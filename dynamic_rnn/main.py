# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

import data
import model
import os

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--session', type=str, default='model',
                    help='identify a training process')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping default=0.25')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay ')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--start_epochs', type=int, default=1,
                    help='start epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--hard_thresh', type=float, default=0.0,
                    help='output threshold for gates (0.0, 1.0)')
parser.add_argument('--norm_thresh', type=float, default=0.0,
                    help='output threshold for gates in l1norm (0.0, 1.0)')
parser.add_argument('--gate_decay_rate', type=float, default=0.0,
                    help='decay rate for l1norm of gate')
parser.add_argument('--group_lasso_rate', type=float, default=0.002,
                    help='for iis')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--optim', type=str, default='sgd',
                    help='sgd or adam')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--checkpoint_path', type=str,  default='dump',
                    help='path to save the final model')
parser.add_argument('--start_from', type=str, default='',
                    help='path to start from')
parser.add_argument('--save_every', type=int, default=2)
parser.add_argument('--eval_only', type=int, default=0)
args = parser.parse_args()

# set tensorboard
try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


tf_summary_writer = tf and tf.summary.FileWriter(args.checkpoint_path)


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# set to distable cudnn
torch.backends.cudnn.enabled = False

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 1
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    #hidden = model.init_hidden(eval_batch_size)
    hidden = None
    tmp_time = []
    print(data_source.size(0))
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        start_time = time.time()
        output, hidden, gate_norm, hard_norm = model(data, hidden)
        tmp_time.append(time.time() - start_time)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    #print('--------------')
    #print(np.mean(tmp_time[10: -10]))
    return total_loss[0] / len(data_source)


# At any point you can hit Ctrl + C to break out of training early.
criterion = nn.CrossEntropyLoss()

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
print('vocab : %d'%(ntokens))

def build_model():
    if args.start_from != '':
        model = start_from()
    else:
        import model
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, args.hard_thresh, args.norm_thresh)
    return model

def start_from():
    print('start from %s'%(args.start_from))
    with open(os.path.join(args.start_from, 'model.pt'), 'rb') as f:
        model = torch.load(f)
        #model.cpu()
        #model = torch.load(f, map_location=lambda storage, loc: storage)
    return model

model = build_model()
if args.cuda:
    model.cuda()

if args.start_from != '':
    # Run on val data.
    val_loss = evaluate(val_data)
    print('=' * 89)
    print('| restart of training | val loss {:5.2f} | val ppl {:8.2f}'.format(
        val_loss, math.exp(val_loss)))
    print('=' * 89)
    np.save(open('np_dump/t0.0_val.np', 'w'), model.rnn.dropout_state[0])
    model.rnn.dropout_state[0] = []
    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    np.save(open('np_dump/t0.0_test.np', 'w'), model.rnn.dropout_state[0])

# Loop over epochs.
lr = args.lr
best_val_loss = None
for epoch in range(args.start_epochs, args.epochs+1):
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    else:
        print('this optim is not supported ...')
    
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    total_norm_loss = 0
    total_norm_count = 0
    total_hard_loss = 0
    total_hard_count = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    iter_per_epoch = len(range(0, train_data.size(0) - 1, args.bptt))
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        output, hidden, gate_norm, hard_norm = model(data, hidden)
        gate_norm, gate_norm_c = gate_norm
        hard_norm, hard_norm_c = hard_norm
        #group_lasso, zero_counts = model.group_lasso()
        group_lasso = model.group_lasso()
        loss = criterion(output.view(-1, ntokens), targets) 
        LOSS = loss + args.gate_decay_rate * gate_norm + args.group_lasso_rate * group_lasso
        #LOSS = loss + args.group_lasso_rate * group_lasso
        LOSS.backward()
        
        total_loss += loss.data[0]
        total_norm_loss += gate_norm.data[0]
        total_norm_count += gate_norm_c
        total_hard_loss += hard_norm.data[0]
        total_hard_count += hard_norm_c
        if (batch + 1) % args.log_interval == 0 or batch == 0:
            cur_loss = total_loss
            cur_norm_loss = total_norm_loss
            cur_norm_count = total_norm_count
            cur_hard_loss = total_hard_loss
            cur_hard_count = total_hard_count
            elapsed = (time.time() - start_time) * 1000 
            if batch > 0:
                tmp = [cur_loss, cur_norm_loss, cur_norm_count, cur_hard_loss, cur_hard_count, elapsed]
                tmp = [item / args.log_interval for item in tmp]
                cur_loss, cur_norm_loss, cur_norm_count, cur_hard_loss, cur_hard_count, elapsed = tmp

            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | hard {:7d}/{:7d}={:.5f} | norm {:7d}/{:7d}={:.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed, cur_loss, int(cur_hard_loss), int(cur_hard_count), cur_hard_loss / cur_hard_count, \
                        int(cur_norm_loss), int(cur_norm_count), cur_norm_loss / cur_norm_count, math.exp(cur_loss)))
            
            add_summary_value(tf_summary_writer, 'train_loss', cur_loss, batch + (epoch - 1) * iter_per_epoch)
            add_summary_value(tf_summary_writer, 'train_ppl', math.exp(cur_loss), batch + (epoch - 1) * iter_per_epoch)
            tf_summary_writer.flush()
            if batch > 0:
                total_loss = 0
                total_norm_loss = 0
                total_norm_count = 0
                total_hard_loss = 0
                total_hard_count = 0
                start_time = time.time()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        '''
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
        '''

        optimizer.step()
        optimizer.zero_grad()


    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)
    add_summary_value(tf_summary_writer, 'val_loss', val_loss, epoch * iter_per_epoch)
    add_summary_value(tf_summary_writer, 'val_ppl', math.exp(val_loss), epoch * iter_per_epoch)
    tf_summary_writer.flush()
    # Save the model if the validation loss is the best we've seen so far.
    with open(os.path.join(args.checkpoint_path, 'model.pt'), 'wb') as f:
        torch.save(model, f)
    if epoch % args.save_every == 0:
        with open(os.path.join(args.checkpoint_path, 'model-epoch%d.pt'%(epoch)), 'wb') as f:
            torch.save(model, f)
    if not best_val_loss or val_loss < best_val_loss:
        with open(os.path.join(args.checkpoint_path, 'model-best.pt'), 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        lr /= 4.0
        with open(os.path.join(args.checkpoint_path, 'model-best.pt'), 'rb') as f:
            model = torch.load(f)

        '''
        if epoch >= 30 :
            if lr == 20:
                lr /= 10.0
                with open(os.path.join(args.checkpoint_path, 'model-best.pt'), 'rb') as f:
                    model = torch.load(f)
            elif epoch >= 40:
                if lr == 2:
                    lr /= 10.0
                    with open(os.path.join(args.checkpoint_path, 'model-best.pt'), 'rb') as f:
                        model = torch.load(f)
                elif epoch >= 50:
                    if lr == 0.2:
                        lr /= 10.0
                        with open(os.path.join(args.checkpoint_path, 'model-best.pt'), 'rb') as f:
                            model = torch.load(f)
        '''                     
    #if epoch >= 14:
    #    lr /= 1.15


# Load the best saved model.
with open(os.path.join(args.checkpoint_path, 'model-best.pt'), 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
