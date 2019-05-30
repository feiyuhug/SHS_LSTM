#########################################################################
# File Name: train_fromscratch.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Mon 30 Apr 2018 09:42:12 PM HKT
#########################################################################
#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python ptb_word_lm_heter.py \
  --model large \
  --data_path simple-examples/data/ \
  --hidden_size1 1500 \
  --hidden_size2 1500 \
  --config_file from_scratch.json 


