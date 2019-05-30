CUDA_VISIBLE_DEVICES=6 python ptb_word_lm_heter.py \
        --model large \
        --data_path simple-examples/data/ \
        --hidden_size1 1500 \
        --hidden_size2 1500 \
        --config_file from_scratch.json 
