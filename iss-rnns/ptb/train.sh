#!/bin/bash
#CUDA_VISIBLE_DEVICES=5 python tmp.py --model sparselarge \
#  --data_path simple-examples/data/  \
#  --hidden_size1 1500 \
#  --hidden_size2 1500 \
#  --config_file structure_grouplasso.json

CUDA_VISIBLE_DEVICES=2 python tmp.py --model large \
  --data_path simple-examples/data/  \
  --hidden_size1 1500 \
  --hidden_size2 1500 \
  --config_file from_scratch.json 
