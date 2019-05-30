#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python tmp.py \
  --model validtestlarge \
  --data_path simple-examples/data/ \
  --display_weights True \
  --config_file structure_grouplasso.json \
  --restore_path dump/thresh0.3/ \


