#!/bin/bash
export PYTHONWARNINGS="ignore"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u finetune.py \
--savepath='/data/eval/PAM_train' \
--config-file '/code/configs/local_train_gan.yaml' \
--print_freq 1 \
--save_freq 4