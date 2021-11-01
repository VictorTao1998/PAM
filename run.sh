#!/bin/bash
export PYTHONWARNINGS="ignore"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/PAM/finetune.py \
--savepath='/jianyu-fast-vol/PAM/PAM_train' \
--config-file '/jianyu-fast-vol/PAM/configs/remote_train_gan.yaml' \
--print_freq 100 \
--save_freq 1