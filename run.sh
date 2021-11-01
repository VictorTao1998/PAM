#!/bin/bash
export PYTHONWARNINGS="ignore"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u /jianyu-fast-vol/PAM/finetune.py \
--savepath='/jianyu-fast-vol/eval/ActiveStereoRui_train' \
--config-file '/jianyu-fast-vol/ActiveStereoRui/configs/remote_train_gan.yaml' \
--print_freq 100
--save_freq 1