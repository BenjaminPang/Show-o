#!/bin/bash

nohup accelerate launch \
    --config_file accelerate_configs/1_gpu.yaml \
    --main_process_port=8888 \
    training/train_w_clip_vit.py \
    config=configs_a100/showo_instruction_tuning_1_w_clip_vit.yaml \
    > "${log_file}" 2>&1 &

echo "Process started with PID: $!"