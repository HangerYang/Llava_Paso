#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=2
export OMP_NUM_THREADS=2
export CUDA_LAUNCH_BLOCKING=1  # Enable CUDA synchronous debug mode

# Set the log file path
LOGFILE="./results/dpo_lora/training.log"

# Ensure the log file directory exists
mkdir -p "$(dirname "$LOGFILE")"

# Run command and log output to the log file
{
    echo "Starting training at $(date)"
    accelerate launch --config_file ./scripts/config.yaml \
        ./train_dpo.py \
        --do_train \
        --seed 42 \
        --keywords keyword1 \
        --config_train_path "/home/hyang/llava_paso/data/dict.json" \
        --dpo_beta 0.1 \
        --lora_enable True \
        --finetune_mm_projector True \
        --batch_size 1 \
        --policy_model_name_or_path liuhaotian/llava-v1.5-7b \
        --learning_rate 2e-5 \
        --warmup_steps 10 \
        --output_dir ./results/dpo_lora \
        --total_epochs 1 \
        --evaluation_strategy "no" \
        --weight_decay 0.0 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --ddp_backend "nccl" \
        --bf16 True \
        --ddp_find_unused_parameters False \
        --max_grad_norm 1.0 \
        --clean_tokens_after_eos True \
        --temperature 1.0 \
        --model_max_length 2048 \
        --image_folder /home/hyang/llava_paso/data/figs \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio 'pad'
    echo "Training completed at $(date)"
} | tee "$LOGFILE"
