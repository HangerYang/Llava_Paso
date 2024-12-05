python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.5-7b" \
    --tasks hades \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_hades \
    --output_path ./output/