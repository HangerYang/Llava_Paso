CUDA_VISIBLE_DEVICES=0 python evaluate.py \
    --eval_dataset ../evaluation/custom_eval.json \
    --model_path PKU-Alignment/beaver-dam-7b \
    --max_length 512 \
    --output_dir output/evaluation