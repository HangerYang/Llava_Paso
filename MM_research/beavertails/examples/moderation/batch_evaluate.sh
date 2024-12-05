CUDA_VISIBLE_DEVICES=2 python batch_evaluate.py \
    --eval_dataset /home/spencers/multimodal/MQT-LLaVA/batch_model_outputs.json \
    --model_path PKU-Alignment/beaver-dam-7b \
    --max_length 512 \
    --output_dir output/evaluation