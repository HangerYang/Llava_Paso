CUDA_VISIBLE_DEVICES=2 python batch_evaluate_booster.py \
    --eval_dataset /home/spencers/mm_github/beavertails/examples/moderation/llava15_mocha__outputs_grenadebomb.json \
    --model_path PKU-Alignment/beaver-dam-7b \
    --max_length 512 \
    --output_dir output/evaluation \
    --beta_val 1