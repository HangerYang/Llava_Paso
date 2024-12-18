python batch_evaluate_booster.py --eval_dataset /home/hyang/llava_paso/output/llava-v1.5-7bimgblank.json --model_path PKU-Alignment/beaver-dam-7b --output_dir /home/hyang/llava_paso/output_judge_img --beta_val 1

python batch_evaluate_booster.py --eval_dataset /home/hyang/llava_paso/output_llm/llava-v1.6-mistral-7b.json --model_path PKU-Alignment/beaver-dam-7b --output_dir /home/hyang/llava_paso/output_judge --beta_val 1

python batch_evaluate_booster.py --eval_dataset /home/hyang/llava_paso/output_llm/llava-v1.6-vicuna-7b.json --model_path PKU-Alignment/beaver-dam-7b --output_dir /home/hyang/llava_paso/output_judge --beta_val 1

python batch_evaluate_booster.py --eval_dataset /home/hyang/llava_paso/output_llm/llava-v1.6-vicuna-13b.json --model_path PKU-Alignment/beaver-dam-7b --output_dir /home/hyang/llava_paso/output_judge --beta_val 1