import torch
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(tokenizer, model, query, max_attempts, keywords):
    """
    Generate a response from the language model with retry logic to ensure keywords are in the response.
    """
    for attempt in range(max_attempts):
        input_ids = tokenizer(query, return_tensors="pt")["input_ids"].cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        
        if not keywords or keywords in outputs:
            return outputs

    # If we exhausted max_attempts without success
    return outputs


def evaluate(args):
    """
    Evaluate the language model on text-based tasks.
    """
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset_path)["test"]

    # Filter test set based on step argument
    if args.step == "last":
        test_split = [item for item in dataset if item["step"] == 5]
    elif args.step == "all":
        test_split = dataset

    results = []

    print("Starting evaluation...")
    for item in tqdm(test_split):
        inst = item["instruction"]
        keywords = item.get("keywords", None)

        # Generate a response
        response = generate(tokenizer, model, inst, args.max_attempts, keywords)

        # Prepare the result dictionary
        result = {
            "id": item["id"],
            "scenario": item["scenario"],
            "keywords": item["keywords"],
            "step": item["step"],
            "category": item["category"],
            "instruction": item["instruction"],
            "response": response,
        }
        results.append(result)

        # Save result to file incrementally
        with open(args.output_path + "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    print("Evaluation complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Monosail/HADES")
    parser.add_argument("--model_path", type=str, default="gpt2")  # Specify any LM model here
    parser.add_argument("--output_path", type=str, default="./output/")
    parser.add_argument("--ensure_accurate_OCR", type=bool, default=False)  # Placeholder, no OCR required
    parser.add_argument("--max_attempts", type=int, default=5)  # Retry attempts to match keywords
    parser.add_argument("--step", type=str, default="last")  # Evaluate on last-step or all images
    args = parser.parse_args()

    evaluate(args)
