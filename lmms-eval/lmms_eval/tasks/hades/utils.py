import os
import json

def save_results(doc, response):
    """Save model outputs to a JSONL file."""
    result = {
        "id": doc["id"],
        "scenario": doc["scenario"],
        "keywords": doc["keywords"],
        "step": doc["step"],
        "category": doc["category"],
        "instruction": doc["instruction"],
        "response": response
    }
    output_dir="/home/hyang/llava_paso/lmms-eval/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "hades_results.jsonl")
    
    with open(output_file, "w") as f:
        f.write(json.dumps(result) + "\n")

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extracts text input for the LLaVA model from the dataset document."""
    question = doc['instruction'].strip()
    return question  # You can customize this further if necessary

