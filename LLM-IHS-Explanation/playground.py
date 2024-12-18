# import torch
# import json
# import argparse
# from tqdm import tqdm
# from datasets import load_dataset
# from PIL import Image
# import sys
# import numpy as np
# import os
# from w2s_utils import get_layer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler
# import accelerate
# from visualization import topk_intermediate_confidence_heatmap, accuracy_line
# from pdb import set_trace as breakpoint
# sys.path.append('/home/hyang/llava_paso/LLaVA')

# from llava.mm_utils import get_model_name_from_path
# from llava.constants import (
#     IMAGE_TOKEN_INDEX,
#     DEFAULT_IMAGE_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IM_END_TOKEN,
# )
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import (
#     tokenizer_image_token,
#     get_model_name_from_path,
#     KeywordsStoppingCriteria,
# )


# def load_hades(mm_use_im_start_end=False, model_name="vicuna-7b-v1.5"):
#     hades = load_dataset("Monosail/HADES")['test']
#     hades = [item for item in hades if item['step'] == 5]
#     malicious = []
#     for item in hades:
#         image = item['image'].convert('RGB')
#         inst = item['instruction']

#         # if item["category"] == "behavior":
#         #     mask = "conduct this behavior on"
#         # else:
#         #     mask = f"this {item['category']}"
#         # new_inst = f"The bottom of the image shows a {item['category']}. " + inst.replace(item["keywords"], mask)
#         qs = inst
#         if mm_use_im_start_end:
#             qs = (
#                 DEFAULT_IM_START_TOKEN
#                 + DEFAULT_IMAGE_TOKEN
#                 + DEFAULT_IM_END_TOKEN
#                 + "\n"
#                 + qs
#             )
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#         if "llama-2" in model_name.lower():
#             conv_mode = "llava_llama_2"
#         elif "v1" in model_name.lower():
#             conv_mode = "llava_v1"
#         elif "mpt" in model_name.lower():
#             conv_mode = "mpt"
#         else:
#             conv_mode = "llava_v0"

#         conv = conv_templates[conv_mode].copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], None)
#         # breakpoint()
#         prompt = conv.get_prompt()
#         malicious.append({'id':item['id'], 'image':image,'prompt': prompt})
#     return malicious



# def load_normal(dataset_path, mm_use_im_start_end=False,model_name="vicuna-7b-v1.5"):
#     with open(dataset_path, "r") as file:
#         content = file.read()
#     normal_dataset = json.loads(content)
#     normal = []
#     for item in tqdm(normal_dataset[:750]):
#         image = Image.open("/home/hyang/llava_paso/coco/" + item['image']).convert('RGB').resize((1024, 1324))
#         for conv_item in item['conversations']:
#             if conv_item['from'] == 'human':
#                 qs = conv_item['value']
#                 if mm_use_im_start_end:
#                     qs = (
#                         DEFAULT_IM_START_TOKEN
#                         + DEFAULT_IMAGE_TOKEN
#                         + DEFAULT_IM_END_TOKEN
#                         + "\n"
#                         + qs
#                     )
#                 elif DEFAULT_IMAGE_TOKEN not in qs:
#                     qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#                 if "llama-2" in model_name.lower():
#                     conv_mode = "llava_llama_2"
#                 elif "v1" in model_name.lower():
#                     conv_mode = "llava_v1"
#                 elif "mpt" in model_name.lower():
#                     conv_mode = "mpt"
#                 else:
#                     conv_mode = "llava_v0"
                
#                 conv = conv_templates[conv_mode].copy()
#                 conv.append_message(conv.roles[0], qs)
#                 conv.append_message(conv.roles[1], None)
#                 # breakpoint()
#                 prompt = conv.get_prompt()
#                 normal.append({'id':item['id'], 'image':image,'prompt': prompt})
#     return normal

# def step_forward_vlm(model, tokenizer, image_processor, input, decoding=True, k_indices=5):
#     image_tensor = (
#         image_processor.preprocess(input['image'], return_tensors="pt")["pixel_values"]
#         .half()
#         .cuda()
#     )

#     input_ids = (
#         tokenizer_image_token(input['prompt'], tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#         .unsqueeze(0)
#         .cuda()
#     )
#     with torch.no_grad():
#         outputs = model(
#             input_ids=input_ids,
#             images=image_tensor,  # Pass image tensor to the model
#             output_hidden_states=True
#         )
#         tl_pair = []
#         lm_head = model.lm_head
#         # Get normalization layer
#         if hasattr(model, "model") and hasattr(model.model, "norm"):
#             norm = model.model.norm
#         elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
#             norm = model.transformer.ln_f
#         else:
#             raise ValueError("Incorrect Model: Missing norm layer.")

#         # Process hidden states layer by layer
#         for i, hidden_state in enumerate(outputs.hidden_states):
#             layer_logits = []
#             layer_output = norm(hidden_state)
#             logits = lm_head(layer_output)

#             # Get logits for the next token
#             next_token_logits = logits[:, -1, :]
#             top_values, top_indices = torch.topk(next_token_logits, k_indices, dim=-1)

#             # Decode or return raw token indices
#             decoded_texts = [tokenizer.decode([idx], skip_special_tokens=False) for idx in top_indices.squeeze().tolist()]
#             top_values = top_values.detach().cpu()
#             if decoding:
#                 for value, token in zip(top_values.squeeze().tolist(), decoded_texts):
#                     layer_logits.append([token, value])
#             else:
#                 for value, token in zip(top_values.squeeze().tolist(), top_indices.squeeze().tolist()):
#                     layer_logits.append([token, value])
#             tl_pair.append(layer_logits)

#         # Extract hidden states for analysis
#         res_hidden_states = [hidden_state.detach().cpu().numpy() for hidden_state in outputs.hidden_states]
#     return res_hidden_states, tl_pair


# class Weak2StrongClassifier:
#     def __init__(self, return_report=True, return_visual=False):
#         self.return_report = return_report
#         self.return_visual = return_visual

#     @staticmethod
#     def _process_data(forward_info):
#         features = []
#         labels = []
#         for key, value in forward_info.items():
#             for hidden_state in value["hidden_states"]:
#                 features.append(hidden_state.flatten())
#                 labels.append(value["label"])

#         features = np.array(features)
#         labels = np.array(labels)
#         X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#         return X_train, X_test, y_train, y_test

#     def svm(self, forward_info):
#         X_train, X_test, y_train, y_test = self._process_data(forward_info)
#         svm_model = SVC(kernel='linear')
#         svm_model.fit(X_train, y_train)
#         y_pred = svm_model.predict(X_test)
#         report = None
#         if self.return_report:
#             print("SVM Test Classification Report:")
#             print(classification_report(y_test, y_pred, zero_division=0.0))
#         if self.return_visual:
#             report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
#         return X_test, y_pred, report

#     def mlp(self, forward_info):
#         X_train, X_test, y_train, y_test = self._process_data(forward_info)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.01,
#                             solver='adam', verbose=0, random_state=42,
#                             learning_rate_init=.01)

#         mlp.fit(X_train_scaled, y_train)
#         y_pred = mlp.predict(X_test_scaled)
#         report = None
#         if self.return_report:
#             print("MLP Test Classification Report:")
#             print(classification_report(y_test, y_pred, zero_division=0.0))
#         if self.return_visual:
#             report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)
#         return X_test, y_pred, report
    
# class Weak2StrongExplanation_vlm:
#     def __init__(self, model_path, layer_nums = 32, return_report=True, return_visual=True):
#         self.model_name = get_model_name_from_path(model_path)
#         self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
#             model_path=model_path, model_base=None, model_name=self.model_name,cache_dir = "/home/hyang/llava_paso/.cache"
#     )
#         self.layer_sums = layer_nums + 1
#         self.forward_info = {}
#         self.return_report = return_report
#         self.return_visual = return_visual
#     def get_forward_info(self, inputs_dataset, class_label, debug=True):
#         offset = len(self.forward_info)
#         for _, i in enumerate(inputs_dataset):
#             if debug and _ > 100:
#                 break
#             list_hs, tl_pair = step_forward_vlm(self.model, self.tokenizer, self.image_processor, i)
#             last_hs = [hs[:, -1, :] for hs in list_hs]
#             self.forward_info[_ + offset] = {"hidden_states": last_hs, "top-value_pair": tl_pair, "label": class_label}
#     def explain(self, datasets, classifier_list=None, debug=True, accuracy=True):
#         self.forward_info = {}
#         if classifier_list is None:
#             classifier_list = ["svm", "mlp"]
#         if isinstance(datasets, list):
#             for class_num, dataset in enumerate(datasets):
#                 self.get_forward_info(dataset, class_num, debug=debug)
#         elif isinstance(datasets, dict):
#             for class_key, dataset in datasets.items():
#                 self.get_forward_info(dataset, class_key, debug=debug)
        
#         classifier = Weak2StrongClassifier(self.return_report, self.return_visual)

#         rep_dict = {}
#         if "svm" in classifier_list:
#             rep_dict["svm"] = {}
#             for _ in range(0, self.layer_sums):
#                 x, y, rep = classifier.svm(get_layer(self.forward_info, _))
#                 rep_dict["svm"][_] = rep

#         if "mlp" in classifier_list:
#             rep_dict["mlp"] = {}
#             for _ in range(0, self.layer_sums):
#                 x, y, rep = classifier.mlp(get_layer(self.forward_info, _))
#                 rep_dict["mlp"][_] = rep
        
#         if not self.return_visual:
#             return
        
#         if accuracy and classifier_list != []:
#             accuracy_line(rep_dict, self.model_name)
#     def vis_heatmap(self, dataset, left=0, right=33, debug=True, model_name=""):
#         self.forward_info = {}
#         self.get_forward_info(dataset, 0, debug=debug)
#         topk_intermediate_confidence_heatmap(self.forward_info, left=left, right=right,model_name=model_name)


# normal = load_normal(dataset_path="/home/hyang/llava_paso/coco/conversation_58k.json")
# malicious = load_hades()
# test = Weak2StrongExplanation_vlm(model_path = 'liuhaotian/llava-v1.5-7b')
# model_name = get_model_name_from_path('liuhaotian/llava-v1.5-7b')

# # classifier_list needs weak classifiers like ["svm", "mlp"] 
# # if accuracy = True, W2SE will return line charts,
# # and heatmap requires you to call the vis_headmap function.
# test.explain({"norm":normal, "mali":malicious}, classifier_list=["svm", "mlp"], accuracy=True)
# # In vis_heatmap function, you need to manually specify the starting layer and the ending layer
# test.vis_heatmap(normal, 16, 32, model_name=model_name)

from huggingface_hub import hf_hub_download

import torch

projector_path = hf_hub_download(
    repo_id="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
    filename="mm_projector.bin"
)

# Load the projector weights
projector_state_dict = torch.load(projector_path)
# print(projector_state_dict.keys())

print(projector_state_dict['model.mm_projector.0.weight'].shape)
print(projector_state_dict['model.mm_projector.2.weight'].shape)

import sys
sys.path.append('/home/hyang/llava_paso/LLaVA')
from llava.model.builder import load_pretrained_model_paper, load_pretrained_model_no_tokenizer
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import tqdm as notebook_tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from llava.model import *
from peft import PeftModel



model_path = "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"
model_base = "lmsys/vicuna-7b-v1.5"

tokenizer, model, image_processor, fcontext_len = load_pretrained_model_paper(
    model_path=model_path,
    model_base="lmsys/vicuna-7b-v1.5",
    model_name=get_model_name_from_path(model_path),
    projector_state_dict=projector_state_dict
)