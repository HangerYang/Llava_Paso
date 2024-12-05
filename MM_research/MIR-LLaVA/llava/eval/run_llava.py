# This was modified with the eval2 function for calculating the MIR from the MIR github.
# Functions that eval2 calls were also added here



import argparse
import torch
import math
import os
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model_paper
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    # print("PRINTING INFO")
    # print(args.model_path)
    # print(args.model_base)
    # print(model_name)
    model_name = get_model_name_from_path(args.model_path)
    model.cuda()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    layer_name = "model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.q_proj.weight"  # Adjust this based on your model structure
    print("Selected layer weights:", model.state_dict()[layer_name][:5, :5])
    if torch.isnan(model.state_dict()[layer_name][:5, :5]).any():
        print("nans exist")
    else:
        print("no nans")
    

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            output_hidden_states=True
        )

    # hidden_states = output_ids.hidden_states
    # print(hidden_states)
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


def replace_outliers_with_median_l2(data):
    norms = torch.norm(data, p=2, dim=-1)  # Compute L2 norms along the last dimension
    median_norm = torch.median(norms)
    std_dev = torch.std(norms)
    outliers = torch.abs(norms - median_norm) > 3 * std_dev  # Outliers based on norms

    median_values = torch.median(data, dim=0).values
    data[outliers, :] = median_values
    return data

class MatrixSquareRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Ensure the input is a square matrix
        assert input.shape[0] == input.shape[1], "Input must be a square matrix"
        
        # Use the Newton-Schulz iteration to approximate the square root
        max_iter = 5
        eye = torch.eye(input.shape[0], device=input.device, dtype=input.dtype)
        Y = input
        Z = eye
        for i in range(max_iter):
            Y = 0.5 * (Y + torch.inverse(Z) @ input)
            Z = 0.5 * (Z + torch.inverse(Y) @ input)
        
        ctx.save_for_backward(Y, Z)
        return Y
    
    @staticmethod
    def backward(ctx, grad_output):
        Y, Z = ctx.saved_tensors
        grad_input = grad_output @ torch.inverse(Y).t() @ torch.inverse(Y).t()
        return grad_input

def matrix_sqrt(input):
    return MatrixSquareRoot.apply(input)

@torch.no_grad()
def calculate_fid_pytorch(tensor_A, tensor_B):
    print(f"tensor A shape: {tensor_A.shape}")
    print(f"tensor B shape: {tensor_B.shape}")
    print("calculting fid pytorch")
    mu_A = tensor_A.mean(dim=0)
    mu_B = tensor_B.mean(dim=0)
    print("means done")
    tensor_A_centered = tensor_A - mu_A
    tensor_B_centered = tensor_B - mu_B
    cov_A = tensor_A_centered.T @ tensor_A_centered / (tensor_A.size(0) - 1)
    cov_B = tensor_B_centered.T @ tensor_B_centered / (tensor_B.size(0) - 1)
    print("more done")
    covmean = matrix_sqrt(cov_A @ cov_B)
    print("sqrt done")
    if torch.is_complex(covmean):
        covmean = covmean.real

    mean_diff = mu_A - mu_B
    fid = torch.sum(mean_diff**2) + torch.trace(cov_A) + torch.trace(cov_B) - 2 * torch.trace(covmean)
    print("sum done")
    return fid.item()

def read_story_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    parts = content.split('@highlight')
    story = parts[0].strip()
    highlights = [part.strip() for part in parts[1:]]
    return story, highlights

def eval_model2(args, tokenizer, model, image_processor, data_images, data_texts, text_data_path, image_base_path):
    # Model
    disable_torch_init()

    all_hidden_states = {"vision": [], "text": []}

    model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     args.model_path, args.model_base, model_name
    # )

    # layer_name = "model.vision_tower.vision_tower.vision_model.encoder.layers.10.self_attn.q_proj.weight"  # Adjust this based on your model structure
    # print("Selected layer weights:", model.state_dict()[layer_name][:5, :5])
    # if torch.isnan(model.state_dict()[layer_name][:5, :5]).any():
    #     print("nans exist")
    # else:
    #     print("no nans")

    for idx in range(args.eval_num):

        qs = read_story_file(os.path.join(text_data_path, data_texts[idx]))[0]
        # qs = data_texts[idx]
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    conv_mode, args.conv_mode, args.conv_mode
                )
            )
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print(f"\n\n--------------Prompt: {prompt} -------------\n\n")

        raw_image = os.path.join(image_base_path, data_images[idx])
        print(raw_image)
        raw_image = Image.open(raw_image)
        raw_image = raw_image.convert("RGB")
        images_tensor = process_images([raw_image], image_processor, model.config)[0].unsqueeze(0)
        image_sizes = raw_image.size

        # image_files = image_parser(args)
        # images = load_images(image_files)
        # image_sizes = [x.size for x in images]
        # images_tensor = process_images(
        #     images,
        #     image_processor,
        #     model.config
        # ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        image_start_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]

        print("Input ids and image tensor shapes:")
        print(input_ids.shape)
        print(images_tensor.shape)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                #do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                #top_p=args.top_p,
                #num_beams=args.num_beams,
                #max_new_tokens=args.max_new_tokens,
                #use_cache=True,
                do_sample=False,
                num_beams=1,
                max_new_tokens=1,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
        
        print(type(output_ids))
        generated_tokens = output_ids.sequences
        decoded_text = tokenizer.decode(generated_tokens[0])
        print(decoded_text)

        hidden_states = output_ids.hidden_states
        # print(type(hidden_states))
        # print(hidden_states[0])
        # print(len(hidden_states[0]))
        # print(hidden_states[0].shape)
        latent_hidden_states = [hidden_state.squeeze() for hidden_state in hidden_states[0]]
        print(f"Latent hidden states:")
        print(len(latent_hidden_states))
        print(latent_hidden_states[5])
        vision_hidden_states = [latent[image_start_idx:image_start_idx+576,:].detach().cpu() for latent in latent_hidden_states]
        text_hidden_states = [latent[image_start_idx+576:,:].detach().cpu() for latent in latent_hidden_states]
        all_hidden_states["vision"].append(vision_hidden_states) # 100 * [33, 576, 4096]
        all_hidden_states["text"].append(text_hidden_states)



    layer_length = len(all_hidden_states["vision"][0])
    plot_data = {"Per-Layer-MIR":[]}
    for layer_idx in range(1, layer_length):

        vision_features = [hidden_states[layer_idx].float().cuda() for hidden_states in all_hidden_states["vision"]]
        text_features = [hidden_states[layer_idx].float().cuda() for hidden_states in all_hidden_states["text"]]
        vision_features = torch.cat(vision_features, dim=0)
        text_features = torch.cat(text_features, dim=0)

        # Text-Centric Normalization
        scale_factor = 1. / text_features.norm(p=2, dim=-1).mean(0)
        vision_features = scale_factor * vision_features
        text_features = scale_factor * text_features
        # print(f"Scale factor: {scale_factor}")

        # 3-Sigma Outlier Removal
        vision_features = replace_outliers_with_median_l2(vision_features)
        text_features = replace_outliers_with_median_l2(text_features)

        # Switch between fast mode and accurate mode, we use fast mode by default
        if True:
            plot_data["Per-Layer-MIR"].append(calculate_fid_pytorch(vision_features, text_features))
        else:
            plot_data["Per-Layer-MIR"].append(calculate_fid(vision_features, text_features))

        print("Layer #{}\tPer-Layer MIR: {}".format(layer_idx, plot_data["Per-Layer-MIR"][-1]))
    
    final_mir = math.log(sum(plot_data["Per-Layer-MIR"]), 10)
    print(f"Overall MIR: {final_mir}")
    #outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # print(outputs)
    # return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
