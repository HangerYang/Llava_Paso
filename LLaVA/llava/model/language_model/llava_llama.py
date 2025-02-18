#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
import torch.nn as nn
from pdb import set_trace as breakpoint
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
IMAGE_TOKEN_INDEX = -200
class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.ema_losses = {}
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    

    def update_ema(self, curr_loss, loss_type, alpha=0.1, epsilon=1e-8):
        curr_loss = curr_loss.detach()
        if loss_type not in self.ema_losses:
            # Initialize the EMA for this loss type if not present.
            self.ema_losses[loss_type] = curr_loss
            print(f"Initializing EMA for {loss_type} loss: {curr_loss}")
        else:
            old_val = self.ema_losses[loss_type]
            new_val = alpha * curr_loss + (1 - alpha) * old_val
            self.ema_losses[loss_type] = new_val
        
        return self.ema_losses[loss_type] + epsilon
    def compute_l2_loss(self, tensor, num_image_embeddings=576, image_start_idx=None):
            """
            Computes the L2-based loss between image and text embeddings in the batch.
            
            Args:
                tensor: Tensor of shape [batch_size, num_embedding_vectors, 4096]
                num_image_embeddings: Number of image embeddings per batch element (default: 576)
                sample_size: Number of image embeddings to sample (default: 100)
                
            Returns:
                L2 loss (scalar)
            """
            # breakpoint()
            image_start_idx = image_start_idx.tolist()
            batch_size = tensor.size(0)
            loss = torch.tensor(0.0, device=tensor.device)
            for b in range(batch_size):
                start_idx = image_start_idx[b]
                image_embeddings = tensor[b, start_idx:start_idx+num_image_embeddings, :]  # Shape: [576, 4096]
                text_embeddings = tensor[b, start_idx+num_image_embeddings:, :]   # Shape: [num_text_embeddings, 4096]
                
                # randomly sample 100 image embeds
                # sampled_indices = torch.randperm(num_image_embeddings)[:sample_size]
                # sampled_image_embeddings = image_embeddings[sampled_indices]  # Shape: [100, 4096]
                
                # Compute pairwise L2 distances
                distances = torch.cdist(image_embeddings, text_embeddings, p=2)
                loss += distances.sum()

            # Average the loss over the batch size
            loss = loss / batch_size
            return loss
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        image_start_idx = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        sim_loss_coeff = 0.1
        original_sim_loss = self.compute_l2_loss(inputs_embeds, image_start_idx=image_start_idx)
        max_sim = self.update_ema(original_sim_loss, "sim")
        training_list = [0, 4, 8, 12, 16, 20, 24, 28, 32]
        output_hidden_states = [x.detach() for x in outputs.hidden_states]
        total_normalized_hidden_loss= torch.tensor(0.0, device=output_hidden_states[0].device)
        # for i in training_list:
        #     hidden_states_loss = self.compute_l2_loss(output_hidden_states[i], image_start_idx=image_start_idx)
        #     max_hidden = self.update_ema(hidden_states_loss, f"hidden_{i}")
        #     total_normalized_hidden_loss += hidden_states_loss/max_hidden
        # total_normalized_hidden_loss = sim_loss_coeff * total_normalized_hidden_loss
        
        normalized_sim_loss = sim_loss_coeff * (original_sim_loss / max_sim)
        normalized_util_loss = loss / self.update_ema(loss, "util")
        total_loss = normalized_sim_loss + normalized_util_loss + total_normalized_hidden_loss
        # breakpoint()
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     image_start_idx=image_start_idx
        # )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
