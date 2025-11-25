from vllm.model_executor.models.llama import (
    LlamaForCausalLM,
    LlamaModel,
)

from transformers import AutoTokenizer
from vllm.attention import AttentionMetadata
from typing import List, Optional
import torch
from transformers import PretrainedConfig

class MergedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, vllm_config, prefix: str = "", **kwargs):
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        
        # Extract model-token → layer indices from config
        # Expect config.layer_ownership = {model_name: [layer_indices]}
        self.layer_ownership = getattr(self.config, "layer_ownership", {})

        # # Map model-token IDs to model names
        # self.token_id_to_model = {}
        # if hasattr(self.config, "tokenizer"):
        #     for model_name, token_str in getattr(self.config, "special_tokens", {}).items():
        #         token_id = self.config.tokenizer.convert_tokens_to_ids(token_str)
        #         self.token_id_to_model[token_id] = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # --- STEP 1: Determine which model to use based on first token ---
        if input_ids.dim() == 1:
            first_token = input_ids[0].item()
        else:
            first_token = input_ids[0, 0].item()

        dummy_run = False
        print("===", first_token)
        if first_token not in self.layer_ownership.keys():
            first_token = list(self.layer_ownership.keys())[0] # setting a default because we need it for gpu_model_runner.py -> _dummy_run
            dummy_run = True
            # print("====", self.layer_ownership.keys())
            # raise RuntimeError(f"Unknown routing token: {first_token}")

        # --- STEP 2: Lookup layer indices from config ---
        layer_indices = self.layer_ownership.get(first_token)
        if layer_indices is None:
            raise RuntimeError(f"No layer indices found for routing token: {first_token}")

        # --- STEP 3: Remove routing token from input ---
        if not dummy_run:
            if input_ids.dim() == 1:
                input_ids, positions = input_ids[1:], positions[1:]
            else:
                input_ids, positions = input_ids[:, 1:], positions[:, 1:]

        # --- STEP 4: Embed tokens if needed ---
        hidden_states = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
        residual = None

        # --- STEP 5: Apply selected layers ---
        for physical_layer_idx in layer_indices:
            layer = self.model.layers[physical_layer_idx]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        hidden_states = self.model.norm(hidden_states)
        return hidden_states
