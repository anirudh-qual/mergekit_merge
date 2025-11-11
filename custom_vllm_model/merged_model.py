from vllm.model_executor.models.llama import (
    LlamaForCausalLM,
    LlamaModel,
)

from transformers import AutoTokenizer
from vllm.attention import AttentionMetadata
from typing import List, Optional
import torch
from transformers import PretrainedConfig

class MergedModel (LlamaForCausalLM):
    def __init__(self,vllm_config, prefix: str = "", **kwargs):
        # vLLM uses keyword-only arguments, so we need to match that signature
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        # Initialize with default values
        self.la = []
        self.lb = []
        
        self.lbid = getattr(self.config, 'lbid', None)
        self.laid = getattr(self.config, 'laid', None)
        
       
        if hasattr(self.config, 'la') and hasattr(self.config, 'lb'):
            self.la = self.config.la
            self.lb = self.config.lb


    def set_layer_indices (self, la, lb):
        self.la = la
        self.lb = lb

    def set_token_ids(self, laid, lbid):
        self.laid = laid
        self.lbid = lbid
    def set_routing_logic(self, inputs):
        # Handle both 1D and 2D tensors
        if inputs.dim() == 1:
            first_token = inputs[0]
        else:
            first_token = inputs[0, 0]
        
        if first_token == self.lbid:
            return False
        return True
    
    def remove_routing_token(self, inputs, positions):
        # Handle both 1D and 2D tensors
        if inputs.dim() == 1:
            return inputs[1:], positions[1:]
        else:
            return inputs[:, 1:], positions[:, 1:]
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        intermediate_tensors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,  # Accept any other kwargs vLLM might pass
    ) -> torch.Tensor:
      
        
        is_base = self.set_routing_logic(input_ids)
        vsz = self.config.vocab_size
        if torch.any(input_ids < 0) or torch.any(input_ids >= vsz):
            bad = input_ids[(input_ids < 0) | (input_ids >= vsz)]
            raise RuntimeError(f"Bad token id(s): {bad[:10].tolist()} vs vocab_size={vsz}")
        # === STEP 2: Remove Routing Token ===
        input_ids, positions = self.remove_routing_token(input_ids, positions)
        
        
        if is_base:
            layer_indices = self.la
        else:
            layer_indices = self.lb
            
        
        # Use inputs_embeds if provided, otherwise embed input_ids
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.model.embed_tokens(input_ids)
        
        # Initialize residual to None (vLLM layers will compute it)
        residual = None
        
        # Handle layers
        if kv_caches is not None and attn_metadata is not None:
            # Normal inference path with KV caching
            for physical_layer_idx in layer_indices:
                layer = self.model.layers[physical_layer_idx]
                
                
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
        else:
            # Initialization/profiling path - skip layers or call without kv_cache
            # During init, vLLM might not provide kv_caches
            for physical_layer_idx in layer_indices:
                layer = self.model.layers[physical_layer_idx]
                
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                )
           
        
        
        hidden_states = self.model.norm(hidden_states)
        return hidden_states
        
    
