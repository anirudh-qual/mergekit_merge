from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, os

model_dir = "with_special_tokens"

tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)

print("tokenizer len:", len(tok))
print("tokenizer.vocab_size:", tok.vocab_size)
print("config.vocab_size:", model.config.vocab_size)
print("embed_tokens:", model.get_input_embeddings().num_embeddings)
print("lm_head:", model.lm_head.out_features)

# Optional: print special IDs you’re using
print("eos:", model.config.eos_token_id, "pad:", model.config.pad_token_id)