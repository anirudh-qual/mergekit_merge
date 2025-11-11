import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    DynamicCache,
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
)
from transformers import AutoTokenizer

OUTPUT_PATH = "./merged"  # folder to store the result in
LORA_MERGE_CACHE = "./cache"  # change if you want to keep these for some reason
CONFIG_YML = "./config.yaml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

import os
HF_DATASETS_CACHE = os.path.join("hf_cache", "datasets")
TRANSFORMERS_CACHE = os.path.join("hf_cache", "transformers")
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE

hf_home = "hf_cache"
hf_home = os.getenv("HF_HOME", hf_home)

tokenizer = AutoTokenizer.from_pretrained("merged",trust_remote_code=True)
print(f"Original vocab size: {len(tokenizer)}")  

special_tokens_dict = {
    'additional_special_tokens': ['<|use_instruct|>', '<|use_base|>']
}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"New vocab size: {len(tokenizer)}")

instruct_token_id = tokenizer.convert_tokens_to_ids('<|use_instruct|>')
base_token_id = tokenizer.convert_tokens_to_ids('<|use_base|>')

print(f"<|use_instruct|> token ID: {instruct_token_id}")  
print(f"<|use_base|> token ID: {base_token_id}")
outpath = "with_special_tokens"
tokenizer.save_pretrained(outpath)

model = AutoModelForCausalLM.from_pretrained("merged",torch_dtype="auto",device_map="auto",trust_remote_code=True)
model.resize_token_embeddings(len(tokenizer))
model.config.vocab_size = len(tokenizer)
model.save_pretrained(outpath)




