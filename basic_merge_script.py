INTERMEDIATE_PATH = "/scratch/shared_dir/nmeda6/merge_intermediate/"  # folder to store the result in
FINAL_OUTPUT = "/scratch/shared_dir/nmeda6/merge_final/"
LORA_MERGE_CACHE = "./cache"  # change if you want to keep these for some reason
CONFIG_YML = "./config_deepseek_0merge.yaml"  # merge configuration file
COPY_TOKENIZER = True  
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    DynamicCache,
    PretrainedConfig,
    PreTrainedModel,
    GenerationMixin,
    AutoTokenizer,
    AutoConfig,
)

# Parse layer ownership from MergeKit slices
def parse_mergekit_slices(config):
    slices = config["slices"]
    model_layers = {}  # model_name -> list of output layer indices

    out_idx = 0

    for slc in slices:
        for src in slc["sources"]:
            src_model = src["model"]
            layer_range = src["layer_range"]
            start, end = layer_range 
            length = end - start

        slice_output_range = range(out_idx, out_idx + length)

        for src in slc["sources"]:
            model = model_name_token(src["model"])
            model_layers.setdefault(model, [])
            model_layers[model].extend(slice_output_range)

        out_idx += length

    return model_layers

# Convert real model names to safe token strings
def model_name_token(name: str):
    name = name.replace("/", "_").replace("-", "_")
    return  f"<|use_{name}|>"

with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config_yaml = yaml.safe_load(fp)
    merge_config = MergeConfiguration.model_validate(merge_config_yaml)

model_layer_map = parse_mergekit_slices(merge_config_yaml)

run_merge(
    merge_config,
    out_path=INTERMEDIATE_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
        allow_crimes=True,
        write_model_card=True
    ),
)

model_tokens = [name for name in model_layer_map.keys()]

tokenizer = AutoTokenizer.from_pretrained(INTERMEDIATE_PATH, trust_remote_code=True)
print("Original vocab size:", len(tokenizer))

special_tokens_dict = {"additional_special_tokens": model_tokens}
num_added = tokenizer.add_special_tokens(special_tokens_dict)

print("New vocab size:", len(tokenizer))

tokenizer.save_pretrained(FINAL_OUTPUT)

model = AutoModelForCausalLM.from_pretrained(
    INTERMEDIATE_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
model.config.vocab_size = len(tokenizer)
model.config.layer_ownership = model_layer_map
model_arch = model.config.architectures[0]
model.save_pretrained(FINAL_OUTPUT)

config = AutoConfig.from_pretrained(FINAL_OUTPUT, trust_remote_code=True)
config.architectures = [f"Merged{model_arch}"]
config.save_pretrained(FINAL_OUTPUT)

print("Done. Saved tokenizer + model to", FINAL_OUTPUT)


