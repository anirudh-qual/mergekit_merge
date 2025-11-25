from vllm import LLM, SamplingParams, ModelRegistry
from custom_vllm_model import MergedLlamaForCausalLM

ModelRegistry.register_model("MergedLlamaForCausalLM", MergedLlamaForCausalLM)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)
llm = LLM (model = "/scratch/shared_dir/nmeda6/merge_final/", trust_remote_code=True, enforce_eager=True)

input = "<|use__scratch_shared_dir_unified_models_deepseek_coder_7b_instruct_v1.5|> Tell me a joke about cats."
outputs = llm.generate([input], sampling_params)
print(outputs)

# from vllm import LLM
# from transformers import AutoTokenizer
# import torch

# # 1️⃣ Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("/scratch/shared_dir/nmeda6/merge_final/", trust_remote_code=True)

# # Test a sample input
# text = "<|use__scratch_shared_dir_unified_models_deepseek_coder_7b_instruct_v1.5|>Tell me a joke about cats."
# encoded = tokenizer(text)
# print("Token IDs:", encoded["input_ids"])


# tokenizer = AutoTokenizer.from_pretrained("/scratch/shared_dir/unified_models/deepseek-coder-7b-instruct-v1.5", trust_remote_code=True)

# # Test a sample input
# text = "<|use__scratch_shared_dir_unified_models_deepseek_coder_7b_instruct_v1.5|>Tell me a joke about cats."
# encoded = tokenizer(text)
# print("Token IDs:", encoded["input_ids"])


# print("Decoded:", tokenizer.decode(input_ids))
# print("Number of tokens:", len(input_ids))

# # 2️⃣ Load vLLM model pointing to the folder with safetensors
# llm = LLM(model="/scratch/shared_dir/nmeda6/merge_final/", trust_remote_code=True, enforce_eager=True)

# # 3️⃣ Quick embedding check
# # Access the underlying model through llm.model
# # Note: LLM wraps the model, the underlying HF model is llm.model.model
# hf_model = llm.model.model
# embeddings = hf_model.get_input_embeddings().weight

# print("Embedding shape:", embeddings.shape)
# print("Tokenizer vocab size:", len(tokenizer))

# # 4️⃣ Ensure vocab matches embedding size
# if embeddings.shape[0] == len(tokenizer):
#     print("✅ Embeddings and tokenizer vocab are aligned")
# else:
#     print("❌ Mismatch between embeddings and tokenizer vocab!")

# # 5️⃣ Optional: run a single forward pass (prefill)
# outputs = llm.model.forward(input_ids=torch.tensor([input_ids]))
# print("Forward pass logits shape:", outputs.logits.shape)
