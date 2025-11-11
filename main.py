from vllm import LLM, SamplingParams, ModelRegistry
from custom_vllm_model import MergedModel


ModelRegistry.register_model("MergedModel", MergedModel)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)
llm = LLM (model = "with_special_tokens", trust_remote_code=True)

input = "Tell me a joke about cats."
outputs = llm.generate([input], sampling_params)
print(outputs)