import os
from transformers import AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
model_id = "/data/home/wangyongxin/model/llama3/8b/origin/Meta-Llama-3-8B-Instruct"

# model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
model_hf = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)

sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)