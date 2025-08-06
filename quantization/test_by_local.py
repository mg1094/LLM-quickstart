import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/cache"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 直接从本地路径加载量化模型
local_model_path = "models/opt-2.7b-gptq"

# 加载量化模型 - transformers会自动识别量化配置
quant_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map='auto',
    torch_dtype=torch.float16  # 可选：进一步优化内存
)

# 加载对应的分词器
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")  # 或者也保存分词器到本地

# 测试推理
text = "Merry Christmas! I'm glad to"
inputs = tokenizer(text, return_tensors="pt").to(quant_model.device)
out = quant_model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))