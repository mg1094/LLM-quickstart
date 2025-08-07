import os
os.environ["HF_HOME"] = "/root/autodl-tmp/cache"

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "facebook/opt-2.7b"
quant_model_dir = "models/opt-2.7b-awq"

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# 量化模型
model.quantize(tokenizer, quant_config=quant_config)
print(quant_config)

from transformers import AwqConfig, AutoConfig

# 修改配置文件以使其与transformers集成兼容
quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower(),
).to_dict()

# 预训练的transformers模型存储在model属性中，我们需要传递一个字典
model.model.config.quantization_config = quantization_config

# 保存模型权重
model.save_quantized(quant_model_dir)
# 保存分词器
tokenizer.save_pretrained(quant_model_dir)  

model.eval()

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(quant_model_dir)
model = AutoModelForCausalLM.from_pretrained(quant_model_dir, device_map="cuda").to(0)

def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(0)

    out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

result = generate_text("Merry Christmas! I'm glad to")
print(result)

result = generate_text("The woman worked as a")
print(result)