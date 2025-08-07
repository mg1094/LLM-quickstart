import os
os.environ["HF_HOME"] = "/root/autodl-tmp/cache"

from transformers import pipeline

model_path = "facebook/opt-125m"

generator = pipeline(
    "text-generation",
    model=model_path,
    device=0,
    do_sample=True,
    num_return_sequences=3
)

prompt = "hello, i am a student"
outputs = generator(prompt)
for output in outputs:
    print(output["generated_text"])
    print("="*50)


    
    
# import os
# import torch
# import shutil
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AwqConfig
# from awq import AutoAWQForCausalLM
# from huggingface_hub import snapshot_download

# # 设置环境变量
# os.environ["HF_HOME"] = "/root/autodl-tmp/cache"

# def download_model_safely(model_id, local_path):
#     """安全下载模型"""
#     try:
#         print(f"下载模型 {model_id} 到 {local_path}...")
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
#         snapshot_download(
#             repo_id=model_id,
#             local_dir=local_path,
#             local_dir_use_symlinks=False,
#             ignore_patterns=["*.safetensors*"]  # 避免safetensors问题
#         )
#         print("模型下载完成!")
#         return local_path
#     except Exception as e:
#         print(f"本地下载失败: {e}")
#         print("尝试直接从Hub加载...")
#         return model_id

# # 模型配置
# model_path = "facebook/opt-125m"
local_model_path = "./models/opt-125m-original"
# quant_path = "models/opt-125m-awq"

# # 安全下载模型
# actual_model_path = download_model_safely(model_path, local_model_path)

# try:
#     # 加载模型
#     print("加载模型进行量化...")
#     model = AutoAWQForCausalLM.from_pretrained(
#         actual_model_path,
#         device_map="cuda",
#         torch_dtype=torch.float16,
#         use_safetensors=False  # 避免safetensors问题
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         actual_model_path, 
#         trust_remote_code=True
#     )
    
#     # 继续您的量化流程...
    
# except Exception as e:
#     print(f"模型加载失败: {e}")
#     print("请检查网络连接或尝试手动下载模型")    

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "models/opt-125m-awq"
quant_config = {"zero_point":True, "q_group_size":128, "w_bit":4, "version":"GEMM"}

model = AutoAWQForCausalLM.from_pretrained(local_model_path, force_download=True, device_map="cuda")
# trust_remote_code=True 表示允许从远程加载自定义的分词器代码（比如模型作者自定义的分词器实现），
# 这样可以确保即使模型仓库中有特殊的分词器实现，也能正确加载和使用。
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

model.quantize(tokenizer, quant_config=quant_config)

print(quant_config)

from transformers import AwqConfig, AutoConfig

quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size = quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"]
).to_dict()

model.model.config.quantization_config = quantization_config
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="cuda").to(0)

def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(0)
    out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

result = generate_text("hello, i am a student")
print(result)

