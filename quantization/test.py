import os
os.environ["HF_HOME"] = "/root/autodl-tmp/cache"

from transformers import TRANSFORMERS_CACHE
from transformers import __version__ as transformers_version
from datasets import __version__ as datasets_version
from huggingface_hub import __version__ as huggingface_hub_version


# 打印库版本（可选）
print("=== Hugging Face 库版本 ===")
print(f"transformers: {transformers_version}")
print(f"datasets:     {datasets_version}")
print(f"hub:          {huggingface_hub_version}")
print()

# 打印环境变量
print("=== 环境变量设置情况 ===")
print(f"HF_HOME:              {os.getenv('HF_HOME', '未设置')}")
print(f"TRANSFORMERS_CACHE:   {os.getenv('TRANSFORMERS_CACHE', '未设置')}")
print(f"HF_DATASETS_CACHE:    {os.getenv('HF_DATASETS_CACHE', '未设置')}")
print(f"HUGGINGFACE_HUB_CACHE:{os.getenv('HUGGINGFACE_HUB_CACHE', '未设置')}")
print()

# 打印实际使用的缓存路径（transformers 使用的）
print("=== 实际生效的缓存路径 ===")
print(f"Transformers 默认缓存目录: {TRANSFORMERS_CACHE}")

# 验证 datasets 缓存路径
try:
    from datasets import config
    print(f"Datasets 缓存目录: {config.HF_DATASETS_CACHE}")
except ImportError:
    print("Datasets 缓存目录: 无法导入 datasets 模块")

# 验证 huggingface_hub 缓存
try:
    from huggingface_hub import constants
    hub_cache = constants.HUGGINGFACE_HUB_CACHE
    print(f"Hugging Face Hub 缓存目录: {hub_cache}")
except ImportError:
    print("Hugging Face Hub 缓存目录: 无法导入 huggingface_hub 模块")
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_name_or_path = "facebook/opt-2.7b"

# wikitext2数据集是一个广泛用于语言建模任务的英文文本数据集，内容主要来自维基百科的文章。它被设计用于评估和训练自然语言处理模型，尤其是在文本生成和理解方面。wikitext2数据集包含了大量的自然语言句子，结构接近真实世界的百科全书条目，适合用作模型量化时的校准数据集。

quantization_config = GPTQConfig(
     bits=4,              # 量化精度，表示将权重量化为4比特
     group_size=128,      # 分组大小，每128个权重为一组进行量化
     dataset="wikitext2", # 用于校准量化的校准数据集，这里选择"wikitext2"，它是一个英文维基百科语料库，适合语言模型的校准
     desc_act=False,      # 是否对激活值进行有序量化，False表示不进行有序激活量化
)

quant_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    device_map='auto',
force_download=True)

print(quant_model.model.decoder.layers[0].self_attn.q_proj.__dict__)
# 保存模型权重
quant_model.save_pretrained("models/opt-2.7b-gptq")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

text = "Merry Christmas! I'm glad to"
inputs = tokenizer(text, return_tensors="pt").to(0)

out = quant_model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))


