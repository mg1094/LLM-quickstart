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