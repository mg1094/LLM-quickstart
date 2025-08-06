import os
os.environ["HF_HOME"] = "/root/autodl-tmp/cache"

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

# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name_or_path = "facebook/opt-2.7b"
# custom_dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]

# custom_quantization_config = GPTQConfig(
#     bits=4,
#     group_size=128,
#     desc_act=False,
#     dataset=custom_dataset
# )

# custom_quant_model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     quantization_config=custom_quantization_config,
#     torch_dtype=torch.float16,
#     device_map='auto'
# )

# text = "Merry Christmas! I'm glad to"
# # return_tensors="pt" 的作用是让分词器（tokenizer）将文本编码后直接返回 PyTorch 张量（torch.Tensor）格式，方便后续直接输入到模型中进行推理或训练。
# inputs = tokenizer(text, return_tensors="pt").to(0)
# out = custom_quant_model.generate(**inputs, max_new_tokens=64)
# print(tokenizer.decode(out[0],skip_special_tokens=True))


