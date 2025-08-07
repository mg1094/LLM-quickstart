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


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "models/opt-125m-awq"
quant_config = {"zero_point":True, "q_group_size":128, "w_bit":4, "version":"GEMM"}

model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="cuda")
# trust_remote_code=True 表示允许从远程加载自定义的分词器代码（比如模型作者自定义的分词器实现），
# 这样可以确保即使模型仓库中有特殊的分词器实现，也能正确加载和使用。
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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

