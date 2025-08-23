from datasets import load_dataset
from random import randrange

dataset = load_dataset("databricks/databricks-dolly-15k",split="train")
print(dataset)
print(dataset[randrange(len(dataset))])

def format_instruction(sample):
    if 'response' not in sample or 'instruction' not in sample:
        return "Error: 'response' or 'instruction' key missing in the input data"

    return f"""### Instruction:
    Use the Input below to create an instruction, which could have been used to generate the input using an LLM.
    ### Input:
    {sample['response']}

    ### Response:
    {sample['instruction']}
    """

print(format_instruction(dataset[randrange(len(dataset))]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "NousResearch/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    # load_in_4bit：将模型加载为4位量化模型，可显著减少模型内存占用，从而支持在显存较小的设备上加载大模型。
    load_in_4bit=True,
    # bnb_4bit_use_double_quant：启用二次量化，即在第一次量化的基础上对量化常数进行再次量化，进一步减少内存使用。
    bnb_4bit_use_double_quant=True,
    # bnb_4bit_quant_type：指定4位量化的类型，"nf4" 表示使用NormalFloat 4位量化，这是一种专为Transformer模型设计的量化类型。
    bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype：指定模型计算时使用的数据类型，torch.bfloat16 是一种16位浮点数格式，在保持计算精度的同时能提升计算速度。
    bnb_4bit_compute_dtype=torch.bfloat16   
)

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            quantization_config=bnb_config,
                                            use_cache=False,
                                            device_map="auto"
                                            )

# 设置模型配置中的预训练张量并行度(pretraining_tp)为1，即不使用张量并行，单张量处理。
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Set the padding token to the end-of-sequence token to ensure padding uses the end token
tokenizer.pad_token = tokenizer.eos_token
# Set the padding direction to the right
tokenizer.padding_side = "right"

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
qlora_model = get_peft_model(model,peft_config)
qlora_model.print_trainable_parameters()

import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
demo_train = True
output_dir = f"models/llama-7-int4-dlly-{timestamp}"

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir=output_dir,  # 模型训练过程中输出文件（如检查点、日志等）的保存目录
    num_train_epochs=3,  # 训练的总轮数
    # max_steps=100,  # 训练的最大步数，与num_train_epochs互斥，优先使用max_steps
    per_device_train_batch_size=3,  # 每个设备上的训练批次大小
    gradient_accumulation_steps=4,  # 梯度累积步数，用于模拟更大的批次大小
    gradient_checkpointing=True,  # 是否启用梯度检查点，以减少内存使用但会增加计算时间
    optim="paged_adamw_32bit",  # 使用的优化器类型，这里是32位的分页AdamW优化器
    logging_steps=100,  # 每隔多少步记录一次训练日志
    save_steps=100,  # 每隔多少步保存一次模型检查点
    learning_rate=2e-4,  # 学习率，控制模型参数更新的步长
    bf16=True,  # 是否使用bfloat16精度进行训练，可减少内存占用并加速训练
    max_grad_norm=0.3,  # 梯度裁剪的最大范数，用于防止梯度爆炸
    warmup_ratio=0.03,  # 学习率热身的比例，在训练初期逐渐增加学习率
    lr_scheduler_type="constant",  # 学习率调度器的类型，这里是常数学习率
)

from trl import SFTTrainer 
max_seq_length = 2048

trainer = SFTTrainer(
    model=qlora_model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting=format_instruction,
    args=args,
)

trainer.train()
trainer.save_model()