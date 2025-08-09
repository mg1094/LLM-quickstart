result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/cache"

model_name_or_path = 'THUDM/chatglm-6b'
train_data_path = 'HasturOfficial/adgen'
eval_data_path = None
seed = 8
max_input_length = 512
max_output_length = 1536
lora_rank = 4
lora_alpha = 32
lora_dropout = 0.05
resume_from_checkpoint = None
prompt_text = ''
compute_dtype = 'fp32'

from datasets import load_dataset 
dataset = load_dataset(train_data_path)
print(dataset)

#显示数据
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ,ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
        display(HTML(df.to_html()))

print(show_random_elements(dataset['train'],num_examples=1))

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,
revision='b098244')

def tokenize_func(example, tokenizer, ignore_lable_id=-100):
    question = prompt_text +  example['content']
    if example.get('input',None) and example['input'].strip():
        question += f'\n{example["input"]}'

    answer = example['summary']

    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    if len(q_ids) > max_input_length - 2:
        q_ids = q_ids[:max_input_length - 2]
    if len(a_ids) > max_output_length - 1:
        a_ids = a_ids[:max_output_length - 1]
    
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2

    labels = [ignore_lable_id] * question_length + input_ids[question_length:]

    return {'input_ids': input_ids, 'labels': labels}


column_names = dataset['train'].column_names
print(column_names)
tokenized_dataset = dataset['train'].map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False,
    remove_columns=column_names
)

print(len(tokenized_dataset))
print(show_random_elements(tokenized_dataset,num_examples=1))

tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
tokenized_dataset = tokenized_dataset.flatten_indices()

#定义数据整理器
import torch
from typing import List, Dict, Optional

class DataCollatorForChatGLM:
    def __init__(self, pad_token_id: int, max_length: int = 2048, ignore_lable_id: int = -100):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.ignore_lable_id = ignore_lable_id

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)

        input_dis, labels = [],[]

        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_lable_id] * pad_len

            if batch_max_len > self.max_length:
                ids = ids[:self.max_length]
                label = label[:self.max_length]
            input_dis.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))
       
        input_dis = torch.stack(input_dis, dim=0)
        labels = torch.stack(labels, dim=0)
        return {'input_ids': input_dis, 'labels': labels}


data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)

from transformers import AutoModel, BitsAndBytesConfig

_compute_dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


q_config = BitsAndBytesConfig(
    load_int_4bit = True,
    load_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type = _compute_dtype_map['bf16']

)

model = AutoModel.from_pretrained(model_name_or_path,
trust_remote_code=True,
revision='b098244',
quantization_config=q_config,
device_map='auto')

memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)
print(f"{memory_footprint_mib:.2f} MiB")

from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_kbit_training

kbit_model = prepare_model_for_kbit_training(model)

from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

print(target_modules)

lora_config = LoraConfig(
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)


qlora_model = get_peft_model(kbit_model, lora_config)
qlora_model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = f"models/{model_name_or_path}",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    num_train_epochs=1,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy='steps',
    save_steps=100,
    optimizer='adamw_torch',
    fp16=True,
)

trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.model.save_pretrained(f"models/demo/{model_name_or_path}")
