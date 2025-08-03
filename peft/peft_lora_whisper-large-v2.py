import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/cache"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/cache/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/cache/datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/cache/hub"      

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

import torch

# 检查 CUDA 是否可用
is_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {is_available}")

if is_available:
    # 获取 GPU 数量
    device_count = torch.cuda.device_count()
    print(f"可用的 GPU 数量: {device_count}")
    
    # 获取当前 GPU 的名称
    current_device_name = torch.cuda.get_device_name(0)
    print(f"当前 GPU 名称: {current_device_name}")
else:
    print("PyTorch 未能检测到任何可用的 CUDA 设备。")    


from datasets import load_dataset, DatasetDict, Audio
from transformers import (AutoFeatureExtractor, AutoTokenizer, AutoProcessor,
                          AutoModelForSpeechSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# 全局参数
model_name_or_path = "openai/whisper-large-v2"
model_dir = "models/whisper-large-v2-asr-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"
batch_size = 64

# 使用镜像加速（如不需要可注释）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 加载数据集
common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset(dataset_name, language_abbr, split="validation", trust_remote_code=True)

# 2. 移除无用字段
common_voice = common_voice.remove_columns([
    "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
])

# 3. 降采样音频
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# 4. 特征提取器、分词器、处理器
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch     

# 5. 多进程加速特征提取（如阻塞可调小 num_proc）
if os.path.exists("tokenized_common_voice"):
    from datasets import load_from_disk
    tokenized_common_voice = load_from_disk("tokenized_common_voice")
else:
    tokenized_common_voice = common_voice.map(prepare_dataset, num_proc=4)
    tokenized_common_voice.save_to_disk("tokenized_common_voice")

# 6. 数据整理器
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 7. 加载模型（int8）
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model = prepare_model_for_int8_training(model)

# 8. LoRA 配置
config = LoraConfig(
    r=4,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# 9. 训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=1e-3,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    fp16=True,  # 混合精度加速
    per_device_eval_batch_size=batch_size,
    generation_max_length=128,
    logging_steps=10,
    remove_unused_columns=False,
    label_names=["labels"],
)

# 10. Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=peft_model,
    train_dataset=tokenized_common_voice["train"],
    eval_dataset=tokenized_common_voice["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
peft_model.config.use_cache = False

# 11. 训练
trainer.train()

# 12. 保存模型
trainer.save_model(model_dir)
