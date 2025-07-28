modle_name_or_path = "openai/whisper-large-v2"
modle_dir = "modles/whisper-large-v2-asr-int8"
language = "Chinese(China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"
batch_size = 64

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset, DatasetDict
common_voice = DatasetDict()
common_voice["train"]  = load_dataset(dataset_name, language_abbr, split="train", trust_remote_code=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="validation", trust_remote_code=True)
print(common_voice["train"][0])


from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor

feature_extractor = AutoFeatureExtractorer.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

print(common_voice["train"][0])


from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling=16000))
print(common_voice["train"][0])

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

small_common_voice = DatasetDict()
small_common_voice["train"] = common_voince["train"].shuffe(seed=16).select(range(640))
small_common_voice["validation"] = common_voice["validation"].shuffe(seed=16).select(range(320))

tokenized_common_voice = small_common_voice.map(prepare_dataset)


import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddding:
    precessor: Any

    def __call__(self, fetures: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.atttention_mask.ne(1),-100)

        if(label[:,0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:,1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)        

from transformers import AutoModelForSpeechSeq2Seq
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, load_int8=Ture,device_map="auto")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model)

from peft import LoraConfig, PeftModel, LoraConfig, get_peft_model

config = LoraConfig(
    r=4,
    lora_alpha=64,
    target_modudels=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

from transformers import Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir=modle_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=1e-3,
    num_train_epochs=10,
    evaluation_strategy="steps",
    per_device_eval_batch_size=batch_size,
    generation_max_length=128,
    logging_steps=10,
    remove_unused_columns=False,
    label_names=["lablels"],
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=peft_model,
    trian_dataset=tokenized_common_voice["train"],
    eval_dataet=tokenized_common_voice["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
peft_model.config.use_cache = False
trainer.train()

trainer.save_model(modle-dir)
peft_model.eval()