#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取式问答数据预处理演示
展示 prepare_train_features 方法处理前后的数据变化
"""

from transformers import AutoTokenizer
import json

def prepare_train_features_demo(examples, tokenizer, max_length=384, doc_stride=128, pad_on_right=True):
    """
    演示版本的 prepare_train_features 方法
    """
    # 1. 清理问题文本
    examples["question"] = [q.lstrip() for q in examples["question"]]
    
    print("=== 步骤1: 清理后的问题 ===")
    print(f"原始问题: '  什么是人工智能？'")
    print(f"清理后: '{examples['question'][0]}'")
    print()
    
    # 2. 标记化处理
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    print("=== 步骤2: 标记化结果 ===")
    print(f"输入ID长度: {len(tokenized_examples['input_ids'][0])}")
    print(f"注意力掩码长度: {len(tokenized_examples['attention_mask'][0])}")
    print(f"偏移映射数量: {len(tokenized_examples['offset_mapping'][0])}")
    print()
    
    # 3. 提取映射信息
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    print("=== 步骤3: 映射信息 ===")
    print(f"样本映射: {sample_mapping}")
    print(f"偏移映射前5个: {offset_mapping[0][:5]}")
    print()
    
    # 4. 计算答案位置
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            # 找到上下文的token范围
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            
            # 检查答案是否在当前span内
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # 计算答案的token位置
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    
    return tokenized_examples

def demo_qa_data_processing():
    """演示问答数据处理流程"""
    
    # 初始化tokenizer
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("=" * 60)
    print("抽取式问答数据预处理演示")
    print("=" * 60)
    print()
    
    # 模拟原始数据
    raw_data = {
        "question": ["  什么是人工智能？"],
        "context": [
            "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。"
        ],
        "answers": {
            "text": ["人工智能"],
            "answer_start": [0]
        }
    }
    
    print("=== 原始数据 ===")
    print(f"问题: {raw_data['question'][0]}")
    print(f"上下文: {raw_data['context'][0][:100]}...")
    print(f"答案: '{raw_data['answers']['text'][0]}'")
    print(f"答案起始位置: {raw_data['answers']['answer_start'][0]}")
    print()
    
    # 处理数据
    processed_data = prepare_train_features_demo(
        raw_data, 
        tokenizer, 
        max_length=128,  # 使用较小的长度便于演示
        doc_stride=64
    )
    
    print("=== 处理后的数据 ===")
    print(f"输入ID: {processed_data['input_ids'][0][:20]}...")
    print(f"注意力掩码: {processed_data['attention_mask'][0][:20]}...")
    print(f"答案起始位置: {processed_data['start_positions']}")
    print(f"答案结束位置: {processed_data['end_positions']}")
    print()
    
    # 验证答案位置
    print("=== 答案位置验证 ===")
    start_pos = processed_data['start_positions'][0]
    end_pos = processed_data['end_positions'][0]
    
    if start_pos != tokenizer.cls_token_id and end_pos != tokenizer.cls_token_id:
        # 解码答案token
        answer_tokens = processed_data['input_ids'][0][start_pos:end_pos+1]
        decoded_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        print(f"预测的答案: '{decoded_answer}'")
        print(f"原始答案: '{raw_data['answers']['text'][0]}'")
        print(f"答案匹配: {decoded_answer == raw_data['answers']['text'][0]}")
    else:
        print("答案位置使用CLS token标记（可能超出当前span）")
    
    print()
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    demo_qa_data_processing() 