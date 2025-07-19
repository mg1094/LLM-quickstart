#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取式问答数据预处理演示（简化版）
展示 prepare_train_features 方法处理前后的数据变化
"""

import json

def simple_tokenizer(text, max_length=128):
    """
    简化的tokenizer，将文本按字符分割
    """
    # 简单的按字符分割，实际中会使用更复杂的tokenizer
    tokens = list(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokens

def prepare_train_features_simple_demo(examples, max_length=128, doc_stride=64):
    """
    简化版的 prepare_train_features 方法演示
    """
    print("=" * 60)
    print("抽取式问答数据预处理演示")
    print("=" * 60)
    print()
    
    # 1. 清理问题文本
    original_question = examples["question"][0]
    cleaned_question = original_question.lstrip()
    examples["question"] = [cleaned_question]
    
    print("=== 步骤1: 文本清理 ===")
    print(f"原始问题: '{original_question}'")
    print(f"清理后: '{cleaned_question}'")
    print()
    
    # 2. 模拟标记化处理
    question_tokens = simple_tokenizer(cleaned_question, max_length//2)
    context_tokens = simple_tokenizer(examples["context"][0], max_length//2)
    
    # 组合问题和上下文
    input_tokens = question_tokens + context_tokens
    if len(input_tokens) > max_length:
        input_tokens = input_tokens[:max_length]
    
    print("=== 步骤2: 标记化结果 ===")
    print(f"问题tokens: {question_tokens[:10]}...")
    print(f"上下文tokens: {context_tokens[:10]}...")
    print(f"组合后tokens长度: {len(input_tokens)}")
    print()
    
    # 3. 计算答案位置
    answers = examples["answers"]
    answer_text = answers["text"][0]
    answer_start_char = answers["answer_start"][0]
    
    print("=== 步骤3: 答案位置计算 ===")
    print(f"答案文本: '{answer_text}'")
    print(f"答案在原文中的起始位置: {answer_start_char}")
    
    # 在token序列中找到答案位置
    question_length = len(question_tokens)
    answer_start_token = question_length + answer_start_char
    answer_end_token = answer_start_token + len(answer_text)
    
    print(f"答案在token序列中的起始位置: {answer_start_token}")
    print(f"答案在token序列中的结束位置: {answer_end_token}")
    print()
    
    # 4. 构建最终数据
    processed_data = {
        "input_ids": input_tokens,
        "attention_mask": [1] * len(input_tokens),
        "start_positions": [answer_start_token],
        "end_positions": [answer_end_token]
    }
    
    print("=== 步骤4: 处理后的数据 ===")
    print(f"输入tokens: {input_tokens[:20]}...")
    print(f"注意力掩码: {processed_data['attention_mask'][:20]}...")
    print(f"答案起始位置: {processed_data['start_positions']}")
    print(f"答案结束位置: {processed_data['end_positions']}")
    print()
    
    # 5. 验证答案
    print("=== 步骤5: 答案验证 ===")
    start_pos = processed_data['start_positions'][0]
    end_pos = processed_data['end_positions'][0]
    
    if start_pos < len(input_tokens) and end_pos <= len(input_tokens):
        predicted_answer = ''.join(input_tokens[start_pos:end_pos])
        print(f"预测的答案: '{predicted_answer}'")
        print(f"原始答案: '{answer_text}'")
        print(f"答案匹配: {predicted_answer == answer_text}")
    else:
        print("答案位置超出token序列范围")
    
    print()
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    return processed_data

def demo_with_multiple_examples():
    """演示多个样本的处理"""
    
    print("\n" + "=" * 60)
    print("多样本处理演示")
    print("=" * 60)
    
    # 多个样本数据
    examples = {
        "question": [
            "  什么是人工智能？",
            "  AI的主要应用领域有哪些？",
            "  机器学习的定义是什么？"
        ],
        "context": [
            "人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "人工智能的主要应用领域包括机器人、语言识别、图像识别、自然语言处理和专家系统等。",
            "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。"
        ],
        "answers": {
            "text": ["人工智能", "机器人、语言识别、图像识别、自然语言处理和专家系统", "机器学习"],
            "answer_start": [0, 15, 0]
        }
    }
    
    print("原始数据:")
    for i in range(len(examples["question"])):
        print(f"样本{i+1}:")
        print(f"  问题: {examples['question'][i]}")
        print(f"  上下文: {examples['context'][i]}")
        print(f"  答案: '{examples['answers']['text'][i]}'")
        print()
    
    # 处理每个样本
    for i in range(len(examples["question"])):
        print(f"处理样本 {i+1}:")
        single_example = {
            "question": [examples["question"][i]],
            "context": [examples["context"][i]],
            "answers": {
                "text": [examples["answers"]["text"][i]],
                "answer_start": [examples["answers"]["answer_start"][i]]
            }
        }
        prepare_train_features_simple_demo(single_example)
        print()

if __name__ == "__main__":
    # 单个样本演示
    raw_data = {
        "question": ["  什么是人工智能？"],
        "context": [
            "人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        ],
        "answers": {
            "text": ["人工智能"],
            "answer_start": [0]
        }
    }
    
    prepare_train_features_simple_demo(raw_data)
    
    # 多样本演示
    demo_with_multiple_examples() 