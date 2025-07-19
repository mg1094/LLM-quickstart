#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取式问答模型训练概念演示
展示 start_positions 和 end_positions 在训练过程中的作用（纯概念版）
"""

import numpy as np
import math

def simulate_qa_model_forward(input_ids, start_positions=None, end_positions=None):
    """
    模拟问答模型的前向传播
    
    Args:
        input_ids: 输入的token ID序列
        start_positions: 答案起始位置的标签（训练时使用）
        end_positions: 答案结束位置的标签（训练时使用）
    """
    batch_size, seq_len = input_ids.shape
    
    # 模拟模型输出：为每个位置生成logits
    np.random.seed(42)  # 固定随机种子以便演示
    start_logits = np.random.randn(batch_size, seq_len)
    end_logits = np.random.randn(batch_size, seq_len)
    
    # 模拟模型对答案位置的偏好（让模型稍微倾向于正确答案）
    if start_positions is not None and end_positions is not None:
        for i in range(batch_size):
            # 增加正确答案位置的logits
            start_logits[i, start_positions[i]] += 0.5
            end_logits[i, end_positions[i]] += 0.5
    
    return start_logits, end_logits

def calculate_loss(start_logits, end_logits, start_positions, end_positions, attention_mask):
    """
    计算交叉熵损失
    
    Args:
        start_logits: 起始位置预测logits
        end_logits: 结束位置预测logits
        start_positions: 真实起始位置
        end_positions: 真实结束位置
        attention_mask: 注意力掩码
    """
    batch_size, seq_len = start_logits.shape
    
    # 应用注意力掩码，忽略填充位置
    total_loss = 0
    start_loss = 0
    end_loss = 0
    active_count = 0
    
    for i in range(batch_size):
        for j in range(seq_len):
            if attention_mask[i, j] == 1:  # 只计算有效位置的损失
                # 计算起始位置损失
                start_probs = softmax(start_logits[i])
                start_loss -= math.log(start_probs[start_positions[i]] + 1e-8)
                
                # 计算结束位置损失
                end_probs = softmax(end_logits[i])
                end_loss -= math.log(end_probs[end_positions[i]] + 1e-8)
                
                active_count += 1
    
    if active_count > 0:
        start_loss /= active_count
        end_loss /= active_count
        total_loss = (start_loss + end_loss) / 2
    
    return total_loss, start_loss, end_loss

def softmax(x):
    """计算softmax概率"""
    exp_x = np.exp(x - np.max(x))  # 数值稳定性
    return exp_x / np.sum(exp_x)

def demo_training_process():
    """演示训练过程中如何使用 start_positions 和 end_positions"""
    
    print("=" * 60)
    print("抽取式问答模型训练概念演示")
    print("=" * 60)
    print()
    
    # 1. 准备训练数据
    print("=== 步骤1: 准备训练数据 ===")
    
    # 模拟经过 prepare_train_features 处理后的数据
    batch_data = {
        'input_ids': np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0],  # 样本1
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # 样本2
        ]),
        'attention_mask': np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 样本1
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # 样本2
        ]),
        'start_positions': np.array([8, 10]),   # 答案起始位置
        'end_positions': np.array([12, 14])     # 答案结束位置
    }
    
    print(f"输入形状: {batch_data['input_ids'].shape}")
    print(f"注意力掩码形状: {batch_data['attention_mask'].shape}")
    print(f"答案起始位置: {batch_data['start_positions']}")
    print(f"答案结束位置: {batch_data['end_positions']}")
    print()
    
    # 2. 前向传播（训练模式）
    print("=== 步骤2: 前向传播（训练模式）===")
    
    # 传入标签进行训练
    start_logits, end_logits = simulate_qa_model_forward(
        batch_data['input_ids'],
        start_positions=batch_data['start_positions'],
        end_positions=batch_data['end_positions']
    )
    
    print(f"起始logits形状: {start_logits.shape}")
    print(f"结束logits形状: {end_logits.shape}")
    print()
    
    # 3. 损失计算
    print("=== 步骤3: 损失计算 ===")
    
    total_loss, start_loss, end_loss = calculate_loss(
        start_logits, end_logits,
        batch_data['start_positions'],
        batch_data['end_positions'],
        batch_data['attention_mask']
    )
    
    print(f"总损失: {total_loss:.4f}")
    print(f"起始位置损失: {start_loss:.4f}")
    print(f"结束位置损失: {end_loss:.4f}")
    print()
    
    # 4. 预测结果分析
    print("=== 步骤4: 预测结果分析 ===")
    
    # 计算预测的起始和结束位置
    pred_start_positions = np.argmax(start_logits, axis=-1)
    pred_end_positions = np.argmax(end_logits, axis=-1)
    
    print("预测结果 vs 真实标签:")
    for i in range(len(batch_data['start_positions'])):
        print(f"样本{i+1}:")
        print(f"  预测起始位置: {pred_start_positions[i]}, 真实: {batch_data['start_positions'][i]}")
        print(f"  预测结束位置: {pred_end_positions[i]}, 真实: {batch_data['end_positions'][i]}")
        print(f"  起始位置正确: {pred_start_positions[i] == batch_data['start_positions'][i]}")
        print(f"  结束位置正确: {pred_end_positions[i] == batch_data['end_positions'][i]}")
        print(f"  答案完全正确: {pred_start_positions[i] == batch_data['start_positions'][i] and pred_end_positions[i] == batch_data['end_positions'][i]}")
    print()
    
    # 5. 模拟参数更新
    print("=== 步骤5: 模拟参数更新 ===")
    
    # 模拟梯度更新后的模型输出（损失应该下降）
    print("模拟梯度更新...")
    
    # 再次前向传播，模拟更新后的模型
    updated_start_logits, updated_end_logits = simulate_qa_model_forward(
        batch_data['input_ids'],
        start_positions=batch_data['start_positions'],
        end_positions=batch_data['end_positions']
    )
    
    # 增加正确答案位置的logits（模拟学习效果）
    for i in range(len(batch_data['start_positions'])):
        updated_start_logits[i, batch_data['start_positions'][i]] += 1.0
        updated_end_logits[i, batch_data['end_positions'][i]] += 1.0
    
    # 计算更新后的损失
    updated_total_loss, updated_start_loss, updated_end_loss = calculate_loss(
        updated_start_logits, updated_end_logits,
        batch_data['start_positions'],
        batch_data['end_positions'],
        batch_data['attention_mask']
    )
    
    print(f"更新前损失: {total_loss:.4f}")
    print(f"更新后损失: {updated_total_loss:.4f}")
    print(f"损失变化: {total_loss - updated_total_loss:.4f}")
    print()
    
    # 6. 推理模式演示
    print("=== 步骤6: 推理模式演示 ===")
    
    # 不传入标签，只进行预测
    inference_start_logits, inference_end_logits = simulate_qa_model_forward(
        batch_data['input_ids']
    )
    
    print("推理结果:")
    print(f"起始logits样本: {inference_start_logits[0][:10]}...")
    print(f"结束logits样本: {inference_end_logits[0][:10]}...")
    
    # 获取最可能的答案位置
    pred_start = np.argmax(inference_start_logits, axis=-1)
    pred_end = np.argmax(inference_end_logits, axis=-1)
    
    print(f"预测的答案位置: 起始={pred_start.tolist()}, 结束={pred_end.tolist()}")
    
    print()
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)

def demo_loss_function_details():
    """详细演示损失函数的计算过程"""
    
    print("\n" + "=" * 60)
    print("损失函数计算详解")
    print("=" * 60)
    
    # 模拟数据
    batch_size, seq_len = 2, 10
    start_logits = np.random.randn(batch_size, seq_len)
    end_logits = np.random.randn(batch_size, seq_len)
    start_positions = np.array([3, 5])  # 真实起始位置
    end_positions = np.array([6, 8])    # 真实结束位置
    attention_mask = np.ones((batch_size, seq_len))
    
    print("=== 输入数据 ===")
    print(f"起始logits形状: {start_logits.shape}")
    print(f"结束logits形状: {end_logits.shape}")
    print(f"真实起始位置: {start_positions}")
    print(f"真实结束位置: {end_positions}")
    print()
    
    # 计算损失
    print("=== 损失计算步骤 ===")
    
    # 1. 应用注意力掩码
    active_positions = attention_mask == 1
    print(f"激活位置数量: {np.sum(active_positions)}")
    print()
    
    # 2. 计算每个样本的损失
    for i in range(batch_size):
        print(f"样本{i+1}的损失计算:")
        
        # 起始位置损失
        start_probs = softmax(start_logits[i])
        start_loss = -math.log(start_probs[start_positions[i]] + 1e-8)
        print(f"  起始位置概率分布: {start_probs[:5]}...")
        print(f"  真实起始位置: {start_positions[i]}")
        print(f"  对应概率: {start_probs[start_positions[i]]:.4f}")
        print(f"  起始位置损失: {start_loss:.4f}")
        
        # 结束位置损失
        end_probs = softmax(end_logits[i])
        end_loss = -math.log(end_probs[end_positions[i]] + 1e-8)
        print(f"  结束位置概率分布: {end_probs[:5]}...")
        print(f"  真实结束位置: {end_positions[i]}")
        print(f"  对应概率: {end_probs[end_positions[i]]:.4f}")
        print(f"  结束位置损失: {end_loss:.4f}")
        
        total_sample_loss = (start_loss + end_loss) / 2
        print(f"  样本总损失: {total_sample_loss:.4f}")
        print()
    
    # 3. 分析预测结果
    print("=== 预测结果分析 ===")
    pred_start = np.argmax(start_logits, axis=-1)
    pred_end = np.argmax(end_logits, axis=-1)
    
    for i in range(batch_size):
        print(f"样本{i+1}:")
        print(f"  预测: 起始={pred_start[i]}, 结束={pred_end[i]}")
        print(f"  真实: 起始={start_positions[i]}, 结束={end_positions[i]}")
        print(f"  起始位置正确: {pred_start[i] == start_positions[i]}")
        print(f"  结束位置正确: {pred_end[i] == end_positions[i]}")
        print(f"  答案完全正确: {pred_start[i] == start_positions[i] and pred_end[i] == end_positions[i]}")
    
    print()
    print("=" * 60)

def demo_gradient_flow():
    """演示梯度流动过程"""
    
    print("\n" + "=" * 60)
    print("梯度流动演示")
    print("=" * 60)
    
    print("=== 梯度计算过程 ===")
    print("1. 模型输出 start_logits 和 end_logits")
    print("2. 计算与真实标签 start_positions 和 end_positions 的损失")
    print("3. 损失函数: CrossEntropyLoss(start_logits, start_positions) + CrossEntropyLoss(end_logits, end_positions)")
    print("4. 反向传播计算梯度")
    print("5. 更新模型参数")
    print()
    
    print("=== 关键点 ===")
    print("• start_positions 和 end_positions 作为监督信号指导模型学习")
    print("• 模型需要学会预测答案在输入序列中的确切位置")
    print("• 损失函数确保模型输出的logits在正确答案位置有更高的值")
    print("• 通过反向传播，模型逐渐学会定位答案")
    print()
    
    print("=== 训练效果 ===")
    print("• 模型学会理解问题和上下文的关系")
    print("• 能够准确定位答案的起始和结束位置")
    print("• 提高答案抽取的准确性和完整性")
    print()
    
    print("=" * 60)

if __name__ == "__main__":
    demo_training_process()
    demo_loss_function_details()
    demo_gradient_flow() 