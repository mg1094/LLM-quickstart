#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取式问答模型训练演示
展示 start_positions 和 end_positions 在训练过程中的作用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleQAModel(nn.Module):
    """
    简化的问答模型，用于演示 start_positions 和 end_positions 的使用
    """
    def __init__(self, vocab_size=1000, hidden_size=128, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # 答案起始位置预测头
        self.start_head = nn.Linear(hidden_size, 1)
        # 答案结束位置预测头
        self.end_head = nn.Linear(hidden_size, 1)
        
        self.max_length = max_length
        
    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        """
        前向传播
        
        Args:
            input_ids: 输入的token ID序列
            attention_mask: 注意力掩码
            start_positions: 答案起始位置的标签（训练时使用）
            end_positions: 答案结束位置的标签（训练时使用）
        """
        # 1. 词嵌入
        embeddings = self.embedding(input_ids)
        
        # 2. 编码器处理
        lstm_out, _ = self.encoder(embeddings)
        
        # 3. 预测答案起始和结束位置的logits
        start_logits = self.start_head(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        end_logits = self.end_head(lstm_out).squeeze(-1)      # [batch_size, seq_len]
        
        # 4. 计算损失（仅在训练时）
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 忽略填充位置的损失
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits.view(-1)[active_loss]
            active_end_logits = end_logits.view(-1)[active_loss]
            active_start_positions = start_positions.view(-1)[active_loss]
            active_end_positions = end_positions.view(-1)[active_loss]
            
            # 计算交叉熵损失
            start_loss = F.cross_entropy(active_start_logits, active_start_positions)
            end_loss = F.cross_entropy(active_end_logits, active_end_positions)
            total_loss = (start_loss + end_loss) / 2
            
        return {
            'loss': total_loss,
            'start_logits': start_logits,
            'end_logits': end_logits
        }

def demo_training_process():
    """演示训练过程中如何使用 start_positions 和 end_positions"""
    
    print("=" * 60)
    print("抽取式问答模型训练演示")
    print("=" * 60)
    print()
    
    # 1. 准备训练数据
    print("=== 步骤1: 准备训练数据 ===")
    
    # 模拟经过 prepare_train_features 处理后的数据
    batch_data = {
        'input_ids': torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0],  # 样本1
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # 样本2
        ]),
        'attention_mask': torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # 样本1
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # 样本2
        ]),
        'start_positions': torch.tensor([8, 10]),   # 答案起始位置
        'end_positions': torch.tensor([12, 14])     # 答案结束位置
    }
    
    print(f"输入形状: {batch_data['input_ids'].shape}")
    print(f"注意力掩码形状: {batch_data['attention_mask'].shape}")
    print(f"答案起始位置: {batch_data['start_positions']}")
    print(f"答案结束位置: {batch_data['end_positions']}")
    print()
    
    # 2. 初始化模型
    print("=== 步骤2: 初始化模型 ===")
    model = SimpleQAModel(vocab_size=1000, hidden_size=64, max_length=18)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print()
    
    # 3. 前向传播（训练模式）
    print("=== 步骤3: 前向传播（训练模式）===")
    model.train()
    
    # 传入标签进行训练
    outputs = model(
        input_ids=batch_data['input_ids'],
        attention_mask=batch_data['attention_mask'],
        start_positions=batch_data['start_positions'],
        end_positions=batch_data['end_positions']
    )
    
    print(f"损失值: {outputs['loss'].item():.4f}")
    print(f"起始logits形状: {outputs['start_logits'].shape}")
    print(f"结束logits形状: {outputs['end_logits'].shape}")
    print()
    
    # 4. 损失计算详解
    print("=== 步骤4: 损失计算详解 ===")
    
    # 获取当前预测的logits
    start_logits = outputs['start_logits']
    end_logits = outputs['end_logits']
    
    # 计算预测的起始和结束位置
    pred_start_positions = start_logits.argmax(dim=-1)
    pred_end_positions = end_logits.argmax(dim=-1)
    
    print("预测结果 vs 真实标签:")
    for i in range(len(batch_data['start_positions'])):
        print(f"样本{i+1}:")
        print(f"  预测起始位置: {pred_start_positions[i].item()}, 真实: {batch_data['start_positions'][i].item()}")
        print(f"  预测结束位置: {pred_end_positions[i].item()}, 真实: {batch_data['end_positions'][i].item()}")
        print(f"  预测正确: {pred_start_positions[i].item() == batch_data['start_positions'][i].item() and pred_end_positions[i].item() == batch_data['end_positions'][i].item()}")
    print()
    
    # 5. 反向传播
    print("=== 步骤5: 反向传播 ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 计算梯度
    outputs['loss'].backward()
    
    # 获取梯度信息
    start_head_grad = model.start_head.weight.grad
    end_head_grad = model.end_head.weight.grad
    
    print(f"起始位置预测头梯度范数: {start_head_grad.norm().item():.4f}")
    print(f"结束位置预测头梯度范数: {end_head_grad.norm().item():.4f}")
    print()
    
    # 6. 参数更新
    print("=== 步骤6: 参数更新 ===")
    optimizer.step()
    optimizer.zero_grad()
    
    # 再次前向传播，查看损失是否下降
    outputs_after_update = model(
        input_ids=batch_data['input_ids'],
        attention_mask=batch_data['attention_mask'],
        start_positions=batch_data['start_positions'],
        end_positions=batch_data['end_positions']
    )
    
    print(f"更新前损失: {outputs['loss'].item():.4f}")
    print(f"更新后损失: {outputs_after_update['loss'].item():.4f}")
    print(f"损失变化: {outputs['loss'].item() - outputs_after_update['loss'].item():.4f}")
    print()
    
    # 7. 推理模式演示
    print("=== 步骤7: 推理模式演示 ===")
    model.eval()
    
    with torch.no_grad():
        # 不传入标签，只进行预测
        inference_outputs = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask']
        )
        
        print("推理结果:")
        print(f"起始logits: {inference_outputs['start_logits'][0][:10]}...")
        print(f"结束logits: {inference_outputs['end_logits'][0][:10]}...")
        
        # 获取最可能的答案位置
        pred_start = inference_outputs['start_logits'].argmax(dim=-1)
        pred_end = inference_outputs['end_logits'].argmax(dim=-1)
        
        print(f"预测的答案位置: 起始={pred_start.tolist()}, 结束={pred_end.tolist()}")
    
    print()
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)

def demo_loss_function():
    """详细演示损失函数的计算过程"""
    
    print("\n" + "=" * 60)
    print("损失函数计算详解")
    print("=" * 60)
    
    # 模拟数据
    batch_size, seq_len = 2, 10
    start_logits = torch.randn(batch_size, seq_len)
    end_logits = torch.randn(batch_size, seq_len)
    start_positions = torch.tensor([3, 5])  # 真实起始位置
    end_positions = torch.tensor([6, 8])    # 真实结束位置
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("=== 输入数据 ===")
    print(f"起始logits形状: {start_logits.shape}")
    print(f"结束logits形状: {end_logits.shape}")
    print(f"真实起始位置: {start_positions}")
    print(f"真实结束位置: {end_positions}")
    print()
    
    # 计算损失
    print("=== 损失计算步骤 ===")
    
    # 1. 应用注意力掩码
    active_loss = attention_mask.view(-1) == 1
    active_start_logits = start_logits.view(-1)[active_loss]
    active_end_logits = end_logits.view(-1)[active_loss]
    active_start_positions = start_positions.view(-1)[active_loss]
    active_end_positions = end_positions.view(-1)[active_loss]
    
    print(f"激活的起始logits数量: {len(active_start_logits)}")
    print(f"激活的结束logits数量: {len(active_end_logits)}")
    print()
    
    # 2. 计算交叉熵损失
    start_loss = F.cross_entropy(active_start_logits, active_start_positions)
    end_loss = F.cross_entropy(active_end_logits, active_end_positions)
    total_loss = (start_loss + end_loss) / 2
    
    print(f"起始位置损失: {start_loss.item():.4f}")
    print(f"结束位置损失: {end_loss.item():.4f}")
    print(f"总损失: {total_loss.item():.4f}")
    print()
    
    # 3. 分析预测结果
    print("=== 预测结果分析 ===")
    pred_start = start_logits.argmax(dim=-1)
    pred_end = end_logits.argmax(dim=-1)
    
    for i in range(batch_size):
        print(f"样本{i+1}:")
        print(f"  预测: 起始={pred_start[i].item()}, 结束={pred_end[i].item()}")
        print(f"  真实: 起始={start_positions[i].item()}, 结束={end_positions[i].item()}")
        print(f"  起始位置正确: {pred_start[i].item() == start_positions[i].item()}")
        print(f"  结束位置正确: {pred_end[i].item() == end_positions[i].item()}")
        print(f"  答案完全正确: {pred_start[i].item() == start_positions[i].item() and pred_end[i].item() == end_positions[i].item()}")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    demo_training_process()
    demo_loss_function() 