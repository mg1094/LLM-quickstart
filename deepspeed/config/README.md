# DeepSpeed 配置文件说明

该目录包含了用于不同 ZeRO 优化级别的 DeepSpeed 配置文件。

## ds_config_zero2.json

此配置文件使用 ZeRO-2 优化策略，主要特点包括：

1. **混合精度训练**：支持 FP16 和 BF16 混合精度训练，通过动态损失缩放来保持训练稳定性。
2. **优化器配置**：使用 AdamW 优化器，具有自动调整的学习率、beta 系数、epsilon 和权重衰减。
3. **学习率调度**：使用 WarmupLR 调度器，自动调整热身阶段的学习率。
4. **ZeRO-2 优化**：将优化器状态和梯度进行分片，显著减少 GPU 内存使用。
   - 支持将优化器状态卸载到 CPU 内存
   - 通过分片和通信优化来提高训练效率

## ds_config_zero3.json

此配置文件使用 ZeRO-3 优化策略，是最高级别的内存优化技术，主要特点包括：

1. **混合精度训练**：支持 FP16 和 BF16 混合精度训练。
2. **优化器配置**：使用 AdamW 优化器。
3. **学习率调度**：使用 WarmupLR 调度器。
4. **ZeRO-3 优化**：将模型参数、梯度和优化器状态都进行分片，实现最大化的内存节省。
   - 支持将优化器状态和模型参数卸载到 CPU 内存
   - 多种参数控制内存使用和性能平衡

### ZeRO-3 特有参数说明

1. `offload_param`：将模型参数卸载到 CPU 内存，类似把不常用的书籍放到书架上。
2. `stage3_prefetch_bucket_size`：预取参数的数量，类似提前准备下一章节的内容。
3. `stage3_param_persistence_threshold`：决定哪些参数需要一直保留在 GPU 内存中。
4. `stage3_max_live_parameters`：控制同时在 GPU 内存中的参数数量，类似限制工作台上的书本数量。
5. `stage3_max_reuse_distance`：控制参数在被再次使用之前可以被移动到多远的存储位置。
6. `stage3_gather_16bit_weights_on_model_save`：在保存模型时整合所有分片的权重，类似整理散落的笔记成完整文档。

### stage3_max_live_parameters 和 stage3_max_reuse_distance 详解

- `stage3_max_live_parameters`：控制任何时候可以保留在 GPU 内存中的最大参数数量。设置得越大，GPU 内存中保留的参数就越多，相应地需要在 CPU 内存中保存的参数就越少。

- `stage3_max_reuse_distance`：控制参数在被再次使用之前可以被移动到 CPU 内存的最大距离。设置得越大，参数可以被移动到更远的存储位置，从而减少 GPU 内存使用，但可能会增加数据传输开销。

这两个参数协同工作来优化内存使用，就像图书馆管理员管理书籍一样，需要在可用空间和访问效率之间找到平衡。