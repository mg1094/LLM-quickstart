# DeepSpeed 框架安装指南

## 更新 GCC 和 G++ 版本（如需）

首先，添加必要的 PPA 仓库，然后更新 `gcc` 和 `g++`：

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-7 g++-7
```

更新系统的默认 `gcc` 和 `g++` 指向：

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7
sudo update-alternatives --config gcc
```

## 创建隔离的 Anaconda 环境

如果想要隔离环境，建议采用 clone 方式，新建一个 DeepSpeed 专用的 Anaconda 环境：

```bash
conda create -n deepspeed --clone base
```

## 安装 Transformers 和 DeepSpeed

### 源代码安装 Transformers

遵循[官方文档](https://huggingface.co/docs/transformers/installation#install-from-source)，通过下面的命令安装 Transformers：

```bash
pip install git+https://github.com/huggingface/transformers
或者
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
离线下载包
pip install "transformers[torch]"
```

### 源代码安装 DeepSpeed

根据你的 GPU 实际情况设置参数 `TORCH_CUDA_ARCH_LIST`。如果你需要使用 CPU Offload 优化器参数，设置参数 `DS_BUILD_CPU_ADAM=1`；如果你需要使用 NVMe Offload，设置参数 `DS_BUILD_UTILS=1`：

3090显卡 对应的是 8.6
```bash
git clone https://github.com/microsoft/DeepSpeed/
git checkout tags/v0.13.1
cd DeepSpeed
rm -rf build
# TORCH_CUDA_ARCH_LIST="8.6": 设置 CUDA 架构列表为 8.6，适用于 3090 显卡
# DS_BUILD_CPU_ADAM=1: 启用 CPU Offload 优化器参数功能
# DS_BUILD_UTILS=1: 启用 NVMe Offload 功能
# --global-option="build_ext": 指定全局选项为构建扩展模块
# --global-option="-j8": 使用 8 个并行任务进行编译，加快编译速度
# --no-cache: 不使用缓存，确保获取最新的包
# -v: 启用详细输出模式，显示更多安装过程信息
# --disable-pip-version-check: 禁用 pip 版本检查
# 2>&1 | tee build.log: 将标准错误输出重定向到标准输出，并将输出内容同时保存到 build.log 文件中
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

**注意：不要在项目内 clone DeepSpeed 源代码安装，容易造成误提交。**

### 使用 DeepSpeed 训练 T5 系列模型

- 单机单卡训练脚本：[train_on_one_gpu.sh](train_on_one_gpu.sh)
- 分布式训练脚本：[train_on_multi_nodes.sh](train_on_multi_nodes.sh)