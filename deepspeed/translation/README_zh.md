<!---
版权所有 2020 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版（"许可证"）授权；
除非符合许可证的规定，否则您不得使用此文件。
您可以在以下位置获得许可证副本

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则软件
根据许可证分发是在"按原样"的基础上分发的，
不附带任何明示或暗示的担保或条件。
请参阅许可证了解特定语言的管理权限和限制。
-->

## 翻译

此目录包含在翻译任务上微调和评估 transformers 的示例。
如有任何问题/意外行为，请标记 @patil-suraj 或发送 PR！
有关已弃用的 `bertabs` 指令，请参见 [`bertabs/README.md`](https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertabs/README.md)。
有关旧版 `finetune_trainer.py` 和相关工具，请参见 [`examples/legacy/seq2seq`](https://github.com/huggingface/transformers/blob/main/examples/legacy/seq2seq)。

### 支持的架构

- `BartForConditionalGeneration`
- `FSMTForConditionalGeneration`（仅限翻译）
- `MBartForConditionalGeneration`
- `MarianMTModel`
- `PegasusForConditionalGeneration`
- `T5ForConditionalGeneration`
- `MT5ForConditionalGeneration`

`run_translation.py` 是一个轻量级示例，展示如何从 [🤗 Datasets](https://github.com/huggingface/datasets) 库下载和预处理数据集，或使用您自己的文件（jsonlines 或 csv），然后在上述架构之一上进行微调。

有关 `jsonlines` 格式的自定义数据集，请参见：https://huggingface.co/docs/datasets/loading_datasets#json-files
您也可以在下面找到这些示例。


## 使用 Trainer

以下是使用 MarianMT 模型进行翻译微调的示例：

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

MBart 和一些 T5 模型需要特殊处理。

T5 模型 `t5-small`、`t5-base`、`t5-large`、`t5-3b` 和 `t5-11b` 必须使用附加参数：`--source_prefix "translate {source_lang} to {target_lang}"`。例如：

```bash
python translation/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

如果您得到的 BLEU 分数很差，请确保您没有忘记使用 `--source_prefix` 参数。

对于上述 T5 模型组，重要的是要记住，如果您切换到不同的语言对，请确保调整所有 3 个特定于语言的命令行参数中的源和目标值：`--source_lang`、`--target_lang` 和 `--source_prefix`。

MBart 模型需要不同的 `--source_lang` 和 `--target_lang` 值格式，例如，它期望 `en_XX` 而不是 `en`，对于 `ro` 它期望 `ro_RO`。MBart 语言代码的完整规范可以在[这里](https://huggingface.co/facebook/mbart-large-cc25)找到。例如：

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path facebook/mbart-large-en-ro  \
    --do_train \
    --do_eval \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

以下是您如何在自己的文件上使用翻译微调，在调整参数 `--train_file`、`--validation_file` 的值以匹配您的设置后：

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --train_file path_to_jsonlines_file \
    --validation_file path_to_jsonlines_file \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

翻译任务仅支持自定义 JSONLINES 文件，每行是一个字典，键为 `"translation"`，其值是另一个字典，键为语言对。例如：

```json
{ "translation": { "en": "Others have dismissed him as a joke.", "ro": "Alții l-au numit o glumă." } }
{ "translation": { "en": "And some are holding out for an implosion.", "ro": "Iar alții așteaptă implozia." } }
```
这里的语言是罗马尼亚语（`ro`）和英语（`en`）。

如果您想使用预处理的数据集以获得高 BLEU 分数，但针对 `en-de` 语言对，您可以使用 `--dataset_name stas/wmt14-en-de-pre-processed`，如下所示：

```bash
python examples/pytorch/translation/run_translation.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name stas/wmt14-en-de-pre-processed \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate
```

## 使用 Accelerate

基于脚本 [`run_translation_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation_no_trainer.py)。

与 `run_translation.py` 类似，此脚本允许您在翻译任务上微调任何受支持的模型，主要区别是此脚本暴露了裸训练循环，允许您快速试验并添加任何想要的自定义。

它提供的选项比使用 `Trainer` 的脚本少（例如，您可以直接在脚本中轻松更改优化器或数据加载器的选项），但仍可在分布式设置中运行，在 TPU 上运行，并通过 [🤗 `Accelerate`](https://github.com/huggingface/accelerate) 库支持混合精度。安装后您可以正常使用该脚本：

```bash
pip install git+https://github.com/huggingface/accelerate
```

然后

```bash
python run_translation_no_trainer.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir ~/tmp/tst-translation
```

然后您可以使用常用的启动器在分布式环境中运行它，但最简单的方法是运行

```bash
accelerate config
```

并回答提出的问题。然后

```bash
accelerate test
```

这将检查训练所需的一切是否已准备就绪。最后，您可以使用以下命令启动训练：

```bash
accelerate launch run_translation_no_trainer.py \
    --model_name_or_path Helsinki-NLP/opus-mt-en-ro \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir ~/tmp/tst-translation
```

此命令是相同的，适用于：

- 仅 CPU 设置
- 单 GPU 设置
- 多 GPU 的分布式训练（单节点或多节点）
- TPU 上的训练

请注意，此库处于 alpha 发布阶段，因此如果您在使用过程中遇到任何问题，欢迎提供反馈。