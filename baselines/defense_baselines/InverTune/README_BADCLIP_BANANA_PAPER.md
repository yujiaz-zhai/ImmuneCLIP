# InverTune 对 banana-BadCLIP 的修正版复现说明

本文档说明如何直接使用已有 banana-BadCLIP RN50 中毒模型，复现
InverTune 的触发器反演、Activation Tuning 和 ImageNet CA/ASR 评测。
本实验不重新训练 BadCLIP，因此不会使用 CC3M。

论文地址：<https://arxiv.org/abs/2506.12411>

## 1. 本机实验输入

| 内容 | 路径 |
|---|---|
| ImageNet-1K 验证集（Parquet，50,000 张） | `/root/autodl-tmp/datasets/imagenet1k` |
| OpenAI CLIP RN50 预训练权重 | `/root/autodl-tmp/checkpoints/clip-clean-pretrained/RN50.pt` |
| 已有 banana-BadCLIP 中毒模型 | `/root/autodl-tmp/experiments/badclip_sbl/baseline_1_1_readme_strict/logs/nodefence_badCLIP/checkpoints/epoch_10.pt` |
| 原始 BadCLIP 补丁 | `/root/workspace/aaai-backdoor/baselines/BadCLIP/opti_patches/badCLIP.jpg` |
| 修正版配置 | `/root/autodl-tmp/experiments/invertune/config/badclip_banana_paper.yaml` |

目标类别为 ImageNet 标签 `954`，类别名为 `banana`。真实攻击评测严格按照
原 BadCLIP 实现：先把图像缩放到 `224x224`，再在中央放置 `16x16`
的 `badCLIP.jpg`，最后执行 CLIP 归一化。

## 2. 第一次复现为什么失败

第一次运行得到：

```text
防御前：CA=58.712%，ASR=82.028%
防御后：CA=58.646%，ASR=80.362%
```

该结果不是简单的评测口径差异，而是反演阶段实际上失败了。具体原因如下。

### 2.1 反演对齐损失实现错误

第一次实现使用“中毒图像特征与目标类图像特征的余弦距离”作为主要对齐项。
论文和作者原代码使用的是 1,000 个 ImageNet 文本类别上的对比交叉熵：

```text
CE((image_feature @ all_text_features) / temperature, target_label)
```

其中 `temperature=0.07`。错误实现缺少所有非目标文本类别形成的负样本，
对 banana 的优化信号太弱。第一次运行中 InfoNCE 从约 `0.535` 到 `0.538`
几乎没有改善。

### 2.2 反演迭代次数错误

第一次配置遍历完整 781 个 batch，并训练 10 个 epoch，共执行：

```text
781 x 10 = 7,810 次反演更新
```

论文补充材料说明反演通常在数百次迭代内收敛，作者公开配置中的
`20 batches x 10 epochs` 对应 200 次更新。7,810 次更新使
`lambda4=0.01` 的 L1 稀疏项长期占据主导，最终把掩码压缩到近似全零。

第一次反演结果的直接证据：

```text
反演触发器在 1,024 张图像上的 ASR：0%
mask 元素总数：150,528
mask L1：约 20
mask > 0.1 的元素：2
```

因此 Activation Tuning 实际比较的是“干净图像”和“几乎没有变化的图像”，
没有暴露真实后门路径。

### 2.3 Activation Tuning 的干净批次没有类别多样性

ImageNet Parquet 数据按标签近似有序。第一次实现直接取 DataLoader 的第一个
batch，64 张图几乎都来自开头类别，而不是论文所述的任意、有代表性的
64 张干净图。修正版从完整 50,000 张图中固定随机种子抽取 64 张跨类别样本。

### 2.4 神经元和参数掩码过窄

第一次实现先对空间维度求平均，只得到通道级差异，并且仅更新第一维恰好等于
关键层输出通道数的参数。失败运行只识别到 24 个通道和 12 个参数张量。

修正版先对 `[C,H,W]` 空间神经元差异做 KMeans，再将空间神经元映射到通道和
关键层内所有匹配的参数张量。本次修复运行识别到：

```text
关键层：visual.layer4
关键通道：211
受梯度掩码约束的参数张量：30
```

## 3. 修复后的实测结果

修正版反演只执行 200 次更新，随机 512 张诊断图像上的反演 ASR 为：

```text
epoch 1：0.39%
epoch 4：75.20%
epoch 8：87.11%
epoch 10：88.67%
```

论文报告的 BadCLIP 反演触发器 ASR 为 89.72%，两者基本一致。修复后的掩码
L1 约为 454，不再坍缩到 20。

Activation Tuning 后，完整 50,000 张 ImageNet 验证集结果为：

```text
CA：57.020%
CA Top-5：84.918%
ASR：0.052%
排除真实 banana 样本后的 ASR：0.016%
```

论文 banana-BadCLIP 表格结果是 `CA=57.01%`、`ASR=1.14%`。本机 CA
几乎完全一致，ASR 更低。需要注意，本机中毒模型的防御前 ASR 为 82.028%，
而论文 banana 模型为 98.16%，两者并非完全相同的攻击检查点，因此不应要求
每一个 ASR 小数位与论文一致。

## 4. 环境准备

```bash
conda activate aaai
cd /root/autodl-tmp/experiments/invertune
```

确认使用的是 OpenAI CLIP：

```bash
python -c "import clip, torch, pyarrow, sklearn; print(clip.__file__); print(torch.cuda.is_available())"
```

## 5. 推荐的分阶段复现命令

### 5.1 防御前基线

```bash
bash run_badclip_banana_paper.sh baseline
```

该阶段加载原始中毒模型，在 50,000 张干净图像上计算 CA，并使用原始
`badCLIP.jpg` 计算 ASR。

### 5.2 触发器反演

```bash
bash run_badclip_banana_paper.sh inversion
```

该阶段运行 `BadCLIPTriggerInversionPaper.py`，执行 10 个 epoch、每个 epoch
20 个 batch，总计 200 次 Adam 更新。

必须检查日志末尾是否出现类似输出：

```text
accepted inverted trigger: ASR=88.67%
```

配置要求反演 ASR 至少达到 70%。如果低于 70%，程序会报错并停止，不允许用
无效触发器继续微调。论文参考值为 89.72%，实际复现建议达到 80% 以上。

### 5.3 Activation Tuning

```bash
bash run_badclip_banana_paper.sh tuning
```

该阶段运行 `BadCLIPActivationTuningPaper.py`：

1. 随机抽取 64 张非 banana 干净图像。
2. 比较干净图像和反演触发图像在 RN50 四个 stage 的激活。
3. 使用 `mean + std` 选择关键层。
4. 对关键层 `[C,H,W]` 激活差异做二类 KMeans。
5. 只允许关键神经元对应参数接收梯度。
6. 使用学习率 `8e-6`、`beta=0.5` 微调 200 个 epoch。

本次运行中，反演触发器 ASR 在第 30 个 epoch 降到 7.81%，从第 40 个
epoch 起降到 0%。

### 5.4 防御后完整评测

```bash
bash run_badclip_banana_paper.sh evaluate
```

该阶段使用原始 BadCLIP 补丁而不是反演触发器进行最终评测。最终论文指标
必须从该阶段读取。

### 5.5 一键执行全部阶段

```bash
bash run_badclip_banana_paper.sh all
```

执行顺序为：基线评测、触发器反演、Activation Tuning、防御后评测。

## 6. 结果文件路径

修正版结果根目录：

```text
/root/autodl-tmp/experiments/invertune/results/badclip_banana_paper
```

### 6.1 反演结果

```text
inversion/epoch_1.pth ... inversion/epoch_10.pth
```

每个反演 epoch 的 `mask`、`trigger`、损失和诊断 ASR。

```text
inversion/best.pth
inversion/latest.pth
```

`best.pth` 是诊断 ASR 最高的反演结果；`latest.pth` 被设置为同一最佳结果，
也是 Activation Tuning 默认读取的文件。

```text
inversion/history.json
```

包含每轮总损失、交叉熵、embedding 损失、SSIM、mask L1、有效 mask 数量和
反演 ASR。

### 6.2 Activation Tuning 结果

```text
critical_neurons.json
```

保存候选层差异、选择阈值、关键层、空间关键神经元、关键通道和实际更新的
参数名称。

```text
tuning_history.json
```

保存 200 个 epoch 的激活对齐损失、特征保持损失和反演 ASR。

```text
checkpoints/epoch_50.pt
checkpoints/epoch_100.pt
checkpoints/epoch_150.pt
checkpoints/defended_model.pt
```

`defended_model.pt` 是最终防御模型。

### 6.3 CA/ASR 和日志

```text
evaluation/baseline_metrics.json
evaluation/defended_metrics.json
```

最终防御结果位于：

```text
/root/autodl-tmp/experiments/invertune/results/badclip_banana_paper/evaluation/defended_metrics.json
```

所有阶段日志位于：

```text
logs/baseline_evaluation.log
logs/trigger_inversion.log
logs/activation_tuning.log
logs/defended_evaluation.log
```

## 7. 复现时的必要判断标准

不要只观察训练损失下降，应依次检查：

1. 防御前 ASR 是否明显高于正常误分类率。
2. 反演触发器 ASR 是否达到至少 70%，建议达到 80% 以上。
3. mask 是否没有坍缩为近似全零。
4. Activation Tuning 过程中反演 ASR 是否持续下降。
5. 最终必须使用原始 `badCLIP.jpg`，而不是反演触发器计算 ASR。
6. 防御后 ASR 应显著下降，同时 CA 下降应有限。

## 8. 数据划分限制

论文使用独立的 50K ImageNet 训练子集进行 InverTune，再使用 ImageNet
验证集评测。本机目前提供的 `/root/autodl-tmp/datasets/imagenet1k` 实际是
50K 验证集，因此防御和评测复用了同一批图像。该设置足以验证代码和防御机制，
但若要形成严格的论文对等实验，应额外准备独立的 50K ImageNet 训练子集，
并保持现有 50K 验证集只用于最终 CA/ASR。

## 9. 修正版项目文件构成

清理后的目录只保留修正版论文复现所需代码、配置和有效结果：

```text
invertune/
├── BadCLIPEvaluate.py
├── BadCLIPTriggerInversionPaper.py
├── BadCLIPActivationTuningPaper.py
├── run_badclip_banana_paper.sh
├── README_BADCLIP_BANANA_PAPER.md
├── config/
│   └── badclip_banana_paper.yaml
├── data/
│   ├── __init__.py
│   ├── classes.py
│   └── imagenet.py
├── models/
│   ├── invertune_badclip.py
│   └── paper_inversion.py
├── utils/
│   ├── __init__.py
│   ├── clip_model.py
│   └── repro.py
└── results/
    └── badclip_banana_paper/
```

各文件职责如下。

| 文件 | 作用 |
|---|---|
| `BadCLIPEvaluate.py` | 在 ImageNet 上计算干净 CA、Top-5 CA、真实 BadCLIP ASR 和排除目标类后的 ASR |
| `BadCLIPTriggerInversionPaper.py` | 使用 1,000 类文本对比交叉熵执行 200 次触发器反演，并用诊断 ASR 筛选最佳触发器 |
| `BadCLIPActivationTuningPaper.py` | 选择关键层和关键神经元，应用参数梯度掩码并微调 200 个 epoch |
| `run_badclip_banana_paper.sh` | 按 `baseline → inversion → tuning → evaluate` 顺序组织实验 |
| `config/badclip_banana_paper.yaml` | 集中管理模型、数据、目标类、真实补丁、超参数和输出目录 |
| `data/classes.py` | ImageNet 1,000 类名称和 CLIP prompt templates |
| `data/imagenet.py` | 读取 ImageNet Parquet、构造目标/非目标数据和应用真实 BadCLIP 补丁 |
| `models/paper_inversion.py` | 论文反演损失、mask/pattern 参数化和反演 ASR 计算 |
| `models/invertune_badclip.py` | CLIP 归一化、反归一化以及反演触发器叠加函数 |
| `utils/clip_model.py` | 加载本地 OpenAI CLIP 和中毒/防御 checkpoint，构建零样本文本分类器 |
| `utils/repro.py` | YAML、随机种子、JSON 和 checkpoint 辅助函数 |
| `results/badclip_banana_paper/` | 当前有效反演结果、防御模型、日志和 CA/ASR |

其中 `.git/` 是版本控制元数据，不参与实验运行，但保留它便于查看和管理代码
变更。`results/badclip_banana_paper/` 是本次已验证成功的实验结果，不属于待删除
冗余文件。

## 10. 更换输入中毒模型

一键脚本现在会自动从 YAML 的 `model.model_path` 读取中毒模型，不再在 shell
脚本中硬编码 checkpoint 路径。更换同为 RN50、同为 banana 目标的中毒模型时，
只需修改：

```yaml
model:
  clean_model_path: /root/autodl-tmp/checkpoints/clip-clean-pretrained/RN50.pt
  model_path: /绝对路径/新的中毒模型.pt
  clip_type: RN50
  device: auto
```

支持以下两种 checkpoint 格式：

```python
# 格式一：直接 state_dict
{"visual.conv1.weight": tensor, ...}

# 格式二：训练 checkpoint
{"state_dict": {"visual.conv1.weight": tensor, ...}, "epoch": 10, ...}
```

参数名称带 `module.` 前缀时，加载器会自动移除该前缀。模型架构必须与
`clean_model_path` 对应；当前关键层配置针对 RN50。若输入模型是 RN101 或
ViT，需要同时提供同架构的干净 CLIP 权重，并调整 `tuning.candidate_layers`。

### 10.1 更换模型后的推荐配置

为避免覆盖当前成功结果，建议同时给新实验设置独立输出目录。例如：

```yaml
model:
  model_path: /绝对路径/banana_badclip_new.pt

inversion:
  trigger_path: /root/autodl-tmp/experiments/invertune/results/banana_badclip_new/inversion/latest.pth

evaluation:
  checkpoint: /root/autodl-tmp/experiments/invertune/results/banana_badclip_new/checkpoints/defended_model.pt
  output: /root/autodl-tmp/experiments/invertune/results/banana_badclip_new/evaluation/defended_metrics.json

output:
  root: /root/autodl-tmp/experiments/invertune/results/banana_badclip_new
```

`output.root`、`inversion.trigger_path`、`evaluation.checkpoint` 和
`evaluation.output` 必须指向同一个实验目录体系。

### 10.2 目标类或真实补丁发生变化时

如果新模型不是 banana 目标，还必须同步修改：

```yaml
data:
  target_class_id: 对应的_ImageNet_WNID
  target_word: 新目标类别英文名
  target_label: 新目标在 ImageNet_1K 中的整数标签

attack:
  patch_path: /新攻击对应的真实补丁路径
  patch_size: 16
  patch_location: middle
```

`attack` 部分只用于最终真实攻击评测，不参与 InverTune 反演和微调。若真实
补丁未知，仍可执行反演和防御，但无法计算具有攻击语义的最终真实 ASR。

### 10.3 更换模型后的运行顺序

先只运行基线：

```bash
conda activate aaai
cd /root/autodl-tmp/experiments/invertune
bash run_badclip_banana_paper.sh baseline
```

确认新 checkpoint 能正确加载，并且基线 CA/ASR 合理后，再运行：

```bash
bash run_badclip_banana_paper.sh inversion
```

必须确认反演 ASR 通过 70% 硬门槛，建议达到 80% 以上。随后执行：

```bash
bash run_badclip_banana_paper.sh tuning
bash run_badclip_banana_paper.sh evaluate
```

也可以在配置确认无误后执行完整流程：

```bash
bash run_badclip_banana_paper.sh all
```

不要直接复用其他中毒模型生成的 `inversion/latest.pth` 或
`checkpoints/defended_model.pt`。每个中毒 checkpoint 都必须重新执行
`inversion` 和 `tuning`。

### 10.4 更换前检查 checkpoint

可以先查看 checkpoint 结构：

```bash
python -c "import torch; p='/绝对路径/新的中毒模型.pt'; x=torch.load(p,map_location='cpu'); print(type(x)); print(x.keys() if isinstance(x,dict) else 'not dict')"
```

如果 `baseline` 报告 missing/unexpected keys，说明 checkpoint 架构或
state_dict 命名与当前 RN50 不兼容，不能通过关闭严格检查来忽略；应使用匹配
架构的 OpenAI CLIP 基模型和候选层配置。
