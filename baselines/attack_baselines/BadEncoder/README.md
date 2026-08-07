# InverTune 中的 BadEncoder 基线复现

本目录用于复现论文《InverTune: Removing Backdoors from Multimodal Contrastive Learning Models via Trigger Inversion and Activation Tuning》中的 CLIP RN50 BadEncoder 基线。

## 实验配置

- 干净模型：OpenAI CLIP RN50
- 影子数据集：CC3M 500K
- 攻击目标：ImageNet 第 954 类 `banana`
- 触发器：BadEncoder 原始右下角位置的 16×16 纯白补丁
- 优化器：SGD
- 学习率：`1e-6`
- batch size：128
- 训练轮数：10 epoch
- 下游评估：ImageNet-1K 零样本分类
- 文本分类器：ImageNet 1,000 类及 InverTune 使用的 80 个 CLIP prompt templates

未防御 BadEncoder 攻击结果为：

- 干净准确率：`CA=58.88%`
- 攻击成功率：`ASR=97.91%`

## 本机数据和权重

```text
OpenAI CLIP RN50:
  /root/autodl-tmp/checkpoints/clip-clean-pretrained/RN50.pt

CC3M 500K:
  /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/train.csv
  /root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/images/

ImageNet-1K 验证集:
  /root/autodl-tmp/datasets/imagenet1k_badclip/validation/

ImageNet 类名和 prompt templates:
  /root/autodl-tmp/experiments/invertune/data/classes.py
```

训练阶段直接流式读取 CC3M JPEG，不需要生成体积巨大的 `train_224.npz`。

## 主要文件

- `badencoder.py`：BadEncoder 视觉编码器后门注入训练。
- `datasets/clip_paper_dataset.py`：CC3M 500K 影子数据集加载和触发器叠加。
- `zero_shot_imagenet.py`：ImageNet-1K 零样本 CA、CA@5、ASR 评估。
- `scripts/run_invertune_badencoder.py`：论文配置的一键训练和评估入口。
- `trigger/trigger_pt_white_173_50_ap_replace.npz`：224×224 输入对应的 16×16 纯白触发器。

训练生成的 checkpoint 包含完整 CLIP 状态，包括未修改的文本编码器，因此可以被 `/root/autodl-tmp/experiments/invertune` 中的模型加载器直接读取。

## 训练进度、日志和 checkpoint

训练默认启用 AMP，并按损失项分阶段反向传播，以降低 RN50 在 batch size 128 下的显存峰值。终端每秒刷新一次进度条，显示当前 epoch、batch、总损失、三个分项损失和学习率。

运行脚本会把同一份输出实时显示在终端并追加写入：

```text
log/invertune_badencoder_banana.log
```

不需要再使用 `2>&1 | tee ...`。每完成一个 epoch 会原子保存：

```text
output/CLIP/invertune_badencoder_banana/model_1.pth
output/CLIP/invertune_badencoder_banana/model_2.pth
...
output/CLIP/invertune_badencoder_banana/model_10.pth
output/CLIP/invertune_badencoder_banana/latest.pth
```

`latest.pth` 指向最近一个完整 checkpoint，包含完整 CLIP 参数、优化器、AMP scaler 和随机状态。

## 冒烟测试

冒烟测试只验证数据、训练、checkpoint 和评估链路，其指标不能与论文结果比较。

```bash
cd /root/autodl-tmp/experiments/badEncoder

python -u scripts/run_invertune_badencoder.py \
  --stage all \
  --epochs 1 \
  --batch_size 128 \
  --eval_batch_size 2 \
  --num_workers 0 \
  --max_shadow_samples 128 \
  --max_eval_samples 4 \
  --output_dir /tmp/badencoder_invertune_smoke \
  --log_file /tmp/badencoder_invertune_smoke.log
```

## 正式训练和评估

终端已经显示 `(aaai)` 时直接运行：

```bash
cd /root/autodl-tmp/experiments/badEncoder

python -u scripts/run_invertune_badencoder.py \
  --stage all \
  --gpu 0 \
  --batch_size 128
```

如果尚未激活环境，必须加入 `--no-capture-output`：

```bash
conda run --no-capture-output -n aaai \
  python -u scripts/run_invertune_badencoder.py \
  --stage all \
  --gpu 0 \
  --batch_size 128
```

该命令依次完成 CC3M 500K 上的 10 epoch 后门注入和 ImageNet-1K 全量零样本评估。

## 中断后续训

从最近一个完整 epoch 继续训练到第 10 轮：

```bash
python -u scripts/run_invertune_badencoder.py \
  --stage train \
  --epochs 10 \
  --gpu 0 \
  --resume output/CLIP/invertune_badencoder_banana/latest.pth
```

如果在某一 epoch 中途停止，该 epoch 会从头重跑；已经完成的 epoch 不会重跑。续训结束后单独评估：

```bash
python -u scripts/run_invertune_badencoder.py \
  --stage eval \
  --epochs 10 \
  --gpu 0
```

## 输出文件

```text
output/CLIP/invertune_badencoder_banana/model_1.pth ... model_10.pth
output/CLIP/invertune_badencoder_banana/latest.pth
output/CLIP/invertune_badencoder_banana/imagenet_zero_shot_metrics.json
log/invertune_badencoder_banana.log
```

评估 JSON 包含：

- `CA`：ImageNet-1K Top-1 干净准确率。
- `CA_top5`：ImageNet-1K Top-5 干净准确率。
- `ASR`：所有触发样本被预测为 `banana` 的比例。
- `ASR_non_target`：排除真实 `banana` 样本后的攻击成功率。

## 复现说明

InverTune 论文指定了 BadEncoder 方法和目标类别，但没有公开具体使用的目标参考图。本实现固定使用 ImageNet 验证集中标签为 `banana` 的图像：


