#!/usr/bin/env python3
"""下游微调 + ASR/CA 轨迹记录（Week1 核心）。

两种下游目标（objective）：

- ``contrastive``（默认，推荐）：用干净图文对（CC3M）继续做 **CLIP 原生对比微调**。
  这保持 CLIP 的图文对齐空间，与 zero-shot ASR/CA 评估口径一致，是论文中
  “用户拿清洗后的 CLIP 继续正常微调导致后门 rebound” 的真实场景。

- ``supervised``：在 image encoder 上加线性头做 CE 分类微调（cifar10 / imagenet_local）。
  **注意**：full-FT + 监督 CE 会把整个图文嵌入空间朝分类任务重塑，破坏 CLIP 的
  zero-shot 通道，导致 CA / ASR 双双坍塌——此时 rebound 会被“评估通道失效”掩盖。
  仅作为对照/兼容保留，不建议用它判定 rebound。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from clip_eval import eval_asr_ca, load_clip_model
from config import BADCLIP_ROOT, LOG_ROOT, RESULT_ROOT


IMAGENET_ROOT_DEFAULT = "/root/autodl-tmp/datasets/imagenet1k_badclip/validation"
IMAGENET_LABELS_DEFAULT = os.path.join(IMAGENET_ROOT_DEFAULT, "labels.csv")

CC3M_ROOT_DEFAULT = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K"
# 该 csv 已剔除 banana 相关样本，避免下游数据把 target 概念重新引入。
CC3M_CSV_DEFAULT = os.path.join(CC3M_ROOT_DEFAULT, "cc3m_natural_10K_WObanana.csv")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clip_train_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# 监督分类数据（对照用）
# --------------------------------------------------------------------------- #
def get_cifar10_loader(batch_size: int, num_workers: int = 4):
    root = os.path.join(RESULT_ROOT, "cifar10_cache")
    ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=_clip_train_transform()
    )
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        10,
    )


class CsvImageNetDataset(Dataset):
    def __init__(self, root: str, labels_csv: str, transform):
        self.root = root
        self.transform = transform
        self.rows = []
        with open(labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.join(root, row["image"])
                if os.path.exists(path):
                    self.rows.append((path, int(row["label"])))
        if not self.rows:
            raise RuntimeError(f"No ImageNet rows found from {labels_csv} under {root}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def get_imagenet_local_loader(
    batch_size: int,
    root: str = IMAGENET_ROOT_DEFAULT,
    labels_csv: str = IMAGENET_LABELS_DEFAULT,
    num_workers: int = 4,
):
    ds = CsvImageNetDataset(root=root, labels_csv=labels_csv, transform=_clip_train_transform())
    return (
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        1000,
    )


# --------------------------------------------------------------------------- #
# CLIP 对比微调数据（图文对，推荐）
# --------------------------------------------------------------------------- #
class Cc3mPairDataset(Dataset):
    """CC3M 干净图文对：返回 (image_tensor, caption_str)。"""

    def __init__(self, root: str, csv_path: str, transform, image_key="image", caption_key="caption"):
        self.root = root
        self.transform = transform
        self.rows = []
        missing = 0
        examples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get(image_key)
                cap = row.get(caption_key)
                if not img or cap is None:
                    continue
                path = os.path.join(root, img)
                if not os.path.exists(path):
                    missing += 1
                    if len(examples) < 5:
                        examples.append(path)
                self.rows.append((path, str(cap)))
        if not self.rows:
            raise RuntimeError(f"No CC3M pairs found from {csv_path} under {root}")
        if missing:
            ratio = missing / len(self.rows)
            msg = (
                f"Missing {missing}/{len(self.rows)} CC3M images under root={root} "
                f"for csv={csv_path}. Examples: {examples}"
            )
            if ratio > 0.01:
                raise RuntimeError(msg)
            print(f"[warn] {msg}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, caption = self.rows[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # 损坏样本回退到全零图，避免整个训练崩溃
            image = Image.new("RGB", (224, 224))
        return self.transform(image), caption


def _collate_pairs(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return images, captions


def get_cc3m_loader(
    batch_size: int,
    root: str = CC3M_ROOT_DEFAULT,
    csv_path: str = CC3M_CSV_DEFAULT,
    num_workers: int = 4,
):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CC3M csv not found: {csv_path}. "
            "Refusing to fall back to train.csv because rebound results depend on the exact clean CSV."
        )
    ds = Cc3mPairDataset(root=root, csv_path=csv_path, transform=_clip_train_transform())
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate_pairs,
        drop_last=True,
    )
    return loader, csv_path


# --------------------------------------------------------------------------- #
# 可训练参数配置
# --------------------------------------------------------------------------- #
def _set_requires_grad_by_ft(model, ft: str):
    """按 ft 模式设置 backbone 的可训练性，返回 ft_mode 描述。"""
    if ft == "full":
        for p in model.parameters():
            p.requires_grad = True
        return "full"

    if ft == "linear":
        # 对比目标下 “linear” = 只调最终投影层（保持 CLIP 结构）
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if any(k in name for k in ("text_projection", "visual.attnpool.c_proj")):
                p.requires_grad = True
        return "linear_proj"

    if ft == "lora":
        # 半解冻高层（对 RN50 命名友好的 fallback）
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if any(k in name for k in ("visual.layer4", "visual.attnpool", "text_projection", "ln_final")):
                p.requires_grad = True
        return "lora_partial"

    raise ValueError(ft)


def setup_supervised(model, ft: str, device: str, num_classes: int):
    """监督分类：backbone + 线性头。返回 (head, params, ft_mode)。"""
    ft_mode = _set_requires_grad_by_ft(model, ft)
    if ft == "linear":
        # 监督 linear-probe 应冻结整个 backbone，只训头
        for p in model.parameters():
            p.requires_grad = False
        ft_mode = "linear_head"
    dim = model.text_projection.shape[1]
    head = nn.Linear(dim, num_classes).to(device)
    trainable = list(filter(lambda x: x.requires_grad, model.parameters())) + list(head.parameters())
    return head, trainable, ft_mode


def setup_contrastive(model, ft: str):
    """对比微调：无分类头。返回 (params, ft_mode)。"""
    ft_mode = _set_requires_grad_by_ft(model, ft)
    trainable = list(filter(lambda x: x.requires_grad, model.parameters()))
    return trainable, ft_mode


def save_model_ckpt(model, path: str):
    torch.save({"state_dict": model.state_dict(), "epoch": 0}, path)


def clip_contrastive_loss(model, images, input_ids, attention_mask, criterion):
    image_feats = model.get_image_features(images)
    text_feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_feats @ text_feats.t()
    logits_per_text = logits_per_image.t()
    targets = torch.arange(images.size(0), device=images.device)
    return (criterion(logits_per_image, targets) + criterion(logits_per_text, targets)) / 2.0


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_downstream(
    ckpt: str,
    ft: str = "full",
    objective: str = "contrastive",
    steps: int = 500,
    eval_every: int = 50,
    eval_steps: str = "",
    batch_size: int = 128,
    lr: float = 1e-5,
    device: str = "cuda:0",
    seed: int = 42,
    tag: str = "run",
    subset: int = 5000,
    downstream: str = "cc3m",
    imagenet_root: str = IMAGENET_ROOT_DEFAULT,
    imagenet_labels: str = IMAGENET_LABELS_DEFAULT,
    cc3m_root: str = CC3M_ROOT_DEFAULT,
    cc3m_csv: str = CC3M_CSV_DEFAULT,
    revival_threshold: float = 0.5,
):
    set_seed(seed)
    os.makedirs(RESULT_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)

    traj_path = os.path.join(RESULT_ROOT, f"traj_{tag}_{objective}_{ft}_s{seed}.csv")
    log_path = os.path.join(LOG_ROOT, f"downstream_{tag}_{objective}_{ft}_s{seed}.log")

    # ---- 数据 ----
    data_meta = {}
    if objective == "contrastive":
        if downstream != "cc3m":
            raise ValueError("contrastive objective 仅支持 downstream=cc3m（图文对）")
        loader, used_csv = get_cc3m_loader(batch_size, root=cc3m_root, csv_path=cc3m_csv)
        num_classes = 0
        data_meta["cc3m_csv"] = used_csv
    else:  # supervised
        if downstream == "cifar10":
            loader, num_classes = get_cifar10_loader(batch_size)
        elif downstream == "imagenet_local":
            loader, num_classes = get_imagenet_local_loader(
                batch_size, root=imagenet_root, labels_csv=imagenet_labels
            )
        else:
            raise ValueError(f"supervised objective 不支持 downstream={downstream}")
    it = iter(loader)

    # ---- 模型 ----
    model, processor = load_clip_model(ckpt, device)
    # CLIP 权重默认 fp16；训练需转 fp32 以保证 AdamW 数值稳定。
    model = model.float().to(device)

    head = None
    if objective == "contrastive":
        params, ft_mode = setup_contrastive(model, ft)
    else:
        head, params, ft_mode = setup_supervised(model, ft, device, num_classes)

    optimizer = optim.AdamW(params, lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    eval_subset = subset if subset > 0 else None
    scheduled_eval_steps = None
    if eval_steps:
        scheduled_eval_steps = {
            int(x.strip()) for x in eval_steps.split(",") if x.strip()
        }
        scheduled_eval_steps = {
            step for step in scheduled_eval_steps if 0 < step <= steps
        }
        scheduled_eval_steps.add(steps)

    def evaluate_current(step_label: int, train_loss: float = 0.0):
        # 直接评估内存中的模型，避免每次 eval 把 ~1.2GB 权重落盘再读回
        # （既慢又脆弱：RAID 偶发读失败会整条轨迹崩溃）。
        was_training = model.training
        model.eval()
        mets = eval_asr_ca(None, device=device, subset=eval_subset, model=model, processor=processor)
        if was_training:
            model.train()
        return {"step": step_label, "train_loss": train_loss, **mets}

    rows = [evaluate_current(0)]
    with open(log_path, "w") as logf:
        logf.write(f"# ImmuneCLIP downstream | objective={objective} ft={ft} ckpt={ckpt}\n")
        logf.write(
            f"# steps={steps} eval_every={eval_every} "
            f"eval_steps={sorted(scheduled_eval_steps) if scheduled_eval_steps else None} "
            f"lr={lr} seed={seed} "
            f"downstream={downstream} num_classes={num_classes} meta={data_meta}\n"
        )
        logf.write(json.dumps(rows[0]) + "\n")
        logf.flush()

        model.train()
        pbar = tqdm(range(1, steps + 1), desc=f"{tag}/{objective}/{ft}")
        for step in pbar:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            optimizer.zero_grad()
            if objective == "contrastive":
                images, captions = batch
                images = images.to(device)
                tokens = processor.process_text(list(captions))
                input_ids = tokens["input_ids"].to(device)
                attn = tokens["attention_mask"].to(device)
                loss = clip_contrastive_loss(model, images, input_ids, attn, criterion)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                if ft == "linear":
                    with torch.no_grad():
                        feats = model.get_image_features(images)
                    logits = head(feats.detach())
                else:
                    feats = model.get_image_features(images)
                    logits = head(feats)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            should_eval = (
                step in scheduled_eval_steps
                if scheduled_eval_steps is not None
                else (step % eval_every == 0 or step == steps)
            )
            if should_eval:
                row = evaluate_current(step, loss.item())
                rows.append(row)
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                pbar.set_postfix(
                    asr=f"{row['asr_top1']:.3f}",
                    ca=f"{row['ca_top1']:.3f}",
                    rebound=f"{row['asr_top1'] - rows[0]['asr_top1']:.3f}",
                )

    with open(traj_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    asr_vals = [r["asr_top1"] for r in rows]
    ca_vals = [r["ca_top1"] for r in rows]
    if len(rows) > 1 and rows[-1]["step"] > rows[0]["step"]:
        area = 0.0
        for left, right in zip(rows[:-1], rows[1:]):
            width = right["step"] - left["step"]
            area += width * 0.5 * (left["asr_top1"] + right["asr_top1"])
        aurc_asr = area / (rows[-1]["step"] - rows[0]["step"])
    else:
        aurc_asr = rows[0]["asr_top1"]
    revival_step = None
    for r in rows:
        if r["asr_top1"] >= revival_threshold:
            revival_step = r["step"]
            break
    summary = {
        "tag": tag,
        "objective": objective,
        "ft": ft,
        "ft_mode": ft_mode,
        "downstream": downstream,
        "num_classes": num_classes,
        "ckpt": ckpt,
        "traj_csv": traj_path,
        "log_path": log_path,
        "asr_step0": rows[0]["asr_top1"],
        "asr_final": rows[-1]["asr_top1"],
        "asr_max": max(asr_vals),
        "rebound_delta": rows[-1]["asr_top1"] - rows[0]["asr_top1"],
        "rebound_max_delta": max(asr_vals) - rows[0]["asr_top1"],
        "aurc_asr": aurc_asr,
        "revival_threshold": revival_threshold,
        "revival_step": revival_step,
        "ca_step0": rows[0]["ca_top1"],
        "ca_final": rows[-1]["ca_top1"],
        "ca_min": min(ca_vals),
        "eval_steps": [r["step"] for r in rows],
        **data_meta,
    }
    summary_path = traj_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ft", choices=["linear", "lora", "full"], default="full")
    parser.add_argument(
        "--objective",
        choices=["contrastive", "supervised"],
        default="contrastive",
        help="contrastive=CLIP图文对比微调(推荐,保持zero-shot口径); supervised=线性头CE(对照,会破坏zero-shot空间)",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument(
        "--eval_steps",
        type=str,
        default="",
        help="Optional comma-separated evaluation steps. Step 0 is always evaluated.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--subset", type=int, default=5000)
    parser.add_argument(
        "--downstream",
        choices=["cc3m", "cifar10", "imagenet_local"],
        default="cc3m",
    )
    parser.add_argument("--imagenet_root", type=str, default=IMAGENET_ROOT_DEFAULT)
    parser.add_argument("--imagenet_labels", type=str, default=IMAGENET_LABELS_DEFAULT)
    parser.add_argument("--cc3m_root", type=str, default=CC3M_ROOT_DEFAULT)
    parser.add_argument("--cc3m_csv", type=str, default=CC3M_CSV_DEFAULT)
    parser.add_argument(
        "--revival_threshold",
        type=float,
        default=0.5,
        help="First step with ASR@1 >= threshold; null if never reached.",
    )
    args = parser.parse_args()
    run_downstream(**vars(args))


if __name__ == "__main__":
    main()
