#!/usr/bin/env python3
"""Target-agnostic Stage0 inversion for ImmuneCLIP.

This script estimates both a universal proxy trigger and a target class from the
defended model itself. It never reads the BadCLIP patch or a known target label.
The output checkpoint is compatible with immuneclip_train.py proxy mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from clip_eval import load_clip_model  # noqa: E402


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CC3M_ROOT_DEFAULT = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K"
CC3M_CSV_DEFAULT = (
    "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/"
    "cc3m_natural_10K_no_banana_strict.csv"
)
IMAGENET_CLASSES_DEFAULT = "/root/autodl-tmp/datasets/imagenet1k_badclip/validation/classes.py"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class CleanImageDataset(Dataset):
    def __init__(self, root: str, csv_path: str, limit_rows: int = 0):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        self.root = root
        self.transform = train_transform()
        self.paths: List[str] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get("image")
                if not img:
                    continue
                path = img if os.path.isabs(img) else os.path.join(root, img)
                if os.path.exists(path):
                    self.paths.append(path)
                if limit_rows > 0 and len(self.paths) >= limit_rows:
                    break
        if not self.paths:
            raise RuntimeError(f"No usable clean images from {csv_path}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


def load_classes(classes_path: str):
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(classes_path)
    config = eval(open(classes_path, "r").read())
    return config["classes"], config["templates"]


@torch.no_grad()
def build_text_classifier(model, processor, classes: Sequence[str], templates, device: str) -> torch.Tensor:
    embeddings = []
    for c in tqdm(classes, desc="text_embed", leave=False):
        texts = [template(c) for template in templates]
        tokens = processor.process_text(texts)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        text_embedding = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = F.normalize(text_embedding, dim=-1).mean(dim=0)
        embeddings.append(F.normalize(text_embedding, dim=0))
    return torch.stack(embeddings, dim=1).to(device)


def tv_loss(mask: torch.Tensor) -> torch.Tensor:
    return (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean() + (
        mask[:, :, :, 1:] - mask[:, :, :, :-1]
    ).abs().mean()


def normalized_from_pixel(pixel: torch.Tensor) -> torch.Tensor:
    mean = pixel.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = pixel.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (pixel - mean) / std


def apply_proxy(images: torch.Tensor, mask: torch.Tensor, trigger_norm: torch.Tensor) -> torch.Tensor:
    return (1 - mask) * images + mask * trigger_norm


def fixed_patch_mask(
    patch_size: int,
    device: str,
    dtype: torch.dtype,
    location: str = "middle",
) -> torch.Tensor:
    mask = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
    if location == "middle":
        top = left = 112 - patch_size // 2
    elif location == "bottom_right":
        top = left = 224 - patch_size
    else:
        raise ValueError(f"Unsupported patch location: {location}")
    mask[:, :, top : top + patch_size, left : left + patch_size] = 1.0
    return mask


def apply_fixed_patch(
    images: torch.Tensor,
    patches_norm: torch.Tensor,
    patch_size: int,
    location: str,
) -> torch.Tensor:
    """Apply one normalized patch per image batch.

    images: [K, B, 3, 224, 224], patches_norm: [K, 3, P, P]
    returns: [K * B, 3, 224, 224]
    """
    k, b = images.shape[:2]
    out = images.clone()
    if location == "middle":
        top = left = 112 - patch_size // 2
    elif location == "bottom_right":
        top = left = 224 - patch_size
    else:
        raise ValueError(f"Unsupported patch location: {location}")
    out[:, :, :, top : top + patch_size, left : left + patch_size] = patches_norm[:, None]
    return out.reshape(k * b, 3, 224, 224)


@torch.no_grad()
def diagnose(model, loader, text_classifier, classes, mask, trigger_norm, device, max_batches: int):
    clean_sum = torch.zeros(text_classifier.shape[1], device=device)
    trig_sum = torch.zeros_like(clean_sum)
    pred_counts = torch.zeros_like(clean_sum)
    total = 0
    for i, images in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        clean = F.normalize(model.get_image_features(images), dim=-1)
        trig = F.normalize(model.get_image_features(apply_proxy(images, mask, trigger_norm)), dim=-1)
        clean_logits = clean @ text_classifier
        trig_logits = trig @ text_classifier
        clean_sum += clean_logits.sum(dim=0)
        trig_sum += trig_logits.sum(dim=0)
        pred_counts += trig_logits.argmax(dim=1).bincount(minlength=text_classifier.shape[1]).float()
        total += images.size(0)
    mean_clean = clean_sum / max(total, 1)
    mean_trig = trig_sum / max(total, 1)
    delta = mean_trig - mean_clean
    top = torch.topk(delta, k=min(10, delta.numel()))
    pred_top = torch.topk(pred_counts / max(total, 1), k=min(10, delta.numel()))
    topk = [
        {
            "rank": int(r + 1),
            "class_index": int(idx),
            "class_name": classes[int(idx)],
            "delta_logit": float(val),
            "trigger_logit": float(mean_trig[int(idx)]),
            "clean_logit": float(mean_clean[int(idx)]),
            "trigger_pred_freq": float(pred_counts[int(idx)] / max(total, 1)),
        }
        for r, (val, idx) in enumerate(zip(top.values, top.indices))
    ]
    pred_topk = [
        {
            "rank": int(r + 1),
            "class_index": int(idx),
            "class_name": classes[int(idx)],
            "trigger_pred_freq": float(val),
            "delta_logit": float(delta[int(idx)]),
        }
        for r, (val, idx) in enumerate(zip(pred_top.values, pred_top.indices))
    ]
    target_idx = int(top.indices[0])
    return {
        "num_diagnostic_images": int(total),
        "target_index": target_idx,
        "target_name": classes[target_idx],
        "target_delta_logit": float(top.values[0]),
        "topk_delta": topk,
        "topk_trigger_predictions": pred_topk,
    }


@torch.no_grad()
def diagnose_target(
    model,
    loader,
    text_classifier,
    target_idx: int,
    mask: torch.Tensor,
    trigger_norm: torch.Tensor,
    device: str,
    max_batches: int,
) -> Dict[str, float]:
    total = 0
    correct = 0
    ce_sum = 0.0
    logit_sum = 0.0
    labels = None
    for i, images in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        logits = F.normalize(
            model.get_image_features(apply_proxy(images, mask, trigger_norm)),
            dim=-1,
        ) @ text_classifier
        if labels is None or labels.numel() != images.size(0):
            labels = torch.full((images.size(0),), target_idx, dtype=torch.long, device=device)
        ce_sum += float(F.cross_entropy(logits / 0.07, labels[: images.size(0)], reduction="sum").cpu())
        correct += int(logits.argmax(dim=1).eq(target_idx).sum().cpu())
        logit_sum += float(logits[:, target_idx].sum().cpu())
        total += images.size(0)
    return {
        "num_diagnostic_images": int(total),
        "target_ce": ce_sum / max(total, 1),
        "target_asr": correct / max(total, 1),
        "target_logit": logit_sum / max(total, 1),
    }


def parse_candidates(value: str, num_classes: int) -> List[int]:
    if value == "all":
        return list(range(num_classes))
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= num_classes:
            raise ValueError(f"Candidate target index out of range: {idx}")
        result.append(idx)
    if not result:
        raise ValueError("No scan candidates were provided")
    return result


def scan_target_candidates(
    model,
    loader,
    text_classifier: torch.Tensor,
    classes: Sequence[str],
    args,
    log_path: str,
) -> Tuple[int, List[dict]]:
    """Black-box target scan using only model outputs and clean images.

    The scan optimizes a small proxy patch for each candidate target. It does not
    use the real trigger or the real target label. Candidates that reach high
    target ASR with lower CE and small patch energy are treated as suspicious.
    """
    candidates = parse_candidates(args.scan_candidates, text_classifier.shape[1])
    device = args.device
    mean = text_classifier.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = text_classifier.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    data_iter = iter(loader)
    all_records: List[dict] = []
    chunk_bar = tqdm(
        range(0, len(candidates), args.scan_chunk_size),
        desc="stage0_scan",
    )
    for start_idx in chunk_bar:
        chunk = candidates[start_idx : start_idx + args.scan_chunk_size]
        k = len(chunk)
        patch_logits = torch.nn.Parameter(
            torch.randn((k, 3, args.scan_patch_size, args.scan_patch_size), device=device) * 0.01
        )
        optimizer = torch.optim.Adam([patch_logits], lr=args.scan_lr)
        target_tensor = torch.tensor(chunk, dtype=torch.long, device=device)
        last_ce = None
        last_asr = None
        for _step in range(1, args.scan_steps + 1):
            try:
                images = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images = next(data_iter)
            images = images.to(device)
            if images.size(0) > args.scan_batch_size:
                images = images[: args.scan_batch_size]
            optimizer.zero_grad(set_to_none=True)
            patches_norm = (torch.sigmoid(patch_logits) - mean) / std
            expanded = images.unsqueeze(0).expand(k, -1, -1, -1, -1)
            poisoned = apply_fixed_patch(expanded, patches_norm, args.scan_patch_size, args.scan_location)
            logits = F.normalize(model.get_image_features(poisoned), dim=-1) @ text_classifier
            logits = logits.reshape(k, images.size(0), -1)
            labels = target_tensor[:, None].expand(-1, images.size(0)).reshape(-1)
            ce_each_sample = F.cross_entropy(
                (logits.reshape(k * images.size(0), -1) / args.scan_temperature),
                labels,
                reduction="none",
            ).reshape(k, images.size(0))
            ce_each = ce_each_sample.mean(dim=1)
            patch_l2 = torch.sigmoid(patch_logits).flatten(1).pow(2).mean(dim=1)
            loss = (ce_each + args.scan_patch_l2_weight * patch_l2).mean()
            loss.backward()
            optimizer.step()
            last_ce = ce_each.detach()
            last_asr = logits.detach().argmax(dim=-1).eq(target_tensor[:, None]).float().mean(dim=1)
        with torch.no_grad():
            patches_norm = (torch.sigmoid(patch_logits) - mean) / std
            try:
                images = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                images = next(data_iter)
            images = images.to(device)
            if images.size(0) > args.scan_batch_size:
                images = images[: args.scan_batch_size]
            expanded = images.unsqueeze(0).expand(k, -1, -1, -1, -1)
            logits = F.normalize(
                model.get_image_features(
                    apply_fixed_patch(expanded, patches_norm, args.scan_patch_size, args.scan_location)
                ),
                dim=-1,
            ) @ text_classifier
            logits = logits.reshape(k, images.size(0), -1)
            target_logits = logits.gather(2, target_tensor[:, None, None].expand(-1, images.size(0), 1)).squeeze(-1)
            final_ce = F.cross_entropy(
                (logits.reshape(k * images.size(0), -1) / args.scan_temperature),
                target_tensor[:, None].expand(-1, images.size(0)).reshape(-1),
                reduction="none",
            ).reshape(k, images.size(0)).mean(dim=1)
            final_asr = logits.argmax(dim=-1).eq(target_tensor[:, None]).float().mean(dim=1)
            patch_mean = torch.sigmoid(patch_logits).mean(dim=(1, 2, 3))
            patch_std = torch.sigmoid(patch_logits).std(dim=(1, 2, 3))
            for j, target_idx in enumerate(chunk):
                ce = float(final_ce[j].cpu())
                asr = float(final_asr[j].cpu())
                mean_logit = float(target_logits[j].mean().cpu())
                record = {
                    "stage": "scan",
                    "target_index": int(target_idx),
                    "target_name": classes[target_idx],
                    "scan_ce": ce,
                    "scan_asr": asr,
                    "scan_target_logit": mean_logit,
                    "patch_mean": float(patch_mean[j].cpu()),
                    "patch_std": float(patch_std[j].cpu()),
                    "score": asr - args.scan_ce_weight * ce + args.scan_logit_weight * mean_logit,
                    "scan_steps": args.scan_steps,
                    "scan_patch_size": args.scan_patch_size,
                }
                all_records.append(record)
                with open(log_path, "a") as f:
                    f.write(json.dumps(record, sort_keys=True) + "\n")
        best_so_far = max(all_records, key=lambda r: r["score"])
        chunk_bar.set_postfix(target=best_so_far["target_name"], score=f"{best_so_far['score']:.3f}")
    all_records.sort(key=lambda r: r["score"], reverse=True)
    return int(all_records[0]["target_index"]), all_records[: args.scan_topk]


def invert_for_target(
    model,
    loader,
    text_classifier: torch.Tensor,
    classes: Sequence[str],
    target_idx: int,
    args,
    log_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    device = args.device
    data_iter = iter(loader)
    if args.target_invert_style == "invertune":
        mask_param = torch.nn.Parameter(torch.rand((1, 3, 224, 224), device=device))
        mean = mask_param.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = mask_param.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        low = (torch.zeros_like(mean) - mean) / std
        high = (torch.ones_like(mean) - mean) / std
        trigger_param = torch.nn.Parameter(low + torch.rand_like(mask_param) * (high - low))
        optimizer = torch.optim.Adam([mask_param, trigger_param], lr=args.target_invert_lr)
    else:
        mask_logits = torch.full(
            (1, 3, 224, 224),
            args.mask_init_logit,
            device=device,
            requires_grad=True,
        )
        trigger_pixel_logits = torch.nn.Parameter(
            torch.randn((1, 3, 224, 224), device=device) * 0.01
        )
        optimizer = torch.optim.Adam([mask_logits, trigger_pixel_logits], lr=args.lr)
    target = torch.empty(0, dtype=torch.long, device=device)
    start = time.time()
    last_record = {}
    pbar = tqdm(range(1, args.steps + 1), desc="stage0_invert")
    for step in pbar:
        try:
            images = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images = next(data_iter)
        images = images.to(device)
        if target.numel() != images.size(0):
            target = torch.full((images.size(0),), target_idx, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        if args.target_invert_style == "invertune":
            mask = mask_param.clamp(0, 1)
            trigger_norm = trigger_param.clamp(low, high)
        else:
            mask = torch.sigmoid(mask_logits)
            trigger_norm = normalized_from_pixel(torch.sigmoid(trigger_pixel_logits))
        logits = F.normalize(
            model.get_image_features(apply_proxy(images, mask, trigger_norm)),
            dim=-1,
        ) @ text_classifier
        ce = F.cross_entropy(logits / args.temperature, target)
        target_logit = logits[:, target_idx].mean()
        if args.target_invert_style == "invertune":
            loss = (
                args.target_infonce_weight * ce
                - args.target_logit_weight * target_logit
                + args.target_mask_sum_weight * mask.abs().sum()
                + args.mask_tv_weight * tv_loss(mask)
            )
        else:
            loss = (
                ce
                - args.target_logit_weight * target_logit
                + args.mask_l1_weight * mask.mean()
                + args.mask_tv_weight * tv_loss(mask)
                + args.min_mask_weight * torch.relu(args.min_mask_mean - mask.mean())
            )
        loss.backward()
        optimizer.step()
        if args.target_invert_style == "invertune":
            with torch.no_grad():
                mask_param.clamp_(0, 1)
                trigger_param.clamp_(low, high)
        if step == 1 or step % 10 == 0 or step == args.steps:
            asr = logits.detach().argmax(dim=1).eq(target_idx).float().mean()
            record = {
                "stage": "invert",
                "step": step,
                "target_index": int(target_idx),
                "target_name": classes[target_idx],
                "loss": float(loss.detach().cpu()),
                "target_ce": float(ce.detach().cpu()),
                "target_asr": float(asr.cpu()),
                "target_logit": float(target_logit.detach().cpu()),
                "mask_mean": float(mask.detach().mean().cpu()),
                "mask_l1": float(mask.detach().abs().sum().cpu()),
                "mask_active_gt_0.1": int((mask.detach() > 0.1).sum().cpu()),
                "elapsed_sec": time.time() - start,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            last_record = record
            pbar.set_postfix(asr=f"{record['target_asr']:.3f}", target=classes[target_idx])
    if args.target_invert_style == "invertune":
        return (
            mask_param.detach().clamp(0, 1).cpu(),
            trigger_param.detach().clamp(low, high).cpu(),
            last_record,
        )
    return (
        torch.sigmoid(mask_logits).detach().cpu(),
        normalized_from_pixel(torch.sigmoid(trigger_pixel_logits)).detach().cpu(),
        last_record,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Target-agnostic proxy trigger inversion")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=["target_agnostic", "scan_then_invert", "target_given"],
        default="target_agnostic",
        help=(
            "target_agnostic keeps the old delta-concentration objective; "
            "scan_then_invert first discovers a target with an InverTune-style "
            "candidate scan; target_given only uses --target_index for ablation."
        ),
    )
    parser.add_argument("--target_index", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean_csv", default=CC3M_CSV_DEFAULT)
    parser.add_argument("--cc3m_root", default=CC3M_ROOT_DEFAULT)
    parser.add_argument("--classes_path", default=IMAGENET_CLASSES_DEFAULT)
    parser.add_argument("--limit_rows", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--mask_init_logit", type=float, default=-4.0)
    parser.add_argument("--mask_l1_weight", type=float, default=0.0001)
    parser.add_argument("--mask_tv_weight", type=float, default=0.0001)
    parser.add_argument("--entropy_weight", type=float, default=0.2)
    parser.add_argument("--target_logit_weight", type=float, default=0.0)
    parser.add_argument("--target_invert_style", choices=["invertune", "sigmoid"], default="invertune")
    parser.add_argument("--target_invert_lr", type=float, default=0.01)
    parser.add_argument("--target_infonce_weight", type=float, default=5.0)
    parser.add_argument("--target_mask_sum_weight", type=float, default=0.01)
    parser.add_argument("--min_mask_mean", type=float, default=0.01)
    parser.add_argument("--min_mask_weight", type=float, default=0.1)
    parser.add_argument("--diagnostic_batches", type=int, default=8)
    parser.add_argument("--scan_candidates", default="all")
    parser.add_argument("--scan_topk", type=int, default=20)
    parser.add_argument("--scan_steps", type=int, default=20)
    parser.add_argument("--scan_batch_size", type=int, default=8)
    parser.add_argument("--scan_chunk_size", type=int, default=8)
    parser.add_argument("--scan_patch_size", type=int, default=16)
    parser.add_argument("--scan_location", choices=["middle", "bottom_right"], default="middle")
    parser.add_argument("--scan_lr", type=float, default=0.08)
    parser.add_argument("--scan_temperature", type=float, default=0.07)
    parser.add_argument("--scan_patch_l2_weight", type=float, default=0.0)
    parser.add_argument("--scan_ce_weight", type=float, default=0.05)
    parser.add_argument("--scan_logit_weight", type=float, default=0.5)
    parser.add_argument("--refine_topk", type=int, default=20)
    parser.add_argument("--refine_score_asr_weight", type=float, default=1.0)
    parser.add_argument("--refine_score_ce_weight", type=float, default=0.02)
    parser.add_argument("--refine_score_logit_weight", type=float, default=0.5)
    parser.add_argument(
        "--refine_score_mask_l1_weight",
        type=float,
        default=0.0,
        help="Applied to mask_l1 / numel(mask). Default ranks by diagnostic ASR.",
    )
    parser.add_argument(
        "--refine_rank_policy",
        choices=["weighted", "asr_ce"],
        default="asr_ce",
        help=(
            "weighted ranks by the scalar refine_score. asr_ce treats near-tied "
            "high-ASR candidates as saturated InverTune inversions and picks the "
            "lowest target CE, then highest target logit."
        ),
    )
    parser.add_argument(
        "--refine_asr_tolerance",
        type=float,
        default=0.02,
        help="ASR window used by --refine_rank_policy asr_ce.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    model, processor = load_clip_model(args.ckpt, args.device)
    model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    classes, templates = load_classes(args.classes_path)
    text_classifier = build_text_classifier(model, processor, classes, templates, args.device)
    ds = CleanImageDataset(args.cc3m_root, args.clean_csv, args.limit_rows)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    log_path = os.path.splitext(args.out)[0] + ".jsonl"

    if args.mode in {"scan_then_invert", "target_given"}:
        if args.mode == "scan_then_invert":
            target_idx, scan_topk = scan_target_candidates(
                model,
                loader,
                text_classifier,
                classes,
                args,
                log_path,
            )
        else:
            if args.target_index < 0:
                raise ValueError("--mode target_given requires --target_index")
            target_idx = args.target_index
            scan_topk = []
        refine_records = []
        if args.mode == "scan_then_invert" and args.refine_topk > 0:
            refine_states = {}
            for candidate in scan_topk[: args.refine_topk]:
                candidate_idx = int(candidate["target_index"])
                mask_i, trigger_i, last_i = invert_for_target(
                    model,
                    loader,
                    text_classifier,
                    classes,
                    candidate_idx,
                    args,
                    log_path,
                )
                diag_i = diagnose_target(
                    model,
                    eval_loader,
                    text_classifier,
                    candidate_idx,
                    mask_i.to(args.device),
                    trigger_i.to(args.device),
                    args.device,
                    args.diagnostic_batches,
                )
                mask_l1 = float(mask_i.abs().sum())
                mask_mean = float(mask_i.mean())
                refine_score = (
                    args.refine_score_asr_weight * diag_i["target_asr"]
                    - args.refine_score_ce_weight * diag_i["target_ce"]
                    + args.refine_score_logit_weight * diag_i["target_logit"]
                    - args.refine_score_mask_l1_weight * mask_mean
                )
                refine_record = {
                    "stage": "refine",
                    "target_index": candidate_idx,
                    "target_name": classes[candidate_idx],
                    "scan_record": candidate,
                    "target_asr": diag_i["target_asr"],
                    "target_ce": diag_i["target_ce"],
                    "target_logit": diag_i["target_logit"],
                    "mask_l1": mask_l1,
                    "mask_mean": mask_mean,
                    "mask_active_gt_0.1": int((mask_i > 0.1).sum()),
                    "refine_score": refine_score,
                    "last_record": last_i,
                }
                refine_records.append(refine_record)
                with open(log_path, "a") as f:
                    f.write(json.dumps(refine_record, sort_keys=True) + "\n")
                refine_states[candidate_idx] = {
                    "target_idx": candidate_idx,
                    "mask": mask_i,
                    "trigger": trigger_i,
                    "last_record": last_i,
                    "diagnostics": diag_i,
                    "record": refine_record,
                }
            if not refine_records:
                raise RuntimeError("No refine records were produced")
            refine_records.sort(key=lambda r: r["refine_score"], reverse=True)
            if args.refine_rank_policy == "asr_ce":
                max_asr = max(float(r["target_asr"]) for r in refine_records)
                eligible = [
                    r
                    for r in refine_records
                    if float(r["target_asr"]) >= max_asr - args.refine_asr_tolerance
                ]
                selected_record = sorted(
                    eligible,
                    key=lambda r: (
                        float(r["target_ce"]),
                        -float(r["target_logit"]),
                        float(r["mask_mean"]),
                        -float(r["target_asr"]),
                    ),
                )[0]
            else:
                selected_record = refine_records[0]
            best_refine = refine_states[int(selected_record["target_index"])]
            target_idx = int(best_refine["target_idx"])
            mask, trigger_norm = best_refine["mask"], best_refine["trigger"]
            last_record = best_refine["last_record"]
            target_diagnostics = {
                **best_refine["diagnostics"],
                "refine_rank_policy": args.refine_rank_policy,
                "refine_asr_tolerance": args.refine_asr_tolerance,
                "selected_refine_record": selected_record,
            }
        else:
            mask, trigger_norm, last_record = invert_for_target(
                model,
                loader,
                text_classifier,
                classes,
                target_idx,
                args,
                log_path,
            )
            target_diagnostics = diagnose_target(
                model,
                eval_loader,
                text_classifier,
                target_idx,
                mask.to(args.device),
                trigger_norm.to(args.device),
                args.device,
                args.diagnostic_batches,
            )
        delta_diagnostics = diagnose(
            model,
            eval_loader,
            text_classifier,
            classes,
            mask.to(args.device),
            trigger_norm.to(args.device),
            args.device,
            args.diagnostic_batches,
        )
        diagnostics = {
            **target_diagnostics,
            "target_index": int(target_idx),
            "target_name": classes[target_idx],
            "scan_topk": scan_topk,
            "refine_topk": refine_records,
            "delta_diagnostics": delta_diagnostics,
        }
        checkpoint = {
            "mask": mask,
            "trigger": trigger_norm,
            "target_name": classes[target_idx],
            "target_index": int(target_idx),
            "target_label": classes[target_idx],
            "diagnostics": diagnostics,
            "args": vars(args),
            "last_record": last_record,
            "created_at": datetime.now().isoformat(),
            "stage0_mode": f"{args.mode}_blackbox",
        }
        torch.save(checkpoint, args.out)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
            json.dump({k: v for k, v in checkpoint.items() if k not in {"mask", "trigger"}}, f, indent=2)
        print(json.dumps({k: v for k, v in checkpoint.items() if k not in {"mask", "trigger"}}, indent=2))
        return

    data_iter = iter(loader)
    mask_logits = torch.full(
        (1, 3, 224, 224),
        args.mask_init_logit,
        device=args.device,
        requires_grad=True,
    )
    trigger_pixel_logits = torch.randn((1, 3, 224, 224), device=args.device, requires_grad=True) * 0.01
    trigger_pixel_logits = torch.nn.Parameter(trigger_pixel_logits)
    optimizer = torch.optim.Adam([mask_logits, trigger_pixel_logits], lr=args.lr)

    start = time.time()
    last_record = {}
    pbar = tqdm(range(1, args.steps + 1), desc="stage0_blackbox")
    for step in pbar:
        try:
            images = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images = next(data_iter)
        images = images.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        mask = torch.sigmoid(mask_logits)
        trigger_norm = normalized_from_pixel(torch.sigmoid(trigger_pixel_logits))
        with torch.no_grad():
            clean_feats = F.normalize(model.get_image_features(images), dim=-1)
            clean_logits = clean_feats @ text_classifier
        trig_feats = F.normalize(model.get_image_features(apply_proxy(images, mask, trigger_norm)), dim=-1)
        trig_logits = trig_feats @ text_classifier
        delta_logits = trig_logits.mean(dim=0) - clean_logits.mean(dim=0)
        distribution = F.softmax(delta_logits / args.temperature, dim=0)
        entropy = -(distribution * torch.log(distribution.clamp_min(1e-12))).sum()
        concentration = torch.logsumexp(delta_logits / args.temperature, dim=0) * args.temperature
        loss = (
            -concentration
            + args.entropy_weight * entropy
            + args.mask_l1_weight * mask.mean()
            + args.mask_tv_weight * tv_loss(mask)
            + args.min_mask_weight * torch.relu(args.min_mask_mean - mask.mean())
        )
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            top_idx = int(delta_logits.detach().argmax())
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "concentration": float(concentration.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "mask_mean": float(mask.detach().mean().cpu()),
                "mask_active_gt_0.1": int((mask.detach() > 0.1).sum().cpu()),
                "target_index": top_idx,
                "target_name": classes[top_idx],
                "target_delta_logit": float(delta_logits.detach()[top_idx].cpu()),
                "elapsed_sec": time.time() - start,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            last_record = record
            pbar.set_postfix(target=classes[top_idx], d=f"{record['target_delta_logit']:.3f}")

    mask = torch.sigmoid(mask_logits).detach().cpu()
    trigger_norm = normalized_from_pixel(torch.sigmoid(trigger_pixel_logits)).detach().cpu()
    diagnostics = diagnose(
        model,
        eval_loader,
        text_classifier,
        classes,
        mask.to(args.device),
        trigger_norm.to(args.device),
        args.device,
        args.diagnostic_batches,
    )
    checkpoint = {
        "mask": mask,
        "trigger": trigger_norm,
        "target_name": diagnostics["target_name"],
        "target_index": diagnostics["target_index"],
        "target_label": diagnostics["target_name"],
        "diagnostics": diagnostics,
        "args": vars(args),
        "last_record": last_record,
        "created_at": datetime.now().isoformat(),
        "stage0_mode": "target_agnostic_blackbox",
    }
    torch.save(checkpoint, args.out)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump({k: v for k, v in checkpoint.items() if k not in {"mask", "trigger"}}, f, indent=2)
    print(json.dumps({k: v for k, v in checkpoint.items() if k not in {"mask", "trigger"}}, indent=2))


if __name__ == "__main__":
    main()
