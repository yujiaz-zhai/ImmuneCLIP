#!/usr/bin/env python3
"""Proxy trigger inversion for ImmuneCLIP Week3.

The first Week3 implementation uses this as a practical Tier-2 proxy: optimize
a universal patch on clean CC3M images to maximize similarity to one or more
candidate target texts, then save the best patch for immune training.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from clip_eval import load_clip_model  # noqa: E402


CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
CC3M_ROOT_DEFAULT = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K"
CC3M_CSV_DEFAULT = (
    "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/"
    "cc3m_natural_10K_no_banana_strict.csv"
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CLIP_MEAN.flatten().tolist(), CLIP_STD.flatten().tolist()),
        ]
    )


class Cc3mImageDataset(Dataset):
    def __init__(self, root: str, csv_path: str, limit_rows: int):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Clean CSV not found: {csv_path}")
        self.root = root
        self.transform = clip_transform()
        self.rows: List[str] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = row.get("image")
                if not image:
                    continue
                path = image if os.path.isabs(image) else os.path.join(root, image)
                if os.path.exists(path):
                    self.rows.append(path)
                if limit_rows > 0 and len(self.rows) >= limit_rows:
                    break
        if not self.rows:
            raise RuntimeError(f"No images found from {csv_path} under {root}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.rows[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img)


def tokenize(processor, texts: Sequence[str], device: str):
    tokens = processor.process_text(list(texts))
    return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)


@torch.no_grad()
def encode_target(model, processor, target: str, device: str):
    texts = [
        f"a photo of a {target}.",
        f"a close-up photo of a {target}.",
        f"a cropped photo of the {target}.",
    ]
    input_ids, attention_mask = tokenize(processor, texts, device)
    feats = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    feats = F.normalize(feats, dim=-1).mean(dim=0)
    return F.normalize(feats, dim=0)


def apply_patch(images: torch.Tensor, patch_pixel: torch.Tensor, location: str):
    mean = CLIP_MEAN.to(images.device, images.dtype)
    std = CLIP_STD.to(images.device, images.dtype)
    patch = (patch_pixel - mean) / std
    out = images.clone()
    _, _, h, w = out.shape
    _, ph, pw = patch.shape
    if location == "middle":
        top, left = int(h / 2 - ph / 2), int(w / 2 - pw / 2)
    elif location == "bottom_right":
        top, left = h - ph, w - pw
    else:
        raise ValueError(location)
    out[:, :, top : top + ph, left : left + pw] = patch.unsqueeze(0)
    return out


def save_patch_image(patch_pixel: torch.Tensor, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    torchvision.utils.save_image(patch_pixel.detach().cpu().clamp(0, 1), out_png)


def optimize_for_target(args, model, processor, target: str) -> Tuple[float, str, str]:
    out_dir = os.path.join(args.out_dir, target.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(
        Cc3mImageDataset(args.cc3m_root, args.clean_csv, args.limit_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    target_text = encode_target(model, processor, target, args.device)
    patch_pixel = torch.full(
        (3, args.patch_size, args.patch_size),
        0.5,
        device=args.device,
        requires_grad=True,
    )
    opt = torch.optim.Adam([patch_pixel], lr=args.lr)
    last_score = 0.0
    data_iter = iter(loader)
    for step in tqdm(range(1, args.steps + 1), desc=f"invert:{target}"):
        try:
            images = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images = next(data_iter)
        images = images.to(args.device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        patched = apply_patch(images, patch_pixel.clamp(0, 1), args.patch_location)
        feats = F.normalize(model.get_image_features(patched), dim=-1)
        score = (feats @ target_text).mean()
        tv = (
            torch.abs(patch_pixel[:, :, 1:] - patch_pixel[:, :, :-1]).mean()
            + torch.abs(patch_pixel[:, 1:, :] - patch_pixel[:, :-1, :]).mean()
        )
        loss = -score + args.lambda_tv * tv
        loss.backward()
        opt.step()
        with torch.no_grad():
            patch_pixel.clamp_(0, 1)
        last_score = float(score.detach().cpu())
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            with open(os.path.join(out_dir, "invert_log.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "score": last_score, "loss": float(loss.detach().cpu())}) + "\n")

    pt_path = os.path.join(out_dir, "delta_hat_pixel.pt")
    png_path = os.path.join(out_dir, "delta_hat.png")
    torch.save({"patch_pixel": patch_pixel.detach().cpu(), "target": target, "score": last_score}, pt_path)
    save_patch_image(patch_pixel, png_path)
    return last_score, pt_path, png_path


def parse_args():
    p = argparse.ArgumentParser(description="Invert a proxy universal patch for ImmuneCLIP")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean_csv", default=CC3M_CSV_DEFAULT)
    p.add_argument("--cc3m_root", default=CC3M_ROOT_DEFAULT)
    p.add_argument("--limit_rows", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda_tv", type=float, default=0.001)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--patch_location", choices=["middle", "bottom_right"], default="middle")
    p.add_argument("--candidate_targets", default="banana")
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    model, processor = load_clip_model(args.ckpt, args.device)
    model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    records = []
    for target in [t.strip() for t in args.candidate_targets.split(",") if t.strip()]:
        score, pt_path, png_path = optimize_for_target(args, model, processor, target)
        records.append({"target": target, "score": score, "pt_path": pt_path, "png_path": png_path})
    records.sort(key=lambda r: r["score"], reverse=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": args.ckpt,
        "best": records[0],
        "records": records,
        "args": vars(args),
    }
    with open(os.path.join(args.out_dir, "target_hat.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
