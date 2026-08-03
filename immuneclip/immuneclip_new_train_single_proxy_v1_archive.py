#!/usr/bin/env python3
"""ImmuneCLIP new FT implementation.

Post-purification immunization for CLIP. This implementation follows the
reachable-set rebound-risk formulation:

  L = L_util + lambda_anchor L_anchor + lambda_dir L_dir + lambda_reach L_reach

The defense uses proxy triggers/targets only. Oracle triggers are intentionally
not part of the main training path.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - compatibility with older torch
    from torch.nn.utils.stateless import functional_call

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_SCRIPT_DIR = "/root/workspace/usenix/scripts"
if REPO_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, REPO_SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from clip_eval import eval_asr_ca, load_clip_model  # noqa: E402


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CC3M_ROOT_DEFAULT = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K"
CC3M_CSV_DEFAULT = (
    "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/"
    "cc3m_natural_10K_no_banana_strict.csv"
)
CLASSES_PATH_DEFAULT = "/root/autodl-tmp/datasets/imagenet1k_badclip/validation/classes.py"
PAR_ALIGN_CKPT = (
    "/root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/"
    "checkpoints/par_cleaned_rn50.pt"
)
PROXY_TRIGGER_DEFAULT = (
    "/root/autodl-tmp/experiments/immuneclip_week3_blackbox_stage0_formal_ce_rank/"
    "proxy_trigger.pt"
)


@dataclass
class ProxySpec:
    name: str
    mask: torch.Tensor
    trigger: torch.Tensor
    target_index: int
    target_name: str
    source_path: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_attention() -> None:
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return
    for name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_math_sdp", True),
    ):
        fn = getattr(cuda_backends, name, None)
        if fn is not None:
            fn(enabled)


def write_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def save_json(path: str, record: dict) -> None:
    with open(path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)


def clip_train_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class Cc3mPairDataset(Dataset):
    def __init__(self, root: str, csv_path: str, transform, limit_rows: int = 0):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Clean CSV not found: {csv_path}")
        self.root = root
        self.transform = transform
        self.rows: list[tuple[str, str]] = []
        missing = 0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image = row.get("image")
                caption = row.get("caption")
                if not image or caption is None:
                    continue
                path = image if os.path.isabs(image) else os.path.join(root, image)
                if not os.path.exists(path):
                    missing += 1
                    continue
                self.rows.append((path, str(caption)))
                if limit_rows > 0 and len(self.rows) >= limit_rows:
                    break
        if not self.rows:
            raise RuntimeError(f"No usable CC3M rows from {csv_path}")
        if missing:
            print(f"[warn] skipped {missing} missing CC3M images from {csv_path}", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, caption = self.rows[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        return self.transform(image), caption


def collate_pairs(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return images, captions


def get_loader(args, batch_size: int | None = None) -> DataLoader:
    ds = Cc3mPairDataset(
        root=args.cc3m_root,
        csv_path=args.clean_csv,
        transform=clip_train_transform(),
        limit_rows=args.limit_rows,
    )
    return DataLoader(
        ds,
        batch_size=batch_size or args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_pairs,
    )


def tokenize(processor, captions: Sequence[str], device: str):
    tokens = processor.process_text(list(captions))
    return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)


def clip_contrastive_loss(model, processor, images, captions, device: str):
    input_ids, attention_mask = tokenize(processor, captions, device)
    image_feats = F.normalize(model.get_image_features(images), dim=-1)
    text_feats = F.normalize(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)
    logits = model.logit_scale.exp() * image_feats @ text_feats.t()
    targets = torch.arange(images.size(0), device=images.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_t), input_ids, attention_mask


def visual_selected_overrides(named_params, offsets: Sequence[torch.Tensor] | None = None):
    overrides = {}
    for idx, (name, param) in enumerate(named_params):
        if not name.startswith("visual."):
            raise RuntimeError(
                f"Functional reach currently supports visual-only parameters, got {name}"
            )
        key = name[len("visual.") :]
        if offsets is None:
            overrides[key] = param
        else:
            overrides[key] = param + offsets[idx].to(device=param.device, dtype=param.dtype)
    return overrides


def visual_features_functional(model, images, named_params, offsets=None):
    feats = functional_call(model.visual, visual_selected_overrides(named_params, offsets), (images,))
    if isinstance(feats, (tuple, list)):
        feats = feats[0]
    return feats


def clip_contrastive_loss_functional_visual(
    model,
    processor,
    images,
    captions,
    device: str,
    named_params,
    offsets=None,
):
    input_ids, attention_mask = tokenize(processor, captions, device)
    image_feats = F.normalize(visual_features_functional(model, images, named_params, offsets), dim=-1)
    with torch.no_grad():
        text_feats = F.normalize(
            model.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
            dim=-1,
        )
    logits = model.logit_scale.exp().detach() * image_feats @ text_feats.t()
    targets = torch.arange(images.size(0), device=images.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_t)


def kd_loss(model, ref_model, images, input_ids, attention_mask):
    with torch.no_grad():
        ref_i = F.normalize(ref_model.get_image_features(images), dim=-1)
        ref_t = F.normalize(
            ref_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
            dim=-1,
        )
    cur_i = F.normalize(model.get_image_features(images), dim=-1)
    cur_t = F.normalize(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)
    return (1.0 - (cur_i * ref_i).sum(dim=-1).mean()) + (
        1.0 - (cur_t * ref_t).sum(dim=-1).mean()
    )


def load_classes(classes_path: str):
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(classes_path)
    config = eval(open(classes_path, "r").read())
    return config["classes"], config["templates"]


@torch.no_grad()
def build_text_classifier(model, processor, classes, templates, device: str) -> torch.Tensor:
    embeddings = []
    for c in tqdm(classes, desc="text_embed", leave=False):
        texts = [template(c) for template in templates]
        tokens = processor.process_text(texts)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        text_embedding = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = F.normalize(text_embedding, dim=-1).mean(dim=0)
        embeddings.append(F.normalize(text_embedding, dim=0))
    return torch.stack(embeddings, dim=1).detach()


def resolve_target_index(classes: Sequence[str], target_label: str, metadata: dict) -> tuple[int, str]:
    if target_label == "auto":
        idx = metadata.get("target_index")
        name = metadata.get("target_name") or metadata.get("target_label")
        if idx is not None and 0 <= int(idx) < len(classes):
            return int(idx), classes[int(idx)]
        if name:
            target_label = str(name)
    matches = [i for i, c in enumerate(classes) if c == target_label or target_label in c]
    if not matches:
        raise ValueError(f"Target label {target_label!r} not found in ImageNet classes")
    return int(matches[0]), classes[int(matches[0])]


def load_proxy_checkpoint(path: str, device: str, dtype: torch.dtype):
    ckpt = torch.load(path, map_location="cpu")
    if "mask" not in ckpt or "trigger" not in ckpt:
        raise KeyError(f"Proxy checkpoint must contain mask/trigger: {path}")
    mask = ckpt["mask"].to(device=device, dtype=dtype).clamp(0, 1)
    trigger = ckpt["trigger"].to(device=device, dtype=dtype)
    mean = trigger.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = trigger.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    trigger = trigger.clamp((0 - mean) / std, (1 - mean) / std)
    metadata = {
        "target_label": ckpt.get("target_label"),
        "target_name": ckpt.get("target_name"),
        "target_index": ckpt.get("target_index"),
        "diagnostics": ckpt.get("diagnostics", {}),
        "stage0_mode": ckpt.get("stage0_mode"),
    }
    return mask, trigger, metadata


def roll_proxy(mask: torch.Tensor, trigger: torch.Tensor, shift_y: int, shift_x: int):
    return torch.roll(mask, shifts=(shift_y, shift_x), dims=(-2, -1)), torch.roll(
        trigger, shifts=(shift_y, shift_x), dims=(-2, -1)
    )


def make_single_proxy_augmentations(args, classes, device: str) -> list[ProxySpec]:
    paths = [p.strip() for p in args.proxy_trigger_paths.split(",") if p.strip()]
    if not paths:
        raise ValueError("--proxy_trigger_paths must contain at least one proxy checkpoint")
    if len(paths) > 1:
        raise ValueError(
            "Single Proxy ImmuneCLIP expects exactly one proxy checkpoint. "
            "Use one InverTune-style inversion result and treat variants only as augmentations."
        )
    specs: list[ProxySpec] = []
    variant_plan = [
        ("base", 0, 0, 1.00, 0.00),
        ("weak", 0, 0, 0.75, 0.00),
        ("shift_up_left", -4, -4, 1.00, 0.00),
        ("shift_down_right", 4, 4, 1.00, 0.00),
        ("noisy", 0, 0, 0.90, 0.01),
        ("shift_up_right", -4, 4, 1.00, 0.00),
        ("shift_down_left", 4, -4, 1.00, 0.00),
        ("faint", 0, 0, 0.60, 0.00),
    ]
    path = paths[0]
    mask, trigger, metadata = load_proxy_checkpoint(path, device, torch.float32)
    target_index, target_name = resolve_target_index(classes, args.target_label, metadata)
    for suffix, sy, sx, strength, noise_std in variant_plan[: max(1, args.proxy_variants)]:
        vmask, vtrigger = roll_proxy(mask, trigger, sy, sx)
        if strength != 1.0:
            vtrigger = strength * vtrigger
        if noise_std > 0:
            vtrigger = vtrigger + noise_std * torch.randn_like(vtrigger)
        specs.append(
            ProxySpec(
                name=f"{os.path.basename(path)}:{suffix}",
                mask=vmask.detach(),
                trigger=vtrigger.detach(),
                target_index=target_index,
                target_name=target_name,
                source_path=path,
            )
        )
    return specs


def select_named_params(model, scope: str, keywords: Sequence[str]):
    if scope == "all_visual":
        selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith("visual.")]
    elif scope == "selected":
        selected = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in keywords)
        ]
    elif scope == "all":
        selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    else:
        raise ValueError(scope)
    if not selected:
        raise RuntimeError(f"No trainable parameters selected for scope={scope}")
    return selected


def set_trainable(model, train_scope: str, keywords: Sequence[str]) -> None:
    for name, p in model.named_parameters():
        if train_scope == "all":
            enabled = True
        elif train_scope == "all_visual":
            enabled = name.startswith("visual.")
        elif train_scope == "selected":
            enabled = any(k in name for k in keywords)
        else:
            raise ValueError(train_scope)
        p.requires_grad_(enabled)


def dot_like(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> torch.Tensor:
    total = None
    for x, y in zip(xs, ys):
        v = (x * y).sum()
        total = v if total is None else total + v
    if total is None:
        raise RuntimeError("empty tensor list")
    return total


def grad_list(loss, params: Sequence[torch.nn.Parameter], create_graph: bool, retain_graph: bool):
    grads = torch.autograd.grad(
        loss,
        list(params),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def unit_direction(grads: Sequence[torch.Tensor], sign: float = -1.0, eps: float = 1e-12):
    norm = dot_like(grads, grads).detach().sqrt().clamp_min(eps)
    return [(sign * g.detach()) / norm for g in grads], norm


def adam_sign_direction(grads: Sequence[torch.Tensor], sign: float = -1.0, eps: float = 1e-12):
    direction = [(sign * torch.sign(g.detach())) for g in grads]
    norm = dot_like(direction, direction).detach().sqrt().clamp_min(eps)
    return [d / norm for d in direction], norm


def apply_proxy(images: torch.Tensor, proxy: ProxySpec) -> torch.Tensor:
    mask = proxy.mask.to(device=images.device, dtype=images.dtype)
    trigger = proxy.trigger.to(device=images.device, dtype=images.dtype)
    return (1 - mask) * images + mask * trigger


def targeted_margin_risk(
    model,
    images: torch.Tensor,
    text_classifier: torch.Tensor,
    proxy: ProxySpec,
    margin_tau: float,
) -> torch.Tensor:
    feats = F.normalize(model.get_image_features(apply_proxy(images, proxy)), dim=-1)
    logits = feats @ text_classifier
    target = logits[:, proxy.target_index]
    masked = logits.clone()
    masked[:, proxy.target_index] = -1e4
    margin = target - masked.max(dim=1).values
    return F.softplus(margin / margin_tau).mean()


def targeted_margin_risk_functional_visual(
    model,
    images: torch.Tensor,
    text_classifier: torch.Tensor,
    proxy: ProxySpec,
    margin_tau: float,
    named_params,
    offsets=None,
) -> torch.Tensor:
    feats = F.normalize(
        visual_features_functional(model, apply_proxy(images, proxy), named_params, offsets),
        dim=-1,
    )
    logits = feats @ text_classifier
    target = logits[:, proxy.target_index]
    masked = logits.clone()
    masked[:, proxy.target_index] = -1e4
    margin = target - masked.max(dim=1).values
    return F.softplus(margin / margin_tau).mean()


def single_proxy_augmented_risk(
    model,
    images: torch.Tensor,
    text_classifier: torch.Tensor,
    proxy_augmentations: Sequence[ProxySpec],
    margin_tau: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    scores = [
        targeted_margin_risk(model, images, text_classifier, proxy_aug, margin_tau)
        for proxy_aug in proxy_augmentations
    ]
    return torch.stack(scores).mean(), scores


def single_proxy_augmented_risk_functional_visual(
    model,
    images: torch.Tensor,
    text_classifier: torch.Tensor,
    proxy_augmentations: Sequence[ProxySpec],
    margin_tau: float,
    named_params,
    offsets=None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    scores = [
        targeted_margin_risk_functional_visual(
            model, images, text_classifier, proxy_aug, margin_tau, named_params, offsets
        )
        for proxy_aug in proxy_augmentations
    ]
    return torch.stack(scores).mean(), scores


def smooth_max(values: Sequence[torch.Tensor], mode: str, temperature: float, top_frac: float):
    if not values:
        return torch.zeros(())
    stacked = torch.stack([v.reshape(()) for v in values])
    if mode == "mean":
        return stacked.mean()
    if mode == "max":
        return stacked.max()
    if mode == "cvar":
        k = max(1, int(round(stacked.numel() * top_frac)))
        return torch.topk(stacked, k=k).values.mean()
    if mode == "lse":
        return temperature * torch.logsumexp(stacked / temperature, dim=0)
    raise ValueError(mode)


def add_param_direction(params, direction, scale: float):
    deltas = []
    with torch.no_grad():
        for p, u in zip(params, direction):
            delta = scale * u.to(device=p.device, dtype=p.dtype)
            p.add_(delta)
            deltas.append(delta)
    return deltas


def remove_param_direction(params, deltas):
    with torch.no_grad():
        for p, d in zip(params, deltas):
            p.sub_(d)


def save_ckpt(model, path: str, args, step: int, metadata: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "step": step,
            "args": vars(args),
            "metadata": metadata,
        },
        path,
    )


def evaluate_to_json(model, processor, args, path: str, step: int, tag: str):
    model.eval()
    cwd = os.getcwd()
    try:
        os.chdir("/root/workspace/usenix/baselines/BadCLIP")
        metrics = eval_asr_ca(
            None,
            device=args.device,
            subset=args.eval_subset,
            model=model,
            processor=processor,
        )
    finally:
        os.chdir(cwd)
        model.train()
    record = {"step": step, "tag": tag, **metrics}
    save_json(path, record)
    return record


def parse_args():
    p = argparse.ArgumentParser(description="ImmuneCLIP new reachable-set FT immunization")
    p.add_argument("--init_ckpt", default=PAR_ALIGN_CKPT)
    p.add_argument("--ref_ckpt", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clean_csv", default=CC3M_CSV_DEFAULT)
    p.add_argument("--cc3m_root", default=CC3M_ROOT_DEFAULT)
    p.add_argument("--limit_rows", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--update_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_scope", choices=["selected", "all_visual", "all"], default="selected")
    p.add_argument("--param_keywords", nargs="+", default=["visual.layer4", "visual.attnpool"])

    p.add_argument("--proxy_trigger_paths", default=PROXY_TRIGGER_DEFAULT)
    p.add_argument(
        "--proxy_variants",
        type=int,
        default=4,
        help="Number of augmentations for the single proxy; not multiple proxy hypotheses.",
    )
    p.add_argument(
        "--proxies_per_step",
        type=int,
        default=0,
        help="Deprecated compatibility argument. Single Proxy mode always averages all proxy augmentations.",
    )
    p.add_argument("--target_label", default="auto")
    p.add_argument("--classes_path", default=CLASSES_PATH_DEFAULT)
    p.add_argument("--margin_tau", type=float, default=0.05)
    p.add_argument("--anchor_threshold", type=float, default=0.08)

    p.add_argument("--num_update_dirs", type=int, default=2)
    p.add_argument(
        "--update_dir_modes",
        nargs="+",
        choices=["grad", "sign_precond", "adam_sign"],
        default=["grad"],
        help="Clean downstream update directions to cover in L_dir/L_reach.",
    )
    p.add_argument("--dir_smooth", choices=["lse", "cvar", "mean", "max"], default="cvar")
    p.add_argument("--dir_temperature", type=float, default=0.05)
    p.add_argument("--dir_top_frac", type=float, default=0.35)
    p.add_argument("--reach_steps", type=int, default=1)
    p.add_argument("--reach_radius", type=float, default=1e-4)
    p.add_argument(
        "--reach_mode",
        choices=["traj_global", "checkpoint_rho"],
        default="traj_global",
        help=(
            "traj_global optimizes a global SmoothMax over finite directional risk "
            "increments. checkpoint_rho performs a detached 1-step virtual clean "
            "optimizer step and recomputes rho_hat at theta'."
        ),
    )
    p.add_argument("--virtual_optimizer", choices=["sgd", "sign_precond"], default="sgd")
    p.add_argument("--virtual_lr", type=float, default=1e-6)
    p.add_argument(
        "--virtual_lrs",
        type=str,
        default="",
        help="Optional comma-separated virtual SGD learning rates for reachable checkpoints.",
    )
    p.add_argument("--reach_smooth", choices=["lse", "cvar", "mean", "max"], default="cvar")
    p.add_argument("--reach_temperature", type=float, default=0.05)
    p.add_argument("--reach_top_frac", type=float, default=0.35)

    p.add_argument("--lambda_clip", type=float, default=1.0)
    p.add_argument("--lambda_kd", type=float, default=0.5)
    p.add_argument("--lambda_anchor", type=float, default=1.0)
    p.add_argument("--lambda_dir", type=float, default=0.2)
    p.add_argument("--lambda_reach", type=float, default=0.2)

    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--eval_subset", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    args.update_dir_modes = [
        "sign_precond" if mode == "adam_sign" else mode for mode in args.update_dir_modes
    ]
    args.virtual_lr_values = (
        [float(x.strip()) for x in args.virtual_lrs.split(",") if x.strip()]
        if args.virtual_lrs
        else [args.virtual_lr]
    )
    if args.reach_mode == "checkpoint_rho" and args.reach_steps != 1:
        raise ValueError("--reach_mode checkpoint_rho currently implements exactly --reach_steps 1")
    configure_attention()
    set_seed(args.seed)
    for d in ("logs", "results", "checkpoints"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)
    save_json(os.path.join(args.out_dir, "results", f"{args.tag}_args.json"), vars(args))

    model, processor = load_clip_model(args.init_ckpt, args.device)
    model.float().train()
    ref_model, _ = load_clip_model(args.ref_ckpt, args.device)
    ref_model.float().eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    set_trainable(model, args.train_scope, args.param_keywords)
    named_params = select_named_params(model, args.train_scope, args.param_keywords)
    non_visual = [name for name, _p in named_params if not name.startswith("visual.")]
    if non_visual:
        raise RuntimeError(
            "immuneclip_new_train.py currently supports visual-only immunization. "
            f"Non-visual trainable parameters would require refreshing text classifier: {non_visual[:5]}"
        )
    params = [p for _n, p in named_params]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    classes, templates = load_classes(args.classes_path)
    text_classifier = build_text_classifier(model, processor, classes, templates, args.device)
    proxy_augmentations = make_single_proxy_augmentations(args, classes, args.device)
    single_proxy = proxy_augmentations[0]

    loader = get_loader(args, args.batch_size)
    update_loader = get_loader(args, args.update_batch_size)
    data_iter = itertools.cycle(loader)
    update_iter = itertools.cycle(update_loader)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "init_ckpt": args.init_ckpt,
        "selected_param_count": int(sum(p.numel() for p in params)),
        "selected_param_names": [n for n, _p in named_params],
        "single_proxy_mode": True,
        "proxy_source_path": single_proxy.source_path,
        "proxy_target_index": single_proxy.target_index,
        "proxy_target_name": single_proxy.target_name,
        "target_label_arg": args.target_label,
        "target_source_note": (
            "target is read from proxy checkpoint metadata when target_label=auto; "
            "if that checkpoint was created with banana supervision, report as target-known."
        ),
        "proxy_augmentations_not_hypotheses": True,
        "proxy_augmentations": [
            {
                "name": p.name,
                "target_index": p.target_index,
                "target_name": p.target_name,
                "source_path": p.source_path,
            }
            for p in proxy_augmentations
        ],
    }
    save_json(os.path.join(args.out_dir, "results", f"{args.tag}_init_summary.json"), metadata)

    train_log = os.path.join(args.out_dir, "logs", f"{args.tag}_train.jsonl")
    eval_pre = evaluate_to_json(
        model,
        processor,
        args,
        os.path.join(args.out_dir, "results", "eval_pre.json"),
        0,
        "pre",
    )
    print(json.dumps({"stage": "eval_pre", **eval_pre}), flush=True)

    start = time.time()
    last_record = {}
    pbar = tqdm(range(1, args.steps + 1), desc=args.tag)
    for step in pbar:
        images, captions = next(data_iter)
        images = images.to(args.device, non_blocking=True)

        update_dirs = []
        update_norms = []
        update_mode_names = []
        virtual_offsets = []
        virtual_offset_norms = []
        for _ in range(args.num_update_dirs):
            u_images, u_captions = next(update_iter)
            u_images = u_images.to(args.device, non_blocking=True)
            u_loss, _ids, _mask = clip_contrastive_loss(model, processor, u_images, u_captions, args.device)
            u_grads = grad_list(u_loss, params, create_graph=False, retain_graph=False)
            if "grad" in args.update_dir_modes:
                u_dir, u_norm = unit_direction(u_grads, sign=-1.0)
                update_dirs.append(u_dir)
                update_norms.append(float(u_norm.cpu()))
                update_mode_names.append("grad")
            if "sign_precond" in args.update_dir_modes:
                u_dir, u_norm = adam_sign_direction(u_grads, sign=-1.0)
                update_dirs.append(u_dir)
                update_norms.append(float(u_norm.cpu()))
                update_mode_names.append("sign_precond")
            if args.reach_mode == "checkpoint_rho":
                if args.virtual_optimizer == "sgd":
                    offset_list = [
                        [(-virtual_lr * g.detach()) for g in u_grads]
                        for virtual_lr in args.virtual_lr_values
                    ]
                elif args.virtual_optimizer == "sign_precond":
                    step_dir, _step_norm = adam_sign_direction(u_grads, sign=-1.0)
                    offset_list = [[(args.reach_radius * d.detach()) for d in step_dir]]
                else:
                    raise ValueError(args.virtual_optimizer)
                for offsets in offset_list:
                    virtual_offsets.append(offsets)
                    virtual_offset_norms.append(float(dot_like(offsets, offsets).detach().sqrt().cpu()))

        optimizer.zero_grad(set_to_none=True)
        loss_clip, input_ids, attention_mask = clip_contrastive_loss(
            model, processor, images, captions, args.device
        )
        loss_kd = kd_loss(model, ref_model, images, input_ids, attention_mask)
        loss_util = args.lambda_clip * loss_clip + args.lambda_kd * loss_kd

        base_score, aug_scores = single_proxy_augmented_risk(
            model, images, text_classifier, proxy_augmentations, args.margin_tau
        )
        anchor_terms = [F.relu(base_score - args.anchor_threshold)]
        loss_anchor = smooth_max(anchor_terms, args.dir_smooth, args.dir_temperature, args.dir_top_frac)

        s_grads = grad_list(base_score, params, create_graph=True, retain_graph=True)
        dir_terms = []
        dir_raw = []
        for u_dir in update_dirs:
            d_k = dot_like(s_grads, u_dir)
            dir_raw.append(float(d_k.detach().cpu()))
            dir_terms.append(F.relu(d_k))
        loss_dir = smooth_max(dir_terms, args.dir_smooth, args.dir_temperature, args.dir_top_frac)

        reach_values = []
        reach_terms = []
        if args.lambda_reach > 0 and args.reach_steps > 0:
            base_detached = base_score.detach()
            if args.reach_mode == "traj_global":
                for u_dir in update_dirs:
                    for h in range(1, args.reach_steps + 1):
                        radius = args.reach_radius * h
                        offsets = [(radius * u.detach()) for u in u_dir]
                        future_score, _future_aug_scores = single_proxy_augmented_risk_functional_visual(
                            model,
                            images,
                            text_classifier,
                            proxy_augmentations,
                            args.margin_tau,
                            named_params,
                            offsets,
                        )
                        val = F.relu(future_score - base_detached) / (radius + 1e-12)
                        reach_terms.append(val)
                        reach_values.append(float(val.detach().cpu()))
            elif args.reach_mode == "checkpoint_rho":
                for offsets in virtual_offsets:
                    r_images, r_captions = next(update_iter)
                    r_images = r_images.to(args.device, non_blocking=True)
                    r_loss = clip_contrastive_loss_functional_visual(
                        model,
                        processor,
                        r_images,
                        r_captions,
                        args.device,
                        named_params,
                        offsets,
                    )
                    r_grads = grad_list(r_loss, params, create_graph=False, retain_graph=True)
                    checkpoint_dirs = []
                    if "grad" in args.update_dir_modes:
                        r_dir, _r_norm = unit_direction(r_grads, sign=-1.0)
                        checkpoint_dirs.append(r_dir)
                    if "sign_precond" in args.update_dir_modes:
                        r_dir, _r_norm = adam_sign_direction(r_grads, sign=-1.0)
                        checkpoint_dirs.append(r_dir)
                    future_score, _future_aug_scores = single_proxy_augmented_risk_functional_visual(
                        model,
                        images,
                        text_classifier,
                        proxy_augmentations,
                        args.margin_tau,
                        named_params,
                        offsets,
                    )
                    s_prime_grads = grad_list(future_score, params, create_graph=True, retain_graph=True)
                    for r_dir in checkpoint_dirs:
                        val = dot_like(s_prime_grads, r_dir)
                        reach_terms.append(F.relu(val))
                        reach_values.append(float(val.detach().cpu()))
            else:
                raise ValueError(args.reach_mode)
        loss_reach = (
            smooth_max(reach_terms, args.reach_smooth, args.reach_temperature, args.reach_top_frac)
            if reach_terms
            else torch.zeros((), device=args.device)
        )

        total_loss = (
            loss_util
            + args.lambda_anchor * loss_anchor
            + args.lambda_dir * loss_dir
            + args.lambda_reach * loss_reach
        )
        total_loss.backward()

        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        else:
            grad_norm = torch.zeros(())
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            record = {
                "stage": "train",
                "step": step,
                "loss_clip": float(loss_clip.detach().cpu()),
                "loss_kd": float(loss_kd.detach().cpu()),
                "loss_anchor": float(loss_anchor.detach().cpu()),
                "loss_dir": float(loss_dir.detach().cpu()),
                "loss_reach": float(loss_reach.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "single_proxy_score": float(base_score.detach().cpu()),
                "aug_score_mean": float(torch.stack(aug_scores).mean().detach().cpu()),
                "aug_score_max": float(torch.stack(aug_scores).max().detach().cpu()),
                "aug_score_min": float(torch.stack(aug_scores).min().detach().cpu()),
                "dir_raw_max": max(dir_raw) if dir_raw else 0.0,
                "dir_raw_mean": sum(dir_raw) / max(1, len(dir_raw)),
                "dir_active_ratio": sum(1 for v in dir_raw if v > 0) / max(1, len(dir_raw)),
                "reach_raw_max": max(reach_values) if reach_values else 0.0,
                "reach_raw_mean": sum(reach_values) / max(1, len(reach_values)),
                "reach_active_ratio": sum(1 for v in reach_values if v > 0) / max(1, len(reach_values)),
                "anchor_active_ratio": float(
                    float((base_score.detach() - args.anchor_threshold).cpu()) > 0
                ),
                "update_norm_mean": sum(update_norms) / max(1, len(update_norms)),
                "virtual_offset_norm_mean": sum(virtual_offset_norms) / max(1, len(virtual_offset_norms)),
                "update_dir_modes_used": update_mode_names,
                "reach_mode": args.reach_mode,
                "virtual_optimizer": args.virtual_optimizer,
                "virtual_lr_values": args.virtual_lr_values,
                "grad_norm": float(grad_norm.detach().cpu()),
                "single_proxy": single_proxy.name,
                "proxy_augmentations": [p.name for p in proxy_augmentations],
                "elapsed_sec": time.time() - start,
            }
            last_record = record
            write_jsonl(train_log, record)
            print(json.dumps(record), flush=True)
            pbar.set_postfix(
                anchor=f"{record['loss_anchor']:.3g}",
                dir=f"{record['loss_dir']:.3g}",
                reach=f"{record['loss_reach']:.3g}",
            )

        if args.save_every > 0 and step % args.save_every == 0:
            save_ckpt(
                model,
                os.path.join(args.out_dir, "checkpoints", f"{args.tag}_step{step}.pt"),
                args,
                step,
                metadata,
            )

        if args.eval_every > 0 and step % args.eval_every == 0:
            record = evaluate_to_json(
                model,
                processor,
                args,
                os.path.join(args.out_dir, "results", f"eval_step{step}.json"),
                step,
                f"step{step}",
            )
            print(json.dumps({"stage": "eval", **record}), flush=True)

    final_ckpt = os.path.join(args.out_dir, "checkpoints", f"{args.tag}_final.pt")
    save_ckpt(model, final_ckpt, args, args.steps, metadata)
    eval_final = evaluate_to_json(
        model,
        processor,
        args,
        os.path.join(args.out_dir, "results", "eval_final.json"),
        args.steps,
        "final",
    )
    summary = {
        "created_at": datetime.now().isoformat(),
        "tag": args.tag,
        "out_dir": args.out_dir,
        "final_ckpt": final_ckpt,
        "train_log": train_log,
        "eval_pre": eval_pre,
        "eval_final": eval_final,
        "last_train_record": last_record,
        "args": vars(args),
        "metadata": metadata,
    }
    save_json(os.path.join(args.out_dir, "results", f"{args.tag}_summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
