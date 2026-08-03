#!/usr/bin/env python3
"""ImmuneCLIP Week3 immune fine-tuning.

This script is intentionally additive: it reuses the Week1/2 CLIP loading,
ASR/CA evaluation and CC3M clean-data protocol, while adding the Week3
first-order/second-order immune objectives.
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
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from clip_eval import eval_asr_ca, load_clip_model  # noqa: E402


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CC3M_ROOT_DEFAULT = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K"
CC3M_CSV_DEFAULT = (
    "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/"
    "cc3m_natural_10K_no_banana_strict.csv"
)
ORACLE_PATCH_DEFAULT = (
    "/root/autodl-tmp/experiments/immuneclip_week2/"
    "BadCLIP_GradAlign/opti_patches/badCLIP.jpg"
)
IMAGENET_CLASSES_DEFAULT = "/root/autodl-tmp/datasets/imagenet1k_badclip/validation/classes.py"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_higher_order_attention() -> None:
    """Use SDP kernels that support the higher-order gradients needed by L_fo/L_so."""
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
    def __init__(
        self,
        root: str,
        csv_path: str,
        transform,
        limit_rows: int = 0,
        image_key: str = "image",
        caption_key: str = "caption",
    ):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"CC3M csv not found: {csv_path}. Refusing to fall back because "
                "Week3 immune/rebound results depend on the exact clean CSV."
            )
        self.root = root
        self.transform = transform
        self.rows: List[Tuple[str, str]] = []
        missing = 0
        missing_examples: List[str] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get(image_key)
                cap = row.get(caption_key)
                if not img or cap is None:
                    continue
                path = img if os.path.isabs(img) else os.path.join(root, img)
                if not os.path.exists(path):
                    missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(path)
                self.rows.append((path, str(cap)))
                if limit_rows > 0 and len(self.rows) >= limit_rows:
                    break
        if not self.rows:
            raise RuntimeError(f"No CC3M pairs found from {csv_path} under {root}")
        if missing:
            ratio = missing / len(self.rows)
            msg = (
                f"Missing {missing}/{len(self.rows)} CC3M images for csv={csv_path}; "
                f"examples={missing_examples}"
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
            image = Image.new("RGB", (224, 224))
        return self.transform(image), caption


def collate_pairs(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return images, captions


def get_loader(args) -> DataLoader:
    ds = Cc3mPairDataset(
        root=args.cc3m_root,
        csv_path=args.clean_csv,
        transform=clip_train_transform(),
        limit_rows=args.limit_rows,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_pairs,
    )


def tokenize(processor, captions: Sequence[str], device: str):
    tokens = processor.process_text(list(captions))
    return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)


def encode_texts(model, processor, texts: Sequence[str], device: str) -> torch.Tensor:
    input_ids, attention_mask = tokenize(processor, texts, device)
    text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    text_features = F.normalize(text_features, dim=-1)
    text_embedding = text_features.mean(dim=0)
    return F.normalize(text_embedding, dim=0)


def target_templates(target_label: str, classes_path: str) -> List[str]:
    if os.path.isfile(classes_path):
        config = eval(open(classes_path, "r").read())
        classes, templates = config["classes"], config["templates"]
        matches = [c for c in classes if c == target_label or target_label in c]
        if matches:
            return [template(matches[0]) for template in templates]
    return [
        f"a photo of a {target_label}.",
        f"a close-up photo of a {target_label}.",
        f"a cropped photo of the {target_label}.",
    ]


@torch.no_grad()
def build_target_text(model, processor, target_label: str, classes_path: str, device: str) -> torch.Tensor:
    return encode_texts(model, processor, target_templates(target_label, classes_path), device).detach()


def clip_contrastive_loss(model, processor, images: torch.Tensor, captions: Sequence[str], device: str):
    input_ids, attention_mask = tokenize(processor, captions, device)
    image_feats = F.normalize(model.get_image_features(images), dim=-1)
    text_feats = F.normalize(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_feats @ text_feats.t()
    targets = torch.arange(images.size(0), device=images.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_t), input_ids, attention_mask


def kd_loss(model, ref_model, images, input_ids, attention_mask):
    with torch.no_grad():
        ref_i = F.normalize(ref_model.get_image_features(images), dim=-1)
        ref_t = F.normalize(
            ref_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
            dim=-1,
        )
    cur_i = F.normalize(model.get_image_features(images), dim=-1)
    cur_t = F.normalize(model.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)
    loss_i = 1.0 - (cur_i * ref_i).sum(dim=-1).mean()
    loss_t = 1.0 - (cur_t * ref_t).sum(dim=-1).mean()
    return loss_i + loss_t


def load_patch(path: str, patch_size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Patch not found: {path}")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((patch_size, patch_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    patch = transform(Image.open(path).convert("RGB")).to(device=device, dtype=dtype)
    return patch


def apply_trigger(images: torch.Tensor, patch: torch.Tensor, location: str) -> torch.Tensor:
    out = images.clone()
    _, _, h, w = out.shape
    _, ph, pw = patch.shape
    if location == "middle":
        top, left = int(h / 2 - ph / 2), int(w / 2 - pw / 2)
    elif location == "bottom_right":
        top, left = h - ph, w - pw
    else:
        raise ValueError(f"Unsupported patch location: {location}")
    out[:, :, top : top + ph, left : left + pw] = patch.unsqueeze(0)
    return out


def load_proxy_trigger(path: str, device: str, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    if not path:
        raise ValueError("--proxy_trigger_path is required when --trigger_mode=proxy")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Proxy trigger checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if "mask" not in checkpoint or "trigger" not in checkpoint:
        raise KeyError(f"Invalid proxy trigger checkpoint: {path}; expected keys mask/trigger")
    mask = checkpoint["mask"].to(device=device, dtype=dtype).clamp(0, 1)
    trigger = checkpoint["trigger"].to(device=device, dtype=dtype)
    mean = trigger.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = trigger.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    trigger = trigger.clamp((0 - mean) / std, (1 - mean) / std)
    metadata = {
        "target_label": checkpoint.get("target_label"),
        "target_name": checkpoint.get("target_name"),
        "target_index": checkpoint.get("target_index"),
        "diagnostics": checkpoint.get("diagnostics", {}),
        "stage0_mode": checkpoint.get("stage0_mode"),
    }
    return mask, trigger, metadata


def apply_proxy_trigger(images: torch.Tensor, mask: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=images.device, dtype=images.dtype)
    trigger = trigger.to(device=images.device, dtype=images.dtype)
    return (1 - mask) * images + mask * trigger


def apply_backdoor_trigger(
    images: torch.Tensor,
    trigger_mode: str,
    oracle_patch: torch.Tensor | None,
    proxy_trigger: Tuple[torch.Tensor, torch.Tensor, dict] | None,
    location: str,
) -> torch.Tensor:
    if trigger_mode == "oracle":
        if oracle_patch is None:
            raise RuntimeError("oracle_patch is required for trigger_mode=oracle")
        return apply_trigger(images, oracle_patch, location)
    if trigger_mode == "proxy":
        if proxy_trigger is None:
            raise RuntimeError("proxy_trigger is required for trigger_mode=proxy")
        mask, trigger, _metadata = proxy_trigger
        return apply_proxy_trigger(images, mask, trigger)
    raise NotImplementedError("trigger_mode=free is not implemented in this Week3 script")


def backdoor_score(
    model,
    images: torch.Tensor,
    target_text: torch.Tensor,
    trigger_mode: str,
    oracle_patch: torch.Tensor | None,
    proxy_trigger: Tuple[torch.Tensor, torch.Tensor, dict] | None,
    location: str,
):
    x_trig = apply_backdoor_trigger(images, trigger_mode, oracle_patch, proxy_trigger, location)
    image_feats = F.normalize(model.get_image_features(x_trig), dim=-1)
    return (image_feats @ target_text).mean()


def select_named_params(model, keywords: Sequence[str]):
    selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in keywords)]
    if not selected:
        raise RuntimeError(f"No trainable parameters matched keywords={keywords}")
    return selected


def grad_list(loss, params: Sequence[torch.nn.Parameter], create_graph: bool, retain_graph: bool):
    grads = torch.autograd.grad(
        loss,
        list(params),
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def flatten(xs: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([x.reshape(-1) for x in xs])


def cosine_from_grads(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], eps: float = 1e-12):
    af, bf = flatten(a), flatten(b)
    return torch.dot(af, bf) / (af.norm() * bf.norm() + eps)


def unit_like(grads: Sequence[torch.Tensor], eps: float = 1e-12):
    flat = flatten(grads)
    norm = flat.norm().detach().clamp_min(eps)
    return [g.detach() / norm for g in grads]


def dot_like(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]):
    return sum((x * y).sum() for x, y in zip(a, b))


def add_param_perturbation(params: Sequence[torch.nn.Parameter], direction: Sequence[torch.Tensor], rho: float):
    perturbations = []
    with torch.no_grad():
        for p, d in zip(params, direction):
            delta = rho * d.to(device=p.device, dtype=p.dtype)
            p.add_(delta)
            perturbations.append(delta)
    return perturbations


def remove_param_perturbation(params: Sequence[torch.nn.Parameter], perturbations: Sequence[torch.Tensor]):
    with torch.no_grad():
        for p, delta in zip(params, perturbations):
            p.sub_(delta)


def project_selected_grads(params: Sequence[torch.nn.Parameter], s_grads: Sequence[torch.Tensor], eps: float = 1e-12):
    raw_grads = [torch.zeros_like(p) if p.grad is None else p.grad for p in params]
    dot = dot_like(raw_grads, s_grads)
    norm2 = dot_like(s_grads, s_grads).detach().clamp_min(eps)
    if dot.item() < 0:
        coeff = dot.detach() / norm2
        for p, s in zip(params, s_grads):
            if p.grad is not None:
                p.grad.sub_(coeff * s.detach())
    return dot.detach(), norm2.detach().sqrt()


def save_ckpt(model, path: str, args, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "step": step, "args": vars(args)}, path)


def write_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="ImmuneCLIP Week3 immune fine-tuning")
    parser.add_argument("--init_ckpt", required=True, help="Poisoned or cleaned checkpoint to immunize")
    parser.add_argument("--ref_ckpt", default=None, help="Clean reference checkpoint. Default uses pretrained RN50.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tag", default="immuneclip_oracle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--clean_csv", default=CC3M_CSV_DEFAULT)
    parser.add_argument("--cc3m_root", default=CC3M_ROOT_DEFAULT)
    parser.add_argument("--limit_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--trigger_mode", choices=["oracle", "proxy"], default="oracle")
    parser.add_argument("--patch_path", default=None)
    parser.add_argument("--proxy_trigger_path", default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--patch_location", choices=["middle", "bottom_right"], default="middle")
    parser.add_argument(
        "--target_label",
        default="banana",
        help="'auto' reads target_name/target_label from the proxy trigger checkpoint.",
    )
    parser.add_argument("--classes_path", default=IMAGENET_CLASSES_DEFAULT)

    parser.add_argument("--mode", choices=["loss", "surgery"], default="loss")
    parser.add_argument(
        "--lambda_clip",
        type=float,
        default=1.0,
        help="Weight of the clean CLIP contrastive loss in the immune update. "
        "L_fo/L_so still use the clean CLIP gradient even when this is reduced.",
    )
    parser.add_argument("--lambda_kd", type=float, default=0.5)
    parser.add_argument("--lambda_fo", type=float, default=0.1)
    parser.add_argument("--lambda_so", type=float, default=0.0)
    parser.add_argument(
        "--lambda_nb",
        type=float,
        default=0.0,
        help="Stage2 neighborhood robustness weight for L_fo/L_so at theta+rho*grad_S.",
    )
    parser.add_argument("--neighborhood_rho", type=float, default=1e-4)
    parser.add_argument(
        "--lambda_supp",
        type=float,
        default=0.0,
        help="Direct suppression weight for ReLU(S - suppression_margin).",
    )
    parser.add_argument("--suppression_margin", type=float, default=0.05)
    parser.add_argument("--align_param_keywords", nargs="+", default=["visual.layer4", "visual.attnpool"])

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=0, help="0 disables in-training ASR/CA eval")
    parser.add_argument("--eval_subset", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_higher_order_attention()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    result_dir = os.path.join(args.out_dir, "results")
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    for d in (log_dir, result_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    args_path = os.path.join(result_dir, f"{args.tag}_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    model, processor = load_clip_model(args.init_ckpt, args.device)
    model.float().train()
    ref_model, _ = load_clip_model(args.ref_ckpt, args.device)
    ref_model.float().eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    for p in model.parameters():
        p.requires_grad_(True)
    theta_named = select_named_params(model, args.align_param_keywords)
    theta_names = [n for n, _ in theta_named]
    theta_params = [p for _, p in theta_named]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loader = get_loader(args)
    data_iter = itertools.cycle(loader)
    oracle_patch = None
    proxy_trigger = None
    proxy_metadata = {}
    if args.trigger_mode == "oracle":
        oracle_patch = load_patch(args.patch_path, args.patch_size, args.device, torch.float32)
    elif args.trigger_mode == "proxy":
        proxy_trigger = load_proxy_trigger(args.proxy_trigger_path, args.device, torch.float32)
        proxy_metadata = proxy_trigger[2]
    else:
        raise NotImplementedError(f"Unsupported trigger_mode={args.trigger_mode}")
    target_label = args.target_label
    if target_label == "auto":
        target_label = proxy_metadata.get("target_name") or proxy_metadata.get("target_label")
        if not target_label:
            raise ValueError(
                "--target_label auto requires proxy checkpoint metadata target_name/target_label"
            )
    target_text = build_target_text(model, processor, target_label, args.classes_path, args.device)

    train_log = os.path.join(log_dir, f"{args.tag}_train.jsonl")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "tag": args.tag,
        "init_ckpt": args.init_ckpt,
        "out_dir": args.out_dir,
        "selected_param_count": sum(p.numel() for p in theta_params),
        "selected_param_names": theta_names,
        "resolved_target_label": target_label,
        "proxy_metadata": proxy_metadata,
        "args_path": args_path,
        "train_log": train_log,
    }
    with open(os.path.join(result_dir, f"{args.tag}_init_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    start = time.time()
    last_record = {}
    pbar = tqdm(range(1, args.steps + 1), desc=args.tag)
    for step in pbar:
        images, captions = next(data_iter)
        images = images.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        loss_clip, input_ids, attention_mask = clip_contrastive_loss(
            model, processor, images, captions, args.device
        )
        loss_kd = kd_loss(model, ref_model, images, input_ids, attention_mask)
        score_s = backdoor_score(
            model,
            images,
            target_text,
            args.trigger_mode,
            oracle_patch,
            proxy_trigger,
            args.patch_location,
        )
        loss_supp = torch.relu(score_s - args.suppression_margin)

        if args.mode == "surgery":
            s_grads = grad_list(score_s, theta_params, create_graph=False, retain_graph=True)
            loss_base = args.lambda_clip * loss_clip + args.lambda_kd * loss_kd + args.lambda_supp * loss_supp
            loss_base.backward()
            dot_before, s_norm = project_selected_grads(theta_params, s_grads)
            loss_fo = torch.relu(-dot_before / (s_norm * s_norm + 1e-12))
            loss_so = torch.zeros((), device=args.device)
            cos_s_gc = dot_before / (s_norm * s_norm + 1e-12)
            total_loss = loss_base.detach()
        else:
            need_higher = args.lambda_fo > 0 or args.lambda_so > 0
            g_clean = grad_list(loss_clip, theta_params, create_graph=need_higher, retain_graph=True)
            s_grads = grad_list(score_s, theta_params, create_graph=need_higher, retain_graph=True)
            cos_s_gc = cosine_from_grads(s_grads, g_clean)
            loss_fo = torch.relu(-cos_s_gc)
            loss_so = torch.zeros((), device=args.device)
            if args.lambda_so > 0:
                ghat = unit_like(g_clean)
                hv_scalar = dot_like(s_grads, ghat)
                hvp = grad_list(hv_scalar, theta_params, create_graph=True, retain_graph=True)
                curvature = dot_like(hvp, ghat)
                loss_so = torch.relu(curvature)
            total_loss = (
                args.lambda_clip * loss_clip
                + args.lambda_kd * loss_kd
                + args.lambda_supp * loss_supp
                + args.lambda_fo * loss_fo
                + args.lambda_so * loss_so
            )
            total_loss.backward()

        loss_nb_fo = torch.zeros((), device=args.device)
        loss_nb_so = torch.zeros((), device=args.device)
        cos_nb = torch.zeros((), device=args.device)
        if args.lambda_nb > 0 and args.mode == "loss":
            direction = unit_like(s_grads)
            perturbations = add_param_perturbation(theta_params, direction, args.neighborhood_rho)
            try:
                nb_loss_clip, _nb_input_ids, _nb_attention_mask = clip_contrastive_loss(
                    model, processor, images, captions, args.device
                )
                nb_score_s = backdoor_score(
                    model,
                    images,
                    target_text,
                    args.trigger_mode,
                    oracle_patch,
                    proxy_trigger,
                    args.patch_location,
                )
                nb_need_higher = args.lambda_so > 0
                nb_g_clean = grad_list(
                    nb_loss_clip,
                    theta_params,
                    create_graph=nb_need_higher,
                    retain_graph=True,
                )
                nb_s_grads = grad_list(
                    nb_score_s,
                    theta_params,
                    create_graph=nb_need_higher,
                    retain_graph=True,
                )
                cos_nb = cosine_from_grads(nb_s_grads, nb_g_clean)
                loss_nb_fo = torch.relu(-cos_nb)
                if args.lambda_so > 0:
                    nb_ghat = unit_like(nb_g_clean)
                    nb_hv_scalar = dot_like(nb_s_grads, nb_ghat)
                    nb_hvp = grad_list(nb_hv_scalar, theta_params, create_graph=True, retain_graph=True)
                    nb_curvature = dot_like(nb_hvp, nb_ghat)
                    loss_nb_so = torch.relu(nb_curvature)
                nb_total = args.lambda_nb * (loss_nb_fo + args.lambda_so * loss_nb_so)
                nb_total.backward()
            finally:
                remove_param_perturbation(theta_params, perturbations)

        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        else:
            grad_norm = torch.tensor(0.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            record = {
                "step": step,
                "loss": float(total_loss.detach().cpu()),
                "loss_clip": float(loss_clip.detach().cpu()),
                "loss_kd": float(loss_kd.detach().cpu()),
                "loss_supp": float(loss_supp.detach().cpu()),
                "loss_fo": float(loss_fo.detach().cpu()),
                "loss_so": float(loss_so.detach().cpu()),
                "loss_nb_fo": float(loss_nb_fo.detach().cpu()),
                "loss_nb_so": float(loss_nb_so.detach().cpu()),
                "cos_s_gc": float(cos_s_gc.detach().cpu()),
                "cos_nb": float(cos_nb.detach().cpu()),
                "score_s": float(score_s.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "elapsed_sec": time.time() - start,
            }
            if args.eval_every > 0 and step % args.eval_every == 0:
                model.eval()
                metrics = eval_asr_ca(None, device=args.device, subset=args.eval_subset, model=model, processor=processor)
                model.train()
                record.update({f"quick_{k}": v for k, v in metrics.items()})
            write_jsonl(train_log, record)
            last_record = record
            pbar.set_postfix(
                loss=f"{record['loss']:.4f}",
                cos=f"{record['cos_s_gc']:.3f}",
                S=f"{record['score_s']:.3f}",
            )

        if args.save_every > 0 and step % args.save_every == 0:
            save_ckpt(model, os.path.join(ckpt_dir, f"{args.tag}_step{step}.pt"), args, step)

    final_ckpt = os.path.join(ckpt_dir, f"{args.tag}_final.pt")
    save_ckpt(model, final_ckpt, args, args.steps)
    summary.update(
        {
            "finished_at": datetime.now().isoformat(),
            "final_ckpt": final_ckpt,
            "elapsed_sec": time.time() - start,
            "last_record": last_record,
        }
    )
    summary_path = os.path.join(result_dir, f"{args.tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved final checkpoint: {final_ckpt}")
    print(f"Saved train log: {train_log}")


if __name__ == "__main__":
    main()
