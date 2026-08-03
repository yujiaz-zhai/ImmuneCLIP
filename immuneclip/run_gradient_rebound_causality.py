#!/usr/bin/env python3
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
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = "/root/workspace/usenix"
SCRIPT_DIR = os.path.join(PROJECT_ROOT, "scripts")
BADCLIP_ROOT = os.path.join(PROJECT_ROOT, "baselines", "BadCLIP")
sys.path.insert(0, SCRIPT_DIR)

from clip_eval import eval_asr_ca, load_clip_model  # noqa: E402
from immuneclip_train import (  # noqa: E402
    ORACLE_PATCH_DEFAULT,
    apply_backdoor_trigger,
    backdoor_score,
    build_target_text,
    clip_contrastive_loss,
    configure_higher_order_attention,
    get_loader,
    load_patch,
    load_proxy_trigger,
)

os.chdir(PROJECT_ROOT)


CC3M_ROOT_DEFAULT = "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K"
CC3M_CSV_STRICT = (
    "/root/autodl-tmp/datasets/cc3m_badclip/GCC_Training500K/"
    "cc3m_natural_10K_no_banana_strict.csv"
)
CLASSES_PATH = "/root/autodl-tmp/datasets/imagenet1k_badclip/validation/classes.py"
PAR_ALIGN_CKPT = (
    "/root/autodl-tmp/experiments/immuneclip_week2/defense_align_ep10/"
    "checkpoints/par_cleaned_rn50.pt"
)
PROXY_TRIGGER = (
    "/root/autodl-tmp/experiments/immuneclip_week3_blackbox_stage0_formal_ce_rank/"
    "proxy_trigger.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def save_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dot_like(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> torch.Tensor:
    total = None
    for x, y in zip(xs, ys):
        v = (x * y).sum()
        total = v if total is None else total + v
    if total is None:
        raise RuntimeError("empty gradient list")
    return total


def grad_list(loss, params: Sequence[torch.nn.Parameter], retain_graph: bool):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(grads, params)]


def clean_target_score(model, images, target_text) -> torch.Tensor:
    image_features = model.get_image_features(images)
    image_features = F.normalize(image_features, dim=-1)
    return (image_features @ target_text).mean()


@torch.no_grad()
def build_class_text_matrix(model, processor, classes_path: str, target_label: str, device: str):
    config = eval(open(classes_path, "r").read())
    classes, templates = config["classes"], config["templates"]
    matches = [i for i, c in enumerate(classes) if c == target_label or target_label in c]
    if not matches:
        raise ValueError(f"Target label {target_label!r} not found in {classes_path}")
    text_embeddings = []
    for c in classes:
        text = [template(c) for template in templates]
        tokens = processor.process_text(text)
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        text_embedding = model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_embedding = F.normalize(text_embedding, dim=-1).mean(dim=0)
        text_embeddings.append(F.normalize(text_embedding, dim=0))
    return torch.stack(text_embeddings, dim=1).detach(), int(matches[0])


def image_features(model, images):
    return F.normalize(model.get_image_features(images), dim=-1)


def target_margin_score(image_feats, class_texts, target_index: int) -> torch.Tensor:
    logits = image_feats @ class_texts
    target = logits[:, target_index]
    non_target = (logits.sum(dim=1) - target) / max(1, logits.size(1) - 1)
    return (target - non_target).mean()


def readout_score(
    model,
    images,
    target_text,
    class_texts,
    target_index,
    trigger_kind: str,
    oracle_patch,
    proxy_trigger,
    patch_location: str,
    score_definition: str,
) -> torch.Tensor:
    x_trig = apply_backdoor_trigger(
        images,
        trigger_kind,
        None if trigger_kind == "proxy" else oracle_patch,
        proxy_trigger if trigger_kind == "proxy" else None,
        patch_location,
    )
    triggered_feats = image_features(model, x_trig)
    triggered = (triggered_feats @ target_text).mean()
    if score_definition == "raw":
        return triggered
    if score_definition == "trigger_delta":
        return triggered - clean_target_score(model, images, target_text)
    if score_definition == "trigger_margin_delta":
        if class_texts is None or target_index is None:
            raise RuntimeError("class_texts/target_index are required for trigger_margin_delta")
        clean_feats = image_features(model, images)
        return (
            target_margin_score(triggered_feats, class_texts, target_index)
            - target_margin_score(clean_feats, class_texts, target_index)
        )
    raise ValueError(f"unknown score_definition={score_definition}")


def cosine_update_to_s(g_clean, s_grads, eps: float = 1e-12) -> torch.Tensor:
    # User-facing quantity is cos(-g_c, grad S), not cos(g_c, grad S).
    dot = dot_like(g_clean, s_grads)
    g_norm = dot_like(g_clean, g_clean).sqrt()
    s_norm = dot_like(s_grads, s_grads).sqrt()
    return -dot / (g_norm * s_norm + eps)


def set_projected_grads(params, g_clean, s_grads, enabled: bool, eps: float = 1e-12):
    dot = dot_like(g_clean, s_grads)
    s_norm2 = dot_like(s_grads, s_grads).clamp_min(eps)
    coeff = dot / s_norm2
    for p, g, s in zip(params, g_clean, s_grads):
        if not p.requires_grad:
            continue
        if enabled:
            p.grad = (g - coeff * s).detach().clone()
        else:
            p.grad = g.detach().clone()
    return dot.detach(), s_norm2.detach().sqrt(), coeff.detach()


def project_existing_grads(
    params,
    s_grads,
    enabled: bool,
    eps: float = 1e-12,
    harmful_only: bool = False,
):
    """Project the already-populated p.grad values along s_grads in-place.

    This keeps the downstream update path equivalent to run_downstream.py:
    first run loss.backward(), then optionally remove the backdoor-direction
    component from the clean gradient before optimizer.step().
    """
    dot = None
    g_norm2 = None
    s_norm2 = None
    usable = []
    for p, s in zip(params, s_grads):
        if s is None:
            s = torch.zeros_like(p)
        else:
            s = s.detach()
        if p.grad is None:
            g = torch.zeros_like(p)
        else:
            g = p.grad.detach()
        cur_dot = (g * s).sum()
        cur_g = (g * g).sum()
        cur_s = (s * s).sum()
        dot = cur_dot if dot is None else dot + cur_dot
        g_norm2 = cur_g if g_norm2 is None else g_norm2 + cur_g
        s_norm2 = cur_s if s_norm2 is None else s_norm2 + cur_s
        usable.append((p, s))
    if dot is None or g_norm2 is None or s_norm2 is None:
        raise RuntimeError("empty gradient list")

    s_norm2 = s_norm2.clamp_min(eps)
    coeff = dot / s_norm2
    should_project = enabled and (not harmful_only or float(dot.detach().cpu()) < 0.0)
    if should_project:
        for p, s in usable:
            if p.requires_grad and p.grad is not None:
                p.grad.add_(s, alpha=-float(coeff.detach().cpu()))
    cos_update = -dot / (g_norm2.sqrt() * s_norm2.sqrt() + eps)
    return dot.detach(), s_norm2.sqrt().detach(), coeff.detach(), cos_update.detach()


def project_matched_random_grads(
    params,
    oracle_grads,
    enabled: bool,
    eps: float = 1e-12,
    harmful_only: bool = True,
):
    """Remove the oracle positive-reactivation component along a random direction.

    Let g be the clean loss gradient and s be grad S. The downstream update is
    proportional to -g, so the harmful component is active when <g, s> < 0.
    This control removes exactly that oracle component magnitude, but along an
    unrelated random unit direction, then rescales the final gradient norm to
    match the oracle-projected update. This blocks the "smaller step size"
    alternative explanation for the causal projection experiment.
    """
    dot = None
    g_norm2 = None
    s_norm2 = None
    r_norm2 = None
    usable = []
    for p, s in zip(params, oracle_grads):
        if s is None:
            s = torch.zeros_like(p)
        else:
            s = s.detach()
        if p.grad is None:
            g = torch.zeros_like(p)
        else:
            g = p.grad.detach()
        r = torch.randn_like(g)
        dot = (g * s).sum() if dot is None else dot + (g * s).sum()
        g_norm2 = (g * g).sum() if g_norm2 is None else g_norm2 + (g * g).sum()
        s_norm2 = (s * s).sum() if s_norm2 is None else s_norm2 + (s * s).sum()
        r_norm2 = (r * r).sum() if r_norm2 is None else r_norm2 + (r * r).sum()
        usable.append((p, g, s, r))
    if dot is None or g_norm2 is None or s_norm2 is None or r_norm2 is None:
        raise RuntimeError("empty gradient list")

    s_norm2 = s_norm2.clamp_min(eps)
    r_norm2 = r_norm2.clamp_min(eps)
    dot_g_shat = dot / s_norm2.sqrt()
    should_project = enabled and (not harmful_only or float(dot.detach().cpu()) < 0.0)
    if not should_project:
        cos_update = -dot / (g_norm2.sqrt() * s_norm2.sqrt() + eps)
        return dot.detach(), s_norm2.sqrt().detach(), torch.tensor(0.0, device=dot.device), cos_update.detach(), torch.tensor(1.0, device=dot.device)

    # Oracle projection target norm: ||g - <g, shat> shat||.
    target_norm2 = (g_norm2 - dot_g_shat.pow(2)).clamp_min(eps)
    r_norm = r_norm2.sqrt()
    tmp_norm2 = None
    for _p, g, _s, r in usable:
        tmp = g - dot_g_shat * (r / r_norm)
        cur = (tmp * tmp).sum()
        tmp_norm2 = cur if tmp_norm2 is None else tmp_norm2 + cur
    scale = (target_norm2.sqrt() / tmp_norm2.clamp_min(eps).sqrt()).detach()

    for p, g, _s, r in usable:
        if p.requires_grad and p.grad is not None:
            p.grad = ((g - dot_g_shat * (r / r_norm)) * scale).detach().clone()
    cos_update = -dot / (g_norm2.sqrt() * s_norm2.sqrt() + eps)
    return dot.detach(), s_norm2.sqrt().detach(), dot_g_shat.detach(), cos_update.detach(), scale.detach()


def random_direction_like(params):
    return [
        torch.randn_like(p.grad if p.grad is not None else p)
        for p in params
    ]


def shuffled_direction_like(s_grads):
    shuffled = []
    for s in s_grads:
        if s is None:
            shuffled.append(None)
            continue
        flat = s.detach().reshape(-1)
        if flat.numel() <= 1:
            shuffled.append(flat.clone().reshape_as(s))
            continue
        idx = torch.randperm(flat.numel(), device=flat.device)
        shuffled.append(flat[idx].reshape_as(s).contiguous())
    return shuffled


def unit_projection_from_existing_grads(params, s_grads, eps: float = 1e-12) -> torch.Tensor:
    dot, s_norm, _coeff, _cos = project_existing_grads(params, s_grads, enabled=False, eps=eps)
    return -dot / (s_norm + eps)


def random_null_unit_projections(params, n: int) -> list[float]:
    samples = []
    for _ in range(n):
        rnd = random_direction_like(params)
        value = unit_projection_from_existing_grads(params, rnd)
        samples.append(float(value.cpu()))
        del rnd
    return samples


def select_params(model, scope: str, keywords: Sequence[str]):
    if scope == "all":
        return [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    result = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and any(k in n for k in keywords)
    ]
    if not result:
        raise RuntimeError(f"No parameters matched {keywords}")
    return result


def resolve_ckpt(value: str | None):
    if value is None or value in {"", "none", "clean"}:
        return None
    return value


def parse_eval_steps(eval_steps: str | None, eval_every: int, total_steps: int) -> set[int]:
    steps = {0, total_steps}
    if eval_steps:
        for part in eval_steps.split(","):
            part = part.strip()
            if not part:
                continue
            value = int(part)
            if 0 <= value <= total_steps:
                steps.add(value)
        return steps
    if eval_every > 0:
        steps.update(range(eval_every, total_steps + 1, eval_every))
    return steps


def run(args):
    configure_higher_order_attention()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    model, processor = load_clip_model(resolve_ckpt(args.ckpt), args.device)
    model.float().train()
    for p in model.parameters():
        p.requires_grad_(True)
    named_params = select_params(model, args.param_scope, args.param_keywords)
    params = [p for _n, p in named_params]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loader = get_loader(args)
    data_iter = itertools.cycle(loader)

    oracle_patch = load_patch(args.patch_path, args.patch_size, args.device, torch.float32)
    proxy_trigger = None
    if args.proxy_trigger_path:
        proxy_trigger = load_proxy_trigger(args.proxy_trigger_path, args.device, torch.float32)

    target_text = build_target_text(model, processor, args.target_label, args.classes_path, args.device)
    class_texts = None
    target_index = None
    if args.score_definition == "trigger_margin_delta":
        class_texts, target_index = build_class_text_matrix(
            model, processor, args.classes_path, args.target_label, args.device
        )
    jsonl_path = os.path.join(args.out_dir, f"{args.tag}_steps.jsonl")
    traj_path = os.path.join(args.out_dir, f"traj_{args.tag}.csv")
    summary_path = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    args_path = os.path.join(args.out_dir, f"{args.tag}_args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    rows = []
    start = time.time()
    eval_steps = parse_eval_steps(args.eval_steps, args.eval_every, args.steps)

    def evaluate(step: int, train_loss: float, cos_oracle=None, cos_proxy=None):
        if step not in eval_steps:
            return
        model.eval()
        cwd = os.getcwd()
        try:
            os.chdir(BADCLIP_ROOT)
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
        row = {
            "step": step,
            "train_loss": train_loss,
            "cos_update_oracle": cos_oracle,
            "cos_update_proxy": cos_proxy,
            **metrics,
        }
        rows.append(row)
        save_csv(traj_path, rows)
        write_jsonl(jsonl_path, {"stage": "eval", **row, "elapsed_sec": time.time() - start})

    evaluate(0, 0.0)

    for step in range(1, args.steps + 1):
        images, captions = next(data_iter)
        images = images.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        loss_clip, _input_ids, _attention_mask = clip_contrastive_loss(
            model, processor, images, captions, args.device
        )

        cos_proxy = None
        score_proxy_value = None
        score_oracle_value = None
        dot_before = torch.tensor(0.0, device=args.device)
        s_norm = torch.tensor(0.0, device=args.device)
        coeff = torch.tensor(0.0, device=args.device)
        cos_oracle = None
        projected = False
        random_match_scale = None

        loss_clip.backward()

        unit_projection = None
        null_unit_projection_samples = None

        if args.mode == "project":
            trigger_kind = args.project_trigger
            if trigger_kind == "proxy" and proxy_trigger is None:
                trigger_kind = "oracle"
            if trigger_kind == "proxy_shuffled" and proxy_trigger is None:
                raise RuntimeError("--project_trigger proxy_shuffled requires --proxy_trigger_path")
            if trigger_kind == "random":
                score_project = None
                s_project = random_direction_like(params)
            elif trigger_kind == "random_matched":
                score_project = readout_score(
                    model,
                    images,
                    target_text,
                    class_texts,
                    target_index,
                    "oracle",
                    oracle_patch,
                    proxy_trigger,
                    args.patch_location,
                    args.score_definition,
                )
                s_project = torch.autograd.grad(
                    score_project,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            elif trigger_kind == "proxy_shuffled":
                score_project = readout_score(
                    model,
                    images,
                    target_text,
                    class_texts,
                    target_index,
                    "proxy",
                    oracle_patch,
                    proxy_trigger,
                    args.patch_location,
                    args.score_definition,
                )
                proxy_grads = torch.autograd.grad(
                    score_project,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                s_project = shuffled_direction_like(proxy_grads)
                del proxy_grads
            else:
                score_project = readout_score(
                    model,
                    images,
                    target_text,
                    class_texts,
                    target_index,
                    trigger_kind,
                    oracle_patch,
                    proxy_trigger,
                    args.patch_location,
                    args.score_definition,
                )
                s_project = torch.autograd.grad(
                    score_project,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
            if trigger_kind in {"proxy", "proxy_shuffled"}:
                score_proxy_value = float(score_project.detach().cpu())
            elif trigger_kind in {"oracle", "random_matched"}:
                score_oracle_value = float(score_project.detach().cpu())
            if trigger_kind == "random_matched":
                dot_before, s_norm, coeff, cos_tensor, random_match_scale = project_matched_random_grads(
                    params,
                    s_project,
                    enabled=True,
                    harmful_only=args.project_harmful_only,
                )
            else:
                dot_before, s_norm, coeff, cos_tensor = project_existing_grads(
                    params, s_project, enabled=True, harmful_only=args.project_harmful_only
                )
            if trigger_kind in {"proxy", "proxy_shuffled"}:
                cos_proxy = float(cos_tensor.cpu())
            elif trigger_kind in {"oracle", "random_matched"}:
                cos_oracle = float(cos_tensor.cpu())
            unit_projection = float((-dot_before / (s_norm + 1e-12)).cpu())
            projected = True
            del s_project
        elif args.diagnostic_every > 0 and (
            step % args.diagnostic_every == 0 or step == 1 or step == args.steps
        ):
            score_oracle = readout_score(
                model,
                images,
                target_text,
                class_texts,
                target_index,
                "oracle",
                oracle_patch,
                None,
                args.patch_location,
                args.score_definition,
            )
            score_oracle_value = float(score_oracle.detach().cpu())
            s_oracle = torch.autograd.grad(
                score_oracle,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            dot_before, s_norm, coeff, cos_tensor = project_existing_grads(
                params, s_oracle, enabled=False
            )
            cos_oracle = float(cos_tensor.cpu())
            unit_projection = float((-dot_before / (s_norm + 1e-12)).cpu())
            if args.null_samples > 0:
                null_unit_projection_samples = random_null_unit_projections(
                    params, args.null_samples
                )
            if proxy_trigger is not None and args.compute_proxy_diagnostic:
                score_proxy = readout_score(
                    model,
                    images,
                    target_text,
                    class_texts,
                    target_index,
                    "proxy",
                    None,
                    proxy_trigger,
                    args.patch_location,
                    args.score_definition,
                )
                score_proxy_value = float(score_proxy.detach().cpu())
                s_proxy = torch.autograd.grad(
                    score_proxy,
                    params,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                _dot, _s_norm, _coeff, proxy_cos_tensor = project_existing_grads(
                    params, s_proxy, enabled=False
                )
                cos_proxy = float(proxy_cos_tensor.cpu())
            del s_oracle

        if args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        else:
            grad_norm = torch.tensor(0.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            record = {
                "stage": "train",
                "step": step,
                "mode": args.mode,
                "projected": projected,
                "project_trigger": args.project_trigger,
                "loss_clip": float(loss_clip.detach().cpu()),
                "score_oracle": score_oracle_value,
                "score_proxy": score_proxy_value,
                "cos_update_oracle": cos_oracle,
                "cos_update_proxy": cos_proxy,
                "dot_g_s": float(dot_before.cpu()),
                "s_norm": float(s_norm.cpu()),
                "unit_projection_neg_g_on_s": unit_projection,
                "null_unit_projection_samples": null_unit_projection_samples,
                "projection_coeff": float(coeff.cpu()),
                "random_match_scale": None if random_match_scale is None else float(random_match_scale.cpu()),
                "grad_norm": float(grad_norm.cpu()),
                "elapsed_sec": time.time() - start,
            }
            write_jsonl(jsonl_path, record)
            print(json.dumps(record), flush=True)

        if step in eval_steps:
            evaluate(step, float(loss_clip.detach().cpu()), cos_oracle, cos_proxy)

    if rows:
        asrs = [r["asr_top1"] for r in rows]
        summary = {
            "tag": args.tag,
            "mode": args.mode,
            "ckpt": args.ckpt,
            "out_dir": args.out_dir,
            "traj_csv": traj_path,
            "jsonl": jsonl_path,
            "asr_step0": rows[0]["asr_top1"],
            "asr_final": rows[-1]["asr_top1"],
            "asr_max": max(asrs),
            "rebound_delta": rows[-1]["asr_top1"] - rows[0]["asr_top1"],
            "rebound_max_delta": max(asrs) - rows[0]["asr_top1"],
            "ca_step0": rows[0]["ca_top1"],
            "ca_final": rows[-1]["ca_top1"],
            "created_at": datetime.now().isoformat(),
            "args": vars(args),
        }
    else:
        summary = {
            "tag": args.tag,
            "mode": args.mode,
            "ckpt": args.ckpt,
            "out_dir": args.out_dir,
            "jsonl": jsonl_path,
            "created_at": datetime.now().isoformat(),
            "args": vars(args),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=PAR_ALIGN_CKPT)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=["normal", "project"], default="normal")
    p.add_argument(
        "--project_trigger",
        choices=["oracle", "proxy", "proxy_shuffled", "random", "random_matched"],
        default="oracle",
    )
    p.add_argument("--project_harmful_only", action="store_true")
    p.add_argument(
        "--score_definition",
        choices=["raw", "trigger_delta", "trigger_margin_delta"],
        default="raw",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument(
        "--eval_steps",
        default=None,
        help="Optional comma-separated evaluation steps. Step 0 and final step are always evaluated.",
    )
    p.add_argument("--eval_subset", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--diagnostic_every", type=int, default=0)
    p.add_argument("--compute_proxy_diagnostic", action="store_true")
    p.add_argument("--null_samples", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit_rows", type=int, default=0)
    p.add_argument("--clean_csv", default=CC3M_CSV_STRICT)
    p.add_argument("--cc3m_root", default=CC3M_ROOT_DEFAULT)
    p.add_argument("--patch_path", default=ORACLE_PATCH_DEFAULT)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--patch_location", choices=["middle", "bottom_right"], default="middle")
    p.add_argument("--proxy_trigger_path", default=PROXY_TRIGGER)
    p.add_argument("--target_label", default="banana")
    p.add_argument("--classes_path", default=CLASSES_PATH)
    p.add_argument("--param_scope", choices=["all", "selected"], default="all")
    p.add_argument("--param_keywords", nargs="+", default=["visual.layer4", "visual.attnpool"])
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
