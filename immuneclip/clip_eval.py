"""CLIP RN50 零样本 CA / ASR 评估（对齐 BadCLIP 口径）。"""
from __future__ import annotations

import os
import sys
import types
from typing import Dict, Optional

import torch
from tqdm import tqdm

from config import (
    BADCLIP_ROOT,
    IMAGENET_EVAL_SUBSET,
    IMAGENET_VAL_DIR,
    PATCH_LOCATION,
    PATCH_NAME,
    PATCH_SIZE,
    PATCH_TYPE,
    TARGET_LABEL,
)

# BadCLIP 代码在仓库根目录下 import
if BADCLIP_ROOT not in sys.path:
    sys.path.insert(0, BADCLIP_ROOT)
os.chdir(BADCLIP_ROOT)


def load_clip_model(checkpoint: Optional[str], device: str, model_name: str = "RN50"):
    from pkgs.openai.clip import load as load_model

    model, processor = load_model(name=model_name, pretrained=True)
    model.to(device)
    if checkpoint and not os.path.isfile(checkpoint):
        if "clip-clean-pretrained" not in checkpoint:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint and os.path.isfile(checkpoint):
        # OpenAI 官方 RN50.pt 由 load_model(pretrained=True) 已从 cache 加载，无需再 load
        if "clip-clean-pretrained" in checkpoint:
            pass
        else:
            ckpt = torch.load(checkpoint, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                if next(iter(state_dict)).startswith("module."):
                    state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    raise RuntimeError(
                        "Checkpoint key mismatch while loading "
                        f"{checkpoint}: missing={missing}, unexpected={unexpected}"
                    )
    model.eval()
    return model, processor


def _build_eval_options(
    device: str,
    add_backdoor: bool = False,
    asr: bool = False,
    subset: Optional[int] = IMAGENET_EVAL_SUBSET,
):
    from src.data import get_eval_test_dataloader

    opts = types.SimpleNamespace(
        eval_data_type="ImageNet1K",
        eval_test_data_dir=IMAGENET_VAL_DIR,
        eval_test_data_csv=None,
        add_backdoor=add_backdoor,
        asr=asr,
        label=TARGET_LABEL,
        patch_type=PATCH_TYPE,
        patch_location=PATCH_LOCATION,
        patch_name=os.path.join(BADCLIP_ROOT, PATCH_NAME),
        patch_size=PATCH_SIZE,
        scale=None,
        blended_alpha=None,
        tigger_pth=None,
        save_files_name=None,
        backdoor_sufi=False,
        distributed=False,
        batch_size=64,
        num_workers=0,
        name="immuneclip_eval",
        device=device,
    )
    return opts


@torch.no_grad()
def eval_zeroshot(
    model,
    processor,
    device: str,
    add_backdoor: bool = False,
    asr: bool = False,
    subset: Optional[int] = IMAGENET_EVAL_SUBSET,
) -> Dict[str, float]:
    """返回 zeroshot_top1/3/5/10。"""
    from src.data import get_eval_test_dataloader

    options = _build_eval_options(device, add_backdoor, asr, subset)
    test_loader = get_eval_test_dataloader(options, processor)
    if subset is not None and subset < len(test_loader.dataset):
        # 均匀跨类采样（labels.csv 按类排序，取前 N 会只覆盖前 N/50 个类，
        # 使 CA 估计严重有偏）。用等间隔 stride 覆盖全部 1000 类。
        n = len(test_loader.dataset)
        stride = max(1, n // subset)
        indices = list(range(0, n, stride))[:subset]
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_loader.dataset, indices),
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    test_loader.num_samples = len(test_loader.dataset)

    umodel = model
    config = eval(open(f"{IMAGENET_VAL_DIR}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    target_index = None
    if asr:
        matches = [i for i, c in enumerate(classes) if c == TARGET_LABEL or TARGET_LABEL in c]
        if not matches:
            raise ValueError(f"Target label {TARGET_LABEL!r} not found in ImageNet classes")
        target_index = int(matches[0])

    text_embeddings = []
    for c in tqdm(classes, desc="text_embed", leave=False):
        text = [template(c) for template in templates]
        tokens = processor.process_text(text)
        text_input_ids = tokens["input_ids"].to(device)
        text_attention_mask = tokens["attention_mask"].to(device)
        text_embedding = umodel.get_text_features(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        )
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.mean(dim=0)
        text_embedding /= text_embedding.norm()
        text_embeddings.append(text_embedding)
    text_embeddings = torch.stack(text_embeddings, dim=1).to(device)

    topk = [1, 3, 5, 10]
    correct = {k: 0 for k in topk}
    total = 0
    for image, label in tqdm(test_loader, desc="zeroshot", leave=False):
        image, label = image.to(device), label.to(device)
        if target_index is not None:
            label = torch.full_like(label, target_index)
        image_embedding = umodel.get_image_features(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        logits = image_embedding @ text_embeddings
        ranks = logits.topk(max(topk), 1)[1].T
        predictions = ranks == label
        total += predictions.shape[1]
        for k in topk:
            correct[k] += torch.sum(torch.any(predictions[:k], dim=0)).item()

    return {f"zeroshot_top{k}": correct[k] / total for k in topk}


def eval_asr_ca(
    checkpoint: Optional[str],
    device: str = "cuda:0",
    subset: Optional[int] = IMAGENET_EVAL_SUBSET,
    model=None,
    processor=None,
) -> Dict[str, float]:
    if model is None:
        model, processor = load_clip_model(checkpoint, device)
    ca = eval_zeroshot(model, processor, device, add_backdoor=False, asr=False, subset=subset)
    asr = eval_zeroshot(model, processor, device, add_backdoor=True, asr=True, subset=subset)
    return {
        "ca_top1": ca["zeroshot_top1"],
        "ca_top5": ca["zeroshot_top5"],
        "asr_top1": asr["zeroshot_top1"],
        "asr_top5": asr["zeroshot_top5"],
    }
