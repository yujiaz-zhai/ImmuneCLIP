import os
import warnings
from pathlib import Path

import clip
import torch


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint does not contain a state_dict.")

    state_dict = {}
    for name, value in checkpoint.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        state_dict[name] = value
    return state_dict


def resolve_device(model_config):
    requested = os.environ.get(
        "INVERTUNE_DEVICE", str(model_config.get("device", "auto"))
    ).strip().lower()
    if requested == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested}', but CUDA is unavailable. "
            "The current container does not expose an NVIDIA GPU. Run "
            "`nvidia-smi` and restore/attach the AutoDL GPU, or use "
            "`INVERTUNE_DEVICE=cpu` for a slow CPU-only run."
        )

    if device.type == "cpu":
        warnings.warn(
            "InverTune is running on CPU because no CUDA GPU is available. "
            "The full 50,000-image pipeline can take many hours. Restore the "
            "AutoDL GPU before a formal reproduction.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        print(
            f"Using device: {device} "
            f"({torch.cuda.get_device_name(device.index or 0)})",
            flush=True,
        )
    return device


def load_clip_model(config, checkpoint_path=None, train=False):
    model_config = config["model"]
    device = resolve_device(model_config)

    clean_model_path = Path(model_config["clean_model_path"]).expanduser()
    if not clean_model_path.is_file():
        raise FileNotFoundError(f"Clean CLIP checkpoint not found: {clean_model_path}")

    model, preprocess = clip.load(str(clean_model_path), device=device, jit=False)
    model = model.float()

    checkpoint_path = checkpoint_path or model_config.get("model_path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(
            _extract_state_dict(checkpoint), strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing keys: {missing}; "
                f"unexpected keys: {unexpected}"
            )

    model.train(train)
    return model, preprocess, device


@torch.no_grad()
def build_zeroshot_classifier(model, class_names, templates, device):
    weights = []
    for class_name in class_names:
        prompts = [template(class_name) for template in templates]
        tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        class_feature = text_features.mean(dim=0)
        class_feature = class_feature / class_feature.norm()
        weights.append(class_feature)
    return torch.stack(weights, dim=1)
