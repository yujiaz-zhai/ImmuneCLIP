import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from data.imagenet import build_defense_loaders, load_imagenet_metadata
from models.invertune_badclip import apply_inverted_trigger, load_inverted_trigger
from utils.clip_model import build_zeroshot_classifier, load_clip_model
from utils.repro import cpu_state_dict, load_config, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-faithful activation tuning.")
    parser.add_argument("--config", default="config/badclip_banana_paper.yaml")
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def diverse_clean_batch(dataset, count, target_label, seed):
    candidates = np.flatnonzero(np.asarray(dataset.targets) != target_label)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(candidates, size=count, replace=False))
    samples = [dataset[int(index)] for index in indices]
    return torch.stack([sample[0] for sample in samples]), torch.tensor(
        [sample[1] for sample in samples]
    )


def capture_activations(model, images, layer_names):
    modules = dict(model.named_modules())
    outputs = {}
    hooks = []
    for name in layer_names:
        if name not in modules:
            raise KeyError(f"Model does not contain layer {name}")
        hooks.append(
            modules[name].register_forward_hook(
                lambda _module, _inputs, output, key=name: outputs.__setitem__(
                    key, output
                )
            )
        )
    features = model.encode_image(images)
    for hook in hooks:
        hook.remove()
    return features, outputs


@torch.no_grad()
def identify_critical_neurons(model, clean_images, triggered_images, layer_names):
    _, clean_outputs = capture_activations(model, clean_images, layer_names)
    _, trigger_outputs = capture_activations(model, triggered_images, layer_names)
    differences = {}
    layer_scores = {}
    for name in layer_names:
        clean_mean = clean_outputs[name].float().mean(dim=0)
        trigger_mean = trigger_outputs[name].float().mean(dim=0)
        difference = (clean_mean - trigger_mean).abs()
        differences[name] = difference
        layer_scores[name] = float(
            torch.norm(clean_mean - trigger_mean, p=2)
            / torch.norm(clean_mean, p=2).clamp_min(1e-8)
        )

    scores = np.asarray(list(layer_scores.values()))
    threshold = float(scores.mean() + scores.std())
    critical_layers = [
        name for name in layer_names if layer_scores[name] > threshold
    ]
    if not critical_layers:
        critical_layers = [max(layer_scores, key=layer_scores.get)]

    spatial_masks = {}
    channel_masks = {}
    for name in critical_layers:
        shape = differences[name].shape
        values = differences[name].cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(values)
        cluster_means = [
            values[kmeans.labels_ == cluster].mean() for cluster in (0, 1)
        ]
        critical_cluster = int(np.argmax(cluster_means))
        spatial = torch.from_numpy(
            (kmeans.labels_ == critical_cluster).reshape(shape)
        ).to(clean_images.device)
        channel = spatial.flatten(start_dim=1).any(dim=1)
        spatial_masks[name] = spatial
        channel_masks[name] = channel

    analysis = {
        "layer_scores": layer_scores,
        "threshold": threshold,
        "critical_layers": critical_layers,
        "critical_spatial_neurons": {
            name: int(mask.sum()) for name, mask in spatial_masks.items()
        },
        "total_spatial_neurons": {
            name: int(mask.numel()) for name, mask in spatial_masks.items()
        },
        "critical_channels": {
            name: int(mask.sum()) for name, mask in channel_masks.items()
        },
        "total_channels": {
            name: int(mask.numel()) for name, mask in channel_masks.items()
        },
    }
    return spatial_masks, channel_masks, analysis


def configure_parameter_masks(model, channel_masks):
    parameter_masks = {}
    selected_parameters = []
    selected_names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        for layer_name, channel_mask in channel_masks.items():
            if not name.startswith(f"{layer_name}.") or parameter.ndim == 0:
                continue
            critical_indices = torch.where(channel_mask)[0]
            valid_indices = critical_indices[critical_indices < parameter.shape[0]]
            if valid_indices.numel() == 0:
                break
            mask = torch.zeros_like(parameter)
            mask[valid_indices] = 1
            parameter.requires_grad_(True)
            parameter_masks[name] = mask
            selected_parameters.append(parameter)
            selected_names.append(name)
            break
    if not selected_parameters:
        raise RuntimeError("No model parameters matched the critical neurons.")
    return parameter_masks, selected_parameters, selected_names


@torch.no_grad()
def inverted_asr(model, images, classifier, mask, trigger, target_label):
    features = model.encode_image(apply_inverted_trigger(images, mask, trigger))
    features = features / features.norm(dim=-1, keepdim=True)
    predictions = (features @ classifier).argmax(dim=-1)
    return float(predictions.eq(target_label).float().mean() * 100)


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 123))
    seed_everything(seed)
    model, preprocess, device = load_clip_model(config)
    original_model, _, _ = load_clip_model(config)
    original_model.eval()
    for parameter in original_model.parameters():
        parameter.requires_grad_(False)

    dataset, _, _ = build_defense_loaders(config, preprocess)
    target_label = int(config["data"]["target_label"])
    clean_images, _ = diverse_clean_batch(
        dataset,
        int(config["data"]["batch_size"]),
        target_label,
        seed + 2,
    )
    clean_images = clean_images.to(device)

    trigger_path = Path(config["inversion"]["trigger_path"])
    checkpoint = torch.load(trigger_path, map_location="cpu")
    trigger_asr = float(checkpoint.get("diagnostics", {}).get("inverted_ASR", -1))
    minimum_asr = float(config["inversion"].get("minimum_asr", 70.0))
    if trigger_asr < minimum_asr:
        raise RuntimeError(
            f"Refusing to tune with an ineffective inverted trigger: "
            f"ASR={trigger_asr:.2f}%, required={minimum_asr:.2f}%."
        )
    mask, trigger = load_inverted_trigger(trigger_path, device)
    triggered_images = apply_inverted_trigger(clean_images, mask, trigger)

    layer_names = config["tuning"]["candidate_layers"]
    spatial_masks, channel_masks, analysis = identify_critical_neurons(
        model, clean_images, triggered_images, layer_names
    )
    parameter_masks, selected_parameters, selected_names = (
        configure_parameter_masks(model, channel_masks)
    )
    analysis["updated_parameters"] = selected_names
    analysis["inverted_trigger_ASR_before_tuning"] = trigger_asr

    optimizer = torch.optim.AdamW(
        selected_parameters,
        lr=float(config["tuning"]["learning_rate"]),
        weight_decay=float(config["tuning"].get("weight_decay", 1e-4)),
    )
    metadata = load_imagenet_metadata()
    classifier = build_zeroshot_classifier(
        original_model, metadata["classes"], metadata["templates"], device
    )
    with torch.no_grad():
        original_features = original_model.encode_image(clean_images)
        original_features = original_features / original_features.norm(
            dim=-1, keepdim=True
        )

    output_dir = Path(config["output"]["root"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "critical_neurons.json", analysis)
    print(f"critical layers: {analysis['critical_layers']}", flush=True)
    print(f"critical channels: {analysis['critical_channels']}", flush=True)
    print(f"masked parameter tensors: {len(selected_names)}", flush=True)

    epochs = args.epochs or int(config["tuning"]["epochs"])
    beta = float(config["tuning"]["beta"])
    checkpoint_interval = int(config["tuning"].get("checkpoint_interval", 50))
    history = []
    model.eval()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        clean_features, clean_outputs = capture_activations(
            model, clean_images, analysis["critical_layers"]
        )
        _, trigger_outputs = capture_activations(
            model, triggered_images, analysis["critical_layers"]
        )

        alignment = clean_features.new_zeros(())
        for name, spatial_mask in spatial_masks.items():
            difference = trigger_outputs[name] - clean_outputs[name]
            alignment = alignment + difference[:, spatial_mask].square().mean()

        normalized_clean = clean_features / clean_features.norm(
            dim=-1, keepdim=True
        )
        preservation = torch.norm(normalized_clean - original_features, p=2)
        total = alignment + beta * preservation
        total.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(parameter_masks[name])
        torch.nn.utils.clip_grad_norm_(selected_parameters, 1.0)
        optimizer.step()

        record = {
            "epoch": epoch,
            "total": float(total.detach()),
            "alignment": float(alignment.detach()),
            "preservation": float(preservation.detach()),
        }
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            record["inverted_ASR"] = inverted_asr(
                model,
                clean_images,
                classifier,
                mask,
                trigger,
                target_label,
            )
            print(
                f"epoch={epoch}/{epochs} loss={record['total']:.6f} "
                f"alignment={record['alignment']:.6f} "
                f"preservation={record['preservation']:.6f} "
                f"inverted_ASR={record['inverted_ASR']:.2f}%",
                flush=True,
            )
        history.append(record)
        save_json(output_dir / "tuning_history.json", history)

        if epoch % checkpoint_interval == 0 and epoch != epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": cpu_state_dict(model),
                    "critical_neurons": analysis,
                },
                checkpoint_dir / f"epoch_{epoch}.pt",
            )

    defended_path = checkpoint_dir / "defended_model.pt"
    torch.save(
        {
            "epoch": epochs,
            "state_dict": cpu_state_dict(model),
            "critical_neurons": analysis,
            "source_checkpoint": config["model"]["model_path"],
        },
        defended_path,
    )
    print(f"saved defended model: {defended_path}", flush=True)


if __name__ == "__main__":
    main()
