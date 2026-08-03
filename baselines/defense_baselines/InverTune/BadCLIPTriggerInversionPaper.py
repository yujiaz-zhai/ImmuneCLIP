import argparse
from pathlib import Path

import numpy as np
import torch

from data.imagenet import build_defense_loaders, load_imagenet_metadata
from models.paper_inversion import PaperBackdoorInversion
from utils.clip_model import build_zeroshot_classifier, load_clip_model
from utils.repro import load_config, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-faithful BadCLIP inversion.")
    parser.add_argument("--config", default="config/badclip_banana_paper.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def diverse_batch(dataset, count, target_label, seed):
    candidates = np.flatnonzero(np.asarray(dataset.targets) != target_label)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(candidates, size=count, replace=False))
    images = [dataset[int(index)][0] for index in indices]
    return torch.stack(images)


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 123))
    seed_everything(seed)
    model, preprocess, device = load_clip_model(config)
    dataset, target_loader, non_target_loader = build_defense_loaders(
        config, preprocess
    )

    metadata = load_imagenet_metadata()
    classifier = build_zeroshot_classifier(
        model, metadata["classes"], metadata["templates"], device
    )
    target_label = int(config["data"]["target_label"])
    inversion = PaperBackdoorInversion(
        model=model,
        text_classifier=classifier,
        target_label=target_label,
        weights=config["optimizer"],
        learning_rate=float(config["inversion"]["learning_rate"]),
        temperature=float(config["inversion"].get("temperature", 0.07)),
    )
    target_images, _ = next(iter(target_loader))
    target_feature = inversion.encode_target(target_images.to(device))
    diagnostic_images = diverse_batch(
        dataset,
        int(config["inversion"].get("diagnostic_samples", 512)),
        target_label,
        seed + 1,
    ).to(device)

    epochs = args.epochs or int(config["inversion"]["epochs"])
    max_batches = args.max_batches or int(config["inversion"]["max_batches"])
    output_dir = Path(config["output"]["root"]) / "inversion"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_asr = -1.0
    best_mask_l1 = float("inf")

    for epoch in range(1, epochs + 1):
        totals = {}
        batches = 0
        for batch_index, (clean_images, _) in enumerate(non_target_loader):
            if batch_index >= max_batches:
                break
            losses = inversion.optimize(
                clean_images.to(device, non_blocking=True), target_feature
            )
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1

        averages = {name: value / batches for name, value in totals.items()}
        inverted_asr = inversion.attack_success_rate(diagnostic_images)
        mask_l1 = float(inversion.mask.detach().abs().sum())
        mask_active = int((inversion.mask.detach() > 0.1).sum())
        record = {
            "epoch": epoch,
            "batches": batches,
            **averages,
            "inverted_ASR": inverted_asr,
            "mask_l1_end": mask_l1,
            "mask_active_gt_0.1": mask_active,
        }
        history.append(record)
        checkpoint = inversion.export_state(record)
        checkpoint["epoch"] = epoch
        torch.save(checkpoint, output_dir / f"epoch_{epoch}.pth")
        if inverted_asr > best_asr or (
            inverted_asr == best_asr and mask_l1 < best_mask_l1
        ):
            best_asr = inverted_asr
            best_mask_l1 = mask_l1
            torch.save(checkpoint, output_dir / "best.pth")
        save_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch}/{epochs} loss={record['total']:.4f} "
            f"CE={record['alignment']:.4f} mask_l1={mask_l1:.2f} "
            f"inverted_ASR={inverted_asr:.2f}%",
            flush=True,
        )

    best = torch.load(output_dir / "best.pth", map_location="cpu")
    torch.save(best, output_dir / "latest.pth")
    minimum_asr = float(config["inversion"].get("minimum_asr", 70.0))
    if best_asr < minimum_asr:
        raise RuntimeError(
            f"Trigger inversion failed: best ASR is {best_asr:.2f}%, "
            f"below the required {minimum_asr:.2f}%. Do not run activation tuning."
        )
    print(
        f"accepted inverted trigger: ASR={best_asr:.2f}% "
        f"path={output_dir / 'latest.pth'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
