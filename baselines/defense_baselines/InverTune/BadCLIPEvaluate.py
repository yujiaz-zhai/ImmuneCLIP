import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.imagenet import (
    BadCLIPTrigger,
    PairedTriggeredDataset,
    build_imagenet_dataset,
    load_imagenet_metadata,
)
from utils.clip_model import build_zeroshot_classifier, load_clip_model
from utils.repro import load_config, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate clean accuracy and BadCLIP attack success rate."
    )
    parser.add_argument("--config", default="config/badclip_banana.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 123)))
    checkpoint = args.checkpoint or config["evaluation"]["checkpoint"]
    model, preprocess, device = load_clip_model(
        config, checkpoint_path=checkpoint, train=False
    )

    base_dataset = build_imagenet_dataset(
        config["data"]["data_root"], transform=None, max_samples=args.max_samples
    )
    trigger = BadCLIPTrigger(
        patch_path=config["attack"]["patch_path"],
        patch_size=int(config["attack"]["patch_size"]),
        location=config["attack"]["patch_location"],
    )
    dataset = PairedTriggeredDataset(base_dataset, preprocess, trigger)
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"].get("batch_size", 256)),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    metadata = load_imagenet_metadata()
    classifier = build_zeroshot_classifier(
        model, metadata["classes"], metadata["templates"], device
    )
    target_label = int(config["data"]["target_label"])
    total = 0
    clean_correct = 0
    clean_top5_correct = 0
    target_predictions = 0
    non_target_total = 0
    non_target_success = 0

    for batch_index, (clean_images, poisoned_images, labels) in enumerate(loader):
        clean_images = clean_images.to(device, non_blocking=True)
        poisoned_images = poisoned_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clean_features = model.encode_image(clean_images)
        poisoned_features = model.encode_image(poisoned_images)
        clean_features = clean_features / clean_features.norm(dim=-1, keepdim=True)
        poisoned_features = poisoned_features / poisoned_features.norm(
            dim=-1, keepdim=True
        )
        clean_logits = clean_features @ classifier
        poisoned_logits = poisoned_features @ classifier

        clean_predictions = clean_logits.argmax(dim=-1)
        poisoned_predictions = poisoned_logits.argmax(dim=-1)
        clean_correct += int((clean_predictions == labels).sum())
        clean_top5_correct += int(
            clean_logits.topk(5, dim=-1).indices.eq(labels[:, None]).any(dim=1).sum()
        )
        target_predictions += int((poisoned_predictions == target_label).sum())
        non_target = labels != target_label
        non_target_total += int(non_target.sum())
        non_target_success += int(
            ((poisoned_predictions == target_label) & non_target).sum()
        )
        total += labels.numel()
        if batch_index % 25 == 0:
            print(f"evaluated {total}/{len(dataset)} images", flush=True)

    metrics = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "target_class": config["data"]["target_word"],
        "target_label": target_label,
        "num_samples": total,
        "CA": 100.0 * clean_correct / total,
        "CA_top5": 100.0 * clean_top5_correct / total,
        "ASR": 100.0 * target_predictions / total,
        "ASR_non_target": 100.0 * non_target_success / non_target_total,
    }
    output = args.output or config["evaluation"]["output"]
    save_json(output, metrics)
    print(
        f"CA={metrics['CA']:.2f}%  ASR={metrics['ASR']:.2f}%  "
        f"ASR(non-target)={metrics['ASR_non_target']:.2f}%",
        flush=True,
    )
    print(f"saved metrics: {Path(output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
