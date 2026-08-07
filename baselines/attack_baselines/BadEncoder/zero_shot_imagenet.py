import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import clip.clip as clip
from config import CLIP_RN50_PATH


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageNetTriggeredDataset(Dataset):
    def __init__(self, root, trigger_file, max_samples=None):
        self.root = Path(root)
        with (self.root / "labels.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.records = []
            for row in reader:
                relative_path = row.get("image") or row.get("filename") or row.get("path")
                image_path = self.root / relative_path
                if not image_path.is_file():
                    image_path = self.root / "images" / relative_path
                self.records.append((image_path, int(row["label"])))
                if max_samples and len(self.records) >= max_samples:
                    break

        with np.load(trigger_file) as trigger:
            self.trigger_patch = trigger["t"][0].copy()
            self.trigger_mask = trigger["tm"][0].copy()

        self.prepare = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, label = self.records[index]
        image = self.prepare(Image.open(image_path).convert("RGB"))
        image_array = np.asarray(image, dtype=np.uint8)
        poisoned = np.clip(
            image_array * self.trigger_mask + self.trigger_patch, 0, 255
        ).astype(np.uint8)
        return self.normalize(image), self.normalize(Image.fromarray(poisoned)), label


def load_imagenet_metadata(path):
    return eval(Path(path).read_text(), {"__builtins__": {}})


def extract_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint)
    return {
        name[len("module."):] if name.startswith("module.") else name: value
        for name, value in state_dict.items()
    }


def load_backdoored_model(clean_checkpoint, backdoor_checkpoint, device):
    model, _ = clip.load(clean_checkpoint, device=device, jit=False)
    model = model.float()
    checkpoint = torch.load(backdoor_checkpoint, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    model_keys = set(model.state_dict())
    if set(state_dict).issubset(model_keys):
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            non_visual_missing = [key for key in missing if not key.startswith("visual.")]
            if non_visual_missing:
                print(
                    "Loaded a visual-only checkpoint; clean text encoder weights are retained.",
                    flush=True,
                )
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    else:
        visual_state = {
            name[len("visual."):]: value
            for name, value in state_dict.items()
            if name.startswith("visual.")
        }
        if not visual_state:
            visual_state = state_dict
        model.visual.load_state_dict(visual_state)

    model.eval()
    return model


@torch.no_grad()
def build_zeroshot_classifier(model, class_names, templates, device):
    weights = []
    for index, class_name in enumerate(class_names):
        prompts = [template(class_name) for template in templates]
        tokens = clip.tokenize(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        class_feature = features.mean(dim=0)
        weights.append(class_feature / class_feature.norm())
        if (index + 1) % 100 == 0:
            print(f"Built text weights for {index + 1}/{len(class_names)} classes", flush=True)
    return torch.stack(weights, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the InverTune BadEncoder baseline on ImageNet-1K."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--classes_file", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--clip_model", default=CLIP_RN50_PATH)
    parser.add_argument("--trigger_file", required=True)
    parser.add_argument("--target_label", default=954, type=int)
    parser.add_argument("--target_word", default="banana")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the formal ImageNet evaluation.")
    device = torch.device(f"cuda:{args.gpu}")

    dataset = ImageNetTriggeredDataset(
        args.data_root, args.trigger_file, args.max_samples
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    metadata = load_imagenet_metadata(args.classes_file)
    if metadata["classes"][args.target_label] != args.target_word:
        raise ValueError(
            f"Target mismatch: class {args.target_label} is "
            f"{metadata['classes'][args.target_label]!r}, not {args.target_word!r}."
        )

    model = load_backdoored_model(args.clip_model, args.encoder, device)
    classifier = build_zeroshot_classifier(
        model, metadata["classes"], metadata["templates"], device
    )

    total = clean_correct = clean_top5_correct = target_predictions = 0
    non_target_total = non_target_success = 0
    for clean_images, poisoned_images, labels in tqdm(loader):
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
        poisoned_predictions = (poisoned_features @ classifier).argmax(dim=-1)

        clean_correct += int((clean_logits.argmax(dim=-1) == labels).sum())
        clean_top5_correct += int(
            clean_logits.topk(5, dim=-1).indices.eq(labels[:, None]).any(dim=1).sum()
        )
        target_predictions += int((poisoned_predictions == args.target_label).sum())
        non_target = labels != args.target_label
        non_target_total += int(non_target.sum())
        non_target_success += int(
            ((poisoned_predictions == args.target_label) & non_target).sum()
        )
        total += labels.numel()

    metrics = {
        "checkpoint": str(Path(args.encoder).resolve()),
        "target_class": args.target_word,
        "target_label": args.target_label,
        "num_samples": total,
        "CA": 100.0 * clean_correct / total,
        "CA_top5": 100.0 * clean_top5_correct / total,
        "ASR": 100.0 * target_predictions / total,
        "ASR_non_target": 100.0 * non_target_success / non_target_total,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"CA={metrics['CA']:.2f}%  CA@5={metrics['CA_top5']:.2f}%  "
        f"ASR={metrics['ASR']:.2f}%  "
        f"ASR(non-target)={metrics['ASR_non_target']:.2f}%",
        flush=True,
    )
    print(f"Saved metrics to {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
