import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_reference_images(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as reference:
            return [Image.fromarray(image).convert("RGB") for image in reference["x"]]
    return [Image.open(path).convert("RGB")]


class CC3MShadowDataset(Dataset):
    def __init__(self, csv_path, data_root, trigger_file, reference_file, max_samples=None):
        self.data_root = Path(data_root)
        with Path(csv_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.image_paths = [self.data_root / row["image"] for row in reader]
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        if not self.image_paths:
            raise RuntimeError(f"No CC3M records found in {csv_path}")

        with np.load(trigger_file) as trigger:
            self.trigger_patches = trigger["t"].copy()
            self.trigger_masks = trigger["tm"].copy()
        self.reference_images = _load_reference_images(reference_file)

        self.prepare = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
        ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        self.reference_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.prepare(Image.open(self.image_paths[index]).convert("RGB"))
        image_array = np.asarray(image, dtype=np.uint8)
        clean_image = self.normalize(image)

        backdoor_images = []
        for patch, mask in zip(self.trigger_patches, self.trigger_masks):
            poisoned = np.clip(image_array * mask + patch, 0, 255).astype(np.uint8)
            backdoor_images.append(self.normalize(Image.fromarray(poisoned)))

        reference_images = []
        augmented_references = []
        for reference in self.reference_images:
            prepared_reference = self.prepare(reference)
            reference_images.append(self.normalize(prepared_reference))
            augmented_references.append(
                self.reference_augmentation(prepared_reference)
            )

        return clean_image, backdoor_images, reference_images, augmented_references


def get_shadow_cc3m(args):
    dataset = CC3MShadowDataset(
        csv_path=args.shadow_csv,
        data_root=args.shadow_data_root,
        trigger_file=args.trigger_file,
        reference_file=args.reference_file,
        max_samples=args.max_shadow_samples,
    )
    print(f"Loaded {len(dataset)} CC3M shadow images")
    return dataset, None, None, None
