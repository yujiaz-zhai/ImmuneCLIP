import bisect
import csv
import io
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


def load_imagenet_metadata():
    namespace = {}
    classes_path = Path(__file__).with_name("classes.py")
    exec(compile(classes_path.read_text(), str(classes_path), "exec"), namespace)
    return eval(classes_path.read_text(), {"__builtins__": {}})


class ParquetImageNetDataset(Dataset):
    def __init__(self, root, transform=None, max_samples=None):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading the supplied ImageNet dataset requires pyarrow."
            ) from exc

        self.pq = pq
        self.root = Path(root)
        self.transform = transform
        self.files = sorted((self.root / "data").glob("*.parquet"))
        if not self.files:
            self.files = sorted(self.root.glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No parquet shards found below {self.root}")

        lengths = [self.pq.ParquetFile(path).metadata.num_rows for path in self.files]
        self.offsets = []
        total = 0
        for length in lengths:
            total += length
            self.offsets.append(total)
        self.size = min(total, max_samples) if max_samples else total

        labels = []
        remaining = self.size
        for path, length in zip(self.files, lengths):
            if remaining <= 0:
                break
            count = min(length, remaining)
            table = self.pq.read_table(path, columns=["label"])
            labels.extend(table.column("label")[:count].to_pylist())
            remaining -= count
        self.targets = labels

        self._cached_shard = None
        self._cached_table = None

    def __len__(self):
        return self.size

    def _locate(self, index):
        if index < 0:
            index += self.size
        if index < 0 or index >= self.size:
            raise IndexError(index)
        shard = bisect.bisect_right(self.offsets, index)
        start = self.offsets[shard - 1] if shard else 0
        return shard, index - start

    def get_raw(self, index):
        shard, row = self._locate(index)
        if shard != self._cached_shard:
            self._cached_table = self.pq.read_table(
                self.files[shard], columns=["image", "label"]
            )
            self._cached_shard = shard

        image_record = self._cached_table.column("image")[row].as_py()
        label = int(self._cached_table.column("label")[row].as_py())
        if image_record.get("bytes") is not None:
            image = Image.open(io.BytesIO(image_record["bytes"]))
        elif image_record.get("path"):
            image = Image.open(image_record["path"])
        else:
            raise ValueError(f"Image {index} has neither bytes nor a path.")
        return image.convert("RGB"), label

    def __getitem__(self, index):
        image, label = self.get_raw(index)
        if self.transform:
            image = self.transform(image)
        return image, label


class FlatImageNetDataset(Dataset):
    def __init__(self, root, transform=None, max_samples=None):
        self.root = Path(root)
        self.transform = transform
        csv_path = self.root / "labels.csv"
        images_root = self.root / "images"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing labels file: {csv_path}")

        records = []
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = (
                    row.get("filename")
                    or row.get("image")
                    or row.get("path")
                    or next(iter(row.values()))
                )
                label_value = (
                    row.get("label")
                    or row.get("target")
                    or list(row.values())[-1]
                )
                records.append((images_root / filename, int(label_value)))
                if max_samples and len(records) >= max_samples:
                    break
        self.records = records
        self.targets = [label for _, label in records]

    def __len__(self):
        return len(self.records)

    def get_raw(self, index):
        path, label = self.records[index]
        return Image.open(path).convert("RGB"), label

    def __getitem__(self, index):
        image, label = self.get_raw(index)
        if self.transform:
            image = self.transform(image)
        return image, label


def build_imagenet_dataset(root, transform=None, max_samples=None):
    root = Path(root)
    if list((root / "data").glob("*.parquet")) or list(root.glob("*.parquet")):
        return ParquetImageNetDataset(root, transform, max_samples)
    if (root / "labels.csv").is_file():
        return FlatImageNetDataset(root, transform, max_samples)
    raise ValueError(
        f"Unsupported ImageNet layout at {root}. Expected parquet shards or "
        "images/ plus labels.csv."
    )


def build_defense_loaders(config, preprocess, max_samples=None):
    data_config = config["data"]
    dataset = build_imagenet_dataset(
        data_config["data_root"], preprocess, max_samples
    )
    target_label = int(data_config["target_label"])
    target_indices = [
        index for index, label in enumerate(dataset.targets) if label == target_label
    ]
    non_target_indices = [
        index for index, label in enumerate(dataset.targets) if label != target_label
    ]
    if not target_indices:
        raise RuntimeError(f"No samples found for target label {target_label}.")

    common = {
        "num_workers": int(data_config.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    batch_size = int(data_config.get("batch_size", 64))
    target_loader = DataLoader(
        Subset(dataset, target_indices), batch_size=batch_size, shuffle=False, **common
    )
    non_target_loader = DataLoader(
        Subset(dataset, non_target_indices),
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    return dataset, target_loader, non_target_loader


class BadCLIPTrigger:
    def __init__(self, patch_path, patch_size=16, location="middle"):
        self.patch = Image.open(patch_path).convert("RGB")
        self.patch_size = int(patch_size)
        self.location = location

    def __call__(self, image):
        image = image.convert("RGB").resize((224, 224))
        patch = self.patch.resize((self.patch_size, self.patch_size))
        if self.location != "middle":
            raise ValueError("This reproduction supports the BadCLIP middle trigger.")
        start = (224 - self.patch_size) // 2
        image.paste(patch, (start, start))
        return image


class PairedTriggeredDataset(Dataset):
    def __init__(self, base_dataset, preprocess, trigger):
        self.base_dataset = base_dataset
        self.preprocess = preprocess
        self.trigger = trigger

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset.get_raw(index)
        clean = self.preprocess(image)
        poisoned = self.preprocess(self.trigger(image.copy()))
        return clean, poisoned, label
