"""Official DeepWeeds CSV splits, deterministic evaluation and train augmentation."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium",
               "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _validate_frame(df, name):
    # Official subset CSVs contain only Filename/Label; Species is optional
    # metadata in labels.csv. Display names come from CLASS_NAMES.
    if not {"Filename", "Label"}.issubset(df.columns):
        raise ValueError(f"{name}: required columns are Filename, Label")
    if df.empty or df["Filename"].isna().any() or df["Filename"].duplicated().any():
        raise ValueError(f"{name}: empty split, missing or duplicate filenames")
    labels = pd.to_numeric(df["Label"], errors="raise")
    if not (labels.notna() & labels.between(0, 8) & (labels == np.floor(labels))).all():
        raise ValueError(f"{name}: labels must be integers 0..8")


def load_split(labels_dir: str | Path, fold: int = 0):
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(Path(labels_dir) / f"{split}_subset{fold}.csv")
        _validate_frame(df, split)
        frames.append(df)
    return tuple(frames)


def check_split(train_df, val_df, test_df, images_dir, *, expected_total=17509) -> dict:
    """Check metadata of all splits, including test; never decode test images.

    expected_total is overridden only in unit tests using synthetic datasets.
    Production train.run always uses the official total of 17,509.
    """
    frames = dict(zip(("train", "val", "test"), (train_df, val_df, test_df)))
    for split, df in frames.items():
        _validate_frame(df, split)
    names = {s: set(df["Filename"]) for s, df in frames.items()}
    overlaps = {f"{a}_{b}": len(names[a] & names[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    union = set.union(*names.values())
    if any(overlaps.values()):
        raise ValueError(f"Overlapping splits: {overlaps}")
    if len(union) != expected_total:
        raise ValueError(f"Union has {len(union)} images; expected {expected_total}")
    for split, df in frames.items():
        expected_ratio = 0.6 if split == "train" else 0.2
        if abs(len(df) / expected_total - expected_ratio) > 0.01:
            raise ValueError(f"{split}: split ratio differs by more than 1 percentage point")
    base = Path(images_dir).resolve()
    missing = []
    for filename in sorted(union):
        path = (base / filename).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"Image path escapes images_dir: {filename}")
        if not path.is_file():
            missing.append(filename)
    if missing:
        raise ValueError(f"Missing {len(missing)} images, examples: {missing[:5]}")
    result = {"n": {s: len(df) for s, df in frames.items()},
              "per_class": {s: [int((df.Label == c).sum()) for c in range(9)]
                            for s, df in frames.items()},
              "overlap": overlaps, "union": len(union), "missing_files": 0}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic", *,
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, interpolation="bilinear"):
    """Train: crop/flip and optional augmentation; eval: resize/center crop.

    Mean/std/interpolation can be supplied from the actual timm pretrained_cfg.
    Mixup/CutMix are batch operations in losses.py.
    """
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    interp = getattr(T.InterpolationMode, interpolation.upper())
    if train:
        ops = [T.RandomResizedCrop(img_size, interpolation=interp), T.RandomHorizontalFlip()]
        if aug == "color":
            ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        elif aug == "trivial":
            ops.append(T.TrivialAugmentWide(interpolation=interp))
        elif aug == "randaug":
            ops.append(T.RandAugment(interpolation=interp))
        elif aug != "basic":
            raise ValueError(f"Unknown augmentation: {aug}")
    else:
        ops = [T.Resize(round(img_size * 256 / 224), interpolation=interp), T.CenterCrop(img_size)]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir, self.transform = Path(images_dir), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.images_dir / row.Filename) as image:
            image = image.convert("RGB")
            image = self.transform(image) if self.transform else image.copy()
        return image, int(row.Label), str(row.Filename)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df, images_dir, transform, batch_size, train, sampler=None,
                num_workers=2, *, seed=0):
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if sampler not in (None, "balanced") or (sampler and not train):
        raise ValueError("Only train may use the balanced sampler")
    generator = torch.Generator().manual_seed(seed)
    weights_sampler = None
    if sampler == "balanced":
        counts = df.Label.value_counts()
        weights = [1.0 / counts[int(label)] for label in df.Label]
        weights_sampler = WeightedRandomSampler(weights, len(df), replacement=True,
                                                generator=generator)
    return DataLoader(DeepWeedsDataset(df, images_dir, transform), batch_size=batch_size,
                      shuffle=train and weights_sampler is None, sampler=weights_sampler,
                      drop_last=train and len(df) >= batch_size,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=generator)
