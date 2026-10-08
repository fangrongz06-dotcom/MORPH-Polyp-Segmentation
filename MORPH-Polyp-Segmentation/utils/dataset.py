from __future__ import annotations

import random
import math
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from utils.prompts import Click
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class JointTransform:
    """Synchronized image, mask and click augmentation."""

    def __init__(self, horizontal_flip_probability, vertical_flip_probability, rotation_degrees, intensity_perturbation,
                 rotation_probability=1.0, intensity_probability=1.0):
        values = (horizontal_flip_probability, vertical_flip_probability, rotation_degrees, intensity_perturbation)
        if any(value is None for value in values):
            raise ValueError("Augmentation parameters must be specified")
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)
        self.rotation_probability = float(rotation_probability)
        self.intensity_probability = float(intensity_probability)
        if not 0 <= self.rotation_probability <= 1 or not 0 <= self.intensity_probability <= 1:
            raise ValueError("rotation/intensity probabilities must be in [0,1]")
        self.rotation_range = tuple(map(float, rotation_degrees)) if isinstance(rotation_degrees, (list, tuple)) else (-float(rotation_degrees), float(rotation_degrees))
        if len(self.rotation_range) != 2 or self.rotation_range[0] > self.rotation_range[1]:
            raise ValueError("rotation_degrees must be a nonnegative magnitude or ordered [min,max]")
        self.jitter = ColorJitter(
            brightness=float(intensity_perturbation), contrast=float(intensity_perturbation)
        )

    def __call__(self, image, mask, clicks=None):
        width, height = image.size
        points = list(clicks) if clicks is not None else []
        if random.random() < self.horizontal_flip_probability:
            image, mask = TF.hflip(image), TF.hflip(mask)
            points = [Click(p.y, width - 1 - p.x, p.positive) for p in points]
        if random.random() < self.vertical_flip_probability:
            image, mask = TF.vflip(image), TF.vflip(mask)
            points = [Click(height - 1 - p.y, p.x, p.positive) for p in points]
        rotate = self.rotation_probability > 0 and (self.rotation_probability == 1 or random.random() < self.rotation_probability)
        angle = random.uniform(*self.rotation_range) if rotate else 0.0
        if rotate:
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        radians = math.radians(angle); sine, cosine = math.sin(radians), math.cos(radians)
        cx, cy = (width - 1) / 2, (height - 1) / 2
        transformed = []
        for point in points:
            x, y = point.x - cx, point.y - cy
            rx, ry = round(cosine * x + sine * y + cx), round(-sine * x + cosine * y + cy)
            if 0 <= rx < width and 0 <= ry < height:
                transformed.append(Click(ry, rx, point.positive))
        jitter = self.intensity_probability > 0 and (self.intensity_probability == 1 or random.random() < self.intensity_probability)
        result = self.jitter(image) if jitter else image, mask
        return (*result, transformed) if clicks is not None else result


TEST_DIRECTORIES = {
    "kvasir": "Kvasir", "clinicdb": "CVC-ClinicDB", "colondb": "CVC-ColonDB",
    "etis": "ETIS-LaribPolypDB", "endoscene": "CVC-300", "bkai": "BKAI",
}


def read_split(manifest, data_root, training=False):
    root = Path(data_root).resolve()
    with Path(manifest).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty split: {Path(manifest).name}")
    records, identities = [], set()
    for row in rows:
        dataset, filename = row["dataset"], row["filename"]
        if Path(filename).name != filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError("Split filenames must be basenames")
        identity = (dataset, filename)
        if identity in identities:
            raise ValueError(f"Duplicate split entry: {identity}")
        identities.add(identity)
        if training:
            if dataset not in {"kvasir", "clinicdb"}:
                raise ValueError("Training uses Kvasir-SEG and CVC-ClinicDB only")
            directory, image_folder = root / "TrainDataset", "image"
        else:
            directory = root / "TestDataset" / TEST_DIRECTORIES[dataset]
            image_folder = "images"
        image_path, mask_path = directory / image_folder / filename, directory / "masks" / filename
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image/mask for {dataset}/{filename}")
        records.append({"dataset": dataset, "image_id": Path(filename).stem,
                        "image_path": str(image_path), "mask_path": str(mask_path)})
    return records


class PolypDataset(Dataset):
    def __init__(self, manifest, data_root, training=False, transform=None, geometry=1024):
        self.records = read_split(manifest, data_root, training)
        self.transform, self.geometry = transform, geometry

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB").resize((self.geometry, self.geometry), Image.Resampling.BILINEAR)
        with Image.open(record["mask_path"]) as source:
            mask = source.convert("L").resize((self.geometry, self.geometry), Image.Resampling.NEAREST)
        if self.transform:
            image, mask = self.transform(image, mask)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
        mask = torch.from_numpy((np.asarray(mask).copy() > 0).astype(np.float32))[None]
        return {"image": image, "mask": mask, **record}


def check_train_test_overlap(training, tests):
    source = {(row["dataset"], row["image_id"]) for row in training.records}
    for dataset in tests:
        overlap = source & {(row["dataset"], row["image_id"]) for row in dataset.records}
        if overlap:
            raise ValueError(f"Train/test membership overlap: {sorted(overlap)[:3]}")
