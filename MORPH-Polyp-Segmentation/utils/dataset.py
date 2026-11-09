from __future__ import annotations

import random
import math
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
        self.intensity_magnitude = float(intensity_perturbation)
        if not 0 <= self.intensity_magnitude <= 1:
            raise ValueError("intensity_perturbation must be in [0,1]")

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
        # Brightness and contrast have independent Bernoulli draws (Sec. 4.2).
        for adjust in (TF.adjust_brightness, TF.adjust_contrast):
            if random.random() < self.intensity_probability:
                image = adjust(image, random.uniform(1-self.intensity_magnitude, 1+self.intensity_magnitude))
        result = image, mask
        return (*result, transformed) if clicks is not None else result


TEST_DIRECTORIES = {
    "kvasir": "Kvasir", "clinicdb": "CVC-ClinicDB", "colondb": "CVC-ColonDB",
    "etis": "ETIS-LaribPolypDB", "endoscene": "CVC-300", "bkai": "BKAI",
}


def training_directory(data_root):
    folder = Path(data_root) / "TrainDataset"
    return folder if (folder/"image").is_dir() else folder/"TrainDataset"


def read_split(manifest, data_root, training=False, source_partition=False):
    root = Path(data_root).resolve()
    with Path(manifest).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"dataset", "filename"} <= set(reader.fieldnames or []):
            raise ValueError("A split requires dataset,filename columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty split: {Path(manifest).name}")
    records, identities = [], set()
    for row in rows:
        dataset, filename = row["dataset"], row["filename"]
        if Path(filename).name != filename or "\\" in filename or filename in {".", ".."}:
            raise ValueError("Split filenames must be basenames")
        mask_filename = row.get("mask_filename") or filename
        if Path(mask_filename).name != mask_filename or "\\" in mask_filename:
            raise ValueError("Mask filenames must be basenames")
        identity = (dataset, Path(filename).stem)
        if identity in identities:
            raise ValueError(f"Duplicate split entry: {identity}")
        identities.add(identity)
        if training or source_partition:
            if dataset not in {"kvasir", "clinicdb"}:
                raise ValueError("Training uses Kvasir-SEG and CVC-ClinicDB only")
            directory, image_folder = training_directory(root), "image"
        else:
            if dataset not in TEST_DIRECTORIES:
                raise ValueError(f"Unknown dataset: {dataset}")
            directory = root / "TestDataset" / TEST_DIRECTORIES[dataset]
            image_folder = "images"
        image_path, mask_path = directory / image_folder / filename, directory / "masks" / mask_filename
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image/mask for {dataset}/{filename}")
        records.append({"dataset": dataset, "image_id": Path(filename).stem,
                        "image_path": str(image_path), "mask_path": str(mask_path)})
    return records


IMAGENET_MEAN = (0.485, 0.456, 0.406)


def read_binary_mask(source, dataset):
    if dataset == "bkai":
        rgb = np.asarray(source.convert("RGB"))
        if source.format not in {"PNG", "TIFF", "BMP"} or not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 1], rgb[..., 2]):
            raise ValueError("BKAI requires explicitly prepared lossless binary GT; raw color JPEG labels are not binary masks")
        if not set(np.unique(rgb[..., 0]).tolist()) <= {0, 1, 255}:
            raise ValueError("Prepared BKAI GT values must be 0/1 or 0/255")
    return Image.fromarray((np.asarray(source.convert("L")) > 0).astype(np.uint8)*255)


def resize_and_pad(image, mask, geometry=1024):
    """Resize the longest side, then pad raw RGB with ImageNet means."""
    width, height = image.size
    scale = int(geometry) / max(height, width)
    resized_h, resized_w = max(1, int(height*scale+0.5)), max(1, int(width*scale+0.5))
    image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    mask = mask.resize((resized_w, resized_h), Image.Resampling.NEAREST)
    image_tensor = torch.tensor(IMAGENET_MEAN).view(3,1,1).expand(3,geometry,geometry).clone()
    image_tensor[:, :resized_h, :resized_w] = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2,0,1)/255
    target = torch.zeros(1,geometry,geometry)
    target[:, :resized_h, :resized_w] = torch.from_numpy((np.asarray(mask)>0).astype(np.float32))[None]
    valid = torch.zeros(1,geometry,geometry,dtype=torch.bool)
    valid[:, :resized_h, :resized_w] = True
    return image_tensor, target, valid, (resized_h, resized_w)


class PolypDataset(Dataset):
    def __init__(self, manifest, data_root, training=False, transform=None, geometry=1024, source_partition=False):
        self.records = read_split(manifest, data_root, training, source_partition)
        self.transform, self.geometry = transform, geometry

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        original_size = (image.height, image.width)
        with Image.open(record["mask_path"]) as source:
            if source.size != image.size:
                raise ValueError("Image and GT dimensions differ")
            mask = read_binary_mask(source, record["dataset"])
        if self.transform:
            image, mask = self.transform(image, mask)
        image, mask, valid, resized_size = resize_and_pad(image, mask, self.geometry)
        return {"image": image, "mask": mask, "valid_mask": valid,
                "original_size": torch.tensor(original_size), "resized_size": torch.tensor(resized_size), **record}


def check_train_test_overlap(training, tests):
    tests = list(tests)
    source = {(row["dataset"], row["image_id"]) for row in training.records}
    for dataset in tests:
        overlap = source & {(row["dataset"], row["image_id"]) for row in dataset.records}
        if overlap:
            raise ValueError(f"Train/test membership overlap: {sorted(overlap)[:3]}")
    from utils.content import decoded_hash
    hashes = {decoded_hash(row["image_path"]) for row in training.records}
    for dataset in tests:
        if any(decoded_hash(row["image_path"]) in hashes for row in dataset.records):
            raise ValueError("Training/evaluation decoded-RGB content overlap")
