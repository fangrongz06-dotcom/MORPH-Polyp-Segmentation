from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    parent = config.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        merge_config(base, config)
        return base
    return config


def merge_config(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_config(base[key], value)
        else:
            base[key] = value


def apply_overrides(config, overrides):
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("Overrides must use key=value")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    return config


def resolve_config(config, data_root, seed=None, sam_checkpoint=None):
    config["data"]["root"] = str(Path(data_root).expanduser().resolve())
    if seed is not None:
        config["seed"] = int(seed)
    if sam_checkpoint:
        config["model"]["sam_checkpoint"] = sam_checkpoint
    def local_path(value):
        path = Path(value).expanduser()
        return str(path.resolve() if path.is_absolute() else REPOSITORY_ROOT / path)
    config["model"]["sam_checkpoint"] = local_path(config["model"]["sam_checkpoint"])
    config["data"]["train_manifest"] = local_path(config["data"]["train_manifest"])
    config["data"]["test_manifests"] = {
        name: local_path(value) for name, value in config["data"]["test_manifests"].items()
    }
    config["output_dir"] = local_path(config["output_dir"])
    return config


def set_seed(seed, *, deterministic_warn_only=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)


def save_config(config, path):
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_code_hash():
    digest = hashlib.sha256()
    for directory in ("models", "utils"):
        for path in sorted((REPOSITORY_ROOT / directory).rglob("*.py")):
            digest.update(path.relative_to(REPOSITORY_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    for name in ("train.py", "test.py"):
        digest.update((REPOSITORY_ROOT / name).read_bytes())
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def select_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def sample_seed(base_seed, dataset, image_id):
    identity = f"{int(base_seed)}\0{dataset}\0{image_id}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
