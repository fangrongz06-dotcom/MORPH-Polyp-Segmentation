from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train MORPH on the fixed source split")
    parser.add_argument("--config", default="configs/morph_1m.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sam_checkpoint")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader
    from models import MORPH
    from utils.common import apply_overrides, load_config, resolve_config, save_config, select_device, set_seed
    from utils.dataset import PolypDataset, check_train_test_overlap
    from utils.engine import load_checkpoint
    from utils.precision import configure_precision
    from utils.engine import Trainer
    from utils.losses import MORPHSegmentationLoss
    from utils.dataset import JointTransform

    config = resolve_config(apply_overrides(load_config(args.config), args.set),
                            args.data_root, args.seed, args.sam_checkpoint)
    execution, data, training = config["execution"], config["data"], config["train"]
    if args.device:
        execution["device"] = args.device
    device = select_device(execution["device"])
    configure_precision(execution, device)
    set_seed(config["seed"], deterministic_warn_only=execution.get("deterministic_warn_only", False))
    if int(training["batch_size"]) < 2:
        raise ValueError("CBAF BatchNorm requires a training batch size of at least 2")
    run_dir = Path(config["output_dir"]) / str(config["seed"])
    if (run_dir / "checkpoint_last.pth").exists() and not args.resume:
        raise FileExistsError("A run exists; supply --resume or a new output_dir")
    if (run_dir / "config.yaml").is_file() and load_config(run_dir / "config.yaml") != config:
        raise ValueError("Existing run configuration differs; use a new output_dir")
    augmentation = JointTransform(
        data["horizontal_flip_probability"], data["vertical_flip_probability"],
        data["rotation_degrees"], data["intensity_perturbation"],
        rotation_probability=data["rotation_probability"], intensity_probability=data["intensity_probability"],
    )
    dataset = PolypDataset(data["train_manifest"], data["root"], training=True,
                          transform=augmentation, geometry=data["geometry"])
    tests = [PolypDataset(manifest, data["root"], geometry=data["geometry"])
             for manifest in data["test_manifests"].values()]
    check_train_test_overlap(dataset, tests)
    batch_size = int(training["batch_size"])
    if len(dataset) % batch_size == 1:
        raise ValueError("The last batch has one image; choose a compatible batch size")
    workers = int(execution["num_workers"])
    loader_options = dict(batch_size=batch_size, shuffle=True, num_workers=workers,
                          pin_memory=bool(execution["pin_memory"]))
    if workers > 0:
        loader_options["prefetch_factor"] = int(execution.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_options)
    model = MORPH(config).to(device)
    criterion = MORPHSegmentationLoss(training["bce_weight"], training["dice_weight"], training["dice_smooth"])
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training["scheduler_step"],
                                               gamma=training["scheduler_gamma"])
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, restore_rng=True,
                                     expected_config=config, expected_manifest=data["train_manifest"])
        start_epoch = int(checkpoint["epoch"]) + 1
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    Trainer(model, criterion, optimizer, scheduler, config, device, run_dir).run(loader, start_epoch)


if __name__ == "__main__":
    main()
