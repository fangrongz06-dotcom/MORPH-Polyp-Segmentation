from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MORPH with 0, 1 or 2 clicks")
    parser.add_argument("--config", default="configs/morph_0m.yaml")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam_checkpoint")
    parser.add_argument("--protocol", choices=["0-m", "1-m", "2-m"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output")
    parser.add_argument("--bkai_split", help="CSV for strict BKAI zero-shot evaluation")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from models import MORPH
    from utils.common import apply_overrides, load_config, resolve_config, sample_seed, select_device, set_seed
    from utils.dataset import PolypDataset, check_train_test_overlap
    from utils.engine import evaluate_sample
    from utils.metrics import aggregate_rows

    config = resolve_config(apply_overrides(load_config(args.config), args.set),
                            args.data_root, sam_checkpoint=args.sam_checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not {"model", "config", "epoch", "seed"} <= checkpoint.keys():
        raise ValueError("Expected a MORPH training checkpoint with source configuration")
    trained, requested = dict(checkpoint["config"]["model"]), dict(config["model"])
    for key in ("sam_checkpoint", "activation_checkpointing", "activation_checkpointing_scope"):
        trained.pop(key, None)
        requested.pop(key, None)
    if trained != requested:
        raise ValueError("Model configuration differs from the training checkpoint")
    source_evaluation = checkpoint["config"]["evaluation"]
    for key in ("threshold", "prompt_seed", "boundary_width", "metric_epsilon"):
        if config["evaluation"][key] != source_evaluation[key]:
            raise ValueError(f"Evaluation setting changed: {key}")
    if checkpoint["epoch"] != checkpoint["config"]["train"]["epochs"]:
        raise ValueError("Evaluation requires the configured final-epoch checkpoint")
    data = config["data"]
    if args.bkai_split:
        data["test_manifests"] = {"bkai": str(Path(args.bkai_split).resolve())}
    datasets = {name: PolypDataset(manifest, data["root"], geometry=data["geometry"])
                for name, manifest in data["test_manifests"].items()}
    for name, dataset in datasets.items():
        if any(record["dataset"] != name for record in dataset.records):
            raise ValueError(f"Incorrect dataset membership in {name} split")
    source = PolypDataset(data["train_manifest"], data["root"], training=True, geometry=data["geometry"])
    check_train_test_overlap(source, datasets.values())
    device = select_device(args.device or config["execution"]["device"])
    set_seed(config["evaluation"]["prompt_seed"])
    model = MORPH(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    protocol = args.protocol or config["evaluation"]["protocol"]
    output = Path(args.output or Path(config["output_dir"]) / str(checkpoint["seed"]) / protocol)
    if (output / "summary.json").exists():
        raise FileExistsError("Evaluation output exists; select a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    rows, click_records = [], []
    for name, dataset in datasets.items():
        for sample in dataset:
            image, target = sample["image"][None].to(device), sample["mask"][None].to(device)
            metrics, clicks, logits = evaluate_sample(
                model, image, target, protocol,
                seed=sample_seed(config["evaluation"]["prompt_seed"], name, sample["image_id"]),
                radius=config["model"]["click_radius"], threshold=config["evaluation"]["threshold"],
                metric_epsilon=config["evaluation"]["metric_epsilon"],
                boundary_width=config["evaluation"]["boundary_width"],
            )
            rows.append({"dataset": name, "image_id": sample["image_id"], "seed": checkpoint["seed"],
                         "protocol": protocol, "num_clicks": len(clicks), **metrics})
            click_records.append({"dataset": name, "image_id": sample["image_id"],
                                  "clicks": [vars(click) for click in clicks]})
            directory = output / "predictions" / name
            directory.mkdir(parents=True, exist_ok=True)
            prediction = torch.sigmoid(logits.float())[0, 0].ge(config["evaluation"]["threshold"])
            Image.fromarray(prediction.cpu().numpy().astype(np.uint8) * 255).save(directory / f"{sample['image_id']}.png")
    summaries, overall = aggregate_rows(rows)
    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"protocol": protocol, "seed": checkpoint["seed"], "zero_shot": bool(args.bkai_split),
               "datasets": summaries, **overall}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "clicks.json").write_text(json.dumps(click_records, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
