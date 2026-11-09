from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MORPH with 0, 1 or 2 clicks")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs/morph_0m.yaml"))
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam_checkpoint")
    parser.add_argument("--protocol", choices=["0-m", "1-m", "2-m"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output")
    parser.add_argument("--bkai_split", help="CSV for strict BKAI zero-shot evaluation")
    parser.add_argument("--scope", choices=["benchmark", "development"], default="benchmark")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from models import MORPH
    from utils.common import apply_overrides, load_config, resolve_config, sample_seed, select_device, set_seed, manifest_hash, source_code_hash
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
    if data["geometry"] != checkpoint["config"]["data"]["geometry"]:
        raise ValueError("Evaluation geometry differs from training")
    phase = checkpoint["config"]["train"].get("phase")
    if phase != ("development" if args.scope == "development" else "final"):
        raise ValueError("Checkpoint phase does not match evaluation scope")
    if args.scope == "development":
        if args.bkai_split:
            raise ValueError("BKAI cannot be used for model development")
        data["train_manifest"] = data["development_fit_manifest"]
        data["test_manifests"] = {"development": data["development_manifest"]}
    if args.bkai_split:
        data["test_manifests"] = {"bkai": str(Path(args.bkai_split).resolve())}
    datasets = {name: PolypDataset(manifest, data["root"], geometry=data["geometry"], source_partition=args.scope == "development")
                for name, manifest in data["test_manifests"].items()}
    for name, dataset in datasets.items():
        if name != "development" and any(record["dataset"] != name for record in dataset.records):
            raise ValueError(f"Incorrect dataset membership in {name} split")
    source = PolypDataset(data["train_manifest"], data["root"], training=True, geometry=data["geometry"])
    check_train_test_overlap(source, datasets.values())
    device = select_device(args.device or config["execution"]["device"])
    set_seed(config["evaluation"]["prompt_seed"])
    model = MORPH(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    protocol = args.protocol or config["evaluation"]["protocol"]
    if checkpoint["manifest_hash"] != manifest_hash(data["train_manifest"]):
        raise ValueError("The training manifest differs from the checkpoint source split")
    output = Path(args.output or Path(config["output_dir"]) / phase / str(checkpoint["seed"]) / protocol)
    if (output / "summary.json").exists():
        raise FileExistsError("Evaluation output exists; select a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    rows, click_records = [], []
    for name, dataset in datasets.items():
        for sample in dataset:
            dataset_name = sample["dataset"]
            image, target = sample["image"][None].to(device), sample["mask"][None].to(device)
            metrics, clicks, logits = evaluate_sample(
                model, image, target, protocol,
                seed=sample_seed(config["evaluation"]["prompt_seed"], dataset_name, sample["image_id"]),
                radius=config["model"]["click_radius"], threshold=config["evaluation"]["threshold"],
                metric_epsilon=config["evaluation"]["metric_epsilon"],
                boundary_width=config["evaluation"]["boundary_width"],
                valid_mask=sample["valid_mask"][None].to(device),
            )
            rows.append({"dataset": dataset_name, "image_id": sample["image_id"], "seed": checkpoint["seed"],
                         "protocol": protocol, "num_clicks": len(clicks), **metrics})
            click_records.append({"dataset": dataset_name, "image_id": sample["image_id"],
                                  "clicks": [vars(click) for click in clicks]})
            directory = output / "predictions" / dataset_name
            directory.mkdir(parents=True, exist_ok=True)
            prediction = torch.sigmoid(logits.float())[0, 0].ge(config["evaluation"]["threshold"])
            Image.fromarray(prediction.cpu().numpy().astype(np.uint8) * 255).save(directory / f"{sample['image_id']}.png")
    summaries, overall = aggregate_rows(rows)
    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"protocol": protocol, "seed": checkpoint["seed"], "scope": args.scope,
                "model_implementation": config["model"]["implementation"],
                "training_config": {"model":trained,"train":checkpoint["config"]["train"]},
                "evaluation_config": {k:v for k,v in config["evaluation"].items() if k != "protocol"},
                "geometry": data["geometry"], "preprocessing": "aspect_resize_mean_pad_normalize_zero_pad",
                "metric_space": "padded_grid", "click_radius": config["model"]["click_radius"],
                "manifest_hashes": {name:manifest_hash(path) for name,path in data["test_manifests"].items()},
                "checkpoint_sha256": manifest_hash(args.checkpoint), "source_code_sha256": source_code_hash()}
    metadata["metrics_sha256"] = manifest_hash(output/"per_image_metrics.csv")
    summary = {"protocol": protocol, "seed": checkpoint["seed"], "scope": args.scope, "zero_shot": bool(args.bkai_split),
               "datasets": summaries, **overall}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "clicks.json").write_text(json.dumps(click_records, indent=2), encoding="utf-8")
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
