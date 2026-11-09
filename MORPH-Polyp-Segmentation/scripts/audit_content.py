"""Audit user-supplied benchmark or BKAI manifests without changing membership."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dataset import PolypDataset
from utils.content import audit_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--training_manifest", type=Path, default=ROOT/"splits/local/train.csv")
    parser.add_argument("--evaluation_manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty audit output directory")
    train = PolypDataset(args.training_manifest, args.data_root, training=True)
    datasets = [PolypDataset(path,args.data_root) for path in args.evaluation_manifests]
    rows, summary = audit_records(train.records, [r for dataset in datasets for r in dataset.records])
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"content_audit.jsonl").open("x",encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
