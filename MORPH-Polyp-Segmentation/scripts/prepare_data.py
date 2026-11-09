"""Create private manifests from an already partitioned PraNet-style download."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dataset import PolypDataset, TEST_DIRECTORIES, training_directory
from utils.content import audit_records

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COUNTS = {"kvasir":100, "clinicdb":62, "colondb":380, "etis":196, "endoscene":60}


def write_split(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_splits(data_root, output, source_train_manifest=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty manifest output directory")
    directory = training_directory(data_root)
    images = sorted(p.name for p in (directory/"image").iterdir() if p.suffix.lower() in EXTENSIONS)
    if source_train_manifest:
        with Path(source_train_manifest).open(encoding="utf-8-sig", newline="") as stream:
            train = [{"dataset":r["dataset"], "filename":r["filename"]} for r in csv.DictReader(stream)]
        if sorted(r["filename"] for r in train) != images:
            raise ValueError("Source manifest differs from the supplied training image directory")
    else:
        train = []
        for filename in images:
            stem = Path(filename).stem
            if stem.isdecimal():
                dataset = "clinicdb"
            elif re.fullmatch(r"c[a-z0-9]{24}", stem):
                dataset = "kvasir"
            else:
                raise ValueError("Nonstandard source filenames: supply --source_train_manifest with dataset,filename")
            train.append({"dataset":dataset, "filename":filename})
    if Counter(r["dataset"] for r in train) != {"kvasir":900, "clinicdb":550}:
        raise ValueError("Expected the original 900 Kvasir + 550 ClinicDB source split; no repartition is performed")
    manifests = {"train":train}
    for dataset, expected in COUNTS.items():
        files = sorted(p.name for p in (data_root/"TestDataset"/TEST_DIRECTORIES[dataset]/"images").iterdir() if p.suffix.lower() in EXTENSIONS)
        if len(files) != expected:
            raise ValueError(f"Expected {expected} already-held-out records for {dataset}; found {len(files)}")
        manifests["test_"+dataset] = [{"dataset":dataset, "filename":name} for name in files]
    development_keys = set()
    for dataset, count in (("kvasir",90), ("clinicdb",55)):
        ranked = sorted((r for r in train if r["dataset"] == dataset), key=lambda r:
                        hashlib.sha256(f"1234|{dataset}|{Path(r['filename']).stem}".encode()).hexdigest())
        development_keys.update((r["dataset"],r["filename"]) for r in ranked[:count])
    manifests["development"] = [r for r in train if (r["dataset"],r["filename"]) in development_keys]
    manifests["development_fit"] = [r for r in train if (r["dataset"],r["filename"]) not in development_keys]
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in manifests.items():
        write_split(output/(name+".csv"), rows)
    source = PolypDataset(output/"train.csv", data_root, training=True)
    tests = [PolypDataset(output/f"test_{dataset}.csv", data_root) for dataset in COUNTS]
    rows, summary = audit_records(source.records, [r for test in tests for r in test.records])
    if summary["training_overlap_groups"]:
        raise ValueError("Training/test RGB overlap found: investigate the supplied dataset before training")
    with (output/"content_audit.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    (output/"content_audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return {"counts":{name:len(rows) for name,rows in manifests.items()},"content_audit":summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"splits/local")
    parser.add_argument("--source_train_manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_splits(args.data_root, args.output, args.source_train_manifest), indent=2))


if __name__ == "__main__":
    main()
