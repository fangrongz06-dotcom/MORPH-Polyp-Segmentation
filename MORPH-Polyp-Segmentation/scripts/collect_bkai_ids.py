"""Collect BKAI IDs from local files, a ZIP inventory, or evaluation records.

Only writes JSON. It does not extract images, select a new split, validate
GT pixels, or substitute a competition split for a completed experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def index_images(names):
    index = {}
    for name in sorted(names):
        file = PurePosixPath(name)
        if file.suffix.lower() not in IMAGE_EXTENSIONS or file.name.startswith("."):
            continue
        key = file.stem.casefold()
        if key in index:
            raise ValueError(f"Ambiguous image ID: {file.stem}")
        index[key] = file
    return index


def paired_inventory(image_names, mask_names):
    images, masks = index_images(image_names), index_images(mask_names)
    if not images or not masks:
        raise ValueError("Both image and GT mask sources must contain image files")
    shared = sorted(images.keys() & masks.keys())
    if not shared:
        raise ValueError("No common image/GT stems; check the selected directories or ZIP prefixes")
    pairs = [{"image_id": images[key].stem, "image_filename": images[key].name,
              "mask_filename": masks[key].name,
              "same_filename_for_morph_loader": images[key].name == masks[key].name}
             for key in shared]
    return {"status": "FILENAME_PAIRED_NOT_EVALUATED", "matched_pairs": len(pairs),
            "unmatched_image_ids": [images[k].stem for k in sorted(images.keys() - masks.keys())],
            "unmatched_mask_ids": [masks[k].stem for k in sorted(masks.keys() - images.keys())],
            "pairs": pairs,
            "scope": "Stem-based inventory only. GT pixels and overlap with training/development are not audited."}


def archive_inventory(archive_path, image_prefix, mask_prefix):
    def prefix(value):
        clean = value.replace("\\", "/").strip("/")
        if not clean or any(part in (".", "..", "") for part in clean.split("/")):
            raise ValueError("Specify a nonempty relative ZIP folder prefix")
        return clean + "/"
    ip, mp = prefix(image_prefix), prefix(mask_prefix)
    if ip.startswith(mp) or mp.startswith(ip):
        raise ValueError("Image and GT prefixes must be disjoint")
    with ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
    return paired_inventory([n for n in names if n.startswith(ip)], [n for n in names if n.startswith(mp)])


def evaluation_inventory(metric_files):
    runs, sources = defaultdict(set), []
    for file in metric_files:
        raw = file.read_bytes()
        rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if not {"dataset", "image_id", "seed", "protocol"} <= set(rows.fieldnames or []):
            raise ValueError("Expected per_image_metrics.csv with dataset,image_id,seed,protocol")
        n = 0
        for row in rows:
            if row["dataset"].strip().casefold() != "bkai":
                continue
            image_id, seed, protocol = row["image_id"], row["seed"], row["protocol"]
            if not image_id or image_id != image_id.strip() or not seed or protocol not in {"0-m", "1-m", "2-m"}:
                raise ValueError("Invalid BKAI image ID, seed or protocol")
            key = (int(seed), protocol)
            if image_id in runs[key]:
                raise ValueError(f"Repeated BKAI record for seed {seed}, protocol {protocol}, ID {image_id}")
            runs[key].add(image_id)
            n += 1
        sources.append({"file": file.name, "sha256": hashlib.sha256(raw).hexdigest(), "bkai_rows": n})
    if not runs:
        raise ValueError("No dataset=bkai rows were found; main benchmark rows are not BKAI")
    sets = list(runs.values())
    return {"status": "IDS_RECORDED_IN_SUPPLIED_EVALUATION_FILES", "sources": sources,
            "same_membership_across_runs": all(ids == sets[0] for ids in sets[1:]),
            "runs": [{"seed": seed, "protocol": protocol, "image_count": len(ids), "image_ids": sorted(ids)}
                     for (seed, protocol), ids in sorted(runs.items())],
            "scope": "Counts are from supplied records, not proof of intended manifest completeness."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    folders = commands.add_parser("folders")
    folders.add_argument("--images", type=Path, required=True)
    folders.add_argument("--masks", type=Path, required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--zip", type=Path, required=True)
    archive.add_argument("--image-prefix", required=True)
    archive.add_argument("--mask-prefix", required=True)
    evaluated = commands.add_parser("evaluated")
    evaluated.add_argument("--metrics", type=Path, nargs="+", required=True)
    for command in (folders, archive, evaluated):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "folders":
        result = paired_inventory([p.name for p in args.images.iterdir() if p.is_file()],
                                  [p.name for p in args.masks.iterdir() if p.is_file()])
    elif args.mode == "archive":
        result = archive_inventory(args.zip, args.image_prefix, args.mask_prefix)
    else:
        result = evaluation_inventory(args.metrics)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Wrote {args.output.name}: {result['status']}")


if __name__ == "__main__":
    main()
