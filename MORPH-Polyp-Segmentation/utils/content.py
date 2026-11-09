"""Native RGB/GT content audit. No resizing or benchmark record removal."""
import hashlib
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


def decoded_hash(path, mode="RGB"):
    with Image.open(path) as image:
        image = image.convert(mode)
        channels = 3 if mode == "RGB" else 1
        return hashlib.sha256(struct.pack(">QQQ", image.height, image.width, channels) + image.tobytes()).hexdigest()


def audit_records(training, evaluation, workers=4):
    def inspect(item):
        split, row = item
        with Image.open(row["image_path"]) as image, Image.open(row["mask_path"]) as mask:
            geometry_matches = image.size == mask.size
        return {"dataset": row["dataset"], "image_id": row["image_id"], "split": split,
                "geometry_matches": geometry_matches,
                "rgb_sha256": decoded_hash(row["image_path"]),
                "mask_sha256": decoded_hash(row["mask_path"], "L")}
    records = [("training", row) for row in training] + [("evaluation", row) for row in evaluation]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(inspect, records))
    train_hashes = {r["rgb_sha256"] for r in rows if r["split"] == "training"}
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "evaluation":
            groups[row["rgb_sha256"]].append(row)
    for rgb, members in groups.items():
        conflict = len({r["mask_sha256"] for r in members}) > 1
        overlap = rgb in train_hashes
        invalid_geometry = not all(r["geometry_matches"] for r in members)
        for row in members:
            row.update(rgb_group_id="RGB_"+rgb, group_size=len(members),
                       training_overlap=overlap, gt_conflict=conflict,
                       content_eligible=not (overlap or conflict or invalid_geometry))
    summary = {"training_records": len(training), "training_unique_rgb": len(train_hashes),
               "evaluation_records": len(evaluation), "evaluation_unique_rgb": len(groups),
               "redundant_evaluation_records": len(evaluation)-len(groups),
               "duplicate_rgb_groups": sum(len(g)>1 for g in groups.values()),
               "training_overlap_groups": sum(g[0]["training_overlap"] for g in groups.values()),
               "gt_conflict_groups": sum(g[0]["gt_conflict"] for g in groups.values()),
               "eligible_content_groups": sum(g[0]["content_eligible"] for g in groups.values()),
               "rgb_hash_specification":"SHA256(three big-endian uint64 H,W,3 + native contiguous uint8 RGB bytes)",
               "mask_hash_specification":"SHA256(three big-endian uint64 H,W,1 + native uint8 grayscale bytes)"}
    return rows, summary
