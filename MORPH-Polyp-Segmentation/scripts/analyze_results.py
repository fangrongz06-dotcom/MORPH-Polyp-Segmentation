"""Analyze measured three-seed outputs; one predefined Holm family per call."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.statistics import grouped_differences, signed_rank, holm, run_summaries

FAMILIES = {"main_0m":("0-m",),"main_prompt":("1-m","2-m"),"bkai_prompt":("1-m","2-m")}


def read_measured(paths):
    rows,metadata = [],[]
    for path in paths:
        with path.open(encoding="utf-8-sig",newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"dataset","image_id","seed","protocol","dice","iou"} <= set(reader.fieldnames or []):
                raise ValueError("Expected measured per_image_metrics.csv fields")
            current = list(reader)
        item = json.loads((path.parent/"evaluation_metadata.json").read_text(encoding="utf-8"))
        if item.get("metrics_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Metrics CSV differs from its evaluation provenance hash")
        if not current or {int(r["seed"]) for r in current}!={int(item["seed"])} or {r["protocol"] for r in current}!={item["protocol"]}:
            raise ValueError("CSV and evaluation provenance differ")
        if item.get("scope") != "benchmark":
            raise ValueError("Development results are not formal benchmark comparisons")
        rows.extend(current)
        metadata.append(item)
    if any(m["training_config"]!=metadata[0]["training_config"] for m in metadata):
        raise ValueError("Within-method runs differ in settings other than training seed")
    return rows,metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left",type=Path,nargs="+",required=True)
    parser.add_argument("--right",type=Path,nargs="+",required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--family",choices=FAMILIES,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output exists")
    left,lmeta = read_measured(args.left)
    right,rmeta = read_measured(args.right)
    protocols = FAMILIES[args.family]
    if any({r["protocol"] for r in rows} != set(protocols) for rows in (left,right)):
        raise ValueError("Files must contain exactly the protocols of this Holm family")
    with args.audit.open(encoding="utf-8") as stream:
        audit = [json.loads(line) for line in stream if line.strip()]
    datasets = {r["dataset"] for r in audit if r["split"]=="evaluation"}
    if (datasets=={"bkai"}) != (args.family=="bkai_prompt"):
        raise ValueError("BKAI and main benchmark families must remain separate")
    expected = {"bkai":1000} if args.family=="bkai_prompt" else {"kvasir":100,"clinicdb":62,"colondb":380,"etis":196,"endoscene":60}
    if Counter(r["dataset"] for r in audit if r["split"]=="evaluation") != expected:
        raise ValueError("Audit record counts differ from the selected paper benchmark family")
    comparisons,summaries = [],{}
    for protocol in protocols:
        provenance = [m for m in lmeta+rmeta if m["protocol"]==protocol]
        for key in ("evaluation_config","geometry","preprocessing","metric_space","click_radius","manifest_hashes"):
            if any(m[key]!=provenance[0][key] for m in provenance):
                raise ValueError(f"Paired evaluation setting differs: {key}")
        summaries[protocol] = {}
        for metric in ("dice","iou"):
            groups,audit_counts = grouped_differences(left,right,audit,protocol,metric)
            result = signed_rank([g["difference"] for g in groups])
            comparisons.append({"protocol":protocol,"metric":metric,**audit_counts,**result})
            summaries[protocol][metric] = {"left":run_summaries(left,protocol,metric),"right":run_summaries(right,protocol,metric)}
    for result,p in zip(comparisons,holm([r["p_value"] for r in comparisons])):
        result["p_holm"] = p
        result["p_holm_display"] = "<0.001" if p<.001 else "<0.01" if p<.01 else "<0.05" if p<.05 else ">=0.05"
    report = {"family":args.family,"holm_comparison_count":len(comparisons),"paired":comparisons,
              "three_run_summaries":summaries,"statistical_unit":"audited RGB content group; patient independence not established"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
