"""Three-run summaries and paired inference on audited image-content groups."""
import math
from collections import defaultdict

import numpy as np

SEEDS = {1234,2345,3456}


def repeated_run_ci(values):
    values = np.asarray(values,dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Exactly three finite independent-run values are required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    margin = 4.3026527299*sd/math.sqrt(3)
    return {"mean":mean,"sample_sd":sd,"ci95":[mean-margin,mean+margin],"n_runs":3}


def grouped_differences(left, right, audit, protocol, metric):
    records = [r for r in audit if r["split"] == "evaluation"]
    expected = {(r["dataset"],r["image_id"]) for r in records}
    if not expected or len(expected) != len(records):
        raise ValueError("Audit must contain unique, nonempty evaluation record keys")
    methods = []
    for rows in (left,right):
        selected = [r for r in rows if r["protocol"] == protocol]
        if {int(r["seed"]) for r in selected} != SEEDS:
            raise ValueError("Each protocol requires seeds 1234,2345,3456")
        by_seed = {}
        for seed in sorted(SEEDS):
            run = [r for r in selected if int(r["seed"]) == seed]
            index = {(r["dataset"],r["image_id"]):float(r[metric]) for r in run}
            if len(index) != len(run) or set(index) != expected:
                raise ValueError("Every method/run must cover exactly all audited evaluation IDs")
            if not all(math.isfinite(v) and 0<=v<=1 for v in index.values()):
                raise ValueError("Dice/IoU must be finite values in [0,1]")
            by_seed[seed] = index
        methods.append({key:sum(by_seed[seed][key] for seed in sorted(SEEDS))/3 for key in expected})
    groups = defaultdict(list)
    for record in records:
        if not isinstance(record["content_eligible"],bool):
            raise ValueError("Audit eligibility must be a JSON boolean")
        groups[record["rgb_sha256"]].append(record)
    output, excluded = [], 0
    for rgb,members in sorted(groups.items()):
        if not all(r["content_eligible"] for r in members):
            excluded += 1
            continue
        if len({r["mask_sha256"] for r in members}) != 1:
            raise ValueError("Unresolved GT differences cannot be averaged")
        keys = [(r["dataset"],r["image_id"]) for r in members]
        a,b = (sum(method[k] for k in keys)/len(keys) for method in methods)
        # Eq. 18 averages the record differences, in manifest order.
        difference = sum(methods[0][key]-methods[1][key] for key in keys)/len(keys)
        output.append({"rgb_sha256":rgb,"member_count":len(keys),"left_mean":a,"right_mean":b,"difference":difference})
    if not output:
        raise ValueError("No eligible matched content groups")
    return output,{"benchmark_records_per_run":len(records),"rgb_content_groups":len(groups),"excluded_groups":excluded}


def signed_rank(difference):
    d = np.asarray(difference,float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError("A nonempty finite paired-difference vector is required")
    nonzero = d[d != 0]
    order = np.argsort(np.abs(nonzero))
    values = np.abs(nonzero)[order]
    ranks = np.empty(len(nonzero),float)
    start = 0
    while start<len(nonzero):
        end = start+1
        while end<len(nonzero) and values[end] == values[start]:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    positive,negative = float(ranks[nonzero>0].sum()),float(ranks[nonzero<0].sum())
    distribution = np.array([1.0])
    for rank in np.rint(2*ranks).astype(int):
        updated = np.zeros(len(distribution)+rank)
        updated[:len(distribution)] += distribution*.5
        updated[rank:] += distribution*.5
        distribution = updated
    observed = int(round(2*positive))
    p = min(1.,2*min(distribution[:observed+1].sum(),distribution[observed:].sum()))
    return {"N_pair":len(d),"N_zero":int((d==0).sum()),"n":len(nonzero),
            "mean_difference":float(d.mean()),"median_difference":float(np.median(d)),
            "rank_biserial":(positive-negative)/(positive+negative) if len(nonzero) else 0.,
            "W_plus":positive,"W_minus":negative,"p_value":float(p),
            "p_value_method":"two_sided_exact_conditional_sign_permutation_DP",
            "zero_policy":"discard_strict_zero","ties":"average_ranks"}


def holm(pvalues):
    p = np.asarray(pvalues,float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError("Invalid p-values")
    result = np.empty_like(p)
    running = 0.
    for rank,index in enumerate(np.argsort(p)):
        running = max(running,(len(p)-rank)*p[index])
        result[index] = min(1.,running)
    return result.tolist()


def run_summaries(rows,protocol,metric):
    selected = [r for r in rows if r["protocol"] == protocol]
    per_dataset = sorted({r["dataset"] for r in selected})
    summaries,macro_runs = {},[]
    for dataset in per_dataset:
        means = [np.mean([float(r[metric]) for r in selected if int(r["seed"])==seed and r["dataset"]==dataset]) for seed in sorted(SEEDS)]
        summaries[dataset] = repeated_run_ci(means)
    for seed in sorted(SEEDS):
        macro_runs.append(float(np.mean([np.mean([float(r[metric]) for r in selected if int(r["seed"])==seed and r["dataset"]==ds]) for ds in per_dataset])))
    return {"datasets":summaries,"unweighted_dataset_macro":repeated_run_ci(macro_runs)}
