"""Analyse E5 (tree-sitter chunks: BM25 vs dense embedders vs BM25+dense RRF) on the 460 held-out tasks,
stratified by the E1 hop depth. Also picks the agent's embedder on the 40 DEV tasks only (budget-hit, then
throughput as tie-break), so the end-to-end choice is not tuned on the test tasks.

python analyze_dense.py results/kaggle_e5 results/e1_localization_tasks.jsonl results/e5_summary.json
"""
import glob
import json
import os
import sys

import numpy as np

DEV = {"psf/requests", "pallets/flask", "mwaskom/seaborn", "pylint-dev/pylint", "pytest-dev/pytest"}
RNG = np.random.default_rng(20261112)


def ci(x):
    x = np.asarray(x, float)
    m = x[RNG.integers(0, len(x), (2000, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)]


def main(d, e1p, outp):
    e1 = {json.loads(l)["instance_id"]: json.loads(l) for l in open(e1p)}
    hop = {i: ("unanchored" if r.get("hop") is None else "0" if r["hop"] == 0 else "1" if r["hop"] == 1 else "2+") for i, r in e1.items()}
    per = {}
    thr = {}
    for f in sorted(glob.glob(os.path.join(d, "e5_*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if "skip" in r:
                per.setdefault(r["instance_id"], {"skip": r["skip"], "repo": r["repo"]})
                continue
            cur = per.setdefault(r["instance_id"], {"repo": r["repo"]})
            for k, v in r.items():
                if isinstance(v, dict):
                    cur[k] = v
    for f in glob.glob(os.path.join(d, "*.throughput.json")):
        thr.update(json.load(open(f)))
    names = sorted({k for r in per.values() for k in r if isinstance(r.get(k), dict)}, key=lambda k: (k != "bm25", k))
    ok = {i: r for i, r in per.items() if "skip" not in r and all(n in r for n in names)}
    test = [i for i, r in ok.items() if r["repo"] not in DEV]
    dev = [i for i, r in ok.items() if r["repo"] in DEV]
    S = {"n_tasks_scored": len(ok), "n_test": len(test), "n_dev": len(dev), "skipped": [i for i, r in per.items() if "skip" in r],
         "retrievers": names, "throughput_t4_fp16": thr, "table_test": {}}
    for n in names:
        row = {}
        for b in ("0", "1", "2+", "unanchored", "all"):
            ids = [i for i in test if b == "all" or hop.get(i) == b]
            if not ids:
                continue
            hits = [float(ok[i][n]["budget_hit"]) for i in ids]
            row[b] = {"n": len(ids), "budget_hit_pct": 100 * float(np.mean(hits)), "ci95": ci(hits),
                      "file_recall@1_pct": 100 * float(np.mean([(ok[i][n]["file_rank"] or 1e9) <= 1 for i in ids])),
                      "chunk_recall@5_pct": 100 * float(np.mean([(ok[i][n]["chunk_rank"] or 1e9) <= 5 for i in ids]))}
        S["table_test"][n] = row
    dense_cands = [n for n in names if n.startswith("rrf_bm25_")]
    dev_score = {n: float(np.mean([ok[i][n]["budget_hit"] for i in dev])) for n in dense_cands}
    S["dev_budget_hit"] = dev_score
    best = max(dense_cands, key=lambda n: (dev_score[n], (thr.get(n.replace("rrf_bm25_", ""), {}) or {}).get("chunks_per_s") or 0))
    S["agent_embedder_selected_on_dev"] = best.replace("rrf_bm25_", "")
    json.dump(S, open(outp, "w"), indent=1)
    for n in names:
        print(f"{n:24s}", {b: round(v["budget_hit_pct"], 1) for b, v in S["table_test"][n].items()})
    print("dev:", {k: round(v, 3) for k, v in dev_score.items()}, "-> agent embedder:", S["agent_embedder_selected_on_dev"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
