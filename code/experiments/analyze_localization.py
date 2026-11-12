"""Analyse E1 records: hop-stratified localization tables, router cross-validation, CIs.

Protocol (fixed before looking at test numbers):
  dev  = all tasks from psf/requests, pallets/flask, mwaskom/seaborn, pylint-dev/pylint,
         pytest-dev/pytest (40 tasks) -> used only to choose the graph retriever variant.
  test = remaining 460 tasks (7 repositories) -> all reported numbers.
Router label y = 1 iff the graph-side retriever ranks the first gold entity strictly
better than BM25 (unranked = +inf); ties go to BM25 (cheaper). Evaluated with stratified
5-fold CV (repeated 5x) and leave-one-repository-out (LORO).

Usage: python analyze_localization.py results/e1_localization_tasks.jsonl results/e1_summary.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swe_local.router import FEATURE_NAMES, LogisticRouter, MLPRouter, PCRouter, auroc  # noqa: E402

DEV_REPOS = {"psf/requests", "pallets/flask", "mwaskom/seaborn", "pylint-dev/pylint", "pytest-dev/pytest"}
CANDIDATES = ["graph", "hybrid", "boost_b0.5_g0.5", "boost_b1.0_g0.5", "boost_b2.0_g0.5", "boost_b4.0_g0.5"]
RNG = np.random.default_rng(20261112)


def hop_bucket(r):
    h = r.get("hop")
    return "unanchored" if h is None else ("0" if h == 0 else "1" if h == 1 else "2+")


def bh(r, name):
    return float(r[name]["budget_hit"])


def rank(r, name, key="ent_rank"):
    v = r[name][key]
    return math.inf if v is None else v


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 2
    idx = RNG.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)]


def mcnemar_exact(a, b):
    """two-sided exact McNemar p-value for paired binary outcomes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p), n01, n10


def table(recs, names):
    out = {}
    for name in names:
        row = {}
        for b in ["0", "1", "2+", "unanchored", "all"]:
            S = [r for r in recs if b == "all" or hop_bucket(r) == b]
            if not S:
                continue
            hits = [bh(r, name) for r in S]
            row[b] = {
                "n": len(S),
                "budget_hit_pct": 100 * float(np.mean(hits)),
                "budget_hit_ci95": boot_ci(hits),
                "ent_recall@5_pct": 100 * float(np.mean([rank(r, name) <= 5 for r in S])),
                "ent_recall@10_pct": 100 * float(np.mean([rank(r, name) <= 10 for r in S])),
                "file_recall@1_pct": 100 * float(np.mean([rank(r, name, "file_rank") <= 1 for r in S])),
                "file_recall@5_pct": 100 * float(np.mean([rank(r, name, "file_rank") <= 5 for r in S])),
                "mean_ctx_tokens": float(np.mean([r[name]["ctx_tokens"] for r in S])),
            }
        out[name] = row
    return out


def cv_predict(model_fn, X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        m = model_fn().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])
    return p


def strat_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    idx_pos = rng.permutation(np.where(y == 1)[0])
    idx_neg = rng.permutation(np.where(y == 0)[0])
    parts = [np.concatenate([idx_pos[i::k], idx_neg[i::k]]) for i in range(k)]
    all_idx = np.arange(len(y))
    return [(np.setdiff1d(all_idx, te), te) for te in parts]


def main(inp, outp):
    recs = [json.loads(l) for l in open(inp) if l.strip()]
    recs = [r for r in recs if "skip" not in r]
    dev = [r for r in recs if r["repo"] in DEV_REPOS]
    test = [r for r in recs if r["repo"] not in DEV_REPOS]
    dev_scores = {c: float(np.mean([bh(r, c) for r in dev])) for c in CANDIDATES}
    chosen = max(CANDIDATES, key=lambda c: (dev_scores[c], -CANDIDATES.index(c)))

    summary = {"n_total": len(recs), "n_dev": len(dev), "n_test": len(test),
               "dev_budget_hit_by_candidate": dev_scores, "graph_side_selected_on_dev": chosen,
               "hop_counts_test": {b: sum(hop_bucket(r) == b for r in test) for b in ["0", "1", "2+", "unanchored"]},
               "hop_counts_all": {b: sum(hop_bucket(r) == b for r in recs) for b in ["0", "1", "2+", "unanchored"]}}
    summary["table_test"] = table(test, ["bm25", "graph"] + [c for c in CANDIDATES if c != "graph"])

    # ---------------- router
    X = np.array([r["features"] for r in test])
    y = np.array([1 if rank(r, chosen) < rank(r, "bm25") else 0 for r in test])
    repos = np.array([r["repo"] for r in test])
    hit_bm = np.array([bh(r, "bm25") for r in test])
    hit_gr = np.array([bh(r, chosen) for r in test])
    models = {"logistic": LogisticRouter, "mlp_backprop": MLPRouter, "predictive_coding": PCRouter}
    router = {"label_positive_rate": float(y.mean()), "features": FEATURE_NAMES, "models": {}}
    for name, M in models.items():
        aucs, hits, t_fit = [], [], []
        for rep in range(3):
            folds = strat_folds(y, 5, 1000 + rep)
            t0 = time.perf_counter()
            p = cv_predict(lambda: M(seed=rep), X, y, folds)
            t_fit.append(time.perf_counter() - t0)
            aucs.append(auroc(y, p))
            hits.append(np.where(p >= 0.5, hit_gr, hit_bm))
        hits_mean = np.mean(hits, 0)
        # LORO
        p_loro = np.zeros(len(y))
        for rp in sorted(set(repos)):
            te = np.where(repos == rp)[0]
            tr = np.where(repos != rp)[0]
            p_loro[te] = M(seed=0).fit(X[tr], y[tr]).predict_proba(X[te])
        hit_loro = np.where(p_loro >= 0.5, hit_gr, hit_bm)
        m = M(seed=0).fit(X, y)
        t0 = time.perf_counter()
        for _ in range(200):
            m.predict_proba(X[:1])
        lat_ms = (time.perf_counter() - t0) / 200 * 1000
        rep_hits = hits[0].astype(bool)
        pval, n01, n10 = mcnemar_exact(hit_bm.astype(bool), rep_hits)
        by_hop = {}
        for b in ["0", "1", "2+", "unanchored"]:
            sel = np.array([hop_bucket(r) == b for r in test])
            by_hop[b] = 100 * float(hits_mean[sel].mean()) if sel.any() else None
        router["models"][name] = {
            "cv5x3_auroc_mean": float(np.mean(aucs)), "cv5x3_auroc_sd": float(np.std(aucs)),
            "cv_routed_budget_hit_pct": 100 * float(hits_mean.mean()),
            "cv_routed_budget_hit_ci95": boot_ci(hits[0]),
            "cv_routed_budget_hit_by_hop_pct": by_hop,
            "loro_auroc": auroc(y, p_loro), "loro_routed_budget_hit_pct": 100 * float(hit_loro.mean()),
            "mcnemar_vs_bm25_rep0": {"p": pval, "bm25_only_hits": n10, "routed_only_hits": n01},
            "inference_latency_ms_cpu": lat_ms, "cv_fit_seconds_total": float(np.mean(t_fit)),
        }
        if name == "predictive_coding":
            router["models"][name]["energy_first_last"] = [m.energy_trace[0], m.energy_trace[-1]]
    oracle = np.maximum(hit_bm, hit_gr)
    router["oracle_budget_hit_pct"] = 100 * float(oracle.mean())
    router["bm25_budget_hit_pct"] = 100 * float(hit_bm.mean())
    router["graph_side_budget_hit_pct"] = 100 * float(hit_gr.mean())
    router["oracle_by_hop_pct"] = {b: 100 * float(oracle[np.array([hop_bucket(r) == b for r in test])].mean())
                                   for b in ["0", "1", "2+", "unanchored"]}
    summary["router"] = router

    # ---------------- cost (CPU, this machine)
    tim = {k: [r["timing_s"][k] for r in test] for k in test[0]["timing_s"]}
    summary["timing_test_median_s"] = {k: float(np.median(v)) for k, v in tim.items()}
    summary["timing_test_p95_s"] = {k: float(np.percentile(v, 95)) for k, v in tim.items()}
    summary["graph_size_median"] = {"files": float(np.median([r["n_files"] for r in test])),
                                    "entities": float(np.median([r["n_entities"] for r in test])),
                                    "edges": float(np.median([r["n_edges"] for r in test]))}
    summary["parse_failures_total"] = int(sum(r["parse_failures"] for r in recs))
    with open(outp, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ["n_test", "graph_side_selected_on_dev", "hop_counts_test"]}, indent=1))
    for n in ["bm25", chosen]:
        print(n, {b: round(v["budget_hit_pct"], 1) for b, v in summary["table_test"][n].items()})
    print("oracle", round(router["oracle_budget_hit_pct"], 1), router["oracle_by_hop_pct"])
    for n, v in router["models"].items():
        print(n, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k not in ("cv_routed_budget_hit_ci95",)})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
