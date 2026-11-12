"""Figure 1 from results/e1_summary.json only (no hand-entered numbers)."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

s = json.load(open(sys.argv[1]))
out = sys.argv[2]
T = s["table_test"]
hops = ["0", "1", "2+"]
labels = [f"Hop {h}\n(n={T['bm25'][h]['n']})" for h in hops]
series = [("bm25", "BM25 (lexical)", "#8da0a8"), ("boost_b0.5_g0.5", "Graph-boosted BM25 (β=0.5)", "#0072B2"),
          ("hybrid", "BM25 ⊕ graph (RRF)", "#D55E00")]
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=200)
w = 0.26
for i, (k, name, c) in enumerate(series):
    ys = [T[k][h]["budget_hit_pct"] for h in hops]
    lo = [T[k][h]["budget_hit_pct"] - T[k][h]["budget_hit_ci95"][0] for h in hops]
    hi = [T[k][h]["budget_hit_ci95"][1] - T[k][h]["budget_hit_pct"] for h in hops]
    xs = [j + (i - 1) * w for j in range(3)]
    a.bar(xs, ys, w, yerr=[lo, hi], color=c, label=name, capsize=2, error_kw={"lw": 0.8})
    for x, y in zip(xs, ys):
        a.text(x, y + 3.2, f"{y:.0f}", ha="center", fontsize=8)
a.set_xticks(range(3))
a.set_xticklabels(labels)
a.set_ylabel("Gold entity inside 3,600-token context (%)")
a.set_title("Turn-1 retrieval by dependency depth\n(SWE-bench Verified, 460 held-out tasks, 95% bootstrap CI)", fontsize=10)
a.set_ylim(0, 108)
a.legend(fontsize=8, loc="upper right")
a.grid(axis="y", alpha=0.3)

R = s["router"]
names = ["BM25\nalways", "Graph-side\nalways", "Logistic\nrouter", "MLP\nrouter", "Predictive-\ncoding router", "Oracle\nrouter"]
vals = [R["bm25_budget_hit_pct"], R["graph_side_budget_hit_pct"], R["models"]["logistic"]["cv_routed_budget_hit_pct"],
        R["models"]["mlp_backprop"]["cv_routed_budget_hit_pct"], R["models"]["predictive_coding"]["cv_routed_budget_hit_pct"],
        R["oracle_budget_hit_pct"]]
cols = ["#8da0a8", "#D55E00", "#999999", "#999999", "#009E73", "#333333"]
b.bar(range(6), vals, color=cols)
for x, v in enumerate(vals):
    b.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
b.set_xticks(range(6))
b.set_xticklabels(names, fontsize=8)
b.set_ylim(50, 72)
auc = {k: R["models"][k]["cv5x3_auroc_mean"] for k in R["models"]}
b.set_title("Per-task routing between BM25 and graph side\n(5-fold CV; AUROC " + " / ".join(f"{v:.2f}" for v in auc.values()) + " ≈ chance)", fontsize=10)
b.set_ylabel("Gold entity inside context (%)")
b.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(out)
