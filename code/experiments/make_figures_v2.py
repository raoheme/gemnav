"""Writeup figures from measured results only: fig2 (all retrievers by hop depth) and the 560x280 card image."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1]
e1 = json.load(open(f"{R}/e1_summary.json"))["table_test"]
e5 = json.load(open(f"{R}/e5_summary.json"))["table_test"]
hops = ["0", "1", "2+"]
series = [
    ("BM25 (AST entities)", e1["bm25"], "#8da0a8", "-"),
    ("BM25 ⊕ code graph (RRF)", e1["hybrid"], "#D55E00", "-"),
    ("Graph-boosted BM25 β=0.5", e1["boost_b0.5_g0.5"], "#E69F00", "--"),
    ("BM25 (tree-sitter chunks)", e5["bm25"], "#56B4E9", "--"),
    ("MiniLM dense", e5["dense_minilm"], "#009E73", "--"),
    ("BM25 ⊕ MiniLM (RRF)", e5["rrf_bm25_minilm"], "#0072B2", "-"),
    ("CodeBERT dense", e5["dense_codebert"], "#CC79A7", ":"),
]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
for name, t, c, ls in series:
    ys = [t[h]["budget_hit_pct"] for h in hops]
    lw = 3 if name in ("BM25 ⊕ code graph (RRF)", "BM25 ⊕ MiniLM (RRF)") else 1.8
    ax.plot(range(3), ys, marker="o", color=c, ls=ls, lw=lw, label=name)
    ax.annotate(f"{ys[-1]:.0f}", (2, ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=c)
ax.set_xticks(range(3))
n = e1["bm25"]
ax.set_xticklabels([f"Hop {h}\n(n={n[h]['n']})" for h in hops])
ax.set_ylabel("Faulty entity inside 3,600-token context (%)")
ax.set_ylim(-2, 100)
ax.set_xlim(-0.2, 2.35)
ax.grid(alpha=0.3)
ax.set_title("HopScope: retrieval for small agents, by dependency depth (SWE-bench Verified, 460 held-out tasks)", fontsize=10)
ax.legend(fontsize=8, loc="lower left", ncol=2)
plt.tight_layout()
plt.savefig(f"{R}/fig2_retrievers_by_depth.png")

# card 560x280
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.patch.set_facecolor("#0f1720")
ax = fig.add_axes([0.06, 0.16, 0.5, 0.62])
ax.set_facecolor("#0f1720")
for name, t, c, lw in (("BM25", e1["bm25"], "#9fb3bd", 2), ("+ code graph", e1["hybrid"], "#ff7a3d", 3), ("+ MiniLM", e5["rrf_bm25_minilm"], "#3db2ff", 3)):
    ys = [t[h]["budget_hit_pct"] for h in hops]
    ax.plot(range(3), ys, marker="o", color=c, lw=lw)
    ax.text(2.1, ys[-1], name, color=c, fontsize=7, va="center")
ax.set_xticks(range(3))
ax.set_xticklabels(["hop 0", "hop 1", "hop 2+"], color="#cfd8dc", fontsize=7)
ax.set_yticks([0, 50, 100])
ax.tick_params(colors="#cfd8dc", labelsize=7)
for s in ax.spines.values():
    s.set_color("#33424d")
ax.set_ylim(0, 100)
ax.set_xlim(-0.2, 2.9)
fig.text(0.06, 0.86, "gemnav · HopScope", color="white", fontsize=15, weight="bold")
fig.text(0.62, 0.62, "Code graphs help\nshallow bugs,\nhurt deep ones.", color="white", fontsize=11, weight="bold", va="top")
fig.text(0.62, 0.24, "Gemma 4 E2B/E4B · offline\nSWE-bench Verified × depth", color="#9fb3bd", fontsize=7.5)
plt.savefig(f"{R}/card_560x280.png", facecolor=fig.get_facecolor())
