"""All writeup figures, generated from results/ only. python make_writeup_figures.py results results/figs"""
import collections
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
GREY, ORANGE, BLUE, GREEN, PINK = "#8a9ba5", "#D55E00", "#0072B2", "#009E73", "#CC79A7"
DEV = {"psf/requests", "pallets/flask", "mwaskom/seaborn", "pylint-dev/pylint", "pytest-dev/pytest"}
e1 = [json.loads(l) for l in open(f"{R}/e1_localization_tasks.jsonl")]
S1 = json.load(open(f"{R}/e1_summary.json"))
S5 = json.load(open(f"{R}/e5_summary.json"))
G = json.load(open(f"{R}/e2e3_summary.json"))


def hb(r):
    h = r.get("hop")
    return "unanchored" if h is None else ("0" if h == 0 else "1" if h == 1 else "2+")


def save(name):
    plt.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- 1. what "hop depth" means (real task)
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.axis("off")
ax.set_xlim(-0.1, 10.7)
ax.set_ylim(0, 3.6)
nodes = [(1.15, 1.8, "Issue text", "“requests.get is ALWAYS\nsending content length”", "#eef2f4"),
         (3.9, 1.8, "requests/__init__.py", "anchor found in the issue\n(hop 0)", "#dbe9f6"),
         (6.6, 1.8, "PreparedRequest", "class  ·  hop 1", "#fde7d9"),
         (9.4, 1.8, "prepare_\ncontent_length", "method · hop 2 · the fix", "#f7c9a8")]
for x, y, t, sub, c in nodes:
    ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.55), 2.2, 1.1, boxstyle="round,pad=0.05", fc=c, ec="#455a64", lw=1))
    ax.text(x, y + 0.15, t, ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(x, y - 0.33, sub, ha="center", va="center", fontsize=8, color="#37474f")
for (a, b, lab) in [(1.15, 3.9, "issue mentions\n“requests”"), (3.9, 6.6, "imports\n(from .models import ...)"), (6.6, 9.4, "contains")]:
    ax.add_patch(FancyArrowPatch((a + 1.2, 1.8), (b - 1.2, 1.8), arrowstyle="-|>", mutation_scale=14, color="#455a64"))
    ax.text((a + b) / 2, 2.55, lab, ha="center", fontsize=8, color="#455a64")
ax.text(0.0, 3.35, "What we mean by dependency depth: a real task (psf__requests-1142)", fontsize=11, weight="bold")
r = next(x for x in e1 if x["instance_id"] == "psf__requests-1142")
ax.text(0.0, 0.35, f"Rank of the faulty method: BM25 #{r['bm25']['ent_rank']}, BM25 ⊕ graph #{r['hybrid']['ent_rank']}, "
        f"graph only #{r['graph']['ent_rank']}. The issue names the package, not the method that has to change.", fontsize=8.5, color="#37474f")
save("f1_hop_depth_example.png")

# ---------------------------------------------------------------- 2. benchmark composition
repos = collections.defaultdict(collections.Counter)
for r in e1:
    repos[r["repo"]][hb(r)] += 1
order = sorted(repos, key=lambda k: -sum(repos[k].values()))
fig, ax = plt.subplots(figsize=(9.5, 4.2))
left = [0] * len(order)
for b, c in (("0", "#9ecae1"), ("1", "#4292c6"), ("2+", "#08519c"), ("unanchored", "#d9d9d9")):
    v = [repos[k][b] for k in order]
    ax.barh(range(len(order)), v, left=left, color=c, label=f"hop {b}" if b != "unanchored" else "no anchor found")
    left = [a + x for a, x in zip(left, v)]
for i, k in enumerate(order):
    ax.text(left[i] + 2, i, f"{left[i]}" + ("   dev" if k in DEV else ""), va="center", fontsize=8, color="#555")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("Tasks")
tot = collections.Counter(hb(r) for r in e1)
ax.set_title(f"HopScope labels for all 500 SWE-bench Verified tasks  (hop 0: {tot['0']}, hop 1: {tot['1']}, hop 2+: {tot['2+']}, "
             f"no anchor: {tot['unanchored']})", fontsize=10)
ax.legend(fontsize=8, loc="lower right", frameon=False)
save("f2_benchmark_composition.png")

# ---------------------------------------------------------------- 3. retrievers by depth (headline)
T1, T5 = S1["table_test"], S5["table_test"]
hops = ["0", "1", "2+"]
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for name, t, c, ls, lw in [("BM25", T1["bm25"], GREY, "-", 2.2), ("BM25 + code graph (RRF)", T1["hybrid"], ORANGE, "-", 3.2),
                            ("graph-boosted BM25, β = 0.5", T1["boost_b0.5_g0.5"], "#E69F00", "--", 1.8),
                            ("MiniLM dense only", T5["dense_minilm"], GREEN, "--", 1.8),
                            ("BM25 + MiniLM (RRF)", T5["rrf_bm25_minilm"], BLUE, "-", 3.2),
                            ("CodeBERT dense only", T5["dense_codebert"], PINK, ":", 1.8)]:
    ys = [t[h]["budget_hit_pct"] for h in hops]
    ax.plot(range(3), ys, marker="o", color=c, ls=ls, lw=lw, label=name)
    dy = {"MiniLM dense only": 5, "BM25 + MiniLM (RRF)": -5}.get(name, 0)
    ax.annotate(f"{ys[-1]:.0f}%", (2, ys[-1]), xytext=(7, dy), textcoords="offset points", va="center", fontsize=8, color=c)
ax.set_xticks(range(3))
ax.set_xticklabels([f"hop {h}\n({T1['bm25'][h]['n']} tasks)" for h in hops])
ax.set_ylabel("tasks where the code to fix fits in\nthe 3,600-token context (%)")
ax.set_ylim(-2, 100)
ax.set_xlim(-0.2, 2.4)
ax.grid(axis="y", alpha=0.3)
ax.set_title("Retrieval quality falls with dependency depth; the code graph falls fastest\n460 held-out tasks, 7 repositories",
             fontsize=10.5)
ax.legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.95)
save("f3_retrievers_by_depth.png")

# ---------------------------------------------------------------- 4. gained vs lost when adding the graph
test = [r for r in e1 if r["repo"] not in DEV]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, b in enumerate(hops):
    S = [r for r in test if hb(r) == b]
    gain = sum(1 for r in S if r["hybrid"]["budget_hit"] and not r["bm25"]["budget_hit"])
    lost = sum(1 for r in S if r["bm25"]["budget_hit"] and not r["hybrid"]["budget_hit"])
    ax.barh(i, gain, color=BLUE)
    ax.barh(i, -lost, color=ORANGE)
    ax.text(gain + 0.6, i, f"+{gain} rescued", va="center", fontsize=9, color=BLUE)
    ax.text(-lost - 0.6, i, f"{lost} lost", va="center", ha="right", fontsize=9, color=ORANGE)
ax.axvline(0, color="#333", lw=0.8)
ax.set_yticks(range(3))
ax.set_yticklabels([f"hop {h}" for h in hops])
ax.invert_yaxis()
ax.set_xlim(-40, 40)
ax.set_xlabel("tasks (relative to BM25 alone)")
ax.set_title("What fusing the code graph into BM25 does, task by task", fontsize=10.5)
ax.spines["left"].set_visible(False)
save("f4_graph_gains_losses.png")

# ---------------------------------------------------------------- 5. routing: the gap nobody can close
Rr = S1["router"]
labels = ["BM25\nalways", "graph side\nalways", "logistic\nrouter", "MLP\nrouter", "predictive-\ncoding router", "oracle\n(upper bound)"]
vals = [Rr["bm25_budget_hit_pct"], Rr["graph_side_budget_hit_pct"], Rr["models"]["logistic"]["cv_routed_budget_hit_pct"],
        Rr["models"]["mlp_backprop"]["cv_routed_budget_hit_pct"], Rr["models"]["predictive_coding"]["cv_routed_budget_hit_pct"],
        Rr["oracle_budget_hit_pct"]]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.bar(range(6), vals, color=[GREY, ORANGE, "#b0bec5", "#b0bec5", GREEN, "#37474f"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(6))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(50, 72)
ax.set_ylabel("code to fix inside context (%)")
auc = Rr["models"]["logistic"]["cv5x3_auroc_mean"]
ax.set_title(f"Choosing graph vs BM25 per task: learned routers stay near chance (AUROC ≈ {auc:.2f})\n"
             f"while a perfect choice would add {Rr['oracle_budget_hit_pct'] - Rr['graph_side_budget_hit_pct']:.1f} points", fontsize=10.5)
save("f5_routing_gap.png")

# ---------------------------------------------------------------- 6. agent pipeline diagram
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axis("off")
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.4)


def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="#455a64", lw=1))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=9, weight="bold")
    ax.text(x + w / 2, y + h - 0.6, sub, ha="center", va="top", fontsize=7.4, color="#37474f", linespacing=1.35)


def arrow(a, b, lab=""):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#455a64", lw=1))
    if lab:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, lab, ha="center", fontsize=7, color="#455a64")


box(0.1, 2.5, 2.1, 1.6, "Index (once per repo)", "tree-sitter chunks\nSQLite + MiniLM vectors\nBM25 · pylsp (Jedi)\nfile summary map", "#e3f2fd")
box(0.1, 0.3, 2.1, 1.7, "Phase 0", "model writes a repro\nscript, frozen read-only\noutside the repo;\nregression baseline B0", "#f3e5f5")
box(2.8, 1.4, 2.4, 2.0, "Gemma 4 E2B / E4B", "llama.cpp, QAT q4_0\n8K ctx, q4_0 KV cache\ngreedy, seed 42\none JSON action per turn\n(per-tool GBNF grammar)", "#fff3e0")
box(5.8, 2.6, 2.3, 1.5, "Read tools", "semantic_search · ripgrep\nget_definition · references\nread_file · list_symbols", "#e8f5e9")
box(5.8, 0.3, 2.3, 1.9, "Edit tools", "replace_in_file · create_file\n1. protected-file guard\n2. py_compile\n3. repro + regression checks", "#ffebee")
box(8.7, 0.3, 2.2, 1.9, "Git transaction", "keep the edit if the score\n(clean, −broken, repro)\ndid not get worse,\nelse reset + digest check", "#eceff1")
box(8.7, 2.6, 2.2, 1.5, "Output", "git diff of non-test\nfiles → official\nSWE-bench harness", "#e0f7fa")
arrow((2.2, 3.2), (2.8, 2.9), "context")
arrow((2.2, 1.2), (2.8, 1.8), "status")
arrow((5.2, 2.9), (5.8, 3.3), "read")
arrow((5.2, 1.9), (5.8, 1.3), "edit")
arrow((8.1, 1.25), (8.7, 1.25))
arrow((8.1, 3.35), (8.7, 3.35), "submit")
ax.add_patch(FancyArrowPatch((6.95, 2.6), (5.2, 2.4), connectionstyle="arc3,rad=0.3", arrowstyle="-|>", mutation_scale=12, color="#90a4ae", ls="--"))
ax.text(5.9, 2.25, "observation\n(≤15 steps, 30 min)", fontsize=7, color="#607d8b")
ax.text(0.1, 4.25, "gemnav: the offline repair loop", fontsize=11, weight="bold")
save("f6_agent_pipeline.png")

# ---------------------------------------------------------------- 7. speed: the silent CPU fallback
v1 = [json.loads(l) for l in open(f"{R}/kaggle_pilot_v1/e2e_gemma4-e4b-qat-q4_0-gemnav-pilot_tasks.jsonl")]
v3 = [json.loads(l) for l in open(f"{R}/kaggle_pilot_v3/e2e_gemma4-e4b-qat-q4_0-gemnav-pilot_tasks.jsonl")]


def med(recs, key):
    xs = [c[key + "_n"] / (c[key + "_ms"] / 1000) for r in recs for c in r.get("llm_calls", []) if c.get(key + "_ms")]
    return statistics.median(xs), len(xs)


rows = [("Kaggle T4, CUDA libs not found\n→ silent CPU fallback (pilot 1)", *med(v1, "prompt"), med(v1, "predicted")[0]),
        ("Kaggle T4 × 2,\nGPU working (pilot 3)", *med(v3, "prompt"), med(v3, "predicted")[0])]
lap = f"{R}/laptop/speed_gate.json"
if os.path.exists(lap):
    g = json.load(open(lap, encoding="utf-8-sig"))
    for b in ("cpu", "vulkan"):
        if isinstance(g.get(b), dict) and g[b].get("prompt_tok_s"):
            rows.append((f"Ryzen 5 5500U laptop\n({b.upper()}, 6 threads)", g[b]["prompt_tok_s"], 1, g[b]["gen_tok_s"]))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, idx, lab in ((axs[0], 1, "prompt processing (tokens/s)"), (axs[1], 3, "generation (tokens/s)")):
    v = [r[idx] for r in rows]
    ax.barh(range(len(rows)), v, color=[ORANGE, BLUE, GREEN, "#7cb342"][: len(rows)])
    for i, x in enumerate(v):
        ax.text(x * 1.05, i, f"{x:,.1f}", va="center", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows] if idx == 1 else [""] * len(rows), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(lab + ", log scale")
fig.suptitle("Gemma 4 throughput reported by llama-server (median over agent calls; laptop = speed-gate prompt, E2B)\nKaggle rows: E4B QAT q4_0", fontsize=10)
save("f7_throughput.png")

# ---------------------------------------------------------------- 8. guards
E = G["E2_guards"]
items = [("edits to existing tests", E["A_test_tampering"]["blocked"], E["A_test_tampering"]["n"]),
         ("new conftest.py injected", E["B_config_injection"]["blocked"], E["B_config_injection"]["n"]),
         ("module shadowing a test import", E["C_import_shadow"]["blocked"], E["C_import_shadow"]["n"]),
         ("rollbacks restored bit-exactly", G["E3_rollback"]["exact_restores"], G["E3_rollback"]["trials"]),
         ("gold patches wrongly rejected", E["benign_false_reject"], E["n"])]
fig, ax = plt.subplots(figsize=(8.5, 3.2))
for i, (lab, k, n) in enumerate(items):
    bad = lab.startswith("gold")
    ax.barh(i, 100 * k / n, color=ORANGE if bad else GREEN)
    ax.text(100 * k / n + 1, i, f"{k} / {n}", va="center", fontsize=9)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([x[0] for x in items])
ax.invert_yaxis()
ax.set_xlim(0, 115)
ax.set_xlabel("%")
ax.set_title("Safety checks, run on all 500 SWE-bench Verified gold patches\n(attack cases built by us from the real benchmark tests)", fontsize=10)
save("f8_guards.png")
print(sorted(os.listdir(OUT)))

# ---------------------------------------------------------------- 9. end-to-end funnel (measured; official harness)
p = f"{R}/e2e/e2e_e4b_summary.json"
if os.path.exists(p):
    E2 = json.load(open(p))
    steps = [("tasks attempted", E2["n"]), ("produced a patch", E2["patches"]), ("patch edits the right file", E2["patch_in_gold_file"]),
             ("patch applies (official harness)", E2["applied"]), ("best: 167 / 168 target tests", 1), ("resolved", E2["resolved"])]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (lab, v) in enumerate(steps):
        ax.barh(i, v, color=BLUE if i < 4 else (GREEN if i == 4 else ORANGE))
        ax.text(v + 0.6, i, str(v), va="center", fontsize=10, weight="bold")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([s[0] for s in steps])
    ax.invert_yaxis()
    ax.set_xlim(0, 56)
    ax.set_xlabel("tasks (50 stratified, held-out)")
    ax.set_title("Gemma 4 E4B end-to-end: it finds the right file, then can't write the exact edit", fontsize=10.5)
    save("f9_e2e_funnel.png")
