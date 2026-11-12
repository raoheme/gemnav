"""Cohesive 16:9 gallery panels + 560x280 card, all numbers read from results/. python make_gallery.py results results/gallery"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BG, INK, MUTED, GRID = "#fbfaf7", "#1f2a30", "#5f6f78", "#e6e2da"
GREY, ORANGE, BLUE, GREEN, PINK = "#9aa7ae", "#d4572a", "#1d6fb8", "#2b9a6e", "#b5609a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
S1 = json.load(open(f"{R}/e1_summary.json"))
S5 = json.load(open(f"{R}/e5_summary.json"))
G = json.load(open(f"{R}/e2e3_summary.json"))
E2 = json.load(open(f"{R}/e2e/e2e_e4b_summary.json"))
T1, T5 = S1["table_test"], S5["table_test"]
hops = ["0", "1", "2+"]


def panel(n, kicker, headline, sub):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    fig.text(0.05, 0.92, f"{n:02d}  ·  {kicker.upper()}", fontsize=15, color=ORANGE, weight="bold")
    fig.text(0.05, 0.835, headline, fontsize=34, color=INK, weight="bold")
    fig.text(0.05, 0.785, sub, fontsize=16, color=MUTED)
    fig.text(0.05, 0.03, "gemnav · HopScope   |   SWE-bench Verified · Gemma 4 E2B/E4B · llama.cpp   |   github.com/raoheme/gemnav",
             fontsize=11, color=MUTED)
    return fig


def ax_in(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.grid(axis="y", color=GRID)
    ax.tick_params(labelsize=14)
    return ax


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", facecolor=BG)
    plt.close(fig)


# 1 headline
fig = panel(1, "the finding", "Code graphs help shallow bugs and hurt deep ones",
            "Share of 460 held-out tasks where the code to fix fits in a 3,600-token context")
ax = ax_in(fig, [0.07, 0.12, 0.62, 0.6])
for name, t, c, lw, ls in [("BM25", T1["bm25"], GREY, 3, "-"), ("BM25 + code graph", T1["hybrid"], ORANGE, 5, "-"),
                            ("BM25 + MiniLM", T5["rrf_bm25_minilm"], BLUE, 5, "-"), ("CodeBERT", T5["dense_codebert"], PINK, 2.5, ":")]:
    ys = [t[h]["budget_hit_pct"] for h in hops]
    ax.plot(range(3), ys, color=c, lw=lw, ls=ls, marker="o", ms=10)
    ax.text(2.08, ys[-1], f"{name}  {ys[-1]:.0f}%", color=c, fontsize=15, va="center", weight="bold")
ax.set_xticks(range(3))
ax.set_xticklabels([f"hop {h}\n{T1['bm25'][h]['n']} tasks" for h in hops], fontsize=15)
ax.set_ylim(-3, 100)
ax.set_xlim(-0.15, 2.9)
ax.set_ylabel("%", fontsize=14)
fig.text(0.74, 0.62, f"{T1['bm25']['0']['budget_hit_pct']:.0f}% → {T1['hybrid']['0']['budget_hit_pct']:.0f}%", fontsize=40, color=ORANGE, weight="bold")
fig.text(0.74, 0.57, "graph on direct-mention bugs", fontsize=15, color=MUTED)
fig.text(0.74, 0.43, f"{T1['bm25']['2+']['budget_hit_pct']:.0f}% → {T1['hybrid']['2+']['budget_hit_pct']:.0f}%", fontsize=40, color=ORANGE, weight="bold")
fig.text(0.74, 0.38, "graph on 2+ hop bugs", fontsize=15, color=MUTED)
fig.text(0.74, 0.24, f"{T1['bm25']['2+']['budget_hit_pct']:.0f}% → {T5['rrf_bm25_minilm']['2+']['budget_hit_pct']:.0f}%", fontsize=40, color=BLUE, weight="bold")
fig.text(0.74, 0.19, "a 22M MiniLM on 2+ hop bugs", fontsize=15, color=MUTED)
save(fig, "g01_finding.png")

# 2 what depth means
fig = panel(2, "the benchmark", "Depth: hops from what the issue names to the fix",
            "Real task psf__requests-1142, measured on our code graph of the repository")
ax = fig.add_axes([0.03, 0.12, 0.94, 0.6])
ax.axis("off")
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
nodes = [(2, "Issue", "“requests.get is ALWAYS\nsending content length”", "#eef1f2"), (6, "requests/__init__.py", "named in the issue\nhop 0", "#dce9f5"),
         (10, "PreparedRequest", "imported class\nhop 1", "#fbe3d6"), (14, "prepare_content_length", "where the fix is\nhop 2", "#f4b99a")]
for x, t, s, c in nodes:
    ax.add_patch(FancyBboxPatch((x - 1.45, 2.2), 2.9, 2.2, boxstyle="round,pad=0.08", fc=c, ec=INK, lw=1.5))
    ax.text(x, 3.75, t.replace("prepare_content_length", "prepare_\ncontent_length"), ha="center", va="center", fontsize=15 if len(t) < 20 else 14, weight="bold", color=INK)
    ax.text(x, 2.85, s, ha="center", va="center", fontsize=13, color=MUTED)
for a, b, lab in [(2, 6, "mentions"), (6, 10, "imports"), (10, 14, "contains")]:
    ax.add_patch(FancyArrowPatch((a + 1.55, 3.3), (b - 1.55, 3.3), arrowstyle="-|>", mutation_scale=28, color=INK, lw=2))
    ax.text((a + b) / 2, 4.75, lab, ha="center", fontsize=13, color=MUTED)
c = S1["hop_counts_all"]
ax.text(0.3, 0.8, f"All 500 tasks labelled:  hop 0 = {c['0']}   hop 1 = {c['1']}   hop 2+ = {c['2+']}   no anchor = {c['unanchored']}",
        fontsize=17, color=INK, weight="bold")
save(fig, "g02_depth.png")

# 3 gains/losses
e1 = [json.loads(l) for l in open(f"{R}/e1_localization_tasks.jsonl")]
DEV = {"psf/requests", "pallets/flask", "mwaskom/seaborn", "pylint-dev/pylint", "pytest-dev/pytest"}
test = [r for r in e1 if r["repo"] not in DEV]


def hb(r):
    h = r.get("hop")
    return "u" if h is None else ("0" if h == 0 else "1" if h == 1 else "2+")


fig = panel(3, "why averages hide it", "On deep bugs the graph rescues 2 tasks and loses 28",
            "Tasks flipped by adding the code graph to BM25 (blue: miss → hit, orange: hit → miss)")
ax = ax_in(fig, [0.12, 0.14, 0.8, 0.56])
ax.grid(False)
for i, b in enumerate(hops):
    S = [r for r in test if hb(r) == b]
    g = sum(1 for r in S if r["hybrid"]["budget_hit"] and not r["bm25"]["budget_hit"])
    lo = sum(1 for r in S if r["bm25"]["budget_hit"] and not r["hybrid"]["budget_hit"])
    ax.barh(i, g, color=BLUE, height=0.6)
    ax.barh(i, -lo, color=ORANGE, height=0.6)
    ax.text(g + 0.8, i, f"+{g}", va="center", fontsize=22, color=BLUE, weight="bold")
    ax.text(-lo - 0.8, i, f"−{lo}", va="center", ha="right", fontsize=22, color=ORANGE, weight="bold")
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(range(3))
ax.set_yticklabels([f"hop {h}" for h in hops], fontsize=18)
ax.invert_yaxis()
ax.set_xlim(-40, 40)
ax.set_xlabel("tasks", fontsize=14)
ax.spines["left"].set_visible(False)
save(fig, "g03_gains_losses.png")

# 4 embedders
fig = panel(4, "embeddings", "A 22M general-text model beats code-pretrained encoders",
            "Gold inside context, all 460 held-out tasks. CodeBERT/GraphCodeBERT were never trained for retrieval.")
ax = ax_in(fig, [0.22, 0.13, 0.7, 0.58])
rows = [("BM25 + MiniLM (RRF)", T5["rrf_bm25_minilm"]["all"]["budget_hit_pct"], BLUE), ("MiniLM-L6-v2 alone", T5["dense_minilm"]["all"]["budget_hit_pct"], GREEN),
        ("BM25 over chunks", T5["bm25"]["all"]["budget_hit_pct"], GREY), ("GraphCodeBERT", T5["dense_graphcodebert"]["all"]["budget_hit_pct"], PINK),
        ("CodeBERT", T5["dense_codebert"]["all"]["budget_hit_pct"], PINK)]
ax.grid(axis="x", color=GRID)
ax.grid(axis="y", visible=False)
for i, (n, v, c) in enumerate(rows):
    ax.barh(i, v, color=c, height=0.62)
    ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=20, weight="bold", color=c)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=16)
ax.invert_yaxis()
ax.set_xlim(0, 80)
th = S5["throughput_t4_fp16"]
fig.text(0.62, 0.2, f"T4 fp16 throughput: MiniLM {th['minilm']['chunks_per_s']:.0f} chunks/s,\nCodeBERT {th['codebert']['chunks_per_s']:.0f}, GraphCodeBERT {th['graphcodebert']['chunks_per_s']:.0f}",
         fontsize=13, color=MUTED)
save(fig, "g04_embedders.png")

# 5 routing
Rr = S1["router"]
fig = panel(5, "the open problem", "Nobody can tell from the issue whether a bug is deep",
            f"Three learned routers choose graph vs BM25 per task. AUROC ≈ {Rr['models']['logistic']['cv5x3_auroc_mean']:.2f} (chance = 0.50).")
ax = ax_in(fig, [0.08, 0.13, 0.86, 0.58])
lab = ["BM25\nalways", "graph side\nalways", "logistic", "MLP", "predictive\ncoding", "perfect\nrouter"]
vals = [Rr["bm25_budget_hit_pct"], Rr["graph_side_budget_hit_pct"], Rr["models"]["logistic"]["cv_routed_budget_hit_pct"],
        Rr["models"]["mlp_backprop"]["cv_routed_budget_hit_pct"], Rr["models"]["predictive_coding"]["cv_routed_budget_hit_pct"], Rr["oracle_budget_hit_pct"]]
ax.bar(range(6), vals, color=[GREY, ORANGE, "#c3cbd0", "#c3cbd0", GREEN, INK], width=0.65)
for i, v in enumerate(vals):
    ax.text(i, v + 0.35, f"{v:.1f}", ha="center", fontsize=20, weight="bold", color=INK)
ax.set_xticks(range(6))
ax.set_xticklabels(lab, fontsize=15)
ax.set_ylim(50, 72)
ax.set_ylabel("% gold in context", fontsize=14)
ax.annotate("", xy=(5, Rr["oracle_budget_hit_pct"] - 0.3), xytext=(1, Rr["graph_side_budget_hit_pct"] + 1.3),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.5, connectionstyle="arc3,rad=-0.2"))
ax.text(2.9, 69.5, f"+{Rr['oracle_budget_hit_pct'] - Rr['graph_side_budget_hit_pct']:.1f} points left on the table", fontsize=17, color=ORANGE, weight="bold")
save(fig, "g05_routing.png")

# 6 agent
fig = panel(6, "the agent", "gemnav: a repair loop that cannot cheat",
            "Everything local: llama.cpp, SQLite, tree-sitter, pylsp, git. One JSON tool call per turn under a per-tool grammar.")
ax = fig.add_axes([0.03, 0.08, 0.94, 0.66])
ax.axis("off")
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
B = [(0.3, 4.2, 3.3, 2.3, "1  Index", "tree-sitter chunks\nSQLite + MiniLM + BM25\npylsp (Jedi)", "#dce9f5"),
     (0.3, 0.8, 3.3, 2.6, "2  Freeze a test", "model writes a repro script;\nstored read-only\noutside the repo", "#ece3f3"),
     (4.6, 2.3, 3.4, 2.8, "3  Gemma 4", "E2B / E4B, QAT q4_0\n8K context\ngreedy, seed 42", "#fdf0dc"),
     (9.0, 4.2, 3.2, 2.3, "4a  Read", "search · definition\nreferences · read_file", "#dff2e7"),
     (9.0, 0.8, 3.2, 2.6, "4b  Edit", "guard → py_compile →\nrepro → old tests", "#f9e0db"),
     (13.0, 0.8, 2.8, 2.6, "5  Keep or undo", "worse? git reset\n+ digest check", "#e9ecee"),
     (13.0, 4.2, 2.8, 2.3, "6  Patch", "git diff →\nofficial harness", "#d9f1f3")]
for x, y, w, h, t, s, c in B:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=c, ec=INK, lw=1.4))
    ax.text(x + w / 2, y + h - 0.45, t, ha="center", fontsize=16, weight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - 0.3, s, ha="center", va="center", fontsize=12.5, color=MUTED, linespacing=1.4)
for a, b in [((3.6, 5.3), (4.6, 4.4)), ((3.6, 2.1), (4.6, 3.0)), ((8.0, 4.4), (9.0, 5.3)), ((8.0, 3.0), (9.0, 2.1)), ((12.2, 2.1), (13.0, 2.1)), ((12.2, 5.3), (13.0, 5.3))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=22, color=INK, lw=1.8))
save(fig, "g06_agent.png")

# 7 guards
E = G["E2_guards"]
fig = panel(7, "safety", "Every cheating attempt we built was blocked",
            "Attacks constructed from the real benchmark tests for all 500 tasks; 1 legitimate fix wrongly rejected (it edits setup.cfg)")
ax = ax_in(fig, [0.32, 0.13, 0.6, 0.58])
ax.grid(False)
items = [("edit an existing test", E["A_test_tampering"]["blocked"], E["A_test_tampering"]["n"], GREEN),
         ("inject a conftest.py", E["B_config_injection"]["blocked"], E["B_config_injection"]["n"], GREEN),
         ("shadow a module the tests import", E["C_import_shadow"]["blocked"], E["C_import_shadow"]["n"], GREEN),
         ("random agent mutations rolled back", G["E3_rollback"]["exact_restores"], G["E3_rollback"]["trials"], GREEN),
         ("correct fixes wrongly rejected", E["benign_false_reject"], E["n"], ORANGE)]
for i, (lab_, k, n, c) in enumerate(items):
    ax.barh(i, 100 * k / n, color=c, height=0.6)
    ax.text(max(100 * k / n, 0) + 1.5, i, f"{k} / {n}", va="center", fontsize=19, weight="bold", color=c)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([x[0] for x in items], fontsize=15)
ax.invert_yaxis()
ax.set_xlim(0, 120)
ax.set_xticks([0, 50, 100])
ax.set_xlabel("%", fontsize=14)
save(fig, "g07_guards.png")

# 8 laptop + speed
gate = json.load(open(f"{R}/laptop/speed_gate.json", encoding="utf-8-sig"))
fig = panel(8, "consumer hardware", "On an 8 GB laptop it runs, slowly, on CPU only",
            "Ryzen 5 5500U, 6.8 GB usable RAM, Gemma 4 E2B QAT q4_0. The Radeon iGPU could not allocate llama.cpp's buffers.")
ax = ax_in(fig, [0.3, 0.14, 0.62, 0.56])
ax.grid(axis="x", color=GRID)
ax.grid(axis="y", visible=False)
rows = [("Kaggle T4, E4B (GPU working)", E2["prompt_tok_s_median"], E2["gen_tok_s_median"], BLUE),
        ("laptop CPU, E2B (best run)", 114.0, 5.4, GREEN),
        ("laptop CPU, E2B (throttled)", gate["cpu"]["prompt_tok_s"], gate["cpu"]["gen_tok_s"], "#8bc4a4"),
        ("Kaggle T4, CUDA libs missing", 13.8, 3.7, ORANGE)]
for i, (n, p, g, c) in enumerate(rows):
    ax.barh(i, p, color=c, height=0.6)
    ax.text(p * 1.12, i, f"{p:,.0f} prompt · {g:.1f} gen tok/s", va="center", fontsize=15, color=INK)
ax.set_xscale("log")
ax.set_xlim(8, 60000)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=14)
ax.invert_yaxis()
ax.set_xlabel("prompt tokens per second (log scale)", fontsize=13)
fig.text(0.05, 0.1, "≈ 50 s per agent turn\non the laptop", fontsize=17, color=GREEN, weight="bold")
save(fig, "g08_hardware.png")

# 9 e2e
fig = panel(9, "end-to-end, honestly", "It finds the right file, then fails the exact edit",
            "Gemma 4 E4B on 50 held-out tasks, scored by the official SWE-bench harness (gold patch resolves in all 8 checked environments)")
ax = ax_in(fig, [0.3, 0.13, 0.62, 0.58])
ax.grid(False)
steps = [("attempted", E2["n"], GREY), ("produced a patch", E2["patches"], BLUE), ("edited the right file", E2["patch_in_gold_file"], BLUE),
         ("applied cleanly", E2["applied"], BLUE), ("passed 167 of 168 target tests", 1, GREEN), ("resolved", E2["resolved"], ORANGE)]
for i, (n, v, c) in enumerate(steps):
    ax.barh(i, v, color=c, height=0.62)
    ax.text(v + 0.8, i, str(v), va="center", fontsize=22, weight="bold", color=c)
ax.set_yticks(range(len(steps)))
ax.set_yticklabels([s[0] for s in steps], fontsize=15)
ax.invert_yaxis()
ax.set_xlim(0, 58)
fig.text(0.05, 0.1, f"{E2['edit_nomatch']} edit attempts\ncould not quote\nthe code exactly", fontsize=16, color=ORANGE, weight="bold")
save(fig, "g09_end_to_end.png")

# card 560x280
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.patch.set_facecolor("#14212b")
fig.text(0.055, 0.8, "HopScope", fontsize=24, color="white", weight="bold")
fig.text(0.055, 0.69, "where code graphs help small coding agents", fontsize=9.5, color="#b9c6cd")
ax = fig.add_axes([0.06, 0.22, 0.5, 0.4])
ax.set_facecolor("#14212b")
for t, c, lw in ((T1["bm25"], "#9aa7ae", 2), (T1["hybrid"], "#ff7a45", 3.2), (T5["rrf_bm25_minilm"], "#4fb3ff", 3.2)):
    ax.plot(range(3), [t[h]["budget_hit_pct"] for h in hops], color=c, lw=lw, marker="o", ms=4)
ax.set_xticks(range(3))
ax.set_xticklabels(["hop 0", "hop 1", "hop 2+"], fontsize=7.5, color="#b9c6cd")
ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_ylim(0, 100)
fig.text(0.62, 0.62, f"{T1['hybrid']['0']['budget_hit_pct']:.0f}%", fontsize=22, color="#ff7a45", weight="bold")
fig.text(0.79, 0.645, "graph,\nshallow bugs", fontsize=7.5, color="#b9c6cd", linespacing=1.1)
fig.text(0.62, 0.40, f"{T1['hybrid']['2+']['budget_hit_pct']:.0f}%", fontsize=22, color="#ff7a45", weight="bold")
fig.text(0.79, 0.425, "graph,\ndeep bugs", fontsize=7.5, color="#b9c6cd", linespacing=1.1)
fig.text(0.62, 0.18, f"{T5['rrf_bm25_minilm']['2+']['budget_hit_pct']:.0f}%", fontsize=22, color="#4fb3ff", weight="bold")
fig.text(0.79, 0.205, "MiniLM,\ndeep bugs", fontsize=7.5, color="#b9c6cd", linespacing=1.1)
fig.text(0.055, 0.035, "Gemma 4 · SWE-bench Verified · 500 tasks · offline", fontsize=7, color="#7f939e")
fig.savefig(f"{OUT}/card_560x280.png", facecolor=fig.get_facecolor())
print(sorted(os.listdir(OUT)))
