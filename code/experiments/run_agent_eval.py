"""E4 protocol (needs a local model): tool-loop localization with Gemma 4 E2B / E4B on SWE-bench Verified.

For each task: materialise the indexable source tree at base_commit, run the gemnav tool loop with the
issue text as the question, and score whether the final answer names a gold file (file-level) and a
gold entity name (entity-level). No patches are generated and no tests are run, so results are
directly comparable to the retrieval-only numbers of E1 (same gold labels).

  # 1. serve a model (see README for the llama.cpp commands), then
  python experiments/run_agent_eval.py --parquet verified.parquet --mirror-root /tmp/mirrors \
      --server http://127.0.0.1:8080 --tag gemma4-e4b-q4km --limit 100 --out results/e4_gemma4-e4b.jsonl
  # harness smoke test without a model (answers with the gold file; proves plumbing ONLY, not capability):
  python experiments/run_agent_eval.py ... --stub-oracle --limit 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyarrow.parquet as pq  # noqa: E402

from exp_localization import ensure_mirror, gold_entities, read_tree  # noqa: E402
from gemnav.client import LlamaServerClient, ScriptedClient  # noqa: E402
from gemnav.loop import run_agent  # noqa: E402
from gemnav.tools import Toolbox  # noqa: E402
from swe_local.codegraph import build_graph  # noqa: E402


def materialise(files: dict[str, str], root: str) -> None:
    for rel, src in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--mirror-root", default="/tmp/mirrors")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--budget", type=int, default=5800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stub-oracle", action="store_true")
    a = ap.parse_args()
    rows = sorted(pq.read_table(a.parquet).to_pylist(), key=lambda r: r["instance_id"])
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.limit]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["instance_id"] for l in open(a.out) if l.strip()}
    with open(a.out, "a") as out:
        for row in rows:
            if row["instance_id"] in done:
                continue
            mirror = ensure_mirror(row["repo"], a.mirror_root)
            files = read_tree(mirror, row["base_commit"])
            g = build_graph(files)
            gold, gold_files = gold_entities(g, row["patch"])
            with tempfile.TemporaryDirectory() as td:
                materialise(files, td)
                client = (ScriptedClient([json.dumps({"answer": "Fix is in " + (sorted(gold_files) or ["?"])[0]})])
                          if a.stub_oracle else LlamaServerClient(a.server))
                res = run_agent(row["problem_statement"][:6000], Toolbox(td), client, a.max_iters, a.budget)
            ans = res["answer"] or ""
            rec = {"instance_id": row["instance_id"], "tag": a.tag, "answered": res["answer"] is not None,
                   "iterations": res["iterations"], "max_prompt_tokens": res["max_prompt_tokens"],
                   "mean_latency_s": sum(t["latency_s"] for t in res["trajectory"]) / max(len(res["trajectory"]), 1),
                   "file_hit": any(f in ans for f in gold_files),
                   "entity_hit": any(e.split("::", 1)[1].split(".")[-1] in ans for e in gold if "::" in e),
                   "protocol_errors": sum(1 for t in res["trajectory"] if "error" in t),
                   "stub": a.stub_oracle}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec)


if __name__ == "__main__":
    main()
