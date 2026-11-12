"""E1 - Turn-1 localization on SWE-bench Verified, stratified by dependency hop depth.

No language model is involved: this measures what the retrieval layer hands a small model
under a fixed 3,600-token budget. For every task we
  1. read every indexable .py file at `base_commit` from a local bare clone,
  2. build the typed code graph and a BM25 index over code entities,
  3. derive gold entities = innermost entity containing each original-side line changed by
     the gold patch (tests excluded),
  4. anchor the issue text to graph seeds and compute hop depth d = BFS distance seed->gold,
  5. rank entities with BM25, graph propagation and RRF hybrid; record ranks, budgeted hits,
     router features and timings.
Output: one JSON line per task (resumable).

Usage: python exp_localization.py --parquet verified.parquet --repo django/django \
           --mirror /tmp/mirrors --out results/e1_tasks.jsonl [--limit N] [--sample-frac F]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pyarrow.parquet as pq  # noqa: E402

from swe_local.codegraph import build_graph, is_indexable, is_test_path, issue_seeds  # noqa: E402
from swe_local.patches import parse_unified_diff  # noqa: E402
from swe_local.retrieval import (BM25, budgeted_context, file_ranking, first_hit_rank,  # noqa: E402
                                 graph_boost_rank, graph_rank, graph_scores, rrf)
from swe_local.router import route_features  # noqa: E402

BUDGET = 3600


def ensure_mirror(repo: str, mirror_root: str) -> str:
    path = os.path.join(mirror_root, repo.replace("/", "__") + ".git")
    if not os.path.isdir(path):
        os.makedirs(mirror_root, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--bare", "--filter=blob:limit=300k",
                        f"https://github.com/{repo}.git", path], check=True)
    return path


def read_tree(mirror: str, commit: str) -> dict[str, str]:
    ls = subprocess.run(["git", "-C", mirror, "ls-tree", "-r", commit], capture_output=True, text=True, check=True).stdout
    want = []
    for line in ls.splitlines():
        meta, path = line.split("\t", 1)
        _mode, typ, sha = meta.split()
        if typ == "blob" and is_indexable(path):
            want.append((path, sha))
    if not want:
        return {}
    p = subprocess.run(["git", "-C", mirror, "cat-file", "--batch"], input="\n".join(s for _, s in want).encode() + b"\n",
                       capture_output=True, check=True)
    out, buf, i = {}, p.stdout, 0
    for path, sha in want:
        nl = buf.index(b"\n", i)
        header = buf[i:nl].decode().split()
        if len(header) < 3 or header[1] == "missing":
            i = nl + 1
            continue
        size = int(header[2])
        data = buf[nl + 1 : nl + 1 + size]
        i = nl + 1 + size + 1
        try:
            out[path] = data.decode("utf-8")
        except UnicodeDecodeError:
            out[path] = data.decode("latin-1")
    return out


def gold_entities(g, patch: str) -> tuple[set[str], set[str]]:
    ents, files = set(), set()
    for fp in parse_unified_diff(patch):
        if fp.is_new_file or not fp.old_path or not fp.old_path.endswith(".py") or is_test_path(fp.old_path):
            continue
        if fp.old_path not in g.entities:
            continue
        files.add(fp.old_path)
        for ln in fp.old_changed_lines:
            e = g.innermost_entity(fp.old_path, ln)
            if e:
                ents.add(e)
    return ents, files


def run_task(row: dict, mirror: str) -> dict:
    t0 = time.perf_counter()
    files = read_tree(mirror, row["base_commit"])
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = build_graph(files)
    t_graph = time.perf_counter() - t0
    t0 = time.perf_counter()
    bm = BM25(g)
    t_bm25_index = time.perf_counter() - t0
    gold, gold_files = gold_entities(g, row["patch"])
    issue = row["problem_statement"]
    rec = {"instance_id": row["instance_id"], "repo": row["repo"], "difficulty": row.get("difficulty"),
           "n_files": len(files), "n_entities": len(g.entities), "n_edges": len(g.typed_edges),
           "parse_failures": g.parse_failures, "gold": sorted(gold), "gold_files": sorted(gold_files)}
    if not gold:
        rec["skip"] = "no_gold_entity_in_index"
        return rec
    t0 = time.perf_counter()
    seeds = issue_seeds(g, issue)
    t_seed = time.perf_counter() - t0
    dist = g.bfs_dist(seeds, max_depth=8)
    d = min((dist[e] for e in gold if e in dist), default=None)
    rec["hop"] = d
    rec["n_seeds"] = len(seeds)

    t0 = time.perf_counter()
    bsc = bm25_scores = bm.scores(issue)
    r_bm = sorted(bsc, key=lambda k: (-bsc[k], k))
    t_bm_q = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_gr = graph_rank(g, seeds, bm25_scores) if seeds else list(r_bm)
    t_gr_q = time.perf_counter() - t0
    r_hy = rrf(r_bm, r_gr)
    t0 = time.perf_counter()
    r_boost = {f"boost_b{b}_g{gm}": graph_boost_rank(g, seeds, bm25_scores, dist, beta=b, gamma=gm)
               for b in (0.5, 1.0, 2.0, 4.0) for gm in (0.5,)}
    t_boost_q = (time.perf_counter() - t0) / len(r_boost)
    gs = graph_scores(g, seeds) if seeds else {}
    gsorted = sorted(gs.values(), reverse=True)[:2]
    two_hop = sum(1 for v in dist.values() if v <= 2)

    rec["features"] = route_features(issue, [bsc[k] for k in r_bm[:2]], seeds, gsorted, r_bm[:10], r_gr[:10], two_hop).tolist()
    for name, r in [("bm25", r_bm), ("graph", r_gr), ("hybrid", r_hy)] + sorted(r_boost.items()):
        ctx, used = budgeted_context(g, r, BUDGET)
        fr = file_ranking(g, r)
        rec[name] = {
            "ent_rank": first_hit_rank(r, gold, 2000),
            "file_rank": first_hit_rank(fr, gold_files, 2000),
            "budget_hit": any(e in gold for e in ctx),
            "budget_gold_frac": sum(1 for e in gold if e in ctx) / len(gold),
            "budget_files_hit": any(g.entities[e].path in gold_files for e in ctx),
            "ctx_entities": len(ctx),
            "ctx_tokens": used,
        }
    rec["timing_s"] = {"read": t_read, "graph_build": t_graph, "bm25_index": t_bm25_index, "seed": t_seed,
                       "bm25_query": t_bm_q, "graph_query": t_gr_q, "boost_query": t_boost_q}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mirror", default="/tmp/mirrors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th task starting at i")
    ap.add_argument("--time-budget", type=float, default=540.0, help="stop cleanly after this many seconds")
    a = ap.parse_args()
    rows = [r for r in pq.read_table(a.parquet).to_pylist() if r["repo"] == a.repo]
    rows.sort(key=lambda r: r["instance_id"])
    rng = random.Random(a.seed)
    if a.sample_frac < 1.0:
        k = max(1, round(len(rows) * a.sample_frac))
        rows = sorted(rng.sample(rows, k), key=lambda r: r["instance_id"])
    if a.limit:
        rows = rows[: a.limit]
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = rows[si::sn]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(l)["instance_id"] for l in f if l.strip()}
    mirror = ensure_mirror(a.repo, a.mirror)
    start = time.time()
    n = 0
    with open(a.out, "a") as f:
        for row in rows:
            if row["instance_id"] in done:
                continue
            if time.time() - start > a.time_budget:
                print(f"time budget reached; {n} new tasks", flush=True)
                break
            try:
                rec = run_task(row, mirror)
            except Exception as ex:  # record and continue; failures are reported in the paper
                rec = {"instance_id": row["instance_id"], "repo": row["repo"], "skip": f"error:{type(ex).__name__}:{ex}"[:300]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            print(rec["instance_id"], rec.get("hop"), rec.get("bm25", {}).get("ent_rank"), rec.get("graph", {}).get("ent_rank"),
                  round(sum(rec.get("timing_s", {}).values()), 1), flush=True)
    remaining = len([r for r in rows if r["instance_id"] not in done]) - n
    print(f"REMAINING {remaining}")


if __name__ == "__main__":
    main()
