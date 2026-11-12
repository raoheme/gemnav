"""E5 - Blueprint retrieval on SWE-bench Verified: tree-sitter chunks x {BM25, 4 dense embedders, BM25+dense RRF}.

Same tasks, gold patches and 3,600-token budget as E1, but the retrieval unit is the blueprint's tree-sitter chunk
(classes > 200 lines split into header + methods). Gold chunks = chunks whose line span contains an original-side
changed line (most specific chunk wins; module chunk if no definition contains the line).

Embeddings are content-addressed (sha256 of chunk text) in an SQLite cache, so each unique chunk is embedded once
per model across all 500 commits. Query = issue text (truncated by the model's tokenizer).
Output: one JSON line per task with ranks / budget hits per retriever, plus per-model embedding throughput.

python exp_dense.py --parquet verified.parquet --mirror /kaggle/tmp/mirrors --models minilm,codebert \
    --device cuda:0 --cache /kaggle/tmp/emb_cache.sqlite --out e5_part0.jsonl [--shard 0/2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from exp_localization import ensure_mirror, read_tree  # noqa: E402
from gemnav.bm25 import ChunkBM25  # noqa: E402
from gemnav.chunker import chunk_file  # noqa: E402
from gemnav.embedder import Embedder  # noqa: E402
from gemnav.vector_store import chunk_text_for_embedding  # noqa: E402
from swe_local.codegraph import is_test_path  # noqa: E402
from swe_local.patches import parse_unified_diff  # noqa: E402
from swe_local.retrieval import BM25, approx_tokens, rrf, tokenize  # noqa: E402

import hashlib  # noqa: E402
import math  # noqa: E402
import sqlite3  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

BUDGET = 3600


def gold_chunks(chunks, patch):
    by_file = defaultdict(list)
    for i, c in enumerate(chunks):
        by_file[c.file_path].append(i)
    gold, gold_files = set(), set()
    for fp in parse_unified_diff(patch):
        if fp.is_new_file or not fp.old_path or is_test_path(fp.old_path) or fp.old_path not in by_file:
            continue
        gold_files.add(fp.old_path)
        for ln in fp.old_changed_lines:
            cands = [i for i in by_file[fp.old_path] if chunks[i].node_type != "module" and chunks[i].start_line <= ln <= chunks[i].end_line]
            cands = [i for i in cands if chunks[i].node_type != "class_header"] or cands
            if cands:
                gold.add(min(cands, key=lambda i: chunks[i].end_line - chunks[i].start_line))
            else:
                gold.update(i for i in by_file[fp.old_path] if chunks[i].node_type == "module")
    return gold, gold_files


def budget_pack(chunks, ranking, budget=BUDGET, cap=1200):
    used, chosen = 0, []
    for i in ranking:
        t = min(approx_tokens(chunks[i].raw_text) + 12, cap)
        if used + t > budget:
            if used >= budget - 50:
                break
            continue
        chosen.append(i)
        used += t
    return chosen, used


def metrics(chunks, ranking, gold, gold_files):
    ctx, used = budget_pack(chunks, ranking)
    first = next((k + 1 for k, i in enumerate(ranking[:2000]) if i in gold), None)
    files, seen = [], set()
    for i in ranking[:2000]:
        p = chunks[i].file_path
        if p not in seen:
            seen.add(p); files.append(p)
    ffirst = next((k + 1 for k, p in enumerate(files) if p in gold_files), None)
    return {"chunk_rank": first, "file_rank": ffirst, "budget_hit": any(i in gold for i in ctx), "ctx_tokens": used}


class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb_cache (text_sha TEXT, model TEXT, vector BLOB, PRIMARY KEY (text_sha, model))")

    def get(self, model, shas):
        out = {}
        for i in range(0, len(shas), 900):
            q = shas[i:i + 900]
            for s, v in self.db.execute(f"SELECT text_sha, vector FROM emb_cache WHERE model=? AND text_sha IN ({','.join('?' * len(q))})", [model, *q]):
                out[s] = np.frombuffer(v, dtype=np.float32)
        return out

    def put(self, model, items):
        self.db.executemany("INSERT OR REPLACE INTO emb_cache VALUES (?,?,?)", [(s, model, v.astype(np.float32).tobytes()) for s, v in items])
        self.db.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--models", default="minilm,codebert,graphcodebert,jina-code")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default=None, help="force onnx|torch for all models")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--time-budget", type=float, default=1e9)
    a = ap.parse_args()
    rows = sorted(pq.read_table(a.parquet).to_pylist(), key=lambda r: (r["repo"], r["instance_id"]))
    si, sn = (int(x) for x in a.shard.split("/"))
    rows = rows[si::sn][: a.limit or None]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["instance_id"] for l in open(a.out) if l.strip()}
    models = a.models.split(",")
    emb = {}
    for m in models:
        kw = {"device": a.device}
        if a.backend:
            kw["backend"] = a.backend
        if m == "jina-code":  # its remote modeling code is incompatible with current transformers; use the official ONNX export
            kw["backend"] = "onnx"
        emb[m] = Embedder(m, **kw)
        if a.device.startswith("cuda") and emb[m].backend == "torch":
            emb[m].model.half()
    cache = Cache(a.cache)
    thr = {m: {"embedded": 0, "embed_seconds": 0.0, "cache_hits": 0} for m in models}
    start = time.time()
    with open(a.out, "a") as f:
        for row in rows:
            if row["instance_id"] in done:
                continue
            if time.time() - start > a.time_budget:
                break
            t0 = time.perf_counter()
            files = read_tree(ensure_mirror(row["repo"], a.mirror), row["base_commit"])
            chunks = []
            for p, s in files.items():
                try:
                    chunks += chunk_file(s, p)
                except Exception:
                    pass
            t_chunk = time.perf_counter() - t0
            gold, gold_files = gold_chunks(chunks, row["patch"])
            rec = {"instance_id": row["instance_id"], "repo": row["repo"], "n_chunks": len(chunks), "n_gold": len(gold),
                   "t_read_chunk_s": t_chunk}
            if not gold:
                rec["skip"] = "no_gold_chunk"
                f.write(json.dumps(rec) + "\n"); continue
            texts = [chunk_text_for_embedding(c) for c in chunks]
            issue = row["problem_statement"]
            t0 = time.perf_counter()
            bm = ChunkBM25(texts)
            r_bm = bm.rank(issue)
            r_bm += [i for i in range(len(chunks)) if i not in set(r_bm)]
            rec["t_bm25_s"] = time.perf_counter() - t0
            rec["bm25"] = metrics(chunks, r_bm, gold, gold_files)
            shas = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
            for m in models:
                e = emb[m]
                have = cache.get(m, shas)
                todo = sorted({s for s in shas if s not in have})
                thr[m]["cache_hits"] += len(shas) - len(todo)
                if todo:
                    idx = {s: shas.index(s) for s in todo}
                    st = {}
                    vec = e.embed([texts[idx[s]] for s in todo], batch=a.batch, stats=st)
                    thr[m]["embedded"] += st["embedded"]; thr[m]["embed_seconds"] += st["embed_seconds"]
                    cache.put(m, zip(todo, vec))
                    have.update(zip(todo, vec))
                mat = np.stack([have[s] for s in shas])
                t0 = time.perf_counter()
                q = e.embed([issue])[0]
                sims = mat @ q
                r_d = list(np.argsort(-sims, kind="stable"))
                rec[f"dense_{m}"] = metrics(chunks, r_d, gold, gold_files)
                rec[f"rrf_bm25_{m}"] = metrics(chunks, rrf(r_bm, r_d), gold, gold_files)
                rec[f"t_query_{m}_s"] = time.perf_counter() - t0
            f.write(json.dumps(rec, default=int) + "\n"); f.flush()
            print(row["instance_id"], len(chunks), {k: v["budget_hit"] for k, v in rec.items() if isinstance(v, dict)}, flush=True)
    with open(a.out + ".throughput.json", "w") as g:
        json.dump({m: dict(v, device=a.device, backend=emb[m].backend,
                           chunks_per_s=(v["embedded"] / v["embed_seconds"]) if v["embed_seconds"] else None) for m, v in thr.items()}, g, indent=1)


if __name__ == "__main__":
    main()
