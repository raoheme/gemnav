"""Blueprint performance targets, measured on the machine it runs on.

Targets (from the blueprint, "8 GB RAM, CPU-only"):
  index 10K files ~3 min (chunking + embedding + symbols) | vector search < 5 ms (10K vectors)
  LSP definition < 50 ms (cached) | ripgrep < 100 ms (10K files)
Stages are timed separately; peak RSS of this process is sampled every 0.2 s (psutil).

python bench_index.py --repo /scratch/core --embedder minilm --out /work/results/laptop/bench_index.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np  # noqa: E402
import psutil  # noqa: E402

from gemnav.chunker import chunk_file, symbols  # noqa: E402
from gemnav.vector_store import VectorStore  # noqa: E402

PEAK = {"rss_mib": 0}


def sampler(stop):
    p = psutil.Process()
    while not stop.is_set():
        try:
            rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
            PEAK["rss_mib"] = max(PEAK["rss_mib"], rss // 1048576)
        except Exception:
            pass
        time.sleep(0.2)


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--embedder", default="minilm")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    stop = threading.Event()
    threading.Thread(target=sampler, args=(stop,), daemon=True).start()
    R = {"repo": a.repo, "embedder": a.embedder, "threads": a.threads, "cpu": os.cpu_count(),
         "ram_total_mib": psutil.virtual_memory().total // 1048576}
    try:
        R["repo_commit"] = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    t0 = time.perf_counter()
    files = []
    for dp, dn, fn in os.walk(a.repo):
        dn[:] = [d for d in dn if d not in (".git", "node_modules", "__pycache__")]
        files += [os.path.join(dp, f) for f in fn]
    py = [f for f in files if f.endswith(".py")]
    R["files_total"], R["files_py"] = len(files), len(py)
    R["walk_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    chunks = []
    for f in py:
        try:
            chunks += chunk_file(open(f, "rb").read(), os.path.relpath(f, a.repo).replace(os.sep, "/"))
        except Exception:
            pass
    R["chunks"] = len(chunks)
    R["chunk_s"] = time.perf_counter() - t0
    db = os.path.join(os.path.dirname(a.out) or ".", "bench_index.db")
    if os.path.exists(db):
        os.remove(db)
    t0 = time.perf_counter()
    vs = VectorStore(db)
    vs.add_chunks(chunks, symbols(chunks), {})
    R["sqlite_insert_s"] = time.perf_counter() - t0
    R["rss_after_chunk_mib"] = PEAK["rss_mib"]
    from gemnav.embedder import Embedder
    t0 = time.perf_counter()
    e = Embedder(a.embedder, backend="onnx" if a.embedder == "minilm" else None, threads=a.threads, cache_dir=os.environ.get("HF_CACHE", "/cache/hf"))
    R["embedder_load_s"] = time.perf_counter() - t0
    st = vs.embed_all(e, batch=a.batch)
    R["embed"] = st
    R["embed_chunks_per_s"] = st.get("embedded", 0) / st["embed_seconds"] if st.get("embed_seconds") else None
    R["index_total_s"] = R["walk_s"] + R["chunk_s"] + R["sqlite_insert_s"] + st["total_seconds"]
    # vector search latency (exact numpy cosine over all chunks; 200 random queries)
    rng = random.Random(0)
    qs = [e.embed([chunks[rng.randrange(len(chunks))].raw_text[:500]])[0] for _ in range(50)]
    lat = []
    for q in qs * 4:
        t = time.perf_counter(); vs.search(e.name, q, 10); lat.append((time.perf_counter() - t) * 1000)
    R["vector_search_ms"] = {"p50": pct(lat, 50), "p95": pct(lat, 95), "n_vectors": len(chunks)}
    # ripgrep latency over the whole repo, 20 identifier queries
    names = [c.identifier for c in rng.sample(chunks, min(20, len(chunks))) if c.identifier]
    lat = []
    for n in names:
        t = time.perf_counter()
        subprocess.run(["rg", "-n", "-m", "5", "--no-heading", "-e", n, a.repo], capture_output=True, timeout=120)
        lat.append((time.perf_counter() - t) * 1000)
    R["ripgrep_ms"] = {"p50": pct(lat, 50), "p95": pct(lat, 95), "n": len(lat)}
    # LSP definition latency (first call = cold, then repeated = warm/cached)
    from gemnav.lsp import PylspClient
    t = time.perf_counter(); lsp = PylspClient(a.repo, timeout=120); R["lsp_init_s"] = time.perf_counter() - t
    cold, warm = [], []
    funcs = [c for c in chunks if c.node_type in ("function", "method")]
    for c in rng.sample(funcs, min(10, len(funcs))):
        src = open(os.path.join(a.repo, c.file_path), errors="replace").read().splitlines()
        line = src[c.start_line - 1] if c.start_line - 1 < len(src) else ""
        col = line.find(c.identifier)
        if col < 0:
            continue
        try:
            t = time.perf_counter(); lsp.definition(c.file_path, c.start_line, col); cold.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter(); lsp.definition(c.file_path, c.start_line, col); warm.append((time.perf_counter() - t) * 1000)
        except Exception as ex:
            R.setdefault("lsp_errors", []).append(str(ex)[:100])
    lsp.close()
    R["lsp_definition_ms"] = {"cold_p50": pct(cold, 50), "warm_p50": pct(warm, 50), "warm_p95": pct(warm, 95), "n": len(warm)}
    stop.set()
    R["peak_rss_mib"] = PEAK["rss_mib"]
    R["targets"] = {"index_10k_files_le_180s": R["index_total_s"] <= 180, "vector_search_p50_lt_5ms": (R["vector_search_ms"]["p50"] or 1e9) < 5,
                    "lsp_warm_p50_lt_50ms": (R["lsp_definition_ms"]["warm_p50"] or 1e9) < 50,
                    "ripgrep_p50_lt_100ms": (R["ripgrep_ms"]["p50"] or 1e9) < 100}
    json.dump(R, open(a.out, "w"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
