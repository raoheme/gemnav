"""Summarise end-to-end runs: resolve rate (official harness), latency (llama-server timings), memory (samplers).

python summarize_e2e.py results/laptop [results/kaggle]
Writes <dir>/summary_e2e.json. Resolve rates are reported over (a) all sampled tasks and (b) tasks whose gold patch
resolves in the same environment (environment-valid denominator), with Wilson 95% intervals.
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def q(xs, p):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else None


def latency(recs):
    calls = [c for r in recs for c in r.get("llm_calls", [])]
    pp = [c["prompt_n"] / (c["prompt_ms"] / 1000) for c in calls if c.get("prompt_ms") and c.get("prompt_n")]
    gg = [c["predicted_n"] / (c["predicted_ms"] / 1000) for c in calls if c.get("predicted_ms") and c.get("predicted_n")]
    wall = [c["wall_s"] for c in calls if c.get("wall_s") is not None]
    newtok = [c.get("prompt_n") for c in calls]
    cached = [c.get("cache_n") for c in calls if c.get("cache_n") is not None]
    return {"llm_calls": len(calls), "call_wall_s_p50": q(wall, 50), "call_wall_s_p95": q(wall, 95),
            "prompt_tok_s_p50": q(pp, 50), "gen_tok_s_p50": q(gg, 50), "new_prompt_tokens_p50": q(newtok, 50),
            "cached_tokens_p50": q(cached, 50),
            "task_agent_wall_s_p50": q([r.get("wall_s") for r in recs], 50), "task_agent_wall_s_p95": q([r.get("wall_s") for r in recs], 95),
            "steps_mean": float(np.mean([r.get("steps", 0) for r in recs])) if recs else None}


def main(d, kaggle_dir=None):
    recs = [json.loads(l) for l in open(os.path.join(d, "e2e_laptop_tasks.jsonl"))] if os.path.exists(os.path.join(d, "e2e_laptop_tasks.jsonl")) else []
    extra = os.path.join(d, "e2e_laptop_rescore_e4b.jsonl")
    if os.path.exists(extra):
        add = {json.loads(l)["instance_id"]: json.loads(l)["scores"] for l in open(extra)}
        for r in recs:
            r.setdefault("scores", {}).update(add.get(r["instance_id"], {}))
    S = {"n_tasks": len(recs), "errors": sum(1 for r in recs if r.get("error"))}
    models = sorted({m for r in recs for m in r.get("scores", {})})
    gold_ok = {r["instance_id"] for r in recs if r.get("scores", {}).get("gold", {}).get("resolved")}
    S["gold_resolved"] = len(gold_ok)
    S["resolve"] = {}
    for m in models:
        if m == "gold":
            continue
        scored = [r for r in recs if m in r.get("scores", {})]
        k = sum(1 for r in scored if r["scores"][m].get("resolved"))
        kv = sum(1 for r in scored if r["instance_id"] in gold_ok and r["scores"][m].get("resolved"))
        nv = sum(1 for r in scored if r["instance_id"] in gold_ok)
        S["resolve"][m] = {"n": len(scored), "resolved": k, "pct": 100 * k / len(scored) if scored else None, "ci95": wilson(k, len(scored)),
                           "env_valid_n": nv, "env_valid_resolved": kv, "env_valid_pct": 100 * kv / nv if nv else None,
                           "env_valid_ci95": wilson(kv, nv),
                           "empty_patches": sum(1 for r in scored if r["scores"][m].get("empty_patch")),
                           "patch_apply_fail": sum(1 for r in scored if r["scores"][m].get("patch_applied") is False),
                           "harness_unknown": sum(1 for r in scored if r["scores"][m].get("resolved") is None)}
        by = {}
        for h in ("0", "1", "2+", "unanchored"):
            sub = [r for r in scored if r.get("hop") == h]
            by[h] = [sum(1 for r in sub if r["scores"][m].get("resolved")), len(sub)]
        S["resolve"][m]["by_hop_resolved_n"] = by
    ok = [r for r in recs if not r.get("error")]
    S["latency_e2b"] = latency(ok)
    S["agent"] = {k: int(sum(r.get(k, 0) or 0 for r in ok)) for k in ("edits_committed", "edits_rolled_back", "guard_rejects", "duplicates", "protocol_errors")}
    S["repro_status"] = {s: sum(1 for r in ok if r.get("repro_status") == s) for s in ("ALIGNED", "PASSES_ON_BUGGY", "BROKEN", "NO_SCRIPT")}
    S["index_s_p50"] = q([sum(v for k, v in r.get("index", {}).items() if k.endswith("_s")) for r in ok], 50)
    emb = [r.get("index", {}).get("embed", {}) for r in ok]
    S["embed_cache_hit_rate"] = (sum(e.get("cache_hits", 0) for e in emb) / max(1, sum(e.get("chunks", 0) for e in emb)))
    csvp = os.path.join(d, "ram_samples.csv")
    if os.path.exists(csvp):
        rows = list(csv.DictReader(open(csvp)))
        def mx(k):
            v = [float(r[k]) for r in rows if r.get(k) not in (None, "")]
            return max(v) if v else None
        S["memory_mib"] = {"sys_used_max": mx("sys_used_mib"), "sys_total": mx("sys_total_mib"), "llama_peak_ws_max": mx("llama_peak_ws_mib"),
                           "llama_ws_max": mx("llama_ws_mib"), "docker_vm_ws_max": mx("vmmem_ws_mib"), "containers_max": mx("docker_containers_mib"),
                           "samples": len(rows)}
    for f in ("speed_gate.json", "bench_index.json", "hardware.json"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            S[f[:-5]] = json.load(open(p, encoding="utf-8-sig"))
    if kaggle_dir:
        kr = [json.loads(l) for f in glob.glob(os.path.join(kaggle_dir, "e2e_*_tasks.jsonl")) for l in open(f)]
        info = [json.load(open(f)) for f in glob.glob(os.path.join(kaggle_dir, "run_info_*.json"))]
        S["kaggle_e4b"] = {"n_tasks": len(kr), "errors": sum(1 for r in kr if r.get("error")), "latency": latency([r for r in kr if not r.get("error")]),
                           "run_info": info[0] if info else None}
    json.dump(S, open(os.path.join(d, "summary_e2e.json"), "w"), indent=1)
    print(json.dumps({k: S[k] for k in ("n_tasks", "errors", "gold_resolved", "resolve")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
