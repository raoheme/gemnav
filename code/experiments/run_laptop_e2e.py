"""Laptop end-to-end run (inside the gemnav-agent Linux container, Docker socket mounted).

Per task (sequential, one task image on disk at a time):
  1. pull the official instance image, start the task container (DockerEnv), copy /testbed out
  2. index (tree-sitter + SQLite + dense embeddings, content-cached across tasks) + pylsp
  3. gemnav guarded repair with Gemma 4 E2B served by llama-server on the Windows host
  4. official SWE-bench harness (v5.0.1 venv from the swe-paper-swebench volume) scores, for this instance:
     gold patch (environment validity), the E2B prediction, and the E4B prediction if present
  5. remove the task container and the instance image (env/base images kept); record disk before/after
Resumable: tasks already in the stats file are skipped. All timestamps are UTC epoch seconds so the Windows-side
RAM sampler CSV can be aligned with agent events.

python run_laptop_e2e.py --tasks /work/results/e2e_tasks50.json --server http://host.docker.internal:8080 \
   --out /work/results/laptop --embedder minilm [--e4b-preds /work/results/kaggle/predictions_e4b.jsonl]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pyarrow.parquet as pq  # noqa: E402

from gemnav.client import LlamaServerClient  # noqa: E402
from gemnav.envs import DockerEnv, image_for  # noqa: E402
from gemnav.repair import Index, RepairAgent  # noqa: E402

SWEBENCH_PY = os.environ.get("SWEBENCH_PY", "/swebench/swe-paper-eval/SWE-bench/.venv/bin/python")
SWEBENCH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SWEBENCH_PY)))
DATASET = "SWE-bench/SWE-bench_Verified"


def docker_images_gb() -> float:
    o = subprocess.run(["docker", "system", "df", "--format", "{{.Type}} {{.Size}}"], capture_output=True, text=True, timeout=120).stdout
    for line in o.splitlines():
        if line.startswith("Images"):
            v = line.split()[1]
            mult = {"GB": 1, "MB": 1e-3, "kB": 1e-6, "B": 1e-9, "TB": 1e3}
            for u, m in mult.items():
                if v.endswith(u):
                    return float(v[: -len(u)]) * m
    return -1.0


def score(iid: str, preds: dict[str, str], outdir: str) -> dict:
    """Run the official harness once per prediction set for this instance; return {name: resolved|None}."""
    res = {}
    for name, patch in preds.items():
        run_id = f"gemnav-{name}-{iid}"
        pf = os.path.join(outdir, "preds_tmp", f"{run_id}.jsonl")
        os.makedirs(os.path.dirname(pf), exist_ok=True)
        with open(pf, "w") as f:
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": name, "model_patch": patch}) + "\n")
        if not patch.strip():
            res[name] = {"resolved": False, "empty_patch": True}
            continue
        t = time.time()
        r = subprocess.run([SWEBENCH_PY, "-m", "swebench.harness.run_evaluation", "--dataset_name", DATASET,
                            "--predictions_path", pf, "--max_workers", "1", "--run_id", run_id, "--instance_ids", iid,
                            "--timeout", "1800"], cwd=SWEBENCH_DIR, capture_output=True, text=True, timeout=5400)
        rep = glob.glob(os.path.join(SWEBENCH_DIR, "logs", "run_evaluation", run_id, name, iid, "report.json"))
        entry = {"harness_rc": r.returncode, "eval_s": time.time() - t}
        if rep:
            d = json.load(open(rep[0]))[iid]
            entry.update(resolved=bool(d.get("resolved")), patch_applied=d.get("patch_successfully_applied"),
                         f2p=d.get("tests_status", {}).get("FAIL_TO_PASS"), p2p_fail=len(d.get("tests_status", {}).get("PASS_TO_PASS", {}).get("failure", [])))
            shutil.copy(rep[0], os.path.join(outdir, "reports", f"{run_id}.report.json"))
        else:
            entry.update(resolved=None, harness_tail=(r.stdout + r.stderr)[-1500:])
        res[name] = entry
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--parquet", default="/cache/verified.parquet")
    ap.add_argument("--server", default="http://host.docker.internal:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--embedder", default="minilm")
    ap.add_argument("--e4b-preds", default="")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--wall-s", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="gemma4-e2b-qat-q4_0-gemnav-laptop")
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--rescore-e4b", action="store_true", help="only score E4B predictions for tasks already run here")
    ap.add_argument("--score-only", action="store_true", help="no agent: score gold + E4B predictions for every task")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "reports"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "traj"), exist_ok=True)
    if not os.path.exists(a.parquet):
        import urllib.request
        os.makedirs(os.path.dirname(a.parquet), exist_ok=True)
        urllib.request.urlretrieve("https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet", a.parquet)
    rows = {r["instance_id"]: r for r in pq.read_table(a.parquet).to_pylist()}
    tasks = json.load(open(a.tasks))["tasks"][: a.limit or None]
    e4b = {}
    if a.e4b_preds and os.path.exists(a.e4b_preds):
        e4b = {json.loads(l)["instance_id"]: json.loads(l)["model_patch"] for l in open(a.e4b_preds) if l.strip()}
    stats_path = os.path.join(a.out, "e2e_laptop_tasks.jsonl")
    done = {json.loads(l)["instance_id"] for l in open(stats_path)} if os.path.exists(stats_path) else set()
    if a.score_only:
        name = "gemma4-e4b-qat-q4_0-gemnav"
        sp = os.path.join(a.out, "score_only_e4b.jsonl")
        have = {json.loads(l)["instance_id"] for l in open(sp)} if os.path.exists(sp) else set()
        for t in tasks:
            iid = t["instance_id"]
            if iid in have or iid not in e4b:
                continue
            t0 = time.time()
            preds = {"gold": rows[iid]["patch"], name: e4b[iid]}
            try:
                sc = score(iid, preds, a.out)
            except Exception as ex:
                sc = {"error": f"{type(ex).__name__}: {ex}"[:300]}
            if not a.keep_images:
                subprocess.run(["docker", "rmi", "-f", image_for(iid)], capture_output=True, timeout=600)
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, timeout=600)
            with open(sp, "a") as f:
                f.write(json.dumps({"instance_id": iid, "hop": t["hop"], "repo": t["repo"], "scores": sc, "t": time.time() - t0,
                                    "images_gb_after": docker_images_gb()}) + "\n")
            print("scored", iid, {k: (v or {}).get("resolved") for k, v in sc.items() if isinstance(v, dict)}, round(time.time() - t0), "s", flush=True)
        return
    if a.rescore_e4b:
        name = "gemma4-e4b-qat-q4_0-gemnav"
        recs = [json.loads(l) for l in open(stats_path)] if os.path.exists(stats_path) else []
        extra = os.path.join(a.out, "e2e_laptop_rescore_e4b.jsonl")
        have = {json.loads(l)["instance_id"] for l in open(extra)} if os.path.exists(extra) else set()
        for r in recs:
            iid = r["instance_id"]
            if iid in have or iid not in e4b or name in r.get("scores", {}):
                continue
            t0 = time.time()
            sc = score(iid, {name: e4b[iid]}, a.out)
            if not a.keep_images:
                subprocess.run(["docker", "rmi", "-f", image_for(iid)], capture_output=True, timeout=600)
                subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, timeout=600)
            with open(extra, "a") as f:
                f.write(json.dumps({"instance_id": iid, "scores": sc, "t": time.time() - t0}) + "\n")
            print("rescored", iid, sc[name].get("resolved"), flush=True)
        return
    emb = None
    if a.embedder != "none":
        from gemnav.embedder import Embedder
        emb = Embedder(a.embedder, backend="onnx" if a.embedder == "minilm" else None, threads=int(os.environ.get("EMB_THREADS", "4")),
                       cache_dir=os.environ.get("HF_CACHE", "/cache/hf"))
    client = LlamaServerClient(a.server, n_predict=1536)
    for t in tasks:
        iid = t["instance_id"]
        if iid in done:
            continue
        row = rows[iid]
        rec = {"instance_id": iid, "t_start": time.time(), "hop": t["hop"], "repo": t["repo"], "images_gb_before": docker_images_gb()}
        env = None
        try:
            t0 = time.time()
            env = DockerEnv(iid, "/scratch", mem=os.environ.get("TASK_MEM", "2g"))
            rec["env"] = dict(env.timings, setup_s=time.time() - t0)
            t0 = time.time()
            ix = Index(env.workdir, "/scratch/ix.db", embedder=emb, cache_path="/cache/embcache.sqlite", lsp=True)
            rec["index"] = ix.stats
            n0 = len(client.calls)
            res = RepairAgent(env, row["problem_statement"], client, ix, max_steps=a.max_steps, wall_s=a.wall_s,
                              ctx_budget=7000).run()
            ix.close()
            rec["llm_calls"] = client.calls[n0:]
            rec.update({k: v for k, v in res.items() if k not in ("model_patch", "events")})
            json.dump(res["events"], open(os.path.join(a.out, "traj", f"{iid}.json"), "w"))
            patch = res["model_patch"]
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"[:500]
            rec["trace"] = traceback.format_exc()[-2000:]
            patch = ""
        finally:
            if env is not None:
                env.close(remove_image=False)
        with open(os.path.join(a.out, f"predictions_{a.tag}.jsonl"), "a") as f:
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": a.tag, "model_patch": patch}) + "\n")
        rec["agent_end"] = time.time()
        preds = {"gold": row["patch"], a.tag: patch}
        if iid in e4b:
            preds["gemma4-e4b-qat-q4_0-gemnav"] = e4b[iid]
        try:
            rec["scores"] = score(iid, preds, a.out)
        except Exception as ex:
            rec["scores_error"] = f"{type(ex).__name__}: {ex}"[:300]
        if not a.keep_images:
            subprocess.run(["docker", "rmi", "-f", image_for(iid)], capture_output=True, timeout=600)
            subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, timeout=600)
        rec["images_gb_after"] = docker_images_gb()
        rec["t_end"] = time.time()
        with open(stats_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(iid, "patch_lines", patch.count("\n"), {k: (v or {}).get("resolved") for k, v in rec.get("scores", {}).items()},
              "wall", round(rec["t_end"] - rec["t_start"]), flush=True)


if __name__ == "__main__":
    main()
