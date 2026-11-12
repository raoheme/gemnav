
# ---------------- E2E driver: gemnav guarded repair with Gemma 4 E4B (QAT q4_0) on Kaggle T4x2 ----------------
# One llama-server per T4, one worker per server. Each task runs in its official SWE-bench image unpacked as a
# rootfs (chroot/proot; Kaggle has no Docker). Predictions are scored later with the official Docker harness.
import concurrent.futures as cf, hashlib, json, queue, shutil, subprocess, tarfile, threading, time, traceback, urllib.request
T0 = time.time()
W = "/kaggle/working"; TMP = "/kaggle/tmp"; os.makedirs(TMP, exist_ok=True)
TASKS = json.loads(r'''__TASKS_JSON__''')
SKIP = set(json.loads(r'''__SKIP_JSON__'''))
EMBEDDER = "__EMBEDDER__"
TAG = "gemma4-e4b-qat-q4_0-gemnav"
TIME_LIMIT = 11.0 * 3600
LLAMA = [("cudart-llama-b12211-bin-ubuntu-cuda-12.8-x64.tar.gz", "d66dba34f0e45c66ce7d4484116c681aa71ff257ee1e47b02d9775aa59b5ef77"),
         ("llama-b12211-bin-ubuntu-cuda-12.8-x64.tar.gz", "103fac975e807da6cdb5717e87ca594e25bb7703dd8a7f10c22269c124e1a552")]
MODEL = ("google/gemma-4-E4B-it-qat-q4_0-gguf", "gemma-4-E4B_q4_0-it.gguf", "4b4a2c1d584be7264f87aac328a1bc739ce81b6c",
         "676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee")
SERVER_FLAGS = ["-ngl", "999", "-c", "8192", "--parallel", "1", "-fa", "on", "--jinja", "--seed", "42",
                "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--swa-full", "--host", "127.0.0.1"]
LOG = open(f"{W}/e2e_driver.log", "a")
_lk = threading.Lock()
def L(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    with _lk:
        print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tree-sitter==0.26.1", "tree-sitter-python==0.25.0", "onnxruntime==1.23.2",
                "python-lsp-server==1.15.0", "jedi", "psutil", "pyarrow"], check=False)
subprocess.run("apt-get install -y -q ripgrep >/dev/null 2>&1 || true", shell=True)
import psutil
# ---- llama.cpp (prebuilt, sha-verified) + model (pinned revision, sha-verified)
LD = f"{TMP}/llama"; os.makedirs(LD, exist_ok=True)
for name, dig in LLAMA:
    p = f"{LD}/{name}"
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"https://github.com/ggml-org/llama.cpp/releases/download/b12211/{name}", p)
    assert sha256(p) == dig, name
    tarfile.open(p).extractall(LD)
SRV = next(str(x) for x in __import__("pathlib").Path(LD).rglob("llama-server"))
os.chmod(SRV, 0o755)
from huggingface_hub import hf_hub_download
GG = hf_hub_download(MODEL[0], MODEL[1], revision=MODEL[2], cache_dir=f"{TMP}/hf")
assert sha256(GG) == MODEL[3], "model sha mismatch"
L("assets ok", round(time.time() - T0), "s")
import pathlib
LIBS = ":".join(sorted({str(p.parent) for p in pathlib.Path(LD).rglob("*.so*")}) + [os.environ.get("LD_LIBRARY_PATH", "")])
dev = subprocess.run([SRV, "--list-devices"], env=dict(os.environ, LD_LIBRARY_PATH=LIBS), capture_output=True, text=True)
L("list-devices", (dev.stdout + dev.stderr).strip()[-600:])
if dev.stdout.count("CUDA") < 2:
    raise SystemExit("FAIL-FAST: llama.cpp does not see 2 CUDA devices")
servers, t_load = [], {}
for g in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), LD_LIBRARY_PATH=LIBS)
    t = time.time()
    servers.append(subprocess.Popen([SRV, "-m", GG, "--port", str(8000 + g), "-t", "2", *SERVER_FLAGS], env=env,
                                    stdout=open(f"{W}/llama_server_{g}.log", "w"), stderr=subprocess.STDOUT))
    for _ in range(600):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{8000 + g}/health", timeout=5)).get("status") == "ok": break
        except Exception: pass
        time.sleep(1)
    t_load[g] = time.time() - t
L("servers up", t_load)

# ---- resource monitor (GPU memory per device, llama-server RSS, system RAM), 2 s period
MON = {"gpu_mem_mib_max": [0, 0], "server_rss_mib_max": [0, 0], "sys_ram_used_mib_max": 0, "samples": 0}
def monitor():
    while True:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split()
            for i, v in enumerate(o[:2]): MON["gpu_mem_mib_max"][i] = max(MON["gpu_mem_mib_max"][i], int(v))
            for i, p in enumerate(servers):
                MON["server_rss_mib_max"][i] = max(MON["server_rss_mib_max"][i], psutil.Process(p.pid).memory_info().rss // 1048576)
            MON["sys_ram_used_mib_max"] = max(MON["sys_ram_used_mib_max"], (psutil.virtual_memory().total - psutil.virtual_memory().available) // 1048576)
            MON["samples"] += 1
        except Exception: pass
        time.sleep(2)
threading.Thread(target=monitor, daemon=True).start()

# ---- speed gate (same prompt as the laptop gate)
from gemnav.client import LlamaServerClient
gate = {}
for g in (0, 1):
    c = LlamaServerClient(f"http://127.0.0.1:{8000 + g}", n_predict=128)
    c.chat("You are a helpful assistant.", "Explain what a Python decorator is in about 100 words. " * 20, grammar=None)
    x = c.calls[-1]
    gate[g] = {"prompt_n": x["prompt_n"], "prompt_tok_s": x["prompt_n"] / (x["prompt_ms"] / 1000), "gen_tok_s": x["predicted_n"] / (x["predicted_ms"] / 1000)}
L("speed gate", gate)

# ---- data + code
import pyarrow.parquet as pq
PQ = f"{TMP}/verified.parquet"
urllib.request.urlretrieve("https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet", PQ)
ROWS = {r["instance_id"]: r for r in pq.read_table(PQ).to_pylist()}
exec(open(f"{CODE_DIR}/kaggle/rootfs.py").read())          # defines pull_image, Sandbox
from gemnav.envs import RootfsEnv, image_for
from gemnav.repair import Index, RepairAgent
emb = None
if EMBEDDER != "none":
    from gemnav.embedder import Embedder
    # GPU 0 hosts a llama-server too; MiniLM (~90 MB) fits alongside. Same weights as the laptop's ONNX model.
    emb = Embedder(EMBEDDER, backend="torch", device="cuda:0")
    if hasattr(emb, "model"): emb.model.half()
    _emb_lock = threading.Lock(); _embed = emb.embed
    def _locked_embed(*a, **k):
        with _emb_lock:
            return _embed(*a, **k)
    emb.embed = _locked_embed

PRED = open(f"{W}/predictions_{TAG}.jsonl", "a"); STAT = open(f"{W}/e2e_{TAG}_tasks.jsonl", "a")
q = queue.Queue()
for t in TASKS:
    if t["instance_id"] not in SKIP: q.put(t["instance_id"])

def worker(g):
    client = LlamaServerClient(f"http://127.0.0.1:{8000 + g}", n_predict=1536)
    while time.time() - T0 < TIME_LIMIT:
        try: iid = q.get_nowait()
        except queue.Empty: return
        row, t_task = ROWS[iid], time.time()
        root = f"{TMP}/rootfs/{iid}"
        rec = {"instance_id": iid, "gpu": g}
        try:
            t = time.time(); info = pull_image(image_for(iid), root, cache_dir=f"{TMP}/blobs_{g}"); rec["pull_s"] = time.time() - t
            sb = Sandbox(root); env = RootfsEnv(sb)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", env.workdir])
            subprocess.run(["git", "-C", env.workdir, "-c", "core.fileMode=false", "status", "-s"], capture_output=True)
            t = time.time()
            ix = Index(env.workdir, f"{TMP}/ix_{g}.db", embedder=emb, cache_path=f"{TMP}/embcache_{g}.sqlite", lsp=True)
            rec["index"] = ix.stats; rec["index_s"] = time.time() - t
            n0 = len(client.calls)
            agent = RepairAgent(env, row["problem_statement"], client, ix, max_steps=15, wall_s=1800, ctx_budget=7000,
                                log=lambda *a: L(iid, *a))
            res = agent.run()
            ix.close()
            calls = client.calls[n0:]
            rec.update({k: v for k, v in res.items() if k not in ("model_patch", "events")})
            rec["llm_calls"] = calls
            rec["patch_lines"] = res["model_patch"].count("\n")
            json.dump(res["events"], open(f"{W}/traj_{iid}.json", "w"))
            with _lk:
                PRED.write(json.dumps({"instance_id": iid, "model_name_or_path": TAG, "model_patch": res["model_patch"]}) + "\n"); PRED.flush()
        except Exception as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"[:500]; rec["trace"] = traceback.format_exc()[-1500:]
            with _lk:
                PRED.write(json.dumps({"instance_id": iid, "model_name_or_path": TAG, "model_patch": ""}) + "\n"); PRED.flush()
        finally:
            try: sb.close()
            except Exception: pass
            shutil.rmtree(root, ignore_errors=True)
        rec["task_wall_s"] = time.time() - t_task
        with _lk:
            STAT.write(json.dumps(rec, default=str) + "\n"); STAT.flush()
        L("DONE", iid, "patch_lines", rec.get("patch_lines"), "err", rec.get("error", "")[:80], "wall", round(rec["task_wall_s"]))

with cf.ThreadPoolExecutor(2) as ex:
    list(ex.map(worker, (0, 1)))
remaining = []
while not q.empty(): remaining.append(q.get())
json.dump({"tag": TAG, "model": MODEL, "server_flags": SERVER_FLAGS, "llama_tag": "b12211", "embedder": EMBEDDER,
           "load_s": t_load, "speed_gate": gate, "monitor": MON, "remaining": remaining,
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout,
           "cpus": os.cpu_count(), "ram_total_mib": psutil.virtual_memory().total // 1048576, "wall_s": time.time() - T0},
          open(f"{W}/run_info_{TAG}.json", "w"), indent=1)
for p in servers: p.terminate()
sums = {os.path.basename(p): sha256(p) for p in sorted(__import__("glob").glob(f"{W}/*")) if os.path.isfile(p) and not p.endswith(".log")}
json.dump(sums, open(f"{W}/SHA256SUMS.json", "w"), indent=1)
L("finished", round(time.time() - T0), "s; remaining", len(remaining))
