
# ---------------- E5 driver: dense-embedder benchmark on all 500 SWE-bench Verified tasks, T4x2 ----------------
import json, subprocess, time, urllib.request, hashlib, glob, platform
T0 = time.time()
W = "/kaggle/working"; TMP = "/kaggle/tmp"
os.makedirs(TMP, exist_ok=True)
log = open(f"{W}/e5_driver.log", "a")
def L(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tree-sitter==0.26.1", "tree-sitter-python==0.25.0",
                "pyarrow"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu==1.23.2"], check=False)
ONLY = {"minilm", "codebert", "graphcodebert", "jina-code"}
PQ = f"{TMP}/verified.parquet"
urllib.request.urlretrieve("https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/data/test-00000-of-00001.parquet", PQ)
L("parquet sha256", hashlib.sha256(open(PQ, "rb").read()).hexdigest())
import pyarrow.parquet as pq
repos = sorted({r["repo"] for r in pq.read_table(PQ).to_pylist()})
MIR = f"{TMP}/mirrors"; os.makedirs(MIR, exist_ok=True)
procs = []
for r in repos:  # clone all mirrors in parallel
    dst = f"{MIR}/{r.replace('/', '__')}.git"
    if not os.path.isdir(dst):
        procs.append(subprocess.Popen(["git", "clone", "-q", "--bare", "--filter=blob:limit=300k", f"https://github.com/{r}.git", dst]))
for p in procs:
    p.wait()
L("mirrors ready", round(time.time() - T0), "s")
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception:
    gpu = "?"
env_info = {"gpu": gpu, "python": platform.python_version(), "cpus": os.cpu_count()}
L("env", env_info)

PLAN = {0: [m for m in ["minilm", "codebert", "graphcodebert", "jina-code"] if m in ONLY]}
def run_gpu(g, models):
    out = []
    for m in models:
        o = f"{W}/e5_{m}.jsonl"
        cmd = [sys.executable, f"{CODE_DIR}/experiments/exp_dense.py", "--parquet", PQ, "--mirror", MIR, "--models", m,
               "--device", "cuda:0", *([] if m == "jina-code" else ["--backend", "torch"]), "--cache", f"{TMP}/emb_{m}.sqlite", "--out", o, "--batch", "64"]
        e = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        t = time.time()
        r = subprocess.run(cmd, env=e, capture_output=True, text=True)
        open(f"{W}/e5_{m}.stderr.log", "w").write(r.stderr[-20000:])
        L(m, "rc", r.returncode, "wall_s", round(time.time() - t), "lines", sum(1 for _ in open(o)) if os.path.exists(o) else 0)
        out.append((m, r.returncode))
    return out

import threading
res = {}
ths = [threading.Thread(target=lambda g=g: res.__setitem__(g, run_gpu(g, PLAN[g]))) for g in PLAN]
for t in ths: t.start()
for t in ths: t.join()
json.dump({"env": env_info, "results": res, "wall_s": time.time() - T0}, open(f"{W}/e5_run_info.json", "w"), indent=1)
sums = {}
for p in sorted(glob.glob(f"{W}/*")):
    if os.path.isfile(p) and not p.endswith(".log"):
        sums[os.path.basename(p)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(sums, open(f"{W}/SHA256SUMS.json", "w"), indent=1)
L("done", round(time.time() - T0), "s")
