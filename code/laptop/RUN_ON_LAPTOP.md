# Laptop run — Gemma 4 E2B, gemnav guarded repair, official SWE-bench scoring

Machine: Ryzen 5 5500U (6C/12T, Radeon Vega 7 iGPU), 8 GB RAM, Docker Desktop. Expected wall time: roughly one to two
days for 50 tasks (agent + three harness evaluations per task: gold, E2B, E4B). The run is **resumable** — rerun the
same command after a reboot, sleep or crash and it continues where it stopped.

## Before you start
1. Docker Desktop running; at least 25 GB free on C: (you have ~60 GB). Disable sleep while plugged in.
2. Close browsers/IDEs if you can — RAM is the bottleneck (model ~3.2 GB + WSL VM for task containers).
3. Your `swe_paper` folder must sit next to `Final_Paper` (it does). The run reuses
   `swe_paper\local_runs\tools\` (E2B model, llama.cpp CPU build, `windows_ca.pem`) and the Docker volume
   `swe-paper-swebench` (the SWE-bench v5.0.1 venv bootstrapped by your earlier setup).
4. If Windows Firewall asks about `llama-server.exe`, allow **Private networks** — the agent container reaches it via
   `host.docker.internal:8080`.

## Run
```powershell
cd C:\eurykosmotron-knowledge-continuum\Final_Paper
powershell -ExecutionPolicy Bypass -File code\laptop\run_laptop_e2e.ps1 -Stage all -Limit 2   # 2-task smoke first
powershell -ExecutionPolicy Bypass -File code\laptop\run_laptop_e2e.ps1 -Stage all            # full 50 tasks
```
Stages: `gate` (CPU vs Vulkan speed + RAM, picks the faster backend) → `bench` (index a real 10K+-file repo:
home-assistant/core; times every blueprint target) → `e2e` (50 tasks) → `summary`.

## What gets written (all under `results\laptop\`)
`hardware.json`, `speed_gate.json`, `bench_index.json`, `ram_samples.csv` (every ~5 s), `predictions_*.jsonl`,
`e2e_laptop_tasks.jsonl` (per task: steps, LLM timings from llama-server, edits/rollbacks, harness verdicts),
`reports\*.report.json` (official harness reports), `traj\*.json`, `summary_e2e.json`, `run.log`.
Nothing is uploaded anywhere; tell me when it finishes (or paste `run.log` if it stops) and I analyse the files.

## If something fails
* Vulkan download blocked: download `llama-b12211-bin-win-vulkan-x64.zip` from the llama.cpp b12211 release page in
  your browser into `swe_paper\local_runs\tools\`, rerun. (SHA-256 is verified.)
* `swe-paper-swebench` volume missing: run `swe_paper\scripts\windows\swebench.ps1 -Action bootstrap` once with
  `SWEBENCH_RUNNER=docker`.
* Out of memory / very slow: in Docker Desktop → Settings → Resources, keep the WSL limit at 5 GB (your `.wslconfig`).
