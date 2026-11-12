# Laptop: run these now (deadline plan)

Open PowerShell in `C:\eurykosmotron-knowledge-continuum\Final_Paper`. Docker Desktop must be running.

**1. Speed gate + 10K-file indexing benchmark (~1 h)** — start now:
```powershell
powershell -ExecutionPolicy Bypass -File code\laptop\run_laptop_e2e.ps1 -Stage bench
```
(The speed gate CPU vs Vulkan runs first automatically, then the indexing benchmark.)

**2. Gemma 4 E2B on 5 real tasks (~2-3 h)** — right after step 1:
```powershell
powershell -ExecutionPolicy Bypass -File code\laptop\run_laptop_e2e.ps1 -Stage e2e -Limit 5
```

**3. Official scoring of the Kaggle E4B patches (~3-6 h)** — when I tell you the file
`results\kaggle\predictions_gemma4-e4b-qat-q4_0-gemnav.jsonl` exists:
```powershell
powershell -ExecutionPolicy Bypass -File code\laptop\run_laptop_e2e.ps1 -Stage score
```
Each task image is deleted after its score is saved. Rerun the same command if anything stops; it resumes.

If a step fails, paste the last 30 lines of `results\laptop\run.log` into the chat.
