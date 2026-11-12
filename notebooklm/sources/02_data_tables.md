# HopScope data tables (all measured; source of truth for every number)

## Table A. Retrieval: share of 460 held-out SWE-bench Verified tasks where the code to fix fits in a 3,600-token context (%)

| Retriever | Hop 0 (183 tasks) | Hop 1 (137) | Hop 2+ (118) | All 460 | 95% CI (all) |
|---|---|---|---|---|---|
| BM25 over functions/classes | 77.0 | 48.9 | 33.1 | 56.7 | 52.0 to 61.1 |
| Code graph only | 91.3 | 33.6 | 0.8 | 49.6 | 45.2 to 54.1 |
| BM25 + code graph (RRF fusion) | 92.9 | 59.1 | 11.0 | 60.4 | 55.9 to 65.0 |
| BM25 with mild graph boost (beta 0.5) | 87.4 | 49.6 | 30.5 | 60.4 | 56.1 to 65.0 |
| BM25 over tree-sitter chunks | 73.8 | 49.6 | 32.2 | 55.2 | 50.7 to 59.6 |
| MiniLM-L6-v2 embeddings | 68.9 | 56.2 | 45.8 | 58.5 | 53.7 to 63.0 |
| BM25 + MiniLM (RRF fusion) | 79.8 | 64.2 | 44.9 | 65.7 | 61.3 to 70.2 |
| CodeBERT embeddings | 1.6 | 0.0 | 0.8 | 0.9 | 0.2 to 1.7 |
| GraphCodeBERT embeddings | 2.2 | 0.7 | 1.7 | 2.0 | 0.9 to 3.3 |

## Table B. HopScope labels for all 500 tasks

| Depth | Tasks | Meaning |
|---|---|---|
| Hop 0 | 196 | the issue names the code that must change |
| Hop 1 | 145 | one graph edge away |
| Hop 2+ | 135 | two or more edges away |
| No anchor | 24 | the issue names nothing found in the code graph |

## Table C. Per-task routing between BM25 and the code-graph side

| Policy | Gold in context (%) | AUROC (5-fold CV) | AUROC (held-out repository) |
|---|---|---|---|
| Always BM25 | 56.7 | n/a | n/a |
| Always graph side | 60.4 | n/a | n/a |
| Logistic regression router | 58.6 | 0.560 | 0.506 |
| MLP router (backprop) | 58.8 | 0.563 | 0.517 |
| Predictive-coding router (local learning) | 58.7 | 0.560 | 0.511 |
| Perfect router (oracle) | 68.5 | 1.0 | 1.0 |

## Table D. Embedding throughput (one NVIDIA T4, fp16) over 145,348 unique chunks

| Model | Parameters | Chunks per second |
|---|---|---|
| MiniLM-L6-v2 | 22M | 614 |
| CodeBERT | 125M | 145 |
| GraphCodeBERT | 125M | 138 |

Content hashing: 95% of 3.08 million chunk lookups across 500 commits were cache hits.

## Table E. Guards and rollback (all 500 SWE-bench Verified tasks)

| Check | Result |
|---|---|
| Gold patch + edits to existing tests blocked | 497 of 497 |
| Gold patch + injected conftest.py blocked | 500 of 500 |
| New module shadowing a test import blocked | 122 of 122 |
| Correct gold patches wrongly rejected | 1 of 500 (pylint-4661 edits setup.cfg) |
| Random agent mutations rolled back bit-exactly | 200 of 200 |
| Hard-coded constant patches flagged | 378 of 400 |
| Correct patches also flagged (so only a tie-breaker) | 61 of 500 |

## Table F. Gemma 4 E4B end-to-end (50 held-out tasks, official SWE-bench harness v5.0.1)

| Stage | Count |
|---|---|
| Tasks attempted | 50 |
| Produced a patch | 8 |
| Patch edits the file the real fix changes | 7 |
| Patch applies cleanly | 8 |
| Best partial: django-14672 | 167 of 168 target tests pass |
| Resolved | 0 |
| Patches that broke existing tests | 3 |
| Edit attempts whose quoted code was not in the file | 46 |
| Gold patch resolves in the scoring environment | 8 of 8 checked |
| Reproduction scripts: usable / broken / already passing | 21 / 19 / 10 |
| Run wall time (2x T4) | 73 minutes |
| Median time per task | 139 s |

## Table G. Hardware and speed (tokens per second reported by llama-server)

| Setup | Model | Prompt tok/s | Generated tok/s | Memory |
|---|---|---|---|---|
| Kaggle T4, GPU working | E4B QAT q4_0 | 1877 | 39.4 | 3.8 GB GPU per T4 |
| Kaggle T4, CUDA libraries missing (silent CPU fallback) | E4B | 14 | 3.7 | n/a |
| Laptop Ryzen 5 5500U CPU, best run | E2B QAT q4_0 | 114 | 5.4 | llama-server peak 1.67 GB |
| Laptop CPU, throttled | E2B | 59 | 2.2 | 1.49 GB |
| Laptop Radeon iGPU (Vulkan) | E2B | failed | failed | could not allocate 547-830 MB compute buffers |

Laptop: AMD Ryzen 5 5500U (6 cores / 12 threads), 6.82 GB usable RAM, Windows 11, Docker Desktop, no discrete GPU. One agent turn on the laptop (about 1,500 new prompt tokens + 200 generated) takes about 50 seconds.
