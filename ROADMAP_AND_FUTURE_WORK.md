# gemnav / HopScope: future work and roadmap

*Companion document to the Kaggle writeup. Everything in "What we measured" is a measured result; everything after it is a plan.*

## What we measured (the starting point)

| Stage | Result | Source |
|---|---|---|
| Retrieval, hop 0 | BM25 77.0% → BM25 + code graph 92.9% | `results/e1_summary.json` |
| Retrieval, hop 2+ | BM25 33.1% → BM25 + code graph 11.0%; BM25 + MiniLM 44.9% | `e1_summary.json`, `e5_summary.json` |
| Per-task routing | AUROC 0.56 (CV), 0.51 (held-out repo); oracle headroom +8.0 points | `e1_summary.json` |
| Gemma 4 E4B, 50 tasks | 8 patches, 7 in the right file, 8 apply, 1 at 167/168 tests, **0 resolved** | `results/e2e/e2e_e4b_summary.json` |
| Main agent failure | 46 edit attempts quoted code that was not in the file | `results/kaggle/traj_*.json` |
| Laptop (Ryzen 5 5500U, 8 GB) | E2B CPU: 114 prompt / 5.4 gen tok/s (59 / 2.2 throttled); iGPU (Vulkan) cannot allocate buffers | `results/laptop/` |

The pattern is clear. **Finding the code is mostly solved for shallow bugs. Turning located code into an exact edit is the bottleneck for a 4B model.** The roadmap is ordered by how directly each item attacks that bottleneck.

## Phase 1: edit precision (highest impact, 1–2 weeks)

1. **Line-anchored edits instead of quoted text.** Today the model must copy `old` text exactly, and that failed 46 times. Change the tool to `replace_lines {path, start, end, new}` with line numbers taken from the last `read_file`. The harness then knows exactly which lines change. The model only has to produce the new text, which small models are much better at.
2. **Edit-precision SFT (QLoRA).**
   - *Data:* 2–3K synthetic examples built from SWE-bench *train* repositories (no Verified leakage). Given numbered file lines and a described change, emit the exact tool call.
   - *Training:* E2B and E4B, rank-16 LoRA on attention and MLP, 1–2 epochs.
   - *Hardware:* a single 16–24 GB GPU (Kaggle T4 works for E2B; a T4 or L4 for E4B).
   - *Target metric:* edit-match rate on held-out repositories; then re-run the same 50 tasks.
3. **Show the model the target test.** Let `run_tests` run the reproduction script and report the failing assertion verbatim each turn, so the model can see why 167/168 isn't 168/168.
4. **Wider regression set.** Three patches broke existing tests outside the two nearby test files we checked. Pick the regression set from test files that import the edited module (we have the import graph), not just by name.

## Phase 2: knowing when a bug is deep (2–4 weeks)

5. **Better routing signals.** Surface features fail (AUROC 0.56). Candidates:
   - *anchor quality:* is the anchor a hub? how many definitions share its name?
   - *disagreement:* how far apart the BM25 and graph top-10 lists are;
   - *learned relevance:* a small cross-encoder that scores (issue, anchor) pairs.

   Train it on HopScope labels.
6. **Iterative retrieval instead of routing.** Retrieve with BM25 + MiniLM first; let the agent expand graph neighbours of chunks it has actually *read*, rather than of anchors guessed from the issue. This uses the graph where it is reliable, near confirmed-relevant code.
7. **Retrieval-trained code embedders.** CodeBERT and GraphCodeBERT failed because they were never trained for retrieval. Fine-tune a small encoder contrastively on (issue, gold function) pairs from SWE-bench train, and compare it with MiniLM on HopScope. The Jina v2 code embedder run did not finish this time; finish it.

## Phase 3: learning from outcomes (1–2 months)

8. **Trajectory SFT.** Collect successful navigation and repair trajectories, including dead ends that were recovered from. Sources: the agent itself after Phase 1, plus a stronger teacher model used offline. Train on SWE-bench train repositories only.
9. **DPO on test outcomes.** Build preference pairs, each with the same task and two trajectories: one whose patch passes the official tests and one that doesn't. This rewards the model with real test results, which our guards already make hard to game (tests are read-only; rollback is verified).
10. **Self-improvement loop, carefully.** Accept new training data only from patches that pass the frozen reproduction test *and* the regression set, and replay-verify them on held-out tasks before adding them. The guard layer already gives us the integrity checks this needs.

## Phase 4: consumer hardware (in parallel)

11. **10K-file indexing benchmark.** Finish the planned benchmark on a real 10K-file repository on the laptop. Measure chunking, embedding and SQLite time, peak RAM, vector-search, ripgrep and LSP latency against the blueprint targets.
12. **E2B laptop end-to-end.** Run the same 50 tasks on the laptop CPU, with RAM sampled every 5 s; the runner (`code/laptop/run_laptop_e2e.ps1`) is ready.
13. **Prompt-cache-aware context.** On CPU, prompt processing dominates (about 50 s per turn). Order the prompt so the stable parts (system, file map, retrieved code) come first and only the history grows, maximising llama.cpp cache reuse.
14. **iGPU.** Vulkan failed because the driver exposes too little device memory. Test larger UMA settings in BIOS, the OpenVINO build, or offloading only a few layers.

## Phase 5: HopScope as a public resource

15. **Release HopScope labels** as a dataset with the code graph, anchors and depth for every SWE-bench Verified task, so others can report retrieval *by depth* instead of one average.
16. **Extend beyond Python**, using tree-sitter graphs for JavaScript/TypeScript and Java (SWE-bench Multilingual).
17. **Leaderboard view.** Re-score published agents' trajectories by depth to see whether large models show the same graph collapse at hop 2+.

## Success criteria for the next submission

- Edit-match rate on held-out repositories ≥ 80% (today: most edit attempts fail).
- E4B resolves ≥ 5 of the same 50 tasks, scored by the official harness.
- A router with AUROC ≥ 0.65 on held-out repositories.
- Laptop E2B: full 50-task run with measured per-turn latency and peak RAM.
