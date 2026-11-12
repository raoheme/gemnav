
## The question

I wanted a coding agent that runs on my own laptop: a Ryzen 5 5500U with 8 GB of RAM and no discrete GPU. At that size, Gemma 4 E2B or E4B can hold about 8K tokens, while a repository like Django has over 10,000 functions and classes. So the model never reads the repo. It reads whatever the retriever hands it, and whether the bug gets fixed depends mostly on that choice.

Papers on code graphs for software agents (RepoGraph, CodexGraph, LocAgent) report end-to-end gains with large models. Those numbers mix retrieval, model skill and prompting together, so they don't tell you what to build for a 4B model. I measured one thing on its own: **does the code that has to change land inside the context window?** I measured it as a function of how far that code is from what the issue actually mentions.

That distance turns out to matter more than anything else.

## Dependency depth, with a real example

[Figure: What dependency depth means, see image g02_depth.png]
*Task psf__requests-1142. The issue mentions `requests.get`; the fix is in `PreparedRequest.prepare_content_length`, two graph hops from anything the issue names.*

For every task I parse the repository at its base commit with Python's `ast` into module, class and function nodes, connected by *contains*, *calls*, *imports* and *inherits* edges. A call edge is dropped when the name matches more than five definitions, so names like `get` don't become hubs.

- **Anchors** are the nodes the issue points at: file paths, dotted names, identifiers and stack-trace frames.
- **Gold** is the innermost function or class containing each line the reference patch changes.
- **Depth** is the shortest path from any anchor to any gold node. Hop 0 means the issue names the code that changes; hop 2+ means it describes a symptom somewhere else.

I call the resulting labels **HopScope**; every task in SWE-bench Verified gets one. Of the 500 tasks, 196 are hop 0, 145 hop 1, 135 hop 2+, and 24 have no anchor. Five small repositories (40 tasks) were used only to choose settings. **All numbers below are on the other 460 tasks across 7 repositories.**

The measurement is simple. Retrieved entities are packed greedily into a fixed **3,600-token** budget (tokens ≈ characters/4, at most 1,200 per entity), and I count whether a gold entity made it in. Intervals are 2,000-sample bootstraps over tasks.

## What I found

[Figure: Retrieval by dependency depth, see image g01_finding.png]

| Retriever (460 held-out tasks) | Hop 0 (183) | Hop 1 (137) | Hop 2+ (118) | All [95% CI] |
|---|---|---|---|---|
| BM25 over functions/classes | 77.0 | 48.9 | 33.1 | 56.7 [52.0, 61.1] |
| Code graph only | 91.3 | 33.6 | 0.8 | 49.6 [45.2, 54.1] |
| BM25 + code graph (RRF) | **92.9** | 59.1 | 11.0 | 60.4 [55.9, 65.0] |
| BM25 with a mild graph boost (β = 0.5) | 87.4 | 49.6 | 30.5 | 60.4 [56.1, 65.0] |
| BM25 over tree-sitter chunks | 73.8 | 49.6 | 32.2 | 55.2 [50.7, 59.6] |
| MiniLM-L6-v2 embeddings | 68.9 | 56.2 | **45.8** | 58.5 [53.7, 63.0] |
| BM25 + MiniLM (RRF) | 79.8 | **64.2** | 44.9 | **65.7** [61.3, 70.2] |
| CodeBERT embeddings (mean-pooled) | 1.6 | 0.0 | 0.8 | 0.9 |
| GraphCodeBERT embeddings (mean-pooled) | 2.2 | 0.7 | 1.7 | 2.0 |

**1. A code graph amplifies whatever the issue points at, right or wrong.** When the issue names the code (hop 0), adding the graph to BM25 lifts retrieval from 77% to 93%, and file-level top-1 goes from 41.7% to 50.4%. On deep tasks it drops from 33% to 11%. The averages hide this: on hop 2+ the graph rescues 2 tasks and loses 28. My reading is that a misleading anchor spends the token budget on the wrong neighbourhood and pushes out the lexical hits that would have worked. I didn't isolate that mechanism, so treat it as a hypothesis.

[Figure: Gains and losses, see image g03_gains_losses.png]

If you want a graph in a small agent's retriever, use it as a gentle boost on BM25 scores, not as an equal partner. The β = 0.5 boost matches RRF overall and keeps 30.5% at depth instead of 11%.

**2. A tiny general-purpose embedder beat the code models.** MiniLM has 22M parameters and was not trained on code. Fused with BM25, it was the most depth-robust option I tried: 44.9% on hop 2+ against BM25's 33.1%, and 65.7% overall. CodeBERT and GraphCodeBERT, mean-pooled as published, sit at 1–2%. That is expected once you look closer: they are masked-language-model encoders, never trained to put similar code close together. "Pretrained on code" and "useful for retrieval" are different properties.

The cost is small. On one T4 in fp16, MiniLM embedded 614 chunks/s, against 145 for CodeBERT and 138 for GraphCodeBERT. Hashing chunk contents made 95% of the 3.08M chunk lookups across all 500 commits cache hits, so each repository version only pays for what changed.

[Figure: Embedders, see image g04_embedders.png]

**3. You can't tell from the issue text whether a bug is deep.** If the graph helps shallow tasks and hurts deep ones, the obvious move is to route per task. I trained three routers on ten cheap Turn-1 features: BM25 and graph score margins, anchor count, ranking overlap, issue length and a stack-trace flag. The three were logistic regression, a small MLP, and a predictive-coding network trained with local updates only.

All three land at AUROC 0.56 in cross-validation, and 0.51 when the test repository is held out entirely, i.e. chance. A perfect router would reach 68.5%, at least 8 points above any fixed policy. This is the most useful open problem the benchmark exposes: the signal that a bug is deep is not on the surface of the issue.

[Figure: Routing, see image g05_routing.png]

## gemnav: putting it on a laptop

[Figure: Agent, see image g06_agent.png]

The agent takes the retrieval result and adds what a small model needs to avoid hurting itself.

- **Serving.** llama.cpp (b12211) runs Google's QAT q4_0 GGUFs of Gemma 4 with an 8K context, a q4_0 KV cache and greedy decoding.
- **Index.** Tree-sitter splits code into whole functions and classes; classes over 200 lines are split per method. Chunks go into SQLite with MiniLM vectors and BM25 alongside, plus `pylsp` for definitions and references.
- **One action per turn.** Each turn is one JSON tool call, held to the right shape by a GBNF grammar written per tool, then checked again in Python.
- **Before editing,** the model writes a script that reproduces the bug. It is stored read-only outside the repository, so the model can't later "fix" the test instead of the code.
- **Every edit** goes through four checks in order:
  1. a guard refusing changes to existing tests and test configuration;
  2. `py_compile`;
  3. the reproduction script;
  4. nearby existing tests.

  If anything gets worse, git restores the tree and a digest confirms it.

[Figure: Guards, see image g07_guards.png]

The attacks above are built by me from the real benchmark tests, so 100% shows the rules are implemented, not that a determined model can't get around them. The one false rejection, pylint-4661, is a correct fix that edits `setup.cfg`.

Short pilot runs on Kaggle (2×T4) turned up five problems that raised no error yet would have quietly ruined the results. Each is fixed in the released code:

1. **Silent CPU fallback.** llama.cpp couldn't find its CUDA libraries, ran at 14 tokens/s and said nothing. It now refuses to start unless both GPUs are listed.
2. **No cache reuse.** Gemma's sliding-window cache isn't reused between requests unless you pass `--swa-full`. With it, turns reuse about 4,000 cached tokens.
3. **Grammar rejected.** llama.cpp rejects grammar repetition bounds above an internal limit. Strings are now bounded at 600 characters, which I measured costs no speed.
4. **Runaway newlines.** Greedy decoding wrote thousands of `\n` inside one JSON string; a deterministic DRY repetition penalty stops it.
5. **Edits that couldn't match.** Small models drop indentation when copying code. Edits now use whitespace-tolerant matching that re-indents and refuses ambiguous matches.

[Figure: Hardware, see image g08_hardware.png]

**On my laptop** (6.8 GB usable RAM), E2B runs on CPU at 114 prompt and 5.4 generated tokens/s, with llama-server peaking at 1.7 GB. That is about 50 seconds per agent turn: slow, but workable for a background agent. When the laptop throttled, the same test gave 59 and 2.2 tokens/s.

The Radeon iGPU (Vulkan) could not create a context at all: llama.cpp's 547–830 MB compute buffers didn't fit in the memory the driver exposes. For this class of machine, CPU is the only path.

## End-to-end, briefly

I ran Gemma 4 E4B through the full loop on 50 held-out tasks, stratified by depth, on Kaggle T4s. The run took 73 minutes, with 3.8 GB of GPU memory per card. I scored the patches with the official SWE-bench harness (v5.0.1, Docker) on the laptop; the gold patch resolved in every environment I checked.

[Figure: End to end, see image g09_end_to_end.png]

**Resolved: 0 of 50.** The funnel shows where it breaks:

- 8 runs produced a patch, 7 of them in the file the real fix changes, and all 8 applied cleanly.
- One (django-14672) passed 167 of 168 target tests with the right idea on the right line.
- Most runs never got that far: 46 edit attempts quoted code that wasn't in the file exactly.
- Three patches broke existing tests outside the two nearby test files the agent checks.

At this size, retrieval is no longer the bottleneck; turning located code into a precise edit is. That is what the next round of work targets (see the attached roadmap).

## Limitations

- **The retrieval numbers are an upper bound** on what an agent can use.
- **The graph is Python-only and static,** and depth is defined on it, so some "deep" tasks are deep partly because an edge was missed.
- **Gold comes from the reference patch,** so valid fixes elsewhere count as misses.
- **The token budget and token estimate are fixed choices.**
- **The guard attacks are mine.**
- **Fifty end-to-end tasks is a small sample.** The E2B agent run and the 10K-file indexing benchmark on the laptop were not completed in time. Jina code embeddings did not finish on Kaggle.

## Related work

SWE-bench and its Verified subset [1]; agent designs SWE-agent [2], OpenHands [3] and Agentless [4]. Graph-based repository access: AutoCodeRover [5], CodexGraph [6], RepoGraph [7] and LocAgent [8]. LocAgent's own ablation found the lexical entry point mattered most, which fits finding 1. Sentence-BERT and MiniLM [9]; CodeBERT [10] and GraphCodeBERT [11] use masked-language pretraining, not retrieval training. Predictive coding with local learning [12]; reward hacking [13]; Gemma 4 [14]; llama.cpp and GBNF grammars [15].

## Reproducing this

Code, per-task records and every figure are at **github.com/raoheme/gemnav** (Apache-2.0) and in the attached zip, along with the roadmap and the raw harness reports. `python -m pytest code/tests` runs 20 tests. Every chart is regenerated from the result files by `make_gallery.py`.

**References.** [1] Jimenez et al., SWE-bench, ICLR 2024 (arXiv:2310.06770). [2] Yang et al., SWE-agent, 2024 (arXiv:2405.15793). [3] Wang et al., OpenHands, 2024 (arXiv:2407.16741). [4] Xia et al., Agentless, 2024 (arXiv:2407.01489). [5] Zhang et al., AutoCodeRover, ISSTA 2024 (arXiv:2404.05427). [6] Liu et al., CodexGraph, 2024 (arXiv:2408.03910). [7] Ouyang et al., RepoGraph, ICLR 2025 (arXiv:2410.14684). [8] Chen et al., LocAgent, ACL 2025 (arXiv:2503.09089). [9] Reimers & Gurevych, Sentence-BERT, EMNLP 2019; Wang et al., MiniLM, NeurIPS 2020. [10] Feng et al., CodeBERT, Findings of EMNLP 2020. [11] Guo et al., GraphCodeBERT, ICLR 2021. [12] Whittington & Bogacz, Neural Computation 29(5), 2017. [13] Skalse et al., Defining and Characterizing Reward Hacking, NeurIPS 2022. [14] Google DeepMind, Gemma 4 model card, 2026. [15] ggml-org/llama.cpp.
