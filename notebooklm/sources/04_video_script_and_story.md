# HopScope video: story, scene plan and narration guide

This document tells the story of the project in the order a viewer should hear it. Every number here appears in the data tables source (02_data_tables.md). Do not use any other numbers.

## The one-sentence story

Small open models like Gemma 4 can run a coding agent on an ordinary 8 GB laptop. But they only ever see the few thousand tokens the retriever hands them, and we found that the popular fix, code graphs, helps when the issue names the right code and badly hurts when the bug is a few hops away.

## Scene 1. The problem (about 40 seconds)

- Today's best coding agents run on huge cloud models. That means cost, sending your code to someone else's servers, and no help when you are offline.
- A small open model, Gemma 4 E2B or E4B, fits on a normal laptop. Ours is a Ryzen 5 5500U with 8 GB of RAM and no graphics card.
- The catch: the model can look at about 8,000 tokens at a time. A real project like Django has more than 10,000 functions. The model never sees the project; it sees whatever the search step picks for it.
- So the real question is: does the code that has to change end up in front of the model?

## Scene 2. A real bug, and what "depth" means (about 45 seconds)

Show image g02_depth.png.

- Bug report: "requests.get is ALWAYS sending content length."
- The report mentions the requests package. The fix lives in a method called prepare_content_length, inside a class called PreparedRequest.
- That is two steps away in the project's dependency graph: the package imports the class, the class contains the method.
- We call that distance the bug's depth. Depth 0 means the report names the exact code. Depth 2 or more means the report describes a symptom far from the cause.
- We labelled every one of the 500 tasks in SWE-bench Verified this way: 196 at depth 0, 145 at depth 1, 135 at depth 2 or more, and 24 where the report names nothing we could find. We call the benchmark HopScope.

## Scene 3. The main finding (about 60 seconds)

Show image g01_finding.png, then g03_gains_losses.png.

- We gave every search method the same budget: 3,600 tokens of code, enough for a small model.
- Plain keyword search (BM25) finds the right code 77% of the time when the report names it, but only 33% when the bug is two or more hops away.
- Adding a code graph lifts the easy case from 77% to 93%. On deep bugs it falls from 33% to 11%.
- Counting task by task: on deep bugs the graph rescued 2 tasks and lost 28.
- Why: a graph follows whatever the report points at. If the report points at the wrong place, the graph spends the whole budget there.

## Scene 4. The surprise about embeddings (about 40 seconds)

Show image g04_embedders.png.

- We expected models trained on code to search code best. They did not.
- CodeBERT and GraphCodeBERT, used as published, found the right code about 1 to 2% of the time; they were never trained for search.
- A tiny 22-million-parameter general text model, MiniLM, combined with keyword search, was the best option: 65.7% overall and 45% on deep bugs, against keyword search's 33%.
- It is also fast: 614 chunks per second on one T4 GPU, and with caching 95% of lookups were free.

## Scene 5. The open problem (about 30 seconds)

Show image g05_routing.png.

- If graphs help easy bugs and hurt deep ones, just pick the right tool per bug, right?
- We trained three different models to make that choice from the bug report. All three were no better than a coin flip (AUROC 0.56).
- A perfect chooser would gain 8 more points. You cannot tell from the surface of a bug report how deep the bug is. That is the open problem this benchmark puts on the table.

## Scene 6. The agent, and keeping it honest (about 50 seconds)

Show image g06_agent.png, then g07_guards.png.

- We built gemnav, an agent that runs fully offline: llama.cpp for the model, SQLite and tree-sitter for search, git for safety.
- Before it edits anything, the model writes a small script that reproduces the bug. That script is locked away read-only, so the model cannot "fix" the test instead of the code.
- Every edit is checked: no touching existing tests, it must compile, and the bug script and old tests must not get worse. Otherwise git undoes it.
- Every cheating attempt we built from the real benchmark was blocked. Only 1 of 500 correct fixes was wrongly refused.
- Along the way we found five silent failures that would have ruined the results without any error message. One example: on Kaggle the model quietly ran on the CPU at 14 tokens per second instead of the GPU at about 1,900.

## Scene 7. The honest end-to-end result (about 45 seconds)

Show image g09_end_to_end.png, then g08_hardware.png.

- We ran Gemma 4 E4B on 50 of these tasks and scored it with the official SWE-bench test harness.
- 8 attempts produced a patch, 7 of them in exactly the right file, and all 8 applied cleanly. One passed 167 of 168 target tests.
- None fully resolved its bug. The main reason: 46 times the model tried to edit code it could not quote exactly.
- On the laptop the model runs on the CPU at about 114 tokens per second of reading and 5 of writing, roughly 50 seconds per step. The laptop's built-in graphics chip could not run it at all.

## Scene 8. What it means, and what is next (about 40 seconds)

Show the roadmap points.

- For small local agents, finding the code is no longer the main problem for easy bugs. Turning the found code into an exact edit is.
- Next steps:
  1. edits by line number instead of copied text;
  2. a small fine-tune focused on exact edits;
  3. retrieval that expands the graph only around code the agent has already confirmed;
  4. releasing HopScope labels so everyone reports search quality by depth, not one average.
- Everything, including code, data and every chart, is open source at github.com/raoheme/gemnav.

## Tone

Plain, curious, honest. A developer explaining what they found to other developers. No hype words. Say what failed as clearly as what worked.
