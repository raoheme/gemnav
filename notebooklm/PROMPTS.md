# NotebookLM: setup and the exact prompts to paste

## 1. Create the notebook (about 3 minutes)
1. Go to notebooklm.google.com, then **New notebook**. Name it `HopScope · gemnav`.
2. **Add sources** and upload, in this order:
   - `sources/01_paper_writeup.md`
   - `sources/02_data_tables.md`  ← the source of truth for every number
   - `sources/03_roadmap_future_work.md`
   - `sources/04_video_script_and_story.md`
   - all 9 images in `images/` (g01 to g09; skip the card)
3. Wait until every source shows as processed. Ask in chat: *"List every number about hop 2+ retrieval."* Check that the answer matches Table A (33.1, 11.0, 44.9). If it matches, the sources loaded correctly.

## 2. Video Overview  (Studio panel, Video Overview, ✏️ Customize)
Format: **Explainer**. Language: English. Paste:

```
Make a 6-8 minute explainer video for software engineers and ML researchers, following the scene order in
"04_video_script_and_story.md" exactly: 1 problem, 2 what dependency depth means, 3 main finding, 4 embeddings
surprise, 5 open problem, 6 the agent and its guards, 7 honest end-to-end result, 8 what is next.
Open with the problem: the best coding agents need huge cloud models, while a small open model like Gemma 4 fits
on an 8 GB laptop but only ever sees the few thousand tokens its search step picks.
Use ONLY numbers from "02_data_tables.md" and say them exactly as written (77% to 93% at hop 0; 33% to 11% at
hop 2+; BM25 + MiniLM 45% at hop 2+; routers AUROC 0.56; 0 of 50 resolved, 8 patches, 7 in the right file,
1 at 167 of 168 tests). Never round differently, never invent numbers, never claim the agent fixed bugs.
Show the uploaded charts (g01 to g09) when the matching scene is discussed; prefer them over generated visuals.
Be plain and honest, like a developer explaining results to peers; no hype. End with the roadmap and the
repository name github.com/raoheme/gemnav.
```

After it renders, **watch the whole video once and check every number against Table A–G.** If one is wrong, regenerate with the line *"Correction: <the right number> (Table X)."* added to the top of the prompt.
Download it, then upload it to YouTube as **Unlisted**, titled *"HopScope: where code graphs help small coding agents (Gemma 4, SWE-bench Verified)"*. Paste the YouTube link into the Kaggle media gallery.

## 3. Slide deck  (Studio, Slide Deck, Customize)
```
A 12-slide deck for a research talk at an ML venue. One idea per slide, title states the takeaway.
Order: problem; what dependency depth is (requests-1142 example); HopScope benchmark composition; main result
table by depth; gains vs losses when adding the graph; embeddings result; routing open problem; the gemnav agent;
guards; hardware and speed; end-to-end funnel (0 of 50, with the reason); roadmap.
Use only numbers from "02_data_tables.md". Reuse the uploaded charts where they fit.
```

## 4. Report  (Studio, Reports, Create your own)
```
Write a 2-page technical brief titled "HopScope: retrieval for small coding agents, by dependency depth".
Sections: Summary (5 sentences), Method, Results (with one table copied from Table A), What failed end-to-end
and why, Recommendations for anyone building a local coding agent (5 concrete rules derived from the
results), Roadmap. Use only numbers from "02_data_tables.md"; mark anything not measured as "planned".
```
Second report (optional), using **Study guide** or **Briefing doc**: *"Briefing doc for a hackathon judge: what is new, what was measured, and what is honestly not yet working."*

## 5. Data table  (Studio, Data Table)
```
Build one table with a row per retriever and columns: retriever, hop 0 %, hop 1 %, hop 2+ %, all %,
change at hop 2+ versus BM25 (percentage points). Copy values exactly from Table A of "02_data_tables.md".
```
Then a second one:
```
One row per stage of the Gemma 4 E4B end-to-end funnel from Table F, with a column "what went wrong at this stage".
```

## 6. Infographic  (Studio, Infographic)
Make two. Use them as extra gallery images **only after checking every number by eye.**
```
Portrait infographic: "Code graphs help shallow bugs and hurt deep ones". Three big numbers: 77% -> 93% (graph,
bug named in the report), 33% -> 11% (graph, bug 2+ hops away), 33% -> 45% (BM25 + MiniLM, 2+ hops).
Below: one line each on the benchmark (500 SWE-bench Verified tasks labelled by depth), the laptop
(8 GB, CPU only, Gemma 4 E2B) and the open problem (routers AUROC 0.56). Clean, light background, no clip art.
```
```
Landscape infographic: "Where a small coding agent fails". A funnel from Table F: 50 tasks, 8 patches,
7 in the right file, 8 apply, 1 passes 167/168 tests, 0 resolved; side note "46 edits quoted code that was not
in the file". Footer: next step "edit by line number + exact-edit fine-tune".
```

## 7. Mind map  (optional, Studio, Mind Map)
No prompt needed. Export it as an image if it looks clean; it makes a good gallery item showing the project's structure.

## Rules for every output
- Every number must appear in `02_data_tables.md`. NotebookLM sometimes rounds or invents numbers, so check before you upload.
- Never say the agent "solved" or "fixed" bugs. It resolved 0 of 50; say that plainly.
- The charts in `images/` are exact. Prefer them over any generated chart.
