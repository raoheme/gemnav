"""Deterministic 50-task stratified sample for the end-to-end runs (seed 42).

Population: the 460 held-out tasks of E1 (development repositories excluded, as in the paper).
Strata: E1 hop depth (0 / 1 / 2+ / unanchored); allocation proportional to stratum size (largest remainder),
then within each stratum a seeded shuffle. Writes results/e2e_tasks50.json with per-task repo, hop, difficulty.
"""
import json
import random
import sys
from collections import defaultdict

import pyarrow.parquet as pq

DEV = {"psf/requests", "pallets/flask", "mwaskom/seaborn", "pylint-dev/pylint", "pytest-dev/pytest"}


def main(e1_path, parquet, out, n=50, seed=42):
    e1 = {json.loads(l)["instance_id"]: json.loads(l) for l in open(e1_path)}
    rows = {r["instance_id"]: r for r in pq.read_table(parquet).to_pylist()}
    pop = sorted(i for i, r in e1.items() if r["repo"] not in DEV and "skip" not in r)
    strata = defaultdict(list)
    for i in pop:
        h = e1[i].get("hop")
        strata["unanchored" if h is None else ("0" if h == 0 else "1" if h == 1 else "2+")].append(i)
    quota = {k: n * len(v) / len(pop) for k, v in strata.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: -(quota[k] - alloc[k]))[: n - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    chosen = []
    for k in sorted(strata):
        ids = sorted(strata[k])
        rng.shuffle(ids)
        chosen += [(i, k) for i in ids[: alloc[k]]]
    chosen.sort()
    data = {"seed": seed, "population": len(pop), "allocation": alloc,
            "tasks": [{"instance_id": i, "repo": rows[i]["repo"], "hop": k, "difficulty": rows[i]["difficulty"]} for i, k in chosen]}
    json.dump(data, open(out, "w"), indent=1)
    from collections import Counter
    print(alloc, Counter(t["repo"] for t in data["tasks"]), Counter(t["difficulty"] for t in data["tasks"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
