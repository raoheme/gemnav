"""Entity retrievers and token-budgeted context assembly.

All retrievers return a ranked list of entity ids. Token counts use the
documented approximation tokens ~= ceil(chars / 4) (no tokenizer dependency).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .codegraph import CodeGraph

_TOK = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOP = set("""a an the and or of to in is it this that for on with as be are was were by not no if
then else from at but can will would should could have has had do does did i we you they he she
when what which who how why there here so such only also any all some more most very just than too
into out up down over under again use used using get set self none true false return def class import
""".split())


def tokenize(text: str) -> list[str]:
    out = []
    for w in _TOK.findall(text):
        lw = w.lower()
        if lw not in STOP and len(lw) > 1:
            out.append(lw)
        if "_" in w or any(c.isupper() for c in w[1:]):
            for p in _CAMEL.findall(w):
                lp = p.lower()
                if lp != lw and lp not in STOP and len(lp) > 1:
                    out.append(lp)
    # snake_case parts
    for w in re.findall(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+", text):
        for p in w.split("_"):
            lp = p.lower()
            if lp and lp not in STOP and len(lp) > 1:
                out.append(lp)
    return out


def approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def entity_doc(g: CodeGraph, eid: str) -> str:
    e = g.entities[eid]
    return f"{e.path.replace('/', ' ')} {eid.split('::')[-1].replace('.', ' ')} {e.name} {e.text[:6000]}"


class BM25:
    def __init__(self, g: CodeGraph, k1: float = 1.2, b: float = 0.75):
        self.ids = list(g.entities)
        self.k1, self.b = k1, b
        self.tf: list[Counter] = []
        df: Counter = Counter()
        for eid in self.ids:
            c = Counter(tokenize(entity_doc(g, eid)))
            self.tf.append(c)
            df.update(c.keys())
        self.N = len(self.ids)
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / self.N) if self.N else 1.0
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.post: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)

    def scores(self, query: str) -> dict[str, float]:
        q = Counter(tokenize(query))
        s: dict[int, float] = defaultdict(float)
        for t, qn in q.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return {self.ids[i]: v for i, v in s.items()}

    def rank(self, query: str) -> list[str]:
        sc = self.scores(query)
        return sorted(sc, key=lambda k: (-sc[k], k))


def graph_scores(g: CodeGraph, seeds: dict[str, float], hops: int = 2, gamma: float = 0.5) -> dict[str, float]:
    """Degree-normalised seed propagation: s_{k+1}(v) = sum_{u~v} s_k(u)/deg(u); total = sum_k gamma^k s_k."""
    total = dict(seeds)
    frontier = dict(seeds)
    for k in range(1, hops + 1):
        nxt: dict[str, float] = defaultdict(float)
        for u, su in frontier.items():
            d = g.degree(u)
            if d == 0:
                continue
            share = su / d
            for v in g.adj[u]:
                nxt[v] += share
        for v, sv in nxt.items():
            total[v] = total.get(v, 0.0) + (gamma ** k) * sv
        frontier = nxt
    return total


def graph_rank(g: CodeGraph, seeds: dict[str, float], bm25_scores: dict[str, float], hops: int = 2) -> list[str]:
    gs = graph_scores(g, seeds, hops)
    mx = max(bm25_scores.values(), default=1.0) or 1.0
    # graph score dominates; BM25 (normalised, x1e-3) breaks ties; unreached entities follow in BM25 order
    reached = sorted(gs, key=lambda k: (-(gs[k] + 1e-3 * bm25_scores.get(k, 0.0) / mx), k))
    rest = sorted((k for k in bm25_scores if k not in gs), key=lambda k: (-bm25_scores[k], k))
    return reached + rest


def graph_boost_rank(g: CodeGraph, seeds: dict[str, float], bm25_scores: dict[str, float],
                     dist: dict[str, int], beta: float = 1.0, gamma: float = 0.5, max_hops: int = 2) -> list[str]:
    """Graph-boosted BM25: s(v) = bm25(v) * (1 + beta * w(v) * gamma**d(v)) for d(v) <= max_hops.

    Content relevance comes from BM25; the code graph only re-weights entities that are
    structurally close to issue anchors (d(v) = BFS hops from the nearest anchor).
    Entities with zero BM25 score but inside the neighbourhood get a small floor so that
    lexically silent neighbours (the multi-hop case) can still surface.
    """
    if not seeds:
        return sorted(bm25_scores, key=lambda k: (-bm25_scores[k], k))
    mx = max(bm25_scores.values(), default=1.0) or 1.0
    floor = 0.05 * mx
    s = dict(bm25_scores)
    for v, d in dist.items():
        if d <= max_hops:
            s[v] = max(s.get(v, 0.0), floor * (gamma ** d)) * (1 + beta * (gamma ** d))
    return sorted(s, key=lambda k: (-s[k], k))


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    s: dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, e in enumerate(r):
            s[e] += 1.0 / (k + i + 1)
    return sorted(s, key=lambda e: (-s[e], e))


def budgeted_context(g: CodeGraph, ranking: list[str], budget_tokens: int = 3600, per_entity_cap: int = 1200) -> tuple[list[str], int]:
    """Greedily pack ranked entities (each truncated to per_entity_cap tokens) under the budget."""
    used, chosen = 0, []
    for eid in ranking:
        t = min(approx_tokens(g.entities[eid].text) + 12, per_entity_cap)
        if used + t > budget_tokens:
            if used >= budget_tokens - 50:
                break
            continue
        chosen.append(eid)
        used += t
    return chosen, used


def first_hit_rank(ranking: list[str], gold: set[str], limit: int = 10**9) -> int | None:
    for i, e in enumerate(ranking[:limit]):
        if e in gold:
            return i + 1
    return None


def file_ranking(g: CodeGraph, ranking: list[str]) -> list[str]:
    seen, out = set(), []
    for e in ranking:
        p = g.entities[e].path
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
