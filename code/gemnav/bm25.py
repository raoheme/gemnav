"""BM25 over chunk texts (same tokenizer/parameters as swe_local.retrieval.BM25)."""
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from swe_local.retrieval import tokenize  # noqa: E402


class ChunkBM25:
    """BM25 over chunk texts (same tokenizer/parameters as swe_local.retrieval.BM25)."""

    def __init__(self, texts, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.N = len(texts)
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / self.N) if self.N else 1.0
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)

    def rank(self, q):
        s = defaultdict(float)
        for t in Counter(tokenize(q)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return sorted(s, key=lambda i: (-s[i], i))
