"""Per-task retrieval router: decides between graph propagation and BM25 at Turn 1.

Three classifiers with identical inputs (10 standardised features):
  LogisticRouter : L2-regularised logistic regression (full-batch gradient descent)
  MLPRouter      : 10-32-1 tanh MLP trained with backpropagation
  PCRouter       : 10-32-1 predictive-coding network trained with local prediction-error
                   minimisation (Whittington & Bogacz, 2017): hidden states are relaxed by
                   gradient descent on the energy F = 1/2 sum_l ||x_l - mu_l||^2 with the input
                   and the label clamped; weights then follow purely local Hebbian-like updates
                   dW_l ~ eps_{l+1} f(x_l)^T. No backpropagated gradient is used.
All use NumPy only, a fixed seed and CPU execution.
"""
from __future__ import annotations

import math
import re

import numpy as np

FEATURE_NAMES = [
    "bm25_top1_log",
    "bm25_margin_ratio",
    "n_seeds_log",
    "bm25_graph_top10_overlap",
    "max_seed_weight",
    "graph_top1_score",
    "graph_margin_ratio",
    "two_hop_neighbourhood_log",
    "issue_tokens_log",
    "has_stack_trace",
]


def route_features(issue: str, bm25_sorted_scores: list[float], seeds: dict[str, float],
                   graph_sorted_scores: list[float], bm25_top10: list[str], graph_top10: list[str],
                   two_hop_size: int, n_path_mentions: int | None = None) -> np.ndarray:
    b1 = bm25_sorted_scores[0] if bm25_sorted_scores else 0.0
    b2 = bm25_sorted_scores[1] if len(bm25_sorted_scores) > 1 else 0.0
    g1 = graph_sorted_scores[0] if graph_sorted_scores else 0.0
    g2 = graph_sorted_scores[1] if len(graph_sorted_scores) > 1 else 0.0
    overlap = len(set(bm25_top10) & set(graph_top10)) / 10.0
    stack = 1.0 if re.search(r"Traceback \(most recent call last\)|File \".*\", line \d+", issue) else 0.0
    return np.array([
        math.log1p(b1),
        (b1 - b2) / b1 if b1 > 0 else 0.0,
        math.log1p(len(seeds)),
        overlap,
        max(seeds.values(), default=0.0),
        g1,
        (g1 - g2) / g1 if g1 > 0 else 0.0,
        math.log1p(two_hop_size),
        math.log1p(len(issue.split())),
        stack,
    ], dtype=float)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class _Std:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class LogisticRouter:
    def __init__(self, l2=1e-2, lr=0.1, epochs=2000, seed=0):
        self.l2, self.lr, self.epochs, self.seed = l2, lr, epochs, seed

    def fit(self, X, y):
        self.s = _Std().fit(X)
        Z = self.s(X)
        self.w = np.zeros(Z.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sig(Z @ self.w + self.b)
            g = p - y
            self.w -= self.lr * (Z.T @ g / len(y) + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        return _sig(self.s(X) @ self.w + self.b)


class MLPRouter:
    def __init__(self, hidden=32, lr=0.05, epochs=1500, l2=1e-3, seed=0):
        self.h, self.lr, self.epochs, self.l2, self.seed = hidden, lr, epochs, l2, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.s = _Std().fit(X)
        Z = self.s(X)
        d = Z.shape[1]
        self.W0 = rng.normal(0, 1 / math.sqrt(d), (d, self.h))
        self.b0 = np.zeros(self.h)
        self.W1 = rng.normal(0, 1 / math.sqrt(self.h), (self.h, 1))
        self.b1 = np.zeros(1)
        n = len(y)
        for _ in range(self.epochs):
            a = np.tanh(Z @ self.W0 + self.b0)
            p = _sig(a @ self.W1 + self.b1)[:, 0]
            g = (p - y)[:, None] / n
            gW1 = a.T @ g + self.l2 * self.W1
            ga = (g @ self.W1.T) * (1 - a ** 2)
            gW0 = Z.T @ ga + self.l2 * self.W0
            self.W1 -= self.lr * gW1
            self.b1 -= self.lr * g.sum(0)
            self.W0 -= self.lr * gW0
            self.b0 -= self.lr * ga.sum(0)
        return self

    def predict_proba(self, X):
        a = np.tanh(self.s(X) @ self.W0 + self.b0)
        return _sig(a @ self.W1 + self.b1)[:, 0]


class PCRouter:
    """Two-layer predictive-coding network (10 -> 32 -> 1)."""

    def __init__(self, hidden=32, lr=0.1, epochs=300, infer_steps=10, infer_lr=0.2, l2=1e-3, seed=0):
        self.h, self.lr, self.epochs = hidden, lr, epochs
        self.T, self.ilr, self.l2, self.seed = infer_steps, infer_lr, l2, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.s = _Std().fit(X)
        x0 = self.s(X)
        d = x0.shape[1]
        self.W0 = rng.normal(0, 1 / math.sqrt(d), (d, self.h))
        self.b0 = np.zeros(self.h)
        self.W1 = rng.normal(0, 1 / math.sqrt(self.h), (self.h, 1))
        self.b1 = np.zeros(1)
        f = np.tanh
        Y = y[:, None].astype(float)
        n = len(y)
        self.energy_trace = []
        for _ in range(self.epochs):
            mu1 = x0 @ self.W0 + self.b0          # prediction of hidden layer from input layer
            x1 = mu1.copy()                       # initialise hidden state at its feed-forward value
            for _t in range(self.T):              # inference: relax x1 with x0 and label clamped
                fx1 = f(x1)
                mu2 = _sig(fx1 @ self.W1 + self.b1)
                e1 = x1 - mu1
                e2 = Y - mu2                      # output prediction error (Bernoulli / cross-entropy form)
                dx1 = -e1 + (e2 @ self.W1.T) * (1 - fx1 ** 2)
                x1 = x1 + self.ilr * dx1
            fx1 = f(x1)
            mu2 = _sig(fx1 @ self.W1 + self.b1)
            e1 = x1 - mu1
            e2 = Y - mu2
            self.energy_trace.append(float(0.5 * (e1 ** 2).sum() / n + 0.5 * (e2 ** 2).sum() / n))
            # local weight updates: each layer uses only its own pre-synaptic activity and post-synaptic error
            self.W1 += self.lr * (fx1.T @ e2 / n - self.l2 * self.W1)
            self.b1 += self.lr * e2.mean(0)
            self.W0 += self.lr * (x0.T @ e1 / n - self.l2 * self.W0)
            self.b0 += self.lr * e1.mean(0)
        return self

    def predict_proba(self, X):
        x1 = self.s(X) @ self.W0 + self.b0
        return _sig(np.tanh(x1) @ self.W1 + self.b1)[:, 0]


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U with tie handling
    allp = np.concatenate([pos, neg])
    order = allp.argsort()
    ranks = np.empty(len(allp))
    sp = allp[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
