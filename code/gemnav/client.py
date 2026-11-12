"""llama.cpp `llama-server` client (OpenAI-compatible chat endpoint + GBNF) and a scripted stub for tests.

Every call records llama-server's own `timings` (prompt_n, prompt_ms, predicted_n, predicted_ms) plus wall time,
so latency numbers in the paper come from the server, not from estimates.
"""
from __future__ import annotations

import json
import time
import urllib.request

from .grammar import GBNF


class LlamaServerClient:
    def __init__(self, url: str = "http://127.0.0.1:8080", temperature: float = 0.0, n_predict: int = 768,
                 seed: int = 42, timeout: int = 1800):
        self.url = url.rstrip("/").removesuffix("/v1")
        self.temperature, self.n_predict, self.seed, self.timeout = temperature, n_predict, seed, timeout
        self.calls: list[dict] = []

    # DRY ("don't repeat yourself") sampler: penalises long verbatim repeats (e.g. runaway "\\n\\n\\n..." inside a JSON
    # string) while allowing short ones (indentation); deterministic under top_k=1, so greedy decoding is preserved.
    DRY = {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 4, "dry_penalty_last_n": 512}

    def chat(self, system: str, user: str, grammar: str | None = GBNF, n_predict: int | None = None, dry: bool = False) -> str:
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.temperature, "top_k": 1 if self.temperature == 0 else 40, "seed": self.seed,
                "max_tokens": n_predict or self.n_predict, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        if grammar:
            body["grammar"] = grammar
        if dry:
            body.update(self.DRY)
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.loads(r.read())
        wall = time.perf_counter() - t0
        t = res.get("timings", {})
        u = res.get("usage", {})
        self.calls.append({"wall_s": wall, "prompt_n": t.get("prompt_n", u.get("prompt_tokens")),
                           "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n", u.get("completion_tokens")),
                           "predicted_ms": t.get("predicted_ms"), "cache_n": t.get("cache_n"),
                           "finish": res["choices"][0].get("finish_reason")})
        return res["choices"][0]["message"]["content"]

    def complete(self, prompt: str) -> str:
        """Navigation loop entry point: the whole rendered context goes in the user turn."""
        from .context import SYSTEM
        return self.chat(SYSTEM, prompt)


class ScriptedClient:
    """Deterministic stand-in used by unit tests. NOT a model; used only to test the harness."""

    def __init__(self, outputs: list[str]):
        self.outputs, self.prompts, self.calls = list(outputs), [], []

    def chat(self, system: str, user: str, grammar=None, n_predict=None, dry=False) -> str:
        self.prompts.append(user)
        self.calls.append({"wall_s": 0.0})
        return self.outputs.pop(0) if self.outputs else json.dumps({"tool": "submit", "args": {}})

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0) if self.outputs else json.dumps({"answer": "no more scripted output"})
