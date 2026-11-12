"""Agent loop: generate -> parse/validate -> execute tool -> collapse history -> repeat.

Safeguards: repeated identical calls are answered with a cached notice (not re-executed),
malformed output is fed back as an error observation (max 2 consecutive), hard iteration cap,
every prompt/response/tool result is appended to a JSONL trajectory log (append-only).
"""
from __future__ import annotations

import json
import time

from .context import Context, toks
from .grammar import ProtocolError, parse_action
from .tools import Toolbox


def run_agent(query: str, tb: Toolbox, client, max_iters: int = 10, budget: int = 5800, log_path: str | None = None) -> dict:
    ctx = Context(query, tb.file_summary_map(), budget=budget)
    seen: set[str] = set()
    bad = 0
    traj = []
    final = None
    for i in range(max_iters):
        prompt = ctx.render()
        t0 = time.perf_counter()
        raw = client.complete(prompt)
        dt = time.perf_counter() - t0
        rec = {"iter": i, "prompt_tokens_approx": toks(prompt), "latency_s": dt, "raw": raw}
        try:
            act = parse_action(raw)
        except ProtocolError as ex:
            bad += 1
            rec["error"] = str(ex)
            traj.append(rec)
            if bad > 2:
                break
            ctx.add("protocol_error", {}, f"Your last output was invalid: {ex}. Reply with one JSON object.")
            continue
        bad = 0
        if "answer" in act:
            final = act["answer"]
            traj.append(rec)
            break
        key = json.dumps(act, sort_keys=True)
        if key in seen:
            result = "(duplicate call - see HISTORY; choose a different action or answer)"
        else:
            seen.add(key)
            result = tb.dispatch(act["tool"], act["args"])
        rec.update({"tool": act["tool"], "args": act["args"], "result_chars": len(result)})
        traj.append(rec)
        ctx.add(act["tool"], act["args"], result)
    if log_path:
        with open(log_path, "a") as f:
            for r in traj:
                f.write(json.dumps(r) + "\n")
    return {"answer": final, "iterations": len(traj), "max_prompt_tokens": max((r["prompt_tokens_approx"] for r in traj), default=0),
            "trajectory": traj}
