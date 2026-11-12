"""Hierarchical context: always-on file summary map + sliding window of tool observations.

Older observations are collapsed to one-line digests (tool, args, first line, size) so the model
never re-reads them, and the total prompt stays under a hard token budget. Digests are produced
deterministically (no extra LLM call). Tokens are approximated as ceil(chars/4).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

SYSTEM = ("You are a code-navigation agent. You cannot see the repository directly; use tools. "
          "Reply with exactly one JSON object: either {\"tool\": NAME, \"args\": {...}} or {\"answer\": TEXT}. "
          "Cite file:line in answers. Do not repeat a tool call that already appears in HISTORY.")


def toks(s: str) -> int:
    return (len(s) + 3) // 4


@dataclass
class Context:
    query: str
    summary_map: str
    budget: int = 5800
    keep_full: int = 2
    steps: list[tuple[str, dict, str]] = field(default_factory=list)

    def add(self, tool: str, args: dict, result: str) -> None:
        self.steps.append((tool, args, result))

    def _digest(self, tool, args, result) -> str:
        first = result.strip().splitlines()[0][:100] if result.strip() else ""
        return f"- {tool}({json.dumps(args, sort_keys=True)}) -> {len(result)} chars; first line: {first}"

    def render(self) -> str:
        head = f"{SYSTEM}\n\nFILE MAP:\n{self.summary_map}\n\nQUESTION: {self.query}\n"
        n = len(self.steps)
        parts_old = [self._digest(*s) for s in self.steps[: max(0, n - self.keep_full)]]
        recent = self.steps[max(0, n - self.keep_full):]

        def build(recent_steps, old):
            out = head
            if old:
                out += "HISTORY (collapsed):\n" + "\n".join(old) + "\n"
            for t, a, r in recent_steps:
                out += f"CALL {t} {json.dumps(a, sort_keys=True)}\nRESULT:\n{r}\n"
            return out + "NEXT ACTION (JSON):\n"

        text = build(recent, parts_old)
        # shrink: truncate the largest recent result, then collapse oldest recent into history
        while toks(text) > self.budget and recent:
            big = max(range(len(recent)), key=lambda i: len(recent[i][2]))
            t, a, r = recent[big]
            if len(r) > 600:
                recent[big] = (t, a, r[: len(r) // 2] + "\n...[trimmed]")
            else:
                parts_old.append(self._digest(*recent[0]))
                recent = recent[1:]
            text = build(recent, parts_old)
        while toks(text) > self.budget and len(parts_old) > 1:  # last resort: drop oldest digests
            parts_old.pop(0)
            text = build(recent, parts_old)
        return text
