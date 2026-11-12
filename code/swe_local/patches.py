"""Minimal unified-diff parsing (no external dependencies)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    old_changed_lines: set[int] = field(default_factory=set)  # 1-based lines in the original file
    added_lines: list[str] = field(default_factory=list)
    deleted_lines: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def is_new_file(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted_file(self) -> bool:
        return self.new_path is None


def _strip(p: str) -> str | None:
    p = p.strip().split("\t")[0]
    if p == "/dev/null":
        return None
    if p.startswith("a/") or p.startswith("b/"):
        return p[2:]
    return p


def parse_unified_diff(diff: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    old_ln = 0
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            cur = FilePatch(_strip(line[4:]), _strip(lines[i + 1][4:]))
            files.append(cur)
            i += 2
            continue
        if cur is None:
            i += 1
            continue
        m = _HUNK.match(line)
        if m:
            old_ln = int(m.group(1))
            i += 1
            continue
        if line.startswith("-") and not line.startswith("---"):
            cur.old_changed_lines.add(old_ln)
            cur.deleted_lines.append(line[1:])
            old_ln += 1
        elif line.startswith("+") and not line.startswith("+++"):
            # insertion: anchored to the original line it follows (robust to appends at the end of a
            # function). Lines replacing deleted lines are already covered by the '-' lines.
            if not (i > 0 and lines[i - 1][:1] in ("-", "+") and not lines[i - 1].startswith(("---", "+++"))):
                cur.old_changed_lines.add(max(old_ln - 1, 1))
            cur.added_lines.append(line[1:])
        elif line.startswith(" "):
            old_ln += 1
        i += 1
    return files


def touched_paths(diff: str) -> list[str]:
    out = []
    for fp in parse_unified_diff(diff):
        for p in (fp.old_path, fp.new_path):
            if p and p not in out:
                out.append(p)
    return out
