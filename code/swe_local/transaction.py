"""Git-native transactions with a canonical state digest, plus baseline-identity regression logic.

Each repair turn: begin() -> apply candidate -> evaluate -> commit() or rollback().
rollback() restores tracked AND untracked (non-ignored) files with
`git reset --hard <checkpoint>` + `git clean -ffd`, then asserts that the canonical digest
equals the digest recorded at begin(). A mismatch raises RollbackIntegrityError.

Canonical digest fields
  head          : HEAD commit id
  index_tree    : tree id of the real index (git write-tree)
  worktree_tree : tree id of the full working tree incl. untracked non-ignored files
                  (computed through a throw-away GIT_INDEX_FILE, so the real index is untouched)
  status        : `git status --porcelain=v1 -uall` text hash
  harness_state : hash of caller-supplied JSON-serialisable state (observations, manifests, B0)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass


class RollbackIntegrityError(AssertionError):
    pass


def _git(repo: str, *args: str, env: dict | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def canonical_digest(repo: str, harness_state: dict | None = None) -> dict[str, str]:
    head = _git(repo, "rev-parse", "HEAD").strip()
    index_tree = _git(repo, "write-tree").strip()
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(td, "idx"))
        _git(repo, "read-tree", "HEAD", env=env)
        _git(repo, "add", "-A", env=env)
        wt = _git(repo, "write-tree", env=env).strip()
    status = hashlib.sha256(_git(repo, "status", "--porcelain=v1", "-uall").encode()).hexdigest()
    hs = hashlib.sha256(json.dumps(harness_state or {}, sort_keys=True, default=str).encode()).hexdigest()
    return {"head": head, "index_tree": index_tree, "worktree_tree": wt, "status": status, "harness_state": hs}


def digest_hex(d: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


@dataclass
class GitNativeTransaction:
    repo: str
    clean_ignored: bool = False

    def begin(self, harness_state: dict | None = None) -> dict[str, str]:
        self._state = json.loads(json.dumps(harness_state or {}, default=str))
        self.checkpoint = _git(self.repo, "rev-parse", "HEAD").strip()
        self.before = canonical_digest(self.repo, self._state)
        return self.before

    def commit(self, message: str = "agent turn") -> str:
        _git(self.repo, "add", "-A")
        _git(self.repo, "-c", "user.name=agent", "-c", "user.email=agent@local", "commit", "-q",
             "--allow-empty", "--no-verify", "-m", message)
        self.checkpoint = _git(self.repo, "rev-parse", "HEAD").strip()
        return self.checkpoint

    def rollback(self) -> dict:
        _git(self.repo, "reset", "-q", "--hard", self.checkpoint)
        _git(self.repo, "clean", "-ffdxq" if self.clean_ignored else "-ffdq")
        after = canonical_digest(self.repo, self._state)
        if after != self.before:
            diff = {k: (self.before[k], after[k]) for k in after if after[k] != self.before[k]}
            raise RollbackIntegrityError(f"digest mismatch after rollback: {diff}")
        return {"state": self._state, "digest": after}


# ----------------------------------------------------------------------------- regression logic
PASS, FAIL, SKIP, ERROR = "PASS", "FAIL", "SKIP", "ERROR"


def regression_clean(b0: dict[str, str], outcome: dict[str, str]) -> tuple[bool, list[str]]:
    """Baseline identity: every test that passed on the clean checkout must still pass."""
    broken = [t for t, s in b0.items() if s == PASS and outcome.get(t) != PASS]
    return (len(broken) == 0, broken)


def turn_score(b0: dict[str, str], outcome: dict[str, str], repro_status: str) -> tuple[int, int, int]:
    """Lexicographic objective (regression_clean, -n_broken, R) with R in {2 PASS, 1 TARGET_FAILURE, 0 other}."""
    ok, broken = regression_clean(b0, outcome)
    r = {"PASS": 2, "TARGET_FAILURE": 1}.get(repro_status, 0)
    return (int(ok), -len(broken), r)
