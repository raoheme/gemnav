"""CLI:  python -m gemnav.cli query REPO "question" [--server URL]
        python -m gemnav.cli grammar            # print GBNF for llama-server --grammar-file
"""
from __future__ import annotations

import argparse
import json
import sys

from .client import LlamaServerClient
from .grammar import GBNF
from .loop import run_agent
from .tools import Toolbox


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gemnav")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("query")
    q.add_argument("repo")
    q.add_argument("question")
    q.add_argument("--server", default="http://127.0.0.1:8080")
    q.add_argument("--max-iters", type=int, default=10)
    q.add_argument("--budget", type=int, default=5800)
    q.add_argument("--log", default=None)
    sub.add_parser("grammar")
    a = ap.parse_args(argv)
    if a.cmd == "grammar":
        print(GBNF)
        return 0
    res = run_agent(a.question, Toolbox(a.repo), LlamaServerClient(a.server), a.max_iters, a.budget, a.log)
    print(json.dumps({k: v for k, v in res.items() if k != "trajectory"}, indent=1))
    print(res["answer"] or "(no answer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
