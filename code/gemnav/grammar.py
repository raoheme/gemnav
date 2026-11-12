"""GBNF grammar for llama.cpp plus a strict python-side validator (defence in depth).

The grammar fixes the key *order* and the tool-name vocabulary. Arguments are a flat object of
string/integer values with escape-aware strings, so the output is always valid JSON (the
blueprint's `[^"]*` string rule cannot encode a quote, backslash or newline, which code edits need).
Validation after parsing is still required: a grammar guarantees syntax, not semantics.
"""
from __future__ import annotations

import json

NAV_ARGS = {
    "search_code": {"pattern", "path"}, "get_definition": {"symbol"}, "find_references": {"symbol"},
    "read_file": {"path", "start", "end"}, "list_symbols": {"path"}, "semantic_search": {"query", "k"},
    "get_file_structure": {"path"}, "get_callers": {"symbol"},
}
REPAIR_ARGS = dict(NAV_ARGS, **{
    "replace_in_file": {"path", "old", "new"}, "create_file": {"path", "content"},
    "run_tests": set(), "submit": set(),
})
ALLOWED_ARGS = NAV_ARGS


# fixed, ordered argument schemas: (key, "s" string | "i" integer). Every key is required.
SCHEMAS = {
    "search_code": [("pattern", "s"), ("path", "s")], "get_definition": [("symbol", "s")],
    "find_references": [("symbol", "s")], "read_file": [("path", "s"), ("start", "i"), ("end", "i")],
    "list_symbols": [("path", "s")], "semantic_search": [("query", "s"), ("k", "i")],
    "get_file_structure": [("path", "s")], "get_callers": [("symbol", "s")],
    "replace_in_file": [("path", "s"), ("old", "s"), ("new", "s")], "create_file": [("path", "s"), ("content", "s")],
    "run_tests": [], "submit": [],
}


def build_strict_gbnf(tool_names, allow_answer: bool = False) -> str:
    """Per-tool grammar: tool name, then exactly that tool's argument keys in order with typed values."""
    alts, rules = [], []
    for t in tool_names:
        rid = "call-" + t.replace("_", "-")
        parts = []
        for k, typ in SCHEMAS[t]:
            parts.append(f'"\\"{k}\\"" ws ":" ws {"string" if typ == "s" else "integer"}')
        args = ' ws "," ws '.join(parts)
        body = f'"{{" ws {args} ws "}}"' if parts else '"{" ws "}"'
        rules.append(f'{rid} ::= "{{" ws "\\"tool\\"" ws ":" ws "\\"{t}\\"" ws "," ws "\\"args\\"" ws ":" ws {body} ws "}}"')
        alts.append(rid)
    root = "root ::= " + " | ".join(alts + (["final-answer"] if allow_answer else []))
    return "\n".join([root] + rules) + r'''
integer ::= "-"? [0-9]{1,9}
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})){0,600} "\""
final-answer ::= "{" ws "\"answer\"" ws ":" ws string ws "}"
ws ::= [ \t\n]{0,8}
'''


def build_gbnf(tool_names, allow_answer: bool = True) -> str:
    names = " | ".join(f'"\\"{t}\\""' for t in tool_names)
    root = "root ::= tool-call | final-answer" if allow_answer else "root ::= tool-call"
    return root + r'''
tool-call ::= "{" ws "\"tool\"" ws ":" ws tool-name ws "," ws "\"args\"" ws ":" ws args ws "}"
tool-name ::= ''' + names + r'''
args ::= "{" ws (pair (ws "," ws pair)*)? ws "}"
pair ::= key ws ":" ws value
key ::= "\"" [a-z_]+ "\""
value ::= string | integer
integer ::= "-"? [0-9]{1,9}
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
final-answer ::= "{" ws "\"answer\"" ws ":" ws string ws "}"
ws ::= [ \t\n]{0,8}
'''


GBNF = build_gbnf(list(NAV_ARGS))


class ProtocolError(ValueError):
    pass


def parse_action(text: str, allowed: dict | None = None) -> dict:
    allowed = allowed or ALLOWED_ARGS
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as ex:
        raise ProtocolError(f"invalid JSON: {ex}") from ex
    if not isinstance(obj, dict):
        raise ProtocolError("top-level value must be an object")
    if set(obj) == {"answer"} and isinstance(obj["answer"], str):
        return obj
    if set(obj) == {"tool", "args"} and obj["tool"] in allowed and isinstance(obj["args"], dict):
        extra = set(obj["args"]) - allowed[obj["tool"]]
        if extra:
            raise ProtocolError(f"unexpected args for {obj['tool']}: {sorted(extra)}")
        return obj
    raise ProtocolError("object is neither a valid tool call nor a final answer")
