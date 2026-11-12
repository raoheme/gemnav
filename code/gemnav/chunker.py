"""Tree-sitter semantic chunking (Python). Each chunk is a complete AST node, never split mid-function.

Top-level function/class/decorated definitions become chunks. Classes longer than `max_class_lines`
are split into a header chunk (signature + class-level statements) plus one chunk per method, with
`parent_scope` = class name. Remaining top-level statements are grouped into one `module` chunk per
file (imports, constants, assignments) so module-level code is still retrievable.
"""
from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY = Language(tspython.language())
_PARSER = Parser(PY)
DEF_TYPES = {"function_definition", "class_definition", "decorated_definition"}


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    node_type: str      # function | class | method | class_header | module
    identifier: str
    parent_scope: str
    raw_text: str


def _inner(node):
    """decorated_definition -> the function/class it wraps."""
    if node.type == "decorated_definition":
        d = node.child_by_field_name("definition")
        return d if d is not None else node
    return node


def _name(node, src: bytes) -> str:
    n = _inner(node).child_by_field_name("name")
    return src[n.start_byte:n.end_byte].decode("utf-8", "replace") if n is not None else ""


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def chunk_file(source: bytes | str, file_path: str, max_class_lines: int = 200) -> list[Chunk]:
    src = source.encode("utf-8") if isinstance(source, str) else source
    tree = _PARSER.parse(src)
    chunks: list[Chunk] = []
    module_lines: list[str] = []
    mod_start = None
    mod_end = 0
    for node in tree.root_node.children:
        if node.type not in DEF_TYPES:
            module_lines.append(_text(node, src))
            mod_start = node.start_point[0] + 1 if mod_start is None else mod_start
            mod_end = node.end_point[0] + 1
            continue
        inner = _inner(node)
        name = _name(node, src)
        s, e = node.start_point[0] + 1, node.end_point[0] + 1
        if inner.type == "function_definition":
            chunks.append(Chunk(file_path, s, e, "function", name, "", _text(node, src)))
            continue
        # class
        if e - s + 1 <= max_class_lines:
            chunks.append(Chunk(file_path, s, e, "class", name, "", _text(node, src)))
            continue
        body = inner.child_by_field_name("body")
        header_parts = [src[node.start_byte:body.start_byte].decode("utf-8", "replace") if body is not None else _text(node, src)]
        for st in (body.children if body is not None else []):
            if st.type in DEF_TYPES:
                ms, me = st.start_point[0] + 1, st.end_point[0] + 1
                chunks.append(Chunk(file_path, ms, me, "method", _name(st, src), name, _text(st, src)))
            else:
                header_parts.append(_text(st, src))
        chunks.append(Chunk(file_path, s, e, "class_header", name, "", "\n".join(header_parts)))
    if module_lines:
        chunks.append(Chunk(file_path, mod_start or 1, mod_end or 1, "module", file_path.rsplit("/", 1)[-1][:-3], "",
                            "\n".join(module_lines)[:8000]))
    return chunks


def symbols(chunks: list[Chunk]) -> list[tuple[str, str, int, str, str]]:
    """(symbol_name, file_path, line, kind, scope) rows for the `symbols` table."""
    return [(c.identifier, c.file_path, c.start_line, c.node_type, c.parent_scope)
            for c in chunks if c.node_type in ("function", "class", "method", "class_header")]
