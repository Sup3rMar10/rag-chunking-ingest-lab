"""Parse markdown-ish documentation into typed blocks and heading paths.

Block kinds: heading, para, list, table, code. Fenced code is recognised first so a
"# comment" line inside a code block is never mistaken for a heading.
"""

import bisect
import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
LIST_ITEM = re.compile(r"^\s*(?:\d+\.|[-*])\s+")


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    start: int
    end: int
    level: int = 0
    title: str = ""


def parse_blocks(text):
    blocks = []
    cur_kind = None
    cur_lines = []
    cur_start = 0
    cur_end = 0
    pos = 0
    in_fence = False

    def flush():
        nonlocal cur_kind, cur_lines
        if cur_kind is not None and cur_lines:
            blocks.append(Block(cur_kind, "\n".join(cur_lines), cur_start, cur_end))
        cur_kind, cur_lines = None, []

    def begin(kind, line_start, line_end, line):
        nonlocal cur_kind, cur_lines, cur_start, cur_end
        cur_kind, cur_lines, cur_start, cur_end = kind, [line], line_start, line_end

    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        line_start, line_end = pos, pos + len(line)
        pos += len(raw)
        if in_fence:
            cur_lines.append(line)
            cur_end = line_end
            if line.strip().startswith("```"):
                flush()
                in_fence = False
            continue
        if line.strip().startswith("```"):
            flush()
            begin("code", line_start, line_end, line)
            in_fence = True
            continue
        if not line.strip():
            flush()
            continue
        m = HEADING.match(line)
        if m:
            flush()
            blocks.append(
                Block("heading", line, line_start, line_end, len(m.group(1)), m.group(2))
            )
            continue
        if line.lstrip().startswith("|"):
            kind = "table"
        elif LIST_ITEM.match(line) or (cur_kind == "list" and line[:1] in " \t"):
            kind = "list"
        else:
            kind = "para"
        if kind != cur_kind:
            flush()
            begin(kind, line_start, line_end, line)
        else:
            cur_lines.append(line)
            cur_end = line_end
    flush()
    return blocks


def title_of(text, default=""):
    for b in parse_blocks(text):
        if b.kind == "heading" and b.level == 1:
            return b.title
    return default


def heading_paths(text):
    """Sorted [(offset, path)] where path is 'H1 > H2 > H3' in force from offset on."""
    out = []
    stack = []
    for b in parse_blocks(text):
        if b.kind != "heading":
            continue
        while stack and stack[-1][0] >= b.level:
            stack.pop()
        stack.append((b.level, b.title))
        out.append((b.start, " > ".join(t for _, t in stack)))
    return out


def path_at(paths, offset, default=""):
    """The heading path in force at a character offset."""
    if not paths:
        return default
    i = bisect.bisect_right([p[0] for p in paths], offset) - 1
    return paths[i][1] if i >= 0 else default
