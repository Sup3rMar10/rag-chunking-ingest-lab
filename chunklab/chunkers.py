"""Five ways to cut a document into retrieval units.

fixed        word windows with overlap. Structure-blind; the usual baseline.
sentence     sentences packed up to a size. Oversize "sentences" (a table or code block
             has no punctuation) are hard-split by words.
recursive    a recursive splitter (paragraph, then line, then sentence, then word) that
             merges small pieces back up to the size. The widely used strong baseline.
structure    parses headings, tables, lists and code. Tables, lists and code blocks stay
             whole when they fit; oversize tables split by rows with the header repeated;
             a heading is never left orphaned at the end of a chunk. Chunks stay within
             the size except that a heading glued to a block that does not fit beside it
             may exceed it by the heading's own length (and a single table row wider than
             the size is emitted alone).
parent_child small windows are indexed, but the whole parent section is returned
             ("small to big").

Every strategy yields the same Chunk record. `text` is what is indexed, `context_text`
is what would be handed to the model (the parent for parent_child, else the same text),
`prefix` is the optional metadata line (document title or heading path) that is indexed
together with the text but is NOT part of the span a question must find.
"""

import hashlib
import json
import re
from dataclasses import dataclass

from .blocks import heading_paths, parse_blocks, path_at, title_of
from .text import count_tokens, split_sentences, word_spans

STRATEGIES = ("fixed", "sentence", "recursive", "structure", "parent_child")
ENRICH = ("none", "title", "path")
SEPARATORS = ("\n\n", "\n", ". ", " ")
TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@dataclass(frozen=True)
class ChunkerConfig:
    strategy: str = "structure"
    size: int = 128  # words per chunk (child size for parent_child)
    overlap: int = -1  # words; -1 means size // 8 (size // 4 for parent_child children)
    enrich: str = "title"
    parent_size: int = 400

    def resolved_overlap(self):
        if self.overlap >= 0:
            return self.overlap
        return self.size // 4 if self.strategy == "parent_child" else self.size // 8

    def validate(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.enrich not in ENRICH:
            raise ValueError(f"unknown enrich mode {self.enrich!r}")
        if self.size < 4:
            raise ValueError("size must be at least 4 words")
        if self.resolved_overlap() >= self.size:
            raise ValueError("overlap must be smaller than size")
        if self.strategy == "parent_child" and self.parent_size < self.size:
            raise ValueError("parent_size must be at least the child size")

    def fingerprint(self):
        return json.dumps(
            [self.strategy, self.size, self.resolved_overlap(), self.enrich, self.parent_size]
        )


@dataclass(frozen=True)
class Chunk:
    key: str  # identity of this stored record (doc + prefix + text + context)
    embed_key: str  # identity of the text that gets embedded (prefix + text only)
    doc_id: str
    ordinal: int
    text: str
    context_text: str
    prefix: str
    path: str

    @property
    def embed_text(self):
        return f"{self.prefix}\n{self.text}" if self.prefix else self.text


def _h(*parts):
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- fixed
def _fixed_windows(text, size, overlap):
    """[(start, end, anchor)] character offsets; anchor = start of the non-overlap part."""
    spans = word_spans(text)
    n = len(spans)
    if n == 0:
        return []
    step = size - overlap
    out = []
    k = 0
    while True:
        a = k * step
        b = min(n, a + size)
        anchor_idx = min(a + (overlap if k > 0 else 0), n - 1)
        out.append((spans[a][0], spans[b - 1][1], spans[anchor_idx][0]))
        if b == n:
            break
        k += 1
    return out


def _chunk_fixed(text, cfg):
    return [(text[s:e], text[s:e], anchor) for s, e, anchor in
            _fixed_windows(text, cfg.size, cfg.resolved_overlap())]


# ------------------------------------------------------------ generic unit packer
def _pack_units(unit_tokens, size, overlap):
    """Greedy packing of consecutive units. Returns [(first, last_excl, first_new)];
    the units before first_new are overlap carried from the previous chunk."""
    chunks = []
    cur = []
    cur_t = 0
    first_new = 0
    for idx, t in enumerate(unit_tokens):
        if cur and cur_t + t > size:
            chunks.append((cur[0], cur[-1] + 1, first_new))
            carry, ct = [], 0
            for j in reversed(cur):
                if ct + unit_tokens[j] > overlap:
                    break
                carry.insert(0, j)
                ct += unit_tokens[j]
            if ct + t > size:
                carry, ct = [], 0
            cur, cur_t = carry, ct
            first_new = idx
        elif not cur:
            first_new = idx
        cur.append(idx)
        cur_t += t
    if cur:
        chunks.append((cur[0], cur[-1] + 1, first_new))
    return chunks


def _hard_split(text, base, size):
    """Tile text into word windows of at most `size` words, no overlap, exact cover."""
    spans = word_spans(text)
    if len(spans) <= size:
        return [(base, base + len(text))]
    units = []
    prev = 0
    for a in range(0, len(spans), size):
        b = min(len(spans), a + size)
        end = len(text) if b == len(spans) else spans[b - 1][1]
        units.append((base + prev, base + end))
        prev = end
    return units


def _pack_text_units(text, units, cfg):
    tokens = [count_tokens(text[s:e]) for s, e in units]
    out = []
    for first, last, first_new in _pack_units(tokens, cfg.size, cfg.resolved_overlap()):
        s, e = units[first][0], units[last - 1][1]
        chunk = text[s:e].strip()
        if not chunk:
            continue
        anchor = units[first_new][0]
        anchor += len(text[anchor:]) - len(text[anchor:].lstrip())
        out.append((chunk, chunk, anchor))
    return out


# ------------------------------------------------------------------ sentence
def _sentence_units(text, size):
    units = []
    last = 0
    pieces = []
    for m in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text):
        pieces.append((last, m.start()))
        last = m.end()
    pieces.append((last, len(text)))
    for s, e in pieces:
        seg = text[s:e]
        if not seg.strip():
            continue
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        s2, e2 = s + lead, e - trail
        units.extend(_hard_split(text[s2:e2], s2, size))
    return units


def _chunk_sentence(text, cfg):
    return _pack_text_units(text, _sentence_units(text, cfg.size), cfg)


# ------------------------------------------------------------------ recursive
def _recursive_units(text, base, size, seps):
    if count_tokens(text) <= size:
        return [(base, base + len(text))]
    if not seps:
        return _hard_split(text, base, size)
    sep = seps[0]
    out = []
    pos = 0
    while pos < len(text):
        j = text.find(sep, pos)
        end = len(text) if j == -1 else j + len(sep)
        piece = text[pos:end]
        if count_tokens(piece) <= size:
            out.append((base + pos, base + end))
        else:
            out.extend(_recursive_units(piece, base + pos, size, seps[1:]))
        pos = end
    return out


def _chunk_recursive(text, cfg):
    return _pack_text_units(text, _recursive_units(text, 0, cfg.size, SEPARATORS), cfg)


# ------------------------------------------------------------------ structure
def _atoms_for_block(block, size, pid):
    """Split one block into atoms of at most `size` words where possible.
    Returns [(text, kind, para_id_or_None)]."""
    text = block.text
    if block.kind == "heading" or count_tokens(text) <= size:
        return [(text, block.kind, None)]
    if block.kind == "para":
        sents = []
        for s in split_sentences(text):
            sents.extend(s[a:b] for a, b in _hard_split(s, 0, size))
        return [(t, "para", pid) for t in _greedy_join(sents, size, " ")]
    lines = text.split("\n")
    if block.kind == "table":
        header = []
        if len(lines) >= 3 and TABLE_RULE.match(lines[1]):
            header, lines = lines[:2], lines[2:]
        budget = max(1, size - sum(count_tokens(h) for h in header))
        return [("\n".join(header + g), "table", None) for g in _group_lines(lines, budget)]
    if block.kind == "list":
        items = []
        for ln in lines:
            if items and ln[:1] in (" ", "\t"):
                items[-1] += "\n" + ln
            else:
                items.append(ln)
        return [(g, "list", None) for g in _greedy_join(items, size, "\n")]
    return [(g, block.kind, None) for g in _greedy_join(lines, size, "\n")]  # code


def _group_lines(lines, budget):
    groups, cur, ct = [], [], 0
    for ln in lines:
        t = count_tokens(ln)
        if cur and ct + t > budget:
            groups.append(cur)
            cur, ct = [], 0
        cur.append(ln)
        ct += t
    if cur:
        groups.append(cur)
    return groups


def _greedy_join(parts, size, joiner):
    out, cur, ct = [], [], 0
    for p in parts:
        t = count_tokens(p)
        if cur and ct + t > size:
            out.append(joiner.join(cur))
            cur, ct = [], 0
        cur.append(p)
        ct += t
    if cur:
        out.append(joiner.join(cur))
    return out


def _trailing_sentences(text, limit):
    picked, total = [], 0
    for s in reversed(split_sentences(text)):
        t = count_tokens(s)
        if total + t > limit:
            break
        picked.insert(0, s)
        total += t
    return " ".join(picked)


def _chunk_structure(text, cfg, size=None, overlap=None):
    size = size or cfg.size
    overlap = cfg.resolved_overlap() if overlap is None else overlap
    sections = []  # (path, [blocks])
    stack = []
    cur = None
    for b in parse_blocks(text):
        if b.kind == "heading":
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.title))
            cur = (" > ".join(t for _, t in stack), [b])
            sections.append(cur)
        elif cur is None:
            cur = ("", [b])
            sections.append(cur)
        else:
            cur[1].append(b)

    out = []
    pid = 0
    for path, blocks in sections:
        atoms = []
        for b in blocks:
            pid += 1
            atoms.extend(_atoms_for_block(b, size, pid))
        cur_atoms, cur_t = [], 0

        def flush():
            out.append(("\n\n".join(a[0] for a in cur_atoms), path))

        for atom in atoms:
            t = count_tokens(atom[0])
            only_heading = len(cur_atoms) == 1 and cur_atoms[0][1] == "heading"
            if cur_atoms and cur_t + t > size and not only_heading:
                flush()
                last = cur_atoms[-1]
                cur_atoms, cur_t = [], 0
                if atom[2] is not None and atom[2] == last[2] and overlap:
                    carry = _trailing_sentences(last[0], overlap)
                    ct = count_tokens(carry)
                    if carry and ct + t <= size:
                        cur_atoms, cur_t = [(carry, "carry", None)], ct
            cur_atoms.append(atom)
            cur_t += t
        if cur_atoms:
            flush()
    return [(t, t, p) for t, p in out]  # path is returned in the third slot


def _chunk_parent_child(text, cfg):
    parents = _chunk_structure(text, cfg, size=cfg.parent_size, overlap=0)
    out = []
    for ptext, _, path in parents:
        for s, e, _ in _fixed_windows(ptext, cfg.size, cfg.resolved_overlap()):
            out.append((ptext[s:e], ptext, path))
    return out


# --------------------------------------------------------------------- public
def chunk_document(doc_id, text, cfg):
    cfg.validate()
    title = title_of(text, doc_id)
    paths = heading_paths(text)
    if cfg.strategy == "fixed":
        raw = _chunk_fixed(text, cfg)
    elif cfg.strategy == "sentence":
        raw = _chunk_sentence(text, cfg)
    elif cfg.strategy == "recursive":
        raw = _chunk_recursive(text, cfg)
    elif cfg.strategy == "structure":
        raw = _chunk_structure(text, cfg)
    else:
        raw = _chunk_parent_child(text, cfg)

    chunks, seen = [], set()
    for body, context, where in raw:
        if cfg.strategy in ("structure", "parent_child"):
            path = where or title
        else:
            path = path_at(paths, where, title)
        prefix = {"none": "", "title": title, "path": path}[cfg.enrich]
        key = _h(doc_id, prefix, body, context)
        if key in seen:
            continue
        seen.add(key)
        chunks.append(
            Chunk(
                key=key,
                embed_key=_h(prefix, body),
                doc_id=doc_id,
                ordinal=len(chunks),
                text=body,
                context_text=context,
                prefix=prefix,
                path=path,
            )
        )
    return chunks
