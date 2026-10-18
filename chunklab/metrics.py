"""Retrieval metrics that depend on the chunking, not just on ranking.

A question is ANSWERED by a retrieved context only if the context belongs to the gold
document and contains every required span (`must_contain`) in one piece. So a table row
cut away from its header, or half of a numbered procedure, does not count.

  cover@k    some context in the top k answers the question (answer-span coverage)
  dochit@k   some context in the top k comes from the gold document (it was located)
  ctx@k      words handed to the model for the top k
"""

import math

from .text import count_tokens, norm

KS = (1, 3, 5)


def answers(question, chunk, core=False):
    """core=True is the lenient answer-only check (a table row without its header)."""
    if chunk.doc_id != question.doc_id:
        return False
    ctx = norm(chunk.context_text)
    spans = question.core_spans() if core else question.must_contain
    return all(norm(m) in ctx for m in spans)


def wilson(successes, n, z=1.96):
    """95% Wilson interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_p(a, b):
    """Exact two-sided McNemar p-value for paired boolean outcomes a and b.
    Returns (only_a_right, only_b_right, p)."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * tail)


def evaluate(index, questions, ks=KS):
    kmax = max(ks)
    rows = []
    for q in questions:
        res = index.search(q.text, kmax)
        first = next((i + 1 for i, c in enumerate(res) if answers(q, c)), None)
        first_core = next((i + 1 for i, c in enumerate(res) if answers(q, c, core=True)), None)
        rows.append((q, res, first, first_core))
    n = len(rows)
    out = {"n": n, "cover": {}, "cover_ci": {}, "dochit": {}, "ctx": {}, "by_kind": {},
           "outcomes": {}, "cover_core": {}, "by_fmt": {}}
    for k in ks:
        hits = sum(1 for _, _, first, _ in rows if first is not None and first <= k)
        out["cover"][k] = hits / n if n else 0.0
        out["outcomes"][k] = [first is not None and first <= k for _, _, first, _ in rows]
        out["cover_ci"][k] = wilson(hits, n)
        out["cover_core"][k] = (
            sum(1 for *_, fc in rows if fc is not None and fc <= k) / n if n else 0.0
        )
        out["dochit"][k] = (
            sum(1 for q, res, _, _ in rows if any(c.doc_id == q.doc_id for c in res[:k])) / n
            if n else 0.0
        )
        out["ctx"][k] = (
            sum(count_tokens(c.context_text) for _, res, _, _ in rows for c in res[:k]) / n
            if n else 0.0
        )
    out["mrr"] = sum(1 / first for _, _, first, _ in rows if first) / n if n else 0.0
    for kind in sorted({q.kind for q in questions}):
        sub = [(q, first) for q, _, first, _ in rows if q.kind == kind]
        out["by_kind"][kind] = {
            "n": len(sub),
            **{k: sum(1 for _, f in sub if f is not None and f <= k) / len(sub) for k in ks},
        }
    for fmt in sorted({q.fmt for q in questions}):
        sub = [first for q, _, first, _ in rows if q.fmt == fmt]
        out["by_fmt"][fmt] = {
            "n": len(sub),
            **{k: sum(1 for f in sub if f is not None and f <= k) / len(sub) for k in ks},
        }
    return out
