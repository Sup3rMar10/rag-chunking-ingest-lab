"""A BM25 index over chunks. Searching returns retrieval units, de-duplicated by the
context they would hand to the model (so two child windows of one parent count once)."""

from .bm25 import BM25
from .text import terms

CANDIDATE_DEPTH = 60


class Index:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self._bm25 = BM25([terms(f"{c.prefix} {c.text}") for c in self.chunks])

    def __len__(self):
        return len(self.chunks)

    def search(self, query, k):
        """Top-k distinct contexts, best first. Each result is the matching Chunk."""
        out, seen = [], set()
        for i, _ in self._bm25.top(terms(query), CANDIDATE_DEPTH):
            c = self.chunks[i]
            ident = (c.doc_id, c.context_text)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(c)
            if len(out) == k:
                break
        return out
