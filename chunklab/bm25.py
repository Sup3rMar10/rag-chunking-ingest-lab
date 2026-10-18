"""BM25 in the standard library. Deterministic: ties break by document position."""

import math
from collections import Counter


class BM25:
    def __init__(self, docs_terms, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_terms)
        self.lengths = [len(d) for d in docs_terms]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings = {}
        for i, terms in enumerate(docs_terms):
            for term, tf in Counter(terms).items():
                self.postings.setdefault(term, []).append((i, tf))

    def idf(self, term):
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_terms):
        acc = {}
        for term in set(query_terms):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf(term)
            for i, tf in posting:
                norm = 1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1.0)
                acc[i] = acc.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return acc

    def top(self, query_terms, n):
        scored = self.scores(query_terms)
        return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
