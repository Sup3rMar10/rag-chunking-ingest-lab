"""Incremental ingestion with content hashes, tombstones and a content-addressed
embedding cache.

The rules:
  * A document is re-chunked only when its text hash or the chunker config changed.
  * Chunks are identified by content (doc + prefix + text + context). A chunk that
    survives an edit is kept as is; a new chunk gets embedded; a chunk that no longer
    exists is TOMBSTONED (hidden from search, still stored for `grace` runs).
  * Embeddings are cached by the text that is embedded, so identical text is embedded
    once, and a chunk revived inside the grace period needs no new embedding.
  * Each document is a transaction. If embedding fails half way, the document keeps its
    previous version, nothing partial becomes visible, and the next run retries it.
  * Re-running with the same source changes nothing (idempotent).
  * `compact()` physically drops old tombstones and unreferenced embeddings.

The "embedding" is a deterministic hashing vector. It stands in for a model call so the
cost can be counted; swap `embed` for a real one.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field

from .chunkers import Chunk, ChunkerConfig, chunk_document
from .retrieval import Index
from .text import terms


def hash_embed(text, dim=32):
    vec = [0.0] * dim
    for t in terms(text):
        h = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class IngestReport:
    run: int = 0
    docs_added: int = 0
    docs_updated: int = 0
    docs_unchanged: int = 0
    docs_deleted: int = 0
    docs_failed: list = field(default_factory=list)
    chunks_added: int = 0
    chunks_kept: int = 0
    chunks_revived: int = 0
    chunks_tombstoned: int = 0
    embeds_computed: int = 0
    embeds_reused: int = 0

    def changed(self):
        return bool(self.docs_added or self.docs_updated or self.docs_deleted
                    or self.chunks_added or self.chunks_tombstoned or self.chunks_revived)


@dataclass
class StaleReport:
    changed_docs: list
    missing_docs: list  # in the source, not in the store
    orphan_docs: list  # in the store, gone from the source
    config_drift_docs: list
    stale_chunk_keys: list  # live chunks the next ingest would tombstone

    def clean(self):
        return not (self.changed_docs or self.missing_docs or self.orphan_docs
                    or self.config_drift_docs or self.stale_chunk_keys)


class IngestStore:
    def __init__(self, cfg=None, embed=hash_embed, grace=2):
        self.cfg = cfg or ChunkerConfig()
        self.cfg.validate()
        self.embed = embed
        self.grace = grace
        self.run_id = 0
        self.docs = {}  # doc_id -> {"hash", "config", "keys"}
        self.chunks = {}  # key -> Chunk (live or tombstoned)
        self.tombstones = {}  # key -> run in which it was tombstoned
        self.vectors = {}  # embed_key -> vector
        self.embed_calls = 0
        self._index = None

    # ------------------------------------------------------------------ ingest
    def ingest(self, source):
        self.run_id += 1
        rep = IngestReport(run=self.run_id)
        fp = self.cfg.fingerprint()
        for doc_id in sorted(source):
            text = source[doc_id]
            h = text_hash(text)
            rec = self.docs.get(doc_id)
            if rec and rec["hash"] == h and rec["config"] == fp:
                rep.docs_unchanged += 1
                continue
            new_chunks = chunk_document(doc_id, text, self.cfg)
            known = set(self.chunks)
            staged = {}
            try:
                for c in new_chunks:
                    if c.key in known or c.embed_key in self.vectors or c.embed_key in staged:
                        continue
                    self.embed_calls += 1
                    rep.embeds_computed += 1
                    staged[c.embed_key] = self.embed(c.embed_text)
            except Exception:
                rep.docs_failed.append(doc_id)
                continue
            self.vectors.update(staged)
            new_keys = [c.key for c in new_chunks]
            for c in new_chunks:
                if c.key in self.chunks:
                    if c.key in self.tombstones:
                        del self.tombstones[c.key]
                        rep.chunks_revived += 1
                    else:
                        rep.chunks_kept += 1
                else:
                    rep.chunks_added += 1
                    if c.embed_key not in staged:
                        rep.embeds_reused += 1
                self.chunks[c.key] = c
            for k in (rec["keys"] if rec else []):
                if k not in new_keys and k not in self.tombstones:
                    self.tombstones[k] = self.run_id
                    rep.chunks_tombstoned += 1
            self.docs[doc_id] = {"hash": h, "config": fp, "keys": new_keys}
            if rec:
                rep.docs_updated += 1
            else:
                rep.docs_added += 1
        for doc_id in sorted(set(self.docs) - set(source)):
            for k in self.docs[doc_id]["keys"]:
                if k not in self.tombstones:
                    self.tombstones[k] = self.run_id
                    rep.chunks_tombstoned += 1
            del self.docs[doc_id]
            rep.docs_deleted += 1
        self._index = None
        return rep

    # ------------------------------------------------------------------- views
    def live_chunks(self):
        keys = [k for k in self.chunks if k not in self.tombstones]
        return sorted((self.chunks[k] for k in keys), key=lambda c: (c.doc_id, c.ordinal))

    def index(self):
        if self._index is None:
            self._index = Index(self.live_chunks())
        return self._index

    def search(self, query, k):
        return self.index().search(query, k)

    def compact(self):
        """Drop tombstones older than the grace period and unreferenced embeddings.
        Returns (chunks_removed, vectors_removed)."""
        old = [k for k, run in self.tombstones.items() if self.run_id - run >= self.grace]
        for k in old:
            del self.tombstones[k]
            del self.chunks[k]
        referenced = {c.embed_key for c in self.chunks.values()}
        gone = [e for e in self.vectors if e not in referenced]
        for e in gone:
            del self.vectors[e]
        self._index = None
        return len(old), len(gone)

    # ------------------------------------------------------------------- stale
    def stale_report(self, source):
        fp = self.cfg.fingerprint()
        changed, drift, stale = [], [], []
        for doc_id in sorted(set(self.docs) & set(source)):
            rec = self.docs[doc_id]
            is_changed = rec["hash"] != text_hash(source[doc_id])
            is_drift = rec["config"] != fp
            if is_changed:
                changed.append(doc_id)
            if is_drift:
                drift.append(doc_id)
            if is_changed or is_drift:
                fresh = {c.key for c in chunk_document(doc_id, source[doc_id], self.cfg)}
                stale += [k for k in rec["keys"] if k not in fresh and k not in self.tombstones]
        orphans = sorted(set(self.docs) - set(source))
        for doc_id in orphans:
            stale += [k for k in self.docs[doc_id]["keys"] if k not in self.tombstones]
        return StaleReport(changed, sorted(set(source) - set(self.docs)), orphans, drift,
                           sorted(stale))

    # ------------------------------------------------------------- persistence
    def snapshot(self):
        """Canonical state, for equality checks (the run counter is excluded)."""
        return {
            "docs": {d: self.docs[d] for d in sorted(self.docs)},
            "chunks": sorted(self.chunks),
            "tombstones": dict(sorted(self.tombstones.items())),
            "vectors": sorted(self.vectors),
        }

    def to_json(self):
        return json.dumps({
            "cfg": self.cfg.__dict__,
            "grace": self.grace,
            "run_id": self.run_id,
            "docs": self.docs,
            "chunks": {k: c.__dict__ for k, c in self.chunks.items()},
            "tombstones": self.tombstones,
            "vectors": self.vectors,
        })

    @classmethod
    def from_json(cls, blob, embed=hash_embed):
        d = json.loads(blob)
        store = cls(ChunkerConfig(**d["cfg"]), embed=embed, grace=d["grace"])
        store.run_id = d["run_id"]
        store.docs = d["docs"]
        store.chunks = {k: Chunk(**c) for k, c in d["chunks"].items()}
        store.tombstones = d["tombstones"]
        store.vectors = d["vectors"]
        return store
