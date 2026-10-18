"""The measurements behind the README. Every function returns plain data; `format_*`
turns it into the tables the demo prints.

Protocol (so the held-out numbers mean something):
  * questions are split by DOCUMENT (a document is entirely dev or entirely held-out);
  * the chunk size of each strategy is chosen on dev only, under a context budget (a real
    system pays for every word it sends to the model), then scored once on held-out;
    without a budget, bigger chunks win trivially, so that choice is reported separately;
  * a second table fixes every strategy at the same size, so "structure won" cannot be
    explained by a different chunk size;
  * enrichment (title / heading path prefix) is an orthogonal switch with its own ablation.
"""

from .chunkers import ChunkerConfig, chunk_document
from .corpus import apply_edits
from .ingest import IngestStore
from .metrics import KS, evaluate, mcnemar_p
from .retrieval import Index

STRATEGY_ORDER = ("fixed", "sentence", "recursive", "structure", "parent_child")
TUNE_SIZES = {
    "fixed": (64, 128, 256, 512),
    "sentence": (64, 128, 256, 512),
    "recursive": (64, 128, 256, 512),
    "structure": (64, 128, 256, 512),
    "parent_child": (32, 48, 64, 96),  # child size; the parent is a 400-word section
}
MATCHED = {"fixed": 128, "sentence": 128, "recursive": 128, "structure": 128,
           "parent_child": 48}
CTX_BUDGET = 400  # words of context at k = 3
WHOLE_DOC = ChunkerConfig("fixed", size=10 ** 6, overlap=0, enrich="title")


def build_chunks(docs, cfg):
    return [c for d in sorted(docs) for c in chunk_document(d, docs[d], cfg)]


class _Cache:
    """Index per config, so a table does not re-chunk the corpus for every row."""

    def __init__(self, docs):
        self.docs, self._idx, self._n = docs, {}, {}

    def index(self, cfg):
        if cfg not in self._idx:
            chunks = build_chunks(self.docs, cfg)
            self._idx[cfg] = Index(chunks)
            self._n[cfg] = len(chunks)
        return self._idx[cfg]

    def n_chunks(self, cfg):
        self.index(cfg)
        return self._n[cfg]


def _mean_chunk_words(index):
    from .text import count_tokens
    sizes = [count_tokens(c.text) for c in index.chunks]
    return sum(sizes) / len(sizes) if sizes else 0.0


def tune_size(cache, strategy, dev, enrich="title", budget=CTX_BUDGET):
    """Best dev cover@3 among sizes whose dev context at k=3 fits the budget (if none fit,
    the smallest context). Ties go to the smaller context, then the smaller size.
    Returns (budgeted size, unconstrained size, rows)."""
    rows = []
    for size in TUNE_SIZES[strategy]:
        cfg = ChunkerConfig(strategy, size=size, enrich=enrich)
        r = evaluate(cache.index(cfg), dev)
        rows.append({"size": size, "cover3": r["cover"][3], "ctx3": r["ctx"][3],
                     "chunks": cache.n_chunks(cfg)})
    fits = [r for r in rows if r["ctx3"] <= budget] or [min(rows, key=lambda r: r["ctx3"])]
    key = lambda r: (-r["cover3"], r["ctx3"], r["size"])  # noqa: E731
    return min(fits, key=key)["size"], min(rows, key=key)["size"], rows


def _row(cache, cfg, questions):
    r = evaluate(cache.index(cfg), questions)
    r["chunks"] = cache.n_chunks(cfg)
    r["mean_words"] = _mean_chunk_words(cache.index(cfg))
    return r


def run_quality(corpus):
    cache = _Cache(corpus.docs)
    dev, held = corpus.split("dev"), corpus.split("heldout")
    out = {"n_dev": len(dev), "n_heldout": len(held), "tuned": {}, "unconstrained": {},
           "tune_rows": {}, "ctx_budget": CTX_BUDGET}
    for s in STRATEGY_ORDER:
        size, size_free, rows = tune_size(cache, s, dev)
        out["tuned"][s], out["unconstrained"][s], out["tune_rows"][s] = size, size_free, rows
    out["heldout_tuned"] = {s: _row(cache, ChunkerConfig(s, size=out["tuned"][s]), held)
                            for s in STRATEGY_ORDER}
    out["heldout_unconstrained"] = {
        s: _row(cache, ChunkerConfig(s, size=out["unconstrained"][s]), held)
        for s in STRATEGY_ORDER}
    out["heldout_matched"] = {s: _row(cache, ChunkerConfig(s, size=MATCHED[s]), held)
                              for s in STRATEGY_ORDER}
    out["whole_doc"] = _row(cache, WHOLE_DOC, held)
    out["ablation"] = {
        s: {e: evaluate(cache.index(ChunkerConfig(s, size=out["tuned"][s], enrich=e)), held)
            for e in ("none", "title", "path")}
        for s in STRATEGY_ORDER
    }
    tuned = out["heldout_tuned"]
    out["paired"] = []
    for a, b in (("structure", "recursive"), ("structure", "fixed"), ("recursive", "fixed"),
                 ("parent_child", "structure"), ("parent_child", "recursive")):
        only_a, only_b, p = mcnemar_p(tuned[a]["outcomes"][3], tuned[b]["outcomes"][3])
        out["paired"].append({"a": a, "b": b, "only_a": only_a, "only_b": only_b, "p": p})
    return out


# ------------------------------------------------------------------- ingestion
INGEST_CFGS = {
    "fixed": ChunkerConfig("fixed", size=128),
    "sentence": ChunkerConfig("sentence", size=128),
    "recursive": ChunkerConfig("recursive", size=128),
    "structure": ChunkerConfig("structure", size=128),
    "parent_child": ChunkerConfig("parent_child", size=48),
}


def run_ingest(corpus, trials=10, fraction=0.05):
    out = {"trials": trials, "fraction": fraction, "strategies": {}}
    for name, cfg in INGEST_CFGS.items():
        base = IngestStore(cfg)
        base.ingest(corpus.docs)
        blob = base.to_json()
        base_index = base.index()
        agg = {"full": [], "doc_level": [], "chunk_hash": [], "identical": True,
               "kinds": {"value": [], "insert": [], "delete": []}, "per_doc_chunks": [],
               "stale_hits": 0, "fresh_hits": 0, "fresh_n": 0}
        for t in range(trials):
            new_docs, edits = apply_edits(corpus, seed=100 + t, fraction=fraction)
            store = IngestStore.from_json(blob)
            rep = store.ingest(new_docs)
            fresh = build_chunks(new_docs, cfg)
            agg["identical"] &= (sorted(c.key for c in store.live_chunks())
                                 == sorted(c.key for c in fresh))
            agg["full"].append(len(fresh))
            agg["chunk_hash"].append(rep.embeds_computed)
            agg["doc_level"].append(
                sum(len(chunk_document(e.doc_id, new_docs[e.doc_id], cfg)) for e in edits))
            for e in edits:
                old = {c.embed_key for c in chunk_document(e.doc_id, corpus.docs[e.doc_id], cfg)}
                new = chunk_document(e.doc_id, new_docs[e.doc_id], cfg)
                agg["kinds"][e.kind].append(len({c.embed_key for c in new} - old))
                agg["per_doc_chunks"].append(len(new))
            qs = [e.question for e in edits if e.question]
            if qs:
                stale = evaluate(base_index, qs)
                fresh_r = evaluate(store.index(), qs)
                agg["stale_hits"] += round(stale["cover"][3] * len(qs))
                agg["fresh_hits"] += round(fresh_r["cover"][3] * len(qs))
                agg["fresh_n"] += len(qs)
        out["strategies"][name] = agg
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------------------- lifecycle
class FlakyEmbed:
    """An embed function that raises on its Nth call (then keeps working)."""

    def __init__(self, fail_on, inner):
        self.fail_on, self.calls, self.inner = fail_on, 0, inner

    def __call__(self, text):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("embedding service unavailable")
        return self.inner(text)


def run_lifecycle(corpus):
    from .ingest import hash_embed
    cfg = ChunkerConfig("structure", size=128)
    out = {}
    store = IngestStore(cfg, grace=2)
    first = store.ingest(corpus.docs)
    base_blob = store.to_json()
    snap = store.snapshot()
    second = store.ingest(corpus.docs)
    out["first_run"] = {"docs": first.docs_added, "chunks": first.chunks_added,
                        "embeds": first.embeds_computed}
    out["idempotent"] = {"changed": second.changed(), "embeds": second.embeds_computed,
                         "snapshot_equal": snap == store.snapshot()}

    new_docs, edits = apply_edits(corpus, seed=100)
    sr = store.stale_report(new_docs)
    out["stale_before"] = {"changed_docs": len(sr.changed_docs),
                           "stale_chunks": len(sr.stale_chunk_keys)}
    rep = store.ingest(new_docs)
    out["after_edit"] = {"docs_updated": rep.docs_updated, "chunks_added": rep.chunks_added,
                         "chunks_kept": rep.chunks_kept,
                         "chunks_tombstoned": rep.chunks_tombstoned,
                         "embeds": rep.embeds_computed,
                         "stale_after": len(store.stale_report(new_docs).stale_chunk_keys)}

    # deletion: tombstone, hidden from search, grace, compact, revive
    ids = sorted(new_docs)
    gone = ids[:: max(1, len(ids) // 10)][:10]
    kept = {d: t for d, t in new_docs.items() if d not in gone}
    live_before = len(store.live_chunks())
    snapshot_blob = store.to_json()
    d1 = store.ingest(kept)
    leaked = sum(1 for q in corpus.questions if q.doc_id in gone
                 for c in store.search(q.text, 5) if c.doc_id in gone)
    early = store.compact()
    store.ingest(kept)
    store.ingest(kept)
    late = store.compact()
    out["deletion"] = {"docs_deleted": d1.docs_deleted, "chunks_tombstoned": d1.chunks_tombstoned,
                       "live_before": live_before, "live_after": len(store.live_chunks()),
                       "leaked_results": leaked, "compact_before_grace": early,
                       "compact_after_grace": late}
    revive = IngestStore.from_json(snapshot_blob)
    revive.ingest(kept)
    r = revive.ingest(new_docs)
    out["revive"] = {"chunks_revived": r.chunks_revived, "embeds": r.embeds_computed}

    # failure: embedding dies half way through an update batch
    flaky = FlakyEmbed(fail_on=3, inner=hash_embed)
    s2 = IngestStore.from_json(base_blob, embed=flaky)
    probe = edits[0]
    before = s2.live_chunks()
    rep = s2.ingest(new_docs)
    out["failure"] = {"docs_failed": len(rep.docs_failed),
                      "docs_updated": rep.docs_updated,
                      "failed_doc_kept_old_chunks": all(
                          any(c.key == k for c in before) for k in
                          s2.docs[rep.docs_failed[0]]["keys"]) if rep.docs_failed else None}
    rep2 = s2.ingest(new_docs)
    out["failure"]["retry_updated"] = rep2.docs_updated
    out["failure"]["retry_failed"] = len(rep2.docs_failed)
    out["failure"]["probe_doc"] = probe.doc_id
    return out


# ------------------------------------------------------------------- formatting
def _pct(x):
    return f"{100 * x:5.1f}%"


def _ci(r, k=3):
    lo, hi = r["cover_ci"][k]
    return f"{100 * lo:.0f}-{100 * hi:.0f}"


def format_quality(q):
    L = []
    L.append(f"Questions: {q['n_dev']} dev / {q['n_heldout']} held-out (split by document)")
    L.append("")
    L.append(f"Chunk size sweep on dev: size: cover@3 / context words@3 "
             f"(* = chosen, context budget {q['ctx_budget']} words):")
    for s in STRATEGY_ORDER:
        L.append("  " + f"{s:<13}" + "  ".join(
            f"{r['size']}:{100 * r['cover3']:.1f}%/{r['ctx3']:.0f}w"
            + ("*" if r["size"] == q["tuned"][s] else "")
            for r in q["tune_rows"][s]))
    hdr = (f"{'strategy':<14}{'size':>5}{'chunks':>8}{'words':>7}{'cover@1':>9}{'cover@3':>9}"
           f"{'95% CI':>9}{'cover@5':>9}{'answer-only@3':>15}{'ctx words@3':>13}")

    def table(title, rows, sizes):
        L.append("")
        L.append(title)
        L.append(hdr)
        for name, r, size in rows:
            L.append(f"{name:<14}{size:>5}{r['chunks']:>8}{r['mean_words']:>7.0f}"
                     f"{_pct(r['cover'][1]):>9}{_pct(r['cover'][3]):>9}{_ci(r):>9}"
                     f"{_pct(r['cover'][5]):>9}{_pct(r['cover_core'][3]):>15}"
                     f"{r['ctx'][3]:>13.0f}")

    table("HELD-OUT, size chosen on dev within the context budget:",
          [(s, q["heldout_tuned"][s], q["tuned"][s]) for s in STRATEGY_ORDER], None)
    table("HELD-OUT, size chosen on dev with NO budget (big chunks buy coverage with context):",
          [(s, q["heldout_unconstrained"][s], q["unconstrained"][s]) for s in STRATEGY_ORDER],
          None)
    table("HELD-OUT, every strategy at the same size (128 words; parent_child child = 48):",
          [(s, q["heldout_matched"][s], MATCHED[s]) for s in STRATEGY_ORDER], None)
    table("Reference (not a real option): one chunk per whole document:",
          [("whole-doc", q["whole_doc"], "all")], None)

    L.append("")
    L.append("Held-out cover@3 by question kind (size chosen on dev):")
    kinds = sorted(next(iter(q["heldout_tuned"].values()))["by_kind"])
    L.append(f"{'strategy':<14}" + "".join(f"{k:>10}" for k in kinds))
    ns = next(iter(q["heldout_tuned"].values()))["by_kind"]
    L.append(f"{'(n)':<14}" + "".join(f"{ns[k]['n']:>10}" for k in kinds))
    for s in STRATEGY_ORDER:
        bk = q["heldout_tuned"][s]["by_kind"]
        L.append(f"{s:<14}" + "".join(f"{100 * bk[k][3]:>9.1f}%" for k in kinds))

    L.append("")
    L.append("Held-out cover@3 by document format (md = markdown, flat = structure stripped):")
    fmts = sorted(next(iter(q["heldout_tuned"].values()))["by_fmt"])
    nf = next(iter(q["heldout_tuned"].values()))["by_fmt"]
    L.append(f"{'strategy':<14}" + "".join(f"{f + ' (n=' + str(nf[f]['n']) + ')':>14}" for f in fmts))
    for s in STRATEGY_ORDER:
        bf = q["heldout_tuned"][s]["by_fmt"]
        L.append(f"{s:<14}" + "".join(f"{100 * bf[f][3]:>13.1f}%" for f in fmts))

    L.append("")
    L.append("Enrichment ablation, held-out cover@3 (prefix indexed with the chunk text):")
    L.append(f"{'strategy':<14}{'none':>9}{'title':>9}{'path':>9}")
    for s in STRATEGY_ORDER:
        a = q["ablation"][s]
        L.append(f"{s:<14}" + "".join(f"{_pct(a[e]['cover'][3]):>9}" for e in ("none", "title", "path")))

    L.append("")
    L.append("Paired comparison on held-out cover@3 (exact McNemar; only-A / only-B = questions "
             "one answered and the other did not):")
    for p in q["paired"]:
        L.append(f"  {p['a']:<13} vs {p['b']:<13} only-A {p['only_a']:>3}  only-B {p['only_b']:>3}"
                 f"  p = {p['p']:.3f}")
    return "\n".join(L)


def format_ingest(res):
    L = [f"Edit {res['fraction']:.0%} of the documents ({res['trials']} trials, each edits a "
         "different random set, cycling value / insert / delete)."]
    L.append(f"{'strategy':<14}{'chunks/doc':>11}{'full rebuild':>14}{'doc-level':>11}"
             f"{'chunk-hash':>12}{'saved vs full':>15}{'value':>7}{'insert':>8}{'delete':>8}"
             f"{'== rebuild':>12}")
    for name, a in res["strategies"].items():
        full, doc, ch = _mean(a["full"]), _mean(a["doc_level"]), _mean(a["chunk_hash"])
        k = a["kinds"]
        L.append(f"{name:<14}{_mean(a['per_doc_chunks']):>11.1f}{full:>14.0f}{doc:>11.1f}{ch:>12.1f}"
                 f"{100 * (1 - ch / full):>14.1f}%{_mean(k['value']):>7.1f}{_mean(k['insert']):>8.1f}"
                 f"{_mean(k['delete']):>8.1f}{str(a['identical']):>12}")
    L.append("(full rebuild / doc-level / chunk-hash = chunks embedded per run; value / insert / "
             "delete = chunks embedded per edited document of that kind)")
    L.append("")
    L.append("Edited-fact questions (the new value must be retrievable), cover@3:")
    for name, a in res["strategies"].items():
        n = a["fresh_n"]
        L.append(f"  {name:<13} before re-ingest {a['stale_hits']:>3}/{n}   after {a['fresh_hits']:>3}/{n}")
    return "\n".join(L)


def format_lifecycle(lc):
    f, i, a, d, r, fl = (lc["first_run"], lc["idempotent"], lc["after_edit"], lc["deletion"],
                         lc["revive"], lc["failure"])
    return "\n".join([
        f"first ingest: {f['docs']} docs, {f['chunks']} chunks, {f['embeds']} embeddings",
        f"re-run unchanged: changed={i['changed']}, embeddings={i['embeds']}, "
        f"state identical={i['snapshot_equal']}",
        f"before re-ingest of the edited docs, stale report: {lc['stale_before']['changed_docs']} "
        f"changed docs, {lc['stale_before']['stale_chunks']} stale chunks",
        f"after editing {a['docs_updated']} docs: +{a['chunks_added']} chunks, "
        f"{a['chunks_kept']} kept, -{a['chunks_tombstoned']} tombstoned, {a['embeds']} embeddings; "
        f"stale chunks left: {a['stale_after']}",
        f"delete {d['docs_deleted']} docs: {d['chunks_tombstoned']} chunks tombstoned, live "
        f"{d['live_before']} -> {d['live_after']}, results leaked from deleted docs: "
        f"{d['leaked_results']}",
        f"compact right away (inside grace): removed {d['compact_before_grace']} "
        f"(chunks, embeddings); after grace: {d['compact_after_grace']}",
        f"re-add the deleted docs inside the grace period: {r['chunks_revived']} chunks revived "
        f"with {r['embeds']} embeddings",
        f"embedding outage on call 3: {fl['docs_failed']} doc failed, "
        f"{fl['docs_updated']} others updated, failed doc kept its old chunks="
        f"{fl['failed_doc_kept_old_chunks']}; retry updated {fl['retry_updated']} "
        f"(failed again: {fl['retry_failed']})",
    ])


def run_all(corpus, trials=10):
    return {"quality": run_quality(corpus), "ingest": run_ingest(corpus, trials=trials),
            "lifecycle": run_lifecycle(corpus)}
