import pytest

from chunklab.chunkers import STRATEGIES, ChunkerConfig, chunk_document
from chunklab.corpus import apply_edits, generate_corpus
from chunklab.experiments import FlakyEmbed
from chunklab.ingest import IngestStore, hash_embed

CFG = ChunkerConfig("structure", size=64)


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_docs=24, seed=11)


def fresh_keys(docs, cfg):
    return sorted(c.key for d in docs for c in chunk_document(d, docs[d], cfg))


def test_first_ingest_adds_everything_and_embeds_each_unique_text_once(corpus):
    store = IngestStore(CFG)
    rep = store.ingest(corpus.docs)
    assert rep.docs_added == 24 and rep.docs_unchanged == 0
    assert len(store.live_chunks()) == rep.chunks_added
    unique = {c.embed_key for d in corpus.docs for c in chunk_document(d, corpus.docs[d], CFG)}
    assert rep.embeds_computed == len(unique) == store.embed_calls == len(store.vectors)


def test_rerunning_unchanged_source_is_a_no_op(corpus):
    store = IngestStore(CFG)
    store.ingest(corpus.docs)
    snap = store.snapshot()
    rep = store.ingest(corpus.docs)
    assert not rep.changed()
    assert rep.docs_unchanged == 24 and rep.embeds_computed == 0
    assert store.snapshot() == snap


def test_an_edit_embeds_only_the_new_chunks_and_tombstones_the_old(corpus):
    store = IngestStore(CFG)
    store.ingest(corpus.docs)
    new_docs, edits = apply_edits(corpus, seed=4, fraction=0.05)
    rep = store.ingest(new_docs)
    assert rep.docs_updated == len(edits) and rep.docs_unchanged == 24 - len(edits)
    assert rep.chunks_added == rep.embeds_computed + rep.embeds_reused
    assert 0 < rep.embeds_computed < 12
    assert rep.chunks_tombstoned >= 1 and rep.chunks_kept > 0
    assert sorted(c.key for c in store.live_chunks()) == fresh_keys(new_docs, CFG)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_incremental_state_equals_a_full_rebuild(corpus, strategy, seed):
    cfg = ChunkerConfig(strategy, size=48 if strategy == "parent_child" else 64)
    store = IngestStore(cfg)
    store.ingest(corpus.docs)
    new_docs, _ = apply_edits(corpus, seed=seed, fraction=0.2)
    store.ingest(new_docs)
    rebuilt = IngestStore(cfg)
    rebuilt.ingest(new_docs)
    assert store.live_chunks() == rebuilt.live_chunks()  # same records, same order


def test_embedding_cost_equals_the_set_difference_the_experiment_assumes(corpus):
    store = IngestStore(CFG)
    store.ingest(corpus.docs)
    new_docs, edits = apply_edits(corpus, seed=7, fraction=0.1)
    expected = 0
    for e in edits:
        old = {c.embed_key for c in chunk_document(e.doc_id, corpus.docs[e.doc_id], CFG)}
        new = {c.embed_key for c in chunk_document(e.doc_id, new_docs[e.doc_id], CFG)}
        expected += len(new - old)
    assert store.ingest(new_docs).embeds_computed == expected


def test_deleted_documents_are_tombstoned_and_hidden_from_search(corpus):
    store = IngestStore(CFG)
    store.ingest(corpus.docs)
    gone = sorted(corpus.docs)[:3]
    q = next(q for q in corpus.questions if q.doc_id == gone[0])
    assert any(c.doc_id == gone[0] for c in store.search(q.text, 5))
    rep = store.ingest({d: t for d, t in corpus.docs.items() if d not in gone})
    assert rep.docs_deleted == 3 and rep.chunks_tombstoned > 0
    assert all(c.doc_id not in gone for c in store.live_chunks())
    assert all(c.doc_id != gone[0] for c in store.search(q.text, 5))
    assert len(store.tombstones) == rep.chunks_tombstoned


def test_revived_chunks_need_no_new_embeddings(corpus):
    store = IngestStore(CFG)
    store.ingest(corpus.docs)
    gone = sorted(corpus.docs)[:3]
    store.ingest({d: t for d, t in corpus.docs.items() if d not in gone})
    rep = store.ingest(corpus.docs)
    assert rep.docs_added == 3 and rep.chunks_revived > 0 and rep.embeds_computed == 0
    assert store.tombstones == {}
    assert sorted(c.key for c in store.live_chunks()) == fresh_keys(corpus.docs, CFG)


def test_compact_keeps_tombstones_inside_the_grace_period(corpus):
    store = IngestStore(CFG, grace=2)
    store.ingest(corpus.docs)
    kept = {d: t for d, t in corpus.docs.items() if d != sorted(corpus.docs)[0]}
    store.ingest(kept)
    assert store.compact() == (0, 0) and store.tombstones
    store.ingest(kept)
    assert store.compact() == (0, 0)  # one run old, grace is 2
    store.ingest(kept)
    removed_chunks, removed_vectors = store.compact()
    assert removed_chunks > 0 and removed_vectors > 0
    assert store.tombstones == {}
    assert set(store.chunks) == {c.key for c in store.live_chunks()}
    assert len(store.vectors) == len({c.embed_key for c in store.live_chunks()})


def test_embedding_failure_leaves_the_old_version_and_the_retry_succeeds(corpus):
    base = IngestStore(CFG)
    base.ingest(corpus.docs)
    flaky = FlakyEmbed(fail_on=2, inner=hash_embed)
    store = IngestStore.from_json(base.to_json(), embed=flaky)
    new_docs, edits = apply_edits(corpus, seed=5, fraction=0.2)
    before = store.live_chunks()
    rep = store.ingest(new_docs)
    assert len(rep.docs_failed) == 1
    failed = rep.docs_failed[0]
    old_keys = {c.key for c in before if c.doc_id == failed}
    assert old_keys <= {c.key for c in store.live_chunks()}  # old version still serving
    assert store.docs[failed]["hash"] == base.docs[failed]["hash"]  # not marked done
    assert store.stale_report(new_docs).changed_docs == [failed]
    rep2 = store.ingest(new_docs)
    assert rep2.docs_failed == [] and rep2.docs_updated == 1
    assert sorted(c.key for c in store.live_chunks()) == fresh_keys(new_docs, CFG)


def test_a_failed_first_ingest_stores_nothing_for_that_document():
    docs = {"a": "# A guide\n\nalpha text one two three", "b": "# B guide\n\nbeta text four five"}
    store = IngestStore(CFG, embed=FlakyEmbed(1, hash_embed))
    rep = store.ingest(docs)
    assert rep.docs_failed == ["a"] and rep.docs_added == 1
    assert "a" not in store.docs and all(c.doc_id == "b" for c in store.live_chunks())


def test_identical_text_is_embedded_once_across_documents():
    text = "# T\n\nthe very same paragraph of words"
    store = IngestStore(ChunkerConfig("fixed", size=20, enrich="none"))
    rep = store.ingest({"a": text, "b": text, "c": text})
    assert rep.chunks_added == 3  # three records
    assert rep.embeds_computed == 1 and rep.embeds_reused == 2  # one embedding


def test_a_config_change_marks_every_document_stale_and_reingest_rechunks(corpus):
    store = IngestStore(CFG)
    store.ingest(corpus.docs)
    assert store.stale_report(corpus.docs).clean()
    store.cfg = ChunkerConfig("structure", size=32)
    sr = store.stale_report(corpus.docs)
    assert sr.config_drift_docs == sorted(corpus.docs) and sr.stale_chunk_keys
    rep = store.ingest(corpus.docs)
    assert rep.docs_updated == 24
    assert sorted(c.key for c in store.live_chunks()) == fresh_keys(corpus.docs, store.cfg)
    assert store.stale_report(corpus.docs).clean()


def test_stale_report_lists_changed_missing_and_orphan_documents(corpus):
    store = IngestStore(CFG)
    ids = sorted(corpus.docs)
    store.ingest({d: corpus.docs[d] for d in ids[:-1]})  # last doc not ingested yet
    new_docs, edits = apply_edits(corpus, seed=9, fraction=0.05)
    source = {d: t for d, t in new_docs.items() if d != ids[0]}  # first doc deleted at source
    sr = store.stale_report(source)
    assert sr.missing_docs == [ids[-1]] and sr.orphan_docs == [ids[0]]
    assert set(sr.changed_docs) <= {e.doc_id for e in edits}
    assert sr.stale_chunk_keys and not sr.clean()
    store.ingest(source)
    assert store.stale_report(source).clean()


def test_save_and_load_round_trips_and_resumes(corpus):
    store = IngestStore(CFG)
    store.ingest(corpus.docs)
    clone = IngestStore.from_json(store.to_json())
    assert clone.snapshot() == store.snapshot()
    assert clone.live_chunks() == store.live_chunks()
    assert not clone.ingest(corpus.docs).changed()
    new_docs, _ = apply_edits(corpus, seed=1)
    assert clone.ingest(new_docs).docs_updated == store.ingest(new_docs).docs_updated


def test_search_sees_the_new_value_only_after_the_edit_is_ingested(corpus):
    store = IngestStore(ChunkerConfig("structure", size=64))
    store.ingest(corpus.docs)
    new_docs, edits = apply_edits(corpus, seed=1, fraction=0.2)
    e = next(e for e in edits if e.kind == "value")
    span = e.question.must_contain[0]
    hit = lambda: any(span in c.context_text for c in store.search(e.question.text, 3))  # noqa: E731
    assert not hit()
    store.ingest(new_docs)
    assert hit()
