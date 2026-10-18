import pytest

from chunklab import __main__ as cli
from chunklab.chunkers import ChunkerConfig
from chunklab.corpus import generate_corpus
from chunklab.experiments import (CTX_BUDGET, STRATEGY_ORDER, TUNE_SIZES, _Cache, format_ingest,
                                  format_lifecycle, format_quality, run_ingest, run_lifecycle,
                                  run_quality, tune_size)


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_docs=40, seed=17)


@pytest.fixture(scope="module")
def quality(corpus):
    return run_quality(corpus)


def test_tuning_uses_dev_only_and_respects_the_context_budget(corpus):
    cache = _Cache(corpus.docs)
    dev = corpus.split("dev")
    size, free, rows = tune_size(cache, "fixed", dev)
    assert size in TUNE_SIZES["fixed"] and free in TUNE_SIZES["fixed"]
    chosen = next(r for r in rows if r["size"] == size)
    assert chosen["ctx3"] <= CTX_BUDGET
    free_row = next(r for r in rows if r["size"] == free)
    assert free_row["cover3"] >= chosen["cover3"]
    # a budget nothing fits falls back to the smallest context
    size0, _, rows0 = tune_size(cache, "fixed", dev, budget=1)
    assert size0 == min(rows0, key=lambda r: r["ctx3"])["size"]


def test_quality_tables_have_every_strategy_and_sane_ranges(quality):
    for table in ("heldout_tuned", "heldout_unconstrained", "heldout_matched"):
        assert set(quality[table]) == set(STRATEGY_ORDER)
        for r in quality[table].values():
            assert 0 <= r["cover"][1] <= r["cover"][3] <= r["cover"][5] <= 1
            assert r["dochit"][3] >= r["cover"][3]
            assert r["cover_core"][3] >= r["cover"][3]  # lenient is never stricter
    assert quality["whole_doc"]["cover"][3] >= 0.99  # the ceiling: the whole document


def test_dev_and_heldout_are_disjoint_by_document(corpus):
    assert not ({q.doc_id for q in corpus.split("dev")} & {q.doc_id for q in corpus.split("heldout")})


def test_whole_document_costs_far_more_context_than_any_chunker(quality):
    whole = quality["whole_doc"]["ctx"][3]
    assert all(r["ctx"][3] < whole for r in quality["heldout_matched"].values())


def test_quality_report_prints_the_tables(quality):
    text = format_quality(quality)
    for needle in ("HELD-OUT", "answer-only@3", "Enrichment ablation", "McNemar", "flat"):
        assert needle in text


def test_ingest_cost_ordering_and_equivalence_to_a_rebuild(corpus):
    res = run_ingest(corpus, trials=2, fraction=0.1)
    for name, a in res["strategies"].items():
        assert a["identical"], name
        for full, doc, ch in zip(a["full"], a["doc_level"], a["chunk_hash"]):
            assert ch <= doc <= full, name
        assert a["stale_hits"] == 0 and a["fresh_hits"] <= a["fresh_n"]
    assert "chunk-hash" in format_ingest(res)


def test_structure_chunking_re_embeds_no_more_than_fixed_on_an_insert(corpus):
    res = run_ingest(corpus, trials=3, fraction=0.1)["strategies"]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert mean(res["structure"]["kinds"]["delete"]) <= mean(res["fixed"]["kinds"]["delete"])
    assert mean(res["fixed"]["kinds"]["value"]) <= 2.5  # an in-place edit touches ~1-2 chunks


def test_lifecycle_invariants(corpus):
    lc = run_lifecycle(corpus)
    assert lc["idempotent"] == {"changed": False, "embeds": 0, "snapshot_equal": True}
    assert lc["stale_before"]["changed_docs"] > 0 and lc["after_edit"]["stale_after"] == 0
    assert lc["deletion"]["leaked_results"] == 0
    assert lc["deletion"]["compact_before_grace"][0] <= lc["deletion"]["compact_after_grace"][0]
    assert lc["revive"]["chunks_revived"] == lc["deletion"]["chunks_tombstoned"]
    assert lc["revive"]["embeds"] == 0
    assert lc["failure"]["docs_failed"] == 1 and lc["failure"]["failed_doc_kept_old_chunks"]
    assert lc["failure"]["retry_failed"] == 0
    assert "idempotent" not in format_lifecycle(lc) and "state identical=True" in format_lifecycle(lc)


def test_cli_demo_small_and_show(capsys):
    assert cli.main(["demo", "--small"]) == 0
    out = capsys.readouterr().out
    assert "=== 1." in out and "=== 2." in out and "=== 3." in out
    assert cli.main(["show", "--docs", "40", "--strategy", "fixed", "--size", "64"]) == 0
    assert "chunks (fixed" in capsys.readouterr().out


def test_default_matched_configs_are_valid():
    for s in STRATEGY_ORDER:
        ChunkerConfig(s, size=TUNE_SIZES[s][0]).validate()
