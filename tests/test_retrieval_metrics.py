import math

import pytest

from chunklab.bm25 import BM25
from chunklab.chunkers import Chunk, ChunkerConfig, chunk_document
from chunklab.corpus import Question
from chunklab.metrics import answers, evaluate, mcnemar_p, wilson
from chunklab.retrieval import Index


def mk(doc_id, text, context=None, prefix="", ordinal=0):
    return Chunk(key=f"{doc_id}:{ordinal}:{text[:8]}", embed_key="e", doc_id=doc_id,
                 ordinal=ordinal, text=text, context_text=context or text, prefix=prefix,
                 path=prefix)


def q(doc_id, text, must, kind="fact", core=()):
    return Question(f"{doc_id}:{kind}", doc_id, kind, text, tuple(must), "dev", "md", core)


# -------------------------------------------------------------------- bm25
def test_bm25_ranks_the_matching_document_first():
    bm = BM25([["red", "apple"], ["green", "pear"], ["red", "pear"]])
    assert bm.top(["apple"], 3)[0][0] == 0
    assert [i for i, _ in bm.top(["red", "pear"], 3)][0] == 2


def test_bm25_rarer_terms_weigh_more():
    bm = BM25([["common", "rare"], ["common"], ["common"], ["common"]])
    assert bm.idf("rare") > bm.idf("common")


def test_bm25_length_normalisation_prefers_the_shorter_match():
    bm = BM25([["x"] + ["pad"] * 30, ["x"]])
    assert bm.top(["x"], 2)[0][0] == 1


def test_bm25_is_deterministic_on_ties_and_handles_empty_queries():
    bm = BM25([["a"], ["a"], ["a"]])
    assert [i for i, _ in bm.top(["a"], 3)] == [0, 1, 2]
    assert bm.top([], 3) == []
    assert bm.top(["unknown"], 3) == []
    assert BM25([]).top(["a"], 3) == []


# --------------------------------------------------------------- retrieval
def test_prefix_is_indexed_so_the_title_disambiguates_identical_text():
    a = mk("alpha", "the default timeout is 30 seconds", prefix="Alpha guide")
    b = mk("beta", "the default timeout is 30 seconds", prefix="Beta guide")
    idx = Index([a, b])
    assert idx.search("default timeout for Beta", 1)[0].doc_id == "beta"
    assert idx.search("default timeout for Alpha", 1)[0].doc_id == "alpha"


def test_without_a_prefix_identical_chunks_cannot_be_told_apart():
    a, b = mk("alpha", "same text here"), mk("beta", "same text here")
    assert Index([a, b]).search("same text for beta", 1)[0].doc_id == "alpha"  # tie, by position


def test_search_dedupes_children_of_one_parent():
    parent = "whole section text about retries and timeouts"
    kids = [mk("d", "retries backoff", parent, ordinal=0), mk("d", "retries timeouts", parent, ordinal=1),
            mk("d", "unrelated words", "other parent", ordinal=2)]
    res = Index(kids).search("retries", 5)
    assert [r.context_text for r in res] == [parent]


def test_search_respects_k_and_returns_nothing_for_no_match():
    idx = Index([mk("d", f"word{i} shared", ordinal=i) for i in range(10)])
    assert len(idx.search("shared", 3)) == 3
    assert idx.search("zzz", 3) == []


# ----------------------------------------------------------------- metrics
def test_answers_requires_every_span_in_one_context():
    c = mk("d", "Open the page.\nPress the button.")
    assert answers(q("d", "x", ["Open the page.", "Press the button."]), c)
    assert not answers(q("d", "x", ["Open the page.", "Missing step."]), c)


def test_answers_needs_the_gold_document():
    c = mk("other", "the default timeout is 30 seconds")
    assert not answers(q("d", "x", ["default timeout is 30 seconds"]), c)


def test_answers_ignores_whitespace_differences():
    assert answers(q("d", "x", ["a b   c"]), mk("d", "a\nb c"))


def test_answers_core_is_the_lenient_check_for_tables():
    row_only = mk("d", "| Team | 20 |")
    question = q("d", "x", ["| Team | 20 |", "Seats"], kind="table", core=("| Team | 20 |",))
    assert not answers(question, row_only)
    assert answers(question, row_only, core=True)


def test_evaluate_cover_dochit_and_context_size():
    good = mk("d", "alpha answer span here", ordinal=0)
    near = mk("d", "alpha but no span", ordinal=1)
    other = mk("e", "alpha elsewhere", ordinal=2)
    idx = Index([near, good, other])
    r = evaluate(idx, [q("d", "alpha answer span", ["answer span here"])], ks=(1, 3))
    assert r["cover"][1] == 1.0 and r["cover"][3] == 1.0
    assert r["dochit"][1] == 1.0
    assert r["mrr"] == 1.0
    assert r["ctx"][3] == sum(len(c.context_text.split()) for c in idx.search("alpha answer span", 3))


def test_evaluate_counts_a_miss_when_the_span_was_cut_off():
    cut = [mk("d", "step one", ordinal=0), mk("d", "step two", ordinal=1)]
    r = evaluate(Index(cut), [q("d", "step", ["step one", "step two"])], ks=(1, 3))
    assert r["cover"][3] == 0.0
    assert r["dochit"][3] == 1.0  # located the document, still could not answer


def test_evaluate_reports_by_kind_and_format():
    idx = Index([mk("d", "hello world one", ordinal=0)])
    qs = [q("d", "hello", ["world one"], kind="fact"), q("d", "hello", ["nope"], kind="code")]
    r = evaluate(idx, qs, ks=(1,))
    assert r["by_kind"]["fact"][1] == 1.0 and r["by_kind"]["code"][1] == 0.0
    assert r["by_fmt"]["md"]["n"] == 2
    assert r["cover"][1] == 0.5


def test_wilson_interval_behaves():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and 0.40 < lo and hi < 0.60
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(100, 100)[1] == pytest.approx(1.0) and wilson(100, 100)[0] > 0.95


def test_mcnemar_exact_values():
    assert mcnemar_p([1, 1, 1, 0], [0, 0, 0, 0]) == (3, 0, 0.25)
    assert mcnemar_p([1, 0], [0, 1])[2] == 1.0
    assert mcnemar_p([1, 1], [1, 1]) == (0, 0, 1.0)
    only_a, only_b, p = mcnemar_p([1] * 10 + [0] * 5, [0] * 10 + [1] * 5)
    assert (only_a, only_b) == (10, 5) and math.isclose(p, 0.3018, abs_tol=1e-3)


@pytest.mark.parametrize("strategy", ["fixed", "structure"])
def test_end_to_end_small_corpus(strategy):
    docs = {"d1": "# One guide\n\n## Limits\n\nThe default timeout is 30 seconds.\n",
            "d2": "# Two guide\n\n## Limits\n\nThe default timeout is 45 seconds.\n"}
    chunks = [c for d in sorted(docs) for c in chunk_document(d, docs[d], ChunkerConfig(strategy, size=20))]
    r = evaluate(Index(chunks), [q("d2", "default timeout of Two", ["The default timeout is 45 seconds."])], ks=(1,))
    assert r["cover"][1] == 1.0
