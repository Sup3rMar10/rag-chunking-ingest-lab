import re
from collections import Counter

import pytest

from chunklab.corpus import (KINDS, Rng, apply_edits, generate_corpus, render, split_of)
from chunklab.text import norm


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(n_docs=60, seed=3)


def test_generation_is_deterministic_and_seed_sensitive():
    a, b = generate_corpus(n_docs=15, seed=5), generate_corpus(n_docs=15, seed=5)
    assert a.docs == b.docs and a.questions == b.questions
    assert generate_corpus(n_docs=15, seed=6).docs != a.docs


def test_rng_is_stable_and_in_range():
    r1, r2 = Rng(9), Rng(9)
    assert [r1.int(3, 7) for _ in range(20)] == [r2.int(3, 7) for _ in range(20)]
    r = Rng(1)
    assert all(3 <= r.int(3, 7) <= 7 for _ in range(200))
    assert len(set(Rng(2).sample(range(10), 10))) == 10


def test_product_names_are_unique(corpus):
    names = [s.name for s in corpus.specs.values()]
    assert len(set(names)) == len(names)


def test_every_question_span_is_in_its_gold_document(corpus):
    for q in corpus.questions:
        doc = norm(corpus.docs[q.doc_id])
        assert all(norm(m) in doc for m in q.must_contain), q.qid
        assert all(norm(m) in doc for m in q.core_spans()), q.qid


def test_questions_name_their_product_and_every_kind_appears(corpus):
    assert {q.kind for q in corpus.questions} == set(KINDS)
    for q in corpus.questions:
        assert corpus.specs[q.doc_id].name in q.text


def test_split_is_by_document_and_both_halves_are_populated(corpus):
    dev = {q.doc_id for q in corpus.split("dev")}
    held = {q.doc_id for q in corpus.split("heldout")}
    assert dev and held and not (dev & held)
    assert all(split_of(d) == "dev" for d in dev)


def test_flat_documents_have_no_markdown_structure(corpus):
    flat = [s for s in corpus.specs.values() if s.flat]
    md = [s for s in corpus.specs.values() if not s.flat]
    assert flat and md
    for s in flat:
        text = corpus.docs[s.doc_id]
        assert "|" not in text and "```" not in text and not re.search(r"^#", text, re.M)
    for s in md:
        assert "## Limits" in corpus.docs[s.doc_id] and "|---" in corpus.docs[s.doc_id]
    assert {q.fmt for q in corpus.questions} == {"md", "flat"}


def test_table_questions_need_the_header_but_core_is_only_the_row(corpus):
    tq = [q for q in corpus.questions if q.kind in ("table", "bigtable")]
    assert tq
    for q in tq:
        assert len(q.must_contain) == 2 and len(q.core) == 1
        assert q.core[0] == q.must_contain[0]


def test_corpus_has_no_card_numbers_ssns_or_real_looking_personal_data(corpus):
    text = "\n".join(corpus.docs.values()) + "\n".join(q.text for q in corpus.questions)
    assert not re.search(r"\b\d{9,}\b", text)  # no long digit runs (cards, ids, phone numbers)
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", text)  # SSN shape
    assert not re.search(r"@[a-z]+\.(com|org|net)", text)  # no email addresses


def test_edits_touch_about_five_percent_and_do_not_mutate_the_corpus(corpus):
    before = dict(corpus.docs)
    new_docs, edits = apply_edits(corpus, seed=1, fraction=0.05)
    assert corpus.docs == before
    changed = [d for d in corpus.docs if new_docs[d] != corpus.docs[d]]
    assert sorted(changed) == sorted(e.doc_id for e in edits)
    assert len(edits) == 3  # round(0.05 * 60)
    assert Counter(e.kind for e in edits) == {"value": 1, "insert": 1, "delete": 1}


def test_edit_kinds_do_what_they_say(corpus):
    new_docs, edits = apply_edits(corpus, seed=2, fraction=0.15)
    for e in edits:
        old, new = corpus.docs[e.doc_id], new_docs[e.doc_id]
        if e.kind == "value":
            assert len(old.split()) == len(new.split())  # in place: same word count
            span = e.question.must_contain[0]
            assert span in new and span not in old
        elif e.kind == "insert":
            assert len(new.split()) > len(old.split())
        else:
            assert "2.4.0:" in old and "2.4.0:" not in new  # the changelog entries are gone


def test_render_is_a_pure_function_of_the_spec(corpus):
    s = next(iter(corpus.specs.values()))
    assert render(s) == render(s) == corpus.docs[s.doc_id]
