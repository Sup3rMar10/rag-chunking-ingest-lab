import pytest

from chunklab.chunkers import (STRATEGIES, ChunkerConfig, _hard_split, _pack_units,
                               _recursive_units, chunk_document)
from chunklab.text import count_tokens, norm

PROSE = " ".join(f"Sentence number {i} says something plain." for i in range(60))

TABLE = "\n".join(
    ["| Region | Endpoint | Budget |", "|---|---|---|"]
    + [f"| r-{i} | https://r-{i}.example.test | {i} |" for i in range(30)]
)

DOC = f"""# Gadget guide

Gadget is a fictional thing.

## Limits

The default timeout is 30 seconds.

| Plan | Seats |
|---|---|
| Free | 3 |
| Team | 20 |

## Setup

1. Open the page.
2. Press the button.
3. Wait.

```yaml
sync:
  batch_size: 100
  flush_ms: 500
```

## Regions

{TABLE}

## Essay

{PROSE}
"""


def cfg(strategy, size=40, **kw):
    return ChunkerConfig(strategy, size=size, **kw)


def texts(strategy, size=40, **kw):
    return [c.text for c in chunk_document("g", DOC, cfg(strategy, size, **kw))]


# ------------------------------------------------------------------ config
def test_config_validation():
    with pytest.raises(ValueError):
        ChunkerConfig("nope").validate()
    with pytest.raises(ValueError):
        ChunkerConfig("fixed", size=2).validate()
    with pytest.raises(ValueError):
        ChunkerConfig("fixed", size=32, overlap=32).validate()
    with pytest.raises(ValueError):
        ChunkerConfig("structure", enrich="bogus").validate()
    with pytest.raises(ValueError):
        ChunkerConfig("parent_child", size=64, parent_size=32).validate()


def test_default_overlap_and_fingerprint_change_with_params():
    assert ChunkerConfig("fixed", size=128).resolved_overlap() == 16
    assert ChunkerConfig("parent_child", size=48).resolved_overlap() == 12
    a, b = ChunkerConfig("fixed", size=64), ChunkerConfig("fixed", size=65)
    assert a.fingerprint() != b.fingerprint()


# ------------------------------------------------------------------- fixed
def test_fixed_windows_have_the_right_size_and_overlap():
    cs = texts("fixed", 20, overlap=5)
    words = [c.split() for c in cs]
    assert all(len(w) <= 20 for w in words)
    assert all(len(w) == 20 for w in words[:-1])
    for a, b in zip(words, words[1:]):
        assert a[-5:] == b[:5]


def test_fixed_covers_every_word_of_the_document():
    cs = texts("fixed", 20, overlap=5)
    assert set(" ".join(cs).split()) == set(DOC.split())


def test_fixed_chunks_are_exact_substrings_of_the_source():
    for c in texts("fixed", 20, overlap=0):
        assert c in DOC


def test_fixed_splits_a_table_away_from_its_header():
    big = [c for c in texts("fixed", 40) if "r-25 " in c][0]
    assert "| Region | Endpoint | Budget |" not in big


# ---------------------------------------------------------------- sentence
def test_sentence_chunks_respect_the_size_and_cover_the_words():
    cs = texts("sentence", 30)
    assert all(count_tokens(c) <= 30 for c in cs)
    assert set(" ".join(cs).split()) == set(DOC.split())


def test_sentence_keeps_sentences_whole_when_they_fit():
    cs = texts("sentence", 30)
    assert any("The default timeout is 30 seconds." in c for c in cs)


def test_oversize_sentence_is_hard_split_not_dropped():
    units = _hard_split("w " * 50, 0, 20)
    assert len(units) == 3
    assert units[0][0] == 0 and units[-1][1] == len("w " * 50)


# --------------------------------------------------------------- recursive
def test_recursive_units_tile_the_text_exactly():
    text = DOC
    units = _recursive_units(text, 0, 25, ("\n\n", "\n", ". ", " "))
    assert "".join(text[a:b] for a, b in units) == text
    assert all(count_tokens(text[a:b]) <= 25 for a, b in units)


def test_recursive_chunks_respect_the_size():
    assert all(count_tokens(c) <= 40 for c in texts("recursive", 40))


def test_recursive_keeps_a_small_table_whole():
    assert any("| Free | 3 |" in c and "| Team | 20 |" in c for c in texts("recursive", 60))


def test_pack_units_overlap_carries_trailing_units():
    chunks = _pack_units([4, 4, 4, 4, 4], size=10, overlap=4)
    assert chunks[0] == (0, 2, 0)
    assert chunks[1][0] == 1 and chunks[1][2] == 2  # unit 1 carried, unit 2 first new


# --------------------------------------------------------------- structure
def test_structure_keeps_small_table_list_and_code_whole():
    cs = texts("structure", 60)
    assert any("| Plan | Seats |" in c and "| Team | 20 |" in c for c in cs)
    assert any("1. Open the page." in c and "3. Wait." in c for c in cs)
    assert any("batch_size: 100" in c and "flush_ms: 500" in c for c in cs)


def test_structure_splits_a_big_table_by_rows_and_repeats_the_header():
    cs = [c for c in texts("structure", 40) if "| r-" in c]
    assert len(cs) > 2
    assert all("| Region | Endpoint | Budget |" in c for c in cs)
    rows = [ln for c in cs for ln in c.splitlines() if ln.startswith("| r-")]
    assert len(rows) == 30 and len(set(rows)) == 30  # no row lost or duplicated


def test_structure_chunks_stay_within_the_size():
    # a repeated header can never push a chunk past the size here
    assert all(count_tokens(c) <= 40 for c in texts("structure", 40))


def test_structure_never_leaves_a_heading_orphaned_at_a_chunk_end():
    for c in texts("structure", 20):
        last = c.strip().splitlines()[-1]
        assert not last.startswith("#")


def test_a_heading_is_glued_to_a_block_that_does_not_fit_beside_it():
    # the paragraph (19 words) cannot share a 20-word chunk with a 3-word heading line;
    # the heading still travels with it, so the chunk is allowed to exceed the size by
    # the heading's own length rather than leave "## Sec title" alone at a chunk end
    para = " ".join(f"w{i}" for i in range(19))
    chunks = [c.text for c in chunk_document("g", f"## Sec title\n\n{para}\n\nTail text.",
                                             cfg("structure", 20, overlap=0))]
    assert chunks[0] == f"## Sec title\n\n{para}"
    assert count_tokens(chunks[0]) == 22
    assert chunks[1] == "Tail text."


def test_structure_loses_no_content():
    body = norm(" ".join(texts("structure", 40)))
    for line in DOC.splitlines():
        if line.strip() and not line.startswith("|---"):
            assert norm(line) in body or all(w in body for w in line.split())


def test_structure_chunk_path_follows_headings():
    chunks = chunk_document("g", DOC, cfg("structure", 60, enrich="path"))
    limits = [c for c in chunks if "default timeout" in c.text][0]
    assert limits.path == "Gadget guide > Limits"
    assert limits.prefix == "Gadget guide > Limits"


def test_structure_overlaps_a_split_paragraph():
    chunks = [c.text for c in chunk_document("g", DOC, cfg("structure", 30, overlap=8))
              if "Sentence number" in c.text]
    assert len(chunks) > 3
    shared = sum(1 for a, b in zip(chunks, chunks[1:])
                 if set(a.split()[-3:]) & set(b.split()[:12]))
    assert shared >= 1


# ------------------------------------------------------------ parent/child
def test_parent_child_returns_the_parent_but_indexes_the_child():
    chunks = chunk_document("g", DOC, cfg("parent_child", 16, parent_size=80))
    assert all(c.text in c.context_text for c in chunks)
    assert all(count_tokens(c.text) <= 16 for c in chunks)
    assert any(c.text != c.context_text for c in chunks)
    parent = [c for c in chunks if "default timeout" in c.text][0].context_text
    assert "| Plan | Seats |" in parent  # the child's parent holds the whole section


# --------------------------------------------------------------- enrichment
def test_enrichment_prefix_modes():
    none = chunk_document("g", DOC, cfg("fixed", 30, enrich="none"))
    title = chunk_document("g", DOC, cfg("fixed", 30, enrich="title"))
    path = chunk_document("g", DOC, cfg("fixed", 30, enrich="path"))
    assert all(c.prefix == "" for c in none)
    assert all(c.prefix == "Gadget guide" for c in title)
    assert {c.prefix for c in path} > {"Gadget guide"}
    assert any(c.prefix.endswith("> Regions") for c in path)
    assert [c.text for c in none] == [c.text for c in title]  # the prefix is not in the text


def test_untitled_documents_fall_back_to_the_doc_id():
    chunks = chunk_document("my-doc", "no heading\n\nsome text here", cfg("fixed", 10))
    assert chunks[0].prefix == "my-doc"


# ------------------------------------------------------------------- keys
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_chunking_is_deterministic_and_keys_are_unique(strategy):
    a = chunk_document("g", DOC, cfg(strategy, 40))
    b = chunk_document("g", DOC, cfg(strategy, 40))
    assert a == b
    assert len({c.key for c in a}) == len(a)
    assert [c.ordinal for c in a] == list(range(len(a)))


def test_same_text_in_two_documents_has_different_keys_but_the_same_embed_key():
    a = chunk_document("a", "# T\n\nhello there world", cfg("fixed", 10, enrich="title"))
    b = chunk_document("b", "# T\n\nhello there world", cfg("fixed", 10, enrich="title"))
    assert a[0].key != b[0].key
    assert a[0].embed_key == b[0].embed_key


def test_empty_document_has_no_chunks():
    for strategy in STRATEGIES:
        assert chunk_document("e", "", cfg(strategy, 10)) == []
