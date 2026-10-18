from chunklab.blocks import heading_paths, parse_blocks, path_at, title_of
from chunklab.text import count_tokens, norm, split_sentences, terms, word_spans

DOC = """# Widget guide

Intro paragraph. It has two sentences.

## Limits

The default timeout is 30 seconds.

| Plan | Seats |
|---|---|
| Free | 3 |
| Team | 20 |

### Notes

1. First step.
2. Second step.

```yaml
# not a heading
retry:
  max_attempts: 4
```

After the code.
"""


def kinds(text):
    return [b.kind for b in parse_blocks(text)]


def test_count_tokens_is_whitespace_words():
    assert count_tokens("a  b\nc\td") == 4
    assert count_tokens("") == 0


def test_norm_collapses_whitespace():
    assert norm("a \n  b\t c") == "a b c"


def test_terms_lowercase_and_drop_stopwords():
    assert terms("What is the Default timeout_ms?") == ["default", "timeout_ms"]


def test_split_sentences_on_punctuation_and_blank_lines():
    assert split_sentences("One. Two!\n\nThree row\nstill three") == [
        "One.", "Two!", "Three row\nstill three"]


def test_word_spans_cover_words_exactly():
    text = "ab  cd\nef"
    assert [text[a:b] for a, b in word_spans(text)] == ["ab", "cd", "ef"]


def test_block_kinds_in_order():
    assert kinds(DOC) == ["heading", "para", "heading", "para", "table", "heading", "list",
                          "code", "para"]


def test_fenced_code_hides_hash_lines_from_the_heading_parser():
    headings = [b.title for b in parse_blocks(DOC) if b.kind == "heading"]
    assert headings == ["Widget guide", "Limits", "Notes"]
    code = [b for b in parse_blocks(DOC) if b.kind == "code"][0]
    assert "# not a heading" in code.text and code.text.startswith("```yaml")


def test_table_and_list_blocks_keep_all_their_lines():
    blocks = {b.kind: b.text for b in parse_blocks(DOC)}
    assert blocks["table"].count("\n") == 3
    assert blocks["list"].count("\n") == 1


def test_block_offsets_point_back_into_the_source():
    for b in parse_blocks(DOC):
        assert DOC[b.start:b.end] == b.text


def test_heading_levels_and_titles():
    hs = [b for b in parse_blocks(DOC) if b.kind == "heading"]
    assert [h.level for h in hs] == [1, 2, 3]
    assert title_of(DOC) == "Widget guide"
    assert title_of("no heading here", default="fallback") == "fallback"


def test_heading_path_nests_and_pops():
    paths = [p for _, p in heading_paths(DOC)]
    assert paths == ["Widget guide", "Widget guide > Limits",
                     "Widget guide > Limits > Notes"]
    flat = "# A\n\n## B\n\n## C\n"
    assert [p for _, p in heading_paths(flat)] == ["A", "A > B", "A > C"]


def test_path_at_uses_the_heading_in_force():
    paths = heading_paths(DOC)
    assert path_at(paths, DOC.index("default timeout")) == "Widget guide > Limits"
    assert path_at(paths, DOC.index("First step")) == "Widget guide > Limits > Notes"
    assert path_at([], 5, default="x") == "x"


def test_text_with_no_structure_is_one_paragraph():
    assert kinds("plain line one\nplain line two") == ["para"]
