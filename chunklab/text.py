"""Small text helpers shared by the chunkers, the index and the metrics.

A "token" in this repo is a whitespace-delimited word. It is a stand-in for an LLM
token (a real tokenizer would count differently, roughly 1.3x more for English), so
every size and context number here is in words, and comparisons between strategies
are fair because they all use the same unit.
"""

import re

WORD = re.compile(r"\S+")
TERM = re.compile(r"[a-z0-9_]+")
SENT_BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the this to "
    "what which with does do".split()
)


def count_tokens(text):
    return len(WORD.findall(text))


def norm(text):
    """Collapse all whitespace so a span can be matched across line breaks."""
    return " ".join(text.split())


def terms(text):
    return [t for t in TERM.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text):
    """Split on sentence-ending punctuation or a blank line. Naive on purpose:
    table rows and code lines carry no punctuation, so they glue together."""
    return [s.strip() for s in SENT_BREAK.split(text) if s.strip()]


def word_spans(text):
    """(start, end) character offsets of every word."""
    return [(m.start(), m.end()) for m in WORD.finditer(text)]
