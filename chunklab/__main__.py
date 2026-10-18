"""python -m chunklab demo [--small] [--json PATH]
   python -m chunklab show [--doc ID] [--strategy S] [--size N] [--enrich E]
"""

import argparse
import json
import sys

from .chunkers import STRATEGIES, ChunkerConfig, chunk_document
from .corpus import generate_corpus
from .experiments import (format_ingest, format_lifecycle, format_quality, run_ingest,
                          run_lifecycle, run_quality)
from .text import count_tokens


def _corpus_summary(corpus):
    words = sum(count_tokens(t) for t in corpus.docs.values())
    flat = sum(1 for s in corpus.specs.values() if s.flat)
    kinds = {}
    for q in corpus.questions:
        kinds[q.kind] = kinds.get(q.kind, 0) + 1
    return (f"corpus: {len(corpus.docs)} synthetic docs ({flat} flat), "
            f"{words / len(corpus.docs):.0f} words each, {len(corpus.questions)} questions "
            f"{dict(sorted(kinds.items()))}")


def _strip(obj):
    """Drop the per-question outcome lists (hundreds of booleans per row) from the JSON."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "outcomes"}
    return obj


def demo(args):
    corpus = generate_corpus(n_docs=40 if args.small else 200, seed=args.seed)
    trials = 3 if args.small else 10
    print(_corpus_summary(corpus))
    print("\n=== 1. Retrieval quality by chunking strategy ===")
    quality = run_quality(corpus)
    print(format_quality(quality))
    print("\n=== 2. Cost of keeping the index fresh ===")
    ingest = run_ingest(corpus, trials=trials)
    print(format_ingest(ingest))
    print("\n=== 3. Ingestion lifecycle (structure chunker, 128 words) ===")
    lifecycle = run_lifecycle(corpus)
    print(format_lifecycle(lifecycle))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_strip({"quality": quality, "ingest": ingest, "lifecycle": lifecycle}),
                      fh, indent=1, default=str)
        print(f"\nwrote {args.json}")


def show(args):
    corpus = generate_corpus(n_docs=args.docs, seed=args.seed)
    doc_id = args.doc or next(d for d, s in sorted(corpus.specs.items())
                              if s.regions and not s.flat)
    cfg = ChunkerConfig(args.strategy, size=args.size, enrich=args.enrich)
    chunks = chunk_document(doc_id, corpus.docs[doc_id], cfg)
    print(f"{doc_id}: {count_tokens(corpus.docs[doc_id])} words -> {len(chunks)} chunks "
          f"({args.strategy}, size {args.size}, enrich {args.enrich})")
    for c in chunks:
        head = c.text.replace("\n", " / ")[:90]
        print(f"[{c.ordinal:>2}] {count_tokens(c.text):>3}w  path='{c.path}'  {head}...")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="chunklab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run every measurement")
    d.add_argument("--small", action="store_true", help="40 docs, 3 trials (CI smoke run)")
    d.add_argument("--seed", type=int, default=17)
    d.add_argument("--json", default="")
    d.set_defaults(fn=demo)
    s = sub.add_parser("show", help="print how one document is chunked")
    s.add_argument("--doc", default="")
    s.add_argument("--strategy", default="structure", choices=STRATEGIES)
    s.add_argument("--size", type=int, default=128)
    s.add_argument("--enrich", default="path", choices=("none", "title", "path"))
    s.add_argument("--docs", type=int, default=200)
    s.add_argument("--seed", type=int, default=17)
    s.set_defaults(fn=show)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
