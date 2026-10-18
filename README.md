# rag-chunking-ingest-lab

![ci](https://github.com/Sup3rMar10/rag-chunking-ingest-lab/actions/workflows/ci.yml/badge.svg)

The **ingestion half of RAG**: how a document is cut into retrieval units, and how the index is kept fresh when documents change. Five chunking strategies are measured on the same questions with the same BM25 retriever, and an incremental ingester (content hashes, tombstones, a content-addressed embedding cache, transactional documents) is measured against a full rebuild.

> **What is verified, and what is not.** 114 tests pass locally on Python 3.13 (standard library only), and the CI workflow in this repo runs them on Python 3.11 and 3.13. Retrieval numbers come from `python -m chunklab demo` (about two minutes; the output is committed in `results/demo_output.txt`). I also broke the code on purpose in 16 places (a table split that drops its header, a metric that ignores the gold document, tombstones that stay visible, a failed document marked as done, ...) and the tests failed each time. **What this does not show:** the corpus is *synthetic and regular* (200 invented product guides generated from templates, 642 words each), the questions are single-hop with one exact answer span, the retriever is plain BM25 (no dense model, no re-ranker, no LLM), and the "embedding" is a hashing stand-in used only to count how many embeddings an ingest run would pay for. A "token" is a whitespace-delimited word. Real, messier documents will not give these numbers; the point is the method and the failure modes it exposes.

## What counts as "answered"

Chunking can only guarantee that the answer is *in one retrieved unit*. So the metric is **cover@k**: some context in the top k comes from the **gold document** and contains **every required span in one piece**.

| Question kind | Required spans | Why it is a chunking test |
|---|---|---|
| `fact` | one sentence ("The default timeout is 32 seconds.") | easy for any chunker if the right chunk is ranked |
| `para` | two consecutive sentences | a window boundary can fall between them |
| `table` | the whole row **and the column header** | a number with no header is not an answer |
| `bigtable` | same, in a 24-row table | wider than most chunks; the header only exists at the top |
| `code` | two lines of a YAML block | a block can be cut between the two |
| `steps` | first, middle and last step of a numbered list | half a procedure is not a procedure |
| `prose` | one sentence buried in five long paragraphs | the "structure does not help" control |

`answer-only@3` relaxes table questions to just the row (no header), to show how much of a gap is the header rule. `dochit@k` (the right document was located) and context size (words handed to the model) are reported too.

## The five strategies

| Strategy | What it does |
|---|---|
| `fixed` | word windows with 12.5% overlap. Structure-blind. |
| `sentence` | sentences packed up to a size. Sentences longer than the size (a table has no punctuation) are hard-split. |
| `recursive` | split on paragraph, then line, then sentence, then word; merge pieces back up to the size. The widely used strong baseline. |
| `structure` | parses headings, tables, lists and code. Small tables/lists/code stay whole; a big table splits **by rows with the header repeated**; a heading is glued to its first block; the section's heading path is recorded. |
| `parent_child` | indexes small child windows but returns the whole parent section ("small to big"). |

**Metadata enrichment** is an independent switch: `none`, `title` (the document title), or `path` (`Title > Section > Subsection`) is indexed together with the chunk text but is *not* part of the span a question has to find.

Look at what each strategy actually produces: `python -m chunklab show --strategy fixed --size 64 --enrich title` prints one document's chunks (words, heading path, first words). On the sample document the fixed chunker starts a chunk in the middle of a table (`16173 | 405 | 365 |`) and separates a table from its header; the structure chunker does not.

## Protocol

- Questions are split **by document** (319 dev / 296 held-out). A document is entirely one or the other.
- Each strategy's chunk size is chosen **on dev only**, then scored once on held-out. The choice is made under a **context budget** (dev context at k = 3 must be at most 400 words), because without a budget the biggest chunk always wins. That unbudgeted choice is reported separately.
- A second table fixes every strategy at the **same size**, so a win cannot be explained by a different chunk size.
- 95% Wilson intervals on cover@3, and exact McNemar tests on paired outcomes (same questions, different chunkers).

## Results (held-out, 296 questions)

**Size chosen on dev within the context budget** (the realistic setting):

| Strategy | Size | Chunks | Words/chunk | cover@1 | cover@3 (95% CI) | cover@5 | answer-only@3 | Context words@3 |
|---|---|---|---|---|---|---|---|---|
| fixed | 128 | 1203 | 120 | 66.9% | 87.8% (84-91) | 93.9% | 91.2% | 369 |
| sentence | 128 | 1245 | 112 | 76.7% | 93.9% (91-96) | 97.6% | 95.3% | 345 |
| recursive | 128 | 1313 | 102 | 82.1% | 92.2% (89-95) | 96.6% | 95.6% | 310 |
| **structure** | 64 | 3011 | 43 | 92.6% | 96.6% (94-98) | 99.0% | 96.6% | **102** |
| **parent_child** (child size) | 96 | 2156 | 65 | 95.9% | 100.0% (99-100) | 100.0% | 100.0% | 290 |

**Same size for everyone** (128 words; parent_child child = 48):

| Strategy | cover@1 | cover@3 (95% CI) | answer-only@3 | Context words@3 |
|---|---|---|---|---|
| fixed | 66.9% | 87.8% (84-91) | 91.2% | 369 |
| sentence | 76.7% | 93.9% (91-96) | 95.3% | 345 |
| recursive | 82.1% | 92.2% (89-95) | 95.6% | 310 |
| structure | 89.2% | 94.6% (91-97) | 94.6% | 189 |
| parent_child | 98.3% | 99.3% (98-100) | 99.3% | 383 |

**No budget:** fixed, sentence and recursive all reach **100.0% cover@3 at 512 words**, with about 1,200 context words (structure and parent_child keep their budgeted choice). One chunk per whole document also gets 100% and sends 1,777 words. Coverage is easy to buy with context; the interesting number is coverage *per word sent*.

By question kind (cover@3, size chosen within the budget):

| Strategy | bigtable (n=17) | code (42) | fact (69) | para (51) | prose (27) | steps (38) | table (52) |
|---|---|---|---|---|---|---|---|
| fixed | 41.2% | 100.0% | 84.1% | 90.2% | 100.0% | 86.8% | 90.4% |
| sentence | 76.5% | 100.0% | 89.9% | 96.1% | 100.0% | 92.1% | 96.2% |
| recursive | 41.2% | 100.0% | 88.4% | 90.2% | 100.0% | 100.0% | 100.0% |
| structure | 88.2% | 95.2% | 95.7% | 100.0% | 92.6% | 100.0% | 98.1% |
| parent_child | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

The clearest chunking effect is `bigtable`: fixed and recursive split a 24-row table and lose the header (41%); structure repeats the header on every piece (88%); parent_child returns the whole section (100%).

## Where structure-aware chunking does NOT help

Read these before quoting the table above.

1. **At the same size the cover@3 gap is small.** Structure 94.6% vs sentence 93.9% vs recursive 92.2% at 128 words, with overlapping intervals. Where it clearly wins is cover@1 (89% vs 77% and 82%) and context size (189 vs 310-369 words). The paired test on the budgeted configurations (structure 21 questions better, recursive 8 better) gives p = 0.024, and I ran five paired comparisons, so it does not survive a Bonferroni threshold of 0.01.
2. **On "flat" documents it does nothing.** 15% of the documents are the same content with headings, pipes and fences stripped (a PDF-to-text export). Held-out cover@3 on those 47 questions: structure 80.9%, sentence 80.9%, recursive 80.9%, fixed 76.6%. A structure-aware chunker needs structure to read. (parent_child reaches 100% on them because it returns a large parent.)
3. **Its own weaker rows.** At the budgeted size (64) structure scores 95.2% on `code` and 92.6% on `prose` where the baselines score 100%. I checked every miss: the two `code` misses are flat documents (the intact chunk exists but the Quickstart chunk outranks it), and the two `prose` misses are ranking misses (the top three results are *other* documents' near-identical "the supervisor restarts it" sentence). Chunking cannot fix ranking; a hybrid or re-ranking stage would. In each of the four cases exactly one intact chunk containing the answer existed.
4. **Enrichment is a bigger lever than the chunker.** Held-out cover@3 with no prefix is 32-41% for *every* strategy (the product name lives only in the title); with the title it is 88-100%. Adding the heading path on top of the title changes it by at most 1 point (structure 96.6% to 97.6%, fixed 87.8% to 88.5%) because in this corpus the heading is already inside the structure chunk's text and the documents are short. The corpus is built so that the title identifies the product, as in real documentation, so this ablation is by construction.
5. **Cost savings are mostly "only re-embed what changed", not chunk hashing.** A full rebuild after a 5% edit embeds 1,204-3,879 chunks; every incremental variant (doc-level or chunk-hash) embeds 0.7-5% of that, simply because 95% of the documents did not change. The honest comparison is document-level vs chunk-level (next section).
6. **The strong baseline is strong.** `recursive` (paragraph, then line, then sentence) is within the noise of `sentence` and close to `structure` on markdown documents.

## Enrichment ablation (held-out cover@3, size chosen within the budget)

| Strategy | none | title | path |
|---|---|---|---|
| fixed | 33.1% | 87.8% | 88.5% |
| sentence | 35.8% | 93.9% | 93.9% |
| recursive | 38.2% | 92.2% | 92.2% |
| structure | 32.1% | 96.6% | 97.6% |
| parent_child | 41.2% | 100.0% | 100.0% |

## Keeping the index fresh

`chunklab.ingest.IngestStore` follows these rules:

- A document is re-chunked only if its **text hash** or the **chunker config** changed.
- Chunks are identified by content. A chunk that survives an edit is kept; a new chunk is embedded; a chunk that no longer exists is **tombstoned** (hidden from search, stored for a grace period).
- Embeddings are cached by the text that is embedded, so identical text is embedded once and a chunk revived inside the grace period needs no new embedding.
- Each document is a **transaction**: if embedding fails half way, the document keeps its previous version, nothing partial becomes visible, and the next run retries it.
- Re-running with the same source changes nothing (**idempotent**; the state is compared byte for byte). `compact()` drops old tombstones and unreferenced embeddings.
- `stale_report(source)` lists changed, missing and orphaned documents, config drift, and the exact live chunks the next ingest would tombstone.
- The state of an incremental run is **identical to a full rebuild** (checked by a test for every strategy and several edit seeds, and in every experiment trial).

**Cost of a 5% edit** (10 documents out of 200, 10 trials, each trial edits a different random set cycling *value* / *insert* / *delete*). Chunks embedded per run:

| Strategy | Chunks/doc | Full rebuild | Doc-level (re-embed all chunks of changed docs) | Chunk-hash | Chunk-hash vs doc-level | value edit | insert edit | delete edit |
|---|---|---|---|---|---|---|---|---|
| fixed | 6.1 | 1204 | 60.8 | 33.0 | -46% | 1.1 | 6.8 | 2.7 |
| sentence | 6.3 | 1246 | 62.7 | 26.0 | -59% | 1.2 | 4.5 | 2.6 |
| recursive | 6.6 | 1313 | 65.8 | 19.1 | -71% | 1.1 | 3.1 | 1.8 |
| structure | 9.4 | 1923 | 94.5 | 12.6 | -87% | 1.0 | 2.4 | 0.5 |
| parent_child | 19.4 | 3879 | 193.8 | 25.9 | -87% | 1.0 | 6.1 | 1.1 |

*value / insert / delete* = chunks embedded per edited document of that kind. An in-place edit (same word count) costs about one chunk for every strategy. An **insertion shifts every window after it** for fixed chunking (6.8 chunks) but only the touched section for structure (2.4); deleting a section does the same (2.7 vs 0.5). So chunk-level hashing saves the most when chunk boundaries are stable, which is a property of the chunker, not of the hashing.

**Freshness.** After an edit, the new value was retrievable (cover@3, 40 edited-fact questions) in 0/40 cases before re-ingest for every strategy (the index serves the old value), and in 30/40 (fixed), 35/40 (sentence), 37/40 (recursive), 39/40 (structure) and 40/40 (parent_child) after.

**Lifecycle run** (structure, 128 words, 200 documents): first ingest 200 docs, 1923 chunks, 1923 embeddings; re-run unchanged: no change, 0 embeddings, state identical; stale report before re-ingest: 10 changed docs, 14 stale chunks; after editing 10 docs: +13 chunks, 75 kept, -14 tombstoned, 13 embeddings, 0 stale left; delete 10 docs: 102 chunks tombstoned, live 1922 to 1820, 0 results leaked from deleted documents; compact inside the grace period removes nothing, after the grace period removes 116 chunks and 116 embeddings; re-adding the deleted docs inside the grace period revives 102 chunks with 0 embeddings; an embedding outage on the 3rd call fails 1 document, updates the other 9, the failed document keeps serving its old chunks, and the retry completes it.

## Run it

```
python -m pytest                          # 114 tests, about a minute
python -m chunklab demo                   # every table above (about 2 minutes, stdlib only)
python -m chunklab demo --small           # 40 docs, 3 trials (the CI smoke run)
python -m chunklab show --strategy structure --size 64
python -m chunklab show --strategy fixed --size 64 --enrich title
```

```
chunklab/
  text.py        words, terms, sentences
  blocks.py      markdown-ish block parser (heading / para / list / table / code), heading paths
  chunkers.py    the five strategies and the Chunk record
  bm25.py        BM25 in the standard library
  retrieval.py   index + context de-duplication (parent_child)
  corpus.py      deterministic synthetic docs, questions with exact spans, edit simulator
  metrics.py     cover@k, dochit, context size, Wilson interval, McNemar
  ingest.py      IngestStore: hashes, tombstones, embedding cache, transactions, stale report
  experiments.py the measurements and the table formatting
```

The corpus generator uses only `random.random()` (the other `random.*` helpers may change between Python versions), so the numbers reproduce on 3.11 and 3.13.

## Limits

- Synthetic, template-generated documents with very regular sections (about 70 words each). Real manuals, wikis, tickets and contracts are messier, so absolute numbers will differ.
- BM25 only. A dense retriever, a hybrid, or a re-ranker would change which chunks are found (the two `prose` misses above are exactly that); `rag-tuning-lab` looks at that side.
- Single-hop questions with one exact span. No multi-section answers, no paraphrased questions (the questions reuse the document's own words).
- Word counts, not model tokens. The embedding is a hashing stand-in used only to count cost. No live model, no API key, nothing to untest.
- Chunk sizes were swept over four values each, and the budgeted choice for `structure` is a dev tie (64 and 128 are both 97.2%) broken toward less context; the same-size table shows 128 as well.
- 296 held-out questions: differences under about 4 points are within noise, and I ran several comparisons.
