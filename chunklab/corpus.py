"""A deterministic synthetic documentation corpus, with questions that have exact answer
spans, and a small edit simulator for the incremental-ingestion experiments.

Every product is invented ("Ledgerly guide", "Quilnim guide", ...). Every number is random.
There is no real data, and no payment-card or health data of any kind.

Each question carries `must_contain`: strings that must ALL appear in one retrieved
context of the gold document for the question to count as answered. For a table cell the
spans are the whole row line plus the column header, because a number with no header is
not an answer (`core` holds the row alone, for the lenient "answer-only" metric). For a
procedure they are the first, a middle and the last step, so a procedure cut in half does
not count.

About 15% of the documents are "flat": the same content with headings, table pipes and
code fences stripped, like a PDF-to-text export. A structure-aware chunker has nothing to
parse there, and the experiments report that group separately.

The corpus uses its own tiny generator on top of random.random() so the numbers in the
README reproduce on Python 3.11 and 3.13 (the other random.* helpers may change between
versions).
"""

import hashlib
import random
from dataclasses import dataclass, field, replace

KINDS = ("fact", "para", "table", "bigtable", "code", "steps", "prose")

SYLLABLES = ["led", "ger", "ly", "quil", "base", "nim", "bus", "vor", "tex", "kel", "pan",
             "dra", "flu", "mox", "zen", "art", "cro", "sol", "ban", "wex"]
NOUNS = ["records", "events", "tickets", "orders", "files", "messages", "reports", "assets"]
SYSTEMS = ["the warehouse", "the CRM", "the ticket desk", "the object store", "the data lake",
           "the metrics store", "the mail queue", "the search index"]
REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-west-2",
           "eu-west-3", "eu-central-1", "eu-north-1", "ap-south-1", "ap-south-2",
           "ap-northeast-1", "ap-northeast-2", "ap-northeast-3", "ap-southeast-1",
           "ap-southeast-2", "ap-southeast-3", "ca-central-1", "sa-east-1", "me-south-1",
           "af-south-1", "il-central-1", "mx-central-1", "eu-south-1", "eu-south-2",
           "us-gov-west-1", "us-gov-east-1", "cn-north-1", "cn-northwest-1", "ap-east-1"]
ERROR_MEANINGS = ["Invalid credentials", "Quota exceeded", "Source unavailable",
                  "Mapping failed", "Destination rejected the batch", "Schema mismatch",
                  "Checkpoint corrupted", "Payload too large"]
FEATURES = ["dry run mode", "pause and resume", "tag filters", "run history export",
            "mapping preview", "webhook retries", "workspace roles", "bulk import"]
FILLER = [
    "Operators can raise the timeout when the network between regions is slow.",
    "Each run writes a summary that lists the {n} it moved and the {n} it skipped.",
    "Large batches are split into smaller requests so that one slow request does not block the rest.",
    "The console shows the status of every connector in a single list.",
    "Keep the credentials for each environment in a separate secret.",
    "A dry run reports what would change without writing any {n}.",
    "Webhooks notify your own service when a run finishes.",
    "Use tags to group connectors by team, by environment or by cost centre.",
    "If a record fails validation it is written to the run log with a short reason.",
    "Schedules are evaluated in UTC, so plan around daylight saving changes.",
    "The retry count can be lowered for jobs that must not repeat.",
    "Audit entries are kept for the retention period of your plan.",
    "Plan limits apply per workspace, not per connector.",
    "Rate limit errors include a header that tells you when to try again.",
    "Latency depends on the region and on the size of each batch of {n}.",
    "Use the preview environment to test a new mapping before it reaches production.",
    "Every connector has an owner, and the owner receives the failure alerts.",
    "Pausing a connector keeps its checkpoint so the next run resumes where it stopped.",
    "Mappings translate source fields to destination fields and can rename or drop them.",
    "The default behaviour is safe for most teams, and the settings below are optional.",
    "A failed run never deletes {n} that were already written.",
    "Check the status page before opening a support request about slow runs.",
    "Connectors are stateless, so you can move one to another host at any time.",
    "Older log files are compressed after a week to save space.",
]
PLANS = [("Free", 30, 120, 1, 5), ("Team", 200, 900, 10, 50),
         ("Business", 1000, 5000, 60, 300), ("Enterprise", 6000, 20000, 400, 2000)]
PLAN_COLUMNS = ["Requests per minute", "Seats", "Retention (days)"]
CONFIG_KEYS = [("sync", "batch_size", 50, 900), ("sync", "flush_interval_ms", 200, 5000),
               ("sync", "max_inflight", 2, 64)]
SUPPORT = ("For help, contact the platform team through the console. Include the request id "
           "from the failing call. Response times depend on your plan.")


class Rng:
    """Version-stable helpers built only on random.random()."""

    def __init__(self, seed):
        self._r = random.Random(seed)

    def rand(self):
        return self._r.random()

    def int(self, a, b):
        return a + int(self.rand() * (b - a + 1))

    def choice(self, seq):
        return seq[int(self.rand() * len(seq))]

    def chance(self, p):
        return self.rand() < p

    def sample(self, seq, k):
        pool = list(seq)
        out = []
        for _ in range(min(k, len(pool))):
            out.append(pool.pop(int(self.rand() * len(pool))))
        return out


@dataclass(frozen=True)
class Question:
    qid: str
    doc_id: str
    kind: str
    text: str
    must_contain: tuple
    split: str
    fmt: str = "md"  # "md" (markdown structure) or "flat" (structure stripped)
    core: tuple = ()  # the answer value alone (table: the row without its header)

    def core_spans(self):
        return self.core or self.must_contain


@dataclass(frozen=True)
class DocSpec:
    doc_id: str
    name: str
    nouns: tuple
    source: str
    dest: str
    overview: tuple  # two paragraphs
    extra_paragraph: str  # inserted after the first overview paragraph ("" = none)
    timeout: int
    limits_note: str
    limits_para: str
    plans: tuple  # ((plan, rpm, seats, retention), ...)
    config: tuple  # ((section, key, value), ...)
    retries_ms: int
    attempts: int
    errors: tuple  # ((code, meaning, retryable), ...)
    key_days: int
    key_hours: int
    regions: tuple  # ((region, latency_ms), ...) possibly empty
    changelog: tuple  # () means the section is absent
    flat: bool = False  # headings/tables/fences stripped, like a PDF-to-text export
    trouble: tuple = ()  # long prose paragraphs; () means no troubleshooting section
    stall_minutes: int = 0


@dataclass
class Corpus:
    docs: dict
    questions: list
    specs: dict = field(repr=False, default_factory=dict)

    def split(self, name):
        return [q for q in self.questions if q.split == name]


def _paragraph(rng, noun):
    sentences = rng.sample(FILLER, rng.int(3, 5))
    return " ".join(s.replace("{n}", noun) for s in sentences)


def _make_name(rng, used):
    while True:
        name = "".join(rng.choice(SYLLABLES) for _ in range(rng.int(2, 3))).capitalize()
        if name not in used:
            used.add(name)
            return name


def make_spec(rng, name):
    nouns = tuple(rng.sample(NOUNS, 2))
    plans = []
    for plan, lo, hi, slo, shi in PLANS:
        plans.append((plan, rng.int(lo, hi), rng.int(slo, shi), rng.choice([7, 30, 90, 180, 365])))
    config = tuple((s, k, rng.int(lo, hi)) for s, k, lo, hi in CONFIG_KEYS)
    attempts = rng.int(3, 9)
    retries_ms = rng.choice([100, 200, 250, 400, 500, 800])
    codes = rng.sample(range(101, 120), 6)
    errors = tuple(
        (f"E{c}", rng.choice(ERROR_MEANINGS), rng.choice(["yes", "no"])) for c in sorted(codes)
    )
    regions = ()
    if rng.chance(0.2):
        regions = tuple((r, rng.int(20, 400)) for r in rng.sample(REGIONS, 24))
    versions = ["2.4.0", "2.3.1", "2.3.0", "2.2.2", "2.2.0", "2.1.0"]
    changelog = tuple(
        f"- {v}: " + rng.choice([
            f"Added {rng.choice(FEATURES)}.",
            "Fixed a crash when a record had an empty name.",
            f"Improved the speed of large {rng.choice(nouns)} batches.",
            "Updated the console layout.",
        ])
        for v in versions
    )
    flat = rng.chance(0.15)
    trouble, stall = (), 0
    if rng.chance(0.35):
        stall = rng.int(5, 45)
        paras = [_paragraph(rng, nouns[0]) for _ in range(5)]
        pos = rng.int(0, 4)
        paras[pos] += f" If a run stalls for more than {stall} minutes the supervisor restarts it."
        trouble = tuple(paras)
    return DocSpec(
        doc_id=name.lower(),
        name=name,
        nouns=nouns,
        source=rng.choice(SYSTEMS),
        dest=rng.choice(SYSTEMS),
        overview=(_paragraph(rng, nouns[0]), _paragraph(rng, nouns[1])),
        extra_paragraph="",
        timeout=rng.int(10, 90),
        limits_note=rng.choice(FILLER).replace("{n}", nouns[0]),
        limits_para=_paragraph(rng, nouns[1]),
        plans=tuple(plans),
        config=config + (("retry", "max_attempts", attempts), ("retry", "backoff_ms", retries_ms)),
        retries_ms=retries_ms,
        attempts=attempts,
        errors=errors,
        key_days=rng.choice([30, 60, 90, 180]),
        key_hours=rng.choice([1, 2, 4, 12, 24]),
        regions=regions,
        changelog=changelog,
        flat=flat,
        trouble=trouble,
        stall_minutes=stall,
    )


# ------------------------------------------------------------------ rendering
def row_text(spec, cells):
    """One table row as it appears in the document (pipes in markdown, spaces when flat)."""
    cells = [str(c) for c in cells]
    return " ".join(cells) if spec.flat else "| " + " | ".join(cells) + " |"


def _heading(spec, level, text):
    return text if spec.flat else "#" * level + " " + text


def _table(spec, header, rows):
    if spec.flat:
        return [" ".join(header)] + [row_text(spec, r) for r in rows]
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + [row_text(spec, r) for r in rows])


def fact_sentence(spec):
    return f"The default timeout is {spec.timeout} seconds."


def retry_sentences(spec):
    return (
        f"Retries use exponential backoff starting at {spec.retries_ms} ms.",
        f"After {spec.attempts} attempts the job moves to the dead-letter queue.",
    )


def step_lines(spec):
    return (
        f"Open the Settings page in the {spec.name} console.",
        "Choose API keys and press Create key.",
        f"Set the key lifetime to {spec.key_days} days.",
        "Copy the new key into your secret store.",
        f"Revoke the old key after {spec.key_hours} hours.",
    )


def region_cells(spec, i):
    region, ms = spec.regions[i]
    return [region, f"https://{region}.{spec.doc_id}.example.test", ms]


def stall_sentence(spec):
    return f"If a run stalls for more than {spec.stall_minutes} minutes the supervisor restarts it."


def render(spec):
    p = spec.name

    def H(level, text):
        return _heading(spec, level, text)

    out = [H(1, f"{p} guide"), "",
           f"{p} is a fictional connector in this lab. It copies {spec.nouns[0]} from "
           f"{spec.source} to {spec.dest} on a schedule you control.", "",
           H(2, "Overview"), "", spec.overview[0], ""]
    if spec.extra_paragraph:
        out += [spec.extra_paragraph, ""]
    out += [spec.overview[1], "",
            H(2, "Quickstart"), "", f"Follow these steps to run your first sync with {p}.", "",
            f"1. Install the {p} agent on a host that can reach both systems.",
            "2. Sign in and create a workspace.",
            "3. Add a source and a destination.",
            "4. Start a run and watch the status page.",
            f"5. Check the run log for skipped {spec.nouns[0]}.", "",
            H(2, "Configuration"), "",
            f"Settings live in a YAML file named {spec.doc_id}.yaml. Every key is optional.", ""]
    if not spec.flat:
        out.append("```yaml")
    section = None
    for sec, key, value in spec.config:
        if sec != section:
            out.append(f"{sec}:")
            section = sec
        out.append(f"  {key}: {value}")
    if not spec.flat:
        out.append("```")
    out += ["", spec.limits_para, "",
            H(2, "Limits"), "", f"{fact_sentence(spec)} {spec.limits_note}", ""]
    out += _table(spec, ["Plan"] + PLAN_COLUMNS, spec.plans)
    s1, s2 = retry_sentences(spec)
    out += ["", H(2, "Error handling"), "", f"{s1} {s2}", ""]
    out += _table(spec, ["Code", "Meaning", "Retryable"], spec.errors)
    out.append("")
    if spec.changelog:
        out += [H(2, "Changelog"), ""] + list(spec.changelog) + [""]
    if spec.trouble:
        out += [H(2, "Troubleshooting"), ""]
        for para in spec.trouble:
            out += [para, ""]
    out += [H(2, "Rotating the API key"), "",
            "Rotate keys on a schedule that matches your security policy.", ""]
    out += [f"{i}. {s}" for i, s in enumerate(step_lines(spec), 1)]
    out.append("")
    if spec.regions:
        out += [H(2, "Regions"), "", "Endpoints are listed per region.", ""]
        out += _table(spec, ["Region", "Endpoint", "Latency budget (ms)"],
                      [region_cells(spec, i) for i in range(len(spec.regions))])
        out.append("")
    out += [H(2, "Support"), "", SUPPORT, ""]
    return "\n".join(out)


# ------------------------------------------------------------------ questions
def _questions_for(spec, rng, kinds):
    """{kind: (question text, must_contain, core)}"""
    p, qs = spec.name, {}
    if "fact" in kinds:
        text = rng.choice([f"What is the default timeout for {p}?",
                           f"How many seconds is the default timeout in {p}?"])
        qs["fact"] = (text, (fact_sentence(spec),), ())
    if "para" in kinds:
        qs["para"] = (f"How do retries back off in {p}, and what happens after the attempts "
                      "are used up?", retry_sentences(spec), ())
    if "table" in kinds:
        row = rng.choice(spec.plans)
        col = PLAN_COLUMNS[rng.int(0, 2)]
        qs["table"] = (f"In {p}, what is the {col.lower()} on the {row[0]} plan?",
                       (row_text(spec, row), col), (row_text(spec, row),))
    if "bigtable" in kinds and spec.regions:
        i = rng.int(0, len(spec.regions) - 1)
        row = row_text(spec, region_cells(spec, i))
        qs["bigtable"] = (f"What is the latency budget in ms for region {spec.regions[i][0]} "
                          f"in {p}?", (row, "Latency budget (ms)"), (row,))
    if "code" in kinds:
        keys = rng.sample(list(spec.config), 2)
        lines = tuple(f"{k}: {v}" for _, k, v in keys)
        qs["code"] = (f"In {p}, what are the {keys[0][1]} and {keys[1][1]} settings?", lines, ())
    if "steps" in kinds:
        steps = step_lines(spec)
        qs["steps"] = (f"What are the steps to rotate the API key in {p}?",
                       (steps[0], steps[2], steps[4]), ())
    if "prose" in kinds and spec.trouble:
        qs["prose"] = (f"After how many minutes does the supervisor restart a stalled run "
                       f"in {p}?", (stall_sentence(spec),), ())
    return qs


def split_of(doc_id):
    return "dev" if int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 2 == 0 else "heldout"


def generate_corpus(n_docs=200, seed=17):
    rng = Rng(seed)
    used, specs, docs, questions = set(), {}, {}, []
    weights = [("fact", 3), ("para", 2), ("table", 3), ("code", 2), ("steps", 2)]
    for _ in range(n_docs):
        spec = make_spec(rng, _make_name(rng, used))
        specs[spec.doc_id] = spec
        docs[spec.doc_id] = render(spec)
        pool = [k for k, w in weights for _ in range(w)]
        chosen = set()
        for _ in range(rng.int(2, 3)):
            chosen.add(rng.choice(pool))
            pool = [k for k in pool if k not in chosen]
        if spec.regions:
            chosen.add("bigtable")
        if spec.trouble:
            chosen.add("prose")
        built = _questions_for(spec, rng, chosen)
        for kind in KINDS:
            if kind in built:
                text, must, core = built[kind]
                questions.append(Question(f"{spec.doc_id}:{kind}", spec.doc_id, kind, text,
                                          must, split_of(spec.doc_id),
                                          "flat" if spec.flat else "md", core))
    return Corpus(docs=docs, questions=questions, specs=specs)


# ----------------------------------------------------------------------- edits
@dataclass(frozen=True)
class Edit:
    doc_id: str
    kind: str  # value | insert | delete
    question: Question = None  # for value edits: the fact question with the NEW span


def apply_edits(corpus, seed, fraction=0.05):
    """Edit ~fraction of the documents, cycling value / insert / delete.

    value   changes the default timeout to another number (same word count: in place)
    insert  adds a paragraph to the overview (shifts every word after it)
    delete  removes the changelog section (shifts every word after it)
    Returns (new_docs, edits). The input corpus is not modified.
    """
    rng = Rng(seed)
    ids = sorted(corpus.docs)
    n = max(1, round(fraction * len(ids)))
    new_docs = dict(corpus.docs)
    edits = []
    cycle = ("value", "insert", "delete")
    for i, doc_id in enumerate(rng.sample(ids, n)):
        spec, kind = corpus.specs[doc_id], cycle[i % 3]
        question = None
        if kind == "value":
            spec = replace(spec, timeout=spec.timeout + rng.int(1, 9))
            question = Question(f"{doc_id}:fact-edited", doc_id, "fact",
                                f"What is the default timeout for {spec.name}?",
                                (fact_sentence(spec),), split_of(doc_id),
                                "flat" if spec.flat else "md")
        elif kind == "insert":
            spec = replace(spec, extra_paragraph=_paragraph(rng, spec.nouns[0]))
        else:
            spec = replace(spec, changelog=())
        new_docs[doc_id] = render(spec)
        edits.append(Edit(doc_id, kind, question))
    return new_docs, edits
