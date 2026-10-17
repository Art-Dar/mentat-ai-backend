# Gold sets for retrieval tuning (SB-54)

`goldset.example.json` is a worked example against the demo corpus. It is
a template, not a gold set for your library — the spans in it only match
those documents.

## Writing your own

```json
[
  {
    "question": "When is invoice 2026-0042 due?",
    "answer_spans": ["net 14 days from the invoice date"],
    "tags": ["invoice", "precise-fact"]
  }
]
```

**`answer_spans` are literal text from your documents**, not the answer
you would like to read. A retrieval counts as a hit when one of these
strings appears in a retrieved chunk. They are labelled this way, rather
than by chunk id, because changing chunk size is the point of the
exercise and chunk 7 does not survive it — a span is defined in the
document's own character space, which no chunking changes.

Several spans are **alternatives**: any one of them matching counts as a
hit. Use that when the same fact appears twice, for instance in a table
and again in a sentence.

**Spans under 12 characters are refused.** "480.00" appears in almost any
invoice chunk, so it scores a hit for a retrieval that found nothing
relevant — and it inflates every setting equally, which looks like a
stable result rather than a broken one.

`tags` are free text and only used for slicing the results. They are
worth filling in: tuning that lifts the average while destroying one
category is common and invisible in a single number.

## How many, and of what

Twenty is the minimum that says anything; forty is comfortable.

**Include questions you expect to fail.** A gold set of easy lookups
reports a high number and tells you nothing. The demo set deliberately
contains a transliterated-Ukrainian receipt, which the offline embedder
gets wrong every time — that miss is the most useful row in the output.

**Your corpus needs distractors.** With four documents and k=5, retrieval
returns the whole library and recall is 1.000 by construction. The
harness flags this (`← k covers the corpus`) and refuses to draw a
conclusion from it, but the real fix is a corpus with plenty of documents
that do *not* answer any question in the set.

## Running

```bash
# sanity-check the labels against the corpus, then sweep
python scripts/tune_retrieval.py --gold eval/goldset.json \
    --user-id <uuid> --offline

# the numbers you actually act on
python scripts/tune_retrieval.py --gold eval/goldset.json --user-id <uuid>
```

`--offline` uses a local TF-IDF embedder: no API key, no bill, real
lexical semantics. It is for developing the sweep and proving the harness
works. It matches words rather than meaning, so do not change a setting
based on it.

The sweep **re-chunks the documents it evaluates**, repeatedly, and
leaves the corpus at the last setting tried. Run
`scripts/backfill_chunks.py` afterwards.
