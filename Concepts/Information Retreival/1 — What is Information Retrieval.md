# Chapter 1 — What is Information Retrieval?

## What is it?

Information Retrieval (IR) is the problem of finding relevant documents from a large collection given a user query. When you type into Google, Elasticsearch, or a RAG pipeline — that's IR. The system doesn't "understand" your question the way a human does. It computes a score between your query and every document, then sorts by that score.

The core problem: **the corpus is huge, the query is short, and you need an answer in milliseconds.**

---

## The intuition

Imagine a librarian with a million books. You ask: *"anything about neural networks?"* She doesn't read every book on the spot. She uses an **index** — a pre-built lookup table that maps words to books. You say "neural" → she checks the index → gets a list of 3,000 books → picks the most relevant ones.

That index is the heart of IR. Everything else (TF-IDF, BM25, embeddings) is just a smarter way to define "most relevant."

---

## The core equation

```
score(q, d) → rank documents by this score → return top-k
```

Where:
- `q` = query
- `d` = a document in the corpus
- `score` = any relevance function (we build this chapter by chapter)

The entire field of IR is about what `score(q, d)` should be.

---

## The IR pipeline — step by step

### Step 1 · Query processing

```
raw query:   "Running dogs BARKED"
tokenize:    ["Running", "dogs", "BARKED"]
lowercase:   ["running", "dogs", "barked"]
stopwords:   remove "the", "is", "on" etc. → no change here
stem:        ["run", "dog", "bark"]
```

**Why stem?** "Running", "runs", "ran" all collapse to "run" — they match the same posting list. Without it, "running" and "runs" are treated as completely different terms.

**Why remove stopwords?** "The", "is", "on" appear in nearly every document. They carry no signal and waste space in the index.

---

### Step 2 · Build the inverted index

A normal index: document → words it contains. An inverted index flips it: word → documents that contain it.

```
corpus:
  D1 = "cat sat on mat"
  D2 = "cat wore a hat"
  D3 = "mat and hat"

inverted index:
  "cat" → [D1, D2]
  "sat" → [D1]
  "mat" → [D1, D3]
  "hat" → [D2, D3]
```

Each entry is called a **posting list**. At query time you look up your terms and score across posting lists — never touching documents that contain no query terms. That's the speed trick.

---

### Step 2b · How is the index maintained when docs are added or removed?

This is a real FAANG system design question. The naive approach — rebuild the whole index on every change — is obviously too slow. Real systems use two techniques:

**Adding documents — segment-based indexing (how Lucene/Elasticsearch works):**

```
new docs arrive → written to a small in-memory index (buffer)
buffer fills up → flushed to disk as a new immutable "segment"
segments accumulate → periodically merged into larger segments
```

Each segment is its own mini inverted index. At query time, you search all segments and merge results. This means writes are cheap (just append a new segment) and reads fan out across segments. Lucene calls this a **log-structured merge (LSM)** approach.

**Removing documents — tombstones, not deletion:**

Physically removing a doc from a posting list is expensive — you'd have to rewrite the entire list. Instead:

```
delete D2 → write a tombstone file: {D2: deleted}
query returns D2 → filter it out using the tombstone → user never sees it
segment merge → tombstoned docs are physically dropped during merge
```

So deletes are lazy. The doc disappears from results immediately (tombstone filters it) but the storage is only reclaimed during the next merge. This is the same pattern used in Cassandra, RocksDB, and most LSM-based systems.

**Updating documents:**

An update = delete (tombstone) + add (new segment). There's no in-place edit.

**Why is this cumbersome?** Yes — it is. The tradeoff is write speed vs. read complexity. More segments = more fan-out at query time. Elasticsearch exposes `forcemerge` to collapse segments manually when you want to optimize read latency.

---

### Step 3 · Score

For every document in the relevant posting lists, compute `score(q, d)`:

```
query: "cat mat"
docs:  D1, D2, D3
scores (placeholder — Chapter 2 defines this properly):
  D1 → 0.82   (contains both "cat" and "mat")
  D2 → 0.41   (contains "cat" only)
  D3 → 0.35   (contains "mat" only)
```

### Step 4 · Rank and return

```
ranked: D1 > D2 > D3
return top-k: [D1, D2, D3]
```

---

## Worked numeric example — Boolean retrieval

```
corpus:
  D1 = "deep learning neural networks"
  D2 = "neural networks for NLP"
  D3 = "deep reinforcement learning"
  D4 = "support vector machines"

query: "deep neural"

posting list for "deep":   [D1, D3]
posting list for "neural": [D1, D2]

AND → intersect: {D1, D3} ∩ {D1, D2} = {D1}

result: D1
```

D2 has "neural" but not "deep." D3 has "deep" but not "neural." D4 has neither. Only D1 has both.

**Why Boolean retrieval isn't enough:** If 10,000 docs match, you get 10,000 equally-weighted results with no way to rank them. That's the problem TF-IDF (Chapter 2) solves.

---

## Why it works / why it fails

**Why it works:**
- Inverted index makes lookup O(1) per term regardless of corpus size
- Pre-processing reduces vocabulary noise significantly
- The pipeline is modular — swap the scoring function without touching indexing

**Why it fails:**
- Boolean retrieval has no notion of degree — only match or no match
- Vocabulary mismatch: "car" vs "automobile" — no match despite same meaning
- No word order awareness — "dog bites man" = "man bites dog" to a bag-of-words system
- Short queries are ambiguous — "apple" could be fruit or company

---

## The one thing to remember

IR converts "find me relevant stuff" into a scoring and sorting problem, made fast by a pre-built inverted index. Everything else in IR is a better answer to what `score(q, d)` should be.

---

## Formulas used in this chapter

| Formula | Meaning |
|---------|---------|
| `score(q, d)` | Generic relevance function between query and document — placeholder for TF-IDF, BM25, cosine sim etc. |
| `posting(t)` = [d : t ∈ d] | The posting list for term t — all documents containing t |
| `Boolean AND` = ∩ posting lists | Result set = intersection of posting lists for all query terms |
| `top-k` = argsort(score)[-k:] | Return the k documents with the highest scores |

---

## Interview Q&A

**Q1. Why is the index called "inverted"?**

Because the natural direction is document → words (a document contains these words). The inverted index flips it to word → documents (this word appears in these documents). It's inverted relative to the document-centric view.

**Q2. Is building and maintaining an inverted index cumbersome? What happens when a doc is added or removed?**

Yes, it's non-trivial at scale. Real systems (Lucene, Elasticsearch) use segment-based indexing: new documents are written to small in-memory buffers, flushed to immutable disk segments, and merged periodically. Deletes use tombstones — the doc is flagged as deleted immediately but only physically removed during a segment merge. Updates are delete + re-insert. The tradeoff is that more segments mean more fan-out at query time, which is why Elasticsearch exposes `forcemerge`.

**Q3. Why does the inverted index make retrieval fast even on billion-document corpora?**

Because you never scan all documents. For a query term `t`, you look up `posting(t)` — only the (usually small) fraction of docs containing `t`. Lookup is O(1) (hash or B-tree). Even for common terms, posting lists are compressed and traversed sequentially, which is cache-friendly. The score computation only touches docs in the posting lists, not the full corpus.

**Q4. What is the fundamental limitation of Boolean retrieval that motivates ranked retrieval?**

Boolean retrieval is binary — a document either matches or doesn't. It produces an unordered set. If the result set is large (which it often is for common terms), there's no way to tell the user which documents are most relevant. Ranked retrieval assigns a score to every document and sorts by it, so the most relevant result is always at position 1.

**Q5. Name two things that go wrong when queries and documents use different vocabulary for the same concept.**

First, vocabulary mismatch: a query for "car" won't match a document about "automobiles" even if the content is identical — the posting lists don't overlap. Second, stemming only partially helps (it handles morphological variation like "run/running/ran") but can't bridge true synonymy. This is the core motivation for dense/embedding-based retrieval in Chapter 4, where "car" and "automobile" map to nearby vectors regardless of surface form.

---

\
