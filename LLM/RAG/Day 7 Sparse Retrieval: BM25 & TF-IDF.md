# RAG Interview Prep — Day 7
## Sparse Retrieval: BM25 & TF-IDF

---

## 🚀 Quick Summary

Before embeddings existed, and still very much alongside them today, **sparse retrieval** finds relevant documents by matching actual keywords/terms between the query and the corpus — "sparse" because each document's representation is a huge vector where almost every entry is zero (one entry per possible word in the vocabulary, mostly unused). TF-IDF was the classical scoring function; **BM25** is its modern, more robust successor and remains the single most common baseline (and frequent production component) in real retrieval systems, including inside most RAG pipelines via hybrid search (Day 9). Today is about exactly how these scores are computed, why BM25 fixes TF-IDF's weaknesses, and — critically for interviews — *when keyword search still beats embeddings*, which is a very common trap question ("why not just use embeddings for everything?").

**Think of it like judging a book by how often and how uniquely it uses your search words.** If you search "diabetes treatment," a document that mentions "diabetes" fifty times is probably relevant — but if "the" appears fifty times too, that tells you nothing, because "the" is everywhere. TF-IDF and BM25 are both formalizations of the same intuition: reward documents where your search terms appear often, but *discount* terms that are common across the whole corpus (since a common word being present isn't informative), and — BM25's key addition — stop over-rewarding a term once it's already appeared "enough" times, and correct for document length so a long document doesn't win purely by being long.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Sparse retrieval** | Retrieval based on exact term/keyword matching, represented as high-dimensional mostly-zero vectors |
| **Dense retrieval** | Retrieval based on learned embedding similarity (Day 2/8) — the opposite of sparse |
| **Term Frequency (TF)** | How often a term appears within a specific document |
| **Inverse Document Frequency (IDF)** | A measure of how rare/informative a term is across the whole corpus |
| **TF-IDF** | A scoring function combining TF and IDF — classic sparse retrieval baseline |
| **BM25** | "Best Matching 25" — an improved, saturating, length-normalized version of TF-IDF; the modern sparse retrieval standard |
| **Term saturation** | The property that additional occurrences of a term contribute diminishing score, rather than linearly forever |
| **Inverted index** | A data structure mapping each term to the list of documents containing it — what makes sparse retrieval fast |

---

# PHASE 1 — Intuition & Visual Map

## Sparse vs. Dense — the fundamental contrast

```
  SPARSE (BM25/TF-IDF)                    DENSE (embeddings, Day 2)

  "Apple ID password reset"               "Apple ID password reset"
         │                                        │
         ▼                                        ▼
  vector over the ENTIRE vocabulary        vector over a LEARNED
  (e.g. 50,000 dims), almost all           semantic space (e.g. 768 dims),
  zero, nonzero only at the exact          every dimension carries some
  words that appear                        distributed meaning signal

  "password" and "passcode" are            "password" and "passcode" can
  COMPLETELY unrelated (different          be CLOSE in vector space,
  vocabulary slots, zero overlap)          because the model learned
                                            they're semantically similar
```

**The core trade-off in one line:** sparse retrieval is *exact* but *literal* (it has zero notion that "cheap" and "affordable" mean the same thing); dense retrieval is *semantic* but *fuzzy* (it can miss an exact product code or rare acronym because it's reasoning in a compressed, generalized space that wasn't built to preserve exact tokens).

## Why sparse retrieval still matters in a RAG interview

A very common trap: assuming embeddings have made sparse retrieval obsolete. In practice, **keyword search still wins in specific, well-known scenarios**:
- ✅ **Exact identifiers** — product SKUs, error codes, legal citation numbers, part numbers. Embeddings compress these into a fuzzy semantic neighborhood; BM25 matches them exactly.
- ✅ **Rare or out-of-vocabulary terms** — a brand-new technical term, a specific person's name, an uncommon acronym the embedding model never saw enough of during training to represent well.
- ✅ **Queries where exact phrasing matters** — legal/compliance search where a specific word choice has legal significance.
- ✅ **Low-resource or highly technical domains** where general embedding models weren't trained on enough in-domain text to build a good semantic map (tie-in to Day 2's domain-specific embedding discussion).

This is exactly why hybrid search (Day 9) — combining sparse and dense — is the production default in most serious systems, rather than picking one or the other.

---

# PHASE 2 — Math & Formulas

## Notation table

| Symbol | Meaning |
|---|---|
| `t` | A specific term (word) |
| `d` | A specific document |
| `D` | The full document corpus |
| `f(t, d)` | Raw count of how many times term `t` appears in document `d` |
| `N` | Total number of documents in the corpus |
| `n(t)` | Number of documents containing term `t` at least once |
| `\|d\|` | Length of document `d` (number of terms) |
| `avgdl` | Average document length across the whole corpus |
| `k1`, `b` | BM25 tunable hyperparameters |

---

### 1. Term Frequency — Inverse Document Frequency (TF-IDF)

```
TF-IDF(t, d) = TF(t,d) × IDF(t)

TF(t, d) = f(t,d) / |d|              (a common normalization variant)

IDF(t) = log( N / n(t) )
```

**Plain English:**
- **TF** — how prevalent is this term *within this specific document*, relative to the document's length? A term appearing 5 times in a 50-word document is more prevalent than the same 5 occurrences in a 5,000-word document.
- **IDF** — how rare is this term *across the whole corpus*? A term that appears in almost every document (like "the") gets an IDF close to `log(1) = 0` — essentially worthless as a distinguishing signal. A term that appears in only a few documents out of many gets a high IDF — very informative when it does show up.
- **TF-IDF** — multiply them together: reward a term for being locally frequent, but only to the extent that it's also globally rare/informative.

**Worked numerical example:**

Corpus of `N = 1000` documents. Query term is **"diabetes"**.
```
"diabetes" appears in n(t) = 20 documents out of 1000
IDF("diabetes") = log(1000/20) = log(50) ≈ 3.91
```
Now compare two candidate documents:
```
Document A: "diabetes" appears 8 times, |d| = 200 words
  TF(A) = 8/200 = 0.04
  TF-IDF(A) = 0.04 × 3.91 ≈ 0.1564

Document B: "diabetes" appears 3 times, |d| = 60 words
  TF(B) = 3/60 = 0.05
  TF-IDF(B) = 0.05 × 3.91 ≈ 0.1955
```
Despite Document A mentioning "diabetes" more times in absolute terms (8 vs. 3), Document B scores higher — because "diabetes" makes up a larger *proportion* of Document B's (shorter) content, and TF-IDF is measuring concentration/prevalence, not raw count.

**Why it matters in practice:** TF-IDF was the workhorse of search engines for decades, and its two core ideas — reward local frequency, penalize global commonness — remain the conceptual foundation every sparse retrieval method (including BM25) builds on. But it has real weaknesses (next section) that motivated BM25's development.

---

### 2. BM25 — The Modern Standard

```
BM25(t, d) = IDF(t) × [ f(t,d) × (k1 + 1) ] / [ f(t,d) + k1 × (1 - b + b × |d|/avgdl) ]
```
(Summed across all query terms to get a document's total BM25 score for a multi-word query.)

**Plain English:** Same core spirit as TF-IDF (reward local frequency, weight by global rarity), but with two crucial fixes: **term frequency saturation** (more occurrences help less and less, rather than linearly forever) and **explicit, tunable document-length normalization** (rather than TF-IDF's simple division-by-length).

**Term-by-term breakdown:**
- `IDF(t)` — same rarity concept as before (BM25's actual IDF formula has a small smoothing variant, but the core idea is identical to TF-IDF's).
- `f(t,d)` — raw term frequency, same as TF-IDF's numerator.
- `k1` (typically ~1.2–2.0) — controls how quickly term frequency **saturates**. Higher `k1` means additional occurrences of the term keep contributing more before leveling off; lower `k1` means the score plateaus faster after just a few occurrences.
- `b` (typically ~0.75, range 0–1) — controls how strongly document length is penalized. `b=1` means full length normalization (long documents are fully penalized for their length); `b=0` means no length normalization at all (raw term frequency counts fully, regardless of document length).
- `|d|/avgdl` — this document's length relative to the corpus average — the mechanism through which `b` actually adjusts the score based on whether this document is longer or shorter than typical.

**The saturation effect, illustrated numerically:**

Using `k1 = 1.5` (fixing IDF and length normalization aside for a moment, focusing purely on how the `f(t,d)` term behaves inside the saturating fraction `f/(f+k1)`):
```
f(t,d) = 1:   1/(1+1.5) = 0.400
f(t,d) = 3:   3/(3+1.5) = 0.667
f(t,d) = 10:  10/(10+1.5) = 0.870
f(t,d) = 30:  30/(30+1.5) = 0.952
f(t,d) = 100: 100/(100+1.5) = 0.985
```
Going from 1 → 3 occurrences jumps the score contribution by +0.267. Going from 30 → 100 occurrences (70 more occurrences!) only adds +0.033. **This is term frequency saturation** — BM25 explicitly encodes the intuition that a term appearing 100 times isn't meaningfully "more relevant" than one appearing 30 times; after a point, more repetition stops being informative (and might even suggest keyword-stuffing rather than genuine relevance). Plain TF-IDF's raw `f(t,d)/|d|` has no such ceiling — it keeps rewarding raw frequency linearly (modulo the length division), which BM25 explicitly fixes.

**Full worked BM25 example:**

Let `k1 = 1.5`, `b = 0.75`, `avgdl = 250` words (corpus average document length).

Query term: **"battery"**. `IDF("battery") = 2.8` (given, for this example).

**Document X:** `f("battery", X) = 4`, `|X| = 150` words (shorter than average)
```
length ratio = |X|/avgdl = 150/250 = 0.6
denominator adjustment = k1 × (1 - b + b × 0.6)
                        = 1.5 × (1 - 0.75 + 0.75×0.6)
                        = 1.5 × (0.25 + 0.45)
                        = 1.5 × 0.70
                        = 1.05

BM25(X) = 2.8 × [4 × 2.5] / [4 + 1.05]
        = 2.8 × 10 / 5.05
        = 2.8 × 1.980
        ≈ 5.545
```

**Document Y:** `f("battery", Y) = 4`, `|Y| = 500` words (twice the average length)
```
length ratio = 500/250 = 2.0
denominator adjustment = 1.5 × (1 - 0.75 + 0.75×2.0)
                        = 1.5 × (0.25 + 1.5)
                        = 1.5 × 1.75
                        = 2.625

BM25(Y) = 2.8 × [4 × 2.5] / [4 + 2.625]
        = 2.8 × 10 / 6.625
        = 2.8 × 1.509
        ≈ 4.226
```

**Interpretation:** Both documents mention "battery" exactly 4 times — identical raw term frequency — but Document X (shorter, 150 words) scores meaningfully higher than Document Y (longer, 500 words): **5.545 vs. 4.226**. This is BM25's length normalization at work: 4 mentions in a short document is a much stronger relevance signal (the term makes up more of the document's content) than 4 mentions in a document more than 3x longer, where those same 4 mentions are comparatively diluted. TF-IDF's simpler length handling captures a version of this too, but BM25's `b` parameter makes the *strength* of this normalization explicitly tunable.

### What happens as k1 and b change

| Parameter | Low value effect | High value effect |
|---|---|---|
| `k1` (saturation point) | Score saturates quickly — even 2-3 occurrences nearly maxes out the term-frequency contribution | Score keeps rewarding additional occurrences for longer before saturating |
| `b` (length normalization strength) | `b=0`: raw term frequency counts fully, no penalty for document length at all | `b=1`: full length normalization — score is heavily adjusted for how the document's length compares to the corpus average |

---

## TF-IDF vs. BM25 — Comparison Table (frequently asked directly)

| | TF-IDF | BM25 |
|---|---|---|
| **Term frequency handling** | Roughly linear (more occurrences → proportionally more score, modulo length division) | Saturating — diminishing returns past a tunable point (`k1`) |
| **Length normalization** | Simple division by document length | Explicit, tunable normalization relative to *corpus average* length (`b`) |
| **Tunability** | Minimal — mostly a fixed formula | Two interpretable hyperparameters (`k1`, `b`) that can be tuned per corpus/use case |
| **Modern usage** | Largely superseded, but conceptually foundational | The current standard sparse retrieval baseline (default in Elasticsearch, Lucene, most search infra) |
| **Robustness to keyword stuffing** | Weaker — doesn't cap the benefit of repeating a term many times | Stronger — saturation explicitly limits the reward for excessive repetition |

---

## The Inverted Index — Why Sparse Retrieval Is Fast

**Mechanism:** Instead of scanning every document to check if it contains a query term (which would be `O(N)` per term, similar to Day 4's brute-force vector search problem), an inverted index pre-computes, for every term in the vocabulary, the **list of document IDs that contain it** (often with the term frequency and position stored alongside).

```
   Term "battery"  →  [doc_12, doc_45, doc_203, doc_890, ...]
   Term "password" →  [doc_3, doc_45, doc_501, ...]
   Term "reset"    →  [doc_3, doc_45, doc_200, doc_890, ...]
```

At query time, for a multi-term query, the system looks up each query term's posting list directly (fast, no scanning) and combines/scores only the documents that appear in at least one relevant list — dramatically narrowing the search space compared to checking every document in the corpus, analogous in spirit to how ANN indexes (Day 4) avoid brute-force comparison for dense vectors, just via an entirely different mechanism suited to discrete tokens rather than continuous vectors.

---

## Practical Implementation Notes

- **Tokenization for sparse retrieval matters a lot** — typically involves lowercasing, removing stopwords ("the," "a," "is" — since these have near-zero IDF and add noise/cost with no benefit), and often **stemming/lemmatization** (reducing "running," "runs," "ran" to a common root "run") so that different word forms of the same concept still match.
- **Common production tools:** Elasticsearch and Apache Lucene both implement BM25 as their default scoring function — this is worth knowing by name, since "how would you implement sparse retrieval in production" often expects "BM25 via Elasticsearch/Lucene," not a from-scratch implementation.
- **BM25 is stateless per-query relative to the corpus statistics** (`avgdl`, `IDF` values) — but those corpus-level statistics need to be recomputed or incrementally updated as the corpus changes, similar in spirit to IVF's centroid staleness problem (Day 4), though generally cheaper to refresh than a full re-clustering.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** In one sentence, what's the fundamental difference between sparse and dense retrieval?

<details>
<summary>Show answer</summary>

Sparse retrieval matches based on exact term/keyword overlap (high-dimensional, mostly-zero vectors over the vocabulary), while dense retrieval matches based on learned semantic similarity in a compressed embedding space where related-but-different words can be close together even with zero literal overlap.
</details>

---

**Q2 (Easy — calculation).** A term appears in 50 documents out of a corpus of 2000. Compute its IDF.

<details>
<summary>Show answer</summary>

```
IDF = log(2000/50) = log(40) ≈ 3.69
```
</details>

---

**Q3 (Medium — conceptual).** Name three specific scenarios where keyword-based (sparse) search still outperforms dense embedding search, and explain why in each case.

<details>
<summary>Show answer</summary>

(1) Exact identifiers (product SKUs, error codes, legal citation numbers) — embeddings compress these into a fuzzy semantic neighborhood and can't guarantee exact matching, while BM25 matches them precisely. (2) Rare or out-of-vocabulary terms (uncommon acronyms, brand-new technical terms, specific names) — an embedding model may not have seen enough examples during training to represent these well in vector space, while sparse retrieval just needs the literal token to appear. (3) Queries where exact phrasing carries legal/compliance significance — a specific word choice can matter in ways a "close enough" semantic match would miss.
</details>

---

**Q4 (Medium — conceptual).** What specific problem does BM25's term-frequency saturation solve that plain TF-IDF doesn't handle well?

<details>
<summary>Show answer</summary>

Plain TF-IDF's term frequency component grows roughly linearly (modulo length normalization) with raw occurrence count — a document mentioning a term 100 times scores proportionally higher than one mentioning it 10 times, with no ceiling. BM25 introduces a saturating function (`f/(f+k1)`) where additional occurrences contribute diminishing score gains past a tunable point — reflecting the intuition that after enough repetitions, more occurrences stop being a meaningfully stronger relevance signal (and may even indicate keyword stuffing rather than genuine topical relevance).
</details>

---

**Q5 (Medium — calculation).** Using `k1=1.2`, compare the saturation contribution `f/(f+k1)` at `f=2` and `f=20`. What does the gap between them demonstrate?

<details>
<summary>Show answer</summary>

```
f=2:  2/(2+1.2) = 2/3.2 = 0.625
f=20: 20/(20+1.2) = 20/21.2 ≈ 0.943
```
Going from 2 to 20 occurrences (18 more) only increases the contribution by about 0.318, and the curve is clearly flattening — demonstrating diminishing returns from additional term occurrences, which is exactly the saturation behavior BM25 is designed to produce.
</details>

---

**Q6 (Hard — conceptual + parameter reasoning).** Two documents both mention "warranty" exactly 5 times. Document A is 100 words long; Document B is 800 words long (well above the corpus average of ~250 words). Without doing the full calculation, explain qualitatively how BM25's `b` parameter would affect their relative scores, and what setting `b=0` would do differently.

<details>
<summary>Show answer</summary>

With a typical `b` (e.g., 0.75), Document A (short, 100 words) would score meaningfully higher than Document B (long, 800 words) for the same raw term frequency, because BM25's length normalization penalizes documents that are longer relative to the corpus average — 5 mentions in a short document is a stronger concentration/relevance signal than 5 mentions diluted across a much longer document. If `b` were set to 0, length normalization would be disabled entirely — both documents would be scored based on raw term frequency and IDF alone, with no adjustment for the fact that Document B is over 3x longer than the corpus average, likely making their scores much closer (or even causing the longer document to not be penalized despite its lower term concentration).
</details>

---

**Q7 (Hard — system design / synthesis).** You're building the retrieval layer for a technical support RAG system covering a mix of general product FAQs and a large database of specific error codes (e.g., "ERR-4471"). A pure dense-embedding retrieval system is underperforming on error-code queries specifically. Diagnose why, and propose a fix.

<details>
<summary>Show answer</summary>

Error codes like "ERR-4471" are exact identifiers with essentially no semantic content for an embedding model to generalize from — the model has no meaningful notion of what makes "ERR-4471" similar or different from "ERR-4472" beyond surface token similarity, and a dense embedding space built for general semantic meaning isn't optimized to preserve this kind of exact-match distinction reliably. This is a textbook case for sparse retrieval: BM25 (via an inverted index) would match the literal "ERR-4471" token directly and reliably, with no fuzziness. The fix is to add a sparse retrieval path (BM25) alongside the existing dense retrieval and combine them via hybrid search (Day 9) — likely with a mechanism that lets exact-match sparse hits on high-value tokens like error codes take priority, rather than relying on dense embeddings alone for a query type it's structurally poorly suited for.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Assuming embeddings have made keyword search obsolete — exact identifiers, rare terms, and phrasing-sensitive domains are real, common cases where sparse still wins.
- ❌ Confusing TF-IDF's linear term-frequency reward with BM25's saturating one — this distinction is asked about directly and often.
- ❌ Forgetting that BM25's length normalization (`b`) is relative to the *corpus average* document length, not an absolute threshold.
- ❌ Treating `k1` and `b` as fixed constants rather than tunable hyperparameters worth adjusting per corpus (e.g., a corpus of very short documents vs. very long ones may want different `b`).
- ❌ Not knowing that Elasticsearch/Lucene implement BM25 by default — a very commonly expected practical detail.
- ❌ Ignoring stopword removal and stemming as part of the sparse retrieval pipeline, treating "raw tokens" as sufficient.

---

# 📌 Cheat Sheet (Day 7)

**TF-IDF:** `TF(t,d) × IDF(t)`, where `IDF(t) = log(N/n(t))`. Rewards local frequency, discounts globally common terms. Weakness: roughly linear term-frequency reward, simple length handling.

**BM25:** `IDF(t) × [f(t,d)×(k1+1)] / [f(t,d) + k1×(1-b+b×|d|/avgdl)]`. Fixes TF-IDF via (1) **saturation** — `k1` controls how fast additional term occurrences stop mattering, and (2) **tunable length normalization** — `b` controls how strongly a document's length (relative to corpus average) penalizes its score. Typical defaults: `k1 ≈ 1.2–2.0`, `b ≈ 0.75`.

**When sparse wins over dense:** exact identifiers (SKUs, error codes, citations), rare/out-of-vocabulary terms, phrasing-sensitive domains, low-resource specialized corpora.

**Infrastructure:** inverted index (term → document list) makes sparse retrieval fast, analogous in purpose (not mechanism) to Day 4's ANN indexes for dense vectors. Elasticsearch/Lucene = BM25 by default in production.

---

*End of Day 7. Next up — Day 8: Dense Retrieval (bi-encoders vs. cross-encoders).*
