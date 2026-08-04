# RAG Interview Prep — Day 12
## Review Day: Retrieval Week (Days 7–11) — Closed Book

---

## 📋 How to run this review

1. **No notes.** Close Days 7–11 (and ideally Days 1–5 too) before starting.
2. Answer out loud or in writing, then expand `<details>` to check.
3. Log anything shaky in the **Weak Spot Tracker** at the bottom — this feeds directly into Day 18's Generation-week review and Day 25's mock interview.
4. Target: 60–90 minutes for all 28 questions.

---

## Section A — Day 7: Sparse Retrieval (BM25 / TF-IDF)

**A1.** What's the key weakness of TF-IDF that BM25's term-frequency saturation fixes?

<details>
<summary>Show answer</summary>
TF-IDF's term frequency component grows roughly linearly with raw occurrence count, with no ceiling — a term appearing 100 times scores proportionally higher than one appearing 10 times. BM25's saturating function (f/(f+k1)) gives diminishing returns past a tunable point, reflecting that excessive repetition stops being a meaningfully stronger relevance signal (and may indicate keyword stuffing).
</details>

**A2 (calculation).** A term appears in 30 of 1500 documents. Compute its IDF.

<details>
<summary>Show answer</summary>

```
IDF = log(1500/30) = log(50) ≈ 3.91
```
</details>

**A3.** Name two scenarios where sparse retrieval still outperforms dense embeddings.

<details>
<summary>Show answer</summary>
Exact identifiers (SKUs, error codes, legal citations) where embeddings compress meaning fuzzily but BM25 matches literally; and rare/out-of-vocabulary terms an embedding model wasn't trained on enough to represent well.
</details>

---

## Section B — Day 8: Dense Retrieval (Bi-Encoders vs. Cross-Encoders)

**B1.** Why can bi-encoder document vectors be precomputed but cross-encoder scores cannot?

<details>
<summary>Show answer</summary>
Bi-encoders encode the query and document independently — a document's vector doesn't depend on any specific query, so it can be computed once, offline. Cross-encoders require the query and document to be present together during a joint forward pass (for self-attention across both), so nothing about the score can be computed before a specific query arrives.
</details>

**B2 (calculation).** A cross-encoder takes 18ms/pair. How long to score 40 candidates vs. 2,000,000 documents?

<details>
<summary>Show answer</summary>

```
40 × 18ms = 720ms (feasible reranking)
2,000,000 × 18ms = 36,000,000ms = 10 hours (infeasible as first-stage retrieval)
```
</details>

**B3.** What's a hard negative, and why does it improve dense retriever training more than in-batch negatives?

<details>
<summary>Show answer</summary>
A hard negative is a document that's superficially similar to the correct answer but actually irrelevant — often surfaced by a weaker retriever or BM25. It forces the model to learn fine-grained distinctions between genuinely relevant and merely similar-looking content, whereas in-batch negatives (other examples' positives in the same training batch) are usually easy/random and teach a coarser distinction.
</details>

---

## Section C — Day 9: Hybrid Search & RRF

**C1.** Why can't BM25 and cosine similarity scores be combined by simple addition?

<details>
<summary>Show answer</summary>
They're on incompatible scales — BM25 is unbounded and corpus/length-dependent, cosine similarity is bounded [-1,1]. Direct addition lets whichever retriever produces numerically larger values dominate, regardless of actual relevance.
</details>

**C2 (calculation).** Using RRF with k=60, a document ranks 3rd in sparse and 1st in dense. Compute its RRF score.

<details>
<summary>Show answer</summary>

```
1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.01587 + 0.01639 = 0.03226
```
</details>

**C3.** Why is RRF generally preferred over min-max score normalization + weighted sum?

<details>
<summary>Show answer</summary>
RRF only uses rank position, sidestepping the scale-incompatibility problem entirely with no normalization needed. Min-max normalization is computed per query using that query's own min/max scores, making it sensitive to outliers within that specific query's result set — RRF has no equivalent fragility.
</details>

---

## Section D — Day 10: Reranking

**D1.** Why does reranking typically move nDCG more than Recall@k?

<details>
<summary>Show answer</summary>
Reranking reorders an already-retrieved candidate set rather than discovering new documents, so Recall@k (which only checks whether relevant docs appear anywhere in top-k) often barely changes. nDCG is position-sensitive and directly rewards moving already-recalled relevant documents into better positions — exactly what reranking does.
</details>

**D2.** What is MaxSim in ColBERT-style late interaction, and why does it allow partial precomputation unlike a cross-encoder?

<details>
<summary>Show answer</summary>
MaxSim computes, for each query token, its maximum similarity against any document token, then sums these across query tokens. Document token embeddings can be precomputed offline (they don't require knowing the query), only the MaxSim comparison happens at query time — unlike a cross-encoder, which requires query and document together during the model's forward pass for joint attention.
</details>

**D3.** Why shouldn't you use a reranker's raw output score as a fixed threshold across different queries?

<details>
<summary>Show answer</summary>
Reranker scores are trained to produce good relative ordering within one query's candidate set, not to be comparable in absolute terms across different queries — a fixed threshold can inconsistently over-filter some queries and under-filter others depending on how strong that particular query's candidate set happened to be.
</details>

---

## Section E — Day 11: Query Transformation

**E1.** What problem does HyDE solve, and why doesn't the hypothetical document's factual accuracy matter?

<details>
<summary>Show answer</summary>
HyDE addresses query-document style asymmetry — queries and real answer documents are structurally different kinds of text. A generated hypothetical answer is stylistically similar to real documents even if factually imperfect; only its embedding (used to search) matters, and it's discarded afterward — never shown to the user or fact-checked.
</details>

**E2.** Why would you decompose a comparative query like "which is faster, X or Y, and which is cheaper" rather than retrieving for it directly?

<details>
<summary>Show answer</summary>
This bundles multiple sub-questions (two products × two attributes); a single embedding would need to match a chunk simultaneously covering all of it, which likely doesn't exist since real documents typically cover one product's specs at a time. Decomposing into individually-answerable sub-questions lets each retrieve against chunks that actually match well.
</details>

**E3.** Why shouldn't query transformation techniques be applied unconditionally to every query?

<details>
<summary>Show answer</summary>
Each technique adds real latency (extra LLM calls and/or extra retrieval calls) — e.g., multi-query generation alone can add several hundred milliseconds. Simple, unambiguous, single-hop queries gain little from these techniques while still paying their full latency cost, so a query-aware routing/triggering strategy is preferable to blanket application.
</details>

---

## Section F — Cross-Week Synthesis (Retrieval × Foundations — the hardest section)

**F1.** A corpus was chunked with fixed-size 100-token chunks and no overlap (Day 3). Retrieval uses hybrid search with RRF (Day 9). Explain a failure mode this combination is especially prone to, and how you'd diagnose it.

<details>
<summary>Show answer</summary>
Fixed-size chunking with no overlap risks splitting a critical sentence or instruction across a chunk boundary. Both sparse (BM25) and dense retrieval would independently struggle with a chunk containing only half of a relevant idea — sparse because key terms might land on the wrong side of the split, dense because the embedding of a half-idea is a poor, ambiguous representation. Since RRF only rewards documents that rank well in at least one retriever, if the same underlying boundary-split problem affects both retrievers similarly (both missing the split chunk), fusion won't rescue it — RRF fuses rankings across retrievers, it doesn't fix an underlying chunk quality problem shared by both. Diagnosis: check whether relevant content spans a chunk boundary in the raw source document for failing queries; a chunk-size/overlap sweep (Day 3) against a golden eval set would be the fix, not further retrieval-side tuning.
</details>

**F2.** Why would switching from a general-purpose embedding model (Day 2) to a domain-specific one potentially reduce your reliance on query transformation techniques (Day 11) like HyDE?

<details>
<summary>Show answer</summary>
HyDE largely exists to bridge a style/vocabulary gap between how queries are phrased and how documents are written, which is partly a symptom of the embedding model's semantic space not representing domain-specific query-answer relationships well. A domain-specific embedding model, trained on in-domain query-document pairs, may already capture that relationship more natively — reducing (though not necessarily eliminating) the gap HyDE is designed to close. This doesn't make query transformation obsolete, but it's a case where a Day 2 fix (better embedding model) and a Day 11 fix (query transformation) address overlapping symptoms from different angles.
</details>

**F3.** You're designing a multi-tenant system (Day 5) with per-tenant HNSW namespaces (Day 4), using cross-encoder reranking (Day 10). A tenant complains their reranked results seem inconsistent day-to-day even for the same query. What Day 5 concept is the likely culprit, separate from anything about the reranker itself?

<details>
<summary>Show answer</summary>
Likely embedding drift or index staleness from ongoing document updates within that tenant's namespace — if new documents are being added/updated continuously (Day 4's incremental update discussion) and the first-stage candidate set changes as a result, the reranker (Day 10) is only reordering whatever candidate set first-stage retrieval hands it — a shifting candidate set naturally produces shifting reranked output, even with a perfectly stable reranker. Worth ruling out before assuming the reranker itself is misbehaving; also worth checking that all replicas in that tenant's namespace are consistent (Day 4/5's eventual consistency discussion) rather than serving from different replica states.
</details>

**F4.** Combine Day 8 (bi-encoder vs. cross-encoder) and Day 11 (multi-query) reasoning: why does applying multi-query generation before a two-stage retrieve-and-rerank pipeline compound latency more than applying it before a single-stage bi-encoder-only pipeline?

<details>
<summary>Show answer</summary>
Multi-query generation multiplies the number of first-stage retrieval calls (N variants → N retrievals). In a single-stage bi-encoder-only pipeline, this only multiplies the (already fast) ANN search cost. In a two-stage pipeline, the N variants' combined/fused candidate set (via RRF, Day 9) still needs to be reranked (Day 10) — and if the fused candidate set ends up larger or the union across variants surfaces more unique candidates than a single-query retrieval would have, the reranking stage (the most expensive part of the pipeline, per Day 10's latency math) now has more candidates to score, compounding the added multi-query latency with additional reranking latency on top — not just the retrieval-side cost alone.
</details>

**F5.** Why does BM25's inverted index (Day 7) conceptually parallel HNSW/IVF's role (Day 4), and what's the key difference in what problem each solves?

<details>
<summary>Show answer</summary>
Both exist to avoid the same fundamental problem: comparing a query against every single item in the corpus (O(N) brute-force). The inverted index avoids scanning every document by pre-mapping terms to document lists; HNSW/IVF avoid scanning every vector by using graph traversal or cluster-narrowing. The key difference is the underlying data type and matching definition — inverted indexes operate on discrete, exact tokens (a term either appears in a document or it doesn't), while HNSW/IVF operate on continuous vector space and approximate geometric closeness — different mechanisms solving the same "don't brute-force scan everything" problem for fundamentally different kinds of matching (exact/discrete vs. semantic/continuous).
</details>

---

## 📊 Weak Spot Tracker

| Question # | Topic | Got it cold? | Needs repair? |
|---|---|---|---|
| A1–A3 | Sparse retrieval (BM25/TF-IDF) | ☐ | ☐ |
| B1–B3 | Dense retrieval (bi/cross-encoder) | ☐ | ☐ |
| C1–C3 | Hybrid search / RRF | ☐ | ☐ |
| D1–D3 | Reranking | ☐ | ☐ |
| E1–E3 | Query transformation | ☐ | ☐ |
| F1–F5 | Cross-week synthesis | ☐ | ☐ |

**Rule carried over from Day 6:** synthesis-question misses (Section F) matter more than single-topic recall misses — they're what a real interview conversation actually probes, since interviewers build follow-ups on your first answer rather than asking isolated definitions back to back.

---

*Retrieval week complete. Next up — Day 13: Context Construction & Lost-in-the-Middle, starting the Generation week.*
