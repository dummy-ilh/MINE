# Chapter 17 — Multi-Source / Federated Retrieval

## What is it?

Every chapter so far has assumed a single, homogeneous corpus: one index, one embedding space, one document type. Real search systems never look like this. A single query to Siri or Spotlight might need to draw from web documents, app content, structured product data, contacts, calendar events, and on-device files — sources with completely different schemas, update frequencies, and even different retrieval mechanisms (some are dense-retrieved, some are pure database lookups). The Apple JD names this directly: **"combine information from multiple structured and unstructured sources."**

**Multi-source (or federated) retrieval** is the architecture and set of techniques for querying several heterogeneous sources in parallel and merging their results into one coherent ranked list — despite the sources having incomparable score scales, different latencies, and different relevance semantics.

---

## The intuition

**Observation 1 — different information lives in fundamentally different shapes.** "What's the weather tomorrow" needs a structured API call with an exact numeric answer. "Best pizza place near me" needs a geo-filtered structured database of businesses. "How does photosynthesis work" needs unstructured document retrieval. One retrieval architecture cannot efficiently serve all three — this is why Query Understanding (Chapter 16)'s intent classification exists: it decides *which* source(s) a query should even be sent to.

**Observation 2 — even when you know which sources to query, their scores are not comparable.** A BM25 score of 8.2 from a document index and a cosine similarity of 0.71 from a dense product-embedding index and a hand-tuned relevance heuristic from a structured contacts lookup are three numbers on three completely different scales with no inherent meaning relative to each other. You cannot simply sort all candidates by "raw score" across sources — you need a principled way to make them comparable.

**Observation 3 — the merging problem is really a ranking problem on top of ranking problems.** Each source already internally ranks its own candidates well (that's Chapters 3-9's job). Federation adds a *second*, separate ranking layer whose only job is: given several already-good-internally rankings from different sources, produce one final good ranking across all of them.

---

## The merging strategies

### 1. Score normalization

Convert each source's raw scores into a common scale before comparing, most commonly via min-max normalization per source, per query:

```
normalized_score(d) = (raw_score(d) − min_score_in_source) / (max_score_in_source − min_score_in_source)
```

This forces every source's results into [0, 1], but it's fragile: it assumes each source's min/max for *this specific query* is meaningful, and it says nothing about whether "0.9 from source A" is actually as relevant as "0.9 from source B" — it only guarantees within-source ordering is preserved.

### 2. Reciprocal Rank Fusion (RRF)

Instead of trying to make raw scores comparable at all, throw away the scores and use only each source's **rank order**, then combine:

```
RRF_score(d) = Σ_{source s where d appears} 1 / (k + rank_s(d))
```

where `k` is a constant (commonly 60, empirically found to work well across many settings) that dampens the impact of very top ranks so a #1 result doesn't dominate too extremely, and `rank_s(d)` is the document's position (1-indexed) in source `s`'s ranked list.

This is the most widely used production technique precisely because it **sidesteps the score-comparability problem entirely** — it never touches the underlying scores, only the ordinal position, which every source can produce regardless of how internally different their scoring mechanisms are.

### 3. Learned fusion (a second-stage LTR model)

Treat the outputs of each source as *features* into a Learning-to-Rank model (Chapter 8): `rank_in_source_A`, `score_in_source_A` (normalized), `rank_in_source_B`, source-type indicator, freshness of the item, and so on — then train a model (e.g., LambdaMART or a small neural ranker) to predict final relevance directly from these cross-source features, using real click/relevance labels.

This is the most accurate approach at scale (it can learn, e.g., that structured local-business results should be boosted for queries with geo-intent, in a way neither RRF nor score normalization can express), but it requires labeled training data spanning the merged, federated results — which RRF does not.

---

## Worked numeric example — Reciprocal Rank Fusion

```
Query: "jaguar"

Source A (Web document index) top-5, by rank:
  rank 1: doc_car_review    (raw BM25 score: 14.2)
  rank 2: doc_animal_wiki   (raw BM25 score: 11.8)
  rank 3: doc_car_history   (raw BM25 score: 9.5)

Source B (Structured local-business index) top-5, by rank:
  rank 1: store_jaguar_dealership   (raw relevance heuristic: 0.95)
  rank 2: doc_animal_wiki           (does NOT appear — wrong source type)
```

Note `doc_animal_wiki` only appears in Source A; `store_jaguar_dealership` only appears in Source B; nothing appears in both here except for illustration, let's also add one overlapping doc:

```
Source A rank 4: store_jaguar_dealership  (BM25 score: 3.1 — weakly matched as a web page)

Using k = 60:

RRF(doc_car_review)         = 1/(60+1)                = 0.01639
RRF(doc_animal_wiki)        = 1/(60+2)                = 0.01613
RRF(doc_car_history)        = 1/(60+3)                = 0.01587
RRF(store_jaguar_dealership)= 1/(60+1) + 1/(60+4)      = 0.01639 + 0.01563 = 0.03202
```

**Final fused ranking (sorted by RRF score, descending):**

```
1. store_jaguar_dealership  (0.03202)  ← appeared in BOTH sources, even at a weak rank-4 in one
2. doc_car_review           (0.01639)
3. doc_animal_wiki          (0.01613)
4. doc_car_history           (0.01587)
```

**The key mechanism to notice:** `store_jaguar_dealership` wasn't even Source A's top result (it was rank 4, a fairly weak BM25 match), and its raw structured-relevance score (0.95) from Source B is on a completely different, incomparable scale to BM25's 14.2. Yet RRF correctly promotes it to the #1 fused position — purely because it's the *only* document that shows up as reasonably-ranked in **both independent sources**, which is a strong cross-validated signal of true relevance that no single source alone could have produced.

---

## Why it works / why it fails

**Why it works:**
- RRF requires no score calibration between sources — it works even when sources use fundamentally incompatible scoring mechanisms (BM25 vs. cosine similarity vs. hand-tuned business relevance heuristics vs. exact structured-field matches), because it only ever looks at rank position, which is universally defined for any ranked list.
- Agreement across independent sources is a strong relevance signal precisely because each source's ranking errors are largely uncorrelated — a document ranked well by two structurally different retrieval mechanisms is far less likely to be a false positive than one ranked well by only one.
- Learned fusion (LTR-based) can go further and model genuine source-specific value (e.g., "for geo-intent queries, boost structured local results") that a purely rank-based method like RRF has no way to express, since RRF is intent-agnostic by construction.

**Why it fails / risks:**
- **RRF ignores score magnitude entirely**, which can be a real loss of information: a document with an overwhelming BM25 score of 40 vs. the next-best at 12 is treated identically to one that barely edged out its rank-2 neighbor (14.2 vs 14.1) — both just contribute `1/(k+1)`. In domains where score magnitude genuinely reflects confidence gaps, RRF discards that.
- **The `k` constant in RRF is a global hyperparameter, not adaptive per query or per source** — it implicitly assumes all sources' rank positions carry roughly comparable meaning, which may not hold if one source's "top 10" are all excellent and another's "top 10" are mediocre by rank 3.
- **Learned fusion requires labeled data spanning federated results**, which is expensive and slow to collect (you need judgments on the *merged* output, not just each source independently), and the model can go stale as sources evolve or new sources are added, requiring retraining.
- **Latency is bounded by the slowest source queried in parallel.** Federating N sources means the end-to-end latency is at best the max of all N sources' individual latencies (if truly parallel) — a single slow structured backend (e.g., a live API call) can become the bottleneck for the entire federated response, which is why production systems often set hard per-source timeouts and degrade gracefully (return without that source) rather than block indefinitely.

---

## The one thing to remember

Federated retrieval's hardest problem isn't querying multiple sources — it's making their results *comparable* despite incompatible scoring mechanisms, and rank-based fusion (RRF) solves this by discarding scores entirely and trusting cross-source rank agreement as the relevance signal, while learned fusion trades that simplicity for the ability to model genuine source-specific value at the cost of needing federated training labels.

---

## Formulas used in this chapter

| Formula | Meaning |
|---|---|
| `normalized_score(d) = (raw − min) / (max − min)` | Min-max score normalization within a single source, per query |
| `RRF_score(d) = Σₛ 1/(k + rank_s(d))` | Reciprocal Rank Fusion — combine ranks across sources without needing comparable scores |

---

## Interview Q&A

**Q1. Why is Reciprocal Rank Fusion preferred over score normalization in most production federated search systems?**

Score normalization (e.g., min-max) still relies on the assumption that after rescaling, a normalized score of 0.9 means roughly the same thing across different sources — but this is rarely true, since it's only guaranteed to preserve *within-source* relative ordering, not cross-source comparability of magnitude. RRF sidesteps the entire problem by using only rank position, which is universally meaningful across any ranked list regardless of how the underlying scores were computed — a BM25-ranked list and a structured-relevance-heuristic-ranked list both produce well-defined rank-1, rank-2, etc., even though their raw scores are completely incomparable. This makes RRF far more robust to adding new, unpredictably-scored sources without recalibration.

**Q2. In the worked RRF example, why did the structured local-business result end up ranked #1 in the fused list even though it wasn't the top result in either individual source?**

It appeared in both sources (rank 1 in the structured business index, rank 4 in the web document index), and RRF sums contributions across every source a document appears in. Even though its rank-4 appearance in the web index was a relatively weak individual signal, the fact that two independent, structurally different retrieval mechanisms both surfaced it as relevant is a stronger combined signal than any single source's top-1 result that only one source found. This cross-source agreement effect is the core mechanism that makes RRF work well in practice — it rewards consensus across independently-erring sources.

**Q3. What's the main limitation of RRF, and when would you reach for a learned fusion model instead?**

RRF completely discards score magnitude — it treats a document that overwhelmingly won its source's ranking the same as one that barely edged out its next competitor, as long as both are rank 1. It's also intent-agnostic: it has no way to learn that, say, geo-intent queries should systematically favor structured local-business results over general web documents. When you have (or can collect) labeled relevance judgments on the federated, merged output, and you need the fusion to be sensitive to query intent or source-specific value beyond simple rank agreement, a learned fusion model (LTR over cross-source features: rank, normalized score, source type, freshness, geo-relevance, etc.) can capture those patterns explicitly, at the cost of needing that federated training data and ongoing retraining as sources evolve.

**Q4. How would you handle a structured source (e.g., a live weather API) that's occasionally slow, in a federated retrieval system with a strict end-to-end latency budget?**

Since federated retrieval typically queries sources in parallel, end-to-end latency is bounded by the slowest source in the critical path — an occasionally slow API call can become the bottleneck for the entire response if the system waits unconditionally. The standard mitigation is to set a hard per-source timeout and degrade gracefully: if the structured source doesn't respond within its budget, proceed with fusion over whatever sources did respond in time, rather than blocking the whole query. This trades a small chance of missing that source's contribution on slow requests for a guaranteed latency bound on every request — usually the right trade-off in a user-facing system where a late result is often worse than a slightly incomplete one.

**Q5. Why can't you just always send every query to every available source and let fusion sort it out, instead of relying on query understanding/intent classification (Chapter 16) to decide routing?**

You technically could, but it's costly and often lower quality in practice: querying every source for every query means paying the latency and compute cost of the slowest/most-expensive source (e.g., a live structured API) even for queries that source is irrelevant to (e.g., hitting a contacts lookup for "how does gradient descent work"), and it increases the chance of low-quality or nonsensical results from an irrelevant source diluting the fused ranking, especially under score-agnostic fusion like RRF where an irrelevant source's rank-1 result still gets a non-trivial RRF contribution. Intent classification lets the system selectively route to only the sources plausibly relevant to the query, improving both latency and precision, with federation and fusion still handling the (usually smaller) set of genuinely relevant multi-source cases.
