# RAG Module 5 — Reranking

---

## 5.1 Why a second-stage reranker exists

Recall Module 1.3's bi-encoder vs cross-encoder tradeoff: bi-encoders are fast (precomputed doc embeddings, O(1) ANN lookup) but lossy — no cross-attention between query and doc tokens means the model can't capture fine-grained interactions (a document that shares topic-level similarity but misses the actual answer still scores highly).

**The two-stage funnel pattern**:
```
Full corpus (millions) 
  → Stage 1: bi-encoder + ANN search → top-k (e.g. k=50-100), fast, cheap, high recall target
  → Stage 2: cross-encoder reranker → top-n (e.g. n=5-10), slow, expensive, high precision target
  → generation
```

This is the same pattern as IVF-PQ's "approximate shortlist, then exact re-score" from Module 3.2 — a recurring architectural motif in RAG: **cheap broad recall, then expensive narrow precision**, applied at every layer (indexing, retrieval, reranking).

**Why not skip stage 1 and rerank everything**: cross-encoders require a full model forward pass *per (query, doc) pair* — no precomputation possible since query and doc are jointly encoded. Running this over a full corpus of millions of documents per query is computationally infeasible at interactive latency. Reranking only makes sense on an already-small candidate set.

---

## 5.2 Cross-encoder rerankers

Concatenate `[CLS] query [SEP] document [SEP]`, pass through a transformer, take a single relevance score from the output (often a linear layer on top of the `[CLS]` token or pooled output).

**Why more accurate than bi-encoders**: full self-attention across *all* query and document tokens jointly — the model can directly attend from a query token to a specific document token and vice versa, capturing fine-grained lexical/semantic interactions that bi-encoders lose by compressing query and document into independent fixed vectors before any comparison happens.

**Cost**: O(n) forward passes for n candidates, no caching/precomputation possible (document representation is query-dependent — you can't precompute "the" embedding for a document since its representation *changes* depending on what query it's paired with).

---

## 5.3 Late-interaction models (ColBERT) — the middle ground

Recall from Module 1.3: ColBERT encodes query and document tokens **separately** (like a bi-encoder — document token embeddings are precomputable and stored in the index) but at query time computes a **fine-grained token-to-token interaction** via the **MaxSim** operator:

```
score(Q,D) = Σ_{q ∈ Q} max_{d ∈ D} (q · d)
```

For every query token, find its single most-similar document token (max over all doc tokens), sum these max-similarities across all query tokens.

**Why this is a genuine middle ground, not just "cheaper cross-encoder"**:
- Document token embeddings are precomputed and stored (like a bi-encoder) — no query-time document encoding needed
- But relevance scoring still involves token-level interaction (like a cross-encoder) rather than a single compressed vector comparison — captures much of the cross-encoder's precision advantage
- **Cost**: higher than a pure bi-encoder (must store token-level embeddings for every document, not just one pooled vector — larger index; MaxSim computation at query time is more expensive than a single dot product) but far cheaper than a true cross-encoder (no joint forward pass required per candidate)

**When to reach for it**: when reranking latency/cost budget is tight but bi-encoder-only precision isn't good enough — ColBERT-style late interaction is a strong "first-stage retrieval with reranking-level quality baked in" option, sometimes used *instead of* a separate rerank stage rather than *alongside* one.

---

## 5.4 Production rerankers

- **Cohere Rerank** — API-based cross-encoder reranker, common production default because it requires no self-hosting/fine-tuning, just an API call with (query, candidate list) → returns reordered list with scores
- **Open cross-encoder rerankers** (e.g. `ms-marco-MiniLM`-family, BGE-reranker) — self-hostable, trained on large-scale relevance-labeled data (MS MARCO is the standard training/benchmark set for rerankers, worth naming), fine-tunable on domain-specific relevance data the same way embedders are (Module 1.7)

**Practical usage pattern**: retrieve top-k (e.g. 50) via hybrid retrieval (Module 4) → send all 50 (query, doc) pairs to the reranker in a batch → take top-n (e.g. 5) by reranker score → feed to generation. The reranker call is usually the single largest latency contributor in the whole pipeline after generation itself — worth explicitly flagging in system design discussions (Module 9).

---

## 5.5 LLM-as-reranker

Use a general-purpose LLM (not a specialized cross-encoder) to score or reorder candidates, via prompting rather than a dedicated fine-tuned model.

### Pointwise
Prompt the LLM once per (query, document) pair: "On a scale of 1-10, how relevant is this document to this query?" — score each candidate independently, sort by score.
- Simple, parallelizable, but **no comparative signal** — the LLM never sees other candidates, so its absolute scoring can be poorly calibrated (a "7" from one call and a "7" from another aren't guaranteed to be truly equivalent in relevance).

### Pairwise
Prompt the LLM with two documents at once: "Which of these two documents is more relevant to the query, A or B?" — use pairwise comparisons (e.g. via a tournament or merge-sort-style comparison scheme) to derive a full ranking.
- Better calibration (relative judgments are usually more reliable than absolute scores for LLMs — the same "which is better" psychology as RLHF preference data), but requires O(n log n) or O(n²) comparison calls depending on the sorting scheme — much more expensive than pointwise.

### Listwise
Prompt the LLM with the *entire* candidate list at once: "Given this list of N documents, output them reordered from most to least relevant to the query."
- Most token-efficient (one call instead of many), and gives the LLM full comparative context across all candidates simultaneously — but is bounded by context window (can't listwise-rank hundreds of candidates in one call) and is more prone to position bias (documents earlier in the prompt getting systematically favored/disfavored regardless of true relevance) — same phenomenon as "lost in the middle," previewed here, covered fully in Module 6.

**Cost/latency tradeoff summary**: pointwise is cheapest and most parallelizable but weakest signal; pairwise is highest quality per comparison but most expensive at scale; listwise is a middle ground on cost but has context-length ceilings and position-bias risk. In practice, dedicated cross-encoder rerankers (5.2/5.4) are usually cheaper *and* more reliable than LLM-as-reranker for pure relevance scoring — LLM-as-reranker is more commonly reached for when the ranking criterion is complex/instruction-dependent (e.g. "rank by relevance AND recency AND avoid redundant near-duplicates") in a way a fixed cross-encoder wasn't trained to handle.

---

## 5.6 Where reranking fits in the latency budget

Rough latency profile of a typical RAG request (illustrative, not universal):
```
Query embedding:       ~10-30ms
ANN retrieval (top-50): ~10-50ms
Reranking (50 → top-5): ~100-300ms   ← often the single biggest non-generation cost
Generation:            ~500-3000ms+ (dominates total latency, but reranking is non-trivial)
```

**Top-k → top-n funnel design is a real tuning decision, not an afterthought**:
- Retrieve too few candidates into the reranker (small k) → reranker has nothing good to promote if the true best document wasn't in the initial top-k at all (a *retrieval* recall failure that reranking cannot fix — reranking only reorders what it's given, it can't retrieve documents that were never fetched)
- Retrieve too many candidates into the reranker (large k) → reranking latency/cost scales roughly linearly with k, since it's effectively O(k) forward passes for cross-encoders
- Typical production values: k=50-100 into the reranker, n=5-10 out to generation — but this should be tuned against an eval set (Module 7) measuring how often the true relevant document falls outside top-k at various k values, exactly the same empirical sweep methodology as chunk-size tuning in Module 2.8.

---

## Interview Q&A drill

**Q: Your retrieval pipeline retrieves top-20 candidates and reranks to top-5, but the final answer is still wrong. How do you determine if this is a retrieval problem or a reranking problem?**
A: Inspect whether the actually-relevant document appears *anywhere* in the initial top-20 candidate list before reranking. If it's absent, this is a retrieval recall failure — the reranker can only reorder what it receives, it cannot promote a document that was never fetched, so the fix is in Module 4 (retrieval strategy, hybrid search, k tuning) not the reranker. If the relevant document *is* present in the top-20 but doesn't make the final top-5 after reranking, that's a genuine reranking failure — worth checking the reranker's score distribution, whether it's a mismatch between the reranker's training domain and your corpus (same domain-adaptation issue as Module 1.7, applies to rerankers too), or a bug in how candidates are batched/passed to the reranker.

**Q: Why would you choose ColBERT-style late interaction over a standard bi-encoder + cross-encoder two-stage pipeline?**
A: When you want reranking-level precision without paying the cross-encoder's per-query joint-encoding cost for every candidate, and you're willing to accept a larger index (storing per-token embeddings rather than one pooled vector per document) as the tradeoff. It's especially attractive when latency budget is tight and a separate reranking stage would push past acceptable response time, since ColBERT folds much of the precision benefit into the first-stage retrieval itself rather than requiring a second expensive pass.

**Q: Compare pointwise, pairwise, and listwise LLM-as-reranker approaches on cost and quality.**
A: Pointwise scores each document independently against the query — cheapest and fully parallelizable, but absolute scores from separate LLM calls aren't guaranteed to be consistently calibrated against each other. Pairwise compares two documents at a time and is generally the most reliable per-comparison signal (LLMs are typically better at relative judgments than absolute scoring), but requires many comparison calls to produce a full ranking, scaling poorly with candidate count. Listwise ranks the entire candidate set in a single call — most token-efficient and gives full comparative context, but is limited by context window size for large candidate sets and is more susceptible to position bias, where documents' position in the prompt influences their perceived rank independent of true relevance.

**Q: Why is the reranking stage often the biggest latency cost in the pipeline besides generation itself, and how would you reduce it?**
A: Reranking requires a full model forward pass per (query, document) pair for true cross-encoders, so latency scales with the number of candidates being reranked (k), unlike the initial ANN retrieval stage which is sub-linear in corpus size. Reduction strategies: lower k into the reranker (tuned against eval recall to avoid cutting off the true relevant document), use a smaller/distilled cross-encoder model, batch reranking calls efficiently, or substitute a ColBERT-style late-interaction model that shifts more of the cost to index-time (precomputed token embeddings) rather than query-time joint encoding.

---

**Next up: Module 6 — Augmentation & generation.** Say the word when ready.
