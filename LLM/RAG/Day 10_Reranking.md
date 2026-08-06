# RAG Interview Prep — Day 10
## Reranking

---

## 🚀 Quick Summary

Reranking is the second stage of the two-stage retrieval pipeline introduced on Day 8: a fast first-stage retriever (bi-encoder + ANN, or BM25, or hybrid via Day 9) narrows a huge corpus down to a small candidate set, and a much more accurate — but much slower — model **reorders just that small set** to push the truly best results to the top. Today covers the reranker landscape in full: cross-encoder rerankers (the standard choice), the middle-ground **late-interaction** approach (ColBERT), LLM-based reranking, and — critically — how to actually budget latency across a multi-stage pipeline, since reranking's entire value proposition depends on getting that budget right.

**Think of it like a hiring pipeline.** A resume-screening tool (first-stage retrieval) quickly filters 10,000 applications down to 50 promising candidates using cheap, scalable heuristics. A panel of expert interviewers (the reranker) then spends real, focused time with just those 50, producing a much more reliable final ranking than the resume screener alone ever could — because you obviously can't have the expert panel interview all 10,000 applicants, but you also can't trust the cheap screener's coarse ranking as your final answer for who gets the job.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Reranking** | A second-stage process that reorders a small candidate set from first-stage retrieval using a more accurate (and more expensive) model |
| **Cross-encoder reranker** | A cross-encoder (Day 8) used specifically in the reranking role — scores each (query, candidate) pair jointly |
| **Late interaction (ColBERT-style)** | A middle-ground architecture that keeps per-token embeddings (not fully joint like a cross-encoder, not fully collapsed like a bi-encoder) and computes fine-grained similarity at query time |
| **MaxSim** | The core operation in late-interaction models — for each query token, find its maximum similarity against any document token |
| **LLM-based reranking** | Using a general-purpose LLM, via prompting, to score or reorder candidates instead of a purpose-trained reranker model |
| **Candidate set size** | How many first-stage results get passed into the reranker — the primary lever controlling reranking latency |
| **Score calibration** | Whether a reranker's output scores are meaningfully comparable across different queries, not just internally consistent within one query's ranking |

---

# PHASE 1 — Intuition & Visual Map

## Where reranking sits in the pipeline

```
   FULL CORPUS (millions/billions)
            │
            ▼
   ┌─────────────────────┐
   │  FIRST-STAGE          │   BM25 (Day 7), bi-encoder + ANN (Day 8),
   │  RETRIEVAL            │   or hybrid/RRF (Day 9)
   └─────────────────────┘
            │  narrows to top ~50-500 candidates
            ▼
   ┌─────────────────────┐
   │  RERANKING            │   cross-encoder, late-interaction, or
   │  (today's topic)       │   LLM-based — reorders just this small set
   └─────────────────────┘
            │  narrows/reorders to top ~5-10
            ▼
   PASSED TO GENERATOR (context construction, later week)
```

**Why this two-stage shape is basically universal in production RAG:** first-stage retrieval optimizes for *recall at scale* (don't miss the relevant documents, out of millions), while reranking optimizes for *precision at the top* (given a good-but-imperfect candidate set, get the actual best few into the exact positions that matter most for generation). Neither stage alone is sufficient — Day 8 already showed why a cross-encoder can't be the first stage (latency), and a bi-encoder-only pipeline alone leaves real accuracy on the table (Day 8's information-bottleneck argument), so the combination consistently outperforms either alone.

## When reranking matters most

- ✅ When first-stage retrieval's *ranking order* is unreliable even if its *recall* is good — i.e., the relevant document is somewhere in the top-50, but not necessarily near the top, and position matters (Day 4's nDCG discussion, and lost-in-the-middle from Day 1/13 both apply directly here).
- ✅ When you have a meaningful latency budget to spend on a second stage — reranking always costs additional time, so it's a trade you make deliberately.
- ❌ Less valuable when first-stage retrieval is already highly precise at top-1/2 for your query distribution (rare in practice, but worth knowing reranking isn't "free value" — it's a genuine cost/accuracy trade-off, not a strictly-always-do-it step).

---

# PHASE 2 — The Reranker Landscape

## 1. Cross-Encoder Rerankers (the standard choice)

This is Day 8's cross-encoder architecture, applied specifically to the reranking role. Quick recap of the mechanism, now with reranking-specific framing:

- Each (query, candidate) pair is scored via a joint forward pass through the model — full self-attention between query and candidate tokens.
- Because reranking only operates on a *small* candidate set (not the full corpus), the latency math from Day 8 that made cross-encoders infeasible as first-stage retrievers becomes perfectly manageable here.

**Worked latency budget example (recap + extension):**
```
Reranker: ~15ms per (query, candidate) pair

Candidate set = 50:   50 × 15ms = 750ms
Candidate set = 100:  100 × 15ms = 1,500ms
Candidate set = 20:   20 × 15ms = 300ms
```
This table is the direct, practical version of "candidate set size is the primary reranking latency lever" — if your end-to-end latency budget is tight (say, a 300ms total budget for the whole retrieval+reranking step), you're forced to either use a smaller candidate set, a faster reranker, or parallelize reranking calls (batching the candidate set through the model together rather than strictly sequential, which most real implementations do — the sequential numbers above are illustrative of the *cost driver*, not necessarily literal wall-clock time in an optimized production system).

**Common production rerankers to know by name:** Cohere's Rerank API (a popular managed option), open-source cross-encoder models like `bge-reranker` and MS MARCO-trained cross-encoders (`cross-encoder/ms-marco-*` on Hugging Face) — worth knowing these exist by name, since "what would you actually use" is a common practical follow-up.

---

## 2. Late Interaction — ColBERT

**The problem ColBERT addresses:** cross-encoders are accurate but require a full joint forward pass per candidate (expensive, not precomputable — Day 8). Bi-encoders are fast and precomputable but lose fine-grained token-level interaction by collapsing everything into one vector. ColBERT is a genuine architectural middle ground.

**Mechanism:**
1. Encode the query and each document into **per-token embeddings** (not collapsed into one vector, unlike a standard bi-encoder) — this part *can* be precomputed for documents, since it doesn't require seeing the query first.
2. At query time, compute a **MaxSim** operation: for each query token, find its maximum similarity against *any* token in the candidate document, then sum these per-query-token maximum similarities to get the final relevance score.

```
MaxSim_score(q, d) = Σ_{i ∈ query tokens} max_{j ∈ doc tokens} sim(q_i, d_j)
```

**Plain English:** For every word in the query, ask "what's the single best-matching word anywhere in this document?" and add up those best-match scores across all query words. This preserves genuine token-level granularity (unlike a bi-encoder's single collapsed vector) while still allowing document token embeddings to be precomputed offline (unlike a cross-encoder, which requires the query to already be present during encoding).

**Worked conceptual example:**
```
Query: "battery life AirPods"  → tokens: [battery, life, AirPods]
Document tokens include: [..., battery, duration, AirPods, Pro, ...]

For "battery":  max similarity found against document's "battery" token → high (e.g. 0.95)
For "life":     max similarity found against document's "duration" token → moderate (e.g. 0.71,
                since "life" and "duration" are semantically related but not identical tokens)
For "AirPods":  max similarity found against document's "AirPods" token → high (e.g. 0.97)

MaxSim_score = 0.95 + 0.71 + 0.97 = 2.63
```
Notice this captures something a single collapsed bi-encoder vector might blur together — the query word "life" specifically finding its best match against "duration" elsewhere in the document is a genuinely token-level signal, similar in spirit to what a cross-encoder's attention would capture, but achieved without needing a full joint forward pass.

**Where it fits:** ColBERT-style late interaction sits between bi-encoders and cross-encoders on the speed/accuracy spectrum — more accurate than a pure bi-encoder, meaningfully faster than a full cross-encoder at the same candidate-set size (since document token embeddings are precomputed), but with a real cost: storing *per-token* embeddings for every document is substantially more storage than a bi-encoder's single vector per document (roughly proportional to average document token count), a trade-off worth naming explicitly.

---

## 3. LLM-Based Reranking

**Mechanism:** instead of a purpose-trained reranker model, prompt a general-purpose LLM to score or directly reorder a list of candidates given the query — e.g., "given this query and these 10 passages, rank them from most to least relevant" or "rate this passage's relevance to the query from 1-10."

**Trade-offs:**
- **Pros:** no separate model to train/maintain, can leverage the same strong general-purpose LLM already used for generation, sometimes captures more nuanced relevance judgments (especially for complex, multi-part, or reasoning-heavy queries) than a purpose-trained reranker.
- **Cons:** typically higher latency and cost per candidate than a dedicated cross-encoder reranker (LLM inference is generally more expensive than a small purpose-built cross-encoder), and is subject to the same LLM-judge biases covered in Module 7's evaluation content (position bias, verbosity bias) if not carefully prompted — e.g., a naive "rank these 10 passages" prompt can be sensitive to the *order* the candidates were presented in, echoing the position-bias problem from LLM-as-judge evaluation.

**When to reach for it:** often used as a **third stage**, applied to an already-small shortlist (e.g., top 5-10 from a cross-encoder rerank) where the extra cost is easily affordable and the query/domain benefits from more nuanced reasoning than a purpose-trained reranker provides — rather than as a full replacement for a cross-encoder reranker across a larger candidate set, where the cost would compound quickly.

---

## Reranker Landscape — Comparison Table

| | Cross-Encoder | ColBERT (Late Interaction) | LLM-Based |
|---|---|---|---|
| **Precomputable document representation?** | No | Yes (per-token embeddings) | No |
| **Accuracy** | High | High (close to cross-encoder in many benchmarks) | Variable — can be very high for complex queries, sensitive to prompting |
| **Speed at moderate candidate-set scale** | Moderate | Faster than cross-encoder | Typically slowest and most expensive per candidate |
| **Storage overhead** | Low (no special storage need beyond the base index) | Higher (per-token embeddings, not single vectors) | Low |
| **Typical role** | Standard second-stage reranker | Alternative second-stage reranker, especially at larger candidate-set sizes | Optional third-stage refinement on an already-small shortlist |

---

## Why Reranking's Impact Concentrates at the Top of the List

Tie-in to Day 4's nDCG discussion and the lost-in-the-middle effect: reranking's business value isn't just "does the relevant document appear somewhere in the final list" (that was largely already settled by first-stage recall) — it's specifically about **getting the best documents into position 1-2**, because:
1. Generation quality (a later-week topic) is sensitive to *where* in the context window key evidence sits — burying the best evidence at position 8 instead of position 1 can meaningfully hurt faithfulness even if it's technically "in there."
2. nDCG's logarithmic position discount (Day 4) means moving a highly-relevant document from position 5 to position 1 provides substantially more score improvement than moving an already-mediocre document from position 20 to position 15 — reranking effort is best spent, and best evaluated, on top-of-list precision specifically.

> **Why This Matters callout:** If asked "how do you measure whether your reranker is actually helping," the strong answer references nDCG specifically (not just Recall@k) — since a reranker mostly *reorders* an already-recalled candidate set rather than finding new documents, Recall@k often won't move much from reranking alone, while nDCG (position-sensitive) is exactly the metric built to detect a reranker's real contribution.

---

## Score Calibration — A Frequently Missed Gotcha

**The problem:** a reranker's output scores are trained to produce a good *relative ordering* within one query's candidate set — they are not necessarily meaningfully comparable *across different queries*. A score of 0.7 for one query's top candidate and a score of 0.7 for a completely different query's top candidate don't necessarily represent the same "true" level of relevance.

**Why this matters in practice:** if you try to use raw reranker scores as an absolute relevance threshold (e.g., "only pass documents scoring above 0.5 to the generator, regardless of query") without calibration, you can inconsistently over- or under-filter depending on the query — some queries might have all genuinely relevant documents scoring below 0.5 (because none of the candidates were a great match, but query still needs *some* answer), while others might have irrelevant documents scoring above 0.5 (because the candidate set happened to have unusually similar-looking wrong answers). This is directly relevant to the "no good answer exists" eval slice from Module 7 — a well-calibrated confidence signal is what lets a system reasonably decide when to say "I don't know" versus force an answer from mediocre candidates.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** Why does a two-stage retrieve-then-rerank pipeline consistently outperform either stage alone?

<details>
<summary>Show answer</summary>

First-stage retrieval (BM25, bi-encoder+ANN, or hybrid) is optimized for recall at scale — quickly narrowing a huge corpus to a manageable candidate set — but can't afford the joint query-document attention that produces the most accurate relevance judgments. Reranking (typically a cross-encoder) provides that accuracy but is far too slow to run over a full corpus (Day 8's latency math). Neither alone is sufficient: first-stage-only leaves accuracy on the table at the top of the list, rerank-only is architecturally infeasible at corpus scale — the two-stage combination gets both scale and precision.
</details>

---

**Q2 (Easy — calculation).** A cross-encoder reranker takes 12ms per candidate. Your latency budget for reranking is 400ms. What's the maximum candidate set size you can rerank sequentially within budget?

<details>
<summary>Show answer</summary>

```
400ms / 12ms per candidate ≈ 33.3 → 33 candidates
```
</details>

---

**Q3 (Medium — conceptual).** Explain the MaxSim operation in ColBERT-style late interaction, and why it allows document representations to be precomputed while a standard cross-encoder's cannot.

<details>
<summary>Show answer</summary>

MaxSim computes, for each query token, its maximum similarity against any token in the candidate document, then sums these per-query-token best-match scores to produce the overall relevance score. Document token embeddings can be precomputed offline because encoding a document's tokens doesn't require knowing the query in advance — the query-specific computation (finding each query token's best match) only happens at query time, using the precomputed document token embeddings. A cross-encoder, by contrast, requires the query and document to be present *together* during the model's forward pass (for self-attention across both), so nothing about a cross-encoder's scoring can be precomputed ahead of a specific query.
</details>

---

**Q4 (Medium — conceptual).** Why would you use LLM-based reranking as a third stage on top of a cross-encoder rerank, rather than as a full replacement for the cross-encoder across a larger candidate set?

<details>
<summary>Show answer</summary>

LLM-based reranking is typically higher latency and cost per candidate than a dedicated, purpose-trained cross-encoder reranker, so applying it across a larger candidate set (e.g., 50-100 documents) would compound that cost significantly. Applying it only as a refinement step on an already-small shortlist (e.g., the top 5-10 from a cross-encoder rerank) keeps the extra cost affordable while still gaining LLM-based reranking's potential benefit — more nuanced relevance judgment, especially for complex or reasoning-heavy queries — on exactly the candidates where getting the final order right matters most.
</details>

---

**Q5 (Medium — conceptual, ties to Day 4/Module 7).** Your reranker is deployed, but Recall@k barely changed while nDCG improved noticeably. Is this expected, and why?

<details>
<summary>Show answer</summary>

Yes, this is expected and actually the typical signature of a working reranker. Reranking operates on a candidate set that first-stage retrieval already produced — it reorders those candidates rather than finding new ones, so Recall@k (which only measures whether relevant documents appear anywhere in the top-k, regardless of position) often won't move much from reranking alone. nDCG, being position-sensitive, is specifically built to detect the kind of improvement reranking provides — moving already-recalled relevant documents into better (earlier) positions, which nDCG's logarithmic position discount rewards directly.
</details>

---

**Q6 (Hard — conceptual, gotcha).** A team uses their reranker's raw output score as a fixed confidence threshold across all queries (e.g., "only show results scoring above 0.6") and finds this behaves inconsistently — sometimes filtering out genuinely relevant results, sometimes letting through irrelevant ones. What's the likely root cause, and how would you address it?

<details>
<summary>Show answer</summary>

The likely root cause is a score calibration problem: reranker scores are generally trained to produce a good *relative* ordering within a single query's candidate set, not to be meaningfully comparable in absolute terms *across different queries*. A score of 0.6 for one query's candidates doesn't necessarily represent the same true relevance level as 0.6 for a different query — some queries may have all their genuinely relevant candidates score below the threshold (because none of the retrieved candidates were a strong match), while others may have irrelevant candidates score above it (because the candidate set happened to contain unusually similar-looking wrong answers). Rather than relying on a fixed absolute threshold, I'd consider calibration techniques (e.g., validating and adjusting score interpretation against a labeled eval set, or using relative signals like "is there a large score gap between the top result and the rest" rather than a fixed cutoff), or reframe the decision around relative confidence within each query's own candidate set instead of a single global threshold.
</details>

---

**Q7 (Hard — system design synthesis).** Design a full multi-stage retrieval pipeline (first-stage through final context construction) for a RAG system needing sub-300ms end-to-end retrieval+reranking latency over a 20-million-document corpus, balancing recall and precision. Justify each stage and its approximate latency budget allocation.

<details>
<summary>Show answer</summary>

Stage 1 — hybrid first-stage retrieval (Day 9): BM25 + bi-encoder/ANN in parallel, fused via RRF, targeting the top ~100 candidates; budget roughly 50-80ms (query encoding + ANN search + BM25 lookup, largely parallelizable). Stage 2 — cross-encoder (or ColBERT, if storage budget allows and latency is tighter) reranking of those ~100 candidates down to the top ~10-15; at ~10-15ms per candidate for a cross-encoder, batched/parallelized inference rather than strictly sequential, targeting roughly 150-200ms for this stage, which is the dominant cost in the pipeline and the primary lever to tune (reducing candidate set size from stage 1 is the main way to control this budget if latency is tight). Remaining budget (~50-70ms) covers final selection/formatting for context construction. I would not add an LLM-based third-stage rerank given the tight 300ms overall budget — that's better suited to a use case with a looser latency requirement, since LLM inference per candidate is typically more expensive than the cross-encoder's per-candidate cost already dominating the budget.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Treating reranking as "free extra accuracy" without acknowledging it's a genuine latency/cost trade-off, tuned via candidate set size.
- ❌ Forgetting ColBERT-style late interaction exists as a real middle ground — jumping straight from "bi-encoder" to "cross-encoder" as if those are the only two options.
- ❌ Using LLM-based reranking across a large candidate set instead of reserving it for an already-small shortlist, given its typically higher per-candidate cost.
- ❌ Evaluating reranker impact using only Recall@k — nDCG is the metric actually built to detect a reranker's contribution, since reranking reorders rather than discovers new candidates.
- ❌ Using raw reranker scores as a fixed cross-query threshold without accounting for score calibration issues.
- ❌ Not naming concrete production rerankers (Cohere Rerank, bge-reranker, MS MARCO cross-encoders) when asked what you'd actually use.

---

# 📌 Cheat Sheet (Day 10)

**Role:** second-stage reordering of a small first-stage candidate set — optimizes top-of-list precision, not new recall.

**Cross-encoder reranker:** standard choice, joint query-candidate scoring, ~10-20ms/candidate typical — feasible only because the candidate set is small (Day 8's latency math flips once you're not scoring the full corpus).

**ColBERT / late interaction:** MaxSim = sum of each query token's best-matching document token similarity. Precomputable document token embeddings (unlike cross-encoders), more storage than bi-encoders, sits between bi-encoder and cross-encoder on speed/accuracy.

**LLM-based reranking:** most flexible/nuanced, most expensive — reserve for an already-small shortlist as an optional third stage, watch for the same position/verbosity biases as LLM-as-judge evaluation.

**Evaluation:** measure reranker impact via nDCG (position-sensitive), not Recall@k (reranking reorders, doesn't discover).

**Gotcha to remember:** reranker scores aren't calibrated across queries — don't use a fixed absolute threshold; think in relative/within-query terms instead.

---

*End of Day 10. Next up — Day 11: Query Transformation (HyDE, multi-query, decomposition).*
