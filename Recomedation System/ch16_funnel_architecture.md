# Chapter 16: Candidate Generation vs. Ranking vs. Re-Ranking (Funnel Architecture)

## 1. Intuition

Every model discussed so far (Chapters 4-15) has an implicit cost profile: some are cheap and coarse (embedding dot products), some are expensive and precise (cross-feature MLPs, LambdaMART, self-attention over sequences). No production system at Google/YouTube/Pinterest scale can afford to run the expensive, precise models against the **entire catalog** for every single request — that's the central constraint this chapter resolves.

The answer, used universally across large-scale recsys, is a **multi-stage funnel**: start with a huge candidate pool (the whole catalog, potentially billions of items), progressively narrow it down through stages of increasing model complexity and decreasing candidate count, so the most expensive computation only ever runs on a small, already-promising set. This chapter is where Modules 2-4's individual algorithms get assembled into an actual system — it's the single most commonly asked system-design framing in L5 recsys interviews.

## 2. The Three Stages

**Stage 1 — Candidate Generation (Retrieval)**
- Goal: reduce billions of items down to hundreds/low-thousands of plausible candidates, fast and cheap, prioritizing **recall** (don't miss good items) over precision.
- Typical models: two-tower embeddings + ANN search (Ch. 12, 17), sometimes multiple parallel candidate sources (e.g., one two-tower model for "similar to recently watched," another for "trending," another for GNN-based co-engagement, Ch. 15) whose outputs are unioned together.
- Latency budget: extremely tight (milliseconds), since this runs against the full catalog.
- Why precision doesn't matter much here: any irrelevant candidate that slips through gets filtered out at later stages — the real danger at this stage is **missing** a genuinely great candidate, since it never gets a second chance if it's not retrieved at all. This is why recall, not precision, is the guiding metric here.

**Stage 2 — Ranking**
- Goal: take the hundreds/thousands of candidates from Stage 1 and score them with a richer, more expensive model that jointly considers user-item feature interactions.
- Typical models: DeepFM, Wide & Deep (Ch. 13), LambdaMART (Ch. 10), or neural rankers incorporating sequence features (Ch. 14) — models that were too expensive to run against the full catalog but are entirely affordable against a few thousand candidates.
- Latency budget: looser than Stage 1 but still real-time-bound (tens of milliseconds), since it now runs per-request but only over the much smaller candidate set.
- This stage optimizes ranking-quality metrics directly (NDCG, MAP — Ch. 2), using pairwise/listwise LTR objectives (Ch. 8-10) since the goal is now genuinely about getting the *order* right among plausible candidates, not just identifying *whether* something is plausible at all.

**Stage 3 — Re-Ranking**
- Goal: apply business logic, diversity constraints, freshness boosts, and policy filters to the ranked list from Stage 2, producing the final list actually shown to the user.
- Typical operations: diversity re-ranking (avoid showing 10 nearly-identical items even if they all score highly — Ch. 21), deduplication, freshness/recency boosting, business rules (e.g., promoting certain content categories, applying content policy filters, ensuring a minimum quota of a certain type of content), and sometimes explicit exploration injection (Ch. 22's bandits) to gather data on under-explored items.
- This stage is often **rule-based or lightweight-model-based** rather than another full ML model — it's about applying constraints and adjustments on top of an already-good ranking, not re-deriving relevance from scratch.

## 3. Why This Specific Ordering (Not the Reverse)

The funnel is structured cheap-to-expensive specifically because model cost and candidate-set size have an inverse relationship in what's affordable: expensive, feature-rich models (Stage 2) are precisely the ones that **cannot** scale to run against the full catalog (this is the exact limitation Chapter 12 identified for NCF/joint architectures), so they can only be used once the candidate set has already been narrowed down by something cheap enough to run at full-catalog scale (Stage 1's embedding-based retrieval). Trying to run Stage 2-caliber models directly against the full catalog would blow every realistic latency budget; trying to make final business-logic decisions (Stage 3) before ranking quality is established (Stage 2) would mean applying diversity/freshness rules to a poorly-ordered list, wasting that logic's value.

## 4. Multiple Candidate Sources at Stage 1

Real production candidate generation is rarely a single two-tower model — it's typically an **ensemble of multiple retrieval sources**, each capturing a different kind of relevance signal, unioned together before ranking:

- A two-tower model trained on general engagement signal (broad personalization)
- A separate two-tower or GNN-based model specifically for "similar items to what you just interacted with" (short-term intent, Ch. 14/15's sequence/graph signal)
- A popularity/trending source (Ch. 3's baseline, still present as a fallback/supplementary signal)
- Sometimes a freshness-focused source (recently published items, to ensure new content gets exposure despite lacking interaction history — a partial cold-start mitigation, Ch. 1)

Each source might contribute, say, 100-300 candidates, and the union (with deduplication) forms the full candidate set that Stage 2 then ranks. This multi-source design directly addresses a limitation of any single retrieval model — no one embedding space captures every kind of relevance (long-term taste, short-term intent, trending, freshness) equally well, so diversity of retrieval *sources* is itself a design lever, separate from diversity of final *results* (which is Stage 3's job).

## 5. Worked Example — Funnel Sizing and Latency Budget

A concrete illustrative funnel for a video recommendation system with a catalog of 500 million videos and a 200ms total latency budget:

| Stage | Input size | Output size | Approx. latency budget | Model type |
|---|---|---|---|---|
| Candidate Generation | 500,000,000 | ~1,000 (union of sources) | ~50ms | Two-tower + ANN (Ch. 12, 17), GNN embeddings (Ch. 15), popularity |
| Ranking | ~1,000 | ~100 | ~100ms | DeepFM / neural ranker with sequence features (Ch. 13, 14) |
| Re-Ranking | ~100 | ~10-20 (shown to user) | ~30ms | Diversity/business rules, lightweight scoring |

Note the massive reduction at Stage 1 (500M → 1,000 is a 500,000x reduction) achieved in the tightest latency budget, precisely because it uses the cheapest per-item computation (a single dot product against a precomputed ANN index, Ch. 17) — this is only possible because two-tower embeddings can be precomputed offline for the item side (Ch. 12's key structural property). Stage 2's much smaller reduction (1,000 → 100, only 10x) gets a larger latency budget because each candidate now requires a genuinely more expensive joint computation.

## 6. Production Considerations

- Each stage typically has **separately trained models with different objectives** — Stage 1's retrieval model is often trained with a recall-oriented objective (in-batch softmax over a broad positive set, Ch. 12) while Stage 2's ranking model is trained with a precision/order-oriented objective (pairwise/listwise LTR, Ch. 8-10) — using the same model/objective for both stages is a common design mistake, since what makes a good retriever (broad recall) and a good ranker (precise ordering) are genuinely different optimization targets.
- A known failure mode: **training-serving skew** between stages — if Stage 2's ranking model is trained on candidates that came from a different (e.g., older, or randomly sampled) retrieval distribution than what Stage 1 actually produces in production, the ranker sees a distribution mismatch at serving time, degrading quality. This is why retraining pipelines often need coordination between stages (e.g., regenerating ranking training data using current-Stage-1-retrieved candidates).
- Latency budgets are hard real-time constraints in consumer-facing systems — the funnel design is as much a **systems engineering discipline** as it is a modeling exercise, and L5 candidates are expected to reason about both simultaneously, not just describe algorithms in isolation.

## 7. Interview Traps

- Proposing to run an expensive ranking model (DeepFM, LambdaMART) directly against the full catalog — a classic sign of not internalizing the cost-vs-catalog-size trade-off that motivates the entire funnel design.
- Describing only a single retrieval model at Stage 1, without mentioning that real systems typically union multiple candidate sources — a common oversimplification.
- Treating re-ranking as "just re-running the ranking model again" rather than recognizing it as a distinct stage focused on business logic, diversity, and constraints layered on top of an already-good ranking.
- Not mentioning training-serving skew as a risk across funnel stages when asked about production challenges — a frequently probed systems-level failure mode.

## 8. L5-Differentiating Talking Points

- Explicitly reason about **latency budgets and candidate-set sizes numerically** at each stage (as in Section 5's table) rather than describing the funnel purely qualitatively — grounding the architecture in concrete engineering constraints is a strong L5 signal.
- Bring up multi-source candidate generation and the recall-vs-precision objective mismatch between Stage 1 and Stage 2 unprompted — this shows you understand that the funnel isn't just "big model → small model," but a deliberate change in optimization target at each stage.
- Mention training-serving skew across funnel stages as a specific, named production risk — a detail that separates genuine production experience from purely academic algorithm knowledge.
- Frame the entire funnel as the synthesis point of the whole curriculum so far: cheap embedding-based retrieval (Module 2/4's MF, two-tower) feeds richer feature-interaction ranking (Module 3/4's LTR, DeepFM) feeds business-logic re-ranking (Module 6's diversity/fairness) — explicitly narrating this pipeline view is exactly the kind of systems synthesis that distinguishes L5 system design answers.

## 9. Comprehension Check

1. Why can't the ranking-stage model (e.g., DeepFM, LambdaMART) simply be run against the entire catalog directly, skipping candidate generation?
2. Why is recall the primary metric of concern at the candidate generation stage, while precision/ranking-quality metrics matter more at the ranking stage?
3. Why do real production systems typically use multiple candidate generation sources rather than a single retrieval model?
4. What is training-serving skew in the context of a multi-stage funnel, and why is it a risk specifically at the boundary between candidate generation and ranking?
5. What kinds of operations belong in the re-ranking stage that don't belong in the ranking stage itself?
