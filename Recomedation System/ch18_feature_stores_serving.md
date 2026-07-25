# Chapter 18: Multi-Stage Ranking, Feature Stores, Real-Time vs. Batch Serving

## 1. Intuition

Chapter 16 established the funnel's stages; Chapter 17 solved retrieval's core scalability problem. This chapter addresses the plumbing question that makes the whole funnel actually work in production: **where do the features come from, at each stage, within the latency budget?** A model is only as good as the features it can actually access at serving time — and a huge fraction of real-world recsys engineering effort goes into feature infrastructure, not model architecture. This is a chronically underweighted topic in interview prep relative to how much it matters in practice.

## 2. The Feature Freshness Spectrum

Features used across the funnel fall on a spectrum from **batch/precomputed** (stale but cheap) to **real-time/streaming** (fresh but expensive):

- **Batch features**: computed periodically (hourly/daily) via offline pipelines — e.g., a user's 30-day aggregate watch-time-by-category, an item's historical average engagement rate, precomputed two-tower embeddings (Ch. 12). Cheap to serve (just a lookup), but can be hours-to-a-day stale.
- **Near-real-time / streaming features**: computed continuously as events arrive (e.g., via a stream processing system), updated within seconds-to-minutes — e.g., "items the user clicked in the last 10 minutes," "current trending rate of an item in the last hour."
- **Real-time / request-time features**: computed at the moment of the request itself, using only information available right now — e.g., current session's items-viewed-so-far, current device/context, time-of-day.

The general rule: **the earlier funnel stages (Ch. 16) lean more on batch features** (since Stage 1 needs to be extremely cheap per item, across the whole catalog), while **later stages can afford richer, fresher features** (since they operate over a much smaller candidate set). A user's precomputed two-tower embedding (batch, refreshed periodically) is what powers Stage 1 retrieval; by Stage 2 ranking, the system can afford to also pull in near-real-time signals (what the user clicked in the last few minutes) as additional input features to the ranker.

## 3. Feature Stores

A **feature store** is the infrastructure component that solves a specific, recurring production problem: **training-serving skew from feature computation mismatch**. If the features used to train a model are computed via one code path (e.g., a batch Spark job over historical logs) and the features used at serving time are computed via a different code path (e.g., an online service reimplementing similar logic), subtle discrepancies between the two implementations silently degrade model quality — the model was trained on features computed one way but is served features computed a slightly different way, and this mismatch is notoriously hard to detect without dedicated infrastructure.

A feature store centralizes feature definitions so that **the same feature computation logic is used for both training (offline) and serving (online)** — typically via a dual-write/dual-read architecture: an offline store (e.g., a data warehouse table) for generating training datasets, and an online store (e.g., a low-latency key-value store like Redis or Bigtable) for serving-time lookups, both populated from the same underlying feature definitions/pipeline. This training-serving consistency guarantee is the feature store's core value proposition — it's an infrastructure answer to a correctness problem, not just a performance optimization.

## 4. Worked Example — A Concrete Feature Store Flow

Consider a single feature: "user's average watch time per session over the last 7 days."

**Without a feature store** (the failure mode): the ML training pipeline computes this via a batch SQL/Spark job reading historical logs. Separately, the serving system implements its own version of this computation (perhaps in a different language, with a slightly different windowing definition — e.g., using a rolling 7-day window vs. a calendar-week window) to serve it at request time. These two implementations can silently diverge (different edge-case handling, different timezone assumptions, different definitions of "a session"), and the model sees different feature distributions at train vs. serve time, which is a common and hard-to-detect source of a model performing worse in production than offline metrics predicted.

**With a feature store**: the feature "avg_watch_time_7d" is defined **once**, in a single pipeline definition. The offline batch job populates historical values into the offline store for generating training data (joining feature values as they existed at each historical point in time — critical for avoiding a subtler failure, **temporal leakage**, i.e., accidentally using a feature's *current* value when training on a *past* example, which would let the model implicitly "see the future"). The same pipeline also computes and continuously refreshes each user's *current* value into the online store, which the serving system queries directly at request time — no separate reimplementation, no drift.

## 5. Real-Time vs. Batch Serving — The Full Picture

Putting this together with Chapter 16's funnel:

| Component | Feature freshness | Why |
|---|---|---|
| Stage 1 (Retrieval) item embeddings | Batch (precomputed, Ch. 12) | Must be looked up instantly across huge catalog; recomputing per-request is infeasible |
| Stage 1 user embedding | Often computed at request time from a mix of batch (long-term profile) + real-time (current session) features | User tower forward pass is cheap (one user, one embedding), so some freshness is affordable even at this stage |
| Stage 2 (Ranking) features | Mix of batch (historical aggregates), near-real-time (recent session activity), and request-time (context) | Smaller candidate set affords richer, fresher feature computation per candidate |
| Stage 3 (Re-ranking) | Mostly request-time (diversity within *this* returned list) plus business-rule lookups | Operates on the final small list; needs to reason about the list as a whole, which is inherently request-time |

This table is a direct continuation of Chapter 16's Section 5 latency table — the freshness spectrum and the funnel's cost spectrum are two views of the same underlying constraint: the amount of computation (and therefore the achievable feature freshness) you can afford per item scales inversely with how many items you're processing per request.

## 6. Production Considerations

- Feature stores are a genuinely major piece of ML infrastructure at large companies (e.g., internally-built systems at Google/Meta, or open-source options like Feast) — mentioning this by name and describing the train/serve consistency problem it solves is a strong, concrete signal of production awareness.
- Temporal correctness (point-in-time correctness) in the offline store is a subtle but critical correctness requirement — training data must be joined with feature values *as they existed at the time of the historical event*, not current values, or the model implicitly trains on information it wouldn't have had at serving time (a specific, checkable form of data leakage distinct from the general concept).
- Real-time features add genuine infrastructure complexity (stream processing systems, low-latency serving stores) — teams make a deliberate cost/benefit call about which features are worth the engineering investment to make real-time versus accepting batch staleness, not adding real-time freshness everywhere by default.

## 7. Interview Traps

- Not being able to explain concretely *why* training-serving feature skew happens (two different code paths computing "the same" feature slightly differently) — a vague "there can be inconsistencies" answer is much weaker than naming the dual-implementation root cause.
- Forgetting point-in-time correctness when describing how training data is generated from a feature store — using a feature's current value for a historical training example is a specific, nameable leakage bug, not just generic "be careful with data."
- Assuming every feature should be real-time — ignoring the genuine cost/complexity trade-off and the fact that most large-scale systems intentionally keep most features batch-computed, reserving real-time infrastructure for features where freshness measurably matters.
- Describing feature freshness as uniform across the whole funnel, rather than recognizing (as in Section 5's table) that freshness affordability changes stage by stage, mirroring Chapter 16's latency/cost funnel logic.

## 8. L5-Differentiating Talking Points

- Name the specific failure mode a feature store solves (dual-implementation training-serving skew) rather than describing it only as "a place to store features" — this precision is what separates genuine infrastructure understanding from a surface-level buzzword mention.
- Bring up point-in-time correctness for offline training data generation unprompted — this is a specific, well-known-in-industry but often-missed detail that signals real production ML engineering experience.
- Explicitly connect the feature-freshness spectrum to Chapter 16's funnel-stage cost structure — showing that "how fresh can this feature be" and "how expensive can this stage's computation be" are the same underlying trade-off viewed from two angles.
- Discuss the deliberate cost/benefit decision-making behind which features get real-time treatment, rather than treating "more real-time is always better" as an assumption — reflecting engineering judgment about where complexity investment is actually worth it.

## 9. Comprehension Check

1. What specific problem does a feature store solve, and why is it hard to solve without one?
2. What is point-in-time correctness, and what would go wrong in a training pipeline without it?
3. Why do earlier funnel stages (candidate generation) lean more heavily on batch-computed features than later stages (ranking, re-ranking)?
4. Give an example of a feature that would typically be computed at request time versus one computed via a nightly batch job, and explain why each fits its category.
5. Why might a team deliberately choose *not* to make a given feature real-time, even if real-time infrastructure is available?
