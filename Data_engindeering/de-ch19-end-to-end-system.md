# Chapter 19: End-to-End Worked System

## Why this chapter is structured differently

Ch1–18 each introduced one concept in isolation. This chapter is the
opposite exercise: **a single interview question, answered start to
finish**, using nothing but what you've already learned — with commentary
showing exactly which chapter each design decision comes from. This is
also, not coincidentally, close to the actual shape of an MLE system-design
interview: you won't be asked "what's a shuffle" cold, you'll be asked to
design something and expected to bring these concepts up yourself, unprompted,
at the right moments.

---

## The question

**"Design the data pipeline for a real-time product recommendation system
— 'customers who viewed/bought X also viewed/bought Y' — for an e-commerce
site with millions of daily users."**

This is intentionally the same running example threaded through the whole
syllabus — seeing it assembled end to end is the point.

---

## Step 1: Clarify scope and requirements (before designing anything)

A strong answer starts here, not with architecture. Questions worth
raising out loud:

- **Latency requirement:** does "real-time" mean recommendations update
  within milliseconds of a new event, or is "fresh within the hour"
  actually fine? (Ch2 — this single answer reshapes the whole design.)
- **Scale:** roughly how many events/day? (Shapes partition counts, Ch8/
  Ch11, and whether a single Spark cluster is even sufficient.)
- **Cold start:** what happens for a brand-new product/user with no
  history yet? (Not this syllabus's focus, but worth a one-line
  acknowledgment — shows you're thinking about the whole system, not just
  the pipeline.)

**Assume for this walkthrough:** recommendations should reflect activity
from the last few minutes for trending/urgency signals, but the core
co-purchase model can retrain daily. This mixed requirement is realistic
and is exactly what justifies a **hybrid batch + streaming** design rather
than picking one purely.

---

## Step 2: Sources and ingestion (Ch1, Ch10-11)

- **Sources:** website emits `{user_id, product_id, action, timestamp}`
  events on every view/purchase; a production orders database holds
  authoritative transaction records.
- **Ingestion:** events are published to a **Kafka** topic,
  `product-events`, **partitioned by `user_id`** (Ch11) — keeping a given
  user's activity ordered while spreading load across partitions for
  throughput. Partition count sized generously relative to expected peak
  events/sec, since re-partitioning later is disruptive.
- Say explicitly: *this decouples the website's checkout/browsing code
  from every downstream consumer (Ch10)* — new consumers (a future fraud
  model, an analytics team) can subscribe later with zero changes to the
  website.

---

## Step 3: Storage layer (Ch3-4)

- Raw events get written from Kafka into a **data lake**, as **Parquet**,
  **partitioned by date** (Ch4) — chosen because downstream jobs
  (retraining, ad-hoc analysis) predominantly filter by recent date
  ranges, enabling partition pruning.
- Explicitly note: this is an **ELT** pattern (Ch17) — raw events land
  before any transformation, preserving the ability to compute entirely
  new features later without re-extracting from source.
- If the team also needs fast BI/dashboard queries over aggregated stats,
  mention a **lakehouse** layer (Ch3) on top (e.g., Delta Lake) rather
  than standing up a fully separate warehouse — avoids duplicating data
  unnecessarily.

---

## Step 4: Batch processing — training data & daily model refresh (Ch5-9)

- A nightly **Spark** job reads the last N days of events from the lake
  (**partition pruning** on date, Ch4/Ch7).
- **Joins** raw events with a small product-metadata table — Catalyst
  (Ch7) picks a **broadcast join** automatically since the metadata table
  is small, avoiding a shuffle of the much larger events table.
- **GroupBy** product pairs to compute co-purchase statistics — this step
  **does shuffle** (Ch8); flag proactively that if certain products are
  extreme outliers (viral/limited items), this groupBy is where **skew**
  would show up, and salting the key is the fix if profiling confirms it.
- Point-in-time correctness (Ch9): training data must reflect only
  information known *as of* each historical example's timestamp, avoiding
  label leakage — worth stating explicitly, since it's a common way this
  kind of pipeline goes subtly wrong.
- Output (co-purchase stats / trained model artifacts) written back to the
  lake/warehouse, **idempotently** — overwriting that day's partition
  rather than appending (Ch15), so retries and backfills stay safe.

---

## Step 5: Streaming path — real-time trending signal (Ch2, Ch12-13)

- A lightweight stream processor (**Kafka Streams**, chosen over spinning
  up Spark Structured Streaming for this simpler, focused transformation —
  Ch13) consumes `product-events` and maintains a rolling count of recent
  views/purchases per product, updated continuously.
- Consumer group sized so partition count ≥ number of consumer instances
  (Ch12) — no wasted idle consumers.
- Processing logic is **idempotent** — the rolling count is recomputed/set
  from a time window rather than incremented, so at-least-once delivery
  (Kafka's practical default, Ch12) can't cause double-counting on retry.
- Result written to a low-latency **online feature store** (e.g., Redis) —
  this is what the live recommendation service actually queries at
  request time; it never talks to Kafka directly (Ch13).

---

## Step 6: Feature store — tying batch and streaming together (Ch18)

- Both paths — nightly Spark output (offline store) and the real-time
  rolling counts (online store) — are framed as populating a **feature
  store**, ideally from a shared feature definition, specifically to
  minimize the risk that "recent product popularity" is computed
  differently for training vs. serving.
- This is the direct, architectural answer to **training-serving skew**
  (a thread since Ch1) — worth stating explicitly as the reason this
  design has two paths sharing one source (the Kafka topic) rather than
  two independently-built pipelines.

---

## Step 7: Orchestration & reliability (Ch14-16)

- The batch path (ingest → validate → join → aggregate → write) is a
  DAG in **Airflow** — dependencies enforced (no job runs on incomplete
  upstream data), automatic retries on transient failures, and
  idempotent tasks make those retries safe (Ch15).
- **Validation tasks** (Ch16) run right after ingest — schema, null,
  volume checks — so a broken upstream event format fails fast, before an
  expensive Spark join runs on bad data.
- **Monitoring** tracks pipeline freshness (has the nightly job run
  successfully today?), event volume (any anomalous drop?), and —
  specifically for this design — periodically spot-checks the batch and
  streaming paths' outputs against each other as a skew-detection
  mechanism (Ch16/Ch18).

---

## Step 8: Serving

- The live recommendation service, when generating recommendations for a
  user, queries: the **online feature store** (real-time trending signal)
  + the precomputed **co-purchase stats table** (from the batch model,
  loaded into a fast-lookup store) — combining a stable, high-quality
  batch signal with a responsive, fresh real-time signal.

---

## Full picture, one diagram

```
Website ──▶ Kafka (product-events, partitioned by user_id)
              │
              ├──▶ [ELT: load raw] ──▶ Data Lake (Parquet, partitioned by date)
              │                             │
              │                    Airflow-orchestrated Spark job:
              │                    validate → broadcast-join products →
              │                    groupBy (watch skew) → idempotent write
              │                             │
              │                             ▼
              │                     Offline Feature Store ──┐
              │                                              │
              └──▶ Kafka Streams (rolling counts,            │
                    idempotent) ──▶ Online Feature Store ────┤
                                                              ▼
                                                  Recommendation Service
                                                  (serves live predictions)
```

---

## Common interviewer follow-ups, and how to answer them

**"What happens if the nightly Spark job fails?"**
Airflow retries automatically (Ch15); if it keeps failing, the online
(real-time) path still serves fresh trending signals, and the batch
co-purchase stats simply stay one day stale rather than breaking entirely
— worth naming this graceful degradation as an intentional property of the
hybrid design, not an accident.

**"How would you detect training-serving skew here specifically?"**
Compare the streaming-computed and batch-computed versions of any
overlapping signal (Ch16/Ch18) — e.g., periodically recompute yesterday's
"purchases in last 10 minutes" from the batch path and diff it against
what the streaming path reported at that time; alert if they diverge
beyond a tolerance.

**"How would this scale if daily events grew 100x?"**
More Kafka partitions (Ch11) and more consumer instances up to that
partition count (Ch12); more Spark executors and attention to whether the
groupBy shuffle (Ch8) needs repartitioning/salting at the new scale;
possibly reconsidering the lake's partitioning scheme (Ch4) if per-day
files become unmanageably large or small.

**"Why not just make everything real-time streaming, given it's a
'real-time' recommendation system?"**
Because the co-purchase model itself doesn't need millisecond freshness —
daily retraining is both sufficient and far simpler/cheaper (Ch2) — while
only the "recent trending" signal genuinely benefits from streaming
freshness. Using streaming everywhere would add unnecessary complexity and
cost (Ch2's core tradeoff) for signals that don't need it.

---

## Quick recap

- A strong system-design answer starts with clarifying latency/scale
  requirements, not architecture — the requirements determine whether you
  need batch, streaming, or (usually, for real systems) both.
- This worked example combined nearly every earlier chapter: Kafka
  ingestion (Ch10-13), lake storage in Parquet (Ch3-4), Spark batch
  processing with attention to joins/shuffle/skew (Ch5-9), Airflow
  orchestration with validation and idempotency (Ch14-16), ELT (Ch17), and
  a feature store tying the batch and streaming paths together to limit
  skew (Ch18).
- Being able to proactively name *why* each piece is there — not just
  what it is — is what separates a strong interview answer from a
  name-dropping one.

---

Next: **Ch20 — Rapid-Fire Interview Q&A**, drilling the "explain X in one
minute" version of every concept in this syllabus. Say "ch20" when ready.
