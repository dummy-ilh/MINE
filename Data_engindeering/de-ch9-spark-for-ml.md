# Chapter 9: Spark for ML Workloads

## Closing out Module B

Chapters 5–8 built up Spark's mental model from first principles. This
chapter is the payoff: applying that model specifically to the kind of work
an MLE actually does with Spark — feature engineering at scale, reading
training data efficiently, and knowing where Spark's job ends and the
training framework's job begins.

---

## Feature engineering at scale: common patterns

Most ML feature engineering with Spark falls into a few recurring shapes.
Recognizing them (and their relevant chapter, i.e. their cost) is the
actual skill:

**1. Aggregations over a window (e.g., "user's average order value over
last 30 days")**
- This is a `groupBy` + aggregate — which means it triggers a **shuffle**
  (Ch8). If this feature needs to be computed for millions of users daily,
  the shuffle cost is the thing to be aware of and optimize (e.g., careful
  partitioning by `user_id` up front, watching for skew if some users have
  vastly more activity than others).

**2. Joining multiple data sources (e.g., combining user demographics +
transaction history + product metadata into one feature table)**
- This is exactly the join patterns from Ch8. If one of the sources is
  small (e.g., product metadata), it's a broadcast-join candidate — cheap.
  If all sources are large, it's a shuffle join — expensive, and worth
  double-checking Catalyst's plan (via `.explain()`) rather than assuming.

**3. Point-in-time correctness ("as-of" joins)**
- A subtler but very real ML-specific pattern: when building training
  data, a feature must reflect only what was *known at the time* of the
  training label, not future information — otherwise you get **label
  leakage** (the model implicitly learns from data it wouldn't have had at
  prediction time). E.g., "user's total purchases" as of the moment a
  churn label was recorded, not total purchases *ever* (which would leak
  future information for users who churned early but purchased more
  later). Spark doesn't automatically prevent this — it's a query-design
  responsibility, and it's one of the more common subtle bugs behind
  suspiciously good offline model performance.

**4. Sampling/splitting (train/val/test)**
- Needs to be done carefully at scale — a naive `.sample()` per partition
  can introduce bias if data isn't randomly distributed across partitions
  to begin with (e.g., if data is partitioned by date and you sample
  within each partition, you might inadvertently stratify by date in a way
  you didn't intend). Worth explicitly checking that your partitioning
  scheme doesn't accidentally correlate with your split logic.

---

## Reading training data efficiently

This section is really "Ch4 and Ch7's optimizations, applied on purpose."

- **Partition pruning:** if your lake is partitioned by date (Ch4) and you
  only need the last 90 days of data for training, filtering on date lets
  Spark skip reading everything outside that range — instead of scanning
  the entire historical lake and filtering after the fact.
- **Column pruning:** if your table has 200 columns but your model only
  uses 30 of them as features, `select`ing just those 30 (rather than
  reading everything and dropping columns later in the pipeline) lets
  Catalyst's column pruning (Ch7) skip reading the other 170 off disk
  entirely.
- **Avoiding premature `.collect()`:** a common beginner mistake in ML
  feature pipelines is calling `.toPandas()` or `.collect()` too early —
  pulling a huge distributed DataFrame back into a single machine's memory
  (the driver) before it's actually necessary, defeating the entire point
  of using Spark and risking an out-of-memory crash (the failure mode
  flagged back in Ch6).
- **Caching intermediate results deliberately:** if a feature table gets
  reused across several downstream steps (e.g., computing multiple
  different features off the same joined base table), explicitly caching
  it avoids Spark recomputing that lineage from scratch every single time
  it's referenced — a direct, practical use of the lineage concept from
  Ch6.

---

## Where Spark ends and the training framework begins

This is a boundary worth being precise about, because interviewers will
sometimes probe exactly where you'd draw this line.

**Spark's job:** large-scale, distributed data preparation — reading raw
data, joining, aggregating, computing features, producing a clean training
dataset (or writing it out as a table/file that a training job will later
read).

**The training framework's job (PyTorch/TensorFlow/etc.):** once you have a
prepared, feature-complete dataset that's small enough to reasonably fit
into a training pipeline's data loader (even if it's still large — sharded
across GPUs, streamed in batches), the *model training* itself — gradient
computation, backprop, optimizer steps — is not something Spark does. Spark
doesn't train neural networks; it prepares the data neural networks train
on. (There's a legacy library, MLlib, for classical ML directly in Spark —
worth knowing it exists, but for deep learning workloads, the handoff to a
dedicated framework is the standard modern pattern.)

**The natural boundary, in one sentence:** *Spark owns "get from raw,
scattered data to a clean, feature-complete training set"; the training
framework owns "turn that training set into a trained model."* Some teams
draw this line at "Spark writes Parquet, training framework's data loader
reads Parquet" — a clean handoff via storage rather than in-process
integration, which also nicely sidesteps a lot of complexity.

---

## Worked example, tying the whole module together

Full picture for the recommendation model's training data prep:

1. Read raw events (Ch4's Parquet, partitioned by date) — apply **partition
   pruning** to only read the last 90 days (Ch4/Ch7).
2. **Join** with product metadata (small table) → Catalyst picks a
   **broadcast join** (Ch8) — cheap.
3. **GroupBy** user + time window to compute rolling features like "purchases
   in last 30 days" → this **shuffles** (Ch8) — watch for skew if some
   users are far more active than others (e.g., bot accounts).
4. Carefully construct features to reflect only information available
   **as of** the training label's timestamp — avoiding label leakage.
5. `select` only the final feature columns needed (**column pruning**,
   Ch7) and write the result out as Parquet.
6. **Handoff:** a PyTorch training job's data loader reads that Parquet
   output directly — Spark's job is done; it never touches gradients or
   model weights.

---

## Downstream considerations

1. **Latency:** Inefficient feature pipelines (missing pruning, avoidable
   shuffles, premature collects) directly delay when fresh training data
   is available — which can push back model retraining cadence and leave
   production models stale longer than intended.
2. **Consistency:** Label leakage (point-in-time correctness failures) is
   a *consistency* problem in disguise — it's the feature pipeline computing
   something that couldn't actually have existed at prediction time,
   producing training data that's subtly inconsistent with real production
   conditions. This is worth naming explicitly as distinct from
   training-serving skew (Ch1/Ch2) — leakage is wrong *training* data;
   skew is a mismatch between *training and serving* computation.
3. **Cost/scale:** Every optimization in this chapter (pruning, broadcast
   joins, avoiding premature collects, caching) is directly a cost lever —
   feature pipelines that run daily or hourly compound small inefficiencies
   into meaningful recurring cloud spend.
4. **Failure mode:** A feature pipeline that "succeeds" but has a
   point-in-time bug doesn't fail loudly — it silently produces a model
   that looks great offline and disappoints in production, which is often
   *harder* to diagnose than an outright pipeline failure, because nothing
   in the pipeline itself errored.

---

## Quick recap

- Common ML feature-engineering patterns in Spark: windowed aggregations
  (shuffle-heavy), multi-source joins (watch broadcast vs. shuffle),
  point-in-time-correct ("as-of") features (leakage risk), careful sampling.
- Efficient training-data reads lean on partition pruning, column pruning,
  avoiding premature `.collect()`, and deliberate caching of reused
  intermediate results.
- Spark's boundary: it prepares data, it doesn't train models — the
  handoff to a training framework typically happens via a clean storage
  boundary (Spark writes Parquet, training framework reads it).
- Label leakage from point-in-time mistakes is a distinct, common,
  hard-to-detect failure mode — worth explicitly separating from
  training-serving skew in how you talk about it.

---

## Interview-style Q&A

**Q: You're computing a "user's total purchases" feature for a churn
model. What's a subtle mistake that could inflate offline model
performance but hurt it in production?**
A: Computing total purchases using all-time data (including purchases that
happened *after* the churn label's cutoff date) rather than only what was
known as of that point in time. This is label leakage — the model
effectively gets to see the future during training, which won't be
available at real prediction time, so offline metrics look artificially
good.

**Q: Where does Spark's responsibility end in an ML training pipeline?**
A: Spark handles distributed data preparation — reading, joining,
aggregating, and computing features at scale — and typically writes the
resulting clean training dataset out to storage (e.g., Parquet). Actual
model training (gradient computation, backprop) is handled by a dedicated
training framework that reads that prepared dataset; Spark itself doesn't
train deep learning models.

**Q: What's a common Spark mistake specific to ML feature pipelines that
you'd watch for?**
A: Calling `.collect()` or `.toPandas()` too early, pulling a large
distributed dataset back onto a single driver machine before it's
necessary — this can crash the driver on large data and defeats the point
of using Spark's distributed processing in the first place.

---

That closes **Module B (Spark)**. Next up is **Module C — Kafka**, starting
with **Ch10: Messaging Systems 101**. Say "ch10" when ready.
