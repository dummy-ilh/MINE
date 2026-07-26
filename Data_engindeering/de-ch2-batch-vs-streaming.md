# Chapter 2: Batch vs. Streaming

## Why this is the first real "decision" in the pipeline

Back in Ch1, "Ingest" and "Process" were just boxes. This chapter is about
the single biggest fork in the road for both of them: **do you handle data
in bounded chunks on a schedule, or as an unbounded, continuous flow?**
Almost every other decision in this syllabus (which storage system, which
processing engine, how fresh your features are) flows downstream from this
one choice.

---

## What batch processing is

**Batch = you wait until you have a defined, bounded set of data, then
process all of it at once.**

Plain-language version: imagine doing laundry once a week instead of
washing each item the moment it gets dirty. You collect a "batch" (a week's
worth of clothes), then run one big load.

Concretely in data terms:
- A job runs on a **schedule** (e.g., every night at 2am) or is **triggered**
  by an event (e.g., "the upstream file finally landed").
- It reads a **fixed, known set of data** — "yesterday's orders table" —
  process it, and produce output.
- Once the job finishes, that batch is done. The next batch is a completely
  separate run.

**Where you've already seen this:** the nightly Spark job in Ch1's worked
example that joins events with product metadata is a batch job.

## What streaming processing is

**Streaming = you process each piece of data (or a tiny "micro-batch" of a
second/minutes' worth) as soon as it arrives, continuously, forever.**

Plain-language version: instead of a weekly laundry pile, you wash each item
the instant it gets dirty. There's no "batch" — there's just an endless
stream of individual events, and your system reacts to each one as it comes.

Concretely:
- Data is treated as **unbounded** — it never "finishes." The job doesn't
  have a start/end, it just runs continuously, always waiting for the next
  event.
- Processing happens with very low latency — milliseconds to seconds after
  an event occurs.
- The Kafka events from Ch1 ("user viewed product X") are naturally a
  stream — they happen continuously, with no natural "batch boundary."

---

## The core tradeoff: latency vs. throughput/cost/simplicity

| | Batch | Streaming |
|---|---|---|
| **Latency** | Minutes to a day+ (however often it runs) | Milliseconds to seconds |
| **Throughput per unit cost** | High — big jobs process huge volumes efficiently | Lower — always-on infra is more expensive per record processed |
| **Correctness / debugging** | Easier — a batch is a known, fixed input; reruns are reproducible | Harder — out-of-order events, late-arriving data, need to handle "what if this event shows up 3 minutes late" |
| **Infra complexity** | Simpler — schedule a job, it runs, it ends | More complex — always-on system, state management, backpressure handling |
| **Good for** | Training data prep, daily reports, model retraining | Fraud detection, real-time recommendations, live dashboards, alerting |

The one-line version interviewers want to hear: **streaming buys you low
latency, and you pay for it with system complexity and harder correctness
guarantees.** Never reach for streaming just because it sounds more
impressive — it's the wrong tradeoff for most ML problems.

---

## Micro-batching: the practical middle ground

In practice, "pure" streaming (react to literally every single event,
one at a time) is rare. Most production streaming systems actually use
**micro-batching**: collect events for a very short window (say, every 1–10
seconds), then process that tiny batch. This gets you *most* of the latency
benefit of streaming with *some* of the simplicity of batch.

- **Spark Structured Streaming** works this way under the hood — it's
  literally the same Spark DataFrame engine from Module B, just triggered
  repeatedly on small windows of new data instead of one big historical
  scan.
- This is why understanding Spark's DataFrame model (Ch7) pays off in both
  the batch and streaming world — it's genuinely the same mental model
  applied at different time scales.

---

## Worked example: same feature, batch or streaming

Let's take one concrete feature an ML system might need: **"has this user
made a purchase in the last 10 minutes?"** (useful for e.g. detecting
account takeover / fraud right after a suspicious login).

- **Batch approach:** A nightly job scans yesterday's `orders` table and
  computes "purchases in last 10 minutes" — except this is nonsensical for
  batch, because by the time the job runs, "last 10 minutes" from a day ago
  is stale and useless. **This is the tell that a feature genuinely needs
  streaming**, not just "streaming would be nice."
- **Streaming approach:** A Kafka consumer watches the purchase event
  stream, maintains a small rolling window of each user's last 10 minutes
  of activity, and can answer "has this user purchased in last 10 min?" the
  instant it's asked. This is the right tool for this specific feature.

Contrast with: **"average customer lifetime value"** — this changes slowly,
so recomputing it nightly in batch is not just acceptable but *preferable*
(cheaper, simpler, easier to debug) — streaming would be over-engineering.

**The interview skill here isn't memorizing definitions — it's being able
to look at a feature/requirement and say which one it obviously needs, and
why.**

---

## Downstream considerations (applying Ch1's four questions)

1. **Latency:** This chapter *is* the latency decision. Ask: what's the
   actual business/ML requirement for freshness? Don't assume — "real-time"
   is thrown around loosely; often "within an hour" is actually fine.
2. **Consistency:** This is where **training-serving skew risk is born.**
   If your training data is computed via a nightly batch job but your live
   model queries a streaming-computed feature, you now have two different
   code paths computing (hopefully) the same logic. This is one of the
   most common real causes of skew — worth stating explicitly in an
   interview if it comes up.
3. **Cost/scale:** Streaming infrastructure (Kafka clusters, always-on
   consumers) costs money 24/7 whether or not there's traffic. Batch jobs
   only cost money while they run. At low data volume, streaming can
   actually be *more* expensive relative to the value it provides.
4. **Failure mode:** A failed batch job is usually easy to detect and
   simply rerun (idempotency permitting — more in Ch15–16). A failed or
   lagging streaming consumer can silently fall behind, and by the time
   you notice, you may have a backlog of stale/unprocessed events to catch
   up on — a much messier recovery.

---

## Quick recap

- Batch = bounded, scheduled, wait-then-process-all-at-once. Simpler,
  cheaper, higher-latency.
- Streaming = unbounded, continuous, process-as-it-arrives. Lower latency,
  more complex, harder correctness guarantees.
- Micro-batching (e.g., Spark Structured Streaming) is the realistic middle
  ground most production systems actually use.
- Decide based on the actual freshness requirement of the specific feature
  or use case — not on which sounds more sophisticated.
- Mixing batch (training) and streaming (serving) paths for the "same"
  feature is a leading cause of training-serving skew.

---

## Interview-style Q&A

**Q: When would you choose batch over streaming for an ML pipeline, even if
low latency were free/easy to get?**
A: When correctness and reproducibility matter more than freshness — e.g.,
training data prep, where you want a fixed, auditable, easily-rerunnable
snapshot rather than a moving target. Also whenever the underlying feature
genuinely doesn't change quickly (e.g., a user's account age), so freshness
beyond daily buys nothing.

**Q: What's a concrete symptom that would tell you a system has
training-serving skew rooted in a batch/streaming mismatch?**
A: The model performs well in offline evaluation (trained/tested on
batch-computed features) but degrades in production, and the drop is worse
for time-sensitive features specifically — a strong signal that the
real-time (streaming) computation of a feature diverges from how it was
computed in the batch training pipeline.

**Q: Is Spark Structured Streaming "true" streaming?**
A: Not in the strictest sense — it's micro-batching: it processes small
windows of new data on a very short repeating trigger, using the same
underlying batch DataFrame engine. It gets you low-latency behavior without
needing a fundamentally different processing model, which is exactly why
it's popular.

---

Next: **Ch3 — Data Lake vs. Data Warehouse vs. Lakehouse.** Say "ch3" when
ready.
