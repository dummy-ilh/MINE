# Chapter 13: Kafka in an ML Context

## Closing out Module C

Chapters 10–12 covered Kafka's mechanics from first principles. This
chapter is the payoff, same shape as Ch9 for Spark: how does this actually
show up in ML systems, and where does Kafka's job end?

---

## Feeding real-time features into a feature store

This is the most common ML-adjacent Kafka pattern. Recall the feature
store concept teased in Ch1 and Ch18's preview: the same feature needs to
be available both for training (historical, batch) and for serving (live,
low-latency), and they need to agree with each other.

The typical real-time path:
```
Website → Kafka topic → Stream processor → Online feature store (low-latency)
                      ↘ Data lake (Parquet)  → Offline feature store (for training)
```

- Kafka is the **entry point** for real-time events into this whole
  system — it's not itself the feature store, it's the durable, replayable
  pipe that both the real-time path (online store) and the batch path
  (offline store, via the lake) branch off from.
- Writing the *same* topic to both paths (rather than computing the
  feature two separate ways in two separate places) is exactly how you
  avoid the training-serving skew problem raised back in Ch1/Ch2 — one
  source of truth for the raw event, even if it's processed slightly
  differently for the two destinations.

## Streaming model input vs. batch retraining

Two genuinely different ML use cases both touch Kafka, and it's worth
keeping them mentally separate:

**1. Real-time inference input** (e.g., ranking search results, live fraud
scoring): the model needs a *feature value* that reflects very recent
activity — "how many times has this user searched in the last 5 minutes?"
This is where a stream processor (Kafka Streams, ksqlDB, or Spark
Structured Streaming from Ch2) consumes the Kafka topic continuously,
maintains a rolling aggregate, and makes it queryable with low latency at
prediction time. The *model itself* isn't reading Kafka directly — a
serving layer/feature store sits in between, and the model queries that.

**2. Batch retraining trigger/input** (e.g., "retrain the recommendation
model weekly on the last 30 days of purchase events"): here Kafka is
really just acting as a durable log that eventually lands in the lake
(Ch1's worked example) — the actual retraining is a Spark batch job (Ch9)
reading from Parquet, not from Kafka directly. Kafka's role here is
upstream and indirect: it's how the raw data *got into* the lake in the
first place.

**The pattern to recognize:** Kafka is almost never something a model
queries directly at prediction time. It's the transport/durability layer
that feeds both a low-latency online store (for #1) and a lake (for #2) —
the model (or its serving layer) talks to those downstream systems, not to
Kafka itself.

---

## Kafka Streams / ksqlDB vs. "just pipe into Spark"

Two ways to process a Kafka stream, worth knowing the tradeoff between,
even at a conceptual level:

**Kafka Streams / ksqlDB:** lightweight stream-processing libraries built
specifically for Kafka, running as a normal application (no separate
cluster needed beyond Kafka itself). Good fit for simpler, focused
transformations directly on a topic — e.g., "maintain a rolling count per
user" — with lower operational overhead than standing up a full Spark
cluster.

**Piping into Spark (Structured Streaming, from Ch2):** better fit when
you need heavier processing — complex joins across multiple topics/tables,
integration with the same Spark-based batch pipelines already used for
training data prep (Ch9), or when your team already has Spark expertise
and infrastructure and doesn't want a second, separate streaming stack to
maintain.

**Interview framing:** this is a "simple, dedicated tool vs. heavier,
more powerful, already-in-use tool" tradeoff — the same shape of decision
you'll see in lots of infra choices. Neither is universally "correct";
it depends on the complexity of the transformation and what's already in
your stack.

---

## Worked example, tying Module C together

Full real-time path for the "purchases in last 10 minutes" fraud feature
from Ch2:

1. Website (**producer**) publishes purchase events to the
   `purchase-events` **topic** (Ch11), partitioned by `user_id` so a
   given user's events stay ordered.
2. Kafka Streams (a lightweight consumer) continuously maintains a rolling
   10-minute purchase count per user, writing the result into a
   low-latency **online feature store** (e.g., Redis) — this is the
   real-time inference path.
3. Separately, a **consumer group** (Ch12) reads the same topic and lands
   raw events into the data **lake** as Parquet (Ch1/Ch4) — this feeds the
   nightly Spark job (Ch9) that builds the **offline** training dataset.
4. Both paths originate from the *same* Kafka topic — a single source of
   truth — reducing (though not eliminating) the risk that "purchases in
   last 10 minutes" is computed inconsistently between training and
   serving.
5. The consumer's feature-update logic is **idempotent** (Ch12) — if it
   crashes and reprocesses a few events, the rolling count is recomputed
   from source rather than incremented, so duplicates don't corrupt the
   feature.

---

## Downstream considerations

1. **Latency:** The real-time path (Kafka → stream processor → online
   store) is built specifically to minimize latency to milliseconds/
   seconds; the batch path (Kafka → lake → Spark) accepts hours of latency
   in exchange for simpler, cheaper, more thorough processing. Knowing
   which latency budget a given feature actually needs (back to Ch2) is
   what determines which path — or both — you build.
2. **Consistency:** This whole chapter is really an extended answer to
   "how do you avoid training-serving skew in practice" — the single-topic,
   dual-consumer pattern (one topic feeding both an online and offline
   store) is the standard architectural answer, though it only actually
   works if both consumers compute the "same" feature the same way, which
   requires real discipline (often solved by feature-store tooling
   specifically designed to share transformation logic between the two
   paths — more in Ch18).
3. **Cost/scale:** Running both a stream processor and a batch pipeline
   off the same topic is more infrastructure than just picking one — worth
   being able to justify it's worth the cost specifically because the
   *latency requirements differ* between training and serving use cases,
   not because "more infrastructure is inherently better."
4. **Failure mode:** If the stream processor (online path) falls behind or
   crashes, live features go stale, degrading real-time predictions
   immediately and visibly. If the batch path lags, training data is
   stale for the *next* retraining cycle — a slower-burning, easier-to-miss
   problem. Both are real, but they fail on very different timescales and
   warrant different monitoring.

---

## Quick recap

- Kafka is the durable, replayable entry point that typically feeds
  *both* a low-latency online feature store (for real-time inference) and
  a data lake (for offline/batch training data) — the shared source is
  what helps limit training-serving skew.
- Models/serving layers query the online store, not Kafka directly;
  training jobs query the lake, not Kafka directly — Kafka is transport,
  not the destination.
- Kafka Streams/ksqlDB suit lightweight, dedicated stream transformations;
  piping into Spark Structured Streaming suits heavier processing or
  teams already invested in a Spark-based stack.
- Real-time and batch paths off the same topic fail on very different
  timescales — worth monitoring both distinctly.

---

## Interview-style Q&A

**Q: Does an ML model typically query Kafka directly at prediction time?**
A: No — Kafka is a transport/durability layer, not a queryable store
optimized for low-latency point lookups. A stream processor consumes the
Kafka topic and materializes the result into a proper low-latency store
(e.g., Redis, an online feature store), and that's what the model or its
serving layer actually queries.

**Q: How does feeding both an online and offline feature store from the
same Kafka topic help with training-serving skew?**
A: It ensures both paths start from the exact same raw event data — one
source of truth — rather than the online and offline systems separately
capturing or defining "the same" event in potentially inconsistent ways.
It doesn't eliminate skew risk entirely (the transformation logic on each
path can still diverge), but it removes one common source of it.

**Q: When would you reach for Kafka Streams/ksqlDB instead of Spark
Structured Streaming?**
A: For simpler, more focused transformations directly on one or a few
Kafka topics, where standing up a full Spark cluster would be unnecessary
operational overhead. Spark Structured Streaming makes more sense for
heavier processing — complex multi-source joins, or when the team already
runs Spark for batch training-data pipelines and wants to reuse that
infrastructure and expertise rather than maintaining a separate stack.

---

That closes **Module C (Kafka)**. Next up is **Module D — Orchestration**,
starting with **Ch14: Why Orchestration Is Its Own Problem**. Say "ch14"
when ready.
