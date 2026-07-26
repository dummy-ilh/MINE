# Chapter 11: Kafka Core Concepts

## Getting specific

Ch10 gave you the generic producer/broker/consumer model. Now we get
concrete: what does *Kafka specifically* call things, and what are the
mechanisms (topics, partitions, offsets, replication) that make it work?

---

## Topics: the named stream of messages

A **topic** is a named category/feed that messages get published to —
think of it as the name of the "queue" from Ch10. Producers publish to a
topic; consumers subscribe to a topic.

Example: `purchase-events`, `page-views`, `fraud-alerts` would each be
separate topics. A producer publishing a purchase doesn't send it to
"Kafka" generically — it sends it to the `purchase-events` topic
specifically, and only consumers subscribed to that topic receive it.

This mirrors the pattern from Ch1's worked example: the website publishes
purchase/view events to a topic; the nightly Spark job, the fraud model,
and the analytics dashboard all separately subscribe to (consume from)
that same topic.

---

## Partitions: why topics scale

A topic isn't stored as one single ordered log — it's split into
**partitions**, and this is the concept that makes Kafka scale, in exactly
the same spirit as Spark's partitions (Ch5/Ch8) making distributed
processing scale.

```
Topic: purchase-events
  Partition 0:  [msg][msg][msg][msg]...
  Partition 1:  [msg][msg][msg][msg]...
  Partition 2:  [msg][msg][msg][msg]...
```

Each partition is an **ordered, append-only log** — new messages are always
added to the end, and once written, a message's position never changes.
Order is only guaranteed **within a single partition**, not across the
whole topic — messages in partition 0 and partition 1 have no defined
ordering relative to each other.

**Why split into partitions at all?** Two reasons, mirroring Ch5's
motivation for distributed processing generally:
- **Parallelism:** different partitions can be written to and read from
  independently, by different brokers and different consumers, at the
  same time — a topic isn't limited to the throughput of a single machine.
- **Scale:** a topic's total data can exceed what fits (or what's fast to
  process) on a single machine, so spreading it across partitions (and
  therefore across multiple broker machines) is what lets Kafka handle
  very high message volumes.

**How does a message get assigned to a partition?** Typically by a **key**
— e.g., partitioning `purchase-events` by `user_id` means all of a given
user's events always land in the same partition (and therefore stay in
order relative to each other), while different users' events can be spread
across different partitions for parallelism. Messages without a key are
usually distributed round-robin across partitions instead.

**This key-based partitioning choice matters a lot** — it's directly
analogous to choosing a partition key for a data lake (Ch4) or worrying
about skew from an unevenly-distributed key (Ch8). If one user (or one
key) produces a disproportionate share of events, that one partition
becomes a hot spot — same skew problem, different system.

---

## Offsets: how consumers track position

Each message within a partition has a sequential **offset** — 0, 1, 2, 3...
— which is just its position in that partition's log.

A consumer doesn't "receive" messages pushed at it in real time the way you
might picture a phone notification — instead, **the consumer tracks which
offset it has read up to**, and pulls the next messages from there. This
tracked position is often called the consumer's **committed offset**.

Why this design matters:
- **Consumers control their own pace.** A slow consumer can fall behind
  without losing data — the messages just sit in the log waiting, and the
  consumer catches up whenever it's ready (this is the "buffering"
  superpower from Ch10, made concrete).
- **Replay is just "reset the offset."** If a consumer needs to reprocess
  old data (e.g., after fixing a bug, or a brand-new consumer wants
  history), it can simply set its offset back to an earlier point and
  re-read messages it already technically saw before — messages aren't
  deleted just because they've been read (deletion is governed separately,
  by a **retention policy** — e.g., "keep messages for 7 days" — not by
  consumption).
- This offset-based model is precisely what makes Kafka fundamentally
  different from many older/simpler message queues, where a message is
  typically removed once consumed — Kafka instead behaves more like a
  durable, replayable log that multiple independent readers can each
  progress through at their own pace.

---

## Replication: durability basics

Each partition isn't stored on just one broker machine — it's
**replicated** across several (commonly 3 in production), for the same
fault-tolerance reason RDDs recompute lost partitions (Ch6): machines fail,
and you don't want to lose data when they do.

- One replica is the **leader** — all reads and writes for that partition
  go through the leader.
- The other replicas are **followers** — they continuously copy the
  leader's data, staying in sync, ready to take over if the leader fails.
- If the leader broker goes down, Kafka promotes one of the in-sync
  followers to be the new leader, and producers/consumers transparently
  start talking to the new leader — this is what "durability" and "high
  availability" mean concretely in Kafka, not just marketing words.

**Note the difference in fault-tolerance strategy from Spark:** Spark
tolerates failure by *recomputing* lost data from lineage (Ch6) — there's
no need to keep redundant copies, because the source data + transformations
are enough to reproduce it. Kafka instead tolerates failure by *keeping
redundant copies* (replication) — because the data itself (a raw event) has
no "recipe" to recompute it from; once it happened, it happened, and if you
lose all copies, it's gone for good. This is a good contrast to be able to
articulate if asked.

---

## Worked example, tying it together

The `purchase-events` topic, concretely:

- Split into, say, 12 partitions, keyed by `user_id` — so all of a given
  user's purchase history stays in order within one partition, while the
  overall topic's write/read load is spread across 12 partitions'
  worth of parallelism.
- Each partition is replicated 3x across different broker machines — if
  one machine dies, no purchase data is lost; a follower is promoted to
  leader and things continue.
- The nightly Spark job (a consumer) tracks its own offset per partition —
  it processed up through offset 40,000 in partition 3 last night, and
  tonight it resumes from there, reading only the new messages since.
- If a bug is found in the fraud-detection consumer's logic, it can reset
  its offset back a few hours and reprocess that window of purchase events
  with the fixed logic — without needing the website to "resend" anything;
  the data is still sitting right there in the log.

---

## Downstream considerations

1. **Latency:** Consumers reading near the "head" (most recent offset) of
   a partition get near-real-time data; a consumer that's fallen far
   behind is reading "old" data relative to wall-clock time, even though
   the messages themselves are being delivered correctly and in order.
   Monitoring **consumer lag** (how far behind the latest offset a
   consumer is) is the standard way to catch this before it becomes a
   stale-features problem downstream.
2. **Consistency:** Ordering is only guaranteed *within* a partition — if
   an ML feature depends on strict event ordering across a whole topic
   (not just per-user), partitioning by the wrong key can silently
   introduce ordering bugs. Always double check: does my downstream logic
   assume global ordering, when Kafka only promises per-partition
   ordering?
3. **Cost/scale:** More partitions = more parallelism, but also more
   per-partition overhead (similar tradeoff to Spark's partition count,
   Ch8) — and more replicas means more storage cost for the same data
   (3x replication literally triples raw storage need). These are real
   dials, not "always max them out."
4. **Failure mode:** A poorly chosen partition key (e.g., a key with a few
   very hot values) creates a hot partition — same skew problem as Ch8,
   but here it can also mean that one partition's consumer falls behind
   while others keep up, since partitions are consumed somewhat
   independently (more on this exact mechanic in Ch12's consumer groups).

---

## Quick recap

- Topics are named message streams; partitions split a topic into
  ordered, parallel, independently-writable/readable logs — order is only
  guaranteed within a partition.
- A partition key determines which partition a message lands in;
  choosing it well (or badly) directly affects both ordering guarantees
  and skew/hot-partition risk.
- Offsets track each consumer's position in a partition — this is what
  enables independent consumer pacing and replay, and is a fundamentally
  different model from "delete on consume" queues.
- Replication keeps multiple copies of each partition across brokers for
  durability — Kafka's fault-tolerance strategy (redundant copies) versus
  Spark's (recompute from lineage) is a good contrast to know cold.

---

## Interview-style Q&A

**Q: Why does Kafka split a topic into multiple partitions instead of
storing it as one single log?**
A: Partitioning is what enables parallelism and scale — different
partitions can be written to and read from independently, across multiple
brokers and consumers simultaneously, rather than being bottlenecked by a
single machine's throughput. The tradeoff is that Kafka only guarantees
message ordering within a single partition, not across the whole topic.

**Q: How does a Kafka consumer know what to read next, and what does that
enable?**
A: Each consumer tracks a committed offset per partition — its position in
that partition's log — rather than having messages pushed at it and
discarded. This lets consumers process at their own pace independently,
and lets a consumer replay old messages simply by resetting its offset
backward, since messages aren't deleted on consumption (deletion is
governed separately by a retention policy).

**Q: How does Kafka's approach to fault tolerance differ from Spark's?**
A: Spark tolerates failure by recomputing lost data from its lineage — the
recipe of source + transformations is enough to reproduce a lost
partition. Kafka instead relies on replication — keeping multiple copies
of each partition across different brokers — because raw event data has
no "recipe" to regenerate it from if all copies are lost.

---

Next: **Ch12 — Consumer Groups & Delivery Semantics** (parallel consumption,
at-least-once vs. exactly-once). Say "ch12" when ready.
