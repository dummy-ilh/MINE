# Chapter 12: Consumer Groups & Delivery Semantics

## Two separate questions this chapter answers

1. How does Kafka let multiple consumers process a topic **in parallel**
   without duplicating work?
2. When something goes wrong (a crash, a retry), what guarantee do you
   actually get about whether a message was processed **zero, one, or more
   times**?

These are genuinely different concerns, and interviewers often test them
separately — don't blend them together.

---

## Consumer groups: how parallel consumption works

Recall from Ch11: a topic is split into partitions. A **consumer group** is
a named set of consumer instances that **cooperate to consume a topic
together**, with Kafka automatically assigning each partition to exactly
one consumer *within that group*.

```
Topic: purchase-events (6 partitions)

Consumer Group "feature-pipeline":
  Consumer A  ← partitions 0, 1
  Consumer B  ← partitions 2, 3
  Consumer C  ← partitions 4, 5
```

- Each partition is read by **exactly one consumer in the group at a
  time** — this is what prevents duplicate processing *within* a group:
  two consumers in the same group never read the same partition
  simultaneously.
- If you add a 4th consumer to this group, Kafka **rebalances** —
  redistributing partitions so the new consumer gets some too (e.g., 0-1,
  2, 3, 4-5). If a consumer dies, its partitions get reassigned to the
  remaining consumers in the group.
- **This is your parallelism knob:** the maximum useful parallelism within
  one consumer group is capped by the number of partitions — 6 partitions
  means at most 6 consumers can be doing useful work simultaneously in that
  group; a 7th consumer would sit idle with nothing assigned.

### Multiple groups = independent, parallel "views" of the same topic

Different consumer groups are completely independent of each other — each
group tracks its **own** offsets per partition. This is how the Ch1/Ch10
pattern works: the `feature-pipeline` group, the `fraud-detection` group,
and the `analytics` group can all separately consume the *entire*
`purchase-events` topic, each at their own pace, without interfering with
each other at all — because "has this message been read" is tracked
per-group, not globally.

**One-sentence summary:** consumer groups give you parallelism *within* a
group (partitions split across group members) and independence *across*
groups (each group gets its own full, independent read of the topic).

---

## Delivery semantics: what you're actually promised

This is where things get genuinely tricky, and it's a favorite interview
topic because the "obvious" answer (exactly-once) is usually not free.

**At-most-once:** a message is delivered **zero or one** times — it might
get lost, but it will never be processed twice. (Consumer commits its
offset *before* processing — if it crashes mid-processing, that message is
just skipped on restart.) Rarely what you actually want for anything that
matters.

**At-least-once:** a message is delivered **one or more** times — it will
never be silently lost, but it might get processed twice. (Consumer
processes the message *first*, then commits its offset — if it crashes
after processing but before committing, it'll reprocess that same message
on restart.) This is Kafka's practical default and the one you'll work
with most.

**Exactly-once:** a message is delivered and processed **exactly one**
time, no duplicates, no loss. Kafka does support exactly-once semantics
(via transactional producers/consumers) in certain configurations, but it
comes with real cost — more coordination overhead, reduced throughput, and
it generally only holds end-to-end if *every* system in the pipeline
participates in the same transactional guarantee. In practice, many teams
find it easier (and often just as correct in effect) to accept
at-least-once delivery and make their processing **idempotent** instead.

---

## Idempotency: the practical fix for at-least-once

**Idempotent** means: processing the same message multiple times produces
the **same end result** as processing it once. If your consumer logic is
idempotent, at-least-once delivery becomes just as safe as exactly-once —
duplicates simply don't matter anymore, because reprocessing one doesn't
change the outcome.

Concrete patterns for making processing idempotent:
- **Use the message's natural identity as a deduplication key.** E.g.,
  writing "user 123 purchased product 456 at timestamp T" as an upsert
  keyed on `(user_id, product_id, timestamp)` rather than a blind
  `INSERT` — processing the same event twice just overwrites the same row
  with the same values, instead of creating a duplicate row.
- **Prefer "set to X" over "increment by X."** `total_purchases = 40` is
  idempotent (writing it twice gives the same result); `total_purchases +=
  1` is not (writing it twice double-counts). This is a very concrete,
  interview-friendly example.
- **Track processed message IDs** and skip reprocessing ones you've
  already handled, if the downstream operation genuinely can't be made
  naturally idempotent.

**This directly echoes Ch6's determinism requirement for Spark's fault
tolerance** — same underlying idea, different system: distributed systems
retry work after failures, and your processing logic needs to tolerate
being run more than once on the same input.

---

## Worked example

The `feature-pipeline` consumer group reads `purchase-events` and updates a
"user's rolling 30-day purchase count" feature in a feature store.

- **Parallelism:** if the topic has 12 partitions and the group has 4
  consumer instances, each consumer handles 3 partitions — 4x the
  throughput of a single consumer, with Kafka handling the
  partition-to-consumer assignment automatically.
- **Delivery semantics in practice:** the consumer processes a message
  (updates the feature) and *then* commits its offset. If it crashes
  between those two steps, it'll reprocess that same purchase event on
  restart — at-least-once, meaning this update could theoretically run
  twice for the same event.
- **Idempotency fix:** instead of `rolling_count += 1` (which would
  double-count on a duplicate), the consumer recomputes and **sets** the
  rolling count from a windowed query/state (e.g., "count of purchases in
  the last 30 days as of now"), so reprocessing the same event twice
  naturally produces the same final value both times.

---

## Downstream considerations

1. **Latency:** Consumer group rebalancing (when a consumer joins/leaves)
   briefly pauses partition consumption while reassignment happens — this
   is a real, measurable latency blip worth knowing about if asked "what
   happens when you scale up a Kafka consumer group."
2. **Consistency:** Non-idempotent processing under at-least-once delivery
   is a genuine, common source of subtle data bugs — e.g., a feature that
   slowly drifts upward over time because of occasional duplicate
   processing after crashes/retries, which can be very hard to notice
   until someone audits the numbers against ground truth.
3. **Cost/scale:** Adding more consumers to a group only helps parallelism
   up to the partition count — beyond that, extra consumers are pure waste
   (idle, doing nothing), which directly echoes the "too many
   executors/partitions" theme from Ch5/Ch8. If you need more parallelism
   than your partition count allows, you actually need to increase
   partitions, not just add consumers.
4. **Failure mode:** At-most-once delivery (rare, but exists in some
   configurations) can silently drop data on any crash — a genuinely
   dangerous default for anything feeding an ML feature pipeline, since a
   dropped event just quietly never shows up rather than raising any
   alarm.

---

## Quick recap

- Consumer groups split a topic's partitions across their member
  consumers for parallelism; different groups are fully independent of
  each other, each tracking their own offsets.
- Parallelism within a group is capped by partition count — extra
  consumers beyond that sit idle.
- At-least-once (Kafka's practical default) means messages are never
  lost but can be processed more than once; exactly-once exists but is
  costly and requires end-to-end participation.
- Idempotent processing (e.g., upsert-by-key, "set" instead of
  "increment") makes at-least-once delivery just as safe as exactly-once
  in practice, without the coordination overhead.

---

## Interview-style Q&A

**Q: If you add more consumers to a Kafka consumer group than the topic
has partitions, what happens?**
A: The extra consumers beyond the partition count sit idle — each
partition can only be assigned to one consumer within a group at a time,
so partition count is the hard ceiling on useful parallelism for that
group. To get more parallelism, you'd need to increase the topic's
partition count, not just add more consumers.

**Q: Why is at-least-once delivery usually preferred over trying to
guarantee exactly-once?**
A: True exactly-once semantics require significant coordination overhead
and typically need every system in the pipeline to participate in the same
transactional guarantee, which reduces throughput and adds complexity. In
practice, making your consumer logic idempotent achieves the same
practical safety as exactly-once — reprocessing a duplicate under
at-least-once delivery simply produces the same result — without that
overhead.

**Q: Give an example of making a feature-update operation idempotent.**
A: Instead of incrementing a counter (`count += 1`, which would
double-count on a retried/duplicate message), recompute and set the value
directly from a windowed query (`count = number of purchases in the last
30 days as of now`) — reprocessing the same event twice then yields the
same final value both times, since it's an overwrite rather than an
accumulation.

---

Next: **Ch13 — Kafka in an ML Context**, the last chapter of Module C. Say
"ch13" when ready.
