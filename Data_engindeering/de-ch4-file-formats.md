# Chapter 4: File Formats & Storage Layout

## Why this chapter matters more than it sounds like it should

This feels like the most "boring" chapter so far — just file formats,
right? But this is the chapter that most directly explains **why one
Spark job takes 2 minutes and a seemingly similar one takes 2 hours**, and
it's a very common place for interviewers to probe whether you actually
understand what's under the hood, versus just knowing tool names.

---

## Row-based vs. columnar storage — the core idea

Imagine a table:

| user_id | age | country | purchase_amount |
|---|---|---|---|
| 1 | 25 | US | 40 |
| 2 | 31 | UK | 15 |
| 3 | 29 | US | 80 |

**Row-based storage** writes this to disk row by row:
```
1,25,US,40 | 2,31,UK,15 | 3,29,US,80
```
(CSV and JSON work this way.)

**Columnar storage** writes it column by column instead:
```
1,2,3 | 25,31,29 | US,UK,US | 40,15,80
```
(Parquet and ORC work this way.)

### Why this distinction matters — plain language

Think about the query: *"What's the average purchase_amount for US
users?"*

- **Row-based:** to answer this, you must read every full row (all four
  columns), even though you only actually need 2 of the 4 columns
  (`country`, `purchase_amount`). You're reading a lot of data you'll
  immediately throw away.
- **Columnar:** you can read *only* the `country` column and the
  `purchase_amount` column, skipping `user_id` and `age` entirely. For a
  table with 4 columns this saves 50%; for a real table with 200 columns
  and you only need 3, you just avoided reading 98.5% of the data.

This single idea — **read only the columns you need** — is called **column
pruning**, and it's why analytical/ML workloads (which usually aggregate or
select specific columns over huge tables) overwhelmingly prefer columnar
formats. Row-based formats win when you need to read/write *entire records*
frequently (e.g., a live application updating one user's full profile) —
that's a transactional/OLTP pattern, not an analytical one.

### Compression is a side benefit, and it's a big one

Columnar layout also compresses far better, because similar values are
stored next to each other (e.g., a column of `country` values like
`US,US,US,UK,US,UK...` compresses much better than that same data
scattered between unrelated row values). This is part of why Parquet files
are often 5-10x smaller than the equivalent CSV.

---

## Parquet vs. Avro — the two you'll actually hear named

Both are common in the Spark/Kafka world, but for different jobs:

**Parquet — columnar, built for analytical reads**
- Used for: data lake storage, Spark DataFrame reads, ML training data.
- Optimized for "read a few columns out of a huge table."
- Self-describing (embeds its own schema in the file).

**Avro — row-based, built for fast writes and schema evolution**
- Used for: Kafka messages, streaming pipelines, anywhere you're writing
  one full record at a time.
- Row-based because in streaming, you're producing/consuming one complete
  event at a time — you don't get the "only read some columns" benefit
  because you always need the whole record anyway.
- Has strong support for **schema evolution** — the same field can gain a
  new optional field over time without breaking old consumers, which
  matters a lot in Kafka where producers and consumers get deployed at
  different times and must stay compatible with each other.

**The one-line pattern to remember:** *streaming/event data (write-heavy,
whole-record access) → Avro; analytical/lake data (read-heavy, partial-
column access) → Parquet.* This is also why it's completely normal for the
same pipeline to use both: Kafka events in Avro → Spark job reads them,
processes, writes the result out as Parquet into the lake.

---

## Partitioning on disk

Separate from Spark's *in-memory* partitioning (that's Ch8) — this is about
how files are physically organized in the lake/warehouse.

A partitioned lake layout looks like:
```
/orders/
  date=2026-07-23/
    part-0001.parquet
    part-0002.parquet
  date=2026-07-24/
    part-0001.parquet
  date=2026-07-25/
    part-0001.parquet
```

If a query filters `WHERE date = '2026-07-24'`, the engine can skip
reading the `2026-07-23` and `2026-07-25` folders entirely — it never even
opens those files. This is called **partition pruning**, and combined with
column pruning (above), it's the main reason a well-laid-out lake query can
be dramatically faster than a naively-organized one, without changing any
compute engine at all.

**Choosing a partition key** is a real design decision, not a formality:
- Partition by something commonly filtered on (date is the classic choice
  for event data — most queries/jobs naturally filter to "yesterday" or
  "last 7 days").
- Don't over-partition: partitioning by something high-cardinality (like
  `user_id`, with millions of distinct values) creates millions of tiny
  files, which is *slower* — the overhead of opening many small files
  outweighs the benefit of pruning. This is a real, commonly-cited gotcha.

---

## Worked example, tying it together

Back to the recommendation pipeline: the nightly Spark job reads Kafka
events (stored as Avro), processes them, and writes co-purchase stats out
as Parquet, partitioned by `date`.

- **Why Avro in Kafka:** events are produced one at a time by many
  producers (website servers) over time, and the event schema might grow
  new optional fields (e.g., adding a `device_type` field later) — Avro
  handles this evolution gracefully without breaking existing consumers.
- **Why Parquet in the lake:** the Spark job later needs to scan months of
  history but only cares about `user_id`, `product_id`, `action` — column
  pruning means it skips reading anything else.
- **Why partition by date:** because both the nightly job ("process
  yesterday's events") and ad-hoc analyst queries ("show me last week")
  naturally filter on date — partition pruning avoids scanning the entire
  history every single run.

---

## Downstream considerations

1. **Latency:** Poor file layout (no partitioning, wrong format) doesn't
   cause *incorrect* results — it causes jobs to take much longer, which
   can push a "should be a 20 minute nightly job" into "still running when
   the next day's data needs to be ready," cascading delays downstream.
2. **Consistency:** Format mismatches (Avro schema evolves in a
   breaking way — e.g., a field's *type* changes rather than just adding
   an optional field) can cause a consumer to fail to parse messages it
   used to handle fine. Since ML pipelines often have several consumers
   (training pipeline, feature pipeline, monitoring) reading the same
   stream, an uncoordinated schema change can silently break just one of
   them.
3. **Cost/scale:** Storing raw JSON logs at scale, uncompressed and
   row-based, can cost meaningfully more in both storage *and* compute
   (every scan reads everything) than converting to partitioned Parquet —
   this is a very common, very concrete "easy win" cost optimization in
   real ML data platforms.
4. **Failure mode:** Over-partitioning (too many tiny files) doesn't fail
   loudly — it just makes everything slowly, mysteriously slower as file
   listing/open overhead dominates. This is a classic "why is my Spark job
   suddenly so slow" root cause worth naming in an interview.

---

## Quick recap

- Columnar (Parquet) formats let you skip reading unneeded columns —
  huge win for analytical/ML workloads. Row-based (Avro, CSV, JSON) formats
  are better for whole-record, write-heavy workloads like streaming.
- Avro's schema evolution support is why it's the default for Kafka events;
  Parquet's column pruning + compression is why it's the default for lake/
  training data.
- Partitioning data on disk (commonly by date) lets queries skip entire
  files via partition pruning — but over-partitioning on high-cardinality
  keys backfires by creating too many tiny files.
- A single pipeline commonly uses both formats: Avro for the streaming
  leg, Parquet for the lake/training leg.

---

## Interview-style Q&A

**Q: Why is Parquet generally preferred over CSV/JSON for ML training
data?**
A: Parquet is columnar, so a training job that only needs a subset of
columns (e.g., specific features) can skip reading the rest entirely,
saving both I/O and time. It also compresses much better than row-based
text formats, reducing storage cost and read time further.

**Q: Why would you use Avro instead of Parquet for Kafka messages?**
A: Kafka events are produced and consumed one whole record at a time, so
Parquet's column-pruning benefit doesn't apply — there's no "partial
record" read pattern in streaming. Avro's strength instead is schema
evolution: producers and consumers are deployed independently over time,
and Avro lets the event schema add fields without breaking existing
consumers.

**Q: What's a partitioning mistake you'd watch for when designing lake
storage for an ML pipeline?**
A: Partitioning by a high-cardinality field (like `user_id`) instead of
something like `date` — this creates a huge number of very small files,
and the overhead of opening/listing all those files ends up slower overall
than not partitioning at all, even though partitioning is "supposed" to
speed things up.

---

Next: **Ch5 — Why Distributed Processing Exists**, the on-ramp into the
Spark module. Say "ch5" when ready.
