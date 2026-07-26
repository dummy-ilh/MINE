# Chapter 20: Rapid-Fire Interview Q&A

## How to use this chapter

This is a drilling reference, not new material — every answer below is a
compressed "one-minute version" of a full chapter (Ch1–19). If any answer
here feels unfamiliar or thin, that's a signal to go back and reread that
chapter rather than just memorizing the line here. Read through once,
then use it as a self-quiz: cover the answers, look at just the question,
and see if you can produce something close to it unaided.

---

### Module A — Foundations

**Q: What are the five stages of a data pipeline?**
A: Source → Ingest → Store → Process → Serve. Sources are built for
operations, not analytics, which is why the rest of the pipeline exists at
all.

**Q: Batch vs. streaming — how do you decide?**
A: Ask what freshness the actual use case needs. Batch is simpler, cheaper,
higher-latency (minutes to a day+); streaming is lower-latency but more
complex and harder to debug (out-of-order/late data, always-on infra
cost). Don't default to streaming just because it sounds more advanced.

**Q: What is micro-batching, and where does it show up?**
A: Processing small time-windows of data on a short repeating trigger —
gets most of streaming's latency benefit with more of batch's simplicity.
Spark Structured Streaming works this way under the hood.

**Q: Data lake vs. data warehouse vs. lakehouse, in one line each?**
A: Warehouse = structured, schema-on-write, fast SQL, higher cost. Lake =
any data type, schema-on-read, cheap storage, needs a compute engine to
query well. Lakehouse = adds a transactional/schema layer on top of lake
storage to get warehouse-like reliability without a separate system.

**Q: Why is columnar storage (Parquet) preferred for analytical/ML
workloads over row-based (CSV/JSON)?**
A: Analytical queries usually need a subset of columns across many rows —
columnar storage lets you read only those columns (column pruning),
skipping the rest, and compresses much better since similar values sit
together.

**Q: Why is Avro preferred for Kafka messages instead of Parquet?**
A: Streaming reads/writes whole records at a time, so column pruning
doesn't help — Avro's strength instead is schema evolution, letting
producers and consumers stay compatible as the event schema changes over
time.

**Q: What is partition pruning, and what breaks it?**
A: Skipping entire files/folders that don't match a query's filter (e.g.,
`WHERE date = X` skipping other date partitions). Over-partitioning on a
high-cardinality key (like `user_id`) breaks the benefit by creating too
many tiny files, adding overhead that outweighs the pruning gain.

---

### Module B — Spark

**Q: Why does distributed processing exist?**
A: A single machine hits real memory and time ceilings on large-enough
data. Horizontal scaling (many machines) has effectively no ceiling and is
usually more cost-efficient than continuing to scale one machine
vertically.

**Q: Driver vs. executor?**
A: The driver plans and schedules work but doesn't process bulk data
itself. Executors hold data partitions in memory and do the actual
computation, in parallel, each on their own slice.

**Q: What is an RDD, really?**
A: Not a container of data — a lineage graph: a recipe (source + chain of
transformations) for how to (re)produce data. This is what enables fault
tolerance — a lost partition is recomputed from its lineage, not restored
from a backup.

**Q: Transformations vs. actions?**
A: Transformations (map, filter) are lazy — they just extend the lineage
graph, nothing runs yet. Actions (count, collect) trigger actual
execution. Laziness lets Spark optimize the whole chain at once before
running anything.

**Q: What does a DataFrame add over a raw RDD?**
A: A known schema, which unlocks the Catalyst optimizer — automatic
predicate pushdown, column pruning, operation reordering, and join
strategy selection, none of which Spark can do on opaque RDD objects.

**Q: Catalyst vs. Tungsten?**
A: Catalyst decides the efficient query plan (what to do, in what order).
Tungsten executes that plan fast at the memory/CPU level (compact binary
layout, generated code instead of generic interpretation).

**Q: What causes a shuffle, and why is it expensive?**
A: Operations needing data regrouped across partitions by key — groupBy,
join, repartition, distinct, orderBy. Expensive because it requires
physically moving data across the network (write, transfer, re-read)
rather than each partition working independently in place.

**Q: What is data skew, and how do you fix it?**
A: Uneven distribution across a key means one task gets far more data than
the rest — since a job's total time is bounded by its slowest task, one
skewed task can dominate the whole job's runtime. Fixes: salting the key
(spreading a hot key across sub-keys), broadcast joins where applicable,
targeted repartitioning.

**Q: When does Spark use a broadcast join, and why is it faster?**
A: When one side of a join is small enough to fit on every executor — it
copies that small table to all executors instead of shuffling the large
table, avoiding the large table's (much more expensive) shuffle entirely.

**Q: What's a common ML-specific Spark pipeline mistake?**
A: Point-in-time correctness failures — computing a feature using
information that wasn't actually available at the historical label's
timestamp (label leakage), which inflates offline metrics but hurts
production performance. Also: calling `.collect()`/`.toPandas()` too
early, risking a driver OOM crash.

---

### Module C — Kafka

**Q: Why use a message queue instead of direct service-to-service calls?**
A: Decouples producers from consumers (new consumers can be added without
touching the producer), buffers against slow/down consumers, and (for
Kafka specifically) enables replay of historical messages.

**Q: Topics vs. partitions?**
A: A topic is a named stream; it's split into partitions for parallelism
and scale. Ordering is only guaranteed within a single partition, not
across the whole topic.

**Q: What determines which partition a message lands in, and why does it
matter?**
A: Usually a key (e.g., `user_id`) — same key always lands in the same
partition, preserving per-key order. A poorly chosen key can create a hot
partition (same skew problem as Spark, different system).

**Q: What is an offset, and what does it enable?**
A: A message's sequential position within a partition. Consumers track
their own committed offset rather than having messages pushed and
discarded — this is what lets consumers process at independent paces and
replay old messages by resetting their offset backward.

**Q: How does Kafka's fault tolerance differ from Spark's?**
A: Spark recomputes lost data from lineage (no redundant copies needed).
Kafka instead relies on replication — multiple copies of each partition
across brokers — because raw event data has no "recipe" to regenerate it
from if lost.

**Q: Consumer groups — what's the parallelism ceiling?**
A: Partition count. Each partition is read by exactly one consumer within
a group at a time; consumers beyond the partition count sit idle.
Different consumer groups are fully independent, each with their own
offsets, so multiple groups can consume the same topic in parallel without
interfering.

**Q: At-least-once vs. exactly-once — and the practical fix?**
A: At-least-once (Kafka's practical default) never loses messages but can
process one more than once. True exactly-once is costly and requires
end-to-end participation. The practical fix is making processing
idempotent — e.g., overwrite/set instead of increment — so a duplicate
doesn't change the outcome.

**Q: Does a model query Kafka directly?**
A: No — Kafka is a transport/durability layer. A stream processor
materializes results into a proper low-latency store (online feature
store), and that's what the model/serving layer actually queries.

---

### Module D — Orchestration

**Q: Why isn't cron enough for a multi-step pipeline?**
A: Cron only knows wall-clock time, not dependencies — it can't guarantee
a downstream job only runs after an upstream job actually succeeds. This
causes the most dangerous failure mode: a job silently running on stale or
incomplete data with no error raised. Cron also has no built-in retries,
backfills, or cross-job observability.

**Q: What does a DAG give you over a flat list of scheduled scripts?**
A: The actual dependency structure between jobs (including parallel
branches), which an orchestrator can enforce at runtime — only start a job
once its declared dependencies have completed — plus automatic retries,
backfill support, and visibility into what's blocked and why.

**Q: Why does task idempotency matter so much in Airflow?**
A: Automatic retries and backfills mean a task might run more than once
for the same logical unit of work. Non-idempotent tasks (e.g., append-only
writes) turn a recoverable transient failure into silent data duplication;
idempotent tasks (overwrite/upsert) make retries and backfills safe.

**Q: What's a sensor, and why is it different from a regular task?**
A: A sensor polls for an external condition (e.g., a file landing in
storage) rather than performing work directly — it lets a DAG depend on
something outside Airflow's own task graph, not just on other Airflow
tasks.

**Q: Validation vs. monitoring — what's the difference?**
A: Validation checks a single run's data for correctness (schema, nulls,
ranges, referential integrity), ideally early in the DAG to fail fast.
Monitoring tracks pipeline health across time (freshness, volume trends,
schema drift), catching gradual or silent problems no single run's
validation would flag.

**Q: Why doesn't validation alone catch training-serving skew?**
A: Skew is a mismatch between two separate pipelines (training vs.
serving) that can each individually pass validation while still computing
"the same" feature differently. Catching it requires comparing the two
paths against each other or architecturally sharing the transformation
logic (feature stores), not just validating each in isolation.

---

### Module E — Tying it together

**Q: ETL vs. ELT — what changed, and why?**
A: ETL transforms before loading (used when storage/compute were
expensive and coupled). ELT loads raw data first, transforms in place
afterward — became preferred once cloud storage got cheap, since keeping
the raw copy adds flexibility for future, unanticipated use cases. ETL
still applies when data must be cleaned/redacted before it's allowed to
land anywhere (e.g., PII).

**Q: What problem does a feature store actually solve?**
A: Training-serving skew, architecturally — define a feature's
transformation logic once, and populate an offline store (full history,
for training) and an online store (current value, low-latency, for
serving) from that single shared definition, rather than two
independently-implemented pipelines that can quietly diverge.

**Q: Offline store vs. online store?**
A: Offline: full history, point-in-time-correct access, used to build
training datasets. Online: current value only, extremely low-latency
lookup, used for live predictions.

**Q: Does a feature store fully eliminate training-serving skew?**
A: No — it reduces *definitional* skew (divergent logic) but not
*freshness* skew (the online store can still lag behind due to
stream-processing delays), which still requires separate freshness
monitoring.

**Q: In one sentence, why does training-serving skew happen in the first
place?**
A: Because a feature's training-time computation (usually batch, over
history) and serving-time computation (usually real-time, over current
state) are implemented as two separate code paths, and any divergence
between them — however small — produces two different answers for what's
supposed to be the same feature.

---

## Final check — the four questions to ask about anything in this
## syllabus

If you can answer these four questions for *any* concept covered in
Ch1–19, you're in good shape for an interview conversation about it:

1. **Latency:** how does this affect how fresh data/features are?
2. **Consistency:** could this produce a different answer computed twice
   (training vs. serving, retried vs. not)?
3. **Cost/scale:** what happens at 100x the data volume, and what does
   this choice cost?
4. **Failure mode:** if this breaks or lags, what does it look like
   downstream — loud failure, or silent bad output?

That's the whole syllabus. Good luck.
