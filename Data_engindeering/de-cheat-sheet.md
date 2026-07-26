# Data Engineering Basics — Day-Before Cheat Sheet

One page of scannable signal per module. If a line doesn't ring a bell,
that's your flag to skim that chapter tonight, not tomorrow morning.

---

## The 4 questions to apply to anything (use these to fill gaps live)
1. **Latency** — how fresh does this need to be?
2. **Consistency** — could this give a different answer computed twice?
3. **Cost/scale** — what breaks at 100x volume?
4. **Failure mode** — loud crash, or silent bad output?

---

## Module A — Foundations

| Concept | One-liner |
|---|---|
| Pipeline | Source → Ingest → Store → Process → Serve |
| Batch | Bounded, scheduled, simple, cheap, high(er) latency |
| Streaming | Unbounded, continuous, low latency, complex, costly always-on |
| Micro-batch | Small time windows on a fast repeating trigger (Spark Structured Streaming) |
| Warehouse | Structured, schema-on-write, fast SQL, pricier |
| Lake | Any data type, schema-on-read, cheap, needs compute engine to query |
| Lakehouse | Transaction log + optional schema on top of lake files = warehouse reliability, lake cost |
| Columnar (Parquet) | Read only needed columns (column pruning) + great compression → analytics/ML |
| Row-based (Avro) | Whole-record read/write, strong schema evolution → Kafka/streaming |
| Partition pruning | Skip whole files/folders that don't match filter (e.g. date=) |
| Over-partitioning trap | High-cardinality key (user_id) → too many tiny files → slower, not faster |

**Reflex line:** "Sources are built for operations, not analytics — that mismatch is why pipelines exist."

---

## Module B — Spark

| Concept | One-liner |
|---|---|
| Why distributed | Single machine hits memory/time ceilings → horizontal > vertical scaling |
| Driver | Plans/schedules, doesn't process bulk data |
| Executor | Holds partitions, does the actual work, in parallel |
| RDD | NOT stored data — a lineage graph (recipe: source + transforms) |
| Transformations | map/filter — LAZY, just extend the recipe |
| Actions | count/collect — trigger real execution |
| Why lazy | Lets Spark optimize the *whole chain* before running anything |
| Fault tolerance | Recompute lost partition from lineage — requires deterministic, side-effect-free transforms |
| DataFrame | RDD + schema → unlocks Catalyst optimizer |
| Catalyst | Plans: predicate pushdown, column pruning, reordering, join strategy |
| Tungsten | Executes fast: off-heap binary layout, generated code |
| Shuffle | Data moves across network to regroup by key — triggered by groupBy/join/repartition/distinct/orderBy — usually THE cost |
| Skew | Uneven key distribution → one slow task bottlenecks whole job (job time = slowest task) |
| Skew fixes | Salting the key, broadcast join, targeted repartition |
| Broadcast join | Small table copied to every executor → avoids shuffling the big table |
| Label leakage | Feature uses info not available as-of the label's timestamp → inflated offline metrics |
| Spark boundary | Prepares data (ETL/features); does NOT train models — hands off via storage (Parquet) to PyTorch/TF |

**Reflex line:** "Shuffle = data physically moves across the network — usually the dominant cost. Job finishes when the SLOWEST task finishes, which is why skew is so damaging."

---

## Module C — Kafka

| Concept | One-liner |
|---|---|
| Why a queue | Decouples producer/consumer, buffers slow consumers, enables replay |
| Producer/Broker/Consumer | Publish / store+route / read — producer & consumer never talk directly |
| Topic | Named stream |
| Partition | Ordered, append-only log; topic split for parallelism/scale |
| Ordering | Guaranteed WITHIN a partition only, never across the whole topic |
| Partition key | Same key → same partition (order preserved per key); bad key choice → hot partition (skew, Kafka-flavored) |
| Offset | Message's position; consumer tracks its OWN offset (pull model) |
| Replay | Just reset the offset backward — data isn't deleted on read (retention policy governs deletion) |
| Replication | Leader + follower copies across brokers — durability via redundancy |
| Kafka vs Spark fault tolerance | Kafka = redundant copies (replication); Spark = recompute from lineage |
| Consumer group | Partitions split across group members — parallelism ceiling = partition count |
| Multiple groups | Fully independent, own offsets — many groups can read same topic in parallel |
| At-least-once | Kafka's practical default — no loss, possible duplicates |
| Exactly-once | Possible but costly, needs end-to-end participation |
| Idempotency fix | SET/overwrite instead of INCREMENT — duplicate processing becomes harmless |
| Model & Kafka | Model never queries Kafka directly — stream processor writes to online store; model queries THAT |

**Reflex line:** "At-least-once + idempotent processing = as safe as exactly-once, without the coordination cost."

---

## Module D — Orchestration

| Concept | One-liner |
|---|---|
| Cron's gap | No dependency awareness → silently runs on stale/incomplete data; no retries, backfills, observability |
| DAG | Directed Acyclic Graph — jobs + dependencies, not jobs + fixed times |
| Airflow: DAG/Task/Operator | DAG = pipeline as code; Task = one unit of work; Operator = the template/type of action |
| Scheduler vs Executor | Scheduler decides what should run now (schedule + dependency state); Executor actually runs it |
| Idempotent tasks | Overwrite/upsert, not append — makes retries & backfills SAFE not dangerous |
| Sensor | Polls an external condition (file landed?) before letting the DAG proceed |
| Validation | Within-run checks: schema, range, null, referential — put EARLY, fail fast/cheap |
| Monitoring | Across-time checks: freshness, volume, schema drift |
| Skew detection | Validation alone won't catch it — need to compare batch vs. streaming outputs directly |

**Reflex line:** "The scariest failure: a pipeline that runs 'successfully' but silently produces stale or wrong data — no error anywhere."

---

## Module E — Tying it together

| Concept | One-liner |
|---|---|
| ETL | Transform BEFORE load — needed when data must be cleaned/redacted before landing anywhere (PII) |
| ELT | Load raw FIRST, transform in place after — modern default; cheap storage + keeps raw copy for future use cases |
| Feature store | ONE feature definition → populates offline store (history, training) + online store (current value, serving) |
| Offline store | Point-in-time-correct, full history, batch access |
| Online store | Current value only, millisecond lookup, serving path |
| Feature store fixes | Definitional skew (shared logic) — does NOT fix freshness skew (still need lag monitoring) |
| Skew root cause, 1 sentence | Training (batch) and serving (real-time) compute "the same" feature via two separate code paths that can quietly diverge |

**Reflex line:** "Training-serving skew is a data-pipeline consistency problem, not a modeling problem."

---

## The recommendation-pipeline example (reuse this skeleton for ANY "design a pipeline" question)

```
Website → Kafka (topic, partitioned by user_id)
             │
             ├─ Load raw → Data Lake (Parquet, partitioned by date)   [ELT]
             │        │
             │   Airflow DAG: validate → broadcast-join → groupBy (watch skew)
             │        → idempotent write → Offline Feature Store
             │
             └─ Kafka Streams (idempotent rolling counts) → Online Feature Store
                                                                  │
                                                     Recommendation Service (serves)
```

**Structure any system-design answer as:**
1. Clarify latency/scale requirements first (don't assume "real-time")
2. Ingestion (Kafka, partition key choice)
3. Storage (lake, format, partitioning)
4. Batch processing (Spark: joins, shuffle/skew awareness, point-in-time correctness)
5. Streaming path if freshness requires it (what actually needs real-time vs. what doesn't)
6. Feature store tying batch+streaming together (skew mitigation)
7. Orchestration + validation + monitoring
8. Serving layer
9. Proactively mention: failure modes, scaling levers, skew-detection strategy

---

## Common traps to say out loud (shows depth, not just recall)

- Don't reach for streaming by default — justify it against the actual latency need.
- Don't forget point-in-time correctness when describing any training feature — mention leakage risk unprompted.
- Don't say "exactly-once" as the default answer — say idempotent + at-least-once.
- Don't over-partition on high-cardinality keys (lake OR Kafka).
- Don't forget: a feature store reduces skew, it doesn't eliminate freshness lag.
- Don't say "use Airflow" without saying WHY (dependency-aware retries + idempotent backfills).

Good luck tomorrow.
