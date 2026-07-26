# Data Engineering Basics — Syllabus for Google/Apple MLE Interviews

**Goal:** Go from zero knowledge to being able to comfortably discuss how data
flows from raw source → trained model, in an MLE interview system-design or
ML-infra round. This is not a full Data Engineer's syllabus — it's the subset
an MLE is expected to reason about: how pipelines are built, why certain
tools/architectures are chosen, and the tradeoffs interviewers probe.

Each chapter below will be built out the same way as your other prep material
(plain-language first, math/architecture from scratch, diagrams, numeric or
worked examples, then interview-style Q&A at the end).

---

## Module A — Foundations: How Data Moves and Is Stored

**Ch1. The Big Picture: What Is a Data Pipeline**
- Source → Ingest → Store → Process → Serve, as one mental model
- Where an MLE sits in this pipeline (feature pipelines, training data, serving data)
- Why interviewers ask this at all (data quality/availability breaks more ML systems than model choice does)

**Ch2. Batch vs. Streaming**
- What "batch" actually means (bounded data, scheduled runs)
- What "streaming" actually means (unbounded data, continuous processing)
- Latency vs. throughput vs. cost tradeoffs
- Micro-batching as the middle ground (Spark Structured Streaming)
- When an MLE would choose one over the other (e.g., fraud detection vs. monthly churn model)

**Ch3. Data Lake vs. Data Warehouse vs. Lakehouse**
- Schema-on-read vs. schema-on-write
- Structured vs. semi-structured vs. unstructured data
- Cost and query-pattern differences
- Where feature stores fit in relative to these
- Modern lakehouse idea (Delta Lake / Iceberg / Hudi) and why it emerged

**Ch4. File Formats & Storage Layout**
- Row-based vs. columnar storage (CSV/JSON vs. Parquet/ORC)
- Why columnar formats dominate analytics and ML training reads
- Compression, schema evolution, partitioning on disk (date=/region= style layouts)
- Avro vs. Parquet (row-oriented serialization for streaming vs. columnar for analytics)

---

## Module B — Distributed Processing (Spark)

**Ch5. Why Distributed Processing Exists**
- Single-machine limits, the scale-out idea
- Cluster anatomy: driver, executors, cluster manager (conceptual, not YARN/K8s internals-heavy)

**Ch6. RDDs — The Low-Level Model**
- What an RDD actually is (a lineage graph, not a container of data)
- Transformations vs. actions, lazy evaluation
- Why RDDs matter to understand even though nobody hand-writes them anymore

**Ch7. DataFrames & the Catalyst/Tungsten Engine**
- DataFrames as RDDs + schema + query optimizer
- Why DataFrames are almost always preferred over raw RDDs today
- How the optimizer changes your mental model of "what runs when"

**Ch8. Partitioning, Shuffling, and Performance**
- What a partition is and why partition count matters
- Shuffles: what causes them (groupBy, join, repartition) and why they're expensive
- Data skew: what it is, why it kills performance, common fixes (salting, broadcast joins)
- Broadcast joins vs. shuffle joins — when Spark picks each

**Ch9. Spark for ML Workloads**
- Feature engineering at scale: common patterns and pitfalls
- Reading training data efficiently (partition pruning, predicate pushdown)
- Where Spark ends and a training framework (PyTorch/TF) begins

---

## Module C — Streaming Systems (Kafka)

**Ch10. Messaging Systems 101**
- Producer/consumer/broker vocabulary
- Why you'd put a message queue between systems at all (decoupling, buffering, replay)

**Ch11. Kafka Core Concepts**
- Topics and partitions: what they are and why partitions = parallelism unit
- Offsets and how consumers track position
- Replication and leader/follower partitions (durability basics)

**Ch12. Consumer Groups & Delivery Semantics**
- How consumer groups enable parallel + independent consumption
- At-most-once / at-least-once / exactly-once — what each means and where the tradeoff bites
- Idempotency as the practical fix for at-least-once

**Ch13. Kafka in an ML Context**
- Feeding real-time features into a feature store
- Streaming model input (e.g., clickstream → real-time ranking) vs. batch retraining
- Where Kafka Streams / ksqlDB fit vs. just using Kafka as a pipe into Spark

---

## Module D — Orchestration & Pipeline Reliability

**Ch14. Why Orchestration Is Its Own Problem**
- Cron's limitations at scale (dependencies, retries, backfills, observability)
- DAGs as the core abstraction

**Ch15. Airflow Fundamentals**
- DAGs, tasks, operators, scheduler, executor — what each actually does
- Idempotent tasks and why they matter for retries/backfills
- Sensors and triggers (waiting on upstream data)

**Ch16. Pipeline Reliability & Data Quality**
- Data validation / schema checks (great expectations-style thinking, conceptually)
- Backfills and reprocessing historical data safely
- Monitoring: freshness, volume, schema drift
- Training-serving skew as a data-pipeline problem, not just a modeling problem (ties back to your MLOps notes)

---

## Module E — Tying It Together for Interviews

**Ch17. ETL vs. ELT**
- What changed (cheap compute/storage → push transformation downstream)
- When each is still the right call

**Ch18. Feature Stores**
- The problem they solve (train/serve consistency, feature reuse)
- Offline store vs. online store, and how batch + streaming both feed them

**Ch19. End-to-End Worked System**
- A full worked example: "Design the data pipeline for a real-time recommendation model" — walked start to finish using everything above
- Common interviewer follow-ups and how to answer them

**Ch20. Rapid-Fire Interview Q&A**
- The "explain X in one minute" versions of every concept above, drilled

---

### Suggested order
Modules A → B → C → D → E, since B (Spark) and C (Kafka) both lean on the
storage/format vocabulary from Module A, and Module E assumes you've seen all
of B–D at least once.

Want me to start writing out Ch1 in full (same style as your other chapters — plain language, no assumed prior knowledge, diagrams + worked examples)?
