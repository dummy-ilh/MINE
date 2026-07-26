# Chapter 3: Data Lake vs. Data Warehouse vs. Lakehouse

## Why this comes right after batch vs. streaming

Ch2 was about *when* data gets processed. This chapter is about *where it
lives* once it's landed — the "Store" box from Ch1. This choice determines
how expensive/fast it is to later query that data, and (crucially for you)
how easy it is to pull training data or build a feature pipeline off it.

---

## What a data warehouse is

**A data warehouse stores structured data, in a predefined schema, optimized
for fast analytical queries (SQL) over large volumes.**

Plain-language analogy: a warehouse is like a library with a strict cataloging
system — every book has to be classified and shelved in exactly the right
spot *before* it goes on the shelf. This upfront work makes finding things
later extremely fast.

Key properties:
- **Schema-on-write**: you must define the table structure (columns, types)
  *before* loading data in. Data that doesn't fit the schema either gets
  rejected or has to be transformed to fit first.
- Data is **structured** (rows and columns) — think: an `orders` table, a
  `users` table.
- Optimized for **SQL-style analytical queries**: aggregations, joins,
  filters over millions/billions of rows.
- Examples: Snowflake, BigQuery, Redshift.

## What a data lake is

**A data lake stores raw data of any type — structured, semi-structured, or
unstructured — in its native format, with no schema enforced at write time.**

Plain-language analogy: a lake is like a giant storage unit where you just
drop boxes in as they arrive, unlabeled, and you figure out what's inside
and how to organize it only when you actually need something.

Key properties:
- **Schema-on-read**: you don't define structure upfront. You dump raw
  files in (JSON logs, images, CSVs, Parquet, video, whatever), and figure
  out the schema/structure only when you later read and process it.
- Can store **structured, semi-structured (JSON, XML), and unstructured**
  (images, audio, raw text) data — this is the big difference from a
  warehouse.
- Cheaper storage (usually object storage like S3/GCS/ADLS under the hood).
- Examples: raw files sitting in S3, often organized with tools like Delta
  Lake/Iceberg/Hudi on top (see "Lakehouse" below).

## Schema-on-write vs. schema-on-read — the concept that actually matters

This is the single idea worth internalizing, because everything else in this
chapter falls out of it:

- **Schema-on-write** (warehouse): you pay the cost of structuring data
  *once*, upfront, before it's stored. Every future read is fast and
  reliable because the structure is guaranteed.
- **Schema-on-read** (lake): you pay *no* cost upfront — just dump the data
  in — but you pay the cost of interpreting/structuring it *every single
  time* something reads it. Flexible, but slower per-query and prone to
  "garbage in" problems (nothing stopped bad data from landing).

This is a classic pay-now-vs-pay-later tradeoff, and it's exactly the kind
of thing interviewers like to hear articulated in your own words.

---

## Why the lakehouse exists

For years, the standard setup was: dump raw data in a lake (cheap, flexible,
handles anything) → run ETL jobs to clean/structure it → load the cleaned
result into a warehouse for fast, reliable analytical querying. Two systems,
two copies of data, and an ETL pipeline stitching them together.

The **lakehouse** idea (Delta Lake, Apache Iceberg, Apache Hudi) is:
what if we could get warehouse-like reliability (schema enforcement, ACID
transactions, fast queries) *directly on top of* cheap lake storage (S3/GCS),
without needing to duplicate data into a separate warehouse system?

How it does this, conceptually:
- Adds a **transaction log** on top of raw files in the lake, so you get
  ACID guarantees (no half-written/corrupted reads) even though the
  underlying storage is just files.
- Enforces **schema** optionally, catching bad data at write time if you
  want that safety — you're not forced into pure schema-on-read chaos.
- Supports **time travel** (query the table as it looked yesterday) because
  the transaction log keeps a history of changes.
- Still uses cheap object storage and open file formats (Parquet)
  underneath — you're not locked into a proprietary warehouse.

**One-line summary for interviews:** *a lakehouse tries to give you warehouse
reliability at lake cost/flexibility, by adding a transactional metadata
layer on top of files instead of requiring a separate system.*

---

## Comparison table

| | Data Warehouse | Data Lake | Lakehouse |
|---|---|---|---|
| **Schema** | On-write (enforced upfront) | On-read (flexible, deferred) | Optionally enforced, transaction-logged |
| **Data types** | Structured only | Structured + semi-structured + unstructured | Same as lake, with structure guarantees available |
| **Query speed** | Fast, optimized for SQL | Slower — often needs a compute engine (Spark) on top | Fast, close to warehouse speed |
| **Cost** | Higher (compute+storage bundled) | Cheap storage (object storage) | Cheap storage + warehouse-like features |
| **Best for** | BI dashboards, reliable reporting | Raw event logs, ML training data (images, text, huge volumes), data science exploration | Modern default: one copy of data serving both BI and ML |

---

## Worked example: where does your ML data actually sit?

Back to the recommendation model from Ch1:

- The **raw click/view/purchase events from Kafka** land in a **data
  lake** (as Parquet, partitioned by date) — this data is high-volume,
  needs to support many different future uses, and you don't want to
  commit to a rigid schema before you know exactly what features you'll
  need.
- The **cleaned, aggregated co-purchase statistics** (the output of the
  nightly Spark job) might get loaded into a **data warehouse table** —
  now it's structured, relatively small, and analysts/dashboards want fast
  SQL access to it.
- If the company uses a **lakehouse** (e.g., Delta Lake), both of the above
  might just live as different tables in the *same* system — raw events as
  a lightly-structured Delta table, aggregated stats as a strongly-typed
  Delta table, no separate warehouse needed.

This is also exactly why data lakes (and lakehouses) are usually where ML
training data lives rather than a pure warehouse: training often needs raw,
large-volume, sometimes unstructured data (e.g., raw event logs, images)
that a rigid warehouse schema isn't built for.

---

## Downstream considerations

1. **Latency:** Warehouses are optimized for fast *query* latency once
   data's loaded, but data typically arrives via a batch ETL load, so
   *freshness* latency can lag. Lakes can ingest raw data with lower
   latency (dump-and-go), but querying it fast may then require an
   additional processing layer (Spark) — you've shifted the cost, not
   removed it.
2. **Consistency:** If your training pipeline reads from the lake but your
   BI/reporting reads from a warehouse that's ETL'd from that same lake,
   watch for **drift between the two copies** if the ETL job lags or a
   schema changes on one side without the other. This is the storage-layer
   version of training-serving skew.
3. **Cost/scale:** Lakes are cheap to just keep dumping raw data into
   indefinitely (object storage is cheap) — a common ML pattern is
   "store everything raw, decide later what's useful." Warehouses charge
   more for that same flexibility, so teams are pickier about what they
   load there.
4. **Failure mode:** Schema-on-write (warehouse) fails loudly and early —
   bad data gets rejected at load time. Schema-on-read (lake) fails
   silently and late — bad/malformed data can sit unnoticed until some
   downstream job tries to read it and breaks (or worse, doesn't break
   and just produces wrong results).

---

## Quick recap

- Warehouse = structured, schema-on-write, fast SQL, higher cost.
- Lake = any data type, schema-on-read, cheap storage, needs a compute
  engine to query well.
- Lakehouse = adds a transactional/schema layer on top of lake storage to
  get warehouse-like reliability without a separate system.
- ML training data usually favors lakes/lakehouses (raw volume, flexible
  types); BI/reporting usually favors warehouses (or the warehouse-like
  layer of a lakehouse).
- Watch for drift between lake and warehouse copies of "the same" data —
  it's a storage-layer version of training-serving skew.

---

## Interview-style Q&A

**Q: Why wouldn't you just store all your ML training data in a data
warehouse?**
A: Warehouses require schema-on-write and are optimized for structured,
tabular data — but ML training data is often raw, high-volume, and
sometimes unstructured (images, text, logs), which fits a lake's
schema-on-read flexibility much better and is far cheaper to store at that
scale.

**Q: What problem does a lakehouse actually solve, in one sentence?**
A: It avoids maintaining two separate copies of data (a lake copy and a
warehouse copy) by adding transactional guarantees and optional schema
enforcement directly on top of cheap lake storage.

**Q: What's a concrete risk of schema-on-read you'd want to guard against
in an ML pipeline?**
A: Malformed or unexpected data can silently land in the lake without being
caught at write time, and only surface as a failure (or, worse, a silent
data-quality bug) much later when a Spark job or training pipeline tries to
read and parse it — so lakes typically need extra validation logic
downstream to catch what schema-on-write would've caught for free.

---

Next: **Ch4 — File Formats & Storage Layout** (Parquet, Avro, partitioning
on disk). Say "ch4" when ready.
