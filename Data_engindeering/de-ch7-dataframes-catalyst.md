# Chapter 7: DataFrames & the Catalyst/Tungsten Engine

## The one-sentence upgrade from Ch6

**A DataFrame is an RDD plus a known schema, which unlocks a query
optimizer.** That's genuinely most of what you need to remember — this
chapter is about what that extra "schema + optimizer" actually buys you,
and why it's the reason nobody hand-writes RDD code anymore.

---

## Why schema changes everything

With a raw RDD, Spark just sees opaque Python/Scala objects being passed
through your functions (`map(lambda x: ...)`). Spark has **no idea** what's
inside those objects — it can't tell that `x.age` is an integer, or that
`x.country` only has a handful of distinct values, or that you're about to
filter on a column that's sitting right there in a Parquet file's metadata.
Spark just blindly runs your function on every row.

A DataFrame, by contrast, knows its structure: `age: int, country: string,
purchase_amount: double`. Because Spark now *understands the shape and
types of the data*, it can reason about your query the way a database's SQL
engine reasons about a `SELECT` statement — and that reasoning is exactly
what the **Catalyst optimizer** does.

---

## What Catalyst actually does (plain language)

When you write:
```python
df.filter(df.country == "US").select("user_id", "purchase_amount")
```

With an RDD, this would translate to: read every row, then check `country`,
then keep only `user_id`/`purchase_amount` fields, in exactly the order you
wrote it. With a DataFrame, Catalyst instead looks at your *whole query* and
rewrites it into a more efficient plan before running anything — the same
idea you saw in Ch6 (laziness enabling optimization), just far more
powerful because now the engine understands the data's structure, not just
opaque function calls.

Common things Catalyst does automatically:

- **Predicate pushdown:** if the data is stored in Parquet (Ch4), Catalyst
  can push the `country == "US"` filter down to the file-reading step
  itself — some formats/storage systems can skip entire blocks of data
  that don't match, without even reading them into Spark at all. You
  never wrote code to do this; the optimizer figured it out from your
  query.
- **Column pruning:** since you only `select`ed 2 of maybe 20 columns,
  Catalyst ensures only those 2 columns are actually read off disk — the
  columnar-format benefit from Ch4, applied automatically based on what
  your query actually needs, not what's in the file.
- **Reordering operations:** e.g., applying a cheap filter *before* an
  expensive join, even if you wrote the join first in your code, because
  filtering first means less data flows into the expensive join step.
- **Join strategy selection:** deciding whether to use a broadcast join
  or a shuffle join (more in Ch8) based on estimated table sizes — you
  just write `.join(...)`, Catalyst picks the strategy.

**The core shift to internalize:** with RDDs, you (the programmer) are
responsible for writing efficient code. With DataFrames, you write *what*
you want, and Catalyst is responsible for figuring out an efficient *how*.
This is the same "declarative vs. imperative" idea that separates SQL from
hand-written loop-based code — and it's not a coincidence that DataFrames
feel SQL-like.

---

## Tungsten — the execution-side complement to Catalyst

Catalyst decides *what plan to run*. **Tungsten** is the underlying engine
that makes *running* that plan fast, at the level of raw memory and CPU:

- **Off-heap, binary memory layout:** instead of storing data as regular
  Java/Python objects (which have significant memory overhead and trigger
  garbage collection pauses), Tungsten packs data into compact binary
  representations closer to how columnar formats already store it —
  smaller memory footprint, less GC overhead, faster processing.
- **Whole-stage code generation:** rather than interpreting each operation
  (filter, then map, then...) one row at a time through generic Spark
  internals, Tungsten generates specialized, compiled code for your
  *specific* query pipeline — closer to what a human would write by hand
  for that exact task, which runs much faster than generic interpreted
  code.

You don't need to be able to reproduce these mechanisms in an interview —
you need to be able to say, at a high level: *"Catalyst optimizes the query
plan; Tungsten optimizes how that plan executes physically, at the memory
and CPU level."* That one sentence covers what most interviewers are
checking for.

---

## Worked example: same query, RDD vs. DataFrame

**RDD version** (you optimize everything by hand):
```python
raw = sc.textFile("s3://lake/events/*.parquet")
parsed = raw.map(parse_event)                       # you parse every row yourself
us_events = parsed.filter(lambda e: e.country == "US")   # you wrote the filter logic
result = us_events.map(lambda e: (e.user_id, e.purchase_amount))
```
Spark has no idea what `parse_event` does internally — it just runs your
function on every row of every partition. No column pruning, no predicate
pushdown; you get exactly the plan you wrote.

**DataFrame version** (Catalyst optimizes for you):
```python
df = spark.read.parquet("s3://lake/events/")
result = df.filter(df.country == "US").select("user_id", "purchase_amount")
```
Catalyst can push the `country == "US"` filter down to the Parquet read
itself, and only read the `country`, `user_id`, `purchase_amount` columns
off disk — skipping any other columns in the file entirely, without you
writing any of that logic yourself.

This is why, in almost all modern practice (including Spark Structured
Streaming, from Ch2), you reach for DataFrames by default and drop to RDDs
only for the rare case where you need very fine-grained, non-tabular
control that the DataFrame API doesn't expose.

---

## Downstream considerations

1. **Latency:** Catalyst's optimizations (pushdown, pruning, reordering)
   are exactly what makes the difference between a job that respects a
   tight nightly SLA and one that blows past it — this is the concrete
   payoff of "just use DataFrames," not an abstract nicety.
2. **Consistency:** Because Catalyst may reorder/rewrite your operations,
   the *order you wrote code in* is not necessarily the order it executes
   in — this matters if you're relying on side effects happening in a
   particular sequence (a general reason to avoid side-effecting code
   inside Spark transformations, echoing Ch6's determinism point).
3. **Cost/scale:** Better plans directly reduce compute cost — reading
   fewer columns and fewer rows (via pruning/pushdown) shrinks I/O and CPU
   time, which is real money at cloud-scale data processing. This is a
   legitimate answer if asked "how would you reduce this pipeline's cost."
4. **Failure mode:** Occasionally Catalyst's automatic join-strategy choice
   (broadcast vs. shuffle, Ch8) picks wrong if table size statistics are
   stale or missing (e.g., after a big data load without refreshing table
   stats), leading to a much slower plan than expected — worth knowing
   this is a real, debuggable failure mode, not a black box you can't
   reason about.

---

## Quick recap

- A DataFrame = an RDD + a schema, and that schema is what unlocks
  Catalyst's query optimization.
- Catalyst decides *what* efficient plan to run (predicate pushdown,
  column pruning, operation reordering, join strategy selection).
- Tungsten makes *running* that plan fast at the memory/CPU level
  (compact binary layout, generated code).
- With RDDs you hand-optimize everything; with DataFrames, you describe
  intent and the optimizer figures out the efficient execution — which is
  why DataFrames are the default choice in virtually all modern Spark code.

---

## Interview-style Q&A

**Q: What's the practical difference between working with RDDs and
DataFrames?**
A: RDDs are untyped, schema-less collections — Spark has no visibility into
their structure, so all optimization responsibility falls on the
programmer. DataFrames carry a known schema, which lets Spark's Catalyst
optimizer automatically apply techniques like predicate pushdown, column
pruning, and smart join-strategy selection — turning what you *wrote* into
a more efficient plan than what you'd get from equivalent hand-written RDD
code.

**Q: What's the difference between what Catalyst and Tungsten each do?**
A: Catalyst is the query planner/optimizer — it decides the logical and
physical execution plan for your query (what order to do things in, which
join strategy, what to push down). Tungsten is the execution engine that
makes running that chosen plan fast, via compact off-heap memory layout and
generated, specialized code instead of generic row-by-row interpretation.

**Q: Would you ever still use RDDs today?**
A: Mostly no for typical tabular ETL/ML feature work — DataFrames and their
optimizer make that faster and simpler to write. RDDs still show up for
low-level, non-tabular processing where you need fine-grained control
Catalyst's tabular optimizations don't apply to, but this is now a
minority case.

---

Next: **Ch8 — Partitioning, Shuffling, and Performance**, where a lot of the
real interview "war stories" (data skew, broadcast joins) live. Say "ch8"
when ready.
