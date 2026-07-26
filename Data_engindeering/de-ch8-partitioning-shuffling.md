# Chapter 8: Partitioning, Shuffling, and Performance

## Why this is the chapter interviewers dig into most

Everything so far has been "here's how Spark works." This chapter is "here's
what actually goes wrong in practice, and how to talk about fixing it."
Data skew and shuffle costs are probably the single most common Spark
performance topic in interviews, because they show up constantly in real
production pipelines — including ML feature pipelines.

---

## What a partition is, precisely (in-memory version)

You've seen "partition" twice already: Ch4 (files split on disk) and Ch5
(the split-process-combine idea). Here's the precise in-memory version:

**A partition is a chunk of a DataFrame/RDD that lives on one executor and
gets processed by one task, independently of the other chunks.**

If your DataFrame has 200 partitions and your cluster has 50 executor
cores, Spark can run up to 50 partitions' worth of work simultaneously,
with the remaining 150 partitions queueing up as cores free up. **Partition
count is effectively your unit of parallelism** — too few partitions and
you're leaving cores idle; too many and you pay excessive per-task
scheduling overhead (a smaller-scale version of the "too many tiny files"
problem from Ch4).

---

## What a shuffle is, and why it's the expensive part

Some operations can be computed **within** a single partition, independently
— e.g., `filter`, `map`. Spark just runs these locally on each partition's
data, no coordination needed between executors.

Other operations require data that's currently scattered across *different*
partitions to be **regrouped together** before the operation can proceed.
This regrouping is a **shuffle**: data physically moves across the network
between executors, gets written to disk temporarily, and gets re-read on
the receiving side.

**Operations that typically cause a shuffle:**
- `groupBy` — all rows with the same key need to end up on the same
  executor to be grouped together.
- `join` (in the general case) — matching rows from two tables need to be
  co-located to compare keys.
- `repartition` — you're explicitly asking Spark to redistribute data
  across a different number of partitions.
- `distinct`, `orderBy` — both require comparing/deduplicating across the
  whole dataset, not just within one partition.

### Why shuffles are expensive (plain language)

Think back to the alphabetizing-index-cards analogy from Ch5. `filter` is
like each person independently throwing away cards they don't need — no
coordination required. But `groupBy` is like: "everyone, pass me all your
cards starting with the letter M, and I'll pass you all mine starting with
your assigned letters" — now everyone has to stop, exchange piles with each
other over the network, and only then can each person continue working on
their now-regrouped pile. That exchange step — writing data out, sending it
over the network, reading it back in on another machine — is inherently
slower than any single machine just working on data it already has locally.
**This is the shuffle**, and it's usually the single biggest cost in a
real-world Spark job.

---

## Data skew: what it is and why it kills performance

**Skew** happens when the data isn't evenly distributed across the key
you're partitioning/grouping by — some partitions end up with far more data
than others.

Concrete example: you're computing "purchases per product," and one
product (say, a wildly popular limited-edition item) accounts for 40% of
all purchase events while thousands of other products share the remaining
60%. After the shuffle, the executor handling that one popular product's
partition has to process a massive chunk of data, while every other
executor finishes quickly and sits idle waiting.

**Why this is worse than it sounds:** a Spark job doesn't finish until its
*slowest* task finishes. 49 executors finishing in 2 minutes and 1
executor taking 40 minutes means the whole job takes 40 minutes — you paid
for 50 executors but only got the benefit of roughly 1. This is one of the
most common real-world "why is this job so slow" root causes, and it's
worth being able to describe it precisely in an interview, not just say
"skew is bad."

### Common fixes for skew

- **Salting:** artificially split a skewed key into several sub-keys (e.g.,
  `product_id + random_number_0_to_9`), spreading that one popular
  product's data across 10 partitions instead of 1, then combining the
  partial results afterward. This trades a bit of extra complexity for
  much more even load.
- **Broadcast joins** (below) — sidestep the shuffle entirely for one
  common skew-causing case: joining a huge table against a small one.
- **Increasing partition count / using `repartition` on the skewed key**
  — sometimes just gives Spark more, smaller pieces to spread out, though
  this doesn't fix true key-level skew (one key is still one key, however
  you slice partition *count*).

---

## Broadcast joins vs. shuffle joins

This is the concrete "when does Catalyst make a smart decision for you"
example promised back in Ch7.

**Shuffle join (default for two large tables):** both tables get
shuffled/reorganized so matching keys end up on the same executor, then
joined locally. Expensive — both sides pay the shuffle cost.

**Broadcast join (when one table is small):** instead of shuffling both
huge tables, Spark just **copies the small table in full to every
executor** (broadcasts it), so each executor can join its local chunk of
the big table against a full local copy of the small table — **no shuffle
of the big table needed at all.**

**When this applies:** joining a huge `events` table (billions of rows)
against a small `products` lookup table (a few thousand rows) — exactly
the pattern from the recommendation pipeline example in earlier chapters.
Catalyst automatically chooses a broadcast join here if the small table is
under a size threshold — you usually don't have to force it, but knowing
*why* it's faster, and being able to force it via a hint if Catalyst's size
estimate is wrong, is a real, practical interview-worthy detail.

---

## Worked example, tying it all together

Back to the co-purchase stats job: `events.join(products, "product_id")
.groupBy("product_id").count()`.

- The `join` against a small `products` table → Catalyst picks a
  **broadcast join**, avoiding a shuffle of the (huge) `events` table.
- The subsequent `groupBy("product_id")` **does** require a shuffle — rows
  need to be regrouped by `product_id` so counts can be computed together.
- If one product is disproportionately popular, this `groupBy` step is
  where **skew** shows up — that product's partition after the shuffle is
  much bigger than the rest, and that one task becomes the job's
  bottleneck.
- Fix: salt the `product_id` key for the groupBy, or accept that the
  popular product needs special-case handling (e.g., precomputed
  separately) if it's a known, persistent hot key.

---

## Downstream considerations

1. **Latency:** Shuffles are usually the dominant cost in job runtime —
   minimizing shuffle stages (favoring broadcast joins, filtering before
   joining, avoiding unnecessary `repartition`/`distinct`) is the most
   direct lever for making a pipeline finish faster and meet its downstream
   SLA (e.g., "training data must be ready by 6am").
2. **Consistency:** Not directly a correctness issue for well-written
   Spark code (shuffles produce the same logical result, just slower) —
   but severe skew can cause a job to run so long it misses its window,
   which can cascade into stale features reaching a model — tying back to
   Ch1's failure-mode framing.
3. **Cost/scale:** Skew wastes cluster resources — you pay for N
   executors but only effectively use 1 if a single skewed task dominates
   runtime. This is a very concrete, quantifiable cost-optimization talking
   point ("we identified and salted a skewed key, cutting job runtime and
   cluster cost by X%") that reads well in an interview.
4. **Failure mode:** Extreme skew can cause a single task to run out of
   memory (too much data on one executor) and fail the whole job outright,
   rather than just running slowly — worth knowing skew isn't purely a
   performance issue, it can be a hard failure too.

---

## Quick recap

- A partition is the unit of parallel work; partition count roughly sets
  your job's parallelism ceiling.
- A shuffle is data physically moving across the network to regroup by
  key — usually the most expensive part of a job (triggered by groupBy,
  join, repartition, distinct, orderBy).
- Skew = uneven key distribution → one task becomes the bottleneck → the
  whole job waits on it, wasting the rest of the cluster's capacity.
- Broadcast joins avoid shuffling a huge table by copying a small table to
  every executor instead — Catalyst does this automatically for
  small-enough tables.
- Fixes for skew: salting keys, broadcast joins where applicable,
  targeted repartitioning.

---

## Interview-style Q&A

**Q: Why is a `groupBy` typically much more expensive than a `filter` in
Spark?**
A: `filter` can be applied independently within each partition with no
coordination needed. `groupBy` requires rows sharing the same key to end
up on the same executor, which requires a shuffle — physically moving data
across the network, writing and re-reading it — and that network/disk
overhead is usually the dominant cost in a Spark job.

**Q: You have a Spark job that's taking much longer than expected. What
would you check first?**
A: I'd look at the Spark UI's stage/task breakdown for signs of skew —
specifically, one or a few tasks taking dramatically longer than the rest
in a shuffle stage, since the job's total time is bounded by its slowest
task. If confirmed, I'd look at whether a broadcast join could eliminate a
shuffle entirely, or salt the skewed key to spread that hot key's data
across multiple partitions.

**Q: When does Spark use a broadcast join, and why is it faster?**
A: When one side of a join is small enough to fit comfortably on every
executor (below a configurable size threshold), Spark can copy that whole
small table to every executor instead of shuffling the large table to
co-locate matching keys. This avoids shuffling the large table entirely,
which is usually the more expensive side of the join.

---

Next: **Ch9 — Spark for ML Workloads**, the last chapter of Module B before
we move to Kafka. Say "ch9" when ready.
