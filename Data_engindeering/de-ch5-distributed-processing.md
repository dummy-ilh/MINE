# Chapter 5: Why Distributed Processing Exists

## Opening Module B

Chapters 1–4 covered where data lives and how it moves. Starting here, we
zoom into the "Process" box: **Spark**. Before touching RDDs or DataFrames
(Ch6–7), you need the "why" — what problem does a distributed engine like
Spark actually solve, and what does a cluster look like underneath the
API calls you'll eventually write?

---

## The problem: a single machine runs out of room

Imagine you have a table of 50 billion rows of clickstream events and you
need to compute "average session length per user." Two hard limits show up
on a single machine:

1. **Memory:** you can't load 50 billion rows into RAM on one machine — it
   just won't fit, no matter how much RAM you buy (there's a ceiling, and
   the data will keep growing past it).
2. **Time:** even if it *did* fit, processing 50 billion rows sequentially,
   one core at a time, could take hours or days — unacceptable if this
   needs to feed a nightly pipeline that other things depend on.

Vertical scaling (buying a bigger single machine) helps for a while but
hits diminishing returns fast, and there's always a ceiling on how big one
machine can get. **Horizontal scaling** — using many ordinary machines
together instead of one huge one — is the way out, and it's the founding
idea behind Spark (and Hadoop before it, and basically all "big data" tools).

## The core idea: split the data, split the work, combine the results

This is the same idea as, say, a group of people alphabetizing a giant pile
of index cards: instead of one person going through all of them, you split
the pile into chunks, hand a chunk to each person, let them each sort their
chunk in parallel, then merge the sorted chunks back together. The work
finishes roughly (num_people)x faster, and no single person needed to hold
the entire pile in front of them at once.

In distributed processing terms:
- The dataset gets **split into partitions** — chunks small enough to fit
  in memory on one machine (this "partition" word comes back constantly —
  Ch8 goes deep on it).
- Each partition gets processed **independently, in parallel**, on a
  different machine.
- Results get **combined/aggregated** at the end.

This only works well if the work can actually be split up like this — which
is why Spark's programming model (transformations that apply the same
operation to every partition independently) is shaped the way it is. You'll
see this shape explicitly in Ch6.

---

## Anatomy of a cluster (conceptual, not admin-level)

You don't need to know how to *operate* a cluster for an MLE interview, but
you should be able to describe what's in one and each piece's job:

```
                     ┌─────────────┐
                     │   DRIVER    │   ← plans the work, coordinates everything
                     │  (your app) │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ EXECUTOR │  │ EXECUTOR │  │ EXECUTOR │   ← do the actual work,
        │ (worker) │  │ (worker) │  │ (worker) │      each on its own partitions
        └──────────┘  └──────────┘  └──────────┘
```

- **Driver:** the process running your actual Spark code. It builds the
  execution plan (what work needs doing, in what order) and hands out
  tasks to executors. It doesn't process the bulk data itself.
- **Executors:** worker processes, usually one per machine (or several per
  machine), that actually hold data partitions in memory and run the
  computation on them.
- **Cluster manager:** the layer that decides which physical machines the
  driver and executors actually run on (YARN, Kubernetes, or Spark's own
  standalone manager). For interview purposes, it's enough to know this
  layer exists and handles resource allocation — you won't be asked to
  configure it.

**Key mental model:** when you write `df.filter(...).groupBy(...).sum()`,
you (the driver) are *describing* work. The driver figures out how to split
that work across partitions, ships the instructions to executors, and each
executor does its slice — you never manually decide which machine does
what.

---

## Worked example: connecting this back to the pipeline

Recall the nightly job from Ch1/Ch4: joining view/purchase events with
product metadata, computing co-purchase stats, reading from Parquet files
in the lake.

- That data — millions/billions of events across many partitioned Parquet
  files (Ch4) — gets read in, and each file/partition can be assigned to a
  different executor to read and process in parallel.
- The **driver** builds the plan: read events, join with product metadata,
  group by product pair, count. It doesn't do this work itself — it
  schedules it across executors.
- Each **executor** processes its own chunk of the data independently, and
  partial results (e.g., partial counts per product pair) get combined at
  the end — this "combine" step is a **shuffle**, which you'll meet
  properly in Ch8, and it's usually the expensive part.

This is also why the file layout from Ch4 (Parquet, partitioned by date)
matters so directly here: Spark can assign one partition/file to one task,
so a well-partitioned lake maps naturally onto a well-parallelized Spark
job. Badly laid-out data (one giant unsplit file, or millions of tiny
files) makes it harder for Spark to parallelize efficiently — this is the
concrete link between Ch4 and Ch5–8.

---

## Downstream considerations

1. **Latency:** More machines (more parallelism) generally means faster
   wall-clock completion for a fixed amount of data — but only up to a
   point. Coordination overhead (driver scheduling, combining results)
   doesn't shrink, so throwing infinite machines at a small job eventually
   stops helping and can even get slower (overhead dominates the actual
   work).
2. **Consistency:** Distributed processing introduces the possibility of
   **partial failure** — one executor can crash mid-job while others
   succeed. Spark is designed to detect this and simply re-run the failed
   partition's task elsewhere, which only works cleanly if the computation
   is *deterministic and re-runnable* (a theme that recurs in Ch6's
   discussion of RDD lineage, and again in Ch15–16 on pipeline reliability).
3. **Cost/scale:** More executors = more cost, roughly linearly. Interview
   framing: distributed processing doesn't make computation "free" — it
   trades money (more machines) for time (faster completion), and knowing
   when that trade is worth it matters more than just "always use Spark."
4. **Failure mode:** If a single machine (in a non-distributed world) dies
   mid-job, you lose everything and start over. In a well-designed
   distributed system, one executor dying only costs you the re-run of
   that executor's partition, not the whole job — this fault tolerance is
   one of the actual reasons distributed frameworks are worth the added
   complexity, not just "handles bigger data."

---

## Quick recap

- A single machine hits real memory and time ceilings on large-enough
  data — distributed processing scales horizontally (more machines) instead
  of vertically (bigger machine).
- The core pattern: split data into partitions, process each in parallel,
  combine results.
- A cluster = driver (plans/coordinates) + executors (do the actual work)
  + a cluster manager (allocates physical resources).
- Good data layout (from Ch4) directly enables good parallelism here —
  they're not separate concerns.
- Distributed systems trade cost (more machines) for speed, and gain fault
  tolerance (one executor failing doesn't kill the whole job) as a real
  side benefit, not just raw scale.

---

## Interview-style Q&A

**Q: Why not just use a bigger single machine instead of a Spark cluster?**
A: Vertical scaling (bigger machine) has a hard ceiling — there's a limit
to how much RAM/CPU a single machine can have, and it gets disproportionately
expensive as you approach that ceiling. Horizontal scaling (many ordinary
machines) has effectively no ceiling and is usually far more cost-efficient
per unit of compute at large scale.

**Q: What's the role of the driver vs. the executors?**
A: The driver plans the work — it builds the execution graph from your code
and schedules tasks — but doesn't process the bulk data itself. Executors
are the workers that actually hold data partitions in memory and run the
computation, each handling its own slice in parallel.

**Q: What happens if one executor crashes mid-job in Spark?**
A: Spark detects the failure and re-runs just that executor's failed
partition/task elsewhere in the cluster, rather than failing the entire
job — this works because Spark tracks how each partition's result was
computed (its lineage, covered in Ch6) and can deterministically recompute
just the lost piece.

---

Next: **Ch6 — RDDs: The Low-Level Model.** Say "ch6" when ready.
