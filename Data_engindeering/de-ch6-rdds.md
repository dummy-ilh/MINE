# Chapter 6: RDDs — The Low-Level Model

## Why learn this if "nobody writes RDDs anymore"

Fair question — in modern Spark code, you'll almost always use DataFrames
(Ch7), not raw RDDs. But DataFrames are *built on top of* RDDs, and several
behaviors that seem mysterious at the DataFrame level (lazy evaluation, why
a failed task can just be retried, why `.collect()` can crash your driver)
only make sense once you understand the RDD underneath. Interviewers
sometimes probe exactly this — "why is Spark lazy?" — to check you're not
just calling APIs without understanding what they do.

---

## What an RDD actually is

**RDD = Resilient Distributed Dataset.** Let's take that name apart, because
each word is doing real work:

- **Distributed:** the data is split into partitions spread across multiple
  machines (Ch5's split-process-combine idea, made concrete).
- **Resilient:** if a partition is lost (a machine crashes), Spark can
  *recompute* just that partition from scratch — it doesn't need a backup
  copy sitting around. This is the "fault tolerance" mentioned at the end
  of Ch5.
- **Dataset:** conceptually, a collection of records — but here's the part
  that surprises people:

**An RDD is not actually a container holding your data in memory. It's a
recipe for how to produce that data, plus a record of where it came from.**
Spark calls this recipe the **lineage graph**: "start by reading these
files, then filter with this function, then map with this function..." The
actual data only gets computed when it's needed.

This is *why* resilience works the way it does: if partition 7 is lost,
Spark doesn't need a backup — it just re-runs partition 7's recipe (read
its input file, apply the same filter/map steps) from scratch. The lineage
*is* the backup.

---

## Transformations vs. actions — the two kinds of RDD operations

This distinction is the single most important thing to understand, and it's
also where "laziness" comes from.

**Transformations** — operations that produce a *new* RDD from an existing
one, describing more of the recipe. Examples: `map`, `filter`, `groupBy`.
**Transformations are lazy** — calling `.filter(...)` does not actually
filter anything yet. It just adds a step to the recipe/lineage graph.

**Actions** — operations that actually trigger computation and produce a
real result (a number, a collected list, a write to disk). Examples:
`count()`, `collect()`, `saveAsTextFile()`. **Actions are what actually run
the recipe** — this is the moment Spark looks at the entire chain of
transformations you've built up and finally executes it.

```python
rdd2 = rdd1.filter(lambda x: x > 10)   # transformation — nothing happens yet
rdd3 = rdd2.map(lambda x: x * 2)       # transformation — still nothing happens
result = rdd3.count()                  # ACTION — now Spark actually runs
                                        # filter + map across all partitions
```

### Why bother being lazy? (this is the actual interview-worthy insight)

If Spark executed every transformation immediately, it would have to
materialize (fully compute and hold in memory) the result of `filter`
*before* even knowing you were about to `map` it right afterward — wasteful.

By staying lazy and waiting until an action is called, Spark can look at
the **entire chain of transformations at once** and optimize it — e.g.,
combine the filter and map into a single pass over the data instead of two
separate passes, or skip work for partitions that won't contribute to the
final result. Laziness isn't just an implementation quirk — it's what
*enables* the optimizer (which becomes much more powerful at the DataFrame
level in Ch7).

---

## Worked example: lineage and fault tolerance together

```python
raw = sc.textFile("s3://lake/events/date=2026-07-24/*.parquet")   # RDD 1
parsed = raw.map(parse_event)                                     # RDD 2 (transformation)
purchases = parsed.filter(lambda e: e.action == "purchase")       # RDD 3 (transformation)
total = purchases.count()                                         # ACTION → triggers execution
```

Nothing runs until `.count()`. At that point, Spark:
1. Looks at the full lineage: read files → parse → filter.
2. Splits the input files into partitions, assigns each to an executor.
3. Each executor runs the *entire* read→parse→filter recipe on just its
   own partition.
4. Partial counts are combined into the final total.

Now suppose the executor handling partition 5 crashes halfway through.
Spark doesn't panic or fail the whole job — it looks at partition 5's
lineage (which input file, which transformations) and simply reruns that
one partition's recipe on a different executor. This only works because:
- The lineage record says exactly how to reproduce partition 5 from
  scratch.
- The transformations are (assumed to be) **deterministic** — running
  `parse_event` and the filter again on the same input produces the same
  output every time. If your transformation function has hidden randomness
  or depends on external mutable state, this guarantee breaks — a real
  gotcha worth naming if it comes up.

---

## Downstream considerations

1. **Latency:** Laziness means errors in your *transformation logic* often
   don't surface until the action runs — which can be much later in your
   code, and much later in time, than where the actual bug is. This is a
   common debugging trap: the stack trace points at the `.count()` line,
   but the real bug is in a `.map()` three lines earlier.
2. **Consistency:** Because a lost partition is *recomputed* rather than
   restored from a saved copy, your transformation functions must be
   deterministic and side-effect-free for Spark's fault tolerance to
   actually produce correct, consistent results after a retry. A
   transformation that calls an external API with side effects (e.g.,
   "increment a counter in Redis") could double-count if that partition
   gets retried — a subtle but real correctness risk.
3. **Cost/scale:** Because nothing computes until an action, you can chain
   many transformations "for free" and let Spark's optimizer decide the
   most efficient way to execute the whole chain at once — versus forcing
   eager execution at every step, which would waste both time and memory
   materializing intermediate results you didn't need to keep.
4. **Failure mode:** `.collect()` (an action) pulls *all* the data from
   every executor back to the single driver machine. If the result is
   large, this can crash the driver with an out-of-memory error — a classic
   real-world Spark mistake, and one worth mentioning proactively in an
   interview as evidence you understand the driver/executor split from
   Ch5, not just the API surface.

---

## Quick recap

- An RDD isn't stored data — it's a lineage graph: a recipe for how to
  (re)produce data from its original source plus a chain of transforms.
- Transformations (map, filter, ...) are lazy — they just extend the
  recipe. Actions (count, collect, ...) trigger actual execution.
- Laziness exists so Spark can optimize the *entire* chain of operations
  at once, rather than executing each step eagerly and wastefully.
- Fault tolerance works by recomputing lost partitions from their lineage,
  which requires transformation functions to be deterministic.
- `.collect()` pulls all data to the driver — a common cause of driver
  out-of-memory crashes on large results.

---

## Interview-style Q&A

**Q: Why does Spark use lazy evaluation instead of executing each
transformation immediately?**
A: Laziness lets Spark see the full chain of transformations before running
anything, so it can optimize across the whole chain at once — e.g., fusing
multiple operations into a single pass over the data — rather than
eagerly materializing every intermediate result, which would waste memory
and time.

**Q: How does Spark recover from a lost partition without keeping backup
copies of the data?**
A: Each RDD tracks its lineage — the sequence of transformations and the
original source needed to reproduce it. If a partition is lost, Spark
simply reruns that partition's lineage on another executor, recomputing it
from scratch, rather than restoring from a stored backup.

**Q: What has to be true about your transformation functions for this
fault-tolerance model to give correct results?**
A: They need to be deterministic and free of external side effects —
otherwise, recomputing a partition after a failure could produce a
different result than the original run (non-determinism) or cause
duplicate side effects (e.g., double-writing to an external system).

---

Next: **Ch7 — DataFrames & the Catalyst/Tungsten Engine.** Say "ch7" when
ready.
