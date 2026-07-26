# Chapter 15: Airflow Fundamentals

## From the concept to the concrete tool

Ch14 established *why* you need a DAG-based orchestrator. This chapter maps
that concept onto **Airflow** specifically — the tool name you'll actually
hear in interviews — and introduces the idempotency requirement that makes
the whole retry/backfill story from Ch14 actually work correctly.

---

## The core building blocks

**DAG:** in Airflow, this is literally the term used — a Python object
defining the set of tasks and their dependencies, exactly as pictured in
Ch14. You write a DAG as code (a Python file), not a config file or a
drag-and-drop UI — this "pipelines as code" approach is a deliberate design
choice, letting you use normal software practices (version control, code
review, testing) on your pipeline definitions.

**Task:** a single node in the DAG — one unit of work (e.g.,
"run this Spark job," "check that this file exists," "run this SQL
query"). Tasks are connected with dependency operators
(`task_a >> task_b` means "task_a must succeed before task_b runs" — this
directly encodes the arrows from Ch14's diagrams).

**Operator:** the *type* of work a task performs — a template for a
specific kind of action. Airflow ships with many built-in operators:
`BashOperator` (run a shell command), `PythonOperator` (run a Python
function), `SparkSubmitOperator` (submit a Spark job), and dozens more for
specific systems (databases, cloud services, etc.). A task is really "an
operator, configured with specific parameters" — e.g., a
`SparkSubmitOperator` configured to run your specific co-purchase-stats
Spark job.

**Scheduler:** the Airflow component that continuously evaluates all your
DAGs and decides what should run right now — based on schedule (e.g.,
"daily at 1am") *and* on whether each task's dependencies have actually
been satisfied (the DAG-aware behavior from Ch14, not just clock time).

**Executor:** the component that actually *runs* the tasks the scheduler
decides need running — this can be configured to run tasks locally, or
distribute them across a cluster of worker machines, depending on scale
needs. (You don't need deep executor-configuration knowledge for an MLE
interview — just know this is the "who actually executes the task" layer,
distinct from the scheduler's "who decides what needs running" layer.)

---

## Idempotent tasks: the requirement that makes retries actually safe

This is the single most important practical concept in this chapter, and
it directly extends the idempotency discussion from Ch12 (Kafka consumers)
and the determinism discussion from Ch6 (Spark RDDs) — **same underlying
principle, now applied to orchestrated pipeline tasks.**

Recall from Ch14: Airflow can automatically retry a failed task. But
automatic retries are only *safe* if the task is **idempotent** — running
it twice (or three times, after two failed attempts) produces the same
correct final result as running it once.

**Concrete example of a non-idempotent task (bad):**
```python
# BAD: appends new rows every time this task runs
spark_job.write.mode("append").save("warehouse.copurchase_stats")
```
If this task fails halfway through (having already written some rows) and
Airflow retries it, you now have **duplicate rows** from the partial first
attempt plus the full successful retry — silently corrupting the output
table, with no error raised anywhere.

**Concrete example of an idempotent task (good):**
```python
# GOOD: overwrites the specific partition for this run's date,
# regardless of how many times this task has been attempted
spark_job.write.mode("overwrite").partitionBy("date") \
    .save("warehouse.copurchase_stats", partition_date=run_date)
```
Now, running this task 1 time or 5 times (due to retries) produces the
exact same final state — each attempt fully overwrites just that day's
partition rather than appending on top of whatever partial state a
previous failed attempt left behind.

**The general pattern to recognize and articulate:** prefer `overwrite`/
`upsert`-style operations over `append`-style operations for anything that
might be retried — this is the orchestration-layer version of the
"set vs. increment" idempotency lesson from Ch12.

---

## Sensors and triggers: waiting on upstream data

Sometimes a task shouldn't just depend on *another Airflow task*
succeeding — it needs to wait on something **external**, like "has this
file actually landed in S3 yet" or "has an upstream team's pipeline (which
isn't part of your DAG at all) finished." This is what **sensors** are for.

A sensor is a special kind of task that **polls** for a condition
(checking periodically: "does this file exist yet? not yet... check
again... yes, it exists now") and only lets the DAG proceed once that
condition is true. Example: `S3KeySensor` waits for a specific file to
appear in a bucket before letting the downstream Spark job start reading
it — directly solving the Ch14 problem of "the 2am job ran before the 1am
job's output was actually fully ready," but generalized to depend on
*external* signals, not just other Airflow tasks.

---

## Worked example, tying Ch14 and Ch15 together

The recommendation pipeline DAG, now with Airflow-specific pieces named:

```python
wait_for_raw_events = S3KeySensor(task_id="wait_for_events", ...)
                                    # SENSOR — polls until the day's raw
                                    # event files have actually landed

spark_join = SparkSubmitOperator(task_id="join_and_aggregate", ...)
                                    # writes with mode="overwrite" on the
                                    # run's specific date partition —
                                    # IDEMPOTENT, safe to retry

write_warehouse = PythonOperator(task_id="load_to_warehouse", ...)
                                    # also idempotent — upserts by
                                    # (product_id, date) rather than
                                    # blindly appending

wait_for_raw_events >> spark_join >> write_warehouse
```

If `spark_join` fails partway (say, a transient cluster issue), Airflow's
scheduler automatically retries it (per configured retry policy). Because
the task was written idempotently, that retry safely overwrites the same
partition rather than duplicating data — this is the concrete payoff of
everything in this chapter working together.

---

## Downstream considerations

1. **Latency:** Sensors that poll too infrequently add unnecessary delay
   (waiting up to the full poll interval after data is actually ready
   before noticing); polling too frequently wastes resources checking a
   condition that isn't true yet. This is a real, tunable tradeoff worth
   naming if asked about sensor design.
2. **Consistency:** Idempotency is the direct guarantee that a retried (or
   backfilled) task produces correct, non-duplicated output — without it,
   Airflow's automatic retry/backfill features (Ch14's whole selling
   point) actually become a liability, silently corrupting data instead of
   safely recovering from failure.
3. **Cost/scale:** Non-idempotent tasks that get retried can silently
   inflate storage/compute costs over time (duplicate data accumulating,
   or downstream jobs processing that duplicated data unnecessarily) —
   this is a subtle but real cost driver worth being aware of.
4. **Failure mode:** The most dangerous Airflow anti-pattern is a
   non-idempotent task combined with automatic retries — this actively
   makes a transient failure *worse* (silent data corruption) rather than
   Airflow's retry mechanism safely recovering from it. Worth stating
   explicitly if asked "what's a common mistake teams make with Airflow."

---

## Quick recap

- DAG = the pipeline definition (as Python code); Task = one unit of work;
  Operator = the template/type of action a task performs; Scheduler =
  decides what should run based on schedule + dependency state; Executor =
  actually runs the tasks.
- Idempotent tasks (prefer overwrite/upsert over append) are what make
  Airflow's automatic retries and backfills *safe* rather than dangerous —
  this is the single most important practical takeaway of the module.
- Sensors let a DAG wait on external conditions (a file landing, an
  upstream system finishing) rather than only depending on other Airflow
  tasks — generalizing Ch14's dependency idea beyond just your own DAG.

---

## Interview-style Q&A

**Q: Why does task idempotency matter so much in an orchestration tool
like Airflow?**
A: Airflow's core value proposition includes automatic retries and
backfills — but both of those mean a task might run more than once for the
same logical unit of work. If the task isn't idempotent (e.g., it appends
rows rather than overwriting a partition), a retry after a partial failure
can silently duplicate data, turning a recoverable transient failure into
a data-corruption bug.

**Q: What's the difference between a sensor and a regular task in
Airflow?**
A: A regular task performs work and completes. A sensor instead polls
repeatedly for an external condition to become true (e.g., a file landing
in cloud storage) and only allows downstream tasks to proceed once that
condition is satisfied — it's how a DAG can depend on something outside
Airflow's own task graph, like an upstream team's pipeline finishing.

**Q: How would you fix a Spark task in an Airflow DAG that writes output
using append mode, given that Airflow might retry it on failure?**
A: Change it to overwrite the specific partition for that run (e.g., by
date), or upsert by a natural key, rather than blindly appending. That way,
whether the task runs once or is retried multiple times due to transient
failures, the final output is the same — making the task idempotent and
safe to retry.

---

Next: **Ch16 — Pipeline Reliability & Data Quality**, the last chapter of
Module D. Say "ch16" when ready.
