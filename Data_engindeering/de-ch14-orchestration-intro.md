# Chapter 14: Why Orchestration Is Its Own Problem

## Opening Module D

Modules B and C covered *processing* (Spark) and *moving* (Kafka) data.
This module is about **scheduling and coordinating** all the jobs that make
up a real pipeline — the piece that turns "a Spark job" and "a Kafka
consumer" into an actual, reliable, repeatable production system. This
chapter is the "why isn't cron enough" chapter before Airflow specifics
(Ch15).

---

## The naive starting point: just use cron

Cron (run this script at 2am every day) feels like it should be enough.
For a single, isolated job, it often is. The problems start the moment you
have **more than one job that depends on another**.

Recall the recommendation pipeline from Ch1: raw events land in the lake →
a Spark job joins events with product metadata → co-purchase stats get
written to a warehouse table → a downstream dashboard/model reads that
table. That's already 3-4 separate steps with real dependencies between
them. Cron has no concept of "step 2 should only run after step 1
succeeds" — it just runs things at fixed times and hopes for the best.

## Concrete problems cron doesn't solve

**1. Dependencies between jobs**
If the ingest job (step 1) is scheduled for 1am and normally takes 45
minutes, you might schedule the Spark job (step 2) for 2am to be safe. But
what happens the one night ingest takes 90 minutes because of an unusually
large data volume? Cron doesn't know or care — it fires the 2am job
regardless, and it now runs on incomplete/stale data. This is a real,
common, and quietly dangerous failure mode: **the pipeline doesn't crash,
it just silently produces wrong output.**

**2. Retries**
If a job fails at 2am — a transient network blip, a momentarily
unavailable database — cron just... doesn't run it again until tomorrow.
Someone has to notice and manually rerun it. At any real scale (dozens or
hundreds of jobs), this becomes unmanageable by hand.

**3. Backfills**
Suppose you discover a bug in a transformation that's been running for the
last 2 weeks, and you need to rerun that job for each of the last 14 days
with the fix. Cron has no built-in concept of "rerun this job as if it were
each of these past 14 dates" — you'd be manually invoking the script 14
times with different date parameters, hoping you don't miss one or
duplicate one.

**4. Observability**
When something breaks in a pipeline of 50 interdependent jobs, "which job
failed, why, and what depends on it" is a real question that needs an
actual answer — a dashboard, logs, alerting — not just an email cron
happens to send on failure (if you even configured that).

---

## The core abstraction: DAGs

The concept that solves all four problems at once is representing your
pipeline as a **DAG** — a **Directed Acyclic Graph**. Breaking that name
down:

- **Directed:** edges have a direction — "job A must finish before job B
  starts" (not a symmetric relationship).
- **Acyclic:** no cycles — you can't have "A depends on B, and B depends on
  A" (that would be a deadlock; nothing could ever start).
- **Graph:** a set of nodes (jobs/tasks) connected by edges (dependencies).

```
   ingest_events
        │
        ▼
   spark_join_and_aggregate
        │
        ▼
   write_to_warehouse
        │
        ▼
   refresh_dashboard
```

Once your pipeline is expressed this way, an orchestrator (Airflow, Ch15)
can automatically:
- **Only run a job once its upstream dependencies have actually
  succeeded** — solving problem #1. `spark_join_and_aggregate` literally
  cannot start until `ingest_events` has reported success, regardless of
  wall-clock time.
- **Retry a failed job automatically** (with configurable retry counts/
  backoff) — solving problem #2.
- **Rerun the whole DAG (or part of it) for a specific past date** —
  solving problem #3 (this is what "backfilling" concretely means in an
  orchestration tool).
- **Show you, visually, exactly which job failed and what's blocked
  waiting on it** — solving problem #4.

**This is the single most important idea in this module:** orchestration
tools aren't really about "running scripts on a schedule" (cron already
does that) — they're about **managing dependencies, failures, and history**
across many interrelated jobs, which is a fundamentally different, harder
problem.

---

## Worked example

Back to the recommendation pipeline, as a DAG:

```
ingest_kafka_events_to_lake
        │
        ▼
spark_join_events_with_products   ──┐
        │                            │
        ▼                            ▼
compute_copurchase_stats      compute_daily_summary_report
        │                            │
        ▼                            ▼
write_stats_to_warehouse      refresh_analytics_dashboard
```

Notice this isn't a single straight line — `spark_join_events_with_products`
has **two** downstream jobs that can run in parallel once it completes
(the co-purchase stats path and a separate daily summary report path).
This is a completely normal DAG shape, and it's exactly the kind of
structure cron has no way to express — cron can only say "run this at this
time," not "run this once these two independent things are both done."

If `ingest_kafka_events_to_lake` fails at 1am, an orchestrator holds
*everything* downstream — nothing runs on stale/missing data, and once the
ingest job is retried and succeeds (whether automatically or manually
triggered), the rest of the DAG proceeds from there.

---

## Downstream considerations

1. **Latency:** A DAG-based approach can actually reduce end-to-end
   latency compared to conservative, padded cron scheduling — instead of
   guessing generous buffer times between jobs "just in case," downstream
   jobs start the *moment* their dependencies actually finish, not at a
   fixed clock time chosen defensively.
2. **Consistency:** This chapter is fundamentally about preventing a
   pipeline from silently running on incomplete or stale upstream data —
   the DAG dependency model is what guarantees a downstream job only ever
   sees data from an upstream job that's actually confirmed complete.
3. **Cost/scale:** Manual firefighting (someone noticing a failed cron job
   and re-running it by hand) doesn't scale past a handful of jobs —
   orchestration tooling is what lets a data platform grow to dozens or
   hundreds of interdependent pipelines without needing proportionally
   more people babysitting them.
4. **Failure mode:** The scariest cron failure mode isn't a job crashing
   loudly — it's a job running "successfully" on stale or partial upstream
   data because the timing assumption quietly broke. This is worth naming
   explicitly as the core motivation for this whole module if asked "why
   not just use cron."

---

## Quick recap

- Cron handles scheduling but has no concept of dependencies, automatic
  retries, backfills, or built-in observability across multiple jobs.
- The core failure mode this causes: a downstream job silently runs on
  stale/incomplete data because a timing assumption broke, with no error
  raised.
- A DAG (directed acyclic graph) represents a pipeline as jobs +
  dependencies rather than jobs + fixed times, which is what lets an
  orchestrator enforce "only run once dependencies actually succeed,"
  retry failures, backfill history, and expose what's happening.

---

## Interview-style Q&A

**Q: Why isn't cron sufficient for a multi-step data pipeline?**
A: Cron only knows about wall-clock time, not job dependencies — it can't
express "only run step 2 once step 1 has actually succeeded." This means a
downstream job can silently run on stale or incomplete upstream data if
the upstream job runs longer than expected or fails, with no built-in
retry, backfill, or visibility into what happened.

**Q: What does a DAG give you that a linear list of scheduled scripts
doesn't?**
A: It captures the actual dependency structure between jobs — including
jobs that can run in parallel once a shared dependency completes — and
lets an orchestrator enforce those dependencies at runtime (only start a
job once its upstream jobs report success), rather than relying on
manually-tuned, defensively-padded time gaps between scheduled scripts.

**Q: What's the most dangerous kind of pipeline failure, and how does
DAG-based orchestration address it?**
A: A downstream job running "successfully" on stale or partial upstream
data — no error is raised, but the output is silently wrong. DAG-based
orchestration prevents this by only allowing a job to start once its
declared upstream dependencies have actually completed successfully,
rather than assuming they're done because enough wall-clock time has
passed.

---

Next: **Ch15 — Airflow Fundamentals** (DAGs, tasks, operators, scheduler,
idempotency). Say "ch15" when ready.
