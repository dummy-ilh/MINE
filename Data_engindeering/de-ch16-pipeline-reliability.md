# Chapter 16: Pipeline Reliability & Data Quality

## Closing out Module D

Ch14–15 covered *how* pipelines get scheduled and run reliably from an
execution standpoint. This chapter is about a different, equally important
kind of reliability: **is the data itself correct**, and how do you catch
it when it isn't — including tying together training-serving skew, which
has been building since Ch1, into one coherent picture.

---

## Data validation: catching bad data before it spreads

The core idea: don't wait for a human to notice a model degrading — build
automated checks into the pipeline itself that catch bad data at the
moment it's produced, before it flows downstream into training data,
features, or dashboards.

**Common categories of checks** (this is the "great expectations"-style
thinking mentioned in the syllabus — conceptual, not a specific tool):

- **Schema checks:** does the data have the expected columns, with the
  expected types? (E.g., did `purchase_amount` suddenly start arriving as
  a string instead of a number, because an upstream service changed its
  logging format?)
- **Range/sanity checks:** are values within plausible bounds? (E.g.,
  `age` should be between 0 and 120; a `purchase_amount` of -500 or
  50,000,000 is almost certainly a bug, not a real transaction.)
- **Null/completeness checks:** is a critical field unexpectedly null or
  missing for a large fraction of rows? (E.g., if `user_id` is null for
  30% of today's events, something upstream broke.)
- **Referential checks:** do foreign keys actually resolve? (E.g., does
  every `product_id` in the events table actually exist in the products
  table?)

**Where these checks live:** typically as their own tasks in the Airflow
DAG (Ch15) — e.g., a validation task runs right after ingest and *before*
the expensive Spark join, so a bad batch fails fast and cheap, rather than
silently flowing through the whole pipeline and corrupting a training
dataset or a dashboard.

---

## Backfills and reprocessing historical data safely

Ch14 introduced backfilling as a reason orchestration tools exist. Now the
practical question: what makes a backfill *safe*?

This connects directly back to **idempotency (Ch15)**: if every task
overwrites/upserts rather than appends, rerunning a DAG for a past date
(a backfill) produces the exact same correct result as if it had run
correctly the first time — no duplicate data, no manual cleanup needed.

A concrete scenario: you find a bug in the co-purchase stats calculation
that's been live for 2 weeks. Because each day's task is idempotent
(overwrites that day's partition), you can trigger a backfill for the last
14 days, and Airflow will simply rerun each day's DAG with the fix,
overwriting each day's previously-wrong output with the corrected version
— cleanly, with no risk of duplicating anything. This is the concrete
payoff of investing in idempotency upfront.

---

## Monitoring: freshness, volume, schema drift

Validation (above) catches bad data *within* a single run. Monitoring is
about catching problems **across time** — is the pipeline behaving
normally compared to its own history?

- **Freshness:** how old is the most recent data in a table? If a table
  that should update daily hasn't updated in 3 days, something's broken
  upstream — even if no single run "failed" (e.g., maybe the DAG is
  disabled, or a sensor is stuck waiting on a condition that will never
  arrive).
- **Volume:** is today's row count wildly different from a typical day?
  (E.g., 10x fewer purchase events than a normal day likely means an
  upstream ingestion problem, not that purchases genuinely dropped 90%
  overnight.)
- **Schema drift:** did the shape of the data change unexpectedly over
  time — a new column appearing, a column's type silently changing, a
  categorical field suddenly containing new/unexpected values? This is
  especially dangerous for ML because a model trained on the old schema
  can silently receive malformed input without erroring.

These are typically implemented as scheduled checks or dashboards
comparing recent runs against historical baselines — distinct from the
per-run validation checks above, and worth being able to articulate as a
separate concern in an interview.

---

## Training-serving skew as a data-pipeline problem — pulling the thread together

This concept has been seeded since Ch1, referenced again in Ch2, Ch9, and
Ch13 — this is the chapter to state it as one coherent idea, because it's
one of the most likely deep-dive questions in an MLE interview.

**The core claim:** training-serving skew is very often *not* a modeling
problem at all — it's a data pipeline consistency problem, and everything
in Module D exists partly to prevent it:

- If training data is computed via a batch Spark job (Ch9) but serving
  features are computed via a separate real-time stream processor (Ch13),
  any divergence in their logic — even a subtle one, like slightly
  different time-window boundaries or different handling of null values —
  produces two different answers for "the same" feature. The model was
  trained on one version of reality and is being asked to predict on
  another.
- **Validation checks (this chapter)** catch data *quality* problems
  (nulls, bad ranges, broken schemas) but generally do **not** by
  themselves catch skew — skew is a subtler *consistency between two
  pipelines* problem, not a within-one-pipeline correctness problem. This
  distinction is worth being precise about if pressed in an interview.
- **The most robust fix** is architectural, not just "be careful": share
  the actual transformation logic between the training and serving paths
  wherever possible (rather than maintaining two independently-written
  implementations of "the same" feature) — this is exactly the problem
  feature stores (Ch18) are built to solve, and why they're worth knowing
  about even briefly.
- Monitoring (this chapter's freshness/volume/drift checks) is your
  detection mechanism for skew *after the fact* — e.g., if offline model
  evaluation metrics and online production metrics start diverging over
  time in a way validation checks don't explain, pipeline-level skew
  between training and serving logic is a prime suspect worth
  investigating first.

---

## Worked example, tying the whole module together

Full reliability picture for the recommendation pipeline:

1. `wait_for_raw_events` (sensor, Ch15) confirms the day's data has landed.
2. `validate_raw_events` (new — this chapter) runs schema, null, and
   volume checks on the raw events before anything expensive happens — if
   volume is 10x lower than the historical baseline, the DAG fails fast
   here and alerts a human, rather than silently producing a
   nearly-empty training dataset.
3. `spark_join` (idempotent, Ch15) computes co-purchase stats.
4. `validate_output` (this chapter) checks the output table's row counts
   and schema look sane before downstream consumers (dashboard, training
   pipeline) read it.
5. Separately, an ongoing **monitoring** dashboard tracks freshness of
   this whole pipeline's output daily, and compares the real-time
   (streaming) version of the "purchases in last 10 minutes" feature
   against the batch-computed version for spot-checking — a direct,
   concrete skew-detection mechanism.
6. If a bug is later found in step 3's logic, the pipeline's idempotent
   design (Ch15) means a clean 14-day backfill fixes history without
   manual cleanup.

---

## Downstream considerations

1. **Latency:** Validation checks add some latency to every pipeline run —
   worth explicitly weighing against the cost of *not* catching bad data
   (which is almost always higher, since bad data caught late means
   redoing more downstream work, or worse, a degraded model already
   serving predictions).
2. **Consistency:** This entire chapter is fundamentally about
   consistency — within a run (validation), across time (monitoring), and
   across systems (training-serving skew) — worth explicitly naming these
   as three distinct flavors of the same underlying concern if asked to
   compare them.
3. **Cost/scale:** Catching a data quality issue early (right after
   ingest) is far cheaper than catching it late (after it's already
   propagated through an expensive Spark join and corrupted a training
   dataset) — this "fail fast, fail cheap" framing is a strong, concrete
   answer if asked to justify investing in validation infrastructure.
4. **Failure mode:** The scariest failure mode in this whole module,
   worth naming explicitly: a pipeline that runs "successfully" (no errors
   anywhere) but silently produces bad data or skewed training/serving
   features — nothing crashes, no alert fires, and the first sign of
   trouble is degraded model performance in production, discovered days
   or weeks later.

---

## Quick recap

- Data validation (schema, range, null, referential checks) catches bad
  data within a single run, ideally placed early in the DAG to fail fast
  and cheap.
- Idempotent tasks (Ch15) are what make backfills safe — rerunning
  history overwrites cleanly rather than duplicating.
- Monitoring (freshness, volume, schema drift) catches problems across
  time, distinct from per-run validation.
- Training-serving skew is fundamentally a data-pipeline consistency
  problem between two separate paths (training/batch vs. serving/
  real-time) — validation alone doesn't catch it; shared transformation
  logic (feature stores) is the architectural fix, and monitoring is the
  after-the-fact detection mechanism.

---

## Interview-style Q&A

**Q: What's the difference between data validation and data monitoring in
a pipeline?**
A: Validation checks a single run's data for correctness — schema, nulls,
value ranges, referential integrity — typically as a task early in the
DAG, so bad data fails fast before propagating downstream. Monitoring
instead tracks pipeline health across time — freshness, volume trends,
schema drift — catching gradual or silent problems that don't show up as
an outright failure in any single run.

**Q: Why doesn't data validation, by itself, catch training-serving
skew?**
A: Validation confirms that data within one pipeline is well-formed and
plausible — but skew is a mismatch *between two separate pipelines*
(training's batch computation vs. serving's real-time computation) that
can both individually pass all validation checks while still computing
"the same" feature differently. Catching that requires comparing the two
paths against each other, or architecturally sharing the transformation
logic between them, not just validating each path in isolation.

**Q: Why does idempotency (from Ch15) matter specifically for backfills?**
A: A backfill reruns historical DAG executions, often for many past dates
at once. If tasks aren't idempotent, rerunning them duplicates or corrupts
already-existing output for those dates. Idempotent tasks (overwrite/
upsert rather than append) make backfills a clean, safe operation —
rerunning history produces the same correct end state regardless of how
many times it's run.

---

That closes **Module D (Orchestration)**. Next up is **Module E — Tying It
Together**, starting with **Ch17: ETL vs. ELT**. Say "ch17" when ready.
