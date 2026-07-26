# Chapter 1: The Big Picture — What Is a Data Pipeline

## Why an MLE even needs this chapter

You will never be asked to build an Airflow DAG in an MLE interview. But you
will absolutely be asked things like:

- "How would the training data for this model get produced and refreshed?"
- "Your model's predictions look great offline but bad in production — where
  would you look first?"
- "Design a real-time recommendation system" (this is, underneath, a data
  pipeline design question wearing an ML costume)

Almost every one of these questions is really asking: **do you understand
where data comes from, what happens to it before it reaches your model, and
what can go wrong along the way?** That's the whole subject of this syllabus.
This chapter gives you the skeleton everything else hangs on.

---

## What is a data pipeline? (plain language, no jargon)

A data pipeline is just a series of steps that moves data from where it's
*produced* to where it's *useful*, transforming it along the way.

Think of it like a physical supply chain:

```
  Raw materials  →  Factory  →  Warehouse  →  Store shelf  →  Customer
  (source data)     (process)   (storage)     (serving)       (consumer:
                                                                 a model,
                                                                 a dashboard,
                                                                 a person)
```

In data terms, the same chain looks like:

```
  SOURCE  →  INGEST  →  STORE  →  PROCESS  →  SERVE
```

Let's define each link, because every later chapter is really just "zoom in
on one of these five boxes."

### 1. Source
Where data is born. Examples:
- App/website event logs (user clicked X, user viewed Y)
- Production databases (the Postgres/MySQL table backing your app)
- Third-party APIs (payment processor, ad platform)
- Sensors/IoT devices
- Files dropped by another team/vendor

**Key property to notice:** sources are almost never designed for analytics
or ML. A production database is optimized for your app being fast for one
user at a time, not for you scanning 5 years of history. This mismatch is
*the* reason data pipelines exist at all — you can't just point your model
training job at the live production database.

### 2. Ingest
The act of *getting* data out of the source and into a system built for
processing/analysis. This is where **batch vs. streaming** (Ch2) becomes a
decision: do you pull a snapshot every night, or do you capture every event
as it happens?

### 3. Store
Where the data lands and lives. This is where **data lake vs. warehouse**
(Ch3) and **file formats** (Ch4) come in — different storage choices make
different downstream work fast or slow, cheap or expensive.

### 4. Process
Transforming raw data into something usable: cleaning, joining, aggregating,
computing features. This is where **Spark** (Module B) lives.

### 5. Serve
Making the processed data available to whoever/whatever needs it: a BI
dashboard, an analyst running SQL, or — the case you care about most — a
**model**, either reading a table of training data or querying a **feature
store** for a live prediction.

---

## How it actually works, end to end (a worked example)

Let's ground this in something concrete: **an e-commerce "customers who
bought X also bought Y" recommendation model.**

1. **Source:** Every time a user views or buys a product, the website emits
   an event: `{user_id, product_id, action: "view"|"purchase", timestamp}`.
2. **Ingest:** These events are published to a message queue (Kafka) in
   real time as they happen, *and* a separate nightly job pulls a full
   snapshot of the `orders` table from the production database.
3. **Store:** The Kafka events get written into a data lake (as Parquet
   files, partitioned by date) for historical access; the nightly order
   snapshot lands in a data warehouse table.
4. **Process:** A Spark job runs nightly, joining view/purchase events with
   product metadata, and computes co-purchase statistics ("users who bought
   A bought B 40% of the time") — this becomes your training data.
5. **Serve:** The trained model's outputs (or the co-purchase table itself)
   get pushed into a low-latency store (e.g., Redis) that the live website
   queries in milliseconds when a user loads a product page.

Notice: this single example already used batch *and* streaming, a lake *and*
a warehouse, and both a "training-time" path and a "serving-time" path. Real
systems mix these constantly — there's rarely one pure approach.

---

## Why this matters — the MLE-specific lens

Here's the mental shift from "data engineer" thinking to "ML engineer"
thinking: **you care about this pipeline because your model has two
lifelines running through it, and they must agree with each other.**

```
   Training-time path:  Source → Ingest → Store → Process → Training data
   Serving-time path:   Source → Ingest → Store → Process → Live features
```

If the *process* step computes a feature differently in these two paths
(e.g., "average order value over last 30 days" computed one way in a nightly
Spark job, and a slightly different way in a real-time service), your model
will perform well offline and mysteriously worse in production. This is
**training-serving skew**, and it is one of the most common real-world ML
bugs — and it is fundamentally a *data pipeline* problem, not a modeling
problem. You'll see this again when we cover feature stores (Ch18), because
feature stores exist specifically to solve this.

---

## Downstream considerations (what to keep asking at every later chapter)

As we go chapter by chapter, keep interrogating every new tool/concept with
these four questions — they're the ones interviewers actually care about:

1. **Latency:** How fresh does this data need to be by the time it reaches
   the model? (Milliseconds? Hours? Once a day?)
2. **Consistency:** Could this step produce a different answer if computed
   twice (e.g., once for training, once for serving)? If yes, that's a skew
   risk.
3. **Cost/scale:** Does this approach still work at 100x the data volume?
   What breaks first?
4. **Failure mode:** If this step fails or lags, what does the model
   downstream actually experience — stale features? Missing data? A crash?

Every remaining chapter (Spark, Kafka, Airflow, lake vs. warehouse) is best
understood as "here's a tool, and here's how it answers those four
questions differently than the alternatives."

---

## Quick recap

- A data pipeline = Source → Ingest → Store → Process → Serve.
- Sources are built for operations, not analytics — that mismatch is why
  pipelines exist.
- Real systems blend batch and streaming, lakes and warehouses — it's rarely
  one clean approach.
- As an MLE, your real concern is that the **training path** and **serving
  path** through this pipeline stay consistent — inconsistency there is the
  root cause of training-serving skew.
- Every later chapter is a deep dive into one box in this pipeline, viewed
  through: latency, consistency, cost/scale, failure mode.

---

## Interview-style Q&A

**Q: Why can't you just train your model directly off the production
database?**
A: Production databases are optimized for fast, small, transactional reads/
writes for the live app — not for scanning millions of historical rows,
which is what training needs. Running heavy analytical queries against it
also risks slowing down or crashing the live application. So data is
extracted into a separate system (a lake/warehouse) built for large scans.

**Q: What's the difference between the "process" step happening for
training data vs. for serving?**
A: For training, processing usually happens in batch, over historical data,
with no strict latency requirement. For serving, processing may need to
happen in real time (or the result needs to already be precomputed and
cached) so a live prediction request isn't waiting on a slow Spark job. The
risk is that these two code paths compute the "same" feature differently —
training-serving skew.

**Q: Give an example of a failure mode at the ingest step and its downstream
effect on a model.**
A: If the nightly ingest job silently fails for a day, the model's training
data (or a feature store's daily-refreshed features) goes stale by a day.
The model doesn't crash — it just quietly serves worse predictions using
yesterday's-yesterday's data, which is often *harder* to catch than an
outright failure.

---

Next: **Ch2 — Batch vs. Streaming**, where we unpack the ingest-side decision
that shapes almost everything downstream. Say the word when you want it.
