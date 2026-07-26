# Chapter 17: ETL vs. ELT

## Opening Module E

This module pulls the whole syllabus together. Ch17–18 fill in two
remaining vocabulary/concept gaps (ETL/ELT, feature stores) that lean on
everything built so far, before Ch19's full worked system and Ch20's rapid
review.

---

## ETL: the traditional pattern

**Extract → Transform → Load.** The order tells you exactly what happens:

1. **Extract:** pull raw data out of a source system (Ch1's sources —
   production databases, APIs, logs).
2. **Transform:** clean, join, aggregate, reshape the data — **before** it
   ever lands in its destination. This transformation typically happens on
   a separate processing system (historically, dedicated ETL tools; today,
   often Spark, Ch5-9).
3. **Load:** write the *already-transformed* result into the destination
   (traditionally a data warehouse, Ch3).

**Why this order historically made sense:** warehouses used to be
expensive, and compute/storage were tightly coupled and costly. You didn't
want to load messy, oversized raw data into an expensive warehouse and
then pay warehouse compute prices to clean it up there — better to do the
heavy transformation work on cheaper, separate infrastructure first, and
only load the clean, final result.

## ELT: the modern, cheaper-storage pattern

**Extract → Load → Transform.** Same first two words, different order —
and that reordering reflects a real shift in the underlying economics:

1. **Extract:** same as before.
2. **Load:** load the **raw** data into the destination *immediately*,
   before transforming it — this is only reasonable because modern lakes/
   warehouses/lakehouses (Ch3) have made storage cheap and decoupled from
   compute.
3. **Transform:** do the cleaning/joining/aggregating **inside** the
   destination system itself (e.g., running SQL transformations directly
   in a warehouse like Snowflake/BigQuery, or Spark jobs directly against
   a lakehouse), rather than on a separate upstream system.

**Why this became preferable:** cheap object storage (Ch3) means loading
raw, un-transformed data first "just in case" costs very little, and
modern warehouses/lakehouses have powerful enough built-in compute that
transforming data *in place* is often simpler and just as fast as staging
it externally first. It also means the raw data is always available and
preserved — if a transformation turns out to be wrong or a new use case
appears later, you still have the untouched original to reprocess from,
rather than only ever having kept the already-transformed result.

---

## The core tradeoff, made concrete

| | ETL | ELT |
|---|---|---|
| **Transform happens** | Before loading, on separate infra | After loading, inside the destination |
| **Raw data preserved?** | Often not — only the transformed result is kept | Yes — raw data sits in the lake/warehouse too |
| **Flexibility for new use cases** | Lower — if you need a different transformation later, you may need to re-extract from source | Higher — raw data is already there; just write a new transform |
| **Upfront infra needed** | A separate transformation system before the destination | The destination itself needs to be powerful enough to transform in place |
| **Historical fit** | Pre-cloud, when storage/compute were expensive and coupled | Modern cloud lakes/warehouses/lakehouses with cheap, decoupled storage |

**One-line summary for interviews:** *ELT became the modern default because
storage got cheap enough that "load first, transform later, and keep the
raw copy around" beats "transform before loading and only keep the
result," especially given how often new use cases emerge that need the
raw data you didn't think to keep under ETL.*

---

## When ETL is still the right call

ELT being "modern" doesn't mean ETL is obsolete — worth being able to name
exceptions, since a blanket "always ELT" answer reads as memorized rather
than understood:

- **Sensitive data that must be cleaned/redacted before landing anywhere**
  (e.g., PII that legally cannot be stored raw even temporarily) — you
  have to transform (redact) *before* load, not after, by necessity.
  ELT's whole premise (raw data lands first) doesn't work for a
  constraint like this.
- **Very constrained destination systems** where in-place transformation
  compute is genuinely expensive or limited — sometimes still cheaper to
  pre-process elsewhere first.
- **Extremely high-volume raw data you have no intention of ever keeping
  in full** (e.g., verbose debug logs) — loading 100% of it just to
  immediately discard 95% during an in-place transform can be wasteful
  compared to filtering during extraction.

---

## Worked example: mapping this onto the recommendation pipeline

The pipeline built up across earlier chapters is, in fact, an **ELT**
pattern, even though it was never labeled that way until now:

1. **Extract:** Kafka events, production order snapshots (Ch1).
2. **Load:** raw events land in the data lake as Parquet, **before** any
   joining/aggregation happens (Ch1, Ch4) — this is the "load raw first"
   ELT signature.
3. **Transform:** the Spark job (Ch5-9) — which itself runs *against* the
   lake, i.e., inside the broader lakehouse ecosystem — does the actual
   join/aggregate/feature-computation work, producing the co-purchase
   stats table.

Because the raw events were preserved in the lake at step 2 (rather than
discarded after an upfront ETL transform), if a new team later wants a
completely different feature computed from the same raw purchase events
(say, "average basket size per session" for a totally different model),
they can simply write a new transform against the already-loaded raw data
— no need to go back and re-extract from the original Kafka topic/source
system, which might not even retain that history anymore (recall Ch11's
retention policy discussion).

---

## Downstream considerations

1. **Latency:** ELT can actually get raw data into the destination faster
   (skip the separate transform-before-load step), even though the
   *transformed* result might not be ready any sooner — worth
   distinguishing "when is raw data available" from "when is the
   transformed/usable result available" as two different latency
   questions.
2. **Consistency:** Preserving raw data (ELT's core advantage) is directly
   useful for training-serving skew debugging (Ch16) — if you suspect a
   transformation diverged between training and serving, having the
   original raw data still available lets you recompute and compare both
   paths from the same source, rather than only having two already-
   transformed, hard-to-reconcile outputs.
3. **Cost/scale:** ELT trades cheaper storage cost (keeping raw data
   around) for compute cost inside the destination system when
   transforming — usually a good trade given how cheap object storage has
   become, but worth being able to state as an explicit tradeoff rather
   than an unqualified win.
4. **Failure mode:** Under ELT, if a transformation has a bug, you can
   simply fix it and rerun against the still-present raw data (echoing
   Ch16's backfill discussion). Under strict ETL where raw data wasn't
   retained, the same bug might require re-extracting from the original
   source — which may no longer be possible if that source has since
   changed or the data has aged out (e.g., past a Kafka retention window).

---

## Quick recap

- ETL transforms data before loading it into the destination; ELT loads
  raw data first and transforms it in place afterward.
- ELT became the modern default because cheap, decoupled cloud storage
  makes "keep the raw copy" affordable, and it preserves flexibility for
  future, not-yet-anticipated use cases.
- ETL still makes sense when data must be cleaned/redacted before landing
  anywhere (legal/compliance constraints), or when raw volume genuinely
  isn't worth preserving.
- The recommendation pipeline built across this syllabus is itself an ELT
  pattern — raw events load into the lake first, transformation happens
  afterward via Spark.

---

## Interview-style Q&A

**Q: Why did ELT become more common than traditional ETL in modern data
platforms?**
A: Cloud storage got cheap and decoupled from compute, making it
affordable to load raw data into a lake/warehouse first and transform it
in place afterward, rather than transforming on separate infrastructure
before loading. This also preserves the original raw data for future,
not-yet-anticipated use cases — a real advantage over ETL, where often
only the already-transformed result is kept.

**Q: Is there a case where ETL is still the right choice over ELT?**
A: Yes — when data must be cleaned or redacted before it's allowed to land
anywhere at all, such as PII that can't legally be stored raw even
temporarily. In that case transformation has to happen before load by
necessity, which is exactly the ETL pattern; ELT's core premise of loading
raw data first doesn't work under that constraint.

**Q: How does the ELT pattern help when debugging training-serving skew?**
A: Because raw data is preserved in the lake rather than discarded after
an upfront transform, you can recompute a feature from the original raw
source and directly compare it against both the training-path and
serving-path outputs, rather than only having two already-transformed
results with no shared, unprocessed baseline to reconcile them against.

---

Next: **Ch18 — Feature Stores** (offline vs. online, and how they solve
skew architecturally). Say "ch18" when ready.
