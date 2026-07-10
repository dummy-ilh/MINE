# Session 1: Data for ML Systems — First Principles

## The Core Question

Before any ML system, you must answer:

> **What human behavior am I trying to predict, and what evidence of that behavior exists in the world?**

This is the most important question in ML system design. Everything else — storage, pipelines, models — is just machinery to answer it.

---

## Mental Model: The Signal Pyramid

```
                    /\
                   /  \
                  / LA \          <- Explicit Labels (rare, expensive)
                 /------\
                / IMPL.  \        <- Implicit Signals (abundant, noisy)
               /----------\
              /   CONTEXT   \     <- Contextual Features (who, when, where)
             /--------------\
            /   RAW EVENTS   \    <- Everything the system logs
           /------------------\
```

Every ML system is built on this pyramid. Let's go layer by layer.

---

## Layer 1 — Raw Events (The Foundation)

Every time a user does **anything**, your system should log it.

For YouTube, this means:

```
User 123 opened app          → timestamp, device, location
User 123 saw video A         → impression
User 123 clicked video A     → click, timestamp
User 123 watched 30 seconds  → watch time
User 123 closed app          → session end
```

These are **raw events**. They have no label yet. They are just facts.

**Why log everything?**
Because you don't know today what your model will need tomorrow. Storage is cheap. Missing historical data is **irreversible**.

This is called **data logging for future ML** and it's one of the most important engineering decisions a team makes early on.

---

## Layer 2 — Implicit Signals

Raw events become signals when you interpret them as **evidence of user preference**.

| Raw Event | Implicit Signal | Why |
|---|---|---|
| Clicked video | Mild interest | But could be clickbait |
| Watched 90% of video | Strong interest | Revealed preference |
| Watched 10% then left | Probably didn't like it | But maybe wrong format |
| Clicked Like | Explicit positive | But rare, biased users |
| Shared video | Very strong signal | High-effort action |
| Opened then immediately closed | Negative signal | Misleading thumbnail |

**Key insight:** Not all signals are equal. Higher effort = stronger signal of true preference.

This is called **engagement signal hierarchy** and it matters enormously for label construction.

---

## Layer 3 — Explicit Labels

Sometimes you ask users directly:

- Thumbs up / thumbs down
- Star ratings
- Surveys

**Why not just use these?**

Three problems:

**1. Sparsity** — Only ~1% of users rate anything. You can't train a model on 1% of interactions.

**2. Selection bias** — Who rates things? Extreme responders. People who loved it or hated it. The middle majority is silent.

**3. Reporting bias** — People rate what they think they *should* like, not what they actually enjoyed. (Someone might rate a documentary 5 stars but watch reality TV for hours.)

So explicit labels are **high quality but low volume**. Implicit signals are **low quality but high volume**. Production systems use both.

---

## Layer 4 — Context

The same user wanting different things at different times is one of the hardest problems in recommendations.

Context includes:

```
WHO    → user_id, age, history, preferences
WHAT   → query, current video, session history  
WHEN   → time of day, day of week, recency
WHERE  → device type, country, connection speed
HOW    → how did they arrive here? search? autoplay?
```

**Why does context matter so much?**

The same person on Monday morning commuting on mobile wants something different than Friday night on a TV.

Without context, your model learns *average* behavior. With context, it learns *situational* behavior. Big difference in quality.

---

## The Label Construction Problem

Here's something most data scientists don't think about:

> **Your model learns exactly what you measure. If you measure the wrong thing, you get the wrong model.**

This is called **proxy label problem** and it causes some of the biggest real-world ML failures.

Example at YouTube:

```
If you optimize for → CLICKS
Model learns       → recommend clickbait thumbnails

If you optimize for → WATCH TIME  
Model learns       → recommend long videos, even low quality ones

If you optimize for → SATISFACTION (survey)
Model learns       → but data is too sparse to generalize

Real solution      → weighted combination of signals
```

This is why YouTube moved to a **multi-objective** approach. No single metric captures "did the user genuinely enjoy this?"

---

## Data Collection Architecture — First Principles

Now we can talk infrastructure. But notice: we chose **what** to collect before **how** to collect it.

```
USER ACTION
    │
    ▼
[Client SDK]  ← logs event with metadata
    │
    ▼
[Event Bus]   ← Kafka / Pub-Sub
(Kinesis)       high throughput, durable, ordered
    │
    ├──────────────────────┐
    ▼                      ▼
[Stream Processing]   [Raw Event Store]
(Flink / Dataflow)    (GCS / S3)
Sessionization         Cold storage
Feature computation    Replay capability
    │
    ▼
[Feature Store / Data Warehouse]
(BigQuery / Hive)
Training data ready
```

**Why an Event Bus in the middle?**

Because producers (apps) and consumers (ML pipelines, analytics, monitoring) should be **decoupled**. If your training pipeline is slow, it shouldn't slow down your app. The bus acts as a buffer.

This pattern is called **event-driven architecture** and it's foundational at Google scale.

---

## Common Pitfalls

**1. Logging too late**
Teams build the product first, then realize they have no training data. Always instrument logging on day one.

**2. Survivorship bias in data**
You only log what your *current* system showed users. Your model never sees what it *didn't* recommend. This is called **exposure bias** and it causes feedback loops.

**3. Position bias**
Users click the first result more regardless of quality. If you don't correct for this, your model thinks position-1 items are just better.

**4. Training-serving skew**
Features computed differently at training time vs serving time. Causes silent model degradation. Very common, very painful.

---

## Summary — What to Remember

```
1. Log raw events first. Storage is cheap, missing data is permanent.

2. Implicit signals > explicit labels for volume.
   Explicit labels > implicit signals for quality.
   Use both.

3. What you measure is what you get. 
   Wrong proxy label = wrong model behavior.

4. Context transforms a good model into a great one.

5. Decouple data producers from consumers via an event bus.
```

---

