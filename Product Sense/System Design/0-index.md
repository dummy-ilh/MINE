
# Full Curriculum — Teaching Phase

Here's exactly what we'll cover and in what order:

```
WEEK 1 — DATA FOUNDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━
Session 1  ✓ Data Collection & Signals (done)
Session 2  → Feature Engineering
Session 3  → Feature Stores
Session 4  → Embeddings
Session 5  → Offline Evaluation

WEEK 2 — CORE ML SYSTEMS
━━━━━━━━━━━━━━━━━━━━━━━━━
Session 6  → Candidate Generation & Retrieval
Session 7  → Ranking
Session 8  → Training Pipelines
Session 9  → Distributed Training
Session 10 → Hyperparameter Tuning

WEEK 3 — SERVING & INFRASTRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Session 11 → Online Serving & Real-time Inference
Session 12 → Batch & Streaming Inference
Session 13 → Vector Search & ANN
Session 14 → Caching
Session 15 → Model Registry & Deployment

WEEK 4 — PRODUCTION MATURITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Session 16 → Canary, Shadow Deployment & Rollback
Session 17 → Monitoring & Drift Detection
Session 18 → Data Validation
Session 19 → Retraining Strategies
Session 20 → Reliability, Cost & Privacy
```

---

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



# Session 2: Feature Engineering

## The Core Question

> **How do you turn raw human behavior into numbers a model can reason about?**

This is feature engineering. It sounds mechanical but it's actually where most of your modeling intuition lives.

---

## First Principle: Models Only See Numbers

A model cannot see:

```
"User Alice watched a cooking video on Saturday night on her iPhone"
```

It can only see:

```
[0.23, 1.0, 0, 6, 22, 0.87, ...]
```

Feature engineering is the translation layer between the human world and the mathematical world.

**Key insight:** The quality of this translation determines your ceiling. A perfect model on bad features will lose to a mediocre model on great features. Every time.

---

## Feature Categories

There are five types of features in production ML systems:

```
┌─────────────────────────────────────────────────────┐
│  1. USER FEATURES      Who is this person?          │
│  2. ITEM FEATURES      What are we scoring?         │
│  3. INTERACTION        History between user & item  │
│  4. CONTEXT FEATURES   When/where/how?              │
│  5. CROSS FEATURES     Combinations of the above    │
└─────────────────────────────────────────────────────┘
```

Let's go through each with YouTube as our example.

---

## 1. User Features

These describe the user independent of any specific item.

```
STATIC (rarely change)
├── age_bucket         [18-24, 25-34, ...]
├── country            [US, IN, BR, ...]
├── language           [en, hi, pt, ...]
└── account_age_days   [1, 30, 365, ...]

DYNAMIC (change over time)
├── watch_history_30d         [video_id list]
├── avg_session_length_7d     [minutes]
├── preferred_categories_7d   [cooking:0.4, tech:0.3, ...]
├── active_hours              [morning, evening, ...]
└── device_type               [mobile, TV, desktop]
```

**Why separate static from dynamic?**

Static features can be precomputed once and cached. Dynamic features must be recomputed frequently. This separation matters enormously at serving time — you don't want to recompute account age on every request.

---

## 2. Item Features

These describe the video independent of any specific user.

```
CONTENT FEATURES
├── title_embedding        [dense vector]
├── thumbnail_embedding    [dense vector from CV model]
├── transcript_embedding   [dense vector from NLP model]
├── duration_seconds       [120, 600, 3600, ...]
├── category               [cooking, tech, news, ...]
└── language               [en, hi, ...]

POPULARITY FEATURES
├── views_total            [raw count]
├── views_7d               [recency-weighted]
├── ctr_7d                 [click-through rate]
├── avg_watch_pct_7d       [how much people finish it]
└── like_dislike_ratio     [engagement quality]

FRESHNESS
└── upload_timestamp       [hours since upload]
```

**Key insight:** Popularity features are powerful but dangerous. If you weight them too heavily, your system only recommends already-popular content. New creators never break through. This is called the **popularity bias** problem.

---

## 3. Interaction Features

These describe the historical relationship between a specific user and a specific item or category.

```
├── user_watched_this_video          [bool]
├── user_watched_same_creator        [bool]
├── user_avg_watch_pct_this_category [float]
├── days_since_user_last_watched_category [int]
└── user_engagement_with_similar_videos [float]
```

These are often your most predictive features because they capture **personalized affinity** directly.

---

## 4. Context Features

These capture the situation in which the recommendation happens.

```
├── time_of_day_bucket    [morning, afternoon, evening, night]
├── day_of_week           [weekday, weekend]
├── device_type           [mobile, tablet, TV, desktop]
├── connection_type       [wifi, 4g, 3g]
├── session_length_so_far [minutes]
├── last_video_watched    [video_id → embedding]
└── entry_point           [home, search, autoplay, notification]
```

**Why entry point matters:**

A user who arrived via search has explicit intent. A user on the home feed is browsing. A user on autoplay is in a passive consumption mode. The same user, three completely different mental states. Your model needs to know this.

---

## 5. Cross Features

These are combinations of features that capture interactions the model might not learn on its own.

```
user_country × video_language    → local content preference
time_of_day × device_type        → mobile morning = short videos
user_age × category              → gaming content skews younger
session_length × video_duration  → long session = ok with long videos
```

**Why not let the model learn these automatically?**

Deep learning models theoretically can. But in practice:

- You need enough data for the model to discover the interaction
- Explicit cross features help the model learn faster with less data
- They also make model behavior more interpretable

This is a classic **manual feature engineering vs learned representation** tradeoff.

---

## Feature Transformation

Raw values are rarely model-ready. You need to transform them.

### Numerical Features

```
PROBLEM: Raw numbers have very different scales
  age: 25
  views: 4,500,000
  duration: 347

SOLUTIONS:

1. Normalization (0 to 1)
   x_norm = (x - min) / (max - min)
   Good for: bounded features
   Bad for: outliers destroy the scale

2. Standardization (mean=0, std=1)
   x_std = (x - mean) / std
   Good for: normally distributed features
   Bad for: heavy-tailed distributions

3. Log Transform
   x_log = log(1 + x)
   Good for: counts, views, revenue (power law distributed)
   Very common in production

4. Bucketization / Binning
   age → [18-24], [25-34], [35-44]
   Good for: capturing non-linear relationships
   Model treats each bucket as a category
```

**Why log transform for view counts?**

View counts follow a power law. A few videos have billions of views, most have hundreds. Raw counts would make the model obsess over outliers. Log transform compresses the scale to something the model can reason about linearly.

---

### Categorical Features

```
PROBLEM: Models can't handle strings
  category = "cooking"

SOLUTIONS:

1. One-Hot Encoding
   cooking → [1, 0, 0, 0, 0]
   tech    → [0, 1, 0, 0, 0]
   Good for: low cardinality (< ~100 categories)
   Bad for: high cardinality (explodes dimensionality)

2. Label Encoding
   cooking → 3, tech → 7
   Bad for most models (implies ordering that doesn't exist)
   OK for tree-based models

3. Embedding Lookup
   cooking → [0.2, -0.5, 0.8, ...]  (learned dense vector)
   Good for: high cardinality (user IDs, video IDs, words)
   This is the standard approach at scale
```

**High cardinality is the key challenge.** YouTube has 800 million videos. You cannot one-hot encode video IDs. You must learn embeddings. This is why embeddings are their own entire session.

---

## Temporal Features — A Special Case

Time is one of the most important and most underused feature dimensions.

```
RAW TIMESTAMP: 2024-01-15 22:34:07

DERIVED FEATURES:
├── hour_of_day      → 22 (late night)
├── day_of_week      → 1 (Monday)
├── is_weekend       → 0
├── days_since_event → 45
└── cyclical encoding:
    sin(2π × hour/24), cos(2π × hour/24)
```

**Why cyclical encoding for time?**

Hour 23 and hour 0 are numerically far apart (23 vs 0) but temporally adjacent (11pm and midnight). Cyclical encoding using sin/cos wraps the number line into a circle so the model understands temporal proximity.

```
sin(2π × 23/24) ≈ sin(2π × 0/24)  ✓ close in value
```

This is a subtle but important detail that separates strong ML engineers from average ones.

---

## Recency Weighting

Not all historical behavior is equally relevant.

```
What you watched 2 years ago     → low relevance
What you watched last week       → medium relevance  
What you watched 10 minutes ago  → high relevance
```

A common pattern is **exponential decay weighting:**

```
weight = exp(-λ × days_since_event)

λ controls decay speed:
  λ = 0.01 → slow decay (long memory)
  λ = 0.1  → fast decay (short memory)
```

Your model's effective memory length is a hyperparameter. Some systems use different decay rates for different feature types.

---

## The Training-Serving Skew Problem

This is one of the most common and painful production ML bugs.

```
TRAINING TIME:
  You compute features from a database snapshot
  avg_watch_time_7d = query(database, user_id, last_7_days)

SERVING TIME:
  You compute features from a live cache
  avg_watch_time_7d = redis.get(f"user:{user_id}:watch_7d")

IF THESE DON'T MATCH EXACTLY:
  Your model sees different data in production than in training
  Performance degrades silently
  Very hard to debug
```

**This is why Feature Stores exist.** They enforce that the exact same computation runs at both training and serving time. We'll cover this next session.

---

## Feature Selection

Not all features are equally useful. Adding bad features can hurt model performance.

```
METHODS:

1. Correlation analysis
   Remove features correlated > 0.95 with another feature
   (Redundant information, adds noise)

2. Feature importance
   Tree-based models give you importance scores
   Remove features with near-zero importance

3. Ablation studies
   Remove one feature group at a time
   Measure impact on offline metrics
   The production-grade approach

4. Regularization
   L1 regularization drives weak feature weights to zero
   Model learns its own feature selection
```

---

## Common Pitfalls

**1. Feature leakage**
Using information at training time that wouldn't be available at serving time.

```
LEAKED: using "final watch time" to predict "will user watch?"
REASON: you don't know final watch time before the user watches
FIX:    only use features available at the moment of recommendation
```

**2. Data freshness mismatch**
Training on stale features while serving fresh features. Or vice versa.

**3. Null handling inconsistency**
Filling nulls with mean at training, with zero at serving. Silent but deadly.

**4. Scale mismatch**
Normalizing with training set statistics, but not applying same statistics at serving time.

---

## Summary

```
FIVE FEATURE TYPES:
  User → Item → Interaction → Context → Cross

TRANSFORMATIONS:
  Numerical  → normalize / log / bucketize
  Categorical → embed (high cardinality) or one-hot (low)
  Temporal   → cyclical encoding + recency decay

GOLDEN RULE:
  Training and serving must compute features identically
  Feature Store enforces this

BIGGEST TRAP:
  Feature leakage — using future information to predict the future
```

---

# Session 3: Feature Stores

## The Core Question

> **How do you make features available consistently, at scale, to both training pipelines and serving systems?**

---

## Why Feature Stores Exist

Imagine you have 50 ML models at a company. Each one needs user features.

Without a feature store:

```
Team A computes avg_watch_time their own way
Team B computes avg_watch_time a slightly different way
Team C uses a 3-month-old version

Result:
├── Duplicated engineering effort
├── Inconsistent feature definitions
├── Training-serving skew everywhere
└── No reusability across models
```

A feature store solves all of this by being a **single source of truth for features.**

---

## The Two Fundamental Needs

Every feature in an ML system has two access patterns:

```
TRAINING TIME (offline)
  "Give me user features for all users over the last 6 months"
  → High volume, historical, batch access
  → Latency doesn't matter (hours is fine)
  → Correctness and point-in-time accuracy matter enormously

SERVING TIME (online)
  "Give me user features for user_123 right now"
  → Low volume, current, real-time access
  → Latency matters enormously (< 10ms)
  → Freshness matters
```

These two access patterns require completely different storage systems.

---

## The Feature Store Architecture

```
                    FEATURE COMPUTATION
                   (Spark / Flink / dbt)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────┐       ┌──────────────────┐
    │   OFFLINE STORE │       │   ONLINE STORE   │
    │   (data lake)   │       │   (low latency)  │
    │                 │       │                  │
    │ • Hive / BQ     │       │ • Redis          │
    │ • Parquet       │       │ • DynamoDB       │
    │ • S3 / GCS      │       │ • Cassandra      │
    │                 │       │                  │
    │ Historical data │       │ Current snapshot │
    │ Point-in-time   │       │ Fast lookup      │
    └────────┬────────┘       └────────┬─────────┘
             │                         │
             ▼                         ▼
    ┌─────────────────┐       ┌──────────────────┐
    │ TRAINING JOBS   │       │  SERVING SYSTEM  │
    │                 │       │                  │
    │ Fetch historical│       │ Fetch current    │
    │ features for    │       │ features for     │
    │ labeled dataset │       │ online inference │
    └─────────────────┘       └──────────────────┘
```

---

## The Offline Store

**Purpose:** Historical feature storage for training.

**Requirements:**
- Store features with timestamps
- Support point-in-time correct lookups
- Handle petabyte scale

**Technologies:** BigQuery, Hive, S3 + Parquet

### Point-in-Time Correctness — Critical Concept

This is subtle and almost always overlooked.

```
PROBLEM:

You're training a model to predict if a user clicks a video.
Your training label: "user clicked video at timestamp T"
Your features: "user's watch history"

WRONG approach:
  Join user features as they exist TODAY
  User's history today includes videos watched AFTER time T
  You've leaked future information into training
  Model looks great offline, terrible in production

CORRECT approach:
  For each training example at time T,
  fetch user features as they existed AT time T
  This is point-in-time correct feature lookup
```

```
Timeline:
──────────────────────────────────────────────►
         T-30d    T-7d    T (label)    NOW
           │       │        │           │
           └───────┘        │           │
        Features to use     │      Features you
        for training at T   │      should NOT use
                            │      (future leakage)
```

This requires the offline store to store features with timestamps and support "give me the feature value as of time T" queries.

---

## The Online Store

**Purpose:** Real-time feature serving at low latency.

**Requirements:**
- Sub-10ms p99 latency
- High availability (99.99%)
- Current feature values (not historical)

**Technologies:** Redis, DynamoDB, Bigtable

### What lives in the online store?

Not all features need to be in the online store. Only features that:
1. Change frequently (need freshness)
2. Are needed at serving time
3. Are expensive to compute on the fly

```
IN ONLINE STORE:
  user_watch_history_7d        ← changes daily
  user_session_features        ← changes per session
  video_view_count_1h          ← changes hourly

NOT IN ONLINE STORE (compute on the fly):
  time_of_day                  ← trivial to compute
  days_since_account_created   ← static, compute from account data
```

---

## Feature Materialization

How do features get from computation into the stores?

```
BATCH MATERIALIZATION (most common)
  Spark job runs every hour/day
  Computes features for all users/items
  Writes to offline store
  Syncs latest values to online store

  Pros: Simple, reliable, cheap
  Cons: Features can be stale (up to 1 hour old)

STREAMING MATERIALIZATION
  Flink/Dataflow job processes events in real time
  Updates online store within seconds of event occurring
  Also writes to offline store for training

  Pros: Fresh features
  Cons: Complex, expensive, harder to backfill

ON-DEMAND COMPUTATION
  Features computed at request time
  Result optionally cached
  
  Pros: Always fresh, no precomputation
  Cons: Adds latency to serving path
```

**The tradeoff:**

```
BATCH          → simple but stale
STREAMING      → fresh but complex
ON-DEMAND      → freshest but slow
```

Most production systems use all three, choosing per feature based on how much freshness matters.

---

## Feature Registry

The feature store also needs a **registry** — a catalog of all features.

```
FEATURE: user_avg_watch_time_7d
  Description: Average watch time per video in last 7 days
  Owner: recommendations-team
  Entity: user_id
  Data type: float32
  Freshness SLA: 1 hour
  Computation: SELECT AVG(watch_time) FROM events 
               WHERE timestamp > NOW() - 7d
               GROUP BY user_id
  Offline store: gs://features/user/watch_time_7d/
  Online store: redis key: user:{id}:watch_7d
  Used by: [ranking_model_v2, explore_model_v1]
```

This registry enables:
- Feature discovery (teams reuse instead of recompute)
- Lineage tracking (which models depend on which features)
- Impact analysis (if I change this feature, what breaks?)

---

## Real Systems

**Google:** Internal feature store integrated with their ML platform (Vertex AI Feature Store publicly)

**Apple:** Privacy-constrained — many features are computed **on device** and never leave the phone. The feature store problem looks very different when you can't centralize user data.

**Uber Michelangelo:** One of the first public feature store architectures. Influenced most systems built after it.

**Feast:** Open-source feature store. Good reference architecture.

---

## Summary

```
FEATURE STORE = OFFLINE STORE + ONLINE STORE + REGISTRY + PIPELINE

OFFLINE STORE: Historical features, point-in-time correct, for training
ONLINE STORE:  Current features, low latency, for serving

THREE MATERIALIZATION PATTERNS:
  Batch     → simple, stale
  Streaming → fresh, complex  
  On-demand → freshest, slower

BIGGEST TRAP:
  Point-in-time incorrectness → feature leakage → model looks
  great offline, fails in production
```

---

# Session 4: Embeddings

## The Core Question

> **How do you represent complex objects — users, videos, words, products — as numbers in a way that captures meaning and similarity?**

---

## The Fundamental Problem

You have 800 million videos. You need to compare them.

One-hot encoding:
```
video_001 → [1, 0, 0, 0, ..., 0]  (800M dimensions)
video_002 → [0, 1, 0, 0, ..., 0]  (800M dimensions)
```

Problems:
- 800M dimensional vectors are unusable
- All videos are equally dissimilar (orthogonal vectors)
- No semantic meaning captured

What you want:
```
cooking_video_1 → [0.8, 0.2, -0.3, ...]
cooking_video_2 → [0.7, 0.3, -0.2, ...]  ← similar to cooking_video_1
gaming_video_1  → [-0.5, 0.8, 0.6, ...]  ← different from cooking
```

This is what embeddings give you: **dense, low-dimensional vectors where geometric proximity means semantic similarity.**

---

## Mental Model: Embedding Space

```
          EMBEDDING SPACE (2D simplified)

  +1.0 │                    • gaming
       │              • tech
  +0.5 │
       │
   0.0 │──────────────────────────────────
       │
  -0.5 │    • cooking
       │              • fitness
  -1.0 │  • food_review
       │
       └──────────────────────────────────
           -1.0    -0.5    0.0    +0.5   +1.0

  cooking and food_review are CLOSE → similar content
  gaming and cooking are FAR → different content
```

Real embeddings are 64 to 512 dimensions. The principle is the same.

---

## How Embeddings Are Learned

### Method 1: Word2Vec / Item2Vec (Shallow)

The key intuition comes from distributional semantics:

> **You shall know a word by the company it keeps.**
> — J.R. Firth, 1957

Applied to YouTube:

> **You shall know a video by the sessions it appears in.**

```
USER SESSION (sequence of videos watched):
[cooking_basics, knife_skills, pasta_recipe, italian_cuisine]

INSIGHT:
  pasta_recipe and knife_skills appear in similar sessions
  → they should have similar embeddings

TRAINING:
  Given a video, predict the videos that appear near it in sessions
  → learns which videos co-occur
  → co-occurring videos get similar vector representations
```

This is **Item2Vec** — the recommendation system analogue of Word2Vec. Simple, fast, surprisingly effective.

### Method 2: Matrix Factorization

```
OBSERVATION MATRIX R (users × items):
         vid1  vid2  vid3  vid4
user1  [  1     0     1     ?  ]
user2  [  0     1     0     1  ]
user3  [  1     1     ?     0  ]

FACTORIZE into:
R ≈ U × V^T

U = user embedding matrix  (users × k)
V = item embedding matrix  (items × k)

k = embedding dimension (e.g., 64)

The ? values are predicted by U[user] · V[item]
```

This is **collaborative filtering**. The embeddings encode latent user preferences and item characteristics without knowing what those characteristics actually are.

### Method 3: Deep Learning (Most Common Today)

```
INPUT: user_id, video_id, context features
         │
         ▼
    [Embedding Lookup Layers]
    user_id → 256-dim vector
    video_id → 256-dim vector
         │
         ▼
    [Neural Network Layers]
    concatenate + dense layers
         │
         ▼
    OUTPUT: probability of engagement
```

The embeddings are learned end-to-end as part of the model training. They encode whatever information is useful for the prediction task.

---

## Two-Tower Architecture

This is the dominant architecture for recommendation systems at scale.

```
     USER TOWER              ITEM TOWER
         │                       │
   [user features]          [item features]
   user_id embedding        video_id embedding
   watch history            title embedding
   demographics             category
         │                       │
   [dense layers]           [dense layers]
         │                       │
         ▼                       ▼
   user_embedding           item_embedding
      (256-dim)                (256-dim)
         │                       │
         └───────────┬───────────┘
                     │
               dot product
                     │
                     ▼
            similarity score
```

**Why two towers?**

Because at serving time you can:
1. Precompute all item embeddings offline
2. Compute user embedding at request time
3. Find nearest item embeddings via ANN search

This scales to billions of items because item embeddings are precomputed. You only need to compute one user embedding per request.

**This is the foundation of modern recommendation at Google, Meta, Netflix, Spotify.**

---

## Types of Embeddings in Production

```
TEXT EMBEDDINGS
  Word2Vec, GloVe → word-level, shallow
  BERT, Sentence-BERT → contextual, deep
  Use for: titles, descriptions, queries, transcripts

IMAGE / VIDEO EMBEDDINGS
  CNN features (ResNet, EfficientNet) → visual content
  Video transformers → temporal understanding
  Use for: thumbnail quality, content understanding

USER EMBEDDINGS
  Learned from interaction history
  Encode long-term preferences
  Updated periodically (daily or real-time)

ITEM EMBEDDINGS
  Learned from co-occurrence or content
  Encode item characteristics
  Updated when item metadata changes

QUERY EMBEDDINGS (for search)
  Encode search intent
  Must be compatible with item embedding space
  Enables semantic search beyond keyword matching
```

---

## Embedding Freshness

Embeddings go stale. This matters.

```
USER EMBEDDING FRESHNESS:
  New user     → no history → cold start problem
  Active user  → preferences shift → need updates
  
  Update strategies:
  ├── Daily batch recompute (most common)
  ├── Session-level updates (expensive)
  └── Streaming updates with approximation

ITEM EMBEDDING FRESHNESS:
  New video    → no interactions yet → cold start
  Trending     → popularity shifts fast
  
  Update strategies:
  ├── Use content embeddings for new items (no interaction needed)
  └── Blend content + interaction embeddings as data accumulates
```

---

## Cold Start Problem

New users and new items have no interaction history. You can't learn embeddings from nothing.

```
COLD START SOLUTIONS:

NEW USER:
  1. Use demographic features as proxy
  2. Ask onboarding questions (explicit preferences)
  3. Use popular/trending items until enough data
  4. Session-based recommendations (no user history needed)

NEW ITEM:
  1. Content-based embeddings (from title, thumbnail, description)
  2. Publisher embeddings (new video from popular creator)
  3. Similar item embeddings from content similarity
  4. Explore/exploit: show to some users to gather signal
```

This is one of the hardest problems in production recommendation systems.

---

## Embedding Storage and Serving

```
TRAINING:
  Embeddings stored as model weights
  Updated during training

SERVING:
  Item embeddings → precomputed, stored in vector database
  User embeddings → computed at request time OR cached

VECTOR DATABASE:
  Stores millions/billions of item embeddings
  Supports fast nearest-neighbor search
  Examples: Pinecone, Weaviate, Milvus, Vertex AI Matching Engine
  (We cover ANN search in Session 13)
```

---

## Summary

```
EMBEDDING = dense vector that encodes semantic meaning
  Proximity in embedding space = semantic similarity

THREE LEARNING METHODS:
  Item2Vec   → co-occurrence in sessions
  Matrix Factorization → collaborative filtering
  Deep Learning → end-to-end, most flexible

TWO-TOWER MODEL:
  User tower + Item tower → dot product similarity
  Enables billion-scale retrieval via precomputed item embeddings

COLD START:
  New users → demographics, onboarding, popularity
  New items → content embeddings, creator signals

FRESHNESS:
  User embeddings → daily or streaming updates
  Item embeddings → content-based for new, interaction-based as data grows
```

---

# Session 5: Offline Evaluation

## The Core Question

> **How do you know if your model is good before you deploy it to real users?**

---

## Why Offline Evaluation Is Hard

Online evaluation (A/B testing) is the ground truth. But you can't A/B test every model — it's slow, expensive, and risky.

Offline evaluation lets you filter bad models before they ever see users. But it has fundamental limitations that you must understand.

---

## The Dataset Split Problem

```
NAIVE SPLIT (WRONG for time-series data):

ALL DATA: ─────────────────────────────────
           random 80%           random 20%
           [TRAIN]              [TEST]

PROBLEM:
  Training data contains events AFTER test events
  Model sees the future during training
  Offline metrics look great, online metrics disappoint
```

```
CORRECT SPLIT (temporal):

ALL DATA: ──────────────────────────────────────►  time
           [──── TRAIN ────][─ VAL ─][─ TEST ─]

  Train on past → validate on near future → test on far future
  Mimics real deployment conditions
```

**Always split by time for recommendation and prediction systems.**

---

## Core Offline Metrics

### Classification Metrics

```
For binary outcomes (click / no click):

ACCURACY: fraction correct
  Problem: useless with class imbalance
  If 99% of labels are negative, predict all negative → 99% accuracy

AUC-ROC: area under ROC curve
  Measures ranking quality across all thresholds
  0.5 = random, 1.0 = perfect
  Most common metric for recommendation ranking

PR-AUC (Area Under Precision-Recall Curve)
  Better than AUC-ROC for highly imbalanced datasets
  Focuses on the minority class (positives)
  Use when false positives are very costly
```

### Ranking Metrics

These matter most for recommendation systems because you return a ranked list.

```
PRECISION@K:
  Of the top K items you recommended, how many were relevant?
  P@10 = (relevant items in top 10) / 10

RECALL@K:
  Of all relevant items, how many appeared in your top K?
  R@10 = (relevant items in top 10) / (total relevant items)

NDCG@K (Normalized Discounted Cumulative Gain):
  Measures ranking quality, position-aware
  Relevant items ranked higher are rewarded more

  DCG@K = Σ (relevance_i / log2(i+1)) for i=1 to K

  The log2 discount means:
    Position 1 contributes much more than position 10
    
  Normalized by ideal ranking (NDCG range: 0 to 1)

MRR (Mean Reciprocal Rank):
  For queries with one correct answer
  MRR = mean(1 / rank_of_first_correct_answer)
  Common in search, Q&A systems
```

**Why NDCG over Precision@K?**

A model that puts the best item at position 1 should score better than one that puts it at position 5, even if both have the same precision@5. NDCG captures this position sensitivity.

---

## The Metric-Reality Gap

This is the most important concept in offline evaluation.

```
OFFLINE METRIC       ≠       REAL-WORLD OUTCOME
(what you measure)           (what you care about)

High AUC-ROC         ≠       Users happier
High NDCG            ≠       More time spent on platform
Low loss             ≠       Better recommendations
```

**Why does this gap exist?**

**1. Popularity bias in test data**
Your test data was generated by your old model. It only contains items your old model showed users. Your new model might be better but recommend items users never had a chance to see.

**2. Implicit feedback noise**
Click ≠ satisfaction. A click on a misleading thumbnail doesn't mean the user liked the video.

**3. Position bias in labels**
Items shown in position 1 get clicked more regardless of quality. Your test labels are contaminated by position.

**4. Distribution shift**
Test data is from the past. Real deployment is in the future. User behavior changes.

---

## Addressing Evaluation Biases

### Inverse Propensity Scoring (IPS)

```
PROBLEM: Items shown at position 1 have higher click probability
         regardless of quality

SOLUTION: Reweight training examples by inverse of their exposure probability

          corrected_label = observed_click / P(item was shown)

This debiases the metric to reflect true user preference
rather than system-induced bias
```

### Counterfactual Evaluation

```
PROBLEM: We only know what happened with the items we showed.
         We don't know what would have happened with different items.

SOLUTION: Use logged exploration data (random policy)
          Evaluate new policy on that unbiased sample
          
          This is called off-policy evaluation
          Requires deliberate logging of random/exploratory traffic
```

---

## Offline Evaluation Pipeline

```
┌─────────────────────────────────────────────────────┐
│                 EVALUATION PIPELINE                  │
│                                                      │
│  Historical Data                                     │
│       │                                              │
│       ▼                                              │
│  [Temporal Split]                                    │
│  train / val / test                                  │
│       │                                              │
│       ▼                                              │
│  [Feature Generation]                                │
│  same feature code as production                     │
│       │                                              │
│       ▼                                              │
│  [Model Training]                                    │
│  train on train set                                  │
│       │                                              │
│       ▼                                              │
│  [Offline Metrics]                                   │
│  AUC, NDCG, P@K on test set                         │
│       │                                              │
│       ▼                                              │
│  [Slice Analysis]                                    │
│  metrics per user segment, country, device           │
│       │                                              │
│       ▼                                              │
│  [Regression Test]                                   │
│  must beat baseline model metrics                    │
│       │                                              │
│       ▼                                              │
│  PASS → Shadow Deploy → A/B Test                    │
└─────────────────────────────────────────────────────┘
```

---

## Slice Analysis — Critical for L5

Don't just look at overall metrics. Look at metrics per slice.

```
OVERALL AUC: 0.82  ← looks good

BY SLICE:
  New users (< 7 days):     AUC = 0.61  ← terrible
  Mobile users:             AUC = 0.84  ← good
  Non-English speakers:     AUC = 0.71  ← below average
  Low-activity users:       AUC = 0.65  ← bad

A model can look great overall while failing specific populations.
This is how you get biased, unfair, or unsafe models in production.
```

Slice analysis is also how you **find where to improve next.** The worst-performing slice is your next modeling opportunity.

---

## Backtesting

For time-sensitive systems (e.g., news recommendation, trending content):

```
BACKTESTING:
  Train on data up to time T
  Evaluate on data from T to T+∆
  Repeat for multiple T values (rolling window)
  
  Average metric across windows gives stable estimate
  Also reveals if model degrades over time (staleness)
```

---

## Summary

```
ALWAYS split by time, never randomly for temporal data

CORE METRICS:
  Classification: AUC-ROC, PR-AUC
  Ranking:        NDCG@K, P@K, Recall@K, MRR

OFFLINE ≠ ONLINE: the gap is real and caused by:
  Popularity bias, position bias, feedback noise, distribution shift

SLICE ANALYSIS: overall metrics hide failures in subpopulations

COUNTERFACTUAL EVAL: use exploration data for unbiased offline eval

OFFLINE EVALUATION IS A FILTER, NOT A GUARANTEE
The only ground truth is an A/B test
```

---

# Session 6: Candidate Generation & Retrieval

## The Core Question

> **You have 800 million videos and one user. You need 10 recommendations in under 100ms. How?**

You cannot score all 800 million videos. You need to narrow down to a manageable set first.

This is the **retrieval problem.**

---

## The Funnel Architecture

Every large-scale recommendation system uses this pattern:

```
800,000,000 items
      │
      │  RETRIEVAL / CANDIDATE GENERATION
      │  Fast, approximate, high recall
      ▼
   1,000 candidates
      │
      │  PRE-RANKING (optional)
      │  Light model, filters and coarsely ranks
      ▼
    200 candidates
      │
      │  RANKING
      │  Expensive model, accurate scoring
      ▼
     50 candidates
      │
      │  RE-RANKING / BUSINESS RULES
      │  Diversity, freshness, policy enforcement
      ▼
     10 final recommendations
```

Each stage trades **speed for accuracy**:
- Retrieval: very fast, approximate
- Ranking: slower, precise

**Why not use the ranking model on all items?**

A deep neural network scoring 800M items at 1ms each = 800,000 seconds. Completely infeasible. The funnel solves this.

---

## Retrieval Methods

### Method 1: Embedding-Based Retrieval (Dominant)

```
OFFLINE:
  Precompute embedding for every video → store in vector index

ONLINE (per request):
  1. Compute user embedding (256-dim vector)
  2. Find K nearest video embeddings
  3. Return those K videos as candidates

QUERY:  user_embedding = [0.3, -0.5, 0.8, ...]
SEARCH: "which of the 800M video embeddings are most similar?"
ANSWER: top-1000 nearest neighbors (by cosine similarity)
```

This requires fast **Approximate Nearest Neighbor (ANN) search**, which we cover in Session 13.

### Method 2: Collaborative Filtering Retrieval

```
"Users similar to you also watched..."

STEPS:
  1. Find users similar to target user (by embedding similarity)
  2. Collect items those users interacted with
  3. Return unseen items as candidates
```

### Method 3: Content-Based Retrieval

```
"Because you watched [cooking video], here are similar videos..."

STEPS:
  1. Take user's recent watch history
  2. Find videos with similar content embeddings
  3. Return as candidates
```

### Method 4: Rule-Based / Heuristic Retrieval

```
"Trending in your country"
"New from creators you follow"
"Your saved/watchlater list"
"Previously started but not finished"
```

Simple but important. Rules encode domain knowledge and handle edge cases that ML can miss.

---

## Multiple Retrieval Sources

Production systems don't use one retrieval method. They use many:

```
CANDIDATE SOURCES:
  ├── Embedding ANN retrieval          → 300 candidates
  ├── Collaborative filtering          → 200 candidates
  ├── Content-based (recent watches)   → 150 candidates
  ├── Trending in user's country       → 100 candidates
  ├── New from followed creators       → 100 candidates
  ├── Continue watching                →  50 candidates
  └── Saved items                      →  50 candidates
                                         ──────────────
  TOTAL (deduplicated)               →  ~750 candidates
```

**Why multiple sources?**

- No single source covers all user intents
- Diversity: embedding retrieval alone produces similar items
- Coverage: rule-based catches things ML misses (followed creators)
- Freshness: trending + new uploads from ANN alone

---

## Retrieval Quality Metrics

```
RECALL@K:
  Of all items the user would have engaged with,
  what fraction appear in your K candidates?
  
  If your ranker is perfect but retrieval recall is 0.3,
  you can only ever surface 30% of good items.
  Retrieval recall is the ceiling on overall system quality.

COVERAGE:
  What fraction of the item catalog appears in candidates
  across all users? Low coverage = popularity bias.

FRESHNESS:
  Are new items being retrieved for users who'd like them?
  Important for new creator discovery.
```

---

## The Exploration vs Exploitation Problem

```
EXPLOITATION: Recommend what we're confident the user will like
              → Safe, but creates filter bubbles
              → User's taste never expands
              → Over time, engagement drops as content gets repetitive

EXPLORATION:  Sometimes recommend something new/different
              → Risky short-term (might not engage)
              → Builds richer user profiles
              → Discovers new interests
              → Better long-term engagement

SOLUTION: ε-greedy or Thompson Sampling
  ε-greedy: with probability ε, retrieve random candidates
             with probability 1-ε, retrieve best candidates
  
  ε is tuned carefully — too high and UX suffers,
  too low and the system stagnates
```

---

## Session 7: Ranking

## The Core Question

> **Given 1,000 candidates, how do you order them to maximize the chance the user engages with the top results?**

---

## Why Ranking Is Harder Than Retrieval

Retrieval is about **recall** — did we find the right items?
Ranking is about **precision and order** — did we put the best items first?

Ranking models can be much more expensive because they only process ~1,000 items, not billions.

---

## What the Ranking Model Learns

The ranker takes a (user, item, context) tuple and produces a score.

```
INPUT:
  user_features         [demographics, history, preferences]
  item_features         [content, popularity, freshness]
  interaction_features  [user affinity for this creator/category]
  context_features      [time, device, session]
  cross features        [user × item interactions]

MODEL:
  Deep neural network (typically)

OUTPUT:
  P(click)    = 0.23
  P(watch>50%)= 0.18
  P(like)     = 0.04
  P(share)    = 0.01
```

Notice: **multiple outputs**, one per engagement type.

---

## Multi-Task Learning for Ranking

You almost never optimize a single objective in production.

```
SINGLE OBJECTIVE PROBLEM:
  Optimize P(click) → model learns clickbait
  Optimize P(watch_time) → model favors long videos
  Optimize P(like) → data too sparse

SOLUTION: Multi-Task Learning (MTL)

  SHARED LAYERS                    TASK-SPECIFIC HEADS
  (learn shared representation)    (learn task-specific patterns)

  [user features]                  ┌→ P(click)
  [item features]    → [shared] ───┼→ P(watch > 50%)
  [context features]  network      ├→ P(like)
                                   └→ P(share)

  All tasks are trained simultaneously
  Shared layers benefit from all signals
  Task heads specialize

FINAL SCORE:
  score = w1 × P(click) + w2 × P(watch>50%) + w3 × P(like) + ...
  
  Weights w1, w2, w3 are business decisions, not ML decisions
```

This is the architecture used by YouTube, TikTok, and most major platforms.

---

## Learning to Rank

Three paradigms:

```
POINTWISE:
  Treat each (user, item) pair independently
  Learn: is this item relevant? (binary classification)
  Loss: cross-entropy
  Pros: simple
  Cons: doesn't optimize ranking order directly

PAIRWISE:
  Learn: given item A and item B, which should rank higher?
  Loss: hinge loss or logistic loss on pairs
  Pros: directly optimizes relative ordering
  Cons: O(n²) pairs, noisy with implicit feedback

LISTWISE:
  Learn: given a list of items, what's the optimal ordering?
  Loss: approximates NDCG directly (LambdaRank, LambdaMART)
  Pros: directly optimizes what you care about
  Cons: more complex, harder to train at scale

PRODUCTION REALITY:
  Most large-scale systems use POINTWISE with careful metric design
  Pairwise used in some neural ranking models
  Listwise in search (LambdaMART is standard in Bing, Google)
```

---

## Position Bias in Ranking

```
PROBLEM:
  Items ranked position 1 get more clicks regardless of quality
  Your training data has position-biased labels
  Model learns position is good, not item quality

EVIDENCE:
  Same item moved from position 5 to position 1
  Gets 2-5x more clicks with zero change in content

SOLUTIONS:

1. Position feature injection (during training):
   Add position as a feature
   At serving time, set position = 0 for all items
   Model learns to separate position effect from item quality

2. Inverse Propensity Scoring:
   Downweight clicks from position 1 (over-represented)
   Upweight clicks from lower positions (under-represented)

3. Randomization experiments:
   Randomly shuffle results for a small % of traffic
   Collect unbiased click data for training
```

---

## Re-Ranking and Business Rules

After the main ranker, you apply final adjustments:

```
DIVERSITY:
  Don't show 10 cooking videos in a row
  Ensure variety of creators, categories, lengths

FRESHNESS BOOST:
  Slightly boost recently uploaded content
  Prevents the feed from stagnating on old content

POLICY ENFORCEMENT:
  Filter age-restricted content for minors
  Remove flagged/reviewed content
  Apply geographic content restrictions

FATIGUE PENALTY:
  Reduce score of items user has already seen
  Reduce score of items from same creator shown recently

BUSINESS RULES:
  Boost promoted content (ads insertion point)
  Ensure minimum threshold of local language content
```

These rules are often **not ML** — they're explicit business logic. This is intentional. Some constraints are too important to learn from data.

---

## Summary: The Full Recommendation Stack

```
USER REQUEST
     │
     ▼
[RETRIEVAL]
Multiple sources → ~1,000 candidates
Fast, approximate, high recall
Embedding ANN + rules + collab filtering
     │
     ▼
[PRE-RANKING] (optional)
Light model → ~200 candidates
Filter obvious bad matches cheaply
     │
     ▼
[RANKING]
Heavy model → scores all ~200 candidates
Multi-task: P(click), P(watch), P(like)
Corrects for position bias
     │
     ▼
[RE-RANKING]
Business rules, diversity, freshness
Policy enforcement
     │
     ▼
FINAL 10 RECOMMENDATIONS
```

---

# Session 8: Training Pipelines

## The Core Question

> **How do you go from raw data to a trained model reliably, repeatably, and at scale?**

---

## Why Training Pipelines Are Hard

Training a model in a notebook is easy. Building a production training pipeline is hard because:

```
NOTEBOOK:               PRODUCTION:
One run                 Thousands of runs
Your laptop             Cluster of machines
Manual steps            Fully automated
Hours of data           Months or years of data
You know what happened  Must be debuggable by anyone
Run once                Runs every day (retraining)
```

---

## Anatomy of a Training Pipeline

```
RAW DATA
  │
  ▼
[DATA VALIDATION]           ← Is the data sane?
  │
  ▼
[FEATURE GENERATION]        ← Same code as serving (critical)
  │
  ▼
[DATASET CONSTRUCTION]      ← Sample, split, balance
  │
  ▼
[TRAINING]                  ← Model fitting
  │
  ▼
[OFFLINE EVALUATION]        ← Is the model good enough?
  │
  ▼
[MODEL REGISTRATION]        ← Version, tag, store
  │
  ▼
[DEPLOYMENT GATE]           ← Human or automated approval
  │
  ▼
[SERVING]
```

Each step should be independently testable and rerunnable.

---

## Dataset Construction in Detail

```
1. SAMPLING
   You often can't use all data (too much)
   
   Negative sampling: clicks are rare (1% CTR)
   Without sampling: 99% of data is "no click"
   Model predicts "no click" always → 99% accuracy, useless
   
   Solution: downsample negatives to ~1:4 ratio (positive:negative)
   Correction: adjust model output probabilities post-hoc
   
2. TEMPORAL SPLIT
   As discussed in Session 5
   
3. LABEL CONSTRUCTION
   Multiple engagement signals → weighted combination
   Decide the label delay: 
     Immediate (click) vs delayed (did they finish watching?)
     Longer delay = better label, but slower retraining

4. HARD NEGATIVE MINING
   Easy negatives: random items user didn't engage with (too easy)
   Hard negatives: items user saw but didn't engage with (informative)
   
   Hard negatives make the model learn finer distinctions
   Critical for embedding models and two-tower architectures
```

---

## Pipeline Orchestration

```
WHAT IS ORCHESTRATION?
  Managing dependencies between pipeline steps
  "Run step B only after step A completes successfully"

TOOLS:
  Apache Airflow → most common open source
  Google Cloud Composer → managed Airflow
  Kubeflow Pipelines → Kubernetes-native ML pipelines
  Metaflow (Netflix) → Python-native ML pipelines
  Vertex AI Pipelines (Google)

PIPELINE DAG (Directed Acyclic Graph):

  data_validation
       │
       ▼
  feature_generation ──→ fails here? stop. alert.
       │
       ▼
  train_model
       │
  ┌────┴────┐
  ▼         ▼
eval_val  eval_test
  │         │
  └────┬────┘
       ▼
  [compare to baseline]
       │
       ▼
  register_model (if better)
```

---

## Experiment Tracking

Every training run should log:

```
WHAT TO LOG:
  ├── Code version (git commit hash)
  ├── Data version (which dataset, date range)
  ├── Hyperparameters (learning rate, batch size, architecture)
  ├── Training metrics (loss curve per epoch)
  ├── Evaluation metrics (AUC, NDCG on val and test)
  ├── Training time and compute cost
  ├── Model size and inference latency
  └── Feature importance / ablation results

TOOLS:
  MLflow (open source)
  Weights & Biases (W&B)
  Google Vertex AI Experiments
  Neptune.ai

PURPOSE:
  Reproducibility: anyone can recreate any past run
  Comparison: was this model actually better?
  Debugging: why did this run fail?
  Compliance: audit trail for regulated industries
```

---

## Data Versioning

Your model is only as good as the data it was trained on. You must version your data.

```
WHY:
  Bug in feature computation? Need to retrain from before the bug.
  Need to reproduce a specific model? Need exact same data.
  Regulatory audit? Must prove what data the model saw.

HOW:
  Immutable dataset snapshots in object storage (S3/GCS)
  Name datasets with timestamps and versions
  Track dataset → model mapping in model registry
  
  gs://ml-data/recommendations/training/v2024-01-15/
  gs://ml-data/recommendations/training/v2024-01-22/
  gs://ml-data/recommendations/training/v2024-01-29/
```

---

# Session 9: Distributed Training

## The Core Question

> **Your model has 100 billion parameters and your dataset has 1 trillion examples. A single machine can't handle this. How do you train across hundreds of machines?**

---

## Why Distribution Is Necessary

```
TYPICAL DEEP LEARNING SCALE:
  Dataset:         1TB - 1PB
  Model:           millions to billions of parameters
  Training time:   days to weeks on single GPU

SOLUTION: Distribute across many machines/GPUs

CHALLENGE: Machines must coordinate
  They share a model but see different data
  How do they agree on what the model weights should be?
```

---

## Data Parallelism

The most common form of distributed training.

```
IDEA:
  Split data across N machines
  Each machine has a FULL COPY of the model
  Each machine trains on its own data shard
  Machines synchronize gradients periodically

PROCESS:
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │
  │ Model    │  │ Model    │  │ Model    │  ← identical copies
  │ Data 1/3 │  │ Data 2/3 │  │ Data 3/3 │  ← different data
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
               [GRADIENT SYNC]
               Average gradients
               Update all models
                      │
                      ▼
              All workers now have
              identical updated model
```

**Synchronous vs Asynchronous:**

```
SYNCHRONOUS (All-Reduce):
  All workers compute gradients
  Wait for ALL workers to finish
  Average gradients, update model
  
  Pros: Deterministic, stable training
  Cons: Slowest worker bottlenecks everyone (straggler problem)

ASYNCHRONOUS (Parameter Server):
  Workers don't wait for each other
  Update parameters as gradients arrive
  
  Pros: No straggler bottleneck
  Cons: Gradient staleness, less stable, harder to tune
```

---

## Model Parallelism

When the model itself is too large for a single machine.

```
PIPELINE PARALLELISM:
  Split model layers across machines
  
  Machine 1: layers 1-10
  Machine 2: layers 11-20
  Machine 3: layers 21-30
  
  Data flows through machines like a pipeline
  
  CHALLENGE: Machines are idle while waiting for previous stage
  SOLUTION: Micro-batching — pipeline multiple micro-batches
            to keep all machines busy

TENSOR PARALLELISM:
  Split individual matrix operations across machines
  Each machine computes part of a matrix multiplication
  Results combined via all-reduce
  
  Used for very large layers (e.g., 10B parameter attention layers)
  Requires very high-bandwidth interconnects (NVLink, InfiniBand)
```

---

## Google's Approach: TPUs

```
TPU (Tensor Processing Unit):
  Custom ASIC designed specifically for matrix operations
  Organized in TPU Pods: 256-4096+ TPUs connected by high-speed fabric

ADVANTAGES:
  Much faster matrix multiply than GPU for transformers
  High bandwidth interconnect between TPUs
  Tight integration with GCS (data storage) and TF

TRAINING SETUP:
  Data: Google Cloud Storage
  Training: TPU Pod
  Orchestration: Borg (internal) → Kubernetes externally
  Framework: JAX or TensorFlow
  
  A TPU v4 Pod can train a 540B parameter model (PaLM)
  in weeks rather than years
```

---

## Gradient Checkpointing and Mixed Precision

```
MEMORY PROBLEM:
  Training requires storing activations for backprop
  Large models run out of GPU memory

GRADIENT CHECKPOINTING:
  Don't store all activations
  Recompute them during backward pass
  Trades compute for memory
  ~30% more compute, ~4x less memory

MIXED PRECISION TRAINING (FP16 + FP32):
  Store weights in FP32 (for numerical stability)
  Compute forward and backward pass in FP16 (2x faster, 2x less memory)
  Accumulate gradients in FP32
  
  Standard practice for all large model training today
  Libraries: PyTorch AMP (Automatic Mixed Precision)
```

---

# Session 10: Hyperparameter Tuning

## The Core Question

> **How do you efficiently find the best configuration for your model without burning your entire compute budget?**

---

## The Search Space Problem

```
TYPICAL HYPERPARAMETERS:
  learning_rate:    [1e-5, 1e-4, 1e-3, 1e-2]
  batch_size:       [256, 512, 1024, 2048]
  num_layers:       [2, 4, 6, 8]
  hidden_dim:       [128, 256, 512, 1024]
  dropout:          [0.0, 0.1, 0.2, 0.3]
  optimizer:        [Adam, AdamW, SGD]
  lr_schedule:      [cosine, linear, constant]

GRID SEARCH COMBINATIONS: 4×4×4×4×4×3×3 = 18,432 combinations
  At 2 hours per run = 36,864 hours = 4.2 years
  Obviously infeasible
```

---

## Tuning Strategies

```
1. GRID SEARCH
   Try all combinations
   Guarantees finding the best in the grid
   Only feasible for 1-2 hyperparameters

2. RANDOM SEARCH
   Sample random combinations
   Surprisingly effective
   Finds good solutions faster than grid search
   Why? Most parameters don't matter equally.
        Random search explores important dims better.

3. BAYESIAN OPTIMIZATION
   Model the relationship between hyperparameters and performance
   Use past results to intelligently select next trial
   
   Process:
   a. Run a few random trials
   b. Fit a surrogate model (Gaussian Process) to results
   c. Use acquisition function to pick most promising next trial
   d. Run trial, add to data, repeat
   
   Pros: Very sample-efficient (fewer trials needed)
   Cons: Sequential (can't fully parallelize)
   Tools: Optuna, Vizier (Google), HyperOpt

4. SUCCESSIVE HALVING / HYPERBAND
   Start many trials with small compute budget
   Kill the worst half
   Give survivors more compute
   Repeat until one winner
   
   Pros: Very efficient, parallelizable
   Cons: Assumes performance ranking is stable early
   
   This is the dominant approach at Google scale
```

---

## Learning Rate — The Most Important Hyperparameter

```
LEARNING RATE INTUITION:
  Too high → model diverges, loss explodes
  Too low  → model trains too slowly, gets stuck
  
  LR Schedule (how LR changes during training):
  
  WARMUP + COSINE DECAY:
  
  LR  │    ╭─╮
      │   ╱   ╲
      │  ╱     ╲
      │ ╱       ╲_______
      │╱
      └─────────────────► steps
       warmup  decay
  
  Warmup: start small, ramp up (avoids early instability)
  Cosine decay: smoothly reduce LR as training progresses
  
  This is standard practice for transformer training
```

---

Now let's move into serving infrastructure.

---

# Session 11: Online Serving & Real-Time Inference

## The Core Question

> **Your model is trained. How do you serve predictions to millions of users with low latency and high reliability?**

---

## The Serving Challenge

```
TRAINING:                     SERVING:
  Batch, offline                Request/response, online
  Throughput matters            Latency matters (< 100ms)
  Errors are recoverable        Errors affect real users
  Single machine OK             Must scale horizontally
  Python notebook OK            Production-grade system
```

---

## Online Serving Architecture

```
USER
  │  HTTP/gRPC request
  ▼
[LOAD BALANCER]
  │  routes to available server
  ▼
[API GATEWAY]
  │  auth, rate limiting, request validation
  ▼
[FEATURE FETCHING]
  │  parallel fetch from feature store
  ▼  online store (Redis) → < 5ms
[MODEL SERVER]
  │  TensorFlow Serving / TorchServe / Triton
  │  runs model inference
  ▼
[POST-PROCESSING]
  │  business rules, re-ranking, formatting
  ▼
[RESPONSE]
  returned to user
```

---

## Model Serving Systems

```
TENSORFLOW SERVING:
  Optimized for TF models
  gRPC + REST APIs
  Handles model versioning and hot-swapping
  Used heavily at Google

TORCHSERVE:
  PyTorch equivalent
  Handler-based architecture

NVIDIA TRITON:
  Framework-agnostic (TF, PyTorch, ONNX, TensorRT)
  GPU-optimized batching
  Standard in high-throughput serving
  
ONNX RUNTIME:
  Convert models to ONNX format
  Run optimized inference across hardware
  Good for cross-platform portability

VLLM / TGI (for LLMs):
  Optimized for transformer generation
  Continuous batching for high throughput
```

---

## Latency Breakdown

```
END-TO-END LATENCY BUDGET: 100ms total

  Network (client to server):    ~20ms
  Load balancer + routing:        ~2ms
  Auth + rate limiting:           ~3ms
  Feature fetching (Redis):       ~5ms
  Model inference:               ~30ms
  Post-processing:                ~5ms
  Response serialization:         ~3ms
  Network (server to client):    ~20ms
                                 ──────
  TOTAL:                         ~88ms

BOTTLENECKS TO WATCH:
  Feature fetching: if Redis is cold or slow
  Model inference: if model is too large
  Sequential dependencies: A must finish before B
```

---

## Batching Strategies

```
WHY BATCH?
  GPUs are designed for parallel computation
  Single inference wastes GPU capacity
  Batching amortizes fixed costs (memory transfers, kernel launch)

STATIC BATCHING:
  Wait for N requests, process together
  Problem: requests don't arrive uniformly
           waiting introduces latency

DYNAMIC BATCHING (Triton, TF Serving):
  Accumulate requests for up to X microseconds
  Process whatever arrived in that window
  Balances latency vs throughput dynamically

CONTINUOUS BATCHING (LLMs):
  For generative models, requests finish at different times
  Insert new requests into the batch as others complete
  Keeps GPU utilization high
  This is how vLLM achieves high throughput
```

---

## Hardware for Serving

```
CPU SERVING:
  Pros: cheap, simple, no GPU management
  Cons: slow for large models
  Good for: small models, < 10ms inference

GPU SERVING:
  Pros: fast for neural networks, high throughput
  Cons: expensive, complex, memory-constrained
  Good for: large models, embedding computation

INFERENCE-OPTIMIZED CHIPS:
  Google TPU for serving (Cloud TPU Inference)
  AWS Inferentia
  Apple Neural Engine (on-device)
  Lower cost than GPU for inference-only workloads
```

---

# Session 12: Batch and Streaming Inference

## Batch Inference

> **How do you generate predictions for millions of users/items without waiting for a request?**

```
USE CASES:
  Precompute recommendations for all users overnight
  Generate embeddings for all catalog items
  Score all (user, item) pairs for newsletter personalization
  Run model on historical data for analysis

ARCHITECTURE:
  Data in GCS/S3
       │
       ▼
  Spark / Beam job
  distributes data across workers
       │
       ▼
  Each worker runs model inference
  on its data shard
       │
       ▼
  Results written back to GCS/BigQuery
  or synced to online store (Redis)

ADVANTAGES:
  Amortize compute cost
  No latency pressure during serving (precomputed)
  Can use larger, more expensive models

DISADVANTAGES:
  Predictions are stale (computed hours ago)
  Wasted compute if user never visits
  Cold start for new users between batch runs
```

## Streaming Inference

> **How do you generate predictions in near real-time as events occur?**

```
USE CASES:
  Update user embedding as they watch videos (within minutes)
  Fraud detection on transactions (within seconds)
  Adjust recommendations based on current session

ARCHITECTURE:
  USER EVENT
       │
       ▼
  [Kafka / Pub-Sub]
       │
       ▼
  [Stream Processor - Flink/Dataflow]
  Aggregates events, computes features
       │
       ▼
  [Model Inference] (lightweight model)
       │
       ▼
  [Online Store - Redis]
  Updated prediction/features available immediately

TRADEOFFS VS BATCH:
  Fresher predictions
  More complex infrastructure
  Higher cost
  Harder to debug
```

---

# Session 13: Vector Search & Approximate Nearest Neighbors

## The Core Question

> **You have 800 million video embeddings (each 256-dimensional). A user walks in. How do you find the 1000 most similar videos to their embedding in under 10ms?**

---

## Why Exact Search Doesn't Scale

```
EXACT NEAREST NEIGHBOR:
  Compare user vector to every item vector
  800M × 256 dimensions × float32
  = 800M dot products per query
  
  At 1 billion operations per second:
  800M operations = 0.8 seconds per query
  
  WAY too slow. Need approximation.
```

---

## ANN Algorithms

### HNSW (Hierarchical Navigable Small World)

The most widely used ANN algorithm today.

```
INTUITION:
  Build a multi-layer graph where:
  - Top layers have few nodes, long-range connections (highways)
  - Bottom layers have all nodes, short-range connections (streets)
  
  SEARCH:
  1. Enter at top layer, find approximate nearest neighbor
  2. Drop down a layer, use previous result as starting point
  3. Refine search with denser connections
  4. Repeat until bottom layer
  
  LAYER 2 (sparse):    A ─────────── E
                       │             │
  LAYER 1 (medium):    A ── B ── C ─ E
                       │    │    │   │
  LAYER 0 (dense):     A─B──B─C─C─D─E
  
  Query "find nearest to Q":
  Start at top: A or E? → E closer
  Drop to layer 1: E, D, C? → C closer
  Drop to layer 0: C, B? → B closest
  
PROPERTIES:
  Build time: O(n log n)
  Query time: O(log n)
  Memory: high (stores graph structure)
  Accuracy: very high (tunable recall)
```

### IVF (Inverted File Index)

```
INTUITION:
  Cluster all vectors into K clusters (e.g., K=10,000)
  At query time:
  1. Find which clusters the query is closest to
  2. Search only within those clusters (nprobe clusters)
  
  Reduces search space from 800M to ~80K vectors
  
PROPERTIES:
  Build time: moderate (clustering)
  Query time: fast
  Memory: low (no graph)
  Accuracy: depends on nprobe
  nprobe↑ → accuracy↑ but latency↑
```

### Product Quantization (PQ) — Compression

```
PROBLEM: 800M × 256-dim × 4 bytes = 800GB. Doesn't fit in RAM.

SOLUTION: Compress vectors using Product Quantization
  Split 256-dim vector into 32 sub-vectors of 8-dim each
  For each sub-vector, store which of 256 cluster centers it's closest to
  8-bit index instead of 8 × 4 = 32 bytes
  
  Compression ratio: 32× smaller
  800GB → 25GB. Fits in RAM.
  
  Accuracy tradeoff: slight approximation in distance computation
  Usually combined with IVF (IVF-PQ)
```

---

## Production ANN Systems

```
FAISS (Facebook AI Similarity Search):
  Open source, GPU-accelerated
  Supports IVF, HNSW, PQ and combinations
  Battle-tested at Meta scale
  The foundation of most vector DBs

DEDICATED VECTOR DATABASES:
  Pinecone      → managed, easy to use
  Weaviate      → open source, full-featured
  Milvus        → open source, high scale
  Qdrant        → open source, Rust-based
  
GOOGLE'S SOLUTION:
  Vertex AI Matching Engine (ScaNN internally)
  ScaNN: Google's own ANN library, very fast
  Supports streaming updates to the index

KEY METRICS:
  Recall@K: what fraction of true K-NN does ANN return?
  QPS: queries per second
  Latency: p50, p99
  Memory footprint: must fit in RAM for fast serving
```

---

# Session 14: Caching

## The Core Question

> **How do you reduce latency and cost by avoiding redundant computation?**

---

## What to Cache

```
FEATURE CACHE:
  Cache user features in Redis
  TTL = 1 hour (balance freshness vs cost)
  Hit rate typically 80-90%
  Eliminates feature store roundtrip for active users

EMBEDDING CACHE:
  Cache user embedding (256-dim vector)
  Reuse across requests in same session
  Embed recompute only when session changes

RESULT CACHE:
  Cache final ranked list for user
  TTL = 5-10 minutes
  Aggressive caching: same user same context → same result
  
  RISK: Stale results if item availability changes
        (e.g., item goes out of stock, content removed)

CANDIDATE CACHE:
  Cache retrieval results (1000 candidates)
  Ranking is cheap relative to retrieval
  Can rerank fresh but use cached candidates
```

---

## Cache Invalidation

The hardest problem in caching.

```
STRATEGIES:

TTL (Time-to-Live):
  Cache expires after N seconds
  Simple, predictable
  May serve stale data until TTL expires

EVENT-BASED INVALIDATION:
  When user watches a video → invalidate their recommendation cache
  When video is removed → invalidate any cached results containing it
  
  More complex, but fresher data

WRITE-THROUGH:
  Update cache and database simultaneously on writes
  Cache never stale
  Higher write latency

LAZY INVALIDATION:
  Serve stale, rebuild asynchronously
  Great for non-critical staleness (e.g., 5-min old recommendations)
```

---

# Session 15: Model Registry & Deployment

## Model Registry

> **How do you track, version, and manage models in production?**

```
WHAT A MODEL REGISTRY STORES:
  ├── Model artifact (weights, architecture)
  ├── Model version (v1.0, v1.1, v2.0)
  ├── Training metadata (commit, dataset, hyperparameters)
  ├── Evaluation metrics (AUC, NDCG, latency)
  ├── Deployment status (staging, canary, production)
  ├── Lineage (which data, which features, which code)
  └── Tags (approved, deprecated, experimental)

TOOLS:
  MLflow Model Registry (open source)
  Vertex AI Model Registry (Google)
  SageMaker Model Registry (AWS)
  W&B Model Registry

WORKFLOW:
  Train → Evaluate → Register → Review → Approve → Deploy
  
  Only approved models get deployed
  Every deployment is traceable to a specific registered version
```

---

# Session 16: Deployment Patterns

## The Core Question

> **How do you ship a new model to production without breaking things?**

---

## Canary Deployment

```
IDEA:
  Don't give new model to all users at once
  Start with a small percentage, monitor, then ramp up

PROCESS:
  Week 1: 1% of traffic → new model
          99% of traffic → old model
          
  Monitor metrics. If no issues:
  
  Week 2: 10% → new model
  Week 3: 50% → new model
  Week 4: 100% → new model

METRICS TO WATCH:
  Latency (p50, p99)
  Error rate
  Business metrics (CTR, watch time)
  System metrics (CPU, memory)

AUTOMATIC ROLLBACK:
  If error rate > threshold → automatically revert to old model
  No human intervention required for obvious failures
```

---

## Shadow Deployment

```
IDEA:
  Run new model in parallel with old model
  New model doesn't affect users
  Compare outputs to validate behavior

ARCHITECTURE:
  REQUEST
     │
     ├──────────────────────────────┐
     ▼                              ▼
[OLD MODEL]                    [NEW MODEL] (shadow)
produces response              produces response
serves to user                 discarded (not shown)
     │                              │
     ▼                              ▼
[RESPONSE TO USER]            [LOGGED FOR COMPARISON]
                                    │
                                    ▼
                              [ANALYSIS]
                              Do outputs match?
                              Are there regressions?

USE CASES:
  Validating a major model architecture change
  Testing a new model on real traffic patterns
  Catching edge cases before canary
  
COST: Doubles inference compute during shadow period
```

---

## Blue/Green Deployment

```
BLUE (current production)     GREEN (new version)
  ────────────────              ─────────────────
  Serving 100% traffic          Idle, ready
  
  SWITCH:
  Load balancer flips 100% traffic from blue to green
  Instant cutover
  
  ROLLBACK:
  Flip back to blue immediately if issues
  Blue remains live throughout
  
  PROS: Instant cutover, instant rollback
  CONS: Requires 2x infrastructure cost during transition
```

---

## Rollback

```
TRIGGERS FOR ROLLBACK:
  Error rate > threshold
  Latency p99 > SLA
  Business metrics drop > X%
  Manual trigger by on-call engineer
  
ROLLBACK TYPES:
  Model rollback: revert to previous model version
  Feature rollback: revert to previous feature computation
  Full service rollback: revert entire service version

ROLLBACK REQUIREMENTS:
  Old model still registered and available
  Feature computation of old model still works
  Can complete rollback in < 5 minutes
  Ideally automated, no manual steps
```

---

# Session 17: Monitoring & Drift Detection

## The Core Question

> **Your model is deployed. How do you know it's still working correctly tomorrow? Next month? Next year?**

---

## Why Models Degrade

```
CONCEPT DRIFT:
  The relationship between features and labels changes
  User behavior changes (COVID changed everything)
  World events affect what content is popular
  
DATA DRIFT:
  The distribution of input features changes
  New user demographics
  New device types
  Seasonal changes in usage patterns

MODEL STALENESS:
  Model trained on past data
  World has moved on
  Model's "mental model" of the world is outdated

UPSTREAM DATA CHANGES:
  Someone changes a data pipeline
  Feature computation changes silently
  Model sees different features than it was trained on
```

---

## What to Monitor

```
LAYER 1: INFRASTRUCTURE METRICS (always)
  ├── Request volume (QPS)
  ├── Latency (p50, p99, p999)
  ├── Error rate
  ├── CPU / GPU utilization
  └── Memory usage

LAYER 2: INPUT DATA METRICS
  ├── Feature distribution (mean, std, percentiles)
  ├── Missing value rates
  ├── Null/invalid value rates
  └── Category distribution shifts

LAYER 3: PREDICTION METRICS
  ├── Prediction distribution (mean score, percentiles)
  ├── Score variance (model becoming overconfident?)
  └── Prediction volume per class

LAYER 4: BUSINESS METRICS (ground truth, delayed)
  ├── Click-through rate
  ├── Watch time
  ├── User satisfaction scores
  └── Conversion rates
```

---

## Drift Detection Methods

```
STATISTICAL TESTS:

Population Stability Index (PSI):
  Measures how much a distribution has shifted
  PSI < 0.1: no significant change
  PSI 0.1-0.2: moderate change, investigate
  PSI > 0.2: significant change, take action

Kolmogorov-Smirnov Test:
  Compares two distributions
  Tests if they come from same distribution
  Good for continuous features

Chi-Square Test:
  For categorical feature distributions
  Tests if observed vs expected distribution differ

WINDOW-BASED MONITORING:
  Compare current week vs baseline week
  Alert if mean shifts by > 2 standard deviations
  Simple but effective

LEARNED DRIFT DETECTION:
  Train a classifier to distinguish
  "training distribution" vs "current distribution"
  If classifier achieves high AUC, distributions are different
  This is called the "two-sample test" approach
```

---

## Alerting Strategy

```
ALERT FATIGUE IS REAL:
  Too many alerts → on-call engineers ignore them
  Too few alerts → problems go undetected

TIERED ALERTING:
  SEV 1 (page immediately):
    Error rate > 5%
    Latency p99 > 2x SLA
    Service down
    
  SEV 2 (alert in business hours):
    Business metrics drop > 10%
    Significant data drift detected
    Model performance regression
    
  SEV 3 (create ticket):
    Gradual drift trend
    Minor metric degradation
    New device/feature not in training distribution

RUNBOOKS:
  Every alert should have a runbook
  "When X alert fires, do Y steps"
  On-call shouldn't have to improvise
```

---

# Session 18: Data Validation

## The Core Question

> **How do you catch data problems before they corrupt your model or serving system?**

---

## TensorFlow Data Validation (TFDV) Approach

```
SCHEMA DEFINITION:
  After training, generate schema from training data:
  
  feature: user_age
    type: INT
    min_value: 13
    max_value: 120
    missing_rate: 0.02
    
  feature: video_category
    type: STRING
    allowed_values: [cooking, tech, gaming, ...]
    missing_rate: 0.0

VALIDATION AT PIPELINE TIME:
  When new training data arrives:
    → Validate against schema
    → Alert on anomalies
    → Block pipeline if critical
    
  When serving data arrives:
    → Real-time validation per request
    → Return fallback if feature invalid
```

---

## What to Validate

```
SCHEMA VALIDATION:
  Feature exists in request
  Feature has correct type
  Feature value in expected range
  
DISTRIBUTION VALIDATION:
  Compare new data distribution to reference (training)
  Use PSI, KS test, or mean/std comparison
  Alert on significant shifts
  
CARDINALITY VALIDATION:
  Categorical feature has unexpected new values?
    new country code, new device type
  → May need model retraining or special handling

COMPLETENESS VALIDATION:
  Missing rate increased?
  Some upstream service stopped sending a feature?
  Downstream model will see nulls instead of values
  → Silent degradation
  
PIPELINE VALIDATION:
  Does feature computation produce same result as training?
  Run shadow comparison periodically
```

---

# Session 19: Retraining Strategies

## The Core Question

> **How do you keep your model fresh as the world changes?**

---

## Retraining Triggers

```
SCHEDULED RETRAINING:
  Retrain every N days regardless of performance
  Simple, predictable, easy to operate
  Risk: retraining when not needed (wastes compute)
        not retraining when needed (if schedule too slow)

METRIC-TRIGGERED RETRAINING:
  Monitor business metrics and drift metrics
  If metric drops below threshold → trigger retraining
  More efficient but requires good monitoring

EVENT-TRIGGERED RETRAINING:
  Major world event (holiday, news, product launch)
  Sudden change in user behavior
  Data pipeline change
  → Manual trigger by ML team
```

---

## Full Retrain vs Fine-tuning

```
FULL RETRAIN:
  Start from scratch on all available data
  Learns everything fresh
  Most expensive
  Use for: major distribution shifts, architecture changes

FINE-TUNING:
  Start from existing model weights
  Train on recent data only
  Much cheaper
  Use for: gradual drift, moderate updates
  Risk: catastrophic forgetting (model forgets old patterns)

CONTINUAL LEARNING:
  Never fully retrain
  Continuously update model on stream of new data
  Maintains memory of old data via various techniques
  (Elastic Weight Consolidation, experience replay)
  Very complex, used in specialized systems
```

---

## The Retraining Pipeline

```
AUTOMATED RETRAINING SYSTEM:
  
  TRIGGER (scheduled or metric-based)
       │
       ▼
  [DATA FRESHNESS CHECK]
  Is new data available?
  Is data volume sufficient?
       │
       ▼
  [FULL TRAINING PIPELINE]
  (as in Session 8)
       │
       ▼
  [COMPARISON TO CHAMPION MODEL]
  Must beat current production model
  on all key metrics
       │
       ▼
  [AUTO-PROMOTION or HUMAN REVIEW]
  Small improvements → auto-promote
  Large changes → human review
       │
       ▼
  [CANARY DEPLOY]
  (as in Session 16)
```

---

# Session 20: Reliability, Cost, Privacy & Security

## Reliability

```
SLA vs SLO vs SLI:

SLI (Service Level Indicator):
  A specific metric you measure
  "p99 latency" or "error rate"

SLO (Service Level Objective):
  Your internal target for an SLI
  "p99 latency < 100ms"
  "error rate < 0.1%"

SLA (Service Level Agreement):
  External commitment to customers
  Consequence if violated (refunds, etc.)
  Usually looser than SLO (buffer)

ERROR BUDGET:
  If SLO is 99.9% availability:
  Error budget = 0.1% = 8.7 hours/year
  Team can use error budget for risky changes
  When budget exhausted: freeze deployments until recovered

RELIABILITY PATTERNS:
  Circuit breakers: stop calling failing dependency
  Fallbacks: if model fails, return popularity-based recs
  Timeouts: don't wait forever for slow responses
  Retries with backoff: handle transient failures
  Bulkheads: isolate failures to one component
```

---

## Cost Optimization

```
TRAINING COSTS:
  Use spot/preemptible instances (60-90% cheaper)
  Checkpoint frequently for preemption recovery
  Efficient data loading (avoid I/O bottleneck)
  Use mixed precision (FP16) to halve compute
  
SERVING COSTS:
  Right-size instances (don't over-provision)
  Autoscaling: scale down during low traffic
  Model distillation: train small model to mimic large model
  Quantization: INT8 instead of FP32 (4x smaller, faster)
  Batch inference for non-real-time use cases
  Caching: avoid recomputing for same user/context
  
STORAGE COSTS:
  Lifecycle policies: archive old data to cold storage
  Compression: Parquet + Snappy for feature data
  Deduplication: don't store same feature computed multiple ways
```

---

## Privacy & Security

### Privacy

```
DATA MINIMIZATION:
  Only collect what you need
  Aggregate where possible, don't store individual events
  Delete data after retention period

DIFFERENTIAL PRIVACY:
  Add calibrated noise to training gradients
  Guarantees individual user's data can't be inferred
  Used by Apple extensively
  Cost: slight accuracy reduction
  DP-SGD: differentially private stochastic gradient descent

FEDERATED LEARNING:
  Don't centralize data at all
  Model trains ON the user's device
  Only model updates (gradients) sent to server
  Gradients don't reveal individual data
  Apple uses this for keyboard autocomplete, Siri suggestions
  
  CHALLENGES:
    Devices are heterogeneous (different compute, different data)
    Network unreliable (phones go offline)
    Aggregation must be secure (secure aggregation protocols)

k-ANONYMITY:
  Any individual's data is indistinguishable from k-1 others
  Prevents re-identification
```

---

## Apple vs Google — Final Comparison

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│ DIMENSION          │ GOOGLE               │ APPLE                │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Data location      │ Centralized          │ On-device            │
│ Training           │ TPU Pods, datacenter │ Federated learning   │
│ Model size         │ Very large (billions)│ Small (MBs)          │
│ Privacy approach   │ Anonymization, DP    │ On-device, FL, DP    │
│ Inference          │ Cloud serving        │ Core ML, Neural Eng. │
│ Scale              │ Billions of users    │ Billions of devices  │
│ Latency            │ 50-200ms cloud       │ <10ms on-device      │
│ Freshness          │ Near-real-time       │ Limited by sync      │
│ Infra              │ Borg/K8s/GCS/BigQuery│ On-device + iCloud   │
│ Optimization       │ Throughput + recall  │ Battery + memory     │
└────────────────────┴──────────────────────┴──────────────────────┘
```

---

# You Now Have the Full Foundation

Here's what we've covered:

```
✓ Session 1:  Data Collection & Signals
✓ Session 2:  Feature Engineering
✓ Session 3:  Feature Stores
✓ Session 4:  Embeddings
✓ Session 5:  Offline Evaluation
✓ Session 6:  Candidate Generation & Retrieval
✓ Session 7:  Ranking
✓ Session 8:  Training Pipelines
✓ Session 9:  Distributed Training
✓ Session 10: Hyperparameter Tuning
✓ Session 11: Online Serving
✓ Session 12: Batch & Streaming Inference
✓ Session 13: Vector Search & ANN
✓ Session 14: Caching
✓ Session 15: Model Registry & Deployment
✓ Session 16: Canary, Shadow, Rollback
✓ Session 17: Monitoring & Drift Detection
✓ Session 18: Data Validation
✓ Session 19: Retraining Strategies
✓ Session 20: Reliability, Cost, Privacy
```

---
