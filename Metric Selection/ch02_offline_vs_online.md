# Chapter 2: Offline vs. Online Evaluation

> *"A model that looks great on your test set and fails in production is not a good model. It's a model that passed the wrong exam."*

---

## 2.1 The Core Distinction

Every ML system lives a double life:

- **Offline**: You control the data, the labels, the environment. Fast, cheap, reproducible.
- **Online**: Real users, real traffic, real consequences. Slow, expensive, ground truth.

Neither is sufficient alone. Your offline metrics tell you what *could* work. Your online metrics tell you what *does* work.

```
Offline evaluation  →  "Does this model look good on historical data?"
Online evaluation   →  "Does this model improve the product in the real world?"
```

The gap between these two is where most production ML surprises live.

---

## 2.2 Offline Evaluation

### The Train / Validation / Test Split

The foundational setup:

```
Full Dataset
├── Training set      (~70%)   — model learns from this
├── Validation set    (~15%)   — hyperparameter tuning, model selection
└── Test set          (~15%)   — final, held-out, one-time evaluation
```

**Hard rules:**
- The test set is used **exactly once**. The moment you tune on it, it becomes a validation set.
- If you evaluate on test 10 times to pick the best model, your test error is optimistically biased.
- Splits must respect **time** for time-series data (no future leaking into past).

### Cross-Validation

When data is scarce, k-fold CV gives a more reliable estimate:

```
K-Fold (k=5):

Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]
Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]

Final score = mean ± std across folds
```

**When NOT to use standard k-fold:**
- Time-series data → use **TimeSeriesSplit** (always train on past, validate on future)
- Grouped data → use **GroupKFold** (keep user/patient groups intact across folds)
- Highly imbalanced → use **StratifiedKFold** (preserve class ratios per fold)

### Offline Evaluation Failure Modes

| Failure | What Went Wrong |
|---|---|
| **Data leakage** | Future information in training features |
| **Label leakage** | Target signal embedded in a feature |
| **Distribution mismatch** | Test set doesn't reflect production traffic |
| **Temporal leakage** | Random split on time-series data |
| **Duplicate contamination** | Same sample in train and test |

> **Leakage is the silent killer.** A model with 99% offline accuracy that drops to 60% in production almost always has a leakage problem. Always ask: *at inference time, would this feature actually be available?*

---

## 2.3 The Offline-Online Gap

Even a perfectly clean offline evaluation can diverge from online performance. Here's why:

### Reasons the Gap Exists

**1. Distribution shift**
The data you trained and tested on is historical. Production traffic is the future. User behavior, language, and patterns shift over time.

**2. Position bias and feedback loops**
In recommender systems, users only interact with what they're shown. Your logged data reflects the *previous model's decisions*, not user preferences in the abstract. A better model might never get credit in offline eval because its recommendations were never logged.

**3. Missing counterfactuals**
Offline data tells you what happened. It doesn't tell you what *would have happened* if the model had made a different decision. Online evaluation actually runs the counterfactual.

**4. System effects**
Latency, caching, UI changes, and downstream system behavior don't exist in offline eval. They exist in production.

**5. Novelty and primacy effects**
Users sometimes engage more with any change simply because it's new. Online A/B tests need to run long enough for this to wash out.

---

## 2.4 Online Evaluation Methods

### A/B Testing (Randomized Controlled Experiment)

The gold standard for online evaluation.

```
Incoming traffic
       │
       ▼
  [Randomize users]
       │
   ┌───┴───┐
   ▼       ▼
Control   Treatment
(Model A) (Model B)
   │       │
   └───┬───┘
       ▼
  Measure metric
  Run significance test
  Make decision
```

**Key design decisions:**

| Decision | Guidance |
|---|---|
| Unit of randomization | User-level (not request-level) to avoid carryover effects |
| Traffic split | Start 90/10 or 95/5; go 50/50 once confident in safety |
| Run duration | At least 1–2 full weekly cycles to capture weekly patterns |
| Primary metric | Pre-register one metric before launch; avoid p-hacking |
| Guardrail metrics | Metrics that must not regress (latency, error rate, revenue) |

**When A/B testing is hard:**
- Small user base (underpowered tests)
- Long feedback loops (retention takes months)
- Social/network effects (contamination between control and treatment)
- Cold-start (new users have no history)

### Interleaving

Used heavily in search and recommendation. Instead of splitting users, you interleave results from two rankers and see which items users prefer.

```
Model A ranking:  [Item1_A, Item2_A, Item3_A, ...]
Model B ranking:  [Item1_B, Item2_B, Item3_B, ...]

Interleaved:      [Item1_A, Item1_B, Item2_A, Item2_B, ...]
                   (deduplicated, order balanced)

Winner = model whose items got more clicks
```

**Advantage:** Much more statistically efficient than A/B — detects differences with far less traffic. Used by Netflix, Google, Spotify.

### Bandit-Based Evaluation (Exploration-Exploitation)

Rather than a fixed split, dynamically route more traffic to the better-performing model.

```
ε-greedy:   With probability ε → explore (random model)
            With probability 1-ε → exploit (best known model)

UCB:        Route to model with highest upper confidence bound
            on expected reward

Thompson:   Sample from posterior of each model's reward
            distribution, route to the sample winner
```

**Use when:** You can't afford to run a fixed A/B experiment and must learn while minimizing regret (e.g., real-time bidding, personalization).

---

## 2.5 Shadow Mode

Before you run a real A/B test, run your new model in **shadow mode**:

```
Production request
        │
        ├──► Model A (Production) ──► User sees this response
        │
        └──► Model B (Shadow)     ──► Response logged but NOT shown to user
```

**What you learn:**
- Does the model run without errors in production?
- How does latency compare?
- How different are the outputs from the current model?
- Are there edge cases you didn't catch offline?

Shadow mode is **zero-risk** — users are never affected. It's your safety check before committing to a real experiment.

---

## 2.6 Canary Deployments

A canary release sends a small slice of real traffic to the new model before a full rollout.

```
Rollout stages:

Day 1:   1% of traffic  → monitor for errors, latency, crashes
Day 3:   5% of traffic  → monitor business metrics
Day 7:   20% of traffic → run significance check
Day 14:  50% of traffic → A/B conclusion
Day 21:  100% rollout   or  rollback
```

**Canary vs. A/B test:**
- Canary is about **safety** (catching disasters early)
- A/B test is about **measurement** (detecting metric differences)
- You can do both simultaneously — canary is a subset of A/B traffic

**Automatic rollback triggers** (define these before launch):
- Error rate increases by > X%
- p99 latency exceeds threshold
- Guardrail metric drops by > Y%

---

## 2.7 Choosing the Right Evaluation Strategy

```
Is your change safe to show to users?
    No  → Shadow mode first
    Yes ↓

Is your user base large enough for an A/B test?
    No  → Offline eval + careful canary rollout
    Yes ↓

Is feedback available quickly (< 1 week)?
    No  → Consider surrogate metrics, longer experiment window
    Yes ↓

Do you have network/social effects?
    Yes → Cluster randomization, holdout cells
    No  → Standard A/B test
```

---

## 2.8 The Evaluation Ladder

Think of evaluation as a ladder you climb before deploying:

```
Level 5: Long-term online metrics (retention, LTV)        ← weeks/months
Level 4: Short-term online metrics (CTR, engagement)      ← days
Level 3: A/B test / canary                                ← days
Level 2: Shadow mode                                      ← hours
Level 1: Offline test set evaluation                      ← minutes
Level 0: Validation set / cross-validation                ← seconds
```

You move up the ladder as confidence grows. You never skip levels on high-stakes systems. Each level is more expensive but more trustworthy.

---

## 2.9 Logging for Online Evaluation

Online evaluation is only as good as your logging infrastructure. You need:

| Log Event | Why |
|---|---|
| Request ID + timestamp | Join events across systems |
| Model version + feature hash | Know exactly which model served the request |
| Raw features at inference time | Debug distribution shift later |
| Model output (score/ranking) | Reproduce decisions |
| User action (click, purchase, skip) | The label for online eval |
| Delay between request and action | Account for delayed feedback |

> **Don't log just outputs. Log the full decision context.** Six months from now you'll want to know exactly what the model saw when it made that prediction.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Offline eval | Fast, controlled, but can diverge from production |
| Train/val/test | Test set is sacred — use it once |
| Leakage | The #1 cause of optimistic offline metrics |
| Offline-online gap | Caused by shift, bias, feedback loops, system effects |
| A/B testing | Gold standard; pre-register metrics, respect run duration |
| Interleaving | More efficient than A/B for ranking problems |
| Shadow mode | Zero-risk production check before experiments |
| Canary | Safety-first rollout with automatic rollback triggers |
| Evaluation ladder | Climb levels; never skip on high-stakes systems |

---

## Further Reading

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments* (the A/B testing bible)
- Chapelle et al. — *Offline Evaluation of Contextual-Bandit-Based News Article Recommendation*
- Schölkopf et al. — *Toward Causal Representation Learning* (on counterfactuals in eval)

---

*Next: Chapter 3 — Ranking Metrics (NDCG, MAP, MRR)*
