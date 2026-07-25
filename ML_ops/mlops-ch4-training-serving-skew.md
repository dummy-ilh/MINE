# Chapter 4 — Training-Serving Skew

*(Module 4 of the syllabus — a favorite interview topic)*

---

## 1. Set up the scenario first

You've done everything right so far: clean versioned data (Ch2), a properly containerized, well-served model (Ch3). You evaluate the model offline and it looks great — 94% accuracy on held-out test data. You deploy it. And in production... it performs noticeably worse. Not broken, not crashing — just quietly *worse* than what you measured.

This is one of the most common, most frustrating, and most interview-tested failure modes in ML systems, and it has a name: **training-serving skew**.

## 2. Definition

**Training-serving skew** is any difference between the data/features the model saw during training and the data/features it actually sees at serving (prediction) time. The model was trained on one "shape" of data and is now, unknowingly, being asked to make decisions on a different "shape" of data.

The word "skew" is key — it's not that the production data is *invalid*, it's that it's *different in some way* from what the model learned from, and that difference degrades performance.

---

## 3. The three classic causes — know these cold

### Cause 1: Different code paths computing the same feature, offline vs. online

This is the single most common real-world cause, and it's subtle. Here's the setup:

- **Training time**: your data scientists compute features using an offline pipeline — maybe a batch SQL query or a Spark job that scans historical data.
- **Serving time**: a completely *different* piece of code, written by a different team (often the backend/serving team), computes "the same" feature in real time, on live data.

Even if both pieces of code are *intended* to compute "average purchase amount over the last 30 days," subtle implementation differences creep in: maybe the offline job uses calendar days and the online code uses rolling 30×24 hours; maybe one includes the current, in-progress day and the other doesn't; maybe there's a rounding difference. Individually tiny — but the model was trained on the offline version's exact numbers, and now it's being fed the online version's slightly different numbers. The model's learned weights don't know how to compensate for that, because it never saw that version of the feature.

**Why this happens so often in practice**: training pipelines and serving pipelines are frequently built by different teams, at different times, sometimes in different programming languages entirely (a Python/Spark offline job vs. a low-latency Java/C++ online service). Nobody *intends* skew — it emerges from the two pipelines drifting apart over time as each is maintained independently.

### Cause 2: Data distribution shift over time

Even with identical feature computation code, the *real world* changes. User behavior, market conditions, product features — all evolve. A model trained on last year's data implicitly assumes this year looks statistically similar. When it doesn't, the model's learned patterns become stale. This is technically a slightly different phenomenon from Cause 1 — it's not a *bug*, it's just time passing — but it produces the same symptom (production data looks different from training data) and gets grouped under the same skew umbrella in most interview contexts. (We'll separate "data drift" vs. "concept drift" precisely in Chapter 7 on monitoring.)

### Cause 3: Time-travel leakage (subtle, and a great one to bring up unprompted)

This is the sneakiest cause, and mentioning it unprompted is a strong signal in an interview. It happens when your *offline training pipeline* accidentally uses information that would not actually have been available at the time of the real prediction.

Concrete example: you're training a model to predict "will this user churn in the next 30 days," and one of your features is "total lifetime purchases." If you compute that feature using the user's *entire* history up to *today* (including purchases made *after* the point you're trying to predict from), you've leaked future information into training. The model learns to rely on a feature that, at real serving time, simply cannot exist yet — because the future hasn't happened. Offline, this can even make your evaluation metrics look artificially *better* than they'll ever be in production, which makes it especially dangerous: you ship a model that looks great on paper and immediately underperforms in the real world, without an obvious explanation.

---

## 4. How feature stores address Cause 1 (the big one)

This is where the feature store concept from the syllabus fully clicks into place. A **feature store** is a centralized system where a feature is defined and computed **once**, in a single piece of logic, and both the training pipeline and the serving pipeline pull from that *same* definition — rather than each team writing their own separate implementation.

Concretely, a feature store typically has two matched components:
- An **offline store** — historical feature values, used to assemble training datasets
- An **online store** — the latest feature values, kept fresh, used to serve real-time predictions

The critical property: both stores are populated by the *same underlying feature definition/transformation logic*, just materialized in two different ways (one for bulk historical access, one for fast point-in-time lookup). This directly eliminates Cause 1 — there's no longer a second, independently-written implementation to drift apart from the first.

Feature stores also directly prevent Cause 3 (time-travel leakage), through a mechanism called **point-in-time correctness**: when assembling a training example for "user X at date Y," the feature store only returns feature values *as they actually existed at date Y* — never anything computed using data from after that date. This is a specific, named capability worth citing if asked how you'd prevent leakage systematically, rather than just "being careful."

---

## 5. How to detect skew — the interview-ready debugging process

If an interviewer says "your model performs great offline but poorly in production — walk me through debugging this," here is the structured process to describe:

1. **Compare feature distributions, offline vs. online.** For every feature, log its actual value at serving time and compare the distribution (mean, spread, min/max, missing-value rate) against the distribution seen in the training data. A feature whose live distribution looks nothing like its training distribution is your prime suspect.
2. **Trace the computation path for suspect features.** For any feature showing a mismatch, go find the *two* pieces of code that compute it — the offline version and the online version — and diff their logic directly. This is where Cause 1 bugs get found.
3. **Check for leakage in the offline pipeline.** Review whether any feature used in training could only have existed *after* the prediction point in the real timeline. This is where Cause 3 gets found.
4. **Check the calendar.** If features and code both check out, consider Cause 2 — has enough time passed since training that the world itself has shifted? (This pushes you toward Chapter 10 — retraining.)

Notice the ordering: check code/implementation mismatches (Cause 1, Cause 3) *before* assuming "the world changed" (Cause 2). Cause 1/3 are bugs you can fix immediately; Cause 2 requires a retraining cycle. A good engineer rules out the fixable bug first.

---

## 6. Common pitfall interviewers listen for

Don't describe skew as just "the model is out of date." That description only covers Cause 2. If you *only* mention "the data changed over time," you're missing the two causes (mismatched feature code, and leakage) that are actually more common in real production incidents and are far more clearly a system-design/engineering failure — which is exactly the kind of thing an MLE is expected to catch.

---

## Worked example (to test your own understanding against)

*A fraud detection model performs at 92% precision in offline evaluation. In production, precision drops to 78% within the first week, then stays roughly flat — it doesn't keep degrading further over time.*

Think about this before reading on: does the "stays roughly flat, doesn't keep degrading" detail point you more toward Cause 1/3 (implementation mismatch) or Cause 2 (gradual world drift)? *(Gradual drift would typically show progressive degradation over time; an immediate drop that then plateaus points strongly toward a fixed implementation mismatch between the offline and online feature computation — Cause 1 — rather than the world slowly changing.)*

---

## Comprehension check

1. In your own words, explain the difference between Cause 1 and Cause 2 — one is a bug, one basically isn't. Why does that distinction matter for how you'd respond to each?
2. What does "point-in-time correctness" mean, and which of the three causes does it directly prevent?
3. Describe your own version of the 4-step debugging process, in the order you'd actually check things, and briefly say why that order makes sense.

Say "c5" when ready for **Chapter 5: Deployment Strategies** (shadow, canary, blue-green, A/B testing).
