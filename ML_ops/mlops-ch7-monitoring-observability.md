# Chapter 7 — Monitoring & Observability

*(Module 7 of the syllabus)*

---

## 1. Why "is the server up" isn't enough for ML

In normal software monitoring, you mostly care about: is the service running, is it responding, is latency acceptable, is the error rate low. All of that still matters for an ML service — but remember Chapter 1's core insight: **an ML model can be fully "up," responding fast, throwing zero errors, and still be silently giving wrong answers.** A model that's degrading doesn't crash. It just gets worse, quietly, while every normal infrastructure metric looks perfectly healthy.

This is why ML monitoring needs an extra layer beyond standard infrastructure monitoring: you have to watch the *content* of what's flowing through the system — the inputs, the outputs, and their relationship to reality — not just whether the system is technically alive.

---

## 2. What to actually monitor — four layers

### Layer 1: Infrastructure health (the standard stuff)
Is the service up, what's the error rate, is latency within budget. Standard DevOps monitoring — necessary but, per above, not sufficient on its own for ML.

### Layer 2: Latency percentiles specifically
Worth calling out on its own because averages are misleading for latency. **p50** (median — half of requests are faster than this) tells you the typical experience. **p95** and **p99** (95th/99th percentile — only 5% or 1% of requests are slower than this) tell you about your worst-case tail. A service can have a great average latency while a meaningful fraction of users experience terrible slowness — which is exactly what p95/p99 catch and averages hide. If asked "how would you monitor latency," naming p95/p99 specifically (not just "average latency") is a strong, easy signal of production experience.

### Layer 3: Feature/input distribution
Are the *inputs* the model is currently receiving statistically similar to what it was trained on? This is your early-warning layer — a shift here often precedes a visible drop in output quality, and catching it here means you can investigate before business metrics actually suffer. This is also your direct connection back to Chapter 4 — this is literally how you'd detect training-serving skew in an ongoing way, not just as a one-time debugging exercise.

### Layer 4: Business/output metrics
Is the model actually still achieving its real-world purpose — conversion rate, fraud caught, click-through, whatever the model exists to move. This is the layer that ultimately matters most, but it's also often the *slowest* to get a signal from (you may need real outcomes — did the transaction turn out fraudulent — which can take days to resolve). This is why Layer 3 (input distribution) matters as an earlier warning sign than waiting on Layer 4 alone.

**The mental model to state explicitly in an interview:** these four layers form a chain, roughly in order of how fast you get a signal and how directly it reflects real business impact. Infrastructure metrics are fastest but least informative about model quality; business metrics are slowest but most directly meaningful. A mature monitoring setup watches all four, using the faster layers as early warnings for the slower ones.

---

## 3. Data drift vs. concept drift — the distinction interviewers love to probe

This is the precise version of what Chapter 4 grouped loosely under "the world changes over time." Now let's separate it properly, because interviewers specifically test whether you can tell these apart.

### Data drift
**Definition:** the distribution of the model's **input features** changes over time, while the underlying relationship between inputs and the correct output stays the same.

**Example:** an e-commerce recommendation model was trained when most traffic came from desktop browsers; now most traffic comes from mobile. The *inputs* look different (different behavioral patterns, different session lengths) — but "what makes a good recommendation given someone's browsing pattern" hasn't fundamentally changed. The world generating the inputs shifted; the true underlying pattern the model is trying to learn did not.

### Concept drift
**Definition:** the actual relationship between the inputs and the correct output changes over time — the same input now genuinely warrants a *different* answer than it used to.

**Example:** a fraud detection model was trained on historical fraud patterns. Fraudsters adapt their tactics specifically to evade detection — so a transaction pattern that used to reliably indicate "not fraud" might now genuinely indicate fraud, because fraudsters have deliberately started mimicking legitimate-looking behavior. The *relationship itself* — what a given pattern actually means — has changed, not just the frequency of different input patterns.

### Why the distinction matters practically
- **Data drift** can sometimes be handled by ensuring your training data is representative of current input patterns — you may not need the model's *learned relationships* to change, just make sure it's seeing (and was trained on) realistic current inputs.
- **Concept drift** is more serious: no amount of "more of the same kind of data" fixes it, because the ground truth itself has changed. This requires retraining on *recent* labeled data that reflects the new reality — and if drift is happening continuously (like adversarial fraud), you may need meaningfully more frequent retraining, or even online learning (Chapter 10), not just occasional batch updates.

**Quick way to keep these straight:** data drift = the questions being asked changed. Concept drift = the correct answers to the same question changed.

---

## 4. How drift actually gets detected (conceptually)

You don't need to derive the statistics from scratch, but you should be able to describe the general approach:

- **Statistical distance between distributions** — compare the distribution of a feature (or of model outputs) in a recent time window against the distribution seen during training, using some measure of how different two distributions are. A large distance triggers investigation.
- **Population Stability Index (PSI)** — a specific, commonly-cited metric for exactly this purpose: it buckets a feature's values into ranges and measures how much the proportion of data falling into each bucket has shifted between the training population and the current population. Worth knowing by name even without deriving the formula — it's a common "have you heard of this" check in interviews.

The practical setup: for every important feature (and often for the model's output distribution too), compute this kind of distance metric on a rolling basis (e.g., daily), and alert when it crosses a pre-defined threshold.

---

## 5. Alerting design

Detecting drift or a metric drop is only useful if it reliably reaches a human (or an automated response) in time to matter. Key design questions to be ready to answer:

- **What threshold triggers an alert?** Too sensitive → alert fatigue, people start ignoring alerts (a very real, very common failure mode). Too loose → real problems go unnoticed for too long.
- **Who gets paged, and how urgently?** Not every anomaly deserves to wake someone up at 3am — severity should scale the response (a Slack notification for something worth reviewing tomorrow vs. an urgent page for something actively harming users right now).
- **What's the escalation path if the first responder doesn't act?** Production monitoring generally shouldn't have a single point of failure in the *human* response chain either.

A strong interview answer connects this back to Chapter 5's rollback principle: ideally, some alerts don't just notify a human — they trigger an *automatic* response (like an automatic canary rollback), precisely because human response time is slower and less consistent than a pre-built automated trigger.

---

## 6. Logging predictions — and the privacy angle

For debugging, auditing (Chapter 9), and future retraining (Chapter 10), production systems typically log predictions: what input came in, what the model output, sometimes what actually happened afterward (ground truth, once known).

**Worth flagging unprompted in an interview:** if the inputs contain personal or sensitive user data, logging *raw* predictions and inputs indefinitely can create real privacy/compliance exposure. A thoughtful answer mentions considerations like: retention limits (don't keep logs forever), anonymization/pseudonymization where feasible, and access controls on who can query prediction logs — rather than treating "just log everything" as a free, consequence-free default. This is a good place to demonstrate that you think about production systems holistically, not just the ML mechanics.

---

## 7. Common pitfall interviewers listen for

Don't answer "how do you monitor a model in production" with only infrastructure metrics (uptime, latency, error rate). That answer would be correct for *any* backend service and demonstrates no ML-specific understanding. The layer that actually distinguishes ML monitoring — and the one to lead with — is watching feature/input distributions and output/business metrics for drift, precisely because (as established at the top of this chapter) an ML service can look perfectly healthy on every standard metric while quietly making worse and worse predictions.

---

## Comprehension check

1. In your own words: why can a model be "fully up" with zero errors and normal latency, yet still be failing at its actual job? Which monitoring layer catches that, and why don't the others?
2. Give your own example (different from the ones above) of data drift, and a separate example of concept drift. Be explicit about which one is which and why.
3. Explain why "alert on every anomaly, no matter how small" is actually a bad monitoring design, despite sounding maximally safe.

Say "c8" when ready for **Chapter 8: Scaling & Infrastructure**.
