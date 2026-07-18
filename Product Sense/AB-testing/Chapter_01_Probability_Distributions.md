
## 1. Definition

A probability distribution describes how likely different outcomes of a random variable are. In product analytics, every metric you look at — conversion rate, session count, time-to-purchase — is a realization of some underlying random process, and the distribution tells you what "normal" variation looks like before you ever run a test.

The four you need cold for A/B testing work:

- **Binomial(n, p)**: number of successes in *n* independent yes/no trials, each with success probability *p*.
- **Poisson(λ)**: number of events in a fixed interval of time/space, when events happen independently at a constant average rate λ.
- **Normal(μ, σ²)**: continuous, symmetric, bell-shaped distribution defined by mean and variance; the limiting distribution for sums/averages (via CLT).
- **Exponential(λ)**: time between independent events in a Poisson process; models "time until next event."

## 2. Layman Explanation

Think of these as four different "shapes" of randomness, each matching a different kind of question:

- **Binomial** = "Out of 1,000 users who saw the button, how many clicked it?" Each user is a coin flip (click or not), and you're counting heads.
- **Poisson** = "How many customer support tickets come in per hour?" You're not counting flips of a coin — you're counting rare events over time, with no fixed "number of trials."
- **Normal** = "What's the distribution of average order value across all our stores?" Once you average or sum enough things, the bell curve shows up almost no matter what the underlying data looked like.
- **Exponential** = "How long until the next user churns?" It's the flip side of Poisson — instead of counting events, you're measuring the gap between them.

The intuition to hold onto: **Binomial counts binary outcomes, Poisson counts rare events over time, Normal describes averages/sums, Exponential describes waiting times.**

## 3. Formal Explanation

**Binomial(n, p)**
- PMF: P(X = k) = C(n,k) pᵏ(1-p)ⁿ⁻ᵏ
- Mean: np
- Variance: np(1-p)
- Use case: conversion rate experiments — X = number of converters out of n users.

**Poisson(λ)**
- PMF: P(X = k) = (λᵏ e⁻λ) / k!
- Mean = Variance = λ
- Derived as the limit of Binomial(n, p) as n → ∞, p → 0, np → λ (rare events, many trials).
- Use case: modeling counts — page views per session, error rates per day.

**Normal(μ, σ²)**
- PDF: f(x) = (1 / (σ√(2π))) · e^(-(x-μ)²/(2σ²))
- Fully described by mean and variance; symmetric, unbounded.
- Why it matters for A/B testing: the Central Limit Theorem says the *sampling distribution of a sample mean* is approximately Normal regardless of the underlying metric's distribution, provided n is large enough. This is what licenses using t-tests/z-tests on metrics that aren't themselves Normal.

**Exponential(λ)**
- PDF: f(x) = λe^(-λx), x ≥ 0
- Mean = 1/λ, Variance = 1/λ²
- Memoryless property: P(X > s+t | X > s) = P(X > t) — the process "forgets" how long it's already waited. Important trap: many people assume churn/retention is memoryless, but real user behavior usually isn't (hazard rates change over tenure).

## 4. Levers — What Controls It, What Moves It

**Binomial**
- *n* (sample size) shrinks variance of the estimated proportion (np(1-p)/n²  → p(1-p)/n for the sample proportion) — more users = tighter estimate.
- *p* itself affects variance: variance is maximized at p = 0.5, and shrinks as p approaches 0 or 1. This is why very rare or very common conversion events are "easier" to estimate precisely in relative terms but harder in absolute terms (you need more samples to see rare events at all).

**Poisson**
- λ is both the mean and the variance — so as event rate increases, variance increases proportionally. This matters for metrics like "errors per session": a feature that increases baseline error rate will also inflate variance, making it harder to detect real signal without adjusting for this.
- Aggregating over more time/space (e.g., week instead of day) increases λ, changing your effective power.

**Normal**
- μ shifts the whole distribution; σ controls spread/precision.
- CLT convergence speed depends on the skewness of the underlying distribution — highly skewed metrics (e.g., revenue with big whales) need larger n before the Normal approximation for the sample mean is trustworthy. This is a classic reason average revenue per user (ARPU) tests need bigger samples or transformations (log, winsorizing, CUPED) than click-through-rate tests.

**Exponential**
- λ (rate) is the only lever — inverse of mean waiting time.
- If real-world data shows time-varying hazard (e.g., churn risk that changes with tenure), Exponential no longer fits — you'd need Weibull or a Cox proportional hazards model instead. Recognizing when memorylessness *fails* is itself an interview signal.

## 5. Famous Q&A (Google / Apple style)

**Q: Why can we use a t-test on conversion rate data when individual user outcomes are Bernoulli (0/1), not Normal?**
A: Because we're not testing individual observations — we're testing the *sample mean* (the conversion rate), and by the Central Limit Theorem, the sampling distribution of a mean approaches Normal as n grows, regardless of the underlying distribution's shape. With typical A/B test sample sizes (thousands of users), this approximation holds well unless p is extremely close to 0 or 1, in which case you'd want a normal approximation check (np ≥ 5 and n(1-p) ≥ 5) or an exact test.

**Q: Your error-count metric (errors per session) has variance much larger than its mean. Is Poisson still the right model?**
A: Not necessarily — this is overdispersion, a common trap. Poisson forces mean = variance. Real event-count data often shows variance > mean because event rates vary across users (some users are just more error-prone). The fix is typically a Negative Binomial distribution, which adds a dispersion parameter. Flagging this distinction is a strong L5 signal — it shows you check model assumptions rather than defaulting to the textbook distribution.

**Q: We want to detect a 2% relative lift in a metric with a very low base rate (0.1% conversion). What's the challenge?**
A: Variance for a Binomial proportion is p(1-p)/n, which is maximized around p=0.5 and small near p=0.1%. But because the base rate itself is tiny, the *absolute* signal you're trying to detect (0.1% × 2% = 0.002 percentage points) is also tiny relative to the noise floor — you need a much larger sample size to hit sufficient power. This is why rare-event experiments (e.g., purchase conversion in a low-intent flow) often need either much longer run times, a proxy metric with a higher base rate, or variance reduction techniques like CUPED.

**Q: A stakeholder asks why average session duration "looks Normal in the histogram of daily averages" but the raw per-session data is heavily right-skewed. Explain the discrepancy.**
A: This is the Central Limit Theorem in action. The raw per-session durations are skewed (most sessions are short, a few are very long), but once you average across many sessions per day, that average becomes approximately Normally distributed — the skew washes out as n grows. The individual data being non-Normal doesn't prevent the *aggregate* (mean) from being Normal; this is exactly why we can build valid confidence intervals on daily average metrics even when raw user-level data looks nothing like a bell curve.

---
Yes. The key is distinguishing **relative lift** from **absolute lift**.

### What is the base rate?

The **base rate** is the current value of the metric **before** any experiment. For a conversion metric, it's the percentage of users who convert under the existing experience.

If the base rate is **0.1%**, it means:

* Out of **100 users**, about **0.1** convert on average (not very intuitive).
* Out of **1,000 users**, about **1** converts.
* Out of **100,000 users**, about **100** convert.

So:

* Number of users = 100,000
* Conversions = 100
* Conversion rate = **100 / 100,000 = 0.001 = 0.1%**

---

### What does a 2% relative lift mean?

A **2% relative lift** means increasing the conversion rate by **2% of its current value**, **not** by 2 percentage points.

Current conversion rate:

[
0.1% = 0.001
]

A 2% relative increase is:

[
0.001 \times 1.02 = 0.00102
]

Converting back to percentages:

* Before: **0.100%**
* After: **0.102%**

Absolute increase:

[
0.102% - 0.100% = 0.002%
]

or **0.002 percentage points**.

---

### Example with 1,000,000 users

Suppose:

|                 |   Control | Treatment |
| --------------- | --------: | --------: |
| Users           | 1,000,000 | 1,000,000 |
| Conversion rate |    0.100% |    0.102% |
| Conversions     |     1,000 |     1,020 |

The treatment generated **20 extra conversions** out of **1 million users**.

That's a **2% relative lift**:

[
\frac{1020-1000}{1000}=0.02=2%
]

Even though the improvement is real, it's only **20 additional conversions**, which is a very small signal.

---

### Why does this require a huge sample?

Each user either:

* converts (1), or
* doesn't convert (0).

Most users (99.9%) don't convert.

The improvement you're trying to detect is:

* from **1000 conversions**
* to **1020 conversions**

Because the difference is only **20 conversions**, random variation can easily hide it unless you have a very large number of users.

---

### Is a 2% lift possible?

Absolutely. Many mature products target lifts of:

* **0.5%**
* **1%**
* **2%**
* **5%**

A 2% lift is often considered valuable at scale.

For example, suppose an e-commerce site gets **100 million visitors/year** with a **0.1%** purchase rate:

* Current purchases:
  [
  100,000,000 \times 0.1% = 100,000
  ]

* After a **2% relative lift**:
  [
  100,000 \times 1.02 = 102,000
  ]

That's **2,000 extra purchases per year**. If each purchase is worth $100, that's **$200,000 in additional revenue** from what appears to be a tiny improvement.

---

### Relative lift vs. percentage-point lift

These are commonly confused:

| Statement                       | New conversion rate (starting from 0.1%) |
| ------------------------------- | ---------------------------------------: |
| **2% relative lift**            |                               **0.102%** |
| **2 percentage-point increase** |                                 **2.1%** |

A **2 percentage-point increase** (from 0.1% to 2.1%) would be an enormous change—about a **2,000% relative lift**—and is extremely unlikely in most real-world A/B tests. That's why experiment discussions almost always specify **relative lift** rather than percentage-point changes.
