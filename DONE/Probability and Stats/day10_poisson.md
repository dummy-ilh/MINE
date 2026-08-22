# Day 10 — Poisson Distribution & Poisson Process
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 4 (Section 4.6)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Poisson Matters in ML and Data Science

The Poisson distribution is the natural model whenever you're counting **rare events in a fixed interval**.

| Scenario | Poisson Model |
|---|---|
| Server requests per second | X ~ Poisson(λ) where λ = avg requests/sec |
| Errors in a log file per hour | X ~ Poisson(λ) |
| Rare word occurrences in a document | X ~ Poisson(λ) |
| Bugs found per 1000 lines of code | X ~ Poisson(λ) |
| Customer arrivals per minute | X ~ Poisson(λ) |
| Anomalies detected per day | X ~ Poisson(λ) |
| Mutations per genome segment | X ~ Poisson(λ) |

In NLP, the **Poisson language model** is a baseline. In systems ML, **Poisson processes** model request arrivals to APIs. In anomaly detection, departures from Poisson are the signal.

---

## 2. Poisson Distribution — Definition

> **Definition:** X ~ Poisson(λ) if:
> ```
>            e^(−λ) · λᵏ
> P(X = k) = ————————————     for k = 0, 1, 2, 3, ...
>                 k!
> ```
> where λ > 0 is the **rate parameter** (average number of events).

### Verify it's a valid PMF

```
Σₖ₌₀^∞ e^(−λ)λᵏ/k! = e^(−λ) · Σₖ₌₀^∞ λᵏ/k! = e^(−λ) · eˡ = 1  ✓
```

(Using the Taylor series: eˡ = Σ λᵏ/k!)

### Parameters

```
E[X]   = λ
Var(X) = λ
SD(X)  = √λ
```

**The remarkable fact:** Mean = Variance = λ. This is unique to Poisson.

**Proof of E[X]:**
```
E[X] = Σₖ₌₀^∞ k · e^(−λ)λᵏ/k!
     = Σₖ₌₁^∞ k · e^(−λ)λᵏ/k!        [k=0 term vanishes]
     = Σₖ₌₁^∞ e^(−λ)λᵏ/(k−1)!        [cancel k with k!]
     = λ · Σⱼ₌₀^∞ e^(−λ)λʲ/j!        [let j = k−1]
     = λ · 1 = λ  ∎
```

**Proof of Var(X):**
```
E[X(X−1)] = Σₖ₌₂^∞ k(k−1) · e^(−λ)λᵏ/k! = λ² · Σⱼ₌₀^∞ e^(−λ)λʲ/j! = λ²

E[X²] = E[X(X−1)] + E[X] = λ² + λ

Var(X) = E[X²] − (E[X])² = λ² + λ − λ² = λ  ∎
```

---

## 3. Derivation: Poisson as Limit of Binomial

This is the key insight — Poisson is what Binomial becomes when events are rare.

**Setup:** n trials, each with probability p = λ/n of success, where λ is fixed.

As n → ∞, p → 0, but np = λ stays constant.

```
P(X=k) = C(n,k) · (λ/n)ᵏ · (1 − λ/n)^(n−k)

        = [n(n−1)···(n−k+1)/k!] · (λᵏ/nᵏ) · (1−λ/n)^n · (1−λ/n)^(−k)
```

As n → ∞:
```
n(n−1)···(n−k+1)/nᵏ → 1         [k fixed, n→∞]
(1 − λ/n)^n         → e^(−λ)    [definition of e]
(1 − λ/n)^(−k)      → 1

Therefore: P(X=k) → e^(−λ)λᵏ/k!  ∎
```

**Rule of thumb for when to use Poisson approximation:**
- n ≥ 20 and p ≤ 0.05, OR
- n ≥ 100 and np ≤ 10

---

## 4. Key Properties of Poisson

### Additivity (Reproductive Property)

If X ~ Poisson(λ₁) and Y ~ Poisson(λ₂) are **independent**:
```
X + Y ~ Poisson(λ₁ + λ₂)
```

**ML use:** If server A gets λ₁ requests/sec and server B gets λ₂, the combined system sees Poisson(λ₁+λ₂) requests/sec.

### Poisson Splitting (Thinning)

If X ~ Poisson(λ) and each event is independently type 1 with prob p or type 2 with prob 1−p:
```
X₁ ~ Poisson(λp)     [type 1 events]
X₂ ~ Poisson(λ(1−p)) [type 2 events]
X₁ and X₂ are INDEPENDENT
```

**ML use:** If λ = total requests and p = fraction that are ML inference requests, then ML requests ~ Poisson(λp) — independent of other requests.

### Mode of Poisson

```
Mode = floor(λ)    if λ is not an integer
Mode = λ and λ−1  if λ is an integer (bimodal)
```

### Poisson vs Binomial: When to Use Which

| Situation | Use |
|---|---|
| Fixed n, fixed p, counting successes | Binomial(n,p) |
| Rare events, large n, fixed rate λ=np | Poisson(λ) |
| Count data with no fixed upper bound | Poisson(λ) |
| Count data where Var ≈ Mean | Poisson(λ) |
| Count data where Var > Mean | Negative Binomial (overdispersion) |

---

## 5. The Poisson Process

> **Definition:** A **Poisson process** with rate λ is a model for random events occurring over time (or space) where:
> 1. Events in non-overlapping intervals are **independent**
> 2. P(1 event in small interval Δt) ≈ λΔt
> 3. P(2+ events in Δt) ≈ 0 (no simultaneous events)

### Key Result

The number of events in any interval of length t follows:
```
N(t) ~ Poisson(λt)
```

### Inter-arrival Times

The time between consecutive events in a Poisson process follows:
```
T ~ Exponential(λ)
```

This is the critical link: **Poisson counts ↔ Exponential waiting times** (Day 11).

```
Poisson process
    ↓ count events in [0,t]     → N(t) ~ Poisson(λt)
    ↓ measure time between events → T ~ Exponential(λ)
```

### Why Exponential is Memoryless

Since inter-arrival times are Exponential, and Exponential is the continuous memoryless distribution, the Poisson process has **no memory** — the next event time doesn't depend on when the last one occurred. This is the defining property of Poisson processes.

---

## 6. Poisson Regression (ML Application)

When your target variable is a count (non-negative integer), use **Poisson regression**:

```
log(E[Y|X]) = β₀ + β₁X₁ + ... + βₚXₚ

⟺  E[Y|X] = exp(β₀ + β₁X₁ + ... + βₚXₚ)
```

The log link ensures predicted counts are always non-negative.

**Loss function:** Negative Poisson log-likelihood:
```
L = Σᵢ [λᵢ − yᵢ log λᵢ]    where λᵢ = exp(Xᵢβ)
```

**Used for:** Click counts, purchase counts, error counts, document word frequencies, disease case counts.

---

## 7. Testing for Poisson: Mean = Variance

A quick diagnostic: if data is truly Poisson, mean ≈ variance.

```
Dispersion ratio = Var(X)/E[X]

≈ 1:  Poisson (equidispersion)
> 1:  Overdispersed — use Negative Binomial
< 1:  Underdispersed — rare, use Conway-Maxwell-Poisson
```

**In NLP:** Word counts in documents are often overdispersed (Var > Mean) — this is why topic models use Negative Binomial or Dirichlet-Multinomial rather than pure Poisson.

---

## 8. Worked Numericals

---

### 🔢 Numerical 1 — Basic Poisson PMF

**Problem:** A machine learning API receives an average of 3 requests per second. Model requests as Poisson(λ=3).

**(a)** P(exactly 3 requests in 1 second)
**(b)** P(no requests in 1 second)
**(c)** P(more than 5 requests in 1 second)
**(d)** P(at least 1 request in 1 second)

**Solution:**

X ~ Poisson(3), e^(−3) ≈ 0.0498

**(a)**
```
P(X=3) = e^(−3) · 3³/3! = 0.0498 · 27/6 = 0.0498 · 4.5 = 0.2240
```

**(b)**
```
P(X=0) = e^(−3) · 3⁰/0! = e^(−3) = 0.0498
```
About 5% of seconds have zero requests.

**(c)**
```
P(X>5) = 1 − P(X≤5)
P(X=0) = 0.0498
P(X=1) = e^(−3)·3 = 0.1494
P(X=2) = e^(−3)·9/2 = 0.2240
P(X=3) = 0.2240   [from above]
P(X=4) = e^(−3)·81/24 = 0.1680
P(X=5) = e^(−3)·243/120 = 0.1008

P(X≤5) = 0.0498+0.1494+0.2240+0.2240+0.1680+0.1008 = 0.9160
P(X>5) = 1 − 0.9160 = 0.0840
```
About 8.4% of seconds have more than 5 requests.

**(d)**
```
P(X≥1) = 1 − P(X=0) = 1 − e^(−3) = 1 − 0.0498 = 0.9502
```

---

### 🔢 Numerical 2 — Poisson Approximation to Binomial

**Problem:** A dataset has 10,000 samples. Each sample is independently mislabeled with probability 0.0003.

Model the number of mislabeled samples using:
**(a)** Exact Binomial
**(b)** Poisson approximation
**(c)** Find P(more than 5 mislabeled)

**Solution:**

X ~ Binomial(10000, 0.0003)
λ = np = 10000 × 0.0003 = 3

**(a)** Exact: P(X=k) = C(10000,k) × 0.0003ᵏ × 0.9997^(10000−k)
— Computationally intensive

**(b)** Poisson approximation: X ≈ Poisson(3)
— n=10000 is large, p=0.0003 is small ✓

**(c)** P(X>5) = 1 − P(X≤5) ≈ 0.0840 (from Numerical 1)

**Exact Binomial answer:** 0.0839 — virtually identical to Poisson approximation 0.0840. ✓

**ML insight:** When checking for label noise in large datasets, Poisson is the right model. If your dataset has 1M samples and 0.01% mislabeling rate, λ=100 mislabeled samples — Poisson(100) is exact in the limit.

---

### 🔢 Numerical 3 — Poisson Process: Server Requests

**Problem:** A model inference server receives requests at rate λ = 10/minute.

**(a)** P(exactly 15 requests in 1 minute)
**(b)** P(fewer than 5 requests in 30 seconds)
**(c)** Expected time between consecutive requests
**(d)** P(next request arrives within 3 seconds)

**Solution:**

N(t) ~ Poisson(λt), T ~ Exponential(λ)

**(a)** N(1 min) ~ Poisson(10):
```
P(N=15) = e^(−10) · 10¹⁵/15!
         = 4.540×10⁻⁵ · 10¹⁵/1,307,674,368,000
         ≈ 0.0347
```

**(b)** N(0.5 min) ~ Poisson(10×0.5) = Poisson(5):
```
P(N<5) = P(N≤4) = Σₖ₌₀⁴ e^(−5)·5ᵏ/k!

e^(−5) = 0.00674

P(N=0) = 0.00674
P(N=1) = 0.03369
P(N=2) = 0.08422
P(N=3) = 0.14037
P(N=4) = 0.17547

P(N<5) = 0.44049 ≈ 44.0%
```

**(c)** Inter-arrival time T ~ Exponential(10/min):
```
E[T] = 1/λ = 1/10 min = 6 seconds
```

**(d)** T ~ Exponential(10/min) = Exponential(1/6 per second)
P(T ≤ 3 sec) = 1 − e^(−10×(3/60)) = 1 − e^(−0.5) = 1 − 0.6065 = **0.3935**

About 39.4% chance the next request arrives within 3 seconds.

---

### 🔢 Numerical 4 — Poisson Splitting: Traffic Routing

**Problem:** An ML platform receives λ = 20 requests/second total.
- 30% are image classification requests
- 50% are text generation requests
- 20% are tabular inference requests

Each type is routed to a different server independently.

**(a)** Model each request type as a Poisson process.
**(b)** P(image server receives exactly 5 requests in 1 second)
**(c)** P(text server is idle for a given second)
**(d)** Are image and text request counts independent? Why?

**Solution:**

By Poisson splitting:
- Image: Poisson(20×0.30) = Poisson(6)
- Text: Poisson(20×0.50) = Poisson(10)
- Tabular: Poisson(20×0.20) = Poisson(4)

**(a)** ✓ Each is Poisson by the splitting theorem.

**(b)**
```
P(Image=5) = e^(−6)·6⁵/5! = 0.002479·7776/120 = 0.002479·64.8 = 0.1606
```

**(c)**
```
P(Text=0) = e^(−10) = 4.54×10⁻⁵ ≈ 0.00454%
```
Text server is almost never idle.

**(d)** Yes — **by the Poisson splitting theorem**, the split streams are independent even though they came from the same process. This is non-obvious but mathematically exact.

**ML insight:** This is why microservices can be scaled independently — if you model total traffic as Poisson and routing as random splitting, each service's traffic is Poisson and independent of others.

---

### 🔢 Numerical 5 — Anomaly Detection with Poisson

**Problem:** A monitoring system detects model errors. Historical rate: λ₀ = 2 errors/hour (normal operation).

One hour, you observe 7 errors.

**(a)** Under normal operation (λ=2), what is P(X ≥ 7)?
**(b)** Is 7 errors statistically unusual? (Use threshold P < 0.01)
**(c)** If the true rate has doubled to λ=4, what is P(X ≥ 7)?
**(d)** What threshold k should trigger an alert if P(X ≥ k | λ=2) < 0.05?

**Solution:**

**(a)** X ~ Poisson(2):
```
P(X≥7) = 1 − P(X≤6)

P(X=0) = e^(−2) = 0.1353
P(X=1) = 0.2707
P(X=2) = 0.2707
P(X=3) = 0.1804
P(X=4) = 0.0902
P(X=5) = 0.0361
P(X=6) = 0.0120

P(X≤6) = 0.9955
P(X≥7) = 0.0045
```

**(b)** P(X≥7) = 0.0045 < 0.01 → **Yes, statistically unusual.** Trigger alert.

**(c)** X ~ Poisson(4):
```
P(X≥7) = 1 − P(X≤6)

P(X=0) = e^(−4) = 0.0183
P(X=1) = 0.0733
P(X=2) = 0.1465
P(X=3) = 0.1954
P(X=4) = 0.1954
P(X=5) = 0.1563
P(X=6) = 0.1042

P(X≤6) = 0.8893
P(X≥7) = 0.1107
```

With doubled rate, 7 errors is no longer unusual (11% chance).

**(d)** Find k where P(X≥k | λ=2) < 0.05:
```
P(X≥5) = 1 − P(X≤4) = 1 − (0.1353+0.2707+0.2707+0.1804+0.0902) = 0.0527 > 0.05
P(X≥6) = 1 − P(X≤5) = 1 − 0.9473 = 0.0527... let me recompute
P(X≤5) = 0.1353+0.2707+0.2707+0.1804+0.0902+0.0361 = 0.9834
P(X≥6) = 0.0166 < 0.05 ✓
```

Alert threshold: **k = 6** (trigger alert when 6 or more errors observed).

**ML insight:** This is how **statistical process control** works for ML monitoring. You fit a Poisson baseline during normal operation, then flag observations in the tail as anomalies. This is simpler and more principled than ad-hoc thresholds.

---

### 🔢 Numerical 6 — Poisson in NLP: Word Counts

**Problem:** In a corpus, the word "the" appears on average λ = 50 times per document. The word "serendipity" appears on average λ = 0.5 times per document.

**(a)** P("serendipity" appears exactly once in a document)
**(b)** P("serendipity" appears 0 times)
**(c)** P("the" appears fewer than 40 times) — use Normal approximation
**(d)** Why is Poisson appropriate for word counts?

**Solution:**

**(a)** X_s ~ Poisson(0.5):
```
P(X_s=1) = e^(−0.5)·0.5¹/1! = 0.6065·0.5 = 0.3033
```

**(b)**
```
P(X_s=0) = e^(−0.5) = 0.6065
```
60.65% of documents contain no instance of "serendipity".

**(c)** X_t ~ Poisson(50). For large λ, Poisson(λ) ≈ Normal(λ, λ):
```
P(X_t < 40) = P(Z < (40−50)/√50) = P(Z < −10/7.07) = P(Z < −1.414)
             = Φ(−1.414) ≈ 0.0786
```
About 7.9% of documents have fewer than 40 occurrences of "the".

**(d)** Word counts are non-negative integers, have no fixed upper bound, and in many models are assumed to arise from independent "slots" each potentially generating the word — matching Poisson assumptions. This underpins the **bag-of-words model** and naive language models.

**ML insight:** The Poisson assumption in NLP fails when words cluster (overdispersion). This is why Latent Dirichlet Allocation (LDA) uses a Dirichlet-Multinomial (which accounts for burstiness) rather than Poisson.

---

### 🔢 Numerical 7 — Poisson Regression Setup

**Problem:** You're building a model to predict daily bug counts for software projects. Features: X₁ = lines of code (thousands), X₂ = team size.

Training data:

| Project | X₁ (kloc) | X₂ (team) | Y (bugs/day) |
|---|---|---|---|
| A | 10 | 5 | 3 |
| B | 20 | 8 | 7 |
| C | 5  | 3 | 1 |
| D | 15 | 6 | 5 |

Fit Poisson regression: log(λ) = β₀ + β₁X₁ + β₂X₂

Suppose fitted model gives β₀=−1.0, β₁=0.08, β₂=0.15.

**(a)** Predict expected bugs/day for project E: X₁=12, X₂=7.
**(b)** Interpret β₁.
**(c)** Check: is mean ≈ variance for training data?

**Solution:**

**(a)**
```
log(λ_E) = −1.0 + 0.08×12 + 0.15×7
          = −1.0 + 0.96 + 1.05
          = 1.01
λ_E = e^(1.01) ≈ 2.75 bugs/day
```

**(b)** β₁ = 0.08 means: each additional 1000 lines of code multiplies expected bugs by e^(0.08) ≈ 1.083 — about 8.3% more bugs per kloc, holding team size constant.

**(c)** Check dispersion:
```
Observed Y: 3, 7, 1, 5
Mean = (3+7+1+5)/4 = 4.0
Variance = [(3−4)²+(7−4)²+(1−4)²+(5−4)²]/4 = [1+9+9+1]/4 = 5.0
```
Var(5.0) > Mean(4.0) → slight overdispersion. Poisson might underfit — consider Negative Binomial regression.

---

## 9. Mean = Variance: The Poisson Diagnostic

In practice, always check if Poisson is appropriate:

```
If Var/Mean ≈ 1:  Poisson is appropriate
If Var/Mean >> 1: Overdispersed — use Negative Binomial
If Var/Mean << 1: Underdispersed — rare, check data
```

**Real-world count data is almost always overdispersed** — Poisson is a starting point, not the final model.

---

## 10. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is the Poisson distribution and when do you use it?" | Count data, rare events, fixed rate λ; Mean=Var=λ |
| "Derive Poisson from Binomial" | n→∞, p→0, np=λ fixed: Binomial(n,p)→Poisson(λ) |
| "What does Mean=Variance mean in practice?" | Diagnostic for Poisson fit; overdispersion → Negative Binomial |
| "What is a Poisson process?" | Independent increments, Poisson counts, Exponential inter-arrivals |
| "What is the Poisson splitting theorem?" | Splitting Poisson stream independently → independent Poisson streams |
| "How does Poisson regression differ from linear regression?" | Log link ensures non-negative predictions; loss is Poisson log-likelihood |
| "Why are word counts modeled as Poisson?" | Non-negative integers, no fixed upper bound, independent occurrences |
| "What is overdispersion and how do you handle it?" | Var > Mean in count data → Negative Binomial regression |

---

## 11. Key Formulas — Cheat Sheet for Day 10

```
Poisson(λ):
    P(X=k) = e^(−λ)λᵏ/k!    k = 0,1,2,...
    E[X] = λ
    Var(X) = λ
    SD(X) = √λ

Poisson as Binomial limit:
    Bin(n, λ/n) → Poisson(λ)  as n→∞

Additivity:
    Poisson(λ₁) + Poisson(λ₂) = Poisson(λ₁+λ₂)  [independent]

Splitting (Thinning):
    Poisson(λ) split with prob p → Poisson(λp), Poisson(λ(1−p)), independent

Poisson Process:
    N(t) ~ Poisson(λt)         [counts in interval t]
    T ~ Exponential(λ)         [inter-arrival times]

Normal approximation (large λ):
    Poisson(λ) ≈ Normal(λ, λ)

Dispersion ratio:
    Var(X)/E[X] ≈ 1 → Poisson
                 > 1 → Negative Binomial (overdispersed)

Poisson regression:
    log(E[Y|X]) = Xβ  →  E[Y|X] = exp(Xβ)
    Loss: Σᵢ [λᵢ − yᵢ log λᵢ]
```

---

## 12. Practice Problems (Solve Before Day 11)

1. A fraud detection system flags λ=0.5 fraudulent transactions per hour on average. Find:
   - P(no fraud in 2 hours)
   - P(at least 3 fraudulent transactions in 4 hours)
   - Expected time between consecutive fraud events

2. A website gets 1000 visitors/day. Each independently makes a purchase with probability 0.002. Use the Poisson approximation to find P(exactly 3 purchases in a day).

3. **Prove** the Poisson additivity property: if X~Poisson(λ₁) and Y~Poisson(λ₂) are independent, then X+Y~Poisson(λ₁+λ₂). *(Hint: use the convolution formula P(X+Y=k) = Σⱼ P(X=j)P(Y=k−j) and the Binomial theorem.)*

4. A model API handles requests at λ=5/min. What is the probability the server is idle (0 requests) for a full minute? If you add a second independent server also at λ=5/min, what is the probability the combined system is idle?

5. *(Interview-level)* You observe the following daily error counts over 10 days: 2, 5, 1, 3, 8, 2, 4, 1, 6, 3.
   - Estimate λ̂ (MLE for Poisson is the sample mean).
   - Compute the dispersion ratio.
   - Is Poisson a good fit? What would you use instead?

---

## 13. Looking Ahead

**Day 11** — **Continuous Random Variables: PDF, CDF, Uniform & Exponential.** We move from discrete to continuous distributions — the natural home of model weights, probabilities, loss values, and waiting times. The Exponential distribution is the continuous counterpart of the Geometric and the inter-arrival time of the Poisson process.

---
*End of Day 10 | Next: Day 11 — Continuous Random Variables, PDF, CDF, Uniform & Exponential*
