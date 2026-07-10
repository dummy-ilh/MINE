# Module 2: Random Variables & Distributions
---

## 1. What is a Random Variable?

A **random variable (RV)** is a function that maps outcomes of a random experiment to real numbers.

```
X : Ω → ℝ
```

It's not "random" in a chaotic sense — it's a precise numerical encoding of uncertainty.

**Example**: Roll a die. Ω = {1,2,3,4,5,6}. Define X = outcome. Then X is a random variable that takes values 1 through 6.

**Example**: Flip 3 coins. Define X = number of heads. X can be 0, 1, 2, or 3. The randomness is in which coins land heads; X just counts them.

### Two Types

| Type | Values | Described by | Example |
|---|---|---|---|
| Discrete | Countable (finite or countably infinite) | PMF | # of clicks, # of bugs |
| Continuous | Uncountable (intervals of ℝ) | PDF | Load time, revenue, height |

---

## 2. Discrete Random Variables

### Probability Mass Function (PMF)

The PMF gives the probability that X takes each specific value:

```
p(x) = P(X = x)
```

**Requirements** (these must always hold — check these in interviews):
1. p(x) ≥ 0 for all x
2. Σₓ p(x) = 1

### Cumulative Distribution Function (CDF)

The CDF gives the probability that X is at most x:

```
F(x) = P(X ≤ x) = Σ_{t ≤ x} p(t)
```

The CDF is always:
- Non-decreasing
- Right-continuous
- F(−∞) = 0, F(+∞) = 1

**Example**: X = roll of a fair die.

```
p(1) = p(2) = ... = p(6) = 1/6

F(3) = P(X ≤ 3) = p(1)+p(2)+p(3) = 3/6 = 0.5
```

---

## 3. Expectation & Variance

### Expectation (Mean)

The **expected value** is the probability-weighted average of all possible values:

```
E[X] = Σₓ x · p(x)         (discrete)
E[X] = ∫ x · f(x) dx       (continuous)
```

**Intuition**: If you ran the experiment infinitely many times and averaged the outcomes, you'd get E[X]. It's the center of mass of the distribution.

### Properties of Expectation

```
E[aX + b]    = a·E[X] + b            (linearity)
E[X + Y]     = E[X] + E[Y]           (always, no independence needed)
E[XY]        = E[X]·E[Y]             (only if X,Y independent)
E[g(X)]      = Σₓ g(x)·p(x)         (law of the unconscious statistician)
```

### Variance

Variance measures the **spread** of the distribution around its mean:

```
Var(X) = E[(X − μ)²] = E[X²] − (E[X])²
```

The second form `E[X²] − (E[X])²` is almost always easier to compute.

**Derivation of the shortcut**:
```
Var(X) = E[(X−μ)²]
       = E[X² − 2μX + μ²]
       = E[X²] − 2μ·E[X] + μ²
       = E[X²] − 2μ² + μ²
       = E[X²] − μ²
       = E[X²] − (E[X])²
```

### Standard Deviation

```
SD(X) = σ = √Var(X)
```

Same units as X — easier to interpret than variance.

### Properties of Variance

```
Var(aX + b) = a²·Var(X)        (shifting doesn't affect spread; scaling does)
Var(X + Y)  = Var(X) + Var(Y)  (only if X,Y independent)
Var(X − Y)  = Var(X) + Var(Y)  (independent — note: still ADD variances)
```

The last line trips people up. Subtracting two independent RVs increases variance.

---

## 4. Key Discrete Distributions

### 4.1 Bernoulli(p)

Models a **single trial** with success probability p.

```
X = 1 (success) with probability p
X = 0 (failure) with probability 1−p

E[X] = p
Var(X) = p(1−p)
```

**Variance is maximized at p = 0.5** — most uncertain when 50/50.

**Google example**: Did a user click an ad? (1 = yes, 0 = no)

---

### 4.2 Binomial(n, p)

Models the **number of successes in n independent Bernoulli trials**.

```
P(X = k) = C(n,k) · pᵏ · (1−p)^(n−k)

E[X] = np
Var(X) = np(1−p)
```

**Where C(n,k) = n! / (k!(n−k)!)** — the number of ways to choose k successes from n trials.

**Intuition**: n independent coins each with P(H) = p. X counts total heads.

### 📌 Example 1: Ad Click-Through

**Q**: An ad has a 3% CTR. 200 users see it. What's the expected number of clicks and the standard deviation?

```
X ~ Binomial(n=200, p=0.03)

E[X] = 200 × 0.03 = 6 clicks

Var(X) = 200 × 0.03 × 0.97 = 5.82

SD(X) = √5.82 ≈ 2.41 clicks
```

So you expect **6 clicks, ± ~2.4**.

### 📌 Google-Level Q: Binomial Approximation

**Q**: You're testing a new Search feature on 10,000 queries. Baseline success rate is 0.1%. You expect the feature to increase it to 0.15%. How many successes do you expect under each scenario, and is the difference meaningful relative to natural variation?

```
Baseline:  E[X] = 10000 × 0.001 = 10,    SD = √(10000×0.001×0.999) ≈ 3.16
Treatment: E[X] = 10000 × 0.0015 = 15,   SD = √(10000×0.0015×0.9985) ≈ 3.87
```

The difference in means = 5. Combined SD ≈ √(3.16² + 3.87²) ≈ 5.0. Signal-to-noise ≈ 1.0 — you'd need far more queries to detect this reliably. **This is why sample size matters before running experiments.**

---

### 4.3 Poisson(λ)

Models the **number of events in a fixed interval of time or space**, when events occur independently at a constant average rate λ.

```
P(X = k) = e^(−λ) · λᵏ / k!     for k = 0, 1, 2, ...

E[X] = λ
Var(X) = λ
```

**Remarkable property**: Mean = Variance = λ. This is a diagnostic — if you see count data where variance >> mean, it's overdispersed (consider Negative Binomial).

**Poisson arises from Binomial** when n → ∞, p → 0, np → λ. Rule of thumb: use Poisson approximation when n > 20 and p < 0.05.

### 📌 Example 2: Server Errors

**Q**: A server receives on average 3 errors per hour. What's the probability of exactly 5 errors in the next hour? At least 1 error?

```
X ~ Poisson(λ=3)

P(X=5) = e^(−3) · 3⁵ / 5!
        = 0.0498 × 243 / 120
        = 0.1008 ≈ 10.1%

P(X≥1) = 1 − P(X=0)
        = 1 − e^(−3) · 3⁰ / 0!
        = 1 − e^(−3)
        = 1 − 0.0498
        ≈ 0.950
```

**95% chance of at least one error per hour.**

### 📌 Google-Level Q: Poisson Process

**Q**: Google Maps processes an average of 1,200 route requests per minute. What's the probability of 0 requests in any given second?

```
Rate per second: λ = 1200/60 = 20 requests/sec

X ~ Poisson(λ=20)

P(X=0) = e^(−20) ≈ 2.06 × 10⁻⁹
```

**Essentially zero.** At this scale, the system is never idle. This is useful for capacity planning: at λ=20, P(X>35) is also extremely small, giving you headroom estimates.

---

### 4.4 Geometric(p)

Models the **number of trials until the first success**.

```
P(X = k) = (1−p)^(k−1) · p       for k = 1, 2, 3, ...

E[X] = 1/p
Var(X) = (1−p) / p²
```

**Intuition**: How many coin flips until the first head? If p=0.5, expect 2 flips on average.

**Key property — Memorylessness**:
```
P(X > m+n | X > m) = P(X > n)
```

The geometric distribution "forgets" its past. If you've already failed m times, your remaining wait has the same distribution as if you'd just started.

### 📌 Example 3: User Retention

**Q**: A user returns to an app each day with probability 0.7. What's the expected number of days until they churn (first non-return)?

Model churn as X ~ Geometric(p = 0.3) where "success" = churn.

```
E[X] = 1/0.3 ≈ 3.33 days
```

Expected to churn after about **3.3 days**. The memoryless property also means: given they haven't churned yet, each additional day still has a 30% churn probability.

---

### 4.5 Negative Binomial(r, p)

Models the **number of trials until the r-th success**.

```
P(X = k) = C(k−1, r−1) · pʳ · (1−p)^(k−r)     k = r, r+1, ...

E[X] = r/p
Var(X) = r(1−p)/p²
```

Geometric is a special case with r=1. Used in modeling overdispersed count data (variance > mean) — e.g., comment counts per video, where some videos go viral.

---

## 5. Continuous Random Variables

### Probability Density Function (PDF)

For continuous RVs, P(X = x) = 0 for any single point. Instead:

```
P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx
```

**Requirements**:
1. f(x) ≥ 0 for all x
2. ∫_{−∞}^{∞} f(x) dx = 1

**Critical point**: f(x) is NOT a probability. It's a density. f(x) can exceed 1. What matters is the area under the curve.

### CDF for Continuous RVs

```
F(x) = P(X ≤ x) = ∫_{−∞}^{x} f(t) dt

f(x) = F'(x)     (PDF is derivative of CDF)
```

---

## 6. Key Continuous Distributions

### 6.1 Uniform(a, b)

Every value in [a,b] is equally likely.

```
f(x) = 1/(b−a)     for a ≤ x ≤ b

E[X] = (a+b)/2
Var(X) = (b−a)²/12
```

**Example**: A bus arrives every 10 minutes. You arrive at a random time. Your wait X ~ Uniform(0, 10). E[X] = 5 minutes.

---

### 6.2 Normal (Gaussian) — N(μ, σ²)

The most important distribution in statistics.

```
f(x) = (1/√(2πσ²)) · exp(−(x−μ)²/(2σ²))

E[X] = μ
Var(X) = σ²
```

**Standard Normal**: Z ~ N(0,1). Any Normal can be standardized:

```
Z = (X − μ) / σ
```

### Key Percentiles to Memorize

```
P(|Z| < 1) ≈ 68%     (μ ± 1σ)
P(|Z| < 2) ≈ 95%     (μ ± 2σ)
P(|Z| < 3) ≈ 99.7%   (μ ± 3σ)

z₀.₀₅  = 1.645    (one-tailed 95%)
z₀.₀₂₅ = 1.960    (two-tailed 95%)
z₀.₀₀₅ = 2.576    (two-tailed 99%)
```

### Why Normal is Everywhere: CLT Preview

Sums of many independent RVs (from ANY distribution) converge to Normal as n grows. This is the Central Limit Theorem (covered in Module 3) — it's why the Normal appears in A/B testing, regression, and nearly every inference procedure.

### 📌 Example 4: Page Load Time

**Q**: Page load times follow N(μ=2.0s, σ=0.4s). What fraction of pages load in under 2.8 seconds?

```
P(X < 2.8) = P(Z < (2.8−2.0)/0.4)
            = P(Z < 2.0)
            ≈ 0.9772
```

**97.72% of pages load in under 2.8 seconds.** (2s above the mean = 97.7th percentile.)

### 📌 Google-Level Q: SLA Threshold

**Q**: Latency for a Google API call ~ N(120ms, 20²). Your SLA requires p99 latency under 180ms. Are you meeting it?

```
P(X < 180) = P(Z < (180−120)/20)
            = P(Z < 3.0)
            ≈ 0.9987
```

**p99 latency ≈ 99.87th percentile → yes, 180ms is your ~p999.** You're well within SLA. The actual p99 is:

```
p99 = μ + 2.326σ = 120 + 2.326×20 = 120 + 46.5 ≈ 166.5ms
```

p99 is 166.5ms, comfortably below 180ms.

---

### 6.3 Exponential(λ)

Models the **time between events** in a Poisson process. Continuous analogue of the Geometric.

```
f(x) = λ·e^(−λx)       for x ≥ 0

E[X] = 1/λ
Var(X) = 1/λ²
SD(X) = 1/λ
```

**Memorylessness** (same as Geometric, now continuous):
```
P(X > s+t | X > s) = P(X > t)
```

If a user hasn't clicked in 5 minutes, the remaining time until click still has the same Exponential distribution.

### 📌 Example 5: Time Between Purchases

**Q**: Users make purchases at a rate of 2 per month on average. What's the probability a user makes their next purchase within 2 weeks (0.5 months)?

```
X ~ Exponential(λ=2)

P(X ≤ 0.5) = 1 − e^(−λ·0.5)
            = 1 − e^(−1)
            = 1 − 0.368
            ≈ 0.632
```

**63.2% chance of a purchase within 2 weeks.**

Note: This 63.2% ≈ 1−1/e appears naturally for X < E[X]. At the mean, about 63% of the distribution has already occurred.

---

### 6.4 Beta(α, β)

Models **probabilities** — values constrained to [0,1]. This is the natural prior for a conversion rate or CTR.

```
f(x) = x^(α−1) · (1−x)^(β−1) / B(α,β)     for 0 ≤ x ≤ 1

E[X] = α / (α+β)
Var(X) = αβ / ((α+β)²(α+β+1))
```

**Interpretation of parameters**:
- α−1 = prior successes
- β−1 = prior failures
- α+β = strength of prior (total prior observations)

**Used in Bayesian A/B testing**: If you observe k successes in n trials (Binomial likelihood), and your prior is Beta(α,β), then the posterior is:

```
Beta(α+k, β+n−k)
```

This is the conjugate prior relationship.

### 📌 Google-Level Q: Bayesian CTR

**Q**: You believe an ad's CTR is around 5% based on 100 historical impressions (5 clicks). You then run a new campaign with 200 impressions and get 14 clicks. What's your posterior belief about the CTR?

**Prior**: Beta(α=5, β=95) — from 5 clicks, 95 non-clicks historically.

**Likelihood update**: 14 successes, 186 failures from new data.

**Posterior**:
```
Beta(5+14, 95+186) = Beta(19, 281)

Posterior mean = 19 / (19+281) = 19/300 ≈ 0.0633
```

**Posterior CTR estimate: ~6.3%** — pulled between the prior (5%) and new data (7%), weighted by sample sizes.

---

### 6.5 Log-Normal

If X ~ Normal(μ, σ²), then Y = eˣ is **Log-Normal**.

```
E[Y] = e^(μ + σ²/2)
Var(Y) = (e^(σ²) − 1) · e^(2μ+σ²)
```

**When to use**: Revenue, session length, file sizes — anything positive and right-skewed. A common mistake is modeling these as Normal when they're log-Normal, leading to poor estimates.

### 📌 Google-Level Q: Revenue Distribution

**Q**: Why shouldn't you model user revenue as Normal? What should you use instead, and how does it change your A/B test analysis?

**Answer**: Revenue is:
- **Bounded below by 0** (can't have negative revenue)
- **Right-skewed** — a few users drive disproportionate revenue
- **Often log-Normal** in practice

Using Normal:
- Can predict negative revenue (meaningless)
- Mean is dragged by outliers; variance is inflated
- t-test assumes normality; violated by skew at small samples

Better approaches:
1. **Log-transform** revenue and test on log scale
2. Use **Mann-Whitney U test** (non-parametric)
3. Use **CUPED** to reduce variance
4. **Cap outliers** and analyze separately

At Google, revenue experiments are often analyzed with log-transformed metrics or bootstrapped CIs precisely because of this.

---

## 7. Moment Generating Functions (MGFs)

### Definition

```
M_X(t) = E[eᵗˣ]
```

The MGF encodes all moments of X. The k-th moment is:

```
E[Xᵏ] = M_X^(k)(0)     (k-th derivative of MGF evaluated at t=0)
```

### Why MGFs Matter

1. **Uniqueness**: If two RVs have the same MGF, they have the same distribution.
2. **Sums of independent RVs**: M_{X+Y}(t) = M_X(t) · M_Y(t)
3. **Proving CLT**: MGF of standardized sum → MGF of Normal.

### Common MGFs

| Distribution | MGF M(t) |
|---|---|
| Bernoulli(p) | 1−p + p·eᵗ |
| Binomial(n,p) | (1−p + p·eᵗ)ⁿ |
| Poisson(λ) | exp(λ(eᵗ−1)) |
| Normal(μ,σ²) | exp(μt + σ²t²/2) |
| Exponential(λ) | λ/(λ−t) for t < λ |

**Key use**: To find E[X] and Var(X) quickly without integration.

**Example** — Normal MGF gives:
```
M'(t)  = (μ + σ²t)·exp(μt + σ²t²/2)
M'(0)  = μ            → E[X] = μ  ✓

M''(t) at t=0 = σ² + μ²   → E[X²] = σ² + μ²
Var(X) = E[X²]−(E[X])² = σ²  ✓
```

---

## 8. Chebyshev's Inequality

A universal bound on how much probability can be far from the mean, for ANY distribution:

```
P(|X − μ| ≥ kσ) ≤ 1/k²
```

Or equivalently:
```
P(|X − μ| < kσ) ≥ 1 − 1/k²
```

**Example**: k=2 → at least 75% of probability lies within 2 standard deviations. (Compare: Normal gives 95%. Chebyshev is a weaker, distribution-free guarantee.)

**Use in interviews**: When you can't assume a distribution, Chebyshev gives conservative but valid bounds.

---

## Q&A Section

### Q1 (Warm-up)
**Q**: X ~ Binomial(10, 0.4). What are E[X] and Var(X)?

**A**:
```
E[X] = np = 10 × 0.4 = 4
Var(X) = np(1−p) = 10 × 0.4 × 0.6 = 2.4
SD(X) = √2.4 ≈ 1.55
```

---

### Q2 (PMF Reasoning)
**Q**: You roll two dice. X = the maximum of the two rolls. What is P(X = 4)?

**A**: X = 4 means both dice ≤ 4, and at least one die = 4.

```
P(both ≤ 4) = (4/6)² = 16/36
P(both ≤ 3) = (3/6)² = 9/36

P(X = 4) = 16/36 − 9/36 = 7/36 ≈ 0.194
```

---

### Q3 (Google-Level: Poisson)
**Q**: A Google data center has two independent server clusters. Cluster A fails λ_A = 0.5 times/day on average. Cluster B fails λ_B = 0.3 times/day. What is the expected total failures per day and the probability of zero total failures?

**A**: Sum of independent Poissons is Poisson with λ = λ_A + λ_B.

```
Total X ~ Poisson(0.5 + 0.3) = Poisson(0.8)

E[X] = 0.8 failures/day

P(X=0) = e^(−0.8) ≈ 0.449
```

**~45% chance of a failure-free day.**

---

### Q4 (Normal Standardization)
**Q**: Session durations ~ N(4.5 min, 1.2²). What % of sessions are between 3 and 6 minutes?

**A**:
```
P(3 < X < 6) = P((3−4.5)/1.2 < Z < (6−4.5)/1.2)
             = P(−1.25 < Z < 1.25)
             = 2·Φ(1.25) − 1
             ≈ 2(0.8944) − 1
             = 0.7888
```

**~78.9% of sessions fall between 3 and 6 minutes.**

---

### Q5 (Google-Level: Geometric / Memorylessness)
**Q**: Each time a user sees a Google ad, they click with probability 0.08. A user has seen the ad 5 times without clicking. What's the expected total number of impressions until their first click, starting from now?

**A**: By memorylessness of the Geometric distribution:

```
E[additional impressions] = 1/p = 1/0.08 = 12.5
```

The 5 past failures are irrelevant. They still need **12.5 more impressions on average**. Total expected impressions from the start = 5 + 12.5 = 17.5, but from now: **12.5**.

---

### Q6 (Google-Level: Distribution Choice)
**Q**: You're modeling the number of YouTube comments per video. You assume Poisson(λ=50). But you notice the sample variance is 2,400 — much larger than 50. What's wrong and what should you use?

**A**: Poisson requires **mean = variance**. Variance of 2,400 >> mean of 50 signals **overdispersion**. Poisson is too restrictive — it can't model this spread.

**Better choice**: **Negative Binomial distribution**, which allows variance > mean. It adds a dispersion parameter r:

```
Variance = μ + μ²/r
```

As r → ∞, Negative Binomial → Poisson. Small r = heavy overdispersion.

Overdispersion often arises from heterogeneity in the population: some videos go viral (massive comments), most get very few. A mixture of Poissons with varying λ produces exactly a Negative Binomial.

---

## Cheat Sheet: Module 2

```
┌─────────────────────────────────────────────────────────────────────┐
│               RANDOM VARIABLES & DISTRIBUTIONS                       │
├─────────────────────────────────────────────────────────────────────┤
│  EXPECTATION & VARIANCE                                              │
│  E[aX+b]      = a·E[X] + b                                          │
│  E[X+Y]       = E[X] + E[Y]           (always)                      │
│  Var(aX+b)    = a²·Var(X)                                           │
│  Var(X+Y)     = Var(X)+Var(Y)         (if independent)              │
│  Var(X)       = E[X²] − (E[X])²       (shortcut)                   │
├─────────────────────────────────────────────────────────────────────┤
│  DISCRETE DISTRIBUTIONS                                              │
│                                                                       │
│  Bernoulli(p)         E=p          Var=p(1−p)                        │
│  Binomial(n,p)        E=np         Var=np(1−p)                       │
│  Poisson(λ)           E=λ          Var=λ          ← Mean=Var         │
│  Geometric(p)         E=1/p        Var=(1−p)/p²   ← Memoryless      │
│  Neg.Binomial(r,p)    E=r/p        Var=r(1−p)/p²                    │
├─────────────────────────────────────────────────────────────────────┤
│  CONTINUOUS DISTRIBUTIONS                                            │
│                                                                       │
│  Uniform(a,b)         E=(a+b)/2    Var=(b−a)²/12                    │
│  Normal(μ,σ²)         E=μ          Var=σ²                            │
│  Exponential(λ)       E=1/λ        Var=1/λ²       ← Memoryless      │
│  Beta(α,β)            E=α/(α+β)    Var=αβ/((α+β)²(α+β+1))          │
├─────────────────────────────────────────────────────────────────────┤
│  NORMAL PERCENTILES                                                  │
│  ±1σ → 68%    ±2σ → 95%    ±3σ → 99.7%                             │
│  z(95% one-tail) = 1.645                                            │
│  z(95% two-tail) = 1.960                                            │
│  z(99% two-tail) = 2.576                                            │
├─────────────────────────────────────────────────────────────────────┤
│  DISTRIBUTION SELECTION GUIDE                                        │
│  Single 0/1 trial              → Bernoulli                          │
│  Count of successes in n       → Binomial                           │
│  Count of events in time/space → Poisson                            │
│  Trials until first success    → Geometric                          │
│  Time between events           → Exponential                        │
│  Probability/rate estimation   → Beta                               │
│  Positive skewed (revenue)     → Log-Normal                         │
│  Count with variance >> mean   → Negative Binomial                  │
├─────────────────────────────────────────────────────────────────────┤
│  CHEBYSHEV (distribution-free)                                       │
│  P(|X−μ| ≥ kσ) ≤ 1/k²                                              │
│  k=2: at least 75% within 2σ                                        │
│  k=3: at least 89% within 3σ                                        │
├─────────────────────────────────────────────────────────────────────┤
│  TRAPS                                                               │
│  ✗ f(x) is density, not probability — can exceed 1                  │
│  ✗ Var(X−Y) = Var(X)+Var(Y), NOT Var(X)−Var(Y)                     │
│  ✗ Poisson mean=variance; if not, use Neg. Binomial                 │
│  ✗ Revenue/session time ≠ Normal → use log-transform               │
│  ✗ Memorylessness only holds for Geometric & Exponential            │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Next → Module 3: Joint Distributions, Covariance, CLT & LLN*
