# Module 2: Random Variables & Distributions


---

## 🎯 The 5 Things You Must Know

If you remember nothing else from this module, remember these five — they show up in almost every quant interview question at Google:

1. **Linearity of Expectation** — `E[aX + bY + c] = aE[X] + bE[Y] + c`, and it **never requires independence**. This is the single most powerful tool for solving "expected number of X" problems without brute-force enumeration.
2. **Variance of sums** — `Var(X ± Y) = Var(X) + Var(Y)` **only if independent**, and note the `+` even for subtraction. Combining independent sources of randomness always increases uncertainty, never cancels it.
3. **Normal percentiles** — `68/95/99.7` rule, plus `z₀.₀₅=1.645`, `z₀.₀₂₅=1.960`, `z₀.₀₀₅=2.576`. You should be able to compute a p95 or p99 latency SLA in your head.
4. **Distribution picker** — given a word problem, you should be able to name the right distribution in under 5 seconds (table below).
5. **Memorylessness** — only Geometric and Exponential have it. Everything else "remembers" its history, and assuming otherwise is a classic trap.

### Quick Reference Table

| Distribution | Use it when... | E[X] | Var(X) | Google Example |
|---|---|---|---|---|
| Bernoulli(p) | Single yes/no trial | p | p(1−p) | Did the user click this ad? |
| Binomial(n,p) | # successes in n trials | np | np(1−p) | Conversions out of n users in an A/B test |
| Poisson(λ) | # events in fixed time/space, constant rate | λ | λ | Outages per quarter, requests per second |
| Geometric(p) | # trials until first success | 1/p | (1−p)/p² | Days until a user churns |
| Neg. Binomial(r,p) | # trials until r-th success | r/p | r(1−p)/p² | Comment counts on videos (overdispersed) |
| Uniform(a,b) | Equally likely over a range | (a+b)/2 | (b−a)²/12 | Wait time for a load-balanced request |
| Normal(μ,σ²) | Aggregated/averaged metrics | μ | σ² | A/B test lift, latency averages |
| Exponential(λ) | Time between events | 1/λ | 1/λ² | Time between ad clicks |
| Beta(α,β) | Estimating a probability/rate | α/(α+β) | αβ/((α+β)²(α+β+1)) | Bayesian CTR estimation |
| Log-Normal | Positive, right-skewed metrics | e^(μ+σ²/2) | (e^σ²−1)e^(2μ+σ²) | Revenue per user, session length |

---

## 1. What is a Random Variable?

Every time Google serves a search result, there's a random variable running quietly in the background. The user might click. They might not. That single bit of uncertainty — click or no click — is exactly what a random variable encodes as math.

Formally, a **random variable (RV)** is a function that maps outcomes of a random experiment to real numbers:

```
X : Ω → ℝ
```

It's not "random" in a chaotic sense — it's a precise numerical encoding of uncertainty. The randomness lives in which outcome occurs; X is just the rule for turning that outcome into a number you can do math with.

**Example**: Roll a die. Ω = {1,2,3,4,5,6}. Define X = outcome. Then X is a random variable that takes values 1 through 6.

**Example**: Flip 3 coins. Define X = number of heads. X can be 0, 1, 2, or 3. The randomness is in which coins land heads; X just counts them.

### Two Types

| Type | Values | Described by | Example |
|---|---|---|---|
| Discrete | Countable (finite or countably infinite) | PMF | # of clicks, # of bugs |
| Continuous | Uncountable (intervals of ℝ) | PDF | Load time, revenue, height |

**Why this matters in your Google interview**: The interviewer is checking whether you can translate a messy word problem into a clean random variable before touching any formulas. Candidates who jump straight to "np(1−p)" without first stating "let X = number of successes" tend to make sign errors and mis-specify independence later.

---

## 2. Discrete Random Variables

### Probability Mass Function (PMF)

The PMF gives the probability that X takes each specific value:

```
p(x) = P(X = x)
```

**Intuitive example (not the die)**: Say Google Ads shows a user 3 ad slots, and X = number of slots the user engages with (clicks or hovers meaningfully). Suppose historical data gives:

```
p(0) = 0.50, p(1) = 0.30, p(2) = 0.15, p(3) = 0.05
```

This is a full description of the user's engagement behavior — no more, no less. Everything you'd ever want to know about X (its mean, variance, probability of at least one engagement) is derivable from this table.

**PMF Checklist** — before you trust any function is a valid PMF, verify:
1. **Non-negativity**: p(x) ≥ 0 for all x
2. **Normalization**: Σₓ p(x) = 1

Check the ad example: 0.50+0.30+0.15+0.05 = 1.00 ✓, and all terms are non-negative ✓. This is a valid PMF.

**PMF and sets**: The PMF doesn't just answer "what's the probability of exactly this value" — it lets you compute the probability of any event A directly:

```
P(X ∈ A) = Σ_{x ∈ A} p(x)
```

Using the ad example, P(at least 1 engagement) = p(1)+p(2)+p(3) = 0.30+0.15+0.05 = 0.50.

### Cumulative Distribution Function (CDF)

The CDF gives the probability that X is at most x:

```
F(x) = P(X ≤ x) = Σ_{t ≤ x} p(t)
```

**Why CDF is more general than PMF**: The PMF only exists cleanly for discrete RVs (continuous RVs have P(X=x)=0 everywhere). The CDF, by contrast, is defined identically for both discrete and continuous random variables — it's always `P(X ≤ x)`. This is why almost every statistical table (z-tables, t-tables) is really a CDF table, and why software libraries standardize around `.cdf()` methods rather than `.pmf()` methods when comparing across distribution types.

The CDF is always:
- Non-decreasing
- Right-continuous
- F(−∞) = 0, F(+∞) = 1

**PMF → CDF → PMF, step by step**, using the ad engagement example:

```
p(0)=0.50, p(1)=0.30, p(2)=0.15, p(3)=0.05

F(0) = p(0)                = 0.50
F(1) = p(0)+p(1)           = 0.80
F(2) = p(0)+p(1)+p(2)      = 0.95
F(3) = p(0)+p(1)+p(2)+p(3) = 1.00
```

Going backward (CDF → PMF), you just take successive differences:

```
p(0) = F(0)        = 0.50
p(1) = F(1)−F(0)   = 0.30
p(2) = F(2)−F(1)   = 0.15
p(3) = F(3)−F(2)   = 0.05
```

**The staircase visualization**: For a discrete RV, F(x) is a step function — flat everywhere except at each possible value of X, where it **jumps up** by exactly p(x). Picture the ad example as stairs: flat at height 0 until x=0, then a jump of 0.50 straight up, flat until x=1, then a jump of 0.30, and so on until you reach the ceiling at 1.00. The height of each jump *is* the PMF; the height of the stair *is* the CDF. This mental picture is the fastest way to sanity-check a CDF problem on a whiteboard — if your steps don't sum to 1, you've made an arithmetic error somewhere.

**Signal you know this**: If you say "the jump size at each point of the CDF staircase equals the PMF at that point," you've shown you understand these aren't two separate objects — they're two views of the same distribution.

---

## 3. Expectation & Variance

### Expectation (Mean)

Expected value is what Google's ad auction optimizes. Every bid, every impression, every revenue forecast — it all traces back to E[X].

The **expected value** is the probability-weighted average of all possible values:

```
E[X] = Σₓ x · p(x)         (discrete)
E[X] = ∫ x · f(x) dx       (continuous)
```

**Intuition**: If you ran the experiment infinitely many times and averaged the outcomes, you'd get E[X]. It's the center of mass of the distribution.

### Linearity of Expectation — the superpower

```
E[aX + bY + c] = a·E[X] + b·E[Y] + c
```

This holds **always** — no independence required. That last part is what makes it so powerful: you can decompose a hard, correlated problem into simple pieces, sum their expectations, and get the right answer without ever touching the joint distribution.

**Memory hook**: Linearity of expectation is like splitting a dinner bill — you don't need to know how the whole group ordered to calculate your share. Each person's expected cost adds up regardless of who ordered the lobster.

**Classic interview problem — Expected tosses to get 3 consecutive heads**

*The complicated way*: Set up a Markov chain with states {0, 1, 2, 3} representing the current streak of consecutive heads, solve a system of equations for expected hitting time from each state. It works, but it's slow and error-prone under interview pressure.

*The elegant way (linearity + recursion is still needed here, but a cleaner linearity-style problem is more illustrative — so let's use the standard textbook example instead)*: **Coupon Collector Problem.** You want to collect all n distinct items (e.g., n distinct badge types Google randomly attaches to app installs), drawing one uniformly at random each time, with replacement. What's the expected number of draws to collect all n?

Let T = total draws, and break it into stages: T = T₁ + T₂ + ... + Tₙ, where Tᵢ = number of draws while you have exactly i−1 distinct items, until you get a new one. Each Tᵢ is Geometric with success probability `p_i = (n−i+1)/n` (probability the next draw is a "new" item).

```
E[Tᵢ] = n / (n−i+1)

E[T] = Σᵢ₌₁ⁿ E[Tᵢ] = n·Σᵢ₌₁ⁿ 1/i = n·Hₙ ≈ n·ln(n)
```

For n=50 badge types: E[T] ≈ 50 × ln(50) ≈ 50 × 3.91 ≈ 195.6 draws.

You never needed the joint distribution of all n draws. You just summed n separate, individually-tractable expectations. That's the power of linearity — it turns an intractable combinatorial mess into n one-line Geometric calculations.

**Google-scale example**: Expected total ad revenue from n impressions equals the sum of individual expected revenues per impression — **even if the impressions are correlated** (e.g., the same user seeing multiple ads in one session, where seeing one ad changes the odds of engaging with the next). This is exactly why linearity is the backbone of bidding and revenue forecasting systems: you can forecast total expected revenue without ever modeling the correlation structure between impressions.

**Why this matters in your Google interview**: The interviewer wants to know if you can think in terms of averages, not just isolated outcomes. They'll ask questions like "what's the expected profit of this campaign?" and you need to handle correlation, linearity, and conditional expectation naturally.

**The trap**: Reaching for the raw definition `E[X] = Σx·p(x)` on every problem, even when the joint distribution is nasty or unknown. In most elegant interview solutions, you decompose X into a sum of simpler indicator or sub-variables and apply linearity — you almost never need the full joint distribution.

**Signal you know this**: If you can say *"I don't need the joint distribution here — linearity of expectation lets me work with the marginals,"* you've just separated yourself from most candidates.

### E[g(X)] — The Law of the Unconscious Statistician (LOTUS)

```
E[g(X)] = Σₓ g(x)·p(x)       (discrete)
E[g(X)] = ∫ g(x)·f(x) dx     (continuous)
```

**What it means**: You don't need to derive the distribution of g(X) to compute its expectation — you can plug g(x) directly into the weights given by X's original distribution. It "unconsciously" skips a step that seems necessary but isn't.

**Interview trap**: People try to compute E[X²] by first finding the distribution of the new random variable Y=X², then computing E[Y] = Σy·p_Y(y). That's unnecessary work. LOTUS says: just compute Σx²·p(x) directly, using X's own PMF. This is *exactly* how the variance shortcut formula works — you never derive the distribution of X².

### Variance

Variance measures the **spread** of the distribution around its mean — the expected squared distance from the mean:

```
Var(X) = E[(X − μ)²] = E[X²] − (E[X])²
```

**Why "squared"**: If you used `E[X − μ]` directly, positive and negative deviations would cancel out and you'd always get 0, no matter how spread out the distribution actually is. Squaring forces every deviation to contribute positively, so variance actually captures spread rather than washing it out.

**Units trap**: If X is measured in seconds, `Var(X)` is in **seconds²** — an awkward, hard-to-interpret unit. This is exactly why standard deviation, `SD(X) = √Var(X)`, is usually reported instead: it's back in the original units (seconds), which is far more interpretable to a stakeholder.

**Derivation of the shortcut formula**:
```
Var(X) = E[(X−μ)²]
       = E[X² − 2μX + μ²]
       = E[X²] − 2μ·E[X] + μ²
       = E[X²] − 2μ² + μ²
       = E[X²] − μ²
       = E[X²] − (E[X])²
```

The second form is almost always easier to compute — this is another direct application of LOTUS, since `E[X²]` is just `Σx²·p(x)`.

### Chebyshev's Inequality (a variance concept, first and foremost)

Variance measures spread, but Chebyshev's inequality tells you *how much* probability that spread guarantees stays near the mean — for **any** distribution, no shape assumptions required:

```
P(|X − μ| ≥ kσ) ≤ 1/k²
```

Or equivalently:
```
P(|X − μ| < kσ) ≥ 1 − 1/k²
```

**Memory hook**: Chebyshev's inequality is the "fire alarm" of statistics — it works for every distribution, but it gives you a conservative, worst-case bound. It won't tell you the exact probability, but it guarantees you're never blindsided.

**Concrete Google example**: Say a service's response time has mean μ=100ms and SD σ=15ms, but you don't know (or don't want to assume) its shape — maybe it's a weird bimodal distribution from mixed traffic types. What's a guaranteed lower bound on the fraction of requests within ±30ms of the mean (i.e., within 2σ)?

```
k = 2 → P(|X−100| < 30) ≥ 1 − 1/4 = 0.75
```

**At least 75% of requests fall within 70–130ms — guaranteed, regardless of the true distribution shape.** Compare this to the Normal-distribution answer of ~95% for the same k=2 — Chebyshev is deliberately weaker but distribution-free, which is exactly why it's the right tool when you can't assume normality (e.g., early in an experiment, before you have enough data to check shape).

**Use in interviews**: When you can't assume a distribution, Chebyshev gives conservative but valid bounds — and saying so explicitly (rather than defaulting to Normal-style percentiles) is a strong signal of statistical maturity.

### Properties of Expectation & Variance — reference block

```
E[aX + b]    = a·E[X] + b            (linearity)
E[X + Y]     = E[X] + E[Y]           (always, no independence needed)
E[XY]        = E[X]·E[Y]             (only if X,Y independent)
E[g(X)]      = Σₓ g(x)·p(x)         (LOTUS)

Var(aX + b) = a²·Var(X)        (shifting doesn't affect spread; scaling does)
Var(X + Y)  = Var(X) + Var(Y)  (only if X,Y independent)
Var(X − Y)  = Var(X) + Var(Y)  (independent — note: still ADD variances)
```

The last line trips people up constantly. Subtracting two independent random variables **increases** variance — you're adding two independent sources of uncertainty together, and the direction of subtraction doesn't cancel that out. This shows up directly in A/B testing: the variance of the difference in means between treatment and control is the *sum* of their variances, never the difference.

### Mode, Median, and Skew

These describe the "center" of a distribution from different angles, and interviewers love asking when to use which.

- **Mode**: the value with the highest probability (discrete) or highest density (continuous) — the peak of the distribution.
- **Median**: the 50th percentile — the value where F(x) = 0.5. Half the probability mass lies below it, half above.
- **Mean**: the probability-weighted average, E[X].

**How they relate to skew**:

```
Symmetric distribution:     mean = median = mode
Right-skewed (long right tail): mean > median > mode
Left-skewed (long left tail):   mean < median < mode
```

**Interview context — "When should you use median instead of mean?"** The classic answer is **revenue data**. A handful of whale users can pull the mean revenue per user far above what a "typical" user actually generates, while the median stays anchored near where most of the mass actually sits. If you report mean revenue lift in an A/B test with a few extreme outliers, you can convince yourself a change worked when in fact one whale user's random large purchase drove the entire effect. Reporting the median (or a trimmed/winsorized mean) alongside the mean is a strong signal of statistical maturity in a product-sense interview.

---

## 4. Key Discrete Distributions

### 4.1 Bernoulli(p)

**Google in the wild**: Did a user click this ad? Yes/No. That's a Bernoulli trial — and Google runs billions of these every single day.

Models a **single trial** with success probability p.

```
X = 1 (success) with probability p
X = 0 (failure) with probability 1−p

E[X] = p
Var(X) = p(1−p)
```

**Variance is maximized at p = 0.5** — most uncertain when it's a true coin flip. As p moves toward 0 or 1, the outcome becomes more predictable and variance shrinks toward 0.

### 4.2 Binomial(n, p)

**Google in the wild**: Search ranking experiments — X = number of successful queries out of n. Binomial gives you the variance you need to know whether your change actually worked, or whether you just got lucky noise.

Models the **number of successes in n independent Bernoulli trials**.

```
P(X = k) = C(n,k) · pᵏ · (1−p)^(n−k)

E[X] = np
Var(X) = np(1−p)
```

Where `C(n,k) = n! / (k!(n−k)!)` is the number of ways to choose k successes from n trials.

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

**Google in the wild**: Every time you see "About 1,200 results" — that's not an exact count. But for rare events at Google scale — Cloud incident reports, hardware failures, spam spikes — Poisson is your first call.

Models the **number of events in a fixed interval of time or space**, when events occur independently at a constant average rate λ.

```
P(X = k) = e^(−λ) · λᵏ / k!     for k = 0, 1, 2, ...

E[X] = λ
Var(X) = λ
```

**Remarkable property**: Mean = Variance = λ. This is a diagnostic — if you see count data where variance >> mean, it's overdispersed (consider Negative Binomial).

**Poisson arises from Binomial** when n → ∞, p → 0, np → λ. Rule of thumb: use the Poisson approximation to Binomial when n > 20 and p < 0.05.

### 📌 Example 2: Server Errors

**Q**: A server receives on average 3 errors per hour. What's the probability of exactly 5 errors in the next hour? At least 1 error?

```
X ~ Poisson(λ=3)

P(X=5) = e^(−3) · 3⁵ / 5! = 0.0498 × 243 / 120 ≈ 0.1008 (10.1%)

P(X≥1) = 1 − P(X=0) = 1 − e^(−3) ≈ 1 − 0.0498 ≈ 0.950
```

**95% chance of at least one error per hour.**

### 📌 Google-Level Q: Poisson Process

**Q**: Google Maps processes an average of 1,200 route requests per minute. What's the probability of 0 requests in any given second?

```
Rate per second: λ = 1200/60 = 20 requests/sec
X ~ Poisson(λ=20)
P(X=0) = e^(−20) ≈ 2.06 × 10⁻⁹
```

**Essentially zero.** At this scale, the system is never idle. This is useful for capacity planning: at λ=20, P(X>35) is also extremely small, giving you a headroom estimate for provisioning.

---

### 4.4 Geometric(p)

**Google in the wild**: Retention — how many days until a user stops coming back? Geometric models the "waiting time to churn."

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

Expected to churn after about **3.3 days**. The memoryless property also means: given they haven't churned yet, each additional day still has a 30% churn probability — no "wearing down" effect.

---

### 4.5 Negative Binomial(r, p)

**Google in the wild**: Comment counts per YouTube video, where some videos go viral and most get very few — a classic overdispersed count.

Models the **number of trials until the r-th success**.

```
P(X = k) = C(k−1, r−1) · pʳ · (1−p)^(k−r)     k = r, r+1, ...

E[X] = r/p
Var(X) = r(1−p)/p²
```

Geometric is the special case r=1. Used to model overdispersed count data (variance > mean).

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

**Critical point — the trap**: f(x) is **NOT** a probability. It's a density. f(x) can exceed 1 (e.g., a Uniform(0, 0.5) has f(x)=2 everywhere on its support). What matters is the **area under the curve**, never the height at a single point.

### CDF for Continuous RVs

```
F(x) = P(X ≤ x) = ∫_{−∞}^{x} f(t) dt

f(x) = F'(x)     (PDF is the derivative of the CDF)
```

---

## 6. Key Continuous Distributions

### 6.1 Uniform(a, b)

**Google in the wild**: Load balancing — when requests hit a server uniformly at random within a time window, wait times follow Uniform. That's how Google's frontends distribute traffic evenly.

Every value in [a,b] is equally likely.

```
f(x) = 1/(b−a)     for a ≤ x ≤ b

E[X] = (a+b)/2
Var(X) = (b−a)²/12
```

**Example**: A bus arrives every 10 minutes. You arrive at a random time. Your wait X ~ Uniform(0, 10). E[X] = 5 minutes.

---

### 6.2 Normal (Gaussian) — N(μ, σ²)

**Google in the wild**: Basically everything after the Central Limit Theorem kicks in — A/B test lift, latency averages, any aggregated metric.

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

Sums of many independent RVs — from **any** underlying distribution — converge to Normal as n grows. This is the Central Limit Theorem (covered in full in Module 3), and it's why the Normal appears in A/B testing, regression, and nearly every inference procedure, even when the raw underlying data isn't Normal at all.

### 📌 Example 4: Page Load Time

**Q**: Page load times follow N(μ=2.0s, σ=0.4s). What fraction of pages load in under 2.8 seconds?

```
P(X < 2.8) = P(Z < (2.8−2.0)/0.4) = P(Z < 2.0) ≈ 0.9772
```

**97.72% of pages load in under 2.8 seconds.**

### 📌 Google-Level Q: SLA Threshold

**Q**: Latency for a Google API call ~ N(120ms, 20²). Your SLA requires p99 latency under 180ms. Are you meeting it?

```
P(X < 180) = P(Z < (180−120)/20) = P(Z < 3.0) ≈ 0.9987
```

180ms sits at roughly the 99.87th percentile. Your actual p99 latency is:

```
p99 = μ + 2.326σ = 120 + 2.326×20 = 120 + 46.5 ≈ 166.5ms
```

**p99 ≈ 166.5ms, comfortably below the 180ms SLA.**

---

### 6.3 Exponential(λ)

**Google in the wild**: The time between your searches, or the gap between ad clicks — these are Exponential, and memorylessness makes them surprisingly simple to model.

Models the **time between events** in a Poisson process. The continuous analogue of the Geometric.

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

If a user hasn't clicked in 5 minutes, the remaining time until click still has the same Exponential distribution — the wait "so far" gives you zero information about the wait "from here."

### 📌 Example 5: Time Between Purchases

**Q**: Users make purchases at a rate of 2 per month on average. What's the probability a user makes their next purchase within 2 weeks (0.5 months)?

```
X ~ Exponential(λ=2)

P(X ≤ 0.5) = 1 − e^(−2×0.5) = 1 − e^(−1) ≈ 1 − 0.368 = 0.632
```

**63.2% chance of a purchase within 2 weeks.** Note: this 63.2% ≈ 1−1/e appears naturally for any Exponential evaluated at its own mean — a useful mental shortcut.

---

### 6.4 Beta(α, β)

**Google in the wild**: Bayesian CTR estimation for ads. Your prior is Beta; your data updates it. This is exactly how Google's ad systems learn CTR estimates online.

Models **probabilities** — values constrained to [0,1]. The natural prior for a conversion rate or CTR.

```
f(x) = x^(α−1) · (1−x)^(β−1) / B(α,β)     for 0 ≤ x ≤ 1

E[X] = α / (α+β)
Var(X) = αβ / ((α+β)²(α+β+1))
```

**Interpretation of parameters**:
- α−1 = prior successes
- β−1 = prior failures
- α+β = strength of prior (total prior observations)

**Conjugate prior relationship**: If you observe k successes in n trials (Binomial likelihood), and your prior is Beta(α,β), the posterior is simply:

```
Beta(α+k, β+n−k)
```

No integration required — you just add counts to the parameters.

### 📌 Google-Level Q: Bayesian CTR

**Q**: You believe an ad's CTR is around 5% based on 100 historical impressions (5 clicks). You then run a new campaign with 200 impressions and get 14 clicks. What's your posterior belief about the CTR?

```
Prior: Beta(α=5, β=95)
New data: 14 successes, 186 failures
Posterior: Beta(5+14, 95+186) = Beta(19, 281)
Posterior mean = 19/300 ≈ 0.0633
```

**Posterior CTR estimate: ~6.3%** — pulled between the prior (5%) and the new data's raw rate (7%), weighted by relative sample sizes.

---

### 6.5 Log-Normal

**Google in the wild**: Revenue per user, session duration, file sizes — all right-skewed, all modeled as Log-Normal in practice.

If X ~ Normal(μ, σ²), then Y = eˣ is **Log-Normal**.

```
E[Y] = e^(μ + σ²/2)
Var(Y) = (e^(σ²) − 1) · e^(2μ+σ²)
```

**When to use**: Revenue, session length, file sizes — anything positive and right-skewed. A common mistake is modeling these as Normal when they're log-Normal, leading to poor estimates and nonsensical negative predictions.

### 📌 Google-Level Q: Revenue Distribution

**Q**: Why shouldn't you model user revenue as Normal? What should you use instead, and how does it change your A/B test analysis?

**Answer**: Revenue is:
- **Bounded below by 0** (can't have negative revenue)
- **Right-skewed** — a few users drive disproportionate revenue
- **Often log-Normal** in practice

Modeling it as Normal can predict negative revenue (meaningless), lets the mean get dragged by outliers, inflates variance, and violates the normality assumption that t-tests rely on at small sample sizes.

Better approaches:
1. **Log-transform** revenue and test on the log scale
2. Use the **Mann-Whitney U test** (non-parametric)
3. Use **CUPED** to reduce variance
4. **Cap outliers** and analyze separately

At Google, revenue experiments are often analyzed with log-transformed metrics or bootstrapped confidence intervals precisely because of this skew.

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
2. **Sums of independent RVs**: `M_{X+Y}(t) = M_X(t) · M_Y(t)`.
3. **Proving CLT**: the MGF of a standardized sum converges to the MGF of a Normal — this is the cleanest proof route for the Central Limit Theorem.

### Common MGFs

| Distribution | MGF M(t) |
|---|---|
| Bernoulli(p) | 1−p + p·eᵗ |
| Binomial(n,p) | (1−p + p·eᵗ)ⁿ |
| Poisson(λ) | exp(λ(eᵗ−1)) |
| Normal(μ,σ²) | exp(μt + σ²t²/2) |
| Exponential(λ) | λ/(λ−t) for t < λ |

**Key use**: To find E[X] and Var(X) quickly without integration.

**Example — Normal MGF**:
```
M'(t)  = (μ + σ²t)·exp(μt + σ²t²/2)
M'(0)  = μ            → E[X] = μ  ✓

M''(0) = σ² + μ²   → E[X²] = σ² + μ²
Var(X) = E[X²]−(E[X])² = σ²  ✓
```

---

## 8. Common Interview Questions (Extended)

### Problem A: Linearity of Expectation — Coupon Collector

*See the full derivation in Section 3 above.* The key insight: decompose the total wait into n independent-in-effect Geometric stages, sum their expectations. `E[T] = n·Hₙ ≈ n·ln(n)`.

**Wrong answer**: Trying to directly compute `E[T] = Σ t·P(T=t)` from the full joint distribution over n draws — this is combinatorially painful and error-prone under time pressure.

**Right approach**: Define Tᵢ as the wait during stage i (having i−1 items, waiting for a new one), note each is Geometric(p=(n−i+1)/n), and sum expectations via linearity.

**Signal**: "I can decompose this into independent Geometric stages and just sum their expectations — I don't need to track the joint state space."

---

### Problem B: Distribution Identification

**Q**: Google is monitoring the number of failed login attempts per user account per day. Some users have 0, most have very few, but a small number of accounts (likely under attack) have hundreds. Which distribution should you use, and why?

**Wrong answer**: "Poisson, since it's a count of events." This ignores the shape of the data — a small number of extreme accounts suggests massive overdispersion, which plain Poisson cannot represent (it forces mean=variance).

**Right approach**: Compute sample mean and variance first. If variance >> mean, the correct choice is a **Negative Binomial**, which explicitly allows variance to exceed the mean via its dispersion parameter r. This heterogeneity (most users behaving one way, a rare subset behaving very differently) is often best thought of as a mixture of Poissons with different λs, which mathematically produces a Negative Binomial.

**Signal**: "Before I commit to Poisson, I'd check whether variance equals mean in the data — if it doesn't, that tells me there's heterogeneity I need to model, likely with a Negative Binomial."

---

### Problem C: Variance Intuition — Why Var(X+Y) > Var(X) Alone

**Q**: Explain, without formulas, why combining two independent random sources of variation (say, two independent stages of a funnel, each with its own randomness) always produces *more* total variance than either stage alone — never less.

**Wrong answer**: "Sometimes variances could cancel out, like they do for the mean when things are negative." This confuses expectation (which can involve cancellation because of sign) with variance (which is a sum of *squared* deviations — always non-negative contributions).

**Right approach**: Variance is built from squared terms, so every independent source of randomness can only ever add non-negative uncertainty to the total. There is no way for one independent random variable's spread to "cancel" another's spread, because you never subtract variances — even when the underlying random variables are subtracted from each other. The direction of combination (add or subtract the RVs) is irrelevant to the variance; only independence and each variable's own variance matter.

**Signal**: "Variance combination is fundamentally different from expectation combination — sign doesn't matter for variance because everything is squared, so independent variances only ever add."

---

### Problem D: Poisson Approximation to Binomial

**Q**: When should you approximate a Binomial(n,p) with a Poisson(λ=np) instead of computing the Binomial directly, and why does this matter operationally at Google's scale?

**Wrong answer**: "Always, since Poisson is simpler to compute." This ignores that the approximation only holds well under specific conditions.

**Right approach**: The Poisson approximation is good when n is large and p is small, with np staying moderate — the standard rule of thumb is n > 20 and p < 0.05. This matters operationally because at Google's scale, many events genuinely fit this regime (rare failures across millions of servers, rare fraud across billions of transactions), and using Poisson instead of exact Binomial avoids numerically unstable factorial computations for huge n, while still giving an accurate approximation.

**Signal**: "I'd check that p is small and n is large with np roughly constant before defaulting to the Poisson approximation — otherwise I'd just compute the Binomial directly, especially now that computing power makes the 'simplicity' argument less relevant than the accuracy argument."

---

### Problem E: Beta Posterior Update

**Q**: A new feature's opt-in rate is believed to be around 20%, based on a weak prior equivalent to 10 historical observations. After launching to 500 users, 80 opt in. What's your updated belief about the true opt-in rate, and how confident should you be?

**Wrong answer**: Just reporting the raw observed rate, 80/500 = 16%, and ignoring the prior entirely, or ignoring the new data and sticking with the 20% prior.

**Right approach**: Translate the "weak prior equivalent to 10 observations at 20%" into `Beta(α=2, β=8)` (2 prior successes, 8 prior failures, summing to 10 total prior observations). Update with the new data (80 successes, 420 failures):

```
Posterior: Beta(2+80, 8+420) = Beta(82, 428)
Posterior mean = 82/510 ≈ 0.1608
```

**Posterior opt-in rate ≈ 16.1%** — very close to the raw observed rate, because the new data (500 observations) overwhelms the weak prior (10 observations). The posterior variance also shrinks substantially as more data comes in, which you could quantify with the Beta variance formula to state a credible interval.

**Signal**: "The weight of the prior versus the data is literally just a comparison of observation counts — with a weak prior and a large new sample, the posterior converges toward the empirical rate, which is exactly what I'd expect and can show mathematically."

---

## 9. Traps — Expanded

| Trap | Why It's Wrong | What To Say Instead |
|------|---------------|---------------------|
| "PDF f(x) is the probability at x" | f(x) is a density, not a probability — it can exceed 1 | "Probability is the area under the curve, not the height of the curve" |
| "Var(X−Y) = Var(X)−Var(Y)" | Subtracting two independent variables still adds their uncertainty; variances never subtract | "Var(X−Y) = Var(X)+Var(Y) when X, Y are independent" |
| "Poisson is for any count data" | Poisson strictly requires mean = variance | "Check for overdispersion first — if variance >> mean, use Negative Binomial instead" |
| "Normal is always the right choice for aggregated metrics" | Revenue, session time, and many counts are naturally skewed | "Log-transform the metric, or use robust/non-parametric statistics" |
| "Memorylessness works for all distributions" | Only Geometric and Exponential have this property | "Most distributions 'remember' — only Geometric and Exponential forget their past" |
| "E[XY] = E[X]E[Y] always" | This only holds under independence | "I need independence to factor the expectation of a product — otherwise I need the covariance term" |
| "A larger sample always reduces overdispersion" | Overdispersion is a property of the data-generating process, not the sample size | "More data gives a better *estimate* of the dispersion, but doesn't change the underlying process" |
| "Chebyshev gives the exact probability" | It's a worst-case bound, not an exact value, and is often very loose | "Chebyshev guarantees a lower bound — the true probability is usually much higher, e.g. under Normality" |

---

## 10. Bridge to Module 3

Everything covered here — expectation, variance, distributions — is the foundation for Module 3. There, you'll see how **multiple random variables interact** (joint distributions, covariance, correlation), why **sums of variables** behave the way they do (the Central Limit Theorem), and how **sample averages** converge to population values (the Law of Large Numbers).

The key bridge: if X₁,...,Xₙ are iid with mean μ and variance σ², their sum has mean nμ and variance nσ² — and as n grows, that sum becomes approximately Normal. That's the Central Limit Theorem, and it's the mathematical fact that makes A/B testing at Google possible at all.

---

## Q&A Section (Original Six, Retained)

### Q1 (Warm-up)
**Q**: X ~ Binomial(10, 0.4). What are E[X] and Var(X)?
```
E[X] = np = 10 × 0.4 = 4
Var(X) = np(1−p) = 10 × 0.4 × 0.6 = 2.4
SD(X) = √2.4 ≈ 1.55
```

### Q2 (PMF Reasoning)
**Q**: You roll two dice. X = the maximum of the two rolls. What is P(X = 4)?
```
P(both ≤ 4) = (4/6)² = 16/36
P(both ≤ 3) = (3/6)² = 9/36
P(X = 4) = 16/36 − 9/36 = 7/36 ≈ 0.194
```

### Q3 (Google-Level: Poisson)
**Q**: A Google data center has two independent server clusters. Cluster A fails λ_A = 0.5 times/day on average. Cluster B fails λ_B = 0.3 times/day. What is the expected total failures per day and the probability of zero total failures?
```
Total X ~ Poisson(0.5 + 0.3) = Poisson(0.8)
E[X] = 0.8 failures/day
P(X=0) = e^(−0.8) ≈ 0.449
```
**~45% chance of a failure-free day.**

### Q4 (Normal Standardization)
**Q**: Session durations ~ N(4.5 min, 1.2²). What % of sessions are between 3 and 6 minutes?
```
P(3 < X < 6) = P(−1.25 < Z < 1.25) = 2·Φ(1.25) − 1 ≈ 2(0.8944) − 1 = 0.7888
```
**~78.9% of sessions fall between 3 and 6 minutes.**

### Q5 (Google-Level: Geometric / Memorylessness)
**Q**: Each time a user sees a Google ad, they click with probability 0.08. A user has seen the ad 5 times without clicking. What's the expected total number of impressions until their first click, starting from now?
```
E[additional impressions] = 1/p = 1/0.08 = 12.5
```
By memorylessness, the 5 past failures are irrelevant — they still need **12.5 more impressions on average**, starting now.

### Q6 (Google-Level: Distribution Choice)
**Q**: You're modeling the number of YouTube comments per video. You assume Poisson(λ=50). But you notice the sample variance is 2,400 — much larger than 50. What's wrong and what should you use?
**A**: Poisson requires mean = variance. Variance of 2,400 >> mean of 50 signals overdispersion. Poisson is too restrictive. The better choice is **Negative Binomial**, which allows variance > mean via a dispersion parameter r (`Variance = μ + μ²/r`). As r → ∞, Negative Binomial → Poisson. Overdispersion here likely reflects heterogeneity — most videos get few comments, some go viral — which is exactly a mixture-of-Poissons scenario, mathematically equivalent to Negative Binomial.

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
