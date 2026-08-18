# Day 11 — Continuous Random Variables: PDF, CDF, Uniform & Exponential
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 5 (Sections 5.1–5.4)
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Continuous Distributions Dominate ML

Almost every quantity in ML is continuous:

| ML Quantity | Distribution |
|---|---|
| Model weights | Normal (after initialization) |
| Loss value | Continuous, often right-skewed |
| Learning rate | Uniform (log-scale search) |
| Dropout probability | Uniform(0,1) |
| Time between events | Exponential |
| Predicted probability | Uniform(0,1) under null model |
| Pixel intensity (normalized) | Uniform(0,1) |
| Activation values (pre-ReLU) | Approximately Normal |
| p-values under H₀ | Uniform(0,1) — critical fact |

The transition from discrete to continuous requires one key conceptual shift: **individual points have zero probability.** Probability lives in intervals, not points.

---

## 2. From Discrete to Continuous — The Key Shift

### Discrete: PMF sums to 1
```
Σₓ P(X=x) = 1
P(X=x) > 0 for values in support
```

### Continuous: PDF integrates to 1
```
∫₋∞^∞ f(x) dx = 1
P(X=x) = 0 for any single point x
```

**Why P(X = x) = 0 for continuous RVs?**

The probability of hitting any exact point on a continuous line is zero — like the probability of a dart hitting an exact mathematical point (zero area). This seems strange but is mathematically consistent.

**Consequence:**
```
P(a < X < b) = P(a ≤ X < b) = P(a < X ≤ b) = P(a ≤ X ≤ b)
```
Endpoints don't matter for continuous RVs.

---

## 3. Probability Density Function (PDF)

> **Definition:** A continuous random variable X has **PDF** f(x) if:
> ```
> P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx    for all a ≤ b
> ```

### PDF Requirements

```
1. f(x) ≥ 0          for all x        [non-negative]
2. ∫₋∞^∞ f(x) dx = 1                 [normalizes to 1]
```

### Critical: PDF Values Can Exceed 1

Unlike PMF values, PDF values are **not** probabilities — they are **densities**.

```
f(x) can be > 1      [perfectly valid]
P(a ≤ X ≤ b) ≤ 1    [always]
```

**Intuition:** f(x) is like mass per unit length. A very thin, tall spike can have f(x) >> 1 as long as the area under it is ≤ 1.

---

## 4. Cumulative Distribution Function (CDF)

For continuous RVs:
```
F(x) = P(X ≤ x) = ∫₋∞ˣ f(t) dt
```

### Fundamental Theorem of Calculus Applied

```
f(x) = F'(x) = dF/dx        [PDF is derivative of CDF]
F(x) = ∫₋∞ˣ f(t) dt        [CDF is integral of PDF]
```

This is the most-used relationship in continuous probability.

### CDF Properties (same as discrete)

```
F(−∞) = 0,    F(+∞) = 1
F is non-decreasing
F is continuous (for continuous RVs)

P(a < X ≤ b) = F(b) − F(a)
P(X > a)      = 1 − F(a)
```

---

## 5. Expectation and Variance for Continuous RVs

Replace sums with integrals:

```
E[X]   = ∫₋∞^∞ x · f(x) dx

E[g(X)] = ∫₋∞^∞ g(x) · f(x) dx    [LOTUS for continuous]

Var(X) = E[X²] − (E[X])²
       = ∫₋∞^∞ x² f(x) dx − (∫₋∞^∞ x f(x) dx)²
```

All properties from Day 8 (linearity, Var(aX+b) = a²Var(X), etc.) hold identically.

---

## 6. Uniform Distribution

> **Definition:** X ~ Uniform(a, b) if X is equally likely to be any value in [a, b].

```
         ⎧ 1/(b−a)    if a ≤ x ≤ b
f(x) =   ⎨
         ⎩ 0          otherwise
```

### CDF

```
         ⎧ 0           x < a
F(x) =   ⎨ (x−a)/(b−a) a ≤ x ≤ b
         ⎩ 1           x > b
```

### Parameters

```
E[X]   = (a+b)/2          [midpoint]
Var(X) = (b−a)²/12
SD(X)  = (b−a)/√12 = (b−a)/2√3
```

**Proof of E[X]:**
```
E[X] = ∫ₐᵇ x · 1/(b−a) dx = 1/(b−a) · [x²/2]ₐᵇ
     = 1/(b−a) · (b²−a²)/2 = (b+a)(b−a)/[2(b−a)] = (a+b)/2  ∎
```

**Proof of Var(X):**
```
E[X²] = ∫ₐᵇ x² · 1/(b−a) dx = (b³−a³)/[3(b−a)] = (a²+ab+b²)/3

Var(X) = E[X²] − (E[X])² = (a²+ab+b²)/3 − (a+b)²/4
       = (b−a)²/12  ∎
```

### Special Case: Uniform(0,1)

```
f(x) = 1      0 ≤ x ≤ 1
E[X] = 0.5
Var(X) = 1/12 ≈ 0.0833
```

**ML uses of Uniform(0,1):**
- Random number generation (foundation of all simulation)
- p-values under H₀ are Uniform(0,1)
- Dropout mask generation: sample u ~ Uniform(0,1), drop if u < p
- Probability calibration: well-calibrated model outputs ~ Uniform(0,1) on unlabeled data
- Inverse CDF method: generate any distribution from Uniform(0,1)

### Inverse CDF (Quantile) Method

> If U ~ Uniform(0,1) and F is a CDF, then X = F⁻¹(U) has CDF F.

**ML use:** Generate samples from any distribution using uniform random numbers. This is how numpy generates non-uniform random samples internally.

---

## 7. Exponential Distribution

> **Definition:** X ~ Exponential(λ) if:
> ```
> f(x) = λe^(−λx)    for x ≥ 0
> f(x) = 0            for x < 0
> ```

where λ > 0 is the **rate parameter**.

### CDF

```
F(x) = 1 − e^(−λx)    for x ≥ 0
```

**Derivation:**
```
F(x) = ∫₀ˣ λe^(−λt) dt = [−e^(−λt)]₀ˣ = 1 − e^(−λx)  ✓
```

### Parameters

```
E[X]   = 1/λ
Var(X) = 1/λ²
SD(X)  = 1/λ
```

Note: Mean = SD = 1/λ for Exponential. Coefficient of variation = SD/Mean = 1 always.

**Proof of E[X]:**
```
E[X] = ∫₀^∞ x · λe^(−λx) dx

Integration by parts: u=x, dv=λe^(−λx)dx → du=dx, v=−e^(−λx)

= [−xe^(−λx)]₀^∞ + ∫₀^∞ e^(−λx) dx
= 0 + [−e^(−λx)/λ]₀^∞
= 0 + 1/λ = 1/λ  ∎
```

### The Memoryless Property — Exponential Version

> **The Exponential is the ONLY continuous memoryless distribution:**
> ```
> P(X > s+t | X > s) = P(X > t)    for all s, t ≥ 0
> ```

**Proof:**
```
P(X > s+t | X > s) = P(X > s+t) / P(X > s)
                   = e^(−λ(s+t)) / e^(−λs)
                   = e^(−λt)
                   = P(X > t)  ∎
```

**Intuition:** If you've already waited s minutes for a bus, the remaining wait has the same distribution as if you just arrived. Past waiting time gives no information.

### Connection to Poisson and Geometric

```
Exponential(λ) = continuous analog of Geometric(p)
Poisson process with rate λ:
    → Counts: Poisson(λt)
    → Inter-arrival times: Exponential(λ)
```

### Alternative Parameterization

Some books use scale parameter β = 1/λ:
```
f(x) = (1/β)e^(−x/β)
E[X] = β
```

Be careful — always check which parameterization is being used!

---

## 8. Comparing Uniform and Exponential

| Property | Uniform(a,b) | Exponential(λ) |
|---|---|---|
| Support | [a, b] — bounded | [0, ∞) — unbounded |
| Shape | Flat (constant density) | Decreasing exponentially |
| Memoryless? | No | Yes |
| Mean | (a+b)/2 | 1/λ |
| Variance | (b−a)²/12 | 1/λ² |
| ML use | Random init, p-values | Waiting times, survival |

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — Uniform: Hyperparameter Search

**Problem:** You do random search over learning rate η ~ Uniform(0.0001, 0.1).

**(a)** What is the PDF of η?
**(b)** P(η < 0.01) — probability of selecting a small learning rate
**(c)** E[η] and SD[η]
**(d)** Why is log-uniform search preferred over uniform?

**Solution:**

**(a)** b−a = 0.1 − 0.0001 = 0.0999
```
f(η) = 1/0.0999 ≈ 10.01    for 0.0001 ≤ η ≤ 0.1
```

Note: f(η) ≈ 10 > 1 — valid since it's a density, not a probability.

**(b)**
```
P(η < 0.01) = (0.01 − 0.0001)/0.0999 = 0.0099/0.0999 ≈ 0.099 ≈ 9.9%
```

Only ~10% of uniform random samples give η < 0.01, even though values in (0.001, 0.01) are often the best range.

**(c)**
```
E[η] = (0.0001 + 0.1)/2 ≈ 0.0501
Var(η) = (0.0999)²/12 ≈ 0.000832
SD(η) ≈ 0.0289
```

**(d)** With uniform search, 90% of samples have η > 0.01 — you waste most budget on large learning rates. With **log-uniform** (sample log(η) ~ Uniform(log(0.0001), log(0.1))):
- Equal probability in each order of magnitude: [0.0001, 0.001], [0.001, 0.01], [0.01, 0.1]
- Better coverage of the useful range

**This is why random search outperforms grid search for hyperparameters** — and why log-scale is used for rates, regularization strengths, etc.

---

### 🔢 Numerical 2 — Uniform CDF and Inverse CDF

**Problem:** X ~ Uniform(0, 1). 

**(a)** Find P(0.3 ≤ X ≤ 0.7)
**(b)** Find the value x₀ such that P(X ≤ x₀) = 0.95 (the 95th percentile)
**(c)** Verify that F(x) = x for Uniform(0,1)
**(d)** Use inverse CDF to explain how to generate Exponential(λ) samples from Uniform(0,1)

**Solution:**

**(a)**
```
P(0.3 ≤ X ≤ 0.7) = F(0.7) − F(0.3) = 0.7 − 0.3 = 0.4
```

**(b)**
```
F(x₀) = x₀ = 0.95  →  x₀ = 0.95
```

**(c)**
```
F(x) = ∫₀ˣ 1 dt = x    for 0 ≤ x ≤ 1  ✓
```

**(d)** For Exponential(λ): F(x) = 1 − e^(−λx)

Inverse: F⁻¹(u) = −log(1−u)/λ

So: if U ~ Uniform(0,1), then X = −log(1−U)/λ ~ Exponential(λ)

Since 1−U ~ Uniform(0,1) as well: X = −log(U)/λ ~ Exponential(λ)

**Algorithm:**
```python
# Generate Exponential(λ) from Uniform(0,1):
u = random.uniform(0, 1)
x = -np.log(u) / lambda_param
```

This is the **inverse CDF method** — used internally by numpy and scipy.

---

### 🔢 Numerical 3 — Exponential: Server Response Time

**Problem:** Server response time T ~ Exponential(λ=2) (rate = 2 responses/second, mean = 0.5 sec).

**(a)** P(response within 1 second)
**(b)** P(response takes more than 2 seconds)
**(c)** Median response time
**(d)** Given server hasn't responded in 1 second, P(responds within next 0.5 seconds)

**Solution:**

T ~ Exponential(λ=2): F(t) = 1 − e^(−2t)

**(a)**
```
P(T ≤ 1) = 1 − e^(−2×1) = 1 − e^(−2) = 1 − 0.1353 = 0.8647
```
86.5% of requests respond within 1 second.

**(b)**
```
P(T > 2) = e^(−2×2) = e^(−4) = 0.0183
```
Only 1.8% of requests take more than 2 seconds.

**(c)** Median: solve F(m) = 0.5:
```
1 − e^(−2m) = 0.5
e^(−2m) = 0.5
−2m = ln(0.5) = −0.693
m = 0.347 seconds
```

Median (0.347s) < Mean (0.5s) — Exponential is right-skewed. The median is always ln(2)/λ.

**(d)** **Memoryless property:**
```
P(T ≤ 1.5 | T > 1) = P(T ≤ 0.5) = 1 − e^(−2×0.5) = 1 − e^(−1) = 0.6321
```

The conditional probability equals the unconditional probability of responding within 0.5 seconds — as if the clock reset.

**ML insight:** This is used in **survival analysis** for time-to-event models. The hazard rate of Exponential is constant (λ at all times) — called the **constant hazard** property. Models like Cox Proportional Hazards extend this to non-constant hazard rates.

---

### 🔢 Numerical 4 — PDF Validation and Computation

**Problem:** Suppose f(x) = cx² for 0 ≤ x ≤ 2, 0 otherwise.

**(a)** Find c so f(x) is a valid PDF.
**(b)** Find the CDF F(x).
**(c)** Find E[X].
**(d)** Find P(1 ≤ X ≤ 1.5).
**(e)** Find the median.

**Solution:**

**(a)** Normalization:
```
∫₀² cx² dx = c[x³/3]₀² = c·8/3 = 1
c = 3/8
```

**(b)** CDF for 0 ≤ x ≤ 2:
```
F(x) = ∫₀ˣ (3/8)t² dt = (3/8)·[t³/3]₀ˣ = x³/8
```
So F(x) = 0 for x<0, x³/8 for 0≤x≤2, 1 for x>2.

**(c)**
```
E[X] = ∫₀² x·(3/8)x² dx = (3/8)∫₀² x³ dx = (3/8)·[x⁴/4]₀² = (3/8)·4 = 3/2 = 1.5
```

**(d)**
```
P(1 ≤ X ≤ 1.5) = F(1.5) − F(1) = (1.5)³/8 − 1³/8
               = 3.375/8 − 1/8 = 2.375/8 = 0.297
```

**(e)** Median: solve F(m) = 0.5:
```
m³/8 = 0.5  →  m³ = 4  →  m = 4^(1/3) ≈ 1.587
```

---

### 🔢 Numerical 5 — Exponential: Time to Model Convergence

**Problem:** Training loss decreases below threshold after T hours, where T ~ Exponential(λ = 0.5) (average 2 hours).

**(a)** P(model converges within 1 hour)
**(b)** P(model takes between 2 and 4 hours)
**(c)** If you've been training for 3 hours without convergence, what's the expected additional time needed?
**(d)** 90th percentile of convergence time

**Solution:**

T ~ Exponential(0.5): F(t) = 1 − e^(−0.5t), E[T] = 2 hours

**(a)**
```
P(T ≤ 1) = 1 − e^(−0.5) = 1 − 0.6065 = 0.3935
```
Only 39.4% chance of converging within 1 hour.

**(b)**
```
P(2 < T ≤ 4) = F(4) − F(2)
             = (1−e^(−2)) − (1−e^(−1))
             = e^(−1) − e^(−2)
             = 0.3679 − 0.1353 = 0.2326
```

**(c)** **Memoryless:** Given T > 3, the remaining time ~ Exponential(0.5).
```
E[remaining time | T > 3] = E[T] = 1/0.5 = 2 hours
```
After 3 hours of waiting, you still expect 2 more hours — same as the start!

**(d)** 90th percentile: solve F(t₀.₉) = 0.9:
```
1 − e^(−0.5t) = 0.9
e^(−0.5t) = 0.1
−0.5t = ln(0.1) = −2.303
t = 4.605 hours ≈ 4 hours 36 minutes
```

**General formula for p-th percentile of Exponential(λ):**
```
tₚ = −ln(1−p)/λ
```

---

### 🔢 Numerical 6 — Uniform: p-values Under H₀

**Problem:** Under the null hypothesis H₀, p-values are Uniform(0,1). You run 100 independent hypothesis tests, all under H₀ (no real effect).

**(a)** Expected number of p-values below 0.05
**(b)** P(at least one p-value below 0.05) — the multiple testing problem
**(c)** Bonferroni correction: what threshold α* should you use for each test to maintain family-wise error rate of 5%?

**Solution:**

Each p-value P_i ~ Uniform(0,1) under H₀.
P(P_i < 0.05) = 0.05 (since CDF of Uniform(0,1) is F(x)=x).

**(a)** Expected false positives:
```
E[false positives] = 100 × 0.05 = 5
```
Even with no real effects, you expect 5 "significant" results!

**(b)**
```
P(at least one p < 0.05) = 1 − P(all p ≥ 0.05)
                         = 1 − (0.95)^100
                         = 1 − 0.00592 = 0.9941
```
**99.4% chance of at least one false positive** — almost certain! This is the **multiple testing problem**.

**(c)** Bonferroni correction: use α* = α/m = 0.05/100 = **0.0005**

P(any false positive with threshold α*) ≤ m × α* = 100 × 0.0005 = 0.05 ✓

**ML insight:** When evaluating models on many metrics or running many ablation studies, the multiple testing problem means some "improvements" are false positives. This is a major source of unreproducible ML results. Always correct for multiple comparisons.

---

### 🔢 Numerical 7 — Mixed: Connecting Uniform, Exponential, Poisson

**Problem:** A streaming ML system:
- Requests arrive as Poisson(λ=4/min)
- Each request takes T ~ Exponential(μ=6/min) to process
- Processing times are i.i.d.

**(a)** Expected inter-arrival time between requests
**(b)** P(processing time > inter-arrival time) — probability of queue buildup for one request
**(c)** Expected number of requests arriving during one processing period
**(d)** What condition on λ and μ ensures the queue is stable (doesn't grow forever)?

**Solution:**

**(a)** Inter-arrival time A ~ Exponential(4):
```
E[A] = 1/4 = 0.25 minutes = 15 seconds
```

**(b)**
```
P(T > A) where T ~ Exp(6), A ~ Exp(4), independent

P(T > A) = ∫₀^∞ P(T > a) · f_A(a) da
         = ∫₀^∞ e^(−6a) · 4e^(−4a) da
         = 4∫₀^∞ e^(−10a) da
         = 4 · [−e^(−10a)/10]₀^∞
         = 4/10 = 0.4
```

**General result:** If T~Exp(μ) and A~Exp(λ), P(T>A) = λ/(λ+μ) = 4/(4+6) = **0.4**

40% of requests cause queue buildup.

**(c)** Requests during processing time T ~ Poisson(λT) = Poisson(4T):
```
E[requests during T] = E[4T] = 4 · E[T] = 4 · (1/6) = 2/3
```

On average, 0.67 new requests arrive during each processing period.

**(d)** Queue stability: service rate must exceed arrival rate:
```
μ > λ   →   6 > 4  ✓
```

The system is stable (traffic intensity ρ = λ/μ = 4/6 = 0.667 < 1).

**ML insight:** This is **queueing theory** — the M/M/1 queue. It models ML inference servers: if inference takes longer than inter-request time on average, the queue grows without bound and latency explodes. The condition μ > λ is why serving teams monitor QPS (queries per second) vs throughput capacity.

---

## 10. Common Interview Questions

| Question | Key Idea |
|---|---|
| "Why is P(X=x)=0 for continuous RVs?" | Probability = area, single point has zero area |
| "Can a PDF value exceed 1?" | Yes — it's a density, not a probability |
| "What is the relationship between PDF and CDF?" | f(x) = F'(x); F(x) = ∫f(t)dt |
| "What distribution models waiting times?" | Exponential — memoryless, inter-arrival of Poisson |
| "What is the memoryless property of Exponential?" | P(X>s+t\|X>s) = P(X>t) — past doesn't matter |
| "Under H₀, what distribution do p-values follow?" | Uniform(0,1) |
| "How do you generate Exponential samples from Uniform?" | X = −log(U)/λ (inverse CDF method) |
| "What is the Bonferroni correction?" | Use α/m threshold for m tests to control family-wise error |
| "When does a queue become unstable?" | When arrival rate λ > service rate μ |

---

## 11. Key Formulas — Cheat Sheet for Day 11

```
Continuous RV:
    P(a≤X≤b) = ∫ₐᵇ f(x)dx
    P(X=x) = 0
    f(x) ≥ 0,  ∫f(x)dx = 1

PDF ↔ CDF:
    F(x) = ∫₋∞ˣ f(t)dt
    f(x) = F'(x)
    P(a<X≤b) = F(b) − F(a)

Expectation/Variance:
    E[X] = ∫x·f(x)dx
    E[g(X)] = ∫g(x)·f(x)dx    [LOTUS]
    Var(X) = E[X²] − (E[X])²

Uniform(a,b):
    f(x) = 1/(b−a)        [a≤x≤b]
    F(x) = (x−a)/(b−a)
    E[X] = (a+b)/2
    Var(X) = (b−a)²/12

Uniform(0,1) special:
    F(x) = x
    p-values ~ Uniform(0,1) under H₀

Exponential(λ):
    f(x) = λe^(−λx)       [x≥0]
    F(x) = 1 − e^(−λx)
    E[X] = 1/λ
    Var(X) = 1/λ²
    Median = ln(2)/λ
    p-th percentile = −ln(1−p)/λ
    Memoryless: P(X>s+t|X>s) = P(X>t)

Inverse CDF (generate Exp from Uniform):
    If U~Uniform(0,1): X = −log(U)/λ ~ Exponential(λ)

P(Exp(μ) > Exp(λ)) = λ/(λ+μ)   [independent]

Queue stability: λ < μ  (traffic intensity ρ = λ/μ < 1)

Bonferroni correction: α* = α/m for m tests
```

---

## 12. Practice Problems (Solve Before Day 12)

1. X ~ Uniform(−1, 1). Find E[X], Var(X), P(|X| < 0.5), and the PDF and CDF. This distribution models weight initialization in some neural networks — why is mean 0 desirable?

2. T ~ Exponential(λ=3). Find:
   - P(T > 1)
   - P(0.5 < T < 1.5)
   - The 75th percentile
   - E[T²] using LOTUS

3. **Prove** the memoryless property of the Exponential distribution from the CDF.

4. f(x) = ke^(−2x) for x ≥ 0. Find k, verify it's a valid PDF, identify the distribution, and compute E[X] and Var(X).

5. *(Interview-level)* You run 20 independent A/B tests simultaneously. Under H₀ for all tests, what is the probability that at least 2 tests show p < 0.05? *(Hint: number of significant results ~ Binomial(20, 0.05) under H₀.)*

6. *(Hard)* An ML model serves requests. Inference time T ~ Exponential(μ=10/sec). Requests arrive at λ=8/sec.
   - What fraction of time is the server busy? (Utilization = λ/μ)
   - What is P(next request arrives before current inference finishes)?
   - If inference time doubles (μ=5/sec), is the system still stable?

---

## 13. Looking Ahead

**Day 12** — **The Normal (Gaussian) Distribution.** The most important distribution in all of statistics and ML. We derive its properties, understand why it appears everywhere (Central Limit Theorem preview), and connect it to maximum likelihood estimation, regularization, and neural network weight initialization.

---
*End of Day 11 | Next: Day 12 — The Normal / Gaussian Distribution*
