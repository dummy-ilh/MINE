# Day 21 — Sampling Distributions: t, Chi-Squared & F
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Blitzstein & Hwang, *Introduction to Probability* — Chapter 10; Casella & Berger Chapter 5
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why These Three Distributions Matter in ML

The Normal distribution (Day 12) assumes you know σ. In reality, **you never know σ** — you estimate it from data. This one change propagates into three new distributions that underlie all of classical inference.

| Distribution | When It Appears | ML/DS Application |
|---|---|---|
| **t(ν)** | Unknown σ, small n | Confidence intervals, one/two-sample tests |
| **χ²(ν)** | Sum of squared Normals | Variance estimation, goodness-of-fit tests |
| **F(ν₁,ν₂)** | Ratio of variances | ANOVA, regression F-test, model comparison |

Every time you run a t-test, compute a p-value for a regression coefficient, or compare two model variances, you're using these distributions.

---

## 2. The Sample Variance and Why It Matters

Before the distributions, we need the **sample variance**:

```
S² = 1/(n−1) · Σᵢ(Xᵢ − X̄)²
```

**Why n−1 instead of n?**

- With n in denominator: E[S²_biased] = (n−1)/n · σ² ≠ σ² (biased)
- With n−1 in denominator: E[S²] = σ² (unbiased)

**Degrees of freedom:** We use n−1 because we "use up" 1 degree of freedom estimating the mean. After fixing X̄, only n−1 of the n deviations (Xᵢ−X̄) are free.

```
The constraint: Σᵢ(Xᵢ − X̄) = 0
```

This forces the last deviation, given the other n−1. Hence n−1 free parameters.

---

## 3. Chi-Squared Distribution

> **Definition:** If Z₁, Z₂, ..., Zₖ are i.i.d. N(0,1), then:
> ```
> X = Z₁² + Z₂² + ... + Zₖ² ~ χ²(k)
> ```
> X follows a **chi-squared distribution with k degrees of freedom**.

### PDF and Parameters

```
f(x) = x^(k/2−1) · e^(−x/2) / (2^(k/2) · Γ(k/2))    for x > 0

E[X] = k
Var(X) = 2k
Mode = max(k−2, 0)
```

### Key Properties

```
1. χ²(k) = Gamma(k/2, 1/2)    [special case of Gamma]
2. If X~χ²(m) and Y~χ²(n) independent: X+Y~χ²(m+n)  [additive]
3. As k→∞: χ²(k) ≈ N(k, 2k)   [CLT applies]
4. χ²(1) = Z²  where Z~N(0,1)  [squared standard Normal]
5. χ²(2) = Exponential(1/2)    [special case]
```

### Critical Result: Sample Variance Distribution

> **Theorem:** If X₁,...,Xₙ ~ i.i.d. N(μ,σ²), then:
> ```
> (n−1)S²/σ² ~ χ²(n−1)
> ```
> and X̄ and S² are **independent**.

**Proof sketch:**
```
Σᵢ(Xᵢ−X̄)²/σ² = Σᵢ[(Xᵢ−μ)/σ]² − n[(X̄−μ)/σ]²
                = χ²(n) − χ²(1) = χ²(n−1)
```

(Each (Xᵢ−μ)/σ is N(0,1), but (Xᵢ−X̄)/σ only has n−1 free parameters due to the constraint Σ(Xᵢ−X̄)=0.)

**ML use:** Confidence intervals for σ², hypothesis tests about variance, goodness-of-fit tests (Pearson's χ²).

---

## 4. The t-Distribution

### Motivation

You want to test whether μ=μ₀, but σ is unknown. You estimate it with S.

If σ were known: Z = (X̄−μ)/(σ/√n) ~ N(0,1)

Replace σ with S: T = (X̄−μ)/(S/√n) ← this is NOT N(0,1). It has heavier tails.

> **Definition:** If Z ~ N(0,1) and V ~ χ²(ν) are **independent**, then:
> ```
>          Z
> T = ————————— ~ t(ν)
>       √(V/ν)
> ```
> T follows a **t-distribution with ν degrees of freedom**.

### Connection to Sampling

For X₁,...,Xₙ ~ i.i.d. N(μ,σ²):
```
T = (X̄ − μ) / (S/√n) ~ t(n−1)
```

Here: Z = (X̄−μ)/(σ/√n) ~ N(0,1), V = (n−1)S²/σ² ~ χ²(n−1), independent.

Substituting: T = Z/√(V/(n−1)) = t(n−1) ✓

### PDF and Parameters

```
f(t) = Γ((ν+1)/2) / [√(νπ)·Γ(ν/2)] · (1 + t²/ν)^(−(ν+1)/2)

E[T] = 0              for ν > 1
Var(T) = ν/(ν−2)      for ν > 2
```

### Key Properties

```
1. Symmetric around 0 (like Normal)
2. Heavier tails than Normal (more probability in extremes)
3. As ν→∞: t(ν) → N(0,1)    [normal is limit]
4. t(1) = Cauchy(0,1)        [heaviest-tailed case]
5. Quantiles: t_{α,ν} > z_α  [always wider than Normal]
```

### t vs Normal: Critical Values

| Significance | N(0,1) | t(5) | t(10) | t(30) | t(∞)=N(0,1) |
|---|---|---|---|---|---|
| 90% (one-sided) | 1.282 | 2.015 | 1.812 | 1.697 | 1.282 |
| 95% (one-sided) | 1.645 | 2.571 | 2.228 | 2.042 | 1.645 |
| 97.5% (two-sided 95%) | 1.960 | 2.571 | 2.228 | 2.042 | 1.960 |
| 99.5% (two-sided 99%) | 2.576 | 4.032 | 3.169 | 2.750 | 2.576 |

**Key takeaway:** With small n, t-critical values are larger than z-critical values — confidence intervals are wider. The extra uncertainty from estimating σ is captured in the heavier tails.

---

## 5. The F-Distribution

> **Definition:** If U ~ χ²(ν₁) and V ~ χ²(ν₂) are **independent**, then:
> ```
>      U/ν₁
> F = ——————— ~ F(ν₁, ν₂)
>      V/ν₂
> ```
> F follows an **F-distribution with ν₁ and ν₂ degrees of freedom**.

### Parameters

```
E[F] = ν₂/(ν₂−2)         for ν₂ > 2
Var[F] = 2ν₂²(ν₁+ν₂−2)/[ν₁(ν₂−2)²(ν₂−4)]   for ν₂ > 4
```

### Key Properties

```
1. F(ν₁,ν₂) > 0  always    [ratio of non-negative quantities]
2. 1/F(ν₁,ν₂) ~ F(ν₂,ν₁)  [reciprocal flips df]
3. [t(ν)]² ~ F(1,ν)         [squared t is F]
4. As ν₁,ν₂→∞: F → 1       [ratio of equals = 1]
5. F(1,ν) = [t(ν)]²
```

### Connection to Variance Comparison

For two independent Normal samples:
```
F = S₁²/S₂² ~ F(n₁−1, n₂−1)    under H₀: σ₁²=σ₂²
```

This is the **F-test for equality of variances**.

### F-test in Regression

For a regression model with p predictors and n observations:

```
F = (R²/p) / ((1−R²)/(n−p−1)) ~ F(p, n−p−1)    under H₀: all βᵢ=0
```

This tests whether the regression as a whole is significant.

---

## 6. Relationships Between the Three Distributions

```
Z ~ N(0,1)
│
├── Z² ~ χ²(1)
│        │
│        └── Σ Zᵢ² ~ χ²(k)      [sum of k squared Normals]
│                  │
│                  └── (n−1)S²/σ² ~ χ²(n−1)    [sample variance]
│
└── Z/√(χ²(ν)/ν) ~ t(ν)          [t from Normal + chi-squared]
         │
         └── [t(ν)]² ~ F(1,ν)    [squared t is F]
                   │
                   └── χ²(ν₁)/ν₁ / χ²(ν₂)/ν₂ ~ F(ν₁,ν₂)  [general F]
```

All three distributions are derived from the Normal. They are the sampling distributions that arise when you estimate parameters from Normal data.

---

## 7. Confidence Intervals — The Full Picture

### For μ with known σ (Normal):
```
X̄ ± z_{α/2} · σ/√n
```

### For μ with unknown σ (t-distribution):
```
X̄ ± t_{α/2, n−1} · S/√n
```

### For σ² (chi-squared):
```
[(n−1)S²/χ²_{α/2,n−1}, (n−1)S²/χ²_{1−α/2,n−1}]
```

Note: CI for σ² is **asymmetric** — chi-squared is not symmetric.

### For ratio σ₁²/σ₂² (F-distribution):
```
[S₁²/(S₂²·F_{α/2,n₁−1,n₂−1}), S₁²·F_{α/2,n₂−1,n₁−1}/S₂²]
```

---

## 8. Worked Numericals

---

### 🔢 Numerical 1 — Chi-Squared: Variance Estimation

**Problem:** 10 measurements of model latency (ms): 45, 52, 48, 55, 50, 47, 53, 49, 51, 46.

Assuming Normal distribution:
**(a)** Compute sample mean X̄ and sample variance S².
**(b)** Find the 95% CI for true variance σ².
**(c)** Test H₀: σ² = 10 vs H₁: σ² ≠ 10 at α=0.05.

**Solution:**

**(a)**
```
X̄ = (45+52+48+55+50+47+53+49+51+46)/10 = 496/10 = 49.6

Deviations from mean:
−4.6, 2.4, −1.6, 5.4, 0.4, −2.6, 3.4, −0.6, 1.4, −3.6

Σ(Xᵢ−X̄)² = 21.16+5.76+2.56+29.16+0.16+6.76+11.56+0.36+1.96+12.96 = 92.4

S² = 92.4/(10−1) = 92.4/9 = 10.267 ms²
S = √10.267 ≈ 3.204 ms
```

**(b)** 95% CI for σ²:

χ²_{0.025, 9} = 2.700 (lower tail)
χ²_{0.975, 9} = 19.023 (upper tail)

```
CI: [(n−1)S²/χ²_{upper}, (n−1)S²/χ²_{lower}]
  = [9×10.267/19.023, 9×10.267/2.700]
  = [92.4/19.023, 92.4/2.700]
  = [4.858, 34.222]
```

**95% CI for σ²: (4.86, 34.22) ms²**

The interval is wide and asymmetric — typical for chi-squared CIs with small n.

**(c)** Test statistic:
```
χ² = (n−1)S²/σ₀² = 9×10.267/10 = 9.24
```

Under H₀, χ²~χ²(9). Critical values: 2.700 and 19.023.

Since 2.700 < 9.24 < 19.023, **fail to reject H₀**. Insufficient evidence that σ² ≠ 10.

---

### 🔢 Numerical 2 — One-Sample t-Test: Model Performance

**Problem:** You claim a model achieves μ=0.90 F1 score. You test on 8 independent datasets:
```
F1 scores: 0.87, 0.92, 0.85, 0.91, 0.88, 0.94, 0.89, 0.90
```

Test H₀: μ=0.90 vs H₁: μ≠0.90 at α=0.05.

**Solution:**

```
X̄ = (0.87+0.92+0.85+0.91+0.88+0.94+0.89+0.90)/8
   = 7.16/8 = 0.895

Σ(Xᵢ−X̄)² = 0.000625+0.000625+0.002025+0.000225+0.000225+0.002025+0.000025+0.000025
           = 0.005800

S² = 0.005800/7 = 0.000829
S = 0.02878

SE = S/√n = 0.02878/√8 = 0.02878/2.828 = 0.01018

t = (X̄ − μ₀)/SE = (0.895 − 0.90)/0.01018 = −0.005/0.01018 = −0.491
```

Degrees of freedom: ν = n−1 = 7

Critical value: t_{0.025, 7} = 2.365 (two-sided 5%)

|t| = 0.491 < 2.365 → **Fail to reject H₀**

p-value: P(|t₇| > 0.491) ≈ 2×P(t₇ > 0.491) ≈ 2×0.32 = **0.64**

The observed mean (0.895) is entirely consistent with the claimed μ=0.90. No evidence of a difference.

**ML insight:** With only n=8 datasets, the t-test has low power. Even if the true mean were 0.87, you'd likely fail to reject H₀. This is why benchmarking requires many evaluation runs.

---

### 🔢 Numerical 3 — Two-Sample t-Test: Comparing Two Models

**Problem:** Two models evaluated on 12 datasets each:
- Model A: X̄_A = 0.82, S_A = 0.05, n_A = 12
- Model B: X̄_B = 0.78, S_B = 0.06, n_B = 12

Test H₀: μ_A = μ_B vs H₁: μ_A > μ_B at α=0.05.

**Solution:**

**Welch's t-test** (unequal variances assumed — safer in practice):

```
SE_{diff} = √(S_A²/n_A + S_B²/n_B)
          = √(0.0025/12 + 0.0036/12)
          = √(0.000208 + 0.000300)
          = √0.000508 = 0.02254

t = (X̄_A − X̄_B) / SE_{diff} = (0.82 − 0.78) / 0.02254 = 0.04/0.02254 = 1.775
```

**Welch-Satterthwaite degrees of freedom:**
```
ν = (S_A²/n_A + S_B²/n_B)² / [(S_A²/n_A)²/(n_A−1) + (S_B²/n_B)²/(n_B−1)]
  = (0.000508)² / [(0.000208)²/11 + (0.000300)²/11]
  = 2.581×10⁻⁷ / [3.94×10⁻⁹ + 8.18×10⁻⁹]
  = 2.581×10⁻⁷ / 1.212×10⁻⁸
  ≈ 21.3  →  use ν=21
```

One-sided critical value: t_{0.05, 21} = 1.721

t = 1.775 > 1.721 → **Reject H₀** at α=0.05

p-value ≈ P(t₂₁ > 1.775) ≈ 0.045

**Model A is significantly better than Model B at the 5% level.**

**ML insight:** Two-sample t-tests are the standard tool for comparing two ML models. "My model beats the baseline" — quantify it with a two-sample t-test over multiple evaluation runs.

---

### 🔢 Numerical 4 — F-Test: Comparing Model Variance

**Problem:** Two training runs (different random seeds) produce:
- Run 1: n₁=16 evaluations, S₁²=0.008 (model A)
- Run 2: n₂=21 evaluations, S₂²=0.003 (model B)

Test H₀: σ₁²=σ₂² vs H₁: σ₁²≠σ₂² at α=0.10.

**Solution:**

```
F = S₁²/S₂² = 0.008/0.003 = 2.667

ν₁ = n₁−1 = 15,   ν₂ = n₂−1 = 20
```

Two-sided test: compare to F_{0.05, 15, 20} (upper 5%) and F_{0.95, 15, 20} (lower 5%).

From F-tables: F_{0.05, 15, 20} ≈ 2.20

Since F = 2.667 > 2.20 → **Reject H₀** at α=0.10 (two-sided).

Model A has significantly higher variance than Model B — Model B is more stable across seeds.

**ML insight:** Variance of model performance across seeds/folds is as important as mean performance. A model with lower variance (more stable) is often preferred in production even if its mean is slightly lower. The F-test quantifies this comparison.

---

### 🔢 Numerical 5 — t vs Normal: When the Difference Matters

**Problem:** Model accuracy μ is unknown. Sample of n=5 evaluations: X̄=0.88, S=0.04.

**(a)** 95% CI using Normal (wrongly assuming σ=S is known).
**(b)** 95% CI using t-distribution (correct).
**(c)** How much wider is the correct interval?
**(d)** At what n do the two intervals become practically equal?

**Solution:**

SE = S/√n = 0.04/√5 = 0.04/2.236 = 0.01789

**(a) Normal CI (wrong):**
```
X̄ ± 1.96 × SE = 0.88 ± 1.96×0.01789 = 0.88 ± 0.0351
= (0.845, 0.915)
```

**(b) t-CI (correct, ν=4):**
```
t_{0.025, 4} = 2.776

X̄ ± 2.776 × SE = 0.88 ± 2.776×0.01789 = 0.88 ± 0.0497
= (0.830, 0.930)
```

**(c)** t-interval width: 0.0994 vs Normal width: 0.0702

**41% wider** — using Normal when you should use t dramatically underestimates uncertainty with n=5.

**(d)** t_{0.025,ν} vs z_{0.025}=1.96:

| ν (df) | n | t_{0.025,ν} | % wider than z |
|---|---|---|---|
| 4 | 5 | 2.776 | 41.6% |
| 9 | 10 | 2.262 | 15.4% |
| 19 | 20 | 2.093 | 6.8% |
| 29 | 30 | 2.045 | 4.3% |
| 59 | 60 | 2.000 | 2.0% |
| 119 | 120 | 1.980 | 1.0% |
| ∞ | ∞ | 1.960 | 0% |

**At n≥30:** t-interval is only 4.3% wider — practically indistinguishable. This is why "n≥30, use Normal" is a common rule.

**ML insight:** For model evaluation with few training runs (n=5 seeds), ALWAYS use t-intervals. For large-scale evaluations (n≥30), Normal approximation is adequate.

---

### 🔢 Numerical 6 — F-Test in Regression: Overall Significance

**Problem:** A regression model predicts house price from p=3 features (size, age, location) on n=50 observations.

R²=0.72 (model explains 72% of variance).

**(a)** Compute the F-statistic.
**(b)** Is the regression significant at α=0.01?
**(c)** What does this F-test actually test?

**Solution:**

**(a)** Regression F-statistic:
```
F = (R²/p) / ((1−R²)/(n−p−1))
  = (0.72/3) / ((0.28/46))
  = 0.24 / 0.006087
  = 39.43
```

Degrees of freedom: ν₁=p=3, ν₂=n−p−1=46

**(b)** F_{0.01, 3, 46} ≈ 4.24 (from F-table)

F = 39.43 >> 4.24 → **Highly significant** (p << 0.01)

**(c)** The F-test tests: **H₀: β₁=β₂=β₃=0** (all slope coefficients are zero — model is useless) vs H₁: at least one βᵢ ≠ 0.

Rejecting H₀ means the regression as a whole explains significantly more variance than just predicting the mean.

**Important distinction:** F-test tests the whole model. Individual t-tests (for each βᵢ) test each coefficient separately. It's possible to have a significant F-test (model matters) but no individually significant coefficients (multicollinearity distributes significance across correlated features).

---

### 🔢 Numerical 7 — Chi-Squared Goodness-of-Fit Test

**Problem:** A model classifies images into 4 categories. On n=200 test images, you expect equal distribution (50 per category) but observe:

| Category | Observed | Expected |
|---|---|---|
| Cat | 60 | 50 |
| Dog | 45 | 50 |
| Bird | 55 | 50 |
| Fish | 40 | 50 |

Test H₀: equal distribution across categories at α=0.05.

**Solution:**

**Pearson chi-squared statistic:**
```
χ² = Σ (Oᵢ − Eᵢ)² / Eᵢ
   = (60−50)²/50 + (45−50)²/50 + (55−50)²/50 + (40−50)²/50
   = 100/50 + 25/50 + 25/50 + 100/50
   = 2.0 + 0.5 + 0.5 + 2.0
   = 5.0
```

Degrees of freedom: k−1 = 4−1 = 3

Critical value: χ²_{0.05, 3} = 7.815

χ² = 5.0 < 7.815 → **Fail to reject H₀**

p-value = P(χ²₃ > 5.0) ≈ 0.172

No significant evidence that the model's classification is unequal across categories.

**ML insight:** The chi-squared goodness-of-fit test is the go-to test for:
- Checking if model output class distribution matches expected
- Testing if data follows a specific distribution (e.g., Poisson, Normal)
- Detecting label imbalance introduced by a model
- Testing independence in contingency tables (chi-squared test of independence)

---

## 9. ANOVA: F-Distribution for Multiple Groups

**ANOVA (Analysis of Variance)** tests whether k group means are equal:

H₀: μ₁ = μ₂ = ... = μₖ    vs    H₁: at least one differs

```
F = Between-group variance / Within-group variance
  = MSB / MSW ~ F(k−1, n−k)    under H₀

MSB = Σᵢ nᵢ(X̄ᵢ−X̄)² / (k−1)     [Between Mean Square]
MSW = ΣᵢΣⱼ(Xᵢⱼ−X̄ᵢ)² / (n−k)    [Within Mean Square]
```

**ML use:** Comparing k different models or hyperparameter settings across multiple evaluation runs. F-test generalizes the two-sample t-test to k groups.

---

## 10. Common Interview Questions

| Question | Key Idea |
|---|---|
| "Why use t instead of Normal?" | σ unknown — estimated by S adds uncertainty → heavier tails |
| "What is the chi-squared distribution?" | Sum of k squared independent N(0,1) variables |
| "What is (n−1)S²/σ² distributed as?" | χ²(n−1) — key result for variance inference |
| "What does degrees of freedom mean?" | Number of free parameters after estimating constraints |
| "When does t(ν) converge to Normal?" | As ν→∞; practically at ν≥30 |
| "What does the F-test in regression test?" | H₀: all slope coefficients = 0 (model is useless) |
| "What is t² distributed as?" | F(1,ν) — squared t equals F |
| "When do you use chi-squared goodness-of-fit?" | Testing if observed counts match expected distribution |
| "Why is the chi-squared CI for variance asymmetric?" | χ² distribution is right-skewed (not symmetric) |

---

## 11. Key Formulas — Cheat Sheet for Day 21

```
Sample Variance:
    S² = Σ(Xᵢ−X̄)²/(n−1)    [unbiased, denominator n−1]
    E[S²] = σ²

Chi-Squared χ²(k):
    X = Z₁²+...+Zₖ² ~ χ²(k)   where Zᵢ~N(0,1) i.i.d.
    E[X] = k,   Var(X) = 2k
    (n−1)S²/σ² ~ χ²(n−1)      [sample variance result]
    CI for σ²: [(n−1)S²/χ²_{α/2}, (n−1)S²/χ²_{1−α/2}]

t-distribution t(ν):
    T = Z/√(V/ν) ~ t(ν)        [Z~N(0,1), V~χ²(ν) indep]
    T = (X̄−μ)/(S/√n) ~ t(n−1)  [one-sample t]
    E[T]=0, Var(T)=ν/(ν−2)
    As ν→∞: t(ν)→N(0,1)
    t(1) = Cauchy

F-distribution F(ν₁,ν₂):
    F = (U/ν₁)/(V/ν₂) ~ F(ν₁,ν₂)  [U~χ²(ν₁), V~χ²(ν₂) indep]
    [t(ν)]² ~ F(1,ν)
    Regression F: (R²/p)/((1−R²)/(n−p−1)) ~ F(p,n−p−1)

Confidence Intervals:
    μ (σ known):   X̄ ± z_{α/2}·σ/√n
    μ (σ unknown): X̄ ± t_{α/2,n−1}·S/√n
    σ²:            [(n−1)S²/χ²_{α/2,n−1}, (n−1)S²/χ²_{1−α/2,n−1}]

Pearson χ² Goodness-of-Fit:
    χ² = Σ(Oᵢ−Eᵢ)²/Eᵢ ~ χ²(k−1)    under H₀

Key relationships:
    N(0,1) → χ²(1) → χ²(k) → F(ν₁,ν₂)
    N(0,1) / √(χ²(ν)/ν) → t(ν)
    [t(ν)]² → F(1,ν)
```

---

## 12. Practice Problems (Solve Before Day 22)

1. 6 model accuracy scores: 0.91, 0.88, 0.93, 0.87, 0.90, 0.92. Assuming Normal:
   - Compute X̄ and S².
   - Find 90% CI for μ using t-distribution.
   - Test H₀: μ=0.90 vs H₁: μ≠0.90 at α=0.10.

2. Why does (n−1)S²/σ² have n−1 degrees of freedom instead of n? Explain intuitively using the constraint Σ(Xᵢ−X̄)=0.

3. Model A tested on 10 datasets (S_A²=0.016), Model B on 15 datasets (S_B²=0.004). Test H₀: σ_A²=σ_B² vs H₁: σ_A²>σ_B² at α=0.05. Compute F-statistic and decision.

4. A classifier output has observed class counts: [35, 25, 30, 10] on n=100 samples. Test if output is uniformly distributed (expected 25 each) using chi-squared test at α=0.01.

5. *(Interview-level)* A regression of sales (Y) on advertising spend (X) with n=30 observations gives R²=0.45 with p=1 predictor. Compute:
   - The F-statistic
   - The t-statistic for the slope coefficient (Hint: t=√F for p=1)
   - Interpret: are advertising spend and sales significantly related?

---

## 13. Looking Ahead

**Day 22** — **Maximum Likelihood Estimation (MLE).** The most important estimation method in statistics and ML. We formally derive MLE, prove its properties (consistency, asymptotic normality, efficiency), and apply it to Gaussian, Bernoulli, and Poisson models — connecting back to the loss functions used in deep learning.

---
*End of Day 21 | Next: Day 22 — Maximum Likelihood Estimation*
