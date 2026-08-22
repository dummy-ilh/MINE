# Day 25 — Confidence Intervals
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Casella & Berger Chapter 9; Wasserman, *All of Statistics* Chapter 9
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Confidence Intervals Matter in ML

Every reported model metric needs an uncertainty estimate. A number without an interval is incomplete.

| ML Context | Confidence Interval |
|---|---|
| Model accuracy | "92% ± 1.5% (95% CI)" |
| A/B test improvement | "CTR increased by 2.3% (CI: 0.8%–3.8%)" |
| Regression coefficient | "β₁ = 0.34 (CI: 0.21–0.47)" |
| Latency SLA | "P95 latency = 145ms (CI: 138–152ms)" |
| Feature importance | SHAP value ± bootstrap CI |
| Cross-validation accuracy | "84.3% ± 2.1% across 10 folds" |

In ML interviews, you will be asked: "Is your model actually better?" A confidence interval is the rigorous answer.

---

## 2. Definition and Correct Interpretation

> **Definition:** A **95% confidence interval** [L(data), U(data)] for parameter θ satisfies:
> ```
> P(L(data) ≤ θ ≤ U(data)) = 0.95
> ```
> where the probability is over **repeated sampling of the data**, with θ fixed.

### The Critical Correct Interpretation

**WRONG (common mistake):**
> "There is a 95% probability that θ is in [L, U]."

**CORRECT:**
> "If we repeated this experiment many times and constructed a CI each time, 95% of those intervals would contain the true θ."

**Why the difference matters:**
- In frequentist statistics, θ is fixed (not random). The interval is random (it varies with each sample).
- The 95% refers to the procedure's long-run coverage, not a probability statement about this specific interval.
- The Bayesian equivalent (credible interval) DOES support the "probability θ is here" interpretation (Day 24).

### For a Specific Realized Interval

Once you observe data and compute [L, U], that specific interval either contains θ or it doesn't — there's no probability. But we say the method that produced it has 95% coverage.

---

## 3. General Construction of Confidence Intervals

### Method 1: Pivot Method

A **pivot** is a function Q(data, θ) whose distribution is known and doesn't depend on θ.

```
Find Q(data, θ) ~ known distribution (e.g., N(0,1), t(ν))
Find a, b: P(a ≤ Q ≤ b) = 0.95
Invert to get: P(L(data) ≤ θ ≤ U(data)) = 0.95
```

**Example pivot for Normal mean (σ known):**
```
Q = (X̄ − μ)/(σ/√n) ~ N(0,1)    [pivot]
P(−1.96 ≤ Q ≤ 1.96) = 0.95
→ P(X̄ − 1.96σ/√n ≤ μ ≤ X̄ + 1.96σ/√n) = 0.95
```

### Method 2: Likelihood Ratio / Profile Likelihood

### Method 3: Bootstrap (Day 25 Section 7)

---

## 4. Standard Confidence Intervals

### For μ with Known σ (z-interval)

```
X̄ ± z_{α/2} · σ/√n

95% CI: X̄ ± 1.96 · σ/√n
99% CI: X̄ ± 2.576 · σ/√n
```

Margin of error: E = z_{α/2} · σ/√n

Sample size for margin E: **n = (z_{α/2} · σ/E)²**

### For μ with Unknown σ (t-interval)

```
X̄ ± t_{α/2, n−1} · S/√n

where S = √[(1/(n−1))Σ(Xᵢ−X̄)²]
```

Use t-table with ν = n−1 degrees of freedom.

### For Proportion p (Wald Interval)

```
p̂ ± z_{α/2} · √(p̂(1−p̂)/n)

where p̂ = k/n
```

**Wilson Interval** (better for small n or extreme p̂):
```
p̃ ± z_{α/2} · √(p̂(1−p̂)/n + z²_{α/2}/(4n²)) / (1 + z²_{α/2}/n)

where p̃ = (k + z²_{α/2}/2)/(n + z²_{α/2})
```

### For σ² (Chi-squared interval)

```
[(n−1)S²/χ²_{α/2, n−1}, (n−1)S²/χ²_{1−α/2, n−1}]
```

Asymmetric because chi-squared is skewed.

### For Difference of Means μ₁ − μ₂ (Two-sample)

```
(X̄₁ − X̄₂) ± t_{α/2, ν} · √(S₁²/n₁ + S₂²/n₂)

ν = Welch-Satterthwaite df (Day 21)
```

---

## 5. Factors Affecting CI Width

```
Width = 2 × z_{α/2} × σ/√n

Width INCREASES when:
    - Confidence level increases (z_{α/2} larger)
    - Population variance σ² increases (more spread)
    - Sample size n decreases

Width DECREASES when:
    - Confidence level decreases
    - Population variance decreases
    - Sample size n increases (as 1/√n)
```

**To halve the CI width: quadruple the sample size (4×n).**

---

## 6. Bootstrap Confidence Intervals

When the distribution of the estimator is unknown or hard to derive analytically, use the **bootstrap**:

### Algorithm

```
Given: data X = {x₁,...,xₙ}, estimator θ̂ = T(X)

For b = 1 to B:
    1. Draw bootstrap sample X*ᵦ of size n with replacement from X
    2. Compute θ̂*ᵦ = T(X*ᵦ)

Result: bootstrap distribution {θ̂*₁, ..., θ̂*_B}
```

### Bootstrap CI Methods

**Percentile Method** (simplest):
```
95% CI: [θ̂*_{(0.025B)}, θ̂*_{(0.975B)}]    [2.5th and 97.5th percentile of bootstrap distribution]
```

**Basic Bootstrap (Reflected):**
```
95% CI: [2θ̂ − θ̂*_{(0.975B)}, 2θ̂ − θ̂*_{(0.025B)}]
```

**Bootstrap-t (Studentized):**
```
95% CI: [θ̂ − t*_{(0.975)} · SE, θ̂ − t*_{(0.025)} · SE]
```

Most accurate; requires SE estimate per bootstrap sample.

### When to Use Bootstrap

- Non-Normal data
- Complex estimators (median, correlation, AUC, F1)
- No closed-form sampling distribution
- Small samples where CLT hasn't kicked in

---

## 7. CI vs Credible Interval — The Critical Distinction

| | Frequentist CI | Bayesian Credible Interval |
|---|---|---|
| **Definition** | P(interval contains θ) = 95% over repeated sampling | P(θ ∈ interval \| data) = 95% |
| **Probability statement** | About the procedure | About θ itself |
| **θ treated as** | Fixed unknown constant | Random variable |
| **Requires prior?** | No | Yes |
| **Interpretation** | 95% of such intervals cover true θ | 95% probability θ is here |
| **More intuitive?** | Less | More |
| **Computation** | Often closed form | May require MCMC |

**The Bayesian credible interval is what most people think they're getting from frequentist CIs.**

In practice, for large n with reasonable priors, they're numerically similar. For small n, they can differ substantially.

---

## 8. Worked Numericals

---

### 🔢 Numerical 1 — CI for Model Accuracy (z-interval)

**Problem:** Model tested on n=500 samples: 440 correct. Compute:

**(a)** Point estimate p̂ and SE.
**(b)** 95% CI (Wald).
**(c)** 99% CI.
**(d)** How many samples needed to get CI width ≤ 2%?

**Solution:**

**(a)**
```
p̂ = 440/500 = 0.880
SE = √(p̂(1−p̂)/n) = √(0.880×0.120/500) = √(0.0002112) = 0.01453
```

**(b) 95% CI (Wald):**
```
p̂ ± 1.96 × SE = 0.880 ± 1.96×0.01453 = 0.880 ± 0.0285
= (0.852, 0.909)
```

**(c) 99% CI:**
```
p̂ ± 2.576 × SE = 0.880 ± 2.576×0.01453 = 0.880 ± 0.0374
= (0.843, 0.917)
```

**(d)** Width = 2 × 1.96 × √(p̂(1−p̂)/n) ≤ 0.02

Using p̂ = 0.88 (conservative, use p(1-p)=0.5×0.5=0.25 for truly conservative):
```
n ≥ (1.96)² × p̂(1−p̂) / (E/2)²
  = (1.96)² × 0.880×0.120 / (0.01)²
  = 3.8416 × 0.1056 / 0.0001
  = 0.4058 / 0.0001
  = 4058
```

Need **n ≈ 4,058 samples** for a 95% CI with width ≤ 2%.

**ML insight:** Current test with n=500 gives CI width ≈ 5.7%. For publication-quality reporting of model accuracy to ±1%, you need ~4,000+ test samples. This is why large-scale ML benchmarks use thousands of test examples.

---

### 🔢 Numerical 2 — CI for Mean with Unknown σ (t-interval)

**Problem:** 8 evaluation runs of a model give F1 scores:
0.87, 0.91, 0.88, 0.93, 0.85, 0.90, 0.92, 0.89.

**(a)** Point estimate and sample std.
**(b)** 95% CI using t-distribution.
**(c)** What would CI be if n=8 were mistakenly using z (Normal)?
**(d)** 90% CI — narrower or wider?

**Solution:**

**(a)**
```
X̄ = (0.87+0.91+0.88+0.93+0.85+0.90+0.92+0.89)/8
   = 7.15/8 = 0.89375

Deviations: −0.024, 0.016, −0.014, 0.036, −0.044, 0.006, 0.026, −0.004
Σdev² = 0.000576+0.000256+0.000196+0.001296+0.001936+0.000036+0.000676+0.000016
      = 0.004988

S² = 0.004988/7 = 0.000713
S = 0.02670
SE = S/√8 = 0.02670/2.828 = 0.009445
```

**(b) 95% t-CI, ν=7:**

t_{0.025, 7} = 2.365

```
X̄ ± t × SE = 0.89375 ± 2.365×0.009445
            = 0.89375 ± 0.02234
            = (0.871, 0.916)
```

**(c) 95% z-CI (wrong for n=8):**

z_{0.025} = 1.960

```
X̄ ± 1.960×SE = 0.89375 ± 1.960×0.009445 = 0.89375 ± 0.01851 = (0.875, 0.912)
```

**t-CI: (0.871, 0.916) width=0.045**
**z-CI: (0.875, 0.912) width=0.037**

Using z instead of t with n=8 gives a CI that's **18% too narrow** — falsely precise. This is why you must use t-distribution for small samples.

**(d) 90% t-CI:** t_{0.05, 7} = 1.895

```
0.89375 ± 1.895×0.009445 = 0.89375 ± 0.01789 = (0.876, 0.912)
```

Width = 0.036. **Narrower** than 95% CI (0.045) — accepting more risk of missing θ gives a tighter interval. ✓

---

### 🔢 Numerical 3 — CI for Difference: Comparing Two Models

**Problem:** A/B test results:
- Model A: n_A=200 samples, 164 correct (p̂_A=0.820)
- Model B: n_B=200 samples, 176 correct (p̂_B=0.880)

**(a)** 95% CI for p_B − p_A.
**(b)** Does the CI include 0? What does this mean?
**(c)** P-value for H₀: p_A=p_B.
**(d)** Minimum effect size detectable with 80% power at α=0.05.

**Solution:**

**(a)**
```
p̂_B − p̂_A = 0.880 − 0.820 = 0.060

SE(diff) = √(p̂_A(1−p̂_A)/n_A + p̂_B(1−p̂_B)/n_B)
         = √(0.820×0.180/200 + 0.880×0.120/200)
         = √(0.000738 + 0.000528)
         = √0.001266 = 0.03558

95% CI: 0.060 ± 1.96×0.03558 = 0.060 ± 0.0697 = (−0.010, 0.130)
```

**(b)** CI includes 0 → **not statistically significant at α=0.05**.

The observed difference of 6 percentage points could plausibly be zero — we cannot reject H₀: p_A=p_B.

**(c)** Pooled p̂ = (164+176)/(200+200) = 340/400 = 0.850

SE_pooled = √(0.850×0.150×(1/200+1/200)) = √(0.000638×2) = √0.001275 = 0.03571

Z = 0.060/0.03571 = 1.68

p-value = 2P(Z > 1.68) = 2×0.0465 = **0.093 > 0.05**

Not significant. Consistent with CI including 0. ✓

**(d)** Minimum detectable effect (MDE) for 80% power, α=0.05 (one-sided z_{0.80}=0.842, z_{0.05}=1.645):

```
MDE = (z_{α} + z_{β}) × SE_pooled
    = (1.645 + 0.842) × 0.03571
    = 2.487 × 0.03571
    ≈ 0.089 = 8.9%
```

With n=200 per group, you can only reliably detect differences of 8.9%+ at 80% power. The observed 6% difference was too small to detect — **underpowered test**.

---

### 🔢 Numerical 4 — Bootstrap CI for AUC

**Problem:** 10 test predictions with true labels:

| Score | Label |
|---|---|
| 0.9 | 1 |
| 0.8 | 1 |
| 0.7 | 0 |
| 0.6 | 1 |
| 0.5 | 0 |
| 0.4 | 1 |
| 0.3 | 0 |
| 0.2 | 0 |
| 0.1 | 1 |
| 0.05 | 0 |

Observed AUC = 0.76 (computed from the data).

**(a)** Why use bootstrap for AUC CI?
**(b)** Describe one bootstrap resample and how to compute AUC.
**(c)** If bootstrap gives AUC values {0.72, 0.68, 0.80, 0.76, 0.84, 0.70, 0.78, 0.74, 0.82, 0.66, ...} with mean=0.756, SD=0.056, give 95% percentile CI.
**(d)** Interpret the interval.

**Solution:**

**(a)** AUC has no simple closed-form sampling distribution — it depends on the rank ordering of all predictions, which involves complex combinatorics. Bootstrap provides a distribution-free CI that works for any estimator.

**(b)** One bootstrap resample: draw 10 points with replacement. Example:
```
{(0.9,1), (0.6,1), (0.6,1), (0.2,0), (0.9,1), (0.4,1), (0.3,0), (0.7,0), (0.1,1), (0.5,0)}
```
Compute AUC on this resample (count concordant pairs / total pairs).

**(c)** With B=1000 bootstrap AUC values, 95% percentile CI:

Take 25th and 975th ordered values. If SD=0.056 and approximately Normal:

Approximate percentile CI: 0.756 ± 1.96×0.056 = 0.756 ± 0.110 = **(0.646, 0.866)**

**(d)** Interpretation: We are 95% confident the true AUC is between 0.646 and 0.866.

The wide interval reflects the small test set (n=10). With n=1000, the CI would be roughly 0.76 ± 0.034 — much tighter.

**ML insight:** AUC, F1, precision@k, NDCG — all complex metrics should be reported with bootstrap CIs. A model with AUC=0.80 on 50 test samples has CI roughly ±0.14 — the difference from a model with AUC=0.75 is not statistically meaningful.

---

### 🔢 Numerical 5 — CI Width and Sample Size Planning

**Problem:** You need to report accuracy within ±2% (i.e., half-width E=0.02) with 95% confidence. How many test samples do you need?

**(a)** Conservative estimate (p̂=0.5).
**(b)** If you expect p̂≈0.9 (good model).
**(c)** For 99% CI instead of 95%.
**(d)** For ±1% half-width.

**Solution:**

Formula: n = z²_{α/2} × p̂(1−p̂) / E²

**(a) Conservative (p̂=0.5, maximum variance):**
```
n = (1.96)² × 0.5×0.5 / (0.02)²
  = 3.8416 × 0.25 / 0.0004
  = 0.9604 / 0.0004 = 2401
```

**2,401 samples** for ±2% CI (conservative).

**(b) Expected p̂=0.9:**
```
n = (1.96)² × 0.9×0.1 / (0.02)²
  = 3.8416 × 0.09 / 0.0004
  = 0.3457 / 0.0004 = 864
```

**864 samples** — much fewer when model is good (less variance near p=1).

**(c) 99% CI (z=2.576), p̂=0.5:**
```
n = (2.576)² × 0.25 / 0.0004 = 6.635×0.25/0.0004 = 4,148
```

**(d) ±1% half-width (E=0.01), p̂=0.5:**
```
n = (1.96)² × 0.25 / (0.01)² = 3.8416×0.25/0.0001 = 9,604
```

**Quadrupling precision requires 4× samples.**

| Scenario | n required |
|---|---|
| ±2%, 95%, p̂=0.5 | 2,401 |
| ±2%, 95%, p̂=0.9 | 864 |
| ±2%, 99%, p̂=0.5 | 4,148 |
| ±1%, 95%, p̂=0.5 | 9,604 |

**ML insight:** Before running an experiment, always compute the required sample size. Many ML papers report improvements on test sets of n=100–500 — too small for reliable ±1% accuracy claims. Proper benchmarking requires thousands of samples.

---

### 🔢 Numerical 6 — CI for Regression Coefficient

**Problem:** Linear regression y = β₀ + β₁x + ε on n=20 observations.

Results: β̂₁ = 0.45, S_{β̂₁} = 0.12 (standard error of slope estimate).

**(a)** 95% CI for β₁.
**(b)** Test H₀: β₁=0.
**(c)** Test H₀: β₁=0.5.
**(d)** Relationship between CI and hypothesis test.

**Solution:**

**(a)** t-CI with ν = n−2 = 18 degrees of freedom:

t_{0.025, 18} = 2.101

```
β̂₁ ± t × SE = 0.45 ± 2.101×0.12 = 0.45 ± 0.252 = (0.198, 0.702)
```

**(b)** H₀: β₁=0:

```
t = (β̂₁ − 0)/SE = 0.45/0.12 = 3.75

|t| = 3.75 > 2.101  → Reject H₀
p-value ≈ 2P(t₁₈ > 3.75) ≈ 0.0014
```

β₁ is significantly different from 0 — x is a significant predictor.

**(c)** H₀: β₁=0.5:

```
t = (0.45 − 0.5)/0.12 = −0.05/0.12 = −0.417

|t| = 0.417 < 2.101  → Fail to reject H₀
```

Cannot reject that the true slope is 0.5.

**(d)** **Key relationship:** H₀: β₁ = β₀ is rejected at level α if and only if β₀ is NOT in the (1−α) CI.

CI = (0.198, 0.702):
- β₁=0: NOT in CI → rejected ✓
- β₁=0.5: IN CI → not rejected ✓

**The confidence interval is the set of all null hypothesis values you would fail to reject.**

---

### 🔢 Numerical 7 — Frequentist vs Bayesian Intervals: Side by Side

**Problem:** Test set: n=20 samples, k=16 correct. Compare:

**(a)** Frequentist 95% CI (Wald and Wilson).
**(b)** Bayesian 95% credible interval (uniform prior Beta(1,1)).
**(c)** Bayesian 95% credible interval (informative prior Beta(10,5)).
**(d)** Interpret differences.

**Solution:**

p̂ = 16/20 = 0.80

**(a) Frequentist CIs:**

Wald:
```
SE = √(0.80×0.20/20) = √0.008 = 0.0894
95% CI: 0.80 ± 1.96×0.0894 = 0.80 ± 0.175 = (0.625, 0.975)
```

Wilson (better for small n):
```
p̃ = (16 + 1.96²/2)/(20 + 1.96²) = (16+1.92)/(20+3.84) = 17.92/23.84 = 0.7517
SE_wilson = √(0.80×0.20/20 + 1.96²/(4×400)) / (1 + 1.96²/20)
≈ (0.625, 0.944)    [Wilson is narrower and more accurate for small n]
```

**(b) Bayesian, uniform prior Beta(1,1):**

Posterior: Beta(1+16, 1+4) = Beta(17, 5)
```
Mean = 17/22 = 0.773
95% credible interval: use Beta quantiles
≈ (0.531, 0.927)
```

**(c) Bayesian, informative prior Beta(10,5):**

This encodes prior belief that p ≈ 10/15 ≈ 0.67 (prior says model is decent but not great).

Posterior: Beta(10+16, 5+4) = Beta(26, 9)
```
Mean = 26/35 = 0.743
95% credible interval ≈ (0.573, 0.888)
```

**(d) Comparison:**

| Method | Interval | Width | Interpretation |
|---|---|---|---|
| Wald | (0.625, 0.975) | 0.350 | Procedure coverage |
| Wilson | (0.625, 0.944) | 0.319 | Better coverage |
| Bayes (uniform) | (0.531, 0.927) | 0.396 | P(p in interval\|data)=95% |
| Bayes (informative) | (0.573, 0.888) | 0.315 | Incorporates prior belief |

**Key observations:**
- Wald extends above 0.975 — implausible for probability
- Bayesian intervals respect [0,1] bounds naturally
- Informative prior shrinks interval (more information → less uncertainty)
- All intervals are wide with n=20 — small sample, large uncertainty

**ML insight:** For small evaluation sets (n<50), Bayesian credible intervals with sensible priors often outperform frequentist CIs in coverage and interpretability. The Wald interval fails badly near p=0 or p=1.

---

## 9. Common Mistakes in CI Interpretation

**Mistake 1:** "There is a 95% probability that the true accuracy is between 85% and 91%."
- Wrong: frequentist θ is fixed, interval is random
- Right: "Our estimation procedure produces intervals that contain the true accuracy 95% of the time."

**Mistake 2:** "Since the CI is (0.85, 0.91), accuracies outside this range are impossible."
- Wrong: values outside are just less consistent with data
- Right: the CI is the range compatible with the data at the 95% level

**Mistake 3:** "A wider CI means the model is worse."
- Wrong: wider CI means more uncertainty (small n or high variance)
- Right: CI width reflects data quality, not model quality

**Mistake 4:** "If two CIs overlap, there's no significant difference."
- Wrong: overlapping CIs don't necessarily mean non-significant difference
- Right: test the difference directly (CI for the difference)

**Mistake 5:** "Larger confidence level = better."
- Wrong: 99% CI is wider — less informative about the true value
- Right: choose confidence level based on the cost of Type I error

---

## 10. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is a confidence interval?" | Interval [L,U] where P(L≤θ≤U)=95% over repeated sampling |
| "Correct interpretation of 95% CI?" | If repeated, 95% of such intervals contain true θ — NOT P(θ in interval)=95% |
| "How does CI relate to hypothesis testing?" | Reject H₀:θ=θ₀ iff θ₀ not in CI |
| "When do you use t vs z interval?" | t when σ unknown (always in practice); z when σ known or n large |
| "What is a bootstrap CI?" | Use resampling to estimate sampling distribution of any estimator |
| "CI vs credible interval?" | Frequentist (procedure coverage) vs Bayesian (direct probability statement) |
| "How to reduce CI width by half?" | Quadruple sample size (n scales as 1/E²) |
| "Why is Wilson better than Wald for proportions?" | Wald fails near p=0 or p=1; Wilson has better coverage for small n |
| "Two CIs overlap — are they significantly different?" | Not necessarily — test the difference directly |

---

## 11. Key Formulas — Cheat Sheet for Day 25

```
z-interval (σ known):
    X̄ ± z_{α/2} · σ/√n

t-interval (σ unknown):
    X̄ ± t_{α/2,n−1} · S/√n

Proportion CI (Wald):
    p̂ ± z_{α/2} · √(p̂(1−p̂)/n)

Two-sample difference CI:
    (p̂₁−p̂₂) ± z_{α/2}·√(p̂₁(1−p̂₁)/n₁ + p̂₂(1−p̂₂)/n₂)

Variance CI (chi-squared):
    [(n−1)S²/χ²_{α/2}, (n−1)S²/χ²_{1−α/2}]

Sample size for proportion CI:
    n = z²_{α/2} · p̂(1−p̂) / E²
    Conservative (p̂=0.5): n = z²_{α/2} / (4E²)

Key z-values:
    90% CI: z = 1.645
    95% CI: z = 1.960
    99% CI: z = 2.576

Bootstrap percentile CI:
    [θ̂*_{(α/2)}, θ̂*_{(1−α/2)}]

CI ↔ Hypothesis test:
    Reject H₀:θ=θ₀ at level α ⟺ θ₀ ∉ (1−α) CI

Width factors:
    Width ∝ z_{α/2} · σ / √n
    Halve width → 4× sample size
```

---

## 12. Practice Problems (Solve Before Day 26)

1. A model achieves 450/500 correct on test A and 380/450 correct on test B. Compute 95% CI for accuracy on each test. Do the CIs overlap? Compute CI for the difference in accuracy.

2. 6 training runs give losses: 0.32, 0.28, 0.35, 0.30, 0.29, 0.33. Compute 95% CI for mean loss using t-distribution. Test H₀: mean loss = 0.30.

3. You want to detect an accuracy improvement of ε=0.03 with 90% power and α=0.05 (two-sided). Baseline accuracy p₀=0.80. How many samples needed per group? (Use formula: n = (z_α + z_β)² × 2p(1−p) / ε²)

4. **Prove** that the 95% CI is the set of all θ₀ values for which H₀:θ=θ₀ is not rejected at the 5% level. (Hint: show that θ₀ ∉ CI ⟺ the test statistic |X̄−θ₀|/(S/√n) > t_{0.025}.)

5. *(Interview-level)* Model A is evaluated on 1000 samples (accuracy 0.85) and Model B on 500 samples (accuracy 0.87). A colleague says "B is better." You say "the CIs overlap." They say "so what?" Resolve the debate: compute the CI for the difference p_B−p_A, and explain whether B is significantly better, accounting for the different sample sizes.

---

## 13. Looking Ahead

**Day 26** — **Hypothesis Testing: p-values, Type I/II Errors & Test Statistics.** The formal framework for making decisions from data. We'll cover z-tests, t-tests, chi-squared tests, and the multiple testing problem — with direct connections to model evaluation, A/B testing, and feature selection in ML.

---
*End of Day 25 | Next: Day 26 — Hypothesis Testing*
