# Day 26 — Hypothesis Testing: p-values, Type I/II Errors & Test Statistics
### *30-Day Probability & Statistics for AI/ML Interviews*
> **Reference:** Casella & Berger Chapter 8; Wasserman *All of Statistics* Chapter 10
> **Style:** Andrew Ng — intuition first, math second, numericals always

---

## 1. Why Hypothesis Testing is Central to ML Practice

Every model comparison, every A/B test, every feature importance claim involves hypothesis testing.

| ML Question | Hypothesis Test |
|---|---|
| Is Model A better than Model B? | Two-sample test on accuracy |
| Does feature X improve performance? | Test coefficient β = 0 in regression |
| Is the click rate different after launch? | Proportion test (A/B test) |
| Does the data follow a Normal distribution? | Kolmogorov-Smirnov test |
| Are these two features independent? | Chi-squared independence test |
| Is this word count different from expected? | Poisson goodness-of-fit test |
| Are all group means equal? | ANOVA F-test |
| Is this anomaly statistically significant? | Z-test for deviation from baseline |

Hypothesis testing is how you distinguish "this looks better" from "this IS better."

---

## 2. The Framework: Four Key Components

### Step 1 — Hypotheses

```
H₀: Null hypothesis  — the "nothing is happening" claim
H₁: Alternative      — what you want to show evidence for
```

**H₀ is what you assume by default.** You only reject it when evidence is overwhelming.

### Step 2 — Test Statistic

A function T(data) that measures evidence against H₀.

```
Under H₀, T follows a known distribution (N(0,1), t(ν), χ²(k), F(ν₁,ν₂))
Large |T| → evidence against H₀
```

### Step 3 — p-value

```
p-value = P(T is at least as extreme as observed | H₀ is true)
```

Small p-value → observed data is unlikely under H₀ → evidence against H₀.

### Step 4 — Decision

```
Reject H₀ if p-value < α    [α = significance level, typically 0.05]
```

---

## 3. The p-value — Deep Understanding

> **Definition:** The p-value is the probability of observing a test statistic at least as extreme as the one computed, **assuming H₀ is true**.

### What the p-value IS and ISN'T

| Statement | Correct? |
|---|---|
| p-value = P(H₀ is true) | ❌ WRONG |
| p-value = P(data | H₀) | ✓ Correct |
| Small p means H₁ is true | ❌ WRONG |
| Small p means data is unlikely under H₀ | ✓ Correct |
| p = 0.05 means 5% chance of error | ❌ WRONG |
| p < 0.05 means the effect is large | ❌ WRONG |
| p < 0.05 means we reject H₀ at 5% significance | ✓ Correct |

**The p-value is NOT the probability that H₀ is true.** This is the most common misinterpretation in science and ML.

### Under H₀, p-values are Uniform(0,1)

If H₀ is true, p-values follow Uniform(0,1). This means:
- 5% of the time, you'll get p < 0.05 even when H₀ is true (Type I error rate = α)
- Running 100 tests under H₀ → expect ~5 false positives

---

## 4. Type I and Type II Errors

|  | H₀ True | H₀ False |
|---|---|---|
| **Reject H₀** | Type I Error (False Positive) α | Correct (Power = 1−β) |
| **Fail to Reject H₀** | Correct (1−α) | Type II Error (False Negative) β |

```
α = P(Type I Error)  = P(Reject H₀ | H₀ true)   = significance level
β = P(Type II Error) = P(Fail to reject | H₀ false)
Power = 1 − β        = P(Reject H₀ | H₁ true)   = sensitivity
```

### The Tradeoff

```
Decreasing α (stricter threshold): fewer Type I errors, more Type II errors
Increasing α (looser threshold):   fewer Type II errors, more Type I errors
```

**In ML terms:**
- Type I: saying "model B is better" when it isn't → deploy inferior model
- Type II: saying "models are the same" when B is better → miss an improvement

The business cost determines which error is worse.

### Power Analysis

```
Power = P(Reject H₀ | true effect size = δ)

Power depends on:
    α    [significance level]
    n    [sample size]
    δ    [effect size = signal]
    σ    [noise level]

Power = Φ(|δ|√n/σ − z_{α/2})  [for two-sided z-test]
```

To achieve power 1−β at significance α:
```
n = (z_{α/2} + z_β)² · σ²/δ²
```

---

## 5. Key Test Statistics

### One-Sample z-test (σ known)
```
H₀: μ = μ₀
Z = (X̄ − μ₀)/(σ/√n) ~ N(0,1)    under H₀
```

### One-Sample t-test (σ unknown)
```
H₀: μ = μ₀
T = (X̄ − μ₀)/(S/√n) ~ t(n−1)    under H₀
```

### Two-Sample z-test (proportions)
```
H₀: p₁ = p₂
Z = (p̂₁ − p̂₂)/√[p̂(1−p̂)(1/n₁+1/n₂)]    where p̂ = pooled proportion
Z ~ N(0,1)    under H₀
```

### Chi-Squared Tests
```
Goodness-of-fit:   χ² = Σ(Oᵢ−Eᵢ)²/Eᵢ ~ χ²(k−1)
Independence:      χ² = Σᵢⱼ(Oᵢⱼ−Eᵢⱼ)²/Eᵢⱼ ~ χ²((r−1)(c−1))
```

### F-test (ANOVA / Regression)
```
F = MSB/MSW ~ F(k−1, n−k)    [ANOVA]
F = (R²/p)/((1−R²)/(n−p−1)) ~ F(p, n−p−1)    [regression]
```

---

## 6. One-Sided vs Two-Sided Tests

```
Two-sided (H₁: μ ≠ μ₀):
    Reject when |T| > t_{α/2}
    p-value = 2·P(T > |t_obs|)

One-sided right (H₁: μ > μ₀):
    Reject when T > t_α
    p-value = P(T > t_obs)

One-sided left (H₁: μ < μ₀):
    Reject when T < −t_α
    p-value = P(T < t_obs)
```

**Rule:** Use two-sided when you have no directional prior. Use one-sided when you only care about one direction (e.g., "is the new model better?" — one-sided).

One-sided test has higher power for the same α — it "bets" on the direction.

---

## 7. Multiple Testing Problem

Running m tests at level α: P(at least one false positive) = 1−(1−α)ᵐ.

For m=20, α=0.05: P(≥1 false positive) = 1−0.95²⁰ = 64%!

### Corrections

**Bonferroni:** Use α* = α/m per test.
- Controls Family-Wise Error Rate (FWER): P(any false positive) ≤ α
- Very conservative (loses power)

**Benjamini-Hochberg (BH):** Controls False Discovery Rate (FDR)
```
Sort p-values: p_{(1)} ≤ p_{(2)} ≤ ... ≤ p_{(m)}
Find k* = max{k: p_{(k)} ≤ k·α/m}
Reject all H₀_{(i)} for i ≤ k*
```
- Controls E[FP/Rejections] ≤ α
- More powerful than Bonferroni for many simultaneous tests

**ML applications:**
- Feature selection: testing each feature's effect (hundreds of tests)
- Genome-wide association studies: millions of tests
- Neural architecture search: comparing many configurations
- Hyperparameter tuning: evaluating many settings

---

## 8. Statistical vs Practical Significance

**Statistical significance:** p < α → the effect is real (not due to chance)

**Practical significance:** The effect is large enough to matter in practice

These are different:
- With n=1,000,000: even a 0.001% accuracy improvement is statistically significant
- With n=10: even a 10% improvement might not be statistically significant

### Effect Size Measures

```
Cohen's d (means):    d = (μ₁−μ₂)/σ_pooled
                      Small: 0.2, Medium: 0.5, Large: 0.8

Proportion difference: h = 2arcsin(√p₁) − 2arcsin(√p₂)

R²:                   Proportion of variance explained
```

**Always report both p-value AND effect size.**

---

## 9. Worked Numericals

---

### 🔢 Numerical 1 — One-Sample t-test: Is Model Accuracy Above 80%?

**Problem:** You claim your model achieves >80% accuracy. You test on 15 samples:
```
Accuracies: 0.83, 0.79, 0.85, 0.82, 0.88, 0.80, 0.84, 0.81, 0.86, 0.79, 0.83, 0.87, 0.82, 0.85, 0.80
```

Test H₀: μ=0.80 vs H₁: μ>0.80 (one-sided) at α=0.05.

**Solution:**

```
n=15, X̄ = Σxᵢ/15

Σxᵢ = 0.83+0.79+0.85+0.82+0.88+0.80+0.84+0.81+0.86+0.79+0.83+0.87+0.82+0.85+0.80
     = 12.44
X̄ = 12.44/15 = 0.8293

Deviations from 0.8293:
−0.0007, −0.0393, 0.0207, −0.0093, 0.0507, −0.0293, 0.0107, −0.0193, 0.0307, −0.0393, 0.0007, 0.0407, −0.0093, 0.0207, −0.0293

Σ(dev)² = 0.0000049+0.0015445+0.0004285+0.0000865+0.0025705+0.0008585+0.0001145+0.0003725+0.0009425+0.0015445+0.0000049+0.0016565+0.0000865+0.0004285+0.0008585
        = 0.01150

S² = 0.01150/14 = 0.000821
S = 0.02866
SE = S/√15 = 0.02866/3.873 = 0.007400
```

Test statistic:
```
T = (X̄ − μ₀)/SE = (0.8293 − 0.80)/0.007400 = 0.0293/0.0074 = 3.959
```

Degrees of freedom: ν=14

One-sided critical value: t_{0.05, 14} = 1.761

T = 3.959 > 1.761 → **Reject H₀**

p-value = P(t₁₄ > 3.959) ≈ **0.0008**

Strong evidence that mean accuracy exceeds 80%.

**ML insight:** With only 15 evaluation points, a 2.93% improvement is statistically significant (t=3.96). The small SE (0.74%) combined with consistent results makes the signal clear.

---

### 🔢 Numerical 2 — Two-Sample z-test: A/B Test

**Problem:** You launch a new recommendation algorithm.

- Control (A): n_A=2000 users, 240 purchases (p̂_A=0.120)
- Treatment (B): n_B=2000 users, 280 purchases (p̂_B=0.140)

Test H₀: p_A=p_B vs H₁: p_B>p_A (one-sided) at α=0.05.

**Solution:**

Pooled proportion:
```
p̂ = (240+280)/(2000+2000) = 520/4000 = 0.130
```

Standard error:
```
SE = √[p̂(1−p̂)(1/n_A + 1/n_B)]
   = √[0.130×0.870×(1/2000+1/2000)]
   = √[0.1131×0.001]
   = √0.0001131 = 0.01063
```

Test statistic:
```
Z = (p̂_B − p̂_A)/SE = (0.140 − 0.120)/0.01063 = 0.020/0.01063 = 1.882
```

One-sided p-value:
```
p-value = P(Z > 1.882) = 1 − Φ(1.882) = 1 − 0.9700 = 0.030
```

p-value = 0.030 < 0.05 → **Reject H₀**

The new algorithm shows a statistically significant improvement.

**Absolute improvement:** 2.0 percentage points

**Relative improvement:** 2.0/12.0 = 16.7%

**ML insight:** Even a 2pp improvement at this scale is statistically detectable. With n=2000 per arm, the test has good power. Before deploying, also check: is 16.7% improvement practically meaningful? What's the revenue impact?

---

### 🔢 Numerical 3 — Chi-Squared Goodness-of-Fit: Distribution Check

**Problem:** You expect model prediction scores to be approximately Uniform(0,1). You bin 500 predictions into 5 buckets:

| Bin | [0,0.2) | [0.2,0.4) | [0.4,0.6) | [0.6,0.8) | [0.8,1.0] |
|---|---|---|---|---|---|
| Observed | 85 | 95 | 115 | 120 | 85 |
| Expected | 100 | 100 | 100 | 100 | 100 |

Test H₀: uniform distribution at α=0.01.

**Solution:**

```
χ² = Σ(Oᵢ−Eᵢ)²/Eᵢ
   = (85−100)²/100 + (95−100)²/100 + (115−100)²/100 + (120−100)²/100 + (85−100)²/100
   = 225/100 + 25/100 + 225/100 + 400/100 + 225/100
   = 2.25 + 0.25 + 2.25 + 4.00 + 2.25
   = 11.00
```

Degrees of freedom: k−1 = 5−1 = 4

Critical value: χ²_{0.01, 4} = 13.277

11.00 < 13.277 → **Fail to reject H₀** at α=0.01

p-value = P(χ²₄ > 11.0) ≈ **0.026**

At α=0.05 we'd reject; at α=0.01 we don't. The scores are somewhat non-uniform but the deviation isn't overwhelming.

**ML insight:** This test checks if model confidence scores are well-calibrated (uniform distribution of scores means the model uses all confidence levels equally). Overconfident models pile up near 1.0. This chi-squared test quantifies calibration deviation.

---

### 🔢 Numerical 4 — Chi-Squared Independence Test: Feature Independence

**Problem:** Are gender and model prediction correlated in your dataset?

| | Predict = 1 | Predict = 0 | Total |
|---|---|---|---|
| Female | 60 | 140 | 200 |
| Male | 80 | 120 | 200 |
| **Total** | **140** | **260** | **400** |

Test H₀: gender and prediction are independent at α=0.05.

**Solution:**

Expected counts under independence:
```
E_{ij} = (Row total × Col total) / Grand total

E(Female, 1) = 200×140/400 = 70
E(Female, 0) = 200×260/400 = 130
E(Male, 1)   = 200×140/400 = 70
E(Male, 0)   = 200×260/400 = 130
```

Chi-squared statistic:
```
χ² = (60−70)²/70 + (140−130)²/130 + (80−70)²/70 + (120−130)²/130
   = 100/70 + 100/130 + 100/70 + 100/130
   = 1.429 + 0.769 + 1.429 + 0.769
   = 4.396
```

Degrees of freedom: (r−1)(c−1) = (2−1)(2−1) = 1

Critical value: χ²_{0.05, 1} = 3.841

4.396 > 3.841 → **Reject H₀**

p-value = P(χ²₁ > 4.396) ≈ **0.036**

Gender and model prediction are significantly associated.

**ML insight:** This is **disparate impact testing** — checking if a model treats demographic groups differently. A statistically significant result means the model's outputs depend on gender, raising fairness concerns. In production ML, this test (and related metrics) is required for model governance.

---

### 🔢 Numerical 5 — Type I/II Errors and Power Analysis

**Problem:** You're testing whether a new model (Model B) improves accuracy over baseline (Model A, p_A=0.75). You want:
- α=0.05 (5% Type I error rate)
- Power = 80% to detect δ=0.05 improvement (p_B=0.80)

**(a)** Required sample size per group.
**(b)** P(Type I error) and P(Type II error).
**(c)** If true improvement is δ=0.02 (smaller), what is the power with n=500?
**(d)** What n gives 80% power for δ=0.02?

**Solution:**

**(a)** Sample size formula for two proportions:

z_{0.025}=1.960 (two-sided α=0.05), z_{0.20}=0.842 (power=80%)

```
n = (z_{α/2} + z_β)² × [p_A(1−p_A) + p_B(1−p_B)] / δ²
  = (1.960 + 0.842)² × [0.75×0.25 + 0.80×0.20] / (0.05)²
  = (2.802)² × [0.1875 + 0.1600] / 0.0025
  = 7.851 × 0.3475 / 0.0025
  = 2.728 / 0.0025
  = 1,091
```

Need **n ≈ 1,091 per group** (≈2,182 total) to detect δ=0.05 with 80% power.

**(b)**

P(Type I error) = α = **0.05** — set by design.

P(Type II error) = β = 1 − power = 1 − 0.80 = **0.20** — set by design.

**(c)** Power with n=500, δ=0.02:

SE = √[p̄(1−p̄)(2/n)] where p̄ ≈ (0.75+0.77)/2 = 0.76 ≈ √[0.76×0.24×2/500] = √0.000730 = 0.02701

Critical z: Reject when Z > z_{0.025}=1.96 (one-sided at α/2 for two-sided test)

Non-centrality parameter: δ/SE = 0.02/0.02701 = 0.741

Power = P(Z > 1.96 − 0.741) + P(Z < −1.96 − 0.741)
      ≈ P(Z > 1.219) [dominant term]
      = 1 − Φ(1.219) ≈ 1 − 0.889 = **11.1%**

Only **11% power** to detect a 2% improvement with n=500. Almost certainly miss it.

**(d)** n for 80% power with δ=0.02:

```
n = (1.960 + 0.842)² × [0.75×0.25 + 0.77×0.23] / (0.02)²
  = 7.851 × [0.1875 + 0.1771] / 0.0004
  = 7.851 × 0.3646 / 0.0004
  = 7,169
```

Need **n ≈ 7,169 per group** (~14,338 total) to detect δ=0.02 at 80% power.

**ML insight:** Small improvements require huge samples to detect reliably. Before running an experiment, always do power analysis. Many ML papers are underpowered — they would miss real improvements 80%+ of the time.

---

### 🔢 Numerical 6 — Multiple Testing: Bonferroni and BH Correction

**Problem:** You test 20 features for their association with the target. Raw p-values (sorted):

```
0.0003, 0.0012, 0.0045, 0.0089, 0.0120, 0.0231, 0.0340, 0.0412, 0.0501, 0.0611,
0.0742, 0.0891, 0.1023, 0.1230, 0.1567, 0.2341, 0.3012, 0.4123, 0.5678, 0.7234
```

**(a)** Without correction: how many are significant at α=0.05?
**(b)** With Bonferroni correction.
**(c)** With Benjamini-Hochberg correction (FDR=0.05).
**(d)** Which features would you select?

**Solution:**

**(a) No correction:** Features with p < 0.05:
p-values 0.0003, 0.0012, 0.0045, 0.0089, 0.0120, 0.0231, 0.0340, 0.0412 → **8 features**

Expected false positives: 20×0.05 = 1.0 (could be 1 of those 8 is spurious).

**(b) Bonferroni:** α* = 0.05/20 = 0.0025

Features with p < 0.0025: 0.0003, 0.0012 → **2 features**

Very conservative — likely missing real signals (0.0045, 0.0089 were genuine).

**(c) Benjamini-Hochberg:**

Threshold for rank k: p_{(k)} ≤ k × (0.05/20) = k × 0.0025

| Rank k | p-value | Threshold k×0.0025 | Significant? |
|---|---|---|---|
| 1 | 0.0003 | 0.0025 | Yes |
| 2 | 0.0012 | 0.0050 | Yes |
| 3 | 0.0045 | 0.0075 | Yes |
| 4 | 0.0089 | 0.0100 | Yes |
| 5 | 0.0120 | 0.0125 | Yes |
| 6 | 0.0231 | 0.0150 | **No** (0.0231 > 0.0150) |

Find largest k where condition holds: k*=5.

BH rejects all H₀ for ranks 1 through 5 → **5 features**

**(d) Feature selection:**

| Method | Features Selected | Comment |
|---|---|---|
| No correction | 8 | ~1 expected false positive |
| Bonferroni | 2 | Very conservative |
| BH (FDR=5%) | 5 | Balanced: 5% of selected are expected false positives |

**Recommendation:** Use BH correction for feature selection. Bonferroni is too conservative (misses real features); no correction gives too many false positives. BH allows ~5% of selected features to be noise.

---

### 🔢 Numerical 7 — Permutation Test: Model Comparison Without Distributional Assumptions

**Problem:** Model A and Model B are evaluated on 10 test samples each. Observed accuracy difference: Δ = 0.15 (B better).

**(a)** Why might standard tests be inappropriate here?
**(b)** Permutation test setup.
**(c)** If out of 10,000 random permutations, 380 give |Δ*| ≥ 0.15, what is the p-value?
**(d)** Interpret the result.

**Solution:**

**(a)** Standard tests assume:
- i.i.d. samples (may not hold if test samples are correlated)
- Normal approximation (n=10 is too small for CLT to kick in)
- Known test statistic distribution (not always the case for complex metrics)

Permutation test is distribution-free and exact for small samples.

**(b)** Permutation test:

```
1. Pool all 20 accuracy measurements (10 from A, 10 from B)
2. Randomly assign 10 to "A" and 10 to "B"
3. Compute Δ* = mean(B*) − mean(A*)
4. Repeat 10,000 times
5. p-value = fraction of permutations where |Δ*| ≥ |Δ_observed|
```

Under H₀ (A and B are identical), the labeling A/B is arbitrary — permuting should give similar differences.

**(c)**

```
p-value = 380/10000 = 0.038
```

**(d)** p-value = 0.038 < 0.05 → **Reject H₀** at α=0.05.

The observed difference of 0.15 is unlikely (p=3.8%) to occur by chance if A and B have equal performance.

**ML insight:** Permutation tests are the gold standard for small-sample model comparisons — they make no distributional assumptions and are exact. McNemar's test (for paired binary outcomes) and the Wilcoxon signed-rank test are common ML alternatives that don't assume normality.

---

## 10. Common Pitfalls in ML Hypothesis Testing

### Pitfall 1: p-hacking / HARKing
Testing many models, selecting the best, then reporting the test on the same data.

**Fix:** Pre-register your hypothesis; hold out test data until one final evaluation.

### Pitfall 2: Stopping Early
Checking p-values repeatedly and stopping when p < 0.05.

**Fix:** Sequential testing with proper correction (e.g., alpha spending functions).

### Pitfall 3: Forgetting Multiple Testing
Testing model on 50 metrics, reporting the best 5.

**Fix:** BH correction for all metrics tested.

### Pitfall 4: Confusing Statistical and Practical Significance
"p=0.001, we improved accuracy by 0.01%" — statistically significant but useless.

**Fix:** Always report effect size alongside p-value.

### Pitfall 5: Using Training Set for Testing
Testing model improvement on the same data used to train/select the model.

**Fix:** Always test on held-out data.

### Pitfall 6: Violation of Independence
Test samples are from the same user, time period, or cluster — correlated observations.

**Fix:** Account for correlation structure; use cluster-robust standard errors.

---

## 11. Common Interview Questions

| Question | Key Idea |
|---|---|
| "What is the p-value?" | P(data this extreme \| H₀ true) — NOT P(H₀ true) |
| "What is α?" | P(Type I error) = significance level |
| "What is power?" | P(reject H₀ \| H₁ true) = 1−β |
| "How does sample size affect power?" | Larger n → higher power → easier to detect real effects |
| "What is the multiple testing problem?" | Running m tests at α gives ~m×α false positives |
| "Bonferroni vs BH correction?" | Bonferroni: controls FWER (conservative); BH: controls FDR (more powerful) |
| "Statistical vs practical significance?" | Statistical: p<α. Practical: effect is large enough to matter |
| "When to use permutation test?" | Small samples, no distributional assumptions, complex statistics |
| "What is p-hacking?" | Selectively reporting tests that give p<0.05 — inflates false positive rate |
| "What is the two-sample t-test for model comparison?" | T = (X̄_A−X̄_B)/SE_{diff} ~ t(ν) under H₀ |

---

## 12. Key Formulas — Cheat Sheet for Day 26

```
Framework:
    H₀ (null) vs H₁ (alternative)
    p-value = P(T ≥ t_obs | H₀)
    Reject H₀ if p < α

Errors:
    α = P(Type I) = P(reject | H₀ true)
    β = P(Type II) = P(fail to reject | H₁ true)
    Power = 1−β = P(reject | H₁ true)

Key test statistics:
    One-sample z:    Z = (X̄−μ₀)/(σ/√n) ~ N(0,1)
    One-sample t:    T = (X̄−μ₀)/(S/√n) ~ t(n−1)
    Two-proportion:  Z = (p̂₁−p̂₂)/SE ~ N(0,1)
    Chi-squared GOF: χ² = Σ(O−E)²/E ~ χ²(k−1)
    Chi-squared ind: χ² = Σ(O−E)²/E ~ χ²((r−1)(c−1))

Sample size for power 1−β at significance α:
    n = (z_{α/2}+z_β)²σ²/δ²    [means, one-sample]
    n = (z_{α/2}+z_β)²[p₁(1−p₁)+p₂(1−p₂)]/δ²  [proportions]

Multiple testing:
    Bonferroni:  α* = α/m         [controls FWER]
    BH:          reject p_{(k)} ≤ k·α/m    [controls FDR]

p-value under H₀: Uniform(0,1)
p-value under H₁: concentrated near 0

Effect size:
    Cohen's d = (μ₁−μ₂)/σ_pooled
    Small=0.2, Medium=0.5, Large=0.8
```

---

## 13. Practice Problems (Solve Before Day 27)

1. Model accuracy over 12 test sets: {0.82, 0.85, 0.79, 0.88, 0.83, 0.81, 0.86, 0.84, 0.80, 0.87, 0.83, 0.85}. Test H₀: μ=0.82 vs H₁: μ>0.82 at α=0.05. Compute T, p-value, and conclusion.

2. An experiment has α=0.05 and power=0.80. Fill in the 2×2 error table with probabilities. If P(H₀ true)=0.70 (prior), what is P(H₀ true | reject H₀)? (This is the positive predictive value — connect to Bayes' theorem.)

3. You run a chi-squared test on a 3×4 contingency table. What are the degrees of freedom? If χ²=18.5, is it significant at α=0.05? At α=0.01?

4. You test 50 features using BH at FDR=0.10. Raw p-values sorted: the 12th smallest is 0.048 and the 13th is 0.053. Determine the BH cutoff and how many features are selected.

5. *(Interview-level)* A large-scale ML system makes 1 million binary predictions per day. You run a chi-squared independence test between prediction and a sensitive attribute. With n=1,000,000, you get χ²=4.2 (df=1, p=0.040). Is there a fairness problem? How does the concept of statistical vs practical significance apply here?

---

## 14. Looking Ahead

**Day 27** — **A/B Testing: The Complete ML Practitioner's Guide.** We bring together everything — hypothesis testing, confidence intervals, power analysis, multiple testing — into a complete framework for running and analyzing A/B tests in production ML systems. This is the single most tested topic in DS/ML interviews at tech companies.

---
*End of Day 26 | Next: Day 27 — A/B Testing*
