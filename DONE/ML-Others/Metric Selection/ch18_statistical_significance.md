# Chapter 18: Statistical Significance in Experiments

> *"A p-value tells you how surprised you should be if the null hypothesis were true. It does not tell you whether your experiment worked, whether the effect is real, or whether it matters. Confusing these four things is the source of most bad experimental science."*

---

## 18.1 Why Statistical Rigor Matters in ML Experiments

Every time you run an A/B test, compare two models, or claim a metric improvement, you are making a statistical inference. Without rigor:

- You mistake random fluctuations for real effects
- You run tests until you see significance (p-hacking)
- You stop tests early when results look good
- You run many tests and don't correct for multiple comparisons
- You report improvements that don't replicate

The consequences in ML:
- Models deployed that don't actually improve the product
- Engineering effort wasted chasing noise
- Loss of credibility when improvements don't hold in production
- Goodhart's Law accelerated (optimize metrics that fluctuate in your favor)

```
An experiment shows CTR improved by 2.1%, p = 0.047.
You ship the model.
In production: CTR unchanged.

What happened?
  - Sample size was marginal; random fluctuation crossed the threshold
  - You peaked at the data mid-experiment and stopped when it looked good
  - You tried 8 variants and reported only the one that worked
  - Weekly seasonality wasn't accounted for

Statistical rigor would have caught all of these.
```

---

## 18.2 Hypothesis Testing Fundamentals

### The Null and Alternative Hypotheses

```
H₀ (Null):        No effect. Any observed difference is due to chance.
H₁ (Alternative): There is a real effect.

In A/B testing:
  H₀: CTR_treatment = CTR_control   (model change has no effect)
  H₁: CTR_treatment ≠ CTR_control   (two-tailed)
  H₁: CTR_treatment > CTR_control   (one-tailed; directional claim)
```

**Pre-register your hypothesis.** Choose H₁ before looking at data. Switching from two-tailed to one-tailed after seeing the direction of results is p-hacking.

### The Two Types of Errors

```
                    H₀ True           H₀ False (H₁ True)
Reject H₀:    Type I error (FP)    Correct (True Positive)
              False alarm           Detected real effect
              Rate: α               Rate: Power = 1 - β

Fail to Reject H₀: Correct (TN)   Type II error (FN)
                   Correctly        Missed real effect
                   no alarm         Rate: β
```

- **α (significance level):** Probability of Type I error. You control this. Typically 0.05.
- **β:** Probability of Type II error.
- **Power = 1 - β:** Probability of detecting a real effect when it exists. Typically target ≥ 0.80.

### The p-value

```
p-value = P(observing data at least this extreme | H₀ is true)
```

**What a p-value IS:**
- The probability that random chance alone produces an effect at least as large as observed
- A measure of evidence against H₀

**What a p-value IS NOT:**
- The probability that H₀ is true
- The probability that the effect is real
- A measure of effect size or practical importance
- A binary verdict (significant ≠ important; not significant ≠ no effect)

### Significance Level vs. p-value

```
α = 0.05: Your pre-specified Type I error tolerance
p = 0.03: What you observed

p < α → Reject H₀ (statistically significant)
p > α → Fail to reject H₀ (not statistically significant)
```

The threshold α must be set **before** the experiment. Changing it after seeing results invalidates the test.

---

## 18.3 Effect Size

Statistical significance tells you whether an effect exists. Effect size tells you how big it is.

A tiny effect can be statistically significant with enough data. A large effect can be statistically insignificant with too little data. You need both.

### Common Effect Size Measures

**Cohen's d (continuous outcomes):**

```
d = (μ₁ - μ₂) / σ_pooled

σ_pooled = √[(σ₁² + σ₂²) / 2]

|d| < 0.2:  Small effect
|d| = 0.5:  Medium effect
|d| > 0.8:  Large effect
```

**Relative lift (proportional metrics):**

```
Relative Lift = (metric_treatment - metric_control) / metric_control × 100%
```

More interpretable than absolute differences for business metrics.

**Odds Ratio (binary outcomes):**

```
OR = (p_treatment / (1 - p_treatment)) / (p_control / (1 - p_control))

OR = 1.0: No effect
OR > 1.0: Treatment increases odds
OR < 1.0: Treatment decreases odds
```

**Cohen's h (proportions):**

```
h = 2 × arcsin(√p₁) - 2 × arcsin(√p₂)
```

Stabilizes variance for proportion comparisons.

### Minimum Detectable Effect (MDE)

The smallest effect size your experiment is powered to detect reliably. Set this before running:

```
MDE = minimum business-meaningful improvement

Example:
  Current CTR: 2.0%
  A CTR improvement of 0.1% is not worth shipping
  A CTR improvement of 0.3% justifies the engineering cost
  → MDE = 0.3% relative (or 0.006% absolute)
```

MDE drives sample size calculation. A smaller MDE requires a larger sample.

---

## 18.4 Power Analysis and Sample Size

Power analysis answers: **how many samples do I need?**

The four quantities are linked — specify any three, solve for the fourth:
- **α**: Significance level (typically 0.05)
- **Power (1-β)**: Target power (typically 0.80 or 0.80)
- **MDE**: Minimum detectable effect
- **n**: Required sample size per group

### Sample Size Formula (Two-Sample Proportions)

For comparing two proportions (e.g., CTR):

```
n = (z_{α/2} + z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²

z_{α/2} = 1.96  (for α = 0.05, two-tailed)
z_β     = 0.84  (for power = 0.80)
z_β     = 1.28  (for power = 0.90)
```

**Example:**

```
Control CTR:    p₁ = 0.020
Treatment CTR:  p₂ = 0.023   (15% relative lift; our MDE)
α = 0.05, Power = 0.80

n = (1.96 + 0.84)² × (0.020×0.980 + 0.023×0.977) / (0.023 - 0.020)²
  = 7.84 × (0.0196 + 0.0225) / 0.000009
  = 7.84 × 0.0421 / 0.000009
  ≈ 36,700 per group   (73,400 total)
```

At 1,000 users/day: need 73 days. Too long? Either increase MDE threshold or accept lower power.

### Sample Size Formula (Two-Sample Means)

For continuous metrics (revenue, session length):

```
n = 2 × (z_{α/2} + z_β)² × σ² / δ²

σ = standard deviation of the metric
δ = minimum detectable difference (absolute)
```

### Python Power Analysis

```python
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# For proportions
effect_size = proportion_effectsize(0.023, 0.020)   # Cohen's h
analysis = NormalIndPower()
n = analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.80)
print(f"Required n per group: {int(np.ceil(n))}")   # ~36,700

# For means (Cohen's d)
analysis = TTestIndPower()
n = analysis.solve_power(effect_size=0.2, alpha=0.05, power=0.80)
print(f"Required n per group (d=0.2): {int(np.ceil(n))}")   # ~394
```

### The Power Curve

Plot power as a function of sample size for your MDE:

```
Power
1.0 |                    ___________
    |                ___/
    |             __/
0.8 |          __/         ← Target power
    |       __/
    |     _/
0.5 |   _/
    | _/
0.0 +________________________
    0   5K   10K  20K  50K  100K
              Sample Size per Group
```

Use this to communicate to stakeholders: "We need 37K users per group, which takes 37 days at current traffic."

---

## 18.5 The t-Test and Z-Test

### Two-Sample Z-Test (Large Samples)

For comparing proportions with n > 30:

```python
from statsmodels.stats.proportion import proportions_ztest

count = np.array([control_conversions, treatment_conversions])
nobs  = np.array([n_control, n_treatment])

z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
```

### Two-Sample t-Test (Means, Any Sample Size)

```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(
    control_metric,
    treatment_metric,
    equal_var=False   # Welch's t-test; doesn't assume equal variances
)

# Effect size
pooled_std = np.sqrt((control_metric.std()**2 + treatment_metric.std()**2) / 2)
cohens_d = (treatment_metric.mean() - control_metric.mean()) / pooled_std

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Relative lift: {(treatment_metric.mean()/control_metric.mean()-1)*100:.2f}%")
```

### Confidence Intervals

Always report confidence intervals alongside p-values:

```python
import scipy.stats as stats

diff = treatment_mean - control_mean
se = np.sqrt(treatment_var/n_treatment + control_var/n_control)
ci_lower, ci_upper = diff - 1.96*se, diff + 1.96*se

print(f"Effect: {diff:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
```

**Reading the CI:**
- CI excludes 0: statistically significant (consistent with p < 0.05)
- CI entirely above 0: positive effect
- CI entirely below 0: negative effect
- Wide CI: underpowered; uncertain effect size
- Narrow CI crossing 0: precisely measured null result

---

## 18.6 Multiple Testing Problem

Run 20 independent tests at α = 0.05. Even if all null hypotheses are true, you expect 1 false positive (0.05 × 20 = 1). Run 100 tests, expect 5 false positives.

This is the **multiple comparisons problem** — one of the most common sources of false discoveries in ML experimentation.

### Sources of Multiple Testing in ML

- Testing multiple variants against control
- Checking many metrics for significance
- Peeking at results repeatedly during the experiment
- Testing subgroup effects (many user segments)
- Running experiments sequentially and claiming any winner

### Family-Wise Error Rate (FWER) Control

Control the probability of making **any** Type I error across all tests.

**Bonferroni correction (conservative):**

```
α_adjusted = α / m   (m = number of tests)

20 tests, α = 0.05:
α_adjusted = 0.05 / 20 = 0.0025

Use α = 0.0025 as threshold for each individual test.
```

**Holm-Bonferroni (less conservative):**

```
1. Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. Reject H₀ᵢ if pᵢ ≤ α / (m - i + 1)
3. Stop at first non-rejection; accept all remaining H₀
```

More powerful than Bonferroni because the threshold adjusts as you go.

### False Discovery Rate (FDR) Control

Control the **fraction** of discoveries that are false, rather than the probability of any false discovery. Less conservative than FWER; more suitable when running many tests.

**Benjamini-Hochberg (BH) procedure:**

```python
from statsmodels.stats.multitest import multipletests

p_values = [0.001, 0.008, 0.039, 0.041, 0.210, 0.040, 0.890]

# BH correction at FDR = 0.05
reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

for i, (r, p_orig, p_adj) in enumerate(zip(reject, p_values, p_adjusted)):
    print(f"Test {i+1}: p={p_orig:.3f}, adj_p={p_adj:.3f}, reject={r}")
```

**When to use:**
- **Bonferroni/Holm:** High stakes; any false positive is very costly (drug trials, legal decisions)
- **BH/FDR:** Exploratory analysis; some false positives acceptable; prioritizing discovery

---

## 18.7 Common Experimental Design Pitfalls

### P-Hacking (Data Dredging)

Running the test and checking significance repeatedly. Stopping when p < 0.05.

```
Simulation: Flip a fair coin. Check p-value at n=50, 100, 150, ... 1000.
Probability of seeing p < 0.05 at least once: ~50%
(Even though coin is fair and H₀ is true)
```

**Solution:** Pre-register the sample size. Do not peek. Analyze once.

If you must peek: use **sequential testing** (see 18.8).

### Underpowered Experiments

Running tests with insufficient sample size. Significant results from underpowered tests:
1. Overestimate effect size (winner's curse)
2. Have low reproducibility
3. Have high variance in estimated effect

```
Power = 0.20 → 80% of real effects are missed
              → Detected effects have inflated estimated sizes (publication bias)
              → Winner's curse: observed effect is systematically larger than true effect
```

**Solution:** Always do power analysis before running the experiment.

### Novelty Effect

Users engage more with anything new, regardless of quality. A new recommendation algorithm might see a CTR boost just because it's different.

```
Week 1: Treatment CTR = 2.4%  ← novelty boost
Week 2: Treatment CTR = 2.2%
Week 3: Treatment CTR = 2.1%  ← settling to true effect
Week 4: Treatment CTR = 2.1%

If you stopped after Week 1: false positive (novelty, not real improvement)
If you ran 4 weeks: true effect = +0.1%, not +0.4%
```

**Solution:** Run experiments long enough to let novelty wear off. For high-frequency interactions (search, feed), 1–2 weeks is usually sufficient. For low-frequency interactions (purchase), longer.

### SRM (Sample Ratio Mismatch)

The actual split between control and treatment doesn't match the intended split.

```
Intended: 50/50 split
Observed: Control = 52,341  Treatment = 47,823

Chi-squared test: p < 0.001 → SRM detected

Why it happens:
  - Bot traffic filtered asymmetrically
  - Caching effects
  - Assignment bugs
  - Self-selection (users who see the treatment behave differently)
```

**Always check for SRM before analyzing results.** SRM invalidates the experiment.

```python
from scipy.stats import chisquare

observed = [n_control, n_treatment]
expected = [total/2, total/2]  # For 50/50 split

chi2, p = chisquare(observed, expected)
if p < 0.01:
    print("WARNING: Sample Ratio Mismatch detected. Results invalid.")
```

### Interference (Spillover)

Treatment affects the control group through shared infrastructure, social networks, or markets.

```
Example: Pricing experiment
  Treatment group: users shown lower prices
  Control group: users shown normal prices

  But users talk to each other: "I got it for $20!"
  Control users are now less likely to buy at normal price.
  → Spillover; SUTVA (Stable Unit Treatment Value Assumption) violated
```

**Solutions:**
- Cluster randomization (randomize by group, not individual)
- Geographic holdout (different regions for control/treatment)
- Time-based holdout (control period vs. treatment period)

### Carryover Effects

Previous experiment affects current experiment. User behavior changed by exposure to an earlier variant.

**Solution:** Washout period between experiments. Do not run experiments back-to-back on the same users without a gap.

---

## 18.8 Sequential Testing

Classic hypothesis testing requires a fixed sample size decided upfront. In practice, business pressures create desire to peek early. Sequential testing provides a statistically valid framework for continuous monitoring.

### The Problem with Naive Peeking

```
Run test. Check significance daily.
Stop and ship when p < 0.05.

Result: True Type I error rate ≈ 25-30% (not 5%)
Even when H₀ is true, continuous peeking nearly guarantees finding significance.
```

### Sequential Probability Ratio Test (SPRT)

After each observation, compute likelihood ratio and compare to boundaries:

```
LR = P(data | H₁) / P(data | H₀)

Stop for H₁ if: LR ≥ (1-β)/α    ← Reject H₀
Stop for H₀ if: LR ≤ β/(1-α)    ← Accept H₀
Continue if:    between boundaries
```

Guarantees Type I and Type II error rates regardless of when you stop.

### Always Valid p-values (mSPRT)

A modern sequential testing approach that lets you peek at any time while controlling Type I error:

```
At each time t, compute an "always valid" p-value pₜ.
Stop whenever pₜ < α — at any time — without inflating Type I error.
```

Implemented in:
- Spotify's `sequential` package
- Optimizely's Stats Engine
- Statsig's sequential testing

```python
# Conceptual always-valid p-value (mSPRT for proportions)
from scipy.stats import norm

def always_valid_p(n_control, conversions_control, n_treatment, conversions_treatment):
    p_c = conversions_control / n_control
    p_t = conversions_treatment / n_treatment
    diff = p_t - p_c
    se = np.sqrt(p_c*(1-p_c)/n_control + p_t*(1-p_t)/n_treatment)

    # Standard z-test p-value (for illustration; use mSPRT library for production)
    z = diff / se
    return 2 * (1 - norm.cdf(abs(z)))
```

### Bayesian A/B Testing

Instead of p-values, compute the posterior probability that treatment beats control:

```
P(treatment > control | data) = ?

Using Beta-Binomial model for conversion rates:

Prior: Beta(α₀, β₀)  (e.g., Beta(1,1) = uniform)
Posterior: Beta(α₀ + conversions, β₀ + non-conversions)

P(p_treatment > p_control) = ∫∫ P(p_t > p_c) dP_t dP_c
```

```python
import numpy as np

def bayesian_ab_test(control_conv, control_total,
                     treatment_conv, treatment_total,
                     n_samples=100_000):

    # Sample from Beta posteriors
    p_control   = np.random.beta(1 + control_conv,
                                  1 + control_total - control_conv,
                                  n_samples)
    p_treatment = np.random.beta(1 + treatment_conv,
                                  1 + treatment_total - treatment_conv,
                                  n_samples)

    prob_treatment_better = (p_treatment > p_control).mean()
    expected_lift = (p_treatment - p_control).mean()
    credible_interval = np.percentile(p_treatment - p_control, [2.5, 97.5])

    return {
        'prob_treatment_better': prob_treatment_better,
        'expected_lift': expected_lift,
        'credible_interval': credible_interval
    }
```

**Bayesian advantages:**
- Intuitive: "85% probability treatment is better" vs. p-value interpretation
- Valid at any sample size (no fixed stopping rule required)
- Can incorporate prior beliefs
- Reports expected lift with uncertainty

**Bayesian limitations:**
- Requires prior (though flat prior minimizes its influence)
- Type I error rate not formally controlled in the frequentist sense
- Harder to audit and standardize across teams

---

## 18.9 Variance Reduction Techniques

Reducing metric variance increases experiment sensitivity — you can detect smaller effects with the same sample size, or detect the same effect with fewer users.

### CUPED (Controlled-experiment Using Pre-Experiment Data)

The most widely used variance reduction technique in industry (Deng et al., 2013).

```
Adjusted metric = Y - θ × (X - E[X])

Y = post-experiment metric
X = pre-experiment metric (same metric, before experiment started)
θ = cov(Y, X) / var(X)   ← optimal coefficient
```

By subtracting the predictable variation in Y explained by pre-experiment behavior, the residual variance is much smaller.

```python
def cuped_adjustment(y_post, x_pre):
    """Apply CUPED variance reduction."""
    theta = np.cov(y_post, x_pre)[0, 1] / np.var(x_pre)
    y_adjusted = y_post - theta * (x_pre - x_pre.mean())
    variance_reduction = 1 - np.var(y_adjusted) / np.var(y_post)
    return y_adjusted, variance_reduction

y_adjusted_control, var_red = cuped_adjustment(y_control, x_pre_control)
y_adjusted_treatment, _     = cuped_adjustment(y_treatment, x_pre_treatment)

print(f"Variance reduction: {var_red:.1%}")  # Typically 20-60% reduction
```

### Stratified Sampling

Pre-stratify users by known high-variance segments (country, device, plan type). Ensure equal representation of each stratum in control and treatment. Reduces variance by eliminating between-stratum differences.

### Delta Method for Ratio Metrics

Ratio metrics (CTR = clicks/impressions) don't have simple variance formulas. The delta method provides a variance approximation:

```
Var(clicks/impressions) ≈ (1/n̄²) × [Var(clicks) + (clicks̄/impressions̄)² × Var(impressions)
                                       - 2 × (clicks̄/impressions̄) × Cov(clicks, impressions)]
```

Use the delta method when your primary metric is a ratio. Ignoring this leads to incorrect standard errors and invalid p-values.

---

## 18.10 The Pre-Experiment Checklist

Run through this before launching any experiment:

**Hypothesis and design:**
- [ ] H₀ and H₁ stated clearly; one-tailed vs. two-tailed decided
- [ ] Primary metric pre-registered (not chosen after seeing data)
- [ ] Guardrail metrics defined (revenue, latency, error rate)
- [ ] MDE defined based on business significance, not just statistical convenience

**Sample size:**
- [ ] Power analysis completed (n per group, expected duration)
- [ ] Power ≥ 0.80; α = 0.05 (or justified deviation)
- [ ] Traffic sufficient to complete in reasonable time

**Validity checks:**
- [ ] Randomization unit appropriate (user, session, cookie)
- [ ] SRM check planned (chi-squared test on split ratio)
- [ ] Interference/spillover risks assessed
- [ ] Washout period from previous experiment included

**Analysis plan:**
- [ ] Statistical test pre-specified (t-test, z-test, etc.)
- [ ] Multiple testing correction pre-specified if multiple metrics
- [ ] One analysis time point (or sequential testing if peeking required)
- [ ] Effect size and CI will be reported alongside p-value

---

## 18.11 Worked Example: Search Ranking A/B Test

**Context:** New learning-to-rank model. Primary metric: NDCG@10. Traffic: 500K searches/day.

```
Step 1: Power analysis
  Current NDCG@10:    0.742
  MDE:                0.010 (1.35% relative lift; engineering threshold)
  Metric std dev σ:   0.089 (estimated from historical data)
  α = 0.05, Power = 0.90

  n = 2 × (1.96 + 1.28)² × (0.089)² / (0.010)²
    = 2 × 10.50 × 0.00792 / 0.0001
    = 1,664 searches per group
    → With 500K searches/day and 50/50 split: 1 day more than sufficient

  → But minimum 7 days to capture weekly seasonality

Step 2: SRM check (after 7 days)
  Control:   1,742,831 searches
  Treatment: 1,758,204 searches
  Chi-squared: p = 0.41  ✓ (no SRM)

Step 3: CUPED variance reduction
  Pre-experiment NDCG (previous 7 days) used as covariate
  Variance reduction: 43%
  Effective sample size increase: ~75%

Step 4: Primary analysis
  Control NDCG@10:    0.742 ± 0.0012 (SE)
  Treatment NDCG@10:  0.754 ± 0.0011 (SE)
  Absolute lift:      +0.012 (95% CI: [0.009, 0.015])
  Relative lift:      +1.6%
  t-statistic:        7.43
  p-value:            < 0.0001   ✓

Step 5: Guardrail metrics
  Latency (p99):      +4ms   (within 10ms budget) ✓
  Error rate:         flat                         ✓
  Revenue/session:    +0.4% (95% CI: [-0.2%, +1.0%]) → NS, but positive ✓

Step 6: Novelty effect check
  NDCG@10 by week:
    Week 1: +1.9%
    Week 2: +1.6%   ← stabilizing
    Week 3: +1.5%   (extended 1 extra week to confirm)
  → Slight novelty effect; true effect ≈ +1.5%

Step 7: Decision
  Primary metric: significant, above MDE ✓
  Guardrails: all passing ✓
  Novelty: adjusted estimate still above MDE ✓
  → Ship the model
```

**Lesson:** CUPED reduced variance by 43%, making the experiment conclusive in 7 days instead of 12. The novelty check was essential — the initial 1.9% lift would have been an overestimate.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| p-value | P(data this extreme | H₀ true); not P(H₀ true) |
| Effect size | How large is the effect, independent of sample size |
| MDE | Smallest effect worth detecting; drives sample size |
| Power analysis | Calculate n before running; target power ≥ 0.80 |
| Type I / II errors | False positive vs. false negative; α and β trade off |
| Confidence interval | Always report; tells you effect size uncertainty |
| Multiple testing | Bonferroni (FWER) or BH (FDR) correction required |
| P-hacking | Peeking invalidates tests; use sequential testing instead |
| SRM | Always check split ratio; SRM invalidates the experiment |
| CUPED | Pre-experiment covariate reduces variance 20–60% |
| Sequential testing | Valid peeking via always-valid p-values or Bayesian methods |

---

## Further Reading

- Kohavi, Tang, Xu — *Trustworthy Online Controlled Experiments* (2020) — the definitive A/B testing book
- Deng et al. — *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (CUPED, WSDM 2013)
- Johari et al. — *Always Valid Inference: Continuous Monitoring of A/B Tests* (Operations Research, 2022)
- Benjamini & Hochberg — *Controlling the False Discovery Rate* (JRSS-B, 1995)
- Luedtke & Van der Laan — *Statistical Inference for Treatment Rules in Studies* (2016)
- Imbens & Rubin — *Causal Inference for Statistics, Social, and Biomedical Sciences* (2015)

---

*Next: Chapter 19 — Distribution Shift & Monitoring*
