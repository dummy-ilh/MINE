# 🎚️ 05 — Significance Level (α) & 06 — Confidence Intervals

---

# PART A: Significance Level (α)

## Definition

The **significance level** $\alpha$ is the maximum probability of a Type I error you are willing to tolerate — the threshold for deciding a result is "statistically significant."

$$\alpha = P(\text{Reject } H_0 \mid H_0 \text{ is true})$$

**Decision rule:**
$$\text{Reject } H_0 \text{ if } p\text{-value} < \alpha$$

### Common Values

| α | Critical Z (two-tailed) | Use Case |
|---|---|---|
| 0.10 | ±1.645 | Exploratory, low-stakes |
| **0.05** | **±1.960** | **Industry standard default** |
| 0.01 | ±2.576 | Higher-stakes decisions |
| 0.001 | ±3.291 | Medical trials, genomics |
| $5 \times 10^{-8}$ | ~±5.45 | Genome-wide association studies |

---

## How to Choose α: Business Implications

The choice of α is **a business decision**, not a statistical one. It encodes the cost of false positives.

### Framework: Cost-Benefit Analysis

$$\alpha^* = \arg\min \left[ \alpha \cdot C_{\text{FP}} + \beta(\alpha) \cdot C_{\text{FN}} \right]$$

Where $C_{\text{FP}}$ = cost of false positive, $C_{\text{FN}}$ = cost of false negative.

### Domain-Specific Guidance

| Domain | Recommended α | Reasoning |
|---|---|---|
| Drug approval (safety) | 0.001 | FP = patients harmed by useless drug |
| Cancer screening | 0.10 | FN = missed cancer (worse) |
| A/B testing (feature launch) | 0.05 | Balanced, industry standard |
| Early-stage exploration | 0.10 | More discoveries, more follow-up |
| Multiple simultaneous tests | 0.05/k (Bonferroni) | Control family-wise error rate |
| Financial fraud detection | 0.01 | False alarms are expensive |

### ⚠️ α Is Set Before Seeing Data

Setting α after seeing results (e.g., "my p-value is 0.03, so I'll use α = 0.05") is circular and invalid. Pre-specify α in the study design.

---

## The α/2 Split for Two-Tailed Tests

For a two-tailed test at level α, the rejection region is:

$$\text{Reject if } |Z| > z_{\alpha/2}$$

At $\alpha = 0.05$: critical value is $z_{0.025} = 1.96$ on each side.

```
       Reject              Accept              Reject
       region              region              region
  ─────────|──────────────────────────────────|─────────
         -1.96                                +1.96
          α/2 = 2.5%                          α/2 = 2.5%
```

---

## Interview Questions (Part A)

**Q1: Why is 0.05 the standard α? Is it actually the right choice?**

> A: 0.05 was popularized by R.A. Fisher in the 1920s — largely an arbitrary convention. In practice, it's often the wrong choice. For consumer product A/B tests where moving fast matters and downside risk is low, some teams use α = 0.10. For decisions affecting user safety or involving large costs, α = 0.01 is more appropriate. The right α depends on the cost asymmetry between Type I and Type II errors.

**Q2: If you lower α from 0.05 to 0.01, what happens to power?**

> A: Power decreases (for fixed n). A lower α means the rejection region shrinks — the critical value moves further into the tail. This makes it harder to reject H₀, increasing β. To maintain power, you'd need to increase sample size.

---

---

# PART B: Confidence Intervals

## Definition

A $(1-\alpha) \times 100\%$ **confidence interval** is a range of values constructed from sample data such that, if we repeated the sampling procedure many times, $(1-\alpha) \times 100\%$ of the intervals would contain the true parameter.

$$CI = \hat{\theta} \pm z_{\alpha/2} \cdot SE(\hat{\theta})$$

For a mean with known σ:

$$CI = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

For a mean with unknown σ (use t):

$$CI = \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

---

## Visual Intuition

![Confidence interval visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Normal_distribution_and_confidence_intervals.svg/800px-Normal_distribution_and_confidence_intervals.svg.png)

Imagine 100 separate studies, each constructing a 95% CI. Approximately 95 of those intervals will capture the true mean. You don't know which ones — you just know the long-run coverage is 95%.

---

## Relationship to Hypothesis Testing

**CIs and hypothesis tests are dual frameworks:**

| Operation | Hypothesis Test | Confidence Interval |
|---|---|---|
| Test H₀: μ = μ₀ at level α | p < α → reject | μ₀ outside (1−α)CI → reject |
| Equivalent? | ✅ Yes, exactly equivalent | ✅ Yes |

### Example

Battery life: $n=36$, $\bar{x}=492$, $\sigma=24$, $\mu_0=500$

$$95\% \text{ CI} = 492 \pm 1.96 \times \frac{24}{\sqrt{36}} = 492 \pm 7.84 = [484.16, 499.84]$$

Since $\mu_0 = 500$ is **outside** the 95% CI → **Reject H₀** at α = 0.05. (Same conclusion as our Z-test with p = 0.0456.)

---

## CI Width Determinants

$$\text{Width} = 2 \cdot z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

| Factor | Effect on Width |
|---|---|
| Increase $n$ | Narrower ✅ (better precision) |
| Increase $\sigma$ | Wider (more variable data) |
| Increase confidence level (e.g., 95%→99%) | Wider (wider net = more coverage) |

---

## 🚨 CRITICAL: Interpretation Pitfalls

### ❌ Wrong Interpretation #1

> "There is a 95% probability that the true mean is between 484 and 500."

**Why wrong:** The true mean is a fixed (unknown) constant — it either is or isn't in that interval. The probability refers to the *procedure*, not a specific interval.

### ✅ Correct Interpretation

> "We used a procedure that, in repeated sampling, would capture the true mean 95% of the time. Our specific interval is [484, 500]."

OR more practically:

> "We are 95% confident the true mean is between 484 and 500 hours."

*(The word "confident" is preferred over "probability" when talking about a specific interval.)*

---

### ❌ Wrong Interpretation #2

> "95% of the data falls within the CI."

**Why wrong:** The CI is about the **parameter** (population mean), not individual data points. The interval containing 95% of data is the **prediction interval**, which is wider.

---

### Confidence Interval vs Prediction Interval

| | Confidence Interval | Prediction Interval |
|---|---|---|
| What it estimates | Population parameter (μ) | Future individual observation |
| Width | Narrower | Wider |
| Formula (normal, known σ) | $\bar{x} \pm z \cdot \sigma/\sqrt{n}$ | $\bar{x} \pm z \cdot \sigma\sqrt{1 + 1/n}$ |

---

## Bootstrap Confidence Intervals

When you can't assume normality or the distribution is complex:

```python
import numpy as np

data = np.array([...])  # your sample
n_bootstrap = 10000
boot_means = [np.mean(np.random.choice(data, len(data), replace=True))
              for _ in range(n_bootstrap)]

ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)
```

Bootstrap CIs are distribution-free and work for any statistic (median, correlation, etc.).

---

## CI in A/B Testing Context

For a two-proportion test (control $p_1$, treatment $p_2$):

$$CI_{\Delta p} = (\hat{p}_2 - \hat{p}_1) \pm z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}$$

**Business interpretation:**

If the CI for $\Delta p$ is $[0.01, 0.05]$:
- The true improvement is between 1% and 5%
- The entire CI is above 0 → statistically significant improvement
- Use the *lower bound* (1%) for conservative business planning

---

## Interview Questions (Part B)

**Q1: Explain confidence intervals to a product manager without statistics background.**

> A: Imagine you're trying to estimate your app's true conversion rate. You measured it at 4.2% from a sample. A 95% confidence interval of [3.8%, 4.6%] means: "We're pretty confident the true conversion rate sits somewhere between 3.8% and 4.6%. We might be wrong 1 in 20 times, but that's our best range estimate." It's like a net — we're 95% sure the fish (true value) is inside the net.

**Q2: A CI contains 0. What does that mean for the hypothesis test?**

> A: If the 95% CI for the difference in means contains 0, it means we fail to reject H₀: μ₁ = μ₂ at α = 0.05. There's no statistically significant difference detected. But check the CI bounds — if it's [−0.001, 0.002], that's a narrow CI excluding large effects, so the true difference is likely small. If it's [−10, 11], the CI is wide, and we're simply underpowered.

**Q3: Why is a wider CI sometimes *better*?**

> A: It's not — but it's honest. A CI that's too narrow (from underestimating SE or using wrong assumptions) gives false precision. A properly wide CI that reflects genuine uncertainty is more trustworthy. In sequential/adaptive testing, CIs can be wider by design to maintain valid coverage.

---

*← [04 — p-value](04_pvalue.md) | [07 — Z-test →](07_ztest.md)*
