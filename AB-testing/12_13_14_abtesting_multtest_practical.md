# 🧪 12, 13, 14 — A/B Testing, Multiple Testing & Practical Significance

---

# PART A: A/B Testing — The Applied Framework

## What is A/B Testing?

An **A/B test** is a controlled experiment where users are randomly assigned to two (or more) variants to measure the causal effect of a change on a metric. It is the operational expression of hypothesis testing in product development.

```
Users Arrive
     ↓
Random Assignment (50/50)
    /        \
Control (A)  Variant (B)
(old button)  (new button)
    ↓              ↓
Metric measured: conversion rate, CTR, revenue, etc.
     ↓
Statistical test to detect difference
```

---

## Mapping Hypothesis Testing to A/B Tests

| Hypothesis Testing Concept | A/B Test Equivalent |
|---|---|
| H₀ | No difference between A and B |
| H₁ | Variant B changes the metric |
| Test statistic | Z or t statistic for proportion/mean difference |
| α | Significance level (usually 0.05) |
| Power | Probability of detecting a real improvement |
| Effect size | Minimum Detectable Effect (MDE) |
| p-value | Probability of this gap if A=B |

---

## Key Metrics in A/B Testing

### Binary Metrics (Proportions)

- **Click-Through Rate (CTR):** `clicks / impressions`
- **Conversion Rate:** `conversions / sessions`
- **Retention Rate:** `users retained / users exposed`

**Test:** Two-proportion Z-test

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}}$$

### Continuous Metrics (Means)

- **Revenue per user**
- **Session duration**
- **Pages per session**

**Test:** Two-sample t-test (Welch's)

$$t = \frac{\bar{x}_B - \bar{x}_A}{\sqrt{s_A^2/n_A + s_B^2/n_B}}$$

---

## Sample Size Calculation

Before running the test, determine how many users you need:

$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \cdot (p_A(1-p_A) + p_B(1-p_B))}{(p_B - p_A)^2}$$

### Practical Example

> Baseline CTR: $p_A = 0.05$ (5%)
> MDE: detect a 20% relative lift → $p_B = 0.06$ (6%)
> $\alpha = 0.05$, Power = 80% ($\beta = 0.20$)

$$n = \frac{(1.96 + 0.84)^2 \times (0.05 \times 0.95 + 0.06 \times 0.94)}{(0.06-0.05)^2}$$

$$= \frac{7.84 \times (0.0475 + 0.0564)}{0.0001} = \frac{7.84 \times 0.1039}{0.0001} \approx 8146 \text{ per group}$$

You need ~8,146 users per variant (16,292 total).

---

## A/B Testing Lifecycle

```
1. HYPOTHESIS FORMATION
   "We believe adding social proof will increase checkout CTR by ≥15%"
   
2. METRIC SELECTION
   Primary: checkout CTR
   Guardrail: bounce rate, page load time
   
3. SAMPLE SIZE CALCULATION
   Based on baseline, MDE, α, and power
   
4. RANDOMIZATION
   User-level (not session-level) to prevent carryover
   
5. EXPERIMENT RUNNING
   Do NOT peek! Run until planned n is reached
   
6. ANALYSIS
   Compute test statistic, p-value, CI
   Check guardrail metrics
   
7. DECISION
   Ship / No-ship / Run longer / Investigate
```

---

## Practical Pitfalls in A/B Testing

### 1. 🕵️ The Novelty Effect
New features get initial engagement boost just because they're new. Run test long enough to see steady-state behavior (typically 2+ weeks for returning users).

### 2. 📅 Seasonality
Don't run tests during major holidays, sales events, or product launches — confounds the results.

### 3. 🌐 Network Effects (SUTVA Violation)
If users interact with each other (social networks, marketplaces), the treatment can "spill over" to the control group. Solution: cluster randomization (randomize by unit with no cross-group interactions).

### 4. 🔁 Carryover Effects
If users experience both conditions over time (A-B-A design), prior exposure can influence current behavior. Prefer parallel designs.

### 5. ⚠️ Simpson's Paradox
Aggregate results can reverse within subgroups.

> Example: Variant B "wins" overall, but loses in both Mobile and Desktop segments separately because Mobile users are overrepresented in variant B.

Always segment and sanity-check results.

### 6. 📊 Multiple Metrics
Testing 10 metrics → 1 expected false positive at α=0.05. Designate **one primary metric** pre-test. Others are secondary/diagnostic.

---

---

# PART B: The Multiple Testing Problem

## Why It Matters

Every additional hypothesis test you run at α = 0.05 gives a 5% chance of a false positive, **independent of whether H₀ is true**. Running multiple tests without correction inflates the error rate.

### Family-Wise Error Rate (FWER)

$$FWER = P(\text{at least one false positive among } m \text{ tests})$$

$$FWER = 1 - (1-\alpha)^m$$

| # Tests (m) | FWER at α=0.05 |
|---|---|
| 1 | 5% |
| 5 | 23% |
| 10 | 40% |
| 20 | 64% |
| 100 | 99.4% |

This is the **multiple comparisons problem**.

---

## P-Hacking

**P-hacking** is the practice of selectively reporting or manipulating analyses to get p < 0.05. Forms include:

1. **Outcome switching**: Testing many metrics and reporting the one with p < 0.05
2. **Optional stopping**: Peeking at results and stopping when significant
3. **Subgroup mining**: Running tests in many subgroups until one "works"
4. **One vs. two-tailed switching**: Choosing tails based on the data direction
5. **Removing outliers selectively**: Until significance is reached

> **Key insight:** If you try enough things, you will get a false positive. A genuine result should survive pre-registration and correction for multiple comparisons.

---

## Bonferroni Correction

The simplest method: divide α by the number of tests.

$$\alpha_{adjusted} = \frac{\alpha}{m}$$

For $m = 10$ tests at $\alpha = 0.05$:

$$\alpha_{adjusted} = \frac{0.05}{10} = 0.005$$

**Pros:** Simple, controls FWER strictly.
**Cons:** Very conservative — increases Type II errors (misses real effects). Assumes tests are independent.

---

## False Discovery Rate (FDR) — Benjamini-Hochberg

A less conservative alternative. Instead of controlling "any false positive," control the **proportion of false positives among all rejections**.

$$FDR = E\left[\frac{\text{False positives}}{\text{Total rejections}}\right]$$

### Benjamini-Hochberg (BH) Procedure

1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find the largest $k$ such that:
$$p_{(k)} \leq \frac{k}{m} \cdot q$$
Where $q$ is the desired FDR level (e.g., 0.05).
3. Reject all null hypotheses for tests $1, 2, \ldots, k$.

### Example

> 6 tests, q = 0.05:

| Rank (k) | p-value | $k/m \times q = k/6 \times 0.05$ | Reject? |
|---|---|---|---|
| 1 | 0.001 | 0.008 | ✅ (0.001 < 0.008) |
| 2 | 0.010 | 0.017 | ✅ (0.010 < 0.017) |
| 3 | 0.025 | 0.025 | ✅ (0.025 ≤ 0.025) |
| 4 | 0.040 | 0.033 | ❌ (0.040 > 0.033) — Stop |
| 5 | 0.080 | 0.042 | ❌ |
| 6 | 0.150 | 0.050 | ❌ |

Reject tests 1, 2, 3.

**BH is standard in genomics and large-scale testing (e.g., testing 20,000 genes).**

---

## Bonferroni vs. BH

| | Bonferroni | BH (FDR) |
|---|---|---|
| Controls | FWER | FDR |
| Strictness | Very conservative | Less conservative |
| Power | Lower | Higher |
| Best for | Few tests, high stakes | Many tests, discovery setting |
| Used in | Clinical trials | Genomics, A/B testing at scale |

---

---

# PART C: Statistical vs. Practical Significance

## The Core Distinction

| | Statistical Significance | Practical Significance |
|---|---|---|
| Answers | "Is the effect real?" | "Is the effect *meaningful*?" |
| Depends on | Sample size, variance | Effect magnitude, business context |
| Measured by | p-value | Effect size, absolute lift, ROI |

---

## The Sample Size Problem

With infinite data, any non-zero effect becomes statistically significant:

$$Z = \frac{\delta}{\sigma / \sqrt{n}} \to \infty \text{ as } n \to \infty$$

At n = 10,000,000 users, a CTR improvement of 0.00001% will have p < 0.001 — but is completely irrelevant.

### Real Example

> Website redesign test:
> - Old CTR: 4.0000%
> - New CTR: 4.0001%
> - n = 50 million users
> - p = 0.0001 ✅ statistically significant
> - Absolute lift: 0.0001% = ~50 additional clicks per day
> - Revenue impact: $0.10/day
> - Cost to ship: $200,000 engineering time
> **→ Not practically significant.** Statistical significance ≠ "worth shipping."

---

## Effect Size Measures

### Cohen's d (for means)

$$d = \frac{\mu_1 - \mu_2}{\sigma_{pooled}}, \quad \sigma_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$$

| d | Interpretation |
|---|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

### Cohen's h (for proportions)

$$h = 2\arcsin(\sqrt{p_1}) - 2\arcsin(\sqrt{p_2})$$

### Relative Lift (A/B testing standard)

$$\text{Lift} = \frac{p_B - p_A}{p_A}$$

Always report absolute AND relative lift. A 100% relative improvement sounds impressive; if baseline is 0.001%, the absolute lift is 0.001%.

---

## The Complete Reporting Framework

For any test result, always report:

```
1. Test statistic and p-value
2. Effect size (Cohen's d, absolute lift, relative lift)
3. Confidence interval for the effect
4. Power / sample size
5. Business context: What does this effect mean in $, users, or UX?
```

---

## 💬 Interview Questions (Sections 12–14)

**Q1: An A/B test shows p = 0.03 with a 0.1% absolute lift in conversion. Should you ship it?**

> A: Not automatically. At 0.03, it's statistically significant, but the business question is: What's the 0.1% lift worth? Calculate the revenue impact: if you have 100k users/day at $50 average order value, 0.1% is 100 extra conversions × $50 = $5,000/day. Annualized that's $1.8M. Now weigh against: engineering cost to maintain, impact on guardrail metrics, confidence interval (what's the lower bound?), and whether this generalizes beyond the test period. If all checks out, ship.

**Q2: How do you handle multiple metrics in an A/B test?**

> A: Pre-specify one **primary metric** (the North Star metric tied to the hypothesis) and a few secondary/diagnostic ones and guardrail metrics. Apply Bonferroni or BH correction if testing multiple primary hypotheses. Avoid fishing through 50 metrics post-hoc — the rate of false positives explodes. If a metric you didn't predict comes out significant, treat it as exploratory, not confirmatory — plan a follow-up test.

**Q3: Your experiment shows p = 0.049. A colleague wants to run it for one more week "just to be sure." Is this valid?**

> A: No — this is optional stopping / "peeking," a form of p-hacking. Once your pre-specified sample size is reached, the result stands. Running one more week after seeing p = 0.049 is cherry-picking a stopping point. If p goes to 0.04, you can't claim a pre-planned decision was made. Use sequential testing methods (e.g., sequential probability ratio test, alpha-spending functions) if you need valid peeking.

**Q4: Explain the difference between statistical and practical significance for a PM audience.**

> A: Statistical significance tells you the effect is *real* — not just random noise in the data. Practical significance tells you the effect is *meaningful* — worth acting on. You can have one without the other. Very large samples make tiny, irrelevant effects statistically significant. Very small samples may not detect large, important effects. Both together make a strong case: "The effect is real AND meaningful enough to ship."

---

## 🚨 Common Pitfalls Summary

1. **Not planning sample size before the experiment** — underpowered or overpowered
2. **Peeking at results and stopping early** — inflates false positives
3. **Testing many metrics and reporting the best one** — p-hacking
4. **Confusing statistical and practical significance** — classic interview trap
5. **Ignoring guardrail metrics** — a feature might improve CTR while breaking session depth
6. **Not accounting for network effects** — A/B contamination in social products
7. **Treating "no significant difference" as "proven equal"** — underpowered ≠ no effect

---

*← [09-11 — Chi-Square/ANOVA/Non-Param](09_10_11_chisq_anova_nonparam.md) | [15 — Interview Traps →](15_interview_traps.md)*
