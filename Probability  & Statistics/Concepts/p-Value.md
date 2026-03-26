# p-Value 

---

## 1. Definition & Formula

A **p-value** is the probability of observing a test statistic **as extreme as, or more extreme than**, the one computed from your sample data — *assuming the null hypothesis is true*.

> **Plain English:** If there were truly no effect (null hypothesis), how likely is it that you'd see data this extreme just by random chance? A small p-value means "this result would be very unlikely by chance alone — something real might be going on."

> **Critical reminder:** The p-value is **not** the probability that the null hypothesis is true. It is the probability of your *data* given the null hypothesis — P(data | H₀), not P(H₀ | data).

---

### Formula

The p-value depends on the test used. The general structure is:

```
p-value = P( |T| ≥ |t_observed| | H₀ is true )
```

Where **T** is the test statistic and **t_observed** is the value computed from your sample.

#### For a z-test (large samples, known σ):

```
z = (X̄ - μ₀) / (σ / √n)

Two-tailed:  p = 2 × P(Z ≥ |z|)
Left-tailed: p = P(Z ≤ z)
Right-tailed:p = P(Z ≥ z)
```

#### For a t-test (small samples, unknown σ):

```
t = (X̄ - μ₀) / (s / √n)

p = P(T_{n-1} ≥ |t|)   [two-tailed, multiply by 2]
```

#### For a proportion z-test (A/B testing):

```
z = (p̂₁ - p̂₂) / √[ p̂(1 - p̂)(1/n₁ + 1/n₂) ]

where p̂ = (x₁ + x₂) / (n₁ + n₂)   [pooled proportion]
```

---

### Key Terms

| Term | Symbol | Meaning |
|------|--------|---------|
| Null hypothesis | H₀ | Assumption of no effect / no difference |
| Alternative hypothesis | H₁ or Hₐ | What you're trying to detect |
| Significance level | α | Threshold for rejecting H₀ (commonly 0.05) |
| p-value | p | Probability of observed data under H₀ |
| Test statistic | z, t, χ² | Standardized measure of observed effect |
| Type I error | α | Rejecting H₀ when it's actually true (false positive) |
| Type II error | β | Failing to reject H₀ when it's actually false (false negative) |
| Statistical power | 1 − β | Probability of correctly detecting a real effect |

---

### Decision Rule

```
If p ≤ α  →  Reject H₀  (result is "statistically significant")
If p > α  →  Fail to reject H₀  (insufficient evidence)
```

Common significance levels: **α = 0.05** (95% confidence), **α = 0.01** (99% confidence), **α = 0.001** (99.9% confidence — used in high-stakes decisions).

---

## 2. Explanation

### The Thought Experiment

Hypothesis testing works by contradiction. You assume H₀ is true, then ask: *"If this were true, how surprising is my data?"*

- **Large p-value (e.g. 0.6):** Your data is very consistent with H₀. No reason to doubt it.
- **Small p-value (e.g. 0.02):** Your data would be very unlikely if H₀ were true. Either H₀ is false, or you got unlucky — and p = 0.02 means only a 2% chance it's just luck.

### Visualizing the p-value

The p-value is the **area in the tail(s)** of the null distribution beyond your observed test statistic:

```
Two-tailed test (most common in A/B testing):

       H₀ distribution
            ____
           /    \
  p/2 →  /      \  ← p/2
_________|        |_________
   -|z|      0      +|z|
   ←tail→         ←tail→
   
p = area of both shaded tails combined
```

- **Two-tailed:** Tests if effect is different in either direction (≠)
- **One-tailed:** Tests if effect is specifically larger (>) or smaller (<)

### What p-value Does NOT Mean

These are extremely common misconceptions — interviewers love to probe these:

| Misconception | Reality |
|---------------|---------|
| p = 0.05 means 5% chance H₀ is true | p-value says nothing about P(H₀) — that requires Bayesian methods |
| p < 0.05 means the result is practically important | Statistical significance ≠ practical significance. A huge sample can make tiny, meaningless effects significant |
| p > 0.05 means H₀ is true | It only means insufficient evidence to reject H₀. Absence of evidence is not evidence of absence |
| p = 0.04 is "better" than p = 0.06 | The 0.05 threshold is arbitrary. Both values carry uncertainty |
| Smaller p = larger effect | p depends on both effect size *and* sample size. Large n inflates significance |

### p-value and Sample Size

This is critical for FAANG interviews. The p-value is **sensitive to sample size**:

```
z = effect_size × √n / σ
```

With billions of users, even a 0.001% difference in conversion rate produces p ≈ 0. Every experiment becomes "statistically significant." This is why **practical significance (effect size)** matters more than p-values at scale.

### Relationship to Confidence Intervals

```
If a 95% CI does not contain the null value (e.g. 0 for differences):
  → p < 0.05  (statistically significant at α = 0.05)

If a 95% CI contains the null value:
  → p > 0.05  (fail to reject H₀)
```

Confidence intervals are generally more informative than p-values alone — they tell you both the significance *and* the magnitude of the effect.

---

## 3. Uses & Applications

### A/B Testing (Core Use Case at FAANG)

The p-value is the engine of every online experiment. You test whether a new feature changes a metric (CTR, conversion rate, revenue per user). You set α = 0.05 (or stricter for high-stakes decisions), run the experiment until reaching your pre-specified sample size, then compute the p-value.

If p < α, you have statistical evidence the variant performs differently. But at Google/Meta/Amazon scale, you always pair this with **minimum detectable effect (MDE)** and **practical significance** checks.

### Clinical Trials & Medical Research

p-values determine whether a drug treatment shows statistically significant improvement over a placebo. Regulatory bodies (FDA) require p < 0.05 (often p < 0.01) before approval. Crucially, multiple trial phases are used to avoid false positives from repeated testing.

### Quality Control

In manufacturing, hypothesis tests check whether a production process has drifted from specifications. A p-value below threshold triggers an investigation or shutdown of the line.

### Machine Learning Model Evaluation

- **Feature selection:** Testing whether a feature's coefficient is significantly different from zero (in linear/logistic regression, software outputs p-values per coefficient)
- **Model comparison:** Statistical tests (e.g. McNemar's test) compare whether two models differ significantly in performance
- **Chi-squared tests:** Test whether categorical features are independent of the target

### Finance

Backtesting trading strategies: p-values test whether a strategy's returns are statistically different from zero (i.e. not just random noise). Also used in risk models and factor analysis.

### Search & Ranking (Google, Amazon)

p-values validate whether changes to ranking algorithms produce significantly better user satisfaction scores (measured via click-through rate, dwell time, purchases) before global rollout.

---

## 4. FAANG Interview Q&A

### Conceptual Questions

---

**Q: What is a p-value? Explain it simply.**

> A p-value is the probability of observing your data — or something more extreme — assuming the null hypothesis is true. If p = 0.03, it means there's only a 3% chance you'd see this result by random chance alone if H₀ were true. A small p-value is evidence *against* H₀, not proof that your hypothesis is correct.

---

**Q: What is the most common misconception about p-values?**

> The most common one: "p = 0.05 means there's a 5% chance the null hypothesis is true." This is wrong. The p-value is P(data | H₀), not P(H₀ | data). To get P(H₀ | data), you'd need Bayes' theorem and a prior on H₀. The p-value tells you how surprising your data is under H₀ — nothing more.

---

**Q: What is the difference between statistical significance and practical significance?**

> Statistical significance (p < α) just means the observed effect is unlikely due to chance. Practical significance means the effect is large enough to matter in the real world. With millions of users, you can detect a 0.001% improvement with p < 0.0001 — but shipping a feature for a 0.001% lift may not be worth the engineering cost. Always report effect size (e.g. Cohen's d, relative lift) alongside p-values.

---

**Q: What is a Type I error and Type II error? How do they relate to p-values?**

> - **Type I error (α):** Rejecting H₀ when it's true — a false positive. The significance level α is the probability you're willing to tolerate for this. Setting α = 0.05 means you accept a 5% chance of a false positive.
> - **Type II error (β):** Failing to reject H₀ when it's false — a false negative. Related to statistical power (1 − β). Lower α (stricter threshold) reduces Type I errors but increases Type II errors. There's an inherent trade-off.

---

**Q: What happens to p-values as sample size increases?**

> p-values decrease as n increases, even if the true effect size stays the same. This is because the test statistic z = effect × √n / σ grows with n. At FAANG scale with billions of users, nearly every experiment reaches p < 0.001. This is why you must define a **minimum detectable effect (MDE)** before the experiment and focus on effect size, not just p-values.

---

**Q: What is the difference between a one-tailed and two-tailed test? Which should you use?**

> - **Two-tailed:** Tests whether the effect is different in *either* direction (H₁: μ ≠ μ₀). Splits α across both tails.
> - **One-tailed:** Tests whether the effect is specifically larger or smaller (H₁: μ > μ₀ or μ < μ₀). All α in one tail.
>
> In A/B testing, **two-tailed is almost always preferred** — you want to detect if a feature helps *or* hurts. One-tailed tests are easier to "pass" and are sometimes used inappropriately to squeeze significance out of weak results. Use one-tailed only when the direction is pre-specified by strong domain knowledge.

---

**Q: What is the p-value under the null hypothesis? What distribution does it follow?**

> Under H₀, the p-value follows a **Uniform(0, 1) distribution**. This means if H₀ is true, you'll see p < 0.05 exactly 5% of the time by chance — which is the definition of the Type I error rate. This fact is exploited in **p-value histograms** for multiple testing analysis: a uniform distribution across [0,1] suggests no true effects; a spike near 0 suggests real signals exist.

---

### Practical / Case-Based Questions

---

**Q: You run an A/B test and get p = 0.04. Your PM wants to ship the feature. What do you say?**

> p = 0.04 clears the α = 0.05 threshold, but I'd check several things before recommending a ship:
> 1. **Was the sample size pre-specified?** If we stopped early because p crossed 0.05, we have peeking bias (inflated false positive rate).
> 2. **What's the effect size?** A statistically significant but tiny effect may not justify the engineering cost.
> 3. **Are there guardrail metrics?** Did any important secondary metrics (latency, revenue) degrade?
> 4. **Was this a single test or one of many?** Multiple comparisons inflate false positive risk.
> 5. **Is the result reproducible?** Can we validate on a holdout?

---

**Q: You run 100 A/B tests simultaneously. How many do you expect to show p < 0.05 by chance?**

> If all null hypotheses are true (no real effects), you'd expect **5 false positives** — 5% of 100 tests. This is the **multiple comparisons problem**. Corrections include:
> - **Bonferroni correction:** Divide α by the number of tests (α = 0.05/100 = 0.0005). Very conservative.
> - **Benjamini-Hochberg (FDR control):** Controls the False Discovery Rate — the expected proportion of significant results that are false positives. More powerful than Bonferroni, preferred in practice.
> - **Sequential testing / always-valid p-values:** Used at companies like Airbnb and Netflix for continuous monitoring.

---

**Q: What is p-hacking and how do you prevent it?**

> p-hacking is the practice of manipulating analysis choices until p < 0.05 — by trying multiple metrics, stopping early when significant, segmenting until a subgroup shows significance, or adding/removing covariates. It inflates the false positive rate and produces unreliable results.
>
> Prevention: (1) **Pre-register** your hypothesis, primary metric, and sample size before running the experiment. (2) Never peek at results and stop early based on significance. (3) Limit the number of metrics tested. (4) Use corrections for multiple comparisons. (5) Require replication for important decisions.

---

**Q: Your experiment shows p = 0.20. Your PM says "so the feature has no effect." How do you respond?**

> p = 0.20 means we *failed to reject* H₀ — it does not mean the effect is zero. Two scenarios can produce a large p-value: (1) there truly is no effect, or (2) the experiment was underpowered — sample size was too small to detect a real effect. I'd check the **confidence interval**: if it includes both zero and practically meaningful effect sizes, the experiment is inconclusive, not negative. We may need a larger sample or a longer run.

---

**Q: How do you handle p-values for revenue metrics, which are heavily skewed?**

> Revenue per user is typically right-skewed (most users spend $0, a few spend a lot). The standard z-test may not be valid with small samples. Approaches:
> 1. **Check if CLT has kicked in** — with large n (typically 10k+), the sampling distribution of the mean normalizes regardless of skew.
> 2. **Winsorize or cap** extreme values to reduce variance and get reliable p-values.
> 3. **Use a non-parametric test** (Mann-Whitney U) that doesn't assume normality.
> 4. **Bootstrap the p-value** — permutation tests make no distributional assumptions.
> 5. **Log-transform** revenue and test on the log scale, then back-transform the effect.

---

**Q: At Google/Meta scale, every experiment is "significant." How do you make decisions?**

> At massive scale, statistical significance is nearly guaranteed. The decision framework shifts to:
> 1. **Practical significance:** Is the effect large enough to matter? Define MDE (minimum detectable effect) before the experiment based on business value.
> 2. **Effect size metrics:** Report relative lift (%) and absolute change, not just p-values.
> 3. **Guardrail metrics:** Check that no critical metrics (latency, crash rate, revenue) were harmed — even if the primary metric improved.
> 4. **Cost-benefit analysis:** Engineering cost, maintenance burden, and risk vs. the measured gain.
> 5. **Novelty effects:** Is the lift real or just users reacting to something new? Run longer or use holdout groups.

---

**Q: What is the relationship between p-values and confidence intervals?**

> They are mathematically equivalent for the same test. If a 95% confidence interval for the difference **excludes zero**, then p < 0.05. If it **includes zero**, then p > 0.05. CIs are generally more informative because they show:
> - The direction and magnitude of the effect
> - The precision of the estimate (wide CI = high uncertainty)
> - The range of plausible true values
>
> At FAANG, reporting the CI alongside the p-value is considered best practice over reporting p alone.

---

**Q: What is the difference between p-values and Bayesian approaches? When would you prefer Bayesian?**

> The frequentist p-value answers: *"How unlikely is this data if H₀ is true?"* The Bayesian approach answers: *"What is the probability that H₁ is true, given my data?"* — which is often what decision-makers actually want.
>
> Prefer Bayesian when:
> - You want to quantify P(variant is better) directly (intuitive for business stakeholders)
> - You have strong prior knowledge about the effect size
> - You need to stop experiments early without inflating false positive rates
> - You want to report "there's a 94% probability the variant lifts conversion" rather than "p = 0.03"
>
> Companies like Airbnb, Booking.com, and VWO have moved toward Bayesian A/B testing frameworks for these reasons.

