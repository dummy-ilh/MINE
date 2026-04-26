# Hypothesis Testing — Complete Notes & FAANG Q&A

---

## 1. Definition & Formula

**Hypothesis testing** is a formal statistical procedure to decide whether there is enough evidence in your sample data to reject a default assumption (the null hypothesis) in favour of an alternative claim.

> **Plain English:** You start by assuming "nothing interesting is happening." You then collect data and ask: "Is my data surprising enough to disprove that assumption?" If yes — you have a finding. If no — you don't have enough evidence to claim otherwise.

> **Analogy:** It works like a court trial. The null hypothesis is "innocent until proven guilty." You need sufficient evidence (data) to convict (reject H₀). Failing to convict does not prove innocence — it just means the evidence wasn't strong enough.

---

### The Two Hypotheses

```
H₀  (Null hypothesis)        — The default assumption. No effect, no difference.
H₁  (Alternative hypothesis) — What you're trying to detect. Some effect exists.
```

**Examples:**

| Context | H₀ | H₁ |
|---------|----|----|
| A/B test | New feature has no effect on CTR | New feature changes CTR |
| Drug trial | Drug has no effect vs placebo | Drug reduces blood pressure |
| Manufacturing | Machine produces bolts of 50mm | Machine produces bolts ≠ 50mm |
| ML model | New model accuracy = old model | New model accuracy > old model |

---

### The General Formula

Every hypothesis test computes a **test statistic** of this form:

```
Test statistic = (Observed value − Expected value under H₀) / Standard Error

                       signal
              =       ────────
                        noise
```

A large test statistic means your observation is many standard errors away from what H₀ predicts — strong evidence against H₀.

**Specific formulas by test:**

```
z-test (large n or σ known):     z = (X̄ − μ₀) / (σ / √n)

t-test (small n, σ unknown):     t = (X̄ − μ₀) / (s / √n)

Proportion z-test (A/B):         z = (p̂₁ − p̂₂) / √[p̂(1−p̂)(1/n₁ + 1/n₂)]

Chi-squared test:                 χ² = Σ (Observed − Expected)² / Expected
```

---

### The Decision Rule

```
Compute test statistic  →  find p-value  →  compare to α

If  p ≤ α   →  Reject H₀    ("statistically significant")
If  p > α   →  Fail to reject H₀  ("insufficient evidence")
```

Common significance levels:

| α | Confidence | Used when |
|---|-----------|-----------|
| 0.05 | 95% | Most standard experiments |
| 0.01 | 99% | Higher-stakes decisions |
| 0.001 | 99.9% | Medical, safety-critical |
| 0.1 | 90% | Exploratory research |

---

### Key Terms at a Glance

| Term | Symbol | Meaning |
|------|--------|---------|
| Null hypothesis | H₀ | Assumption of no effect |
| Alternative hypothesis | H₁ | Claim you want to test |
| Significance level | α | Max tolerable false positive rate |
| p-value | p | P(data this extreme \| H₀ true) |
| Test statistic | z, t, χ² | Standardised signal-to-noise ratio |
| Critical value | z*, t* | Threshold for rejecting H₀ |
| Type I error | α | False positive — reject true H₀ |
| Type II error | β | False negative — miss real effect |
| Statistical power | 1 − β | Probability of detecting a real effect |
| Effect size | d, δ | Magnitude of the true difference |

---

## 2. Explanation

### The Step-by-Step Process

```
Step 1 — State hypotheses
         Define H₀ and H₁ clearly before looking at data.

Step 2 — Choose significance level (α)
         Typically 0.05. Set this BEFORE running the experiment.

Step 3 — Select the right test
         z-test, t-test, chi-squared, ANOVA — based on data type and design.

Step 4 — Compute the test statistic
         Plug your sample data into the formula.

Step 5 — Find the p-value
         Area in the tail(s) of the null distribution beyond your test statistic.

Step 6 — Make a decision
         p ≤ α → reject H₀. p > α → fail to reject H₀.

Step 7 — Report effect size + CI
         Statistical significance alone is not enough. Always report magnitude.
```

---

### One-Tailed vs Two-Tailed Tests

```
Two-tailed (H₁: μ ≠ μ₀)       — effect could go either direction
   Reject if  |z| > z*          α split equally across both tails

One-tailed right (H₁: μ > μ₀) — testing for an increase only
   Reject if  z > z*            all α in right tail

One-tailed left  (H₁: μ < μ₀) — testing for a decrease only
   Reject if  z < −z*           all α in left tail
```

> In A/B testing, **always use two-tailed** unless you have a pre-specified directional hypothesis grounded in domain knowledge. One-tailed is easier to achieve significance with — and therefore easier to misuse.

---

### Type I and Type II Errors — The Trade-off

```
                        Reality
                   H₀ True    H₀ False
              ┌──────────────────────────┐
  Reject H₀  │  Type I error │  Correct  │
  Decision   │      (α)      │  (Power)  │
             ├──────────────────────────┤
  Fail to    │   Correct     │ Type II   │
  reject H₀  │               │  error(β) │
              └──────────────────────────┘
```

- **Type I error (α):** False positive. You conclude there's an effect when there isn't. Controlled by your choice of α.
- **Type II error (β):** False negative. You miss a real effect. Controlled by sample size and effect size.
- **Power (1 − β):** Probability of correctly detecting a real effect. Typical target: 80% or 0.80.

The trade-off: lowering α (stricter threshold) reduces Type I errors but increases Type II errors. You need a larger sample size to reduce both simultaneously.

---

### Statistical Power — Deep Dive

Power depends on four factors:

```
Power  ↑  when:
  Effect size   is larger    (bigger signal is easier to detect)
  Sample size   is larger    (more data reduces noise)
  α             is larger    (looser threshold easier to cross)
  Variance      is smaller   (less noise → cleaner signal)
```

**Power formula (approximate for z-test):**

```
n = (z_α/2 + z_β)² × σ² / δ²

Where:
  z_α/2 = critical value for significance (1.96 for α=0.05 two-tailed)
  z_β   = critical value for power (0.84 for 80% power)
  σ²    = variance of the metric
  δ     = minimum detectable effect (MDE)
```

You always do power analysis **before** running an experiment to determine required sample size.

---

### The Multiple Testing Problem

Running many tests simultaneously inflates your false positive rate:

```
P(at least 1 false positive across m tests) = 1 − (1 − α)ᵐ

m =  1  →  5.0%  false positive rate
m =  5  →  22.6%
m = 10  →  40.1%
m = 20  →  64.2%
m = 50  →  92.3%
```

**Corrections:**

| Method | Adjustment | When to use |
|--------|-----------|-------------|
| Bonferroni | α_new = α / m | Conservative, few tests |
| Benjamini-Hochberg | Controls FDR (false discovery rate) | Many simultaneous tests |
| Holm-Bonferroni | Step-down, less conservative | Moderate number of tests |
| Pre-specify 1 primary metric | No correction needed | Best practice in A/B testing |

---

### Parametric vs Non-Parametric Tests

| If assumptions hold | Use parametric | Examples |
|--------------------|---------------|---------|
| Normality + continuous | z-test, t-test | Latency, revenue |
| If assumptions violated | Use non-parametric | — |
| Two groups, non-normal | Mann-Whitney U | Revenue with outliers |
| Paired, non-normal | Wilcoxon signed-rank | Before/after, skewed |
| Categorical data | Chi-squared | CTR buckets, survey |
| 3+ groups | ANOVA (parametric) or Kruskal-Wallis | Multi-arm experiments |

---

## 3. Uses & Applications

### A/B Testing at Tech Companies

The dominant use case. Every feature launch at Google, Meta, Amazon, Netflix is preceded by an experiment. Hypothesis testing determines whether observed metric changes are real or noise. The framework: pre-register H₀ and H₁, set α and MDE, run until target n, compute p-value, decide.

### Drug and Clinical Trial Approval

Regulatory agencies (FDA, EMA) require hypothesis tests showing p < 0.05 (often p < 0.01 for approval). Phase III trials are structured hypothesis tests comparing treatment vs placebo with pre-specified endpoints and sample sizes.

### Quality Control in Manufacturing

Six Sigma methodology is built on hypothesis testing. Control charts, capability studies, and process improvement all rely on tests to determine whether process changes produce statistically significant improvements in defect rates or measurements.

### Machine Learning Model Selection

- Testing whether Model B is significantly better than Model A (paired t-test on CV folds)
- Feature selection: testing whether a feature coefficient is significantly non-zero (t-test in regression)
- Checking whether training loss differences are real or due to random seed variation

### Search Engine & Recommendation Ranking

Ranking algorithm changes are validated via online controlled experiments (A/B tests) using hypothesis testing on metrics like NDCG, CTR, dwell time, and conversion before any global rollout.

### Finance & Econometrics

Testing whether a trading strategy's returns are significantly different from zero. Testing whether two assets are cointegrated. Testing whether a macro factor significantly predicts equity returns. All use hypothesis testing frameworks.

### Social Science & Policy Research

Testing whether a policy intervention (e.g. job training programme, subsidy) produced a significant improvement in outcomes vs a control group. Randomised controlled trials use hypothesis testing as their core inferential tool.

---

## 4. FAANG Interview Q&A

### Conceptual Questions

---

**Q: Explain hypothesis testing from scratch as if to a non-technical PM.**

> Imagine we're testing a new button colour. We start by assuming it makes no difference — that's our null hypothesis. We then show the old button to half our users and the new one to the other half, and measure how many click. After enough data, we ask: "If the button truly made no difference, how likely would we see this big a gap in clicks just by chance?" If the answer is "less than 5% likely," we conclude the button probably does make a difference and consider shipping it. If the gap could easily happen by chance, we don't have enough evidence to act.

---

**Q: What is the difference between H₀ and H₁? Can you reject H₁?**

> H₀ is the null — the default assumption of no effect. H₁ is the alternative — what you're trying to detect. Hypothesis testing only ever makes a decision about H₀: you either reject it or fail to reject it. You never "accept H₁" — you can only say there is sufficient evidence against H₀. You also never "accept H₀" — failing to reject it just means insufficient evidence, not proof it's true.

---

**Q: What is statistical power and why does it matter in experiment design?**

> Power (1 − β) is the probability of correctly detecting a real effect when one exists. A test with 80% power means that if the true effect is real, you'll detect it 80% of the time. Low power means you risk missing real improvements — a Type II error. Power matters because under-powered experiments waste resources and may incorrectly conclude "no effect" when the variant actually works. Before running any experiment, you must compute the required sample size to achieve adequate power for your minimum detectable effect.

---

**Q: What is the difference between statistical significance and practical significance?**

> Statistical significance (p < α) only tells you the effect is unlikely due to chance. Practical significance tells you whether the effect is large enough to matter in the real world. At massive scale, you can detect a 0.001% improvement with p < 0.0001 — but shipping a feature for that lift may not justify engineering cost, added complexity, or risk. Always pair p-values with effect sizes (absolute lift, relative lift, Cohen's d) and a business cost-benefit analysis.

---

**Q: Explain Type I and Type II errors with a real-world example.**

> Using an A/B test on a new checkout flow:
>
> **Type I error (false positive, rate = α):** The new flow has no real effect, but random chance makes it look better in our sample, and we ship it. We wasted engineering effort and potentially added complexity for nothing.
>
> **Type II error (false negative, rate = β):** The new flow genuinely reduces drop-off by 3%, but our sample was too small to detect it, so we conclude "no effect" and don't ship. We miss a real business win.
>
> The trade-off: tightening α (e.g. 0.01 instead of 0.05) reduces false positives but makes it harder to detect real effects. The fix is a larger sample size — which reduces both error types simultaneously.

---

**Q: What is the multiple comparisons problem? How do you handle it at FAANG?**

> When you run multiple hypothesis tests simultaneously (e.g. testing 20 metrics in one experiment), the probability of at least one false positive at α = 0.05 rises to ~64%. Solutions at FAANG:
> 1. **Pre-specify one primary metric** — this is the most important fix. All other metrics are secondary or exploratory with explicit caveats.
> 2. **Benjamini-Hochberg FDR control** — for exploratory analysis across many metrics, control the false discovery rate rather than the family-wise error rate.
> 3. **Bonferroni correction** — divide α by the number of tests. Very conservative, use sparingly.
> 4. **Hierarchical metric framework** — primary metric gets full α; guardrail metrics use a separate threshold; secondary metrics are directional signals only.

---

**Q: What is a p-value? State what it is and what it is NOT.**

> A p-value is the probability of observing data as extreme as yours, or more extreme, assuming H₀ is true. Formally: P(data | H₀).
>
> It is NOT:
> - The probability that H₀ is true
> - The probability that your result is due to chance
> - A measure of the size of the effect
> - Evidence that your alternative hypothesis is correct
>
> The only valid interpretation: a small p-value means your data would be very unlikely if H₀ were true. That's all.

---

**Q: What is a confidence interval and how does it relate to hypothesis testing?**

> A 95% confidence interval contains all values of the null parameter for which you would NOT reject H₀ at α = 0.05. So:
> - CI excludes zero → p < 0.05 → reject H₀
> - CI includes zero → p > 0.05 → fail to reject H₀
>
> CIs are more informative than p-values alone because they show both significance and the plausible range of effect sizes. Reporting "lift = 2.1% (95% CI: 0.4%, 3.8%)" is far more actionable than "p = 0.02."

---

### Practical / Case-Based Questions

---

**Q: You're designing an A/B test for a new search ranking algorithm. Walk me through your full hypothesis testing setup.**

> 1. **State hypotheses:** H₀: new ranking produces same NDCG/CTR as old. H₁: new ranking changes NDCG/CTR (two-tailed).
> 2. **Choose α:** 0.05 standard; consider 0.01 if ranking changes are high-risk.
> 3. **Define MDE:** What's the smallest lift worth shipping? e.g. 0.5% relative CTR improvement.
> 4. **Power analysis:** Target 80% power. Calculate n from MDE, baseline variance, α, and power. This gives experiment duration.
> 5. **Randomisation:** Randomly assign users (or queries) to control/treatment. Check for novelty effects and SRM (sample ratio mismatch).
> 6. **Run the experiment:** Do not peek. Do not stop early based on significance.
> 7. **Compute test statistic and p-value:** Two-sample z-test or t-test depending on n. Check guardrail metrics.
> 8. **Report:** Effect size, CI, p-value, guardrail metric health. Make a ship/no-ship decision including business judgment.

---

**Q: Your experiment shows p = 0.03 after just 2 days. Your PM wants to ship. What do you say?**

> I'd advise against shipping yet for several reasons:
> 1. **Peeking bias:** Stopping early because p crossed 0.05 inflates the false positive rate. The actual Type I error rate is much higher than 5% when you peek repeatedly.
> 2. **Underpowered duration:** Two days may miss weekly seasonality (weekend vs weekday behaviour differs), novelty effects, and long-term user adaptation.
> 3. **Sample size check:** Did we reach the pre-specified n from our power analysis? If not, the result is unreliable.
> 4. **Guardrail metrics:** We need to verify no secondary metrics degraded.
> 5. **Fix:** Either use a sequential testing framework (always-valid p-values) that allows early stopping with controlled error rates, or commit to running the full planned duration.

---

**Q: How would you handle hypothesis testing when your metric is not normally distributed?**

> Several approaches depending on the situation:
> 1. **Rely on CLT** — for large n (typically 10k+ per arm), the sampling distribution of the mean normalises regardless of the underlying distribution. z-test or t-test on the mean remains valid.
> 2. **Transform the data** — log-transform right-skewed metrics (revenue, session length) before testing.
> 3. **Winsorise** — cap extreme values at the 95th or 99th percentile to reduce variance without removing data.
> 4. **Bootstrap/permutation test** — resample the data empirically to build the null distribution. Makes no distributional assumption.
> 5. **Non-parametric test** — Mann-Whitney U for two groups, Wilcoxon signed-rank for paired data. These test distributional shift rather than mean difference.
> 6. **Change the metric** — test median instead of mean, which is more robust to outliers.

---

**Q: What is a sample ratio mismatch (SRM) and why does it matter?**

> SRM occurs when the actual ratio of users assigned to control vs treatment differs significantly from the intended ratio (e.g. you aimed for 50/50 but got 52/48). This is detected by a chi-squared test on the assignment counts.
>
> SRM matters because it indicates something went wrong in the randomisation or logging pipeline — a bug in the experiment infrastructure, bot traffic, or differential attrition between arms. Any metric results from an SRM experiment are unreliable and should be discarded until the root cause is fixed. At FAANG, SRM detection is always the first check before analysing experiment results.

---

**Q: Walk me through how you'd use hypothesis testing to decide whether to roll out a new ML model.**

> 1. **Offline evaluation first:** Paired t-test or McNemar's test on held-out data comparing Model A vs Model B accuracy/F1/AUC across cross-validation folds. This gives a signal but doesn't reflect production distribution.
> 2. **Shadow mode / canary:** Run new model in shadow — it scores but doesn't serve. Verify predictions are sane and latency is acceptable.
> 3. **Online A/B test:** Randomly route a % of traffic (e.g. 10%) to new model. Primary metric: business outcome (CTR, conversion, revenue). Secondary metrics: engagement, latency, error rate.
> 4. **Hypothesis test on online metrics:** Two-sample z-test (proportions) or t-test (continuous) depending on metric. Set α = 0.05, target 80%+ power, pre-specify MDE.
> 5. **Ship criteria:** p < α on primary metric AND no guardrail metric degradation AND effect size justifies operational complexity of maintaining a new model.

---

**Q: What is the difference between a one-tailed and two-tailed test? When would you use each?**

> A **two-tailed test** tests whether an effect exists in either direction (H₁: μ ≠ μ₀). It splits α across both tails (e.g. 2.5% each for α = 0.05). Use this by default in A/B testing — you want to detect both improvements and regressions.
>
> A **one-tailed test** tests for an effect in a specific direction only (H₁: μ > μ₀ or μ < μ₀). All of α goes into one tail, making it easier to achieve significance. Use only when: the direction is pre-specified from strong domain knowledge before seeing data, and a result in the other direction would be treated identically to no effect. Never switch from two-tailed to one-tailed after seeing data — this is p-hacking.

---

**Q: What is Bayesian hypothesis testing and how does it differ from frequentist? When would you prefer it?**

> **Frequentist** testing asks: "How likely is this data if H₀ is true?" (p-value). It cannot directly answer "What is the probability my variant is better?"
>
> **Bayesian** testing asks: "Given my data and prior beliefs, what is the probability that the variant is better?" It outputs P(variant > control | data) — directly interpretable by stakeholders.
>
> Prefer Bayesian when:
> - You want to communicate results as "94% probability the variant wins" — clearer for non-technical PMs
> - You need to stop experiments early with controlled error rates (posterior probability thresholds)
> - You have strong prior knowledge about expected effect sizes
> - You're running many low-traffic experiments where frequentist power requirements are prohibitive
>
> Companies like Airbnb, Booking.com, and VWO have adopted Bayesian A/B testing frameworks for these reasons. The trade-off: results depend on prior choice and are harder to audit for regulatory purposes.

---

*Tags: Statistics · Hypothesis Testing · A/B Testing · p-Value · Power · Type I Error · Type II Error · FAANG Prep · Experimentation*
