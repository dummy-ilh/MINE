# 🏆 16 — Master Cheat Sheet, Decision Tree & FAANG Questions

---

# 📋 PART A: Complete Cheat Sheet

## Core Definitions

| Concept | Definition | Symbol |
|---|---|---|
| Null Hypothesis | Default assumption (no effect) | H₀ |
| Alternative Hypothesis | What you're trying to detect | H₁ |
| Significance Level | Max tolerable Type I error | α |
| p-value | P(data this extreme \| H₀ true) | p |
| Type I Error | Reject true H₀ (false positive) | α |
| Type II Error | Fail to reject false H₀ (false negative) | β |
| Power | P(reject H₀ \| H₀ false) | 1−β |
| Effect Size | Magnitude of true effect | d, h, etc. |
| Standard Error | SD of the sampling distribution | SE |
| Test Statistic | Standardized distance from H₀ | Z, t, F, χ² |
| Confidence Interval | Range containing true param. at (1−α)×100% | CI |

---

## Key Formulas at a Glance

### Test Statistics

$$Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} \quad \text{(Z-test, known }\sigma\text{)}$$

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \quad df = n-1 \quad \text{(one-sample t)}$$

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}} \quad \text{(Welch's t)}$$

$$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}} \quad \text{(two-proportion Z)}$$

$$\chi^2 = \sum \frac{(O-E)^2}{E} \quad \text{(chi-square)}$$

$$F = \frac{MS_{between}}{MS_{within}} \quad \text{(ANOVA)}$$

### Confidence Intervals

$$CI_\mu = \bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \quad \text{(known }\sigma\text{)}$$

$$CI_\mu = \bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}} \quad \text{(unknown }\sigma\text{)}$$

$$CI_p = \hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

### Sample Size

$$n = \left(\frac{(z_{\alpha/2} + z_\beta)\sigma}{\delta}\right)^2 \quad \text{(means)}$$

$$n = \frac{(z_{\alpha/2} + z_\beta)^2(p_1(1-p_1) + p_2(1-p_2))}{(p_1-p_2)^2} \quad \text{(proportions)}$$

### Critical Z-Values

| α | $z_{\alpha/2}$ (two-tailed) | $z_\alpha$ (one-tailed) |
|---|---|---|
| 0.10 | 1.645 | 1.282 |
| 0.05 | 1.960 | 1.645 |
| 0.01 | 2.576 | 2.326 |
| 0.001 | 3.291 | 3.090 |

### Power

$$\text{Power} = \Phi\left(\frac{|\mu_1 - \mu_0|}{\sigma/\sqrt{n}} - z_{\alpha/2}\right)$$

Standard: Power ≥ 0.80

### Multiple Testing

$$\text{Bonferroni: } \alpha_{adj} = \alpha/m$$

$$\text{BH FDR: reject test }k\text{ if } p_{(k)} \leq \frac{k}{m} \cdot q$$

---

## Test Selection Summary

| Data Type | Groups | Distribution | Test |
|---|---|---|---|
| Continuous mean | 1 | Normal, σ known | Z-test |
| Continuous mean | 1 | Normal, σ unknown | One-sample t |
| Continuous mean | 2 independent | Normal | Welch's t |
| Continuous mean | 2 paired | Normal differences | Paired t |
| Continuous mean | 3+ independent | Normal | One-way ANOVA |
| Proportions | 1 | Large n | Z-test |
| Proportions | 2 | Large n | Two-proportion Z |
| Categorical | Any | Count data | Chi-square |
| Any | 2 independent | Non-normal | Mann-Whitney U |
| Any | 2 paired | Non-normal | Wilcoxon signed-rank |
| Any | 3+ independent | Non-normal | Kruskal-Wallis |

---

## Error & Decision Matrix

| Reality | Decision: Reject H₀ | Decision: Fail to Reject H₀ |
|---|---|---|
| H₀ True | Type I Error (prob = α) | Correct (prob = 1−α) |
| H₀ False | Correct / Power (prob = 1−β) | Type II Error (prob = β) |

---

## The 10 Most Common Misstatements (MEMORIZE)

1. ❌ "p = probability H₀ is true" → ✅ p = P(data | H₀)
2. ❌ "Accept H₀" → ✅ Fail to reject H₀
3. ❌ "95% prob. true value is in CI" → ✅ 95% of such CIs contain true value
4. ❌ "Small p = large effect" → ✅ p depends on n, not just effect
5. ❌ "Non-significant = no effect" → ✅ Insufficient evidence; may be underpowered
6. ❌ "Significant = important" → ✅ Also need practical significance
7. ❌ "p < 0.05 is always the right threshold" → ✅ α depends on costs of errors
8. ❌ "After peeking, we can stop early" → ✅ Optional stopping inflates Type I error
9. ❌ "Post-hoc subgroup finding is confirmed result" → ✅ It's exploratory, needs replication
10. ❌ "Higher n always makes tests better" → ✅ Higher n detects trivially small effects

---

---

# 🌳 PART B: Decision Tree — "Which Test to Use?"

```
START: What kind of data do you have?
│
├── CATEGORICAL DATA
│   │
│   ├── One variable (distribution test)?
│   │   └── Chi-Square Goodness of Fit
│   │
│   └── Two variables (relationship test)?
│       └── Chi-Square Test of Independence
│           (if any expected cell < 5: Fisher's Exact)
│
└── NUMERICAL DATA
    │
    ├── What is your goal?
    │   │
    │   ├── Test a SINGLE mean against known value
    │   │   ├── σ known → Z-test
    │   │   └── σ unknown
    │   │       ├── n ≥ 30 → t-test (≈ Z)
    │   │       ├── n < 30 AND normal → t-test
    │   │       └── n < 30 AND non-normal → Wilcoxon signed-rank
    │   │
    │   ├── Compare TWO groups
    │   │   ├── Independent groups?
    │   │   │   ├── Normal + n ≥ 30 → Welch's t-test
    │   │   │   ├── Normal + n < 30 → Welch's t-test (check normality)
    │   │   │   └── Non-normal or ordinal → Mann-Whitney U
    │   │   │
    │   │   └── Paired/matched groups?
    │   │       ├── Normal differences → Paired t-test
    │   │       └── Non-normal differences → Wilcoxon signed-rank
    │   │
    │   └── Compare 3+ groups
    │       ├── Normal + equal variances → One-way ANOVA
    │       │   └── Significant? → Post-hoc (Tukey HSD)
    │       ├── Normal + unequal variances → Welch's ANOVA
    │       └── Non-normal → Kruskal-Wallis
    │           └── Significant? → Post-hoc (Dunn's test)
    │
    └── Testing PROPORTIONS?
        ├── Single proportion → One-proportion Z-test
        └── Two proportions → Two-proportion Z-test
```

---

---

# 💼 PART C: 10 FAANG-Style Interview Questions

---

### Q1: Google — A/B Test Design

**"We're testing a new search ranking algorithm. How would you design and analyze this experiment?"**

**Answer:**

**Setup:** H₀: New algorithm has no effect on click satisfaction. H₁: It improves click-through rate or dwell time (both are proxies for satisfaction).

**Metric selection:** Primary: CTR on top-3 results (or NDCG for ranking). Guardrail: overall query volume, latency, user complaints.

**Randomization:** At user level (same user always gets same algorithm). Avoid session-level (same user seeing both — contaminates).

**Power analysis:** Based on historical CTR variance, set MDE (e.g., 0.5% absolute lift), α=0.05, power=80% → compute n.

**Test:** Two-sample t-test or Z-test for proportion. For ranking metrics: possibly non-parametric (Wilcoxon) if distribution is skewed.

**Duration:** ≥ 2 weeks to capture day-of-week effects.

**Analysis:** Compute test statistic, p-value, effect size, CI. Check for novelty effect by analyzing early vs. late week data. Segment by query type, geography, device.

---

### Q2: Meta — Multiple Metrics

**"Your A/B test on the News Feed ranking shows: DAU +0.2% (p=0.04), Session time -1% (p=0.03), Ad CTR +0.5% (p=0.001). How do you interpret this?"**

**Answer:**

Multiple significant results across metrics require correction (BH FDR or Bonferroni). With 3 tests at α=0.05, Bonferroni threshold = 0.0167.

- DAU: p=0.04 > 0.0167 → not significant after correction
- Session time: p=0.03 > 0.0167 → not significant after correction
- Ad CTR: p=0.001 < 0.0167 → significant

The ad CTR increase is the only robust finding. However, the directional signals matter: if session time is going down while ad CTR goes up, users may be seeing more ads but spending less time (worse experience). This tradeoff needs business evaluation. I would not ship without understanding the session time decrease further — it's a guardrail metric failure.

---

### Q3: Amazon — p-value vs. Effect Size

**"A/B test result: p = 0.0001, but conversion rate increased by 0.001%. Your PM wants to ship immediately. What do you say?"**

**Answer:**

The p-value is highly significant because we have a very large sample size. But the absolute lift is 0.001% — essentially negligible. Let's calculate business impact: if Amazon has 100M customers × 0.001% × $100 AOV = $100,000/year. Compared to the engineering and maintenance cost of shipping this feature, this is likely not worth it.

I'd report: (1) statistically significant at p<0.05; (2) practical impact is ~$100K/year; (3) Cohen's h ≈ tiny; (4) CI for the lift: the lower bound of the CI is nearly zero. Recommendation: do not ship based purely on p-value — practical significance is not there.

---

### Q4: Netflix — Confidence Intervals

**"You run an experiment measuring change in watch time. The 95% CI is [-2 min, +8 min]. What do you conclude?"**

**Answer:**

The CI contains zero, so we fail to reject H₀ at α=0.05 — the result is not statistically significant. However, the CI tells us more: the true effect could be anywhere from a 2-minute loss to an 8-minute gain. This is a wide interval — the test appears underpowered.

For business decisions: if a +5 minute increase would justify shipping, and the CI extends to +8, we can't confidently say we'd achieve that. I'd recommend checking power retrospectively and running a larger experiment. Don't ship, but don't kill the idea — we simply don't have enough data.

---

### Q5: Uber — Peeking

**"An engineer says 'the A/B test hit significance on Day 7, let's end it now and ship!' How do you respond?"**

**Answer:**

This is optional stopping / peeking bias. When you monitor a test continuously and stop as soon as p < 0.05, you inflate the true Type I error rate well above 5%. The significance achieved on Day 7 is likely a false positive — random fluctuations will cross the 0.05 line during any multi-day experiment.

Options: (1) If the test was pre-planned for 14 days, continue to Day 14 regardless. (2) If early stopping is genuinely needed (cost considerations), use sequential testing methods with proper alpha-spending (O'Brien-Fleming). These adjust the threshold for each interim look to maintain overall α=0.05. Simply stopping at p=0.04 on Day 7 is not valid.

---

### Q6: Airbnb — Non-Normality

**"Revenue per booking is right-skewed with a few very high outliers. Which test do you use for an A/B test?"**

**Answer:**

Several options:
1. **Log transformation**: If log(revenue) is approximately normal, apply t-test on the log scale.
2. **Winsorizing**: Cap outliers at the 95th or 99th percentile, then use t-test. Reduces sensitivity to whales.
3. **Mann-Whitney U**: Non-parametric, rank-based, not affected by extreme values. Tests if one distribution tends to produce higher values.
4. **Bootstrap CI**: Resample with replacement and compute CI for the mean difference — no distributional assumptions.
5. **Delta method / ratio metrics**: For revenue per booking as a ratio metric, use the delta method to compute SE correctly.

I'd use a combination: winsorized t-test as primary (easy to explain) + bootstrap as robustness check. If they agree, more confidence in the result.

---

### Q7: LinkedIn — Chi-Square Application

**"You want to know if connection acceptance rate differs across user tenure segments (new, mid, veteran). What test do you use?"**

**Answer:**

This is a test of independence between a categorical variable (tenure segment: 3 levels) and a binary outcome (accepted: yes/no). Use the **chi-square test of independence** on a 3×2 contingency table.

H₀: Connection acceptance rate is independent of user tenure.
H₁: Acceptance rate differs across tenure segments.

Compute expected counts as (row total × col total)/grand total. If any expected cell < 5, consider collapsing segments or using Fisher's exact test. A significant result tells us segments differ — follow up with pairwise comparisons with Bonferroni correction to identify which pairs.

---

### Q8: Microsoft — Power & Sample Size

**"Your experiment ran but results were inconclusive (p=0.3). A PM says 'the test failed.' What do you say?"**

**Answer:**

The test being inconclusive doesn't mean the experiment "failed" — it may have been underpowered. Key questions: (1) What was our planned MDE? (2) What was our achieved power? (3) What effect size did we actually observe?

If observed effect = 0.5% lift and planned MDE was 1% lift, we were looking for something bigger than what's there. The 0.5% might be real but we couldn't detect it. 

Check the CI: if CI = [−0.3%, 1.3%], we can't distinguish zero from a 1.3% lift. Underpowered.

Recommendation: run a power analysis for the observed effect and compute required n. Re-run with adequate power if the effect is practically meaningful.

---

### Q9: Apple — Sequential Testing

**"We run experiments for many weeks. Is there a valid way to peek at results mid-experiment?"**

**Answer:**

Yes — **sequential testing** methods allow valid interim analyses without inflating the Type I error:

1. **Alpha-spending functions (O'Brien-Fleming)**: Pre-specify interim looks (e.g., at 50% and 100% of target n). The threshold for early stopping is very strict (e.g., p < 0.005 at 50%) and relaxes toward the final look. Total α is preserved.

2. **Sequential Probability Ratio Test (SPRT)**: A likelihood-ratio based method that controls both α and β continuously. Can stop for significance OR futility.

3. **mSPRT (mixture SPRT)**: Used by Optimizely / Netflix — provides "always valid p-values" you can check anytime while maintaining Type I control.

Key: The method must be specified before the experiment. You can't retroactively justify peeking.

---

### Q10: Stripe — Practical Significance Framing

**"Define statistical significance vs. practical significance. Give a financial example."**

**Answer:**

**Statistical significance** answers: "Is the observed effect larger than what we'd expect from random chance?" It's controlled by sample size and variability. At large n, tiny effects are statistically significant.

**Practical significance** answers: "Is the effect large enough to matter for our business goals?" It's measured by effect size, absolute lift, and ROI.

**Financial example:** Stripe tests a new checkout flow. 10M transactions tested. Result: 0.002% reduction in abandonment rate, p < 0.001. 

- Statistically significant? Yes (massive sample).
- Practically significant? 0.002% × 10M annual transactions × avg. $200 = $40,000/year recovered. If the feature costs $2M to build and maintain, ROI is negative. **Not practically significant.**

Conversely: a 2% lift in conversion with p = 0.08 from a small-sample pilot might not be statistically confirmed yet, but the potential impact (millions in revenue) makes it worth a larger confirmatory experiment.

---

---

# 🗺️ PART D: Study Roadmap

```
WEEK 1: FOUNDATIONS
□ Foundations (H₀, H₁, one/two-tailed)
□ Type I/II Errors and Power
□ Statistical Concepts (CLT, SE, sampling distribution)
□ p-value (definition + all misinterpretations)

WEEK 2: TESTS
□ Z-test and t-test (one/two-sample/paired)
□ Chi-square tests
□ ANOVA + post-hoc
□ Non-parametric tests

WEEK 3: APPLIED
□ A/B testing end-to-end
□ Multiple testing + Bonferroni + FDR
□ Statistical vs. practical significance
□ Interview traps (MEMORIZE the table)

WEEK 4: FAANG PREP
□ Work through all 10 FAANG questions
□ Practice explaining each concept in <60 seconds
□ Implement Z-test, t-test, chi-square, bootstrap in Python
□ Practice sample size calculations by hand
```

---

## Python Quick Reference

```python
import numpy as np
from scipy import stats

# One-sample t-test
t_stat, p_val = stats.ttest_1samp(data, popmean=mu0)

# Two-sample Welch's t-test
t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

# Paired t-test
t_stat, p_val = stats.ttest_rel(before, after)

# Z-test for proportions
from statsmodels.stats.proportion import proportions_ztest
z_stat, p_val = proportions_ztest([x1, x2], [n1, n2])

# Chi-square test of independence
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

# One-way ANOVA
f_stat, p_val = stats.f_oneway(group1, group2, group3)

# Mann-Whitney U
u_stat, p_val = stats.mannwhitneyu(group1, group2, alternative='two-sided')

# Sample size calculation (proportions)
from statsmodels.stats.power import zt_ind_solve_power
n = zt_ind_solve_power(effect_size=h, alpha=0.05, power=0.8)

# Bootstrap CI
boot_means = [np.mean(np.random.choice(data, len(data), replace=True)) 
              for _ in range(10000)]
ci = np.percentile(boot_means, [2.5, 97.5])
```

---

*← [15 — Interview Traps](15_interview_traps.md)*

---

**You've mastered Hypothesis Testing. 🎓**

*Built as a complete interview reference — from CLT to FAANG. Review the cheat sheet before every interview.*
