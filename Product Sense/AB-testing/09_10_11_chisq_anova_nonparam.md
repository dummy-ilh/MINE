# 📊 09, 10, 11 — Chi-Square, ANOVA & Non-Parametric Tests

---

# PART A: Chi-Square Test (χ²)

## Overview

The chi-square test is used for **categorical data**. No means, no variances — just counts.

$$\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}$$

Where $O_i$ = observed count, $E_i$ = expected count under H₀.

**Key property:** $\chi^2$ is always non-negative. Large $\chi^2$ → evidence against H₀.

---

## 1. Goodness of Fit Test

### Use When
Testing whether observed **frequency distribution** matches a hypothesized distribution.

### Example: Die Fairness Test

> Roll a die 120 times. Expected: 20 per face. Observed:

| Face | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Observed | 15 | 22 | 18 | 25 | 19 | 21 |
| Expected | 20 | 20 | 20 | 20 | 20 | 20 |

$$\chi^2 = \frac{(15-20)^2}{20} + \frac{(22-20)^2}{20} + \frac{(18-20)^2}{20} + \frac{(25-20)^2}{20} + \frac{(19-20)^2}{20} + \frac{(21-20)^2}{20}$$

$$= \frac{25+4+4+25+1+1}{20} = \frac{60}{20} = 3.0$$

df = k − 1 = 5. Critical value at α = 0.05: $\chi^2_{0.05, 5} = 11.07$

Since $3.0 < 11.07$ → **Fail to Reject H₀**. Die appears fair.

---

## 2. Chi-Square Test of Independence

### Use When
Testing whether two **categorical variables are independent** in a contingency table.

### Hypotheses
- H₀: Variable A and Variable B are independent
- H₁: They are associated (not independent)

### Expected Counts Formula

$$E_{ij} = \frac{(\text{Row } i \text{ total}) \times (\text{Column } j \text{ total})}{\text{Grand total}}$$

### Example: Click vs. Device Type

|  | Clicked | Not Clicked | Total |
|---|---|---|---|
| Mobile | 120 | 380 | 500 |
| Desktop | 200 | 300 | 500 |
| **Total** | **320** | **680** | **1000** |

Expected counts:
$$E_{11} = \frac{500 \times 320}{1000} = 160, \quad E_{12} = \frac{500 \times 680}{1000} = 340$$
$$E_{21} = \frac{500 \times 320}{1000} = 160, \quad E_{22} = \frac{500 \times 680}{1000} = 340$$

$$\chi^2 = \frac{(120-160)^2}{160} + \frac{(380-340)^2}{340} + \frac{(200-160)^2}{160} + \frac{(300-340)^2}{340}$$

$$= \frac{1600}{160} + \frac{1600}{340} + \frac{1600}{160} + \frac{1600}{340} = 10 + 4.71 + 10 + 4.71 = 29.41$$

df = (rows−1)(cols−1) = 1. $\chi^2_{0.05, 1} = 3.84$

Since $29.41 \gg 3.84$ → **Reject H₀**. Click rate depends on device type.

### Assumption: Minimum Expected Count ≥ 5

If any $E_{ij} < 5$, chi-square approximation is unreliable. Use Fisher's Exact Test instead.

---

## 💬 Interview Questions (Chi-Square)

**Q1: When would you use a chi-square test in product analytics?**

> A: Testing whether user behavior (e.g., feature adoption) differs across categorical segments like country, device, or cohort. Example: "Is the distribution of subscription plan types the same across mobile and desktop users?" or "Does the clicked/not-clicked proportion differ between ad formats?"

**Q2: What's the difference between chi-square goodness-of-fit and independence test?**

> A: Goodness-of-fit: one variable, testing if its distribution matches a theoretical/expected one. Independence: two variables in a contingency table, testing if they're related.

---

---

# PART B: ANOVA (Analysis of Variance)

## The Problem ANOVA Solves

Comparing means across **3 or more groups**. Why not multiple t-tests?

Running $\binom{k}{2}$ t-tests at α = 0.05 inflates the family-wise error rate:

$$P(\text{at least one false positive}) = 1 - (1-0.05)^{\binom{k}{2}}$$

For k=5 groups: $1 - 0.95^{10} = 1 - 0.60 = 0.40$ (40% false positive rate!).

ANOVA tests **all groups simultaneously** while keeping α intact.

---

## One-Way ANOVA

### Hypotheses

$$H_0: \mu_1 = \mu_2 = \cdots = \mu_k$$
$$H_1: \text{At least one pair } \mu_i \neq \mu_j$$

### Variance Decomposition — The Key Intuition

$$SS_{total} = SS_{between} + SS_{within}$$

| Term | Formula | Meaning |
|---|---|---|
| $SS_{between}$ (SSB) | $\sum_i n_i (\bar{x}_i - \bar{x})^2$ | Variation due to group differences |
| $SS_{within}$ (SSW) | $\sum_i \sum_j (x_{ij} - \bar{x}_i)^2$ | Variation within each group (noise) |

### F-Statistic

$$F = \frac{MS_{between}}{MS_{within}} = \frac{SSB/(k-1)}{SSW/(n-k)}$$

**Intuition:** F measures signal-to-noise ratio. A large F means group differences are large relative to within-group variability → evidence against H₀.

Under H₀, $F \sim F_{k-1, n-k}$.

### ANOVA Table

| Source | SS | df | MS | F |
|---|---|---|---|---|
| Between groups | SSB | k−1 | SSB/(k−1) | MS_B/MS_W |
| Within groups | SSW | n−k | SSW/(n−k) | |
| Total | SST | n−1 | | |

### Step-by-Step Example

> Three email subject line variants tested on CTR (%):
> - A: [21, 25, 19, 22] → $\bar{x}_A = 21.75$
> - B: [28, 30, 27, 31] → $\bar{x}_B = 29.0$
> - C: [18, 20, 17, 19] → $\bar{x}_C = 18.5$
>
> Grand mean: $\bar{x} = (21.75+29.0+18.5)/3 = 23.08$

SSB = $4(21.75-23.08)^2 + 4(29.0-23.08)^2 + 4(18.5-23.08)^2$
$= 4(1.77) + 4(35.05) + 4(20.97) = 7.09 + 140.19 + 83.89 = 231.17$

SSW (sum of within-group squared deviations) ≈ 24.75

$F = \frac{231.17/2}{24.75/9} = \frac{115.58}{2.75} = 42.03$

$F_{0.05, 2, 9} = 4.26$. Since $42.03 \gg 4.26$ → **Reject H₀**. Subject lines have significantly different CTRs.

### Post-Hoc Tests

ANOVA's rejection of H₀ only tells you *something differs* — not *which pairs*. Post-hoc tests identify the specific differences:

| Test | Use | Description |
|---|---|---|
| Tukey's HSD | Equal n, control-all comparisons | Compares all pairs, controls FWER |
| Bonferroni | Any design | Conservative, divides α by comparisons |
| Dunnett's | Compare all vs. control | More powerful when only comparing to baseline |

### Assumptions

1. **Independence** of observations
2. **Normality** within each group (robust to violations for large n)
3. **Homogeneity of variance** (Levene's test or Bartlett's test)

---

---

# PART C: Non-Parametric Tests

## When to Use Non-Parametric Tests

Non-parametric tests make **no distributional assumptions**. Use them when:

- Data is severely non-normal (especially with small $n$)
- Data is ordinal (Likert scale: "rate 1-5")
- Presence of extreme outliers that can't be removed
- Small sample size where normality can't be assumed

**Trade-off:** Non-parametric tests are typically less powerful than their parametric counterparts when parametric assumptions hold.

---

## Mann-Whitney U Test

**Parametric equivalent:** Two-sample t-test (independent groups)

**Null hypothesis:** The two populations have the same distribution (often stated as equal medians).

### Procedure

1. Combine both samples and rank all observations
2. Sum the ranks for each group: $W_1$ (Group 1), $W_2$ (Group 2)

$$U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - W_1$$

$$U_2 = n_1 n_2 + \frac{n_2(n_2+1)}{2} - W_2$$

$$U = \min(U_1, U_2)$$

Compare U to critical values from Mann-Whitney tables. For large samples, use normal approximation:

$$Z = \frac{U - \frac{n_1 n_2}{2}}{\sqrt{\frac{n_1 n_2 (n_1+n_2+1)}{12}}}$$

### Example

> Group A (5 users): session durations [2, 5, 9, 14, 20]
> Group B (5 users): session durations [3, 7, 8, 12, 16]

Rank all 10 values: 2(1), 3(2), 5(3), 7(4), 8(5), 9(6), 12(7), 14(8), 16(9), 20(10)

$W_1 = 1+3+6+8+10 = 28$, $W_2 = 2+4+5+7+9 = 27$

$U_1 = 25 + 15 - 28 = 12$, $U_2 = 25 + 15 - 27 = 13$

$U = 12$. Critical value (α=0.05, n1=n2=5): 2. Since $12 > 2$ → **Fail to Reject H₀**.

---

## Wilcoxon Signed-Rank Test

**Parametric equivalent:** Paired t-test

**Use:** Paired samples, non-normal differences.

### Procedure

1. Compute differences $d_i = x_{i,2} - x_{i,1}$
2. Ignore zero differences
3. Rank absolute differences $|d_i|$
4. Assign signs back to ranks
5. $W^+ = $ sum of positive ranks, $W^- = $ sum of negative ranks
6. $T = \min(W^+, W^-)$

For large n, approximate with normal distribution.

---

## Non-Parametric Equivalents Summary Table

| Parametric Test | Non-Parametric Equivalent | Test Statistic |
|---|---|---|
| One-sample t-test | Wilcoxon signed-rank (vs. median) | W |
| Two-sample t-test | Mann-Whitney U | U |
| Paired t-test | Wilcoxon signed-rank | T |
| One-way ANOVA | Kruskal-Wallis H | H |
| Pearson correlation | Spearman rank correlation | ρ_s |

---

## Kruskal-Wallis Test

**Parametric equivalent:** One-way ANOVA  
**Use:** Comparing 3+ groups with non-normal data.

$$H = \frac{12}{n(n+1)} \sum_{i=1}^k \frac{R_i^2}{n_i} - 3(n+1)$$

Where $R_i$ = sum of ranks for group $i$.

Under H₀: $H \sim \chi^2_{k-1}$ (for large samples).

---

## 💬 Interview Questions

**Q1: When would you choose a non-parametric test over a t-test in a product experiment?**

> A: When sample sizes are small (n < 30 per group), the metric is ordinal (e.g., satisfaction ratings 1–5), or the data has extreme outliers that reflect real user behavior (like revenue per user — a handful of whales can make the distribution extremely right-skewed). Non-parametric tests are also useful when stakeholders are more comfortable with median-based comparisons.

**Q2: What are the limitations of non-parametric tests?**

> A: Lower statistical power when parametric assumptions hold — meaning you need larger samples to detect the same effect. They also test distributional differences, not specifically mean differences, which can make interpretation less intuitive. Finally, they don't easily extend to multi-factor designs or covariate adjustment (unlike ANOVA/regression).

**Q3: ANOVA rejects H₀ across 5 groups. What do you do next?**

> A: Run post-hoc pairwise comparisons while controlling the family-wise error rate. Tukey's HSD is standard. Practically: compute 95% simultaneous confidence intervals for all pairwise differences — any interval not containing 0 indicates a significant pair. Also examine effect sizes (Cohen's d) for each pair to understand practical significance.

---

## 🚨 Common Pitfalls

**Chi-Square:**
- Expected cell counts < 5 → use Fisher's exact test
- Confusing independence and goodness-of-fit applications
- Forgetting df = (r-1)(c-1) for independence test

**ANOVA:**
- Stopping at ANOVA without post-hoc comparisons
- Assuming equal variances without testing
- Using one-way ANOVA for factorial designs (need two-way ANOVA)

**Non-Parametric:**
- Using non-parametric by default (sacrificing power unnecessarily)
- Forgetting that Kruskal-Wallis needs post-hoc tests too (Dunn's test)

---

*← [07/08 — Z & T Tests](07_08_z_t_tests.md) | [12 — A/B Testing →](12_ab_testing.md)*
