# ANOVA — Analysis of Variance: Complete Notes

---

## Table of Contents

1. [What is ANOVA?](#what-is-anova)
2. [Core Intuition](#core-intuition)
3. [Types of ANOVA](#types-of-anova)
4. [Key Terminology](#key-terminology)
5. [One-Way ANOVA: Full Walkthrough](#one-way-anova-full-walkthrough)
6. [Two-Way ANOVA](#two-way-anova)
7. [Repeated Measures ANOVA](#repeated-measures-anova)
8. [Post-Hoc Tests](#post-hoc-tests)
9. [Assumptions of ANOVA](#assumptions-of-anova)
10. [Effect Size](#effect-size)
11. [FAANG QA Interview Questions](#faang-qa-interview-questions)

---

## What is ANOVA?

**ANOVA (Analysis of Variance)** is a statistical method used to test whether the means of **three or more groups** are significantly different from each other.

- **Null Hypothesis H₀**: All group means are equal → μ₁ = μ₂ = μ₃ = ... = μk
- **Alternative Hypothesis H₁**: At least one group mean is different

> Why not just run multiple t-tests?  
> Running k groups pairwise creates C(k,2) tests. Each test has a Type I error rate α. This inflates the **family-wise error rate**:  
> FWER = 1 − (1 − α)^m  
> For 4 groups and α = 0.05: FWER = 1 − (0.95)^6 ≈ **26.5%** — far too high.  
> ANOVA solves this with a single test.

---

## Core Intuition

ANOVA compares **two sources of variance**:

| Source | Question |
|--------|----------|
| **Between-group variance** | How much do group means differ from the overall mean? |
| **Within-group variance** | How much do individual observations vary within each group? |

The **F-statistic** is the ratio:

$$F = \frac{\text{Variance Between Groups}}{\text{Variance Within Groups}} = \frac{MS_{Between}}{MS_{Within}}$$

- If F ≈ 1 → group means are similar (noise ≈ signal)
- If F >> 1 → group means differ more than chance would predict → reject H₀

---

## Types of ANOVA

| Type | Description | When to Use |
|------|-------------|-------------|
| **One-Way ANOVA** | One independent variable, one dependent variable | Comparing 3+ groups on a single factor |
| **Two-Way ANOVA** | Two independent variables | Studying two factors and their interaction |
| **Repeated Measures** | Same subjects measured multiple times | Within-subject designs |
| **MANOVA** | Multiple dependent variables | When DVs are correlated |
| **ANCOVA** | Adds a covariate to control for confounds | Controlling for a continuous variable |

---

## Key Terminology

| Term | Symbol | Meaning |
|------|--------|---------|
| Grand Mean | $\bar{\bar{X}}$ | Mean of all observations |
| Group Mean | $\bar{X}_j$ | Mean of group j |
| Total observations | N | All data points across all groups |
| Number of groups | k | How many groups |
| Observations per group | nj | Sample size of group j |
| Sum of Squares | SS | Squared deviations, partitioned by source |
| Degrees of Freedom | df | Constraining values in estimation |
| Mean Square | MS | SS ÷ df |
| F-ratio | F | MS_between ÷ MS_within |

---

## One-Way ANOVA: Full Walkthrough

### Step 1 — Partition the Total Variance

The fundamental identity:

$$SS_{Total} = SS_{Between} + SS_{Within}$$

### Formulas

#### Sum of Squares Between (SS_B) — also called SS_Treatment

$$SS_B = \sum_{j=1}^{k} n_j (\bar{X}_j - \bar{\bar{X}})^2$$

Measures how far each group mean is from the grand mean, weighted by group size.

#### Sum of Squares Within (SS_W) — also called SS_Error

$$SS_W = \sum_{j=1}^{k} \sum_{i=1}^{n_j} (X_{ij} - \bar{X}_j)^2$$

Measures variability of individual scores around their own group mean.

#### Sum of Squares Total (SS_T)

$$SS_T = \sum_{j=1}^{k} \sum_{i=1}^{n_j} (X_{ij} - \bar{\bar{X}})^2$$

### Step 2 — Degrees of Freedom

| Source | df |
|--------|----|
| Between | $df_B = k - 1$ |
| Within | $df_W = N - k$ |
| Total | $df_T = N - 1$ |

### Step 3 — Mean Squares

$$MS_B = \frac{SS_B}{df_B} = \frac{SS_B}{k-1}$$

$$MS_W = \frac{SS_W}{df_W} = \frac{SS_W}{N-k}$$

### Step 4 — F-Statistic

$$F = \frac{MS_B}{MS_W}$$

Under H₀, this follows an **F-distribution** with (k−1, N−k) degrees of freedom.

### Step 5 — p-value Decision

- If **p < α** (commonly 0.05) → **Reject H₀** (at least one group mean differs)
- If **p ≥ α** → **Fail to reject H₀**

---

### ANOVA Table Format

| Source | SS | df | MS | F | p-value |
|--------|----|----|----|---|---------|
| Between (Treatment) | SS_B | k − 1 | SS_B / (k−1) | MS_B / MS_W | from F-table |
| Within (Error) | SS_W | N − k | SS_W / (N−k) | | |
| **Total** | **SS_T** | **N − 1** | | | |

---

### Worked Example

**Scenario**: A company tests 3 different UI designs (A, B, C) for task completion time (seconds).

```
Group A (Design A): 12, 15, 14, 11, 13   → n_A = 5, X̄_A = 13.0
Group B (Design B): 20, 22, 19, 21, 23   → n_B = 5, X̄_B = 21.0
Group C (Design C): 16, 14, 17, 15, 18   → n_C = 5, X̄_C = 16.0
```

**Grand Mean:**

$$\bar{\bar{X}} = \frac{(13.0 \times 5) + (21.0 \times 5) + (16.0 \times 5)}{15} = \frac{65 + 105 + 80}{15} = \frac{250}{15} = 16.67$$

**SS_Between:**

$$SS_B = 5(13.0 - 16.67)^2 + 5(21.0 - 16.67)^2 + 5(16.0 - 16.67)^2$$
$$= 5(13.47) + 5(18.75) + 5(0.449)$$
$$= 67.35 + 93.75 + 2.245 = 163.35$$

**SS_Within (calculated per group):**

Group A: $(12-13)^2 + (15-13)^2 + (14-13)^2 + (11-13)^2 + (13-13)^2 = 1+4+1+4+0 = 10$

Group B: $(20-21)^2 + (22-21)^2 + (19-21)^2 + (21-21)^2 + (23-21)^2 = 1+1+4+0+4 = 10$

Group C: $(16-16)^2 + (14-16)^2 + (17-16)^2 + (15-16)^2 + (18-16)^2 = 0+4+1+1+4 = 10$

$$SS_W = 10 + 10 + 10 = 30$$

**Degrees of Freedom:**

$$df_B = k - 1 = 3 - 1 = 2$$
$$df_W = N - k = 15 - 3 = 12$$

**Mean Squares:**

$$MS_B = \frac{163.35}{2} = 81.68$$
$$MS_W = \frac{30}{12} = 2.50$$

**F-statistic:**

$$F = \frac{81.68}{2.50} = 32.67$$

**ANOVA Table:**

| Source | SS | df | MS | F | p-value |
|--------|----|----|----|---|---------|
| Between | 163.35 | 2 | 81.68 | **32.67** | < 0.0001 |
| Within | 30.00 | 12 | 2.50 | | |
| Total | 193.35 | 14 | | | |

**Conclusion**: F(2, 12) = 32.67, p < 0.0001. We reject H₀. At least one UI design results in significantly different task completion time.

---

## Two-Way ANOVA

Tests the effect of **two independent variables** (factors) and their **interaction**.

### Variance Partitioning

$$SS_{Total} = SS_A + SS_B + SS_{A \times B} + SS_{Within}$$

| Source | What It Tests |
|--------|--------------|
| SS_A | Main effect of Factor A |
| SS_B | Main effect of Factor B |
| SS_{A×B} | Interaction — does the effect of A change across levels of B? |
| SS_W | Random error within cells |

### F-ratios

$$F_A = \frac{MS_A}{MS_W}, \quad F_B = \frac{MS_B}{MS_W}, \quad F_{A \times B} = \frac{MS_{A \times B}}{MS_W}$$

### Interpreting Interaction Effects

- **No interaction**: Lines on an interaction plot are parallel
- **Interaction present**: Lines cross or diverge — the effect of one factor depends on the level of the other

**Example**: Testing page load speed (Factor A: CDN vs No CDN) × device type (Factor B: Mobile vs Desktop). If CDN helps mobile users drastically but barely helps desktop users → significant interaction.

---

## Repeated Measures ANOVA

Used when the **same subjects** are measured under multiple conditions or time points.

### Why It's More Powerful

It removes **individual differences** as a source of error:

$$SS_{Total} = SS_{Between-subjects} + SS_{Within-subjects}$$
$$SS_{Within-subjects} = SS_{Treatment} + SS_{Error}$$

The error term is smaller because subject-level variation is isolated → higher F-ratio → more statistical power.

### Degrees of Freedom

| Source | df |
|--------|----|
| Between subjects | n − 1 |
| Treatment | k − 1 |
| Error | (n−1)(k−1) |

### Sphericity

Repeated measures ANOVA assumes **sphericity** — the variances of the differences between all pairs of conditions are equal.

- Test with **Mauchly's Test of Sphericity**
- If violated, apply corrections: **Greenhouse-Geisser** or **Huynh-Feldt** epsilon adjustments

---

## Post-Hoc Tests

ANOVA tells you *that* means differ — post-hoc tests tell you *which* pairs differ.

| Test | Best For | Controls FWER? |
|------|----------|----------------|
| **Tukey's HSD** | All pairwise comparisons, equal n | Yes |
| **Bonferroni** | Few planned comparisons | Yes (conservative) |
| **Scheffé** | All contrasts, unequal n | Yes (very conservative) |
| **Fisher's LSD** | Only after significant F | Partial |
| **Dunnett's** | Comparing all groups to a control | Yes |
| **Games-Howell** | Unequal variances | Yes |

### Tukey's HSD Formula

$$HSD = q \cdot \sqrt{\frac{MS_W}{n}}$$

Where q is the **studentized range statistic** from a q-table at the desired α level.

Two group means are significantly different if:

$$|\bar{X}_i - \bar{X}_j| > HSD$$

---

## Assumptions of ANOVA

| Assumption | Description | How to Check |
|------------|-------------|--------------|
| **Independence** | Observations are independent | Study design review |
| **Normality** | DV is normally distributed within each group | Shapiro-Wilk test, Q-Q plot |
| **Homogeneity of Variance** | Equal variances across groups (homoscedasticity) | Levene's test, Bartlett's test |
| **Random Sampling** | Data is a random sample | Study design review |

### When Assumptions Are Violated

| Violation | Remedy |
|-----------|--------|
| Non-normality (large n) | CLT makes ANOVA robust — usually fine |
| Non-normality (small n) | Use **Kruskal-Wallis** (non-parametric alternative) |
| Unequal variances | Use **Welch's ANOVA** |
| Non-sphericity (RM) | Apply **Greenhouse-Geisser** correction |

---

## Effect Size

Statistical significance ≠ practical significance. Always report effect size.

### Eta-Squared (η²)

$$\eta^2 = \frac{SS_B}{SS_T}$$

Proportion of total variance explained by the treatment. **Biased** — tends to overestimate in small samples.

### Partial Eta-Squared (η²_p)

$$\eta^2_p = \frac{SS_{Effect}}{SS_{Effect} + SS_{Error}}$$

Used in multi-factor ANOVA. Each effect's size relative to its own error.

### Omega-Squared (ω²) — preferred

$$\omega^2 = \frac{SS_B - (k-1) \cdot MS_W}{SS_T + MS_W}$$

Less biased than η². Better for generalizing to population.

### Effect Size Benchmarks (Cohen, 1988)

| Size | η² / ω² |
|------|---------|
| Small | 0.01 |
| Medium | 0.06 |
| Large | 0.14 |

---

## FAANG QA Interview Questions

---

### Q1: What is ANOVA and why would you use it instead of multiple t-tests?

**Answer:**  
ANOVA tests whether the means of three or more groups are significantly different using a single F-test. Running multiple t-tests inflates the Type I error rate (family-wise error rate). For example, with 4 groups and α = 0.05, doing all 6 pairwise t-tests gives a FWER of ~26.5%. ANOVA controls this at the desired α.

---

### Q2: You ran an A/B/C test at Google — 3 page designs. ANOVA gives p = 0.03. What do you conclude, and what do you do next?

**Answer:**  
p < 0.05 → reject H₀. At least one design has a different mean outcome (e.g., click-through rate). But ANOVA doesn't tell you *which* designs differ. The next step is a **post-hoc test** — likely Tukey's HSD since we're comparing all pairs. Also report effect size (η² or ω²) and check if the difference is practically meaningful, not just statistically significant.

---

### Q3: What is the F-statistic and what happens when F = 1?

**Answer:**  
F = MS_Between / MS_Within. When F = 1, the variability between group means is no greater than variability within groups — consistent with H₀ being true (all means equal). As F grows, it indicates group means are more spread out than noise would explain.

---

### Q4: You're comparing load times across 5 server configurations. Levene's test gives p = 0.01. What does this mean and how do you handle it?

**Answer:**  
p < 0.05 on Levene's test → **homogeneity of variance is violated** (heteroscedasticity). Standard ANOVA assumes equal variances. The fix is to use **Welch's ANOVA**, which doesn't require equal variances. Follow up with **Games-Howell** post-hoc test instead of Tukey's.

---

### Q5: At Meta, you test a new feed ranking algorithm across 4 regions over 6 weeks (same users measured weekly). What type of ANOVA do you use?

**Answer:**  
**Repeated Measures ANOVA** (or mixed ANOVA if there are also between-subject factors). The same users are measured 6 times, so observations are not independent across time. Repeated measures ANOVA accounts for this by partitioning out between-subject variance, increasing power. Must check **sphericity** with Mauchly's test and apply Greenhouse-Geisser correction if violated.

---

### Q6: Explain the difference between SS_Between and SS_Within in plain language.

**Answer:**  
- **SS_Between**: How much do the group averages differ from the overall average? Large SS_B means groups are far apart — treatment has an effect.  
- **SS_Within**: Within each group, how much do individuals vary around their own group mean? This is noise/random error — things ANOVA can't explain.  
ANOVA asks: is the signal (between) large relative to the noise (within)?

---

### Q7: You use ANOVA and get a significant F, but η² = 0.02. What does this tell you?

**Answer:**  
The result is statistically significant, but the **effect size is very small** (η² = 0.02 < 0.06 is small by Cohen's conventions). Only 2% of the variance in the outcome is explained by the grouping factor. This often happens with very large sample sizes — tiny differences become statistically detectable. The finding may not be **practically or business meaningful**. Always pair p-values with effect sizes.

---

### Q8: What is an interaction effect in Two-Way ANOVA and why does it matter?

**Answer:**  
An interaction means the effect of one factor **depends on the level of another**. Example: A new checkout UI (Factor A) might significantly improve conversion on mobile (Factor B = mobile) but have no effect on desktop. The main effects alone would miss this nuance. Interaction terms reveal when you can't generalize a treatment effect uniformly — critical for product decisions about segment-specific rollouts.

---

### Q9: An engineer at Amazon says "we should just use ANOVA to compare all our metrics across the test groups." What concerns would you raise?

**Answer:**  
Several concerns:
1. **Multiple comparisons across metrics**: Running ANOVA on 20 metrics inflates FWER — need correction (Bonferroni, BH procedure).
2. **Correlated outcomes**: If metrics are correlated (e.g., CTR and revenue), use **MANOVA** instead.
3. **Non-independence**: If users appear in multiple groups, standard ANOVA is violated.
4. **Practical significance**: Significant F doesn't mean the magnitude matters for business decisions.
5. **Assumption checks**: Need to verify normality and homoscedasticity for each metric before blindly applying ANOVA.

---

### Q10: What non-parametric alternative would you use if ANOVA assumptions are severely violated?

**Answer:**  
**Kruskal-Wallis H Test** — the non-parametric equivalent of one-way ANOVA. It ranks all observations and tests whether the rank distributions differ across groups. It doesn't assume normality. For post-hoc comparisons after Kruskal-Wallis, use **Dunn's test** with Bonferroni correction. For repeated measures, the alternative is **Friedman's test**.

---

### Q11: Walk me through calculating an F-statistic from scratch.

**Answer (step-by-step):**
1. Compute group means $\bar{X}_j$ and grand mean $\bar{\bar{X}}$
2. Compute $SS_B = \sum_j n_j(\bar{X}_j - \bar{\bar{X}})^2$
3. Compute $SS_W = \sum_j \sum_i (X_{ij} - \bar{X}_j)^2$
4. Degrees of freedom: $df_B = k-1$, $df_W = N-k$
5. $MS_B = SS_B / df_B$, $MS_W = SS_W / df_W$
6. $F = MS_B / MS_W$
7. Compare F to critical F-value at $(df_B, df_W)$ or compute p-value

---

### Q12: What's the difference between eta-squared and omega-squared?

**Answer:**  
Both measure effect size (proportion of variance explained), but:
- **η²** = SS_B / SS_T — biased upward, especially in small samples, because it uses sample SS which overestimates population variance explained.
- **ω²** subtracts a correction term: $\omega^2 = (SS_B - (k-1) \cdot MS_W) / (SS_T + MS_W)$. It's an unbiased estimator of the population effect size and is preferred when reporting results.

---

*Notes prepared for statistical testing & experimentation interviews.*
*Topics: One-Way ANOVA · Two-Way ANOVA · Repeated Measures · Post-Hoc Tests · Effect Size · FAANG QA*
