# ANOVA — Analysis of Variance: Complete Notes

---

## Table of Contents

1. [What is ANOVA?](#what-is-anova)
2. [Core Intuition](#core-intuition)
3. [Why It Actually Works](#why-it-actually-works)
4. [Types of ANOVA](#types-of-anova)
5. [Key Terminology](#key-terminology)
6. [One-Way ANOVA: Full Walkthrough](#one-way-anova-full-walkthrough)
7. [Two-Way ANOVA](#two-way-anova)
8. [Repeated Measures ANOVA](#repeated-measures-anova)
9. [Post-Hoc Tests](#post-hoc-tests)
10. [Assumptions of ANOVA](#assumptions-of-anova)
11. [Effect Size](#effect-size)
12. [FAANG QA Interview Questions](#faang-qa-interview-questions)

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

## Why It Actually Works

This is the part most notes skip, and interviewers love probing it — knowing the formulas gets you halfway, knowing *why* they're built this way is what separates a memorized answer from a real understanding.

### Why does SS_Total split cleanly into SS_Between + SS_Within?

It's not a coincidence — it's an algebraic identity. Add and subtract the group mean inside the total deviation:

$$X_{ij} - \bar{\bar{X}} = (X_{ij} - \bar{X}_j) + (\bar{X}_j - \bar{\bar{X}})$$

Square both sides and sum over all i, j. The cross-product term vanishes because, within any single group, $\sum_i (X_{ij} - \bar{X}_j) = 0$ by definition of a mean. That's the entire trick — an **orthogonal decomposition**, the same geometric idea behind the Pythagorean theorem applied to variance. This is exactly why SS_B and SS_W can be computed independently on separate data slices and simply added to get SS_T.

### Why the ratio F, specifically — why not just compare SS_B and SS_W directly?

SS_B and SS_W are sums of squared deviations built from different numbers of independent quantities, so they aren't on comparable scales — you first divide each by its degrees of freedom to get a per-unit variance estimate (MS_B, MS_W). Under H₀ (all means truly equal), both MS_B and MS_W are independent, unbiased estimators of the *same* population variance σ². The ratio of two independent chi-square-distributed quantities, each scaled by its own df, is by definition distributed as an **F-distribution**. This isn't a heuristic — Cochran's theorem guarantees SS_B and SS_W are statistically independent under the normality assumption, which is what makes the F-test valid in the first place.

So F is large only when MS_B overestimates σ² relative to MS_W — which only happens when group means are genuinely spread out beyond what sampling noise alone would produce.

### Why is df_Within = N − k, not N?

Each group mean $\bar{X}_j$ is itself estimated from that group's data, which "uses up" one degree of freedom per group. Across k groups you lose k degrees of freedom in total, leaving N − k free values to estimate within-group variance. It's the same logic as losing 1 df for a single-sample variance (n − 1), just applied k times over.

### Why does unequal variance hurt ANOVA more than unequal sample size?

MS_W pools every group into one shared variance estimate, which implicitly assumes a single true σ² across groups. If the real variances differ, that pooled estimate becomes distorted — dragged toward whichever group happens to have the most observations, not necessarily the group with the "true" spread. This is exactly why **Welch's ANOVA** doesn't pool at all — it replaces MS_W with a variance-weighted term and adjusts the degrees of freedom (via the Welch-Satterthwaite equation) instead.

### Why does a larger sample size make small effects "significant"?

MS_W shrinks as N grows (more data → tighter estimate of within-group noise), so F = MS_B / MS_W increases even if the true effect (the gap between group means) hasn't changed at all. This is precisely why **p < 0.05 is not the same as "the effect matters"** — it's why effect size (η², ω²) has to be reported alongside the F-test, especially in large-N settings like product experiments with millions of users.

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

### Worked Numerical Example

**Scenario**: An engineering team tests **checkout button color** (Factor A: Red vs Green) × **device type** (Factor B: Mobile vs Desktop) on conversion rate (%), with 3 users per cell.

```
              Mobile              Desktop
Red      (A1B1): 10, 12, 11       (A1B2): 20, 22, 21
Green    (A2B1): 18, 20, 19       (A2B2): 21, 23, 22
```

**Cell means:**

| | Mobile | Desktop | Row mean (Factor A) |
|---|---|---|---|
| Red | 11.0 | 21.0 | 16.0 |
| Green | 19.0 | 22.0 | 20.5 |
| **Col mean (Factor B)** | **15.0** | **21.5** | **Grand mean = 18.25** |

**SS_A (Color: Red vs Green)** — treat each color's overall mean vs grand mean, weighted by n = 6 per color:

$$SS_A = 6(16.0 - 18.25)^2 + 6(20.5 - 18.25)^2 = 6(5.0625) + 6(5.0625) = 30.375 + 30.375 = 60.75$$

**SS_B (Device: Mobile vs Desktop)** — n = 6 per device:

$$SS_B = 6(15.0 - 18.25)^2 + 6(21.5 - 18.25)^2 = 6(10.5625) \times 2 = 126.75$$

**SS_{A×B} (Interaction)** — for each cell, compare (cell mean) to (row mean + col mean − grand mean), squared, times n = 3 per cell:

$$\hat{X}_{cell} = \bar{X}_{row} + \bar{X}_{col} - \bar{\bar{X}}$$

- A1B1 (Red-Mobile): predicted = 16.0 + 15.0 − 18.25 = 12.75; actual = 11.0; diff = −1.75
- A1B2 (Red-Desktop): predicted = 16.0 + 21.5 − 18.25 = 19.25; actual = 21.0; diff = 1.75
- A2B1 (Green-Mobile): predicted = 20.5 + 15.0 − 18.25 = 17.25; actual = 19.0; diff = 1.75
- A2B2 (Green-Desktop): predicted = 20.5 + 21.5 − 18.25 = 23.75; actual = 22.0; diff = −1.75

$$SS_{A \times B} = 3\left[(-1.75)^2 + (1.75)^2 + (1.75)^2 + (-1.75)^2\right] = 3(3.0625 \times 4) = 3(12.25) = 36.75$$

**SS_Within** — variance inside each cell (all cells have variance from values ±1 around their mean, e.g. Red-Mobile: 10,12,11 → mean 11 → $(10-11)^2+(12-11)^2+(11-11)^2 = 1+1+0=2$; same pattern in all 4 cells = 2 each):

$$SS_W = 2 + 2 + 2 + 2 = 8$$

**Degrees of Freedom** (a = 2 levels of A, b = 2 levels of B, n = 3 per cell, N = 12):

| Source | df |
|--------|----|
| A | a − 1 = 1 |
| B | b − 1 = 1 |
| A×B | (a−1)(b−1) = 1 |
| Within | ab(n−1) = 8 |
| Total | N − 1 = 11 |

**Mean Squares and F:**

| Source | SS | df | MS | F |
|--------|----|----|----|----|
| A (Color) | 60.75 | 1 | 60.75 | 60.75 / 1.0 = **60.75** |
| B (Device) | 126.75 | 1 | 126.75 | 126.75 / 1.0 = **126.75** |
| A×B | 36.75 | 1 | 36.75 | 36.75 / 1.0 = **36.75** |
| Within | 8.00 | 8 | 1.0 | |
| Total | 232.25 | 11 | | |

**Conclusion**: All three F-values are large relative to df(1,8) critical value (~5.32 at α=0.05). Both main effects and the interaction are significant — meaning color matters, device matters, **and** the size of the color effect genuinely differs by device (button color helps far more on mobile than desktop here). This is exactly the pattern a product team would use to justify a **segment-specific rollout** (e.g., green button on mobile only) rather than a blanket change.

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

### Worked Numerical Example (using the One-Way UI Design data)

From the earlier worked example: $MS_W = 2.50$, $n = 5$ per group, $k = 3$ groups, $df_W = 12$.

Look up the studentized range statistic q for k = 3, df = 12, α = 0.05 → **q ≈ 3.77**.

$$HSD = 3.77 \times \sqrt{\frac{2.50}{5}} = 3.77 \times \sqrt{0.5} = 3.77 \times 0.707 = 2.665$$

**Pairwise mean differences:**

- $|\bar{X}_A - \bar{X}_B| = |13.0 - 21.0| = 8.0 \to 8.0 > 2.665$ → **significant**
- $|\bar{X}_A - \bar{X}_C| = |13.0 - 16.0| = 3.0 \to 3.0 > 2.665$ → **significant**
- $|\bar{X}_B - \bar{X}_C| = |21.0 - 16.0| = 5.0 \to 5.0 > 2.665$ → **significant**

**Conclusion**: All three pairwise comparisons exceed HSD — every design differs significantly from every other. Design A (13.0s) is fastest, Design C (16.0s) is second, Design B (21.0s) is slowest. Note this differs from the overall F-test conclusion only in *specificity*: the F-test told us "something differs," Tukey's HSD tells us "everything differs, pairwise."

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

### Worked Numerical Example (using the One-Way UI Design data)

From the earlier worked example: $SS_B = 163.35$, $SS_W = 30.00$, $SS_T = 193.35$, $k = 3$, $MS_W = 2.50$.

**Eta-squared:**

$$\eta^2 = \frac{163.35}{193.35} = 0.845$$

**Omega-squared:**

$$\omega^2 = \frac{163.35 - (3-1)(2.50)}{193.35 + 2.50} = \frac{163.35 - 5.0}{195.85} = \frac{158.35}{195.85} = 0.808$$

**Interpretation**: Both η² (0.845) and ω² (0.808) are far above the "large" threshold of 0.14 — the UI design choice explains roughly **80–85% of the variance** in task completion time. This is a case where the effect is both statistically significant (F = 32.67, p < 0.0001) **and** practically enormous — unlike Q7 below, where significance without meaningful effect size is the whole point of the question.

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

### Q13: Why does the F-statistic follow an F-distribution — what's the mathematical justification, not just "it's defined that way"?

**Answer:**
Under H₀, both $SS_B/\sigma^2$ and $SS_W/\sigma^2$ are independently distributed as chi-square variables (χ² with k−1 and N−k df respectively) — this independence is guaranteed by **Cochran's theorem** when the data are normally distributed and the sums of squares come from an orthogonal decomposition. The ratio of two independent chi-square variables, each divided by its own degrees of freedom, is by definition an F-distributed random variable. So F = MS_B/MS_W isn't an arbitrary test statistic — it falls directly out of the assumption structure (normality + independence) that ANOVA requires.

---

### Q14: Your 5 treatment groups have very different sample sizes (n = 200, 15, 50, 300, 10). Does this break ANOVA?

**Answer:**
Unequal n (an "unbalanced design") doesn't invalidate the F-test mathematically — the formulas still work with weighted SS_B — but it does have consequences: (1) it makes ANOVA more sensitive to violations of the equal-variance assumption, since MS_W becomes dominated by the larger groups; (2) it reduces power for detecting differences involving the smallest groups; (3) some post-hoc tests (classic Tukey's HSD) assume equal n and need a modification (Tukey-Kramer) for unbalanced designs. Best practice: check Levene's test carefully here, and prefer Games-Howell or Tukey-Kramer over vanilla Tukey's HSD.

---

### Q15: How does increasing sample size affect statistical power in ANOVA, and is there a downside?

**Answer:**
Larger N shrinks MS_W (a tighter estimate of within-group noise), which inflates F for a given true effect size — so power increases and even tiny true differences eventually become "significant." The downside is exactly the η²=0.02 scenario in Q7: with large N, statistical significance stops being informative about practical importance. This is why experimentation teams at scale (Google, Meta, Amazon) pre-register a **minimum detectable effect (MDE)** and power analysis before running the test, rather than just running until p < 0.05.

---

### Q16: What's ANCOVA and when would you reach for it instead of ANOVA?

**Answer:**
ANCOVA (Analysis of Covariance) adds one or more continuous **covariates** to the ANOVA model to statistically control for a confound before comparing group means. Example: comparing 3 training programs' effect on test scores, while controlling for **pre-test score** as a covariate — this removes variance in the outcome that's explained by prior ability, shrinking SS_Within and increasing power to detect the true treatment effect. Use ANCOVA whenever you have a measured confound that correlates with the outcome but isn't the variable you're testing.

---

### Q17: When would you use MANOVA instead of running separate ANOVAs on each outcome metric?

**Answer:**
MANOVA is appropriate when you have **multiple correlated dependent variables** measured on the same groups — e.g., testing 3 onboarding flows on both "day-7 retention" and "day-7 revenue," which are likely correlated. Running separate ANOVAs ignores that correlation and inflates the overall Type I error rate; MANOVA tests all DVs jointly using a multivariate test statistic (Wilks' Lambda, Pillai's trace) that accounts for their covariance structure. If MANOVA is significant, follow up with univariate ANOVAs or discriminant analysis to see which DV(s) actually drove the difference.

---

### Q18: A colleague ran a Bonferroni correction on 10 post-hoc pairwise comparisons at α = 0.05. What's the adjusted threshold, and what's the tradeoff?

**Answer:**
Bonferroni divides α by the number of comparisons: $\alpha_{adjusted} = 0.05 / 10 = 0.005$. Each individual pairwise test must now clear p < 0.005 to be called significant. This tightly controls the family-wise error rate, but the tradeoff is a large loss of statistical power — the more comparisons, the more conservative (and more prone to Type II errors / false negatives) Bonferroni becomes. For a large number of comparisons, Tukey's HSD or the Benjamini-Hochberg (FDR) procedure is often preferred as a less punishing alternative.

---

### Q19: Two-way ANOVA gives significant main effects for both factors but a non-significant interaction. How do you interpret and report this?

**Answer:**
A non-significant interaction means each factor's effect is **additive and consistent** across levels of the other factor — e.g., a UI change improves conversion by roughly the same amount on both mobile and desktop, and device type shifts the baseline but doesn't change the *size* of the UI effect. In this case, it's safe to report and act on the main effects independently ("roll out the UI change everywhere") without needing to segment by device. This is the opposite conclusion from Q8, where a significant interaction meant the effect had to be reported and acted on per-segment.

---

*Notes prepared for statistical testing & experimentation interviews.*
*Topics: One-Way ANOVA · Two-Way ANOVA · Repeated Measures · Post-Hoc Tests · Effect Size · FAANG QA*
