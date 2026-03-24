# MIT 18.05 — Introduction to Probability and Statistics
## Classes 18 & 19: NHST II — t-Tests, Chi-Square, ANOVA, and the Gallery of Common Tests
### Complete Study Notes | Spring 2022

> **Authors:** Jeremy Orloff and Jonathan Bloom  
> **Source:** MIT OpenCourseWare — 18.05, Spring 2022  
> **Topics Covered:** NHST review, one-sample t-test, two-sample t-test, paired t-test, one-way ANOVA (F-test), chi-square goodness-of-fit, chi-square for independence/homogeneity, multiple testing, Type I/II error analysis

---

## Table of Contents

1. [Learning Goals](#1-learning-goals)
2. [The 8-Step NHST Protocol](#2-the-8-step-nhst-protocol)
3. [Review: Types of Errors and Their Probabilities](#3-review-types-of-errors-and-their-probabilities)
4. [Understanding a Significance Test — Questions to Ask](#4-understanding-a-significance-test--questions-to-ask)
5. [The z-Test (Review and Full Reference)](#5-the-z-test-review-and-full-reference)
6. [The Student t Distribution](#6-the-student-t-distribution)
7. [One-Sample t-Test](#7-one-sample-t-test)
8. [Two-Sample t-Test (Equal Variances)](#8-two-sample-t-test-equal-variances)
9. [Two-Sample t-Test (Welch's: Unequal Variances)](#9-two-sample-t-test-welshs-unequal-variances)
10. [Paired Two-Sample t-Test](#10-paired-two-sample-t-test)
11. [One-Way ANOVA (F-Test for Equal Means)](#11-one-way-anova-f-test-for-equal-means)
12. [Chi-Square Test for Goodness of Fit](#12-chi-square-test-for-goodness-of-fit)
13. [Chi-Square Test for Independence / Homogeneity](#13-chi-square-test-for-independence--homogeneity)
14. [Multiple Testing Problem](#14-multiple-testing-problem)
15. [Critical Discussions: Type I Error Rate and Publication Bias](#15-critical-discussions-type-i-error-rate-and-publication-bias)
16. [All Worked Examples — Complete Solutions](#16-all-worked-examples--complete-solutions)
17. [Concept Questions with Deep Explanations](#17-concept-questions-with-deep-explanations)
18. [Gallery of Tests — Quick Reference](#18-gallery-of-tests--quick-reference)
19. [Common Mistakes](#19-common-mistakes)
20. [Quick Reference Summary](#20-quick-reference-summary)

---

## 1. Learning Goals

**Class 18:**
1. List the steps common to all null hypothesis significance tests.
2. Define and compute the probability of Type I and Type II errors.
3. Apply one-sample and two-sample $t$-tests.

**Class 19:**
1. Given hypotheses and data, identify the appropriate significance test.
2. Apply any significance test after looking up the details.
3. Understand chi-square tests for goodness-of-fit and independence/homogeneity.
4. Understand ANOVA (F-test) for comparing means across multiple groups.

---

## 2. The 8-Step NHST Protocol

### 2.1 Standard Steps

Every null hypothesis significance test follows this universal protocol:

**DESIGN PHASE (before collecting data):**

**Step 1:** Design an experiment and choose a test statistic $x$ to be computed from the data. Identify the null distribution $\phi(x \mid H_0)$.

**Step 2:** Decide if the test is one-sided or two-sided based on $H_A$.

**Step 3:** Choose a significance level $\alpha$ (typically 0.05, 0.01, or 0.10).

**Step 4:** Determine how much data you need to achieve the desired power.

**EXECUTION PHASE (after collecting data):**

**Step 5:** Run the experiment to collect data $x_1, x_2, \ldots, x_n$.

**Step 6:** Compute the test statistic $x$ from the data.

**Step 7:** Compute the p-value corresponding to $x$ using the null distribution.

**Step 8:** If $p < \alpha$, reject $H_0$ in favor of $H_A$. Otherwise, do not reject $H_0$.

### 2.2 Key Notes on the Protocol

> **Note 1:** Instead of choosing $\alpha$ and computing a p-value, you could choose a rejection region directly and reject $H_0$ if $x$ falls in it. The p-value approach is equivalent but more convenient.

> **Note 2:** The null hypothesis is the cautious choice. Lower $\alpha$ = more evidence required before rejection. It is standard practice to publish the p-value itself so readers can draw their own conclusions.

> **Note 3 (Critical):** A significance level of 0.05 does NOT mean the test only makes mistakes 5% of the time. It means: if $H_0$ is true, the probability of mistakenly rejecting it is 5%. Power measures accuracy when $H_A$ is true.

> **Note 4:** The p-value is conceptually a computational shortcut. The logical order is: (1) set $\alpha$, (2) define the rejection region. The p-value just tells us in one computation whether the test statistic is in the rejection region.

---

## 3. Review: Types of Errors and Their Probabilities

### 3.1 The Error Table

|  | $H_0$ is true | $H_A$ is true |
|---|---|---|
| **Reject $H_0$** | **Type I Error** | Correct (Power) |
| **Don't reject $H_0$** | Correct | **Type II Error** |

### 3.2 Probabilities

$$P(\text{Type I error}) = P(\text{test statistic in rejection region} \mid H_0) = \alpha = \text{significance level}$$

$$P(\text{Type II error}) = P(\text{test statistic NOT in rejection region} \mid H_A) = 1 - \text{Power}$$

$$\text{Power} = P(\text{test statistic in rejection region} \mid H_A)$$

### 3.3 Medical Testing Analogy

| NHST Term | Medical Screening | Criminal Justice |
|---|---|---|
| Type I Error | False positive (alarm a healthy patient) | Convicting an innocent person |
| Type II Error | False negative (miss a sick patient) | Acquitting a guilty person |
| Power | Sensitivity (true positive rate) | Correctly finding the guilty party guilty |
| Significance | False positive rate | Probability of false conviction |

### 3.4 Important Clarification: Significance ≠ P(mistake)

$$\text{Significance} = P(\text{reject } H_0 \mid H_0 \text{ is TRUE})$$

$$\text{NOT: } P(H_0 \text{ is TRUE} \mid \text{rejected } H_0)$$

The latter is a Bayesian posterior probability and requires a prior. A frequentist cannot compute this without knowing the base rate (prior probability) of $H_0$ being true.

---

## 4. Understanding a Significance Test — Questions to Ask

When evaluating any published significance test, ask these critical questions:

1. **Data collection:** How was data collected? What is the experimental setup? Is there potential for bias?

2. **Hypotheses:** What are $H_0$ and $H_A$? Are they appropriate for the scientific question?

3. **Test type:** What type of significance test was used? Does the data satisfy the assumptions needed for this test?

4. **Assumptions check:** Does the data match the criteria? (E.g., is the normality assumption justified? Are variances equal?) How robust is the test to violations?

5. **p-value computation:** How is the p-value computed? Is the test one- or two-sided? Is "at least as extreme" defined correctly?

6. **Significance level:** What $\alpha$ was used? Was it set in advance or after seeing the data?

7. **Power:** What is the power of the test? Could a failure to reject simply be due to insufficient sample size?

> **Key Warning:** A non-significant result (failing to reject $H_0$) does NOT mean $H_0$ is true. It may simply mean the sample size was too small to detect the effect.

---

## 5. The z-Test (Review and Full Reference)

### 5.1 When to Use

Use the z-test when:
- Data is normally distributed: $x_1, \ldots, x_n \sim \text{N}(\mu, \sigma^2)$
- The mean $\mu$ is unknown but the variance $\sigma^2$ **is known**
- Testing whether the population mean equals a specific value $\mu_0$

### 5.2 Full Specification

| Component | Formula/Value |
|---|---|
| **Data** | $x_1, \ldots, x_n \sim \text{N}(\mu, \sigma^2)$, $\mu$ unknown, $\sigma$ **known** |
| **$H_0$** | $\mu = \mu_0$ |
| **$H_A$** | Two-sided: $\mu \neq \mu_0$; One-sided: $\mu > \mu_0$ or $\mu < \mu_0$ |
| **Test statistic** | $z = \dfrac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$ |
| **Null distribution** | $Z \sim \text{N}(0,1)$ |

### 5.3 p-Value Formulas

$$\text{Right-sided: } p = P(Z \geq z \mid H_0) = \texttt{1-pnorm(z,0,1)}$$

$$\text{Left-sided: } p = P(Z \leq z \mid H_0) = \texttt{pnorm(z,0,1)}$$

$$\text{Two-sided: } p = P(|Z| \geq |z| \mid H_0) = \texttt{2*(1-pnorm(abs(z),0,1))}$$

### 5.4 Why the Factor of 2 for Two-Sided Tests

For a two-sided test, the rejection region has $\alpha/2$ probability in each tail. The test statistic is in the rejection region if and only if the one-sided tail probability (beyond the test statistic) is less than $\alpha/2$:

$$x \text{ in rejection region} \iff P(Z \geq z) \leq \frac{\alpha}{2} \iff 2P(Z \geq z) \leq \alpha \iff p \leq \alpha$$

So we compute $p = 2P(Z \geq z)$ for the two-sided test — the factor of 2 ensures the p-test is equivalent to the rejection region test.

### 5.5 Worked Example 1 — One-Sided z-Test

**Problem:** Data from $\text{N}(\mu, 4)$ ($\sigma^2 = 4$, $\sigma = 2$ known). $H_0$: $\mu = 2$, $H_A$: $\mu > 2$. Data: $3, 2, 5, 7, 1$. Significance $\alpha = 0.05$.

**Step 1:** Compute sample mean.
$$\bar{x} = \frac{3+2+5+7+1}{5} = \frac{18}{5} = 3.6$$

**Step 2:** Compute z-statistic.
$$z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}} = \frac{3.6 - 2}{2/\sqrt{5}} = \frac{1.6}{0.894} = 1.79$$

**Step 3:** Since $H_A$: $\mu > 2$ is right-sided, compute one-sided p-value.
$$p = P(Z > 1.79) = 1 - \Phi(1.79) = 0.037$$

**Step 4:** Since $p = 0.037 < \alpha = 0.05$: **Reject $H_0$**.

**Conclusion:** "We reject the null hypothesis in favor of the alternative hypothesis that $\mu > 2$, at significance level 0.05 with p-value 0.037."

---

### 5.6 Worked Example 1b — Two-Sided z-Test

**Same setup, but now $H_A$: $\mu \neq 2$.**

Using the same $z = 1.79 > 0$:

$$p = 2P(Z > 1.79) = 2 \times 0.037 = 0.074$$

Since $p = 0.074 > \alpha = 0.05$: **Do not reject $H_0$**.

**Important lesson:** Whether we reject or not depends critically on whether the test is one-sided or two-sided. The SAME data ($z = 1.79$) leads to rejection for a one-sided test but not for a two-sided test at the same $\alpha = 0.05$. This is why the choice of one- vs. two-sided must be made before seeing the data.

---

## 6. The Student t Distribution

### 6.1 Why Do We Need It?

The z-test requires knowing $\sigma$. In practice, $\sigma$ is almost never known — it must be estimated from the data using the sample standard deviation $s$. When we substitute $s$ for $\sigma$, the standardized statistic no longer follows $\text{N}(0,1)$; it follows a $t$-distribution.

### 6.2 Definition and Properties

> **Definition:** The $t$-distribution with parameter $df$ (degrees of freedom), written $t(df)$, is a symmetric, bell-shaped distribution centered at 0.

**Key properties:**
- Symmetric around 0 (like the standard normal)
- Heavier tails than $\text{N}(0,1)$ for small $df$
- As $df \to \infty$, $t(df) \to \text{N}(0,1)$

**Intuition for heavier tails:** When estimating $\sigma$ from a small sample, our estimate $s$ may be off — sometimes much smaller than the true $\sigma$. This inflates the t-statistic, putting more probability in the tails compared to the z-distribution where $\sigma$ is known exactly.

### 6.3 Comparison Table

| $df$ | $t_{0.025}$ (two-sided 5% critical value) |
|---|---|
| 1 | 12.71 |
| 2 | 4.30 |
| 5 | 2.57 |
| 10 | 2.23 |
| 20 | 2.09 |
| 30 | 2.04 |
| $\infty$ | 1.96 (standard normal) |

**Observation:** With $df = 5$, the critical value is 2.57 vs. 1.96 for the normal. You need a larger test statistic to reject $H_0$ when you've estimated $\sigma$ from only a few data points — the t-distribution naturally builds in this extra uncertainty.

### 6.4 R Functions for t Distribution

| Function | Purpose |
|---|---|
| `pt(x, df)` | CDF: $P(T \leq x)$ |
| `dt(x, df)` | PDF: density at $x$ |
| `qt(p, df)` | Quantile: $t$ such that $P(T \leq t) = p$ |
| `rt(n, df)` | Generate $n$ random samples |

**Example:** `pt(1.65, 3)` computes $P(T \leq 1.65)$ where $T \sim t(3)$.

---

## 7. One-Sample t-Test

### 7.1 When to Use

Use the one-sample t-test when:
- Data is normal: $x_1, \ldots, x_n \sim \text{N}(\mu, \sigma^2)$
- Both $\mu$ AND $\sigma$ are **unknown**
- Testing whether $\mu$ equals a specific value $\mu_0$

This is the most common replacement for the z-test in practice.

### 7.2 Full Specification

| Component | Formula/Value |
|---|---|
| **Data** | $x_1, \ldots, x_n \sim \text{N}(\mu, \sigma^2)$, both $\mu$ and $\sigma$ **unknown** |
| **$H_0$** | $\mu = \mu_0$ |
| **$H_A$** | Two-sided: $\mu \neq \mu_0$; or one-sided |
| **Sample variance** | $s^2 = \dfrac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$ |
| **Test statistic** | $t = \dfrac{\bar{x} - \mu_0}{s/\sqrt{n}}$ (the Studentized mean) |
| **Null distribution** | $T \sim t(n-1)$ (t-distribution with $n-1$ degrees of freedom) |

### 7.3 p-Value Formulas

$$\text{Right-sided: } p = P(T \geq t \mid H_0) = \texttt{1-pt(t, n-1)}$$

$$\text{Left-sided: } p = P(T \leq t \mid H_0) = \texttt{pt(t, n-1)}$$

$$\text{Two-sided: } p = P(|T| \geq |t| \mid H_0) = \texttt{2*(1-pt(abs(t), n-1))}$$

### 7.4 Why $n-1$ Degrees of Freedom?

We lose one degree of freedom because we use the data to estimate $\bar{x}$ first, and then compute $s^2$ based on deviations from $\bar{x}$. Since the deviations $x_i - \bar{x}$ must sum to zero, only $n-1$ of them are "free." This is why we divide by $n-1$ in $s^2$ (making it an unbiased estimator of $\sigma^2$).

### 7.5 Worked Example 2 — One-Sample t-Test

**Problem:** Data from $\text{N}(\mu, \sigma^2)$ with both $\mu$ and $\sigma$ unknown. $H_0$: $\mu = 0$, $H_A$: $\mu > 0$. Data: $1, 2, 3, 6, -1$. Significance $\alpha = 0.05$.

**Step 1:** Compute sample mean and variance.

$$\bar{x} = \frac{1+2+3+6+(-1)}{5} = \frac{11}{5} = 2.2$$

$$s^2 = \frac{1}{4}\left[(1-2.2)^2 + (2-2.2)^2 + (3-2.2)^2 + (6-2.2)^2 + (-1-2.2)^2\right]$$

$$= \frac{1}{4}\left[1.44 + 0.04 + 0.64 + 14.44 + 10.24\right] = \frac{26.8}{4} = 6.7$$

$$s = \sqrt{6.7} \approx 2.588$$

**Step 2:** Compute t-statistic (Studentized mean).

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} = \frac{2.2 - 0}{\sqrt{6.7}/\sqrt{5}} = \frac{2.2}{\sqrt{6.7/5}} = \frac{2.2}{\sqrt{1.34}} = \frac{2.2}{1.158} = 1.901$$

**Step 3:** Null distribution is $t(4)$ (since $n - 1 = 5 - 1 = 4$). Right-sided test since $H_A$: $\mu > 0$.

$$p = P(T > 1.901) = \texttt{1-pt(1.901, 4)} = 0.065$$

**Step 4:** Since $p = 0.065 > \alpha = 0.05$: **Do not reject $H_0$**.

**Comparison with z-test:** If we had known $\sigma^2 = 6.7$ exactly and used a z-test:

$$z = \frac{2.2}{\sqrt{6.7/5}} = 1.901, \quad p = P(Z > 1.901) = 0.028 < 0.05 \to \text{reject}$$

The t-test gives $p = 0.065 > 0.05$ → don't reject, while the z-test with the same statistic gives $p = 0.028 \to $ reject. This illustrates how using the t-distribution (which accounts for uncertainty in $\sigma$) makes the test more conservative — it requires stronger evidence to reject $H_0$.

**Diagram:**
```
φ(y|H₀) ~ t(4)

             ________
            /        \        ████████
___________/          \______████████___
                       1.90  2.13
         ← non-reject H₀ →|← reject →

Critical value t₀.₀₅ = 2.13 (95th percentile of t(4))
t = 1.90 < 2.13 → NOT in rejection region
p = 0.065 > 0.05 → do not reject H₀
```

---

## 8. Two-Sample t-Test (Equal Variances)

### 8.1 When to Use

Use the two-sample t-test (equal variances) when:
- Two independent groups of normal data
- Both distributions have the **same unknown variance** $\sigma^2$
- Testing whether the population means differ

**Typical use cases:** Comparing two treatments, two groups, two populations.

### 8.2 Full Specification

| Component | Formula/Value |
|---|---|
| **Data** | $x_1, \ldots, x_n \sim \text{N}(\mu_x, \sigma^2)$ and $y_1, \ldots, y_m \sim \text{N}(\mu_y, \sigma^2)$ |
| **Assumption** | Same variance $\sigma^2$ for both groups (unknown) |
| **$H_0$** | $\mu_x - \mu_y = \Delta\mu$ (usually $\Delta\mu = 0$) |
| **Pooled variance** | $s_P^2 = \dfrac{(n-1)s_x^2 + (m-1)s_y^2}{n+m-2}\left(\dfrac{1}{n} + \dfrac{1}{m}\right)$ |
| **Test statistic** | $t = \dfrac{\bar{x} - \bar{y} - \Delta\mu}{s_P}$ |
| **Null distribution** | $T \sim t(n+m-2)$ |

**Note on pooled variance notation:** Some textbooks define $s_{P,\text{other}}^2 = \frac{(n-1)s_x^2 + (m-1)s_y^2}{n+m-2}$ (without the $1/n + 1/m$ factor). Our $s_P^2$ includes this factor and represents the estimated variance of $\bar{x} - \bar{y}$.

### 8.3 Intuition for the Pooled Variance

> **Intuition:** We have two independent estimates of the same variance $\sigma^2$: $s_x^2$ from the $x$ group and $s_y^2$ from the $y$ group. The pooled variance is a weighted average of these two estimates, weighted by their degrees of freedom. More data in a group → that group's estimate gets more weight.

### 8.4 p-Values (Same Structure as One-Sample)

$$\text{Two-sided: } p = P(|T| > |t| \mid H_0) = \texttt{2*(1-pt(abs(t), n+m-2))}$$

$$\text{Right-sided: } p = P(T > t \mid H_0) = \texttt{1-pt(t, n+m-2)}$$

### 8.5 R Code

```r
t.test(x, y, var.equal=TRUE, alternative="two.sided")
```

### 8.6 Worked Example 3 — Two-Sample t-Test (Maternity Hospital)

**Problem:** 1408 women admitted to a maternity hospital.
- Medical group: $n = 775$, $\bar{x}_M = 39.08$, $s_M^2 = 7.77$
- Emergency group: $m = 633$, $\bar{x}_E = 39.60$, $s_E^2 = 4.95$

Test whether mean pregnancy duration differs between the two groups. $H_0$: $\mu_M = \mu_E$, $H_A$: $\mu_M \neq \mu_E$ (two-sided).

**Step 1:** Compute pooled variance.

$$s_{P,\text{other}}^2 = \frac{774 \times 7.77 + 632 \times 4.95}{1406} = \frac{6014.0 + 3128.4}{1406} = \frac{9142.4}{1406} = 6.503$$

Including the $1/n + 1/m$ factor:

$$s_P^2 = 6.503 \times \left(\frac{1}{775} + \frac{1}{633}\right) = 6.503 \times 0.002873 = 0.01868$$

$$s_P = \sqrt{0.01868} = 0.1367$$

**Step 2:** Compute t-statistic.

$$t = \frac{\bar{x}_M - \bar{x}_E}{s_P} = \frac{39.08 - 39.60}{0.1367} = \frac{-0.52}{0.1367} = -3.806$$

**Step 3:** Degrees of freedom: $df = 775 + 633 - 2 = 1406$.

**Step 4:** Compute p-value.

With 1406 degrees of freedom, $t(1406) \approx \text{N}(0,1)$. Our $|t| = 3.806$ is nearly 4 standard deviations from the mean.

$$p = P(|T| > 3.806) \approx P(|Z| > 3.806) \approx 0.00015$$

This is far smaller than $\alpha = 0.05$ or $\alpha = 0.01$.

**Step 5:** **Reject $H_0$**.

**Conclusion:** "We reject the null hypothesis in favor of the alternative that there is a difference in mean pregnancy duration between medical and emergency admissions."

**Assumptions made:**
- Both distributions are approximately normal ✓ (large sample supports this via CLT)
- Independence between the two groups ✓ (different patients)
- Equal variances: $s_M^2 = 7.77$ vs $s_E^2 = 4.95$ — the large discrepancy in sample variances raises concern about this assumption. An F-test for equality of variances should ideally be run first.

---

## 9. Two-Sample t-Test (Welch's: Unequal Variances)

### 9.1 When to Use

Use Welch's t-test when:
- Two independent groups of normal data
- Variances are **not assumed equal** (or you're not sure if they're equal)

Many statisticians recommend always using Welch's test rather than the equal-variance test, as it costs very little when variances are equal and protects against violations when they're not.

### 9.2 Full Specification

| Component | Formula/Value |
|---|---|
| **Data** | $x_i \sim \text{N}(\mu_x, \sigma_x^2)$, $y_j \sim \text{N}(\mu_y, \sigma_y^2)$, different variances |
| **$H_0$, $H_A$** | Same as equal-variance case |
| **Pooled variance** | $s_P^2 = \dfrac{s_x^2}{n} + \dfrac{s_y^2}{m}$ |
| **Test statistic** | $t = \dfrac{\bar{x} - \bar{y} - \Delta\mu}{s_P}$ |
| **Degrees of freedom** | $df = \dfrac{(s_x^2/n + s_y^2/m)^2}{(s_x^2/n)^2/(n-1) + (s_y^2/m)^2/(m-1)}$ (Welch-Satterthwaite) |
| **Null distribution** | $T \sim t(df)$ (note: $df$ is typically non-integer) |

### 9.3 R Code

```r
t.test(x, y, var.equal=FALSE, alternative="two.sided")  # var.equal=FALSE is the default
```

---

## 10. Paired Two-Sample t-Test

### 10.1 When to Use

Use the paired t-test when:
- Data naturally comes in pairs $(x_i, y_i)$ (same subject measured twice, or matched subjects)
- You want to test whether the mean difference is zero

### 10.2 The Key Idea

Instead of comparing two separate groups, compute the **differences** $w_i = x_i - y_i$ and run a **one-sample t-test** on $w_i$ with $H_0$: $\mu_w = 0$.

The paired test is more powerful than an unpaired two-sample t-test when the within-pair correlation is positive (as is typical in before/after designs) because pairing removes subject-to-subject variability.

### 10.3 Full Specification

| Component | Formula/Value |
|---|---|
| **Data** | Pairs $(x_1, y_1), \ldots, (x_n, y_n)$; differences $w_i = x_i - y_i$ |
| **Assumption** | $w_i \sim \text{N}(\mu, \sigma^2)$ independently (both $\mu, \sigma$ unknown) |
| **$H_0$** | $\mu = \mu_0$ (usually $\mu_0 = 0$: no effect) |
| **$H_A$** | Two-sided: $\mu \neq \mu_0$; or one-sided |
| **Test statistic** | $t = \dfrac{\bar{w} - \mu_0}{s/\sqrt{n}}$ where $s^2 = \dfrac{1}{n-1}\sum(w_i - \bar{w})^2$ |
| **Null distribution** | $T \sim t(n-1)$ |

### 10.4 Examples of Paired Data

**Example 5 — Cholesterol study:** Measure each subject's cholesterol before and after drug treatment. The pair is (before, after) for the same person.

**Example 6 — Cancer treatment:** Match each treated subject with an untreated subject of similar disease stage, age, and sex. The pair is (treated, matched control).

> **Why pair?** Pairing controls for individual variation. If some people naturally have high cholesterol, an unpaired test would confuse individual variation with treatment effect. Pairing removes this confounder by looking only at the change for each person.

### 10.5 Worked Example 7 — Cigarette Smoking and Platelet Aggregation

**Problem:** Blood from 11 subjects measured before and after smoking a cigarette.

| | Before (B) | After (A) | Difference ($w = A - B$) |
|---|---|---|---|
| 1 | 25 | 27 | 2 |
| 2 | 25 | 29 | 4 |
| 3 | 27 | 37 | 10 |
| 4 | 44 | 56 | 12 |
| 5 | 30 | 46 | 16 |
| 6 | 67 | 82 | 15 |
| 7 | 53 | 57 | 4 |
| 8 | 53 | 80 | 27 |
| 9 | 52 | 61 | 9 |
| 10 | 60 | 59 | -1 |
| 11 | 28 | 43 | 15 |

$H_0$: $\mu_w = 0$ (no effect), $H_A$: $\mu_w \neq 0$ (two-sided).

**R code:**
```r
before.cig <- c(25,25,27,44,30,67,53,53,52,60,28)
after.cig  <- c(27,29,37,56,46,82,57,80,61,59,43)
t.test(after.cig, before.cig, alternative="two.sided", mu=0, paired=TRUE)
```

**R output:**
```
Paired t-test
t = 4.2716, df = 10, p-value = 0.001633
mean of the differences: 10.27273
```

**Step 1:** Compute mean difference.
$$\bar{w} = \frac{2+4+10+12+16+15+4+27+9+(-1)+15}{11} = \frac{113}{11} = 10.27$$

**Step 2:** With $t = 4.27$ and $df = 10$, the two-sided p-value is 0.0016.

**Step 3:** Since $p = 0.0016 \ll 0.05$: **Reject $H_0$**.

**Conclusion:** Cigarette smoking significantly increases platelet aggregation (p = 0.0016). The mean increase is 10.27 units.

**Equivalently:** `t.test(after.cig - before.cig, mu=0)` gives the same result — the paired t-test is exactly a one-sample t-test on the differences.

---

## 11. One-Way ANOVA (F-Test for Equal Means)

### 11.1 When to Use

Use one-way ANOVA when:
- You have $n \geq 3$ groups (for 2 groups, use the two-sample t-test)
- Each group has independent normal data with the same variance
- Testing whether all group means are equal

> **Why not run multiple t-tests?** With 6 groups, there are $\binom{6}{2} = 15$ pairwise comparisons. At $\alpha = 0.05$ each, the probability of at least one false rejection is much higher than 5% (the multiple testing problem — see Section 14).

### 11.2 Full Specification

| Component | Formula/Value |
|---|---|
| **Data** | $n$ groups, $m$ obs. per group: $x_{i,j} \sim \text{N}(\mu_i, \sigma^2)$ |
| **Assumption** | Independent normal data, same variance $\sigma^2$ across groups |
| **$H_0$** | $\mu_1 = \mu_2 = \cdots = \mu_n$ |
| **$H_A$** | Not all means are equal |

### 11.3 Key Quantities

**Group means and grand mean:**

$$\bar{x}_i = \frac{1}{m}\sum_{j=1}^m x_{i,j}, \qquad \bar{x} = \frac{1}{nm}\sum_i\sum_j x_{i,j}$$

**Between-group variance (MS$_B$):** Measures how much group means vary around the grand mean.

$$\text{MS}_B = \frac{m}{n-1}\sum_{i=1}^n (\bar{x}_i - \bar{x})^2$$

**Within-group variance (MS$_W$):** Average of the within-group sample variances; measures random variation within each group.

$$\text{MS}_W = \frac{s_1^2 + s_2^2 + \cdots + s_n^2}{n}, \quad \text{where } s_i^2 = \frac{1}{m-1}\sum_{j=1}^m (x_{i,j} - \bar{x}_i)^2$$

### 11.4 The F-Statistic

$$\boxed{f = \frac{\text{MS}_B}{\text{MS}_W}}$$

**Intuition:** If all means are equal ($H_0$ true), both MS$_B$ and MS$_W$ estimate the same $\sigma^2$, so $f \approx 1$. If means differ ($H_A$ true), MS$_B$ grows (because group means are spread out) while MS$_W$ stays the same, so $f > 1$.

**Null distribution:** $F \sim F(n-1,\; n(m-1))$ (the F-distribution with numerator df = $n-1$ and denominator df = $n(m-1)$).

**p-value:** Always right-sided (large $f$ is evidence against $H_0$):

$$p = P(F > f \mid H_0) = \texttt{1-pf(f, n-1, n*(m-1))}$$

### 11.5 The F-Distribution

The F-distribution is the ratio of two chi-square random variables divided by their degrees of freedom. It is right-skewed and always non-negative.

| Distribution | Shape |
|---|---|
| $F(3,4)$ | Very skewed, heavy right tail |
| $F(10,15)$ | Less skewed |
| $F(30,15)$ | Nearly symmetric |

### 11.6 Worked Example 8 — Pain Levels After Medical Procedures

**Problem:** Pain levels (scale 1–6) after 3 treatments.

| $T_1$ | $T_2$ | $T_3$ |
|---|---|---|
| 2 | 3 | 2 |
| 4 | 4 | 1 |
| 1 | 6 | 3 |
| 5 | 1 | 3 |
| 3 | 4 | 5 |

$H_0$: $\mu_1 = \mu_2 = \mu_3$, $H_A$: not all means equal.

**R code:**
```r
T1 <- c(2,4,1,5,3); T2 <- c(3,4,6,1,4); T3 <- c(2,1,3,3,5)
procedure <- c(rep('T1',5), rep('T2',5), rep('T3',5))
pain <- c(T1, T2, T3)
data.pain <- data.frame(procedure, pain)
aov.data <- aov(pain ~ procedure, data=data.pain)
summary(aov.data)
```

**Output:** $F = 0.325$, $p = 0.729$.

**Decision:** $p = 0.729 \gg 0.05$. **Fail to reject $H_0$**.

**Careful interpretation:** We do NOT conclude that all three treatments have equal mean pain levels. With only 5 data points per procedure, we may simply lack the power to detect differences that exist.

---

### 11.7 Board Problem — Recovery Time (Numerical ANOVA)

**Problem:** Recovery times (days) for 3 treatments, 6 patients each.

| $T_1$ | $T_2$ | $T_3$ |
|---|---|---|
| 6 | 8 | 13 |
| 8 | 12 | 9 |
| 4 | 9 | 11 |
| 5 | 11 | 8 |
| 3 | 6 | 7 |
| 4 | 8 | 12 |

$H_0$: $\mu_1 = \mu_2 = \mu_3$, $H_A$: not all equal. Significance $\alpha = 0.05$.

**Step 1:** Compute group means.

$$\bar{x}_1 = \frac{6+8+4+5+3+4}{6} = \frac{30}{6} = 5, \quad \bar{x}_2 = \frac{8+12+9+11+6+8}{6} = \frac{54}{6} = 9, \quad \bar{x}_3 = \frac{13+9+11+8+7+12}{6} = \frac{60}{6} = 10$$

$$\bar{x} = \frac{30+54+60}{18} = \frac{144}{18} = 8$$

**Step 2:** Compute MS$_B$ (between-group variance).

$$\text{MS}_B = \frac{m}{n-1}\sum_{i=1}^3 (\bar{x}_i - \bar{x})^2 = \frac{6}{2}\left[(5-8)^2 + (9-8)^2 + (10-8)^2\right] = 3[9 + 1 + 4] = 42$$

**Step 3:** Compute within-group sample variances.

$T_1$: deviations from 5: $(1, 3, -1, 0, -2, -1)$, $s_1^2 = (1+9+1+0+4+1)/5 = 16/5 = 3.2$  
$T_2$: deviations from 9: $(-1, 3, 0, 2, -3, -1)$, $s_2^2 = (1+9+0+4+9+1)/5 = 24/5 = 4.8$  
$T_3$: deviations from 10: $(3, -1, 1, -2, -3, 2)$, $s_3^2 = (9+1+1+4+9+4)/5 = 28/5 = 5.6$

$$\text{MS}_W = \frac{s_1^2 + s_2^2 + s_3^2}{3} = \frac{3.2 + 4.8 + 5.6}{3} = \frac{13.6}{3} \approx 4.533$$

*(The PDF states MS$_W = 68/15 \approx 4.533$, consistent with this.)*

**Step 4:** Compute F-statistic.

$$f = \frac{\text{MS}_B}{\text{MS}_W} = \frac{42}{68/15} = \frac{42 \times 15}{68} = \frac{630}{68} \approx 9.26$$

**Step 5:** Null distribution: $F(n-1, n(m-1)) = F(2, 15)$. Critical value at $\alpha = 0.05$: $F_{0.05} = 3.68$.

$$p = P(F_{2,15} > 9.26) = \texttt{1-pf(9.26, 2, 15)} \approx 0.0024$$

**Step 6:** Since $f = 9.26 > 3.68$ and $p = 0.0024 < 0.05$: **Reject $H_0$**.

**Conclusion:** There is significant evidence that the mean recovery times differ across the three treatments.

---

## 12. Chi-Square Test for Goodness of Fit

### 12.1 When to Use

Use the chi-square goodness-of-fit test when:
- You have counts of categorical outcomes
- You want to test whether these counts fit a specific hypothesized distribution

**Examples:** Does the distribution of dice rolls fit a fair die? Do genetic phenotype counts match Mendel's predicted ratios? Do customer arrivals by day fit the owner's claimed distribution?

### 12.2 Full Specification

| Component | Formula/Value |
|---|---|
| **Data** | Observed counts $O_i$ for each outcome $\omega_i$ ($i = 1, \ldots, k$) |
| **$H_0$** | Data drawn from a specific distribution with probabilities $p_1, \ldots, p_k$ |
| **Expected counts** | $E_i = N \cdot p_i$ where $N = \sum O_i$ is total count |
| **G statistic** | $G = 2\sum_i O_i \ln(O_i/E_i)$ (likelihood ratio) |
| **$X^2$ statistic** | $X^2 = \sum_i (O_i - E_i)^2/E_i$ (Pearson's) |
| **Degrees of freedom** | $df = k - 1 - (\text{number of parameters estimated from data})$ |
| **Null distribution** | Both $G$ and $X^2$ approximately follow $\chi^2(df)$ under $H_0$ |
| **p-value** | $p = P(\chi^2 > G) = \texttt{1-pchisq(G, df)}$ |

### 12.3 Degrees of Freedom — The Key Concept

> **Computing $df$:** Start with $k$ cells. Subtract 1 for the constraint that counts sum to total $N$. Subtract 1 more for each parameter you estimated from the data.
>
> $$df = k - 1 - (\text{parameters estimated from data})$$

**Examples:**
- $k = 6$ outcomes, no parameters estimated: $df = 6 - 1 = 5$
- $k = 6$ outcomes, estimated $\theta$ from data: $df = 6 - 1 - 1 = 4$
- $k = 4$ outcomes (e.g., $2 \times 2$ table), no estimated parameters: $df = 4 - 1 = 3$

### 12.4 G vs. $X^2$: Which to Use?

Both statistics measure how far observed counts are from expected counts. Under $H_0$, both approximately follow $\chi^2(df)$ for large samples.

- $G$ (likelihood ratio) is theoretically preferred and is more robust.
- $X^2$ (Pearson's) was historically used because it's easier to compute by hand.
- For large samples, $G \approx X^2$ and both give the same conclusion.
- In borderline cases, they may differ (see Example 10).

### 12.5 Worked Example 9 — Binomial Goodness-of-Fit

**Problem:** 51 trials with outcomes in $\{0, 1, 2, 3, 4, \geq 5\}$.

| Outcome | 0 | 1 | 2 | 3 | 4 | $\geq 5$ |
|---|---|---|---|---|---|---|
| Observed | 3 | 10 | 15 | 13 | 7 | 3 |

$H_0$: data from $\text{Binomial}(8, 0.5)$. $H_A$: from some other distribution.

**Step 1:** Compute expected counts from Binomial$(8, 0.5)$.

$$E_i = 51 \times P(\text{Binomial}(8,0.5) = i)$$

| Outcome | $H_0$ Probability | Expected |
|---|---|---|
| 0 | 0.0039 | 0.20 |
| 1 | 0.0313 | 1.59 |
| 2 | 0.1094 | 5.58 |
| 3 | 0.2188 | 11.16 |
| 4 | 0.2734 | 13.95 |
| $\geq 5$ | 0.3633 | 18.53 |

**Step 2:** Compute statistics.

$$X^2 = \sum \frac{(O_i - E_i)^2}{E_i} = \frac{(3-0.20)^2}{0.20} + \cdots = 116.41$$

$$G = 2\sum O_i \ln(O_i/E_i) = 66.08$$

**Step 3:** Degrees of freedom: 6 cells, 1 constraint (total = 51). $df = 6 - 1 = 5$.

**Step 4:** p-values are essentially 0 for both statistics.

**Decision:** Reject $H_0$ at any reasonable significance level. The data does not fit a Binomial$(8, 0.5)$ distribution.

---

### 12.6 Worked Example 10 — Degrees of Freedom with Estimated Parameter

**Problem:** Observed counts:

| Outcome | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Observed | 6 | 9 | 13 | 12 | 7 | 3 |

$H_0$: data from Binomial$(5, \theta)$ with $\theta$ **unknown**. $H_A$: other distribution.

**Step 1:** Estimate $\theta$ from data.

$$\text{Sample mean} = \frac{6(0)+9(1)+13(2)+12(3)+7(4)+3(5)}{50} = \frac{114}{50} = 2.28$$

Since $E[\text{Binomial}(5,\theta)] = 5\theta$:

$$\hat\theta = \frac{2.28}{5} = 0.456$$

**Step 2:** Compute expected counts using Binomial$(5, 0.456)$.

| Outcome | Expected |
|---|---|
| 0 | 2.38 |
| 1 | 9.98 |
| 2 | 16.74 |
| 3 | 14.03 |
| 4 | 5.88 |
| 5 | 0.99 |

**Step 3:** Degrees of freedom.

- 6 cells
- 2 constraints: (1) total count = 50, (2) estimated $\hat\theta$ from the mean
- $df = 6 - 2 = 4$

**Step 4:** Compute statistics and p-values.

$$G = 8.01, \quad p_G = \texttt{1-pchisq(8.01, 4)} = 0.09$$

$$X^2 = 11.05, \quad p_{X^2} = \texttt{1-pchisq(11.05, 4)} = 0.026$$

**Divergent conclusion:** $G$ gives $p = 0.09 > 0.05$ → don't reject; $X^2$ gives $p = 0.026 < 0.05$ → reject! This is a borderline case where the choice of statistic matters. In such cases, prefer $G$ as it is more theoretically sound.

**Lesson:** For borderline cases, the most we can say is that the data does not strongly support rejecting $H_0$.

---

### 12.7 Worked Example 11 — Mendel's Genetics (Famous Historical Example)

**Background:** Mendel crossed $F_1$ plants with genes $SsYy$ (smooth/yellow dominant). Under Mendel's laws of independent assortment, the $F_2$ generation should show a 9:3:3:1 ratio.

| Phenotype | Prob. | Expected (from 556) | Observed |
|---|---|---|---|
| Smooth Yellow | 9/16 | 312.75 | 315 |
| Smooth Green | 3/16 | 104.25 | 108 |
| Wrinkled Yellow | 3/16 | 104.25 | 102 |
| Wrinkled Green | 1/16 | 34.75 | 31 |

$H_0$: data follows Mendel's 9:3:3:1 distribution.

**Statistics:**

$$G = 2\left[315\ln\!\frac{315}{312.75} + 108\ln\!\frac{108}{104.25} + 102\ln\!\frac{102}{104.25} + 31\ln\!\frac{31}{34.75}\right] = 0.618$$

$$X^2 = \frac{2.75^2}{312.75} + \frac{3.75^2}{104.25} + \frac{3.75^2}{104.25} + \frac{2.25^2}{34.75} = 0.604$$

**Degrees of freedom:** 4 cells, 1 constraint (total = 556), 0 estimated parameters. $df = 3$.

**p-value:** $p = \texttt{1-pchisq(0.618, 3)} = 0.892$.

**Decision:** Do NOT reject $H_0$. The data fits Mendel's predicted 9:3:3:1 ratio extremely well (p = 0.89 means results this close to expected would occur 89% of the time even by chance).

**Historical note:** Mendel's data famously fits too well — the p-value is suspiciously high. Some statisticians have argued that the data was manipulated or that Mendel stopped recording data once he had enough to confirm his theory.

---

## 13. Chi-Square Test for Independence / Homogeneity

### 13.1 Chi-Square for Independence

**When to use:** Test whether two categorical variables are independent in a population, using a contingency table.

**Setup:** A contingency table with $r$ rows and $c$ columns, where each cell contains the observed count for a combination of the two variables.

**$H_0$:** The two variables are independent (cell probabilities = product of marginal probabilities).

**Expected counts under $H_0$:**

$$E_{ij} = \frac{(\text{row } i \text{ total}) \times (\text{column } j \text{ total})}{\text{grand total}}$$

**Degrees of freedom:** $df = (r-1)(c-1)$

**Test statistics and p-values:** Same formulas as goodness-of-fit.

### 13.2 Chi-Square for Homogeneity

**When to use:** Test whether several independent datasets are drawn from the same distribution.

**Setup:** $m$ groups, each with counts for $n$ possible outcomes. Organize in an $m \times n$ table.

**$H_0$:** All groups are drawn from the same distribution (unspecified).

**Degrees of freedom:** $df = (m-1)(n-1)$ (same formula as independence test).

### 13.3 Worked Example — Chi-Square for Independence (Education and Marriage)

**Problem (from Rice):** 1436 women surveyed on education level and number of marriages.

| Education | Married once | Married multiple times | Total |
|---|---|---|---|
| College | 550 | 61 | 611 |
| No college | 681 | 144 | 825 |
| Total | 1231 | 205 | 1436 |

$H_0$: Education and number of marriages are independent. $\alpha = 0.01$.

**Step 1:** Compute expected counts under $H_0$ using marginal totals.

$$E_{11} = \frac{611 \times 1231}{1436} = \frac{751,741}{1436} = 523.5$$

$$E_{12} = \frac{611 \times 205}{1436} = \frac{125,255}{1436} = 87.2$$

$$E_{21} = \frac{825 \times 1231}{1436} = \frac{1,015,575}{1436} = 707.2$$

$$E_{22} = \frac{825 \times 205}{1436} = \frac{169,125}{1436} = 117.8$$

**Step 2:** Observed vs. expected:

| | Married once | Married multiple times |
|---|---|---|
| College | (550, 523.5) | (61, 87.2) |
| No college | (681, 707.2) | (144, 117.8) |

**Step 3:** Compute statistics.

$$X^2 = \frac{(550-523.5)^2}{523.5} + \frac{(61-87.2)^2}{87.2} + \frac{(681-707.2)^2}{707.2} + \frac{(144-117.8)^2}{117.8}$$

$$= \frac{702.25}{523.5} + \frac{686.44}{87.2} + \frac{686.44}{707.2} + \frac{686.44}{117.8} = 1.34 + 7.87 + 0.97 + 5.83 = 16.01$$

$$G = 16.55$$

**Step 4:** Degrees of freedom: $(2-1)(2-1) = 1$.

**Step 5:** p-value.

$$p = \texttt{1-pchisq(16.55, 1)} = 0.000047$$

**Decision:** $p \ll 0.01$. **Reject $H_0$**. Education level and number of marriages are NOT independent — they are significantly associated.

---

### 13.4 Worked Example — Chi-Square for Homogeneity (Shakespeare Authorship)

**Problem:** Test whether a "long lost work" has the same word frequencies as King Lear.

| Word | a | an | this | that | Total |
|---|---|---|---|---|---|
| King Lear | 150 | 30 | 30 | 90 | 300 |
| Long lost work | 90 | 20 | 10 | 80 | 200 |
| Total | 240 | 50 | 40 | 170 | 500 |

$H_0$: Both texts have the same relative word frequencies. Significance $\alpha = 0.1$.

**Step 1:** Estimate common relative frequencies under $H_0$ (total counts / grand total).

| Word | Relative freq. |
|---|---|
| a | 240/500 = 0.48 |
| an | 50/500 = 0.10 |
| this | 40/500 = 0.08 |
| that | 170/500 = 0.34 |

**Step 2:** Expected counts = (row total) × (relative frequency).

| | a | an | this | that |
|---|---|---|---|---|
| King Lear expected | 144 | 30 | 24 | 102 |
| Lost work expected | 96 | 20 | 16 | 68 |

**Step 3:** Compute $X^2$.

$$X^2 = \frac{(150-144)^2}{144} + \frac{(30-30)^2}{30} + \frac{(30-24)^2}{24} + \frac{(90-102)^2}{102}$$
$$+ \frac{(90-96)^2}{96} + \frac{(20-20)^2}{20} + \frac{(10-16)^2}{16} + \frac{(80-68)^2}{68}$$

$$= \frac{36}{144} + 0 + \frac{36}{24} + \frac{144}{102} + \frac{36}{96} + 0 + \frac{36}{16} + \frac{144}{68}$$

$$= 0.25 + 0 + 1.50 + 1.41 + 0.375 + 0 + 2.25 + 2.12 \approx 7.9$$

**Step 4:** Degrees of freedom: $df = (m-1)(n-1) = (2-1)(4-1) = 3$.

**Step 5:** $p = \texttt{1-pchisq(7.9, 3)} = 0.048$.

**Decision:** $p = 0.048 < \alpha = 0.1$. **Reject $H_0$**.

**Scientific conclusion:** The word frequency patterns differ significantly. Assuming all of Shakespeare's works have similar word distributions, this suggests the long lost work was probably NOT written by Shakespeare.

---

### 13.5 Board Problem — Genetic Linkage (Bateson, Saunders, Punnett 1905)

**Problem:** Sweet pea plants. $H_0$: Color and pollen shape are inherited independently (Mendel's law of independent assortment).

**Expected under $H_0$ (independent assortment):**

| | Long | Round | Total |
|---|---|---|---|
| Purple | 9/16 | 3/16 | 3/4 |
| Red | 3/16 | 1/16 | 1/4 |
| Total | 3/4 | 1/4 | 1 |

From 2132 plants:

| Phenotype | Expected | Observed |
|---|---|---|
| Purple, Long | $2132 \times 9/16 = 1199$ | 1528 |
| Purple, Round | $2132 \times 3/16 = 400$ | 106 |
| Red, Long | $2132 \times 3/16 = 400$ | 117 |
| Red, Round | $2132 \times 1/16 = 133$ | 381 |

**Statistics:** $G = 972.0$, $X^2 = 966.6$. Both p-values are effectively 0.

**Decision:** Decisively reject $H_0$.

**Biological interpretation:** The data violates Mendel's law of independent assortment for color and shape. There are far more purple-long (dominant) and red-round (recessive) plants than expected, and far fewer of the mixed types. This is evidence for **genetic linkage** — the genes for color and shape are on the same chromosome and tend to be inherited together. This was one of the early discoveries of genetic linkage.

---

## 14. Multiple Testing Problem

### 14.1 The Problem

When running many tests simultaneously, the probability of at least one false rejection (Type I error) grows substantially, even if each individual test is run at $\alpha = 0.05$.

### 14.2 Worked Concept Question — Multiple t-Tests

**Problem:** 6 treatments; test if all have the same mean recovery time. Using pairwise two-sample t-tests.

**(a) How many tests?**

$$\binom{6}{2} = \frac{6!}{2!\,4!} = 15 \text{ tests}$$

**(b) Probability of at least one false rejection at $\alpha = 0.05$?**

For 3 independent tests (e.g., pairs 1-2, 3-4, 5-6):

$$P(\text{at least one false rejection}) = 1 - P(\text{no false rejections}) = 1 - (0.95)^3 = 1 - 0.857 = 0.143$$

The other 12 tests are not independent of these, but they can only increase the probability further. In simulations, the false rejection rate for all 15 tests is approximately **0.36** — much higher than the intended 0.05.

> **Key Insight:** The more tests you run, the more likely you are to find a "significant" result purely by chance. This is why ANOVA tests all means simultaneously — it controls the Type I error rate at the global level.

### 14.3 Solutions to Multiple Testing

- **ANOVA (F-test):** Tests all means simultaneously with a single test at level $\alpha$.
- **Bonferroni correction:** Use significance level $\alpha/k$ for each of $k$ tests.
- **False Discovery Rate (FDR) control:** For high-dimensional data (e.g., genomics).

---

## 15. Critical Discussions: Type I Error Rate and Publication Bias

### 15.1 Discussion 1 — What Fraction of Published Papers Have Type I Errors?

**Question:** A journal only publishes results significant at $\alpha = 0.05$. What percentage of its papers contain Type I errors?

**Answer:** **Unknown without a prior.**

This question asks for $P(H_0 \mid \text{rejected } H_0)$ — the probability that $H_0$ is true given we rejected it. This requires knowing the **base rate** (prior probability) of $H_0$ being true.

$$P(H_0 \mid \text{rejected}) = \frac{P(\text{rejected} \mid H_0)\, P(H_0)}{P(\text{rejected})} = \frac{\alpha \cdot P(H_0)}{\alpha \cdot P(H_0) + \text{Power} \cdot P(H_A)}$$

Without knowing $P(H_0)$, the answer could be anywhere from 0% to 100%.

> **Lesson:** The significance level alone does not tell you how often published findings are true. The base rate of true effects in a field matters enormously. In fields where most tested hypotheses are true (e.g., physics), false positives are rare. In fields where most hypotheses are speculative (e.g., early-stage biomedical research), false positives may dominate.

---

### 15.2 Discussion 2 — Jerry (Treatments Never Work)

**Scenario:** Jerry tests treatments that are **never effective**. He uses $\alpha = 0.05$ and publishes when $p < 0.05$.

**(a) What percentage of his experiments result in publications?**

Since all his $H_0$'s are true, he only rejects $H_0$ due to chance. By the definition of $\alpha$:

$$P(\text{publish}) = P(\text{reject } H_0 \mid H_0 \text{ true}) = \alpha = 0.05 = 5\%$$

**(b) What percentage of his published papers contain Type I errors?**

All of his experiments have $H_0$ true (his treatments never work). Therefore, every rejection is a Type I error:

$$P(H_0 \mid \text{published}) = 1 = 100\%$$

All of his published papers describe spurious effects.

---

### 15.3 Discussion 3 — Jen (Treatments Always Work)

**Scenario:** Jen tests treatments that are **always effective**. She uses $\alpha = 0.05$.

**(a) What percentage result in publications?**

This depends on **power**. If her treatments produce a tiny effect barely above placebo, power might be close to $\alpha = 5\%$. If her treatments are dramatically better, power might approach 100%.

$$P(\text{publish}) = P(\text{reject } H_0 \mid H_A \text{ true}) = \text{Power}$$

Without knowing the effect size, we can only say: "it depends on power."

**(b) What percentage of her published papers contain Type I errors?**

Since all her treatments are effective ($H_A$ is always true), no rejection can be a Type I error:

$$P(H_0 \mid \text{published}) = 0 = 0\%$$

None of her published papers contain Type I errors.

> **Lesson from Jerry and Jen:** The same $\alpha = 0.05$ level leads to 100% false positives for Jerry and 0% for Jen. The significance level alone doesn't determine the quality of published science — the base rate of true effects in the field is equally important.

---

## 16. All Worked Examples — Complete Solutions

### 16.1 Board Problem 1 — Significance Level and Power (Class 18)

**Setup:** Data $x \sim \text{Binomial}(\theta, 10)$. Rejection region: $\{0, 1, 2, 8, 9, 10\}$.

| $x$ | **0** | **1** | **2** | 3 | 4 | 5 | 6 | 7 | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $\theta=0.5$ | **0.001** | **0.010** | **0.044** | 0.117 | 0.205 | 0.246 | 0.205 | 0.117 | **0.044** | **0.010** | **0.001** |
| $\theta=0.6$ | 0.000 | 0.002 | 0.011 | 0.042 | 0.111 | 0.201 | 0.251 | 0.215 | **0.121** | **0.040** | **0.006** |
| $\theta=0.7$ | 0.000 | 0.000 | 0.001 | 0.009 | 0.037 | 0.103 | 0.200 | 0.267 | **0.233** | **0.121** | **0.028** |

**(a) Significance level:**

$$\alpha = P(\text{rejection region} \mid H_0) = 0.001+0.010+0.044+0.044+0.010+0.001 = \mathbf{0.11}$$

**(b) Power:**

$$\text{Power}(\theta=0.6) = 0.000+0.002+0.011+0.121+0.040+0.006 = \mathbf{0.180}$$

$$\text{Power}(\theta=0.7) = 0.000+0.000+0.001+0.233+0.121+0.028 = \mathbf{0.383}$$

**(c) Probabilities of errors:**

$$P(\text{Type I error}) = \alpha = \mathbf{0.11}$$

$$P(\text{Type II error} \mid \theta=0.6) = 1 - \text{Power}(0.6) = 1 - 0.180 = \mathbf{0.820}$$

$$P(\text{Type II error} \mid \theta=0.7) = 1 - \text{Power}(0.7) = 1 - 0.383 = \mathbf{0.617}$$

---

### 16.2 Board Problem 2 — z and One-Sample t-Test (Class 18)

**Setup:** Data 2, 4, 4, 10 from $\text{N}(\mu, \sigma^2)$. $H_0$: $\mu = 0$, $H_A$: $\mu \neq 0$. $\alpha = 0.05$.

**Preliminary calculations:**

$$\bar{x} = \frac{2+4+4+10}{4} = \frac{20}{4} = 5$$

$$s^2 = \frac{(2-5)^2+(4-5)^2+(4-5)^2+(10-5)^2}{3} = \frac{9+1+1+25}{3} = \frac{36}{3} = 12, \quad s = \sqrt{12} = 2\sqrt{3}$$

**(a) One or two-sided?**

**Two-sided.** The alternative $\mu \neq 0$ says the true mean could be either above or below 0. Data far in either direction is evidence against $H_0$.

**(b) z-test (assuming $\sigma^2 = 16$ known):**

$$z = \frac{\bar{x} - 0}{\sigma/\sqrt{n}} = \frac{5}{4/\sqrt{4}} = \frac{5}{4/2} = \frac{5}{2} = 2.5$$

Rejection region for two-sided $\alpha = 0.05$: $|z| > 1.96$.

Since $|2.5| = 2.5 > 1.96$: **Reject $H_0$**.

P-value verification:

$$p = 2P(Z \geq 2.5) = 2 \times (1 - \Phi(2.5)) = 2 \times 0.0062 = 0.012 < 0.05 \checkmark$$

**(c) t-test (assuming $\sigma^2$ unknown):**

$$t = \frac{\bar{x} - 0}{s/\sqrt{n}} = \frac{5}{\sqrt{12}/\sqrt{4}} = \frac{5}{\sqrt{12}/2} = \frac{5}{\sqrt{3}} = \frac{5}{1.732} \approx 2.887$$

Null distribution: $t(3)$ (with $n-1 = 3$ degrees of freedom).

$$p = 2P\!\left(T \geq \frac{5}{\sqrt{3}}\right) = 2 \times \texttt{(1-pt(5/sqrt(3), 3))} \approx 0.063$$

Since $p = 0.063 > \alpha = 0.05$: **Do not reject $H_0$**.

**Key observation:** Using the z-test (assuming $\sigma^2 = 16$ known): reject. Using the t-test (unknown variance, estimated $s^2 = 12$): don't reject. The t-test correctly accounts for the additional uncertainty from estimating $\sigma$, leading to a more conservative conclusion.

---

### 16.3 Board Problem — Khan's Restaurant Chi-Square (Class 19)

**Problem:** Sal wants to test whether the owner's claimed customer distribution is correct.

| Day | M | T | W | Th | F | Sa |
|---|---|---|---|---|---|---|
| Owner's dist. | 0.10 | 0.10 | 0.15 | 0.20 | 0.30 | 0.15 |
| Observed | 30 | 14 | 34 | 45 | 57 | 20 |

Total observed customers: $N = 30+14+34+45+57+20 = 200$.

Expected counts: $E_i = N \times p_i$.

| Day | Observed | Expected | $O-E$ | $(O-E)^2/E$ |
|---|---|---|---|---|
| M | 30 | 20 | 10 | 5.00 |
| T | 14 | 20 | -6 | 1.80 |
| W | 34 | 30 | 4 | 0.53 |
| Th | 45 | 40 | 5 | 0.625 |
| F | 57 | 60 | -3 | 0.15 |
| Sa | 20 | 30 | -10 | 3.33 |

$$X^2 = 5.00 + 1.80 + 0.53 + 0.625 + 0.15 + 3.33 = 11.44$$

$$G = 2\left[30\ln\!\frac{30}{20} + 14\ln\!\frac{14}{20} + 34\ln\!\frac{34}{30} + 45\ln\!\frac{45}{40} + 57\ln\!\frac{57}{60} + 20\ln\!\frac{20}{30}\right] = 11.39$$

**Degrees of freedom:** 6 cells, 1 constraint (total count): $df = 5$.

**p-value:** $p = \texttt{1-pchisq(11.39, 5)} = 0.044$.

**Decision at $\alpha = 0.05$:** $p = 0.044 < 0.05$. **Reject $H_0$**.

**Conclusion:** The owner's claimed distribution for customers by day is not consistent with Sal's observed data. The biggest discrepancies are on Monday (+10 more than expected) and Saturday (-10 fewer than expected).

---

### 16.4 z-Test Problem (Class 19 Extra)

**Problem:** 16 samples from Normal$(\theta, 8^2)$. $\bar{x} = 4$. $H_0$: $\theta = 2$, $H_A$: $\theta \neq 2$. $\alpha = 0.04$.

$$z = \frac{\bar{x} - 2}{8/\sqrt{16}} = \frac{4 - 2}{8/4} = \frac{2}{2} = 1$$

Two-sided, $z > 0$:

$$p = 2P(Z > 1) = 2 \times 0.1587 = 0.317$$

Since $p = 0.317 > 0.04$: **Do not reject $H_0$**.

---

## 17. Concept Questions with Deep Explanations

### 17.1 Concept Question 1 — NHST (Class 18)

**Problem:** Left-sided z-test, $\alpha = 0.1$, $z = 1.8$.

**(i) Which R command computes the critical value?**

For a **left-sided** test at $\alpha = 0.1$, the rejection region is $z \leq z_{0.1}$ (the 0.1-quantile of the standard normal).

- **Answer: (g) `qnorm(0.1, 0, 1)`** = the 10th percentile $\approx -1.28$.

**(ii) Which command computes the p-value?**

The p-value for a left-sided test is $p = P(Z \leq z) = P(Z \leq 1.8)$.

- **Answer: (d) `pnorm(1.8, 0, 1)`** $\approx 0.964$.

**(iii) Reject?**

$p = 0.964 > \alpha = 0.10$. **No, do not reject $H_0$.**

**Draw the picture:**
```
φ(z|H₀) ~ N(0,1)

████            ___________
████           /           \
████__________/             \__________
z=-1.28       0          z=1.8
←rej.→       ← non-reject H₀ →

Rejection region: z ≤ -1.28 (left tail)
α = 10% (shaded, left tail)
z = 1.8 is on the RIGHT — far from rejection region
p = P(Z ≤ 1.8) = 0.964 >> 0.10 → do NOT reject
```

**Why is $z = 1.8$ not in the rejection region for a left-sided test?** A left-sided test rejects when the data is far to the LEFT (suggesting $\mu < \mu_0$). A z-value of 1.8 is far to the RIGHT, which is actually evidence FOR the null (or against the left-sided alternative). The p-value of 0.964 reflects that the data is very consistent with $H_0$ or even more extreme in the non-rejected direction.

---

### 17.2 Concept Question 2 — Power (Class 18)

**Question:** Power = area of which region in the diagram?

**Answer: (c) $R_1 + R_2$**

Power = $P(\text{rejection region} \mid H_A)$ = area under the $H_A$ distribution that lies in the rejection region.

- $R_1$ = area under $H_A$ curve that is clearly in the rejection region
- $R_2$ = area under $H_A$ curve in the overlapping region (still in rejection region)
- $R_1 + R_2$ = total area under $H_A$ in rejection region = **power**

---

### 17.3 Concept Question 3 — Higher Power (Class 18)

**Question:** Which of the two diagrams shows higher power?

**Answer: (2) Bottom graph.**

In the bottom graph, the $H_A$ distribution is far to the left of the null, and almost all of its probability mass lies in the rejection region. Power $\approx 1$.

In the top graph, there is more overlap between $H_A$ and the non-rejection region, so power is lower.

---

### 17.4 Concept Question 1 — t-Test Odds (Class 19)

**Question:** Two-sample t-test gives $p = 0.04$. What are the odds the two distributions have the same mean?

**Answer: (e) Unknown.**

A frequentist test computes $P(\text{data} \mid H_0)$, not $P(H_0 \mid \text{data})$. The latter requires a Bayesian prior. Without knowing the prior probability that the means are equal, we cannot compute the posterior odds. The p-value only tells us that, if $H_0$ were true, data this extreme would occur 4% of the time — not that $H_0$ has 4% probability of being true.

---

## 18. Gallery of Tests — Quick Reference

### 18.1 Choosing the Right Test

| Situation | Test | Null distribution |
|---|---|---|
| One group, test mean, $\sigma$ **known** | z-test | $Z \sim \text{N}(0,1)$ |
| One group, test mean, $\sigma$ **unknown** | One-sample t-test | $T \sim t(n-1)$ |
| Two groups, compare means, equal variance | Two-sample t-test | $T \sim t(n+m-2)$ |
| Two groups, compare means, unequal variance | Welch's t-test | $T \sim t(df)$ (Welch) |
| Paired data, compare means | Paired t-test | $T \sim t(n-1)$ |
| $\geq 3$ groups, compare means | ANOVA (F-test) | $F \sim F(n-1, n(m-1))$ |
| Categorical data fit to distribution | Chi-square goodness-of-fit | $\chi^2(df)$ |
| Two categorical variables, test independence | Chi-square independence | $\chi^2((r-1)(c-1))$ |
| Multiple groups, same distribution? | Chi-square homogeneity | $\chi^2((m-1)(n-1))$ |

### 18.2 Key Formulas at a Glance

**z-test:**

$$z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}, \quad Z \sim \text{N}(0,1)$$

**One-sample t-test:**

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \quad s^2 = \frac{\sum(x_i-\bar{x})^2}{n-1}, \quad T \sim t(n-1)$$

**Two-sample t-test (equal variances):**

$$t = \frac{\bar{x} - \bar{y} - \Delta\mu}{s_P}, \quad s_P^2 = \frac{(n-1)s_x^2+(m-1)s_y^2}{n+m-2}\left(\frac{1}{n}+\frac{1}{m}\right), \quad T \sim t(n+m-2)$$

**ANOVA F-test:**

$$f = \frac{\text{MS}_B}{\text{MS}_W}, \quad \text{MS}_B = \frac{m}{n-1}\sum(\bar{x}_i - \bar{x})^2, \quad \text{MS}_W = \frac{\sum s_i^2}{n}, \quad F \sim F(n-1, n(m-1))$$

**Chi-square statistics:**

$$X^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}, \quad G = 2\sum_i O_i \ln\!\frac{O_i}{E_i}$$

### 18.3 All Tests Assume Normality (Except Chi-Square)

| Test | Normality Required? | Notes |
|---|---|---|
| z-test | Yes | $\sigma$ known |
| One-sample t | Yes | Both $\mu, \sigma$ unknown |
| Two-sample t | Yes | Check equal variance assumption |
| Paired t | Yes (for differences) | Strong when pairs are correlated |
| ANOVA | Yes | Check equal variance across groups |
| Chi-square | **No** | Only assumes large expected counts |

### 18.4 R Commands

| Test | R Command |
|---|---|
| One-sample t | `t.test(x, mu=mu0, alternative="two.sided")` |
| Two-sample t (equal var) | `t.test(x, y, var.equal=TRUE)` |
| Two-sample t (Welch) | `t.test(x, y, var.equal=FALSE)` |
| Paired t | `t.test(x, y, paired=TRUE)` |
| ANOVA | `aov(y ~ group, data=df)` then `summary()` |
| Chi-square | `chisq.test(observed, p=expected_probs)` |
| F distribution CDF | `pf(f, df1, df2)` |
| t distribution CDF | `pt(t, df)` |
| Chi-square CDF | `pchisq(x, df)` |

---

## 19. Common Mistakes

### 19.1 Choosing the Wrong Test

| Mistake | Correction |
|---|---|
| Using z-test when $\sigma$ is unknown | Use t-test with sample variance $s^2$ |
| Using one-sample t for two groups | Use two-sample t-test |
| Using unpaired t for paired data | Use paired t-test (more powerful for paired designs) |
| Running 15 pairwise t-tests instead of ANOVA | Use ANOVA to control Type I error rate |

### 19.2 Degrees of Freedom

| Mistake | Correction |
|---|---|
| Using $n$ instead of $n-1$ for t-test df | One-sample t-test has $df = n-1$ |
| Using $n+m-1$ for two-sample t-test df | Two-sample t-test has $df = n+m-2$ |
| Not subtracting estimated parameters in chi-square df | $df = k - 1 - (\text{parameters estimated})$ |

### 19.3 Interpreting Results

| Mistake | Correction |
|---|---|
| "Not rejecting $H_0$ means $H_0$ is true" | Failing to reject only means data doesn't provide enough evidence against $H_0$ |
| "p-value is probability $H_0$ is true" | p-value is $P(\text{data this extreme} \mid H_0)$, NOT $P(H_0 \mid \text{data})$ |
| "Significant = important effect" | Statistical significance ≠ practical significance; with large $n$, tiny effects become significant |
| "Assuming equal variance without checking" | Always check variance equality before applying the equal-variance t-test |

### 19.4 Multiple Testing

| Mistake | Correction |
|---|---|
| Running many tests and treating each at $\alpha = 0.05$ independently | Apply Bonferroni correction or use ANOVA |
| Selecting which test to run based on seeing the data first | Decide on test statistic and significance level BEFORE collecting data |

---

## 20. Quick Reference Summary

### 20.1 Universal NHST Pattern

All significance tests follow: **specify $H_0$, $H_A$, test statistic, rejection region → compute test statistic from data → reject if p < α.**

### 20.2 Error Summary

$$\alpha = P(\text{Type I error}) = P(\text{reject } H_0 \mid H_0) = \text{significance level}$$

$$\text{Power} = P(\text{correct rejection}) = P(\text{reject } H_0 \mid H_A) = 1 - P(\text{Type II error})$$

### 20.3 z vs. t Summary

| | z-Test | t-Test |
|---|---|---|
| $\sigma$ | **Known** | **Unknown** (estimated by $s$) |
| Null distribution | $\text{N}(0,1)$ | $t(n-1)$ |
| Critical value at $\alpha=0.05$ (two-sided) | 1.96 | Larger (e.g., 2.57 for $df=5$) |
| Conservatism | Less conservative | More conservative (wider tails) |

### 20.4 Degrees of Freedom Rules

| Test | df |
|---|---|
| One-sample t | $n-1$ |
| Two-sample t (equal var) | $n+m-2$ |
| Paired t | $n-1$ (where $n$ = number of pairs) |
| ANOVA | Numerator: $n-1$; Denominator: $n(m-1)$ |
| Chi-square goodness-of-fit | $k - 1 - (\text{estimated params})$ |
| Chi-square independence/homogeneity | $(r-1)(c-1)$ |

### 20.5 Key Principle on Base Rates and Publication

The false positive rate in published science is NOT determined by $\alpha$ alone. It depends on the **base rate** of true effects in the field:

$$P(H_0 \mid \text{published}) = \frac{\alpha \cdot P(H_0)}{\alpha \cdot P(H_0) + \text{Power} \cdot (1-P(H_0))}$$

Fields with low true-effect base rates will have high rates of published false positives even with strict $\alpha$.

---

*End of MIT 18.05 Classes 18 & 19 Study Notes.*  
*Source: MIT OpenCourseWare, https://ocw.mit.edu — 18.05 Introduction to Probability and Statistics, Spring 2022.*
