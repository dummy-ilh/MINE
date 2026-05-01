# 🔬 07 & 08 — Z-test and T-test: Complete Guide

---

# PART A: Z-Test

## When to Use Z-Test

✅ Use the Z-test when:
- Population standard deviation $\sigma$ is **known**
- Sample size is **large** ($n \geq 30$), CLT justifies normality
- Testing proportions (with large $n$)

❌ Do NOT use Z-test when:
- $\sigma$ is unknown and $n$ is small
- Population is severely non-normal with small $n$

---

## One-Sample Z-Test

### Hypotheses

$$H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0$$

### Test Statistic

$$Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1) \text{ under } H_0$$

### Decision Rule

| Test | Reject H₀ if |
|---|---|
| Two-tailed | $\|Z\| > z_{\alpha/2}$ |
| Right-tailed | $Z > z_{\alpha}$ |
| Left-tailed | $Z < -z_{\alpha}$ |

### Critical Values (from N(0,1))

| α | Two-tailed ($z_{\alpha/2}$) | One-tailed ($z_{\alpha}$) |
|---|---|---|
| 0.10 | 1.645 | 1.282 |
| 0.05 | 1.960 | 1.645 |
| 0.01 | 2.576 | 2.326 |

### Step-by-Step Example

> A factory claims its machines produce bolts with mean diameter $\mu_0 = 10.0$ mm and known $\sigma = 0.5$ mm. You sample $n = 100$ bolts and find $\bar{x} = 10.12$ mm. Test at $\alpha = 0.05$.

**Step 1:** H₀: μ = 10.0 | H₁: μ ≠ 10.0 (two-tailed)

**Step 2:** $\alpha = 0.05$, critical value: $z_{0.025} = 1.96$

**Step 3:** Test statistic:
$$Z = \frac{10.12 - 10.0}{0.5/\sqrt{100}} = \frac{0.12}{0.05} = 2.4$$

**Step 4:** p-value:
$$p = 2 \times (1 - \Phi(2.4)) = 2 \times 0.0082 = 0.0164$$

**Step 5:** Since $p = 0.0164 < 0.05$, and $|Z| = 2.4 > 1.96$ → **Reject H₀**

**Conclusion:** There is statistically significant evidence at α = 0.05 that the mean bolt diameter differs from 10.0 mm.

---

## Two-Proportion Z-Test (A/B Testing)

### Hypotheses

$$H_0: p_1 = p_2 \quad \text{vs} \quad H_1: p_1 \neq p_2$$

### Test Statistic

$$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

Where $\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}$ is the **pooled proportion** (estimate under H₀ that $p_1 = p_2$).

### Example: A/B Test for CTR

> Control: $n_1 = 5000$, $x_1 = 250$ clicks → $\hat{p}_1 = 0.05$
> Variant: $n_2 = 5000$, $x_2 = 300$ clicks → $\hat{p}_2 = 0.06$

$$\hat{p} = \frac{250 + 300}{10000} = 0.055$$

$$Z = \frac{0.06 - 0.05}{\sqrt{0.055 \times 0.945 \times (1/5000 + 1/5000)}} = \frac{0.01}{\sqrt{0.055 \times 0.945 \times 0.0004}}$$

$$= \frac{0.01}{\sqrt{0.0000208}} = \frac{0.01}{0.00456} \approx 2.19$$

$$p\text{-value} = 2 \times (1 - \Phi(2.19)) = 2 \times 0.0142 = 0.0284$$

At $\alpha = 0.05$: **Reject H₀** — the CTR improvement is statistically significant.

---

---

# PART B: T-Test

## Why T Instead of Z?

When $\sigma$ is unknown (almost always in practice), we estimate it with $s$. This introduces **extra uncertainty**. The t-distribution accounts for this by having heavier tails than the normal.

$$T = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \sim t_{n-1} \text{ under } H_0$$

### T vs Z Distribution

![t-distribution vs normal](https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Student_t_pdf.svg/640px-Student_t_pdf.svg.png)

- t-distribution has heavier tails (more probability in extremes)
- As $n \to \infty$, $t_{n-1} \to \mathcal{N}(0,1)$
- At $n \geq 30$, the difference is minimal

---

## One-Sample t-Test

### Use When
- Testing a sample mean against a known/hypothesized value
- Population σ unknown
- Normal population (or n ≥ 30 by CLT)

### Test Statistic

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \quad df = n - 1$$

### Example

> A coffee shop claims average wait time is ≤ 5 minutes. You time 16 random customers: $\bar{x} = 5.8$ min, $s = 1.5$ min. Test at α = 0.05.

H₀: μ ≤ 5 | H₁: μ > 5 (one-tailed, right)

$$t = \frac{5.8 - 5}{1.5/\sqrt{16}} = \frac{0.8}{0.375} = 2.133$$

df = 15. Critical value: $t_{0.05, 15} = 1.753$

Since $2.133 > 1.753$ → **Reject H₀**. Evidence supports mean wait time exceeds 5 minutes.

---

## Independent (Two-Sample) t-Test

### Use When
- Comparing means of **two independent groups**
- E.g., treatment vs. control, users in region A vs. B

### Two Variants

#### Equal Variances (Pooled)

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}, \quad s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}$$

$$df = n_1 + n_2 - 2$$

#### Unequal Variances — Welch's t-Test (Preferred in Practice)

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

$$df \approx \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}} \quad \text{(Welch-Satterthwaite)}$$

> **Rule of thumb:** Use Welch's t-test by default unless you have strong theoretical reasons to assume equal variances. It's robust and more general.

### Example: Welch's t-test

> Treatment (n₁=25): $\bar{x}_1 = 82$, $s_1 = 12$
> Control (n₂=30): $\bar{x}_2 = 75$, $s_2 = 18$

$$t = \frac{82 - 75}{\sqrt{\frac{144}{25} + \frac{324}{30}}} = \frac{7}{\sqrt{5.76 + 10.8}} = \frac{7}{\sqrt{16.56}} = \frac{7}{4.07} \approx 1.72$$

Compute df via Welch-Satterthwaite ≈ 48. At $\alpha = 0.05$ (two-tailed), $t_{0.025, 48} \approx 2.01$.

Since $1.72 < 2.01$ → **Fail to Reject H₀**. No significant difference.

---

## Paired t-Test

### Use When

- Observations are paired (same subject measured twice: before/after, left/right, matched pairs)
- Within-subject design removes individual variation

### Key Insight

Convert paired data to **differences** $d_i = x_{i,2} - x_{i,1}$. Then perform a one-sample t-test on $d_i$.

$$t = \frac{\bar{d} - 0}{s_d / \sqrt{n}}, \quad df = n - 1$$

Where $\bar{d}$ = mean of differences, $s_d$ = SD of differences.

### Why Paired Tests Are More Powerful

The pairing removes between-subject variance. If individuals differ a lot (e.g., different baseline health levels), the paired test eliminates that noise, leaving only the treatment effect.

$$\text{Var}(\bar{d}) = \text{Var}(\bar{X}_1 - \bar{X}_2) = \frac{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}{n}$$

When $\rho > 0$ (positive correlation between pairs), the variance of differences is smaller → more powerful test.

### Example

> Blood pressure before and after treatment for 8 patients:

| Patient | Before | After | Difference |
|---|---|---|---|
| 1 | 145 | 138 | -7 |
| 2 | 130 | 125 | -5 |
| 3 | 150 | 140 | -10 |
| ... | ... | ... | ... |

$\bar{d} = -6.5$, $s_d = 3.2$, $n = 8$

$$t = \frac{-6.5 - 0}{3.2/\sqrt{8}} = \frac{-6.5}{1.131} = -5.75, \quad df = 7$$

$t_{0.025, 7} = 2.365$. Since $|{-5.75}| > 2.365$ → **Reject H₀**. Treatment significantly reduces blood pressure.

---

## 📊 Z-test vs T-test Decision Guide

```
                    Is σ known?
                   /           \
                YES              NO
                 |                |
            Z-test            n ≥ 30?
                            /       \
                          YES         NO
                           |           |
                    t-test (approx Z)  t-test (mandatory)
```

| Scenario | Test |
|---|---|
| Known σ, any n | Z-test |
| Unknown σ, n ≥ 30 | t-test (≈ Z in practice) |
| Unknown σ, n < 30, normal pop. | t-test |
| Unknown σ, n < 30, non-normal | Non-parametric |
| Two independent groups | Two-sample t (Welch's) |
| Before/after, matched pairs | Paired t-test |

---

## 💬 Interview Questions & Answers

**Q1: Why do we use the t-distribution instead of normal for small samples?**

> A: Because when we estimate σ with s, we introduce additional variability. The t-distribution has heavier tails to account for this uncertainty. With small n, s can be far from σ, so we need a distribution that allows for more extreme values. As n increases, s converges to σ and the t-distribution converges to the normal.

**Q2: When should you use a paired t-test vs an independent t-test?**

> A: Use paired when the same subjects or naturally matched units are measured under both conditions (before/after, left eye/right eye, twins in treatment/control). This removes between-subject variability and increases power. Use independent when the two groups are separate, unrelated individuals.

**Q3: Google runs an A/B test. How would you test whether the mean session duration differs?**

> A: Use a two-sample t-test (Welch's variant, since variances are likely unequal between groups). H₀: μ_control = μ_treatment. Compute the test statistic using sample means and standard deviations. Check if |t| exceeds the critical value at chosen α. Also check assumptions: independence of users (no network effects), equal exposure period, and check for outliers (session durations can be right-skewed — may need to winsorize or use a non-parametric test).

**Q4: Your p-value is exactly 0.05. Do you reject H₀?**

> A: Technically at α = 0.05, the rule is p < α (strict), so p = 0.05 would not reject. In practice, this is a borderline case — report the exact p-value, confidence intervals, and effect size, and let stakeholders make an informed decision. Don't treat the boundary as meaningful without context.

---

## 🚨 Common Pitfalls

1. **Using Z when σ is unknown** — always use t unless you truly know σ (rare in practice).
2. **Running two one-sample t-tests instead of a two-sample t-test** — increases Type I error.
3. **Using pooled t-test when variances differ greatly** — use Welch's.
4. **Applying independent t-test to paired data** — wastes the pairing, loses power.
5. **Not checking normality for small samples** — use Shapiro-Wilk test or Q-Q plots.

---

*← [05/06 — Significance & CI](05_06_significance_CI.md) | [09 — Chi-Square →](09_chi_square.md)*
