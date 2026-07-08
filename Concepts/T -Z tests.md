# T-test variants and the Z-test for A/B testing

All numbers below were computed and verified (not hand-guessed) — you can trust them for practice.

---

## 1. One-sample t-test

**Question it answers:** does this sample's mean differ from a known/claimed value?

$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \quad df = n - 1$$

| Symbol | Meaning |
|---|---|
| $\bar{x}$ | sample mean |
| $\mu_0$ | the claimed/hypothesized population mean |
| $s$ | sample standard deviation |
| $n$ | sample size |

### Worked example

A delivery company claims average delivery time is **30 minutes**. You sample 10 deliveries:
`32, 29, 35, 31, 28, 33, 34, 30, 36, 32`

- $\bar{x} = 32.0$, $s = 2.582$, $n = 10$
- $SE = s/\sqrt{n} = 2.582/\sqrt{10} = 0.816$
- $t = (32.0 - 30)/0.816 = 2.449$
- $df = 9$
- Two-tailed $p = 0.0368$

**Conclusion:** at α = 0.05, $p = 0.0368 < 0.05$ → reject H₀. Actual delivery times are significantly different (higher) than the claimed 30 minutes.

---

## 2. Independent two-sample t-test — Student's (equal variances assumed)

**Question it answers:** do two independent groups have different means, assuming both populations have roughly the same variance?

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}, \quad s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}, \quad df = n_1+n_2-2$$

$s_p^2$ is the **pooled variance** — a weighted average of both groups' variances, used because we're assuming they share one true variance.

### Worked example

Comparing exam scores, teaching method A vs B (10 students each):
- Group A: `85,88,82,90,86,84,89,87,91,83` → mean 86.5, sd 3.028
- Group B: `78,81,76,84,79,80,77,82,85,79` → mean 80.1, sd 2.923

- Pooled variance $s_p^2 = 8.856$
- $SE_{pooled} = 1.331$
- $t = (86.5 - 80.1)/1.331 = 4.809$
- $df = 18$
- $p = 0.00014$

**Conclusion:** extremely small p — reject H₀. Method A produces significantly higher scores.

**Rule of thumb for "equal variance" assumption:** if the larger sample variance is no more than ~4x the smaller one, Student's t is usually safe. Otherwise, use Welch's below.

---

## 3. Independent two-sample t-test — Welch's (unequal variances)

**Question it answers:** same as above, but doesn't assume equal population variances — safer default in practice, especially with unequal sample sizes or visibly different spreads.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$

**Degrees of freedom (Welch-Satterthwaite equation)** — this is the part people forget:

$$df = \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1}+\frac{(s_2^2/n_2)^2}{n_2-1}}$$

df is not simply $n_1+n_2-2$ here — it's often a non-integer, calculated to reflect how unequal the variances/sample sizes are.

### Worked example

Two groups with very different spread and size:
- Group C (n=7): `50,52,49,51,50,53,48` → mean 50.43, sd 1.718 (tight cluster)
- Group D (n=12): `40,65,35,70,30,75,20,80,45,60,25,55` → mean 50.0, sd 20.338 (wide spread)

- $SE = \sqrt{1.718^2/7 + 20.338^2/12} = $ computed → $t = 0.0726$
- Welch df = **11.27** (non-integer — this is expected and correct)
- $p = 0.943$

**Conclusion:** p is huge — fail to reject H₀. Despite wildly different variances, the means are statistically indistinguishable. This example is deliberately chosen to show Welch's test handling unequal variance/sample-size gracefully — a naive Student's t here (pooling such different variances) would give a misleading result.

**Interview flag:** in practice, **Welch's t-test is the safer default** — many statisticians recommend always using Welch's unless you have strong reason to believe variances are equal, since it degrades gracefully to Student's t when variances *are* equal, but Student's t can be badly wrong when they aren't.

---

## 4. Paired t-test (dependent samples)

**Question it answers:** is there a significant difference between two *matched/related* measurements — same subjects measured twice (before/after), or naturally paired items?

The trick: reduce it to a **one-sample t-test on the differences**.

$$t = \frac{\bar{d} - 0}{s_d/\sqrt{n}}, \quad df = n-1$$

where $d_i = x_{1i} - x_{2i}$ for each pair, $\bar{d}$ is the mean difference, $s_d$ is the standard deviation of the differences.

### Worked example

Weight (kg) before and after an 8-week program, same 8 people:

| Before | 80 | 92 | 75 | 88 | 70 | 95 | 83 | 78 |
|---|---|---|---|---|---|---|---|---|
| After | 75 | 90 | 70 | 86 | 68 | 90 | 80 | 74 |
| Diff | 5 | 2 | 5 | 2 | 2 | 5 | 3 | 4 |

- $\bar{d} = 3.5$, $s_d = 1.414$, $n = 8$
- $SE = 1.414/\sqrt{8} = 0.5$
- $t = 3.5/0.5 = 7.0$
- $df = 7$
- $p = 0.00021$

**Conclusion:** reject H₀ — the program produced a statistically significant average weight loss of 3.5 kg.

**Why paired beats independent-samples here:** by differencing within the same person, you cancel out each person's individual baseline variability (some people just run heavier/lighter) — this dramatically shrinks the noise (compare $s_d = 1.414$ to the raw before/after spreads) and gives you far more power than treating before/after as two independent groups.

---

## 5. Z-test for two proportions — the standard A/B testing test

**Question it answers:** did the treatment group's conversion rate differ from control's? This is the workhorse test for A/B testing binary outcomes (click/no-click, convert/no-convert) because with large samples, the sampling distribution of a proportion is approximately normal (CLT) and — critically — the variance under H₀ is *known* as a function of the pooled proportion, so you don't need to estimate an unknown σ from scratch the way a t-test does. That's why it's a Z-test, not a t-test.

$$z = \frac{\hat{p}_2 - \hat{p}_1}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}, \quad \hat{p} = \frac{x_1+x_2}{n_1+n_2}$$

| Symbol | Meaning |
|---|---|
| $\hat{p}_1, \hat{p}_2$ | observed conversion rates, control and treatment |
| $\hat{p}$ | pooled conversion rate under H₀ (assumes no true difference) |
| $x_1, x_2$ | conversion counts |
| $n_1, n_2$ | sample sizes |

### Worked example

A/B test, 10,000 users per arm:
- Control: 620 conversions → $\hat{p}_1 = 0.0620$ (6.20%)
- Treatment: 690 conversions → $\hat{p}_2 = 0.0690$ (6.90%)

- Pooled rate $\hat{p} = (620+690)/20000 = 0.0655$
- $SE_{pooled} = \sqrt{0.0655 \times 0.9345 \times (1/10000+1/10000)} = 0.00350$
- $z = (0.0690 - 0.0620)/0.00350 = 2.001$
- Two-tailed $p = 0.0454$

**Conclusion:** at α = 0.05, $p = 0.0454 < 0.05$ → statistically significant lift. Treatment converts better than control.

**Confidence interval for the actual lift** (uses *unpooled* SE — a subtlety worth knowing: pooled SE is used for the hypothesis test under H₀, unpooled SE is used for the CI since it doesn't assume p₁ = p₂):

$$SE_{unpooled} = \sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1} + \frac{\hat p_2(1-\hat p_2)}{n_2}}$$

Difference = 0.70 percentage points, 95% CI = **[0.014%, 1.386%]**. Notice the CI barely clears zero — this result is significant but not by a wide margin, worth flagging to stakeholders as "real but modest, keep monitoring."

---

## Cheat sheet: which test, when

| Scenario | Test |
|---|---|
| One sample vs a known/claimed value | One-sample t-test |
| Two independent groups, similar variances | Student's independent t-test |
| Two independent groups, unequal variances (default choice in practice) | Welch's independent t-test |
| Same subjects measured twice (before/after) | Paired t-test |
| Two independent proportions (A/B test conversion rates) | Two-proportion Z-test |
| Comparing more than 2 group means | ANOVA (not covered here) |

---

## Interview Q&A

**Q1: Why is a two-proportion comparison a Z-test but a two-group mean comparison is a t-test?**
A: For proportions, the variance under H₀ is a deterministic function of the proportion itself ($p(1-p)$) — once you have $\hat p$, you know the variance exactly, no separate estimation needed. For means, the population variance is genuinely unknown and must be estimated from the sample, which injects extra uncertainty — that's exactly the uncertainty the t-distribution's fatter tails account for.

**Q2: Why does the two-proportion Z-test use pooled SE for the test but unpooled SE for the confidence interval?**
A: The hypothesis test assumes H₀ is true (p₁ = p₂), so it's valid — and more powerful — to pool both samples into a single estimate of that shared proportion. The confidence interval, by contrast, is estimating the *actual* difference without presupposing it's zero, so each group's own observed variance is used instead.

**Q3: When would you deliberately use Welch's t-test over Student's, even if you suspect variances are similar?**
A: As a default habit — Welch's converges to the same answer as Student's when variances truly are equal, but protects you when they're not. There's little downside to always using Welch's unless you have a specific reason (e.g., a course requires the pooled-variance formula).

**Q4: Why is the paired t-test almost always more powerful than an independent two-sample t-test on the same data, if you could run it either way?**
A: Pairing removes between-subject variability from the noise term — you're only measuring the *within-subject* change, which is usually far less variable than the raw values across different people. Less noise in the denominator means a larger t-statistic for the same true effect, hence higher power.

**Q5: Your A/B test shows p = 0.045, barely under 0.05, with a CI of [0.01%, 1.4%]. What do you tell the stakeholder?**
A: The result is statistically significant, but the interval is wide relative to the effect and barely excludes zero — the true lift could be as small as 0.01 percentage points. I'd recommend treating this as a promising but not conclusive signal: consider running longer, checking for practical significance (is even the low end of that interval worth shipping?), and being cautious about peeking/multiple testing that could have inflated the apparent significance.

**Q6 (curveball): You run a Welch's t-test and get a non-integer degrees of freedom, like 11.27. Is that an error?**
A: No — this is expected and correct. Student's t assumes equal variances and gives a clean integer df ($n_1+n_2-2$); Welch's t explicitly does not assume equal variances, and the Welch-Satterthwaite equation that computes its df is a weighted approximation that is only occasionally an integer. A non-integer df is a signature that Welch's correction is doing its job.

**Q7: In the paired t-test example, what would go wrong if you'd mistakenly run an independent two-sample t-test on the before/after data instead?**
A: You'd throw away the pairing information and treat "before" and "after" as two separate groups of 8 unrelated people each. This ignores that the same individual's before and after values are correlated (heavier people tend to stay relatively heavier even after losing weight), which inflates the apparent variance and lowers your power to detect the true within-subject effect — you could easily get a non-significant result even when a real effect exists.
