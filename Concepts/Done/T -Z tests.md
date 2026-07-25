# T-test variants and the Z-test for A/B testing

All numbers below were computed and verified (not hand-guessed) — you can trust them for practice. Each test now includes at least one **significant** and one **non-significant** worked example, since interviewers often probe whether you understand what "fail to reject" looks like in practice, not just how to reject H₀.

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

### Worked example 1 — reject H₀

A delivery company claims average delivery time is **30 minutes**. You sample 10 deliveries:
`32, 29, 35, 31, 28, 33, 34, 30, 36, 32`

- $\bar{x} = 32.0$, $s = 2.582$, $n = 10$
- $SE = s/\sqrt{n} = 2.582/\sqrt{10} = 0.816$
- $t = (32.0 - 30)/0.816 = 2.449$
- $df = 9$
- Two-tailed $p = 0.0368$

**Conclusion:** at α = 0.05, $p = 0.0368 < 0.05$ → reject H₀. Actual delivery times are significantly different (higher) than the claimed 30 minutes.

### Worked example 2 — fail to reject H₀

A factory claims average bolt weight is **40 g**. You sample 10 bolts:
`41, 38, 45, 39, 43, 40, 42, 37, 44, 41`

- $\bar{x} = 41.0$, $s = 2.582$, $n = 10$
- $SE = 2.582/\sqrt{10} = 0.8165$
- $t = (41.0 - 40)/0.8165 = 1.2247$
- $df = 9$
- Two-tailed $p = 0.2518$

**Conclusion:** $p = 0.2518 > 0.05$ → fail to reject H₀. The 1 g sample deviation is well within what you'd expect from sampling noise alone; there's no evidence the true mean differs from the claimed 40 g. Notice the sample size, spread, and even the raw deviation magnitude are almost identical in shape to Example 1 — the difference in conclusion comes entirely from the smaller *relative* gap (1.0 vs. 2.0 units) relative to the same noise level, which is exactly the ratio the t-statistic is built to capture.

### Worked example 3 — one-tailed test

A machine is supposed to fill bottles to **at least 6.0 L**. You only care if it's *underfilling* (one-directional concern), so a one-tailed test is appropriate. Sample of 12 bottles:
`5.8, 6.1, 5.9, 6.3, 6.0, 6.2, 5.7, 6.4, 6.1, 5.9, 6.2, 6.0`

- $\bar{x} = 6.05$, $s = 0.2067$, $n = 12$
- $SE = 0.2067/\sqrt{12} = 0.0597$
- $t = (6.05 - 6.0)/0.0597 = 0.8379$
- $df = 11$
- One-tailed $p = 0.2100$ (compare: two-tailed would be $0.4199$)

**Conclusion:** even one-tailed, $p = 0.21 > 0.05$ → fail to reject. No evidence of systematic underfilling.

**Interview flag on one-tailed tests:** a one-tailed test roughly halves your p-value compared to two-tailed for the same data — which is exactly why using one when you should've used two is a classic way p-hacking sneaks in. The direction of concern must be decided *before* looking at the data, never chosen after seeing which way the sample happened to lean.

---

## 2. Independent two-sample t-test — Student's (equal variances assumed)

**Question it answers:** do two independent groups have different means, assuming both populations have roughly the same variance?

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}, \quad s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}, \quad df = n_1+n_2-2$$

$s_p^2$ is the **pooled variance** — a weighted average of both groups' variances, used because we're assuming they share one true variance.

### Worked example 1 — reject H₀

Comparing exam scores, teaching method A vs B (10 students each):
- Group A: `85,88,82,90,86,84,89,87,91,83` → mean 86.5, sd 3.028
- Group B: `78,81,76,84,79,80,77,82,85,79` → mean 80.1, sd 2.923

- Pooled variance $s_p^2 = 8.856$
- $SE_{pooled} = 1.331$
- $t = (86.5 - 80.1)/1.331 = 4.809$
- $df = 18$
- $p = 0.00014$

**Conclusion:** extremely small p — reject H₀. Method A produces significantly higher scores.

### Worked example 2 — fail to reject H₀ (near-identical means)

Comparing reaction times (ms), two caffeine dosages, 10 subjects each:
- Group A: `72,75,71,78,74,73,76,77,70,75` → mean 74.1, sd 2.601
- Group B: `74,73,76,72,75,71,77,74,73,76` → mean 74.1, sd 1.912

- Pooled variance $s_p^2 = 5.2111$
- $SE_{pooled} = 1.0209$
- $t = (74.1 - 74.1)/1.0209 = 0.0000$
- $df = 18$
- $p = 1.0000$

**Conclusion:** the means are identical to the decimal shown — $t=0$, $p=1$, about as clean a "fail to reject" as you'll ever see. Worth being able to recognize instantly: when $\bar x_1 = \bar x_2$ exactly, the numerator of $t$ is zero regardless of the pooled SE or sample size, so $t=0$ and $p=1$ always, no further computation needed — a good mental shortcut/sanity check to state out loud in an interview.

**Rule of thumb for "equal variance" assumption:** if the larger sample variance is no more than ~4x the smaller one, Student's t is usually safe. Otherwise, use Welch's below.

---

## 3. Independent two-sample t-test — Welch's (unequal variances)

**Question it answers:** same as above, but doesn't assume equal population variances — safer default in practice, especially with unequal sample sizes or visibly different spreads.

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$

**Degrees of freedom (Welch-Satterthwaite equation)** — this is the part people forget:

$$df = \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1}+\frac{(s_2^2/n_2)^2}{n_2-1}}$$

df is not simply $n_1+n_2-2$ here — it's often a non-integer, calculated to reflect how unequal the variances/sample sizes are.

### Worked example 1 — fail to reject H₀ (unequal spread, no real difference)

Two groups with very different spread and size:
- Group C (n=7): `50,52,49,51,50,53,48` → mean 50.43, sd 1.718 (tight cluster)
- Group D (n=12): `40,65,35,70,30,75,20,80,45,60,25,55` → mean 50.0, sd 20.338 (wide spread)

- $SE = \sqrt{1.718^2/7 + 20.338^2/12} = $ computed → $t = 0.0726$
- Welch df = **11.27** (non-integer — this is expected and correct)
- $p = 0.943$

**Conclusion:** p is huge — fail to reject H₀. Despite wildly different variances, the means are statistically indistinguishable. This example is deliberately chosen to show Welch's test handling unequal variance/sample-size gracefully — a naive Student's t here (pooling such different variances) would give a misleading result.

### Worked example 2 — reject H₀ (unequal spread, real difference this time)

Same shape of problem — tight-vs-wide groups, unequal n — but now the true means genuinely differ, to show Welch's correctly detecting a real effect and not just being "conservative":

- Group E (n=10): `55,58,54,57,56,59,55,58,56,57` → mean 56.5, sd 1.581 (tight cluster)
- Group F (n=14): `20,50,15,55,25,45,10,60,30,40,22,48,18,52` → mean 35.0, sd 16.793 (wide spread)

- $SE = \sqrt{1.581^2/10 + 16.793^2/14} = 4.5158$
- $t = (56.5-35.0)/4.5158 = 4.761$
- Welch df = **13.32**
- $p = 0.00035$

**Conclusion:** reject H₀ — Group E is significantly higher than Group F.

**Side-by-side with Student's on the *same* data** (to make the "Welch is the safer default" point concrete, not just asserted): running the naive equal-variance formula on this same E/F data gives pooled $s_p^2 = 167.66$, $SE_{pooled} = 5.361$, $t = 4.010$, $df = 22$, $p = 0.00059$. Both tests reject here — the true effect is large enough that it survives either way — but notice **Student's $t$ is smaller and its $p$ is larger** than Welch's, because pooling variance across a tight group and a wide group overstates the tight group's true uncertainty. In borderline cases (smaller effect sizes), this gap between the two methods is exactly what can flip a result from significant to non-significant depending on which formula you reach for — which is the whole argument for defaulting to Welch's.

**Interview flag:** in practice, **Welch's t-test is the safer default** — many statisticians recommend always using Welch's unless you have strong reason to believe variances are equal, since it degrades gracefully to Student's t when variances *are* equal, but Student's t can be badly wrong when they aren't.

---

## 4. Paired t-test (dependent samples)

**Question it answers:** is there a significant difference between two *matched/related* measurements — same subjects measured twice (before/after), or naturally paired items?

The trick: reduce it to a **one-sample t-test on the differences**.

$$t = \frac{\bar{d} - 0}{s_d/\sqrt{n}}, \quad df = n-1$$

where $d_i = x_{1i} - x_{2i}$ for each pair, $\bar{d}$ is the mean difference, $s_d$ is the standard deviation of the differences.

### Worked example 1 — reject H₀

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

### Worked example 2 — fail to reject H₀ (differences cancel out)

Resting heart rate (bpm) before and after a *placebo* intervention, same 10 people:

| Before | 150 | 162 | 148 | 171 | 155 | 168 | 159 | 163 | 152 | 166 |
|---|---|---|---|---|---|---|---|---|---|---|
| After | 151 | 160 | 150 | 169 | 156 | 167 | 158 | 164 | 150 | 168 |
| Diff | 1 | -2 | 2 | -2 | 1 | -1 | -1 | 1 | -2 | 2 |

- $\bar{d} = -0.10$, $s_d = 1.663$, $n = 10$
- $SE = 1.663/\sqrt{10} = 0.526$
- $t = -0.10/0.526 = -0.190$
- $df = 9$
- $p = 0.8534$

**Conclusion:** fail to reject H₀ — the differences are small and scattered in both directions (some subjects went up, some down), so they nearly cancel in the mean, and what's left is indistinguishable from noise. This is exactly the pattern you'd want to see for a working placebo/control arm — a good "negative control" example to have in your pocket, since interviewers sometimes ask you to sanity-check that your test *correctly fails to find an effect where none should exist*, not just correctly finds one where it should.

**Why paired beats independent-samples here:** by differencing within the same person, you cancel out each person's individual baseline variability (some people just run heavier/lighter) — this dramatically shrinks the noise (compare $s_d = 1.414$ or $1.663$ to the raw before/after spreads, which are an order of magnitude larger) and gives you far more power than treating before/after as two independent groups.

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

### Worked example 1 — significant, but marginal (the "watch the CI, not just the p-value" case)

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

### Worked example 2 — not significant, small sample per arm

Smaller A/B test, 500 users per arm (e.g., an early readout before the test has fully ramped):
- Control: 48 conversions → $\hat p_1 = 0.0960$ (9.60%)
- Treatment: 55 conversions → $\hat p_2 = 0.1100$ (11.00%)

- Pooled rate $\hat p = (48+55)/1000 = 0.1030$
- $SE_{pooled} = \sqrt{0.1030 \times 0.8970 \times (1/500+1/500)} = 0.01922$
- $z = (0.1100 - 0.0960)/0.01922 = 0.728$
- Two-tailed $p = 0.4665$

**Conclusion:** fail to reject H₀. Note the observed *relative* lift here (14.6%) is actually larger than Example 1's (11.3%), but it's not significant — because the sample size is 20x smaller, the standard error is far larger relative to the effect. 95% CI on the difference (unpooled SE, same value here since $n_1=n_2$): $[-2.37\%, 5.17\%]$ — the interval comfortably spans zero, unlike Example 1. **This pair of examples is the cleanest way to demonstrate the core A/B testing lesson: statistical significance is driven by effect size relative to noise, and noise is driven by sample size — a bigger raw or relative lift is not automatically more "real" than a smaller one if it wasn't measured precisely enough.**

### Worked example 3 — huge sample, tiny effect, still significant

At the other extreme — very large A/B test (200,000 users per arm), where even a fraction-of-a-percentage-point difference becomes detectable:
- Control: 12,000 conversions → $\hat p_1 = 0.06000$ (6.000%)
- Treatment: 12,300 conversions → $\hat p_2 = 0.06150$ (6.150%)

- Pooled rate $\hat p = 0.06075$
- $SE_{pooled} = 0.000755$
- $z = (0.06150-0.06000)/0.000755 = 1.986$
- Two-tailed $p = 0.0471$

**Conclusion:** significant, but the absolute lift is only 0.15 percentage points (95% CI on the difference: $[0.002\%, 0.298\%]$ — a razor-thin interval). This is the classic "statistically significant but is it *practically* significant" trap large-scale A/B tests run into: at internet-company sample sizes, almost any nonzero true effect eventually clears $p<0.05$, so the interview-worthy skill is pairing the p-value with the CI/effect size and asking whether 0.15pp of lift is worth the engineering cost and risk of shipping — not treating "p < 0.05" as the end of the analysis.

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
A: As a default habit — Welch's converges to the same answer as Student's when variances truly are equal, but protects you when they're not. There's little downside to always using Welch's unless you have a specific reason (e.g., a course requires the pooled-variance formula). The side-by-side E/F example above (§3) shows the two methods can meaningfully disagree on both the test statistic and p-value even when they reach the same reject/fail-to-reject conclusion — in borderline cases that gap can flip the decision entirely.

**Q4: Why is the paired t-test almost always more powerful than an independent two-sample t-test on the same data, if you could run it either way?**
A: Pairing removes between-subject variability from the noise term — you're only measuring the *within-subject* change, which is usually far less variable than the raw values across different people. Less noise in the denominator means a larger t-statistic for the same true effect, hence higher power.

**Q5: Your A/B test shows p = 0.045, barely under 0.05, with a CI of [0.01%, 1.4%]. What do you tell the stakeholder?**
A: The result is statistically significant, but the interval is wide relative to the effect and barely excludes zero — the true lift could be as small as 0.01 percentage points. I'd recommend treating this as a promising but not conclusive signal: consider running longer, checking for practical significance (is even the low end of that interval worth shipping?), and being cautious about peeking/multiple testing that could have inflated the apparent significance.

**Q6 (curveball): You run a Welch's t-test and get a non-integer degrees of freedom, like 11.27. Is that an error?**
A: No — this is expected and correct. Student's t assumes equal variances and gives a clean integer df ($n_1+n_2-2$); Welch's t explicitly does not assume equal variances, and the Welch-Satterthwaite equation that computes its df is a weighted approximation that is only occasionally an integer. A non-integer df is a signature that Welch's correction is doing its job.

**Q7: In the paired t-test example, what would go wrong if you'd mistakenly run an independent two-sample t-test on the before/after data instead?**
A: You'd throw away the pairing information and treat "before" and "after" as two separate groups of 8 (or 10) unrelated people each. This ignores that the same individual's before and after values are correlated (heavier people tend to stay relatively heavier even after losing weight), which inflates the apparent variance and lowers your power to detect the true within-subject effect — you could easily get a non-significant result even when a real effect exists.

**Q8: Two A/B tests both hit p < 0.05 — one on 1,000 users/arm with a 5-point relative lift, one on 200,000 users/arm with a 2.5% relative lift. Which result would you trust more, and which would you ship?**
A: "Trust" and "ship" aren't the same question. Statistically, the larger-sample result is measured far more precisely (tighter CI relative to the effect, as in §5 Example 3), so there's less concern about the estimate itself being noisy. But "ship" depends on the *absolute* size of the effect and its business cost/benefit — a huge, precisely-measured 0.15-percentage-point lift (§5 Example 3) might not be worth shipping if it doesn't clear an engineering or risk threshold, while a smaller-sample but larger relative lift might still be worth a longer follow-up test to firm up the estimate before deciding. The p-value alone answers neither question.

**Q9: In the one-tailed bottle-filling example (§1), why does switching from two-tailed to one-tailed roughly halve the p-value, and why is that risky?**
A: A two-tailed test splits the significance threshold across both directions of surprise (too high *or* too low); a one-tailed test puts the entire α budget in one direction, so the same observed deviation looks "more extreme" relative to a one-sided rejection region — mechanically, for a symmetric distribution like $t$, the one-tailed p-value is exactly half the two-tailed one when the effect is in the hypothesized direction. It's risky because if you decide *after* seeing the data that you only care about one direction (rather than committing to that beforehand), you're effectively giving yourself a free halving of your p-value on demand — a subtle but common form of p-hacking.