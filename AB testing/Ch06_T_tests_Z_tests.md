# Chapter 6: T-tests & Z-tests

## 1. Intuition

By now you know the hypothesis testing logic (Ch 2), what a p-value means (Ch 3), and how to size an experiment (Ch 5). This chapter answers a narrower but frequently-tested question: **which specific test statistic do you actually compute, and why does it matter which one you pick?**

The core intuition: the choice between a t-test and a z-test isn't about the metric type (conversion rate vs. revenue) — it's about **whether you know the true population variance or have to estimate it from your sample, and how large your sample is.** In practice, at Google-scale traffic, this distinction mostly stops mattering numerically (t and z converge for large $n$), but interviewers still test whether you understand *why* it stops mattering, not just that it does.

## 2. The Formal Distinction

**Z-test**: assumes the population standard deviation $\sigma$ is known, or the sample is large enough that the sample standard deviation $s$ is treated as if it equals $\sigma$ (via the Law of Large Numbers). Test statistic:

$$z = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}$$

**T-test**: used when $\sigma$ is unknown and estimated from the sample as $s$, introducing extra uncertainty that the normal distribution doesn't account for. This extra uncertainty is captured by using the **t-distribution** instead of the normal, which has fatter tails for small $n$ and converges to the normal distribution as $n \to \infty$ (specifically, as degrees of freedom → ∞).

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$

Same formula structurally — the difference is entirely in **which distribution you compare the statistic to** (normal vs. t with $df$ degrees of freedom), because the t-distribution correctly accounts for the added uncertainty of estimating $\sigma$ from a finite sample.

**Why proportions typically use z, not t**: for a binomial proportion, the variance $p(1-p)$ is *fully determined by the mean* $p$ itself — there's no separate variance parameter to estimate. So once you estimate $\hat{p}$, you've automatically got your variance estimate too, and by CLT for reasonably large $n$, the sampling distribution of $\hat{p}$ is well-approximated by a normal. This is why Chapters 2-4's conversion rate examples all used z, not t.

**Why continuous metrics (revenue, session length, latency) typically use t**: these have a true variance $\sigma^2$ that's a separate, independent parameter from the mean, and must be estimated separately as $s^2$ from the sample — hence the extra "estimation uncertainty" the t-distribution accounts for.

## 3. Paired vs. Unpaired (Independent) Tests

- **Unpaired/independent t-test**: used when treatment and control are different, unrelated groups of users — the standard A/B test setup.
- **Paired t-test**: used when you have before/after measurements on the *same* units (e.g., comparing each user's spending in the week before vs. the week after a feature launch, for the same set of users). Paired tests are more powerful because they control for individual-level baseline differences — this is conceptually the ancestor of CUPED (Chapter 9), which generalizes the "control for individual baseline" idea to the regression-adjustment setting.

$$t_{paired} = \frac{\bar{d}}{s_d/\sqrt{n}}$$

where $\bar{d}$ is the mean of the individual differences and $s_d$ is the standard deviation of those differences — because you're differencing out each user's own baseline, the variance is often dramatically smaller than in the unpaired case, which is exactly why paired designs are more powerful when available.

## 4. Welch's Correction

The standard two-sample t-test (Student's t-test) assumes **equal variances** between the two groups. In A/B testing, this assumption is frequently violated — e.g., if the treatment changes not just the mean but also the *spread* of the metric (common with revenue metrics where a treatment might create more high-spenders).

**Welch's t-test** relaxes the equal-variance assumption:

$$t_{Welch} = \frac{\bar{X}_1-\bar{X}_2}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}$$

(same numerator/structure as the general formula above), but uses the **Welch-Satterthwaite equation** to compute adjusted degrees of freedom that account for unequal variances and unequal sample sizes:

$$df \approx \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1}+\frac{(s_2^2/n_2)^2}{n_2-1}}$$

**Practical rule**: default to Welch's t-test rather than Student's t-test in production A/B testing. It's strictly more robust — when variances happen to be equal, Welch's converges to the same answer as Student's; when they're not, Welch's protects you from an invalid test. Nearly all modern statistical software (including scipy's `ttest_ind` with `equal_var=False`) makes Welch's the trivial default choice, so there's essentially no cost to defaulting to it.

## 5. Worked Example

You're comparing average session duration (a continuous metric, not a proportion) between treatment and control.

- Control: $n_0=500$, $\bar{X}_0 = 12.4$ min, $s_0 = 5.1$ min
- Treatment: $n_1=500$, $\bar{X}_1 = 13.1$ min, $s_1 = 7.8$ min (notice: variances look quite different — 5.1 vs 7.8 — a Welch's correction candidate)

**Step 1 — compute the difference and pooled-free SE:**

$$SE = \sqrt{\frac{5.1^2}{500}+\frac{7.8^2}{500}} = \sqrt{\frac{26.01}{500}+\frac{60.84}{500}} = \sqrt{0.0520+0.1217} = \sqrt{0.1737} \approx 0.4168$$

**Step 2 — t-statistic:**

$$t = \frac{13.1-12.4}{0.4168} = \frac{0.7}{0.4168} \approx 1.68$$

**Step 3 — degrees of freedom (Welch-Satterthwaite)**, approximately (skipping intermediate arithmetic): given the substantial variance difference, $df \approx 850$ (less than the naive $n_1+n_2-2=998$ you'd get from Student's t-test, reflecting the "penalty" for unequal variances).

**Step 4 — critical value**: for $df\approx850$, the t-distribution is already extremely close to normal, so $t_{critical}\approx1.96$ for $\alpha=0.05$ two-sided.

Since $1.68 < 1.96$, we fail to reject $H_0$ — not statistically significant, despite treatment showing a numerically higher mean.

**Key teaching point**: notice that at $df=850$, the t-critical value (1.96ish) is essentially identical to the z-critical value (1.96) — this is the "t converges to z for large n" fact made concrete. If this were $n=15$ per arm instead of 500, the t-critical value would be closer to 2.05-2.15 (fatter tails), and the distinction would actually matter.

## 6. Production Considerations

- **Default to Welch's t-test for continuous metrics** unless you have strong reason to believe variances are equal — there's no real downside.
- **For proportions, use a z-test** (or equivalently a chi-square test for the 2x2 contingency table version, which gives an identical p-value for a 2-group comparison) — not a t-test, since variance is determined by the mean.
- **For heavily skewed metrics** (revenue per user, with many zeros and a long right tail from whales), neither a naive t-test nor z-test may be appropriate — consider a Mann-Whitney U test (non-parametric, tests for stochastic dominance rather than mean difference) or bootstrap confidence intervals, which don't rely on normality assumptions. Worth mentioning if asked about revenue metrics specifically.
- **At Google-scale sample sizes** (often hundreds of thousands to millions of users per arm), t vs. z is essentially a non-issue numerically — but understanding *why* (CLT + large df convergence) is what interviewers are actually checking for.

## 7. Interview Traps

- **Trap #1**: Saying "we use a t-test for small samples and a z-test for large samples" as if it's purely about sample size. It's actually about whether $\sigma$ is known/derivable from the mean (proportions) vs. estimated separately (continuous metrics) — sample size affects convergence, but isn't the defining distinction.
- **Trap #2**: Using Student's t-test (equal-variance assumption) by default instead of Welch's, especially when the interviewer mentions or implies unequal variances between arms.
- **Trap #3**: Using an independent/unpaired test when the data is actually paired (e.g., before/after on the same users), losing statistical power unnecessarily.
- **Trap #4**: Not recognizing when a metric is so skewed that a mean-based test (t-test) is inappropriate altogether, and non-parametric alternatives should be considered.

## 8. L5-Differentiating Talking Points



- Explaining WHY proportions use z (variance determined by the mean) rather than reciting "small n → t, big n → z" shows real understanding of the underlying mechanics.
- Defaulting to Welch's t-test and explaining why it's the safer choice, unprompted, signals production experience over textbook knowledge.
- Bringing up non-parametric alternatives (Mann-Whitney, bootstrap) for skewed revenue metrics shows you know the limits of the standard toolkit and have a next move ready.
- Connecting paired t-tests conceptually to CUPED as "controlling for individual baseline" bridges this chapter forward and shows you see the throughline in variance reduction across the curriculum.

## 9. Comprehension Check

1. What determines whether you should use a t-test or z-test — is it fundamentally about sample size, or something else? Explain.
2. Why do proportions typically use a z-test while continuous metrics like revenue typically use a t-test?
3. What problem does Welch's correction solve, and why might you default to it even when you're not sure if variances are equal?
4. Why is a paired t-test more powerful than an unpaired t-test when both are valid options for the same data?
5. Your revenue-per-user metric has a huge right tail from a handful of whale customers. Why might a standard t-test give a misleading result here, and what's an alternative?

---
*Next: Chapter 7 — Randomization Units & Interference*
