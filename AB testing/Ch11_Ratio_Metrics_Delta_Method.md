# Chapter 11: Ratio Metrics & the Delta Method

## 1. Intuition

So far every metric we've analyzed has been a simple average per randomization unit (mean conversion rate per user, mean session length per user). But many of the most important product metrics are **ratios of two random quantities**, both of which vary across users: click-through rate (clicks/impressions), revenue per session (revenue/sessions), average order value (revenue/orders), watch-time-per-view (total watch time/number of views).

The intuition for why this matters: **you cannot just treat a ratio metric as if it were a simple average and plug it into the standard SE formula from Chapter 6.** The numerator and denominator are both random variables that vary user-to-user and are typically *correlated* with each other — and naively ignoring this gives you a **wrong, usually too-small, standard error**, which means false confidence and inflated false-positive rates.

## 2. Why Simple Averaging Breaks for Ratios

Consider CTR computed at the **user level**: for user $i$, $R_i = \frac{clicks_i}{impressions_i}$. If you naively average $R_i$ across users and compute a standard t-test on those per-user ratios, you're treating each user's ratio as if it carries equal weight and equal reliability — but a user with 1 impression and 1 click has $R_i=1.0$, exactly like a user with 1,000 impressions and 1,000 clicks, even though the second user's ratio is a vastly more reliable estimate of their true click propensity. Naive per-user-ratio averaging throws away this information.

The more common (and more correct) approach computes the **aggregate ratio**: $R = \frac{\sum_i clicks_i}{\sum_i impressions_i}$ — the overall pooled ratio, not the average of individual ratios. But now you have a new problem: the standard error of a ratio of two summed random variables is **not** simply derived from the individual variances of numerator and denominator alone — you need to account for their covariance too, and the ratio itself is a *nonlinear* function of two random variables, so straightforward variance formulas for sums don't directly apply.

## 3. The Delta Method

The **delta method** is a general technique for approximating the variance of a (nonlinear) function of random variables, using a first-order Taylor expansion around the means.

For a ratio metric $R = X/Y$ where $X$ (numerator, e.g., total clicks) and $Y$ (denominator, e.g., total impressions) are both random (varying by randomization unit, e.g., per-user), the delta method gives:

$$Var(R) \approx \frac{1}{\bar{Y}^2}\left[Var(X) + \bar{R}^2 Var(Y) - 2\bar{R}\cdot Cov(X,Y)\right]$$

where $\bar{X}, \bar{Y}$ are the means and $\bar{R}=\bar{X}/\bar{Y}$.

**Where this comes from, intuitively**: this is just a first-order Taylor expansion of $R=X/Y$ around $(\bar{X},\bar{Y})$, using the standard result that for a function $g(X,Y)$:

$$Var(g(X,Y)) \approx \left(\frac{\partial g}{\partial X}\right)^2 Var(X) + \left(\frac{\partial g}{\partial Y}\right)^2 Var(Y) + 2\frac{\partial g}{\partial X}\frac{\partial g}{\partial Y}Cov(X,Y)$$

with $\frac{\partial g}{\partial X}=\frac{1}{Y}$ and $\frac{\partial g}{\partial Y}=-\frac{X}{Y^2}$, evaluated at the means.

**The critical, easy-to-miss term**: the $-2\bar{R}\cdot Cov(X,Y)$ term. If clicks and impressions are positively correlated (very typically true — users who get shown more impressions also tend to click more in absolute terms), **ignoring this covariance term overstates the variance**, making your test overly conservative (too few false positives, but also underpowered — you'd miss real effects). In other cases, depending on the sign and magnitude of the correlation, ignoring it can understate variance instead — the direction of the error depends entirely on the actual covariance structure of your specific metric, which is exactly why you can't skip this term and hope it's negligible.

## 4. Worked Example

You're computing CTR = clicks/impressions at the user level, aggregated across 1,000 users in the treatment group.

Suppose (from your data):
- $\bar{X}$ (mean clicks per user) = 5
- $\bar{Y}$ (mean impressions per user) = 50
- $\bar{R} = 5/50 = 0.10$ (10% CTR)
- $Var(X) = 20$, $Var(Y) = 400$, $Cov(X,Y) = 60$ (clicks and impressions are positively correlated, as expected — more active users generate more of both)

Plugging into the delta method formula:

$$Var(R) \approx \frac{1}{50^2}\left[20 + (0.10)^2 \times 400 - 2(0.10)(60)\right] = \frac{1}{2500}\left[20 + 4 - 12\right] = \frac{12}{2500} = 0.0048$$

$$SE(R) = \sqrt{0.0048/1000} \approx 0.00219$$

(Dividing by $n=1000$ here because $Var(X), Var(Y), Cov(X,Y)$ were per-user variances, and we need the variance of the *mean* ratio across the sample — same logic as dividing a per-observation variance by $n$ to get the SE of a sample mean.)

**Contrast with the naive (wrong) approach**: if you'd ignored the covariance term entirely (a common shortcut), you'd compute:

$$Var(R)_{naive} \approx \frac{1}{2500}[20+4] = \frac{24}{2500}=0.0096$$

This is **exactly double** the correct variance in this example — the naive approach overstates the SE by ignoring the positive covariance, making your CI unnecessarily wide and your test underpowered. This concrete numeric contrast (correct 0.0048 vs naive 0.0096) is the kind of thing worth being able to reproduce live if asked "why does the delta method matter here."

## 5. Production Considerations

- **Most experimentation platforms at scale (Google, Meta-style infrastructure) compute ratio metric variances via the delta method or via bootstrap resampling** — bootstrap is a more computationally expensive but assumption-free alternative that naturally captures the numerator-denominator covariance without needing the closed-form formula. Worth mentioning as the practical alternative when the delta method's linear approximation may not hold well (e.g., very skewed or small-sample ratios).
- **Randomization unit vs. analysis unit mismatch is a related, adjacent trap**: if you randomize by user but analyze at the impression or session level (treating each impression as an independent observation), you dramatically overstate your effective sample size and understate your true SE — a distinct but related error to the ratio-metric problem, since impressions from the same user are correlated, not independent (this connects back to the clustering/interference ideas in Chapter 7).
- **CUPED (Chapter 12/next module) applies to ratio metrics too**, but requires extending the delta-method variance formula to also incorporate covariance with the pre-experiment covariate — a natural extension worth flagging if you're asked to go deep on ratio metric variance reduction specifically.

## 6. Interview Traps

- **Trap #1**: Treating a ratio metric like a simple mean and plugging observed per-user ratios directly into the standard two-sample t-test SE formula from Chapter 6 — ignoring both the weighting problem (Section 2) and the covariance problem (Section 3).
- **Trap #2**: Using the delta method formula but forgetting the covariance term entirely, silently introducing a (possibly large) bias in your variance estimate — direction of the bias depends on the sign of the correlation, so it's not automatically "safe" to omit.
- **Trap #3**: Not recognizing when the delta method's linear approximation breaks down (e.g., ratios with denominators close to zero, or highly skewed numerator/denominator distributions) — in these cases, flag bootstrap as the safer alternative rather than forcing the delta method to apply.
- **Trap #4**: Confusing the "ratio metric" variance problem with the "clustered/non-independent observations" problem (analysis unit ≠ randomization unit) — these are related but distinct sources of SE misestimation, and interviewers may probe whether you can tell them apart.

## 7. L5-Differentiating Talking Points

- Being able to write out the delta method's Taylor expansion derivation, even briefly, rather than just quoting the final formula, shows you understand where it comes from rather than having memorized a lookup-table result.
- Proactively raising that the covariance term's sign/magnitude determines whether ignoring it makes your test too conservative or too liberal — rather than assuming it's "always fine to ignore" or "always makes you conservative" — shows precise, non-hand-wavy understanding.
- Mentioning bootstrap resampling as the assumption-free alternative when the delta method's approximation may be shaky (small denominators, heavy skew) demonstrates breadth beyond the one canonical formula.
- Connecting this chapter to both Chapter 7 (randomization unit vs. analysis unit) and the upcoming CUPED chapter shows you see ratio-metric variance estimation as one node in a connected web of "getting your standard errors right" problems, not an isolated formula to memorize.

## 8. Comprehension Check

1. Why can't you just compute the average of each user's individual ratio (clicks/impressions per user) and run a standard t-test on those averages?
2. Write the delta method variance formula for a ratio $R=X/Y$ and explain where the covariance term comes from.
3. In the worked example, ignoring the covariance term doubled the estimated variance. Under what circumstances would ignoring the covariance term instead *understate* the true variance?
4. What's a practical, assumption-free alternative to the delta method when the ratio's denominator is close to zero or highly skewed?
5. Explain the difference between the "ratio metric variance" problem in this chapter and the "randomization unit vs. analysis unit" problem — are these the same issue or different ones?

---
*Next: Chapter 12 — Variance Reduction (CUPED)*
