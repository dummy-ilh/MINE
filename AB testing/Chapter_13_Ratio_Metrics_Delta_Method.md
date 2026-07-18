# Chapter 13: Ratio Metrics & the Delta Method
---

## 1. Intuition

So far, most metrics discussed elsewhere in this curriculum are simple averages per randomization unit (mean conversion rate per user, mean session length per user). But many of the most important product metrics are **ratios of two random quantities**, both of which vary across users: click-through rate (clicks/impressions), revenue per session (revenue/sessions), average order value (revenue/orders), clicks per session, watch-time-per-view.

**The core problem**: you cannot just treat a ratio metric as if it were a simple average and plug it into the standard SE-of-a-mean formula. The numerator and denominator are both random variables that vary user-to-user and are typically *correlated* with each other — and naively ignoring this gives you a **wrong** standard error (too small OR too large, depending on the correlation's sign), which means either false confidence and inflated false-positive rates, or an unnecessarily wide interval and lost power.

### Layman analogy
Imagine measuring "clicks per session" for two users:
- User A: 10 clicks across 2 sessions → 5 clicks/session
- User B: 3 clicks across 1 session → 3 clicks/session

If User A logs in more (more sessions) AND clicks more per session, the numerator (clicks) and denominator (sessions) are moving together — they're correlated. If you treat "clicks per session" like a simple, independent measurement per user and compute variance the naive way, you completely ignore that correlation, and your estimated variance ends up wrong — potentially too small, making you overconfident (your confidence intervals look tighter than they should, and you might see "significant" results that are really just noise) — or too large, understating your power.

The **Delta Method** is a mathematical trick that says: "since a ratio is a nonlinear function of two random things, let's approximate how it wobbles by looking at how small wobbles in the numerator and denominator translate into wobbles in the ratio — accounting for how those two things move together." It's like estimating how much a seesaw tips based on small pushes on both ends simultaneously, rather than assuming only one end is moving.

---

## 2. Core Definitions

- **Ratio metric**: a metric expressed as one quantity divided by another, where **both** the numerator and denominator vary randomly across the randomization unit — e.g., "clicks per session," "revenue per visitor," "average watch time per view," CTR (clicks/impressions when impressions vary per user).
- This **differs** from a simple average of a single per-user value (e.g., "average number of clicks per user," where each user contributes one fixed observation and the "denominator" — 1 user — isn't random at all).
- **The Delta Method**: the standard statistical technique for approximating the variance of a ratio (or any nonlinear function of random variables) correctly, using a first-order Taylor expansion around the means.

---

## 3. Why Simple Averaging Breaks for Ratios

### Problem 1 — Naive per-user-ratio averaging discards reliability information
Consider CTR computed at the **user level**: for user $i$, $R_i = \frac{clicks_i}{impressions_i}$. If you naively average $R_i$ across users and run a standard t-test on those per-user ratios, you're treating each user's ratio as if it carries equal weight and equal reliability — but a user with 1 impression and 1 click has $R_i=1.0$, exactly like a user with 1,000 impressions and 1,000 clicks, even though the second user's ratio is a vastly more reliable estimate of their true click propensity. Naive per-user-ratio averaging throws away this information.

The more common (and more correct) approach computes the **aggregate/pooled ratio**: $R = \frac{\sum_i clicks_i}{\sum_i impressions_i}$ — the overall pooled ratio, not the average of individual ratios.

### Problem 2 — The pooled ratio still needs the right variance formula
Even using the pooled ratio, the standard error of a ratio of two summed random variables is **not** simply derived from the individual variances of numerator and denominator alone — you need to account for their **covariance** too, because the ratio itself is a *nonlinear* function of two random variables, so straightforward variance formulas for sums don't directly apply. This is exactly what the Delta Method solves.

**When this matters most**: any metric that's a ratio *at the user level* where both parts vary — CTR (when impressions vary per user), revenue per session, avg items per cart. It does **NOT** apply to simple per-user averages like "average clicks per user," where the denominator (1 user = 1 observation) is fixed, not random — the standard mean-variance formula is already correct there, and applying the Delta Method would be unnecessary complexity.

---

## 4. The Delta Method — Formal Derivation

For a ratio metric $R = X/Y$ where $X$ (numerator, e.g., total/average clicks) and $Y$ (denominator, e.g., total/average impressions or sessions) are both random (varying by randomization unit, e.g., per-user), the delta method gives:

$$Var(R) \approx \frac{1}{\bar{Y}^2}\left[Var(X) + \bar{R}^2 Var(Y) - 2\bar{R}\cdot Cov(X,Y)\right]$$

where $\bar{X}, \bar{Y}$ are the means and $\bar{R}=\bar{X}/\bar{Y}$. (Equivalently, in relative terms: $\frac{Var(R)}{R^2} \approx \frac{Var(X)}{\bar{X}^2} - \frac{2\,Cov(X,Y)}{\bar{X}\bar{Y}} + \frac{Var(Y)}{\bar{Y}^2}$.)

### Where this comes from
This is a first-order Taylor expansion of $R=X/Y$ around $(\bar{X},\bar{Y})$, using the standard result that for a function $g(X,Y)$:

$$Var(g(X,Y)) \approx \left(\frac{\partial g}{\partial X}\right)^2 Var(X) + \left(\frac{\partial g}{\partial Y}\right)^2 Var(Y) + 2\frac{\partial g}{\partial X}\frac{\partial g}{\partial Y}Cov(X,Y)$$

with $\frac{\partial g}{\partial X}=\frac{1}{Y}$ and $\frac{\partial g}{\partial Y}=-\frac{X}{Y^2}$, evaluated at the means.

### The critical, easy-to-miss term: covariance
This is the single most common practical mistake to get wrong — either by omitting the covariance term entirely, or by wrongly assuming it always pushes variance in one direction:

- **If X and Y are positively correlated** (numerator and denominator move together — very typically true, since users who get shown more impressions/sessions also tend to generate more clicks in absolute terms): the covariance term **reduces** the variance of the ratio relative to treating them as independent, because when the denominator goes up, the numerator tends to go up proportionally too, partially canceling out. **Ignoring this covariance term overstates the variance** — making your test overly conservative (CI too wide, test underpowered — you'd miss real effects), NOT overconfident.
- **If X and Y are negatively correlated** (rarer, but possible — e.g., users who visit more often might convert at a lower rate per visit due to habituation): the covariance term **inflates** the true variance relative to naive estimates. Ignoring it makes your CI too narrow, meaning you could see false "significant" results — the overconfidence failure mode.
- **If uncorrelated**, the covariance term drops to zero and the formula simplifies — but it's still not the same as the naive single-mean variance formula, since $Var(Y)$ still contributes.

**The key takeaway**: the direction of the error from ignoring covariance depends entirely on the actual covariance structure of your specific metric — you can't skip this term and assume it's automatically "safe" or "conservative" in one particular direction.

---

## 5. Worked Examples

### Example A — CTR (clicks/impressions), positive covariance

You're computing CTR = clicks/impressions at the user level, aggregated across 1,000 users in the treatment group.

Per-user statistics:
- $\bar{X}$ (mean clicks per user) = 5
- $\bar{Y}$ (mean impressions per user) = 50
- $\bar{R} = 5/50 = 0.10$ (10% CTR)
- $Var(X) = 20$, $Var(Y) = 400$, $Cov(X,Y) = 60$ (clicks and impressions positively correlated — more active users generate more of both)

**Delta Method (correct):**

$$Var(R) \approx \frac{1}{50^2}\left[20 + (0.10)^2 \times 400 - 2(0.10)(60)\right] = \frac{1}{2500}\left[20 + 4 - 12\right] = \frac{12}{2500} = 0.0048$$

$$SE(R) = \sqrt{0.0048/1000} \approx 0.00219$$

(Dividing by $n=1000$ because $Var(X), Var(Y), Cov(X,Y)$ were per-user variances, and we need the variance of the *mean* ratio across the sample — the same logic as dividing a per-observation variance by n to get the SE of a sample mean.)

**Naive (wrong) approach**, ignoring the covariance term entirely:

$$Var(R)_{naive} \approx \frac{1}{2500}[20+4] = \frac{24}{2500}=0.0096$$

This is **exactly double** the correct variance in this example — the naive approach overstates the SE by ignoring the positive covariance, making your CI unnecessarily wide and your test underpowered. This concrete numeric contrast (correct 0.0048 vs. naive 0.0096) is worth being able to reproduce live if asked "why does the delta method matter here."

### Example B — Clicks per session, positive covariance (different numbers, same pattern)

Per-user data (aggregated over a test period):
- $\bar{X}$ (average clicks per user) = 8.0, $Var(X) = 16$
- $\bar{Y}$ (average sessions per user) = 4.0, $Var(Y) = 4$
- $Cov(X,Y) = 5$ (clicks and sessions move together — heavier users have both more clicks and more sessions)
- $R = \bar{X}/\bar{Y} = 8.0/4.0 = 2.0$ clicks per session

**Delta Method (correct):**

$$Var(R) \approx \frac{1}{16}\left[16 - 2(2.0)(5) + (2.0)^2(4)\right] = \frac{1}{16}[16-20+16] = \frac{12}{16} = 0.75$$

Notice the covariance term (−20) substantially reduced what would have been a larger variance (16 + 16 = 32 without it) down to 12 before dividing by $\bar{Y}^2$. This reflects that clicks and sessions move together — a user with an unusually high session count also tends to have proportionally more clicks, so the *ratio* itself is more stable than either raw number alone.

**Naive (wrong) approach**, ignoring covariance and treating X, Y as independent:

$$Var(R)_{naive} \approx \frac{1}{16}(16+16) = 2.0$$

**2.0 vs. the correct 0.75 — nearly 3x too large**, producing an unnecessarily wide, overly conservative confidence interval and understating your true statistical power. (Note: this example's naive baseline still uses the ratio-formula structure with $\bar{Y}^2$ in the denominator, just omitting the covariance cross-term — a different, cruder mistake than Example A's naive approach, which also omitted covariance from the same formula structure. Either way, omitting covariance is the shared error.)

**Takeaway from both examples**: whenever numerator and denominator are positively correlated (the common case), omitting the covariance term makes your variance estimate too large, not too small — the opposite of what many people instinctively assume "ignoring a complication" would do.

---

## 6. Levers — What Controls Ratio-Metric Variance

**Correlation between numerator and denominator**
- Strong positive correlation shrinks the true variance of the ratio relative to naive estimates — ignoring this makes your CI too wide (overly conservative, underpowered, but not falsely overconfident).
- Strong negative correlation (rarer — e.g., users who visit more often might convert at a lower rate per visit due to habituation) inflates the true variance relative to naive estimates — ignoring this makes your CI too narrow, risking false "significant" results.

**Granularity / level at which the ratio is computed**
- A ratio metric defined at the user level (e.g., avg clicks-per-session, averaged first within-user then across users) behaves differently statistically than one computed as a pooled ratio across all sessions (total clicks / total sessions, ignoring user boundaries) — the latter can bias toward heavy users (a user with many sessions contributes many "observations," implicitly overweighting them) and needs its own correction via stratification or clustering-aware variance estimation.
- Computing session-level ratios and averaging those (rather than aggregating to the user level first) risks implicitly weighting all sessions equally regardless of how many sessions each user contributes, and can mask the true numerator/denominator covariance structure at the user level.

**Choice of denominator / metric redefinition**
- Some ratio metrics can be redefined to avoid the problem entirely — e.g., instead of "clicks per session" (both random), you might use "did the user click at all" (binary, single random variable per user) as a simpler, more robust proxy metric, trading some information for statistical simplicity.

---

## 7. Production Considerations

- **Most experimentation platforms at scale compute ratio-metric variances via the delta method or via bootstrap resampling.** Bootstrap is a more computationally expensive but assumption-free alternative that naturally captures the numerator-denominator covariance without needing the closed-form formula — worth mentioning as the practical alternative when the delta method's linear approximation may not hold well (e.g., very skewed or small-sample ratios, or denominators close to zero).
- **Randomization unit vs. analysis unit mismatch is a related, adjacent trap**: if you randomize by user but analyze at the impression or session level (treating each impression as an independent observation), you dramatically overstate your effective sample size and understate your true SE — a distinct but related error to the ratio-metric problem, since impressions/sessions from the same user are correlated, not independent (this connects to the clustering/interference concepts covered elsewhere in this curriculum).
- **Variance-reduction techniques (CUPED) apply to ratio metrics too**, but require extending the delta-method variance formula to also incorporate covariance with the pre-experiment covariate — a natural extension worth flagging if asked to go deep on ratio-metric variance reduction specifically.
- **When NOT to worry about the Delta Method**: when the denominator isn't actually random — e.g., "average clicks per user" where the denominator is just "1 user" (a fixed unit of observation, not a random count). This is a simple mean, and the standard variance formula applies directly with no ratio-metric correction needed.

---

## 8. Interview Traps (Consolidated)

1. **Treating a ratio metric like a simple mean** and plugging observed per-user ratios directly into the standard two-sample t-test SE formula — ignoring both the weighting problem (Section 3, Problem 1) and the covariance problem (Section 3, Problem 2).
2. **Using the delta method formula but forgetting the covariance term entirely**, silently introducing a (possibly large) bias in your variance estimate — direction of the bias depends on the sign of the correlation, so it's not automatically "safe" to omit.
3. **Assuming ignoring covariance is automatically conservative (or automatically liberal)** — it can go either way depending on the correlation's sign (Section 4).
4. **Not recognizing when the delta method's linear approximation breaks down** (e.g., ratios with denominators close to zero, or highly skewed numerator/denominator distributions) — in these cases, flag bootstrap as the safer alternative rather than forcing the delta method to apply.
5. **Confusing the "ratio metric" variance problem with the "clustered/non-independent observations" problem** (analysis unit ≠ randomization unit) — related but distinct sources of SE misestimation; interviewers may probe whether you can tell them apart.
6. **Computing ratio metrics by pooling at the wrong level** (e.g., session-level instead of user-level), implicitly overweighting heavy users.
7. **Applying the Delta Method to a metric where the denominator is actually fixed** (e.g., a true simple per-user average) — unnecessary complexity where the standard formula is already correct.

---

## 9. Common Mistakes / Red Flags — Quick Review

- ❌ Applying the standard variance-of-a-mean formula directly to a ratio metric without accounting for denominator variance and covariance
- ❌ Assuming ignoring the covariance term is automatically "safe" or conservative — the direction of the error depends on the sign of the correlation
- ❌ Computing ratio metrics by pooling at the wrong level (e.g., session-level instead of user-level), implicitly overweighting heavy users
- ❌ Applying the Delta Method to a metric where the denominator is actually fixed (e.g., a true simple per-user average) — unnecessary complexity where the standard formula is already correct
- ❌ Forcing the delta-method linear approximation onto a ratio with a denominator near zero or heavy skew, instead of switching to bootstrap
- ✅ Check whether both numerator and denominator vary at your unit of randomization before deciding a ratio-metric correction is needed
- ✅ Estimate Cov(X,Y) from your data rather than assuming independence by default
- ✅ Aggregate to the user level (or your randomization unit) before computing the ratio, rather than pooling at a finer, non-independent level

---

## 10. Famous Interview Q&A

**Q: You're testing a new search ranking algorithm and using "clicks per search session" as your primary metric. Why can't you just use the standard variance-of-the-mean formula on this ratio?**
A: Because both clicks and sessions vary randomly per user, and treating "clicks per session" as if it were a simple per-user average ignores the covariance between the two — if users who search more often also click proportionally more (a very plausible pattern), that covariance meaningfully changes the true variance of the ratio, and ignoring it produces a biased (often overly conservative, sometimes overconfident) estimate of the metric's variance. I'd apply the Delta Method to correctly account for Var(X), Var(Y), and Cov(X,Y) together rather than just using Var(X)/n.

**Q: If the numerator and denominator of your ratio metric are positively correlated, does ignoring the correlation make your test too conservative or too aggressive?**
A: Positive correlation between numerator and denominator actually *reduces* the true variance of the ratio relative to what you'd naively estimate by treating them as independent — so ignoring it means your naive variance estimate is too large, making your confidence interval too wide and your test too conservative (you might fail to detect a real effect that a properly-calculated, tighter interval would have caught). This is the opposite direction of failure from negative correlation, which would make naive estimates too small and the test falsely overconfident — so the direction of the mistake genuinely depends on the correlation's sign, and can't be assumed to always be "safe."

**Q: A junior analyst computes a confidence interval on "revenue per session" by just applying the standard formula for the variance of a mean to the ratio values directly (session-level revenue/session, averaged). What's the issue?**
A: The core issue is which level the ratio is computed and averaged at. If they're computing revenue-per-session for each individual session and then averaging those session-level ratios, they may be implicitly weighting all sessions equally regardless of how many sessions each user contributes — heavy users (many sessions) get proportionally more influence just by having more observations, and simple pooling can also mask the true numerator/denominator covariance structure at the user level. The Delta Method, applied correctly at the user level (using per-user aggregated numerator and denominator, X̄ and Ȳ), gives a more defensible variance estimate that properly accounts for the user-level covariance structure, rather than pretending session-level ratios are independent, identically distributed observations.

**Q: When would you NOT need to worry about the Delta Method for a metric that looks like a ratio?**
A: When the denominator isn't actually random — e.g., "average clicks per user" where the denominator is just "1 user" (a fixed unit of observation, not a random count). This is a simple mean, and the standard variance formula applies directly with no ratio-metric correction needed. The Delta Method specifically matters when BOTH the numerator and denominator are random quantities that vary across your randomization unit — like sessions-per-user or revenue-per-visit where the "per" part itself fluctuates.

**Q: What's a practical, assumption-free alternative to the delta method when the ratio's denominator is close to zero or highly skewed?**
A: Bootstrap resampling — repeatedly resample your randomization units (e.g., users) with replacement, recompute the pooled ratio on each resample, and use the empirical distribution of those resampled ratios to estimate variance/confidence intervals directly. It's more computationally expensive than the closed-form delta method, but it naturally captures the numerator-denominator covariance and doesn't rely on the delta method's linear (Taylor-expansion) approximation, which can break down for small or skewed denominators.

---

## 11. L5-Differentiating Talking Points

- Being able to write out the delta method's Taylor expansion derivation, even briefly, rather than just quoting the final formula, shows you understand where it comes from rather than having memorized a lookup-table result.
- Proactively raising that the covariance term's sign/magnitude determines whether ignoring it makes your test too conservative or too liberal — rather than assuming it's "always fine to ignore" or "always makes you conservative" — shows precise, non-hand-wavy understanding.
- Mentioning bootstrap resampling as the assumption-free alternative when the delta method's approximation may be shaky (small denominators, heavy skew) demonstrates breadth beyond the one canonical formula.
- Connecting this topic to both the randomization-unit-vs-analysis-unit problem and to CUPED/variance-reduction shows you see ratio-metric variance estimation as one node in a connected web of "getting your standard errors right" problems, not an isolated formula to memorize.
- Being able to reproduce a concrete numeric contrast (correct vs. naive variance, as in both worked examples) live, on request, is a strong differentiator over reciting the formula alone.

---

## 12. Comprehension Check (Self-Test)

1. Why can't you just compute the average of each user's individual ratio (clicks/impressions per user) and run a standard t-test on those averages?
2. Write the delta method variance formula for a ratio $R=X/Y$ and explain where the covariance term comes from.
3. In the worked examples, ignoring the covariance term overstated the variance (by 2x and ~2.7x respectively). Under what circumstances would ignoring the covariance term instead *understate* the true variance?
4. What's a practical, assumption-free alternative to the delta method when the ratio's denominator is close to zero or highly skewed?
5. Explain the difference between the "ratio metric variance" problem in this tutorial and the "randomization unit vs. analysis unit" problem — are these the same issue or different ones?
6. When would you NOT need to apply the Delta Method to a metric that superficially looks like a ratio?
7. A ratio metric is computed by averaging session-level ratios instead of aggregating to the user level first. What two problems does this introduce?
8. Compute, by hand, the delta-method variance for a ratio with $\bar{X}=10$, $\bar{Y}=20$, $Var(X)=25$, $Var(Y)=16$, $Cov(X,Y)=8$. Compare it to the naive estimate that omits the covariance term.

---
*This tutorial merges two chapters on ratio metrics and the Delta Method — one framed around the CTR example and a Taylor-expansion derivation, the other framed around a layman clicks-per-session analogy and a second worked example with a ~3x naive/correct discrepancy. Both worked examples were kept since they illustrate the same underlying error (omitting covariance) with different metrics and different magnitudes of consequence. All levers, traps, and Q&A from both sources are consolidated with duplicates merged.*
