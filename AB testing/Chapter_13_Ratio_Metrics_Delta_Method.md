# Chapter 13: Ratio Metrics & the Delta Method

## 1. Definition

A **ratio metric** is a metric expressed as one quantity divided by another, where both the numerator and denominator vary randomly across the randomization unit — e.g., "clicks per session" (numerator: clicks, denominator: sessions, both random per user), "revenue per visitor" (both numerator and denominator vary), or "average watch time per view."

This differs from a simple average of a single per-user value (e.g., "average number of clicks per user," where each user contributes one fixed observation). When both numerator and denominator are themselves random variables, the naive formula for variance of a ratio (just plugging into the standard variance-of-a-mean formula) is **wrong**, because it ignores the covariance between numerator and denominator.

The **Delta Method** is the standard statistical technique for approximating the variance of a ratio (or any non-linear function of random variables) correctly, using a first-order Taylor expansion around the means.

## 2. Layman Explanation

Imagine you're measuring "clicks per session" for two users:
- User A: 10 clicks across 2 sessions → 5 clicks/session
- User B: 3 clicks across 1 session → 3 clicks/session

If User A logs in more (more sessions) AND clicks more per session, the numerator (clicks) and denominator (sessions) are moving together — they're correlated. If you treat "clicks per session" like a simple, independent measurement per user and compute variance the naive way, you completely ignore that correlation, and your estimated variance ends up wrong — usually too small, making you overconfident (your confidence intervals look tighter than they should, and you might see "significant" results that are really just noise).

The Delta Method is a mathematical trick that says: "since a ratio is a nonlinear function of two random things, let's approximate how it wobbles by looking at how small wobbles in the numerator and denominator translate into wobbles in the ratio — accounting for how those two things move together." It's like estimating how much a seesaw tips based on small pushes on both ends simultaneously, rather than assuming only one end is moving.

## 3. Formal Explanation

**Why naive variance formulas fail:**

For a ratio metric R = X̄/Ȳ (e.g., X̄ = average clicks, Ȳ = average sessions, both per user), the naive approach might compute:

Var(R) ≈ Var(X̄)/n (treating R like a simple mean)

This is wrong because it ignores:
1. Var(Ȳ) — variance from the denominator itself
2. Cov(X̄, Ȳ) — the covariance between numerator and denominator

**Delta Method formula (first-order Taylor approximation):**

For R = X/Y, the approximate variance is:

Var(R) ≈ (1/Ȳ²) × [Var(X) - 2R·Cov(X,Y) + R²·Var(Y)]

Or equivalently, in relative terms:

Var(R)/R² ≈ Var(X)/X̄² - 2·Cov(X,Y)/(X̄·Ȳ) + Var(Y)/Ȳ²

**Intuition behind the formula:**
- If X and Y are positively correlated (numerator and denominator move together, as in the clicks/sessions example), the covariance term *reduces* the variance of the ratio relative to treating them as independent — because when sessions go up, clicks tend to go up proportionally too, partially canceling out.
- If X and Y are uncorrelated, the covariance term drops to zero and the formula simplifies, though it's still not the same as the naive single-mean variance formula.
- Getting the sign of the covariance term wrong (or ignoring it) is the single most common practical mistake — it can make your variance estimate too large OR too small depending on the true correlation structure, so you can't just "assume it's conservative to ignore it."

**When this matters most:**
Any metric that's a ratio *at the user level* where both parts vary — CTR (clicks/impressions, when impressions vary per user), revenue per session, avg items per cart. It does NOT apply to simple per-user averages like "average clicks per user" where the denominator (1 user = 1 observation) is fixed, not random.

## 4. Levers — What Controls It, What Moves It

**Correlation between numerator and denominator**
- Strong positive correlation shrinks the true variance of the ratio relative to naive estimates — ignoring this makes your CI too wide (overly conservative, but at least not falsely overconfident).
- Strong negative correlation (rarer, but possible — e.g., users who visit more often might convert at a lower rate per visit due to habituation) inflates the true variance relative to naive estimates — ignoring this makes your CI too narrow, meaning you could see false "significant" results.

**Granularity of the ratio's denominator**
- A ratio metric defined at the user level (e.g., avg clicks-per-session, averaged first within-user then across users) behaves differently statistically than one computed as a pooled ratio across all sessions (total clicks / total sessions, ignoring user boundaries) — the latter can bias toward heavy users and needs its own correction (see also Chapter 14 on stratification).

**Choice of denominator**
- Some ratio metrics can be redefined to avoid the problem entirely — e.g., instead of "clicks per session" (both random), you might use "did the user click at all" (binary, single random variable per user) as a simpler, more robust proxy metric, trading some information for statistical simplicity.

## 5. Worked Example

Suppose per-user data (aggregated over a test period) gives you:
- X̄ (average clicks per user) = 8.0, Var(X) = 16
- Ȳ (average sessions per user) = 4.0, Var(Y) = 4
- Cov(X, Y) = 5 (clicks and sessions move together — heavier users have both more clicks and more sessions)

Ratio R = X̄/Ȳ = 8.0/4.0 = 2.0 clicks per session.

**Naive (wrong) approach** — treating this like a simple mean's variance:
Var(R)_naive ≈ Var(X)/n = 16/n (completely ignoring Y's variance and the covariance — this isn't even using the right formula structure)

**Delta Method (correct) approach:**
Var(R) ≈ (1/Ȳ²) × [Var(X) - 2R·Cov(X,Y) + R²·Var(Y)]
Var(R) ≈ (1/16) × [16 - 2(2.0)(5) + (2.0)²(4)]
Var(R) ≈ (1/16) × [16 - 20 + 16]
Var(R) ≈ (1/16) × 12
Var(R) ≈ 0.75

Notice how the covariance term (-20) substantially reduced what would have been a larger variance (16 + 16 = 32 without it) down to 12 before dividing by Ȳ². This reflects the fact that clicks and sessions move together — a user with an unusually high session count also tends to have proportionally more clicks, so the *ratio* itself is more stable than either raw number alone. If you'd ignored the covariance term entirely (treating X and Y as independent), you'd have overestimated Var(R) as (1/16)(16+16) = 2.0 instead of the correct 0.75 — nearly 3x too large, producing an unnecessarily wide, overly conservative confidence interval and understating your true statistical power.

## 6. Famous Q&A (Google / Apple style)

**Q: You're testing a new search ranking algorithm and using "clicks per search session" as your primary metric. Why can't you just use the standard variance-of-the-mean formula on this ratio?**
A: Because both clicks and sessions vary randomly per user, and treating "clicks per session" as if it were a simple per-user average ignores the covariance between the two — if users who search more often also click proportionally more (a very plausible pattern), that covariance meaningfully changes the true variance of the ratio, and ignoring it produces a biased (often overconfident) estimate of the metric's variance. I'd apply the Delta Method to correctly account for Var(X), Var(Y), and Cov(X,Y) together rather than just using Var(X)/n.

**Q: If the numerator and denominator of your ratio metric are positively correlated, does ignoring the correlation make your test too conservative or too aggressive?**
A: Positive correlation between numerator and denominator actually *reduces* the true variance of the ratio relative to what you'd naively estimate by treating them as independent — so ignoring it means your naive variance estimate is too large, making your confidence interval too wide and your test too conservative (you might fail to detect a real effect that a properly-calculated, tighter interval would have caught). This is the opposite direction of failure from negative correlation, which would make naive estimates too small and the test falsely overconfident — so the direction of the mistake genuinely depends on the correlation's sign, and can't be assumed to always be "safe."

**Q: A junior analyst computes a confidence interval on "revenue per session" by just applying the standard formula for the variance of a mean to the ratio values directly (session-level revenue/session, averaged). What's the issue?**
A: The core issue is which level the ratio is computed and averaged at. If they're computing revenue-per-session for each individual session and then averaging those session-level ratios, they may be implicitly weighting all sessions equally regardless of how many sessions each user contributes — heavy users (many sessions) get proportionally more influence just by having more observations, and simple pooling can also mask the true numerator/denominator covariance structure at the user level. The Delta Method, applied correctly at the user level (using per-user aggregated numerator and denominator, X̄ and Ȳ), gives a more defensible variance estimate that properly accounts for the user-level covariance structure, rather than pretending session-level ratios are independent, identically distributed observations.

**Q: When would you NOT need to worry about the Delta Method for a metric that looks like a ratio?**
A: When the denominator isn't actually random — e.g., "average clicks per user" where the denominator is just "1 user" (a fixed unit of observation, not a random count), this is a simple mean, and the standard variance formula applies directly with no ratio-metric correction needed. The Delta Method specifically matters when BOTH the numerator and denominator are random quantities that vary across your randomization unit — like sessions-per-user or revenue-per-visit where the "per" part itself fluctuates.

## 7. Common Mistakes / Red Flags (Quick Review)

- ❌ Applying the standard variance-of-a-mean formula directly to a ratio metric without accounting for denominator variance and covariance
- ❌ Assuming ignoring the covariance term is automatically "safe" or conservative — the direction of the error depends on the sign of the correlation
- ❌ Computing ratio metrics by pooling at the wrong level (e.g., session-level instead of user-level), implicitly overweighting heavy users
- ❌ Applying the Delta Method to a metric where the denominator is actually fixed (e.g., a true simple per-user average) — unnecessary complexity where the standard formula is already correct
- ✅ Do: check whether both numerator and denominator vary at your unit of randomization before deciding a ratio-metric correction is needed
- ✅ Do: estimate Cov(X,Y) from your data rather than assuming independence by default

---
*Next: Chapter 14 — Variance Reduction: CUPED, Stratified Sampling.*
