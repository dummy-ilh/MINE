# 100-Day A/B Testing Interview Prep
### FAANG Interviewer POV | 1 Question/Day | Deep Follow-ups
### Days 1–50: L4 Level | Days 51–100: L6 Level

> Format per day:
> 🎯 **Main Q** — What the interviewer opens with
> 💬 **Model Answer** — What a strong candidate says
> ↳ **Follow-up** — What the interviewer probes next
> 🚩 **Red Flag** — What a weak candidate says
> ⭐ **What earns the role** — The differentiating insight

---

# DAYS 1–50: L4 LEVEL
*Foundations, core statistics, design, common pitfalls*

---

## Day 1 — What is A/B Testing?

🎯 **"Explain A/B testing to me like I'm a PM who's never heard of it."**

💬 **Model Answer**
A/B testing is a controlled experiment. You split your users randomly into two groups — one sees the current experience (control), one sees a change (treatment). You measure a metric on both groups and use statistics to determine whether the difference is real or just random noise. The goal is to make causal claims: did *this change* cause *this outcome*?

↳ **"Why do we need randomization? Why not just compare before vs after?"**
Before-after comparisons confound the treatment with time — seasonality, news events, other product changes all happen simultaneously. Randomization ensures the only systematic difference between groups is the change being tested, so we can attribute any metric difference to that change alone.

↳ **"Give me a real example of where before-after analysis would mislead you."**
You launch a new homepage in December. Revenue goes up 20%. Is it the homepage or Christmas shopping? You can't tell. A concurrent A/B test with a holdout would have isolated the homepage effect from seasonal lift.

🚩 **Red Flag**
"A/B testing is when you test two versions and pick the better one." — No mention of randomization, statistical validity, or causal inference.

⭐ **What earns the role**
Proactively mentioning SUTVA, or noting that "better" requires a pre-specified metric and statistical threshold — not eyeballing results.

---

## Day 2 — Hypothesis Formulation

🎯 **"Before running an A/B test, what do you define upfront and why?"**

💬 **Model Answer**
Before touching any data: (1) Null and alternative hypothesis, (2) Primary metric, (3) Guardrail metrics, (4) Significance level α, (5) Desired power, (6) MDE — minimum detectable effect, (7) Randomization unit, (8) Expected experiment duration. Defining these upfront prevents p-hacking and HARKing (Hypothesizing After Results are Known).

↳ **"Why do you need to define the metric before running the test, not after?"**
If you look at 20 metrics post-hoc and pick whichever moved, you exploit multiple comparisons — expected false positive rate explodes. The primary metric must be locked before data collection so statistical guarantees hold.

↳ **"A PM says 'let's just see what moves.' What do you do?"**
Educate: we can track many secondary metrics for learning, but we must pre-specify exactly one primary metric that drives the launch decision. Everything else is exploratory. I'd also document the hypothesis in our experiment system before enabling the flag.

🚩 **Red Flag**
Not mentioning MDE or treating metric selection as something you figure out after launch.

⭐ **What earns the role**
Mentioning pre-registration and distinguishing confirmatory (primary metric) from exploratory (secondary metrics) analysis — the same distinction used in clinical trials.

---

## Day 3 — Null and Alternative Hypothesis

🎯 **"What's the difference between H₀ and H₁ in an A/B test? Write them out for a CTR experiment."**

💬 **Model Answer**
- H₀: CTR_treatment = CTR_control (no difference)
- H₁: CTR_treatment ≠ CTR_control (two-tailed)

The null is the default assumption — we assume no effect until data convinces us otherwise. The alternative is what we believe if we reject H₀.

↳ **"When would you use a one-tailed test?"**
Only when it's truly impossible for the treatment to harm the metric — which is rare. One-tailed tests have more power but miss regressions. In practice, two-tailed is almost always correct because even "safe" changes can have unexpected negative effects.

↳ **"If p = 0.04 and α = 0.05, what do you conclude?"**
We reject H₀. The result is statistically significant. But I'd immediately ask: what's the effect size? What's the CI? Is the lift practically meaningful? Significance alone isn't a launch decision.

🚩 **Red Flag**
"One-tailed because we expect treatment to be better." — Expectation doesn't justify one-tailed; it just introduces bias toward false positives.

⭐ **What earns the role**
Noting that the choice of one- vs two-tailed must be made *before* seeing results, otherwise it's cherry-picking.

---

## Day 4 — p-value

🎯 **"What is a p-value? What does p = 0.02 tell you?"**

💬 **Model Answer**
The p-value is the probability of observing a result at least as extreme as what we saw, *assuming H₀ is true*. p = 0.02 means: if there were truly no difference between control and treatment, we'd see this large a difference only 2% of the time by chance. It does NOT mean there's a 98% chance treatment is better.

↳ **"What's the most common misinterpretation of p-values you've seen?"**
Three classics: (1) "p = 0.02 means 98% probability treatment works" — wrong, it's a statement about data under H₀, not probability of H₁. (2) "p > 0.05 means no effect" — no, it means insufficient evidence to reject H₀. (3) "smaller p = larger effect" — completely false, p conflates effect size with sample size.

↳ **"A stakeholder says 'p = 0.049, we're good to ship.' What do you check?"**
Effect size and CI — is the lift meaningful or trivially small? Guardrail metrics — are we harming anything? SRM — is randomization valid? How many metrics did we look at — did we correct for multiple comparisons? Duration — did we run long enough to rule out novelty?

🚩 **Red Flag**
"p-value is the probability that the null hypothesis is true." — Textbook misinterpretation that disqualifies a candidate.

⭐ **What earns the role**
Explaining that p-values are uniformly distributed under H₀ — so with many experiments you'll naturally get some below 0.05. That's why pre-registration and experiment governance matter.

---

## Day 5 — Type I and Type II Errors

🎯 **"Explain Type I and Type II errors. Which is worse in product experimentation?"**

💬 **Model Answer**
- **Type I (α):** False positive — we ship a feature that doesn't work. We rejected H₀ when it was true.
- **Type II (β):** False negative — we miss a real improvement. We failed to reject H₀ when H₁ was true.

Which is worse depends on context. Shipping a broken feature wastes engineering resources and may harm users. Missing a real improvement means slower product growth. In practice, Type I errors are more costly for irreversible decisions (billing changes, core UX rework). Type II errors are more costly when the change is cheap and the opportunity is large.

↳ **"How do you control each?"**
Type I: set α (e.g., 0.05). Type II: set desired power (1-β, e.g., 0.80) and ensure adequate sample size. They trade off — reducing α (fewer false positives) requires larger n to maintain power.

↳ **"Give me a scenario where you'd use α = 0.01."**
Changing pricing, removing a feature used by millions, medical or safety-critical changes. Anything where a false positive causes irreversible harm.

🚩 **Red Flag**
"Type II errors don't matter as much" — shows no understanding that missing real improvements is a real business cost.

⭐ **What earns the role**
Framing the α/power tradeoff as a business decision, not a statistics default — and connecting it to the cost asymmetry of being wrong in each direction.

---

## Day 6 — Statistical Power

🎯 **"What is statistical power and what factors affect it?"**

💬 **Model Answer**
Power = 1 - β = probability of detecting a true effect when one exists. Standard target: 80%. Four factors:
1. **Sample size** — more data → more power (strongest lever)
2. **Effect size (MDE)** — larger effect → easier to detect → higher power
3. **Significance level α** — higher α → more power but more false positives
4. **Variance** — lower metric variance → higher power

↳ **"Your experiment is underpowered. What are your options?"**
(1) Run longer to get more users. (2) Increase traffic allocation (e.g., 50/50 instead of 10/90). (3) Reduce metric variance via CUPED. (4) Choose a more sensitive metric. (5) Increase MDE — accept you can only detect larger effects. (6) Accept lower power — but document the risk.

↳ **"You ran a test, got p = 0.12, and the team wants to ship. What do you say?"**
p = 0.12 is not significant at α = 0.05. But I'd check: was the test adequately powered? If power was 20%, a null result is uninformative. Compute the confidence interval — if the CI excludes any meaningful negative effect (e.g., [−0.1%, +3%]), there may be enough evidence to ship conservatively. Don't just look at the p-value in isolation.

🚩 **Red Flag**
"80% power means we'll be right 80% of the time." — Conflates power with overall accuracy.

⭐ **What earns the role**
Noting that underpowered experiments have a high false negative rate AND a high false positive rate (among significant results) — the winner's curse / Type M error.

---

## Day 7 — Sample Size Calculation

🎯 **"Walk me through how you'd calculate the sample size for an A/B test on conversion rate."**

💬 **Model Answer**
Inputs needed:
- Baseline conversion rate (p₀) — from historical data, e.g., 10%
- MDE — e.g., detect a 1pp lift → p₁ = 11%
- α = 0.05 (two-tailed), so z_{α/2} = 1.96
- Power = 0.80, so z_β = 0.84

Formula per variant:
n = (z_{α/2} + z_β)² × [p₀(1-p₀) + p₁(1-p₁)] / (p₁ - p₀)²

For p₀=0.10, p₁=0.11: n ≈ 14,750 per variant → ~29,500 total.

↳ **"How does MDE affect sample size?"**
Inverse square relationship — halving the MDE quadruples the required sample. This is why detecting small effects is so expensive. A 0.1% lift needs ~100x more data than a 1% lift.

↳ **"A PM says 'we only have 1,000 users a day, we need results in a week.' What do you do?"**
7,000 users total. For 80% power at α=0.05, we can only detect very large effects (~5% absolute lift on a 10% baseline). Options: (1) Extend timeline, (2) Use a more sensitive metric, (3) Use CUPED, (4) Accept lower power with a documented risk, (5) Re-evaluate if the change is worth testing at this traffic level.

🚩 **Red Flag**
Not knowing the formula components or saying "just run until we have enough data" without specifying what "enough" means.

⭐ **What earns the role**
Proactively noting that sample size must account for both variants — many candidates compute n for one group only.

---

## Day 8 — Confidence Intervals

🎯 **"What is a confidence interval and how do you interpret it in an A/B test?"**

💬 **Model Answer**
A 95% CI is a range constructed such that if we repeated the experiment many times, 95% of those intervals would contain the true parameter. In an A/B test, if the CI for the lift is [1.2%, 4.8%], it means the true effect is plausibly between 1.2% and 4.8% — we're reasonably confident the treatment helps. If the CI includes 0, we cannot claim a significant effect.

↳ **"Is it correct to say 'there's a 95% probability the true value is in this interval'?"**
In frequentist statistics — no. The true value is fixed; it's either in the interval or not. The 95% refers to the long-run frequency of intervals containing the true value. A Bayesian credible interval does give a direct probability statement about the parameter.

↳ **"CI is [0.1%, 0.3%]. Statistically significant. Do you ship?"**
The effect is real but tiny. I'd ask: is a 0.1%–0.3% lift meaningful for the business? What's the engineering cost to maintain this feature? What do guardrail metrics show? Statistical significance ≠ practical significance.

🚩 **Red Flag**
"The CI contains the true value with 95% probability." — Classic frequentist/Bayesian confusion.

⭐ **What earns the role**
Noting that CIs are more informative than p-values — they show effect size and uncertainty simultaneously, not just a binary reject/fail-to-reject.

---

## Day 9 — Randomization

🎯 **"How do you assign users to control and treatment in an A/B test?"**

💬 **Model Answer**
Hash-based assignment: compute hash(user_id + experiment_id) mod 100 → assign to a bucket (0–99) → map buckets to variants. This ensures:
- Determinism: same user always sees same variant
- Uniform distribution: buckets are balanced
- Independence: different experiments use different salts (experiment_id) so assignments are uncorrelated across experiments

↳ **"Why do you include the experiment_id in the hash?"**
Without it, users assigned to bucket 42 in experiment A are the same users in bucket 42 in experiment B. This creates correlation between experiments, which is fine if experiments are in different layers but dangerous if they're not.

↳ **"What is the randomization unit and how do you choose it?"**
The unit can be user, session, device, request, or group. Choose the unit that: (1) matches how users experience the feature, (2) is stable enough to avoid re-assignment, (3) avoids SUTVA violations. User-level is most common. Session-level makes sense for anonymous traffic. Group/cluster-level for features with network effects.

🚩 **Red Flag**
"We use random() at request time." — Creates inconsistent experiences (user sees control on one page load, treatment on another), pollutes the experiment.

⭐ **What earns the role**
Discussing the tradeoff: user-level ensures consistency but reduces effective sample size for low-engagement users. Session-level increases sample size but risks contamination for logged-in users.

---

## Day 10 — A/A Testing

🎯 **"What is an A/A test and when would you run one?"**

💬 **Model Answer**
An A/A test splits users into two groups that both receive identical experiences. Expected result: no statistically significant difference on any metric. Used to:
1. Validate that the randomization system works correctly
2. Detect instrumentation bugs (tracking firing differently per group)
3. Estimate the false positive rate of the testing infrastructure
4. Calibrate variance estimates for sample size calculations
5. Check for pre-existing user imbalances (e.g., treatment bucket has historically higher-value users)

↳ **"Your A/A test shows p = 0.02 on session duration. What does that mean?"**
Something is wrong. Possible causes: biased hash function, bot traffic filtered asymmetrically, logging differences per group, a different experiment leaking into one bucket. Do not run any A/B tests until this is resolved — your p-values are untrustworthy.

↳ **"How often should you run A/A tests?"**
When setting up a new experiment system, when the system is modified, when a new metric or logging pipeline is added, or whenever a series of A/B tests shows unexpectedly high win rates (could signal systematic false positives).

🚩 **Red Flag**
"A/A tests are a waste — we know they'll show no difference." — Misses the entire point of validation.

⭐ **What earns the role**
Running repeated A/A tests and checking that the distribution of p-values is uniform (U[0,1]) — if it's skewed toward small values, the system is producing too many false positives.

---

## Day 11 — Metric Selection

🎯 **"How do you choose the primary metric for an A/B test?"**

💬 **Model Answer**
A good primary metric is: (1) **Sensitive** — moves in response to the treatment. (2) **Aligned** — correlated with long-term business value. (3) **Fast-moving** — responds within the experiment window. (4) **Measurable** — observable and attributable. (5) **Resistant to gaming** — can't be trivially inflated without real value creation.

Example: for a new search ranking feature, "click-through rate on top result" is sensitive but gameable (show a clickbait title). "Long-click rate" (user clicks and doesn't return quickly) is better — proxies satisfaction.

↳ **"What's the difference between a primary metric and a guardrail metric?"**
Primary drives the launch decision — it's the one metric you pre-powered for. Guardrail metrics are monitors: if they degrade significantly, we don't ship regardless of the primary metric result. Example: testing a new ad format — primary = ad revenue, guardrail = page load time and organic CTR.

↳ **"Primary metric improved 3%. Guardrail metric degraded 1% (not significant). Do you ship?"**
Depends. If the guardrail degradation is not significant *and* the CI excludes practically meaningful harm, I'd lean toward shipping with monitoring. But if 1% degradation would matter at scale (e.g., retention), I'd iterate to fix the guardrail before launching.

🚩 **Red Flag**
Choosing the metric that moved the most after seeing results. That's p-hacking.

⭐ **What earns the role**
Mentioning the concept of "surrogate metrics" — short-term proxies for long-term outcomes — and the risk that optimizing a surrogate can diverge from the true goal (Goodhart's Law).

---

## Day 12 — Guardrail Metrics

🎯 **"Walk me through how you'd set up guardrail metrics for an experiment on a new checkout flow."**

💬 **Model Answer**
Primary metric: conversion rate (% of users who complete purchase).
Guardrails I'd set up:
- Page load time — a slower checkout would hurt conversion long-term
- Error rate / crash rate — must not increase
- Cart abandonment rate — a leading indicator of checkout friction
- Customer support tickets related to checkout
- Return/refund rate — a fast but bad checkout experience

For each guardrail: define a threshold (e.g., page load ≤ 200ms increase) and a significance level (usually stricter: α = 0.01). Flag but don't auto-reject — some tradeoffs are worth it.

↳ **"Conversion rate up 5%, but page load increased 300ms. Do you ship?"**
No, not yet. 300ms is a significant regression — research shows latency directly affects conversion and retention. I'd investigate root cause, fix the performance issue, then re-test. Never ship a performance regression for a conversion gain — it erodes over time as users develop negative associations.

↳ **"How many guardrail metrics is too many?"**
If you have 50 guardrails, you'll almost certainly see one degrade by chance. Too many also makes decision-making slow. Pick 5–10 that represent distinct risk dimensions. Apply appropriate multiple testing correction across guardrails.

🚩 **Red Flag**
"We only track the primary metric." — Missing the entire point of holistic experiment evaluation.

⭐ **What earns the role**
Distinguishing between **hard guardrails** (must not degrade — e.g., crash rate) and **soft guardrails** (monitor and flag — e.g., session length) with different decision thresholds.

---

## Day 13 — Duration

🎯 **"How do you decide how long to run an A/B test?"**

💬 **Model Answer**
Duration = ceil(required_sample_size / daily_traffic_per_variant). But also: run for at least one full week to capture day-of-week effects (weekday vs weekend behavior differs significantly for most products). For consumer apps, 2 weeks is standard. For B2B with weekly work cycles, longer.

Upper bound: don't run indefinitely. After ~4–6 weeks, user composition changes (new users enter who weren't present at start), seasonal effects drift in, and carryover from the experiment itself can confound results.

↳ **"Why must duration be fixed upfront, not adjusted mid-experiment?"**
Stopping based on observed results is optional stopping — it inflates Type I error. If you stop every time p dips below 0.05, you'll see that happen by chance in ~22% of null experiments (not 5%). Duration must be pre-committed.

↳ **"The test reached significance on Day 3. Can you stop?"**
Not with standard frequentist tests. The p-value naturally fluctuates — being below 0.05 at Day 3 doesn't mean it will stay there. You'd need a sequential testing method (like SPRT or always-valid p-values) to stop early with Type I error control.

🚩 **Red Flag**
"I'd stop when we hit significance." — This is the most common experimentation mistake in industry.

⭐ **What earns the role**
Explaining the peeking problem mathematically: if you peek daily and stop at first significance, the effective Type I error rate is ~22–30%, not 5%.

---

## Day 14 — Peeking Problem

🎯 **"What is the peeking problem and how do you solve it?"**

💬 **Model Answer**
Peeking means checking results while the experiment is running and making decisions based on interim results. The p-value fluctuates across time — with repeated looks, the probability of ever crossing α = 0.05 grows far beyond 5% even when H₀ is true.

Example: with daily peeking over 20 days, the effective false positive rate is ~30% at α = 0.05.

Solutions:
1. **Fixed-horizon test:** Commit to a sample size, look only once at the end.
2. **Group sequential testing:** Pre-planned interim looks with adjusted α thresholds (alpha spending functions — O'Brien-Fleming, Pocock).
3. **Sequential probability ratio test (SPRT):** Continuously valid, allows stopping at any time.
4. **Always-valid p-values / e-values:** Modern approach used at Netflix, Booking.com.

↳ **"Your company wants a live dashboard so PMs can monitor experiments daily. How do you design it?"**
Show the dashboard but use always-valid confidence sequences instead of fixed-horizon CIs. Display a "do not call" indicator until the pre-specified end date. Educate stakeholders: "this is for monitoring anomalies (crashes, SRM), not for making launch decisions mid-flight."

🚩 **Red Flag**
"Peeking is fine if the effect is large." — Effect size doesn't change the statistical validity of early stopping.

⭐ **What earns the role**
Knowing that always-valid p-values (based on e-values or martingale theory) provide anytime-valid inference — the CI is valid regardless of when you look.

---

## Day 15 — Sample Ratio Mismatch (SRM)

🎯 **"What is SRM and what do you do when you detect it?"**

💬 **Model Answer**
SRM (Sample Ratio Mismatch) is when the observed ratio of users in control vs treatment doesn't match the intended ratio. Example: you expect 50/50 but observe 48,000 control / 52,000 treatment. Detect with a chi-squared test on observed vs expected counts.

When SRM is detected: **stop all analysis immediately**. The experiment is invalid — selection bias is present, meaning the groups are no longer comparable. The statistical results cannot be trusted.

↳ **"What causes SRM?"**
- Bots filtered differently per variant
- Client-side crashes more common in one variant (users drop out before logging)
- Redirect differences (treatment has an extra redirect, losing some users)
- CDN caching serving control to users before assignment
- Logging failures in one variant
- Variant assignment and logging happening at different code paths

↳ **"How do you debug SRM?"**
Stratify by: device type, OS, geography, acquisition channel, new vs returning users. Look for where the imbalance concentrates — that's usually the root cause. Also check the time series of group sizes — when did the imbalance start?

🚩 **Red Flag**
"SRM means we should just reweight the groups." — Reweighting doesn't fix selection bias; it's not a valid correction.

⭐ **What earns the role**
Building SRM checks into automated pre-analysis pipelines so experiments with SRM never surface results to analysts in the first place.

---

## Day 16 — Two-Proportion Z-Test

🎯 **"Walk me through the statistical test for comparing two conversion rates."**

💬 **Model Answer**
Use the two-proportion z-test.

Let p̂_c = conversions_control / n_control, p̂_t = conversions_treatment / n_treatment.
Pooled proportion: p̂ = (conversions_c + conversions_t) / (n_c + n_t)
Standard error: SE = sqrt(p̂(1-p̂)(1/n_c + 1/n_t))
Z-statistic: z = (p̂_t - p̂_c) / SE
Compare to z_{α/2} = 1.96 for α = 0.05 two-tailed.

Assumptions: independence (hash-based assignment), large enough n (np > 5 for both groups), observations are Bernoulli (0/1 per user).

↳ **"What if the metric isn't binary — like revenue per user?"**
Use Welch's t-test (two-sample t-test with unequal variances). Revenue is continuous and typically right-skewed, but CLT kicks in for large n. For small samples or extreme skew, use bootstrap or Mann-Whitney U.

↳ **"When would the z-test give wrong results for a proportion?"**
When analysis unit ≠ randomization unit — e.g., session-level analysis when assignment is at user level. Sessions from the same user are correlated, so the standard error is underestimated, inflating significance. Use the delta method or cluster-robust SEs.

🚩 **Red Flag**
Using a chi-squared test without knowing it's equivalent to the z-test for 2x2 tables.

⭐ **What earns the role**
Flagging the unit-of-analysis problem without prompting — this is one of the most common bugs in industry experimentation.

---

## Day 17 — Central Limit Theorem

🎯 **"Why can we use normal-based tests for revenue, which is heavily skewed?"**

💬 **Model Answer**
The Central Limit Theorem: the sampling distribution of the sample mean approaches a normal distribution as n grows, regardless of the underlying distribution. So even though individual revenue values are right-skewed (most users spend $0, a few spend thousands), the *mean* revenue across n users is approximately normally distributed for large n.

How large? For mild skew: n > 30 is often sufficient. For heavy skew (revenue, streaming hours): n > 500–1000 is safer. Always validate empirically via bootstrap.

↳ **"What if n is small — say, you're testing on a niche B2B product with 200 users?"**
CLT doesn't save you. Options: (1) Bootstrap (resample empirically), (2) Permutation test (distribution-free), (3) Mann-Whitney U (rank-based, non-parametric), (4) Bayesian methods with appropriate priors.

↳ **"How do you check whether n is large enough for CLT to apply?"**
Simulate: bootstrap the distribution of the sample mean and check if it looks normal. Or look at skewness of the raw metric — high skewness requires larger n.

🚩 **Red Flag**
"CLT means the data is normally distributed." — No, CLT applies to the sampling distribution of the mean, not the raw data.

⭐ **What earns the role**
Mentioning that high-skew metrics like revenue are better analyzed after outlier capping or log-transformation, or via CUPED which also reduces variance.

---

## Day 18 — Multiple Testing

🎯 **"You're running an experiment and tracking 30 metrics. How do you handle the multiple testing problem?"**

💬 **Model Answer**
With 30 independent tests at α = 0.05, we expect ~1.5 false positives even if nothing works. Solutions depend on what we're controlling:

- **FWER (Familywise Error Rate):** Bonferroni (divide α by 30 → α' = 0.0017). Very conservative. Holm-Bonferroni is less conservative and uniformly more powerful.
- **FDR (False Discovery Rate):** Benjamini-Hochberg — controls the expected proportion of false positives among significant results. Better when running many metrics and some false positives are tolerable.

In practice: pre-specify the primary metric (no correction needed for one), apply Holm or BH to secondary metrics, treat everything else as exploratory.

↳ **"Why is Bonferroni too conservative for 30 metrics?"**
It assumes all tests are independent — if metrics are correlated (e.g., clicks and sessions), Bonferroni over-penalizes. It also drives α so low that real effects become undetectable, requiring far more data.

↳ **"Your experiment shows 2 significant metrics out of 30. How do you report this?"**
With caution. Apply FDR correction. Report the adjusted p-values. Note that at 30 tests, 1–2 false positives are expected. Treat these as hypothesis-generating, not confirmatory, unless they were pre-specified primary/secondary metrics.

🚩 **Red Flag**
"Each metric is independent so there's no problem." — Metrics in the same experiment are rarely independent.

⭐ **What earns the role**
Distinguishing confirmatory tests (pre-specified, needs correction) from exploratory tests (all post-hoc — generate hypotheses, don't make decisions).

---

## Day 19 — CUPED

🎯 **"What is CUPED and why does it matter?"**

💬 **Model Answer**
CUPED (Controlled-experiment Using Pre-Experiment Data) reduces metric variance by removing the component explained by a pre-experiment covariate. Lower variance → higher power → smaller required sample size.

Implementation:
1. Collect pre-experiment metric X (e.g., revenue in the prior 2 weeks)
2. Compute θ = Cov(Y, X) / Var(X)
3. Compute Y_cuped = Y - θ(X - E[X])
4. Run t-test on Y_cuped

This doesn't introduce bias because θ is estimated on pre-experiment data and the adjustment is applied equally to both groups.

↳ **"How much variance reduction can CUPED achieve?"**
Depends on the correlation between pre- and post-experiment metric. If ρ = 0.8, variance reduces by 1 - ρ² = 36%. If ρ = 0.9, variance reduces by 19%. In practice, 30–50% variance reduction is common, which roughly halves required sample size.

↳ **"What's the best covariate to use for CUPED?"**
The same metric as the outcome, measured in the pre-experiment period. Revenue → prior revenue. Sessions → prior sessions. If that's not available, use a strong correlated proxy (e.g., historical page views for a new-user metric).

🚩 **Red Flag**
"CUPED is just linear regression." — Partially true, but missing the key insight: the covariate is measured *before* the experiment starts, guaranteeing no treatment contamination.

⭐ **What earns the role**
Noting that CUPED and regression adjustment are equivalent when the covariate is orthogonal to treatment assignment — which is guaranteed by randomization.

---

## Day 20 — Novelty Effect

🎯 **"Your experiment shows a huge lift in the first week, which decays to zero by week 3. What's happening?"**

💬 **Model Answer**
Classic novelty effect: users engage more with anything new, not because it's better, but because it's unfamiliar. The lift is real but not durable — once the novelty wears off, behavior returns to baseline.

This is a major risk when launch decisions are made on short experiments.

↳ **"How do you detect novelty vs a real effect?"**
Plot the treatment effect over time (rolling weekly). If it starts high and monotonically decreases, suspect novelty. If it's stable or increases over time, the effect is likely genuine. Stratify by new vs returning users — novelty only affects users who are familiar with the old experience.

↳ **"How do you design experiments to mitigate novelty?"**
Run for at least 2–4 weeks (longer for habit-forming features). Use holdout groups to measure long-term effects post-launch. Focus on new users as a segment where novelty is not a factor (they have no prior experience to compare to).

🚩 **Red Flag**
"The first week data is most accurate because users engage most." — Completely backwards.

⭐ **What earns the role**
Distinguishing novelty (positive inflation that decays) from primacy (negative initial resistance that recovers) — they require opposite interpretations and different mitigation strategies.

---

## Day 21 — Primacy Effect

🎯 **"What is the primacy effect and how does it differ from the novelty effect?"**

💬 **Model Answer**
Primacy effect: existing users resist change, performing worse initially with a new UX because muscle memory and habits are disrupted. The effect starts negative and recovers as users adapt. The long-term effect may be positive even though the short-term A/B result looks negative.

Novelty: new = exciting → inflated short-term metrics that decay.
Primacy: new = different → depressed short-term metrics that recover.

↳ **"How do you make a launch decision when you suspect primacy?"**
You can't run the test long enough for full adaptation — that could take months. Instead: (1) Analyze new users separately (no primacy for them). (2) Look at the trend in returning users — is the effect recovering? (3) Use qualitative research to understand if the negative is discomfort or genuine dissatisfaction. (4) Consider a phased rollout with extended monitoring.

↳ **"Give me a real product example of primacy effect."**
Any major navigation redesign — Facebook's 2009 redesign, Twitter's layout changes. Users petition to revert, but metrics improve long-term as users adapt to a more efficient layout.

🚩 **Red Flag**
Treating a recovering negative trend as proof the feature is "getting better over time" without understanding *why*.

⭐ **What earns the role**
Proposing to use new user cohorts as a "primacy-free" proxy for long-term returning user behavior.

---

## Day 22 — Selection Bias

🎯 **"What is selection bias in the context of A/B testing and how do you prevent it?"**

💬 **Model Answer**
Selection bias occurs when the groups being compared differ in ways unrelated to the treatment — typically because assignment wasn't truly random, or because post-assignment behavior differentially filters users from the analysis.

Prevention: (1) Use proper hash-based randomization. (2) Run SRM checks. (3) Compare pre-experiment covariates between groups (balance check). (4) Don't filter users post-assignment based on behavior that could be affected by treatment (e.g., analyzing only "active users" when activity itself is affected by the treatment).

↳ **"Give me an example of post-assignment selection bias."**
Testing a new onboarding flow. You filter analysis to "users who completed step 3." But the treatment changes how many users reach step 3. The users who complete step 3 in treatment vs control are different populations — comparing them is biased. Analyze all users assigned to treatment, regardless of step completion.

↳ **"What is survivor bias in experiments?"**
Analyzing only users who survived some process (e.g., made a purchase, didn't churn). Treatment may affect survival, so survivors in each group are not comparable. Solution: intent-to-treat analysis on all assigned users.

🚩 **Red Flag**
"We only analyze engaged users to reduce noise." — This introduces selection bias worse than the noise it removes.

⭐ **What earns the role**
Framing intent-to-treat (ITT) as the default: always analyze all users assigned to treatment, not just those who received or engaged with it.

---

## Day 23 — Network Effects & SUTVA

🎯 **"You're at a social network. A user is assigned to treatment and sees a new sharing feature. Does this affect other users?"**

💬 **Model Answer**
Yes — this violates SUTVA (Stable Unit Treatment Value Assumption), which requires no interference between units. If treatment users share more, their friends (who may be in control) are exposed to more shares, changing their behavior too. This contaminates the control group, making the treatment effect estimate biased.

↳ **"How do you run experiments that handle social interference?"**
(1) **Ego-network randomization:** Assign entire social clusters to treatment or control — users in the same cluster see the same variant. (2) **Geographic randomization:** Assign at city/DMA level. (3) **Time-based isolation:** Not practical for social. (4) **Measure spillover explicitly** via users at the boundary of treated/untreated clusters.

↳ **"How do you estimate the true network effect if spillover is present?"**
Compare cluster-level outcomes: average metric in fully-treated clusters vs fully-untreated clusters. The difference captures both direct effect and network amplification. You can also model spillover explicitly using the fraction of treated neighbors as a covariate.

🚩 **Red Flag**
"Social features don't need special treatment — just randomize at user level." — This is how you get biased results and wrong product decisions.

⭐ **What earns the role**
Noting that SUTVA violations almost always **underestimate** the true treatment effect — treatment users are compared to contaminated control users who are also benefiting from network spillover.

---

## Day 24 — Two-Sided Marketplace

🎯 **"How do you run A/B tests at Airbnb or Uber where there are buyers and sellers on the same platform?"**

💬 **Model Answer**
User-level randomization creates interference: if you treat 50% of riders with a new matching algorithm, drivers experience a mix of treatment and control demand — their behavior changes regardless of their own assignment. The groups are no longer isolated.

Solutions:
1. **Geo-based randomization:** Assign markets (cities) to treatment or control. Supply and demand within a city are exposed to the same variant.
2. **Switchback experiments:** Alternate treatment and control over time windows within the same market.
3. **Supply-side holdout:** Hold back a fraction of supply from the feature.
4. **Quasi-experimental methods:** Difference-in-differences at market level.

↳ **"What are the tradeoffs of geo-based experiments?"**
Fewer randomization units (cities vs users) → much lower statistical power. Cities are heterogeneous (NYC ≠ rural town) → need matched pairs or regression adjustment. Spillover across city borders. Time and cost of running city-level experiments.

↳ **"How does Uber run switchback experiments?"**
Alternate between treatment and control in 30-minute or 1-hour windows within the same market. Drivers experience both conditions sequentially. Requires careful modeling to remove time-of-day effects and carryover.

🚩 **Red Flag**
"We randomize riders and just accept some noise." — The noise isn't random; it's systematic bias.

⭐ **What earns the role**
Knowing that geo experiments require a pre-treatment period to validate parallel trends, and that synthetic control methods can substitute for a matched control market.

---

## Day 25 — Interleaving

🎯 **"What is interleaving and why is it more sensitive than A/B testing for ranking experiments?"**

💬 **Model Answer**
Interleaving mixes results from two algorithms on the same result page for the same user. Example: for a search query, algorithm A ranks results [R1, R3, R5...] and algorithm B ranks [R2, R4, R6...]. The page shows a blend. User clicks reveal implicit preference between A and B.

Why more sensitive: within-user comparison eliminates user-level variance (a major noise source). The same user's click behavior directly compares algorithms, so effect sizes are detectable with ~100x less traffic than A/B tests.

↳ **"What is balanced interleaving?"**
Alternate which algorithm gets to place the top result first (to control for position bias). Otherwise, whichever algorithm always places result at position 1 gets a structural advantage.

↳ **"What can't interleaving measure?"**
Long-term effects (novelty/primacy), effects on non-search behavior (retention, revenue), absolute metric values. Interleaving tells you "A > B" but not by how much or what the business impact is. Use interleaving for signal, A/B for sizing.

🚩 **Red Flag**
"Interleaving is just showing users both versions." — Missing the statistical mechanism of within-user comparison.

⭐ **What earns the role**
Knowing that interleaving is used at Google, Bing, Netflix, and Spotify precisely because it dramatically reduces the traffic cost of ranking experiments.

---

## Day 26 — Dilution / Intent-to-Treat

🎯 **"Only 30% of treatment users actually triggered the new feature. How does this affect your analysis?"**

💬 **Model Answer**
This is the dilution problem. The remaining 70% of treatment users had the same experience as control — they dilute the treatment effect. The observed effect in treatment vs control is smaller than the true effect of the feature on users who actually experienced it.

Two analysis approaches:
1. **Intent-to-Treat (ITT):** Analyze all assigned users regardless of exposure. Unbiased but underestimates the feature's true effect. Conservative and valid.
2. **Complier Average Causal Effect (CACE) / IV:** Estimate the effect only on users who actually received treatment, using assignment as an instrument. Unbiased if instrument is valid.

↳ **"When would you use CACE over ITT?"**
When you want to understand the true effect of the feature itself (for feature improvement), not the effect of rolling it out to everyone (for launch decision). For launch decisions, ITT is appropriate because at launch, not everyone will trigger the feature either.

↳ **"How do you increase treatment take-up rate?"**
Ensure the feature is shown early in the session, lower the trigger threshold, improve discoverability. High take-up rates reduce dilution and improve experiment sensitivity.

🚩 **Red Flag**
"Analyze only treatment users who triggered the feature." — This creates post-assignment selection bias (triggering is itself affected by the treatment).

⭐ **What earns the role**
Connecting ITT/CACE to the concept of treatment compliance — a concept borrowed directly from clinical trial methodology.

---

## Day 27 — Carryover Effects

🎯 **"You're running a sequence of experiments on the same users. What is carryover and how do you handle it?"**

💬 **Model Answer**
Carryover (or spillover over time) is when the effect of experiment A persists and contaminates experiment B, even after experiment A ends. Common in: personalization systems (models trained on treatment data), recommendation history, learned user habits, email preferences.

Solutions:
1. **Washout period:** Wait between experiments for effects to dissipate (usually 1–2 weeks, but may be longer for personalization).
2. **Holdout groups:** Permanent control groups not exposed to any experiments.
3. **New user cohorts:** Carryover doesn't affect users who weren't in the prior experiment.
4. **Careful bucketing:** Ensure experiment buckets are mutually exclusive over time.

↳ **"How long of a washout period do you need?"**
Depends on the mechanism. For UI changes: 1–2 weeks. For recommendations/ML models: longer — the model may have been retrained on biased data. For email campaigns: at least one send cycle. There's no universal rule — measure decay of the effect over time.

↳ **"Is carryover always a problem?"**
No — if you're measuring the steady-state effect of a feature (which includes any carryover), and you run experiments long enough to reach steady state, carryover is baked in. The problem is when carryover from experiment A biases the measurement of experiment B.

🚩 **Red Flag**
"We just start the next experiment immediately after the previous one ends." — Very common in practice; almost always wrong.

⭐ **What earns the role**
Knowing that personalization and ML systems are especially susceptible to carryover because the model itself is the mechanism — switching off the experiment doesn't undo the model's learned behavior.

---

## Day 28 — Segmentation & HTE

🎯 **"Your experiment shows no overall effect, but you suspect the feature helps mobile users. What do you do?"**

💬 **Model Answer**
This is a subgroup / heterogeneous treatment effect (HTE) analysis. If mobile was not a pre-specified segment, any finding is exploratory — not confirmatory. Steps:

1. Run the interaction test: outcome ~ treatment + device + treatment×device. A significant interaction coefficient means the effect differs between mobile and desktop.
2. If significant: report as exploratory, replicate with a dedicated mobile-only experiment.
3. Correct for multiple comparisons if you're testing many segments.
4. Do not launch based on a post-hoc segment finding without replication.

↳ **"What's the risk of looking at many segments?"**
With 10 segments at α=0.05, you expect ~0.5 false positive segment findings even with no true heterogeneity. This is why pre-registration of key segments is important.

↳ **"How do large companies like Meta handle HTE at scale?"**
Using machine learning methods like causal forests, X-learner, or T-learner to estimate individualized treatment effects. These methods identify which user features predict HTE without requiring pre-specified segments.

🚩 **Red Flag**
"We found a significant effect in mobile users — we'll launch for mobile." Without pre-registration or replication — this is p-hacking.

⭐ **What earns the role**
Understanding that HTE analyses are valuable for *learning and personalization* even when the overall test is null — the information about who benefits drives future targeting.

---

## Day 29 — Regression to the Mean

🎯 **"You target users who had their worst week ever and show them a new feature. Results look amazing. Is the feature really working?"**

💬 **Model Answer**
Not necessarily. This is regression to the mean. Users who had an extreme low (worst week ever) will, on average, perform better the following week purely due to random variation — regardless of any treatment. The improvement you observe conflates the feature's effect with this natural recovery.

↳ **"How do you avoid this in experiment design?"**
Never target a group based on an extreme recent outcome. If you must target based on behavior, use older historical data (to reduce recency bias) or use randomization within the target group and compare treated vs untreated targeted users.

↳ **"Where does regression to the mean appear most in industry?"**
- Targeting churned or at-risk users (they were extreme lows; some would have recovered anyway)
- Targeting high-value users who had a bad month
- "Intervention" programs for low-performing employees or products

🚩 **Red Flag**
Not knowing the concept exists at all. Very common gap.

⭐ **What earns the role**
Connecting to the design fix: even within a targeted group, randomize — compare treated high-risk users vs untreated high-risk users. Regression to the mean affects both equally, canceling out.

---

## Day 30 — Bayesian A/B Testing

🎯 **"Walk me through Bayesian A/B testing. How does it differ from frequentist?"**

💬 **Model Answer**
Frequentist: we compute a p-value under H₀, make a binary reject/fail-to-reject decision at a fixed α. No probability statement about whether treatment is better.

Bayesian: we start with prior beliefs about the conversion rates (e.g., Beta(α, β) prior), update with observed data to get a posterior distribution, and report:
- P(treatment > control) — intuitive probability statement
- Expected loss from choosing incorrectly
- Posterior distribution over effect size

↳ **"What are the practical advantages of Bayesian testing?"**
(1) No fixed sample size — stop when posterior is conclusive. (2) Intuitive output for stakeholders: "87% probability treatment is better." (3) Naturally handles multiple comparisons within the Bayesian framework. (4) Incorporates prior knowledge from similar past experiments.

↳ **"What are the weaknesses?"**
(1) Sensitive to prior choice — informative priors can dominate with small samples. (2) Less standardized across organizations. (3) "No fixed sample size" can be misused — without discipline, it's just continuous peeking under another name. (4) Harder to audit and explain to regulators.

🚩 **Red Flag**
"Bayesian is just better than frequentist." — Both are valid frameworks with different assumptions and properties. The choice depends on context.

⭐ **What earns the role**
Knowing that Bayesian credible intervals (e.g., "95% probability the lift is between 1% and 3%") are what most practitioners *think* frequentist CIs mean — Bayesian framing is more intuitive but requires a valid prior.

---

## Days 31–50: L4 Continued — Applied & Scenario-Based

---

## Day 31 — Decision Framework

🎯 **"An A/B test shows: primary metric +2% (significant), guardrail metric −1% (not significant), CI = [−2.5%, +0.5%]. Do you ship?"**

💬 **Model Answer**
Do not ship yet. The guardrail CI includes −2.5% — a 2.5% degradation is practically significant and the data is consistent with it. "Not significant" ≠ "no harm." The CI tells us we can't rule out meaningful harm.

Decision: investigate the guardrail degradation. Run a larger experiment to narrow the CI. Or redesign the feature to protect the guardrail metric.

↳ **"What if the PM says the −1% on the guardrail is noise?"**
Show the CI. The question isn't whether the −1% point estimate is noise — it's whether we can confidently rule out a −2.5% degradation at scale. We can't. Data shows ambiguity; the right call is to resolve the ambiguity before shipping.

🚩 **Red Flag**
"Not significant means no effect — ship it." — Statistical non-significance does not equal practical safety.

⭐ **What earns the role**
Reframing: absence of evidence is not evidence of absence. CI width is the key information, not just the p-value.

---

## Day 32 — Metric Sensitivity

🎯 **"Your experiment ran for 4 weeks and shows no significant effect on revenue. How do you diagnose this?"**

💬 **Model Answer**
Three possibilities: (1) No true effect exists. (2) Effect exists but experiment is underpowered. (3) Revenue is too noisy a metric for this feature.

Diagnosis:
- Check power: what effect size was the test powered to detect? If MDE was 10% but you can only realistically expect 1%, you were always going to fail.
- Check CI width: if CI is [−15%, +18%], the experiment is massively underpowered. If CI is [−1%, +2%], you've effectively ruled out meaningful effects.
- Look at intermediate metrics: did clicks, sessions, or add-to-cart change? Revenue may be too downstream to see quickly.
- Check metric variance: revenue has huge variance due to outliers. Cap outliers, log-transform, or use CUPED.

↳ **"How would you redesign this experiment to detect a revenue effect?"**
Use CUPED with prior revenue as covariate. Cap at 99th percentile. Use a more sensitive proxy metric (e.g., items viewed, add-to-cart rate) that predicts revenue. Increase sample size.

🚩 **Red Flag**
"No significant effect means the feature doesn't work." — Could just mean wrong metric or underpowered test.

⭐ **What earns the role**
Distinguishing "this experiment proved no effect" (CI is narrow around zero) from "this experiment was unable to detect an effect" (CI is wide).

---

## Day 33 — Outliers

🎯 **"You're analyzing average order value. A few users spent $50,000 each. What do you do?"**

💬 **Model Answer**
High-value outliers inflate the mean and variance, making it very difficult to detect real treatment effects in a standard t-test. Options:

1. **Winsorize/cap at percentile** (e.g., 99th percentile). Replace all values above the cap with the cap value. Reduces variance without removing observations.
2. **Log-transform** the metric. Makes the distribution more symmetric. Effect measured on log scale = relative/multiplicative effect.
3. **Median or quantile regression** — robust to outliers by construction.
4. **Bootstrap** — handles non-normality without parametric assumptions.
5. **Separate analysis** for whale users — they may need different features anyway.

↳ **"Is capping valid? Doesn't it introduce bias?"**
It introduces a small bias in the point estimate (you're measuring something slightly different) but dramatically reduces variance. The bias-variance tradeoff usually favors capping. The key: apply the same cap to control and treatment, and pre-specify the cap level before seeing results.

🚩 **Red Flag**
"Remove outliers." — Removing outliers (deleting observations) is worse than capping — it reduces sample size and may differentially affect groups.

⭐ **What earns the role**
Mentioning that the cap value should be pre-specified based on the historical distribution (e.g., 99th percentile over the past 90 days), not chosen after seeing the experiment data.

---

## Day 34 — Ratio Metrics

🎯 **"How do you run a t-test on a ratio metric like sessions per user?"**

💬 **Model Answer**
Ratio metrics (clicks/sessions, revenue/user, sessions/day) require special handling because their variance isn't simply the variance of the numerator or denominator — it involves both.

Two approaches:
1. **Delta method:** Analytically approximates the variance of a ratio using first-order Taylor expansion. Gives closed-form SE for the ratio metric. Used at LinkedIn, Google.
2. **Bootstrap:** Resample users with replacement, compute the ratio each time, build empirical distribution of the ratio. More accurate for small samples.

Naive approach (just run t-test on per-user ratios) works if each user has a ratio value — but be careful about users with zero denominator.

↳ **"Why can't you just average sessions per user across the two groups?"**
You can — if every user has at least one session. The issue is zero-denominator users. More subtly, the ratio of averages ≠ average of ratios. For population-level statements (overall CTR), use ratio of sums (total clicks / total sessions). For user-level statements, use average of per-user ratios.

🚩 **Red Flag**
Running a standard t-test on session count without accounting for exposure differences between users.

⭐ **What earns the role**
Knowing the difference between ratio of averages and average of ratios, and when each is appropriate.

---

## Day 35 — Long-Term Effects

🎯 **"A 2-week experiment shows a 5% retention improvement. How confident are you this will hold post-launch?"**

💬 **Model Answer**
Not very confident without more evidence. Short-term experiments suffer from: (1) Novelty effects — lift may be transient. (2) Incomplete learning — users haven't fully adapted their behavior. (3) Survivorship bias in the measurement window. (4) Seasonal effects.

To measure long-term: (1) Run a holdout group post-launch — keep a small % of users (1–5%) on the old experience for 2–3 months. (2) Use leading indicators that predict long-term retention. (3) Analyze by user cohorts — does the effect persist for cohorts as they age?

↳ **"What is a holdback experiment?"**
A long-running holdout where a small fraction of users never sees the launched feature. Measures cumulative impact over months. Used at Facebook, Google for major features.

↳ **"How do you handle the ethical concern of holdback experiments?"**
If you believe the feature genuinely helps users, keeping 1% from it indefinitely is ethically uncomfortable. Common practice: limit holdback to 3–6 months, focus on learning-value, and sunset the holdback once the learning is complete.

🚩 **Red Flag**
"2-week A/B result is sufficient for long-term claims." — Almost never true for behavior-change features.

⭐ **What earns the role**
Quantifying the risk: "If novelty accounts for 50% of the observed lift, our true steady-state effect may be only 2.5% — which might still be worth launching, but we should size the expected impact correctly for roadmap planning."

---

## Day 36 — Experiment Governance

🎯 **"How do you prevent p-hacking and HARKing in an experimentation culture?"**

💬 **Model Answer**
Systematic solutions, not just trust:
1. **Pre-registration:** Experiment system requires declaration of primary metric, α, power, and expected duration before enabling the flag. Can't be changed after launch.
2. **Automated analysis:** System runs the pre-specified analysis — analysts can't cherry-pick the test.
3. **Audit trail:** All experiment decisions logged with timestamps.
4. **Peer review:** Experiment design reviewed before launch, results reviewed before decision.
5. **Null result culture:** Celebrate null results — learning that something doesn't work is valuable. Punishing nulls incentivizes p-hacking.

↳ **"A senior PM overrules a null result and ships anyway. What do you do?"**
Document your recommendation and the statistical evidence clearly. Escalate to the experimentation team lead. Propose a re-test after launch with a holdback to validate the decision. This is a cultural problem — individual pushback is insufficient; governance systems must enforce standards.

🚩 **Red Flag**
"We trust our analysts not to p-hack." — Trust is not a system.

⭐ **What earns the role**
Noting that the file drawer problem (null results not reported) creates organizational overconfidence — teams think their hit rate is 70% when it's actually 30%, because failures aren't tracked.

---

## Day 37 — Cookie/ID Issues

🎯 **"A user deletes their cookies mid-experiment. What happens?"**

💬 **Model Answer**
The user gets a new anonymous ID on their next visit. When hashed for experiment assignment, they may land in a different bucket — potentially switching from control to treatment or vice versa. This:
1. Creates an inconsistent user experience (sees both variants)
2. Contaminates the experiment (one user counted in both groups)
3. Biases toward null (adds noise)

↳ **"How do you fix this?"**
Use login-based assignment instead of cookie-based when possible. The user_id is stable across sessions and devices. For anonymous users: accept the noise (it's random and biases toward null, not toward false positives), or use fingerprinting (privacy concerns).

↳ **"What fraction of users typically delete cookies?"**
Varies by platform — 10–20% of browser sessions have no cookies in some studies. Mobile apps with logged-in users don't have this problem. This is why login-based experiments are strongly preferred for measuring retention metrics.

🚩 **Red Flag**
Not knowing this is a problem at all.

⭐ **What earns the role**
Noting that cookie deletion is not random — privacy-conscious users are more likely to delete cookies, creating a systematic (non-random) bias if not handled.

---

## Day 38 — Experiment Sizing for Rare Events

🎯 **"You want to measure the effect of a feature on 7-day churn (rare event, ~2% rate). How do you design this?"**

💬 **Model Answer**
Rare events require large samples because variance is high relative to the signal. For 2% baseline churn and wanting to detect 10% relative reduction (to 1.8%), sample size calculation gives tens of millions of users — often impractical.

Alternatives:
1. **Use a more frequent proxy metric** — e.g., Day-1 or Day-3 retention (higher base rate, faster to measure, predictive of Day-7 churn).
2. **Longer experiment window** — accumulate more events over time.
3. **Bayesian methods with informative priors** — prior knowledge from similar experiments reduces required data.
4. **Aggregate at higher level** — if 2% churn still means 10K events at your scale, you may be fine.

↳ **"How do you validate that a proxy metric (Day-3 retention) predicts Day-7 churn?"**
Historical calibration: show that experiments that improved Day-3 retention in the past also improved Day-7 churn, and quantify the relationship. This is called surrogate validation — the proxy must have a consistent, quantifiable relationship with the true outcome.

🚩 **Red Flag**
"We'll just wait until we have enough churn events." — May take months, blocking the roadmap.

⭐ **What earns the role**
Proposing a formal surrogate metric validation framework — not just assuming that a proxy correlates, but proving it empirically across historical experiments.

---

## Day 39 — Experiment on Pricing

🎯 **"How would you A/B test a price change from $9.99 to $11.99 on a subscription product?"**

💬 **Model Answer**
Pricing experiments are uniquely sensitive. Standard user-level randomization creates fairness problems (two users paying different prices for the same product) and SUTVA violations (users who talk to each other or compare prices online).

Approach:
1. **Geo-based experiment:** Different markets (countries or DMAs) see different prices. Reduces within-market fairness concerns.
2. **New user only:** Existing users see current price; only new users are randomized. Avoids existing user outrage.
3. **Time-based (switchback):** Not ideal for pricing — users remember prices.
4. **Legal review:** Price discrimination laws vary by jurisdiction (EU, US states).

Metrics: conversion rate, LTV, churn rate, revenue per user, support tickets about pricing.

↳ **"How do you handle the survivorship bias in a pricing experiment?"**
Users who convert under the higher price are self-selected as less price-sensitive. Comparing revenue per *converting* user is biased. Compare revenue per *exposed* user — this captures the conversion rate penalty of higher prices.

🚩 **Red Flag**
"We'd just randomize users to see different prices." — Ignores fairness, legal, and SUTVA issues.

⭐ **What earns the role**
Connecting pricing experiments to price elasticity estimation — what you ultimately want to know is the demand curve, not just the binary "did $11.99 outperform $9.99."

---

## Day 40 — Cohort Analysis

🎯 **"How do cohort analysis and A/B testing differ? When would you use each?"**

💬 **Model Answer**
**A/B testing:** Randomized, concurrent, causally valid. Best for measuring the effect of a specific change.

**Cohort analysis:** Observational, follows groups defined by some characteristic (sign-up date, acquisition channel) over time. Describes how different cohorts behave, but can't establish causation.

Use cohort analysis when: (1) Randomization is impossible (historical data only). (2) You want to understand long-term retention patterns without an experiment. (3) You want to see how product changes affected users who joined at different points in time.

↳ **"Can you use cohort analysis to validate A/B test results?"**
Yes — compare the behavior of cohorts exposed to a feature (post-launch) vs cohorts who weren't (pre-launch), controlling for cohort age. This is difference-in-differences thinking. Not as clean as an A/B test but useful for post-hoc validation.

↳ **"What is the problem with cohort analysis for causal inference?"**
Cohort membership is not random — users who signed up in January vs March may differ systematically (different marketing channels, seasonality, product version). These confounders make causal claims unreliable.

🚩 **Red Flag**
"Cohort analysis proves the feature caused retention to improve." — Cohort analysis is observational; causation requires randomization.

⭐ **What earns the role**
Using cohort analysis proactively to check for novelty effects and long-term decay in A/B test results — plotting the treatment effect by user cohort age.

---

## Day 41 — Funnel Analysis in Experiments

🎯 **"Your experiment increases top-of-funnel clicks by 10% but checkout conversion is flat. What's happening?"**

💬 **Model Answer**
The treatment is attracting more clicks (higher quantity) but those clicks are lower intent — the marginal users clicking in treatment don't convert. This is a classic quality-quantity tradeoff. Possible causes: the new design attracts exploratory clicks that don't lead to purchase; the change appeals to a different (less purchase-ready) user segment.

Diagnose: measure conversion rate at *each funnel step* — where does the funnel drop off? If click→cart is flat but cart→checkout drops, the issue is cart experience, not click quality.

↳ **"How do you define funnel metrics for an A/B test?"**
Each funnel step as a conversion rate: step_n / exposed_users (not step_n / step_{n-1}) to avoid survivorship bias. Using exposed users as denominator ensures all steps are comparable and compositional effects don't confuse the analysis.

↳ **"Should you measure funnel step rates per exposed user or per user who completed the prior step?"**
Per exposed user is cleaner for A/B tests — it avoids selection bias at each step. Per prior step is useful for diagnosing where drop-off occurs, but treat it as exploratory.

🚩 **Red Flag**
"Clicks went up so the experiment worked." — Top-of-funnel metrics without downstream validation are insufficient for most products.

⭐ **What earns the role**
Proposing to measure revenue per exposed user (not per click, not per converter) as the unbiased summary metric for the full funnel.

---

## Day 42 — Simpson's Paradox

🎯 **"Explain Simpson's Paradox and give me an example from A/B testing."**

💬 **Model Answer**
Simpson's Paradox: a trend appears in several subgroups but disappears or reverses when groups are combined. Caused by a lurking confounding variable that creates a weighted combination effect.

A/B testing example: Treatment has higher conversion overall (8% vs 7%). But split by mobile/desktop: mobile shows 5% treatment vs 6% control (treatment worse), desktop shows 10% treatment vs 9% control (treatment slightly better). The overall result is Simpson's Paradox — the aggregate is dominated by the device mix, not the true treatment effect.

↳ **"How would this happen in a real experiment?"**
If treatment users are disproportionately desktop users (maybe the feature only renders on desktop), and desktop has higher baseline conversion, the treatment group looks better in aggregate — but only because of the device composition, not the feature.

↳ **"How do you catch and prevent this?"**
SRM checks by segment catch composition differences. Always stratify results by major segments (device, geography, user type) before reporting. If composition differs between groups, this is effectively a form of selection bias.

🚩 **Red Flag**
Not knowing what Simpson's Paradox is — it comes up in almost every FAANG data interview.

⭐ **What earns the role**
Connecting Simpson's Paradox to the general problem of confounding in aggregate statistics — and proposing stratified analysis as the standard remedy.

---

## Day 43 — Switchback Experiments

🎯 **"What is a switchback experiment and when would you use one?"**

💬 **Model Answer**
A switchback alternates treatment and control over time windows within the same market or unit (e.g., odd hours = treatment, even hours = control, cycling across days). Used when user-level randomization causes interference — typically in marketplaces (ride-sharing, food delivery, real-time pricing).

Why: supply and demand interact so much that user-level experiments are contaminated. By alternating over time, you isolate the effect of the algorithm on the entire market.

↳ **"What are the statistical challenges of switchback experiments?"**
(1) **Carryover:** Effects from one period bleed into the next (e.g., surge pricing in treatment changes driver positioning that persists into control period). (2) **Time confounds:** Morning vs evening has different baselines — need to model time-of-day effects. (3) **Few independent observations:** Each market has few time windows — low power. Requires careful analysis with mixed-effects models.

↳ **"How long should each switchback window be?"**
Long enough for the market to reach a new equilibrium after the switch (washout period), but short enough to get many windows. For ride-sharing: typically 30 minutes to 2 hours. Must be determined empirically by measuring how quickly the market equilibrates.

🚩 **Red Flag**
"Switchback is just A/B testing over time." — Missing the specific challenges of carryover and market equilibration.

⭐ **What earns the role**
Knowing that switchback experiments estimate the Average Treatment Effect (ATE) over the time windows, not over users — and that the unit of analysis is the time window, not the user.

---

## Day 44 — Difference-in-Differences

🎯 **"You can't randomize. A new feature launched in Germany but not France. How do you measure its effect?"**

💬 **Model Answer**
Use Difference-in-Differences (DiD). Compare the *change* in metric from before to after launch in Germany (treatment) vs France (control). The DiD estimator:

Δ = (Germany_after − Germany_before) − (France_after − France_before)

This controls for time trends (if both countries were trending up, we subtract that trend) and fixed differences between countries (Germany might always convert better — we subtract that too).

↳ **"What is the key assumption of DiD?"**
Parallel trends: in the absence of treatment, Germany and France would have followed the same trend. Testable in the pre-treatment period — if trends diverged pre-launch, DiD is invalid.

↳ **"What if parallel trends don't hold?"**
Use synthetic control: create a "synthetic France" as a weighted combination of multiple control countries that best matches Germany's pre-treatment trajectory. More robust to non-parallel trends.

🚩 **Red Flag**
Just comparing Germany to France post-launch without the pre-period — this is pure cross-sectional comparison with all its confounders.

⭐ **What earns the role**
Proactively checking parallel pre-trends and knowing that DiD is a causal inference method, not just a descriptive comparison.

---

## Day 45 — Experiment Velocity & Culture

🎯 **"How do you build a culture of experimentation in a company that doesn't have one?"**

💬 **Model Answer**
Structural changes matter more than training:
1. **Make experimentation easy:** Build self-serve tooling so engineers can launch experiments in hours, not weeks.
2. **Democratize data:** Any PM or engineer can see results without waiting for a data scientist.
3. **Celebrate null results:** Publicly recognize experiments that showed nothing — they prevented bad launches.
4. **Tie decisions to experiments:** Require experiment evidence for major launches. Make it the default, not the exception.
5. **Reduce the cost of being wrong:** Small experiments, fast cycles. The goal is learning, not just winning.

↳ **"What's the biggest bottleneck to experiment velocity?"**
Usually: analysis time (waiting for DS to analyze) or engineering time (building instrumentation per experiment). Self-serve platforms (like Optimizely, Statsig, or internal tools) and standardized metrics pipelines solve both.

↳ **"What's a bad experimentation culture look like?"**
HiPPO decisions (Highest Paid Person's Opinion), experiments run retroactively to justify decisions already made, celebrating wins but killing experiments that show regressions, and shipping without experiments for "obvious" features.

🚩 **Red Flag**
"Just train people to run experiments." — Training without structural support doesn't change behavior.

⭐ **What earns the role**
Quoting the research: companies with mature experimentation cultures (Amazon, Booking.com, LinkedIn) attribute significant revenue growth directly to experiment-driven decision-making and run 1000+ experiments per year.

---

## Day 46 — Incremental Lift & Holdout

🎯 **"What is incremental lift and how do you measure it?"**

💬 **Model Answer**
Incremental lift is the additional conversion/revenue/action caused by a treatment *beyond what would have happened anyway*. It's the true causal effect, not the total observed metric.

Measurement via holdout: run a campaign (e.g., email, ad) to treatment group. Keep a matched holdout group who sees nothing (or a placebo). Incremental lift = (conversion rate of treated) − (conversion rate of holdout).

↳ **"Why is holdout important for measuring ad campaign effectiveness?"**
Without a holdout, you measure total conversions in the exposed group. But many of those users would have converted organically. The incremental lift subtracts the organic baseline. Companies regularly find that 30–50% of attributed conversions are non-incremental.

↳ **"How do you create the holdout group for an ad campaign?"**
Randomly withhold a fraction (e.g., 10%) from seeing the ad at targeting time, not at impression time. The holdout is selected before any targeting decisions. This ensures comparable groups.

🚩 **Red Flag**
"We measure ad effectiveness by looking at conversion rate of users who saw the ad." — This confuses correlation (users who see ads tend to be intenders) with causation.

⭐ **What earns the role**
Knowing that the holdout group must be selected at the same stage as treatment, and that late-stage holdouts (after targeting) re-introduce selection bias.

---

## Day 47 — Multi-Armed Bandit

🎯 **"When would you use a multi-armed bandit instead of an A/B test?"**

💬 **Model Answer**
Bandits are better when:
1. **Regret matters:** You don't want to show an inferior variant to many users during the experiment (e.g., personalized recommendations, ads — every bad impression costs).
2. **Fast-moving environments:** Bandit adapts to changes in user behavior over time; A/B test is static.
3. **Many variants:** Testing 20 content pieces — A/B testing would require massive traffic; bandits explore efficiently.

A/B tests are better when:
1. You need clean statistical inference for a launch decision.
2. The experiment is short and regret is low.
3. You need reproducible, auditable results.

↳ **"What is the exploration-exploitation tradeoff in bandits?"**
Exploration: trying variants to estimate their value. Exploitation: showing the best known variant to maximize reward. Pure exploitation stops exploring too early (misses better variants). Pure exploration ignores what you've learned. Algorithms like UCB (Upper Confidence Bound) and Thompson Sampling balance this optimally.

↳ **"Does Thompson Sampling give valid p-values?"**
No — adaptive sampling biases the posterior toward the true best variant, which invalidates frequentist inference. You can't use a bandit's results as if they were from a randomized A/B test. This is a critical limitation for statistical inference.

🚩 **Red Flag**
"Bandits are always better than A/B tests." — They optimize for regret, not inference. Different objective.

⭐ **What earns the role**
Knowing that bandits create a fundamental inference problem — the sampling distribution of results is not the same as in a fixed randomized experiment, so post-hoc significance testing is invalid.

---

## Day 48 — Heterogeneous Populations

🎯 **"Your experiment shows +5% CTR on average, but you know your user base is 60% mobile and 40% desktop. Should you stratify your analysis?"**

💬 **Model Answer**
Yes, for two reasons:
1. **Precision:** Stratified analysis reduces variance — if mobile and desktop have different baseline CTRs, pooling adds unnecessary noise. Stratified estimates are more precise.
2. **Validity:** If the treatment effect differs between mobile and desktop (HTE), the average effect may mask important differences that affect the launch decision.

Post-stratification: compute weighted average of stratum-specific effects, weighted by stratum size. This is equivalent to controlling for device type in a regression.

↳ **"Is stratified randomization better than simple randomization?"**
Yes when strata are known upfront. Stratified randomization ensures balance across strata within each arm, increasing precision. Particularly valuable for small experiments where imbalance is more likely by chance.

↳ **"If you didn't stratify at assignment time, can you still stratify at analysis time?"**
Yes — post-stratification is valid with random assignment. It adjusts for any observed imbalance in stratum composition between groups, similar to CUPED.

🚩 **Red Flag**
"Stratification isn't necessary — randomization guarantees balance." — True in expectation, but not in finite samples. Stratification reduces variance from finite-sample imbalance.

⭐ **What earns the role**
Connecting stratification to regression adjustment — both achieve the same goal of removing variance from known covariates, with regression being more flexible for continuous variables.

---

## Day 49 — Experiment Automation & Monitoring

🎯 **"How do you automatically detect when an experiment has gone wrong?"**

💬 **Model Answer**
Build automated checks that run continuously:
1. **SRM detector:** Chi-squared test on group sizes every hour. Alert if p < 0.001.
2. **Crash rate monitor:** If treatment crash rate spikes significantly → auto-pause.
3. **Metric outlier detection:** Alert if any guardrail metric changes by >3σ from historical baseline.
4. **Data completeness check:** Alert if logging volume drops unexpectedly (instrumentation failure).
5. **Assignment audit:** Verify that user-level assignments are consistent over time (no re-assignment).

↳ **"Should automated monitors pause experiments automatically?"**
For clear safety signals (crash spike, massive SRM) — yes, auto-pause. For ambiguous metric movements — alert a human. Automated pausing prevents disasters; human review prevents over-reacting to noise.

↳ **"What's the difference between monitoring and peeking?"**
Monitoring for anomalies (bugs, crashes, SRM) is good practice. Making launch decisions based on interim metric significance is peeking. The dashboard should show guardrail metrics for safety monitoring, not primary metric significance for decision-making.

🚩 **Red Flag**
"We check results every day and stop when we see something significant." — Conflates anomaly monitoring with statistical decision-making.

⭐ **What earns the role**
Designing the monitoring system with always-valid confidence sequences for primary metrics — valid for monitoring at any time, but clearly labeled as "monitoring" not "decision criteria."

---

## Day 50 — Wrapping Up L4

🎯 **"If you could only teach a new data scientist one thing about A/B testing, what would it be?"**

💬 **Model Answer**
That statistical significance is not the same as practical significance, and neither is sufficient for a launch decision. A significant result means the data is unlikely under the null — it doesn't mean the feature is good, the metric is right, the experiment was designed correctly, or the effect will persist post-launch. Good experimentation is a system: right metric, right design, right analysis, right interpretation, and the humility to know what you can't measure.

↳ **"What's the most common mistake you'd want them to avoid?"**
Peeking — making decisions before the pre-specified sample size is reached. It's subtle (everyone wants to know early), the damage is invisible (you don't see the false positives you're generating), and it's rampant in industry. The fix is simple: commit to a sample size and honor it.

↳ **"What does a mature experimentation culture look like at FAANG level?"**
Pre-registration is enforced by tooling. Null results are reported and analyzed. Every major launch requires an experiment. HTE analysis is routine. Long-term holdouts are maintained for top features. The experimentation platform gives real-time anomaly monitoring with always-valid inference. Data scientists review experiment design before launch, not just results after.

🚩 **Red Flag**
"I'd teach them how to run a t-test in Python." — Technical execution is the easiest part.

⭐ **What earns the role**
Articulating that experimentation is ultimately about decision-making under uncertainty — and that the goal is calibrated, reproducible decisions at scale, not individual experiment wins.

---

*— END OF DAYS 1–50 (L4) —*
*Days 51–100 (L6) continue in: faang_ab_testing_days51_100.md*
