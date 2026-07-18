# Chapter 13: Proxy Metrics for Long-Term Outcomes

*(Closes Module 2: Metrics & Measurement)*

## 1. Intuition

Chapter 9 flagged this tension but deferred it: your OEC should ideally reflect long-term value (customer LTV, sustained retention), but you usually can't wait 6-12 months to observe that for every experiment — by the time you know the true long-term effect, you've either shipped something harmful for months or delayed a genuinely good feature for no reason.

This chapter formalizes the actual solution used at large tech companies: instead of picking one short-term proxy and hoping it correlates with long-term value (the informal approach from Chapter 9), you build a **statistically validated model connecting short-term surrogate metrics to the long-term outcome**, and use that model's *predicted* long-term effect as your decision criterion — even though you never actually waited to observe the real long-term outcome in the current experiment.

## 2. The Surrogate Index Method

This approach, formalized in a well-known paper by Athey, Chetty, Imbens, and colleagues ("Surrogate Index" methodology, used at companies like Google, Netflix), works as follows:

**Step 1 — Historical calibration**: using *past* experiments or observational data where you *did* eventually observe the true long-term outcome (e.g., 12-month LTV), collect a rich set of short-term surrogate metrics available within the first 1-4 weeks (e.g., week-1 retention, week-2 engagement, early conversion, early support-ticket rate) for those same historical units.

**Step 2 — Fit a predictive model**: regress the long-term outcome $Y_{long}$ on the vector of short-term surrogates $S = (S_1, S_2, ..., S_k)$ using this historical data:

$$Y_{long} = g(S_1, S_2, ..., S_k) + \epsilon$$

This gives you a **surrogate index**: a predicted long-term value as a function of observable short-term metrics.

**Step 3 — Apply the model to your current experiment**: for your live experiment, measure the same short-term surrogates $S$ for both treatment and control (available quickly, within days/weeks), plug them into the fitted model $g(\cdot)$ from Step 2, and use the *predicted* long-term outcome difference between arms as your effective OEC — without needing to wait for the actual long-term outcome to materialize in this specific experiment.

**The key statistical assumption this relies on**: the short-term surrogates must be sufficient statistics for the treatment's effect on the long-term outcome — formally, treatment must affect the long-term outcome *only through* its effect on the observed surrogates (no unobserved "side channel" through which treatment affects long-term value without affecting any measured surrogate). This is a strong, non-trivial assumption, and it's exactly what critics of the method push on — good short-term surrogates need to be validated repeatedly against realized long-term outcomes when they eventually do become available, not assumed to hold forever.

## 3. Simpler Alternative: Empirically-Validated Single Proxy

Not every team has the infrastructure or historical data volume to build a full surrogate index model. The simpler, more common version used in practice: **empirically demonstrate, using historical data, that your chosen short-term proxy (e.g., week-1 retention) is strongly correlated with the long-term outcome you actually care about (e.g., 6-month LTV)**, then use that single proxy as your OEC with documented confidence in the correlation, re-validating periodically.

This is weaker than the full surrogate index (single correlation vs. a fitted multi-variable model) but far more practically achievable, and is likely the answer to lead with in an interview unless specifically asked about the more advanced surrogate index methodology — most day-to-day experimentation at even large companies uses this simpler validated-proxy approach, reserving full surrogate index modeling for particularly high-stakes or company-wide metric decisions.

## 4. Worked Example

A subscription streaming product wants to test a new onboarding flow. The true goal: 12-month subscriber retention (LTV). Waiting 12 months per experiment is untenable — the team wants to ship decisions within 2-3 weeks.

**Historical validation (done once, reused across experiments)**: using 2 years of historical cohort data where 12-month retention is now known, the team finds:
- Week-1 app open rate: correlation with 12-month retention, $\rho=0.35$ (weak — noisy, too easily gamed by notification spam)
- Week-2 "completed 3+ core actions" rate: correlation with 12-month retention, $\rho=0.68$ (moderate-strong)
- Combined surrogate index (week-2 core actions + week-1 open rate + early support ticket rate, combined via regression): correlation with 12-month retention, $\rho=0.81$ (much stronger than any single proxy alone)

**Decision**: the team adopts the combined surrogate index (Step 2 model) as their OEC for onboarding-related experiments specifically, since the $\rho=0.81$ combined model is meaningfully more predictive than any single 2-3 week metric alone, and onboarding experiments are frequent enough and high-stakes enough (first impression drives the whole subscriber lifetime) to justify the extra modeling investment.

**For a lower-stakes, more frequent experiment** (e.g., a minor UI copy change), the same team instead just uses "week-2 core actions rate" alone as a validated single proxy ($\rho=0.68$) — good enough for a lower-stakes decision, without the overhead of maintaining a full surrogate index model for every minor test.

## 5. Production Considerations

- **Proxy validation is not a one-time task.** User behavior and product context shift over time — a proxy that was well-correlated with long-term value two years ago may have decayed in predictive power (e.g., if the product or user base has changed meaningfully). Schedule periodic re-validation, not a one-and-done calibration.
- **Beware proxy metrics becoming Goodhart's Law targets** (same risk flagged in Chapter 9) — if teams know "week-2 core actions" is the de facto OEC, they'll optimize that specific metric, potentially in ways that stop reflecting genuine long-term value (e.g., nudging users into actions that count toward the metric but don't reflect real engagement).
- **Full surrogate index modeling requires substantial historical data infrastructure** (needing large enough historical cohorts with realized long-term outcomes) — this is often the actual bottleneck to adopting the more rigorous approach, not the statistical methodology itself.
- **This entire chapter is essentially the OEC problem (Chapter 9) solved with more rigor** — worth stating explicitly if asked to relate the two: Chapter 9 informally says "pick a good short-term proxy," this chapter gives you the statistically validated machinery to actually justify that choice.

## 6. Interview Traps

- **Trap #1**: Proposing to just "wait for the long-term metric" as the default answer without acknowledging this is usually operationally untenable — interviewers want to see you propose the proxy/surrogate solution, not sidestep the tension.
- **Trap #2**: Not being able to articulate the key assumption behind the surrogate index method (surrogates must fully mediate treatment's effect on the long-term outcome) — this is the crux of why the method can fail, and a sophisticated interviewer will probe exactly this.
- **Trap #3**: Treating proxy validation as a one-time setup rather than an ongoing process requiring re-validation as user behavior shifts.
- **Trap #4**: Not distinguishing between the lightweight "empirically validated single proxy" approach (common, practical) and the full "surrogate index" methodology (rigorous, data-intensive) — conflating these, or assuming every company runs the full academic version, signals a gap between theory and practice.

## 7. L5-Differentiating Talking Points

- Being able to name the surrogate index methodology and its core mediating-variable assumption, while also being pragmatic about when the lightweight single-proxy version is "good enough," shows both theoretical depth and practical judgment about when each level of rigor is warranted.
- Proactively raising the Goodhart's Law risk (proxies becoming gamed targets) connects this chapter back to Chapter 9 and shows a consistent, principled worry about metric-driven optimization throughout the curriculum, not an isolated concern.
- Framing this whole chapter as "OEC design (Chapter 9), but with the statistical machinery to actually validate the choice" demonstrates you see the curriculum as an integrated system rather than 21 disconnected topics.
- Mentioning that proxy validation needs to be periodically refreshed (not "set once, use forever") signals genuine production experience with metrics degrading over time.

## 8. Comprehension Check

1. What is the key statistical assumption required for the surrogate index method to give an unbiased estimate of the long-term treatment effect?
2. Why might a company use a simpler single validated proxy for most experiments, but invest in a full surrogate index model for a smaller subset of high-stakes decisions?
3. Explain, using the worked example, why combining multiple short-term surrogates (via regression) produced a higher correlation with the long-term outcome than any single surrogate alone.
4. How does this chapter's solution relate to the OEC design problem introduced in Chapter 9?
5. A team says "our proxy metric was validated against long-term LTV two years ago, so we're confident using it going forward indefinitely." What's the risk in this reasoning?

---
*End of Module 2: Metrics & Measurement (Chapters 9-13).*
*Next: Chapter 14 — Sample Ratio Mismatch (start of Module 3: Failure Modes & Diagnostics)*
