# Causal Inference — Interview Question Bank
### Google MLE / Data Scientist prep (fresh questions, not repeated from the main guide)

Organized by theme. Each question has a model answer. Use these to mock-interview yourself out loud — the goal is fluency, not memorization.

---

## Section 1 — Experiment Design & A/B Testing

**Q1. You launch a fully randomized A/B test and see a statistically significant 3% lift in a key metric. Your VP asks "can we expect this globally at 100% rollout?" What's your answer?**
A: Not automatically. A user-randomized experiment estimates the effect *holding the rest of the system fixed at ~50/50*. At 100% rollout you need to worry about: (1) interference/SUTVA violations — shared resources like ad inventory, search index load, or marketplace supply that behave differently at full scale; (2) novelty effects that fade (or grow) over time; (3) whether the 2-week window captured long-run behavior; (4) segment mix — if certain user segments were excluded from the test, the global population differs. I'd recommend a ramp-up (5%→25%→50%→100%) with monitoring, plus a long-running holdback group, before declaring the 3% is the true global effect.

**Q2. What's the difference between "intent-to-treat" (ITT) and "per-protocol" analysis, and which should you report by default?**
A: ITT analyzes everyone as assigned, regardless of whether they actually complied with/received the treatment. Per-protocol only analyzes those who complied. ITT preserves the unbiasedness from randomization (comparing as-assigned groups); per-protocol can reintroduce selection bias, since compliers may differ systematically from non-compliers. Default to ITT for the headline causal effect; per-protocol (or IV/LATE) is a secondary, assumption-heavy analysis.

**Q3. Your experiment metric is "revenue per user," but the distribution is extremely right-skewed with a few whale users. How does this affect your test, and what would you do?**
A: High variance from outliers inflates the standard error, hurting statistical power and making the t-test's normal approximation less reliable in small-to-moderate samples. Options: cap/winsorize extreme values, use a log or rank transform, use a variance-reduction technique like CUPED (regression-adjust using pre-experiment revenue as a covariate), or bootstrap the confidence interval instead of relying on asymptotic normality.

**Q4. Explain CUPED (Controlled-experiment Using Pre-Experiment Data) in plain terms and why it helps.**
A: CUPED adjusts the outcome using a pre-experiment covariate that's correlated with the outcome but unaffected by treatment (e.g., last month's revenue). You compute Y_adjusted = Y − θ(X_pre − X̄_pre), choosing θ to minimize variance (θ = Cov(Y,X_pre)/Var(X_pre)). Since X_pre is uncorrelated with treatment (measured before treatment), this doesn't bias the treatment effect estimate, but it strips out variance explained by pre-existing user differences, shrinking the standard error and increasing power — letting you detect smaller effects or need less sample size.

**Q5. How would you size a sample for a metric that's a ratio (e.g., click-through rate = clicks/impressions), where impressions vary a lot across users?**
A: Ratio metrics at the user level need the **delta method** to approximate the variance of a ratio, since simple per-user averaging can be misleading when the denominator (impressions) varies across users. Compute Var(clicks/impressions) ≈ using a first-order Taylor expansion around the means of numerator and denominator, incorporating their covariance — then plug that into the standard power formula. Ignoring this and treating it as a simple proportion can under- or over-estimate the needed sample size.

---

## Section 2 — Observational Data / Confounding

**Q6. A DS finds that users who enable dark mode have 20% higher retention. Product wants to force dark mode on everyone. What's your first reaction?**
A: I'd flag this as likely confounded before recommending anything. Users who *seek out and enable* a settings option are probably more engaged, more tech-savvy, or more invested in the product to begin with — that self-selection (not dark mode itself) may drive retention. I'd check for observable confounders (tenure, device type, engagement level) via matching/regression, but ultimately I'd push for a randomized experiment (force dark mode on a random subset) before any global rollout — dark mode toggling is exactly the kind of low-cost feature where an RCT is easy and observational data is unreliable.

**Q7. What is "immortal time bias" and where might it show up in product analytics?**
A: It occurs when there's a gap between when a unit is classified as "treated" and when treatment actually starts, during which the unit couldn't have experienced the outcome (e.g., churn) yet — inflating the apparent benefit of treatment. Example: comparing "retention of users who eventually upgrade to premium" vs. free users — users must survive long enough to *decide* to upgrade, so the "premium" group is implicitly conditioned on survival during the pre-upgrade window, biasing retention comparisons upward for premium. Fix: align time zero properly (e.g., use a landmark/time-varying treatment analysis, not a fixed retrospective label).

**Q8. Explain the "table 2 fallacy."**
A: In a regression with multiple covariates, people often interpret *every* coefficient (not just the treatment variable) causally. But covariates included purely to control confounding for the main treatment effect may not have their own coefficients interpretable as causal effects — they could be mediators, could have their own unmeasured confounders, or could be "adjusted for" in a way that's only valid for the treatment variable's coefficient, not for themselves. Only the coefficient you specifically designed the model to identify (usually the treatment variable) should be read causally.

**Q9. You control for "number of prior purchases" when estimating the causal effect of a loyalty program (D) on future purchases (Y). Is this a good idea?**
A: It depends on timing. If "number of prior purchases" is measured *before* enrollment in the loyalty program, it's a legitimate pre-treatment confounder to control for (people with more purchase history may be more likely to join and more likely to buy again anyway). But if it's measured *after* or *concurrently* with program enrollment, it could be a mediator (loyalty program → more purchases → counted in this measure) — controlling for a mediator blocks part of the true causal path and biases the estimated effect toward zero. Always check timing before deciding to control for a variable.

**Q10. How do negative control outcomes help validate an observational causal analysis?**
A: A negative control outcome is one that treatment plausibly *cannot* causally affect, but that would be affected by the same confounders you're worried about. If your causal method finds a "significant effect" on the negative control, that's strong evidence your adjustment strategy isn't fully removing confounding (a red flag for the main analysis too). No effect on the negative control increases (but doesn't prove) confidence in your design.

---

## Section 3 — Method-Specific Judgment Calls

**Q11. You're choosing between propensity score matching and inverse probability weighting (IPW) for the same dataset. What practical factors tip you toward one vs the other?**
A: Matching discards unmatched units, which is transparent about lack of overlap but throws away data and can be sensitive to matching algorithm/caliper choices; it's easier to visually inspect balance. IPW uses all the data (weighted) but is very sensitive to extreme propensity scores (weight explosion) if overlap is poor. If overlap is strong, IPW (or doubly-robust methods) is usually more efficient. If overlap is weak in some regions, matching (or trimming) makes the lack of common support visible rather than papering over it with huge weights. In practice, doubly-robust estimators (e.g., AIPW / TMLE) that combine an outcome model and a propensity model are often preferred, since they're consistent if *either* model is correctly specified.

**Q12. What is a "doubly robust" estimator and why might Google DS teams favor it over plain PSM?**
A: A doubly robust (DR) estimator combines an outcome regression model and a propensity score model such that the estimate is consistent if *at least one* of the two models is correctly specified (not necessarily both). This gives a safety net against misspecifying either model — plain regression adjustment fails if the outcome model is wrong; plain IPW fails if the propensity model is wrong; DR (e.g., Augmented IPW, TMLE) fails only if *both* are wrong. Given how often functional forms are uncertain in messy product data, this extra robustness is valuable.

**Q13. In a DiD setup, why might you prefer to also include unit fixed effects and time fixed effects rather than just the four-cell (2x2) comparison?**
A: With many units and many time periods, unit fixed effects absorb any time-invariant unit-level confounder (controlling for it automatically, not just the two groups' average), and time fixed effects absorb any shock common to all units in a given period (macro trends, seasonality). This generalizes the simple 2x2 DiD to a full panel setting and is more efficient, using all periods rather than collapsing to before/after averages — but it inherits the same parallel trends assumption, and (as covered in the main guide) can be biased under staggered treatment timing with heterogeneous effects.

**Q14. Your RDD analysis is on Google Play app ratings, where a "featured" badge kicks in at 4.5 stars average rating. What's a serious threat to this design that's specific to ratings data, beyond general manipulation concerns?**
A: Ratings are computed from a *finite, growing sample* of user reviews — the running variable itself is noisy and mechanically "regresses" as more reviews come in (a newly-launched app's 4.5 average is far less stable than an app with 10,000 reviews at 4.5). This means units near the cutoff aren't necessarily comparable in the way RDD assumes — apps that just barely cross 4.5 by chance (due to few reviews / sampling noise) will regress toward their "true" rating over time regardless of the badge, which can masquerade as or obscure a badge effect. You'd want to control for/stratify by review count, or model the measurement error in the running variable explicitly.

**Q15. When would you use a "difference-in-differences" versus a "changes-in-changes" (CIC) approach?**
A: DiD is testable/valid on the *mean* and assumes parallel trends in means; it's not invariant to monotonic transformations of Y (a DiD estimate on log(Y) can imply a different conclusion than on Y directly). Changes-in-changes generalizes the idea to the whole outcome *distribution* rather than just the mean, under a weaker/different assumption (that the distribution of Y(0) is stable in a rank sense over time, invariant to monotonic transforms), and lets you estimate the treatment effect at different quantiles, not just the average. Use CIC when treatment effects are plausibly heterogeneous across the distribution and DiD's mean-only, transform-sensitive answer feels too thin.

---

## Section 4 — Google/Big-Tech Context Questions

**Q16. Search ranking team wants to know the causal effect of "page load speed" on user satisfaction, using observational logs (can't randomly assign slow load times to real users). How do you approach this?**
A: I'd look for quasi-experimental leverage rather than pure observational regression: (1) natural variation from server/CDN incidents or infra rollouts that changed load speed for reasons unrelated to user behavior (a natural experiment / usable as an instrument if it satisfies exclusion); (2) geographic or device-based discontinuities (e.g., users just inside vs. outside a CDN edge boundary) as an RDD-style design; (3) if truly stuck with observational data, use rich covariate adjustment (device, connection type, historical engagement) with heavy sensitivity analysis, while being explicit that unmeasured confounders (e.g., user patience/context) are a real risk. I'd also flag that a controlled, artificially-throttled speed experiment (even on a small % of sessions) is usually feasible and far more trustworthy than any observational workaround.

**Q17. YouTube wants to measure the causal effect of video recommendations on watch time, but recommendations are personalized (i.e., treatment assignment is a function of the user's own history — not random). Why is this a fundamentally hard causal problem, and what's a common industry approach?**
A: The recommendation itself is endogenous — chosen *because* the algorithm predicts the user will like it, so comparing "watched recommended video" vs not is deeply confounded by the same signals the model used to make the recommendation. A common approach is **off-policy evaluation using randomization already embedded in the system** — e.g., exploiting random perturbations/exploration slots that recommender systems intentionally inject (small % random slot, or randomized re-ranking) to get unbiased counterfactual signal, combined with techniques like inverse propensity scoring over the *logged* recommendation policy (treating the known serving probabilities as propensity scores) — this is the "counterfactual/off-policy learning" framework, an application of IPW ideas from Chapter 6 to recommender systems specifically.

**Q18. A regional team says "we saw a huge revenue increase right after our marketing campaign launched, causation is obvious." What questions do you ask before agreeing?**
A: Was there a comparison/control region or a pre-existing trend (to rule out this being a seasonal or macro pattern — i.e., can we build a DiD or synthetic control rather than trust a raw before/after)? Was the campaign timed to coincide with anything else (holiday, other launches, pricing change)? Is "right after" precise — could this be reverse causation (campaign launched *because* leadership anticipated a strong period)? Is the "huge" increase robust to removing outlier days/whale customers? Without a counterfactual, a raw before/after is not a causal claim, just a coincidence in time.

**Q19. Ads team proposes: "let's just compare revenue from advertisers who opted into the new bidding tool vs those who didn't." What's the single biggest problem, and what would you propose instead?**
A: Biggest problem: self-selection — advertisers who opt in are systematically different (likely more sophisticated, larger budgets, more growth-oriented) from those who don't, so the comparison conflates "who opts in" with "what the tool does." I'd propose a randomized encouragement design: randomly *encourage* (e.g., via UI prompt/nudge) a subset of advertisers to try the tool, and use that random encouragement as an instrument for actual adoption (an IV/LATE approach) — this avoids the need to force adoption on anyone while still getting a causal read on compliers.

**Q20. You're told an experiment must run for exactly 1 week due to a launch deadline, but your power calculation says you need 3 weeks of data for adequate power. What are your options, and what would you actually recommend?**
A: Options: (a) increase the traffic allocation to treatment/control (e.g., 50/50 instead of 10/90) to gain power faster; (b) use variance reduction (CUPED, stratification) to shrink required sample size; (c) accept a larger minimum detectable effect (only sized to catch a bigger lift, explicitly communicating that smaller effects won't be reliably detected); (d) push back on the deadline with the power/risk tradeoff spelled out in business terms (e.g., "at 1 week we can only reliably detect an effect above X%; if the true effect is smaller, we may ship based on noise"). I would not silently ship the analysis without flagging the power shortfall — that's the deliverable Google interviewers want to hear (transparency over false confidence).

---

## Section 5 — Statistical/Mathematical Depth

**Q21. Derive (informally) why the OLS coefficient on a randomized binary treatment equals the difference in means.**
A: For Y = α + τD + ε with D∈{0,1} randomly assigned and no other regressors, OLS minimizes squared residuals; the closed-form solution for a single binary regressor is τ̂ = Cov(D,Y)/Var(D). With D binary with mean p, Var(D)=p(1−p). Cov(D,Y) = E[DY] − E[D]E[Y] = p·E[Y|D=1] − p·(p·E[Y|D=1]+(1−p)E[Y|D=0]) = p(1−p)(E[Y|D=1]−E[Y|D=0]). Dividing by Var(D)=p(1−p) gives τ̂ = E[Y|D=1]−E[Y|D=0], exactly the difference in means.

**Q22. What is the "curse of dimensionality" in matching, quantitatively — why does it get so much harder with more covariates?**
A: With k continuous covariates, the volume of a neighborhood needed to find "close" matches grows exponentially with k — the number of data points needed to maintain the same matching quality (average distance to nearest neighbor) scales roughly like n^(1/k) in some sense, meaning even large datasets become sparse in high-dimensional covariate space, and exact/near matches become rare, forcing you to either loosen matching criteria (introducing bias) or drop to a lower-dimensional summary like the propensity score.

**Q23. In a linear model with treatment effect heterogeneity (i.e., true τ_i varies by unit), what does OLS with just D (no interactions) actually estimate?**
A: Under random assignment, OLS-on-D-alone (no covariates) recovers the ATE — the population-average τ_i — regardless of heterogeneity, because it's just a weighted average of the difference in means. But if you add covariate interactions or run a saturated/weighted regression, OLS can implicitly put different weight on different units' treatment effects (a "variance-weighted" average that isn't the simple ATE) — this is a known subtlety (the "OLS weighting" result), so simple specifications are safer for interpretability unless you know what weighting a more complex specification implies.

**Q24. Explain why bootstrapping is often used for confidence intervals in matching/synthetic control settings instead of standard analytic formulas.**
A: Estimators like nearest-neighbor matching or synthetic control involve a discrete, non-smooth optimization step (choosing matches / weights), for which standard asymptotic variance formulas either don't exist cleanly or are known to be invalid/conservative in certain matching regimes (e.g., Abadie-Imbens showed naive bootstrap can fail for some matching estimators specifically, motivating specialized variance estimators — but in many applied settings a well-designed bootstrap or permutation/placebo approach is the pragmatic choice when no clean closed form exists).

**Q25. If Y(1) and Y(0) are both observed for a subset of "always measurable" units (e.g., via a proxy outcome or simulation), how would you validate a causal method before trusting it on real, partially-observed data?**
A: Use it as a *validation/backtest*: apply your causal estimator (matching, IPW, DiD, etc.) to the subset where you can compute the "true" effect directly (both potential outcomes known, or a gold-standard RCT effect available as ground truth), and check whether the observational method recovers a similar estimate. This is essentially what's done in "causal benchmarking" — running an observational method on data from a *known* RCT (ignoring the randomization) to see how close it gets to the RCT's answer, as an empirical credibility check for the method/covariates chosen.

---

## Section 6 — "Explain Like I'm Reviewing Your Work" (Communication)

**Q26. Your director says "just tell me: did the feature work, yes or no?" How do you answer for a borderline result (p=0.08, estimated lift +1.2%, CI [-0.3%, +2.7%])?**
A: I'd avoid a binary yes/no and instead frame it as: "We estimate a +1.2% lift, but we can't rule out a small negative effect — the data are consistent with anything from a slight decline to a solid gain. This isn't strong enough evidence to confidently say it worked, but it's not evidence it *didn't* work either." Then give a recommendation tied to cost/risk: e.g., "given the low cost and no signal of harm, I'd extend the test for more power" or "given rollout cost, I'd want more certainty before shipping."

**Q27. How would you explain "parallel trends" to a product manager with no stats background?**
A: "We're comparing our test market to a similar market. This only works if, absent our change, both markets *would have kept moving the same way* they were before — like two friends walking side by side. If they were already drifting apart before we did anything, we can't cleanly credit our change for any gap afterward."

**Q28. A stakeholder says "the confidence interval includes zero, so there's definitely no effect." How do you correct this, tactfully?**
A: "A CI including zero means we can't statistically distinguish the effect from zero with the data we have — it doesn't mean the true effect *is* zero. It's also consistent with a real, non-zero effect that we just don't have enough power to detect precisely. I'd frame it as 'inconclusive' rather than 'no effect,' and if this matters for the decision, I'd suggest gathering more data before concluding either way."

---

## Cheat List: One-Line Answers for Rapid Recall

- **Fundamental problem of causal inference** → never observe both potential outcomes for the same unit.
- **Why randomize** → breaks D's dependence on potential outcomes, killing selection bias.
- **SUTVA** → one unit's outcome depends only on its own treatment; violated by shared resources/networks.
- **Ignorability** → no unmeasured confounders given X; untestable, argue plausibility.
- **Propensity score** → collapses many confounders into one balancing scalar.
- **IV gives you** → LATE (compliers only), not ATE.
- **DiD needs** → parallel trends; check via pre-trend plots.
- **RDD gives you** → a local effect at the cutoff; check for manipulation/bunching.
- **Synthetic control fits** → the whole pre-treatment trajectory using a weighted donor pool.
- **E-value** → how strong a hidden confounder would need to be to erase your result.
- **CUPED** → pre-experiment covariate adjustment to cut variance, not bias.
- **"CI includes zero"** → inconclusive, not proof of no effect.
