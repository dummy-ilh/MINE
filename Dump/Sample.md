# Google Data Scientist Interview Prep: Stats, Causal Inference & Applied ML

Reference doc covering 12 real, candidate-reported Google DS interview questions across three clusters: statistical foundations, experimentation/causal inference, and applied ML judgment.

---

## Part 1: Statistical Foundations

### 1. "You've built an arbitrary distribution — how do you sample from it?"

**Intuition:** If you know the shape of a distribution but don't have a built-in sampler for it, you need a way to convert draws from something easy to sample (uniform random numbers) into draws that follow your target shape.

**The core techniques, in order of what interviewers want to hear:**

**a) Inverse Transform Sampling** — the default answer.
- If you know the CDF `F(x)`, and `U ~ Uniform(0,1)`, then `X = F⁻¹(U)` follows your target distribution.
- Why it works: the CDF maps any distribution onto a uniform distribution on [0,1]. Running that map backwards takes a uniform sample and "un-flattens" it back into your target shape.
- Worked example: exponential distribution, `F(x) = 1 - e^(-λx)`. Solve for x: `x = -ln(1-U)/λ`. Draw `U`, plug in, done.
- Limitation: only works cleanly when `F⁻¹` has a closed form. Many arbitrary/empirical distributions don't.

**b) Rejection Sampling** — when you don't have an invertible CDF.
- Pick an envelope/proposal distribution `g(x)` that you *can* sample from, and a constant `M` such that `M·g(x) ≥ f(x)` everywhere (f is your target density).
- Sample `x` from `g`, sample `u ~ Uniform(0, M·g(x))`. Accept `x` if `u ≤ f(x)`, else reject and retry.
- Intuition: you're sampling uniformly under the envelope curve, then only keeping points that also fall under your target curve — geometrically this recovers the target shape.
- Tradeoff to mention: efficiency depends entirely on how tight your envelope is. A loose envelope means high rejection rate.

**c) Markov Chain Monte Carlo (MCMC)** — for high-dimensional or unnormalized densities.
- If you only know `f(x)` up to a normalizing constant (very common — e.g., a Bayesian posterior), Metropolis-Hastings or Gibbs sampling let you construct a Markov chain whose stationary distribution *is* your target distribution.
- You don't need the normalizing constant because MH only requires the *ratio* `f(x')/f(x)`.

**How to structure the answer live:** Ask what you actually know about the distribution — closed-form CDF? Only a density, unnormalized? Only empirical samples/histogram? That determines which of the three tools applies. This is the answer Google is fishing for — not reciting all three, but triaging correctly.

**L5-differentiator:** Mention that for empirical/arbitrary distributions given as a histogram or dataset, you can also do **inverse transform on the empirical CDF** directly (interpolate between order statistics), which is simpler than rejection sampling when you literally have the data.

---

### 2. Walk through calculating a t-statistic and a z-statistic

**Z-statistic** — used when population standard deviation σ is known (or n is large enough that sample std is a good stand-in and CLT applies cleanly):

```
z = (x̄ - μ₀) / (σ / √n)
```
- `x̄`: sample mean, `μ₀`: hypothesized population mean, `σ`: population std dev, `n`: sample size.
- Compare against the standard normal distribution.

**T-statistic** — used when σ is unknown and estimated from the sample (the realistic case almost always):

```
t = (x̄ - μ₀) / (s / √n)
```
- Identical formula, but `s` (sample standard deviation) replaces `σ`.
- Compare against the **t-distribution** with `n-1` degrees of freedom, not the normal distribution.

**Why the distinction matters (this is the actual interview signal):** Using `s` instead of `σ` introduces extra uncertainty — you're estimating two things (mean and variance) from the same sample. The t-distribution has fatter tails than the normal to account for that added uncertainty. As `n` grows, the t-distribution converges to the normal, which is why at large n people get sloppy and use z-tests interchangeably with t-tests — technically incorrect at small n, functionally fine at large n.

**Two-sample version** (more likely what they actually want, given the DS context):
```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)     [Welch's t-test, unequal variances]
```
Mention Welch's by name — assuming equal variances (pooled t-test) is a common interview trap since real-world A/B test groups rarely have identical variance.

**Interview trap:** If asked "when do you use z vs t" and you just say "z for large n, t for small n" without mentioning *why* (extra estimation uncertainty from using sample variance), that reads as memorized rather than understood.

---

### 3. What's the difference between MLE and MAP estimation?

**Maximum Likelihood Estimation (MLE):** find the parameter value that makes the observed data most probable, with no prior assumption about what the parameter should be.

```
θ_MLE = argmax_θ  P(Data | θ)
```

**Maximum A Posteriori (MAP):** same idea, but you fold in a prior belief about θ before seeing data, using Bayes' rule.

```
θ_MAP = argmax_θ  P(Data | θ) · P(θ)
       = argmax_θ  P(θ | Data)          [posterior, up to normalizing constant]
```

**The clean way to say it in an interview:** MAP is MLE plus a prior. If your prior is uniform/flat (i.e., you assume nothing), MAP collapses exactly to MLE.

**Concrete worked example — coin flipping:**
- You flip a coin 3 times, get 3 heads. MLE says `p = 1.0` (heads probability) — the data alone says "always heads."
- If you have a prior belief that coins are usually close to fair (say a Beta(10,10) prior), MAP pulls the estimate back toward 0.5, because your prior is skeptical of a 100%-biased coin from just 3 flips.
- This is the single clearest way to explain *why* MAP exists: it regularizes small-sample estimates using outside information.

**Connection interviewers want you to make:** L2 regularization in linear/logistic regression is literally MAP estimation with a Gaussian prior on the weights. L1 regularization corresponds to a Laplace prior. If you can draw this line, it signals you understand regularization isn't just "a penalty term" but has a principled Bayesian interpretation.

**Interview trap:** Don't say "MAP is always better than MLE." With enough data, the likelihood dominates the prior and they converge — MAP mainly helps in low-data or high-variance regimes.

---

### 4. How do you know if your sample size is large enough to trust your conclusions?

This is really a **statistical power** question in disguise.

**Core idea:** "Large enough" means: if a real effect of a given size exists, your sample is big enough to detect it with high probability, not just a coin flip.

**The formal approach — power analysis, done before collecting data:**
You need four ingredients, any three of which determine the fourth:
1. **Significance level (α)** — typically 0.05, your false-positive tolerance.
2. **Power (1-β)** — typically 0.80, your probability of detecting a true effect if it exists.
3. **Minimum detectable effect (MDE)** — the smallest effect size that matters to the business.
4. **Sample size (n)**.

Rough sample size formula for a two-proportion test:
```
n ≈ 16 · σ² / δ²
```
where δ is the MDE and σ² is the outcome variance — this is the standard back-of-envelope Google/tech-company interviewers want you to be able to derive or at least explain the shape of (bigger MDE → smaller n needed; more noise → more n needed).

**Practical checks beyond the formula:**
- Post-hoc: check your confidence interval width. If the CI on your effect is wide relative to the MDE that matters to the business, you're underpowered regardless of what the p-value says.
- Check for adequate **sample ratio** between groups (SRM — sample ratio mismatch) — if your 50/50 split actually landed 52/48, something's wrong upstream (randomization bug), and no amount of "n is big enough" saves you.

**L5-differentiator:** Tie it to the actual business question — "large enough" is meaningless without first defining the MDE that would change a decision. A statistically significant but practically trivial effect (e.g., 0.01% lift found because n=50 million) is the classic trap Google interviewers probe for at senior levels.

---

### 5. (Glassdoor, Paris) How do you compute a p-value with only one sample?

This sounds like a trick question but it's really testing **one-sample hypothesis testing** fundamentals.

**Setup:** You have one sample (not "one data point" — one sample/dataset), and you want to test whether its mean differs from some hypothesized value μ₀.

**Steps:**
1. State H₀: population mean = μ₀. Hₐ: population mean ≠ μ₀ (or one-sided).
2. Compute the one-sample t-statistic: `t = (x̄ - μ₀) / (s/√n)`, using the sample's own mean and standard deviation.
3. Compare t against the t-distribution with n-1 degrees of freedom to get the p-value — the probability of observing a t-statistic this extreme (or more) if H₀ were true.

**The key insight the question is fishing for:** you don't need a second sample to compute a p-value — you only need a *hypothesized reference value* to test against. People sometimes freeze on "one sample" thinking you need two groups to compare, but one-sample tests (comparing sample mean to a fixed benchmark) are standard.

**If the interviewer meant literally one data point (n=1):** then you can't estimate variance from the sample itself, and the honest answer is: you need an assumed or externally known σ (making it a z-test), or a Bayesian approach with an informative prior — with n=1 and unknown variance, classical frequentist inference breaks down since you have zero degrees of freedom to estimate spread.

**How to handle the ambiguity live:** Say both readings out loud and ask which the interviewer means — this is exactly the sort of interviewer-intent clarification that reportedly mattered in the Google 2026 loop for other questions.

---

## Part 2: Experimentation & Causal Inference

### 6. Google Meet: G Suite-only → public. How do you define success? What metrics?

**Framing approach:** Anchor in the objective first, then build a metric tree, then flag risks.

**Step 1 — clarify the goal.** Is success about adoption, engagement, retention, or business impact (e.g., upsell to Workspace)? Ask before assuming.

**Step 2 — primary metrics (North Star candidates):**
- **Adoption:** new signups/activations from non-Workspace users, weekly active meeting hosts.
- **Engagement/depth:** meetings per active user per week, average meeting duration, % of meetings with 3+ participants (signals real collaborative use vs. one-off testing).
- **Retention:** week-over-week or month-over-month retention of new cohort of free users.

**Step 3 — guardrail metrics (things that shouldn't get worse):**
- Existing G Suite/Workspace user engagement (make sure the free tier doesn't cannibalize paid usage or degrade infra performance/call quality for paying customers).
- Call quality metrics (latency, drop rate) under increased load.
- Abuse/spam rate — opening a product to the public introduces bad actors (meeting bombing was a literal real-world issue here).

**Step 4 — business metric:** conversion rate from free → paid Workspace tier over some window, since ultimately the point of opening the funnel is (likely) top-of-funnel growth for Google's paid suite.

**L5-differentiator:** Explicitly separate **vanity metrics** (raw signups) from **quality metrics** (retained, engaged usage) — Google interviewers reportedly push on exactly this kind of metric-tradeoff reasoning.

---

### 7. City-by-city rollout, can't randomize at the user level — how do you measure impact?

**Why this is hard:** Your unit of randomization (city) is different from — and much coarser than — your unit of analysis (user). Standard A/B testing assumes independent, randomly assigned units; here you have a small number of clusters (cities), which breaks that assumption and shrinks your effective sample size to the number of cities, not the number of users.

**Approaches, roughly in order of rigor:**

1. **Difference-in-Differences (DiD):** compare the change in the treated cities before vs. after rollout, against the change in untreated cities over the same period. This nets out both the city-level baseline differences and any global time trend.
   ```
   Effect = (Y_treated,after - Y_treated,before) - (Y_control,after - Y_control,before)
   ```
   Key assumption to state out loud: **parallel trends** — absent treatment, treated and control cities would have moved the same way over time. You'd want to check this with pre-period trend plots.

2. **Synthetic Control:** if you don't have naturally comparable control cities, construct a "synthetic" control city as a weighted combination of many untreated cities that best matches the treated city's pre-period trend. Useful when you have one or very few treated units.

3. **Cluster-robust standard errors:** whatever method you use, you must account for the fact that observations within a city are correlated (not independent) — otherwise you'll understate your standard errors and overstate significance.

**Interview trap:** Don't propose a plain t-test on individual users pooled across cities — that ignores clustering and will make you look overconfident about statistical significance you don't actually have.

---

### 8. Global launch, no holdout group — how do you estimate impact?

**Core problem:** No contemporaneous control means you can't do a clean randomized comparison. You need a way to construct a *counterfactual* — what would have happened without the change.

**Approaches:**
1. **Pre/post comparison with trend adjustment:** compare metrics before and after launch, but explicitly model out seasonality/trend (e.g., using a time-series model or year-over-year comparison) rather than a naive before/after delta, which conflates the treatment effect with any pre-existing trend.
2. **Synthetic control** using other geographies/products/segments that weren't exposed, as in Q7.
3. **Interrupted time series analysis:** fit a model to the pre-period, forecast forward as if the launch never happened, and compare actual post-launch values to that forecast — the gap is your estimated effect.
4. **Retroactive holdout / "hold-back" test:** if it's not too late, carve out a small % of traffic to *not* receive the feature even after "full" launch — a common real-world fix data scientists push for precisely to avoid this problem next time.
5. **Quasi-experiment via staggered rollout:** if the launch wasn't literally simultaneous everywhere (e.g., rolled out region by region, or by app version), you can use the users who got it later as a temporary control group.

**L5-differentiator:** Proactively note that the *real* fix is a process one — advocate for holding back even 1% of traffic on any "full" launch specifically so you retain the ability to measure impact later. This is the kind of forward-looking answer that separates senior candidates.

---

### 9. When would you use difference-in-differences vs. synthetic control?

**DiD** — use when:
- You have a reasonably comparable control group that already exists (similar cities/segments not exposed to treatment).
- Parallel trends assumption is plausible and can be checked with pre-period data.
- You typically have more than one or two treated units (though it works with one treated unit vs one/few controls too).

**Synthetic Control** — use when:
- You have very few (even just one) treated units, and no single existing untreated unit looks like a good comparison on its own.
- You have a longer pre-treatment time series available to construct weights (you need enough pre-period data points to fit the donor pool weights reliably).
- You're worried the parallel-trends assumption doesn't hold for any single natural control, but a *weighted combination* of several controls might track the treated unit's pre-trend well.

**Key distinguishing intuition:** DiD assumes your control group *already* tracks the treated group in expectation (just needs one subtraction to remove baseline offset). Synthetic control doesn't assume any single group is a good match — it *constructs* the best-matching comparison as a weighted blend, using the pre-period data to solve for the weights that best replicate the treated unit's history.

**Interview trap:** Don't say "synthetic control is just better DiD" — they answer different situations. If you have many treated and control units with a plausible parallel-trends story, DiD is simpler, more standard, and easier to communicate to stakeholders. Reach for synthetic control specifically when you lack a natural comparison group.

---

### 10. Observational data, want a treatment effect — how do you approach it?

**Core challenge:** No randomization means treatment and control groups likely differ systematically (confounding) — people who chose the treatment differ from those who didn't in ways that also affect the outcome.

**Approach, as a decision tree:**

1. **Identify confounders** — variables that affect both treatment assignment and outcome. Talk through a DAG (directed acyclic graph) mentally, even if you don't draw one, to reason about what needs to be controlled for.

2. **Matching / Propensity Score Matching:** estimate the probability of receiving treatment given observed covariates (the propensity score), then match treated units to control units with similar propensity scores. This approximates "comparing like with like."

3. **Regression adjustment:** control for confounders directly in a regression model — simplest approach, but only removes confounding from variables you actually measured and included.

4. **Instrumental Variables (IV):** if there's a confounder you *can't* measure, find an instrument — a variable that affects treatment assignment but has no direct effect on the outcome except through treatment. Classic example: distance to a clinic affecting treatment uptake but not health outcomes directly.

5. **Regression Discontinuity (RDD):** if treatment is assigned based on a cutoff of some running variable (e.g., users above a usage threshold get a feature), compare units just above vs. just below the cutoff — locally, this acts like a natural randomization.

6. **DiD / synthetic control** if you have a time dimension, as covered above.

**The one sentence that signals seniority:** "All of these are attempts to approximate the counterfactual we'd get from randomization, and the right one depends on what data and identifying assumptions are actually available" — showing you understand these are all substitutes for the RCT you don't have, each with a different assumption you're betting on.

**Interview trap:** Naming techniques without naming their required assumption (unconfoundedness for matching, exclusion restriction for IV, continuity at the cutoff for RDD) reads as vocabulary without understanding.

---

## Part 3: Applied ML / Product Judgment

### 11. Forecast weekly CPU demand, only 18 months of data, promos/incidents create spikes

**Framing:** This is a time-series forecasting problem complicated by short history and irregular shocks — the answer should walk through model choice, feature engineering, and validation, not just name-drop a model.

**Step 1 — decompose the signal:** trend, weekly/seasonal pattern, and residual spikes from promotions/incidents. With only 18 months (~78 weeks), you likely have at most 1 full annual cycle — call this out explicitly as a limitation, since you can't reliably separate yearly seasonality from a one-time trend with only one year of data.

**Step 2 — handle the spikes, don't let them dominate:**
- Promotions and incident weeks are *known* events (usually), so treat them as **exogenous regressors** rather than noise — include a promo-calendar indicator and incident-flag feature. This lets the model learn "capacity during promo week is baseli
