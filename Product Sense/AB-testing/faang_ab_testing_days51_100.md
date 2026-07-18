# 100-Day A/B Testing Interview Prep
### Days 51–100: L6 Level
### FAANG Interviewer POV | Deep Follow-ups | Senior/Staff Complexity

> At L6, the interviewer expects you to:
> - Design systems, not just run tests
> - Identify failure modes before they're pointed out
> - Make judgment calls under ambiguity
> - Connect statistics to business strategy
> - Know what you don't know — and what to do about it

---

# DAYS 51–100: L6 LEVEL
*Systems thinking, causal inference, platform design, org influence*

---

## Day 51 — Experiment Platform Design

🎯 **"Design an experimentation platform from scratch for a company with 50M daily users."**

💬 **Model Answer**
Core components:

**1. Assignment Service**
- Hash-based: hash(user_id + experiment_id + salt) mod 10000 → bucket
- Deterministic, low-latency (<1ms), stateless
- Supports user/session/device/group level randomization
- Orthogonal layering for concurrent experiments

**2. Feature Flag System**
- Decoupled from deployment — ship code, control exposure separately
- Supports gradual rollout (1% → 10% → 50% → 100%)
- Real-time kill switch per experiment

**3. Event Logging & Attribution**
- Experiment assignment logged at exposure, not at assignment
- Joins assignment log with event log on user_id + timestamp
- Handles late-arriving events (impression → conversion hours later)

**4. Metric Pipeline**
- Pre-computes per-user aggregates daily
- Supports custom metric definitions via SQL or config
- SRM check runs automatically on ingestion

**5. Analysis Engine**
- Runs pre-specified tests: z-test, t-test, CUPED, bootstrap
- Sequential testing for monitoring-safe dashboards
- Outputs: point estimate, CI, p-value, power, SRM status

**6. Decision & Governance Layer**
- Pre-registration enforced before flag is enabled
- Peer review queue for design and results
- Audit log: who made what decision when, with what evidence

↳ **"How do you handle experiment overlap — two experiments on the same user?"**
Orthogonal layering: divide the experiment space into layers. Within a layer, users get one experiment. Across layers, assignments are independent (different salts). Experiments that touch the same surface must be in the same layer (and therefore mutually exclusive) or proven to be non-interacting.

↳ **"What's your biggest scaling concern at 50M DAU?"**
Assignment service latency — must be sub-millisecond and globally consistent. Cache experiment configs at the edge. Use consistent hashing to avoid assignment drift if configs change. Log at the right granularity — logging every page view for 50M users is petabytes/day; log only on first exposure per experiment per session.

🚩 **Red Flag**
"Use a third-party tool like Optimizely." — Fine for small companies, but L6 at FAANG must understand the full system design.

⭐ **What earns the role**
Identifying that the hardest problem isn't statistics — it's the attribution join (matching assignment to outcomes that happen hours or days later), especially when users span devices and sessions.

---

## Day 52 — Always-Valid Inference

🎯 **"Your company wants a live dashboard showing experiment results. How do you make it statistically valid?"**

💬 **Model Answer**
Standard frequentist CIs are only valid at the pre-specified sample size — peeking inflates false positives. Three valid approaches for continuous monitoring:

**1. Always-Valid Confidence Sequences (AVCS)**
Based on martingale theory. Provides CIs that are valid at every sample size simultaneously — if the true effect is zero, the CI contains zero at every peek with at least 1-α probability. Used at Netflix (Lindon et al.) and Spotify.

**2. Alpha Spending Functions (Group Sequential)**
Pre-plan a fixed number of interim looks. Allocate the α budget across looks using spending functions (O'Brien-Fleming: conservative early, liberal late; Pocock: equal α at each look). Requires pre-specifying looks upfront.

**3. E-values / E-processes**
Game-theoretic approach to hypothesis testing. E-values can be multiplied across sequential data without inflating error. Composable, interpretable as "evidence against H₀."

Dashboard design: show always-valid CIs (wider than fixed-horizon CIs). Add clear messaging: "Do not use for launch decisions before [date]." Show guardrail metrics on standard monitoring — anomaly detection, not significance testing.

↳ **"How do AVCS differ from standard CIs in width?"**
AVCS are wider — they must be valid at all sample sizes including small ones, so they're more conservative. As n grows, they narrow and approach standard CIs. The price of anytime validity is slightly wider intervals.

↳ **"Stakeholder asks: 'Why can't I just use the p-value on the dashboard to decide?' How do you explain?"**
The p-value displayed at 10K users is not the same as the p-value you'd get at 100K users. If you stop when it crosses 0.05, you're not running a 5% false positive rate test — you might be running a 30% false positive rate test. AVCS guarantees that no matter when you stop, you're still controlling the error rate.

🚩 **Red Flag**
"We just show the p-value with a warning not to use it." — Humans can't resist acting on significant p-values they see in real time.

⭐ **What earns the role**
Knowing the mathematical basis: AVCS are based on the optional stopping theorem — a valid test statistic must be a martingale under H₀, so stopping at any stopping time preserves the Type I error guarantee.

---

## Day 53 — Causal Forests & HTE at Scale

🎯 **"How do you identify which users benefit most from a feature, at scale?"**

💬 **Model Answer**
This is heterogeneous treatment effect (HTE) estimation. At scale:

**1. Causal Forests (Wager & Athey, 2018)**
Extension of random forests for causal inference. Estimates individualized treatment effects τ(x) = E[Y(1) - Y(0) | X = x]. Automatically discovers which features drive HTE without pre-specifying segments. Honest estimation (uses separate subsamples for splitting and estimation) gives valid confidence intervals.

**2. T-Learner / S-Learner / X-Learner**
- S-Learner: one model on (X, W) → predict outcome; treatment effect = difference when W=1 vs W=0
- T-Learner: separate models for control and treatment; τ(x) = μ₁(x) - μ₀(x)
- X-Learner: uses T-Learner estimates to impute counterfactuals, then trains a model on the imputed effects. Better for unequal group sizes.

**3. Double ML (Chernozhukov et al.)**
Partialing out confounders using flexible ML. Valid for observational data as well.

↳ **"What do you do with HTE estimates once you have them?"**
(1) **Targeting:** Only ship the feature to users with τ(x) > threshold (expected to benefit). (2) **Personalization:** Serve different variants to different user segments. (3) **Policy learning:** Optimize a treatment assignment rule. (4) **Insight generation:** What user features drive the HTE? Informs product strategy.

↳ **"What's the risk of acting on HTE estimates?"**
Estimated HTEs are noisy. If you only target users where τ̂(x) > 0, you're selecting on noise — some of those users may actually be harmed. Requires holdout validation: run a new experiment targeting only the "high benefit" segment to confirm the HTE.

🚩 **Red Flag**
"We just segment by device/age/location manually." — L6 at FAANG works with 100s of features; manual segmentation misses interactions.

⭐ **What earns the role**
Knowing that HTE methods require careful validation — you can't evaluate a causal model the way you evaluate a predictive model (no ground truth for individual treatment effects). Use held-out experiment arms or CATE validation methods (AIPW, DR-learner).

---

## Day 54 — Interference at Scale

🎯 **"You're at LinkedIn. You test a feature that shows users how many of their connections work at a company they're viewing. How does this create interference?"**

💬 **Model Answer**
Direct interference: if user A (treatment) can now see that user B works at Acme Corp, user B's profile gets more views — regardless of whether B is in treatment or control. B's behavior (feeling more discoverable, responding more to recruiter messages) is affected by A's treatment.

At LinkedIn's scale, nearly every user is connected to users in both treatment and control — the contamination is pervasive. Standard user-level A/B test would severely underestimate the true treatment effect (control users benefit from their treated connections).

Solutions:
1. **Ego-network randomization:** Assign by social cluster. Users in the same cluster get the same assignment. Minimize cross-cluster edges.
2. **Two-stage randomization:** Randomly select clusters; within clusters, randomize users. Estimates direct + indirect effects separately.
3. **Graph cluster randomization:** Partition the social graph into clusters with minimal between-cluster edges, then randomize at cluster level.

↳ **"How do you measure the network effect itself?"**
Compare outcomes at the cluster boundary: users with many treated neighbors vs users with few treated neighbors (both in "control"). The difference is the spillover effect. Can also use the Horvitz-Thompson estimator for network experiments.

↳ **"What's the statistical cost of cluster randomization?"**
Fewer independent units → much lower power. If you have 100M users but only 10K clusters, your effective sample size is closer to 10K. You need much larger effects to detect or run much longer experiments.

🚩 **Red Flag**
"Social networks are hard so we just accept the bias." — At L6, you're expected to propose viable solutions, not accept limitations.

⭐ **What earns the role**
Knowing that network interference *always* biases toward underestimating the treatment effect — control users are partially treated via their network neighbors. This means A/B tests at social companies are systematically conservative.

---

## Day 55 — Instrumentation & Observability

🎯 **"Your experiment shows inconsistent results across two data pipelines. How do you debug?"**

💬 **Model Answer**
First, establish ground truth: which pipeline should be authoritative? Then systematically compare:

1. **Event volume:** Do both pipelines receive the same number of events? If pipeline A has 2% fewer events than pipeline B for the same users, there's a logging discrepancy.
2. **Assignment consistency:** Do both agree on which users are in control vs treatment? Pull a sample of user IDs and compare assignments.
3. **Join logic:** How each pipeline joins assignment logs to outcome events — different timestamp logic, different session definitions, or different deduplication can cause divergence.
4. **Latency handling:** Late-arriving events (conversions that happen days after exposure). If pipelines cut off differently, results diverge.
5. **Deduplication:** Does either pipeline deduplicate events differently? A/B click events may be logged multiple times.

↳ **"What's the most common cause of pipeline divergence in practice?"**
Different attribution windows. Pipeline A joins conversions within 24 hours of exposure; pipeline B within 7 days. This alone causes huge divergence for metrics with delayed conversions (purchases, subscriptions).

↳ **"How do you prevent this systematically?"**
Canonical metric definitions stored in a central metric registry. All pipelines use the same SQL/config to compute each metric. Any deviation is flagged in a metric consistency check that runs daily.

🚩 **Red Flag**
"Use whichever pipeline shows the better result." — This is p-hacking at the infrastructure level.

⭐ **What earns the role**
Proposing a metric governance system: before any experiment, the metric definition (including attribution window, deduplication logic, filtering rules) is frozen and versioned. Changes to metric definitions require re-approval.

---

## Day 56 — Variance Reduction at Scale

🎯 **"Beyond CUPED, what other variance reduction techniques exist and when do you use each?"**

💬 **Model Answer**

**1. CUPED** (Controlled-experiment Using Pre-Experiment Data)
Best when: strong pre-experiment covariate available. Reduces variance by 1-ρ².

**2. Stratified sampling**
Pre-assign users to strata (device, country), ensure balanced representation in each arm. Reduces variance from stratum-level differences. Best for known discrete confounders.

**3. Post-stratification**
Same as stratified sampling but applied at analysis time. Works even if randomization wasn't stratified.

**4. Regression adjustment (ANCOVA)**
Include pre-experiment covariates in a regression model. More flexible than CUPED — handles multiple covariates simultaneously, non-linear relationships.

**5. Paired experiments**
Match users in control and treatment on pre-experiment characteristics. Compare pairs directly. Very powerful but expensive to implement.

**6. Outlier capping / winsorization**
Reduces variance from heavy-tailed distributions. Easy to implement, highly effective for revenue metrics.

**7. Metric transformation**
Log-transform or rank-transform the metric. Reduces skewness → lower variance → more power.

↳ **"CUPED reduces variance. Does it change the point estimate?"**
No — in expectation. The CUPED adjustment is unbiased because the covariate X is pre-experiment and therefore orthogonal to treatment assignment (by randomization). The point estimate may shift slightly in practice due to finite-sample correlation, but the expected value of the estimate is unchanged.

↳ **"When does CUPED fail?"**
When the pre-experiment metric is unavailable (new users, first-time events), when ρ is low (covariate weakly correlated with outcome), or when the pre-period is itself noisy or too short.

🚩 **Red Flag**
"We just increase sample size to handle variance." — Variance reduction is often more efficient and faster than waiting for more data.

⭐ **What earns the role**
Knowing that CUPED, regression adjustment, and stratification are all special cases of the general framework: any variable correlated with the outcome but orthogonal to treatment can reduce variance without introducing bias — by Frisch-Waugh-Lovell theorem.

---

## Day 57 — Synthetic Control

🎯 **"You want to measure the impact of launching in Brazil, but you can't run an A/B test. How do you use synthetic control?"**

💬 **Model Answer**
Synthetic control (Abadie, Diamond, Hainmueller 2010):

**Setup:** Brazil is the treated unit. We have pre-treatment data on Brazil and several control countries (Argentina, Colombia, Chile, Mexico...).

**Method:**
1. Find a weighted combination of control countries that best matches Brazil's pre-launch trajectory on the outcome metric (minimize MSE in pre-period).
2. Weights: non-negative, sum to 1. Solved via constrained optimization.
3. Post-launch: compare Brazil's actual outcome to the synthetic Brazil (counterfactual).
4. Treatment effect = Brazil_actual − synthetic_Brazil in the post-period.

**Inference:** Use permutation/placebo tests — apply synthetic control to each control country (as if they were treated) and compare the fit. Brazil's effect is significant if it's an outlier relative to the placebo distribution.

↳ **"What's the key assumption?"**
The synthetic control must fit Brazil's pre-period well. If it doesn't (MSE is high in pre-period), the counterfactual is unreliable. Also: no spillover from Brazil to control countries (SUTVA at country level).

↳ **"When does synthetic control break down?"**
(1) Poor pre-period fit — no good donor pool. (2) Too few pre-period time points. (3) Treated unit is very different from all controls (Brazil is unique). (4) Global events affect all countries post-launch (can't separate treatment from event).

🚩 **Red Flag**
"Compare Brazil post-launch to a similar country pre-launch." — This is just a cross-sectional comparison with no causal validity.

⭐ **What earns the role**
Knowing the inference approach: permutation/placebo tests are synthetic control's equivalent of p-values — valid even with one treated unit (N=1), which is where standard frequentist inference breaks down.

---

## Day 58 — Regression Discontinuity

🎯 **"How would you use regression discontinuity to measure the effect of a loyalty program that gives benefits at 100 purchases?"**

💬 **Model Answer**
RDD exploits the sharp cutoff: users with exactly 99 purchases are nearly identical to users with exactly 100 purchases — the only systematic difference is which side of the cutoff they're on (no vs yes benefits). We compare outcomes just below and just above the threshold.

**Implementation:**
1. Running variable: number of purchases
2. Cutoff: 100
3. Treatment: receives loyalty benefits (1 if purchases ≥ 100, else 0)
4. Outcome: retention, next-month purchases, LTV
5. Estimate: local linear regression on each side of the cutoff, estimate discontinuity at the threshold

**Key assumption:** No manipulation around the cutoff. Users shouldn't be able to precisely control whether they're at 99 or 100 purchases. Test: density of running variable should be smooth at the cutoff (McCrary density test).

↳ **"What's the limitation of RDD?"**
External validity: the estimate is valid only for users near the cutoff (local average treatment effect at the threshold). Users with 50 purchases might respond very differently to loyalty benefits than users at 100. Can't generalize far from the cutoff.

↳ **"What's a fuzzy RDD?"**
When the cutoff determines probability of treatment, not certainty. E.g., users above 100 purchases are *recommended* for benefits but don't always receive them. Handled with instrumental variables — the cutoff is the instrument.

🚩 **Red Flag**
"Compare users above 100 purchases to users below 100." — Without restricting to a bandwidth around the cutoff, you're comparing very different users.

⭐ **What earns the role**
Choosing bandwidth optimally (e.g., MSE-optimal bandwidth selector from Calonico, Cattaneo, Titiunik) and validating with covariate balance just below/above the cutoff.

---

## Day 59 — Instrumental Variables

🎯 **"You want to measure the effect of a user watching a tutorial video on retention, but watching is self-selected. How do you use IV?"**

💬 **Model Answer**
Self-selection problem: users who watch tutorials are different (more motivated, less confused) — we can't compare them to non-watchers. Instrumental variable approach:

**Find an instrument Z:** A variable that (1) affects whether the user watches the tutorial (relevance) and (2) only affects retention through watching the tutorial, not directly (exclusion restriction).

Example instrument: random variation in tutorial prompt placement — users who see the tutorial button at the top of the screen (vs buried in menu) are more likely to watch, but the button position itself doesn't affect retention except through tutorial watching.

**Estimation:**
1. First stage: regress tutorial_watched on button_position → get ŷ
2. Second stage: regress retention on ŷ → IV estimate of tutorial effect

This gives the LATE (Local Average Treatment Effect) — effect on compliers (users who watch only because of the nudge, not inherently motivated).

↳ **"What's the exclusion restriction and why is it untestable?"**
The instrument must affect the outcome only through the treatment — no direct effect. This is an assumption, not testable from data. You can support it with domain knowledge and falsification tests (does the instrument predict outcomes in a group where the treatment has no effect?).

↳ **"When does IV give biased estimates?"**
Weak instruments (low first-stage F-statistic < 10) → finite sample bias toward OLS, inflated variance. Always report first-stage F-statistic.

🚩 **Red Flag**
"Just control for user motivation in a regression." — Unobserved motivation is the problem; you can't control for what you can't measure.

⭐ **What earns the role**
Knowing that IV estimates LATE, not ATE — results generalize to compliers, not all users. Whether that's the policy-relevant population depends on the question.

---

## Day 60 — Observational Causal Inference

🎯 **"You have no randomization. How do you estimate causal effects from observational data?"**

💬 **Model Answer**
Hierarchy of methods by assumption strength:

**1. Matching & Propensity Score**
Match treated users to similar untreated users on observable covariates. Propensity score: P(treated | X) — balance on the score balances all observed covariates (Rosenbaum-Rubin). Assumes no unobserved confounders (strong ignorability).

**2. Inverse Probability Weighting (IPW)**
Reweight observations by 1/P(treated|X) to create a pseudo-population where treatment is independent of X. Doubly robust when combined with outcome model.

**3. Doubly Robust / AIPW**
Combines propensity score model and outcome model. Consistent if *either* model is correctly specified — more robust than either alone.

**4. Difference-in-Differences**
Controls for time-invariant confounders. Requires parallel trends.

**5. Instrumental Variables**
Controls for unobserved confounders with a valid instrument.

**6. Regression Discontinuity**
Local randomization near a threshold.

↳ **"What's the fundamental problem of causal inference?"**
The fundamental problem (Holland 1986): we can never observe both Y(0) and Y(1) for the same unit simultaneously. Causal inference is always about estimating a counterfactual. All methods make assumptions to fill this gap — the question is which assumptions are most credible in context.

↳ **"How do you choose between these methods?"**
Match method to data structure: DiD when panel data with parallel trends. RDD when sharp threshold. IV when valid instrument exists. Matching/IPW when rich observables and no instruments. Always prefer the method with the weakest (most credible) assumptions.

🚩 **Red Flag**
"Add confounders as controls in a regression." — Works only if all confounders are observed and the functional form is correct. Misses unobserved confounders entirely.

⭐ **What earns the role**
Knowing that no observational method solves the fundamental problem — they all make untestable assumptions. The job is to be transparent about which assumptions you're making and how sensitive results are if those assumptions are violated (sensitivity analysis).

---

## Day 61 — Metric Design at System Level

🎯 **"Design a metric system for an e-commerce company that connects experiment metrics to business OKRs."**

💬 **Model Answer**
**Metric hierarchy:**

**North Star:** Long-term revenue per user (LTV proxy)
↓
**L1 — Driver metrics:** Acquisition rate, purchase frequency, average order value, retention rate
↓
**L2 — Leading indicators:** CTR, add-to-cart rate, checkout completion, search-to-purchase
↓
**L3 — Experiment metrics:** Feature-specific metrics (e.g., recommendation CTR, search result diversity)

**Causal chain:** L3 metrics should be shown (via historical validation) to predict L2 metrics, which predict L1, which predict the North Star. This is surrogate metric validation — each level is a causal proxy for the level above.

**Guardrails at each level:** page load time, error rate, customer support volume, return rate.

↳ **"Why not just measure LTV in every experiment?"**
LTV takes months to observe. Experiments would need to run 6–12 months — impractical. L3 metrics are fast (measurable in days), but only valid if causally connected to LTV via validated surrogates.

↳ **"How do you validate that a surrogate metric predicts the North Star?"**
Historical validation: across past experiments, does improvement in the surrogate predict improvement in LTV? Requires a library of past experiments with both short-term and long-term outcomes. This is why maintaining experiment archives with long-term followups is valuable.

🚩 **Red Flag**
"We optimize for CTR because it's easy to measure." — Without validating that CTR predicts revenue, you may be optimizing for clicks while degrading purchase intent.

⭐ **What earns the role**
Proposing a formal surrogate validation protocol, inspired by the oncology concept of validated surrogate endpoints — a surrogate is valid if the treatment effect on the surrogate fully mediates the treatment effect on the true outcome.

---

## Day 62 — Experiment Interaction Effects

🎯 **"Two experiments are running simultaneously. Experiment A tests a new search ranking. Experiment B tests a new checkout button. Can they interact?"**

💬 **Model Answer**
Maybe — depends on whether they share a surface or user journey. Search ranking and checkout button are on different pages and unlikely to interact mechanically. However:

**Interaction check:**
1. **Same user journey:** Do users who interact with the search ranking also reach checkout? If search determines what they buy, and checkout determines whether they buy, the experiments are connected.
2. **Metric interaction:** If the primary metric for both is conversion rate, and the search experiment changes who reaches checkout, the checkout experiment's treatment population is different than expected.
3. **Statistical interaction:** Run a 2×2 factorial analysis — users are bucketed into one of four groups (A-control/B-control, A-treatment/B-control, A-control/B-treatment, A-treatment/B-treatment). Test whether the interaction term is significant.

↳ **"If there is a significant interaction, what do you do?"**
If you're running them in separate layers (orthogonal), the overall average effect of each experiment is still unbiased — the interaction averages out. But if you want to know the effect of A for users also in B-treatment (to plan a joint launch), you need the factorial estimate.

↳ **"How does Google handle thousands of simultaneous experiments?"**
OEI (Overlapping Experiment Infrastructure): experiments are layered. Within each layer, users see one experiment. Across layers, assignments are independent. Experiments on the same surface must share a layer (mutually exclusive). Experiments on different surfaces are in different layers (can overlap).

🚩 **Red Flag**
"Experiments on different pages can't interact." — Interaction occurs through the user journey and shared metrics, not just shared UI surfaces.

⭐ **What earns the role**
Knowing that the orthogonal layering design ensures average treatment effects remain unbiased even in the presence of interactions — but joint effects and personalized policies require factorial analysis.

---

## Day 63 — Power in Low-Traffic Settings

🎯 **"You're at a B2B SaaS company. You have 500 enterprise accounts, not 50M users. How do you experiment?"**

💬 **Model Answer**
Traditional A/B testing requires large n for power — 500 accounts is severely limited. Adapted approaches:

**1. Account-level randomization**
Randomize accounts (not users within accounts). Reduces SUTVA violations within accounts. With 500 accounts, expect very low power for detecting small effects.

**2. Bayesian methods**
With informative priors from similar past experiments, Bayesian approaches extract more signal from small samples. Results expressed as posterior probability, not p-values.

**3. Pre/post designs with matched pairs**
Match accounts on size, industry, usage history. Pair them, assign one to treatment, compare change from baseline. Paired t-test has higher power than two-sample t-test.

**4. Within-account experiments**
Randomize at user level within accounts. Higher n, but SUTVA may be violated (users in same account interact, share workflows).

**5. Sequential rollout analysis**
Roll out to accounts in batches, treat batch timing as quasi-random. DiD between early and late adopters.

**6. Increase MDE**
Accept that you can only detect large effects (>15–20% relative lift). Adjust roadmap expectations accordingly.

↳ **"How do you communicate this limitation to stakeholders?"**
Show the power curve: "With 500 accounts, we can detect a 20% relative lift with 80% power. We cannot reliably detect 5% lifts. Our results are directionally informative but may miss real improvements." Set expectations before experiments run, not after.

🚩 **Red Flag**
"Just run the standard A/B test." — With 500 accounts, standard tests will be chronically underpowered, producing misleading null results.

⭐ **What earns the role**
Designing a portfolio approach: use small-n experiments for directional signal, combine with qualitative research (user interviews, session recordings), and use larger customers as natural experiments when features are adopted unevenly.

---

## Day 64 — Experiment Cannibalization

🎯 **"Your new recommendation feature improves CTR on recommendations but reduces organic search usage. How do you think about this?"**

💬 **Model Answer**
This is a cannibalization problem — the feature may be moving engagement from one surface (search) to another (recommendations) rather than growing total engagement. Net effect on the user and business could be zero or negative, even though the feature-specific metric looks good.

**Analysis:**
1. Measure total sessions, total page views, total engagement — not just per-surface metrics.
2. Check search volume in treatment vs control — does it decline?
3. Compute total revenue per user (not per surface) — does the product mix shift toward lower-margin items (recommendations) from higher-margin (search-intent purchases)?

**Framework:** Any time a surface-specific metric improves, ask: is this incremental or substitution? Incremental = new behavior on top of existing. Substitution = same behavior on a different surface.

↳ **"When is substitution acceptable?"**
When the new surface provides a better user experience for the same task (users prefer recommendations to search for discovery). The goal is user value, not surface-level metrics. If total satisfaction and retention are flat or improving — substitution is fine.

↳ **"How do you design metrics to catch cannibalization?"**
Always include "total" metrics alongside surface-specific metrics in the experiment: total sessions, total revenue per user, total content consumed. Surface metrics are intermediate; total metrics are the outcome.

🚩 **Red Flag**
"Recommendation CTR went up — great success." — Celebrating surface metrics without checking total impact.

⭐ **What earns the role**
Connecting to portfolio thinking: the product is a portfolio of surfaces. Optimizing one surface at the expense of others is a local maximum, not a global one.

---

## Day 65 — Structural Breaks & Seasonality

🎯 **"An experiment runs from November 15 to December 15. How do you handle holiday seasonality?"**

💬 **Model Answer**
Running an experiment entirely in a seasonal period creates two problems:
1. **External validity:** Results may not generalize to non-holiday periods. Users in holiday mode buy differently, engage differently.
2. **Confounding with time:** If the holiday effect is non-uniform across control and treatment (e.g., one variant is better suited to gift purchases), the holiday inflates the treatment effect.

**Mitigations:**
1. **Run pre/during/post:** Start the experiment before the holiday and run it through. If feasible, compare results in each period to see if the effect differs.
2. **Holiday-aware sample size:** Holiday weeks have higher variance (higher revenue spikes). Adjust sample size accordingly.
3. **Segment by purchase intent:** Holiday gift buyers are a distinct segment — analyze separately.
4. **Report results with caveat:** "This result reflects holiday-period behavior. Expected effect in normal periods may be [lower/different]."

↳ **"Would you delay the experiment to avoid the holiday entirely?"**
Depends on the product. For a holiday-specific feature (gift wrapping option), run it during the holiday — that's the target context. For a general feature, delaying is better unless the holiday creates natural urgency for the launch decision.

↳ **"How do you detect if seasonality is confounding your results?"**
Compare day-of-week and week-by-week treatment effects. If the effect is only present in the last 2 weeks of the experiment (holiday rush), suspect seasonality-treatment interaction. Compute the experiment effect separately for holiday vs non-holiday days.

🚩 **Red Flag**
"Seasonality is just noise — randomization handles it." — Randomization controls for between-group differences; it doesn't eliminate seasonality as a within-period confounder if the seasonal effect is non-uniform across variants.

⭐ **What earns the role**
Knowing that time-varying treatments and seasonal experiments require caution about external validity — the experiment is internally valid but the effect size may not generalize to other time periods.

---

## Day 66 — Experiment ROI

🎯 **"A VP asks: 'Is our experimentation program worth the investment?' How do you answer?"**

💬 **Model Answer**
Frame it as a decision-quality improvement problem, not just cost-benefit:

**Quantifiable value:**
1. **Wins shipped:** Revenue attributed to experiments that showed positive results × % of roadmap driven by experiments.
2. **Bad launches prevented:** Estimate how many experiments showed significant negative effects (and therefore prevented bad launches). What would those launches have cost at full rollout?
3. **Velocity:** How much faster do we iterate vs competitors who don't experiment?
4. **Compounding:** Each experiment generates learning that informs the next. The value is not linear — it compounds.

**Example framing:** "Last year, we ran 300 experiments. 40% showed positive results (shipped). 15% showed significant regressions (prevented). Assuming prevented regressions would have each cost $X in lost revenue, and shipped experiments generated $Y in aggregate lift, the net value of the program is $Z."

↳ **"How do you calculate the value of a prevented bad launch?"**
Estimate the negative effect size × exposed user base × duration it would have run before being caught via post-launch monitoring. Usually: weeks to months of degraded metrics before an org without experiments catches a silent regression.

↳ **"What's the cost of the experimentation program?"**
Engineering hours to build/maintain the platform, data infrastructure costs, experiment design time, analysis time. Typically dominated by engineering costs. For large companies: $5–50M/year for a mature platform. Compare to value generated.

🚩 **Red Flag**
"We can't measure the value of experimentation." — This is a cop-out. Every component of value is estimable.

⭐ **What earns the role**
Proposing a holdout study of the experimentation program itself: identify a set of decisions that went through experimentation vs a matched set that didn't. Compare outcomes. This is the meta-experiment.

---

## Day 67 — Interference: Marketplace Equilibrium

🎯 **"You work at DoorDash. You test a new dispatch algorithm that routes orders more efficiently. Why is user-level A/B testing inadequate?"**

💬 **Model Answer**
The dispatch algorithm operates at the market level — it allocates delivery capacity across all active orders simultaneously. If 50% of orders use the new algorithm, it doesn't operate in isolation: it's competing with the old algorithm for the same pool of dashers. The algorithms interact with each other and with the shared resource (dashers).

The observed effect is not the effect of fully deploying the new algorithm — it's the effect of running it in a mixed market. The true effect (all orders using new algorithm) could be significantly different, potentially better (full optimization) or worse (algorithm optimized for a homogeneous environment but degraded in a hybrid one).

↳ **"What's the right experimental design?"**
Geo-based or switchback experiments. Assign entire markets to one algorithm or the other. Each market operates as a unified system. Compare market-level outcomes (average delivery time, dasher utilization, order cancellation rate).

↳ **"How do you handle the low power of geo experiments here?"**
(1) Use many markets with small effect sizes and power via t-test at market level. (2) Use synthetic control if markets are few. (3) Use regression adjustment on market-level covariates (market size, density, historical delivery time) to reduce variance. (4) Run the experiment longer to accumulate market-level observations.

🚩 **Red Flag**
"We randomize at the order level." — Creates the mixed-market problem described above.

⭐ **What earns the role**
Understanding that marketplace algorithms have equilibrium effects — the steady-state performance of the algorithm in a fully deployed environment is qualitatively different from its performance in a partial deployment. This is why the treatment effect in a geo experiment is the policy-relevant estimand, not the individual-level ATE.

---

## Day 68 — Metric Decomposition

🎯 **"Revenue per user declined 3% in an experiment. How do you decompose this to understand why?"**

💬 **Model Answer**
Revenue per user = (% users who purchase) × (avg order value among purchasers)
= Conversion rate × AOV

Further:
- Conversion rate = traffic × click rate × add-to-cart rate × checkout rate
- AOV = avg items per order × avg item price

Decompose each component in treatment vs control:
1. Did fewer users purchase (lower conversion rate)?
2. Did purchasers buy cheaper items (lower AOV)?
3. Did purchasers buy fewer items per order?
4. Was it specific product categories (mix shift)?

This is the "variance decomposition" of the revenue metric — attributing the aggregate change to its components.

↳ **"Revenue is down because AOV dropped 5% but conversion is flat. Now what?"**
Users are buying, but buying cheaper. Could mean: (1) The feature surfaces cheaper alternatives prominently. (2) The feature attracts price-sensitive users to complete purchases they'd otherwise abandon (could be good — lower AOV but more volume). (3) The feature degrades product discovery for premium items. Dig into category-level data.

↳ **"When is a revenue decline acceptable?"**
If it's accompanied by a retention improvement — lower short-term revenue, higher LTV. Or if it democratizes access (lower AOV from more diverse customers, not just fewer sales). Always connect revenue change to user value, not just business metrics.

🚩 **Red Flag**
"Revenue went down, so the experiment failed." — Without decomposition, you don't know how to fix it or whether it's actually a problem.

⭐ **What earns the role**
Proposing a pre-experiment metric decomposition: document the hypothesized causal chain before the experiment, then verify each link post-experiment. Unexpected decompositions reveal new product insights.

---

## Day 69 — Switchback Design Deep Dive

🎯 **"Design a switchback experiment for Uber surge pricing in NYC."**

💬 **Model Answer**
**Design:**
- Unit: NYC market (city-wide, not borough — surge is interdependent across boroughs)
- Time windows: 1-hour windows (shorter → more observations, longer → less carryover)
- Schedule: Alternating windows over 4 weeks (treatment hour, control hour, treatment hour...)
- Balance: Equal number of treatment and control windows for each hour-of-day (control for time-of-day effects)
- Randomization: Which hour-of-day gets treatment first is randomized (to prevent systematic confound with time)

**Analysis:**
- Unit of analysis: 1-hour window
- Covariates: hour-of-day, day-of-week, weather, local events
- Model: mixed-effects regression with window as observation, market as random effect (if multiple cities)
- Carryover: include prior window's treatment as a lagged covariate; estimate and subtract carryover

**Metrics:** Rider wait time, driver earnings, order completion rate, surge multiplier distribution

↳ **"How do you estimate and correct for carryover?"**
Model carryover explicitly: in the regression, include a dummy for "prior window was treatment." If the coefficient is significant, carryover exists. Correct by either: (a) removing the first window after a switch, (b) including the carryover term in the model, (c) using longer windows to allow equilibration.

↳ **"How do you determine the right window length?"**
Empirically: measure how long it takes for the market to reach equilibrium after a treatment switch. Plot key metrics (driver availability, request rate) as a function of time since the switch. The equilibration time is your minimum window length.

🚩 **Red Flag**
"Run switchback for 1 day to get results." — Too few windows → very low power. Need 4+ weeks of hourly windows to get adequate observations.

⭐ **What earns the role**
Knowing that the analysis must account for autocorrelation in outcomes across adjacent time windows (market state in hour T+1 is correlated with hour T) — requires clustered or robust standard errors, not independent-observations t-test.

---

## Day 70 — Sensitive Metrics & Privacy

🎯 **"You need to measure the effect of a feature on user mental health proxies (e.g., time-of-day usage, session frequency). How do you handle privacy?"**

💬 **Model Answer**
Mental health proxies from behavioral data (late-night usage, weekend isolation patterns) are sensitive — even aggregate signals can be stigmatizing or re-identifying.

**Privacy-preserving approaches:**
1. **Differential privacy (DP):** Add calibrated noise to aggregate statistics such that individual-level data cannot be inferred. Trade-off: lower statistical precision.
2. **k-anonymity:** Ensure each reported segment contains at least k users (e.g., k=50) — prevents re-identification.
3. **Aggregation only:** Never compute individual-level sensitive features in the experiment system; only aggregate metrics (e.g., % of users with late-night sessions) are logged.
4. **On-device computation:** Compute sensitive features on-device and only send aggregated, noised statistics to the server (federated analytics).

**Ethical framework:**
- Institutional review before the experiment
- User consent for mental-health-related measurements
- Limit data retention
- Transparency report

↳ **"Does DP significantly reduce statistical power?"**
Yes — DP adds noise proportional to sensitivity/ε. For small ε (strong privacy), noise can overwhelm real effects. For large user populations (millions), the noise is small relative to signal. Design the experiment with sufficient n to absorb DP noise.

↳ **"How do you measure wellbeing effects without sensitive proxies?"**
Use validated survey instruments (PHQ-2 for depression screening, SWLS for life satisfaction) sampled from a subset of users. Self-reported measures are noisier but direct. Combine with behavioral proxies for triangulation.

🚩 **Red Flag**
"We aggregate the data so it's fine." — Aggregation alone doesn't prevent inference attacks. DP is the only method with formal privacy guarantees.

⭐ **What earns the role**
Proposing a privacy budget framework: each experiment consumes some ε from the total privacy budget. As the budget depletes, subsequent experiments require stronger justification. This is the same framework Google uses for Chrome privacy-preserving analytics.

---

## Day 71 — Long-Term Treatment Effects

🎯 **"A feature shows +5% Day-7 retention in the experiment. How do you predict its effect on 90-day retention?"**

💬 **Model Answer**
Short-term retention doesn't linearly extrapolate to long-term. Several approaches:

**1. Historical calibration (surrogate validation)**
Collect past experiments with both Day-7 and Day-90 retention outcomes. Fit a model: Day-90 effect = f(Day-7 effect). If the relationship is stable, use it to predict. But: this is only valid if the causal mechanism is the same across experiments.

**2. Extended holdout / long-run holdback**
Keep a 1% holdout from the launch. After 90 days, compare to fully-exposed users. Gives direct measurement of long-term effect. Cost: delayed certainty.

**3. Mechanistic reasoning**
Why does the feature affect retention? If it's habit-forming (e.g., daily streak), Day-7 effect will compound. If it's novelty-driven, it will decay. The mechanism determines whether short-term is optimistic or pessimistic.

**4. Cohort tracking**
Track cohorts exposed to the feature at launch. Compare their 90-day retention to pre-launch cohorts (with DiD adjustment for time trends).

↳ **"What's the risk of using Day-7 as a proxy for Day-90?"**
If the feature improves Day-7 retention by changing *what type of users* complete onboarding (not just whether they complete it), the Day-7 cohort is compositionally different — and this composition effect doesn't necessarily persist to Day-90.

↳ **"What experiments specifically are bad predictors of long-term effects?"**
Anything with novelty effects, push notification experiments (users may ignore or unsubscribe over time), gamification experiments (initial engagement decays), or experiments targeting acquisition rather than habit formation.

🚩 **Red Flag**
"Day-7 is highly correlated with Day-90, so the prediction is reliable." — Correlation across users is not the same as correlation of treatment *effects* across experiments.

⭐ **What earns the role**
Distinguishing between user-level correlation (high-retention users are retained at Day-7 and Day-90) and experiment-level correlation (features that improve Day-7 also improve Day-90). The latter is what matters for prediction; it requires a library of historical experiments, not just user data.

---

## Day 72 — Decision Theory Under Uncertainty

🎯 **"The experiment is borderline: p=0.06, CI=[−0.1%, +4%], guardrails are clean. How do you decide?"**

💬 **Model Answer**
This is a decision under uncertainty — the data is ambiguous. Pure statistical significance threshold (ship if p < 0.05) gives a false sense of objectivity. The right framework:

**Expected value of shipping:**
E[value | ship] = P(effect > 0) × E[effect | effect > 0] − P(effect < 0) × E[harm | effect < 0]

From the CI [−0.1%, +4%]: P(effect > 0) is high (~95%+), but P(effect < 0) is non-zero. The expected harm is small (CI lower bound is only −0.1%). The expected gain is meaningful (~2% effect × scale).

**Factors favoring shipping:**
- Expected harm is trivially small (−0.1%)
- Engineering cost of the feature is sunk
- Feature is reversible (can roll back)
- Strategic alignment

**Factors against:**
- Cannot rule out zero effect (results may be noise)
- p=0.06 means 1-in-17 chance this is a false positive
- Future experiments will be biased if we routinely ship at p=0.06

↳ **"If you always use a fixed p < 0.05 threshold, what are you implicitly assuming?"**
That the cost of a false positive (shipping something useless) equals the cost of a false negative (missing a real improvement). This is almost never true. Decision theory allows you to incorporate the actual asymmetric costs.

↳ **"What's the Bayesian decision rule?"**
Compute expected loss of each decision: ship if E[loss(don't ship)] > E[loss(ship)]. This is the Bayesian optimal decision. Requires specifying a prior and a loss function — explicit but subjective.

🚩 **Red Flag**
"p=0.06 is close to 0.05, so we can ship." — This is not a principled argument; it's approximation by proximity.

⭐ **What earns the role**
Proposing the explicit decision framework: document the decision with the evidence (p=0.06, CI=[-0.1%, +4%]), the reasoning (expected harm is minimal, expected value is positive), and the reversibility (rollback plan if post-launch monitoring shows degradation). This is defensible, transparent, and reproducible.

---

## Day 73 — False Discovery Rate in Large Organizations

🎯 **"Your organization runs 500 experiments per year. How do you think about the aggregate false positive rate?"**

💬 **Model Answer**
At α=0.05 and 500 experiments, if all null hypotheses were true, you'd expect 25 false positives. In reality, the null is often partially true — many experiments have small real effects. The empirical false discovery rate (eFDR) is:

eFDR = False positives shipped / Total positives shipped

To estimate this: use the distribution of p-values. Under H₀, p-values are uniform U[0,1]. Excess mass near 0 indicates real effects. The proportion of true nulls π₀ can be estimated from the p-value distribution, and eFDR computed from Storey's q-value method.

**Organizational implications:**
1. If win rate is 70% (350/500 positive), and π₀ = 40%, eFDR ≈ 6% — about 21 false positives shipped per year.
2. This accumulates: if features persist for 2 years, you have 42 neutral features cluttering the product.
3. Remedy: lower α for low-stakes experiments, maintain a replication track (re-run top experiments before permanent launch).

↳ **"How do you build a replication culture?"**
Require that any experiment result above some business impact threshold be replicated before full launch. The replication experiment uses independent data (new time period or cohort). Replication reduces eFDR substantially.

↳ **"What's the prior probability that any given experiment works?"**
At mature companies, typically 30–50% of well-designed experiments show positive results. Bayesian updating with this prior dramatically improves the positive predictive value of significant results.

🚩 **Red Flag**
"Each experiment is independent, so there's no organization-level FDR concern." — The product is not independent — all features coexist and interact.

⭐ **What earns the role**
Connecting to portfolio theory: the organization is making a portfolio of bets. The aggregate quality of those bets — not just individual experiment validity — determines product quality. eFDR is a portfolio-level metric.

---

## Day 74 — Designing for Irreversible Decisions

🎯 **"You're testing whether to permanently delete a legacy feature used by 5% of users. How do you design this experiment?"**

💬 **Model Answer**
Irreversible or high-consequence decisions require more conservative experimental design:

1. **Lower α:** Use α=0.01 or 0.001 — fewer false positives. The cost of mistakenly concluding it's safe to delete is high.
2. **Higher power:** Run for longer or use larger allocation to treatment (but be careful — "deleting" is the treatment, which harms if wrong).
3. **Pre-experiment user research:** Before the experiment, survey the 5% who use the feature. Understand use cases. Are there segments for whom this is critical?
4. **Staged rollout:** Don't go 50/50 immediately. Start with 1% deletion, monitor heavily, then scale.
5. **Long duration:** Legacy feature users may use it infrequently. Run for 4–8 weeks to capture all usage patterns.
6. **Rollback plan:** Even for "permanent" deletion, ensure technical rollback is possible for at least 90 days post-experiment.

Guardrails: support ticket rate, user churn, direct complaints, usage of alternative feature.

↳ **"What metrics do you watch most closely?"**
Churn rate (irreversible for the user) and support volume (leading indicator of user distress). Any significant increase in churn attributable to the deletion → rollback immediately.

↳ **"5% of users sounds small. Should you care?"**
At any large company, 5% may be millions of users. And they're likely the most loyal, high-value, long-tenured users — overrepresented in LTV. Churning 5% of users who use a legacy feature could cost far more than the technical debt saved by deleting it.

🚩 **Red Flag**
"Only 5% use it so it's fine to delete." — Percentage doesn't determine business impact; absolute user count and user value do.

⭐ **What earns the role**
Proposing a "migration experiment" instead of a "deletion experiment": redirect feature users to the alternative, measure adoption and satisfaction, then delete only after successful migration is confirmed.

---

## Day 75 — Organizational Influence

🎯 **"A senior leader wants to ship a feature without an experiment because 'it's obviously good.' How do you respond?"**

💬 **Model Answer**
This is a leadership and influence challenge, not just a statistics problem.

**First:** Understand their reasoning. Is the timeline critical (competitive threat)? Is the feature reversible? Are there safety concerns with running an experiment? Not all situations require an experiment.

**Then, if the experiment is warranted:**
1. Show the cost of being wrong: "If this feature doesn't work as expected and we don't detect it, the expected cost is $X over Y months before post-launch monitoring catches it."
2. Show the cost of the experiment: "A 2-week experiment delays launch by 2 weeks. Opportunity cost is $Z in revenue during that window."
3. Propose a compromise: "We can do a 1% ramp for 1 week — not a full experiment, but enough to catch critical regressions before full rollout."
4. Document: "I'm logging my recommendation to run an experiment. If we proceed without one, I want to ensure we have post-launch monitoring in place."

↳ **"What if they override you?"**
Document your recommendation, ensure rollback capability, set up post-launch monitoring, and propose a holdback group to measure the effect after the fact. You've done your job; you can't force the decision but can minimize the risk.

↳ **"When is it actually right to skip the experiment?"**
Critical bug fix. Legal/regulatory requirement. Extremely low-risk cosmetic change. Feature with no measurable metric impact. Time-sensitive competitive response where delay is worse than uncertainty. These are legitimate exceptions — not every decision needs an experiment.

🚩 **Red Flag**
"I'd insist on the experiment no matter what." — Dogmatism is not L6 behavior. L6 understands when rules apply and when judgment overrides them.

⭐ **What earns the role**
Framing experimentation as risk management, not bureaucracy. The question is never "do we follow the process" — it's "what is our best decision given the available evidence and time constraints?"

---

## Day 76 — Power Analysis for Complex Metrics

🎯 **"How do you power an experiment for a metric that's a ratio of two metrics (e.g., revenue per session)?"**

💬 **Model Answer**
Standard sample size formulas assume a simple mean. For ratio metrics, use the delta method to compute variance:

For R = A/B (e.g., revenue/sessions):
Var(R) ≈ (1/μ_B²) × [Var(A) + R² × Var(B) − 2R × Cov(A,B)]

Where:
- μ_B = mean of denominator (sessions per user)
- Var(A) = variance of numerator (revenue per user)
- Var(B) = variance of denominator (sessions per user)
- Cov(A,B) = covariance between revenue and sessions per user

Plug this into the standard sample size formula with the delta-method variance.

↳ **"Why can't you just compute variance of revenue/sessions at the user level?"**
If users have different session counts, the ratio at user level is noisier than the population ratio. The correct approach is to estimate variance of the *population* ratio, which is what the delta method gives.

↳ **"What's the easiest way to get this variance empirically?"**
Bootstrap: resample users with replacement, compute revenue/sessions for each bootstrap sample, compute variance of the bootstrap distribution. More accurate than delta method for small samples or heavy-tailed data.

🚩 **Red Flag**
"Just divide revenue by sessions and run a t-test on that." — Correct in spirit but ignores the complex variance structure of ratio metrics.

⭐ **What earns the role**
Knowing that ratio metrics measured at the population level (total revenue / total sessions) vs user level (avg of per-user ratios) have different statistical properties and different interpretations — and choosing the right one for the business question.

---

## Day 77 — Causal Discovery

🎯 **"You have 6 months of observational data and want to understand what causes user churn. How do you approach this?"**

💬 **Model Answer**
Causal discovery from observational data is hard. Framework:

**Step 1: Domain knowledge**
List hypothesized causes of churn: poor onboarding, unresolved support tickets, no social connections, pricing dissatisfaction, competitive switching. Build a directed acyclic graph (DAG) based on domain knowledge.

**Step 2: Identify estimable effects**
For each hypothesized cause, identify an estimation strategy:
- Is there a natural experiment? (e.g., random variation in onboarding flow version)
- Is there a valid instrument? (e.g., randomized email timing)
- Is DiD applicable? (e.g., support ticket resolution SLA changed on a specific date)
- Is matching sufficient? (e.g., users with/without social connections, controlling for observables)

**Step 3: Rank by actionability**
Some causes are unmeasurable or unmodifiable. Focus on causes where: (a) causal evidence is credible, (b) the company can intervene, (c) the effect size is large enough to matter.

**Step 4: Validate with experiments**
Observational estimates are hypothesis-generating. Validate top causes with targeted experiments before investing in interventions.

↳ **"What is a DAG and why is it useful for causal analysis?"**
Directed Acyclic Graph: nodes are variables, edges are causal relationships, no cycles. Forces you to be explicit about what you believe causes what. Reveals which variables to control for (confounders), which to avoid controlling for (mediators, colliders), and which identification strategies are valid.

🚩 **Red Flag**
"Run a correlation matrix and find what correlates most with churn." — Correlation is not causation; this approach will find many spurious relationships.

⭐ **What earns the role**
Knowing the backdoor criterion: to estimate the causal effect of X on Y, control for all variables that open a backdoor path from X to Y (i.e., confounders), and don't control for colliders or mediators. This requires the DAG to identify the right adjustment set.

---

## Day 78 — Replication Crisis & Reliability

🎯 **"How do you prevent a replication crisis in your experimentation program?"**

💬 **Model Answer**
The replication crisis in academia (many published findings fail to replicate) has direct parallels in industry experimentation. Causes and solutions:

**Causes in industry:**
1. **p-hacking / multiple metrics:** Solved by pre-registration and single primary metric.
2. **HARKing:** Solved by pre-registration.
3. **Underpowered experiments:** Solved by power analysis before launch.
4. **Optional stopping:** Solved by fixed-horizon or sequential testing.
5. **Publication bias:** Solved by tracking and reviewing null results.
6. **Winner's curse:** Effects in the winning experiment arm are overestimated due to selection (took the best of many noisy estimates). Solved by shrinkage estimators (James-Stein) or Bayesian hierarchical models.

**Organizational solutions:**
- Require replication for high-impact features before permanent launch
- Track experiment hit rate over time — if it's unusually high, something is wrong
- Maintain a "pre-mortem" culture: before launching, ask "why might this result be false?"

↳ **"What is the winner's curse and how does it apply to A/B tests?"**
When you select the best-performing variant from many noisy estimates, the selected winner's effect is upwardly biased — it contains both the true effect and positive noise. At scale, features that barely pass the significance threshold may have much smaller real effects than estimated. This is why effect size estimates shrink post-launch.

↳ **"How do you correct for winner's curse?"**
Shrinkage: apply Empirical Bayes or James-Stein shrinkage to effect estimates. The shrunken estimate is closer to the population mean — conservative but more accurate. Particularly important for multi-armed experiments where you select the best arm.

🚩 **Red Flag**
"Our experiments always replicate because we run them correctly." — This is naively optimistic. Even correctly-run experiments have winner's curse and selection effects.

⭐ **What earns the role**
Connecting winner's curse to the broader concept of selection bias in statistics: anytime you condition on being "selected" (significant, high-performing), you introduce upward bias. This is a fundamental statistical phenomenon, not a methodological failure.

---

## Day 79 — Federated Experimentation

🎯 **"Your company has products on web, iOS, Android, and Smart TV. How do you run a unified experiment across all platforms?"**

💬 **Model Answer**
Cross-platform experiments require:

**1. Unified user identity**
Assign experiments at the user_id level (not device or session). Requires login — anonymous users can't be tracked cross-platform. For logged-in users: same hash(user_id + experiment_id) regardless of platform.

**2. Consistent feature flag delivery**
Experiment config served from a central service. Each client queries the assignment service with user_id → gets back variant assignment. Must be platform-agnostic (REST API, not platform-specific SDK).

**3. Unified event schema**
All platforms log to the same event schema: user_id, experiment_id, variant, event_type, timestamp, platform. Analysis joins across platforms on user_id.

**4. Platform as a dimension**
Always segment results by platform. iOS users behave differently from Smart TV users — a feature that helps web may hurt TV. Platform is a key HTE variable.

**5. Consistent metric definition**
"Session" means different things on TV vs mobile. Pre-define all metrics with platform-specific carve-outs if needed.

↳ **"What if a user is in the experiment on web but not logged in on TV?"**
They get a new random assignment on TV (as an anonymous user). This creates cross-device contamination — the user sees both variants. Options: (1) Accept the noise (biases toward null). (2) Require login for treatment eligibility. (3) Use device fingerprinting (privacy concerns).

↳ **"How do you handle platform-specific features in a cross-platform experiment?"**
Some features are platform-specific (e.g., TV has no keyboard — can't test a search bar feature there). Restrict the experiment to platforms where the feature is applicable. Report platform-stratified results for completeness.

🚩 **Red Flag**
"Run separate experiments on each platform." — You'll have conflicting results, different assignment logic, and no cross-platform view of user behavior.

⭐ **What earns the role**
Designing the user identity graph: a probabilistic model that links devices to users, enabling cross-platform attribution even for partially-logged-in users. Used at Airbnb, Netflix, and Spotify for cross-device experiment analysis.

---

## Day 80 — Experiment Debt

🎯 **"What is experiment debt and how do you manage it?"**

💬 **Model Answer**
Experiment debt is the accumulation of:
1. **Ramped but never fully launched features** — stuck at 50% exposure
2. **Holdback groups** — small holdouts running for years
3. **Concurrent experiments** — so many running simultaneously that interactions are unmanageable
4. **Stale experiment configs** — old experiments left running after decisions are made
5. **Entangled metrics** — metrics defined for old experiments that persist and pollute new analyses

**Consequences:**
- Users have inconsistent experiences
- New experiments are harder to interpret (entangled with legacy experiments)
- Engineering systems carry technical debt from feature flag spaghetti
- Analysis time increases as analysts navigate complex experiment state

**Management:**
- Experiment TTL (time-to-live): all experiments expire after N weeks unless explicitly extended
- Graduation ceremonies: experiments move through states (design → running → decided → graduated/deprecated)
- Dashboard showing "experiment age" — anything over 8 weeks is flagged
- Annual experiment cleanup sprint

↳ **"What happens if you never clean up old experiments?"**
At scale: 100 active experiments × 3 years = 300 accumulated experiments. Users are in 300 overlapping experiments simultaneously. Any new experiment's results are potentially confounded by any of the 300 existing ones. Analysis becomes intractable.

🚩 **Red Flag**
"Experiments that finished are just archived, they don't affect anything." — Feature flags from "finished" experiments that were never fully rolled out or cleaned up continue to affect user experience.

⭐ **What earns the role**
Proposing a formal experiment lifecycle: Draft → Pre-launch review → Running → Decision → Graduated (feature fully launched or killed, flag removed from code). Each state transition requires explicit human action. Automated nudges after 8 weeks with no decision.

---

## Days 81–100: L6 Capstone — System & Strategy Level

---

## Day 81 — Experimentation Strategy

🎯 **"How do you build a 1-year experimentation roadmap for a new product area?"**

💬 **Model Answer**
Frame as a learning plan, not just a test plan:

**Q1: Foundation**
- A/A tests to validate infrastructure
- Baseline metrics: establish pre-experiment distributions for all key metrics
- First experiments: high-confidence, high-impact hypotheses (low-hanging fruit)
- Goal: build trust in the system and win organizational buy-in

**Q2: Exploration**
- Broad experiments across the funnel (acquisition → activation → retention → revenue)
- Identify which funnel stage has the highest leverage
- Run HTE analyses to understand user segments
- Goal: identify the biggest opportunities

**Q3: Exploitation**
- Deep experiments in the highest-leverage funnel stage
- Multi-armed experiments to test many variants efficiently
- Sequential experiments based on Q2 learnings
- Goal: maximize metric improvement in identified opportunity areas

**Q4: Consolidation**
- Validate long-term effects of Q2/Q3 launches via holdbacks
- Run replication experiments for highest-impact findings
- Build surrogate metric validation for long-term metrics
- Goal: ensure Q2/Q3 results are durable and calibrate effect size estimates

↳ **"How do you prioritize which experiments to run first?"**
ICE framework: Impact × Confidence × Ease. High impact (large addressable user base, large expected effect), high confidence (strong prior evidence, clear mechanism), high ease (fast to build, fast to measure). Combine with strategic alignment — some high-ICE experiments may not be on the product roadmap.

🚩 **Red Flag**
"Run experiments on whatever the PM wants to build." — Reactive experimentation; no strategic allocation of experiment capacity.

⭐ **What earns the role**
Noting that the experimentation roadmap and product roadmap must be co-designed — you can't run an experiment on a feature that doesn't exist, and you shouldn't build features without planning to test them.

---

## Day 82 — Metric Wars

🎯 **"Two teams are arguing about whose metric should be the primary metric for a shared experiment. How do you resolve this?"**

💬 **Model Answer**
Metric disagreements are usually proxy for business priority disagreements. Resolution process:

**1. Map to outcomes:** What user or business outcome does each metric represent? Often the teams want the same thing but are measuring different proxies.

**2. Causal chain:** Which metric is more upstream (causes the other) vs downstream (caused by the other)? The more upstream metric is usually better for detecting early effects.

**3. Empirical validation:** Which metric has been shown historically to predict the long-term North Star metric better? Use the historical experiment library.

**4. Sensitivity comparison:** Which metric has lower variance and higher sensitivity to the changes being tested? Run a power analysis for each.

**5. Decision authority:** Who owns the North Star metric for this product area? That team should have primary metric authority.

**Resolution:** Use both as co-primary metrics with multiplicity correction. Or designate one as primary (higher sensitivity) and one as secondary (business outcome). Document the decision and rationale.

↳ **"What if both teams just want their metric to 'win' for political reasons?"**
This is a data culture problem. Escalate to product leadership with a clear framing: "The question is not which team wins — it's which decision rule best serves users and the business." Propose a structured review with a neutral arbiter.

🚩 **Red Flag**
"Use both and pick whichever is significant." — This is the multiple metrics problem — guaranteed to inflate false positives.

⭐ **What earns the role**
Proposing a metric governance council: a cross-functional group that owns the canonical metric definitions and resolves disputes based on empirical evidence, not political negotiation.

---

## Day 83 — Experimentation for ML Models

🎯 **"How do you A/B test a new ML model (e.g., a ranking model) in production?"**

💬 **Model Answer**
ML model experiments have unique challenges:

**Canary vs Full A/B:**
Start with a canary (1% traffic) to catch catastrophic failures before full exposure. Then ramp to 50/50.

**Online vs Offline Evaluation:**
Offline metrics (AUC, NDCG on held-out data) often don't predict online A/B results. Always validate offline gains with online experiments — the gap is caused by distribution shift between training data and live traffic, and feedback loop effects.

**Metric selection:**
Use the metric the model was NOT directly optimized for — otherwise you're just confirming the model optimized its training objective. Use downstream business metrics (conversion, retention) as primary, with model-specific metrics (precision@K, CTR) as secondary.

**Feedback loops:**
A better ranking model shows better content → users click more → more training data for that content → further reinforcement. This feedback loop means short-term A/B results may understate long-term gains. Monitor with a holdback experiment.

**Shadow mode:**
Before A/B, run the new model in shadow mode — it receives the same requests but its output isn't shown to users. Compare shadow outputs to production outputs to identify edge cases and bugs.

↳ **"What is the training-serving skew problem?"**
The model is trained on data collected under the old policy (users exposed to old model). When deployed, it faces a different data distribution (user behavior changes with the new model). This makes online performance hard to predict from offline evaluation. Requires online evaluation as the gold standard.

↳ **"How do you handle long-term model degradation?"**
Model performance typically degrades over time as user behavior and content distribution drift. Run regular A/B tests comparing the current model to a retrained version. Set up automated retraining triggers when offline metrics degrade past a threshold.

🚩 **Red Flag**
"Offline AUC improved 2% so we'll ship it." — Offline metrics are necessary but not sufficient for ML model launches.

⭐ **What earns the role**
Knowing that "Goodhart's Law for ML models" — optimizing a proxy (CTR) causes it to diverge from the true goal (user satisfaction) over time. The A/B test must measure outcomes the model was NOT directly optimizing to get an unbiased evaluation.

---

## Day 84 — Adaptive Experiments

🎯 **"What is response-adaptive randomization and when should you use it?"**

💬 **Model Answer**
Response-adaptive randomization (RAR) adjusts the allocation ratio between arms based on accumulating data — allocating more traffic to the better-performing arm as the experiment progresses.

**Thompson Sampling** is the most common RAR method: at each step, sample a parameter from the current posterior of each arm, allocate to whichever arm has the higher sampled value. Over time, allocates most traffic to the best arm.

**When to use:**
- High regret per observation (medical trials, expensive interventions)
- Many arms (exploring many content variants)
- Non-stationary environments where you want to adapt to change

**When NOT to use:**
- When you need clean statistical inference for a binary launch decision
- When allocation imbalance would introduce selection bias into analysis
- When the experiment is short (overhead of adaptation isn't worth it)

**Statistical challenges:**
RAR induces correlation between sample sizes and outcomes. Standard frequentist tests are invalid (biased toward the true best arm). Requires specialized inference methods: resampling-based tests, or Bayesian inference (which is naturally compatible with RAR).

↳ **"Why is standard statistical inference invalid after RAR?"**
Standard tests assume samples are drawn i.i.d. RAR violates this — later samples are systematically from better arms. The sampling distribution of the test statistic is no longer what the standard formula assumes, inflating false positive rates or biasing effect estimates.

↳ **"How does RAR differ from bandits?"**
They're essentially the same concept. "Multi-armed bandit" is the framing from reinforcement learning (minimize cumulative regret). "Response-adaptive randomization" is the clinical trials framing (maximize patient benefit). Thompson Sampling is used in both literatures.

🚩 **Red Flag**
"RAR is just a faster A/B test." — RAR optimizes for a different objective (minimize regret during the experiment) vs A/B (minimize error in the launch decision).

⭐ **What earns the role**
Knowing that the regret-minimization objective and the inference objective are fundamentally in tension — you can't fully optimize both simultaneously. The choice of method should reflect which objective is primary for the decision at hand.

---

## Day 85 — Experiment Simulation

🎯 **"How do you validate that your experimentation system is working correctly before you trust its results?"**

💬 **Model Answer**
Simulation-based validation:

**1. A/A test battery**
Run many A/A tests (1000+) on historical data by randomly assigning users to "treatment" and "control" within historical logs. Compute p-values for each. The distribution should be U[0,1]. If excess mass near 0: system is generating false positives. Plot the QQ plot against uniform — deviations indicate systematic issues.

**2. Simulated treatment effects**
Inject synthetic treatment effects of known size. Verify the system detects effects of size δ at the expected rate (i.e., verify power is ≈80% when designed for 80%). Verify the point estimate is unbiased (recovered δ ≈ true δ on average).

**3. SRM injection test**
Artificially create an SRM (e.g., assign 60% to treatment, 40% to control but tell the system to expect 50/50). Verify SRM is detected with high sensitivity.

**4. Multi-arm power simulation**
Simulate experiments with 5, 10, 20 arms at various effect sizes. Verify that multiple testing correction maintains eFDR at the nominal level.

**5. Cross-pipeline validation**
Compare results from the production system to a known-correct hand-coded analysis for a sample of experiments. Differences reveal implementation bugs.

↳ **"How often should you run these validations?"**
On system setup, on any major platform change, quarterly audit, and whenever a suspicious pattern appears in live experiment results (unusually high win rate, persistent SRM in some segments).

🚩 **Red Flag**
"We trust the system because it was built by experienced engineers." — Trust but verify. Systems accumulate bugs; validation is not a one-time event.

⭐ **What earns the role**
Proposing a continuous integration test suite for the experimentation platform: automated tests run on every deploy that verify the p-value distribution, SRM detection, and point estimate accuracy using simulated data.

---

## Day 86 — Product Sense + Experimentation

🎯 **"Instagram is considering hiding likes from posts. Design the experimentation strategy."**

💬 **Model Answer**
This is a high-stakes, hard-to-measure, socially-sensitive experiment. Framework:

**Why it's hard:**
1. **Network effects:** If User A's likes are hidden, their friends' behavior changes — contamination.
2. **Long-term effects:** The true impact is on mental health and creator motivation — not measurable in weeks.
3. **Heterogeneous effects:** Heavy creators vs casual users vs passive scrollers are affected very differently.
4. **SUTVA violation:** Hiding A's likes changes B's information, regardless of B's assignment.

**Experimental design:**
1. **Geo-based experiment:** Countries in treatment (likes hidden) vs control. Reduces social interference.
2. **Duration:** 3–6 months minimum (short-term behavior change is not the target).
3. **Metrics:** Creator posting frequency, follower growth rate, time spent, user-reported wellbeing (survey), mental health proxies, platform NPS.
4. **Guardrails:** Total content volume, advertiser satisfaction, overall engagement.
5. **Holdout:** Keep 10% of treatment country users with likes visible to measure within-country spillover.
6. **Qualitative:** User interviews and diary studies in parallel with the experiment.

↳ **"What's your primary metric?"**
No single metric captures the goal (user wellbeing). Use a composite: weighted combination of creator frequency, user satisfaction survey, and time spent on health-positive content. Declare the weighting before the experiment.

↳ **"What if the experiment shows mixed results — some users better, some worse?"**
Expected outcome. Segment analysis to understand who benefits. Design for the most vulnerable segment if protecting them is the product goal. Consider optional like-hiding (user choice) as a compromise.

🚩 **Red Flag**
"Measure engagement — if it drops, don't ship." — Engagement is exactly what you'd expect to drop if users are spending time more intentionally. Engagement loss may be the right outcome.

⭐ **What earns the role**
Recognizing that this experiment is fundamentally about product values, not just metrics. The experiment can inform the decision, but cannot make it — that requires a value judgment about what Instagram is for.

---

## Day 87 — Sequential Experiments & Iteration

🎯 **"You run Experiment A, it fails. You iterate and run Experiment B. How do you prevent p-value inflation across the sequence?"**

💬 **Model Answer**
Sequential experiments (iterate-and-test) have a multiple testing problem across time — if you run experiments until one succeeds, the effective false positive rate grows with the number of experiments run.

**Solutions:**

**1. Pre-specify the iteration plan**
Before the first experiment, document: "We will test up to 3 versions. If version 3 fails, we stop." Then apply multiplicity correction across all planned tests (e.g., Bonferroni with k=3).

**2. Treat each experiment independently**
If experiments are truly separate hypotheses (not the same hypothesis retested), no correction is needed. Experiment B tests "redesigned feature" — a different hypothesis from "original feature." Independence depends on how different B is from A.

**3. Bayesian approach**
Each experiment updates your prior. The posterior after Experiment A informs the prior for Experiment B. No p-value inflation — Bayesian updating is coherent across experiments.

**4. Raise the bar for later experiments**
If Experiment A failed with p=0.4 (strong null), the prior for B should be more skeptical. Informally, require stronger evidence (lower α) for experiments that are testing the same core hypothesis.

↳ **"When does iterating and retesting inflate Type I error?"**
When the same hypothesis is being tested (essentially retested with a minor tweak), and you stop and ship when any iteration crosses p < 0.05. This is functionally the same as optional stopping — it's sequential testing without correction.

↳ **"How do you communicate this risk to a PM who wants to keep iterating?"**
"Each iteration costs us some false positive budget. We can do 3 iterations with overall error rate controlled at 5% by using α=0.017 per test. If we do more, we're accepting a higher false positive rate. I want to make that tradeoff explicit."

🚩 **Red Flag**
"Each experiment is independent, so no correction needed." — Only true if the hypotheses are genuinely different. If it's "same feature, tweaked UX," it's not independent.

⭐ **What earns the role**
Knowing that the distinction between "different hypothesis" and "same hypothesis retested" is fundamentally a judgment call — and that pre-specifying the hypothesis (including acceptable variations) before any iteration is the only rigorous solution.

---

## Day 88 — Multi-Market Experiments

🎯 **"You want to test a feature across 5 different countries simultaneously. How do you design and analyze this?"**

💬 **Model Answer**
Multi-market experiments allow estimation of the treatment effect in each market and a pooled overall effect.

**Design:**
- Randomize at user level within each country
- Ensure each country has adequate sample size independently (country-specific power analysis)
- Same randomization salt across countries (consistent assignment for globally traveling users)
- Pre-specify whether the primary estimate is country-specific or global

**Analysis:**

**1. Country-stratified analysis**
Estimate treatment effect separately per country. Combine with a fixed-effects meta-analysis (weighted by inverse variance). Gives a precision-weighted global estimate.

**2. Random effects meta-analysis**
If treatment effects differ by country (HTE at country level), use random effects — allows for country-level heterogeneity in the effect. Reports: average effect + between-country variance.

**3. Test for heterogeneity**
Cochran's Q test or I² statistic: is there significant variation in treatment effects across countries? If yes, report country-specific estimates, not just the global average.

↳ **"US shows +5%, Brazil shows -2%, others are near zero. What do you conclude?"**
Significant HTE across countries. Do not average and conclude "+1% globally." The feature may harm Brazilian users — investigate why (localization issue, cultural context, different competitive environment). Consider country-specific rollout decisions.

↳ **"How does this change your metric definitions?"**
Some metrics need country-specific definitions (e.g., revenue is in local currency, purchasing power differs). Normalize metrics (e.g., revenue as % of ARPU) or analyze in local currency with careful interpretation.

🚩 **Red Flag**
"Pool all countries together and run one t-test." — Ignores country-level heterogeneity and may be dominated by the largest country (US) even if others show opposite effects.

⭐ **What earns the role**
Proposing a hierarchical Bayesian model: each country's treatment effect is drawn from a global distribution. Partial pooling — small countries borrow strength from the global estimate; large countries influence the global estimate. More principled than fixed-effects meta-analysis.

---

## Day 89 — The Trolley Problem of Experiments

🎯 **"You're running an experiment and halfway through you detect that treatment users have a significantly higher refund rate. Do you stop?"**

💬 **Model Answer**
This is an ethical and statistical question simultaneously.

**Statistical view:**
Refund rate is a guardrail metric. If you pre-specified a stopping rule for guardrail violations (e.g., "stop if refund rate increases >10% with p < 0.01"), and that threshold is crossed — yes, stop. This was planned. If there was no pre-specified rule, stopping now is reactive and introduces bias.

**Ethical view:**
Users in treatment are experiencing more refunds — a signal of product harm (worse experience, misleading descriptions, defective items). The ethical obligation to prevent ongoing harm may override the statistical obligation to complete the experiment.

**Decision framework:**
1. How severe is the harm? Refund = inconvenience vs safety issue?
2. Is the signal robust? Is the increase statistically significant on the guardrail metric?
3. Can you mitigate without stopping? (e.g., reduce traffic allocation to 1%)

**If the harm is clear and significant:** Stop. Log the decision and reason. The statistical integrity of the experiment is secondary to user welfare.

↳ **"How do you prevent this situation in future experiments?"**
Pre-specify guardrail stopping rules for all high-risk metrics before the experiment starts. Build automated monitoring that triggers alerts or automatic pauses when guardrail thresholds are crossed. These are not peeking — they're pre-committed decision rules.

↳ **"Does stopping early invalidate the results?"**
Yes, for the primary metric analysis (biased by early stopping). But the guardrail finding (refund rate increase) is valid evidence that the feature is harmful — that conclusion stands regardless of whether the primary metric was ready for analysis.

🚩 **Red Flag**
"We never stop experiments early — it biases results." — Statistical purity does not override ethical responsibility.

⭐ **What earns the role**
Knowing that pre-specified stopping rules (including guardrail stops) are statistically valid — they don't inflate Type I error because they're committed before data is seen. The key is "pre-specified." Ad hoc stopping is problematic; planned stopping is fine.

---

## Day 90 — Experiment Governance at Scale

🎯 **"You're the Head of Experimentation at a FAANG company. How do you govern 1,000+ experiments per year?"**

💬 **Model Answer**
Four pillars:

**1. Automated gatekeeping**
- Experiment system enforces pre-registration (metric, α, power, duration) before flag can be enabled
- SRM check runs automatically; experiments with SRM are blocked from decision until resolved
- Experiments with no decision after 8 weeks get automated escalation
- TTL enforcement: experiments expire unless extended

**2. Tiered review**
- Tier 1 (low risk, small scope): self-serve, auto-approved with standard template
- Tier 2 (medium risk, broad scope): peer review by experiment team
- Tier 3 (high risk: pricing, core UX, irreversible): full review committee with sign-off
- Risk classification is based on: user exposure %, reversibility, metric risk (health, safety, financial)

**3. Experiment quality metrics**
Track at org level: SRM rate, underpowered experiment rate, peeking rate, null result reporting rate, time to decision, experiment velocity. Report quarterly to leadership. Use these to identify teams that need training or tooling support.

**4. Learning infrastructure**
- Experiment library: searchable database of all past experiments with results, decisions, and long-term outcomes
- Meta-analysis: periodically aggregate experiments in the same product area to get better effect estimates
- Surrogate validation: ongoing calibration of short-term metrics against long-term outcomes

↳ **"What's the #1 thing that breaks experimentation culture at scale?"**
HiPPO overrides — when senior leaders routinely override experimental evidence. It signals that experiments are theater, not decision-making. Solution: leadership alignment that experimental evidence is the input to decisions, not a box to check. Start with wins — show cases where experiments prevented bad launches.

↳ **"How do you scale data science support for 1,000+ experiments?"**
Automate the routine (standard metrics, standard tests). Data scientists focus on: complex analyses (HTE, causal inference), novel metrics, cross-experiment meta-analysis, and mentoring. Self-serve tooling handles the tail of straightforward experiments.

🚩 **Red Flag**
"Have a centralized DS team review every experiment." — Doesn't scale. Creates a bottleneck and slows velocity.

⭐ **What earns the role**
Proposing that governance is a product: the experiment platform itself must be designed to make the right thing easy and the wrong thing hard. Governance through UX (you can't enable a flag without completing pre-registration) is more effective than governance through policy.

---

## Days 91–100: Final Scenarios & Integration

---

## Day 91 — End-to-End Experiment Design

🎯 **"Design a complete experiment to test a new Instagram Reels recommendation algorithm."**

💬 **Full Model Answer**
**Business context:** New algorithm uses a transformer model vs current CF model. Hypothesis: better content understanding improves watch time and creator growth.

**Randomization unit:** User (consistent experience; creator-viewer interactions are complex but user-level is standard at Instagram scale)

**Primary metric:** Total Reels watch time per user per day (captures engagement quality and quantity)

**Secondary metrics:** Reels shares, profile visits from Reels, creator follower growth rate

**Guardrail metrics:** App crash rate, page load time, Stories/Feed engagement (cannibalization check), user satisfaction score (survey sample)

**MDE:** 1% relative lift in watch time. Baseline: 30 min/day, SD: 45 min. Power = 80%, α = 0.05. → ~200K users per arm.

**Duration:** 3 weeks (>1 week for weekly cycle; 3 weeks to check novelty decay)

**Pre-experiment:** A/A test on the assignment system, SRM check setup, shadow mode for 1 week to catch bugs

**Analysis:** CUPED with prior week's watch time as covariate. Segment by: creator vs viewer, content category, user tenure, device.

**Monitoring:** Daily SRM check, daily crash/load time monitoring. Alert if any guardrail degrades >5%.

**Decision rule:** Ship if primary metric is significant at p<0.05 with positive direction, all hard guardrails clean, and no significant negative in secondary metrics. If guardrails show marginal signals, escalate.

**Post-launch:** 2% holdback for 90 days to measure long-term effect.

---

## Day 92 — The Anti-Pattern Gallery

🎯 **"Name and explain 5 anti-patterns in A/B testing you've seen or can anticipate."**

💬 **Model Answer**

**1. The Eager Stopper**
Stops the experiment the moment p < 0.05 is seen on the dashboard. Classic peeking. Result: inflated false positive rate, overestimated effect sizes.

**2. The Metric Tourist**
After seeing null results on the primary metric, goes "shopping" across 30 secondary metrics until finding one that's significant. Reports it as the finding. Result: certain false positive.

**3. The Segment Safari**
After seeing a null overall result, runs post-hoc analysis across 20 segments (age, device, location, tenure...) until one shows significance. "It works for Android users in Brazil!" Result: almost certainly noise.

**4. The Hopeful HiPPO**
Leadership "knows" the feature works and ships it anyway. The experiment was cosmetic — designed to confirm, not to learn. Result: bad features ship, culture erodes.

**5. The Zombie Experiment**
An experiment that showed a null result was never formally closed. The feature flag is still live, creating an unofficial split test that contaminates other experiments. Result: uncontrolled confounding for months or years.

↳ **"Which anti-pattern is hardest to fix culturally?"**
The Hopeful HiPPO — it requires leadership buy-in and is self-reinforcing (leaders who override experiments and get lucky become more confident overriding). The fix is institutional: require experiment evidence for major launches, and track the outcomes of override decisions.

---

## Day 93 — Communicating Results

🎯 **"How do you present a complex experiment result to a C-suite audience?"**

💬 **Model Answer**
**Structure:**
1. **One-sentence headline:** "The new checkout flow increases revenue per user by 3.2% with high confidence."
2. **Business impact:** At current scale, this is $X million/year. Annualized, assuming effect is stable.
3. **Confidence:** "We ran this for 4 weeks with 2M users. The result would only occur by chance 2% of the time if there were no real effect."
4. **What we checked:** "We verified page load time didn't increase, refund rate was flat, and the effect is consistent across mobile and desktop."
5. **Recommendation:** "Ship. We'll run a 2% holdback for 90 days to validate long-term retention impact."
6. **Risk:** "The one uncertainty is long-term retention — we'll have data in 90 days."

**Avoid:** p-values (say "2% chance this is noise"), CIs (say "we're confident the lift is between 2% and 4.5%"), jargon.

↳ **"How do you handle a result you don't agree with but the data clearly shows?"**
Present the data faithfully, with your interpretation of why it might be happening, and any caveats about metric validity. Separate "here's what the data shows" from "here's my hypothesis about why." Don't editorialize the data.

---

## Day 94 — Connecting Experiments to Strategy

🎯 **"You have $10M in engineering capacity for experimentation this year. How do you allocate it?"**

💬 **Model Answer**
Portfolio allocation framework:

**30% — Platform & Infrastructure**
Investment in the experiment platform itself: faster assignment service, better variance reduction (CUPED at scale), always-valid dashboards, self-serve metric definitions. Every dollar here multiplies the value of all future experiments.

**40% — Core Product Bets**
High-confidence experiments in areas with the highest expected return on the North Star metric. Typically: top-of-funnel (acquisition), activation (onboarding), and retention (habit formation). These have the highest expected value per experiment.

**20% — Exploration**
Lower-confidence experiments in new product areas, new user segments, or novel hypotheses. Expected failure rate is higher but learnings compound. Treat as R&D investment, not immediate ROI.

**10% — Measurement Innovation**
New metrics development, long-term holdback studies, surrogate metric validation, causal inference for observational data. Enables higher-quality decisions in all other categories.

↳ **"What's the opportunity cost of over-investing in platform?"**
At some point, diminishing returns on platform investment — once it's "good enough," additional infrastructure investment doesn't improve decisions. The threshold is when platform limitations are no longer the binding constraint on experiment quality or velocity.

---

## Day 95 — Self-Critique

🎯 **"What are the fundamental limitations of A/B testing that no methodology can fully solve?"**

💬 **Model Answer**
Honest intellectual humility is a key L6 trait:

**1. External validity**
A/B tests measure effects on current users, in the current competitive environment, in the current season. Results may not generalize to future users, markets, or contexts. Every A/B test is a local estimate.

**2. Ethical constraints**
You can't ethically randomize users to experiences you believe are harmful. So you can never measure the true causal effect of things like "deleting a critical safety feature" or "deliberately showing misinformation."

**3. SUTVA violations at scale**
At massive scale (billions of users), almost every user's behavior is influenced by other users. Perfect isolation is impossible. All estimates are contaminated by some spillover.

**4. The fundamental problem of causal inference**
You never observe the counterfactual. Every causal estimate is an approximation based on assumptions. If those assumptions are wrong (non-random assignment, SUTVA violation, model misspecification), the estimate is biased — and you often can't tell.

**5. Short time horizons**
Experiments that run for weeks cannot measure effects on annual churn, lifetime value, or competitive positioning. The most strategically important effects are the hardest to measure.

**6. Metric validity**
A/B tests are only as good as the metrics they measure. If your metrics don't capture user value — if they're gameable, noisy, or misaligned with long-term outcomes — no amount of statistical rigor helps.

↳ **"Given these limitations, should we still run A/B tests?"**
Absolutely. The question is not "are A/B tests perfect" — they're not. The question is "are A/B tests better than the alternative?" The alternative is HiPPO decisions, which are subject to all these limitations plus cognitive bias, political pressure, and no quantitative feedback loop. Imperfect experimentation beats confident intuition.

---

## Day 96 — Crisis Scenario

🎯 **"An experiment accidentally exposed 10M users to a bug that caused data loss. How do you respond?"**

💬 **Model Answer**
**Immediate (0–30 min):**
1. Kill switch — disable the experiment flag immediately
2. Incident declaration — page on-call, declare P0
3. Scope assessment — which users, what data, since when?
4. User communication team on standby

**Short-term (30 min–24 hr):**
5. Root cause analysis — how did the bug pass experimentation review?
6. Impact quantification — how many users, how much data?
7. Recovery — can data be restored from backups?
8. User notification — legal/comms lead on notification requirements (GDPR, CCPA)

**Post-incident (1 week+):**
9. Blameless postmortem — what failed in the experiment design, code review, canary process?
10. Systemic fixes: (a) Require data integrity tests before any experiment flag is enabled. (b) Lower canary threshold (0.1% instead of 1%). (c) Add automated data loss detection to monitoring.

**Experimentation system fix:**
This experiment should never have reached 10M users without a canary catching the bug. The canary process failed. Every high-risk experiment (touching data storage, payments, auth) needs a mandatory 0.1% canary with 24-hour hold before any ramp.

↳ **"How do you prevent this in future?"**
Tiered experiment risk classification. Data-touching experiments are Tier 3: require architecture review, mandatory 0.1% canary for 48 hours, automated data integrity checks in monitoring, manual approval to ramp beyond 1%.

---

## Day 97 — Experimentation Failures Hall of Fame

🎯 **"Tell me about a famous experiment failure and what you'd have done differently."**

💬 **Model Answer**
**Example: Facebook's 2012 Emotional Contagion Study**

Facebook manipulated 689,000 users' News Feeds to show more positive or negative content, studying whether emotions spread through social networks. It was published in PNAS. Massive public backlash.

**What went wrong:**
1. No informed consent — users weren't told they were in a psychological experiment
2. Emotional harm potential — users seeing more negative content may have experienced real harm
3. No IRB review for the academic publication
4. Unclear whether standard Terms of Service constitutes valid consent for psychological experiments

**What I'd do differently:**
1. IRB review before any experiment touching psychological outcomes
2. Explicit disclosure or consent waiver process for sensitive experiments
3. Risk classification: any experiment designed to affect emotional state = Tier 3, requires ethics review
4. Measure harm guardrails: self-reported mood, session-ending behavior, mental health signals

**The broader lesson:**
Statistical validity ≠ ethical validity. An experiment can be perfectly designed from a statistics standpoint and still be wrong to run. Experimentation ethics is not a legal compliance checkbox — it's a genuine obligation.

---

## Day 98 — The Future of Experimentation

🎯 **"Where is experimentation methodology heading in the next 5 years?"**

💬 **Model Answer**
**1. Always-valid inference as default**
E-values and confidence sequences will replace fixed-horizon frequentist tests as the industry standard. Continuous monitoring with valid inference becomes the default UX for experiment dashboards.

**2. Causal ML at scale**
HTE estimation via causal forests and double ML becomes routine, not research. Every experiment automatically estimates individualized treatment effects and surfaces them for personalized policies.

**3. Automated experiment design**
AI systems suggest optimal experiment designs (sample size, metric, duration, segment stratification) based on the hypothesis and historical data. Reduces design errors and time-to-launch.

**4. Cross-experiment learning**
Meta-analysis across thousands of experiments becomes automated. Effect estimates for common features are pooled across experiments, improving precision and enabling empirical priors for Bayesian testing.

**5. Privacy-preserving experimentation**
On-device computation and federated analytics enable experiments without centralizing sensitive user data. Differential privacy becomes standard for sensitive metrics.

**6. Simulation-based validation**
Before running an experiment, simulate its results on synthetic data generated from historical user models. Validate expected power, SRM risk, and metric sensitivity.

↳ **"What's the biggest unsolved problem in experimentation?"**
Long-term causal effects. We can measure 2-week effects reliably. We cannot measure 1-year effects without holdbacks that take a year. Developing validated surrogate metrics that reliably predict long-term outcomes from short-term signals remains the hardest open problem.

---

## Day 99 — Your Experimentation Philosophy

🎯 **"What is your personal philosophy on experimentation?"**

💬 **Model Answer**
*This question tests whether you've internalized experimentation as a way of thinking, not just a technique.*

"Experimentation is institutionalized humility. It's the acknowledgment that our intuitions are unreliable, our models are wrong, and our best ideas often don't work the way we expect. The goal is not to prove we're right — it's to find out what's actually true.

The best experimentation cultures I've seen share three traits: they celebrate learning over winning, they treat null results as valuable as positive ones, and they use evidence to make decisions even when that evidence is uncomfortable.

The worst mistake is treating experiments as a bureaucratic hurdle — something to run so you can say you tested it, not because you genuinely want to know the answer. An experiment run without genuine willingness to be wrong is a waste of everyone's time.

At the end of the day, experimentation is just structured learning under uncertainty. The statistics are the how; the humility is the why."

---

## Day 100 — The Final Round

🎯 **"You have 30 minutes left in the interview. Ask me anything about the role."**

💬 **Model Answer**
*What separates L6 candidates: they ask questions that reveal strategic thinking and genuine curiosity.*

Strong questions to ask:
1. "What is the most important experiment this team has run in the last year — and what did you learn from it, whether it succeeded or failed?"
2. "Where does the experimentation culture here have the most room to grow? What's the hardest behavior to change?"
3. "How does the experimentation team influence product roadmap prioritization? Is there a seat at the table early in the process?"
4. "What's the biggest open methodological problem your team is working on?"
5. "What does success look like for this role in 18 months?"

*These questions signal:*
- You think about culture, not just craft
- You're interested in influence, not just execution
- You want to learn, not just demonstrate knowledge
- You've already started thinking about the role

---

*— END OF DAYS 51–100 (L6) —*

---

# APPENDIX: Quick Reference

## Statistical Tests Cheat Sheet
| Metric Type | Large n | Small n |
|---|---|---|
| Proportion (CTR) | Two-proportion z-test | Fisher's exact |
| Continuous (revenue) | Welch's t-test | Bootstrap / Mann-Whitney |
| Ratio (revenue/session) | Delta method | Bootstrap |
| Count (sessions) | Poisson regression | Bootstrap |
| Ordinal (ratings) | Mann-Whitney | Mann-Whitney |

## Sample Size Rule of Thumbs
- Halve the MDE → 4x the sample size
- Double the baseline rate → ~halve the sample size (for proportions)
- CUPED with ρ=0.7 → ~50% variance reduction → ~50% sample size reduction
- Two-tailed → ~25% more sample than one-tailed

## Decision Checklist
- [ ] SRM check passed?
- [ ] Primary metric direction correct?
- [ ] Primary metric statistically significant?
- [ ] All hard guardrails clean?
- [ ] Effect size practically meaningful?
- [ ] Novelty/primacy effects accounted for?
- [ ] Segments checked for HTE?
- [ ] Post-launch monitoring plan in place?
- [ ] Rollback plan if needed?

## Red Flags in Any Experiment
- p-value fluctuating — someone is peeking
- Win rate > 70% — p-hacking likely
- Effect size shrinks post-launch — winner's curse, novelty, or metric gaming
- SRM present — invalid experiment, stop analysis
- Guardrail "not significant" used to justify shipping — check the CI

---
*100 days. One question per day. Every follow-up answered.*
*You now know more about A/B testing than 95% of people who interview for FAANG ML roles.*
*Go get it.* 🚀
