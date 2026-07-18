# 🎯 100-Day A/B Testing Interview Prep
### FAANG Interviewer POV | 1 Question/Day | Deep Follow-Ups
### Days 1–50: L4 Level → Days 51–100: L6 Level

> Format per day:
> - 🎤 **Interviewer Q** — what they actually ask
> - 💡 **Model Answer** — what a strong candidate says
> - ↳ **Follow-up 1, 2, 3...** — where interviewers go next
> - 🚩 **Red flags** — what weak candidates say
> - ⭐ **What makes it L5/L6** — the extra mile

---

# DAYS 1–50: L4 FOUNDATION

---

## Day 1 — What is A/B Testing?

**🎤 Interviewer:** "Walk me through what an A/B test is and why companies run them."

**💡 Model Answer:**
An A/B test is a randomized controlled experiment where users are split into two groups — control (A) sees the current experience, treatment (B) sees the new one. We measure a pre-defined metric to determine whether the change caused a meaningful difference in user behavior. Companies run them to make causal claims about product changes, not just correlational ones. Without randomization, we can't separate the effect of the change from pre-existing differences between user groups.

↳ **Follow-up 1:** "Why randomization specifically? Why not just compare before vs after?"
Before/after comparisons conflate the change with everything else that changed simultaneously — seasonality, marketing campaigns, other product changes. Randomization ensures the only systematic difference between groups is the treatment.

↳ **Follow-up 2:** "What's the difference between correlation and causation in this context?"
Observational data might show that users who use feature X have higher retention — but maybe high-retention users are more likely to discover feature X, not the other way around. A/B testing breaks this ambiguity by forcing random assignment.

↳ **Follow-up 3:** "Give me a real example of a company making a bad decision without A/B testing."
A classic: a company notices users who see a certain banner convert more. They roll it out to everyone. Conversion drops. Turns out high-intent users were clicking the banner regardless — the banner actually annoyed casual users.

🚩 **Red flags:** "A/B testing compares two versions to see which is better" — too vague. No mention of randomization or causality.

⭐ **L5/L6 addition:** Mention experimentation as a learning system, not just a decision tool. The goal is building an accurate model of user behavior, not just shipping winners.

---

## Day 2 — Null and Alternative Hypothesis

**🎤 Interviewer:** "How do you set up hypotheses for an A/B test? Walk me through it."

**💡 Model Answer:**
Before running any test, we state two hypotheses:
- **H₀ (null):** The treatment has no effect — any observed difference is due to random chance.
- **H₁ (alternative):** The treatment has a real effect.

We assume H₀ is true and ask: how likely is it to observe data this extreme if H₀ were true? If that probability (the p-value) is below our threshold α, we reject H₀.

↳ **Follow-up 1:** "One-tailed or two-tailed — which do you use and why?"
Two-tailed by default. It tests whether there's any difference (positive or negative). One-tailed tests for improvement only — they have more power, but miss regressions. In industry, you almost always want to know if your change hurts users, so two-tailed is standard.

↳ **Follow-up 2:** "What does it mean to 'reject the null hypothesis'?"
It means the data is inconsistent with H₀ at the chosen significance level. It does NOT mean H₁ is proven — it means H₀ is implausible given the data.

↳ **Follow-up 3:** "Can you ever accept the null hypothesis?"
No — in frequentist statistics, you can only fail to reject it. Absence of evidence is not evidence of absence. If your test is underpowered, a null result is uninformative.

🚩 **Red flags:** Saying "we accept H₀" or confusing p-value with probability that H₁ is true.

⭐ **L5/L6 addition:** Discuss pre-registration of hypotheses — deciding before seeing data prevents HARKing (Hypothesizing After Results are Known).

---

## Day 3 — p-value

**🎤 Interviewer:** "What is a p-value? I want the real definition, not a textbook oversimplification."

**💡 Model Answer:**
The p-value is the probability of observing a test statistic at least as extreme as the one computed from your data, assuming the null hypothesis is true. It measures how compatible your data is with H₀ — a small p-value means the data is unlikely under H₀, giving us reason to reject it.

Key clarification: p-value is NOT the probability that H₀ is true, NOT the probability your result is due to chance, and NOT the probability of a false positive.

↳ **Follow-up 1:** "If p = 0.04, what exactly can and can't you conclude?"
Can conclude: the observed difference is unlikely (4% chance) if there were truly no effect. Cannot conclude: there's a 96% chance the treatment works, or that the effect is practically significant.

↳ **Follow-up 2:** "What's wrong with using p < 0.05 as a universal threshold?"
It's arbitrary (Fisher himself said so). It creates binary thinking — 0.049 and 0.051 are treated very differently despite carrying nearly identical information. Better to report effect size + CI and make decisions on the full picture.

↳ **Follow-up 3:** "What is p-hacking?"
Running many tests, subgroups, or metrics until one shows p < 0.05, then reporting only that result. It exploits the fact that by chance, 5% of tests will show significance even when nothing is real. Inflates the false positive rate severely.

🚩 **Red flags:** "p-value is the probability the result is due to chance" — this is the most common misstatement in interviews.

⭐ **L5/L6 addition:** Discuss the replication crisis in science as a consequence of p-value misuse at scale. Connect to organizational experimentation: with 1000 experiments at α=0.05, expect ~50 false launches.

---

## Day 4 — Type I and Type II Errors

**🎤 Interviewer:** "Explain Type I and Type II errors and how they trade off."

**💡 Model Answer:**
- **Type I error (α):** We reject H₀ when it's actually true — a false positive. We launch a feature that has no real effect. Probability = significance level α.
- **Type II error (β):** We fail to reject H₀ when H₁ is true — a false negative. We miss a real improvement. Probability = β.

They trade off: reducing α (making the significance bar higher) makes it harder to reject H₀ — which increases β. You can't minimize both simultaneously without increasing sample size.

↳ **Follow-up 1:** "Which error is worse — Type I or Type II?"
It depends on context. Type I: you ship something with no benefit (wasted engineering, potential harm). Type II: you miss a real win (opportunity cost). For high-stakes changes (billing, safety), Type I is worse. For exploratory features, Type II may be worse.

↳ **Follow-up 2:** "What is statistical power and how does it relate to Type II error?"
Power = 1 - β. It's the probability of detecting a true effect when one exists. Industry standard is 80% power, meaning we accept a 20% chance of missing a real effect.

↳ **Follow-up 3:** "How do you increase power without changing α?"
Increase sample size — the most direct lever. Also: reduce metric variance (CUPED, outlier capping), increase effect size (not always controllable), or use more sensitive metrics.

🚩 **Red flags:** Confusing which error is which, or not knowing that power = 1 - β.

⭐ **L5/L6 addition:** Frame Type I/II in business terms — "false launch rate" and "missed opportunity rate." Show how you'd set α and power based on cost of each error in a specific product context.

---

## Day 5 — Statistical Power

**🎤 Interviewer:** "Tell me about statistical power. How do you use it when designing an experiment?"

**💡 Model Answer:**
Power is the probability of detecting a true effect given that the effect exists. When designing an experiment, I use power to determine the required sample size. I need to specify:
1. α (significance level) — typically 0.05
2. Desired power — typically 0.80
3. Minimum Detectable Effect (MDE) — smallest effect worth detecting
4. Baseline metric value and variance

From these, I compute n per group. Power analysis happens before the experiment — not after.

↳ **Follow-up 1:** "What is MDE and how do you choose it?"
MDE is the smallest effect size that would change a launch decision. It's a business choice, not a statistical one. If a 0.1% CTR lift wouldn't cause us to ship, don't power for it. Smaller MDE = larger required sample.

↳ **Follow-up 2:** "What happens if you compute power after seeing results?"
Post-hoc power analysis is widely considered misleading. If results are significant, power is trivially high; if not, you're estimating power from an observed effect that includes noise. It doesn't add information beyond what the CI already tells you.

↳ **Follow-up 3:** "Your test ran for 2 weeks and wasn't significant. A PM asks you to check if it was underpowered. What do you say?"
Present the observed effect and CI. If the CI excludes the MDE, the test was adequately powered and the effect is simply smaller than the MDE. If the CI is wide and includes the MDE, the test was underpowered — recommend re-running with more traffic.

🚩 **Red flags:** Confusing power with significance, or not knowing that power must be specified before the experiment.

⭐ **L5/L6 addition:** Discuss the tradeoff between power and experiment velocity — higher power requires more traffic/time, which slows learning cycles. Some orgs deliberately accept 70-75% power to ship more experiments.

---

## Day 6 — Sample Size Calculation

**🎤 Interviewer:** "How do you calculate the sample size for an A/B test?"

**💡 Model Answer:**
For a two-proportion z-test:

n = 2 × (z_α/2 + z_β)² × p̄(1-p̄) / δ²

Where:
- z_α/2 = 1.96 for α=0.05 two-tailed
- z_β = 0.84 for 80% power
- p̄ = pooled baseline proportion
- δ = MDE (absolute difference)

For a continuous metric, replace p̄(1-p̄) with pooled variance σ².

In practice, I use a power calculator or Python's statsmodels — but I understand what each input does.

↳ **Follow-up 1:** "If I double the MDE, what happens to sample size?"
Sample size is proportional to 1/δ². Doubling MDE reduces sample size by 4x. Detecting smaller effects is exponentially more expensive.

↳ **Follow-up 2:** "How does traffic split affect sample size?"
Equal split (50/50) is most efficient. Unequal splits require more total users. If you must use 90/10 (e.g., risky treatment), adjust the formula — you'll need significantly more traffic.

↳ **Follow-up 3:** "You calculate you need 500K users per group but only get 100K/week. What do you do?"
Options: (1) Increase MDE — accept detecting only larger effects, (2) Increase α — accept more false positives, (3) Use variance reduction (CUPED), (4) Use a more sensitive metric, (5) Run longer, (6) Combine with other experiments using factorial design.

🚩 **Red flags:** Not knowing what inputs go into the formula, or thinking more traffic always means shorter experiment duration (ignores weekly cycles).

⭐ **L5/L6 addition:** Discuss adaptive sample sizes — sequential testing methods that don't require fixing n upfront. Connect to always-valid p-values used at Booking.com and Netflix.

---

## Day 7 — Randomization

**🎤 Interviewer:** "How does randomization work in practice at a tech company?"

**💡 Model Answer:**
We hash a combination of user ID + experiment ID using a deterministic hash function (e.g., MD5, MurmurHash), take the output modulo 100, and assign users to buckets 0–99. Buckets are then mapped to variants (e.g., 0–49 = control, 50–99 = treatment).

This ensures:
- **Determinism:** Same user always gets same variant
- **Independence:** Different experiment IDs give independent assignments
- **Balance:** Buckets are uniformly distributed
- **No database lookup needed:** Assignment computed on the fly

↳ **Follow-up 1:** "Why use user ID rather than session ID?"
Session-level randomization means the same user could see different variants in different sessions — inconsistent experience and introduces within-user correlation that inflates apparent significance. User-level ensures a consistent experience and cleaner statistics.

↳ **Follow-up 2:** "What if you want to randomize at device level for a mobile app?"
Use device ID. But be aware users may use multiple devices — they'd see different variants on different devices. For features where consistency matters, login-based (user ID) is better.

↳ **Follow-up 3:** "What is namespace / experiment layer isolation?"
To run multiple experiments simultaneously without interference, we use independent namespaces. Each namespace uses a different salt in the hash, ensuring independent assignment across experiments. Users can be in multiple experiments simultaneously (overlapping), as long as the experiments are in different layers.

🚩 **Red flags:** Not knowing that randomization must be deterministic, or confusing random assignment with random sampling.

⭐ **L5/L6 addition:** Discuss stratified randomization — pre-stratifying by a covariate (e.g., country, device) to ensure balance on key dimensions. Reduces variance and improves power.

---

## Day 8 — Control Group

**🎤 Interviewer:** "What should the control group experience? Is it always 'do nothing'?"

**💡 Model Answer:**
The control group should see the current production experience — the status quo. This is not always "nothing." If you're testing a new recommendation algorithm, control gets the current algorithm. If testing a new UI, control gets the current UI.

The key is that control = baseline you're comparing against. What you're measuring is the *incremental* effect of the change.

↳ **Follow-up 1:** "What is a placebo control and when do you need one?"
A placebo control gives the control group something that mimics the treatment structurally but without the active ingredient. Used in medicine but also relevant in product: if treatment users receive an email and control receive nothing, any effect could be due to receiving communication, not the email content. A placebo (generic email) isolates the specific content effect.

↳ **Follow-up 2:** "Should the control group ever receive a degraded experience?"
Generally no — deliberately degrading control is unethical and also creates a biased baseline (you'd be measuring "new feature vs bad experience" not "new feature vs current experience"). Exception: some stress-testing experiments, done carefully and briefly.

↳ **Follow-up 3:** "What is a concurrent control vs historical control?"
Concurrent: control and treatment run simultaneously (standard A/B). Historical: compare current users to past users without treatment. Historical controls are almost always biased due to temporal confounds — avoid unless no other option.

🚩 **Red flags:** Saying control group gets "nothing" without qualification.

⭐ **L5/L6 addition:** Discuss holdback groups as a type of control that persists after launch, enabling long-term measurement.

---

## Day 9 — A/A Testing

**🎤 Interviewer:** "What is an A/A test? When would you run one and what do you expect to see?"

**💡 Model Answer:**
An A/A test splits users into two groups that both receive identical experiences. It's a sanity check. Expected result: no statistically significant difference in any metric. We use it to:
1. Validate the randomization pipeline
2. Detect pre-existing imbalances or bugs in logging
3. Estimate the false positive rate of our testing system
4. Calibrate variance estimates for sample size calculations

↳ **Follow-up 1:** "If your A/A test shows p < 0.05, what happened?"
Something is wrong. Possible causes: biased randomization (users aren't truly randomly assigned), logging bugs (events double-counted in one group), selection bias (different user pools for each group), or metric computation errors. Investigate before running any real experiments.

↳ **Follow-up 2:** "If you run 100 A/A tests at α=0.05, how many would you expect to be 'significant'?"
About 5 — this is the definition of α. So seeing a few false positives in A/A tests is expected. If you see significantly more than 5%, your system is miscalibrated.

↳ **Follow-up 3:** "Can you use A/A test data to estimate variance for power calculations?"
Yes — A/A tests give you a clean estimate of metric variance under real traffic conditions. Better than using historical data which may not reflect current user behavior.

🚩 **Red flags:** Not knowing what to expect (some think the test should definitely show no significance — but random chance means some false positives are expected).

⭐ **L5/L6 addition:** Continuous A/A monitoring as an automated health check of the experimentation platform — running perpetual A/A tests and alerting when false positive rate drifts above α.

---

## Day 10 — Confidence Intervals

**🎤 Interviewer:** "Explain what a 95% confidence interval means. Why is it more useful than a p-value alone?"

**💡 Model Answer:**
A 95% CI is a range computed from the data such that, if we repeated the experiment many times, 95% of those intervals would contain the true parameter. It gives a range of plausible effect sizes.

It's more useful than a p-value because:
- It shows the direction AND magnitude of the effect
- It communicates uncertainty quantitatively
- It separates statistical from practical significance — you can see if even the upper bound of the CI is too small to matter
- It directly maps to the p-value (if CI excludes 0, p < α)

↳ **Follow-up 1:** "What's the common misinterpretation of a CI?"
"There's a 95% probability the true value is in this interval." Wrong — the true value is fixed; it's either in the interval or not. The 95% refers to the procedure's long-run coverage, not any single interval.

↳ **Follow-up 2:** "Your CI is [+0.1%, +0.3%] CTR lift. Is this a good result?"
Statistically yes (excludes 0). Practically — depends. Is a 0.1%–0.3% CTR lift meaningful for the business? If baseline CTR is 5%, this is a 2–6% relative lift. Whether to launch depends on implementation cost, guardrail metrics, and business strategy.

↳ **Follow-up 3:** "How does sample size affect CI width?"
CI width ∝ 1/√n. Quadrupling sample size halves the CI width. This is why large-traffic companies can detect tiny effects — their CIs are very tight.

🚩 **Red flags:** Saying "95% probability the true value is in the interval."

⭐ **L5/L6 addition:** Discuss one-sided confidence bounds for non-inferiority testing — when you want to prove a new (cheaper/simpler) treatment is not worse than the current one.

---

## Day 11 — Choosing Metrics

**🎤 Interviewer:** "How do you choose the right metric for an A/B test?"

**💡 Model Answer:**
A good metric must be: (1) **Sensitive** — moves in response to real changes, (2) **Attributable** — caused by the treatment, not confounders, (3) **Timely** — changes within the experiment window, (4) **Aligned** — proxies for the long-term business goal.

Process: Start from the business goal, identify user behaviors that indicate progress toward it, pick the measurable signal closest to that behavior that's also fast-moving enough for the experiment timeline.

↳ **Follow-up 1:** "What is a guardrail metric? Give an example."
A guardrail metric is one you monitor to ensure you haven't caused harm, even if the primary metric improves. Example: testing a new ad format — primary metric is ad revenue, guardrail is user session time. If users earn you more ad revenue but leave the app sooner, it's a bad trade.

↳ **Follow-up 2:** "What's the difference between a leading and lagging metric?"
Leading: moves quickly, early signal (e.g., click rate). Lagging: measures real outcome but slowly (e.g., 90-day retention). A/B tests usually need leading metrics. The risk: leading metrics may not correlate with lagging outcomes.

↳ **Follow-up 3:** "How do you validate that your metric is actually correlated with the long-term goal?"
Run retrospective analysis — look at historical experiments where you measured both the leading metric and the long-term outcome. Check if experiments that moved the leading metric also moved the long-term one. This is called "metric validation" or "surrogate endpoint validation."

🚩 **Red flags:** Just picking "whatever is easy to measure" without connecting to business value.

⭐ **L5/L6 addition:** Discuss the OEC (Overall Evaluation Criterion) — a composite metric that combines multiple signals into one score. Reduces multiple comparisons, forces explicit prioritization. Used at Microsoft/Bing.

---

## Day 12 — Experiment Duration

**🎤 Interviewer:** "How do you decide how long to run an A/B test?"

**💡 Model Answer:**
Duration is driven by two things: (1) reaching the pre-calculated sample size, and (2) covering at least one full weekly cycle (7 days), ideally two. Duration = required sample size / daily traffic per variant.

You should never extend the experiment purely because results aren't significant yet — that's peeking and inflates Type I error. The duration is pre-committed before the experiment starts.

↳ **Follow-up 1:** "Why do you need to cover a full weekly cycle?"
User behavior varies significantly by day of week — weekday vs weekend behavior, different user demographics, usage patterns. If your experiment runs Mon–Wed only, the results may not represent the full user population.

↳ **Follow-up 2:** "What if results are clearly significant after 2 days — can you stop early?"
Only if you're using a sequential testing framework (SPRT, always-valid p-values) that controls Type I error under continuous monitoring. With standard tests, stopping at p < 0.05 when you peek is invalid — the threshold should be lower (e.g., 0.001 at interim looks).

↳ **Follow-up 3:** "What if the experiment needs to run for 6 months to reach sample size?"
Rethink the MDE — you may be targeting an effect too small to be detectable given your traffic. Consider: variance reduction methods (CUPED), more sensitive metrics, restricting to a high-traffic surface, or accepting a larger MDE.

🚩 **Red flags:** "Run until significant" — this is the classic mistake.

⭐ **L5/L6 addition:** Discuss the tradeoff between experiment duration and opportunity cost — a 3-month experiment delays all other experiments on that surface. Portfolio thinking: sometimes better to accept lower power for shorter runtime.

---

## Day 13 — Sample Ratio Mismatch (SRM)

**🎤 Interviewer:** "What is Sample Ratio Mismatch? How do you detect and handle it?"

**💡 Model Answer:**
SRM occurs when the observed ratio of users in groups doesn't match the intended ratio. If you designed a 50/50 split but see 52% in treatment and 48% in control, something is wrong. SRM is detected using a chi-squared test on observed vs expected group sizes.

SRM invalidates the experiment — you can't trust any metric results because the groups are no longer comparable.

↳ **Follow-up 1:** "What causes SRM in practice?"
Common causes: (1) bots/crawlers filtered unevenly between groups, (2) client-side crash in one variant losing events, (3) redirect in treatment variant causing some users to bounce before logging, (4) CDN caching serving stale assignments, (5) logging pipeline dropping events for one group.

↳ **Follow-up 2:** "How do you diagnose which cause is responsible?"
Look at where in the funnel the discrepancy appears — is it at assignment time or event logging time? Compare bot traffic, crash rates, load times between groups. Look at the time series of group sizes — does the imbalance appear suddenly or gradually?

↳ **Follow-up 3:** "Can you recover a valid analysis from an experiment with SRM?"
Rarely. If you understand and can model the root cause (e.g., specific devices crashed in treatment), you can potentially exclude those users from both groups. But this requires strong assumptions. Generally, treat SRM as experiment invalidation and re-run.

🚩 **Red flags:** Not checking for SRM before analyzing results, or not knowing that SRM invalidates the experiment.

⭐ **L5/L6 addition:** Build SRM detection into the experiment platform as an automated pre-check before any metrics are shown — prevents analysts from making decisions on compromised data.

---

## Day 14 — Novelty Effect

**🎤 Interviewer:** "What is the novelty effect and how does it affect A/B test results?"

**💡 Model Answer:**
The novelty effect is when users engage more with a new feature simply because it's new and unfamiliar, not because it's genuinely better. This inflates short-term metrics during the experiment, leading to false positive launch decisions. The lift decays over time as novelty wears off.

↳ **Follow-up 1:** "How do you detect a novelty effect?"
Plot the treatment effect over time (day-by-day or week-by-week). A novelty effect shows as a large initial lift that decays monotonically. Compare new users (who see both versions for the first time) vs returning users — returning users show the novelty effect, new users don't.

↳ **Follow-up 2:** "How do you handle it?"
Run the experiment long enough for novelty to wear off — typically 2–4 weeks. Alternatively, analyze only returning users who had prior exposure to the old experience. Or use a holdback group post-launch for long-term measurement.

↳ **Follow-up 3:** "What is the opposite of novelty effect?"
Primacy effect (or change aversion) — users resist the new change and perform worse initially, even if it's objectively better. The effect starts negative and recovers over time. Look at the same time trend but in the opposite direction. Stratify by user tenure to distinguish.

🚩 **Red flags:** Not knowing novelty vs primacy, or not knowing how to detect it from time trends.

⭐ **L5/L6 addition:** Design experiments with a "learning period" — exclude the first N days of data from the primary analysis to let novelty effects dissipate. Pre-register this decision to avoid cherry-picking.

---

## Day 15 — Peeking Problem

**🎤 Interviewer:** "What is the peeking problem in A/B testing? Why is it a problem?"

**💡 Model Answer:**
Peeking means checking results during the experiment and stopping when you see p < 0.05, rather than waiting until the pre-specified sample size is reached. This inflates the Type I error rate dramatically. A p-value computed at a fixed sample size has a known false positive rate (α). But if you check multiple times and stop at the first favorable crossing, you're exploiting random fluctuations — the effective false positive rate can be 3–5x the nominal α.

↳ **Follow-up 1:** "Why does the p-value fluctuate over time?"
As new data comes in, the estimated effect size and standard error both change. Early on, with small samples, the effect estimate is noisy and can randomly cross the significance threshold. The p-value is only calibrated at the sample size it was designed for.

↳ **Follow-up 2:** "How do you allow monitoring without inflating error rates?"
Use sequential testing methods: (1) Pocock or O'Brien-Fleming group sequential tests with spending functions, (2) SPRT (Sequential Probability Ratio Test), (3) Always-valid p-values or e-values. These methods pre-specify how significance thresholds change at each look.

↳ **Follow-up 3:** "What if a PM insists on looking at results daily?"
Build dashboards that show the time series but clearly label the experiment as "in-flight" and not ready for decision. Show whether sample size has been reached. Use always-valid confidence sequences that are valid at any sample size so stakeholders can look without statistical harm.

🚩 **Red flags:** "Peeking is fine as long as you're careful" — it's not. Or not knowing that it inflates Type I error.

⭐ **L5/L6 addition:** Discuss the Bayesian alternative — Bayesian methods allow updating beliefs at any time without the peeking problem, because they don't use p-values. Connect to how companies like VWO and Optimizely shifted to Bayesian testing platforms.

---

## Day 16 — Multiple Testing Problem

**🎤 Interviewer:** "If you test 20 metrics in a single experiment, what's the problem and how do you fix it?"

**💡 Model Answer:**
With 20 independent tests at α=0.05, the probability of at least one false positive is 1 - (0.95)²⁰ ≈ 64%. This is the familywise error rate (FWER) problem. Even if nothing is real, you'll almost certainly find "something significant."

Solutions:
- **Bonferroni correction:** Divide α by number of tests (α/20 = 0.0025). Very conservative.
- **Holm-Bonferroni:** Sequential Bonferroni, less conservative.
- **Benjamini-Hochberg (FDR):** Controls the expected proportion of false positives among rejections. Better for many metrics.

↳ **Follow-up 1:** "What's the difference between FWER and FDR?"
FWER: probability of ANY false positive. Very strict — used in clinical trials. FDR: expected proportion of false positives among all significant results. More lenient — better for exploratory analysis with many metrics.

↳ **Follow-up 2:** "In practice, do companies apply these corrections?"
Often not rigorously. Instead: pre-specify ONE primary metric (which drives the launch decision), treat all others as exploratory. Only the primary metric gets the full significance claim; secondary metrics inform but don't decide.

↳ **Follow-up 3:** "What about testing multiple segments post-hoc?"
Same problem — each segment test is a comparison. Pre-register segments of interest. Post-hoc segmentation is hypothesis-generating only; never launch based purely on a post-hoc segment finding.

🚩 **Red flags:** Running 10 metrics and picking the one that's significant. Not knowing FWER vs FDR.

⭐ **L5/L6 addition:** Discuss the primary/secondary/guardrail hierarchy as a practical organizational solution to multiple testing — pre-commitment to what matters prevents fishing expeditions.

---

## Day 17 — CUPED

**🎤 Interviewer:** "What is CUPED and why would you use it?"

**💡 Model Answer:**
CUPED (Controlled-experiment Using Pre-Experiment Data) is a variance reduction technique. The idea: subtract the part of the outcome metric that's predictable from pre-experiment behavior. This reduces residual variance, tightening confidence intervals and increasing power — without any bias.

The adjusted metric is: Y_cuped = Y - θ × (X_pre - E[X_pre])

Where X_pre is a pre-experiment covariate (e.g., last week's revenue), and θ = Cov(Y, X_pre) / Var(X_pre).

↳ **Follow-up 1:** "Why doesn't CUPED introduce bias?"
Because we subtract the same adjustment from both control and treatment. The expected value of the adjustment is zero (since E[X_pre - E[X_pre]] = 0), so the mean effect estimate is unchanged. Only variance is reduced.

↳ **Follow-up 2:** "How much variance reduction can you get?"
Depends on the correlation between pre and post metric. Variance reduction = ρ² × 100%, where ρ is the correlation. If last week's sessions correlates 0.7 with this week's, variance reduces by 49%. In practice, 30–70% reduction is common.

↳ **Follow-up 3:** "What's the relationship between CUPED and regression adjustment (ANCOVA)?"
They're equivalent when the covariate is from before the experiment. CUPED is a special case of linear regression adjustment. The OLS estimate of the treatment effect in a regression with pre-experiment covariates is the same as CUPED.

🚩 **Red flags:** Not knowing CUPED, or thinking it changes the point estimate (it doesn't — only variance).

⭐ **L5/L6 addition:** Extend to multiple covariates (multivariate CUPED), or non-linear adjustment using ML-based residualization (also called ML-CUPED or post-stratification).

---

## Day 18 — SUTVA

**🎤 Interviewer:** "What is SUTVA and when is it violated in tech experiments?"

**💡 Model Answer:**
SUTVA (Stable Unit Treatment Value Assumption) has two components:
1. **No interference:** One unit's treatment doesn't affect another unit's outcome.
2. **No hidden treatment versions:** Everyone assigned to treatment receives the same treatment.

When SUTVA holds, we can estimate causal effects cleanly. When it's violated, observed outcomes in control are contaminated by treatment — biasing our estimate toward zero (if spillover is positive).

↳ **Follow-up 1:** "Give me three examples of SUTVA violations in tech."
(1) Social network: showing one user a new sharing feature changes their friends' behavior. (2) Marketplace: treating buyers with better recommendations reduces supply for control buyers. (3) Ride-sharing: assigning drivers to a new routing algorithm changes pickup times for all riders, including control.

↳ **Follow-up 2:** "How do you handle SUTVA violations?"
Design the experiment to cluster interference: geo-based randomization (cities, not users), ego-network randomization (entire social clusters), time-based switchback. Accept that you're estimating a different estimand — the total effect including spillover.

↳ **Follow-up 3:** "What is the dilution bias from SUTVA violations?"
When treatment spills over to control, control users experience some of the treatment effect. This makes control and treatment look more similar than they are, shrinking the apparent effect. You underestimate the true impact of the feature.

🚩 **Red flags:** Not knowing SUTVA, or not being able to give concrete examples.

⭐ **L5/L6 addition:** Distinguish between the direct effect (treatment on treated) and total effect (including spillover). Discuss how some companies deliberately measure both — direct effect via user-level randomization, total effect via geo-level randomization.

---

## Day 19 — Network Effects in Experiments

**🎤 Interviewer:** "How would you design an A/B test for a feature on a social network where users interact with each other?"

**💡 Model Answer:**
Standard user-level randomization causes interference — if I'm in treatment and my friend is in control, my new behavior affects their metrics. Options:

1. **Ego-network (cluster) randomization:** Assign entire friend groups to the same variant. Reduces interference within clusters. Requires enough clusters for power.
2. **Graph-based partitioning:** Use community detection (e.g., Louvain) to find dense clusters, then randomize clusters.
3. **Geographic randomization:** Use city/country as the unit.
4. **Bipartite graph treatment:** If the network is bipartite (e.g., creators and consumers), randomize one side only.

↳ **Follow-up 1:** "What's the tradeoff of cluster randomization?"
Fewer independent units → lower power. You go from millions of user-level units to thousands of cluster-level units. Need to account for intra-cluster correlation (design effect) in power calculations.

↳ **Follow-up 2:** "How do you measure the spillover itself?"
Use a two-level design: some users are in treatment, some are control. Within each group, vary whether their neighbors are in treatment or control. Compare: treated user with treated neighbors vs treated user with control neighbors. The difference is the spillover effect.

↳ **Follow-up 3:** "What's wrong with just ignoring the interference?"
Your ATE estimate is biased. If positive spillover, you underestimate the true effect. If negative spillover (crowding out), you overestimate. Either way, your launch decision is based on wrong numbers.

🚩 **Red flags:** Not recognizing interference as a problem, or proposing user-level randomization without acknowledging the issue.

⭐ **L5/L6 addition:** Discuss the difference between direct effects, indirect effects (spillover), and total effects in the potential outcomes framework. Show you know the correct estimand for each design.

---

## Day 20 — Two-Sided Marketplace Experiments

**🎤 Interviewer:** "How would you run an A/B test on Uber or Airbnb where buyers and sellers interact?"

**💡 Model Answer:**
Standard user-level A/B testing fails because supply and demand interact. If treatment buyers get better recommendations and book more, that reduces available inventory for control buyers. The control experience is contaminated.

Solutions:
1. **Supply-side holdout:** Randomly hold back a fraction of supply from the new feature. Measure effect on demand-side separately.
2. **Geographic (market-level) randomization:** Entire cities get treatment or control. Markets are mostly independent.
3. **Switchback experiments:** Alternate treatment and control over time within the same market.
4. **Quasi-experimental methods:** Difference-in-differences with matched markets.

↳ **Follow-up 1:** "What is a switchback experiment in detail?"
Time periods are alternated between treatment and control (e.g., odd hours = treatment, even hours = control). Handles the interference problem because the whole market is in the same state at any moment. Requires time randomization — which period starts treatment should be randomized.

↳ **Follow-up 2:** "What are the limitations of geo-based experiments?"
Fewer units (cities vs users) → lower power. Geographic spillover (users near border between treatment/control cities). Cities aren't identical — need matched pairs or regression adjustment. Seasonal/event effects can confound.

↳ **Follow-up 3:** "How do you analyze a switchback experiment?"
Use a within-market, within-period regression: outcome ~ treatment + market_fixed_effects + time_fixed_effects. The treatment coefficient captures the causal effect while controlling for market and temporal variation.

🚩 **Red flags:** Proposing user-level A/B test for a marketplace without acknowledging the interference issue.

⭐ **L5/L6 addition:** Discuss the concept of "market equilibrium" — some features only show their true value when fully deployed (e.g., surge pricing works because everyone can see it). Partial rollout in A/B underestimates equilibrium effects.

---

## Day 21 — Segmentation Analysis

**🎤 Interviewer:** "After running an A/B test, your aggregate result is not significant but you find a 15% lift for mobile users. What do you do?"

**💡 Model Answer:**
I treat this as exploratory, not confirmatory. Post-hoc segment analysis is hypothesis-generating, not sufficient for a launch decision. The 15% finding could be a false positive due to multiple comparisons — if I tested 10 segments, I'd expect one to show significance by chance.

Steps: (1) Check how many segments were tested — apply FDR correction. (2) Ask if the mobile hypothesis was pre-specified or discovered post-hoc. (3) Run a follow-up experiment powered specifically for mobile users. (4) See if the mobile effect is directionally consistent with a mechanism — does it make sense?

↳ **Follow-up 1:** "When is it okay to act on a post-hoc segment finding?"
If the effect is large, mechanistically plausible, consistent across multiple dimensions (not just mobile — also iOS, App Store users, etc.), and you can run a confirmatory experiment. Never launch a feature to all mobile users based solely on a post-hoc subgroup.

↳ **Follow-up 2:** "What is a heterogeneous treatment effect (HTE) and how do you estimate it properly?"
HTE = the treatment effect varies across individuals. Proper estimation requires pre-registered interaction tests (treatment × subgroup covariate) or causal ML methods (causal forests, X-learner). These are more powerful and statistically valid than simple stratified analysis.

↳ **Follow-up 3:** "If you do find a robust HTE, what are the product implications?"
Launch only to the benefiting segment. Or use it to personalize — users who would benefit get the feature, others don't. HTEs are the foundation of treatment effect personalization.

🚩 **Red flags:** Launching based on a post-hoc segment finding, or not knowing the multiple testing implications.

⭐ **L5/L6 addition:** Discuss pre-registration of interaction hypotheses — if you believe mobile is different before the experiment, specify the mobile interaction test upfront. Then it's a pre-specified, valid test.

---

## Day 22 — Practical vs Statistical Significance

**🎤 Interviewer:** "Your A/B test shows p = 0.001 and a 0.01% CTR lift. Do you launch?"

**💡 Model Answer:**
Not necessarily. p = 0.001 tells me the result is highly unlikely under H₀ — the data is compelling. But 0.01% absolute CTR lift may be practically meaningless. If baseline CTR is 10%, this is a 0.1% relative improvement. I need to ask: is this lift worth the engineering cost? Does it improve user experience? Does it affect revenue meaningfully?

Statistical significance ≠ practical significance. With millions of users, tiny effects become detectable — but tiny doesn't mean valuable.

↳ **Follow-up 1:** "How do you communicate this to a PM?"
Frame it in business terms: "The experiment detected a statistically real but very small effect. At our scale, 0.01% CTR lift translates to approximately X additional clicks per day / $Y additional revenue per quarter. Given the implementation cost of Z, the ROI is approximately W."

↳ **Follow-up 2:** "What's an effect size measure you'd use beyond p-value?"
For proportions: Cohen's h or relative risk. For continuous: Cohen's d. In practice: just report the absolute and relative effect size with CI. "Treatment increased CTR from 10.00% to 10.01% (95% CI: [+0.005%, +0.015%])."

↳ **Follow-up 3:** "Can you have practical significance without statistical significance?"
Yes. With a small sample, a 15% lift might have CI [-5%, +35%] — not significant but potentially important. The right response: increase sample size to determine if the effect is real, not to lower the bar.

🚩 **Red flags:** "Significant p-value means we should launch." Ignoring effect size entirely.

⭐ **L5/L6 addition:** Introduce the concept of "minimum worthwhile effect" — the smallest effect that justifies the cost. Pre-specify this as the MDE. Then if the CI lower bound exceeds this threshold, you have clear evidence of practical value.

---

## Day 23 — Revenue Metrics and Variance

**🎤 Interviewer:** "You're running an A/B test where the primary metric is revenue per user. What challenges do you expect and how do you address them?"

**💡 Model Answer:**
Revenue per user has extremely high variance — a small number of whale users (big spenders) dominate. This makes the metric sensitive to outliers and requires very large samples to detect small effects.

Approaches:
1. **Outlier capping:** Cap at 99th or 95th percentile. Reduces variance significantly; small bias from ignoring top values.
2. **Log transformation:** Stabilizes variance for right-skewed distributions.
3. **CUPED:** Use pre-experiment revenue as covariate. Often reduces variance by 50%+.
4. **Use a different primary metric:** Conversion rate (binary) has much lower variance. Then check revenue per converter as secondary.

↳ **Follow-up 1:** "Is outlier capping valid? Doesn't it introduce bias?"
Yes, it introduces a small downward bias on the effect estimate for the capped metric. But if capping is applied identically to both groups, the bias is symmetric — it affects the point estimate but not the comparison between groups, so inference is valid. The bias in the metric itself is the tradeoff for tractable variance.

↳ **Follow-up 2:** "When would you use median revenue instead of mean?"
Median is robust to outliers and doesn't require capping. But it's harder to estimate variance for (need bootstrap or Greenwood's formula), and less interpretable in business terms. Use when distribution is extremely skewed and outliers can't be capped.

↳ **Follow-up 3:** "What is the delta method and when do you need it for revenue metrics?"
If your metric is a ratio (e.g., revenue per session, not revenue per user), the standard formula for variance doesn't apply because it's a ratio of two random variables. The delta method provides a first-order approximation of the variance: Var(Y/X) ≈ (μ_Y/μ_X)² × [Var(Y)/μ_Y² - 2Cov(X,Y)/(μ_X×μ_Y) + Var(X)/μ_X²].

🚩 **Red flags:** Treating revenue per user like a simple proportion, or not knowing why variance is a problem.

⭐ **L5/L6 addition:** Discuss quantile treatment effects (QTE) — instead of comparing mean revenue, compare the 90th or 95th percentile revenue. Better for detecting shifts in the right tail of the distribution where most revenue lives.

---

## Day 24 — Selection Bias

**🎤 Interviewer:** "Walk me through how selection bias can enter an A/B test and how you'd catch it."

**💡 Model Answer:**
Selection bias occurs when the groups being compared are systematically different in ways unrelated to the treatment. In a properly randomized A/B test, selection bias shouldn't exist — randomization eliminates it by design. It creeps in through:
1. Non-random assignment (bugs in randomization)
2. Self-selection after assignment (users opting in/out)
3. Attrition (dropouts differ between groups)
4. Survivor bias (analyzing only users who completed a flow)

Catch it with: SRM checks, covariate balance checks (compare pre-experiment metrics between groups), time-trend checks, and A/A tests.

↳ **Follow-up 1:** "Give a concrete example of survivorship bias in an A/B test."
Testing a new checkout flow. Treatment reduces cart abandonment. The users who complete checkout in treatment are a *less price-sensitive* group than those who complete in control (because treatment removed friction for marginal buyers). Comparing revenue-per-completor between groups is biased — treatment completors are structurally different from control completors.

↳ **Follow-up 2:** "How do you do a covariate balance check?"
Take pre-experiment features (age, historical purchase rate, device type) and compare their distributions between control and treatment. Use t-tests or KS tests for each covariate. A significant difference suggests randomization failure — even if group sizes match.

↳ **Follow-up 3:** "What is the difference between selection bias and omitted variable bias?"
Selection bias is when group composition is systematically different (a design flaw). Omitted variable bias is when a confounder affects both treatment and outcome but isn't controlled for (an analysis flaw). Both lead to biased causal estimates but through different mechanisms.

🚩 **Red flags:** Thinking randomization automatically protects against all bias, including post-assignment bias like attrition.

⭐ **L5/L6 addition:** Discuss inverse probability weighting (IPW) to correct for selection bias when it's measurable — weight observations by the inverse probability of being assigned to their group, given observed covariates.

---

## Day 25 — Launch Decision Framework

**🎤 Interviewer:** "Walk me through your complete framework for making a launch decision from an A/B test."

**💡 Model Answer:**
Step-by-step:

1. **Pre-flight checks:** SRM check, covariate balance, A/A calibration
2. **Primary metric:** Is it statistically significant? In the right direction? Effect size meaningful?
3. **Guardrail metrics:** Are any guardrails violated? If yes — do not launch regardless of primary metric.
4. **Secondary metrics:** Consistent story? Any surprising signals?
5. **Segment analysis:** Is the effect concentrated in a specific group? Robust across key segments?
6. **Time trend:** Novelty or primacy effects? Is the effect stable?
7. **Practical significance:** Is the effect size worth the implementation/maintenance cost?
8. **External factors:** Is the experiment representative? Any confounds (holiday, incident)?
9. **Decision:** Launch / Don't launch / Iterate / Run longer

↳ **Follow-up 1:** "Primary metric significant, but one guardrail is borderline (p=0.08 in negative direction). Do you launch?"
Borderline guardrail violation is a red flag. I would not launch immediately. Investigate the guardrail more deeply — is it consistent across segments? Is it mechanistically connected to the treatment? Consider running a larger experiment to resolve the uncertainty on the guardrail.

↳ **Follow-up 2:** "Stakeholder insists on launching despite a failed experiment. What do you do?"
Present data clearly and quantify the expected harm. Escalate to data leadership if needed. Propose alternatives: iterate on the feature, run a follow-up test, launch to a limited segment with monitoring. Document your recommendation regardless of the outcome.

↳ **Follow-up 3:** "What is a launch criteria document and why is it valuable?"
A pre-commitment document specifying: primary metric, guardrails, significance threshold, MDE, experiment duration, and decision rules — written before the experiment runs. Prevents goalpost-moving and p-hacking. Aligns stakeholders before anyone sees results.

🚩 **Red flags:** Launching based on secondary metrics when primary is null. Not knowing what to do with guardrail violations.

⭐ **L5/L6 addition:** Discuss decision theory approach — framing the launch decision as minimizing expected loss given uncertainty. Bayesian methods give the probability of each outcome; combine with cost of false positive and false negative to make the optimal decision under uncertainty.

---

## Day 26 — Bayesian A/B Testing

**🎤 Interviewer:** "What is Bayesian A/B testing and how does it differ from frequentist?"

**💡 Model Answer:**
In frequentist A/B testing, we compute p-values and reject or fail to reject H₀. The output is binary. In Bayesian A/B testing, we compute the posterior distribution over the effect size given the data and a prior. The output is a full probability distribution — we can say "there's an 87% probability treatment is better" or "the expected loss from choosing control is 0.3%."

Key differences:
- Bayesian is interpretable: probability statements about hypotheses
- No fixed sample size needed — update continuously as data arrives
- Sensitive to prior choice
- No multiple testing correction needed in the same framework

↳ **Follow-up 1:** "How do you choose a prior?"
For a metric like CTR: use a Beta distribution with parameters set from historical data (e.g., last month's CTR). Or use a weakly informative prior that assumes the effect is probably small. The prior should be pre-specified and documented. Sensitivity analysis: check how much the conclusion changes with different priors.

↳ **Follow-up 2:** "What is expected loss and how do you use it for decisions?"
Expected loss from choosing B over A = E[max(0, μ_A - μ_B)] — the expected amount B is worse, weighted by probability. If expected loss < threshold (e.g., 0.1%), you can launch B. It's a risk-calibrated decision criterion, more nuanced than a binary threshold.

↳ **Follow-up 3:** "What's the main criticism of Bayesian A/B testing?"
Prior sensitivity — if your prior is wrong, your posterior can be biased. Also, different teams using different priors get different answers from the same data. Harder to audit and standardize. Frequentist methods are more objective and easier to communicate.

🚩 **Red flags:** Thinking Bayesian and frequentist always give the same answer — they can diverge substantially with small samples.

⭐ **L5/L6 addition:** Discuss Thompson Sampling — a Bayesian bandit algorithm that samples from posterior distributions to make assignment decisions. Connection between Bayesian inference and online decision-making.

---

## Day 27 — Dilution Problem

**🎤 Interviewer:** "40% of users assigned to treatment never actually see the new feature. How does this affect your experiment?"

**💡 Model Answer:**
This is the dilution problem. The true causal effect of the feature (on users who see it) is diluted by the 60% who don't — they're counted in treatment but experienced the control. This shrinks the observed treatment effect toward zero.

If the true effect on exposed users is δ, and only 40% are exposed, the observed ITT effect is approximately 0.4 × δ.

↳ **Follow-up 1:** "What is Intent-to-Treat (ITT) analysis?"
Analyze all users as assigned, regardless of actual exposure. ITT gives an unbiased estimate of the policy effect — "what happens if we roll this out, knowing some users won't encounter it?" It's conservative and valid.

↳ **Follow-up 2:** "What is Complier Average Causal Effect (CACE) / LATE?"
The effect on users who would actually comply with the treatment. Estimated via instrumental variables, using assignment as the instrument and actual exposure as the treatment. CACE = ITT effect / compliance rate. Only valid under monotonicity and exclusion restriction assumptions.

↳ **Follow-up 3:** "When would you report CACE vs ITT?"
ITT for launch decisions (reflects real-world rollout). CACE for feature evaluation (what's the true effect on users who experience it?). Always report both with clear labeling.

🚩 **Red flags:** Not knowing the terms ITT / CACE, or thinking you should just analyze exposed users only (that's biased — exposure is not random).

⭐ **L5/L6 addition:** Filtering to only exposed users creates selection bias (exposure is itself affected by the treatment). Instrument variables / CACE is the right approach. Discuss the monotonicity assumption — no users would be de-exposed by treatment assignment.

---

## Day 28 — Interleaving

**🎤 Interviewer:** "What is interleaving and when would you use it over a standard A/B test for ranking systems?"

**💡 Model Answer:**
Interleaving is an evaluation method for ranking algorithms where results from two algorithms are combined on the same page and shown to a single user. User behavior (clicks) reveals which algorithm produced better results, without needing to split users into groups.

Advantages over A/B: dramatically more sensitive (detects the same signal with 100–1000x less traffic), faster results, no between-subject variance.

Used at Google, Bing, Netflix for evaluating search/recommendation ranking changes.

↳ **Follow-up 1:** "How does balanced interleaving work?"
Alternate which algorithm places its top result at each rank position. Then track which algorithm's results get more clicks. This removes position bias — neither algorithm is systematically advantaged by higher positions.

↳ **Follow-up 2:** "What can't interleaving measure that A/B testing can?"
Long-term effects, engagement metrics beyond immediate clicks, multi-session behavior, business metrics (revenue, retention). Interleaving only measures immediate result quality — it's a fast signal, not a comprehensive one. Use interleaving for ranking hypothesis filtering, A/B for final validation.

↳ **Follow-up 3:** "What is team-draft interleaving vs balanced interleaving?"
Team-draft: each algorithm alternately selects its next highest-ranked result, avoiding duplicates. Balanced: strictly alternates positions regardless of duplicates. Team-draft is more commonly used; balanced can have issues when algorithms overlap heavily.

🚩 **Red flags:** Not knowing what interleaving is, or claiming it replaces A/B testing entirely.

⭐ **L5/L6 addition:** Describe the online evaluation pipeline: interleaving for quick ranking signal → A/B test for metric validation → holdback for long-term measurement. Three-stage funnel for responsible evaluation.

---

## Day 29 — Experimentation Ethics

**🎤 Interviewer:** "What ethical considerations apply to A/B testing? Have you thought about when testing is wrong?"

**💡 Model Answer:**
A/B testing is powerful but has ethical limits. Key considerations:

1. **Harm:** Never deliberately degrade a control group experience to make treatment look better by contrast.
2. **Consent:** Users implicitly consent to improvement experiments, but deliberate manipulation (dark patterns) is unethical.
3. **Vulnerable populations:** Extra caution for experiments on children, users in distress, or financially vulnerable users.
4. **Privacy:** Experiments that reveal sensitive user attributes through behavior require review.
5. **Facebook's emotional contagion study (2014):** Manipulated news feeds to induce positive/negative emotions without consent — a widely criticized example.

↳ **Follow-up 1:** "Is it ethical to test pricing variations on different users?"
It's legally and ethically complex. Dynamic pricing based on personal data can constitute price discrimination in some jurisdictions. Most companies avoid user-level pricing experiments and use geo or time-based designs instead.

↳ **Follow-up 2:** "Should users know they're in an experiment?"
Industry norm varies. GDPR and CCPA require transparency about data collection but not always about experimentation. Best practice: include in privacy policy, don't run experiments with potential for significant harm without explicit consent.

↳ **Follow-up 3:** "What is an institutional review process for experiments?"
Some companies (especially those in health, finance, or with large scale) have ethics review boards for experiments. They review experiments for potential harm, fairness implications, and privacy considerations before launch. Data scientists should proactively flag borderline cases.

🚩 **Red flags:** "Ethics isn't really a concern in product A/B testing." Naïve view that randomization makes everything acceptable.

⭐ **L5/L6 addition:** Discuss fairness in experimentation — if an experiment randomly assigns some users to a worse experience, are there equity implications? E.g., if control users are disproportionately in a lower-income segment, a worse experience has disparate impact.

---

## Day 30 — Regression to the Mean

**🎤 Interviewer:** "What is regression to the mean and how can it mislead A/B test analysis?"

**💡 Model Answer:**
Regression to the mean: extreme values tend to be less extreme on re-measurement, purely due to random variation. If you target users who had unusually low retention last month, they'll likely improve next month — not because of your intervention, but because their previous low performance was partly random.

In A/B testing: if you segment post-hoc on extreme values of a metric, those users will appear to improve regardless of treatment. This can create a false impression of a treatment effect.

↳ **Follow-up 1:** "How do you avoid regression to the mean artifacts in your analysis?"
Always measure treatment vs control, not before vs after. Proper randomization ensures both groups regress to the mean at the same rate — the comparison is still valid. Pre-post analysis without a control group is vulnerable to RTM.

↳ **Follow-up 2:** "You're told 'we gave our worst-performing users the new feature and they improved dramatically.' How do you respond?"
This is classic RTM contamination. Without a control group of equally bad-performing users, you can't attribute the improvement to the feature. The users would likely have improved regardless. Proper experiment design is the antidote.

↳ **Follow-up 3:** "How does RTM relate to the winner's curse in experiments?"
The winner's curse in experiments: when you run many experiments and select the one with the highest observed effect, you've selected partly on the basis of a lucky draw. The true effect is smaller — the observed winner overstates its real effect. Replications typically show smaller effects. Related to multiple testing and publication bias.

🚩 **Red flags:** Not knowing what RTM is, or not recognizing it as a threat to before-after analysis.

⭐ **L5/L6 addition:** Discuss Stein's paradox — when estimating multiple effects simultaneously, shrinkage estimators (like James-Stein) dominate the MLE by pulling estimates toward the mean. This is the optimal response to the RTM phenomenon at scale.

---

## Days 31–50: Continuing L4 Foundation

---

## Day 31 — Stratified Randomization

**🎤 Interviewer:** "What is stratified randomization and when does it help?"

**💡 Model Answer:**
Stratified randomization pre-divides users into groups (strata) based on a key covariate (e.g., country, device type, user tenure), then randomizes within each stratum. This guarantees balance on those covariates, even in small samples.

Benefits: reduces variance of the effect estimate, ensures no accidental imbalance on important covariates, improves power. Especially valuable when the covariate strongly predicts the outcome.

↳ **Follow-up 1:** "How is this different from post-stratification?"
Stratified randomization is done at assignment time. Post-stratification is a statistical adjustment done after the fact — reweight observations by stratum to correct for imbalance. Post-stratification is valid but less efficient than pre-stratification.

↳ **Follow-up 2:** "What happens if you stratify on too many variables?"
With many strata and few users, some strata may have very few observations — randomization within tiny strata is less meaningful. Limit to 2–3 key stratification variables. Alternatively, use a covariate-adaptive randomization algorithm.

↳ **Follow-up 3:** "When doesn't stratification help?"
When the stratification variable isn't correlated with the outcome. Stratifying on hair color for a purchase experiment adds complexity without benefit. Only stratify on strong predictors of the outcome metric.

🚩 **Red flags:** Confusing stratified randomization with stratified analysis (post-hoc segmentation).

⭐ **L5/L6 addition:** Discuss rerandomization — run randomization multiple times, keep only assignments that satisfy balance criteria on key covariates. More powerful than stratification for high-dimensional covariate sets.

---

## Day 32 — Holdback Groups

**🎤 Interviewer:** "What is a holdback group and how is it different from a standard A/B test control?"

**💡 Model Answer:**
A holdback group is a small percentage of users (e.g., 1–5%) who are permanently excluded from a feature even after it's launched to everyone else. Used to measure long-term effects that can't be captured in a short A/B test — e.g., habit formation, 90-day retention, churn effects.

Standard control: temporary, during the experiment. Holdback: permanent, post-launch.

↳ **Follow-up 1:** "What metrics would you use a holdback to measure?"
Long-term retention (90-day, 180-day), lifetime value, subscription renewal rates, feature habituation curves, cumulative notification effects. Any metric where the treatment effect takes months to manifest.

↳ **Follow-up 2:** "What are the ethical and UX concerns with holdbacks?"
Holding users back from a beneficial feature for months can harm those users (if the feature is genuinely valuable). Holdback users receive a degraded experience relative to the rest of the user base. Balance learning value against user harm. Make holdbacks small (1%) and time-limited.

↳ **Follow-up 3:** "How do you handle holdbacks for features with network effects?"
If the feature has network effects, the holdback users interact with fully-treated users — their experience is influenced by the treatment, even though they're not in it. The holdback estimate in this case includes both direct effects and spillover, making interpretation complex.

🚩 **Red flags:** Not knowing the difference between holdback and control, or not understanding long-term measurement use cases.

⭐ **L5/L6 addition:** Cumulative holdbacks — keeping a small fraction of users off ALL new features simultaneously. Measures the aggregate value of the entire experimentation program. Used to justify the experimentation investment to leadership.

---

## Day 33 — Metric Variance Reduction Techniques

**🎤 Interviewer:** "What techniques can you use to reduce metric variance in an experiment?"

**💡 Model Answer:**
1. **CUPED:** Subtract pre-experiment covariate correlation. Most powerful method.
2. **Outlier capping/trimming:** Remove or winsorize extreme values.
3. **Log transformation:** For right-skewed distributions.
4. **Stratification:** Pre-stratify on high-variance subgroups.
5. **Ratio metrics:** Revenue/session instead of revenue/user (reduces between-user variance if sessions are controlled).
6. **Switching to a lower-variance metric:** Binary conversion vs continuous revenue.

↳ **Follow-up 1:** "Which of these is most commonly used in industry and why?"
CUPED — because it's mathematically principled (no bias), doesn't require changing the metric, can reduce variance by 30–70%, and is straightforward to implement. Outlier capping is also common but introduces mild bias.

↳ **Follow-up 2:** "Can you combine multiple variance reduction techniques?"
Yes — apply CUPED first, then cap outliers on the CUPED-adjusted metric. Or use stratified CUPED. The order matters slightly; generally apply CUPED before capping.

↳ **Follow-up 3:** "What's the tradeoff between variance reduction and interpretability?"
The more you transform a metric, the harder it is to interpret. "CUPED-adjusted log-winsorized revenue" is statistically clean but hard to explain to stakeholders. Balance rigor with communication. Report the raw effect in interpretable units; use transformed metric for inference.

🚩 **Red flags:** Not knowing CUPED or only knowing outlier removal.

⭐ **L5/L6 addition:** ML-based variance reduction — train a regression model on pre-experiment features to predict the outcome metric, then use residuals as the analysis metric. More flexible than linear CUPED, can capture non-linear relationships.

---

## Day 34 — Carryover Effects

**🎤 Interviewer:** "What are carryover effects and how do you handle them in sequential experiments?"

**💡 Model Answer:**
Carryover effects occur when the effect of a previous experiment persists and contaminates a subsequent experiment on the same users. Common in personalization — if users' preferences or model weights are updated by experiment A, experiment B may see a different baseline than expected.

Handling strategies: washout period between experiments, separate user cohorts for sequential experiments, model and adjust for the carryover effect, or design experiments that explicitly account for history.

↳ **Follow-up 1:** "How long should a washout period be?"
Depends on the carryover mechanism. If it's a model weight effect, washout = model retraining cycle. If it's a user behavior habit, washout could be weeks. There's no universal rule — understand the persistence of the specific effect.

↳ **Follow-up 2:** "What is crossover design and when is it used?"
A crossover (or within-subject) design exposes the same user to both treatment and control at different times. Reduces between-subject variance because each user is their own control. Requires no carryover (washout period). Used in ad testing, search ranking experiments.

↳ **Follow-up 3:** "How is a switchback experiment different from a crossover?"
Both alternate treatment over time, but switchback is at the market level (all users in a market switch simultaneously) while crossover is at the user level (individual users alternate). Switchback handles marketplace interference; crossover reduces user-level variance.

🚩 **Red flags:** Not knowing what carryover effects are, or thinking washout always solves the problem.

⭐ **L5/L6 addition:** Discuss the Balaam design and other Latin square designs for crossover experiments with multiple treatments — minimizes carryover bias across multiple conditions.

---

## Day 35 — Surrogate Metrics

**🎤 Interviewer:** "Your long-term goal is 1-year retention but experiments only run for 2 weeks. How do you handle this?"

**💡 Model Answer:**
Use a surrogate metric — a short-term metric that's highly correlated with and causally upstream of the long-term goal. For retention, surrogates might be: 7-day retention, session frequency in week 2, feature adoption rate, or NPS proxy metrics.

The risk: the surrogate may not perfectly predict the long-term metric. Optimizing a surrogate can mislead if the correlation breaks down.

↳ **Follow-up 1:** "How do you validate a surrogate metric?"
Retrospective analysis: look at past experiments where you have both short-term and long-term outcomes. Check if experiments that moved the short-term metric consistently moved the long-term metric in the same direction and magnitude. Prentice criterion: the surrogate fully captures the treatment effect on the outcome.

↳ **Follow-up 2:** "Give an example where a surrogate metric misleads."
Click-through rate as a surrogate for user satisfaction. A clickbait change dramatically increases CTR but users are disappointed with content — long-term engagement drops. The surrogate moved in the opposite direction from the true goal.

↳ **Follow-up 3:** "What is the OEC (Overall Evaluation Criterion) and how does it address this?"
OEC is a weighted composite of multiple metrics designed to capture overall user value. At Microsoft, OEC combines session success rate, session length, query reformulation rate, etc. By incorporating multiple signals, it's harder to game and more predictive of long-term outcomes than any single metric.

🚩 **Red flags:** Just picking any fast-moving metric without validating its connection to the business goal.

⭐ **L5/L6 addition:** Discuss causal mediation analysis — formally testing whether the surrogate fully mediates the treatment's effect on the outcome. Allows quantification of how much indirect effect goes through the surrogate vs direct paths.

---

## Day 36 — External Validity

**🎤 Interviewer:** "Your A/B test ran on US users only. The PM wants to roll out globally. What concerns do you have?"

**💡 Model Answer:**
External validity — the results from US users may not generalize to:
1. **Different languages/cultures:** UI changes may have different effects in different cultural contexts
2. **Different infrastructure:** Mobile vs desktop mix differs by country; performance varies by network
3. **Different user behavior:** Purchasing behavior, social norms, and product usage patterns vary globally
4. **Regulatory context:** Some features may not be legal in all markets

I'd recommend: run a follow-up experiment in key international markets (EU, APAC) before global rollout. At minimum, monitor key metrics by region after launch with rapid rollback capability.

↳ **Follow-up 1:** "How would you design a multi-market experiment efficiently?"
Use a geo-based factorial design — randomize at market level within each region. Ensures regional effects are captured while reducing the number of separate experiments needed.

↳ **Follow-up 2:** "What is the difference between internal and external validity?"
Internal validity: the experiment correctly estimates the causal effect within its scope. External validity: the effect generalizes to other populations, contexts, or times. RCTs (and A/B tests) maximize internal validity; generalizability requires additional evidence.

↳ **Follow-up 3:** "When would you NOT need to worry about external validity?"
When the treatment is generic enough to work across populations (e.g., basic UX improvements like reducing load time), or when you're launching only to the exact population tested. The less similar the target population to the experiment population, the more external validity matters.

🚩 **Red flags:** Assuming US results automatically generalize, or not understanding the distinction between internal and external validity.

⭐ **L5/L6 addition:** Meta-analysis of experiments across multiple markets — pool effect estimates from regional experiments with appropriate weighting to estimate a global average treatment effect.

---

## Day 37 — Triggering and Exposure Logging

**🎤 Interviewer:** "What is trigger analysis in A/B testing and when is it necessary?"

**💡 Model Answer:**
Not all assigned users will actually encounter the feature being tested. Trigger analysis restricts analysis to users who were exposed to the feature (triggered the treatment). This is more sensitive than analyzing all assigned users (ITT), but requires careful implementation.

Key requirement: the trigger condition must be identical for control and treatment users — you log a "triggered" event when a user would have seen the feature, regardless of their assignment. This is called "logging the trigger in control."

↳ **Follow-up 1:** "Why must you log the trigger for control users?"
If you only log triggered users in treatment, you're selecting on different criteria in control vs treatment. Triggered treatment users are self-selected (they hit a condition); triggered control users must be defined by the same condition. Without this, the comparison is biased.

↳ **Follow-up 2:** "What are the risks of trigger analysis?"
If trigger rate differs between control and treatment (SUTVA violation — the treatment itself affects whether users hit the trigger), you have a biased sample. Always check that trigger rates are similar between groups. If they differ, prefer ITT.

↳ **Follow-up 3:** "How does trigger analysis affect sample size?"
Only triggered users are included in analysis. If 20% of users trigger the feature, your effective sample size is 20% of assigned users — you need to assign 5x more users to achieve the same power. Account for this in experiment design.

🚩 **Red flags:** Analyzing only triggered treatment users without logging control triggers — a very common implementation bug.

⭐ **L5/L6 addition:** Discuss the difference between "trigger on intent" vs "trigger on exposure" — intent-based triggers (user tries to use the feature) vs exposure-based (feature rendered on screen). Intent triggers are noisier but closer to causal mechanism; exposure triggers are cleaner but may miss abandoned attempts.

---

## Day 38 — Experiment Interaction Effects

**🎤 Interviewer:** "Two experiments are running on the same users simultaneously. How do you check if they interfere with each other?"

**💡 Model Answer:**
Experiments can interfere if they modify the same user experience or if their effects interact. Steps to check:

1. **Check for overlap:** How many users are in both experiments? If small, interaction effects are limited.
2. **Factorial analysis:** Treat the combination of assignments as four groups: AA, AB, BA, BB. Test the interaction term (treatment_A × treatment_B). A significant interaction means the experiments aren't independent.
3. **Business logic review:** Do the features modify the same surface? If experiment A changes a button and experiment B changes the same page layout, they may conflict.

↳ **Follow-up 1:** "What if you find a significant interaction between two experiments?"
Don't interpret either experiment in isolation. Run a clean factorial design that explicitly tests both features simultaneously and estimates their interaction. Report the joint effect.

↳ **Follow-up 2:** "What is orthogonal experiment design?"
Ensuring experiments are statistically independent — assignment to one experiment is uncorrelated with assignment to another. Achieved by using independent hash salts for different experiments. Orthogonality ensures the experiments don't contaminate each other's estimates.

↳ **Follow-up 3:** "Can you always run experiments orthogonally?"
Not if they're on the same feature surface. If two experiments both modify the home screen, they can't be truly independent. One solution: mutual exclusion — users are in at most one experiment at a time on a given surface. Reduces interference at the cost of slower experimentation.

🚩 **Red flags:** Not knowing that simultaneous experiments can interfere, or assuming orthogonal hashing always prevents interaction.

⭐ **L5/L6 addition:** Discuss the experimentation layer framework (Google OEI) — different layers are orthogonal by design, each with independent randomization. Features in different layers are tested simultaneously without interference; features in the same layer are mutually exclusive.

---

## Day 39 — Funnel Analysis in Experiments

**🎤 Interviewer:** "Your A/B test shows treatment increases product page views by 15% but checkout conversion rate drops by 3%. Net revenue is flat. How do you interpret this?"

**💡 Model Answer:**
The treatment is driving more top-of-funnel activity (page views) but attracting lower-intent users. This is a classic funnel composition shift — the treatment changes WHO enters the funnel, not just what they do once in it.

15% more page views × (1 - 0.03) lower conversion = approximately net flat revenue. But this depends on absolute numbers. Need to check: absolute conversions, revenue per visitor, and whether the lower-intent users have value (e.g., browsing now, buying later).

↳ **Follow-up 1:** "How do you diagnose funnel composition shifts?"
Compare user segment distributions between control and treatment at each funnel stage. Are the users reaching checkout in treatment systematically different from control? Look at pre-experiment features (past purchase history, session depth) of users at each stage.

↳ **Follow-up 2:** "What metric should be primary when funnel composition shifts?"
Revenue per exposed user (not revenue per visitor or revenue per checkout initiator). This is a user-level metric that captures both the increased top of funnel AND the lower conversion rate without compositional bias.

↳ **Follow-up 3:** "When is it okay to launch despite lower conversion rate?"
If revenue per user is flat/positive AND the lower-intent users have long-term value (e.g., they become loyal customers), or if there's a strategic goal to grow top-of-funnel awareness. Decision depends on the business model and long-term value of those marginal users.

🚩 **Red flags:** Focusing only on one funnel metric without considering the full picture. Concluding "the experiment failed" based only on conversion rate.

⭐ **L5/L6 addition:** Discuss mediation analysis within funnels — decompose the total treatment effect into the path through page views, through conversion rate, and any direct effects. This reveals which mechanism is driving the revenue outcome.

---

## Day 40 — Measuring Long-Term Effects

**🎤 Interviewer:** "How do you measure the long-term impact of a feature that you launched 6 months ago?"

**💡 Model Answer:**
Several approaches:
1. **Holdback group:** If you maintained a holdback, compare holdback users to fully-treated users at 6 months.
2. **Pre-post analysis with control:** If you have a holdback or a comparable control group, difference-in-differences gives a clean estimate.
3. **Synthetic control:** Construct a synthetic control group from users who couldn't receive the feature (e.g., different market).
4. **Interrupted time series:** Model the metric trend before launch and extrapolate forward; compare to actual post-launch values.
5. **Cohort analysis:** Compare retention curves of cohorts that first encountered the product before vs after the feature launched.

↳ **Follow-up 1:** "What's the problem with before-after analysis for long-term effects?"
Many things change in 6 months — seasonality, other product launches, market conditions. Without a concurrent control, you can't attribute metric changes to the specific feature. Every change is confounded.

↳ **Follow-up 2:** "What is an interrupted time series and what assumptions does it require?"
Model the metric as a time series with a structural break at the launch date. Estimate the counterfactual trend (what would have happened without launch) and compare to actual. Key assumption: the pre-launch trend would have continued unchanged. Violated if other changes occurred simultaneously.

↳ **Follow-up 3:** "Can you use difference-in-differences for this?"
Yes, if you have a control group unaffected by the launch (e.g., a geographic holdout, or users on a platform where the feature wasn't launched). DiD controls for common time trends. Key assumption: parallel trends in the pre-period — validate by checking that control and treatment moved similarly before launch.

🚩 **Red flags:** Not having a plan beyond "look at the metric over time."

⭐ **L5/L6 addition:** Discuss the Rubin Causal Model framework for these post-launch analyses — clearly defining the potential outcomes and counterfactual of interest. What exactly are you estimating? ATE? ATT? Effect at 6 months vs cumulative?

---

## Day 41 — Estimating Experiment Value

**🎤 Interviewer:** "How do you estimate the dollar value of an A/B test result before launching?"

**💡 Model Answer:**
Value = Effect size × Scale × Time horizon

1. **Effect size:** CI from the experiment (use lower bound for conservative estimate)
2. **Scale:** Current user base or traffic that will see the feature
3. **Time horizon:** How long will this effect persist? (accounting for novelty decay, competition)
4. **Metric monetization:** Convert metric lift to revenue (e.g., CTR → conversions → revenue per conversion)

Example: 1% CTR lift × 10M impressions/day × $0.05 revenue per click = $5,000/day = $1.8M/year.

↳ **Follow-up 1:** "What uncertainties do you communicate alongside this estimate?"
CI on the effect estimate, uncertainty in metric-to-revenue conversion, risk of external validity failure, potential for novelty decay, interaction effects with other features. Present as a range, not a point estimate.

↳ **Follow-up 2:** "What is the opportunity cost of not launching?"
Each day of delay foregoes the expected lift. If the experiment was positive and conclusive, the cost of delay is: effect × scale × days delayed. This is a lever for prioritization discussions with engineering.

↳ **Follow-up 3:** "How do you value experiments that produced null results?"
Null results have learning value — they tell you the feature doesn't work, saving future investment. Value = cost of building a feature that would have been shipped without the experiment, multiplied by probability it would have been shipped. Null results prevent bad launches, which is often worth more than positive experiments.

🚩 **Red flags:** Not knowing how to translate experiment metrics to business value, or ignoring uncertainty in the estimate.

⭐ **L5/L6 addition:** Expected value under uncertainty — use the full posterior distribution of effect sizes (or CI) to compute expected revenue, integrating over possible effect sizes. Accounts for uncertainty better than using the point estimate.

---

## Day 42 — Experiment Governance

**🎤 Interviewer:** "How do you prevent p-hacking and bad statistical practices at an organizational level?"

**💡 Model Answer:**
Governance mechanisms:
1. **Pre-registration:** Document H₀, primary metric, MDE, α, duration BEFORE starting. Lock in the analysis plan.
2. **Single primary metric:** Only one metric drives the launch decision. Secondary metrics are exploratory.
3. **Automated platforms:** Calculate required sample size upfront; don't show results until sample size is reached (or use sequential testing).
4. **Peer review:** Data scientists review each other's experiment designs.
5. **Experiment culture:** Celebrate null results, don't punish them. Don't reward "shipping" — reward learning.
6. **Post-mortem reviews:** Audit launched features against experiment predictions.

↳ **Follow-up 1:** "What does a pre-registration document contain?"
Hypothesis, primary metric and direction, guardrail metrics, MDE and justification, α and power, randomization unit, experiment duration, pre-specified segments for subgroup analysis, definition of "launch" outcome.

↳ **Follow-up 2:** "How do you handle the tension between speed and rigor?"
Use sequential testing methods (faster conclusive decisions), CUPED (less traffic needed), tiered review (quick review for low-risk experiments, full review for high-stakes). Speed and rigor aren't always in conflict — proper design upfront often saves time vs running inconclusive experiments.

↳ **Follow-up 3:** "How would you build experimentation culture at a company that doesn't have it?"
Start by showing the value of one great experiment that prevented a bad launch. Build simple tooling. Share null results openly. Connect metric movements to business outcomes. Train teams on basic statistical concepts. Create lightweight templates for experiment design.

🚩 **Red flags:** Thinking individual vigilance is sufficient — governance must be systematic.

⭐ **L5/L6 addition:** Discuss the FAANG scale — at Google or Meta, thousands of experiments run simultaneously. Governance at that scale requires: automated pre-checks, ML-based anomaly detection, experiment interaction monitoring, and dedicated experimentation teams.

---

## Day 43 — Quasi-Experimental Methods

**🎤 Interviewer:** "When can't you run a proper A/B test and what do you do instead?"

**💡 Model Answer:**
A/B tests require the ability to randomly assign users to conditions. This isn't always possible: regulatory constraints, features that can't be partially rolled out, ethical constraints, or irreversible interventions. Quasi-experimental alternatives:

1. **Difference-in-differences:** Compare change over time between treated and untreated groups.
2. **Regression discontinuity:** Exploit natural cutoffs in assignment.
3. **Instrumental variables:** Use an exogenous variable that affects treatment but not outcome directly.
4. **Synthetic control:** Construct a counterfactual from untreated units.
5. **Propensity score matching:** Match treated and control units on observed covariates.

↳ **Follow-up 1:** "What is the key assumption behind DiD?"
Parallel trends — in the absence of treatment, the treated and control groups would have followed the same trend. Validate by checking pre-treatment trend alignment.

↳ **Follow-up 2:** "When would you use regression discontinuity?"
When assignment is determined by crossing a threshold on a continuous variable. E.g., users who cross 100 purchases get a loyalty reward; compare users near 99 vs 101 purchases. Local causal effect at the cutoff. Assumes no manipulation of the assignment variable around the cutoff.

↳ **Follow-up 3:** "How much do you trust quasi-experimental results vs RCTs?"
Less. Quasi-experiments rely on untestable assumptions (parallel trends, exclusion restriction). They're valuable when RCTs are impossible but results should be held to higher scrutiny, replicated across methods, and treated as directional rather than definitive.

🚩 **Red flags:** Thinking observational analysis is equivalent to an experiment, or not knowing any quasi-experimental methods.

⭐ **L5/L6 addition:** Discuss the credibility revolution in economics (Angrist, Imbens, Card — 2021 Nobel Prize) — the formalization of quasi-experimental methods as the gold standard when RCTs aren't feasible.

---

## Day 44 — A/B Testing for Rare Events

**🎤 Interviewer:** "The metric you care about (purchase) only happens for 0.1% of users. How do you design a powerful experiment?"

**💡 Model Answer:**
Rare events require enormous sample sizes for direct measurement. Options:

1. **Funnel-up metric:** Test a higher-funnel metric correlated with purchase (e.g., add-to-cart rate, which is more frequent) as primary metric. Purchase as secondary.
2. **Longer experiment:** More time = more events.
3. **Increased traffic allocation:** Broader rollout to treatment.
4. **Bayesian methods with informative priors:** Use historical purchase data to inform priors, reducing sample needed.
5. **Pooling across segments:** Combine data across related markets or time periods.
6. **Sensitivity analysis:** Accept larger MDE — you can only detect large effects.

↳ **Follow-up 1:** "What if your funnel-up metric doesn't correlate with purchase?"
Validate the correlation first. If add-to-cart doesn't predict purchase (e.g., users abandon carts at high rates), it's not a valid surrogate. Look further up the funnel for something that does correlate, or reframe what you're measuring.

↳ **Follow-up 2:** "How do you calculate sample size for a 0.1% conversion rate?"
Use the proportion z-test formula. With p=0.001 and MDE=0.0001 (10% relative lift), you'd need: approximately (1.96+0.84)² × 2 × 0.001 × 0.999 / (0.0001)² ≈ 156 million users per group. That's often infeasible — hence the need for funnel-up metrics.

↳ **Follow-up 3:** "What is a negative binomial regression and when would you use it?"
For count outcomes (e.g., number of purchases) that are overdispersed (variance > mean), negative binomial regression models the distribution better than Poisson. Useful when purchase counts per user have high variance (most users buy 0, a few buy many).

🚩 **Red flags:** Powering for the rare event without considering funnel-up alternatives, or not knowing the sample size implications.

⭐ **L5/L6 addition:** Discuss survival analysis for time-to-event outcomes — instead of "did they purchase," analyze "how long until purchase." Cox proportional hazards model can extract more information from the same data by using time-to-event rather than binary outcome.

---

## Day 45 — Variance of Ratio Metrics

**🎤 Interviewer:** "Your metric is revenue per session (not revenue per user). What statistical challenges does this create?"

**💡 Model Answer:**
Revenue per session is a ratio metric where both numerator (revenue) and denominator (sessions) are random variables. Standard formulas for variance of proportions or means don't apply to ratios.

Two approaches:
1. **Delta method:** First-order Taylor approximation of the variance of a ratio.
2. **Bootstrap:** Resample users (not sessions), compute the ratio statistic each time, build empirical distribution.

Critical: resample at the user level, not session level — sessions from the same user are correlated.

↳ **Follow-up 1:** "Write out the delta method variance for a ratio."
If μ_Y = E[Y] (revenue), μ_X = E[X] (sessions), r = μ_Y/μ_X:

Var(r) ≈ (1/n) × [Var(Y)/μ_X² - 2(μ_Y/μ_X)Cov(X,Y)/μ_X² + (μ_Y²/μ_X⁴)Var(X)]

↳ **Follow-up 2:** "Why can't you just compute revenue per session for each user and then run a t-test?"
You can — this is valid and equivalent to the delta method under mild conditions. It correctly handles the correlation between numerator and denominator at the user level. Many practitioners use this approach as a simpler alternative.

↳ **Follow-up 3:** "Is revenue per session or revenue per user a better metric?"
Depends on what you're controlling. If treatment changes session frequency (e.g., increases sessions), revenue per session can be misleading — users have more sessions but same revenue per session, masking a potential per-user decline. Revenue per user is more comprehensive as the primary metric.

🚩 **Red flags:** Treating ratio metrics as simple continuous metrics, or not knowing the delta method.

⭐ **L5/L6 addition:** Discuss when ratio metrics are appropriate — when you want to control for denominator variation (e.g., revenue per active day when activity itself varies). Contrast with when they introduce bias in causal interpretation.

---

## Day 46 — Experiment Logging and Data Quality

**🎤 Interviewer:** "What data quality issues can invalidate an A/B test and how do you catch them?"

**💡 Model Answer:**
Key data quality issues:

1. **Assignment logging failure:** Some users' assignments not logged → incomplete group data
2. **Event duplication:** Double-counting conversions/clicks
3. **Missing events:** Server errors causing lost outcome data
4. **Time zone bugs:** Events attributed to wrong day
5. **Bot traffic:** Bots skewing metrics (often asymmetrically if IP filtering affects one group)
6. **Client-side vs server-side mismatch:** Assignment logged server-side, outcome logged client-side — join failures

Detection: SRM checks, metric sanity checks (plausible ranges), trend monitoring, cross-source validation.

↳ **Follow-up 1:** "How do you detect event duplication?"
Look at per-user event counts distribution. If many users have implausibly high counts (e.g., 100 purchases in an hour), duplication is likely. Cross-validate with backend database counts. Check for duplicate event IDs.

↳ **Follow-up 2:** "How do you handle bot traffic in experiments?"
Identify bots using: IP reputation lists, user agent strings, click pattern analysis (too fast, too regular). Filter bots before analysis — but filter them symmetrically (same criteria for control and treatment). If bot traffic differs between groups (SRM cause), investigate the asymmetry.

↳ **Follow-up 3:** "What is the impact of 5% data loss that's random (MCAR)?"
If data loss is completely random (MCAR — Missing Completely At Random), the analysis is still unbiased but slightly underpowered (you have fewer effective observations). If data loss is related to the treatment (MNAR — Missing Not At Random), estimates are biased. Always investigate missingness patterns.

🚩 **Red flags:** Not having a data quality checklist, or assuming the data is clean without verification.

⭐ **L5/L6 addition:** Build automated data quality monitoring into the experimentation platform: real-time SRM alerts, event rate monitors (alert if event rate drops >20%), automated A/A checks on a rolling basis.

---

## Day 47 — Bayesian vs Frequentist in Practice

**🎤 Interviewer:** "Your company is deciding between Bayesian and frequentist A/B testing frameworks. What do you recommend?"

**💡 Model Answer:**
Both are valid; the right choice depends on organizational needs.

**Frequentist advantages:** Objective, no prior sensitivity, standard in industry, easier to audit, well-understood error rates, easier to communicate to non-statisticians ("significant" is a familiar concept).

**Bayesian advantages:** Intuitive output (probability statements), valid for continuous monitoring without peeking problems, handles prior knowledge formally, gives expected loss for decision-making.

**My recommendation:** Frequentist with sequential testing for most experiments (handles peeking, widely understood). Bayesian for high-frequency personalization decisions where you're making many decisions and want to minimize total regret.

↳ **Follow-up 1:** "What would make you choose Bayesian for a specific use case?"
Personalization systems, content ranking, ad targeting — where you make millions of decisions, have strong priors from historical data, and want to minimize expected loss across decisions. Multi-armed bandits are naturally Bayesian.

↳ **Follow-up 2:** "How do you explain p-values to a non-technical PM?"
"Imagine the feature had zero effect. The p-value is the probability of seeing results this extreme just by chance. 5% means we'd see this kind of result 1 in 20 times even if nothing was real. We've set our threshold such that we only call something a win when there's a low chance of being fooled by randomness."

↳ **Follow-up 3:** "What's wrong with the statement 'p < 0.05 means there's a 95% chance the feature works'?"
Classic misinterpretation. The p-value is about data probability under H₀, not hypothesis probability. Assigning probability to hypotheses is Bayesian, not frequentist. The correct statement: "If the feature had no effect, there's only a 5% chance we'd see this or a more extreme result."

🚩 **Red flags:** Strong dogmatic preference for one approach without understanding the tradeoffs.

⭐ **L5/L6 addition:** Discuss empirical Bayes — use data to set priors automatically. Pool information across many experiments to estimate the prior distribution of effect sizes, then use this as the prior for each individual experiment. Used at Netflix and Airbnb.

---

## Day 48 — Search Experiments

**🎤 Interviewer:** "How would you design an A/B test for a change to a search ranking algorithm?"

**💡 Model Answer:**
Search ranking experiments have special challenges: (1) implicit feedback (clicks, not ratings), (2) position bias (users click top results more), (3) query mix variation (different queries may have different sensitivity to the change).

Approach:
1. **Interleaving first:** Blend results from two algorithms per query, use clicks to identify which algorithm produced better results. 100–1000x more sensitive than user-level split.
2. **User-level A/B test:** After interleaving shows promise, run a full A/B test with user-level assignment. Measure session success rate, query reformulation rate, long-click rate.
3. **Metrics:** Avoid raw CTR (position-biased). Use normalized CTR, session success rate, reformulation rate.

↳ **Follow-up 1:** "How do you handle position bias in click data?"
Use position-adjusted metrics (e.g., clicks weighted by inverse position), or use models that explicitly debias position effects (propensity-weighted click models). Or use metrics that are less position-biased (e.g., time-to-next-query).

↳ **Follow-up 2:** "Your new ranking algorithm improves CTR by 2% but query reformulation increases by 5%. How do you interpret this?"
Mixed signal. More clicks but more reformulations suggests users are clicking but not finding what they need — surface quality may be worse despite higher CTR. Prioritize the reformulation signal; it's a stronger indicator of user satisfaction failure.

↳ **Follow-up 3:** "How do you test a ranking change that only affects rare queries (long tail)?"
Bucket users by query type (head vs torso vs tail). Analyze the long-tail segment separately. May need to run the experiment longer to accumulate enough tail-query events. Or analyze at the query level rather than user level for tail queries.

🚩 **Red flags:** Using raw CTR without acknowledging position bias, or not knowing interleaving.

⭐ **L5/L6 addition:** Counterfactual evaluation using logged data — use inverse propensity scoring to evaluate a new ranking policy using historical data without running an experiment. Connects to offline evaluation before online testing.

---

## Day 49 — Experimentation ROI

**🎤 Interviewer:** "A VP asks you to justify the cost of running an experimentation program. How do you frame it?"

**💡 Model Answer:**
Frame it in three ways:

1. **Bad launch prevention:** What would happen if we shipped features without testing? Estimate: % of experiments that show negative effects × average revenue impact of a bad launch × number of experiments per year. Even preventing 2 bad launches/year often exceeds program cost.

2. **Positive value shipped:** Sum of revenue attributed to positive experiments shipped. A/B results give counterfactual estimates of what was gained vs not shipping.

3. **Compounding learning:** Experimentation culture creates an organizational knowledge base. Over time, teams make better product decisions, intuition improves, and experiment success rates increase. Long-term ROI compounds.

↳ **Follow-up 1:** "How do you measure experiment success rate and what's a healthy number?"
Success rate = % of experiments where primary metric is significantly positive. Industry benchmarks: ~10–30% for mature product areas (most ideas don't work), up to 50%+ for product areas with strong theory. Low success rate indicates you're testing the right things; high success rate might mean you're playing it too safe.

↳ **Follow-up 2:** "What if the VP says 'we don't need experiments, we trust our PMs'?"
Present evidence that intuition is systematically overconfident. Microsoft's data: 2/3 of ideas that PMs were confident about failed in A/B testing. Even senior leaders' ideas fail at high rates. Experimentation replaces HiPPO (Highest Paid Person's Opinion) with data.

↳ **Follow-up 3:** "How does experiment velocity relate to product velocity?"
High experiment velocity = more learning cycles per time period = faster convergence to product-market fit. Each experiment, even negative, rules out a hypothesis and focuses resources. Companies with high experiment velocity (Booking.com: 1000+/year) consistently outperform on user experience.

🚩 **Red flags:** Unable to connect experimentation to business outcomes. Pure statistical framing without business context.

⭐ **L5/L6 addition:** Marginal value of experimentation — at what point do you have enough experiments running? Model the marginal learning value of each additional experiment versus the opportunity cost of running it (traffic, engineering time). Optimum isn't "as many as possible."

---

## Day 50 — Comprehensive Case Study (L4 Capstone)

**🎤 Interviewer:** "We're launching a new notification system. Design the complete A/B test from scratch."

**💡 Model Answer:**

**Step 1 — Define the question**
We want to know: does the new notification system increase 7-day retention vs the current system?

**Step 2 — Hypothesis**
H₀: New system has no effect on 7-day retention. H₁: New system changes 7-day retention (two-tailed).

**Step 3 — Metrics**
- Primary: 7-day retention rate (% of users active on day 7)
- Guardrails: notification opt-out rate, session duration, app store rating
- Secondary: Day 1/3 retention, notification CTR, notification volume

**Step 4 — Sample size**
Baseline 7-day retention: 40%. MDE: 2% absolute lift (5% relative). α=0.05, power=0.80.
n ≈ 2 × (1.96+0.84)² × 0.4 × 0.6 / (0.02)² ≈ 4,700 users per group.
With 50K DAU, that's <1 day of traffic — run for 14 days minimum (weekly cycles).

**Step 5 — Randomization**
Unit: user ID. Hash(user_id + experiment_id) % 100 → 0–49 control, 50–99 treatment.

**Step 6 — Pre-launch checks**
A/A test on the logging pipeline. SRM check. Covariate balance check.

**Step 7 — Analysis**
Two-proportion z-test on day-7 retention. Check guardrails. Segment analysis (new vs returning users, iOS vs Android). Time-trend check for novelty effects.

**Step 8 — Decision**
Launch if: primary metric significant and positive + no guardrail violations + effect is stable over experiment window.

↳ **Follow-up 1:** "Retention improved 3% but opt-out rate increased by 0.5% (borderline p=0.07). Do you launch?"
Borderline opt-out increase is a red flag. Long-term, high opt-out rates degrade the notification channel value. I would not launch — iterate to reduce opt-out while maintaining retention benefit. Possibly: personalize notification frequency, improve content relevance.

↳ **Follow-up 2:** "Results show new system works for Android but not iOS. What next?"
Check if the iOS difference is statistically significant (pre-specified or post-hoc?). If robust, investigate: is the feature implemented correctly on iOS? Are notifications delivered differently? Consider Android-only launch while fixing iOS. Run a dedicated iOS experiment.

↳ **Follow-up 3:** "How would you measure the long-term impact of notifications on retention?"
Holdback 1% of users from the notification system permanently. Compare 90-day and 180-day retention against fully-notified users. Also model: do notifications increase short-term engagement at the cost of long-term fatigue? Use time-to-uninstall as a long-term signal.

🚩 **Red flags:** Missing any major step (especially sample size, guardrails, or SRM check). Launching with a guardrail violation.

⭐ **L5/L6 addition:** Discuss notification personalization as a future step — use experiment results to train a model that predicts optimal notification frequency per user. Transition from one-size-fits-all A/B to multi-armed bandit per user segment.

---

*End of Days 1–50 (L4 Foundation)*
*Days 51–100 (L6 Advanced) continue in the next file.*

---

**Quick Reference: L4 Topics Covered**
- Days 1–5: Hypothesis, p-value, errors, power
- Days 6–10: Sample size, randomization, control, A/A, CI
- Days 11–15: Metrics, duration, SRM, novelty, peeking
- Days 16–20: Multiple testing, CUPED, SUTVA, network effects, marketplace
- Days 21–25: Segmentation, practical significance, revenue, selection bias, launch decisions
- Days 26–30: Bayesian intro, dilution, interleaving, ethics, regression to mean
- Days 31–35: Stratification, holdbacks, variance reduction, carryover, surrogate metrics
- Days 36–40: External validity, triggering, interactions, funnel analysis, long-term effects
- Days 41–45: Value estimation, governance, quasi-experiments, rare events, ratio metrics
- Days 46–50: Data quality, Bayesian vs frequentist, search experiments, ROI, case study
