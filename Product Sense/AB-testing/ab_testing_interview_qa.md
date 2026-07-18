# A/B Testing — 100+ Interview Q&A
### FAANG-Level | With Follow-Ups | Exhaustive

> Format: **Q** → Answer → ↳ Follow-up Q → Answer
> Covers: Foundations · Statistics · Design · Pitfalls · Advanced · Product Sense

---

## SECTION 1: Foundations

---

**Q1. What is A/B testing and why do we do it?**
A/B testing is a controlled experiment where users are randomly split into groups — one sees variant A (control) and one sees variant B (treatment) — and we measure which performs better on a metric. We do it to make causal claims about product changes rather than relying on correlation or intuition.

↳ **Why causal and not just observational analysis?**
Observational data has confounders — users who opt into a feature may behave differently regardless of the feature. Randomization ensures the only systematic difference between groups is the treatment itself, allowing causal inference.

---

**Q2. What is the difference between A/B testing and multivariate testing?**
A/B testing changes one variable at a time. Multivariate testing (MVT) tests multiple variables simultaneously (e.g., headline + button color) and measures all combinations. MVT requires much larger sample sizes and is harder to interpret.

↳ **When would you choose MVT over A/B?**
When you suspect interaction effects between variables (e.g., headline copy might work differently depending on button color). But it needs significantly more traffic to be powered.

---

**Q3. What is a hypothesis in an A/B test?**
A statistical hypothesis is a formal statement about a population parameter. In A/B testing:
- **Null hypothesis (H₀):** No difference between control and treatment.
- **Alternative hypothesis (H₁):** There is a difference (two-tailed) or treatment is better (one-tailed).

↳ **Should you use a one-tailed or two-tailed test?**
Two-tailed by default in industry — it detects both improvements and degradations. One-tailed is used only when degradation is impossible or irrelevant, which is rare. One-tailed tests have more power but risk missing regressions.

---

**Q4. What is a p-value? What does p=0.03 mean?**
The p-value is the probability of observing a result at least as extreme as what was observed, assuming H₀ is true. p=0.03 means there's a 3% chance of seeing this result (or more extreme) if there were truly no effect.

↳ **Does p=0.03 mean there's a 97% chance the treatment works?**
No — this is a common misconception. The p-value is not the probability that H₁ is true. It's a measure of data compatibility with H₀. You need Bayesian reasoning to assign probability to hypotheses.

↳ **What's wrong with p-hacking?**
If you run multiple tests and pick the one with p < 0.05, you inflate the false positive rate. With 20 tests, you expect one false positive by chance even when all null hypotheses are true.

---

**Q5. What is statistical significance?**
A result is statistically significant if the p-value falls below a pre-chosen threshold α (usually 0.05), meaning we reject H₀. It tells us the effect is unlikely due to chance — but says nothing about the size or practical importance of the effect.

↳ **Difference between statistical significance and practical significance?**
Statistical significance is about noise; practical significance is about magnitude. A result can be statistically significant but too small to matter (e.g., 0.001% CTR lift on a button). Always look at effect size alongside p-value.

---

**Q6. What is Type I error and Type II error?**
- **Type I error (α):** Rejecting H₀ when it's actually true — a false positive. Probability = α (significance level).
- **Type II error (β):** Failing to reject H₀ when H₁ is true — a false negative. Probability = β.

↳ **What is statistical power?**
Power = 1 - β. It's the probability of detecting a true effect. Industry standard is 80% power. Low power means real effects are often missed.

↳ **How do you increase power?**
Increase sample size, increase effect size (not in your control), increase α (risky), reduce variance (e.g., use CUPED), or run the experiment longer.

---

**Q7. What is the significance level α and how do you choose it?**
α is the acceptable false positive rate — how often you're willing to declare a winner when there isn't one. Typically 0.05 (5%). For high-stakes decisions (e.g., billing changes), use 0.01. For exploratory tests, 0.10 may be acceptable.

↳ **What happens if you set α too low?**
You need more data to reach significance, tests run longer, and you miss real effects (higher Type II error). There's a fundamental power-significance tradeoff.

---

**Q8. What is a confidence interval?**
A 95% CI is a range of values that, if you repeated the experiment many times, would contain the true parameter 95% of the time. It gives a range of plausible effect sizes, not just a binary yes/no.

↳ **Is a 95% CI the same as saying there's a 95% probability the true value is in that interval?**
No — this is the frequentist vs Bayesian distinction. In frequentist stats, the true value is fixed (not random); the interval is what varies across experiments. A Bayesian credible interval does give a probability statement about the parameter.

---

## SECTION 2: Test Design

---

**Q9. How do you calculate sample size for an A/B test?**
Sample size depends on:
- α (significance level, e.g., 0.05)
- Power (1-β, e.g., 0.80)
- Baseline metric value (e.g., current CTR = 10%)
- Minimum detectable effect (MDE) — smallest lift you care about

Formula uses the z-scores for α and β and the variance of the metric. Larger MDE → smaller sample needed. Smaller variance → smaller sample needed.

↳ **What is MDE and how do you choose it?**
MDE (Minimum Detectable Effect) is the smallest effect size worth detecting. It's a business decision — if a 0.1% CTR lift isn't actionable, don't power for it. Set MDE based on what would change a launch decision.

↳ **What if you can't get enough sample size?**
Run the test longer, increase traffic allocation, use variance reduction (CUPED), simplify the metric, or reconsider whether the change is worth testing.

---

**Q10. How do you decide how long to run an A/B test?**
Run until the pre-calculated sample size is reached — not based on peeking at results. Duration = required sample size / daily traffic per variant. Also account for day-of-week effects (run at least 1–2 full weeks to capture weekly cycles).

↳ **What happens if you stop early when you see significance?**
This is "peeking" and inflates Type I error. The p-value fluctuates over time — stopping when p < 0.05 happens to be crossed biases toward false positives. Use sequential testing methods (e.g., always-valid p-values) if you need to monitor continuously.

---

**Q11. What is randomization and how do you implement it?**
Randomization assigns users to variants in a way that's unbiased and reproducible. Common approach: hash the user ID + experiment ID, map to a bucket (0–99), assign to control/treatment based on bucket ranges. This ensures:
- Same user always sees same variant (consistency)
- Buckets are balanced
- Independent across experiments

↳ **What is a randomization unit and how do you choose it?**
The unit can be user, session, device, or request. Choose the unit that matches the granularity of the decision and avoids SUTVA violations. User-level is most common — avoids within-user contamination across sessions.

---

**Q12. What is the SUTVA assumption?**
SUTVA = Stable Unit Treatment Value Assumption. It requires:
1. No interference between units (one user's treatment doesn't affect another).
2. No hidden versions of treatment.

↳ **When is SUTVA violated in practice?**
- Social networks: showing a feature to one user changes their friends' experience (network effects).
- Marketplaces: treating buyers affects sellers.
- Shared resources: caching, inventory.
↳ **How do you handle SUTVA violations?**
Cluster randomization (randomize by groups), geo-based experiments, or switchback experiments.

---

**Q13. What is a control group and what should it receive?**
The control group receives the current production experience (status quo). It's the baseline. Every experiment should have a control — even "no treatment" is a valid treatment.

↳ **Should the control group receive a placebo?**
Yes, when possible. A/A tests validate the setup. For UI changes, the control sees the old UI. Avoid giving control users nothing when treatment users receive something that creates awareness (novelty effect).

---

**Q14. What is an A/A test and why run one?**
An A/A test splits users into two groups that both receive the same experience. Expected result: no statistically significant difference. Used to:
- Validate the randomization is correct
- Check for pre-existing imbalances
- Estimate false positive rate of your testing system
- Calibrate variance estimates

↳ **If your A/A test shows p < 0.05, what does that mean?**
Something is wrong — possibly biased randomization, instrumentation bugs, selection bias, or a misconfigured test. Investigate before running any A/B tests.

---

**Q15. What is a holdout group?**
A holdout group is a set of users permanently excluded from a feature, even after it launches. Used to measure long-term effects of a feature that might not be visible in short-term A/B tests (e.g., habit formation, churn reduction).

↳ **When would you use a holdout instead of an A/B test?**
When the effect takes months to materialize (e.g., email frequency on annual churn), or when you need to measure cumulative impact of many simultaneous experiments.

---

## SECTION 3: Metrics

---

**Q16. How do you choose a metric for an A/B test?**
A good metric is:
- **Sensitive:** Moves in response to the treatment
- **Attributable:** Caused by the change, not confounders
- **Fast-moving:** Changes within the experiment window
- **Aligned with business goals:** Proxy for what you actually care about

↳ **What is a guardrail metric?**
A metric you monitor to ensure the treatment doesn't cause harm, even if it improves the primary metric. E.g., testing a recommendation algorithm on CTR while guarding against session time reduction.

↳ **What is the difference between a primary metric and a secondary metric?**
Primary metric drives the launch decision. Secondary metrics provide supporting evidence and detect unintended effects. Only the primary metric should be used for the power calculation.

---

**Q17. What is metric sensitivity and why does it matter?**
Sensitivity is how much a metric moves in response to a real effect. Low sensitivity = experiments need more data or time. High variance metrics (like revenue per user) are less sensitive than rate metrics (like CTR).

↳ **How do you improve metric sensitivity?**
Use CUPED (covariate adjustment), cap outliers, use relative rather than absolute metrics, or choose a more granular unit of analysis.

---

**Q18. What is Goodhart's Law and how does it apply to A/B testing?**
"When a measure becomes a target, it ceases to be a good measure." If you optimize for a proxy metric (e.g., clicks), the model may game it in ways that don't align with the true goal (e.g., clickbait that reduces satisfaction).

↳ **How do you guard against this?**
Use a portfolio of metrics, including counter-metrics (e.g., alongside CTR, measure dwell time and return rate). Include user satisfaction proxies like long clicks, ratings, or NPS.

---

**Q19. What is the difference between click-through rate and click-through probability?**
- **CTR:** Total clicks / total page views (can be > 1 if a user clicks multiple times).
- **CTP:** Number of users who clicked / number of users who visited (bounded [0,1], user-level).

CTP is usually preferred for A/B tests because it's user-level, avoids double-counting, and has lower variance.

↳ **Why does the choice of metric unit matter?**
Using event-level metrics when your randomization is user-level violates the independence assumption and underestimates variance, leading to inflated significance.

---

**Q20. How do you handle metrics with high variance (e.g., revenue)?**
- Cap outliers at a percentile (e.g., 99th)
- Use log transformation
- Use CUPED to reduce variance via pre-experiment covariates
- Use median or quantile-based metrics instead of mean
- Increase sample size

↳ **What is CUPED?**
CUPED (Controlled-experiment Using Pre-Experiment Data) uses a pre-experiment covariate (e.g., last week's revenue) to reduce variance in the outcome metric. It works by subtracting the part of variance explained by the covariate. Can reduce variance by 50%+ without introducing bias.

---

## SECTION 4: Statistical Tests

---

**Q21. What statistical test do you use for an A/B test on a proportion metric (e.g., CTR)?**
Two-proportion z-test. Assumes large enough samples (np > 5 for both groups). Test statistic = (p̂₁ - p̂₂) / pooled standard error.

↳ **What if your sample size is small?**
Use Fisher's exact test (works for small counts). Chi-squared test is equivalent to z-test for large samples.

---

**Q22. What test do you use for a continuous metric (e.g., session duration)?**
Welch's t-test (two-sample t-test with unequal variances). More robust than Student's t-test when variances differ between groups.

↳ **When would you use a non-parametric test instead?**
When the metric is heavily skewed, has outliers, or sample size is too small to invoke CLT. Mann-Whitney U test is the non-parametric alternative. Less powerful but more robust.

---

**Q23. What is the Central Limit Theorem and why is it important for A/B testing?**
CLT states that the sample mean of any distribution approaches a normal distribution as sample size grows, regardless of the underlying distribution. This is why t-tests and z-tests work even for non-normal metrics like revenue or session time — with large enough n, the sampling distribution of the mean is approximately normal.

↳ **What is "large enough" n?**
Rule of thumb: n > 30 for roughly symmetric distributions, but n > 1000 may be needed for heavily skewed distributions (e.g., revenue). Always check via bootstrapping if unsure.

---

**Q24. What is bootstrapping and when do you use it?**
Bootstrapping resamples the data with replacement many times, computes the statistic each time, and builds an empirical distribution. Used when:
- Metric distribution is unknown or non-normal
- Analytic formulas for variance don't exist (e.g., median, ratios)
- Sample size is too small for CLT

↳ **What's the tradeoff with bootstrapping?**
Computationally expensive, especially for large datasets. It's accurate but slow. Delta method is an alternative for ratio metrics.

---

**Q25. What is the delta method?**
A technique to approximate the variance of a function of random variables (e.g., ratio metrics like CTR = clicks/sessions). Uses first-order Taylor expansion. More efficient than bootstrapping and gives closed-form variance estimates for ratio metrics.

---

**Q26. What is multiple testing and why is it a problem?**
When you test multiple hypotheses simultaneously, the probability of at least one false positive increases. With 20 independent tests at α=0.05, expected false positives = 1. This is called the familywise error rate (FWER) problem.

↳ **How do you correct for multiple testing?**
- **Bonferroni:** Divide α by number of tests. Very conservative.
- **Holm-Bonferroni:** Sequential, less conservative.
- **Benjamini-Hochberg (FDR):** Controls false discovery rate rather than FWER. Better for many tests.

↳ **When does multiple testing arise in A/B testing?**
- Testing multiple metrics
- Testing multiple variants (A/B/C/D)
- Peeking at results repeatedly (sequential testing issue)
- Segmented analysis post-hoc

---

**Q27. What is sequential testing / always-valid inference?**
Traditional tests require fixed sample size. Sequential testing allows continuous monitoring with valid Type I error control. Methods include:
- **Sequential probability ratio test (SPRT)**
- **Always-valid p-values** (e-values)
- **Group sequential testing** with spending functions

Used at companies like Netflix, Booking.com to allow early stopping while controlling false positives.

---

## SECTION 5: Common Pitfalls

---

**Q28. What is the novelty effect?**
Users behave differently when they encounter something new, not because the feature is better, but because it's unfamiliar. Leads to inflated short-term metrics that decay over time.

↳ **How do you detect and handle it?**
Run the experiment long enough for the novelty to wear off (2–4 weeks). Plot the treatment effect over time — if it decreases monotonically, suspect novelty. Use holdout groups to measure long-term effects.

---

**Q29. What is selection bias in A/B testing?**
When the groups being compared are not comparable — e.g., users assigned to treatment differ systematically from control users, not due to the treatment. Causes: non-random assignment, self-selection, survivor bias.

↳ **How do you detect selection bias?**
Run an SRM check and compare pre-experiment covariates between groups (age, historical behavior). If they differ significantly, the randomization is compromised.

---

**Q30. What is Sample Ratio Mismatch (SRM)?**
SRM occurs when the observed ratio of users in control vs treatment doesn't match the intended ratio (e.g., expected 50/50 but got 48/52). Indicates a bug in randomization, logging, or assignment.

↳ **How do you detect SRM?**
Chi-squared test on the observed group sizes vs expected. Always check for SRM before interpreting results.

↳ **What causes SRM?**
Bots (often filtered unevenly), client-side crashes in one variant, logging failures, CDN caching issues, redirect differences. SRM invalidates the experiment.

---

**Q31. What is survivorship bias in experiments?**
Analyzing only users who complete an action (e.g., only users who made a purchase) instead of all users exposed to the experiment. The groups are no longer comparable because completion itself is affected by the treatment.

↳ **Example?**
Testing a checkout flow change. If treatment reduces cart abandonment, the users who complete checkout in treatment are a different (less price-sensitive) group than control completers. Comparing revenue per purchaser is biased.

---

**Q32. What is primacy effect?**
The opposite of novelty — existing users resist change and perform worse initially with the new variant. Long-term performance may be better once users adapt.

↳ **How do you distinguish novelty from primacy?**
Look at the time trend of the treatment effect. Novelty decays; primacy starts negative and recovers. Stratify by user tenure — new users won't show either effect.

---

**Q33. What is interference / spillover between experiment groups?**
When the treatment given to one user affects another user's behavior. Violates SUTVA. Common in:
- Social networks (viral effects)
- Two-sided marketplaces
- Shared inventory / resources

↳ **Solutions?**
Cluster randomization, ego-network randomization, geo experiments, time-based (switchback) experiments.

---

**Q34. What is the dilution problem?**
When not all users in the treatment group actually receive the treatment (e.g., 40% of treatment users never trigger the new feature). This dilutes the observed effect.

↳ **How do you handle it?**
Use **Intent-to-Treat (ITT)** analysis (analyze all assigned users regardless of exposure) — conservative but unbiased. Or use **Complier Average Causal Effect (CACE)** / instrumental variables for the effect on actually treated users.

---

**Q35. What is Twyman's Law?**
"The more unusual or interesting the data, the more likely it is to be wrong." In A/B testing: dramatic results are often bugs. Always verify large effects with sanity checks before celebrating.

---

**Q36. What is carryover effect?**
When the effect of a previous experiment persists and influences a subsequent experiment on the same users. Common in personalization and ranking systems.

↳ **How do you handle it?**
Introduce a washout period between experiments. Use holdout groups. Design experiments to be independent via careful bucketing.

---

## SECTION 6: Advanced Topics

---

**Q37. What is a factorial experiment?**
An experiment that simultaneously varies multiple factors. A 2x2 factorial has 4 conditions (control/A × control/B). Allows estimation of main effects and interaction effects.

↳ **When is it better than sequential A/B tests?**
When you suspect interactions between features. Sequential tests can't detect interactions and may give misleading results if features interact.

---

**Q38. What is a bandit algorithm and how does it differ from A/B testing?**
A/B testing uses fixed allocation (e.g., 50/50) throughout. Multi-armed bandits dynamically allocate more traffic to better-performing variants during the experiment, reducing regret (lost value from showing inferior variants).

↳ **What is the exploration-exploitation tradeoff?**
Exploration = trying variants to learn. Exploitation = showing the best known variant. Bandits balance this; A/B tests are pure exploration until decision time.

↳ **When would you prefer A/B over bandits?**
When you need clean statistical inference (bandits introduce bias), when experiments are short, or when you care about rigorously controlling Type I error.

---

**Q39. What is a switchback experiment?**
A time-based experiment where the treatment alternates between groups over time (e.g., odd hours = treatment, even hours = control). Used in marketplaces and ride-sharing where user-level randomization causes interference.

↳ **What are the limitations?**
Temporal confounding (time of day effects), requires careful design to separate treatment from time effects, can't detect carry-over effects easily.

---

**Q40. What is geo-based experimentation?**
Randomizing at the geographic unit level (city, DMA, country) rather than user level. Used when user-level randomization causes interference (e.g., advertising, marketplace liquidity).

↳ **What are the challenges?**
Fewer randomization units → lower power. Geographic units are not identical → need matched pairs or synthetic control. Spillover across geographic boundaries.

---

**Q41. What is a synthetic control method?**
Used when you have very few treated units (e.g., one country gets a new policy). Creates a "synthetic" control unit as a weighted combination of untreated units that best matches the treated unit pre-treatment.

---

**Q42. What is difference-in-differences (DiD)?**
A causal inference method that compares the change in outcome over time between a treatment group and a control group. Controls for time-invariant confounders and common time trends.

↳ **Key assumption?**
Parallel trends — in the absence of treatment, both groups would have followed the same trend. This is untestable but can be partially validated by checking pre-treatment trends.

---

**Q43. What is regression discontinuity design (RDD)?**
A quasi-experimental method that exploits a cutoff in an assignment variable. Users just above and below the cutoff are compared — they're assumed to be similar except for the treatment. E.g., users with exactly 100 days of tenure get a feature; users with 99 days don't.

---

**Q44. What is Bayesian A/B testing?**
Instead of p-values, uses prior beliefs + observed data to compute a posterior distribution over the effect. Reports the probability that treatment is better than control, and the expected loss from choosing incorrectly.

↳ **Advantages over frequentist?**
- Intuitive output: "87% probability treatment is better"
- No fixed sample size needed
- Naturally incorporates prior knowledge
- No multiple comparison issues in same framework

↳ **Disadvantages?**
Sensitive to prior choice. Less standardized. Harder to audit.

---

**Q45. What is the false discovery rate (FDR)?**
FDR = expected proportion of rejected hypotheses that are false positives. Benjamini-Hochberg procedure controls FDR at level q. More powerful than FWER-controlling methods when running many tests.

↳ **When would you prefer FDR over FWER control?**
When running large-scale experiments with many metrics/segments, and some false positives are acceptable. FWER is better when any false positive is very costly (e.g., medical trials).

---

## SECTION 7: Network Effects & Marketplace

---

**Q46. How do you run A/B tests on a two-sided marketplace?**
User-level randomization causes interference (treating buyers affects sellers). Solutions:
- **Supply-side holdout:** Hold out a fraction of supply from the new feature
- **Geo-based randomization:** Randomize at city level
- **Switchback experiments:** Time-based alternation
- **Ego-network randomization:** Assign by social clusters

↳ **Which approach does Airbnb/Uber typically use?**
Geo experiments and switchbacks are common. Ego-network randomization for social features.

---

**Q47. How do you measure network effects in an experiment?**
Look for spillover from treatment to control (if control metrics improve, network effects are present). Use cluster-level randomization and compare cluster-level outcomes. Can also use the "market-level" design to capture equilibrium effects.

---

**Q48. What is a holdback experiment?**
A long-running experiment where a small fraction of users (e.g., 1%) is permanently held back from a launched feature to measure cumulative long-term impact. Different from a standard holdout used during launch decision.

---

## SECTION 8: Product & Decision Making

---

**Q49. How do you make a launch decision from an A/B test?**
1. Check for SRM
2. Verify primary metric is significant with right direction
3. Check guardrail metrics are not harmed
4. Assess effect size for practical significance
5. Consider secondary metrics for holistic picture
6. Consider novelty / primacy effects
7. Align with business strategy

↳ **What if the primary metric is not significant but secondary metrics improve?**
Do not launch based on secondary metrics alone — this is p-hacking. Consider running a larger test to detect a real but small effect, or reconsider whether you're measuring the right thing.

---

**Q50. What do you do when an experiment shows a statistically significant negative result?**
Do not launch. Investigate root cause — is the effect consistent across segments? Is it a bug? Is the metric sensitive to novelty? If the negative is real, roll back and learn.

↳ **What if the feature is strategically important?**
Escalate. Show stakeholders the data. Propose iteration. Never overrule clear statistical evidence for political reasons.

---

**Q51. How do you handle an inconclusive A/B test?**
- Verify power was adequate (was the test underpowered?)
- Increase sample size and re-run
- Check if effect exists in specific segments
- Re-examine the metric — maybe you're measuring the wrong thing
- Consider that there may genuinely be no effect

↳ **What's the difference between "no effect" and "insufficient power to detect effect"?**
With enough power, absence of significance supports absence of effect. Without adequate power, a null result is uninformative. Always report confidence intervals, not just p-values.

---

**Q52. What is the difference between launching a feature and learning from an experiment?**
Launch decision uses a pre-specified primary metric and threshold. Learning uses the full distribution of metric movements, segment analyses, and qualitative signals to understand why the effect occurred — informing future experiments.

---

**Q53. How do you handle pressure from stakeholders to ship a feature that failed an A/B test?**
Present the data clearly. Quantify the expected harm of shipping (e.g., "this feature will reduce retention by X% = $Y in annual revenue"). Propose alternatives: iterate on the feature, run a follow-up experiment, or ship to a limited segment. Never manufacture significance.

---

## SECTION 9: Segmentation & Heterogeneity

---

**Q54. What is subgroup analysis (segmentation)?**
Analyzing the treatment effect within subgroups (e.g., by device, geography, user tenure). Can reveal heterogeneous treatment effects (HTE) — the feature may help some users and hurt others.

↳ **What is the risk of subgroup analysis?**
Multiple comparisons inflate false positive rate. Pre-register segments of interest before running the experiment. Post-hoc segment analysis is exploratory only.

---

**Q55. What is a heterogeneous treatment effect (HTE)?**
When the treatment effect varies across individuals or subgroups. Detecting HTEs requires pre-registered interaction tests or methods like causal forests, X-learner, T-learner.

↳ **Why does this matter for product decisions?**
You may launch to only the benefiting segment, or redesign the feature for the hurt segment. Personalization at scale uses HTE estimates directly.

---

**Q56. What is the interaction test for HTE?**
Include an interaction term in a regression: outcome ~ treatment + subgroup + treatment × subgroup. The interaction coefficient tests whether the effect differs between subgroups. Requires larger sample than a main effect test.

---

## SECTION 10: Experiment at Scale

---

**Q57. How do large companies (Google, Meta, Netflix) run thousands of experiments simultaneously?**
- **Orthogonal experimentation:** Users are divided into layers; within each layer, users are re-randomized. Experiments in different layers are independent.
- **Traffic splitting frameworks:** Experimentation platforms assign users to multiple non-overlapping or orthogonal buckets.
- **Automated guardrails:** Triggered alerts if guardrail metrics degrade.

↳ **What is layering / overlapping experiments?**
Multiple experiments run simultaneously on the same users. Requires independence across experiments or careful disentanglement. Meta's PlanOut, Google's Overlapping Experiment Infrastructure (OEI) handle this.

---

**Q58. What is an experimentation platform and what does it do?**
A system that handles: user assignment, experiment configuration, metric logging, statistical analysis, and reporting. Examples: Airbnb's ERF, Netflix's Experimentation Platform, Uber's XP, Meta's Deltoid.

---

**Q59. How do you prioritize which experiments to run?**
Prioritize by:
- Expected impact (size × likelihood of success)
- Cost to build and run
- Strategic alignment
- Learning value even if result is null
Use an ICE (Impact, Confidence, Ease) or RICE framework for backlog prioritization.

---

**Q60. What is experiment velocity and why does it matter?**
The number of experiments run per unit time. Higher velocity = faster learning cycle = faster product improvement. Companies like Booking.com and LinkedIn report running 1000+ experiments/year as a competitive advantage.

---

## SECTION 11: Special Scenarios

---

**Q61. How do you A/B test a search ranking algorithm?**
Use interleaving — mix results from two algorithms on the same page and measure which results users click. Much more sensitive than user-level split testing. Detects small ranking improvements with less traffic.

↳ **What is balanced interleaving?**
Alternate which algorithm gets to place its top result first. Removes position bias from the comparison.

---

**Q62. How do you test email or notification campaigns?**
Randomize at user level. Control = no email / current email. Treatment = new email. Measure downstream actions (open rate, click rate, conversion, long-term retention). Beware: email timing effects, unsubscribe rate as guardrail.

---

**Q63. How do you A/B test pricing?**
Extremely sensitive — users may communicate prices to each other (social interference), causing unfairness and SUTVA violations. Solutions:
- Geo-based pricing experiments
- Time-based (switchback) experiments
- Segment by new vs existing users
- Legal review required in many jurisdictions

---

**Q64. How do you test features for low-traffic pages or rare events?**
- Use a more sensitive metric (funnel step before the rare event)
- Run longer experiments
- Pool data across similar surfaces
- Use Bayesian methods with informative priors
- Consider quasi-experimental designs

---

**Q65. How do you handle cookie deletion / user ID changes?**
Users who delete cookies or switch devices get re-assigned randomly, which can contaminate groups (control users see treatment). Solutions: login-based assignment (more stable), device fingerprinting (privacy concerns), or accept the noise (it biases toward null, conservative).

---

## SECTION 12: Metrics Deep Dive

---

**Q66. What is NDCG and when is it used as an A/B metric?**
Normalized Discounted Cumulative Gain — measures ranking quality, accounting for position (higher positions matter more). Used in search, recommendations. Compares actual ranking to ideal ranking of relevant items.

↳ **Why discount by position?**
Users are less likely to see/click lower results. Gains from relevant items ranked higher should count more.

---

**Q67. What is long-click rate and why is it used at Google/Bing?**
A long click is when a user clicks a search result and doesn't return to the SERP quickly (indicating satisfaction). More reliable than raw CTR because it measures engagement quality, not just clicking.

---

**Q68. What is Days Active / DAU / MAU and how are they used in experiments?**
- **DAU:** Daily Active Users — short-term engagement
- **MAU:** Monthly Active Users — broader retention signal
- **DAU/MAU ratio:** Stickiness metric — what fraction of monthly users engage daily

These are used as guardrail metrics. A feature shouldn't ship if it reduces DAU/MAU.

---

**Q69. What is session-based analysis vs user-based analysis?**
Session-based: each session is an observation. User-based: each user is an observation (aggregating their sessions). User-based is correct when randomization is at user level — session-based analysis violates independence (same user's sessions are correlated).

---

**Q70. What is the difference between average treatment effect (ATE), ATT, and ATC?**
- **ATE:** Average Treatment Effect — average over the whole population.
- **ATT:** Average Treatment Effect on the Treated — average for those who received treatment.
- **ATC:** Average Treatment Effect on the Control — average for those who didn't.

In randomized experiments, ATE = ATT = ATC. They differ in observational studies.

---

## SECTION 13: Coding & Analysis

---

**Q71. How do you compute a z-test for proportions in Python/SQL?**

```python
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

count = np.array([converted_treatment, converted_control])
nobs = np.array([n_treatment, n_control])
stat, pvalue = proportions_ztest(count, nobs)
```

In SQL:
```sql
SELECT
  variant,
  COUNT(*) AS n,
  SUM(converted) AS conversions,
  AVG(converted) AS rate
FROM experiment_table
GROUP BY variant
```

---

**Q72. How do you compute confidence intervals for a proportion?**

```python
from statsmodels.stats.proportion import proportion_confint
lower, upper = proportion_confint(conversions, n, alpha=0.05, method='wilson')
```
Wilson interval is preferred over normal approximation for small samples or extreme proportions.

---

**Q73. How do you compute CUPED in Python?**

```python
import numpy as np

# theta = Cov(Y, X) / Var(X)
theta = np.cov(Y, X_pre)[0,1] / np.var(X_pre)
Y_cuped = Y - theta * (X_pre - X_pre.mean())

# Now run t-test on Y_cuped
from scipy.stats import ttest_ind
stat, pvalue = ttest_ind(Y_cuped[treatment==1], Y_cuped[treatment==0])
```

---

**Q74. How do you detect SRM in code?**

```python
from scipy.stats import chisquare

observed = [n_control, n_treatment]
expected = [total/2, total/2]  # or based on intended split
stat, pvalue = chisquare(observed, f_exp=expected)
if pvalue < 0.01:
    print("SRM detected — investigate before analyzing results")
```

---

**Q75. How do you compute power analysis in Python?**

```python
from statsmodels.stats.power import NormalIndPower

effect_size = (p_treatment - p_control) / np.sqrt(p_pooled * (1 - p_pooled))
power_analysis = NormalIndPower()
n = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8)
```

---

## SECTION 14: Scenario-Based Questions

---

**Q76. DAU drops by 5% the day after you launch a feature. What do you do?**
1. Check for bugs/instrumentation issues first
2. Check SRM
3. Look at the time series — is the drop sustained or a spike?
4. Segment by device, geography, user type
5. Check if it correlates with the feature rollout timing
6. If real and sustained → roll back immediately, investigate root cause

---

**Q77. Your A/B test shows a 10% CTR improvement but revenue is flat. What's happening?**
Possible explanations:
- Users click more but buy less (low-intent clicks)
- Revenue metric has high variance and is underpowered
- There's a segment where revenue drops offsetting gains
- Cannibalization of other surfaces

↳ Actions: Segment revenue analysis, look at revenue per click, check basket size, run longer.

---

**Q78. You run an experiment and find that treatment users click 20% more but only on mobile. Desktop shows no effect. What do you conclude?**
There is a heterogeneous treatment effect — the feature works on mobile but not desktop. Do not average and conclude "10% lift." Instead: consider mobile-only launch, investigate why desktop differs (feature may not render well, interaction model differs), run a dedicated mobile experiment to confirm.

---

**Q79. A PM says "let's just run the test for 3 days — we have enough traffic." What do you say?**
3 days may not capture weekly seasonality (weekday vs weekend behavior differs). You might also hit novelty effects. More importantly: did you pre-calculate required sample size? If that requires 3 days of traffic, fine — but the duration must be pre-specified, not chosen post-hoc.

---

**Q80. You're testing a new onboarding flow. New users love it (conversion +15%) but the effect disappears for returning users who accidentally hit the flow. What do you do?**
Segment the analysis — analyze new users and returning users separately. The aggregate effect is diluted by returning users who aren't the target. Launch the feature only for new users if that segment shows a robust significant effect. Ensure the feature is properly gated by user state in production.

---

**Q81. Two experiments are running simultaneously on the same users. How do you ensure they don't interfere?**
Use orthogonal layers — users are re-randomized independently for each experiment. Verify that the two features don't interact (do they modify the same UI surface?). If they do, either serialize the experiments or use a factorial design.

---

**Q82. You observe a 2% conversion lift but confidence interval is [-0.5%, +4.5%]. What do you recommend?**
The result is not statistically significant — zero is within the CI. Do not launch based on this. Either: (1) run longer to narrow the CI, or (2) if the MDE was set at 2% and this is underpowered, revisit sample size calculation. Present to stakeholders with context: "we see a directional positive signal but cannot rule out noise."

---

**Q83. An experiment has been running for 6 weeks. Is that a problem?**
Potentially yes — long experiments risk: (1) novelty wearing off (good), (2) user composition drift (new users keep entering), (3) seasonal confounds, (4) stakeholder impatience leading to peeking. Check if the sample size was reached earlier — if so, why wasn't it stopped? If still underpowered after 6 weeks, the MDE may be too small to be detectable given your traffic.

---

**Q84. How would you design an experiment to test a new recommendation algorithm on Netflix?**
- **Unit of randomization:** User (consistent experience)
- **Primary metric:** Streaming hours in first 2 weeks
- **Guardrail metrics:** Cancellation rate, title diversity, customer satisfaction score
- **Duration:** 2–4 weeks (capture weekly patterns)
- **SRM check:** Pre-launch
- **Segments:** New vs returning users, content type preferences
- **Novelty concern:** Monitor effect over time, check if new releases confound results
- **Interleaving:** For ranking sensitivity, run interleaving alongside A/B for signal

---

**Q85. What would cause treatment users to have higher revenue but lower sessions?**
Users in treatment may be completing goals faster (more efficient UX) → fewer sessions but more value per session. Or: treatment drives a specific high-value cohort to engage more intensely. Check revenue per session, not just absolute revenue. This is a good result if revenue per user is up.

---

## SECTION 15: Tricky Concepts

---

**Q86. Can you have a statistically significant result with no practical significance?**
Yes. With huge sample sizes, even tiny effects (e.g., 0.001% CTR lift) become significant. Always report effect size and confidence interval alongside p-value. Ask: is this effect large enough to matter for the business?

---

**Q87. Can you have practical significance without statistical significance?**
Yes. With small samples, a large effect (e.g., 15% lift) may not reach significance because the confidence interval is wide. The solution is to increase the sample size, not to lower the significance threshold.

---

**Q88. Why is it wrong to run an experiment until p < 0.05?**
This is "optional stopping" and inflates false positives dramatically. The p-value fluctuates over time — if you check repeatedly and stop when it crosses 0.05, you exploit these fluctuations. Use sequential testing methods (SPRT, always-valid p-values) if you need to monitor continuously.

---

**Q89. What is the difference between internal and external validity?**
- **Internal validity:** The experiment correctly measures the causal effect within the study conditions (right randomization, no confounds).
- **External validity:** The results generalize to the real world / other populations / future time periods.

A/B tests have high internal validity but limited external validity — results may not hold for all geographies, device types, or seasons.

---

**Q90. What is regression to the mean and how does it affect experiments?**
Extreme values tend to be less extreme on re-measurement due to random variation. If you target users who performed unusually poorly (or well) in a baseline period, their metric will naturally improve (or regress) even without treatment. This is why randomization and proper control groups matter.

---

**Q91. What is Berkson's paradox?**
A selection bias where conditioning on a collider (a variable caused by both the treatment and outcome) creates a spurious correlation. In experiments: if you analyze only users who completed a conversion, you may see paradoxical relationships because completion itself is the collider.

---

**Q92. Why might your A/B test results differ from post-launch metrics?**
- Novelty/primacy effects during experiment resolve post-launch
- Holdback group contamination
- Different traffic mix after full rollout
- Interaction with other features launched simultaneously
- Seasonal changes
- Survivorship: experiment may have run on a specific cohort

---

**Q93. What is the interference between the measurement and the event (observer effect)?**
Logging and tracking itself can affect user behavior — e.g., excessive instrumentation slows page load, affecting the metric being measured. Always measure and minimize the performance overhead of experiment infrastructure.

---

**Q94. What is variance inflation from clustering?**
When randomization unit (e.g., user) is different from analysis unit (e.g., session), observations within the same user are correlated. Using standard errors that assume independence underestimates true variance → inflated significance. Use cluster-robust standard errors or delta method.

---

**Q95. What is the "file drawer problem" in experimentation?**
Experiments with null results often don't get reported or acted upon. This creates publication bias in your org's experiment history — you only see the successes. Leads to overconfidence in new features and poor calibration of expected success rates.

---

## SECTION 16: Final Hard Questions

---

**Q96. How would you design an experimentation system from scratch for a mid-sized company?**
Key components:
1. Assignment service — hash-based, deterministic, user/session-level
2. Feature flag system — integrates with assignment
3. Event logging — client + server side
4. Metric pipeline — joins assignments with events, computes per-user aggregates
5. Analysis layer — runs statistical tests, SRM checks, CI computation
6. Dashboard — shows results, guardrails, segment breakdowns
7. Decision workflow — review, approval, launch tracking

---

**Q97. How do you measure the ROI of an experimentation program?**
- Cumulative revenue attributed to shipped experiments
- Experiment velocity (tests per quarter) as a leading indicator
- Win rate — % of experiments that show positive results
- Speed to decision (days per experiment)
- Cost of bad launches prevented by experiments

---

**Q98. What is the difference between a randomized controlled trial (RCT) and an A/B test?**
Conceptually identical — both randomly assign units to treatment and control. RCT is the medical/academic terminology; A/B test is the industry term. Practical differences: RCTs often involve consent, blinding, and regulatory requirements; A/B tests operate at scale without explicit consent in many jurisdictions.

---

**Q99. If you could only track one metric for an experiment, what would it be and why?**
The metric most closely aligned with the long-term value the company creates for users. For a social network: DAU or meaningful sessions. For e-commerce: repeat purchase rate. For streaming: retention / renewal rate. The answer depends on the business — the key is picking a metric resistant to gaming and predictive of long-term health.

---

**Q100. What is the biggest mistake you've seen (or can think of) in A/B testing?**
Common answer: peeking (stopping when significant). Better answer: launching based on surrogate metrics that don't correlate with long-term outcomes — e.g., optimizing for clicks that degrade satisfaction, or optimizing for short-term revenue that increases churn. The hardest problem in experimentation is not statistics — it's choosing the right thing to measure.

---

**Q101. How do you think about experimentation ethics?**
Users deserve consistent, functional experiences. Experiments should not:
- Deliberately degrade experience for the control group
- Test dark patterns or manipulative UX
- Expose users to harm (even for learning)
- Violate privacy regulations (GDPR, CCPA) in data collection

Informed consent norms vary by company. Best practice: have an ethics review for experiments with potential for harm.

---

**Q102. What is the "multiple comparison problem" at the organizational level?**
Even if each experiment is properly designed, if your org runs 1000 experiments with α=0.05, you expect ~50 false positive launches. This is why consistent methodology, pre-registration, and replication matter at scale. Some orgs use lower α (0.01) for high-stakes experiments.

---

*End of A/B Testing Interview Q&A — 102 questions, 17 sections.*
*Pair with: Statistics fundamentals, Product sense, and SQL for complete FAANG prep.*
