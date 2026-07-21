# A/B Testing Interview Prep — SL6 Level
> Google · Meta · Apple | Past Questions + 100 Medium-Hard Q&A

---

## PART A — Real Interview Questions from Google, Meta & Apple

### Google (Data Scientist / Senior DS)

1. **"We launched a new ranking feature in Search. The click-through rate went up 5% but revenue per session dropped 2%. How do you decide whether to ship it?"**
2. **"How would you design an A/B test for a change to Google Maps that might have long-term learning effects?"**
3. **"You ran an experiment and got p = 0.04 on your primary metric, but 20 secondary metrics all showed no change. What do you do?"**
4. **"How do you handle novelty effects when testing UI changes in Gmail?"**
5. **"Walk me through how you'd detect if your randomization is broken in an A/B test with 50M users."**
6. **"A product team wants to test 8 variants simultaneously. What are the risks and how do you manage them?"**
7. **"Your experiment shows a statistically significant lift, but it disappears when you segment by device type. What's happening and what do you conclude?"**
8. **"Google Search runs thousands of experiments simultaneously. How do you ensure they don't interfere with each other?"**
9. **"How would you measure the long-term impact of a change to YouTube recommendations on user retention?"**
10. **"A/B test results are positive in the US but negative in India. Do you ship globally?"**

---

### Meta (Research Scientist / DS)

1. **"How would you test whether adding Stories to Facebook Feed hurts or helps overall engagement?"**
2. **"You see a 3% lift in DAU but a 1.5% drop in time-per-session. How do you frame this trade-off to leadership?"**
3. **"How do you measure the network effects of a feature change on a social platform like Instagram?"**
4. **"An experiment on Messenger shows 0% change for new users but a 6% lift for power users. What's going on and what do you recommend?"**
5. **"How do you design an A/B test that avoids contamination when the treatment is 'show ads to user's friends'?"**
6. **"Meta runs an experiment and the treatment group shows better engagement but more reports of harmful content. How do you decide?"**
7. **"How would you test a change to the News Feed algorithm when users interact with each other?"**
8. **"What is your process for choosing a primary metric vs guardrail metrics for a Marketplace feature test?"**
9. **"An engineer claims a feature shipped 3 weeks ago is causing a long-term decline in retention. How do you investigate?"**
10. **"How do you calculate the minimum detectable effect for an experiment on a metric that's log-normally distributed?"**

---

### Apple (Data Scientist / Senior Analytics)

1. **"How would you A/B test a change to the App Store search algorithm?"**
2. **"You want to test a new onboarding flow for iPhone. How do you randomize when devices are tied to Apple IDs?"**
3. **"How do you measure the causal impact of a Siri improvement on user satisfaction?"**
4. **"Apple releases features to a 5% population first. How do you decide when to ramp to 100%?"**
5. **"How would you test a privacy-related feature change where you have limited data collection?"**
6. **"An iCloud feature shows a significant positive effect in testing, but ops says it will increase server costs by 20%. Walk me through the decision."**
7. **"How do you design an experiment where the unit of randomization is a device but the metric is at the user level?"**
8. **"You want to measure the impact of a Maps feature on driving safety. What metrics would you use and what are the ethical constraints?"**
9. **"How do you A/B test a payment flow change on the App Store given very low event rates?"**
10. **"How would you detect if the Hawthorne effect is influencing results in a study of Apple Watch activity features?"**

---
Here are full SL6-level answers to all 30 interview questions:

---

## Google

**Q1. CTR up 5%, revenue per session down 2% — ship?**

This is a metric conflict requiring decomposition, not a binary call. First, quantify the net business impact: if revenue per session dropped 2% but sessions per user increased due to more clicks, the net revenue effect might still be positive. Compute total revenue impact = (change in sessions) × (change in revenue per session) × (users). If that's positive and the magnitude exceeds the noise floor, the feature is revenue-positive overall.

If total revenue is flat or negative, investigate the mechanism. Did lower-quality results get clicked more (novelty, clickbait)? Did users bounce faster after clicking (lower dwell time, higher reformulation rate)? Pull downstream engagement metrics: pogo-sticking rate, session depth, task completion. A CTR increase that comes from users clicking worse results and immediately bouncing back is not a win.

Also check for Simpson's paradox — revenue per session may be down in aggregate because the feature changed the mix of query types served, not because any individual query type is worse.

Decision framework: if the net revenue effect is positive and quality signals (dwell, task completion, NPS) are stable or improving, ship. If quality signals decline alongside revenue, do not ship — you're trading long-run user trust for a short-run click metric.

---

**Q2. A/B test for a Google Maps change with long-term learning effects**

Long-term learning effects (primacy effects) mean users initially perform worse with a new UI because they're unfamiliar with it, but improve over time. Standard 2-week A/B tests underestimate the eventual benefit.

Design: randomize users at the account level (not session — Maps learns from your history). Run for a minimum of 4 weeks. Plot the treatment effect week-by-week; if the effect grows over time, you're seeing learning. Fit an asymptotic model to the weekly effects to estimate the steady-state lift.

Additionally, segment by new vs. returning users. New users have no habit for the old UI, so their behavior approximates the long-run equilibrium. If new users show a strong, stable positive effect from day 1 while returning users start negative and trend positive, the primacy interpretation is confirmed.

For primary metrics: route efficiency (did users take the optimal route), task completion rate (did they reach the destination successfully), re-route rate (did they need corrections). These are more reliable than raw sessions or time-in-app, which can be ambiguous for a navigation context.

Consider a long-horizon holdout: after shipping to 90%, keep 10% on the old experience and compare 3-month outcomes. This is the only way to confirm the asymptotic estimate.

---

**Q3. p = 0.04 on primary, 20 secondary metrics all zero — what do you do?**

At α = 0.05, a p-value of 0.04 has a reasonable probability of being a false positive, especially since you've looked at 21 metrics total. However, you declared a primary metric upfront, so the multiple testing concern applies primarily to secondary metrics, not the primary. The primary result is valid on its own.

That said, the complete null on 20 secondary metrics is important context. If the feature genuinely lifts the primary metric, you'd typically expect correlated downstream metrics to move directionally as well, even if not significantly. A completely flat secondary profile suggests either: (a) the primary metric move is noise, (b) the feature has a very narrow, targeted effect isolated to exactly that metric, or (c) the secondary metrics are insensitive or measuring different things.

What I'd do: first, check if the primary metric result is robust — does it hold across segments, across time periods within the experiment, across different randomization checks? If robustness checks pass, examine the causal mechanism. Is there a plausible story for why the primary metric moves while everything downstream stays flat? If yes, the result is credible. If no plausible mechanism exists, treat with skepticism.

Recommendation: don't ship on this result alone. Either extend the experiment to increase power on a key correlated secondary metric, or run a follow-up confirmatory experiment with the same primary metric. A single p = 0.04 at the boundary, unaccompanied by any corroborating signal, doesn't meet the bar for a production ship decision.

---

**Q4. Handling novelty effects in Gmail UI changes**

Novelty effects in Gmail are real — Gmail users have deeply ingrained habits built over years, so any UI change will initially disrupt workflows before users adapt.

Detection: plot the daily treatment effect (not cumulative) over the experiment duration. A decaying effect (large in week 1, smaller in week 2, near zero by week 3) is the novelty signature. A growing effect is primacy (users are learning the new UI). A stable effect means neither.

Mitigation strategies, in order of rigor:

Analyze new users only — users who created Gmail accounts after the experiment started have no prior habit. Their response approximates the long-run steady state because they never experienced the old UI. If new users show a stable positive effect, you can be confident the long-run effect is positive.

Run longer — for deeply habitual UIs like Gmail, 6–8 weeks is more appropriate than 2. The effect should stabilize by then.

Fit a decay model — model the treatment effect over time as an exponential decay toward an asymptote: `δ(t) = δ_steady + (δ_0 - δ_steady) * exp(-λt)`. Estimate δ_steady as the long-run effect. If δ_steady is significantly positive, ship.

Cohort-based analysis — within the treatment group, compute the effect size as a function of "days since first exposure." Users 1–7 days post-exposure vs. 14–21 days post-exposure. The latter approximates the stabilized effect.

---

**Q5. Detecting broken randomization in an A/B test with 50M users**

The primary tool is a Sample Ratio Mismatch (SRM) check. Given an intended 50/50 split, if you observe 25.3M in control and 24.7M in treatment, a chi-squared test on those counts will tell you if the deviation is statistically significant. With 50M users, even a 0.1% imbalance (25.025M vs 24.975M) will be highly significant. The SRM check should be run immediately after the experiment starts, not just at the end.

Beyond counts, check covariate balance: are pre-experiment distributions of key user characteristics (tenure, device type, geography, activity level) the same in both arms? Run a logistic regression predicting treatment assignment from these covariates. If the model achieves AUC > 0.52, randomization is likely broken.

Check temporal patterns: plot the assignment rate (treatment/(treatment+control)) over time. It should be stable around 0.5. A sudden shift on a specific date suggests a deployment event corrupted the randomization.

Check by assignment mechanism: if users are assigned via a hash function, verify the hash output distribution is uniform. Run `hash(user_id) mod 100` on a sample and check the distribution.

Common causes with 50M users: cookie deletion (users reassigned on re-visit), bot traffic that isn't filtered consistently between arms, logging pipeline drops that differ between arms, or a gradual rollout system that changed the allocation mid-flight.

If SRM is detected, do not analyze the experiment. The data is untrustworthy. Identify and fix the root cause, then restart.

---

**Q6. Risks of testing 8 variants simultaneously**

The primary risks:

Multiple comparisons: with 8 variants vs. control (8 tests), the family-wise error rate at α = 0.05 is `1 - (0.95)^8 ≈ 34%`. One-third chance of at least one false positive. Mitigation: apply Bonferroni (α = 0.05/8 = 0.006) or Dunnett's test (designed for multiple comparisons against a single control, more powerful than Bonferroni).

Traffic dilution: with 9 arms (8 variants + control), each arm gets 1/9 of traffic. For the same power as a two-arm test, you need roughly the same per-arm sample size, but the experiment runs 4.5x longer to achieve it (since traffic is split across 9 arms instead of 2). At Google scale this is often fine, but it must be planned.

Interaction risks: if the 8 variants differ on multiple dimensions, you can't isolate which dimension drove an effect. If the goal is to understand which specific change works, a factorial design or sequential testing approach is better.

Contamination: if users switch devices or accounts, they might get inconsistent variant assignments.

Management: define a primary metric, pre-specify a winner selection protocol (e.g., pick the highest significant lift, use Bonferroni threshold), and document which variant is the "intended ship candidate" before launching. Never pick the winner after seeing results without adjustment.

---

**Q7. Significant lift disappears when segmented by device type — what's happening?**

Several explanations are possible:

Simpson's paradox: the aggregate shows a lift, but when you condition on device type, the effect disappears or reverses in both segments. This happens when device type is imbalanced between treatment and control and correlates with both the treatment allocation and the metric. The device-stratified analysis is the correct one if device type is a true confounder.

Interaction effect without main effect: the feature only works on one device type (e.g., only on desktop), and that device type has higher baseline values. The aggregate average shows a lift driven entirely by that segment. When you stratify, the desktop segment shows a strong lift and mobile shows zero.

Power issue: the aggregate n is sufficient to detect the effect, but each device stratum has lower n and lower power. The effect was always only real in aggregate and in desktop, and mobile is simply underpowered.

What to conclude: first run a formal interaction test — `outcome ~ treatment * device_type`. If the interaction is significant, HTEs are real. If not, the segmented finding may be noise. Then, ask whether the feature works on the device types where most of your users or revenue are. If it works strongly on desktop and desktop users drive 70% of revenue, a desktop-only ship may be warranted, with the mobile implementation going back for redesign.

Never conclude "the experiment failed" just because a subgroup analysis shows heterogeneity. That's information, not a failure.

---

**Q8. Ensuring thousands of simultaneous experiments don't interfere**

Google's solution is the Overlapping Experiment Framework, which uses independent hash functions for each experiment. A user's assignment in experiment A is determined by `hash(user_id + seed_A) mod 100`, and in experiment B by `hash(user_id + seed_B) mod 100`. Because different seeds produce statistically independent hash outputs, the two experiments are independent even though they run on the same users.

Experiments are grouped into domains (e.g., search ranking, UI, ads). Within a domain, experiments are independent and can overlap. Across domains, you can use disjoint splits when experiments in different domains might interact with each other (e.g., a ranking experiment and an ads experiment that both affect SERP layout).

Interaction detection is automated: monitor whether the presence of experiment A significantly changes the effect of experiment B. This is done by running two-way ANOVA on the (experiment A assignment × experiment B assignment) cells for each pair of overlapping experiments. Flag pairs where the interaction term is significant.

Experiment exclusion: some experiments are classified as "launch experiments" that exclude all other traffic overlaps, used for sensitive or high-risk changes.

The framework's key insight is that statistical independence between experiments doesn't require disjoint user pools — it requires independent randomization mechanisms.

---

**Q9. Measuring long-term impact of a YouTube recommendations change on retention**

Standard A/B test: run the experiment with user-level randomization, measuring D30 and D90 retention as secondary metrics. The problem: D90 is too slow — you'd need to run the experiment for 3+ months just to measure the primary outcome.

Surrogate metric approach: identify fast-moving metrics that are causally predictive of D90 retention. Candidates include: 7-day re-engagement rate (did the user return within a week?), video completion rate (are users finishing what they start?), satisfaction signals (thumbs up/down ratio, explicit "not interested" rate). Validate these surrogates using historical experiments where you know D90 outcomes — compute the correlation between the short-term surrogate and the D90 outcome across past experiments.

Long-horizon holdout: after shipping to 90% of users, keep 10% permanently on the old algorithm. Compare D90 and D365 retention between the holdout and the shipped population. This is the gold standard but takes time.

Hazard modeling: fit a survival model (Cox proportional hazards) to early churn data (first 30 days) in each arm. Compare the hazard ratios. Extrapolate to 90 days using the parametric model. This gives a probabilistic estimate of the long-run effect.

Cohort analysis: after shipping, compare retention curves for cohorts exposed to the new algorithm vs. a matched control cohort (using propensity matching on pre-ship characteristics). Useful but requires careful confounding adjustment.

At SL6, you'd combine the surrogate approach (fast, enables rapid iteration) with long holdouts (slow, ground truth) and use the holdouts to calibrate the surrogates over time.

---

**Q10. A/B test positive in US, negative in India — ship globally?**

Do not ship globally without understanding the mechanism. The different effects are real signal, not noise (assuming both are statistically significant with adequately powered samples for each market).

Investigation steps:

First, run a formal interaction test: `outcome ~ treatment * market`. Confirm the HTE is statistically significant and not just sampling noise.

Then investigate why. Common causes: the feature renders differently in the Indian locale (different fonts, RTL issues, translation quality). The user population differs substantially (device type distribution — India is more mobile and lower-spec hardware; connectivity — the feature may be latency-sensitive; user behavior patterns — session length, query types). The metric itself may behave differently (e.g., "revenue per session" may be near-zero in India if monetization is limited, making a negative effect essentially a noise result).

Decision options: ship only to the US if the Indian market is small and the mechanism is unclear. Ship globally with India excluded if the Indian market is significant and the harm is confirmed. Fix the root cause for India (e.g., optimize for low-bandwidth connections, fix localization) and re-test in India before expanding. If India's metric is near-floor and the negative effect is practically tiny, a global ship may still be net positive.

The worst outcome is shipping globally and causing sustained harm in a market because you didn't investigate a statistically significant warning sign.

---

## Meta

**Q11. Testing whether adding Stories to Facebook Feed hurts or helps overall engagement**

This is one of the hardest Meta-style questions because Stories and Feed are both content and distribution surfaces that compete for attention within the same session.

The core measurement challenge: adding Stories may cannibalize Feed scroll time, meaning individual metrics go in opposite directions. You need a holistic session-level metric, not individual surface metrics.

Design: randomize users. Treatment group sees Stories in Feed; control group does not. Both groups still have access to Stories via the Stories bar (top of app) — so this isn't testing Stories existence, it's testing placement.

Primary metric: total meaningful social interactions (MSI) per session, defined as the composite of comments, reactions, shares, and direct messages weighted by depth of engagement. Alternatively, overall time-well-spent as measured by post-session satisfaction surveys. Do not use raw time-in-app — more time can mean worse experience if users are scrolling mindlessly.

Secondary metrics: Stories views (does placement increase discovery?), Feed scroll depth (does Stories presence reduce feed consumption?), friend-to-friend interactions (does Stories exposure increase direct communication?).

Guardrails: total MSI must not decline, and user-reported app satisfaction (measured via periodic in-app surveys) must not decline.

Network effects consideration: if Stories from treatment users are seen by control users (via their Friends' stories), you have cross-arm contamination. Mitigate by using ego-network cluster randomization — randomize at the level of connected social clusters, not individual users.

---

**Q12. 3% lift in DAU, 1.5% drop in time-per-session — how do you frame this to leadership?**

Don't frame it as a trade-off to resolve — frame it as a distinction between breadth and depth of engagement, and then explain what the data says about which matters more for the business.

The math question first: does total engagement (DAU × time-per-session) go up or down? If DAU +3% and time-per-session -1.5%, total time = 1.03 × 0.985 = 1.015 — a net 1.5% increase in total time. If that's the case, this is actually a positive result and the framing is "broader, slightly shorter sessions."

The interpretation question: why are users coming back more often but spending less time per visit? Hypothesis A: the feature improved content discovery, so users find what they need faster and leave satisfied (positive). Hypothesis B: the feature added friction or reduced content quality, so users bounce faster (negative). Distinguish these using downstream signals: did post-session satisfaction go up or down? Did click-through to content increase? Did reformulation or back-navigation rates change?

The business question: for Meta, does revenue scale more with DAU (ad auction participation depends on users being present) or with session length (more ad inventory per user per day)? The answer is both, but the marginal value differs by market and ad product. Frame the trade-off in terms of revenue impact, not raw metric direction.

Leadership framing: "Total engagement is up 1.5%. Users are returning more frequently but with shorter sessions. We believe this reflects [mechanism]. Our monetization estimate shows a net $X impact. Here's what additional data would resolve any remaining uncertainty."

---

**Q13. Measuring network effects of a feature change on Instagram**

Network effects mean that the value a user gets from the feature depends on how many of their connections also have it. Standard user-level A/B tests ignore this — they assign half of connected users to control and half to treatment, creating an artificial world where no one has all their friends in the same condition.

Design options:

Ego-network randomization: randomize at the level of ego + all their immediate connections (1-hop neighborhood). A user and all their followers/following are in the same arm. Reduces inter-arm contamination but requires identifying non-overlapping ego networks, which is hard in a dense graph.

Graph cluster randomization: use community detection (Louvain algorithm) to partition the social graph into dense, weakly-interconnected clusters. Randomize clusters to treatment or control. Within clusters, most connections are intra-cluster, minimizing cross-arm spillover. Analyze at the cluster level using cluster-robust standard errors.

Two-stage design: randomize the fraction of a user's connections that receive the treatment (low, medium, high saturation). Estimate the spillover function: how does outcome change as a function of connection saturation? This allows decomposing direct effects (from the feature itself) and indirect effects (from connected users also having the feature).

Measurement: compute peer treatment dose for each user (fraction of friends/followers in treatment). Include this as a covariate to separate direct and indirect effects: `Y = β₀ + β₁*T_i + β₂*mean(T_j for j in friends of i) + ε`. The coefficient β₂ is the spillover effect.

---

**Q14. 0% change for new users, 6% lift for power users on Messenger — what's happening?**

Several hypotheses worth investigating:

Baseline difference: power users use Messenger heavily and have many active threads. New users have sparse networks and few conversations. The feature may require a certain density of connections to activate (e.g., a smart reply feature that needs message history, or a group feature that needs existing groups). New users simply lack the substrate for the feature to work.

Feature discoverability: power users actively explore app features; new users follow narrow usage patterns. The treatment may be visible and engaging to power users but effectively invisible to new users who haven't reached the relevant screen.

Metric sensitivity: with near-zero baseline engagement for new users, a 6% lift would also be near-zero in absolute terms and undetectable. Check the absolute effect sizes: if new users average 2 messages/day, a 6% lift would be 0.12 messages/day — undetectable. Power users average 50 messages/day, so 6% = 3 messages — easily detectable. The HTE may be real but the null result for new users may be a power issue, not a true zero.

Recommendation: run a power analysis within the new user segment. What would 80% power look like? If you'd need 10x more new users than you have, extend the experiment or declare new user results inconclusive. If you have sufficient power and new users truly show no effect, ship the feature with the understanding that it's a power user enhancement. Set up onboarding or tooltips to surface the feature to new users at the right moment in their lifecycle (e.g., after their 5th conversation thread).

---

**Q15. Avoiding contamination when the treatment is 'show ads to user's friends'**

This is a network interference problem by design — the treatment is explicitly defined as changing another user's experience based on user A's actions.

The contamination mechanism: if user A (treatment) sees an ad and their friend B (control) sees that same ad because it was targeted to friends of A, then B is in control but exposed to treatment-derived content. Standard user-level A/B is invalid.

Design solution: Cluster randomization by social graph. Partition users into clusters where within-cluster connections are dense (communities). Assign whole clusters to treatment or control. In treatment clusters, all users see friend-activity ads. In control clusters, no users see friend-activity ads. Cross-cluster edges are a small fraction of total edges in a well-partitioned graph, limiting contamination.

Analysis: compute outcomes at the cluster level. Use cluster-level means as the unit of analysis. Run a two-sample test on cluster-level averages. Standard errors must account for within-cluster correlation.

Measuring spillover: in the treatment clusters, compare outcomes for the "friends of users who saw the ad" to the baseline from control clusters. This quantifies the spillover effect — does seeing a friend's ad activity change your behavior, above and beyond being in a treatment cluster?

An alternative: randomize at the friendship pair level. For each (A, B) friend pair, both are assigned to the same arm. Ensures consistent treatment within friend dyads. Harder to implement at Instagram scale.

---

**Q16. Better engagement but more harmful content reports — how do you decide?**

This is a values and ethics question as much as a statistics question. At SL6, you're expected to have a framework, not just say "it depends."

First, characterize both effects precisely. How large is the engagement lift? How large is the harmful content report increase? Are the reports driven by a broader user base seeing more content (scale effect), or by a higher rate of harmful content per impression (quality effect)? These are very different. If the feature increases the exposure of every user to harmful content, that's a quality problem. If harmful content reports increase because more users are engaging with the platform and thus have more opportunities to report, that's a scale effect that may not mean content quality worsened.

Second, decompose the metric: harmful content reports / total content impressions should be the guardrail metric, not raw report count. If the rate is stable or declining despite more total reports, the absolute count increase is a volume effect.

Third, apply a clear decision rule: harmful content safety metrics are non-negotiable guardrails at Meta. An increase in harmful content rate (not just count) is a hard block on shipping, regardless of engagement lift. This is a stated company value, not a data question.

Fourth, investigate whether the engagement lift is driven by the harmful content. If users engaging more are specifically engaging with the content being reported, the engagement lift is contaminated — it's measuring harm engagement, not healthy engagement. Separate the engagement metrics for flagged vs. unflagged content.

Decision: if the harmful content rate increases, do not ship. If only the count increases due to volume and the rate is stable, the engagement lift can be weighed on its merits. In either case, escalate to policy and legal teams before making a ship decision on a feature with any harmful content signal.

---

**Q17. Testing a News Feed algorithm change when users interact with each other**

The standard challenge: users interact with each other's posts. A user in control who gets less algorithmic amplification of their posts affects what treatment users see in their feeds, and vice versa. There's bidirectional contamination.

Best design: graph cluster randomization. Use the Louvain or METIS algorithm to partition the social graph into dense communities. Randomize at the community level. In treatment communities, all users get the new algorithm. In control communities, all users get the old algorithm. Cross-community edges are sparse by construction of the partition, so contamination is minimal.

Implementation at Meta scale: the social graph has billions of nodes and edges. Community detection needs to run on a distributed graph processing system (e.g., Giraph or similar). The partition must be pre-computed and stable for the experiment duration.

Analysis: cluster-level analysis. Average the metric within each cluster (mean engagement per cluster, mean time-per-session per cluster). Run a weighted two-sample test on cluster averages, weighting by cluster size. Standard errors must be cluster-robust.

Primary metric: meaningful social interactions (MSI) — reactions, comments, shares, and messages per user per day. This is a composite that captures different types of engagement rather than optimizing a single modality.

For the network effect estimation: within treatment clusters, compare outcomes for users with high vs. low fractions of their friends also in treatment. This estimates the network amplification factor (how much does having more treated friends magnify the direct effect?).

---

**Q18. Choosing primary metric vs guardrail metrics for a Marketplace feature test**

Process for a Marketplace test:

Start from the business objective. Marketplace's goal is efficient matching of buyers and sellers — more transactions, with satisfied parties on both sides. The primary metric should measure matching efficiency or transaction success.

Primary metric candidates: Gross Merchandise Value (GMV) per active user, transaction completion rate (listing view → purchase), or buyer-seller match rate. Among these, transaction completion rate is best for a feature test because it's less noisy than GMV (which has high variance from large transactions) and directly measures the core matching function.

Why not GMV as primary: it has high variance due to outliers (a single large purchase in the treatment arm can swing results). Use it as a secondary metric with winsorization.

Guardrail metrics: any metric that must not be harmed regardless of primary metric improvement.

Essential guardrails: seller satisfaction rate (are sellers getting fair transactions?), fraudulent transaction rate (does the feature create gaming opportunities?), buyer complaint rate (post-transaction dissatisfaction), and listing quality score (does the feature encourage low-quality listings?).

The process: in the experiment protocol document, define the primary metric, guardrail thresholds, and the decision rule upfront. The rule should state that a positive primary metric result only leads to a ship decision if all guardrails are within their pre-specified tolerances. This prevents post-hoc rationalization of guardrail failures.

---

**Q19. Engineer claims a feature shipped 3 weeks ago is causing a long-term retention decline — how do you investigate?**

This is a causal attribution problem with observational post-ship data. The experiment ended when the feature shipped, so you don't have a clean control group.

Step 1: characterize the retention signal. Is the decline statistically significant? Plot D7, D14, D30 retention for weekly cohorts over the past 3 months. Is the decline specific to cohorts that joined after the feature shipped, or does it affect older cohorts as well? If only post-ship cohorts show the decline, the feature is a plausible cause. If all cohorts decline simultaneously, it's more likely a platform-wide cause (algorithm change, seasonal effect, competitor move).

Step 2: check for confounds. Did anything else change 3 weeks ago? Other feature ships, a notification change, an algorithm update, external events? Build a change log and check for co-occurrences.

Step 3: look for a dose-response relationship. Among users who joined after the ship date, do those who interacted more with the feature show lower retention? A correlation between feature usage and churn (with appropriate lag) supports causality, but is not proof.

Step 4: check if there's a holdout. Was there any percentage kept on the old experience? Even a 1% holdout gives you a small but useful control group.

Step 5: if no holdout exists, consider a rollback experiment. Disable the feature for a new random 10% of users and monitor their retention relative to the remaining 90% over the next 2 weeks. This is an ad-hoc A/B test of "feature off" vs "feature on." If the 10% shows better retention, the causal hypothesis is supported.

If the causal evidence is strong, recommend a staged rollback with continuous monitoring.

---

**Q20. Calculating MDE for a log-normally distributed metric**

For a log-normally distributed metric (e.g., revenue per user, session length), the mean and variance are related in ways that affect MDE calculation.

If `X` is log-normal, then `log(X) ~ Normal(μ, σ²)`. The mean of `X` is `exp(μ + σ²/2)` and the variance of `X` is `[exp(σ²) - 1] * exp(2μ + σ²)`.

Two approaches for MDE calculation:

Approach 1 — Transform to normal, compute MDE on log scale: Apply log transformation to the metric. The transformed metric is approximately normal. Compute the standard deviation `σ_log` from historical data (take logs of raw observations, compute SD). MDE on the log scale: `δ_log = (z_α/2 + z_β) * σ_log * sqrt(2/n)`. Convert back to the original scale: a δ_log change on the log scale corresponds to a multiplicative factor of `exp(δ_log)` on the original scale. This is your MDE as a percentage change.

Approach 2 — Direct delta method on original scale: Compute mean `μ_X` and variance `σ²_X` from historical data on the original scale (with winsorization to cap outliers that inflate variance). MDE: `δ = (z_α/2 + z_β) * σ_X * sqrt(2/n)`. This gives MDE in absolute units. Express as a percentage: `MDE% = δ / μ_X`.

Practical considerations: log-normal revenue metrics often have σ_log ≈ 2–3, meaning the distribution is extremely heavy-tailed. Winsorize before computing variance (e.g., cap at the 99th percentile). Use winsorized variance in the MDE formula to get a realistic sample size estimate. Without winsorization, the required n can be impossibly large due to a small number of very high-revenue users dominating the variance.

---

## Apple

**Q21. A/B testing a change to the App Store search algorithm**

The App Store search has unique characteristics: queries are sparse for any individual user (you don't search the App Store 50 times a day), the user population skews toward intent (searching = high purchase/download intent), and the outcome is binary (install or not) with a very low per-query rate.

Design: randomize at the Apple ID level (not device, since users have multiple devices). Users in treatment get the new ranking algorithm for all their App Store searches; control gets the current algorithm.

Primary metric: search-to-install conversion rate, computed using the delta method since it's a ratio metric.

Secondary metrics: search success rate (did the user find an app and interact positively — defined as installing and not immediately deleting), search abandonment rate, refund rate for search-driven installs (a quality signal), App Store session depth.

Challenges: the install rate is very low for most queries (maybe 5–15% of searches result in an install). This means you need large samples. Use CUPED with pre-experiment search behavior as the covariate to reduce variance. Also use triggered analysis — only include users who actually conducted a search during the experiment; don't dilute with the large population of App Store visitors who browsed but didn't search.

Guardrails: developer revenue per search session (don't harm the developer ecosystem), user-reported app quality (ratings of apps found via search), and App Store crash rates for newly installed apps (search should surface quality apps).

---

**Q22. Testing a new iPhone onboarding flow when devices are tied to Apple IDs**

The randomization unit question is critical here: should you randomize by device or by Apple ID?

The answer is Apple ID, for two reasons. First, many users have multiple Apple devices; you want consistent onboarding across all of them. Second, device-level randomization creates a scenario where the same person gets different onboarding experiences on their iPhone vs. iPad, which is confusing and technically problematic.

Implementation: at account creation or first device setup (linked to Apple ID), assign the user to treatment or control. Store the assignment server-side, linked to the Apple ID. When any device belonging to that Apple ID is set up, retrieve the assignment and serve the appropriate onboarding flow.

Edge cases to handle: users who set up a device before creating an Apple ID (they see default flow, then get assigned when they create the ID — exclude these from the analysis or analyze separately). Users who already have an Apple ID but are setting up a new device (this is not a new-user onboarding scenario; segment these separately). Family sharing accounts (one Apple ID linked to multiple family members' devices — exclude or handle carefully since you'd be treating multiple people as one unit).

Primary metrics: onboarding completion rate, 7-day activation (at least N app opens or feature interactions within 7 days), and first-party app adoption (did the user set up Apple Pay, iCloud, Face ID?). These are leading indicators of long-term retention.

---

**Q23. Measuring the causal impact of a Siri improvement on user satisfaction**

Measuring causal impact of a conversational AI improvement is hard because: Siri interactions are sparse for most users, satisfaction is subjective and hard to measure directly, and users don't always know what's possible so they may not try new capabilities.

Causal design: randomize users to treatment (improved Siri) vs control (current Siri). This is a standard A/B at the device/Apple ID level. The causal impact question is answered by the randomization.

Measurement of satisfaction: this is the harder part. Options:

Implicit signals: Siri task completion rate (did the action the user requested get successfully executed?), retry rate (did the user immediately re-ask the same question, indicating failure?), follow-up question rate (did the user need to clarify, suggesting the first response was incomplete?), Siri usage growth over time in treatment (if users find Siri more useful, they'll use it more).

Explicit signals: in-Siri thumbs up/down feedback after responses. Rates are low (most users don't rate), but they're direct satisfaction signals. Periodic satisfaction surveys: show a small fraction of users an in-app survey ("How satisfied are you with Siri?") and compare treatment vs control responses.

Downstream signals: smart speaker re-engagement (did HomePod users engage more after the improvement?), third-party app integrations (did users start using Siri with more apps?).

Primary metric: task completion rate plus immediate retry rate (as a quality penalty). Combine into a single "first-attempt success rate" that captures whether Siri understood and executed the intent correctly.

---

**Q24. Deciding when to ramp from 5% to 100%**

This is a staged rollout decision framework. The 5% launch is a risk mitigation tool — you're looking for operational failures and safety issues before broad exposure.

Ramp decision criteria:

Guardrail stability window: after the 5% rollout, monitor all critical guardrails (crash rate, error rate, battery impact, network usage, customer support volume) for a minimum holding period — typically 48–72 hours for high-frequency features, or until enough users have experienced the feature for statistical confidence. If all guardrails are within pre-defined tolerance thresholds (e.g., crash rate < baseline + 0.01%), begin ramping.

Statistical significance check: at 5% allocation, you may not have sufficient power to measure the effect on user experience metrics. But you should check for directional signals. If the primary metric is moving strongly negative at 5%, pause before ramping.

Ramp schedule: don't jump from 5% to 100% in one step for high-risk features. Use intermediate steps: 5% → 20% → 50% → 100%, with a 24–48 hour holding period at each step. At each step, re-validate guardrails.

Automatic ramp vs manual: for well-instrumented features with clear automated guardrails, the ramp can be automated — the system advances the percentage if guardrails pass, pauses if they fail. For novel features or significant user-facing changes, require manual approval at each step.

100% decision: ramp to 100% when all guardrails have been stable across the entire ramp period, the feature behavior in high-traffic conditions (50–100% may hit infrastructure limits not visible at 5%) looks normal, and any A/B test intended to measure behavioral impact has been closed or flagged as complete.

---

**Q25. Testing a privacy-related feature change with limited data collection**

Apple's privacy architecture creates genuine constraints: ATT opt-in rates are low, on-device processing means some signals never leave the device, and differential privacy adds noise to aggregate metrics.

Design constraints: you may not be able to track individual users across app boundaries or measure certain downstream behaviors. Focus on metrics that Apple can measure directly: App Store engagement (if the feature relates to an App Store feature), first-party app usage, device-level usage patterns that are aggregated with differential privacy.

Statistical approaches under limited data:

Aggregate statistics with differential privacy: if you can compute aggregate statistics with DP noise, run the experiment at sufficient scale that the signal-to-noise ratio exceeds the DP noise floor. This requires large sample sizes.

On-device computation: compute the metric locally on the device and send only an aggregate summary (mean, count) without raw events. The summary is the experiment observation. Apple's Private Set Intersection and other cryptographic tools can enable this.

Cohort-based analysis: instead of individual-level data, analyze cohorts defined by version update time or random assignment encoded in the device OS, with aggregate outcomes computed server-side from anonymized telemetry.

Survey-based measurement: for subjective outcomes (satisfaction, privacy perception), use in-app surveys with a small sample. The sample size is small but the signal is direct. Ensure survey timing doesn't correlate with treatment assignment.

Pre-specified primary metric: with limited data, you have limited power. Pre-specify one metric (the most sensitive one you can measure cleanly) and accept lower power. Don't run many metrics and use any that cross significance.

---

**Q26. iCloud feature positive in testing, but ops says server costs up 20% — how do you decide?**

This is a net present value calculation with uncertainty.

Step 1: quantify the revenue benefit of the primary metric lift. How does the improvement in the measured metric translate to revenue? If the primary metric is iCloud subscription retention and the experiment shows a 2% improvement in annual renewal rate, estimate the revenue impact: 2% × (current iCloud subscribers) × (average subscription revenue). Include an LTV multiplier for the multi-year retention impact.

Step 2: quantify the server cost impact. A 20% increase in iCloud server costs is a hard number, but it scales with usage. Is the 20% estimate based on the test population's usage? Will it generalize to full rollout? Get the cost estimate from ops for the full user base at 100% rollout. Compute the annual cost increase.

Step 3: compute net value. Net value = (revenue benefit) - (cost increase). If positive, the feature is net beneficial.

Step 4: consider strategic value. Is the iCloud improvement a differentiating feature that prevents churn to Google One or Dropbox? Competitive moat has strategic value beyond direct revenue. Include a qualitative factor.

Step 5: cost optimization options. Can the server cost be reduced through engineering work (compression, caching, CDN optimization) without degrading the user experience? If a 3-month engineering sprint could halve the cost, the decision changes. Recommend a phased approach: ship the feature, simultaneously invest in cost optimization.

Step 6: risk. Is the 20% cost estimate itself uncertain? Get a confidence interval on the ops estimate. If the uncertainty is high, include a cost sensitivity analysis.

Recommendation format: "The expected revenue benefit is $X ± Y. The estimated server cost is $A ± B. Net value is $X-A with these ranges. We recommend shipping if the lower bound of net value is positive [or whatever threshold applies]."

---

**Q27. Unit of randomization is device but metric is at the user level**

Mismatch between randomization unit and measurement unit requires careful handling to avoid biased estimates and incorrect standard errors.

The problem: if you randomize by device but a user has 3 devices, the same user may be in both treatment and control simultaneously (one device in each arm). This creates cross-arm contamination. When you compute user-level metrics, you're mixing treatment and control within the same user.

Solutions:

Upgrade randomization to Apple ID: randomize at the Apple ID level, which is the user-level identifier. All devices belonging to the Apple ID get the same assignment. This resolves the mismatch entirely. This is the preferred solution if technically feasible.

Device-level analysis: if Apple ID-level randomization isn't possible (e.g., the feature is pre-authentication), analyze at the device level instead. Accept that device-level metrics don't perfectly proxy user-level metrics, and note this as a limitation.

Clustering adjustment: if you must randomize at the device level and measure at the user level, cluster your standard errors at the user (Apple ID) level. Regression: `user_metric ~ treatment + controls`, with SEs clustered by Apple ID. This accounts for the within-user correlation between devices.

Exclusion approach: exclude users who have devices in both arms (detected via Apple ID linkage if available). Analyze only "clean" users who have all devices in the same arm. This reduces sample size but eliminates the contamination.

The cleanest solution for Apple's context is Apple ID-level randomization, since Apple controls the identity infrastructure.

---

**Q28. Measuring impact of a Maps feature on driving safety — metrics and ethical constraints**

Driving safety is an ethically distinct domain. The core tension: any metric that measures driving safety requires some notion of unsafe events, which could mean tracking events where users are in danger.

Metrics (from most to least direct):

Direct safety metrics: hard braking events (detected by accelerometer), rapid lane changes, high-speed navigation errors (missed turns taken at speed). These are already collected by CarPlay and some navigation apps. Compare event rates per navigation session between treatment and control.

Proxy safety metrics: recalculation rate (did the user deviate from the suggested route, requiring recalculation — may indicate distraction or confusion), time-on-route variance (unexpected delays may indicate accidents or near-misses), manual interaction with the phone while driving (detected by touch events during movement at speed).

Downstream safety signals: insurance claim rates, accident reports in the geographic areas where the feature is active (matched to local government data, with appropriate privacy protections). These are very lagged but directly measure outcomes.

Ethical constraints:

Do not collect granular location data beyond what's necessary for navigation. Do not track specific driving behaviors in a way that could be used adversarially (e.g., insurers obtaining the data). Aggregate safety metrics before transmission — compute event counts on-device and send only summaries. Obtain explicit consent for any enhanced safety monitoring beyond standard navigation. Ensure the experiment design cannot cause harm: if a feature is suspected to reduce safety, deploy to a very small sample with intensive monitoring and an immediate kill switch.

Avoid using accident or injury data as an experiment metric — this would require experiencing adverse events to measure the treatment. Use leading indicators instead.

---

**Q29. A/B testing a payment flow change with very low event rates**

Payment flow on the App Store has extremely low per-user conversion rates (most users don't make a purchase in any given week). This is the hardest power challenge in A/B testing.

Strategies:

Triggered analysis: only include users who initiated the payment flow (e.g., reached the checkout screen). Don't dilute with the large population who visited the App Store but never started a purchase. This can 10–50x improve effective event rates.

Proxy metrics: use higher-frequency upstream metrics as primary: "add to cart" or "tap buy" rate, payment form fill rate, number of payment attempts. These occur more frequently than completed purchases. Validate that they're causally predictive of actual purchases.

CUPED: use each user's purchase history from the pre-experiment period as a covariate. Purchase behavior is sticky — users who bought recently are more likely to buy again. CUPED on historical purchase behavior can reduce variance by 40–60% for purchase metrics.

Bayesian design: use informative priors from historical conversion rates. This allows drawing conclusions faster than frequentist designs when the effect is in the expected direction.

Extended duration: accept that the experiment will run longer. Calculate required duration honestly: at a 0.5% weekly purchase rate, a 10% relative lift (to 0.55%), with 80% power, you need roughly 2M users per arm. If you have 20M users in the experiment pool, that's a 2-week experiment minimum.

Stratified analysis: stratify by user tier (high-value purchasers, occasional purchasers, never-purchased). Focus primary analysis on users with at least one prior purchase — they're most likely to convert and have higher baseline rates, improving power.

---

**Q30. Detecting if the Hawthorne effect is influencing Apple Watch activity results**

The Hawthorne effect is when users change their behavior because they know they're being observed or studied, not because of the feature itself. In Apple Watch activity studies, this is particularly salient: users in a study may increase activity simply because they're aware their activity is being measured more carefully.

Detection methods:

Temporal decay pattern: if the effect is Hawthorne-driven, it should decay over time as users forget they're in a study and revert to normal behavior. Plot the treatment effect week by week. A large initial effect that decays to near zero is consistent with Hawthorne. A stable effect that persists over 8–12 weeks is more likely a genuine feature effect.

Comparison with passive data: compare the experiment sample's pre-experiment activity levels to a matched cohort of non-experiment users. If experiment participants were already more active before the experiment started (selection into the study correlates with activity motivation), that's a baseline confound, not Hawthorne.

Behavioral discontinuity at awareness: if you can measure when users "discovered" they were in a study (e.g., by looking at the research app or health study notification), compare behavior before and after that discovery. If activity spikes at discovery and decays afterward, Hawthorne is occurring.

Active vs passive measurement: design one arm to receive only passive background monitoring (no in-app study reminders, no progress notifications) and another arm to receive active reminders and progress tracking. The difference in behavior between these arms, holding the feature constant, estimates the Hawthorne effect magnitude.

Blind-to-assignment design: if technically feasible, ensure users don't know their assignment. For a feature being A/B tested in a production app without informed consent for the specific experiment, Hawthorne is less of a concern (users don't know they're in a study). For research studies with IRB consent, blind designs or delayed disclosure of the specific hypothesis being tested can reduce Hawthorne.
---

## PART B — 100 Medium-Hard Practical A/B Testing Q&A (SL6 Level)

---

### SECTION 1: Experimental Design (Q1–20)

---

**Q1. How do you choose the unit of randomization — user, session, or page?**

**A:** The unit of randomization should match the unit at which the treatment has its effect and the level at which interference between units is minimized.

- **User-level** randomization is standard when the feature persists across sessions (e.g., a new recommendation algorithm) or when you fear spillover — user A's experience affecting user B's.
- **Session-level** is acceptable when the treatment is isolated to a single session and there's no user-state carryover (e.g., testing a search result layout within a session). But it inflates sample size because within-user correlation is ignored.
- **Page/request-level** is only valid for truly stateless, no-carryover treatments (e.g., testing an API response format), and variance estimates must be cluster-corrected.

The risk of choosing too granular a unit: **SUTVA violations** (Stable Unit Treatment Value Assumption) — treatment of one unit bleeds into another. The cost of being too coarse: smaller effective sample size and slower experiments.

Rule of thumb: randomize at the coarsest level where you can plausibly claim independence.

---

**Q2. What is SUTVA and when is it violated in practice?**

**A:** SUTVA — Stable Unit Treatment Value Assumption — states that (1) the treatment of one unit does not affect another unit's outcome, and (2) there is only one version of the treatment.

**Violations in practice:**
- **Social platforms:** showing user A a viral post (treatment) can cause A to share it, which affects B (control). Classic interference.
- **Two-sided marketplaces:** treating buyers on Marketplace affects seller behavior, which feeds back to all buyers.
- **Email/notification features:** if you send more emails to treatment users and they reply, that affects control users who receive those replies.
- **SEO tests:** changing ranking for treatment users can affect what gets indexed, affecting control users downstream.

**Mitigations:**
- Network/cluster randomization (randomize by social graph cluster)
- Geographic randomization (holdout by region)
- Time-based switchback designs
- Ego-network analysis to measure spillover

At SL6, you should be able to quantify the spillover using difference-in-differences on cluster-level data and estimate the true ATE under interference.

---

**Q3. How do you calculate the required sample size for an experiment?**

**A:** Sample size for a two-sample z-test on proportions:

```
n = 2 * (z_α/2 + z_β)² * p(1-p) / δ²
```

Where:
- `z_α/2` = critical value for Type I error (1.96 for α=0.05, two-tailed)
- `z_β` = critical value for power (0.84 for 80% power, 1.28 for 90%)
- `p` = baseline conversion rate
- `δ` = minimum detectable effect (MDE) — the smallest lift you care about detecting

**For continuous metrics** (e.g., revenue per user):
```
n = 2 * (z_α/2 + z_β)² * σ² / δ²
```

**Key SL6 considerations:**
- Use the actual metric variance from historical data, not a guess
- If the metric is log-normal (revenue, session length), apply variance-stabilizing transforms or use the delta method
- Account for **multiple testing**: if you have k primary metrics, apply Bonferroni or Benjamini-Hochberg
- For ratio metrics (e.g., CTR = clicks/impressions), use the **delta method** to estimate variance
- If running multiple variants, adjust: `n_per_variant = n_two_arm * correction_factor`

Always pilot: run the power calculation then cross-check with historical split variance.

---

**Q4. What is the difference between frequentist and Bayesian A/B testing? When would you prefer each?**

**A:**

**Frequentist:**
- Define α (Type I error rate) and power upfront
- Run the test to a fixed sample size or duration
- Reject H₀ if p < α
- Guarantees error rate control if you don't peek
- Interpretability: "If the null were true, we'd see this data <5% of the time"

**Bayesian:**
- Place a prior on the effect size
- Update beliefs with data to get a posterior distribution
- Decision is based on probability the treatment is better, or Expected Loss
- Naturally handles early stopping and continuous monitoring
- Output: "There's a 94% probability variant B beats control"

**When to prefer Bayesian:**
- You need to peek at results frequently (e.g., you're running a business and can't wait)
- You have strong prior knowledge (e.g., from a previous experiment)
- You care about magnitude of effect, not just significance
- Small samples where frequentist power is insufficient

**When to prefer frequentist:**
- Regulatory contexts requiring strict error rate control
- Large-scale platform experimentation with standardized tooling
- When stakeholders are trained to interpret p-values

**SL6 nuance:** Most large tech companies use frequentist for consistency and auditability, but use Bayesian for feature flags, multi-armed bandits, and early-stopping decisions. Knowing both and their failure modes is table stakes.

---

**Q5. How do you handle multiple testing in an A/B experiment with 15 metrics?**

**A:** Multiple testing inflates the family-wise Type I error rate (FWER). With 15 independent metrics at α=0.05, the probability of at least one false positive is `1 - (0.95)^15 ≈ 54%`.

**Approaches:**

1. **Define a primary metric upfront** — only one metric drives the go/no-go decision. This is the most disciplined approach. Secondary metrics are exploratory and require replication.

2. **Bonferroni correction** — divide α by the number of tests: `α_adjusted = 0.05/15 = 0.0033`. Conservative, controls FWER.

3. **Benjamini-Hochberg (BH)** — controls False Discovery Rate (FDR), not FWER. More powerful than Bonferroni. Appropriate when you expect some metrics to truly move. Rank p-values, reject those where `p(i) ≤ (i/m) * α`.

4. **Hierarchical testing** — pre-specify a hierarchy: test primary first; if significant, test secondary 1; if significant, test secondary 2, etc. No correction needed if done correctly.

5. **Group sequential methods** — for metrics evaluated at multiple time points.

**SL6 answer pattern:** In practice at top tech companies, you declare one primary metric in the experiment protocol before launch, treat secondary metrics as directional, and use BH for any secondary analysis. This preserves power while limiting false discoveries.

---

**Q6. What is a holdout group and when do you use it vs a traditional A/B test?**

**A:** A **holdout group** is a long-running, persistent control group that never receives a specific feature or class of features, used to measure cumulative and long-term effects.

**When to use:**
- Measuring **long-term ROI** of a launched feature (e.g., does the new recommendation system increase retention 6 months later?)
- Measuring **cumulative impact** when many features launch in sequence (a/b tests end, but holdouts persist)
- Detecting **cannibalization** between features
- Estimating **total value of the experimentation program** (holdout from all experiments combined)

**How it differs from a traditional A/B test:**
- A/B tests run for weeks, then ship or kill. Holdouts run for months to years.
- A/B tests measure short-term lift on a specific metric. Holdouts measure ecosystem-level and longitudinal effects.
- Holdouts require a stable, uncontaminated control group — you must ensure holdout users don't get the treatment through other paths (e.g., a friend shares a feature, the app updates over the air).

**Risks:**
- Holdout groups shrink over time as users churn or change devices
- Maintaining non-contamination is operationally hard
- Long holdouts can be ethically questionable if the treatment is net positive

At SL6, you're expected to design a holdout framework that handles contamination and attrition and can report on it quarterly.

---

**Q7. How do you design an A/B test for a feature with very low baseline conversion (e.g., 0.1%)?**

**A:** Low conversion rates make experiments expensive — required sample sizes blow up because variance in a Bernoulli metric is `p(1-p)`, which is low in absolute terms but large relative to the mean.

**Strategies:**

1. **Increase traffic allocation** — if you have the volume, allocate more users. At 0.1% baseline and MDE of 10% relative lift (to 0.11%), at 80% power you need ~14M users per arm.

2. **Use a proxy metric** — find an upstream metric that predicts the eventual conversion and has higher base rate. E.g., "add to cart" instead of "purchase complete." This must be validated causally first.

3. **Log-transform or use ratio metrics** — for revenue metrics, log-transformation reduces variance and increases power.

4. **Triggered analysis** — only include users who were actually eligible for the conversion event (e.g., reached the checkout page). This sharpens the analysis by removing diluted users.

5. **Sequential / adaptive designs** — use methods that allow stopping early if effect is strong.

6. **Bayesian with informative priors** — if you have good prior beliefs about the effect size, Bayesian designs are more efficient.

7. **Reduce variance via stratification or CUPED** — pre-experiment covariates can explain much of the variance in conversion, dramatically reducing required n.

---

**Q8. What is CUPED and how does it improve experimental power?**

**A:** **CUPED (Controlled-experiment Using Pre-Experiment Data)** is a variance reduction technique that uses pre-experiment behavior to reduce noise in the experiment metric.

**Core idea:** Instead of analyzing the raw metric `Y`, analyze the residualized metric:
```
Y_cuped = Y - θ * X_pre
```
where `X_pre` is a pre-experiment covariate (e.g., metric value in the 2 weeks before the experiment) and `θ = Cov(Y, X_pre) / Var(X_pre)`.

**Why it works:** `X_pre` is uncorrelated with treatment (since it predates the experiment), so subtracting it doesn't bias the estimate. But if `X_pre` is predictive of `Y`, it removes variance, equivalent to running a regression adjustment.

**Variance reduction formula:**
```
Var(Y_cuped) = Var(Y) * (1 - ρ²)
```
where `ρ` is the correlation between `Y` and `X_pre`. If ρ = 0.7, variance drops by 51%.

**In practice:** Top tech companies (Booking.com, Netflix, Airbnb) report 20–50% variance reduction using CUPED, which translates to 20–50% fewer users needed (or faster experiments).

**SL6 considerations:**
- Must verify `X_pre` is truly pre-treatment (no contamination)
- Winsorize or cap extreme values in `X_pre` to avoid leverage points
- Can extend to multiple covariates (regression adjustment)
- Works for any metric (binary, continuous, ratio)

---

**Q9. What is a switchback experiment and when would you use one?**

**A:** A **switchback (time-series crossover) experiment** alternates treatment and control assignments over time for the same units (e.g., geographic regions, driver pools).

**Design:** Divide time into alternating windows (e.g., 30-minute slots). Units are assigned treatment in odd slots and control in even slots (or randomized). Repeat.

**When to use:**
- **Marketplace/supply-demand interference:** In ride-sharing (Uber, Lyft), assigning driver pricing treatment to half the drivers creates imbalanced supply, making rider outcomes incomparable. Switchbacks alternate the whole market between treatment and control over time.
- **Operational systems:** Content delivery, server routing, where user-level randomization isn't feasible.
- **When network effects make user-level randomization invalid.**

**Statistical analysis:** Use regression with fixed effects for unit and time period to recover the ATE:
```
Y_it = α_i + β_t + δ * T_it + ε_it
```
where `T_it` is the treatment indicator.

**Challenges:**
- Carryover effects: if a treatment window effects persist into the next control window, estimates are biased. Must include washout periods.
- Autocorrelation in residuals — must cluster standard errors by unit.
- Requires careful logging of exact treatment assignment timestamps.

---

**Q10. How do you test a feature when you can only randomize at the cluster/group level?**

**A:** This is a **cluster randomized trial (CRT)**. Examples: randomizing by school (educational interventions), city (marketing campaigns), or social group (social feature tests).

**Key challenge:** Users within a cluster are correlated. If you ignore this and treat observations as independent, standard errors are too small → inflated Type I error.

**Analysis approach:**

1. **Cluster-robust standard errors:** Run OLS/logistic regression but cluster SEs at the randomization level. This is the minimum.

2. **Aggregated analysis:** Average the metric within each cluster, then run a simple two-sample test on cluster-level means. Treats the cluster as the unit of analysis — valid if clusters are equal-sized.

3. **Mixed-effects models:** Include a random effect for cluster to account for intra-cluster correlation (ICC).

4. **Design-based methods:** Use randomization inference (permutation tests) that respect the clustering.

**Power implications:** Power depends on the **design effect** (DEFF):
```
DEFF = 1 + (m - 1) * ICC
```
where `m` is average cluster size and ICC is the intra-class correlation. High ICC + large clusters = massive power loss. You need more clusters, not more users per cluster.

**SL6 tip:** At companies like Meta, this arises constantly in social experiments. The ICC for engagement metrics within social clusters can be 0.05–0.15, leading to DEFF of 5–15x. You often need 10–50x more clusters than you'd naively estimate.

---

**Q11. What are guardrail metrics and how do you set thresholds for them?**

**A:** **Guardrail metrics** are metrics you must not harm, regardless of primary metric improvements. They represent safety rails for the business (e.g., you can't ship a feature that improves CTR but crashes 1% of users).

**Examples by category:**
- **Quality guardrails:** crash rate, error rate, latency (p50, p95, p99), broken UI rate
- **Business guardrails:** revenue per user, ad load, subscription cancellation rate
- **User health guardrails:** customer support contact rate, account deletion rate, NPS

**How to set thresholds:**

1. **Relative change threshold:** e.g., guardrail fails if metric degrades >X% relative (e.g., latency increases >2%). Set based on historical variability and business judgment.

2. **Statistical approach:** Run a one-sided test. The guardrail fails if we can reject "metric is ≤ baseline" at α=0.05. This controls the rate of falsely failing a good experiment.

3. **Practical significance threshold:** Combine statistical significance with a minimum meaningful effect. Don't fail a guardrail for a 0.0001% degradation that's statistically significant at n=100M.

**Decision logic:**
- Primary metric significant AND all guardrails pass → Ship
- Primary significant BUT any guardrail fails → Investigate, likely don't ship
- Primary not significant AND guardrails pass → Don't ship (or extend experiment)

**SL6 nuance:** Guardrail thresholds should be documented in the experiment protocol before launch, not set after seeing results. Post-hoc guardrail adjustment is p-hacking.

---

**Q12. How do you handle imbalanced sample sizes between control and treatment?**

**A:** In a perfect A/B test, arms are equal in size. Imbalance happens due to:
- Traffic splitting bugs
- User dropout differential between arms
- Bot filtering applied unevenly

**Detecting imbalance:**
- Run a **SRM (Sample Ratio Mismatch) check**: given the intended split, use a chi-squared test to check if observed counts match expected counts. Even small SRMs (e.g., 50.1%/49.9% vs 50/50) can indicate a logging or randomization bug.

```
χ² = Σ (Observed - Expected)² / Expected
```

**If there's an SRM, do not analyze the experiment.** SRM almost always indicates a bug, not a natural variation, and results are untrustworthy.

**If no SRM, unequal sizes are fine statistically:**
- Variance of the difference in means is: `σ²/n_c + σ²/n_t`
- Welch's t-test handles unequal variances and sizes correctly
- Power is sub-optimal vs equal allocation but estimates are unbiased

**Optimal allocation:** If you want to maximize power with fixed total n, allocate proportionally to the standard deviation in each arm (`Neyman allocation`). For equal variance, 50/50 is optimal. If treatment has higher variance (e.g., a high-variance revenue metric), allocate more users to treatment.

---

**Q13. What is an A/A test and what should you use it for?**

**A:** An **A/A test** assigns users to two groups that receive identical experiences. Both groups see the control. It's a sanity check on your experimentation infrastructure.

**What it validates:**
1. **Randomization quality:** If randomization is truly random, the groups should be statistically indistinguishable. Any significant difference flags a bug.
2. **Variance estimation:** The observed variance of your metric in an A/A test should match your historical or theoretical estimate. If variance is wrong, power calculations are wrong.
3. **Type I error rate calibration:** Running many A/A tests and checking what fraction are significant at α=0.05 should give ~5%. If you see 15%, your p-values are miscalibrated.
4. **Logging and data pipeline integrity:** Confirms that treatment assignment logs, metric computation, and joins are working correctly.

**What it doesn't validate:**
- Whether your metrics are measuring the right thing
- Whether the experiment has sufficient power for your actual treatment

**SL6 use:** Run A/A tests as part of your experimentation platform validation. If your platform is new, run 1000 simulated A/A tests on historical data to verify Type I error rate. In production, run A/A checks any time you change your randomization hash function, logging schema, or experiment assignment system.

---

**Q14. How do you design an experiment to test a ranking algorithm change?**

**A:** Ranking algorithm changes are especially tricky because:
- **Position bias:** clicks depend on position, not just quality. Showing better results in position 1 inflates CTR even if results aren't actually better.
- **Interleaving:** users can't be split cleanly — a single query needs one ranked list.
- **Feedback loops:** clicks on treatment results may improve future rankings, creating non-stationarity.

**Design options:**

1. **User-level A/B test:** Randomize users to "old algorithm" or "new algorithm." All queries from user A use the old algo; all from user B use the new. This is clean for measuring long-term engagement (session depth, return visits) but takes longer to accumulate signal.

2. **Interleaved testing (Netflix/Google approach):** For each query, blend results from both algorithms: odd positions from algo A, even from algo B (or use a fair interleaving algorithm). Record which positions users click. The algo with more clicks on its positions wins. This is **10–100x more sensitive** than user-split A/B because each query is its own experiment. But it only measures immediate relevance, not long-term satisfaction.

3. **Counterfactual evaluation (offline):** Before online testing, replay historical clicks under the new ranking to estimate CTR change. Requires inverse propensity scoring to correct for position bias.

**Primary metric for ranking:** Often **NDCG (Normalized Discounted Cumulative Gain)**, click-through rate adjusted for position, or **session satisfaction** signals (did the user reformulate the query, click back, dwell time).

---

**Q15. What is the "peeking problem" and how do you solve it?**

**A:** The **peeking problem** (also called "optional stopping") occurs when you repeatedly check experiment results during the run and stop as soon as you see p < 0.05. This inflates Type I error dramatically.

**Why it's a problem:**
Under the null hypothesis, a random walk of cumulative p-values will cross α=0.05 with probability much greater than 5% if you check frequently. Simulations show that with daily peeking over a 4-week experiment, the true false positive rate can reach 20–30%.

**Solutions:**

1. **Pre-register the end date and don't peek.** Simple but operationally difficult in business settings.

2. **Sequential testing / Alpha-spending functions:** Use methods like O'Brien-Fleming or Pocock that pre-allocate the α budget across planned interim analyses. Each interim look uses a more stringent threshold (e.g., p < 0.001 at first look, p < 0.01 at second, p < 0.05 at final). Total Type I error remains ≤ 5%.

3. **Always-Valid Inference / Sequential probability ratio tests (SPRT):** Methods like mSPRT (mixture Sequential Probability Ratio Test) allow continuous monitoring while maintaining error control. Used by companies like Optimizely and Airbnb.

4. **Bayesian approaches:** Use Expected Loss or posterior probability as stopping criterion. Naturally supports continuous monitoring without inflating Type I error (though different assumptions apply).

**SL6 answer:** At scale, companies implement always-valid confidence intervals (e.g., using asymptotic confidence sequences) that are valid at any sample size, enabling continuous monitoring dashboards without inflating errors.

---

**Q16. How do you measure the impact of a feature when there's a novelty or primacy effect?**

**A:** 
**Novelty effect:** Users engage more with a new feature simply because it's new. Engagement declines after the novelty wears off — the initial lift is not representative of the long-term effect.

**Primacy effect (or "learning" effect):** Users initially perform worse with a new UI because they're unfamiliar with it. Over time, as they learn, the benefit emerges — the initial result underestimates the long-term effect.

**Detection:**
- Plot the treatment effect (treatment - control) over time (day-by-day). A decaying effect suggests novelty; a growing effect suggests primacy.
- Segment by "new users" (who have no habit for the old experience) vs "existing users." If only existing users show the novelty spike, it's likely novelty.

**Mitigation strategies:**

1. **Run the experiment long enough:** For most consumer features, 2–4 weeks is sufficient to see the effect stabilize.

2. **Day-of-week and time series analysis:** Decompose the effect by time-since-exposure rather than calendar time.

3. **New user analysis:** Analyze users who joined after the experiment started — they have no prior experience with the old product, so no novelty or primacy effect. Their behavior approximates the long-run steady state.

4. **Long-horizon holdout:** Maintain a holdout group for 3–6 months to measure the stabilized long-run effect directly.

---

**Q17. How do you run an experiment when you only have access to aggregate-level data (not user-level)?**

**A:** This is a common constraint in offline, partner, or privacy-restricted settings. Techniques include:

1. **Difference-in-Differences (DiD):** Use geographic or time-based variation. Compare treated regions/periods to control regions/periods, controlling for pre-existing trends.
```
DiD = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
```
Requires the **parallel trends assumption**: absent treatment, both groups would have trended similarly.

2. **Regression Discontinuity Design (RDD):** If treatment is assigned based on a threshold (e.g., users with score > 80 get the feature), compare outcomes just above and below the threshold. Local randomization assumption holds near the cutoff.

3. **Interrupted Time Series (ITS):** For a single unit (e.g., the whole product) with a treatment event at time T, model the trend before T and extrapolate to estimate counterfactual post-T. The gap is the treatment effect. Requires a long pre-period.

4. **Synthetic Control:** Construct a weighted combination of control units that matches the treated unit's pre-period trend. Use the synthetic control as the counterfactual post-treatment.

**SL6 tip:** Each method has critical identifying assumptions. Always test those assumptions explicitly (e.g., test parallel trends pre-treatment; test if assignment is truly as-good-as-random near an RD cutoff).

---

**Q18. How do you handle user attrition/dropout in an A/B test?**

**A:** Attrition occurs when users drop out of the experiment before the measurement period ends. This becomes a problem when attrition is **differential** — dropping out at different rates in treatment vs control. This is called **informative censoring** and can bias results.

**Example:** You test a new email frequency. Control users who find it annoying unsubscribe. Treatment users who find it helpful stay. If you only analyze retained users, you're comparing unhappy control users who stayed vs treatment users — a biased comparison.

**Approach:**

1. **Intent-to-Treat (ITT) Analysis:** Analyze all users based on their initial assignment, regardless of whether they completed the experiment. This is unbiased for the average treatment effect in the full population. It may dilute the measured effect but is interpretable.

2. **Per-Protocol Analysis:** Only analyze users who completed the protocol. Valid only if completion is independent of treatment (unlikely in most cases). Use with caution.

3. **Survival/Time-to-Event Analysis:** Use Kaplan-Meier curves and Cox proportional hazards models to analyze dropout itself as an outcome. Compare survival curves between arms.

4. **Bounds analysis (Manski bounds):** Compute upper and lower bounds on the treatment effect under worst-case assumptions about what dropouts would have done.

5. **Check for differential attrition:** Run a regression predicting dropout on treatment assignment. If coefficient is significant, attrition is differential and you must address it.

---

**Q19. How do you design a multivariate test (MVT) and when is it better than sequential A/B tests?**

**A:** A **multivariate test (MVT)** tests combinations of multiple factors simultaneously in a single experiment. E.g., testing 3 headline options × 2 button colors × 2 layouts = 12 combinations.

**Full factorial design:** All combinations tested simultaneously. Allows estimation of:
- Main effects: impact of each factor independently
- Interaction effects: does the headline effect depend on button color?

**Fractional factorial design:** Test a subset of combinations (orthogonal array). Estimates main effects and selected interactions with fewer cells. Used when you can't run all combinations due to traffic constraints.

**When MVT beats sequential A/B tests:**
- You believe factors interact (e.g., headline only works with the right image)
- You want to find the globally optimal combination, not just the best of each factor individually
- You have enough traffic to fill all cells (each cell needs the same sample size as a standard A/B test)

**When sequential A/B is better:**
- Factors are independent (no interactions) — then just stack A/B tests sequentially
- Traffic is limited — MVTs require much more traffic
- You need fast iteration — MVTs take longer

**Statistical analysis of MVT:** Use ANOVA or regression with dummy variables for each factor and their interactions. Test each main effect and interaction with appropriate corrections for multiple comparisons.

---

**Q20. What is network effect interference and how do you handle it in experimentation?**

**A:** **Network effect interference** occurs on platforms where a user's outcome depends on other users' treatment status. It violates SUTVA and makes standard A/B tests biased.

**Examples:**
- Social feed: treatment users see algorithmically curated content and post more → control users' feeds change because they're connected to treatment users
- Marketplace: treating sellers with new pricing tools affects buyer outcomes
- Messaging: treatment users send more messages to control users

**Types of interference:**
- **Dilution bias:** Some of the control group receives indirect treatment → understates the true effect
- **Amplification bias:** Treatment users' behavior reinforces itself through the network → overstates individual-level effect

**Solutions:**

1. **Ego-network randomization:** Randomize at the level of "ego + all alters." Each user and all their connections are in the same arm. Reduces interference but requires identifying social clusters.

2. **Graph cluster randomization:** Use community detection algorithms (Louvain, METIS) to partition the social graph into dense clusters. Randomize at the cluster level. Estimate the treatment effect using cluster-level aggregates.

3. **Two-stage experiments:** In stage 1, randomize clusters. In stage 2, within treated clusters, randomize individuals to measure direct vs spillover effects separately.

4. **Bipartite experiments:** In two-sided markets, randomize on one side (e.g., all sellers in a market to treatment), measure outcomes on both sides.

5. **Regression-based correction:** After the fact, estimate the "peer treatment dose" for each user (e.g., fraction of friends in treatment), and include it as a covariate to separate direct and indirect effects.

---

### SECTION 2: Statistical Analysis (Q21–40)

---

**Q21. What is the delta method and when do you need it?**

**A:** The **delta method** is a technique for estimating the variance of a function of random variables using a first-order Taylor approximation.

**When you need it:** Anytime your metric is a **ratio** of two random quantities (e.g., CTR = clicks/impressions, revenue per session = revenue/sessions).

**The problem:** If you naively compute CTR = total_clicks / total_impressions and treat it as a single observation, you lose the within-user variance structure. The ratio's variance is not simply `Var(clicks)/impressions²`.

**Delta method formula:**
For metric `r = X/Y` (e.g., CTR):
```
Var(r̂) ≈ (1/n) * [Var(X)/μ_Y² - 2*Cov(X,Y)*μ_X/μ_Y³ + Var(Y)*μ_X²/μ_Y⁴]
```
where `μ_X = E[X]`, `μ_Y = E[Y]`, and Cov(X,Y) is the covariance between numerator and denominator at the user level.

**Practical implementation:**
Compute user-level `x_i` and `y_i` (e.g., clicks per user, impressions per user). The linearized metric is:
```
z_i = (x_i - r̂ * y_i) / μ_Y
```
Then `Var(r̂) = Var(z_i) / n`. Run a t-test on `z_i` values.

**SL6 note:** Not using the delta method for ratio metrics is a common and consequential error. It leads to incorrect confidence intervals and hypothesis tests. Most production experimentation platforms implement it automatically.

---

**Q22. How do you handle metrics with heavy tails or outliers?**

**A:** Heavy-tailed metrics (revenue, session length, API response times) have high variance, which degrades experiment power and makes t-tests less reliable.

**Techniques:**

1. **Winsorization (capping/truncation):** Cap the metric at a high percentile (e.g., 99th or 99.9th). Extreme outliers are brought to the cap value. Reduces variance substantially. The choice of cap is a business decision — must be set before seeing results.

2. **Log transformation:** Applies when the metric is log-normally distributed (common for revenue). `log(Y + 1)` stabilizes variance. Interpret results as percentage changes in geometric mean.

3. **Quantile-based metrics:** Instead of mean, test on median or a percentile (e.g., p75). Use bootstrap to estimate variance of quantile difference.

4. **Trimmed mean:** Remove the top and bottom X% of observations before computing the mean. More robust to outliers, but excludes real users.

5. **Bootstrapping:** Non-parametric resampling to estimate the sampling distribution of the mean without normality assumptions.

6. **Mann-Whitney U test:** Non-parametric test of stochastic dominance — doesn't assume normality. Lower power than t-test if normality holds, but robust otherwise.

**SL6 preference:** Winsorization is the most widely used in industry because it's easy to explain, deterministic, and doesn't change the metric's interpretation dramatically. Always apply the same cap in control and treatment.

---

**Q23. Explain the difference between statistical significance and practical significance.**

**A:** 
**Statistical significance** (p-value < α) tells you only that the observed effect is unlikely under the null hypothesis of zero effect. With a large enough sample, even a 0.0001% change will be statistically significant.

**Practical significance** (effect size vs minimum meaningful effect) tells you whether the effect is large enough to matter for the business. A 0.01% lift in CTR is statistically significant at n=10 billion but completely meaningless in practice.

**How to assess both:**

1. **Define MDE (Minimum Detectable Effect) upfront** — what's the smallest effect that would change the product decision? This is a business judgment call. Use it to size the experiment.

2. **Report effect size with confidence intervals**, not just p-values. A result of "+0.5% ± 0.2% at 95% CI" is far more informative than "p=0.03."

3. **Confidence interval interpretation:**
   - If CI excludes zero AND lower bound > MDE: statistically AND practically significant → ship
   - If CI excludes zero but CI is entirely below MDE: statistically significant but not practically → don't ship
   - If CI includes zero but lower bound > MDE: not significant but effect could be real → extend experiment
   - If CI is wide and includes zero: underpowered → need more data

**SL6 framing:** When briefing leadership, lead with practical significance: "We're 95% confident the feature increases revenue by $1.2M–$3.8M per year." The p-value is a footnote for the statisticians.

---

**Q24. What is regression to the mean and how can it confound A/B results?**

**A:** **Regression to the mean** is the phenomenon where extreme measurements on a first observation tend to be less extreme on a second observation, purely due to random noise — not any real change.

**How it confounds A/B tests:**

The most common scenario: you select users for an experiment based on recent behavior (e.g., "users who churned in the last 30 days" for a re-engagement campaign). These users had unusually low engagement *by definition*. Even without any treatment, their next-period engagement will tend to be higher just because of mean reversion.

**Example:** Select 10K users with zero logins last week. Show them a re-engagement email (treatment) vs nothing (control). Both groups will likely show more logins next week — not because of the email, but because having zero logins was an extreme event that regresses to the mean.

**How to avoid it:**
1. **Random assignment:** The only solution is randomizing after selection, so both arms experience the same regression to the mean. The difference between arms is still the causal effect.
2. **Avoid selecting on the outcome:** Don't select users based on the metric you're measuring. If you must, use a pre-period metric that's different from the outcome metric.
3. **Account for pre-period values:** Use CUPED or regression adjustment with the pre-experiment metric as a covariate. This explicitly controls for where users started.

---

**Q25. How do you compute and interpret a confidence interval for a proportion?**

**A:** For a binary metric (e.g., conversion rate), the difference in proportions `δ̂ = p̂_T - p̂_C` has an approximate sampling distribution:

**Point estimate:** `δ̂ = p̂_T - p̂_C`

**Standard error of the difference:**
```
SE = sqrt(p̂_C(1-p̂_C)/n_C + p̂_T(1-p̂_T)/n_T)
```

**95% Confidence interval:**
```
CI = δ̂ ± 1.96 * SE
```

**Interpretation:** "We are 95% confident that the true difference in conversion rates lies between [lower bound, upper bound]." This means: if we repeated this experiment many times under identical conditions, 95% of the CIs computed this way would contain the true effect.

**Common SL6 pitfalls:**
- **Misinterpretation:** "There's a 95% probability the true effect is in this interval" is technically wrong (frequentist CIs are not probability statements about the parameter). The correct interpretation is about the procedure.
- **Wilson interval for small n:** When counts are small (p close to 0 or 1), the normal approximation breaks down. Use Wilson score interval or Clopper-Pearson.
- **For ratio metrics:** Use delta method (see Q21).
- **For skewed continuous metrics:** Use bootstrap confidence intervals.

---

**Q26. How do you handle seasonality when running an A/B test?**

**A:** Seasonality creates non-stationarity in your metric, which can confound results if the treatment and control periods don't overlap in the same seasonal pattern.

**Risks:**
- Running an experiment only on weekdays gives results that don't generalize to the full week
- Starting an experiment during a holiday sale period inflates all metrics
- Post-holiday dip can make a good treatment look bad

**Best practices:**

1. **Run for full weeks (multiples of 7 days):** Day-of-week effects are strong in most consumer products. A 7-day experiment that starts Tuesday will have slightly different day-mix than one starting Monday. Use 7, 14, or 21-day experiments.

2. **Avoid holiday periods:** Don't start or end experiments within 2 weeks of major holidays (Black Friday, Christmas, Diwali, etc.) unless you specifically want to test in that context.

3. **Run simultaneous treatment and control:** In a proper A/B test, both arms run concurrently, so they experience the same seasonality. This is the main protection. Don't use "before vs after" comparisons as a substitute.

4. **Seasonal decomposition for power estimation:** When estimating variance for sample size calculations, use the same period last year to capture seasonal variance. Don't use an off-peak period to estimate on-peak variance.

5. **Include time as a covariate:** In regression analysis, include day-of-week and week-of-year indicators to absorb seasonal variance.

---

**Q27. What is a Type I vs Type II error, and how do you trade them off?**

**A:**
- **Type I error (α, false positive):** Rejecting H₀ when it's true — concluding there's an effect when there isn't. Probability = α (e.g., 5%).
- **Type II error (β, false negative):** Failing to reject H₀ when it's false — missing a real effect. Probability = β (e.g., 20% if power = 80%).

**The trade-off:**
- Decreasing α (being more stringent) → increases β (lower power), unless you increase sample size
- Increasing power (decreasing β) → requires larger sample size for fixed α

**Business framing of the trade-off:**

| Error | Cost |
|-------|------|
| Type I | Ship a feature that doesn't work → wasted engineering, potential harm |
| Type II | Don't ship a feature that does work → missed revenue, competitive disadvantage |

**How to choose α and power:**
- **High-risk changes** (core algorithm, payments, medical): use α=0.01 and power=90%
- **Low-risk UI changes**: α=0.05 and power=80% is standard
- **Exploratory feature discovery**: α=0.10 and power=70% may be acceptable if cost of misses is high

**SL6 nuance:** α and power aren't symmetrically important. In most product experiments, the cost of shipping a harmful feature (Type I error for negative metrics) is higher than missing a benefit. Set guardrail metrics to very low α (0.01) and primary metric to α=0.05.

---

**Q28. How do you test for heterogeneous treatment effects (HTEs)?**

**A:** **HTEs** (also called **subgroup effects** or **effect modification**) occur when the treatment effect differs across user segments (e.g., the feature works great on iOS but not Android).

**Why it matters:** Average treatment effects can mask important heterogeneity. A null overall effect could hide a +10% effect for new users and a -10% effect for power users.

**Testing for HTEs:**

1. **Pre-specified subgroup analysis:** Before the experiment, define subgroups of interest (e.g., device type, user tenure, geography). Test the treatment effect within each subgroup. Control for multiple comparisons.

2. **Interaction tests:** In a regression model, include the interaction term `Treatment × Subgroup`. A significant interaction means HTEs exist.
```
Y = β₀ + β₁T + β₂X + β₃(T×X) + ε
```
The interaction coefficient β₃ is the HTE estimate.

3. **Causal forest / ML-based methods:** Use algorithms like Generalized Random Forests (Wager & Athey, 2018) to estimate heterogeneous effects across continuous and high-dimensional covariates. Outputs a CATE (Conditional Average Treatment Effect) for each user.

4. **SHAP values for effect attribution:** After estimating CATEs, use SHAP to understand which features drive the heterogeneity.

**Important:** Don't go on fishing expeditions through 50 subgroups after seeing the results. Pre-register your subgroup analyses or apply strict FDR control. Post-hoc subgroup findings require replication.

---

**Q29. What is the bootstrap and when should you use it in experiments?**

**A:** The **bootstrap** is a non-parametric resampling method for estimating the sampling distribution of any statistic. You resample with replacement from your observed data B times (typically B=1000–10000), compute the statistic each time, and use the distribution of bootstrapped statistics to estimate confidence intervals and standard errors.

**When to use it in experiments:**
1. **Non-normal metrics:** When the metric distribution is heavily skewed and the t-test's normality assumption is questionable (especially for small n)
2. **Complex statistics:** When the statistic is a complex function of the data (e.g., the difference in 90th percentiles, or a ratio of ratios)
3. **Ratio metrics:** As an alternative to the delta method
4. **Quantile effects:** "Did the treatment reduce the p95 latency?" — bootstrap the p95 difference

**Bootstrap CI types:**
- **Percentile bootstrap:** Use the 2.5th and 97.5th percentiles of bootstrapped statistics
- **BCa bootstrap (bias-corrected and accelerated):** Corrects for skewness in the bootstrap distribution. More accurate for small samples and non-symmetric distributions
- **Studentized/pivot bootstrap:** Most accurate but computationally expensive

**Practical considerations:**
- Requires large-memory computation at scale (10M users × 1000 bootstraps)
- For large n, the t-test is usually sufficient by CLT — bootstrap is most valuable for small/medium n or complex statistics
- Cluster your bootstrap at the randomization unit (user-level) if you have multiple observations per user

---

**Q30. How do you compute the p-value for a two-sample test on a metric with many zeros (zero-inflated distribution)?**

**A:** Zero-inflated metrics are common: # of purchases (most users buy nothing), # of posts, customer support contacts. The distribution has a point mass at 0 and a heavy right tail.

**The challenge:** The standard t-test performs poorly because the distribution is far from normal. For small n, CLT convergence is slow.

**Approaches:**

1. **Two-part (hurdle) model:** Decompose the test into two parts:
   - Part 1: Is the probability of a non-zero outcome different? → test `P(Y>0)` with a chi-squared or proportions test
   - Part 2: Among non-zero users, is the mean different? → test the conditional mean
   - Combine: overall treatment effect = P(Y>0|T=1) * E[Y|Y>0, T=1] - P(Y>0|T=0) * E[Y|Y>0, T=0]

2. **Permutation test:** Shuffle treatment labels and recompute the mean difference many times. The empirical distribution of permuted differences is the null. No distributional assumptions.

3. **Winsorize + t-test:** Cap heavy upper tail (e.g., at 99.9th percentile) and run a standard t-test. The inflated zeros no longer dominate the variance structure.

4. **Bootstrap:** Resample user-level observations (preserving the zero structure) and compute the mean difference distribution.

5. **Mann-Whitney U test:** Non-parametric test of whether treatment distribution stochastically dominates control. Robust to zeros and skew. Interprets as P(Y_T > Y_C).

---

### SECTION 3: Metrics and Decision-Making (Q41–60)

---

**Q41. How do you define and select a primary metric for an A/B test?**

**A:** A good primary metric should satisfy several properties:

1. **Directionally aligned with the business goal:** It measures what you actually care about (e.g., long-term retention, not just short-term engagement)
2. **Sensitive enough:** It moves meaningfully within the experiment window with realistic sample sizes
3. **Trustworthy and measurable:** You can compute it accurately, without ambiguity
4. **Resistant to manipulation:** The feature team shouldn't be able to "game" it easily

**Selection process:**

1. Start from the **company/product OKR** — what does success look like at the highest level? (e.g., DAU, revenue, retention D30)
2. Identify **leading indicators** that are measurable in the experiment timeframe and causally predictive of the OKR metric
3. Check **historical sensitivity:** Does this metric move in past experiments where you knew the ground truth? If it never moves, it's a vanity metric.
4. Run a **power analysis** for your candidate metric. If you'd need 50M users for 80% power, it's too noisy.

**SL6 anti-patterns:**
- Using too many "primary" metrics (turns into multiple testing chaos)
- Using activity metrics (logins, clicks) as primary when the actual goal is satisfaction or revenue — these are proxies and may not be causally aligned
- Using ratio metrics without computing power correctly (delta method)

---

**Q42. What is a surrogate metric and what are the risks of relying on one?**

**A:** A **surrogate metric** (or proxy metric) is a short-term, easily measurable metric used as a stand-in for a long-term, hard-to-measure goal.

**Examples:**
- CTR as a surrogate for user satisfaction
- # of sessions as a surrogate for D30 retention
- Likes as a surrogate for meaningful social connection

**Why surrogates are necessary:** You can't wait 6 months to measure retention in every experiment. You need a fast-moving signal that predicts long-run outcomes.

**Risks:**

1. **Surrogacy failure:** The proxy may move in the direction you want while the true outcome moves in the opposite direction. A famous example: email open rates went up when subject lines were made more clickbait-y, but actual revenue and retention declined.

2. **Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure." Teams optimize the proxy, decoupling it from the underlying goal.

3. **Simpson's paradox / confounding:** The surrogate and true metric can be confounded by user segments such that improving the surrogate doesn't imply improving the true metric in any segment.

**Validation of surrogates:**
- Run historical experiments where you know the long-term outcome, and check whether the surrogate's short-term movement correctly predicted the long-term direction
- Use causal validation: estimate the fraction of the treatment effect on the true metric "explained by" the surrogate (proportion of treatment effect mediated)

---

**Q43. How do you handle a situation where the primary metric is positive but two guardrail metrics are negative?**

**A:** This is one of the most common and important decisions in applied experimentation. The framing matters enormously.

**Step 1: Characterize the magnitude and significance of each metric.**
- Is the primary lift statistically significant? What's the effect size?
- Are the guardrail violations statistically significant? Are they practically meaningful?
- What's the direction: are guardrails slightly negative (within noise) or strongly negative?

**Step 2: Investigate the mechanism.**
- Are the guardrail movements plausibly causal? Could they be spurious (false positives from multiple testing)?
- Is there a user segment driving both effects? (e.g., new users love the feature but it harms power users)
- Is the guardrail metric measuring the right thing? (e.g., "crash rate" went up because the feature is more widely used, but crash rate per session is fine)

**Step 3: Calculate the net value.**
Convert all metrics to a common currency (e.g., expected revenue impact) and aggregate. If the primary metric lift = +$5M/yr and the guardrail violations = -$2M/yr, net value is positive.

**Step 4: Check reversibility.**
If the guardrail violation is in a critical user health metric (e.g., account deletion rate, safety reports), the bar for shipping is much higher. Some harms are not reversible.

**Step 5: Decision options:**
- Ship: if net value is clearly positive and guardrail harms are minor
- Ship with mitigation: modify the feature to reduce the harm before shipping
- Don't ship: if net value is negative or harm is in a critical guardrail
- Extend experiment: if effects are marginal and more data would resolve ambiguity

---

**Q44. How do you measure engagement vs satisfaction, and why do they diverge?**

**A:** **Engagement metrics** (clicks, sessions, time on site, logins) measure how much users interact with a product. **Satisfaction metrics** (NPS, CSAT, ratings, "did you find what you were looking for?") measure how much users value those interactions.

**Why they diverge:**
- A user watching 3 hours of autoplay videos is "highly engaged" but may feel bad about it afterward (the YouTube recommendation problem)
- A user who finds what they need in 30 seconds and leaves has low engagement but high satisfaction
- Dark patterns (infinite scroll, fear of missing out) increase engagement while decreasing wellbeing
- Outrage-inducing content is highly engaging but harmful to users and society

**What this means for A/B testing:**
- Optimizing purely on engagement can be self-defeating long-term: users burn out, delete the app, or develop negative associations with the brand
- Companies like Netflix and Spotify have moved toward satisfaction signals: "Did you enjoy this?" thumbs up/down rather than raw playtime
- Instagram/Facebook have added "time well spent" metrics and wellbeing dashboards

**SL6 approach:** Define a portfolio of metrics:
- Short-term engagement (sensitive, fast-moving, used for quick iteration)
- Satisfaction/quality signals (slower, but more aligned with retention)
- Long-term retention (ultimate truth, measured via holdouts)

Weight them according to the product's core value proposition. A meditation app should weight wellbeing metrics heavily; a news app might weight comprehension over raw time.

---

**Q45. What is the "North Star Metric" and how does it relate to A/B testing?**

**A:** The **North Star Metric (NSM)** is a single metric that best captures the core value a company delivers to customers and predicts long-term business success. It's meant to align all teams around one shared goal.

**Examples:**
- Airbnb: Nights booked
- Spotify: Time spent listening
- Facebook (historically): Daily Active Users
- Slack: Messages sent within a team
- Netflix: Retention (members watching 60+ hours/month)

**Relationship to A/B testing:**

1. **Primary metric hierarchy:** The NSM is often too slow-moving to be a direct experiment primary metric (e.g., "nights booked" may take months to stabilize). Teams use fast-moving **input metrics** (booking intent rate, search-to-view rate) as primary metrics in experiments, validated to be leading indicators of the NSM.

2. **Guardrail role:** Even if the NSM moves too slowly to be a primary metric, include it as a guardrail or secondary metric. You don't want to optimize input metrics at the expense of the NSM.

3. **Counter-metrics:** Always pair the NSM with counter-metrics to prevent optimization at users' expense (e.g., pair "time in app" with "self-reported satisfaction" to avoid dark patterns).

**SL6 nuance:** A single NSM can create perverse incentives. Netflix discovered that optimizing raw retention led to engagement with mediocre content. They shifted toward quality-adjusted engagement. Be critical of any single-metric framing.

---

**Q46. How do you attribute causality when you can't randomize?**

**A:** When randomization is impossible (ethical, technical, or business constraints), you must use **quasi-experimental methods** to estimate causal effects from observational data.

**Main methods:**

1. **Difference-in-Differences (DiD):** Compare the change in outcome over time for a treated group vs an untreated group. Requires parallel trends (both groups would have trended similarly absent treatment). Strong for policy changes affecting some regions/cohorts but not others.

2. **Regression Discontinuity (RDD):** Exploit a sharp threshold in assignment (e.g., users with score ≥ 80 get the feature). Near the cutoff, assignment is as good as random. Local causal effect is estimated at the threshold.

3. **Instrumental Variables (IV):** Find a variable Z (instrument) that affects treatment T but only affects outcome Y through T. Use Z as an instrument to estimate the causal effect. Example: random assignment to a "you might like" prompt (Z) affects feature adoption (T), which affects engagement (Y). The IV estimate is the Local Average Treatment Effect (LATE) for compliers.

4. **Synthetic Control:** Build a weighted average of control units that matches the treated unit's pre-treatment trends. Post-treatment gap is the causal effect. Used in country/region-level studies.

5. **Propensity Score Matching/Weighting:** Match treated and control units on pre-treatment characteristics (confounders). Estimates ATE under "no unmeasured confounding" assumption.

**SL6 emphasis:** Every quasi-experimental method requires a critical identifying assumption. You must explicitly test that assumption (e.g., test parallel trends in pre-period, test exclusion restriction for IV, test covariate balance for PSM). Causal identification is never guaranteed from observational data.

---

**Q47. How do you measure the long-term retention impact of a short-term A/B test?**

**A:** Short-term experiments (1–4 weeks) often can't directly measure long-term retention (D90, D365). Several approaches bridge this gap:

1. **Long-run holdout:** Keep a holdout group in control permanently after the feature ships. Compare the holdout to the shipped population 3, 6, 12 months later. Gold standard but expensive and operationally complex.

2. **Surrogate metric validation:** Identify short-term metrics that predict long-run retention (e.g., if 7-day re-engagement predicts D90 with high accuracy, use it). Validate this causal chain using historical data from past experiments where you know D90 outcomes.

3. **Cohort analysis:** After shipping, compare retention curves of cohorts who used the new feature vs those who didn't (accounting for selection bias, ideally using propensity scoring).

4. **Causal mediation analysis:** Measure the short-term change in engagement and estimate how much of that change translates to long-run retention based on historical relationships.

5. **Extrapolation models:** Fit a parametric survival model (e.g., Weibull, log-normal) to the early survival curve observed in the experiment. Extrapolate to 90 or 365 days.

**Key caveat:** All non-holdout methods rely on structural assumptions (stability of the surrogate relationship, stationarity of the retention model). Holdouts are the only unbiased approach. Companies like Meta and Google maintain multi-year holdout cells for this reason.

---

**Q48. You have two competing teams, each claiming their feature drives retention. How do you arbitrate?**

**A:** This is a priority conflict that requires both statistical rigor and organizational process.

**Technical approach:**

1. **Check for experiment overlap:** Were both experiments running simultaneously? If so, check if users in both experiments are confounded. Users could be in all four combinations: (A on, B on), (A on, B off), (A off, B on), (A off, B off). Run a 2x2 factorial analysis to estimate main effects and interactions.

2. **Define the causal chain:** Do both features plausibly affect the same intermediate metric (e.g., D7 retention)? Or do they affect different pathways? Draw the DAG (Directed Acyclic Graph) of causal relationships.

3. **Look for interaction effects:** Test `retention ~ feature_A + feature_B + feature_A * feature_B`. A significant interaction means the features are complementary or substitutes.

4. **Sequentially test in isolation:** Run an experiment where only feature A is active, then only feature B, with matched holdouts. This cleanly isolates each effect.

5. **Power analysis for the joint experiment:** A 2x2 factorial has 4 cells, each needing the same n as a standalone A/B test. Confirm you have enough traffic before proceeding.

**Organizational process:**
At SL6, you're also expected to influence the process: propose a **joint prioritization framework** where teams agree on a single retention metric and a protocol for arbitrating conflicts before they launch. This is a data culture problem as much as a statistics problem.

---

**Q49. How do you decide to stop an experiment early?**

**A:** Early stopping is legitimate only if done with proper statistical controls. Uncontrolled early stopping is p-hacking.

**Valid reasons to stop early:**

1. **Pre-planned interim analysis:** You designed a sequential test with O'Brien-Fleming or Pocock spending functions. A pre-specified interim threshold was crossed. Stop and report.

2. **Always-valid sequential inference:** You're using a methodology (e.g., mSPRT, e-values, confidence sequences) that controls error rates under continuous monitoring. The test crossed the decision boundary.

3. **Critical harm detected:** A guardrail metric (crash rate, error rate, safety metric) crossed a harm threshold. You should have a pre-defined "kill switch" rule for this case.

4. **Operational necessity:** The experiment is causing real-time operational problems (e.g., a bug is causing revenue loss). Stop, fix, restart.

**Invalid reasons to stop early:**
- "We peeked and saw p=0.04 today" (peeking problem — inflated Type I error)
- "The business needs an answer faster" (not a valid statistical reason)
- "It looks like it won't be significant, let's kill it" (this is also a form of data dredging)

**Best practice:** At experiment launch, pre-specify:
- Planned end date and sample size
- Any interim analysis schedule
- Harm-based kill switch thresholds
- Whether you're using a sequential or fixed-horizon test

Document everything. Post-experiment, disclose any deviations from protocol.

---

**Q50. What is the difference between ATT, ATE, and LATE? When does each matter?**

**A:** These are three estimands (quantities being estimated) in causal inference:

**ATE — Average Treatment Effect:** The effect of the treatment averaged over the entire population (including non-compliers, never-takers, etc.).
```
ATE = E[Y(1) - Y(0)]
```

**ATT — Average Treatment Effect on the Treated:** The effect of the treatment averaged only over those who actually received it.
```
ATT = E[Y(1) - Y(0) | T = 1]
```

**LATE — Local Average Treatment Effect (IV context):** In an instrumental variables setting, the effect of treatment on "compliers" — those whose treatment status is changed by the instrument.
```
LATE = E[Y(1) - Y(0) | complier]
```

**When each matters:**

| Estimand | When it matters |
|----------|----------------|
| **ATE** | You want to evaluate a policy that would be applied to everyone (e.g., a mandatory UI change) |
| **ATT** | You want to evaluate the impact on users who adopted a feature (e.g., value of a premium subscription for subscribers) |
| **LATE** | You used an IV and want to understand the effect for those who responded to the nudge (e.g., effect of feature adoption for users who adopted because of a prompt) |

**SL6 nuance:** In most A/B tests with perfect compliance (everyone in treatment gets the treatment), ATE = ATT = LATE. The distinction matters most in observational studies, partial-compliance experiments, and IV designs. Misidentifying the estimand leads to wrong policy conclusions.

---

### SECTION 4: Common Scenarios and Edge Cases (Q51–75)

---

**Q51. An experiment shows a big lift in Week 1 that fades to zero by Week 4. What do you conclude?**

**A:** This pattern is the classic **novelty effect**. Users engage more initially because the feature is new and interesting, but the effect decays as the novelty wears off.

**Analysis steps:**

1. **Plot the weekly treatment effect (not cumulative).** If the effect was +5% in week 1, +2% in week 2, +0.5% in week 3, and 0% in week 4, the fading pattern is clear.

2. **Separate new vs returning users.** New users (who joined after the experiment started) have no prior experience with the old product. If new users show a stable effect and returning users show a decaying effect, this confirms novelty.

3. **Test "exposure recency" as a moderator.** Does the effect size correlate with how recently a user last visited? Users who hadn't visited in 30 days may be more excited by novelty.

4. **Extrapolate the asymptote.** Fit an exponential decay model: `effect(t) = δ * exp(-λt) + ε`. The asymptote ε is the estimated steady-state effect. If ε is statistically and practically zero, the feature likely has no long-run value.

**Decision:** If the long-run effect is zero or negligible, do not ship. A feature that delivers no sustained value consumes engineering capacity, adds complexity, and may have been harming one metric (e.g., server cost) for a non-existent benefit.

---

**Q52. Your experiment shows different effects on iOS vs Android. What do you do?**

**A:** Platform-level heterogeneity is common and requires careful investigation.

**Step 1: Check if the HTE is significant.**
Run an interaction test: `Y ~ treatment * platform`. If the interaction is not statistically significant (at p<0.10, given your power), the observed difference may be noise. Don't over-interpret.

**Step 2: Check for SRM or data quality issues by platform.**
Were users randomized identically on iOS and Android? Were there any OS-version-specific bugs in the implementation? Logging differences? Check A/A balance within each platform.

**Step 3: If HTE is real, identify the mechanism.**
- Different UI rendering (the feature looks different on each OS)
- Different user behavior (iOS users are more engaged → higher baseline → different treatment response)
- Different feature implementation on each platform
- Different population demographics (iOS skews higher income in some markets)

**Step 4: Decision options:**
- **Positive on both, larger on one:** Ship to both, report separately
- **Positive on iOS, null on Android:** Ship to iOS only; investigate Android; don't conclude it "works" globally
- **Positive on iOS, negative on Android:** Do not ship. Fix the Android experience first.
- **Positive only after segment split:** Pre-registration violation — treat as exploratory, require replication

---

**Q53. You want to test a pricing change. What special considerations apply?**

**A:** Pricing experiments are among the most ethically and statistically sensitive experiments in tech.

**Ethical and legal considerations:**
- In many jurisdictions, **price discrimination** (showing different prices to different users) is regulated. Random assignment to different prices may violate consumer protection laws in some markets. Always consult legal.
- Users in the control group may find out about the lower price and feel deceived. Transparency risk is real.
- For B2B SaaS, pricing experiments can damage relationships with enterprise customers.

**Statistical considerations:**

1. **Revenue per user has high variance.** Power calculations must account for the heavy-tailed distribution of revenue.

2. **Long-term effects:** A price increase might show short-term revenue lift but increase churn over 30–90 days. Ensure the experiment runs long enough to measure cancellations and downgrades.

3. **Cannibalization:** Test whether users assigned the higher price simply downgrade or delay, showing up as revenue later in the control arm.

4. **Geographical randomization:** If user-level price randomization is legally risky, randomize by geography (country, DMA). Analyze with appropriate cluster corrections.

**Metrics:**
- Primary: revenue per user (or LTV-adjusted)
- Guardrails: cancellation rate, refund rate, customer support contacts, NPS

---

**Q54. How do you run an experiment on a feature that affects both sides of a two-sided marketplace?**

**A:** Two-sided marketplaces (Airbnb, Uber, eBay, Amazon Marketplace) have buyers and sellers. A treatment on one side creates supply-demand spillovers that affect the other side and contaminate standard A/B tests.

**Example:** If you give 50% of Airbnb hosts a better pricing tool, their listings become more competitive. This affects the demand side (buyers see better prices) even in the "control" group — because supply has changed. The control buyers are now contaminated.

**Design strategies:**

1. **Geographic market randomization:** Randomize entire markets (cities/regions) to treatment or control. Within a market, everyone experiences the same condition. Between markets, you estimate the treatment effect. Requires many markets for adequate power.

2. **Switchback experiments:** Alternate treatment and control across time periods for entire markets. Reduces market-level variance. Must include washout periods between switches to prevent carryover.

3. **One-sided treatment with tight market definition:** If you can ensure supply and demand pools don't overlap (e.g., certain seller categories serve distinct buyer segments), user-level randomization on one side may work.

4. **Estimation under interference:** After the experiment, use regression to estimate the "peer effects" (spillover) and the direct effect separately. Compute total ATE = direct effect + spillover.

**At SL6:** You should be able to design and analyze a geo-randomized marketplace experiment, including variance estimation for cluster-randomized designs and the design effect calculation.

---

**Q55. How do you detect and handle a logging bug in the middle of an experiment?**

**A:** Logging bugs mid-experiment are a common operational challenge. The key principle: **a broken measurement instrument gives you no information**.

**Detection:**
1. **Monitor experiment diagnostic metrics in real time:** event volume per arm, session counts per arm, metric distributions. Sudden breaks in trends signal logging changes.
2. **Run automated SRM checks daily:** If the treatment/control split drifts suddenly (e.g., from 50/50 to 52/48 overnight), investigate immediately.
3. **Check for metric anomalies:** Sudden spikes or drops in the variance of a metric, or unusual changes in the fraction of null/missing values.
4. **Correlate with deployment events:** Cross-reference experiment anomalies with backend deployments, A/B system updates, or logging schema changes.

**Handling options (in order of preference):**

1. **Fix the bug and restart the experiment.** This is the safest option. Discard the data from the buggy period entirely.

2. **Truncate the analysis:** If the bug only affected the last 3 days of a 21-day experiment, analyze only the first 18 days. Valid if the bug is truly isolated to that period.

3. **Analyze pre/post separately:** If you can identify the exact logging change timestamp, use a DiD analysis with "pre-bug" as baseline and test whether treatment effects differ pre/post bug.

4. **Never patch noisy data with imputation and proceed as if nothing happened.** This is not legitimate.

**Documentation:** Log all known data quality issues in the experiment record. This is essential for reproducibility and future meta-analysis.

---

**Q56. An executive says "the p-value is 0.06, just round it up to significant." How do you respond?**

**A:** This is a critical moment for scientific integrity. The response needs to be both principled and politically intelligent.

**The honest answer:**
"I understand the pressure to ship, but a p-value of 0.06 means there's a 6% chance we'd see this result by random chance if there's no real effect. Our pre-specified threshold was 0.05. If we move the threshold post-hoc, we lose the ability to trust our experiment results going forward — including the ones that tell us to ship things we're excited about."

**Reframe constructively:**
- "Our 95% CI is [−0.2%, +3.1%]. The point estimate is positive. What we can say is the most likely effect is a small positive lift, but we're not confident enough to claim it's real. Options: (1) extend the experiment by two more weeks to increase power, (2) ship to 10% and monitor, or (3) ship and accept we may be wrong on this one."

- "What was our pre-specified MDE? If the current estimate is within our MDE range, we may have been underpowered to begin with — that's worth understanding separately."

**What not to do:**
- Don't agree to "round up" the result
- Don't silently re-run the test with different subgroups until you find p<0.05
- Don't recommend shipping without flagging the uncertainty

**SL6 behavior:** This is where senior data scientists earn trust — by being the independent check on premature decision-making, while offering real alternatives that address the underlying business need.

---

**Q57. How do you test whether a new feature has differential impact across user cohorts defined by acquisition date?**

**A:** Acquisition cohorts differ systematically — older users have more history, different engagement patterns, and different exposure to prior features. HTEs across cohorts are common and important.

**Analysis:**

1. **Define cohorts:** Group users by acquisition period (e.g., acquired in Q1, Q2, Q3). These are stable, pre-treatment covariates.

2. **Run interaction tests:**
```
Y ~ treatment + cohort + treatment * cohort
```
Test the significance of the interaction terms using F-test or likelihood ratio test (with multiple cohort dummies, test the joint significance of all interaction terms).

3. **Visualize cohort-specific effects:** Plot point estimates and 95% CIs for the treatment effect within each cohort. Look for monotonic trends (effect grows with cohort age) or non-monotonic patterns.

4. **Account for multiple comparisons:** With 6 cohorts, you have 6 interaction terms. Apply Bonferroni or BH correction.

**Interpretation:**

- Effect concentrated in new users → feature is great for acquisition/onboarding; may not help retention of existing users
- Effect concentrated in old users → feature addresses a mature-user need; won't help early retention
- Opposite effects across cohorts → very important — shipping could help some users while hurting others; need cohort-specific decisions or product changes

**SL6 extra:** Also test whether the feature differentially affected **user value** across cohorts. High-LTV old users being harmed is a much bigger concern than low-LTV new users being harmed.

---

**Q58. How do you handle an experiment where the treatment group has significantly higher opt-out rates?**

**A:** This is a variant of the attrition/dropout problem with an important nuance: opt-out behavior itself is a **treatment outcome** and can be analyzed as such.

**Step 1: Analyze opt-out as an outcome.**
Test whether the opt-out rate is different between arms (chi-squared test on opt-out counts). If the treatment causes more opt-outs, that's a signal — the feature may be intrusive or unwanted.

**Step 2: Intent-to-Treat (ITT) analysis.**
Analyze all originally assigned users, regardless of opt-out. This gives you the policy-relevant estimate of the treatment's impact on the full assigned population.

**Step 3: Complier Average Causal Effect (CACE/LATE).**
Among users who stayed in both arms, estimate the effect. But be careful — "users who didn't opt out in treatment" is a selected subset, not comparable to "all users in control."

**Step 4: Bounds analysis.**
Estimate upper and lower bounds on the treatment effect, assuming the opt-outs would have had the best or worst possible outcome. If the bounds both show a positive effect, you can be confident in the result. If the bounds cross zero, the opt-outs are threatening validity.

**Decision implications:**
High opt-out in treatment is a strong signal that users dislike the feature. Even if the primary metric shows a lift among those who stay, you must weigh the fact that you're keeping users who tolerate the feature while losing those who don't. This is survivorship bias in your measured effect.

---

**Q59. How do you measure the impact of server-side latency improvements on user behavior?**

**A:** Latency experiments have several unique features: the treatment is often invisible (users don't know they got faster service), latency effects are non-linear, and the primary metric must capture user behavior change, not just latency itself.

**Experiment design:**
- Randomize users or requests to two server configurations: one at normal latency, one with artificially added latency (or artificially reduced latency if testing an optimization). Use the added-latency arm as control to measure the value of lower latency.
- Alternatively, if you've made a server-side optimization, randomize by user to old vs new code path.

**Primary metrics:**
- Engagement metrics downstream of the page load: task completion rate, session depth, return visits, conversion rate
- Not just "did the page load faster" — you need behavioral response

**Analysis nuances:**

1. **Non-linear dose-response:** The effect of latency is not linear. A 100ms reduction at the 2-second level matters more than at the 200ms level. Analyze the effect size as a function of baseline latency using quantile regression or binning.

2. **Long-tail latency:** Mean latency may improve while p99 gets worse. Check the full distribution (p50, p75, p90, p95, p99). Users who experience extreme latency are often the most valuable or most likely to churn.

3. **Platform differences:** Mobile users are more sensitive to latency than desktop. Desktop users on slow connections are more sensitive than those on fast connections. Analyze HTEs by connection type and device.

4. **Cache effects:** If one arm benefits from a warm cache and the other doesn't, the comparison is unfair. Ensure identical cache warming conditions.

---

**Q60. What's the difference between an online experiment and an offline evaluation?**

**A:**

**Offline evaluation** assesses a model or algorithm using historical data, before any real users are involved.
- **Methods:** Holdout evaluation (train on past, evaluate on future), cross-validation, A/B replay with logged data
- **Pros:** Fast, cheap, no user risk
- **Cons:** Doesn't capture distribution shift, can't measure user behavioral response, can be gamed by training on the test distribution, counterfactual evaluation requires logged propensities

**Online experiment (A/B test)** deploys the change to real users and measures real behavioral outcomes.
- **Methods:** User-split A/B, interleaving, multi-armed bandit
- **Pros:** Measures real causal effect on real user behavior; the gold standard
- **Cons:** Slow, expensive, risks harm to users, novelty/primacy effects

**Why they often disagree:**
- Offline metrics (accuracy, NDCG, AUC) may not correlate with online business metrics (CTR, retention)
- Offline evaluation assumes logged user behavior is a proxy for satisfaction — but users click on recommended items even when they're mediocre (position bias)
- The offline distribution (what was shown in logs) is different from what the new model would show (counterfactual problem)

**The "offline-to-online gap"** is a major research area. Many organizations invest heavily in calibrating their offline metrics to be predictive of online outcomes by running many paired offline+online experiments and measuring correlation.

---

### SECTION 5: Platform, Infrastructure, and Advanced Topics (Q61–100)

---

**Q61. How does Uber/Lyft design experiments in a ride-sharing marketplace?**

**A:** Ride-sharing has severe interference problems: drivers and riders share a two-sided market where supply and demand are globally coupled within a city.

**The core problem:** If you give 50% of drivers a new surge pricing algorithm (treatment), they change their behavior. This shifts supply in the marketplace, affecting prices and wait times for all riders — including those in "control." Standard A/B test is invalid.

**Solutions used in practice:**

1. **City-level randomization:** Randomize entire cities to treatment or control. Within a city, everyone gets the same experience. Compare outcomes across cities. Requires many cities and careful matching on pre-experiment characteristics.

2. **Switchback experiments:** Within a city, alternate treatment and control in time windows (e.g., 30-minute blocks). Both drivers and riders experience alternating conditions. Analyze with regression controlling for time-of-day and day-of-week fixed effects. Must include washout windows between switches to clear carryover.

3. **Driver-side holdout with demand-side measurement:** Treat drivers; measure rider outcomes (wait time, ETAs) as a function of what fraction of nearby drivers are in treatment. This allows estimating both direct (driver) and indirect (rider) effects.

4. **Simulation:** Build a simulator calibrated to historical supply-demand dynamics. Test new algorithms in simulation before deploying. Useful for a first pass but requires model accuracy.

---

**Q62. What is a bandit algorithm and when would you use it instead of an A/B test?**

**A:** A **multi-armed bandit** is an adaptive algorithm that continuously allocates more traffic to better-performing variants, balancing exploration (learning about variants) and exploitation (sending traffic to the best-known variant).

**Types:**
- **ε-greedy:** With probability ε, explore randomly; with 1-ε, exploit the current best arm
- **UCB (Upper Confidence Bound):** Choose the arm with the highest UCB = mean + confidence bonus. Naturally explores uncertain arms.
- **Thompson Sampling:** Sample from the posterior distribution of each arm's expected reward; choose the arm with the highest sample. Bayesian; naturally handles uncertainty.

**When to use bandit over A/B:**

| Scenario | A/B Test | Bandit |
|----------|----------|--------|
| Long experiment runway, can wait for results | ✓ | |
| High cost of showing suboptimal treatment | | ✓ |
| Non-stationary environment (user preferences change) | | ✓ |
| Need unbiased causal estimate for future decisions | ✓ | |
| Many variants (10+) to test | | ✓ |
| Safety-critical — must control Type I error | ✓ | |

**Key limitation of bandits:** They don't produce unbiased causal estimates because traffic allocation is not random — it correlates with the metric being measured. This makes it difficult to use bandit results for causal inference or for building models.

---

**Q63. How does Google's Overlapping Experiment Framework work?**

**A:** Google runs thousands of experiments simultaneously on the same user population. A naive approach would require each experiment to "own" a separate slice of users, quickly running out of traffic. The **Overlapping Experiment Framework** (Kohavi et al.) allows experiments to share the same users.

**Core idea:** Use **independent hash functions** for each layer of experiments. A user's assignment in experiment A is determined by `hash(user_id + "exp_A") mod 100`, and in experiment B by `hash(user_id + "exp_B") mod 100`. Because different hash seeds produce independent assignments, the two experiments are statistically independent even though they run on the same users.

**Layers and domains:**
- Experiments are grouped into **domains** (e.g., search ranking, UI, ads)
- Within a domain, experiments are **independent** — the same user can be in multiple experiments simultaneously
- Domains can be **disjoint** — users are split between domains such that experiments in domain A never overlap with domain B (used when interactions between experiments in different domains are expected)

**Interaction detection:** Since experiments overlap, you must check for experiment-experiment interactions. Monitor whether the presence of experiment A changes the effect of experiment B. Companies use automated interaction detection by running 2x2 factorial analyses on pairs of overlapping experiments.

**SL6 key insight:** The framework is what enables tech companies to run thousands of experiments per year on a fixed user base. Understanding it is essential for SL6 infrastructure roles.

---

**Q64. What is a metric movement budget and how do you use it in experiment analysis?**

**A:** A **metric movement budget** (or sensitivity budget) specifies how much each metric is expected to move in a given experiment, based on historical experiments and product understanding. It's a calibration tool that catches unusual results early.

**Purpose:**
- **Anomaly detection:** If a metric moves 10x its typical range, it likely indicates a bug, data quality issue, or something unexpected in the experiment.
- **Power calibration:** If metrics routinely move less than the MDE you used in your power analysis, your MDEs may be too optimistic. Recalibrate.
- **Inter-experiment comparison:** Understand what a "large" vs "small" effect means for each metric in context.

**How it works:**
1. Collect all past experiments and their treatment effects (with standard errors) for each metric.
2. Compute the empirical distribution of observed effects. The 90th percentile of absolute effects is a reasonable "large effect" benchmark.
3. When a new experiment produces an effect > 5 SD from the historical mean, flag for review.
4. Calibrate MDEs as a fraction of typical effect sizes (e.g., set MDE at 25th percentile of historical effects for "medium sensitivity").

**SL6 usage:** Build this as an automated check in your experiment review system. It surfaces unusual results faster than manual review and reduces false alarms from coding bugs.

---

**Q65. How do you handle an experiment that accidentally exposed the control group to part of the treatment?**

**A:** This is **treatment contamination** — a serious threat to internal validity. The severity depends on the fraction contaminated and the mechanism.

**Detection:**
- Check whether any control users have treatment-specific events in their logs (e.g., control users seeing the new UI component, receiving the treatment email, etc.)
- Measure the contamination rate: `fraction of control users with treatment exposure`

**Impact assessment:**
- If contamination rate is small (<1%) and random (not correlated with user characteristics), the bias in the ATE estimate is approximately `contamination_rate * true_effect`. For a true effect of 5% and contamination of 1%, bias is ~0.05% — likely negligible.
- If contamination is systematic (e.g., high-value users are disproportionately contaminated), the bias is larger and unpredictable.

**Remediation options:**

1. **Exclude contaminated users from analysis:** Analyze only "pure control" (never-treated control users). This is an ITT violation and can introduce selection bias, but may be acceptable if contamination is random.

2. **IV approach:** Use original assignment as an instrument for actual treatment received. This estimates the LATE for compliers.

3. **Restart the experiment:** If contamination is severe, the only clean option is to restart with a fixed randomization system.

4. **Document and bound the bias:** If the experiment can't be restarted, report results with explicit contamination bias bounds: "The estimated effect is X, but given Y% contamination, the true effect could be as high as Z."

---

**Q66. What is experiment triggering and why does it matter?**

**A:** **Triggering** refers to the practice of only including users in the experiment analysis if they actually "triggered" the condition being tested — i.e., they were in a state where the treatment could have affected them.

**Example:** You're testing a new checkout button style. You randomize 1M users to treatment and control. But only 50K users actually visited the checkout page during the experiment. The other 950K were never exposed to the treatment. Including them in the analysis dilutes the measured effect toward zero.

**Triggered analysis:** Only analyze users who reached the checkout page (the trigger event). This is called the "triggered" or "as-treated" population.

**Why it matters:**
- **Increased power:** By restricting to triggered users, the signal-to-noise ratio improves dramatically. For a true effect of 5% on triggered users, the diluted effect on all users is `0.05 * (50K/1M) = 0.25%` — nearly undetectable.
- **Interpretability:** The triggered effect is the causal effect on the right population (users who saw the treatment).

**Requirements for triggered analysis:**
- The trigger event must be **pre-treatment** — it must occur (or be determined) before the treatment affects the user. If the treatment itself changes whether users reach the trigger, you have endogeneity.
- The trigger must be defined in the experiment protocol before launch.
- Both arms should show similar trigger rates (check this!). If treatment causes more/fewer users to trigger, you have a problem.

---

**Q67. How would you build an experimentation platform from scratch at a startup?**

**A:** This is a systems design + strategy question common at SL6 interviews. Here's a phased approach:

**Phase 1: Minimum Viable Platform (0–6 months)**
- **Assignment service:** Random hash-based assignment (`hash(user_id + experiment_id) mod 100 < treatment_percentage`). Store assignments in a key-value store or append to event logs.
- **Logging:** Ensure every user action logs `user_id`, `timestamp`, `experiment_id`, `variant`. Use your existing event logging pipeline.
- **Analysis:** Simple SQL queries on your data warehouse. T-test or proportions test, manual p-value calculation or scipy.
- **Documentation:** Spreadsheet tracking experiment name, hypothesis, primary metric, dates, results.

**Phase 2: Self-Service Platform (6–18 months)**
- **Experiment configuration UI:** Product managers can create experiments, set traffic allocation, define start/end dates.
- **Automated analysis:** Daily/weekly statistical reports with automated p-value, CI, and power computation.
- **SRM detection:** Automated chi-squared check on assignment splits.
- **Metric catalog:** Define all metrics once; reuse across experiments.

**Phase 3: Mature Platform (18+ months)**
- **Sequential testing / always-valid CIs:** Enable continuous monitoring without peeking problem.
- **CUPED integration:** Automated variance reduction using pre-experiment covariates.
- **Holdout management:** Long-running holdout cells for measuring cumulative effects.
- **Interaction detection:** Automated checks for experiment-experiment interactions.
- **Meta-analysis:** Aggregate learnings across experiments to build institutional knowledge.

**Key decisions:**
- **Randomization unit:** User vs session vs device — must match your product's user identity model.
- **Build vs buy:** Statsig, Optimizely, LaunchDarkly, GrowthBook (open source) are alternatives to building. At very early stage, buy. Build once you've outgrown SaaS limitations.

---

**Q68. What are e-values and how do they differ from p-values for sequential testing?**

**A:** **E-values** are a recent (2020s) alternative to p-values that are specifically designed for sequential testing and provide valid inference at any stopping time.

**P-value limitations in sequential settings:**
- A p-value is only valid at the pre-specified sample size. Stopping early based on p<0.05 inflates Type I error.
- Combining p-values across looks requires pre-specified spending functions.

**E-value definition:**
An e-value `E` is a non-negative random variable satisfying:
```
E[E | H₀] ≤ 1
```
Under the null, the expected value of `E` is ≤ 1. An e-value of 20 means: under the null, you'd observe E ≥ 20 with probability ≤ 1/20 = 5%.

**Key properties:**
- **Anytime valid:** You can peek at the e-value at any time and reject H₀ if `E ≥ 1/α` (e.g., E ≥ 20 for α = 0.05). No spending function needed.
- **E-values multiply:** If you run sequential e-tests, you can multiply e-values across looks: `E_total = E_1 * E_2 * ... * E_k`. This gives a valid combined test.
- **Compatible with Bayes Factors:** E-values are closely related to Bayes Factors and have a natural Bayesian interpretation.

**Practical use:** Companies like Spotify and Netflix have begun implementing e-value-based monitoring systems that allow continuous dashboards without inflating false positive rates. At SL6, you should understand the theoretical basis and be able to implement e-value tests using methods like the GRAPA (Generalized Randomized Anytime-valid Permutation Approach).

---

**Q69. How do you build a long-term experimentation culture in an organization resistant to it?**

**A:** This is a leadership and organizational design question, but deeply statistical at its core.

**Common resistance patterns and responses:**

1. **"We don't have time to run experiments"**
Response: Frame experimentation as risk reduction, not added cost. "Without a test, we're shipping blind. A 2-week test can tell us if this feature will cost us users before we invest in the full rollout."

2. **"We already know this will work"**
Response: "HiPPO (Highest Paid Person's Opinion) decisions are right about 10–40% of the time, based on published research from companies like Amazon, Google, and Booking.com. Experiments tell us when intuition is wrong before it's too late to course-correct."

3. **"The experiment killed a good feature"**
Response: "The experiment saved us from shipping something that wouldn't deliver the expected value. That's the experiment working correctly — it's information, not failure."

**What builds culture:**
- **Celebrate learnings, not just wins:** Publish results of experiments that showed null or negative results. This signals that the organization values truth over validation.
- **Executive-level experimentation reviews:** Make experiment results part of quarterly reviews. This signals that leadership takes data seriously.
- **Fast experiments:** Reduce experiment cycle time (through CUPED, better tooling, streamlined review). If experiments take 3 months, teams won't use them. If they take 2 weeks, they'll use them constantly.
- **Training and templates:** Provide accessible guides, training, and templates that lower the activation energy for teams to run their first experiment.

---

**Q70. How do you estimate the value of your experimentation program?**

**A:** This is a strategic and statistical question that comes up at SL6+ performance reviews, exec presentations, and program justification.

**Method 1: Holdout from all experiments**
Run a "meta-holdout" — a small percentage of users never receives any of the features shipped based on A/B test learnings. After 6–12 months, compare this group to the regular population. The gap is the cumulative value of the experimentation program.

**Method 2: Feature-level attribution**
For each shipped feature, record the measured lift and the user population. Sum across all shipped features:
```
Total value = Σ (lift_i * users_i * revenue_per_user)
```
This is an upper bound because it doesn't account for cannibalization between features.

**Method 3: Counterfactual shipping analysis**
Estimate what decisions would have been made without the experiments (based on executive intuition, prior trends, competitor shipping). Compare actual outcomes to this counterfactual. Example: "Without experiments, we estimate we would have shipped 3 features that our experiments showed were harmful, avoiding an estimated $12M in lost revenue."

**Method 4: Win rate benchmarking**
Track the "win rate" — fraction of experiments that show a positive primary metric. Compare to industry benchmarks (~30% of experiments win at mature companies). A higher win rate means better hypothesis generation and feature development.

**SL6 output:** Present the value of experimentation to leadership as an ROI: "The program cost $X in data science time and tooling. It delivered $Y in value through better product decisions. ROI = Y/X."

---

**Q71. How do you analyze an experiment where users can change segments during the experiment?**

**A:** This is the **dynamic treatment regime** or **time-varying confounding** problem. Users are assigned to a segment (e.g., "free" vs "paid") that may change during the experiment because of the treatment itself.

**Example:** You test a feature that encourages free users to upgrade. Some users convert to paid during the experiment. If you analyze by "current segment at measurement time," paid users in treatment include natural upgraders + treatment-induced upgraders, while paid users in control are only natural upgraders. The segments are now non-comparable.

**Solutions:**

1. **Analyze by segment at assignment time (baseline segment):** Lock the segment label at experiment start. A user who was "free" at assignment stays "free" in the analysis, even if they later upgraded. This gives a clean estimate of the treatment effect on the segment of users who were free at baseline.

2. **Intent-to-Treat with all users:** Analyze all users regardless of segment. The ITT estimate includes the upgrade conversion effect, which may be exactly what you want to measure.

3. **Causal mediation analysis:** If you want to separate the "direct" effect of the feature (on free users' engagement) from the "indirect" effect (through conversion to paid), use mediation analysis. Requires strong assumptions.

4. **Marginal Structural Models (MSM):** For complex time-varying treatments, use inverse probability of treatment weighting (IPTW) to estimate marginal effects. Requires modeling the time-varying treatment probability.

---

**Q72. What is a Difference-in-Differences design and what assumptions must hold?**

**A:** **Difference-in-Differences (DiD)** is a quasi-experimental method that compares the change in outcomes over time between a group that received a treatment and a group that did not.

**Setup:**
- Two groups: treated (T) and control (C)
- Two time periods: pre (0) and post (1)
- DiD estimator:
```
DiD = [E(Y|T,post) - E(Y|T,pre)] - [E(Y|C,post) - E(Y|C,pre)]
```

**Critical assumption: Parallel Trends**
"Absent the treatment, the treated group would have experienced the same change in outcome as the control group." This is an **untestable assumption** in the post period. But you can test it in pre-period data — if both groups were trending parallel before treatment, this supports the assumption.

**Testing parallel trends:**
- Plot pre-period trends for both groups visually
- Run a placebo test: use an earlier period as the "treatment" date and check if DiD is zero
- Run a regression with group × time interactions for the pre-period only

**Other assumptions:**
- **No spillover:** Control group is not affected by treatment (SUTVA)
- **No anticipation:** Treated units don't change behavior before the treatment takes effect
- **Stable composition:** No differential attrition between groups

**Extensions:**
- **Staggered DiD:** Treatment rolls out at different times for different units. Requires Callaway-Sant'Anna or Sun-Abraham estimators (standard TWFE can be biased in this setting).
- **Synthetic DiD (Arkhangelsky et al.):** Combines DiD with synthetic control weighting for better pre-treatment fit.

---

**Q73. How do you handle experiments with long observation windows (e.g., annual subscription renewal)?**

**A:** Annual subscription renewal experiments require users to be tracked for 12 months, but you can't wait a year for every experiment.

**Approaches:**

1. **Cohort analysis with earlier renewal cohorts:** Use users who are approaching renewal (within 30 days). Their renewal decision is imminent. You get results in 1–2 months, not 12.

2. **Surrogate metric approach:** Find short-term metrics that predict renewal. Candidates: usage frequency in weeks 1–4, feature adoption rate, NPS score, number of consecutive active weeks. Validate the surrogate on historical cohorts where you know renewal outcomes.

3. **Hazard model extrapolation:** Fit a survival model (Cox proportional hazards) to the early churn data (e.g., monthly cancellations in the first 3 months). Compare hazard ratios between treatment and control. Use the model to extrapolate to 12-month renewal probability.

4. **Long holdout with phased rollout:** Ship the feature to 90% of users. Keep 10% as a permanent holdout. After 12 months, compare renewal rates between the holdout and the shipped population.

5. **Continuous-time event modeling:** Use a proportional hazards model with time-varying covariates to estimate the effect of the treatment on the hazard of cancellation at each time point.

**SL6 consideration:** For subscription businesses (Netflix, Spotify, SaaS), churn is the most important metric. Investing in surrogate validation studies and long holdout infrastructure is essential.

---

**Q74. What is regression discontinuity and how would you apply it to a product problem?**

**A:** **Regression Discontinuity (RDD)** exploits a sharp threshold in an assignment rule to estimate a causal effect. Units just above and just below the threshold are similar in expectation, so comparing their outcomes gives a local causal effect.

**Setup:** Assignment `T = 1 if X ≥ c`, where X is the "running variable" and c is the cutoff. Near the cutoff, treatment is as good as random.

**Product application example:** "Users who join on a day when your app has a > 4.5-star rating see a different onboarding flow (treatment). You want to know if the 4.5+ rating causes better retention."

**Implementation:**
1. Identify a threshold in your assignment rule (score cutoff, engagement threshold, account age, etc.)
2. Plot the outcome variable against the running variable, separately for units above and below the cutoff
3. If the outcome jumps discontinuously at the cutoff, that's the causal effect
4. Estimate using local linear regression on both sides of the cutoff:
```
Y = α + β*X + δ*T + γ*(T*X) + ε
```
where `T = 1[X ≥ c]` and X is centered at c. The coefficient δ is the RD estimate.

**Bandwidth selection:** Choose the window around the cutoff (e.g., only users with X within ±5 of c). Wider bandwidth = more data but more potential for bias. Use MSE-optimal bandwidth (Imbens-Kalyanaraman or CCT selector).

**Validity checks:**
- No "sorting/manipulation" around the cutoff (users shouldn't be able to game their X value)
- No discontinuities in pre-treatment covariates at the cutoff (McCrary density test)
- Placebo cutoffs show no discontinuity

---

**Q75. What is causal discovery and when does it apply to A/B testing analysis?**

**A:** **Causal discovery** is the process of inferring causal structure (a DAG — Directed Acyclic Graph) from observational data, without running experiments.

**Core algorithms:**
- **PC algorithm:** Tests conditional independence in data to infer edges and directions in a DAG. Assumes causal Markov condition and faithfulness.
- **FCI algorithm:** Extension of PC that handles latent confounders.
- **LiNGAM:** Assumes linear non-Gaussian noise structure to identify full causal directions.

**When it applies to A/B testing:**

1. **Interpreting experiment results:** Once you have a significant effect, causal discovery can help identify the mechanism (mediation path). E.g., "Does the feature affect retention directly, or through engagement, or through both?"

2. **Building DAGs for analysis:** Before running a regression or CUPED adjustment, you should have a causal model of the data-generating process. Causal discovery can help confirm or challenge your assumed DAG.

3. **Detecting confounders in observational studies:** When you can't randomize, causal discovery identifies which variables are confounders that must be controlled.

4. **Root cause analysis:** When a metric anomaly occurs (metric suddenly drops), causal discovery on metric time series can help identify which upstream metric is causally responsible.

**SL6 caveat:** Causal discovery is exploratory and assumption-heavy. It generates hypotheses, not confirmed causal structures. Always validate discovered causal relationships with targeted experiments. "Correlation is not causation" applies to output DAGs as much as raw correlations.

---

**Q76–Q100: Rapid-Fire Advanced Questions with Key Answers**

---

**Q76. What is the "winner's curse" in experimentation?**
**A:** When a study is underpowered, only the experiments that "got lucky" with an unusually large observed effect reach significance. The measured effect in significant experiments exceeds the true effect on average. This is called the winner's curse. Effect sizes in underpowered experiments are inflated. Correction: power your experiments adequately; use Bayesian shrinkage to correct for selection bias in published results.

---

**Q77. How do you test whether a metric change is real or due to a change in user composition (Simpson's paradox)?**
**A:** Run the analysis separately within each user segment (device type, geography, new/returning). If the direction of the effect reverses within segments compared to aggregate, you're seeing Simpson's paradox driven by differential sample composition between arms. The within-segment analysis is more causally honest. Use standardization (direct or inverse probability weighting) to control for compositional differences.

---

**Q78. What is an instrumented DiD (event study) and when is it appropriate?**
**A:** An **event study** (or staggered DiD event study) plots the treatment effect at each relative time period (e.g., weeks -4 to -1 before treatment, and weeks 1 to 8 after). The pre-period coefficients test parallel trends; the post-period coefficients trace the dynamic treatment effect. Appropriate when treatment is staggered across units (different rollout dates). Use Callaway-Sant'Anna or Sun-Abraham estimators to avoid TWFE bias with heterogeneous treatment timing.

---

**Q79. What is covariate adjustment in a randomized experiment and does it introduce bias?**
**A:** Covariate adjustment (OLS regression of outcome on treatment + pre-experiment covariates) is valid and unbiased in randomized experiments, even if the model is misspecified. It increases precision by absorbing variance explained by covariates. CUPED is a special case of regression adjustment. No bias is introduced because covariates are independent of treatment by randomization.

---

**Q80. How do you compute a p-value for the difference in medians?**
**A:** Two common approaches: (1) **Permutation test** — permute treatment labels, compute the difference in medians 10,000 times, the fraction of permuted differences exceeding the observed difference is the p-value. (2) **Bootstrap test** — bootstrap the median difference and use the bootstrap distribution. Do not use the Mann-Whitney U test for the median specifically — it tests stochastic dominance, not equality of medians.

---

**Q81. You observe a treatment effect only in week 3 of a 4-week experiment. What do you conclude?**
**A:** Be very skeptical. If you only tested week 3 after seeing no effect in weeks 1–2 and 4, this is data dredging. If week 3 was pre-specified (e.g., you expected a delayed effect), the finding is more credible. Test the week 3 result with appropriate multiple testing correction across the 4 weeks. Consider running a follow-up experiment where the primary metric is measured in week 3 from the start.

---

**Q82. What is an experiment tax and how do you minimize it?**
**A:** An **experiment tax** is the cumulative opportunity cost of running experiments: traffic allocated to control groups that don't receive beneficial features costs value over time. Minimization strategies: (1) run experiments for the minimum duration needed (no longer); (2) use sequential tests to end early when results are clear; (3) use bandits where ethical to reallocate traffic to winning arms; (4) reduce holdout group size when statistical power permits.

---

**Q83. How does network randomization work in practice for a social platform?**
**A:** Partition the social graph into dense clusters using community detection (Louvain algorithm). Randomly assign clusters to treatment or control. Within clusters, edges are mostly intra-cluster, minimizing cross-cluster interference. Analyze at the cluster level: compute the average metric per cluster, run a two-sample test on cluster-level means. Standard error must account for cluster variance. Effective for estimating both direct effects and spillover effects.

---

**Q84. What is the LATE in an intent-to-treat experiment with non-compliance?**
**A:** In ITT analysis, the measured effect is diluted by non-compliers. The LATE (Local Average Treatment Effect) estimates the effect on compliers only — those who take the treatment when assigned. Compute LATE = ITT / compliance rate, where compliance rate = P(took treatment | assigned to treatment) - P(took treatment | assigned to control). This is the Wald IV estimator with assignment as the instrument.

---

**Q85. What is a "false discovery rate" and when should you use it over FWER control?**
**A:** **FDR** (Benjamini-Hochberg) controls the expected fraction of significant results that are false positives. **FWER** (Bonferroni) controls the probability of any false positive. FDR is appropriate when you're exploring many hypotheses and can tolerate some false discoveries (e.g., genomics, secondary metrics analysis). FWER is appropriate when any false positive is costly (e.g., primary metric in a critical safety experiment). At SL6: use FWER for primary metric decisions and FDR for exploratory secondary analysis.

---

**Q86. How do you measure the incremental lift of targeting (i.e., value of showing the right content to the right user)?**
**A:** Run a 2x2 experiment: (1) Targeted content to targeted users, (2) Random content to targeted users, (3) Targeted content to random users, (4) Random content to random users. The incremental lift of targeting = E[Y|targeted content, targeted users] - E[Y|random content, targeted users]. This isolates the value of personalization from the value of the content itself.

---

**Q87. What is the "regression kink design" and how does it differ from RDD?**
**A:** In a **Regression Kink Design (RKD)**, the treatment amount (not assignment) changes slope at a threshold (e.g., a benefit payment that increases more steeply above an income threshold). There's no discontinuity in assignment probability at the cutoff — only a kink in the dosage. The causal effect is estimated as the ratio of the kink in the outcome to the kink in the dosage. Useful when treatment is a continuous variable, not binary.

---

**Q88. How do you estimate treatment effects when the treatment rolls out gradually over time?**
**A:** Use a **staggered DiD** design. Units receive treatment at different times (rollout waves). Standard two-way fixed effects (TWFE) is biased with heterogeneous treatment timing because later-treated units are used as implicit controls for earlier-treated units even while themselves already treated. Use Callaway-Sant'Anna estimator: compute DiD separately for each (cohort, time) pair using never-treated or not-yet-treated units as the control group, then aggregate.

---

**Q89. What is the "file drawer problem" in experimentation and how does it affect meta-analysis?**
**A:** The **file drawer problem** (publication bias) occurs when negative or null experiment results are not shared, creating an inflated view of how often features "work." In internal experimentation, this appears as: teams only share positive results in experiment reviews; null results are quietly discarded. In meta-analysis of past experiments, this biases estimates of feature effectiveness upward. Fix: mandate logging all experiments and results in a central repository, regardless of outcome. Use funnel plots in meta-analysis to detect publication bias.

---

**Q90. How do you design an A/B test for a feature with a viral loop?**
**A:** Viral loops create interference: a user who joins because of a referral from a treatment user is "contaminated" by the treatment. Standard user-level A/B test understates the true effect (because control users also get some viral spillover from treatment users). To estimate the full effect including viral amplification: (1) Randomize at the cohort/network level rather than user level; (2) Use a bipartite graph experiment with separate seeds and adopters; (3) After measuring direct effect in a standard A/B, use a diffusion model to estimate the viral amplification factor (k-factor) and compute: total effect = direct effect × 1/(1 - k).

---

**Q91. What are "canary deployments" and how are they different from A/B tests?**
**A:** A **canary deployment** rolls a new code version to a small fraction of traffic (1–5%) for **operational safety** monitoring — watching for crashes, errors, and latency spikes before full rollout. It is not statistically designed to measure user behavioral outcomes (underpowered for that). An **A/B test** is a statistically designed experiment measuring behavioral impact. In practice: canaries come first (operational validation) → A/B test (behavioral impact measurement) → full rollout. Conflating the two leads to premature conclusions from statistically underpowered canaries.

---

**Q92. How do you handle an experiment with a treatment that has heterogeneous dosage?**
**A:** When users in the treatment arm receive different amounts of treatment (e.g., some users open the notification 3 times, others 0 times), standard ITT dilutes the effect. Options: (1) **ITT analysis** — include everyone; estimates policy effect of assigning the treatment. (2) **Triggered analysis** — restrict to users who received non-zero dosage (endogenous selection risk). (3) **IV/LATE** — use assignment as instrument for dosage; estimates LATE for compliers. (4) **Dose-response analysis** — model `Y ~ f(dosage)` using assignment as an instrument to handle endogeneity in dosage.

---

**Q93. What is the "table 2 fallacy" in causal inference?**
**A:** In regression models, researchers often interpret all coefficients in a multiple regression as causal effects, not just the primary treatment variable. This is the "Table 2 fallacy." Each covariate in the regression has its own causal model requirements (different confounders, different adjustment sets). Including them all in one model doesn't simultaneously provide valid causal estimates for each. At SL6: be explicit about which coefficient you're estimating causally and what identification strategy supports it. Don't report every regression coefficient as a causal estimate.

---

**Q94. How do you use propensity score weighting to estimate ATE from observational data?**
**A:** **Propensity score** `e(X) = P(T=1 | X)` is the probability of treatment given observed covariates X. **Inverse Probability Weighting (IPW)** reweights observations to create a pseudo-population where treatment is independent of X: `ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]`. In practice: (1) Estimate propensity scores using logistic regression or a gradient boosted model; (2) Check overlap (positivity) — each user must have 0 < e(X) < 1; (3) Trim extreme weights to reduce variance; (4) Use doubly robust estimators (combine with outcome regression) for robustness. Assumes no unmeasured confounders.

---

**Q95. What is the "first-stage problem" in IV analysis?**
**A:** In an IV design, the first stage is the regression of treatment T on instrument Z: `T = γ*Z + η`. A **weak instrument** has low `γ` (small first stage), meaning Z barely moves T. Consequences: (1) IV estimates become biased toward OLS estimates (bias toward confounded observational estimate); (2) Standard errors are inflated; (3) The bias-to-variance trade-off worsens. Test: first-stage F-statistic should be >10 (rule of thumb), ideally >16 for 5%-level bias. With multiple instruments, use the Kleibergen-Paap rk F statistic. If first stage is weak, find a stronger instrument or use alternative methods.

---

**Q96. How do you build a "metric tree" for an experiment program?**
**A:** A **metric tree** decomposes a top-level business metric into sub-metrics in a multiplicative or additive structure. Example for an e-commerce revenue metric:

```
Revenue = DAU × Sessions/User × Conversion Rate × Order Value
```

Each factor can be individually tested. The metric tree enables: (1) **Root cause analysis** — when revenue drops, which factor drove it? (2) **Experiment alignment** — product team tests map to specific nodes in the tree; (3) **Counter-metric identification** — a change that improves Conversion Rate may harm Order Value; the tree shows the trade-off; (4) **Sensitivity calibration** — how much does a 1% change in each leaf node affect the root?

---

**Q97. What is "regression adjustment" and how does it differ from CUPED?**
**A:** Both use pre-experiment covariates to reduce variance. **Regression adjustment** adds covariates `X` directly to the treatment effect regression:
```
Y = α + δ*T + β*X + ε
```
The coefficient δ is the adjusted treatment effect. **CUPED** constructs a residualized outcome `Y_cuped = Y - θ*X_pre` and tests on that. They are mathematically equivalent when `X_pre` is the only covariate and the model is linear. CUPED is more interpretable as a variance reduction; regression adjustment is more flexible (multiple covariates, interaction terms). Both are unbiased in randomized experiments.

---

**Q98. How do you measure statistical power for a difference-in-difference design?**
**A:** Power in DiD depends on: (1) Number of treated vs control units; (2) Number of pre and post time periods; (3) Within-unit serial correlation (ICC over time); (4) True effect size. Use the formula for two-sample t-test on the DiD estimate:

```
Var(DiD) = 4 * σ² * [1 + (T-1)*ρ] / (N * T)
```

where T = number of time periods, ρ = autocorrelation within units, N = total units, σ² = residual variance. High within-unit autocorrelation (ρ) inflates variance substantially. When ρ is unknown, use historical data to estimate it. For small numbers of treated units (<10), use randomization inference rather than asymptotic normal approximations.

---

**Q99. What is "interference bias" and how do you quantify it?**
**A:** **Interference bias** occurs when the treatment of one unit affects another unit's outcome, violating SUTVA. The standard ATE estimator under interference estimates a mixture of direct effects and spillover effects. To quantify interference: (1) Design a two-stage experiment: randomize clusters to treatment intensity (0%, 50%, 100% treated within cluster); (2) Estimate the **spillover effect** = outcome difference between untreated units in high-treatment vs low-treatment clusters; (3) Estimate **direct effect** = outcome difference between treated and untreated units within the same cluster; (4) **Total effect** = direct + spillover. Spillover quantification requires the multi-arm design — it cannot be recovered from standard A/B data.

---

**Q100. How do you make a final ship/no-ship recommendation when results are ambiguous?**
**A:** Ambiguous results are the norm, not the exception. A rigorous SL6-level recommendation covers:

1. **Characterize uncertainty explicitly:** Report the point estimate, 95% CI, and the probability that the true effect exceeds the MDE (from a Bayesian posterior or power analysis).

2. **Segment the ambiguity:** Is the overall effect null because effects cancel across segments? Or is the effect uniformly near zero? These have different implications.

3. **Value-of-information analysis:** How much would additional data reduce uncertainty? If 2 more weeks of data would resolve the question, extend. If the CI barely moves with more data (high intrinsic variance), it may never resolve.

4. **Expected value calculation:** `EV(ship) = P(effect > 0) * E[value | effect > 0] + P(effect < 0) * E[cost | effect < 0]`. Compare to `EV(no ship) = 0` (or the opportunity cost of not shipping). If EV(ship) > 0, ship.

5. **Reversibility assessment:** Is the change easily rolled back? If yes, shipping at low confidence is lower risk. If no (e.g., a pricing change, a data deletion), the bar for confidence is higher.

6. **Document the decision:** Record all evidence, the uncertainty, the decision rationale, and a plan to monitor post-ship. This enables organizational learning and accountability.

---

*End of Document*

---

> **Prepared for SL6-level data science interviews at Google, Meta, and Apple.**
> *All statistical methods described are industry-standard; specific implementations vary by company platform.*
