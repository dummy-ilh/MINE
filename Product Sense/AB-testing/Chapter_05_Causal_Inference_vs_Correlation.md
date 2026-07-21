# Chapter 5: Causal Inference vs. Correlation — Counterfactual Framing
# Master Tutorial: Causal Inference & the Counterfactual Framework (Google/Apple-style Interview Prep)

---

## 1. Intuition

Every A/B test is trying to answer one question: **"What would have happened if this specific user had NOT seen the treatment?"**

You can never observe this. If a user sees the new checkout flow and buys, you cannot rewind time and show them the old flow to see if they'd have bought anyway. This is the **fundamental problem of causal inference** — for any single unit, you only ever observe one of the two possible outcomes.

A/B testing is the engineering solution to this philosophical problem. Instead of trying to find the counterfactual for one person, we use **randomization** across many people so that, on average, the treatment group and control group are identical in every way except the treatment itself. Any difference in outcomes can then be attributed to the treatment — not to who happened to get it.

This is why randomization is not just "good practice" — it's the entire mechanism that lets you make **causal** claims instead of just **correlational** ones.

### Layman analogy
Imagine you take a painkiller and your headache goes away 30 minutes later. Did the pill cause the relief, or would the headache have gone away on its own? You can't rewind time and *not* take the pill to check — you only get one version of reality for yourself.

This is exactly the problem in product analytics: if a user sees a new feature and then converts, did the feature *cause* the conversion, or would they have converted anyway? Correlation just tells you the two things happened together. Causal inference is the toolkit for approximating "what would have happened without the feature" — the missing, unobservable alternate reality — using clever comparisons (like a similar group who didn't see the feature).

A/B testing is the cleanest tool for this: by randomly assigning users to see the feature or not, the "control" group becomes your best stand-in for what "would have happened" to the treatment group had they not seen it — because randomization makes the two groups statistically identical in expectation on every other dimension.

---

## 2. Core Definitions

- **Correlation**: measures how two variables move together — it says nothing about whether one causes the other.
- **Causal inference**: the discipline of establishing whether a change in X actually *causes* a change in Y, distinct from X and Y merely co-occurring due to a third factor or reverse causation.
- **Counterfactual**: for a single unit (a user, a store, a market), the causal effect of a treatment is the difference between what *did* happen under treatment and what *would have* happened under control — for the *same* unit, at the *same* time. Since we can never observe both outcomes for the same unit simultaneously, every causal inference method is really a strategy for approximating the missing counterfactual.

---

## 3. The Formal Framework (Potential Outcomes / Rubin Causal Model)

For each unit $i$ (user), define two **potential outcomes**:

- $Y_i(1)$ = the outcome if unit $i$ receives treatment
- $Y_i(0)$ = the outcome if unit $i$ receives control

The **individual treatment effect** is:

$$\tau_i = Y_i(1) - Y_i(0)$$

You can never compute $\tau_i$ directly because you only observe **one** of $Y_i(1)$ or $Y_i(0)$ for any given person — never both. This is the **Fundamental Problem of Causal Inference**, stated formally.

What you *can* estimate is the **Average Treatment Effect (ATE)**:

$$ATE = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]$$

### The key identity that makes A/B testing valid

$$E[Y_i(1) - Y_i(0)] = E[Y_i \mid T_i=1] - E[Y_i \mid T_i=0]$$

This equation says: "the true average treatment effect equals the difference in observed means between treatment and control groups" — **but only if treatment assignment $T_i$ is independent of the potential outcomes.** This independence is exactly what randomization guarantees:

$$E[Y_i(0) \mid T_i=1] = E[Y_i(0) \mid T_i=0]$$

In words: in expectation, the control group's observed outcome IS a valid stand-in for what the treatment group *would have* experienced without treatment — because randomization removes systematic differences between groups on all observed and unobserved confounders. Without randomization, this equation is false, and the gap between the two sides is called **confounding bias** (see Section 4).

---

## 4. Why Observational Data Fails — Three Classic Failure Modes

Say you didn't run an experiment, and instead looked at users who *chose* to enable a new feature vs. those who didn't. You compare their 30-day retention and treatment users retain better. Is the feature causing retention? Maybe. Or one of:

1. **Confounding**: a third variable Z causes both X and Y. E.g., users who are already highly engaged are both more likely to seek out a new feature AND more likely to convert — the feature didn't cause the conversion, prior engagement did.
2. **Reverse causation**: Y actually causes X, not the other way around. E.g., users who are about to churn browse the help center more — help-center visits don't cause churn, impending churn causes help-center visits.
3. **Selection bias**: the sample itself is non-random in a way tied to the outcome. E.g., only power users opt into a beta feature — comparing beta users to non-beta users conflates the feature's effect with pre-existing differences between power users and everyone else.

### The bias term, formally

Without randomization:

$$E[Y_i \mid T_i=1] - E[Y_i \mid T_i=0] = \underbrace{ATE}_{\text{true effect}} + \underbrace{E[Y_i(0)\mid T_i=1] - E[Y_i(0)\mid T_i=0]}_{\text{selection bias}}$$

That second term is zero **only** under randomization, because randomization makes $T_i$ independent of $Y_i(0)$ — treatment and control groups have the same baseline potential outcome in expectation. This is the mathematical reason "correlation isn't causation" — the bias term doesn't vanish on its own; interviewers want you to be able to produce this term, not just recite the slogan.

---

## 5. Worked Examples

### Example A — Dark mode and session length (quantified selection bias)

Observed data:
- Users who opted into "dark mode": avg session length = 25 min
- Users who didn't opt in: avg session length = 18 min

**Naive conclusion**: dark mode increases session length by 7 minutes.

Now suppose the true data-generating process is:
- Power users (heavy usage, avg 24 min baseline) are 3x more likely to opt into new features.
- Dark mode's true causal effect is only **+2 minutes**.

The naive comparison is contaminated by the fact that heavy users self-selected into treatment. The observed 7-minute gap = 2 minutes true effect + ~5 minutes selection bias from pre-existing differences in usage intensity.

**If this had been a randomized experiment** with 50/50 assignment independent of usage pattern, both groups would have the same mix of power users and casual users in expectation, and the observed difference would converge to the true +2 minute effect.

This is precisely why, when a PM says "let's just look at what happened for users who used the feature vs. those who didn't" instead of running an A/B test, the correct pushback is: **that's an observational comparison, not a causal one, and it's very likely confounded.**

### Example B — Dark mode and retention (same trap, different metric)

Users who enable a new "dark mode" setting show 15% higher retention than users who don't. Does dark mode cause retention? Not necessarily — same confounding/selection setup as Example A: users who proactively enable a new setting are likely more engaged, more technically savvy, or more invested in the product to begin with. That prior engagement could be driving both the decision to enable dark mode AND the higher retention, with no causal link between the two. To make a causal claim, you'd want to randomly assign dark mode (or default it on for a random subset) rather than let users self-select, so the "enabled" and "not enabled" groups are comparable on everything except the setting itself.

---

## 6. Levers — What Controls Validity of a Causal Claim

**Randomization**
- The single strongest lever for valid causal inference — when done correctly, it breaks the link between treatment assignment and any confounder, observed or unobserved, by construction.
- Must be truly random and enforced at the right level (the "unit of randomization" — user vs. session vs. device) or its guarantees break down.

**Sample size for balance**
- Randomization guarantees balance *in expectation*, but with small samples, imbalance on some confounder can occur by chance. Larger n makes empirical balance more likely to match the theoretical guarantee — this is why teams still check for balance on pre-treatment covariates even in randomized experiments.

**When randomization isn't possible**
- Quasi-experimental methods step in as substitutes for the missing counterfactual: **difference-in-differences, regression discontinuity, instrumental variables, propensity score matching**. Each is a different strategy for constructing a believable stand-in control group when you can't randomize (see Section 8 for when to reach for each).

**Confounder control**
- In observational settings, controlling for known confounders (via regression, matching, stratification) can partially address bias — but only for confounders you've measured. **Unmeasured confounding remains a threat**, which is exactly why randomized experiments are preferred whenever feasible.

---

## 7. Production Considerations

- **SUTVA (Stable Unit Treatment Value Assumption)**: the potential-outcomes framework assumes one user's treatment doesn't affect another user's outcome. This assumption breaks in social products (referrals, feeds, marketplaces) — a network/interference effect.
- **Compliance/dilution**: even in a randomized experiment, if treatment users don't actually experience the treatment (e.g., ad blockers, feature flags failing to load), your ATE estimate is diluted toward zero. This is the **Intent-to-Treat (ITT) vs. Treatment-on-Treated (TOT)** distinction:
  - **ITT**: compares outcomes based on assigned group, regardless of actual compliance — what a standard A/B test analysis reports by default.
  - **TOT**: attempts to estimate the effect specifically among those who actually received/complied with treatment — requires additional assumptions/methods (e.g., instrumental variables) to estimate validly.
- **External validity**: ATE is specific to your population and time period. A result from a US-only experiment may not generalize to other markets — interviewers sometimes probe this with "would you ship this globally based on this result?"

---

## 8. When You Can't Randomize — Quasi-Experimental Toolkit

Confusing "no randomization" with "no causal claim possible at all" is a trap — quasi-experimental methods exist precisely to recover causal effects without randomization:

- **Difference-in-differences**: useful when there's a comparable market/region that didn't get the treatment; compare the *change over time* in both groups (not just the levels) to net out any pre-existing trend differences.
- **Regression discontinuity**: useful when treatment was assigned based on some threshold (e.g., account tier, a score cutoff) — compare units just above vs. just below the cutoff, who should otherwise be similar.
- **Instrumental variables**: useful when there's a variable that affects treatment assignment but not the outcome directly (other than through treatment) — lets you isolate the causal effect despite unmeasured confounding.
- **Propensity score matching**: match treated and untreated units on observed characteristics to construct a more comparable control group from observational data.

Each is an imperfect substitute for randomization — be explicit about the assumptions each relies on and where they could fail, rather than presenting any of them as a drop-in replacement for a true experiment.

---

## 9. Interview Traps

1. **Saying "correlation isn't causation" without being able to explain *why*, mathematically**, randomization fixes it. Interviewers want the selection-bias term (Section 4), not the slogan.
2. **Confusing "no randomization" with "no causal claim possible at all."** Quasi-experimental methods (Section 8) exist precisely to recover causal effects without randomization — mentioning these signals depth.
3. **Not distinguishing ATE from individual treatment effect.** If asked "does this feature help every user," the honest answer is "we only know the average effect — individual effects can vary and some users may even be negatively affected" (heterogeneous treatment effects).
4. **Concluding causation from a dashboard correlation** (e.g., support response time vs. satisfaction) without considering reverse causation or a lurking confounder like issue complexity (see Q&A below).

---

## 10. Famous Interview Q&A

**Q: A colleague says "engagement went up after we launched feature X, so it's working." What's missing to call this causal?**
A: This is a before/after comparison with no counterfactual — you don't know what engagement would have done anyway, absent the launch (seasonality, concurrent launches, general product growth could all explain it). Without a randomized control group (or a credible quasi-experimental substitute like diff-in-diff), you can't rule out these alternative explanations; you'd need to compare against what a comparable, untreated group experienced over the same period.

**Q: Users who enable a new "dark mode" setting show 15% higher retention than users who don't. Does dark mode cause retention?**
A: Not necessarily — a classic confounding/selection setup. Users who proactively enable a new setting are likely more engaged, more technically savvy, or more invested in the product to begin with — that prior engagement could be driving both the decision to enable dark mode AND the higher retention, with no causal link between the two. To make a causal claim, you'd want to randomly assign dark mode (or default it on for a random subset) rather than let users self-select, so the "enabled" and "not enabled" groups are comparable on everything except the setting itself.

**Q: Why is a randomized controlled experiment considered the "gold standard" for causal inference?**
A: Because randomization solves the Fundamental Problem of Causal Inference in expectation — it ensures the control group is, on average, identical to the treatment group on every observed and unobserved characteristic, making the control group's outcome a valid proxy for what the treatment group would have experienced absent treatment. No observational method can fully guarantee this, because you can never be certain you've measured and controlled for every possible confounder.

**Q: A team can't randomize a pricing change (regulatory/business constraints) but still needs to estimate its causal effect. What's your approach?**
A: I'd reach for quasi-experimental designs that try to approximate the missing counterfactual without randomization. For example, difference-in-differences if there's a comparable market/region that didn't get the price change, using it as a proxy control group and comparing the *change* over time in both groups to net out any pre-existing trend differences. Alternatively, regression discontinuity if the price change was rolled out based on some threshold (e.g., account tier), or an instrumental variable if there's a variable that affects treatment assignment but not the outcome directly. Each is an imperfect substitute for randomization, so I'd be explicit about the assumptions each relies on and where they could fail.

**Q: A metrics dashboard shows that customer support response time and customer satisfaction are negatively correlated (faster responses, happier customers). A VP wants to conclude "cutting response time causes higher satisfaction" and mandate a response-time SLA. What do you flag?**
A: This is observational correlation, not a randomized comparison, so reverse causation and confounding are both live possibilities. For instance, easier/simpler issues might both get resolved faster AND naturally produce happier customers — issue complexity is a confounder driving both variables, not the response time itself. Before mandating an SLA on the strength of this correlation, I'd recommend either a randomized test (e.g., randomly prioritizing some tickets for faster response, holding issue type roughly constant) or at minimum a regression controlling for issue complexity/type as an observed confounder, while being upfront that unmeasured confounders could still bias the observational estimate.

---

## 11. L5-Differentiating Talking Points

- Explicitly stating the fundamental problem of causal inference and the potential-outcomes notation shows formal grounding, not just "I've run experiments before."
- Bringing up **SUTVA violations unprompted** when discussing a social/marketplace product signals you think about interference before it bites you in production — a strong L5 signal, since junior candidates only think about it after being asked.
- Being able to say "if we can't randomize here for [reason X], I'd reach for diff-in-diff or a regression discontinuity design instead" shows you're not a one-tool thinker.
- Producing the selection-bias term (Section 4) instead of just saying "correlation isn't causation" demonstrates you understand the mechanism, not just the slogan.
- Distinguishing ITT from TOT, and knowing which one standard A/B analysis reports by default, shows attention to a subtlety many candidates miss.

---

## 12. Comprehension Check (Self-Test)

1. Write out the fundamental problem of causal inference in potential-outcomes notation. Why can't $\tau_i$ ever be directly observed?
2. A colleague says "engagement went up after we launched feature X, so it's working." What's missing to call this causal?
3. Explain in one sentence why randomization makes $E[Y_i(0) \mid T_i=1] = E[Y_i(0) \mid T_i=0]$.
4. Give an example of a SUTVA violation in a product you've worked with or know well.
5. What's the difference between ITT and TOT, and which one does a standard A/B test analysis usually report by default?
6. Name the three classic failure modes of observational (non-randomized) comparisons, with an example of each.
7. In the dark-mode/session-length worked example, decompose the observed 7-minute gap into its true-effect and selection-bias components.
8. A team can't randomize a pricing change due to regulatory constraints. Name two quasi-experimental methods they could use instead, and briefly state what each requires to be valid.

---
*This tutorial merges two chapters on causal inference — one framed around the potential-outcomes formalism with a dark-mode/session-length worked example; the other framed around the counterfactual/correlation-vs-causation framing with a dark-mode/retention worked example and a broader quasi-experimental toolkit. Overlapping content (the potential-outcomes setup, the ATE identity, the dark-mode example) was merged rather than duplicated; both worked examples were kept since they illustrate different metrics and angles on the same bias.*

---
---

# PART 2 — EXTENDED NOTES (added)

Nothing above is changed. This part adds: deeper mechanics on each quasi-experimental method (with a worked numeric diff-in-diff example), heterogeneous treatment effects (CATE/HTE), Simpson's paradox as a causal trap, DAGs as a reasoning tool, a method-selection decision guide, more failure-mode examples, and additional interview Q&A.

---

## 13. Difference-in-Differences — worked numerically

Diff-in-diff assumes **parallel trends**: absent treatment, the treated and comparison group would have moved in the same direction/magnitude over time. The estimator:

$$DiD = (\bar{Y}_{treated,\,after} - \bar{Y}_{treated,\,before}) - (\bar{Y}_{control,\,after} - \bar{Y}_{control,\,before})$$

**Example**: a company rolls out a pricing change in Market A but not in comparable Market B.

| | Before | After | Change |
|---|---|---|---|
| Market A (treated) | \$40 avg order value | \$46 | +\$6 |
| Market B (control) | \$38 avg order value | \$41 | +\$3 |

$$DiD = (46-40) - (41-38) = 6 - 3 = \$3$$

The naive before/after comparison in Market A alone would have claimed a +\$6 effect; diff-in-diff nets out the \$3 of "would have happened anyway" (general market growth, seasonality) captured by Market B's trend, isolating a more credible **+\$3 causal estimate**.

**Where parallel trends can fail**: if Market A and Market B were already diverging *before* treatment (e.g., A was growing faster than B for unrelated reasons), diff-in-diff will misattribute that pre-existing divergence to the treatment. Best practice: plot pre-treatment trends for both groups and visually/statistically confirm they were roughly parallel before treatment, not just compare two snapshots.

---

## 14. Regression Discontinuity — the mechanics

Used when treatment is assigned by a running variable crossing a threshold (e.g., "accounts with >$10K trailing revenue get a dedicated CSM"). Units just above and just below the cutoff are assumed to be nearly identical in every other way — the cutoff, not user choice, decided their treatment status, which is what recovers a locally randomized-like comparison.

**Key assumption**: no ability to precisely manipulate the running variable to land on the desired side of the cutoff. If sales reps can nudge a customer's trailing revenue just over $10K to get them a CSM, that manipulation reintroduces selection bias exactly at the cutoff — a classic RD validity check is testing for a suspicious "bunching" of the running variable just on the favorable side of the threshold.

**What you get**: a **local average treatment effect (LATE)** — valid only near the cutoff, not necessarily generalizable to units far from it (e.g., the effect of a CSM on a $10K account tells you little about its effect on a $500K account).

---

## 15. Instrumental Variables — the mechanics

An instrument $Z$ must satisfy two conditions:
1. **Relevance**: $Z$ is correlated with the treatment $T$.
2. **Exclusion restriction**: $Z$ affects the outcome $Y$ *only* through $T$, not through any other path.

**Classic product example**: estimating the causal effect of app notifications on engagement, where users self-select into high notification volume (confounded by underlying interest). If a server outage randomly delayed/dropped notifications for a subset of users unrelated to their engagement level, that outage could serve as an instrument — it affects how many notifications a user actually received (relevance) but has no other plausible path to affecting engagement (exclusion), assuming the outage was truly unrelated to user behavior.

**Why IV is hard to use well in practice**: the exclusion restriction is *not testable* — it's an assumption you have to argue for qualitatively, since by definition you can't observe the "other path" you're claiming doesn't exist. This is why IV designs live or die on how convincing the instrument's story is, and why interviewers who bring up IV are often testing whether you'll propose a poor instrument without interrogating its exclusion restriction.

---

## 16. Propensity Score Matching — the mechanics

Estimate $P(T_i=1 \mid X_i)$ (the propensity score) from observed covariates, then match treated and untreated units with similar propensity scores to construct a more comparable comparison group.

**Core limitation**: PSM can only balance on *observed* covariates used to build the propensity model — it does nothing for unmeasured confounders, unlike randomization which balances everything (observed and unobserved) by construction. This is the single most important caveat to state whenever PSM comes up: it narrows observational comparisons, it doesn't fully solve them.

---

## 17. Heterogeneous Treatment Effects (CATE/HTE) — beyond the single ATE number

The original chapter's Interview Trap #3 flags that ATE isn't the individual effect — this extends that point into its own toolkit.

- **Conditional Average Treatment Effect (CATE)**: the average effect within a subgroup defined by covariates $X$, e.g., $E[Y_i(1)-Y_i(0) \mid X_i = \text{power user}]$.
- **Why this matters practically**: an ATE of +2% conversion could hide a +8% effect for new users and a −3% effect for existing power users — shipping based on the aggregate ATE alone could quietly harm a valuable segment even while the topline looks like a clean win. This is the same Simpson's-Paradox-adjacent risk flagged in the metrics-framework material for OEC primary metrics, now shown at the causal-estimation level rather than just the descriptive-metrics level.
- **Common methods**: interaction terms in a regression ($Y \sim T + X + T{\times}X$), causal forests / uplift modeling for more flexible, data-driven subgroup discovery.
- **Caution**: hunting for subgroups with a large effect *after* seeing the data (subgroup fishing) is the causal-inference analog of p-hacking — pre-specify the subgroups you'll examine, or treat post-hoc subgroup findings as hypothesis-generating only, not confirmatory.

---

## 18. Simpson's Paradox as a causal trap

A pattern present in aggregate data can reverse or vanish when the data is split by a relevant subgroup — a vivid illustration of confounding, and a favorite interview follow-up to the ATE/CATE discussion above.

**Illustrative pattern**: a new onboarding flow shows *lower* overall conversion than the old flow. But split by acquisition channel, the new flow actually converts *better* within every single channel — the aggregate reversal happens because the new flow, during its rollout, was disproportionately shown to a lower-converting channel mix. Aggregating across a confounded mix of subgroups can flip the sign of an effect entirely, not just its magnitude.

**Practical takeaway**: whenever an aggregate causal estimate is surprising or contradicts subgroup-level intuition, check whether the treatment and comparison groups have the same *composition* across a plausible confounding dimension — this is conceptually the same balance-check instinct used for SRM/covariate balance in randomized experiments, just applied to observational or rollout data instead.

---

## 19. DAGs (Directed Acyclic Graphs) as a reasoning tool

A lightweight way to reason about confounding, mediation, and colliders without necessarily doing formal math — increasingly common in senior-level interview discussions.

- **Confounder**: $Z \rightarrow X$ and $Z \rightarrow Y$ (Z causes both) — must control for Z to avoid bias, exactly the structure in Example A/B above (prior engagement → feature adoption, prior engagement → retention).
- **Mediator**: $X \rightarrow M \rightarrow Y$ (M is on the causal pathway from X to Y) — controlling for M would be a mistake if you want the *total* effect of X on Y, since it would "block" part of the very effect you're trying to measure.
- **Collider**: $X \rightarrow C \leftarrow Y$ (C is caused by both X and Y) — controlling for or conditioning on a collider (even accidentally, e.g., by filtering your analysis population on it) *induces* a spurious association between X and Y where none causally exists. This is a subtler and less well-known trap than confounding, and a good thing to mention if asked "can controlling for more variables ever make things worse?" — yes, controlling for a collider is a case where adding a covariate actively introduces bias rather than removing it.

---

## 20. Method-selection decision guide

| Situation | Reach for |
|---|---|
| You can randomize | A/B test — always the first choice when feasible |
| No randomization, but a comparable untreated group + pre/post data exists | Difference-in-differences (check parallel trends first) |
| Treatment assigned by a threshold/cutoff on a running variable | Regression discontinuity (check for manipulation/bunching at the cutoff) |
| A plausible variable affects treatment but not the outcome except through treatment | Instrumental variables (be honest about how testable the exclusion restriction really is) |
| Rich observational covariates, no natural experiment available | Propensity score matching (caveat: only balances observed covariates) |
| You suspect the aggregate effect hides subgroup variation | Estimate CATE / run pre-specified subgroup analysis, don't just report ATE |
| An aggregate result contradicts subgroup-level intuition | Check for Simpson's Paradox — inspect composition/mix across the suspected confounding dimension |

---

## 21. More Interview Q&A (new)

**Q: You run a diff-in-diff analysis on a pricing change and get a large effect. What's the first thing you'd check before trusting it?**
A: Whether the parallel-trends assumption plausibly holds — I'd plot the treated and control markets' trends *before* the treatment period and check they were moving similarly. If the treated market was already diverging from the control market prior to the change (for unrelated reasons, like a differing local economy), diff-in-diff will misattribute that pre-existing divergence to the treatment, inflating or even fabricating an apparent effect.

**Q: A team wants to use "whether a customer's account happened to get a system outage that delayed their notifications" as an instrument for notification volume's effect on engagement. What would you push on?**
A: The exclusion restriction — whether the outage could have affected engagement through any path *other than* notification volume. If the outage also degraded other parts of the product experience at the same time (e.g., general app slowness), it violates exclusion, since it would affect engagement directly, not just through notifications. I'd also check relevance — does the outage actually meaningfully move notification volume, or is the effect on the "instrument" itself too weak to be useful (a weak-instrument problem).

**Q: Your ATE shows a new feature improves conversion by 2% overall, but a colleague found it hurts a specific customer segment by 3%. Which number should drive the ship decision?**
A: Neither number alone — this is exactly why relying only on the aggregate ATE can be misleading. I'd want the full CATE breakdown across meaningful, pre-specified segments before deciding, and treat this as a segmented-launch decision rather than a single yes/no: potentially shipping to the segments where the effect is positive or neutral while holding back or iterating for the segment showing harm, rather than let a positive aggregate number paper over a real subgroup regression.

**Q: What's a collider, and why is "just control for more variables to be safe" not always good advice?**
A: A collider is a variable caused by *both* your treatment and outcome variables (X→C←Y). Conditioning on it — even by something as innocuous as filtering your analysis to a subset defined by the collider — can create a spurious statistical association between X and Y where no causal link exists, essentially manufacturing bias rather than removing it. It's a case where "control for everything you can measure" is actively wrong advice; which variables to control for needs to be reasoned about causally (e.g., with a DAG), not just statistically.

---

## 22. Extended Comprehension Check (Part 2, new)

9. Using the worked diff-in-diff numeric example (Section 13), explain what would happen to the estimated effect if Market B had been growing *faster* than Market A even before the treatment.
10. What is a Local Average Treatment Effect (LATE), and why is a regression discontinuity result specifically a LATE rather than a general ATE?
11. Give an example (product-related or otherwise) of a valid-sounding instrumental variable, and identify the specific way its exclusion restriction could plausibly be violated.
12. Why can propensity score matching never fully substitute for randomization, no matter how many covariates you include in the propensity model?
13. Describe, in your own words, a scenario where an aggregate treatment effect and a subgroup-level (CATE) effect point in opposite directions, and explain why this isn't a contradiction.
14. Draw (in words) a DAG distinguishing a confounder, a mediator, and a collider, and state which one you should control for if you want the *total* causal effect of X on Y.

---
*Part 2 added: a numeric diff-in-differences walkthrough, deeper mechanics for regression discontinuity, instrumental variables, and propensity score matching (including their specific failure modes), a full CATE/heterogeneous-treatment-effects section, Simpson's Paradox as a causal trap, DAG-based reasoning (confounders/mediators/colliders), a method-selection decision guide, four new interview Q&As, and six new comprehension questions. Nothing from the original chapter was removed or altered.*
