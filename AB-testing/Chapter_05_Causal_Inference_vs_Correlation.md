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
