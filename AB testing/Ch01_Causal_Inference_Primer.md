# Chapter 1: Causal Inference Primer

## 1. Intuition

Every A/B test is trying to answer one question: **"What would have happened if this specific user had NOT seen the treatment?"**

You can never observe this. If a user sees the new checkout flow and buys, you cannot rewind time and show them the old flow to see if they'd have bought anyway. This is called the **fundamental problem of causal inference** — for any single unit, you only ever observe one of the two possible outcomes.

A/B testing is the engineering solution to this philosophical problem. Instead of trying to find the counterfactual for one person, we use **randomization** across many people so that, on average, the treatment group and control group are identical in every way except the treatment itself. Any difference in outcomes can then be attributed to the treatment — not to who happened to get it.

This is why randomization is not just "good practice" — it's the entire mechanism that lets you make causal claims instead of just correlational ones.

## 2. The Formal Framework (Potential Outcomes / Rubin Causal Model)

For each unit $i$ (user), define two **potential outcomes**:

- $Y_i(1)$ = the outcome if unit $i$ receives treatment
- $Y_i(0)$ = the outcome if unit $i$ receives control

The **individual treatment effect** is:

$$\tau_i = Y_i(1) - Y_i(0)$$

You can never compute $\tau_i$ directly because you only observe one of $Y_i(1)$ or $Y_i(0)$ for any given person — never both. This is the fundamental problem stated formally.

What you *can* estimate is the **Average Treatment Effect (ATE)**:

$$ATE = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]$$

The key identity that makes A/B testing valid:

$$E[Y_i(1) - Y_i(0)] = E[Y_i \mid T_i=1] - E[Y_i \mid T_i=0]$$

This equation says: "the true average treatment effect equals the difference in observed means between treatment and control groups" — **but only if treatment assignment $T_i$ is independent of the potential outcomes.** This independence is exactly what randomization guarantees. Without it, this equation is false, and the gap between the two sides is called **confounding bias**.

## 3. Why Observational Data Fails

Say you didn't run an experiment, and instead looked at users who *chose* to enable a new feature vs. those who didn't. You compare their 30-day retention and treatment users retain better.

Is the feature causing retention? Maybe. Or:
- More engaged users are more likely to explore new features (**selection bias** — the pre-existing difference, not the treatment, drives the outcome)
- Power users who already retain well also tend to try new things (**confounding**: an unmeasured variable, "underlying engagement," drives both feature adoption and retention)

Formally, without randomization:

$$E[Y_i \mid T_i=1] - E[Y_i \mid T_i=0] = \underbrace{ATE}_{\text{true effect}} + \underbrace{E[Y_i(0)\mid T_i=1] - E[Y_i(0)\mid T_i=0]}_{\text{selection bias}}$$

That second term is zero *only* under randomization, because randomization makes $T_i$ independent of $Y_i(0)$ — treatment and control groups have the same baseline potential outcome in expectation. This is the mathematical reason "correlation isn't causation" — the bias term doesn't vanish on its own.

## 4. Worked Example

Suppose a company observes:
- Users who opted into a "dark mode" feature: avg session length = 25 min
- Users who didn't opt in: avg session length = 18 min

Naive conclusion: dark mode increases session length by 7 minutes.

Now suppose the true data generating process is:
- Power users (heavy usage, avg 24 min baseline) are 3x more likely to opt into new features
- Dark mode's true causal effect is only +2 minutes

The naive comparison is contaminated by the fact that heavy users self-selected into treatment. The observed 7-minute gap = 2 minutes true effect + ~5 minutes selection bias from pre-existing differences in usage intensity.

**If this had been a randomized experiment** with 50/50 assignment independent of usage pattern, both groups would have the same mix of power users and casual users in expectation, and the observed difference would converge to the true +2 minute effect.

This is precisely why, when a PM says "let's just look at what happened for users who used the feature vs. those who didn't" instead of running an A/B test, the correct pushback is: **that's an observational comparison, not a causal one, and it's very likely confounded.**

## 5. Production Considerations

- **SUTVA (Stable Unit Treatment Value Assumption)**: potential outcomes framework assumes one user's treatment doesn't affect another user's outcome. This assumption breaks in social products (referrals, feeds, marketplaces) — covered fully in Chapter 7 (Interference).
- **Compliance/dilution**: even in a randomized experiment, if treatment users don't actually experience the treatment (e.g., ad blockers, feature flags failing to load), your ATE estimate is diluted toward zero — this is an Intent-to-Treat (ITT) vs Treatment-on-Treated (TOT) distinction worth knowing at L5.
- **External validity**: ATE is specific to your population and time period. A result from a US-only experiment may not generalize to other markets — interviewers sometimes probe this with "would you ship this globally based on this result?"

## 6. Interview Traps

- **Trap**: Saying "correlation isn't causation" without being able to explain *why*, mathematically, randomization fixes it. Interviewers want the selection bias term, not the slogan.
- **Trap**: Confusing "no randomization" with "no causal claim possible at all" — quasi-experimental methods (diff-in-diff, regression discontinuity, instrumental variables) exist precisely to recover causal effects without randomization, and mentioning these signals depth.
- **Trap**: Not distinguishing ATE from individual treatment effect. If asked "does this feature help every user," the honest answer is "we only know the average effect — individual effects can vary and some users may even be negatively affected" (this connects to heterogeneous treatment effects, Chapter 15).

## 7. L5-Differentiating Talking Points

- Explicitly stating the fundamental problem of causal inference and the potential outcomes notation shows formal grounding, not just "I've run experiments before."
- Bringing up SUTVA violations *unprompted* when discussing a social/marketplace product signals you think about interference before it bites you in production — this is a strong L5 signal because junior candidates only think about it after being asked.
- Being able to say "if we can't randomize here for [reason X], I'd reach for diff-in-diff or a regression discontinuity design instead" shows you're not a one-tool thinker.

## 8. Comprehension Check

1. Write out the fundamental problem of causal inference in potential-outcomes notation. Why can't $\tau_i$ ever be directly observed?
2. A colleague says "engagement went up after we launched feature X, so it's working." What's missing to call this causal?
3. Explain in one sentence why randomization makes $E[Y_i(0) \mid T_i=1] = E[Y_i(0) \mid T_i=0]$.
4. Give an example of a SUTVA violation in a product you've worked with or know well.
5. What's the difference between ITT and TOT, and which one does a standard A/B test analysis usually report by default?

---
*Next: Chapter 2 — Hypothesis Testing Framework*
