# Chapter 4: Hypothesis Testing Basics — Type I/II Errors, P-Values, Power

## 1. Definition

Hypothesis testing is a framework for deciding whether observed data provides enough evidence to reject a default assumption (the null hypothesis, H₀) in favor of an alternative (H₁).

- **Null hypothesis (H₀):** the "no effect" baseline — e.g., the new feature has no impact on conversion rate.
- **Alternative hypothesis (H₁):** the effect exists — e.g., the new feature changes conversion rate.
- **Type I error (α):** rejecting H₀ when it's actually true — a false positive (concluding there's an effect when there isn't).
- **Type II error (β):** failing to reject H₀ when H₁ is actually true — a false negative (missing a real effect).
- **P-value:** the probability of observing data at least as extreme as what you got, *assuming H₀ is true*.
- **Power (1-β):** the probability of correctly detecting a true effect when one exists.

## 2. Layman Explanation

Think of hypothesis testing like a courtroom trial. The default assumption is "innocent" (H₀ = no effect). The evidence (your data) needs to be strong enough to convict (reject H₀) beyond a reasonable doubt.

- **Type I error** = convicting an innocent person (you say the feature works, but it actually doesn't — a false alarm).
- **Type II error** = letting a guilty person go free (the feature actually works, but your test wasn't convincing enough to catch it).
- **P-value** = "if this person were truly innocent, how surprising is this evidence?" A very low p-value means the evidence would be very unusual if H₀ were true — so you doubt H₀.
- **Power** = how good your trial is at catching real guilt when it exists. A trial with low power might let real winners (features that actually help) slip through as "not significant."

Crucially: a low p-value doesn't tell you "there's only a 5% chance the null is true" — it tells you "if the null were true, data this extreme would only show up 5% of the time." That distinction trips up almost everyone, including PMs and even some data scientists.

## 3. Formal Explanation

**Decision framework:**

| | H₀ True | H₀ False |
|---|---|---|
| Reject H₀ | Type I error (α) | Correct (Power = 1-β) |
| Fail to reject H₀ | Correct (1-α) | Type II error (β) |

**P-value formalism:**
p = P(observing test statistic this extreme or more | H₀ is true)

We reject H₀ if p < α (typically α = 0.05).

**Power formalism:**
Power = P(reject H₀ | H₁ is true) = 1 - β

Power depends on:
- Effect size (larger true effect → easier to detect → higher power)
- Sample size (n)
- Significance threshold α (looser α → higher power, but more Type I risk)
- Variance of the underlying metric (σ²) — noisier metrics reduce power for a given n

**Key relationship (for a z-test on means):**
Power is a function of: (effect size × √n) / σ, compared against the critical value at α.

This is why sample size calculations (Chapter 9 in this curriculum) are really "solve for n given a target power, effect size, and variance."

**Common conceptual traps:**
- p-value is NOT P(H₀ is true | data). That's a Bayesian posterior, and requires a prior — frequentist p-values don't provide this.
- Failing to reject H₀ does NOT prove H₀ is true — it just means you don't have enough evidence to reject it (absence of evidence ≠ evidence of absence).
- Statistical significance ≠ practical significance — with a huge enough n, even a trivially small, business-irrelevant effect can produce p < 0.05.

## 4. Levers — What Controls It, What Moves It

**α (significance threshold)**
- Lowering α (e.g., 0.05 → 0.01) reduces Type I error risk but increases Type II error risk (lower power) for the same sample size — a direct tradeoff.
- Google/Apple-scale companies often use stricter α for high-stakes launches (e.g., pricing changes) and looser thresholds for low-risk UI tweaks.

**Sample size (n)**
- Larger n increases power directly — more data means real effects are easier to distinguish from noise. This is the primary lever teams pull when a test is "underpowered."

**Effect size**
- Larger true effects are inherently easier to detect. This isn't something you control, but it affects how you set your Minimum Detectable Effect (MDE) when planning sample size — chasing tiny effects requires disproportionately more data.

**Variance (σ²)**
- Lower variance (via CUPED, stratification, better metric definitions) increases power without needing more users — same lever as prior chapters, now applied to hypothesis testing directly.

**One-sided vs. two-sided tests**
- A one-sided test (testing only for improvement, not degradation) has more power to detect an effect in the specified direction for the same α, because all the rejection region is on one side. But it comes at the cost of being unable to detect an effect in the opposite direction — a risky choice if a feature could plausibly hurt the metric.

## 5. Famous Q&A (Google / Apple style)

**Q: A test returns p = 0.03. Does that mean there's a 97% chance your feature actually works?**
A: No — this is one of the most common misinterpretations of p-values. The p-value of 0.03 means: *if the null hypothesis (no effect) were true*, you'd see data this extreme (or more) only 3% of the time. It says nothing directly about the probability that H₀ or H₁ is true — that would require a Bayesian framework with a prior. The correct statement is "assuming no true effect, this result would be unusual," which is evidence against H₀, but not a direct probability statement about the hypothesis itself.

**Q: You ran a test, got p = 0.12, and conclude "the feature has no effect." What's wrong with this conclusion?**
A: Failing to reject H₀ is not the same as proving H₀ true. A p-value of 0.12 could mean there's genuinely no effect, or it could mean there IS an effect but your test was underpowered (too small a sample, too much noise) to detect it. Before concluding "no effect," you should check the power of your test given your actual sample size and observed variance — if power was low, the honest conclusion is "inconclusive," not "no effect."

**Q: A test on 5 million users shows a statistically significant 0.02% lift in click-through rate (p < 0.001). Should you ship it?**
A: This is the statistical vs. practical significance trap. With a sample size that large, even a trivially small, real effect will produce a very low p-value — the test is highly powered to detect tiny effects. The question isn't "is it significant" but "is 0.02% lift worth the engineering cost, maintenance burden, and any tradeoffs (e.g., latency, complexity) of shipping this?" A senior-level answer weighs the effect size against business cost, not just the p-value.

**Q: Your team lowers the significance threshold from 0.05 to 0.01 to be "more rigorous." What's the tradeoff?**
A: You reduce the Type I error rate (fewer false positives — fewer times you'll incorrectly conclude a dud feature works), but you increase the Type II error rate for the same sample size — real effects now need to clear a higher bar, so you're more likely to miss true positives (lower power). If the team wants both lower α and maintained power, the only lever left is increasing sample size (running the test longer or with more traffic).

---
*Next: Chapter 5 — Causal Inference vs. Correlation: counterfactual framing and why we need experiments at all.*
