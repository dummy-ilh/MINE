# Statistical power: what it means and how to control it

## Definition

**Power = the probability of correctly rejecting H₀ when the alternative H₁ is actually true.**

$$\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true})$$

Where β is the probability of a **Type II error** — failing to detect a real effect.

In plain terms: if there truly is a difference (or effect), power is *how likely your test is to actually catch it*. High power = your experiment is a sensitive instrument. Low power = your experiment could easily miss a real effect and you'd wrongly conclude "no difference."

---

## The four-quadrant picture (memorize this cold)

| | H₀ actually true | H₁ actually true |
|---|---|---|
| **Test says "reject H₀"** | Type I error (α) — false positive | Correct (**power**, 1−β) |
| **Test says "fail to reject H₀"** | Correct (1−α) | Type II error (β) — false negative |

- **α (significance level)** — your tolerance for false positives. You set this (usually 0.05).
- **β** — probability of a false negative. Power is 1−β.
- These trade off against each other, but not one-for-one — they're controlled independently through different levers (see below).

---

## Why power matters

A **non-significant result (p > 0.05) with low power tells you almost nothing.** You didn't confirm "no effect" — you may have just run an underpowered experiment incapable of detecting the effect even if it exists. This is precisely misinterpretation #3 from the p-value notes: absence of evidence ≠ evidence of absence.

Common industry convention: **aim for power ≥ 0.80** (i.e., you'd only miss a true effect 20% of the time) before trusting a null result.

---

## The four levers that control power

Power is driven by four things — this is the core of the topic:

| Lever | Effect on power | Why |
|---|---|---|
| **Sample size (n)** ↑ | Power ↑ | More data shrinks standard error ($\propto 1/\sqrt{n}$), making real effects easier to detect against noise |
| **Effect size** ↑ | Power ↑ | A bigger true difference is easier to distinguish from zero |
| **Significance level (α)** ↑ | Power ↑ | Loosening your false-positive tolerance (e.g., 0.10 instead of 0.05) makes it easier to reject H₀ — but at the cost of more false positives |
| **Variability (σ)** ↓ | Power ↑ | Less noise in the data makes the signal easier to detect |

**The practical control knobs, in order of how much control you usually have:**

1. **Increase sample size** — the primary lever in practice. This is what "power analysis" usually solves for: given a target effect size, α, and desired power, how many samples do you need?
2. **Reduce variability** — better measurement instruments, more homogeneous samples, blocking/stratification, or paired/repeated-measures designs (comparing within-subject reduces noise from between-subject variation).
3. **Choose α thoughtfully** — raising α trades Type I risk for power; usually not the first lever to pull since it directly inflates false positives.
4. **Target a realistic effect size** — you can't manufacture a bigger true effect, but you can decide the *minimum effect size worth detecting* (MDE) up front, which determines how much power/sample size you actually need. Chasing tiny effects always needs disproportionately more data.

---

## Power analysis (a priori sample size calculation)

Before running an experiment, solve for the required n given:
- Desired power (commonly 0.80)
- Significance level α (commonly 0.05)
- Expected/minimum detectable effect size (Cohen's d, or a business-relevant delta)
- Expected variance

For a two-sample z-test, approximate required sample size per group:

$$n \approx \frac{2\sigma^2(z_{\alpha/2} + z_{\beta})^2}{\delta^2}$$

where δ is the minimum effect size you want to detect. Key intuition from this formula: **required sample size scales with the square of 1/δ** — halving the effect size you want to detect quadruples the sample size needed. This is why detecting small effects (e.g., a 0.1% conversion lift) requires massive sample sizes.

---

## Interview Q&A

**Q1: In plain English, what is statistical power?**
A: The probability your test detects a real effect when one truly exists. If power is 0.8, and there's a genuine difference, you'll correctly find it 80% of the time — and miss it (a false negative) 20% of the time.

**Q2: Your A/B test came back p = 0.24, "no significant difference." What's your first question?**
A: What was the power of this test? If the sample size was small or the true effect was modest, a non-significant result could just mean the test was underpowered to detect it — not that there's genuinely no difference. I'd check whether the observed effect size, if real, would have been detectable given the sample size used.

**Q3: How do you increase power without collecting more data?**
A: Reduce variance in the measurement — e.g., switch to a paired/within-subject design, control for confounding variables via stratification, or use a more precise metric. You can also raise α, but that directly increases your false-positive rate, so it's usually the least preferred lever.

**Q4: If you halve the minimum effect size you want to detect, how does the required sample size change?**
A: It roughly quadruples — required n scales with $1/\delta^2$. This is why "surface a smaller lift" requests from stakeholders often carry a steep, non-obvious sample size cost.

**Q5: What's the relationship between Type I error, Type II error, and power?**
A: α is your tolerance for false positives (test says "effect" when there isn't one); β is the false-negative rate (test misses a real effect); power = 1 − β. Lowering α (being stricter about false positives) generally lowers power for a fixed sample size, unless you compensate with more data — that trade-off is the central tension in test design.

**Q6 (curveball): A colleague says "we got p = 0.03, so our test was well-powered." What's wrong with this reasoning?**
A: Power is a property of the test *design* (computed before or independent of the observed result) — it's the probability of detecting a true effect *if one exists*, calculated from sample size, expected effect size, and α. A single significant p-value doesn't retroactively prove the study was well-powered; you could get a significant result from an underpowered study by chance (or by inflating effect size via selective reporting). Power should be planned a priori, not inferred from the outcome.

**Q7: Why do genomics studies use α = 5×10⁻⁸ instead of 0.05, and what does this do to required power/sample size?**
A: With millions of simultaneous hypothesis tests (one per genetic variant), a 0.05 threshold would produce enormous numbers of false positives via multiple comparisons. Tightening α to control the family-wise error rate directly reduces power for a fixed n — which is why genomics studies compensate with extremely large sample sizes (often hundreds of thousands of subjects).
