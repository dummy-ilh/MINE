# 🚨 15 — Common Interview Traps in Hypothesis Testing

> *"The most dangerous phrase in statistics is 'it's significant.'"*

This section catalogs the most common traps interviewers set — and exactly how to avoid them.

---

## 🪤 Trap 1: "Fail to Reject" vs. "Accept"

### The Trap

> "Your A/B test showed p = 0.32. The two variants are the same."

### Why It's Wrong

Failing to reject H₀ does **not** mean H₀ is true. It only means there's insufficient evidence to disprove it. The true state could be:
- H₀ is actually true (no effect)
- H₀ is false but the test was underpowered
- The effect exists but is smaller than your MDE

### ✅ Correct Language

> "We **fail to reject** H₀ at α = 0.05. We do not have sufficient evidence to conclude there is a difference. This does not prove equivalence — we may have been underpowered."

### Equivalence Testing

If you want to *prove* variants are equivalent (e.g., two drugs perform the same), use **Two One-Sided Tests (TOST)**:

- H₀: |μ₁ − μ₂| ≥ Δ (practically different)
- H₁: |μ₁ − μ₂| < Δ (practically equivalent)

This requires *rejecting* bounds of practical significance in both directions.

---

## 🪤 Trap 2: The p-value Probability Fallacy

### The Trap

> "p = 0.02 means there's a 2% chance our hypothesis is wrong."
> "p = 0.02 means we're 98% confident the effect is real."

### Why It's Wrong

The p-value is $P(\text{data} \mid H_0)$, not $P(H_0 \mid \text{data})$.

$$P(\text{data} \mid H_0) \neq P(H_0 \mid \text{data})$$

To compute $P(H_0 \mid \text{data})$, you need Bayes' theorem:

$$P(H_0 \mid \text{data}) = \frac{P(\text{data} \mid H_0) \cdot P(H_0)}{P(\text{data})}$$

This requires a **prior** $P(H_0)$. Without it, the statement is ill-defined.

### Numerical Illustration of the Fallacy

> Suppose a drug test is conducted. Only 1 in 100 drugs tested actually works (prior = 1%). A study with α = 0.05 and 80% power:
>
> - True positives: $0.01 \times 0.80 = 0.008$
> - False positives: $0.99 \times 0.05 = 0.0495$
> - If the result is significant: $P(\text{true effect} \mid p < 0.05) = \frac{0.008}{0.008 + 0.0495} = 14\%$
>
> Even with p < 0.05, there's only a 14% chance the drug works! This is the **false positive risk** / base rate neglect.

---

## 🪤 Trap 3: Confidence Interval Confusion

### The Trap

> "Our 95% CI is [0.02, 0.08]. There's a 95% probability the true value is in this range."

### Why It's Wrong

The true parameter is fixed. The interval either contains it or not (with probability 0 or 1). The 95% is a property of the **procedure**, not the specific interval.

### ✅ Correct Statement

> "We used a procedure that will capture the true parameter in 95% of repeated experiments. Our observed interval is [0.02, 0.08]."

Or colloquially: "We are 95% **confident** (not: certain with 95% probability) the true effect is between 2% and 8%."

---

## 🪤 Trap 4: The Peeking Problem (Optional Stopping)

### The Trap

> "Let's just check the results every day and stop when we see p < 0.05."

### Why It's Wrong

Under continuous monitoring, the probability of ever crossing p = 0.05 by chance (even with H₀ true) is **much higher than 5%**.

**Simulation result:** If you check daily for 20 days and stop when p < 0.05, your true false positive rate is ~30–40% instead of 5%.

```
Day 1:  p = 0.42 (keep going)
Day 5:  p = 0.12 (keep going)
Day 11: p = 0.04 ← STOP! But this is pure chance.
```

### ✅ Solutions

1. **Pre-specify sample size** and only look once at the end.
2. **Sequential testing** with alpha-spending functions (O'Brien-Fleming):
   - The significance threshold is more conservative at early looks and relaxes as the test proceeds.
3. **Always Valid Inference** (Optimizely's SPRT-based approach): control error rates under continuous monitoring.
4. **Bayesian A/B testing**: No fixed stopping rule needed; posterior continuously updated.

---

## 🪤 Trap 5: "More Data = Better Test"

### The Trap

> "We had 1 million users, so our test must be very reliable."

### The Counter-Trap

With massive samples:
- Minuscule effects become highly significant
- Practical significance is dwarfed by statistical significance
- Every guardrail metric will show "significant" changes

### ✅ Response

Large samples increase power (good), but they also detect effects too small to matter (bad if not accounted for). Always pair p-values with effect sizes and confidence intervals. Pre-specify the Minimum Detectable Effect that's practically meaningful.

---

## 🪤 Trap 6: Ignoring Assumptions

### The Trap

> "We ran a t-test. p = 0.03, so the variants are different."

### Hidden Pitfalls

| Assumption | What Happens If Violated |
|---|---|
| Independence | Invalid SE; p-value unreliable |
| Normality (small n) | t-test p-value wrong |
| Equal variances (pooled t) | t-statistic biased |
| i.i.d. observations | All inference breaks |

### In A/B Tests Specifically

- **Non-independence**: Users who see the same content due to caching/cookies (use user-level randomization)
- **Novelty effect**: First-week behavior != steady state
- **Survivorship bias**: Analyzing only users who completed the funnel

---

## 🪤 Trap 7: One-Tailed After Seeing Data

### The Trap

> "The treatment looks better, so let's use a one-tailed test to get a lower p-value."

### Why It's Wrong

Choosing the tail direction after seeing the results doubles your effective Type I error rate. You're implicitly running a two-tailed test but reporting a one-tailed p-value.

### ✅ Rule

One-tailed vs. two-tailed must be decided **before** data collection, based on the scientific question, not the observed direction.

---

## 🪤 Trap 8: Subgroup Mining

### The Trap

> "Overall the test failed, but looking at users in Brazil aged 25–34 who use iOS, it's significant with p = 0.02!"

### Why It's Wrong

With enough subgroups tested, at least one will show significance by chance. If you test 40 subgroups, expect 2 false positives at α = 0.05.

### ✅ Correct Approach

1. **Pre-specify subgroups** of interest before the experiment
2. Apply multiple testing correction
3. Treat any post-hoc subgroup findings as **hypothesis-generating**, not hypothesis-confirming — run a follow-up test

---

## 🪤 Trap 9: Misinterpreting Non-Significant Results for Business

### The Trap

> "The test wasn't significant, so let's roll back to the old design."

### The Problem

If the test was underpowered, non-significance says nothing. The effect might be real but undetected. Check:
1. What was the observed effect size?
2. What was the statistical power?
3. Does the CI include both practically significant and trivial values?

If CI = [−0.001, 0.003] (very narrow, near zero), then "no effect" is a reasonable conclusion.  
If CI = [−0.10, 0.08] (very wide), then we simply don't know — we're underpowered.

---

## 🪤 Trap 10: Confounding in Non-Randomized Tests

### The Trap

> "Users who use feature X have 40% higher retention. Feature X increases retention."

### Why It's Wrong

**Correlation ≠ causation.** Users who opt into feature X may be more engaged *anyway* (selection bias). The feature may not be *causing* higher retention — both could be driven by user engagement levels.

### ✅ Fix

This is why **randomized controlled experiments** (A/B tests) exist. For observational data, use causal inference techniques: propensity score matching, instrumental variables, or difference-in-differences.

---

## 🎤 FAANG-Level Meta Questions

**Q: "Walk me through how you'd design and analyze an A/B test end to end."**

> Perfect structure: (1) Define hypothesis and primary metric; (2) Calculate sample size based on baseline, MDE, α=0.05, power=80%; (3) Randomize at user level; (4) Run until target n is reached — no peeking; (5) Analyze with the pre-specified test (Z-test for proportions, t-test for means); (6) Report p-value, effect size, CI; (7) Check guardrail metrics; (8) Make a ship/no-ship decision with business context.

**Q: "What would you do if your A/B test shows different results across platforms?"**

> This is heterogeneous treatment effects. First, check if this is a pre-specified analysis or post-hoc subgroup mining. If pre-specified: run the test for each platform independently with corrected α (Bonferroni). Investigate the mechanism: Is the UI different? Are users different? Possibly run separate experiments per platform. If post-hoc: treat as exploratory, run confirmatory experiment.

**Q: "Your experiment shows p = 0.06. The PM wants to ship. What do you say?"**

> I'd say: "The result doesn't meet our pre-specified significance threshold (α = 0.05). Shipping based on p = 0.06 is equivalent to accepting a ~6% false positive rate — slightly higher than our agreed tolerance. Let me look at the confidence interval and effect size. If the CI lower bound is still meaningfully positive and the effect is practically significant, I'd recommend extending the experiment to increase power rather than lowering our standards. We should not move the goalposts after seeing the data."

---

## 📋 Quick Reference: Correct vs. Incorrect Statements

| Incorrect ❌ | Correct ✅ |
|---|---|
| "p = 0.03, there's a 3% chance this is due to chance" | "If H₀ were true, data this extreme occurs 3% of the time" |
| "We accept H₀" | "We fail to reject H₀" |
| "95% CI means 95% probability the true value is inside" | "95% of such intervals will contain the true value" |
| "The effect is real because p < 0.05" | "There is statistically significant evidence of an effect at α=0.05" |
| "p = 0.06, just run it longer to hit significance" | "We should not run longer without adjusting for sequential testing" |
| "This subgroup is significant — let's focus there" | "This subgroup finding is exploratory and needs confirmation" |
| "Bigger sample = more reliable conclusion" | "Bigger sample = more power, but also detects trivially small effects" |

---

*← [12-14 — A/B Testing](12_13_14_abtesting_multtest_practical.md) | [16 — Cheat Sheet & Decision Tree →](16_cheatsheet_faang_questions.md)*
