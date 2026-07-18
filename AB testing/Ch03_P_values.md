# Chapter 3: P-values

## 1. Intuition

The p-value is the single most misinterpreted number in all of statistics, and interviewers know this — it's almost guaranteed to come up, often as a direct "explain a p-value to me like I'm a PM" question.

Here's the intuition to internalize before anything formal: **the p-value answers "how surprising is my data, assuming the null hypothesis is exactly true?"** It does NOT answer "how likely is it that the null hypothesis is true" or "how likely is it that my treatment works." Those require Bayesian reasoning with a prior — the p-value is a purely frequentist object that says nothing about the probability of hypotheses, only about the probability of data.

## 2. The Formal Definition

$$p\text{-value} = P(\text{test statistic at least as extreme as observed} \mid H_0 \text{ is true})$$

For a two-sided test with observed z-statistic $z_{obs}$:

$$p = 2 \times P(Z > |z_{obs}|)$$

where $Z$ is a standard normal random variable. This is a **conditional probability about data, computed under a hypothetical world where $H_0$ holds** — it is not a statement about the world you're actually in.

**Decision rule**: reject $H_0$ if $p < \alpha$.

## 3. What the P-value IS and IS NOT

| It IS | It is NOT |
|---|---|
| $P(\text{data this extreme} \mid H_0 \text{ true})$ | $P(H_0 \text{ true} \mid \text{data})$ |
| A statement about data given a hypothesis | A statement about a hypothesis given data |
| A measure of surprise/incompatibility with $H_0$ | The probability the effect is real |
| Dependent on sample size | A measure of effect size |

The single most important sentence in this chapter: **the p-value is $P(\text{data} \mid H_0)$, not $P(H_0 \mid \text{data})$.** Confusing these two is called the "prosecutor's fallacy" in other contexts, and it's the most common p-value error, including among people who use A/B tests daily.

**Why this confusion matters practically**: A p-value of 0.03 does NOT mean "there's a 3% chance the null is true" or "there's a 97% chance the treatment works." To get $P(H_0 \mid \text{data})$ you'd need Bayes' theorem and a prior on $H_0$ being true — which is exactly what Bayesian A/B testing frameworks do differently (worth mentioning if asked about Bayesian vs. frequentist approaches).

## 4. Worked Example — Building the Intuition from Scratch

Suppose you flip what you believe is a fair coin ($H_0: p=0.5$) 100 times and get 60 heads.

Test statistic: $\hat{p} = 0.60$, $SE = \sqrt{0.5 \times 0.5 / 100} = 0.05$

$$z = \frac{0.60 - 0.50}{0.05} = 2.0$$

$$p = 2 \times P(Z > 2.0) = 2 \times 0.0228 = 0.0456$$

**Correct interpretation**: "If the coin really were fair, we'd see a result this extreme (60+ or 40- heads out of 100) about 4.56% of the time, just from random flipping." Since 0.0456 < 0.05, we reject the fair-coin hypothesis.

**Incorrect interpretation** (the trap): "There's a 4.56% chance the coin is fair." This is wrong — we never computed $P(H_0 \mid data)$, only $P(data \mid H_0)$.

**Now the same data, different sample size**, to show p-values scale with $n$, not just effect size: if instead you got 600 heads out of 1,000 flips (same 60% rate):

$$SE = \sqrt{0.5 \times 0.5/1000} \approx 0.0158, \quad z = \frac{0.10}{0.0158} \approx 6.32$$

This gives $p \ll 0.0001$ — vastly more significant, despite the *effect size* (60% vs 50%) being identical. This is the core reason p-values alone are insufficient — they conflate effect size with sample size, which is exactly why Chapter 2 emphasized reporting effect size and CI alongside p.

## 5. Production Considerations

- **P-values as a continuous quantity, not a cliff.** $p=0.049$ and $p=0.051$ represent nearly identical evidence, but a rigid $\alpha=0.05$ cutoff treats them as categorically different (significant vs not). In practice, treat p-values close to the threshold as "inconclusive, need more data" rather than a hard pass/fail.
- **The "p-value is not the probability of replication" trap.** A p=0.01 result does not mean you'd get p<0.05 again 99% of the time if you reran the experiment — replication probability depends on the true effect size and power, which you don't know with certainty.
- **ASA's 2016 statement on p-values** (worth knowing exists, don't need every detail memorized): explicitly warns against exactly these misinterpretations and against using p-value thresholds as the sole basis for decisions. Citing awareness of this shows field literacy.

## 6. Interview Traps

- **Trap #1 (most common)**: "The p-value is the probability the null hypothesis is true." This is the single most-tested misconception in an A/B testing interview. Always be ready to state the correct definition crisply.
- **Trap #2**: "A smaller p-value means a bigger effect." False — as shown above, p-values shrink with both larger effects AND larger sample sizes. A tiny, meaningless effect can have a minuscule p-value if $n$ is large enough.
- **Trap #3**: Treating $p=0.05$ as a magic, universal line rather than a chosen convention that should be calibrated to the cost of false positives for that specific decision.
- **Trap #4**: "Non-significant p-value proves there's no effect." Same fallacy as Chapter 2 — absence of evidence isn't evidence of absence; you may simply be underpowered (Chapter 5).

## 7. L5-Differentiating Talking Points

- Explicitly contrasting $P(\text{data}\mid H_0)$ vs $P(H_0\mid\text{data})$ out loud, ideally with the Bayes' theorem connection ("to get the second one you'd need a prior"), immediately signals you're not reciting a memorized definition.
- Bringing up that p-values conflate effect size and sample size — and that this is *why* production dashboards should always show CI/effect size next to the p-value — is a strong practical-maturity signal.
- Mentioning that at large scale (billions of users), Google-style companies often see "statistically significant" results on nearly everything, making p-value thresholds alone insufficient for ship decisions — this shows awareness of how theory meets real production scale.

## 8. Comprehension Check

1. Write the formal definition of a p-value as a conditional probability, in symbols.
2. Why is "there's a 3% chance the null hypothesis is true" an incorrect interpretation of $p=0.03$?
3. Explain, using the coin-flip example, why a p-value can shrink even when the effect size stays exactly the same.
4. What would you need, in addition to the p-value, to compute $P(H_0 \mid \text{data})$?
5. A stakeholder sees $p=0.048$ on one metric and $p=0.052$ on a nearly identical metric and asks "why did one succeed and one fail?" How do you respond?

---
*Next: Chapter 4 — Confidence Intervals*
