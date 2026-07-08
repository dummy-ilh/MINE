# The p-value: definition, intuition, and misinterpretations

## Definition

$$p\text{-value} = P(\text{data at least as extreme as observed} \mid H_0 \text{ true})$$

In plain terms: **assuming the null hypothesis is true, how likely is it that random chance alone would produce data this extreme (or more)?**

- Small p (e.g., 0.003) → data is unlikely under H₀ → evidence against H₀
- Large p (e.g., 0.42) → data is plausible under H₀ → no strong evidence against H₀

**Formulas**

| Test | Formula |
|---|---|
| Right-tailed | $p = P(T \geq t_{obs} \mid H_0)$ |
| Two-tailed | $p = 2 \cdot P(T \geq \lvert t_{obs}\rvert \mid H_0)$ |
| Z-test | $p = 2(1 - \Phi(\lvert z_{obs}\rvert))$ |

**Worked example:** $z_{obs} = -2.0$ → $p = 2 \times P(Z \leq -2.0) = 2 \times 0.0228 = 0.0456$. At α = 0.05, this is below threshold → reject H₀.

**Coin example:** 16 heads out of 20 flips. H₀: coin is fair (p = 0.5).

$$p = P(X \geq 16 \mid X \sim \text{Bin}(20, 0.5)) \approx 0.0059$$

Under a fair coin, 16+ heads happens only ~0.6% of the time — reject H₀.

---

## The six misinterpretations you must be able to correct on the spot

| # | Wrong statement | Why it's wrong | Correct framing |
|---|---|---|---|
| 1 | "p = probability H₀ is true" | Confuses $P(D\mid H_0)$ with $P(H_0\mid D)$ — this reversal requires Bayes' theorem and a prior, which the p-value doesn't supply | p is a statement about the data given H₀, not about H₀ given the data |
| 2 | "1 − p = probability H₁ is true" | Same reversal, applied to the alternative | p says nothing about H₁'s probability |
| 3 | "Non-significant p means no effect" | Failing to reject ≠ proving the null; could just be underpowered | "Absence of evidence isn't evidence of absence" — the effect may be real but too small to detect at this sample size |
| 4 | "Smaller p = bigger effect" | p is driven by sample size as much as effect size | With n = 1M, a trivial effect can still produce p = 0.0001. p measures *evidence*, not *magnitude* |
| 5 | "p = probability of replication" | Replication rate depends on power, effect size, and variability — not a direct function of p | These are entirely separate quantities |
| 6 | "p = 0.05 is a meaningful cliff" | 0.05 is an arbitrary convention, not a natural boundary | p = 0.049 and p = 0.051 represent essentially identical evidence; treat p as a continuum, not a binary pass/fail |

**One-line test for yourself:** if you can't say whether a "wrong" statement is conditioning on H₀ or on the data, you don't yet have this cold — go back to the definition.

---

## Interview Q&A

**Q1: Explain a p-value to a non-technical product manager.**
A: If variant B in an A/B test converts 5% better than A, the p-value answers: "if there were truly no difference between A and B, how often would random noise alone produce a gap this large?" A p-value of 0.03 means that would happen only 3% of the time by chance — unlikely enough that we treat the difference as real.

**Q2: Give an example where a tiny p-value is practically meaningless.**
A: A company with 10M users finds variant B converts 0.001% better, with p = 0.0001. The effect is statistically significant only because the sample is enormous — the actual revenue impact might be trivial. Statistical significance ≠ practical significance; always pair p with effect size.

**Q3: How does the p-value relate to confidence intervals?**
A: They're two views of the same test. Rejecting H₀ at level α (p < α) is equivalent to the null value falling outside the (1−α) confidence interval. If p < 0.05, the 95% CI excludes the H₀ value, and vice versa.

**Q4: What happens to p if you double the sample size, holding the true effect constant?**
A: The test statistic scales with $\sqrt{n}$, so doubling n multiplies |Z| by ≈1.41, pushing p lower. With enough data, any non-zero effect eventually reaches p ≈ 0 — which is exactly why p alone is a poor stopping criterion; you need effect size and domain judgment alongside it.

**Q5: Why did widespread p-value misuse contribute to the replication crisis?**
A: Practices like p-hacking (testing many hypotheses until one clears 0.05), optional stopping (peeking at data and stopping early once significant), and publication bias (only significant results get published) inflate false-positive rates across a field. Modern practice counters this with pre-registration, reporting effect sizes and CIs alongside p, and in some fields (e.g., genomics) using far stricter thresholds like α = 5×10⁻⁸.

**Q6 (curveball): Two studies report p = 0.04 and p = 0.06 on the same effect. Is the first meaningfully more significant?**
A: No — this tests whether the candidate treats 0.05 as a real threshold (misinterpretation #6). Both represent similar evidence strength; a cliff-edge distinction between them is an artifact of the arbitrary α convention, not a meaningful statistical difference.
