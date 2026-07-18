# Chapter 3: Confidence Intervals — Construction & Correct Interpretation

## 1. Definition

A confidence interval (CI) is a range of values, computed from sample data, that is designed to contain the true population parameter with a specified long-run frequency — most commonly 95%. A 95% CI does **not** mean "there's a 95% probability the true value is in this specific interval." It means: if you repeated this sampling process many times, 95% of the intervals constructed this way would contain the true parameter.

## 2. Layman Explanation

Imagine you're trying to guess the true average height of every adult in a country, but you can only measure a sample of 1,000 people. Your sample average won't be exactly right — it'll be off by some amount due to random sampling. A confidence interval is your way of saying: "Based on this sample, I'm fairly confident the true average is somewhere in this range."

The 95% doesn't describe your certainty about *this one* range — it describes the *method*. If you repeated the whole experiment (new sample of 1,000 people) 100 times, about 95 of those 100 intervals would capture the true average. Any single interval either contains the truth or it doesn't — you just don't know which, but the method is reliable 95% of the time.

## 3. Formal Explanation

**Construction (Normal-based, using CLT):**

CI = X̄ ± z_(α/2) × (σ / √n)

- X̄ = sample mean
- z_(α/2) = critical value from standard Normal (1.96 for 95% CI)
- σ/√n = standard error (use sample std dev *s* if σ unknown, and switch to a t-distribution critical value for small n)

**For proportions (conversion rate):**

CI = p̂ ± z_(α/2) × √(p̂(1-p̂)/n)

**Correct interpretation:**
P(interval contains true parameter) is a statement about the *procedure*, not about a fixed realized interval. Once you've computed the actual numbers (e.g., [0.031, 0.045]), the true parameter either is or isn't in that range — there's no more randomness left; the randomness was in the sampling, not in the fixed parameter.

**Common misinterpretations (frequently tested):**
1. "There's a 95% chance the true value is in this interval" — wrong for frequentist CIs (this is actually the Bayesian *credible interval* interpretation).
2. "95% of the data falls in this interval" — wrong; that's describing a data range (like a percentile range), not a CI on a parameter.
3. "If I repeat the experiment, there's a 95% chance I'll get a mean in this interval" — wrong; the CI describes uncertainty in estimating μ, not the spread of future sample means.
4. Non-overlapping CIs between two groups always implies significance — actually **true** as a conservative rule, but overlapping CIs do *not* necessarily mean no significant difference (overlap can still coexist with a significant difference under certain conditions) — this asymmetry is a classic trap.

## 4. Levers — What Controls It, What Moves It

**Sample size (n)**
- CI width shrinks proportionally to 1/√n (same nonlinearity as Chapter 2 — 4x the data halves the width).

**Confidence level**
- Raising confidence (95% → 99%) widens the interval (z goes from 1.96 to 2.58) — you trade precision for higher certainty of capture.
- Lowering confidence (95% → 90%) narrows the interval but increases the chance of missing the true value.

**Variance of underlying metric (σ²)**
- Noisier metrics (high variance) produce wider CIs for the same n.
- Variance reduction techniques (CUPED, stratification) tighten CIs without needing more data — same lever discussed in CLT chapter, this is where it pays off practically.

**Distribution shape / small-sample corrections**
- For small n or unknown population variance, use the t-distribution instead of z — it has fatter tails, producing wider (more honest) intervals to account for extra uncertainty in estimating σ from the sample.

## 5. Famous Q&A (Google / Apple style)

**Q: Your 95% CI for lift in conversion rate is [0.5%, 3.5%]. A PM says "there's a 95% chance the true lift is between 0.5% and 3.5%." Is that correct?**
A: Not technically, under the frequentist framework. The 95% describes the long-run reliability of the *method* used to construct the interval — if we repeated the experiment many times, 95% of such intervals would contain the true lift. This specific interval either contains the true value or it doesn't; there's no remaining probability once the data is observed. That said, in practice, many teams use this shorthand because it's operationally close enough for decision-making — but it's worth knowing the distinction is real, and if the interviewer probes on it, the correct framing shows statistical maturity. (Note: a Bayesian credible interval *would* support the "95% probability" interpretation directly.)

**Q: Two variants have CIs [1%, 5%] and [3%, 7%] for lift. They overlap. Does that mean the difference isn't statistically significant?**
A: Not necessarily — this is a common trap. Overlapping CIs on two separate estimates don't automatically mean the difference between them is non-significant; the correct test is to compute the CI (or p-value) on the *difference* directly, which accounts for the covariance structure correctly. As a rule of thumb, non-overlapping CIs *do* imply significance, but overlapping CIs are inconclusive without directly testing the difference.

**Q: Why does the CI get wider when you raise your confidence level from 95% to 99%?**
A: Because you're demanding the interval capture the true parameter more often across repeated sampling — to be more sure you haven't missed it, you must cast a wider net. This is a direct precision-vs-confidence tradeoff: z_(α/2) increases from 1.96 to 2.58, directly widening the margin.

**Q: You're running an experiment with only 200 users per arm due to low traffic. Why might using a standard z-based CI be a mistake?**
A: With small n, using the sample standard deviation to estimate σ introduces extra uncertainty that the z-distribution doesn't account for — it assumes σ is known exactly. The t-distribution has fatter tails specifically to compensate for this added uncertainty in small samples, producing a more honest (wider) interval. As n grows large (~30+, though the right threshold depends on skew), t and z converge and the distinction stops mattering much.

---
*Next: Chapter 4 — Hypothesis Testing Basics: Type I/II errors, p-values, and power.*
