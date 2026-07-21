# 🔮 17 — Bayesian Hypothesis Testing & Bayesian A/B Testing

> *"The Bayesian says: I had beliefs before seeing data. The data updated them. Here's my new belief. The Frequentist says: Assuming H₀ is true, this data is unlikely."*

---

## Why Bayesian? The Frequentist Limitation

Frequentist testing answers: **"How surprising is my data, assuming H₀ is true?"**

What most people *actually want* to know: **"Given this data, how likely is H₁ to be true?"**

That's a Bayesian question. Frequentist statistics literally cannot answer it without a prior — yet interviewers catch candidates making this mistake constantly (see Section 15, Trap #2).

---

## Bayes' Theorem — The Foundation

$$P(H_1 \mid \text{data}) = \frac{P(\text{data} \mid H_1) \cdot P(H_1)}{P(\text{data})}$$

In the hypothesis testing context:

| Term | Name | Meaning |
|---|---|---|
| $P(H_1)$ | **Prior** | Your belief in H₁ *before* seeing data |
| $P(\text{data} \mid H_1)$ | **Likelihood** | How probable is the data under H₁ |
| $P(H_1 \mid \text{data})$ | **Posterior** | Your updated belief *after* seeing data |
| $P(\text{data})$ | **Marginal likelihood** | Normalizing constant |

---

## Bayesian vs. Frequentist — Core Comparison

| Aspect | Frequentist | Bayesian |
|---|---|---|
| Parameters | Fixed, unknown constants | Random variables with distributions |
| Probability | Long-run frequency | Degree of belief |
| Output | p-value, CI | Posterior distribution, credible interval |
| Prior knowledge | Ignored | Explicitly incorporated |
| Sample size | Fixed in advance | Can stop when posterior is conclusive |
| Interpretation | "Data is surprising under H₀" | "Here's the updated probability of each hypothesis" |
| Multiple peeking | Inflates Type I error | Valid — posterior updates continuously |

---

## Bayes Factor

The **Bayes Factor** $BF_{10}$ quantifies how much the data shifts the odds in favor of H₁ over H₀:

$$BF_{10} = \frac{P(\text{data} \mid H_1)}{P(\text{data} \mid H_0)}$$

### Interpretation Scale (Jeffreys Scale)

| $BF_{10}$ | Interpretation |
|---|---|
| 1 – 3 | Anecdotal evidence for H₁ |
| 3 – 10 | **Moderate** evidence for H₁ |
| 10 – 30 | **Strong** evidence for H₁ |
| 30 – 100 | **Very strong** evidence for H₁ |
| > 100 | **Extreme** evidence for H₁ |
| < 1 | Evidence for H₀ |
| 1/3 – 1 | Anecdotal evidence for H₀ |

### Bayes Factor vs. p-value

$$\text{Posterior Odds} = BF_{10} \times \text{Prior Odds}$$

$$\frac{P(H_1 \mid \text{data})}{P(H_0 \mid \text{data})} = BF_{10} \times \frac{P(H_1)}{P(H_0)}$$

**Key insight:** A small p-value does not imply a large Bayes Factor. They measure different things.

---

## Bayesian A/B Testing

### Setup for Conversion Rate Test

Let $p_A$ and $p_B$ be the true conversion rates. We model them with Beta distributions (the conjugate prior for Bernoulli data):

$$p_A \sim \text{Beta}(\alpha_A, \beta_A)$$
$$p_B \sim \text{Beta}(\alpha_B, \beta_B)$$

### Prior Choice

**Non-informative (uniform) prior:** $\text{Beta}(1, 1)$ — all values equally likely.

**Informative prior:** If you know baseline is ~5% conversion, use $\text{Beta}(5, 95)$ — encodes prior belief centered around 5%.

### Updating with Data (Conjugate Update)

After observing $s$ successes (conversions) out of $n$ trials:

$$p \mid \text{data} \sim \text{Beta}(\alpha + s, \beta + n - s)$$

This is the **posterior distribution** — your updated belief about $p$ after seeing data.

### Example

> **Control (A):** 500 visitors, 25 conversions → $\hat{p}_A = 5\%$
> **Variant (B):** 500 visitors, 35 conversions → $\hat{p}_B = 7\%$

Starting with $\text{Beta}(1,1)$ priors:

$$p_A \mid \text{data} \sim \text{Beta}(1+25, 1+475) = \text{Beta}(26, 476)$$
$$p_B \mid \text{data} \sim \text{Beta}(1+35, 1+465) = \text{Beta}(36, 466)$$

**Posterior means:** $\mu_A = 26/502 \approx 5.18\%$, $\mu_B = 36/502 \approx 7.17\%$

---

### Key Bayesian Metrics

#### 1. Probability that B is Better

$$P(p_B > p_A \mid \text{data})$$

This is the probability that variant B truly has a higher conversion rate — **the exact thing a PM wants to know**.

Computed by Monte Carlo simulation:

```python
import numpy as np

alpha_A, beta_A = 26, 476
alpha_B, beta_B = 36, 466
N = 100_000

samples_A = np.random.beta(alpha_A, beta_A, N)
samples_B = np.random.beta(alpha_B, beta_B, N)

prob_B_better = np.mean(samples_B > samples_A)
print(f"P(B > A) = {prob_B_better:.3f}")  # ~0.89
```

**Result:** ~89% probability that B is genuinely better.

#### 2. Expected Loss

How much conversion rate are you *expected to lose* if you ship the wrong variant?

$$\text{Loss}(A) = E[\max(p_B - p_A, 0)]$$
$$\text{Loss}(B) = E[\max(p_A - p_B, 0)]$$

```python
loss_choosing_A = np.mean(np.maximum(samples_B - samples_A, 0))
loss_choosing_B = np.mean(np.maximum(samples_A - samples_B, 0))
# Ship B if loss_choosing_A > threshold (e.g., 0.001)
```

This allows **risk-based decisions** — ship when the expected loss of the wrong choice is below a business threshold.

#### 3. Credible Interval (Highest Density Interval)

The **95% credible interval** for $p_B - p_A$:

```python
diff_samples = samples_B - samples_A
ci_lower = np.percentile(diff_samples, 2.5)
ci_upper = np.percentile(diff_samples, 97.5)
```

**Correct interpretation (unlike frequentist CI):** "There is a 95% probability the true difference is between ci_lower and ci_upper." ✅

---

## Credible Interval vs. Confidence Interval

| | Confidence Interval | Credible Interval |
|---|---|---|
| Framework | Frequentist | Bayesian |
| Statement | 95% of such intervals contain true value | 95% probability true value is in this interval |
| Probability on parameter? | ❌ No | ✅ Yes |
| Requires prior? | ❌ No | ✅ Yes |
| Interpretation | Procedure-based | Belief-based |

---

## When to Use Bayesian A/B Testing

✅ **Use Bayesian when:**
- You want probabilistic statements: "P(B is better) = 87%"
- You want to incorporate prior knowledge
- You need to peek at data continuously without inflating errors
- You want expected loss as a decision criterion
- You have small samples (priors help stabilize estimates)

✅ **Use Frequentist when:**
- Regulatory requirements mandate p-values (clinical trials, FDA)
- Strict Type I error control is required
- Stakeholders are familiar with p-values
- No sensible prior exists

---

## The Prior Sensitivity Problem

A key criticism of Bayesian methods: results can depend heavily on prior choice.

| Prior | P(B>A) |
|---|---|
| Beta(1,1) — flat | 89.1% |
| Beta(5,95) — informative at 5% | 85.3% |
| Beta(0.5,0.5) — Jeffreys prior | 89.4% |

With sufficient data, the prior matters less — the data dominates. With small samples, prior choice is critical and should be disclosed.

---

## Visual Intuition

```
Prior:           Posterior (A):      Posterior (B):
Beta(1,1)        Beta(26,476)        Beta(36,466)
───────────       ────────────        ────────────
flat              peaked ~5%          peaked ~7%
(uniform)         narrow bell         narrow bell
                  
                  P(B > A) = shaded overlap area
```

---

## 💬 FAANG Interview Questions & Answers

**Q1: What is the main advantage of Bayesian A/B testing over frequentist?**

> A: Three main advantages: (1) You get a direct, intuitive output — "87% probability B is better" — rather than a p-value that requires careful interpretation. (2) You can continuously monitor results and stop when you're sufficiently confident, without the Type I inflation of frequentist peeking. (3) You can incorporate prior knowledge about baseline rates, which helps with small samples. The tradeoff is that you must choose a prior, and results can be prior-sensitive with small data.

**Q2: How do you explain a Bayesian A/B result to a non-technical PM?**

> A: "We ran an experiment where variant B showed 7% conversion vs. 5% for A. Based on the data, we estimate there's an 87% chance B is genuinely better than A. If we're wrong and ship B when A is better, we'd expect to lose about 0.3% in conversion on average — which is within our acceptable risk threshold. I recommend shipping B."

**Q3: What's a Bayes Factor and when would you use it?**

> A: The Bayes Factor quantifies how much more likely your data is under H₁ vs. H₀. A BF of 15 means the data is 15 times more probable if the effect is real than if it isn't. You'd use it when you want a model-comparison metric that, unlike p-values, also allows you to accumulate evidence *for* H₀. A BF < 1/10 gives strong evidence H₀ is true — frequentist tests can't provide this.

**Q4: Booking.com uses Bayesian testing. What problem does this solve?**

> A: Booking.com runs thousands of tests simultaneously on short booking windows. Frequentist testing with fixed sample sizes doesn't fit their need for fast, continuous decisions. Bayesian testing lets them: (1) stop early when evidence is conclusive, (2) use accumulated site knowledge as priors, (3) make risk-based shipping decisions using expected loss, and (4) handle the multiple-testing burden more naturally through hierarchical Bayesian models.

---

## 🚨 Common Pitfalls

1. **Using a flat prior when you have historical data** — wastes information.
2. **Claiming a high posterior probability without stating the prior** — the prior drives the posterior with small samples.
3. **Confusing P(B > A | data) with P(data | B > A)** — still a conditional probability flip.
4. **Not reporting expected loss** — "P(B > A) = 95%" sounds great, but if the expected gain is 0.01%, it might not be worth shipping.
5. **Treating Bayesian CIs exactly like frequentist CIs** — they have different philosophical interpretations even if numerically similar.

---

*← [16 — Cheat Sheet](16_cheatsheet_faang_questions.md) | [18 — CUPED →](18_cuped_variance_reduction.md)*
