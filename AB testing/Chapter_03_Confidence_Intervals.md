# Chapter 3: Confidence Intervals — Construction & Correct Interpretation
---

## 1. Why CIs Matter (Intuition)

A p-value tells you whether to reject H₀. It does **not** tell you the plausible range of the actual effect size — a p-value of 0.001 is consistent with both a huge effect and a tiny one.

A **confidence interval (CI)** fixes this by giving you a range of plausible values for the true effect — almost always more useful for a ship/no-ship decision than a single yes/no test.

Instead of asking "is the effect exactly zero?" (hypothesis testing), you ask "given my data, what's the range of effect sizes I can't rule out?" This is why practitioners increasingly favor **estimation** (CIs) over pure **testing** (p-values) — a CI carries strictly more information than a p-value alone.

### Layman analogy
Imagine estimating the true average height of every adult in a country, but you can only measure a sample of 1,000 people. Your sample average won't be exactly right — it'll be off by some amount due to random sampling. A CI is your way of saying: "Based on this sample, I'm fairly confident the true average is somewhere in this range."

The 95% doesn't describe your certainty about *this one* range — it describes the *method*. If you repeated the whole experiment (new sample of 1,000 people) 100 times, about 95 of those 100 intervals would capture the true average. Any single interval either contains the truth or it doesn't — you just don't know which, but the method is reliable 95% of the time.

---

## 2. Formal Definition

A confidence interval is a range of values, computed from sample data, designed to contain the true population parameter with a specified long-run frequency — most commonly 95%.

### General construction (Normal-based, using CLT)

$$CI = \bar{X} \pm z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}$$

- $\bar{X}$ = sample mean
- $z_{\alpha/2}$ = critical value from standard Normal (**1.96** for 95% CI, **2.58** for 99%)
- $\sigma/\sqrt{n}$ = standard error (use sample std dev *s* if σ unknown, and switch to a **t-distribution** critical value for small n)

### For proportions (conversion rate)

$$CI = \hat{p} \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

### For a difference in means/proportions (the A/B testing case)

$$\hat{\Delta} \pm z_{\alpha/2} \times SE(\hat{\Delta})$$

where $\hat{\Delta} = \bar{X}_{treatment} - \bar{X}_{control}$, and for a 95% CI, $z_{\alpha/2} = 1.96$.

**Important construction detail**: for CI construction, use the **unpooled** standard error, since you're not assuming H₀ is true — you're estimating the actual gap:

$$SE_{unpooled} = \sqrt{\frac{\hat{p}_0(1-\hat{p}_0)}{n_0} + \frac{\hat{p}_1(1-\hat{p}_1)}{n_1}}$$

This is subtly different from the **pooled SE** used in the Chapter 2-style hypothesis test, which assumes H₀ is true (equal proportions). Using pooled SE when constructing a CI is a classic interview trap.

---

## 3. The Correct Interpretation (Most-Tested Concept)

> **Correct**: "If we repeated this experiment many times, 95% of the confidence intervals constructed this way would contain the true effect."

> **Wrong (the interpretation almost everyone gives)**: "There's a 95% chance the true effect is in this specific interval."

**Why the second one is wrong**: once you've computed a specific interval — say [0.002, 0.014] — the true parameter either is or isn't in that fixed interval; there's no more randomness left. The randomness was in the sampling process, not in the fixed (if unknown) parameter. The 95% is a property of the **procedure** across repeated sampling, not a probability statement about this one realized interval.

A **Bayesian credible interval** *does* let you make the "95% probability" statement directly — but that requires a prior and a different framework entirely. This frequentist-vs-Bayesian distinction is a favorite interview probe.

### Other common misinterpretations
1. "There's a 95% chance the true value is in this interval" — wrong for frequentist CIs (this is the Bayesian credible-interval interpretation).
2. "95% of the data falls in this interval" — wrong; that describes a data range/percentile range, not a CI on a parameter.
3. "If I repeat the experiment, there's a 95% chance I'll get a sample mean in this interval" — wrong; the CI describes uncertainty in estimating the true parameter, not the spread of future sample means.
4. **Overlap asymmetry trap**: Non-overlapping CIs between two groups *does* imply significance (conservative rule) — but overlapping CIs do **not** necessarily mean no significant difference. Overlap can coexist with a significant difference under certain conditions. This asymmetry is a classic trap — see Section 4 for the correct test.

---

## 4. The CI ↔ Hypothesis Test Duality

A CI and a two-sided hypothesis test are mathematically linked:

$$\text{Reject } H_0: \Delta=0 \text{ at level } \alpha \iff 0 \text{ is NOT in the } (1-\alpha)\text{ CI for } \Delta$$

If your 95% CI for the treatment effect is $[0.002, 0.014]$ (doesn't contain 0), you'd also reject H₀ at α=0.05 with p<0.05.

Stating this duality explicitly in an interview is valuable — it shows you see the CI and the hypothesis test as two views of the same underlying math, not two unrelated tools.

**Corollary for the overlap trap above**: the correct way to check significance between two variants is to compute the CI (or p-value) on the **difference** directly — this accounts for the covariance structure correctly — rather than eyeballing whether two separate CIs overlap.

**One-sided nuance**: if your hypothesis test is one-sided, the corresponding CI should be one-sided too, for the duality to hold cleanly. Mismatching them (one-sided test + two-sided CI, or vice versa) is a subtle but real error some dashboards make.

---

## 5. Worked Example (Checkout Redesign)

- Control conversion: $\hat{p}_0 = 0.100$
- Treatment conversion: $\hat{p}_1 = 0.108$
- $\hat{\Delta} = 0.008$
- $n_0 = n_1 = 10{,}000$

Unpooled SE:

$$SE_{unpooled} = \sqrt{\frac{0.09}{10000} + \frac{0.0964}{10000}} \approx 0.00432$$

95% CI:

$$0.008 \pm 1.96 \times 0.00432 = 0.008 \pm 0.00847 = [-0.00047,\ 0.01647]$$

**Interpretation**: We're 95% confident (in the repeated-sampling sense) the true lift is somewhere between -0.05pp and +1.65pp. This interval **contains 0** — consistent with failing to reject H₀ at α=0.05 (confirming the duality from Section 4).

**Why the CI beats the p-value here**: the p-value alone just says "not significant." The CI shows the true effect is very unlikely to be *large and negative* — the plausible range is mostly positive, just not decisively so. That's actionable: it might justify "extend the test for more power" rather than "kill the idea," a nuance a bare p-value hides.

---

## 6. Levers — What Controls CI Width

| Lever | Effect |
|---|---|
| **Sample size (n)** | Width shrinks proportionally to $1/\sqrt{n}$ — quadrupling n only **halves** the width (same nonlinearity as the CLT chapter). Not linear — a common bad intuition to correct in interviews. |
| **Confidence level** | 95% → 99%: interval **widens** (z: 1.96 → 2.58) — trading precision for higher certainty of capture. 95% → 90%: interval **narrows** but increases the chance of missing the true value. |
| **Variance of underlying metric (σ²)** | Noisier metrics → wider CIs for the same n. Variance-reduction techniques (CUPED, stratification) tighten CIs **without needing more data**. |
| **Distribution shape / small-sample correction** | For small n or unknown population variance, use the **t-distribution** instead of z — fatter tails produce wider (more honest) intervals to account for extra uncertainty in estimating σ from the sample. As n grows large (~30+, though the right threshold depends on skew), t and z converge. |

---

## 7. Production Considerations

- **CI width shrinks with $\sqrt{n}$, not n.** When a stakeholder asks "can we just run it twice as long to get a tighter interval," the honest answer involves the square-root law — doubling n only shrinks width by ~29%, not 50%.
- **Always report CIs, not just point estimates, in experiment readouts.** "+2% lift, CI [-8%, +12%]" tells a very different story than "+2%, CI [+1.5%, +2.5%]," even though the point estimate is identical.
- **Match one-sided/two-sided framing** between the test and the CI (see Section 4).

---

## 8. Famous Interview Q&A

**Q: Your 95% CI for lift in conversion rate is [0.5%, 3.5%]. A PM says "there's a 95% chance the true lift is between 0.5% and 3.5%." Is that correct?**
A: Not technically, under the frequentist framework. The 95% describes the long-run reliability of the *method* used to construct the interval — if we repeated the experiment many times, 95% of such intervals would contain the true lift. This specific interval either contains the true value or it doesn't; there's no remaining probability once the data is observed. In practice, many teams use this shorthand because it's operationally close enough for decision-making — but if the interviewer probes on it, the correct framing shows statistical maturity. (A Bayesian credible interval *would* support the "95% probability" interpretation directly.)

**Q: Two variants have CIs [1%, 5%] and [3%, 7%] for lift. They overlap. Does that mean the difference isn't statistically significant?**
A: Not necessarily — a common trap. Overlapping CIs on two separate estimates don't automatically mean the difference between them is non-significant; the correct test is to compute the CI (or p-value) on the *difference* directly, which correctly accounts for the covariance structure. As a rule of thumb, non-overlapping CIs *do* imply significance, but overlapping CIs are inconclusive without directly testing the difference.

**Q: Why does the CI get wider when you raise your confidence level from 95% to 99%?**
A: Because you're demanding the interval capture the true parameter more often across repeated sampling — to be more sure you haven't missed it, you must cast a wider net. Direct precision-vs-confidence tradeoff: $z_{\alpha/2}$ increases from 1.96 to 2.58, directly widening the margin.

**Q: You're running an experiment with only 200 users per arm due to low traffic. Why might a standard z-based CI be a mistake?**
A: With small n, using the sample standard deviation to estimate σ introduces extra uncertainty that the z-distribution doesn't account for — it assumes σ is known exactly. The t-distribution has fatter tails specifically to compensate, producing a more honest (wider) interval. As n grows large (~30+, depending on skew), t and z converge and the distinction stops mattering much.

---

## 9. L5-Differentiating Talking Points

- Correctly stating the CI interpretation on the first try, without prompting, is a strong signal — most candidates get this wrong until corrected.
- Proactively drawing the CI ↔ hypothesis-test duality connects estimation and testing together and shows systemic understanding rather than isolated memorized facts.
- Distinguishing frequentist CIs from Bayesian credible intervals, and knowing when you'd prefer one over the other (credible intervals are more intuitive for stakeholder communication; frequentist CIs are the industry standard for formal experiment readouts) shows breadth.

---

## 10. Comprehension Check (Self-Test)

1. State the correct interpretation of a 95% confidence interval, and explain precisely why the common "95% probability" phrasing is wrong.
2. Prove the CI-hypothesis test duality in words: why does "0 is not in the 95% CI" imply "we reject H₀ at α=0.05"?
3. If you quadruple your sample size, by what factor does your CI width shrink? Why?
4. Why does CI construction use the unpooled standard error while a hypothesis test typically uses the pooled standard error?
5. A stakeholder says "the CI is [-0.05%, +1.65%], so there's a 95% chance the true effect is positive." What's wrong with this statement, and what would you say instead?
6. Two variants' CIs overlap — can you conclude there's no significant difference? What should you check instead?
7. Why might a z-based CI be misleading with only 200 users per arm, and what's the fix?

---
*End of master tutorial — combines confidence interval construction, interpretation, the CI-hypothesis test duality, levers, worked example, and interview Q&A/traps.*


---


The confusing part is the word **"effect."** It just means **the quantity you're trying to measure**.

Depending on the experiment, the effect could be:

* A difference in means (e.g., users spend **5 more minutes** on a new app version).
* A difference in conversion rates (e.g., **+2%** purchases).
* An odds ratio, risk ratio, regression coefficient, etc.

### Example

Suppose you're A/B testing a new checkout page.

* Control conversion rate: **10%**
* Treatment conversion rate: **12%**

The **effect** is:

> **12% − 10% = +2 percentage points**

Now imagine your statistical analysis gives:

* p-value = **0.03**
* 95% CI = **0.3% to 3.7%**

Here's what each tells you:

* **p-value:** If there were actually **no effect** (true difference = 0), seeing data this extreme would happen about 3% of the time. Since 0.03 < 0.05, you reject the null hypothesis.
* **Confidence interval:** Based on your data, plausible values for the **true effect** range from **0.3% to 3.7%**.

Notice how the CI is much more informative:

* It says the effect is probably **positive** (the interval doesn't include 0).
* It also tells you **how big** the improvement might realistically be.

---

### Why the "95%" statement is worded that way

People often want to say:

> "There is a 95% probability the true effect is in this interval."

But that's **not** the frequentist interpretation.

Instead, we say:

> **If we repeated the experiment many times and constructed a 95% confidence interval from each sample, about 95% of those intervals would contain the true effect.**

Why?

In frequentist statistics:

* The **true effect** is treated as a fixed (but unknown) number.
* The **interval** changes from sample to sample because different samples give different estimates.

So the **95%** refers to the **method's long-run success rate**, not to the probability that this particular interval contains the true value.

---

### A visual intuition

Suppose the true effect is actually **+2%**, but you don't know that.

You repeat the experiment 100 times:

```
Experiment 1:   [0.8%, 3.1%]   ✓
Experiment 2:   [1.2%, 2.7%]   ✓
Experiment 3:   [-0.4%, 1.8%]  ✗
Experiment 4:   [0.5%, 3.8%]   ✓
...
```

Each experiment produces a different confidence interval because the data differ.

If you're using a 95% confidence interval procedure, then **about 95 of those 100 intervals will include the true effect (+2%)**, while about 5 will miss it entirely.

---
