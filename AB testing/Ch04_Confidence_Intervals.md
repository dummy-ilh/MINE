# Chapter 4: Confidence Intervals

## 1. Intuition

A p-value tells you whether to reject $H_0$. It does NOT tell you the plausible range of the actual effect size — a p-value of 0.001 is consistent with both a huge effect and a tiny one. A **confidence interval (CI)** fixes this by giving you a range of plausible values for the true effect, which is almost always more useful for a ship/no-ship decision than a single yes/no test.

The intuition: instead of asking "is the effect exactly zero?" (hypothesis testing), you ask "given my data, what's the range of effect sizes I can't rule out?" This reframing is why practitioners increasingly favor "estimation" (CIs) over pure "testing" (p-values) — a CI carries strictly more information than a p-value alone.

## 2. The Formal Definition

For a sample mean difference $\hat{\Delta} = \bar{X}_{treatment} - \bar{X}_{control}$, the $(1-\alpha)$ confidence interval is:

$$\hat{\Delta} \pm z_{\alpha/2} \times SE(\hat{\Delta})$$

For a 95% CI, $z_{\alpha/2} = 1.96$.

**The correct interpretation** (and this is where almost everyone gets it wrong, including working data scientists):

> "If we repeated this experiment many times, 95% of the confidence intervals constructed this way would contain the true effect."

**The interpretation almost everyone gives, which is wrong:**

> "There's a 95% chance the true effect is in this specific interval."

Why is the second one wrong? Once you've computed a specific interval — say [0.002, 0.014] — the true effect either is or isn't in that fixed interval; there's no more randomness left. The 95% is a property of the *procedure* across repeated sampling, not a probability statement about this one interval. (A Bayesian **credible interval** *does* let you make the "95% probability" statement — but that requires a prior and a different framework. This distinction is a favorite interview probe.)

## 3. The CI-Hypothesis Test Duality

A CI and a two-sided hypothesis test are mathematically linked:

$$\text{Reject } H_0: \Delta=0 \text{ at level } \alpha \iff 0 \text{ is NOT in the } (1-\alpha) \text{ CI for } \Delta$$

This means: if your 95% CI for the treatment effect is $[0.002, 0.014]$ (doesn't contain 0), you'd also reject $H_0$ at $\alpha=0.05$ with $p<0.05$. This duality is worth stating explicitly in an interview — it shows you see the CI and the hypothesis test as two views of the same underlying math, not two unrelated tools.

## 4. Worked Example

Continuing the checkout redesign example from Chapter 2:

- Control conversion: $\hat{p}_0 = 0.100$
- Treatment conversion: $\hat{p}_1 = 0.108$
- $\hat{\Delta} = 0.008$

For a CI (unlike the hypothesis test), we use the **unpooled** standard error, since we're not assuming $H_0$ is true anymore — we're estimating the actual difference:

$$SE_{unpooled} = \sqrt{\frac{\hat{p}_0(1-\hat{p}_0)}{n_0} + \frac{\hat{p}_1(1-\hat{p}_1)}{n_1}} = \sqrt{\frac{0.09}{10000} + \frac{0.0964}{10000}} \approx 0.00432$$

**95% CI:**

$$0.008 \pm 1.96 \times 0.00432 = 0.008 \pm 0.00847 = [-0.00047, \ 0.01647]$$

**Interpretation**: We're 95% confident (in the repeated-sampling sense) the true lift is somewhere between -0.05pp and +1.65pp. Note this interval **contains 0** — consistent with our Chapter 2 finding that we failed to reject $H_0$ at $\alpha=0.05$ (confirming the duality above).

**Why the CI is more useful here than the p-value alone**: the p-value just told us "not significant." The CI tells us the true effect is very unlikely to be *large and negative* — the plausible range is mostly positive, just not decisively so. This is actionable: it might justify "let's extend the test for more power" rather than "kill the idea," a nuance a bare p-value hides.

## 5. Production Considerations

- **CI width shrinks with $\sqrt{n}$**, not $n$ — quadrupling sample size only halves the CI width. This matters when a stakeholder asks "can we just run it twice as long to get a tighter interval" — the honest answer involves the square root law.
- **Report CIs, not just point estimates, in every experiment readout.** A treatment showing "+2% lift" with a CI of [-8%, +12%] tells a very different story than "+2%, CI [+1.5%, +2.5%]," even though the point estimate is identical.
- **One-sided vs two-sided CIs**: if your hypothesis test is one-sided (Chapter 8), the corresponding CI should be one-sided too, for the duality to hold cleanly — mismatching them is a subtle but real error some dashboards make.

## 6. Interview Traps

- **Trap #1 (the classic)**: Saying "there's a 95% probability the true value is in this interval." This is the wrong (Bayesian-sounding) interpretation of a frequentist object. Always default to the repeated-sampling definition unless explicitly asked about Bayesian credible intervals.
- **Trap #2**: Not connecting the CI to the hypothesis test — many candidates treat these as separate topics rather than seeing "CI excludes 0" ⟺ "reject $H_0$" as literally the same statement in different clothing.
- **Trap #3**: Forgetting that CI width depends on $\sqrt{n}$, leading to bad intuitions about how much more data is needed to tighten an interval.
- **Trap #4**: Using the pooled SE (from the hypothesis test) when constructing a CI. These are subtly different — pooled SE assumes $H_0$ true (equal proportions); CI construction should use unpooled SE since you're estimating the actual gap, not testing a null.

## 7. L5-Differentiating Talking Points

- Correctly stating the CI interpretation on the first try, without prompting, is a strong signal — most candidates get this wrong until corrected.
- Proactively drawing the CI ↔ hypothesis-test duality connects Chapters 2-4 together and shows systemic understanding rather than three isolated memorized facts.
- Distinguishing frequentist CIs from Bayesian credible intervals, and being able to say when you'd prefer one over the other (e.g., credible intervals are more intuitive for stakeholder communication, frequentist CIs are the industry standard for formal experiment readouts) shows breadth.

## 8. Comprehension Check

1. State the correct interpretation of a 95% confidence interval, and explain precisely why the common "95% probability" phrasing is wrong.
2. Prove the CI-hypothesis test duality in words: why does "0 is not in the 95% CI" imply "we reject $H_0$ at $\alpha=0.05$"?
3. If you quadruple your sample size, by what factor does your CI width shrink? Why?
4. Why does CI construction use the unpooled standard error while the hypothesis test in Chapter 2 used the pooled standard error?
5. A stakeholder says "the CI is [-0.05%, +1.65%], so there's a 95% chance the true effect is positive." What's wrong with this statement, and what would you say instead?

---
*Next: Chapter 5 — Sample Size & Power Analysis*
