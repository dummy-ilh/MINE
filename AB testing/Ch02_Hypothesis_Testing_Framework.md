# Chapter 2: Hypothesis Testing Framework

## 1. Intuition

Every A/B test boils down to a courtroom analogy: the treatment is "innocent until proven guilty." You assume nothing changed (null hypothesis) and only conclude something changed if the evidence is strong enough to make "nothing changed" implausible.

The reason this framework exists at all: random noise alone can make treatment and control means differ even when the treatment does nothing. Hypothesis testing is a formal procedure for deciding whether an observed difference is "real" or "just noise," with known, quantifiable error rates.

The uncomfortable truth interviewers want you to internalize: **you never prove the treatment works.** You only ever quantify how surprising your data would be if it didn't.

## 2. The Formal Framework

**Null hypothesis ($H_0$)**: the treatment has no effect. Formally, in potential-outcomes terms from Chapter 1: $H_0: ATE = 0$, or more generally $H_0: \mu_{treatment} = \mu_{control}$.

**Alternative hypothesis ($H_1$)**: the treatment has an effect. $H_1: \mu_{treatment} \neq \mu_{control}$ (two-sided) or $H_1: \mu_{treatment} > \mu_{control}$ (one-sided).

**The two ways you can be wrong:**

| | $H_0$ is actually True | $H_0$ is actually False |
|---|---|---|
| **You reject $H_0$** | Type I Error (false positive), probability = $\alpha$ | Correct decision (true positive), probability = Power |
| **You fail to reject $H_0$** | Correct decision (true negative) | Type II Error (false negative), probability = $\beta$ |

- **$\alpha$ (significance level)**: the probability you accept, in advance, of declaring a winner when there isn't one. Conventionally 0.05. This is a *policy choice*, not a law of nature — it's the false-positive rate you're willing to tolerate.
- **$\beta$**: probability of missing a real effect. **Power = $1-\beta$**, the probability of correctly detecting a real effect of a given size. Covered in depth in Chapter 5.

**The test statistic and rejection region:** you compute a test statistic (e.g., a t-statistic) from your sample, and compare it to a critical value determined by $\alpha$. If the test statistic falls in the rejection region (i.e., is extreme enough), you reject $H_0$.

$$\text{Reject } H_0 \text{ if } |t| > t_{critical}$$

where $t_{critical}$ is chosen so that, *if $H_0$ were true*, the probability of observing a $|t|$ this extreme is exactly $\alpha$.

## 3. The Logic Chain (this is what people fumble live)

The full logical chain, stated precisely, is:

1. Assume $H_0$ is true.
2. Under that assumption, the sampling distribution of your test statistic is known (e.g., approximately normal/t-distributed by CLT).
3. Ask: "if $H_0$ were really true, how likely is it I'd see a difference this large or larger, just from random sampling noise?"
4. If that probability is very small (below $\alpha$), conclude the data is inconsistent with $H_0$, and reject it.
5. Note carefully: you never assume $H_1$ to test it. You only ever try to discredit $H_0$. This is why it's called **null hypothesis significance testing (NHST)** — the test is entirely about the null.

This asymmetry is the single most commonly misunderstood part of hypothesis testing. You are not measuring the probability that the treatment works. You are measuring how implausible your data would be if it didn't.

## 4. Worked Example

You run an experiment on a checkout page redesign. 10,000 users per arm.

- Control conversion rate: $\hat{p}_0 = 0.100$ (1,000 conversions)
- Treatment conversion rate: $\hat{p}_1 = 0.108$ (1,080 conversions)
- Observed difference: $0.008$ (0.8 percentage points)

**Step 1 — State hypotheses:**
$H_0: p_1 - p_0 = 0$
$H_1: p_1 - p_0 \neq 0$ (two-sided, assuming we don't have a strong prior the change could only help)

**Step 2 — Compute the standard error under $H_0$** (pooled since under $H_0$ both are the same population):

$$\hat{p}_{pooled} = \frac{1000+1080}{20000} = 0.104$$

$$SE = \sqrt{\hat{p}_{pooled}(1-\hat{p}_{pooled})\left(\frac{1}{10000}+\frac{1}{10000}\right)} = \sqrt{0.104 \times 0.896 \times 0.0002} \approx 0.00431$$

**Step 3 — Compute the z-statistic:**

$$z = \frac{0.008}{0.00431} \approx 1.86$$

**Step 4 — Compare to critical value:** for $\alpha = 0.05$ two-sided, $z_{critical} = 1.96$.

Since $1.86 < 1.96$, we **fail to reject $H_0$.** The result is not statistically significant at the 5% level, even though the treatment numerically outperformed control by 8%.

**This is the exact moment where PMs push back** — "but treatment did better, why can't we ship it?" The answer: an 0.8pp gap with this sample size is not distinguishable from noise at our chosen confidence level. It's not that nothing happened; it's that we don't have enough evidence yet to rule out "nothing happened."

## 5. Production Considerations

- **$\alpha$ is a business decision, not a statistical law.** A company might justifiably use $\alpha=0.10$ for low-stakes UI experiments (faster iteration, willing to accept more false positives) and $\alpha=0.01$ for pricing/monetization changes where false positives are expensive to roll back.
- **Statistical significance ≠ practical significance.** With enough sample size, you can detect a statistically significant but practically meaningless 0.01% conversion lift. Always report effect size and confidence interval alongside the p-value, not the p-value alone.
- **One test per decision, ideally pre-registered.** If you run the significance test, look at the result, then decide to test a different metric or sub-segment, you've implicitly run multiple tests and inflated your true false-positive rate (ties to Chapter 14, Multiple Testing).

## 6. Interview Traps

- **Trap**: Saying "we fail to reject $H_0$, so we accept $H_0$ / prove there's no effect." This is wrong — failing to reject is not evidence of no effect, it's *insufficient evidence to conclude there is one*. The absence of proof is not proof of absence.
- **Trap**: Not being able to state, precisely, what $\alpha$ means. It is NOT "the probability our result is wrong." It's "the probability of rejecting $H_0$ given $H_0$ is actually true" — a statement about the long-run behavior of the *procedure*, not about this specific result.
- **Trap**: Forgetting that $H_0$ and $H_1$ must be mutually exclusive and collectively exhaustive, and that the choice of one- vs two-sided (Chapter 8) has to be made *before* seeing the data — choosing it after peeking at the direction of the effect is a subtle form of p-hacking.

## 7. L5-Differentiating Talking Points

- Framing the whole test as "we're trying to discredit the null, not prove the alternative" in your own words, unprompted, signals genuine understanding vs. memorized procedure.
- Volunteering that $\alpha$ and the choice of one-sided/two-sided are *policy decisions tied to business cost of false positives/negatives* — not fixed conventions — is exactly the kind of judgment L5 interviewers are listening for.
- Being fluent enough to derive the z-statistic by hand (as above) without hesitating shows the difference between "I've used A/B testing tools" and "I understand what the tools are doing."

## 8. Comprehension Check

1. In your own words, what does a p-value of 0.03 actually mean? (Careful — full answer requires Chapter 3.)
2. Why is "fail to reject $H_0$" not the same as "prove $H_0$ is true"?
3. In the worked example, if we'd used $\alpha = 0.10$ instead of 0.05, would we have reached a different conclusion? Compute the answer.
4. Why must the choice between one-sided and two-sided testing be made before looking at the data?
5. A PM says "the p-value was 0.06, so basically nothing happened." What's wrong with this statement?

---
*Next: Chapter 3 — P-values*
