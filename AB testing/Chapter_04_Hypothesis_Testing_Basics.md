# Chapter 4: Hypothesis Testing Basics — Type I/II Errors, P-Values, Power

> *"The null hypothesis is never proven or established, but is possibly disproved, in the course of experimentation."* — Karl Popper

---

## 1. What Is Hypothesis Testing? (Intuition)

Hypothesis testing is a formal procedure for deciding whether observed data provides enough evidence to reject a default assumption (the null hypothesis, H₀) in favor of an alternative (H₁).

Every A/B test boils down to a **courtroom analogy**: the treatment is "innocent until proven guilty."

- **H₀ (null hypothesis)**: the defendant is innocent — the treatment has no effect. The default, "nothing interesting is happening" assumption.
- **H₁ (alternative hypothesis)**: the defendant is guilty — the treatment has an effect.
- You need evidence *beyond reasonable doubt* (a statistically significant result) to "convict" — i.e., reject H₀.
- You can only **fail to convict** — you can't prove innocence. Likewise, you can only *fail to reject* H₀, never *accept* it.

**Why this framework exists at all**: random noise alone can make treatment and control means differ even when the treatment does nothing. Hypothesis testing formally decides whether an observed difference is "real" or "just noise," with known, quantifiable error rates.

**The uncomfortable truth interviewers want you to internalize**: you never prove the treatment works. You only ever quantify how surprising your data would be if it didn't. A low p-value doesn't mean "there's only a 5% chance the null is true" — it means "if the null were true, data this extreme would only show up 5% of the time." That distinction trips up almost everyone, including PMs and even some data scientists.

### The scientific process, mapped

```
Real World Question
       ↓
Formulate H₀ and H₁
       ↓
Collect Sample Data
       ↓
Compute Test Statistic
       ↓
Compare to Null Distribution
       ↓
Make Decision (Reject / Fail to Reject H₀)
       ↓
Draw Conclusion in Context
```

---

## 2. Core Definitions

- **Null hypothesis (H₀)**: the "no effect" baseline — e.g., the new feature has no impact on conversion rate. Represents no effect, no difference, no relationship, status quo.
- **Alternative hypothesis (H₁)**: the effect exists — e.g., the new feature changes conversion rate. What you're trying to find evidence *for* — the research claim.
- **Type I error (α)**: rejecting H₀ when it's actually true — a false positive.
- **Type II error (β)**: failing to reject H₀ when H₁ is actually true — a false negative.
- **P-value**: the probability of observing data at least as extreme as what you got, *assuming H₀ is true*.
- **Power (1-β)**: the probability of correctly detecting a true effect when one exists.

### Mathematical form

$$H_0: \theta = \theta_0$$

where $\theta$ is the population parameter of interest (mean, proportion, variance, etc.) and $\theta_0$ is a specific claimed value.

| Context | H₀ |
|---|---|
| One-sample mean | $H_0: \mu = \mu_0$ |
| Two-sample comparison | $H_0: \mu_1 = \mu_2$ |
| Proportion | $H_0: p = p_0$ |
| Correlation | $H_0: \rho = 0$ |
| Regression coefficient | $H_0: \beta_1 = 0$ |

In A/B testing terms (potential-outcomes framing): $H_0: ATE = 0$, or more generally $H_0: \mu_{treatment} = \mu_{control}$. $H_1: \mu_{treatment} \neq \mu_{control}$ (two-sided) or $H_1: \mu_{treatment} > \mu_{control}$ (one-sided).

### Real-world example H₀ statements
- "The new drug has **no effect** on blood pressure compared to placebo."
- "Clicking the blue button **does not change** conversion rate vs. the green button."
- "Customer age is **not correlated** with purchase value."

### The courtroom analogy again — layman explanation
- **Type I error** = convicting an innocent person (you say the feature works, but it actually doesn't — a false alarm).
- **Type II error** = letting a guilty person go free (the feature actually works, but your test wasn't convincing enough to catch it).
- **P-value** = "if this person were truly innocent, how surprising is this evidence?" A very low p-value means the evidence would be very unusual if H₀ were true — so you doubt H₀.
- **Power** = how good your trial is at catching real guilt when it exists. A trial with low power might let real winners (features that actually help) slip through as "not significant."

---

## 3. The Decision Framework (Type I / Type II Errors)

| | H₀ is actually True | H₀ is actually False |
|---|---|---|
| **Reject H₀** | Type I Error (false positive), probability = α | Correct decision (true positive), probability = Power |
| **Fail to reject H₀** | Correct decision (true negative), probability = 1-α | Type II Error (false negative), probability = β |

- **α (significance level)**: the probability you accept, *in advance*, of declaring a winner when there isn't one. Conventionally 0.05. This is a **policy choice**, not a law of nature — the false-positive rate you're willing to tolerate.
- **β**: probability of missing a real effect. **Power = 1−β**, the probability of correctly detecting a real effect of a given size.

---

## 4. Why We Test H₀, Not H₁ Directly (The Deep-Dive Interviewers Love)

**The question**: "Why do we test H₀ and try to disprove it, instead of directly testing whether H₁ is true?"

This is the question that separates someone who memorized the framework from someone who understands *why* it's built this way. Three layers of answer:

### Layer 1 — You need a fully specified distribution to compute anything
A p-value is $P(\text{data this extreme} \mid \text{hypothesis})$. To compute that probability, the hypothesis needs to pin down **one exact distribution** for your test statistic.

- $H_0: \mu = 50$ is a **simple/point hypothesis** — it names one exact value, giving one exact sampling distribution (e.g., $N(50, \sigma^2/n)$). You can calculate exact probabilities against it.
- $H_1: \mu \neq 50$ is a **composite hypothesis** — true for infinitely many values (50.001, 51, 60, 1000...), each with a *different* sampling distribution. There is no single "the H₁ distribution" to compute a probability against.

You cannot compute $P(\text{data} \mid H_1)$ in the two-tailed case — there's no one number to condition on. You *can* always compute $P(\text{data} \mid H_0)$. That's a mathematical necessity, not a philosophical preference — the entire p-value machinery only works because H₀ is precise.

### Layer 2 — It mirrors proof by contradiction
In math, to prove statement A, you often assume ¬A and derive a contradiction. Hypothesis testing does the same thing epistemically: assume the "boring" claim (no effect) is true, and check whether your data would be a bizarre coincidence under that assumption. If the data would be *too* bizarre (p < α), the boring assumption looks untenable — you reject it. You never directly "prove" your research claim; you only make the null look implausible enough to abandon.

### Layer 3 — Falsifiability (Popper): science disproves, it doesn't confirm
You can never collect *enough* evidence to positively confirm a universal claim like "this drug works" — there's always a next patient who might break the pattern. But a *single* disconfirming data point can, in principle, break a precise claim. This is why the entire frequentist apparatus is oriented around **falsifying a precise claim (H₀)** rather than **confirming a vague one (H₁)**. It's logically cheaper and more rigorous to disprove than to prove.

> **One-liner for the interview**: *"We test H₀ because it's the only hypothesis precise enough to have a computable sampling distribution — H₁ is typically composite, so there's no single distribution to test it against directly."*

**Bonus depth (shows range)**: This is a distinctly *frequentist* framing. Bayesian hypothesis testing flips this entirely — it computes $P(H_1 \mid \text{data})$ directly via Bayes' rule, using a prior over H₁. If an interviewer pushes on "why not just compute the probability H1 is true," this is your opening to mention Bayesian A/B testing does exactly that, at the cost of needing a defensible prior.

---

## 5. The Logic Chain (Stated Precisely)

1. Assume $H_0$ is true.
2. Under that assumption, the sampling distribution of your test statistic is known (e.g., approximately normal/t-distributed by CLT).
3. Ask: "if $H_0$ were really true, how likely is it I'd see a difference this large or larger, just from random sampling noise?"
4. If that probability is very small (below α), conclude the data is inconsistent with $H_0$, and reject it.
5. **Note carefully**: you never assume $H_1$ to test it. You only ever try to discredit $H_0$. This is why it's called **null hypothesis significance testing (NHST)** — the test is entirely about the null.

This asymmetry is the single most commonly misunderstood part of hypothesis testing. You are not measuring the probability that the treatment works. You are measuring how implausible your data would be if it didn't.

### ⚠️ Critical nuance: H₀ is falsifiable, not proven
You **never prove H₀ is true**. Failure to reject simply means *you don't have enough evidence to disprove it* — ties directly to the Layer 3 falsifiability point above.

---

## 6. Test Statistics & Rejection Regions

Every hypothesis test computes a **test statistic** of this general form:

```
Test statistic = (Observed value − Expected value under H₀) / Standard Error

                       signal
              =       ────────
                        noise
```

A large test statistic means your observation is many standard errors away from what H₀ predicts — strong evidence against H₀.

$$\text{Reject } H_0 \text{ if } |t| > t_{critical}$$

where $t_{critical}$ is chosen so that, *if $H_0$ were true*, the probability of observing a $|t|$ this extreme is exactly α.

### Specific formulas by test

```
z-test (large n or σ known):     z = (X̄ − μ₀) / (σ / √n)

t-test (small n, σ unknown):     t = (X̄ − μ₀) / (s / √n)

Proportion z-test (A/B):         z = (p̂₁ − p̂₂) / √[p̂(1−p̂)(1/n₁ + 1/n₂)]

Chi-squared test:                 χ² = Σ (Observed − Expected)² / Expected
```

### Decision rule

```
Compute test statistic  →  find p-value  →  compare to α

If  p ≤ α   →  Reject H₀    ("statistically significant")
If  p > α   →  Fail to reject H₀  ("insufficient evidence")
```

| α | Confidence | Used when |
|---|-----------|-----------|
| 0.05 | 95% | Most standard experiments |
| 0.01 | 99% | Higher-stakes decisions |
| 0.001 | 99.9% | Medical, safety-critical |
| 0.1 | 90% | Exploratory research |

---

## 7. One-Tailed vs. Two-Tailed Tests

### Two-tailed test
**Use when**: you care about deviation in *either direction*.

$$H_0: \mu = 50 \quad \text{vs} \quad H_1: \mu \neq 50$$

Rejection region: both tails. $\text{Reject } H_0 \text{ if } |Z| > z_{\alpha/2}$

**Example**: testing whether a coin is *biased at all* (could be heads- or tails-biased).
**Alpha split**: each tail gets α/2. If α = 0.05, critical values are ±1.96.

### Right-tailed (upper-tailed) test
**Use when**: you only care if the parameter is *greater than* the null value.

$$H_0: \mu \leq \mu_0 \quad \text{vs} \quad H_1: \mu > \mu_0$$

Rejection region: right tail only. $\text{Reject } H_0 \text{ if } Z > z_{\alpha}$

**Example**: does the new drug *increase* mean survival time?

### Left-tailed (lower-tailed) test
**Use when**: you only care if the parameter is *less than* the null value.

$$H_0: \mu \geq \mu_0 \quad \text{vs} \quad H_1: \mu < \mu_0$$

Rejection region: left tail only. $\text{Reject } H_0 \text{ if } Z < -z_{\alpha}$

**Example**: is the new manufacturing process *reducing* defect rate?

### Decision guide

| Situation | Test Type |
|---|---|
| "Is there any difference?" | Two-tailed |
| "Is A better than B?" | One-tailed |
| "Did the metric improve?" | One-tailed |
| "Did anything change?" | Two-tailed |
| A/B test: "Does variant beat control?" | One-tailed (directional) |
| A/B test: "Is there any effect?" | Two-tailed |

### ⚠️ Pitfall: choosing tails after seeing data
**Never** switch from two-tailed to one-tailed after seeing the direction of the result — this is a form of p-hacking. The choice of one- vs. two-sided must be made *before* seeing the data.

> **Rule of thumb**: when in doubt, use two-tailed. It's more conservative and more defensible.

### One-sided power tradeoff
A one-sided test (testing only for improvement, not degradation) has **more power** to detect an effect in the specified direction for the same α, because the entire rejection region is on one side. But it comes at the cost of being **unable to detect an effect in the opposite direction** — a risky choice if a feature could plausibly hurt the metric.

---

## 8. Worked Example (Checkout Redesign)

10,000 users per arm.

- Control conversion rate: $\hat{p}_0 = 0.100$ (1,000 conversions)
- Treatment conversion rate: $\hat{p}_1 = 0.108$ (1,080 conversions)
- Observed difference: 0.008 (0.8 percentage points)

**Step 1 — State hypotheses:**
$H_0: p_1 - p_0 = 0$; $H_1: p_1 - p_0 \neq 0$ (two-sided, assuming no strong prior the change could only help)

**Step 2 — Compute the standard error under H₀** (pooled, since under H₀ both arms are the same population):

$$\hat{p}_{pooled} = \frac{1000+1080}{20000} = 0.104$$

$$SE = \sqrt{\hat{p}_{pooled}(1-\hat{p}_{pooled})\left(\frac{1}{10000}+\frac{1}{10000}\right)} = \sqrt{0.104 \times 0.896 \times 0.0002} \approx 0.00431$$

**Step 3 — Compute the z-statistic:**

$$z = \frac{0.008}{0.00431} \approx 1.86$$

**Step 4 — Compare to critical value:** for α = 0.05 two-sided, $z_{critical} = 1.96$.

Since $1.86 < 1.96$, we **fail to reject H₀**. The result is not statistically significant at the 5% level, even though treatment numerically outperformed control by 8% relative.

**This is the exact moment where PMs push back**: "but treatment did better, why can't we ship it?" The answer: an 0.8pp gap with this sample size is not distinguishable from noise at our chosen confidence level. It's not that nothing happened; it's that we don't have enough evidence yet to rule out "nothing happened."

*(Note: this pooled-SE hypothesis test is distinct from the unpooled-SE confidence-interval construction on the same data — see the CI master tutorial for that side of the duality.)*

---

## 9. Levers — What Controls Type I/II Errors & Power

**α (significance threshold)**
- Lowering α (e.g., 0.05 → 0.01) reduces Type I error risk but increases Type II error risk (lower power) for the same sample size — a direct tradeoff.
- Google/Apple-scale companies often use stricter α for high-stakes launches (e.g., pricing changes) and looser thresholds for low-risk UI tweaks.
- α is a **business decision, not a statistical law** — it's the false-positive rate you're willing to tolerate, chosen in advance.

**Sample size (n)**
- Larger n increases power directly — more data means real effects are easier to distinguish from noise. This is the primary lever teams pull when a test is "underpowered."

**Effect size**
- Larger true effects are inherently easier to detect. Not something you control, but it affects how you set your Minimum Detectable Effect (MDE) when planning sample size — chasing tiny effects requires disproportionately more data.

**Variance (σ²)**
- Lower variance (via CUPED, stratification, better metric definitions) increases power without needing more users.

**One-sided vs. two-sided tests**
- See Section 7 — one-sided gains power in the specified direction at the cost of blindness to the opposite direction.

**Key relationship (z-test on means)**: Power is a function of $(\text{effect size} \times \sqrt{n}) / \sigma$, compared against the critical value at α. This is why sample-size calculations are really "solve for n given a target power, effect size, and variance."

---

## 10. Common Conceptual Traps

1. **p-value ≠ P(H₀ is true | data).** That's a Bayesian posterior requiring a prior; frequentist p-values don't provide this. p-value = $P(\text{data} \mid H_0)$, not $P(H_0 \mid \text{data})$.
2. **Failing to reject H₀ does NOT prove H₀ is true.** It just means insufficient evidence to reject it — absence of evidence ≠ evidence of absence. "We fail to reject H₀, so we accept H₀" is wrong.
3. **Statistical significance ≠ practical significance.** With a huge enough n, even a trivially small, business-irrelevant effect can produce p < 0.05.
4. **α is not "the probability our result is wrong."** It's "the probability of rejecting H₀ given H₀ is actually true" — a statement about the long-run behavior of the *procedure*, not about this specific result.
5. **H₀ and H₁ must be mutually exclusive and collectively exhaustive**, and the one- vs. two-sided choice must be made *before* seeing the data — choosing it after peeking at the direction of the effect is a subtle form of p-hacking.
6. **"Reject H₀" is not the same as "H₁ is proven."** It's evidence against H₀, not proof of H₁ — the asymmetry is the whole point of the framework.

---

## 11. Production Considerations

- **α is a business decision, not a statistical law.** A company might justifiably use α=0.10 for low-stakes UI experiments (faster iteration, more tolerance for false positives) and α=0.01 for pricing/monetization changes where false positives are expensive to roll back.
- **Statistical significance ≠ practical significance.** With enough sample size, you can detect a statistically significant but practically meaningless 0.01% conversion lift. Always report effect size and confidence interval alongside the p-value, not the p-value alone.
- **One test per decision, ideally pre-registered.** If you run the significance test, look at the result, then decide to test a different metric or sub-segment, you've implicitly run multiple tests and inflated your true false-positive rate (multiple-comparisons problem — see Q&A below).

---

## 12. Famous Interview Q&A

**Q: A test returns p = 0.03. Does that mean there's a 97% chance your feature actually works?**
A: No — one of the most common misinterpretations of p-values. p = 0.03 means: *if the null hypothesis (no effect) were true*, you'd see data this extreme (or more) only 3% of the time. It says nothing directly about the probability that H₀ or H₁ is true — that would require a Bayesian framework with a prior. The correct statement is "assuming no true effect, this result would be unusual" — evidence against H₀, not a direct probability statement about the hypothesis itself.

**Q: You ran a test, got p = 0.12, and conclude "the feature has no effect." What's wrong with this conclusion?**
A: Failing to reject H₀ is not the same as proving H₀ true. p = 0.12 could mean there's genuinely no effect, or it could mean there IS an effect but your test was underpowered (too small a sample, too much noise) to detect it. Before concluding "no effect," check the power of your test given your actual sample size and observed variance — if power was low, the honest conclusion is "inconclusive," not "no effect."

**Q: A test on 5 million users shows a statistically significant 0.02% lift in click-through rate (p < 0.001). Should you ship it?**
A: The statistical-vs-practical-significance trap. With a sample size that large, even a trivially small, real effect will produce a very low p-value — the test is highly powered to detect tiny effects. The question isn't "is it significant" but "is 0.02% lift worth the engineering cost, maintenance burden, and any tradeoffs (latency, complexity) of shipping this?" A senior-level answer weighs effect size against business cost, not just the p-value.

**Q: Your team lowers the significance threshold from 0.05 to 0.01 to be "more rigorous." What's the tradeoff?**
A: You reduce the Type I error rate (fewer false positives), but you increase the Type II error rate for the same sample size — real effects now need to clear a higher bar, so you're more likely to miss true positives (lower power). If the team wants both lower α and maintained power, the only remaining lever is increasing sample size.

**Q: You ran a one-tailed test expecting the metric to increase, but the data moved sharply in the opposite direction — a huge, obvious decrease. Can you claim significance?**
A: **Trap.** People instinctively say "yes, it's a huge effect, of course it's significant." **Correct answer: No.** A one-tailed test's rejection region lives entirely on one side of the distribution. If you defined $H_1: \mu > \mu_0$, there is *no* rejection region on the left side — no matter how extreme the drop, it falls in the "fail to reject" zone by construction. This is why one-tailed tests must be chosen for principled reasons before the data arrives, and why "when in doubt, use two-tailed" is the safer default.

**Q: We ran 20 independent tests, each with a true null effect (no real difference anywhere). How many would you expect to show p < 0.05 purely by chance?**
A: About 1 (20 × 0.05 = 1). This is the **multiple-comparisons problem**, and it's why large-scale experimentation platforms (Google, Airbnb, Netflix) apply corrections like Bonferroni or Benjamini-Hochberg (FDR) when running many simultaneous tests or subgroup cuts — otherwise "significant" results are largely noise.

**Q: I doubled my sample size and my p-value dropped from 0.08 to 0.01, even though the effect size stayed exactly the same. Did the treatment get "more real"?**
A: No — larger n shrinks the standard error, so the same effect size produces a larger test statistic and smaller p-value. This is why, at Google/Meta scale with millions of users, you can get statistically significant results for effects so tiny they're practically meaningless. Always report effect size and confidence intervals alongside p-values, not p-values alone.

**Q: My p-value came out to exactly 0.05. Is that significant?**
A: It's a boundary case, and treating 0.05 as a sacred cliff is itself bad practice. Two datasets differing by noise alone can flip a result from 0.049 to 0.051 — the underlying evidence didn't meaningfully change. Better: report the exact p-value and confidence interval, and avoid binary "significant/not significant" framing this close to the threshold.

**Q: If I fail to reject H₀, have I shown there's no effect?**
A: No — this only shows insufficient evidence to detect an effect *given your sample size and variance*. It could be a real Type II error (underpowered test). This is why reporting power/minimum detectable effect (MDE) alongside a null result matters — "we found no effect" and "we couldn't have detected an effect this small anyway" are very different claims.

**Q: Why can't we just flip the logic and assume the effect exists (H₁) until proven otherwise?**
A: Because H₁ is typically composite (true for a whole range of values), there's no single sampling distribution to test data against — you'd need to pick one specific effect size to even start computing probabilities, which reintroduces exactly the specificity that made H₀ usable in the first place. (See Section 4, Layer 1.)

**Q: Two teams run the exact same experiment on the exact same data. One reports p=0.04 (two-tailed) and "ships it." The other pre-registered a one-tailed test and gets p=0.02 for the same direction. Which do you trust more?**
A: Trust the one-tailed result *only if* the directional hypothesis was genuinely pre-registered before seeing data — otherwise this is a red flag for "I peeked at the two-tailed result, didn't like the p-value, and switched tests to make it significant," a classic p-hacking move interviewers plant on purpose.

---

## 13. L5/L6-Differentiating Talking Points

- Framing the whole test as "we're trying to discredit the null, not prove the alternative" in your own words, unprompted, signals genuine understanding vs. memorized procedure.
- Volunteering that α and the one-sided/two-sided choice are **policy decisions tied to business cost of false positives/negatives** — not fixed conventions — is exactly the judgment L5 interviewers listen for.
- Being fluent enough to derive the z-statistic by hand (Section 8) without hesitating shows the difference between "I've used A/B testing tools" and "I understand what the tools are doing."
- Being able to give the full "why H₀ not H₁" answer (Section 4) unprompted is a strong differentiator — most candidates have never thought about it.

---

## 14. Rapid-Fire One-Liners

| Question | One-Liner Answer |
|---|---|
| What is a p-value? | The probability of seeing data this extreme or more, **assuming H₀ is true**. |
| Does p-value = P(H₀ is true)? | No — the single most common misinterpretation; p-value is P(data\|H₀), not P(H₀\|data). |
| Why "fail to reject" instead of "accept H₀"? | Absence of evidence isn't evidence of absence — we haven't proven H₀, just failed to disprove it. |
| Why do we negate H₀ instead of testing H₁ directly? | H₀ is a precise, single-value hypothesis with one computable distribution; H₁ is usually composite and has none. |
| What's statistical vs. practical significance? | Statistical significance says the effect is unlikely due to chance; practical significance says it's big enough to act on. |
| What does α = 0.05 mean? | We accept a 5% chance of rejecting H₀ when it's actually true (Type I error rate), decided *before* the test. |
| Can you ever "prove" H₁? | No — you can only make H₀ implausible enough to reject; that's indirect support, not proof. |
| When do you use a one-tailed test? | Only with a strong, pre-registered directional hypothesis, when you genuinely don't care about the opposite direction. |
| What's the danger of one-tailed tests? | More power in your expected direction, but zero ability to detect a significant effect in the opposite direction. |

---

## 15. Key Terms at a Glance

| Term | Symbol | Meaning |
|------|--------|---------|
| Null hypothesis | H₀ | Assumption of no effect |
| Alternative hypothesis | H₁ | Claim you want to test |
| Significance level | α | Max tolerable false positive rate |
| p-value | p | P(data this extreme \| H₀ true) |
| Test statistic | z, t, χ² | Standardised signal-to-noise ratio |
| Critical value | z*, t* | Threshold for rejecting H₀ |
| Type I error | α | False positive — reject true H₀ |
| Type II error | β | False negative — miss real effect |
| Statistical power | 1 − β | Probability of detecting a real effect |
| Effect size | d, δ | Magnitude of the true difference |

---

## 16. Common Pitfalls Checklist

1. **Setting up H₁ as the null** — always put "no effect" in H₀.
2. **Post-hoc directionality** — deciding one-tailed vs. two-tailed after seeing the result.
3. **Conflating statistical and practical significance** — a significant result isn't necessarily meaningful.
4. **Collecting data without specifying H₀ first** — leads to data dredging.
5. **Testing the wrong parameter** — make sure H₀ and H₁ match what you actually want to know.
6. **Treating "reject H₀" as "H₁ is proven"** — it's evidence against H₀, not proof of H₁.
7. **Multiple testing without correction** — running many simultaneous tests/subgroups inflates the true false-positive rate; apply Bonferroni or Benjamini-Hochberg (FDR).

---

## 17. Comprehension Check (Self-Test)

1. In your own words, what does a p-value of 0.03 actually mean?
2. Why is "fail to reject H₀" not the same as "prove H₀ is true"?
3. In the worked example (Section 8), if we'd used α = 0.10 instead of 0.05, would we have reached a different conclusion? Compute the answer.
4. Why must the choice between one-sided and two-sided testing be made before looking at the data?
5. A PM says "the p-value was 0.06, so basically nothing happened." What's wrong with this statement?
6. Why do we test H₀ instead of directly testing H₁? (Give the full three-layer answer.)
7. We ran 20 independent tests, all with a true null effect. About how many would show p < 0.05 by chance, and what's the standard fix?
8. A test on 5 million users shows p < 0.001 for a 0.02% lift. Should you ship it? Why or why not?

---
*This tutorial merges: (1) a Type I/II error, p-value, and power overview; (2) a hypothesis-testing framework chapter with a full worked example; and (3) a deep-dive foundations guide covering one-/two-tailed tests, the "why test H₀" question, and interview traps.*
