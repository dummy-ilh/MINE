# Chapter 5: Sample Size & Power Analysis

## 1. Intuition

This is the chapter where A/B testing stops being abstract statistics and becomes an actual planning problem: **"How many users do I need, and how long do I have to run this experiment, before I can trust the result?"**

Every experiment design is a negotiation between four quantities, and if you fix any three, the fourth is determined:

1. **Significance level ($\alpha$)** — your tolerance for false positives
2. **Power ($1-\beta$)** — your ability to detect a real effect when one exists
3. **Minimum Detectable Effect (MDE)** — the smallest effect size you care about detecting
4. **Sample size ($n$)** — how many users/observations you need

The intuition to internalize: **power analysis is done BEFORE the experiment**, not after. Its entire purpose is to answer "how long do I run this and how many users do I need" *before* you commit engineering and traffic resources — running an underpowered test is often worse than not running one at all, because it wastes traffic and gives you an unreliable null result you can't distinguish from "no effect."

## 2. The Formal Framework

**Power** is defined as:

$$\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ is true, effect} = \delta)$$

In words: given that a real effect of size $\delta$ truly exists, what's the probability your test correctly detects it? Conventionally, power $\geq 0.80$ is the industry standard (Google, Meta, etc. commonly use 0.80, sometimes 0.90 for high-stakes decisions).

**The sample size formula** for comparing two proportions (most common A/B test scenario):

$$n = \frac{2\left(z_{\alpha/2} + z_{\beta}\right)^2 \bar{p}(1-\bar{p})}{\delta^2}$$

where:
- $n$ = required sample size **per arm**
- $z_{\alpha/2}$ = critical value for your significance level (1.96 for $\alpha=0.05$ two-sided)
- $z_{\beta}$ = critical value for your desired power (0.84 for 80% power, 1.28 for 90% power)
- $\bar{p}$ = baseline conversion rate (approximate average of both arms)
- $\delta$ = MDE, the minimum effect size you want to detect

**Where this formula comes from**: it's derived by requiring that the test statistic simultaneously (a) exceeds the critical value under $H_0$ AND (b) has 80%+ probability of doing so if the true effect is $\delta$. The $(z_{\alpha/2}+z_\beta)^2$ term is what encodes "far enough from 0 to reject" AND "not too far to be an achievable target given the true effect."

## 3. The Four-Way Tradeoff (this is the part interviewers actually probe)

Because $n$ depends on $\alpha$, power, and $\delta$, moving any one of them changes what you need for the others:

- **Smaller $\alpha$** (stricter significance, e.g. 0.05 → 0.01) → larger $z_{\alpha/2}$ → **need more sample**
- **Higher power** (e.g., 80% → 90%) → larger $z_\beta$ → **need more sample**
- **Smaller MDE** (want to detect a tinier effect) → $\delta$ shrinks, and since $\delta$ is squared in the denominator → **sample size explodes quadratically**
- **Smaller baseline variance** $\bar{p}(1-\bar{p})$ (e.g., very rare or very common events) → **need less sample**, all else equal

The quadratic relationship between MDE and sample size is the single most important practical fact in this chapter: **halving your MDE requires 4x the sample size.** This is why "let's just detect a smaller lift" is often not a free lunch — it can turn a 2-week experiment into a 2-month one.

## 4. Worked Example

You want to detect a lift from a 10% baseline conversion rate to an 11% conversion rate (a 1 percentage point absolute MDE, i.e., $\delta = 0.01$). You want $\alpha=0.05$ (two-sided) and 80% power.

$$\bar{p} \approx \frac{0.10+0.11}{2} = 0.105, \quad \bar{p}(1-\bar{p}) = 0.105 \times 0.895 \approx 0.094$$

$$z_{\alpha/2} = 1.96, \quad z_\beta = 0.84 \text{ (for 80% power)}$$

$$n = \frac{2 \times (1.96+0.84)^2 \times 0.094}{0.01^2} = \frac{2 \times 7.84 \times 0.094}{0.0001} = \frac{1.474}{0.0001} \approx 14{,}740$$

So you need **~14,740 users per arm** (~29,480 total) to reliably detect this 1pp lift.

**Now watch the quadratic MDE effect**: if you instead wanted to detect a **0.5pp** lift ($\delta=0.005$) instead of 1pp:

$$n = \frac{2 \times 7.84 \times 0.094}{0.005^2} = \frac{1.474}{0.000025} \approx 58{,}960$$

Halving the MDE from 1pp to 0.5pp **quadrupled** the required sample size (14,740 → 58,960) — exactly as the $\delta^2$ term in the denominator predicts. This is the number to have ready if an interviewer asks "what happens to sample size if we want to detect half the effect."

**Converting to duration**: if your product gets 5,000 new users/day eligible for the experiment, split 50/50 into two arms (2,500/arm/day), you'd need $14{,}740 / 2{,}500 \approx 6$ days to hit the 1pp-MDE target, but $58{,}960/2{,}500 \approx 24$ days for the 0.5pp target.

## 5. Production Considerations

- **Always run power analysis before launching, not after.** Post-hoc power analysis (computing power based on your *observed* effect size after the experiment ends) is a well-known statistical fallacy — it's circular and gives you no new information beyond the p-value you already have.
- **Practical MDE selection is a business decision.** The MDE should be chosen as "the smallest effect size that would be worth shipping," not "the smallest effect size we can afford to detect given our traffic." These can conflict — if your traffic can only detect a 5pp lift but the business cares about 1pp lifts, you have a real constraint to surface, not paper over.
- **Diminishing returns of running longer**: because $n$ needed shrinks the CI width by only $\sqrt{n}$, doubling experiment duration doesn't double your precision — it improves it by ~41% ($\sqrt{2}\approx1.41$).
- **Variance reduction techniques (CUPED, covered in Chapter 9)** effectively increase power without collecting more samples, by reducing the variance term $\bar{p}(1-\bar{p})$ (or its continuous-metric analog) — this is a very natural bridge point to bring up your existing CUPED knowledge.

## 6. Interview Traps

- **Trap #1**: Not knowing the quadratic relationship between MDE and sample size — this is probably the single most commonly asked "do the math" question in A/B testing interviews ("what happens if we want to detect half the effect size?").
- **Trap #2**: Confusing $z_\beta$ and $z_{\alpha/2}$, or forgetting that power calculations need BOTH a significance threshold AND a target power — you cannot compute sample size from $\alpha$ alone.
- **Trap #3**: Doing post-hoc power analysis — computing "what was our power, given the effect we observed" after the experiment ran. This is circular reasoning and a well-documented statistical error.
- **Trap #4**: Forgetting that sample size formulas assume a *fixed*, pre-specified sample size — if you're peeking at results continuously and stopping early (Chapter 13), this formula's guarantees no longer hold as stated.

## 7. L5-Differentiating Talking Points

- Being able to derive/state the sample size formula from memory and actually compute a numeric answer live, rather than just saying "we'd use a power calculator," is a strong signal of genuine fluency vs. tool-dependency.
- Proactively mentioning the $\delta^2$ quadratic relationship and framing it as "this is why shrinking your MDE is expensive, not free" shows you think about the cost/tradeoff of experiment design, not just the mechanics.
- Bringing up variance reduction (CUPED) as a lever to increase power *without* more samples — connecting this chapter forward to Chapter 9 — demonstrates you see the full toolkit, not isolated formulas.
- Explicitly separating "statistically detectable" from "worth shipping" (MDE as a business threshold, not just a stats input) is exactly the kind of judgment call L5 interviewers listen for.

## 8. Comprehension Check

1. Write the sample size formula for comparing two proportions and explain what each term represents.
2. If you halve your MDE, by what factor does your required sample size change? Derive this from the formula.
3. Why is post-hoc power analysis (computed after seeing your results) considered invalid?
4. Your PM says "let's just run the experiment for 2 more weeks to get more confidence." Using the $\sqrt{n}$ relationship, what would you tell them about how much more precision that buys?
5. Name one technique (from later chapters) that increases power without requiring additional sample size, and explain briefly why it works.

---
*Next: Chapter 6 — T-tests & Z-tests*
