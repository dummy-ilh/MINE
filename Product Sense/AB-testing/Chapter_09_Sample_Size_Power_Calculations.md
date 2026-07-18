# Chapter 9: Sample Size & Power Calculations (MDE, Baseline Variance)

---

## 1. Intuition

This is the chapter where A/B testing stops being abstract statistics and becomes an actual planning problem: **"How many users do I need, and how long do I have to run this experiment, before I can trust the result?"**

Sample size calculation determines how many users (or units) you need per arm to reliably detect a specified effect size, given your metric's baseline variance, your desired power, and your significance threshold.

Every experiment design is a negotiation between four quantities — fix any three, and the fourth is determined:

1. **Significance level (α)** — your tolerance for false positives
2. **Power (1−β)** — your ability to detect a real effect when one exists
3. **Minimum Detectable Effect (MDE)** — the smallest effect size you care about detecting
4. **Sample size (n)** — how many users/observations you need

**The core idea to internalize**: power analysis is done **before** the experiment, not after. Its entire purpose is to answer "how long do I run this and how many users do I need" *before* you commit engineering and traffic resources. Running an underpowered test is often worse than not running one at all, because it wastes traffic and gives you an unreliable null result you can't distinguish from "no effect."

### Layman analogy
Think of sample size calculation like deciding how many jellybeans you need to taste from a jar to reliably tell if it's "mostly sweet" vs. "mostly sour," when the jar is a mix of both. If the jar is very mixed (high variance) or you want to detect a very subtle shift in sweetness (small MDE), you need to taste a lot more jellybeans. If the jar is mostly one flavor already (low variance) or you're looking for an obvious shift, a handful of tastes will do.

The MDE is you deciding upfront: "I only care if this feature moves conversion by at least 1 percentage point — smaller than that isn't worth acting on anyway." That decision, combined with how noisy conversion normally is, tells you exactly how many users you need before even starting the test — so you don't run a test for 2 weeks only to realize afterward it was never going to be able to detect anything meaningful.

---

## 2. Core Definitions

- **MDE (Minimum Detectable Effect)**: the smallest true effect size you want the test to be able to reliably detect — a **design choice**, not something you observe.
- **Baseline variance (σ²)**: the natural variability of the metric before any treatment.
- **α (significance threshold)**: typically 0.05.
- **Power (1−β)**: typically 0.80 or 0.90 — the probability of detecting the MDE if it truly exists. Formally:

$$\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ is true, effect} = \delta)$$

In words: given that a real effect of size δ truly exists, what's the probability your test correctly detects it? Power ≥ 0.80 is the industry standard (Google, Meta, etc. commonly use 0.80, sometimes 0.90 for high-stakes decisions).

---

## 3. The Formulas

### Sample size — two proportions (most common A/B test scenario, per arm)

$$n = \frac{2\left(z_{\alpha/2} + z_{\beta}\right)^2 \bar{p}(1-\bar{p})}{\delta^2}$$

Equivalently, written with explicit $p_1, p_2$:

$$n = \frac{(z_{\alpha/2} + z_{\beta})^2 \times [p_1(1-p_1) + p_2(1-p_2)]}{(p_1 - p_2)^2}$$

Where:
- $n$ = required sample size **per arm**
- $z_{\alpha/2}$ = critical value for significance level (**1.96** for α=0.05 two-sided)
- $z_{\beta}$ = critical value for desired power (**0.84** for 80% power, **1.28** for 90% power)
- $\bar{p}$ = baseline conversion rate (approx. average of both arms); equivalently $p_1$ = baseline rate, $p_2 = p_1 + \text{MDE}$
- $\delta$ = MDE, the minimum effect size you want to detect, in absolute units

### Sample size — two means (continuous metric)

$$n = \frac{2\sigma^2 (z_{\alpha/2} + z_{\beta})^2}{\delta^2}$$

where δ is the MDE in absolute units and σ² is the baseline variance of the metric.

### Where the formula comes from
It's derived by requiring the test statistic simultaneously (a) exceeds the critical value under H₀ **AND** (b) has 80%+ probability of doing so if the true effect is δ. The $(z_{\alpha/2}+z_\beta)^2$ term encodes "far enough from 0 to reject" **AND** "not too far to be an achievable target given the true effect."

### Key relationships (memorize these for interviews)
- $n$ is **inversely proportional to δ²** — halving your MDE (wanting to detect a smaller effect) requires **4x** the sample size. This is the single most commonly tested fact in this chapter.
- $n$ is **directly proportional to σ²** — doubling variance requires roughly double the sample size.
- $n$ grows with the **square of $(z_{\alpha/2} + z_\beta)$** — tightening α or raising power both increase required n, and since it's the *sum* of the two z-values that's squared, tightening both simultaneously compounds faster than tightening either alone.

### Duration vs. sample size
Once you know required n per arm:

$$\text{Test duration} = \frac{n}{\text{daily unique users allocated to the experiment, per arm}}$$

This is why low-traffic features (e.g., enterprise-tier settings) sometimes need tests running for months, while high-traffic surfaces (homepage) can reach required n in days.

---

## 4. Worked Example (with the quadratic MDE effect made concrete)

You want to detect a lift from a 10% baseline conversion rate to an 11% conversion rate (a 1 percentage point absolute MDE, i.e., δ = 0.01). You want α = 0.05 (two-sided) and 80% power.

$$\bar{p} \approx \frac{0.10+0.11}{2} = 0.105, \quad \bar{p}(1-\bar{p}) = 0.105 \times 0.895 \approx 0.094$$

$$z_{\alpha/2} = 1.96, \quad z_\beta = 0.84 \text{ (for 80% power)}$$

$$n = \frac{2 \times (1.96+0.84)^2 \times 0.094}{0.01^2} = \frac{2 \times 7.84 \times 0.094}{0.0001} = \frac{1.474}{0.0001} \approx 14{,}740$$

So you need **~14,740 users per arm** (~29,480 total) to reliably detect this 1pp lift.

**Now watch the quadratic MDE effect**: if instead you wanted to detect a **0.5pp** lift (δ=0.005):

$$n = \frac{2 \times 7.84 \times 0.094}{0.005^2} = \frac{1.474}{0.000025} \approx 58{,}960$$

Halving the MDE from 1pp to 0.5pp **quadrupled** the required sample size (14,740 → 58,960) — exactly as the δ² term in the denominator predicts. This is the number to have ready if an interviewer asks "what happens to sample size if we want to detect half the effect."

**Converting to duration**: if your product gets 5,000 new users/day eligible for the experiment, split 50/50 into two arms (2,500/arm/day):
- 1pp MDE target: $14{,}740 / 2{,}500 \approx 6$ days
- 0.5pp MDE target: $58{,}960 / 2{,}500 \approx 24$ days

---

## 5. Levers — What Controls Required Sample Size

**MDE (the effect size you're trying to detect)**
- Smaller MDE → dramatically larger required n (squared relationship) — chasing tiny effects is expensive.
- MDE should be chosen based on **business relevance**, not just "whatever we can afford to detect." A common mistake is setting MDE too large just to make the sample size requirement feel achievable, which risks missing real, smaller-but-still-valuable effects.

**Baseline variance (σ²)**
- Higher-variance metrics (revenue, heavily skewed metrics) need larger samples for the same MDE than low-variance metrics (binary conversion near extreme rates).
- Variance-reduction techniques (CUPED, stratified sampling) directly shrink σ², reducing required n without needing more traffic — often the more practical lever than "just wait longer" when traffic is fixed.

**Power target (1−β)**
- Raising power from 80% to 90% increases required n noticeably. Many companies default to 80% power as an acceptable balance, reserving 90%+ for high-stakes launches where missing a real effect is costly.

**Baseline rate (for proportions)**
- $p(1-p)$ is maximized at $p=0.5$ and shrinks near the extremes — but very rare events (low p) require detecting a proportionally tiny absolute effect, which usually dominates and **increases** required n despite the lower $p(1-p)$ term (see fraud-rate Q&A below).

**Traffic allocation**
- Allocating more of your total traffic to the experiment (e.g., 50/50 vs. 90/10) directly shortens the time needed to reach required n, at the cost of exposing more users to an unproven treatment — a risk/speed tradeoff, especially relevant for novel or risky features (often mitigated via staged rollouts: 1% → 10% → 50%).

**α (significance threshold)**
- Smaller α (stricter significance, e.g. 0.05 → 0.01) → larger $z_{\alpha/2}$ → need more sample.

---

## 6. The Four-Way Tradeoff (What Interviewers Actually Probe)

Because n depends on α, power, and δ, moving any one of them changes what you need for the others:

| Change | Effect on required n |
|---|---|
| Smaller α (stricter significance, e.g. 0.05 → 0.01) | Larger $z_{\alpha/2}$ → **more sample needed** |
| Higher power (e.g., 80% → 90%) | Larger $z_\beta$ → **more sample needed** |
| Smaller MDE (want to detect a tinier effect) | δ shrinks, squared in denominator → **sample size explodes quadratically** |
| Smaller baseline variance $\bar{p}(1-\bar{p})$ (rare or very common events) | **Less sample needed**, all else equal |

---

## 7. Important Concepts & How They Affect Other Things

Sample size and power don't live in isolation — they're the hinge that connects experiment *design* to nearly every other part of the A/B testing stack. Here's how these levers ripple outward:

### → Effect on Confidence Intervals
CI width shrinks with $\sqrt{n}$, not n (same square-root law as the sample-size formula's δ² term, just inverted). So:
- The MDE you plan for during power analysis is roughly the width of CI you should expect to see post-hoc — if your test is powered to detect a 1pp MDE, don't be surprised when your resulting CI is also on that order of width.
- **Diminishing returns of running longer**: doubling experiment duration doesn't double your precision — it improves CI width by only ~41% ($\sqrt{2}\approx1.41$). This is the direct link between the sample-size chapter's n and the CI chapter's width formula — the same square root shows up in both because they're the same underlying math (SE ∝ $1/\sqrt{n}$).

### → Effect on Hypothesis Testing / the CI-Test Duality
- An underpowered test (n too small for your true effect size) doesn't just risk "no significant result" — it means a **failure to reject H₀ is uninformative**, not evidence of no effect (Type II error risk). This directly feeds the hypothesis-testing trap of confusing "fail to reject" with "proved no effect."
- Because of the CI ↔ hypothesis-test duality, an underpowered test also produces a CI so wide it straddles both practically-meaningful and practically-irrelevant effect sizes — the test technically "ran," but it can't discriminate between "big win" and "no effect," which is a design failure, not a result.

### → Effect on Business/Roadmap Decisions
- **MDE is fundamentally a business threshold, not a stats input.** If your traffic can only detect a 5pp lift but the business cares about 1pp lifts, that's a real constraint to surface to stakeholders — not something to paper over by quietly running an underpowered test.
- **Statistical cost-to-detect should factor into feature prioritization**, not just engineering cost. A feature with a small expected effect may be cheap to build but expensive to *validate* (needs a huge n or long duration), while a feature with a big expected effect is statistically "cheap" to test even if expensive to build. Total cost = build cost + time-to-signal cost.
- **Rare-event metrics (fraud, churn, safety incidents)** need enormous sample sizes not because $p(1-p)$ is large (it's actually small near extreme p), but because the MDE you're chasing is also proportionally tiny — a "20% relative reduction" in a 0.1% fraud rate is only a 0.02pp absolute change, and n scales with $1/\delta^2$. This is why fraud/abuse teams often need much bigger platforms' worth of traffic, longer windows, or proxy metrics.

### → Effect on Variance-Reduction Techniques (CUPED, stratification)
- These techniques increase power **without** collecting more samples, by shrinking the variance term ($\bar{p}(1-\bar{p})$ or its continuous analog, or σ² directly) in the sample-size/power formula.
- This is the most "free lunch"-like lever available: unlike MDE (a business tradeoff) or α/power (a rigor tradeoff), reducing variance is close to a pure win, limited mainly by the quality of the pre-experiment covariate you have available.

### → Effect on Experiment Velocity & Traffic Allocation Strategy
- Low-traffic surfaces effectively have a *ceiling* on how small an MDE they can detect in a reasonable timeframe — this shapes which experiments get run where (e.g., testing bold, large-effect ideas on low-traffic enterprise tiers; reserving subtle refinements for high-traffic consumer surfaces).
- Traffic allocation (50/50 vs. 90/10, staged rollouts) is a speed/risk tradeoff that only makes sense once you know your target n — you can't reason about "how fast can we get signal" without first anchoring to the sample size formula.

### → Effect on Test Validity / Peeking
- The sample-size formula assumes a **fixed, pre-specified sample size**. If you're continuously peeking at results and stopping early once significance is hit, the formula's guarantees no longer hold — this connects forward to sequential testing / always-valid inference methods, which use different (more permissive but more complex) machinery specifically because the fixed-n assumption is violated.

---

## 8. Production Considerations

- **Always run power analysis before launching, not after.** Post-hoc power analysis (computing power based on your *observed* effect size after the experiment ends) is a well-known statistical fallacy — it's circular and gives you no new information beyond the p-value you already have.
- **Practical MDE selection is a business decision**, not "the smallest effect we can afford to detect given our traffic" (see Section 7).
- **Diminishing returns of running longer**: n needed shrinks CI width by only $\sqrt{n}$ — doubling duration improves precision by ~41%, not 2x.
- **Variance reduction (CUPED)** effectively increases power without collecting more samples — a natural bridge to bring up if you already know CUPED.

---

## 9. Famous Interview Q&A

**Q: Your team wants to detect an MDE half as large as originally planned, to catch smaller wins. What happens to your required sample size?**
A: It roughly quadruples. Required sample size scales with $1/\delta^2$, so halving the minimum detectable effect requires four times the sample size to maintain the same power — this nonlinear relationship is one of the most common traps in interviews, since people often assume it's a simple 2x increase.

**Q: A test on a low-traffic enterprise feature would take 6 months to reach the sample size needed for 80% power at your desired MDE. What options do you have?**
A: Several levers: (1) increase the MDE — accept you can only reliably detect a larger effect, acceptable if smaller effects aren't business-relevant anyway; (2) reduce variance via CUPED or a better pre-experiment covariate, lowering required n without more traffic; (3) increase the fraction of enterprise traffic allocated to the test if risk tolerance allows; (4) use a proxy metric with a higher baseline rate or lower variance that's been validated to correlate with the true outcome, for faster detection; or (5) accept the longer timeline if the decision is high-stakes enough, rather than under-powering the test and risking a false "no effect" conclusion.

**Q: Why does a company running an experiment on a rare event (e.g., 0.1% fraud rate) need such enormous sample sizes, even though p(1-p) is small at low p?**
A: Although the *variance term* p(1-p) is small near p=0.1%, the MDE you're trying to detect is proportionally even smaller — a "20% relative reduction" in fraud at a 0.1% baseline is only a 0.02 percentage point absolute change, and required n scales with $1/\delta^2$ where δ is that tiny absolute effect. The shrinking variance doesn't compensate enough for the shrinking effect size you're chasing, so rare-event experiments typically need very large populations or much longer durations — this is why fraud/abuse teams often rely on much bigger platforms' worth of traffic, longer test windows, or proxy metrics with higher base rates.

**Q: Two candidate features both look promising. One has a big expected effect but is expensive to build; the other has a small expected effect but is cheap. How does sample size planning influence which one you'd prioritize testing first?**
A: The big-effect feature is statistically "cheaper" to test — it needs a smaller sample size (and shorter duration) to reach a confident conclusion, since required n shrinks as MDE grows. The small-effect feature, even if cheap to build, may require a very large sample size (or long duration) just to get a reliable read — meaning the *total* cost of validating it (engineering + time-to-signal) could exceed the big-effect feature's cost. I'd factor in both build cost and statistical cost-to-detect when prioritizing, not just engineering cost alone.

**Q: Your PM says "let's just run the experiment for 2 more weeks to get more confidence." What would you tell them?**
A: Precision (CI width, and correspondingly your ability to resolve smaller effects) scales with $\sqrt{n}$, not n. So doubling the duration doesn't double your confidence/precision — it improves it by only about 41% ($\sqrt{2} \approx 1.41$). If they want meaningfully tighter results, they should think in terms of "how many multiples of duration" rather than "a bit more time," and weigh that against just accepting a wider CI or reducing variance via CUPED instead.

---

## 10. L5-Differentiating Talking Points

- Being able to derive/state the sample size formula from memory and actually compute a numeric answer live, rather than just saying "we'd use a power calculator," is a strong signal of genuine fluency vs. tool-dependency.
- Proactively mentioning the δ² quadratic relationship and framing it as "this is why shrinking your MDE is expensive, not free" shows you think about the cost/tradeoff of experiment design, not just the mechanics.
- Bringing up variance reduction (CUPED) as a lever to increase power *without* more samples demonstrates you see the full toolkit, not isolated formulas.
- Explicitly separating "statistically detectable" from "worth shipping" (MDE as a business threshold, not just a stats input) is exactly the kind of judgment call L5 interviewers listen for.
- Connecting sample size/power explicitly to CI width, the hypothesis-test duality, and prioritization tradeoffs (Section 7) shows you see experimentation as one connected system, not a stack of isolated formulas.

---

## 11. Comprehension Check (Self-Test)

1. Write the sample size formula for comparing two proportions and explain what each term represents.
2. If you halve your MDE, by what factor does your required sample size change? Derive this from the formula.
3. Why is post-hoc power analysis (computed after seeing your results) considered invalid?
4. Your PM says "let's just run the experiment for 2 more weeks to get more confidence." Using the $\sqrt{n}$ relationship, what would you tell them about how much more precision that buys?
5. Name one technique that increases power without requiring additional sample size, and explain briefly why it works.
6. Why do rare-event metrics (like fraud rate) require such large sample sizes despite having small baseline variance?
7. How does an underpowered test undermine the CI-hypothesis-test duality discussed elsewhere in this curriculum?
8. Why is MDE described as a "business decision" rather than a purely statistical one?

---
*This tutorial merges two chapters on sample size and power analysis (one framed around Type I/II error mechanics and business Q&A, one framed around formula derivation and a full worked numeric example), plus an added section mapping how these levers ripple into confidence intervals, hypothesis testing, business prioritization, variance reduction, and test validity.*
