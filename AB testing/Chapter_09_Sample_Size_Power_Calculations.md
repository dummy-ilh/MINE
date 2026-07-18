# Chapter 9: Sample Size & Power Calculations (MDE, Baseline Variance)

## 1. Definition

Sample size calculation is the process of determining how many users (or units) you need per arm to reliably detect a specified effect size, given your metric's baseline variance, your desired power, and your significance threshold. The core inputs are:

- **MDE (Minimum Detectable Effect):** the smallest true effect size you want the test to be able to reliably detect — a design choice, not something you observe.
- **Baseline variance (σ²):** the natural variability of the metric before any treatment.
- **α (significance threshold):** typically 0.05.
- **Power (1-β):** typically 0.80 or 0.90 — the probability of detecting the MDE if it truly exists.

## 2. Layman Explanation

Think of sample size calculation like deciding how many jellybeans you need to taste from a jar to reliably tell if it's "mostly sweet" vs. "mostly sour," when the jar is a mix of both. If the jar is very mixed (high variance) or you want to detect a very subtle shift in sweetness (small MDE), you need to taste a lot more jellybeans. If the jar is mostly one flavor already (low variance) or you're looking for an obvious shift, a handful of tastes will do.

The MDE is you deciding upfront: "I only care if this feature moves conversion by at least 1 percentage point — smaller than that isn't worth acting on anyway." That decision, combined with how noisy conversion normally is, tells you exactly how many users you need before even starting the test — so you don't run a test for 2 weeks only to realize afterward it was never going to be able to detect anything meaningful.

## 3. Formal Explanation

**Sample size formula (two-sample test of proportions, per arm):**

n = (z_(α/2) + z_β)² × [p₁(1-p₁) + p₂(1-p₂)] / (p₁ - p₂)²

Where:
- p₁ = baseline conversion rate
- p₂ = p₁ + MDE (expected conversion rate under treatment)
- z_(α/2) = 1.96 for α=0.05 two-sided
- z_β = 0.84 for 80% power, 1.28 for 90% power

**Sample size formula (two-sample test of means, continuous metric):**

n = 2σ² × (z_(α/2) + z_β)² / δ²

Where δ is the MDE in absolute units and σ² is the baseline variance of the metric.

**Key relationships (memorize these for interviews):**
- n is inversely proportional to δ² — halving your MDE (wanting to detect a smaller effect) requires **4x** the sample size.
- n is directly proportional to σ² — doubling variance requires roughly double the sample size.
- n grows with the square of (z_(α/2) + z_β) — tightening α or raising power both increase required n, but the relationship is driven by the sum of these z-values, so the marginal cost of one more "unit" of power/significance stringency compounds.

**Duration vs. sample size:**
Once you know required n per arm, test duration = n / (daily unique users allocated to the experiment). This is why low-traffic features (e.g., enterprise-tier settings) sometimes need tests running for months, while high-traffic surfaces (homepage) can reach required n in days.

## 4. Levers — What Controls It, What Moves It

**MDE (the effect size you're trying to detect)**
- Smaller MDE → dramatically larger required n (squared relationship) — chasing tiny effects is expensive.
- MDE should be chosen based on business relevance, not just "whatever we can afford to detect" — a common mistake is setting MDE too large just to make the sample size requirement feel achievable, which risks missing real, smaller-but-still-valuable effects.

**Baseline variance (σ²)**
- Higher-variance metrics (e.g., revenue, heavily skewed metrics) need larger samples for the same MDE than low-variance metrics (e.g., binary conversion near extreme rates).
- Variance reduction techniques (CUPED, stratified sampling — Chapter 14) directly shrink σ², reducing required n without needing more traffic — often the more practical lever than "just wait longer" when traffic is fixed.

**Power target (1-β)**
- Raising power from 80% to 90% increases required n noticeably — many companies default to 80% power as an acceptable balance, reserving 90%+ for high-stakes launches where missing a real effect is costly.

**Baseline rate (for proportions)**
- p(1-p) is maximized at p=0.5 and shrinks near the extremes — but as discussed in Chapter 1, very rare events (low p) require detecting a proportionally tiny absolute effect, which usually dominates and increases required n despite the lower p(1-p) term.

**Traffic allocation**
- Allocating more of your total traffic to the experiment (e.g., 50/50 split vs. 90/10) directly shortens the time needed to reach required n, at the cost of exposing more users to an unproven treatment — a risk/speed tradeoff, especially relevant for novel or risky features (often mitigated via staged rollouts: 1% → 10% → 50%).

## 5. Famous Q&A (Google / Apple style)

**Q: Your team wants to detect an MDE half as large as originally planned, to catch smaller wins. What happens to your required sample size?**
A: It roughly quadruples. Because required sample size scales with 1/δ², halving the minimum detectable effect requires four times the sample size to maintain the same power — this nonlinear relationship is one of the most common traps in interviews, since people often assume it's a simple 2x increase.

**Q: A test on a low-traffic enterprise feature would take 6 months to reach the sample size needed for 80% power at your desired MDE. What options do you have?**
A: A few levers: (1) increase the MDE — accept that you can only reliably detect a larger effect, which may still be acceptable if smaller effects aren't business-relevant anyway; (2) reduce variance via CUPED or a better pre-experiment covariate, effectively lowering required n without more traffic; (3) increase the fraction of enterprise traffic allocated to the test if risk tolerance allows; (4) consider a proxy metric with a higher baseline rate or lower variance that's been validated to correlate with the true outcome, allowing faster detection; or (5) accept the longer timeline if the decision is high-stakes enough to warrant it, rather than under-powering the test and risking a false "no effect" conclusion.

**Q: Why does a company running an experiment on a rare event (e.g., 0.1% fraud rate) need such enormous sample sizes, even though p(1-p) is small at low p?**
A: Because although the *variance term* p(1-p) is small near p=0.1%, the MDE you're trying to detect is proportionally even smaller — a "20% relative reduction" in fraud at a 0.1% baseline is only a 0.02 percentage point absolute change, and required n scales with 1/δ² where δ is that tiny absolute effect. The shrinking variance doesn't compensate enough for the shrinking effect size you're chasing, so rare-event experiments typically need very large populations or much longer durations — this is why fraud/abuse teams often rely on much bigger platforms' worth of traffic, longer test windows, or proxy metrics with higher base rates.

**Q: Two candidate features both look promising. One has a big expected effect but is expensive to build; the other has a small expected effect but is cheap. How does sample size planning influence which one you'd prioritize testing first?**
A: The big-effect feature is statistically "cheaper" to test — it needs a smaller sample size (and thus shorter test duration) to reach a confident conclusion, because required n shrinks as MDE grows. The small-effect feature, even if cheap to build, may require a very large sample size (or long duration) just to get a reliable read — meaning the *total* cost of validating it (engineering + time-to-signal) could actually exceed the big-effect feature's cost. I'd factor in both build cost and statistical cost-to-detect when prioritizing, not just engineering cost alone — a genuinely strong senior-level answer surfaces this less obvious tradeoff.

---
*Next: Chapter 10 — Randomization Mechanics (hashing, bucketing, SRM checks).*
