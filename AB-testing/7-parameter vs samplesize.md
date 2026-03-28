
## The core framework

Every A/B test is a decision under uncertainty. You observe data and decide: *is variant B genuinely better, or is this just noise?* The framework uses four parameters:

**Significance level (α)** is your false positive rate — the probability you declare a winner when there's actually no real difference. The standard is α = 0.05, meaning you accept a 5% chance of a false alarm. This directly *is* your FPR.

**Statistical power (1 − β)** is your true positive rate — the probability you detect a real effect when it genuinely exists. Typical target is 80% or 90%. Its complement, β, is the false negative rate (missing a real win).

**Effect size (MDE — minimum detectable effect)** is the smallest lift you care about detecting. A 0.1% improvement in revenue per user might be negligible; a 5% improvement is meaningful. You design the test around this.

**Sample size** is derived from the three above. It's not a free choice — once you fix α, power, and MDE, the required sample size is determined mathematically.

Here's the relationship visually:---

## The two-distribution picture

The deepest intuition for power and significance comes from visualizing the two overlapping distributions. Here's an interactive version — adjust the sliders to see how everything connects:---

## How to select each parameter

**Significance level α:** The industry default is 5%. Use 1% when the cost of a false positive is high (shipping a bad feature, making a costly infrastructure change). Use 10% only for low-stakes exploratory tests. The key insight: α is directly your false positive rate — there's no trick here, it's definitional.

**Power (1−β):** 80% is the minimum standard; 90% is better if you can afford the extra sample size (it requires ~30% more users). The asymmetry matters: in most businesses, the cost of shipping something that doesn't work (false positive) is worse than missing a small win (false negative), which is why α is set tighter than β.

**Effect size / MDE:** This is a business decision, not a statistics decision. Ask: *what's the smallest improvement that would actually change our roadmap?* A 0.5% lift on a low-traffic page might be meaningless; the same lift on your checkout page might be worth millions. Smaller MDEs require vastly larger sample sizes — notice in the widget above how pulling the MDE down sends required samples through the roof.

**Baseline conversion rate:** Lower baselines require larger samples because the variance is different. A 1% CVR test needs far more traffic than a 30% CVR test for the same relative lift.

---

## Controlling your false positive rate

Your FPR can get inflated well beyond α if you're not careful. The main culprits:

**Peeking** — checking results daily and stopping when p < 0.05 dramatically inflates FPR. If you peek 5 times, your actual FPR is closer to 20%, not 5%. Solutions: pre-commit to a sample size and only look once, or use sequential testing methods (like always-valid inference / mSPRT) that are designed for continuous monitoring.

**Multiple comparisons** — running 10 variants simultaneously means under the null hypothesis you'd expect ~0.5 false positives at α = 5%. Use Bonferroni correction (divide α by the number of tests) or a Benjamini-Hochberg correction for a less conservative approach.

**Multiple metrics** — if you test primary metric + 5 secondary metrics, you're doing 6 hypothesis tests. Be explicit about your primary metric before running the test, and treat secondary metrics as exploratory only.

**Novelty effects** — new features often show inflated early results because users are curious. Run tests long enough (at least 1–2 full business cycles) to let novelty decay.

---

## Interpreting results

A p-value below α does *not* mean "the effect is real" — it means "if there were no effect, data this extreme would occur less than α% of the time." The p-value tells you nothing about effect size.

Always report the confidence interval alongside significance. A statistically significant +0.1% lift with a 95% CI of [+0.02%, +0.18%] is meaningful. A +5% lift with CI of [−2%, +12%] might be significant if your sample was huge, but the practical uncertainty is enormous.

The practical checklist before calling a test: did you reach your pre-specified sample size, has the test run for at least one full business cycle, is your primary metric significant, and does the effect size actually matter for the business?



Here's a summary of the key relationships:

---

### How each parameter affects sample size

The sample size formula ties everything together: roughly `n ∝ (z_α + z_β)² / MDE²`. This means:

**MDE has the biggest lever.** Sample grows as `1/MDE²` — halving the minimum detectable effect *quadruples* required sample. This is the single most impactful decision in test design. If your test is running too long, ask whether you actually need to detect that small an effect.

**Power has diminishing returns.** Going from 50% → 80% power is cheap. Going from 80% → 90% costs ~30% more sample. Going from 90% → 99% roughly doubles it. The upper tail is expensive — most teams use 80% as their default and reserve 90% for high-stakes or irreversible decisions.

**Alpha has a moderate, non-linear effect.** Moving from α=5% to α=1% adds ~20–30% more sample because you're raising the critical threshold. The tradeoff is direct: lower α = fewer false positives, but if sample size is constrained, you're implicitly accepting a higher false negative rate (β goes up).

---

### The fundamental tradeoff triangle

These three are locked together. Fix any two, and the third is determined:

- Fix α and power → sample size is determined by MDE
- Fix α and MDE → you can calculate what power a given sample gives you
- Fix power and MDE → tightening α forces you to collect more data

In practice: **set α and power first** (business policy decisions), then use the MDE to calculate how long your test needs to run. If the runtime is infeasible, you adjust the MDE — not the error rates.

---

### The effect of running underpowered tests

This is where many teams get burned. If you run a test that's underpowered (say, 40% power), one of two things happens: you see a non-significant result and incorrectly conclude the variant doesn't work (high β), or you happen to get p < 0.05 and your *effect size estimate is inflated* (winner's curse — small samples only detect effects by exaggerating them). Underpowered tests produce unreliable results in both directions.


**In a single, pre-specified test run correctly — yes.** But in practice, your actual FPR can be much higher than 5% even with α = 0.05. Here's when the guarantee breaks:

---

### When α = 0.05 holds as the FPR ceiling

The 5% guarantee is valid only when all of these are true:
- You run **one test**, testing **one metric**
- You decide your sample size **before** starting
- You look at results **exactly once** (at the end)
- You don't change anything mid-test

Under those conditions, α is exactly your FPR — by definition.

---

### When your actual FPR silently inflates above 5%

**1. Peeking / early stopping**
Checking results daily and stopping when p < 0.05 is the most common mistake. Each peek is an additional chance to cross the threshold by chance.

| Peeks at α = 0.05 | Actual FPR |
|---|---|
| 1 (correct) | 5% |
| 5 | ~14% |
| 10 | ~19% |
| 100 | ~37% |

You *feel* like you're using α = 0.05 but you've actually raised your FPR to 20–40%.

**2. Multiple metrics**
If you test 1 primary + 4 secondary metrics, the probability that *at least one* is significant by chance under the null is:

`1 − (1 − 0.05)⁵ = 22.6%` actual FPR

**3. Multiple variants**
Testing control vs. variant A vs. variant B vs. variant C involves 3+ comparisons. Same inflation applies.

**4. HARKing** (Hypothesising After Results are Known)
Running the test, seeing which metric moved, and then calling *that* your primary metric. Your FPR is now undefined — and almost certainly very high.

---

### How to actually protect your FPR

| Problem | Fix |
|---|---|
| Peeking | Pre-commit to sample size; look once. Or use sequential testing (mSPRT) |
| Multiple metrics | Declare one primary metric before launch; treat others as exploratory |
| Multiple variants | Bonferroni: use α/k per test (k = number of comparisons) |
| Multiple tests over time | Track your experiment-level FPR, not just per-test α |

---

**The short version:** α = 0.05 sets your FPR at 5% *per single correctly-run test*. Every deviation from that ideal — peeking, multiple metrics, multiple variants — compounds the error rate multiplicatively. The 5% is a promise you make to yourself through discipline in test design, not a guarantee the statistics automatically enforce.


Exactly the right question. The core reason is what we just discussed — **each t-test carries its own α, and they multiply**.

---

### The math of the problem

Say you have 5 variants (control + 4) and run all pairwise t-tests at α = 0.05. That's **10 comparisons**. The probability that *at least one* is a false positive under the null:

`1 − (1 − 0.05)¹⁰ = **40.1% actual FPR**`

You set α = 0.05 but your real false positive rate is 40%. You're essentially guaranteeing a spurious winner.---

### What ANOVA actually does differently

A t-test asks: *"are these two specific means different?"*

ANOVA asks: *"are **any** of these k means different from each other?"* — in a **single test**.

It does this by decomposing variance into two buckets:

- **Between-group variance** — how much do the group means differ from the grand mean? (signal)
- **Within-group variance** — how much do individual observations vary within each group? (noise)

The F-statistic is just their ratio: `F = between-group variance / within-group variance`. If F is large, the groups differ more than noise alone would explain.

Because it's **one test**, the FPR stays at exactly α — no inflation.

---

### The two-stage workflow (what you actually do)

ANOVA alone tells you *something* is different, not *what*. So the correct approach is:

**Stage 1 — ANOVA** → Is there any signal at all? If p > α, stop. No variant wins.

**Stage 2 — Post-hoc tests** → *Only if* ANOVA is significant, run pairwise comparisons using a method that corrects for multiplicity:

| Method | When to use | How it works |
|---|---|---|
| **Tukey HSD** | Comparing all pairs | Controls family-wise error rate exactly |
| **Bonferroni** | Any set of comparisons | Divide α by number of tests — conservative but simple |
| **Dunnett's test** | Each variant vs control only | More powerful than Bonferroni for this specific case |
| **Benjamini-Hochberg** | Many comparisons, ok with some FPs | Controls false *discovery* rate, not family-wise |

For A/B testing specifically, **Dunnett's is usually best** — you're comparing variant A, B, C against control, not against each other, so you're doing fewer comparisons and the test is more powerful.

---

The key intuition: ANOVA is the bouncer that says "something's going on here" before you start looking at pairs. Skipping it and running straight to pairwise t-tests is like running 10 coin flips and declaring heads because at least one came up heads.
