# Chapter 8: Hypothesis Formulation & Pre-Registration

## 1. Definition

**Hypothesis formulation** is the process of translating a product idea into a precise, testable statement before running an experiment: what change is being made, what metric it's expected to move, in what direction, and why (the causal mechanism).

**Pre-registration** means committing — in writing, before looking at any results — to the hypothesis, the primary metric, the analysis method, the sample size/duration, and the success criteria. It exists specifically to prevent post-hoc reasoning: deciding what "counts" as a win only after seeing which metrics moved favorably.

## 2. Layman Explanation

A good hypothesis isn't "let's see what happens if we change the button color." It's: "We believe changing the CTA button from grey to blue will increase click-through rate by at least 3%, because grey currently blends into the background and users don't notice it as a clickable action."

Pre-registration is like placing your bet *before* the race starts, not after watching the replay and claiming you "knew" the winner. If you decide after the fact which metric to highlight ("well, CTR didn't move, but look — time-on-page went up!"), you're no longer testing a hypothesis — you're fishing through results until something looks good, which inflates your real false-positive rate even if each individual p-value looks fine in isolation.

## 3. Formal Explanation

**A well-formed hypothesis has four components:**
1. **Change:** what specifically is being modified (the treatment)
2. **Mechanism:** the causal story for *why* this change should affect behavior
3. **Metric + direction:** which specific metric, and whether you expect it to go up or down
4. **Magnitude (MDE):** the minimum effect size that would be considered meaningful — this feeds directly into the sample size calculation (Chapter 9)

**Pre-registration should lock in, before data collection:**
- The primary metric (the one that determines ship/no-ship — not a menu of options to pick from afterward)
- Secondary/guardrail metrics
- The statistical test to be used
- The planned sample size / test duration
- The significance threshold (α) and whether the test is one-sided or two-sided
- Any planned subgroup analyses (specified in advance, not discovered afterward)

**Why this matters statistically — the multiple comparisons / "garden of forking paths" problem:**
If you don't pre-register and instead look at 20 metrics after the fact, picking whichever one is significant, your true Type I error rate across the "experiment" is much higher than the nominal 5% per-metric threshold — even though each individual test still reports p < 0.05 correctly, the *procedure* of cherry-picking after seeing results invalidates the guarantee. This is distinct from (but related to) the formal multiple testing correction covered in Chapter 15 — pre-registration prevents the informal, undocumented version of this problem (analyst degrees of freedom), while multiple testing correction handles the formal version (multiple pre-specified metrics tested simultaneously).

**One-sided vs. two-sided commitment:**
This must be decided at pre-registration, not after seeing the sign of the effect — deciding "oh, it went down, let's test if it's significantly lower" after the fact is a direct form of p-hacking.

## 4. Levers — What Controls It, What Moves It

**Specificity of the mechanism**
- A vague hypothesis ("this might help engagement") gives you no way to pre-specify a single primary metric or a meaningful MDE — it invites fishing after the fact. A precise mechanism ("simplifying checkout reduces friction, which should reduce cart abandonment specifically") naturally points to one clear primary metric.

**Organizational discipline / tooling**
- Companies with mature experimentation platforms (Google, Microsoft, Booking.com, Netflix) typically enforce pre-registration structurally — the experiment platform requires you to declare the primary metric and sample size before the test can even launch, removing the temptation to decide afterward.

**Number of hypotheses tested per experiment**
- The more secondary/exploratory metrics you allow yourself to inspect, the higher the informal false-positive risk unless you either (a) clearly separate confirmatory (primary, pre-registered) from exploratory (hypothesis-generating only, not decision-making) analysis, or (b) apply formal multiple-testing correction.

**Culture around "reruns" and iteration**
- If a team re-runs a "failed" test with minor tweaks until something is significant, without adjusting statistical thresholds for the repeated attempts, this reintroduces the same inflated false-positive risk pre-registration is meant to prevent — sometimes called the "iterative p-hacking" trap, distinct from but related to sequential peeking (Chapter 16).

## 5. Famous Q&A (Google / Apple style)

**Q: A test launches with "let's see how it affects our key metrics" as the stated hypothesis, and 15 metrics are tracked. The test ends with 2 of the 15 showing p < 0.05. Should you ship?**
A: This is a strong signal the experiment wasn't properly pre-registered. With 15 metrics tested at α = 0.05 each, you'd expect roughly 0.75 false positives by chance alone even if nothing real is happening — finding 2 "significant" results isn't strong evidence of anything without knowing (a) which metric was pre-specified as primary, and (b) whether a multiple-comparisons correction was applied. Before shipping, I'd ask: was one of these 2 metrics the pre-declared primary metric? If not, I'd treat this as hypothesis-generating (something to test as a *primary* metric in a follow-up, properly powered experiment) rather than a ship decision.

**Q: Why does deciding "one-sided vs. two-sided test" need to happen before running the experiment rather than after seeing the direction of the effect?**
A: Because choosing the test direction based on the observed sign of the effect is a form of p-hacking — it effectively halves your p-value threshold's true stringency after the fact, since you're only ever "choosing" the side that makes your result look better. A one-sided test should only be used when you have a genuine, pre-specified reason to only care about one direction (e.g., you'd never ship a change that makes the product worse, so you only test for improvement) — and that decision has to be locked in before the data is observed, or the Type I error guarantee is broken.

**Q: An engineer says "we don't need to pre-register — we'll just look at the data with an open mind and see what's true." What's the flaw in this reasoning?**
A: The flaw is that "looking with an open mind" doesn't prevent unconscious cherry-picking — humans are very good at post-hoc rationalizing why the metric that moved favorably is "the one that really matters," even without deliberate dishonesty. Pre-registration isn't about distrust of the analyst's intentions; it's a structural safeguard against a well-documented cognitive bias (sometimes called "the garden of forking paths") where the true false-positive rate of an analysis balloons whenever the analysis plan is flexible and decided after seeing the data, even if each individual step seems reasonable in isolation.

**Q: A PM wants to add a "we'll also check retention at 90 days" as an afterthought once the 2-week experiment concludes. What would you say?**
A: I'd point out that checking 90-day retention wasn't part of the original design — the experiment likely wasn't run long enough or powered to reliably detect an effect on a metric with that much longer a measurement horizon, and adding it after the fact as a new "primary" criterion is a form of moving the goalposts. If 90-day retention is genuinely important, I'd suggest treating it as an input to designing a *new*, properly powered long-horizon holdout experiment (see Chapter 19 — long-term holdouts) rather than retrofitting it onto results from a test that wasn't built to answer that question.

---
*Next: Chapter 9 — Sample Size & Power Calculations (MDE, baseline variance).*
