# Chapter 7: Instrumental Variables (IV)

## 1. Explanation

### When you need this tool

Every method so far — regression adjustment, PSM, IPW — requires ignorability *given observed X*: you must be able to measure and correctly model every confounder. IV is the tool for the situation where you strongly suspect there's an **unmeasured confounder** — something driving both treatment and outcome that you cannot observe, and therefore cannot control for no matter how sophisticated your model. In that situation, no amount of regression, matching, or weighting on observed covariates can fix the bias, because the bias-causing variable was never in your dataset to begin with.

### The trick, built intuitively

Find a third variable $Z$ — an "instrument" — that shifts treatment $D$ around, but for reasons that have **nothing to do with** the unmeasured confounder, and that has **no direct effect on Y** except *through* its effect on D. If such a $Z$ exists, then the variation in D that's driven by Z is "clean" — untainted by the confounder — and you can use just that slice of variation to estimate the causal effect, effectively ignoring the "dirty," confounded variation in D that comes from other sources.

Picture it as a DAG:
```
Z --> D --> Y
       ^
       |
       U (unmeasured confounder) --> Y  and --> D
```
Notice: there is no arrow from Z directly into Y, and no arrow connecting Z to U. Z only reaches Y by first passing through D.

### The three conditions, explained as causal claims (not just a checklist to memorize)

**1. Relevance**: the $Z \to D$ arrow must exist and be non-trivial — Z must actually move D by a meaningful amount. This is the *one* assumption that's empirically checkable: regress D on Z (the "first stage") and confirm the coefficient is far from zero, with a strong F-statistic.

**2. Exclusion restriction**: there must be no arrow from Z to Y that bypasses D entirely. This is **not testable from data** — it's a substantive, structural claim about the world that you must argue using domain knowledge, institutional detail, and logic, not statistics. This is usually the assumption interviewers push hardest on, because it's the one most often quietly violated in real applications.

**3. Independence/exogeneity**: Z must not share any common cause with Y other than through D — i.e., Z itself must be "as good as randomly assigned" with respect to the unmeasured confounder U. This is often satisfied by design if Z comes from genuine randomization (a lottery) or a strongly-argued "as-if random" natural event (an administrative quirk, weather, a policy discontinuity).

### Building the estimator step by step (the Wald estimator)

You want to isolate "the effect of moving D by one unit, using only Z-driven variation." Two pieces:
- **Numerator**: how much does Y differ across values of Z? This is the "reduced form" effect — the total observed effect attributable to the instrument, mixing together "how much Z moves D" and "how much D moves Y."
- **Denominator**: how much does D itself differ across values of Z? This is the "first stage" — literally how much of D's variation Z is responsible for.

Dividing rescales the numerator into a per-unit-of-D effect:
```
τ_IV = [E(Y|Z=1) − E(Y|Z=0)] / [E(D|Z=1) − E(D|Z=0)]
```
In the linear, multi-variable-controls case, this generalizes to **Two-Stage Least Squares (2SLS)**: (Stage 1) regress D on Z and any controls, obtaining predicted values $\hat{D}$; (Stage 2) regress Y on $\hat{D}$ (and the same controls) — the coefficient on $\hat{D}$ is the IV estimate. Using the *predicted* D (driven only by Z and controls, not by the unmeasured confounder) is precisely what isolates the "clean" variation.

### Why this only ever gives you LATE — and it's not an assumption, it's a mathematical fact

The units that contribute to your denominator, $E(D|Z=1) - E(D|Z=0)$, are exactly the **compliers** — people whose treatment status *actually flips* depending on the value of Z. "Always-takers" (who take treatment regardless of Z) and "never-takers" (who refuse treatment regardless of Z) contribute *zero* variation to that denominator — their D value doesn't move when Z moves. This means the estimator has no information whatsoever about their causal effect; it's not that you're *assuming* they're similar to compliers, it's that they are mathematically invisible to this particular estimation strategy. This is the origin of **LATE (Local Average Treatment Effect)** — "local" to the complier subpopulation defined by this specific instrument.

### Weak instruments — a serious, common, practical failure mode

If Z only barely moves D (a small first-stage coefficient, denominator close to zero), the IV estimator becomes extremely unstable: small sampling noise in a near-zero denominator gets massively amplified into huge swings in the final estimate. Worse, in finite samples, weak-instrument 2SLS estimates are known to be biased *toward* the plain, confounded OLS estimate — ironically undoing the entire benefit you sought from using IV in the first place. The standard diagnostic: check the first-stage F-statistic; a common rule of thumb flags F < 10 as a weak-instrument warning sign.

## 2. Example

### Example A — Classic setup, fully worked

Effect of military service (D) on later earnings (Y), instrumented by draft lottery number (Z, binary: drafted vs. not) — the classic Angrist-style design. The concern: people who chose to serve voluntarily may differ systematically (unmeasured ambition, health, family background — call it U) from those who didn't, confounding a naive D-Y comparison. The draft lottery, by contrast, was explicitly randomized.

Suppose:
- $E[Y \mid Z=1 \text{ (drafted)}] = 38{,}000$ (average earnings of the drafted group, including everyone regardless of whether they actually served)
- $E[Y \mid Z=0 \text{ (not drafted)}] = 40{,}000$
- $E[D \mid Z=1] = 0.30$ (30% of drafted individuals actually serve — some get exemptions)
- $E[D \mid Z=0] = 0.05$ (5% volunteer anyway, despite not being drafted)

```
τ_IV = (38,000 − 40,000) / (0.30 − 0.05)
     = −2,000 / 0.25
     = −8,000
```
**Interpretation**: for the compliers — people who serve *specifically because* they were drafted, and who would not have served otherwise — military service causes an **$8,000 decrease** in earnings. This says nothing directly about always-takers (who'd have volunteered regardless) or never-takers (who'd have avoided service regardless, e.g., via exemption) — a common interview trap is to ask "so military service reduces everyone's earnings by $8,000?" — the correct answer is no, this is a complier-specific (LATE) estimate.

### Example B — A Google-flavored IV scenario, showing the exclusion-restriction argument explicitly

Question: does receiving a **customer support callback** (vs. chat-only support) (D) causally improve subsequent subscription retention (Y)? Naive comparison is confounded by unobserved **case severity** (U): more severe issues get escalated to callback *and* independently predict future churn regardless of how the case was resolved.

Instrument idea: due to an unrelated staffing/scheduling quirk, a random subset of support requests submitted **between 2-3pm on Tuesdays** get auto-routed to callback, for reasons entirely about agent shift overlap — not about the case content.

Data:
- $E[Y \mid Z=1 \text{ (2-3pm Tuesday slot)}] = 0.82$ (82% retained)
- $E[Y \mid Z=0 \text{ (other times)}] = 0.78$
- $E[D \mid Z=1] = 0.55$
- $E[D \mid Z=0] = 0.30$

```
τ_IV = (0.82 − 0.78) / (0.55 − 0.30) = 0.04 / 0.25 = 0.16
```
**Interpretation**: for the compliers — support requests routed to callback *specifically because* they happened to land in the 2-3pm Tuesday slot, not because of underlying case severity — receiving a callback causes a **16 percentage point increase** in retention.

**The exclusion-restriction argument you'd need to make explicit in an interview**: "I'm assuming the 2-3pm Tuesday routing quirk affects retention *only* by changing whether someone gets a callback — not through any other channel. To make this plausible, I'd check that case topics/severity indicators are balanced across the 2-3pm slot vs. other times (an indirect, partial check — even though it can't fully prove exclusion, since I can never observe every possible alternate pathway). If, say, 2-3pm Tuesday happened to also coincide with a different, more experienced shift of agents handling *all* channels (not just callback), that would violate exclusion — the timing would affect retention through agent quality too, not just through callback receipt."

## 3. Interview Q&A

**Q: Why is the exclusion restriction the "hardest" of the three IV assumptions to defend, and how would you argue for it in a real setting?**
A: It requires that the instrument has *no* pathway to the outcome except through the treatment — but you can never fully verify the absence of all such pathways statistically; you have to argue it from institutional/domain knowledge. In the callback example, I'd argue the 2-3pm-Tuesday routing quirk is a scheduling/staffing artifact plausibly uncorrelated with anything about the customer's case content or severity — while also proactively checking whether case topics/severity indicators are balanced across the 2-3pm slot vs other times, as an indirect (not conclusive) supporting check.

**Q: What happens to a 2SLS estimate when the instrument is "weak" (small first-stage coefficient), even if all three IV assumptions technically hold?**
A: The estimator's variance blows up (imprecise, unstable estimates), and more insidiously, the estimator becomes biased *toward* the plain, confounded OLS estimate in finite samples — undermining the entire point of using IV in the first place. Always check the first-stage F-statistic (rule of thumb: want F > 10) before trusting a 2SLS result.

**Q: You have a valid instrument but a large fraction of "always-takers" in your population. What's the practical implication for using this result to make a policy decision?**
A: LATE only describes compliers — if always-takers/never-takers make up most of the population, your IV estimate describes a small, possibly unrepresentative slice, and extrapolating it to "the effect of a universal policy" (e.g., mandating callbacks for everyone) is not justified by this analysis alone. You'd want to separately reason about, or find different sources of variation for, the always-taker/never-taker subpopulations.

**Q: Contrast IV with regression adjustment in terms of what kind of confounding each one can handle.**
A: Regression adjustment (and PSM/IPW) can only handle confounding by **observed** variables — you must be able to measure and correctly model the confounder. IV can handle confounding by **unobserved** variables, as long as you can find a valid instrument — but at the cost of a much harder-to-satisfy, non-testable exclusion assumption, and a narrower (LATE, not ATE) estimand than you'd get from a method that could condition on the full confounder.

**Q: How would you empirically test the "relevance" assumption, and is it fully sufficient on its own?**
A: Regress D on Z (the first stage) and check both the coefficient's magnitude and its F-statistic — this is directly testable and should always be reported alongside the IV estimate. It's necessary but not sufficient: a strong first stage doesn't rescue a violated exclusion restriction. You need both relevance (testable) and exclusion (untestable, argued) to hold for the IV estimate to be valid.

**Q: In Example A, why can't you simply say "military service reduces earnings by $8,000 for everyone"?**
A: Because the $8,000 estimate is identified purely from the variation among compliers — those whose service status was actually changed by being drafted. Always-takers (who'd serve regardless of the draft) and never-takers (who'd avoid service regardless) contribute nothing to the estimator's denominator, so it carries no information about their causal effect; extrapolating the complier-specific LATE to the whole population is an unjustified LATE-to-ATE overreach, a very common interview trap question.

**Q: If you found a *second*, different valid instrument for the same treatment, would you expect it to give you the same IV estimate as your first instrument? Why or why not?**
A: Not necessarily — different instruments typically define different complier populations (the specific subset of people whose treatment status responds to *that particular* instrument), and if treatment effects are heterogeneous across the population, two different instruments can validly produce two different LATEs, each correct for its own complier subpopulation. This is a good check for heterogeneity: if two plausible instruments give very different answers, it's informative about how much the effect varies across different types of "movable" units, not necessarily evidence that one estimate is wrong.

---
**Previous: Chapter 6 — Propensity Score Matching & IPW**
**Next: Chapter 8 — Difference-in-Differences**
