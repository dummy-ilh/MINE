# Chapter 12: Sensitivity Analysis, Placebo Tests, and Quasi-Experiments

## 1. Explanation

### The problem this chapter addresses

Every observational method covered so far — regression adjustment (Ch.5), PSM/IPW (Ch.6) — rests on ignorability: "no unmeasured confounders." As established back in Chapter 4, this assumption is, by definition, **impossible to verify directly from the data you have** — if you could measure the confounder, you'd simply control for it, and it wouldn't be "unmeasured" anymore. Given this permanent limitation, mature causal inference practice isn't about pretending the assumption is proven. It's about **quantifying how fragile your conclusion is** to violations of that assumption, and running falsification checks designed to catch obvious pipeline problems before they mislead a business decision.

### E-value: reasoning about it before the formula

Ask yourself: "how strong would a *hypothetical* unmeasured confounder need to be — in terms of how strongly it's associated with *both* treatment and outcome — to fully explain away my observed effect, even after I've already controlled for everything I *did* measure?" If the honest answer is "extremely strong, stronger than any plausible real-world confounder I can imagine in this context," your result is robust. If the answer is "even a weak, easily-imaginable confounder could do it," your result is fragile, and you should say so.

```
E-value ≈ RR + sqrt(RR × (RR−1))     for an observed (adjusted) risk ratio RR ≥ 1
```
(if your observed RR is less than 1, first take its reciprocal 1/RR, then apply the same formula). This is not a proof of "no confounding" — it's a **transparency and communication tool**. It forces an explicit, checkable statement — "a confounder this strong would be needed to overturn this" — rather than a vague, unfalsifiable assurance that you've "controlled for everything important."

### Rosenbaum bounds: the matching-specific analogue

Used alongside matching (Chapter 6), Rosenbaum bounds ask a related but distinctly-framed question: "how much would an unmeasured confounder need to differentially affect the *odds of receiving treatment* between two matched units — who look identical on all observed covariates — before my statistically significant result would stop being significant?" This is expressed via a sensitivity parameter $\Gamma$ (e.g., "results are robust up to $\Gamma=2$," meaning an unmeasured confounder would need to *double* the odds of treatment between matched pairs to overturn statistical significance). The higher the $\Gamma$ needed to overturn your result, the more robust the finding is considered to be.

### Placebo and falsification tests: the general-purpose sanity-check family

These are cheaper, more direct checks that don't require formalizing a sensitivity bound — they're designed to catch a **broken analysis pipeline**, which is often a bigger practical risk than a subtle confounder.

- **Placebo outcome test**: apply your exact causal method to an outcome that treatment *couldn't plausibly* affect. If you find a "significant effect" there anyway, that's a red flag your method has some systematic bias (perhaps a confounder that happens to also affect this placebo outcome, or a bug in how groups were constructed) — not something you should chalk up to a genuine causal signal on your main outcome either, since the same broken machinery produced both.
- **Placebo treatment timing test**: pretend the treatment happened at some earlier point in time (using only data from before it actually occurred), and check that you find a null effect, as covered for DiD in Chapter 8. Finding a spurious "effect" at a fake date suggests your design has some source of bias unrelated to the real treatment.
- **Balance checks**: confirm that treated and control groups, after adjustment, look similar on *observed*, pre-treatment covariates. This can't verify that unobserved confounders are balanced too — but gross imbalance on *observed* variables is itself a red flag that your adjustment procedure isn't working properly, which should make you more (not less) worried that unobserved variables are also poorly balanced.

### Quasi-experiments and natural experiments: a mindset, not a new toolkit

A "natural experiment" is a real-world situation where variation in treatment happened for reasons **plausibly unrelated to potential outcomes**, even though nobody designed it as an experiment. This isn't a separate method from everything you've learned — it's a way of *finding* the raw material that the earlier chapters' tools can then be applied to:
- A policy's arbitrary implementation date, differing by jurisdiction for bureaucratic reasons → a **DiD** candidate.
- A lottery, or an arbitrary administrative eligibility cutoff → an **IV** or **RDD** candidate.
- A localized shock affecting just one region (a factory closure, a local regulation) → a **synthetic control** candidate.

The skill being tested here is recognizing *when* some real-world variation is plausibly "as-if random," and then correctly matching it to the right tool from Chapters 3–10 — not memorizing a new formula.

### What makes a natural experiment convincing versus flimsy

**Convincing**: the source of variation is plausibly unrelated to potential outcomes for reasons you can clearly and specifically articulate (an administrative rule tied to birth-year, a random lottery, a policy rollout driven by unrelated bureaucratic scheduling) — *and* you can show supporting evidence, like balanced pre-treatment covariates or smooth pre-trends, that's at least consistent with this story.

**Flimsy**: the "natural" variation is itself plausibly correlated with the outcome through some channel you haven't ruled out — e.g., a policy that phased in based on regions' *existing* economic conditions is no longer as-good-as-random, since the timing itself is confounded with exactly the kind of factors that would also affect your outcome.

## 2. Example

### A combined worked example — E-value alongside a real business decision

An observational analysis finds that users who enable "two-factor authentication" (2FA) have a churn risk ratio of $RR = 1.8$ relative to non-2FA users (2FA users churn *more*, even after adjusting for tenure, account type, and support-ticket history).

```
E-value = 1.8 + sqrt(1.8 × 0.8) = 1.8 + sqrt(1.44) = 1.8 + 1.2 = 3.0
```
This means an unmeasured confounder would need to be associated with *both* 2FA adoption and churn by a risk ratio of at least **3.0** (each) to fully explain away the observed 1.8 relationship, above and beyond what's already accounted for by tenure/account-type/tickets.

**Now reason about plausibility.** Is there a plausible confounder that strong? A strong candidate: users who were recently the target of a **security incident or account-recovery flow** might be *both* more likely to subsequently enable 2FA (a reactive security response) *and* more likely to churn afterward (frustrated by the incident itself, independent of 2FA). Is "recent security incident" plausibly a risk-ratio-≥3.0 confounder for both outcomes? Quite possibly, yes — security incidents are well known to be strongly associated with both defensive behavior changes and subsequent churn.

**Conclusion: this result is fragile.** A quite plausible, real-world confounder (recent security incidents, not fully captured by tenure/account-type/tickets) could plausibly fully explain the observed association on its own. I would not recommend concluding "2FA causes churn" or discouraging 2FA adoption based on this analysis alone. Instead, I'd (a) control explicitly for recent-security-incident history if that data exists, or (b) push for a randomized encouragement design (randomly nudging a subset of users to enable 2FA) to get a genuinely clean causal answer before making any product decision.

### A second example — placebo outcome test catching a broken pipeline

Suppose a search-ranking team estimates, via PSM, that a new ranking change "improves" a metric completely unrelated to search — say, in-app crash rate on a completely unrelated settings tab. There's no plausible mechanism connecting ranking changes to crashes on an unrelated screen. Finding a "statistically significant" effect there anyway is a **placebo-outcome red flag** — it strongly suggests something is broken in the underlying pipeline (e.g., the "treated" and "control" populations were constructed differently in some way unrelated to ranking itself, perhaps due to a logging or sampling artifact) rather than indicating the ranking change has some mysterious, unrelated causal power. The correct response is to investigate the experiment/analysis infrastructure before trusting *any* of that pipeline's results, including the main-metric finding it was built to measure.

## 3. Interview Q&A

**Q: A stakeholder asks "how confident are you there's no hidden confounder?" How do you answer without either overclaiming or being unhelpfully vague?**
A: I'd say something like: "I can't prove there's no hidden confounder — that's fundamentally unprovable from this kind of data. What I can tell you is how strong a hidden confounder would need to be to change our conclusion, using a sensitivity measure like the E-value, and then use domain judgment to assess whether a confounder that strong is plausible in this context." This gives a calibrated, honest answer rather than false certainty or unhelpful hand-wringing.

**Q: Why is a placebo-outcome test useful even though it doesn't directly validate your main result?**
A: It's a "canary in the coal mine" — if your method finds a spurious effect on something it obviously shouldn't affect, that's strong evidence your overall pipeline (data construction, matching procedure, model) has a systematic bias, which should make you distrust the main result too, even though passing the placebo test doesn't *prove* the main result is correct. Absence of a detected problem isn't proof of the absence of all possible problems, but presence of an obvious one is decisive evidence something is wrong.

**Q: What does a Rosenbaum bound of Γ=1.1 vs Γ=4 tell you differently?**
A: Γ=1.1 means it would take only a very mild unmeasured confounder (barely increasing the odds of treatment for one matched unit relative to its pair) to overturn your result's statistical significance — a fragile finding. Γ=4 means an unmeasured confounder would need to *quadruple* the odds of treatment between matched pairs to overturn significance — a much more robust finding, since few real-world confounders are typically that strong after already controlling for a rich set of observed covariates.

**Q: What makes a "natural experiment" convincing versus flimsy, in an interview answer?**
A: Convincing: the source of variation is plausibly unrelated to potential outcomes for a reason you can clearly articulate (e.g., an administrative rule based on a birth-year cutoff, a lottery, a policy rollout driven by unrelated bureaucratic timing), backed by supporting evidence like balanced pre-treatment covariates or smooth pre-trends. Flimsy: the "natural" variation is itself plausibly correlated with the outcome through some channel you haven't ruled out — e.g., a policy that phased in based on regions' *existing* economic conditions is no longer as-good-as-random with respect to that outcome.

**Q: If you only have time/budget to do ONE robustness check on an observational finding before presenting it, which would you pick and why?**
A: I'd generally pick a **placebo/falsification test** (placebo outcome or placebo timing) over a formal sensitivity bound like the E-value, because it's the cheapest, most direct way to catch a *broken pipeline* — bugs, sample construction errors, or obviously wrong confounding structure. A sensitivity bound is valuable but implicitly assumes your core method is already sound; a placebo test checks that foundational soundness first, before you even get to reasoning about subtler, harder-to-rule-out unmeasured confounders.

**Q: Two E-values are both calculated as 3.0, but you're much more comfortable trusting one result than the other. What could explain this, given the numbers are identical?**
A: The E-value is purely a mathematical translation of the observed risk ratio — it doesn't know anything about your specific domain. Your comfort level should depend on whether you can actually *name* a plausible confounder that meets or exceeds that threshold in each specific context. If, for one result, you genuinely can't think of any real-world factor that plausibly reaches a risk ratio of 3.0 with both treatment and outcome, that result is more trustworthy than one where you can immediately name a very plausible candidate (as in the 2FA/security-incident example) — the E-value is a starting point for domain reasoning, not a replacement for it.

**Q: How do quasi-experiments relate to the other nine methods in this guide — are they a separate, eleventh method?**
A: No — a quasi-experiment is a *source of variation* you've identified as plausibly "as-if random" in the real world (a lottery, an administrative cutoff, a localized policy change), which you then analyze using one of the standard tools already covered: DiD if it's a before/after comparison across groups, IV or RDD if it involves a lottery or sharp threshold, synthetic control if it's a single affected unit. The "quasi-experiment" label describes where the identifying variation came from, not a new estimation technique on top of what you already know.

---
**Previous: Chapter 11 — Interference and SUTVA Violations**
**Next: Chapter 13 — Google-Style Case Studies (Full Walkthroughs)**
