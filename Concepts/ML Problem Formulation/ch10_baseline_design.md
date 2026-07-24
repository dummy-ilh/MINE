# Chapter 10: Baseline Design

## 1. Why This Chapter Exists

Every model needs something to beat, and the choice of *what* counts as "beating baseline" is itself a formulation decision that gets surprisingly little attention compared to model architecture. A weak or ill-chosen baseline makes a mediocre model look impressive; a well-chosen baseline forces honesty about whether ML is even adding value, and by how much. This chapter is about designing baselines rigorously enough that "we beat baseline" is a meaningful claim.

## 2. The Baseline Hierarchy

Baselines exist on a spectrum of sophistication, and a rigorous evaluation reports *several*, not just one:

1. **Trivial/constant baseline**: predict the global mean (regression) or majority class (classification) for every example, ignoring all features.
2. **Simple heuristic baseline**: a rule a domain expert would write by hand — e.g., "flag transactions over $1,000 from a new device" for fraud, or "recommend the most popular item in this category" for recommendations.
3. **Simple statistical/linear baseline**: logistic regression or linear regression on the same features you'd feed the full model, with minimal feature engineering.
4. **Existing production system** (if replacing one): the current heuristic or older model actually running today — this is often the *only* baseline that matters for a launch decision, since it's the honest counterfactual.
5. **Full proposed model**: the thing you actually want to ship.

**The critical point**: your proposed model needs to beat #4 (or #3, if there's no existing system) by a *margin that justifies its added complexity, latency, and maintenance cost* — not just beat #1, which is a very low bar that almost anything clears.

## 3. Worked Numerical Example: A Baseline That Exposes an Overstated Win

Suppose a team reports "our new deep learning fraud model achieves 0.91 AUC-PR, a huge win!" Let's build the baseline hierarchy:

| Baseline | AUC-PR |
|---|---|
| Trivial (predict base rate for everyone) | 0.05 (≈ true positive rate, uninformative) |
| Heuristic rule (amount + device-age flag) | 0.62 |
| Logistic regression, same features | 0.85 |
| Current production model (gradient-boosted trees) | 0.89 |
| **Proposed deep learning model** | **0.91** |

Framed against the trivial baseline, the "win" looks enormous (0.05 → 0.91). Framed against the *actual* relevant baseline — the current production model at 0.89 — the improvement is +0.02 AUC-PR, a real but modest gain. Whether +0.02 justifies a full model migration (new serving infrastructure, higher inference latency, harder-to-debug predictions) is a legitimate cost-benefit question that the trivial-baseline framing completely obscures. **This is precisely why reporting only the trivial baseline is a red flag in an interview or a real launch review** — it inflates the perceived value of the proposed system by comparing it to a strawman nobody would actually deploy.

## 4. What "Beating Baseline" Should Actually Account For

A rigorous "beat baseline" claim needs three things beyond a single point-estimate comparison:

- **Statistical significance**, given evaluation set size — a +0.02 AUC-PR difference on a test set of 500 examples may not be distinguishable from noise; the same gap on 500,000 examples might be highly significant. Report a confidence interval or a paired significance test (e.g., bootstrap resampling), not just the point estimate.
- **Cost-adjusted comparison** — using Chapter 5's cost-weighted metrics, not just a generic AUC number, since a 0.02 AUC-PR gain might correspond to a large or negligible real dollar impact depending on where in the precision-recall curve the improvement occurs.
- **Complexity-adjusted framing** — explicitly stating the added latency, infrastructure, and maintenance cost of the proposed system relative to the baseline, so the improvement can be weighed against its true total cost, not evaluated on accuracy alone.

## 5. The "Is ML Even Needed?" Check

An underrated part of baseline design: sometimes the heuristic baseline (#2 above) captures nearly all the achievable value, and the honest conclusion is "don't build the ML system yet." **Numeric illustration**: if the heuristic baseline (0.62 AUC-PR) and the full deep model (0.91) are compared, that's a large gap and clearly justifies ML investment. But if a *different* problem shows heuristic at 0.88 and the best achievable ML model at 0.90, the story changes — a 0.02 gain from a full ML system, versus a simple, cheap, fully-interpretable heuristic at 0.88, might not be worth the added complexity, especially if the heuristic is easier to explain to regulators/auditors (relevant in domains like credit or fraud where explainability has real compliance value, not just a technical nice-to-have). Stating this tradeoff explicitly, even when the answer is "yes, build the ML model," demonstrates the kind of judgment that distinguishes senior formulation thinking from reflexively defaulting to "more complex model wins."

## 6. Production Considerations

- **Baselines need to be re-run on the same current data as the proposed model, not quoted from an old evaluation** — a baseline's reported performance from six months ago on stale data is not a valid comparison point for a model being evaluated on current data, since the underlying distribution may have shifted for both.
- **The "existing production system" baseline (#4) should be evaluated with the exact same evaluation protocol as the proposed model** — different train/test splits, different label definitions, or different time windows between the two evaluations invalidates the comparison entirely, even if both numbers look plausible in isolation.
- **Heuristic baselines (#2) often need to be maintained as a fallback/sanity-check even after a model is shipped** — if the ML model's serving pipeline fails or an input is wildly out-of-distribution, a documented, still-functioning heuristic path is a valuable production safety net, not just a historical comparison artifact.
- **Reporting baseline comparisons should be standard practice in any model launch review**, not just an interview exercise — the same "beat the real baseline, with significance, cost-adjusted" bar from Section 4 applies to actual production launch decisions.

## 7. Common Interview Traps

- **Comparing only against a trivial baseline** and presenting the resulting large-looking gap as evidence of a strong model, without acknowledging that the real relevant comparison is against the existing production system or a well-built heuristic (Section 3's exact trap).
- **Reporting a point-estimate improvement with no significance context** — an interviewer probing "how do you know that's not just noise?" is testing for Section 4's statistical-significance awareness.
- **Never considering the possibility that a simple baseline is "good enough"** — always assuming the answer must be "and therefore we build the complex model," missing Section 5's judgment call entirely.
- **Ignoring complexity/maintenance cost when reporting a win** — treating "higher AUC" as automatically justifying deployment regardless of the added system complexity.

## 8. L5-Differentiating Talking Points

- Proactively construct the full baseline hierarchy (Section 2) rather than reporting a single baseline number, and explicitly identify which baseline in the hierarchy is the *actually relevant* one for the launch decision.
- Raise statistical significance and cost-adjusted framing (Section 4) unprompted whenever quoting an offline metric improvement — this is a strong, easy-to-demonstrate signal of rigor.
- Explicitly entertain the possibility that a heuristic baseline might be "good enough," and state what evidence would change your mind either way (Section 5) — showing you're not reflexively biased toward the more complex solution.
- Note that baselines require ongoing maintenance and re-evaluation on fresh data (Section 6), not just a one-time comparison at model launch.

## 9. Comprehension Checks

1. A team reports "our model achieves 85% accuracy, beating the 50% random-guess baseline by 35 points." What's wrong with this comparison, and what baseline should have been reported instead, using Section 2's hierarchy?
2. Using Section 3's worked example, explain why a +0.02 AUC-PR improvement over the current production model might or might not justify a full model migration, and list the three factors from Section 4 you'd want quantified before deciding.
3. Describe a scenario (different from Section 5's) where the honest conclusion from baseline comparison is "don't build the ML model," and justify it using both the accuracy gap and a non-accuracy factor (e.g., interpretability, maintenance cost, latency).
4. Why does re-running an old baseline on stale data invalidate a comparison against a newly-evaluated proposed model, even if both individual numbers seem reasonable? Use Section 6's reasoning.

---

*End of Chapter 10. Chapter 11 will cover feedback loops and selection bias — how deployed models reshape their own future training data, and counterfactual framing to correct for it.*
