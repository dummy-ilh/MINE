# Chapter 7 — Partial Dependence & Individual Conditional Expectation (PDP/ICE)

Chapters 4–6 covered attribution methods — assigning credit for a prediction to individual features. This chapter is a different lens entirely: instead of "how much did this feature contribute," it asks **"how does the model's prediction change as this one feature varies, everything else being what it is?"** This is a dependence/relationship question, not an attribution question — and it comes with its own, related-but-distinct correlated-feature problem.

## 7.1 Partial Dependence Plots (PDP) — the definition

**What a PDP shows:** for one chosen feature, PDP plots the model's **average predicted output** as that feature is swept across its range of values, while every other feature is left at its actual, real values from the dataset — averaged over the whole dataset.

**The formal construction:** to compute the PDP value at a specific value x_s of feature s:
1. Take every example in your dataset.
2. For each one, artificially replace its actual value of feature s with the fixed value x_s, leaving every other feature at that example's real, original values.
3. Get the model's prediction for every one of these modified examples.
4. Average all of these predictions together — that average is the PDP's height at x_s.
5. Repeat across a range of x_s values to trace out the full curve.

**Plain-language reading:** *"If everyone in the dataset had this exact value for feature s, but kept everything else about them the same, what would the model predict, on average?"*

## 7.2 Why PDPs can be misleading under correlated features — the extrapolation problem, again

This is the exact same extrapolation issue from permutation importance (Chapter 4, §4.5), showing up in a new guise, and it's worth recognizing the pattern rather than treating it as a brand-new problem each time.

**The mechanism:** step 2 of the PDP construction artificially sets feature s to x_s for *every* example, **regardless of what values that example's other, correlated features hold**. If s is correlated with another feature (recall the `age` / `years_of_work_experience` example from Chapter 4), this can create the same kind of unrealistic combinations — e.g., forcing `years_of_work_experience = 35` onto an example whose `age` happens to be 22, a combination that never occurs in real data.

**The consequence:** the PDP curve partly reflects genuine model behavior with respect to feature s, and partly reflects the model's arbitrary, essentially made-up behavior on unrealistic feature combinations it never encountered during training — exactly the same underlying issue as permutation importance's extrapolation problem, just manifesting as a distorted *curve shape* here rather than a distorted single importance number.

## 7.3 Individual Conditional Expectation (ICE) — showing what a PDP's averaging hides

**The idea:** instead of averaging across the whole dataset (step 4 of the PDP construction), plot **one curve per individual example**, each showing how *that specific example's* prediction changes as feature s is swept across its range, with every other feature held at that specific example's own real values. A PDP is literally the average of all the individual ICE curves.

**What ICE reveals that a PDP alone hides: heterogeneity.** A PDP can show a flat, unremarkable average trend even when the underlying individual relationships are wildly different across subgroups — e.g., feature s might have a strongly *positive* effect for one subset of examples and a strongly *negative* effect for another subset, and if those effects happen to roughly cancel out on average, the PDP curve looks nearly flat, hiding the fact that s matters a great deal, just in opposite directions for different people. Plotting the individual ICE curves would immediately reveal this fanning-out pattern — some curves rising, others falling — that the single averaged PDP curve completely obscures.

**Concrete illustration:** suppose feature s is `monthly_marketing_spend`, and its effect on `conversion_rate` genuinely depends on customer segment — for new customers, more spend strongly increases conversion; for existing loyal customers, more spend has essentially no effect (they'd have converted anyway). A PDP averaged across both segments might show a mild, unremarkable positive slope. The ICE plot would show two clearly distinct clusters of curves — steep upward lines for new customers, flat lines for existing customers — immediately suggesting that a segment-specific marketing strategy, not a one-size-fits-all spend level, is what the model is actually capturing.

## 7.4 Accumulated Local Effects (ALE) — fixing the extrapolation problem directly

**The core fix ALE applies:** instead of forcing feature s to a fixed value x_s across *every* example regardless of their other feature values (which is exactly what creates PDP's extrapolation problem), ALE only ever asks the model about examples using feature values **close to their own actual, real values** — it computes the model's *local* sensitivity to small changes in s, evaluated only within realistic, in-distribution neighborhoods, and then accumulates (sums up) these local, realistic effects across the range of s to build up the final curve.

**The construction, at a conceptual level:** divide feature s's range into small intervals. Within each interval, take only the examples whose actual value of s falls in that interval (so you're not forcing unrealistic values onto examples that don't naturally have them), and measure how much the prediction changes, on average, as s moves across just that small interval, using each example's own real values for every other feature. Sum (accumulate) these small, local, realistic changes across all the intervals to build the full ALE curve.

**Why this specifically avoids PDP's problem:** because ALE never asks the model to evaluate an example with a feature-s value far from that example's neighborhood, it never constructs the kind of unrealistic combination (e.g., "age=22, years_of_work_experience=35") that distorts a PDP under correlated features — every evaluation ALE performs stays within realistic, locally-observed combinations of feature values.

**The tradeoff:** ALE curves are somewhat less intuitive to explain than a PDP ("the accumulated local effect" is a less immediately graspable concept than "the average prediction if everyone had this value"), and the resulting curve represents *differences* from a baseline rather than absolute predicted values in the same direct way a PDP does — but this cost is generally worth paying whenever your features are meaningfully correlated, which in real tabular data is close to the norm rather than the exception.

## 7.5 Choosing among PDP, ICE, and ALE

- **PDP:** use when features of interest are roughly uncorrelated with the rest of the feature set (the extrapolation problem is minimal in that case), and you want the most intuitive, easily-explained "average predicted outcome as this feature varies" curve.
- **ICE:** use alongside a PDP whenever you suspect the effect of a feature might differ meaningfully across subgroups — ICE is a diagnostic for heterogeneity that a PDP alone cannot reveal, and should be a routine companion plot rather than a special-occasion one.
- **ALE:** use whenever features are meaningfully correlated (again, close to the norm for real tabular data) and you specifically want a dependence plot that doesn't suffer from PDP's extrapolation distortion — at the cost of a slightly less intuitive interpretation.

## 7.6 Quick self-check before Chapter 8

- Can you explain, step by step, why forcing a fixed feature value onto every example (as a PDP does) can create unrealistic feature combinations — using a concrete pair of correlated features as your example?
- Can you describe a scenario where a PDP looks flat/unremarkable but the underlying ICE curves reveal something important the PDP completely hides?
- Can you explain, at a conceptual level, why ALE's "only look at local neighborhoods" construction avoids the extrapolation problem that PDP has?

---

**Next: Chapter 8 — Global vs. Local, and Aggregating Local Explanations**, formally distinguishing global and local importance, covering how SHAP values get aggregated into a global ranking and what's lost in that aggregation, and how to measure the stability of an importance ranking across resamples.
