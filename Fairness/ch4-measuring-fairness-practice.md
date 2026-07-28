# Chapter 4 — Measuring Fairness in Practice

Chapters 2 and 3 gave you the math. This chapter is about the messier, more practical questions that come right after: *which* metric do you actually pick for a given problem, how do you slice your population to find hidden disparities, and how do you avoid fooling yourself with noise when your slices get small. This is also where interviewers like to probe — "okay, you know the definitions, but how would you actually set this up?"

## 4.1 Choosing a metric based on the use case

Chapter 3 showed you can't have everything. So the real skill is: given a scenario, which metric is the *right* tradeoff to make? There's no universal answer, but there is a reliable way to reason about it — ask what kind of error is more costly, and to whom.

**Lending:** A false positive (approving someone who defaults) costs the bank money. A false negative (denying someone who would have repaid) costs that person an opportunity, and — at scale, if concentrated in one group — reproduces exactly the historical exclusion problem from Chapter 1. Many practitioners lean toward **equal opportunity** (equal TPR — equal rate of correctly identifying "will repay") or full **equalized odds**, because approval decisions have direct, comparable-across-people financial consequences, and calibration miscalibration would mean interest rates/limits don't actually reflect real risk for a whole group.

**Hiring:** Similar logic to lending — a missed qualified candidate (false negative) is often considered worse than an extra interview slot spent on an unqualified one (false positive), which pushes toward **equal opportunity**. Some organizations instead explicitly want **demographic parity** at the top of the funnel (e.g., "our interview slate should reflect our applicant pool's diversity") as a policy choice, independent of the statistical argument — that's a legitimate business/values decision, not a math error, but it should be named as a *chosen* tradeoff rather than presented as "the fair option."

**Criminal justice risk scores:** This is the highest-stakes category, and it's exactly where COMPAS's fight played out. A false positive (flagging someone as high-risk who wouldn't reoffend) can mean extended detention or a harsher sentence — a direct harm to a specific person. A false negative can mean releasing someone who reoffends — a harm to potential future victims. Because both errors are severe and asymmetric in *who* bears them, **equalized odds** is the most commonly argued-for standard here, even though (as Chapter 3 showed) it will generally break calibration when base rates differ — which is precisely the unresolved tension in the real COMPAS debate.

**Ad targeting / content ranking:** Here the "positive" outcome (a click, a view) is low-stakes per-instance but the *aggregate* effect matters — e.g., job ads shown less often to one gender. **Demographic parity** in exposure/impressions is a common target, because the harm is about unequal *access* to information/opportunity, not about any single prediction being "wrong."

**The interview-ready pattern:** identify (a) what a false positive costs and who bears it, (b) what a false negative costs and who bears it, (c) whether the score itself will be interpreted by a human downstream (favors calibration) or just used as a binary gate (favors TPR/FPR-based metrics), then pick a metric and *name the tradeoff you're accepting*, rather than claiming you've eliminated unfairness.

## 4.2 Intersectional slicing

Real populations aren't just "Group A vs Group B" — people belong to multiple, overlapping groups (gender × race × age bracket × disability status, etc.). A model can look perfectly fair on each attribute *individually* while being badly unfair on a specific *intersection*.

**Classic illustration (from the Gender Shades study in Chapter 1):** facial recognition error rates looked reasonable when sliced by gender alone, and reasonable when sliced by skin tone alone — but the error rate for the intersection of "darker-skinned" and "female" was dramatically worse than any single-attribute slice suggested. If you only check gender and skin tone separately, you miss the actual problem, which lives at their intersection.

**Practical takeaway:** always slice by combinations of protected attributes, not just each attribute in isolation, whenever your sample size allows it (see 4.3 below for the "whenever your sample size allows it" caveat, which is a real constraint, not a footnote).

## 4.3 The small-sample-size problem

Here's the tension that makes 4.2 hard in practice: the more finely you slice your population (more intersections, more subgroups), the smaller each slice gets — and small slices produce noisy metrics.

**Concrete illustration.** Suppose you slice down to a subgroup with only 25 people, 5 of whom are truly positive. Your TPR estimate for that subgroup is based on just 5 positive examples. If the model happens to miss 1 extra person in that tiny group purely by chance, your measured TPR swings by 20 percentage points (1 out of 5) — a huge apparent "disparity" that might just be sampling noise, not a real property of the model.

**What to do about it:**
- **Compute confidence intervals, not just point estimates**, on every fairness metric — a common approach is a normal approximation or bootstrap resampling to get an interval around, e.g., TPR per subgroup, so you can say "this gap is / isn't statistically distinguishable from zero given our sample size."
- **Set a minimum subgroup size** below which you report "insufficient data" rather than a potentially misleading point estimate.
- **Borrow statistical strength across related subgroups** when appropriate (e.g., hierarchical/Bayesian shrinkage), rather than treating each tiny intersection as fully independent — this stabilizes estimates for small groups by partially pooling information from related, larger groups.
- **Aggregate over time** — a subgroup that's small in one deployment window may be large enough once you accumulate several months of data, at the cost of slower detection.

**The interview-ready framing:** "the finer you slice, the more real disparities you can detect, but also the more noise you introduce — so any fairness measurement system needs an explicit sample-size/confidence-interval policy, not just a table of point estimates."

## 4.4 What a fairness dashboard/report typically contains

Bringing 4.1–4.3 together into something you'd actually build or describe in an interview:

1. **Overall model metrics** (accuracy, AUC, etc.) — the baseline everyone already tracks.
2. **Per-group metrics** for each protected attribute individually: TPR, FPR, PPV, approval rate, each with a confidence interval.
3. **Intersectional slices**, down to whatever granularity the sample size supports, flagged with a data-sufficiency indicator (e.g., grayed out or marked "low confidence" below a minimum n).
4. **Trend over time**, not just a single snapshot — a one-time audit can miss a metric that's slowly drifting due to the feedback loops from Chapter 1.
5. **A named "chosen metric"** for the specific use case (per 4.1), with the tradeoff explicitly documented — not just a wall of every possible number with no interpretation.
6. **Thresholds/alerts** — a policy for what gap size triggers a review (this ties directly into Chapter 9's governance material).

---

**Next: Chapter 5 — Mitigation: Pre-processing**, the first of three chapters covering how to actually fix a disparity once you've measured one — starting with techniques applied to the data before training even begins.
