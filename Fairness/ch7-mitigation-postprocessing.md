# Chapter 7 — Mitigation: Post-processing

Chapters 5 and 6 changed the data or the training objective — both require retraining. This chapter covers the third stage: adjusting a model's decisions **after** it's already trained, without touching the model weights at all. You take a frozen model's output scores S and change only how you convert them into final decisions Ŷ.

## 7.1 Threshold adjustment per group

**The idea:** instead of applying one global threshold to convert score S into decision Ŷ (e.g., "approve if S ≥ 0.5" for everyone), apply a **different threshold for each group**, chosen specifically so that the resulting Ŷ satisfies whichever fairness metric you've targeted (Chapter 3).

**Why this can work even though the model itself is unchanged:** the model's *scores* S might already be reasonably calibrated or informative per group, but a single global threshold can still produce unequal TPR/FPR because the *score distributions* differ by group (e.g., if Group B's scores are systematically a bit lower on average, a single cutoff will catch fewer of Group B's true positives). Adjusting the threshold per group directly targets the TPR/FPR gap without needing the underlying score to change at all.

**Worked example, extending Chapters 2 and 3's numbers.** Suppose after training, you have the raw scores for Group A and Group B, and you're currently using a single global threshold of 0.50, producing the TPR/FPR numbers from Chapter 2 (TPR(A)=0.75, TPR(B)=0.60). You want to equalize TPR across groups (targeting equal opportunity, from Chapter 3 §3.3).

Because Group B's score distribution sits a bit lower, you find that **lowering Group B's threshold to 0.42** (while keeping Group A's threshold at 0.50) shifts a few of Group B's borderline true positives from Ŷ=0 to Ŷ=1 — say this recovers 4 of the 8 previously-missed true positives in Group B:

New TP(B) = 12 + 4 = 16, new FN(B) = 8 − 4 = 4
New TPR(B) = 16 / (16+4) = 16/20 = **0.80**

That's now slightly *above* Group A's 0.75 — so in practice you'd search for the threshold that lands closest to 0.75 exactly (say, a threshold of 0.44 recovers only 3 of the 8, landing TPR(B) at 15/20 = 0.75, an exact match). The general procedure is: sweep each group's threshold, recompute the confusion matrix, and pick the threshold values that minimize the remaining gap in your target metric.

**Important side effect to always mention:** shifting Group B's threshold down doesn't only change TPR — it also changes FPR (more of Group B's true negatives may now get wrongly flagged too, since a lower threshold catches more of *everyone* above it, not just true positives). This is Chapter 3's impossibility result showing up again in a very concrete, practical way: fixing one metric (TPR) via threshold adjustment can move another metric (FPR) further out of alignment, if the group base rates differ. You need to explicitly check the metric you *didn't* target, not just the one you did.

## 7.2 Calibration post-hoc adjustment

**The idea:** if a model's raw scores are well-calibrated overall but *not* per group (recall Chapter 2 §2.5 — a score of 0.7 doesn't mean "70% true positive rate" equally well in every group), you can apply a separate recalibration function per group — e.g., **Platt scaling** (fit a logistic function mapping raw score to calibrated probability) or **isotonic regression** (fit a monotonic step function), fit *separately within each group* rather than on the pooled population.

**Why fit separately per group:** a single pooled calibration curve, fit across everyone, will by construction be "average-correct" but can still be systematically off for any individual group whose score distribution or true positive rate differs from the pooled average — exactly the situation calibration-across-groups (Chapter 3 §3.4) is defined to catch. Fitting the recalibration mapping separately for each group directly forces P(Y=1|S=s, A=a) = s to hold within each group, at the cost of now having group-specific score-to-probability mappings (which raises the same "are you allowed to use A at inference time" question covered in 7.4).

## 7.3 Reject option classification

**The idea:** for individuals whose score falls in an ambiguous band near the decision threshold, don't force a hard Ŷ=0/1 decision at all — route them to a human reviewer, or apply the more favorable outcome to disadvantaged groups only within that ambiguous band, rather than everywhere.

**Why this specifically targets fairness:** most of the disagreement between fairness metrics (Chapter 3) concentrates near the threshold — individuals confidently far from the boundary are rarely the source of a TPR/FPR/PPV gap; it's the close calls that are. By treating the close-call band specially (human review, or a favorable tie-break rule for a disadvantaged group specifically within that band), you can meaningfully narrow a fairness gap while only affecting a small, genuinely-ambiguous slice of the population — rather than shifting a threshold globally and affecting everyone above/below it.

## 7.4 Tradeoffs: the legal/ethical sensitivity of post-processing

Post-processing has real practical advantages: **no retraining required**, it's fast to implement, and it lets you adjust the fairness/accuracy tradeoff after deployment as requirements change (contrast this with in-processing from Chapter 6, where changing λ or ε generally means retraining from scratch).

But it has one recurring, serious catch that shows up in almost every interview discussion of this technique: **threshold adjustment and per-group calibration both require using the protected attribute A explicitly at inference time** — you need to know someone's group membership *at the moment you're making the decision* in order to know which threshold or recalibration function to apply to them.

This raises a genuinely thorny issue, not just a technical footnote:
- In many regulated domains (lending, employment, insurance), it may be **illegal** to use a protected attribute directly in a decision, even if the *purpose* is to make the outcome fairer (this is the "disparate treatment vs. disparate impact" distinction in US anti-discrimination law — deliberately treating people differently *because of* a protected attribute, even for a benevolent reason, can itself be treated as a form of discrimination under some legal frameworks).
- Even where it's legal, it can be hard to explain to stakeholders or the public why two people with the *identical* score received different decisions — the explanation ("we apply different thresholds to compensate for a group-level disparity") is defensible, but it needs to be documented and justified explicitly, which is exactly what Chapter 9's model documentation practices are for.

**The interview-ready framing:** "post-processing is the cheapest and most flexible mitigation stage, but it's also the one most likely to trigger a 'wait, are you allowed to do that?' conversation with legal/compliance — which is precisely why it needs to be paired with the documentation and governance practices in Chapter 9, not applied quietly."

---

**Next: Chapter 8 — The Fairness/Accuracy Tradeoff**, where we step back from individual techniques and reason quantitatively about how much accuracy any of these mitigations typically cost, and how to frame that tradeoff well in an interview setting.
