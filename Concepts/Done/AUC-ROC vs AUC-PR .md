# AUC-ROC vs AUC-PR — Interview Notes

## 1. Why These Exist At All

A single confusion matrix (and the precision/recall/F1 derived from it) is a snapshot at **one threshold**. AUC-ROC and AUC-PR summarize performance **across every possible threshold at once**, giving a threshold-independent view of how well the model separates classes by score, not just how well it performs at whatever cutoff you happened to pick.

```
Model outputs a score/probability for each example:
  0.02  0.05  0.11  0.30  0.44  0.51  0.63  0.71  0.88  0.95
   │                                                        │
  sweep threshold from 0 → 1, recompute the confusion       
  matrix (and thus TPR/FPR or Precision/Recall) at EVERY     
  possible cutoff, then plot the resulting curve
```

---

## 2. ROC Curve — Axes and Construction

**ROC = Receiver Operating Characteristic.** Plots **True Positive Rate (Recall/Sensitivity)** on the y-axis against **False Positive Rate** on the x-axis, swept across all thresholds.

$$TPR = \text{Recall} = \frac{TP}{TP+FN} \qquad\qquad FPR = \frac{FP}{FP+TN}$$

```
TPR
 1.0│                                    ╭──────●  (threshold → 0: predict
    │                              ╭─────╯          everyone positive)
    │                        ╭─────╯
    │                  ╭─────╯           ← ROC curve for a good model
    │            ╭─────╯                   (bows toward top-left)
    │      ╭─────╯
    │  ╭───╯
    │●╱          ╲___ diagonal = random guessing (AUC = 0.5)
    │╱          ╱
 0.0●──────────────────────────────────────► FPR
   0.0                                     1.0
   (threshold → 1: predict               
    no one positive)
```

- **Perfect classifier:** curve goes straight up the left edge (TPR=1 while FPR=0) then across the top — AUC = 1.0.
- **Random classifier:** the diagonal line — AUC = 0.5, meaning the model does no better than a coin flip at ranking positives above negatives.
- **AUC-ROC value = the probability that a randomly chosen positive example gets a higher score than a randomly chosen negative example** — this is the single most important interpretation to memorize; it's a ranking metric, not a calibration or accuracy metric.

---

## 3. PR Curve — Axes and Construction

Plots **Precision** on the y-axis against **Recall** on the x-axis, swept across all thresholds.

```
Precision
 1.0│●
    │ ╲___
    │     ╲____
    │          ╲______              ← PR curve for a good model on
    │                 ╲___              an imbalanced dataset
    │                     ╲______
    │                            ╲____
    │                                 ╲___●
    │- - - - - - - - - - - - - - - - - - - - (baseline = positive class prevalence,
    │                                          e.g. 5% for a 5%-fraud dataset)
 0.0└──────────────────────────────────────► Recall
   0.0                                     1.0
```

- **Baseline for PR-AUC is the positive class prevalence** (e.g., if 5% of examples are fraud, a random/no-skill classifier scores PR-AUC ≈ 0.05) — **unlike ROC, where the random baseline is always 0.5 regardless of class balance.** This single fact is the crux of why PR curves are preferred under imbalance (see §4).
- Precision can be non-monotonic as recall increases (can wiggle up and down), unlike TPR which is monotonic by construction as you sweep the threshold — so PR curves are sometimes jagged and are typically summarized via **Average Precision (AP)**, a weighted-sum approximation of the area under the (interpolated) PR curve, rather than naive trapezoidal integration.

---

## 4. The Core Difference: Why ROC Is Misleading Under Imbalance

This is the single most important interview point in this whole topic.

**Worked example:** fraud detection, 1000 transactions, 50 actual fraud (5% prevalence), model produces a mediocre confusion matrix at some threshold: TP=40, FN=10, FP=200, TN=750.

```
TPR = 40/50 = 0.800
FPR = 200/950 = 0.2105    ← FPR looks almost fine (denominator is 950, huge)

Precision = 40/240 = 0.167   ← precision is TERRIBLE — 83% of flagged transactions are false alarms!
Recall    = 40/50 = 0.800
```

**Why this happens:** FPR's denominator is $FP+TN$ — dominated by the huge number of true negatives in an imbalanced dataset, so even a large *absolute* number of false positives (200) barely moves FPR, making the ROC curve look deceptively good. Precision's denominator is $TP+FP$ — directly exposed to how many false alarms are mixed in with true catches, with no huge TN count to dilute it. **This is exactly why PR-AUC is the recommended metric under heavy class imbalance, and ROC-AUC can look artificially strong while the model is actually flooding you with false positives.**

```
Imbalanced dataset intuition:

  ROC-AUC:  "out of a HUGE pool of negatives, what fraction did I wrongly flag?"
             → a small percentage of a huge number can still BE a huge number,
               but the RATE looks small.

  PR-AUC:   "out of everything I flagged as positive, how much was real?"
             → directly exposed to the absolute FP count relative to TP,
               no large TN denominator to hide behind.
```

---

## 5. Side-by-Side Summary Table

| | ROC Curve | PR Curve |
|---|---|---|
| Axes | TPR (y) vs. FPR (x) | Precision (y) vs. Recall (x) |
| Random baseline | Always 0.5, regardless of class balance | Equals positive-class prevalence — shifts with imbalance |
| Sensitive to TN count? | Yes (FPR denominator includes TN) — can look good even with many FPs if TN is huge | No — precision's denominator (TP+FP) never includes TN |
| Best used when | Classes roughly balanced, or you care about both classes symmetrically | Positive class is rare / the class of interest, and false positives are costly |
| Interpretation | P(random positive ranked above random negative) | Directly reflects "trustworthiness" of positive predictions across thresholds |
| Curve shape | Monotonic TPR as threshold sweeps down | Can be non-monotonic / jagged in precision |
| Summary statistic | AUC-ROC | AUC-PR (often called Average Precision, AP) |
| Misleading scenario | Heavy class imbalance — can look "good" while precision is terrible | N/A (this is the fix for the above) — but less standard/familiar outside imbalance-aware teams |

---

## 6. Common Pitfalls (interviewers love probing these)

1. **Reporting AUC-ROC = 0.95 on a fraud/rare-disease dataset as proof the model is great.** As shown in §4, high ROC-AUC can coexist with a model that's flooding users/analysts with false positives — always pair it with (or replace it with) PR-AUC on imbalanced problems.
2. **Forgetting the PR-AUC baseline moves with class balance.** A PR-AUC of 0.3 might be excellent if the positive class is 2% of the data (baseline 0.02, so 0.3 is 15× better than random) or mediocre if the positive class is 40% (baseline 0.4, so 0.3 is actually *worse* than random) — always compare against the correct baseline, not an absolute number.
3. **Comparing ROC-AUC across datasets with different class balances.** ROC-AUC's 0.5 baseline is stable across class ratios, which sounds like a feature, but it also means a 0.85 ROC-AUC on a 50/50 dataset is not directly comparable in difficulty to a 0.85 ROC-AUC on a 1/99 dataset — the underlying separability challenge is very different.
4. **Naive trapezoidal integration for PR-AUC.** Because precision can be non-monotonic, naive interpolation between PR curve points can overestimate the true area; Average Precision (AP) uses a specific interpolation scheme (weighting by the recall increment at each threshold) to avoid this — know that "PR-AUC" in most libraries actually means AP, not literal trapezoidal AUC.
5. **Treating AUC-ROC/PR-AUC as calibration metrics.** Both are purely about *ranking* — whether positives tend to score higher than negatives — not whether the predicted probabilities are numerically well-calibrated (e.g., whether "0.7" really means "70% chance"). A model can have excellent AUC-ROC and terrible calibration simultaneously; use calibration curves / Brier score / Platt scaling to assess that separately.
6. **Choosing ROC by default without checking class balance first.** ROC-AUC is the more commonly reported metric historically, but that's a habit, not a rule — the correct choice depends on class balance and which error type matters more, not which curve is more familiar.
7. **Ignoring that a single AUC number hides *where* on the curve performance is strong or weak.** A model might have great AUC-ROC overall but be poor specifically in the high-precision, low-recall region that the business actually operates in — always also look at the operating point you'll actually use, not just the aggregate area.

---

## 7. FAANG-Level Interview Q&A

**Q1: Your fraud model has AUC-ROC = 0.97, which sounds excellent, but analysts complain they're drowning in false alarms. What's going on, and what metric would you have looked at instead?**
Classic imbalance trap: FPR's denominator ($FP+TN$) is dominated by the huge number of true negatives on a rare-fraud dataset, so even a large absolute number of false positives barely moves FPR, making ROC-AUC look great. Precision (and PR-AUC) is directly exposed to the FP count relative to TP with no large TN denominator to dilute it — I'd look at PR-AUC and the precision at the operating recall the team is actually using, which would likely reveal the false-alarm problem that ROC-AUC was masking.

**Q2: Explain precisely what AUC-ROC = 0.85 means, in a way that doesn't reference "accuracy."**
It means: if you pick one random positive example and one random negative example, there's an 85% probability the model assigns a higher score to the positive one than to the negative one. It's a measure of *ranking quality* — how well the model separates the two classes by score — completely independent of any specific decision threshold, and it says nothing directly about accuracy, precision, or calibration at any particular cutoff.

**Q3: Why does the PR curve's random-baseline change with class balance, while the ROC curve's baseline stays at 0.5 no matter what?**
The ROC diagonal represents a classifier that assigns scores with no signal — TPR and FPR rise together at the same rate regardless of class prevalence, because both are *rates within their own class* (TPR is a rate among positives, FPR is a rate among negatives), so the diagonal is a structural property of random ranking, unaffected by how many of each class exist. Precision, by contrast, directly reflects the proportion of positives among everything predicted positive — for a random/no-skill classifier, that proportion converges to the overall positive-class prevalence in the dataset, which obviously shifts as class balance shifts. This is exactly why PR-AUC "reacts" to imbalance while ROC-AUC structurally doesn't.

**Q4: Two models have the same AUC-ROC of 0.90. Can you conclude they'll perform identically at the specific threshold your business uses?**
No — the same overall AUC-ROC (area under the whole curve) can result from very different curve *shapes*. One model might be excellent at low FPR (dominating in the region the business actually operates in, e.g., high-precision fraud flagging), while the other is only strong at high FPR (a region you'd never actually deploy at) and mediocre where it matters. Always inspect the curve (or the specific operating point) rather than relying on the single aggregate area, especially when comparing two models being considered for the same deployment threshold.

**Q5: When would you prefer ROC-AUC over PR-AUC, even knowing PR-AUC is often recommended for imbalanced problems?**
When classes are roughly balanced and you care about both classes symmetrically (no strong preference between minimizing FP vs FN) — e.g., a balanced A/B classification task, or when you specifically want a metric that's insensitive to shifts in class balance across different evaluation sets (since ROC-AUC's 0.5 baseline is stable, it can be more comparable across datasets with different prevalence, which is sometimes exactly what you want for tracking model quality over time even as the underlying population's class ratio drifts).

**Q6: Your model's PR-AUC is 0.4. Is that good or bad?**
Can't say without the baseline — PR-AUC must be compared against the positive-class prevalence, not judged as an absolute number. If the positive class is 2% of the data, a PR-AUC of 0.4 is roughly 20× better than a no-skill baseline (0.02) — quite good. If the positive class is 45% of the data, 0.4 is actually *worse* than the no-skill baseline (0.45) — a red flag. Always report (or ask for) the baseline alongside the number.

**Q7: How would you compute Average Precision, and why is it preferred over naive trapezoidal integration of the raw PR curve?**
AP is computed as a weighted sum: $AP = \sum_n (R_n - R_{n-1}) \cdot P_n$, summing precision at each threshold weighted by the increase in recall since the previous threshold. This avoids the overestimation risk of naive trapezoidal (linear) interpolation between two PR points, which can imply achievable precision values in between that the model doesn't actually realize — precision isn't guaranteed to vary linearly (or even monotonically) between two adjacent threshold points the way TPR/FPR do on the ROC curve, so a specialized interpolation scheme is needed for a faithful area estimate.

**Q8 (clever): Can a model have a perfect ROC-AUC of 1.0 but a mediocre PR-AUC?**
Yes, in principle they measure different things and a perfect ROC-AUC (perfect *ranking* — every positive scores above every negative) does guarantee precision=1 at *some* threshold (since there's a point where you can perfectly separate them), but it doesn't by itself guarantee the actual PR-AUC number is high in an absolute sense if extremely severe imbalance is involved and only evaluated through certain interpolation/summarization conventions — the more standard and important version of this question, though, is: ROC-AUC of 1.0 combined with any imbalance still yields a PR curve that hugs precision=1 across all recall (since perfect ranking really does mean zero false positives ahead of any positive), so in the *canonical* case perfect ROC-AUC does imply near-perfect PR-AUC too. The deeper point interviewers are fishing for: near-perfect (not literally 1.0) ROC-AUC, e.g., 0.99, absolutely can coexist with mediocre PR-AUC under heavy imbalance, because a tiny residual ranking error near the boundary translates into a much larger relative hit to precision when true negatives vastly outnumber true positives — that's the practically important version of this question, and the one to actually explain in depth if asked.

---

## 8. One-Line Interview Closers

- *"ROC-AUC answers 'can it rank positives above negatives,' PR-AUC answers 'can I trust a positive prediction' — under imbalance those are very different questions with very different answers."*
- *"FPR has true negatives sitting in its denominator, so on an imbalanced dataset it can look great while precision — which has no such cushion — is telling you the model is flooding you with false alarms."*
- *"Any AUC number without its baseline is meaningless for PR-AUC, and any AUC number without the curve's shape is incomplete for ROC-AUC — the aggregate hides exactly where performance is strong or weak."*
