# Chapter 16: Ensemble Outlier Detection & Evaluation Metrics

## 16.1 Motivation

No single method from Chapters 2–15 dominates across every scenario — each one has a specific failure mode the next chapter fixed (Z-score's masking → MAD; Mahalanobis's circularity → MCD; kNN's global density → LOF; distance's curse-of-dimensionality collapse → Isolation Forest/angle methods). Ensembling is the practical response to *not having to pick just one*: combine multiple detectors so that the blind spot of one method is compensated by another's strength. This chapter covers how to combine detectors and — equally important — how to actually **evaluate** whether any of this is working, which is a distinct and often under-prepared topic in interviews.

## 16.2 Ensemble Strategies for Outlier Detection

**1. Feature Bagging**
Run the same base detector (commonly LOF) on multiple random subsets of features, then aggregate the resulting scores (commonly by averaging or taking the max). This directly targets the curse-of-dimensionality problem from Ch.13 — each individual run operates in a lower-dimensional subspace where distances remain meaningful, and combining many random subspaces reduces the risk of missing an anomaly that's only visible in a subset of features.

**2. Score Normalization Before Combining**
Different detectors produce scores on completely different scales (Mahalanobis distance is unbounded; LOF centers around 1; Isolation Forest is bounded in $(0,1)$) — combining them naively (e.g., simple averaging of raw scores) lets whichever detector happens to have the largest numeric range dominate the ensemble regardless of its actual reliability. Standard practice: convert each detector's scores to a common scale first — e.g., a **z-score of the outlier scores themselves** (not the raw data, but the scores each detector emitted), or convert scores to unified ranks/percentiles before combining.

**3. Combination Functions**
- **Average:** $\text{score}_{ens}(x) = \frac{1}{M}\sum_{m=1}^M \text{score}_m(x)$ — smooths out noise from any single detector, good default.
- **Maximum:** $\text{score}_{ens}(x) = \max_m \text{score}_m(x)$ — flags $x$ if *any* detector considers it highly anomalous; more sensitive (higher recall, more false positives).
- **Weighted combination:** weight each detector's contribution by its estimated reliability (e.g., cross-validated performance on a labeled validation subset, if available).

**4. Model Diversity Principle**
Ensembling only helps if the base detectors have **different failure modes** — combining several variants of "distance from a Gaussian center" (e.g., Z-score + Mahalanobis + Grubbs') gives limited benefit since they all share the same blind spots (non-normality, masking). A genuinely diverse ensemble mixes families with different underlying philosophies: e.g., LOF (local density) + Isolation Forest (isolation difficulty) + PCA reconstruction error (subspace violation) — this is the direct practical payoff of having studied methods across Ch.2-15 as *different lenses on density* (Ch.1 §1.2), rather than interchangeable options.

## 16.3 Evaluation — Why This Is Genuinely Hard

**The core problem:** most real outlier detection is fundamentally **unsupervised** — you usually don't have ground-truth labels for "which points are actually outliers." Evaluation therefore splits into two very different regimes:

### 16.3.1 When You DO Have Labels (e.g., fraud with confirmed cases)

**Why accuracy is the wrong metric:** outliers/anomalies are by definition **rare** — e.g., 0.1% fraud rate. A trivial "predict everything is normal" classifier achieves 99.9% accuracy while catching zero fraud. This is the single most important thing to state upfront in any outlier-detection evaluation discussion.

**Precision, Recall, and Precision@k:**
$$
\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall} = \frac{TP}{TP+FN}
$$
$$
\text{Precision@k} = \frac{\text{number of true anomalies in the top-}k\text{ ranked points}}{k}
$$
Precision@k is especially practical for outlier detection: since you're producing a *ranked* list of anomaly scores, and investigation capacity is usually limited (e.g., a fraud team can only review the top 100 flagged transactions per day), Precision@k directly answers "if I only look at my top $k$ most-suspicious points, how many are actually real?" — this is often more operationally meaningful than a global precision/recall number computed at an arbitrary threshold.

**AUC-ROC vs. AUC-PR (Precision-Recall):**
AUC-ROC can be **misleadingly high** on imbalanced data — because the False Positive Rate ($FP/(FP+TN)$) has a huge $TN$ in the denominator, even a large absolute number of false positives barely moves the FPR when negatives vastly outnumber positives. **AUC-PR (area under Precision-Recall curve) is the preferred metric for imbalanced anomaly detection**, since precision is directly sensitive to the false-positive count relative to total flagged, regardless of how many true negatives exist. This "AUC-ROC misleading on imbalanced data, prefer AUC-PR" point is one of the most frequently tested evaluation concepts in ML interviews generally, not just for outliers.

### 16.3.2 When You DON'T Have Labels (the more common real-world case)

- **Internal consistency checks:** do multiple, philosophically different detectors (Ch.16.2's diversity principle) agree on the same top-flagged points? Strong agreement across diverse methods is weak but real evidence of genuine signal.
- **Injected/synthetic outliers:** artificially inject known synthetic anomalies into a clean dataset (e.g., randomly perturb some points far from their neighbors) and check whether your pipeline recovers them — a controlled sanity check, though synthetic anomalies may not resemble real ones.
- **Downstream task impact:** if outlier removal/flagging is a preprocessing step for another model (e.g., regression, forecasting), measure whether the downstream task's performance improves after treatment — an indirect but practically meaningful signal.
- **Domain expert review:** manually audit a sample of flagged points with subject-matter experts — slow, but often the only real "ground truth" available in practice.

## 16.4 Worked Numerical — Precision@k and AUC-PR Intuition

Suppose a fraud model ranks 1000 transactions by anomaly score, and among the top 20 ranked, 6 are confirmed fraud (out of 10 total fraud cases in the full 1000):

$$
\text{Precision@20} = \frac{6}{20} = 0.30
$$
$$
\text{Recall@20} = \frac{6}{10} = 0.60
$$

**Why AUC-ROC would look artificially strong here regardless:** with 990 legitimate transactions, even if the model additionally misclassified 50 more legitimate transactions as suspicious somewhere further down the ranked list, $FPR = 50/990 \approx 0.0505$ — barely moving the ROC curve, since the denominator (990) swamps the numerator. But if precision is what actually matters operationally (analyst time is limited), that same 50 false positives could easily overwhelm precision at whatever threshold captures them, making **AUC-PR far more sensitive to exactly the kind of error a real system cares about.**

## 16.5 Diagnosis: Choosing an Evaluation Approach

| Situation | Recommended metric/approach |
|---|---|
| Have labels, severe class imbalance (fraud, rare disease) | Precision@k, AUC-PR — not accuracy, not AUC-ROC alone |
| Have labels, investigation capacity is the real constraint | Precision@k directly matches the operational question |
| No labels at all | Ensemble agreement, injected synthetic anomalies, downstream task impact |
| Need to compare two candidate models before deployment | AUC-PR + Precision@k on a held-out labeled (or synthetically labeled) set |
| Building intuition/EDA, not final model selection | Visual inspection of top-ranked flagged points is often the fastest, most practical first check |

## 16.6 Production Considerations
- Ensembling multiplies inference cost (running $M$ detectors instead of 1) — worth explicitly weighing against the marginal precision/recall gain, especially for high-throughput, low-latency scoring systems.
- Evaluation metrics computed once at model launch go stale as the true anomaly rate and pattern drift over time — production monitoring should track precision@k (or proxy signals like analyst-confirmed rate among flagged cases) continuously, not just at initial validation.
- When no labels exist, teams often bootstrap a labeled evaluation set incrementally from analyst feedback on flagged cases over time — worth mentioning as the realistic path from "no labels" to "some labels" in a live system.

## 16.7 Interview Traps
- Proposing "accuracy" as an evaluation metric for outlier/anomaly detection without immediately flagging the severe class-imbalance problem — an almost automatic red flag to interviewers.
- Recommending AUC-ROC as the primary metric for a highly imbalanced anomaly detection problem without knowing to prefer AUC-PR — one of the most common, specifically-tested gaps.
- Suggesting only score-averaging for ensembling without addressing that scores across different detectors are on incompatible scales (§16.2.2) — a subtle but important practical detail.
- Ensembling several methods that share the same underlying assumption (e.g., several Gaussian-distance-based methods) and calling it a "diverse ensemble" — diversity must come from genuinely different detection philosophies, not just different formulas computing the same underlying idea.

## 16.8 L5-Differentiating Talking Points
- Immediately naming the class-imbalance problem and AUC-PR-over-AUC-ROC preference as the first thing you'd address in any outlier-detection evaluation discussion — this is one of the highest-value, most frequently rewarded talking points in the entire curriculum.
- Explicitly connecting ensemble diversity back to the Ch.1 unifying framework — "a good ensemble mixes detectors that estimate density/anomaly differently (local density, isolation difficulty, subspace violation), not just different parameterizations of the same underlying assumption" — this single sentence demonstrates that you've internalized the whole curriculum's structure, not just memorized 15 chapters independently.
- Proactively describing practical no-label evaluation strategies (injected anomalies, ensemble agreement, downstream impact) — showing you've thought about the realistic, most common version of this problem, not just the clean textbook labeled case.

## 16.9 Comprehension Check
1. Why is accuracy a fundamentally misleading metric for evaluating an outlier/anomaly detector, using a concrete numerical example?
2. Explain precisely why AUC-ROC can look artificially strong on severely imbalanced anomaly detection tasks, while AUC-PR does not share this weakness.
3. Give an example of two outlier detectors that would make a genuinely diverse ensemble, and two that would NOT (despite looking superficially different), and explain why.
4. Describe two concrete strategies for evaluating an outlier detection pipeline when no ground-truth labels are available at all.

---
*Next: Chapter 17 — Outliers in Regression: Leverage, Studentized Residuals, Cook's Distance, DFFITS/DFBETAS.*
