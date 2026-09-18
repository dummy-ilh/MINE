# Chapter 7: Confusion Matrix & Threshold Analysis

> *"Every classifier makes two kinds of mistakes. The question is never 'how do I eliminate mistakes?' — it's 'which mistakes am I willing to make, and how many?'"*

---

## 7.1 The Confusion Matrix

The confusion matrix is the foundation of classification evaluation. Everything else — precision, recall, F1, ROC, AUC — derives from it.

For a binary classifier:

```
                    Predicted Positive    Predicted Negative
                  ┌────────────────────┬────────────────────┐
Actual Positive   │  True Positive (TP) │ False Negative (FN)│
                  ├────────────────────┼────────────────────┤
Actual Negative   │ False Positive (FP) │  True Negative (TN)│
                  └────────────────────┴────────────────────┘
```

**Memory anchor:**
- **True/False** = was the prediction correct?
- **Positive/Negative** = what did the model predict?

### The Four Cells in Plain Language

| Cell | Also called | Meaning |
|---|---|---|
| TP | Hit | Correctly identified a positive |
| FN | Miss, Type II error | Missed a real positive |
| FP | False alarm, Type I error | Incorrectly flagged a negative as positive |
| TN | Correct rejection | Correctly identified a negative |

---

## 7.2 Derived Metrics

Everything you care about in binary classification is a ratio of these four numbers.

### Precision (Positive Predictive Value)

*Of everything the model called positive, how much actually was?*

```
Precision = TP / (TP + FP)
```

Precision is about **purity of your positives**. When precision is low, your positive bucket is full of false alarms.

### Recall (Sensitivity, True Positive Rate)

*Of everything that actually was positive, how much did the model find?*

```
Recall = TP / (TP + FN)
```

Recall is about **coverage of true positives**. When recall is low, you're missing real positives.

### Specificity (True Negative Rate)

*Of everything that actually was negative, how much did the model correctly identify?*

```
Specificity = TN / (TN + FP)
```

### Fall-out (False Positive Rate)

*Of everything that actually was negative, how much did the model incorrectly flag?*

```
FPR = FP / (FP + TN) = 1 - Specificity
```

### F1 Score

Harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Why harmonic mean and not arithmetic mean? The harmonic mean punishes extreme imbalance between precision and recall. A model with Precision=1.0 and Recall=0.01 has arithmetic mean 0.505 but F1=0.02 — correctly flagged as nearly useless.

### F-beta Score

Generalizes F1 to weight precision and recall differently:

```
F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

- β < 1 → weights precision more (cost of false positives is high)
- β = 1 → equal weight (standard F1)
- β > 1 → weights recall more (cost of false negatives is high)

**Practical examples:**

| Domain | β | Why |
|---|---|---|
| Spam filter | β = 0.5 | False positives (blocking real email) hurt more |
| Cancer screening | β = 2 | False negatives (missing cancer) hurt more |
| Legal discovery | β = 3 | Must find all relevant documents; false negatives catastrophic |

### Matthews Correlation Coefficient (MCC)

The most underrated binary classification metric:

```
MCC = (TP × TN - FP × FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

MCC is a correlation coefficient between predicted and actual labels, ranging from -1 to +1.

**Why MCC is better than F1 for imbalanced data:**

MCC uses all four cells of the confusion matrix. F1 completely ignores TN. On highly imbalanced datasets, a model can achieve high F1 by being aggressive about predicting the positive class — MCC will correctly penalize this.

| Metric | Uses TN? | Range | Recommended for |
|---|---|---|---|
| Accuracy | Yes | [0, 1] | Balanced classes only |
| F1 | No | [0, 1] | Imbalanced; recall and precision equally important |
| MCC | Yes | [-1, +1] | Imbalanced; want a single honest number |

---

## 7.3 The Threshold

Every probabilistic classifier outputs a score, not a label. You choose the **threshold** that converts scores to decisions.

```
Model output:  P(positive | x) = 0.73

Threshold = 0.5:  0.73 ≥ 0.5  → Predict Positive
Threshold = 0.8:  0.73 < 0.8  → Predict Negative
Threshold = 0.6:  0.73 ≥ 0.6  → Predict Positive
```

**The threshold is a business decision, not a modeling decision.** Default threshold of 0.5 is almost never optimal.

### How the Threshold Moves Metrics

As you increase the threshold (become more conservative about predicting positive):

```
Threshold ↑  →  Fewer positives predicted
             →  Precision ↑  (fewer false alarms)
             →  Recall ↓    (more misses)

Threshold ↓  →  More positives predicted
             →  Precision ↓  (more false alarms)
             →  Recall ↑    (fewer misses)
```

This trade-off is fundamental. You cannot simultaneously maximize both precision and recall. Choosing the threshold is choosing where to sit on this trade-off curve.

---

## 7.4 ROC Curve

The **Receiver Operating Characteristic** curve visualizes classifier performance across all possible thresholds simultaneously.

### Building the ROC Curve

For each possible threshold value (sweeping from 0 to 1):
1. Compute TPR (Recall) and FPR
2. Plot (FPR, TPR) as a point

```
TPR (Recall)
    1.0 |          ___----------
        |       __/
        |      /        ← Good classifier
    0.5 |    /
        |   /  ← Random classifier (diagonal)
        |  /
    0.0 +--/------------------------
        0.0       0.5             1.0
                  FPR
```

**Reading the ROC curve:**
- Top-left corner = perfect classifier (TPR=1, FPR=0)
- Diagonal = random classifier (no discrimination)
- Bottom-right = inverse classifier (predicts everything wrong)
- **The curve closer to top-left is better**

### AUC-ROC

The **Area Under the ROC Curve** collapses the curve to a single number.

```
AUC = 0.5    → Random classifier
AUC = 0.7    → Decent
AUC = 0.8    → Good
AUC = 0.9    → Very good
AUC = 1.0    → Perfect (suspicious — check for leakage)
```

**Probabilistic interpretation:** AUC is the probability that the model ranks a random positive example higher than a random negative example.

```
AUC = P(score(positive) > score(negative))
```

This makes AUC a pure ranking metric — it measures discrimination ability regardless of calibration.

### When ROC-AUC Misleads

On highly imbalanced datasets (e.g., 1% positive rate), the ROC curve can look great while the model is practically useless. Why?

- FPR = FP / (FP + TN)
- With millions of negatives, even a large absolute number of FPs produces a small FPR
- The curve looks good; precision on the positive class is terrible

**Solution: Use PR curve instead for imbalanced problems.**

---

## 7.5 Precision-Recall Curve

Plots Precision (y) vs. Recall (x) across all thresholds.

```
Precision
    1.0 |*
        | \
        |  \      ← Good classifier
    0.5 |   \___
        |       \___
        |           \___  ← Weak classifier
    0.0 +--------------------
        0.0       0.5     1.0
                Recall
```

**Baseline:** A random classifier has Precision ≈ (# positives) / (# total) — the class prevalence. On a 1% positive dataset, a random classifier baseline is 0.01 precision, not 0.50.

### AUC-PR (Average Precision)

The area under the PR curve. Also called **Average Precision (AP)** — the same AP from Chapter 3, here applied to binary classification rather than document ranking.

```
AUC-PR = 0.01  → Random (on 1% imbalance)
AUC-PR = 0.10  → Modest lift above random
AUC-PR = 0.50  → Good for highly imbalanced problem
AUC-PR = 0.80  → Very good
```

### ROC vs. PR: Which to Use?

| Situation | Use |
|---|---|
| Balanced classes | ROC-AUC |
| Highly imbalanced (< 10% positive rate) | PR-AUC |
| Both positive and negative class matter | ROC-AUC |
| Positive class is rare and important | PR-AUC |
| You want a ranking metric | ROC-AUC |
| You want to understand precision at operating point | PR curve |

---

## 7.6 Threshold Selection Methods

Given a trained model and its ROC/PR curves, how do you pick the threshold you'll actually deploy?

### Method 1: Business Cost Optimization

The most principled method. Define the cost of each error type:

```
Cost(threshold) = C_FP × FP(threshold) + C_FN × FN(threshold)

Optimal threshold = argmin Cost(threshold)
```

**Example — Fraud detection:**
- Cost of FP: $5 (customer service call + goodwill credit)
- Cost of FN: $200 (average fraud loss)
- Optimal threshold: where C_FP × FP = C_FN × FN at the margin

This requires knowing the cost ratio C_FN/C_FP. Even a rough estimate (e.g., "FN costs 20x FP") dramatically improves threshold selection over defaulting to 0.5.

### Method 2: F-beta Maximization

If you can't get explicit cost estimates, choose the threshold that maximizes F_β on the validation set, with β chosen to reflect your precision/recall preference.

```python
from sklearn.metrics import fbeta_score
import numpy as np

thresholds = np.arange(0.01, 1.0, 0.01)
scores = [fbeta_score(y_val, probs_val >= t, beta=2) for t in thresholds]
optimal_threshold = thresholds[np.argmax(scores)]
```

### Method 3: Youden's J Statistic

Maximizes the geometric distance from the diagonal on the ROC curve:

```
J = Sensitivity + Specificity - 1
  = TPR - FPR

Optimal threshold = argmax J
```

Useful when you genuinely care equally about FP and FN, and want the threshold that best separates the classes symmetrically.

### Method 4: Precision-at-Fixed-Recall (or Recall-at-Fixed-Precision)

Set a hard constraint on one metric and maximize the other:

- "We need at least 90% recall. Find the highest-precision threshold that achieves this."
- "We need at least 95% precision. Find the highest-recall threshold that achieves this."

```python
# Find threshold that achieves recall ≥ 0.90 with maximum precision
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_val, probs_val)
valid = recall >= 0.90
optimal_idx = np.argmax(precision[valid])
optimal_threshold = thresholds[valid][optimal_idx]
```

---

## 7.7 Multi-Class Confusion Matrix

Extends naturally to K classes:

```
                  Predicted
              Cat    Dog    Bird
         Cat [ 50     3      2  ]
Actual   Dog [  4    45      1  ]
        Bird [  1     2     47  ]
```

**Reading the matrix:**
- Diagonal = correct predictions
- Off-diagonal = errors; position tells you which class confused with which
- Row sums = actual class counts; column sums = predicted class counts

### Aggregating Multi-class Metrics

**Macro averaging:** Compute metric per class, take unweighted mean.
```
Macro F1 = (F1_cat + F1_dog + F1_bird) / 3
```
Treats all classes equally regardless of frequency. Use when all classes matter equally.

**Weighted averaging:** Compute metric per class, weight by class frequency.
```
Weighted F1 = (n_cat × F1_cat + n_dog × F1_dog + n_bird × F1_bird) / n_total
```
Reflects overall performance weighted by class prevalence.

**Micro averaging:** Aggregate TP, FP, FN across all classes, compute metric once.
```
Micro Precision = ΣTP_k / (ΣTP_k + ΣFP_k)
```
Dominated by frequent classes. Equivalent to accuracy for multi-class.

---

## 7.8 Calibration and Thresholds Interact

A calibrated model (Chapter 4) makes threshold selection much more interpretable.

**Uncalibrated model:**
- Model outputs 0.9 but true probability is 0.6
- Threshold of 0.5 doesn't mean "predict positive when probability > 50%"
- Threshold choice is arbitrary — must be tuned empirically

**Calibrated model:**
- Model outputs 0.9 and true probability is ~0.9
- Threshold of 0.5 literally means "predict positive when you believe it's more likely than not"
- Threshold maps cleanly to business decisions

> **Calibrate first, then choose threshold.** In that order.

---

## 7.9 Threshold Stability in Production

Your threshold, set on validation data, may not remain optimal over time.

**Causes of threshold drift:**
- Class prior shifts (fraud rate changes, disease prevalence changes)
- Feature distribution shifts (new user behavior, seasonal patterns)
- Model score distribution shifts after retraining

**Monitoring recommendations:**
- Track precision and recall separately in production — not just combined metrics
- Set alerts when FPR or FNR drift beyond bounds
- Re-evaluate threshold whenever you retrain the model or detect distribution shift
- Consider **dynamic thresholding** for applications where the class prior changes predictably

---

## 7.10 Worked Example: Medical Screening

Model: Binary classifier for diabetic retinopathy from fundus images.
Dataset: 10,000 patients, 500 positive (5% prevalence).

```
Confusion matrix at threshold 0.5:
              Predicted+    Predicted-
Actual+  [ TP=420         FN=80   ]
Actual-  [ FP=300         TN=9200 ]

Precision = 420 / (420 + 300) = 0.583
Recall    = 420 / (420 + 80)  = 0.840
F1        = 0.689
AUC-ROC   = 0.94
```

The AUC looks excellent at 0.94. But at the default threshold:
- We're missing 80 patients with diabetic retinopathy (FN=80)
- We're flagging 300 healthy patients for follow-up (FP=300)

**In this domain:** Missing a case of retinopathy leads to preventable blindness. False positives just mean an unnecessary ophthalmology visit.

**Decision:** Lower the threshold to 0.2.

```
At threshold 0.2:
Recall    = 490/500 = 0.980  (only miss 10 cases)
Precision = 490/990 = 0.495  (half of flagged are FP)
F1        = 0.658
```

Lower F1, but far better for this clinical context. The ophthalmology department handles the extra load; patients don't go blind.

**Lesson:** The threshold encodes your values. Let domain experts help set it.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Confusion matrix | Foundation of all binary classification metrics |
| Precision | Purity of your positive predictions |
| Recall | Coverage of true positives |
| F-beta | F1 generalized; β encodes your precision/recall preference |
| MCC | Best single metric for imbalanced data; uses all four cells |
| ROC-AUC | Ranking metric; misleads on severe imbalance |
| PR-AUC | Better for rare positive class; baseline = class prevalence |
| Threshold | Business decision, not modeling decision; never default to 0.5 |
| Cost optimization | Most principled threshold selection method |
| Calibrate first | Calibrate probabilities before choosing threshold |

---

## Further Reading

- Fawcett, T. — *An Introduction to ROC Analysis* (Pattern Recognition Letters, 2006)
- Davis & Goadrich — *The Relationship Between PR and ROC Curves* (ICML 2006)
- Chicco & Jurman — *The Advantages of MCC Over F1 Score* (BioData Mining, 2020)
- Saito & Rehmsmeier — *The Precision-Recall Plot Is More Informative Than the ROC Plot* (PLOS ONE, 2015)

---

*Next: Chapter 8 — Regression Metrics*
