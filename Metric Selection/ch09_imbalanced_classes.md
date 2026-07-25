# Chapter 9: Imbalanced Classes

> *"A model that predicts 'no cancer' for every patient achieves 99% accuracy on a dataset where 1% have cancer. It is also perfectly useless. Accuracy is lying to you — and imbalanced data is the reason."*

---

## 9.1 What Is Class Imbalance?

Class imbalance occurs when one class in your dataset vastly outnumbers another. It is not an edge case — it is the default condition in most high-stakes ML problems.

### Prevalence in Real Problems

| Domain | Positive Class | Typical Imbalance Ratio |
|---|---|---|
| Fraud detection | Fraudulent transaction | 1:1,000 – 1:10,000 |
| Medical diagnosis | Disease positive | 1:100 – 1:10,000 |
| Churn prediction | User churns | 1:20 – 1:100 |
| Spam detection | Spam email | 1:5 – 1:50 |
| Defect detection | Defective unit | 1:100 – 1:1,000 |
| Click prediction | User clicks | 1:50 – 1:500 |
| Rare event detection | System failure | 1:10,000+ |

The minority class is almost always the class you care about most. Imbalance and importance are inversely correlated — a feature of reality, not a quirk of your dataset.

---

## 9.2 Why Standard Metrics Fail

### The Accuracy Illusion

```
Dataset: 10,000 samples — 100 positive (1%), 9,900 negative (99%)

Dummy classifier (always predicts negative):
  Accuracy = 9,900 / 10,000 = 99.0%

Real model with some skill:
  Correctly identifies 80 positives, misses 20
  Incorrectly flags 50 negatives as positive
  Accuracy = (80 + 9,850) / 10,000 = 99.3%
```

The dummy classifier scores 99.0% accuracy. Your real model scores 99.3% — barely distinguishable. But the real model is catching 80 fraud cases the dummy misses entirely.

**Accuracy hides the minority class completely when imbalance is severe.**

### Why Loss Functions Are Affected Too

Cross-entropy loss on imbalanced data:
- 9,900 negative samples contribute 9,900 loss terms
- 100 positive samples contribute 100 loss terms
- The model learns to minimize loss on the majority class
- Minority class effectively ignored during training

This affects not just evaluation but the training process itself.

---

## 9.3 The Right Metrics for Imbalanced Data

### Precision, Recall, and F1 (Revisited)

From Chapter 7 — these are the workhorses for imbalanced classification:

```
Precision = TP / (TP + FP)    → Don't waste resources on false alarms
Recall    = TP / (TP + FN)    → Don't miss real positives
F1        = harmonic mean      → Balance of both
```

But report these **on the minority class specifically**, not as macro or weighted averages.

### Matthews Correlation Coefficient (MCC) — The Honest Metric

As introduced in Chapter 7, MCC uses all four cells:

```
MCC = (TP × TN - FP × FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```

**Why MCC is the gold standard for imbalance:**

Suppose a model predicts everything as negative on a 1:99 imbalanced dataset:

```
TP=0, FN=100, FP=0, TN=9900

Accuracy = 99%     ← lying
F1       = 0%      ← correct (undefined/0 for minority class)
MCC      = 0       ← correct (no better than random)
```

MCC correctly returns 0 for a model with no discrimination ability, regardless of class distribution.

### Balanced Accuracy

*Average recall across classes — treats each class equally:*

```
Balanced Accuracy = (TPR + TNR) / 2
                  = (Recall_positive + Recall_negative) / 2
                  = (TP/(TP+FN) + TN/(TN+FP)) / 2
```

- Balanced accuracy = 0.5 for a random or constant classifier
- Balanced accuracy = 1.0 for a perfect classifier
- Scale-invariant to class distribution

**When to use:** You care about recall on both classes equally, and want a single intuitive number.

### PR-AUC over ROC-AUC

As discussed in Chapter 7:

- ROC-AUC is optimistic on imbalanced data (TN in denominator of FPR inflates the curve)
- PR-AUC correctly shows the precision-recall trade-off under imbalance
- **Random baseline for PR-AUC = class prevalence** (e.g., 0.01 for 1% positive rate)

```
On a 1:99 imbalanced dataset:

Random ROC-AUC = 0.50   (looks bad, correctly)
Random PR-AUC  = 0.01   (looks terrible, correctly)

A good model might achieve:
ROC-AUC = 0.92  (impressive-looking)
PR-AUC  = 0.45  (harder to achieve; more honest signal)
```

Always report PR-AUC alongside ROC-AUC for imbalanced problems.

---

## 9.4 Handling Imbalance: Data-Level Methods

### Oversampling the Minority Class

**Random oversampling:** Duplicate minority samples until balance is achieved.
- Simple and fast
- Risk: model overfits to the duplicated samples

**SMOTE (Synthetic Minority Oversampling Technique):**
Creates synthetic minority samples by interpolating between existing ones.

```
For each minority sample x:
  1. Find k nearest neighbors in minority class
  2. Choose a random neighbor xₙ
  3. Create synthetic sample: x_new = x + λ × (xₙ - x)
     where λ ∈ [0, 1] is random

Result: new minority samples along the line between real samples
```

Variants:
- **SMOTE-NC**: handles mixed numerical/categorical features
- **ADASYN**: generates more samples in harder-to-learn regions
- **Borderline-SMOTE**: focuses synthesis near the decision boundary

**Caution:** Oversampling must happen **inside** cross-validation folds, not before splitting. Oversampling before splitting leaks information and inflates metrics.

```python
# WRONG — data leakage
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test = train_test_split(X_resampled, y_resampled)

# RIGHT — oversample only training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Undersampling the Majority Class

**Random undersampling:** Randomly remove majority samples.
- Risk: discards potentially useful information

**Tomek Links:** Remove majority samples that are close to minority samples (near the decision boundary). Cleans the boundary without aggressive undersampling.

**NearMiss:** Selects majority samples closest to minority samples. More aggressive, for severe imbalance.

**Edited Nearest Neighbors (ENN):** Removes samples whose class label disagrees with the majority of their k nearest neighbors. Noise removal rather than pure undersampling.

### Combined Methods

**SMOTETomek:** SMOTE (oversample minority) + Tomek link removal (clean majority boundary). Widely used in practice.

**SMOTEENN:** SMOTE + ENN cleaning. Tends to produce cleaner class boundaries.

---

## 9.5 Handling Imbalance: Algorithm-Level Methods

### Class Weights

The simplest and most underused technique. Most ML libraries support it directly.

Assign higher loss weight to minority class samples:

```
weight_positive  = n_total / (n_classes × n_positive)
weight_negative  = n_total / (n_classes × n_negative)
```

For 1:99 imbalance:
```
weight_positive  = 10,000 / (2 × 100)  = 50
weight_negative  = 10,000 / (2 × 9,900) ≈ 0.505
```

The minority class now contributes 50× more to the loss than each majority sample.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Automatic class weight computation
model = LogisticRegression(class_weight='balanced')
model = RandomForestClassifier(class_weight='balanced')

# Manual specification
model = LogisticRegression(class_weight={0: 1, 1: 50})

# In XGBoost
model = XGBClassifier(scale_pos_weight=99)  # ratio of negatives to positives
```

**Start here before reaching for SMOTE.** Class weights are simpler, less prone to overfitting, and often equally effective.

### Cost-Sensitive Learning

Generalization of class weights. Define a cost matrix:

```
Cost Matrix:
              Predicted+    Predicted-
Actual+  [       0             C_FN    ]   ← cost of missing positive
Actual-  [    C_FP               0    ]   ← cost of false alarm
```

The learning algorithm minimizes expected cost rather than expected error rate. Available in:
- `sklearn`: `CostSensitiveDecisionTreeClassifier` (via `imbalanced-learn`)
- `H2O`: native cost-sensitive training
- Custom: modify sample weights to reflect cost matrix

### Threshold Moving

After training, adjust the classification threshold (Chapter 7). This is often the most practical first intervention:

```python
# Train normally
model.fit(X_train, y_train)

# Find threshold on validation set that maximizes F1 (or F2, or business metric)
probs = model.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.01, 0.99, 0.01)
f1_scores = [f1_score(y_val, probs >= t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
```

**Order of operations:**
1. Train model (with class weights if helpful)
2. Calibrate probabilities (Chapter 4)
3. Choose threshold on validation set based on business cost

### Focal Loss

Developed by Lin et al. (2017) for object detection, but widely applicable.

```
Focal Loss = -α × (1 - pₜ)^γ × log(pₜ)

pₜ = p      if y = 1
pₜ = 1 - p  if y = 0
```

Two hyperparameters:
- **α**: class weight (addresses class imbalance)
- **γ**: focusing parameter (down-weights easy examples)

When γ > 0: easy examples (high confidence correct predictions) contribute less to the loss. Hard examples (misclassified or low-confidence) dominate. This naturally focuses learning on the hard, often minority class samples.

```
γ = 0  →  Standard cross-entropy
γ = 2  →  Typical value; strong focusing effect
```

---

## 9.6 Evaluation Protocol for Imbalanced Data

### Cross-Validation: Use Stratified Folds

Standard k-fold may put all minority samples in one fold, leaving others with no positives.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    # Each fold has the same class ratio as the full dataset
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

**Always use StratifiedKFold for imbalanced data.** Standard KFold is not appropriate.

### Metric Reporting Checklist for Imbalanced Problems

Never report only accuracy. Always report:

- [ ] Confusion matrix (absolute counts, not just percentages)
- [ ] Precision, Recall, F1 on the **minority class specifically**
- [ ] MCC
- [ ] PR-AUC
- [ ] ROC-AUC (for comparison, with the caveat noted)
- [ ] Balanced accuracy
- [ ] Class distribution in train/val/test splits

### The Sanity Check

Before trusting any imbalanced classifier, run this:

```python
# Dummy baseline
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)

print("Dummy accuracy:", dummy.score(X_test, y_test))
print("Dummy F1 (minority):", f1_score(y_test, dummy.predict(X_test)))
# F1 = 0.0 for always-predicts-majority
# Your model must beat this clearly, not by 0.1%
```

---

## 9.7 When NOT to Resample

Resampling is not always the right answer. Situations where you should skip it:

**When class weights work just as well**
Most algorithms support class weights natively. Try that first. It's simpler, less prone to overfit, and doesn't change your data.

**When the test distribution matters**
Your test set should reflect real-world distribution — do not resample the test set. Ever.
Resampling the test set will give you optimistic minority-class metrics that don't reflect production performance.

**When imbalance is informative**
In some settings, the ratio of positives to negatives is itself signal. Oversampling removes this information. Example: in anomaly detection, the model should know that anomalies are rare; resampling to 50/50 destroys this prior.

**When you have enough data**
With millions of samples, even a 1:1000 imbalance gives you thousands of minority samples. Class weights + threshold tuning is usually sufficient.

---

## 9.8 Extreme Imbalance: One-Class and Anomaly Detection

When the positive rate is < 0.01% (1:10,000+), standard supervised learning breaks down. The minority class is too rare to learn from directly.

### One-Class Classification

Train only on the majority (normal) class. Learn what "normal" looks like. Flag deviations.

**Algorithms:**
- **Isolation Forest**: isolates anomalies by random splits; anomalies are isolated quickly
- **Local Outlier Factor (LOF)**: flags points with low local density compared to neighbors
- **One-Class SVM**: learns a tight boundary around the majority class
- **Autoencoders**: reconstruction error is high for anomalies not seen in training

### Evaluation for One-Class / Anomaly Detection

Standard metrics need adaptation:

```
Precision@K:    Of the top K flagged samples, what fraction are true anomalies?
Average Precision: Area under precision-recall curve on flagged samples
Detection Rate:  At FPR = 1%, what fraction of anomalies are caught?
```

ROC-AUC remains useful here as a ranking metric. Report it alongside precision@K.

---

## 9.9 Worked Example: Credit Card Fraud

Dataset: 284,807 transactions, 492 fraud (0.17% positive rate).

```
Step 1: Baseline check
Dummy classifier accuracy: 99.83%
Dummy F1 (fraud class):    0.00%
Dummy MCC:                 0.00

Step 2: Logistic regression, no imbalance handling
Accuracy: 99.91%
F1 (fraud): 0.72
PR-AUC:    0.68
MCC:       0.72

Step 3: Logistic regression, class_weight='balanced'
Accuracy: 97.82%  ← went DOWN (more false alarms)
F1 (fraud): 0.84  ← went UP (better minority recall)
PR-AUC:    0.76
MCC:       0.80

Step 4: XGBoost + scale_pos_weight=578 (ratio of neg:pos)
Accuracy: 99.94%
F1 (fraud): 0.87
PR-AUC:    0.85
MCC:       0.87

Step 5: Threshold tuning on validation set
Optimal threshold: 0.35 (not 0.5)
F1 (fraud): 0.89
PR-AUC:    0.85
MCC:       0.88
```

**Key lessons:**
- Accuracy went DOWN when imbalance was handled correctly — and that's correct behavior
- PR-AUC and MCC are the honest signals
- Class weights alone gave most of the gain; SMOTE added little here
- Threshold tuning gave a meaningful final boost

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Accuracy illusion | On imbalanced data, accuracy is meaningless; always check minority-class metrics |
| MCC | Best single metric for imbalanced classification; uses all four cells |
| PR-AUC vs ROC-AUC | PR-AUC is more honest for rare positive classes |
| Class weights | Simplest fix; try before resampling |
| SMOTE | Synthetic minority oversampling; always inside CV folds |
| Focal loss | Down-weights easy examples; forces focus on hard minority samples |
| StratifiedKFold | Always use for imbalanced cross-validation |
| Don't resample test set | Ever. Test must reflect true distribution |
| Extreme imbalance | Switch to one-class / anomaly detection methods |

---

## Further Reading

- Chawla et al. — *SMOTE: Synthetic Minority Over-sampling Technique* (JAIR, 2002)
- Lin et al. — *Focal Loss for Dense Object Detection* (ICCV 2017)
- Chicco & Jurman — *The Advantages of MCC Over F1 Score and Accuracy* (BioData Mining, 2020)
- Lemaitre et al. — *Imbalanced-learn: A Python Toolbox* (JMLR, 2017)
- He & Garcia — *Learning from Imbalanced Data* (IEEE TKDE, 2009) — comprehensive survey

---

*Next: Chapter 10 — Multi-label & Multi-class Evaluation*
