# Chapter 10: Multi-label & Multi-class Evaluation

> *"Most real-world classification problems are not 'is this a cat or a dog?' They're 'what is everything that's true about this thing?' The moment you have multiple correct answers, your evaluation strategy must change fundamentally."*

---

## 10.1 Two Very Different Problems

People often conflate multi-class and multi-label. They are structurally different problems with different metrics.

### Multi-class Classification

**One label per sample, chosen from K > 2 classes.**

```
Input: image of an animal
Output: exactly one of {cat, dog, bird, fish, rabbit}

Sample 1: → dog        (one label)
Sample 2: → cat        (one label)
Sample 3: → bird       (one label)
```

Labels are **mutually exclusive**. A sample belongs to exactly one class.

### Multi-label Classification

**Multiple labels per sample, any subset of K labels can be true.**

```
Input: news article
Output: any subset of {politics, economy, health, technology, sports}

Article 1: → {politics, economy}          (two labels)
Article 2: → {health, technology}         (two labels)
Article 3: → {politics, health, economy}  (three labels)
Article 4: → {sports}                     (one label)
```

Labels are **not mutually exclusive**. A sample can belong to zero, one, or many classes simultaneously.

### Why the Distinction Matters

| Property | Multi-class | Multi-label |
|---|---|---|
| Labels per sample | Exactly 1 | 0 to K |
| Label independence | Mutually exclusive | Independent (or correlated) |
| Output layer | Softmax (sums to 1) | Sigmoid per label (independent) |
| Loss function | Categorical cross-entropy | Binary cross-entropy per label |
| Evaluation | Accuracy, macro/micro F1 | Hamming loss, subset accuracy, label F1 |

---

## 10.2 Multi-class Evaluation

### Standard Accuracy

```
Accuracy = # correctly classified samples / # total samples
```

For balanced multi-class problems, accuracy is reasonable. For imbalanced multi-class, it fails exactly as in Chapter 9.

### Per-Class Metrics

Compute precision, recall, and F1 for each class separately (one-vs-rest):

```
For class k:
  TP_k = samples correctly predicted as class k
  FP_k = samples incorrectly predicted as class k (true class ≠ k)
  FN_k = samples of class k predicted as something else
  TN_k = all other samples correctly not predicted as class k

Precision_k = TP_k / (TP_k + FP_k)
Recall_k    = TP_k / (TP_k + FN_k)
F1_k        = 2 × Precision_k × Recall_k / (Precision_k + Recall_k)
```

### Averaging Strategies (Revisited and Extended)

From Chapter 7, now with full multi-class context:

**Macro averaging:**
```
Macro Precision = (1/K) × Σ Precision_k
Macro Recall    = (1/K) × Σ Recall_k
Macro F1        = (1/K) × Σ F1_k
```
Every class counts equally. A failure on a rare class hurts as much as a failure on a common class. Use when all classes are equally important regardless of frequency.

**Weighted averaging:**
```
Weighted F1 = Σ (nₖ/n) × F1_k
```
Each class weighted by its prevalence. Reflects overall performance on the actual distribution. Use when frequent classes matter more.

**Micro averaging:**
```
Micro Precision = ΣTP_k / (ΣTP_k + ΣFP_k)
Micro Recall    = ΣTP_k / (ΣTP_k + ΣFN_k)
Micro F1        = harmonic mean of micro P and R
```
Aggregates all individual predictions. Dominated by frequent classes. For multi-class, Micro F1 = Accuracy when every sample has exactly one true label.

### Choosing the Right Average

```
All classes equally important (rare class matters as much as common):
    → Macro F1

Performance should reflect real distribution:
    → Weighted F1

Sanity check / comparison with accuracy:
    → Micro F1

Debugging class-specific failures:
    → Per-class F1 (report the full table)
```

### Top-K Accuracy

For problems where there are many classes and the model outputs a ranked list:

```
Top-K Accuracy = fraction of samples where true label is in top-K predictions
```

Used heavily in:
- ImageNet: Top-1 and Top-5 accuracy (1000 classes)
- Product search: Top-10 category accuracy
- Medical coding: Top-3 ICD code accuracy

```python
from sklearn.metrics import top_k_accuracy_score

# probs: (n_samples, n_classes) probability matrix
top3_acc = top_k_accuracy_score(y_true, probs, k=3)
```

### Multi-class Confusion Matrix Analysis

With K classes you get a K×K confusion matrix. Key patterns to look for:

```
                Predicted
           Cat  Dog  Bird  Fish
      Cat [ 45    3    2    0 ]   ← Cat often confused with Dog
Actual Dog [  5   40    3    2 ]
     Bird [  1    4   43    2 ]
     Fish [  0    2    1   47 ]   ← Fish rarely confused
```

**What to look for:**
- Off-diagonal clusters → systematic confusion between specific class pairs
- Asymmetric confusion → Model A→B more than B→A (directional bias)
- Row with many errors → a class the model consistently fails on
- Column with many errors → a class the model over-predicts

**Visualization:** Use a normalized confusion matrix (row-normalize by true class counts) for clearer pattern detection when class sizes differ.

---

## 10.3 Multi-label Evaluation

Multi-label evaluation is fundamentally harder because each sample has a variable number of correct answers, and partial credit is possible.

### The Output Format

For K labels, the model outputs a vector of K binary predictions:

```
True labels:       [1, 0, 1, 1, 0]    ← ground truth
Predicted labels:  [1, 0, 1, 0, 0]    ← model output (thresholded)
Predicted scores:  [0.9, 0.1, 0.8, 0.4, 0.2]  ← raw probabilities
```

The model gets partial credit: it got labels 1, 3, 5 right but missed label 4.

### Hamming Loss

*Fraction of label-sample pairs that are incorrectly classified.*

```
Hamming Loss = (1 / (n × K)) × Σᵢ Σₖ |yᵢₖ - ŷᵢₖ|
```

Where n = number of samples, K = number of labels.

**Example:**

```
Sample 1: True [1,0,1,1,0]   Predicted [1,0,1,0,0]  → 1 wrong out of 5
Sample 2: True [0,1,0,1,1]   Predicted [0,1,0,1,0]  → 1 wrong out of 5

Hamming Loss = (1 + 1) / (2 × 5) = 2/10 = 0.20
```

**Interpretation:** 20% of all (sample, label) pairs are incorrect.

**Properties:**
- Range [0, 1]; lower is better
- Treats all labels equally regardless of frequency
- Does not penalize based on how many labels a sample has
- Insensitive to which labels are wrong (missing a rare important label costs the same as getting a common label wrong)

### Subset Accuracy (Exact Match Ratio)

*Fraction of samples where the predicted label set exactly matches the true label set.*

```
Subset Accuracy = (1/n) × Σᵢ 𝟙[ŷᵢ = yᵢ]
```

Where 𝟙[ŷᵢ = yᵢ] = 1 only if the entire label vector is identical.

```
Sample 1: True [1,0,1,1,0]   Predicted [1,0,1,0,0]  → WRONG (one label off)
Sample 2: True [0,1,0,1,1]   Predicted [0,1,0,1,1]  → CORRECT (exact match)

Subset Accuracy = 1/2 = 0.50
```

**The strictest multi-label metric.** Any single wrong label counts as a complete failure. Very harsh — even a model that gets 4 out of 5 labels right scores 0 on subset accuracy.

**When to use:** Tasks where the complete label set must be correct (e.g., legal document tagging where missing any tag has consequences).

### Label-Based Metrics (Macro/Micro)

Apply binary classification metrics per label, then average:

**Macro-averaged F1:**
```
For each label k:
  Compute F1_k using binary TP/FP/FN for that label
Macro F1 = (1/K) × Σ F1_k
```

Treats each label equally. Sensitive to performance on rare labels.

**Micro-averaged F1:**
```
Aggregate TP, FP, FN across all labels and all samples
Micro F1 = 2 × ΣTP / (2 × ΣTP + ΣFP + ΣFN)
```

Dominated by frequent labels. If some labels appear in nearly every sample, micro F1 reflects those labels most.

**Instance-averaged (Sample F1):**
```
For each sample i:
  Compute F1ᵢ between true and predicted label vectors
Sample F1 = (1/n) × Σ F1ᵢ
```

Averages across samples rather than labels. Each sample contributes equally. Natural when samples vary widely in the number of labels.

### Jaccard Similarity (Intersection over Union)

*Overlap between predicted and true label sets:*

```
Jaccard(yᵢ, ŷᵢ) = |yᵢ ∩ ŷᵢ| / |yᵢ ∪ ŷᵢ|
```

```
True:      {politics, economy, health}
Predicted: {politics, health, technology}

Intersection: {politics, health}           → |∩| = 2
Union:        {politics, economy, health, technology} → |∪| = 4

Jaccard = 2/4 = 0.50
```

Average Jaccard across samples gives a natural multi-label metric. Equivalent to F1 in the binary case (since F1 = 2|∩| / (|y| + |ŷ|) and Jaccard = |∩| / |∪|).

---

## 10.4 Threshold Selection in Multi-label Problems

Each label needs its own threshold — not a single global threshold.

### Per-Label Threshold Tuning

```python
import numpy as np
from sklearn.metrics import f1_score

# probs: (n_samples, n_labels)
# y_val: (n_samples, n_labels)

best_thresholds = []
for k in range(n_labels):
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1s = [f1_score(y_val[:, k], probs_val[:, k] >= t) for t in thresholds]
    best_thresholds.append(thresholds[np.argmax(f1s)])

# Apply per-label thresholds
y_pred = np.column_stack([
    probs_test[:, k] >= best_thresholds[k]
    for k in range(n_labels)
])
```

### Label Frequency and Threshold

Rare labels often need lower thresholds (more sensitivity). Common labels can afford higher thresholds (more precision). Per-label tuning captures this automatically.

---

## 10.5 Label Correlation and Structured Prediction

Standard multi-label evaluation treats labels as independent. But labels are often correlated.

**Examples of label correlations:**
```
{python, machine_learning} → often co-occur in tech articles
{beach, summer} → often co-occur in travel photos
{diabetes, obesity} → often co-occur in medical records
```

### Evaluating Label Correlation Capture

If your model exploits label correlations, you can evaluate whether it does so correctly:

**Conditional accuracy:** Given that label A is predicted, how accurately is label B predicted?

**Label co-occurrence matrix:** Compare predicted co-occurrence rates to true co-occurrence rates. A model that misses correlations will produce a flat co-occurrence matrix.

### Structured Prediction Methods

When label correlations are strong, consider:
- **Classifier chains**: Train K classifiers sequentially, each seeing predictions of previous ones
- **Label powerset**: Treat each unique label combination as a single class
- **LSTM/transformer decoders**: Generate label sets sequentially

These methods change training, not evaluation. Evaluate them with the same multi-label metrics.

---

## 10.6 Multi-label Ranking Metrics

When the model outputs scores per label (not just binary predictions), you can use ranking metrics.

### Coverage Error

How many labels must you include in the top predictions to cover all true labels?

```
Coverage = average rank of the last true label
```

Lower is better. If the true label set is {A, C, E} and the model ranks them at positions {1, 3, 7}, coverage = 7.

### Label Ranking Average Precision (LRAP)

For each sample, for each true label, what fraction of labels ranked ahead of it are also true?

```
LRAP = (1/n) × Σᵢ (1/|Yᵢ|) × Σₖ∈Yᵢ |{j: rank_j ≤ rank_k, j ∈ Yᵢ}| / rank_k
```

Range [0, 1]; higher is better. Analogous to MAP (Chapter 3) but for multi-label problems.

### Label Ranking Loss

Fraction of (true label, false label) pairs that are incorrectly ordered:

```
Ranking Loss = fraction of pairs (true label ranked below false label)
```

Lower is better. Range [0, 1]. Penalizes inversions in the label ranking.

---

## 10.7 One-vs-Rest vs. One-vs-One

For multi-class problems with K classes, you can decompose into binary problems:

### One-vs-Rest (OvR)

Train K binary classifiers. Classifier k learns to separate class k from all others.

```
K=4 classes (A, B, C, D):
  Classifier 1: A vs {B, C, D}
  Classifier 2: B vs {A, C, D}
  Classifier 3: C vs {A, B, D}
  Classifier 4: D vs {A, B, C}

Prediction: argmax of K classifier scores
```

**Evaluation consideration:** Each binary classifier faces imbalance (1 class vs K-1 classes). Use class weights.

### One-vs-One (OvO)

Train K(K-1)/2 binary classifiers, one for each pair of classes.

```
K=4 classes: 4×3/2 = 6 classifiers
  A vs B, A vs C, A vs D, B vs C, B vs D, C vs D

Prediction: majority vote across all classifiers
```

**Evaluation consideration:** Each classifier is balanced (equal representation of two classes). Scales poorly with K (quadratic growth).

| | OvR | OvO |
|---|---|---|
| Number of classifiers | K | K(K-1)/2 |
| Training data per classifier | Full dataset | Subset (two classes) |
| Imbalance per classifier | Yes | No |
| Scales to large K | Yes | No |
| Preferred by | SVMs (OvO more common), logistic regression (OvR) | SVMs |

---

## 10.8 Practical Metric Selection Guide

```
Is each sample assigned exactly one class?
    Yes → Multi-class
          ├─ K = 2:        Binary classification (Chapters 7, 9)
          ├─ K small, balanced:   Accuracy + macro F1
          ├─ K small, imbalanced: Macro F1 + MCC per class
          └─ K large (100+):      Top-K accuracy + macro F1

    No  → Multi-label
          ├─ Complete correctness required: Subset accuracy + Hamming loss
          ├─ Partial credit acceptable:     Sample F1 + macro F1
          ├─ Label frequency varies widely: Micro F1 + per-label breakdown
          └─ Model outputs scores (not binary): LRAP + Ranking Loss
```

---

## 10.9 Worked Example: Document Tagging

Task: Tag news articles with topic labels. 10,000 articles, 20 possible tags, average 2.3 tags per article.

```
Model outputs: 20 sigmoid scores per article
Threshold: 0.5 (default) → binary predictions

Results:
  Hamming Loss:      0.08   (8% of label-article pairs wrong)
  Subset Accuracy:   0.31   (only 31% exact matches)
  Sample F1:         0.74
  Macro F1:          0.61   (low — some rare tags poorly predicted)
  Micro F1:          0.79   (high — common tags well predicted)

Per-label analysis:
  Tag 'politics':   F1 = 0.91  (frequent, easy)
  Tag 'economy':    F1 = 0.85  (frequent, easy)
  Tag 'obituary':   F1 = 0.43  (rare, hard)
  Tag 'weather':    F1 = 0.38  (rare, hard)
```

**Diagnosis:**
- Micro F1 flatters the model (dominated by common tags)
- Macro F1 reveals the rare tag problem
- Subset accuracy is harsh but informative for the "must get all tags right" use case

**Actions taken:**
1. Lower threshold for rare labels (per-label tuning) → Macro F1 → 0.71
2. Add class weights for rare labels → Macro F1 → 0.76
3. Augment training data for rare labels → Macro F1 → 0.81

**Lesson:** Always look at per-label F1. Aggregates hide where the model actually fails.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Multi-class vs multi-label | One label per sample vs. any subset of labels |
| Macro F1 | Equal weight per class; sensitive to rare class failures |
| Micro F1 | Dominated by frequent classes; ≈ accuracy for multi-class |
| Weighted F1 | Weighted by class frequency; reflects real distribution |
| Hamming Loss | Fraction of wrong label-sample pairs; lenient |
| Subset Accuracy | Exact match of full label set; very strict |
| Sample F1 | F1 averaged per sample; natural for variable label counts |
| Jaccard | Intersection over union; intuitive overlap measure |
| Per-label threshold | Each label needs its own threshold in multi-label |
| Top-K accuracy | Right answer in top K predictions; for large K problems |

---

## Further Reading

- Tsoumakas & Katakis — *Multi-label Classification: An Overview* (IJDWM, 2007)
- Zhang & Zhou — *A Review on Multi-Label Learning Algorithms* (IEEE TKDE, 2014)
- Read et al. — *Classifier Chains for Multi-label Classification* (Machine Learning, 2011)
- scikit-learn documentation — *Multilabel Ranking Metrics* (comprehensive with examples)

---

*Next: Chapter 11 — Probabilistic Metrics (Log-loss, Brier Score, ECE)*
