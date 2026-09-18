# Chapter 10: Multi-label & Multi-class Evaluation
### Apple MLE Interview Master Notes — Improved & Expanded Edition

> *"Most real-world classification problems are not 'is this a cat or a dog?' They're 'what is everything that's true about this thing?' The moment you have multiple correct answers, your evaluation strategy must change fundamentally."*

---

## 10.0 Master Cheat Sheet

### 10.0.1 Problem Types at a Glance

| Property | Multi-class | Multi-label |
|---|---|---|
| Labels per sample | Exactly 1 | 0 to K (any subset) |
| Label independence | Mutually exclusive | Independent (or correlated) |
| Output layer | Softmax (probabilities sum to 1) | Sigmoid per label (independent) |
| Loss function | Categorical cross-entropy | Binary cross-entropy per label |
| Core metrics | Accuracy, macro/micro/weighted F1 | Hamming loss, subset accuracy, sample F1 |

### 10.0.2 Key Facts to Keep at the Front of Your Mind

| # | Fact | Detail |
|---|---|---|
| 1 | Micro F1 = Accuracy (multi-class) | Only when every sample has exactly one true label |
| 2 | Macro F1 — use when | All classes matter equally, regardless of frequency |
| 3 | Weighted F1 — use when | Frequent classes should count more (real-world distribution) |
| 4 | Subset accuracy is the strictest metric | One wrong label = full failure, even if 9/10 labels are correct |
| 5 | Hamming loss is the most lenient | Measures fraction of individual label-sample pairs that are wrong |
| 6 | Per-label thresholds are essential | Each label in a multi-label model needs its own tuned threshold |
| 7 | Micro F1 flatters imbalanced models | Dominated by frequent labels; always pair with macro F1 |
| 8 | Top-K accuracy for large label spaces | Used in ImageNet (Top-5), search ranking, medical coding |
| 9 | Softmax is wrong for multi-label | Forces probabilities to sum to 1; labels become mutually exclusive |
| 10 | Normalize confusion matrices | Row-normalize by true class counts for fair pattern detection |

---

## 10.1 Two Very Different Problems

### 10.1.1 What Is Multi-class Classification?

**One label per sample, chosen from K > 2 classes.**

Every sample belongs to exactly one class. The classes are **mutually exclusive** — a news article cannot be both "sports" and "politics" in a pure multi-class setup.

**Example:**
```
Input:  Image of an animal
Output: Exactly one of {cat, dog, bird, fish, rabbit}

Sample 1 → dog    (one label)
Sample 2 → cat    (one label)
Sample 3 → bird   (one label)
```

**Plain-English analogy:** Think of sorting mail into exactly one bin. Each letter goes into one and only one bin — there's no "both" option.

---

### 10.1.2 What Is Multi-label Classification?

**Multiple labels per sample — any subset of K labels can be true simultaneously.**

Labels are **not mutually exclusive**. A sample can have zero, one, or many correct labels at once.

**Example:**
```
Input:  News article
Output: Any subset of {politics, economy, health, technology, sports}

Article 1 → {politics, economy}           (2 labels)
Article 2 → {health, technology}          (2 labels)
Article 3 → {politics, health, economy}   (3 labels)
Article 4 → {sports}                      (1 label)
```

**Plain-English analogy:** Think of tagging a photo. A single photo can be tagged "beach," "sunset," and "family" all at once — these tags don't compete with each other.

---

### 10.1.3 Why the Distinction Matters (Architecture Consequences)

Getting this distinction wrong causes silent, hard-to-debug failures in production:

| Decision | Multi-class | Multi-label |
|---|---|---|
| Output layer | Softmax — one distribution over K classes | K independent sigmoid units |
| Loss function | Categorical cross-entropy | Summed binary cross-entropy (one per label) |
| Threshold | Single argmax | Per-label threshold (tuned separately) |
| Evaluation metrics | Accuracy, macro/micro/weighted F1 | Hamming loss, subset accuracy, sample F1 |
| PyTorch loss | `nn.CrossEntropyLoss(logits, y)` | `nn.BCEWithLogitsLoss(logits, y)` |

> **⚠️ Apple Production Pitfall:** Using softmax + categorical cross-entropy for a multi-label task is one of the most common silent bugs in deployed models. Softmax forces probabilities to sum to 1, making the model treat labels as mutually exclusive when they aren't. The model will train without error but produce systematically wrong outputs. Always match your output layer and loss to the problem type.

---

## 10.2 Multi-class Evaluation

### 10.2.1 Standard Accuracy

```
Accuracy = (# correctly classified samples) / (# total samples)
```

**When it works:** Balanced multi-class problems where all classes appear at similar frequencies.

**When it fails:** Any class imbalance. A model that always predicts the majority class can score 95% accuracy while being completely useless for minority classes. This is the most common misleading metric in production.

---

### 10.2.2 Per-Class Metrics (One-vs-Rest Decomposition)

To understand per-class performance, treat each class k as a binary problem:

- **TP_k** — Samples correctly predicted as class k
- **FP_k** — Samples incorrectly predicted as class k (true class ≠ k)
- **FN_k** — Samples of class k predicted as something else
- **TN_k** — All other samples correctly not predicted as class k

```
Precision_k = TP_k / (TP_k + FP_k)   ← Of all times we predicted class k, how often were we right?
Recall_k    = TP_k / (TP_k + FN_k)   ← Of all true class k samples, how many did we catch?
F1_k        = 2 × Precision_k × Recall_k / (Precision_k + Recall_k)
```

Always report the full per-class F1 table during model development. Aggregates hide class-specific failures.

---

### 10.2.3 Averaging Strategies — Choosing the Right One

Once you have per-class F1 scores, you need to combine them into a single summary. Three strategies exist, each with a different philosophy:

#### 1. Macro Averaging — "Every class matters equally"

```
Macro Precision = (1/K) × Σ Precision_k
Macro Recall    = (1/K) × Σ Recall_k
Macro F1        = (1/K) × Σ F1_k
```

- A failure on a rare class hurts exactly as much as a failure on a common class
- **Use when:** All classes are equally important regardless of frequency (e.g., rare disease detection where missing any disease is equally costly)

#### 2. Weighted Averaging — "Frequent classes count more"

```
Weighted F1 = Σ (nₖ / n) × F1_k
```

Where n_k is the number of samples in class k and n is the total.

- Reflects real-world distribution; frequent classes dominate the score
- **Use when:** You care about overall performance on the actual data distribution (e.g., a search engine where common queries are most important)

#### 3. Micro Averaging — "Aggregate all predictions"

```
Micro Precision = ΣTP_k / (ΣTP_k + ΣFP_k)
Micro Recall    = ΣTP_k / (ΣTP_k + ΣFN_k)
Micro F1        = harmonic mean of Micro Precision and Micro Recall
```

- Dominated by frequent classes
- For multi-class where every sample has exactly one true label: **Micro F1 = Accuracy**
- **Use when:** You want a sanity-check comparison against accuracy, or frequent classes genuinely matter most

#### Averaging Strategy Decision Table

| Scenario | Recommended Average |
|---|---|
| All classes equally important (rare = common) | Macro F1 |
| Performance should reflect real distribution | Weighted F1 |
| Sanity check / compare with accuracy | Micro F1 |
| Debugging specific class failures | Per-class F1 (full table) |

---

### 10.2.4 Top-K Accuracy

For problems with many classes where the model outputs a ranked list of predictions:

```
Top-K Accuracy = fraction of samples where the true label appears in the top K predictions
```

**Real-world uses:**

| Domain | Common K | Why |
|---|---|---|
| ImageNet (1,000 classes) | 5 | Top-5 became the standard benchmark from 2010–2017 |
| Product search (10,000+ SKUs) | 10 | A user sees the top 10 results |
| Medical ICD coding (70,000+ codes) | 3 | Clinicians review the top 3 suggestions |
| Recommendation systems | 10–100 | Items shown in a feed or carousel |

```python
from sklearn.metrics import top_k_accuracy_score

# probs: shape (n_samples, n_classes)
top3_acc = top_k_accuracy_score(y_true, probs, k=3)
```

**Plain-English explanation:** Instead of requiring the model to guess the exact correct answer, you give it credit if the right answer is anywhere in its top K guesses. This is more forgiving and often more meaningful when classes are similar or ambiguous.

---

### 10.2.5 Multi-class Confusion Matrix Analysis

With K classes, the confusion matrix is K×K. Each cell (i, j) shows how many true class-i samples were predicted as class j. The diagonal shows correct predictions.

**Example (4-class animal classifier):**

```
                Predicted
           Cat  Dog  Bird  Fish
      Cat [ 45    3    2    0 ]
Actual Dog [  5   40    3    2 ]
     Bird [  1    4   43    2 ]
     Fish [  0    2    1   47 ]
```

**Four patterns to look for:**

| Pattern | What it means | Example above |
|---|---|---|
| Off-diagonal cluster | Systematic confusion between two specific classes | Cat→Dog and Dog→Cat confusion |
| Asymmetric confusion | Model confuses A→B more than B→A | Dog is called Cat 5× but Cat is called Dog only 3× |
| Row with many errors | A class the model consistently fails on | Dog row: 5+3+2 = 10 errors |
| Column with many errors | A class the model over-predicts | — |

> **Apple Production Tip:** Always use **row-normalized confusion matrices** (divide each row by the true class count) when class sizes differ. Raw counts mislead you — a class with 1,000 samples naturally has more absolute errors than one with 50 samples, even if the per-class accuracy is identical. Normalization makes error patterns comparable across classes.

---

## 10.3 Multi-label Evaluation

Multi-label evaluation is fundamentally harder because each sample has a **variable number of correct answers** and **partial credit is possible**. The metrics below handle this in different ways.

### 10.3.1 The Multi-label Output Format

For K labels, the model produces a K-dimensional output per sample:

```
True labels:      [1, 0, 1, 1, 0]   ← ground truth binary vector
Predicted labels: [1, 0, 1, 0, 0]   ← thresholded binary predictions
Predicted scores: [0.9, 0.1, 0.8, 0.4, 0.2]  ← raw sigmoid probabilities
```

The model gets **partial credit**: it correctly predicted labels 1, 3, and 5, but missed label 4.

---

### 10.3.2 Metric 1: Hamming Loss — The Most Lenient

**Question it answers:** "What fraction of individual (sample, label) pairs are wrong?"

```
Hamming Loss = (1 / (n × K)) × Σᵢ Σₖ |yᵢₖ − ŷᵢₖ|
```

Where n = number of samples, K = number of labels.

**Worked example:**

| Sample | True Labels | Predicted | # Wrong |
|---|---|---|---|
| 1 | [1, 0, 1, 1, 0] | [1, 0, 1, 0, 0] | 1 out of 5 |
| 2 | [0, 1, 0, 1, 1] | [0, 1, 0, 1, 0] | 1 out of 5 |

```
Hamming Loss = (1 + 1) / (2 × 5) = 2/10 = 0.20
```

**Interpretation:** 20% of all (sample, label) pairs are incorrect.

**Properties:**

| Property | Detail |
|---|---|
| Range | [0, 1] — lower is better |
| Sensitivity to outliers | Treats all labels equally, regardless of rarity |
| Partial credit | Yes — getting 4/5 labels right is much better than 0/5 |
| Main weakness | Missing a rare, important label costs the same as a trivial common label |

---

### 10.3.3 Metric 2: Subset Accuracy — The Strictest

**Question it answers:** "What fraction of samples have a perfectly correct prediction?"

```
Subset Accuracy = (1/n) × Σᵢ 𝟙[ŷᵢ = yᵢ]
```

Where 𝟙[ŷᵢ = yᵢ] = 1 only if **every single label** in the vector is correct.

**Using the same example:**

| Sample | True | Predicted | Exact Match? |
|---|---|---|---|
| 1 | [1, 0, 1, 1, 0] | [1, 0, 1, 0, 0] | ❌ (one label off) |
| 2 | [0, 1, 0, 1, 1] | [0, 1, 0, 1, 1] | ✅ |

```
Subset Accuracy = 1/2 = 0.50
```

**Plain-English analogy:** Imagine filling out a 20-question form. Subset accuracy only gives you credit if every single answer is correct. Getting 19/20 right still scores zero. This is extremely strict.

**When to use:** Tasks where the complete label set must be correct, with no tolerance for any missing or extra label (e.g., legal document compliance tagging, medical record coding where missing any tag has regulatory consequences).

---

### 10.3.4 Metric 3: Label-based Metrics (Macro / Micro / Sample F1)

Apply binary F1 per label, then aggregate. Three aggregation strategies:

#### Macro-averaged F1
```
For each label k:  compute binary F1_k
Macro F1 = (1/K) × Σ F1_k
```
Treats each label equally. Sensitive to rare-label performance. **Use when all labels matter equally.**

#### Micro-averaged F1
```
Aggregate TP, FP, FN across all labels and all samples
Micro F1 = 2 × ΣTP / (2 × ΣTP + ΣFP + ΣFN)
```
Dominated by frequent labels. **Use when common labels are more important.**

#### Sample (Instance-averaged) F1
```
For each sample i: compute F1ᵢ between true and predicted label vectors
Sample F1 = (1/n) × Σ F1ᵢ
```
Averages across samples rather than labels. Each sample contributes equally, regardless of how many labels it has. **Use when samples vary widely in label count.**

#### Multi-label F1 Strategy Comparison

| Strategy | Sensitivity to rare labels | Dominated by | Best used when |
|---|---|---|---|
| Macro F1 | High | Nothing — equal weight | All labels matter equally |
| Micro F1 | Low | Frequent labels | Common labels are more business-critical |
| Sample F1 | Medium | Samples with many labels | Variable label count per sample |

---

### 10.3.5 Metric 4: Jaccard Similarity (Intersection over Union)

**Question it answers:** "How much do the predicted and true label sets overlap?"

```
Jaccard(yᵢ, ŷᵢ) = |yᵢ ∩ ŷᵢ| / |yᵢ ∪ ŷᵢ|
```

**Worked example:**

```
True:      {politics, economy, health}
Predicted: {politics, health, technology}

Intersection: {politics, health}                          → |∩| = 2
Union:        {politics, economy, health, technology}     → |∪| = 4

Jaccard = 2/4 = 0.50
```

**Key insight:** Jaccard is equivalent to F1 in the binary (single-label) case. For multi-label, it's an intuitive measure of "how much of the right answer did we get, penalized for extra wrong predictions."

**Properties:**
- Range [0, 1] — higher is better
- Penalizes both missed labels (false negatives) and extra labels (false positives)
- More interpretable than micro/macro F1 for non-technical stakeholders

---

## 10.4 Threshold Selection in Multi-label Problems

Unlike multi-class (where you simply take the argmax), multi-label models require a threshold for each label. A common mistake is applying a single global threshold of 0.5 to all labels.

### 10.4.1 Why Per-Label Thresholds Matter

- **Rare labels** need lower thresholds (prioritize recall; you can't afford to miss them)
- **Common labels** can use higher thresholds (precision is more important; false positives are costly)
- A single threshold of 0.5 systematically under-predicts rare labels

### 10.4.2 Per-Label Threshold Tuning (Code)

```python
import numpy as np
from sklearn.metrics import f1_score

# probs: (n_samples, n_labels) — validation set probabilities
# y_val: (n_samples, n_labels) — validation set ground truth

best_thresholds = []
for k in range(n_labels):
    candidate_thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = [
        f1_score(y_val[:, k], probs_val[:, k] >= t)
        for t in candidate_thresholds
    ]
    best_thresholds.append(candidate_thresholds[np.argmax(f1_scores)])

# Apply per-label thresholds to test set
y_pred = np.column_stack([
    probs_test[:, k] >= best_thresholds[k]
    for k in range(n_labels)
])
```

> **Apple Production Tip:** Tune thresholds on a held-out validation set, never the test set. Re-tune thresholds whenever the data distribution shifts (new products, new user demographics, seasonal changes). Thresholds that worked 6 months ago may be wrong today.

---

## 10.5 Label Correlation and Structured Prediction

Standard multi-label evaluation treats labels as independent. In reality, labels are often correlated — and a production model that exploits these correlations performs better.

### 10.5.1 Common Label Correlation Examples

| Domain | Correlated label pairs | Direction |
|---|---|---|
| Tech articles | `python` ↔ `machine_learning` | Co-occurrence |
| Travel photos | `beach` ↔ `summer` | Co-occurrence |
| Medical records | `diabetes` ↔ `obesity` | Co-occurrence |
| News articles | `election` ↔ `candidate` | Co-occurrence |
| E-commerce | `luxury` ↔ `low_price` | Mutual exclusion |

### 10.5.2 Evaluating Whether Your Model Captures Correlations

1. **Conditional accuracy** — Given that label A is predicted, how often is label B predicted? Compare to the true conditional probability P(B|A).
2. **Label co-occurrence matrix** — Compare predicted co-occurrence rates to true co-occurrence rates. A model that ignores correlations produces a flat, near-uniform co-occurrence matrix even when the data is strongly structured.

### 10.5.3 Structured Prediction Methods (Training Approaches)

These are training strategies — the evaluation metrics stay the same:

| Method | How it works | Best for |
|---|---|---|
| Classifier chains | Train K classifiers sequentially; each sees the predictions of all previous ones | Moderate correlation, small K |
| Label powerset | Treat each unique label combination as a single class | Strong correlation, small K |
| LSTM/Transformer decoder | Generate label sets sequentially, one label at a time | Strong correlation, large K |

---

## 10.6 Multi-label Ranking Metrics

When the model outputs a **score per label** (not just a binary prediction), ranking metrics provide richer evaluation without needing a threshold decision.

### 10.6.1 Metric 1: Coverage Error

**Question:** "How many labels must I include in my top predictions to guarantee all true labels are covered?"

```
Coverage Error = average rank of the last (worst-ranked) true label
```

Lower is better. If the true label set is {A, C, E} and the model ranks them at positions 1, 3, 7, then coverage = 7. You'd need to include the top 7 predictions to guarantee all true labels are found.

### 10.6.2 Metric 2: Label Ranking Average Precision (LRAP)

**Question:** "For each true label, how many of the labels ranked above it are also true?"

```
LRAP = (1/n) × Σᵢ (1/|Yᵢ|) × Σₖ∈Yᵢ |{j : rank_j ≤ rank_k, j ∈ Yᵢ}| / rank_k
```

Range [0, 1] — higher is better. This is the multi-label analogue of Mean Average Precision (MAP). A model that always puts all true labels at the very top scores LRAP = 1.0.

### 10.6.3 Metric 3: Label Ranking Loss

**Question:** "How often is a true label ranked below a false label?"

```
Ranking Loss = fraction of (true label, false label) pairs where the true label ranks lower
```

Range [0, 1] — lower is better. A perfect model scores 0.0 (every true label is ranked above every false label for every sample).

### 10.6.4 Ranking Metrics Summary

| Metric | Range | Better = | What it measures |
|---|---|---|---|
| Coverage Error | ≥ 1 | Lower | How deep you must go to find all true labels |
| LRAP | [0, 1] | Higher | How well true labels are concentrated at the top |
| Ranking Loss | [0, 1] | Lower | How often true labels are outranked by false ones |

---

## 10.7 One-vs-Rest vs. One-vs-One Decomposition

For non-native multi-class models (e.g., SVMs), you can decompose K-class problems into binary subproblems.

### 10.7.1 One-vs-Rest (OvR)

Train K binary classifiers. Classifier k learns to distinguish class k from all other classes combined.

**Example (K = 4 classes: A, B, C, D):**
```
Classifier 1: A vs {B, C, D}
Classifier 2: B vs {A, C, D}
Classifier 3: C vs {A, B, D}
Classifier 4: D vs {A, B, C}

Final prediction: argmax of the K classifier scores
```

**Evaluation consideration:** Each classifier faces class imbalance (1 class vs. K−1 classes). Use class weights or oversample the minority class.

### 10.7.2 One-vs-One (OvO)

Train K(K−1)/2 binary classifiers — one for every pair of classes. Final prediction by majority vote.

**Example (K = 4):** 4 × 3 / 2 = **6 classifiers** needed:
```
A vs B, A vs C, A vs D, B vs C, B vs D, C vs D
```

**Evaluation consideration:** Each classifier is balanced (equal representation of exactly two classes). Does not scale to large K — quadratic growth in number of classifiers.

### 10.7.3 OvR vs. OvO Comparison

| Property | One-vs-Rest (OvR) | One-vs-One (OvO) |
|---|---|---|
| Number of classifiers | K | K(K−1)/2 |
| Training data per classifier | Full dataset | Subset (two classes only) |
| Class imbalance per classifier | Yes (1 vs. K−1) | No (balanced pairs) |
| Scales to large K | ✅ Yes | ❌ No (quadratic growth) |
| Common users | Logistic regression | SVMs |
| Best for | Large K, fast training | Small K, balanced data |

---

## 10.8 Practical Metric Selection Guide

Use this decision tree to pick the right metric for any classification problem:

```
Does each sample get exactly one class label?
│
├── YES → Multi-class
│     ├── K = 2:                  Binary classification (precision, recall, F1, AUC-ROC)
│     ├── K small, balanced:      Accuracy + Macro F1
│     ├── K small, imbalanced:    Macro F1 + MCC + per-class F1 table
│     └── K large (100+):         Top-K Accuracy + Macro F1
│
└── NO  → Multi-label
      ├── Complete correctness required:      Subset Accuracy + Hamming Loss
      ├── Partial credit acceptable:          Sample F1 + Macro F1
      ├── Label frequency varies widely:      Micro F1 + per-label F1 breakdown
      └── Model outputs scores (not binary):  LRAP + Ranking Loss
```

---

## 10.9 Worked Example: News Article Document Tagging

### 10.9.1 Problem Setup

- **Task:** Tag news articles with topic labels
- **Dataset:** 10,000 articles, 20 possible tags, average 2.3 tags per article
- **Model output:** 20 sigmoid scores per article, thresholded at 0.5 (default)

### 10.9.2 Initial Results

| Metric | Value | Interpretation |
|---|---|---|
| Hamming Loss | 0.08 | 8% of label-article pairs are wrong |
| Subset Accuracy | 0.31 | Only 31% of articles have an exactly correct tag set |
| Sample F1 | 0.74 | Average per-article F1 is reasonable |
| Macro F1 | 0.61 | ⚠️ Low — some rare tags are poorly predicted |
| Micro F1 | 0.79 | Common tags predicted well |

### 10.9.3 Per-Label Breakdown

| Tag | Frequency | F1 Score | Diagnosis |
|---|---|---|---|
| politics | High | 0.91 | ✅ Frequent, easy to learn |
| economy | High | 0.85 | ✅ Frequent, easy to learn |
| obituary | Low | 0.43 | ⚠️ Rare, model under-predicts |
| weather | Low | 0.38 | ⚠️ Rare, model under-predicts |

### 10.9.4 Diagnosis

The gap between Micro F1 (0.79) and Macro F1 (0.61) immediately reveals the problem:
- **Micro F1 flatters the model** — it's dominated by high-frequency tags like "politics"
- **Macro F1 reveals the true problem** — rare tags are being systematically missed
- **Subset accuracy (0.31)** is harsh but meaningful: 69% of articles have at least one tag wrong

### 10.9.5 Improvement Steps and Results

| Step | Action Taken | Macro F1 |
|---|---|---|
| Baseline | Default threshold 0.5 for all labels | 0.61 |
| Step 1 | Per-label threshold tuning (lower thresholds for rare labels) | 0.71 |
| Step 2 | Add class weights for rare labels in the loss function | 0.76 |
| Step 3 | Augment training data for rare labels | 0.81 |

**Key lesson:** Always examine per-label F1 before relying on any aggregate metric. Aggregates hide exactly where the model fails — and in production, those failures matter most.

---

## 10.10 Interview Q&A Bank

### Q1: When would you report Micro F1 vs. Macro F1, and how would you explain the difference to a non-technical product manager?

**Why interviewers ask this:** This tests your ability to translate metric choices into business implications — a critical skill for Apple MLE roles where models serve diverse user populations.

**Answer:**

**Technical distinction:**
- **Macro F1** computes F1 separately for each class and averages them with equal weight. A class with 10 samples counts the same as a class with 10,000 samples.
- **Micro F1** aggregates all individual TP, FP, FN counts before computing F1. Frequent classes dominate entirely. For multi-class with one label per sample, Micro F1 equals accuracy.

**Decision rule:**

| Situation | Report | Why |
|---|---|---|
| All failure modes are equally costly | Macro F1 | Rare class failures aren't hidden |
| Overall real-world performance matters | Micro F1 | Reflects actual user-facing behavior |
| Class distribution is very imbalanced | Both | Each tells a different story |

**How to explain it to a non-technical PM:**

> "Macro F1 is like grading a student by averaging their score on every subject equally — even if geography class only has one question and math has a hundred. Micro F1 is like grading by total correct answers across all questions — the subject with the most questions dominates the grade. For our rare-event detector, we should use Macro F1 because we care equally about catching every type of problem, not just the common ones."

---

### Q2: A teammate proposes using subset accuracy as the primary metric for a 20-label document tagging system. What are the risks, and what would you recommend instead?

**Why interviewers ask this:** Metric selection is a product decision with real consequences. This tests engineering judgment about trade-offs between strictness and actionability.

**Answer:**

**Risks of subset accuracy as the primary metric:**

1. **It's extremely punishing for reasonable models.** A model that gets 19/20 tags right scores 0 on subset accuracy, identical to a model that gets 0/20 right. This makes it hard to compare models or measure improvement.
2. **It doesn't tell you where the model fails.** A low subset accuracy could mean the model is slightly wrong on one common tag, or completely wrong on rare tags — you can't distinguish these from the metric alone.
3. **It discourages incremental improvement.** If the threshold for "success" is perfection, small gains in quality go completely unmeasured and unrewarded, which can mislead model selection decisions.

**Recommended metric suite:**

| Metric | Role |
|---|---|
| Sample F1 | Primary metric — measures per-article quality with partial credit |
| Macro F1 | Detects rare-label failures |
| Micro F1 | Tracks overall volume-weighted performance |
| Subset accuracy | Secondary, reported for tasks where exact completeness is required |
| Per-label F1 table | Always reported for debugging and iteration |

**When subset accuracy IS appropriate:** Legal, compliance, or medical contexts where missing any tag has real regulatory or safety consequences.

---

### Q3: You deploy a 10-class classifier at Apple with 97% accuracy. One class represents a rare but high-stakes event (0.5% of data). How do you evaluate and monitor this model in production?

**Why interviewers ask this:** This is a production ML question. Apple deploys models to billions of devices — a rare-class failure at scale means millions of real failures. This tests your ability to think beyond benchmark metrics.

**Answer:**

**Why 97% accuracy is meaningless here:**

A model that ignores the rare class entirely can achieve 99.5% accuracy (by always predicting "not that class"). The 97% model might actually be worse than a naive baseline.

**Evaluation suite for this model:**

| Metric | Target | Why |
|---|---|---|
| Per-class Recall for the rare class | ≥ 90% | Catching rare events is the primary goal |
| Per-class Precision for the rare class | Monitor for FP flood | Low precision = alert fatigue in downstream systems |
| Macro F1 | ≥ 0.85 | Ensures rare class is not being ignored |
| Confusion matrix (normalized) | Visualize regularly | Reveals systematic misclassifications |
| MCC (Matthews Correlation Coefficient) | ≥ 0.75 | Single metric robust to class imbalance |

**Production monitoring additions:**

1. **Distribution drift detection** — monitor P(y = rare_class) over time. If the true prevalence changes, your threshold may need re-tuning.
2. **Precision-recall curve** — evaluate at multiple thresholds; don't commit to 0.5 in production.
3. **Calibration check** — does the model's stated confidence match actual accuracy? A 90%-confident prediction should be right 90% of the time.
4. **Slice-based evaluation** — evaluate separately across user segments (device type, region, age group) to detect subgroup disparities.
5. **Error rate alerting** — set production alerts if rare-class recall drops below a defined threshold within a sliding time window.

---

### Q4: Explain why Hamming Loss and Subset Accuracy can tell completely opposite stories about model quality on the same dataset. Give an example.

**Why interviewers ask this:** This tests whether you understand the complementary nature of metrics — a key skill when building production evaluation pipelines.

**Answer:**

**Key difference:**
- **Hamming Loss** measures the fraction of individual label-sample pairs that are wrong. It gives full partial credit.
- **Subset Accuracy** measures the fraction of samples with a perfectly correct full label set. It gives zero credit for partial correctness.

**Example that exposes the gap:**

Consider 3 samples with 5 labels each:

| Sample | True | Predicted | Hamming errors | Subset match? |
|---|---|---|---|---|
| 1 | [1,1,1,1,1] | [1,1,1,1,0] | 1/5 | ❌ |
| 2 | [0,0,0,0,0] | [0,0,0,0,0] | 0/5 | ✅ |
| 3 | [1,0,1,0,1] | [1,0,1,0,0] | 1/5 | ❌ |

```
Hamming Loss     = (1+0+1) / (3×5) = 2/15 = 0.133   ← "87% of labels correct, looks great!"
Subset Accuracy  = 1/3 = 0.33                         ← "Only 33% of samples are perfect"
```

**How to interpret this:** The model is consistently getting one label slightly wrong per sample. Hamming loss sees this as a small, distributed error (13%). Subset accuracy sees this as a near-total failure (67% of samples are "wrong"). Both are true — they just measure different things. Report both, and let the business requirement decide which is more actionable.

---

## 10.11 Rapid-Fire Flashcards

| # | Prompt | Answer |
|---|---|---|
| 1 | Multi-class vs. multi-label: output layer difference? | Multi-class: softmax. Multi-label: K independent sigmoids |
| 2 | Micro F1 equals what, for multi-class? | Accuracy (when every sample has exactly one true label) |
| 3 | Which average weights all classes equally? | Macro averaging |
| 4 | Which average reflects real data distribution? | Weighted averaging |
| 5 | Strictest multi-label metric? | Subset accuracy (exact match) |
| 6 | Most lenient multi-label metric? | Hamming loss |
| 7 | Hamming loss formula? | (1 / n×K) × Σ|yᵢₖ − ŷᵢₖ| |
| 8 | Subset accuracy formula? | (1/n) × Σ 𝟙[ŷᵢ = yᵢ] |
| 9 | Jaccard similarity? | |y ∩ ŷ| / |y ∪ ŷ| |
| 10 | Why use per-label thresholds in multi-label? | Rare labels need lower thresholds for adequate recall |
| 11 | LRAP range and direction? | [0, 1] — higher is better |
| 12 | Ranking Loss range and direction? | [0, 1] — lower is better |
| 13 | OvR classifiers needed for K classes? | K |
| 14 | OvO classifiers needed for K classes? | K(K−1)/2 |
| 15 | Top-5 accuracy is standard for which benchmark? | ImageNet (1,000 classes) |
| 16 | Macro F1 is low but Micro F1 is high — what does this signal? | Rare classes are being poorly predicted; common classes are fine |
| 17 | Best way to visualize multi-class confusion matrix with imbalanced classes? | Row-normalize by true class count |
| 18 | Why is softmax wrong for multi-label classification? | Softmax forces probabilities to sum to 1, making labels mutually exclusive |

---

## 10.12 Summary Table

| Concept | One-line takeaway |
|---|---|
| Multi-class vs. multi-label | One label per sample vs. any subset of labels |
| Macro F1 | Equal weight per class; sensitive to rare-class failures |
| Micro F1 | Dominated by frequent classes; equals accuracy in multi-class |
| Weighted F1 | Weighted by class frequency; reflects real distribution |
| Hamming Loss | Fraction of wrong label-sample pairs; lenient with partial credit |
| Subset Accuracy | Exact match of the full label set; very strict, no partial credit |
| Sample F1 | F1 averaged per sample; natural for variable label counts |
| Jaccard Similarity | Intersection over union; intuitive overlap measure |
| Per-label threshold tuning | Each label needs its own threshold in multi-label problems |
| Top-K Accuracy | True label in top K predictions; suited for large label spaces |
| Coverage Error | How deep you must go to cover all true labels |
| LRAP | Multi-label analogue of MAP; higher is better |
| Ranking Loss | Fraction of (true, false) label pairs incorrectly ordered |

---

## 10.13 Further Reading

1. Tsoumakas & Katakis — *Multi-label Classification: An Overview* (IJDWM, 2007)
2. Zhang & Zhou — *A Review on Multi-Label Learning Algorithms* (IEEE TKDE, 2014)
3. Read et al. — *Classifier Chains for Multi-label Classification* (Machine Learning, 2011)
4. scikit-learn documentation — *Multilabel Ranking Metrics* (comprehensive with code examples)

---

> **Next:** Chapter 11 — Probabilistic Metrics (Log-loss, Brier Score, ECE)
