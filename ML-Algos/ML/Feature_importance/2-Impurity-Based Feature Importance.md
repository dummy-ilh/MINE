# Impurity-Based Feature Importance (Gini / Entropy) — The Complete Guide

> How tree models score feature importance internally, why it's the most commonly misused method, and exactly how and when it breaks.

---

## Table of Contents

1. [What It Is](#1-what-it-is)
2. [Impurity — The Foundation](#2-impurity--the-foundation)
   - 2.1 Gini Impurity
   - 2.2 Entropy
   - 2.3 Variance Reduction (Regression)
   - 2.4 How They Compare
3. [How a Single Split Is Scored](#3-how-a-single-split-is-scored)
4. [MDI — Mean Decrease in Impurity](#4-mdi--mean-decrease-in-impurity)
   - 4.1 Formula
   - 4.2 Full Numerical Example
   - 4.3 In a Random Forest (aggregated)
5. [MDA — Mean Decrease in Accuracy](#5-mda--mean-decrease-in-accuracy)
6. [MDI vs MDA — Head-to-Head](#6-mdi-vs-mda--head-to-head)
7. [Bias 1 — High-Cardinality Bias](#7-bias-1--high-cardinality-bias)
   - 7.1 Why It Happens
   - 7.2 Numerical Example
   - 7.3 Fix
8. [Bias 2 — Training-Set Bias](#8-bias-2--training-set-bias)
   - 8.1 Why It Happens
   - 8.2 Numerical Example
   - 8.3 Fix
9. [Bias 3 — Correlated Features Bias](#9-bias-3--correlated-features-bias)
10. [Bias 4 — Class Imbalance Bias](#10-bias-4--class-imbalance-bias)
11. [Variance of MDI Estimates](#11-variance-of-mdi-estimates)
12. [When MDI Is Fine to Use](#12-when-mdi-is-fine-to-use)
13. [The Strobl Correction — Conditional Permutation Importance](#13-the-strobl-correction--conditional-permutation-importance)
14. [Bias-Variance Summary](#14-bias-variance-summary)
15. [Interview Q&A](#15-interview-qa)
16. [Summary Card](#16-summary-card)

---

## 1. What It Is

Every decision tree makes splits. Each split reduces the **impurity** of the resulting child nodes relative to the parent. The total impurity reduction attributed to each feature — summed across all splits and all trees in a forest — is its **impurity-based importance**, also called:

- **MDI** — Mean Decrease in Impurity
- **Gini Importance** (when using Gini as the impurity measure)
- **feature_importances_** in scikit-learn's RF and GBM

It is the **oldest**, **most available**, and **most misused** importance method.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X_train, y_train)
print(model.feature_importances_)   # This is MDI
```

It requires **no extra computation** — it's a free by-product of training. That convenience is both its selling point and the reason it gets used when it shouldn't be.

---

## 2. Impurity — The Foundation

Impurity measures how **mixed** the class labels are at a node. A pure node has all samples from one class. Three measures exist:

---

### 2.1 Gini Impurity

```
Gini(t) = 1 − Σₖ p(k|t)²

where p(k|t) = fraction of samples at node t belonging to class k
```

**Numerical example — binary classification:**

```
Node A: 90 class-0, 10 class-1  (mostly pure)
  p(0) = 0.90, p(1) = 0.10
  Gini(A) = 1 − (0.90² + 0.10²) = 1 − (0.81 + 0.01) = 0.18

Node B: 50 class-0, 50 class-1  (maximally impure)
  p(0) = 0.50, p(1) = 0.50
  Gini(B) = 1 − (0.50² + 0.50²) = 1 − (0.25 + 0.25) = 0.50

Node C: 100 class-0, 0 class-1  (perfectly pure)
  Gini(C) = 1 − (1.0² + 0.0²) = 0.0
```

Gini ranges from **0** (perfectly pure) to **0.5** (maximally impure, binary case) to **(1 − 1/K)** for K classes.

---

### 2.2 Entropy (Information Gain)

```
Entropy(t) = − Σₖ p(k|t) × log₂(p(k|t))
```

**Same examples:**

```
Node A: p(0)=0.9, p(1)=0.1
  Entropy(A) = −(0.9×log₂0.9 + 0.1×log₂0.1)
             = −(0.9×(−0.152) + 0.1×(−3.322))
             = −(−0.137 − 0.332) = 0.469 bits

Node B: p(0)=0.5, p(1)=0.5
  Entropy(B) = −(0.5×log₂0.5 + 0.5×log₂0.5)
             = −(0.5×(−1) + 0.5×(−1)) = 1.0 bit

Node C: pure → Entropy = 0
```

Entropy ranges from **0** (pure) to **log₂(K)** bits (maximally impure).

---

### 2.3 Variance Reduction (Regression Trees)

For regression, impurity = **variance** of the target at the node:

```
Var(t) = (1/n_t) × Σᵢ (yᵢ − ȳₜ)²

Reduction = Var(parent) − [n_L/n × Var(left)] − [n_R/n × Var(right)]
```

sklearn uses this for `RandomForestRegressor`.

---

### 2.4 How They Compare

```
Property           Gini         Entropy       Variance (regression)
─────────────────────────────────────────────────────────────────────
Computation        Faster        Slower (log)  Variance formula
Max impurity       0.5 (binary)  1.0 (binary)  depends on target
Sensitivity to     Slightly less  Slightly more N/A
  rare classes     sensitive      sensitive
In practice        Nearly         Nearly        N/A
  different?       identical      identical
sklearn default    ✅ Yes         criterion='entropy'  ✅ regression
```

In practice, **Gini and Entropy produce nearly identical trees** and nearly identical importances. The choice rarely matters.

---

## 3. How a Single Split Is Scored

This is the atomic unit of MDI. Every importance number traces back to this calculation.

**Formula for impurity decrease at one split:**

```
ΔImpurity(split) = (n_t / N) × [Impurity(t) − (n_L/n_t)×Impurity(L) − (n_R/n_t)×Impurity(R)]

where:
  n_t = samples reaching this node
  N   = total training samples
  L,R = left and right child nodes
```

The `n_t / N` weighting means splits higher in the tree (with more samples) contribute more to importance.

**Numerical Example:**

```
Dataset: N = 200 samples, binary classification.
Node t (root): n_t=200, Gini=0.45, splits on feature "Age"
  Left child:  n_L=120, Gini=0.28
  Right child: n_R=80,  Gini=0.35

ΔImpurity(Age, root) = (200/200) × [0.45 − (120/200)×0.28 − (80/200)×0.35]
                     = 1.0 × [0.45 − 0.60×0.28 − 0.40×0.35]
                     = 1.0 × [0.45 − 0.168 − 0.140]
                     = 1.0 × 0.142
                     = 0.142

Later split — node deeper in tree:
Node t2: n_t=40 (out of N=200), Gini=0.46, splits on "Income"
  Left:  n=25, Gini=0.20
  Right: n=15, Gini=0.38

ΔImpurity(Income, t2) = (40/200) × [0.46 − (25/40)×0.20 − (15/40)×0.38]
                      = 0.20 × [0.46 − 0.625×0.20 − 0.375×0.38]
                      = 0.20 × [0.46 − 0.125 − 0.1425]
                      = 0.20 × 0.1925
                      = 0.0385
```

Notice: the root split on Age contributes **0.142** while the deep split on Income contributes only **0.0385** — even though the local Gini reduction was similar. The `n_t/N` weight is the key.

---

## 4. MDI — Mean Decrease in Impurity

### 4.1 Formula

```
MDI(j) = Σ_{t: node splits on j} ΔImpurity(j, t)

For a Random Forest with T trees:
MDI(j) = (1/T) × Σ_{tree=1}^{T} Σ_{t in tree: split on j} ΔImpurity(j, t)

Normalised:
MDI_normalised(j) = MDI(j) / Σₖ MDI(k)    (sums to 1.0)
```

---

### 4.2 Full Numerical Example — Single Decision Tree

**Dataset:** 200 samples, predict churn (yes/no). Features: `age`, `tenure`, `monthly_charges`, `random_id`.

The tree makes these splits (summarised):

| Node | Depth | Feature | n_t | Gini (parent) | Gini (L) | n_L | Gini (R) | n_R | ΔImpurity |
|---|---|---|---|---|---|---|---|---|---|
| 1 (root) | 0 | monthly_charges | 200 | 0.47 | 0.31 | 110 | 0.38 | 90 | 0.115 |
| 2 | 1 | tenure | 110 | 0.31 | 0.18 | 70 | 0.42 | 40 | 0.072 |
| 3 | 1 | age | 90 | 0.38 | 0.25 | 55 | 0.44 | 35 | 0.055 |
| 4 | 2 | monthly_charges | 70 | 0.18 | 0.08 | 50 | 0.29 | 20 | 0.038 |
| 5 | 2 | random_id | 40 | 0.42 | 0.20 | 22 | 0.40 | 18 | 0.032 |
| 6 | 2 | tenure | 55 | 0.25 | 0.12 | 35 | 0.31 | 20 | 0.028 |
| 7 | 3 | age | 50 | 0.08 | 0.03 | 30 | 0.14 | 20 | 0.006 |

**Computing raw MDI (not normalised):**

```
MDI(monthly_charges) = Node1 + Node4 = 0.115 + 0.038 = 0.153
MDI(tenure)          = Node2 + Node6 = 0.072 + 0.028 = 0.100
MDI(age)             = Node3 + Node7 = 0.055 + 0.006 = 0.061
MDI(random_id)       = Node5         =          0.032 = 0.032
Total                                                  = 0.346
```

**Normalised:**

```
MDI(monthly_charges) = 0.153 / 0.346 = 0.442  (44.2%)
MDI(tenure)          = 0.100 / 0.346 = 0.289  (28.9%)
MDI(age)             = 0.061 / 0.346 = 0.176  (17.6%)
MDI(random_id)       = 0.032 / 0.346 = 0.093  ( 9.3%)
```

⚠️ **Note `random_id` gets 9.3% importance.** This is the high-cardinality bias — explored fully in Section 7.

---

### 4.3 In a Random Forest (aggregated)

Each tree gives a slightly different MDI (different bootstrap sample, different feature subsampling). The forest averages across all T trees:

```
MDI_RF(j) = (1/T) Σ_{tree} MDI_tree(j)
```

The more trees, the more stable the average. With 100 trees, the MDI values are fairly stable. With 10 trees, they can fluctuate significantly.

---

## 5. MDA — Mean Decrease in Accuracy

This is the **other** importance measure that comes from Random Forests. It's essentially permutation importance applied **on the out-of-bag (OOB) samples** during training.

**Algorithm:**

```
For each tree t in the forest:
  1. Use the OOB samples (samples not in bootstrap for this tree)
  2. Score the tree on OOB: acc_oob(t)
  3. For each feature j:
       Permute column j in the OOB samples
       Score again: acc_permuted(t, j)
       local_decrease(t, j) = acc_oob(t) − acc_permuted(t, j)

MDA(j) = mean over all trees of local_decrease(t, j)
```

**Key difference from plain permutation importance:**
- MDA uses OOB samples (close to a validation set, not truly held-out)
- MDA is computed during training (free, no separate test set needed)
- MDA is slightly pessimistic because OOB samples are ~37% of data per tree

**Numerical Example:**

```
Forest with 3 trees:

Tree 1: OOB acc = 0.82
  Permute age:    OOB acc = 0.74  → decrease = 0.08
  Permute income: OOB acc = 0.79  → decrease = 0.03

Tree 2: OOB acc = 0.79
  Permute age:    OOB acc = 0.72  → decrease = 0.07
  Permute income: OOB acc = 0.77  → decrease = 0.02

Tree 3: OOB acc = 0.85
  Permute age:    OOB acc = 0.76  → decrease = 0.09
  Permute income: OOB acc = 0.83  → decrease = 0.02

MDA(age)    = (0.08 + 0.07 + 0.09) / 3 = 0.080
MDA(income) = (0.03 + 0.02 + 0.02) / 3 = 0.023
```

MDA correctly identifies age as more important.

---

## 6. MDI vs MDA — Head-to-Head

| Property | MDI (Gini Importance) | MDA (OOB Permutation) |
|---|---|---|
| Computation | Free (from training splits) | Small overhead (OOB scoring) |
| Data used | Training data only | OOB (pseudo-validation) |
| High-cardinality bias | ✅ **Yes — major flaw** | ❌ Not affected |
| Correlated feature bias | ✅ **Yes** | ✅ **Yes** (both suffer) |
| Overfitting bias | ✅ **Yes** | Mild (OOB helps) |
| Speed | Instantly available | Slight overhead |
| sklearn availability | `model.feature_importances_` | Must use permutation_importance with oob or separate test |
| Preferred for | Quick sanity check | More reliable ranking |

**Rule:** Use MDI as a fast first-pass. Use MDA or test-set permutation importance for any decision that matters.

---

## 7. Bias 1 — High-Cardinality Bias

![Impurity vs Permutation for High Cardinality](https://scikit-learn.org/stable/_images/sphx_glr_plot_permutation_importance_001.png)

*Image: sklearn's canonical demonstration — impurity importance inflates the random numerical feature; permutation importance correctly scores it near zero.*

### 7.1 Why It Happens

A feature with many unique values (high cardinality) gives the tree **more split opportunities**. A continuous feature can be split at 1000 thresholds. A binary feature can only be split one way.

More split opportunities → more chances to reduce impurity → higher cumulative impurity reduction → inflated MDI.

This is **structural bias** — it doesn't reflect the feature's true predictive power. It reflects how many ways the feature can be sliced.

### 7.2 Numerical Example

**Dataset:** 500 samples, predict house price. Features:
- `size_sqft` — continuous, 500 unique values
- `is_renovated` — binary (0 or 1), 2 unique values
- `random_id` — random integer 1–500, 500 unique values (pure noise)

```
MDI results:
  size_sqft    0.51
  random_id    0.28   ← pure noise, but high cardinality!
  is_renovated 0.21   ← binary feature, only 1 split point

Permutation Importance (test set):
  size_sqft    0.19
  is_renovated 0.07
  random_id    0.00   ← correctly identified as noise
```

MDI gives `random_id` 28% importance because it had 500 unique values to exploit during training. Permutation importance correctly gives it 0%.

Meanwhile, `is_renovated` is underestimated by MDI (21%) because being binary, it only ever contributes one split per node.

### 7.3 Fix

- **Primary:** Use permutation importance on test set instead of MDI
- **Secondary if MDI required:** Use `max_features='sqrt'` in the RF (already default) — this partially mitigates the bias by limiting each tree's feature access
- **For categorical features:** Use target encoding or ordinal encoding before fitting; one-hot encoding creates many binary columns and reduces cardinality bias
- **Conditional permutation importance** (Strobl et al., 2008) — accounts for feature correlation and cardinality; described in Section 13

---

## 8. Bias 2 — Training-Set Bias

### 8.1 Why It Happens

MDI is computed entirely on the **training data**. If the model overfits:

- The tree memorises training samples
- Even noise features find small pockets in the training data where they happen to reduce impurity
- These spurious splits accumulate into a non-trivial MDI score
- On new data, those splits are worthless — but MDI doesn't know that

### 8.2 Numerical Example

**Setup:** Deliberate overfit — RF with no depth limit (max_depth=None) on 200 samples with a noise feature.

```
Train set MDI:                    Test set permutation importance:
  income       0.38                 income       0.15
  age          0.31                 age          0.11
  noise_col    0.18   ← inflated    noise_col    0.01  ← correctly near zero
  zip_code     0.13                 zip_code     0.09
```

`noise_col` gets 18% MDI because the overfit tree found splits on it that perfectly divided the 200 training samples — even though those splits mean nothing on new data.

### 8.3 Fix

- Use test-set permutation importance
- Regularise the model (max_depth, min_samples_leaf, max_features) to reduce overfitting before trusting MDI

---

## 9. Bias 3 — Correlated Features Bias

When two features are correlated, the tree can use either one for a split. Which one it chooses depends on:
- Random feature subsampling (RF's `max_features`)
- The random bootstrap sample
- Minor differences in Gini scores

The one that gets "chosen first" at a high node accumulates more MDI (larger `n_t/N` weight). The other one might get used lower in the tree or not at all, getting lower MDI.

**Result:** Among correlated features, MDI attribution is arbitrary — it depends on random chance in the forest construction, not on the features' actual predictive contributions.

**Numerical Example:**

```
True generating process: y = f(size) + noise
size_sqft and num_rooms have r=0.85 correlation.

RF1 (seed=42):
  size_sqft    0.52
  num_rooms    0.15   ← got used less at top of trees

RF2 (seed=99):
  size_sqft    0.21
  num_rooms    0.46   ← this time num_rooms got the top splits

Both forests have the same accuracy! MDI attribution is random.
```

**Fix:** Grouped permutation importance — permute both features simultaneously.

---

## 10. Bias 4 — Class Imbalance Bias

In highly imbalanced datasets (e.g., 95% negative, 5% positive), Gini impurity is dominated by the majority class.

At most nodes, the majority class dominates. A feature that perfectly predicts the minority class might achieve only a tiny Gini reduction (because the node was already "nearly pure" from the majority class's perspective). Its MDI is underestimated.

**Example:**

```
Node: 95 negatives, 5 positives. Gini = 1 − (0.95² + 0.05²) = 0.095 (already low!)

Feature A perfectly separates positives: 
  Left:  95 neg, 0 pos  → Gini = 0
  Right:  0 neg, 5 pos  → Gini = 0

ΔImpurity = (100/N) × [0.095 − (95/100)×0 − (5/100)×0] = (100/N) × 0.095

Feature B randomly splits:
  Left:  48 neg, 3 pos  → Gini ≈ 0.100
  Right: 47 neg, 2 pos  → Gini ≈ 0.079

ΔImpurity = (100/N) × [0.095 − 0.95×0.100 − 0.05×0.079]
          = (100/N) × [0.095 − 0.095 − 0.004] = small negative ≈ 0
```

Feature A perfectly predicts the minority class but gets barely more MDI than the useless Feature B. Gini impurity's low baseline for imbalanced data makes the ceiling of possible impurity reduction very small.

**Fix:** For imbalanced data, use class-weighted Gini, or switch to permutation importance with AUC or F1 as the metric (much more sensitive to minority class performance).

---

## 11. Variance of MDI Estimates

MDI is deterministic for a single tree but **random** across a Random Forest because:

1. **Bootstrap sampling** — each tree sees a different random 63% of the data
2. **Feature subsampling** — at each split, only `max_features` are considered
3. These introduce randomness in which feature gets the top splits

**Variance decreases with more trees:**

```
Var(MDI_RF(j)) ≈ Var(MDI_single_tree(j)) / T

With T=10 trees:   importances can shift substantially between runs
With T=100 trees:  importances are fairly stable
With T=500 trees:  importances are very stable
```

**Standard practice:** Use at least 100 trees before trusting MDI rankings.

**The std of MDI:** sklearn does not report std of `feature_importances_`. To get it:

```python
importances = [tree.feature_importances_ for tree in model.estimators_]
mean_importance = np.mean(importances, axis=0)
std_importance  = np.std(importances, axis=0)
```

Features with high std relative to mean are **unstably ranked** — their position in the ranking might change with a different random seed.

---

## 12. When MDI Is Fine to Use

Despite its biases, MDI is appropriate when:

1. **Speed is critical** — MDI is free; permutation importance costs extra inference passes
2. **First-pass exploration** — to get a rough ranking before a more rigorous analysis
3. **All features have similar cardinality** — the high-cardinality bias doesn't apply
4. **No highly correlated features** — correlation bias doesn't apply
5. **Model isn't overfit** — training-set bias is small when the model generalises well
6. **The ranking, not the magnitude, is what matters** — even biased MDI often gets the top features right when the above conditions hold

**Checklist before trusting MDI:**

```
☐ Similar cardinality across features?    (if not, don't trust it)
☐ Low feature correlations (VIF < 5)?     (if not, don't trust it)
☐ Model is not heavily overfit?           (check train vs test gap)
☐ No class imbalance > 90/10?             (if imbalanced, use AUC-based permutation)
☐ Just need a quick sanity check?         (then MDI is fine)
```

---

## 13. The Strobl Correction — Conditional Permutation Importance

Strobl et al. (2008) showed that the OOB permutation importance (MDA) is still biased when features are correlated, because OOB permutation still creates out-of-distribution samples (as we discussed in the permutation importance guide).

Their fix: **Conditional Permutation Importance** — permute feature j only within groups of samples that have similar values for all other features (conditioning on the other features' values).

This is mathematically elegant but:
- Computationally expensive
- Hard to implement correctly
- Available in R's `party` package but not standard in sklearn

**Practical stance:** For most production use, test-set permutation importance with grouped permutation for correlated features achieves the same goal more practically.

---

## 14. Bias-Variance Summary

```
MDI (Gini Importance)
│
├── BIASES (systematic errors)
│   ├── High-cardinality bias:    continuous > low-cardinality features
│   ├── Training-set bias:        overfit models inflate noise features
│   ├── Correlation bias:         one of two correlated features absorbs all credit
│   └── Class imbalance bias:     minority-class predictors underestimated
│
└── VARIANCE (random error)
    ├── Source: bootstrap + feature subsampling
    ├── Decreases as: 1/T (number of trees)
    └── Fix: use T≥100, report std across trees
```

**When MDI is most dangerous:**
- Features with very different cardinalities in the same dataset
- High-cardinality categorical features (before encoding)
- Datasets with strong feature correlations
- Overfit models

---

## 15. Interview Q&A

**Q: What is Gini impurity and how is it used in decision trees?**

Gini impurity measures the probability of misclassifying a randomly chosen sample at a node if the label were randomly assigned according to the class distribution at that node. A value of 0 means the node is pure. Trees greedily choose splits that maximise the weighted reduction in Gini impurity across the resulting children.

---

**Q: What is MDI and how does sklearn compute it for a Random Forest?**

MDI (Mean Decrease in Impurity) sums the Gini impurity reduction contributed by every split on a feature across all trees in the forest, weighted by the fraction of training samples reaching each node, then averages across trees. sklearn exposes it as `model.feature_importances_`.

---

**Q: What is the high-cardinality bias in Gini importance, and why does it happen?**

High-cardinality features (e.g., continuous variables, random IDs) offer more possible split thresholds. More thresholds → more opportunities to find a split that reduces impurity → higher cumulative MDI — even for a noise feature. Binary features have only one split point and are systematically underestimated.

---

**Q: A tree model gives a random noise column (unique integers per row) 25% feature importance. Why?**

This is the high-cardinality bias. Each unique integer creates a perfect split at the leaves — splitting off one sample at a time. Each split reduces impurity slightly. Accumulated over many nodes, this produces a large total MDI even though the feature is pure noise. On a held-out test set, permutation importance on this feature would be ~0.

---

**Q: What is the difference between MDI and MDA?**

MDI measures impurity reduction during training — it uses training data only and is fast but biased. MDA (Mean Decrease in Accuracy) permutes features on the OOB samples during forest training and measures accuracy drop — it uses a pseudo-validation set and is less biased toward high-cardinality features and overfitting, at the cost of slightly more computation.

---

**Q: You have two equally-good random forests trained with different random seeds. Their MDI rankings differ substantially for two correlated features. What's happening?**

This is the Rashomon + correlation problem. Both forests achieve the same accuracy but attribute credit differently between the correlated features. The tree that happens to use feature A at the root accumulates more MDI for A; one using feature B at the root accumulates more for B. Since both features carry the same information, either attribution is "correct" from the model's perspective. This illustrates that MDI reflects what the tree happened to use, not the ground truth importance of the features.

---

**Q: When is MDI acceptable to use over permutation importance?**

When speed is critical, when it's a quick first-pass exploration, when features have similar cardinality, when feature correlations are low, and when the model isn't heavily overfit. In all other cases, test-set permutation importance is more reliable.

---

## 16. Summary Card

```
┌─────────────────────────────────────────────────────────────────────┐
│  IMPURITY-BASED IMPORTANCE (MDI / Gini Importance)                  │
├─────────────────────────────────────────────────────────────────────┤
│  WHAT IT IS    Sum of Gini/entropy reductions per feature across     │
│                all splits, all trees, weighted by n_t/N             │
│                                                                     │
│  WHERE IT IS   model.feature_importances_  in sklearn               │
│                                                                     │
│  BIASES        1. High-cardinality → inflates continuous features   │
│                2. Training-set only → inflates overfit noise        │
│                3. Correlated features → credit assigned by chance   │
│                4. Class imbalance → underestimates minority signal  │
│                                                                     │
│  VARIANCE      High with few trees; use T≥100                       │
│                Report std across trees, not just mean               │
│                                                                     │
│  MDI vs MDA    MDI = training splits (biased)                       │
│                MDA = OOB permutation (less biased, costs more)      │
│                                                                     │
│  USE WHEN      Quick first-pass, similar cardinality features,      │
│                low correlations, non-overfit model                  │
│                                                                     │
│  DON'T USE     Mixed cardinality, correlated features, overfit,     │
│  WHEN          class imbalance > 90/10                              │
│                                                                     │
│  BETTER ALT    Permutation importance on test set (always)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

# Impurity-Based Feature Importance (Gini / Entropy) — Deep Reference

> How tree models score feature importance internally — built from first principles, every calculation shown, every bias proven with numbers.

---

## Table of Contents

1. [What It Is and Why It Exists](#1-what-it-is-and-why-it-exists)
2. [Impurity Measures — From Scratch](#2-impurity-measures--from-scratch)
   - 2.1 The Probabilistic Interpretation
   - 2.2 Gini Impurity — Full Derivation
   - 2.3 Entropy — Full Derivation
   - 2.4 Variance Reduction (Regression)
   - 2.5 Gini vs Entropy — When They Differ and When They Don't
3. [How a Tree Finds the Best Split](#3-how-a-tree-finds-the-best-split)
   - 3.1 The Threshold Scan Algorithm
   - 3.2 Full Worked Example — Scanning Every Threshold
   - 3.3 The n_t / N Weighting — Why It Matters
4. [Building a Full Tree — Node by Node](#4-building-a-full-tree--node-by-node)
   - 4.1 The Dataset
   - 4.2 Root Node — Finding the Best First Split
   - 4.3 Depth-1 Nodes — Continuing the Tree
   - 4.4 Final Tree Structure (ASCII)
5. [MDI — Accumulating Importance Across the Tree](#5-mdi--accumulating-importance-across-the-tree)
   - 5.1 The MDI Formula
   - 5.2 Computing MDI for Every Node
   - 5.3 Aggregating to Feature-Level Importance
   - 5.4 Normalisation
   - 5.5 The n_t/N Effect — Why Root Splits Dominate
6. [MDI in a Random Forest](#6-mdi-in-a-random-forest)
   - 6.1 Bootstrap Sampling and Its Effect on MDI
   - 6.2 Feature Subsampling (max_features)
   - 6.3 Aggregation Across Trees
   - 6.4 Variance of Forest-Level MDI
7. [MDA — Mean Decrease in Accuracy](#7-mda--mean-decrease-in-accuracy)
   - 7.1 OOB Samples — What They Are
   - 7.2 The OOB Permutation Algorithm
   - 7.3 Full Worked Example Across 3 Trees
   - 7.4 Why MDA Is Less Biased Than MDI
8. [MDI vs MDA — Head-to-Head](#8-mdi-vs-mda--head-to-head)
9. [Bias 1 — High-Cardinality Bias](#9-bias-1--high-cardinality-bias)
   - 9.1 The Mechanism — Proven with Numbers
   - 9.2 The Threshold Count Argument
   - 9.3 Fix
10. [Bias 2 — Training-Set Bias](#10-bias-2--training-set-bias)
    - 10.1 Mechanism
    - 10.2 Numerical Demonstration
    - 10.3 Fix
11. [Bias 3 — Correlated Features Bias](#11-bias-3--correlated-features-bias)
    - 11.1 The Credit Absorption Mechanism
    - 11.2 Numerical Example
    - 11.3 Fix
12. [Bias 4 — Class Imbalance Bias](#12-bias-4--class-imbalance-bias)
    - 12.1 Why Gini Fails at Low Prevalence
    - 12.2 The Ceiling Problem — Proven with Numbers
    - 12.3 Fix
13. [Variance of MDI Estimates](#13-variance-of-mdi-estimates)
14. [When MDI Is Acceptable](#14-when-mdi-is-acceptable)
15. [The Strobl Correction](#15-the-strobl-correction)
16. [Bias-Variance Summary](#16-bias-variance-summary)
17. [Interview Q&A](#17-interview-qa)
18. [Summary Card](#18-summary-card)

---

## 1. What It Is and Why It Exists

When a decision tree splits a node, it selects the feature and threshold that most reduce **impurity** — the mixedness of class labels. After the tree is fully grown, every split used some feature. Summing the total impurity reduction each feature contributed, weighted by how many samples that split affected, gives its **importance**.

This is called:
- **MDI** — Mean Decrease in Impurity
- **Gini Importance** (when the impurity metric is Gini)
- `model.feature_importances_` in sklearn

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
model.feature_importances_   # MDI, normalised to sum to 1.0
```

It is **free** — no additional computation after training. But free has a cost: it is the most commonly misapplied importance method in practice, for reasons we will prove numerically.

---

## 2. Impurity Measures — From Scratch

### 2.1 The Probabilistic Interpretation

At any node `t`, we have `n_t` training samples. Each belongs to one of K classes. Let:

```
p(k|t) = (number of class-k samples at node t) / n_t
```

**Impurity = how wrong you'd be if you predicted class at random according to p(k|t).**

A pure node (all one class) → impurity = 0. A maximally mixed node → impurity = maximum value.

---

### 2.2 Gini Impurity — Full Derivation

**Intuition:** Pick two samples randomly from node t. What's the probability they belong to different classes?

```
P(different classes) = 1 − P(same class)
                     = 1 − Σₖ P(pick class k twice)
                     = 1 − Σₖ p(k|t)²

Gini(t) = 1 − Σₖ p(k|t)²
```

This is a probability — it ranges from 0 to 1. For K=2 (binary):

```
Maximum when p(0)=p(1)=0.5:
  Gini = 1 − (0.5² + 0.5²) = 1 − 0.5 = 0.5

Minimum when p(0)=1, p(1)=0:
  Gini = 1 − (1² + 0²) = 0
```

**For K classes, maximum Gini = (K−1)/K.** For K=3: max = 2/3 ≈ 0.667.

**Full worked example — 4 classes:**

```
Node: 40 samples. Class counts: A=20, B=10, C=8, D=2

p(A) = 20/40 = 0.500
p(B) = 10/40 = 0.250
p(C) =  8/40 = 0.200
p(D) =  2/40 = 0.050

Gini = 1 − (0.500² + 0.250² + 0.200² + 0.050²)
     = 1 − (0.2500 + 0.0625 + 0.0400 + 0.0025)
     = 1 − 0.3550
     = 0.6450

Maximum possible (4 classes): (4−1)/4 = 0.75
This node is at 0.645 / 0.75 = 86% of maximum impurity.
```

---

### 2.3 Entropy — Full Derivation

**Intuition:** How many bits do you need to encode the class label of a random sample from this node?

If you know the class distribution p(k|t), the optimal code uses `-log₂(p(k))` bits for class k. Expected bits = entropy.

```
Entropy(t) = − Σₖ p(k|t) × log₂(p(k|t))

Convention: 0 × log₂(0) = 0  (pure node contributes 0)
```

**Full worked example — same 4-class node:**

```
p(A)=0.500, p(B)=0.250, p(C)=0.200, p(D)=0.050

log₂(0.500) = −1.000    →  0.500 × (−1.000) = −0.5000
log₂(0.250) = −2.000    →  0.250 × (−2.000) = −0.5000
log₂(0.200) = −2.322    →  0.200 × (−2.322) = −0.4644
log₂(0.050) = −4.322    →  0.050 × (−4.322) = −0.2161

Sum = −1.6805

Entropy = −(−1.6805) = 1.6805 bits

Maximum (4 classes, uniform): log₂(4) = 2.0 bits
This node is at 1.68 / 2.0 = 84% of maximum entropy.
```

**Information Gain** (what a split achieves in entropy terms):

```
IG(split) = Entropy(parent) − [n_L/n_t × Entropy(L) + n_R/n_t × Entropy(R)]
```

This is what sklearn minimises when `criterion='entropy'`.

---

### 2.4 Variance Reduction (Regression Trees)

For regression targets, impurity = **variance** of y-values at the node:

```
Var(t) = (1/n_t) × Σᵢ (yᵢ − ȳₜ)²

Variance Reduction = Var(parent) − [n_L/n_t × Var(L)] − [n_R/n_t × Var(R)]
```

**Worked example:**

```
Parent node: y = [100, 150, 200, 250, 300],  n=5,  ȳ=200
Var(parent) = [(100−200)² + (150−200)² + (200−200)² + (250−200)² + (300−200)²] / 5
            = [10000 + 2500 + 0 + 2500 + 10000] / 5 = 5000

Split: Left = [100,150,200], Right = [250,300]
  ȳ_L=150,  Var(L) = [(−50)²+(0)²+(50)²]/3 = 5000/3 = 1667
  ȳ_R=275,  Var(R) = [(−25)²+(25)²]/2      = 1250/2 =  625

Weighted child variance = (3/5)×1667 + (2/5)×625 = 1000 + 250 = 1250

Variance Reduction = 5000 − 1250 = 3750
```

---

### 2.5 Gini vs Entropy — When They Differ and When They Don't

Both measure impurity, both reach minimum at purity = 0, both reach maximum at uniform distribution. Their curves are nearly identical:

```
p(class 1)   Gini    Entropy/2 (scaled to 0–0.5 for comparison)
0.0          0.000   0.000
0.1          0.180   0.169
0.2          0.320   0.322
0.3          0.420   0.441
0.4          0.480   0.486
0.5          0.500   0.500   ← both max here (binary)
0.6          0.480   0.486
0.9          0.180   0.169
1.0          0.000   0.000
```

The key structural difference: **Entropy penalises extreme probabilities more steeply** — it is more sensitive to rare classes. **Gini is cheaper to compute** (no logarithm) — this is why it's the default.

In practice, trees built with Gini vs Entropy differ in fewer than 5% of splits on typical datasets. **For MDI, the choice barely matters. The importance rankings are nearly always identical.**

---

## 3. How a Tree Finds the Best Split

This is crucial to understand MDI, because MDI sums the outputs of this procedure across all splits.

### 3.1 The Threshold Scan Algorithm

For a **continuous** feature j with values sorted as v₁ < v₂ < ... < vₙ:

```
best_reduction = 0
best_threshold = None

For each candidate threshold τ ∈ {(v₁+v₂)/2, (v₂+v₃)/2, ..., (vₙ₋₁+vₙ)/2}:
    Left  = {samples where x_j ≤ τ}
    Right = {samples where x_j > τ}

    reduction = (n_t/N) × [Gini(t) − (n_L/n_t)×Gini(L) − (n_R/n_t)×Gini(R)]

    if reduction > best_reduction:
        best_reduction = reduction
        best_threshold = τ

→ The feature with the best best_reduction across all features wins the split
→ best_reduction is what gets added to that feature's MDI for this node
```

For a **categorical** feature with K categories, the tree tries all 2^(K−1) − 1 possible binary subsets (or a greedy approximation for large K).

**Key insight:** A continuous feature with n unique values has n−1 candidate thresholds. A binary feature has exactly 1. This is the entire root cause of high-cardinality bias — addressed fully in Section 9.

---

### 3.2 Full Worked Example — Scanning Every Threshold

**Dataset:** 10 samples, binary classification, 1 feature (Age).

```
Sample  Age   Class
  1      22     0
  2      25     0
  3      28     1
  4      31     1
  5      35     0
  6      38     1
  7      42     1
  8      47     0
  9      51     1
 10      58     1

N=10.  4 class-0, 6 class-1.
Root Gini = 1 − (0.4² + 0.6²) = 1 − 0.16 − 0.36 = 0.480
```

Candidate thresholds: 23.5, 26.5, 29.5, 33, 36.5, 40, 44.5, 49, 54.5

**Scanning each:**

```
τ=23.5:  Left={22}→{0:1,1:0}  Right={25..58}→{0:3,1:6}
  Gini(L)=0.000 n_L=1    Gini(R)=1−(3/9)²−(6/9)²=1−0.111−0.444=0.444 n_R=9
  Reduction = 1.0×[0.480 − (1/10)×0.000 − (9/10)×0.444] = 0.480−0−0.400 = 0.080

τ=26.5:  Left={22,25}→{0:2,1:0}  Right={28..58}→{0:2,1:6}
  Gini(L)=0.000 n_L=2    Gini(R)=1−(2/8)²−(6/8)²=1−0.0625−0.5625=0.375 n_R=8
  Reduction = 1.0×[0.480 − (2/10)×0.000 − (8/10)×0.375] = 0.480−0−0.300 = 0.180 ✅ best

τ=29.5:  Left={22,25,28}→{0:2,1:1}  Right={31..58}→{0:2,1:5}
  Gini(L)=1−(2/3)²−(1/3)²=0.444 n_L=3    Gini(R)=1−(2/7)²−(5/7)²=0.408 n_R=7
  Reduction = 0.480−(3/10)×0.444−(7/10)×0.408 = 0.480−0.133−0.286 = 0.061

τ=33:    Left={22..31}→{0:2,1:2}  Right={35..58}→{0:2,1:4}
  Gini(L)=0.500 n_L=4    Gini(R)=0.444 n_R=6
  Reduction = 0.480−(4/10)×0.500−(6/10)×0.444 = 0.480−0.200−0.267 = 0.013

τ=36.5:  Left={22..35}→{0:3,1:2}  Right={38..58}→{0:1,1:4}
  Gini(L)=0.480 n_L=5    Gini(R)=0.320 n_R=5
  Reduction = 0.480−0.5×0.480−0.5×0.320 = 0.480−0.240−0.160 = 0.080

τ=40:    Left={22..38}→{0:3,1:3}  Right={42..58}→{0:1,1:3}
  Gini(L)=0.500    Gini(R)=0.375
  Reduction = 0.480−0.6×0.500−0.4×0.375 = 0.480−0.300−0.150 = 0.030

τ=44.5:  Left={22..42}→{0:3,1:4}  Right={47..58}→{0:1,1:2}
  Gini(L)=0.490 n_L=7    Gini(R)=0.444 n_R=3
  Reduction = 0.480−(7/10)×0.490−(3/10)×0.444 = 0.480−0.343−0.133 = 0.004

τ=49:    Left={22..47}→{0:4,1:4}  Right={51,58}→{0:0,1:2}
  Gini(L)=0.500 n_L=8    Gini(R)=0.000 n_R=2
  Reduction = 0.480−(8/10)×0.500−(2/10)×0.000 = 0.480−0.400 = 0.080

τ=54.5:  Left={22..51}→{0:4,1:5}  Right={58}→{0:0,1:1}
  Gini(L)=0.494 n_L=9    Gini(R)=0.000 n_R=1
  Reduction = 0.480−(9/10)×0.494−0 = 0.480−0.444 = 0.036
```

**All results ranked:**

```
Threshold   Gini Reduction
─────────────────────────────
26.5        0.180   ← BEST SPLIT
23.5        0.080
36.5        0.080
49.0        0.080
54.5        0.036
40.0        0.030
29.5        0.061
33.0        0.013
44.5        0.004
```

**Best split: Age ≤ 26.5, Gini reduction = 0.180.**
Left child (age ≤ 26.5): samples 1 and 2, both class 0 → Gini = 0. Pure.
Right child (age > 26.5): 8 samples, mixed → continues splitting.

The value **0.180** is added to Age's MDI for this node.

---

### 3.3 The n_t / N Weighting — Why It Matters

Notice the formula always includes `(n_t / N)`. This scales each split's contribution by the fraction of training data that reached it.

**Effect:** Splits at the root (n_t = N → weight = 1.0) contribute maximally. Splits deep in the tree (say n_t = 10, N=200 → weight = 0.05) contribute almost nothing, even if the local Gini reduction is just as large.

```
Root split on feature A:    local ΔGini=0.18, n_t/N=1.00 → MDI contribution = 0.180
Depth-4 split on feature A: local ΔGini=0.18, n_t/N=0.05 → MDI contribution = 0.009

Same local split quality → 20× difference in MDI contribution.
```

**Implication:** The feature chosen for the first split wins the MDI race. Any mechanism that causes a feature to be picked earlier — more thresholds (high cardinality), higher correlation with the target, random luck — inflates its MDI disproportionately.

---

## 4. Building a Full Tree — Node by Node

### 4.1 The Dataset

```
N=12 samples. Predict Churn (Yes/No).
Features: tenure (months), monthly_charge ($), is_student (0/1)

 #  tenure  charge  student  churn
 1    3       80       0      Yes
 2    5       90       1      Yes
 3    6       75       0      Yes
 4   12       70       0      No
 5   18       85       0      No
 6   20       60       1      No
 7   24       55       0      No
 8   24       95       0      Yes
 9   30       50       1      No
10   36       65       0      No
11   40       90       1      Yes
12   48       45       0      No

Class distribution: 5 Yes, 7 No
Root Gini = 1 − (5/12)² − (7/12)² = 1 − 0.174 − 0.340 = 0.486
```

---

### 4.2 Root Node — Finding the Best First Split

We evaluate each feature's best threshold.

**monthly_charge, τ=72.5:**
```
Left  (charge≤72.5): samples 4,6,7,9,10,12 → {Yes:0, No:6} → Gini=0.000  n=6
Right (charge>72.5): samples 1,2,3,5,8,11  → {Yes:5, No:1} → Gini=1−(5/6)²−(1/6)²=0.278  n=6

Reduction = (12/12)×[0.486 − (6/12)×0.000 − (6/12)×0.278]
          = 1.0×[0.486 − 0 − 0.139]
          = 0.347   ← highest
```

**tenure, τ=9:**
```
Left  (tenure≤9):  samples 1,2,3 → {Yes:3, No:0} → Gini=0.000  n=3
Right (tenure>9):  samples 4..12 → {Yes:2, No:7} → Gini=1−(2/9)²−(7/9)²=0.346  n=9

Reduction = 1.0×[0.486 − (3/12)×0.000 − (9/12)×0.346]
          = 0.486 − 0 − 0.260 = 0.226
```

**is_student, τ=0.5:**
```
Left  (student=1): samples 2,6,9,11 → {Yes:2, No:2} → Gini=0.500  n=4
Right (student=0): samples 1,3,4,5,7,8,10,12 → {Yes:3, No:5} → Gini=0.469  n=8

Reduction = 1.0×[0.486 − (4/12)×0.500 − (8/12)×0.469]
          = 0.486 − 0.167 − 0.313 = 0.006
```

**Winner: monthly_charge ≤ 72.5 with reduction = 0.347.**

---

### 4.3 Depth-1 Nodes

**Left node (charge ≤ 72.5): samples 4,6,7,9,10,12 — all No. Gini=0. PURE → STOP (leaf: Predict No).**

**Right node (charge > 72.5): samples 1,2,3,5,8,11 → {Yes:5, No:1}. Gini=0.278. Continues.**

Best split on right node: **tenure ≤ 9**

```
Left  (tenure≤9): samples 1,2,3 → {Yes:3, No:0} → Gini=0.000  n=3
Right (tenure>9): samples 5,8,11 → {Yes:2, No:1} → Gini=0.444  n=3

Reduction = (6/12)×[0.278 − (3/6)×0.000 − (3/6)×0.444]
          = 0.500×[0.278 − 0 − 0.222]
          = 0.500×0.056 = 0.028
```

Both child nodes are small (n=3). Majority-class leaf prediction:
- Right-Left (tenure≤9): all Yes → **Predict Yes**
- Right-Right (tenure>9): 2 Yes, 1 No → **Predict Yes** (majority)

---

### 4.4 Final Tree Structure (ASCII)

```
                       ┌─────────────────────────────────┐
                       │  ROOT: N=12, Gini=0.486          │
                       │  Split: monthly_charge ≤ 72.5?   │
                       │  MDI contribution: 0.347          │
                       └──────────────┬──────────────────┘
                                      │
              ┌───────────────────────┴──────────────────────┐
              │ YES (charge ≤ 72.5)                          │ NO (charge > 72.5)
              │ n=6, Gini=0.000                              │ n=6, Gini=0.278
              │ {Yes:0, No:6}                                │ {Yes:5, No:1}
              │ ✅ PURE LEAF                                  │ Split: tenure ≤ 9?
              │ Predict: No                                   │ MDI contribution: 0.028
              └───────────────────────────────────────────────┴────────┬──────────┐
                                                                        │          │
                                                               YES (≤9)          NO (>9)
                                                               n=3, Gini=0        n=3, Gini=0.444
                                                               {Yes:3, No:0}      {Yes:2, No:1}
                                                               ✅ PURE LEAF        LEAF
                                                               Predict: Yes        Predict: Yes
```

**is_student was never used → MDI(is_student) = 0.**

---

## 5. MDI — Accumulating Importance Across the Tree

### 5.1 The MDI Formula

```
MDI(j) = Σ_{all nodes t that split on feature j}  ΔGini(j, t)

ΔGini(j, t) = (n_t / N) × [Gini(t) − (n_L/n_t)×Gini(L) − (n_R/n_t)×Gini(R)]
```

### 5.2 Computing MDI for Every Node

```
Node 1 — ROOT:
  Feature: monthly_charge
  n_t=12, N=12, Gini(t)=0.486, n_L=6, Gini(L)=0.000, n_R=6, Gini(R)=0.278
  ΔGini = (12/12)×[0.486 − (6/12)×0.000 − (6/12)×0.278]
        = 1.0 × [0.486 − 0 − 0.139] = 0.347

Node 2 — DEPTH 1 (right branch):
  Feature: tenure
  n_t=6, N=12, Gini(t)=0.278, n_L=3, Gini(L)=0.000, n_R=3, Gini(R)=0.444
  ΔGini = (6/12)×[0.278 − (3/6)×0.000 − (3/6)×0.444]
        = 0.500 × [0.278 − 0 − 0.222] = 0.500×0.056 = 0.028
```

### 5.3 Aggregating to Feature-Level Importance

```
MDI(monthly_charge) = 0.347   (from Node 1 only)
MDI(tenure)         = 0.028   (from Node 2 only)
MDI(is_student)     = 0.000   (never used)
─────────────────────────────
Total               = 0.375
```

### 5.4 Normalisation

sklearn divides by the total so all importances sum to 1.0:

```
MDI_norm(monthly_charge) = 0.347 / 0.375 = 0.925  (92.5%)
MDI_norm(tenure)         = 0.028 / 0.375 = 0.075  ( 7.5%)
MDI_norm(is_student)     = 0.000 / 0.375 = 0.000  ( 0.0%)
```

### 5.5 The n_t/N Effect — Why Root Splits Dominate

The root split contributed **0.347** (weight = 1.0 × local reduction 0.347).
The depth-1 split contributed only **0.028** (weight = 0.5 × local reduction 0.056).

Even though the local Gini reduction at depth-1 (0.056) wasn't negligible, the `n_t/N = 0.5` weight halved it.

**Thought experiment:** Suppose the tree had been forced to use tenure first:

```
Root split on tenure ≤ 9:
  Left:  {1,2,3} → {Yes:3, No:0} → Gini=0.000  n=3
  Right: {4..12} → {Yes:2, No:7} → Gini=0.346  n=9
  ΔGini = 1.0×[0.486 − 0.25×0.000 − 0.75×0.346] = 0.226
  MDI(tenure) += 0.226

Then depth-1 split on monthly_charge (right branch only, n=9):
  ΔGini ≈ (9/12) × [0.346 − ...]  ≈ 0.75 × similar_local_reduction

MDI(tenure)         ≈ 0.226 + smaller  ≈ 0.28
MDI(monthly_charge) ≈ 0.75 × some_reduction ≈ 0.17
```

**The feature ranking could flip simply based on which one was picked first.** The tree's greedy construction, combined with the n_t/N weighting, means MDI reflects order of use as much as true importance.

---

## 6. MDI in a Random Forest

### 6.1 Bootstrap Sampling and Its Effect on MDI

Each tree trains on a **bootstrap sample** — N samples drawn with replacement. About 63.2% of samples appear at least once (the rest are duplicates); ~36.8% are never selected (OOB samples).

**Effect on MDI:** Different bootstrap samples change the class distributions at each node → different Gini values → possibly different features win splits → different MDI per tree. Two trees can rank the same features very differently.

### 6.2 Feature Subsampling (max_features)

At each split, sklearn only considers a random subset of features — `max_features = sqrt(p)` by default for classification. This means at any given split, many features aren't even evaluated.

**Effect:** A feature might not compete for a root split in some trees (excluded by the random subset), giving it near-zero MDI in those trees. Averaged over many trees this stabilises, but with few trees the variance is high.

### 6.3 Aggregation Across Trees

```
MDI_RF(j) = (1/T) × Σ_{tree=1}^{T}  MDI_tree(j)
```

By the law of large numbers, as T → ∞, MDI_RF converges to the expected MDI of a single tree from the RF distribution.

### 6.4 Variance of Forest-Level MDI

```
Var(MDI_RF(j)) ≈ Var_across_trees(MDI_tree(j)) / T

T=10  trees: high variance; rankings can be unreliable
T=100 trees: reasonable stability
T=500 trees: very stable
```

**Extracting per-tree MDI to compute confidence intervals:**

```python
importances_per_tree = np.array([tree.feature_importances_
                                  for tree in rf_model.estimators_])
# shape: (n_estimators, n_features)

mean_imp = importances_per_tree.mean(axis=0)
std_imp  = importances_per_tree.std(axis=0)
cv_imp   = std_imp / (mean_imp + 1e-10)   # coefficient of variation

# cv > 0.3 means this feature's ranking is unstable
```

**Example output:**

```
Feature          Mean MDI   Std MDI   CV     Stable rank?
monthly_charge   0.347      0.021    0.06   ✅ Yes
tenure           0.028      0.019    0.68   ❌ No — gap ≈ std
is_student       0.000      0.001    —      ✅ Yes (stably zero)
```

Monthly_charge is stably the top feature. Tenure vs is_student cannot be reliably ranked — their gap is within the noise.

---

## 7. MDA — Mean Decrease in Accuracy

### 7.1 OOB Samples — What They Are

For each tree, the ~36.8% of training samples not included in its bootstrap are called **Out-of-Bag (OOB) samples**. The tree was never trained on them → they serve as an honest validation set for that specific tree.

The key insight of MDA: use OOB samples for permutation-based importance, getting a validation-set effect without a separate held-out set.

### 7.2 The OOB Permutation Algorithm

```
For each tree t:
  OOB_t = samples not used in tree t's bootstrap

  Step 1: Score tree on OOB_t with original features:
    score_oob(t) = metric(tree_t.predict(OOB_t), y_OOB_t)

  Step 2: For each feature j:
    OOB_perm = copy of OOB_t
    OOB_perm[:, j] = shuffle(OOB_t[:, j])   ← permute only feature j, only within OOB_t
    score_perm(t, j) = metric(tree_t.predict(OOB_perm), y_OOB_t)
    decrease(t, j) = score_oob(t) − score_perm(t, j)

MDA(j)     = mean over all T trees of decrease(t, j)
MDA_std(j) = std over all T trees of decrease(t, j)
MDA_se(j)  = MDA_std(j) / sqrt(T)   ← standard error

z_score(j) = MDA(j) / MDA_se(j)
           → z > 1.96 suggests feature is significantly important
```

### 7.3 Full Worked Example Across 3 Trees

**Setup:** 20 training samples. Features: A (predictive), B (noise).

```
Tree 1: Bootstrap uses 13 unique samples. OOB = {14, 16, 18, 20}  (4 samples)
  score_oob(1)         = 3/4 correct = 0.750
  Permute A in OOB:  scores 1/4 = 0.250  →  decrease(1, A) = 0.750−0.250 = 0.500
  Permute B in OOB:  scores 3/4 = 0.750  →  decrease(1, B) = 0.750−0.750 = 0.000

Tree 2: OOB = {2, 7, 11, 15, 18}  (5 samples)
  score_oob(2)         = 4/5 = 0.800
  Permute A:  1/5 = 0.200  →  decrease(2, A) = 0.600
  Permute B:  4/5 = 0.800  →  decrease(2, B) = 0.000

Tree 3: OOB = {1, 4, 9, 13, 16, 19}  (6 samples)
  score_oob(3)         = 5/6 = 0.833
  Permute A:  2/6 = 0.333  →  decrease(3, A) = 0.500
  Permute B:  5/6 = 0.833  →  decrease(3, B) = 0.000

MDA(A)     = (0.500 + 0.600 + 0.500) / 3 = 0.533
MDA(B)     = (0.000 + 0.000 + 0.000) / 3 = 0.000

MDA_std(A) = std(0.500, 0.600, 0.500) = 0.047
MDA_se(A)  = 0.047 / sqrt(3) = 0.027
z(A)       = 0.533 / 0.027 = 19.7   → significant
z(B)       = 0 / ...         = 0.0   → not significant
```

### 7.4 Why MDA Is Less Biased Than MDI

**MDI's training-set problem:** The tree memorised its training data. Noise features that happened to correlate with a few label patterns earn MDI through those spurious splits.

**MDA's OOB advantage:** OOB samples were never seen during training. The tree's memorised noise patterns don't transfer → permuting a noise feature on OOB samples has no effect on accuracy → MDA correctly gives it ~0 importance.

**MDA's residual limitation:** OOB data still comes from the training distribution. The correlated-feature problem (model using feature B as substitute when A is permuted) applies equally to MDA as to plain permutation importance.

---

## 8. MDI vs MDA — Head-to-Head

| Property | MDI | MDA |
|---|---|---|
| Where computed | Training splits | OOB permutation |
| sklearn access | `model.feature_importances_` | `permutation_importance` with OOB scoring |
| Extra compute | Zero | ~K extra inference passes per tree |
| High-cardinality bias | ✅ Severe | ❌ None |
| Training-set / overfit bias | ✅ Yes | ❌ Much less (OOB not memorised) |
| Correlated feature bias | ✅ Yes | ✅ Yes (same mechanism) |
| Class imbalance bias | ✅ Yes (Gini-based) | Fix: use AUC/F1 metric |
| Variance | Low per tree; 1/T with ensemble | Higher per tree (small OOB); 1/T with ensemble |
| Significance test | ❌ Not built in | ✅ z = MDA / se |
| Best use | Quick sanity check only | Better when no held-out test set exists |

**Recommendation:** If you have a test set → test-set permutation importance. If not → MDA beats MDI. Never use raw MDI for consequential feature-selection decisions.

---

## 9. Bias 1 — High-Cardinality Bias

### 9.1 The Mechanism — Proven with Numbers

A feature with more unique values has more candidate thresholds. More candidates = more chances to find a split that reduces Gini by luck — even for a pure noise feature. This is a **multiple comparisons problem**: scanning 99 thresholds and taking the best is like running 99 tests and reporting the most significant — the result is inflated.

### 9.2 The Threshold Count Argument

The expected maximum Gini reduction over n random thresholds grows approximately as `O(sqrt(log n))`. It grows without bound. A feature with 1000 unique values will accumulate more MDI than one with 10 unique values even if both are pure noise.

**Concrete demonstration:**

```
Dataset: 500 samples, balanced binary (50/50). 3 noise features:
  binary_noise   (values: 0,1)           → 1 threshold evaluated per node
  ten_cat_noise  (values: 0–9)           → 9 thresholds
  random_id      (unique int per sample) → 499 thresholds

Expected MDI purely from cardinality:
  binary_noise   ~0.06
  ten_cat_noise  ~0.12
  random_id      ~0.31

All three have exactly zero true predictive power.
MDI says random_id is 5× more important than binary_noise.
```

Permutation importance on a held-out test set would give all three ~0.

### 9.3 Fix

Primary: use test-set permutation importance instead of MDI.

Partial mitigations if MDI is required:
- Ensure all features go through consistent encoding (don't mix raw continuous with binary)
- Use `max_features='sqrt'` (default): limits the number of features evaluated per split, somewhat reducing the advantage of high-cardinality features
- Use MDA instead of MDI

---

## 10. Bias 2 — Training-Set Bias

### 10.1 Mechanism

MDI is computed from training splits. An overfit tree memorises patterns in training data. Noise features that spuriously correlated with a few labels in the training set earn split-based MDI. On new data, those splits do nothing — but MDI doesn't know that.

### 10.2 Numerical Demonstration

```
Experiment: N_train=100, N_test=20.
True features: age, income (both predictive).
Noise: one random float column.

RF with max_depth=None (fully grown, overfit):
  Train accuracy: 0.98  Test accuracy: 0.74  ← large overfit gap

  MDI:                        Test-set permutation importance:
    age     0.38                age     0.14
    income  0.41                income  0.16
    noise   0.21 ← inflated     noise   0.01 ← correctly near zero

The noise feature's MDI is 21× its true contribution.
```

### 10.3 Fix

Regularise the RF to reduce the gap before trusting MDI:

```python
RandomForestClassifier(
    max_depth=10,
    min_samples_leaf=5,
    min_impurity_decrease=0.001,
    max_features='sqrt'
)
```

Or, just use test-set permutation importance, which is immune to this bias.

---

## 11. Bias 3 — Correlated Features Bias

### 11.1 The Credit Absorption Mechanism

When features A and B are correlated (r=0.80), both carry nearly the same information. At the root, the tree picks whichever achieves a marginally higher Gini reduction. That feature absorbs the full n_t/N=1.0 weight. The other feature is relegated to lower nodes (n_t/N < 1.0) where its additional information is minimal (A already explained most of it). It accumulates far less MDI.

**Result:** The choice between correlated features at the root is sensitive to the bootstrap sample, feature subsampling randomness, and minor Gini differences. MDI allocation between them is essentially arbitrary.

### 11.2 Numerical Example

```
True model:   churn = f(tenure, monthly_charge) + noise
              cor(tenure, monthly_charge) = 0.80

RF seed=42:  monthly_charge gets root → MDI(charge)=0.48, MDI(tenure)=0.15
RF seed=99:  tenure gets root          → MDI(tenure)=0.42, MDI(charge)=0.21

Both forests: test AUC = 0.87 (identical performance)
Both MDI vectors are wrong. The true joint contribution ~0.60 is
correctly recovered by grouped permutation importance.
```

### 11.3 Fix

```
Step 1: Compute pairwise correlation matrix + VIF for all features
Step 2: Group features where |r| > 0.7 (or VIF > 5)
Step 3: Use grouped permutation importance:
        permute ALL features in a group simultaneously
Step 4: Report group-level importance — within-group attribution is inherently ambiguous
```

---

## 12. Bias 4 — Class Imbalance Bias

### 12.1 Why Gini Fails at Low Prevalence

With 95% negatives and 5% positives, the root node starts with Gini = 1 − 0.95² − 0.05² = 0.095. **The maximum possible Gini reduction from this node is 0.095** — even a perfect split can only earn 0.095 MDI credit. Compare to a balanced node (Gini=0.50) where a perfect split earns 0.50.

The ceiling is 5× lower for imbalanced nodes. Features that perfectly predict the minority class can earn no more MDI than a feature that slightly separates the majority class elsewhere.

### 12.2 The Ceiling Problem — Proven with Numbers

```
Dataset: 1000 samples. 950 negatives, 50 positives (5% positive rate).
Root Gini = 0.095.

Feature A: perfectly separates positives
  Left:  {0 neg, 50 pos} → Gini=0.000  n=50
  Right: {950 neg, 0 pos} → Gini=0.000  n=950
  ΔGini(A) = (1000/1000)×[0.095 − 0 − 0] = 0.095

Feature B: random 52/48 split of negatives (noise for minority class)
  Left:  {494 neg, 25 pos} → Gini = 1−(494/519)²−(25/519)²
                            = 1−0.904−0.002 = 0.094  n=519
  Right: {456 neg, 25 pos} → Gini ≈ 0.093  n=481
  ΔGini(B) = 0.095 − (519/1000)×0.094 − (481/1000)×0.093
           = 0.095 − 0.049 − 0.045 = 0.001

Feature A (perfect minority predictor): MDI contribution = 0.095
Feature B (noise):                      MDI contribution = 0.001

OK so far. BUT if Feature B gets used at depth-1 nodes where class balance
is better (e.g., after Feature A split off the positives):

Remaining right node: {950 neg, 0 pos} → pure already, no more splits.
Remaining left node: 50 positives only → pure, no more splits.

In this case Feature A dominates. But with a more realistic imperfect split:
if A separates 80% of positives, the remaining nodes have better balance,
and B can accumulate MDI in those regions — sometimes more than A.
```

The result is that MDI rankings in imbalanced datasets are highly sensitive to which feature first separates the minority class and to the tree's depth/regularisation settings.

### 12.3 Fix

Switch to permutation importance using AUC, F1, or precision-recall AUC:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    scoring='roc_auc',   # proportionally sensitive to minority class
    n_repeats=20
)
```

---

## 13. Variance of MDI Estimates

**Two sources of MDI variance in a Random Forest:**

1. **Bootstrap sampling** — each tree sees a different random subset → different class distributions at nodes → different splits → different MDI
2. **Feature subsampling** — at each split, only sqrt(p) features are considered → a feature might not compete for important high-level splits in some trees

Both are averaged out with more trees:

```
Var(MDI_RF(j)) ≈ Var_single_tree(j) / T

T=10:   CV ≈ 0.15–0.30  (rankings can be unreliable)
T=100:  CV ≈ 0.05–0.10  (reasonable for top features)
T=500:  CV ≈ 0.02–0.04  (very stable)
```

**Ranking instability — the hidden problem:** Even if the variance of MDI magnitudes is small, the **rank ordering** can be unstable for features with similar importances. Always check:

```
Gap between feature i and feature i+1  vs  std of MDI(i) and MDI(i+1)

If gap < max(std_i, std_{i+1}):  ranking is unreliable — treat as tied
```

---

## 14. When MDI Is Acceptable

MDI is reasonable only when **all** of these hold:

```
✅ Similar cardinality across all features     (all continuous, or all binary, etc.)
✅ Low pairwise correlations (|r| < 0.7)        (VIF < 5 for all features)
✅ Model is not heavily overfit                 (train − test accuracy gap < 0.10)
✅ Reasonable class balance                     (minority class > 10%)
✅ Using ≥ 100 trees                            (variance is otherwise too high)
✅ Goal is a quick rough ranking only           (not a final selection decision)
```

If any condition fails → test-set permutation importance.

---

## 15. The Strobl Correction

**Strobl et al. (2007)** formally proved the high-cardinality and correlation biases in MDI and MDA. Their proposed fix (**Strobl et al., 2008**):

**Conditional Permutation Importance:** Permute feature j only **within strata** of samples with similar values on all other features. This approximates sampling from P(X_j | X_rest) rather than the marginal P(X_j).

Benefits:
- Fixes high-cardinality bias: permuted values stay within realistic ranges given other features
- Fixes correlation bias: model can't use correlated substitute for j because the permuted values are conditioned on those features too

Drawbacks:
- Computationally expensive (nearest-neighbour search per sample)
- Requires tuning the neighbourhood bandwidth
- Only available in R's `party::cforest`, not sklearn
- For most practical work, test-set permutation importance + grouped permutation achieves the same goal

---

## 16. Bias-Variance Summary

```
MDI ERROR DECOMPOSITION
│
├── BIAS — systematic, doesn't average away with more data/trees
│   │
│   ├── High-cardinality bias
│   │     Cause:   n−1 thresholds for continuous vs 1 for binary
│   │     Effect:  continuous & high-cardinality features inflated
│   │     Severity: severe when features have very different cardinalities
│   │     Fix:     permutation importance
│   │
│   ├── Training-set / overfitting bias
│   │     Cause:   tree memorises noise in training data → earns MDI
│   │     Effect:  noise features inflated proportional to overfit degree
│   │     Severity: severe when train−test gap > 0.10
│   │     Fix:     regularise model + test-set permutation importance
│   │
│   ├── Correlated feature bias
│   │     Cause:   whichever correlated feature wins root absorbs credit via n_t/N
│   │     Effect:  MDI allocation between correlated features is arbitrary
│   │     Severity: severe when |r| > 0.7
│   │     Fix:     grouped permutation importance
│   │
│   └── Class imbalance bias
│         Cause:   Gini ceiling is low at imbalanced nodes
│         Effect:  minority-class predictors underestimated
│         Severity: severe when minority < 10%
│         Fix:     AUC/F1-based permutation importance
│
└── VARIANCE — random, averages away with more trees
      Cause:     bootstrap sampling + feature subsampling
      Effect:    MDI values shift across runs; rankings swap for close features
      Magnitude: Var(MDI_RF) ≈ Var(single tree) / T
      Detect:    coefficient of variation = std/mean across trees; flag if cv > 0.3
      Fix:       use T≥100 trees; report std; check gap vs std before trusting rank
```

---

## 17. Interview Q&A

**Q: Derive Gini impurity from first principles.**

Pick two samples randomly from a node. The probability they have different class labels is `1 − Σₖ p(k)²`. This is Gini impurity — the probability of misclassification if labels were assigned randomly according to the node's distribution. It equals 0 when the node is pure and reaches maximum (K−1)/K for K classes at uniform distribution.

---

**Q: Walk me through exactly how a decision tree selects the best split.**

For each feature, scan every candidate threshold (midpoints between sorted consecutive values). For each threshold, compute the weighted Gini reduction: `(n_t/N) × [Gini(parent) − (n_L/n_t)×Gini(L) − (n_R/n_t)×Gini(R)]`. The (feature, threshold) pair with the highest reduction is chosen. This is greedy and locally optimal — it doesn't guarantee globally optimal splits.

---

**Q: Why do root splits dominate MDI values?**

The `n_t/N` weighting in the MDI formula gives root splits a weight of 1.0. A split at depth d with n_t = N/2^d carries weight 1/2^d. A split at depth 4 carries only 1/16th the weight of an equally good root split. This means the feature chosen first accumulates most of the MDI regardless of whether it's truly the most important feature.

---

**Q: Why does a random ID column (unique integer per row) get high MDI in a Random Forest?**

High-cardinality bias. A unique ID column has N−1 candidate thresholds. At each node, the tree can find a threshold that splits off a small group of training samples in a way that slightly reduces Gini by chance. Summed across many nodes, this accumulates into non-trivial MDI. This is a multiple-comparisons problem — more thresholds tested means higher expected maximum reduction even for noise. On a held-out test set, permutation importance correctly assigns it ~0.

---

**Q: What are OOB samples, and how does MDA use them to fix MDI's training-set bias?**

OOB samples are the ~36.8% of training samples not included in a particular tree's bootstrap. Since the tree was never trained on them, they serve as an honest validation set. MDA permutes each feature in the OOB samples and measures the accuracy drop. Because the tree hasn't memorised OOB samples, spurious noise patterns don't transfer — noise features earn near-zero MDA. This is the key advantage over MDI.

---

**Q: You run the same RF twice with different seeds. Feature A's importance changes from 0.38 to 0.21, and feature B's from 0.15 to 0.32. Both models have identical test AUC. What's happening, and what would you do?**

This is the correlated features bias. A and B are likely correlated — both carry the same information, and whichever gets picked for high-level splits absorbs the joint MDI credit via n_t/N weighting. The random seed determines which one wins root splits. To confirm: check pairwise correlation. To fix: use grouped permutation importance (permute A and B simultaneously). Also increase n_estimators to reduce variance. Report group-level importance rather than trying to attribute credit individually.

---

**Q: Give an example of class imbalance making MDI unreliable.**

With 950 negatives and 50 positives, the root Gini is 0.095 — a ceiling 5× lower than a balanced dataset's 0.50. A feature perfectly predicting all 50 positives earns at most 0.095 MDI. A feature that slightly improves separation of the 950-member majority class might earn comparable MDI just by operating in more balanced sub-regions later in the tree. The result is that minority-class predictors are systematically underestimated. Use AUC-based permutation importance instead.

---

## 18. Summary Card

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MDI — IMPURITY-BASED IMPORTANCE (Gini / Entropy / Variance)             │
├──────────────────────────────────────────────────────────────────────────┤
│  FORMULA                                                                 │
│    MDI(j) = Σ_nodes_on_j  (n_t/N) × [Gini(t) − weighted_child_Gini]    │
│    Gini(t) = 1 − Σₖ p(k|t)²                                             │
│    Normalised: Σⱼ MDI(j) = 1.0  (sklearn default)                        │
│                                                                          │
│  IN SKLEARN                                                              │
│    model.feature_importances_   (MDI)                                    │
│    std via: [t.feature_importances_ for t in model.estimators_]          │
│                                                                          │
│  WHY ROOTS DOMINATE                                                      │
│    n_t/N = 1.0 at root; weight at depth d = (n_t/N) ≤ 1/2^d             │
│    Whichever feature gets picked first wins the MDI race                 │
│                                                                          │
│  FOUR BIASES                                                             │
│    1. High-cardinality:  more thresholds → inflated MDI                  │
│    2. Training-set:      overfit trees credit noise                      │
│    3. Correlated:        root-split winner absorbs joint credit           │
│    4. Imbalance:         Gini ceiling collapses at low prevalence        │
│                                                                          │
│  VARIANCE                                                                │
│    Source: bootstrap + feature subsampling                               │
│    Decreases as 1/T. Use T≥100. Check cv=std/mean; flag cv>0.3           │
│                                                                          │
│  MDI vs MDA                                                              │
│    MDI = training splits (biased, free)                                  │
│    MDA = OOB permutation (less biased, small overhead)                   │
│    Best = test-set permutation importance                                │
│                                                                          │
│  USE ONLY WHEN all of these hold:                                        │
│    similar cardinality, |r|<0.7, not overfit, balanced classes, T≥100   │
│                                                                          │
│  ALWAYS BETTER: 3_Permutation_Importance.md                              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

