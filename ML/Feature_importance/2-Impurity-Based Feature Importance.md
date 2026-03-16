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
