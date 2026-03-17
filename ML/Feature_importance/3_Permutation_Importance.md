# Permutation Importance — The Complete Guide


## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [The Core Idea](#2-the-core-idea)
3. [Step-by-Step Numerical Example](#3-step-by-step-numerical-example)
4. [The Algorithm (Formal)](#4-the-algorithm-formal)
5. [Why It Works — The Intuition](#5-why-it-works--the-intuition)
6. [Comparison with Other Feature Importance Methods](#6-comparison-with-other-feature-importance-methods)
7. [Bias-Variance Implications](#7-bias-variance-implications)
8. [Correlated Features — The Big Gotcha](#8-correlated-features--the-big-gotcha)
9. [Train-set vs Test-set Permutation Importance](#9-train-set-vs-test-set-permutation-importance)
10. [Choosing the Number of Repeats (n_repeats)](#10-choosing-the-number-of-repeats-n_repeats)
11. [Negative Importances — What Do They Mean?](#11-negative-importances--what-do-they-mean)
12. [Practical Pitfalls & Checklist](#12-practical-pitfalls--checklist)
13. [Code — From Scratch & With sklearn](#13-code--from-scratch--with-sklearn)
14. [Worked Case Study](#14-worked-case-study)
15. [Summary Card](#15-summary-card)

---

## 1. What Problem Are We Solving?

After training a model, you naturally want to know: **which features actually matter?**

There are several ways to answer this, but they all have flaws:

| Method | Flaw |
|---|---|
| Coefficient magnitude (linear models) | Only works if features are scaled; breaks with interactions |
| Tree split impurity (Gini/entropy gain) | Biased towards high-cardinality features |
| SHAP | Computationally expensive; model-specific variants |
| **Permutation Importance** | ✅ Model-agnostic, intuitive, works on any metric |

Permutation Importance answers a beautifully simple question:

> **"If I scramble the values of feature X so the model can no longer use it, how much worse does the model get?"**

---

## 2. The Core Idea

![Permutation Importance Core Idea](https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/images/permutation-feature-importance-idea.png)

*Image: The fundamental idea — shuffle one column, measure performance drop.*

The logic:

- A **useful feature** → model relies on it → shuffling it **destroys** performance → **large importance**
- A **useless feature** → model ignores it → shuffling it **doesn't matter** → **near-zero importance**

That's it. No gradients. No retraining. Works on **any model** (Random Forest, XGBoost, SVM, Neural Net, Linear Regression — anything).

---

## 3. Step-by-Step Numerical Example

Let's build a tiny dataset and walk through the computation **by hand**.

### The Dataset

We want to predict **house price** from 3 features:

| # | Size (sqft) | Rooms | Random Noise | Price ($k) |
|---|---|---|---|---|
| 1 | 1500 | 3 | 0.42 | 200 |
| 2 | 2000 | 4 | 0.17 | 280 |
| 3 | 1200 | 2 | 0.91 | 160 |
| 4 | 1800 | 3 | 0.55 | 250 |
| 5 | 2500 | 5 | 0.33 | 350 |

We train a model (say, a simple linear regressor or RF — doesn't matter). Suppose our baseline **MAE = $10k** on this dataset.

---

### Step 1: Baseline Score

Score model on original data → **MAE = 10k**. Write it down.

---

### Step 2: Permute "Size"

Shuffle the **Size** column (leave everything else untouched):

| # | Size (shuffled) | Rooms | Random Noise | True Price |
|---|---|---|---|---|
| 1 | **2500** | 3 | 0.42 | 200 |
| 2 | **1200** | 4 | 0.17 | 280 |
| 3 | **2000** | 2 | 0.91 | 160 |
| 4 | **1500** | 3 | 0.55 | 250 |
| 5 | **1800** | 5 | 0.33 | 350 |

Now score the model on this corrupted data → **MAE = 48k**

**Importance(Size) = 48 − 10 = 38k**

---

### Step 3: Permute "Rooms"

Restore Size. Now shuffle **Rooms**:

Score → **MAE = 22k**

**Importance(Rooms) = 22 − 10 = 12k**

---

### Step 4: Permute "Random Noise"

Restore Rooms. Shuffle **Random Noise**:

Score → **MAE = 10.1k**

**Importance(Random Noise) = 10.1 − 10 = 0.1k ≈ 0**

---

### Result Table

| Feature | Baseline MAE | Permuted MAE | Importance (drop) |
|---|---|---|---|
| Size | 10k | 48k | **38k** ← most important |
| Rooms | 10k | 22k | **12k** |
| Random Noise | 10k | 10.1k | **0.1k** ← irrelevant |

This perfectly recovers the ground truth: Size matters most, Rooms somewhat, noise is useless.

---

### Why Repeat? (n_repeats > 1)

One shuffle is **random** — you might get an unusually "good" or "bad" shuffle. Standard practice is to repeat the shuffle `n_repeats` times and report:

- **Mean importance** = average drop across all shuffles
- **Std importance** = spread of drops (tells you stability)

Example with `n_repeats=5` for "Size":

| Repeat | Permuted MAE | Importance |
|---|---|---|
| 1 | 48k | 38k |
| 2 | 45k | 35k |
| 3 | 51k | 41k |
| 4 | 47k | 37k |
| 5 | 49k | 39k |
| **Mean** | | **38k** |
| **Std** | | **2.1k** |

The std is small → importance estimate is **stable**.

---

## 4. The Algorithm (Formal)

```
Input:  trained model f, dataset (X, y), metric M, n_repeats K

1. Compute baseline score:  s_base = M(f(X), y)

2. For each feature j = 1, ..., p:
     scores_j = []
     For k = 1, ..., K:
         X_perm = copy of X
         X_perm[:, j] = shuffle(X[:, j])       # permute column j
         s_perm = M(f(X_perm), y)               # score on corrupted data
         scores_j.append(s_perm)
     
     importance_j = mean(scores_j) - s_base     # for error metrics
     # OR: importance_j = s_base - mean(scores_j)  # for accuracy/R² metrics

3. Return importance_j and std(scores_j) for each j
```

**Sign convention:**
- If `M` is an **error metric** (MAE, MSE, log-loss): importance = permuted_score − baseline (positive = important)
- If `M` is a **performance metric** (accuracy, R², AUC): importance = baseline − permuted_score (positive = important)

---

## 5. Why It Works — The Intuition

![Feature Importance Comparison](https://scikit-learn.org/stable/_images/sphx_glr_plot_permutation_importance_001.png)

*Image: sklearn's own visualization showing permutation importance vs impurity-based importance — notice how impurity-based inflates uninformative features.*

Permutation breaks the **statistical relationship** between feature `j` and the target `y`, while keeping the **marginal distribution** of `j` intact.

This is crucial. You're not replacing values with zeros or means (which changes the distribution). You're asking: **"What if feature j carried no information about y?"**

Mathematically, permuting column j approximates sampling from the **marginal** distribution P(X_j) instead of the **conditional** distribution P(X_j | X_rest, y). The model can still "see" realistic values of X_j — they just don't correspond to the right rows anymore.

---

## 6. Comparison with Other Feature Importance Methods

![Feature Importance Methods](https://christophm.github.io/interpretable-ml-book/images/feature-importance-title.png)

| Property | Permutation | Impurity (Tree) | Coefficient | SHAP |
|---|---|---|---|---|
| Model-agnostic | ✅ Yes | ❌ Trees only | ❌ Linear only | ✅ Yes |
| Any metric | ✅ Yes | ❌ Fixed (Gini/entropy) | ❌ Implicit | ✅ Yes |
| Handles correlated features | ⚠️ Partially | ❌ Poor | ❌ Very poor | ✅ Better |
| Computationally cheap | ✅ Yes (no retraining) | ✅ Yes (free from tree) | ✅ Yes | ❌ Expensive |
| Accounts for interactions | ✅ Yes | ⚠️ Partially | ❌ No | ✅ Yes |
| Individual-level importance | ❌ Global only | ❌ Global only | ❌ Global only | ✅ Per-sample |

### Impurity vs Permutation — Concrete Difference

Impurity-based importance (in trees) measures **how much a feature was used during training**. Permutation importance measures **how much a feature is needed for good predictions**.

These sound similar but are very different when:

1. **Correlated features exist** — the tree might use Feature A everywhere (high impurity importance) but Feature B (correlated with A) would do equally well. Permutation would show both as important (or both as low, if they can substitute for each other).

2. **Overfit features exist** — a random feature that happened to split well in training gets high impurity importance but near-zero permutation importance.

---

## 7. Bias-Variance Implications

This is where it gets deep. Permutation importance has a **bias-variance tradeoff of its own** that most tutorials gloss over.

### 7.1 Variance of the Importance Estimate

Each permutation is a random experiment. With `n_repeats=1`, your estimate has **high variance** — you might get unlucky with a bad shuffle.

With `n_repeats=K`, variance of the importance estimate scales as:

```
Var(importance_j) ≈ Var(single permuted score) / K
```

So **doubling n_repeats halves the variance**. Typically `n_repeats = 10–30` is enough.

The std reported by sklearn (`result.importances_std[j]`) is the std **across repeats**, not across samples — it tells you how stable your permutation estimate is.

### 7.2 Bias from Model Overfitting (Train Set vs Test Set)

![Train vs Test Permutation Importance](https://scikit-learn.org/stable/_images/sphx_glr_plot_permutation_importance_multicollinear_001.png)

*Image: Permutation importance can differ drastically on train vs test set.*

This is the **single most important bias source** in permutation importance.

**On the training set:**
- An overfit model has memorized the data
- Even a noisy feature might have been memorized → permuting it hurts training performance
- Result: **noise features appear important** → biased upward for overfit models

**On the test set:**
- Memorized noise patterns don't generalize → permuting noise doesn't hurt test score
- Result: **noise features appear unimportant** → correct behavior

> **Rule: Always run permutation importance on the test set (or a held-out validation set) unless you have a specific reason for train-set importance.**

### 7.3 Numerical Example of This Bias

Suppose a Random Forest **overfits** and achieves 98% accuracy on train, 75% on test.

There's a random noise feature. On training data:

- Baseline accuracy (train): 98%
- Permuted accuracy (train): 95% (model memorized this noise feature!)
- **Importance(train) = 3%** ← falsely appears important

On test data:

- Baseline accuracy (test): 75%
- Permuted accuracy (test): 75.1% (noise feature adds nothing to generalization)
- **Importance(test) = -0.1%** ← correctly flagged as irrelevant

### 7.4 Model Complexity and Importance Stability

| Model Regime | Effect on Permutation Importance |
|---|---|
| **Underfitting** | All importances are near zero (model predicts poorly regardless of features) |
| **Well-fit** | Importances correctly reflect feature contributions |
| **Overfitting** | Train-set importances are inflated for noise features |

This is actually a useful diagnostic: if a feature shows high importance on train but near-zero on test, your model is **overfitting to that feature**.

### 7.5 Sample Size and Variance

With small datasets, the permuted score is noisy because:
- Few samples → high variance of the metric itself
- Permutation can accidentally create a "worse" or "better" dataset

Mitigation: use more `n_repeats`, or bootstrap the dataset before permuting.

---

## 8. Correlated Features — The Big Gotcha

![Correlated Features Problem](https://scikit-learn.org/stable/_images/sphx_glr_plot_permutation_importance_multicollinear_002.png)

*Image: When features are correlated, permutation importance underestimates both.*

This is the most cited limitation of permutation importance.

### The Problem

Say you have two features that are **99% correlated**:
- `weight_kg` and `weight_lbs` — literally the same information

When you permute `weight_kg`:
- The model can **fall back on** `weight_lbs` and still predict well
- So the importance of `weight_kg` appears **near zero**
- Same happens when you permute `weight_lbs`
- **Both features look unimportant!** But weight is clearly the most important predictor.

### Why This Happens

Permutation breaks the relationship between `X_j` and `y`, but leaves the relationship between `X_j` and `X_other` intact. When you shuffle `weight_kg`, `weight_lbs` is still in the data and still correlated with `y`. The model uses the substitute.

In fact, permuting `weight_kg` creates **out-of-distribution** samples: you now have rows where `weight_kg = 70kg` but `weight_lbs = 198 lbs` (which would correspond to 90kg). The model sees an impossible combination.

### Numerical Example

Dataset: predict fitness score from [weight_kg, weight_lbs, age]

```
Feature correlations:
  weight_kg ↔ weight_lbs: r = 0.99
  weight_kg ↔ age:         r = 0.05
  weight_lbs ↔ age:        r = 0.04
```

Actual causal structure: weight is the dominant predictor, age helps a little.

| Feature | True Importance | Permutation Importance |
|---|---|---|
| weight_kg | High | **Low** (model uses weight_lbs as proxy) |
| weight_lbs | High | **Low** (model uses weight_kg as proxy) |
| age | Medium | **Medium** (no correlated substitute) |

The result is misleading — age looks more important than weight!

### Solutions

1. **Remove one of the correlated features** before computing importance. Use clustering-based feature selection (e.g., hierarchical clustering on the correlation matrix) to group correlated features, then keep one representative per group.

2. **Use grouped permutation importance**: permute the entire group of correlated features simultaneously.

   ```
   Permute weight_kg AND weight_lbs together → now the model truly can't use weight
   ```

3. **Use SHAP with the right explainer** (though SHAP has its own issues with correlations).

4. **Partial Dependence Plots** alongside importance to get the full picture.

### Grouped Permutation — Code Sketch

```python
def grouped_permutation_importance(model, X, y, groups, metric, n_repeats=10):
    """
    groups: list of lists, e.g. [['weight_kg', 'weight_lbs'], ['age']]
    """
    baseline = metric(y, model.predict(X))
    importances = {}
    
    for group in groups:
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            # Shuffle ALL features in this group simultaneously
            perm_idx = np.random.permutation(len(X))
            X_perm[group] = X_perm[group].iloc[perm_idx].values
            scores.append(metric(y, model.predict(X_perm)))
        importances[tuple(group)] = np.mean(scores) - baseline
    
    return importances
```

---

## 9. Train-set vs Test-set Permutation Importance

A deeper look at when to use each:

### Use Test-set Importance When:
- Goal: understand what drives **generalization**
- Doing feature selection for production models
- Detecting overfitting (compare train vs test importances)
- Standard interpretability/explainability use case

### Use Train-set Importance When:
- Debugging data leakage (a feature that's suspiciously important on train but not test might be leaking future information)
- Understanding what the model has **learned** (not necessarily what's useful)
- Analyzing a model that was trained on ALL available data (no test set)

### What the Difference Tells You

```
Define:  ΔImportance_j = Importance_j(train) - Importance_j(test)

ΔImportance_j >> 0   →  model is overfitting to feature j
ΔImportance_j ≈ 0    →  feature j generalizes well
ΔImportance_j << 0   →  rare; possible when feature is noisy on training but
                         captures a true signal on test (small train set)
```

---

## 10. Choosing the Number of Repeats (n_repeats)

The right `n_repeats` is a tradeoff between **accuracy of the importance estimate** and **computational cost**.

### Confidence Interval Approach

After `K` repeats, the 95% CI for importance of feature `j` is approximately:

```
importance_j ± 1.96 * std_j / sqrt(K)
```

You want this CI to be **narrow enough to distinguish important from unimportant features**.

### Practical Heuristic

| Dataset Size | Recommended n_repeats |
|---|---|
| Small (<1k rows) | 30–50 (high variance per permutation) |
| Medium (1k–100k) | 10–20 |
| Large (>100k rows) | 5–10 (low variance already) |

If two features' importance estimates have **overlapping confidence intervals**, you cannot reliably rank them. Consider them tied.

### The Elbow Plot

Plot importance ± std for each feature. If a feature's CI overlaps zero → it's **not reliably important**. This is more informative than just looking at point estimates.

```
Feature A: 0.42 ± 0.03   ← clearly important
Feature B: 0.08 ± 0.12   ← CI overlaps zero; likely unimportant
Feature C: -0.01 ± 0.05  ← unimportant
```

---

## 11. Negative Importances — What Do They Mean?

You'll often see features with **negative permutation importance**. This is not a bug.

Negative importance means: **after permuting this feature, the model does BETTER** than the baseline.

### Why This Happens

1. **Noise features that hurt the model**: The model learned a spurious correlation with this feature. Permuting it removes that bad signal → the model actually improves.

2. **Small dataset + high variance**: The test set is small, so the permuted metric happens to be better by chance.

3. **Strong regularization**: The model barely uses the feature, and the permuted version accidentally aligns better.

### What To Do

- Negative importances close to zero → treat as **zero** (feature doesn't matter)
- Negative importances large in magnitude → the feature is **actively hurting** your model; consider removing it
- Many negative importances → your model may be overfitting; revisit regularization

---

## 12. Practical Pitfalls & Checklist

### ✅ Before You Run

- [ ] Is your model trained and finalized? (Permutation importance should be post-hoc)
- [ ] Do you have a proper held-out test set?
- [ ] Are your features scaled appropriately for your model?
- [ ] Have you checked feature correlations? (If r > 0.8, consider grouped permutation)

### ✅ Running It

- [ ] Use `n_repeats ≥ 10` (20–30 for small datasets)
- [ ] Use the **test set** by default
- [ ] Choose the right metric (match your training objective)
- [ ] Check std alongside mean importance

### ✅ Interpreting Results

- [ ] Features with CI overlapping zero → not reliably important
- [ ] Compare train vs test importances to detect overfitting
- [ ] Don't interpret feature importance as **causal** — it's **predictive**
- [ ] If correlated features both show low importance, inspect them as a group

### Common Mistakes

| Mistake | Consequence | Fix |
|---|---|---|
| Using train set instead of test set | Noise features appear important | Use test/validation set |
| n_repeats = 1 | Unstable estimates | Use n_repeats ≥ 10 |
| Ignoring correlated features | Underestimate group importance | Grouped permutation |
| Confusing predictive with causal importance | Wrong decisions | Be explicit about what importance means |
| Wrong sign convention (error vs score metric) | Flip all conclusions | Check sklearn docs / your own code |

---

## 13. Code — From Scratch & With sklearn

### From Scratch (no dependencies except numpy)

```python
import numpy as np

def permutation_importance(model, X, y, metric, n_repeats=10, random_state=42):
    """
    model:      any object with a .predict() method
    X:          numpy array, shape (n_samples, n_features)
    y:          numpy array, shape (n_samples,)
    metric:     callable(y_true, y_pred) → float
                Use an ERROR metric (MAE, MSE): higher = worse
    n_repeats:  number of permutation repeats per feature
    
    Returns:    dict with 'importances_mean', 'importances_std', 'importances'
                importances shape: (n_features, n_repeats)
    """
    rng = np.random.RandomState(random_state)
    
    baseline_score = metric(y, model.predict(X))
    n_features = X.shape[1]
    importances = np.zeros((n_features, n_repeats))
    
    for j in range(n_features):
        for k in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            permuted_score = metric(y, model.predict(X_perm))
            importances[j, k] = permuted_score - baseline_score
    
    return {
        'baseline_score': baseline_score,
        'importances': importances,
        'importances_mean': importances.mean(axis=1),
        'importances_std': importances.std(axis=1),
    }


# Example usage
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

result = permutation_importance(
    model, X_test, y_test,
    metric=mean_absolute_error,
    n_repeats=20,
    random_state=42
)

feature_names = load_diabetes().feature_names
for name, imp, std in zip(feature_names, result['importances_mean'], result['importances_std']):
    print(f"{name:10s}  importance: {imp:6.2f} ± {std:.2f}")
```

### With sklearn

```python
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

result = permutation_importance(
    model,
    X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring='accuracy'   # ← any sklearn scoring string works here
)

# result.importances_mean  → shape (n_features,)
# result.importances_std   → shape (n_features,)
# result.importances       → shape (n_features, n_repeats) — full matrix

import pandas as pd
import matplotlib.pyplot as plt

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

print(importance_df)

# Plot with error bars
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importance_df['feature'], importance_df['importance'],
        xerr=importance_df['std'], align='center')
ax.set_xlabel('Mean accuracy decrease')
ax.set_title('Permutation Importances (Test Set)')
plt.tight_layout()
plt.show()
```

### Grouped Permutation (for correlated features)

```python
def grouped_permutation_importance(model, X_df, y, groups, metric, n_repeats=10, random_state=42):
    """
    X_df:   pandas DataFrame
    groups: dict of {group_name: [col1, col2, ...]}
    metric: error metric (higher = worse)
    """
    rng = np.random.RandomState(random_state)
    baseline = metric(y, model.predict(X_df))
    results = {}
    
    for group_name, cols in groups.items():
        imps = []
        for _ in range(n_repeats):
            X_perm = X_df.copy()
            perm_idx = rng.permutation(len(X_df))
            X_perm[cols] = X_df[cols].iloc[perm_idx].values
            imps.append(metric(y, model.predict(X_perm)) - baseline)
        results[group_name] = {'mean': np.mean(imps), 'std': np.std(imps)}
    
    return results
```

---

## 14. Worked Case Study

### Problem: House Price Prediction with Leakage Feature

Let's say we have these features:
- `size_sqft` — actual house size
- `num_rooms` — number of rooms (correlated with size, r=0.75)
- `zip_code` — neighborhood
- `price_per_sqft_neighbor_avg` — **data leakage!** This was calculated using the target!
- `random_col` — pure noise

We train a Random Forest.

**Expected impurity-based importances (from RF .feature_importances_):**
```
price_per_sqft_neighbor_avg  0.45  ← leakage inflated this
size_sqft                    0.28
zip_code                     0.14
num_rooms                    0.10
random_col                   0.03
```

**Permutation importance on TEST SET:**
```
Feature                       Mean Imp    Std
price_per_sqft_neighbor_avg   0.31        0.04   ← still high (it's actually useful)
size_sqft                     0.29        0.03
zip_code                      0.17        0.02
num_rooms                     0.05        0.03   ← low (substituted by size_sqft)
random_col                    0.00        0.02   ← correctly zero
```

**Permutation importance on TRAINING SET:**
```
Feature                       Mean Imp    Std
price_per_sqft_neighbor_avg   0.52        0.03   ← overfit model over-relies on this
size_sqft                     0.24        0.03
zip_code                      0.10        0.02
num_rooms                     0.08        0.02
random_col                    0.04        0.02   ← falsely appears important on train!
```

**Diagnoses from comparison:**
1. `random_col`: 0.04 on train, 0.00 on test → **model overfit to noise**
2. `num_rooms`: low on both → correlated with `size_sqft`, the model uses size as proxy
3. `price_per_sqft_neighbor_avg`: high everywhere → investigate for leakage (it knows the target neighborhood pricing, which might use test-set info)

---

## 15. Summary Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PERMUTATION IMPORTANCE                            │
├─────────────────────────────────────────────────────────────────────┤
│ WHAT IT IS    Measure how much model performance drops when a       │
│               feature's values are randomly shuffled               │
│                                                                     │
│ FORMULA       importance_j = metric(shuffled_j) - metric(original) │
│               (for error metrics; flip sign for score metrics)     │
│                                                                     │
│ STRENGTHS     ✓ Model-agnostic                                      │
│               ✓ Any metric                                          │
│               ✓ No retraining needed                                │
│               ✓ Captures interaction effects                        │
│                                                                     │
│ WEAKNESSES    ✗ Biased on train set if model overfits               │
│               ✗ Underestimates importance of correlated features    │
│               ✗ High variance with n_repeats=1                      │
│                                                                     │
│ BIAS-VARIANCE ↑ n_repeats → ↓ variance of importance estimate      │
│               Train set importances → biased if overfit             │
│               Test set importances → unbiased for generalization    │
│                                                                     │
│ KEY PARAMS    n_repeats ≥ 10 (30 for small datasets)               │
│               Use TEST SET by default                               │
│               Choose metric matching your objective                 │
│                                                                     │
│ GOTCHAS       1. Correlated features → use grouped permutation      │
│               2. Negative importance → feature hurts the model      │
│               3. CI overlaps zero → feature not reliably important  │
│               4. Always compare train vs test importances           │
└─────────────────────────────────────────────────────────────────────┘
```

---
# Permutation Importance — Interview Q&A

> Conceptual questions only. No code. Covers depth, edge cases, and the kind of follow-up a good interviewer will push on.

---

## Table of Contents

1. [Fundamentals](#1-fundamentals)
2. [Algorithm & Mechanics](#2-algorithm--mechanics)
3. [Bias-Variance & Overfitting](#3-bias-variance--overfitting)
4. [Correlated Features](#4-correlated-features)
5. [Train vs Test Set](#5-train-vs-test-set)
6. [Edge Cases & Tricky Behaviour](#6-edge-cases--tricky-behaviour)
7. [Comparison with Other Methods](#7-comparison-with-other-methods)
8. [Practical & Design Decisions](#8-practical--design-decisions)
9. [Causal vs Predictive Interpretation](#9-causal-vs-predictive-interpretation)
10. [Hard Follow-ups (Senior Level)](#10-hard-follow-ups-senior-level)

---

## 1. Fundamentals

---

**Q1. What is permutation importance, in one sentence?**

> It measures how much a model's performance degrades when the values of a single feature are randomly shuffled, breaking its relationship with the target.

---

**Q2. Why shuffle the feature instead of just removing it or replacing it with zeros/mean?**

Removing or zero-filling changes the **marginal distribution** of the feature — the model now sees values it has never encountered during training, so the performance drop is partially caused by out-of-distribution inputs, not by the loss of information. Shuffling preserves the original distribution (same min, max, mean, variance) — it only destroys the **predictive signal** by decoupling the feature from the target. The attribution is therefore cleaner.

---

**Q3. Does permutation importance require retraining the model?**

No. The model is trained once and frozen. You simply corrupt the input at inference time and measure the resulting performance change. This makes it very cheap compared to methods like leave-one-feature-out (which retrains a new model per feature).

---

**Q4. Is permutation importance model-agnostic?**

Yes. Because it only requires a trained model with a `predict` method and a performance metric, it works on any model — linear regression, random forest, gradient boosting, SVM, neural networks, k-NN, etc. The model is treated as a black box.

---

**Q5. Can permutation importance be used for both classification and regression?**

Yes. You choose the metric that matches your task:
- Regression: MAE, RMSE, R²
- Classification: accuracy, log-loss, AUC, F1

The metric choice matters because different metrics weight errors differently. For example, if you care about ranking performance, AUC-based permutation importance is more meaningful than accuracy-based importance.

---

**Q6. What does a near-zero permutation importance mean?**

The feature carries no predictive signal that the model is using. Shuffling it doesn't change predictions in a way that affects the metric. This could mean the feature is genuinely uninformative, or it could mean another correlated feature is acting as a substitute (the model doesn't need this one because the other covers it).

---

## 2. Algorithm & Mechanics

---

**Q7. Walk me through the permutation importance algorithm step by step.**

1. Train your model on training data.
2. Evaluate the model on a held-out set → record the **baseline score** (e.g., MAE = 10).
3. For each feature `j`:
   - Copy the dataset.
   - Randomly shuffle only column `j` (all other columns stay intact, in their original row positions).
   - Score the model on this corrupted data.
   - `importance_j = corrupted_score − baseline_score` (for error metrics).
4. Repeat the shuffle `n_repeats` times per feature. Report the **mean** and **std** of importance across repeats.

---

**Q8. Why do we repeat the permutation multiple times (n_repeats > 1)?**

A single shuffle is a random event. By chance, one particular shuffle might break the feature's signal more or less than expected. Repeating produces a distribution of importance estimates per feature, letting you report a mean (accuracy) and a std (stability). A feature whose importance has high std is unreliable — you can't trust the point estimate. Confidence intervals that overlap zero indicate the feature is not reliably important.

---

**Q9. For an error metric (like MAE), should a more important feature have a positive or negative permutation importance?**

Positive. Shuffling an important feature worsens predictions → MAE goes up → `corrupted_MAE − baseline_MAE > 0`. For score metrics like accuracy or R², the sign flips: you compute `baseline_score − corrupted_score`, so a positive value still means "this feature is important."

---

**Q10. What does it mean if permutation importance gives a negative value for a feature?**

It means shuffling the feature actually **improved** the model's performance. This tells you the model was relying on a spurious or harmful signal in that feature. The feature is actively hurting predictions. In practice, you'd investigate and likely consider removing that feature from the model.

---

**Q11. Does the order in which you permute features matter?**

No. Each feature is permuted independently, one at a time, with everything else held at original values. The permutation of feature A does not affect the computed importance of feature B. (This contrasts with methods like Shapley values, which consider all possible orderings.)

---

## 3. Bias-Variance & Overfitting

---

**Q12. Does permutation importance have a bias-variance tradeoff of its own?**

Yes, two distinct ones:

- **Variance**: A single permutation is random → high variance. Increasing `n_repeats` reduces this variance at a rate of `1/sqrt(n_repeats)`.
- **Bias**: If you compute permutation importance on the **training set** and the model is overfit, noise features that were memorised will appear important. This is a systematic bias, not randomness. The fix is to always use the **test set**.

---

**Q13. Why does computing permutation importance on the training set bias the results for overfit models?**

An overfit model has memorised noise patterns in the training data. When you permute a noise feature on the training set, you disrupt those memorised patterns → performance drops → the feature appears important. On the test set, those memorised patterns never existed, so permuting the noise feature has no effect. The training-set importance is inflated for any feature the model overfitted to.

---

**Q14. If you see that a feature has high importance on the training set but near-zero importance on the test set, what does that tell you?**

The model has **overfit to that feature**. It learned a spurious correlation in training data that doesn't hold in general. This is a diagnostic signal — the feature might be noisy, or the model needs stronger regularisation. This comparison (train importance vs test importance) is actually a useful overfitting diagnostic beyond just train/test accuracy.

---

**Q15. If your model is heavily underfitting, what would you expect permutation importance values to look like?**

All importances would be close to zero. An underfitting model predicts poorly regardless of which feature you give it — it's not using any feature well. Shuffling any feature barely changes the already-bad performance. Permutation importance is only meaningful when the model has learned something.

---

**Q16. How does the size of the dataset affect permutation importance reliability?**

With small datasets:
- The metric itself has high variance (few samples → noisy estimate of MAE/accuracy).
- Each permuted score is therefore noisy.
- You need **more repeats** to average out the noise.

With large datasets, each metric evaluation is stable (law of large numbers), so fewer repeats suffice. As a rule of thumb: the smaller the dataset, the higher `n_repeats` should be.

---

## 4. Correlated Features

---

**Q17. What happens to permutation importance when two features are highly correlated?**

Both features tend to show **underestimated or near-zero importance**, even if they're collectively the most predictive features. When you shuffle feature A, the model falls back on feature B (which is correlated with A and still in its original order). It can still predict well, so the performance drop is small. The same thing happens when you shuffle B. Neither appears important despite both being highly predictive.

---

**Q18. Is the underestimation of correlated features a bug or a limitation?**

It's a **known limitation** of the method, not a bug. The algorithm is doing exactly what it's designed to do — measure how much performance drops when one feature is broken. If another feature can fully substitute, then breaking one feature doesn't hurt the model. The issue is one of interpretation: permutation importance measures **individual marginal contribution**, not **group contribution**. The limitation is in our interpretation, not the algorithm.

---

**Q19. How would you fix the correlated feature problem?**

Use **grouped permutation importance**: identify groups of correlated features (e.g., via hierarchical clustering on the correlation matrix), then permute all features in a group simultaneously. Now the model can't fall back on any substitute, and the group gets the importance it deserves. This treats the group as a single unit and measures its collective contribution.

---

**Q20. A colleague says "feature X has near-zero permutation importance, let's drop it." What caution would you raise?**

Check if feature X is correlated with another feature Y. If yes, X might be redundant — not because it's useless, but because Y already carries the same signal. Dropping X might be safe (since Y is there), but dropping both X and Y could significantly hurt the model. Near-zero permutation importance is not a safe automatic deletion signal when correlations are present. Always inspect the correlation structure first.

---

**Q21. Permuting one feature at a time generates out-of-distribution samples when features are correlated. Explain why.**

When features A and B are correlated, their joint distribution P(A, B) occupies a narrow region (e.g., if A is high, B tends to be high). When you permute A, you create combinations like "A is high from row 5, B is low from row 2" which would never naturally co-occur. The model now receives an input it never saw during training — it lies outside the training distribution. This is why permutation importance is sometimes described as producing adversarial or impossible inputs for correlated feature sets.

---

## 5. Train vs Test Set

---

**Q22. Should you compute permutation importance on the training set or the test set? Why?**

The **test set**, almost always. The goal is to understand what drives model generalisation. Test-set permutation importance measures the contribution of each feature to out-of-sample performance. Train-set importance is confounded by overfitting — it measures what the model memorised, not what generalises.

---

**Q23. Is there any legitimate use case for train-set permutation importance?**

Yes, a few:
1. **No test set exists** (model trained on all available data) — train-set importance is the only option.
2. **Debugging data leakage** — a feature with suspiciously high train importance but near-zero test importance might have leaked future information or target information.
3. **Understanding model behaviour** — if you want to know what the model has learned (not whether it generalises), train-set importance is appropriate.

---

**Q24. You compute permutation importance on the test set, but your test set is very small (e.g., 50 samples). Are the results trustworthy?**

Not without careful handling. With 50 samples, the metric is highly variable. Increase `n_repeats` significantly (30–50) to average out noise. Even then, report confidence intervals (mean ± std) rather than point estimates. If two features' confidence intervals overlap, you cannot reliably rank them. Consider bootstrapping the test set to further stabilise estimates.

---

## 6. Edge Cases & Tricky Behaviour

---

**Q25. What if two features are perfectly correlated (r = 1.0)? What would permutation importance show?**

Both would show near-zero importance. The model can perfectly substitute one for the other. When you shuffle feature A, feature B is still intact and provides all the same information. No performance drop. Same for B. Both appear useless despite collectively being highly predictive. This is the most extreme form of the correlated-feature problem.

---

**Q26. You have a feature that's important at training time but becomes unavailable at inference time. How should this affect your permutation importance strategy?**

This is practically important. You can intentionally permute that feature during evaluation to simulate its absence at deployment. This gives you a realistic estimate of model performance degradation when the feature is missing, effectively treating permutation as a proxy for "feature unavailable." This informs whether you need a fallback strategy.

---

**Q27. Can permutation importance detect feature interactions?**

It captures their effect **implicitly**. If features A and B interact, permuting A alone disrupts the interaction — the importance of A reflects both its main effect and its contribution via the interaction with B. However, you cannot decompose "how much of feature A's importance is due to its main effect vs the interaction." For that, you'd need SHAP interaction values or partial dependence-based decomposition.

---

**Q28. If a feature has high permutation importance, does it mean the feature causes the target?**

No. Permutation importance is purely **predictive**, not **causal**. A feature that's highly correlated with the target will show high importance even if it doesn't causally affect it. For example, shoe size predicts reading ability in children — but the cause is age. Shoe size would have high permutation importance in a naive model, yet it's not a cause. Causal attribution requires separate tools (instrumental variables, do-calculus, causal graphs).

---

**Q29. Does permutation importance work for multi-output models?**

Yes, but you need a metric that aggregates across outputs (e.g., mean MAE across targets, or macro-averaged F1). The importance then reflects each feature's contribution to that aggregate metric. If different outputs rely on different features, the aggregate importance might obscure this. You could also compute permutation importance separately per output and compare profiles.

---

**Q30. What happens if you permute all features simultaneously?**

The model receives pure noise — no feature has any relationship with any other feature or with the target. Performance collapses to whatever a random guesser would achieve. This is not meaningful as an importance estimate for any individual feature. Feature-wise importance requires permuting one feature (or one group) at a time.

---

## 7. Comparison with Other Methods

---

**Q31. How does permutation importance differ from impurity-based feature importance (e.g., Gini importance in Random Forests)?**

| Dimension | Permutation | Impurity (Gini) |
|---|---|---|
| Scope | Works on any model | Only trees |
| What it measures | Performance on held-out data | Reduction in node impurity during training |
| Overfit sensitivity | Controlled (use test set) | Biased — training metric |
| High-cardinality bias | No | Yes — favours features with many unique values |
| Interaction capture | Yes | Partial |

The critical practical difference: impurity importance measures what the tree used during training; permutation importance measures what actually helps on new data.

---

**Q32. How does permutation importance compare to SHAP?**

| Dimension | Permutation | SHAP |
|---|---|---|
| Scope | Global (dataset-level average) | Both global and per-sample |
| Correlation handling | Poor | Better (though still imperfect) |
| Computational cost | Low | High (especially for complex models) |
| Interaction decomposition | No | Yes (SHAP interaction values) |
| Model-agnostic | Yes | Yes (KernelSHAP); exact for trees (TreeSHAP) |

Use permutation importance for a fast, global sanity check. Use SHAP when you need per-sample explanations, interaction decomposition, or when you need to explain individual predictions to stakeholders.

---

**Q33. How does permutation importance compare to leave-one-feature-out (LOFO) importance?**

LOFO retrains the model from scratch with each feature removed and measures the performance difference. It answers "how much does this feature contribute to what the model can learn?" Permutation importance asks "how much does this feature contribute to what the already-trained model predicts?" LOFO is more expensive (retrains N+1 models) but captures a different signal — it's not affected by the correlated-feature problem in the same way because the model can adapt its weights when a feature is permanently absent.

---

## 8. Practical & Design Decisions

---

**Q34. How do you choose the right metric for permutation importance?**

Use the metric that matches your **actual business objective**, not just the training objective. Examples:
- If you care about rare-event detection → AUC or F1, not accuracy.
- If you care about large errors more than small ones → RMSE, not MAE.
- If predictions are probabilities → log-loss, not accuracy.

The same feature can have different importance ranks under different metrics, which is meaningful: it tells you the feature matters differently depending on what aspect of performance you care about.

---

**Q35. You have 500 features. How would you efficiently compute permutation importance?**

A few strategies:
1. **Parallelize across features** — each feature's permutation is independent, so all 500 can run in parallel.
2. **Pre-screen with a cheaper method** — use impurity importance or a univariate filter to narrow to top ~50 candidates, then apply permutation importance to those.
3. **Use lower n_repeats for the first pass** (n_repeats=5) to rank features, then apply higher n_repeats (n_repeats=20) to the top candidates to get reliable estimates.
4. **Reduce dataset size** — for very large datasets, subsampling to ~10k rows for the permutation step is usually sufficient and much faster.

---

**Q36. A stakeholder says "feature X has 3× the importance of feature Y — so X is 3× more important." Is that interpretation valid?**

Not strictly. Permutation importance is measured in units of the metric (e.g., "increase in MAE"), not in a normalised or ratio scale. A feature with importance 30 contributes 30 units of MAE when shuffled; a feature with importance 10 contributes 10. Saying it's "3× more important" is colloquially useful but not mathematically rigorous — the relationship is additive (in metric units), not multiplicative in some underlying importance space. Relative rankings are reliable; exact ratios should be communicated with care.

---

**Q37. How would you use permutation importance to decide which features to drop?**

A principled approach:
1. Compute permutation importance with confidence intervals on the test set.
2. Identify features whose CI **fully overlaps zero** — these are candidates for removal.
3. Among those, check for correlation with other features before dropping.
4. If you want to drop a correlated group, use grouped permutation importance first.
5. Retrain without the dropped features and compare test performance — the drop should be minimal.
6. Never drop based on importance alone without this final validation step.

---

## 9. Causal vs Predictive Interpretation

---

**Q38. A feature has high permutation importance. Can you say it's an important driver of the outcome in the real world?**

No. High permutation importance means the model relies on this feature for predictions. It does not imply causality. A feature can be important because:
- It directly causes the target (causal, genuinely important).
- It's correlated with a hidden cause (proxy variable, predictively useful but not causal).
- It leaks information about the target (data leakage, inflated importance without real-world relevance).

Distinguishing these requires domain knowledge, causal graphs, or controlled experiments — not permutation importance alone.

---

**Q39. A doctor asks you to identify which clinical features most strongly drive patient readmission risk. You run permutation importance on a hospital model. Can you give her the results directly as causal drivers?**

No — and this is a critical practical distinction. The results tell you which features the model uses most for prediction, which may or may not reflect clinical causation. For example, "day of week of discharge" might have high predictive importance (due to weekend staffing differences) without being a modifiable clinical driver. You'd present these as "features the model relies on most" and involve clinical experts to interpret which ones represent true, actionable drivers.

---

## 10. Hard Follow-ups (Senior Level)

---

**Q40. Permutation importance measures the marginal contribution of a feature. What does "marginal" mean here and what does it miss?**

"Marginal" means the feature is assessed independently, holding all other features at their observed values. It misses:
- **Interaction effects that depend on combinations** — if feature A is only useful when feature B is above a threshold, permuting A alone might show low marginal importance even though A is critical in that specific context.
- **Redundancy** — if A and B are redundant, both have low marginal importance even though the pair collectively has high importance.

Shapley values address this by averaging the marginal contribution of a feature across all possible subsets of other features — but that's exponentially more expensive.

---

**Q41. In a time series problem, what's wrong with standard permutation importance?**

Standard permutation shuffles rows, breaking temporal order within a feature. In a time series:
- Features often encode temporal lag structure (yesterday's value, rolling average, etc.).
- Shuffling row-wise destroys this temporal dependency in an unrealistic way.
- The model sees impossible temporal combinations (e.g., lag-1 feature that doesn't correspond to any real day).

The fix is **block permutation**: shuffle entire time windows (blocks) rather than individual rows, preserving local temporal structure while still breaking the feature's global signal.

---

**Q42. Can you use permutation importance to compare the importance of the same feature across two different models?**

Only if both models achieve similar baseline performance. Permutation importance is in units of the metric. If model A has baseline MAE = 10 and model B has baseline MAE = 25, a feature with importance 5 in model A and 8 in model B is not necessarily less important in A — model B might be worse overall, so the absolute drop is larger. To compare across models, consider relative importance: `importance_j / baseline_score`, which normalises by the model's overall performance level.

---

**Q43. What is the connection between permutation importance and the concept of "model reliance" in the statistics literature?**

Fisher, Rudin & Dominici (2019) formalised permutation importance as "Model Reliance" (MR). They defined it as the ratio of expected loss under feature permutation to expected loss under original data. They proved that model reliance is related to the **Rashomon set** — when many equally good models exist, a feature can have high importance in some models and zero in others, making any single model's permutation importance potentially misleading. This motivates computing importance across a set of equally performant models (PIMP or ensemble-level importance) rather than a single model.

---

**Q44. If you had to explain why permutation importance is preferred over coefficient magnitude for a regularised linear model, what would you say?**

Coefficient magnitude requires:
1. **Standardised features** — otherwise the scale of the feature contaminates the coefficient.
2. **No multicollinearity** — with correlated features, coefficients are unstable; the regulariser can push one coefficient near zero while another absorbs its effect.
3. **Model-specific structure** — only valid for linear models.

Permutation importance sidesteps all of this:
- No feature scaling required — you permute values, not multiply coefficients.
- Correlated features still have the redundancy problem, but the mechanism is explicit and diagnosable.
- Works on any model.

The only case where coefficient magnitude is more informative is when you want to know the **direction** of a feature's effect (positive or negative), which permutation importance doesn't tell you.

---

**Q45. Is permutation importance sensitive to class imbalance in classification?**

Yes, if you use accuracy as the metric. With severe class imbalance, a model can achieve high accuracy by predicting the majority class always. Permuting any feature won't hurt this trivial strategy much, so all importances will be near zero. Solution: use a metric appropriate for imbalanced data — AUC, F1 (macro or weighted), or PR-AUC. The feature importances will then reflect contributions to meaningful discrimination, not just majority-class prediction.

---

## Summary: What Interviewers Really Want to Hear

| Topic | What they're testing |
|---|---|
| Shuffle vs remove | Whether you understand distribution preservation |
| Train vs test set | Whether you understand overfitting's effect on importance |
| Correlated features | Whether you know the core limitation |
| Negative importance | Whether you understand it's meaningful, not an error |
| Causal vs predictive | Whether you can communicate limitations to non-technical audiences |
| n_repeats | Whether you understand variance in the estimate |
| Comparison to SHAP/Gini | Whether you know when to use which tool |
| Time series gotcha | Senior signal — whether you think about distributional assumptions |
