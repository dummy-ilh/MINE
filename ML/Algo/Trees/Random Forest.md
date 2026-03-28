# Random Forest 
---

## What is a Random Forest?

A Random Forest is an **ensemble** of Decision Trees trained using two sources of randomness:

```
1. Bagging        → each tree trains on a different bootstrap sample of data
2. Feature subsampling → each split considers only a random subset of features
```

```
Training Data (n samples)
        │
   ┌────┴────┬─────────┬─────────┐
Bootstrap  Bootstrap  Bootstrap  ...
Sample 1   Sample 2   Sample 3
   │           │           │
 Tree 1      Tree 2      Tree 3   ...  (B trees total)
   │           │           │
pred_1      pred_2      pred_3
        │
   ┌────┴──────────────────┐
   │  Aggregate predictions │
   │  Classification → majority vote
   │  Regression     → mean
   └───────────────────────┘
```

---

# 1. The Algorithm

### Training

```
Algorithm: Random Forest Training
─────────────────────────────────
Input: Training data D (n samples, d features), B trees, m = max_features

For b = 1 to B:
  1. Draw bootstrap sample D_b:
       Sample n points FROM D WITH REPLACEMENT
       (~63.2% unique samples, ~36.8% duplicates)

  2. Grow a full (or near-full) decision tree T_b on D_b:
       At each node:
         a. Randomly select m features (out of d)
         b. Find the best split among those m features only
         c. Split the node
       Stop when min_samples_leaf is reached (usually 1)
       DO NOT PRUNE

  3. Store T_b

Output: Ensemble {T_1, T_2, ..., T_B}
```

### Prediction

```
Classification:
  For each tree T_b → get predicted class label
  Final prediction = majority vote across all B trees

Regression:
  For each tree T_b → get predicted value
  Final prediction = mean across all B trees
```

---

### Why Two Sources of Randomness?

```
Bagging alone:
  Trees trained on different data → somewhat different
  BUT if one feature dominates → all trees still split on it
  → Trees are still correlated → averaging doesn't help much

Feature subsampling:
  Each split only sees m < d features
  → Dominant feature sometimes excluded
  → Trees forced to find alternative splits
  → Trees are decorrelated
  → Averaging decorrelated trees dramatically reduces variance
```

> **Key insight:** Averaging B independent estimates with variance σ² gives variance σ²/B.  
> Averaging B *correlated* (ρ) estimates gives variance ρσ² + (1-ρ)σ²/B.  
> Reducing ρ (correlation) via feature subsampling is what makes RF powerful.

---

### Bootstrap Sampling — The Math

From n samples, draw n with replacement:
$$P(\text{sample } i \text{ NOT selected in one draw}) = 1 - \frac{1}{n}$$
$$P(\text{sample } i \text{ NOT selected in any of n draws}) = \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} e^{-1} \approx 0.368$$

So each tree trains on **~63.2%** of the data.  
The remaining **~36.8%** = **Out-of-Bag (OOB)** samples for that tree.

---

# 2. Key Parameters

| Parameter | What it controls | Effect of increasing |
|---|---|---|
| `n_estimators` | Number of trees | ↑ accuracy (up to a point), ↑ training time |
| `max_features` | Features at each split | ↓ → more decorrelated trees, ↑ → more like bagging |
| `max_depth` | Max depth of each tree | ↑ → lower bias, higher variance per tree |
| `min_samples_leaf` | Min samples at a leaf | ↑ → smoother, less overfit |
| `min_samples_split` | Min samples to split | ↑ → simpler trees |
| `bootstrap` | Use bootstrap sampling | False = use full data (Pasting, not Bagging) |
| `max_samples` | Size of bootstrap sample | ↓ → more randomness, more diverse trees |
| `oob_score` | Compute OOB error | True = free validation estimate |
| `n_jobs` | Parallel cores | -1 = use all cores |
| `random_state` | Reproducibility seed | — |
| `class_weight` | Handle imbalance | `'balanced'` recommended for imbalanced data |

### Most important parameters to tune:
```
1. n_estimators   → more is usually better; plateau around 100-500
2. max_features   → most impactful on bias-variance tradeoff
3. max_depth      → controls individual tree complexity
4. min_samples_leaf → smooths out noisy splits
```

---

# 3. Splitting Criteria

Random Forest uses the **same criteria as Decision Trees** — just applied at each node with random feature subsets.

| Criterion | Task | Formula |
|---|---|---|
| `gini` | Classification | $1 - \sum p_i^2$ |
| `entropy` | Classification | $-\sum p_i \log_2 p_i$ |
| `log_loss` | Classification | $-\sum p_i \ln p_i$ |
| `squared_error` | Regression | MSE — $\frac{1}{n}\sum(y_i - \bar{y})^2$ |
| `absolute_error` | Regression | MAE — $\frac{1}{n}\sum|y_i - \tilde{y}|$ |

The critical difference from a single tree: the split search is over **m randomly selected features**, not all d features.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',       # or 'entropy'
    max_features='sqrt',    # default for classification
    random_state=42
)

# Regression
rf_reg = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',
    max_features=1.0,       # default for regression (all features)
    random_state=42
)
```

---

### Extra Trees (ExtraTreesClassifier) — Even More Random

Extra Trees pushes randomness further: instead of finding the **optimal** threshold for each feature, it picks a **random threshold**.

```
Random Forest at each node:
  → Pick m random features
  → Find BEST threshold for each → pick best feature+threshold

Extra Trees at each node:
  → Pick m random features
  → Pick RANDOM threshold for each → pick best feature+threshold
```

| | Random Forest | Extra Trees |
|---|---|---|
| Threshold selection | Optimal | Random |
| Bias | Slightly lower | Slightly higher |
| Variance | Slightly higher | Lower |
| Training speed | Slower | ⚡ Faster |
| Overfitting | More likely | Less likely |

```python
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
```

---

# 4. Underfitting vs Overfitting

```
Too few trees          Optimal               Too complex trees
(n_estimators=3)       (n_estimators=200)    (max_depth=None,
                       (max_features='sqrt')  min_samples_leaf=1)
─────────────          ──────────────         ──────────────────
High variance          Low variance           Low bias
(noisy ensemble)       Low bias               High variance
Underfit possible      Great fit ✅           Overfit (rare but possible)
```

### The good news about overfitting in RF

Random Forest is **much harder to overfit** than a single Decision Tree because:

1. **Averaging** smooths out individual tree overfitting
2. **OOB error** provides a live monitor of generalisation

However, overfitting CAN still happen when:
- `n_estimators` is low AND `max_depth=None` with noisy data
- Dataset is tiny with many features
- `max_features` is too high (trees become too correlated)

### Diagnosing fit

```python
rf.fit(X_train, y_train)
train_score = rf.score(X_train, y_train)
oob_score   = rf.oob_score_           # ≈ test score, free!
test_score  = rf.score(X_test, y_test)

# Overfit:  train_score >> oob_score
# Underfit: train_score is low AND oob_score is low
# Good fit: train_score ≈ oob_score (both high)
```

---

# 5. Bias and Variance

### Single Tree vs Random Forest

```
Single Decision Tree (fully grown):
  Low Bias    ←  can fit any training pattern
  High Variance ← tiny data change → totally different tree

Random Forest:
  Low Bias    ←  each tree is still fully grown (low bias)
  Low Variance ← averaging B decorrelated trees ÷ variance
```

### The Variance Reduction Formula

For B trees with pairwise correlation ρ and individual variance σ²:

$$\text{Var}(\text{RF}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

As B → ∞:
$$\text{Var}(\text{RF}) \to \rho \sigma^2$$

**Implications:**
- More trees → variance drops (but plateaus at ρσ²)
- Lower correlation ρ → lower floor on variance
- `max_features` ↓ → ρ ↓ → lower variance floor

### How to control Bias and Variance

| To reduce Bias | To reduce Variance |
|---|---|
| ↑ `max_depth` | ↑ `n_estimators` |
| ↓ `min_samples_leaf` | ↓ `max_features` |
| ↑ `max_features` (more info per split) | ↑ `min_samples_leaf` |
| Use stronger base learners | ↓ `max_depth` |

### The RF Bias-Variance sweet spot

```
max_features = 'sqrt'   → sweet spot for classification
                          balances exploration vs decorrelation

max_features = 1.0      → all features = pure bagging, more correlated
max_features = 1        → one feature = maximum decorrelation, high bias
```

---

# 6. Pruning (or: Why RF Doesn't Need It)

Standard Decision Trees need aggressive pruning to prevent overfitting.  
**Random Forests deliberately grow deep, unpruned trees.**

### Why no pruning needed?

```
Single deep tree:
  Overfits training data badly
  High variance — test error is terrible

100 deep trees (Random Forest):
  Each overfits differently (different bootstrap + feature subsets)
  Averaging cancels out individual errors
  → Low variance without any pruning!
```

> Pruning reduces variance of ONE tree at the cost of increasing its bias.  
> RF achieves low variance by AVERAGING, not by pruning individual trees.  
> So you get low variance AND low bias simultaneously.

### That said — you can still regularise RF

RF has its own "pruning" equivalents:

| RF Parameter | Equivalent effect to |
|---|---|
| `max_depth` | Pre-pruning (max depth) |
| `min_samples_leaf` | Pre-pruning (min leaf size) |
| `max_leaf_nodes` | Pre-pruning (leaf budget) |

But these are less critical than in single trees — the ensemble already handles variance.

---

# 7. Feature Importance

Random Forest provides two types of feature importance.

---

### 7a. Mean Decrease in Impurity (MDI) — Built-in

Same as single tree: sum of impurity reductions weighted by sample count, averaged across all trees.

$$\text{Importance}(f) = \frac{1}{B} \sum_{b=1}^{B} \sum_{\text{nodes in } T_b \text{ splitting on } f} \frac{n_{\text{node}}}{n} \cdot \Delta\text{impurity}$$

```python
rf.fit(X_train, y_train)
importances = rf.feature_importances_   # shape: (d,)
# Averaged across all B trees automatically
```

**Pros:** Fast, no extra computation  
**Cons:** Biased toward high-cardinality and continuous features (same issue as single tree — worse because many trees all share this bias)

---

### 7b. Permutation Importance — Unbiased

Shuffle one feature → measure accuracy drop → importance = that drop.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,        # shuffle each feature 10 times, average results
    random_state=42
)
# result.importances_mean  → mean importance per feature
# result.importances_std   → std across repeats
```

**Always use X_test (not X_train)** — using training data gives inflated importances for overfit features.

**Pros:** Unbiased, model-agnostic, evaluates real generalisation impact  
**Cons:** Slower, requires a test set, can be misleading with correlated features

---

### 7c. SHAP Values (best practice)

SHAP (SHapley Additive exPlanations) gives per-sample, per-feature contribution — not just global averages.

```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Gives: "For THIS prediction, feature X contributed +0.3 toward Class 1"

---

### When importances are misleading — correlated features

```
Features: Height and Arm_Span (highly correlated, both predict weight)

RF splits on Height in half the trees, Arm_Span in the other half.
→ Both show moderate importance.

But if you remove Height, Arm_Span takes over.
→ Removing either doesn't hurt accuracy much.
→ MDI and permutation importance both understate their true importance.
```

Fix: use SHAP or group correlated features before computing importance.

---

# 8. Out-of-Bag (OOB) Error

One of RF's most powerful features — a **free cross-validation estimate**.

### How it works

```
Tree 1 trained on samples: {1,2,3,5,7,8,9,10} (bootstrap)
OOB samples for Tree 1:   {4, 6}               (not used)
→ Use Tree 1 to predict samples 4 and 6

Tree 2 trained on: {1,2,4,5,6,8,9,10}
OOB samples:       {3, 7}
→ Use Tree 2 to predict 3 and 7

...repeat for all B trees

For each sample i:
  Collect predictions only from trees where i was OOB
  → Average/vote those predictions
  → Compare to true label → OOB error
```

```python
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)
print(rf.oob_score_)          # ≈ test accuracy
print(rf.oob_decision_function_)  # per-sample probabilities
```

### OOB vs Cross-Validation

| | OOB | k-Fold CV |
|---|---|---|
| Extra training needed? | ❌ No | ✅ Yes (k fits) |
| Cost | Free | Expensive |
| Accuracy of estimate | Slightly pessimistic | More reliable |
| Works for any model? | ❌ RF only | ✅ Any model |

---

# 9. Handling Missing Values

---

### Strategy 1: Impute Before Training (sklearn RF)

Same as Decision Tree — sklearn's RF cannot handle NaN natively.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(n_estimators=100))
])
pipe.fit(X_train, y_train)
```

---

### Strategy 2: RF-Based Imputation (iterative, powerful)

Random Forests can impute their own missing values iteratively — this is called **MissForest**.

```
Algorithm: MissForest
──────────────────────
1. Initial imputation: fill NaN with median/mode
2. Repeat until convergence:
   For each feature f with missing values:
     a. Train RF using all other features on observed rows of f
     b. Predict f for rows where f is missing
     c. Replace missing f values with RF predictions
3. Stop when imputed values stop changing
```

```python
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10),
    max_iter=10,
    random_state=42
)
X_imputed = imputer.fit_transform(X_train)
```

**Why this works:** RF captures non-linear relationships between features — imputed values respect complex patterns in data, not just column means.

---

### Strategy 3: XGBoost / LightGBM (learn optimal direction)

Same as Decision Tree — these learn the best routing for missing values during training.  
Generally preferred in production.

---

# 10. Regression vs Classification

### RandomForestClassifier

```python
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)

rf_clf.predict(X_test)          # majority vote across trees
rf_clf.predict_proba(X_test)    # averaged class probabilities across trees
```

**Probability estimation:** Each tree gives 0/1 (or class fraction), averaged across trees → smooth probability. RF probabilities are generally **better calibrated** than a single tree.

---

### RandomForestRegressor

```python
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train, y_train)

rf_reg.predict(X_test)          # mean prediction across trees
```

**Variance estimation (bonus):** You can get prediction uncertainty for free:

```python
# Get individual tree predictions
tree_preds = np.array([tree.predict(X_test) for tree in rf_reg.estimators_])
# shape: (n_estimators, n_test_samples)

mean_pred = tree_preds.mean(axis=0)   # final prediction
std_pred  = tree_preds.std(axis=0)    # uncertainty estimate!
```

---

### RF Regression vs Single Tree Regression

| | Single Reg Tree | RF Regressor |
|---|---|---|
| Prediction | Step function (piecewise constant) | Smoother (average of step functions) |
| Extrapolation | Cannot (same leaf value) | Cannot (average of leaf values — still bounded) |
| Variance | High | Low |
| Interpretability | High | Low |
| Uncertainty estimate | ❌ | ✅ (std across trees) |

> **Both** single trees and RF cannot extrapolate beyond training range.  
> This is a fundamental limitation of tree-based models.

---

# 11. Class Imbalance

### How imbalance affects RF

Same root problem as Decision Trees — majority class dominates impurity at each split. Compounded in RF: **all B trees** learn this bias independently.

---

### Fix 1: `class_weight='balanced'` or `'balanced_subsample'`

```python
# 'balanced' — weights computed from full training set, applied to all trees
rf = RandomForestClassifier(class_weight='balanced')

# 'balanced_subsample' — weights recomputed PER bootstrap sample (better for RF)
rf = RandomForestClassifier(class_weight='balanced_subsample')
```

**`balanced_subsample` is preferred for RF** — since each tree sees a different bootstrap sample (with different class proportions), recomputing weights per sample is more accurate.

---

### Fix 2: Balanced Random Forest

Undersample majority class **within each bootstrap sample**:

```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=100,
    sampling_strategy='auto',   # undersample majority to match minority
    random_state=42
)
brf.fit(X_train, y_train)
```

Each tree trains on a balanced sample → no tree is biased toward majority.

---

### Fix 3: Easy Ensemble / RUSBoost

```python
from imblearn.ensemble import EasyEnsembleClassifier
ee = EasyEnsembleClassifier(n_estimators=10, random_state=42)
ee.fit(X_train, y_train)
```

Trains multiple balanced classifiers — each on a different undersampled majority subset.

---

### Fix 4: SMOTE + RF (Pipeline)

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100))
])
pipe.fit(X_train, y_train)
```

---

### Which fix to use?

```
Start:  class_weight='balanced_subsample'   → fast, no data change
Better: BalancedRandomForestClassifier      → principled undersampling
Best:   SMOTE + RF (if data is very scarce for minority)
Always: evaluate with F1 / AUC-PR, not accuracy
```

---

# 12. FAANG Conceptual Q&A

---

**Q1: How does Random Forest reduce variance compared to a single Decision Tree?**  
**A:** Two mechanisms. First, bagging — each tree trains on a different bootstrap sample, so individual overfitting patterns don't correlate. Second, feature subsampling — each split only sees m features, forcing trees to use different features and decorrelating them further. Averaging B decorrelated high-variance trees reduces variance by a factor approaching 1/B (less reduction if trees are correlated). Bias stays low because each tree is still grown deep.

---

**Q2: What is the OOB error and why is it useful?**  
**A:** Each tree in an RF is trained on ~63.2% of data (bootstrap sample). The remaining ~36.8% (out-of-bag samples) are used to evaluate that tree. Aggregating OOB predictions across all trees gives a free estimate of test error — no separate validation set needed. It's slightly pessimistic (each sample is evaluated by fewer trees than the full ensemble) but a good approximation.

---

**Q3: Why does increasing `n_estimators` always help (up to a point)?**  
**A:** More trees → averaging over more independent estimates → variance keeps dropping. Unlike boosting, adding trees to a Random Forest never increases bias. The benefit plateaus (diminishing returns) because variance floor is ρσ² — determined by tree correlation ρ, not tree count. Beyond ~300-500 trees you're trading compute for negligible gain.

---

**Q4: Random Forest vs Decision Tree — when would you use a single tree?**  
**A:** A single tree when interpretability is critical — you need to explain the exact decision path to stakeholders ("customer was denied because age < 25 AND income < 30k"). RF sacrifices interpretability for accuracy. In most production ML settings, use RF (or GBM). Use a single tree for rule extraction, decision support systems, or regulatory environments requiring explainability.

---

**Q5: Does Random Forest overfit? Can you add too many trees?**  
**A:** Unlike boosting, adding more trees to RF never overfits (they don't memorise — they average). However, with very few trees and high `max_depth`, individual trees overfit, and with low `n_estimators` the averaging doesn't sufficiently smooth this out. The practical limit is compute — at 1000+ trees, each additional tree gives negligible improvement.

---

**Q6: What is the difference between Bagging and Random Forest?**  
**A:** Bagging trains B trees on bootstrap samples using ALL features at each split. RF additionally subsamples features at each split (`max_features < d`). This extra randomness decorrelates the trees, which is what makes RF more powerful than pure bagging.

---

**Q7: How does Random Forest handle correlated features?**  
**A:** Poorly, for feature importance. If two features are highly correlated, RF splits arbitrarily between them — each gets roughly half the credit, understating the true importance of either. The model accuracy is fine (it uses whichever is available). For importance, use SHAP or group correlated features. For prediction, RF handles correlations fine.

---

**Q8: Why is `max_features='sqrt'` the default for classification?**  
**A:** Empirical finding across many datasets — `sqrt(d)` provides a good balance between (a) giving each tree enough features to find a meaningful split (low bias) and (b) excluding enough features to decorrelate trees (low variance). For regression, `max_features=1.0` (all features) is default — regression targets are often smoother and require more features per split.

---

**Q9: How would you speed up Random Forest training on a large dataset?**  
**A:** Several options: `n_jobs=-1` (parallelise across all cores — trees are independent), reduce `n_estimators` (fewer trees), reduce `max_depth` (shallower trees), set `max_samples` < 1.0 (smaller bootstrap samples), or switch to `ExtraTreesClassifier` (random thresholds are faster than finding optimal ones).

---

**Q10: RF vs GBM — when would you choose which?**

| | Random Forest | Gradient Boosting |
|---|---|---|
| Training speed | ⚡ Faster (parallel) | Slower (sequential) |
| Tuning difficulty | Easy | Harder |
| Overfitting risk | Low | Higher |
| Accuracy on tabular data | Good | Usually better |
| Handles outliers | Moderate | Sensitive |
| Interpretability | Similar (both use SHAP) | Similar |
| Missing values (XGBoost) | Need imputation | Native support |

> RF first for baselines and when speed/robustness matter. GBM for max accuracy on tabular data.

---

# Summary Cheatsheet

```
RANDOM FOREST
─────────────
Core idea:    Bagging + Feature Subsampling → B decorrelated trees → average

Two randomness sources:
  1. Bootstrap sampling   (~63.2% unique data per tree)
  2. max_features         (only m features per split)

Key params:
  n_estimators  → more = better (plateau ~200-500)
  max_features  → 'sqrt' (classification), 1.0 (regression)
  max_depth     → None by default (grow full trees)
  oob_score     → True (free cross-validation!)

Bias-Variance:
  Low Bias    (deep trees, same as single tree)
  Low Variance (averaging B decorrelated trees)
  Variance = ρσ² + (1-ρ)σ²/B → floor at ρσ²

vs Single Tree:
  Better accuracy, worse interpretability
  No pruning needed (averaging handles variance)

vs GBM:
  Faster, more robust, easier to tune
  Usually slightly lower accuracy on tabular data

Feature Importance:
  MDI   → fast, biased to high cardinality
  Permutation → unbiased, use test set
  SHAP  → best, per-sample explanations

Missing Values:
  sklearn → impute first
  MissForest → iterative RF imputation
  XGBoost → native (learn default direction)

Class Imbalance:
  class_weight='balanced_subsample'  → first fix
  BalancedRandomForestClassifier     → principled
  SMOTE + RF                         → when minority is very rare

OOB Error:
  Free test error estimate
  oob_score=True → no separate val set needed
```
