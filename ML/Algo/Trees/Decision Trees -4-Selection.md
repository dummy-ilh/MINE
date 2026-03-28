# How is a Feature Selected for a Split?
## Decision Trees — Split Selection Deep Dive

---

## The Core Question

At every node, the tree must answer:
```
"Which feature, split at which threshold,
 best separates the classes / reduces error?"
```

The answer involves two nested searches:
```
FOR every feature f:
    FOR every possible threshold t on feature f:
        Compute split quality score
Pick the (f, t) pair with the BEST score
```

---

## Step-by-Step: The Full Algorithm

### Step 1 — Collect candidate splits

For each feature:

**Numerical feature (e.g., Age = [22, 25, 30, 35, 40])**
```
Sort unique values: [22, 25, 30, 35, 40]
Candidate thresholds = midpoints between consecutive values:
  → [23.5, 27.5, 32.5, 37.5]
```

**Categorical feature (e.g., Colour = {Red, Blue, Green})**
```
Binary split: try all subsets of categories
  → {Red} vs {Blue, Green}
  → {Blue} vs {Red, Green}
  → {Green} vs {Red, Blue}
  (2^k - 2) / 2 possible binary splits for k categories
```

---

### Step 2 — Score every (feature, threshold) pair

For each candidate split, compute:
$$\text{Score} = \text{Impurity(parent)} - \frac{n_L}{n}\text{Impurity}(L) - \frac{n_R}{n}\text{Impurity}(R)$$

Using Gini, Entropy, or Variance depending on criterion.

---

### Step 3 — Pick the best split

```python
best_score = -infinity
best_feature = None
best_threshold = None

for feature in all_features:
    for threshold in candidate_thresholds(feature):
        score = compute_gain(feature, threshold)
        if score > best_score:
            best_score = score
            best_feature = feature
            best_threshold = threshold
```

---

### Step 4 — Split the node

```
Data at node → split by best_feature <= best_threshold
  Left child  → samples where feature <= threshold
  Right child → samples where feature >  threshold
```

---

## ✏️ Full Worked Example

### Dataset (10 samples, binary classification)

| # | Age | Income | Purchased? |
|---|---|---|---|
| 1 | 22 | Low | No |
| 2 | 25 | High | Yes |
| 3 | 27 | Low | No |
| 4 | 30 | High | Yes |
| 5 | 32 | Low | No |
| 6 | 35 | High | Yes |
| 7 | 38 | Low | No |
| 8 | 40 | High | Yes |
| 9 | 42 | Low | No |
| 10 | 45 | High | Yes |

**Parent node:** 5 Yes, 5 No
$$\text{Gini}_{parent} = 1 - (0.5^2 + 0.5^2) = 0.5$$

---

### Evaluating Feature 1: Age

Sorted unique ages: [22, 25, 27, 30, 32, 35, 38, 40, 42, 45]
Thresholds to try: [23.5, 26, 28.5, 31, 33.5, 36.5, 39, 41, 43.5]

**Try threshold: Age ≤ 31**
- Left (Age ≤ 31): samples 1,2,3,4 → 2 Yes, 2 No
  $$\text{Gini}(L) = 1-(0.5^2+0.5^2) = 0.5$$
- Right (Age > 31): samples 5–10 → 3 Yes, 3 No
  $$\text{Gini}(R) = 1-(0.5^2+0.5^2) = 0.5$$
- Weighted Gini = 0.5 → **Gain = 0.0** ❌ useless split

**Try threshold: Age ≤ 22**
- Left (Age ≤ 22): sample 1 → 0 Yes, 1 No
  $$\text{Gini}(L) = 1-(0^2+1^2) = 0.0$$
- Right (Age > 22): samples 2–10 → 5 Yes, 4 No
  $$\text{Gini}(R) = 1-\left(\left(\frac{5}{9}\right)^2+\left(\frac{4}{9}\right)^2\right) = 1-(0.309+0.198) = 0.494$$
- Weighted Gini = $\frac{1}{10}(0)+\frac{9}{10}(0.494) = 0.444$
- **Gain = 0.5 − 0.444 = 0.056** (small)

Age alone doesn't split well here — No/Yes alternate by Income, not Age.

---

### Evaluating Feature 2: Income (Categorical)

Income ∈ {Low, High}
Only one meaningful binary split: **Income = High vs Income = Low**

- Left (Income = Low): samples 1,3,5,7,9 → 0 Yes, 5 No
  $$\text{Gini}(L) = 1-(0^2+1^2) = \mathbf{0.0}$$ ← PURE!

- Right (Income = High): samples 2,4,6,8,10 → 5 Yes, 0 No
  $$\text{Gini}(R) = 1-(1^2+0^2) = \mathbf{0.0}$$ ← PURE!

$$\text{Weighted Gini} = \frac{5}{10}(0) + \frac{5}{10}(0) = \mathbf{0.0}$$

$$\text{Gain} = 0.5 - 0.0 = \mathbf{0.5}$$ ← Maximum possible!

### ✅ Winner: Income is selected — Gini drops from 0.5 to 0.0

---

## What About `max_features`? (Random Feature Subsampling)

In a standard Decision Tree, **all features are evaluated** at every node.

In **Random Forests**, only a random subset of features is evaluated at each split — this is the `max_features` parameter.

| `max_features` value | Meaning | Typical use |
|---|---|---|
| `None` / `'all'` | Use all features | Plain Decision Tree |
| `'sqrt'` | $\sqrt{d}$ features | Random Forest (classification) default |
| `'log2'` | $\log_2(d)$ features | Random Forest (alternative) |
| `0.5` | 50% of features | Custom subsampling |
| Integer `k` | Exactly k features | Fixed budget |

### Why random subsampling in forests?
```
If one feature dominates (very high importance),
ALL trees in a forest would split on it near the root.
→ Trees become highly correlated
→ Averaging correlated trees doesn't reduce variance much

Randomly excluding features:
→ Forces trees to find other good splits
→ Decorrelates trees
→ Ensemble variance drops significantly
```

---

## Tie-Breaking

What if two splits have the exact same score?

```python
# sklearn behaviour
# → Takes the first one encountered (scan order)
# → Not explicitly documented — treat ties as arbitrary
```

In practice:
- Ties are rare on continuous features
- More common on small datasets or with many binary features
- Can add `random_state` for reproducibility

---

## Numerical vs Categorical Features

| | Numerical | Categorical |
|---|---|---|
| Candidate thresholds | Midpoints of sorted unique values | All binary subset partitions |
| Number of splits to try | O(n unique values) | O(2^k) for k categories |
| High cardinality risk | Low (still O(n)) | High — exponential blowup |
| sklearn support | ✅ Native | ⚠️ Needs encoding first |

### High-Cardinality Categoricals

For a categorical with 20 unique values:
- Binary partitions = $2^{20-1} - 1 = 524,287$ splits to try 🚨

**Solutions:**
1. **Ordinal encode** — treat as ordered numeric (fast, but imposes order)
2. **Target encode** — replace category with mean target value, then treat as numeric
3. **One-hot encode** — creates binary features (safe but wide)
4. **Use CatBoost / LightGBM** — handle high-cardinality categoricals natively and efficiently

---

## Complexity of Split Selection

At a single node with $n$ samples and $d$ features:

$$\text{Time} = O(d \cdot n \log n)$$

- For each of $d$ features: sort $n$ values → $O(n \log n)$
- Evaluate all $n-1$ thresholds → $O(n)$

For the full tree (up to $n$ nodes):

$$\text{Total Training Time} = O(d \cdot n^2 \log n)$$

*(in the worst case — balanced tree has $O(\log n)$ levels, giving $O(d \cdot n \log^2 n)$)*

---

## What Makes a Good Split? — Intuition

```
Bad Split                          Good Split
──────────                         ──────────
       [Node]                             [Node]
      (5Y, 5N)                           (5Y, 5N)
      /       \                          /       \
  (3Y, 2N)  (2Y, 3N)               (5Y, 0N)  (0Y, 5N)
  impure      impure                PURE ✅    PURE ✅
  Gini=0.48  Gini=0.48              Gini=0     Gini=0
  
  Gain = 0.5 - 0.48 = 0.02          Gain = 0.5 - 0 = 0.5
  ❌ Almost no improvement           ✅ Maximum gain
```

A good split **separates classes cleanly** into children — the two children should be as pure (unmixed) as possible.

---

## FAANG Q&A

---

**Q1: How does a decision tree decide which feature to split on?**
**A:** It tries every feature and every possible threshold. For each (feature, threshold) pair it computes the impurity reduction (Gini gain, Information Gain, or variance reduction). The pair with the highest gain is selected. This is a greedy, exhaustive local search — not globally optimal.

---

**Q2: Is the greedy split selection globally optimal?**
**A:** No. CART's greedy approach finds the locally best split at each node, but a suboptimal split now might enable a much better split below. Optimal tree induction is NP-hard. The greedy approach is a practical approximation, and post-pruning / ensembles compensate for its limitations.

---

**Q3: Why does Random Forest use `max_features = sqrt(d)`?**
**A:** To decorrelate trees in the ensemble. If all features are available, all trees tend to split on the same dominant feature near the root — producing correlated trees. Averaging correlated trees gives little variance reduction. Random subsampling forces diversity, making the average more powerful.

---

**Q4: How are numerical thresholds chosen efficiently?**
**A:** Sort the feature values, then the only meaningful thresholds are the midpoints between consecutive unique values. For $n$ samples there are at most $n-1$ candidate thresholds — not infinite, because only boundaries between adjacent sorted values can change the split outcome.

---

**Q5: What happens if a feature is never selected for any split?**
**A:** Its `feature_importances_` value = 0. It contributed nothing to impurity reduction across all nodes. This can indicate the feature is irrelevant, or that it's redundant (another feature carries the same information and was chosen first due to the greedy search).

---

## Summary Cheatsheet

```
HOW A SPLIT IS CHOSEN
─────────────────────
For each feature:
  For each threshold (midpoints of sorted values):
    Compute: Gini/Entropy reduction
Select: feature + threshold with HIGHEST gain

KEY FACTORS
───────────
• Greedy — locally optimal, not globally
• Numerical: O(n) thresholds per feature
• Categorical: O(2^k) partitions — expensive for high cardinality
• max_features < d → random subsampling (used in Random Forests)
  → decorrelates trees → better ensemble

COMPLEXITY
──────────
Per node : O(d × n log n)
Full tree: O(d × n² log n) worst case

TIE BREAKS
──────────
→ First feature encountered (arbitrary) — set random_state for reproducibility
```


# Decision Trees — Advanced Topics
## Missing Values · Oblique Splits · Regression vs Classification · Class Imbalance

---

# TOPIC 1: Handling Missing Values

---

## The Problem

Real data always has gaps. A decision tree needs to route every sample left or right at every node — but what if the splitting feature is missing for that sample?

```
Node splits on: Age <= 30

Sample X:  Age = ???   Income = High   Purchased = ?

Which way does X go?
```

---

## At TRAINING Time

### Strategy 1: Imputation Before Training (sklearn default approach)

sklearn's `DecisionTreeClassifier` **cannot handle NaN natively** — it throws an error.  
You must impute first.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),   # or median, most_frequent
    ('tree', DecisionTreeClassifier())
])
pipe.fit(X_train, y_train)
```

**Imputation strategies:**

| Strategy | Fills with | Best for |
|---|---|---|
| `mean` | Column average | Numerical, symmetric distribution |
| `median` | Column median | Numerical with outliers |
| `most_frequent` | Most common value | Categorical |
| `constant` | A fixed value | When missing = informative (e.g. fill 0) |

**⚠️ Risk:** Imputing with training statistics then applying to test is fine. Imputing with test data statistics = **data leakage**. Always fit imputer on train only.

---

### Strategy 2: Surrogate Splits (CART — the principled approach)

CART (the algorithm behind sklearn) supports surrogate splits conceptually — though sklearn doesn't expose them directly.

**Idea:**
```
Primary split:  Age <= 30
                (best split overall)

Surrogate split: Income = Low
                 (2nd best split that mimics Age <= 30 most closely)

If Age is missing for a sample → use Income = Low as backup
```

**How surrogates are chosen:**
For each candidate surrogate feature, measure how well it **agrees** with the primary split (same left/right assignment) across training samples where both features are known.

```
Surrogate quality = fraction of samples assigned to same child
                    as the primary split would assign them
```

The surrogate with highest agreement becomes the first backup, next highest becomes second backup, etc.

**Pros:** Principled, uses data structure — missing value handling is learned  
**Cons:** Sklearn doesn't expose this; need custom implementation or XGBoost

---

### Strategy 3: Missing as a Separate Branch

Treat `NaN` as its own category — route all missing-value samples down a dedicated third branch.

```
       [Age <= 30]
      /     |     \
   Yes     No    Missing
   / \           (dedicated branch)
```

Not supported in sklearn. Supported natively in **LightGBM** and indirectly in some other libraries.

---

### Strategy 4: Missing Indicator Feature

Add a binary column: `Age_is_missing ∈ {0, 1}`  
Then impute Age normally.

```python
from sklearn.impute import MissingIndicator
import numpy as np

indicator = MissingIndicator()
missing_flags = indicator.fit_transform(X_train)
X_train_augmented = np.hstack([X_train_imputed, missing_flags])
```

**Why this works:** The tree can now split on `Age_is_missing = 1` and learn that missingness itself is predictive (e.g., patients who skipped a test may be healthier or sicker systematically).

---

## At TEST Time

The imputer/surrogate fitted on training data is applied at test time:

```
Training:
  fit imputer on X_train → learn mean_age = 34.2

Test:
  sample has Age = NaN
  → replace with 34.2 (training mean)
  → route through tree as normal
```

**Critical rule:** Always use **training statistics** to fill test missing values. Never refit on test data.

```python
# CORRECT
imputer.fit(X_train)
X_train_clean = imputer.transform(X_train)
X_test_clean  = imputer.transform(X_test)   # uses training stats

# WRONG — leaks test info into imputation
imputer.fit(X_test)
X_test_clean = imputer.transform(X_test)
```

---

## XGBoost / LightGBM — Native Missing Value Handling

These libraries **learn the best direction** for missing values during training.

**XGBoost algorithm:**
```
At each split node, during training:
  1. Try routing all missing-value samples LEFT → compute gain
  2. Try routing all missing-value samples RIGHT → compute gain
  3. Whichever gives higher gain → that becomes the "default direction"

At test time:
  Missing sample → goes in the learned default direction automatically
```

This is elegant — missingness pattern in training data is used to learn the optimal routing.

---

## Summary Table

| Method | sklearn? | Learns from data? | Notes |
|---|---|---|---|
| Mean/Median imputation | ✅ | ❌ | Simple, works well |
| Surrogate splits | ❌ (not exposed) | ✅ | CART theory, principled |
| Missing branch | ❌ | ✅ | LightGBM native |
| Missing indicator | ✅ | ✅ | Best when missingness is informative |
| XGBoost default direction | ✅ (XGB) | ✅ | Best overall, automatic |

---

## FAANG Q&A — Missing Values

**Q: How does sklearn's Decision Tree handle NaN?**  
**A:** It doesn't — it throws a `ValueError`. You must impute before fitting. Use `SimpleImputer` or `IterativeImputer` in a Pipeline. Always fit the imputer on training data only.

**Q: What are surrogate splits?**  
**A:** A CART concept where backup splits are learned for each node — features that best mimic the primary split's left/right assignments. When the primary feature is missing, the best surrogate takes over.

**Q: Why is XGBoost better at missing values than sklearn trees?**  
**A:** XGBoost learns the optimal routing direction for missing values during training — it figures out which direction leads to lower loss and hardcodes that as the default. sklearn trees have no such mechanism.

---
---

# TOPIC 2: Multivariate / Oblique Splits

---

## The Limitation of Standard (Axis-Aligned) Splits

Standard CART splits on **one feature at a time** → decision boundaries are always axis-aligned:

```
Feature 2
   │         ┌──────────┐
   │    ○ ○  │  ● ● ●  │
   │  ○ ○    │   ● ●   │
   │    ○    │  ● ●    │
   └──────────┴─────────── Feature 1
              ↑
         Split: Feature1 <= 3.5
         (vertical line — axis aligned)
```

This works fine for many problems, but **fails when the true decision boundary is diagonal:**

```
Feature 2
   │  ● ● ● ○ ○ ○
   │ ● ● ● ○ ○ ○
   │● ● ● ○ ○ ○
   └──────────────── Feature 1
   
True boundary: diagonal line
Axis-aligned tree needs MANY splits to approximate it:
  if F1 <=1: left
  elif F1 <=2 and F2 >= 3: ...  (staircase approximation)
```

---

## Oblique Splits — The Solution

An **oblique split** uses a **linear combination of features**:

$$w_1 x_1 + w_2 x_2 + \ldots + w_d x_d \leq \theta$$

```
Standard split:    Age <= 30
                   (one feature, axis-aligned)

Oblique split:     0.6 × Age + 0.4 × Income <= 25
                   (linear combination → diagonal boundary)
```

This produces a **diagonal hyperplane** — a single oblique split can replace dozens of axis-aligned splits.

---

## Visual Comparison

```
Axis-Aligned Tree          Oblique Tree
─────────────────          ────────────
●●●│○○○                    ●●●╲○○○
●●●│○○○                    ●●●●╲○○
●●●│○○○                    ●●●●●╲○

Needs staircase approx     Single diagonal line
→ deep tree                → shallow tree
→ high variance            → lower variance
```

---

## Algorithms That Use Oblique Splits

| Algorithm | Approach |
|---|---|
| **OC1** (Oblique Classifier 1) | Randomised search for best weight vector |
| **CART with linear combos** | Exhaustive search (expensive) |
| **Oblique Random Forest** | Oblique splits + ensemble |
| **RidgeTree / LassoTree** | Fit Ridge/Lasso at each node to find weights |
| **sklearn ExtraTreesClassifier** | Random thresholds (not fully oblique but faster) |

---

## How OC1 Finds Oblique Splits

```
1. Start with a random weight vector w
2. For each weight w_j:
   a. Hold all other weights fixed
   b. Find the optimal w_j that minimises impurity (1D search)
3. Repeat until convergence
4. Perturb weights randomly → restart → keep best
```

This is a **coordinate descent + random restart** approach.

---

## Trade-offs

| | Axis-Aligned | Oblique |
|---|---|---|
| Decision boundary | Staircase | Diagonal / curved |
| Interpretability | ✅ High ("Age <= 30") | ❌ Low ("0.6×Age + 0.4×Income <= 25") |
| Training speed | ✅ Fast O(dn log n) | ❌ Slow |
| Accuracy on diagonal boundaries | ❌ Needs deep tree | ✅ Compact |
| sklearn support | ✅ | ❌ (not built-in) |
| Overfitting risk | Moderate | Higher (more parameters per split) |

---

## When Would You Use Oblique Splits?

- Data has **strong correlations** between features
- You know the decision boundary is **not axis-parallel**
- Interpretability is **not** a requirement
- You have **sufficient data** to fit weights robustly

In practice: most FAANG practitioners just use **Random Forest or GBM** instead — they approximate oblique boundaries implicitly through ensembling many axis-aligned trees.

---

## FAANG Q&A — Oblique Splits

**Q: What is the limitation of standard decision tree splits?**  
**A:** They are axis-aligned — they split on one feature at a time, producing staircase-shaped boundaries. When the true boundary is diagonal or curved, a standard tree needs many deep splits to approximate it, leading to high variance. Oblique splits use linear combinations of features and can represent diagonal boundaries with far fewer splits.

**Q: Why don't we use oblique splits by default?**  
**A:** Three reasons: (1) much slower to compute — optimising a weight vector is harder than a 1D threshold search; (2) much less interpretable — you lose the "if age > 30 then..." readability; (3) higher overfitting risk. Ensembles (Random Forest, XGBoost) achieve similar accuracy gains without these drawbacks.

---
---

# TOPIC 3: Regression Trees vs Classification Trees

---

## The Core Difference

| | Classification Tree | Regression Tree |
|---|---|---|
| Target variable | Discrete class labels | Continuous values |
| Leaf prediction | **Majority class** | **Mean of samples** |
| Split criterion | Gini / Entropy | Variance Reduction / MSE |
| Evaluation metric | Accuracy, F1, AUC | MSE, MAE, R² |
| Output | Class label (+ probabilities) | Continuous value |

---

## Leaf Predictions

### Classification Tree
```
Leaf has samples: [Yes, Yes, No, Yes, Yes]
Prediction = majority class = YES
Probability = 4/5 = 0.8 for Yes
```

### Regression Tree
```
Leaf has samples: [2.1, 3.4, 2.8, 3.1]
Prediction = mean = (2.1 + 3.4 + 2.8 + 3.1) / 4 = 2.85
```

Some variants use **median** instead of mean (more robust to outliers):
```python
DecisionTreeRegressor(criterion='absolute_error')  # uses median at leaves
```

---

## Splitting Criteria for Regression

### Variance Reduction (MSE criterion)

$$\text{MSE}(S) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

$$\text{Score} = \text{MSE}(S) - \frac{n_L}{n}\text{MSE}(L) - \frac{n_R}{n}\text{MSE}(R)$$

Pick the split that **maximises variance reduction**.

### MAE criterion (Friedman MSE)

$$\text{MAE}(S) = \frac{1}{n}\sum_{i=1}^{n}|y_i - \tilde{y}|$$

where $\tilde{y}$ is the **median** (optimal for MAE).

```python
from sklearn.tree import DecisionTreeRegressor

# MSE criterion (default) — mean at leaves
reg_mse = DecisionTreeRegressor(criterion='squared_error')

# MAE criterion — median at leaves
reg_mae = DecisionTreeRegressor(criterion='absolute_error')
```

| Criterion | Leaf value | Robust to outliers? |
|---|---|---|
| `squared_error` (MSE) | Mean | ❌ No |
| `absolute_error` (MAE) | Median | ✅ Yes |
| `friedman_mse` | Mean (smarter formula) | ❌ No |
| `poisson` | Mean (for count data) | Moderate |

---

## ✏️ Worked Example — Regression Tree

### Dataset: Predict house price from size

| # | Size (m²) | Price (£k) |
|---|---|---|
| 1 | 50 | 150 |
| 2 | 60 | 180 |
| 3 | 80 | 250 |
| 4 | 90 | 270 |
| 5 | 120 | 400 |
| 6 | 150 | 500 |

**Parent MSE:**
$$\bar{y} = \frac{150+180+250+270+400+500}{6} = 291.7$$
$$\text{MSE} = \frac{(150-291.7)^2 + \ldots + (500-291.7)^2}{6} = 15{,}972$$

**Try split: Size <= 85**
- Left (50, 60, 80): prices [150, 180, 250] → mean = 193.3
  $$\text{MSE}(L) = \frac{(150-193.3)^2+(180-193.3)^2+(250-193.3)^2}{3} = 1{,}755$$

- Right (90, 120, 150): prices [270, 400, 500] → mean = 390
  $$\text{MSE}(R) = \frac{(270-390)^2+(400-390)^2+(500-390)^2}{3} = 10{,}067$$

$$\text{Weighted MSE} = \frac{3}{6}(1{,}755) + \frac{3}{6}(10{,}067) = 5{,}911$$

$$\text{Variance Reduction} = 15{,}972 - 5{,}911 = \mathbf{10{,}061}$$

Compare all thresholds, pick the one with maximum reduction.

**Final tree:**
```
         [Size <= 85?]
        /              \
  Predict £193k       Predict £390k
  (mean of L group)   (mean of R group)
```

---

## Key Differences in Behaviour

### 1. Regression Trees Can't Extrapolate

```
Training prices: £100k – £500k
Test sample: Size = 300m² (never seen)

Tree prediction: £500k (mean of largest leaf)
Reality: probably £900k+

Classification tree has the same issue, but predictions
are bounded (can't predict a new class it hasn't seen).
```

### 2. Step-Function Predictions

Regression trees produce **piecewise constant** predictions — a staircase function, not a smooth curve.

```
Price
 │         ┌─────────┐
 │         │  £390k  │
 │         │         └──────────
 │ £193k ──┘
 └──────────────────────────── Size
           85
```

More leaves → finer staircase → closer approximation to smooth function.

### 3. Sensitivity to Outliers

One extreme price (e.g., £5M mansion) shifts the mean of its leaf dramatically.  
→ Use `criterion='absolute_error'` (median-based) to mitigate.

---

## sklearn API Differences

```python
# Classification
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini')
clf.predict(X)          # → class labels
clf.predict_proba(X)    # → class probabilities

# Regression
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(criterion='squared_error')
reg.predict(X)          # → continuous values
# reg.predict_proba(X)  # ← doesn't exist for regression!
```

---

## FAANG Q&A — Regression vs Classification Trees

**Q: How does a regression tree make predictions at a leaf?**  
**A:** It averages the target values of all training samples that fall into that leaf. For MSE criterion, this mean minimises the squared error within the leaf. For MAE criterion, the median is used instead.

**Q: Why can't regression trees extrapolate?**  
**A:** The tree partitions feature space into rectangles and assigns each a constant value (the mean of training samples in that region). For inputs outside the training range, the tree falls back to the last leaf's constant — it has no mechanism to predict beyond values it has seen.

**Q: When would you use MAE over MSE as the regression criterion?**  
**A:** When the target variable has significant outliers. MSE penalises large errors quadratically, so one outlier can dominate and distort leaf means. MAE uses the median, which is robust to outliers.

---
---

# TOPIC 4: Class Imbalance

---

## The Problem

Most real datasets are imbalanced. In fraud detection, disease diagnosis, churn prediction:

```
Class 0 (Not Fraud): 9,900 samples  ← 99%
Class 1 (Fraud):       100 samples  ← 1%
```

A tree that **always predicts Class 0** gets **99% accuracy** — but is completely useless.

---

## How Imbalance Corrupts Splits

Recall Gini for a node with 9,900 Class 0 and 100 Class 1:

$$p_0 = 0.99, \quad p_1 = 0.01$$
$$\text{Gini} = 1 - (0.99^2 + 0.01^2) = 1 - (0.9801 + 0.0001) = \mathbf{0.0198}$$

This node already looks **nearly pure** to the Gini criterion — even though it's completely useless for detecting fraud.

The tree has little incentive to split on the minority class → it learns to mostly predict the majority class → **minority class recall ≈ 0**.

---

## Fix 1: `class_weight` — Reweighting the Impurity

Modify the impurity calculation so minority class errors cost **more**.

$$\text{Weighted Gini} = 1 - \sum_{i=1}^{C} (w_i \cdot p_i)^2$$

```python
# sklearn automatically computes weights inversely proportional to frequency
clf = DecisionTreeClassifier(class_weight='balanced')

# Or set manually
clf = DecisionTreeClassifier(class_weight={0: 1, 1: 99})
```

**How `'balanced'` computes weights:**
$$w_i = \frac{n_{\text{total}}}{C \times n_i}$$

For our fraud example:
$$w_0 = \frac{10000}{2 \times 9900} = 0.505, \quad w_1 = \frac{10000}{2 \times 100} = 50$$

Now a misclassified fraud sample costs **50×** more than a non-fraud — the tree is forced to pay attention to the minority class.

---

## ✏️ Worked Example — Effect of class_weight

**Node:** 990 Class 0 (majority), 10 Class 1 (minority)

**Without class_weight:**
$$\text{Gini} = 1 - (0.99^2 + 0.01^2) = 0.0198$$
Node looks nearly pure → tree may not split.

**With class_weight='balanced' (w₀ = 0.5, w₁ = 50):**

Effective proportions after weighting:
$$\tilde{p}_0 = \frac{0.5 \times 990}{0.5 \times 990 + 50 \times 10} = \frac{495}{995} = 0.497$$
$$\tilde{p}_1 = \frac{50 \times 10}{995} = 0.503$$

$$\text{Weighted Gini} = 1 - (0.497^2 + 0.503^2) = 1 - (0.247 + 0.253) = \mathbf{0.50}$$

Now the node looks **maximally impure** (0.50) → the tree is strongly motivated to split on it.

---

## Fix 2: Resampling

Change the data itself instead of the criterion.

### Oversampling — add more minority class samples

**Random Oversampling:**
```python
# Duplicate minority class samples randomly
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
```

**SMOTE (Synthetic Minority Over-sampling Technique):**
```
Rather than duplicating, SMOTE creates SYNTHETIC new minority samples:
1. Pick a minority sample x
2. Find its k nearest minority neighbours
3. Create new point along the line between x and a random neighbour
   new_point = x + λ × (neighbour - x),  λ ∈ [0,1]
```

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

### Undersampling — remove majority class samples

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)
```

**Risk:** Throws away real data — can hurt if majority class has important variation.

---

## Fix 3: Change the Decision Threshold

A classifier outputs probabilities. By default, predict Class 1 if `p >= 0.5`.

For imbalanced classes, lower the threshold:

```python
probs = clf.predict_proba(X_test)[:, 1]  # P(fraud)

# Default threshold
preds_default  = (probs >= 0.5).astype(int)

# Lowered threshold → catch more fraud (higher recall, lower precision)
preds_adjusted = (probs >= 0.2).astype(int)
```

**Use the Precision-Recall curve to pick the optimal threshold** — maximise F1 or set a recall floor.

```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
# Plot and pick threshold
```

---

## Fix 4: Use the Right Evaluation Metric

Accuracy is meaningless on imbalanced data. Use:

| Metric | When to use |
|---|---|
| **F1 Score** | Balance precision and recall |
| **Precision** | Cost of false positives is high |
| **Recall** | Cost of false negatives is high (fraud, disease) |
| **AUC-ROC** | Overall ranking ability |
| **AUC-PR** | Better than ROC when positives are rare |
| **Cohen's Kappa** | Adjusts for chance agreement |

```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, probs))
```

---

## Comparison of Fixes

| Method | What it changes | Pros | Cons |
|---|---|---|---|
| `class_weight` | Impurity calculation | Simple, no data change | Doesn't fix data distribution |
| SMOTE | Training data | Creates diverse minority samples | Synthetic data may be unrealistic |
| Random Oversample | Training data | Simple | Just duplicates — no new info |
| Undersample | Training data | Fast | Wastes majority data |
| Threshold tuning | Decision boundary | No retraining needed | Requires calibrated probabilities |

---

## Best Practice Pipeline

```
1. Use class_weight='balanced'          → quick baseline fix
2. Evaluate with F1/AUC-PR not accuracy → correct metric
3. If still poor recall → add SMOTE     → create synthetic minority samples
4. Tune decision threshold              → trade off precision vs recall
5. Consider ensemble (BalancedRF)       → combines undersampling + bagging
```

```python
from imblearn.ensemble import BalancedRandomForestClassifier
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
```

---

## FAANG Q&A — Class Imbalance

**Q: How does class imbalance affect a decision tree?**  
**A:** The impurity measures (Gini, Entropy) are dominated by the majority class. A near-pure majority-class node has very low Gini already — the tree sees little benefit in splitting further. The minority class effectively gets ignored, leading to near-zero recall for it.

**Q: What does `class_weight='balanced'` actually do mathematically?**  
**A:** It scales each class's contribution to the impurity measure by $\frac{n_{total}}{C \times n_i}$ — inversely proportional to class frequency. Rare classes get high weights, making their misclassification cost more. The tree is then incentivised to find splits that separate the minority class.

**Q: SMOTE vs class_weight — when would you use each?**  
**A:** `class_weight` changes nothing about the data — just the loss function. It's fast and safe. SMOTE creates synthetic samples and changes the training distribution — it can help more when the minority class is extremely rare and needs more examples for the tree to learn its patterns. In practice, try `class_weight` first; add SMOTE if recall is still insufficient.

**Q: Why is AUC-PR better than AUC-ROC for imbalanced data?**  
**A:** AUC-ROC can look deceptively good when negatives dominate — a high true negative rate inflates the score even if the model rarely catches positives. AUC-PR focuses entirely on the minority class performance (precision and recall) and is much harder to game on imbalanced data.

---

## Full Summary Cheatsheet

```
MISSING VALUES
──────────────
sklearn: impute before training (fit on train only!)
Best:    XGBoost — learns optimal direction for missing values
Add:     Missing indicator feature when missingness is informative

OBLIQUE SPLITS
──────────────
Standard: one feature at a time → axis-aligned (staircase)
Oblique:  linear combo of features → diagonal boundary
Trade-off: more expressive BUT slower + less interpretable
Practice:  just use Random Forest/GBM instead

REGRESSION vs CLASSIFICATION
──────────────────────────────
Classification → majority class at leaf  | Gini / Entropy
Regression     → mean at leaf            | Variance / MSE
Both: cannot extrapolate beyond training range
Regression: step-function output; use MAE criterion for outlier robustness

CLASS IMBALANCE
───────────────
Problem:   majority class dominates impurity → minority ignored
Fix 1:     class_weight='balanced'  → reweight impurity calculation
Fix 2:     SMOTE                    → synthetic minority samples
Fix 3:     threshold tuning         → lower p threshold for minority
Fix 4:     AUC-PR / F1              → use the right metric!
```
