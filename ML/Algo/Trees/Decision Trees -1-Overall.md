# Decision Trees — Complete FAANG Interview Notes

---

## 1. The Algorithm

A Decision Tree is a **supervised learning** model that splits data into subsets based on feature values, forming a tree of decisions that leads to a prediction.

### How it builds (top-down, greedy):

```
1. Start with the full dataset at the root node
2. For every feature, evaluate every possible split threshold
3. Pick the split that gives the BEST reduction in impurity (Gini / Entropy / MSE)
4. Create left and right child nodes with the split data
5. Recurse on each child
6. Stop when a stopping criterion is met:
   - max_depth reached
   - min_samples_split not met
   - min_samples_leaf not met
   - No further impurity reduction possible (pure node)
7. Assign leaf value:
   - Classification → majority class
   - Regression → mean of samples
```

### Key terminology

| Term | Meaning |
|---|---|
| Root node | The very first split (top of tree) |
| Internal node | Any split node in the middle |
| Leaf node | Terminal node — makes the prediction |
| Branch | Path from one node to a child |
| Depth | Longest path from root to a leaf |
| Pure node | All samples belong to one class (entropy = 0) |

---

## 2. Key Parameters (sklearn `DecisionTreeClassifier`)

| Parameter | What it does | Effect on complexity |
|---|---|---|
| `max_depth` | Max levels of the tree | ↑ depth → more complex |
| `min_samples_split` | Min samples needed to split a node | ↑ value → simpler tree |
| `min_samples_leaf` | Min samples required at a leaf | ↑ value → simpler tree |
| `max_features` | No. of features considered at each split | ↓ value → more random |
| `criterion` | Splitting metric (`gini`, `entropy`, `log_loss`) | Changes split quality measure |
| `max_leaf_nodes` | Caps total number of leaves | ↑ value → more complex |
| `min_impurity_decrease` | Split only if impurity drops by at least this | ↑ value → simpler tree |
| `ccp_alpha` | Complexity parameter for cost-complexity pruning | ↑ value → more pruning |

### Quick mental model:
```
max_depth ↓         →  underfit (high bias)
max_depth ↑         →  overfit  (high variance)
min_samples_leaf ↑  →  smoother, less overfit
```

---

## 3. Splitting Criteria

### 3a. Gini Impurity (Classification)

Measures the probability of misclassifying a randomly chosen sample.

$$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$

- $p_i$ = proportion of class $i$ in set $S$
- Range: **[0, 0.5]** for binary classification
- **0** = perfectly pure node
- **0.5** = perfectly impure (50-50 split)

**Weighted Gini for a split:**
$$\text{Gini}_{split} = \frac{n_L}{n} \cdot \text{Gini}(L) + \frac{n_R}{n} \cdot \text{Gini}(R)$$

---

### 3b. Entropy & Information Gain (Classification)

$$\text{Entropy}(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$

- Range: **[0, log₂(C)]** — for binary: [0, 1]
- **0** = pure node
- **1** = maximum uncertainty (50-50 binary)

**Information Gain:**
$$\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v} \frac{|S_v|}{|S|} \cdot \text{Entropy}(S_v)$$

> Pick the split with the **highest Information Gain** (or lowest weighted entropy).

---

### 3c. Variance Reduction (Regression)

$$\text{Variance}(S) = \frac{1}{n} \sum_{i=1}^{n}(y_i - \bar{y})^2$$

**Split score:**
$$\text{VarReduction} = \text{Var}(S) - \frac{n_L}{n}\text{Var}(L) - \frac{n_R}{n}\text{Var}(R)$$

> Maximise variance reduction at each split.

---

### 3d. Gini vs Entropy — Practical Difference

| | Gini | Entropy |
|---|---|---|
| Computation | Faster (no log) | Slower |
| Behaviour | Tends to isolate the most frequent class | More balanced splits |
| Default in sklearn | ✅ Yes | Optional |
| Difference in practice | **Very small** — results are almost identical |

> **FAANG tip:** "Gini is computationally cheaper; Entropy tends to be slightly more balanced. In practice the trees produced are nearly identical."

---

## 4. Underfitting vs Overfitting

```
Simple tree (depth=2)         Deep tree (depth=20)
──────────────────────        ──────────────────────
   Underfit                       Overfit
   High Bias                      High Variance
   Low Variance                   Low Bias
   Bad on train AND test          Great on train, bad on test
```

### Visual intuition

| | Training Error | Test Error |
|---|---|---|
| Underfit | High | High |
| Good fit | Low | Low |
| Overfit | Very Low | High |

### Signs of overfitting in a Decision Tree
- Tree has a leaf for nearly every training sample
- `train_accuracy = 1.0`, `test_accuracy << 1.0`
- Very deep tree with tiny leaf sizes

---

## 5. Bias & Variance — How to Increase / Decrease

### The Tradeoff
$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

### Decision Tree Controls

| To... | Do this |
|---|---|
| **Increase Bias** (reduce complexity) | ↓ max_depth, ↑ min_samples_leaf, ↑ min_samples_split |
| **Decrease Bias** (more expressive) | ↑ max_depth, ↓ min_samples_leaf |
| **Increase Variance** (fit more tightly) | ↑ max_depth, ↓ min_samples_leaf |
| **Decrease Variance** (generalise more) | Pruning, ↓ max_depth, ensemble (Random Forest) |

### The sweet spot
```
              Total Error
                   │  \
                   │   \   ← Variance growing
              Bias²│    \
                   │─────\────── Optimal complexity
                   │      \
                   └────────────────►
                       Model Complexity
```

> A fully grown tree = **zero bias, very high variance**  
> A stump (depth=1) = **high bias, near-zero variance**

---

## 6. Pruning

Pruning reduces tree size to prevent overfitting.

### 6a. Pre-Pruning (Early Stopping)
Stop growing the tree before it's fully grown.

Controlled by:
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `min_impurity_decrease`

**Pros:** Fast, simple  
**Cons:** May stop too early — misses useful splits deeper down

---

### 6b. Post-Pruning (Cost-Complexity / Reduced Error Pruning)

Grow the full tree first, then prune back.

**Cost-Complexity Pruning (sklearn `ccp_alpha`):**

$$\text{Cost}(T) = \text{Error}(T) + \alpha \cdot |T|$$

- $|T|$ = number of leaf nodes
- $\alpha$ = regularisation strength (like λ in ridge)
- Higher $\alpha$ → more leaves are pruned → simpler tree

**How to find the best α:**
```python
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# Try each alpha on validation set, pick best
for alpha in alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha)
    clf.fit(X_train, y_train)
    # evaluate on val set
```

---

### 6c. Reduced Error Pruning (Conceptual)
1. Grow full tree
2. For each internal node, consider replacing subtree with a leaf
3. If test accuracy doesn't decrease → prune it
4. Repeat bottom-up until no improvement

---

## 7. Feature Importance

### How it's computed
Feature importance = total impurity **reduction** attributable to a feature, weighted by the number of samples that passed through those splits.

$$\text{Importance}(f) = \sum_{\text{nodes splitting on } f} \frac{n_{\text{node}}}{n_{\text{total}}} \cdot \Delta\text{impurity}$$

Then **normalised** so all importances sum to 1.

```python
clf.fit(X_train, y_train)
importances = clf.feature_importances_
# array like [0.45, 0.30, 0.15, 0.10] — sums to 1.0
```

### Caveats
- **Biased towards high-cardinality features** (features with many unique values get more chances to split)
- Importances are **not** the same as causation
- For unbiased importance → use **Permutation Importance** instead

### Permutation Importance (unbiased alternative)
1. Train model, record baseline accuracy
2. Shuffle one feature column completely (break its relationship with target)
3. Re-evaluate accuracy
4. Drop in accuracy = importance of that feature
5. Repeat for every feature

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(clf, X_test, y_test, n_repeats=10)
```

---

## 8. Numerical Examples

### Example 1 — Gini Impurity

Dataset: 10 samples — 6 class A, 4 class B

$$\text{Gini} = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = \mathbf{0.48}$$

After split:
- Left: 5 samples — 5A, 0B → Gini = 1 − (1² + 0²) = **0.0** (pure!)
- Right: 5 samples — 1A, 4B → Gini = 1 − (0.2² + 0.8²) = 1 − 0.68 = **0.32**

Weighted Gini of split:
$$= \frac{5}{10}(0.0) + \frac{5}{10}(0.32) = 0 + 0.16 = \mathbf{0.16}$$

**Gini reduction = 0.48 − 0.16 = 0.32** → This is a great split!

---

### Example 2 — Entropy & Information Gain

Parent: 10 samples — 5A, 5B

$$\text{Entropy}(S) = -(0.5\log_2 0.5 + 0.5\log_2 0.5) = -(-0.5 - 0.5) = \mathbf{1.0}$$

After split:
- Left (6 samples): 4A, 2B → $-(4/6)\log_2(4/6) - (2/6)\log_2(2/6) ≈ 0.918$
- Right (4 samples): 1A, 3B → $-(1/4)\log_2(1/4) - (3/4)\log_2(3/4) ≈ 0.811$

Weighted entropy:
$$= \frac{6}{10}(0.918) + \frac{4}{10}(0.811) = 0.551 + 0.324 = \mathbf{0.875}$$

$$\text{IG} = 1.0 - 0.875 = \mathbf{0.125}$$

---

### Example 3 — Regression Tree Split

Data: y = [1, 2, 3, 10, 11, 12]

Overall mean = 6.5, Variance = 20.25

Split at 5 (left: [1,2,3], right: [10,11,12]):
- Left mean = 2, Var = 0.67
- Right mean = 11, Var = 0.67

Weighted variance:
$$= \frac{3}{6}(0.67) + \frac{3}{6}(0.67) = \mathbf{0.67}$$

Variance reduction = 20.25 − 0.67 = **19.58** → Excellent split!

---

## 9. FAANG Conceptual Q&A

---

**Q1: Why can't a decision tree extrapolate beyond training data range?**  
**A:** Leaves predict the mean (regression) or majority class (classification) of training samples in that region. There is no function learned beyond the boundaries — if a test value is larger than any training value, the tree just uses the last leaf's prediction.

---

**Q2: Decision trees are said to have high variance. Why?**  
**A:** A small change in the training data (even one point) can cause a completely different split at the root, cascading into a very different tree structure. This sensitivity to data = high variance.

---

**Q3: How does a decision tree handle categorical features?**  
**A:** For binary categoricals, it tries each split. For multi-class categoricals, it tests all possible subsets (2^k − 1 partitions), which is expensive. Libraries like CatBoost handle high-cardinality categoricals more efficiently using target encoding or ordered statistics.

---

**Q4: Why are Random Forests better than a single Decision Tree?**  
**A:** A single tree has high variance. Random Forest trains many trees on bootstrap samples (bagging) with a random subset of features at each split. Averaging many high-variance, low-bias trees reduces overall variance without increasing bias — this is the bias-variance tradeoff in action.

---

**Q5: What happens when two features have equal Information Gain?**  
**A:** sklearn breaks ties arbitrarily (first one encountered). In practice, you'd prefer the feature that is cheaper to measure / more interpretable. FAANG follow-up: "Prefer the feature with lower cardinality to avoid overfitting."

---

**Q6: Can Decision Trees handle missing values natively?**  
**A:** Standard CART in sklearn cannot — missing values must be imputed before training. XGBoost and LightGBM handle them natively by learning a default direction for missing values at each split.

---

**Q7: How would you detect and fix overfitting in a deployed Decision Tree?**  
**A:**  
1. Compare train vs validation accuracy — large gap = overfit  
2. Fix: reduce `max_depth`, increase `min_samples_leaf`, apply `ccp_alpha` post-pruning  
3. Or replace with Random Forest / Gradient Boosting  

---

**Q8: Gini or Entropy — which to use?**  
**A:** In practice, they produce nearly identical trees. Gini is faster (no log computation). Use Entropy if you explicitly want to reason in terms of information theory. Default to Gini unless instructed otherwise.

---

**Q9: How is feature importance computed, and what's its main weakness?**  
**A:** Sum of impurity reduction weighted by samples through each split node, normalised to sum to 1. Weakness: biased toward high-cardinality / continuous features which have more split opportunities. Use **permutation importance** for unbiased estimates.

---

**Q10: How does a Decision Tree compare to Logistic Regression?**

| | Decision Tree | Logistic Regression |
|---|---|---|
| Decision boundary | Axis-aligned, non-linear | Linear |
| Interpretability | Very high | High |
| Handles non-linearity | Yes | No (without feature engineering) |
| Sensitive to scale | No | Yes (needs standardisation) |
| Overfitting risk | High (single tree) | Low-Medium |
| Extrapolation | Cannot | Can |

---

**Q11: What is the time complexity of training a Decision Tree?**  
**A:** $O(n \cdot d \cdot n \log n)$ where $n$ = samples, $d$ = features. At each node, for each feature, we sort the values ($O(n \log n)$) and evaluate all thresholds. Over $d$ features and up to $n$ nodes → $O(n \cdot d \cdot n \log n)$.

---

**Q12: Why don't we need feature scaling for Decision Trees?**  
**A:** Splits are based on thresholds on individual features, not distances or dot products. Whether `age` is in [0,1] or [0,100] doesn't affect the split decision — only the ordering matters, not the magnitude.

---

## Summary Cheatsheet

```
Decision Tree
├── Split Criteria
│   ├── Gini (fast, default)
│   ├── Entropy / Info Gain (balanced)
│   └── Variance Reduction (regression)
│
├── Complexity Control
│   ├── max_depth ↓ → simpler
│   ├── min_samples_leaf ↑ → simpler
│   └── ccp_alpha ↑ → more pruning
│
├── Bias-Variance
│   ├── Deep tree → low bias, HIGH variance
│   └── Shallow tree → HIGH bias, low variance
│
├── Fix Overfit
│   ├── Prune (pre or post)
│   └── Use Random Forest (ensemble)
│
└── Feature Importance
    ├── Impurity-based (fast, biased to high cardinality)
    └── Permutation importance (unbiased, use on test set)
```
