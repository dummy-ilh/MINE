# Pruning Methods in Decision Trees
## Complete Notes with Worked Examples

---

## Why Pruning?

A fully grown decision tree will memorise the training data — creating a leaf for almost every sample. This is **overfitting**.

```
Unpruned Tree                   Pruned Tree
─────────────                   ────────────
       [Root]                        [Root]
      /       \                     /       \
   [A]         [B]               [A]         [B]
   / \         / \               / \
 [C] [D]     [E] [F]          [C] [D]
 /\   /\     /\   /\
[G][H][I][J][K][L][M][N]

Train acc: 100%                 Train acc: 92%
Test acc:  71%                  Test acc:  89%  ✅
```

**Pruning = deliberately simplifying the tree to improve generalisation.**

---

## Two Families of Pruning

```
Pruning
├── Pre-Pruning  (Early Stopping)
│     Stop growing the tree BEFORE it overfits
│     → Prevent nodes from being created
│
└── Post-Pruning (Backward Pruning)
      Grow full tree FIRST, then cut back
      → Remove nodes that don't help on unseen data
```

---

## PRE-PRUNING (Early Stopping)

### Idea
Add stopping conditions *during* tree growth. If a condition is triggered, don't split that node — make it a leaf.

### Methods

---

### 1. Max Depth (`max_depth`)

**Stop splitting when the tree reaches a depth limit.**

```
max_depth = 3

          [Root]          ← depth 0
         /       \
      [d1]       [d1]     ← depth 1
      /  \       /  \
   [d2] [d2] [d2] [d2]   ← depth 2
   / \                    ← depth 3 → STOP, make leaves
[d3][d3]
```

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3)
```

**When to use:** Quick baseline. Always set this as a first guard.  
**Risk:** May stop too early — useful splits at deeper levels are never explored.

---

### 2. Minimum Samples to Split (`min_samples_split`)

**Don't split a node if it has fewer than N samples.**

```
min_samples_split = 10

Node with 8 samples → DON'T SPLIT → make leaf
Node with 15 samples → OK to split → continue
```

```python
clf = DecisionTreeClassifier(min_samples_split=10)
```

**Effect:** Prevents tiny, spurious splits deep in the tree.

---

### 3. Minimum Samples at Leaf (`min_samples_leaf`)

**A split is only allowed if BOTH resulting leaves have at least N samples.**

```
min_samples_leaf = 5

Proposed split:
  Left child:  3 samples  ← LESS than 5 → REJECT this split
  Right child: 12 samples

Another split:
  Left child:  7 samples  ← OK
  Right child: 9 samples  ← OK → ACCEPT
```

```python
clf = DecisionTreeClassifier(min_samples_leaf=5)
```

**This is often more useful than min_samples_split** — it directly controls leaf size.

---

### 4. Minimum Impurity Decrease (`min_impurity_decrease`)

**Only split if the impurity drops by at least δ.**

$$\Delta\text{impurity} = \text{Impurity}(\text{parent}) - \frac{n_L}{n}\text{Impurity}(L) - \frac{n_R}{n}\text{Impurity}(R) \geq \delta$$

```python
clf = DecisionTreeClassifier(min_impurity_decrease=0.01)
```

**Effect:** Ignores trivial splits that don't meaningfully reduce uncertainty.  
**Risk:** Hard to set — depends on the scale of impurity values in your data.

---

### 5. Max Leaf Nodes (`max_leaf_nodes`)

**Cap the total number of leaves in the tree.**

Tree grows using best-first (not depth-first) — always expanding the leaf with highest impurity reduction, until the cap is reached.

```python
clf = DecisionTreeClassifier(max_leaf_nodes=20)
```

**Effect:** Budget-based pruning — use exactly 20 decisions, no more.

---

### Pre-Pruning Summary Table

| Parameter | Controls | ↑ value effect |
|---|---|---|
| `max_depth` | Tree height | Simpler tree |
| `min_samples_split` | Node split eligibility | Simpler tree |
| `min_samples_leaf` | Leaf size | Simpler tree |
| `min_impurity_decrease` | Split quality threshold | Simpler tree |
| `max_leaf_nodes` | Total leaf budget | Simpler tree |

### ⚠️ Pre-Pruning Weakness: The Horizon Problem

```
A split at depth 4 looks useless alone...
BUT it enables a very powerful split at depth 5.

Pre-pruning cuts at depth 4 → never sees the depth-5 benefit.
Post-pruning grows both → then evaluates them together.
```

---

## POST-PRUNING (Backward Pruning)

Grow the **full** tree first, then remove nodes that hurt generalisation.

---

### 1. Reduced Error Pruning (REP)

**Simplest post-pruning method.**

#### Algorithm
```
1. Grow the full tree on training data
2. Hold out a validation set (not used in training)
3. For each internal node (bottom-up):
     a. Temporarily replace the subtree with a leaf
        (leaf predicts majority class of its training samples)
     b. Evaluate accuracy on validation set
     c. If accuracy doesn't decrease → PRUNE IT (keep the leaf)
     d. If accuracy decreases → KEEP the subtree
4. Repeat until no more pruning improves / maintains accuracy
```

#### Example

```
Before pruning:          Val accuracy = 85%
        [A]
       /   \
     [B]   [C]
     / \
   [D] [E]          ← try pruning this subtree

Replace [B] with leaf:   Val accuracy = 86%  → PRUNE ✅
        [A]
       /   \
    [B*]   [C]      ← B* is now a leaf

Replace [C] with leaf:   Val accuracy = 83%  → KEEP ❌
```

**Pros:** Simple, fast, effective  
**Cons:** Requires a separate validation set (wastes some training data)

---

### 2. Cost-Complexity Pruning (CCP) — Weakest Link Pruning

**The method used by sklearn (`ccp_alpha`). Most important for FAANG.**

#### Core Idea
Add a penalty for tree complexity:

$$\text{Cost}(T, \alpha) = \text{Error}(T) + \alpha \cdot |T|$$

- $\text{Error}(T)$ = misclassification rate on training data
- $|T|$ = number of leaf nodes (tree size)
- $\alpha$ = regularisation strength (like λ in Ridge/Lasso)

As $\alpha$ increases → the penalty for size grows → more leaves get pruned.

#### The Algorithm

**Step 1:** For each internal node $t$, compute its **effective alpha** — the α at which pruning that subtree becomes worthwhile:

$$\alpha_{eff}(t) = \frac{\text{Error}(t) - \text{Error}(T_t)}{|T_t| - 1}$$

- $\text{Error}(t)$ = error if we make $t$ a leaf
- $\text{Error}(T_t)$ = error of the subtree rooted at $t$
- $|T_t|$ = number of leaves in subtree at $t$

**Step 2:** Prune the node with the **smallest** $\alpha_{eff}$ (the "weakest link" — least useful subtree).

**Step 3:** Repeat on the pruned tree until only the root remains.

**Step 4:** This produces a sequence of trees $T_0 \supset T_1 \supset T_2 \supset \ldots \supset \{\text{root}\}$

**Step 5:** Use cross-validation to pick the best $\alpha$.

---

#### ✏️ Worked Example — CCP

**Setup:** A node $t$ with 2 leaves below it.

```
         [t]   ← Error if leaf = 0.30 (30% misclassified)
        /   \
      [L1] [L2]
```

- $\text{Error}(T_t)$ = weighted error of L1 + L2 = 0.18 (leaves are purer)
- $|T_t|$ = 2 leaves
- $|T_t| - 1$ = 1

$$\alpha_{eff}(t) = \frac{0.30 - 0.18}{2 - 1} = \frac{0.12}{1} = 0.12$$

**Interpretation:** If we set $\alpha \geq 0.12$, the complexity penalty outweighs the accuracy gain → prune this subtree.

---

#### CCP in sklearn

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Step 1: Get the pruning path (all alpha candidates)
clf_full = DecisionTreeClassifier(random_state=0)
path = clf_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # exclude the trivial root-only tree

# Step 2: Train a tree for each alpha
clfs = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)
    clf.fit(X_train, y_train)
    clfs.append(clf)

# Step 3: Evaluate on validation set and pick best alpha
val_scores = [clf.score(X_val, y_val) for clf in clfs]
best_alpha = ccp_alphas[np.argmax(val_scores)]

# Step 4: Final model
final_clf = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0)
final_clf.fit(X_train, y_train)
```

**Effect of increasing alpha:**
```
alpha = 0.0    → full tree   → overfit
alpha = 0.01   → slight prune
alpha = 0.05   → moderate prune  ← often sweet spot
alpha = 0.2    → heavy prune → underfit
alpha = large  → just root    → worst underfit
```

---

### 3. Minimum Description Length (MDL) Pruning

Based on information theory — **Occam's Razor formalised**.

$$\text{MDL Cost} = \underbrace{L(T)}_{\text{bits to encode tree}} + \underbrace{L(D|T)}_{\text{bits to encode errors given tree}}$$

Prune if the cost of describing the subtree outweighs the cost of the errors it fixes.

**Used more in research than production.** sklearn doesn't implement it directly.

---

### 4. Error-Based Pruning (EBP) — used in C4.5

Instead of using a validation set, EBP uses **statistical confidence intervals** on training error to estimate test error.

$$e_{upper}(t) = \text{Error}(t) + z \cdot \sqrt{\frac{\text{Error}(t)(1 - \text{Error}(t))}{n}}$$

- $z$ = z-score for confidence level (default: 25% → z = 0.69 in C4.5)
- $n$ = number of samples at node

**Algorithm:**
```
For each leaf, compute upper-bound estimated error
For each internal node, compare:
  - Subtree's weighted upper-bound errors
  - Node's own upper-bound error (as leaf)
If leaf error ≤ subtree error → PRUNE
```

**Pros:** No need for separate validation set  
**Cons:** The statistical assumptions are shaky; confidence level is arbitrary

---

## Comparison Table

| Method | Type | Needs Val Set? | sklearn? | Used in |
|---|---|---|---|---|
| Max Depth | Pre | ❌ | ✅ | All trees |
| Min Samples Split | Pre | ❌ | ✅ | All trees |
| Min Samples Leaf | Pre | ❌ | ✅ | All trees |
| Min Impurity Decrease | Pre | ❌ | ✅ | All trees |
| Max Leaf Nodes | Pre | ❌ | ✅ | All trees |
| Reduced Error Pruning | Post | ✅ | ❌ | ID3 |
| Cost-Complexity (CCP) | Post | ✅ (CV) | ✅ (`ccp_alpha`) | CART |
| Error-Based Pruning | Post | ❌ | ❌ | C4.5 |
| MDL Pruning | Post | ❌ | ❌ | Research |

---

## Pre-Pruning vs Post-Pruning — When to Use What

| | Pre-Pruning | Post-Pruning (CCP) |
|---|---|---|
| Speed | ⚡ Faster (tree never grows fully) | Slower (grow then prune) |
| Data efficiency | Better (no val set needed) | Needs val set or CV |
| Horizon problem | ❌ Suffers from it | ✅ Avoids it |
| Fine-grained control | Coarser | More precise |
| Interpretability | Easy to explain | Requires α tuning |
| **Recommended for** | Quick baselines, large data | Production models, careful tuning |

---

## The Effect of Pruning on Bias-Variance

```
No Pruning          Light Pruning       Heavy Pruning
───────────         ─────────────       ─────────────
Low Bias            Slightly ↑ Bias     High Bias
High Variance       ↓ Variance          Low Variance
Overfit             Sweet Spot ✅       Underfit
```

### Pruning parameters and their effect:

| More pruning via... | Bias | Variance |
|---|---|---|
| ↑ `ccp_alpha` | ↑ | ↓ |
| ↓ `max_depth` | ↑ | ↓ |
| ↑ `min_samples_leaf` | ↑ | ↓ |
| ↑ `min_impurity_decrease` | ↑ | ↓ |

---

## FAANG Q&A on Pruning

---

**Q1: What is pruning and why do we need it?**  
**A:** Pruning reduces tree complexity to prevent overfitting. A fully grown tree memorises training data (zero training error, poor test error). Pruning removes branches that add complexity without improving generalisation.

---

**Q2: What's the difference between pre and post pruning?**  
**A:** Pre-pruning stops growth early using stopping criteria (max_depth, min_samples_leaf etc.) — fast but may miss useful deep splits. Post-pruning grows the full tree then removes nodes using a validation set or penalty term (CCP) — more precise but slower.

---

**Q3: Explain ccp_alpha. How do you choose the right value?**  
**A:** `ccp_alpha` is the regularisation parameter in Cost-Complexity Pruning. It penalises each leaf node. Higher alpha → more leaves pruned → simpler tree. To find the best value: use `cost_complexity_pruning_path()` to get candidate alphas, then pick the one with the best cross-validation score.

---

**Q4: What is the "horizon effect" in pre-pruning?**  
**A:** A split that looks useless in isolation may enable a very powerful split one level deeper. Pre-pruning cuts early and never sees this benefit. Post-pruning avoids this because the full tree is grown first — all splits are considered before any are removed.

---

**Q5: Does pruning always help?**  
**A:** On noisy, limited data — yes, almost always. On large, clean datasets — the benefit is smaller since the tree has enough signal to learn without overfitting. Also: if you're going to use Random Forest or GBM, pruning a single tree matters less since ensembles handle variance differently.

---

**Q6: How is CCP different from just limiting max_depth?**  
**A:** `max_depth` is a blunt instrument — it cuts entire levels of the tree uniformly. CCP is surgical — it evaluates each subtree's contribution and prunes only those where the accuracy gain doesn't justify the complexity cost. CCP can produce asymmetric trees (deep on one side, shallow on the other) that `max_depth` cannot.

---

## Summary Cheatsheet

```
PRUNING
│
├── PRE-PRUNING (stop early)
│   ├── max_depth          → hard cap on height
│   ├── min_samples_split  → need enough samples to split
│   ├── min_samples_leaf   → need enough samples at leaf
│   ├── min_impurity_decrease → split must be meaningful
│   └── max_leaf_nodes     → budget cap on leaves
│   ⚠️  Risk: Horizon problem
│
└── POST-PRUNING (cut back)
    ├── Reduced Error       → use val set, prune if no accuracy drop
    ├── Cost-Complexity     → sklearn ccp_alpha, penalty per leaf ✅ MAIN ONE
    ├── Error-Based (C4.5)  → statistical upper bound, no val set needed
    └── MDL                 → info-theoretic, research use
    ✅  Avoids horizon problem
    ⚠️  Needs validation data / cross-validation

KEY RULE:
  ccp_alpha ↑  →  more pruning  →  higher bias, lower variance
  max_depth  ↓  →  more pruning  →  higher bias, lower variance
  
  Best practice: use ccp_alpha + cross-validation for production models
```
