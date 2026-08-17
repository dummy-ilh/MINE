# Pruning, Missing Values & Class Imbalance — Master Notes

---

# PART 1 — PRUNING, END TO END

## 1.1 Why Pruning Exists at All

Left completely unconstrained, a decision tree keeps splitting until every leaf is pure (or has just 1 sample). This **always** drives training error to zero — and it's a textbook case of overfitting: the tree has memorized noise, not learned signal.

$$\text{Unconstrained tree} \implies \text{training error} \to 0, \quad \text{validation error} \to \text{often much worse}$$

Pruning is the fix: deliberately limiting how big/complex the tree is allowed to get, trading a bit of training accuracy for much better generalization.

There are two families: **pre-pruning** (stop growth before it happens) and **post-pruning** (grow fully, then cut back). This file covers both, but spends most of its time on post-pruning's cost-complexity method since it's the one with real math behind it.

---

## 1.2 Pre-Pruning (Early Stopping)

Constraints applied **during** growth, so the tree never gets big enough to overfit in the first place.

| Constraint | Plain meaning |
|---|---|
| `max_depth` | Cap how many questions deep the tree can go |
| `min_samples_split` | A node needs at least this many samples to even be considered for splitting |
| `min_samples_leaf` | Every resulting leaf must have at least this many samples |
| `min_impurity_decrease` | A split must reduce impurity by at least this much to be accepted |
| `max_leaf_nodes` | Cap on the total number of leaves, tree-wide |

**Weakness — the horizon effect:** a split that looks useless *right now* (tiny impurity decrease) might be the necessary stepping stone to a great split one level deeper. Pre-pruning can't see that far ahead — it just cuts the branch before it gets the chance.

**Numerical illustration of the horizon effect:**
Suppose splitting on "is size > 2,000?" barely reduces impurity (say Δ = 0.01, and your `min_impurity_decrease` threshold is 0.02) — so pre-pruning stops here. But if you'd allowed that split anyway, the "size > 2,000" branch might have then split beautifully on "is location = downtown?" (Δ = 0.30). Pre-pruning with a impurity-decrease threshold never finds out — it's blocked at the first, unimpressive split.

---

## 1.3 Post-Pruning: Cost-Complexity Pruning (CCP) — The Full Derivation

This is CART's method (`ccp_alpha` in sklearn), and the one interviewers actually want you to be able to derive and compute by hand.

### 1.3.1 The Formula

For any subtree $T$ cut from the fully-grown tree $T_0$:

$$\boxed{R_\alpha(T) = R(T) + \alpha \cdot |T|}$$

| Symbol | Meaning |
|---|---|
| $R(T)$ | Training error of subtree $T$ (misclassification rate for classification, MSE for regression) |
| $\vert T\vert$ | Number of leaves in $T$ — the complexity penalty |
| $\alpha \geq 0$ | The price you're willing to pay per extra leaf (`ccp_alpha` in sklearn) |
| $R_\alpha(T)$ | "Penalized" error — what you actually minimize |

**Why this specific form (linear in leaf count)?** It's the direct tree analogue of L1/L2 penalties in linear regression — a knob that trades training fit against model complexity, where "complexity" for a tree is naturally measured by how many independent constant-prediction regions (leaves) it carves out.

**The two extremes:**
- $\alpha \to 0$: penalty vanishes, $R_\alpha(T) \to R(T_0)$ — the fully-grown tree wins (no cost for size).
- $\alpha \to \infty$: penalty dominates everything else, so the single-leaf tree (just predict the overall mean/majority class) wins — it minimizes $|T|=1$.

### 1.3.2 The Algorithm — "Weakest Link" Pruning

You do **not** need to search every possible subtree (which would be combinatorially huge). Breiman et al. (1984) proved that as $\alpha$ increases from 0, there's a **nested sequence** of optimal subtrees:

$$T_0 \supset T_1 \supset T_2 \supset \dots \supset \{\text{root}\}$$

Each subtree in this sequence is found by repeatedly removing whichever branch is the "weakest link" — the one whose removal costs the least error increase *per leaf removed*. That "cost per leaf" is the **effective alpha**:

$$\boxed{\alpha_{\text{eff}}(t) = \frac{R(\text{collapse } t \text{ to a leaf}) - R(\text{subtree rooted at } t)}{|T_t| - 1}}$$

where $T_t$ is the subtree hanging below node $t$, and $|T_t|$ is its leaf count.

**The algorithm, step by step:**
1. Compute $\alpha_{\text{eff}}$ for every internal (non-leaf) node in the fully-grown tree.
2. Find the node with the **smallest** $\alpha_{\text{eff}}$ — this is the "weakest link," i.e., the branch that's buying the least accuracy per unit of complexity.
3. Collapse that branch into a single leaf. Record this $\alpha_{\text{eff}}$ as the threshold at which this subtree becomes optimal.
4. Repeat on the now-smaller tree, recomputing $\alpha_{\text{eff}}$ values as needed, until you're down to the root.
5. This produces the entire path of $(\alpha, \text{subtree})$ pairs in one pass — no need to separately test every possible $\alpha$ from scratch.
6. Pick the final $\alpha$ (and its corresponding tree) using cross-validation or a held-out validation set — whichever tree in the path performs best on data it wasn't grown on.

### 1.3.3 Worked Numerical #1 — Single Branch, Full Calculation

Suppose a subtree $T_t$ has 4 leaves, with total training misclassification error $R(T_t) = 0.10$ (10%).

If you collapse this entire subtree into one leaf (predicting the majority class for everything that would've gone into those 4 leaves), the error rises to $R(\text{leaf}) = 0.20$ (20%).

**Step 1 — compute the numerator (error cost of collapsing):**
$$0.20 - 0.10 = 0.10$$

**Step 2 — compute the denominator (leaves saved):**
$$|T_t| - 1 = 4 - 1 = 3$$

**Step 3 — divide:**
$$\alpha_{\text{eff}} = \frac{0.10}{3} = 0.0333$$

**Interpretation:**
- For any $\alpha < 0.0333$: the 4-leaf subtree is worth keeping — the accuracy you gain per extra leaf beats what you're being charged per leaf.
- For any $\alpha > 0.0333$: collapsing to 1 leaf is better — the penalty per leaf outweighs the accuracy gain.
- At exactly $\alpha = 0.0333$: indifferent between the two (this is the threshold).

### 1.3.4 Worked Numerical #2 — Comparing Two Branches to Find the Weakest Link

Say your fully-grown tree has two candidate branches you could collapse:

| Branch | Leaves before collapse | $R$ before collapse | $R$ after collapse to 1 leaf | $\alpha_{\text{eff}}$ |
|---|---|---|---|---|
| Branch A | 4 | 0.10 | 0.20 | $\frac{0.20-0.10}{4-1} = 0.0333$ |
| Branch B | 3 | 0.15 | 0.19 | $\frac{0.19-0.15}{3-1} = 0.0200$ |

**Which gets pruned first?** Branch B — it has the **smaller** $\alpha_{\text{eff}}$ (0.0200 < 0.0333), meaning it's buying less accuracy per leaf than Branch A. It's the "weakest link" — the least cost-effective piece of the tree, so it's the first one sacrificed as $\alpha$ increases from 0.

This is the mechanical heart of weakest-link pruning: at every step, always cut whichever branch has the smallest $\alpha_{\text{eff}}$, because that's the branch contributing the least value per unit of complexity it costs.

### 1.3.5 Worked Numerical #3 — Full Tree Pruning Path, Start to Finish

Toy regression tree, fully grown, with 3 prunable internal nodes:

| Node | Leaves under it ($\vert T_t\vert$) | MSE of subtree $R(T_t)$ | MSE if collapsed to 1 leaf | $\alpha_{\text{eff}}$ |
|---|---|---|---|---|
| Node 1 | 2 | 8.0 | 10.0 | $\frac{10.0-8.0}{2-1} = 2.00$ |
| Node 2 | 3 | 5.0 | 9.5 | $\frac{9.5-5.0}{3-1} = 2.25$ |
| Node 3 | 2 | 6.0 | 6.5 | $\frac{6.5-6.0}{2-1} = 0.50$ |

**Building the pruning path:**

**Step 1:** Smallest $\alpha_{\text{eff}}$ is Node 3 (0.50) → collapse it first. This subtree survives for $\alpha \in [0, 0.50)$.

**Step 2:** Recompute on the now-smaller tree. Say after collapsing Node 3, Node 1's numbers stay the same (2.00) and Node 2's numbers stay the same (2.25) — smallest is now Node 1 (2.00) → collapse it next. Survives for $\alpha \in [0.50, 2.00)$.

**Step 3:** Only Node 2 left (2.25) → collapse it, reaching the root. Survives for $\alpha \in [2.00, 2.25)$.

**Step 4:** For $\alpha \geq 2.25$, everything is collapsed to the single-leaf root tree.

**The full path:**

| $\alpha$ range | Tree state |
|---|---|
| $[0, 0.50)$ | Fully grown tree $T_0$ |
| $[0.50, 2.00)$ | Node 3's branch collapsed |
| $[2.00, 2.25)$ | Node 3 and Node 1's branches collapsed |
| $[2.25, \infty)$ | Everything collapsed — single-leaf root |

sklearn's `cost_complexity_pruning_path()` computes exactly this table for you in one call, given a fitted tree — then you evaluate each candidate tree in the path on a validation set (or via cross-validation) and pick whichever performs best.

---

## 1.4 Pre- vs. Post-Pruning — When to Use Which

| | Pre-pruning | Post-pruning (CCP) |
|---|---|---|
| When applied | During growth | After growing to full depth |
| Sees full picture before deciding? | No — greedy, can't look ahead (horizon effect) | Yes — every branch's full value is known before deciding what to cut |
| Compute cost | Cheaper — tree never gets big | More expensive — must grow the whole tree first, then prune |
| Typical use | Large datasets, many trees (e.g., inside a Random Forest with hundreds of trees) where full-depth growth + full pruning per tree isn't tractable | Single trees, or whenever compute allows — generally the more principled choice |

**Why prefer post-pruning when you can afford it?** It never suffers the horizon effect — it sees the fully realized value of every branch before deciding what to cut. The only real downside is cost, which is why pre-pruning limits often get used anyway inside ensembles, with the ensemble's averaging providing regularization instead of per-tree pruning.

---

## 1.5 sklearn Cheat Sheet — Pruning-Relevant Parameters

| Parameter | Type | What it controls |
|---|---|---|
| `max_depth` | Pre-pruning | Max depth of tree |
| `min_samples_split` | Pre-pruning | Minimum samples to attempt a split |
| `min_samples_leaf` | Pre-pruning | Minimum samples per leaf |
| `min_impurity_decrease` | Pre-pruning | Minimum Δimpurity a split must achieve |
| `max_leaf_nodes` | Pre-pruning | Cap on total leaves |
| `ccp_alpha` | Post-pruning | Complexity penalty per leaf (Section 1.3) — default 0.0 = no pruning |
| `cost_complexity_pruning_path()` | Post-pruning utility | Returns the full $(\alpha, \text{tree})$ path in one call |

---

## 1.6 Quick Q&A — Pruning

**Q: Why is the penalty in cost-complexity pruning linear in leaf count, rather than, say, quadratic?**
A: It's the simplest possible complexity measure that's monotonic in tree size — more leaves always costs more, proportionally. This mirrors L1 regularization's linear penalty on coefficient magnitude; there's no strong theoretical reason it must be linear specifically, but it's simple, interpretable (each leaf has a fixed "price"), and it's what produces the clean nested-subtree-sequence property that makes the whole path computable in one efficient pass.

**Q: If you could compute cost-complexity pruning path in one pass, why does sklearn still make you pick $\alpha$ via cross-validation instead of just picking automatically?**
A: The pruning path tells you *which trees are achievable* at each $\alpha$ — it doesn't tell you which one generalizes best, since that requires data the tree wasn't grown on. Picking $\alpha$ still needs a validation signal because different datasets will have a different "sweet spot" for the bias/complexity trade-off, and that sweet spot can only be estimated by testing candidate trees on held-out data.

**Q: Can `max_depth` and `ccp_alpha` be used together?**
A: Yes, and it's common in practice — use `max_depth` (or another pre-pruning limit) to keep the initial grow tractable on large data, then apply `ccp_alpha` for the finer, more principled trim on top of that. They're not mutually exclusive; they operate at different stages.

---

# PART 2 — MISSING VALUE HANDLING

## 2.1 The Core Problem

Real data almost always has gaps — a lot size never recorded, a sensor reading that failed, a survey question someone skipped. A tree's split logic ("is size > 2,000?") has no defined answer when the value is missing — every method for handling this is really a different answer to "what do we do when we hit that undefined comparison?"

## 2.2 Method 1 — Impute Before Training (what plain sklearn requires)

**What it is:** fill in missing values with something — a fixed constant, the column mean/median, the mode (for categorical), or a value predicted by a separate small model — *before* the tree ever sees the data.

**sklearn's actual behavior:** `DecisionTreeClassifier`/`Regressor` and `RandomForestClassifier`/`Regressor` **do not support missing values at all** in most sklearn versions — you must impute first, using something like `SimpleImputer`.

**Why this can lose information:** imputing lot size with the column average assumes missingness is *random* — but often it isn't. If lot size is missing specifically for older, rural houses that were never formally surveyed, forcing all of them to look like "the average house" erases a real, systematic pattern (those houses might also be systematically cheaper) instead of letting the model learn from it.

**Numerical illustration:** suppose lot size is missing for 20 houses, and those 20 houses have a true average price of $180,000, while the overall dataset average price is $310,000. If you also fill their "lot size" with the overall average lot size (say 8,000 sq ft, when their real average is closer to 3,000 sq ft — small rural parcels), the model will see 20 houses that *look* like average-sized-lot houses but have unusually low prices — creating a confusing, contradictory training signal instead of a clean "small/missing lot size → lower price" pattern it could have learned directly.

## 2.3 Method 2 — Surrogate Splits (classic CART theory)

**What it is:** at every split, in addition to the primary (feature, threshold) rule, CART theory allows storing a ranked list of **backup rules** — other (feature, threshold) pairs chosen because they best *mimic* the primary split's sample assignment on the rows that do have the primary feature observed.

**How it works at prediction time:** if a sample is missing the primary split's feature, fall back to surrogate #1. If that's also missing, fall back to surrogate #2. And so on. If every surrogate is also missing, fall back to the majority direction (whichever branch most training samples went to).

**Important practical note:** sklearn's trees **do not implement surrogate splits** — this is textbook CART theory, not something available in sklearn's actual implementation. If you need this behavior, you're either building it yourself or using a library that supports it.

## 2.4 Method 3 — Native Missing-Value Handling (XGBoost, LightGBM)

**What it is:** during training, for *each split individually*, the algorithm tries sending all missing-valued samples left, then tries sending them all right, and keeps whichever direction produces better training performance at that specific node. Different splits can send missing values in different directions — the best default can genuinely differ node to node.

**Why this beats both imputation and surrogate splits in practice:**
- vs. imputation: no information is thrown away or distorted — the model learns the best handling directly from what actually predicts the target well, so if missingness itself correlates with the outcome, the model can exploit that instead of having it erased.
- vs. surrogate splits: it's built directly into modern gradient boosting libraries (XGBoost, LightGBM) and is cheap to compute — no need for a separate backup-rule search per split, and unlike sklearn's plain trees, it's actually available out of the box.

**Numerical illustration, continuing the lot-size example:** if the 20 houses with missing lot size have a true average price of $180,000, and houses with small (but known) lot sizes similarly average $190,000, XGBoost's training process will naturally discover that sending "missing lot size" down the same branch as "small lot size" produces better splits — capturing the real pattern (missingness correlates with rural/cheap houses) automatically, without you ever telling it to.

## 2.5 Comparison Table

| Method | Where it happens | Needs manual work? | Captures "missingness is informative"? | Available in plain sklearn? |
|---|---|---|---|---|
| Impute upstream | Before training | Yes — you choose the fill value | No — usually erases the pattern | Yes (via `SimpleImputer`, then fit) |
| Surrogate splits | During training, per split | No, but requires a CART implementation that supports it | Partially — backup rule can approximate it | **No** |
| Native handling (XGBoost/LightGBM) | During training, per split | No | Yes — directly, and automatically | No (sklearn's own trees don't have this) |

## 2.6 Quick Q&A — Missing Values

**Q: Why does XGBoost's native handling beat mean/median imputation specifically when missingness correlates with the target?**
A: Mean/median imputation makes every missing-valued row look statistically "typical" for that feature, which actively hides any relationship between *being missing* and the outcome. Native handling never manufactures a fake value at all — it just asks "which branch do these rows predict best in?" directly against the real target, so if missingness correlates with a lower or higher outcome, that correlation shows up naturally in which direction the split search prefers.

**Q: If you're stuck using plain sklearn (no native missing-value support), what's the best practical workaround to avoid erasing a real missingness pattern?**
A: Add an explicit "was this value missing" indicator column (a 0/1 flag) alongside the imputed value, then impute the original column normally. This way, even though the actual imputed number is generic, the tree can still split on the flag itself if missingness itself is predictive — recovering most of the signal that plain single-value imputation would otherwise throw away.

**Q: Is missing-value handling a training-time-only concern, or does it also matter at inference time?**
A: Both, and they have to match exactly. Whatever imputation or native-handling logic ran during training must be reproduced identically at inference time — a different default fill value, or a different fallback direction, causes a silent training/serving mismatch where the model makes a prediction based on logic it was never actually trained under.

---

# PART 3 — CLASS IMBALANCE HANDLING

## 3.1 The Core Problem

Say you're predicting "will this house sell within 30 days," and only 5% of houses do. A model can hit **95% accuracy** by always predicting "no" — while being completely useless for the one thing you actually care about (finding the houses that will sell fast).

**Why plain accuracy hides this:** accuracy treats every correct prediction the same, regardless of class — so a model that's perfect on the common class and terrible on the rare class can still look great by this one number alone.

## 3.2 Fix 1 — Class Weights

**What it is:** tell the model to treat mistakes on the rare class as more costly than mistakes on the common class, during training.

**How it works mechanically:** the impurity calculation at each split (Gini, entropy, or MSE) gets weighted by each sample's class weight instead of treating every sample equally — so a node that's "impure" because of a few misclassified rare-class samples counts for more than the raw counts alone would suggest, pushing the split search to actually pay attention to getting those right.

**sklearn's `class_weight='balanced'` — the actual formula:**

$$w_k = \frac{n_{\text{total}}}{K \cdot n_k}$$

where $n_{\text{total}}$ = total samples, $K$ = number of classes, $n_k$ = number of samples in class $k$.

**Worked numerical:** 1,000 houses total, 950 "didn't sell" (class 0), 50 "sold quickly" (class 1), $K=2$.

$$w_0 = \frac{1000}{2 \times 950} = \frac{1000}{1900} = 0.526$$

$$w_1 = \frac{1000}{2 \times 50} = \frac{1000}{100} = 10.0$$

**Interpretation:** each "sold quickly" sample counts as if it were worth about **19x** as much as a "didn't sell" sample ($10.0 / 0.526 \approx 19$) during split-quality calculations. A wrong guess on one rare-class house now has real, weighted consequence for the impurity score — instead of being drowned out by the sheer number of common-class houses.

## 3.3 Fix 2 — Resampling

**Oversampling:** duplicate (or synthetically generate, e.g. via SMOTE) more examples of the rare class until the classes are closer to balanced.

**Undersampling:** throw away some examples of the common class until the classes are closer to balanced.

**Numerical illustration:** same 950/50 split.
- Oversampling to full balance: duplicate the 50 rare-class rows up to 950 (either by repeating them ~19x each, or generating synthetic variants) → 950/950.
- Undersampling to full balance: randomly keep only 50 of the 950 common-class rows → 50/50, but now training on only 100 total rows instead of 1,000 — a real information cost.

**Trade-off:** oversampling risks the model overfitting to the (possibly small number of) repeated/synthetic rare examples — it can become overconfident about those specific instances rather than generalizing. Undersampling risks throwing away real, useful information from the majority class, especially painful when the total dataset isn't large to begin with.

## 3.4 Fix 3 — Bagging-Specific: Balanced Bootstrap Sampling

Bagging already resamples rows for every tree (with replacement, Chapter 3 of this curriculum). Imbalance handling can be built directly into that same step — deliberately draw each bootstrap sample to be more class-balanced than the raw data, rather than adding a separate resampling stage beforehand. This is a natural fit specifically because bagging is already in the business of resampling rows for every tree anyway — sklearn's `BalancedBaggingClassifier` (imbalanced-learn library) implements exactly this.

## 3.5 Comparison Table

| Method | What changes | Needs extra data? | Risk |
|---|---|---|---|
| Class weights | Impurity calculation during split search | No | None major — cheapest, safest default |
| Oversampling | The training set itself (more rare-class rows) | No (duplicates/synthesizes existing data) | Overfitting to repeated/synthetic rare examples |
| Undersampling | The training set itself (fewer common-class rows) | No (but throws data away) | Losing real majority-class information, especially with small datasets |
| Balanced bagging | Each tree's bootstrap sample specifically | No | Same general trade-offs as oversampling/undersampling, applied per-tree instead of globally |

## 3.6 Why Not Just Use Accuracy to Check If Any of This Worked

With 95% "didn't sell," accuracy stays misleadingly high even for a completely useless model. Better metrics:

| Metric | What it actually measures |
|---|---|
| Precision | Of everything the model flagged as "sold quickly," what fraction really did? |
| Recall | Of everything that really did sell quickly, what fraction did the model catch? |
| F1 | Harmonic mean of precision and recall — a single number balancing both |
| Confusion matrix | Full breakdown — lets you see exactly which errors are happening, not just a summary number |

These directly show whether the model is catching the rare, important cases — which is usually the whole reason you cared about the rare class in the first place.

## 3.7 Quick Q&A — Class Imbalance

**Q: When would you reach for class weights instead of resampling?**
A: Class weights first, as the default — it's cheaper (no extra training data manipulation needed, and no risk of throwing away real data or overfitting to duplicated rows), and it directly targets the actual mechanism causing the problem (the split search under-weighting rare-class errors). Resampling is worth adding on top when class weights alone aren't enough, or when you specifically want to change the effective training-set size for other reasons.

**Q: Why might undersampling be a bad idea on a small dataset even if it perfectly balances the classes?**
A: Because it throws away real majority-class data to get there — with a 950/50 split undersampled down to 50/50, you're training on only 100 total rows instead of 1,000, a 90% reduction in training data. On a small dataset to begin with, that data loss can hurt more than the imbalance itself was hurting, especially since the majority class's full data still carries real, useful signal about what "normal" (not-rare-event) cases look like.

**Q: You've handled imbalance and now the model's F1 score improved. Does that guarantee the model is now "good"?**
A: Not automatically — F1 balances precision and recall but still needs to be read against what actually matters for the use case (e.g., is a false negative on the rare class much more costly than a false positive, or vice versa?). Sometimes precision or recall alone, or a cost-weighted metric, is a better fit than F1's default 50/50 balance between the two — worth checking which error type is actually more expensive before picking your primary metric, not just defaulting to F1 because it's the standard imbalance-friendly choice.

---

**One-line summary to remember:** *Pruning: pre-pruning stops growth early (cheap, but can miss good splits behind bad ones); post-pruning grows fully then trims using $R_\alpha(T)=R(T)+\alpha|T|$, always cutting the branch with the smallest cost-per-leaf ($\alpha_{\text{eff}}$) first. Missing values: plain sklearn needs imputation (which can erase informative missingness); XGBoost/LightGBM learn the best default direction per split automatically. Imbalance: class weights (cheap, safe default) or resampling (oversample/undersample, real trade-offs either way) — and always validate with precision/recall/F1/confusion matrix, never plain accuracy.*
