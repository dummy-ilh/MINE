# Chapter 1 — Decision Tree Fundamentals

---

## 1.1 What a Tree Actually Does: Recursive Partitioning

A decision tree predicts by **carving up feature space into axis-aligned rectangles (regions)** and assigning one prediction per region.

Formally, for feature space $X \in \mathbb{R}^p$, the tree partitions $X$ into disjoint regions $R_1, R_2, \dots, R_M$ such that:

$$
\hat{f}(x) = \sum_{m=1}^{M} c_m \cdot \mathbb{1}(x \in R_m)
$$

- Classification: $c_m$ = majority class in region $R_m$ (or class probability vector)
- Regression: $c_m$ = mean of target values in region $R_m$

**How the regions are built — recursive binary splitting:**

1. Start with the full dataset at the root node (one region = everything).
2. Search over every feature and every possible threshold for that feature. For each candidate split, compute how much it would reduce "impurity" (defined precisely in 1.2).
3. Pick the single (feature, threshold) pair that gives the **largest impurity reduction**. Split the node into two children on that pair.
4. Recurse on each child independently, repeating steps 2–3.
5. Stop when a stopping condition is hit (Section 1.4).

This is **greedy** — at each node you pick the locally best split, never looking ahead to see if a "worse now" split might enable a "better later" split two levels down. This greediness is a deliberate trade-off: exhaustively searching all possible *tree structures* (not just one split at a time) is NP-hard, so CART is a polynomial-time greedy approximation.

**Why greedy and not global optimization?**
The number of possible binary trees of depth $d$ with $p$ features grows combinatorially — for even modest $p$ and $d$, exhaustive search over all tree topologies is computationally intractable (this is a known NP-complete problem, proven by Hyafil & Rivest, 1976). Greedy recursive splitting turns an exponential search into a sequence of cheap local searches, at the cost of possibly missing the globally optimal tree. In practice this cost is usually small relative to the compute saved, and ensembling (later chapters) fixes most of the downside.

---

## 1.2 Splitting Criteria — the core of "how do we pick a split"

At every node, we need a number that tells us "how good is this candidate split." That number is an **impurity function** applied before and after the split; we pick the split that reduces impurity the most.

### 1.2.1 Classification: Gini Impurity

For a node with $K$ classes, let $p_k$ = proportion of samples in the node belonging to class $k$.

$$
\text{Gini}(t) = \sum_{k=1}^{K} p_k (1 - p_k) = 1 - \sum_{k=1}^{K} p_k^2
$$

**Why this formula, intuitively:** Gini impurity is the expected error rate if you classified a randomly drawn sample from the node by **randomly guessing according to the class proportions** rather than always guessing the majority class. If you draw one sample with true label $k$ (probability $p_k$), then guess a label at random also using the class distribution, the probability of a *wrong* guess is $1 - p_k$. Summing over classes weighted by their probability of being drawn:

$$
\text{Gini}(t) = \sum_k p_k (1-p_k)
$$

- Pure node (all one class): Gini = 0 (minimum, best).
- Maximally impure node (uniform across $K$ classes): Gini = $1 - \frac{1}{K}$ (maximum for that $K$).

**Worked numerical — Gini at a node:**
Node has 10 samples: 6 class A, 4 class B.
$p_A = 0.6$, $p_B = 0.4$

$$
\text{Gini} = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48
$$

Compare to a node with 5 A / 5 B (maximum impurity for 2 classes):
$$
\text{Gini} = 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5
$$
Confirms 0.5 is the ceiling for binary classification — matches the closed form $1 - 1/K = 1 - 1/2 = 0.5$.

### 1.2.2 Classification: Entropy / Information Gain

$$
\text{Entropy}(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

**Why this formula:** this is Shannon entropy — the expected number of bits needed to encode the class label of a randomly drawn sample, under an optimal code built from the class distribution. A pure node needs 0 bits (you already know the answer). A 50/50 node needs exactly 1 bit (a fair coin flip). Entropy grows with disorder faster than Gini near the pure ends because of the log term.

**Information Gain (IG)** for a candidate split $S$ that partitions parent node $t$ into children $t_L$ (fraction $w_L$ of samples) and $t_R$ (fraction $w_R$):

$$
IG(t, S) = \text{Entropy}(t) - \left[ w_L \cdot \text{Entropy}(t_L) + w_R \cdot \text{Entropy}(t_R) \right]
$$

**Worked numerical — same 6A/4B node, split into two children:**

Parent entropy:
$$
\text{Entropy}(t) = -(0.6 \log_2 0.6 + 0.4 \log_2 0.4)
$$
$\log_2 0.6 = -0.737$, $\log_2 0.4 = -1.322$
$$
= -(0.6 \times -0.737 + 0.4 \times -1.322) = -(-0.4422 - 0.5288) = 0.9710
$$

Suppose a split sends 5 samples left (4A, 1B) and 5 right (2A, 3B):

Left entropy: $p_A=0.8, p_B=0.2$
$$
\text{Entropy}(t_L) = -(0.8\log_2 0.8 + 0.2\log_2 0.2) = -(0.8 \times -0.322 + 0.2 \times -2.322)
$$
$$
= -(-0.2576 - 0.4644) = 0.7220
$$

Right entropy: $p_A=0.4, p_B=0.6$ — by symmetry with the parent-flip case this equals the same 0.9710 (since 0.4/0.6 has identical entropy to 0.6/0.4).

Weighted child entropy: $w_L = w_R = 0.5$
$$
0.5 \times 0.7220 + 0.5 \times 0.9710 = 0.3610 + 0.4855 = 0.8465
$$

$$
IG = 0.9710 - 0.8465 = 0.1245 \text{ bits}
$$

This split reduces average uncertainty by ~0.125 bits per sample. The tree-building algorithm computes this $IG$ (or the Gini-based equivalent below) for **every** candidate (feature, threshold) pair and keeps the max.

### 1.2.3 Gini vs Entropy — Why one over the other?

| | Gini | Entropy |
|---|---|---|
| Formula | $1-\sum p_k^2$ | $-\sum p_k \log_2 p_k$ |
| Compute cost | Cheaper (no log) | Slightly more expensive |
| Sensitivity | Slightly favors larger partitions / majority class | Slightly more sensitive to class balance changes, penalizes impurity a bit more aggressively |
| Range (binary) | [0, 0.5] | [0, 1] |
| Practical difference | In practice, trees built with Gini vs entropy usually agree on splits >95% of the time (Raileanu & Stoffel, 2004 — the two criteria disagree on the chosen split in only a small minority of cases, and final accuracy is nearly indistinguishable) | Same |

**Why does sklearn default to Gini?** Purely computational — no logarithm means faster training, and empirically the resulting trees are statistically almost indistinguishable from entropy-built trees. There is **no theoretically compelling reason** to strictly prefer one; this is one of the few "why not X" answers in ML where the honest answer is "it barely matters, pick the cheaper one."

There's also a third, rarer classification criterion: **Misclassification error** = $1 - \max_k p_k$. This is almost never used to *grow* trees (only sometimes to *prune* them) because it's insensitive — it can assign identical scores to two splits where Gini/Entropy would clearly prefer one, since it only looks at the majority class and ignores how the rest of the distribution is shaped. Concretely: it isn't strictly concave the way Gini/Entropy are, so it fails to reward splits that produce one pure child even when the other child stays mixed — the exact situations where you most want the impurity measure to give credit.

### 1.2.4 Regression: Variance Reduction / MSE

For regression, "impurity" of a node = variance of the target values in that node:

$$
\text{MSE}(t) = \frac{1}{n_t}\sum_{i \in t} (y_i - \bar{y}_t)^2
$$

where $\bar{y}_t$ is the mean target value in node $t$.

The split criterion is **variance reduction** (identical structure to information gain, just swap entropy for MSE):

$$
\Delta = \text{MSE}(t) - \left[ w_L \cdot \text{MSE}(t_L) + w_R \cdot \text{MSE}(t_R) \right]
$$

**Why the mean and squared error specifically?** Because the sample mean is the value that *minimizes* sum of squared errors within a region (a basic least-squares result — the mean is the least-squares-optimal constant predictor for a set of numbers). So once you decide a leaf will output one constant number, MSE-minimization forces that constant to be the mean, and picking splits that reduce MSE the most directly minimizes the tree's total prediction error on the training set.

**Worked numerical — regression split:**
Node has target values: [10, 12, 14, 20, 22, 24] (n=6)
$\bar{y} = (10+12+14+20+22+24)/6 = 102/6 = 17$

$$
\text{MSE}(t) = \frac{1}{6}[(10-17)^2+(12-17)^2+(14-17)^2+(20-17)^2+(22-17)^2+(24-17)^2]
$$
$$
= \frac{1}{6}[49+25+9+9+25+49] = \frac{166}{6} = 27.67
$$

Candidate split: left = [10,12,14] (mean=12), right = [20,22,24] (mean=22)

$$
\text{MSE}(t_L) = \frac{1}{3}[(10-12)^2+(12-12)^2+(14-12)^2] = \frac{1}{3}[4+0+4]=2.67
$$
$$
\text{MSE}(t_R) = \frac{1}{3}[(20-22)^2+(22-22)^2+(24-22)^2] = \frac{1}{3}[4+0+4]=2.67
$$

$$
\Delta = 27.67 - \left(\frac{3}{6}(2.67) + \frac{3}{6}(2.67)\right) = 27.67 - 2.67 = 25.0
$$

Huge reduction — makes sense, this split perfectly separates the "low" cluster from the "high" cluster.

**Why not MAE (mean absolute error) instead of MSE for regression trees?**
You *can* use MAE — sklearn supports `criterion='absolute_error'`. The trade-offs:
- MSE is differentiable and has a closed-form optimal leaf value (the mean), making the split search fast — for any candidate split you already know the optimal prediction with one division.
- MAE's optimal leaf value is the **median**, which requires sorting (or a selection algorithm) per candidate split — this is why `absolute_error` in sklearn is noted as significantly slower.
- MSE penalizes large errors quadratically, so it's more sensitive to outliers; MAE is robust to outliers since it penalizes linearly.
- **Why not always use MAE then, if it's more robust?** Because the speed cost is real (roughly $O(n \log n)$ extra per split from sorting/median-finding versus $O(1)$ incremental updates for mean/variance), and most regression targets don't have pathological outliers that justify paying it — so MSE remains the default and MAE is an opt-in for known-outlier-heavy targets.

---

## 1.3 How the Algorithm Actually Searches for the Best Split (Feature Selection for Splitting)

This is the mechanical heart of tree-building — worth walking through explicitly since "which feature does it split on and why" is a common interview probe.

**The search procedure at every node:**

1. For **each feature** $j = 1, \dots, p$:
   - If $j$ is continuous: sort the node's samples by their value of feature $j$. Candidate thresholds are the midpoints between consecutive *distinct* sorted values (you never need to test a threshold *within* a block of identical values — it would just move a class-neutral empty gap, not actually reassign any sample). For $n$ distinct values this gives at most $n-1$ candidate thresholds.
   - If $j$ is categorical (see 1.6 for full treatment): candidate splits are subsets of categories, or in sklearn's implementation, categories sorted by a proxy statistic (e.g. mean target for regression) and split like a continuous variable.
2. For **each candidate threshold** of feature $j$: compute the impurity reduction (Gini/entropy/MSE delta from 1.2) that this specific (feature, threshold) pair would produce.
3. Track the single best (feature, threshold) pair across **all features and all their thresholds** — this is a full sweep, not a random sample of features (that randomization only enters in Random Forests, Chapter 4).
4. That global best pair becomes the split at this node.

**Why sort first?** Because impurity for a threshold split can be computed incrementally as you sweep the sorted values left-to-right — you're just moving one sample at a time from the "right" bucket to the "left" bucket and updating running sums (of class counts for Gini/entropy, or of $\sum y$ and $\sum y^2$ for MSE) in $O(1)$ per step. This turns an apparently $O(n^2)$ problem (try every pair of thresholds naively) into $O(n \log n)$ dominated by the sort. Across all $p$ features this makes one node's split search $O(p \cdot n \log n)$.

**Why does this matter for "why not test thresholds anywhere, not just midpoints"?** Because impurity as a function of threshold only *changes value* at points where a sample crosses from one side to the other — between any two candidate thresholds that don't cross a data point, the resulting impurity is identical. Testing more finely than "one threshold between each pair of adjacent sorted values" wastes computation without ever finding a better split.

**Multiway vs binary splits — why CART is strictly binary:** ID3/C4.5 (Section 1.3-adjacent, detailed in 1.3.1 below) allow a categorical feature to split into as many branches as it has categories in one shot. CART restricts every split to exactly two children, even for categorical features (by finding the best partition of categories into two groups). Binary-only splitting is preferred because multiway splits fragment the data fast — a categorical feature with 20 levels splitting into 20 branches immediately can shatter a modestly-sized dataset into tiny, statistically unreliable leaves after a single split, and it also biases the impurity criterion toward high-cardinality features (more branches mechanically produces a bigger raw impurity reduction even when the feature isn't more informative — an inflation, not real signal). Binary splits let the tree revisit the same feature at multiple depths if useful, achieving the same expressive power more gradually and robustly.

---

## 1.3.1 CART vs ID3 vs C4.5 — what actually differs

| | ID3 | C4.5 | CART |
|---|---|---|---|
| Splitting criterion | Information Gain | Gain **Ratio** (IG normalized by split's own entropy, to correct IG's bias toward high-cardinality features) | Gini (classification) or MSE (regression) |
| Split arity | Multiway (one branch per category) | Multiway for categorical | Strictly binary |
| Handles continuous features? | No (must be pre-discretized) | Yes (thresholding, same midpoint-sweep idea as 1.3) | Yes |
| Handles missing values? | No | Yes (probabilistic splitting of missing samples across branches) | Yes (via surrogate splits, 1.6) |
| Pruning | None (prone to overfitting) | Post-pruning (error-based pruning) | Cost-complexity pruning (1.4) |
| Supports regression? | No | No | Yes |
| What sklearn implements | — | — | **CART** (an optimized version) |

**Why does IG bias toward high-cardinality features, and why does Gain Ratio fix it?**
Information Gain rewards a split for reducing entropy, but a feature with many unique values (e.g., an ID column) can trivially create many small, pure-ish partitions, producing large IG purely from fragmentation rather than genuine predictive signal — in the extreme, splitting on a row-unique ID gives one sample per leaf, IG maximized, zero generalization value. Gain Ratio divides IG by the **intrinsic information** of the split itself — literally the entropy of the branch-size distribution:

$$
\text{GainRatio}(S) = \frac{IG(S)}{\text{SplitInfo}(S)}, \quad \text{SplitInfo}(S) = -\sum_{i} \frac{|t_i|}{|t|}\log_2\frac{|t_i|}{|t|}
$$

A split into many small branches has high SplitInfo (large denominator), which discounts its raw IG back down — this is the same "fragmentation penalty" that binary-only CART splits sidestep structurally rather than correcting for numerically.

**Why does sklearn only implement CART?** Binary splits are simpler to optimize computationally (the $O(n\log n)$ sweep in 1.3 assumes binary), CART natively handles both classification and regression with one unified framework, and CART's cost-complexity pruning is a cleaner, more theoretically grounded regularization mechanism than C4.5's heuristic pruning. ID3/C4.5 are taught for historical/conceptual reasons but essentially no production library uses them today.

---

## 1.4 Stopping Rules & Pruning

If left unconstrained, recursive splitting continues until every leaf is pure (or has 1 sample) — this **always** achieves zero training error and is a textbook case of overfitting (memorizing noise, not signal). Two families of fixes:

### 1.4.1 Pre-pruning (early stopping) — stop growth before it happens

Constraints applied *during* growth so the tree never gets big enough to overfit:
- Max depth
- Minimum samples required to split a node
- Minimum samples required in a leaf
- Minimum impurity decrease required to accept a split (if the best available split doesn't clear this bar, stop)
- Maximum number of leaf nodes

**Why pre-pruning can be shortsighted:** it's still greedy — a split that looks useless *right now* (small impurity decrease) might be a necessary stepping stone to a very good split *one level deeper*. Pre-pruning can't see that and will cut off the branch before it gets the chance. This is the classic "horizon effect."

### 1.4.2 Post-pruning — grow fully, then prune back

Grow the tree to full depth (maximum overfit), then remove subtrees that don't earn their complexity. CART's method is **Cost-Complexity Pruning (Minimal Cost-Complexity Pruning, "weakest link" pruning)**.

Define, for any subtree $T$ of the fully-grown tree $T_0$:

$$
R_\alpha(T) = R(T) + \alpha \cdot |T|
$$

- $R(T)$ = total misclassification error (or MSE, for regression) of the tree over the training set
- $|T|$ = number of leaf (terminal) nodes — the complexity penalty
- $\alpha \geq 0$ = complexity parameter (`ccp_alpha` in sklearn) controlling the trade-off

**Why this specific penalty form (linear in leaf count)?** It's the direct tree analogue of L1/L2 penalties in linear models — a knob that trades training fit against model complexity, where "complexity" for a tree is naturally measured by how many independent constant-prediction regions it carves out. As $\alpha \to 0$, $R_\alpha \to R(T_0)$, the fully grown tree wins (no penalty for size). As $\alpha \to \infty$, the penalty dominates and the single-node "just predict the overall mean/majority class" tree wins.

**The algorithm:** for a fixed $\alpha$, find the subtree $T_\alpha$ that minimizes $R_\alpha(T)$. It's proven (Breiman et al., 1984) that as $\alpha$ increases from 0, there's a **nested sequence** of optimal subtrees $T_0 \supset T_1 \supset T_2 \supset \dots \supset \{root\}$ — you don't need to search all possible subtrees, just this one path, found efficiently via "weakest link" removal: at each step, remove the subtree whose removal increases $R(T)$ the least per leaf removed (i.e., the node with smallest **effective alpha**, $\alpha_{\text{eff}} = \frac{R(\text{pruned to leaf}) - R(\text{subtree})}{|T_{\text{subtree}}| - 1}$).

**Worked numerical — cost-complexity pruning:**
Suppose a subtree $T$ has 4 leaves with training misclassification error $R(T) = 0.10$ (10% error). If we collapse that subtree to a single leaf, error becomes $R(\text{leaf}) = 0.20$.

$$
\alpha_{\text{eff}} = \frac{0.20 - 0.10}{4 - 1} = \frac{0.10}{3} = 0.0333
$$

This means: for any $\alpha < 0.0333$, keeping the 4-leaf subtree is worth it (the error reduction per added leaf beats the penalty). For any $\alpha > 0.0333$, collapsing to 1 leaf is better. sklearn's `cost_complexity_pruning_path()` computes this $\alpha_{\text{eff}}$ for every prunable subtree, giving you the full sequence of $(\alpha, \text{tree})$ pairs; you then pick the $\alpha$ (hence the tree) that performs best on a held-out validation set or via cross-validation.

**Why prefer post-pruning over pre-pruning when compute allows?** Post-pruning sees the *fully realized* value of every subtree before deciding what to cut, so it doesn't suffer from the horizon effect described in 1.4.1. The trade-off is pure computational cost — growing a full tree is more expensive than stopping early, and for very large datasets/many trees (as in Random Forests, Chapter 4) pre-pruning limits are often used anyway simply for tractability, with the ensemble's averaging providing regularization instead of per-tree pruning.

---

## 1.5 Bias–Variance Profile of a Single Tree

An unpruned, fully-grown decision tree is the textbook **high-variance, low-bias** learner:

- **Low bias**: with enough depth, a tree can carve out arbitrarily complex, non-linear decision boundaries / regression surfaces — it can fit the training data's true signal (and its noise) almost exactly. It makes no strong parametric assumption about the relationship between features and target (unlike, say, linear regression's assumption of linearity).
- **High variance**: small changes in the training data (a few different samples due to resampling) can lead to a *completely different* sequence of greedy splits near the root, which cascades into a structurally very different tree. This instability is precisely *why* bagging (Chapter 3) is such a good match for trees — bagging's variance reduction is most effective when the base learner is unstable/high-variance, and averaging many differently-grown trees smooths out this instability while barely touching bias.

**Why are trees specifically unstable, more so than say k-NN?** Because splits are chosen greedily and hierarchically — an early split decision constrains and reshapes the entire subtree below it. A different bootstrap sample can easily flip which feature "wins" the very first split (especially when two features have close impurity-reduction scores), and that single flip propagates completely differently for the rest of the tree. This is formally why trees have famously high variance relative to their bias, and it's the direct motivation for Chapter 3 (Bagging) and Chapter 4 (Random Forests) — decorrelating many high-variance trees and averaging them is one of the most effective variance-reduction techniques in classical ML, precisely because trees supply so much variance to reduce.

Shallow/pruned trees shift the other way — as depth shrinks or `ccp_alpha`/leaf-size constraints tighten, bias rises (the tree can't express complex boundaries) and variance falls (fewer, more heavily-populated leaves means less sensitivity to which exact samples landed where). This is the direct tree-specific instance of the general model-complexity U-curve.

---

## 1.6 Handling Categorical Features, Missing Values, Surrogate Splits

**Categorical features.** For a categorical feature with categories $\{c_1,\dots,c_q\}$, an *exhaustive* binary split search would need to try all $2^{q-1}-1$ ways to partition the categories into two groups — exponential, infeasible for even modest $q$. CART uses a shortcut proven optimal for binary classification with Gini impurity (Breiman et al., 1984): sort the categories by the proportion of the positive class (classification) or by mean target value (regression) within each category, then treat this sorted order exactly like a continuous variable and sweep thresholds through it (same $O(n \log n)$ machinery as 1.3). This finds the *globally optimal* binary partition for the 2-class case in $O(n \log n)$ instead of exponential time; for multiclass, it's a good heuristic approximation, not a proven optimum.

**Missing values — surrogate splits.** CART's answer: when the *primary* split's feature is missing for some sample, fall back to a **surrogate split** — a backup (feature, threshold) rule chosen (at training time) to mimic the primary split's sample assignment as closely as possible on the samples that *do* have the primary feature observed. Multiple surrogates are ranked, so if surrogate 1's feature is also missing, fall to surrogate 2, and so on; if all surrogates are missing, fall back to the majority direction. **Why bother with this instead of just imputing beforehand?** Surrogates let the *tree itself* learn context-specific stand-ins per split (a surrogate near the root may differ completely from one deep in a specific branch), which can capture missingness patterns imputation would flatten — though in practice, many modern workflows just impute upstream and skip surrogates entirely because they're expensive to compute and sklearn's `DecisionTreeClassifier`/`Regressor` **do not implement surrogate splits at all** — sklearn requires you to impute missing values yourself before fitting (unlike XGBoost/LightGBM, which have native missing-value handling, previewed in Chapter 5 and detailed in 9.3).

---

## 1.7 Decision Boundary Geometry & Interpretability

Because every split is a threshold on a single feature, every decision boundary a tree can draw is a union of **axis-aligned hyperplane segments** — in 2D, boundaries are always horizontal/vertical line segments, never diagonal. A diagonal true boundary (e.g., $x_1 = x_2$) can only be *approximated* by a staircase of many small axis-aligned steps, which costs many splits (deep tree) to approximate well — a structural weakness relative to models like SVM with a linear kernel, or logistic regression, which can represent an arbitrary linear (diagonal) boundary with a single decision surface.

**Interpretability trade-off, precisely:** a shallow tree (depth 3–4) is one of the few ML models a non-technical stakeholder can read directly as a flowchart of if/else rules — this is the tree's biggest practical selling point over black-box models, and is a running theme in "when to use trees" (Chapter 8). That interpretability degrades quickly with depth (a depth-10 tree has up to $2^{10}$ leaves, i.e. up to 1024 distinct if/else paths — no longer humanly readable) which is part of why ensembles (which sacrifice this interpretability for accuracy, then need SHAP, Chapter 9, to get it back) exist.

---

## sklearn Parameters — `DecisionTreeClassifier` / `DecisionTreeRegressor`

| Parameter | What it controls | Notes / why it matters |
|---|---|---|
| `criterion` | Split quality measure | Classifier: `'gini'` (default), `'entropy'`, `'log_loss'`. Regressor: `'squared_error'` (default, = MSE, Section 1.2.4), `'friedman_mse'` (a variant used more in gradient boosting, Ch. 5), `'absolute_error'` (MAE, slower — Section 1.2.4), `'poisson'` (for count targets) |
| `splitter` | How the split search picks among candidates | `'best'` (default, exhaustive search per 1.3) vs `'random'` (only tries a random subset of thresholds per feature — faster, adds variance, rarely used outside of `ExtraTreesClassifier`-style ensembles) |
| `max_depth` | Maximum tree depth (pre-pruning) | `None` (default) = grow until leaves are pure or hit `min_samples_split`. The single most direct bias/variance dial. |
| `min_samples_split` | Minimum samples a node needs to be eligible to split at all | Default 2. Raising this is a mild pre-pruning constraint. |
| `min_samples_leaf` | Minimum samples required in **each** resulting leaf for a split to be accepted | Default 1. Often more effective than `min_samples_split` at smoothing regression trees, since it directly bounds how few points a leaf's prediction can be based on. |
| `min_weight_fraction_leaf` | Same as `min_samples_leaf` but as a fraction of total (weighted) samples | Useful when using `sample_weight` |
| `max_features` | Number of features considered **at each split** | Default `None` = all features considered (full exhaustive search, Section 1.3). Setting this < total features introduces per-split randomness — this parameter is barely used in a *standalone* tree, but is the exact mechanism Random Forests (Ch. 4) tune to decorrelate trees. |
| `max_leaf_nodes` | Cap on total leaves | `None` (default) = unlimited. Alternative pre-pruning lever to `max_depth`; grows tree in best-first (highest impurity-reduction-first) order rather than depth-first when this is set. |
| `min_impurity_decrease` | Minimum impurity reduction a split must achieve to be accepted | Default 0.0. Direct pre-pruning threshold on the $\Delta$ computed in 1.2. |
| `ccp_alpha` | Complexity parameter for post-pruning | Default 0.0 (no pruning). Set via `cost_complexity_pruning_path()`, Section 1.4.2. |
| `class_weight` | Reweights classes (classification only) | `None`, `'balanced'`, or explicit dict — adjusts impurity computation to counter class imbalance |
| `random_state` | Seed | Matters when `splitter='random'` or `max_features < p` (ties among equally-good splits are also broken randomly, so this affects reproducibility even in default settings when ties occur) |

**Why does `min_samples_leaf` often matter more than `max_depth` in practice?** `max_depth` limits depth uniformly across the whole tree, but real data is often uneven — some regions have plenty of samples and could support more splits, others are already sparse. `min_samples_leaf` adapts locally: it lets the tree keep splitting in dense regions while automatically halting in sparse ones, which tends to produce a better bias/variance trade-off than a single global depth cap.

---

## Quick Interview Q&A

**Q: Why do decision trees need pruning but linear regression doesn't need an analogous structural pruning step?**
A: A tree's capacity (number of leaves) grows with the data structure itself — unconstrained, it will keep splitting until every leaf is pure, so its effective complexity is unbounded and directly tied to how deep you let it grow. Linear regression's capacity is fixed by the number of features you chose at the start; it can't spontaneously add more "regions." Regularizing linear regression (L1/L2) shrinks *coefficients*; regularizing a tree constrains its *structure* — different failure mode, different fix.

**Q: If Gini and Entropy usually agree, why does the choice ever come up in interviews?**
A: Because interviewers are testing whether you understand *what* each formula measures (expected misclassification rate under random guessing, vs. expected bits of surprise) rather than testing whether you have a strong practical opinion — the honest, correct answer includes "in practice this rarely changes results much; Gini is preferred mainly for its lower compute cost."

**Q: Why can't you just always grow the tree fully and rely on `ccp_alpha` alone, skipping `max_depth`/`min_samples_leaf` entirely?**
A: You can, and it's often the most *principled* approach — but growing to full, unconstrained depth on large datasets can be computationally expensive and memory-heavy just to reach the point where you then throw most of it away. Pre-pruning limits are frequently used as practical guardrails to keep training tractable, especially inside ensembles (Chapters 3–5) where you're growing hundreds of trees and can't afford full-depth growth + full pruning-path search on each one.

---

**Next up: Chapter 2 — Ensemble Foundations (why averaging/boosting works, bias-variance decomposition for ensembles) before Chapter 3 goes deep on Bagging.** Want me to continue straight into that, or do you want a numerical-heavy practice set on this chapter first?
