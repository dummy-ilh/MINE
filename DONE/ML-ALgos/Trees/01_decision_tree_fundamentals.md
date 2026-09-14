# Decision Tree Fundamentals — Master Notes

## 1. The Big Idea (in one paragraph)

A decision tree predicts by asking a series of yes/no questions about your data, one at a time, until it lands on an answer. "Is size > 1,800 sq ft?" → yes → "Is location = downtown?" → yes → predict $450,000. Every path of questions ends in a **leaf**, and every sample walks down exactly one path to exactly one leaf. Geometrically, this carves feature space into rectangular boxes, with one prediction per box:

$$\hat{f}(x) = \sum_{m=1}^{M} c_m \cdot \mathbb{1}(x \in R_m)$$

where $c_m$ is the majority class (classification) or mean target value (regression) of region $R_m$.

---

## 2. How a Tree Is Built

1. Start with everything in one group (the root).
2. Try out every feature and every reasonable threshold as a candidate yes/no question.
3. Pick whichever single question best separates the group into two groups that are each "cleaner" (more similar to each other) than the group was before.
4. Repeat separately on each new group.
5. Stop when a stopping rule is hit (Section 5).

**Why "greedy"?** At every step the tree just picks whatever question looks best *right now* — it never checks whether a slightly worse question now might set up a much better one later. It's like solving a maze by always taking whichever path looks most promising at each fork, rather than planning the whole route in advance. Planning the whole tree ahead of time to find the *provably best* tree is computationally intractable (an NP-complete problem — Hyafil & Rivest, 1976), so greedy, one-split-at-a-time search is the practical trade-off every real tree library makes.

---

## 3. Picking the Best Question — Splitting Criteria

You need a number for "how messy/mixed-up is this group" — then pick whichever question makes the two resulting groups the least messy on average.

### 3.1 Gini Impurity (classification)

**Plain meaning:** if you picked one sample at random from the group, and guessed its label using a weighted die based on the group's own label mix, how often would you be wrong?

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

**Numerical:** 6 expensive houses, 4 cheap → $p=0.6, 0.4$
$$\text{Gini} = 1-(0.36+0.16) = 0.48$$

- All one label → Gini = 0 (best, purest).
- 50/50 split (2 classes) → Gini = 0.5 (worst, messiest possible for 2 classes).

### 3.2 Entropy / Information Gain (classification)

**Plain meaning:** the expected number of yes/no questions ("bits") you'd need, on average, to correctly guess a random sample's label from this group.

$$\text{Entropy}(t) = -\sum_k p_k \log_2 p_k$$

- All one label → entropy 0 (already know the answer).
- 50/50 split → entropy 1 (exactly a coin flip's worth of uncertainty).

**Information Gain** = messiness before the split minus the (size-weighted) average messiness of the two groups after. Bigger gain = better question.

**Numerical (6A/4B parent, split into 4A/1B left and 2A/3B right, 5 each):**
- Parent entropy ≈ 0.971
- Left entropy ≈ 0.722, Right entropy ≈ 0.971 (0.4/0.6 mix has same entropy as 0.6/0.4)
- Weighted average = $0.5(0.722)+0.5(0.971) = 0.847$
- $IG = 0.971 - 0.847 = 0.125$ bits

### 3.3 Gini vs. Entropy — does it matter?

| | Gini | Entropy |
|---|---|---|
| Formula | $1-\sum p_k^2$ | $-\sum p_k\log_2 p_k$ |
| Compute cost | Cheaper (no log) | Slightly pricier |
| Range (binary) | [0, 0.5] | [0, 1] |
| Agreement in practice | Same split chosen >95% of the time (Raileanu & Stoffel, 2004) | — |

**Honest answer:** it barely matters. sklearn defaults to Gini purely because it's cheaper to compute — no theoretical reason to strictly prefer one. A third option, **misclassification error** ($1-\max_k p_k$), is almost never used for *growing* trees because it's insensitive to how the rest of the distribution is shaped — it can score two very different splits identically where Gini/entropy clearly prefer one.

### 3.4 Variance Reduction / MSE (regression)

**Plain meaning:** "messiness" of a group = how spread out its target values are (variance). Pick the question that leaves each resulting group more tightly clustered than before.

$$\text{MSE}(t) = \frac{1}{n_t}\sum_{i\in t}(y_i-\bar y_t)^2$$

**Numerical:** group = [10,12,14,20,22,24], mean=17, MSE(parent) = 27.67.
Split into [10,12,14] (mean 12, MSE 2.67) and [20,22,24] (mean 22, MSE 2.67):
$$\Delta = 27.67 - \left(\tfrac{1}{2}(2.67)+\tfrac{1}{2}(2.67)\right) = 27.67-2.67 = 25.0$$
Big reduction — this split perfectly separates the low cluster from the high cluster.

**Why the mean?** The mean is the single number that minimizes total squared distance to every value in a group — so once a leaf must output one flat number, MSE-minimization forces that number to be the mean.

**MSE vs. MAE:** MAE (mean absolute error) is more robust to outliers (one $5M mansion won't drag it around the way it drags a mean), but its optimal leaf value is the *median*, which needs sorting per candidate split — meaningfully slower than MSE's $O(1)$ running-sum updates. MSE stays the default; MAE is opt-in for known outlier-heavy targets.

---

## 4. How the Search Actually Works

**Plain version:** for every feature, and every reasonable threshold within that feature, check how good that yes/no question would be. Keep the single best (feature, threshold) combo across everything tried.

**The speed shortcut:** you never need to test every possible number as a threshold — only the midpoints between *actual, distinct* values that appear in the data. For $n$ distinct values that's at most $n-1$ candidates. Nothing between two adjacent data points can change which samples fall left vs. right, so testing more finely than that wastes computation for zero benefit.

**Why this is fast:** if you first *sort* samples by the feature's value, you can sweep the threshold left to right, moving one sample at a time between "left" and "right" buckets and updating running counts/sums in $O(1)$ per step — instead of recomputing impurity from scratch at every threshold. This turns the search into $O(n\log n)$ (dominated by the sort) per feature, so $O(p \cdot n \log n)$ per node across all $p$ features.

**Why always yes/no (binary), never "pick one of five"?** A multi-way split on a high-cardinality category (e.g., 20 neighborhoods) can shatter the data into 20 tiny, unreliable groups in one step, and it mechanically favors features with more categories — more branches means more chances to look good by luck, not real signal. Binary splits are safer and just as powerful, since the tree can revisit the same feature again at a deeper level if it needs to.

### 4.1 CART vs. ID3 vs. C4.5

| | ID3 | C4.5 | CART (what sklearn uses) |
|---|---|---|---|
| Criterion | Information Gain | Gain Ratio (IG ÷ a penalty for many branches) | Gini / MSE |
| Split type | Multiway | Multiway (categorical) | Always binary |
| Continuous features | ✗ (must pre-bucket) | ✓ | ✓ |
| Missing values | ✗ | ✓ | ✓ (surrogate splits, in theory) |
| Pruning | None | Post-pruning | Cost-complexity pruning |
| Regression? | ✗ | ✗ | ✓ |

**Why did Gain Ratio need to exist?** Raw Information Gain rewards fragmentation — a row-unique ID column can look "maximally informative" purely because it splits everyone into their own tiny leaf, not because it's genuinely predictive. Gain Ratio divides IG by the "spread-out-ness" of the split itself, discounting splits that just fragment the data into many small pieces. CART sidesteps the whole problem structurally, by only ever allowing binary splits in the first place.

**Why does sklearn only implement CART?** Binary splits are simpler to optimize (the $O(n\log n)$ sweep assumes two branches), CART handles classification and regression with one unified framework, and cost-complexity pruning is cleaner and more principled than C4.5's heuristics. ID3/C4.5 today are taught for history, not used in production.

---

## 5. When Does the Tree Stop Growing?

Left alone, a tree keeps splitting until every leaf is pure — that's **guaranteed** zero training error and textbook overfitting (memorizing noise, not signal).

### 5.1 Pre-pruning (stop early)

Rules applied *during* growth: max depth, minimum samples to split, minimum samples per leaf, minimum impurity decrease required, max number of leaves.

**Weakness — the horizon effect:** a question that looks useless right now (tiny impurity decrease) might be the necessary stepping stone to a great question one level deeper. Pre-pruning can't see that far ahead and cuts the branch too early.

### 5.2 Post-pruning (grow fully, then trim)

Grow to full depth (max overfit), then remove branches that don't earn their complexity. CART's method: **cost-complexity pruning**.

$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

- $R(T)$ = training error of subtree $T$
- $|T|$ = number of leaves (the complexity penalty)
- $\alpha$ = the price you're willing to pay per extra leaf (`ccp_alpha` in sklearn)

As $\alpha\to 0$, the fully-grown tree wins. As $\alpha\to\infty$, the single-leaf "just predict the overall average" tree wins. There's a proven nested sequence of optimal subtrees as $\alpha$ increases, so sklearn can compute the *entire* pruning path in one pass — no need to search all possible subtrees.

**Numerical:** a 4-leaf subtree has training error 10%; collapsing it to 1 leaf raises error to 20%.
$$\alpha_{\text{eff}} = \frac{0.20-0.10}{4-1} = 0.033$$
Below this $\alpha$, keep the branch (the accuracy-per-leaf trade is worth it). Above it, cut.

**Pre- vs. post-pruning trade-off:** post-pruning sees the fully realized value of every branch before deciding what to cut, so it avoids the horizon effect — but it costs more compute (you have to grow the whole tree first). For very large datasets or hundreds of trees (Random Forests, later chapters), pre-pruning limits are often used anyway just for tractability, and the ensemble's averaging supplies regularization instead.

---

## 6. Why a Single Tree Is Unstable (sets up Bagging)

A fully grown tree is the textbook **high-variance, low-bias** learner:

- **Low bias** — given enough depth, it can carve out almost any decision boundary, fitting the true signal (and the noise) closely. No assumption of linearity, unlike linear regression.
- **High variance** — swap a handful of training rows for different ones, and the very first split can flip entirely, which cascades into a structurally different tree below it.

**Why so sensitive?** Every later question depends completely on the question before it — change the top split, and everything built underneath changes too. A different training sample can easily flip which feature "wins" a close call at the root, and that single flip ripples through the whole tree.

**Why this matters:** this exact instability is the raw material bagging and Random Forests exploit later — averaging many differently-wobbly trees smooths out each one's individual wobble, and it only works this well *because* a single tree wobbles this much to begin with.

---

## 7. Categorical Features & Missing Values

**High-cardinality categories:** checking every possible way to split many categories into two groups is exponential (infeasible). CART's shortcut: sort categories by their average target value (or positive-class proportion), then treat the sorted order like a numeric feature and sweep thresholds through it — same $O(n\log n)$ machinery as Section 4. This is *provably optimal* for binary classification with Gini; for multiclass it's a strong heuristic, not a proven optimum.

**Missing values:** classic CART theory has "surrogate splits" — backup questions used when a sample is missing the primary feature. **In practice, sklearn's trees don't implement this at all** — you must impute missing values yourself before fitting. XGBoost/LightGBM handle missing values natively; this is a real practical advantage of those libraries over plain sklearn trees.

---

## 8. What Shapes Can a Tree's Boundary Take?

Every split is "is this one number above/below a threshold," so every boundary a tree draws is made of horizontal/vertical steps — never diagonal. A truly diagonal boundary (e.g., $x_1 = x_2$) can only be approximated by a staircase of many small steps, costing a lot of splits (depth) to get close.

**Interpretability trade-off:** a shallow tree (depth 3–4) is one of the few models a non-technical person can read directly as an if/else flowchart — the tree's biggest practical selling point over black-box models. That readability collapses fast with depth — a depth-10 tree has up to $2^{10}=1024$ possible paths, no longer humanly readable. This is part of why ensembles (which sacrifice this and need SHAP-style tools to explain themselves) exist.

---

## 9. sklearn Parameter Cheat Sheet

| Parameter | Plain meaning | Default |
|---|---|---|
| `criterion` | Which messiness score to use | `'gini'` (classifier) / `'squared_error'` (regressor) |
| `splitter` | Exhaustive (`'best'`) vs. random-subset threshold search | `'best'` |
| `max_depth` | How many questions deep the tree can go | `None` (unlimited) |
| `min_samples_split` | Smallest group allowed to ask a new question | 2 |
| `min_samples_leaf` | Smallest group allowed in a final leaf | 1 |
| `max_features` | Features considered *per split* | `None` (all — becomes the key Random Forest knob later) |
| `max_leaf_nodes` | Cap on total leaves | `None` |
| `min_impurity_decrease` | Minimum improvement a split must deliver to be accepted | 0.0 |
| `ccp_alpha` | Cost-per-leaf knob for post-pruning | 0.0 (no pruning) |
| `class_weight` | Upweight rarer classes | `None` |
| `random_state` | Reproducibility (matters for tie-breaking, `splitter='random'`, or `max_features<p`) | — |

**`min_samples_leaf` vs. `max_depth`:** `max_depth` applies one flat limit everywhere. `min_samples_leaf` adapts locally — it lets the tree keep splitting in data-rich regions while automatically stopping in sparse ones, usually giving a better bias/variance trade-off than a single global depth cap.

---

## 10. Quick Q&A (general)

**Q: Why does a tree need pruning but linear regression doesn't need an equivalent step?**
A: A tree's complexity (leaf count) grows on its own as splitting continues — unconstrained, it's unbounded. Linear regression's complexity is fixed at the start by however many features you gave it; it can't spontaneously add more "regions." Regularizing linear regression shrinks *coefficients*; regularizing a tree constrains its *structure* — different failure mode, different fix.

**Q: Gini or entropy — does the choice actually matter?**
A: Barely. They agree on the chosen split the large majority of the time. Gini is cheaper to compute (no logs), which is the real reason it's the default — not some deep statistical advantage.

**Q: Why not always grow fully and rely only on `ccp_alpha`, skipping `max_depth`/`min_samples_leaf`?**
A: You can, and it's often more principled — but growing to full depth on large datasets (or across hundreds of trees in an ensemble) is expensive just to then discard most of it via pruning. Pre-pruning limits are common practical guardrails to keep training tractable.

---

## 11. Google MLE Interview Q&A

**Q: You're building a tree on a dataset with a user-ID-like column that has near-unique values per row. What happens if you don't exclude it, and how does the splitting criterion let this happen?**
A: A near-unique ID column can produce splits that look extremely informative — Information Gain in particular can be maximized by fragmenting into many tiny, pure-ish leaves, since IG doesn't penalize the *number* of resulting groups, only how pure each one becomes. Gini has the same underlying vulnerability, just expressed less dramatically since CART is binary-only (it can't fragment into 20 branches in one step the way multiway ID3 could, but repeated binary splits on the same high-cardinality feature can still overfit to it across depth). Practically: this is a pure memorization split with zero generalization value, and it's exactly the scenario Gain Ratio (C4.5) and feature-importance sanity checks are designed to catch.

**Q: You need to explain, in a system design interview, why greedy tree induction is $O(p \cdot n \log n)$ per node rather than something worse. Walk through it.**
A: For a single feature, sorting the node's $n$ samples costs $O(n\log n)$; after that, sweeping threshold candidates left-to-right lets you update the impurity's running statistics (class counts, or $\sum y$/$\sum y^2$) in $O(1)$ per step rather than recomputing from scratch, so the threshold search itself is $O(n)$ once sorted. Repeating this per feature gives $O(p\cdot n\log n)$ for one node's full split search. Naively recomputing impurity from scratch for every candidate threshold without the running-sum trick would cost $O(n)$ per candidate times up to $O(n)$ candidates times $p$ features — $O(p\cdot n^2)$ — so the sort-and-sweep trick is what keeps tree induction practical at scale.

**Q: Design question — you're training decision trees as part of a pipeline processing hundreds of millions of rows, and induction is too slow. What's the first thing in the splitting algorithm you'd look at optimizing, and why?**
A: The per-node sort dominates the node's cost ($O(n\log n)$ vs. $O(n)$ for the sweep), so for continuous features the biggest lever is avoiding a full re-sort at every node — e.g., pre-sorting each feature once at the root and maintaining sorted order incrementally as data is partitioned into children (the "pre-sorted" approach used by some histogram-based/gradient-boosting implementations), or switching to a histogram/binning approximation of thresholds (bucket continuous values into a fixed number of bins once, then search over bins instead of every distinct value) — trading a small amount of split-quality precision for a large constant-factor speedup, which is exactly the trade-off libraries like LightGBM make by default.

---

## 12. Apple MLE Interview Q&A (on-device / practical flavor)

**Q: You're deploying a single decision tree on-device (e.g., in Core ML) for a fast, interpretable prediction. What tree properties from this chapter matter most for that deployment, and why?**
A: Depth and leaf count matter directly — inference cost and memory footprint scale with how many if/else comparisons a prediction has to walk through and how many leaf values need to be stored, both of which grow with `max_depth`/`max_leaf_nodes`. A shallow, heavily pre- or post-pruned tree is attractive on-device for two independent reasons at once: it's cheap to run (few comparisons, tiny memory footprint) *and* it stays human-readable as an if/else flowchart — useful if the feature needs to be explainable to reviewers or users. That second property is specific to trees; it's part of why you might choose a single shallow tree on-device over a larger ensemble even when the ensemble would be more accurate.

**Q: sklearn's trees don't support surrogate splits for missing data — how would that constraint affect a pipeline where a feature can go missing at inference time (e.g., a sensor reading unavailable on a given device)?**
A: Since sklearn can't fall back to a backup question the way full CART theory allows, missingness has to be handled *before* the tree ever sees the data — typically by imputing with a fixed rule (a sentinel value, a learned default, or a separate "missingness" indicator feature) that's identical at training and inference time. The risk is a training/serving mismatch: if the imputation logic used when training differs even slightly from what runs on-device at inference (different default, different handling of a newly-introduced sensor gap), the tree will silently walk down the wrong branch with no error — worth flagging explicitly in a deployment review, since the tree itself gives no signal that anything went wrong.

**Q: A tree's decision boundary is always axis-aligned (never diagonal). Why would this matter for an on-device feature that combines two correlated sensor signals (e.g., accelerometer and gyroscope readings)?**
A: If the true decision boundary between two classes is diagonal in the space of those two raw signals (e.g., depends on their *ratio* or *sum* rather than either one alone), a tree can only approximate that diagonal with a staircase of many small axis-aligned splits — costing extra depth (and therefore extra on-device compute/memory) to get close, and still leaving a jagged, imperfect approximation. The practical fix is usually feature engineering upstream — precompute the combined signal (e.g., the ratio or magnitude) as its own feature before it reaches the tree — so the tree only needs one clean threshold on the engineered feature instead of many splits approximating a diagonal.

**Q: If you were choosing between a single well-pruned tree and a small ensemble for an on-device feature, what specific trade-off from this chapter would you bring up first?**
A: The bias-variance profile from Section 6 — a single tree is high-variance, so its predictions can be noticeably unstable depending on exactly what training data it saw, while pruning it down to reduce that variance directly increases its bias (it captures less of the true pattern). An ensemble fixes variance without touching bias, but at $M\times$ the on-device inference cost. So the real question isn't "which is more accurate in a vacuum," it's whether the specific feature's latency/memory budget can afford paying for variance reduction via more trees, or whether it has to buy stability the cheaper way — via pruning a single tree — and accept the higher bias that comes with it.

---

**One-line summary to remember:** *A tree = greedy, recursive yes/no splitting that minimizes impurity (Gini/entropy for classification, variance/MSE for regression) at each step → unconstrained growth always overfits, so pre-pruning (stop early) or post-pruning (cost-complexity, grow-then-trim) is required → a single tree is low-bias/high-variance and structurally unstable, which is exactly why it's the base learner ensembles (Bagging, Random Forest) are built around.*
