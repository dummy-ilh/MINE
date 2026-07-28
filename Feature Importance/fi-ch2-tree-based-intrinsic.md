# Chapter 2 — Intrinsic Importance: Tree-Based Models

Chapter 1 gave you the taxonomy. This chapter goes deep on the first row of that table — importance methods that only exist because of a tree's internal structure. You've seen MDI once before (if you did the Feature Selection syllabus); here we go further: a full derivation, a second variant, the boosted-tree-specific importance types, and a precise mechanistic account of *why* the cardinality bias happens, not just that it happens.

## 2.1 Mean Decrease in Impurity (MDI) — full derivation

**The building block: impurity at a node.** For a classification tree, the standard impurity measure is **Gini impurity**:

Gini(node) = 1 − Σₖ pₖ²

where pₖ is the proportion of examples at that node belonging to class k, summed over all classes. Intuitively: if a node is pure (all one class), one pₖ = 1 and the rest are 0, giving Gini = 1 − 1 = 0. If a node is maximally mixed between two classes (50/50), Gini = 1 − (0.5² + 0.5²) = 1 − 0.5 = 0.5 — the maximum possible value for a two-class problem.

**Impurity reduction from a single split.** When a node with N examples is split into a left child (N_L examples) and right child (N_R examples), the impurity reduction that split achieves is:

ΔGini = Gini(parent) − [(N_L/N) · Gini(left) + (N_R/N) · Gini(right)]

This is just "impurity before the split" minus "the (size-weighted) average impurity after the split" — a positive ΔGini means the split successfully separated the classes better than leaving them mixed.

**MDI importance for one feature, across a whole tree:** sum ΔGini over every split in the tree where that feature was used, weighting each split's ΔGini by the fraction of total training examples that passed through that node (N/N_total) — this weighting is what makes a split near the root (seen by nearly all the data) count for more than a split deep in the tree (seen by only a few examples). For a random forest, average this quantity across every tree in the forest. Finally, normalize across all features so the importances sum to 1.

**Regression trees:** the same idea, but impurity is typically variance of the target within the node rather than Gini — a split is good if it produces two children whose target values are each more tightly clustered (lower variance) than the mixed parent.

## 2.2 Mean Decrease in Accuracy — a lesser-known intrinsic variant

**The idea:** instead of summing impurity reductions (a training-set-only quantity, since impurity is computed purely from the training data used to grow the tree), Mean Decrease in Accuracy uses each tree's **out-of-bag (OOB) samples** — recall that in a random forest, each tree is trained on a bootstrap sample, leaving roughly a third of the training data unused ("out-of-bag") for that particular tree.

**The procedure:** for each tree, measure its prediction accuracy on its own OOB samples (a form of free, built-in validation set), then permute one feature's values within just those OOB samples (this is a permutation-importance-style shuffle, previewed here and covered in full generality in Chapter 4) and re-measure accuracy on the same OOB samples. The drop in OOB accuracy, averaged across all trees in the forest, is that feature's Mean Decrease in Accuracy.

**Why this is meaningfully different from MDI, despite sounding similar:** MDI is purely a training-time quantity — it only reflects how much impurity reduction a feature *appeared* to provide during tree construction, with no check against held-out data at all. Mean Decrease in Accuracy explicitly validates on OOB samples the tree never used to grow itself, making it considerably more resistant to MDI's overfitting-to-training-noise problem (previewed in §2.4) — it's conceptually a hybrid, sitting partway between MDI's "free but training-data-only" approach and Chapter 4's full permutation importance (which uses a fully separate held-out set rather than each tree's own OOB samples).

## 2.3 Boosted-tree importance types: gain, weight, cover

Gradient-boosted tree libraries (XGBoost, LightGBM) expose **three distinct built-in importance types**, and a common interview trap is not realizing these can produce meaningfully different rankings for the same trained model.

- **Gain:** the average improvement in the loss function (analogous to ΔGini/impurity reduction above, but generalized to whatever loss the boosted model is optimizing) contributed by splits using this feature, averaged over every split where it appears. This is the closest boosted-tree analogue to MDI, and is generally the most informative of the three for understanding predictive contribution.
- **Weight (sometimes called "frequency" or "split count"):** simply the *number of times* a feature is used to split across all trees, with no regard to how much impurity/loss reduction each split achieved. A feature used in many splits, each contributing only a tiny improvement, can score very high on "weight" while contributing little real predictive value — this measure is the most easily distorted by a feature simply having many available split points (the same cardinality-driven mechanism from §2.4, but even more directly, since it's counting raw split occurrences rather than impurity reduction).
- **Cover:** the average number of training examples affected by splits using this feature (summed across all its splits, then averaged). A feature that produces a small number of very impactful, high-coverage splits scores high on cover even if its "weight" (split count) is low.

**The practical consequence:** always specify *which* boosted-tree importance type you're using when reporting or interpreting results — "gain" is generally the safest default for understanding genuine predictive contribution, while "weight" specifically should be treated with real suspicion for high-cardinality features, since it inherits (in an even more direct form) the same bias mechanism as MDI.

## 2.4 The cardinality bias, mechanistically — not just "it happens"

Chapter 1 (implicitly) and prior material told you MDI is biased toward high-cardinality/continuous features. Here's precisely *why*, worked through mechanically.

**The mechanism, step by step:**
1. At every node, the tree-building algorithm searches over **every feature** and, for each feature, **every possible split threshold**, picking whichever (feature, threshold) combination maximizes impurity reduction on the training data available at that node.
2. A continuous feature, or a categorical feature with many distinct categories, offers **far more candidate thresholds to try** than a low-cardinality feature (e.g., a binary 0/1 flag offers exactly one possible split point; a continuous feature with 1,000 distinct training values offers up to 999 candidate thresholds).
3. With many more candidate thresholds to search over, the algorithm has many more "attempts" to stumble onto a threshold that happens to separate the *training* data favorably **purely due to sampling noise** — this is precisely the same "more attempts, more chances to look good by luck" phenomenon that appears in multiple-hypothesis-testing problems generally (closely related to why the wrapper-method search in feature selection risks overfitting its own validation set, and why wide hyperparameter searches risk finding a configuration that looks great on a validation set purely by chance).
4. Because MDI is computed **entirely from training-data impurity reductions**, with no held-out check at all, it has **no way to distinguish** "a split that reflects a real, generalizable pattern" from "a split that happened to separate the training data well purely because there were 999 chances to find one that did" — both look identical to MDI, since MDI never checks against data the split wasn't fit to.

**Why permutation importance (Chapter 4) and Mean Decrease in Accuracy (§2.2) are comparatively immune:** both explicitly re-measure performance on data the specific pattern wasn't fit to find (a separate validation set, or each tree's own OOB samples) — a threshold that only looked good due to training-sample luck will simply fail to hold up on that held-out check, correctly deflating its apparent importance. MDI has no equivalent check built in anywhere in its computation.

**Concrete synthetic demonstration to hold onto for an interview:** add a completely random, uninformative feature to your dataset, but give it high cardinality (say, 1,000 unique random values) rather than low cardinality (say, a single random 0/1 flag). Train a random forest and compute MDI for both random features. **The high-cardinality random feature will reliably receive a non-trivial, sometimes surprisingly large MDI score, while the low-cardinality random feature correctly scores near zero** — despite both being pure noise with zero true relationship to the target. This single demonstration is the cleanest way to prove the bias exists, and to viscerally show why cardinality (not true relevance) is driving the difference.

## 2.5 Quick self-check before Chapter 3

- Can you write out the Gini impurity formula and compute it by hand for a simple two-class node?
- Can you explain, step by step, *why* a high-cardinality noise feature gets an inflated MDI score — not just that it does?
- Given a boosted-tree model, can you say which importance type (gain/weight/cover) you'd trust most for deciding whether to keep a high-cardinality feature, and why?

---

**Next: Chapter 3 — Intrinsic Importance: Linear & Generalized Linear Models**, covering standardized coefficients, the statistical-significance-vs-practical-importance distinction, VIF derived from first principles, and odds ratios as a classification-specific importance lens.
