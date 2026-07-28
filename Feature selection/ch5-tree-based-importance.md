# Chapter 5 — Feature Importance in Tree-Based Models

Chapters 2–4 were about *selecting* features before or during training. This chapter shifts to a related but distinct question: given a model you've already trained, **which features did it actually rely on, and by how much?** This is the "feature importance" half of the topic, and it starts with tree-based models because they're where importance measures were first popularized and where they're asked about most in interviews.

## 5.1 Mean Decrease in Impurity (MDI / Gini importance)

**Setup you need first — what "impurity" means.** At every node in a decision tree, the splitting algorithm picks the feature and threshold that most reduces "impurity" — a measure of how mixed the classes (or how spread the values) are among the examples at that node. For classification, a common impurity measure is **Gini impurity**: roughly, the probability that two randomly picked examples from the node would have *different* class labels — 0 if the node is perfectly pure (all one class), higher as the mix gets more even. For regression trees, the analogous quantity is typically variance of the target within the node.

**How MDI importance is computed:** for each feature, sum up the impurity reduction it produced across **every split, in every tree** where it was used, weighted by how many training examples passed through that particular node (a split near the root, seen by almost all the data, counts for more than a split deep in the tree, seen by only a handful of examples). Sum this across all trees in the forest, and normalize so all features' importances add up to 1. The result: a single number per feature, representing "how much total impurity reduction is this feature responsible for, across the whole forest."

**Worked numeric example.** Imagine a tiny single tree with 3 splits:
- Root split on `credit_score`, covering all 100 examples, reduces Gini impurity by 0.20.
- Left child splits on `income`, covering 60 of those examples, reduces Gini impurity by 0.10.
- Right child splits on `credit_score` again, covering 40 examples, reduces Gini impurity by 0.05.

`credit_score`'s total weighted impurity reduction: (100 × 0.20) + (40 × 0.05) = 20 + 2 = 22.
`income`'s total weighted impurity reduction: 60 × 0.10 = 6.

Normalizing: `credit_score` importance = 22/(22+6) ≈ **0.79**, `income` importance = 6/28 ≈ **0.21**. `credit_score` is roughly 3.7× as important as `income` by this measure — driven mostly by the fact that its split at the root affected all 100 examples, not just 60.

## 5.2 The known bias of MDI — why it inflates some features artificially

Here's the interview-critical caveat: **MDI systematically overestimates the importance of high-cardinality and continuous features**, even when they're not actually more predictive. Understanding *why* is the whole point of this section.

**The mechanism:** a feature with many possible split points (a continuous feature, or a categorical feature with many distinct categories) gives the tree-building algorithm many more candidate thresholds to try when searching for the best split at each node. Purely by having more chances to be tried, such a feature has a higher probability of *appearing* to produce a great split **purely by chance**, on any given finite training sample — the same "more attempts, more chances of a lucky-looking result" effect that shows up any time you search over many candidates (this is the exact same phenomenon flagged in Chapter 3, §3.6, about wrapper methods risking overfitting the selection process itself — MDI has an analogous, subtler version of that same problem baked into how it scores features).

**Concrete illustration:** imagine adding a completely random, uninformative feature to your dataset, but give it 1,000 distinct random values (high cardinality) rather than a simple binary 0/1 (low cardinality). The high-cardinality random feature will tend to receive a *non-trivial* MDI importance score — not because it's actually informative, but because with 1,000 possible split thresholds to try, the tree-building search is likely to stumble onto some threshold that happens to split the training data favorably by pure chance, and MDI has no way to distinguish "a real pattern" from "a lucky split on a feature with lots of options to try."

**The practical consequence:** never trust an MDI-based feature ranking blindly, especially when your dataset mixes low-cardinality categorical features (e.g., a binary flag) with high-cardinality ones (e.g., a zip code with hundreds of distinct values) or continuous features — the high-cardinality/continuous ones will tend to look artificially more important than they really are.

## 5.3 Permutation importance — a more trustworthy alternative

**The idea:** instead of looking at *how the tree was built* (which is what makes MDI biased), measure **how much the model's actual predictive performance degrades when a feature's information is destroyed**, directly on held-out data.

**The procedure:**
1. Measure the trained model's performance (accuracy, AUC, etc.) on a held-out validation set — call this the baseline.
2. Take one feature column and **randomly shuffle its values** across the rows of the validation set, breaking any real relationship between that feature and the target while leaving every other feature and the target untouched.
3. Re-measure the model's performance on this shuffled data.
4. The importance of that feature is the **drop in performance** caused by the shuffle — a feature the model relied on heavily will cause a big performance drop when scrambled; a feature the model barely used will cause almost no drop.
5. Repeat steps 2–4 for every feature (each shuffled independently, one at a time), to get an importance score per feature.

**Why this fixes MDI's bias.** Permutation importance doesn't care at all how many split thresholds a feature has, or how the tree-building search happened to explore them — it only asks "does actually destroying this feature's information hurt real, held-out predictive performance?" A high-cardinality random feature that MDI mistakenly ranked as important will show **essentially zero** performance drop when shuffled, since it was never actually informative in the first place — permutation importance correctly exposes it as unimportant, where MDI was fooled.

## 5.4 Permutation importance's own pitfalls

Permutation importance isn't free of problems either — two worth knowing:

- **Correlated features split credit unreliably.** If two features are highly correlated (say, `income` and `annual_earnings`, near-duplicates), shuffling just one of them barely hurts performance at all — because the model can still get nearly the same information from the *other*, un-shuffled correlated feature. Both features can end up looking artificially unimportant individually, even though the *pair* of them (or either one alone, if the other weren't present) is genuinely valuable — the exact same redundancy issue that showed up with Lasso and correlated features back in Chapter 4, §4.3, just manifesting as an importance-measurement artifact here instead of a selection artifact.
- **Computational cost.** Unlike MDI (which is essentially free — it's just a byproduct of numbers already computed during training), permutation importance requires re-scoring the model on the full validation set once per feature (plus, for a stable estimate, ideally repeating each feature's shuffle multiple times and averaging, since a single random shuffle can itself be a bit noisy) — meaningfully more expensive for datasets with very many features or very large validation sets.

## 5.5 Practical guidance

Given 5.2–5.4, the practical rule of thumb: **use permutation importance as your primary trusted measure whenever you can afford the extra compute**, especially if your dataset has a mix of high- and low-cardinality features (where MDI's bias is most dangerous). MDI is still fine as a *quick, free, rough first look* — especially when comparing features of similar cardinality/type to each other, where its bias mostly cancels out — but should not be the final word if the ranking will inform an important, especially cardinality-mixed, decision.

---

**Next: Chapter 6 — SHAP and Other Model-Agnostic Importance Methods**, where we move beyond tree-specific measures to Shapley-value-based explanations that work for any model type, plus LIME as a cheaper local-approximation alternative.
