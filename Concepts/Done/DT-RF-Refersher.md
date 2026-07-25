Here's a solid rundown of Decision Trees, Random Forests, and Bagging — the concepts interviewers actually probe, plus the kinds of questions that tend to come up at places like Apple and Google.

## Core Concepts

**Decision Trees**
- Split data recursively based on feature thresholds that best separate the target variable.
- Splitting criteria: **Gini impurity** or **Entropy/Information Gain** (classification), **variance reduction / MSE** (regression).
- Prone to **overfitting** — a fully grown tree can memorize training data (low bias, high variance).
- Pruning (pre-pruning: max_depth, min_samples_leaf; post-pruning: cost-complexity pruning) controls this.
- Greedy algorithm — locally optimal splits at each node, not globally optimal tree.

**Bagging (Bootstrap Aggregating)**
- Train multiple models on **bootstrap samples** (sampled with replacement) of the training data.
- Aggregate predictions: majority vote (classification) or average (regression).
- Reduces **variance** without increasing bias much — works best with high-variance, low-bias base learners (like deep decision trees).
- Each bootstrap sample leaves out ~37% of data (**out-of-bag / OOB samples**) — useful for validation without a separate holdout set.

**Random Forest**
- Bagging + **feature randomness**: at each split, only a random subset of features is considered (not all).
- This decorrelates the trees — if one feature is very strong, bagging alone would let every tree split on it first, making trees similar. Feature subsampling forces diversity.
- Reduces variance further than plain bagging.
- Gives free feature importance estimates (via mean decrease in impurity or permutation importance).

## Conceptual

**1. How does a decision tree decide where to split? Gini vs Entropy?**
At each node, the tree tries every feature and every possible threshold, and picks the split that most reduces impurity in the resulting child nodes.
- **Gini impurity**: 1 − Σp², measures probability of misclassifying a randomly chosen element. Ranges 0 (pure) to 0.5 (max impurity, binary case).
- **Entropy**: −Σp·log₂(p), from information theory, measures disorder.
They usually give similar trees. Gini is preferred in practice because it's cheaper to compute (no log), and entropy sometimes gives slightly more balanced trees since it penalizes impurity a bit more aggressively.

**2. Why do decision trees overfit, and how does RF fix it?**
A fully grown tree keeps splitting until nodes are pure or tiny, effectively memorizing training data — low bias, very high variance. Random Forest averages many such trees trained on different bootstrap samples and feature subsets. Individual trees still overfit their own sample, but their errors are decorrelated, so averaging cancels out noise and variance drops sharply while bias stays roughly the same.

**3. Why feature randomness on top of bagging?**
Bootstrapping alone doesn't decorrelate trees enough — if one or two features are strongly predictive, nearly every tree will pick them for the top splits regardless of the bootstrap sample, making trees highly correlated. Averaging correlated predictors reduces variance much less than averaging independent ones. Restricting each split to a random subset of features forces different trees to rely on different features, decorrelating them and improving the variance reduction from averaging.

**4. Bias-variance tradeoff: bagging vs boosting**
- **Bagging**: trains models independently/in parallel on bootstrap samples, averages them → reduces **variance**, bias stays about the same as the base learner.
- **Boosting**: trains models sequentially, each new model focuses on the errors of the previous ones → reduces **bias** (and some variance), but can be more prone to overfitting if unchecked (hence learning rate, early stopping).

**5. Out-of-bag (OOB) error**
Each bootstrap sample uses roughly 63% of the data (sampling with replacement); the remaining ~37% ("out-of-bag") wasn't seen by that tree. For each training point, you can average predictions from only the trees that didn't see it during training — this gives an unbiased-ish estimate of test error without needing a separate validation set, and it's essentially "free" cross-validation.

**6. Why are trees high variance?**
Small changes in training data can lead to very different splits, especially near the top of the tree, since a different early split cascades into a completely different tree structure. Deep trees have very few samples per leaf, making leaf predictions sensitive to noise.

**7. Can Random Forest overfit?**
Yes, though much less than a single tree. It can happen if trees are very deep, `n_estimators` is small, `max_features` is too high (trees too correlated), or if there's leakage/duplicated data. In general, adding more trees doesn't overfit further (variance reduction saturates), but tree depth and correlation among trees still matter.

**8. Why doesn't bagging help linear regression much?**
Bagging reduces variance, but linear regression is already a low-variance, high-bias (relatively stable) model — small changes in training data don't change the fit much. So there's little variance to reduce, and you just add computational cost for negligible benefit. Bagging helps most for high-variance, unstable learners like deep trees.

## Comparisons

**9. Random Forest vs Gradient Boosted Trees**
- RF: parallel, bagged, reduces variance, trees are usually deep/fully grown, robust to overfitting via more trees, easier to tune.
- GBT: sequential, each tree fits the residual/gradient of the previous ensemble, reduces bias, trees are usually shallow (weak learners), more sensitive to hyperparameters (learning rate, number of estimators, depth) and to overfitting if not tuned carefully, but typically higher accuracy ceiling.

**10. Bagging vs Boosting**
| | Bagging | Boosting |
|---|---|---|
| Training | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias |
| Base learners | Strong, unstable (deep trees) | Weak learners (shallow trees) |
| Outlier sensitivity | More robust | More sensitive (residual-based) |
| Overfitting risk | Low | Higher if untuned |

**11. When to prefer a single decision tree over RF?**
When interpretability matters (you need to explain exact decision rules, e.g. regulatory/compliance contexts), when you need very fast inference/small model size, or when the dataset is small and simple enough that ensemble variance reduction isn't needed.

## Applied / Scenario-Based

**12. Great on train, poor on test — what do you check?**
- Tree depth too high / `min_samples_leaf` too low → overfitting individual trees.
- `max_features` too high → correlated trees, not enough variance reduction.
- Too few trees.
- Data leakage (a feature that encodes the target, or train/test overlap).
- Train/test distribution shift.
- Target leakage through time-based features if it's a time series problem evaluated incorrectly (no proper time split).

**13. Key hyperparameters to tune**
- `n_estimators` — more is generally better (diminishing returns, not overfitting risk), until compute cost concern.
- `max_depth` / `min_samples_leaf` / `min_samples_split` — control individual tree complexity.
- `max_features` — controls decorrelation between trees; smaller = more decorrelated but each tree weaker.
- `bootstrap` (on/off), sample size.
Typically I'd do a randomized or grid search cross-validated on `max_depth`, `max_features`, and `min_samples_leaf` first — those have the most impact.

**14. High-cardinality categorical features**
Options: target/mean encoding (careful of leakage — use out-of-fold encoding), frequency encoding, embedding-based encoding for very high cardinality, or grouping rare categories into "other." One-hot encoding a very high-cardinality feature bloats dimensionality and biases impurity-based splits/importance toward that feature.

**15. Missing data / class imbalance**
- Missing data: some implementations (e.g., XGBoost) handle it natively via learned default directions; scikit-learn's RF doesn't — you'd impute (median/mode, or model-based imputation) beforehand, or add a "missing" indicator feature.
- Class imbalance: use `class_weight='balanced'`, oversample minority class (SMOTE) or undersample majority, adjust decision threshold, or evaluate with precision/recall/F1/AUC-PR rather than accuracy.

**16. Unexpected feature ranking high in importance**
Two common causes to check:
- **Correlated features** inflate/split importance oddly — one correlated feature might "steal" importance from another depending on which gets picked first in each tree.
- **Impurity-based (Gini) importance is biased toward high-cardinality or continuous features**, since they offer more possible split points, giving spurious opportunities to reduce impurity. Switching to **permutation importance** (shuffle the feature and measure performance drop) is a more reliable diagnostic since it's model-agnostic and not biased by cardinality.

**17. Real-time inference — RF or something else?**
Depends on latency/model size constraints:
- RF: reasonably fast inference (parallelizable across trees) but model size can be large (many deep trees) and prediction requires traversing every tree — can be a problem at very tight latency budgets or on-device (mobile) settings.
- If ultra-low latency or on-device: consider a smaller ensemble, gradient boosted trees with fewer/shallower trees, model compression/distillation, or even a well-regularized linear/logistic model or small neural net, depending on accuracy requirements. I'd benchmark actual p99 latency and model size before committing.

## Math / Statistics

**18. Variance reduction from averaging**
If you average N i.i.d. models each with variance σ², the variance of the average is σ²/N (variance of a sum of independent variables scales linearly, divide by N² from the averaging, cancels to σ²/N — pure derivation: Var(X̄) = Var(ΣXᵢ/N) = (1/N²)ΣVar(Xᵢ) = (1/N²)(Nσ²) = σ²/N).

In bagging, trees aren't independent — they're trained on overlapping bootstrap samples from the same data and often share dominant features, so they're correlated with some correlation ρ. The actual variance of the average becomes:
ρσ² + (1−ρ)σ²/N
As N→∞, this doesn't go to zero — it converges to ρσ², the "correlation floor." This is exactly why Random Forest's feature randomization (reducing ρ) matters — it pushes that floor down further than bagging alone could.

**19. Why Gini over entropy computationally**
Entropy requires computing log₂(p) for each class probability at every candidate split, across every feature and threshold — this is expensive at scale. Gini only needs multiplication and subtraction (1 − Σp²), which is much faster, especially significant when trees are built over large datasets with many splits evaluated.

**20. Computing feature importance & its pitfall**
**Impurity-based (Gini/MDI) importance**: for each feature, sum up the (weighted) impurity decrease at every split where that feature is used, across all trees, then average. Fast, built into most libraries.
**Pitfall**: biased toward high-cardinality and continuous features, since they have more possible split points and thus more chances to reduce impurity — even if not truly more predictive. A categorical feature with only 2 levels will systematically look less important than a continuous feature with similar true predictive power.
**Fix**: use **permutation importance** — shuffle each feature's values (breaking its relationship with the target) and measure the drop in model performance on a held-out set. This is model-agnostic and doesn't have the cardinality bias, though it's more computationally expensive and can be misleading with strongly correlated features (importance gets split/diluted between them).

## System-Design-Adjacent

**21. RF for device failure / battery degradation prediction**
Features I'd engineer: charge cycle count, average charge/discharge rate, temperature exposure history (mean/max, especially high-heat events), depth of discharge per cycle, time since manufacture, usage patterns (screen-on time, app CPU load), firmware version, battery capacity fade over recent cycles (rate of change, not just current level), and environmental factors if available (climate region). I'd frame it as either a classification (fails within next X days) or a regression on remaining useful life, and would care a lot about **class imbalance** (failures are rare) and **calibration** of predicted probabilities for maintenance-scheduling decisions.

**22. Parallelizing training/prediction of millions of trees**
- **Training**: trees are trained independently in bagging, so this is embarrassingly parallel — distribute tree-building across machines/cores (e.g., Spark MLlib, or manual sharding with joblib/multiprocessing), each worker builds a subset of trees on its own bootstrap sample.
- **Prediction**: also embarrassingly parallel across trees — scatter the same input to all trees, gather and aggregate (vote/average) results. For very large forests, prediction latency can be reduced by pruning/compressing trees, using a subset of trees for a fast approximate answer, or precomputing/caching predictions for common input patterns.

Let's do both — starting with the toy split, then the variance derivation.

## Part 1: Gini vs Entropy — Worked Example

**Setup:** Suppose we're splitting a node with 10 samples: 6 belong to Class A, 4 belong to Class B.

**Step 1 — Impurity of the parent node**

Gini(parent) = 1 − (p_A² + p_B²) = 1 − (0.6² + 0.4²) = 1 − (0.36 + 0.16) = 1 − 0.52 = **0.48**

Entropy(parent) = −(0.6·log₂0.6 + 0.4·log₂0.4) = −(0.6×(−0.737) + 0.4×(−1.322)) = −(−0.442 − 0.529) = **0.971**

**Step 2 — Try a candidate split**

Say some feature (e.g., "age > 30") splits the 10 samples into:
- Left node: 5 samples → 4 A, 1 B
- Right node: 5 samples → 2 A, 3 B

**Left node impurity:**
- Gini = 1 − (0.8² + 0.2²) = 1 − (0.64+0.04) = 1 − 0.68 = **0.32**
- Entropy = −(0.8·log₂0.8 + 0.2·log₂0.2) = −(0.8×(−0.322)+0.2×(−2.322)) = −(−0.257−0.464) = **0.722**

**Right node impurity:**
- Gini = 1 − (0.4² + 0.6²) = 1 − (0.16+0.36) = 1 − 0.52 = **0.48**
- Entropy = −(0.4·log₂0.4 + 0.6·log₂0.6) = **0.971**

**Step 3 — Weighted impurity after split**

Both nodes have equal weight (5/10 each):

Gini(split) = 0.5×0.32 + 0.5×0.48 = 0.16 + 0.24 = **0.40**
Entropy(split) = 0.5×0.722 + 0.5×0.971 = 0.361 + 0.486 = **0.847**

**Step 4 — Gain from this split**

- Gini gain = 0.48 − 0.40 = **0.08**
- Information Gain (entropy) = 0.971 − 0.847 = **0.124**

**What this shows:** the tree algorithm computes this gain for *every* candidate split across every feature and picks whichever split maximizes the gain. In practice, Gini and Entropy tend to rank splits almost identically — they usually agree on which split is best, even though the raw numbers differ. The main practical difference is computational cost (Gini avoids the log), and entropy occasionally penalizes highly unbalanced splits a bit more, nudging toward more balanced trees. In an interview, showing this exact 4-step process (impurity of parent → impurity of children → weighted average → gain) is what demonstrates you actually understand it rather than just naming the formulas.

---

## Part 2: The Correlation Floor Derivation

This is the one that separates "I've heard of Random Forest" from "I understand *why* it works." Let's build it up carefully.

**Setup**

Say we have N trees, each an unbiased predictor of the true value, each with the same variance σ². Let $X_i$ be the prediction of tree $i$. We assume:
- $\text{Var}(X_i) = \sigma^2$ for every tree (identically distributed)
- $\text{Corr}(X_i, X_j) = \rho$ for every pair $i \ne j$ (some shared correlation, since trees are trained on overlapping/bootstrapped data)

The forest's final prediction is the average: $\bar{X} = \frac{1}{N}\sum_{i=1}^N X_i$

**Step 1 — General variance of a sum**

$$\text{Var}\left(\sum_i X_i\right) = \sum_i \text{Var}(X_i) + \sum_{i \ne j} \text{Cov}(X_i, X_j)$$

- There are N variance terms, each equal to σ² → contributes $N\sigma^2$
- There are N(N−1) covariance terms (ordered pairs, i≠j), each equal to $\rho\sigma^2$ (since $\text{Cov}(X_i,X_j) = \rho \cdot \sigma \cdot \sigma$) → contributes $N(N-1)\rho\sigma^2$

So:
$$\text{Var}\left(\sum_i X_i\right) = N\sigma^2 + N(N-1)\rho\sigma^2$$

**Step 2 — Variance of the average**

Dividing by $N^2$ (since $\bar X = \frac{1}{N}\sum X_i$):

$$\text{Var}(\bar{X}) = \frac{N\sigma^2 + N(N-1)\rho\sigma^2}{N^2} = \frac{\sigma^2}{N} + \frac{(N-1)}{N}\rho\sigma^2$$

**Step 3 — Take the limit as N → ∞**

$$\text{Var}(\bar{X}) = \underbrace{\frac{\sigma^2}{N}}_{\to 0} + \underbrace{\frac{N-1}{N}}_{\to 1}\rho\sigma^2 \;\longrightarrow\; \rho\sigma^2$$

**The punchline:** No matter how many trees you add, the variance of the ensemble never drops below $\rho\sigma^2$. The first term (the part that behaves like the "independent case," $\sigma^2/N$) vanishes as N grows — that's the free lunch you get from having lots of trees. But the second term is a **floor** set entirely by how correlated the trees are with each other. Adding more trees past a certain point buys you almost nothing if ρ is high.

**Why this matters for the RF vs plain-bagging distinction:**
- If ρ = 1 (trees are perfectly correlated, e.g., always splitting on the same dominant feature first), then $\text{Var}(\bar X) = \sigma^2$ — averaging buys you *nothing at all*, you might as well have one tree.
- If ρ = 0 (fully independent trees), you're back to $\sigma^2/N \to 0$ — the ideal case.

Random Forest's random feature subsampling exists specifically to push ρ down. Bootstrapping the training data alone reduces ρ a little (different samples → somewhat different trees), but if there's one or two dominant predictive features, almost every bootstrapped tree will still choose them for its top splits, keeping ρ high. By restricting each split to consider only a random subset of features, RF forces some trees to build their top splits around *different* features, which decorrelates the trees more aggressively — lowering ρ and thus lowering the variance floor $\rho\sigma^2$ further than bagging alone ever could.

**How I'd deliver this on a whiteboard in an interview:** write the covariance expansion first, show the two terms fall out of dividing by N², circle the term that vanishes vs the term that doesn't, then connect the surviving term directly back to *why* `max_features < total_features` is the single hyperparameter that most defines Random Forest as different from plain bagging. That's usually the "aha" moment interviewers are listening for.

