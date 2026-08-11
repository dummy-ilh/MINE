# Chapter 4 — Random Forests

Chapter 3 ended on the exact gap Random Forests fill: bagging alone can only decorrelate trees as much as bootstrap resampling allows, and per the Chapter 2.2 variance formula, correlation $\rho$ sets a hard floor ($\rho\sigma^2$) that more trees alone can never push below. Random Forests add a second, stronger randomization source specifically to drive $\rho$ down further.

---

## 4.1 The Extra Trick: Random Feature Subsampling *at Every Split*

**Plain bagging (Ch.3) randomizes only the rows** each tree sees (via bootstrap resampling). Every tree still searches **all $p$ features** at every split (Chapter 1.3's full exhaustive sweep). This is the reason bagged trees stay fairly correlated: if one or two features are much more predictive than the rest, nearly every bootstrap sample will still surface those same dominant features as the best split near the root, so most trees end up structurally similar high up, even though they were trained on different rows.

**Random Forests randomize the columns too — freshly, at every single split, in every tree:**

At each node (not once per tree — once **per node**), instead of considering all $p$ features, draw a random subset of $k < p$ features and restrict the split search (Chapter 1.3's search procedure) to only those $k$ candidates. The best split is chosen from within this restricted subset, then a **new** random $k$-subset is drawn independently for the next node (even a sibling node in the same tree draws its own fresh subset).

**Why does this decorrelate trees more than row-resampling alone?** Consider a dataset where feature $X_1$ is strongly predictive and $X_2,\dots,X_{20}$ are weakly predictive. With plain bagging, $X_1$ nearly always wins the root split across most bootstrap samples — every tree's root, and much of its structure below, ends up similar (high $\rho$). With per-split feature subsampling, on any given node there's a real chance $X_1$ isn't even in the randomly-drawn candidate set — forcing the tree to split on one of the weaker features instead. This forces genuinely different tree structures across the ensemble, directly lowering the pairwise correlation $\rho$ that Chapter 2.2 showed is the binding constraint on variance reduction. Breiman's original Random Forest paper (2001) frames this explicitly: the goal is minimizing $\rho$ while keeping individual-tree strength (accuracy) as high as possible, since the ensemble's error bound depends on both.

**Standard choices for $k$ (sklearn's `max_features`):**
- Classification: $k = \sqrt{p}$ (default `'sqrt'`)
- Regression: $k = p/3$ (a commonly cited rule of thumb; sklearn's regressor default is actually `1.0`, i.e. all features — see the sklearn params table below for the nuance)

**Worked numerical — why $\sqrt{p}$ specifically, intuition via a simple case:** Suppose $p=100$ features, and exactly 5 of them are genuinely predictive ("strong") while 95 are pure noise. With $k=\sqrt{100}=10$ features randomly drawn per split, the probability that **none** of the 5 strong features appear in a given draw is:

$$
P(\text{no strong feature in the sample}) = \frac{\binom{95}{10}}{\binom{100}{10}}
$$

Using the ratio-of-combinations shortcut (probability none of the 5 "marked" items appear when drawing 10 from 100 without replacement):
$$
P = \prod_{i=0}^{9}\frac{95-i}{100-i} = \frac{95}{100}\times\frac{94}{99}\times\frac{93}{98}\times\cdots\times\frac{86}{91}
$$
Each factor is a bit below $0.95$; multiplying ten such factors together gives approximately $0.95^{10}\approx 0.60$ as a rough estimate (the true value is close to this). So roughly 60% of the time, a given split's candidate pool contains **zero** strong features and the tree is forced to split on noise or weak signal at that node — while about 40% of the time at least one strong feature is available and gets used. This is exactly the mechanism: $k=\sqrt{p}$ is small enough to frequently "starve" the split of the dominant features (forcing structural diversity) while still leaving them available often enough that the ensemble's average is still built from genuinely useful splits. If $k$ were set much larger (say $k=50$ out of 100), strong features would appear in nearly every candidate pool, and you'd be back to bagging's original correlation problem; if $k$ were set much smaller (say $k=1$), individual trees would become too weak (high bias, since they're frequently blocked from ever splitting on the informative features at all) — $\sqrt{p}$ is an empirically-validated middle ground, not a value with a clean closed-form derivation.

**Why is this called "Random Forest" and not just "Bagging v2"?** Because the combination of *two independent randomization sources* (row bootstrap + per-split feature subsampling) is what defines the algorithm — remove either one and you get a different, weaker method (remove feature subsampling → plain Bagging; remove row bootstrap but keep feature subsampling → a real but less common variant). The name specifically signals "a forest where each tree is grown under an additional layer of injected randomness beyond just the training rows."

---

## 4.2 Key Hyperparameters and the Bias-Variance Trade-off Each One Controls

| Hyperparameter | ↑ Increasing it tends to... | Why (mechanism) |
|---|---|---|
| `n_estimators` (number of trees, $M$) | Reduce variance further (diminishing returns), never increases bias | Directly the $M$ in the Chapter 2.2 formula — pushes the $\frac{(1-\rho)\sigma^2}{M}$ term toward 0, floor unchanged |
| `max_features` ($k$) | **Increasing** $k$ toward $p$: raises $\rho$ (less decorrelation, per 4.1's logic) but also raises individual-tree strength/accuracy (less bias per tree, since more of the genuinely useful features are available at each split) | A genuine trade-off knob — this is Random Forest's most consequential hyperparameter precisely because it directly trades off the two competing forces (correlation floor vs. per-tree quality) that Breiman's theory identifies |
| `max_depth` / `min_samples_leaf` | Shallower trees (smaller `max_depth`, larger `min_samples_leaf`): raises bias, lowers per-tree variance | Same Chapter 1.4/1.5 logic as a standalone tree — but note: because Random Forest already gets variance reduction "for free" from averaging, RF trees are conventionally still grown fairly deep/unpruned (often `max_depth=None`), leaning on ensembling rather than per-tree pruning for variance control — pruning individual RF trees more aggressively mainly just adds bias without much variance benefit, since the ensemble was already handling variance |
| `n_estimators` **vs** `max_features` — which to tune first? | — | `max_features` changes the *floor* ($\rho\sigma^2$) that more trees can never overcome; `n_estimators` only closes the gap *to* that floor. In practice: tune `max_features` (and per-tree depth/leaf-size constraints) via cross-validation first, then set `n_estimators` generously high (compute allowing) since it can't hurt accuracy, only cost |

**Worked numerical, extending Chapter 2.2's example to compare `max_features` settings:** Suppose at `max_features` = all features (effectively plain bagging), $\rho=0.5$, $\sigma^2=4.0$; reducing `max_features` to $\sqrt{p}$ decorrelates trees down to $\rho=0.25$ but weakens each tree slightly, raising $\sigma^2$ to $4.5$ (weaker trees are often *more* variable individually, even as their *correlation* drops). At $M=200$ trees:

Plain-bagging-like setting: $\text{Var} = 0.5(4.0) + \frac{0.5(4.0)}{200} = 2.0 + 0.01 = 2.01$

RF-like setting: $\text{Var} = 0.25(4.5) + \frac{0.75(4.5)}{200} = 1.125 + 0.0169 = 1.142$

Even though the RF-like setting has a *higher* per-tree $\sigma^2$, its much lower $\rho$ wins decisively — final ensemble variance is nearly half. This numerically demonstrates why Random Forest reliably outperforms plain bagging in practice even though each individual RF tree, considered alone, is often *weaker* than an individual bagged tree.

---

## 4.3 Feature Importance

**Impurity-based / Mean Decrease in Impurity (MDI), sklearn's default `.feature_importances_`:**

For a single tree, feature $j$'s importance is the sum, over every node that splits on $j$, of the impurity decrease that split achieved (Chapter 1.2's $\Delta$), weighted by the fraction of samples reaching that node:

$$
\text{Imp}_j^{\text{tree}} = \sum_{\text{nodes } t \text{ splitting on } j} \frac{n_t}{n}\Delta(t)
$$

Averaged across all $M$ trees in the forest, then normalized to sum to 1 across features.

**Why is MDI biased toward high-cardinality / continuous features?** A feature with many possible split thresholds (e.g., a continuous variable, or a categorical variable with many levels) gives the greedy split search (Chapter 1.3) *more chances* to find a threshold that happens to produce a large impurity decrease on the training data **by chance alone**, even when the feature has no true relationship with the target — this is a pure multiple-comparisons/overfitting-to-training-data effect, structurally the same bias problem that motivated Gain Ratio in Chapter 1.3.1's ID3 vs C4.5 discussion, just showing up in importance scores rather than in split selection itself. A binary feature only ever gets one possible threshold to try, so it has far fewer chances to get "lucky."

**Permutation Importance — the fix:**
1. Compute the model's baseline score (accuracy/$R^2$/etc.) on a held-out (or OOB, Ch.3.3) set.
2. For feature $j$: randomly shuffle (permute) just that feature's values across the evaluation samples, breaking its relationship with the target while leaving its marginal distribution and every other feature untouched.
3. Recompute the score on this permuted set. The importance of feature $j$ is the **drop** in score caused by the shuffle: $\text{Imp}_j = \text{Score}_{\text{baseline}} - \text{Score}_{\text{permuted}}$.
4. Repeat multiple times (different random shuffles) and average, for a stable estimate.

**Worked numerical:** baseline OOB accuracy = 0.850. After permuting feature "income": accuracy drops to 0.790 (average over 10 shuffle repeats). Permutation importance of "income" $= 0.850-0.790=0.060$. After permuting feature "favorite_color" (a low-signal feature): accuracy barely moves, 0.850→0.848, importance $=0.002$ — correctly reflecting near-zero true predictive value, regardless of how many split thresholds that feature offered during training.

**Why is permutation importance more trustworthy, and why isn't it simply always used instead of MDI?** It measures importance via actual *predictive contribution on held-out-like data*, immune to the "more thresholds = more lucky training-set impurity decreases" bias described above. The cost: it requires $p$ (or $p\times$ repeats) full re-scoring passes over held-out data after the model is already trained — for large $p$ or large ensembles this is meaningfully more expensive than MDI, which sklearn gets essentially for free as a byproduct of training (it's just bookkeeping already computed during the split search). MDI remains the quick-look default; permutation importance is the more rigorous choice when the ranking itself matters (e.g., for feature selection decisions).

---

## 4.4 Random Forest vs. Plain Bagging — the Precise Delta

| | Bagging (`BaggingClassifier`) | Random Forest |
|---|---|---|
| Row randomization | Bootstrap resampling (Ch.3.1) | Same |
| Feature randomization | None by default (`max_features=1.0`); optional, but sampled **once per tree** if enabled (Ch.3's sklearn table) | **Always on**, sampled **fresh at every split**, in every tree (4.1) |
| Correlation $\rho$ between trees | Higher (only row diversity driving decorrelation) | Lower (two independent randomization sources) |
| Typical base learner depth | Full depth, unpruned (Ch.3.1) | Also typically full depth/unpruned — variance handled by ensembling either way |
| Built-in feature importance | Not standard | `.feature_importances_` (MDI) available natively |
| When they converge to the same algorithm | When `max_features` is set to consider all features at every split | — |

The one-sentence version, worth having ready verbatim for an interview: **Random Forest is bagging plus one additional randomization mechanism — per-split random feature subsampling — that decorrelates trees further than bootstrap resampling alone can, directly lowering the $\rho$ term that caps how much variance any bagging-style ensemble can remove.**

---

## 4.5 Extremely Randomized Trees (Extra Trees)

Extra Trees push randomization one step further than Random Forest:

1. **Typically no bootstrap resampling** — sklearn's `ExtraTreesClassifier`/`Regressor` default to `bootstrap=False`, training each tree on the **full** original dataset (rather than a bootstrap sample).
2. **Split *thresholds* are also randomized**, not just which features are candidates. For each candidate feature (drawn the same random-subset way as Random Forest), instead of doing the full optimal-threshold search from Chapter 1.3, Extra Trees draws **one random threshold** within that feature's observed range (per candidate feature) and picks the best among those random thresholds (rather than the best among *all possible* thresholds for each feature).

**Why does randomizing the threshold too help?** It's the same $\rho$-vs-$\sigma^2$ trade-off as 4.1, pushed further: random thresholds decorrelate trees even more (two trees are now unlikely to agree on a split even when they agree on which feature to use), at the cost of each individual tree being a weaker/noisier fit (since it's no longer using the locally-optimal threshold, Chapter 1.3's exhaustive-search guarantee is given up). Whether this net trade-off wins depends on the dataset — Extra Trees sometimes outperforms Random Forest, particularly on noisy data where the "extra" randomization prevents individual trees from overfitting to spurious optimal-looking thresholds, and it trains faster (skipping Chapter 1.3's $O(n\log n)$ sweep per feature in favor of one random draw is cheaper), but it is not a strict improvement — on cleaner, larger datasets where the optimal-threshold search reliably finds real signal (not noise), giving that up can lose more accuracy than the extra decorrelation buys back.

---

## sklearn Parameters — `RandomForestClassifier` / `RandomForestRegressor`

| Parameter | What it controls | Notes |
|---|---|---|
| `n_estimators` | Number of trees ($M$) | Default **100** (much higher than `BaggingClassifier`'s default of 10 — reflecting that RF trees are cheaper to decorrelate-and-average productively) |
| `criterion` | Split quality measure | Same options as Chapter 1's table (`gini`/`entropy`/`log_loss` for classifier, `squared_error`/etc. for regressor) |
| `max_features` | Features considered **per split** (Section 4.1 — this is the key structural difference from `BaggingClassifier`'s per-*tree* `max_features`) | Classifier default: `'sqrt'` ($k=\sqrt p$, Section 4.1). Regressor default: **`1.0`** (all features) — note this means sklearn's out-of-the-box `RandomForestRegressor` doesn't apply the classic $p/3$ rule of thumb automatically; tuning `max_features` down manually is often worthwhile for regression tasks |
| `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, `ccp_alpha` | Same per-tree controls as Chapter 1's table | Applied identically to every tree in the forest |
| `bootstrap` | Whether row sampling uses bootstrap (with replacement) | Default `True`. Setting `False` trains every tree on the *entire* dataset, relying solely on feature subsampling for diversity |
| `oob_score` | Compute Out-of-Bag error (Ch.3.3), only meaningful when `bootstrap=True` | Default `False` |
| `class_weight` | Reweight classes | Same as Chapter 1's table |
| `n_jobs` | Parallelism across trees | Same embarrassingly-parallel property as Bagging (Ch.3) — RF training parallelizes just as well |
| `max_samples` | Fraction of samples per bootstrap draw (new relative to defaults — lets you subsample rows *below* 100% even with bootstrap on) | Default `None` (draw $n$ samples, i.e. standard bootstrap size) |

**`ExtraTreesClassifier`/`Regressor` parameters** mirror the above almost exactly, with two defaults flipped: `bootstrap=False` by default (4.5), and the split search uses random thresholds per candidate feature rather than Chapter 1.3's exhaustive sweep (not separately exposed as a tunable parameter — it's the defining mechanism of the class itself).

---

## Quick Interview Q&A

**Q: "Does Random Forest overfit as you add more trees?"**
A: No — same argument as bagging (Ch.3.4): each tree is trained independently, so more trees only ever push variance toward the $\rho\sigma^2$ floor (Ch.2.2), never reintroduce overfitting. `n_estimators` is a compute-vs-diminishing-returns knob, not a regularization knob to fear over-tuning.

**Q: "Why does GBM (boosting, Ch.5) need a learning rate but Random Forest doesn't?"**
A: Boosting's rounds are sequential corrections to residual error (Ch.5) — without shrinking each round's contribution (the learning rate), the ensemble can chase training-set noise more and more aggressively round after round, since nothing structurally prevents late rounds from fitting pure residual noise. Random Forest's trees are trained independently and simply averaged — there's no sequential error-chasing mechanism for a learning rate to dampen.

**Q: "You have 500 features, 5 of which are truly predictive. Would you rather use Random Forest with `max_features='sqrt'` or Bagging with `max_features=1.0`? Why?"**
A: Random Forest — precisely the 4.1 scenario. With 500 features and only 5 truly useful, plain bagging's every-split-sees-all-features design means the same handful of strong features dominate nearly every tree's structure (very high $\rho$). Random Forest's `sqrt(500)≈22`-feature-per-split subsampling forces many splits to proceed without the strongest features available, producing meaningfully more structurally diverse trees and a lower correlation floor — exactly the setting where the extra decorrelation mechanism earns its keep the most.

---

**Next up: Chapter 5 — Boosting (AdaBoost's weight-update derivation, Gradient Boosting as functional gradient descent, then XGBoost/LightGBM/CatBoost's specific innovations).**
