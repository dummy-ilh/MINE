# Chapter 3 — Bagging (Bootstrap Aggregating)

Chapter 2 derived, in general terms, why averaging correlated estimators reduces variance and left off with bootstrap sampling theory. This chapter turns that machinery into the concrete algorithm and works through its numerical behavior end to end.

---

## 3.1 The Algorithm, Step by Step

Given a training set $D = \{(x_1,y_1),\dots,(x_n,y_n)\}$ and a base learner (in this curriculum: a decision tree, typically grown deep/unpruned):

1. **For $m = 1$ to $M$:**
   a. Draw a bootstrap sample $D_m$ of size $n$ from $D$, sampling **with replacement** (Chapter 2.4's mechanism — each $D_m$ is the same size as $D$ but contains duplicates and omits ~36.8% of the original samples on average).
   b. Train a base learner $\hat f_m$ on $D_m$, to full depth, with **no pruning**.
2. **Aggregate** the $M$ trained learners into a single prediction:
   - Regression: $\hat f_{\text{bag}}(x) = \frac{1}{M}\sum_{m=1}^M \hat f_m(x)$ (simple average)
   - Classification: either **majority vote** across the $M$ trees' hard predictions, or average the $M$ trees' predicted class *probabilities* and take the argmax (soft voting) — sklearn's `BaggingClassifier`/`RandomForestClassifier` use soft voting (probability averaging) by default when the base estimator supports `predict_proba`, since it uses more information than a hard vote and empirically tends to perform at least as well.

That's the entire algorithm — its power is not architectural cleverness, it's the variance-reduction math from Chapter 2.2 applied with trees as the deliberately high-variance base learner.

**Why grow the base trees deep/unpruned, when Chapter 1.4 spent an entire section on pruning?** Because bagging's whole value proposition is averaging away variance — and per the Chapter 2.2 formula, the amount of variance-reduction benefit available is proportional to how much variance $\sigma^2$ there was to begin with. A pruned, shallow tree already has low variance (and correspondingly higher bias, per Ch.1.5's bias-variance tradeoff) — there's little left for bagging to usefully reduce, while its bias (which bagging cannot touch, per 2.2) stays exactly as high as a single pruned tree's. Deliberately *not* pruning hands bagging maximum raw material to work with.

---

## 3.2 Worked Numerical: Bagging End to End on a Toy Dataset

Original dataset ($n=6$), regression target:

| i | x | y |
|---|---|---|
| 1 | 1 | 10 |
| 2 | 2 | 12 |
| 3 | 3 | 20 |
| 4 | 4 | 22 |
| 5 | 5 | 30 |
| 6 | 6 | 32 |

Suppose we draw $M=3$ bootstrap samples (small $M$ purely to keep the arithmetic tractable by hand):

**Bootstrap sample $D_1$** (drawn with replacement, indices): $\{1,1,3,4,5,6\}$ → out-of-bag for this tree: $\{2\}$
**Bootstrap sample $D_2$**: $\{2,2,3,3,5,6\}$ → OOB: $\{1,4\}$
**Bootstrap sample $D_3$**: $\{1,2,4,4,5,5\}$ → OOB: $\{3,6\}$

Say each tree, trained on its bootstrap sample, is a simple stump (one split) that predicts the mean $y$ of whichever "side" a new $x$ falls on. For a test point $x_0=3.5$ (between the low cluster ~10-20 and high cluster ~22-32), suppose the three trees—shaped differently because they saw different resampled data—predict:

- $\hat f_1(x_0) = 21.0$
- $\hat f_2(x_0) = 24.5$
- $\hat f_3(x_0) = 19.0$

**Bagged prediction (simple average, per 3.1's aggregation rule):**
$$
\hat f_{\text{bag}}(x_0) = \frac{21.0+24.5+19.0}{3} = \frac{64.5}{3} = 21.5
$$

If the true (noise-free) function value at $x_0=3.5$ is $f(x_0)=21.0$: single-tree errors were $|21.0-21.0|=0$, $|24.5-21.0|=3.5$, $|19.0-21.0|=2.0$ — a spread from 0 to 3.5. The bagged prediction's error is $|21.5-21.0|=0.5$ — smaller than two of the three individual trees' errors, illustrating (on one toy point) exactly the variance-smoothing the Chapter 2.2 formula predicts across many such points on average.

---

## 3.3 Out-of-Bag (OOB) Error — Derivation and a Full Numerical Walkthrough

**The idea:** each original sample $i$ was, on average, excluded from about 36.8% of the $M$ bootstrap samples (Chapter 2.4). For sample $i$, gather every tree $m$ for which $i \notin D_m$ (i.e., every tree that *never saw sample $i$ during training*) — call this set of trees $S_i$. Predict $y_i$ using only the trees in $S_i$, aggregated the same way as 3.1. This is, for that one sample, effectively a held-out prediction — without spending any data on a separate validation split.

$$
\hat y_i^{\text{OOB}} = \text{aggregate}\big(\{\hat f_m(x_i) : m \in S_i\}\big)
$$

The **OOB error** is then simply this OOB-predicted value compared to the true value, averaged over all $n$ samples:
$$
\text{OOB Error} = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat y_i^{\text{OOB}})
$$
for whatever loss $L$ is appropriate (squared error for regression, misclassification indicator for classification).

**Worked numerical, continuing the toy example above:**

Sample 1 was OOB only for $D_1$ (looking back at the sets: $D_1$'s OOB was $\{2\}$, not 1 — let me use the actual OOB sets defined above precisely): from the three bootstrap draws, sample 1 was OOB for $D_2$ (OOB set $\{1,4\}$) and $D_3$ (OOB set — wait, $D_3$'s OOB was $\{3,6\}$, not 1). So sample 1 is OOB **only** for $D_2$. With just $M=3$ trees, some samples (like sample 1 here) have only a single OOB tree available — this is exactly why OOB error estimates are noisy/unreliable with small $M$, and why in practice you need $M$ in the hundreds (typical Random Forest defaults, Ch.4) before OOB error stabilizes into a trustworthy estimate: with more trees, every sample accumulates enough OOB "voters" (in expectation, ~36.8% of $M$ trees per sample) for the aggregate OOB prediction to average out noise the same way the main ensemble does.

Say tree 2 (trained on $D_2$) predicts $\hat f_2(x_1) = 11.2$ for sample 1's OOB prediction. Since it's the only OOB tree for sample 1, $\hat y_1^{\text{OOB}} = 11.2$. True $y_1=10$. Squared error contribution: $(10-11.2)^2 = 1.44$.

Repeating this for all 6 samples and averaging the squared errors gives the OOB MSE — a number you can compute **during training, from the training set alone**, that behaves like a genuine test-set estimate because each sample's OOB prediction never used that sample in the trees that produced it.

**Why is this specifically valuable compared to a plain train/validation split?** A held-out validation split permanently removes, say, 20% of your data from ever influencing the trained model — costly when data is limited. OOB estimation gets a comparable unbiased-ish error estimate **while still using 100% of the data to train the final ensemble** (every sample contributes to the ~63.2% of trees that do include it), because the "held-out-ness" is spread across different trees per sample rather than concentrated in one static held-out block. Breiman (the originator of both bagging and Random Forests) showed OOB error closely tracks what you'd get from N-fold cross-validation, at a fraction of the compute cost (no need to retrain $k$ separate models — OOB comes for free as a byproduct of the bagging you're already doing).

---

## 3.4 When Bagging Helps vs. When It Doesn't

Directly from the Chapter 2.2 formula $\text{Var}(\hat f_{\text{avg}}) \to \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}$:

**Bagging helps a lot when:**
- The base learner has **high variance, low bias** (deep/unpruned trees are the canonical example) — there's substantial $\sigma^2$ for the formula to reduce.
- Bootstrap resampling actually induces meaningful diversity between trees, i.e., $\rho$ isn't too close to 1 — for trees, small changes in which samples appear (especially near the root, per Chapter 1.5's discussion of instability) cascade into structurally different trees, giving genuinely useful decorrelation.

**Bagging helps little or not at all when:**
- The base learner is already **low-variance** (e.g., a pruned/shallow tree, or a linear model). Per the 2.2 derivation, averaging doesn't touch bias, and there's little variance left to reduce — you pay the full $M\times$ compute cost of training and predicting with $M$ models for a marginal accuracy gain. This is the concrete, formula-backed answer to "why don't people bag linear regression models" — a linear regression fit is already fairly stable across resamples of reasonably-sized data (low $\sigma^2$ to begin with), so there's little for bagging's variance-reduction mechanism to grab onto.
- The base learners are **highly correlated** ($\rho$ close to 1) regardless of individual variance — e.g., if the dataset is small enough that most bootstrap samples end up looking nearly identical, or if the base learner's fitting procedure is itself highly deterministic/insensitive to which exact rows are present. The formula's floor term $\rho\sigma^2$ dominates and adding more trees can't push past it (this is precisely the motivation for Random Forest's added feature-randomization in Ch.4 — an explicit mechanism to push $\rho$ down further than bootstrap resampling alone achieves).

**Bagging pitfalls / interview traps:**
- **"Does bagging reduce bias?"** No — provably not, per the exact bias term in the 2.2 derivation (bias of the average equals bias of an individual model). If your single trees are systematically biased (e.g., too shallow), bagging 1000 of them will not fix that; it will just average 1000 equally-biased predictions.
- **"Can you overfit by adding too many bagged trees?"** No, and this is a specifically defensible claim: since each tree is trained independently and predictions are simply averaged, adding more trees only ever pushes variance further down toward its floor ($\rho\sigma^2$) — it doesn't reintroduce any new source of overfitting the way adding more boosting rounds can (Ch.5). `n_estimators` in bagging is a compute/diminishing-returns knob, not a bias-variance tradeoff knob to tune carefully against overfitting.
- **"Is OOB error the same as k-fold CV error?"** Closely related but not identical — OOB uses a *variable, sample-dependent* subset of models per prediction (whichever trees happened to leave that sample out), whereas k-fold CV uses a *fixed*, deliberately-constructed held-out fold per prediction. They tend to agree closely in practice for reasonably large $M$, but OOB is a byproduct of bagging specifically, not a general-purpose validation protocol.

---

## sklearn Parameters — `BaggingClassifier` / `BaggingRegressor`

| Parameter | What it controls | Notes |
|---|---|---|
| `estimator` | Base learner to bag | Default `None` → `DecisionTreeClassifier`/`Regressor` (unpruned by default, matching 3.1's recommendation) |
| `n_estimators` | Number of base learners ($M$) | Default 10 — low by Random Forest standards (Ch.4 defaults to 100); per 3.4, more is (almost) always at least as good, bounded by the $\rho\sigma^2$ floor and compute cost |
| `max_samples` | Fraction/count of samples drawn per bootstrap draw | Default 1.0 (draw $n$ samples, matching classic bagging, Section 3.1). Values <1.0 (without replacement, `bootstrap=False`) is a variant sometimes called **pasting**. |
| `max_features` | Fraction/count of *features* sampled per base estimator (sampled once per estimator, applied to all its splits — different from Random Forest's per-*split* feature sampling, Ch.4) | Default 1.0 (all features). Setting <1.0 gives "Random Subspaces"; combining sample **and** feature subsampling is sometimes called "Random Patches." |
| `bootstrap` | Whether samples are drawn with replacement | Default `True` — the defining feature of bagging proper vs. pasting |
| `bootstrap_features` | Whether *features* are sampled with replacement | Default `False` |
| `oob_score` | Whether to compute the OOB error automatically (3.3) | Default `False` — set `True` to get `.oob_score_` after fitting, at the cost of some extra bookkeeping during training |
| `n_jobs` | Parallelism across the $M$ independent base-estimator fits | Bagging is **embarrassingly parallel** (Chapter 3.1's loop has no dependency between iterations $m$) — unlike boosting (Ch.5), which is inherently sequential and cannot be parallelized across rounds the same way. Setting `n_jobs=-1` uses all cores. |
| `random_state` | Seed for reproducible bootstrap draws | — |

**Why does `max_features` exist on `BaggingClassifier` at all if Random Forest already does feature subsampling?** `BaggingClassifier`'s `max_features` samples a *fixed* feature subset once per base estimator (used for every split within that tree), whereas Random Forest resamples the feature subset **fresh at every single split** within every tree (Chapter 4.1). The latter is a substantially stronger decorrelation mechanism — it's the specific difference that defines "Random Forest" as more than "bagging with `max_features<1.0`," and is why Random Forest gets its own dedicated chapter rather than being a footnote to Bagging.

---

## Quick Interview Q&A

**Q: Why is bagging embarrassingly parallel but boosting isn't?**
A: Each bagged tree is trained on an independently-drawn bootstrap sample with no reference to any other tree's output — the $M$ training runs in Section 3.1's loop have zero data dependency between them, so they can run simultaneously on separate cores/machines. Boosting's core mechanism (Ch.5) requires each new model to be trained on the *residual/reweighted errors of the current ensemble so far* — model $m+1$ literally cannot be defined until model $m$ exists, making the training loop inherently sequential.

**Q: If bagging can't overfit by adding trees, why not just set `n_estimators` extremely high always?**
A: Purely a compute/latency trade-off — training time, memory, and inference latency all scale roughly linearly with $M$, for accuracy gains that shrink toward zero (per the 2.2 formula's diminishing-returns numerical example) well before $M$ gets very large. There's no accuracy harm, but there's real cost with no matching benefit past the point of diminishing returns.

**Q: Give a case where bagging would visibly *not* help.**
A: Bagging a `max_depth=1` decision stump on a large, clean dataset. A depth-1 stump is already low-variance (there's only one possible split point family it can choose, so different bootstrap samples tend to agree on roughly the same split) and high-bias (it can only represent one threshold's worth of decision boundary). Per 3.4, bagging leaves bias untouched and there's little variance to reduce — the bagged ensemble's error will look barely better than a single stump's, despite $M\times$ the training cost.

---

**Next up: Chapter 4 — Random Forests (the extra decorrelation trick, feature importance, and the direct sklearn parameter comparison against plain Bagging).**
