# Bagging (Bootstrap Aggregating) 

---

## 0. Quick Map (read this first)

| Piece | Theoretical role | Practical payoff |
|---|---|---|
| Bootstrap sampling | Manufactures $M$ "different enough" training sets from one dataset | Trains many trees without collecting more data |
| Averaging | The variance-killing mechanism | Smooths out any one tree's instability |
| OOB error | Side effect of ~37% of rows being left out of each bootstrap sample | Free validation — no held-out split needed |
| Deep, unpruned trees as base learner | High-variance, low-bias — exactly what averaging fixes | Why bagging pairs with trees, not linear models |
| Correlation $\rho$ between trees | The term averaging *can't* remove | Explains why Random Forest exists at all |

**One-line theory:** Bagging is the most literal possible application of the variance-reduction formula — it does nothing clever to reduce bias, and nothing clever to decorrelate models beyond plain resampling. It's the "control group" ensemble method: understand it fully, and Random Forest is "bagging plus one extra decorrelation trick," and boosting is "the opposite strategy, aimed at the other term of the error decomposition."

**The single sentence an interviewer wants to hear:** *"Bagging trades compute for variance reduction by averaging independently-trained high-variance models; it can't touch bias, and its ceiling is set by how correlated the models end up being."*

---

## 1. The Big Idea (in one paragraph)

You have one model that's jumpy — small changes in the training data swing its predictions a lot (high variance). Instead of training it once, you train it many times on slightly different "reshuffled" versions of the same data, and average the results. Averaging smooths out the jumpiness. That's it — that's bagging.

The classic base learner is a **deep, unpruned decision tree**, because trees are famously unstable (change a few rows, get a very different tree) — which makes them the perfect candidate for variance-smoothing.

### 1.1 A practical analogy (good for interviews)

> Imagine asking one person to estimate a jar of jellybeans by eye — their guess depends heavily on which part of the jar they glanced at, how the light hit it, and pure luck. Ask the *same type* of person to guess 50 times, each time glancing at a slightly different, randomly-reshuffled subset of beans, then average all 50 guesses. No single guess got smarter, but the average is far more stable — because the guesses' *errors* (not their skill) are what's getting averaged out.

### 1.2 Why this is worth an interviewer's time

Bagging is the cleanest possible illustration of the bias-variance tradeoff in action — it isolates the variance term and shows you exactly what happens when you attack it in isolation, with nothing else changing. That's why it's a favorite whiteboard question: it tests whether you actually understand the decomposition, or just memorized "ensembles are good."

---

## 2. The Algorithm (simplified)

**Step 1 — Make $M$ "reshuffled" datasets.**
For each of $M$ rounds, build a new dataset the same size as the original by sampling **with replacement** (so some rows get picked multiple times, and ~37% of rows get left out entirely, on average).

**Step 2 — Train one tree per dataset.**
Grow each tree deep, don't prune it.

**Step 3 — Combine the $M$ trees.**
- Regression → average the predictions.
- Classification → majority vote, or average predicted probabilities and pick the top class ("soft voting" — sklearn's default, usually a bit better because it uses more information).

| Term | Plain meaning |
|---|---|
| Bootstrap sample | A resampled dataset of the same size, drawn with replacement |
| $M$ | Number of trees you train |
| Bagging | Bootstrap + Aggregating = resample + combine |

### 2.1 Theory — why not prune the trees?

Pruning already makes a tree stable (low variance) but worse at fitting (higher bias). Bagging's whole job is *removing variance* — it can't fix bias at all. So if you prune first, you've thrown away the exact thing bagging was going to fix, and you're left averaging a bunch of similarly-mediocre, similarly-biased trees for no benefit.

> **Interview soundbite:** *"Pruning before bagging is like hiring a decorrelation specialist and then giving them models that were already stable — you've solved the wrong half of the problem before the tool designed to solve the other half even gets to run."*

### 2.2 Why exactly ~37%? (derive it, don't memorize it)

For a dataset of size $n$, the probability a specific row is *not* chosen in one draw is $\frac{n-1}{n}$. Draw $n$ times with replacement (to build a bootstrap sample the same size as the original):

$$P(\text{row never chosen}) = \left(\frac{n-1}{n}\right)^n = \left(1-\frac{1}{n}\right)^n \xrightarrow[n\to\infty]{} e^{-1} \approx 0.368$$

So ~36.8% of rows are left out of any given bootstrap sample, and ~63.2% are included (with duplicates making up the rest of the $n$ slots). This is the exact number behind "~37% OOB" and it's a clean derivation to reproduce on a whiteboard — interviewers love that this is a fully deducible constant, not a magic number.

> **Interview soundbite:** *"The 37% isn't an empirical rule of thumb — it falls straight out of $(1-1/n)^n \to e^{-1}$ as $n$ grows. If you can derive that on a whiteboard, you've shown you understand bootstrap sampling, not just quoted it."*

---

## 3. Formal Bias-Variance Mechanics (the part interviewers actually probe)

Let $\hat f_1, \dots, \hat f_M$ be trees trained on $M$ bootstrap samples, each with variance $\sigma^2$ and pairwise correlation $\rho$ (assume identical marginal variance across trees — a standard simplifying assumption). The bagged predictor is $\hat f_{\text{bag}} = \frac{1}{M}\sum_m \hat f_m$.

**Bias:**
$$\mathbb{E}[\hat f_{\text{bag}}] = \frac{1}{M}\sum_m \mathbb{E}[\hat f_m] = \mathbb{E}[\hat f_1]$$
Averaging is a linear operation on unbiased-in-expectation identically-distributed estimators — the expected value of the average equals the expected value of one tree. **Bias is completely unchanged by bagging.** This is not an approximation; it falls straight out of linearity of expectation and identical distribution across trees.

**Variance:**
$$\text{Var}(\hat f_{\text{bag}}) = \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}$$

Derivation sketch: $\text{Var}\left(\frac{1}{M}\sum \hat f_m\right) = \frac{1}{M^2}\left[\sum_m \text{Var}(\hat f_m) + \sum_{m\ne m'}\text{Cov}(\hat f_m,\hat f_{m'})\right] = \frac{1}{M^2}\left[M\sigma^2 + M(M-1)\rho\sigma^2\right]$, which simplifies to the formula above as $M\to\infty$ terms cancel appropriately.

**Two regimes worth internalizing:**
- As $M \to \infty$: the second term vanishes, leaving a **floor of $\rho\sigma^2$** — you can never average your way below this, no matter how many trees you add.
- If $\rho = 0$ (impossible in practice, but instructive): variance would shrink to $\sigma^2/M$ — approaching zero as $M$ grows. This is the theoretical ceiling of what resampling-based decorrelation could ever buy you.
- If $\rho = 1$ (trees are identical, e.g. deterministic base learner with no randomness): variance stays exactly at $\sigma^2$ regardless of $M$ — bagging does *nothing*.

> **Interview soundbite:** *"There are exactly two knobs in this formula, $M$ and $\rho$. $M$ is free — just train more trees. $\rho$ is the hard part, and it's precisely the thing Random Forest was invented to push down further than plain resampling can."*

### 3.1 Why trees specifically achieve low $\rho$

Bootstrap resampling only changes model structure meaningfully if the base learner is **sensitive to which rows it sees** — i.e., high-variance to begin with. A deep unpruned tree's very first split can flip entirely if you swap out a handful of rows near a decision boundary, which cascades into a completely different tree downstream. A shallow model (or one heavily regularized) barely notices the same row swaps, so its bootstrap replicas stay highly correlated with each other — high $\rho$, and correspondingly little to gain. This is the mechanical link between "trees are unstable" and "trees are the classic bagging base learner": instability *is* what keeps $\rho$ low enough for averaging to pay off.

---

## 4. Worked Numerical Example #1 (original, regression)

Toy dataset ($n=6$):

| i | x | y |
|---|---|---|
| 1 | 1 | 10 |
| 2 | 2 | 12 |
| 3 | 3 | 20 |
| 4 | 4 | 22 |
| 5 | 5 | 30 |
| 6 | 6 | 32 |

Draw $M=3$ bootstrap samples:

| Tree | Sampled indices (with replacement) | Left-out (OOB) indices |
|---|---|---|
| $D_1$ | 1,1,3,4,5,6 | 2 |
| $D_2$ | 2,2,3,3,5,6 | 1,4 |
| $D_3$ | 1,2,4,4,5,5 | 3,6 |

Each tree is a simple stump. For a test point $x_0 = 3.5$, the three trees predict:

$$\hat f_1(x_0)=21.0 \quad \hat f_2(x_0)=24.5 \quad \hat f_3(x_0)=19.0$$

**Bagged (averaged) prediction:**
$$\hat f_{\text{bag}}(x_0) = \frac{21.0+24.5+19.0}{3} = 21.5$$

If the true value is $21.0$: the individual trees were off by $0,\ 3.5,\ 2.0$ — a wide spread. The averaged prediction is off by only $0.5$, smaller than two of the three individual trees. One data point isn't proof, but it's exactly the smoothing effect you'd expect on average across many points.

### 4.1 Extension — variance of this toy ensemble, numerically

The three predictions $\{21.0, 24.5, 19.0\}$ have sample mean $21.5$ and sample variance $\approx \frac{(0.5)^2+(3.0)^2+(2.5)^2}{3} = \frac{0.25+9.0+6.25}{3} \approx 5.17$. A single tree's squared error against the mean averaged $5.17$; the ensemble's squared error against the true value ($0.5^2=0.25$) is far smaller — a concrete (if tiny-$n$) illustration of the $\rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}$ shrinkage from Section 3, well below the $M=1$ single-tree spread.

---

## 5. Worked Numerical Example #2 (classification, soft vs. hard voting)

Predicting "will this house sell within 30 days" (1 = yes, 0 = no) for a new house $x_0$, using $M=5$ bagged trees. Each tree outputs a probability (soft voting):

| Tree | $D_1$ | $D_2$ | $D_3$ | $D_4$ | $D_5$ |
|---|---|---|---|---|---|
| $P(\text{sells fast})$ | 0.72 | 0.55 | 0.30 | 0.68 | 0.61 |

**Soft voting (sklearn default):** average the probabilities.
$$\bar P = \frac{0.72+0.55+0.30+0.68+0.61}{5} = \frac{2.86}{5} = 0.572$$

At the default 0.5 threshold → predict **class 1 (sells fast)**.

**Hard/majority voting**, for comparison: convert each tree's probability to a class first (threshold 0.5), *then* vote.
- Classes: 1, 1, 0, 1, 1 → 4 votes for class 1, 1 vote for class 0 → majority says **class 1** too, here.

**Why they can disagree in general:** imagine tree $D_3$'s probability had instead been 0.49 instead of 0.30 — hard voting still counts it as a single "0" vote (same weight as a confident "0.02"), while soft voting would barely move the average, since 0.49 is nearly a wash against the threshold. **Soft voting uses the *magnitude* of each tree's confidence; hard voting throws that information away and only keeps the final class label.**

> **Interview soundbite:** *"Soft voting is almost always the better default because it preserves confidence information hard voting discards — a tree that's 51% sure and a tree that's 99% sure get treated identically by majority vote, but very differently by averaging probabilities."*

### 5.1 When hard voting actually wins

Soft voting assumes each tree's predicted probability is **reasonably calibrated** — that "0.9" from tree A means roughly the same thing as "0.9" from tree B. Deep unpruned trees are notoriously **poorly calibrated** (leaves with few samples produce overconfident 0/1-ish probabilities). If calibration is bad and inconsistent across trees, averaging raw probabilities can be *worse* than a simple vote, because you're averaging numbers that don't mean the same thing across models. Practical fix: calibrate each tree (e.g., Platt scaling / isotonic regression) before soft-voting, or fall back to hard voting if calibration isn't feasible.

> **Interview soundbite:** *"Soft voting is only as good as the calibration of the probabilities you're averaging — average garbage-calibrated probabilities and you can do worse than a plain majority vote."*

---

## 6. Out-of-Bag (OOB) Error — "Free" Validation

### 6.1 The theory

**Idea:** For each row $i$, some trees never saw it during training (it was "out of bag" for them, ~37% of trees on average). Use *only those* trees to predict row $i$. Since those trees never trained on row $i$, this prediction behaves like a held-out/test prediction — you get validation performance without setting aside a validation set.

$$\hat y_i^{\text{OOB}} = \text{aggregate of } \hat f_m(x_i) \text{ over every tree } m \text{ that didn't see row } i$$

$$\text{OOB Error} = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat y_i^{\text{OOB}})$$

### 6.2 Worked numerical (one row)

Using the table above, sample 1's OOB set is just $\{D_2\}$ — one single tree. Say $D_2$'s tree predicts $\hat f_2(x_1) = 11.2$:
$$\hat y_1^{\text{OOB}} = 11.2, \qquad \text{true } y_1 = 10, \qquad \text{squared error} = (10-11.2)^2 = 1.44$$

### 6.3 Worked numerical — full 6-row pass

| Row $i$ | Trees where $i$ is OOB | OOB predictions | True $y_i$ | OOB prediction $\hat y_i^{\text{OOB}}$ | Squared error |
|---|---|---|---|---|---|
| 1 | $D_2$ | 11.2 | 10 | 11.2 | 1.44 |
| 2 | $D_1$ | 12.9 | 12 | 12.9 | 0.81 |
| 3 | $D_2, D_3$ | 21.5, 18.7 | 20 | avg = 20.1 | 0.01 |
| 4 | $D_2$ | 21.0 | 22 | 21.0 | 1.00 |
| 5 | *(none — in-bag for all 3 trees)* | — | 30 | undefined | *excluded* |
| 6 | $D_3$ | 33.1 | 32 | 33.1 | 1.21 |

**OOB MSE** (averaging over the 5 rows with at least one OOB voter): $\frac{1.44+0.81+0.01+1.00+1.21}{5} = \frac{4.47}{5} = 0.894$

**Row 5 had to be excluded** — it landed in-bag for all 3 trees in this tiny example, an instance of the "small $M$ is noisy" catch below. At realistic $M$ (hundreds of trees), $P(\text{row always in-bag}) \approx (1-0.368)^M$ is under 1% by $M \approx 15$, so this edge case essentially disappears in practice.

### 6.4 Why is this useful compared to a normal train/validation split?

| | Train/validation split | OOB |
|---|---|---|
| Data used to train final model | ~80% (rest held out forever) | 100% |
| Extra models needed | 0 | 0 (comes free from trees you already trained) |
| "Held-out" portion | One fixed chunk | Different for every row, spread across trees |

**Catch:** with small $M$, some rows have only 1 OOB voter — noisy. You need $M$ in the hundreds (typical Random Forest default) before OOB error becomes trustworthy.

> **Interview soundbite:** *"OOB error at small $M$ is like asking one juror to render a verdict — technically 'held-out,' but a sample of one. You need enough trees before OOB behaves like a jury, not a single opinion."*

### 6.5 OOB vs. k-fold CV — precise comparison

| | OOB | k-fold CV |
|---|---|---|
| Extra training needed | None (byproduct of the ensemble you already built) | $k$ full retrains |
| Held-out set per row | Variable-size, random subset of trees | One fixed fold |
| Works for any model? | No — only ensembles built from resampling (bagging, RF) | Yes — model-agnostic |
| Bias of the estimate | Slightly pessimistic at small $M$ (each row averaged over fewer, "weaker" sub-ensembles than the full $M$) | Unbiased estimate of the $k$-fold-sized model's performance |
| Typical use | "Free" sanity check during/after training an ensemble | Standard estimate for any model type, including non-ensembles |

> **Interview soundbite:** *"OOB isn't a replacement for CV in general — it's a shortcut available specifically because bagging already gives you $M$ sub-models with built-in held-out rows for free. For a model that isn't built from resampling, there's no OOB to compute."*

---

## 7. When Bagging Helps vs. Doesn't

Comes straight from the variance formula: $\text{Var}(\hat f_{\text{avg}}) = \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}$

**Helps a lot when:**
- Base learner has **high variance** (deep/unpruned trees) — there's a lot of $\sigma^2$ to reduce.
- Trees end up meaningfully different from each other ($\rho$ isn't close to 1) — resampling actually changes tree structure.

**Helps little when:**
- Base learner is already **low-variance** (shallow tree, linear model). Bagging can't touch bias, and there's barely any variance left to remove — you pay $M\times$ the compute for almost nothing. This is *why nobody bags linear regression*.
- Trees end up **highly correlated** ($\rho \approx 1$) — e.g. dataset is tiny so all bootstrap samples look almost the same. The $\rho\sigma^2$ floor term dominates, and adding more trees can't push below it. (This is the exact motivation for Random Forest's extra feature-randomization.)

### 7.1 Worked numerical — quantifying "helps little"

Take a shallow stump (max_depth=1) with $\sigma^2 = 0.3$ (already low variance) and $\rho = 0.6$ (bootstrap samples don't change a shallow stump's single split much). At $M=100$:

$$\text{Var}(\hat f_{\text{bag}}) = 0.6\times0.3 + \frac{(1-0.6)\times0.3}{100} = 0.18 + 0.0012 = 0.1812$$

Compare to a single stump's variance, $0.3$. Bagging bought a drop from 0.3 → 0.1812 — real but modest, and **99% of that reduction was already captured by $M=100$** (max possible reduction is $0.3-0.18=0.12$; achieved so far is $0.3-0.1812=0.1188$). Meanwhile bias is completely unchanged — if Bias² was, say, 9.0, total error barely moved: from $9.3$ to $9.1812$, a **2% relative improvement**, for 100× the training/serving cost.

### 7.2 Worked numerical — quantifying "helps a lot" (NEW, the contrasting case)

Now take a deep unpruned tree with $\sigma^2 = 8.0$ (genuinely high variance) and $\rho = 0.2$ (deep trees restructure substantially across bootstrap draws). At $M=100$:

$$\text{Var}(\hat f_{\text{bag}}) = 0.2\times8.0 + \frac{(1-0.2)\times8.0}{100} = 1.6 + 0.064 = 1.664$$

Single-tree variance was $8.0$; bagged variance is $1.664$ — roughly a **79% reduction**. If Bias² is small (say $0.5$, since deep trees barely underfit), total error goes from $8.5 \to 2.164$, a **~75% relative improvement**. Put side-by-side with Section 7.1's 2% improvement for the stump, this is the cleanest possible numeric contrast for "bagging's payoff is entirely a function of how much $\sigma^2$ there was to remove in the first place."

> **Interview soundbite:** *"Same algorithm, same formula, two base learners — one gets a 2% improvement, the other gets 75%. The only thing that changed was how much variance was sitting there to begin with. That's the whole 'why trees, not stumps or linear models' argument in one comparison."*

---

## 8. Bagging vs. Random Forest vs. Boosting vs. Pasting — full comparison (NEW)

| Dimension | Bagging | Random Forest | Boosting (e.g. AdaBoost/GBM) | Pasting |
|---|---|---|---|---|
| Row sampling | Bootstrap (with replacement) | Bootstrap (with replacement) | Full dataset, but reweighted/re-gradient each round | Sampling **without** replacement |
| Feature sampling | Optional, once per tree if used | Fresh random subset **at every split** | Typically all features (unless subsampled, e.g. XGBoost `colsample`) | Optional, once per tree if used |
| Trees trained | Independently, in parallel | Independently, in parallel | Sequentially — each depends on the last | Independently, in parallel |
| What's reduced | Variance only | Variance (more aggressively than bagging) | Primarily bias (and some variance, depending on the method) | Variance only |
| Overfitting from more rounds? | No (`n_estimators` is a compute knob) | No | Yes — possible with too many boosting rounds | No |
| Sensitive to noisy/outlier labels? | Moderately (bootstrap can amplify or dilute an outlier's row weight, but no re-weighting toward errors) | Moderately | Highly (repeatedly up-weights hard/misclassified points, which can include mislabeled data) | Moderately |
| Parallelizable? | Fully | Fully | No (sequential dependency) | Fully |
| Typical base learner | Deep, unpruned tree | Deep, unpruned tree | Shallow tree ("weak learner") | Deep, unpruned tree |
| When it shines | High-variance base learner, want robustness + parallel training | Same as bagging, plus want to push correlation down further | Want to squeeze maximum accuracy from clean data, can afford sequential training | Very large datasets where "with replacement" duplication cost isn't worth it |

> **Interview soundbite:** *"If someone asks 'isn't Random Forest just bagging?' — the precise answer is: bagging decorrelates trees only through row resampling; Random Forest adds a second, stronger decorrelation lever by re-randomizing the feature subset at every split, not just once per tree. That's the whole delta between the two algorithms."*

> **Interview soundbite:** *"Bagging and boosting attack opposite terms of the same decomposition — bagging reduces variance by averaging independent high-variance learners; boosting reduces bias by sequentially correcting a chain of weak (high-bias) learners. Neither is a 'better' ensemble method in general — they're solutions to opposite problems."*

---

## 9. sklearn Cheat Sheet — `BaggingClassifier` / `BaggingRegressor`

| Parameter | What it does | Default |
|---|---|---|
| `estimator` | Base learner | `None` → unpruned decision tree |
| `n_estimators` | Number of trees $M$ | 10 (low — Random Forest defaults to 100) |
| `max_samples` | Rows drawn per bootstrap | 1.0 (full $n$, with replacement) |
| `max_features` | Features sampled **once per tree**, reused for all its splits | 1.0 (all features) |
| `bootstrap` | Sample rows with replacement? | `True` |
| `bootstrap_features` | Sample features with replacement? | `False` |
| `oob_score` | Auto-compute OOB error? | `False` |
| `n_jobs` | Parallel training across trees | — |
| `random_state` | Reproducibility | — |

**`max_features` vs. Random Forest's feature sampling — the key distinction:**
`BaggingClassifier` picks a feature subset **once per tree** (same subset used for every split in that tree). Random Forest re-picks a **fresh random subset at every single split**. The latter decorrelates trees much more aggressively — the one specific mechanism that makes Random Forest more than "bagging with fewer features."

### 9.1 Practical hyperparameter tuning guide (NEW)

| Symptom | Likely cause | What to try |
|---|---|---|
| Train and test accuracy both low | Bias problem — bagging can't fix this | Switch to deeper trees, or a lower-bias base learner; bagging is the wrong lever entirely |
| Train accuracy high, test accuracy much lower, and bagging barely narrows the gap | $\rho$ is high — trees aren't diversifying | Add `max_features < 1.0`, reduce `max_samples` slightly, or switch to Random Forest for per-split feature randomization |
| OOB score noisy / jumps around as you add trees | $M$ too small | Increase `n_estimators` until the OOB curve visibly flattens |
| Training/serving too slow | $M$ too large relative to marginal benefit | Plot OOB error vs. $M$; trim to just past the flattening point |
| OOB score much worse than a held-out test set score | Possible data leakage in the held-out set, or `max_samples` set so high that too few rows are ever truly OOB | Check `max_samples`; verify the external test set wasn't touched during any preprocessing fit |

---

## 10. Common Interview Traps

| Claim | True? | Why |
|---|---|---|
| Bagging reduces bias | ❌ No | The bias of the average = bias of a single model, exactly (Section 3). Biased trees stay biased no matter how many you average. |
| Adding more trees can overfit | ❌ No | Trees are trained independently; more trees only ever pushes variance down toward its floor. `n_estimators` is a compute knob, not an overfitting knob (unlike boosting rounds). |
| OOB error = k-fold CV error | ≈ Close, not identical | OOB uses a *different, variable* subset of trees per row; k-fold uses one *fixed* held-out fold per row (Section 6.5). They agree closely for large $M$. |
| Bagging always beats a single tree | ❌ No | Only if the single tree has meaningful variance to begin with (Section 7.1 vs. 7.2). A well-regularized single tree may already be competitive. |
| Random Forest = Bagging with `max_features` set less than 1.0 | ❌ Close, but not exact | `BaggingClassifier(max_features<1)` samples a feature subset **once per tree**; Random Forest resamples features **at every split**. Materially different amount of decorrelation (Section 8). |
| More bootstrap diversity is always better | ❌ Not unconditionally | Too aggressive row/feature subsampling (very low `max_samples`/`max_features`) can starve individual trees of enough signal, raising each tree's own bias — there's a sweet spot, not a monotonic "more randomness = better" relationship. |

---

## 11. Related Methods — Quick Reference (NEW)

Interviewers sometimes probe adjacent methods to see if you understand bagging's *generalization*, not just the one named algorithm.

| Method | Relationship to bagging |
|---|---|
| **Random Forest** | Bagging + per-split feature randomization (Section 8) |
| **Extra Trees (Extremely Randomized Trees)** | Random Forest, but also randomizes the **split threshold** itself (not just which feature), pushing $\rho$ down even further at the cost of slightly higher per-tree bias |
| **Pasting** | Bagging's row sampling done **without** replacement — used when the dataset is so large that "with replacement" duplication isn't needed to get useful diversity |
| **Random Subspaces** | Bagging's *feature* analogue: keep all rows, but randomly sample the feature set per model (the same idea `BaggingClassifier(max_features<1, bootstrap=False)` implements) |
| **Isolation Forest** | Reuses bagging's "many independent randomized trees" machinery, but repurposes tree *path length* (not majority vote) as an anomaly score — a good example of the base infrastructure being reused for a completely different objective |
| **Stacking** | A different ensembling philosophy entirely: instead of *averaging* independently-trained models of the *same* type, it trains a meta-model to *learn how to combine* outputs from different model types — not a variance-reduction argument at all |

> **Interview soundbite:** *"Once you see bagging as 'inject randomness, train independently, average,' Random Forest, Extra Trees, and Random Subspaces are all just different choices of where you inject the randomness — rows, features, or split thresholds. It's one idea with three knobs."*

---

## 12. Whiteboard One-Pager (NEW — for the last 60 seconds before you walk in)

1. **What it is:** resample with replacement → train $M$ independent high-variance learners → average.
2. **What it fixes:** variance only. $\text{Bias}(\hat f_{\text{bag}}) = \text{Bias}(\hat f_1)$, exactly.
3. **The formula:** $\text{Var} = \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}$. Two knobs: $M$ (free, just costs compute) and $\rho$ (hard — this is what RF improves on).
4. **Why trees:** they're unstable (high $\sigma^2$, resampling meaningfully changes structure → low $\rho$) — the ideal base learner for this trick.
5. **Why not prune first:** pruning removes the variance bagging exists to remove.
6. **OOB:** ~37% of rows left out per tree ($\to e^{-1}$ as $n\to\infty$) → free validation, needs $M$ in the hundreds to be reliable.
7. **vs. Random Forest:** RF re-randomizes features at every split, not once per tree — a stronger decorrelation lever.
8. **vs. Boosting:** opposite target (bias, not variance), sequential not parallel, more sensitive to label noise, can overfit with too many rounds (bagging can't).
9. **Inference cost:** $M\times$ a single tree, always — bagging's parallelism benefit is training-time only.

---

## 13. Quick Q&A (general, easy warm-ups)

**Q: Why is bagging embarrassingly parallel but boosting isn't?**
A: Each tree only depends on its own bootstrap sample — no tree needs to know what any other tree did. Boosting's next model is trained on the *current ensemble's errors*, so model $m+1$ literally can't exist before model $m$ does.

**Q: If more trees never hurts, why not set `n_estimators` huge always?**
A: Pure compute/latency cost for shrinking returns — no accuracy downside, but no point paying for trees past where the curve flattens.

**Q: Give an example where bagging visibly doesn't help.**
A: Bagging a depth-1 stump on a large clean dataset. A stump is already low-variance (few possible splits to disagree on) and high-bias (can only represent one threshold). Bagging leaves bias untouched — the ensemble barely beats a single stump, despite $M\times$ the cost (see Section 7.1's 2% number).

---

## 14. Google MLE Interview Q&A

**Q: You're told a Random Forest and a single deep decision tree get nearly the same training accuracy, but very different test accuracy. Explain why, using bagging's mechanics.**
A: Training accuracy mainly reflects bias, and bagging doesn't change bias — so it's expected they'd be similar there. Test accuracy reflects bias *and* variance; the single deep tree has overfit to its specific training rows (high variance, low bias on train), while the forest averaged away most of that variance, so its test performance holds up much better even though its training fit looked "equal."

**Q: How would you use OOB error to pick `n_estimators` without a separate validation set, and what's a failure mode of doing this?**
A: Plot OOB error against $M$ as you grow the forest incrementally — it should decrease and then flatten. Failure mode: at small $M$, OOB error is noisy (few OOB voters per row), so you can mistake noise for a real plateau or a real improvement. Don't make a stopping decision on a jumpy early-$M$ curve.

**Q: Design question — you have a massive dataset that doesn't fit in memory on one machine. How does bagging's structure help you here?**
A: Since each tree's training is fully independent, you can shard the bootstrap sampling and tree training across machines — each worker draws its own bootstrap sample (or a sample from a distributed store) and trains one or more trees, and you only need to gather the final trees for prediction-time aggregation. This is a direct consequence of bagging having zero sequential dependency between rounds.

**Q: A colleague says "bagging is basically a crude version of ensembling, we should just always prefer boosting since it usually gets higher accuracy." How do you push back?**
A: Boosting often does win on accuracy, but it comes with different failure modes: boosting is sequential (can't parallelize across rounds), more sensitive to noisy labels/outliers (since it keeps re-weighting toward hard/misclassified points), and can overfit if you add too many rounds — none of which is true for bagging. The right framing isn't "bagging is a lesser boosting," it's a trade-off: bagging for parallelism, robustness to noisy labels, and a training loop that's structurally overfitting-proof in $M$; boosting when you can afford sequential training and want to squeeze out more accuracy from a clean dataset.

**Q: (NEW) Your team's Random Forest OOB score looks great, but production accuracy is noticeably worse. Walk through how you'd debug this, tying it back to what OOB actually measures.**
A: First check for train/serve skew unrelated to bagging (feature drift, a preprocessing step fit on all data including "future" leakage, etc.) — OOB error is only as trustworthy as the assumption that OOB rows are truly unseen and identically distributed to production data. Specifically check: (1) was any preprocessing (scaling, target encoding, imputation) fit on the *full* dataset before bagging, which would leak information into every tree's OOB rows too, silently inflating OOB score; (2) is production data distributionally different from training data (covariate shift) — OOB can only ever reflect the training distribution; (3) is `max_samples` set high enough that OOB voters per row are too few to be a stable estimate (Section 6.3's noise problem). OOB isn't wrong when this happens — it's answering a narrower question ("how well does this ensemble generalize *within the training distribution*") than the one production performance is actually asking.

**Q: (NEW) How would you estimate feature importance from a bagged ensemble of trees, and what's a known pitfall?**
A: The standard approach is either (a) mean decrease in impurity across all trees' splits on that feature, or (b) permutation importance — shuffle one feature's values (ideally on OOB rows specifically, so you're not reusing training rows) and measure how much OOB error degrades. Pitfall: impurity-based importance is biased toward high-cardinality / continuous features, which get more opportunities to produce a "good-looking" split purely by having more possible thresholds to try, even if they're not truly more predictive. Permutation importance (especially computed on OOB data) is generally more trustworthy for comparing features fairly.

---

## 15. Apple MLE Interview Q&A (on-device / practical flavor)

**Q: You want an on-device ensemble model (e.g., inside Core ML) for a latency-sensitive feature. Would you reach for bagging, and what's the practical trade-off?**
A: Bagging's inference cost scales linearly with $M$ — you run all $M$ trees and aggregate, every single prediction, on every request. On a phone/watch, that's $M\times$ the compute and memory footprint compared to a single tree, which matters a lot more on-device than in a data center. In practice this pushes you toward either a small $M$ (accepting less variance reduction), a much shallower/lighter base learner, or switching to a single well-regularized model or a distilled model rather than shipping the full ensemble — bagging's parallelism benefit (useful for training) doesn't help you at inference time on a single constrained device.

**Q: How does bagging's training-time parallelism map onto a Private Cloud Compute–style setup, where training might happen off-device before a model is distributed to devices?**
A: Since each tree trains independently on its own bootstrap sample, training can be fully parallelized across the compute cluster used before distribution — this is a training-time-only benefit and doesn't change anything about the on-device footprint discussed above; you still ship (and pay the inference cost for) all $M$ trained trees to the device afterward, unless you prune the ensemble down or distill it into something smaller first.

**Q: If you were using bagging as part of a privacy-sensitive pipeline (e.g., data that can't leave a device, differential-privacy constraints), what does bootstrap sampling interact with?**
A: Bootstrap sampling means the same row can be selected multiple times in one bootstrap draw and each tree sees a different resampled subset — if you're layering differential privacy on top, that resampling changes how many times any individual record influences a given tree's output, which affects your privacy-budget accounting per tree. It's not something bagging handles automatically; whatever DP mechanism you use has to be aware that "one bootstrap sample" isn't the same as "one pass over each record exactly once."

**Q: OOB error gave you a validation-free way to estimate error — is that still meaningful in a federated-learning setting where trees might be trained across many separate on-device datasets?**
A: The core requirement for OOB — that some trees genuinely never saw a given row — still holds as long as each device's local bootstrap sample leaves out some of that device's own local rows. But OOB in a federated setup only estimates per-device local error unless predictions and OOB bookkeeping are aggregated back centrally, which itself is more coordination than plain on-device bagging assumes; it's a case where the "free validation" framing needs to be revisited rather than assumed to transfer directly.

**Q: (NEW) A model needs to update as new on-device data streams in, without a full retrain. Is bagging naturally compatible with online/incremental updates?**
A: Not cleanly — each tree in a bagged ensemble is trained on a fixed bootstrap snapshot of the data it saw at training time, and standard decision tree induction isn't naturally incremental (splits are chosen greedily using the full sample available at that node). The practical options are: periodically retrain a subset of trees (e.g., replace the oldest $k$ of $M$ trees on a rolling schedule) rather than the whole ensemble at once, use an incremental-tree variant (e.g., Hoeffding/streaming trees) as the base learner instead of standard CART, or accept periodic full retrains and treat bagging as a batch method. The parallelism that makes bagging attractive for *batch* training doesn't translate into a native online-learning story.

---

## 16. Interview-Ready Soundbites (collected in one place)

1. *"Bagging is the control-group ensemble method — no cleverness beyond resample-and-average. Understand it fully and Random Forest is just 'bagging plus one extra decorrelation trick,' boosting is 'the opposite strategy, aimed at the other term.'"*
2. *"Pruning before bagging solves the wrong half of the problem — you've stabilized a model right before handing it to a tool whose entire purpose was stabilizing it."*
3. *"Soft voting keeps the magnitude of each tree's confidence; hard voting throws that information away and keeps only the label. That's why soft voting is the better default almost everywhere — as long as the probabilities are actually calibrated."*
4. *"OOB error at small $M$ is a jury of one — you need enough trees for it to behave like a jury instead of an opinion."*
5. *"`n_estimators` is a compute knob, not an overfitting knob — trees train independently, so more of them can only push variance further down toward its floor, never up."*
6. *"Bagging's parallelism is a training-time property. At inference, on a single device, you still pay for every one of the M trees — the free lunch is on the training side only."*
7. *"There are exactly two knobs in the variance formula: $M$, which is free, and $\rho$, which is hard. Every named variant of bagging — Random Forest, Extra Trees, Random Subspaces — is just a different way of attacking $\rho$."*
8. *"The 37% OOB figure isn't folklore — it's $(1-1/n)^n \to e^{-1}$, fully derivable on a whiteboard."*
9. *"Bagging and boosting aren't rivals on the same axis — they attack opposite terms of the same decomposition. Comparing them on 'which is better' misses that they solve different problems."*
10. *"Same formula, two base learners, wildly different payoff: bagging a stump buys ~2% total error reduction; bagging a deep tree buys ~75%. The formula doesn't change — the amount of variance sitting there to remove does."*

---

## 17. Practice Q&A

**Q1 (easy).** Why does `BaggingRegressor`'s default `n_estimators=10` usually get bumped up in practice?
<details><summary>Answer</summary>Per the variance formula, $M=10$ still leaves a meaningful chunk of the shrinking term $\frac{(1-\rho)\sigma^2}{M}$ un-shrunk — going from 10 to, say, 100+ trees (Random Forest's typical default) captures most of the remaining achievable variance reduction. 10 trees is a reasonable quick baseline but rarely the endpoint.</details>

**Q2 (easy).** A single decision tree gets 85% train accuracy and 60% test accuracy. After bagging with 200 trees, train accuracy is 84% and test accuracy is 79%. What happened?
<details><summary>Answer</summary>Classic variance reduction at work: bias stayed roughly the same (train accuracy 85%→84%, essentially unchanged, consistent with bagging not touching bias), while the large train/test gap — the fingerprint of high variance — shrank substantially (25 points → 5 points) because averaging 200 trees smoothed out the instability that was hurting generalization.</details>

**Q3 (medium).** Using Section 6.3's OOB table, why did row 5 end up excluded, and what does that imply about choosing $M$ in real settings?
<details><summary>Answer</summary>Row 5 happened to be sampled into every one of the 3 bootstrap training sets, leaving it with zero OOB voters — possible at small $M$ purely by chance ($\approx (1-0.368)^M$ probability of a row landing in-bag everywhere, non-negligible at $M=3$). In real settings with $M$ in the hundreds, this probability becomes vanishingly small, so essentially every row ends up with a usable OOB estimate.</details>

**Q4 (medium).** Why can't you compute OOB error for a `BaggingRegressor` where `max_samples` is set so high that no rows are ever left out?
<details><summary>Answer</summary>OOB error fundamentally depends on some rows being excluded from some trees' training sets — if `max_samples` samples the entire dataset without leaving room for exclusions (or if sampling is done without replacement, so all $n$ rows always appear), there's no "unseen" subset per tree to compute a held-out prediction from. OOB is a direct byproduct of the with-replacement exclusion mechanic; remove that mechanic and OOB has nothing to work with.</details>

**Q5 (hard).** A teammate wants to bag an XGBoost model (i.e., train several independent XGBoost models on different bootstrap samples and average them). Is this a reasonable idea?
<details><summary>Answer</summary>It's unusual but not nonsensical — it depends on what XGBoost's own variance looks like after its typical regularization (shallow trees, learning rate, etc.). A well-tuned, regularized boosted model is often already fairly low-variance, meaning there's less variance left for bagging to remove — similar to the "helps little" case in Section 7.1. It can still provide *some* benefit (different bootstrap samples can still produce meaningfully different boosted models), but the expected gain is usually smaller than bagging a single deep unpruned tree, and the extra training cost is much higher since each "base learner" is itself a full boosting run.</details>

**Q6 (medium, NEW).** Derive, in one line, why the bias of a bagged ensemble equals the bias of a single tree.
<details><summary>Answer</summary>$\mathbb{E}[\hat f_{\text{bag}}] = \mathbb{E}\left[\frac{1}{M}\sum_m \hat f_m\right] = \frac{1}{M}\sum_m \mathbb{E}[\hat f_m] = \mathbb{E}[\hat f_1]$, using linearity of expectation and the fact that every $\hat f_m$ is identically distributed (same training procedure, same bootstrap distribution). Averaging never appears inside the expectation in a way that could change it — it's a purely linear operation on the fitted values, and bias is defined entirely in terms of that same linear expectation.</details>

**Q7 (hard, NEW).** You bag 500 trees and notice the OOB error and the 5-fold CV error on the same data disagree by more than you'd expect from noise. What are two structural (not just random-noise) reasons this could happen?
<details><summary>Answer</summary>(1) OOB's held-out set for each row is a *variable-size subset of the full $M$-tree ensemble* (only the ~37% of trees that missed that row), which is a weaker sub-ensemble than the full $M$ trees — so OOB error is a mild pessimistic estimate of the full ensemble's true error, while 5-fold CV (if you retrained a fresh $M$-tree ensemble per fold) estimates the full-strength ensemble directly, creating a small systematic gap, not just noise. (2) If `max_samples` or `max_features` differs from what you'd naturally use in a from-scratch CV retrain (e.g., CV code path accidentally uses different bagging hyperparameters), you're not comparing like-for-like models at all — worth checking the two evaluation pipelines are actually configured identically before attributing the gap to OOB vs. CV methodology itself.</details>

---

**One-line summary to remember:** *Bagging = resample with replacement → train many high-variance learners independently → average them to kill variance (never touches bias) → payoff is entirely proportional to how much variance was there to remove and how uncorrelated the trees end up → OOB error gives you validation for free (once $M$ is large enough) → embarrassingly parallel to train, but $M\times$ cost at inference.*
