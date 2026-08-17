# Random Forests — Deep Dive Notes

Self-contained reference. Builds on: Ch.1 (single trees, split search, impurity $\Delta$), Ch.2.2 (bagging variance formula $\rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}$), Ch.3 (bootstrap bagging, OOB).

---

## 1. The One Idea Random Forest Adds

Bagging (Ch.3) only randomizes **rows** (bootstrap). Every tree still does a full $p$-feature sweep at every split, so dominant features win at (or near) the root **almost every time**, across almost every bootstrap sample → trees stay structurally similar → correlation $\rho$ stays high → the $\rho\sigma^2$ floor in the variance formula stays high, and more trees ($M\uparrow$) can never push below that floor.

**Random Forest's fix:** randomize **columns too — freshly, at every node, in every tree.**

> At each node: draw a random subset of $k<p$ features → search for the best split *only* within those $k$ → discard the subset → draw a brand-new independent subset for the next node (even a sibling node gets its own fresh draw).

This is the entire algorithm. Everything else (tree growing, splitting criterion, leaf prediction, aggregation by majority-vote/average) is identical to bagged trees (Ch.1 + Ch.3).

**Mental model:** bagging decorrelates by showing each tree *different data*. Random Forest additionally decorrelates by *blindfolding* each split to most of the feature space, forcing the tree to occasionally use a feature it would never have chosen if it could see everything.

---

## 2. Why Column-Subsampling Actually Lowers $\rho$ — Worked Numerical

Setup: $p=100$ features, 5 "strong" (truly predictive), 95 pure noise. `max_features` $k=\sqrt{100}=10$.

**Question:** what's the probability a given split's candidate pool contains *zero* strong features (forcing a split on noise/weak signal, i.e., forcing structural difference)?

$$
P(\text{0 strong features drawn}) = \prod_{i=0}^{9}\frac{95-i}{100-i} = \frac{95}{100}\cdot\frac{94}{99}\cdots\frac{86}{91} \approx 0.95^{10} \approx 0.60
$$

**Reading this:** ~60% of splits are "starved" of the dominant signal and forced to diverge structurally from tree to tree; ~40% of splits still get access to a strong feature so the ensemble isn't crippled. That 60/40 split *is* the decorrelation mechanism, made concrete.

**Sanity-check the boundaries (this is the interview-favorite follow-up):**

| $k$ | Effect |
|---|---|
| $k=100$ (all features) | $P(\text{starved})=0$ → identical to plain bagging → $\rho$ stays high |
| $k=10=\sqrt{p}$ | $P(\text{starved})\approx0.60$ → sweet spot: frequent diversity, occasional access to signal |
| $k=1$ | Trees almost never see a strong feature together with the right context → individual trees become **weak/biased**, not just diverse |

$\sqrt{p}$ has **no clean closed-form derivation** — it's empirically validated, not provably optimal. Say this explicitly in interviews; don't imply it's derived from first principles.

---

## 3. The Trade-off, Formalized

Recall Ch.2.2: $\text{Var}(\text{ensemble}) = \rho\sigma^2 + \dfrac{(1-\rho)\sigma^2}{M}$.

Decreasing `max_features` ($k\downarrow$):
- $\rho \downarrow$ (good — lowers the floor)
- $\sigma^2 \uparrow$ (bad — each individual tree is weaker/noisier since it's often blocked from the best available split)

This is a genuine two-sided trade-off — **not** "always shrink $k$." The whole value of Random Forest is that empirically, for many datasets, the $\rho$-reduction dominates the $\sigma^2$-increase.

**Worked numerical (extending Ch.2.2, $M=200$):**

| Setting | $\rho$ | $\sigma^2$ | $\rho\sigma^2$ (floor) | $\frac{(1-\rho)\sigma^2}{M}$ | Total Var |
|---|---|---|---|---|---|
| Bagging-like ($k=p$) | 0.50 | 4.0 | 2.000 | 0.010 | **2.010** |
| RF-like ($k=\sqrt p$) | 0.25 | 4.5 | 1.125 | 0.017 | **1.142** |

Even though each RF tree is individually *worse* ($\sigma^2$: 4.0→4.5), the ensemble variance nearly halves, because the floor term ($\rho\sigma^2$) dominates the total and $\rho$ dropped by half. **This is the single most important numeric intuition to have memorized for an RF interview question.**

---

## 4. Hyperparameters — What Each One Actually Trades Off

| Hyperparameter | ↑ effect | Mechanism |
|---|---|---|
| `n_estimators` ($M$) | Variance ↓ (diminishing returns), bias unaffected, **never overfits** | Shrinks $\frac{(1-\rho)\sigma^2}{M}$ toward 0; floor $\rho\sigma^2$ is untouched by $M$ |
| `max_features` ($k$) | The real trade-off knob: $k\uparrow$ → $\rho\uparrow$ (worse) but per-tree strength ↑ (better) | Section 3 above |
| `max_depth` / `min_samples_leaf` | Shallower → bias ↑, per-tree variance ↓ | Same as a standalone tree (Ch.1), but RF conventionally leaves trees **deep/unpruned** because averaging already handles variance — pruning here mostly just adds bias for little benefit |

**Tuning order that interviewers want to hear:** tune `max_features` (and depth/leaf constraints) via CV first — it changes the *floor*. Then set `n_estimators` as high as compute allows — it can only help (or plateau), never hurt, since it just closes the gap to a floor that's already fixed by the other knobs.

**Why RF never overfits by adding trees, but boosting (Ch.5 preview) needs a learning rate:** RF trees are trained *independently* and averaged — no sequential mechanism exists for later trees to chase residual noise. Boosting trees are sequential corrections to error; without a learning rate to shrink each round's contribution, later rounds can increasingly fit pure noise. This asymmetry (bagging-family vs boosting-family) is a very common interview probe.

---

## 5. Feature Importance — Two Methods, and Why They Disagree

### 5.1 Mean Decrease in Impurity (MDI) — sklearn's default `.feature_importances_`

For one tree:
$$
\text{Imp}_j^{\text{tree}} = \sum_{\text{nodes }t\text{ splitting on }j} \frac{n_t}{n}\Delta(t)
$$
Average across all $M$ trees, normalize to sum to 1.

**The bias, precisely:** a feature with many candidate split thresholds (continuous, or high-cardinality categorical) gives the greedy search *more chances* to find a threshold that happens to produce a large impurity drop **by chance on the training set alone**, even with zero true relationship to the target. This is a multiple-comparisons problem — structurally identical to the Gain-vs-Gain-Ratio bias in ID3/C4.5 (Ch.1.3.1), just surfacing in importance scores instead of split selection. A binary feature gets exactly one threshold to try, so it can't get "lucky" the same way.

**Cost of MDI:** effectively free — it's bookkeeping already computed during training.

### 5.2 Permutation Importance — the correction

1. Baseline score (accuracy / $R^2$) on held-out or OOB data.
2. Shuffle feature $j$'s values only (breaks its target relationship, preserves its marginal distribution and every other feature).
3. Rescore. $\text{Imp}_j = \text{Score}_\text{baseline} - \text{Score}_\text{permuted}$.
4. Repeat several shuffles, average (stability).

**Worked numerical:** baseline OOB accuracy 0.850.
- Permute "income" → 0.790 → importance $=0.060$ (genuinely predictive)
- Permute "favorite_color" → 0.848 → importance $=0.002$ (correctly near-zero, *regardless* of how many thresholds that feature had at training time)

**Cost of permutation importance:** $p\times$(repeats) full re-scoring passes over held-out data — meaningfully more expensive than MDI, especially for large $p$ or large ensembles.

**When to use which:** MDI for a quick free look during training; permutation importance when the *ranking itself* drives a decision (e.g., feature selection, explaining a model to stakeholders) — because MDI's high-cardinality bias can actively mislead that decision.

---

## 6. Random Forest vs. Plain Bagging — the Precise Delta

| | Bagging | Random Forest |
|---|---|---|
| Row randomization | Bootstrap | Same |
| Feature randomization | Off by default; if enabled, sampled **once per tree** | **Always on**, sampled **fresh per split** |
| $\rho$ | Higher | Lower |
| Base tree depth | Full depth, unpruned | Same (variance handled by ensembling) |
| Built-in importance | Not standard | `.feature_importances_` (MDI) native |
| Converge to same algorithm when... | `max_features` set to consider *all* features at every split | — |

**One-liner to have memorized verbatim:** *"Random Forest is bagging plus one additional randomization mechanism — per-split random feature subsampling — that decorrelates trees further than bootstrap resampling alone can, directly lowering the $\rho$ term that caps how much variance any bagging-style ensemble can remove."*

---

## 7. Extremely Randomized Trees (Extra Trees) — One Step Further

Two changes relative to Random Forest:

1. **No bootstrap by default** (`bootstrap=False`) — every tree trains on the full dataset; row-level diversity is dropped entirely.
2. **Split *thresholds* are randomized too.** For each candidate feature (drawn the same random-subset way as RF), instead of an exhaustive optimal-threshold search (Ch.1.3), draw **one random threshold** within that feature's observed range, and pick the best among those random draws (not the best among all possible thresholds).

**Why this can help:** same $\rho$-vs-$\sigma^2$ trade-off, pushed further — two trees now rarely agree on a split even when they agree on the feature, so $\rho$ drops even more. Cost: each tree is a noisier/weaker fit since it gives up the locally-optimal threshold guarantee. Also **faster to train** — skips the $O(n\log n)$ sweep per feature for a single random draw.

**Not a strict improvement:** on noisy data, the extra randomization prevents overfitting to spurious "optimal-looking" thresholds (Extra Trees wins). On clean, larger datasets where exhaustive search reliably finds real signal, giving that up can lose more accuracy than the added decorrelation buys back (RF wins). This is dataset-dependent — no universal winner.

---

## 8. sklearn Parameter Reference

**`RandomForestClassifier` / `RandomForestRegressor`**

| Parameter | Controls | Notes |
|---|---|---|
| `n_estimators` | $M$ | Default **100** (vs `BaggingClassifier`'s default 10 — RF trees are cheap to productively decorrelate-and-average) |
| `max_features` | $k$, **per split** | Classifier default `'sqrt'`; Regressor default `1.0` (all features — sklearn does **not** auto-apply the classic $p/3$ rule for regression; worth tuning down manually) |
| `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, `min_impurity_decrease`, `ccp_alpha` | Per-tree controls | Same as Ch.1, applied identically to every tree |
| `bootstrap` | Row sampling with replacement | Default `True`. `False` → every tree sees the full dataset, diversity comes only from feature subsampling |
| `oob_score` | Compute OOB error (Ch.3.3) | Default `False`; only meaningful with `bootstrap=True` |
| `max_samples` | Fraction of rows per bootstrap draw | Default `None` (full $n$-size bootstrap); lets you subsample rows below 100% even with bootstrap on |
| `n_jobs` | Parallelism across trees | Embarrassingly parallel, same as Bagging |

**`ExtraTreesClassifier`/`Regressor`:** same table, with `bootstrap=False` by default, and threshold search replaced by the random-draw mechanism (Section 7) — not a separate tunable, it's the defining behavior of the class.

**Key gotcha (asked often):** `BaggingClassifier`'s `max_features` samples **once per tree**; `RandomForestClassifier`'s `max_features` samples **fresh per split**. Same parameter name, structurally different mechanism — don't conflate them.

---

## 9. Interview Q&A Bank

**Q: Does more trees ever cause Random Forest to overfit?**
A: No. Trees are independent and averaged; more trees only push variance toward the $\rho\sigma^2$ floor (Ch.2.2), never reintroduce overfitting. `n_estimators` is a compute/diminishing-returns knob, not something to regularize against.

**Q: Why does boosting need a learning rate but RF doesn't?**
A: Boosting is sequential residual-correction — without shrinkage, later rounds can increasingly chase training noise. RF trees are independent; there's no sequential mechanism for a learning rate to dampen.

**Q: 500 features, 5 truly predictive. RF with `max_features='sqrt'` or Bagging with `max_features=1.0`?**
A: RF. With 500 features and only 5 useful, bagging's every-split-sees-everything design means the same handful of strong features dominate nearly every tree (very high $\rho$). RF's $\sqrt{500}\approx22$-feature subsampling forces many splits to proceed without them, producing more structurally diverse trees and a lower correlation floor — exactly where the extra decorrelation mechanism earns its keep most.

**Q: Why is MDI importance biased, and what fixes it?**
A: Continuous/high-cardinality features get more candidate thresholds → more chances to find a training-set-only "lucky" impurity drop (multiple-comparisons effect) → inflated MDI score even with zero true signal. Permutation importance fixes this by measuring the actual *predictive contribution on held-out data* — immune to threshold-count bias, at the cost of $p\times$repeats extra scoring passes.

**Q: `max_features` for Bagging vs Random Forest — same knob?**
A: Same name, different mechanism. Bagging: sampled once per tree (if enabled at all, off by default). Random Forest: always on, sampled fresh at every node. This *is* the structural difference that defines RF vs Bagging (Section 6).

**Q: When would you prefer Extra Trees over Random Forest?**
A: Noisy datasets where exhaustive threshold search tends to overfit to spurious "optimal" splits, and/or when training speed matters (Extra Trees skips the $O(n\log n)$ per-feature sweep). On cleaner/larger data where the optimal threshold reliably reflects real signal, RF usually wins — Extra Trees isn't a strict upgrade.

---

## 10. Compressed Summary (for last-minute review)

- **Core mechanism:** bootstrap rows (from bagging) + fresh random $k$-feature subset **per split** (RF's addition).
- **Why it works:** lowers pairwise tree correlation $\rho$, which is the hard floor ($\rho\sigma^2$) on ensemble variance that more trees alone cannot beat.
- **Trade-off:** $k\downarrow$ → $\rho\downarrow$ (good) but per-tree $\sigma^2\uparrow$ (bad); $\sqrt p$ (classification) / all-or-$p/3$ (regression) are empirical sweet spots, not derived optima.
- **`n_estimators`:** pure variance-reduction, no overfitting risk, tune last/generously.
- **Importance:** MDI = free but biased toward high-cardinality features; permutation = costly but trustworthy.
- **Extra Trees:** RF + randomized thresholds (+ usually no bootstrap) → more decorrelation, weaker individual trees, faster training; wins on noisy data, not universally.
