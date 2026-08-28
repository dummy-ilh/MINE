# Bagging (Bootstrap Aggregating) — Master Notes

---

## 0. Quick Map (read this first)

| Piece | Theoretical role | Practical payoff |
|---|---|---|
| Bootstrap sampling | Manufactures $M$ "different enough" training sets from one dataset | Enables training many trees without collecting more data |
| Averaging | The variance-killing mechanism (Ensemble Foundations, Section 3) | Smooths out any one tree's instability |
| OOB error | A side effect of ~37% of rows being left out of each bootstrap sample | Free validation — no held-out split needed |
| Deep, unpruned trees as base learner | High-variance, low-bias — exactly what averaging is good at fixing | The reason bagging pairs with trees, not linear models |

**One-line theory:** Bagging is the most literal possible application of the variance formula from Ensemble Foundations — it does nothing clever to reduce bias, and nothing clever to decorrelate models beyond plain resampling. It's the "control group" ensemble method: understand it fully, and Random Forest is just "bagging plus one extra decorrelation trick," and boosting is "the opposite strategy, aimed at the other term."

---

## 1. The Big Idea (in one paragraph)

You have one model that's jumpy — small changes in the training data swing its predictions a lot (high variance). Instead of training it once, you train it many times on slightly different "reshuffled" versions of the same data, and average the results. Averaging smooths out the jumpiness. That's it — that's bagging.

The classic base learner for this is a **deep, unpruned decision tree**, because trees are famously unstable (change a few rows, get a very different tree) — which makes them the perfect candidate for variance-smoothing.

### 1.1 A practical analogy (good for interviews)

> Imagine asking one person to estimate a jar of jellybeans by eye — their guess depends heavily on which part of the jar they glanced at, how the light hit it, and pure luck. Ask the *same type* of person to guess 50 times, each time letting them glance at a slightly different, randomly-reshuffled subset of beans, then average all 50 guesses. No single guess got smarter, but the average is far more stable than any one of them — because the guesses' *errors* (not their skill) are what's getting averaged out.

---

## 2. The Algorithm (simplified)

**Step 1 — Make M "reshuffled" datasets.**
For each of $M$ rounds, build a new dataset the same size as the original by sampling **with replacement** (so some rows get picked multiple times, and ~37% of rows get left out entirely, on average).

**Step 2 — Train one tree per dataset.**
Grow each tree deep, don't prune it.

**Step 3 — Combine the M trees.**
- Regression → average the predictions.
- Classification → either majority vote, or average predicted probabilities and pick the top class ("soft voting" — this is sklearn's default and usually works a bit better because it uses more information than a plain vote).

| Term | Plain meaning |
|---|---|
| Bootstrap sample | A resampled dataset of the same size, drawn with replacement |
| $M$ | Number of trees you train |
| Bagging | Bootstrap + Aggregating = resample + combine |

### 2.1 Theory — why not prune the trees?

Pruning already makes a tree stable (low variance) but worse at fitting (higher bias). Bagging's whole job is *removing variance* — it can't fix bias at all. So if you prune first, you've thrown away the exact thing bagging was going to fix, and you're left averaging a bunch of similarly-mediocre, similarly-biased trees for no benefit.

> **Interview soundbite:** *"Pruning before bagging is like hiring a decorrelation specialist and then giving them models that were already stable — you've solved the wrong half of the problem before the tool designed to solve the other half even gets to run."*

---

## 3. Worked Numerical Example #1 (original)

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

---

## 4. Worked Numerical Example #2 (NEW — a full pass with the classification case)

The example above is regression. Here's a matching **classification** worked example, since interviews often flip the question.

Predicting "will this house sell within 30 days" (1 = yes, 0 = no) for a new house $x_0$, using $M=5$ bagged trees. Each tree outputs a probability (soft voting):

| Tree | $D_1$ | $D_2$ | $D_3$ | $D_4$ | $D_5$ |
|---|---|---|---|---|---|
| $P(\text{sells fast})$ | 0.72 | 0.55 | 0.30 | 0.68 | 0.61 |

**Soft voting (sklearn default):** average the probabilities.
$$\bar P = \frac{0.72+0.55+0.30+0.68+0.61}{5} = \frac{2.86}{5} = 0.572$$

At the default 0.5 threshold → predict **class 1 (sells fast)**.

**Hard/majority voting**, for comparison: convert each tree's probability to a class first (threshold 0.5), *then* vote.
- Classes: 1, 1, 0, 1, 1 → 4 votes for class 1, 1 vote for class 0 → majority says **class 1** too, here.

**Why they can disagree in general:** imagine tree $D_3$'s probability had instead been 0.49 instead of 0.30 — hard voting still counts it as a single "0" vote (same weight as a confident "0.02"), while soft voting would barely move the average at all, since 0.49 is so close to the threshold it's nearly a wash. **Soft voting uses the *magnitude* of each tree's confidence; hard voting throws that information away and only keeps the final class label.** This is exactly why soft voting is usually a bit better — it doesn't waste the information contained in "how sure" each tree was.

> **Interview soundbite:** *"Soft voting is almost always the better default because it preserves confidence information hard voting discards — a tree that's 51% sure and a tree that's 99% sure get treated identically by majority vote, but very differently by averaging probabilities."*

---

## 5. Out-of-Bag (OOB) Error — "Free" Validation

### 5.1 The theory

**Idea:** For each row $i$, some trees never saw it during training (it was "out of bag" for them, ~37% of trees on average — this ~37% figure comes directly from the bootstrap-exclusion math in Ensemble Foundations, Section 5). Use *only those* trees to predict row $i$. Since those trees never trained on row $i$, this prediction behaves like a held-out/test prediction — you get validation performance without setting aside a validation set.

$$\hat y_i^{\text{OOB}} = \text{aggregate of } \hat f_m(x_i) \text{ over every tree } m \text{ that didn't see row } i$$

$$\text{OOB Error} = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat y_i^{\text{OOB}})$$

### 5.2 Worked numerical (original — one row)

Using the table above:

Sample 1 is out-of-bag only for $D_2$ (check the table: it's missing from $D_2$'s sampled list, and also missing from $D_3$'s... but $D_3$'s OOB column only lists 3 and 6, meaning $D_3$ *did* include 1). So sample 1's OOB set is just $\{D_2\}$ — one single tree.

Say $D_2$'s tree predicts $\hat f_2(x_1) = 11.2$. Since it's the only OOB voter for row 1:
$$\hat y_1^{\text{OOB}} = 11.2, \qquad \text{true } y_1 = 10, \qquad \text{squared error} = (10-11.2)^2 = 1.44$$

Do this for all 6 rows, average the squared errors → OOB MSE, computed entirely from training data.

### 5.3 Worked numerical (NEW — full 6-row pass, to see the complete OOB pipeline)

Extend the example above across all 6 rows, using the bootstrap table from Section 3:

| Row $i$ | Trees where $i$ is OOB | OOB predictions | True $y_i$ | OOB prediction $\hat y_i^{\text{OOB}}$ | Squared error |
|---|---|---|---|---|---|
| 1 | $D_2$ | 11.2 | 10 | 11.2 | 1.44 |
| 2 | $D_1$ | 12.9 | 12 | 12.9 | 0.81 |
| 3 | $D_2, D_3$ | 21.5, 18.7 | 20 | avg = 20.1 | 0.01 |
| 4 | $D_2$ | 21.0 | 22 | 21.0 | 1.00 |
| 5 | *(none — in-bag for all 3 trees)* | — | 30 | undefined | *excluded* |
| 6 | $D_3$ | 33.1 | 32 | 33.1 | 1.21 |

*(Predictions here are illustrative, consistent with the trees' behavior implied in Section 3.)*

**OOB MSE** (averaging over the 5 rows that have at least one OOB voter): $\frac{1.44+0.81+0.01+1.00+1.21}{5} = \frac{4.47}{5} = 0.894$

**Notice row 5 had to be excluded entirely** — it happened to land in-bag for all 3 trees in this tiny toy example. This is exactly the "small $M$ is noisy" catch flagged below: with only 3 trees, it's plausible for a row to get zero OOB voters at all. At realistic $M$ (hundreds of trees), the chance any row has zero OOB voters is astronomically small ($\approx (1 - 0.368)^{M}$, which is already under 1% by $M \approx 15$), so this edge case essentially disappears in practice.

### 5.4 Why is this useful compared to a normal train/validation split?

| | Train/validation split | OOB |
|---|---|---|
| Data used to train final model | ~80% (rest is held out forever) | 100% |
| Extra models needed | 0 | 0 (comes free from the trees you already trained) |
| "Held-out" portion | One fixed chunk | Different for every row, spread across trees |

**Catch:** with small $M$ (like $M=3$ above), some rows only have 1 OOB voter — noisy. You need $M$ in the hundreds (typical Random Forest default) before OOB error becomes a trustworthy estimate, since each row then accumulates enough OOB votes to average out noise too.

> **Interview soundbite:** *"OOB error at small $M$ is like asking one juror to render a verdict — it's technically 'held-out' but it's a sample of one. You need enough trees that OOB behaves like a jury, not a single opinion."*

---

## 6. When Bagging Helps vs. Doesn't

Comes straight from the variance formula: $\text{Var}(\hat f_{\text{avg}}) \to \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{M}$

**Helps a lot when:**
- Base learner has **high variance** (deep/unpruned trees) — there's a lot of $\sigma^2$ to reduce.
- Trees end up meaningfully different from each other (correlation $\rho$ isn't close to 1) — resampling actually changes tree structure.

**Helps little when:**
- Base learner is already **low-variance** (shallow tree, linear model). Bagging can't touch bias, and there's barely any variance left to remove — you pay $M\times$ the compute for almost nothing. This is *why nobody bags linear regression*.
- Trees end up **highly correlated** ($\rho \approx 1$) — e.g. dataset is tiny so all bootstrap samples look almost the same. The $\rho\sigma^2$ floor term dominates, and adding more trees can't push below it. (This is the exact motivation for Random Forest's extra feature-randomization — pushing $\rho$ down further than resampling alone can.)

### 6.1 Worked numerical (NEW) — quantifying "helps little"

Take a shallow stump (max_depth=1) with $\sigma^2 = 0.3$ (already low variance) and $\rho = 0.6$ (bootstrap samples don't change a shallow stump's single split much, so trees stay fairly correlated). At $M=100$:

$$\text{Var}(\hat f_{\text{bag}}) = 0.6\times0.3 + \frac{(1-0.6)\times0.3}{100} = 0.18 + 0.0012 = 0.1812$$

Compare to a single stump's variance, $0.3$. Bagging bought a drop from 0.3 → 0.1812 — real, but modest, and note **99.3% of that reduction was already achieved by the floor term alone** ($0.18$ out of the total $0.1812$ drop of $0.1188$... more precisely: single-tree variance 0.3, floor 0.18, so max possible reduction is $0.3-0.18=0.12$, and at $M=100$ you've already captured $0.3-0.1812=0.1188$, i.e. **99% of the max possible gain, achieved already by around $M=100$**). Meanwhile bias is completely unchanged — if this stump's Bias² was, say, 9.0 (large, because a stump badly underfits), total error barely moved: from $9.0+0.3=9.3$ to $9.0+0.1812=9.1812$ — a **2% relative improvement**, for 100x the training/serving cost. This is the concrete number behind "why nobody bags shallow, high-bias learners."

### 6.2 Common interview traps

| Claim | True? | Why |
|---|---|---|
| Bagging reduces bias | ❌ No | The bias of the average = bias of a single model, exactly. Biased trees stay biased no matter how many you average. |
| Adding more trees can overfit | ❌ No | Trees are trained independently; more trees only ever pushes variance down toward its floor. `n_estimators` is a compute knob, not an overfitting knob (unlike boosting rounds). |
| OOB error = k-fold CV error | ≈ Close, not identical | OOB uses a *different, variable* subset of trees per row; k-fold uses one *fixed* held-out fold per row. They agree closely in practice for large $M$. |

---

## 7. sklearn Cheat Sheet — `BaggingClassifier` / `BaggingRegressor`

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
`BaggingClassifier` picks a feature subset **once per tree** (same subset used for every split in that tree). Random Forest re-picks a **fresh random subset at every single split**. The latter decorrelates trees much more aggressively — it's the one specific mechanism that makes Random Forest more than "bagging with fewer features."

> **Interview soundbite:** *"If someone asks 'isn't Random Forest just bagging?' — the precise answer is: bagging decorrelates trees only through row resampling; Random Forest adds a second, stronger decorrelation lever by re-randomizing the feature subset at every split, not just once per tree. That's the whole delta between the two algorithms."*

---

## 8. Quick Q&A (general)

**Q: Why is bagging embarrassingly parallel but boosting isn't?**
A: Each tree only depends on its own bootstrap sample — no tree needs to know what any other tree did. Boosting's next model is trained on the *current ensemble's errors*, so model $m+1$ literally can't exist before model $m$ does.

**Q: If more trees never hurts, why not set `n_estimators` huge always?**
A: Pure compute/latency cost for shrinking returns — no accuracy downside, but no point paying for trees past where the curve flattens.

**Q: Give an example where bagging visibly doesn't help.**
A: Bagging a depth-1 stump on a large clean dataset. A stump is already low-variance (few possible splits to disagree on) and high-bias (can only represent one threshold). Bagging leaves bias untouched — the ensemble barely beats a single stump, despite $M\times$ the cost.

---

## 9. Google MLE Interview Q&A

**Q: You're told a Random Forest and a single deep decision tree get nearly the same training accuracy, but very different test accuracy. Explain why, using bagging's mechanics.**
A: Training accuracy mainly reflects bias, and bagging doesn't change bias — so it's expected they'd be similar there. Test accuracy reflects bias *and* variance; the single deep tree has overfit to its specific training rows (high variance, low bias on train), while the forest averaged away most of that variance, so its test performance holds up much better even though its training fit looked "equal."

**Q: How would you use OOB error to pick `n_estimators` without a separate validation set, and what's a failure mode of doing this?**
A: Plot OOB error against $M$ as you grow the forest incrementally — it should decrease and then flatten. Failure mode: at small $M$, OOB error is noisy (few OOB voters per row, as shown in the numerical above), so you can mistake noise for a real plateau or a real improvement. Don't make a stopping decision on a jumpy early-$M$ curve.

**Q: Design question — you have a massive dataset that doesn't fit in memory on one machine. How does bagging's structure help you here?**
A: Since each tree's training is fully independent (no cross-tree communication needed, per the parallel argument above), you can shard the bootstrap sampling and tree training across machines — each worker draws its own bootstrap sample (or a sample from a distributed store) and trains one or more trees, and you only need to gather the final trees for prediction-time aggregation. This is a direct consequence of bagging having zero sequential dependency between rounds — the same property that makes it embarrassingly parallel on one machine makes it embarrassingly *distributable* across machines.

**Q: A colleague says "bagging is basically a crude version of ensembling, we should just always prefer boosting since it usually gets higher accuracy." How do you push back?**
A: Boosting often does win on accuracy, but it comes with different failure modes: boosting is sequential (can't parallelize across rounds), more sensitive to noisy labels/outliers (since it keeps re-weighting toward hard/misclassified points, which can include mislabeled data), and can overfit if you add too many rounds — none of which is true for bagging. So the right framing isn't "bagging is a lesser boosting," it's a trade-off: bagging for parallelism, robustness to noisy labels, and a training loop that's structurally overfitting-proof in $M$; boosting when you can afford sequential training and want to squeeze out more accuracy from a clean dataset.

---

## 10. Apple MLE Interview Q&A (on-device / practical flavor)

**Q: You want an on-device ensemble model (e.g., inside Core ML) for a latency-sensitive feature. Would you reach for bagging, and what's the practical trade-off?**
A: Bagging's inference cost scales linearly with $M$ — you run all $M$ trees and aggregate, every single prediction, on every request. On a phone/watch, that's $M\times$ the compute and memory footprint compared to a single tree, which matters a lot more on-device than in a data center. In practice this pushes you toward either a small $M$ (accepting less variance reduction), a much shallower/lighter base learner, or switching to a single well-regularized model or a distilled model rather than shipping the full ensemble — bagging's parallelism benefit (useful for training) doesn't help you at inference time on a single constrained device.

**Q: How does bagging's training-time parallelism map onto a Private Cloud Compute–style setup, where training might happen off-device before a model is distributed to devices?**
A: Since each tree trains independently on its own bootstrap sample, training can be fully parallelized across the compute cluster used before distribution — this is a training-time-only benefit and doesn't change anything about the on-device footprint discussed above; you still ship (and pay the inference cost for) all $M$ trained trees to the device afterward, unless you prune the ensemble down or distill it into something smaller first.

**Q: If you were using bagging as part of a privacy-sensitive pipeline (e.g., data that can't leave a device, differential-privacy constraints), what does bootstrap sampling interact with?**
A: Bootstrap sampling means the same row can be selected multiple times in one bootstrap draw and each tree sees a different resampled subset — if you're layering differential privacy on top, that resampling changes how many times any individual record influences a given tree's output, which affects your privacy-budget accounting per tree. It's not something bagging handles automatically; whatever DP mechanism you use has to be aware that "one bootstrap sample" isn't the same as "one pass over each record exactly once."

**Q: OOB error gave you a validation-free way to estimate error — is that still meaningful in a federated-learning setting where trees might be trained across many separate on-device datasets?**
A: The core requirement for OOB — that some trees genuinely never saw a given row — still holds as long as each device's local bootstrap sample leaves out some of that device's own local rows. But OOB in a federated setup only estimates per-device local error unless predictions and OOB bookkeeping are aggregated back centrally, which itself is more coordination than plain on-device bagging assumes; it's a case where the "free validation" framing needs to be revisited rather than assumed to transfer directly.

---

## 11. Interview-Ready Soundbites (collected in one place)

1. *"Bagging is the control-group ensemble method — no cleverness beyond resample-and-average. Understand it fully and Random Forest is just 'bagging plus one extra decorrelation trick,' boosting is 'the opposite strategy, aimed at the other term.'"*
2. *"Pruning before bagging solves the wrong half of the problem — you've stabilized a model right before handing it to a tool whose entire purpose was stabilizing it."*
3. *"Soft voting keeps the magnitude of each tree's confidence; hard voting throws that information away and keeps only the label. That's why soft voting is the better default almost everywhere."*
4. *"OOB error at small $M$ is a jury of one — you need enough trees for it to behave like a jury instead of an opinion."*
5. *"`n_estimators` is a compute knob, not an overfitting knob — trees train independently, so more of them can only push variance further down toward its floor, never up."*
6. *"Bagging's parallelism is a training-time property. At inference, on a single device, you still pay for every one of the M trees — the free lunch is on the training side only."*

---

## 12. Practice Q&A (NEW — to mirror the format used elsewhere in the curriculum)

**Q1 (easy).** Why does `BaggingRegressor`'s default `n_estimators=10` usually get bumped up in practice?
<details><summary>Answer</summary>Per the variance formula, $M=10$ still leaves a meaningful chunk of the shrinking term $\frac{(1-\rho)\sigma^2}{M}$ un-shrunk — going from 10 to, say, 100+ trees (Random Forest's typical default) captures most of the remaining achievable variance reduction, per the diminishing-returns pattern from Ensemble Foundations Section 3.5. 10 trees is a reasonable quick baseline but rarely the endpoint.</details>

**Q2 (easy).** A single decision tree gets 85% train accuracy and 60% test accuracy. After bagging with 200 trees, train accuracy is 84% and test accuracy is 79%. What happened?
<details><summary>Answer</summary>Classic variance reduction at work: bias stayed roughly the same (train accuracy 85%→84%, essentially unchanged, consistent with bagging not touching bias), while the large train/test gap — the fingerprint of high variance — shrank substantially (25 points → 5 points) because averaging 200 trees smoothed out the instability that was hurting generalization.</details>

**Q3 (medium).** Using Section 5.3's OOB table, why did row 5 end up excluded, and what does that imply about choosing $M$ in real settings?
<details><summary>Answer</summary>Row 5 happened to be sampled into every one of the 3 bootstrap training sets, leaving it with zero OOB voters — possible at small $M$ purely by chance ($\approx (1-0.368)^M$ probability of a row landing in-bag everywhere, which is non-negligible at $M=3$). In real settings with $M$ in the hundreds, this probability becomes vanishingly small, so essentially every row ends up with a usable OOB estimate — another reason OOB reliability specifically depends on $M$ being reasonably large.</details>

**Q4 (medium).** Why can't you compute OOB error for a `BaggingRegressor` where `max_samples` is set so high that no rows are ever left out?
<details><summary>Answer</summary>OOB error fundamentally depends on some rows being excluded from some trees' training sets — if `max_samples` samples the entire dataset without leaving room for exclusions (or if sampling is done without replacement, so all $n$ rows always appear), there's no "unseen" subset per tree to compute a held-out prediction from. OOB is a direct byproduct of the with-replacement exclusion mechanic; remove that mechanic and OOB has nothing to work with.</details>

**Q5 (hard).** A teammate wants to bag an XGBoost model (i.e., train several independent XGBoost models on different bootstrap samples and average them). Is this a reasonable idea?
<details><summary>Answer</summary>It's unusual but not nonsensical — it depends on what XGBoost's own variance looks like after its typical regularization (shallow trees, learning rate, etc.). A well-tuned, regularized boosted model is often already fairly low-variance (that's partly the point of boosting's incremental, regularized corrections), meaning there's less variance left for bagging to remove — similar to the "helps little" case in Section 6. It can still provide *some* benefit (different bootstrap samples can still produce meaningfully different boosted models, especially with different random feature/row subsampling inside each XGBoost run), but the expected gain is usually smaller than bagging a single deep unpruned tree, and the extra training cost is much higher since each "base learner" is itself a full boosting run.</details>

---

**One-line summary to remember:** *Bagging = resample with replacement → train many high-variance learners independently → average them to kill variance (never touches bias) → OOB error gives you validation for free → embarrassingly parallel to train, but $M\times$ cost at inference.*
