# Feature Importance — MDI vs. Permutation vs. SHAP — Master Notes

## 1. The Big Idea (in one paragraph)

"Feature importance" always answers some version of the question **"how much does this feature matter?"** — but the three common methods answer it in genuinely different ways, using different information, at different costs. MDI asks it *during training*, using only the tree's own bookkeeping. Permutation importance asks it *after training*, by breaking each feature one at a time and seeing how much worse predictions get. SHAP asks a more specific question entirely — not "how important is this feature overall," but "for this one specific prediction, how much did each feature push it up or down." None of the three is strictly "best" — they trade off cost, trustworthiness, and what exactly they tell you.

---

## 2. Method 1 — MDI (Mean Decrease in Impurity)

### 2.1 What it is

**Plain idea:** every time a feature is used for a split anywhere in the tree, that split reduces impurity by some amount (Gini/entropy/MSE — see the Decision Tree Fundamentals notes). MDI just adds up all the impurity reduction a feature is responsible for, across every split, across every tree in the forest.

### 2.2 The Formula

For a single tree:

$$\text{MDI}(f) = \sum_{t \,:\, \text{split on } f} p(t) \cdot \Delta i(t)$$

| Symbol | Meaning |
|---|---|
| $t$ | A node that splits on feature $f$ |
| $p(t)$ | Fraction of all training samples that reach node $t$ (weights splits near the root more heavily than deep, rarely-visited splits) |
| $\Delta i(t)$ | Impurity decrease at that split (parent impurity minus weighted child impurity — same $\Delta$ from the splitting-criteria chapter) |

For a forest, average this across all $M$ trees:

$$\text{MDI}_{\text{forest}}(f) = \frac{1}{M}\sum_{m=1}^{M} \text{MDI}_m(f)$$

Then usually normalized so all features' importances sum to 1.

### 2.3 Worked Numerical

Suppose feature "size" is used at 2 splits in a tree:

| Split | $p(t)$ (fraction of samples reaching it) | $\Delta i(t)$ |
|---|---|---|
| Root split | 1.00 (100% of samples) | 0.30 |
| A deeper split | 0.25 (25% of samples) | 0.10 |

$$\text{MDI(size)} = (1.00 \times 0.30) + (0.25 \times 0.10) = 0.30 + 0.025 = 0.325$$

Suppose feature "age" only appears once, deep in the tree, at a node reached by 10% of samples, with $\Delta i = 0.05$:

$$\text{MDI(age)} = 0.10 \times 0.05 = 0.005$$

**Size (0.325) looks far more important than age (0.005)** by this measure — mostly because size was used at the root, where it affects 100% of samples, while age's one split only ever touches 10% of them.

### 2.4 The Bias Problem

**MDI is systematically biased toward features with more possible split points** — continuous features, and high-cardinality categorical features. Why: at every node, the split search tries every threshold of every feature and keeps whichever gives the best impurity reduction. A feature with 1,000 distinct values gets 999 chances to get lucky and find a threshold that reduces impurity somewhat by pure chance, purely from having more options to try — even if it has zero real relationship to the target. A binary feature only ever gets 1 chance.

**Extreme illustration:** add a column of pure random noise, but make it continuous with many unique values (e.g., `np.random.rand()` per row). MDI will often rank it as moderately important — not because it's predictive, but because with enough distinct threshold options, *some* split on it will reduce training impurity a little, purely by chance. A near-unique ID column is the worst-case version of this trap (see the Decision Tree Fundamentals and Google Interview Q&A notes for the full ID-column discussion).

### 2.5 Cost

**Free.** MDI comes directly out of information the tree already computed while training — no extra passes over the data, no retraining, nothing extra to run. This is its single biggest practical advantage.

---

## 3. Method 2 — Permutation Importance

### 3.1 What it is

**Plain idea:** if a feature actually matters, scrambling (randomly shuffling) its values across all rows should make predictions noticeably worse — because the model is now getting garbage information for that feature while everything else stays correct. If a feature doesn't matter, scrambling it barely changes anything.

### 3.2 The Formula

$$\text{PermImp}(f) = \text{Score}(\text{model}, X, y) - \text{Score}(\text{model}, X_{\text{shuffled}(f)}, y)$$

Where "Score" is whatever evaluation metric you care about (accuracy, R², negative MSE, etc.), and $X_{\text{shuffled}(f)}$ is the dataset with feature $f$'s column randomly permuted across rows (every other column left untouched).

**The full procedure:**
1. Compute the baseline score on the (held-out, ideally) dataset with everything intact.
2. For each feature $f$: shuffle just that one column, re-score the model (no retraining — same trained model, just different input), record how much the score dropped.
3. Repeat step 2 multiple times per feature (different random shuffles) and average, to reduce noise from any one unlucky/lucky shuffle.
4. Bigger score drop = more important feature.

### 3.3 Worked Numerical

Baseline model accuracy on held-out data: **0.90**

| Feature shuffled | Accuracy after shuffling | Importance (drop) |
|---|---|---|
| size | 0.72 | $0.90 - 0.72 = 0.18$ |
| location | 0.81 | $0.90 - 0.81 = 0.09$ |
| age | 0.895 | $0.90 - 0.895 = 0.005$ |
| random noise column | 0.899 | $0.90 - 0.899 = 0.001$ |

**Reading this:** size matters a lot (scrambling it costs 18 points of accuracy). Age barely matters (0.5 points). The random noise column, correctly, shows almost zero importance — this is the key property MDI lacks (Section 2.4).

### 3.4 Why This Fixes MDI's Bias

Permutation importance measures **actual predictive contribution on held-out data**, not "how many chances did this feature get to look good during training." A random noise column, however many unique values it has, genuinely won't help predict the target — so shuffling it changes nothing, correctly showing ~0 importance regardless of its cardinality.

### 3.5 Cost

**Expensive relative to MDI** — you need one full re-scoring pass per feature (times however many repeats you do for stability), on top of already having a trained model. For $p$ features and $R$ repeats, that's $p \times R$ extra scoring passes over the dataset. Still far cheaper than retraining the model $p$ times, but not free the way MDI is.

### 3.6 A Caveat Worth Knowing

Permutation importance can understate the importance of **correlated features**. If "size" and "num_rooms" are highly correlated, shuffling just "size" alone doesn't hurt predictions much — because "num_rooms" still carries almost the same information, and the model can lean on it instead. Both features might show artificially low individual importance even though the *pair* is jointly very important.

---

## 4. Method 3 — SHAP (SHapley Additive exPlanations)

### 4.1 What it is — and how it's a different question entirely

MDI and permutation importance both answer **"overall, across the whole dataset, how much does this feature matter?"** — a single ranking. SHAP answers a different question: **"for this one specific prediction, how much did each feature push it up or down from the average?"**

**Plain idea:** start from the average prediction across the whole dataset (say, $300,000). For one particular house predicted at $420,000 (a $120,000 gap from average), SHAP splits that $120,000 gap into a contribution from each feature: "+$80,000 because it's unusually large," "+$50,000 for location," "−$10,000 for age" — and these add up exactly to the total gap.

### 4.2 The Formula (conceptual)

SHAP borrows **Shapley values** from cooperative game theory — originally designed to fairly split a group payout among players who might contribute differently in different combinations. Treat each *feature* as a "player," and the *prediction* as the "payout":

$$\phi_f = \sum_{S \subseteq F \setminus \{f\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{f\}) - v(S)\Big]$$

| Symbol | Meaning |
|---|---|
| $F$ | The full set of features |
| $S$ | Some subset of features *not including* $f$ |
| $v(S)$ | The model's expected prediction using only the features in $S$ (others "unknown"/averaged out) |
| $\phi_f$ | Feature $f$'s fair share of credit for this prediction |

**Plain reading, no notation:** consider every possible order in which features could be "revealed" one at a time. For each order, measure how much the prediction changes the moment $f$ gets revealed. Average that change across every possible order. That average is $f$'s fair contribution — "fair" specifically because it accounts for features mattering differently depending on which other features are already known, rather than just looking at one fixed order.

**The property that makes this "additive":**

$$\hat f(x) = \phi_0 + \sum_{f} \phi_f$$

where $\phi_0$ is the average prediction across the dataset — every feature's contribution plus the baseline adds up exactly to this one prediction, no residual left over.

### 4.3 Worked Numerical (simplified, 2-feature toy example)

Average house price across the dataset: $\phi_0 = \$300{,}000$. One specific house: size = large, location = downtown, predicted price = $420,000.

Suppose we can compute (via the averaging-over-orderings procedure):
- Revealing "size = large" first (before location is known): pushes prediction from $300K to $370K → contributes **+$70K**
- Then revealing "location = downtown" (with size already known): pushes prediction from $370K to $420K → contributes **+$50K**

But orderings matter — reveal location first instead:
- Revealing "location = downtown" first: pushes prediction from $300K to $335K → contributes **+$35K**
- Then revealing "size = large" (with location already known): pushes prediction from $335K to $420K → contributes **+$85K**

**Averaging both orderings** (this is the actual Shapley procedure — average across all possible reveal-orders):
$$\phi_{\text{size}} = \frac{70+85}{2} = 77.5K \qquad \phi_{\text{location}} = \frac{50+35}{2} = 42.5K$$

**Check the additive property:** $300K + 77.5K + 42.5K = 420K$ ✓ — matches the actual prediction exactly.

### 4.4 TreeSHAP — Making This Actually Computable

Computing the formula above literally (checking every possible subset ordering) is exponential in the number of features — infeasible beyond a handful of features. **TreeSHAP**, built specifically for tree-based models, computes *exact* SHAP values efficiently by exploiting the tree's structure directly (tracking, for each node, how training data would flow under every feature subset simultaneously, rather than evaluating each subset independently). Roughly $O(T L D^2)$ across $T$ trees, $L$ leaves, $D$ depth — polynomial, not exponential. This efficiency is specifically why SHAP is used so heavily with tree models, more than with model types where only slower approximations (like KernelSHAP) exist.

### 4.5 Cost

**Most expensive of the three**, even with TreeSHAP's efficiency gains — it computes a full per-feature, per-prediction breakdown, not just one number per feature. Cheap enough to be practical for tree ensembles thanks to TreeSHAP, but noticeably more expensive than either MDI (free) or permutation importance (moderate).

---

## 5. Head-to-Head: Which Is Better?

**Short answer: there isn't a single "best" — they answer different questions at different costs, and the honest interview answer is knowing which one fits which situation.**

| | MDI | Permutation Importance | SHAP |
|---|---|---|---|
| Question answered | "How much impurity reduction is this feature responsible for, overall?" | "How much worse do predictions get if I scramble this feature, overall?" | "For this one specific prediction, how much did each feature contribute?" |
| Scope | Whole-model ranking | Whole-model ranking | Per-prediction breakdown (can be averaged into a whole-model ranking too) |
| Cost | Free (byproduct of training) | Moderate (one re-score pass per feature × repeats) | High, even with TreeSHAP (per-prediction computation) |
| Biased toward high-cardinality features? | **Yes — significant bias** | No | No |
| Handles correlated features well? | Somewhat distorted (can over- or under-credit correlated features depending on split order) | Can understate importance of correlated features (Section 3.6) | Handles better in principle (averages over orderings), though correlated features are a known hard case for all three methods |
| Needs held-out data to be trustworthy? | No (computed purely from training) | Yes, ideally (measures real predictive contribution) | Not strictly required, but more meaningful evaluated against real/held-out data |
| Gives per-instance explanations? | No | No | **Yes — its whole purpose** |
| Best use case | A free first look, early in exploration | Before making a real decision off the ranking (e.g., dropping features) | Explaining one specific prediction to a stakeholder/customer/regulator |

### 5.1 The Practical Decision Rule

1. **Use MDI first, for a quick free look.** It costs nothing — you already have it the moment training finishes. Treat it as a rough first pass, not a final answer.
2. **Switch to permutation importance before acting on the ranking.** If you're about to drop features, justify a modeling decision, or report "what matters" to someone who'll act on it, permutation importance is the more trustworthy number, since it measures real predictive contribution rather than training-time impurity bookkeeping.
3. **Reach for SHAP when you need to explain one specific prediction.** Neither MDI nor permutation importance can tell you why *this one* house, loan applicant, or transaction got the specific output it did — only SHAP answers that question, which matters a lot in regulated or customer-facing settings (lending, insurance, fraud, medical risk).

### 5.2 Why "Better" Is the Wrong Frame

Asking "which is better" implicitly assumes they're competing to answer the same question — they aren't. MDI and permutation importance both give overall rankings but disagree because they measure different things (training-time impurity bookkeeping vs. actual held-out predictive contribution). SHAP isn't even in the same category — it's a per-prediction explainer that happens to be *averageable* into something ranking-like, not a competitor to the other two on their own terms. In an interview, the strongest answer names all three trade-offs rather than picking one as universally superior.

---

## 6. Quick Q&A (general)

**Q: You see MDI rank a near-random ID-like column as moderately important. What's happening, and what would you check next?**
A: Classic MDI bias toward high-cardinality/continuous features — more possible split thresholds means more chances to find a split that reduces training impurity by pure chance, even with zero real signal. Recompute with permutation importance; if the ID column's importance collapses to ~0, that confirms it was an MDI artifact, and the feature should likely be dropped or investigated for a possible data leak.

**Q: Why might permutation importance and MDI disagree on which feature is #1?**
A: They measure genuinely different things. MDI accumulates impurity reduction *during training*, which is biased toward features with many split options and can be inflated even for features with weak real predictive value. Permutation importance measures the *actual drop in held-out predictive performance* when a feature is destroyed — a feature can score high on MDI (lots of training-time splits) while scoring low on permutation importance (little real held-out predictive contribution), especially for high-cardinality features.

**Q: Why can't you just use SHAP's per-prediction values as your only feature-importance tool and skip MDI/permutation entirely?**
A: You can average |SHAP value| across all predictions to get an overall ranking, and many practitioners do exactly that — but it costs meaningfully more compute (even with TreeSHAP) than either MDI (free) or permutation importance (moderate), so it's often not worth the cost for a quick exploratory pass. SHAP's real advantage is the per-prediction breakdown, which MDI and permutation importance simply can't provide — using SHAP only for its aggregate ranking throws away the thing it's uniquely good at while paying its full cost.

---

## 7. Google MLE Interview Q&A

**Q: Design question — you need to build an automated feature-selection step into a training pipeline that runs nightly on fresh data. Which of the three would you use, and why?**
A: MDI, as the first-pass filter — it's free (already computed during training, no extra pass over data) and running nightly means the cost of a repeated expensive step compounds fast. Use it to flag *candidates* for removal (very low MDI, consistently, across several nightly runs) rather than final decisions, and route only borderline/high-stakes removals through a periodic (not nightly) permutation-importance check to correct for MDI's high-cardinality bias before actually dropping anything from the pipeline. Pure MDI-only automation risks silently dropping a genuinely useful high-cardinality feature that MDI over- *or* under-rates — worth a human-reviewed permutation check before anything is permanently removed.

**Q: A colleague claims "permutation importance is strictly more correct than MDI, so we should just always use it and stop computing MDI at all." How do you push back?**
A: Permutation importance is more *trustworthy* for making decisions, but "strictly more correct" overstates it — it has its own known failure mode (Section 3.6): correlated features can each show artificially low individual importance because shuffling one alone doesn't hurt much when a correlated partner still carries the signal. MDI, despite its high-cardinality bias, doesn't have exactly this failure mode in the same way, since it credits whichever feature actually won the split search at each node. The stronger position isn't "replace MDI with permutation importance," it's "use MDI as a free first pass, permutation importance before real decisions, and be aware both can mislead on correlated feature groups — check correlation structure directly if that's a concern."

**Q: You're asked why TreeSHAP is specifically efficient for trees but not generally available for, say, a neural network in the same polynomial time. What's the actual mechanism that breaks?**
A: TreeSHAP's efficiency comes from being able to track, at each tree node, how training data flows under every feature subset simultaneously — exploiting the fact that a tree's prediction only depends on which path a sample takes through a *fixed, discrete* structure of splits. A neural network has no equivalent discrete structure to exploit this way — there's no small set of "nodes" whose combinatorics can be shared across feature subsets the way tree splits can. That's why neural networks need model-agnostic approximations (like KernelSHAP, which treats the model as a black box and samples subsets rather than computing them exactly) — slower, and only approximate, unlike TreeSHAP's exact and polynomial computation for trees specifically.

---

## 8. Apple MLE Interview Q&A (on-device / practical flavor)

**Q: You want to explain individual on-device predictions to users without a server round-trip (e.g., "why did this transaction get flagged"). Which of the three methods is even feasible on-device, and what's the actual constraint?**
A: Only SHAP answers the question being asked ("why this one prediction"), and TreeSHAP's computation is self-contained — it walks the already-on-device tree structure, no external calls needed, which fits an on-device-privacy requirement well. The real constraint isn't feasibility in principle, it's cost: TreeSHAP's compute scales with the ensemble's total size (tree count × leaves × depth²), so a large boosted ensemble that computes SHAP values cheaply server-side may need pruning down, or explanations computed for only a subset of predictions, to fit on-device latency/battery budgets — the same per-prediction cost tension that shows up for on-device inference generally, just applied to explanation instead of prediction.

**Q: A personalization model trained per-device shows high MDI for a feature that's rarely populated (mostly missing/default) for most users. What's the concrete risk if this gets used to decide which features to keep in a slimmed-down on-device model?**
A: MDI's high-cardinality bias is a real risk here on its own, but there's an added on-device-specific one: a rarely-populated feature can accumulate meaningful impurity-reduction credit from the (small) subset of users who do have it populated, even if it contributes nothing for the majority — MDI doesn't distinguish "very important for a few users" from "moderately important for everyone" in its raw total. Before trusting this to decide what stays in a memory/latency-constrained on-device model, check permutation importance specifically *segmented* by "feature populated" vs. "feature missing" — if the feature only matters for a small slice of users, keeping it in a slimmed-down model may be a bad trade of memory/compute for benefit that only accrues to a few people, which the raw MDI number wouldn't tell you.

**Q: If you could only afford to compute one of the three importance methods on-device (e.g., as part of a periodic on-device model health check), which would you pick and why?**
A: MDI, specifically because it's free — it requires no extra scoring passes or per-prediction computation, just reading back bookkeeping the tree already produced during training, which matters when the check has to run within an already-tight on-device compute/battery budget alongside everything else the device is doing. It's not the most trustworthy of the three, so it should be read as a coarse "did anything change dramatically" signal (e.g., a feature's MDI swinging wildly between two model versions might flag a data pipeline issue) rather than as a basis for real feature-selection decisions — those decisions are better made server-side, off-device, where permutation importance or SHAP's extra cost is easier to absorb.

---

**One-line summary to remember:** *MDI = free, but biased toward high-cardinality features — good for a first look, not a final decision. Permutation importance = costs a re-score pass per feature, but measures real held-out predictive contribution — the trustworthy number before acting. SHAP = answers a different question entirely (why THIS prediction, not overall ranking), made computationally feasible for trees via TreeSHAP, but the most expensive of the three — use it when you need to explain one specific output, not to replace the other two.*
