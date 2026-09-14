# Evaluation & Tuning — Master Notes

## 0. Quick Map (read this first)

| Topic | Core question it answers | The one thing to remember |
|---|---|---|
| Hyperparameter tuning | "How do I search for good settings without trying everything?" | Grid for 1–2 knobs, random for several, Bayesian when each run is expensive |
| Feature importance | "Which inputs does my model actually rely on?" | Impurity to skim for free, permutation to trust, SHAP to explain one case |
| Class imbalance | "My rare class is the one that matters — how do I stop the model from ignoring it?" | Never trust accuracy alone; fix with class weights or resampling |

**One-line theory:** All three topics are really about the same underlying trap — **a metric or a search process can look fine while quietly optimizing for the wrong thing** (the majority class, a spuriously flexible feature, an untested hyperparameter region). Evaluation & tuning is the discipline of catching that before deployment, not after.

---

## 1. Theory & Practice — Hyperparameter Tuning Strategies

### 1.1 The four methods at a glance

| Method | Plain idea | Best for |
|---|---|---|
| **Grid search** | Pick a small list of values per hyperparameter, try every combination | 1–2 hyperparameters you know matter a lot, with a few sensible values each |
| **Random search** | Try a fixed number of *random* combinations instead of every combination | Tuning several hyperparameters at once |
| **Bayesian optimization** | Use results from combinations already tried to make an educated guess about what to try next | When each training run is expensive and you want to search smartly, not blindly |
| **Early stopping (boosting only)** | Keep adding rounds, watch validation error, stop once it stops improving | Sidesteps tuning "how many rounds" as its own grid dimension entirely |

### 1.2 The theory behind why grid search struggles

**Why grid search struggles fast (the combinatorial explosion):** combinations explode — 3 values × 3 values is only 9, but 5 hyperparameters × 4 values each is 1,024.

**Worked numeric example — the explosion in practice:**

Suppose you want to tune a Random Forest with 4 hyperparameters, and you (reasonably) pick 4 candidate values for each:

| Hyperparameter | Candidate values |
|---|---|
| `n_estimators` | 100, 200, 400, 800 |
| `max_depth` | 4, 8, 16, None |
| `min_samples_leaf` | 1, 2, 5, 10 |
| `max_features` | "sqrt", "log2", 0.5, 1.0 |

Grid search total combinations = 4 × 4 × 4 × 4 = **256 full training runs**. If each run takes 10 minutes, that's **~43 hours** of sequential compute just for the search — before you've touched the model that ships.

Compare that to **random search with a budget of 30 runs**: you get to try 30 combinations spread across that same 256-point space, at ~5 hours of compute — roughly 8x cheaper — and (per Bergstra & Bengio's argument below) you likely find a *near-as-good* setting anyway, because most of the 256 grid points were wasted finely gridding hyperparameters that don't move the needle much.

### 1.3 Why random search often beats grid search anyway (the actual mechanism)

**Bergstra & Bengio, 2012 — the key insight:** most hyperparameters don't matter equally. Random search naturally spends more of its budget exploring the ones that actually move the needle, while grid search wastes effort finely gridding hyperparameters that barely matter.

**Why, mechanically:** Imagine (as is common in practice) that only 1 of your 4 hyperparameters is "important" (has a big effect on validation error) and the other 3 barely matter. In a 4×4×4×4 grid, only 4 *distinct values* of the important hyperparameter are ever tried — no matter how many total grid points you use, because the grid structure forces you to repeat those same 4 values across every combination of the other three. In random search with the same total budget, every single trial gets a fresh, different value of the important hyperparameter — so with 30 random trials, you've effectively tried 30 different values of the one hyperparameter that matters, not just 4.

> **Interview soundbite:** *"Grid search's problem isn't that it's exhaustive, it's that it's exhaustive over the wrong axis — it spends equal resolution on hyperparameters that matter and ones that don't. Random search reallocates that resolution automatically, without you having to know in advance which hyperparameter is the important one."*

### 1.4 Bayesian optimization — theory and analogy

**Plain analogy:** grid/random search is like buying lottery tickets — each one is chosen without learning from the last. Bayesian optimization is a treasure hunter who updates their guess about where to dig next based on what they've already found.

**A bit more theory (for interviews that go deeper):** Bayesian optimization builds a probabilistic *surrogate model* (commonly a Gaussian Process) over "hyperparameter setting → validation score," using every run tried so far. It then uses an **acquisition function** (e.g., Expected Improvement) to pick the next point to try — balancing:
- **Exploitation:** trying settings near the best one found so far, and
- **Exploration:** trying settings in regions the surrogate model is still uncertain about.

This is exactly why it shines when runs are expensive — every single trial gets to be informed by all prior trials, instead of being chosen blind.

### 1.5 Putting it together (house-price example)

Grid-search `max_depth` (big, clear effect, few distinct values worth trying), random-search `min_samples_leaf` and `max_features` together (many reasonable values, effects interact), and use early stopping instead of manually tuning `n_estimators` at all.

### 1.6 Practical decision checklist

1. **How many hyperparameters am I tuning?** 1–2 with known-good ranges → grid is fine and easy to reason about. 3+ → random search as a strong default.
2. **How expensive is one training run?** Seconds/minutes and parallelizable → random search, just run more trials. Hours+ → Bayesian optimization, since wasted trials are costly.
3. **Is one of my "hyperparameters" actually a training-length knob (rounds/epochs)?** Use early stopping instead of putting it in the search grid at all — it's strictly more efficient because it finds the right value in a single run instead of one run per candidate value.
4. **Am I short on total budget?** A common hybrid: a handful of random trials first to seed a rough map of the space, then Bayesian optimization to refine, especially useful in interviews as a "sophisticated but pragmatic" answer.

---

## 2. Theory & Practice — Feature Importance

### 2.1 The three methods at a glance

| Method | Plain meaning | Watch out for |
|---|---|---|
| **Impurity-based (MDI)** | How much did splits on this feature reduce impurity, on average, across the forest? | Free to compute, but biased toward features with lots of possible split points (continuous or high-cardinality) — can overrate a flexible-but-useless feature |
| **Permutation importance** | If I scramble this feature's values, how much worse do predictions get? | More trustworthy, but costs extra compute — one re-score per feature |
| **SHAP** | For this one specific house, how much did each feature push the prediction up or down from the average? | Most detailed (per-prediction, not just overall ranking), also the most expensive |

### 2.2 Theory — why impurity-based importance is biased

A tree-building algorithm picks, at each split, whichever feature/threshold combination reduces impurity the most **on the training data available at that node**. A continuous feature (or a categorical one with many distinct values) offers many candidate thresholds to try — so purely by chance, at least one of those thresholds is likely to look like it reduces impurity somewhat, even if the feature carries zero real signal. A binary feature only ever offers one possible split, so it never gets this "many chances to get lucky" advantage. The result: impurity importance systematically inflates high-cardinality/continuous features relative to their true predictive value.

**Worked numeric example — impurity importance being fooled:**

Imagine two features for predicting house price:
- `has_garage` — binary (yes/no), genuinely predictive (garages add ~$15k on average).
- `listing_id_hash` — a random hash of the internal listing ID, continuous, carries **zero real signal**.

Because `listing_id_hash` has (say) 1,000 distinct values in the training set, a sufficiently deep, unpruned tree can find splits on it that partition the training houses into groups with different average prices *purely by chance* — with 1,000 possible thresholds to try, some will look like they reduce training impurity noticeably, even though this is training-set noise, not real signal. `has_garage`, meanwhile, only offers one possible split point ever.

Result: impurity-based importance might rank `listing_id_hash` **above** `has_garage`, even though on a held-out test set, permuting `has_garage` would hurt predictions a lot (it's real signal) while permuting `listing_id_hash` would barely move test error at all (it was never real signal, just noise the tree overfit to). This mirrors the ID-column overfitting trap from tree induction — a high-cardinality identifier-like feature is close to worst-case for this bias.

### 2.3 Theory — why permutation importance fixes this

Permutation importance is computed on **held-out data**, and directly measures the thing you actually care about: how much does scrambling this feature hurt real predictive performance? Because `listing_id_hash`'s apparent training-time signal was noise, scrambling it on held-out data changes almost nothing — its permutation importance collapses toward zero, correctly reflecting that it's useless.

> **Interview soundbite:** *"Impurity importance asks 'how much did this feature help fit the training data,' which rewards flexibility. Permutation importance asks 'how much does this feature actually help predict new data,' which is the question you meant to ask in the first place."*

### 2.4 SHAP — theory and when it earns its cost

SHAP (SHapley Additive exPlanations) borrows from cooperative game theory: treat each feature as a "player" contributing to the "payout" (the prediction for one specific house), and fairly distribute credit for how far that prediction landed from the average prediction, across all features, accounting for interactions between them. This gives a **per-prediction** breakdown, not just a global ranking — e.g., "for *this* house, `sqft` pushed the price up $30k, but `distance_to_downtown` pushed it down $12k."

### 2.5 How to choose (practical decision rule)

Impurity-based for a quick free first look, permutation importance before making a real decision off the ranking (like dropping features), SHAP when you need to explain one specific prediction to someone (a stakeholder, a regulator, a customer asking "why was my loan denied / my house valued at X").

---

## 3. Theory & Practice — Handling Class Imbalance

### 3.1 The problem, stated numerically

**The problem:** predicting "will this house sell within 30 days," and only 5% do. A model can hit 95% accuracy by always guessing "no" — while being completely useless.

**Worked numeric example — the accuracy trap in full:**

Say your test set has 1,000 houses: 50 "sold within 30 days" (positive), 950 "didn't" (negative).

A model that **always predicts "no"** achieves:
- Accuracy = 950 / 1,000 = **95%**
- Recall on the positive class = 0 / 50 = **0%** — it catches literally zero of the houses that matter
- Precision on the positive class = undefined (it never predicts positive at all)

Compare to a genuinely useful model that catches 35 of the 50 fast-sellers, at the cost of 60 false alarms:
- True positives = 35, False negatives = 15, False positives = 60, True negatives = 890
- Accuracy = (35 + 890) / 1,000 = **92.5%** — *lower* accuracy than the useless model!
- Recall = 35 / 50 = **70%** — actually useful
- Precision = 35 / (35 + 60) = **36.8%**

This is the crux of the whole section: **the useless model has higher accuracy than the useful one.** If accuracy is the only number you look at, you'd pick the worse model.

### 3.2 Fix 1 — class weights (the theory)

`class_weight='balanced'` in sklearn weights each class inversely to how common it is, so a wrong guess on a rare "sold quickly" house counts for much more than a wrong guess on a common "didn't sell" house. Pushes the model to actually pay attention to the minority class.

**Concretely, in the example above:** with 50 positives and 950 negatives, `balanced` weighting assigns each positive example a weight roughly `1000 / (2 × 50) = 10`, and each negative example a weight roughly `1000 / (2 × 950) ≈ 0.53`. During training, getting a positive example wrong now costs the loss function about **19x** more than getting a negative example wrong (10 / 0.53 ≈ 19) — which is exactly the imbalance ratio (950/50 = 19), inverted. The model is mathematically nudged to stop treating "always predict no" as a good strategy.

### 3.3 Fix 2 — resampling (the theory)

Oversample the rare class (duplicate/synthesize more examples) or undersample the common class (throw some away) so training data is more balanced.

- **Oversampling:** duplicate the 50 positive examples (or synthesize new similar ones, e.g. via SMOTE) until the classes are closer to balanced — e.g., duplicate 19x to get ~950 vs. 950.
- **Undersampling:** randomly discard negative examples down to, say, 50 vs. 50 — much less data overall, but balanced.

**Trade-off, in one line:** oversampling risks overfitting to the (possibly noisy) minority examples you have, since you're seeing the exact same handful of points many times; undersampling risks throwing away real, useful information about the majority class.

### 3.4 Fits naturally with bagging

Bagging already resamples rows for every tree — imbalance handling can be built directly into that same step, drawing each bootstrap sample to be more balanced than the raw data, rather than bolting on a separate resampling stage beforehand.

> **Interview soundbite:** *"Balanced bagging isn't a new algorithm — it's just changing what 'random sample of rows' means for each tree's bootstrap draw, so imbalance handling rides for free on machinery you already have."*

### 3.5 Why not just check accuracy?

With 95% "didn't sell," accuracy stays misleadingly high even for a useless model. Use precision/recall, F1, or the confusion matrix instead — these actually show whether the model catches the rare, important cases, which is usually the whole point.

**Quick reference — which metric answers which question:**

| Metric | Question it answers | When it's the one that matters |
|---|---|---|
| Precision | "Of the houses I flagged as fast-sellers, how many actually were?" | False alarms are costly (e.g., sales team wastes time chasing bad leads) |
| Recall | "Of the actual fast-sellers, how many did I catch?" | Missing a positive is costly (e.g., fraud detection, disease screening, fall detection) |
| F1 | Harmonic mean of precision & recall — a single balance-point number | You need one number and care about both errors roughly equally |
| Confusion matrix | The full raw breakdown (TP/FP/FN/TN) | You want to reason about costs yourself rather than trust a pre-baked formula |

---

## 4. Quick Q&A (general)

**Q: Grid search or random search — which and when?**
A: Random search when tuning several hyperparameters at once — it explores more efficiently and doesn't waste budget finely gridding hyperparameters that don't matter much. Grid search is fine, and easier to reason about, with just one or two hyperparameters you know matter a lot.

**Q: Your model gets 95% accuracy predicting a rare event. Happy?**
A: Not necessarily — check the "always guess the common class" baseline first. If 95% of cases are the common class, a model that ignores the rare class entirely already hits 95% while being useless. Look at precision/recall or the confusion matrix instead.

**Q: Why might impurity-based feature importance mislead you?**
A: It tends to rate features with many possible split thresholds (continuous or high-cardinality) as more important than they really are, since they get more chances to find a split that reduces training impurity somewhat by chance. Permutation importance, which measures actual held-out predictive value, doesn't have this bias.

---

## 5. Google MLE Interview Q&A

**Q: You have a training pipeline where each run takes 6 hours and you have a fixed budget of 20 runs to tune 4 hyperparameters. Would you use grid search, random search, or Bayesian optimization, and why?**
A: With only 20 runs and 4 hyperparameters, grid search is essentially ruled out — even 3 values each is already 81 combinations, far past budget. Between random search and Bayesian optimization: random search is a reasonable baseline and trivially parallelizable (all 20 runs can be launched at once), but with an expensive 6-hour-per-run cost, Bayesian optimization's core advantage — using earlier results to pick smarter next candidates — is worth the extra sequential coordination, since each wasted run is expensive. In practice a hybrid is common: a handful of random runs first to seed the search space, then Bayesian optimization for the remaining budget.

**Q: A model shows 95% accuracy in offline eval but the on-call team reports it's "missing almost everything that matters" once deployed. Walk through how you'd diagnose this using what's in this chapter.**
A: First check the class balance of the eval set — if the target event is rare (e.g., 5%), 95% accuracy could just be the "always predict the majority class" baseline, meaning the model may be contributing nothing beyond that baseline. Recompute using precision/recall/F1 or the full confusion matrix on the same eval set to see if recall on the rare class is actually near zero — that would confirm the "missing almost everything" complaint is about the minority class specifically, and points toward class-weighting or resampling (Section 3) as the fix, not toward tuning `max_depth` or other capacity knobs that wouldn't address a class-balance problem at all.

**Q: You compute impurity-based feature importance and a continuous "user ID hash" feature ranks near the top, well above features you'd expect to matter. What's happening, and what would you check next?**
A: This is the classic impurity-importance bias toward high-cardinality/continuous features — a feature with many possible split thresholds gets many chances to find a split that reduces training impurity by pure chance, inflating its apparent importance even with zero real signal (an ID-like feature is close to worst-case for this, similar to the ID-column overfitting trap from tree induction). Next step: recompute with permutation importance, which measures actual held-out predictive value rather than training-set impurity reduction — if the ID feature's importance collapses under permutation importance, that confirms it was an artifact, and the feature should likely be dropped or investigated for a possible data leak (an ID column ranking highly can also sometimes indicate leakage, e.g. IDs assigned in a way that correlates with the label).

---

## 6. Apple MLE Interview Q&A (on-device / practical flavor)

**Q: You're tuning a model that will ship on-device, where you can only run a small number of full training + on-device profiling cycles because each cycle needs a device farm run, not just a training run. How does this change your choice of tuning strategy from the standard grid/random/Bayesian menu?**
A: The expensive step here isn't training alone — it's training *plus* on-device profiling (latency, memory, battery) per candidate, so the effective cost per trial is much higher than a typical cloud-only sweep. That pushes hard toward Bayesian optimization or a similarly sample-efficient method over grid or plain random search, since minimizing the number of full training-and-profile cycles matters more than it would in a cloud setting where extra runs are comparatively cheap. It's also worth tuning parameters that affect both accuracy *and* on-device cost jointly (like `n_estimators`, `max_depth`) as part of the same search rather than tuning for accuracy first and checking on-device cost afterward — a setting that looks best offline might be a poor fit once profiling cost is factored in, and re-running the search from scratch after the fact wastes the exact expensive cycles you were trying to conserve.

**Q: A personalization model trained per-device shows strong permutation importance for a feature that's only non-null for a small fraction of users. What's the practical concern with shipping the model as-is?**
A: Permutation importance measures how much predictions degrade when that feature is scrambled — if it ranks highly, the model has learned to lean on it meaningfully. But if the feature is only populated for a small slice of users, the model's real-world usefulness for everyone else depends on how it behaves when that feature is missing/imputed, which is a scenario permutation importance (computed on the eval set as a whole) doesn't specifically stress-test. Worth explicitly checking model behavior segmented by "has this feature" vs. "doesn't," since a feature that's important *on average* can still mean the model performs unevenly across a user base where the feature's availability itself varies — an on-device fairness/quality concern as much as an accuracy one.

**Q: Class imbalance shows up differently on-device than server-side — e.g., a rare-event detector (like a fall-detection feature) needs to be tuned per-device from very few positive examples locally. Which of the two imbalance fixes in this chapter transfers better to that setting, and why?**
A: Class weighting transfers more cleanly than resampling in a very-low-data local setting. Resampling (oversampling the rare class) needs enough real examples to duplicate or synthesize meaningfully from — with only a handful of true positives locally, oversampling mostly just repeats the same few examples many times, which can make the model overconfident about those specific instances rather than generalizing. Class weighting doesn't need extra examples to work — it just changes how costly a mistake on the existing rare examples is treated during training, so it degrades more gracefully when positive examples are scarce, which is the common case for rare on-device events trained or fine-tuned per-user.

---

## 7. Interview-Ready Soundbites (collected in one place)

Use these as crisp, one-breath answers when an interviewer asks about tuning, feature importance, or imbalance:

1. *"Grid search's real weakness isn't computation, it's that it spends equal resolution on hyperparameters that matter and ones that don't — random search reallocates that resolution automatically."*
2. *"Bayesian optimization is worth its coordination overhead exactly when each trial is expensive — it's the only method of the three where every trial is chosen using everything learned from every prior trial."*
3. *"Early stopping isn't a fourth item on the tuning menu, it's a way to remove one dimension (training length) from the search entirely, in a single run instead of one run per candidate value."*
4. *"Impurity importance rewards features for having many chances to look useful during training; permutation importance checks whether that usefulness survives contact with held-out data."*
5. *"A high-cardinality ID-like feature ranking suspiciously high in impurity importance isn't a red flag about the feature — it's a red flag about the importance method."*
6. *"SHAP earns its computational cost when the question changes from 'what matters overall' to 'why did the model say this, for this one case.'"*
7. *"Accuracy on an imbalanced target measures how well you'd do by ignoring the thing you actually care about — that's the whole reason it stays high even for a useless model."*
8. *"Class weighting changes the cost of being wrong; resampling changes what the model gets to see. In low-data settings, weighting degrades more gracefully because it doesn't need extra examples to work."*
9. *"Balanced bagging isn't new machinery — imbalance handling rides for free on the same bootstrap-sampling step bagging already does."*

---

## 8. Practice Q&A (new — to mirror the format used elsewhere in the curriculum)

**Q1 (easy).** Why would you ever choose grid search over random search?
<details><summary>Answer</summary>When you're only tuning 1–2 hyperparameters that you already know matter a lot, and you have a small, sensible list of candidate values for each. In that regime the combinatorial explosion is mild, and grid search's exhaustiveness (and easier interpretability — you can look at a full 2D heatmap of results) outweighs random search's efficiency advantage, which mostly shows up once you're tuning 3+ hyperparameters at once.</details>

**Q2 (medium).** You permutation-test a feature and its importance comes back essentially zero, but the same feature had high impurity-based importance. What should you conclude, and what shouldn't you conclude?
<details><summary>Answer</summary>You should conclude the feature likely isn't providing real predictive signal on held-out data — its apparent importance during training was probably an artifact of it being continuous/high-cardinality (many chances to find a training-set-only pattern). You shouldn't immediately conclude the feature is definitely useless in every context — it's worth a quick sanity check for bugs (e.g., is the permutation being applied correctly, is there some interaction effect permutation importance underweights) before dropping it, but the default working conclusion should be that impurity importance was misleading here.</details>

**Q3 (medium).** A dataset has 1% positive class. A model achieves 99% accuracy. A colleague says "that's basically as good as it gets." What's the one number you'd ask for next, and why?
<details><summary>Answer</summary>Recall on the positive class (or the full confusion matrix). With a 1% positive rate, "always predict negative" already achieves 99% accuracy while catching zero positives — so 99% accuracy alone is uninformative about whether the model is doing anything useful. Recall tells you directly what fraction of the cases that matter are actually being caught.</details>

**Q4 (hard).** Explain, in terms of the acquisition function trade-off, why Bayesian optimization doesn't just always pick the point predicted to have the best score.
<details><summary>Answer</summary>If Bayesian optimization only exploited (always picked the surrogate model's current best-predicted point), it could get stuck in a locally good region while missing a better region it hasn't sampled yet — the surrogate's confidence there is just an artifact of having no data, not evidence it's bad. Acquisition functions like Expected Improvement balance exploitation (near the current best) against exploration (regions of high uncertainty), because a point the model is very uncertain about could turn out to be much better than anything tried so far — sampling it is informative even if it doesn't pan out.</details>

**Q5 (hard — practical judgment).** Your team wants to both oversample the minority class AND apply class weighting on top, "to be extra safe" about imbalance. What's the risk with stacking both fixes?
<details><summary>Answer</summary>They attack the same problem the same way — oversampling already changes the effective class ratio seen during training, and weighting on top of that further inflates the loss contribution of the (already duplicated) minority examples. Stacking both can over-correct, pushing the model to over-predict the minority class and hurt precision, and it also makes the "effective imbalance ratio" harder to reason about/tune deliberately. Usually better to pick one lever, tune it, and only add the second if the first alone under-corrects, checking precision/recall together (not just recall) so you can see if you've overshot.</details>

---

**One-line summary to remember:** *Random search (or Bayesian, if runs are expensive) beats grid search once you're tuning more than 1-2 hyperparameters → use impurity importance to skim, permutation importance to decide, SHAP to explain one prediction → for imbalanced targets, fix it with class weights or resampling and check precision/recall/confusion matrix, never plain accuracy.*
