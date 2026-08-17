# Evaluation & Tuning — Master Notes

## 1. Hyperparameter Tuning Strategies

| Method | Plain idea | Best for |
|---|---|---|
| **Grid search** | Pick a small list of values per hyperparameter, try every combination | 1–2 hyperparameters you know matter a lot, with a few sensible values each |
| **Random search** | Try a fixed number of *random* combinations instead of every combination | Tuning several hyperparameters at once |
| **Bayesian optimization** | Use results from combinations already tried to make an educated guess about what to try next | When each training run is expensive and you want to search smartly, not blindly |
| **Early stopping (boosting only)** | Keep adding rounds, watch validation error, stop once it stops improving | Sidesteps tuning "how many rounds" as its own grid dimension entirely |

**Why grid search struggles fast:** combinations explode — 3 values × 3 values is only 9, but 5 hyperparameters × 4 values each is 1,024.

**Why random search often beats grid search anyway** (Bergstra & Bengio, 2012): most hyperparameters don't matter equally. Random search naturally spends more of its budget exploring the ones that actually move the needle, while grid search wastes effort finely gridding hyperparameters that barely matter.

**Bayesian optimization, plain analogy:** grid/random search is like buying lottery tickets. Bayesian optimization is a treasure hunter who updates their guess about where to dig next based on what they've already found.

**Putting it together (house-price example):** grid-search `max_depth` (big, clear effect, few distinct values worth trying), random-search `min_samples_leaf` and `max_features` together (many reasonable values, effects interact), and use early stopping instead of manually tuning `n_estimators` at all.

---

## 2. Feature Importance — Recap

| Method | Plain meaning | Watch out for |
|---|---|---|
| **Impurity-based (MDI)** | How much did splits on this feature reduce impurity, on average, across the forest? | Free to compute, but biased toward features with lots of possible split points (continuous or high-cardinality) — can overrate a flexible-but-useless feature |
| **Permutation importance** | If I scramble this feature's values, how much worse do predictions get? | More trustworthy, but costs extra compute — one re-score per feature |
| **SHAP** | For this one specific house, how much did each feature push the prediction up or down from the average? | Most detailed (per-prediction, not just overall ranking), also the most expensive |

**How to choose:** impurity-based for a quick free first look, permutation importance before making a real decision off the ranking (like dropping features), SHAP when you need to explain one specific prediction to someone.

---

## 3. Handling Class Imbalance

**The problem:** predicting "will this house sell within 30 days," and only 5% do. A model can hit 95% accuracy by always guessing "no" — while being completely useless.

**Fix 1 — class weights:** `class_weight='balanced'` in sklearn weights each class inversely to how common it is, so a wrong guess on a rare "sold quickly" house counts for much more than a wrong guess on a common "didn't sell" house. Pushes the model to actually pay attention to the minority class.

**Fix 2 — resampling:** oversample the rare class (duplicate/synthesize more examples) or undersample the common class (throw some away) so training data is more balanced.

**Fits naturally with bagging:** bagging already resamples rows for every tree — imbalance handling can be built directly into that same step, drawing each bootstrap sample to be more balanced than the raw data, rather than bolting on a separate resampling stage beforehand.

**Why not just check accuracy?** With 95% "didn't sell," accuracy stays misleadingly high even for a useless model. Use precision/recall, F1, or the confusion matrix instead — these actually show whether the model catches the rare, important cases, which is usually the whole point.

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

**One-line summary to remember:** *Random search (or Bayesian, if runs are expensive) beats grid search once you're tuning more than 1-2 hyperparameters → use impurity importance to skim, permutation importance to decide, SHAP to explain one prediction → for imbalanced targets, fix it with class weights or resampling and check precision/recall/confusion matrix, never plain accuracy.*
