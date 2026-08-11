# Chapter 7 — Evaluation & Tuning

Same house-price example. This chapter is shorter — it's mostly tying together tuning knobs already introduced in Chapters 1-6, plus one new topic (imbalanced data).

---

## 7.1 Hyperparameter Tuning Strategies

**Grid search:** pick a small list of values for each hyperparameter (say, `max_depth` in [3, 5, 7] and `n_estimators` in [100, 300, 500]) and try every combination. Simple and thorough, but the number of combinations explodes fast if you tune more than 2-3 things at once — 3 values × 3 values is only 9 combos, but 5 hyperparameters with 4 values each is 1,024 combos.

**Random search:** instead of trying every combination, just try a fixed number of random combinations from the same ranges. Sounds worse, but there's a well-known result (Bergstra & Bengio, 2012) that random search usually finds a similarly good setting faster than grid search, because most hyperparameters don't matter equally — random search naturally spends more of its budget exploring the ones that actually move the needle, while grid search wastes a lot of effort finely gridding over hyperparameters that barely matter.

**Bayesian optimization:** instead of picking combinations blindly (grid) or randomly, use the results of combinations you've already tried to make an educated guess about which untried combination is most likely to be good — then try that one next, and repeat. Plain analogy: grid/random search is like trying lottery tickets; Bayesian optimization is like a treasure hunter who updates their guess about where to dig next based on what they've found so far.

**Early stopping for boosting (a shortcut specific to boosting, Ch.5):** instead of deciding `n_estimators` ahead of time, just keep adding rounds and watch validation error — stop as soon as it stops improving for a while (`n_iter_no_change` in sklearn's Gradient Boosting, Ch.5a's table). This sidesteps having to separately tune "how many rounds" as its own grid dimension.

**House price example, put together:** you might grid-search `max_depth` (a few clearly distinct values) since it has a big, clear effect, random-search `min_samples_leaf` and `max_features` together since there are many reasonable values and their effects interact, and use early stopping instead of manually tuning `n_estimators` at all.

---

## 7.2 Feature Importance, Recap and Comparison

Already covered in depth in Chapter 4.3 — quick recap table for reference:

| Method | Plain meaning | Watch out for |
|---|---|---|
| Impurity-based (MDI) | "How much did splits on this feature reduce impurity, on average, across the forest?" | Free to compute, but biased toward features with lots of possible split points (like continuous or high-cardinality features) — can overrate a useless-but-flexible feature |
| Permutation importance | "If I scramble this feature's values, how much worse do predictions get?" | More trustworthy, but costs extra compute since you have to re-score the model once per feature |
| SHAP (previewed here, full treatment in Ch.9) | "For this one specific house, how much did each feature push the prediction up or down from the average?" | Gives per-house explanations, not just an overall ranking — the most detailed option, also the most expensive to compute |

**Simple way to choose:** use impurity-based importance for a quick first look (it's free), switch to permutation importance if you're about to make a real decision based on the ranking (like dropping features), and reach for SHAP when you need to explain one specific prediction to someone (like explaining why one particular house got the price estimate it did).

---

## 7.3 Handling Class Imbalance with Trees

**The problem, simply:** say instead of predicting price, you're predicting "will this house sell within 30 days" — and only 5% of houses do. A model can get 95% accuracy just by always guessing "no," while being completely useless.

**Fix 1 — class weights:** tell the model to treat mistakes on the rare class as more costly. In sklearn's trees/forests, `class_weight='balanced'` automatically weights each class inversely to how common it is — so a wrong guess on one of the rare "sold in 30 days" houses counts for much more than a wrong guess on a common "didn't sell" house, pushing the model to actually pay attention to the minority class instead of ignoring it.

**Fix 2 — resampling:** either oversample the rare class (duplicate/synthesize more "sold quickly" examples) or undersample the common class (throw away some "didn't sell" examples) so the training data is more balanced.

**How this interacts with bagging specifically:** bagging (Ch.3) already resamples rows for every tree — you can build imbalance-handling directly into that same resampling step, so each bootstrap sample is deliberately drawn to be more balanced than the original data, rather than adding a completely separate resampling step beforehand. This is a natural fit since bagging is already in the business of resampling rows for every tree anyway.

**Why not just use plain accuracy to check if any of this worked?** With 95% of houses being "didn't sell," accuracy stays misleadingly high even for a useless model. Better metrics for this situation: precision/recall, F1 score, or looking at the confusion matrix directly — these actually show whether the model is catching the rare, important cases, which is usually the whole point of caring about the rare class in the first place.

---

## Quick, Simple Interview Answers

**Q: "Grid search or random search — which would you pick, and when?"**
A: Random search when tuning several hyperparameters at once, since it explores more efficiently and doesn't waste budget finely gridding hyperparameters that don't matter much. Grid search is fine, and easier to reason about, when you only have one or two hyperparameters you know matter a lot and just a few sensible values to try.

**Q: "Your model gets 95% accuracy predicting a rare event. Should you be happy?"**
A: Not necessarily — check what the "always guess the common class" baseline accuracy would be first. If 95% of cases are the common class, a model that ignores the rare class entirely already hits 95% accuracy while being useless. Look at precision/recall or the confusion matrix instead.

**Q: "Why might impurity-based feature importance mislead you?"**
A: It tends to rate features with many possible split thresholds (continuous, or high-cardinality categorical features) as more important than they really are, just because they get more chances to find a split that reduces training impurity somewhat by chance — permutation importance, which measures actual held-out predictive value, doesn't have this bias.

---

**Next up: Chapter 8 — Practical Interview Ground (when to use trees vs. linear models, overfitting diagnostics, complexity, and common traps). Want me to continue?**
