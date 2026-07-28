# Chapter 3 — Wrapper Methods

Chapter 2 covered filter methods — score each feature alone, no model involved. This chapter is the opposite extreme: **actually train models on different feature subsets and use real performance to guide the search.** Wrapper methods are the most direct way to optimize what you actually care about, at the cost of being the most computationally expensive family.

## 3.1 The core idea: search over subsets, not features

A filter method scores each feature independently and never asks "how does the model actually do with this exact combination of features?" A wrapper method asks exactly that question, repeatedly: pick a candidate subset of features, **train a real model on it**, measure its **real performance** (via cross-validation, so the measurement itself is trustworthy — see 3.4), and use that performance to decide what to try next.

**Why not just try every possible subset?** With n features, there are 2^n possible subsets (each feature is either in or out) — for even a modest 30 features, that's over a billion combinations. Exhaustively training and evaluating a model on every one is computationally infeasible for almost any real dataset. This is exactly why wrapper methods use a **greedy search strategy** rather than brute force — they build up (or tear down) a subset one feature at a time, using the model's measured performance at each step to decide the next move, which turns an exponential search into a much cheaper, roughly quadratic one (see 3.5 for the actual cost accounting).

## 3.2 Forward selection

**The procedure:**
1. Start with an empty set of selected features.
2. For every feature not yet selected, temporarily add it to the current set, train a model, and measure performance (via cross-validation).
3. Permanently add whichever single feature produced the best performance improvement.
4. Repeat steps 2–3, adding one feature at a time, until performance stops improving (or you reach a target number of features).

**Worked walkthrough.** Suppose you have 4 candidate features: `age`, `income`, `zip_code`, `credit_score`, and you're predicting loan default, measuring performance by cross-validated AUC.

- **Round 1:** try each feature alone. Suppose the results are: `age` alone → AUC 0.58; `income` alone → AUC 0.61; `zip_code` alone → AUC 0.55; `credit_score` alone → AUC 0.68. **Best single feature: `credit_score` (0.68).** Permanently keep it.
- **Round 2:** try adding each remaining feature to `credit_score`. Suppose: `credit_score + age` → 0.70; `credit_score + income` → 0.74; `credit_score + zip_code` → 0.69. **Best addition: `income` (0.74).** Keep `{credit_score, income}`.
- **Round 3:** try adding each remaining feature to `{credit_score, income}`. Suppose: `+ age` → 0.745; `+ zip_code` → 0.741. The improvement from adding `age` (0.74 → 0.745) is tiny, and if it's below your chosen stopping threshold, **you stop here** with the final set `{credit_score, income}` (or `{credit_score, income, age}` if you decide the marginal gain is still worth it, depending on your threshold).

Notice something important: `zip_code` never gets selected in this walkthrough, even though it might have looked somewhat useful on its own — because once `credit_score` and `income` are already in the model, `zip_code` adds very little *additional* information (it may be substantially redundant with `income`, a proxy relationship worth flagging, and one you'll see again from a fairness angle if you recall the proxy discussion from your Fairness & Responsible AI prep).

## 3.3 Backward elimination and stepwise selection

**Backward elimination** runs the same idea in reverse: start with *all* features included, and repeatedly remove whichever single feature's removal hurts performance the *least* (or improves it), stopping when removing anything further would hurt performance meaningfully. It's the mirror image of forward selection — same greedy, one-feature-at-a-time logic, just starting full and shrinking instead of starting empty and growing.

**Stepwise selection** combines both directions: at each step, consider both adding a not-yet-included feature and removing an already-included one, taking whichever single move improves performance most. This can recover from an earlier greedy mistake that pure forward or pure backward selection would have been stuck with — e.g., if an early-added feature turns out to be made redundant by a later addition, stepwise selection can remove it, whereas forward selection alone never reconsiders a feature once it's in.

**When to prefer which:** forward selection is cheaper when you expect only a small number of features to matter (you stop early, having trained far fewer models); backward elimination is cheaper when you expect most features to matter and only a few to be useless (you're only removing a handful of features rather than adding almost all of them one by one). Stepwise selection costs more than either alone but guards against the greedy-mistake failure mode of both.

## 3.4 Recursive Feature Elimination (RFE)

**The idea:** RFE is a specific, popular variant of backward elimination that uses a model's *own* internal importance signal (coefficients for a linear model, feature importances for a tree-based model — previewed here, covered in full in Chapters 5 and 7) to decide what to remove, rather than retraining on every possible remaining subset at each step.

**The procedure:**
1. Train the model on all features.
2. Rank features by the model's own importance measure (e.g., coefficient magnitude, or tree-based importance).
3. Remove the single lowest-ranked feature (or a fixed batch of the lowest-ranked features, for speed).
4. Retrain on the remaining features, and repeat from step 2, until you reach a target number of features.

**Why this is cheaper than plain backward elimination:** plain backward elimination has to try removing *every* remaining feature at each step (retraining once per candidate removal) to find the single best one to drop; RFE instead trusts the model's own importance ranking to pick the removal candidate directly, requiring only **one** retrain per round instead of one retrain **per remaining feature** per round. This is a meaningful speed difference — for k remaining features, plain backward elimination trains k models to decide one removal, while RFE trains just 1.

## 3.5 The cost of wrapper methods, made concrete

For n total features, forward selection (as walked through in 3.2) trains roughly n models in round 1 (one for the first feature choice), n−1 in round 2, n−2 in round 3, and so on — summing to roughly n²/2 total model trainings if you go all the way to a full ranking. That's a huge improvement over the 2^n subsets of brute force, but it's still quadratic in the number of features and, critically, **each of those "model trainings" should itself be a full cross-validation run**, not a single train/test split — multiplying the cost further. This is exactly why wrapper methods are typically applied *after* a cheap filter pass (Chapter 2) has already cut the feature count down from, say, thousands to a few dozen — running a quadratic search over a few dozen features is very feasible; running it over thousands is not.

## 3.6 Why cross-validation is essential here, not optional

Every step above says "measure performance" — and it's tempting to just check performance on a single held-out validation set. Here's why that's a mistake specific to wrapper methods: **you are performing a search over a huge number of candidate feature subsets, and any search over many candidates is prone to finding one that looks great on your validation set purely by chance**, the same way trying enough random seeds eventually finds one that happens to do well by luck. If you use the same single validation set to guide every single decision in the search, you risk **overfitting the feature selection process itself** to that one particular validation set — the selected features may look excellent on that set and generalize poorly elsewhere, because you've implicitly searched over many subsets looking for whichever one happens to fit the validation set's particular noise.

**The fix:** use cross-validation (average performance across multiple folds) at every comparison step in the search, not a single train/validation split — this makes each comparison more robust to any one split's particular noise, though it doesn't eliminate the risk entirely (with a genuinely huge number of candidate subsets, some degree of overfitting the selection process is nearly unavoidable, which is part of why wrapper methods are usually reserved for a moderate number of candidate features to begin with, per 3.5).

---

**Next: Chapter 4 — Embedded Methods**, where feature selection stops being a separate search step entirely and instead happens automatically as a side effect of training a single model — starting with L1/Lasso regularization and the geometric reason it drives coefficients exactly to zero.
