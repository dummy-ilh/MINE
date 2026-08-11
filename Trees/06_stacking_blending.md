# Chapter 6 — Stacking & Blending

Same house-price example. So far every method in this curriculum has combined many *copies of the same kind of model* (many trees). Stacking is different: it combines **different kinds of models entirely**, and learns the best way to combine them.

---

## 6.1 The Core Idea

Say you've trained three very different models on the house-price data: a Random Forest, an XGBoost model, and a plain linear regression. Each one makes its own guess for every house. Instead of just averaging their three guesses equally, stacking trains a **fourth, small model** — called the **meta-learner** — whose job is: "given these three models' guesses for a house, what's the best combined guess?"

**Plain analogy:** imagine three different appraisers each give you their price estimate for a house. Instead of just averaging the three numbers, you hire a fourth person whose whole job is to learn, over time, that "Appraiser A tends to run a bit high on older houses, Appraiser B is usually spot-on for expensive homes, Appraiser C is best on small houses" — and combine their three numbers accordingly, weighting each one differently depending on the situation. That fourth person is the meta-learner.

**Why this can beat even the best single model, or a simple average:** different model types make *different kinds* of mistakes. A linear model might be bad at capturing weird non-linear price jumps but good at smooth overall trends; a tree-based model might be great at capturing those jumps but occasionally noisy on smooth trends. If those mistakes don't overlap much, a meta-learner that's learned *when to trust which model* can do better than any one model alone, and better than a plain unweighted average (which treats every model as equally trustworthy in every situation).

---

## 6.2 Avoiding the Leakage Trap

**The naive (wrong) way to build a stack:** train your three base models (Random Forest, XGBoost, linear regression) on all 1,000 houses. Have them each predict on those same 1,000 houses. Use those predictions as input to train the meta-learner.

**Why this is broken:** each base model has already *seen* every house it's predicting on during its own training — so its "prediction" on a house it trained on is suspiciously close to the true answer (it's basically just recalling, not really predicting, especially for a model like an unpruned tree that can memorize training data closely — Ch.1.5's high-variance behavior). The meta-learner ends up training on predictions that are artificially too good, and it won't work nearly as well once it sees a genuinely new house.

**The fix — out-of-fold predictions, same "no peeking" idea as Chapter 5.6's CatBoost fix:**

1. Split the 1,000 houses into, say, 5 folds of 200 each.
2. For each fold: train the base models on the *other* 4 folds (800 houses), then predict on this held-out fold (200 houses) — these 200 predictions are "honest," since those houses were never seen during that training.
3. Repeat for all 5 folds, so every house eventually gets an honest, held-out prediction from each base model.
4. Train the meta-learner on these honest predictions.
5. For a genuinely new house at prediction time: run it through all base models (now retrained on the full 1,000 houses) to get their three predictions, then feed those into the trained meta-learner for the final answer.

This is exactly the same principle as cross-validation itself: never let a model's prediction on a sample be influenced by having trained on that very sample.

---

## 6.3 Stacking vs. Bagging vs. Boosting — When Does Stacking Actually Help?

**Plain answer:** stacking is worth the extra complexity mainly when your base models are genuinely different in *how* they make mistakes — different algorithm types (tree-based + linear + maybe a neural net), not just different random seeds of the same algorithm. If all your base models are just slightly different Random Forests, stacking them barely beats just averaging them, because they all tend to get the same kinds of houses wrong (their mistakes overlap a lot) — there's not much for the meta-learner to actually learn.

**Blending — the simpler cousin of stacking:** instead of the full 5-fold out-of-fold setup, blending just holds out one single validation chunk (say, the last 200 of the 1,000 houses), trains base models on the rest, gets their honest predictions on that one held-out chunk, and trains the meta-learner on just that. Simpler and faster to set up, but the meta-learner sees less data (only 200 houses' worth of honest predictions instead of all 1,000), so it's a rougher, noisier version of the same idea.

**Where stacking sits relative to the rest of this curriculum:** Bagging (Ch.3-4) fights instability (variance) by averaging many similar models. Boosting (Ch.5) fights systematic blind spots (bias) by correcting mistakes round after round. Stacking doesn't neatly fit into "fixes bias" or "fixes variance" — it's really just "learn the smartest way to combine whatever good, differently-flawed models you already have." In real-world settings (like Kaggle competitions), it's common to see a Random Forest, an XGBoost model, and a LightGBM model all stacked together — since they're built differently enough under the hood (Ch.5b) that their mistakes genuinely don't overlap perfectly.

---

## Quick, Simple Interview Answers

**Q: "Why not just average your models' predictions instead of building a whole meta-learner?"**
A: Simple averaging treats every model as equally trustworthy everywhere. A meta-learner can learn that, say, the linear model should be trusted more for typical mid-size houses but less for unusual mansion-sized ones — a smarter, situation-dependent combination that a flat average can't express.

**Q: "What's the single most important thing to get right when building a stack?"**
A: Avoiding leakage — always generate the base models' predictions using out-of-fold (or held-out) data, never predictions on data those models were trained on, or your meta-learner will be trained on artificially inflated, dishonest inputs.

**Q: "When would stacking NOT be worth the extra complexity?"**
A: When your base models are all similar in how they fail (e.g., three Random Forests with different seeds) — there's little genuinely different information for the meta-learner to combine, so the gain over a simple average is small relative to the added complexity and risk of leakage bugs.

---

**Next up: Chapter 7 — Evaluation & Tuning for trees/ensembles. Want me to continue?**
