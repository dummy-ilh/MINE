# Trees & Ensembles — Apple / Google Style Interview Practice Q&A

Mixed practice set, covering the full curriculum (Ch.1-9). Questions are phrased the way they tend to show up in Apple/Google-style ML interviews — often starting concrete/practical, then pushing into "why" follow-ups. Answers are kept in the same plain-language style as the rest of the curriculum.

---

### Warm-up / Foundations

**Q1. "Walk me through how a decision tree decides where to split."**
At every node, it checks every feature and every possible threshold for that feature, and picks whichever single (feature, threshold) pair reduces impurity the most — Gini or entropy for classification, variance/MSE for regression. It repeats this on each resulting child, growing the tree one split at a time. It's a greedy process — it never looks ahead to see if a "worse now" split might set up a better split later.

**Q2. "Why does a fully-grown, unpruned decision tree usually overfit?"**
Left alone, it keeps splitting until every leaf is pure (or has just one sample) — at that point it's basically memorized the training data, including its noise, not just the real underlying pattern. That's why pruning (limiting depth, minimum leaf size, or cost-complexity pruning) exists.

**Q3. "Gini impurity or entropy — which would you pick, and does it matter?"**
In practice it barely matters — the two usually agree on the best split. Gini is cheaper to compute (no logarithm), which is why it's the common default. This is a good question to answer honestly rather than inventing a strong preference that isn't really justified.

---

### Bagging & Random Forest

**Q4. "Explain Random Forest to someone who's never heard of it, in under 30 seconds."**
Train a bunch of decision trees, each on a random resample of the data, and each only allowed to consider a random subset of features at every split. Average all their predictions together. The randomness in both rows and features makes the trees different enough from each other that averaging them smooths out each individual tree's mistakes.

**Q5. "If I keep adding more trees to my Random Forest, will it eventually start overfitting?"**
No. Each tree is trained completely independently, so adding more of them only ever averages away more noise — it can't add new overfitting risk the way, say, adding more boosting rounds can. More trees costs more compute/memory, but it won't hurt accuracy.

**Q6. "Why does Random Forest randomly restrict which features each split can consider, instead of just letting every tree see every feature like in plain bagging?"**
If one or two features are clearly the strongest predictors, plain bagging's trees will nearly all pick those same features for their top splits anyway, no matter which random rows they trained on — so the trees end up fairly similar to each other, and averaging similar trees doesn't help much. Forcing each split to only consider a random handful of features means the strongest feature isn't always available, pushing trees to genuinely differ from each other more — and averaging genuinely different trees helps a lot more than averaging similar ones.

**Q7. "What's Out-of-Bag error, and why would you use it instead of a separate validation set?"**
Since each tree is trained on a random resample, roughly a third of the original data is left out of any given tree's training set. For each data point, you can get an honest prediction from just the trees that never saw it, and compare that to the real answer — giving you a validation-like error estimate for free, without setting aside any data purely for validation.

---

### Boosting

**Q8. "How is boosting fundamentally different from bagging?"**
Bagging trains many trees independently, all at once, and averages them to smooth out unstable, high-variance mistakes. Boosting trains trees one after another, where each new tree is specifically aimed at correcting whatever the current combined model is still getting wrong — it's targeting systematic, repeatable mistakes (bias), not random instability.

**Q9. "Why does Gradient Boosting typically use small, shallow trees, while Random Forest often uses big, deep ones?"**
Random Forest wants each individual tree to be as strong and detailed an opinion as possible, since the averaging step is what handles instability. Boosting is taking a series of small, cautious correction steps — a big, detailed tree at any one boosting round would try to fix everything in one shot, which is exactly the kind of aggressive move that leads boosting to start fitting noise instead of real signal.

**Q10. "You're training a Gradient Boosting model and notice training error keeps dropping but validation error has started rising. What's happening, and what would you do?"**
Classic boosting overfitting — extra rounds are now fixing mistakes that were actually just noise, not real signal. Fixes: stop training earlier (early stopping based on validation error), lower the learning rate, reduce tree depth, or use a smaller number of rounds overall.

**Q11. "What does XGBoost actually add on top of plain Gradient Boosting?"**
Two big things: it directly penalizes trees for being too complex (too many leaves, or leaves with extreme prediction values) as part of what it's optimizing, so it naturally builds simpler, more conservative trees. And it uses curvature information (not just the basic slope/gradient) to decide how big a correction each step should make, which tends to get to a good answer in fewer, more accurate steps.

**Q12. "Your dataset is huge — millions of rows — and training is too slow. Would you switch from XGBoost to LightGBM, and why?"**
Likely yes. LightGBM buckets feature values into a fixed number of bins before searching for splits, instead of checking every possible threshold, which is much faster on large data. It also grows trees by chasing whichever single leaf would benefit most from splitting, rather than expanding every branch evenly — spending its computation where it matters most. Trade-off: keep an eye on `num_leaves`, since this leaf-wise growth can overfit more easily on smaller datasets (less of a concern here, given the data size).

**Q13. "You have a dataset full of categorical features with many unique values — what's your approach?"**
Reach for CatBoost or LightGBM, both of which handle categorical features natively without needing manual one-hot encoding — which would otherwise blow up into hundreds of sparse columns and make it harder for the model to find good splits. CatBoost specifically also guards against a subtle leakage issue where a categorical encoding can end up quietly using a sample's own target value to encode itself.

---

### Stacking & Practical Judgment

**Q14. "Would you rather stack a Random Forest and an XGBoost model, or a Random Forest and three different XGBoost models with different seeds? Why?"**
The first option — Random Forest and XGBoost are genuinely different algorithms that tend to make different kinds of mistakes, giving a meta-learner real signal to combine. Three XGBoost models with different seeds are likely to fail on similar houses/cases in similar ways, so there's much less for stacking to gain over just averaging them.

**Q15. "What's the single most common bug when people build a stacked model, and why does it matter?"**
Leakage — training the meta-learner on base models' predictions on data those same base models were trained on. Those predictions look artificially accurate (the models have partially memorized those rows), so the meta-learner learns from dishonest inputs and performs worse than expected on genuinely new data. The fix is generating base model predictions using out-of-fold or held-out data only.

**Q16. "When would you choose a linear model over a tree-based model, even if the tree-based model tests slightly more accurate?"**
When you need simple, direct explanations for predictions (clean per-feature coefficients), when you expect the model to need to extrapolate beyond the range of training data, or when you have very little data and want the extra stability that comes from a simpler model with fewer moving parts.

**Q17. "Explain to a non-technical product manager why your model priced one specific house the way it did."**
This is where SHAP values fit in — rather than explaining the whole model, SHAP breaks down one specific prediction into how much each feature pushed it up or down from the average. E.g., "this house priced $120K above average: +$80K from its unusually large size, +$50K from its location, −$10K from its age" — a clear, additive, per-prediction explanation a non-technical audience can follow.

**Q18. "Your model predicts that a bigger house is worth less than a smaller one in the same neighborhood, all else equal. Business stakeholders are unhappy — how do you fix this?"**
Add a monotonic constraint forcing the prediction to never decrease as size increases. It's enforced directly during tree-building — the split search simply isn't allowed to consider splits that would violate the constraint — rather than trying to patch the finished model's predictions after the fact.

---

### Rapid-fire (short-answer style, common in phone screens)

- **"Does more trees in Random Forest ever hurt?"** → No.
- **"Does more rounds in Gradient Boosting ever hurt?"** → Yes, can overfit.
- **"Can Random Forest trees train in parallel?"** → Yes.
- **"Can boosting rounds train in parallel?"** → No, each depends on the last.
- **"What does a decision tree use to decide the best split?"** → Gini/entropy (classification) or variance/MSE (regression) reduction.
- **"What's the main risk of impurity-based feature importance?"** → Biased toward features with many possible split points.
- **"What fixes that bias?"** → Permutation importance.
- **"What's the fix for stacking leakage?"** → Out-of-fold predictions for the meta-learner.
- **"Can trees extrapolate beyond training data's range?"** → No.
- **"Which handles missing values natively, sklearn's trees or XGBoost?"** → XGBoost (and LightGBM); plain sklearn trees do not.

---

Want a second, harder-difficulty pass (system-design-style questions — e.g., "design an ML pipeline for X using tree-based models," latency/scale trade-offs), or is this the right level to practice with?
