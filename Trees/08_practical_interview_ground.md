# Chapter 8 — Practical Interview Ground

Same house-price example throughout. This chapter is the "big picture, practical judgment" one — the questions interviewers ask to see if you actually know when and why to reach for these tools, not just how the math works.

---

## 8.1 When to Reach for Trees/RF/GBM vs. Linear/Regularized Models

**Plain answer, the short version:** reach for tree-based models when you expect the relationship between features and target to be messy, non-linear, or full of interactions (this feature matters differently depending on that other feature's value). Reach for linear models when you expect a fairly smooth, additive relationship, or when you specifically need to explain the model with simple coefficients, or when you have very little data.

| Situation | Favors trees/RF/GBM | Favors linear/regularized models |
|---|---|---|
| Relationship shape | Non-linear, lots of interactions (e.g., "size matters a lot in the suburbs but barely in the city") | Roughly linear, additive effects |
| Data size | Works well with plenty of data; can struggle to find real signal with very little | Can work reasonably even with limited data, especially with regularization |
| Feature types | Handles a mix of numeric and categorical easily, no scaling needed | Needs features scaled, categorical variables encoded carefully |
| Need for a simple explanation ("each extra bedroom adds $X") | Harder — a tree/forest doesn't give clean per-feature coefficients (SHAP, Ch.9, helps but adds complexity) | Coefficients are directly interpretable out of the box |
| Extrapolation beyond the training data's range (predicting a 10,000 sq ft mansion when your data tops out at 4,000 sq ft) | Poor — trees can only predict values seen in training leaves, they can't extrapolate a trend | Can extrapolate a trend, for better or worse |
| Outliers/noisy data | Reasonably robust, especially bagging/RF | Can be thrown off by outliers unless using a robust loss |

**Why can't trees extrapolate?** A tree's prediction for any input is just the average (or majority class) of whatever training examples ended up in the matching leaf, per Chapter 1.1's fundamental "regions of feature space" idea. If nothing in training ever had a house bigger than 4,000 sq ft, there's no leaf built to handle a 10,000 sq ft house — the tree will just fall back to whatever leaf covers "biggest houses seen," which flatlines instead of continuing the price trend upward. A linear model, by contrast, will happily (and sometimes wrongly) keep extending its straight-line trend forever.

---

## 8.2 Overfitting Diagnostics Specific to Trees

**The classic tool: train vs. validation error, plotted against a complexity knob.** Plot both curves as you increase `max_depth` (or decrease `min_samples_leaf`, or increase `n_estimators` for boosting specifically, Ch.5.7). Training error keeps dropping the more complex you allow the tree to get — that part is expected and not by itself a warning sign. The warning sign is **validation error starting to rise while training error keeps falling** — that gap opening up is the direct, visual signature of overfitting.

**Tree-specific version of this check:** for a single decision tree, plot this against `max_depth`. For Random Forest, this curve is usually pretty flat/stable against `n_estimators` (Ch.4's "adding more trees doesn't overfit" result) — if you see validation error rising as you add more RF trees, that's a red flag suggesting a bug or a data problem, not normal behavior. For boosting, this curve against `n_estimators` **is** expected to eventually turn upward (Ch.5.7) — that's exactly the point where early stopping should kick in.

**House price example:** if a single decision tree's validation error is much worse than its training error at `max_depth=15`, but the gap nearly disappears at `max_depth=5`, that's a clear sign the deeper tree was memorizing specific houses (like remembering the exact price of one weird 1920s mansion) rather than learning a generalizable pattern.

---

## 8.3 Time & Space Complexity, Kept Simple

**Plain summary, no heavy notation:**

- **A single tree:** building it takes longer the more rows and more features you have — roughly, doubling your data or your feature count roughly doubles (or worse) the build time. Once built, making a prediction for one new house is fast — just walking down one path of the tree.
- **Random Forest / Bagging:** training time is roughly "single tree cost" × "number of trees" — but since every tree can be built at the same time on different CPU cores (Ch.3-4's parallel property), the real-world wall-clock time can stay much lower if you have multiple cores available. Prediction needs to walk every tree and average, so prediction time also scales with the number of trees, though this is usually still fast in absolute terms.
- **Boosting:** training time is also roughly "single tree cost" × "number of rounds," but rounds **must** happen one after another (Ch.5.7) — no shortcut from extra CPU cores here, since round 2 genuinely needs round 1 to finish first. This is the practical reason boosting often takes noticeably longer to train than Random Forest for a comparable number of trees.

**Memory, simply:** a Random Forest with hundreds of deep trees can end up using a fair bit of memory just to store all those trees' structures — this is a real practical consideration when deciding how many trees / how deep to go, separate from the accuracy question.

---

## 8.4 Common Interview Traps

**"Does Random Forest overfit as you add more trees?"**
No — already covered properly in Ch.4's Q&A, but worth having ready fast: each tree trains independently, so more trees only ever average away more noise, never add new overfitting risk. This is a very commonly asked "gotcha" question specifically because the intuitive-sounding wrong answer ("more trees = more complex model = more overfitting") is wrong here, unlike for boosting.

**"Why does Gradient Boosting need a learning rate but Random Forest doesn't?"**
Also covered in Ch.4/5's Q&A — boosting's rounds build on each other and can keep chasing noise round after round without a brake; Random Forest's trees are independent and averaged, with no such chasing mechanism to need braking.

**"If I have categorical features with hundreds of unique values, should I one-hot encode them for a tree model?"**
Usually not a good idea — one-hot encoding a 200-category feature creates 200 new sparse columns, and impurity-based splitting (Ch.1) tends to struggle finding good splits across many thin, sparse columns compared to working with the original categorical feature directly. Tools like LightGBM and CatBoost (Ch.5b) specifically handle categorical features natively without needing this — often the better choice when you have many high-cardinality categorical columns.

**"A tree-based model and a linear model get similar accuracy — which do you ship?"**
Depends on what matters beyond raw accuracy: if you need to explain individual predictions simply to non-technical stakeholders, or you strongly suspect the real relationship is close to linear and want it to extrapolate sensibly outside the training range, lean linear. If you expect messier real-world relationships going forward (new data with different patterns than what you've seen) or don't need simple explanations, the tree-based model is usually the safer, more flexible default.

---

## Quick, Simple Interview Answers

**Q: "Give one clear reason NOT to use a decision tree/forest for predicting something like stock prices."**
A: Trees can't extrapolate beyond the range of values they saw in training (8.1) — if stock prices trend to new all-time highs never seen in the training data, a tree-based model just flatlines at its highest-seen prediction instead of continuing the trend, which a linear (or trend-aware) model would at least attempt to do.

**Q: "How would you check if your Random Forest is overfitting?"**
A: Compare training vs. validation error. If Random Forest's validation error is well above training error even after trying reasonable `max_depth`/`min_samples_leaf` constraints, that's a real overfitting signal — but simply having *many trees* is not itself a red flag (8.2), unlike for boosting.

**Q: "Why might boosting take longer to train than Random Forest, even with the same number of trees?"**
A: Random Forest's trees can all be built simultaneously across CPU cores since none depend on each other; boosting's rounds must happen strictly one after another since each round needs the previous round's results first (8.3) — no amount of extra hardware parallelism can skip that dependency.

---

**Next up: Chapter 9 — Applied/Advanced (SHAP values, monotonic constraints, native missing-value handling). This is the last chapter in the syllabus. Want me to continue?**
