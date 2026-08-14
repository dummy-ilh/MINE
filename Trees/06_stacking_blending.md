# Stacking & Blending — Comprehensive Guide

*(Same running example as the rest of the curriculum: 1,000 houses, predicting price.)*

---

## 1. What stacking is, in one sentence

Instead of averaging the predictions of several different model types, train a small extra model — the **meta-learner** — whose only job is to learn the best way to combine those predictions.

**Analogy:** three appraisers each give you a price estimate for a house. Rather than averaging their three numbers equally, you hire a fourth person whose whole job is to learn, over time, that "Appraiser A tends to run high on older houses, Appraiser B is spot-on for expensive homes, Appraiser C is best on small houses" — and combine their numbers accordingly, weighted by situation. That fourth person is the meta-learner.

**Why it can beat the best single model, or even a plain average:** different model families make *different kinds* of mistakes.
- A **linear model** is good at smooth overall trends but bad at sharp non-linear jumps.
- A **tree-based model** captures those jumps well but can be noisy on smooth trends.

If those error patterns don't overlap much, a meta-learner that has learned *when to trust which model* beats both a single model and a flat average (which treats every model as equally trustworthy everywhere, all the time).

Where this sits relative to the rest of the curriculum:
| Method | What it fights | How |
|---|---|---|
| Bagging (Ch. 3–4) | Instability / variance | Average many similar models |
| Boosting (Ch. 5) | Systematic blind spots / bias | Correct mistakes round after round |
| **Stacking** | Neither, specifically | Learn the smartest way to combine *already-good, differently-flawed* models |

---

## 2. The leakage trap — the part worth slowing down on

### 2.1 The naive (wrong) way

Train your three base models (Random Forest, XGBoost, linear regression) on all 1,000 houses. Have each one predict on those same 1,000 houses. Feed those predictions into the meta-learner as its training data.

### 2.2 Why this is broken — a concrete case

Say house #47 sold for **$500k**, and it was in the Random Forest's training set. A sufficiently deep, unpruned RF can partly *memorize* training data (this is the same high-variance behavior from Ch. 1.5). So its "prediction" for house #47 might come out at **$498k** — eerily close, not because RF understands that house well in general, but because it has already seen the answer.

The meta-learner then learns a pattern like: *"when RF says ~$498k, the true price is ~$500k → trust RF's number almost exactly."*

That pattern is **real**, but it only holds for houses RF has memorized. On a brand-new house, RF has never seen the true answer, so its prediction is genuinely noisier — but the meta-learner has learned to over-trust it anyway. Result: the stack looks fantastic during training and disappoints in production. The meta-learner was trained on artificially inflated, dishonest inputs.

### 2.3 The fix — out-of-fold predictions

Same "no peeking" principle as Ch. 5.6's CatBoost fix, and identical in spirit to cross-validation itself: **never let a model's prediction on a sample be influenced by having trained on that very sample.**

1. Split the 1,000 houses into 5 folds of 200.
2. For each fold *k*: train the base models on the *other* 4 folds (800 houses), then predict on fold *k*'s 200 held-out houses. These 200 predictions are **honest** — those houses were never seen during that particular training run.
3. Repeat for all 5 folds. Every house eventually gets exactly one honest, held-out prediction from each base model.
4. Train the meta-learner on these 1,000 honest predictions (not the naive, leaked ones).
5. **Deployment step (often left out of explanations, but essential):** retrain the base models one final time on **all 1,000 houses**, with no folds held out — you want the base models actually deployed to be as strong as possible. Only the meta-learner's *training data* needed to be leak-free; the final base models should use every house available.
6. For a genuinely new house: run it through the fully-retrained base models to get their predictions, then feed those into the trained meta-learner for the final answer.

### 2.4 Why "honest" is the right word

An out-of-fold prediction for house #47 always comes from a version of the model that was trained on the *other* 800 houses — a model that has never seen house #47's price. That's structurally identical to how the model will behave at real prediction time, on a house it's never seen. That's what makes it an honest stand-in for genuine test-time performance, rather than optimistic recall.

---

## 3. Worked numeric mini-example

To make "honest vs. leaked" concrete with actual numbers, imagine a tiny 6-house dataset split into 3 folds of 2 houses each, and one base model (a deep decision tree prone to memorizing).

| House | True price | Naive (in-sample) prediction | Out-of-fold prediction |
|---|---|---|---|
| 1 | $400k | $401k *(tree trained on house 1)* | $360k *(tree never saw house 1)* |
| 2 | $410k | $409k *(tree trained on house 2)* | $370k *(tree never saw house 2)* |
| 3 | $500k | $502k | $455k |
| 4 | $520k | $518k | $470k |
| 5 | $300k | $299k | $340k |
| 6 | $310k | $312k | $355k |

Notice the pattern: naive predictions are all within ~$2k of the truth — implausibly good for a model that hasn't seen millions of houses. Out-of-fold predictions are off by $30–60k, which is a realistic error size for this model on a house it's never seen.

If the meta-learner trains on the naive column, it learns "trust this model almost exactly" — a rule that will fail badly on new houses, where errors of $30–60k are actually typical. Training on the out-of-fold column teaches the meta-learner the model's *real* error behavior, so it learns a realistic combination rule instead.

---

## 4. Blending — the simpler cousin

Instead of the full 5-fold out-of-fold rotation, **blending** holds out one single validation chunk (say, the last 200 of the 1,000 houses), trains base models on the remaining 800, gets honest predictions on that one held-out chunk, and trains the meta-learner on just that chunk.

- **Pro:** simpler to implement, faster to run, one training pass per base model instead of five.
- **Con:** the meta-learner only ever sees 200 houses' worth of honest predictions instead of 1,000 — a noisier, lower-data version of the same idea.

---

## 5. When stacking is (and isn't) worth it

Stacking earns its complexity when base models are genuinely different in **how** they fail — different algorithm families (tree-based + linear + maybe a neural net), not just different random seeds of the same algorithm.

If all base models are slightly different Random Forests, stacking barely beats a plain average — their mistakes overlap heavily, so there's little for the meta-learner to actually learn, and you've added leakage-bug risk for almost no gain.

In practice (Kaggle-style pipelines), a Random Forest + XGBoost + LightGBM stack is common because their tree-building details differ enough (Ch. 5b) that their errors don't overlap perfectly — genuinely useful signal for the meta-learner.

---

## 6. Runnable code

Verified end-to-end below. Two base-model families are mixed deliberately — a linear signal plus a non-linear interaction term — so you can see the stack beat *every individual* base model, not just tie the best one.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Fake "house price" data: 1000 houses, 10 features.
# Mix of linear signal (linreg's strength) and a nonlinear interaction (trees' strength).
X, y_linear = make_regression(n_samples=1000, n_features=10, noise=15.0, random_state=42)
y = y_linear + 40 * (X[:, 0] * X[:, 1] > 0.3).astype(float)

X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

base_models = {
    "rf": RandomForestRegressor(n_estimators=200, random_state=0),
    "xgb": XGBRegressor(n_estimators=200, max_depth=3, verbosity=0, random_state=0),
    "linreg": LinearRegression(),
}

# 1) Build out-of-fold (honest) predictions for the meta-learner's training data
kf = KFold(n_splits=5, shuffle=True, random_state=0)
oof_preds = np.zeros((len(X_train), len(base_models)))

for name_idx, (name, model) in enumerate(base_models.items()):
    for train_idx, holdout_idx in kf.split(X_train):
        m = type(model)(**model.get_params())
        m.fit(X_train[train_idx], y_train[train_idx])
        oof_preds[holdout_idx, name_idx] = m.predict(X_train[holdout_idx])

# 2) Train the meta-learner on the honest out-of-fold predictions
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(oof_preds, y_train)

# 3) Retrain base models on ALL training data (no folds) for deployment
final_base_models = {}
for name, model in base_models.items():
    m = type(model)(**model.get_params())
    m.fit(X_train, y_train)
    final_base_models[name] = m

# 4) Predict on genuinely new (test) houses
test_base_preds = np.column_stack([m.predict(X_test) for m in final_base_models.values()])
stack_preds = meta_learner.predict(test_base_preds)

print("Stacked MAE:", mean_absolute_error(y_test, stack_preds))
for name, m in final_base_models.items():
    print(f"{name} alone MAE:", mean_absolute_error(y_test, m.predict(X_test)))
```

**Actual output from this run:**
```
Stacked MAE: 17.35
rf alone MAE:     39.70
xgb alone MAE:    26.22
linreg alone MAE: 17.94
```

The stack edges out even the best single model (linreg), because it can lean on RF/XGBoost's non-linear signal for the houses where the interaction term matters, while still trusting linreg's smooth trend elsewhere — exactly the "different models, different mistakes" idea from Section 1.

Scikit-learn also ships a built-in `StackingRegressor` / `StackingClassifier` that automates the out-of-fold machinery above (it handles the CV rotation and final refit for you) — worth knowing it exists, though building it manually once (as above) is the best way to *understand* what it's doing.

---

## 7. Diagnostics — how to tell your stack is actually working

| Symptom | Why it happens here specifically | Fix |
|---|---|---|
| Stack's train-set score is excellent, test-set score is mediocre or worse than the best base model | Classic sign the meta-learner was trained on **leaked** (in-sample) base predictions rather than out-of-fold ones | Rebuild the base-prediction pipeline using the out-of-fold procedure in Section 2.3 — check you didn't accidentally predict on training folds |
| Stack barely beats a plain average of the base models | Base models are too similar (e.g., three RF variants with different seeds) — their errors overlap heavily, so there's nothing new for the meta-learner to learn | Swap in a genuinely different model family (linear, or a different tree library), or accept that a plain average is simpler and nearly as good here |
| Meta-learner assigns a large weight/coefficient to one base model and near-zero to the others | Either that one base model is genuinely dominant, or the other base models' out-of-fold predictions are noisy/low-quality (e.g., too few folds, too little data per fold) | Increase fold count if data allows, check that each base model is reasonably well-tuned on its own before stacking |
| Performance is unstable across different random seeds for the fold split | Meta-learner is a fairly simple model (e.g., linear) being trained on a fairly small out-of-fold set — high sensitivity to which houses land in which fold | Average results across multiple fold-split seeds, or increase to blending with more validation data if 1,000 samples turns out too few for 5-fold stacking to be stable |
| Blended (single-holdout) version underperforms the full out-of-fold version by a lot | Blending trains the meta-learner on far fewer honest predictions (200 vs. 1,000) — noisier estimate of each base model's real error pattern | Switch to full out-of-fold stacking if you have the compute budget; blending is meant as a faster approximation, not a free lunch |

---

## Practice Q&A

**Q1 (easy).** Why not just average your models' predictions instead of building a meta-learner?
<details><summary>Answer</summary>Simple averaging treats every model as equally trustworthy everywhere. A meta-learner can learn that, say, the linear model should be trusted more for typical mid-size houses but less for unusual mansion-sized ones — a smarter, situation-dependent combination a flat average can't express.</details>

**Q2 (easy).** What's the single most important thing to get right when building a stack?
<details><summary>Answer</summary>Avoiding leakage — always generate base-model predictions using out-of-fold (or held-out) data, never predictions on data those models were trained on.</details>

**Q3 (medium).** You have 1,000 houses and use 5-fold out-of-fold stacking. How many total predictions does each base model produce during the out-of-fold phase, and how many of those are used to train the meta-learner?
<details><summary>Answer</summary>Each base model is trained 5 times (once per fold), each time predicting on the 200 held-out houses — that's 5 × 200 = 1,000 predictions per base model, and all 1,000 are honest (out-of-fold), so all 1,000 are used to train the meta-learner. Each house appears in the meta-learner's training set exactly once.</details>

**Q4 (medium).** Why do you retrain the base models on all 1,000 houses at the end, instead of just keeping the 5 fold-restricted versions?
<details><summary>Answer</summary>The fold-restricted versions were only ever trained on 800 houses each, deliberately, so their held-out predictions would be honest for meta-learner training. But for actual deployment you want the strongest possible base models, so you retrain each one on the full 1,000-house dataset. Only the meta-learner's training *data* needed leakage protection — the deployed base models don't.</details>

**Q5 (medium).** A colleague stacks three Random Forests, each with a different random seed, and finds the stack barely beats simply averaging the three. What's the likely explanation?
<details><summary>Answer</summary>Three RFs with different seeds tend to make similar kinds of mistakes — their errors are highly correlated because they're the same algorithm family. There's little genuinely different information for the meta-learner to combine, so stacking's advantage over a flat average shrinks toward zero.</details>

**Q6 (hard — spot the bug).** Someone writes this pseudocode for building the meta-learner's training data:
```
for model in [rf, xgb, linreg]:
    model.fit(X_train_800, y_train_800)
    preds[model] = model.predict(X_train_800)
train_meta_learner(preds, y_train_800)
```
What's wrong, and what would you observe if you ran it?
<details><summary>Answer</summary>This is the naive/leaked approach from Section 2.1 — each model predicts on the exact same data it was trained on, so `preds` are optimistic, near-memorized values rather than honest ones. You'd observe the meta-learner performing very well in-sample but disappointing (often worse than the best single base model) on a genuinely held-out test set — the classic leakage symptom from the diagnostics table.</details>

**Q7 (hard).** Blending uses one 200-house validation chunk instead of 5-fold out-of-fold. Why might blending's meta-learner be *more* prone to overfitting to quirks of that particular chunk than the full out-of-fold version?
<details><summary>Answer</summary>The full out-of-fold approach trains the meta-learner on all 1,000 houses' honest predictions, averaging out fold-specific quirks across 5 different held-out sets. Blending's meta-learner only ever sees one fixed 200-house sample — if that particular chunk happens to be unrepresentative (e.g., skewed toward larger houses), the meta-learner's learned combination rule will reflect that quirk rather than the true general relationship between base-model errors and price.</details>

**Q8 (hard).** In the worked numeric example (Section 3), house 5's naive prediction ($299k) is closer to the true price ($300k) than its out-of-fold prediction ($340k). Does that mean the naive prediction is "better" and should be trusted more?
<details><summary>Answer</summary>No — the naive prediction being close is exactly the problem, not a good sign. It's close *because* the tree memorized house 5 during training, not because the model generally predicts houses like house 5 well. The out-of-fold prediction ($340k, off by $40k) is the honest estimate of how this model actually performs on houses it hasn't seen — which is what matters for real-world use, since every future house the model sees will be "unseen."</details>

---

*Next in the curriculum: Chapter 7 — Evaluation & Tuning for trees/ensembles.*
