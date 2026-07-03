# Bias-Variance & the Bias-Variance Tradeoff — Interview Notes

## 1. Core Definitions

**Setup:** We're trying to learn a function $f(x)$ that generated data $y = f(x) + \epsilon$, where $\epsilon$ is irreducible noise with mean 0, variance $\sigma^2$. We fit a model $\hat{f}(x)$ from a training set.

- **Bias** — Error from wrong assumptions in the learning algorithm. $\text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x)$. It measures how far the *average prediction* (over many training sets) is from the true function. High bias → model is too simple → **underfitting**.

- **Variance** — Error from sensitivity to the specific training set. $\text{Var}[\hat{f}(x)] = E\left[(\hat{f}(x) - E[\hat{f}(x)])^2\right]$. It measures how much predictions swing if you retrain on a different sample. High variance → model is too complex/flexible → **overfitting**.

- **Irreducible error** — $\sigma^2$, noise inherent in the data-generating process. No model can remove this.

## 2. The Decomposition (for squared error loss)

For a point $x$, expected test MSE over random training sets $D$ and noise $\epsilon$:

$$E\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\text{Bias}[\hat{f}(x)]\right)^2}_{\text{bias}^2} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}$$

**Key derivation intuition** (good to be able to sketch on a whiteboard):
$$E[(y-\hat f)^2] = E[(f+\epsilon-\hat f)^2] = E[(f-E[\hat f])^2] + E[(E[\hat f]-\hat f)^2] + \sigma^2$$
using $E[\epsilon]=0$, independence of $\epsilon$, and adding/subtracting $E[\hat f(x)]$.

**Important caveat (common interview trap):** This clean additive decomposition holds exactly for **squared error / regression**. For **0-1 loss / classification**, there's no single clean additive decomposition — bias and variance interact multiplicatively in places (Domingos' unified decomposition), and variance can sometimes *help* near a decision boundary. Mention this if asked — shows depth.

## 3. Intuition

| | Low Variance | High Variance |
|---|---|---|
| **Low Bias** | Ideal — rare in practice | Overfit: complex model, memorizes noise |
| **High Bias** | Underfit: too simple, misses pattern | Worst of both — poorly specified *and* unstable (rare, e.g. tiny noisy model with bad hyperparams) |

Classic dartboard analogy: bias = systematically off-center throws; variance = scattered throws (even if centered on average).

**Model complexity axis:**
- Simple models (linear regression, shallow trees, low-degree polynomials, high-k KNN) → **high bias, low variance**.
- Complex models (deep trees, high-degree polynomials, low-k KNN, deep nets w/o regularization) → **low bias, high variance**.

Total error vs. complexity is the classic **U-shaped curve**: bias² decreases, variance increases, total test error has a minimum at the "sweet spot."

## 4. Pitfalls (interviewers love probing these)

1. **Equating bias-variance tradeoff with train/test error gap always.** A big train-test gap is *often* variance, but not proof of it — could be distribution shift, data leakage in training only, or a buggy eval pipeline.

2. **Assuming the tradeoff is a strict, unavoidable seesaw.** In classical statistical learning theory yes, but it's an *empirical tendency*, not a law. You can reduce both simultaneously with more/better data, better features, or better inductive bias (e.g., CNNs vs MLPs have both lower bias AND lower variance for images due to the right inductive bias).

3. **Double descent (modern deep learning gotcha).** Very over-parameterized models (way more params than data points) can have test error *decrease again* past the interpolation threshold — violating the classic U-curve. Worth mentioning at L5+ if the conversation goes into DL territory.

4. **Ignoring irreducible error.** Chasing zero error when $\sigma^2 > 0$ is a fool's errand — teams sometimes over-engineer models when the residual is just noise.

5. **Bias-variance is model+algorithm+data joint property, not just "model family."** Same model class (e.g. decision tree) can be high or low bias/variance depending on hyperparameters (depth, min_samples_leaf), so "which has more bias, tree or linear regression?" is underspecified without hyperparameters.

6. **Conflating with regularization strength naively.** More regularization → lower variance, higher bias — true in general, but the *rate* differs by problem; over-regularizing can spike bias faster than variance drops, net worse.

7. **Cross-validation variance itself.** People diagnose "high variance" from a single train/val split — but CV *scores themselves* have variance, especially with small data/small k. Misdiagnosis risk.

8. **More data always fixes variance — not quite.** More data reduces variance but does nothing for bias. If a linear model underfits, adding 10x data won't help; you need a more flexible model or better features.

9. **Feature engineering treated as separate from bias/variance.** Bad/missing features → high bias (model literally can't represent the relationship no matter how much data). This is often the actual root cause in production systems, not "model complexity."

## 5. Diagnosis

**Primary tool: Learning curves** (train error & val error vs. training set size)

- **High bias signature:** Train error and val error converge to a similarly **high** error value; adding more data doesn't close the gap much, and the gap between train/val is *small*.
- **High variance signature:** Large **gap** between train error (low) and val error (much higher); the gap tends to shrink (but not vanish) as training size increases.

**Other diagnostics:**
- **Train vs. validation error snapshot** (even without a full curve): train low + val high → variance; train high + val ≈ train → bias.
- **Complexity/hyperparameter sweep** (e.g., tree depth, polynomial degree, regularization λ, k in KNN): plot train/val error vs. complexity — classic U-curve for val error, monotonic decreasing for train error.
- **Bootstrap / repeated resampling:** retrain on many bootstrap samples, look at variance of predictions at fixed points — direct empirical estimate of the variance term.
- **Residual analysis:** structured/systematic residual patterns (e.g., curvature in residual plot) → bias (model misspecified). Random, unstructured but large residuals that change a lot across resamples → variance.
- **Ablation on data size:** if val error keeps dropping steeply with more data → variance-dominated regime; if it plateaus early → bias-dominated (need better model/features, not more rows).

## 6. Solutions

**If high bias (underfitting):**
- Increase model complexity (deeper trees, higher-degree features, more layers/units, less regularization)
- Add more/better features, feature crosses, domain-informed features
- Reduce regularization strength (λ)
- Use a more expressive model class (linear → GBM/NN)
- Boosting (sequentially reduces bias by fitting residuals)
- Decrease k in KNN

**If high variance (overfitting):**
- More training data (helps variance directly, not bias)
- Regularization (L1/L2, dropout, early stopping, pruning)
- Reduce model complexity (fewer features, shallower trees, lower-degree polynomial)
- **Bagging / ensembling** (Random Forest) — averages out variance across many high-variance, low-bias learners while roughly preserving bias
- Cross-validation for hyperparameter selection (prevents overfitting to a single split)
- Data augmentation
- Increase k in KNN
- Feature selection / dimensionality reduction (PCA) to reduce noise sensitivity

**Ensemble methods as a lens (common follow-up question):**
- **Bagging (Random Forest):** reduces **variance**, bias roughly unchanged — works because averaging many independent-ish, unbiased-but-noisy estimators cancels variance (variance of average of $n$ i.i.d. estimators $= \sigma^2/n$, and RF decorrelates trees via feature subsampling to push closer to that ideal).
- **Boosting (GBM/XGBoost/AdaBoost):** reduces **bias** by sequentially fitting weak learners to residuals/errors; can increase variance if run too long (too many rounds/too deep base learners) — controlled via learning rate, number of estimators, early stopping, shrinkage.

**General framing to give in an interview:** "Diagnose first via learning curves or a train/val gap, identify whether you're bias- or variance-dominated, then pick the lever (data, features, capacity, regularization, ensembling) that targets that specific term — don't just 'add more data' or 'regularize more' reflexively."

---

Happy to take questions — e.g., I can go deeper on double descent, the classification-loss decomposition, how this ties into hyperparameter tuning strategy, or how to present this on a whiteboard in an actual interview.
