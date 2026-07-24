# Overfitting & Underfitting — Interview Notes

## 1. Core Definitions

- **Underfitting** — the model is too simple (or undertrained) to capture the true structure in the data. High error on **both** training and validation/test data.
- **Overfitting** — the model has learned the training data too well, including its noise and idiosyncrasies, and fails to generalize. Low training error, much higher validation/test error.
- Both are diagnosed relative to the **irreducible error** floor ($\sigma^2$) — no model can beat that, and chasing it is a fool's errand (see Bias-Variance notes, §4 pitfall 4).

**Direct link to bias-variance (don't present these as separate topics in an interview — they're the same phenomenon from a different angle):**
- Underfitting ⇔ **high bias**
- Overfitting ⇔ **high variance**

```
Error
  │
  │\                                          ╱
  │ \                                        ╱
  │  \                                      ╱
  │   \  Bias² (↓ as complexity ↑)         ╱  Variance (↑ as complexity ↑)
  │    \                                  ╱
  │     \                                ╱
  │      \______________________________╱
  │              Total Test Error (U-shape)
  │                    *  ← sweet spot
  └────────────────────────────────────────────► Model Complexity
    UNDERFIT ◄───────────────────────────► OVERFIT
    (high bias,                          (low bias,
     low variance)                        high variance)
```

---

## 2. Visualizing the Fit (the classic polynomial-degree picture)

```
UNDERFIT (degree 1, straight line through curved data):

  y │     ×
    │   ×   ×        ×
    │ ×___________×____×___   ← model (straight line) ignores the curve
    │×        ×
    └─────────────────────── x
  Training error: HIGH  |  Val error: HIGH  |  Gap: SMALL


GOOD FIT (degree ~3-4, follows the true trend):

  y │     ×  ╭╮
    │   ×  ╭╯  ╰╮      ×
    │ ×  ╭╯      ╰╮  ×
    │×  ╱          ╰╮×
    └─────────────────────── x
  Training error: LOW-ish  |  Val error: LOW-ish  |  Gap: SMALL


OVERFIT (degree 15, wiggles through every single point):

  y │     ×╮ ╭╮╭╮  ╭╮
    │   ×  ╰╯╰╯╰╯╲╭╯ ╲    ×
    │ ×  ╱╲    ╱ ╲╯    ╲ ╱
    │×  ╱  ╲  ╱          ╳  ← wild oscillation between points (Runge's phenomenon)
    └─────────────────────── x
  Training error: ~0  |  Val error: HIGH  |  Gap: LARGE
```

---

## 3. Symptoms Cheat Sheet

| Signature | Train Error | Val/Test Error | Gap | Diagnosis |
|---|---|---|---|---|
| Both high, roughly equal | High | High | Small | **Underfitting** (high bias) |
| Train low, val much higher | Low | High | Large | **Overfitting** (high variance) |
| Both low, roughly equal | Low | Low | Small | **Good fit** |
| Both low in offline eval, but prod metric drops | Low | Low (offline) | N/A | **Not over/underfitting** — likely data/concept drift or a leakage bug (see Cross-Validation notes §14) |

**Learning curve signatures (train/val error vs. training set size):**

```
UNDERFITTING:                          OVERFITTING:
Error                                  Error
  │  val error                           │  val error
  │ ─────────────────                    │╲
  │╲                                     │ ╲___________________
  │ ╲___________________                 │
  │  train error                         │       train error
  │ ─────────────────                    │ ───────────────────────
  │╲                                     │______________________
  │ ╲___________________                 │
  └──────────────────────► n             └──────────────────────► n

Both curves converge to a HIGH          Large, persistent GAP between
value; more data won't close            train (low) and val (high);
the gap much.                           gap shrinks slowly with more data
                                         but doesn't vanish.
```

---

## 4. Root Causes

**Underfitting causes:**
- Model class too simple relative to the true relationship (linear model on a nonlinear problem)
- Too much regularization (λ too high, dropout too aggressive)
- Insufficient training (too few epochs/iterations, learning rate too low, stopped too early)
- Missing/poor features — model literally cannot represent the true function no matter how much data (a very common *actual* root cause in production, more often than "model too simple")
- Too aggressive dimensionality reduction (over-compressed PCA, throwing away signal)

**Overfitting causes:**
- Model class too complex relative to available data (too many parameters vs. sample size)
- Too little training data, or data not representative of the true distribution
- Too little regularization, training for too many epochs without early stopping
- Noisy or low-quality labels the model ends up memorizing
- Too many irrelevant/noisy features (increases effective capacity without adding real signal — recall from Bias-Variance notes: even a useless feature increases variance)
- Data leakage that lets the model "cheat" on validation, masking true overfitting until production

---

## 5. Fixes

**If underfitting:**
- Increase model complexity (deeper trees/nets, higher-degree features, more layers/units)
- Reduce regularization strength
- Train longer / raise learning rate (if optimization hasn't converged)
- Add better features, feature crosses, domain knowledge
- Use a fundamentally more expressive model class
- Reduce excessive dimensionality reduction

**If overfitting:**
- More training data (directly attacks variance, does nothing for bias — won't help underfitting)
- Regularization (L1/L2, dropout, weight decay, pruning)
- Reduce model complexity (fewer features/params, shallower trees, lower-degree polynomial)
- Early stopping (halt training once val error starts rising even as train error keeps falling)
- Data augmentation (synthetically expands effective training set)
- Ensembling (bagging averages out variance across many high-variance learners)
- Cross-validation for hyperparameter selection instead of eyeballing a single split
- Feature selection / dimensionality reduction to remove noisy, non-informative features

```
Early stopping illustration:

Error
  │           val error
  │  ╲              ╭────────────  ← val error starts rising = overfitting begins
  │   ╲___________╱
  │                              train error keeps dropping
  │  ╲______________________________
  │        ↑
  │   STOP HERE (best val checkpoint)
  └──────────────────────────────► epochs
```

---

## 6. Common Pitfalls (interviewers love probing these)

1. **Assuming any train/val gap = overfitting.** Could be distribution shift between train/val, a leakage-in-training-only bug, or a buggy eval pipeline (see Cross-Validation notes for the leakage angle) — always rule out pipeline issues first.
2. **"More data always fixes overfitting" — true, but "more data always fixes underfitting" is false.** More data reduces variance but does nothing for bias; if the model is fundamentally too simple, 10x the data won't help — you need a more flexible model or better features.
3. **Regularizing reflexively without checking direction.** If the model is *underfitting*, adding *more* regularization makes it worse — diagnose first (learning curves), then pick the lever.
4. **Zero training error ≠ "good model," and non-zero training error ≠ "underfitting."** Zero train error is a red flag for overfitting risk (especially with high-capacity models), not a badge of honor. Conversely, some irreducible noise means you should *expect* nonzero train error even in a well-fit model.
5. **Confusing "the model overfit" with "the model overfit to a bug."** E.g., a leaked target-correlated feature can produce train AND val metrics that look great — that's not "no overfitting," that's the leakage masking overfitting or making the whole eval meaningless. Always sanity-check feature importances for suspiciously dominant leaky features.
6. **Over-regularizing can spike bias faster than it saves variance** — net worse total error even though "regularization reduces variance" is directionally true. This is why you sweep regularization strength via CV rather than picking one large "safe" value.
7. **Double descent (deep learning gotcha).** Extremely over-parameterized models can start overfitting, then — counter to classical intuition — test error *improves again* past the interpolation threshold, thanks to implicit regularization from the optimizer (e.g., SGD's minimum-norm bias). Worth mentioning at L5+ if the conversation goes into DL territory.
8. **Believing training loss curves alone tell the whole story.** A smoothly decreasing training loss says nothing about generalization by itself — you need the paired validation curve to diagnose over/underfitting at all.

---

## 7. FAANG-Level Interview Q&A

**Q1: Your model has 99% train accuracy and 65% val accuracy. Is this definitely overfitting?**
Likely, but not certain — first rule out: (a) train/val distribution mismatch (different time periods, sampling bias), (b) label noise differing between splits, (c) a leakage bug where a target-correlated feature is present in train but computed differently (or absent) at val/serving time, (d) an outright pipeline bug (e.g., val set accidentally harder or mislabeled). Only after ruling those out do you confidently call it classic overfitting and reach for regularization/more data/simpler model.

**Q2: Why doesn't "just add more data" fix underfitting, but it does help overfitting?**
More data reduces the *variance* term of the bias-variance decomposition — with more samples, any given model's fit becomes less sensitive to which particular training set it saw. But bias is a property of the model class's ability to represent the true function at all — more rows don't change what functions the hypothesis space can express. If a linear model can't represent a quadratic relationship, no amount of additional linear-model training data closes that structural gap; you need a different (more expressive) model or better features.

**Q3: You're told to "reduce overfitting" on a model that's already underfitting. What's wrong with that instruction, and how do you push back?**
The instruction conflates the two directions of the bias-variance tradeoff. If the model is underfitting (high bias — both train and val error high and similar), adding regularization or simplifying it further will make performance *worse*, not better, because you're moving further along the "high bias" end of the complexity axis. The correct diagnostic step is learning curves or a train/val gap check before picking a direction — show the gap is small and both errors are high, which is the underfitting signature, then argue for *more* capacity or better features instead.

**Q4: Model has zero training error on a dataset with known label noise. What does this imply?**
It implies the model has enough capacity to memorize noise, including mislabeled points — a strong overfitting red flag, not evidence of a "perfect" model. With known label noise, zero train error is essentially never desirable; the theoretically ideal model (matching the true underlying function) should still have *some* residual training error equal to the noise level, because it shouldn't be fitting noise. Zero error below that floor is a memorization signature.

**Q5: Explain why double descent seems to violate the classic overfitting narrative, and whether it actually does.**
Classical theory says test error should worsen monotonically past the point where a model can perfectly interpolate the training data (the "interpolation threshold") — this is the classic overfitting regime. Empirically, in heavily over-parameterized deep models (far more parameters than data points), test error can *decrease again* past that threshold. It doesn't violate the bias-variance decomposition itself (still mathematically true) — it violates the *naive intuition* that more capacity monotonically increases overfitting risk. The explanation usually invoked is implicit regularization from the optimizer (e.g., SGD converging to minimum-norm solutions among the many that fit the training data), which keeps effective variance in check even as raw parameter count explodes.

**Q6: A colleague says "training loss is going down every epoch, so the model is learning well." What's the gap in that reasoning?**
Training loss going down only shows the optimizer is fitting the *training set* better — it says nothing about generalization on its own. Without a paired validation curve, you can't tell whether you're still in the "good fit improving" regime or already past the point where val error has started rising while train loss keeps dropping (classic overfitting signature, see the early-stopping diagram). You need both curves, not just one, to make any claim about "learning well" in the generalization sense.

**Q7: You add 50 new engineered features and train accuracy goes up but val accuracy stays flat or drops slightly. What's your diagnosis and next step?**
Likely overfitting from added capacity — some (or all) of the new features are adding noise-sensitive degrees of freedom without real signal (recall: even a purely non-informative feature can increase variance by letting the model fit spurious training-set correlations). Next step: check feature importances/permutation importance for the new features — if several show near-zero or unstable importance across folds, remove them; consider regularization strength re-tuning since the added dimensionality likely needs more regularization to compensate; re-run with cross-validation (not a single split) to make sure the "flat val accuracy" isn't just single-split noise itself.

**Q8: Why can heavy regularization make BOTH training and validation error worse simultaneously — isn't regularization supposed to help generalization?**
Over-regularizing pushes the model past the sweet spot into the underfitting regime — bias increases faster than variance decreases, so total error (which is bias² + variance + irreducible noise) goes up on both train and val. Regularization strength is a *dial*, not a strictly "more is always better for generalization" lever; the correct approach is sweeping the regularization hyperparameter via cross-validation and picking the point that minimizes validation error, not just picking the largest "safe-looking" value.

---

## 8. One-Line Interview Closers

- *"Underfitting and overfitting aren't separate topics from bias-variance — they're just the qualitative names for the two ends of the same complexity axis."*
- *"Zero training error isn't a badge of honor — with any label noise or irreducible error, it's a memorization red flag, not a sign of a perfect model."*
- *"Before I call something overfitting, I rule out leakage and distribution shift first — a train/val gap has more than one possible cause."*
