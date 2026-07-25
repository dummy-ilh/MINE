# Cross-Validation — Interview Notes (All Types)

## 1. Why Cross-Validation Exists

A single train/test split gives you **one noisy estimate** of generalization error — that estimate itself has variance (see Bias-Variance notes, §4 pitfall 7). Cross-validation (CV) reduces this estimation variance by evaluating on multiple different splits and averaging, giving a more reliable estimate of how a model will perform on unseen data — and it lets you use nearly all your data for both training and validation instead of permanently sacrificing a chunk.

**Two things CV is used for (don't conflate):**
1. **Model/hyperparameter selection** — comparing configs, picking the best one
2. **Performance estimation** — reporting an unbiased-ish estimate of how the *final* model will generalize

Using the *same* CV loop for both without a held-out test set leaks information — this is a classic interview trap (see Q&A below).

---

## 2. Hold-Out Validation (the baseline, not really "cross" validation)

```
[==================== TRAIN (70%) ====================][=== TEST (30%) ===]
                     train on this                        validate on this
```

- Single split, train once, evaluate once.
- **Pros:** cheap, fast, simple.
- **Cons:** high variance in the error estimate — which 30% you happened to get matters a lot on small datasets. Wastes data (that 30% never contributes to training the final model).
- Use when: data is huge (millions of rows) and one split is already stable enough, or CV is too expensive (large deep learning models).

---

## 3. k-Fold Cross-Validation

Split data into $k$ equal-sized folds. Train on $k-1$ folds, validate on the remaining fold. Repeat $k$ times so every fold is the validation set exactly once. Average the $k$ scores.

```
Fold 1: [ VAL ][ TRAIN ][ TRAIN ][ TRAIN ][ TRAIN ]
Fold 2: [ TRAIN ][ VAL ][ TRAIN ][ TRAIN ][ TRAIN ]
Fold 3: [ TRAIN ][ TRAIN ][ VAL ][ TRAIN ][ TRAIN ]
Fold 4: [ TRAIN ][ TRAIN ][ TRAIN ][ VAL ][ TRAIN ]
Fold 5: [ TRAIN ][ TRAIN ][ TRAIN ][ TRAIN ][ VAL ]
         ---------------------------------------
                  Final score = mean(5 scores)
```

- Typical $k$: 5 or 10. Lower $k$ → less compute, more bias in the estimate (each training set is smaller than full data). Higher $k$ → more compute, lower bias, but higher variance across folds (training sets increasingly overlap, so fold scores become correlated).
- **Formula for the estimate:** $\widehat{\text{Err}}_{CV} = \frac{1}{k}\sum_{i=1}^{k} \text{Err}_i$
- **Bias-variance of the CV estimate itself:** $k=n$ (LOOCV) gives an almost unbiased estimate of test error but high variance (folds are nearly identical → highly correlated errors). Small $k$ (e.g. 2–3) gives a more biased estimate (pessimistic, since training set is much smaller than full data) but lower variance. $k=5$ or $10$ is the empirically-supported sweet spot (Kohavi, 1995).

---

## 4. Stratified k-Fold

Same as k-fold, but each fold preserves the overall class distribution — critical for imbalanced classification.

```
Full data class ratio:  ●●●●●●●●○○   (80% ● / 20% ○)

Naive k-fold fold 3 might land:      [●●●●●●●●●●]   ← 0% of minority class!
Stratified fold 3 guarantees:        [●●●●●●●●○○]   ← preserves 80/20 in every fold
```

- **Always prefer this over plain k-fold for classification**, especially with any class imbalance — plain k-fold can accidentally produce a fold with zero minority-class examples, making that fold's metric undefined or meaningless.
- Also relevant for regression when you have a skewed target (can stratify on binned target quantiles).

---

## 5. Leave-One-Out CV (LOOCV)

Special case of k-fold where $k = n$. Train on all but one point, validate on that single point, repeat for every point.

```
n = 6 points:  [1][2][3][4][5][6]

Iter 1: VAL=1, TRAIN={2,3,4,5,6}
Iter 2: VAL=2, TRAIN={1,3,4,5,6}
Iter 3: VAL=3, TRAIN={1,2,4,5,6}
...
Iter 6: VAL=6, TRAIN={1,2,3,4,5}
```

- **Pros:** uses maximum possible training data each iteration (n-1 points) → very low bias in the error estimate.
- **Cons:** computationally expensive ($n$ full model fits — infeasible for large $n$ or expensive models); the $n$ error estimates are highly correlated (training sets overlap in $n-2$ points), so the variance of the final averaged estimate can actually be *higher* than 10-fold CV, despite each individual fit being "more accurate." This is a classic counterintuitive interview point.
- Rarely used in practice except for very small datasets or when a closed-form shortcut exists (e.g., linear regression has an $O(n)$ LOOCV shortcut via the hat matrix — no need to refit $n$ times).

---

## 6. Leave-P-Out CV (LpOCV)

Generalization of LOOCV — leave out $p$ points per iteration instead of 1, test on all $\binom{n}{p}$ combinations.

- Combinatorially explosive ($\binom{n}{p}$ grows fast) — almost never used directly for $p>1$ in practice; mentioned mostly as the theoretical generalization that LOOCV is a special case of ($p=1$).

---

## 7. Repeated k-Fold CV

Run k-fold multiple times with different random shuffles/seeds, average across all repetitions.

```
Repetition 1:  [F1][F2][F3][F4][F5]  → shuffle A
Repetition 2:  [F1][F2][F3][F4][F5]  → shuffle B (different fold membership)
Repetition 3:  [F1][F2][F3][F4][F5]  → shuffle C
                        ↓
        Final score = mean across all reps × folds
```

- Reduces the variance introduced by *which specific* random split you happened to draw — directly addresses interview pitfall "CV scores themselves have variance."
- Cost: multiplies compute by the number of repeats. Common: 5-fold × 3 repeats = 15 fits.

---

## 8. Group k-Fold

Ensures all samples from the same **group** (user ID, patient ID, session ID) stay entirely in either train or validation — never split across both.

```
WRONG (plain k-fold) — user_42's records leak across splits:
TRAIN: [user_1, user_42, user_7]     VAL: [user_42, user_9]
                    ^^^^^^^^ same user in both train AND val → LEAKAGE

RIGHT (group k-fold) — grouped by user:
TRAIN: [user_1, user_7, user_9]      VAL: [user_42, user_55]
        entire users held out together
```

- **Critical for any data with repeated/clustered structure**: multiple rows per user, multiple images per patient, multiple transactions per account. Without grouping, the model can "memorize" a user's idiosyncrasies from train and get an inflated validation score — a very common real-world bug and a favorite FAANG interview scenario ("why is our offline AUC much higher than online AUC?").

---

## 9. Time Series Split (Walk-Forward / Rolling-Origin CV)

You cannot shuffle time-ordered data — training on the future to predict the past leaks information. Instead, the validation window always comes *after* the training window, and the split boundary rolls forward.

```
Expanding window:
Split 1: [TRAIN=====][VAL==]
Split 2: [TRAIN========][VAL==]
Split 3: [TRAIN===========][VAL==]
Split 4: [TRAIN==============][VAL==]
         →→→→→→→→→→→→→→→ time →→→→→→→→→→→→→→→

Sliding window (fixed train size):
Split 1: [TRAIN=====][VAL==]
Split 2:      [TRAIN=====][VAL==]
Split 3:           [TRAIN=====][VAL==]
```

- **Expanding window**: training set grows each split (more realistic for "we accumulate more history over time," but early splits have very little training data).
- **Sliding window**: training set size stays fixed, just shifts forward (better when older data becomes less relevant — concept drift).
- Never shuffle. Never use standard k-fold on time series — this is one of the most common "spot the bug" interview questions.
- Often paired with a **gap/embargo period** between train and val to prevent leakage from features that use rolling windows (e.g., a 7-day rolling average feature computed too close to the val period leaks future info backward).

---

## 10. Nested Cross-Validation

Used when you need to do **both** hyperparameter tuning **and** unbiased performance estimation without leaking one into the other.

```
Outer loop (5-fold) — for performance estimation:
┌─────────────────────────────────────────────────────┐
│ Outer Fold 1: [ OUTER-VAL ][ OUTER-TRAIN=========== ]│
│                                    │                  │
│                                    ▼                  │
│                Inner loop (5-fold) on OUTER-TRAIN     │
│                — for hyperparameter selection:        │
│                [i-VAL][i-TR][i-TR][i-TR][i-TR]        │
│                [i-TR][i-VAL][i-TR][i-TR][i-TR]        │
│                        ... picks best hyperparams     │
│                                    │                  │
│                                    ▼                  │
│         Refit best config on all of OUTER-TRAIN,      │
│         evaluate once on OUTER-VAL (never touched     │
│         during hyperparameter search)                 │
└─────────────────────────────────────────────────────┘
   Repeat for Outer Folds 2–5, average outer scores
```

- **Why it matters:** if you tune hyperparameters using the *same* CV folds you report performance on, your reported score is optimistically biased — you've implicitly "fit" to the validation folds via hyperparameter search. Nested CV keeps the outer loop's validation fold completely untouched by the tuning process.
- Expensive: $O(\text{outer } k \times \text{inner } k)$ total fits. Common FAANG interview follow-up: "how would you evaluate a model AND tune it without leakage, given limited data?" → nested CV is the textbook answer.

---

## 11. Monte Carlo CV / Shuffle-Split (Repeated Random Sub-sampling)

Randomly split into train/val some fixed proportion (e.g., 80/20), repeat many times with fresh random shuffles — unlike k-fold, splits are **not** guaranteed to partition the data (a point can appear in the validation set of multiple iterations, or none).

```
Iter 1: [ TRAIN (random 80%) ][ VAL (random 20%) ]
Iter 2: [ TRAIN (different random 80%) ][ VAL (different 20%) ]
Iter 3: [ TRAIN (different random 80%) ][ VAL (different 20%) ]
...
```

- More flexible than k-fold on train/val proportions and number of iterations (decoupled from each other, unlike k-fold where iterations = k is fixed by the fold size).
- Downside: some points may never be validated on, others validated on many times — less "complete coverage" guarantee than k-fold.

---

## 12. Bootstrap (.632 / .632+ estimators)

Resample $n$ points **with replacement** to form a bootstrap training set (same size $n$, but with duplicates); points never selected form the validation ("out-of-bag") set.

```
Original:      [1][2][3][4][5]
Bootstrap:     [2][2][5][1][5]   ← training set (with duplicates)
Out-of-bag:    [3][4]            ← validation set (never sampled)
```

- On average, **~63.2%** of points are selected at least once per bootstrap sample (probability a point is *never* picked in $n$ draws is $(1-1/n)^n \to e^{-1} \approx 0.368$, so $1-0.368 = 0.632$ are included).
- **.632 estimator:** $\text{Err}_{.632} = 0.368 \times \text{Err}_{\text{train}} + 0.632 \times \text{Err}_{\text{OOB}}$ — blends the (optimistic) training error with the (pessimistic, small-sample) out-of-bag error to reduce bias in either direction alone.
- Directly the mechanism behind **Random Forest's OOB error estimate** — free validation without a separate CV loop, since each tree already has its own out-of-bag set.

---

## 13. Summary Comparison Table

| Method | Data shuffled? | Handles imbalance? | Handles groups? | Handles time? | Compute cost | Bias of estimate | Variance of estimate |
|---|---|---|---|---|---|---|---|
| Hold-out | Yes | No | No | No | Lowest | High (pessimistic, small train) | Highest |
| k-Fold | Yes | No | No | No | Medium | Medium | Medium |
| Stratified k-Fold | Yes | **Yes** | No | No | Medium | Medium | Medium (lower than plain k-fold for imbalanced data) |
| LOOCV | N/A | No | No | No | Highest ($n$ fits) | Lowest | High (correlated folds) |
| Group k-Fold | Yes (by group) | No | **Yes** | No | Medium | Medium | Medium |
| Time Series Split | **No — never** | No | No | **Yes** | Medium | Depends on window | Depends on window |
| Nested CV | Yes | Can combine w/ stratified | Can combine w/ group | Can combine w/ time split | Highest (k×k) | Lowest (unbiased for both tuning + estimate) | Medium |
| Monte Carlo / Shuffle-Split | Yes | No | No | No | Flexible | Medium | Medium-High (uneven coverage) |
| Bootstrap (.632) | Yes (w/ replacement) | No | No | No | Medium | Low (blended) | Low-Medium |

---

## 14. Common Pitfalls (interviewers love probing these)

1. **Shuffling time series data.** Instant disqualifying bug in a live-coding round — always ask "is this data temporal?" before picking a CV strategy.
2. **Leaking groups across train/val** (same user/patient in both) — inflates validation metrics, silently fails in production. The single most common "why does prod underperform offline eval" root cause.
3. **Using the same CV folds for both hyperparameter tuning and final performance reporting** — optimistic bias; use nested CV or a separate untouched test set.
4. **Not stratifying on imbalanced classification** — a fold can end up with zero (or too few) minority-class examples, making precision/recall undefined or wildly noisy for that fold.
5. **Scaling/normalizing before splitting.** Fitting a `StandardScaler` (or any preprocessing that uses statistics of the data) on the *full* dataset before CV leaks validation-set information into training. Always fit preprocessing **inside** each fold's training data only.
6. **Feature engineering leakage** — e.g., target encoding a categorical variable using the full dataset's target means before splitting. Must be computed fold-internally (or via nested/out-of-fold encoding).
7. **Believing k=10 is always "correct."** It's an empirically-good default (Kohavi 1995), not a law — small data may want LOOCV or repeated k-fold; huge data may not need CV at all (hold-out is stable enough).
8. **Ignoring CV score variance itself when comparing models.** A 0.3% CV improvement might be well within the noise of the CV estimate — report std across folds, not just the mean, and consider a paired test across folds.
9. **Time series CV without an embargo/gap** — rolling-window features computed near the split boundary can leak future information backward even with a walk-forward split.

---

## 15. FAANG-Level Interview Q&A

**Q1: You have 10-fold CV showing model A beats model B by 0.4% accuracy. Do you ship model A?**
Not automatically. Compute the per-fold scores for both models and look at variance/spread — if the 0.4% gap is smaller than the standard deviation across folds, it may not be statistically meaningful. Use a paired t-test (or Wilcoxon signed-rank) across the matched folds rather than comparing single mean scores. This tests whether the candidate understands CV *estimates* have their own variance (a very common gotcha).

**Q2: Why can LOOCV have higher variance than 10-fold CV despite each individual model being trained on more data?**
Because the $n$ training sets in LOOCV overlap in $n-2$ points — they're nearly identical to each other, so the $n$ validation errors are highly *correlated*. Averaging correlated estimates doesn't reduce variance nearly as much as averaging independent ones (recall $\text{Var}(\bar X)$ shrinks less when samples are correlated — same math idea as why bagging needs decorrelated trees). 10-fold's training sets overlap less, so its fold errors are more independent, giving a lower-variance average despite each individual fold using less data.

**Q3: Your model has grouped structure (multiple rows per user). You ran plain stratified k-fold and got 92% AUC offline, but only 71% AUC in production. What's your hypothesis and how do you fix it?**
Prime suspect: group leakage — the same user's records likely appear in both train and validation folds, so the model partially memorized user-specific patterns (e.g., a user ID-correlated feature, or just overfitting to that user's idiosyncratic behavior) rather than learning generalizable signal. Fix: switch to **Group k-Fold**, keyed on user ID, and re-evaluate. If the AUC drops sharply toward the production number, that confirms the diagnosis.

**Q4: How would you cross-validate a model that both needs hyperparameter tuning AND needs an unbiased final performance number, on a dataset too small to carve out a separate untouched test set?**
Nested cross-validation: an outer loop provides the unbiased performance estimate (each outer validation fold is never touched during tuning), and an inner loop (run only on the outer training data) handles hyperparameter search. Refit the selected best config on the full outer-training data and score once on the untouched outer-validation fold. This is the standard answer to "no data to spare but need both tuning and honest eval."

**Q5: Time series forecasting model — offline backtest looks great, but a rolling 7-day-average feature is involved. What CV mistake could be inflating your offline number, and what's the fix?**
If the CV split doesn't include a gap/embargo between train and validation, a rolling feature computed near the split boundary can implicitly use information from timestamps that are technically "in the future" relative to some validation points within the window used to compute the rolling stat — a subtle leak. Fix: add an embargo period equal to (or larger than) the rolling window size between the end of train and start of val in each walk-forward split.

**Q6: True or false: stratified k-fold guarantees the same class ratio in every single fold, matching the full dataset exactly.**
Mostly true in practice for reasonably-sized folds, but not an exact mathematical guarantee at extreme edge cases — if a class has fewer members than $k$, stratification can't perfectly replicate the ratio in every fold (e.g., 3 minority samples with k=10 folds — at most 3 folds can contain one). Good answer shows awareness that stratification is "best effort proportional allocation," not a hard mathematical identity, at small sample sizes.

**Q7: When would you deliberately choose hold-out validation over k-fold, despite k-fold being "more rigorous"?**
When the dataset is large enough that a single split already gives a stable, low-variance estimate (e.g., hundreds of millions of rows), or when the model is too expensive to retrain $k$ times (large-scale deep learning, e.g., LLM fine-tuning runs) — the marginal reduction in estimate variance from k-fold isn't worth the k× compute cost. This shows judgment about when rigor isn't worth the price, not just reciting "always use k-fold."

**Q8: Why is fitting a StandardScaler on the full dataset before splitting into CV folds a leakage bug, and how big a deal is it really?**
The scaler's mean/std are computed using validation-fold data too, so the "unseen" fold isn't truly unseen — it has influenced the preprocessing statistics applied to the training data. It's usually a *small* leak for large datasets (global mean/std barely shifts by excluding one fold) but can meaningfully inflate metrics on small datasets or with features sensitive to outliers. Correct practice: wrap scaling inside a pipeline (e.g., sklearn `Pipeline`) so it's refit on each fold's training data only — shows the candidate knows the *mechanism*, not just the rule "don't do that."

**Q9: Explain the .632 bootstrap estimator and why it isn't just the OOB error alone.**
Out-of-bag (OOB) error alone tends to be **pessimistically biased** because the bootstrap training set only contains ~63.2% unique points (effectively a smaller, "hold-out-like" training set from the OOB sample's perspective), while the plain training error is **optimistically biased** (model evaluated on data it was fit on). The .632 estimator — $0.368 \cdot \text{Err}_{train} + 0.632 \cdot \text{Err}_{OOB}$ — blends both to correct each other's bias direction, landing closer to true generalization error than either alone.

**Q10 (clever): Can two models have identical mean CV accuracy across 10 folds but very different reliability in production? How would you tell?**
Yes — identical means can hide very different fold-to-fold variance. Model A might score consistently around 80% every fold (std ≈ 0.5%); Model B might swing between 65% and 95% depending on the fold (same mean, std ≈ 10%). Model B is far riskier in production because its performance is highly sensitive to the specific data distribution it happens to see. Always report (and compare) the standard deviation across folds, not just the mean — a classic "look past the headline number" interview signal.

**Q11: Your manager says "just use LOOCV, it's basically unbiased so it's strictly better than 10-fold." How do you respond?**
Push back precisely: LOOCV does reduce *bias* of the error estimate (each model trains on n-1 points, nearly the full dataset), but it can *increase variance* of that estimate due to high correlation between the n training sets — and it's far more expensive computationally. "Basically unbiased" isn't "strictly better" once you account for the bias-variance tradeoff *of the estimator itself* (a nice callback: the bias-variance tradeoff applies recursively to your evaluation methodology, not just your model). 10-fold is usually the better practical tradeoff.

**Q12: How does cross-validation interact with the bias-variance tradeoff of the model you're evaluating?**
Two separate but related things: (1) CV helps you *diagnose* the model's bias-variance regime (e.g., via a complexity sweep — plot CV error vs. hyperparameter, get the classic U-curve to find the sweet spot), and (2) CV's *own* estimate has its own bias-variance profile as an estimator (see Q11/Q2). It's a favorite meta-question because it tests whether the candidate can hold both frames — the model's tradeoff and the evaluation method's tradeoff — without conflating them.

---

## 16. One-Line Interview Closers (memorize-worthy)

- *"K-fold reduces the variance of your error estimate by averaging over multiple splits — but the CV estimate itself still has variance, which is why I'd report std across folds, not just the mean."*
- *"Group and time-series structure change what 'valid split' even means — the failure mode isn't underfitting or overfitting, it's leakage, and no amount of regularization fixes a leaky split."*
- *"Nested CV exists because tuning and evaluating on the same folds double-dips — the outer loop is the only honest number."*
