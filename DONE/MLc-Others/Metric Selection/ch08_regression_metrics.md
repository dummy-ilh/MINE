# Chapter 8: Regression Metrics

> *"In regression, there is no confusion matrix to hide behind. The model's mistakes are laid bare as numbers — and how you measure those numbers determines everything about what your model learns to care about."*

---

## 8.1 The Regression Problem

In classification, errors are categorical — right or wrong. In regression, errors are continuous — and they have magnitude, direction, and distribution.

This creates choices that classification doesn't:

- Do large errors matter more than small ones?
- Does the sign of the error matter (over-prediction vs. under-prediction)?
- Are outliers real signal or noise?
- Do you care about relative error or absolute error?

The metric you choose answers these questions — explicitly or implicitly. Choose wrong and your model will optimize for the wrong thing.

---

## 8.2 The Error Zoo

All regression metrics are functions of residuals:

```
Residual (error) at sample i:  eᵢ = yᵢ - ŷᵢ

yᵢ  = true value
ŷᵢ  = predicted value
eᵢ > 0  → model under-predicted
eᵢ < 0  → model over-predicted
```

The metrics differ in **how they aggregate these residuals** across samples.

---

## 8.3 Mean Absolute Error (MAE)

*Average magnitude of errors, regardless of direction.*

```
MAE = (1/n) × Σ |eᵢ|
    = (1/n) × Σ |yᵢ - ŷᵢ|
```

**Properties:**
- Units: same as the target variable (interpretable)
- All errors weighted equally — a $10 error and a $1000 error contribute proportionally
- Robust to outliers: a single huge error doesn't dominate
- Not differentiable at zero (subgradient needed for optimization)

**Interpretation:** "On average, our predictions are off by X units."

**When to use:**
- You want a metric that's easy to explain to non-technical stakeholders
- Outliers exist but should not dominate evaluation
- Over- and under-prediction are equally costly
- The error distribution has heavy tails

**Example:** Predicting delivery time in minutes. A model with MAE=8 minutes is off by 8 minutes on average. Easy to explain to a logistics team.

---

## 8.4 Mean Squared Error (MSE) and RMSE

*Average squared error — penalizes large mistakes disproportionately.*

```
MSE  = (1/n) × Σ eᵢ²
     = (1/n) × Σ (yᵢ - ŷᵢ)²

RMSE = √MSE
```

**Properties:**
- MSE units: target² (awkward); RMSE restores original units
- Differentiable everywhere — convenient for gradient-based optimization
- **Sensitive to outliers**: squaring amplifies large errors dramatically
- Penalizes variance in errors, not just mean

**Interpretation:** "The typical error magnitude is X units" (RMSE) — but this understates that a few large errors are driving the number.

### MAE vs. RMSE: The Key Difference

Suppose you have two models on 5 predictions:

```
True values:     [10, 20, 30, 40, 50]
Model A errors:  [ 3,  3,  3,  3,  3]   (consistent)
Model B errors:  [ 1,  1,  1,  1, 11]   (one big miss)

Model A:  MAE = 3.0,  RMSE = 3.0
Model B:  MAE = 3.0,  RMSE = 5.0
```

MAE says they're equal. RMSE correctly identifies Model B as worse — because that one big error matters more in most applications.

**Rule of thumb:**
```
RMSE >> MAE  →  outliers / high-variance errors are present
RMSE ≈ MAE  →  errors are roughly uniform
```

**When to use RMSE:**
- Large errors are genuinely worse than proportional (structural damage forecasting, financial risk)
- You want the metric to penalize inconsistency
- Downstream systems are sensitive to worst-case errors

**When to use MAE:**
- Outliers are measurement noise, not real signal
- Errors are roughly symmetric and you want a stable metric
- Interpretability is important

---

## 8.5 Mean Absolute Percentage Error (MAPE)

*Error as a percentage of the true value.*

```
MAPE = (100/n) × Σ |eᵢ / yᵢ|
     = (100/n) × Σ |(yᵢ - ŷᵢ) / yᵢ|
```

**Properties:**
- Scale-free: works across different target magnitudes
- Intuitive: "we're off by X% on average"
- **Asymmetric**: penalizes under-prediction more than over-prediction
- **Undefined when yᵢ = 0**: division by zero
- **Biased toward low-value predictions**: large true values contribute less to MAPE even with large absolute errors

### The Asymmetry Problem

Suppose true value = 100:

```
Over-prediction by 50:   ŷ = 150  →  |error/y| = 50/100 = 50%
Under-prediction by 50:  ŷ = 50   →  |error/y| = 50/100 = 50%  ← same
```

Seems symmetric? Now try true value = 100:

```
Over-prediction by 100:  ŷ = 200  →  |error/y| = 100/100 = 100%
Under-prediction by 100: ŷ = 0    →  |error/y| = 100/100 = 100%  ← same
```

But over-prediction has no ceiling (ŷ can be infinitely large), while under-prediction is capped at 100% (ŷ can't go below 0). This makes models trained with MAPE systematically **biased toward under-prediction**.

**When to use MAPE:**
- Target values span multiple orders of magnitude (sales forecasting: $100 to $10M)
- Percentage error is the natural unit (revenue, volume, growth rate)
- True values are reliably > 0

### Variants

**SMAPE (Symmetric MAPE):**
```
SMAPE = (200/n) × Σ |yᵢ - ŷᵢ| / (|yᵢ| + |ŷᵢ|)
```
Symmetrizes the denominator. Still has issues near zero and is not truly symmetric.

**WMAPE (Weighted MAPE):**
```
WMAPE = Σ |eᵢ| / Σ |yᵢ|
```
Divides total absolute error by total actual volume. Better for demand forecasting; avoids the zero-division problem by aggregation.

---

## 8.6 R² (Coefficient of Determination)

*How much better is the model than just predicting the mean?*

```
R² = 1 - (SS_res / SS_tot)

SS_res = Σ (yᵢ - ŷᵢ)²     ← residual sum of squares (model's errors)
SS_tot = Σ (yᵢ - ȳ)²      ← total sum of squares (mean baseline's errors)
```

**Interpretation:**
```
R² = 1.0   →  Perfect predictions
R² = 0.0   →  Model is no better than predicting the mean every time
R² < 0.0   →  Model is worse than predicting the mean (this happens)
```

**Example:**
- SS_tot = 1000 (variance around the mean)
- SS_res = 200 (model's remaining error)
- R² = 1 - 200/1000 = 0.80

The model explains 80% of the variance in the target.

### When R² Misleads

**Problem 1: Adding features always increases R²**

Even irrelevant features reduce SS_res slightly. R² will always improve or stay the same when you add a feature. Use **Adjusted R²** instead:

```
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]
```

Where n = samples, k = number of features. Penalizes unnecessary complexity.

**Problem 2: R² doesn't tell you if the model is right for the right reasons**

Anscombe's Quartet: four datasets with identical R², mean, and variance — but completely different shapes. Always plot your residuals.

**Problem 3: R² is not always between 0 and 1**

On a test set (not training set), R² can be negative if the model is worse than the mean. This is common with overfit models applied out-of-distribution.

---

## 8.7 Huber Loss

*A blend of MAE and MSE — robust to outliers but differentiable.*

```
Huber(eᵢ, δ) = {
    0.5 × eᵢ²          if |eᵢ| ≤ δ   ← MSE-like for small errors
    δ × (|eᵢ| - δ/2)   if |eᵢ| > δ   ← MAE-like for large errors
}
```

The hyperparameter δ defines the transition point:
- Errors smaller than δ are penalized quadratically (smooth, sensitive)
- Errors larger than δ are penalized linearly (robust to outliers)

**Behavior:**
```
δ → 0    →  Approaches MAE (fully robust)
δ → ∞    →  Approaches MSE (fully sensitive to outliers)
```

**When to use:**
- Outliers are present but you still want differentiability
- Robust regression in deep learning (object detection bounding box regression)
- Financial forecasting with occasional extreme events

**Practical note:** δ is a hyperparameter. Set it at the scale of typical errors on your validation set. If your typical error is ~10, try δ = 5 to 15.

---

## 8.8 Quantile Loss (Pinball Loss)

*Evaluates predictions at a specific quantile, not the mean.*

```
Quantile Loss (τ) = {
    τ × eᵢ           if eᵢ ≥ 0    (under-prediction)
    (τ - 1) × eᵢ     if eᵢ < 0    (over-prediction)
}
```

Where τ ∈ (0, 1) is the target quantile.

**Key insight:** Standard regression predicts the **conditional mean**. But sometimes you want the conditional **median** (τ=0.5) or a risk quantile (τ=0.9).

**When to use:**
- You need asymmetric cost of over vs. under-prediction
- You want prediction intervals (predict τ=0.1 and τ=0.9 simultaneously)
- Inventory planning: better to have too much stock (over-predict demand) than too little → use τ=0.9

**Example — Ride-hailing ETA:**
- Under-predicting wait time → customer cancels (bad)
- Over-predicting wait time → customer pleasantly surprised (okay)
- Use τ=0.8: model predicts the 80th percentile wait time

---

## 8.9 Choosing the Right Metric

```
Start here:
│
├─ Are outliers noise or real signal?
│       Noise   → MAE or Huber
│       Signal  → RMSE or MSE
│
├─ Does relative error matter more than absolute?
│       Yes     → MAPE or WMAPE
│       No      → MAE / RMSE
│
├─ Are there zeros in the target?
│       Yes     → Never use MAPE; use MAE, RMSE, or WMAPE
│       No      → MAPE is viable
│
├─ Do you need asymmetric cost?
│       Yes     → Quantile loss (set τ to reflect asymmetry)
│       No      → Symmetric metrics (MAE, RMSE)
│
├─ Do you want a normalized, scale-free measure?
│       Yes     → R², MAPE, WMAPE
│       No      → MAE, RMSE (same units as target)
│
└─ Are you training a deep learning model?
        Yes     → MSE (differentiable) or Huber (robust + differentiable)
        No      → Any of the above
```

---

## 8.10 Always Look at the Residual Distribution

Scalar metrics compress distributions into single numbers. Two models can have identical RMSE with completely different error patterns.

### Residual Plots to Always Make

**1. Residuals vs. Predicted Values**
```
eᵢ
 |  *  *
 |     *   *
 |  *     * *
─┼──────────────── ŷ
 |   * *
 |  *
```
Should look like random scatter around zero. Patterns indicate systematic bias.

**2. Residual Distribution (Histogram)**
Should be approximately normal and centered at zero. Heavy tails → RMSE will be dominated by outliers.

**3. Residuals vs. Time (for time series)**
Autocorrelation in residuals means the model is missing temporal patterns.

**4. Residuals by Segment**
Slice residuals by user segment, geography, feature bins. A model with MAE=5 overall might have MAE=20 for a specific user group that you care about.

### The Four Failure Patterns

```
Pattern 1: Heteroscedasticity
Residuals grow with predicted value → variance is not constant
→ Consider log-transforming the target

Pattern 2: Systematic Bias
Residuals are consistently positive or negative in a region
→ Model is missing a feature or has wrong functional form

Pattern 3: Outlier Dominance
A small number of residuals are enormous
→ Check for data errors; consider Huber loss or log transform

Pattern 4: Temporal Autocorrelation
Residuals at time t correlate with residuals at t-1
→ Add lag features; model temporal structure explicitly
```

---

## 8.11 Worked Example: House Price Prediction

Dataset: 10,000 houses. True prices range from $80,000 to $2,000,000.

```
Model A (Linear Regression):
  MAE  = $35,000
  RMSE = $72,000
  MAPE = 12.4%
  R²   = 0.82

Model B (Gradient Boosting):
  MAE  = $28,000
  RMSE = $61,000
  MAPE = 9.8%
  R²   = 0.88
```

Model B wins on all metrics. But let's look deeper:

```
Residual analysis:

Model A: Errors normally distributed, σ ≈ $72K
         Mild heteroscedasticity: errors larger for expensive homes

Model B: 95% of errors within ±$30K
         But 5% of predictions are off by > $300K (luxury homes)
         RMSE inflated by these outliers
```

**Decision:**
- For typical home listings: Model B is clearly better
- For luxury home market: Model A's more consistent errors may be preferred
- For portfolio risk management (worst-case matters): Model A's lower RMSE tail may matter

**Lesson:** No single scalar tells the whole story. Segment, slice, and look at the distribution.

---

## Summary

| Metric | Formula | Outlier Sensitivity | Best For |
|---|---|---|---|
| MAE | mean(\|e\|) | Low | Interpretable, robust baseline |
| MSE | mean(e²) | High | Differentiable training loss |
| RMSE | √MSE | High | Same units as target; penalizes large errors |
| MAPE | mean(\|e/y\|) | Medium | Scale-free; multi-scale targets |
| WMAPE | Σ\|e\|/Σ\|y\| | Low | Demand forecasting; avoids zero division |
| R² | 1 - SS_res/SS_tot | High | Normalized goodness of fit |
| Huber | Quadratic + linear blend | Low-Medium | Robust deep learning regression |
| Quantile | Asymmetric absolute error | Low | Asymmetric costs; prediction intervals |

---

## Further Reading

- Chai & Draxler — *Root Mean Square Error vs. Mean Absolute Error* (Geoscientific Model Development, 2014)
- Hyndman & Koehler — *Another Look at Measures of Forecast Accuracy* (International Journal of Forecasting, 2006) — definitive treatment of MAPE issues
- Huber, P. — *Robust Estimation of a Location Parameter* (Annals of Statistics, 1964) — original Huber loss
- Koenker & Bassett — *Regression Quantiles* (Econometrica, 1978) — quantile regression foundations

---

*Next: Chapter 9 — Imbalanced Classes*
