# Chapter 4: Model Adequacy Checking


> ⚠️ **Note on images:** The source PDF at `home.iitk.ac.in` blocked direct access (SSL/robots restriction), so embedded figures could not be extracted. Images can be viewed in the [original PDF](https://home.iitk.ac.in/~shalab/regression/Chapter4-Regression-ModelAdequacyChecking.pdf).

---

## 1. Is the Model Good Enough for Prediction?

Rejecting the null hypothesis `H₀: β₁ = 0` **and** having a non-significant lack-of-fit F-ratio does **not** guarantee the model will be a good predictor.

**Rule of thumb:** The F-ratio should be **at least 4–5×** the critical value `F(m, n−m−2)` for the regression model to be genuinely useful for prediction.

### Checking Prediction Quality

Compare the **range of fitted values** against their **average standard error**:

| Quantity | Formula |
|---|---|
| Range of fitted values | `ŷ_max − ŷ_min` |
| Average variance of fitted values | `(k/n) · σ²` |
| Estimated standard error | `√(k·σ̂²/n)` |

Where:
- `k` = number of parameters in the model
- `n` = number of observations
- `σ̂²` = model-independent estimate of error variance

> **Verdict:** If the range of `ŷᵢ` is **not large** relative to `√(kσ̂²/n)`, the model is unlikely to be a satisfactory predictor.

---

## 2. Estimating Pure Error — The Core Problem

### Standard approach (requires repeat observations)
Split the residual SS into:
```
SSₑᵣ = SS_PE + SS_LOF
```
- `SS_PE` (Pure Error) — uses repeat observations at the **same** x-levels; model-independent estimate of σ²
- `SS_LOF` (Lack of Fit) — detects systematic model misfit

**Problem in practice:** Exact repeat observations rarely occur in multiple regression, making this standard split useless.

---

## 3. Near-Neighbours Method (when no repeat points exist)

When exact repeats are unavailable, find observations taken at **near-identical** levels of `x₁, x₂, ..., xₖ` — these act as pseudo-repeat points.

### Step 1 — Measure distance between points

Use the **Weighted Sum of Squared Distances (WSSD)**:

```
D²ᵢᵢ' = Σⱼ [ (xᵢⱼ − xᵢ'ⱼ)² / MS_res ]
```

- Small `D²ᵢᵢ'` → points are near-neighbours (close in x-space)
- `D²ᵢᵢ' > 1` → points are widely separated; do **not** use as pseudo-repeats

### Step 2 — Estimate pure error from near-neighbours

For two near-neighbour points `i` and `i'`, compute the range of residuals:

```
Eᵢ = eᵢ − eᵢ'
```

For a sample of size 2 from a normal population, the standard deviation relates to range by:

```
σ ≈ E / 1.128    (equivalently: σ ≈ 0.886 · E)
```

---

## 4. Algorithm for the Near-Neighbours Estimate

1. **Sort** all data points by increasing fitted value `ŷᵢ`.

2. **Compute `D²ᵢᵢ'`** for all adjacent pairs in the sorted list, then repeat for pairs separated by 1, 2, and 3 intermediate `ŷ` values. This yields `~(4n − 10)` candidate pairs.

3. **Sort** the `(4n − 10)` values of `D²ᵢᵢ'` from smallest to largest. Let `Eᵤ` be the range of residuals for the `u`-th pair.

4. **Estimate pure error standard deviation** using the `m` smallest `D²` values:

```
σ̂ = (1/m) · Σᵤ₌₁ᵐ (0.886 · Eᵤ)
```

> **Choosing `m`:** Inspect the sorted `D²` values. Only include pairs where `D²ᵢᵢ'` is genuinely small — exclude any pair where the weighted distance is too large, as those are not true near-neighbours.

---

## Summary of Key Ideas

| Concept | Key Takeaway |
|---|---|
| F-ratio threshold for prediction | Must be 4–5× the critical value, not just significant |
| Prediction check | Range of `ŷ` must be large relative to `√(kσ̂²/n)` |
| Pure error (standard) | Needs exact repeat observations — rarely available in practice |
| Near-neighbours | Substitute for repeats using WSSD distance metric |
| WSSD cutoff | Exclude pairs with `D²ᵢᵢ' > 1` from pure error estimation |
