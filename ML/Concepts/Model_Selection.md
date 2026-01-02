## 📘 6.1 Subset Selection

Subset selection methods aim to identify a **subset of predictors** that provides the best trade-off between **model complexity** and **predictive performance**. These methods are central to linear model selection and are often contrasted with regularization techniques (Ridge, Lasso).

---

## 6.1.1 Best Subset Selection

### Idea

Fit **all possible regression models** formed by different subsets of the (p) predictors, and then choose the best model according to some criterion.

Total number of possible models:
[
2^p
]
(including the null model and the full model)

---

### Procedure (Algorithm 6.1)

1. **Null model** (M_0): contains no predictors.
   Prediction = sample mean of (Y).

2. For (k = 1, 2, \dots, p):

   * Fit **all (\binom{p}{k})** models with exactly (k) predictors.
   * Select the best model (M_k) among them:

     * smallest RSS
     * or equivalently, largest (R^2)

3. Choose the final model from (M_0, M_1, \dots, M_p) using:

   * Cross-validation error
   * (C_p)
   * AIC
   * BIC
   * Adjusted (R^2)

---

### Computational Complexity

* Number of models grows exponentially: (2^p)
* Example:

  * (p = 20 \Rightarrow 2^{20} > 1,000,000) models
  * (p > 40): computationally infeasible

Branch-and-bound algorithms can prune the search space, but:

* still expensive for large (p)
* only applicable to least squares regression

---

### Statistical Issues

* Large search space increases chance of **overfitting**
* High variance in coefficient estimates
* May find models that look good on training data but generalize poorly

---

### Pros & Cons

**Pros**

* Conceptually simple
* Finds the best model (under chosen criterion)

**Cons**

* Exponential time complexity
* High variance for large (p)
* Not scalable

---

## 6.1.2 Stepwise Selection

To address the computational and statistical limitations of best subset selection, **stepwise methods** search a **restricted model space**.

---

## Forward Stepwise Selection

### Idea

Start with no predictors and **add one predictor at a time**, choosing the variable that improves the model the most at each step.

---

### Procedure (Algorithm 6.2)

1. Start with null model (M_0)

2. For (k = 0, 1, \dots, p-1):

   * Consider all (p - k) models formed by adding **one new predictor** to (M_k)
   * Choose the best model (M_{k+1}):

     * smallest RSS
     * or largest (R^2)

3. Select the final model from (M_0, \dots, M_p) using CV, (C_p), AIC, BIC, or adjusted (R^2)

---

### Complexity

* Total models fit ≈ (1 + p(p+1)/2)
* Polynomial time → computationally feasible for large (p)

---

### Limitations

* Once a variable is added, **it cannot be removed**
* Greedy procedure → may miss the optimal subset

---

## Backward Stepwise Selection

### Idea

Start with the **full model** and remove predictors one at a time.

⚠️ Requires (n > p) so that the full model can be fit.

---

### Procedure (Algorithm 6.3)

1. Start with full model (M_p)

2. For (k = p, p-1, \dots, 1):

   * Consider all (k) models formed by removing **one predictor** from (M_k)
   * Choose the best model (M_{k-1})

3. Select the final model from (M_0, \dots, M_p) using CV, (C_p), AIC, BIC, or adjusted (R^2)

---

### Comparison: Forward vs Backward

| Aspect                | Forward    | Backward   |
| --------------------- | ---------- | ---------- |
| Start                 | Null model | Full model |
| Can remove variables? | ❌          | ✅          |
| Requires (n > p)?     | ❌          | ✅          |
| Greedy?               | Yes        | Yes        |

---

## Hybrid Stepwise Selection

Hybrid approaches combine forward and backward steps:

* Add predictors sequentially (like forward selection)
* After each addition, **reconsider and possibly remove** predictors

Goal:

* Approximate best subset selection
* Retain computational efficiency

Commonly used in practice (e.g., stepwise AIC in software)

---

## 6.1.3 Choosing the Optimal Model

Subset selection methods produce **multiple candidate models** with different numbers of predictors.

### Why Not RSS or (R^2)?

* RSS **always decreases** as predictors are added
* (R^2) **always increases** with more predictors
* Both measure **training error**, not test error

Thus, they cannot be used directly for model comparison.

---

## Estimating Test Error

Two general strategies:

### 1. Indirect Estimation (Penalty-Based)

Adjust training error to penalize model complexity:

* Mallows’ (C_p)
* AIC
* BIC
* Adjusted (R^2)

These methods add a penalty proportional to the number of predictors.

---

### 2. Direct Estimation

Estimate test error explicitly:

* Validation set approach
* Cross-validation

Preferred when sufficient data is available.

---

## Key Intuition

* **Best subset**: exhaustive but impractical
* **Stepwise methods**: greedy but scalable
* **Model selection** is about minimizing **test error**, not training error
* Bias–variance trade-off is central

---
## 📘 6.1 Subset Selection

Subset selection methods aim to identify a **subset of predictors** that provides the best trade-off between **model complexity** and **predictive performance**. These methods are central to linear model selection and are often contrasted with regularization techniques (Ridge, Lasso).

---

## 6.1.1 Best Subset Selection

### Idea

Fit **all possible regression models** formed by different subsets of the (p) predictors, and then choose the best model according to some criterion.

Total number of possible models:
[
2^p
]
(including the null model and the full model)

---

### Procedure (Algorithm 6.1)

1. **Null model** (M_0): contains no predictors.
   Prediction = sample mean of (Y).

2. For (k = 1, 2, \dots, p):

   * Fit **all (\binom{p}{k})** models with exactly (k) predictors.
   * Select the best model (M_k) among them:

     * smallest RSS
     * or equivalently, largest (R^2)

3. Choose the final model from (M_0, M_1, \dots, M_p) using:

   * Cross-validation error
   * (C_p)
   * AIC
   * BIC
   * Adjusted (R^2)

---

### Computational Complexity

* Number of models grows exponentially: (2^p)
* Example:

  * (p = 20 \Rightarrow 2^{20} > 1,000,000) models
  * (p > 40): computationally infeasible

Branch-and-bound algorithms can prune the search space, but:

* still expensive for large (p)
* only applicable to least squares regression

---

### Statistical Issues

* Large search space increases chance of **overfitting**
* High variance in coefficient estimates
* May find models that look good on training data but generalize poorly

---

### Pros & Cons

**Pros**

* Conceptually simple
* Finds the best model (under chosen criterion)

**Cons**

* Exponential time complexity
* High variance for large (p)
* Not scalable

---

## 6.1.2 Stepwise Selection

To address the computational and statistical limitations of best subset selection, **stepwise methods** search a **restricted model space**.

---

## Forward Stepwise Selection

### Idea

Start with no predictors and **add one predictor at a time**, choosing the variable that improves the model the most at each step.

---

### Procedure (Algorithm 6.2)

1. Start with null model (M_0)

2. For (k = 0, 1, \dots, p-1):

   * Consider all (p - k) models formed by adding **one new predictor** to (M_k)
   * Choose the best model (M_{k+1}):

     * smallest RSS
     * or largest (R^2)

3. Select the final model from (M_0, \dots, M_p) using CV, (C_p), AIC, BIC, or adjusted (R^2)

---

### Complexity

* Total models fit ≈ (1 + p(p+1)/2)
* Polynomial time → computationally feasible for large (p)

---

### Limitations

* Once a variable is added, **it cannot be removed**
* Greedy procedure → may miss the optimal subset

---

## Backward Stepwise Selection

### Idea

Start with the **full model** and remove predictors one at a time.

⚠️ Requires (n > p) so that the full model can be fit.

---

### Procedure (Algorithm 6.3)

1. Start with full model (M_p)

2. For (k = p, p-1, \dots, 1):

   * Consider all (k) models formed by removing **one predictor** from (M_k)
   * Choose the best model (M_{k-1})

3. Select the final model from (M_0, \dots, M_p) using CV, (C_p), AIC, BIC, or adjusted (R^2)

---

### Comparison: Forward vs Backward

| Aspect                | Forward    | Backward   |
| --------------------- | ---------- | ---------- |
| Start                 | Null model | Full model |
| Can remove variables? | ❌          | ✅          |
| Requires (n > p)?     | ❌          | ✅          |
| Greedy?               | Yes        | Yes        |

---

## Hybrid Stepwise Selection

Hybrid approaches combine forward and backward steps:

* Add predictors sequentially (like forward selection)
* After each addition, **reconsider and possibly remove** predictors

Goal:

* Approximate best subset selection
* Retain computational efficiency

Commonly used in practice (e.g., stepwise AIC in software)

---

## 6.1.3 Choosing the Optimal Model

Subset selection methods produce **multiple candidate models** with different numbers of predictors.

### Why Not RSS or (R^2)?

* RSS **always decreases** as predictors are added
* (R^2) **always increases** with more predictors
* Both measure **training error**, not test error

Thus, they cannot be used directly for model comparison.

---

## Estimating Test Error

Two general strategies:

### 1. Indirect Estimation (Penalty-Based)

Adjust training error to penalize model complexity:

* Mallows’ (C_p)
* AIC
* BIC
* Adjusted (R^2)

These methods add a penalty proportional to the number of predictors.

---

### 2. Direct Estimation

Estimate test error explicitly:

* Validation set approach
* Cross-validation

Preferred when sufficient data is available.

---

## Key Intuition

* **Best subset**: exhaustive but impractical
* **Stepwise methods**: greedy but scalable
* **Model selection** is about minimizing **test error**, not training error
* Bias–variance trade-off is central

---

## Exam & Interview Takeaways

* Best subset = exponential complexity (2^p)
* Forward/backward = greedy search
* Stepwise methods reduce variance at the cost of bias
* Penalized criteria approximate test error
* Cross-validation is the gold standard when feasible

---

> Subset selection methods search explicitly over models, while regularization methods shrink coefficients continuously — two different philosophies for controlling complexity.

---

## Validation Set and Cross-Validation for Model Selection

### Motivation

As an alternative to penalty-based criteria (AIC, BIC, Mallows’ Cp, Adjusted R²), we can **directly estimate the test error** using:

* a **validation set**, or
* **cross-validation (CV)**

The key idea is simple: *choose the model that performs best on data not used for fitting*.

---

## Validation Set Approach

### Procedure

1. Randomly split the data into:

   * Training set
   * Validation set
2. Fit each candidate model on the training set
3. Compute the validation MSE for each model
4. Select the model with the **smallest validation error**

---

### Pros

* Direct estimate of test error
* Very easy to implement
* Makes minimal assumptions about the true model

---

### Cons

* Validation error can be **highly variable** (depends on random split)
* Uses less data for training → higher bias in model fitting
* Inefficient when data size is small

---

## Cross-Validation (CV)

### Idea

Cross-validation improves upon the validation set approach by **repeatedly fitting models on different subsets of the data** and averaging the results.

Most common choice:

* **k-fold cross-validation** (e.g. k = 5 or 10)

---

### k-Fold CV Procedure

1. Split data into k roughly equal folds
2. For each fold:

   * Fit the model on the other k−1 folds
   * Compute prediction error on the held-out fold
3. Average the errors across all folds

This gives an estimate of the **test error**.

---

### Why CV is Attractive for Model Selection

Compared to AIC, BIC, Cp, and Adjusted R²:

* Provides a **direct estimate** of test error
* Requires **fewer assumptions** about:

  * linearity
  * error distribution
  * degrees of freedom
* Works even when:

  * model complexity is hard to quantify
  * σ² is unknown or difficult to estimate
* Applicable to a **wide range of models**, not just linear regression

---

## Computational Considerations

Historically:

* CV was computationally expensive for large n or large p
* Penalized criteria were preferred

Today:

* Modern computing makes CV **practically feasible** in most settings
* CV is now one of the **most widely used methods** for model selection

---

## Comparing CV with Information Criteria

| Criterion        | Nature   | Assumptions              | Output                   |
| ---------------- | -------- | ------------------------ | ------------------------ |
| AIC / Cp         | Indirect | Correct model form, σ²   | Penalized training error |
| BIC              | Indirect | True model in candidates | Strong penalty           |
| Adjusted R²      | Indirect | Linear model             | Heuristic                |
| Cross-validation | Direct   | Minimal                  | Estimated test error     |

---

## Interpretation of Error Curves (Figure 6.3 intuition)

When plotting error vs model size d:

* Training error: decreases monotonically
* Validation / CV error:

  * initially decreases (bias reduction)
  * then increases (variance dominates)

The optimal model corresponds to the **minimum of the validation or CV error curve**.

---

## Key Takeaways

* Validation and CV focus on **generalization performance**
* CV is preferred when data allows
* Penalized criteria are approximations to test error
* Model selection is fundamentally about the **bias–variance trade-off**

> In modern practice, cross-validation is often the default choice for selecting among competing models.
