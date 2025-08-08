
# Resampling Methods 

---

## 5.1 Cross-Validation

###  The Validation Set Approach

- **Procedure**:
  1. Randomly split data into **training** and **validation** sets.
  2. Train model on training set.
  3. Evaluate MSE on validation set.

- **Validation MSE**:
  \\[
  \\text{MSE}_{val} = \\frac{1}{n_{val}} \\sum_{i \\in val} (y_i - \\hat{f}(x_i))^2
  \\]

- **Issues**:
  - High variance (sensitive to random split)
  - Underutilizes data (only part is used for training)

---

###  Leave-One-Out Cross-Validation (LOOCV)

- **Procedure**:
  - For each observation \\( i \\), train model on the remaining \\( n - 1 \\) points.
  - Test on the \\( i^{th} \\) point.

- **LOOCV Error Estimate**:
  \\[
  CV_{(n)} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{f}^{(-i)}(x_i))^2
  \\]

- **Pros**:
  - Very low bias (uses nearly full dataset)
- **Cons**:
  - High variance
  - Computationally expensive

---

###  k-Fold Cross-Validation

- **Procedure**:
  1. Split data into \\( k \\) folds.
  2. For each fold \\( j \\), train on \\( k-1 \\) folds and validate on the remaining.
  3. Average the validation MSEs.

- **k-Fold CV Estimate**:
  \\[
  CV_{(k)} = \\frac{1}{k} \\sum_{j=1}^{k} MSE_j
  \\]

- **Common values**: \\( k = 5 \\), \\( k = 10 \\)

---

###  Bias-Variance Trade-off for k-Fold CV

| Method     | Bias | Variance | Computation |
|------------|------|----------|-------------|
| LOOCV      | Low  | High     | High        |
| 5/10-Fold  | Higher | Lower | Moderate    |

- **Insight**: LOOCV has less bias but higher variance. k-fold is preferred for balanced performance and efficiency.

---

###  CV for Classification Problems

- Instead of MSE, use:
  - Classification Error Rate: \\( \\frac{1}{n} \\sum \\mathbb{I}(y_i \\neq \\hat{y}_i) \\)
  - Other metrics: Precision, Recall, F1-score, ROC AUC

- **Stratified k-Fold CV**: Keeps class distribution same across folds.

---

##  The Bootstrap

- **Purpose**: Estimate bias, variance, and SE of a statistic (e.g., coefficient, accuracy).

- **Procedure**:
  1. Resample data with replacement to get bootstrap samples.
  2. Compute statistic on each sample.
  3. Use distribution of statistics to estimate properties.

- **Bootstrap SE Estimate**:
  \\[
  SE_{boot} = \\sqrt{\\frac{1}{B - 1} \\sum_{b=1}^{B} (\\hat{\\theta}_b^* - \\bar{\\hat{\\theta}}^*)^2}
  \\]

- **Pros**:
  - Non-parametric
  - Useful for small data

- **Cons**:
  - Computational cost
  - Less effective on time series or highly dependent data

---

## 🔮 How to Predict using Cross-Validation

1. Use cross-validation to **select the best model** or **tune hyperparameters**.
2. Retrain the final model on the **entire training data** using best parameters.
3. Make predictions on **unseen test data**.

**Important**: Do **not** test on CV folds — always keep final test data separate.

---
