
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
Here are **advanced research-oriented interview questions** on **resampling methods** (especially based on ISLR Chapter 5) — designed to test deep conceptual understanding, edge-case handling, and real-world trade-offs. Each comes with a **strong answer**.

---

## 🔍 Resampling Methods – Research-Level Interview Questions

---

### 🔹 Q1. Why is LOOCV high variance despite using nearly the entire dataset for training?

**Answer:**
LOOCV trains on $n-1$ points and tests on 1, for each observation. Each model is almost identical, so the predictions are highly correlated. This leads to **high variance in the estimate of test error**, especially for unstable models like decision trees. Small changes in the train set can result in big changes in the fitted function on the one left-out point.

---

### 🔹 Q2. In what cases does the bootstrap give **biased** standard error estimates?

**Answer:**
Bootstrap tends to **underestimate variance** when:

* The sample size is **very small**.
* The **statistic is not smooth** (e.g., median, max).
* For highly skewed distributions.
  It assumes the sample is a good proxy for the population. If the original sample is unrepresentative, the bootstrap replicates the same bias.

---

### 🔹 Q3. Why is k-Fold CV (with $k = 5$ or $10$) often preferred over LOOCV or Validation Set?

**Answer:**

* **Less variance** than LOOCV due to averaging over more varied splits.
* **Less bias** than validation set (uses more data for training).
* **Efficient**: Trains only $k$ models instead of $n$.
* Offers a **sweet spot** in the bias–variance trade-off.

---

### 🔹 Q4. How does cross-validation fail in time series data?

**Answer:**
Standard CV assumes data points are i.i.d. In time series, observations are **autocorrelated**. Shuffling or random folds break the temporal structure. Instead, use:

* **Rolling forecast origin**
* **TimeSeriesSplit** (scikit-learn)
* **Blocked CV**

---

### 🔹 Q5. How would you estimate the **bias** of a model using resampling?

**Answer:**
Use **bootstrap**:

1. Generate $B$ bootstrap datasets.
2. Compute predictions $\hat{f}^*_b(x)$ for each.
3. Estimate bias as:

   $$
   \text{Bias}(x) = \mathbb{E}[\hat{f}^*(x)] - f(x)
   $$

If the true $f(x)$ is unknown, estimate bias on a held-out validation set or use simulations.

---

### 🔹 Q6. Can resampling help with **model selection**? How?

**Answer:**
Yes, especially:

* **k-Fold CV**: Used to choose hyperparameters (like λ in Ridge/Lasso).
* **Nested CV**: Prevents information leakage when tuning and evaluating.

Model with the **lowest average CV error** is selected.

---

### 🔹 Q7. Compare bootstrap and CV for **model performance estimation**.

| Criterion   | Bootstrap                      | Cross-Validation               |
| ----------- | ------------------------------ | ------------------------------ |
| Assumptions | i.i.d. samples                 | i.i.d. samples                 |
| Focus       | Accuracy of estimator (SE, CI) | Test error estimation          |
| Bias        | Sometimes high                 | Lower (k-Fold), higher (LOOCV) |
| Variance    | High (depends on B)            | Depends on k                   |
| Speed       | Slower (many resamples)        | Faster (esp. low k)            |

---

### 🔹 Q8. Why doesn’t bootstrap work well for estimating test error?

**Answer:**
Bootstrap samples are **not disjoint** — many points are repeated, and about **36.8%** of original points are left out in each sample. This breaks the idea of testing on **truly unseen data**. For estimating test error, **CV is more reliable**, especially **k-Fold**.

---

### 🔹 Q9. What’s the effect of choosing a very high value of $k$ in k-Fold CV?

**Answer:**
As $k \to n$, k-Fold becomes LOOCV:

* **Bias decreases** (almost all data used for training)
* **Variance increases** (models are highly correlated)
* **Computation increases**

Use moderate $k = 5$ or $10$ to balance.

---

### 🔹 Q10. How would you evaluate model **stability** using resampling?

**Answer:**
Use **bootstrap or repeated k-Fold CV**:

* Train multiple models on resampled sets.
* Measure variability in predictions or coefficients.
* If results vary a lot → **unstable model**.
  This helps in comparing models not just on accuracy, but also **robustness**.
