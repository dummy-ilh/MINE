```md
# 📘 ISLR — Multiple Linear Regression: Key Concepts & Notes

This section covers **multiple linear regression (MLR)**, including hypothesis testing, variable selection, model fit, and prediction.

---

## 1️⃣ Important Questions in Multiple Regression

When performing MLR, we usually ask:

1. Is at least one predictor $X_1, X_2, ..., X_p$ useful in predicting $Y$?  
2. Are all predictors useful, or only a subset?  
3. How well does the model fit the data?  
4. How to predict $Y$ given a set of predictors and quantify uncertainty?

---

## 2️⃣ Relationship Between Response and Predictors

- In **simple regression**, check if $\beta_1 = 0$.  
- In **multiple regression**, test if all coefficients are zero:

$$
H_0: \beta_1 = \beta_2 = ... = \beta_p = 0 \\
H_a: \text{At least one } \beta_j \neq 0
$$

### 🔹 F-statistic

$$
F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)}
$$

- $TSS = \sum_i (y_i - \bar{y})^2$, $RSS = \sum_i (y_i - \hat{y}_i)^2$  
- $F \approx 1$ → no relationship  
- $F \gg 1$ → evidence at least one predictor matters  

> Example (Advertising data): $F = 570$ → very strong evidence that at least one medium affects sales.

---

### 🔹 Testing a Subset of Predictors

- For $q$ predictors:

$$
F = \frac{RSS_0 - RSS}{q} \Big/ \frac{RSS}{n-p-1}
$$

- Equivalent to **t-test** for a single variable ($q=1$)  
- Individual p-values can indicate partial effects (e.g., TV and radio significant; newspaper not)

---

## 3️⃣ Variable Selection

- **Goal:** Identify predictors that truly affect the response.
- **Problem:** $2^p$ possible subsets → exhaustive search infeasible for moderate $p$.
- **Classical approaches:**
  1. **Forward selection:** Start with null model, add variables one at a time minimizing RSS.
  2. **Backward selection:** Start with full model, remove least significant variables iteratively.
  3. **Mixed selection:** Combines forward and backward; adds variables but removes if p-value rises above threshold.

> Backward selection cannot be used if $p > n$. Forward selection always works but is greedy.

---

## 4️⃣ Model Fit

- **Key metrics:**
  1. **Residual Standard Error (RSE):** Measure of average deviation of observed $Y$ from predicted $\hat{Y}$  
     
     $$
     RSE = \sqrt{\frac{RSS}{n-p-1}}
     $$

  2. **$R^2$:** Fraction of variance explained  

     $$
     R^2 = \text{Cor}(Y, \hat{Y})^2
     $$

- **Properties:**
  - $R^2$ always increases when adding variables.
  - Small increase in $R^2$ → added variable may be unnecessary (risk of overfitting).
  
> Example (Advertising data):  
> - TV only: $R^2 = 0.61$, RSE = 3.26  
> - TV + Radio: $R^2 = 0.897$, RSE = 1.681  
> - TV + Radio + Newspaper: $R^2 \approx 0.8972$, RSE = 1.686 → newspaper adds little value

---

### 🔹 Residual Plots

- Useful for detecting non-linearity, interaction effects, or patterns not captured by the model.
- Advertising example: Residuals show a **synergy effect** (TV + Radio together increase sales more than individually).

---

## 5️⃣ Predictions

- Multiple sources of uncertainty:
  1. **Coefficient estimates ($\hat{\beta}_0, ..., \hat{\beta}_p$)** → reducible error  
     - Use **confidence intervals** to quantify uncertainty in $\hat{f}(X)$
  2. **Model bias** → linear model may approximate true $f(X)$
  3. **Random error ($\varepsilon$)** → irreducible error  

- **Confidence interval:** Predicts mean response for given $X$  

$$
\hat{Y} \pm 2 \cdot SE(\hat{Y})
$$

- **Prediction interval:** Predicts individual response for given $X$ (wider than CI because it includes irreducible error)  

> Example (Advertising data, $X = [\text{TV}=100, \text{Radio}=20]$):
> - 95% CI for mean sales: [10,985, 11,528]  
> - 95% Prediction interval for a single city: [7,930, 14,580]  

---

## ✅ Summary

| Concept | Purpose | Notes |
|---------|---------|-------|
| F-statistic | Test if at least one predictor matters | Adjusts for multiple predictors |
| Variable selection | Identify meaningful predictors | Forward, backward, mixed selection |
| RSE & $R^2$ | Measure model fit | $R^2$ rises with added variables; RSE may increase if added variables add little |
| Residual plots | Detect non-linearity & interaction | Look for systematic patterns |
| Confidence interval | Predict mean response | Narrower than prediction interval |
| Prediction interval | Predict individual response | Wider; includes irreducible error |

> Core workflow: **Fit model → assess significance (F & t-tests) → select variables → evaluate fit (RSE, $R^2$, residuals) → make predictions with uncertainty**
```
````md
# 📘 ISLR — Other Considerations in Regression & Marketing Plan Notes

---

## 1️⃣ Qualitative Predictors

### a) Predictors with Two Levels
- Example: Gender
  - Encode as:  
    ```
    xi = 1 if female
    xi = -1 if male
    ```
- Regression model:
  $$
  y_i = \beta_0 + \beta_1 x_i + \epsilon_i
  $$
  - $\beta_0 + \beta_1 + \epsilon_i$ if female  
  - $\beta_0 - \beta_1 + \epsilon_i$ if male  

### b) Predictors with More than Two Levels
- Example: Race (Asian, Caucasian, African American)
  - Dummy variables:
    ```
    xi1 = 1 if Asian, 0 otherwise
    xi2 = 1 if Caucasian, 0 otherwise
    ```
- Regression model:
  $$
  y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \epsilon_i
  $$
  - Asian: $\beta_0 + \beta_1 + \epsilon_i$  
  - Caucasian: $\beta_0 + \beta_2 + \epsilon_i$  
  - African American: $\beta_0 + \epsilon_i$  

---

## 2️⃣ Removing the Additive Assumption
- Linear regression assumes additive effects:
  $$
  y = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p + \epsilon
  $$
- Non-linear relationships can be accommodated using:
  - Transformations of predictors
  - Interaction terms

---

## 3️⃣ Potential Problems in Linear Regression
1. Non-linearity of response-predictor relationships  
2. Correlated errors  
3. Non-constant variance (heteroscedasticity)  
4. Outliers  
5. High-leverage points  
6. Collinearity (strong correlation among predictors)

---

## 4️⃣ The Marketing Plan: Seven Key Questions

1. **Is there a relationship between advertising and sales?**  
   - Fit MLR: $Sales \sim TV + Radio + Newspaper$  
   - F-statistic and p-value indicate strong evidence for a relationship.

2. **How strong is the relationship?**  
   - RSE = 1,681 units (~12% error)  
   - $R^2 \approx 0.90$ → 90% variance explained

3. **Which media contribute to sales?**  
   - Examine p-values: TV and radio significant; newspaper not significant

4. **Effect size of each medium**  
   - 95% CI for $\beta$:
     - TV: (0.043, 0.049)  
     - Radio: (0.172, 0.206)  
     - Newspaper: (-0.013, 0.011) → not significant  
   - VIF scores: no collinearity detected (TV=1.005, Radio=1.145, Newspaper=1.145)

5. **Prediction accuracy**  
   - Use **confidence intervals** for mean response  
   - Use **prediction intervals** for individual response (wider)

6. **Is the relationship linear?**  
   - Residual plots reveal non-linear patterns  
   - Can include transformations to model non-linearity

7. **Synergy among media?**  
   - Interaction terms capture non-additive effects  
   - Example: TV + Radio interaction increases $R^2$ from 90% → 97%

---

## 5️⃣ Parametric vs Non-Parametric Regression

| Aspect | Linear Regression (Parametric) | KNN Regression (Non-Parametric) |
|--------|-------------------------------|--------------------------------|
| Assumption | Linear form of $f(X)$ | No explicit functional form |
| Coefficients | Easy to interpret | No coefficients; prediction based on neighbors |
| Flexibility | Less flexible; poor if linearity assumption fails | Flexible; adapts to data shape |
| Ease of fitting | Simple, efficient | Requires distance calculations & neighbor selection |

> Linear regression is easy to interpret and statistically test, but strong assumptions can hurt accuracy if true $f(X)$ is non-linear. Non-parametric methods like KNN offer flexibility but less interpretability.
````
```md
# 📘 ISLR — Linear Regression vs K-Nearest Neighbors (KNN) Notes

---

## 1️⃣ Parametric vs Non-Parametric

### Linear Regression (Parametric)
- Assumes a linear functional form for \(f(X)\)
- Advantages:
  - Simple to fit (few coefficients)
  - Coefficients are interpretable
  - Statistical tests (t-tests, F-tests) are easy
- Disadvantages:
  - Strong assumptions about the functional form
  - Poor performance if true relationship is non-linear

### K-Nearest Neighbors Regression (KNN, Non-Parametric)
- No explicit form for \(f(X)\)
- Predict \(f(x_0)\) as average of responses of K nearest neighbors:
  $$
  \hat{f}(x_0) = \frac{1}{K} \sum_{i \in N_0} y_i
  $$
- K small → low bias, high variance (rough fit)  
- K large → higher bias, lower variance (smooth fit)
- Flexible for non-linear relationships

---

## 2️⃣ Bias-Variance Tradeoff in KNN
- K = 1: perfect interpolation → step-function, high variance  
- K > 1: smoother fit → reduces variance, may increase bias  
- Optimal K depends on minimizing test MSE

---

## 3️⃣ When Linear Regression Outperforms KNN
- If the true relationship is linear:
  - Linear regression has almost zero bias, low variance → very accurate  
  - KNN suffers from extra variance, especially for small K  
- Test MSE: Linear regression < KNN (K small)  
- Larger K in KNN reduces variance but may mask structure (bias increases)

---

## 4️⃣ When KNN Outperforms Linear Regression
- When the true relationship is non-linear:
  - KNN can adapt to the shape of \(f(X)\)
  - Linear regression incurs high bias → worse MSE
- Non-linear example:  
  - Slight non-linearity: KNN better for K ≥ 4  
  - Strong non-linearity: KNN outperforms linear regression for all K

---

## 5️⃣ Effect of Dimensionality
- Increasing number of predictors (p) affects KNN more than linear regression:
  - Curse of dimensionality: in high dimensions, neighbors are far away → poor KNN predictions  
  - Example: 100 observations  
    - p = 1 → KNN accurate  
    - p = 20 → KNN MSE increases >10x, linear regression stable
- Rule: Parametric methods (linear regression) tend to outperform non-parametric methods (KNN) when:
  - Sample size per predictor is small
  - High-dimensional data (p large)
  - Interpretability is important

---

## 6️⃣ Summary Guidelines
| Scenario | Preferred Method | Reason |
|----------|-----------------|--------|
| True relationship linear, p small | Linear Regression | Low bias, low variance, interpretable |
| True relationship non-linear, p small | KNN | Can capture non-linearity, flexible |
| p large / high-dimensional | Linear Regression | KNN suffers from curse of dimensionality |
| Interpretability important | Linear Regression | Coefficients and p-values available |
| Max prediction accuracy for non-linear, low-dimensional | KNN | Non-parametric, flexible |

> ⚠️ Key Takeaways:
> - Linear regression = parametric → efficient & interpretable, performs best if linear assumption holds  
> - KNN = non-parametric → flexible, adapts to non-linearity, suffers in high dimensions  
> - Tradeoff between bias, variance, and interpretability determines method choice
```
