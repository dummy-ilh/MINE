
# $R^2$ and Adjusted $R^2$

---

## 1. What is the concept?

### $R^2$ (Coefficient of Determination)
$R^2$ measures the **proportion of variance in the response variable** that is explained by the regression model.

It answers:
> *How much better is this model compared to predicting the mean of $Y$?*

---

### Adjusted $R^2$
Adjusted $R^2$ is a **penalized version of $R^2$** that accounts for the **number of predictors** in the model.

It answers:
> *How much variance is explained **after correcting for model complexity**?*

---

## 2. Intuition

### $R^2$ intuition
- Baseline model: predict $\bar{Y}$ for all observations.
- Regression model: predict $\hat{Y}$ using features.
- $R^2$ compares **error reduction** relative to the baseline.

Interpretation:
- $R^2 = 0$: model is no better than predicting the mean
- $R^2 = 1$: perfect prediction
- $R^2 < 0$: model is worse than predicting the mean

---

### Adjusted $R^2$ intuition
- Adding predictors **always increases or keeps $R^2$ the same**, even if the feature is pure noise.
- Adjusted $R^2$ introduces a **penalty for unnecessary variables**.
- It increases **only if a new variable improves the model more than expected by chance**.

---

## 3. Mathematical formulation

### Notation
Let:
- $n$ = number of observations  
- $p$ = number of predictors (excluding intercept)  
- $Y_i$ = true response  
- $\hat{Y}_i$ = predicted response  
- $\bar{Y}$ = mean of $Y$

---

### Total Sum of Squares (TSS)
$$
TSS = \sum_{i=1}^{n} (Y_i - \bar{Y})^2
$$

### Residual Sum of Squares (RSS)
$$
RSS = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

---

### $R^2$
$$
R^2 = 1 - \frac{RSS}{TSS}
$$

Equivalent interpretation:
$$
R^2 = \frac{\text{Explained Variance}}{\text{Total Variance}}
$$

---

### Adjusted $R^2$
$$
\text{Adjusted } R^2
= 1 - \left( \frac{RSS / (n - p - 1)}{TSS / (n - 1)} \right)
$$

Expanded form:
$$
\text{Adjusted } R^2
= 1 - (1 - R^2)\frac{n - 1}{n - p - 1}
$$

---

## 4. Why the concept matters

### Theoretical importance
- Quantifies **goodness-of-fit**
- Enables **model comparison** against a null baseline
- Central to **ANOVA decomposition** in linear models

---

### Practical importance
- Quick sanity check for regression performance
- Used in:
  - Feature selection
  - Model diagnostics
  - Reporting model quality to stakeholders

---

## 5. Biases, assumptions, and limitations

### Shared assumptions
- Assumes **linear regression framework**
- Sensitive to:
  - Outliers
  - Heteroscedasticity
  - Model misspecification

---

### Limitations of $R^2$
| Issue | Explanation |
|-----|------------|
| Always increases | Adding useless features inflates $R^2$ |
| Not predictive | High $R^2$ does not imply good generalization |
| Scale-dependent | Depends on variance of target |
| Not comparable | Cannot compare across different datasets |

---

### Limitations of Adjusted $R^2$
- Still assumes **linearity**
- Penalizes only by **count of variables**, not their complexity
- Can decrease even when a variable is **statistically significant**

---

## 6. Common pitfalls and misconceptions

### Misconception 1: High $R^2$ means good model
False.
- Overfitting can yield high $R^2$
- Poor out-of-sample performance is common

---

### Misconception 2: $R^2$ measures causality
False.
- $R^2$ is purely **descriptive**
- No causal inference without experimental design

---

### Misconception 3: Adjusted $R^2$ solves overfitting
Partially false.
- It corrects **in-sample inflation**
- Does **not guarantee generalization**

---

## 7. How to detect issues (diagnostics)

### Red flags
| Symptom | Likely Issue |
|------|------------|
| High $R^2$, poor test RMSE | Overfitting |
| Negative $R^2$ on test set | Severe misspecification |
| Large gap between $R^2$ and Adjusted $R^2$ | Too many weak predictors |

---

### Diagnostic checks
- Train vs test performance
- Residual plots
- Cross-validated $R^2$
- Compare with baseline predictors

---

## 8. How to fix or improve issues

### Better evaluation metrics
| Scenario | Prefer |
|-------|-------|
| Prediction | RMSE, MAE |
| Model selection | AIC, BIC |
| Generalization | Cross-validated $R^2$ |

---

### Regularization
- Ridge: reduces variance
- Lasso: feature selection
- Elastic Net: balance

---

### Model improvements
- Feature engineering
- Transformations (log, polynomial)
- Nonlinear models

---

## 9. Connections to other ML concepts

| Concept | Connection |
|------|-----------|
| ANOVA | Variance decomposition |
| Bias–Variance Tradeoff | High $R^2$ may indicate high variance |
| AIC/BIC | Penalized model selection |
| Cross-validation | Reliable $R^2$ estimation |
| Likelihood | $R^2$ relates loosely to explained deviance |

---

## 10. Real-world / FAANG-style applications

### Product analytics
- Measuring impact of features on engagement metrics

### Forecasting
- Baseline comparison for demand or revenue models

### Experiment analysis
- Regression adjustment in A/B tests

### Risk & pricing
- Interpreting explanatory power vs predictability

---

## 11. Deep interview questions (with answers)

### Q1. Can $R^2$ be negative?
**Yes.**  
If $RSS > TSS$, the model performs worse than predicting $\bar{Y}$.

---

### Q2. Why does adding a random variable increase $R^2$?
Because least squares **minimizes RSS**, and adding dimensions can only reduce or keep RSS constant.

---

### Q3. Why might Adjusted $R^2$ decrease after adding a statistically significant variable?
Because the penalty depends on **variance explained**, not $p$-values.

---

### Q4. Is Adjusted $R^2$ suitable for nonlinear models?
No. It is derived under the **linear regression ANOVA framework**.

---

### Q5. Why do FAANG teams rarely rely on $R^2$ alone?
Because:
- It is in-sample
- It ignores distributional assumptions
- It does not measure business impact

---

## Follow-up / Deeper Questions

1. How does $R^2$ relate to likelihood under Gaussian noise?
2. Why is $R^2$ undefined for models without intercept?
3. How does $R^2$ differ from explained deviance in GLMs?
4. When is Adjusted $R^2$ preferable to AIC or BIC?
5. How does cross-validated $R^2$ behave under dataset shift?

---


