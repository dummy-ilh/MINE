
# $R^2$ and Adjusted $R^2$

---

## 1. What is the concept?

### $R^2$ (Coefficient of Determination)
$R^2$ measures the **proportion of variance in the response variable** that is explained by the regression model.
Ranges from 0 and 1
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
R^2 = \frac{\text{Explained Var}}{\text{Total Var}}
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

It sounds like you're preparing for a Data Science or Machine Learning interview at a FAANG+ company. These roles often test your deep understanding of model evaluation metrics beyond just knowing the formulas.

Here is a breakdown of  and Adjusted  through the lens of medium-to-hard interview questions.

---

## 1. The Fundamentals

** (Coefficient of Determination):** Measures the proportion of variance in the dependent variable that is predictable from the independent variables.


**Adjusted :** Penalizes the model for adding features that do not improve its predictive power.



*Where  is the number of data points and  is the number of independent regressors.*

---

## 2. FAANG Interview Questions

### Question 1: "Can  ever be negative? If so, what does it signify?"

**Answer:**
Yes,  can be negative. While it is often thought of as a "squared" value (suggesting it must be positive), it is actually a comparison to a baseline model.

* **When it happens:** If your model performs **worse** than a horizontal line representing the mean of the data.
* **The Significance:** It indicates that your chosen model does not follow the trend of the data at all, or you have applied a linear model to data that is non-linear without proper transformation.

### Question 2: "Why do we need Adjusted  if we already have ? Give a specific mathematical scenario."

**Answer:**
The mathematical flaw of  is that it is **non-decreasing**. Every time you add a new predictor—even if it's just random noise— will likely decrease (or stay the same) by pure chance, causing  to increase.

* **The Scenario:** If you are predicting house prices and you add a column of "Random Numbers" to your dataset, your  will technically go up.
* **The Solution:** Adjusted  incorporates  (number of features). If the increase in  is not significant enough to offset the increase in , the Adjusted  will decrease, signaling that the new feature is adding noise rather than value.

### Question 3: "If you have a high  but your Adjusted  is significantly lower, what does this tell you about your feature set?"

**Answer:**
This is a classic sign of **overfitting** and redundant features.

* It suggests that you have "kitchen-sinked" your model—adding many variables that are not truly informative.
* In a FAANG context (where datasets have thousands of features), this signals a need for **Feature Selection** or **Regularization** (Lasso/Ridge) to prune the model.

### Question 4: "Is a high  always 'good'? Why or why not?"

**Answer:**
Not necessarily.

1. **Overfitting:** A very high  (e.g., 0.99) on training data might mean the model has just memorized the noise.
2. **Spurious Correlation:** In time-series data, two unrelated variables that both trend upward over time will show a high , even if there is no causal link.
3. **Nature of the Field:** In social sciences, an  of 0.3 might be great because human behavior is hard to predict. In physics, an  of 0.9 might be considered poor for a controlled experiment.

---

## 3. Comparison Table

| Feature |  | Adjusted  |
| --- | --- | --- |
| **Primary Purpose** | Explains variance. | Evaluates feature quality. |
| **Effect of Adding Features** | Always increases or stays same. | Can increase OR decrease. |
| **Reliability** | Low (prone to overfitting). | High (better for model selection). |
| **Value Range** | Usually 0 to 1 (can be negative). | Can be negative; usually . |

---

md
# Deep Dive: Advanced Questions on $R^2$, Adjusted $R^2$, and Related Concepts

This document answers **advanced, interview-level questions** around $R^2$, its statistical meaning, limitations, and modern alternatives. The focus is on **theory + implications**, as expected in FAANG L5/L6 interviews.

---

## 1. How does $R^2$ relate to likelihood under Gaussian noise?

### Setup
Assume the classical linear regression model:
$$
Y_i = X_i^\top \beta + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

---

### Log-likelihood
The Gaussian log-likelihood is:
$$
\log L(\beta, \sigma^2)
= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} RSS
$$

where:
$$
RSS = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2
$$

---

### Connection to $R^2$
Recall:
$$
R^2 = 1 - \frac{RSS}{TSS}
$$

- Maximizing likelihood $\iff$ minimizing $RSS$
- Since $TSS$ is fixed for a given dataset, **maximizing $R^2$ is equivalent to maximizing likelihood** under Gaussian noise.

---

### Key insight (interview-grade)
> Under **homoscedastic Gaussian errors**, $R^2$ is a **monotonic transformation of the log-likelihood**.

---

### Limitation
- This equivalence **breaks** if:
  - Errors are non-Gaussian
  - Variance is heteroscedastic
  - Model is misspecified

---

## 2. Why is $R^2$ undefined (or misleading) for models without an intercept?

### Core reason
$R^2$ relies on the decomposition:
$$
TSS = ESS + RSS
$$

This identity holds **only when the model includes an intercept**.

---

### Why the intercept matters
With an intercept:
- Residuals sum to zero:
$$
\sum_{i=1}^n (Y_i - \hat{Y}_i) = 0
$$
- $\bar{Y}$ is the optimal constant predictor

Without intercept:
- Baseline model (predicting $\bar{Y}$) is **not nested** inside the regression model
- Variance decomposition breaks

---

### Consequences
- $RSS$ can be **larger or smaller arbitrarily**
- $R^2$ can exceed $1$ or be highly negative
- Interpretation as "variance explained" is invalid

---

### Interview takeaway
> $R^2$ is only meaningful when the model contains an intercept, because it depends on ANOVA-based variance decomposition.

---

## 3. Why can $R^2$ be negative?

### Mathematical condition
From definition:
$$
R^2 = 1 - \frac{RSS}{TSS}
$$

If:
$$
RSS > TSS \Rightarrow R^2 < 0
$$

---

### Intuitive explanation
- The model predicts **worse than the constant mean predictor**
- Happens when:
  - Model is badly misspecified
  - Severe overfitting evaluated on test data
  - Dataset shift
  - No intercept model

---

### Common scenarios
| Scenario | Explanation |
|------|-----------|
| Test-set evaluation | Model fits train well but generalizes poorly |
| High noise | Model chases noise |
| Wrong features | No predictive signal |
| Distribution shift | Mean baseline is better |

---

### Interview framing
> Negative $R^2$ means the model is **actively harmful** relative to a trivial baseline.

---

## 4. What is the usual range of $R^2$?

### Theoretical range
- **Training set (with intercept)**:
$$
0 \le R^2 \le 1
$$

- **Test set / CV**:
$$
-\infty < R^2 \le 1
$$

---

### Practical interpretation
| Value | Interpretation |
|----|---------------|
| $R^2 \approx 1$ | Near-perfect fit |
| $R^2 \approx 0$ | No better than mean |
| $R^2 < 0$ | Worse than baseline |

---

### Important nuance
There is **no universal "good" $R^2$**.
- $0.05$ may be excellent in economics
- $0.90$ may be poor in physics

---

## 5. How does $R^2$ differ from explained deviance in GLMs?

### $R^2$ (Linear Regression)
- Based on **squared error loss**
- Uses **variance decomposition**
- Assumes Gaussian noise

---

### Explained Deviance (GLMs)
Defined as:
$$
\text{Explained Deviance}
= 1 - \frac{D_{\text{model}}}{D_{\text{null}}}
$$

Where:
- $D = -2 \log L$
- Likelihood depends on the **exponential family**

---

### Key differences
| Aspect | $R^2$ | Explained Deviance |
|----|------|----------------|
| Loss | Squared error | Log-likelihood |
| Noise | Gaussian | Exponential family |
| Interpretation | Variance explained | Likelihood improvement |
| Use case | Linear regression | GLMs |

---

### Interview insight
> Explained deviance is the **likelihood-based analogue of $R^2$**, suitable for non-Gaussian targets.

---

## 6. When is Adjusted $R^2$ preferable to AIC or BIC?

### Adjusted $R^2$
$$
\text{Adj } R^2 = 1 - (1 - R^2)\frac{n - 1}{n - p - 1}
$$

---

### Comparison
| Criterion | Penalizes | Goal |
|------|--------|------|
| Adjusted $R^2$ | Number of predictors | Variance explanation |
| AIC | Likelihood + parameters | Prediction |
| BIC | Likelihood + parameters + $n$ | Model recovery |

---

### Prefer Adjusted $R^2$ when:
- Comparing **nested linear models**
- Goal is **interpretability**
- Dataset size is moderate
- Gaussian assumptions hold

---

### Prefer AIC/BIC when:
- Nonlinear models
- Non-Gaussian noise
- Comparing non-nested models
- Production model selection

---

### Interview-grade statement
> Adjusted $R^2$ is a descriptive, variance-based criterion; AIC/BIC are likelihood-based and asymptotically grounded.

---

## 7. How does cross-validated $R^2$ behave under dataset shift?

### Definition
Cross-validated $R^2$:
$$
R^2_{\text{CV}} = 1 - \frac{\sum (Y - \hat{Y}_{\text{CV}})^2}{\sum (Y - \bar{Y})^2}
$$

---

### Under dataset shift
Types of shift:
- Covariate shift: $P(X)$ changes
- Label shift: $P(Y)$ changes
- Concept drift: $P(Y|X)$ changes

---

### Observed behavior
| Shift Type | Effect on $R^2_{\text{CV}}$ |
|------|----------------------------|
| Mild covariate shift | Gradual degradation |
| Label shift | Baseline mean changes |
| Concept drift | Often negative $R^2$ |

---

### Why it fails
- Mean baseline $\bar{Y}$ becomes unstable
- Variance of $Y$ changes
- Learned relationships no longer hold

---

### Interview insight
> Cross-validated $R^2$ is **not robust to dataset shift** and should be paired with business metrics and stability checks.

---

## 8. Additional High-Value Interview Questions (with Answers)

### Q1. Can two models have the same $R^2$ but very different predictions?
**Yes.**  
$R^2$ ignores:
- Calibration
- Local errors
- Distributional correctness

---

### Q2. Does higher Adjusted $R^2$ imply better generalization?
**No.**  
It is still an **in-sample statistic**.

---

### Q3. Why is $R^2$ inappropriate for classification?
Because:
- No notion of variance around class labels
- Loss is not squared error
- Likelihood-based metrics are preferred

---

### Q4. Why do FAANG teams often report $R^2$ alongside RMSE?
- $R^2$: relative explanatory power
- RMSE: absolute error magnitude

---





