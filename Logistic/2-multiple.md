Great! Let’s build a **comprehensive set of notes on multiple logistic regression**, written in a clear, detailed way based on Hosmer et al., suitable for deep understanding without needing other sources.

---

# **Multiple Logistic Regression**

Multiple logistic regression extends the simple logistic regression model to include **two or more independent variables**. It allows modeling the effect of several predictors simultaneously on a binary outcome.

---

## **1. Model Definition**

For a binary outcome (Y \in {0,1}) and predictors (X_1, X_2, \dots, X_p), the **logit (log-odds) of the probability** of success is modeled as a linear combination of predictors:

[
\text{logit}[\pi(x)] = \ln \frac{\pi(x)}{1 - \pi(x)} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p
]

Where:

* (\pi(x) = P(Y=1 \mid X_1, \dots, X_p))
* (\beta_0) = intercept
* (\beta_j) = log-odds coefficient for (X_j)

Equivalently, the **predicted probability** is:

[
\pi(x) = \frac{e^{\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p}}
]

---

## **2. Interpretation of Coefficients**

1. **(\beta_j)**: change in the **log-odds** of the outcome per unit increase in (X_j), holding all other predictors constant.

2. **Odds ratio**:

[
OR_j = e^{\beta_j}
]

* Represents the multiplicative change in odds for a **one-unit increase in (X_j)**, holding other variables constant.
* Example: (\beta_2 = 0.5 \Rightarrow OR_2 = e^{0.5} \approx 1.65), meaning the odds increase by 65% per unit increase in (X_2).

---

## **3. Likelihood Function**

For (n) independent observations:

[
L(\beta) = \prod_{i=1}^{n} \pi_i^{y_i} (1 - \pi_i)^{1 - y_i},
\quad \pi_i = \frac{e^{\beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}}}{1 + e^{\beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}}}
]

The **log-likelihood** is:

[
\ell(\beta) = \sum_{i=1}^{n} \big[ y_i \ln(\pi_i) + (1 - y_i) \ln(1 - \pi_i) \big]
]

* Maximum likelihood estimation (MLE) is used to find (\hat{\beta}_0, \dots, \hat{\beta}_p).
* Unlike linear regression, no closed-form solution exists; iterative methods (e.g., Newton-Raphson) are used.

---

## **4. Testing Significance**

### **4.1 Likelihood Ratio Test (LRT)**

* Compare **nested models**: full model with all predictors vs. reduced model excluding a variable (or set of variables).
* Statistic:

[
G = -2 \ln \frac{L_{\text{reduced}}}{L_{\text{full}}} = D_{\text{reduced}} - D_{\text{full}}
]

* (G \sim \chi^2_{\text{df}}), df = # of variables added.
* Large (G) → variable(s) significantly improve fit.

### **4.2 Wald Test**

* For each coefficient (\beta_j):

[
W_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}, \quad W_j^2 \sim \chi^2_1
]

* Tests (H_0: \beta_j = 0) (no effect), holding other variables constant.

### **4.3 Score Test**

* Based on derivatives of log-likelihood at null model.
* Useful when fitting full model is expensive; less commonly implemented in software for multiple predictors.

---

## **5. Model Fit Assessment**

1. **Deviance**:

[
D = -2 \ell(\text{fitted model})
]

* Compares model fit to **saturated model** (perfect prediction).
* Reduction in deviance when adding variables indicates improved fit.

2. **Pseudo R² measures** (Hosmer & Lemeshow):

* **Cox & Snell R²**, **Nagelkerke R²**: give an approximate measure of explained variability.

3. **Goodness-of-fit tests**:

* Hosmer-Lemeshow test compares observed vs predicted outcomes in groups.

---

## **6. Multicollinearity**

* Highly correlated predictors can inflate standard errors of coefficients.
* Diagnostic: variance inflation factor (VIF) or condition indices.
* If severe multicollinearity exists: consider dropping or combining predictors.

---

## **7. Interaction Terms**

* Interaction between variables allows **effect modification**:

[
\text{logit}(\pi) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \cdot X_2)
]

* (\beta_3) represents the additional effect on log-odds when both (X_1) and (X_2) occur together.
* Test significance using LRT or Wald test.

---

## **8. Confidence Intervals**

* For coefficients: Wald-based CI:

[
\hat{\beta}*j \pm z*{1-\alpha/2} SE(\hat{\beta}_j)
]

* For odds ratio:

[
CI_{\text{OR}} = \big[ e^{\hat{\beta}_j - z SE}, , e^{\hat{\beta}_j + z SE} \big]
]

* For predicted probability:

[
\pi(x) = \frac{e^{\hat{g}(x)}}{1 + e^{\hat{g}(x)}}, \quad
\text{CI for } \pi(x) = \frac{e^{\hat{g}(x) \pm z SE[\hat{g}(x)]}}{1 + e^{\hat{g}(x) \pm z SE[\hat{g}(x)]}}
]

---

## **9. Practical Interpretation**

1. Multiple logistic regression allows **adjustment** for confounders.
2. Coefficients represent **independent effects** holding other predictors constant.
3. Model-based predicted probabilities can be used for **classification** (risk prediction).
4. Assess model carefully using **deviance, pseudo-R², and goodness-of-fit tests**.

---

### **10. Summary Table: Multiple Logistic Regression Concepts**

| Concept                    | Formula / Description                            |
| -------------------------- | ------------------------------------------------ |
| Model                      | logit(π) = β₀ + β₁X₁ + … + β_p X_p               |
| Predicted probability      | π(x) = e^{β₀+…}/(1+e^{β₀+…})                     |
| Coefficient interpretation | β_j: change in log-odds per unit increase of X_j |
| Odds ratio                 | OR_j = e^{β_j}                                   |
| Significance tests         | Likelihood Ratio, Wald, Score                    |
| Confidence interval (β_j)  | β̂_j ± z SE(β̂_j)                                |
| Confidence interval (OR)   | [e^{β̂_j - zSE}, e^{β̂_j + zSE}]                 |
| Model fit                  | Deviance, pseudo R², Hosmer-Lemeshow test        |

---

If you want, I can **draw a visual diagram showing the structure of multiple logistic regression**, including **predictors, coefficients, logit, odds, and predicted probability**, which makes it extremely intuitive how the model works.

Do you want me to make that diagram next?
