Here's a **structured, interview-oriented note** on Lesson 8 (Regression with ARIMA errors, CCF, and relationships between time series). This condenses the key theory, equations, and R code you’ll likely need.

---

# **Lesson 8 – Regression with ARIMA Errors, Cross-Correlation Functions, and Relationships Between Time Series**

---

## **🎯 Objectives**

* Fit regression models where **errors are autocorrelated**
* Combine regression and ARIMA modeling
* Understand **cross-correlation functions (CCF)** between two series
* Interpret **lagged relationships** and build **transfer function models**

---

## **8.1 Linear Regression Models with Autoregressive Errors**

### **1️⃣ Standard Regression Setup**

Basic regression model:

$$
Y_t = \beta_0 + \beta_1 X_t + \epsilon_t
$$

* Assumes $\epsilon_t$ are **independent, identically distributed** (iid) white noise.

If residuals $\epsilon_t$ **are autocorrelated**, OLS estimates:

* Still **unbiased** but **not efficient**
* Standard errors and tests become invalid → wrong inference

---

### **2️⃣ Regression with AR Errors**

Suppose errors follow an AR(p) process:

$$
\epsilon_t = \phi_1 \epsilon_{t-1} + \dots + \phi_p \epsilon_{t-p} + a_t
$$

where $a_t$ is white noise.

Model:

$$
Y_t = \beta_0 + \beta_1 X_t + \epsilon_t
$$

$$
\epsilon_t = \phi_1 \epsilon_{t-1} + \dots + \phi_p \epsilon_{t-p} + a_t
$$

Equivalent to:

$$
(1 - \phi_1 B - \dots - \phi_p B^p) Y_t = \beta_0(1 - \phi_1 B - \dots - \phi_p B^p) + \beta_1 (1 - \phi_1 B - \dots - \phi_p B^p)X_t + a_t
$$

where $B$ = backshift operator.

**Estimation process:**

1. Fit OLS regression → get residuals $\hat{\epsilon_t}$.
2. Fit ARIMA model to residuals.
3. Re-estimate regression coefficients accounting for AR structure (e.g., via GLS).

---

### **3️⃣ Regression with ARIMA Errors**

More generally:

$$
\epsilon_t = \text{ARIMA}(p,d,q)
$$

So:

$$
Y_t = \beta_0 + \beta_1 X_t + \epsilon_t
$$

with dependent noise.

Handled in R via:

```r
# Using the arima() function with exogenous regressor X
fit <- arima(Y, order=c(p,d,q), xreg=X)
```

This approach jointly estimates:

* Regression coefficients $\beta$
* ARIMA parameters $(p,d,q)$

---

## **8.2 Cross-Correlation Functions (CCF) and Lagged Regressions**

### **1️⃣ Purpose**

* Identify **lead-lag relationships** between two time series $X_t$ and $Y_t$.
* Helps decide whether:

  * $X_t$ predicts $Y_t$ (transfer function)
  * Relationship occurs at a lag $k$

---

### **2️⃣ Definition of Cross-Correlation**

For lag $k$:

$$
r_{XY}(k) = \frac{\sum_t (X_{t-k} - \bar X)(Y_t - \bar Y)}
{\sqrt{\sum_t (X_{t-k} - \bar X)^2 \sum_t (Y_t - \bar Y)^2}}
$$

* **Positive $k$**: $X$ **leads** $Y$ (predictive effect)
* **Negative $k$**: $Y$ **leads** $X$

---

### **3️⃣ Prewhitening**

Issue:

* Both series may have autocorrelation → spurious cross-correlations.

Solution:

1. Fit ARIMA model to $X_t$ (predictor).
2. Filter both $X_t$ and $Y_t$ using same ARIMA model (remove autocorrelation).
3. Compute CCF on filtered (prewhitened) series.

---

### **4️⃣ Transfer Function Models**

If $X_t$ influences $Y_t$ dynamically (with delay):

$$
Y_t = c + \sum_{k=0}^s \omega_k X_{t-k} + N_t
$$

* $N_t$: noise, usually ARIMA
* $\omega_k$: impulse response weights

This is a **dynamic regression model** or ARIMAX.

---

### **5️⃣ R Code Examples**

```r
# Regression with ARIMA errors
fit <- arima(Y, order=c(1,0,0), xreg=X) # AR(1) errors

# Cross-correlation
ccf(X, Y, main="Cross Correlation between X and Y")

# Prewhitening
modX <- arima(X, order=c(1,0,0))
resX <- residuals(modX)
resY <- residuals(filter(Y, filter=modX$model$phi))
ccf(resX, resY)
```

---

## **💡 Interview Cheat Sheet**

| Concept                      | Key Idea / Formula                      | Notes                          |
| ---------------------------- | --------------------------------------- | ------------------------------ |
| Regression with AR errors    | Residuals follow AR(p)                  | GLS improves efficiency        |
| Regression with ARIMA errors | Errors \~ ARIMA(p,d,q)                  | Use `arima(..., xreg=X)`       |
| Cross-Correlation (CCF)      | Correlation between $X_{t-k}$ and $Y_t$ | Detects lead/lag relationships |
| Prewhitening                 | Remove autocorrelation before CCF       | Avoids spurious results        |
| Transfer Function            | $Y_t = c+\sum \omega_k X_{t-k}+N_t$     | Dynamic regression (ARIMAX)    |

---
Let's delve into the fascinating and crucial area of **Time Series Regression**, where we combine the power of linear regression with the nuanced understanding of time series dynamics. This is particularly important because traditional linear regression assumptions are often violated when applied directly to time series data.

-----

## 8.1 Linear Regression Models with Autoregressive Errors

When we perform a standard linear regression ($Y\_t = \\beta\_0 + \\beta\_1 X\_t + \\epsilon\_t$) on time series data, a common issue arises: the **error terms ($\\epsilon\_t$) are often not independent**. Instead, they frequently exhibit autocorrelation, meaning the error at one time point is correlated with errors at previous time points. This violates a key assumption of Ordinary Least Squares (OLS) regression, leading to:

  * **Inefficient (but still unbiased) coefficient estimates:** The OLS estimates for $\\beta\_0$ and $\\beta\_1$ might still be correct on average, but they won't be the most precise (they'll have larger standard errors than necessary).
  * **Incorrect Standard Errors:** The calculated standard errors of the regression coefficients will be underestimated, leading to overly optimistic (too narrow) confidence intervals and inflated t-statistics. This makes you more likely to incorrectly conclude that a predictor is statistically significant when it's not.
  * **Invalid Hypothesis Tests:** F-tests and t-tests for coefficient significance become unreliable.

To address this, we use **Regression Models with Autoregressive (AR) Errors** (or more generally, ARIMA errors).

### The Regression Model with AR Errors

This model assumes that the errors from a linear regression follow an Autoregressive process.

**Model Form:**

1.  **Regression Component:** $Y\_t = \\beta\_0 + \\beta\_1 X\_t + \\eta\_t$

      * $Y\_t$: The dependent (response) variable at time $t$.
      * $X\_t$: The independent (predictor) variable at time $t$.
      * $\\beta\_0$: The intercept.
      * $\\beta\_1$: The slope (regression coefficient).
      * $\\eta\_t$: The error term at time $t$.

2.  **AR Error Component:** The crucial part is that $\\eta\_t$ is not white noise, but follows an AR(p) process:

      * $\\eta\_t = \\phi\_1 \\eta\_{t-1} + \\phi\_2 \\eta\_{t-2} + \\dots + \\phi\_p \\eta\_{t-p} + \\epsilon\_t$
      * $\\phi\_i$: Autoregressive coefficients.
      * $\\epsilon\_t$: White noise error term (independent and identically distributed, often assumed normal).

**Interpretation:** The model states that $Y\_t$ is linearly related to $X\_t$, but the *unexplained variation* (the error $\\eta\_t$) itself has a time series structure that can be modeled by an AR process.

**Estimation:**
This model is typically estimated using methods like Maximum Likelihood Estimation (MLE) or Generalized Least Squares (GLS), which jointly estimate the regression coefficients ($\\beta$'s) and the AR coefficients ($\\phi$'s). This effectively "filters" out the autocorrelation from the errors, allowing for more accurate and efficient estimation of the regression coefficients.

### The Regression Model with ARIMA Errors

This is a generalization of the AR errors model. Instead of just AR errors, the error term ($\\eta\_t$) is assumed to follow a full ARIMA(p,d,q) process.

**Model Form:**

1.  **Regression Component:** $Y\_t = \\beta\_0 + \\beta\_1 X\_t + \\eta\_t$
2.  **ARIMA Error Component:** The error term $\\eta\_t$ follows an ARIMA(p,d,q) process:
      * $(1 - \\phi\_1 B - \\dots - \\phi\_p B^p)(1-B)^d \\eta\_t = (1 + \\theta\_1 B + \\dots + \\theta\_q B^q) \\epsilon\_t$
      * $B$: Backshift operator.
      * $\\phi\_i$: Autoregressive coefficients.
      * $\\theta\_i$: Moving Average coefficients.
      * $d$: Differencing order (to make $\\eta\_t$ stationary).
      * $\\epsilon\_t$: White noise error term.

**When to Use:**
This model is used when the residuals of a simple linear regression (or the errors of an AR error model) still exhibit significant autocorrelation that can be best described by an ARIMA structure (e.g., they need differencing or an MA component).

**Advantages:**

  * Provides more efficient and accurate estimates of regression coefficients compared to OLS when errors are autocorrelated.
  * Allows for proper hypothesis testing on regression coefficients.
  * Enables improved forecasting by modeling the time series structure of both the dependent variable (through $X\_t$) and the error term.

**How to Recognize When to Use (and "How to adjust for residuals with a time series structure"):**

1.  Perform an initial OLS regression: `lm(Y ~ X)`.
2.  Examine the **residuals** of this OLS model:
      * Plot the residuals over time.
      * Check their ACF and PACF plots.
      * Perform a Ljung-Box test on the residuals.
3.  If the residual diagnostics indicate significant autocorrelation (non-white noise residuals), then you need to adjust for this structure using an ARIMA error model.
4.  The specific (p,d,q) orders for the error process are typically determined by analyzing the ACF/PACF of the OLS residuals, just like in a standard ARIMA model building process.

**Estimating the adjusted intercept and slope:**
Software packages (like `forecast::Arima` in R, or `statsmodels.tsa.arima.model.ARIMA` in Python) allow you to specify exogenous variables directly when fitting an ARIMA model. When you do this, the model estimates the regression coefficients ($\\beta$'s) and the ARIMA parameters ($\\phi$'s, $\\theta$'s) simultaneously, giving you the "adjusted" intercept and slope that account for the error autocorrelation.

### R Code (Conceptual)

```r
# Assuming Y and X are time series objects (ts or xts)
# 1. Initial OLS regression
ols_model <- lm(Y ~ X)

# 2. Check residuals
plot(residuals(ols_model))
acf(residuals(ols_model))
pacf(residuals(ols_model))
Box.test(residuals(ols_model), type = "Ljung-Box")

# 3. If residuals show autocorrelation, fit ARIMA with exogenous regressors
# Using the 'forecast' package's Arima function for example:
# (p,d,q) for the ARIMA error model, xreg for the exogenous variable(s)
# The dependent variable 'Y' is modeled as ARIMA, but with 'X' as a predictor that
# influences the mean, and the residuals of that regression follow an ARIMA(p,d,q) structure.
# Often, 'd' here refers to differencing the 'Y' variable, not the 'error' directly,
# but the effect is the same as the error being differenced.
# A common practice is to model Y with ARIMA(p,d,q) and include X as an external regressor.

# To directly model Y ~ X + ARIMA(p,d,q) error:
library(forecast)
arima_reg_model <- Arima(Y, order=c(p,d,q), xreg=X)

# To see adjusted coefficients
summary(arima_reg_model)
# The 'xreg' coefficient will be your adjusted slope, and the intercept will be adjusted too.
```

## 8.2 Cross Correlation Functions and Lagged Regressions

When working with two or more time series, we're often interested in understanding how changes in one series relate to changes in another, and specifically, if there are **lagged relationships**. That is, does $X\_t$ influence $Y\_{t+k}$ (i.e., $X$ leads $Y$), or does $Y\_t$ influence $X\_{t+k}$ (i.e., $Y$ leads $X$), or are they contemporaneous?

### Cross-Correlation Function (CCF)

  * **Concept:** The Cross-Correlation Function (CCF) measures the correlation between two time series, $X\_t$ and $Y\_t$, at various lags. It helps identify lead-lag relationships.
  * **Formula (Conceptual):** The CCF at lag $k$, denoted $\\rho\_{xy}(k)$, measures the correlation between $X\_t$ and $Y\_{t+k}$.
      * $\\rho\_{xy}(k) = \\text{Corr}(X\_t, Y\_{t+k})$
      * For positive $k$ (e.g., $k=1, 2, ...$): Correlation between $X\_t$ and future values of $Y\_{t+k}$. A significant positive spike here means $X$ is a leading indicator for $Y$.
      * For negative $k$ (e.g., $k=-1, -2, ...$): Correlation between $X\_t$ and past values of $Y\_{t+k}$ (which is equivalent to $Y\_t$ and future values of $X\_{t-k}$). A significant positive spike here means $Y$ is a leading indicator for $X$.
      * For $k=0$: Contemporaneous correlation between $X\_t$ and $Y\_t$.
  * **Interpretation:**
      * A large, statistically significant spike at a positive lag $k$ (e.g., CCF at lag +5 is significant) indicates that $X\_t$ is correlated with $Y\_{t+5}$. This suggests that changes in $X$ tend to precede changes in $Y$ by 5 time units. (X leads Y).
      * A large, statistically significant spike at a negative lag $k$ (e.g., CCF at lag -3 is significant) indicates that $X\_t$ is correlated with $Y\_{t-3}$. This suggests that changes in $Y$ tend to precede changes in $X$ by 3 time units. (Y leads X).
      * Spikes outside the confidence bounds indicate significant correlation.
  * **Important Caveat ("Spurious Correlation"):** Just like with ACF, the CCF can show spurious correlations if either (or both) of the series are non-stationary. If $X\_t$ and $Y\_t$ both have a trend, their CCF will often show high correlations at many lags, even if there's no true causal relationship.
      * **Solution:** Always **pre-whiten** the series before calculating the CCF. This means fitting an ARIMA model to *one* of the series (say, $X\_t$) to make it white noise, then applying the same ARIMA filter to the *other* series ($Y\_t$). The CCF is then calculated between the pre-whitened series. This removes the "noise" and isolates the true cross-correlation structure.

### Lagged Regressions (Transfer Function Models / Dynamic Regression)

  * **Concept:** Lagged regressions explicitly model the relationship between a dependent variable $Y\_t$ and current and/or past values of one or more independent variables $X\_t$. These are often called **Transfer Function Models** or **Dynamic Regression Models**.
  * **Model Form:**
    $Y\_t = \\beta\_0 + \\sum\_{j=0}^{m} \\delta\_j X\_{t-j} + \\eta\_t$
      * $\\delta\_j$: Regression coefficients for $X\_t$ at different lags $j$.
      * $\\eta\_t$: The error term, which often follows an ARIMA process (as discussed in 8.1).
  * **Identifying and Interpreting Transfer Function Models:**
    1.  **Pre-whitening:** This is critical. Use the pre-whitened CCF (as described above) to identify potential significant lags. If the CCF between pre-whitened $X$ and pre-whitened $Y$ shows a strong spike at lag $k=0$ and $k=1$, it suggests including $X\_t$ and $X\_{t-1}$ as predictors in your model.
    2.  **Model Fitting:** Fit a regression model with the identified lagged predictors.
    3.  **Error Modeling:** Analyze the residuals of this lagged regression. If they are not white noise, then model these residuals with an ARIMA process. This leads back to the **Regression with ARIMA Errors** framework.
  * **Interpretation:** The coefficients $\\delta\_j$ directly tell you the impact of $X$ at specific lags on $Y$. For example, if $\\delta\_2$ is significant, it means $X\_{t-2}$ has a direct impact on $Y\_t$, controlling for other included lags.

### Objectives Review

  * **Recognize when and how to adjust for residuals with a time series structure:** When residuals from an OLS regression on time series data show significant autocorrelation (via ACF, PACF, Ljung-Box test), you must adjust. This is done by explicitly modeling the residual series with an ARIMA process (i.e., using Regression with ARIMA errors).
  * **Estimate the adjusted intercept and slope:** This is done by using specialized time series regression functions (e.g., `Arima` in R's `forecast` package) that jointly estimate both the regression coefficients and the ARIMA parameters of the error term.
  * **Interpret the cross-correlation function:** Use CCF plots (ideally after pre-whitening) to identify leading or lagging relationships between two time series. Significant positive lags ($X\_t \\to Y\_{t+k}$) or negative lags ($Y\_t \\to X\_{t+k}$) indicate potential causality or predictive power.
  * **Identify and interpret transfer function models:** These are lagged regression models. Identification involves using the pre-whitened CCF to suggest which lags of the predictor variable(s) should be included. Interpretation involves understanding that the coefficients for each lag represent the direct impact of the predictor at that specific past time point on the current dependent variable, *after* accounting for other lags and the ARIMA structure of the errors.

This area is complex but powerful, allowing for robust modeling of relationships between dynamic variables.
