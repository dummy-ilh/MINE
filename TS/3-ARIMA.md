Alright class, let's bring it all together and formalize the process of building **Non-Seasonal ARIMA Models**. This is often referred to as the **Box-Jenkins methodology**, a systematic approach to time series forecasting.

-----

## Non-Seasonal ARIMA Models: The Box-Jenkins Methodology

ARIMA stands for **A**uto**R**egressive **I**ntegrated **M**oving **A**verage. These models are a powerful class for forecasting time series data. They combine the three components we've discussed:

  * **AR (Autoregressive):** Uses past observations ($Y\_{t-1}, ..., Y\_{t-p}$) as predictors. The parameter is 'p'.
  * **I (Integrated):** Uses differencing to make the series stationary. The parameter is 'd'.
  * **MA (Moving Average):** Uses past forecast errors ($\\epsilon\_{t-1}, ..., \\epsilon\_{t-q}$) as predictors. The parameter is 'q'.

An ARIMA model is typically denoted as **ARIMA(p, d, q)**, where:

  * `p`: The order of the Autoregressive (AR) part.
  * `d`: The degree of differencing (the number of times the raw observations are differenced).
  * `q`: The order of the Moving Average (MA) part.

### The Box-Jenkins Methodology: A Four-Step Iterative Process

The Box-Jenkins approach is an iterative process of model building, rather than a single direct step.

1.  **Model Identification (Step 1):**

      * **Objective:** Determine the appropriate values for p, d, and q.
      * **How:**
          * **Check for Stationarity:**
              * **Visual inspection:** Plot the series. Look for trends, seasonality, and changing variance.
              * **Statistical tests:** Use ADF and/or KPSS tests to formally test for unit roots.
          * **Determine 'd' (Differencing):** If the series is non-stationary, apply differencing until it becomes stationary. The number of differences required is 'd'.
              * For trend: First-order differencing (d=1) often works.
              * For seasonality (if it hasn't been removed by seasonal differencing in SARIMA context): Non-seasonal ARIMA typically handles this by differencing, but it might not be the most efficient.
              * If variance is non-constant, a log transformation might be needed *before* differencing.
          * **Examine ACF and PACF of the *differenced* series:**
              * **Identify 'p' (AR order):** Look for a sharp cut-off in the **PACF** plot. The lag at which the PACF cuts off is a strong candidate for 'p'.
              * **Identify 'q' (MA order):** Look for a sharp cut-off in the **ACF** plot. The lag at which the ACF cuts off is a strong candidate for 'q'.
              * **Consider them together:**
                  * If PACF cuts off at p and ACF decays: Likely AR(p) model.
                  * If ACF cuts off at q and PACF decays: Likely MA(q) model.
                  * If both decay: Likely ARMA(p,q) model (or ARIMA if differenced). This requires more careful consideration and often trying different combinations.

2.  **Parameter Estimation (Step 2):**

      * **Objective:** Once p, d, and q are identified, estimate the coefficients ($\\phi$ for AR, $\\theta$ for MA) of the chosen ARIMA model.
      * **How:** This is typically done using statistical software (like `statsmodels` in Python or `forecast` package in R) that employs methods like Maximum Likelihood Estimation (MLE) or Conditional Sum of Squares.

3.  **Model Diagnostics (Step 3):**

      * **Objective:** Check if the estimated model is adequate and fits the data well, and if its assumptions are met.
      * **How:** This primarily involves analyzing the **residuals** of the fitted model.
          * **Residual Plot:** Plot the residuals over time. They should appear randomly scattered around zero, with no discernible patterns, trends, or changing variance.
          * **ACF/PACF of Residuals:** The ACF and PACF plots of the residuals should show *no significant spikes* (i.e., they should look like white noise). This is the most critical check.
          * **Ljung-Box Test:** Perform a formal Ljung-Box test on the residuals. A high p-value (typically \> 0.05) indicates that the residuals are statistically indistinguishable from white noise, meaning the model is adequate. If the p-value is low, the model is misspecified and needs refinement.
          * **Normality Test (e.g., Shapiro-Wilk):** Check if residuals are normally distributed (important for confidence intervals, less so for just point forecasts).
      * **Iterate:** If the diagnostics reveal problems (e.g., significant autocorrelation in residuals), return to Step 1 (Identification) and try a different model (e.g., adjust p, d, or q, or consider a seasonal model if seasonality was missed).

4.  **Forecasting (Step 4):**

      * **Objective:** Once a satisfactory model is identified, estimated, and validated, use it to forecast future values of the time series.
      * **How:** The model parameters are used to generate point forecasts and prediction intervals (confidence intervals) for future time steps.

-----

### What if More Than One Model Looks Okay? (Model Selection)

It's common for several ARIMA(p,d,q) combinations to appear plausible based on the initial ACF/PACF plots and even pass initial diagnostic checks. When this happens, you need a systematic way to choose the "best" model.

  * **Information Criteria (ICs):** These are statistical measures that balance model fit with model complexity (number of parameters). Lower values are better.
      * **Akaike Information Criterion (AIC):** $AIC = -2 \\ln(L) + 2k$, where L is the maximum likelihood of the model and k is the number of parameters.
      * **Bayesian Information Criterion (BIC):** $BIC = -2 \\ln(L) + k \\ln(n)$, where n is the number of observations.
      * **Preference:** BIC penalizes complexity more heavily than AIC, especially for larger datasets. AIC tends to select models that are good for prediction, while BIC tends to select true models (if one exists). For forecasting, AIC is often preferred.
  * **Parsimony:** Given two models with similar AIC/BIC values, the simpler model (with fewer parameters) is generally preferred.
  * **Forecasting Performance on Hold-out Data:** The ultimate test is how well the models perform on unseen data. Split your data into training and testing sets. Train multiple candidate models on the training data and then evaluate their forecasting accuracy (e.g., using RMSE, MAE) on the test set. The model with the best performance on the test set is often the preferred choice.

-----

### Residual Diagnostics in R (Conceptual)

In R (and Python), after fitting an ARIMA model, you can perform residual diagnostics very easily.

Let's imagine you fit an `ARIMA(1,1,0)` model (meaning AR order 1, 1st differencing, no MA component) to a time series `my_ts` using the `auto.arima()` or `Arima()` function from the `forecast` package:

```r
# Example in R (conceptual, not runnable here)
# library(forecast)
# model <- Arima(my_ts, order=c(1,1,0))
```

Then you would typically:

1.  **Plot Residuals:**

    ```r
    # plot(residuals(model))
    ```

      * **Expected:** Random scatter around zero, no patterns.

2.  **ACF/PACF of Residuals:**

    ```r
    # Acf(residuals(model))
    # Pacf(residuals(model))
    ```

      * **Expected:** No significant spikes beyond lag 0. All spikes should be within the blue confidence bounds.

3.  **Ljung-Box Test:**

    ```r
    # checkresiduals(model) # This function from 'forecast' package also plots ACF and does Ljung-Box
    # or explicitly:
    # Box.test(residuals(model), lag=20, type="Ljung-Box")
    ```

      * **Expected:** High p-value (e.g., $p \> 0.05$).

4.  **Normality (e.g., Histogram, QQ Plot):**

    ```r
    # hist(residuals(model))
    # qqnorm(residuals(model))
    # qqline(residuals(model))
    ```

      * **Expected:** Histogram to be bell-shaped, QQ plot points close to the line.

-----

### A Model with One Too Many Coefficients (Parameter Redundancy)

This is a critical concept often leading to diagnostic issues.

**Example 3.2 (Parameter Redundancy): Suppose that the model for your data is white noise.**

Let's say your underlying true data generating process is actually just white noise: $Y\_t = \\epsilon\_t$.
This means it's an ARIMA(0,0,0) process.

Now, imagine you incorrectly try to fit an **MA(1) model** to this white noise data:
$Y\_t = c + \\epsilon\_t + \\theta\_1 \\epsilon\_{t-1}$

  * If the true process is white noise, then any MA(1) model you fit will essentially try to find a $\\theta\_1$ close to zero. The model will struggle to find a significant $\\theta\_1$ because there's no true MA(1) component in the data.
  * The residuals might still *look* like white noise because the model isn't introducing any *new* structure, but it's adding an unnecessary parameter.
  * **Consequence (as stated in your prompt):**
      * "The MA(1) coefficient is significant (you can check it), but mostly this looks worse than the statistics for the right model." - This might happen if there's some random noise that looks like a small MA(1) effect, leading to a "significant" coefficient by chance, especially if the sample size is small.
      * "The estimate of the variance is 1.87, compared to 1.447 for the AR(1) model." (Assuming the "right model" here was actually AR(1) for a moment, or perhaps ARIMA(0,0,0) with a variance of 1.447). A higher variance of the error term (residuals) implies a worse fit. The model is explaining less of the total variance.
      * "The AIC and BIC statistics are higher for the MA(1) than for the AR(1). That’s not good." Higher AIC/BIC values indicate a worse model. This clearly points to the MA(1) being an over-parameterized or incorrectly specified model for the underlying white noise data (or if the true model was AR(1)).

**What does this illustrate?**

  * **Parsimony:** This example highlights the principle of **parsimony** in model selection: choose the simplest model that adequately explains the data. Adding unnecessary parameters (like the $\\theta\_1$ in the MA(1) when the data is white noise) does not improve the model fit, often makes it worse (higher variance, higher AIC/BIC), and increases the risk of overfitting.
  * **Parameter Redundancy:** When a model contains more parameters than necessary to capture the underlying process, it suffers from parameter redundancy. This can lead to:
      * Less precise parameter estimates (larger standard errors).
      * Higher variance in forecasts.
      * Increased computational cost.
      * Difficulty in interpreting the model.

**In an interview context:** If you're asked about parameter redundancy, you should explain that it's about including more terms than the data supports, leading to less efficient models and poorer generalization, and that information criteria (AIC/BIC) help guard against this by penalizing complexity. The goal is to find the simplest model that makes the residuals behave like white noise.


Class, let's continue our deep dive into the critical step of **Diagnostics** in the Box-Jenkins methodology. After we've identified and estimated a potential ARIMA model, it's absolutely vital to check if that model is actually doing a good job. This is where analyzing the residuals comes in, and specifically, understanding the statistical significance of their autocorrelation values.

---

## 3.2 Diagnostics: Analyzing Possible Statistical Significance of Autocorrelation Values

The core idea of diagnostics is to check if the residuals of your fitted model are **white noise**. If they are, it means your model has successfully captured all the predictable patterns in the original time series. If they are not, your model is inadequate, and you need to go back and refine it.

We visually inspect the ACF and PACF plots of the residuals. However, visual inspection alone isn't enough; we need statistical tests to confirm. This is where the **Ljung-Box test** (which we briefly discussed) comes into full play.

### Use of the Ljung-Box Statistic

The Ljung-Box test is a formal statistical test used to check for the presence of autocorrelation in the residuals of a time series model. It's a "portmanteau" test because it considers a group of autocorrelations together.

The **Ljung-Box Q statistic** for a series of $n$ observations and residuals $e_t$ is calculated as:

$$Q(m) = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{n-k}$$

Where:
* $n$: The number of observations in the time series (or residuals).
* $m$: The number of lags being tested (up to which autocorrelation is being considered). You choose this. A common practice is $m = \min(20, n/4)$.
* $\hat{\rho}_k$: The sample autocorrelation coefficient of the residuals at lag $k$.

**Interpretation:**
The logic here is that if the residuals are truly white noise, their sample autocorrelations ($\hat{\rho}_k$) should be very close to zero. If they are not zero, the $Q(m)$ statistic will be large.

### Distribution of $Q(m)$

The distribution of the Ljung-Box $Q(m)$ statistic depends on whether you are applying it to the original series or to the residuals of a *fitted* model. This distinction is crucial for correctly interpreting the p-value.

**Case 1: $Q(m)$ for the *Original Series* (Testing for Pure White Noise)**

* **Scenario:** You have a raw time series and you want to test if it's simply white noise (i.e., no structure, no predictability). You haven't fit any model yet.
* **Distribution:** If the series is truly white noise, then $Q(m)$ is approximately distributed as a **chi-squared ($\chi^2$) distribution with $m$ degrees of freedom**.
* **Hypotheses:**
    * $H_0$: The series is white noise (all autocorrelations are zero up to lag $m$).
    * $H_1$: The series is not white noise.
* **Decision:**
    * If the calculated $Q(m)$ value is **greater than the critical value** from the $\chi^2_{m}$ distribution (or if the p-value is less than your significance level, e.g., 0.05), you **reject $H_0$**. This means there is significant autocorrelation, and the series is not white noise.
    * If $Q(m)$ is less than the critical value (or p-value is greater than 0.05), you **fail to reject $H_0$**. The series appears to be white noise.

**Case 2: $Q(m)$ for the *Residuals of a Fitted ARIMA Model***

* **Scenario:** You have fitted an ARIMA(p,d,q) model to your time series, and now you want to test if the *residuals* from this model are white noise. This is the more common use case in diagnostics.
* **Distribution:** If the model is correctly specified, then $Q(m)$ is approximately distributed as a **chi-squared ($\chi^2$) distribution with $m - (p+q)$ degrees of freedom**.
    * The degrees of freedom are reduced by the number of estimated parameters (p AR terms and q MA terms) in the ARIMA model. The 'd' for differencing doesn't reduce the degrees of freedom directly here because differencing is applied to the data *before* fitting the AR/MA parts, effectively changing the series being modeled.
* **Hypotheses:**
    * $H_0$: The residuals are white noise (meaning the model has adequately captured the autocorrelation structure).
    * $H_1$: The residuals are not white noise (meaning the model is inadequate).
* **Decision:**
    * If the calculated $Q(m)$ value is **greater than the critical value** from the $\chi^2_{m-(p+q)}$ distribution (or if the p-value is less than 0.05), you **reject $H_0$**. This means there's significant autocorrelation left in the residuals, and your ARIMA model is *not* adequate. You need to go back to model identification.
    * If $Q(m)$ is less than the critical value (or p-value is greater than 0.05), you **fail to reject $H_0$**. This is the desired outcome, indicating that your model has adequately captured the autocorrelations, and the residuals are white noise.

**Important Notes for Diagnostics:**

* **Small p-value is Bad (for residuals):** In diagnostics of *residuals*, a small p-value for the Ljung-Box test is undesirable, as it implies the model is inadequate.
* **Number of Lags (m):** Choosing `m` is important. It should be large enough to capture potential significant autocorrelations but not too large (e.g., typically $m \le n/4$ and often $m=20$ or $24$ for monthly data).
* **Other Checks:** Remember that the Ljung-Box test is one piece of the diagnostic puzzle. Always combine it with visual inspection of residual plots, and consider normality and heteroscedasticity tests if those assumptions are critical for your application.

This detailed understanding of the Ljung-Box test and its degrees of freedom adjustment for fitted models is crucial for properly diagnosing your ARIMA models and ensuring their reliability.


Alright class, let's round out our discussion on ARIMA models by focusing on interview-style questions, particularly those related to prediction bands and the common pitfalls or considerations like deviations from stationarity and normality.

---

## Interview Questions on ARIMA, Prediction Bands, and Assumptions

### I. ARIMA Model Concepts & Implementation

1.  **Q: What does ARIMA stand for, and what is the role of each component (AR, I, MA)?**
    * **A:** ARIMA stands for **A**uto**R**egressive **I**ntegrated **M**oving **A**verage.
        * **AR (p):** Autoregressive component, uses a linear combination of past *observations* ($Y_{t-1}, ..., Y_{t-p}$) to predict the current value.
        * **I (d):** Integrated component, involves differencing the raw observations `d` times to make the series stationary.
        * **MA (q):** Moving Average component, uses a linear combination of past *forecast errors* ($\epsilon_{t-1}, ..., \epsilon_{t-q}$) to predict the current value.

2.  **Q: Describe the typical steps involved in building an ARIMA model using the Box-Jenkins methodology.**
    * **A:** The four-step iterative process:
        1.  **Identification:** Determine p, d, q by visually inspecting the time series for stationarity (trends, seasonality), applying differencing if needed to determine 'd', and then examining the ACF and PACF plots of the differenced series to suggest 'p' and 'q'.
        2.  **Estimation:** Estimate the coefficients of the chosen ARIMA model using methods like Maximum Likelihood Estimation.
        3.  **Diagnostic Checking:** Analyze the residuals of the fitted model to ensure they resemble white noise (no remaining autocorrelation). This involves plotting residuals, their ACF/PACF, and performing the Ljung-Box test. If residuals are not white noise, return to step 1.
        4.  **Forecasting:** Once the model passes diagnostics, use it to generate future forecasts and prediction intervals.

3.  **Q: How do you determine the 'd' (differencing order) in an ARIMA model?**
    * **A:** The 'd' is determined by the number of times the series needs to be differenced to achieve stationarity. This is primarily assessed through:
        * **Visual inspection:** Plotting the differenced series to see if trends/seasonality are removed.
        * **Unit root tests:** Statistical tests like the Augmented Dickey-Fuller (ADF) test (null hypothesis: non-stationary) or KPSS test (null hypothesis: stationary). You typically difference until these tests suggest stationarity.

4.  **Q: Explain the role of ACF and PACF plots in ARIMA model identification.**
    * **A:**
        * **ACF (Autocorrelation Function):** Shows the total correlation between an observation and its lagged values. Used primarily to identify the order 'q' of the MA component (it will cut off after lag 'q' for a pure MA process) and to detect seasonality.
        * **PACF (Partial Autocorrelation Function):** Shows the direct correlation between an observation and its lagged values after removing the influence of intermediate lags. Used primarily to identify the order 'p' of the AR component (it will cut off after lag 'p' for a pure AR process).
    * They are used complementarily to suggest initial 'p' and 'q' values.

### II. Prediction Bands (Confidence Intervals)

1.  **Q: What are prediction bands (or forecast intervals) in time series forecasting, and why are they important?**
    * **A:** Prediction bands are the uncertainty ranges around a point forecast. Instead of just providing a single predicted value, they give a range within which the future observation is expected to fall with a certain probability (e.g., 95% confidence interval).
    * They are crucial because:
        * **Quantify Uncertainty:** All forecasts have uncertainty. Prediction bands provide a quantitative measure of that uncertainty.
        * **Decision Making:** They enable better decision-making by showing the best-case and worst-case scenarios. For example, a business needs to know not just the most likely sales figure, but also the potential lower bound for planning inventory.
        * **Risk Assessment:** They allow stakeholders to assess the risk associated with a forecast. Wider bands indicate higher uncertainty and greater risk.

2.  **Q: How are prediction bands typically calculated for ARIMA models? What assumptions do they rely on?**
    * **A:** For ARIMA models, prediction bands are usually calculated based on the assumption that the **forecast errors are normally distributed and have a constant variance**.
    * The width of the prediction interval increases with the forecast horizon.
    * The formula for a $(1-\alpha)\%$ prediction interval for a forecast $\hat{Y}_{t+h}$ is approximately:
        $$\hat{Y}_{t+h} \pm Z_{\alpha/2} \sqrt{Var(\hat{Y}_{t+h})}$$
        Where $Z_{\alpha/2}$ is the critical value from the standard normal distribution (e.g., 1.96 for 95% CI) and $Var(\hat{Y}_{t+h})$ is the forecast error variance, which increases with $h$.
    * **Assumptions:**
        * **Normally distributed residuals (errors):** This is a key assumption for the validity of the $Z_{\alpha/2}$ multiplier.
        * **Homoscedasticity of errors:** The variance of the forecast errors is constant over time.
        * **Correctly specified model:** The ARIMA model fully captures the underlying data generation process, leaving only white noise residuals.

3.  **Q: What factors can cause prediction bands to be wider or narrower?**
    * **A:**
        * **Forecast Horizon:** Wider bands for longer horizons (more uncertainty further into the future).
        * **Variance of Residuals ($\sigma^2_{\epsilon}$):** Larger residual variance leads to wider bands (more unexplained noise).
        * **Model Complexity/Goodness of Fit:** A poorly fitting model (high residual variance) will have wider bands.
        * **Data Volatility:** Inherently more volatile series will have wider bands.
        * **Parameter Uncertainty:** Uncertainty in the estimated ARIMA coefficients also contributes, though often less significantly than residual variance for short horizons.

### III. Deviations from Assumptions (Stationarity, Normality, etc.)

1.  **Q: What happens if you apply an ARIMA model to a non-stationary series without proper differencing? What are the risks?**
    * **A:** Applying ARIMA to non-stationary data leads to **invalid inferences and unreliable forecasts**.
        * **Spurious Regressions:** You might find statistically significant relationships between variables that are purely coincidental.
        * **Incorrect Standard Errors:** The standard errors of your coefficients will be underestimated, leading to overly optimistic p-values and confidence intervals.
        * **Exploding Forecasts:** Forecasts can quickly diverge and become unrealistic.
        * **Invalid Hypothesis Testing:** Any hypothesis tests about coefficients or model fit will be unreliable.

2.  **Q: How do you handle a time series where the variance changes over time (heteroscedasticity)?**
    * **A:**
        * **Transformations:** A common approach is to apply a variance-stabilizing transformation, such as a **logarithmic transformation** (if variance increases with the mean) or a square root transformation.
        * **ARCH/GARCH Models:** For time series where volatility clustering (periods of high and low variance) is observed, **ARCH (Autoregressive Conditional Heteroscedasticity) or GARCH (Generalized ARCH) models** are specifically designed to model and forecast this changing variance. These can be combined with ARIMA (e.g., ARIMA-GARCH).

3.  **Q: Is normality of residuals a strict requirement for ARIMA forecasting? What if it's violated?**
    * **A:** For **point forecasts**, exact normality of residuals is *not a strict requirement* for the model to be valid. The AR/MA components are essentially linear models.
    * However, normality *is important* for:
        * **Calculating accurate prediction intervals/confidence levels:** The calculation of standard prediction intervals relies on the assumption of normally distributed errors to use Z-scores or t-scores.
        * **Hypothesis testing and parameter inference:** Statistical tests on coefficients assume normal errors.
    * **If violated:**
        * Point forecasts might still be robust.
        * Prediction intervals might be incorrect (too narrow or too wide). You might need to use **bootstrapping methods** to construct empirical prediction intervals that don't rely on the normality assumption.
        * Consider if the non-normality points to a missing component in the model (e.g., outliers, non-linear relationships not captured).

4.  **Q: What does it mean if the Ljung-Box test for your ARIMA model's residuals returns a significant p-value (e.g., < 0.05)? What would be your next steps?**
    * **A:** A significant p-value from the Ljung-Box test means you **reject the null hypothesis that the residuals are white noise**. This indicates that there is **still significant autocorrelation** (predictable patterns) left in your residuals, which your current ARIMA model has failed to capture.
    * **Next Steps:**
        1.  **Re-examine ACF/PACF of Residuals:** Look at the residual plots. If there's a spike at a specific lag, try adding an AR or MA term corresponding to that lag.
        2.  **Consider Higher Order:** Try increasing 'p' or 'q'.
        3.  **Check for Missed Seasonality:** If there are significant spikes at seasonal lags (e.g., 12 for monthly data, 7 for daily data) in the residual ACF/PACF, consider using a Seasonal ARIMA (SARIMA) model.
        4.  **Review 'd':** Ensure the differencing order 'd' is appropriate for stationarity.
        5.  **Look for Outliers/Structural Breaks:** Extreme outliers or sudden shifts in the series might not be handled well by standard ARIMA and could be affecting residuals.
        6.  **Consider Model Complexity:** If you've already added many terms, perhaps the model is too complex or the underlying process is not well-suited for a linear ARIMA model.

These questions cover a comprehensive range of ARIMA knowledge, touching on theory, practical application, and troubleshooting. Mastering these will demonstrate a strong understanding of traditional time series analysis.


Here’s an **interview‑ready summary** of Penn State STAT 510 **Lesson 3: Identifying & Estimating ARIMA Models; Forecasting**—structured clearly, with key points & terminology highlighted for ease of recall:

---

## 🧠 Lesson 3: Non‑Seasonal ARIMA Models & Forecasting

**Lesson Objectives**
After this lesson you should be able to:

1. Identify and interpret a non‑seasonal ARIMA model
2. Use ACF & PACF to distinguish AR/MA components
3. Test residual autocorrelations (e.g. Box‑Ljung)
4. Express an ARIMA model in infinite‑order MA form (psi‑weights)
5. Forecast future values and compute prediction intervals using ARIMA models
   ([PennState: Statistics Online Courses][1])

---

### 🔧 3.1 Specifying & Identifying ARIMA(p,d,q)

* An ARIMA model is denoted **(p,d,q)**:

  * **p** = AR order
  * **d** = number of differences
  * **q** = MA order
* **ARIMA(2,0,0)** = AR(2); **ARIMA(0,0,2)** = MA(2); **ARIMA(1,1,1)** applies to the first-differenced series
* Begin by inspecting:

  * A **time series plot** (trend, variance, outliers)
  * Sample **ACF & PACF** plots

    * AR: PACF cuts off after p; ACF tapers
    * MA: ACF cuts off after q; PACF tapers
    * ARMA: both taper gradually
  * Non-stationarity: ACF stays near 1 → consider differencing
  * White noise: insignificance across both ACF and PACF
    ([PennState: Statistics Online Courses][1], [PennState: Statistics Online Courses][2], [PennState: Statistics Online Courses][3])

---

### 🔍 3.2 Estimation & Diagnostic Checks

* Fit candidate models using R (e.g. `arima()` or `sarima()` from **astsa** package)
* Examine:

  * **Significance of AR/MA coefficients** (t‑values or p‑values >|1.96|)
  * **Residual ACF/PACF**: should show no significant autocorrelation
  * **Box‑Ljung test**: ideally non‑significant overall
  * **Residual plots**: check for patterns, variance issues
* When multiple models are plausible, prefer the simpler one (parsimony)
  ([PennState: Statistics Online Courses][1], [PennState: Statistics Online Courses][2])

---

### 🔮 3.3 Forecasting with ARIMA

* **Psi-weight representation**: convert ARIMA to infinite MA form:

* Use `ARMAtoMA(...)` in R to compute psi‑weights
* **Forecast equation**: plug in observed or forecasted values; assume future errors = 0
* **Forecast error variance**:

* **Standard error** = square root of the above; prediction interval:

* As horizon lengthens, forecast converges toward the series mean with increasing uncertainty
  ([PennState: Statistics Online Courses][1])

---

## 📋 Quick Reference Table

| Topic               | Summary                                                             |
| ------------------- | ------------------------------------------------------------------- |
| ARIMA specification | ARMA(p,q) on differenced data → ARIMA(p,d,q)                        |
| Identification      | Use ACF → MA, PACF → AR; both taper → ARMA                          |
| Diagnostics         | Check coefficient significance, residual ACF/PACF, Box‑Ljung, plots |
| Psi‑weights         | ARIMA → infinite MA representation via `ARMAtoMA`                   |
| Forecasting         | Plug-in method + error variance formula → SE, prediction intervals  |

---

## 💼 Interview Ready Tip

> *“I start by plotting the data to assess stationarity or variance issues. Then I use ACF & PACF patterns to propose ARIMA(p,d,q) candidates, fit them in R, and validate via coefficient significance and residual diagnostics like Box‑Ljung. Finally, I forecast using the fitted model, compute psi‑weights to find forecast standard errors, and build 95% prediction intervals.”*

