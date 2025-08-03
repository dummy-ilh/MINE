Here's a **condensed, interview-ready version** of your Lesson 5 notes, focusing on decomposition and smoothing techniques with ARIMA connections. This version keeps the key theory, R commands, and use-cases while cutting redundant text.

---

# **Lesson 5 – Decomposition & Smoothing Methods**

---

## **🎯 Objectives**

* Understand **additive vs multiplicative decompositions**
* Decompose a time series into **trend, seasonal, irregular** parts
* Use **LOWESS**, **moving averages**, and **exponential smoothing**
* Recognize links to **ARIMA models**

---

## **5.1 Decomposition Models**

### **1️⃣ Purpose**

* Identify trend and seasonal patterns
* Produce **seasonally adjusted values** for clearer trend detection

Example: U.S. unemployment is always lower in June → seasonal adjustment removes this repetitive dip.

---

### **2️⃣ Model Structures**

* **Additive:**

$$
X_t = T_t + S_t + I_t
$$

* **Multiplicative:**

$$
X_t = T_t \times S_t \times I_t
$$

* $I_t$: irregular/random noise

---

### **3️⃣ Choosing the Model**

* **Additive** → Seasonal variation remains constant (e.g., beer production)
* **Multiplicative** → Seasonal variation grows with trend level (e.g., Johnson & Johnson earnings)

---

### **4️⃣ Decomposition Steps**

1. **Estimate trend**:

   * Smoothing (moving averages) OR regression model
2. **De-trend**: subtract (additive) or divide (multiplicative)
3. **Estimate seasonal factors**: average (or median) detrended values for each season
4. **Random component**:

   * Additive: $I_t = X_t - T_t - S_t$
   * Multiplicative: $I_t = X_t / (T_t S_t)$
5. Iterate for improved estimates (software like Minitab does this).

---

### **5️⃣ Decomposition in R**

```r
series <- ts(data, freq=4) # quarterly
decomp <- decompose(series, type="additive") # or "multiplicative"
plot(decomp)
decomp$figure # seasonal effects
```

Example – Seasonal adjustment:

$$
\text{Adjusted} = X_t - S_t \quad (\text{additive})
$$

$$
\text{Adjusted} = X_t / S_t \quad (\text{multiplicative})
$$

---

### **6️⃣ Lowess-based Decomposition (stl)**

* Uses **locally weighted regression** for trend and seasonal estimates
* More flexible than classical decomposition

```r
plot(stl(series, s.window="periodic"))
```

---

## **5.2 Smoothing Methods**

Smoothing ≠ modeling → Used to **reveal trends** or **remove seasonality** before ARIMA fitting.

---

### **A) Moving Averages**

* Average of values around time $t$
* **One-sided:** past values only
* **Centered:** values before and after (common for trend detection)

Quarterly data (centered filter example):

```r
trend <- filter(beerprod, filter=c(1/8,1/4,1/4,1/4,1/8), sides=2)
plot(beerprod); lines(trend, col="red")
```

Removes seasonal variation → reveals underlying trend.

---

### **B) LOWESS Smoothing**

* Nonparametric local regression
* Flexible for nonlinear trends

```r
plot(lowess(unemploy, f=2/3))
```

---

### **C) Single Exponential Smoothing**

Forecast:

$$
\hat{x}_{t+1} = \alpha x_t + (1-\alpha)\hat{x}_t
$$

* $0<\alpha<1$: weight on recent value
* **Equivalent to ARIMA(0,1,1)** model with no constant
* Optimal $\alpha$ found via ARIMA fitting

Example:

```r
# ARIMA fit for exponential smoothing
fit <- arima(oilindex, order=c(0,1,1))
predicteds <- oilindex - resid(fit)
lines(predicteds, col="red")
```

---

### **D) Double Exponential Smoothing**

* Handles **trend without seasonality**
* Two smoothing constants:

  * **Level** (intercept)
  * **Trend** (slope)
* Equivalent to **ARIMA(0,2,2)** model

---

## **💡 Interview Cheat Sheet**

| Method                       | Purpose                                   | Equivalent ARIMA |
| ---------------------------- | ----------------------------------------- | ---------------- |
| Additive Decomposition       | Trend + seasonality, constant effect      | None (pre-step)  |
| Multiplicative Decomposition | Seasonal effect grows with level          | None (pre-step)  |
| Moving Averages              | Smooth noise, see trend                   | None             |
| LOWESS                       | Flexible nonlinear smoothing              | None             |
| Single Exponential           | Short-term forecast, no trend/seasonality | ARIMA(0,1,1)     |
| Double Exponential           | Trend forecast                            | ARIMA(0,2,2)     |

---

Would you like me to prepare a **Markdown version** of this note (with headings, code blocks, formulas, and a quick-reference table) that you can paste directly into your notes repo?


## Overview of Time Series Decomposition and Smoothing Methods

### 5.1 Decomposition Models

**Core Idea:**
Decomposition procedures aim to break down an observed time series ($Y_t$) into distinct, interpretable components:
1.  **Trend ($T_t$):** The long-term direction of the series.
2.  **Seasonal ($S_t$):** Regular, repeating patterns at a fixed frequency (e.g., quarterly, monthly).
3.  **Random/Irregular ($R_t$ or $E_t$):** The residual component that cannot be explained by trend or seasonality; essentially the "noise."

**Primary Objective:**
One of the main goals of decomposition is to **estimate seasonal effects** to create **seasonally adjusted values**. Seasonally adjusted values remove the typical seasonal fluctuations, allowing for a clearer view of the underlying trend. This is crucial for economic indicators (like unemployment rates) to distinguish true shifts from expected seasonal variations.

**Basic Structures:**

* **Additive Decomposition:**
    * **Formula:** $Y_t = T_t + S_t + R_t$
    * **When to Use:** When the **magnitude of seasonal variation (and random error)** is **relatively constant** over time. The size of the seasonal peaks and troughs remains consistent regardless of the series' overall level (as seen in Example 5.1, Australian Beer Production).

* **Multiplicative Decomposition:**
    * **Formula:** $Y_t = T_t \times S_t \times R_t$
    * **When to Use:** When the **magnitude of seasonal variation (and random error) increases or decreases proportionally** with the level of the time series. The seasonal peaks and troughs become larger or smaller as the series' overall level changes (as seen in Example 5.2, Johnson & Johnson Earnings).

**Basic Steps in Classical Decomposition:**

1.  **Estimate Trend ($T_t$):**
    * Most commonly, this is done using a **smoothing procedure** like a **moving average**.
    * For seasonal data, the moving average window length is typically set to the seasonal period (e.g., 4 for quarterly, 12 for monthly) to average out the seasonal fluctuations. A **centered moving average** is used for even seasonal periods to properly align the smoothed value with the original time point.
    * Alternatively, a regression equation can be used to model the trend.

2.  **De-trend the Series:**
    * **Additive:** Subtract the estimated trend from the original series: $Y_t - \hat{T}_t$. This leaves $S_t + R_t$.
    * **Multiplicative:** Divide the original series by the estimated trend: $Y_t / \hat{T}_t$. This leaves $S_t \times R_t$.

3.  **Estimate Seasonal Factors ($S_t$):**
    * For each specific season (e.g., each January for monthly data, each Quarter 1 for quarterly data), average (or median, as Minitab does) the de-trended values.
    * These seasonal effects are then adjusted:
        * For additive: They average to 0 over a full seasonal cycle.
        * For multiplicative: They average to 1 over a full seasonal cycle.

4.  **Determine Random/Irregular Component ($R_t$):**
    * **Additive:** $R_t = Y_t - \hat{T}_t - \hat{S}_t$
    * **Multiplicative:** $R_t = Y_t / (\hat{T}_t \times \hat{S}_t)$
    * The random component can then be analyzed (e.g., for its variance, or to see if it truly is random, possibly indicating a need for an ARIMA model on the remainder).

**Decomposition in R (`decompose` and `stl` functions):**

* **`decompose()`:**
    * Performs classical decomposition.
    * Syntax: `decompose(series_name, type = "additive" or "multiplicative")`.
    * **Crucial Step:** Requires the time series to be defined with its seasonal frequency using `ts()` (e.g., `my_series_ts = ts(my_data, freq = 4)` for quarterly data).
    * Output can be plotted directly (`plot(decompose(earnings, type="multiplicative"))`) or stored in an object for component access (e.g., `decompearn$figure` for seasonal effects).

* **`stl()` (Seasonal-Trend decomposition using Loess):**
    * **Preferred Method:** Generally more robust than `decompose()`, handling outliers and missing values better.
    * Performs an **additive decomposition** by default.
    * Uses **LOESS** (Locally Estimated Scatterplot Smoothing) to estimate both trend and seasonal components.
    * Syntax: `stl(series_name, s.window = "periodic" or integer)`. `"periodic"` causes seasonal effects to be constant; an integer allows seasonal effects to change over time.
    * Uses "remainder" instead of "random" for the irregular component.
    * Example 5.3 and 5.4 illustrate the `decompose` function for additive and multiplicative beer production series, showing the derived trend, seasonal, and random components and how seasonal factors are used for adjustment. Example 5.3 and 5.4 also demonstrate the use of `$figure` to extract seasonal values and their application in de-seasonalizing future values.

### 5.2 Smoothing Time Series

**Concept:** Smoothing aims to "iron out" the irregular fluctuations in a time series to reveal clearer underlying patterns like trend or cycles. It's often a preliminary step for analysis or a basic forecasting method. The term "filter" is sometimes used synonymously with smoothing procedure.

* **Identify and interpret additive and multiplicative decompositions:** (Already covered in 5.1).
* **Decompose a time series:** (Already covered in 5.1).

**Apply a Lowess Smoother (LOESS/LOWESS):**
* **Concept:** A non-parametric smoothing technique that fits local regression models to subsets of data. It's flexible and can capture complex, non-linear trends without assuming a specific functional form.
* **Use:** Excellent for visualizing trends in noisy data, especially when the trend isn't strictly linear. It's the core smoothing algorithm used in the `stl()` decomposition.
* **R Example (Conceptual):** `plot(lowess(unemploy, f = 2/3))` (where `f` is the smoothing parameter, determining the span of data points used in each local regression). Example 5.6 shows its application to US Unemployment data.

**Apply a Moving Averages Smoother:**
* **Concept:** Calculates the average of a specified number of observations (a "window") around each point in time.
* **Use:** Primarily to remove high-frequency noise and seasonality, revealing the trend.
* **Types:**
    * **Simple Moving Average:** All observations in the window are weighted equally.
    * **Centered Moving Average:** For an even window size (like a seasonal period of 4 or 12), the average is "centered" between observations to align with the original time points. This involves averaging two simple moving averages.
* **R Example (`filter` command):**
    * `filter(series, filter = c(...), sides = 2)`: `filter` specifies the weights (e.g., `c(1/8, 1/4, 1/4, 1/4, 1/8)` for a centered moving average over 4 quarters). `sides = 2` indicates a two-sided (centered) filter.
    * Example 5.5 demonstrates this for Australian beer production to extract the trend, and then how to subtract the trend to see the seasonality. `sides = 1` creates a one-sided (trailing) filter.

**Apply a Single Exponential Smoother (SES):**
* **Concept:** A basic forecasting method for time series that are **stable (no trend or seasonality)**. It calculates the forecast as a weighted average of the most recent observation and the previous forecast, with weights decaying exponentially into the past.
* **Forecasting Equation:** $\hat{Y}_{t+1} = \alpha Y_t + (1 - \alpha) \hat{Y}_t$, where $\alpha$ is the smoothing constant (0 to 1).
    * High $\alpha$ (closer to 1): More weight on recent data, less smoothing, quicker response to changes.
    * Low $\alpha$ (closer to 0): More weight on past data, more smoothing, slower response.
* **Equivalence to ARIMA(0,1,1):** Critically, SES is **mathematically equivalent to an ARIMA(0,1,1) model with no constant term**. This means applying SES implies that your data is best modeled by an ARIMA(0,1,1) process.
    * **Implication:** You shouldn't blindly apply SES; it's optimal only if an ARIMA(0,1,1) truly represents the underlying process. The optimal $\alpha$ value can be found by fitting an ARIMA(0,1,1) model.
* **Use:** Primarily for short-run forecasting of stable series.
* **R Example (Conceptual):** `library(forecast); ets(oilindex, model="ANN")` (Additive error, No Trend, No Seasonality). Example 5.7 provides a detailed illustration of fitting an ARIMA(0,1,1) to an oil index and interpreting its equivalence to SES.

**Double Exponential Smoothing:**
* **Concept:** An extension of SES for time series that exhibit a **trend but no seasonality**. It involves two smoothing equations: one for the level of the series and one for the trend (slope).
* **Equivalence:** This method is equivalent to fitting an **ARIMA(0,2,2) model with no constant term**.

Okay, let's craft some tricky interview questions around the concepts of Time Series Decomposition, Smoothing, and their relationship with ARIMA, aiming to probe deeper than just definitions.

## Tricky Interview Questions & Answers: Decomposition, Smoothing, and ARIMA

### Question Set 1: Choosing Decomposition & Its Implications

**Q1: You're analyzing a monthly sales dataset that shows a clear upward trend and distinct seasonality. When you plot the data, you notice the peaks of sales are getting progressively higher over time, but the troughs don't seem to be falling proportionally. They are just increasing slightly. Would you choose an additive or multiplicative decomposition, and why? What's a potential pitfall of choosing the "wrong" one?**

**A1:** This is a tricky one because the description isn't perfectly clean-cut for either.
* **Initial Thought (and most common):** The "peaks getting progressively higher over time" strongly suggests a **multiplicative decomposition**. In a multiplicative model, the seasonal effect scales with the trend.
* **The Nuance:** The "troughs just increasing slightly, not proportionally" introduces a slight ambiguity. Pure multiplicative implies *all* components (peaks and troughs) scale proportionally. This observation might suggest a scenario where seasonality is partially multiplicative and partially additive, or where the variance of the error term might be growing but not perfectly tied to the seasonal factor.
* **Decision Strategy:** Given the stronger indicator of "peaks getting progressively higher," I would **start with a multiplicative decomposition**. If the residuals of this decomposition still show patterns (e.g., heteroscedasticity or remaining seasonality), I might then consider a **log transformation** of the data, which turns a multiplicative relationship into an additive one ($log(Y_t) = log(T_t) + log(S_t) + log(R_t)$), and then apply an additive decomposition. This is a common and robust approach for multiplicative data.
* **Potential Pitfall of Choosing the Wrong One:**
    * **Choosing Additive when Multiplicative is Better:** The **seasonally adjusted series will still exhibit increasing (or decreasing) seasonal amplitude** as the trend changes. The seasonal component removed by the additive model will be of a fixed magnitude, failing to account for the growing seasonal swing. This means the "seasonally adjusted" data would still have a seasonal pattern, leading to incorrect interpretation of the underlying trend and potentially biased forecasts. The residuals might also show heteroscedasticity.
    * **Choosing Multiplicative when Additive is Better:** The **seasonally adjusted series might show artificially shrinking (or growing) variability** at higher (or lower) levels of the series, distorting the trend. The model would "over-adjust" for seasonality, assuming a proportional increase where none exists, making the deseasonalized series look artificially compressed or expanded. The residuals might also display a non-constant variance (homoscedasticity would be violated).

### Question Set 2: The "Random" Component & ARIMA

**Q2: After performing a decomposition, the "Random" (or "Remainder") component of your time series still shows significant autocorrelation. What does this imply, and how would you proceed? Can you still use this decomposed series for forecasting?**

**A2:**
* **Implication:** If the "Random" component (residuals) shows significant autocorrelation, it implies that your decomposition model (whether additive or multiplicative) has **not fully captured all the systematic patterns** in the original time series. There's still predictable information left in what should ideally be pure white noise. This is a violation of the assumption that the random component is truly random.
* **How to Proceed:**
    1.  **Re-evaluate the Decomposition:** First, confirm the decomposition type (additive vs. multiplicative) was correct. Sometimes, a subtle mischoice here can leave patterns in the residuals.
    2.  **Model the Remainder:** If the decomposition itself is robust, the standard practice is to **model the "Random" component using an ARIMA or SARIMA model**. The goal is to capture the remaining autocorrelation. You would treat the "Random" series as a new time series and apply the Box-Jenkins methodology (identification, estimation, diagnostics) to it.
    3.  **Combine Forecasts:** Your final forecast would then be a combination of the forecasted Trend, Seasonal component (which can be extrapolated), and the forecast from the ARIMA model applied to the Random component.
* **Can you still use this decomposed series for forecasting?**
    * **Yes, but with caution and modification.** You shouldn't *directly* use just the trend and seasonal components for forecasting without addressing the autocorrelated remainder. If you ignore the autocorrelation in the remainder, your point forecasts might be less accurate, and crucially, your prediction intervals will be too narrow and unreliable because they won't account for the structured uncertainty in the remainder.
    * Therefore, you need to either fix the decomposition or, more commonly, model the "Random" component separately with an ARIMA/SARIMA model to improve the overall forecast accuracy and interval reliability.

### Question Set 3: Smoothing vs. Modeling & SES

**Q3: You're considering using Simple Exponential Smoothing (SES) for a time series that appears to be stationary. A colleague suggests just fitting an ARIMA(0,1,1) model instead. Are these two approaches fundamentally different, or is there a strong connection? Why might you choose one over the other?**

**A3:** This question probes the equivalence and practical considerations.

* **Strong Connection / Equivalence:** Yes, there's a very strong connection. **Simple Exponential Smoothing (SES) is mathematically equivalent to an ARIMA(0,1,1) model with no constant term.** This means that when you apply SES, you are implicitly assuming that the underlying data generating process can be described by an ARIMA(0,1,1) model. The smoothing parameter $\alpha$ in SES is directly related to the MA(1) coefficient ($\theta_1$) in the ARIMA(0,1,1) model ($\theta_1 = 1 - \alpha$).
* **Why Choose One Over the Other (Practical Considerations):**
    * **Simplicity/Intuition (SES):** SES is often easier to explain and understand conceptually, especially for non-technical stakeholders. It's a "simpler" algorithm to implement manually.
    * **Optimality and Diagnostic Rigor (ARIMA(0,1,1)):**
        * When you fit an ARIMA(0,1,1) model using statistical software, the software typically uses Maximum Likelihood Estimation (MLE) to find the **optimal** MA(1) coefficient (and thus the optimal $\alpha$) that best fits your data. This provides a more statistically rigorous and generally better-performing "smoothing" than just picking an $\alpha$ value arbitrarily.
        * Furthermore, fitting an ARIMA(0,1,1) allows you to perform **diagnostic checks on the residuals** (Ljung-Box test, ACF/PACF of residuals). If the residuals are *not* white noise, it immediately tells you that even an ARIMA(0,1,1) (and thus SES) is an inadequate model for your data, prompting you to explore other ARIMA/SARIMA orders. SES alone doesn't provide these diagnostic insights as directly.
    * **Model Selection:** If your colleague suggests ARIMA(0,1,1), it implies they are thinking about the formal modeling process. The choice between SES and ARIMA(0,1,1) often comes down to the level of rigor and diagnostics desired. For quick, simple forecasts on truly flat data, SES is fine. For anything more serious, leveraging the full ARIMA framework (even if it's just ARIMA(0,1,1)) for optimal parameter estimation and diagnostics is superior.

### Question Set 4: The Trade-off in Smoothing

**Q4: You're using a moving average smoother to identify the trend in a very noisy time series. How does the choice of the moving average window length (`k`) impact the results, and what's the fundamental trade-off you're making when selecting `k`?**

**A4:**
* **Impact of Window Length (`k`):**
    * **Larger `k`:** Results in a **smoother** trend line. It averages over more data points, effectively reducing more of the random noise and short-term fluctuations. However, it introduces more **lag** (the smoothed line will lag behind the true underlying trend) and can **oversmooth or obscure genuine short-term changes** in the trend. It also leads to more `NA` values at the beginning and end of the series.
    * **Smaller `k`:** Results in a **less smooth** trend line. It retains more of the original series' variability, responding more quickly to changes in the underlying trend. However, it will **not effectively remove as much noise**, making the trend harder to discern, and the smoothed series will still appear "rough."
* **Fundamental Trade-off:** The fundamental trade-off when selecting `k` is between **smoothness (noise reduction) and responsiveness (retaining trend detail/avoiding lag)**.
    * You want `k` large enough to effectively filter out the noise and periodicity (if smoothing out seasonality), but not so large that it excessively smooths out genuine changes in the underlying trend or introduces too much lag, causing the smoothed line to misrepresent recent shifts.
    * For identifying trend in seasonal data, `k` is typically chosen to be the seasonal period (`s`) to ensure that each smoothed value averages over a full cycle of seasonal effects. For non-seasonal data, it's more about experimentation and visual assessment of the desired level of "signal" versus "noise."
