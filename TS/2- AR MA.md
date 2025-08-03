Alright, let's turn our attention to one of the foundational building blocks of time series modeling: **Autoregressive (AR) Models**.

---

## Autoregressive (AR) Model

The core idea behind an Autoregressive (AR) model is that the current value of a time series ($Y_t$) can be explained as a **linear combination of its own past values**, plus a random error term. It's like saying that what happened in the past directly influences what's happening now.

The "auto" in autoregressive refers to the regression of the variable on itself (its past values).

**General Form of an AR(p) Model:**

An autoregressive model of order *p*, denoted as **AR(p)**, is mathematically expressed as:

$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$$

Where:
* $Y_t$: The value of the time series at the current time $t$.
* $c$: A constant term (intercept).
* $\phi_1, \phi_2, ..., \phi_p$: The autoregressive coefficients (parameters) that quantify the linear dependence on past values.
* $Y_{t-1}, Y_{t-2}, ..., Y_{t-p}$: The past observed values of the time series at lags 1, 2, ..., p, respectively.
* $\epsilon_t$: A white noise error term (also known as a random shock or disturbance) at time $t$. This term accounts for the unpredictable variation in the series that the model cannot explain. It's assumed to be independently and identically distributed (i.i.d.) with a mean of zero and constant variance.

**Key Characteristics of AR Models:**

1.  **Stationarity Requirement:** AR models require the time series to be **stationary** (specifically, weakly stationary). If the series has trends or seasonality, it usually needs to be differenced or transformed first. For an AR(p) process to be stationary, certain conditions must be met regarding the values of the $\phi$ coefficients (e.g., for an AR(1) process, $|\phi_1| < 1$).
2.  **PACF for Order Selection:** The **Partial Autocorrelation Function (PACF)** plot is the primary tool for determining the order 'p' of an AR model. For a pure AR(p) process, the PACF will show significant spikes up to lag 'p' and then cut off (become non-significant) for lags greater than 'p'.
3.  **ACF Behavior:** The **Autocorrelation Function (ACF)** for an AR(p) process typically decays gradually (either exponentially or in a damped sinusoidal pattern). This is because the correlation at longer lags is indirectly influenced by the direct correlations at shorter lags.
4.  **Forecasting:** Once the coefficients ($\phi$ values) are estimated, the model can be used to forecast future values by recursively plugging in the predicted values.

---

## AR(1) vs. AR(2) Models

The number 'p' in AR(p) defines the "order" of the model, which indicates how many past observations are used to predict the current one. Let's compare AR(1) and AR(2).

### AR(1) Model (First-Order Autoregressive Model)

* **Formula:**
    $$Y_t = c + \phi_1 Y_{t-1} + \epsilon_t$$
* **Interpretation:** The current value of the series ($Y_t$) depends *only* on its immediately preceding value ($Y_{t-1}$) and a random shock.
* **Behavior:**
    * **$\phi_1 > 0$:** Positive autocorrelation. High values tend to be followed by high values, and low values by low values (e.g., a smooth, trending series).
    * **$\phi_1 < 0$:** Negative autocorrelation. High values tend to be followed by low values, and vice versa (e.g., an oscillating or zig-zagging series).
    * **Stationarity Condition:** An AR(1) process is stationary if and only if **$|\phi_1| < 1$**. If $|\phi_1| \ge 1$, the series is non-stationary (e.g., a random walk if $\phi_1 = 1$).
* **ACF/PACF:**
    * **ACF:** Decays exponentially.
    * **PACF:** Has a significant spike at lag 1 and then cuts off to zero (or within the confidence bounds) for all subsequent lags.

### AR(2) Model (Second-Order Autoregressive Model)

* **Formula:**
    $$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \epsilon_t$$
* **Interpretation:** The current value ($Y_t$) depends on its two most recent past values ($Y_{t-1}$ and $Y_{t-2}$) and a random shock.
* **Behavior:** An AR(2) model can exhibit more complex patterns than an AR(1), including:
    * **Damped oscillations:** Depending on the values of $\phi_1$ and $\phi_2$, the series can show wave-like or cyclical behavior that gradually diminishes.
    * **Trend-like behavior:** Similar to AR(1), but potentially more complex.
* **Stationarity Conditions:** More complex than AR(1). The conditions relate to the roots of the characteristic equation $1 - \phi_1 B - \phi_2 B^2 = 0$ (where B is the backshift operator). All roots must lie outside the unit circle. Specifically, for an AR(2) to be stationary:
    1.  $|\phi_2| < 1$
    2.  $\phi_1 + \phi_2 < 1$
    3.  $\phi_2 - \phi_1 < 1$
* **ACF/PACF:**
    * **ACF:** Decays, often with a damped oscillatory pattern.
    * **PACF:** Has significant spikes at lags 1 and 2, and then cuts off to zero for all subsequent lags.

**Key Difference Summary:**

| Feature         | AR(1) Model                                             | AR(2) Model                                                   |
| :-------------- | :------------------------------------------------------ | :------------------------------------------------------------ |
| **Number of Lags** | Uses only the immediate past value ($Y_{t-1}$).         | Uses the two immediate past values ($Y_{t-1}, Y_{t-2}$).     |
| **Complexity** | Simpler, captures direct dependence on the last period. | More complex, can capture more intricate patterns like oscillations. |
| **PACF Cut-off**| At lag 1.                                               | At lag 2.                                                     |
| **ACF Decay** | Pure exponential decay.                                 | Damped exponential or damped sinusoidal decay.                 |
| **Stationarity**| $|\phi_1| < 1$.                                         | More complex conditions involving both $\phi_1$ and $\phi_2$.|

---

## Core Concepts of AR Models

1.  **Autocorrelation:** The fundamental premise is that past values are correlated with current values. This "self-correlation" is what the model attempts to capture.
2.  **Linearity:** AR models assume a linear relationship between the current value and its past values.
3.  **Stationarity:** As discussed, this is crucial. Non-stationary data needs to be transformed (e.g., differenced) to achieve stationarity before applying AR models.
4.  **Order (p):** The number of past observations included in the model, determined by examining the PACF plot.
5.  **White Noise Error Term:** The assumption that the residuals (the unexplained part) are random, uncorrelated, and have constant variance. This is key for statistical inference and model validity.
6.  **Yule-Walker Equations:** These are a set of equations that relate the AR coefficients ($\phi$) to the theoretical autocorrelations ($\rho$) of a stationary AR process. They are used to estimate the AR coefficients.

---

## Interview Questions and Answers on AR Models

1.  **Q: What is an Autoregressive (AR) model in time series analysis?**
    * **A:** An AR model predicts the current value of a time series based on a linear combination of its own past values, plus a white noise error term. It assumes that past observations contain information useful for forecasting future values.

2.  **Q: What is the significance of the order 'p' in an AR(p) model?**
    * **A:** The order 'p' indicates the number of past lagged observations that are included as predictors in the model. For example, an AR(1) uses the immediate past value, while an AR(2) uses the two most recent past values.

3.  **Q: How do you determine the appropriate order 'p' for an AR model?**
    * **A:** The primary tool for determining 'p' is the **Partial Autocorrelation Function (PACF)** plot. For a pure AR(p) process, the PACF plot will show significant spikes up to lag 'p' and then cut off sharply to non-significant values (within the confidence bounds) for lags greater than 'p'.

4.  **Q: What does the ACF plot look like for a typical AR(p) process?**
    * **A:** The ACF plot for an AR(p) process typically **decays gradually**. For AR(1) it's an exponential decay. For AR(2) or higher, it might exhibit a damped oscillatory or sinusoidal pattern before decaying.

5.  **Q: What are the stationarity conditions for an AR(1) model?**
    * **A:** An AR(1) model is stationary if and only if the absolute value of its autoregressive coefficient, $|\phi_1|$, is less than 1 ($|\phi_1| < 1$). If $|\phi_1| \ge 1$, the series is non-stationary (e.g., a random walk if $\phi_1=1$).

6.  **Q: How do AR models differ from standard regression models?**
    * **A:** In standard regression, the independent variables (predictors) are typically distinct from the dependent variable. In AR models, the dependent variable is regressed on *its own* past lagged values. This explicitly accounts for the temporal dependence inherent in time series data, which is usually violated in standard regression assumptions.

7.  **Q: When would you consider using an AR model over an MA model?**
    * **A:** You would consider an AR model when the PACF plot shows a sharp cut-off after a certain lag 'p' and the ACF plot decays gradually. This pattern is characteristic of an autoregressive process.
Class, let's now turn our attention to the second fundamental building block of time series modeling: **Moving Average (MA) Models**. These models offer a different perspective on how past information influences the present.

---

## Moving Average (MA) Models

While Autoregressive (AR) models relate a current observation to past *observations*, Moving Average (MA) models relate a current observation to past *forecast errors* (or "shocks" / "innovations"). The idea is that any unexplained deviation from the mean in the past might still have a lingering effect on the current value.

The "moving average" in the name is a bit misleading, as it's not a simple average of past values, but rather a weighted average of past *error terms*.

**General Form of an MA(q) Model:**

A moving average model of order *q*, denoted as **MA(q)**, is mathematically expressed as:

$$Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

Where:
* $Y_t$: The value of the time series at the current time $t$.
* $c$: A constant term (often the mean of the series if it's mean-zero).
* $\epsilon_t$: The current white noise error term (shock or innovation) at time $t$. This is assumed to be independently and identically distributed (i.i.d.) with a mean of zero and constant variance $\sigma^2_\epsilon$.
* $\epsilon_{t-1}, \epsilon_{t-2}, ..., \epsilon_{t-q}$: The past white noise error terms at lags 1, 2, ..., q, respectively. These are the *unobservable* errors from previous forecasts.
* $\theta_1, \theta_2, ..., \theta_q$: The moving average coefficients (parameters) that quantify the linear dependence on past error terms.

**Key Characteristics of MA Models:**

1.  **Always Stationary (if coefficients are finite):** Unlike AR models, a pure MA process is *always* stationary, provided the MA coefficients ($\theta_i$) are finite. Its mean, variance, and autocovariance structure are constant over time.
2.  **ACF for Order Selection:** The **Autocorrelation Function (ACF)** plot is the primary tool for determining the order 'q' of an MA model. For a pure MA(q) process, the ACF will have **significant spikes only up to lag 'q'**, and then "cut off" (drop to non-significant levels) beyond that lag. This is because the series is directly dependent on only a finite number of past error terms.
3.  **PACF Behavior:** The **Partial Autocorrelation Function (PACF)** for an MA(q) process typically **decays gradually** (either exponentially or in a damped sinusoidal pattern). This is because the direct effect of a past error term propagates indirectly through subsequent observations.
4.  **Forecasting:** Forecasting with MA models involves using past actual error terms (which are estimated by residuals from the model) to predict future values.

---

## MA(1) vs. MA(2) Models

Let's compare the simplest MA models:

### MA(1) Model (First-Order Moving Average Model)

* **Formula:**
    $$Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1}$$
* **Interpretation:** The current value of the series ($Y_t$) depends on the current random shock ($\epsilon_t$) and the random shock from the previous period ($\epsilon_{t-1}$).
* **ACF/PACF:**
    * **ACF:** Has a significant spike at lag 1 and then cuts off to zero (or within the confidence bounds) for all subsequent lags.
    * **PACF:** Decays gradually.
* **Invertibility Condition:** For an MA(1) process to be invertible, **$|\theta_1| < 1$**. (More on invertibility below).

### MA(2) Model (Second-Order Moving Average Model)

* **Formula:**
    $$Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2}$$
* **Interpretation:** The current value ($Y_t$) depends on the current shock ($\epsilon_t$) and the shocks from the two most recent past periods ($\epsilon_{t-1}$ and $\epsilon_{t-2}$).
* **ACF/PACF:**
    * **ACF:** Has significant spikes at lags 1 and 2, and then cuts off to zero for all subsequent lags.
    * **PACF:** Decays gradually, potentially with a damped oscillatory pattern.
* **Invertibility Conditions:** More complex, similar to stationarity conditions for AR(2). The roots of the characteristic equation $1 + \theta_1 B + \theta_2 B^2 = 0$ must lie outside the unit circle. Specifically, for an MA(2) to be invertible:
    1.  $|\theta_2| < 1$
    2.  $\theta_1 + \theta_2 > -1$
    3.  $\theta_1 - \theta_2 < 1$

---

## Theoretical Properties of a Time Series with an AR(1), AR(2), MA(1), and MA(2)

Let's look at their theoretical ACF and PACF properties, which are crucial for model identification.

| Model Type | Theoretical ACF                                 | Theoretical PACF                                |
| :--------- | :---------------------------------------------- | :---------------------------------------------- |
| **AR(p)** | **Decays gradually** (exponentially or damped sine wave) | **Cuts off** after lag `p`                      |
| AR(1)      | Exponential decay ($|\phi_1|^k$)               | Spike at lag 1, then zero                       |
| AR(2)      | Damped exponential or sine wave decay           | Spikes at lags 1 & 2, then zero                 |
| **MA(q)** | **Cuts off** after lag `q`                      | **Decays gradually** (exponentially or damped sine wave) |
| MA(1)      | Spike at lag 1, then zero                       | Exponential decay                               |
| MA(2)      | Spikes at lags 1 & 2, then zero                 | Damped exponential or sine wave decay           |

**Note on "Zero" / "Cuts Off":** When we say "cuts off," it means the theoretical correlation is exactly zero beyond that lag. In practice, with sample data, it means the sample correlations fall within the confidence intervals (not statistically significant).

---

## ACF for General MA(q)

For a general MA(q) process:

$$Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

The theoretical ACF, $\rho_k$, will be:

* **Non-zero for $k \le q$**
* **Exactly zero for $k > q$**

This is the defining characteristic of an MA(q) process's ACF. The direct impact of the error terms only extends for 'q' periods. Beyond that, there's no direct linear relationship.

Specifically, for $k \le q$:

$$\rho_k = \frac{\theta_k + \theta_{k+1}\theta_1 + ... + \theta_q\theta_{q-k}}{\sum_{j=0}^{q} \theta_j^2}$$

(where $\theta_0 = 1$)

This formula shows that the ACF coefficients are a function of the MA coefficients ($\theta_j$).

---

## Invertibility of MA Models

Just as AR models have a stationarity condition, MA models have an **invertibility condition**.

* **What is Invertibility?**
    An MA model is said to be **invertible** if it can be rewritten as an **infinite order AR model**.
    * This means we can express $\epsilon_t$ as a function of current and past $Y$ values.
    * An invertible MA model ensures that the past random shocks ($\epsilon_t$) can be uniquely determined from current and past observations of the time series ($Y_t$).
    * This property is crucial for forecasting and for linking MA models to AR models, particularly when estimating parameters. If an MA model is not invertible, there might be multiple sets of MA coefficients that produce the same ACF, making model identification and estimation ambiguous.

* **Why is it Important?**
    1.  **Uniqueness of Solution:** Ensures a unique and stable relationship between the observed series and the underlying white noise process.
    2.  **Parameter Estimation:** Many estimation algorithms for MA and ARMA models rely on the invertibility assumption.
    3.  **Forecasting:** Allows the model to use an infinite number of past observations to estimate the error term, which is practical for forecasting.

* **Condition for Invertibility:**
    For an MA(q) process, the roots of its characteristic equation (or MA polynomial) must lie **outside the unit circle**.

    For an MA(1) model, $Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1}$, the characteristic equation is $1 + \theta_1 B = 0$. The root is $B = -1/\theta_1$. For invertibility, $|-1/\theta_1| > 1$, which simplifies to **$|\theta_1| < 1$**.

    For an MA(2) model, $Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2}$, the characteristic equation is $1 + \theta_1 B + \theta_2 B^2 = 0$. Both roots must lie outside the unit circle.

---

## Partial Autocorrelation Function

The formula you provided for PACF is conceptual and highlights its nature as a conditional correlation:

$$\frac{\text{Covariance}(y, x_3 | x_1, x_2)}{\sqrt{\text{Variance}(y | x_1, x_2) \text{Variance}(x_3 | x_1, x_2)}}$$

This expression means the correlation between $y$ and $x_3$ *given* (or after controlling for the linear effects of) $x_1$ and $x_2$.

In the context of time series, if we want the PACF at lag $k$, we are looking at the correlation between $Y_t$ and $Y_{t-k}$, after removing the linear effects of $Y_{t-1}, Y_{t-2}, ..., Y_{t-k+1}$.

**Think about the difference between interpreting regression models:**
* **Simple Correlation (like ACF at lag 1):** If you just regress $Y_t$ on $Y_{t-1}$, the coefficient (and correlation) tells you the direct relationship.
* **Partial Correlation (like PACF):** If you regress $Y_t$ on $Y_{t-1}$ and $Y_{t-2}$, the coefficient for $Y_{t-2}$ tells you the relationship between $Y_t$ and $Y_{t-2}$ *after accounting for* $Y_{t-1}$. This is precisely the idea behind PACF – isolating the direct effect.

### Some Useful Facts About PACF and ACF Patterns

* **Identification of an AR model is often best done with the PACF.**
    * **Why?** Because for an AR(p) process, the PACF directly reveals the order 'p' by cutting off exactly after 'p' lags. This clear cut-off makes 'p' easy to identify. The ACF for an AR process, by contrast, decays gradually, making the exact order less clear.

* **Identification of an MA model is often best done with the ACF.**
    * **Why?** Similarly, for an MA(q) process, the ACF directly reveals the order 'q' by cutting off exactly after 'q' lags. The PACF for an MA process, on the other hand, decays gradually, making the exact order less clear.

This complementary nature of ACF and PACF is what makes them indispensable tools for identifying the appropriate AR and MA components in an ARIMA model.

