
## What is a Time Series?


Formally, a **time series** is a sequence of data points indexed (or listed) in time order. Most commonly, these data points are collected at successive, equally spaced points in time. The defining characteristic is that *time* is a crucial variable, and the order of observations matters significantly.

Here's why it's special:
* **Temporal Dependency:** The value of a data point at a given time is often dependent on its past values. For example, today's stock price is highly influenced by yesterday's.
* **Chronological Order:** The sequence of observations is paramount. Shuffling the data points would destroy the inherent patterns and relationships.

**Examples:**
* Daily stock prices
* Monthly unemployment rates
* Quarterly GDP figures
* Hourly sensor readings from a machine
* Annual rainfall data

---

## How Does Time Series Vary from Regression?

This is a common point of confusion, so let's clarify. While both time series analysis and regression analysis deal with relationships between variables and prediction, their core approaches and assumptions differ significantly.

| Feature            | Time Series Analysis                                                              | Regression Analysis                                                               |
| :----------------- | :-------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Primary Goal** | Forecasting future values of a variable based on its own past values and patterns over time. | Identifying relationships between a dependent variable and one or more independent variables. |
| **Role of Time** | **Explicitly central.** Time is the independent variable, and the order of observations is critical. | Often **not explicitly central.** Observations are generally assumed to be independent of each other (unless specific regression models for time-dependent data are used). |
| **Data Structure** | Sequential data, where observations are ordered chronologically. Exhibits temporal dependencies like trends, seasonality, and autocorrelation. | Cross-sectional data or panel data (though panel data has a time component, the focus is often on individual entities over time, not the time sequence itself). Observations are often assumed to be independent. |
| **Key Assumption** | Values at different time points are *not* independent; they are often correlated (autocorrelation). | Standard regression assumes independence of observations (though this can be relaxed in some advanced regression techniques). |
| **Typical Models** | ARIMA, SARIMA, Exponential Smoothing, State Space Models, Prophet, Recurrent Neural Networks (RNNs). | Linear Regression, Logistic Regression, Polynomial Regression, Support Vector Regression, Decision Trees, etc. |
| **"X" Variable** | Often the "X" variable is simply time, or lagged values of the target variable itself. | Typically, there are multiple "X" variables (features/predictors) that are distinct from the "Y" variable. |
| **Prediction** | Often termed "forecasting" – predicting *future* values of the same variable. | Often termed "prediction" or "estimation" – predicting a "Y" value given specific "X" values, which could be in the present or future. |

**Think of it this way:**

* **Time Series:** "What will the sales be *tomorrow* based on *past sales patterns*?"
* **Regression:** "What will the sales be if we increase our advertising spend by X amount, regardless of when it happens?" (Though you *can* use time as an independent variable in regression, it's not the primary characteristic that defines it as "time series analysis.")

While you *can* use regression techniques on time series data (e.g., using past values as predictors in a regression model, known as autoregression), true time series analysis incorporates the unique characteristics of temporal dependence.

---

## What are the Important Features on a Time Series Plot?

When you visualize a time series, typically as a line plot with time on the x-axis and the observed value on the y-axis, several key features tend to stand out. Identifying these features is crucial for understanding the underlying process and choosing appropriate forecasting models.

1.  **Trend:**
    * **Definition:** The long-term direction or overall tendency of the data. It indicates whether the series is generally increasing, decreasing, or remaining relatively stable over time.
    * **Appearance on Plot:** A sustained upward slope (positive trend), a sustained downward slope (negative trend), or a relatively flat line (no significant trend).
    * **Example:** A steadily increasing population over several decades, or a gradual decline in traditional newspaper sales.

2.  **Seasonality:**
    * **Definition:** A regular, predictable pattern of fluctuations that recurs at fixed intervals within a specific period (e.g., a year, a week, a day). These patterns are often driven by calendar-related factors.
    * **Appearance on Plot:** Repeating peaks and troughs at consistent intervals. The length of the cycle is fixed.
    * **Example:** Higher retail sales during the holiday season each year, increased ice cream sales in summer, or daily traffic patterns peaking during rush hour.

3.  **Cyclical Component:**
    * **Definition:** Fluctuations that are not of fixed period but occur in cycles of varying lengths. These are often associated with economic or business cycles, and their duration is typically longer than seasonal patterns (e.g., several years).
    * **Appearance on Plot:** Waves or oscillations in the data that are not as regular or predictable as seasonality. They don't have a strict, fixed period.
    * **Example:** Boom and bust cycles in the economy, or fluctuations in commodity prices due to supply and demand over several years.

4.  **Irregular/Residual Component (Noise):**
    * **Definition:** The random, unpredictable fluctuations in the data that remain after accounting for trend, seasonality, and cyclical components. It represents the "noise" or unexplained variation in the series.
    * **Appearance on Plot:** Erratic, seemingly random movements after the systematic patterns have been considered.
    * **Example:** Sudden, unexpected spikes or dips in stock prices due to unforeseen news events, or daily weather variations not explained by seasonal patterns.

5.  **Outliers:**
    * **Definition:** Individual data points that deviate significantly from the general pattern of the series.
    * **Appearance on Plot:** Isolated points that are much higher or lower than their surrounding observations.
    * **Example:** An unusually high surge in website traffic on a particular day due to a viral social media post, or a sudden drop in production due to a machine breakdown.

6.  **Structural Change (Change Point):**
    * **Definition:** A sudden and permanent shift in the underlying behavior of the time series, affecting its mean, variance, or other statistical properties.
    * **Appearance on Plot:** A clear break or change in the trend, level, or variability of the series.
    * **Example:** A new government policy leading to a sustained shift in inflation rates, or a company merger causing a permanent change in sales trends.



# Important Characteristics to Consider First

Some important questions to first consider when first looking at a time series are:

- **Trend:** Is there a trend, meaning that, on average, the measurements tend to increase (or decrease) over time?
- **Seasonality:** Is there a regularly repeating pattern of highs and lows related to calendar time such as seasons, quarters, months, days of the week, and so on?
- **Outliers:** Are there outliers?  
  - In regression, outliers are far away from your line.  
  - With time series data, outliers are far away from your other data.
- **Long-run cycle:** Is there a long-run cycle or period unrelated to seasonality factors?
- **Variance:** Is there constant variance over time, or is the variance non-constant?
- **Abrupt changes:** Are there any abrupt changes to either the level of the series or the variance?

---

## Example 1.1

The following plot is a time series plot of the annual number of earthquakes in the world with seismic magnitude over 7.0, for 99 consecutive years.  
By a **time series plot**, we simply mean that the variable is plotted against time.

![Time series plot of quakes](https://online.stat.psu.edu/stat510/Lesson01_files/figure-html/fig-Timeseriesplotofquakes-1.png "Fig 1.1: Time series plot of quakes")

### Features of the plot:

- There is no consistent trend (upward or downward) over the entire time span.  
  The series appears to slowly wander up and down. The horizontal line drawn at quakes = 20.2 indicates the mean of the series. Notice that the series tends to stay on the same side of the mean (above or below) for a while and then wanders to the other side.

- Almost by definition, there is no seasonality as the data are annual data.

- There are no obvious outliers.

- It’s difficult to judge whether the variance is constant or not because the series meanders up and down, though the data seem to stay within a general band around the overall meandering. There are a couple of places where the variance spikes beyond the general band.

---

## AR(1) Model Introduction

One of the simplest ARIMA-type models is a model in which we use a linear model to predict the value at the present time using the value at the previous time. This is called an **AR(1) model**, standing for *autoregressive model of order 1*.  

The **order** of the model indicates how many previous time steps we use to predict the present time.

A starting point in evaluating whether an AR(1) might work is to plot values of the series against **lag-1 values** of the series.  

- Let \( Y_t \) denote the value of the series at any particular time \( t \).
- Let \( Y_{t-1} \) denote the value of the series one time before time \( t \).  
- That is, \( Y_{t-1} \) is the **lag-1 value** of \( Y_t \).

---

### Example Data (First 5 observations)

| t   | \( Y_t \) | Lag 1 Value \( Y_{t-1} \) |
|-----|-----------|--------------------------|
| 1   | 13        | *                        |
| 2   | 14        | 13                       |
| 3   | 8         | 14                       |
| 4   | 10        | 8                        |
| 5   | 16        | 10                       |

---

For the complete earthquake dataset, here’s a plot of \( Y_t \) versus \( Y_{t-1} \):

![Scatter plot showing quakes vs lag1](https://online.stat.psu.edu/stat510/Lesson01_files/figure-html/fig-Quakesvslag1scatterplot-1.png "Fig 1.2: Quakes vs lag1 scatterplot")

Although it’s only a moderately strong relationship, there is a positive linear association, so an AR(1) model **might be a useful model**.


## Seasonality vs. Cyclicity

This is a frequently confused pair, but understanding the difference is key to correctly analyzing your data. While both refer to patterns of rises and falls, their characteristics are distinct:

| Feature           | Seasonality                                                                   | Cyclicity                                                                   |
| :---------------- | :---------------------------------------------------------------------------- | :-------------------------------------------------------------------------- |
| **Definition** | A regular, predictable pattern of fluctuations that recurs at **fixed intervals** within a specific period (e.g., year, month, week, day). | Rises and falls that are **not of a fixed period** and usually last longer than seasonal patterns. |
| **Period Length** | **Fixed and known period** (e.g., exactly 12 months for annual seasonality). | **Variable length**; the duration of the cycle is not precisely known beforehand. |
| **Cause** | Driven by **calendar-related factors** (e.g., seasons, holidays, work schedules). | Often associated with **economic or business cycles**, or other long-term phenomena. |
| **Predictability** | **Highly predictable** in terms of timing of peaks and troughs.               | **Less predictable** in terms of exact timing and duration.                 |
| **Magnitude** | Tends to have a **more consistent magnitude**.                                | Magnitude can be **more variable**.                                         |
| **Example** | Increased retail sales every December, higher electricity consumption in summer, daily rush hour traffic peaks. | Economic recessions and expansions, fashion trends that last several years. |

**In essence:** If you can mark it on a calendar and it happens consistently year after year (or month after month, etc.), it's seasonality. If it's a wave-like pattern but the "waves" are of different lengths and less predictable, it's cyclicity.

---

## Univariate vs. Multivariate Time Series

This distinction depends on the number of variables you are observing over time.

* **Univariate Time Series:**
    * **Definition:** A time series that consists of **a single variable** measured over time.
    * **Focus:** Understanding and forecasting the future values of that *one* variable based solely on its own past values and inherent patterns.
    * **Data Structure:** A sequence of values, e.g., $[y_1, y_2, ..., y_n]$.
    * **Examples:** Daily stock prices of a *single* company, monthly sales of *one* product, hourly temperature readings from *one* sensor.
    * **Models:** ARIMA, Exponential Smoothing models (e.g., ETS).

* **Multivariate Time Series:**
    * **Definition:** A time series that involves **two or more variables** measured over the same time intervals.
    * **Focus:** Analyzing the relationships and interdependencies between these multiple variables *over time*, and using that information to forecast one or more of them.
    * **Data Structure:** A matrix or 2D array where each row contains multiple measurements at a given timestamp, e.g., $[ (y_{1a}, y_{1b}), (y_{2a}, y_{2b}), ..., (y_{na}, y_{nb}) ]$.
    * **Examples:** Daily stock prices of *multiple* companies, hourly temperature, humidity, and wind speed readings from a weather station, monthly GDP, inflation, and unemployment rates.
    * **Models:** Vector Autoregression (VAR), Vector Autoregressive Moving Average (VARMA), LSTMs (Long Short-Term Memory networks) that can handle multiple input features.

**Why the distinction matters:** If the variables influence each other over time, a multivariate approach can capture these complex relationships, potentially leading to more accurate forecasts than modeling each series independently. However, multivariate models are often more complex and require more data and computational resources.

---

## Types of "Time Domain" Models

When we talk about "time domain" models, we are contrasting them with "frequency domain" models. Time domain models analyze the series directly in terms of its sequence of observations over time, whereas frequency domain models analyze the series based on its underlying periodic components (using tools like Fourier transforms).

Within time domain models, two basic types are widely used, often combined:

1.  **Autoregressive (AR) Models:**
    * **Concept:** These models predict the future value of a variable based on a **linear combination of its own past values**. The "auto" refers to the fact that it's a regression of the variable on itself.
    * **Intuition:** "What happened yesterday (and the day before, etc.) is a good predictor of what will happen today."
    * **Notation:** AR(p), where 'p' is the order of the autoregressive model, indicating the number of past observations used.
    * **Mathematical Form:** $Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$
        * $Y_t$: value at time t
        * $c$: constant
        * $\phi_i$: autoregressive coefficients
        * $\epsilon_t$: white noise error term

2.  **Moving Average (MA) Models:**
    * **Concept:** These models predict the future value of a variable based on a **linear combination of past forecast errors (or "shocks")**.
    * **Intuition:** "If we made a mistake in our prediction yesterday, that mistake (or a series of past mistakes) gives us information to correct today's prediction."
    * **Notation:** MA(q), where 'q' is the order of the moving average model, indicating the number of past error terms used.
    * **Mathematical Form:** $Y_t = c + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q} + \epsilon_t$
        * $Y_t$: value at time t
        * $c$: constant
        * $\theta_i$: moving average coefficients
        * $\epsilon_t$: current white noise error term
        * $\epsilon_{t-i}$: past white noise error terms

**Combinations of these:**
* **ARMA (Autoregressive Moving Average) Models:** Combine both AR and MA components. Used for stationary time series.
* **ARIMA (Autoregressive Integrated Moving Average) Models:** Extend ARMA by including an "Integrated" (I) component, which involves differencing the data to make it stationary. This is crucial for handling trends.
* **SARIMA (Seasonal ARIMA) Models:** Further extend ARIMA to account for seasonality.
* **ARIMAX/SARIMAX:** Include "exogenous" variables (X), which are external predictors that might influence the time series.

---

## Residual Analysis

Once you've built a time series model, how do you know if it's any good? This is where **residual analysis** comes in.

* **What are Residuals?**
    * Residuals are simply the **differences between the actual observed values and the values predicted by your model** for the same time points.
    * $e_t = Y_t - \hat{Y}_t$
        * $e_t$: residual at time t
        * $Y_t$: actual value at time t
        * $\hat{Y}_t$: predicted value at time t

* **Why is Residual Analysis Important?**
    * **Model Validation:** The primary goal of residual analysis is to check if your model has successfully captured all the systematic information in the time series.
    * **Assumptions Check:** Many time series models assume that the errors (residuals) are "white noise."
        * **White Noise:** A series of random, independent, identically distributed values with a mean of zero and constant variance.
    * **Identifying Misspecification:** If residuals are *not* white noise, it means your model is missing some important patterns or relationships in the data. This indicates that the model is "misspecified" and can be improved.

* **What to Look for in Residuals (and how):**

    1.  **No Pattern (Randomness):**
        * **Plot:** Plot the residuals against time. They should appear randomly scattered around zero, with no discernible trends, seasonality, or cyclical patterns.
        * **ACF/PACF Plots:** The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots of the residuals should show no significant spikes (i.e., most values within the confidence intervals), indicating no remaining autocorrelation. This is the most crucial check.
        * **Statistical Tests:**
            * **Ljung-Box Test:** A common statistical test to check for overall autocorrelation in the residuals up to a certain number of lags. The null hypothesis is that there is no autocorrelation. If the p-value is small (e.g., < 0.05), you reject the null, meaning there's significant autocorrelation, and your model is inadequate.

    2.  **Constant Variance (Homoscedasticity):**
        * **Plot:** The spread of the residuals should be roughly constant across the entire time series. You shouldn't see the variance increasing or decreasing over time (heteroscedasticity).

    3.  **Normally Distributed (Optional but helpful for inference):**
        * **Histogram/QQ Plot:** Residuals should ideally follow a normal distribution, though this assumption is less critical for forecasting than for statistical inference.
        * **Shapiro-Wilk Test:** Tests for normality of residuals.

**In summary:** If your model's residuals look like white noise, you can be more confident that your model has captured the underlying data-generating process effectively. If not, you need to go back, diagnose the patterns in the residuals (e.g., if there's residual seasonality, add a seasonal component to your model), and refine your model.
