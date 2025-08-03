
Spectral analysis is one of the most **intuitive yet often underexplained concepts in time series analysis**. Below is a **detailed, interview-ready explanation**, with both intuition and mathematical foundation.

---

# **1️⃣ What is Spectral Analysis?**

Spectral analysis is a method used to **analyze time series data in the frequency domain**, instead of the time domain.

* In the **time domain**, we model how values change over time (ARIMA, regression).
* In the **frequency domain**, we model how much different **cycles or periodic patterns (frequencies)** contribute to the time series.

📌 **Idea:**
Any complex time series can be thought of as a combination of:

* **Trend** (long-term, very low frequency wave)
* **Seasonality or cycles** (repeating patterns, mid-frequency waves)
* **Noise** (random high-frequency components)

Spectral analysis decomposes the series into these frequencies to:

* Detect hidden cycles
* Identify dominant periodic behaviors
* Improve forecasting

---

# **2️⃣ The Analogy**

Think of a song:

* A song is a combination of multiple musical notes (different frequencies).
* A time series is like a song—each cycle is like a note contributing to the final waveform.

Spectral analysis answers:

* *“Which notes (frequencies) are playing the loudest in this time series?”*

---

# **3️⃣ Mathematical Foundation**

Any stationary time series $x_t$ can be expressed as a sum of sinusoidal functions:

$$
x_t = \sum_{k=1}^{K} A_k \cos(2\pi f_k t + \phi_k) + \text{noise}
$$

Where:

* $A_k$: amplitude (strength) of frequency $f_k$
* $\phi_k$: phase (timing of the cycle)
* $f_k$: frequency (number of cycles per time unit)

The **spectral density function (SDF)** describes **how variance of the series is distributed over different frequencies**.

---

# **4️⃣ Periodogram (Core Tool in Spectral Analysis)**

The **periodogram** estimates the spectral density of a time series.

$$
I(f) = \frac{1}{N} \left| \sum_{t=1}^{N} x_t e^{-i 2\pi f t} \right|^2
$$

* $N$: number of observations
* $f$: frequency
* This is essentially the **Fourier Transform** of the series squared in magnitude.

📌 **Interpretation:**

* Plot $I(f)$ vs frequency $f$.
* **Peaks in the plot** indicate dominant periodicities in the data (e.g., a strong seasonal cycle every 12 months).

---

# **5️⃣ Intuition with an Example**

Suppose you record the **daily electricity consumption** over 3 years.

* Time domain plot: looks noisy with ups and downs.
* Spectral analysis (periodogram):

  * Shows a strong peak at **frequency = 1/7** (weekly cycle).
  * Shows a second peak at **frequency = 1/365** (annual seasonality).
  * Random noise contributes to high frequencies.

This tells you that:

* There's a strong repeating weekly pattern (weekends vs weekdays).
* There's also a yearly pattern (winter heating, summer cooling).

---

# **6️⃣ Connection with Autocorrelation (ACF)**

* ACF tells us how much the series relates to its past values.
* **Spectral density = Fourier transform of the autocovariance function.**

Meaning:

* If a time series has strong periodicity, it shows:

  * Oscillating ACF
  * A sharp peak in the spectrum at the periodic frequency

---

# **7️⃣ Practical Uses**

* Detect **hidden cycles** in noisy data (e.g., financial markets).
* Identify **dominant periodicities** for seasonal adjustment.
* Separate **signal vs noise** components.
* Used in **engineering (signal processing)** and **econometrics**.

---

# **8️⃣ R Example**

```R
data <- AirPassengers
spec.pgram(data, log = "no")
```

* The **periodogram plot** shows peaks for monthly cycles and yearly cycles.

---

# **9️⃣ Visual Intuition**

Think of time series as **waves overlapping**:

* Spectral analysis "untangles" these waves.
* Each peak in the spectrum is a frequency where the data has a lot of "energy" (variance).

---

# ✅ Key Interview Takeaways

* **Spectral analysis** shifts focus from time → frequency domain.
* Helps detect **dominant cycles** or **periodicities**.
* **Periodogram** is the main tool, based on **Fourier transform**.
* Strong link between **autocorrelation** and **spectral density**.
* Useful for **seasonality detection** and **filter design**.

---
Here’s a **detailed, intuitive, and interview-focused explanation of Fractional Differencing and Threshold Models**, with examples.

---

# **1️⃣ Fractional Differencing**

## **a) Concept**

* In ARIMA models, differencing (integer $d = 1,2,...$) is used to make a non-stationary series stationary:

  $$
  y_t' = y_t - y_{t-1}
  $$
* But sometimes the series is **not fully non-stationary**, meaning it has **long memory** (decay in autocorrelation is slow, not abrupt).

  * A **regular first difference** might over-difference it, removing too much information.
  * A **fractional difference** allows a *partial differencing* degree $d$ where:

    $$
    0 < d < 1
    $$
* This creates a **stationary series** but **preserves long-term dependencies**.

---

## **b) Mathematical Idea**

We can generalize the difference operator:

$$
(1 - L)^d y_t
$$

Where:

* $L$ is the lag operator $L y_t = y_{t-1}$
* $d$ can be fractional.

We expand this using a **binomial series**:

$$
(1 - L)^d = 1 - dL + \frac{d(d-1)}{2!}L^2 - \frac{d(d-1)(d-2)}{3!}L^3 + \dots
$$

This gives **fractional weights** to past observations.

So, the differenced series becomes:

$$
y_t' = y_t - d y_{t-1} + \frac{d(d-1)}{2} y_{t-2} - ...
$$

Instead of just subtracting the previous observation, we take a **weighted sum of many past lags**, allowing long memory to remain.

---

## **c) Intuition**

* Ordinary differencing ($d=1$) → kills long-term persistence completely.
* Fractional differencing ($d<1$) → removes just enough memory to make the process stationary but keeps meaningful long-run correlations.

Example:

* Financial volatility series often has long memory (decaying autocorrelations over many lags).
* Using fractional differencing, we preserve this feature, unlike simple differencing that might oversmooth it.

---

## **d) Example in R (ARFIMA model)**

```R
library(fracdiff)
data <- log(AirPassengers)
fit <- fracdiff(data, nar=1, nma=1)
fit$d
```

Output might show:

```
Fractional differencing parameter d = 0.35
```

This means the series needed **partial differencing (0.35)** to achieve stationarity.

---

# **2️⃣ Threshold Models**

## **a) Concept**

Threshold models capture **nonlinear dynamics** in time series where the behavior of the series **changes depending on its level or past value**.

Instead of one single linear model, we have:

* **Different regimes**, e.g., if $y_{t-d}$ crosses a threshold value $\gamma$, the process switches behavior.

📌 Example:

* Interest rates or stock returns may behave differently **above vs below a threshold**.
* A company’s sales may grow fast only **after reaching a critical advertising spend**.

---

## **b) Model Structure (TAR: Threshold Autoregressive Model)**

$$
y_t =
\begin{cases}
\phi_0 + \phi_1 y_{t-1} + \epsilon_t & \text{if } y_{t-d} \leq \gamma \\
\psi_0 + \psi_1 y_{t-1} + \epsilon_t & \text{if } y_{t-d} > \gamma
\end{cases}
$$

Where:

* $d$ = delay parameter (lag used for threshold variable)
* $\gamma$ = threshold value
* Different AR models apply in different regimes.

This is a **piecewise linear AR model**.

---

## **c) Smooth Transition Models (STAR)**

* Instead of a hard switch at threshold $\gamma$, we allow a **smooth transition**:

$$
y_t = \phi(L)y_{t-1}[1 - G(z_{t-d};\gamma,c)] + \psi(L)y_{t-1} G(z_{t-d};\gamma,c) + \epsilon_t
$$

* $G$ is a logistic function controlling transition smoothness.

---

## **d) Intuition**

* Threshold models allow **nonlinearities**:

  * Small shocks may have little effect when the system is below a threshold.
  * Large shocks or reaching a critical level may change system dynamics (new regime).

Example:

* Unemployment may react mildly to small economic shocks but rapidly once a crisis threshold is reached.

---

## **e) R Example**

```R
library(tsDyn)
model <- setar(AirPassengers, m=2, thDelay=1)
summary(model)
```

* Finds a threshold value ($\gamma$) and estimates separate AR models for each regime.

---

# **3️⃣ Key Interview Takeaways**

### Fractional Differencing

* Solves **long memory processes** (slowly decaying autocorrelation).
* Allows **partial differencing** ($0<d<1$) to achieve stationarity without losing information.
* Used in ARFIMA models.

### Threshold Models

* Time series can behave differently in different **regimes**.
* TAR models = abrupt switch; STAR models = smooth transition.
* Useful for **nonlinear, asymmetric, or regime-switching data** (e.g., financial markets, macroeconomics).

---

Would you like me to prepare a **"Side-by-Side 1-Page Cheat Sheet (Fractional Differencing vs Threshold Models)"** with formulas, intuition diagrams, and R code snippets for quick revision?


14  Review
Course Summary
This document outlines the key ideas of the course. We won’t give the details again. The purpose is to jar your memory a bit on what we’ve done and to make some connections between different elements of the course. The summary will be in a loose outline form, with some overlap between sections.

1. General Descriptive Tools for Univariate Time Series
Time series plot of the data – look for trends and seasonality (Lesson 1)
Sample ACF and PACF to identify possible ARIMA models (Lesson 2 to Lesson 4) and we’ll also outline important features in this document)
Plots of 
 versus lagged values of 
 for various lags
Smoothing to more clearly see trends and seasonality (Lesson 5)
Decomposition models (additive and multiplicative) for sorting out trend and seasonality and to estimate seasonally adjusted values (Lesson 5)
Periodogram and Spectral Density to identify dominant frequencies and periods/cycles in data (Lesson 6 to (Lesson 11)
2. ARIMA Model Building – (Lesson 1 to Lesson 4)
Basic structure of ARIMA Models
The basic structure is that a value at time 
 is a function of past values (AR terms) and/or or past errors (MA terms).

A stationary series is one for which the mean, variance, and autocovariance structure remain constant over time. In the presence of non-stationarity, use differencing to create a stationary series.

Identification of ARIMA models
Use the ACF and PACF together to identify possible models. The following table gives some rough guidelines. Unfortunately, it’s not a well-defined process and some guesswork/experimentation is usually needed.

Combined ACF and PACF Pattern		Possible Model
Sample ACF	Sample PACF	
Tapering or sinusoidal pattern that converges to 0, possibly alternating negative and positive signs	Significant values at the first p lags, then non-significant value	AR of order p
Significant values at the first q lags, then non-significant values	Tapering or sinusoidal pattern that converges to 0, possibly alternating negative and positive signs	MA of order q
Tapering or sinusoidal pattern that converges to 0, possibly alternating negative and positive signs	Tapering or sinusoidal pattern that converges to 0, possibly alternating negative and positive signs	ARMA with both AR and MA terms – identifying the order involves some guesswork
Seasonal Models (Lesson 4)
Seasonal models are used for data in which there are repeating patterns related to times of the year (months, quarters, etc.). Seasonal patterns are modeled with terms connected to seasonal periods – e.g., AR or MA terms at lags of 12, 24, etc, for monthly data with seasonal features.
It is often the case that seasonal data will require seasonal differencing – e.g., a 12 th difference for monthly data with a seasonal pattern.
To identify seasonal patterns using the sample ACF and PACF, look at the patterns through the seasonal jumps in lags – e.g., 12, 24, etc. for monthly data
Model Confirmation
For a good model, the sample ACF and PACF of the residuals should have non-significant values at all lags. Sometimes we may see a barely significant residual autocorrelation at an unusual lag. This usually can be ignored as a sampling error quirk. Clearly, significant values at important lags should not be ignored – they usually mean the model is not right.
Ideally, the Box-Ljung test for accumulated residual autocorrelation should be non-significant for all lags. This sometimes is hard to achieve. The test seems to be quite sensitive and there is a multiple inference issue. We’re doing a lot of tests when we look at all lags so about 1 in 20 may be significant even when all null hypotheses are true.
Use MSE, AIC and BIC to compare models. You may see two or more models with about the same characteristics. There can be redundancy between different ARIMA models.
Prediction and Forecasting with ARIMA models (Lesson 3)
Details given in Lesson 3 and software will do the work. The basic steps for substituting values into the equation are straightforward –

For AR type values, use known values when you can and forecasted values when necessary
For MA type values, use known values when you can and 0 when necessary
Exponential smoothing and related methods are sometimes used as more simple forecasting methods. (Lesson 5)

3. Variations of univariate ARIMA Models
We examined three variations of univariate ARIMA models:

ARCH and GARCH models used for volatile variance changes (Lesson 11)

Fractional differencing as an alternative to ordinary differencing (Lesson 13)

Threshold AR models which allow different AR coefficients for values above and below a defined threshold (Lesson 13)

4. Relationships Between Time Series Variables
Ordinary regression with AR errors (Lesson 8) – used when we have a dependent variable (y) and one or more predictors (x-variables), with all variables measured as time series.

We start with ordinary regression methods, then examine the AR structure of the residuals and use that structure to adjust the initial least squares regression estimates.
Lagged Regression for the relationship between a y-variable and an x-variable (Lesson 8 and Lesson 9)

In a lagged regression, we use lags of the x-variable and possibly lags of the y-variable to predict y. The cross-correlation function (CCF) is used to identify possible models.
Examining the CCF to Identify a Lagged Regression Model

The CCF of the original series may provide what you need for identifying the model, but in some instances, you may need to “pre-whiten” the y and x series in some way before looking at the CCF. Lesson 9

Pre-whitening steps might be one of the following –

Difference each series and the look at the CCF for the two differenced series
Determine an ARIMA model for x. Apply that model to both x and y, to get “residuals” for each series. Look at the CCF for the two residual series
When looking at the CCF –

Clear spikes at a lag indicate that lag of x may be useful for predicting y
A tapering or sinusoidal pattern emanating from a clear spike may indicate that a first lag (and/or second lag) of the y-variable may be helpful
Another tool that may be useful is to examine plots of y versus lagged values of x
VAR models (Lesson 11)

Vector autoregressive models are multivariate time series models. A VAR model defines a regression system of models in which each variable is a function of lags of itself and all other variables under consideration. These are useful for relationships between variables which are similar – e.g., rainfall statistics from several different locations or commodities prices at several different locations.

5. Comparison of Groups
Intervention Analysis (Lesson 9)

This is a before/after comparison. We examine how a time series is affected by a new law or procedure that is imposed at some point in time.

Repeated Measures/Longitudinal Analysis (Lesson 10)

Here, we measure a (usually) short time series on different experimental units, divided into two or more treatment groups. The objective is to compare the treatment groups with respect to how differing treatments affect the response variable over time.

6. Frequency Domain (Lesson 6 and Lesson 12)
The periodogram and spectral density consider time series in the frequency domain. The underlying structure represents a time series as a sum of cosine and sine waves of varying frequencies. We look for the dominant frequencies.
