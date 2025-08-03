Here’s a detailed explanation of the **missing concepts from Lesson 9 (STAT 510)**, including **Pre-whitening (CCF interpretation)** and **Intervention Analysis**, structured for interview preparation.

---

# **📌 Lesson 9 – Pre-whitening and Intervention Analysis**

---

## **1️⃣ Pre-whitening and Cross-Correlation Function (CCF)**

### **Problem with Raw CCF**

When you compute the CCF between two time series $X_t$ and $Y_t$:

$$
r_{XY}(k) = \frac{\text{Cov}(X_{t-k}, Y_t)}{\sqrt{\text{Var}(X_{t-k})\text{Var}(Y_t)}}
$$

you may see **high correlations that are spurious**, caused by:

* Strong autocorrelation within $X_t$ or $Y_t$.
* Shared long-term trends or seasonality.
* Random persistence unrelated to causal relationship.

---

### **Goal of Pre-whitening**

To reveal the **true lead-lag relationship** between $X$ and $Y$ by:

1. **Removing autocorrelation** from $X_t$.
2. Applying **the same filter** to $Y_t$.
3. Calculating the CCF on the filtered (pre-whitened) series.

This ensures observed cross-correlations represent actual transfer of information from $X$ to $Y$.

---

### **Steps for Pre-whitening**

1. Fit an ARIMA model to $X_t$:

   $$
   \phi(B)(1-B)^d X_t = \theta(B) a_t
   $$
2. Filter both series using:

   $$
   X^*_t = \frac{\phi(B)(1-B)^d}{\theta(B)} X_t
   $$

   $$
   Y^*_t = \frac{\phi(B)(1-B)^d}{\theta(B)} Y_t
   $$
3. Compute:

   $$
   \text{CCF}(k) = Corr(X^*_{t-k}, Y^*_t)
   $$
4. Identify significant lags (bars outside $\pm 2/\sqrt{N}$).

---

### **Interpretation of CCF Patterns**

* **Positive lag $k>0$:** $X_t$ leads $Y_{t+k}$ (predictive relationship).
* **Negative lag $k<0$:** $Y_t$ leads $X_{t+k}$.
* **Lag 0:** Instantaneous effect.

**Patterns:**

* **Single spike:** Simple delay effect.
* **Geometric decay:** Gradual effect, like an impulse response.
* **Symmetric peaks:** Common cyclical relationship.

---

### **Example in R**

```r
modX <- arima(X, order=c(1,0,0))
Xf <- residuals(modX)
Yf <- filter(Y, filter=c(1,-modX$coef[1]), sides=1)
ccf(Xf, Yf, main="Pre-whitened CCF")
```

---

## **2️⃣ Intervention Analysis**

Intervention analysis models the effect of an **external event (intervention)** on a time series.

### **Goal**

Quantify **how a sudden or gradual intervention changes the mean level or pattern** of a time series.

Example:

* Policy change (new tax law)
* Marketing campaign launch
* Machinery upgrade
* COVID-19 lockdown effect on sales

---

### **General Model**

$$
Y_t = \text{Baseline ARIMA model} + \omega \cdot I_t + N_t
$$

Where:

* $I_t$: **Intervention function**
* $\omega$: Effect size
* $N_t$: ARIMA noise process

---

### **Types of Intervention Functions**

1. **Step function:** Permanent level shift

   $$
   I_t =
   \begin{cases}
   0 & t < T \\
   1 & t \geq T
   \end{cases}
   $$

   Example: New tax permanently changes sales level.

---

2. **Pulse function:** One-time shock

   $$
   I_t =
   \begin{cases}
   0 & t \neq T \\
   1 & t = T
   \end{cases}
   $$

   Example: Power outage disrupts production for 1 day.

---

3. **Ramp function:** Gradual change

   $$
   I_t =
   \begin{cases}
   0 & t < T \\
   t - T & t \geq T
   \end{cases}
   $$

   Example: Slow adoption of a new policy.

---

### **Dynamic Effects**

Sometimes, the intervention effect does not fully manifest immediately but **builds over time**:

$$
Y_t = \frac{\omega B^b}{1-\delta B} I_t + N_t
$$

* $b$: **Delay** before effect starts
* $\delta$: **Decay or persistence** factor (0 < δ < 1)

---

### **Steps in Intervention Analysis**

1. Identify intervention time $T$.
2. Build baseline ARIMA model for $Y_t$.
3. Define intervention variable ($I_t$).
4. Fit:

   $$
   Y_t = \text{ARIMA noise} + \text{Transfer function for } I_t
   $$
5. Test if $\omega$ is statistically significant.
6. Forecast post-intervention effects.

---

### **Example in R**

```r
# Define intervention variable (step)
intervention <- ifelse(time(Y) >= 2020, 1, 0)

# Fit ARIMAX model with intervention
fit <- arima(Y, order=c(1,0,0), xreg=intervention)
summary(fit)
```

---

# ✅ Summary for Interview

| Concept                   | Key Idea                                                            |
| ------------------------- | ------------------------------------------------------------------- |
| **Pre-whitening**         | Removes autocorrelation from $X,Y$ to reveal true lead-lag relation |
| **CCF interpretation**    | Positive lag = X leads Y; significance bounds = ±2/√N               |
| **Intervention analysis** | Measures effect of external event on time series                    |
| Types of intervention     | Step (permanent), Pulse (one-time), Ramp (gradual)                  |
| Dynamic intervention      | Includes delay and decay in effect                                  |

---

Let's break down the concepts of Prewhitening and Intervention Analysis, which are crucial techniques for robust time series analysis, especially when dealing with multiple series or external events.

-----

## 9.1 Prewhitening as an Aid to Interpreting the CCF

As we discussed earlier, the **Cross-Correlation Function (CCF)** is used to identify lead-lag relationships between two time series, say $X\_t$ and $Y\_t$. However, a major problem arises when $X\_t$ or $Y\_t$ (or both) are autocorrelated (i.e., they are not white noise).

**The Problem of Spurious Correlation:**
If $X\_t$ is autocorrelated, then $X\_t$ is correlated with $X\_{t-1}$, $X\_{t-2}$, and so on. If $Y\_t$ is also autocorrelated (and often it will be), then the observed CCF between $X\_t$ and $Y\_t$ can be misleading. A significant cross-correlation at a certain lag might not be due to a direct relationship between $X$ and $Y$ at that lag, but rather due to the internal autocorrelation *within* $X$ or $Y$, or a common underlying unobserved factor driving both. This leads to **spurious correlations**, where the CCF shows many significant spikes that don't represent a true lead-lag relationship.

**When and How to Prewhiten:**

  * **When:** You should **always prewhiten** your time series when you want to interpret the CCF to identify genuine lead-lag relationships between two series. The goal of prewhitening is to remove the internal autocorrelation structure from the input series, so that any remaining cross-correlation is genuinely due to the relationship *between* $X\_t$ and $Y\_t$, not just their individual temporal patterns.
  * **How:** The process of prewhitening involves the following steps:
    1.  **Model the Input Series ($X\_t$):** Fit a suitable ARIMA model to the *input* series $X\_t$. The goal is to transform $X\_t$ into a white noise series. Let this ARIMA model be denoted as $\\Phi\_X(B) X\_t = \\Theta\_X(B) \\alpha\_t$, where $\\alpha\_t$ is the white noise residual series from $X\_t$.
    2.  **Apply the Same Filter to the Output Series ($Y\_t$):** Apply the *exact same ARIMA filter* (using the $\\Phi\_X(B)$ and $\\Theta\_X(B)$ operators identified for $X\_t$) to the *output* series $Y\_t$. This transforms $Y\_t$ into a new series, say $\\beta\_t$.
          * Specifically: $\\Phi\_X(B) Y\_t = \\Theta\_X(B) \\beta\_t$.
    3.  **Compute CCF of Residuals:** Compute the CCF between the **prewhitened series** $\\alpha\_t$ (the residuals from the $X$ model) and $\\beta\_t$ (the filtered $Y$ series).
          * The significant spikes in the CCF($\\alpha\_t, \\beta\_t$) now genuinely indicate the lead-lag relationships where $\\alpha\_t$ leads $\\beta\_t$.
          * A significant spike at a positive lag $k$ in CCF($\\alpha\_t, \\beta\_t$) suggests that $\\alpha\_t$ (and thus the innovations in $X\_t$) affects $\\beta\_{t+k}$ (and thus $Y\_{t+k}$) at lag $k$. This directly helps identify which lags of $X$ predict $Y$.

**CCF Patterns (after Prewhitening):**
After prewhitening, the interpretation of CCF spikes becomes much cleaner:

  * **Significant spike at lag 0 only:** Indicates a contemporaneous relationship between $X\_t$ and $Y\_t$.
  * **Significant spike(s) at positive lags (e.g., $k=1, 2, \\dots$):** Indicates that $X\_t$ *leads* $Y\_t$. Changes in $X$ precede changes in $Y$ by $k$ time units. These are the lags that should be considered for inclusion in a lagged regression (transfer function model) where $X$ predicts $Y$.
  * **Significant spike(s) at negative lags (e.g., $k=-1, -2, \\dots$):** Indicates that $Y\_t$ *leads* $X\_t$. Changes in $Y$ precede changes in $X$ by $|k|$ time units.
  * **No significant spikes:** Suggests no linear relationship between $X\_t$ and $Y\_t$ that can be effectively captured through lagging.

**R Code (Conceptual for Prewhitening):**

```r
# Assuming X and Y are time series objects (ts or xts)
library(forecast)

# Step 1: Model the input series X
arima_x_model <- auto.arima(X) # Or specify order=c(p,d,q) if known
alpha_t <- residuals(arima_x_model) # These are the white noise residuals of X

# Step 2: Apply the SAME filter to Y
# Get the ARIMA coefficients from arima_x_model
phi_x <- arima_x_model$model$phi
theta_x <- arima_x_model$model$theta
d_x <- arima_x_model$arma[6] # differencing order

# Apply the filter to Y. The 'filter' function works differently depending on AR/MA
# A more robust way with 'forecast' package is to use 'Arima' to apply the filter:
# Create a dummy model for Y with the same filter as X but no actual parameters estimated
# This essentially filters Y with the characteristics of X's ARIMA model
beta_t <- Arima(Y, model = arima_x_model, include.mean = FALSE)$residuals

# If you prefer manual filtering using 'filter' base R function:
# beta_t = filter(Y, filter=phi_x, method="recursive", sides=1) # For AR part
# beta_t = filter(beta_t, filter=theta_x, method="recursive", sides=1) # For MA part
# (Need to handle differencing manually as well if d_x > 0)
# This manual filtering can be tricky and 'Arima(Y, model = arima_x_model)' is safer.

# Step 3: Compute CCF of residuals
ccf(alpha_t, beta_t, main="Prewhitened CCF(alpha_t, beta_t)")
```

## 9.2 Intervention Analysis

**Concept:** Intervention analysis is a specialized technique used in time series to assess the impact of a specific, **exogenous event or intervention** on the level or behavior of a time series. This event is typically external to the ongoing time series process (e.g., a policy change, a natural disaster, a marketing campaign, a new law, a strike).

**Why not just include as a dummy variable in OLS?**
Traditional OLS with a dummy variable might work for simple, instantaneous shifts, but it fails to account for:

  * The **time series structure** (autocorrelation) of the series *before* and *after* the intervention.
  * The **dynamic nature** of the intervention's effect (e.g., gradual onset, temporary impact, permanent shift).

**The Model:**
Intervention analysis typically extends an ARIMA model (which captures the intrinsic dynamics of the time series) by adding terms that represent the intervention effect.

The general form of an intervention model is:
$Y\_t = \\text{Intervention Effect}\_t + N\_t$
Where $N\_t$ is an ARIMA process (the "noise" or "background" series *without* the intervention).

**Identifying and Interpreting Various Patterns for Intervention Effects:**

Intervention effects can manifest in different ways:

1.  **Step Change (Permanent Shift):**

      * **Description:** The series experiences a sudden, permanent shift in its level. The effect is immediate and lasts indefinitely.
      * **Indicator Variable ($I\_t$):** A dummy variable where $I\_t = 0$ before the intervention point ($T\_0$) and $I\_t = 1$ from $T\_0$ onwards.
      * **Model Term:** $\\omega\_0 I\_t$
      * **Interpretation:** $\\omega\_0$ represents the immediate and permanent change in the mean level of the series.

2.  **Pulse (Temporary Spike/Dip):**

      * **Description:** The series experiences a sudden, temporary impact at a specific point in time, and then returns to its previous level.
      * **Indicator Variable ($P\_t$):** A dummy variable where $P\_t = 1$ at the intervention point ($T\_0$) and $P\_t = 0$ otherwise.
      * **Model Term:** $\\omega\_0 P\_t$
      * **Interpretation:** $\\omega\_0$ represents the temporary deviation from the mean at time $T\_0$.

3.  **Gradual (Ramp) Change:**

      * **Description:** The series gradually shifts to a new, permanent level over several periods, rather than instantaneously.
      * **Indicator Variable ($R\_t$):** $R\_t = 0$ before $T\_0$, $R\_t = 1$ at $T\_0$, $R\_t = 2$ at $T\_0+1$, and so on. Or, it can be modeled as a delayed step function (see below).
      * **Model Term:** $\\frac{\\omega\_0}{1-\\delta B} I\_t$ (where $I\_t$ is a step variable, and $\\delta$ controls the rate of gradual change).
      * **Interpretation:** The effect takes time to fully manifest. $\\omega\_0$ represents the total magnitude of the shift, and $\\delta$ (between 0 and 1) controls how quickly the effect builds up (e.g., smaller $\\delta$ means faster ramp).

4.  **Decaying Effect:**

      * **Description:** The series experiences an immediate impact that then gradually decays back to the original level over time.
      * **Model Term:** $\\frac{\\omega\_0}{1-\\delta B} P\_t$ (where $P\_t$ is a pulse variable).
      * **Interpretation:** $\\omega\_0$ is the initial impact, and $\\delta$ (between 0 and 1) controls the rate of decay.

5.  **Seasonal Pulse:**

      * **Description:** A recurring temporary effect at a specific season after the intervention (e.g., a policy affecting only summer sales for several years).
      * **Model Term:** This would involve a seasonal pulse variable (1 at specific seasonal lags post-intervention, 0 otherwise) or combinations of terms.

**Estimation of an Intervention Effect:**

The process typically involves:

1.  **Identify the ARIMA Model for the Pre-Intervention Series:** Fit an ARIMA model to the time series data *before* the intervention occurred. This captures the inherent dynamics of the series.
2.  **Define Intervention Variables:** Create appropriate dummy variables (step, pulse, etc.) for the intervention(s).
3.  **Estimate the Combined Model:** Fit an ARIMA model to the *entire* series (including the post-intervention period), incorporating the intervention variables as exogenous regressors (similar to `xreg` in R's `Arima` function).
      * The software will estimate the ARIMA parameters and the $\\omega$ coefficients for the intervention terms simultaneously.
4.  **Interpret Coefficients:** The $\\omega$ coefficients directly estimate the magnitude and pattern of the intervention's effect. Their significance indicates whether the intervention had a statistically discernible impact.
5.  **Diagnostic Checking:** Examine the residuals of the final model (after including interventions). They should ideally be white noise, indicating that both the intrinsic dynamics and the intervention effects have been adequately modeled.

**R Code (Conceptual for Intervention Analysis):**

```r
library(forecast)

# Assuming 'my_ts' is your time series
# Assuming 'intervention_start_time' is the exact time point of intervention

# 1. Define intervention variables
# Example: A simple step change
# Create a dummy variable for a step change at a specific point
# For a ts object:
step_variable <- rep(0, length(my_ts))
# Find the index where the intervention starts
intervention_index <- which(time(my_ts) >= intervention_start_time)[1]
if (!is.na(intervention_index)) {
  step_variable[intervention_index:length(my_ts)] <- 1
} else {
  warning("Intervention start time not found in time series.")
}

# Example: A simple pulse at a specific point
pulse_variable <- rep(0, length(my_ts))
pulse_variable[intervention_index] <- 1


# 2. Fit an ARIMA model to the series, including intervention variables as xreg
# Assuming you've already identified an appropriate ARIMA order (p,d,q) for the background noise
# For a step change:
intervention_model_step <- Arima(my_ts, order=c(p,d,q), xreg=step_variable)
summary(intervention_model_step) # Look at the coefficient for step_variable (omega_0)

# For a pulse:
intervention_model_pulse <- Arima(my_ts, order=c(p,d,q), xreg=pulse_variable)
summary(intervention_model_pulse) # Look at the coefficient for pulse_variable (omega_0)

# For more complex effects (gradual, decaying), you might need to construct
# the xreg variables more intricately, potentially using filter() or specific packages
# like 'tsinterventions' or 'lmfor' if available.
```

Intervention analysis is a powerful technique for attributing changes in a time series to specific external events, providing both statistical evidence and quantitative estimates of their impact.
