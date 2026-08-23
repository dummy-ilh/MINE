# Stage 1 — Interview QA & Cheatsheet
### Module 1: Foundations · Stochastic Processes · ACF/PACF · Stationarity · Smoothing

---

## ⚡ CHEATSHEET — Brush up the morning of

### Definitions (one-liners)

| Term | Definition |
|---|---|
| **Time series** | Sequence of observations indexed by time, at regular intervals |
| **Weak stationarity** | Constant mean, constant variance, autocovariance depends only on lag $k$ (not on $t$) |
| **Strict stationarity** | Full joint distribution unchanged under time shift (stronger; rarely needed) |
| **White noise** | IID, mean 0, constant variance $\sigma^2$, zero autocorrelation at all lags $\neq 0$ |
| **Random walk** | $x_t = x_{t-1} + \varepsilon_t$ — non-stationary, variance grows with $t$ |
| **Unit root** | Root of AR characteristic polynomial = 1 → series is non-stationary |
| **Autocovariance** | $\gamma(k) = \text{Cov}(x_t, x_{t+k})$ |
| **ACF** | $\rho(k) = \gamma(k)/\gamma(0)$ — total correlation at lag $k$ |
| **PACF** | Correlation at lag $k$ after removing effect of all shorter lags |
| **Ergodicity** | Time averages → ensemble averages (lets us estimate from one long series) |
| **Trend** | Long-run direction (up/down) |
| **Seasonality** | Fixed, repeating pattern within a known period |
| **Remainder** | What's left after removing trend and seasonal |

---

### Key Formulas

| Formula | What it is |
|---|---|
| $\hat{T}_t = \frac{1}{m}\sum_{j=-k}^{k}x_{t+j}$ | Centered MA of order $m=2k+1$ |
| $\rho(k) = \gamma(k)/\gamma(0)$ | ACF at lag $k$ |
| $\pm 1.96/\sqrt{n}$ | 95% significance bands on ACF/PACF plot |
| $\nabla x_t = x_t - x_{t-1}$ | First difference |
| $\nabla^2 x_t = \nabla x_t - \nabla x_{t-1}$ | Second difference (removes quadratic trend) |
| $\hat{x}_{t+1} = \alpha x_t + (1-\alpha)\hat{x}_t$ | SES update |
| $\ell_t = \alpha x_t + (1-\alpha)(\ell_{t-1}+b_{t-1})$ | Holt level |
| $b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)b_{t-1}$ | Holt trend |
| $\hat{x}_{t+h} = \ell_t + hb_t$ | Holt $h$-step forecast |
| $\hat{x}_{t+h} = \ell_t + hb_t + s_{t-m+(h \bmod m)}$ | Holt-Winters forecast |

---

### ACF/PACF Signature Table

| Pattern | Model |
|---|---|
| ACF cuts off at lag $q$, PACF tails off | MA($q$) |
| PACF cuts off at lag $p$, ACF tails off | AR($p$) |
| Both tail off | ARMA($p$,$q$) |
| ACF decays very slowly (near 1 at many lags) | Non-stationary → difference first |
| Significant spikes at lags $m, 2m, 3m, \ldots$ | Seasonality of period $m$ |

---

### Tests

| Test | Null hypothesis | Reject means |
|---|---|---|
| **ADF** | Unit root exists (non-stationary) | Series IS stationary |
| **KPSS** | Series IS stationary | Series is non-stationary |
| **Ljung-Box** | Residuals are white noise | Residuals have structure left |

> Use ADF + KPSS together. If both agree → confident conclusion.

---

### ETS Codes

| Method | ETS |
|---|---|
| SES | (A, N, N) |
| Holt | (A, A, N) |
| Holt-Winters additive | (A, A, A) |
| Holt-Winters multiplicative | (A, A, M) |

---
---

## PART 1 — Phase 1: Foundations

---

**Q1. What are the four components of a time series?**

**Trend (T):** long-run direction.
**Seasonal (S):** repeating pattern tied to a fixed calendar period.
**Cyclical (C):** multi-year waves, period not fixed (business cycles).
**Irregular/Remainder (I):** what's left after removing the above.

Additive model: $x_t = T_t + S_t + C_t + I_t$
Multiplicative model: $x_t = T_t \times S_t \times C_t \times I_t$

> Use additive when swings are constant in size; multiplicative when swings grow with the level (log-transform converts multiplicative → additive).

---

**Q2. What's the difference between additive and multiplicative seasonality? How do you tell from a plot?**

- **Additive:** seasonal amplitude is the same regardless of trend level. Peaks/troughs stay the same height over time.
- **Multiplicative:** seasonal amplitude grows with the trend. The series has a "megaphone" or "fan" shape.

Quick test: plot the series. If the seasonal swings widen as values rise → multiplicative. Log-transform to linearize before fitting.

---

**Q3. *(Google-style)* You're given monthly sales data with a clear upward trend and seasonal peaks every December. Walk me through how you'd decompose it.**

1. Apply a 2×12 moving average (12 months, even period) to estimate trend $\hat{T}_t$.
2. Subtract: detrended = $x_t - \hat{T}_t$.
3. Average each month's detrended values across all years → seasonal indices $\hat{S}_t$.
4. Seasonally adjust: $x_t - \hat{S}_t$.
5. Remainder = $x_t - \hat{T}_t - \hat{S}_t$.

---

**Q4. Why use a 2×12 MA instead of a straight 12-point MA for monthly data?**

A 12-point window straddles two time points — there's no integer center. Averaging two adjacent 12-point MAs shifts the center to land exactly on a month. Algebraically this gives half-weight to the endpoints and full weight to the middle 11 — a proper symmetric, centered estimate.

---

**Q5. *(Apple, DS interview)* Why does a moving average of order $m$ remove seasonality when $m$ equals the seasonal period?**

Each window of width $m$ contains exactly one observation from each season. The seasonal highs and lows cancel in the average, leaving only the trend. This is the mathematical reason you must match window width to seasonal period.

---

## PART 2 — Phase 2: Stochastic Processes

---

**Q6. Define weak stationarity. Why does it matter?**

A series is weakly stationary if:
1. $E[x_t] = \mu$ (constant mean)
2. $\text{Var}(x_t) = \sigma^2$ (constant variance)
3. $\text{Cov}(x_t, x_{t+k}) = \gamma(k)$ depends only on lag $k$, not on $t$

It matters because almost every classical model (ARIMA, regression on time series, ACF-based analysis) assumes it. Non-stationary series produce spurious correlations and unreliable forecasts.

---

**Q7. What is white noise? Is it stationary?**

White noise $\{\varepsilon_t\}$: mean 0, variance $\sigma^2$, zero autocorrelation at all lags $\neq 0$. Each observation is independent of all others.

Yes — it is stationary (weakly and strictly). It's the "null model" for residuals. After fitting any model, residuals should look like white noise.

---

**Q8. *(Meta DS interview)* What is a random walk? Why is it non-stationary?**

$x_t = x_{t-1} + \varepsilon_t$

Variance: $\text{Var}(x_t) = t\sigma^2$ — grows with time. Mean may be fixed, but the exploding variance violates stationarity. You cannot forecast a random walk beyond the current level; the best forecast is always $\hat{x}_{t+h} = x_t$ (flat).

Fix: first-difference. $\nabla x_t = x_t - x_{t-1} = \varepsilon_t$ → white noise (stationary).

---

**Q9. What is the difference between a random walk and white noise?**

| | White Noise | Random Walk |
|---|---|---|
| Independence | Observations independent | Each depends on the last |
| Stationarity | Stationary | Non-stationary |
| Variance | Constant | Grows over time |
| Forecast | Mean | Current value |

---

**Q10. What is ergodicity and why does it matter practically?**

A process is ergodic if time averages converge to the true (ensemble) expectation. In practice, we have one realization of the series — not many parallel universes. Ergodicity is what justifies computing the sample mean and autocovariance from a single observed path and treating them as valid estimates.

---

**Q11. *(Amazon-style)* Explain the difference between strict and weak stationarity. When is weak sufficient?**

- **Strict:** entire joint distribution is time-shift invariant. Every moment is constant.
- **Weak:** only mean, variance, and autocovariance at each lag are constant.

Weak stationarity is sufficient for most classical models (ARMA, ARIMA, ACF analysis). Strict is needed only when working with higher moments or non-Gaussian distributions. A Gaussian process where weak stationarity holds is also strictly stationary.

---

## PART 3 — Phase 3: ACF & PACF

---

**Q12. *(Apple DS interview)* What is the difference between ACF and PACF?**

- **ACF at lag $k$:** total correlation between $x_t$ and $x_{t-k}$, including indirect paths through intermediate lags.
- **PACF at lag $k$:** correlation between $x_t$ and $x_{t-k}$ after removing the linear effect of all lags $1, 2, \ldots, k-1$.

Example: if $T_1 \to T_2 \to T_3$, ACF shows $T_1$ and $T_3$ as correlated. PACF at lag 2 removes the $T_2$ intermediary and shows whether $T_1$ and $T_3$ are *directly* related.

---

**Q13. How do you use ACF and PACF to identify model order?**

| Observation | Model |
|---|---|
| ACF cuts off after lag $q$, PACF tails off | MA($q$) |
| PACF cuts off after lag $p$, ACF tails off | AR($p$) |
| Both tail off gradually | ARMA($p$,$q$) |

"Cuts off" = drops inside the $\pm 1.96/\sqrt{n}$ bands abruptly.
"Tails off" = decays slowly/exponentially, stays significant for many lags.

---

**Q14. *(Google-style)* ACF is showing slow decay even after many lags. What does that mean?**

The series is likely **non-stationary**. Slow ACF decay (near 1 at many lags) is the classic signature of a unit root or trend. Always test stationarity (ADF/KPSS) and difference the series before reading ACF/PACF for model order.

---

**Q15. What are the significance bands on an ACF plot, and what do they mean?**

Under the null of no autocorrelation, at any lag $k$ the sample ACF is approximately $N(0, 1/n)$. The 95% bands are $\pm 1.96/\sqrt{n}$. Spikes outside these bands indicate statistically significant autocorrelation at that lag.

Practical caveat: with many lags plotted, some will cross purely by chance (~5% under the null). Look for clear patterns, not isolated borderline spikes.

---

**Q16. You see significant ACF spikes at lags 12, 24, 36 in monthly data. What does that tell you?**

Strong seasonality with period $m = 12$ (annual). The series correlates with itself one year ago, two years ago, etc. This signals you need a seasonal model (SARIMA, Holt-Winters, or STL decomposition) — not just a plain ARIMA.

---

## PART 4 — Phase 4: Stationarity

---

**Q17. *(Meta, commonly asked)* What is the ADF test? What is its null hypothesis?**

ADF (Augmented Dickey-Fuller) tests for a **unit root**.

- $H_0$: unit root exists → series is **non-stationary**
- $H_1$: no unit root → series is **stationary**

Reject $H_0$ (p < 0.05) → stationary. Fail to reject → evidence of non-stationarity, consider differencing.

Caveat: ADF has low power against highly persistent but technically stationary series — may fail to reject even when it should.

---

**Q18. What is KPSS? How does it differ from ADF?**

KPSS tests the **opposite null**:
- $H_0$: series IS stationary
- $H_1$: series has a unit root (non-stationary)

| | ADF | KPSS |
|---|---|---|
| Null | Non-stationary | Stationary |
| Reject → | Stationary | Non-stationary |

Use both. If ADF rejects AND KPSS doesn't reject → confident stationarity. If both reject → strong evidence of non-stationarity.

---

**Q19. *(Google interview)* A series has a trend. What's the difference between differencing and detrending?**

- **Differencing** ($x_t - x_{t-1}$): removes stochastic trends (unit roots). No assumption on trend shape. The right fix for random walks.
- **Detrending** (subtract a fitted trend line): removes deterministic trends. Assumes trend follows a known function (linear, polynomial).

Use ADF/KPSS to decide. If the series has a unit root → difference. If it has a fixed deterministic trend → detrend. Wrong choice leads to over- or under-differencing.

---

**Q20. What happens if you over-difference a series?**

You induce unnecessary MA structure. An already-stationary series differenced once will have $\rho(1) \approx -0.5$ — artificial negative autocorrelation introduced by the differencing. The ACF will look like an MA(1), even though the original series didn't need it. Check with KPSS before differencing.

---

**Q21. When would you apply a log transformation before modeling?**

When the series is multiplicative — seasonal or irregular variance grows with the level. Log converts multiplicative structure to additive:
$$\log(x_t) = \log(T_t) + \log(S_t) + \log(I_t)$$
Also stabilizes variance (deals with heteroscedasticity). Always plot the series first; if the swings fan out over time, log-transform.

---

**Q22. *(Amazon-style)* What is integration of order $d$, i.e., $I(d)$?**

A series is $I(d)$ if it requires $d$ differences to become stationary. Most economic/financial series are $I(1)$ — one difference makes them stationary. $I(0)$ = already stationary. $I(2)$ is rare; it means even the first differences have a trend.

---

## PART 5 — Phase 5: Smoothing & Decomposition

---

**Q23. *(Meta DS)* What is exponential smoothing? How is it different from a simple moving average?**

**SES:** $\hat{x}_{t+1} = \alpha x_t + (1-\alpha)\hat{x}_t$

Weights all past data, decaying exponentially with age. Nothing is fully discarded.

**SMA:** equal weight to the last $m$ observations, zero weight to everything older. Hard cutoff.

Key difference: SES gives more weight to recent observations; SMA treats all points in the window equally. SES is better for series where recent data is more informative. SMA is simpler to explain.

---

**Q24. What does the smoothing parameter $\alpha$ control in SES?**

- $\alpha$ close to 1 → forecast reacts sharply to the latest data (responsive, noisy)
- $\alpha$ close to 0 → forecast barely moves (smooth, slow to adapt)

Chosen by minimizing in-sample sum of squared forecast errors (numerical optimization). Analogous to learning rate in ML.

---

**Q25. *(Google-style)* Why can SES only produce a flat forecast?**

SES maintains only a **single smoothed level**. There is no trend or slope component. Every future period gets the same number: $\hat{x}_{t+1} = \hat{x}_{t+2} = \cdots = \ell_t$. It "chases" the current level but has no mechanism to project forward along a slope.

Fix: Holt's method adds a separately smoothed trend component.

---

**Q26. Explain Holt's method. When would you use it over SES?**

Holt tracks two components:
- **Level:** $\ell_t = \alpha x_t + (1-\alpha)(\ell_{t-1}+b_{t-1})$
- **Trend:** $b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)b_{t-1}$
- **Forecast:** $\hat{x}_{t+h} = \ell_t + hb_t$

Use Holt when the series has a trend but no seasonality. SES would systematically lag behind a trending series. Holt's forecast is a straight-line projection from the current level at the current slope.

---

**Q27. *(Apple-style)* What are the three smoothing parameters in Holt-Winters and what does each control?**

| Parameter | Controls |
|---|---|
| $\alpha$ | How fast the **level** adapts to new observations |
| $\beta^*$ | How fast the **trend** adapts to new slope evidence |
| $\gamma$ | How fast the **seasonal pattern** evolves over time |

All three are between 0 and 1. Higher value → faster adaptation, more reactive. All three are estimated by minimizing in-sample forecast errors.

---

**Q28. When do you use additive vs. multiplicative Holt-Winters?**

- **Additive:** seasonal swings are constant in absolute size. Use when the amplitude of the seasonal pattern doesn't change as the series grows.
- **Multiplicative:** seasonal swings scale with the level. Use when the series fans out (megaphone shape). Multiplicative seasonal equation: $s_t = \gamma(x_t/\ell_t) + (1-\gamma)s_{t-m}$.

Practical shortcut: plot the series. Constant-height peaks → additive. Growing peaks → multiplicative (or log-transform and use additive).

---

**Q29. What is STL decomposition? What are its advantages over classical MA decomposition?**

STL (Seasonal-Trend decomposition using Loess) iterates between Loess-smoothing the trend and Loess-smoothing each season's subseries until convergence.

Advantages:
1. **Evolving seasonality** — the seasonal pattern can gradually shift over time; classical MA forces it constant.
2. **Robustness** — outliers can be down-weighted so they don't distort trend/seasonal estimates.
3. **Any seasonal period** — not restricted to $m = 4$ or $m = 12$.

Limitation: additive only in base form. Log-transform first for multiplicative data.

---

**Q30. *(Meta/Amazon-style)* What is the ETS framework?**

ETS = **E**rror · **T**rend · **S**easonal. Each component is **N** (none), **A** (additive), or **M** (multiplicative). The framework provides:
- A unified taxonomy of all exponential smoothing methods
- Formal state-space representation → proper likelihood, AIC/BIC for model selection, prediction intervals

Example: ETS(A,A,M) = additive errors, additive trend, multiplicative seasonality = Holt-Winters multiplicative.

---

## PART 6 — Cross-cutting / Harder Questions

---

**Q31. *(Google, harder)* What is the Yule-Walker equation conceptually, and what is it used for?**

For an AR($p$) process, the Yule-Walker equations express the autocorrelations $\rho(1), \ldots, \rho(p)$ as a linear system in the AR coefficients $\phi_1, \ldots, \phi_p$:

$$\rho(k) = \phi_1\rho(k-1) + \phi_2\rho(k-2) + \cdots + \phi_p\rho(k-p), \quad k \geq 1$$

Used to: (a) estimate AR coefficients from sample ACF, (b) compute the PACF (solving this system of increasing size is exactly what gives PACF values), (c) derive the ACF signature of AR processes analytically.

---

**Q32. *(Apple)* How do you handle missing values in a time series before modeling?**

Options depend on the gap size and data structure:
- **Forward fill / backward fill:** good for short gaps where the series is slow-moving.
- **Linear interpolation:** fill in a straight line between known values.
- **Seasonal interpolation:** use the same period last cycle to estimate the missing value.
- **Model-based imputation:** fit a model to the non-missing data and impute.

Never drop rows — it breaks the regular time spacing that all classical models assume.

---

**Q33. *(Amazon-style)* If your forecasting model residuals still have autocorrelation, what does that mean and what do you do?**

It means the model has not captured all the structure in the data — there is signal left in the residuals that should have been modeled. Diagnose with ACF of residuals and the Ljung-Box test.

Fixes:
- Increase AR or MA order
- Add a seasonal component (SARIMA / seasonal dummies)
- Check if differencing was sufficient
- Consider a non-linear model

Good residuals should look like white noise: ACF near zero at all lags, Ljung-Box p > 0.05.

---

**Q34. *(Meta)* A time series shows seasonality that seems to be getting stronger over time. What model family would you use?**

Multiplicative seasonality (Holt-Winters multiplicative / ETS(A,A,M)) or log-transform + additive Holt-Winters. STL with a wider seasonal window to allow slow evolution is also appropriate. Classical additive decomposition would underfit because it assumes the seasonal amplitude is fixed.

---

**Q35. *(Google)* What's the difference between trend and cycle in time series decomposition?**

- **Trend:** smooth, long-run direction of the series. Duration: the full length of the data or longer.
- **Cycle:** medium-term wave with a period longer than one year but not fixed (unlike seasonality). Corresponds to business cycles, credit cycles, etc. Period varies.

In practice, cycle and trend are often estimated together as "trend-cycle" because they're hard to separate without very long data. STL and classical decomposition typically produce a combined trend-cycle component.

---

**Q36. *(Numerical — commonly given in DS phone screens)* Last period's forecast was 70. Demand was 60. $\alpha = 0.4$. What is the next SES forecast?**

$$\hat{x}_{t+1} = 0.4 \times 60 + 0.6 \times 70 = 24 + 42 = \mathbf{66}$$

---

**Q37. *(Numerical)* 3-month moving average: October = 100, November = 200, December = 300. What is the MA forecast for January?**

$$\hat{x}_{\text{Jan}} = \frac{100 + 200 + 300}{3} = \mathbf{200}$$

---

**Q38. *(Numerical)* AR(1): $x_t = 0.4 + 0.2x_{t-1} + u_t$, $u_t \sim WN(0,1)$. Find the unconditional mean and variance.**

**Mean:** $\mu = \frac{0.4}{1-0.2} = \frac{0.4}{0.8} = \mathbf{0.5}$

**Variance:** $\text{Var}(x_t) = \frac{\sigma^2}{1-\phi^2} = \frac{1}{1-0.04} = \frac{1}{0.96} \approx \mathbf{1.042}$

---

## Quick Interview Tactics

- **Stationarity question always comes first.** Before any model, say: "I'd test for stationarity with ADF and KPSS, and difference if needed."
- **ACF/PACF question:** memorize the cut-off table (Q13). It will be asked verbatim.
- **Smoothing vs. ARIMA:** exponential smoothing = no need to check stationarity, fast to fit, great baseline. ARIMA = more formal, better for inference, needs stationarity.
- **Residual check:** always mention Ljung-Box test + ACF of residuals when describing model validation.
- **Additive vs. multiplicative:** always say "plot the series first."
