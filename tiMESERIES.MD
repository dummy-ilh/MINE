Here's a breakdown of time series questions commonly asked in FAANG interviews (mostly for Data Science, Applied Scientist, and ML Engineer roles), organized by category:

## 1. Conceptual / Statistical Foundations

- **Stationarity**: What is it? How do you test for it (ADF test, KPSS test)? Why does it matter for ARIMA-type models?
- **Autocorrelation vs Partial Autocorrelation**: How do you read ACF/PACF plots to choose AR/MA orders?
- **Trend vs Seasonality vs Cyclic vs Residual**: How do you decompose a series (additive vs multiplicative)?
- **White noise vs random walk**: How do you distinguish them?
- **Differencing**: Why and when do you difference a series? What's over-differencing?

## 2. Modeling Questions

- Explain ARIMA — what do p, d, q mean? How do you select them (AIC/BIC, grid search, auto_arima)?
- When would you use SARIMA vs ARIMA vs Prophet vs exponential smoothing (Holt-Winters)?
- Compare classical statistical models (ARIMA) vs ML models (XGBoost/LightGBM on lag features) vs deep learning (LSTM, Temporal Fusion Transformer) for forecasting — tradeoffs in data size, interpretability, and accuracy.
- How do you handle multiple seasonalities (daily + weekly + yearly), e.g., in demand forecasting?
- How would you forecast a metric with missing data or irregular timestamps?

## 3. Practical / Applied (very common at FAANG)

- "Forecast daily active users for the next 30 days — walk me through your approach." (Meta, Amazon)
- "How would you detect anomalies in a real-time metrics dashboard?" (Google, Netflix)
- "Design a system to detect a sudden drop in revenue/traffic — is it real or noise?" (Amazon, Uber-style but common across FAANG)
- "How would you forecast demand for a new product with no historical data?" (cold-start problem)
- "How do you choose the right train/validation split for time series?" (rolling-window / walk-forward validation, why you can't shuffle randomly)
- "What metrics would you use to evaluate a forecast?" (MAPE, RMSE, MAE, SMAPE — and pitfalls of each, e.g., MAPE blowing up near zero)

## 4. Anomaly Detection Specific

- How do you build a real-time anomaly detection system on streaming metrics?
- Statistical approaches (z-score, control charts) vs ML approaches (isolation forest, seasonal-hybrid ESD) vs forecasting-residual-based detection
- How do you handle alert fatigue / false positives in production monitoring?

## 5. System Design Flavor (for senior roles)

- "Design a real-time fraud detection pipeline using transaction time series data."
- "How would you architect a forecasting pipeline that retrains daily and serves predictions at scale?"
- Questions about handling concept drift / seasonality shifts in production models.

## 6. SQL/Coding on Time Series Data

- Compute rolling averages, lag/lead values, week-over-week or month-over-month growth using window functions.
- Detect consecutive-day streaks (e.g., user active 7 days in a row).
- Gaps-and-islands problems (find continuous date ranges).

---

Good — here's what's commonly missed in the first pass, especially relevant to Google/Apple DS & MLE interviews:

## 7. Feature Engineering for Time Series (heavily asked in MLE rounds)

- Lag features, rolling mean/std/min/max, expanding window stats
- Fourier/sine-cosine features for seasonality instead of one-hot calendar features
- Holiday/event encoding, lead/lag of holidays (pre-holiday spike effects)
- How do you avoid **data leakage** when creating lag/rolling features (e.g., using future data accidentally in a rolling window)?
- Handling time zones / daylight saving time shifts in global metrics (very Apple/Google-specific — global product metrics)

## 8. Causal Inference + Time Series (Google loves this)

- **Interrupted time series analysis** — did a product launch/feature change actually shift the metric, or is it noise/seasonality?
- **CausalImpact / Bayesian structural time series** — used heavily at Google for measuring impact of launches without a clean A/B test
- How do you do A/B testing when the metric has strong seasonality or weekly cycles?
- Difference-in-differences with time series data
- Synthetic control method — building a "counterfactual" series

## 9. Multivariate Time Series

- Granger causality — does series A help predict series B?
- VAR (Vector Autoregression) models
- Cross-correlation function — lead/lag relationships between two metrics (e.g., does search volume lead sales?)
- Multi-task/multi-series forecasting — forecasting thousands of SKUs/products at once (hierarchical forecasting, reconciliation between SKU-level and category-level forecasts) — **very common in Amazon/Apple supply chain-style questions**

## 10. Changepoint & Drift Detection

- How do you detect a structural break/changepoint in a series (CUSUM, Bayesian changepoint detection, PELT algorithm)?
- Concept drift — how do you know when to retrain a deployed forecasting model?
- Distinguishing a permanent level shift from a temporary anomaly

## 11. Uncertainty & Probabilistic Forecasting

- How do you generate prediction intervals, not just point forecasts? (quantile regression, conformal prediction, Monte Carlo dropout for DL models, Bayesian methods)
- Why point forecasts alone are dangerous for business decisions (e.g., inventory planning)
- Calibration of prediction intervals — are your 90% intervals actually covering 90% of the time?

## 12. Backtesting Pitfalls (very common follow-up after "how do you validate")

- Why a single train/test split can be misleading — need rolling-origin/walk-forward backtesting
- Look-ahead bias — accidentally using future information (common in feature engineering, normalization done globally instead of per-window)
- How sample size at long horizons shrinks and inflates variance of error estimates
- How to backtest when there are structural breaks (e.g., COVID-period data) in history

## 13. Case-Study Style (Apple/Google specific framing)

- "App Store daily downloads dropped 15% — is it real, seasonal, or a tracking bug? Walk me through diagnosis." (classic Apple-style)
- "How would you forecast iPhone sales ahead of a new launch, given cannibalization effects from the previous model?"
- "Search query volume for a topic spiked — forecast next 7 days." (Google-style)
- "YouTube/Google Ads revenue forecasting with multiple seasonalities (daily, weekly, holiday) — how do you architect this at scale?"

## 14. State-Space / Advanced Models (asked more at Senior/Staff level)

- Kalman filters — intuition and when you'd use them over ARIMA
- Structural time series models (trend + seasonal + regression components, e.g., what powers Google's `bsts` package)
- Dynamic linear models

---

Here's a structured breakdown of **STAT 510 Lessons 1–5** with major topics and likely interview-style Q&A for each. This course (Penn State, Shumway & Stoffer-based) is a goldmine for FAANG DS/MLE time series rounds since it builds the theoretical backbone behind ARIMA/SARIMA.

---

## Lesson 1: Time Series Basics

**


---

## Lesson 2: MA Models, PACF, Notation

**Major topics:**
- Moving Average model, MA(1) and MA(q): $x_t = \mu + w_t + \theta_1 w_{t-1} + \dots$
- MA(q) ACF cuts off sharply after lag q (key diagnostic difference from AR)
- Partial Autocorrelation Function (PACF) — correlation between $x_t$ and $x_{t-k}$ after removing the effect of the intermediate lags
- AR(p) → PACF cuts off after lag p; ACF tails off
- MA(q) → ACF cuts off after lag q; PACF tails off
- ARMA(p,q) → both ACF and PACF tail off
- Backshift operator notation $B$, e.g., $(1-\phi B)x_t = w_t$
- Invertibility (for MA models) and stationarity (for AR models) conditions

**Likely Q&A:**
- *Q: How do you tell AR vs MA apart just from ACF/PACF plots?*
  A: AR(p): PACF has a sharp cutoff after lag p, ACF tails off gradually. MA(q): ACF has a sharp cutoff after lag q, PACF tails off gradually. ARMA: both tail off.
- *Q: What does "invertibility" mean for an MA model, and why does it matter?*
  A: An MA process is invertible if it can be rewritten as a convergent AR($\infty$) process. It matters because invertibility ensures a unique, identifiable correspondence between the model and its ACF — without it, multiple parameter sets could give identical autocorrelation structure.
- *Q: Why use backshift notation B?*
  A: It compactly expresses differencing and ARMA structure as polynomial operations, e.g., $(1-B)x_t$ is a first difference — this makes manipulating seasonal/non-seasonal terms (multiplying polynomials) much cleaner, as seen in SARIMA.

---

## Lesson 3: Identifying & Estimating ARIMA Models; Forecasting

**Major topics:**
- ARIMA(p,d,q): combining AR, differencing (d), and MA components
- Why and how to choose `d` — examine trend, run ADF/KPSS, or look at ACF decaying very slowly (a sign of non-stationarity needing differencing)
- Over-differencing pitfalls (introduces unnecessary negative autocorrelation, MA structure)
- Model identification process: plot → difference → ACF/PACF → propose candidate models
- Estimation: typically via Maximum Likelihood (e.g., `sarima()` / `arima()` in R)
- Diagnostics: residual ACF, Ljung-Box / Box-Pierce test for residual whiteness, normal Q-Q plot of residuals
- Model selection via AIC/AICc/BIC
- Forecasting: point forecasts + prediction intervals; forecasts converge to the mean as horizon increases for stationary models

**Likely Q&A:**
- *Q: How do you decide if you need to difference the data?*
  A: Look at the time series plot for trend, check the ACF for a slow, almost linear decay over many lags (a hallmark of non-stationarity), and run a formal test like ADF (null = unit root present) or KPSS (null = stationary).
- *Q: What's the risk of over-differencing?*
  A: It can introduce artificial negative autocorrelation and unnecessary MA terms, inflate variance of the differenced series, and make the model harder to interpret/forecast accurately.
- *Q: How do you choose between two candidate ARIMA models with similar AIC?*
  A: Prefer the more parsimonious model, check that all estimated coefficients are statistically significant, and verify residual diagnostics (Ljung-Box test for autocorrelation, near-Gaussian residuals) are clean for both.
- *Q: Why do ARIMA point forecasts flatten out over long horizons?*
  A: Because the underlying process being modeled (after differencing) is stationary, so forecasts revert toward the unconditional mean as the influence of recent observations decays with the autoregressive structure.

---

## Lesson 4: Seasonal Models (covered in detail above, recap)

**Major topics:**
- Seasonal ARIMA notation: ARIMA$(p,d,q)\times(P,D,Q)_S$
- Seasonal differencing $(1-B^S)$ vs non-seasonal differencing $(1-B)$; sometimes need both
- Multiplicative seasonal model: $\Phi(B^S)\phi(B)(x_t-\mu)=\Theta(B^S)\theta(B)w_t$
- Reading seasonal ACF/PACF: examine early lags for non-seasonal terms, examine lags at multiples of S for seasonal terms
- The "surprise" extra significant lag (e.g., lag 11 in a $(0,0,1)\times(0,0,1)_{12}$ model) from cross-terms when multiplying seasonal × non-seasonal MA polynomials
- Worked examples: Colorado River flow (monthly) and Australian beer production (quarterly)
- Handling non-constant variance in residuals (transformations, or defer to ARCH/GARCH later)

**Likely Q&A:**
- *Q: For monthly data with both trend and seasonality, what differencing would you try first?*
  A: Apply a seasonal (lag-12) difference first, then re-examine for remaining trend; if trend persists, apply a first (non-seasonal) difference on top of that.
- *Q: In a $(0,0,1)\times(0,0,1)_{12}$ model, why is there a significant ACF spike at lag 11, not just at 1, 12, 13?*
  A: When you multiply out the seasonal and non-seasonal MA polynomials, a cross term ($\theta_1 \Theta_1 w_{t-13}$ correlating with $w_{t-12}$ from the lag-11-apart pairing) produces a non-zero covariance at lag 11 — it's a multiplicative side-effect, not an additional "real" MA term.
- *Q: How would you handle non-constant variance discovered in residuals-vs-fits plots after fitting a SARIMA model?*
  A: Options include ignoring it if mild, applying a variance-stabilizing transform (log, sqrt) before re-fitting, or moving to a conditional heteroskedasticity model like ARCH/GARCH if volatility clustering is the real driver.

---

## Lesson 5: Smoothing & Decomposition + More ARIMA Practice

**Major topics:**
- Classical decomposition: additive vs multiplicative (Trend + Seasonal + Residual)
- Moving average smoothing for trend estimation
- Exponential smoothing (simple, Holt's linear trend, Holt-Winters seasonal)
- STL decomposition (Seasonal-Trend decomposition using Loess) — flexibility advantages over classical decomposition
- More practice fitting/diagnosing ARIMA and SARIMA models end-to-end on real datasets
- Comparing decomposition-based forecasting vs ARIMA-based forecasting

**Likely Q&A:**
- *Q: When would you choose an additive vs multiplicative decomposition?*
  A: Additive when seasonal swings are roughly constant in absolute magnitude regardless of the trend level; multiplicative when seasonal swings scale proportionally with the trend (e.g., percentage-based seasonal effects that grow as the series grows).
- *Q: What's the advantage of STL over classical decomposition?*
  A: STL allows the seasonal component to change over time (not fixed/rigid), handles outliers more robustly, and works for any periodicity, whereas classical decomposition assumes a fixed, repeating seasonal pattern.
- *Q: How does Holt-Winters differ from ARIMA in approach?*
  A: Holt-Winters directly smooths/extrapolates level, trend, and seasonal components via exponential weighting of past data, while ARIMA models the autocorrelation structure of the (differenced) series explicitly — Holt-Winters is often simpler and more interpretable, ARIMA is more flexible for complex autocorrelation patterns once correctly specified.
- *Q: If you only have weekly data with both a slow trend and a fixed yearly cycle, which approach handles that more naturally — SARIMA or STL/Holt-Winters?*
  A: This is genuinely a judgment call interviewers want you to reason through — STL/Holt-Winters often handles long, smooth seasonal cycles more gracefully and intuitively, while SARIMA can struggle with very long seasonal periods (S=52) since it requires many seasonal AR/MA parameters; many practitioners instead use Fourier-term regression or ML-based lag features for long seasonal cycles.

---

Want me to do the same breakdown for **Lessons 6–10** (Periodogram, regression with ARIMA errors, cross-correlation, intervention analysis, longitudinal models) — these come up a lot in Apple/Google causal-inference-style interview questions?
