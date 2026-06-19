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

If you want, I can pick 2–3 of these (e.g., CausalImpact-style launch analysis, or the "App Store downloads dropped" diagnosis question) and walk through a full mock-interview answer structure for each.
