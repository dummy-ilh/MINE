# Time Series: Zero to Hero — Master Syllabus
### For Google/Apple/FAANG-level Interview Mastery
*Synthesized from STAT 510 (Penn State), Andrew Ng's ML pedagogy style, Rob Hyndman's "Forecasting: Principles and Practice" (FPP3), Box-Jenkins methodology, Shumway & Stoffer (Time Series Analysis and Its Applications), Hamilton's "Time Series Analysis," and modern DL forecasting literature.*

---

## How This Curriculum Is Structured
Each phase below will (in the full build-out) get: **intuition → formal math derivation → worked numerical example by hand → code implementation → common interview questions → pitfalls/nuances**. This document is the **syllabus only** — the map before the territory. Confirm scope/order, then I build out phases one at a time (each phase will be long and detailed, so we go module by module).

---

## PHASE 0 — Prerequisites Refresher (Foundation Check)
0.1 Probability: random variables, expectation, variance, covariance, conditional expectation
0.2 Joint/marginal/conditional distributions; independence vs uncorrelated
0.3 Linear algebra: eigenvalues/eigenvectors, matrix inversion, quadratic forms (needed for VAR, state space)
0.4 Statistical inference: MLE, method of moments, hypothesis testing, confidence intervals
0.5 Regression refresher: OLS, residuals, R², multicollinearity (bridge to time series regression)
0.6 Basic calculus: differentiation for optimization (MLE of AR/MA parameters)

---

## PHASE 1 — What Is a Time Series? Foundations
1.1 Definition, examples (finance, ops, sensor/IoT, demand forecasting — Apple/Google interview flavor: search query volume, ad revenue, app usage)
1.2 Time series vs cross-sectional vs panel data
1.3 Components of a time series: trend, seasonality, cyclicality, irregular/noise
1.4 Additive vs multiplicative decomposition — formula derivation
1.5 Time series plot diagnostics: what to look for visually
1.6 Sampling frequency, aggregation, resampling issues
1.7 Missing data & irregular timestamps — imputation strategies
1.8 Interview trap questions: "Is this even a time series problem?"

## PHASE 2 — Stochastic Processes Foundations
2.1 Formal definition of a stochastic process
2.2 Strict stationarity vs weak/covariance (2nd order) stationarity — full formula treatment
2.3 White noise process: definition, properties, why it's the "null model"
2.4 Random walk: derivation, variance grows with t, non-stationarity proof
2.5 Random walk with drift
2.6 IID sequences vs martingale difference sequences
2.7 Ergodicity — why it matters for estimation from a single realization

## PHASE 3 — Autocorrelation Theory (The Heart of Time Series)
3.1 Autocovariance function γ(k) — full derivation
3.2 Autocorrelation function (ACF) ρ(k) — formula, properties, bounds
3.3 Partial autocorrelation function (PACF) — Yule-Walker derivation
3.4 Correlogram interpretation — pattern recognition for AR vs MA vs ARMA
3.5 Sample ACF/PACF vs theoretical — bias, confidence bands (±1.96/√n)
3.6 Ljung-Box Q-statistic — full derivation and use
3.7 Numerical worked example: compute ACF by hand for a small dataset

## PHASE 4 — Stationarity: Testing & Transformation
4.1 Why stationarity matters (theoretical justification for model validity)
4.2 Differencing — first order, seasonal differencing, over-differencing symptoms
4.3 Unit root concept — intuition and formal definition
4.4 Augmented Dickey-Fuller (ADF) test — full derivation of test statistic, hypotheses, critical values
4.5 KPSS test — contrast with ADF (opposite null hypothesis) — why use both
4.6 Phillips-Perron test
4.7 Variance stabilization: log, Box-Cox transform — formula and lambda estimation
4.8 Detrending vs differencing — when to use which (stochastic vs deterministic trend)
4.9 Interview nuance: trend-stationary vs difference-stationary processes

## PHASE 5 — Classical Decomposition & Smoothing
5.1 Moving averages — simple, centered, weighted — formula derivation
5.2 Classical decomposition algorithm (additive/multiplicative) step-by-step
5.3 STL decomposition (Seasonal-Trend decomposition using Loess) — how Loess works, parameters
5.4 X-11/X-13-ARIMA-SEATS (industry-standard, used at Census Bureau/Google-scale ops) — conceptual overview
5.5 Exponential smoothing family:
  - Simple Exponential Smoothing (SES) — recursive formula, optimal α derivation
  - Holt's linear trend method — level + trend equations
  - Holt-Winters seasonal method (additive & multiplicative) — full 3-equation system
5.6 State-space formulation of exponential smoothing (ETS framework) — Hyndman's taxonomy (Error/Trend/Seasonal)
5.7 Numerical example: hand-compute SES forecasts for 5-10 points with given α

## PHASE 6 — The Box-Jenkins Methodology (AR, MA, ARMA, ARIMA)
6.1 Autoregressive AR(p) models — formula, characteristic equation, stationarity condition (roots outside unit circle)
6.2 Moving Average MA(q) models — formula, invertibility condition
6.3 Duality between AR and MA (infinite order representations) — derivation
6.4 ARMA(p,q) models — combined formula, Wold decomposition theorem (why ARMA is general)
6.5 ARIMA(p,d,q) — integrating differencing into the model
6.6 Seasonal ARIMA — SARIMA(p,d,q)(P,D,Q)_s — full formula with backshift operator notation
6.7 Backshift/lag operator algebra — essential notation for interviews
6.8 Model identification: using ACF/PACF signatures (cheat-sheet table: AR tails off/PACF cuts off, etc.)
6.9 Parameter estimation:
  - Method of moments / Yule-Walker equations (derivation)
  - Conditional & unconditional Least Squares
  - Maximum Likelihood Estimation (MLE) for ARMA — full derivation of likelihood function
6.10 Model selection: AIC, BIC, AICc — formula derivation and bias-variance tradeoff interpretation
6.11 Diagnostic checking: residual ACF, Ljung-Box on residuals, normality (Q-Q plot), heteroskedasticity checks
6.12 Forecasting with ARIMA — point forecasts + prediction intervals derivation
6.13 Full numerical worked example: fit AR(1), MA(1) by hand with method of moments on toy data
6.14 Reference: STAT 510 Lessons 1–9 map directly to this phase — cross-reference

## PHASE 7 — Regression With Time Series Data (STAT 510 Core)
7.1 Time series regression models — trend + seasonal dummy regressors
7.2 Autocorrelated errors problem — why OLS standard errors are wrong
7.3 Cochrane-Orcutt / generalized least squares (GLS) correction
7.4 Regression with ARIMA errors (transfer function models)
7.5 Spurious regression — Granger-Newbold phenomenon, why R² lies with non-stationary data
7.6 Cointegration intuition (bridge to Phase 11)
7.7 Lagged predictors, distributed lag models
7.8 Intervention analysis / structural breaks — Chow test

## PHASE 8 — Seasonal & Complex Seasonality Models
8.1 Seasonal differencing recap, seasonal unit roots (HEGY test — conceptual)
8.2 SARIMA parameter selection walkthrough
8.3 Multiple seasonalities (e.g., daily + weekly + yearly — relevant to Google/Apple traffic data): TBATS, Fourier terms
8.4 Dynamic harmonic regression — Fourier series as regressors, formula
8.5 Prophet model internals (Facebook/Meta) — piecewise trend + Fourier seasonality + holiday effects — full formula breakdown
8.6 Calendar effects, trading day adjustments, holiday regressors

## PHASE 9 — State Space Models & Kalman Filtering
9.1 State space representation — observation equation + state (transition) equation
9.2 Kalman filter — full derivation: predict step, update step, Kalman gain formula
9.3 Kalman smoother
9.4 Local level model, local linear trend model — connecting to ETS
9.5 Structural time series models (Bayesian Structural Time Series — BSTS, used at Google for causal impact analysis)
9.6 Dynamic linear models (DLMs)
9.7 Numerical example: 1D Kalman filter hand computation over a few timesteps

## PHASE 10 — Volatility Modeling (Finance-flavored, common in quant interviews)
10.1 Stylized facts of financial time series: volatility clustering, fat tails, leverage effect
10.2 ARCH(q) models — full formula derivation, Engle's original motivation
10.3 GARCH(p,q) — derivation, stationarity/persistence condition
10.4 EGARCH, GJR-GARCH — asymmetric volatility
10.5 Estimating GARCH via MLE — likelihood function
10.6 Value-at-Risk / volatility forecasting applications

## PHASE 11 — Multivariate Time Series
11.1 Vector Autoregression (VAR) — matrix formula derivation
11.2 Stability/stationarity conditions for VAR (eigenvalues of coefficient matrix)
11.3 Granger causality — formal definition, F-test derivation, common interview misconception ("Granger causality ≠ causality")
11.4 Impulse response functions — derivation and interpretation
11.5 Forecast error variance decomposition
11.6 Cointegration — Engle-Granger two-step method, formula
11.7 Vector Error Correction Model (VECM) — derivation, connecting short-run/long-run dynamics
11.8 Johansen test for cointegration rank (conceptual + when to use)

## PHASE 12 — Spectral / Frequency Domain Analysis
12.1 Motivation: why look at frequency domain
12.2 Fourier transform of a time series — formula
12.3 Periodogram — definition, relationship to ACF (Wiener-Khinchin theorem)
12.4 Spectral density function — derivation
12.5 Smoothing the periodogram (Daniell filters)
12.6 Applications: detecting hidden periodicities, filtering

## PHASE 13 — Time Series Cross-Validation & Evaluation (Interview-Critical)
13.1 Why standard k-fold CV is invalid for time series (leakage)
13.2 Rolling-origin / expanding window cross-validation — formal procedure
13.3 Time series split, blocked CV, purged CV (finance-specific, embargo periods)
13.4 Forecast accuracy metrics — full derivation & when each is appropriate:
  - MAE, MSE, RMSE
  - MAPE, sMAPE (and why MAPE breaks near zero)
  - MASE (Mean Absolute Scaled Error) — Hyndman's preferred metric, formula & rationale
  - Pinball loss / quantile loss for probabilistic forecasts
  - CRPS (Continuous Ranked Probability Score)
13.5 Prediction intervals vs confidence intervals — conceptual distinction
13.6 Backtesting frameworks at scale (production ML systems)

## PHASE 14 — Feature Engineering for ML-based Forecasting
14.1 Lag features, rolling statistics (mean, std, min, max), expanding window features
14.2 Date/time features: cyclical encoding (sin/cos) of hour/day/month
14.3 Fourier features as regressors
14.4 Target encoding, holiday/event flags
14.5 Handling multiple related series (hierarchical/grouped time series) — feature design
14.6 Lag leakage pitfalls — a classic interview gotcha

## PHASE 15 — Machine Learning Models for Forecasting
15.1 Why tree-based models struggle with trend extrapolation (formal reasoning)
15.2 Gradient Boosted Trees (XGBoost/LightGBM) for forecasting — framing as supervised regression
15.3 Global models vs local models — pooling series (relevant to M4/M5 competition learnings)
15.4 Hierarchical & grouped time series reconciliation — bottom-up, top-down, MinT (trace minimization) — formula derivation
15.5 Ensemble/stacking approaches for forecasting

## PHASE 16 — Deep Learning for Time Series (Andrew Ng-style rigor)
16.1 Why RNNs — sequence modeling motivation, formal recurrence equation
16.2 Vanilla RNN — forward pass derivation, vanishing/exploding gradient problem (math)
16.3 LSTM — full gate equations derivation (forget/input/output gates, cell state) with intuition for each gate
16.4 GRU — derivation, comparison to LSTM (fewer parameters, when to prefer)
16.5 Sequence-to-sequence architectures for multi-step forecasting
16.6 1D CNNs / Temporal Convolutional Networks (TCN) — dilated causal convolutions, formula
16.7 Attention mechanism — formula derivation (query/key/value), why it helps long-range dependencies
16.8 Transformer architecture for time series — positional encoding challenges, Informer/Autoformer/PatchTST overview
16.9 N-BEATS / N-HiTS architecture — basis expansion approach
16.10 DeepAR (Amazon) — probabilistic autoregressive RNN, likelihood-based training
16.11 Temporal Fusion Transformer (TFT) — interpretable multi-horizon forecasting
16.12 Practical nuances: windowing strategy, teacher forcing, scaling/normalization per-series, cold-start problem

## PHASE 17 — Anomaly & Change Point Detection
17.1 Statistical approaches: control charts, CUSUM, EWMA control charts — formula
17.2 STL-residual based anomaly detection
17.3 Seasonal Hybrid ESD (used at Twitter)
17.4 Change point detection: Bayesian online change point detection, PELT algorithm — conceptual
17.5 Isolation Forest / Autoencoder-based anomaly detection for multivariate series

## PHASE 18 — Causal Inference in Time Series (Google-specific relevance)
18.1 CausalImpact (Google's BSTS-based method) — full conceptual + formula walkthrough
18.2 Synthetic control method — connection to time series
18.3 A/B testing pitfalls with time series data (autocorrelation inflates false positive rate)
18.4 Difference-in-differences with time series data

## PHASE 19 — Probabilistic & Bayesian Forecasting
19.1 Why point forecasts are insufficient — business framing
19.2 Bayesian structural time series — priors, posterior forecasting
19.3 Quantile regression forecasting
19.4 Conformal prediction for time series — coverage guarantees
19.5 Ensemble-based uncertainty quantification

## PHASE 20 — System Design for Forecasting at Scale (Google/Apple Interview Specific)
20.1 Designing a forecasting pipeline: data ingestion → feature store → training → serving → monitoring
20.2 Forecasting millions of SKUs/series — global vs per-series model tradeoffs
20.3 Handling cold-start series (new products, new devices)
20.4 Retraining cadence, drift detection, model monitoring in production
20.5 Latency/scale tradeoffs: batch vs real-time forecasting
20.6 Case study frameworks: "Forecast App Store daily downloads," "Forecast search query volume," "Detect anomalies in server latency time series"

## PHASE 21 — Interview Drill Bank
21.1 Conceptual rapid-fire Q&A (stationarity, ACF/PACF reading, AIC vs BIC, etc.)
21.2 "Derive this formula on the whiteboard" set (Yule-Walker, Kalman gain, GARCH likelihood, MASE)
21.3 Case-study / product-sense questions (Apple: device usage forecasting; Google: ads revenue forecasting)
21.4 Coding round simulation: implement ARIMA/backtest/feature pipeline under time pressure
21.5 Common "gotcha" questions list with model answers
21.6 Behavioral bridge: explaining time series tradeoffs to non-technical stakeholders

## PHASE 22 — Capstone Projects (Practical Mastery)
22.1 Classical project: SARIMA/ETS forecasting on a real seasonal dataset (full Box-Jenkins cycle end-to-end)
22.2 ML project: Gradient boosting + feature engineering on a Kaggle-style multi-series dataset (e.g., M5)
22.3 DL project: LSTM/TFT on a multivariate sensor dataset
22.4 Causal project: CausalImpact-style analysis of a marketing intervention
22.5 System design write-up: architect a production forecasting service

---

## Reference Map (Where Each Resource Fits)
- **STAT 510 (Penn State)**: Phases 1, 3, 4, 6, 7, 8, 10, 11 — the classical statistical backbone
- **Hyndman & Athanasopoulos, FPP3**: Phases 1, 5, 8, 13, 14, 15 (modern applied best practices, ETS taxonomy, MASE, reconciliation)
- **Shumway & Stoffer**: Phases 2, 3, 9, 12 (rigorous state-space and spectral treatment)
- **Hamilton, Time Series Analysis**: Phases 6, 9, 11 (deep theoretical VAR/state-space treatment, econometrics rigor)
- **Andrew Ng-style DL pedagogy**: Phase 16 (derive-then-code approach to RNN/LSTM/attention)
- **Google research (CausalImpact, TFT co-authors, N-BEATS/N-HiTS)**: Phases 18, 16.11, 16.9
- **M4/M5 forecasting competition papers**: Phases 15, 20 (what actually wins at scale)

---

## Suggested Pace
- **Weeks 1–2**: Phase 0–4 (prereqs + stationarity foundations)
- **Weeks 3–5**: Phase 5–8 (classical Box-Jenkins mastery — this is 70% of interview weight)
- **Week 6**: Phase 9–10 (state space + volatility)
- **Week 7**: Phase 11–12 (multivariate + spectral)
- **Week 8**: Phase 13–15 (evaluation + ML forecasting)
- **Weeks 9–10**: Phase 16 (deep learning — the heaviest phase)
- **Week 11**: Phase 17–19 (anomaly, causal, probabilistic)
- **Week 12**: Phase 20–22 (system design, interview drills, capstones)

---

### Next Step
Tell me which phase to expand first (I recommend starting at **Phase 1–4**, building the unbreakable statistical foundation before Box-Jenkins). Each phase, when expanded, will include full formula derivations, hand-worked numerical examples, code, and an interview Q&A drill set.
