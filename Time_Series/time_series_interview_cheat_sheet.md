# Time Series Interview Cheat Sheet — Night Before

One page (long) to skim, not to learn from cold. Every item links back to a phase you've already derived — if something doesn't click, that's a signal, not a failure.

---

## PART 1 — MASTER DECISION FLOWCHART

```
START: I have a time series. What do I do?
│
├─ 1. PLOT IT. Look for trend / seasonality / cycle / obvious outliers.
│
├─ 2. IS IT STATIONARY?
│      Run ADF (H0: unit root/non-stationary) AND KPSS (H0: stationary)
│      │
│      ├─ ADF rejects + KPSS fails to reject → STATIONARY. Proceed to step 4.
│      ├─ ADF fails to reject + KPSS rejects → NON-STATIONARY. Difference it (d=1),
│      │      re-test. If still non-stationary, d=2 (rare beyond this — check for bug).
│      ├─ Both "stationary" but conflicting sign → likely TREND-STATIONARY.
│      │      Detrend (fit deterministic trend, don't difference) instead.
│      └─ Both inconclusive → get more data / inspect visually.
│
├─ 3. IS VARIANCE STABLE (not growing with the level)?
│      No → log or Box-Cox transform FIRST, before differencing.
│
├─ 4. PLOT ACF & PACF OF THE (now-stationary) SERIES.
│      ACF tails off, PACF cuts off at lag p        → AR(p)
│      ACF cuts off at lag q, PACF tails off         → MA(q)
│      Both tail off                                  → ARMA(p,q)
│      Strong spike at seasonal lag s (and multiples) → SARIMA, add seasonal terms
│
├─ 5. FIT CANDIDATE MODELS. Compare via AIC (forecast-focused) / BIC (true-model-focused).
│
├─ 6. CHECK RESIDUALS.
│      Residual ACF ~ white noise? Ljung-Box p-value large (>0.05)?
│      No → go back to step 4/5, model missed structure (check the flagged lag).
│      Yes → proceed.
│
├─ 7. DOES VARIANCE ITSELF CLUSTER (volatility clustering in residuals)?
│      Yes (common in finance) → layer ARCH/GARCH on the residuals.
│
├─ 8. MULTIPLE RELATED SERIES THAT INFLUENCE EACH OTHER?
│      Yes → VAR. Test Granger causality. Test for cointegration (Engle-Granger/
│      Johansen) before deciding VAR-on-differences vs. VECM.
│
├─ 9. SCALE / NUMBER OF SERIES?
│      One or few series, need interpretability → classical (ETS/ARIMA/GARCH/VAR).
│      Thousands+ series, rich features, tabular → global ML model (GBM) w/ lag
│      features (mind leakage!) or DeepAR/TFT for probabilistic + cold start.
│      Very long sequences, complex non-linear pattern, budget allows → LSTM/
│      TCN/Transformer.
│
└─ 10. FORECAST + INTERVAL + EVALUATE.
       Point forecast → recursive substitution (own forecasts feed forward).
       Interval → analytic (ARIMA/Kalman), quantile regression, or conformal
       (model-agnostic, but check exchangeability for time series!).
       Evaluate via ROLLING-ORIGIN CV (never k-fold). Metric: MASE > RMSE > MAPE
       (MAPE breaks near zero).
```

---

## PART 2 — CORE DEFINITIONS (one line each)

- **Stationarity (weak):** constant mean, constant variance, autocovariance depends only on lag not time.
- **White noise:** zero mean, constant variance, zero autocorrelation at every lag ≠0.
- **Unit root:** the AR(1) boundary case φ=1 (random walk); variance grows unboundedly with t.
- **ACF ρ(k):** normalized correlation between xₜ and xₜ₋ₖ, includes indirect/chained effects.
- **PACF:** correlation between xₜ and xₜ₋ₖ AFTER removing effects of intermediate lags — the DIRECT relationship only.
- **Invertibility (MA):** |θ|<1; ensures a unique, convergent AR(∞) representation.
- **Cointegration:** two individually non-stationary series whose specific linear combination IS stationary — a genuine long-run relationship, not spurious.
- **Spurious regression:** high R² / "significant" coefficient between two UNRELATED non-stationary (unit-root) series — an artifact of both independently drifting.
- **Granger causality:** X's past improves the STATISTICAL prediction of Y beyond Y's own past. NOT real causation (confounding, no mechanism required).
- **Volatility clustering:** large shocks followed by large shocks (in magnitude, either sign) — what ARCH/GARCH model.
- **Vanishing gradient:** repeated multiplication of <1 terms during backprop-through-time causes long-range learning signal to decay geometrically to ~0.
- **Exchangeability (conformal prediction):** the assumption that calibration and test points are statistically interchangeable — the basis of conformal coverage guarantees, and shaky for time series.
- **Counterfactual:** what WOULD have happened without an intervention — fundamentally unobservable, must be estimated (CausalImpact, synthetic control, DiD).

---

## PART 3 — ACF/PACF SIGNATURE TABLE (memorize cold)

| Process | ACF | PACF |
|---|---|---|
| White noise | ~0 everywhere | ~0 everywhere |
| AR(p) | tails off (φᵏ decay, or oscillating if φ<0) | cuts off after lag p |
| MA(q) | cuts off after lag q | tails off |
| ARMA(p,q) | tails off | tails off |

**Why:** AR(p) directly uses only p lags of x, but the ACF still shows indirect "gossip chain" echoes beyond p; PACF strips those out, revealing the true cutoff. MA(q) directly uses only q lags of noise, giving a hard ACF cliff at q; but expressing it in terms of past x requires infinite lags (AR(∞)), so PACF never cuts cleanly.

---

## PART 4 — KEY FORMULAS BY PHASE

**Phase 2 — Random walk:** xₜ = xₜ₋₁ + εₜ → Var(xₜ) = tσ² (grows forever, non-stationary)

**Phase 4 — ADF regression:** Δxₜ = γxₜ₋₁ + εₜ, H0: γ=0 (unit root). Needs special Dickey-Fuller critical values, NOT ordinary t-table, because xₜ₋₁ is non-stationary under H0.

**Phase 5 — SES:** x̂ₜ₊₁ = αxₜ + (1-α)x̂ₜ. Weights on past decay as (1-α)ʲ. Flat forecast only (no trend/seasonal capability) = ARIMA(0,1,1).

**Phase 6 Pt1 — AR(1):** xₜ=φxₜ₋₁+εₜ. Stationary iff |φ|<1. Var = σ²/(1-φ²). ρ(k)=φᵏ.

**Phase 6 Pt2 — MA(1):** xₜ=εₜ+θεₜ₋₁. ρ(1)=θ/(1+θ²), ρ(k≥2)=0. Always stationary; invertible iff |θ|<1.

**Phase 6 Pt4 — AIC/BIC:** AIC=-2lnL+2k (favors forecasting accuracy). BIC=-2lnL+k·ln(n) (favors true, simpler model; penalizes harder as n grows).

**Phase 6 Pt5 — AR(1) forecast:** x̂_{T+h}=φʰxₜ → 0 (mean) as h→∞. Forecast error variance: σ²Σφ²ʲ → σ²/(1-φ²) (ceiling). Random walk: variance grows unboundedly (hσ²), no ceiling.

**Phase 8 — Fourier seasonality:** s(t)=Σ[βₖsin(2πkt/m)+γₖcos(2πkt/m)]. K controls flexibility; multiple periods = separate additive blocks.

**Phase 9 — Kalman gain:** Kₜ = P_{t|t-1}Z / (Z²P_{t|t-1}+R). High R (noisy sensor) → gain→0 (trust prior). High P (uncertain prior) → gain→1 (trust observation). Reduces to SES on the local-level model.

**Phase 10 — GARCH(1,1):** σₜ² = ω+α₁r²ₜ₋₁+β₁σ²ₜ₋₁. Stationary iff α₁+β₁<1. Long-run variance = ω/(1-α₁-β₁).

**Phase 13 — MASE:** (mean |error| of model) / (mean |error| of naive-lag-1-or-lag-m baseline). <1 = beats naive; >1 = worse than doing nothing clever.

**Phase 13 — Pinball loss:** asymmetric; at q=0.5 reduces exactly to (half) MAE.

**Phase 16 Pt2 — LSTM cell update:** Cₜ=fₜ⊙Cₜ₋₁+iₜ⊙C̃ₜ. ADDITIVE — this is the specific structural fix for vanishing gradients (gradients flow through addition largely unchanged, not repeated multiply-and-squash).

**Phase 19 — Conformal interval:** [ŷ ± q̂], where q̂ = the empirical (1-α) quantile of calibration-set absolute errors. Guarantee rests on exchangeability.

---

## PART 5 — MODEL COMPARISON QUICK TABLE

| Model | Handles trend? | Handles seasonality? | Handles multiple series? | Inference cost |
|---|---|---|---|---|
| ETS/Holt-Winters | Yes (Holt+) | Yes (HW) | No (one model each) | Very cheap |
| ARIMA/SARIMA | Yes (d) | Yes (SARIMA) | No | Cheap |
| VAR/VECM | Yes | Limited | Yes (that's the point) | Cheap-moderate |
| GARCH | N/A (models variance) | N/A | Multivariate GARCH exists | Cheap |
| Prophet | Yes (piecewise) | Yes (Fourier) | No (one per series, but fast) | Cheap |
| Gradient boosting | No (needs detrend first!) | Yes (via features) | Yes (global model) | Moderate |
| LSTM/GRU | Yes (learned) | Yes (learned) | Yes (global) | Higher |
| DeepAR | Yes | Yes | Yes (built for this) | Higher, probabilistic |
| TFT | Yes | Yes | Yes | Highest, interpretable |

---

## PART 6 — TRAPS PEOPLE ACTUALLY FALL INTO (say these out loud)

1. **k-fold CV on time series** → leakage, trains on future to predict past.
2. **MAPE on near-zero data** → division blows up; use MASE instead.
3. **Lag/rolling features that accidentally include today** → looks great in training, fails in production. Always shift-then-window.
4. **Detrending a random walk / differencing a trend-stationary series** → wrong fix for the wrong kind of non-stationarity.
5. **Trusting a high R² time-series regression without checking both series for unit roots first** → spurious regression (Granger-Newbold).
6. **Assuming Granger causality = causality** → it's predictive improvement only; confounding is always a live alternative explanation.
7. **Using a tree-based model on trending data without detrending first** → trees cannot extrapolate past the training range, structurally (leaf outputs are just averages of training targets).
8. **Applying textbook conformal prediction to time series without adapting the calibration window** → exchangeability is violated; coverage guarantee may not actually hold.
9. **Retraining a production model too often** → overreacts to ordinary noise as if it were genuine drift.
10. **Running anomaly detection directly on raw seasonal data** → flags every December as an anomaly. Decompose first, detect on the remainder.

---

## PART 7 — "EXPLAIN THIS IN 30 SECONDS" PROMPTS

- **Prophet:** additive decomposition (trend+seasonal+holiday+noise) where trend is piecewise-linear with estimated changepoints, seasonality is Fourier terms, holidays are dummy regressors — fit as one combined regression.
- **Kalman filter:** predict (project state forward, uncertainty grows) → update (blend prediction with new observation via the gain, uncertainty shrinks) → repeat forever. Generalizes SES/Holt/ARIMA as special cases.
- **Why LSTM fixes vanishing gradients:** cell state updates via addition (gated by forget/input gates) rather than repeated matrix-multiply-and-squash, so gradients can flow backward largely intact when the forget gate is near 1.
- **Why attention fixes the encoder-decoder bottleneck:** decoder can directly query any individual encoder position via Q/K similarity, instead of relying on one fixed-size compressed context vector.
- **CausalImpact:** fit a Bayesian structural time series (Kalman filter) on pre-intervention data relating target to an unaffected control series; project forward through treatment period using control's real values; effect = actual minus this projected counterfactual.
- **Why MASE > MAPE:** MASE's denominator is a stable average over many baseline errors; MAPE's denominator is a single (possibly near-zero) actual value.

---

## PART 8 — SYMBOL DECODER (when a formula looks unfamiliar mid-interview)

φ = AR coefficient · θ = MA coefficient · σ² = noise variance · ρ(k)/γ(k) = ACF/autocovariance at lag k
α (Holt/SES context) = smoothing parameter · α (ARCH context) = shock coefficient · α (ADF context) = significance level — **context always disambiguates, this field reuses Greek letters constantly, don't panic**
d = differencing order · p,q = AR/MA order · P,D,Q,s = seasonal versions + period
Kₜ = Kalman gain · Q,R (state-space) = process/observation noise variance
λ (VECM) = error-correction speed · λ (EWMA) = smoothing parameter — again, context-dependent

---

Good luck. You've derived every one of these from scratch at least once — this page is just a pointer back to that work, not new information.
