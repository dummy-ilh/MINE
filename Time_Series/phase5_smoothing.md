# Phase 5 — Decomposition & Exponential Smoothing

---

## 1. Moving Averages

### Centered MA of order $m$

$$\hat{T}_t = \frac{1}{m}\sum_{j=-k}^{k} x_{t+j}, \qquad m = 2k+1$$

- Averages $k$ points on each side of $t$ to estimate the **trend** at $t$
- $m$ must be **odd** so the window has a natural center point
- **Cost:** you lose $k$ points at each end — a real problem since the most recent data is what you usually need most

### Even-period fix: 2×m MA

When the seasonal period is even (12 monthly, 4 quarterly), center is between points — no single integer index. Fix: average two adjacent $m$-point MAs.

For $m = 12$:
$$\hat{T}_t = \frac{1}{12}\left(\tfrac{1}{2}x_{t-6} + x_{t-5}+\cdots+x_{t+5}+\tfrac{1}{2}x_{t+6}\right)$$

Endpoints get half-weight. Weights remain symmetric and sum to 1.

### Why MA removes seasonality

A window of width $m = $ seasonal period contains **exactly one copy of each season**. Seasonal ups/downs cancel in the average → only trend remains.

---

## 2. STL Decomposition

Classical MA decomposition assumes the seasonal pattern never changes. **STL** (Seasonal-Trend decomposition using Loess) is the modern fix.

**Loess:** fits many small local regressions in a sliding window, instead of one global curve. Think tracing a wiggly coastline with a flexible ruler held against short stretches, not one rigid straightedge.

**STL loop (iterated until convergence):**
1. Loess-smooth the series → rough trend estimate
2. Subtract trend → Loess-smooth each "same-season" subseries separately (all Januaries, all Februaries, ...) → seasonal component
3. Subtract seasonal → re-estimate trend from cleaner residual
4. Repeat until stable. Remainder = what's left.

**Why it matters in interviews:**

| Advantage | What it means |
|---|---|
| Evolving seasonality | Seasonal pattern can drift over time; classical MA forces it fixed |
| Outlier robustness | Optional setting down-weights extremes |
| Any seasonal period | Not restricted to $m = 4$ or $m = 12$ |

**Tradeoff:** additive only in base form. For multiplicative data, log-transform first.

---

## 3. Simple Exponential Smoothing (SES)

### Why not just use MA for forecasting?

MA needs points on **both sides** of the target. The future doesn't exist. We need a method that looks only backward.

### Formula

$$\boxed{\hat{x}_{t+1} = \alpha\, x_t + (1-\alpha)\,\hat{x}_t}$$

| Symbol | Meaning |
|---|---|
| $\hat{x}_{t+1}$ | Forecast for next period |
| $x_t$ | Actual observation at $t$ |
| $\hat{x}_t$ | Previous forecast |
| $\alpha \in (0,1)$ | Smoothing parameter |

**Intuition:** blend the latest data point with the previous forecast. $\alpha \to 1$ = responsive but jumpy. $\alpha \to 0$ = smooth but slow.

### Where "exponential" comes from

Unroll the recursion:

$$\hat{x}_{t+1} = \alpha x_t + \alpha(1-\alpha)x_{t-1} + \alpha(1-\alpha)^2 x_{t-2} + \cdots$$

Each older observation is down-weighted by another factor of $(1-\alpha)$ — exponential decay into the past. Nothing is ever fully discarded; old data just fades.

**Weights sum to 1** (geometric series):
$$\alpha \sum_{j=0}^{\infty}(1-\alpha)^j = \alpha \cdot \frac{1}{\alpha} = 1$$

### Choosing $\alpha$

Minimize sum of squared in-sample forecast errors numerically. Same idea as learning rate optimization in ML.

### Key limitation

SES only tracks a **level** — no trend, no seasonality. Multi-step forecast is **flat forever:**
$$\hat{x}_{t+1} = \hat{x}_{t+2} = \hat{x}_{t+3} = \cdots$$

### Worked example

$x = [10, 12, 11, 14]$, $\alpha = 0.4$, $\hat{x}_1 = 10$

| $t$ | $x_t$ | $\hat{x}_t$ | Calculation |
|---|---|---|---|
| 1 | 10 | 10.000 | (initial) |
| 2 | 12 | 10.000 | $0.4(10)+0.6(10)$ |
| 3 | 11 | 10.800 | $0.4(12)+0.6(10.0)$ |
| 4 | 14 | 10.880 | $0.4(11)+0.6(10.8)$ |
| 5 | — | **12.128** | $0.4(14)+0.6(10.88)$ |

$\hat{x}_6 = \hat{x}_7 = \cdots = 12.128$ — flat, as expected.

---

## 4. Holt's Linear Trend Method

### Idea

Add a second smoothed component: the **trend (slope)**. Two equations, two smoothing parameters.

### Equations

$$\boxed{\ell_t = \alpha\, x_t + (1-\alpha)(\ell_{t-1} + b_{t-1})}$$

$$\boxed{b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)\,b_{t-1}}$$

| Symbol | Meaning |
|---|---|
| $\ell_t$ | Smoothed level at $t$ |
| $b_t$ | Smoothed trend (slope) at $t$ |
| $\alpha$ | Level smoothing parameter |
| $\beta^*$ | Trend smoothing parameter |

**Level:** blends the new observation with (old level + old trend) — i.e., where we were plus how fast we were moving.

**Trend:** SES applied to the *change in level* — blends the newest observed slope with the previous trend estimate.

### Forecast

$$\hat{x}_{t+h} = \ell_t + h\, b_t$$

Linear projection: current level + $h$ steps of current slope. Unlike SES, the forecast is a **sloped line**, not flat.

> **Caveat:** projects the slope forward indefinitely. Can go badly wrong far out if the real trend bends.

### Worked example

$x = [20, 24, 27, 32, 35]$, $\alpha = 0.5$, $\beta^* = 0.3$

**Init:** $\ell_1 = 20$, $b_1 = 24 - 20 = 4$

| $t$ | $x_t$ | $\ell_t$ | $b_t$ |
|---|---|---|---|
| 1 | 20 | 20.000 | 4.000 |
| 2 | 24 | 24.000 | 4.000 |
| 3 | 27 | 27.500 | 3.850 |
| 4 | 32 | 31.675 | 3.948 |
| 5 | 35 | 35.311 | 3.854 |

**Step-by-step for $t = 3$:**
$$\ell_3 = 0.5(27) + 0.5(24 + 4) = 13.5 + 14 = 27.5$$
$$b_3 = 0.3(27.5 - 24) + 0.7(4) = 1.05 + 2.8 = 3.85$$

**Forecasts from $t = 5$:**

| Horizon $h$ | Forecast |
|---|---|
| 1 | $35.311 + 3.854 = \mathbf{39.17}$ |
| 2 | $35.311 + 7.708 = \mathbf{43.02}$ |
| 3 | $35.311 + 11.562 = \mathbf{46.87}$ |

**Sanity check:** raw differences are 4, 3, 5, 3 → avg ≈ 3.75/step. Smoothed trend $b_5 \approx 3.85$. ✓

---

## 5. Holt-Winters Seasonal Method

### Idea

Add a third smoothed component: **seasonality**. Three equations, three parameters.

### Equations (additive)

$$\boxed{\ell_t = \alpha(x_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1})}$$

$$\boxed{b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)\,b_{t-1}}$$

$$\boxed{s_t = \gamma(x_t - \ell_t) + (1-\gamma)\,s_{t-m}}$$

| Symbol | Meaning |
|---|---|
| $s_t$ | Smoothed seasonal component at $t$ |
| $m$ | Seasonal period (12 = monthly, 4 = quarterly, 7 = daily with weekly cycle) |
| $\gamma$ | Seasonal smoothing parameter |

**Level:** strips out the seasonal effect ($x_t - s_{t-m}$) before updating — deseasonalizes on the fly.

**Trend:** identical to Holt's method.

**Seasonal:** $(x_t - \ell_t)$ is today's observed seasonal deviation. Blend it with the same-season estimate from one full cycle ago ($s_{t-m}$). Why $t-m$? Because the seasonal index for "this February" is only found in last February, not last month.

### Forecast

$$\hat{x}_{t+h} = \ell_t + h\, b_t + s_{t - m + (h \bmod m)}$$

Trend projection + the seasonal effect for whichever point in the cycle $h$ steps forward lands on.

### Multiplicative version

Use when seasonal swings scale with the level (megaphone shape on a plot):

| Component | Additive | Multiplicative |
|---|---|---|
| Level | $\alpha(x_t - s_{t-m}) + \cdots$ | $\alpha(x_t / s_{t-m}) + \cdots$ |
| Seasonal | $\gamma(x_t - \ell_t) + \cdots$ | $\gamma(x_t / \ell_t) + \cdots$ |
| Forecast | $(\ell_t + hb_t) + s_{(\cdot)}$ | $(\ell_t + hb_t) \times s_{(\cdot)}$ |

---

## 6. ETS Framework

Modern software (R's `ets()`) names every method: **E**rror · **T**rend · **S**easonal, each **N** (none), **A** (additive), or **M** (multiplicative).

| Method | ETS code |
|---|---|
| SES | ETS(A,N,N) |
| Holt's Linear | ETS(A,A,N) |
| Holt-Winters Additive | ETS(A,A,A) |
| Holt-Winters Multiplicative | ETS(A,A,M) |

ETS also provides formal state-space formulation → proper likelihood, AIC model selection, prediction intervals. Covered in Phase 9.

---

## 7. Quick-Reference Summary

| Method | Tracks | Forecast shape | Parameters |
|---|---|---|---|
| SES | Level only | Flat | $\alpha$ |
| Holt | Level + Trend | Sloped line | $\alpha, \beta^*$ |
| Holt-Winters | Level + Trend + Seasonal | Sloped + seasonal wave | $\alpha, \beta^*, \gamma$ |

---

## 8. Self-Check

**Q1.** Why does SES produce a flat multi-step forecast?  
SES maintains only a single level component — no slope, so every future period gets the same number.

**Q2.** As $\alpha \to 1$, what happens to the weight on an observation from 10 periods ago?  
Weight = $\alpha(1-\alpha)^{10}$. With $\alpha$ near 1, $(1-\alpha)^{10} \approx 0$ — collapses to nearly zero. Only the most recent point matters.

**Q3.** Why does Holt-Winters use $s_{t-m}$ and not $s_{t-1}$?  
$s_{t-1}$ is the *previous* season (e.g., November when you're in December). $s_{t-m}$ is the *same* season one full cycle ago — the correct seasonal index for this point in the year.

**Q4.** What does ETS(A,A,A) stand for and which method is it?  
Additive Error, Additive Trend, Additive Seasonal → Holt-Winters additive.
