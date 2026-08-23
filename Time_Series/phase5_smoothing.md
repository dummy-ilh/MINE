# Phase 5: Decomposition & Exponential Smoothing (SES, Holt, Holt-Winters)

In Phase 1 you informally computed a moving average and subtracted it to find seasonality. In Phase 4 you learned the formal fixes for non-stationarity. Now we build a full family of forecasting methods that are still, today, used as production baselines at real companies — and we derive the recursive formulas behind them, not just the recipe.

---

## 1. Moving Averages, formalized

**Definition.** A moving average of order $m$ (written $\text{MA}(m)$ — a different thing from the "MA(q)" moving-average *model* in Phase 6; same name, unrelated concept, an unfortunate collision of terminology to keep separate by context) is:
$$
\hat{T}_t = \frac{1}{m}\sum_{j=-k}^{k} x_{t+j}, \qquad m = 2k+1
$$
To estimate the trend at time $t$, average the $k$ points before, $x_t$ itself, and the $k$ points after — a symmetric window centered exactly on $t$. This is why $m$ must be odd: an odd-width window has a natural center point.

**The even-order problem.** If your seasonal period is even (4 for quarterly data, 12 for monthly data), you can't center a window of that size on a single point — a 4-point window straddles two time points, not one. Picture a 4-legged table: there's no single leg exactly in the middle. The fix, done by hand in Phase 1: compute the order-$m$ moving average, then average TWO adjacent ones together. This is the **2×m moving average** — the formal name for "average of adjacent 4-quarter averages" from Phase 1, section 5. For $m=12$ (monthly, yearly seasonality):
$$
\hat{T}_t = \frac{1}{12}\Big(\tfrac{1}{2}x_{t-6} + x_{t-5}+\dots+x_{t+5}+\tfrac{1}{2}x_{t+6}\Big)
$$
The endpoints get half weight — exactly what "averaging two adjacent 12-point windows" works out to algebraically, keeping the weights symmetric and summing to 1.

**Why averaging removes seasonality.** If the window width $m$ exactly equals the seasonal period, every window contains exactly one copy of each season — one January, one February, ... one December, for $m=12$. The seasonal ups and downs inside the window cancel out in the average, leaving just the smooth trend. This is why a 4-quarter window works for quarterly seasonal data.

**The unavoidable cost:** you lose $k$ points at each end (you can't center a window on the first or last few points — there's nothing beyond the edge to average). A real practical limitation: production systems often need separate handling for the most recent points, which are exactly the ones you usually care about most for forecasting.

---

## 2. STL Decomposition (Seasonal-Trend decomposition using Loess) — conceptual mechanics

Classical moving-average decomposition has two weaknesses: it assumes the seasonal pattern never changes across cycles, and it handles irregular calendar effects poorly. **STL** is a more flexible, modern alternative.

**Loess** (also LOWESS) — "LOcally Estimated Scatterplot Smoothing." Instead of fitting one single curve to the whole series at once (like ordinary regression), Loess slides a small window across the data and, at each point, fits a small local regression using only the nearby points in that window, then moves on. Think of tracing a wiggly coastline with a flexible ruler held only against short local stretches at a time, rather than trying to draw one straight ruler across the entire map. The result is a smooth curve built from many small, overlapping local fits, letting the trend bend in ways a single global formula couldn't.

**STL's iterative loop:**
1. Estimate a rough trend (Loess smoothing).
2. Subtract it, then estimate the seasonal component from what's left — Loess-smoothing each "same season" subseries separately (all the Januaries smoothed together, apart from all the Februaries). This is also why STL lets the seasonal pattern slowly evolve over time, unlike the classical method, which forces it to be identical every cycle.
3. Subtract the seasonal estimate, re-estimate the trend from what's left (cleaner now that seasonality is gone).
4. Repeat steps 1–3 a few times until estimates stabilize.
5. What remains after removing trend and seasonal is the remainder/irregular component.

**Why STL comes up in interviews:** advantages over classical decomposition are (a) seasonality can gradually change over time instead of being rigidly fixed, (b) robustness to outliers (an optional setting down-weights extreme remainder values so one bad data point doesn't distort trend/seasonal estimates), and (c) it works with any seasonal period. The tradeoff: it's additive-only in basic form — log-transform first for multiplicative-feeling data, as in Phase 4, section 8.

---

## 3. Why a whole new family of methods (exponential smoothing)?

Moving averages are good for describing/decomposing a series you already have. They're awkward for **forecasting**, because a moving average by definition needs points on both sides of the target — and the future doesn't exist yet to average over. We need a method built to look only backward and project forward. That's exponential smoothing.

---

## 4. Simple Exponential Smoothing (SES) — full derivation

**Core idea:** to forecast tomorrow, take a weighted average of all past observations, giving more weight to recent ones and progressively less to older ones — recent data is usually more relevant to what happens next.

**Recursive formula:**
$$
\hat{x}_{t+1} = \alpha\, x_t + (1-\alpha)\,\hat{x}_t
$$
- $\hat{x}_{t+1}$ = forecast for the next step, made using information through time $t$.
- $x_t$ = the actual observed value at time $t$.
- $\hat{x}_t$ = the previous forecast — what was predicted for time $t$ before it was actually observed.
- $\alpha$ (alpha) = the **smoothing parameter**, between 0 and 1 — the single knob controlling this whole method.

**Plain English:** "next period's forecast = a blend of the newest actual data point and what we previously thought would happen, weighted by $\alpha$." Picture adjusting a thermostat that's trying to track the room's "true" comfortable temperature, but only ever nudges toward the latest reading rather than jumping straight to it. $\alpha$ close to 1 → the forecast reacts strongly and quickly to new data (responsive but jumpy). $\alpha$ close to 0 → the forecast barely moves (smooth but slow to react). $\alpha$ is structurally the same idea as a "learning rate" or an exponential moving average decay parameter elsewhere in machine learning.

### 4.1 Unrolling the recursion — where the name comes from

Substitute the recursion into itself repeatedly. Start with:
$$
\hat{x}_{t+1} = \alpha x_t + (1-\alpha)\hat{x}_t
$$
Substitute $\hat{x}_t = \alpha x_{t-1} + (1-\alpha)\hat{x}_{t-1}$:
$$
\hat{x}_{t+1} = \alpha x_t + (1-\alpha)\big[\alpha x_{t-1} + (1-\alpha)\hat{x}_{t-1}\big] = \alpha x_t + \alpha(1-\alpha)x_{t-1} + (1-\alpha)^2\hat{x}_{t-1}
$$
Keep substituting further back:
$$
\hat{x}_{t+1} = \alpha x_t + \alpha(1-\alpha)x_{t-1} + \alpha(1-\alpha)^2 x_{t-2} + \alpha(1-\alpha)^3 x_{t-3} + \dots
$$
The weight pattern — $\alpha$, then $\alpha(1-\alpha)$, then $\alpha(1-\alpha)^2$, then $\alpha(1-\alpha)^3$ — shrinks by another factor of $(1-\alpha)$ each step back. Picture a stack of colored filters: each older observation's light has to pass through one more filter than the observation after it, dimming exponentially with age. Nothing is ever fully discarded (unlike a plain moving average's hard cutoff window) — old data just fades toward irrelevance rather than vanishing outright. This is the literal reason it's called "exponential" smoothing.

**Weights sum to 1** (required for a legitimate weighted average):
$$
\alpha + \alpha(1-\alpha) + \alpha(1-\alpha)^2 + \dots = \alpha \sum_{j=0}^{\infty}(1-\alpha)^j = \alpha \cdot \frac{1}{1-(1-\alpha)} = \alpha \cdot \frac{1}{\alpha} = 1 \checkmark
$$
using the geometric series sum $\sum_{j=0}^\infty r^j = \frac{1}{1-r}$ for $|r|<1$, valid here since $0<1-\alpha<1$.

**Choosing $\alpha$ in practice:** rather than picking it by hand, $\alpha$ (and the starting value $\hat{x}_1$) are typically chosen by **minimizing the sum of squared forecast errors** on historical data — trying different $\alpha$ values and keeping whichever would have produced the smallest historical mistakes. A genuine numerical optimization, foreshadowing the Maximum Likelihood approach used for ARIMA in Phase 6.

**The core limitation: SES can only produce a flat forecast line.** Because the recursion only ever updates a single "level," asking it to forecast 10 steps ahead gives 10 identical flat numbers: $\hat{x}_{t+1} = \hat{x}_{t+2} = \dots$. SES has no concept of trend or seasonality — it assumes the series just wanders around a slowly-updating flat level. This limitation motivates the next two methods directly.

### 4.2 Small worked example of SES

Data: $x = [10, 12, 11, 14]$, $\alpha = 0.4$, starting forecast $\hat{x}_1 = 10$ (a common simple choice: use the first observation itself).

$\hat{x}_2 = \alpha x_1 + (1-\alpha)\hat{x}_1 = 0.4(10) + 0.6(10) = 10.0$
$\hat{x}_3 = 0.4(12) + 0.6(10.0) = 4.8 + 6.0 = 10.8$
$\hat{x}_4 = 0.4(11) + 0.6(10.8) = 4.4 + 6.48 = 10.88$
$\hat{x}_5 = 0.4(14) + 0.6(10.88) = 5.6 + 6.528 = 12.128$

Notice the forecast trails the data — it "chases" each new value only partway, exactly as expected from a weighted blend. And since SES has no trend mechanism, $\hat{x}_5 = 12.128$ would also be the forecast for $\hat{x}_6$, $\hat{x}_7$, and every step beyond — flat forever, regardless of how far ahead you ask.

---

## 5. Holt's Linear Trend Method — adding a trend component

**Idea:** keep SES's smoothed level, but also maintain a second, separately-smoothed estimate of the TREND (the slope), and combine both to forecast. In short: **run SES twice — once on the level, once on the level's own rate of change.**

**Level equation:**
$$
\ell_t = \alpha\, x_t + (1-\alpha)(\ell_{t-1} + b_{t-1})
$$
**Trend equation:**
$$
b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)\,b_{t-1}
$$

New symbols:
- $\ell_t$ (script L) = the smoothed LEVEL at time $t$ (like $\hat{x}_t$ in SES, now separated from trend).
- $b_t$ = the smoothed TREND (slope) at time $t$ — how much the level is currently rising or falling per step.
- $\beta^*$ (beta-star; the asterisk just distinguishes it from an unrelated $\beta$ used elsewhere in statistics) = a second smoothing parameter, controlling how quickly the TREND estimate adapts — playing the same responsiveness-vs-smoothness role $\alpha$ plays for the level.

**Level equation, plain English:** the new level blends the newest actual observation with what would have been predicted using the old level plus old trend ($\ell_{t-1}+b_{t-1}$ — "where we were, plus how fast we were moving"). Exactly SES's logic, except the "previous forecast" now accounts for trend instead of assuming a flat line.

**Trend equation, plain English:** the new trend blends the most recently observed change in level ($\ell_t - \ell_{t-1}$) with the old trend estimate. Literally SES's exact same smoothing logic, applied to the slope instead of the level.

**Forecasting $h$ steps ahead:**
$$
\hat{x}_{t+h} = \ell_t + h\, b_t
$$
Take the current level, project forward in a straight line using the current slope, for $h$ steps. Unlike SES's flat forecast, this produces a sloped line into the future. Caveat: it's a fixed straight-line projection — a real risk if you forecast far ahead and the real trend doesn't hold that long.

---

## 6. Holt-Winters Seasonal Method — adding seasonality on top

**Idea:** take Holt's level + trend structure and add a third smoothed component tracking seasonality — **run SES a third time, now on the seasonal pattern.** Additive version shown here (constant absolute seasonal effects, Phase 1 section 4.1).

**Level:**
$$
\ell_t = \alpha(x_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1})
$$
**Trend:**
$$
b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)b_{t-1}
$$
**Seasonal:**
$$
s_t = \gamma(x_t - \ell_t) + (1-\gamma)\,s_{t-m}
$$

New symbols:
- $s_t$ = the smoothed SEASONAL component at time $t$.
- $m$ = the seasonal period (12 for monthly-yearly, 4 for quarterly, 7 for daily-weekly — same $m$ as section 1).
- $\gamma$ (gamma — yet another reused Greek letter, unrelated to autocovariance $\gamma(k)$ from Phase 3 or the ADF coefficient from Phase 4) = a third smoothing parameter controlling how fast the seasonal pattern is allowed to evolve.

**Level equation, plain English:** $x_t - s_{t-m}$ strips out the seasonal effect from the newest observation before updating the level, using the seasonal estimate from exactly one full cycle ago ($s_{t-m}$ — the best current guess for "this same point in the cycle"). This gives a deseasonalized observation to blend into the level, exactly as done manually in Phase 1's seasonal adjustment exercise.

**Seasonal equation, plain English:** $x_t - \ell_t$ is how far the actual observation deviates from the current smoothed level — that deviation IS the freshly observed seasonal effect. Blend this fresh evidence with the previous estimate for this same point in the cycle ($s_{t-m}$, again one full cycle back), controlled by $\gamma$.

**Forecasting $h$ steps ahead:**
$$
\hat{x}_{t+h} = \ell_t + h\, b_t + s_{t-m+h \bmod m}
$$
Take the trend-projected level (same as Holt's method), then add back the appropriate seasonal effect for whichever point in the cycle the forecast horizon lands on. The $\bmod$ ("modulo") operation wraps the index around so it cycles back to the right season — forecasting 14 steps ahead with $m=12$ lands on "season 2," the same relative position as February if $m=12$ starts at January.

**Multiplicative version:** swap subtraction for division and addition for multiplication throughout ($x_t/s_{t-m}$ instead of $x_t - s_{t-m}$; $s_t = \gamma \frac{x_t}{\ell_t} + (1-\gamma)s_{t-m}$; forecast $= (\ell_t + hb_t)\times s_{t-m+h\bmod m}$) — use when the seasonal swing scales with the trend level (megaphone shape, Phase 1 section 4.2 / Phase 4 section 8).

---

## 7. The "ETS" framework — why this naming matters

Modern software (R's `ets()`) organizes this entire family by naming each method with three letters: **E**rror type, **T**rend type, **S**easonal type — each **N** (none), **A** (additive), or **M** (multiplicative). So:
- SES ≈ ETS(A,N,N) — additive errors, no trend, no seasonality
- Holt's method ≈ ETS(A,A,N) — additive errors, additive trend, no seasonality
- Holt-Winters additive = ETS(A,A,A)
- Holt-Winters multiplicative = ETS(A,A,M)

Worth recognizing by name — it's the standard modern vocabulary (Hyndman's taxonomy) even though we've derived the "classical" versions above. The ETS framework additionally reformulates all of these as formal state-space models (proper likelihood functions, automatic model selection via AIC, genuine prediction intervals), covered explicitly in Phase 9 (state space models & Kalman filtering).

---

## 8. Full numerical worked example: Holt's Linear Trend Method by hand

Data: $x = [20, 24, 27, 32, 35]$ — clearly trending.

**Initialization** (a practical necessity — the recursion needs a starting point):
$\ell_1 = x_1 = 20$ (simple choice: use the first observation as the initial level)
$b_1 = x_2 - x_1 = 24 - 20 = 4$ (simple choice: use the first observed difference as the initial trend)

Use $\alpha = 0.5$, $\beta^* = 0.3$.

**Step $t=2$:**
$\ell_2 = 0.5(24) + 0.5(20+4) = 12 + 12 = 24.0$
$b_2 = 0.3(24-20) + 0.7(4) = 1.2+2.8=4.0$

**Step $t=3$:**
$\ell_3 = 0.5(27) + 0.5(24+4.0) = 13.5+14.0=27.5$
$b_3 = 0.3(27.5-24)+0.7(4.0) = 1.05+2.8=3.85$

**Step $t=4$:**
$\ell_4 = 0.5(32)+0.5(27.5+3.85)=16+15.675=31.675$
$b_4 = 0.3(31.675-27.5)+0.7(3.85)=1.2525+2.695=3.9475$

**Step $t=5$:**
$\ell_5 = 0.5(35)+0.5(31.675+3.9475)=17.5+17.81125=35.31125$
$b_5 = 0.3(35.31125-31.675)+0.7(3.9475)=1.090875+2.76325=3.854125$

**Forecast for $t=6,7,8$ ($h=1,2,3$ steps ahead from $t=5$):**
$\hat{x}_6 = \ell_5 + 1\cdot b_5 = 35.31125+3.854125 = 39.165375$
$\hat{x}_7 = \ell_5 + 2\cdot b_5 = 35.31125+7.70825 = 43.0195$
$\hat{x}_8 = \ell_5 + 3\cdot b_5 = 35.31125+11.562375=46.873625$

**Sanity check:** the raw series rose by roughly 4, 3, 5, 3 per step (differences: 24-20=4, 27-24=3, 32-27=5, 35-32=3) — averaging around +3.75/step. The final smoothed trend estimate $b_5 \approx 3.85$ sits right in that ballpark, and the forecasts extend forward at a similar, slightly-smoothed rate. Always run this check: does the model's implied trend roughly match what's visible just from the raw differences?

---

## 9. Self-check questions

1. Why can Simple Exponential Smoothing (SES) never produce anything other than a flat forecast line?
   *SES only maintains a single smoothed LEVEL, with no separate mechanism to estimate or project a slope/trend — every future forecast just repeats the same current level estimate.*

2. In the geometric-series expansion of SES, what happens to the weight given to an observation from 10 periods ago as $\alpha$ gets closer to 1?
   *The weight on older observations shrinks even faster. Since the decay factor is $(1-\alpha)$, a larger $\alpha$ means a smaller $(1-\alpha)$, so weights collapse toward zero more quickly for older data — with $\alpha$ close to 1, only the very most recent observations matter.*

3. In Holt-Winters, why does the level equation use $s_{t-m}$ (seasonal estimate from one full cycle ago) instead of, say, $s_{t-1}$?
   *The correct seasonal adjustment for "this point in the cycle" (e.g., "this December") is only found exactly one full cycle back. $s_{t-1}$ would be last month's seasonal effect — November's — the wrong season entirely.*

4. What do the three letters in ETS(A,A,A) stand for, and what method does it correspond to among the ones covered in this phase?
   *Error type, Trend type, Seasonal type, all Additive — the additive Holt-Winters seasonal method.*

---

## What's next

Phase 6 is the big one: the full Box-Jenkins methodology — AR, MA, ARMA, ARIMA, and SARIMA models, derived rigorously from first principles (stationarity conditions, invertibility, the Yule-Walker equations promised in Phase 3, Maximum Likelihood estimation, AIC/BIC model selection, and full residual diagnostics), plus the formal proof of the ACF/PACF signature table from Phase 3, section 6. This phase typically carries the largest share of interview weight in classical time series, so it's taken in careful, well-paced sub-steps rather than all at once.
