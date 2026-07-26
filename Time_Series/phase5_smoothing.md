# Phase 5: Decomposition & Exponential Smoothing (SES, Holt, Holt-Winters)

In Phase 1 you informally computed a moving average and subtracted it to find seasonality. In Phase 4 you learned the formal fixes for non-stationarity. Now we build a full family of forecasting methods that are still, today, used as production baselines at real companies — and we derive the recursive formulas behind them, not just the recipe.

---

## 1. Moving Averages, formalized

**Definition.** A moving average of order $m$ (written $\text{MA}(m)$ — careful, this is a DIFFERENT thing from the "MA(q)" moving-average *model* you'll meet in Phase 6; same name, unrelated concept, an unfortunate collision of terminology you just have to keep separate by context) is:
$$
\hat{T}_t = \frac{1}{m}\sum_{j=-k}^{k} x_{t+j}, \qquad m = 2k+1
$$
Plain English: to estimate the trend at time $t$, average the $k$ points before, $x_t$ itself, and the $k$ points after — a symmetric window centered exactly on $t$. This is why $m$ must be odd ($m=2k+1$) — an odd-width window has a natural center point.

**The even-order problem (why we "centered" in Phase 1).** If your seasonal period is even (like $m=4$ for quarterly data, or $m=12$ for monthly data), you can't center a window of that size on a single point — a 4-point window sits between two time points, not on one. The fix, which you already did by hand in Phase 1: compute the order-$m$ moving average, then average TWO adjacent ones together. This is called a **2×m moving average**, and it's the formal name for exactly the "average of adjacent 4-quarter averages" trick you performed in Phase 1, section 5. Its formula, for $m=12$ (monthly data, yearly seasonality):
$$
\hat{T}_t = \frac{1}{12}\Big(\tfrac{1}{2}x_{t-6} + x_{t-5}+\dots+x_{t+5}+\tfrac{1}{2}x_{t+6}\Big)
$$
Notice the endpoints get HALF weight — this is exactly what "averaging two adjacent 12-point windows" works out to algebraically, and it ensures the weights are symmetric and sum to 1.

**Why does averaging remove seasonality?** If the window width $m$ exactly equals the seasonal period, then every window contains exactly one copy of each season (one January, one February, ... one December, for $m=12$) — so the seasonal ups and downs inside the window cancel out in the average, leaving just the smooth trend behind. This is the formal justification for why we used a 4-quarter window for quarterly seasonal data in Phase 1.

**The unavoidable cost:** you lose $k$ points at each end of the series (you can't center a window on the very first or last few points, since there's nothing beyond the edge to average). This is a real practical limitation — production systems often need separate handling for the most recent, un-trend-estimable data points, since those are usually exactly the points you care about most for forecasting.

---

## 2. STL Decomposition (Seasonal-Trend decomposition using Loess) — conceptual mechanics

Classical moving-average decomposition (above) has weaknesses: it assumes the seasonal pattern is perfectly constant every single cycle (never evolving), and it can't handle irregular calendar effects well. **STL** is a more flexible, modern alternative.

**New word: Loess** (also written LOWESS) — "LOcally Estimated Scatterplot Smoothing." Plain English: instead of fitting ONE single trend line/curve to the WHOLE series at once (like ordinary regression would), Loess slides a small window across the data and, at each point, fits a small local regression using ONLY the nearby points inside that window, then moves on. The result is a smooth curve built from many small, local, overlapping fits rather than one global fit — this lets the "trend" bend and flex locally in ways a single global formula couldn't.

**STL's iterative loop, in plain English (no need to derive the full math, just understand the logic):**
1. Estimate a rough trend (via Loess smoothing).
2. Subtract it, then estimate the seasonal component from what's left (by Loess-smoothing each "same season" subseries, e.g., smoothing across all the Januaries, separately from all the Februaries — which is ALSO why STL allows the seasonal pattern to slowly evolve over time, unlike the classical method which forces it to be perfectly identical every cycle).
3. Subtract the seasonal estimate, re-estimate the trend from what's left (now cleaner, since seasonality is removed).
4. Repeat steps 1–3 a few times until the estimates stabilize.
5. Whatever remains after removing both trend and seasonal estimates is the remainder/irregular component.

**Why interviewers like asking about STL:** its key practical advantages over classical decomposition are (a) it allows seasonality to gradually CHANGE over time rather than being rigidly fixed, (b) it's robust to outliers (an optional robustness setting down-weights extreme remainder values so a single bad data point doesn't distort the trend/seasonal estimates), and (c) it works with any seasonal period, not just specific ones. The tradeoff: it's additive-only in its basic form (you'd log-transform first for multiplicative-feeling data, exactly as discussed in Phase 4, section 8).

---

## 3. Why do we need a WHOLE NEW family of methods (exponential smoothing) if we already have moving averages?

Moving averages are good for DESCRIBING/decomposing a series you already have. But they're awkward for **forecasting** — genuinely predicting a value you haven't seen yet — because a moving average by definition needs points on BOTH sides of the target, and the future doesn't exist yet to average over. We need a method built specifically to look only backward and project forward. That's exactly what exponential smoothing does.

---

## 4. Simple Exponential Smoothing (SES) — full derivation

**The core idea, built from scratch:** to forecast tomorrow, use a weighted average of all past observations — but give MORE weight to RECENT observations and progressively LESS weight to older ones, since recent data is usually more relevant to what happens next.

**The recursive (practical) formula, defined term by term:**
$$
\hat{x}_{t+1} = \alpha\, x_t + (1-\alpha)\,\hat{x}_t
$$
- $\hat{x}_{t+1}$ = our forecast for the NEXT time step, made using information available up through time $t$.
- $x_t$ = the actual observed value at time $t$ (the most recent real data point we have).
- $\hat{x}_t$ = our PREVIOUS forecast, i.e., what we had predicted for time $t$ before we actually observed it.
- $\alpha$ (alpha) = the **smoothing parameter**, a single tunable number between 0 and 1 — the entire "knob" that controls this whole method.

**Plain English reading:** "Next period's forecast = a blend of (the newest actual data point) and (what we previously thought would happen), weighted by $\alpha$." If $\alpha$ is close to 1, the forecast reacts strongly and quickly to the newest data point (very responsive, but jumpy/noisy). If $\alpha$ is close to 0, the forecast barely moves from its previous prediction, changing very slowly (smooth but slow to react to real changes). **$\alpha$ is exactly the dial that trades off responsiveness against smoothness** — a concept you'll see echoed constantly across machine learning (it's structurally identical to a "learning rate" or an exponential moving average decay parameter, if you've encountered those elsewhere).

**Unrolling the recursion — proving it really IS a weighted average of ALL past data, with exponentially decaying weights (this is where the name comes from):**

Start with the recursion and substitute repeatedly:
$$
\hat{x}_{t+1} = \alpha x_t + (1-\alpha)\hat{x}_t
$$
Now substitute the definition of $\hat{x}_t = \alpha x_{t-1} + (1-\alpha)\hat{x}_{t-1}$ into the equation above:
$$
\hat{x}_{t+1} = \alpha x_t + (1-\alpha)\big[\alpha x_{t-1} + (1-\alpha)\hat{x}_{t-1}\big] = \alpha x_t + \alpha(1-\alpha)x_{t-1} + (1-\alpha)^2\hat{x}_{t-1}
$$
If you keep substituting further back (one more round shown, then generalize):
$$
\hat{x}_{t+1} = \alpha x_t + \alpha(1-\alpha)x_{t-1} + \alpha(1-\alpha)^2 x_{t-2} + \alpha(1-\alpha)^3 x_{t-3} + \dots
$$
**Look at the pattern of weights:** $\alpha$, then $\alpha(1-\alpha)$, then $\alpha(1-\alpha)^2$, then $\alpha(1-\alpha)^3$... Each older observation's weight is multiplied by another factor of $(1-\alpha)$ — **the weights shrink exponentially/geometrically as you go further into the past.** This is the literal, formal reason it's called "exponential" smoothing: every past observation DOES still contribute to the forecast (nothing is ever fully thrown away, unlike a plain moving average which uses a hard cutoff window), but its influence fades away exponentially the older it gets.

**Sanity check the weights sum to 1** (a requirement for this to be a legitimate weighted average): $\alpha + \alpha(1-\alpha) + \alpha(1-\alpha)^2 + \dots = \alpha \sum_{j=0}^{\infty}(1-\alpha)^j = \alpha \cdot \frac{1}{1-(1-\alpha)} = \alpha \cdot \frac{1}{\alpha} = 1$ ✓ (using the standard geometric series sum formula $\sum_{j=0}^\infty r^j = \frac{1}{1-r}$ for $|r|<1$, which applies here since $0<1-\alpha<1$).

**Choosing $\alpha$ in practice:** rather than picking it by hand, $\alpha$ (and the starting value $\hat{x}_1$) are typically chosen by **minimizing the sum of squared forecast errors** on your historical data — i.e., trying different $\alpha$ values and picking whichever one would have produced the smallest historical forecasting mistakes. This is a genuine optimization procedure (usually solved numerically), foreshadowing the Maximum Likelihood Estimation approach we'll use for ARIMA in Phase 6.

**Critical limitation of SES: it can ONLY produce a FLAT forecast line into the future.** Because the recursive formula only ever updates a single "level" estimate, if you ask it to forecast 10 steps ahead, EVERY one of those 10 forecasts will be the exact same flat number — $\hat{x}_{t+1} = \hat{x}_{t+2} = \dots$. **SES has no concept of trend or seasonality whatsoever** — it assumes the series only wanders around a slowly-updating flat level. This limitation is exactly what motivates the next two methods.

---

## 5. Holt's Linear Trend Method — adding a trend component

**The idea:** keep SES's "smoothed level" idea, but ALSO maintain a second, separately-smoothed estimate of the TREND (the slope/rate of change), and combine both to forecast.

**The two update equations, defined term by term:**

**Level equation:**
$$
\ell_t = \alpha\, x_t + (1-\alpha)(\ell_{t-1} + b_{t-1})
$$
**Trend equation:**
$$
b_t = \beta^*(\ell_t - \ell_{t-1}) + (1-\beta^*)\,b_{t-1}
$$

Breaking down every new symbol:
- $\ell_t$ (script L) = the smoothed LEVEL estimate at time $t$ (analogous to $\hat{x}_t$ in SES, but now explicitly separated from trend).
- $b_t$ = the smoothed TREND (slope) estimate at time $t$ — literally "how much is the level currently rising or falling per step."
- $\beta^*$ (beta-star; the asterisk is just standard notation to distinguish it from an unrelated $\beta$ used elsewhere in statistics, no deeper meaning) = a SECOND smoothing parameter, between 0 and 1, controlling how quickly the TREND estimate adapts to new evidence — playing the same "responsiveness vs. smoothness" role that $\alpha$ played for the level.

**Plain English reading of the level equation:** "Our new level estimate blends the newest actual observation with what we would have PREDICTED for this step using our OLD level plus OLD trend ($\ell_{t-1}+b_{t-1}$, i.e., 'where we were, plus how fast we were moving')." This is exactly SES's logic, except the "previous forecast" term now accounts for trend instead of assuming a flat line.

**Plain English reading of the trend equation:** "Our new trend estimate blends the MOST RECENTLY OBSERVED change in level ($\ell_t - \ell_{t-1}$, i.e., how much the level estimate just moved) with our OLD trend estimate." This is literally SES's exact same smoothing logic, just applied to the SLOPE instead of the level — Holt's method is really just "run SES twice: once on the level, once on the trend of that level."

**Forecasting $h$ steps ahead:**
$$
\hat{x}_{t+h} = \ell_t + h\, b_t
$$
Plain English: "take the current level, and project forward in a straight line using the current estimated slope, for $h$ steps." Unlike SES's flat forecast, Holt's method produces a forecast that's a straight, sloped line extending into the future — fixing SES's core limitation, but note it's now a FIXED straight-line projection (which can be a problem if you forecast very far ahead and the real trend doesn't hold up that long — a real practical caveat).

---

## 6. Holt-Winters Seasonal Method — adding seasonality on top

**The idea:** take Holt's level + trend structure, and add a THIRD smoothed component tracking seasonality. We'll present the ADDITIVE version (recall Phase 1, section 4.1 — additive means seasonal effects are constant absolute amounts).

**Three update equations:**

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
- $m$ = the seasonal period (e.g., 12 for monthly-with-yearly-seasonality, 4 for quarterly, 7 for daily-with-weekly-seasonality — same $m$ as in section 1).
- $\gamma$ (gamma — yet another reused Greek letter, unrelated to the autocovariance $\gamma(k)$ from Phase 3 or the ADF regression coefficient from Phase 4; context is everything in this notation) = a THIRD smoothing parameter, between 0 and 1, controlling how fast the seasonal pattern is allowed to adapt/evolve over time.

**Plain English for the level equation:** notice $x_t - s_{t-m}$ — before updating the level, we first strip out the seasonal effect from the most recent observation (using the seasonal estimate from exactly ONE FULL CYCLE ago, $s_{t-m}$, since that's our best current guess for "this same point in the seasonal cycle"). This gives a deseasonalized observation to blend into the level, exactly as you did manually in Phase 1's seasonal adjustment exercise.

**Plain English for the seasonal equation:** $x_t - \ell_t$ is "how far the actual observation deviates from the current smoothed level" — that deviation IS the current seasonal effect, freshly observed. We blend this fresh evidence with the PREVIOUS estimate of the seasonal effect at this same point in the cycle ($s_{t-m}$, again one full cycle back), controlled by $\gamma$.

**Forecasting $h$ steps ahead:**
$$
\hat{x}_{t+h} = \ell_t + h\, b_t + s_{t-m+h \bmod m}
$$
Plain English: take the trend-projected level (same as Holt's method), then ADD BACK the appropriate seasonal effect for whichever point in the seasonal cycle the forecast horizon lands on (the $\bmod$ — "modulo" — operation just wraps the index around so it correctly cycles back to the right season, e.g., forecasting 14 steps ahead with $m=12$ lands on "season 2," the same relative position as February if $m=12$ started at January).

**The multiplicative version** simply swaps subtraction for division and addition for multiplication in each formula ($x_t/s_{t-m}$ instead of $x_t - s_{t-m}$; $s_t = \gamma \frac{x_t}{\ell_t} + (1-\gamma)s_{t-m}$; forecast $= (\ell_t + hb_t)\times s_{t-m+h\bmod m}$) — use this version when the seasonal swing size scales with the trend level (megaphone shape, Phase 1 section 4.2/Phase 4 section 8).

---

## 7. Connecting back: this is why it's called the "ETS" framework

Modern software (like R's `ets()` function) organizes this ENTIRE family systematically by naming each method with three letters: **E**rror type, **T**rend type, **S**easonal type — each can be **N** (none), **A** (additive), or **M** (multiplicative). So:
- SES = ETS(A,N,N) roughly speaking (additive errors, no trend, no seasonality)
- Holt's method = ETS(A,A,N) (additive errors, additive trend, no seasonality)
- Holt-Winters additive = ETS(A,A,A)
- Holt-Winters multiplicative = ETS(A,A,M)

This is worth recognizing by name since it's the standard modern vocabulary (Hyndman's taxonomy, referenced in the original syllabus) even though we've been deriving the "classical" versions above — the ETS framework additionally reformulates all of these as formal state-space models (giving them proper likelihood functions, automatic parameter/model selection via AIC, and genuine prediction intervals) which we'll connect to explicitly in Phase 9 (state space models & Kalman filtering).

---

## 8. Full numerical worked example: Holt's Linear Trend Method by hand

Data: 5 points, clearly trending: $x = [20, 24, 27, 32, 35]$

**Initialization** (a practical necessity — the recursion needs a starting point before it can run):
$\ell_1 = x_1 = 20$ (a common simple starting choice: just use the first observation as the initial level)
$b_1 = x_2 - x_1 = 24 - 20 = 4$ (a common simple starting choice: use the first observed difference as the initial trend)

Let's use $\alpha = 0.5$ and $\beta^* = 0.3$.

**Step $t=2$:**
$\ell_2 = \alpha x_2 + (1-\alpha)(\ell_1+b_1) = 0.5(24) + 0.5(20+4) = 12 + 12 = 24.0$
$b_2 = \beta^*(\ell_2-\ell_1) + (1-\beta^*)b_1 = 0.3(24-20) + 0.7(4) = 0.3(4)+2.8 = 1.2+2.8=4.0$

**Step $t=3$:**
$\ell_3 = 0.5(27) + 0.5(24+4.0) = 13.5+14.0=27.5$
$b_3 = 0.3(27.5-24)+0.7(4.0) = 0.3(3.5)+2.8=1.05+2.8=3.85$

**Step $t=4$:**
$\ell_4 = 0.5(32)+0.5(27.5+3.85)=16+15.675=31.675$
$b_4 = 0.3(31.675-27.5)+0.7(3.85)=0.3(4.175)+2.695=1.2525+2.695=3.9475$

**Step $t=5$:**
$\ell_5 = 0.5(35)+0.5(31.675+3.9475)=17.5+17.81125=35.31125$
$b_5 = 0.3(35.31125-31.675)+0.7(3.9475)=0.3(3.63625)+2.76325=1.090875+2.76325=3.854125$

**Forecast for $t=6, 7, 8$ (i.e., $h=1,2,3$ steps ahead from $t=5$):**
$\hat{x}_6 = \ell_5 + 1\cdot b_5 = 35.31125+3.854125 = 39.165375$
$\hat{x}_7 = \ell_5 + 2\cdot b_5 = 35.31125+7.70825 = 43.0195$
$\hat{x}_8 = \ell_5 + 3\cdot b_5 = 35.31125+11.562375=46.873625$

**Sanity check against the raw data:** our original series rose by roughly 4, 3, 5, 3 per step (differences: 24-20=4, 27-24=3, 32-27=5, 35-32=3) — averaging around +3.75/step. Our final smoothed trend estimate $b_5 \approx 3.85$ is right in that same ballpark, and the forecasts extend the series forward at a similar, slightly-smoothed rate. This is the correct sanity check to always perform: does the model's implied trend roughly match what you can see just by looking at the raw differences?

---

## 9. Quick self-check questions

1. Why can Simple Exponential Smoothing (SES) never produce anything other than a flat forecast line?
   *(Answer: SES only maintains a single smoothed LEVEL, with no separate mechanism to estimate or project a slope/trend — every future forecast just repeats the same current level estimate.)*
2. In the geometric-series expansion of SES (section 4), what happens to the weight given to an observation from 10 periods ago as $\alpha$ gets closer to 1?
   *(Answer: the weight on older observations shrinks even FASTER — since the decay factor is $(1-\alpha)$, a larger $\alpha$ means a smaller $(1-\alpha)$, so weights collapse toward zero more quickly for older data; effectively, only the very most recent observations matter when $\alpha$ is close to 1.)*
3. In Holt-Winters, why does the level equation use $s_{t-m}$ (seasonal estimate from one full cycle ago) instead of, say, the most recently estimated seasonal value $s_{t-1}$?
   *(Answer: because the correct seasonal adjustment for "this point in the cycle" (e.g., "this December") is only found at exactly one full cycle back — $s_{t-1}$ would be last month's seasonal effect, e.g., November's, which is the wrong season entirely.)*
4. What do the three letters in ETS(A,A,A) stand for, and what method does it correspond to among the ones covered in this phase?
   *(Answer: Error type, Trend type, Seasonal type, all Additive — this corresponds to the additive Holt-Winters seasonal method.)*

---

## What's next
Phase 6 is the big one: the full **Box-Jenkins methodology** — AR, MA, ARMA, ARIMA, and SARIMA models, derived rigorously from first principles (stationarity conditions, invertibility, the Yule-Walker equations promised back in Phase 3, Maximum Likelihood estimation, AIC/BIC model selection, and full residual diagnostics), plus the formal proof of the ACF/PACF signature table from Phase 3, section 6. This phase alone typically carries the largest share of interview weight in classical time series, so we'll take it in careful, well-paced sub-steps rather than all at once.

Say "next" for Phase 6, or ask for more Holt/Holt-Winters hand-computation drills first.
