# Phase 6, Part 5 of 5: Diagnostics & Forecasting

Roadmap: 6.1 AR(p) → 6.2 MA(q) → 6.3 ARMA/ARIMA/SARIMA → 6.4 Estimation & model selection → **6.5 Diagnostics & forecasting (this file, final part)**.

You now know how to identify a model's shape (Parts 1-3) and fit its coefficients (Part 4). This file closes the loop: **how do you check the fitted model is actually good, and how do you turn it into an actual forecast with honest uncertainty?** This is literally the last step of the "Box-Jenkins methodology" named in the original syllabus.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $\hat{e}_t$ | the **residual** at time $t$ — the leftover gap between what the model predicted and what actually happened |
| $\hat{x}_t$ | the model's fitted/predicted value at time $t$ |
| $\hat{x}_{t+h}$ | the forecast made $h$ steps into the future, from the current point $t$ |
| $h$ | the **forecast horizon** — how many steps ahead we're predicting |
| $Q$ | the Ljung-Box test statistic (recall from Phase 3, section 8) |
| $\text{SE}$ | standard error — a measure of how uncertain an estimate is |
| $z_{0.975}$ | a fixed number (1.96) used to build a 95% interval — same constant from Phase 3's ACF confidence bands |

---

## 2. What is a "residual," precisely?

**Plain English:** once you've fit a model, you can ask, at each historical time point, "what did my model SAY would happen here?" ($\hat{x}_t$) versus "what ACTUALLY happened?" ($x_t$). The gap between them is the residual:
$$
\hat{e}_t = x_t - \hat{x}_t
$$
**Why residuals matter so much:** recall the entire philosophy from Phase 2 — a good model should extract ALL the predictable structure out of the data, leaving behind ONLY pure, unpredictable randomness (white noise). **So the residuals are your report card: if they still contain visible pattern/structure, your model missed something and needs to be improved. If they look like structureless white noise, your model has done its job.**

---

## 3. Checking residuals: the full diagnostic checklist

### 3.1 Plot the residuals over time
**What to look for:** the residuals should bounce randomly around zero, with roughly constant spread throughout (no widening/narrowing "megaphone" shape — recall this shape from Phase 1/Phase 4, which would signal non-constant variance, foreshadowing GARCH models much later in the full syllabus). If you instead see any remaining trend, cyclical wave, or sudden shift in spread — the model missed something.

### 3.2 Plot the ACF of the residuals (directly reusing Phase 3!)
**What to look for:** if the model is good, the residual ACF should look like white noise's ACF — essentially zero at every lag (all bars inside the confidence band from Phase 3, section 4). **If you see a significant bar at, say, lag 2 — that's a direct, specific signal: there's still a real lag-2 relationship your model failed to capture, and you should consider increasing $p$ or $q$ to absorb it.** This is a genuinely actionable diagnostic, not just a pass/fail check — the LOCATION of any remaining significant lag tells you WHERE to expand the model.

### 3.3 The Ljung-Box test, formally applied here (completing the promise from Phase 3, section 8)
Recall the formula:
$$
Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho(k)^2}{n-k}
$$
**Now applied specifically to the RESIDUALS** (rather than the raw data, which is how we introduced it back in Phase 3): $\hat\rho(k)$ here means the sample ACF of the residual sequence $\hat{e}_t$, not the original $x_t$.
- **Null hypothesis: the residuals are white noise (no leftover structure).**
- **If the p-value is small (< 0.05): REJECT — bad news, your residuals still have real structure, go back and refine the model (increase $p$/$q$, reconsider $d$, check for a missed seasonal component).**
- **If the p-value is large: FAIL TO REJECT — good news, no strong evidence of leftover structure; your model has plausibly captured what's extractable.**

**One important subtlety worth knowing:** when applying Ljung-Box to residuals (rather than raw data), the degrees of freedom of the reference Chi-squared distribution are adjusted downward by the number of parameters you already estimated ($p+q$) — because you've already "used up" some of the data's information fitting those parameters, so the test needs to account for that, similar in spirit to how ordinary regression adjusts degrees of freedom for the number of predictors used. You don't need to hand-derive this adjustment — software handles it automatically — just recognize WHY it happens if asked.

### 3.4 Normality check (Q-Q plot)
**New term: Q-Q plot** ("quantile-quantile" plot) — a simple diagnostic chart comparing your residuals' actual distribution shape against a perfect theoretical Normal/bell-curve distribution. **Plain English: if you sort your residuals from smallest to largest and plot them against where a perfect bell curve's values WOULD have landed at the same percentile ranks, a good fit shows points falling roughly along a straight diagonal line.** Curving away from that line (especially at the ends) signals **fat tails** (more extreme large/small residuals than a bell curve would predict — genuinely common in financial data, foreshadowing the GARCH volatility models mentioned in the wider syllabus) or skew. **Why this matters practically:** the prediction intervals we're about to derive in section 5 rely on a Normal-distribution assumption — if residuals are clearly non-Normal, those intervals will be systematically too narrow or miscalibrated, understating real risk.

---

## 4. Point forecasting: deriving the actual predicted numbers

**The core idea, in one plain sentence:** to forecast the future, take your fitted model's formula, and for anything you don't know yet (future noise terms), use your BEST GUESS — which for white noise is always exactly 0 (recall Phase 2: white noise has zero mean, so the "expected"/best-guess value of any FUTURE, not-yet-happened shock is always 0).

### 4.1 One-step-ahead forecast, worked from an AR(1) example
Take a fitted AR(1) model: $\hat{x}_t = \hat\phi\, x_{t-1}$ (dropping any constant for simplicity, using our fitted $\hat\phi$ from Part 4). To forecast ONE step beyond our last observed point $x_T$ (T = "the last time point we actually have data for"):
$$
\hat{x}_{T+1} = \hat\phi\, x_T
$$
**Plain English:** just plug the LAST KNOWN real value into the fitted formula — completely mechanical, no new concept here.

### 4.2 Two-steps-ahead: this is where it gets interesting
$$
\hat{x}_{T+2} = \hat\phi\, x_{T+1}
$$
**But we don't KNOW $x_{T+1}$ yet — it hasn't happened!** So we substitute in our own FORECAST of it instead, from section 4.1:
$$
\hat{x}_{T+2} = \hat\phi\, \hat{x}_{T+1} = \hat\phi(\hat\phi\, x_T) = \hat\phi^2\, x_T
$$
**Plain English: forecasting further into the future means FEEDING YOUR OWN EARLIER FORECASTS back into the formula as if they were real data.** This is a genuinely important practical point: **multi-step forecasts are built recursively, each one leaning on the previous forecast rather than real observed data** — this is exactly why forecasts get progressively less reliable/more uncertain the further out you go (a concept you'll see precisely quantified in the very next section).

**General pattern (AR(1)):** $\hat{x}_{T+h} = \hat\phi^h\, x_T$. **Notice something familiar:** as $h$ grows, since $|\hat\phi|<1$ (stationarity, Part 1), $\hat\phi^h \to 0$ — **meaning the forecast smoothly converges toward zero (or toward $\mu$, the series' long-run mean, if a nonzero constant/mean was included) as the horizon grows.** This is a genuinely important, intuitive property: **a stationary AR model's forecast eventually just settles down to the series' long-run average, the further ahead you look — because any specific recent shock's influence fades away geometrically (exactly the "pulled back toward center" mean-reverting behavior we first saw numerically back in Part 1, section 6), and with nothing else to go on, the best guess for the distant future is just "the average."**

**Contrast this directly with the random walk (Phase 2) and Holt's method (Phase 5):** a random walk's forecast for ANY horizon is just a FLAT line at the last observed value (since $\phi=1$ exactly, so $\hat\phi^h=1^h=1$ for every $h$ — never decaying), and Holt's method's forecast was a straight SLOPED line extending forever (Phase 5, section 5). **A stationary AR/ARMA model's forecast instead curves and FLATTENS toward the mean** — three genuinely different long-run forecast SHAPES, directly explainable from what you've now derived across three different phases. Recognizing which shape a model implies, just from knowing its type, is a strong practical/interview skill.

---

## 5. Prediction intervals: quantifying honest uncertainty

**Why a single point forecast number is never the whole story:** recall from the very start of this whole curriculum (Phase 2) — a time series comes from a random PROCESS. Even a perfect model can't predict the exact future number, only a most-likely CENTER, with genuine random spread around it. **A responsible forecast always comes with a range, not just a single number.**

### 5.1 Deriving the forecast error variance (AR(1) case, building directly on section 4)
**Define the forecast error** at horizon $h$ as the (unknown, future) gap between what actually happens and what we forecast: $e_{T+h} = x_{T+h} - \hat{x}_{T+h}$.

For $h=1$ (using AR(1)): $x_{T+1} = \phi\, x_T + \varepsilon_{T+1}$, and our forecast was $\hat{x}_{T+1}=\hat\phi\, x_T$. If we assume our ESTIMATED $\hat\phi$ happens to equal the true $\phi$ (a simplification real analysis relaxes, but it's the right starting point to build intuition), the forecast error is:
$$
e_{T+1} = x_{T+1}-\hat{x}_{T+1} = \varepsilon_{T+1}
$$
**Plain English: the one-step-ahead forecast error is JUST the next period's fresh, unpredictable noise shock — which makes complete sense, since that's the ONE piece of information that genuinely could not have been known in advance.** So $\text{Var}(e_{T+1}) = \sigma^2$ (the noise variance, directly).

**For $h=2$**, unroll the same way as section 4.2:
$$
e_{T+2} = x_{T+2}-\hat{x}_{T+2} = (\phi x_{T+1}+\varepsilon_{T+2}) - \phi\hat{x}_{T+1} = \phi(x_{T+1}-\hat{x}_{T+1}) + \varepsilon_{T+2} = \phi\, e_{T+1} + \varepsilon_{T+2} = \phi\varepsilon_{T+1}+\varepsilon_{T+2}
$$
**This is EXACTLY the same "sum of weighted past shocks" structure you derived in Part 1, section 2 (the MA(∞) representation)!** Using the same independent-variance-adds-up rule from Phase 2/Part 1:
$$
\text{Var}(e_{T+2}) = \phi^2\sigma^2 + \sigma^2 = \sigma^2(1+\phi^2)
$$
**General pattern (matches the structure you've now derived multiple times across this course):**
$$
\text{Var}(e_{T+h}) = \sigma^2(1+\phi^2+\phi^4+\dots+\phi^{2(h-1)}) = \sigma^2\sum_{j=0}^{h-1}\phi^{2j}
$$
**Plain English, the key practical takeaway: forecast uncertainty GROWS with the horizon $h$ — you're always LESS certain about further-out forecasts than near-term ones — but for a STATIONARY model, this growth eventually levels off** (as $h\to\infty$, this sum converges to $\sigma^2/(1-\phi^2)$ — recognize this immediately: **it's exactly the AR(1) unconditional variance formula you derived in Part 1, section 2!** This makes complete intuitive sense: once you're forecasting far enough ahead that the current data provides essentially no more useful information, your best-possible uncertainty is simply the process's own natural, ordinary variance — you're no better off than just guessing from the unconditional distribution.).

**Contrast with a random walk's forecast uncertainty (connecting back to Phase 2, section 5.1):** for a random walk ($\phi=1$), this exact same sum becomes $\text{Var}(e_{T+h}) = h\sigma^2$ — **growing WITHOUT BOUND as $h\to\infty$, never leveling off** — precisely matching Phase 2's original "widening cone of uncertainty" result. **This is a genuinely deep, recurring pattern worth noticing explicitly: stationary models have forecast uncertainty that levels off at a ceiling; non-stationary (unit root) models have forecast uncertainty that grows forever, unboundedly** — a real, practical, and often-tested distinction.

### 5.2 Building the actual interval
Once you have $\text{Var}(e_{T+h})$, the standard error is just its square root: $\text{SE}(h) = \sqrt{\text{Var}(e_{T+h})}$. Assuming (per section 3.4's Normality check) the errors are roughly Normally distributed, a 95% prediction interval is built using the SAME constant 1.96 from Phase 3's ACF confidence bands (not a coincidence — same underlying "95% of a bell curve's mass lies within ±1.96 standard deviations" fact, reused everywhere in statistics):
$$
\hat{x}_{T+h} \pm 1.96\times \text{SE}(h)
$$
**Plain English: the forecast interval is the point forecast, plus-or-minus about two standard errors, WIDENING as $h$ grows (per section 5.1) — visually, this produces exactly the "widening cone/fan" shape you'll see on any real forecast chart**, now fully explained mathematically rather than just visually recognized.

---

## 6. Numerical worked example: a full forecast with interval, by hand

Take our fitted AR(1) from Part 1, section 6: $\hat\phi = 0.6$, $\sigma^2=1$, and suppose the last observed value is $x_T = -0.1104$ (reusing that exact number from Part 1's simulated series, so you can trace the SAME numbers across both files).

**Point forecasts** (section 4.2's general formula, $\hat{x}_{T+h}=\hat\phi^h x_T$):
$\hat{x}_{T+1} = 0.6^1 \times(-0.1104) = -0.06624$
$\hat{x}_{T+2} = 0.6^2\times(-0.1104) = 0.36\times(-0.1104)=-0.039744$
$\hat{x}_{T+3} = 0.6^3\times(-0.1104)=0.216\times(-0.1104)\approx -0.023846$

**Notice the forecasts are already rapidly shrinking toward 0 (the process mean here, since we assumed $c=0$/mean-centered back in Part 1) — exactly the "flattens toward the mean" behavior predicted in section 4.2.**

**Forecast variances** (section 5.1's formula, $\text{Var}(e_{T+h})=\sigma^2\sum_{j=0}^{h-1}\phi^{2j}$):
$h=1$: $\text{Var} = 1\times(0.6^0) = 1\times 1 = 1.0$ → $\text{SE}=1.0$
$h=2$: $\text{Var}=1\times(0.6^0+0.6^2)=1\times(1+0.36)=1.36$ → $\text{SE}=\sqrt{1.36}\approx 1.166$
$h=3$: $\text{Var}=1\times(1+0.36+0.6^4)=1\times(1+0.36+0.1296)=1.4896$ → $\text{SE}=\sqrt{1.4896}\approx 1.2205$

**Notice: variance is growing but CLEARLY slowing down/leveling off** (jump from $h{=}1\to2$ is $+0.36$; from $h{=}2\to3$ is only $+0.1296$) — heading toward the ceiling value $\sigma^2/(1-\phi^2) = 1/0.64=1.5625$ that we ALSO computed independently in Part 1, section 6, as the process's unconditional variance — the two numbers converging is not a coincidence, it's the exact mathematical relationship proven in section 5.1 above.

**95% prediction intervals** (point forecast $\pm 1.96\times$SE):
$h=1$: $-0.06624 \pm 1.96(1.0) = -0.06624\pm 1.96 \Rightarrow (-2.026, 1.894)$
$h=2$: $-0.039744\pm 1.96(1.166) = -0.039744\pm 2.285 \Rightarrow (-2.325, 2.245)$
$h=3$: $-0.023846\pm1.96(1.2205)=-0.023846\pm 2.392\Rightarrow (-2.416,2.368)$

**Full interpretation, tying everything together:** the point forecasts converge quickly toward 0 (the mean), while the intervals widen but are clearly approaching a ceiling width, rather than expanding forever — the complete, quantified signature of a stationary AR(1) forecast, exactly as derived from first principles above, and a dramatically different shape than you would get forecasting a random walk (Phase 2) or a Holt's-method trend line (Phase 5) from the same starting point.

---

## 7. Quick self-check questions

1. What should the ACF of a well-fitted model's residuals look like, and what does a significant bar at a specific lag tell you to actually DO?
   *(Answer: it should look like white noise — all bars inside the confidence band. A significant bar at a specific lag tells you there's real leftover structure at that lag, suggesting you should increase p or q to try to capture it.)*
2. Why does forecasting 3 steps ahead require plugging your OWN 1-step and 2-step forecasts back into the model, rather than using real data?
   *(Answer: because the real future values at T+1 and T+2 haven't happened yet by the time you're standing at T trying to forecast T+3 — so the best available substitute is your own best-guess forecast for those unobserved future points, recursively.)*
3. Why does a stationary AR(1) model's prediction interval eventually stop widening as the horizon increases, while a random walk's interval keeps widening forever?
   *(Answer: AR(1)'s forecast error variance is a geometric series that converges to a finite ceiling (σ²/(1-φ²), the same as the process's own unconditional variance) because past shocks' influence decays by a factor of φ each step; the random walk has φ=1, so shocks never decay, and the error variance grows linearly (hσ²) without any ceiling.)*
4. If a Q-Q plot shows residuals curving away from the diagonal line at the extreme ends, what practical problem does this cause for your prediction intervals?
   *(Answer: it signals the residuals have "fat tails" — more extreme large/small values than a true Normal distribution would produce — meaning intervals built assuming Normality (using the 1.96 constant) will likely be too narrow, understating the real chance of an extreme outcome.)*

---

## Phase 6 complete — full recap of what you can now do

Across these five parts, you can now: identify candidate models from ACF/PACF shapes (Parts 1-2, with full derivations, not memorization); understand why ARMA is a theoretically general, justified building block (Part 3's Wold theorem); estimate parameters via Yule-Walker or MLE, and understand precisely what MLE is doing (Part 4); select between candidate models using AIC/BIC with real trade-off reasoning (Part 4); and run a full residual diagnostic + generate honest point forecasts WITH prediction intervals, all derived from first principles rather than treated as a black box (Part 5, this file). **This is the complete, classical Box-Jenkins cycle — genuinely the single highest-weight topic block in time series interviews**, and you've now built every piece of it from the ground up.

## What's next
Phase 7 moves to **Regression With Time Series Data** — the STAT 510 core material on using time series alongside REGRESSION (predicting $x_t$ from OTHER variables, not just its own past), the danger of autocorrelated regression errors, spurious regression, and a first look at cointegration (fully developed later in the multivariate phase).

Say "next" for Phase 7, or ask for more forecasting/prediction-interval drilling first — or feel free to request a consolidated "Phase 6 formula cheat-sheet" file pulling every key formula from all 5 parts into one quick-reference page, which can be genuinely useful for last-minute interview review.
