# Phase 8: Complex Seasonality — Fourier Terms, TBATS, and Prophet Internals

Phase 5 gave you Holt-Winters, which handles ONE seasonal pattern (one fixed $m$). Phase 6's SARIMA handles one seasonal lag $s$ too. Real data at companies like Google or Apple often has SEVERAL overlapping seasonal patterns at once — e.g., web traffic that's higher on weekdays (weekly pattern) AND higher during business hours (daily pattern) AND higher in Q4 (yearly pattern), all simultaneously. This phase builds the tools for that, and ends with a full plain-English breakdown of how Meta's Prophet model — a genuinely popular production tool — is built entirely from pieces you already know.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $K$ | the number of Fourier term PAIRS used to approximate a seasonal shape (more pairs = more flexible/wiggly shape) |
| $m$ | the seasonal period (same meaning as Phase 5's Holt-Winters $m$) |
| $\sin, \cos$ | sine and cosine — the standard wave functions from trigonometry, used here as building-block shapes |
| $\beta_k$ | a regression coefficient attached to one specific Fourier wave term |
| $g(t)$ | Prophet's trend function (piecewise, explained in section 5) |
| $s(t)$ | Prophet's seasonal function (built from Fourier terms) |
| $h(t)$ | Prophet's holiday-effect function |

---

## 2. Why can't Holt-Winters / SARIMA just handle two seasonal periods at once?

**The core structural limitation, in plain English:** both Holt-Winters (Phase 5) and SARIMA (Phase 6) are built around ONE single seasonal period baked directly into their formulas ($m$ in Holt-Winters, $s$ in SARIMA). There's no natural slot in either formula for "also track a SECOND, differently-sized repeating cycle." You could try to bolt on a second seasonal term, but the formulas get unwieldy fast, and — a genuinely practical problem — **if your seasonal period is very large (e.g., $m=365$ for daily data with yearly seasonality), Holt-Winters and SARIMA need to estimate a SEPARATE parameter for almost every single day of the pattern, which is a huge number of parameters to estimate from limited data.** We need an approach that can represent a seasonal SHAPE using far fewer numbers, and that naturally handles multiple overlapping periods. That's exactly what Fourier terms give us.

---

## 3. Building Fourier terms from scratch: representing ANY repeating shape with waves

**The core mathematical fact this all relies on (stated in plain English, no need to prove it — this is a genuinely deep result called Fourier's theorem):** *any* repeating/periodic pattern, no matter how oddly-shaped, can be built up by ADDING TOGETHER enough simple sine and cosine waves of different speeds (frequencies). Think of it like mixing paint colors: with enough of the right basic wave "ingredients," combined in the right amounts, you can reconstruct almost any repeating shape you want.

**The formula for a set of Fourier terms representing seasonality with period $m$:**
$$
s(t) = \sum_{k=1}^{K}\left[\beta_{k}\sin\left(\frac{2\pi k t}{m}\right) + \gamma_{k}\cos\left(\frac{2\pi k t}{m}\right)\right]
$$

**Breaking this down piece by piece, very slowly:**
- $\sin\left(\frac{2\pi t}{m}\right)$ (the $k=1$ term) is a single smooth wave that completes EXACTLY one full up-and-down cycle every $m$ time steps — e.g., if $m=365$ (daily data, yearly seasonality), this wave rises and falls once per year, a simple, gentle single hump-and-dip shape.
- $\sin\left(\frac{2\pi \cdot 2\cdot t}{m}\right)$ (the $k=2$ term) is a FASTER wave — it completes exactly TWO full cycles within the same $m$ time steps (twice as fast). Adding this to the $k=1$ wave lets you build slightly more complex shapes — e.g., a pattern with two bumps per year instead of one smooth hump.
- Each successive $k$ adds an even faster wave, letting the combination capture finer and finer detail in the seasonal shape.
- We use BOTH sine and cosine (not just sine alone) at each frequency because sine and cosine are the same wave shape just SHIFTED in time relative to each other — using both together lets the combination represent a wave PEAKING at any arbitrary point in the cycle (not just where a pure sine wave happens to peak), giving the model full flexibility to align with wherever the real data's peak actually falls.
- $\beta_k, \gamma_k$ are ordinary regression coefficients (Phase 7's machinery, directly reused!) — literally, you can estimate these using standard linear regression, treating each sine/cosine term as just another predictor column, exactly like any other regressor in Phase 7.

**Why is $K$ (the number of wave-pairs used) the key tuning knob?** **Plain English: $K$ controls how WIGGLY/detailed the reconstructed seasonal shape is allowed to be.** $K=1$ gives just one smooth hump-shaped wave — a simple, single-peak seasonal pattern. Larger $K$ allows sharper, more detailed, multi-peaked seasonal shapes. **This is a direct, familiar bias-variance tradeoff (Phase 0/general ML knowledge): too small a $K$ underfits the true seasonal shape (misses real detail); too large a $K$ overfits (starts fitting to noise, producing an oddly jagged "seasonal" pattern that doesn't generalize)** — chosen in practice using the exact same AIC/BIC logic from Phase 6, Part 4, since $K$ directly determines the parameter count.

**The single biggest practical advantage over Holt-Winters/SARIMA's approach, worth restating clearly: you can represent a FULL YEAR of daily seasonality (365 potential "days" worth of pattern) using maybe just $K=3$ to $K=10$ wave-pairs (6 to 20 total coefficients) — a HUGE reduction in the number of parameters needed, compared to estimating something close to 365 separate seasonal index values directly.** This is called **dynamic harmonic regression** when you combine Fourier seasonal terms with an ARIMA structure for whatever's left over (directly reusing Phase 7, section 8's "regression with ARIMA errors" idea — Fourier terms are just another kind of regressor, plugged into that exact same framework).

---

## 4. Handling MULTIPLE seasonal periods at once: just add more Fourier blocks

**This is the genuinely elegant payoff of the whole approach:** since Fourier terms are just ordinary regression predictors (section 3), handling multiple overlapping seasonalities is as simple as adding a SEPARATE block of Fourier terms for EACH seasonal period, all in the SAME regression simultaneously:
$$
s(t) = \underbrace{\sum_{k=1}^{K_1}\left[\beta_k\sin\left(\tfrac{2\pi k t}{m_1}\right)+\gamma_k\cos\left(\tfrac{2\pi k t}{m_1}\right)\right]}_{\text{weekly block, } m_1=7} + \underbrace{\sum_{k=1}^{K_2}\left[\beta_k'\sin\left(\tfrac{2\pi k t}{m_2}\right)+\gamma_k'\cos\left(\tfrac{2\pi k t}{m_2}\right)\right]}_{\text{yearly block, } m_2=365}
$$
**Plain English: one block of waves cycles every 7 days (weekly pattern), a completely separate block of waves cycles every 365 days (yearly pattern), and you just ADD both blocks together into one combined seasonal estimate.** There's no conflict or interference between the blocks — they're just separate columns in the same regression, and regression naturally handles combining any number of separate predictors. **This is precisely why Fourier terms are the standard practical solution for multi-seasonal data (daily+weekly+yearly, the classic case for web traffic, app usage, ride-sharing demand — genuinely common Apple/Google-flavored interview scenarios) — Holt-Winters and plain SARIMA have no clean equivalent mechanism for this.**

---

## 5. TBATS: briefly, what it adds on top

**New term: TBATS** stands for (unpacking each letter): **T**rigonometric seasonality (= Fourier terms, exactly what we just built), **B**ox-Cox transformation (Phase 4, section 8 — the variance-stabilizing transform, applied automatically), **A**RMA errors (Phase 6 — fitting a residual ARMA structure to whatever's left, exactly like Phase 7 section 8's "regression with ARIMA errors" idea), **T**rend, and **S**easonal components that can change over time (rather than being perfectly fixed forever, similar in spirit to STL's evolving-seasonality flexibility from Phase 5, section 2). **Plain English: TBATS is essentially a single, unified, automated package that bundles together nearly every tool you've learned so far — Box-Cox, Fourier-based multi-seasonality, ARMA error correction, and an evolving trend — into one automatically-fitted model.** You don't need to derive its internals further — you now already understand every individual PIECE it's built from; TBATS is simply a specific, well-engineered COMBINATION of tools you already have.

---

## 6. Prophet: fully explained using only concepts you already know

Meta/Facebook's Prophet model is extremely popular in industry, specifically DESIGNED for business-style time series (daily data, strong multiple seasonality, holiday effects, robust to missing data) — and interviewers frequently ask "how does Prophet actually work?" Let's answer that completely, piece by piece, since EVERY piece is something you've already learned.

**Prophet's overall model structure:**
$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t
$$
**Plain English: this is literally just Phase 1's additive decomposition formula ($x_t = T_t+S_t+I_t$) with a holiday term added and relabeled letters!** $g(t)$ = trend (was $T_t$ in Phase 1), $s(t)$ = seasonal (was $S_t$), $h(t)$ = holiday effect (a new, genuinely useful addition for business data), $\varepsilon_t$ = noise (was $I_t$). **Prophet's fundamental structure is NOT some exotic new idea — it's the very first formula of this entire course, Phase 1 section 4.1, with one extra term.**

### 6.1 The trend function $g(t)$: a "piecewise" linear/logistic trend
**New term: piecewise.** Plain English: instead of forcing ONE single straight (or curved) trend line through the ENTIRE dataset (which real business trends often don't follow — growth rates genuinely change over time, e.g., after a product launch or a market shift), Prophet allows the trend's SLOPE to change at specific points in time, called **changepoints**. Between any two changepoints, the trend is just an ordinary straight line (exactly like the trend-stationary deterministic line from Phase 4, section 4.1) — but the SLOPE of that line is allowed to shift at each changepoint, producing an overall trend that bends at a handful of specific moments rather than following one rigid global shape. **This is conceptually just several short trend-stationary segments (Phase 4) stitched together end to end** — genuinely nothing new mathematically, just cleverly allowing the "fixed" trend line's slope parameter to itself change at chosen points, with the specific changepoint LOCATIONS and sizes estimated from the data (using a regularization/penalty technique that discourages too many/too large changepoints, conceptually similar in spirit to the complexity-penalty logic behind AIC/BIC from Phase 6, Part 4, section 6, though the specific regularization method Prophet uses is a bit different technically).

### 6.2 The seasonal function $s(t)$: exactly section 3-4's Fourier terms
**Plain English: Prophet's seasonal component IS literally the Fourier-term regression you just fully derived in sections 3-4 of this file** — separate Fourier blocks for weekly seasonality (default $K=3$) and yearly seasonality (default $K=10$), simply added together exactly as shown in section 4. **Nothing new here at all — you have already derived, from scratch, precisely the mathematical machinery Prophet uses for this piece.**

### 6.3 The holiday function $h(t)$: just more regressors (directly reusing Phase 7)
**Plain English:** for each specified holiday (e.g., "Christmas," "Black Friday"), Prophet adds a simple indicator/dummy regressor (a predictor that's 1 on that holiday's date, and 0 everywhere else — you may recall this "dummy variable" idea from Phase 7's trend+seasonal-dummy regression discussion, section 1's brief mention) with its own estimated coefficient measuring that specific holiday's typical effect size — optionally also including a small window of days BEFORE/AFTER the holiday (e.g., the days leading into Christmas), each as their own separate dummy regressor. **This is precisely ordinary regression with dummy predictor variables (Phase 7) — nothing new mathematically, just a sensible, business-focused set of predictor choices.**

### 6.4 Fitting it all together
Since $g(t)$, $s(t)$, and $h(t)$ are all, at the end of the day, just combinations of (possibly changepoint-adjusted) linear terms and regression predictors, **Prophet fits the WHOLE thing as one combined regression-flavored optimization problem** (technically using a Bayesian estimation approach under the hood for extra robustness/uncertainty quantification, a refinement beyond what we need to detail here — the CORE structure is what matters for interview-level understanding).

**The genuinely valuable, complete interview answer you can now give: "Prophet is an additive decomposition model (trend + seasonal + holiday + noise, literally Phase 1's formula) where the trend is a piecewise-linear function that can bend its slope at estimated changepoints, the seasonal component is built from Fourier sine/cosine terms (allowing multiple overlapping seasonal periods cheaply, with far fewer parameters than estimating a separate index per calendar unit), and holidays are handled as simple dummy regressors — the whole thing fit as one combined regression-style model, which is why it's fast, interpretable, and robust to missing data compared to a full ARIMA approach."** This is a complete, technically accurate, first-principles answer — not a memorized buzzword description.

---

## 7. Numerical mini-example: building a tiny Fourier seasonal term by hand

Suppose $m=4$ (a period of 4, e.g., quarterly-flavored toy example) and we use just $K=1$ (the simplest possible case, one wave pair). At $t=1,2,3,4$ (one full cycle):

$\sin\left(\frac{2\pi(1)(1)}{4}\right) = \sin(\pi/2) = 1.000$
$\sin\left(\frac{2\pi(1)(2)}{4}\right) = \sin(\pi) = 0.000$
$\sin\left(\frac{2\pi(1)(3)}{4}\right) = \sin(3\pi/2) = -1.000$
$\sin\left(\frac{2\pi(1)(4)}{4}\right) = \sin(2\pi) = 0.000$

$\cos\left(\frac{2\pi(1)(1)}{4}\right) = \cos(\pi/2)=0.000$
$\cos\left(\frac{2\pi(1)(2)}{4}\right)=\cos(\pi)=-1.000$
$\cos\left(\frac{2\pi(1)(3)}{4}\right)=\cos(3\pi/2)=0.000$
$\cos\left(\frac{2\pi(1)(4)}{4}\right)=\cos(2\pi)=1.000$

Suppose regression fitting (Phase 7's OLS, applied to these two columns as predictors) gave us $\beta_1=3, \gamma_1=1$. The fitted seasonal value at each quarter:
$s(1) = 3(1.000)+1(0.000)=3.0$
$s(2)=3(0.000)+1(-1.000)=-1.0$
$s(3)=3(-1.000)+1(0.000)=-3.0$
$s(4)=3(0.000)+1(1.000)=1.0$

**Notice: this produces a clean, smooth, repeating seasonal pattern $[3.0,-1.0,-3.0,1.0]$, then repeating identically $[3.0,-1.0,-3.0,1.0]$ again for the next cycle** — and we built this ENTIRE seasonal shape from just TWO estimated numbers ($\beta_1,\gamma_1$), compared to needing FOUR separate seasonal index values if we'd instead used the classical decomposition approach from Phase 1/Phase 5's Holt-Winters seasonal term directly. **This small-scale example directly demonstrates the parameter-efficiency advantage discussed in section 3 — with a larger, more realistic $m$ (like 365), this efficiency gap becomes enormous.**

---

## 8. Quick self-check questions

1. Why can a small number of Fourier terms (say $K=5$, giving 10 total coefficients) represent a full year's daily seasonal pattern more efficiently than Holt-Winters' approach?
   *(Answer: Holt-Winters needs a separate seasonal index parameter for essentially every point in the cycle (near 365 for daily/yearly data), while Fourier terms represent the seasonal SHAPE using a small number of wave coefficients that combine to approximate the same repeating pattern — a large reduction in parameter count.)*
2. Why does the Fourier approach naturally handle MULTIPLE seasonal periods (e.g., weekly AND yearly) at once, while Holt-Winters/SARIMA do not?
   *(Answer: because Fourier terms are just ordinary regression predictors (Phase 7); you can add a separate, independent BLOCK of Fourier terms for each seasonal period into the same regression, and they combine additively without conflict — Holt-Winters/SARIMA instead have a single seasonal period hard-baked into their core recursive formulas, with no natural slot for a second, differently-sized cycle.)*
3. In Prophet, what earlier-learned concept is the "piecewise trend with changepoints" really just an extension of?
   *(Answer: the trend-stationary deterministic-line idea from Phase 4, section 4.1 — Prophet's trend is several short straight-line segments stitched together, with the slope allowed to change at specific estimated changepoints, rather than forcing one single rigid line/curve across the whole series.)*
4. What earlier-learned regression concept does Prophet's holiday effect term $h(t)$ directly reuse?
   *(Answer: dummy/indicator regressor variables from Phase 7 — a predictor that equals 1 on the holiday date(s) and 0 otherwise, with its own estimated coefficient representing that holiday's typical effect size.)*

---

## What's next
Phase 9 moves into **State Space Models and the Kalman Filter** — a more general mathematical framework that can express AR, MA, ARIMA, AND exponential smoothing (Phase 5) all as special cases of ONE unified structure, plus the actual recursive filtering algorithm (predict-then-update) used to estimate hidden/unobserved states in real time as new data arrives — the same core technology behind Bayesian Structural Time Series (used at Google for causal impact analysis, previewed in the original syllabus) and modern tracking/estimation systems generally.

Say "next" for Phase 9, or ask for more Fourier/Prophet drilling first.
