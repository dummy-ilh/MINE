# Phase 7: Regression With Time Series Data (STAT 510 Core)

Everything in Phase 6 was about predicting $x_t$ from ITS OWN past (AR/MA/ARIMA). Phase 7 shifts to a different, equally common question: **predicting $x_t$ from OTHER variables** — e.g., predicting Apple's revenue from marketing spend, or app downloads from a competitor's price change — while the data STILL has a time order. This turns out to break ordinary regression in specific, well-understood ways, and this phase is exactly where STAT 510 spends serious time, because it's a genuinely common real-world mistake.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $y_t$ | the outcome/target variable at time $t$ (what we're trying to predict) |
| $x_t$ (as a REGRESSOR here) | a predictor variable at time $t$ — careful, in this file $x_t$ means "predictor," a different role than in Phase 6 where $x_t$ was "the series itself" |
| $\beta_0,\beta_1$ | ordinary regression coefficients (intercept and slope) |
| $u_t$ | the regression error/residual term (like $\varepsilon_t$, but reserved for the REGRESSION'S error, distinct from an ARIMA noise term) |
| OLS | **Ordinary Least Squares** — the standard, basic method of fitting a regression line by minimizing squared errors |
| GLS | **Generalized Least Squares** — a corrected version of OLS that accounts for autocorrelated errors |
| $\rho$ (in this file) | the autocorrelation coefficient of the REGRESSION ERRORS specifically (yet another reuse of this letter — track it by context) |
| $R^2$ | the standard "percent of variance explained" regression fit statistic |

---

## 2. Ordinary regression, quick refresher (Phase 0 material, revisited)

Basic linear regression: $y_t = \beta_0 + \beta_1 x_t + u_t$. **OLS finds $\hat\beta_0,\hat\beta_1$ by minimizing the sum of squared errors** $\sum_t \hat{u}_t^2$ — you've likely seen this before. **Crucially, standard OLS theory (the formulas for standard errors, confidence intervals, and p-values you'd get from software) is built on an assumption called "no autocorrelation in the errors"** — i.e., it assumes $u_t$ behaves like white noise (Phase 2): no memory, no relationship between $u_t$ and $u_{t-1}$.

---

## 3. Why time series regression errors are almost never actually white noise

**Plain English intuition:** if $y_t$ is something like monthly revenue, and your regression MISSES some real driver (say, a slow economic trend, or a competitor's pricing pattern that you didn't include as a predictor), that missing effect doesn't just randomly appear and disappear — it tends to PERSIST across nearby time points. If your model under-predicts this month (leaving a positive error $u_t$), it's quite likely to ALSO under-predict next month for the same underlying reason (a positive $u_{t+1}$ too). **This means regression errors from time series data are very often autocorrelated — precisely the opposite of what OLS assumes.**

**Formally, we model this leftover autocorrelation the same way you already know how — as an AR(1) process (Phase 6, Part 1) applied to the ERRORS themselves, not to $y_t$ directly:**
$$
u_t = \rho\, u_{t-1} + e_t
$$
where $e_t$ is now GENUINE white noise, and $\rho$ (careful — a NEW use of this letter, meaning the AR(1) coefficient of the regression errors, NOT the ACF from Phase 3, though it plays a structurally similar role). This specific setup — regression with AR(1) errors — is so common it has its own name: an **AR(1) error model**.

---

## 4. What goes wrong, precisely, if you ignore this and just run plain OLS

**The genuinely important, often-misunderstood nuance: your ESTIMATED coefficients $\hat\beta_0,\hat\beta_1$ themselves are usually still roughly unbiased (not systematically wrong on average) even with autocorrelated errors.** What breaks is something more subtle and more dangerous precisely BECAUSE it's subtle: **the STANDARD ERRORS that software reports for those coefficients become wrong** — typically, badly UNDERSTATED (too small).

**Why does autocorrelation shrink your apparent standard errors, intuitively?** Standard errors are, roughly, a measure of "how much NEW, INDEPENDENT information did we actually have to pin down this estimate." If your errors are autocorrelated, nearby data points are effectively echoing/repeating similar information rather than each contributing a genuinely fresh, independent data point (recall: this "less independent information than the raw count of $n$ suggests" issue is structurally the SAME core problem that made ordinary statistics inappropriate for time series in the very first place, back in Phase 2, section 2 — autocorrelation always reduces your EFFECTIVE sample size below the raw count $n$). **Software, unaware of this, computes standard errors AS IF you had $n$ fully independent data points — overstating your genuine precision, and understating the true standard errors.**

**The practical, real-world consequence: your confidence intervals will be too narrow, and your p-values will be too small — making a predictor look statistically significant when it may not genuinely be.** This is a completely real, commonly-made mistake in industry: someone runs a regression on monthly business data, ignores the strong month-to-month persistence in the errors, and confidently declares a marketing campaign "statistically significant" when the apparent significance is partly an illusion created by autocorrelation deflating the standard errors. **This exact failure mode is a very realistic interview scenario/case-study question.**

---

## 5. The fix: Generalized Least Squares (GLS) / Cochrane-Orcutt procedure

**Core idea, in plain English before formulas:** if we know (or can estimate) HOW the errors are autocorrelated (i.e., estimate $\rho$ from section 3), we can mathematically "undo"/transform the data to remove that autocorrelation BEFORE fitting the regression — turning the problem back into one where ordinary OLS assumptions hold again.

**Deriving the transformation (this is a genuinely elegant, simple trick, let's build it step by step):**

Start with the regression at time $t$ and at time $t-1$:
$$
y_t = \beta_0+\beta_1 x_t + u_t \qquad y_{t-1}=\beta_0+\beta_1 x_{t-1}+u_{t-1}
$$
Multiply the SECOND equation by $\rho$ (our estimated error-autocorrelation coefficient from section 3):
$$
\rho\, y_{t-1} = \rho\beta_0 + \rho\beta_1 x_{t-1} + \rho\, u_{t-1}
$$
**Now subtract this from the FIRST equation:**
$$
y_t - \rho y_{t-1} = \beta_0(1-\rho) + \beta_1(x_t-\rho x_{t-1}) + (u_t - \rho u_{t-1})
$$
**Look closely at the very last piece: $u_t - \rho u_{t-1}$.** Recall from section 3, $u_t = \rho u_{t-1}+e_t$, which rearranges to EXACTLY $u_t - \rho u_{t-1} = e_t$ — **genuine, clean white noise!** So the transformed equation is:
$$
y_t^{*} = \beta_0(1-\rho) + \beta_1 x_t^{*} + e_t \qquad \text{where } y_t^*=y_t-\rho y_{t-1},\ \ x_t^*=x_t-\rho x_{t-1}
$$
**Plain English summary of the whole trick: if you transform BOTH your outcome variable and your predictor variable using this exact same "subtract $\rho$ times the previous value" recipe (notice — this IS a differencing-flavored operation, structurally very close to Phase 4's differencing, just using $\rho$ as the multiplier instead of a flat "1"), the NEW transformed regression has genuinely clean white-noise errors, and ordinary OLS on THIS transformed version gives correct, trustworthy standard errors.** This specific practical recipe (estimate $\rho$ from the residuals, transform the data, refit) is called the **Cochrane-Orcutt procedure**, usually run a couple of times iteratively (refitting $\rho$ from the new residuals, re-transforming, repeating) until the estimates stabilize. **GLS** is the more general theoretical name for this class of "transform to restore clean-error OLS" techniques — Cochrane-Orcutt is a specific, practical, real, widely-implemented version of it for the AR(1)-error case.

**A simpler, extremely common alternative you already fully understand: just difference everything.** If $\rho$ is close to 1 (very persistent errors, close to a unit-root-flavored problem, Phase 4), a very common practical shortcut is to run the regression on FIRST DIFFERENCES of both $y$ and $x$ instead ($\Delta y_t$ on $\Delta x_t$) — literally the $d=1$ differencing operator you already fully know from Phase 4, applied here to a regression setting instead of a pure ARIMA setting. This is the $\rho=1$ special case of the Cochrane-Orcutt transformation above (plug $\rho=1$ into the formulas: $y_t^*=y_t-y_{t-1}=\Delta y_t$, exactly matching).

---

## 6. Spurious Regression: a much more dangerous, deeper problem (genuinely famous, Granger-Newbold)

Section 4-5 covered a "standard errors are wrong" problem — annoying, but fixable, and the coefficients themselves were still basically fine. **This next problem is worse: the coefficients THEMSELVES, and the whole apparent relationship, can be complete garbage — even though the regression output looks great.**

**The setup, and the surprising finding (Granger & Newbold, 1974):** take TWO completely UNRELATED random walks (Phase 2) — literally simulate two independent random walks with no real connection to each other whatsoever — and regress one on the other. **You would expect, correctly, that the true relationship is zero (they're unrelated by construction).** But Granger and Newbold showed that, astonishingly often, you get: a HIGH $R^2$ (looking like a strong relationship), a seemingly statistically significant slope coefficient (small p-value, looking meaningful), and yet the WHOLE THING is a complete illusion, entirely an artifact of both series independently wandering/trending due to their unit-root nature (Phase 4) — NOT any genuine relationship.

**Why does this happen, intuitively, connecting directly to what you already know?** Recall from Phase 2, section 6: a random walk, even with zero real drift, tends to wander away from its starting point for long stretches purely by accumulated chance — producing what LOOKS like a trend even though there's no real trending force (we explicitly flagged this exact visual illusion back in Phase 2!). **If TWO unrelated random walks happen to be wandering in roughly the same general direction over your sample period (which happens often, purely by chance, since neither one has any "pull back" to a fixed level — Phase 4's non-stationarity), regression will "detect" a strong-looking relationship between two numbers that are both simply drifting, for entirely separate, coincidental reasons.**

**The practical, dangerous consequence: this looks EXACTLY like a genuine, meaningful regression result in standard output — high $R^2$, low p-value, "significant" coefficient — with absolutely nothing distinguishing it from a real relationship, UNLESS you specifically know to check for it.** This is precisely why spurious regression is such a commonly tested interview concept: it's a real trap that looks completely convincing on the surface.

**The diagnostic red flag, connecting directly to Phase 4's tools:** a classic tell-tale sign of spurious regression is a very HIGH $R^2$ combined with a very LOW Durbin-Watson statistic (a quick, related cousin of the Ljung-Box idea from Phase 3/Part 6.5 — it specifically detects lag-1 autocorrelation in regression residuals) — meaning the regression LOOKS great by $R^2$, but the RESIDUALS still show strong leftover autocorrelation (violating exactly the assumption from section 3), which is the technical fingerprint that the "relationship" is really just two non-stationary series happening to drift together, not real shared structure.

**The actual fix: TEST both series for stationarity FIRST (Phase 4's ADF/KPSS tests), before ever trusting a time-series regression.** If either series has a unit root (non-stationary), either (a) difference BOTH series first and regress the differenced/stationary versions instead (directly reusing Phase 4's fix), or (b) check specifically for **cointegration** — a special, genuine exception case, previewed next.

---

## 7. Cointegration: a genuine, important exception to "always difference first" (full depth comes later in the multivariate phase)

**Here's a fascinating wrinkle worth knowing now, even before the full formal treatment:** sometimes TWO individually non-stationary (unit-root) series move together in a genuinely REAL, non-spurious way — specifically, when there's some real economic/structural force that keeps them from drifting TOO far apart from each other, even though each one individually wanders. Classic example: a company's quarterly revenue and its quarterly operating costs might BOTH individually be non-stationary/trending upward over time (Phase 4), but the GAP between them (profit margin) tends to stay in a much more stable, bounded range — because real competitive/business forces prevent that gap from wandering off indefinitely in either direction. **When this happens — two non-stationary series whose particular LINEAR COMBINATION is itself stationary — the series are called cointegrated, and this is a genuinely REAL, meaningful, exploitable relationship, not a spurious one.**

**Why can't you just always difference to be safe then?** Because if two series ARE genuinely cointegrated, differencing BOTH of them before regressing would actually throw away real, valuable long-run information about their genuine relationship (an overdifferencing-flavored problem, echoing Phase 4, section 6.3) — you'd be solving the spurious-regression problem in a way that overcorrects and discards a real signal. **This is exactly why cointegration testing (the Engle-Granger method, and the Johansen test) exists as its own dedicated topic — to distinguish "these are genuinely, meaningfully linked despite both being non-stationary" from "this is pure spurious coincidence."** We'll build the full formal Engle-Granger and Johansen procedures, plus the Vector Error Correction Model (VECM) that exploits a real cointegrating relationship for forecasting, in the dedicated Multivariate Time Series phase later in this syllabus — for now, the important takeaway is just recognizing that this genuine exception exists, and knowing its NAME and its intuition, so "spurious regression" and "cointegration" don't get conflated in your head — they are, in a sense, mirror-image concepts (one is a fake relationship between non-stationary series, the other is a REAL one).

---

## 8. Regression with ARIMA errors: briefly connecting Phase 6 and Phase 7 together

**A natural generalization worth flagging:** section 3 modeled the regression errors as AR(1). But nothing stops you from modeling them as a FULL ARIMA process instead (any $p,d,q$ from Phase 6) if a simple AR(1) doesn't fully capture the leftover structure — this combined approach (a regression on external predictors, PLUS a full ARIMA structure fitted to whatever's left in the errors) is called **regression with ARIMA errors**, or a **transfer function model** in some textbooks. **Conceptually, it's nothing new: fit the regression, check the residual ACF/PACF (Phase 3) exactly as you would for model diagnostics (Phase 6, Part 5), identify an appropriate ARIMA structure for what's left, and fit both pieces together** (software like R's `auto.arima()` with an `xreg` argument does exactly this in one combined step). This is a genuinely common real production technique — e.g., forecasting sales using both a promotional-calendar regressor AND an ARIMA structure to capture whatever calendar-independent momentum remains.

---

## 9. Numerical mini-illustration: detecting the problem from section 3

Suppose you fit $y_t = \beta_0+\beta_1 x_t + u_t$ and get these residuals over 6 time points: $\hat{u} = [2, 3, 1, 4, 2, 5]$ (deliberately chosen to show an obvious pattern — real residuals from a well-specified model should NOT show a visible pattern like this).

**Quick, informal check (a lightweight version of Phase 3's ACF machinery, applied here to regression residuals instead of raw data):** compute the mean $\bar{u} = (2+3+1+4+2+5)/6 = 17/6\approx 2.833$. Deviations: $[-0.833, 0.167, -1.833, 1.167, -0.833, 2.167]$. Notice going step to step: below average, then slightly above, then well below, then above, then below, then well above — **actually here it's alternating rather than persisting, which would signal NEGATIVE autocorrelation** (a real but less common pattern than the POSITIVE persistence case emphasized in section 3 — worth knowing residual autocorrelation can go either direction; negative autocorrelation in regression errors sometimes shows up from overdifferencing, section 6.3-style, or certain overcorrection patterns). **The genuinely important takeaway from this small example: don't just eyeball residuals — always formally compute the residual ACF (Phase 3) or run a Durbin-Watson/Ljung-Box test (section 6 / Phase 6 Part 5) rather than relying on intuition**, since patterns can be subtle or even run in the opposite direction from what you might first assume.

---

## 10. Quick self-check questions

1. If regression errors are positively autocorrelated but you ignore it and run plain OLS anyway, what SPECIFICALLY goes wrong — the coefficient estimates, the standard errors, or both?
   *(Answer: primarily the standard errors — they come out too SMALL/understated, making predictors look more statistically significant than they really are; the coefficient point-estimates themselves are typically still roughly unbiased.)*
2. In the Cochrane-Orcutt transformation, why does $u_t - \rho u_{t-1}$ end up being clean white noise?
   *(Answer: because the AR(1) error model from section 3 is exactly $u_t = \rho u_{t-1}+e_t$ with $e_t$ defined as white noise — rearranging that equation directly gives $u_t-\rho u_{t-1}=e_t$, so the transformation is constructed specifically to isolate exactly that white-noise piece.)*
3. What's the single most important thing to check BEFORE trusting a time-series regression's high R² and significant p-values, given the spurious regression risk?
   *(Answer: test both series for stationarity/unit roots first (Phase 4's ADF/KPSS) — a strong-looking relationship between two non-stationary series may be entirely spurious, an artifact of both series independently drifting, rather than a genuine relationship.)*
4. How does cointegration differ from spurious regression, even though both involve non-stationary series showing an apparent relationship?
   *(Answer: in spurious regression, the apparent relationship is a coincidental artifact of both series independently drifting with no real connection; in cointegration, there IS a genuine underlying structural force keeping a specific linear combination of the two non-stationary series stable/stationary over time — a real, exploitable relationship rather than an illusion.)*

---

## What's next
Phase 8 covers **complex/multiple seasonality** — handling data with more than one seasonal pattern at once (e.g., Google search traffic with BOTH a daily AND a weekly AND a yearly pattern simultaneously), Fourier-term regressors, and a full breakdown of how Facebook/Meta's Prophet model works internally (which is really just a clever combination of a piecewise trend, Fourier-based seasonality, and holiday regressors — all things you'll already understand the building blocks of by then).

Say "next" for Phase 8, or ask for more drilling on spurious regression / cointegration first — genuinely one of the most interview-relevant concepts in this whole course.
