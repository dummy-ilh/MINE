# Phase 4: Stationarity, Formally — Differencing, ADF Test, KPSS Test

You've been using "stationarity" as an intuitive placeholder since Phase 2 ("the machine's personality doesn't change over time"). Now we make it precise, learn to test for it with real formulas, and learn exactly how to fix it when it's missing. This phase is the direct payoff of the random-walk-vs-AR(1) discussion from Phase 2, section 5.3 — go back and skim that if it's fuzzy, because the ADF test below is built entirely on that distinction.

---

## 1. Why do we even need stationarity? (Motivation before definition — read this first)

Almost every classical model you'll learn (AR, MA, ARMA, ARIMA's stationary part) is built on an assumption: **the statistical rules governing the series stay the same across the whole time period we're looking at.** If the rules keep changing, then a "coefficient" you estimate from the first half of your data might mean something completely different in the second half — you'd be fitting a moving target. Worse, if you naively run standard formulas (like the ACF formula from Phase 3, which used a SINGLE mean $\mu$ for the whole series) on data where the mean is actually drifting over time, your estimates become **meaningless** — you're averaging apples measured in January with oranges measured in December and calling it one number.

So stationarity isn't a pedantic technicality — it's the condition that makes the whole toolkit (ACF, PACF, AR, MA, ARIMA parameter estimation) valid in the first place.

---

## 2. Two flavors of stationarity, defined precisely

### 2.1 Strict (strong) stationarity — the strongest, purest version
**Definition:** A process is strictly stationary if the *entire joint probability distribution* of any collection of points $(x_{t_1}, x_{t_2}, \ldots, x_{t_k})$ is identical to the joint distribution of the same points shifted by any time lag $h$: $(x_{t_1+h}, x_{t_2+h}, \ldots, x_{t_k+h})$.

**Plain English:** if you took a snapshot of "how these points relate to each other, in every possible statistical way" at one point in the timeline, and then took the exact same shaped snapshot somewhere else in the timeline, they would be statistically IDENTICAL — not just similar averages, but identical distributions in every respect (mean, variance, skewness, every possible joint pattern).

**Why we rarely use this directly:** it's an extremely strong, hard-to-verify condition — you'd basically need to check infinitely many properties. In practice almost nobody tests for this directly.

### 2.2 Weak (covariance/second-order) stationarity — the practical version everyone actually uses
This is a much more usable, relaxed definition that only requires THREE specific conditions to hold:

**Condition 1 — Constant mean:**
$$
E[x_t] = \mu \quad \text{(same } \mu \text{ for every } t\text{)}
$$
Plain English: the series isn't drifting upward or downward over time — no trend.

**Condition 2 — Constant variance:**
$$
\text{Var}(x_t) = \sigma^2 \quad \text{(same } \sigma^2 \text{ for every } t\text{)}
$$
Plain English: the series isn't getting more or less "wild"/volatile as time passes (recall this is the same property white noise had, in Phase 2).

**Condition 3 — Autocovariance depends only on the LAG distance, not on the actual time location:**
$$
\text{Cov}(x_t, x_{t-k}) = \gamma(k) \quad \text{(depends only on } k\text{, not on } t\text{)}
$$
Plain English: the relationship between "today and 3 days ago" should be the SAME strength whether "today" is January 5th or October 12th. The memory structure of the process is time-invariant — it doesn't matter WHEN you measure it, only how far apart the two points are.

**This weak version is what people mean 95% of the time when they just say "stationary" in applied work, including in every future phase of this course.** From now on, "stationary" = these three conditions.

---

## 3. What non-stationarity looks like in practice (connecting back to Phase 1 and 2)

- **Trend** (Phase 1) violates Condition 1 (mean isn't constant — it's drifting).
- **Seasonality** (Phase 1) violates Condition 1 too, in a cyclical way (the mean at "December" differs systematically from the mean at "June").
- **Random walk** (Phase 2) violates Condition 2 — remember we derived $\text{Var}(x_t) = t\sigma^2$, which literally depends on $t$, directly breaking "constant variance."
- **Changing volatility over time** (e.g., a stock that was calm for years then became wild during a crisis) also violates Condition 2 — this specific pattern gets its own dedicated toolkit later (GARCH models, Phase 10).

---

## 4. Trend-Stationary vs. Difference-Stationary: a critical distinction (classic interview topic)

There are actually TWO fundamentally different ways a series can have a trend, and they require completely different fixes. This distinction trips up a huge number of practitioners, so slow down here.

### 4.1 Trend-Stationary process
**Definition, built from a formula:**
$$
x_t = f(t) + \varepsilon_t
$$
where $f(t)$ is some deterministic (non-random) function of time — e.g., a straight line $f(t) = a + bt$ — and $\varepsilon_t$ is stationary noise (like white noise, or even a stationary AR process) layered on top.

**Plain English:** there is ONE fixed, predictable trend line, set in stone from the start, and the data just wiggles around that fixed line due to noise. If you know $t$, you can predict exactly where the trend line sits — the ONLY uncertainty is the noise around it, and critically, **that noise does NOT accumulate — each period's noise is a fresh, independent wiggle around the SAME fixed line.**

**The fix:** if you literally subtract off the known trend function $f(t)$ — this is called **detrending** — what's left, $\varepsilon_t$, is stationary by construction. Example fix: fit a simple linear regression of $x_t$ on $t$, then work with the residuals.

### 4.2 Difference-Stationary process (this is the random walk family from Phase 2!)
**Definition:** the series ITSELF is non-stationary, but if you compute the **first difference** $y_t = x_t - x_{t-1}$, that NEW series $y_t$ IS stationary.

**Plain English:** recall the random walk, $x_t = x_{t-1} + \varepsilon_t$. Rearranged: $x_t - x_{t-1} = \varepsilon_t$. The differenced series is literally just white noise — stationary! But unlike the trend-stationary case, here the "trend-like drift" you visually see isn't a fixed, predictable line — it's the ACCUMULATION of past random shocks (exactly as derived in Phase 2, section 5). **Critically: every past shock permanently and fully changes the future level of the series forever** — there's no fixed anchor line to return to.

**The fix:** apply **differencing** (subtracting each point from the previous point) instead of detrending against a fixed line.

### 4.3 Why the distinction matters so much (the actual interview-relevant consequence)
If you WRONGLY detrend a difference-stationary (random-walk-type) series — i.e., you fit a straight line through it and subtract that line off — you do NOT get a properly stationary result, because the "true" wandering pattern isn't anchored to any fixed line at all; you'll leave behind spurious long-swinging patterns in your "residuals."

If you WRONGLY difference a trend-stationary series — i.e., you compute $x_t - x_{t-1}$ when the series actually had a fixed deterministic trend — you technically DO end up with something stationary, but you've introduced unnecessary extra noise/complexity (a specific technical problem called **overdifferencing**, discussed in section 6.3 below) and thrown away useful signal.

**This is exactly why we need formal statistical tests (ADF, KPSS) to tell these two cases apart, rather than just eyeballing a chart** — visually, both can look like "the series trends upward with wiggles," but the correct fix is completely different for each.

---

## 5. Differencing: the practical fix, in detail

**First-order differencing**, defined precisely:
$$
y_t = x_t - x_{t-1} = \nabla x_t
$$
The symbol $\nabla$ (nabla, sometimes called the "difference operator") is just shorthand notation for "take this series and subtract each point from the one before it." You'll see this symbol used constantly in ARIMA notation later.

**Why does this remove a linear trend?** Suppose $x_t = a + bt + \varepsilon_t$ (a straight-line trend plus noise). Then:
$$
y_t = x_t - x_{t-1} = (a+bt+\varepsilon_t) - (a + b(t-1) + \varepsilon_{t-1}) = b + (\varepsilon_t - \varepsilon_{t-1})
$$
Notice: the $a$ cancels completely, and $bt - b(t-1) = b$ — a CONSTANT, no longer depending on $t$. **The linear trend has been converted into a constant offset, and the remaining part is just noise differences.** This is the algebraic proof of why differencing removes a straight-line trend.

**Second-order differencing:** sometimes one differencing pass isn't enough (e.g., if the trend itself is curving, a quadratic trend rather than a straight line). You then difference the ALREADY-differenced series again:
$$
\nabla^2 x_t = \nabla(\nabla x_t) = (x_t - x_{t-1}) - (x_{t-1}-x_{t-2}) = x_t - 2x_{t-1} + x_{t-2}
$$
In practice, real economic/business data almost never needs more than $d=1$ or $d=2$ differences — needing more is usually a red flag that something else is wrong (wrong transformation, structural break, etc.).

**Seasonal differencing:** if a series has seasonality with period $s$ (e.g., $s=12$ for monthly data with yearly seasonality), you instead subtract the value from ONE FULL SEASONAL CYCLE ago:
$$
\nabla_s x_t = x_t - x_{t-s}
$$
Plain English: instead of comparing today to yesterday, you compare "this December" to "last December" — this directly removes a repeating seasonal pattern the same way ordinary differencing removes a trend. In Phase 6 (SARIMA), you'll often see BOTH ordinary and seasonal differencing applied together.

---

## 6. Unit Root Tests: the formal, rigorous way to actually test for this

### 6.1 Setting up the core question (recall from Phase 2, section 5.3)
We want to distinguish:
$$
x_t = \phi\, x_{t-1} + \varepsilon_t
$$
- If $\phi = 1$: this is exactly a random walk — non-stationary, shocks accumulate forever (this is called having a **unit root**, because in the algebra of this equation, rewritten as $(1-\phi L)x_t = \varepsilon_t$ using the lag operator $L$ where $Lx_t = x_{t-1}$, the "characteristic root" of $1-\phi z=0$ is $z=1/\phi$, and when $\phi=1$ that root sits exactly at 1 — hence "unit" root).
- If $|\phi| < 1$: this is stationary — shocks decay away, the process reverts toward a stable mean.

We need a formal statistical test to distinguish these two cases from real, noisy, finite data — you can't just eyeball whether an estimated $\hat\phi$ is "close enough" to 1.

### 6.2 The Augmented Dickey-Fuller (ADF) Test — full derivation

**Step 1 — Rearrange the AR(1) equation into a differenced form.** Start with $x_t = \phi x_{t-1} + \varepsilon_t$. Subtract $x_{t-1}$ from both sides:
$$
x_t - x_{t-1} = \phi x_{t-1} - x_{t-1} + \varepsilon_t = (\phi - 1)x_{t-1} + \varepsilon_t
$$
Define $\gamma = \phi - 1$ (a new symbol, just relabeling for convenience — note this $\gamma$ is unrelated to the autocovariance $\gamma(k)$ from Phase 3, unfortunately the same Greek letter gets reused across different contexts in this field — a genuine annoyance you'll just have to get used to). So:
$$
\Delta x_t = \gamma\, x_{t-1} + \varepsilon_t \qquad \text{where } \Delta x_t \equiv x_t - x_{t-1}
$$
($\Delta$, "delta," is just another common symbol for "first difference," used interchangeably with the $\nabla$ from section 5.)

**Step 2 — Translate the stationarity question into a question about $\gamma$.** Since $\gamma = \phi - 1$:
- If $\phi = 1$ (unit root / random walk / non-stationary) → $\gamma = 0$.
- If $|\phi| < 1$ (stationary) → $\gamma < 0$ (since $\phi<1$ implies $\phi - 1$ is negative; note $\phi$ is typically assumed positive and less than 1 in the basic case, so $\gamma$ is a negative number, reflecting genuine mean-reversion/pull-back).

**This is the key insight of the whole test: instead of testing "is $\phi$ equal to 1?" directly, we test the mathematically equivalent, but computationally cleaner, question "is $\gamma$ equal to 0?"** using an ordinary regression of $\Delta x_t$ on $x_{t-1}$.

**Step 3 — Set up the hypothesis test.**
- **Null hypothesis $H_0$: $\gamma = 0$** (there IS a unit root — the series is non-stationary, a random walk).
- **Alternative hypothesis $H_1$: $\gamma < 0$** (NO unit root — the series is stationary).

Note this is a ONE-sided test (we only care whether $\gamma$ is negative, not positive) — and note also the somewhat counter-intuitive framing: **the "boring, default" null hypothesis here is non-stationarity**, which is the OPPOSITE convention from KPSS below — a very common point of confusion we'll resolve explicitly in section 7.

**Step 4 — Why is this NOT a standard t-test, and what makes it "Dickey-Fuller"?**
You might think: just run the regression $\Delta x_t = \gamma x_{t-1} + \varepsilon_t$, get the estimated $\hat\gamma$ and its standard error, and run an ordinary t-test. Here's the subtlety: **ordinary t-test theory assumes stationary regressors.** But under the null hypothesis we're testing ($\gamma=0$, meaning $x_{t-1}$ is a random walk, non-stationary!), the regressor $x_{t-1}$ ITSELF is non-stationary. This breaks the standard mathematical assumptions that make the ordinary t-distribution valid. Dickey and Fuller (the statisticians who created this test) worked out that under the null hypothesis, the test statistic (still computed the same way, as $\hat\gamma$ divided by its standard error — sometimes called the "tau" statistic) follows a DIFFERENT, non-standard distribution (now called the **Dickey-Fuller distribution**), which has to be looked up from specially computed/simulated critical value tables, rather than ordinary t-tables. **This is the single most important nuance of this entire test, and a very common interview question: "why can't you just use a normal t-test for a unit root?"** — now you know the precise answer: because the regressor's distribution under the null is itself non-stationary, invalidating the standard t-distribution assumptions, requiring specially derived (and typically more extreme/negative) critical values.

**Step 5 — The "Augmented" part.** The basic Dickey-Fuller test above assumes the noise $\varepsilon_t$ has no leftover autocorrelation of its own. Real data often violates this. The **Augmented** Dickey-Fuller test fixes this by adding extra lagged difference terms as additional control regressors:
$$
\Delta x_t = \gamma\, x_{t-1} + \beta_1 \Delta x_{t-1} + \beta_2 \Delta x_{t-2} + \dots + \beta_p \Delta x_{t-p} + \varepsilon_t
$$
Plain English: we throw in enough recent lagged DIFFERENCES as control variables to "soak up" any leftover autocorrelation in the noise, so that what remains genuinely behaves like proper white noise, keeping the test statistically valid. The core logic (test whether $\gamma=0$) is completely unchanged — this is purely a technical robustness fix. Also commonly included: a constant term and/or a deterministic trend term, depending on whether you suspect the series might be trend-stationary (section 4.1) rather than a pure random walk.

**Step 6 — How to read ADF test output in practice:** Software (R, Python's statsmodels) will report a test statistic and a p-value. **If the p-value is small (conventionally < 0.05), you REJECT the null hypothesis of a unit root — meaning the evidence suggests the series IS stationary.** If the p-value is large, you FAIL to reject — meaning you don't have strong evidence against non-stationarity, so you should treat the series as likely non-stationary and consider differencing it.

### 6.3 A critical practical trap: over-differencing
What if you difference a series that was ALREADY stationary (i.e., you had a false positive concern about a unit root, or you just differenced "to be safe")? You don't get an error — the differenced series is technically still stationary — BUT you've introduced an artificial NEGATIVE autocorrelation at lag 1 into your noise, made your model unnecessarily more complex (needing an extra MA term to fix, foreshadowing Phase 6), and inflated the variance of your forecasts unnecessarily. **The practical rule: only difference as much as needed, guided by formal tests (ADF/KPSS) and by watching whether the ACF of your differenced series still shows the slow decay signature of non-stationarity (Phase 3) — don't difference reflexively "just in case."**

---

## 7. The KPSS Test: the deliberate mirror-image of ADF

**Why do we need a second test at all, if ADF already exists?** Because ADF's null hypothesis is "non-stationary" — meaning ADF is specifically GOOD at confidently detecting stationarity (when it rejects), but a "fail to reject" result from ADF is a weak, ambiguous conclusion (it just means "not enough evidence," which could be because the series really is non-stationary, OR simply because you don't have enough data / enough statistical power to be sure). Using a test with the OPPOSITE null hypothesis lets you cross-check and build much stronger confidence when both tests agree.

**KPSS (Kwiatkowski-Phillips-Schmidt-Shin) hypotheses — deliberately flipped from ADF:**
- **Null hypothesis $H_0$: the series IS stationary** (or trend-stationary, depending on the test specification).
- **Alternative hypothesis $H_1$: the series is non-stationary** (has a unit root).

**Conceptual construction (without the full derivation, which involves partial-sum/Brownian motion theory beyond our current scope):** KPSS decomposes the series into a deterministic trend, a pure random walk component, and stationary noise, then constructs a statistic (based on cumulative sums of residuals from a regression) that tends to be SMALL when the random-walk component has essentially zero variance (i.e., the series is genuinely stationary) and grows LARGE as that random-walk component's variance grows (more non-stationary behavior). **Reading KPSS output: a SMALL p-value here means REJECT stationarity (evidence FOR non-stationarity)** — note this is the OPPOSITE reading direction from ADF's p-value, which is a very common source of mixed-up interpretation, so be careful.

### 7.1 The practical 2x2 combination table (a genuinely common interview question)

| ADF result | KPSS result | Conclusion |
|---|---|---|
| Reject unit root (stationary) | Fail to reject (stationary) | **Strong agreement: series is stationary.** |
| Fail to reject (non-stationary) | Reject (non-stationary) | **Strong agreement: series is non-stationary — difference it.** |
| Reject unit root (stationary) | Reject (non-stationary) | **Conflicting! Often indicates the series is stationary AROUND a trend (trend-stationary, section 4.1) — detrend rather than difference, or the test specifications (constant/trend terms included) don't match.** |
| Fail to reject (non-stationary) | Fail to reject (stationary) | **Conflicting/inconclusive — often means there's genuinely not enough data/power to tell; proceed cautiously, consider more data or visual inspection.** |

**Practical takeaway for real work: never rely on a single test.** Run both ADF and KPSS, use the table above, and combine with a visual inspection of the raw series and its ACF (Phase 3) before deciding whether/how to difference.

---

## 8. Variance-stabilizing transformations: Box-Cox and log

Stationarity Condition 2 (constant variance) can also be violated in a specific way: variance that grows as the LEVEL of the series grows (the "megaphone shape" from Phase 1, section 4.2 — the same pattern that signals a multiplicative rather than additive model). Differencing fixes non-constant MEAN (trend); it does NOT fix non-constant VARIANCE. For that, we need a different tool: transforming the scale of the data itself.

**Log transform**, the simplest case:
$$
y_t = \log(x_t)
$$
Recall from Phase 1 section 4.3: this converts multiplicative relationships into additive ones, and it also compresses large values proportionally more than small values — exactly counteracting a megaphone-shaped variance pattern (where absolute swings are proportional to the level).

**Box-Cox transform: a general family that includes the log as a special case.**
$$
y_t = \begin{cases} \dfrac{x_t^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\[4pt] \log(x_t) & \text{if } \lambda = 0 \end{cases}
$$
Plain English: $\lambda$ (lambda) is a single tunable "knob" that controls HOW aggressively we transform the data. $\lambda=1$ means basically no transformation (just a shift). $\lambda=0$ means the pure log transform (matching the log-multiplicative connection from Phase 1). $\lambda=0.5$ gives something like a square-root transform. In practice, software estimates the "best" $\lambda$ from the data automatically (typically by maximum likelihood — the same general estimation philosophy we'll use heavily in Phase 6) to find whichever transform makes the variance most stable across the series.

**Practical workflow ordering — a genuinely important nuance:** always apply variance-stabilizing transforms (log/Box-Cox) **BEFORE** differencing for trend removal. Fix the variance problem first, then fix the mean/trend problem. Doing it in the wrong order can produce misleading intermediate diagnostics.

---

## 9. Numerical worked example: run through the whole detection-and-fix pipeline by hand

Take a tiny 6-point series suspected to be a random walk: $x = [10, 13, 11, 15, 14, 17]$.

**Step 1 — Compute first differences** $\Delta x_t = x_t - x_{t-1}$:
$\Delta x_2 = 13-10=3$
$\Delta x_3 = 11-13=-2$
$\Delta x_4 = 15-11=4$
$\Delta x_5 = 14-15=-1$
$\Delta x_6 = 17-14=3$

Differenced series: $[3, -2, 4, -1, 3]$

**Step 2 — Eyeball check:** the original series climbs unevenly with no fixed anchor (10→13→11→15→14→17 — generally rising but choppy). The differenced series bounces around with no obvious remaining trend, roughly centered near a small positive number (mean of differences = $(3-2+4-1+3)/5 = 7/5 = 1.4$, suggesting a mild positive drift, consistent with a random-walk-with-drift structure from Phase 2, section 5.2, rather than pure zero-drift random walk).

**Step 3 — What ADF would formally test here:** regress $\Delta x_t$ on $x_{t-1}$ (this toy dataset is far too small for a real test — you'd need at least 30-50+ points for ADF to have any real power — but conceptually): if the estimated $\hat\gamma$ (coefficient on $x_{t-1}$) comes out close to 0 and not statistically distinguishable from 0 given the standard error, that supports "unit root present, treat as random walk, use the differenced series for modeling." If $\hat\gamma$ came out clearly negative and significant (using Dickey-Fuller's special critical values, not ordinary t-tables), that would instead suggest genuine mean-reversion (stationary AR(1) with $\phi<1$), and you would NOT want to difference — you'd model the LEVEL series directly with an AR structure instead (foreshadowing Phase 6).

---

## 10. Quick self-check questions

1. In plain English, what's the difference between a trend-stationary process and a difference-stationary process, and why does it matter which one your data is?
   *(Answer: trend-stationary has ONE fixed deterministic trend line with stationary noise wiggling around it forever — fix by detrending (subtracting the fixed line). Difference-stationary (random-walk type) has NO fixed anchor — past shocks permanently shift the level forever — fix by differencing. Using the wrong fix leaves you with improperly "cleaned" data — either spurious leftover patterns (wrong: detrending a random walk) or unnecessary added noise/complexity (wrong: differencing a trend-stationary series, section 6.3's overdifferencing trap).)*
2. Why can't the ADF test statistic be evaluated with an ordinary t-table?
   *(Answer: under the null hypothesis being tested (γ=0, unit root present), the regressor x_{t-1} is itself non-stationary, which breaks the assumptions needed for the standard t-distribution to be valid — Dickey and Fuller derived a special, different reference distribution instead.)*
3. If ADF says "reject unit root" (stationary) but KPSS ALSO rejects its null (suggesting non-stationary) — what does this conflicting combination typically suggest, per the 2x2 table?
   *(Answer: the series is likely trend-stationary — stationary around a deterministic trend rather than a pure unit root — suggesting detrending rather than differencing, or a mismatch in test specification regarding included trend/constant terms.)*
4. Why should you apply a Box-Cox/log transform BEFORE differencing, rather than after?
   *(Answer: the transform fixes non-constant VARIANCE (Condition 2), while differencing fixes non-constant MEAN/trend (Condition 1) — these are separate problems, and applying them in the wrong order can distort your diagnostics/leave residual variance issues that the differencing step wasn't designed to address.)*

---

## What's next
Phase 5 covers **classical decomposition and smoothing methods** in full formal depth: moving averages (with the exact centering mechanics you saw informally in Phase 1), STL decomposition, and the full Exponential Smoothing family (Simple Exponential Smoothing, Holt's linear trend method, Holt-Winters seasonal method) — including deriving the recursive update formulas and working a full numerical forecast by hand.

Say "next" for Phase 5, or ask for more ADF/KPSS drilling first.
