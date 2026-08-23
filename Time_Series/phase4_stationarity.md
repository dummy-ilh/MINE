# Phase 4: Stationarity, Formally — Differencing, ADF Test, KPSS Test

Since Phase 2, "stationarity" has been an intuitive placeholder — "the machine's personality doesn't change over time." Now we make it precise, test for it with real formulas, and fix it when it's missing. This phase is the direct payoff of the random-walk-vs-AR(1) discussion from Phase 2, section 5.3 — the ADF test below is built entirely on that distinction.

---

## 1. Why we need stationarity at all

Almost every classical model (AR, MA, ARMA, ARIMA's stationary part) assumes the statistical rules governing the series stay the same across the whole period being studied. If the rules keep changing, a coefficient estimated from the first half of the data might mean something completely different in the second half — a moving target dressed up as a fixed one.

Worse: standard formulas like the ACF from Phase 3 use a single mean $\mu$ for the entire series. If the true mean is actually drifting over time, that estimate becomes meaningless — like averaging January's temperature with December's and calling it "the temperature."

Stationarity isn't a technicality. It's the condition that makes the whole toolkit — ACF, PACF, AR, MA, ARIMA parameter estimation — valid in the first place.

---

## 2. Two flavors of stationarity

### 2.1 Strict (strong) stationarity

**Definition:** a process is strictly stationary if the *entire joint probability distribution* of any collection of points $(x_{t_1}, x_{t_2}, \ldots, x_{t_k})$ is identical to the joint distribution of the same points shifted by any lag $h$: $(x_{t_1+h}, \ldots, x_{t_k+h})$.

**Plain English:** take a full statistical snapshot of how a set of points relate to each other — mean, variance, skew, every joint pattern — at one point in the timeline. Take the same shaped snapshot somewhere else. They'd be statistically identical in every respect, not just similar on average.

This is an extremely strong, essentially unverifiable condition — you'd need to check infinitely many properties. Almost nobody tests for it directly.

### 2.2 Weak (covariance) stationarity — the version everyone actually uses

Three conditions, much easier to work with:

**Condition 1 — Constant mean:**
$$
E[x_t] = \mu \quad \text{(same } \mu \text{ for every } t\text{)}
$$
The series isn't drifting upward or downward — no trend.

**Condition 2 — Constant variance:**
$$
\text{Var}(x_t) = \sigma^2 \quad \text{(same } \sigma^2 \text{ for every } t\text{)}
$$
The series isn't getting more or less volatile as time passes (same property white noise had in Phase 2).

**Condition 3 — Autocovariance depends only on lag distance, not on when you measure it:**
$$
\text{Cov}(x_t, x_{t-k}) = \gamma(k) \quad \text{(depends only on } k\text{, not on } t\text{)}
$$
The relationship between "today and 3 days ago" is the same strength whether today is January 5th or October 12th. The memory structure is time-invariant.

**From here on, "stationary" means these three conditions** — this is what 95% of applied work means by the word.

---

## 3. What non-stationarity looks like in practice

- **Trend** (Phase 1) violates Condition 1 — the mean is drifting.
- **Seasonality** (Phase 1) also violates Condition 1, cyclically — December's mean systematically differs from June's.
- **Random walk** (Phase 2) violates Condition 2 — recall $\text{Var}(x_t) = t\sigma^2$, which literally depends on $t$.
- **Changing volatility** (a stock calm for years, then wild during a crisis) also violates Condition 2 — gets its own toolkit later (GARCH, Phase 10).

---

## 4. Trend-Stationary vs. Difference-Stationary

Two fundamentally different ways a series can trend, requiring completely different fixes. This distinction trips up a lot of practitioners.

**The core picture: a dog on a leash vs. a person wandering with no leash at all.**

### 4.1 Trend-Stationary process — the dog on a leash

$$
x_t = f(t) + \varepsilon_t
$$

where $f(t)$ is a fixed, deterministic function of time — e.g., a straight line $f(t) = a + bt$ — and $\varepsilon_t$ is stationary noise layered on top.

**Picture:** a dog walking beside its owner on a leash, along a perfectly straight, predetermined path. The dog wanders left and right of the path, sometimes pulling ahead, sometimes lagging — but it is always tethered to that exact same fixed line. Knowing the time tells you exactly where the path is; the only uncertainty is the dog's wiggle around it. Critically, **the wiggle does not accumulate** — each moment's deviation is fresh, independent of the last, always relative to the same fixed line.

**The fix — detrending:** subtract off the known trend function $f(t)$. What's left, $\varepsilon_t$, is stationary by construction. In practice: fit a linear regression of $x_t$ on $t$, then work with the residuals.

### 4.2 Difference-Stationary process — the leashless wanderer (the random walk family from Phase 2)

**Definition:** the series itself is non-stationary, but its **first difference** $y_t = x_t - x_{t-1}$ IS stationary.

**Picture:** the drunk-walk from Phase 2, with no leash at all. Recall $x_t = x_{t-1} + \varepsilon_t$, rearranged: $x_t - x_{t-1} = \varepsilon_t$. The differenced series is literally just white noise — stationary! But unlike the leashed dog, there's no fixed line anywhere in this picture. What looks like "drift" is just the accumulation of past random steps (Phase 2, section 5). Every past shock permanently and fully relocates the walker — there's no anchor to snap back to.

**The fix — differencing:** subtract each point from the previous one, instead of subtracting a fixed line.

### 4.3 Why the distinction matters

Detrend a leashless wanderer (wrongly fit a straight line through a random walk and subtract it) and you're left with spurious long, slow swings in your "residuals" — because there was never a real line to subtract in the first place.

Difference a leashed dog (wrongly compute $x_t - x_{t-1}$ on a series that actually had a fixed deterministic trend) and you technically do end up with something stationary — but you've introduced unnecessary extra noise and complexity (called **overdifferencing**, section 6.3) and thrown away real signal in the process.

Both mistakes can look identical on a raw chart — "trends upward with wiggles" — which is exactly why we need formal tests (ADF, KPSS) instead of eyeballing.

---

## 5. Differencing: the practical fix, in detail

**First-order differencing:**
$$
y_t = x_t - x_{t-1} = \nabla x_t
$$
$\nabla$ (nabla, the "difference operator") is shorthand for "subtract each point from the one before it." You'll see it constantly in ARIMA notation.

**Why this removes a linear trend — the algebra.** Suppose $x_t = a + bt + \varepsilon_t$ (straight-line trend plus noise):
$$
y_t = x_t - x_{t-1} = (a+bt+\varepsilon_t) - (a + b(t-1) + \varepsilon_{t-1}) = b + (\varepsilon_t - \varepsilon_{t-1})
$$
The $a$ cancels completely, and $bt - b(t-1) = b$ — a constant, no longer depending on $t$. The trend collapses into a fixed offset, and what remains is just noise differences. That's the proof of why differencing kills a straight-line trend.

**Second-order differencing:** if one pass isn't enough (a curving, quadratic trend rather than a straight line), difference the already-differenced series again:
$$
\nabla^2 x_t = \nabla(\nabla x_t) = (x_t - x_{t-1}) - (x_{t-1}-x_{t-2}) = x_t - 2x_{t-1} + x_{t-2}
$$
Real economic/business data rarely needs more than $d=1$ or $d=2$ — needing more is usually a red flag something else is wrong (wrong transformation, structural break).

**Seasonal differencing:** for a series with seasonality of period $s$ (e.g., $s=12$ for monthly data with yearly seasonality), subtract the value from one full seasonal cycle ago:
$$
\nabla_s x_t = x_t - x_{t-s}
$$
Instead of comparing today to yesterday, compare this December to last December — removes a repeating seasonal pattern the same way ordinary differencing removes a trend. In Phase 6 (SARIMA), ordinary and seasonal differencing are often used together.

---

## 6. Unit Root Tests: formally testing for this

### 6.1 Setting up the question (recall Phase 2, section 5.3)

We want to distinguish:
$$
x_t = \phi\, x_{t-1} + \varepsilon_t
$$

- If $\phi = 1$: exactly a random walk — non-stationary, shocks accumulate forever. Called having a **unit root** (rewriting as $(1-\phi L)x_t = \varepsilon_t$ using the lag operator $L$ where $Lx_t = x_{t-1}$, the characteristic root of $1-\phi z=0$ is $z=1/\phi$; when $\phi=1$ that root sits exactly at 1).
- If $|\phi| < 1$: stationary — shocks decay away, the process reverts toward a stable mean. Think back to the leashed dog: $\phi$ close to 1 is a very long, loose leash; $\phi$ noticeably less than 1 is a short, taut leash pulling the dog back quickly.

We need a formal test to distinguish these two from real, noisy, finite data — you can't just eyeball whether an estimated $\hat\phi$ is "close enough" to 1.

### 6.2 The Augmented Dickey-Fuller (ADF) Test — full derivation

**Step 1 — Rearrange the AR(1) equation into differenced form.** Start with $x_t = \phi x_{t-1} + \varepsilon_t$. Subtract $x_{t-1}$ from both sides:
$$
x_t - x_{t-1} = \phi x_{t-1} - x_{t-1} + \varepsilon_t = (\phi - 1)x_{t-1} + \varepsilon_t
$$
Define $\gamma = \phi - 1$ (a relabeling — note this $\gamma$ is unrelated to the autocovariance $\gamma(k)$ from Phase 3; the same Greek letter gets reused across different contexts, a genuine annoyance you'll get used to). So:
$$
\Delta x_t = \gamma\, x_{t-1} + \varepsilon_t \qquad \text{where } \Delta x_t \equiv x_t - x_{t-1}
$$
($\Delta$, "delta," is another common symbol for first difference, used interchangeably with $\nabla$.)

**Step 2 — Translate stationarity into a question about $\gamma$.** Since $\gamma = \phi - 1$:
- $\phi = 1$ (unit root, non-stationary) → $\gamma = 0$
- $|\phi| < 1$ (stationary) → $\gamma < 0$ (real mean-reversion, the leash pulling back)

**Key insight:** instead of testing "is $\phi$ equal to 1?" directly, test the equivalent, computationally cleaner question "is $\gamma$ equal to 0?" via an ordinary regression of $\Delta x_t$ on $x_{t-1}$.

**Step 3 — The hypothesis test.**
- **Null $H_0$: $\gamma = 0$** — there IS a unit root, the series is a random walk.
- **Alternative $H_1$: $\gamma < 0$** — no unit root, the series is stationary.

One-sided (we only care whether $\gamma$ is negative). Note the framing: **the "default" null here is non-stationarity** — the opposite convention from KPSS below, a common point of confusion resolved explicitly in section 7.

**Step 4 — Why this isn't an ordinary t-test.**
You might expect: run the regression, get $\hat\gamma$ and its standard error, run a normal t-test. The subtlety: ordinary t-test theory assumes stationary regressors. But under the null being tested ($\gamma=0$, meaning $x_{t-1}$ is a random walk), the regressor $x_{t-1}$ is itself non-stationary — this breaks the assumptions that make the ordinary t-distribution valid. Dickey and Fuller worked out that, under the null, the test statistic (still $\hat\gamma$ divided by its standard error, sometimes called the "tau" statistic) follows a different, non-standard distribution — the **Dickey-Fuller distribution** — requiring specially simulated critical value tables rather than ordinary t-tables. **This is the key nuance and a classic interview question — "why can't you just use a normal t-test for a unit root?"** Because the regressor's distribution under the null is itself non-stationary, invalidating standard t-distribution assumptions, and requiring specially derived (typically more extreme, more negative) critical values.

**Step 5 — The "Augmented" part.** Basic Dickey-Fuller assumes the noise $\varepsilon_t$ has no leftover autocorrelation. Real data often violates this. Augmented Dickey-Fuller fixes it by adding lagged difference terms as extra controls:
$$
\Delta x_t = \gamma\, x_{t-1} + \beta_1 \Delta x_{t-1} + \beta_2 \Delta x_{t-2} + \dots + \beta_p \Delta x_{t-p} + \varepsilon_t
$$
Throw in enough recent lagged differences to soak up any leftover autocorrelation in the noise, so what remains genuinely behaves like white noise and the test stays valid. The core logic (test whether $\gamma=0$) is unchanged — this is a technical robustness fix. Also commonly included: a constant term and/or deterministic trend term, depending on whether the series might be trend-stationary (section 4.1) rather than a pure random walk.

**Step 6 — Reading ADF output.** Software reports a test statistic and p-value. **Small p-value (conventionally < 0.05) → reject the null of a unit root → evidence the series IS stationary.** Large p-value → fail to reject → not strong evidence against non-stationarity → treat as likely non-stationary and consider differencing.

### 6.3 A critical trap: over-differencing

Difference a series that was already stationary — no error occurs, the result is technically still stationary — but you've introduced an artificial negative autocorrelation at lag 1, made your model needlessly more complex (needing an extra MA term to fix, foreshadowing Phase 6), and inflated forecast variance unnecessarily. **Think of it like sanding a surface that's already smooth — you're not fixing anything, you're just wearing it down and adding noise.** Practical rule: only difference as much as needed, guided by formal tests (ADF/KPSS) and by watching whether the differenced series' ACF still shows the slow-decay signature of non-stationarity (Phase 3) — don't difference reflexively "just in case."

---

## 7. The KPSS Test: the deliberate mirror-image of ADF

**Why a second test at all?** ADF's null hypothesis is "non-stationary" — so ADF is specifically good at confidently detecting stationarity (when it rejects), but a "fail to reject" result is weak and ambiguous — it could mean the series really is non-stationary, or simply that there isn't enough data/power to be sure. Think of it like a court case: ADF presumes non-stationarity innocent until proven guilty. A test with the opposite presumption lets you cross-check, and agreement between both builds much stronger confidence.

**KPSS (Kwiatkowski-Phillips-Schmidt-Shin) hypotheses — deliberately flipped:**
- **Null $H_0$:** the series IS stationary (or trend-stationary, depending on specification).
- **Alternative $H_1$:** the series is non-stationary (has a unit root).

**Conceptual construction** (full derivation involves partial-sum/Brownian motion theory beyond scope here): KPSS decomposes the series into a deterministic trend, a pure random-walk component, and stationary noise, then builds a statistic from cumulative sums of residuals. That statistic stays small when the random-walk component has essentially zero variance (genuinely stationary) and grows large as that component's variance grows (more non-stationary). **Reading KPSS output: a small p-value means REJECT stationarity (evidence FOR non-stationarity)** — the opposite reading direction from ADF's p-value. A very common source of confusion — be careful.

### 7.1 The practical 2×2 combination table

| ADF result | KPSS result | Conclusion |
|---|---|---|
| Reject unit root (stationary) | Fail to reject (stationary) | **Strong agreement: series is stationary.** |
| Fail to reject (non-stationary) | Reject (non-stationary) | **Strong agreement: series is non-stationary — difference it.** |
| Reject unit root (stationary) | Reject (non-stationary) | **Conflicting — often means the series is trend-stationary (section 4.1): stationary around a fixed line, not a true random walk. Detrend rather than difference, or check for a test-specification mismatch (constant/trend terms included).** |
| Fail to reject (non-stationary) | Fail to reject (stationary) | **Conflicting/inconclusive — often means not enough data/power to tell. Proceed cautiously; consider more data or visual inspection.** |

**Practical takeaway:** never rely on a single test. Run both ADF and KPSS, use the table above, and combine with visual inspection of the raw series and its ACF (Phase 3) before deciding whether/how to difference.

---

## 8. Variance-stabilizing transformations: Box-Cox and log

Condition 2 (constant variance) can be violated in a specific way: variance growing as the level of the series grows — the "megaphone shape" from Phase 1, section 4.2, the same pattern signaling a multiplicative rather than additive model. **Differencing fixes non-constant mean; it does nothing for non-constant variance.** For that, transform the scale of the data itself.

**Log transform**, the simplest case:
$$
y_t = \log(x_t)
$$
This converts multiplicative relationships into additive ones (Phase 1, section 4.3), and compresses large values proportionally more than small ones — exactly counteracting a megaphone pattern where absolute swings scale with the level. **Picture shrinking a photo down: the biggest shapes shrink the most in absolute terms, so everything ends up more evenly sized relative to each other.**

**Box-Cox transform** — a general family including log as a special case:
$$
y_t = \begin{cases} \dfrac{x_t^{\lambda} - 1}{\lambda} & \text{if } \lambda \neq 0 \\[4pt] \log(x_t) & \text{if } \lambda = 0 \end{cases}
$$
$\lambda$ (lambda) is a tunable knob controlling how aggressively the data is transformed. $\lambda=1$ ≈ no transformation (just a shift). $\lambda=0$ = pure log transform. $\lambda=0.5$ ≈ a square-root transform. In practice, software estimates the "best" $\lambda$ automatically (typically by maximum likelihood, the same estimation philosophy used heavily in Phase 6) to find whichever transform makes the variance most stable.

**Ordering matters:** always apply variance-stabilizing transforms (log/Box-Cox) **before** differencing for trend removal. Fix the variance problem first, then the mean/trend problem — doing it backward can distort intermediate diagnostics.

---

## 9. Numerical worked example: the full pipeline by hand

A tiny 6-point series suspected to be a random walk: $x = [10, 13, 11, 15, 14, 17]$.

**Step 1 — First differences** $\Delta x_t = x_t - x_{t-1}$:
$\Delta x_2 = 13-10=3$
$\Delta x_3 = 11-13=-2$
$\Delta x_4 = 15-11=4$
$\Delta x_5 = 14-15=-1$
$\Delta x_6 = 17-14=3$

Differenced series: $[3, -2, 4, -1, 3]$

**Step 2 — Eyeball check:** the original series climbs unevenly with no fixed anchor (10→13→11→15→14→17 — generally rising but choppy). The differenced series bounces with no obvious remaining trend, roughly centered near a small positive number (mean of differences $= (3-2+4-1+3)/5 = 7/5 = 1.4$) — suggesting mild positive drift, consistent with a random-walk-with-drift (Phase 2, section 5.2) rather than a pure zero-drift random walk.

**Step 3 — What ADF would formally test here:** regress $\Delta x_t$ on $x_{t-1}$ (this toy dataset is far too small for a real test — you'd need at least 30–50+ points for ADF to have real power — but conceptually): if $\hat\gamma$ comes out close to 0 and not statistically distinguishable from 0, that supports "unit root present, treat as random walk, model the differenced series." If $\hat\gamma$ came out clearly negative and significant (using Dickey-Fuller's special critical values, not ordinary t-tables), that would instead suggest genuine mean-reversion (stationary AR(1) with $\phi<1$) — and you would NOT want to difference, you'd model the level series directly with an AR structure (foreshadowing Phase 6).

---

## 10. Self-check questions

1. In plain English, what's the difference between a trend-stationary process and a difference-stationary process, and why does it matter which one your data is?
   *Trend-stationary is the leashed dog: one fixed deterministic trend line, with stationary noise wiggling around it forever — fix by detrending (subtract the fixed line). Difference-stationary (random-walk type) is the leashless wanderer: no fixed anchor, past shocks permanently shift the level forever — fix by differencing. Using the wrong fix leaves improperly cleaned data — either spurious leftover patterns (wrongly detrending a random walk) or unnecessary added noise/complexity (wrongly differencing a trend-stationary series — overdifferencing, section 6.3).*

2. Why can't the ADF test statistic be evaluated with an ordinary t-table?
   *Under the null being tested (γ=0, unit root present), the regressor $x_{t-1}$ is itself non-stationary, which breaks the assumptions needed for the standard t-distribution to be valid — Dickey and Fuller derived a special, different reference distribution instead.*

3. If ADF says "reject unit root" (stationary) but KPSS also rejects its null (suggesting non-stationary) — what does this conflicting combination typically suggest?
   *The series is likely trend-stationary — stationary around a deterministic trend rather than a pure unit root — suggesting detrending rather than differencing, or a mismatch in test specification regarding included trend/constant terms.*

4. Why apply a Box-Cox/log transform before differencing, rather than after?
   *The transform fixes non-constant variance (Condition 2), while differencing fixes non-constant mean/trend (Condition 1) — separate problems. Applying them in the wrong order can distort diagnostics or leave residual variance issues that differencing wasn't designed to address.*

---

## What's next

Phase 5 covers classical decomposition and smoothing methods in full formal depth: moving averages (with the exact centering mechanics seen informally in Phase 1), STL decomposition, and the full Exponential Smoothing family (Simple Exponential Smoothing, Holt's linear trend method, Holt-Winters seasonal method) — including deriving the recursive update formulas and working a full numerical forecast by hand.
