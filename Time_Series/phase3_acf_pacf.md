# Phase 3: Autocorrelation (ACF) and Partial Autocorrelation (PACF)

From Phase 2: a stochastic process can have memory (like the random walk, where today's value depends on the entire past) or no memory (white noise). Now we build the actual tool that measures how much memory a series has, and at what distance (lag). This tool — the ACF — is the single most-used diagnostic in classical time series, and reading its shape correctly is how you identify which model (AR, MA, ARIMA) fits real data in Phase 6.

---

## 1. What is a "lag"?

**Lag** means "how many time steps back are we looking." If today is $t$:
- lag 1 = comparing today $x_t$ to yesterday $x_{t-1}$
- lag 2 = comparing today $x_t$ to two days ago $x_{t-2}$
- lag $k$ in general = comparing $x_t$ to $x_{t-k}$

Lag is just distance backward in time, measured in steps.

---

## 2. Autocovariance: the raw ingredient

Ordinary covariance between two variables $A$ and $B$ measures whether they move together:

$$
\text{Cov}(A,B) = E[(A-\mu_A)(B-\mu_B)]
$$

Take how far $A$ is from its own average, take how far $B$ is from its own average, multiply those two distances together, then average that product over all observations. If $A$ and $B$ tend to be above-average at the same time (or below-average at the same time), the product is usually positive — they move together. If one is high when the other is low, the product is usually negative. No relationship → the products cancel toward zero.

**The time series trick:** instead of comparing two different variables, compare the same series to a shifted copy of itself. That's what "auto" (self) correlation means — literally checking a series against its own echo.

**Everyday picture:** think of a sentence repeated over a walkie-talkie with delay and echo. Autocovariance is asking: "does what I hear right now resemble what I said $k$ seconds ago?" A strong echo at a 1-second delay means the signal has 1-second memory.

**Autocovariance at lag $k$:**

$$
\gamma(k) = \text{Cov}(x_t, x_{t-k}) = E[(x_t - \mu)(x_{t-k}-\mu)]
$$

- $\gamma$ (gamma) — the standard symbol for autocovariance, just notation.
- $\mu$ (mu) — the overall mean of the series, assumed constant across time (this is the stationarity assumption from Phase 2 — required for a single $\mu$ to even make sense).
- $x_t$ = value today, $x_{t-k}$ = value $k$ steps back.

**Plain English:** take today's deviation from the mean, multiply it by the deviation of the value $k$ steps ago, and average that product across the whole series. Large and positive means: when the series is above average today, it also tended to be above average $k$ steps ago — a real, measurable echo of length $k$.

**Sanity check:** $\gamma(0) = \text{Cov}(x_t,x_t) = \text{Var}(x_t)$ — comparing the series to itself at lag 0 is just its own variance. This matters in the next section.

---

## 3. Autocorrelation Function (ACF): making it interpretable

Raw autocovariance has a units problem: measure revenue in dollars vs. thousands of dollars and $\gamma(k)$ changes size even though the real relationship strength hasn't changed. Fix it the same way ordinary correlation fixes covariance — divide by the variance to force the result between −1 and +1.

$$
\rho(k) = \frac{\gamma(k)}{\gamma(0)} = \frac{\text{Cov}(x_t,x_{t-k})}{\text{Var}(x_t)}
$$

- $\rho$ (rho) — the standard symbol for the **autocorrelation function (ACF)** at lag $k$.
- Dividing by $\gamma(0)$ (the variance) keeps the result on a clean, comparable scale regardless of the data's original units.

**Plain English:** $\rho(k)$ tells you, on a clean −1-to-+1 scale, how strongly the series echoes itself $k$ steps back. $\rho(k)=1$ = perfect positive echo, $\rho(k)=0$ = no linear relationship at that lag, $\rho(k)=-1$ = perfect inverse echo.

**Properties worth knowing:**
- $\rho(0) = 1$ always — a series is perfectly correlated with itself at lag 0. Use this as a sanity anchor.
- $\rho(k) = \rho(-k)$ — the ACF is symmetric; looking $k$ steps forward or backward gives the same value, because covariance itself is symmetric.
- For white noise, $\rho(k) = 0$ for every $k \neq 0$ — the formal restatement of "white noise has zero memory," now expressed with the tool we just built.

### 3.1 The correlogram

A **correlogram** is the standard name for a bar chart of $\rho(k)$ against $k$ — lag on the x-axis, correlation value on the y-axis, one bar per lag. It's one of the most commonly shown plots in time series analysis.

---

## 4. Sample ACF: computing it from real, finite data

Everything above assumed we knew the true underlying process. In real life we have a finite dataset of $n$ observations $x_1, \dots, x_n$ and must **estimate** $\rho(k)$. The estimator is the **sample ACF**, $\hat{\rho}(k)$ (the hat $\hat{}$ is standard statistics notation for "estimated from data," not the unknown true value).

$$
\hat{\rho}(k) = \frac{\sum_{t=k+1}^{n} (x_t - \bar{x})(x_{t-k}-\bar{x})}{\sum_{t=1}^{n}(x_t-\bar{x})^2}
$$

- $\bar{x}$ = the sample mean of all $n$ observations, our estimate of $\mu$.
- Numerator: for every pair of points exactly $k$ steps apart, multiply their deviations from the sample mean, then sum. The sum starts at $t=k+1$ because $x_{t-k}$ must exist — you can't look $k$ steps before the first data point.
- Denominator: sum of squared deviations across all $n$ points — essentially $n$ times the sample variance, playing the role of $\gamma(0)$.

**Confidence bands, intuitively:** plot a sample ACF and you'll see two dashed horizontal lines, often near $\pm 1.96/\sqrt{n}$. Here's why: if the true process were pure white noise (zero real autocorrelation everywhere), the *sample* ACF still won't land exactly on zero — it'll wobble a little just from sampling randomness, the same way a fair coin flipped 20 times won't give you exactly 10 heads every time. For large $n$, that random wobble is approximately Normal with standard deviation $\approx 1/\sqrt{n}$. The constant 1.96 comes from the standard Normal distribution — the cutoff beyond which only 5% of pure random wobble would fall (the same 1.96 you've seen in ordinary 95% confidence intervals).

**Practical reading rule:** if a sample ACF bar pokes out beyond the dashed band, that's evidence of a real relationship at that lag, not just noise-wobble. If it stays inside the band, you can't confidently distinguish it from zero.

---

## 5. Partial Autocorrelation Function (PACF): removing the middleman

**The problem plain ACF has:** if today is strongly related to yesterday (a real lag-1 relationship), and yesterday is related to the day before, that relationship can chain through time and create an *apparent* relationship at lag 2, lag 3, and beyond — even if there's no direct connection between today and 2 days ago. It's an indirect echo transmitted through yesterday.

**Analogy — a gossip chain:** Alice tells Bob a secret. Bob tells Carol. Carol tells Dave. Check "does Dave know a version of the secret" and yes — Dave's knowledge correlates with Alice's original secret. But Dave never heard it directly from Alice; it passed through Bob and Carol. The plain ACF between Alice and Dave shows a positive relationship that's entirely indirect.

**What we actually want:** a tool that measures the DIRECT relationship between $x_t$ and $x_{t-k}$, after stripping away — controlling for — everything in between ($x_{t-1}, \dots, x_{t-k+1}$). This is exactly what "partial correlation" means in general statistics — "partial" meaning "with other variables held constant," the same sense used in "controlling for confounders" in regression.

**Definition in words:** the **partial autocorrelation at lag $k$**, written $\phi_{kk}$, is the correlation between $x_t$ and $x_{t-k}$ after removing the linear effect of everything in between from both of them.

**Why the notation $\phi_{kk}$?** It comes from the Yule-Walker equations (derived properly in Phase 6), where the PACF at lag $k$ turns out to equal the last coefficient $\phi_k$ in a hypothetical AR($k$) regression of $x_t$ on its $k$ most recent past values. The double subscript $\phi_{kk}$ denotes "the $k$-th coefficient, in a model using $k$ total lags." Just recognize the notation for now — full derivation in Phase 6.

**A computational description usable right now:** imagine regressing $x_t$ on $x_{t-1}, x_{t-2}, \ldots, x_{t-k}$ all at once. The PACF at lag $k$ is the regression coefficient attached specifically to $x_{t-k}$ — "how much does $x_{t-k}$ move $x_t$, holding all the closer/intermediate lags constant?" It answers: does lag $k$ add any new, direct predictive information beyond what the closer lags already gave?

---

## 6. Why we need BOTH ACF and PACF: the model identification cheat sheet

This is the most practically useful takeaway of this phase — how you read a series' shape and guess its underlying model.

| Process type | ACF shape | PACF shape |
|---|---|---|
| White noise | Zero everywhere (all bars inside confidence band) | Zero everywhere |
| AR(p) (autoregressive, Phase 6) | Tails off gradually (decays slowly, possibly oscillating) | Cuts off sharply after lag $p$ |
| MA(q) (moving average, Phase 6) | Cuts off sharply after lag $q$ | Tails off gradually |
| ARMA(p,q) | Tails off | Tails off |

**Why does an AR process's PACF cut off sharply but its ACF doesn't?** In an AR(p) process, $x_t$ is directly built from exactly the last $p$ values plus noise — nothing beyond lag $p$ is directly used. Once you strip away the indirect chained effects (exactly what PACF does), there's genuinely nothing left beyond lag $p$, so PACF cuts off cleanly. The plain ACF still shows gradual decay beyond lag $p$ because of the indirect gossip-chain effect — lag $p+1$ is still indirectly connected through the chain of closer lags, even though it's not directly in the model.

**Why the mirror image for MA(q)?** An MA(q) process builds $x_t$ directly out of the last $q$ noise terms, not past values of $x$ itself. That gives a naturally sharp ACF cutoff at lag $q$. But expressing it in terms of past $x$ values (what PACF implicitly does) requires infinitely many past $x$ terms to reconstruct that noise-based structure — hence PACF tails off slowly instead of cutting.

Memorize this table as a lookup tool for now; Phase 6 derives precisely why each row holds, using the actual AR and MA formulas.

---

## 7. Full numerical worked example

Small dataset, 8 points: $x = [4, 6, 5, 7, 6, 8, 7, 9]$

**Step 1 — sample mean.**
$\bar{x} = (4+6+5+7+6+8+7+9)/8 = 52/8 = 6.5$

**Step 2 — deviations from the mean:**

| $t$ | $x_t$ | $x_t - \bar{x}$ |
|---|---|---|
| 1 | 4 | −2.5 |
| 2 | 6 | −0.5 |
| 3 | 5 | −1.5 |
| 4 | 7 | +0.5 |
| 5 | 6 | −0.5 |
| 6 | 8 | +1.5 |
| 7 | 7 | +0.5 |
| 8 | 9 | +2.5 |

**Step 3 — denominator ($\gamma(0)$, sum of squared deviations):**
$(-2.5)^2+(-0.5)^2+(-1.5)^2+(0.5)^2+(-0.5)^2+(1.5)^2+(0.5)^2+(2.5)^2$
$= 6.25+0.25+2.25+0.25+0.25+2.25+0.25+6.25 = 18.0$

**Step 4 — numerator for lag 1** (pair each deviation with the previous one):

| pair (t, t−1) | product |
|---|---|
| (2,1): (−0.5)(−2.5) | 1.25 |
| (3,2): (−1.5)(−0.5) | 0.75 |
| (4,3): (0.5)(−1.5) | −0.75 |
| (5,4): (−0.5)(0.5) | −0.25 |
| (6,5): (1.5)(−0.5) | −0.75 |
| (7,6): (0.5)(1.5) | 0.75 |
| (8,7): (2.5)(0.5) | 1.25 |

Sum = $1.25+0.75-0.75-0.25-0.75+0.75+1.25 = 2.25$

**Step 5 — $\hat\rho(1)$:**

$$
\hat\rho(1) = \frac{2.25}{18.0} = 0.125
$$

**Interpretation:** a weak positive lag-1 relationship. With $n=8$, the confidence band is roughly $\pm 1.96/\sqrt{8} \approx \pm 0.693$, so 0.125 is nowhere near significant. Expected — you need far more than 8 points to reliably estimate autocorrelation. This example is purely mechanical practice, not a real conclusion.

**Step 6 — lag 2** (pair each deviation with the value two steps back):

| pair (t, t−2) | product |
|---|---|
| (3,1): (−1.5)(−2.5) | 3.75 |
| (4,2): (0.5)(−0.5) | −0.25 |
| (5,3): (−0.5)(−1.5) | 0.75 |
| (6,4): (1.5)(0.5) | 0.75 |
| (7,5): (0.5)(−0.5) | −0.25 |
| (8,6): (2.5)(1.5) | 3.75 |

Sum = $3.75-0.25+0.75+0.75-0.25+3.75 = 8.5$

$$
\hat\rho(2) = \frac{8.5}{18.0} = 0.472
$$

Lag 2 came out stronger than lag 1. Looking at the raw data $[4,6,5,7,6,8,7,9]$, there's a visible zig-zag-while-trending pattern where every-other point tends to line up. This is a good reminder that with tiny datasets, sample ACF values can look larger or ordered differently than the "true" underlying pattern would suggest, purely from sampling noise — exactly why real-world ACF analysis is always done with confidence bands in mind, and with as much data as reasonably available.

---

## 8. The Ljung-Box test: formalizing "does this look like white noise?"

$$
Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho(k)^2}{n-k}
$$

- $n$ = number of observations.
- $h$ = number of lags being jointly tested (e.g., lags 1 through 10 at once — a common choice).
- $\hat\rho(k)^2$ = squared sample autocorrelation at each lag — squaring makes every term positive, so positive and negative correlations don't cancel; we care about the magnitude of leftover structure, not its direction.
- Dividing by $(n-k)$ is a small-sample correction: sample ACF at higher lags is based on fewer valid pairs (the sum starts later for larger $k$), so those estimates are noisier, and this division down-weights that extra noise.

**Plain English:** $Q$ adds up evidence of leftover correlation across many lags into a single number. Under the null hypothesis that the true process is white noise, $Q$ follows a known reference distribution (Chi-squared with $h$ degrees of freedom — a standard distribution used to get a p-value, no need to derive it).

**Practical use:** fit a model (e.g., ARIMA, Phase 6), look at its residuals, compute $Q$ on the residuals' ACF. Large $Q$ (small p-value, conventionally < 0.05) means the residuals still have structure — the model missed something and needs refining. Small $Q$/large p-value means you can't reject "residuals are white noise" — a good sign the model extracted everything useful.

---

## 9. Self-check questions

1. What's the difference, in plain English, between what ACF measures and what PACF measures at lag 3?
   *ACF at lag 3 measures the total relationship between $x_t$ and $x_{t-3}$, including indirect effects chained through lags 1 and 2. PACF at lag 3 measures only the direct relationship, after removing the effects of lags 1 and 2.*

2. An ACF plot cuts off sharply to zero after lag 2, and the PACF tails off slowly. What process does this suggest?
   *MA(2) — sharp ACF cutoff at lag $q$ plus slowly tailing PACF is the MA(q) signature.*

3. Why does dividing by $(n-k)$ in the Ljung-Box formula matter, rather than always dividing by $n$?
   *Sample ACF at higher lags $k$ is computed from fewer valid data pairs (only $n-k$ pairs exist), making those estimates inherently noisier — the formula down-weights that reduced reliability accordingly.*

4. For white noise, what should $\rho(k)$ equal for every $k \neq 0$, and why?
   *0 — because white noise has zero covariance between any two different time points by definition (Phase 2, Property 3), and ACF is just normalized covariance, so zero covariance directly implies zero ACF.*

---

## What's next

Phase 4 covers stationarity formally: the precise mathematical definition, why it's required for everything built so far to be valid, how to fix non-stationary data (differencing, transformations), and the actual statistical tests (ADF, KPSS) used to check for it in practice — including the full derivation of the ADF test statistic, which directly builds on the random-walk-vs-AR(1) distinction from Phase 2, section 5.3.
