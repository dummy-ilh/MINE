# Phase 3: Autocorrelation (ACF) and Partial Autocorrelation (PACF) — Absolute Zero, continued

From Phase 2, you know: a stochastic process can have "memory" (like the random walk, where today's value depends on the entire past) or no memory (white noise). Now we build the actual **tool that measures how much memory a series has, and at what distance (lag)**. This tool — the ACF — is the single most-used diagnostic in classical time series, and reading its shape correctly is how you'll identify which model (AR, MA, ARIMA) fits real data in Phase 6.

---

## 1. What is a "lag"? (New word — foundational, everything below depends on it)

**Lag** just means "how many time steps back are we looking." If today is $t$, then:
- "lag 1" means comparing today $x_t$ to yesterday $x_{t-1}$
- "lag 2" means comparing today $x_t$ to two days ago $x_{t-2}$
- "lag $k$" in general means comparing $x_t$ to $x_{t-k}$

That's it — lag is nothing more than "distance backward in time, measured in number of steps."

---

## 2. Autocovariance: the raw ingredient before we get to autocorrelation

Recall from ordinary statistics: **covariance** between two variables $A$ and $B$ measures whether they move together. Formula (defining every symbol):
$$
\text{Cov}(A,B) = E[(A-\mu_A)(B-\mu_B)]
$$
Plain English: take how far $A$ is from its own average ($A - \mu_A$), take how far $B$ is from its own average ($B-\mu_B$), multiply those two "distances from average" together, then average that product over all observations. If $A$ and $B$ tend to be above-average at the same time (or below-average at the same time), the product is usually positive → positive covariance (they move together). If one is high when the other is low, the product is usually negative → negative covariance (they move oppositely). If there's no relationship, the products cancel out to roughly zero.

Now here's the trick for time series: instead of comparing two *different* variables $A$ and $B$, we compare **the same series to a lagged/shifted version of itself**. That's literally what "auto" (self) correlation means.

**Autocovariance at lag $k$**, defining every symbol:
$$
\gamma(k) = \text{Cov}(x_t, x_{t-k}) = E[(x_t - \mu)(x_{t-k}-\mu)]
$$
- $\gamma$ (gamma) is just the conventional symbol name for autocovariance — no deeper meaning, just notation you need to recognize.
- $\mu$ (mu) = the overall mean of the series (assumed constant across time — this assumption is exactly the stationarity idea from Phase 2, and it's *required* for this formula to even make sense, since we're using a single $\mu$ for every time point).
- $x_t$ = value today, $x_{t-k}$ = value $k$ steps back.

**Plain English translation of the whole formula:** "Take today's deviation from the mean, multiply it by the deviation-from-mean of the value $k$ steps ago, and average that product across the whole series." If this comes out positive and large, it means: when the series is above its average today, it tends to have ALSO been above average $k$ steps ago (and vice versa for below-average) — i.e., there's a real, measurable echo/memory of length $k$.

**Special case, sanity check:** $\gamma(0) = \text{Cov}(x_t,x_t) = \text{Var}(x_t)$ — comparing the series to itself at lag 0 is just its own variance. This will matter in the next section.

---

## 3. Autocorrelation Function (ACF): making autocovariance interpretable

Raw autocovariance $\gamma(k)$ has a problem: its *units* depend on the scale of your data (e.g., if you measure revenue in dollars vs. thousands of dollars, $\gamma(k)$ changes size even though the underlying relationship strength hasn't). We fix this by normalizing — dividing by the variance — exactly the same trick ordinary correlation uses to turn covariance into a clean number between −1 and +1.

$$
\rho(k) = \frac{\gamma(k)}{\gamma(0)} = \frac{\text{Cov}(x_t,x_{t-k})}{\text{Var}(x_t)}
$$

- $\rho$ (rho) = the conventional symbol for the **autocorrelation function (ACF)** at lag $k$.
- We divide by $\gamma(0)$ (the variance) so the result is always between $-1$ and $+1$, no matter the original units of your data.

**Plain English:** $\rho(k)$ tells you, on a clean, comparable scale, "how strongly does the series at lag $k$ echo the series today?" $\rho(k)=1$ means perfect positive echo (extremely predictable from $k$ steps back), $\rho(k)=0$ means no linear relationship at that lag, $\rho(k)=-1$ means perfect inverse echo.

**Important properties to just know (no need to prove yet):**
- $\rho(0) = 1$ always (a series is perfectly correlated with itself at lag 0 — trivially true, always check this as a sanity anchor).
- $\rho(k) = \rho(-k)$ — the ACF is symmetric; looking $k$ steps forward or $k$ steps backward gives the same value (because covariance itself is symmetric in its two arguments).
- For white noise (Phase 2, section 4), $\rho(k) = 0$ for every $k \neq 0$ — this is the formal restatement of "white noise has zero memory," now expressed precisely using the tool we just built.

### 3.1 What does an ACF plot ("correlogram") actually look like?
A **correlogram** is just the standard name for a bar chart of $\rho(k)$ against $k$ (lag on the x-axis, correlation value 0 to 1 on the y-axis, with bars for lag 1, 2, 3, ...). This is one of the most commonly shown plots in all of time series analysis — you will see it constantly, including in interviews.

---

## 4. Sample ACF: how do we compute this from REAL, finite data?

Everything above assumed we knew the "true" underlying process. In real life we only have a finite dataset of $n$ observations $x_1, \dots, x_n$, and we must **estimate** $\rho(k)$ from that data. The estimator is called the **sample ACF**, denoted $\hat{\rho}(k)$ (the hat symbol $\hat{}$ is standard notation across all of statistics meaning "this is an estimate computed from data, not the unknown true value").

$$
\hat{\rho}(k) = \frac{\sum_{t=k+1}^{n} (x_t - \bar{x})(x_{t-k}-\bar{x})}{\sum_{t=1}^{n}(x_t-\bar{x})^2}
$$

Let's unpack every piece:
- $\bar{x}$ = the sample mean (ordinary average) of all $n$ observations — our estimate of $\mu$.
- Numerator: for every valid pair of points that are exactly $k$ steps apart, multiply their deviations from the sample mean, then sum all those products. Note the sum starts at $t=k+1$ because you need $x_{t-k}$ to exist — you can't look $k$ steps before the very first data point.
- Denominator: the sum of squared deviations from the mean across ALL $n$ points — this is essentially $n$ times the sample variance, and it plays the role of $\gamma(0)$ from before (normalizing).

**Confidence bands:** In practice, when you plot a sample ACF, you'll see two dashed horizontal lines (often near $\pm 1.96/\sqrt{n}$). Here's where that comes from, in plain English: if the TRUE underlying process were pure white noise (zero real autocorrelation at every lag), then due to random sampling noise alone, the *sample* ACF won't be exactly zero — it'll wobble a little around zero just by chance. Statistical theory tells us that for large $n$, this random wobble is approximately Normally distributed with standard deviation $\approx 1/\sqrt{n}$. The number 1.96 comes from the standard Normal distribution — it's the cutoff point beyond which only 5% of purely random wobble would fall (a 95% confidence band, if you've seen "1.96" before in the context of confidence intervals, this is the exact same constant, doing the exact same job). **Practical reading rule: if a sample ACF bar at some lag pokes out beyond the dashed band, that's evidence of a REAL relationship at that lag — not just random noise-wobble.** If it stays inside the band, you can't confidently distinguish it from zero/white noise.

---

## 5. Partial Autocorrelation Function (PACF): removing the "middleman" effect

This is the trickiest new concept in this phase, so we go extra slow.

**The problem the plain ACF has:** Suppose today's value is strongly related to yesterday's value (lag 1 relationship is real and strong). Because yesterday is related to the day before, and the day before is related to the day before that, this lag-1 relationship can "chain" through time and create an *apparent* relationship at lag 2, lag 3, etc. — even if there's no DIRECT connection between today and 2 days ago; it's all just an indirect echo transmitted through yesterday.

**Analogy:** Imagine gossip spreading. Alice tells Bob a secret. Bob tells Carol. Carol tells Dave. If you only look at "does Dave know a version of the secret," yes — Dave's knowledge is correlated with Alice's original secret. But Dave didn't get it DIRECTLY from Alice — he got it through the Bob→Carol chain. The plain ACF between "Alice" and "Dave" would show a positive relationship, but it's entirely indirect, mediated through Bob and Carol.

**What we actually want:** a tool that measures the DIRECT relationship between $x_t$ and $x_{t-k}$, after stripping away — "controlling for" — all the in-between values $x_{t-1}, x_{t-2}, \dots, x_{t-k+1}$. This is exactly what **partial correlation** means in general statistics (not unique to time series) — "partial" here means "with the effect of other variables removed/held constant," the same way "partial" is used in "controlling for confounders" in regression.

**Formal definition (in words first):** The **partial autocorrelation at lag $k$**, written $\phi_{kk}$ (we'll explain this double-subscript notation below), is the correlation between $x_t$ and $x_{t-k}$ AFTER removing the linear effect of everything in between ($x_{t-1}$ through $x_{t-k+1}$) from both of them.

**Why the notation $\phi_{kk}$?** This comes from a specific computational method (the Yule-Walker equations, which we'll derive properly in Phase 6 when we cover AR models) where the PACF at lag $k$ turns out to equal the LAST coefficient $\phi_k$ in a hypothetical AR($k$) regression of $x_t$ on its $k$ most recent past values. The double subscript $\phi_{kk}$ specifically denotes "the $k$-th coefficient, in a model that uses $k$ total lags." Don't worry about deriving this yet — just recognize the notation when you see it; we'll build the full derivation with Yule-Walker equations in Phase 6.

**A more intuitive computational description you CAN use right now:** to compute the PACF at lag $k$, imagine running a regression of $x_t$ on $x_{t-1}, x_{t-2}, \ldots, x_{t-k}$ (all as predictors simultaneously). The PACF at lag $k$ is the estimated regression coefficient specifically attached to $x_{t-k}$ in that regression — i.e., "how much does $x_{t-k}$ move $x_t$, holding all the closer/intermediate lags constant?" This is literally answering: "does lag $k$ add any NEW, direct predictive information beyond what the closer lags already gave us?"

---

## 6. Why we need BOTH ACF and PACF: the model identification cheat sheet

This is the single most practically useful takeaway of this entire phase — it's how you'll "read" a time series' shape and guess its underlying model, a task you will 100% be asked about in interviews.

| Process type | ACF shape | PACF shape |
|---|---|---|
| White noise | Zero everywhere (all bars inside confidence band) | Zero everywhere |
| AR(p) (autoregressive, Phase 6) | **Tails off** gradually (decays slowly, possibly oscillating) | **Cuts off sharply** after lag $p$ (nothing significant beyond lag $p$) |
| MA(q) (moving average, Phase 6) | **Cuts off sharply** after lag $q$ | **Tails off** gradually |
| ARMA(p,q) | Tails off | Tails off |

**Why does an AR process's PACF cut off sharply but its ACF doesn't?** Intuition (full derivation comes in Phase 6): in an AR(p) process, $x_t$ is DIRECTLY built from exactly the last $p$ values (plus noise) — nothing beyond lag $p$ is DIRECTLY used to build $x_t$, so once you strip away the indirect chained effects (which is exactly what PACF does), there's genuinely nothing left beyond lag $p$ — hence PACF cuts off cleanly. But the plain ACF still shows a gradually decaying pattern beyond lag $p$, because of the indirect "gossip chain" effect discussed in section 5 — lag $p+1$ is still indirectly connected through the chain of closer lags, even though it's not DIRECTLY in the model.

**Why is it the mirror image for MA(q)?** An MA(q) process (full definition in Phase 6) builds $x_t$ directly out of the last $q$ NOISE terms (not past values of $x$ itself) — this gives it a naturally sharp cutoff in the plain ACF at lag $q$, but when you try to express it in terms of past $x$ values (which is what PACF implicitly does), it turns out you need infinitely many past $x$ terms to reconstruct that noise-based structure — hence PACF tails off slowly instead of cutting.

**You don't need to fully derive this yet** — just memorize this table as a practical lookup tool. In Phase 6, we will derive precisely *why* each row is true, using the actual AR and MA formulas.

---

## 7. Full numerical worked example: compute ACF by hand

Let's use a small dataset, 8 points: $x = [4, 6, 5, 7, 6, 8, 7, 9]$

**Step 1 — Compute the sample mean.**
$\bar{x} = (4+6+5+7+6+8+7+9)/8 = 52/8 = 6.5$

**Step 2 — Compute deviations from the mean $(x_t - \bar{x})$:**
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

**Step 3 — Compute the denominator ($\gamma(0)$, sum of squared deviations):**
$(-2.5)^2+(-0.5)^2+(-1.5)^2+(0.5)^2+(-0.5)^2+(1.5)^2+(0.5)^2+(2.5)^2$
$= 6.25+0.25+2.25+0.25+0.25+2.25+0.25+6.25 = 18.0$

**Step 4 — Compute the numerator for lag 1** ($\sum_{t=2}^{8}(x_t-\bar x)(x_{t-1}-\bar x)$ — pair each deviation with the PREVIOUS deviation):
| pair (t, t-1) | product |
|---|---|
| (2,1): (−0.5)(−2.5) | 1.25 |
| (3,2): (−1.5)(−0.5) | 0.75 |
| (4,3): (0.5)(−1.5) | −0.75 |
| (5,4): (−0.5)(0.5) | −0.25 |
| (6,5): (1.5)(−0.5) | −0.75 |
| (7,6): (0.5)(1.5) | 0.75 |
| (8,7): (2.5)(0.5) | 1.25 |

Sum = $1.25+0.75-0.75-0.25-0.75+0.75+1.25 = 2.25$

**Step 5 — Compute $\hat\rho(1)$:**
$$
\hat\rho(1) = \frac{2.25}{18.0} = 0.125
$$

**Interpretation:** a fairly weak positive lag-1 relationship (0.125 is small, close to zero, and would very likely fall inside the confidence band for a dataset this tiny — with $n=8$, the band is roughly $\pm 1.96/\sqrt{8} \approx \pm 0.693$, so 0.125 is nowhere near significant. This is expected — you genuinely need much more data than 8 points to reliably estimate autocorrelation; this example is purely to show you the mechanical calculation, not to draw a real conclusion).

**Step 6 — Lag 2, same process** (pair each deviation with the value TWO steps back):
| pair (t, t-2) | product |
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

Interesting — lag 2 came out STRONGER than lag 1 here. Looking at the raw data $[4,6,5,7,6,8,7,9]$, you can see it visually: there's a zig-zag-while-trending pattern where every-other point tends to line up (4,5,6,7 vs 6,7,8,9 sort of alternating pattern) — this kind of small-sample quirk is a good reminder that with tiny datasets, sample ACF values can look larger or in a different order than you'd expect from the "true" underlying pattern, purely due to sampling noise. This is precisely why real-world ACF analysis is always done with the confidence bands in mind, and with as much data as reasonably available.

---

## 8. The Ljung-Box test: formalizing "does this look like white noise?"

We mentioned this test informally in Phase 1. Now that you understand $\hat\rho(k)$, here's the actual formula:

$$
Q = n(n+2)\sum_{k=1}^{h}\frac{\hat\rho(k)^2}{n-k}
$$

Breaking down every piece:
- $n$ = number of observations.
- $h$ = the number of lags you're jointly testing (e.g., you might test lags 1 through 10 all at once — a common choice).
- $\hat\rho(k)^2$ = the squared sample autocorrelation at each lag (squaring makes every term positive, so positive and negative correlations don't cancel out — we care about MAGNITUDE of leftover structure, not direction).
- Dividing by $(n-k)$ is a small-sample correction — sample ACF estimates at higher lags are based on fewer valid pairs of points (as you saw in section 4, the sum starts later for larger $k$), so they're noisier, and this division down-weights that extra noise appropriately.

**Plain English: $Q$ adds up evidence of "leftover correlation" across many lags at once into a single number.** Under the null hypothesis that the true process is white noise (no real autocorrelation anywhere), $Q$ follows a known reference distribution (a Chi-squared distribution with $h$ degrees of freedom — you don't need to derive this, just know it's a standard statistical distribution used to get a p-value). **Practical use:** you fit a model (say, an ARIMA model, Phase 6), you look at its residuals (leftover unexplained noise), you compute this $Q$ statistic on the residuals' ACF. If $Q$ is large (giving a small p-value, conventionally <0.05), that's evidence your residuals STILL have structure left — meaning your model failed to capture everything, and you should go back and refine it. If $Q$ is small/p-value is large, you can't reject the idea that your residuals are just white noise — a good sign that your model has extracted everything it usefully could.

---

## 9. Quick self-check questions

1. What's the difference, in plain English, between what ACF measures and what PACF measures at lag 3?
   *(Answer: ACF at lag 3 measures the total relationship between $x_t$ and $x_{t-3}$, including indirect effects chained through lags 1 and 2. PACF at lag 3 measures ONLY the direct relationship between $x_t$ and $x_{t-3}$, after removing/controlling for the effects of lags 1 and 2.)*
2. If you see an ACF plot that cuts off sharply to zero after lag 2, and a PACF that tails off slowly — what type of process does this suggest (using the Phase 3 table)?
   *(Answer: MA(2) — sharp ACF cutoff at lag $q$ + slowly tailing PACF is the MA(q) signature.)*
3. Why does dividing by $(n-k)$ in the Ljung-Box formula matter, rather than just always dividing by $n$?
   *(Answer: because sample ACF at higher lags $k$ is computed from fewer valid data pairs — only $n-k$ pairs exist — making those estimates inherently noisier, so the formula appropriately down-weights/adjusts for that reduced reliability.)*
4. For white noise, what should $\rho(k)$ equal for every $k \neq 0$, and why?
   *(Answer: 0 — because by definition, white noise has zero covariance between any two different time points (Phase 2, Property 3), and ACF is just normalized covariance, so zero covariance directly implies zero ACF.)*

---

## What's next
Phase 4 covers **stationarity formally**: the precise mathematical definition, why it's required for everything we've built so far to be valid, how to fix non-stationary data (differencing, transformations), and the actual statistical tests (ADF, KPSS) used to check for it in practice — including the full derivation of the ADF test statistic, which directly builds on the random-walk-vs-AR(1) distinction from Phase 2, section 5.3.

Say "next" for Phase 4, or ask for more ACF/PACF hand-computation drills first.
