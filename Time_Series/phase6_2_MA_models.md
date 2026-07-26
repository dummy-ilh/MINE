# Phase 6, Part 2 of 5: Moving Average MA(q) Models

Recall the roadmap: 6.1 AR(p) [done] → **6.2 MA(q) (this file)** → 6.3 ARMA/ARIMA/SARIMA → 6.4 Estimation & model selection → 6.5 Diagnostics & forecasting.

This part is a genuine mirror image of Part 1 — same derivation style, opposite conclusions. Where AR models had ACF that tails off and PACF that cuts off, MA models will have the reverse. Let's derive why, from scratch.

---

## 1. Building MA(1) from scratch

**The core idea, in plain English before any formula:** instead of today's value depending on YESTERDAY'S VALUE (like AR did), today's value depends on YESTERDAY'S SHOCK (the noise term, not the observed series itself).

$$
x_t = \mu + \varepsilon_t + \theta\, \varepsilon_{t-1}
$$

Defining every symbol:
- $\mu$ = the overall mean of the series (just a constant offset — MA models, unlike AR models, don't need any special derivation to find the mean; since $E[\varepsilon_t]=E[\varepsilon_{t-1}]=0$, taking the expectation of both sides immediately gives $E[x_t]=\mu$, directly, no algebra needed. Contrast this with AR(1) in Part 1 section 3, where finding the mean required solving an equation — MA models are actually simpler in this one specific respect).
- $\varepsilon_t$ = today's fresh white noise shock.
- $\varepsilon_{t-1}$ = YESTERDAY'S white noise shock — notice this already happened and is now a KNOWN, fixed past value by the time we're at time $t$ (it's not "random" from today's viewpoint anymore, it's history).
- $\theta$ (theta) = the **MA coefficient** — how strongly yesterday's shock still echoes into today's observed value.

**Plain English reading:** "today's value = the average level, plus a brand new random surprise, plus a LEFTOVER ECHO of some fraction ($\theta$) of yesterday's surprise." Think of it like ripples in a pond: yesterday's disturbance ($\varepsilon_{t-1}$) hasn't fully died out yet — a fraction of it is still rippling into today's observation, on top of today's own brand-new disturbance.

**Order $q$ meaning:** MA(1) means only YESTERDAY's shock still echoes (memory of length 1 step). We'll generalize to MA(q) — memory of $q$ steps of past shocks — in section 5.

---

## 2. Deriving the ACF of MA(1) — and proving the sharp cutoff

**Compute $\gamma(1)$** (autocovariance at lag 1), working with the mean-centered version $x_t - \mu = \varepsilon_t+\theta\varepsilon_{t-1}$ for cleaner algebra:
$$
\gamma(1) = E[(x_t-\mu)(x_{t-1}-\mu)] = E\big[(\varepsilon_t+\theta\varepsilon_{t-1})(\varepsilon_{t-1}+\theta\varepsilon_{t-2})\big]
$$
Expand this product (multiply out every pair of terms, exactly like expanding $(a+b)(c+d) = ac+ad+bc+bd$ in ordinary algebra):
$$
= E[\varepsilon_t\varepsilon_{t-1}] + \theta E[\varepsilon_t\varepsilon_{t-2}] + \theta E[\varepsilon_{t-1}\varepsilon_{t-1}] + \theta^2 E[\varepsilon_{t-1}\varepsilon_{t-2}]
$$
Now apply the white noise property (Phase 2): $E[\varepsilon_i \varepsilon_j] = 0$ whenever $i \neq j$ (no correlation between DIFFERENT noise terms), and $E[\varepsilon_i\varepsilon_i] = \text{Var}(\varepsilon_i) = \sigma^2$ (same index = just the variance). Looking at each of our four terms: $E[\varepsilon_t\varepsilon_{t-1}]=0$ (different indices), $E[\varepsilon_t\varepsilon_{t-2}]=0$ (different indices), $E[\varepsilon_{t-1}\varepsilon_{t-1}]=\sigma^2$ (SAME index), $E[\varepsilon_{t-1}\varepsilon_{t-2}]=0$ (different indices). Only ONE term survives:
$$
\gamma(1) = \theta\sigma^2
$$

**Now compute $\gamma(2)$** the exact same way:
$$
\gamma(2) = E\big[(\varepsilon_t+\theta\varepsilon_{t-1})(\varepsilon_{t-2}+\theta\varepsilon_{t-3})\big]
$$
Every single term here involves TWO DIFFERENT noise indices ($t$ vs $t-2$, $t$ vs $t-3$, $t-1$ vs $t-2$, $t-1$ vs $t-3$ — none of these pairs ever match up to the same index), so EVERY term is zero by the white noise property:
$$
\gamma(2) = 0
$$
**And by the exact same reasoning, $\gamma(k) = 0$ for EVERY $k \geq 2$** — there's simply no overlapping shock index left to produce a nonzero product once you look 2 or more steps apart, because MA(1) only ever involves TWO consecutive noise terms ($\varepsilon_t$ and $\varepsilon_{t-1}$) at any given time point, and once your lag is 2+, those two-term windows (for $x_t$ and $x_{t-2}$) share no common noise term at all.

**Also compute $\gamma(0)$ for the normalization** (needed to get $\rho$ from $\gamma$, exactly as in Phase 3):
$$
\gamma(0) = \text{Var}(x_t) = E[(\varepsilon_t+\theta\varepsilon_{t-1})^2] = E[\varepsilon_t^2] + 2\theta E[\varepsilon_t\varepsilon_{t-1}] + \theta^2E[\varepsilon_{t-1}^2] = \sigma^2 + 0 + \theta^2\sigma^2 = \sigma^2(1+\theta^2)
$$

**Putting it together, the complete ACF of MA(1):**
$$
\rho(1) = \frac{\theta\sigma^2}{\sigma^2(1+\theta^2)} = \frac{\theta}{1+\theta^2}, \qquad \rho(k) = 0 \text{ for all } k \geq 2
$$

**This is the formal, derived proof of the sharp ACF cutoff from the Phase 3 table.** Unlike AR(1)'s ACF, which decayed smoothly forever ($\phi^k$, never exactly zero at any finite lag), MA(1)'s ACF has EXACTLY ONE nonzero value (at lag 1) and then is EXACTLY, PRECISELY zero at every lag beyond that — a genuine mathematical cliff-edge, not just a "gets very small" approximation. **This is literally why the rule is called a "cutoff" for MA and a "tail-off" for AR** — now you've derived both terms from first principles, not just memorized them.

**General pattern (stated here, will be evident from the derivation logic, generalizes directly):** for MA(q), $\rho(k) \neq 0$ for $k \leq q$ and $\rho(k) = 0$ exactly for $k > q$ — the cutoff happens exactly at the order of the model. **This gives you a genuinely practical, direct way to READ THE ORDER $q$ straight off a real correlogram**: count how many lags show significant bars before it drops to (statistically indistinguishable from) zero — that count IS your candidate $q$.

---

## 3. Why does MA's PACF tail off instead of cutting off? (Completing the mirror image)

We asserted this in the Phase 3 table without derivation; here's the intuition (a full rigorous derivation requires inverting an infinite series, more machinery than we need at this stage — the intuition below is what you actually need for interviews).

**Try to express MA(1) in terms of past $x$ values instead of past $\varepsilon$ values** (this is called "inverting" the MA model). Start from $x_t - \mu = \varepsilon_t + \theta\varepsilon_{t-1}$, so $\varepsilon_t = (x_t-\mu) - \theta\varepsilon_{t-1}$. Now substitute this SAME relationship one step back, recursively, to eliminate $\varepsilon$ terms entirely:
$$
\varepsilon_t = (x_t-\mu) - \theta(x_{t-1}-\mu) + \theta^2(x_{t-2}-\mu) - \theta^3(x_{t-3}-\mu) + \dots
$$
(You can verify this pattern by repeated substitution, the same technique used in Phase 2 and Part 1 of this phase — we won't belabor every algebra step here, just notice the RESULT.)

**This means:** to express $x_t$ purely in terms of ITS OWN past values (rearranging the above), you need an INFINITE number of past lags, with geometrically shrinking coefficients ($\theta, \theta^2, \theta^3,\ldots$). **This is exactly why the PACF tails off (never cuts off cleanly) for an MA process**: since the "true" autoregressive representation of an MA(1) process technically requires infinitely many past lags, controlling for/partialling-out effects at each successive lag (which is what PACF does, recall Phase 3 section 5) never finds a clean point where the direct relationship completely vanishes — it just fades out gradually, mirroring the AR(1) ACF's smooth geometric decay from Part 1, section 4, but now showing up in the PACF instead of the ACF.

**This "AR(∞) representation" of an MA process is the exact mirror image of the "MA(∞) representation" of an AR process that we derived in Part 1, section 2** — every AR process can be rewritten as an infinite MA, and (under a condition we cover next) every MA process can be rewritten as an infinite AR. This symmetry is the deep reason AR and MA are "dual" to each other, and it's WHY the ACF/PACF signature table is a clean mirror image between the two model types.

---

## 4. Invertibility: the MA equivalent of AR's stationarity condition

**Important, often-confused fact stated upfront: MA models are ALWAYS stationary, automatically, no matter what value $\theta$ takes.** Look back at section 2 — we computed $\gamma(0), \gamma(1), \gamma(2),\ldots$ and every single one came out as a fixed constant, not depending on $t$ at all, for ANY value of $\theta$. There's no equivalent "boundary case" like AR's $\phi=1$ — finite-order MA processes are unconditionally, structurally stationary by their very construction (a finite sum of stationary white noise terms is always stationary — no infinite accumulation risk like the random walk had).

**So why do we need a separate condition (invertibility) at all?** Look back at section 3: we showed that recovering $\varepsilon_t$ from past $x$ values requires an infinite series with coefficients $\theta, \theta^2, \theta^3,\ldots$. **For this infinite series to actually converge to a finite, sensible answer (rather than blowing up), we need $|\theta| < 1$.** This condition is called **invertibility** — the model can be "inverted" (rewritten in terms of past $x$'s) in a well-behaved, convergent way.

**Why does invertibility matter practically, if MA is already always stationary?** Two crucial practical reasons: (1) **Non-uniqueness without it**: it turns out that an MA(1) with a specific $\theta$ and noise variance $\sigma^2$ produces EXACTLY the same ACF as another MA(1) with coefficient $1/\theta$ and rescaled variance $\theta^2\sigma^2$ (you can verify this yourself: plug $1/\theta$ into the $\rho(1)=\theta/(1+\theta^2)$ formula and notice it gives the identical value — try it: $\rho(1)$ for $\theta=2$ gives $2/5=0.4$; for $\theta=0.5$ gives $0.5/1.25=0.4$ — IDENTICAL). Without the invertibility restriction $|\theta|<1$, there would be TWO different, equally-valid-looking parameter values fitting the same observed correlation pattern — a genuine identifiability problem. Imposing $|\theta|<1$ picks out exactly ONE of these two mathematically-equivalent solutions as "the" invertible one, resolving the ambiguity. (2) **Forecasting practicality**: since real-world forecasting ultimately needs to reconstruct/estimate the unobserved noise terms from observed data (to make predictions), you need that infinite AR-representation series from section 3 to actually converge — otherwise you can't practically compute one-step-ahead forecasts from real data at all.

**General MA(q) invertibility condition (parallel structure to AR's stationarity condition from Part 1, section 5.1):** all roots of the MA characteristic polynomial $\Theta(z) = 1+\theta_1 z + \theta_2 z^2+\dots+\theta_q z^q$ must lie OUTSIDE the unit circle — structurally the exact same TYPE of condition as AR's stationarity requirement, just applied to the $\theta$ coefficients instead of the $\phi$ coefficients, and named "invertibility" instead of "stationarity" purely by convention (a genuinely common point of confusion — many people are surprised these are two different NAMES for structurally the same kind of mathematical condition, just protecting against two different practical problems: AR's stationarity protects against variance blowing up; MA's invertibility protects against non-unique, non-convergent representations).

---

## 5. Generalizing to MA(q): the full model

$$
x_t = \mu + \varepsilon_t + \theta_1\varepsilon_{t-1} + \theta_2\varepsilon_{t-2} + \dots + \theta_q \varepsilon_{t-q}
$$

Using the lag operator (same tool as Part 1, section 5): $x_t - \mu = (1+\theta_1 L + \theta_2 L^2+\dots+\theta_q L^q)\varepsilon_t = \Theta(L)\varepsilon_t$.

**Plain English summary of the whole model:** today's deviation from the mean is built from a weighted combination of the CURRENT shock plus the last $q$ shocks, each with its own coefficient controlling how much that particular shock's echo still lingers. Following the exact derivation logic from sections 2–3 (just with more cross-terms to track, same underlying technique), the ACF cuts off sharply exactly at lag $q$, and the PACF tails off — completing the full mirror-image relationship to AR(p) promised at the start of this file.

---

## 6. Numerical worked example: MA(1) by hand

Let $\theta = 0.7$, $\sigma^2=1$, $\mu=0$, noise draws $\varepsilon_0 = 0.5$ (needed as the "previous" shock to start), $\varepsilon_1=1.2, \varepsilon_2=-0.8, \varepsilon_3=0.3, \varepsilon_4=-1.0$.

**Step 1 — Generate the series** using $x_t = \varepsilon_t + 0.7\varepsilon_{t-1}$:
$x_1 = \varepsilon_1 + 0.7\varepsilon_0 = 1.2+0.7(0.5)=1.2+0.35=1.55$
$x_2 = \varepsilon_2+0.7\varepsilon_1 = -0.8+0.7(1.2)=-0.8+0.84=0.04$
$x_3 = \varepsilon_3+0.7\varepsilon_2=0.3+0.7(-0.8)=0.3-0.56=-0.26$
$x_4=\varepsilon_4+0.7\varepsilon_3=-1.0+0.7(0.3)=-1.0+0.21=-0.79$

Series: $[1.55, 0.04, -0.26, -0.79]$

**Step 2 — Compute theoretical ACF using our derived formulas:**
$$
\rho(1) = \frac{\theta}{1+\theta^2} = \frac{0.7}{1+0.49}=\frac{0.7}{1.49}\approx 0.4698
$$
$$
\rho(2) = \rho(3) = \dots = 0 \quad\text{(exactly, by construction)}
$$

**Step 3 — Sanity check the invertibility condition:** $|\theta|=0.7 < 1$ ✓ — this MA(1) is invertible, meaning it has a valid, convergent AR(∞) representation and doesn't suffer from the non-uniqueness problem discussed in section 4.

**Step 4 — Contrast with Part 1's AR(1) numerical example:** notice how sharply different the ACF SHAPES are — AR(1) with $\phi=0.6$ gave a smoothly decaying sequence $0.6, 0.36, 0.216, 0.1296,\ldots$ (never hitting zero); this MA(1) with $\theta=0.7$ gives a SINGLE nonzero value $0.4698$ and then EXACT zeros forever after. If you were handed real sample ACF bars that looked like one of these two patterns, you should now be able to immediately identify which underlying model family is more consistent with the data — this pattern recognition, backed by the derivations you've now walked through twice, is precisely the skill Box-Jenkins model identification (and time series interviews) test for.

---

## 7. Quick self-check questions

1. Why does $\gamma(2) = 0$ exactly for an MA(1) process, using the logic from section 2?
   *(Answer: computing $\gamma(2)$ requires expanding a product of $(\varepsilon_t+\theta\varepsilon_{t-1})$ and $(\varepsilon_{t-2}+\theta\varepsilon_{t-3})$ — every possible pairing of indices across these two brackets involves two DIFFERENT time indices (none of $t, t-1$ matches any of $t-2, t-3$), so every term vanishes under the white-noise zero-covariance property, leaving nothing.)*
2. In plain English, what does "invertibility" protect against, and how is that DIFFERENT from what AR's stationarity condition protects against?
   *(Answer: invertibility (|θ|<1) ensures the MA model's implied infinite-AR representation converges and resolves a non-uniqueness problem where two different θ values produce identical ACFs; AR's stationarity condition (|φ|<1) instead ensures the process's VARIANCE stays finite rather than blowing up/accumulating forever. Different underlying problems, structurally similar-looking root conditions.)*
3. If a sample correlogram shows significant bars at lags 1, 2, and 3, then nothing significant afterward, what MA order would you propose as a starting candidate, and why?
   *(Answer: MA(3) — because the ACF cutoff happens exactly at lag q for an MA(q) process, so three significant lags followed by an abrupt drop to insignificance directly matches q=3.)*
4. Are finite-order MA models ever non-stationary? Why or why not?
   *(Answer: No — a finite-order MA process is a finite sum of stationary white noise terms, which is always stationary regardless of the θ coefficient values; there's no accumulation-to-infinity risk the way there is with AR's φ=1 boundary case.)*

---

## What's next
**Part 3 of Phase 6** combines everything so far into **ARMA(p,q)**, then extends to **ARIMA(p,d,q)** (bringing back the differencing from Phase 4 to handle non-stationary data) and **SARIMA** (adding a seasonal structure on top, echoing the seasonal differencing and Holt-Winters seasonal logic from Phase 5). We'll also cover the Wold Decomposition Theorem — the deep theoretical result explaining WHY ARMA models are general enough to approximate almost any stationary process.

Say "next" for Part 3, or ask for more MA(q) drilling first (e.g., deriving the ACF of MA(2) by hand, which has two nonzero lags instead of one).
