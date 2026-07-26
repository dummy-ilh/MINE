# Phase 10: Volatility Modeling — ARCH and GARCH

Every model so far (AR, MA, ARIMA, ETS, Kalman) assumed CONSTANT variance (Phase 4's stationarity Condition 2). This phase tackles data where the variance ITSELF changes over time in a structured way — the hallmark of financial data (stock returns, exchange rates) and a near-guaranteed topic in any quant-flavored interview.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $r_t$ | the return (or mean-adjusted series value) at time $t$ — what we're modeling here |
| $\sigma_t^2$ | the **conditional variance** at time $t$ — plain English: "how volatile/wild do we expect things to be RIGHT NOW, given what just happened recently" — this is the whole new idea of this phase |
| $z_t$ | a standardized white noise term, $z_t \sim N(0,1)$ (mean 0, variance exactly 1 — a "pure," unscaled random shock) |
| $\omega$ (omega) | a baseline/floor constant in the variance equation |
| $\alpha_i$ (in this file) | ARCH coefficients — how much PAST SQUARED SHOCKS drive today's variance (careful: yet another reuse of $\alpha$, unrelated to Phase 5's smoothing parameter or Phase 4's ADF regression coefficient) |
| $\beta_i$ (in this file) | GARCH coefficients — how much PAST VARIANCE ITSELF persists into today's variance (careful: unrelated to Phase 5's Holt trend smoothing parameter $\beta^*$) |
| "conditional" | a recurring word in this phase meaning "given information up to now" — i.e., "conditional variance" = "variance, given what we know so far," which can change moment to moment, unlike the FIXED, unconditional variance from Phase 6 |

---

## 2. The stylized facts: what real financial data actually looks like (motivation before formulas)

If you plot daily stock returns, you'll notice patterns that NONE of our models so far can explain:

**Stylized fact 1 — Volatility clustering.** Big price swings tend to cluster together in TIME — a few genuinely calm weeks, then a turbulent stretch (e.g., around a crisis or earnings surprise) where big moves keep happening for a while, then calm returns. **Plain English: today's volatility being high makes TOMORROW'S volatility more likely to also be high** — this is itself a form of "memory," but memory in the SIZE of shocks, not in their direction or level (the returns themselves might still average out to roughly zero, with no real autocorrelation in the RAW returns — but the SQUARED/absolute size of the returns IS autocorrelated).

**Stylized fact 2 — Fat tails.** Extreme moves (very large gains or losses) happen more often than a plain Normal/bell-curve distribution would predict (recall the Q-Q plot diagnostic from Phase 6, Part 5, section 3.4 — financial returns are a classic case where this shows up).

**Stylized fact 3 — The leverage effect.** Volatility tends to rise MORE after a large NEGATIVE shock (a big price drop) than after an equally-sized POSITIVE shock (a big price gain) — a genuine asymmetry (briefly flagged here; the basic ARCH/GARCH models below don't capture this asymmetry, which is exactly why extensions like EGARCH/GJR-GARCH exist, mentioned at the end of this file).

**Why none of our previous tools handle this:** ARIMA models the MEAN (the expected LEVEL of the series). Even a perfectly-fit ARIMA model, after removing all mean-structure, would leave behind RESIDUALS that still show volatility clustering in their SIZE — recall Phase 6, Part 5, section 3.1's residual diagnostic check for "constant spread over time" — **volatility clustering is precisely a violation of that specific diagnostic check, a pattern in the VARIANCE of the residuals rather than in their mean/level.** We need an entirely separate model, layered on TOP of (or applied directly to) the residuals, specifically for the variance itself.

---

## 3. Building ARCH(1) from scratch — modeling the VARIANCE, not the mean

**The core idea, in plain English before any formula:** just like AR(1) said "today's LEVEL depends on yesterday's level," **ARCH(1) says "today's VARIANCE depends on yesterday's SQUARED shock."** Why squared? Because we care about the MAGNITUDE of yesterday's surprise (how big was it, regardless of direction), and squaring is the standard way to measure "size" while always producing a positive number (exactly the same reason Ljung-Box squared the ACF values back in Phase 3, section 8 — to measure magnitude of deviation without direction/sign canceling things out).

**The two-equation structure (notice the parallel to Phase 9's two-equation state space setup):**

**Equation 1 — the return itself:**
$$
r_t = \sigma_t\, z_t
$$
**Plain English:** "today's return equals today's volatility level, multiplied by a pure standardized random shock." $z_t$ is just plain white noise scaled to have variance exactly 1 (Phase 2's white noise, standardized) — ALL of the actual TIME-VARYING volatility behavior comes from $\sigma_t$, which changes over time; $z_t$ itself is always drawn the same simple way at every step.

**Equation 2 — the conditional variance itself (the genuinely new equation):**
$$
\sigma_t^2 = \omega + \alpha_1\, r_{t-1}^2
$$
**Plain English, piece by piece:**
- $\omega$ = a baseline/floor level of variance — "how volatile things are on average, at minimum, ignoring any recent shocks" (must be a positive number, or variance could come out negative, which is nonsensical).
- $r_{t-1}^2$ = YESTERDAY's squared return — a direct, simple MEASURE of how big yesterday's surprise was.
- $\alpha_1$ = how strongly that squared shock carries forward into TODAY's volatility estimate.

**The whole sentence: "today's expected volatility = a baseline floor level, PLUS an extra kick proportional to how big yesterday's surprise was."** If yesterday had a huge move (large $r_{t-1}^2$), today's variance is predicted to be elevated too — **this IS volatility clustering, built directly into the model's structure, mechanically.**

**Why "conditional" is the exactly right word here (worth pausing on, since it's used constantly in this topic):** $\sigma_t^2$ is not a single fixed number for the whole series (that would be the ordinary, UNCONDITIONAL variance, like Phase 6's $\sigma^2/(1-\phi^2)$) — it's a genuinely different, freshly RECALCULATED number at EVERY time step, CONDITIONAL on (i.e., "given," "based on") what just happened. **This is a fundamentally different kind of object than anything in Phase 6 — there, $\sigma^2$ was always one single constant; here, $\sigma_t^2$ is itself a whole TIME SERIES, changing at every step.**

---

## 4. Deriving that ARCH really does capture volatility clustering (a mini-proof)

**Claim: under ARCH(1), the squared returns themselves behave like an AR(1) process (a genuinely satisfying, derivable connection back to Phase 6).**

Start from $r_t^2 = \sigma_t^2 z_t^2$ (squaring the return equation). Substitute the variance equation:
$$
r_t^2 = (\omega+\alpha_1 r_{t-1}^2)\,z_t^2
$$
Now here's the clever algebraic trick: add and subtract $(\omega+\alpha_1 r_{t-1}^2)$ appropriately to isolate an AR(1)-flavored structure. Define $\eta_t = \sigma_t^2(z_t^2-1)$ (this turns out to be a legitimate, zero-mean noise term, since $E[z_t^2]=\text{Var}(z_t)=1$ by definition of $z_t$ being standardized, so $E[z_t^2-1]=0$). Then:
$$
r_t^2 = \sigma_t^2 z_t^2 = \sigma_t^2 + \sigma_t^2(z_t^2-1) = (\omega+\alpha_1 r_{t-1}^2) + \eta_t = \omega + \alpha_1 r_{t-1}^2 + \eta_t
$$
**This is EXACTLY the AR(1) formula from Phase 6, Part 1, but written in terms of SQUARED returns instead of the raw series!** ($r_t^2$ plays the role of $x_t$, $\alpha_1$ plays the role of $\phi$, $\eta_t$ plays the role of $\varepsilon_t$.) **This is the actual mathematical proof that ARCH(1) produces autocorrelated squared returns (volatility clustering) — a direct structural fact, not just a hand-wavy claim** — and it means everything you already know about AR(1) (stationarity condition, ACF decay shape from Phase 6, Part 1) transfers directly: **ARCH(1) is stationary (in this squared-returns sense) exactly when $\alpha_1 < 1$** (a direct parallel to AR(1)'s $|\phi|<1$ condition, section 5 of Part 1), and if stationary, the ACF of the SQUARED returns decays geometrically, exactly mirroring Phase 6, Part 1, section 4's derivation, just applied to $r_t^2$ instead of $x_t$.

---

## 5. GARCH(1,1): adding "memory of past variance itself," not just past shocks

**The limitation of plain ARCH, motivating GARCH:** ARCH(1) only looks ONE step back, and only at the raw squared shock. In practice, real financial volatility often has LONGER memory than a simple one-lag squared-shock term can capture well — you'd need a large ARCH order (many lags, ARCH(q) for large $q$) to adequately capture realistic volatility persistence, which means estimating a lot of separate $\alpha$ parameters. **GARCH's genuinely clever fix, directly parallel to how Holt-Winters extended SES back in Phase 5: instead of needing MANY lags of squared shocks, add ONE extra term that lets YESTERDAY'S VARIANCE ESTIMATE ITSELF feed into today's — creating the same kind of "infinite effective memory from a small number of parameters" efficiency that Holt-Winters/exponential smoothing already showed you.**

$$
\sigma_t^2 = \omega + \alpha_1\, r_{t-1}^2 + \beta_1\, \sigma_{t-1}^2
$$

**The one new piece: $\beta_1 \sigma_{t-1}^2$** — "today's variance also depends on what YESTERDAY'S variance ESTIMATE itself was" (not the raw squared shock this time — the model's own PREVIOUS OUTPUT feeding back into itself). **This recursive "my own previous estimate feeds into my current estimate" structure should feel very familiar — it's structurally identical to SES's recursion from Phase 5, section 4** ($\hat{x}_{t+1}=\alpha x_t+(1-\alpha)\hat{x}_t$ — SES blends new data with its OWN prior estimate, exactly the same recursive flavor GARCH uses here, just applied to variance instead of level).

**Unrolling GARCH(1,1)'s recursion (exactly the same technique as Phase 5, section 4's SES unrolling, and Phase 6, Part 1, section 2's AR(1) unrolling) reveals GARCH(1,1) is EQUIVALENT to an ARCH($\infty$) model** — an infinite-order ARCH with GEOMETRICALLY DECAYING weights on all past squared shocks, achieved using just 3 parameters ($\omega,\alpha_1,\beta_1$) instead of needing infinitely (or even just very many) many separate $\alpha$ coefficients. **This is exactly the same parameter-efficiency story you've now seen repeated multiple times across this course** (AR representing an infinite MA with few parameters in Phase 6 Part 1; Fourier terms representing many seasonal indices with few coefficients in Phase 8) — **a recurring, genuinely important pattern: a small "feedback" term (using the model's own past output, not just past raw data) can compactly represent what would otherwise require many more parameters of "flat" memory.**

---

## 6. The stationarity/persistence condition for GARCH(1,1)

**Directly analogous to AR(1)'s $|\phi|<1$ (Phase 6, Part 1, section 2) and ARCH(1)'s $\alpha_1<1$ (section 4 above):**
$$
\alpha_1 + \beta_1 < 1
$$
**Plain English: the COMBINED persistence from both the shock term and the memory term must be less than 1, or volatility will explode/never settle down to a stable long-run level (an unstable, non-stationary variance process — directly parallel to Phase 4's unit-root instability, just happening in the VARIANCE equation instead of the mean-level equation).**

**When $\alpha_1+\beta_1$ is close to 1 (but still just under it), volatility shocks are extremely PERSISTENT** — a big shock today keeps elevating expected volatility for a very LONG time before fading back to baseline (real financial data very often shows $\alpha_1+\beta_1$ estimated quite close to 1, e.g., 0.95-0.99 — a genuinely common, real empirical finding worth knowing for interviews, sometimes called observing "near-unit-root behavior in volatility").

**The long-run/unconditional variance** (the level $\sigma_t^2$ would settle to on average, if no new shocks ever arrived — parallel to Phase 6, Part 1, section 3's derivation of AR(1)'s mean): take expectations of the GARCH equation, using $E[\sigma_t^2]=E[\sigma_{t-1}^2]=\bar\sigma^2$ (assuming stationarity) and $E[r_{t-1}^2]=\bar\sigma^2$ too (since the unconditional variance of returns equals this same long-run value):
$$
\bar\sigma^2 = \omega + \alpha_1\bar\sigma^2+\beta_1\bar\sigma^2 \quad\Rightarrow\quad \bar\sigma^2(1-\alpha_1-\beta_1)=\omega \quad\Rightarrow\quad \bar\sigma^2 = \frac{\omega}{1-\alpha_1-\beta_1}
$$
**Notice the exact same algebraic PATTERN as Phase 6, Part 1, section 3's derivation of the AR(1) mean** ($\mu = c/(1-\phi)$) — same technique, same structural "divide by one minus the persistence" result, just now for variance instead of level. And exactly as before, this formula breaks down (divides by zero) precisely at the stationarity BOUNDARY ($\alpha_1+\beta_1=1$) — one more confirming angle on section 6's condition.

---

## 7. Estimating ARCH/GARCH parameters: directly reusing Phase 6, Part 4's MLE machinery

**We don't need new estimation theory here — this is a genuinely pleasant payoff of having built MLE properly in Phase 6, Part 4.** The likelihood-construction LOGIC is identical: assume $z_t \sim N(0,1)$ (a standard, if simplifying, assumption — real applications sometimes use fatter-tailed distributions here, directly addressing stylized fact 2 from section 2), which means $r_t \mid \text{past info} \sim N(0,\sigma_t^2)$ (a Normal distribution, but now with a VARYING variance parameter $\sigma_t^2$ that changes at every time step, rather than Phase 6's single FIXED $\sigma^2$). Plug into the SAME Gaussian probability density formula from Phase 6, Part 4, section 4, multiply across all time points, take the log, and MAXIMIZE over $(\omega,\alpha_1,\beta_1)$ using the same numerical-optimization logic described there. **The ONLY conceptual difference: the variance term inside the likelihood formula is now $\sigma_t^2$ (time-varying, itself computed recursively from the GARCH equation) instead of a single fixed $\sigma^2$** — everything else about HOW maximum likelihood estimation works is identical to what you already learned.

---

## 8. Numerical worked example: run GARCH(1,1) forward by hand

Let $\omega=0.1$, $\alpha_1=0.2$, $\beta_1=0.7$ (check: $\alpha_1+\beta_1=0.9<1$ ✓, stationary/persistent-but-stable).

**Long-run variance (section 6):** $\bar\sigma^2 = \dfrac{0.1}{1-0.9}=\dfrac{0.1}{0.1}=1.0$

**Suppose at $t=1$: $\sigma_1^2 = 1.0$ (starting exactly at the long-run level) and the realized return was $r_1 = 2.5$ (a notably large move relative to a baseline variance of 1.0 — i.e., a genuine "surprise" day).**

**Compute $\sigma_2^2$ (tomorrow's predicted variance, given today's data):**
$$
\sigma_2^2 = \omega+\alpha_1 r_1^2+\beta_1\sigma_1^2 = 0.1+0.2(2.5)^2+0.7(1.0) = 0.1+0.2(6.25)+0.7 = 0.1+1.25+0.7=2.05
$$
**Interpretation: because yesterday had an unusually large move ($r_1=2.5$, squared to 6.25 — well above the long-run variance of 1.0), today's predicted variance JUMPS from the long-run baseline of 1.0 up to 2.05 — MORE THAN DOUBLING.** This is volatility clustering, mechanically produced by the formula, in real numbers.

**Suppose $t=2$ then has a calm day: $r_2 = 0.3$.**

**Compute $\sigma_3^2$:**
$$
\sigma_3^2 = 0.1+0.2(0.3)^2+0.7(2.05) = 0.1+0.2(0.09)+1.435=0.1+0.018+1.435=1.553
$$
**Interpretation: even though $t=2$ itself was calm, predicted variance for $t=3$ is STILL elevated (1.553), well above the long-run 1.0 baseline — because the $\beta_1\sigma_{t-1}^2$ term carries forward a good chunk of yesterday's ALREADY-elevated variance ESTIMATE, regardless of how calm today's actual realized return was.** **This is precisely the "memory that decays gradually rather than resetting instantly" behavior — a large shock's effect on expected future volatility fades out gradually over several periods (governed by how close $\alpha_1+\beta_1$ is to 1, section 6), rather than disappearing the instant a single calm day occurs.** If you continued this forward with more calm days, $\sigma_t^2$ would keep gradually decaying back down toward the 1.0 long-run baseline — exactly the geometric-decay-toward-the-mean behavior you should now recognize from Phase 6, Part 1's AR(1) derivation, just happening in the variance sequence instead of the level sequence.

---

## 9. Brief mention: extensions addressing the leverage effect (stylized fact 3)

**GJR-GARCH and EGARCH** (briefly, by name and core idea, not full derivation — genuinely useful to recognize if asked, without needing to derive from scratch): both add an extra term that lets NEGATIVE shocks (price drops) increase predicted variance MORE than equally-sized POSITIVE shocks (price gains) — directly targeting the leverage effect from section 2 that plain GARCH cannot capture (plain GARCH treats $r_{t-1}^2$ identically regardless of the SIGN of $r_{t-1}$, since squaring destroys sign information entirely — these extensions add a mechanism to let the sign matter again, on top of the magnitude). **If asked in an interview "what does plain GARCH miss?", the leverage effect and these two named extensions are the correct, complete answer.**

---

## 10. Quick self-check questions

1. In plain English, what specific real-world pattern in financial data does ARCH/GARCH exist to capture, that plain ARIMA cannot?
   *(Answer: volatility clustering — periods of large price swings tend to cluster together in time, and periods of calm tend to cluster together too; ARIMA models the mean/level of a series and assumes constant variance, so it cannot represent this time-varying pattern in the SIZE of shocks.)*
2. Why does GARCH(1,1) need far fewer parameters than a high-order ARCH(q) to capture similarly realistic volatility persistence, and what earlier-phase concept is this efficiency directly analogous to?
   *(Answer: GARCH's β₁σ²_{t-1} term lets the model's own previous variance estimate feed back into the current one, creating an effectively infinite, geometrically-decaying memory from just one extra parameter — unrolling the recursion shows GARCH(1,1) is equivalent to an infinite-order ARCH; this is directly analogous to how AR(1) compactly represents an infinite MA (Phase 6, Part 1) and how Fourier terms compactly represent many seasonal indices (Phase 8) — a small feedback/recursive term standing in for many "flat" parameters.)*
3. What does it mean, concretely, if a fitted GARCH(1,1) has α₁+β₁ very close to 1 (like 0.98)?
   *(Answer: volatility shocks are extremely persistent — a large shock today will keep elevating expected future volatility for a long time before gradually decaying back to the long-run baseline, rather than fading out quickly; this is commonly observed in real financial data.)*
4. What real pattern does plain GARCH fail to capture, and what is it called?
   *(Answer: the leverage effect — the tendency for volatility to rise more after a large negative shock (price drop) than after an equally large positive shock (price gain); plain GARCH treats both identically since it only uses the squared (sign-destroying) shock. Extensions like GJR-GARCH and EGARCH add this asymmetry.)*

---

## What's next
Phase 11 moves into **Multivariate Time Series** — Vector Autoregression (VAR), Granger causality (and why "Granger causality" is a famously misleading name that doesn't mean real causality — a near-guaranteed interview question), impulse response functions, and the FULL formal treatment of cointegration and the Vector Error Correction Model (VECM) that we previewed back in Phase 7, section 7.

Say "next" for Phase 11, or ask for more ARCH/GARCH hand-computation drills first (e.g., continuing the numerical example a few more steps to watch the variance decay back toward the long-run baseline).
