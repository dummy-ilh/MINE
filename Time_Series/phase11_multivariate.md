# Phase 11: Multivariate Time Series — VAR, Granger Causality, Cointegration, VECM

Everything so far has modeled ONE series at a time (occasionally using OTHER variables as fixed external regressors, Phase 7). This phase handles the case where SEVERAL time series genuinely influence EACH OTHER, back and forth, simultaneously — e.g., interest rates and inflation, or ad spend and revenue, where each plausibly affects the other over time. This also completes the cointegration story we deliberately left unfinished back in Phase 7, section 7.

---

## 1. Symbol glossary for this file

| Symbol | Plain-English meaning |
|---|---|
| $\mathbf{y}_t$ (bold) | a VECTOR (a list of numbers) containing ALL the series' values at time $t$ together, e.g., $\mathbf{y}_t = \begin{pmatrix}y_{1,t}\\y_{2,t}\end{pmatrix}$ for two series |
| $y_{1,t}, y_{2,t}$ | the individual series' values (e.g., series 1 = interest rate, series 2 = inflation) |
| $\mathbf{A}$ | a MATRIX (a grid of numbers) of coefficients — the multivariate generalization of a single $\phi$ |
| $\mathbf{c}$ | a vector of constants (one constant per series) |
| $\boldsymbol{\varepsilon}_t$ | a vector of noise terms, one fresh shock per series, at time $t$ |
| $\Rightarrow$ (in Granger causality context) | "Granger-causes" — a specific, narrow, TESTABLE notion of one series improving the prediction of another (explained fully in section 3 — deliberately NOT the same as real-world "causes") |
| IRF | Impulse Response Function — traces out what happens to ALL the series over future time, after ONE series gets hit with a single shock today |
| $z_t$ | the cointegrating residual/error-correction term (introduced in section 6) |

**A quick, essential refresher on vectors and matrices (only what you need for this file):** a **vector** is just a list of numbers stacked together — here, we stack the CURRENT VALUES of all our different series into one vector, $\mathbf{y}_t$. A **matrix** is a grid/table of numbers — here, a matrix of coefficients lets us express "how EACH series affects EACH OTHER series" all at once, compactly. Multiplying a matrix by a vector is a specific, mechanical operation (multiply-and-add across each row) — you don't need to be a linear algebra expert; software does this instantly. The KEY plain-English point for everything below: **a matrix equation like $\mathbf{A}\mathbf{y}_{t-1}$ is just a compact way of writing SEVERAL ordinary regression equations at once — one row of the matrix per series being predicted.**

---

## 2. Vector Autoregression (VAR): building it from AR(1), for two series

**Plain English motivation, before any formula:** recall AR(1) from Phase 6, Part 1: $x_t = \phi\, x_{t-1}+\varepsilon_t$ — one series, predicted from its OWN past. **VAR's entire idea: let EVERY series be predicted from EVERYONE'S past values, not just its own.** For two series, spelled out as two ordinary, individually-readable equations (no matrix notation needed yet — build the intuition first):

$$
y_{1,t} = c_1 + a_{11}\, y_{1,t-1} + a_{12}\, y_{2,t-1} + \varepsilon_{1,t}
$$
$$
y_{2,t} = c_2 + a_{21}\, y_{1,t-1} + a_{22}\, y_{2,t-1} + \varepsilon_{2,t}
$$

**Reading the FIRST equation in plain English:** "today's value of series 1 depends on: a constant, PLUS its OWN yesterday's value (exactly like ordinary AR(1)), PLUS ALSO yesterday's value of series 2, PLUS fresh noise." **The genuinely new piece, compared to everything in Phase 6, is that $a_{12}$ term** — series 1 is now allowed to depend on series 2's past, not just its own. The second equation is the exact mirror image — series 2 depends on both its own past AND series 1's past.

**The subscript convention on the $a$ coefficients (worth being very explicit about, since this trips people up):** $a_{12}$ means "the effect of series **2**'s past on series **1**'s present" — read the subscripts as (equation number, variable number) — the FIRST number tells you WHICH equation/series is being predicted, the SECOND tells you WHICH past variable is doing the predicting. So $a_{12} \neq a_{21}$ in general — the effect of series 2 on series 1 need NOT equal the effect of series 1 on series 2 (a genuinely important asymmetry, central to the Granger causality discussion in section 3).

**Writing this compactly using vectors/matrices (purely notational — no new content, just packaging the same two equations above into one line):**
$$
\mathbf{y}_t = \mathbf{c} + \mathbf{A}\,\mathbf{y}_{t-1} + \boldsymbol{\varepsilon}_t, \qquad \mathbf{A} = \begin{pmatrix}a_{11}&a_{12}\\a_{21}&a_{22}\end{pmatrix}
$$
**This single line, once you unpack the matrix multiplication, is EXACTLY the same two equations written out above** — genuinely nothing new mathematically, purely a compact notation so we can write "VAR(1)" instead of writing out a separate equation for every single series every time. "VAR(1)" means each series depends on lag-1 values of everyone; "VAR(p)" extends this to $p$ lags of everyone, exactly the same way AR(p) extended AR(1) back in Phase 6, Part 1, section 5.

**Stationarity condition for VAR (directly generalizing Phase 6, Part 1, section 5.1):** the EIGENVALUES of the matrix $\mathbf{A}$ (a specific, computable set of numbers associated with any square matrix — a genuinely standard linear algebra concept; software computes these instantly) must all have magnitude LESS THAN 1. **This is EXACTLY the same "roots outside the unit circle" idea from AR(p), just phrased using matrix eigenvalues instead of polynomial roots** — for a SINGLE series (a 1×1 "matrix," which is really just an ordinary number), the eigenvalue IS just $\phi$ itself, and this condition reduces to exactly $|\phi|<1$, our original AR(1) condition. **Same underlying concept, generalized to handle multiple interacting series simultaneously.**

---

## 3. Granger Causality — a famously misleading name, explained precisely

**The single most important thing to memorize about this topic, verbatim, because it is asked constantly: "Granger causality" does NOT mean real, true causation. It is a specific, narrow, purely statistical/predictive concept.**

**The precise definition, built from the VAR equations above:** we say **series 2 Granger-causes series 1** if including series 2's PAST values in the equation for series 1 (the $a_{12}$ term above) genuinely improves our ability to PREDICT series 1, beyond what series 1's own past alone could achieve.

**The formal test, directly reusing Phase 4's hypothesis-testing logic:**
- **Null hypothesis $H_0$: $a_{12} = 0$** (series 2's past adds NO predictive value for series 1, once you already know series 1's own past).
- **Alternative: $a_{12} \neq 0$** (series 2's past DOES meaningfully improve the prediction).
- This is tested with an ordinary F-test (a standard statistical test for "does adding this predictor meaningfully improve a regression," which you may have encountered in general regression coursework) comparing the FULL model (with $a_{12}$ included) against the RESTRICTED model (forcing $a_{12}=0$).

**Why this is NOT real causation — the actual, precise reasoning (a genuinely important thing to be able to articulate clearly in an interview, not just assert):**

1. **Confounding.** Both series could be driven by some THIRD, unmeasured factor. Example: ice cream sales "Granger-cause" drowning incidents (ice cream sales predict/precede drowning spikes) — but the REAL common cause is HOT WEATHER, driving both independently. Series 2 (ice cream) genuinely helps PREDICT series 1 (drownings) in a purely statistical sense, without any real causal link between them at all.

2. **It's fundamentally about PREDICTION TIMING, not MECHANISM.** Granger causality only asks "does knowing series 2's past help predict series 1's future, in a statistical, correlational sense" — it says NOTHING about WHY, or through what actual physical/economic mechanism, that predictive relationship exists. Real causation requires an actual causal mechanism; Granger causality requires only a predictive/temporal statistical pattern.

3. **Reverse Granger causality can ALSO hold simultaneously**, which real causation (in the everyday sense) usually shouldn't allow bidirectionally with equal footing. It's entirely possible for series 2 to Granger-cause series 1 AND series 1 to Granger-cause series 2 at the same time (both $a_{12}\neq0$ and $a_{21}\neq0$) — this is called **bidirectional** or **feedback** Granger causality, and it's genuinely common in real economic data (e.g., interest rates and inflation plausibly both Granger-cause each other).

**The complete, correct, interview-ready statement to memorize: "Granger causality is a test of whether one series' past values improve the STATISTICAL PREDICTION of another series, beyond that series' own past — it is a purely predictive, correlational concept, and Clive Granger himself explicitly warned it should not be interpreted as true causation, precisely because of confounding variables and the absence of any requirement for an actual causal mechanism."**

---

## 4. Impulse Response Functions (IRF): tracing a shock's ripple effects through the whole system

**Plain English motivation:** once you've fit a VAR, a genuinely useful practical question is: "if series 1 gets hit with a single, one-time surprise shock TODAY, how does that ripple forward and affect BOTH series 1 AND series 2 over the following days/weeks/months?" **This is EXACTLY the same "unroll the recursion forward" technique you've now used repeatedly (Phase 2's random walk, Phase 6 Part 1's AR(1), Phase 6 Part 5's forecasting) — just applied to a MULTI-series system, tracking how a shock in ONE equation propagates into ALL the equations over time, via their mutual feedback loops.**

**Mechanically, computing an IRF (conceptual walkthrough, no need for a full symbolic derivation):** set every series to zero, then apply exactly ONE unit shock to ONE series' $\varepsilon$ term at time 0 (and zero shocks everywhere else, forever after). Then use the fitted VAR equations (section 2) to compute $\mathbf{y}_1, \mathbf{y}_2, \mathbf{y}_3,\ldots$ forward, step by step, mechanically — exactly the same recursive "plug the previous step's OUTPUT back in as the next step's INPUT" process from Phase 6, Part 5, section 4.2, just now tracking a VECTOR of series simultaneously instead of one number. **The resulting sequence of values for EACH series, over increasing time steps, IS the impulse response function for that series** — literally a plot showing "here's how a single shock to series 1 ripples through and affects series 2 (and series 1 itself) at 1 step later, 2 steps later, 3 steps later, and so on."

**Why this is genuinely useful, practically:** IRFs let you see BOTH the DIRECTION (does a shock to series 1 push series 2 UP or DOWN?) and the DYNAMICS/TIMING (does the effect show up immediately, or with a delay? does it fade away quickly or persist for a long time?) of how the system responds — information a single Granger-causality YES/NO test (section 3) can't convey on its own, since Granger causality only tells you WHETHER a relationship exists, not its SIZE, DIRECTION, or TIME PATTERN.

**Forecast Error Variance Decomposition (briefly, by name): a closely related tool that asks a slightly different question: "of the total forecast UNCERTAINTY (Phase 6, Part 5, section 5's forecast error variance, now computed for a multivariate system) about series 1 at some future horizon, what FRACTION is attributable to series 1's OWN shocks, versus shocks originating from series 2?"** — genuinely useful for understanding which series is the more influential "driver" in a mutually-interacting system.

---

## 5. Cointegration, formally (completing the promise from Phase 7, section 7)

Recall Phase 7's setup: TWO series can each individually be non-stationary (unit root, Phase 4), yet some specific LINEAR COMBINATION of them is stationary — a genuine, real long-run relationship, not a spurious one.

**The formal definition:** $y_{1,t}$ and $y_{2,t}$ are **cointegrated** if (a) each one is individually non-stationary (specifically, each becomes stationary after differencing ONCE — called "integrated of order 1," written **I(1)**, a formal name for exactly the "difference-stationary" idea from Phase 4, section 4.2), AND (b) there exists SOME combination $z_t = y_{1,t} - \beta\, y_{2,t}$ (for some specific constant $\beta$) that IS stationary on its own (I(0), "integrated of order zero," meaning no differencing needed at all — already stationary).

**Plain English: even though both series individually wander around with no fixed anchor (Phase 4), the SPECIFIC GAP between them ($y_{1,t}-\beta y_{2,t}$) does NOT wander — it stays anchored, mean-reverting, bounded.** Revisiting Phase 7's example: revenue and costs might both individually trend upward forever, but (revenue $-\beta\times$costs), for the right $\beta$ (representing, e.g., typical profit margin), stays in a bounded, stable range.

### 5.1 The Engle-Granger two-step method — a genuinely simple, practical test

**Step 1:** run an ordinary OLS regression (Phase 7's basic tool) of one series on the other: $y_{1,t} = \beta\, y_{2,t} + z_t$, and save the residuals $\hat{z}_t$.

**Step 2:** run an ADF test (Phase 4, section 6.2 — the EXACT SAME unit-root test you already fully derived) directly on these residuals $\hat{z}_t$. **If the ADF test REJECTS a unit root in the residuals (i.e., the residuals ARE stationary) → the two series ARE cointegrated (genuine long-run relationship, section 5's definition is satisfied). If the ADF test FAILS to reject (residuals still look non-stationary) → NOT cointegrated (any apparent relationship is likely spurious, exactly Phase 7 section 6's warning).**

**Plain English, tying together Phases 4, 6, and 7 into one clean workflow: Engle-Granger is genuinely nothing more than "run a regression (Phase 7), then run an ADF test (Phase 4) on what's left over" — every individual piece is something you've already fully derived; cointegration testing is simply a clever, specific SEQUENCE of applying two tools you already have.** (One small technical caveat worth knowing: because $\hat z_t$ comes from an ESTIMATED regression rather than being directly observed data, the ADF test's critical values need a small adjustment here — called the Engle-Granger critical values, slightly different from the plain ADF table — software handles this automatically; you just need to know the adjustment exists and why (the residuals are themselves an ESTIMATED quantity, adding a small extra layer of estimation uncertainty beyond plain ADF's original setup)).

### 5.2 The Johansen test — briefly, when Engle-Granger isn't enough
**The Engle-Granger method has a real limitation: it only cleanly handles TWO series at once, and it requires you to arbitrarily pick WHICH series is "the regressor" in step 1** (regressing 1-on-2 vs. 2-on-1 can, awkwardly, give somewhat different results in finite samples). **The Johansen test is a more general, matrix-based method (using the VAR framework's eigenvalues, section 2) that can test for cointegration among ANY NUMBER of series simultaneously, and can even detect MULTIPLE independent cointegrating relationships at once among a larger group of series** — genuinely useful when you have, say, 4-5 related economic indicators and want to find ALL the stable long-run relationships among them, not just check one specific pair. We won't derive its full matrix-eigenvalue machinery here (it requires linear algebra beyond this course's current depth) — the practical, interview-level takeaway is simply: **know it exists, know it generalizes Engle-Granger to more than 2 series, and know it's built on VAR's eigenvalue framework from section 2.**

---

## 6. The Vector Error Correction Model (VECM): USING a real cointegrating relationship

**The motivating question, connecting directly back to Phase 4's overdifferencing warning (section 6.3) and Phase 7's spurious-regression discussion (section 6):** if two series ARE genuinely cointegrated, you should NOT just difference both and fit a plain VAR on the differenced data (as Phase 7 section 6 suggested for the SPURIOUS case) — doing so would DISCARD the real, valuable long-run relationship entirely, an overdifferencing-flavored loss of genuine signal. **VECM is specifically built to use BOTH the short-run dynamics (via differencing, like ordinary VAR) AND the long-run cointegrating relationship (via the stationary combination $z_t$ from section 5) simultaneously, rather than forcing a choice between them.**

**The formula, for two series (building directly on the cointegrating residual $z_{t-1} = y_{1,t-1}-\beta y_{2,t-1}$ from section 5):**
$$
\Delta y_{1,t} = c_1 + \lambda_1\, z_{t-1} + (\text{short-run lagged } \Delta y \text{ terms}) + \varepsilon_{1,t}
$$
$$
\Delta y_{2,t} = c_2 + \lambda_2\, z_{t-1} + (\text{short-run lagged } \Delta y \text{ terms}) + \varepsilon_{2,t}
$$

**Reading this in plain English, piece by piece:**
- $\Delta y_{1,t}$ (the CHANGE in series 1, Phase 4's differencing notation) is what we're predicting now, not the raw level — this is the ordinary VAR-on-differenced-data part (the "short-run dynamics" piece).
- $z_{t-1}$ = the cointegrating gap from ONE PERIOD AGO — "how far out of long-run equilibrium were things, most recently?"
- $\lambda_1$ (lambda) = the **error-correction speed** — how STRONGLY series 1 gets pulled back toward restoring the long-run equilibrium relationship, whenever that gap $z_{t-1}$ isn't zero.

**The genuinely elegant, complete plain-English story: "the SHORT-RUN change in each series depends on ordinary short-run dynamics (like plain VAR), PLUS an extra 'error correction' pull — the further the system has recently drifted from its long-run equilibrium gap $z_{t-1}$, the MORE STRONGLY the series gets pulled back toward restoring that equilibrium."** This is precisely why it's called an ERROR CORRECTION model: **it actively corrects deviations from the long-run relationship, using genuine, non-spurious information that would have been thrown away entirely by naive differencing (Phase 7's blanket "always difference to avoid spurious regression" advice) — VECM is the correct, complete answer specifically for the case where two non-stationary series are genuinely, not spuriously, related.**

**Sign intuition for $\lambda_1$:** typically, $\lambda_1$ comes out NEGATIVE (a genuinely important detail) — plain English: "if the gap $z_{t-1}$ was POSITIVE (series 1 currently ABOVE its long-run equilibrium relationship with series 2), series 1's next CHANGE should tend to be NEGATIVE (pulling it back DOWN toward equilibrium)" — a negative coefficient on a positive deviation produces exactly this stabilizing, corrective pull-back force, directly analogous in SPIRIT to the mean-reversion behavior of a stationary AR(1) with $0<\phi<1$ (Phase 6, Part 1) — deviations get pulled back toward a stable anchor, rather than persisting or exploding.

---

## 7. Numerical mini-illustration: Engle-Granger step 1 by hand (kept simple)

Two tiny series, 5 points each (imagine $y_1$=a company's revenue, $y_2$=its costs, both in $1000s, both clearly trending together):
$y_1 = [10, 14, 17, 22, 25]$
$y_2 = [8, 11, 13, 17, 19]$

**Step 1 — a very rough, simplified regression slope estimate** (ordinary least squares, Phase 7, using the simple formula $\hat\beta = \dfrac{\sum(y_{2,t}-\bar y_2)(y_{1,t}-\bar y_1)}{\sum(y_{2,t}-\bar y_2)^2}$, i.e., regressing $y_1$ on $y_2$):

$\bar y_1 = (10+14+17+22+25)/5 = 88/5=17.6$
$\bar y_2 = (8+11+13+17+19)/5=68/5=13.6$

Deviations $y_1-\bar y_1$: $[-7.6,-3.6,-0.6,4.4,7.4]$
Deviations $y_2-\bar y_2$: $[-5.6,-2.6,-0.6,3.4,5.4]$

Numerator (sum of products): $(-7.6)(-5.6)+(-3.6)(-2.6)+(-0.6)(-0.6)+(4.4)(3.4)+(7.4)(5.4)$
$=42.56+9.36+0.36+14.96+39.96=107.2$

Denominator (sum of squared $y_2$ deviations): $(-5.6)^2+(-2.6)^2+(-0.6)^2+(3.4)^2+(5.4)^2 = 31.36+6.76+0.36+11.56+29.16=79.2$

$$
\hat\beta = \frac{107.2}{79.2}\approx 1.3535
$$

**Step 2 — compute the residuals $\hat z_t = y_{1,t}-\hat\beta\, y_{2,t}$** (using $\hat\beta\approx1.3535$; note we'd also normally estimate an intercept, omitted here purely to keep this hand-calculation simple and focused on the core mechanic):

$\hat z_1 = 10-1.3535(8)=10-10.828=-0.828$
$\hat z_2 = 14-1.3535(11)=14-14.8885=-0.8885$
$\hat z_3 = 17-1.3535(13)=17-17.5955=-0.5955$
$\hat z_4 = 22-1.3535(17)=22-23.0095=-1.0095$
$\hat z_5=25-1.3535(19)=25-25.7165=-0.7165$

**Interpretation: notice the residuals $[-0.828,-0.8885,-0.5955,-1.0095,-0.7165]$ stay in a tight, bounded range (roughly $-0.6$ to $-1.0$), with NO visible trend, even though BOTH original series ($y_1$ and $y_2$) clearly climbed substantially over the same 5 periods (10→25 and 8→19 respectively).** This is EXACTLY the visual signature of a genuinely cointegrated pair: **individually trending series, but a stable, bounded gap between them (after appropriate scaling by $\hat\beta$).** A formal Engle-Granger step 2 would run an ADF test (Phase 4) on this small residual series to CONFIRM stationarity statistically (this toy dataset is far too small for a real, properly-powered test — but the VISUAL pattern here is exactly what you'd hope to see before running that formal confirmation).

---

## 8. Quick self-check questions

1. In the VAR coefficient $a_{12}$, which subscript tells you WHICH equation/series is being predicted, and which tells you WHICH variable's past is doing the predicting?
   *(Answer: the FIRST subscript (1) identifies which series/equation is being predicted; the SECOND subscript (2) identifies which series' past value is the predictor — so a₁₂ specifically means "the effect of series 2's past on series 1's present.")*
2. Give the ice-cream/drowning example (or a similar one) explaining precisely WHY Granger causality is not the same as real causation.
   *(Answer: ice cream sales can "Granger-cause" drownings in the sense that past ice cream sales statistically improve the prediction of drowning incidents — but this is because both are independently driven by a third, unmeasured confounding factor (hot weather), not because ice cream sales actually cause drownings; Granger causality only tests statistical predictive improvement, with no requirement for an actual causal mechanism.)*
3. Why can't you just difference both series and fit a plain VAR on the differenced data, if the two series are genuinely cointegrated?
   *(Answer: differencing both series would discard the real, valuable long-run equilibrium relationship between their levels entirely — an overdifferencing-flavored loss of genuine signal; VECM instead preserves this information by including the cointegrating residual z_{t-1} directly as a predictor alongside the short-run differenced dynamics.)*
4. In a VECM, what does it typically mean if the error-correction coefficient λ₁ is negative and the prior period's cointegrating gap z_{t-1} was positive?
   *(Answer: it means series 1 was recently above its long-run equilibrium relationship with series 2, and the negative λ₁ produces a corrective pull DOWNWARD in series 1's next change — a stabilizing, mean-reverting force pulling the system back toward its long-run equilibrium.)*

---

## What's next
Phase 12 covers **Spectral/Frequency Domain Analysis** — a different lens entirely on everything we've built so far: instead of asking "how does today relate to yesterday" (the time-domain view this whole course has used up to now), we ask "what hidden, repeating CYCLES/frequencies is this series built from," directly formalizing the Fourier-wave intuition from Phase 8 into a full analytical tool (the periodogram), and connecting it back to the ACF via a genuinely elegant result (the Wiener-Khinchin theorem) showing these two seemingly different perspectives (time domain vs. frequency domain) are secretly two views of the exact same information.

Say "next" for Phase 12, or ask for more VAR/Granger/cointegration drilling first.
