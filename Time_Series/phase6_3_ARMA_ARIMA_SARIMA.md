# Phase 6, Part 3 of 5: ARMA, ARIMA, SARIMA

Roadmap: 6.1 AR(p) [done] → 6.2 MA(q) [done] → **6.3 ARMA/ARIMA/SARIMA (this file)** → 6.4 Estimation & model selection → 6.5 Diagnostics & forecasting.

You asked for the math made simpler and every symbol explained — so in this file, **every single symbol gets a one-line plain-English tag the first time it shows up**, even ones repeated from earlier phases. Think of each formula as a sentence built from labeled Lego pieces — I'll name each piece before showing the full sentence.

---

## 1. First, a quick symbol glossary (all symbols used in this file, in one place)

| Symbol | Plain-English meaning |
|---|---|
| $x_t$ | the value of the series at time $t$ (what we observe) |
| $t$ | the current time step (like "today") |
| $t-1, t-2,\dots$ | past time steps (yesterday, two days ago, ...) |
| $c$ | a constant number (just shifts the level up/down) |
| $\phi_i$ (phi) | an AR coefficient — how strongly a past VALUE of $x$ pulls on today's value |
| $\theta_i$ (theta) | an MA coefficient — how strongly a past NOISE SHOCK still echoes into today |
| $\varepsilon_t$ (epsilon) | today's random white-noise shock (unpredictable, mean 0) |
| $p$ | the AR order — how many past $x$ values are used |
| $q$ | the MA order — how many past shocks are used |
| $d$ | the number of times we difference the series to make it stationary |
| $s$ | the seasonal period (e.g. 12 for monthly data, 4 for quarterly) |
| $P, D, Q$ | the SEASONAL versions of $p, d, q$ (same meaning, applied at the seasonal lag) |
| $L$ | the lag operator — a symbol meaning "step back one time unit" ($Lx_t = x_{t-1}$) |
| $\nabla$ or $\Delta$ | the difference operator — means "subtract the previous value" ($\nabla x_t = x_t - x_{t-1}$) |
| $\Phi(L)$, $\Theta(L)$ | shorthand names for the whole AR or MA formula written using $L$ |

Keep this table nearby — every formula below only uses pieces from this table.

---

## 2. Combining AR and MA: the ARMA(p,q) model

**Plain English motivation first:** In Part 1, AR said "today depends on past VALUES." In Part 2, MA said "today depends on past SHOCKS." Why not use both kinds of memory at once? That's literally all ARMA is — an AR piece plus an MA piece, glued together with a plus sign.

**The formula:**
$$
x_t = c + \underbrace{\phi_1 x_{t-1} + \dots + \phi_p x_{t-p}}_{\text{the AR part (Part 1)}} + \underbrace{\varepsilon_t + \theta_1\varepsilon_{t-1}+\dots+\theta_q\varepsilon_{t-q}}_{\text{the MA part (Part 2)}}
$$

**Reading it in one sentence:** "Today's value = a constant, plus a weighted mix of the last $p$ actual values, plus a weighted mix of the last $q$ noise shocks (including today's brand new one)."

**"ARMA(p,q)"** is just a short label meaning "an AR piece of order $p$, glued to an MA piece of order $q$." Nothing more mysterious than that name.

**Do the earlier rules still apply, unchanged?**
- **Stationarity** (Part 1, section 5.1): governed ENTIRELY by the AR piece — the $\phi$'s. The MA piece never causes non-stationarity (Part 2, section 4 already told us finite MA is always stationary). So: check the AR roots exactly the same way as before; ignore the MA part for this specific check.
- **Invertibility** (Part 2, section 4): governed ENTIRELY by the MA piece — the $\theta$'s. Check exactly the same way as before, ignoring the AR part.
**Plain English: the two conditions don't interact — you check each one using only its own half of the formula, completely separately.**

### 2.1 ACF/PACF shape for ARMA — why BOTH tail off
Recall the Phase 3 table said ARMA's ACF AND PACF both tail off (neither cuts off cleanly). **Why?** The AR piece contributes the "infinite decaying tail" behavior to the ACF (proven in Part 1, section 4). The MA piece contributes an infinite decaying tail to the PACF (proven in Part 2, section 3, via the AR(∞) representation). Since ARMA has BOTH pieces at once, **both plots inherit a decaying-tail shape** — neither one gets a clean cutoff point anymore, because there's no pure "AR-only" or "MA-only" structure left to produce one. This is a direct, simple consequence of combining the two derivations you already worked through — no new math, just adding two known effects together.

---

## 3. Why do we even need ARMA if we have separate AR and MA? (Wold Decomposition Theorem)

**The single-sentence version of a famous theorem, explained in plain words:** *Herman Wold proved that (almost) ANY stationary process, no matter how complicated, can be written as an infinite MA process (an infinite weighted sum of past shocks, like the MA(∞) form we met back in Part 1, section 2) plus a perfectly predictable deterministic piece.*

**Why does this matter practically?** It means MA (and by extension ARMA, since AR itself is just a special "compressed" way of writing an infinite MA — remember Part 1's derivation showing AR(1) unrolls into MA(∞)) is not just "one modeling option among many" — it's a THEORETICALLY JUSTIFIED, general-purpose building block that (in principle) can approximate almost any real, well-behaved stationary series. **ARMA models aren't an arbitrary guess — they're backed by a proof that this family of models is rich enough to be a reasonable universal approximation, provided you allow enough parameters ($p$ and $q$ large enough).** In practice, we use FINITE, small $p$ and $q$ instead of an infinite one purely for simplicity and estimability — an ARMA(2,1), say, is a compact/efficient approximation to what might "truly" require an infinite MA, using far fewer numbers to estimate. This is genuinely one of the most commonly asked "why does this method even work" theory questions in top-tier interviews, and now you have the plain-English answer.

---

## 4. From ARMA to ARIMA: adding back differencing

**The problem ARMA alone can't handle:** ARMA REQUIRES stationarity (section 2 above). But Phase 4 taught you most REAL data (with trend, or a random-walk-like wandering pattern) is NOT stationary to begin with. **ARIMA's entire idea: difference the data first (Phase 4's fix) to make it stationary, THEN fit an ARMA model to what's left.**

**The name, unpacked letter by letter:**
- **AR** — the AR part (Part 1), coefficients $\phi_1,\ldots,\phi_p$.
- **I** — stands for **Integrated**. This is genuinely just a fancy statistics word for "differenced." (Slightly confusing naming: "integrating" usually means "summing up" in math, and indeed — if differencing is like taking a derivative (measuring change), then going the OTHER way, from the differenced series back to the original, is like "integrating"/summing those changes back up. The model's NAME refers to this reverse relationship: the model works on the DIFFERENCED data, but to get a real forecast for the ORIGINAL series, you need to "integrate"/sum the differenced forecasts back up. Don't worry about deriving this — just know why the letter "I" is used.)
- **MA** — the MA part (Part 2), coefficients $\theta_1,\ldots,\theta_q$.

**"ARIMA(p,d,q)"**: three numbers in order.
- $p$ = AR order (how many past VALUES of the DIFFERENCED series are used)
- $d$ = how many times we difference the ORIGINAL series before fitting (Phase 4, section 5 — usually 0, 1, or 2)
- $q$ = MA order (how many past SHOCKS are used)

**The formula, built in two clear steps (much simpler than writing it all in one line):**

**Step 1 — difference the raw data $d$ times** (exactly the $\nabla$ operator from Phase 4, section 5):
$$
y_t = \nabla^d x_t
$$
(Plain English: $y_t$ is just a new, shorter series — the result of subtracting each point from the one before it, done $d$ times in a row, as you already practiced in Phase 4.)

**Step 2 — fit an ordinary ARMA(p,q) model to $y_t$** (using EXACTLY the formula from section 2 above, just relabeling $x_t$ as $y_t$ everywhere):
$$
y_t = c + \phi_1 y_{t-1}+\dots+\phi_p y_{t-p} + \varepsilon_t + \theta_1\varepsilon_{t-1}+\dots+\theta_q\varepsilon_{t-q}
$$

**That's genuinely the whole idea — ARIMA is not a new mathematical object, it's "difference first (Phase 4), then apply ARMA (section 2 of this file)."** Everything you already know from both of those places transfers directly; nothing new to derive.

**Special cases you should instantly recognize by name (good interview reflex to build):**
- ARIMA(p,0,0) = plain AR(p) (no differencing needed/applied)
- ARIMA(0,0,q) = plain MA(q)
- ARIMA(0,1,0) = a random walk! (differencing once, then fitting NOTHING — $y_t=\varepsilon_t$, i.e., the differenced series is just pure white noise — exactly matching Phase 2's random walk, $x_t - x_{t-1}=\varepsilon_t$)
- ARIMA(0,1,1) = SES's underlying model (fun fact: Simple Exponential Smoothing from Phase 5 turns out to be mathematically equivalent to this specific ARIMA model — a genuinely nice fact connecting Phase 5 and Phase 6, and a real interview question: "what ARIMA model is equivalent to SES?")

---

## 5. From ARIMA to SARIMA: adding a seasonal ARIMA on top

**The problem ARIMA alone can't handle:** ordinary differencing ($\nabla$) removes a TREND, but it doesn't specifically target a repeating SEASONAL pattern (Phase 1's seasonality, Phase 4's seasonal differencing $\nabla_s$, Phase 5's Holt-Winters seasonal term). **SARIMA's idea: apply a SECOND, separate ARIMA-style structure, but built using SEASONAL lags (steps of size $s$) instead of ordinary lags (steps of size 1).**

**The label: SARIMA$(p,d,q)(P,D,Q)_s$** — notice it's really just TWO ARIMA labels stuck together:
- $(p,d,q)$ = the ORDINARY (non-seasonal) part — EXACTLY the same meaning as plain ARIMA above, working at lag 1, 2, 3, ...
- $(P,D,Q)_s$ = the SEASONAL part — the SAME three ideas (AR order, differencing count, MA order), but now working at lag $s$, $2s$, $3s$, ... instead of lag $1,2,3,\ldots$. Capital letters are used purely to visually distinguish "seasonal versions" from "ordinary versions" of the same three concepts — not a different concept, just a label convention.
- $s$ = the seasonal period (same $s$ as Phase 5's Holt-Winters $m$ — different courses/authors sometimes use $m$ and sometimes $s$ for this identical idea, another naming inconsistency you just have to expect).

**Concretely, what does "seasonal AR order $P=1$" even mean?** It means: today's (differenced) value depends on the value from EXACTLY ONE FULL SEASONAL CYCLE AGO (lag $s$), the same way ordinary AR(1) depended on lag 1 — just measured in "cycles back" instead of "steps back." A seasonal MA term of order $Q=1$ similarly means: today echoes the SHOCK from exactly one seasonal cycle ago.

**Full model construction in words, step by step (this is simpler than one giant formula):**
1. Apply ordinary differencing $d$ times (removes trend) — same as plain ARIMA, section 4.
2. Apply SEASONAL differencing $D$ times, using $\nabla_s$ from Phase 4, section 5 (removes the repeating seasonal pattern) — i.e., subtract the value from exactly one cycle ago, $D$ times in a row.
3. Now fit BOTH an ordinary ARMA structure (using lags $1,2,\ldots,p$ and $1,\ldots,q$) AND a seasonal ARMA structure (using lags $s, 2s,\ldots, Ps$ and $s,\ldots,Qs$) to whatever is left, MULTIPLIED together (technically, using the lag-operator notation from the glossary: $\Phi(L)\Phi_s(L^s)\, \nabla^d\nabla_s^D x_t = \Theta(L)\Theta_s(L^s)\,\varepsilon_t$ — you don't need to expand this by hand; just recognize it's "ordinary AR piece times seasonal AR piece equals ordinary MA piece times seasonal MA piece," each piece built exactly the way you already know).

**Concrete, fully worked-out example of what a real SARIMA label MEANS (no algebra, pure translation exercise):** Take SARIMA$(1,1,1)(1,1,1)_{12}$ — a genuinely very common real-world specification for monthly business data.
- $(1,1,1)$: use 1 ordinary AR lag, difference once (ordinary), use 1 ordinary MA lag.
- $(1,1,1)_{12}$: use 1 seasonal AR lag (12 months back), seasonally-difference once (compare to 12 months ago), use 1 seasonal MA lag (12 months back).
- **In plain English:** "today's sales depend a bit on last month's sales AND last month's surprise, AND ALSO depend a bit on the sales from exactly one year ago AND that same month's surprise from a year ago — after removing both an overall trend and a yearly repeating pattern via differencing." This single sentence is the FULL translation of that six-number label — a genuinely useful skill: being able to read any SARIMA label out loud in plain English is a strong, practical interview signal.

---

## 6. A short numerical illustration (kept deliberately simple, per your request)

We won't re-derive full ACF formulas here (that would just repeat Parts 1 and 2's work with messier algebra) — instead, a SIMPLE arithmetic illustration of the "difference, then model" idea for ARIMA(0,1,1), tying directly to the SES connection from section 4.

Raw data (has a rising trend): $x = [50, 55, 63, 68, 77]$

**Step 1 — difference once ($d=1$):**
$y_2 = 55-50=5$
$y_3 = 63-55=8$
$y_4 = 68-63=5$
$y_5 = 77-68=9$

Differenced series: $[5, 8, 5, 9]$ — notice this hovers around a fairly steady level (average $\approx 6.75$) with no obvious remaining trend — a good visual sign that $d=1$ was probably enough (recall Phase 4's warning against over-differencing — if THIS series still looked trending, you might need $d=2$; since it looks roughly flat/stable, $d=1$ looks sufficient).

**Step 2 — this is now just an MA(1) fitting problem** (ARIMA(0,1,1) means $p=0$: no AR part at all on the differenced series, only an MA(1) part) — exactly the Part 2 machinery, just applied to $y_t$ instead of directly to $x_t$. We won't refit numerically here since you already practiced the MA(1) mechanics by hand in Part 2, section 6 — the point of this example is purely to make concrete what "difference first, then apply the ARMA machinery you already know" looks like on real numbers, not to introduce new estimation math.

---

## 7. Quick self-check questions

1. In ARMA(p,q), which coefficients ($\phi$'s or $\theta$'s) determine whether the model is stationary, and which determine invertibility?
   *(Answer: stationarity depends only on the $\phi$ coefficients (the AR part); invertibility depends only on the $\theta$ coefficients (the MA part) — the two checks are done completely separately.)*
2. What does the letter "I" in ARIMA actually stand for, and why is that name a little confusing at first?
   *(Answer: "Integrated" — meaning differenced. It's confusing because "integrating" usually means summing, but the model itself works on the DIFFERENCED series; the name refers to the fact that you have to sum/"integrate" the differenced forecasts back up to get a forecast for the original series.)*
3. Translate SARIMA(0,1,0)(0,1,0)$_4$ into one plain-English sentence (no formulas).
   *(Answer: "Difference the series once to remove an ordinary trend, and difference it once more at a 4-period seasonal lag to remove a repeating quarterly pattern — with no AR or MA terms of any kind (ordinary or seasonal) fitted to what remains, so the leftover series is assumed to just be white noise.")*
4. Which specific ARIMA(p,d,q) model is mathematically the same as a plain random walk, and which is the same as Simple Exponential Smoothing?
   *(Answer: ARIMA(0,1,0) = random walk; ARIMA(0,1,1) = SES.)*

---

## What's next
**Part 4 of Phase 6** covers HOW we actually ESTIMATE all these $\phi$ and $\theta$ coefficients from real data (Yule-Walker equations for AR, and Maximum Likelihood Estimation for the general case), and how we choose $p$, $d$, $q$ in the first place using AIC/BIC — turning everything from Parts 1–3 from "here's the model shape" into "here's how you actually fit one to real numbers."

Say "next" for Part 4, or ask me to slow down further on anything in this file first (happy to re-explain any single symbol or formula piece again, differently).
