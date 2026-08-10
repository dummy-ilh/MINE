# Chapter 11 — Autocorrelation

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations, extra real-world examples, and step-by-step equation walkthroughs. Uses Chapter 5's residuals (in their original observation order) to work the Durbin-Watson statistic by hand, with an honest caveat about what this dataset can and can't actually demonstrate.*

---

## 11.1 The Motivating Question

Chapters 9–10 handled two ways the LINE assumptions can break: correlated *predictors* (multicollinearity) and unequal error *variance* (heteroscedasticity). This chapter handles a third: **errors correlated with each other** — most commonly, one observation's error being related to the *previous* observation's error, which is why this problem is almost always discussed in the context of **time-ordered (time-series) data**.

**What breaks, precisely (same pattern as Chapter 10):** OLS point estimates $\hat{\boldsymbol{\beta}}$ remain **unbiased** even under autocorrelated errors. What's lost is BLUE-ness (Chapter 6) and the validity of standard errors — and in practice, autocorrelation tends to make OLS standard errors look **artificially small**, making effects appear more statistically significant than they really are. This is arguably the most dangerous of the three violations covered in Chapters 9–11, precisely because it silently inflates your confidence.

### What autocorrelation actually *is*, before any formulas at all

Autocorrelation just means: **the mistake your model makes today is related to the mistake it made yesterday (or last period, or the previous observation).** Instead of every error being a fresh, independent surprise, errors "clump" — a run of positive errors tends to be followed by more positive errors, or a run of negative errors by more negative errors (or, less commonly, errors alternate back and forth predictably).

**Four everyday examples, to build real intuition before touching a single equation:**

1. **Monthly company sales.** If sales were unexpectedly high last month (say, due to a viral social media post), that boost often hasn't fully faded by this month — some of last month's momentum carries over. Your forecasting model's error this month is partly a leftover echo of last month's error, not a brand-new independent surprise.

2. **Daily temperature forecasts.** If a weather model underestimates today's temperature (it's hotter than predicted), tomorrow's error is likely to lean the same direction too — heat waves last for days, not single isolated moments. The errors "stick together" in runs.

3. **Stock price momentum (short-term).** If a stock unexpectedly jumped today, there's often a small tendency for it to keep drifting in the same direction for a little while afterward (before eventually correcting) — again, today's surprise isn't fully independent of tomorrow's.

4. **Repeated sensor readings on a machine.** If a factory sensor reads a bit high due to a temperature drift in the room, that same drift is still present a few seconds later — consecutive readings' errors are correlated because the *underlying cause* of the error (the room being warmer than usual) hasn't gone away.

**The common thread in all four:** whatever *causes* today's error (momentum, weather systems, drift, hype) doesn't vanish instantly — it lingers and partly causes tomorrow's error too. That lingering is autocorrelation.

**Plain-language framing of why this is dangerous (not just annoying):** picture a weather forecaster who doesn't realize that today's forecasting error is basically a rerun of yesterday's — not fresh, independent evidence. If they keep treating each day's error as brand-new information, they'll end up thinking they have *far more independent evidence* of their own accuracy than they really do. That's exactly what happens to your regression: it starts reporting error bars and p-values that are far too confident, because it's silently double (or triple, or tenfold) counting what is really the same underlying pattern of error, repeated.

**An honest framing note for this chapter:** our running dataset is 5 students measured once each — there's no genuine time ordering, so a Durbin-Watson test here is really a demonstration of the *mechanics*, not a meaningful diagnostic (there's no reason student 3's error should relate to student 2's error just because they happen to be listed adjacently). We work the numbers anyway so the formula is concrete, but the right context for this test is genuine time-series or sequentially-collected data — e.g., quarterly sales data, repeated sensor readings — where adjacent-observation correlation is a real physical possibility.

---

## 11.2 The Durbin-Watson Test

**Test statistic:**

$$ DW = \frac{\sum_{i=2}^{n}(e_i-e_{i-1})^2}{\sum_{i=1}^{n}e_i^2} $$

### Breaking this formula into two simple pieces

**Piece 1 — the numerator, $\sum(e_i-e_{i-1})^2$:** this just walks down your list of residuals in order and, for each one, asks "how different is this residual from the one right before it?" It squares each of those differences (so a big gap in either direction counts as "big," and negative gaps don't cancel out positive ones) and adds them all up. In plain words: **"total amount of jumping around between neighbors."**

**Piece 2 — the denominator, $\sum e_i^2$:** this is just the total squared size of all your residuals put together — a measure of "how much error is there overall," regardless of any ordering. You've already seen this exact quantity before — it's simply $SSE$, the same sum of squared errors from every earlier chapter.

**Putting the two pieces together:** $DW$ is asking, "out of all the total error in this model, how much of it shows up as neighbor-to-neighbor jumping around, versus staying similar from one point to the next?" A **small** numerator relative to the denominator means neighbors are staying suspiciously *similar* to each other (errors "sticking together" — positive autocorrelation). A **large** numerator relative to the denominator means neighbors are unusually *different* from each other (errors "bouncing" back and forth — negative autocorrelation).

**Building the intuition further, with the two extreme cases:**

- **Extreme case 1 — errors "stick together" (strong positive autocorrelation).** Imagine residuals like $1.0, 1.1, 0.9, 1.0, 1.05$ — every neighbor is nearly identical to the last. Each difference $(e_i-e_{i-1})$ is tiny, so the numerator shrinks toward zero, and $DW$ collapses toward **0**.
- **Extreme case 2 — errors "bounce" back and forth (strong negative autocorrelation).** Imagine residuals like $1.0, -1.0, 1.0, -1.0, 1.0$ — every neighbor swings to the opposite sign. Each difference is now unusually *large* (subtracting a negative from a positive doubles the gap), pushing the numerator way up, and $DW$ rises toward **4**.
- **The healthy middle ground — no relationship at all.** If each residual is a fresh, independent draw with no memory of the one before it, the differences are neither unusually small nor unusually large on average — landing right around $DW=2$.

**Worked example**, using Chapter 5's residuals in order: $e = 0.2,\ 0.6,\ -1,\ -0.6,\ 0.8$.

**Step 1 — compute each neighbor-to-neighbor difference:**

$$ e_2-e_1 = 0.6-0.2 = 0.4 $$
$$ e_3-e_2 = -1-0.6 = -1.6 $$
$$ e_4-e_3 = -0.6-(-1) = 0.4 $$
$$ e_5-e_4 = 0.8-(-0.6) = 1.4 $$

**Step 2 — square each difference and add them up:**

$$ 0.4^2+(-1.6)^2+0.4^2+1.4^2 = 0.16+2.56+0.16+1.96 = 4.84 $$

**Step 3 — divide by the total squared error (already known from Chapter 5: $SSE=2.4$):**

$$ DW = \frac{4.84}{2.4} \approx 2.02 $$

**Reading it:** $DW\approx2.02$ is almost exactly 2 — no evidence of autocorrelation. As flagged above, this is exactly what you'd expect from **cross-sectional** data where the observation order is arbitrary (alphabetical by student, say) rather than reflecting any real time sequence — there's no mechanism by which this result should show correlation, and it doesn't.

**In plain words, what this confirms:** these 5 students weren't measured in any meaningful sequence — they're just listed in whatever order someone typed them in. There's no reason "student 2's mistake" should have anything to do with "student 3's mistake," and the math agrees: $DW$ landed almost exactly at the "no relationship" value of 2. This is a sanity-check result, not a discovery — it's confirming what we already knew structurally about the data (no real time order = no reason to expect autocorrelation).

**A concrete contrast — what this same dataset would look like WITH real autocorrelation.** Suppose instead these were 5 *consecutive months* of a company's sales-forecast errors, and the pattern had been $e = 0.2,\ 0.5,\ 0.9,\ 1.3,\ 1.6$ — steadily climbing, each one bigger than the last (the model's forecasts keep falling further behind an accelerating trend it hasn't caught onto). The differences would all be small and consistently *positive* ($0.3,\ 0.4,\ 0.4,\ 0.3$), giving a small numerator relative to the total error, and $DW$ would land well below 2 — a real, meaningful signal that the model is missing a trend it should be accounting for.

**Formal hypothesis testing caveat:** Durbin-Watson critical values ($d_L, d_U$) come from specialized tables indexed by $n$ and the number of predictors $k$, and those tables conventionally start around $n=15$ — this chapter's $n=5$ is too small for a formally valid critical-value lookup. The computation above is a legitimate illustration of the *statistic's mechanics*, not a valid hypothesis test at this sample size.

---

## 11.3 Connecting DW to the Autocorrelation Coefficient $\rho$

Under a simple **AR(1)** error structure ($\varepsilon_t = \rho\varepsilon_{t-1}+u_t$, where $u_t$ is genuinely uncorrelated noise), there's a direct approximate relationship:

$$ DW \approx 2(1-\hat{\rho}) \qquad\Rightarrow\qquad \hat{\rho} \approx 1-\frac{DW}{2} $$

### Unpacking "AR(1)" piece by piece, before the formula

$\varepsilon_t = \rho\varepsilon_{t-1}+u_t$ looks intimidating but says something simple: **"today's error equals some fraction of yesterday's error, plus something brand new."**

- $\varepsilon_t$ — today's error.
- $\varepsilon_{t-1}$ — yesterday's error.
- $\rho$ (rho) — a single number between -1 and 1 that controls *how much* of yesterday's error echoes into today. This is the "volume knob" on the echo.
- $u_t$ — a genuinely fresh, unpredictable surprise, unrelated to anything before it.

**Reading the three key values of $\rho$ in plain words:**
- $\rho$ **near 1**: today's error is almost a carbon copy of yesterday's — a very loud, persistent echo (like the sales-momentum example above).
- $\rho$ **near 0**: today's error has nothing to do with yesterday's — no echo at all, which is the "healthy," assumption-satisfying case.
- $\rho$ **near -1**: today's error tends to be the *opposite* of yesterday's — an overcorrection pattern (like a thermostat that overshoots hot, then overshoots cold, back and forth).

**Now the DW-to-$\rho$ formula makes intuitive sense:** since $DW$ collapses toward 0 under strong positive echo and rises toward 4 under strong negative echo (as shown in §11.2), and $\rho$ ranges from +1 (strong positive echo) to -1 (strong negative echo), the two are simply mirror images of each other on related number lines — $DW\approx2(1-\rho)$ is just the formula that translates between them.

**Worked check:**

$$ \hat{\rho} \approx 1 - \frac{2.02}{2} = 1-1.01 = -0.01 $$

Essentially zero, consistent with the "no autocorrelation" reading above. $\hat{\rho}=1$ would indicate perfect positive autocorrelation ($DW\to0$); $\hat{\rho}=-1$ would indicate perfect negative autocorrelation ($DW\to4$); $\hat{\rho}=0$ (our case) gives $DW=2$ exactly.

---

## 11.4 Generalized Least Squares (GLS) — The General Remedy

Chapter 10's WLS handled unequal variances by weighting each observation individually. **GLS** generalizes this fully: instead of just a diagonal weight matrix, it allows the **entire error covariance structure** $\boldsymbol{\Sigma} = \text{Var}(\boldsymbol{\varepsilon})$ to be non-diagonal — capturing not just unequal variances but also *correlations between* different observations' errors.

$$ \hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{y} $$

**Plain-English framing, before the matrix:** WLS (Chapter 10) only had to worry about each point having its *own* level of noise — it never had to worry about points *influencing each other's* noise. GLS drops that restriction entirely: it can handle a situation where point 3's error is partly explained by point 2's error, which is exactly the autocorrelation problem this chapter is about. Practically, GLS is "do the weighting trick from Chapter 10, but let the weights account for relationships *between* points, not just each point's own individual noise level."

**WLS is a special case of GLS** where $\boldsymbol{\Sigma}$ happens to be diagonal (unequal variances, but zero correlation between observations) — this is worth stating explicitly in an interview, since it ties Chapters 10 and 11 together as two instances of the same general fix, just with different assumed structures for $\boldsymbol{\Sigma}$.

**For AR(1) errors specifically**, $\boldsymbol{\Sigma}$ takes a specific banded (Toeplitz) form:

$$ \boldsymbol{\Sigma} = \frac{\sigma_u^2}{1-\rho^2}\begin{bmatrix}1&\rho&\rho^2&\cdots\\\rho&1&\rho&\cdots\\\rho^2&\rho&1&\cdots\\\vdots&&&\ddots\end{bmatrix} $$

### Reading this matrix one entry at a time

Each entry in this grid answers one question: **"how correlated is the error at this row's time point with the error at this column's time point?"**

- **The diagonal (all 1's, scaled):** every observation's error is, of course, perfectly correlated with itself — that's just a mathematical formality, not a discovery.
- **One step off the diagonal ($\rho$):** this is the direct, one-period echo — how strongly *today's* error relates to *yesterday's*.
- **Two steps off the diagonal ($\rho^2$):** this is a weaker echo — the relationship between today and *two days ago* has to "pass through" yesterday, and it fades each time it does (since $\rho^2$ is always smaller than $\rho$, for $-1<\rho<1$).
- **Further off the diagonal, still smaller ($\rho^3, \rho^4, ...$):** the correlation keeps shrinking the further apart in time two points are.

**In plain words, the real-world story this matrix tells:** yesterday strongly predicts today's error; last week only weakly predicts today's; last month barely predicts it at all. That's a very natural, common-sense pattern — recent events matter more than distant ones — and this matrix is simply the precise mathematical version of that intuition.

---

## 11.5 The Cochrane-Orcutt / Prais-Winsten Transformation

Directly inverting $\boldsymbol{\Sigma}$ by hand is impractical even for moderate $n$. The standard practical shortcut transforms the data so that plain OLS on the *transformed* variables is equivalent to GLS on the original ones:

$$ y_t^* = y_t - \hat{\rho}\,y_{t-1}, \qquad x_t^* = x_t - \hat{\rho}\,x_{t-1} \qquad (\text{for } t=2,...,n) $$

Then run **ordinary** OLS of $y_t^*$ on $x_t^*$. The intuition: subtracting off $\hat{\rho}$ times the previous observation "removes" the correlated component, leaving behind approximately independent noise — the same trick, structurally, as differencing in ARIMA time-series modeling. (Prais-Winsten additionally recovers the first observation, which Cochrane-Orcutt simply discards, via a specific transformation for $t=1$ — a minor technical refinement.)

**Plain-English version of the trick, one step at a time:**

1. First, estimate how strong the echo is ($\hat\rho$), using the DW-to-$\rho$ relationship from §11.3 (or a more precise regression-based estimate).
2. For every time point (after the first), subtract off "$\hat\rho$ times last period's value" from both $y$ and $x$. This is literally asking: "how much of this observation is just an echo of last time, and how much is genuinely new?" — and keeping only the genuinely new part.
3. Run an entirely ordinary OLS regression on these adjusted (echo-removed) variables. Because the echo has already been subtracted out, the leftover noise is close to independent again — meaning plain OLS is valid on this transformed version, even though it wasn't valid on the original data.

**Applied to this chapter's data:** since $\hat{\rho}\approx-0.01$ (essentially zero), this transformation would barely change $x_t$ or $y_t$ at all — correctly reflecting that GLS collapses back to ordinary OLS when $\rho=0$ (just as WLS collapses back to OLS when all weights are equal). This is a useful sanity check to state out loud in an interview: **GLS and WLS are both strict generalizations of OLS that reduce exactly to OLS when their respective assumed structures ($\rho=0$, or equal weights) hold.**

**In plain words, why this makes sense:** if there's no real echo between observations ($\rho\approx0$), then "subtracting off the echo" barely changes anything — you subtract almost nothing, because there was almost nothing to subtract. That's a good consistency check: a remedy for a problem you don't actually have should do almost nothing when applied, and that's exactly what happens here.

---

## 11.6 An Alternative Remedy: Newey-West (HAC) Standard Errors

Just as Chapter 10 offered "robust sandwich" standard errors as an alternative to WLS when the variance structure is unknown, **Newey-West standard errors** (also called **HAC** — Heteroscedasticity and Autocorrelation Consistent) serve the same role here: they keep the ordinary OLS point estimates unchanged, but correct the standard errors to remain valid under **both** heteroscedasticity and autocorrelation up to some specified lag, without requiring you to specify the exact AR structure. This is the most common practical choice in applied time-series work when the priority is trustworthy inference rather than fully modeling the error-generating process.

**Plain-English summary:** this is the autocorrelation-flavored cousin of Chapter 10's "sandwich" standard errors. You keep your regular OLS coefficients exactly as they were, and just correct the *error bars* around them to properly account for both "some points being noisier than others" and "nearby points' errors being related to each other" — all without having to first prove or assume exactly how strong that relationship is (no need to estimate $\rho$ precisely, unlike Cochrane-Orcutt).

**A simple decision guide for choosing between §11.5 and §11.6:** if you're confident the errors genuinely follow a clean AR(1)-style echo and you want the *most efficient* possible estimates, model it explicitly with Cochrane-Orcutt/GLS. If you just want honest, trustworthy error bars without betting on a specific structure being exactly right, Newey-West is the safer, more common default in practice.

---

## 11.7 Where the Textbooks Differ

- **Kutner** treats autocorrelation relatively briefly as one instance of the broader "correlated errors" problem, focusing on the Durbin-Watson mechanics and the Cochrane-Orcutt transformation.
- **Montgomery** (reflecting its engineering/quality-control/process-monitoring roots) gives autocorrelation substantial attention, since sequential process measurements are a natural setting for it — control charts and autocorrelation diagnostics are closely related themes there.
- **Sheather** emphasizes Newey-West/HAC standard errors as the modern practical default, mirroring its Chapter 10 emphasis on robust standard errors over WLS.
- **ESL/ISL** essentially don't cover this topic — autocorrelation is a classical time-series-adjacent concern outside the independent-observations framework that most of ESL/ISL assumes; time-series-specific ML methods are treated as a separate subject entirely.

---

## 11.8 Interview Q&A

**Q: What's the difference between heteroscedasticity and autocorrelation, structurally?**
A: Heteroscedasticity means $\text{Var}(\varepsilon_i)$ differs across observations, but errors remain uncorrelated with each other — $\boldsymbol{\Sigma}$ is diagonal but not constant on the diagonal. Autocorrelation means errors are correlated with each other (commonly, with their own past values) — $\boldsymbol{\Sigma}$ has nonzero off-diagonal entries.
*(Simple version: heteroscedasticity = "some points are noisier than others, but independently so"; autocorrelation = "one point's noise leaks into the next point's noise.")*

**Q: What does a Durbin-Watson statistic near 2 indicate? Near 0? Near 4?**
A: Near 2: no autocorrelation. Near 0: strong positive autocorrelation (consecutive errors tend to be similar). Near 4: strong negative autocorrelation (consecutive errors tend to alternate in sign).
*(Simple version: 2 = errors are strangers to each other; 0 = errors copy their neighbor; 4 = errors flip-flop from their neighbor.)*

**Q: Give a real-world example of positive autocorrelation.**
A: Monthly sales forecast errors — if a viral marketing moment boosts sales above forecast one month, some of that boost typically persists into the next month too, so consecutive forecast errors tend to lean the same direction rather than being fresh, independent surprises each time.

**Q: Is WLS a special case of GLS, or the other way around?**
A: WLS is a special case of GLS — GLS allows the full error covariance matrix $\boldsymbol{\Sigma}$ to have any structure, including off-diagonal correlations; WLS restricts $\boldsymbol{\Sigma}$ to be diagonal (unequal variances only, no cross-observation correlation).
*(Simple version: GLS is the general toolbox; WLS is one specific tool inside it, for when points differ in noise level but don't talk to each other.)*

**Q: Why is autocorrelation often considered more dangerous in practice than heteroscedasticity?**
A: It tends to make OLS standard errors appear artificially small, inflating apparent statistical significance — this is especially common (and easy to miss) in time-series regressions with trending or seasonal data, where naive OLS can produce highly significant-looking but spurious relationships.
*(Simple version: it makes your model quietly overconfident — it looks more trustworthy than it actually is, which is a much sneakier failure than just being wrong.)*

**Q: When would you use Newey-West standard errors instead of explicitly modeling the AR structure with GLS?**
A: When you want valid inference without committing to a specific correlation structure (exact AR order, specific $\rho$) — Newey-West corrects standard errors for autocorrelation up to a chosen lag length without requiring you to fully specify or estimate the error-generating process, at the cost of not improving the efficiency of the point estimates the way a correctly-specified GLS would.
*(Simple version: use it when you want honest error bars without having to correctly guess the exact "shape" of how the noise is connected across time.)*

---

*End of Chapter 11. Next: Chapter 12 — Transformations (Box-Cox, log transforms, and polynomial terms as remedies for the linearity violations first flagged back in Chapter 7's residuals-vs-fitted panel).*
