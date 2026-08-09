# Chapter 11 — Autocorrelation

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations. Uses Chapter 5's residuals (in their original observation order) to work the Durbin-Watson statistic by hand, with an honest caveat about what this dataset can and can't actually demonstrate.*

---

## 11.1 The Motivating Question

Chapters 9–10 handled two ways the LINE assumptions can break: correlated *predictors* (multicollinearity) and unequal error *variance* (heteroscedasticity). This chapter handles a third: **errors correlated with each other** — most commonly, one observation's error being related to the *previous* observation's error, which is why this problem is almost always discussed in the context of **time-ordered (time-series) data**.

**What breaks, precisely (same pattern as Chapter 10):** OLS point estimates $\hat{\boldsymbol{\beta}}$ remain **unbiased** even under autocorrelated errors. What's lost is BLUE-ness (Chapter 6) and the validity of standard errors — and in practice, autocorrelation tends to make OLS standard errors look **artificially small**, making effects appear more statistically significant than they really are. This is arguably the most dangerous of the three violations covered in Chapters 9–11, precisely because it silently inflates your confidence.

**Plain-language framing before anything else:** picture tracking a company's monthly sales. If sales were unexpectedly high last month, they're probably *still* somewhat elevated this month too — good months tend to cluster together, and so do bad ones. That "clustering" is autocorrelation: your model's mistakes aren't independent surprises, they're echoes of the previous mistake. The dangerous part isn't that your best-guess predictions go wrong (they don't) — it's that your model becomes *overconfident*, reporting tighter error bars and smaller p-values than it's actually entitled to. It's like a weather forecaster who thinks they're more accurate than they really are because they didn't realize today's forecast error was basically a repeat of yesterday's, not fresh independent evidence.

**An honest framing note for this chapter:** our running dataset is 5 students measured once each — there's no genuine time ordering, so a Durbin-Watson test here is really a demonstration of the *mechanics*, not a meaningful diagnostic (there's no reason student 3's error should relate to student 2's error just because they happen to be listed adjacently). We work the numbers anyway so the formula is concrete, but the right context for this test is genuine time-series or sequentially-collected data — e.g., quarterly sales data, repeated sensor readings — where adjacent-observation correlation is a real physical possibility.

---

## 11.2 The Durbin-Watson Test

**Test statistic:**

$$ DW = \frac{\sum_{i=2}^{n}(e_i-e_{i-1})^2}{\sum_{i=1}^{n}e_i^2} $$

**Plain-English reading:** the numerator measures how much consecutive residuals *differ* from each other; the denominator is just total residual variation (SSE). If consecutive residuals tend to be similar (positive autocorrelation — a positive residual tends to be followed by another positive one), the numerator shrinks relative to the denominator, and $DW$ drops well below 2. If consecutive residuals tend to be dissimilar (negative autocorrelation), $DW$ rises above 2. **$DW\approx2$ indicates no autocorrelation.**

**Building the intuition one step further:** think about what happens in the two extreme cases. If every residual is nearly *identical* to the one before it (strong positive autocorrelation — errors "sticking together"), then each difference $(e_i - e_{i-1})$ is close to zero, so the numerator shrinks toward zero and $DW$ collapses toward 0. If instead residuals *alternate* back and forth — positive, then negative, then positive again (negative autocorrelation — errors "bouncing"), the differences become unusually large (you're subtracting a negative from a positive, doubling the gap), pushing $DW$ up toward 4. Right in the middle, at $DW=2$, the differences are neither unusually small nor unusually large — exactly what you'd expect if each residual has nothing to do with the one before it.

**Worked example**, using Chapter 5's residuals in order: $e = 0.2,\ 0.6,\ -1,\ -0.6,\ 0.8$.

$$ \text{Differences: } (0.6-0.2),\ (-1-0.6),\ (-0.6-(-1)),\ (0.8-(-0.6)) = 0.4,\ -1.6,\ 0.4,\ 1.4 $$

$$ \sum(\text{differences})^2 = 0.16+2.56+0.16+1.96 = 4.84 $$

$$ DW = \frac{4.84}{2.4} \approx 2.02 $$

**Reading it:** $DW\approx2.02$ is almost exactly 2 — no evidence of autocorrelation. As flagged above, this is exactly what you'd expect from **cross-sectional** data where the observation order is arbitrary (alphabetical by student, say) rather than reflecting any real time sequence — there's no mechanism by which this result should show correlation, and it doesn't.

**In plain words, what this confirms:** these 5 students weren't measured in any meaningful sequence — they're just listed in whatever order someone typed them in. There's no reason "student 2's mistake" should have anything to do with "student 3's mistake," and the math agrees: $DW$ landed almost exactly at the "no relationship" value of 2. This is a sanity-check result, not a discovery — it's confirming what we already knew structurally about the data (no real time order = no reason to expect autocorrelation).

**Formal hypothesis testing caveat:** Durbin-Watson critical values ($d_L, d_U$) come from specialized tables indexed by $n$ and the number of predictors $k$, and those tables conventionally start around $n=15$ — this chapter's $n=5$ is too small for a formally valid critical-value lookup. The computation above is a legitimate illustration of the *statistic's mechanics*, not a valid hypothesis test at this sample size.

---

## 11.3 Connecting DW to the Autocorrelation Coefficient $\rho$

Under a simple **AR(1)** error structure ($\varepsilon_t = \rho\varepsilon_{t-1}+u_t$, where $u_t$ is genuinely uncorrelated noise), there's a direct approximate relationship:

$$ DW \approx 2(1-\hat{\rho}) \qquad\Rightarrow\qquad \hat{\rho} \approx 1-\frac{DW}{2} $$

**Plain-English translation of the AR(1) idea, before the formula:** "AR(1)" just means "each error is partly a leftover echo of the previous error, plus something genuinely new." $\rho$ (rho) is a number between -1 and 1 that tells you how strong that echo is: $\rho$ near 1 means "today's mistake is almost a carbon copy of yesterday's," $\rho$ near 0 means "today's mistake has nothing to do with yesterday's," and $\rho$ near -1 means "today's mistake tends to be the *opposite* of yesterday's."

**Worked check:** $\hat{\rho} \approx 1 - 2.02/2 = 1-1.01 = -0.01$ — essentially zero, consistent with the "no autocorrelation" reading above. $\hat{\rho}=1$ would indicate perfect positive autocorrelation ($DW\to0$); $\hat{\rho}=-1$ would indicate perfect negative autocorrelation ($DW\to4$); $\hat{\rho}=0$ (our case) gives $DW=2$ exactly.

---

## 11.4 Generalized Least Squares (GLS) — The General Remedy

Chapter 10's WLS handled unequal variances by weighting each observation individually. **GLS** generalizes this fully: instead of just a diagonal weight matrix, it allows the **entire error covariance structure** $\boldsymbol{\Sigma} = \text{Var}(\boldsymbol{\varepsilon})$ to be non-diagonal — capturing not just unequal variances but also *correlations between* different observations' errors.

$$ \hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{y} $$

**Plain-English framing, before the matrix:** WLS (Chapter 10) only had to worry about each point having its *own* level of noise — it never had to worry about points *influencing each other's* noise. GLS drops that restriction entirely: it can handle a situation where point 3's error is partly explained by point 2's error, which is exactly the autocorrelation problem this chapter is about. Practically, GLS is "do the weighting trick from Chapter 10, but let the weights account for relationships *between* points, not just each point's own individual noise level."

**WLS is a special case of GLS** where $\boldsymbol{\Sigma}$ happens to be diagonal (unequal variances, but zero correlation between observations) — this is worth stating explicitly in an interview, since it ties Chapters 10 and 11 together as two instances of the same general fix, just with different assumed structures for $\boldsymbol{\Sigma}$.

**For AR(1) errors specifically**, $\boldsymbol{\Sigma}$ takes a specific banded (Toeplitz) form:

$$ \boldsymbol{\Sigma} = \frac{\sigma_u^2}{1-\rho^2}\begin{bmatrix}1&\rho&\rho^2&\cdots\\\rho&1&\rho&\cdots\\\rho^2&\rho&1&\cdots\\\vdots&&&\ddots\end{bmatrix} $$

**Reading this matrix in plain words:** each entry tells you "how correlated is the error at this row's observation with the error at this column's observation." The diagonal is all 1's (scaled) because every observation's error is, of course, perfectly correlated with itself. Moving one step off the diagonal, entries are $\rho$ — the direct echo between adjacent points. Two steps away, it's $\rho^2$ — a weaker echo, because the relationship has to "travel through" one intermediate point and fades as it does. This is the mathematical version of "yesterday strongly predicts today, but last week only weakly predicts today" — the correlation fades the further apart in time two points are.

---

## 11.5 The Cochrane-Orcutt / Prais-Winsten Transformation

Directly inverting $\boldsymbol{\Sigma}$ by hand is impractical even for moderate $n$. The standard practical shortcut transforms the data so that plain OLS on the *transformed* variables is equivalent to GLS on the original ones:

$$ y_t^* = y_t - \hat{\rho}\,y_{t-1}, \qquad x_t^* = x_t - \hat{\rho}\,x_{t-1} \qquad (\text{for } t=2,...,n) $$

Then run **ordinary** OLS of $y_t^*$ on $x_t^*$. The intuition: subtracting off $\hat{\rho}$ times the previous observation "removes" the correlated component, leaving behind approximately independent noise — the same trick, structurally, as differencing in ARIMA time-series modeling. (Prais-Winsten additionally recovers the first observation, which Cochrane-Orcutt simply discards, via a specific transformation for $t=1$ — a minor technical refinement.)

**Plain-English version of the trick:** rather than doing the hard matrix math with $\boldsymbol{\Sigma}^{-1}$ directly, there's a shortcut: subtract off "the echo you'd expect from last period" before running an ordinary regression. If $\rho$ tells you how much of today's error is just an echo of yesterday's, then subtracting $\hat{\rho}$ times yesterday's value from today's value effectively "cancels out" that echo, leaving something closer to genuinely fresh, independent noise. Once the echo is removed, plain old OLS works correctly again on the *transformed* data.

**Applied to this chapter's data:** since $\hat{\rho}\approx-0.01$ (essentially zero), this transformation would barely change $x_t$ or $y_t$ at all — correctly reflecting that GLS collapses back to ordinary OLS when $\rho=0$ (just as WLS collapses back to OLS when all weights are equal). This is a useful sanity check to state out loud in an interview: **GLS and WLS are both strict generalizations of OLS that reduce exactly to OLS when their respective assumed structures ($\rho=0$, or equal weights) hold.**

**In plain words, why this makes sense:** if there's no real echo between observations ($\rho\approx0$), then "subtracting off the echo" barely changes anything — you subtract almost nothing, because there was almost nothing to subtract. That's a good consistency check: a remedy for a problem you don't actually have should do almost nothing when applied, and that's exactly what happens here.

---

## 11.6 An Alternative Remedy: Newey-West (HAC) Standard Errors

Just as Chapter 10 offered "robust sandwich" standard errors as an alternative to WLS when the variance structure is unknown, **Newey-West standard errors** (also called **HAC** — Heteroscedasticity and Autocorrelation Consistent) serve the same role here: they keep the ordinary OLS point estimates unchanged, but correct the standard errors to remain valid under **both** heteroscedasticity and autocorrelation up to some specified lag, without requiring you to specify the exact AR structure. This is the most common practical choice in applied time-series work when the priority is trustworthy inference rather than fully modeling the error-generating process.

**Plain-English summary:** this is the autocorrelation-flavored cousin of Chapter 10's "sandwich" standard errors. You keep your regular OLS coefficients exactly as they were, and just correct the *error bars* around them to properly account for both "some points being noisier than others" and "nearby points' errors being related to each other" — all without having to first prove or assume exactly how strong that relationship is (no need to estimate $\rho$ precisely, unlike Cochrane-Orcutt).

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
