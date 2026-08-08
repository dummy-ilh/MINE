# Chapter 11 — Autocorrelation

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Uses Chapter 5's residuals (in their original observation order) to work the Durbin-Watson statistic by hand, with an honest caveat about what this dataset can and can't actually demonstrate.*

---

## 11.1 The Motivating Question

Chapters 9–10 handled two ways the LINE assumptions can break: correlated *predictors* (multicollinearity) and unequal error *variance* (heteroscedasticity). This chapter handles a third: **errors correlated with each other** — most commonly, one observation's error being related to the *previous* observation's error, which is why this problem is almost always discussed in the context of **time-ordered (time-series) data**.

**What breaks, precisely (same pattern as Chapter 10):** OLS point estimates $\hat{\boldsymbol{\beta}}$ remain **unbiased** even under autocorrelated errors. What's lost is BLUE-ness (Chapter 6) and the validity of standard errors — and in practice, autocorrelation tends to make OLS standard errors look **artificially small**, making effects appear more statistically significant than they really are. This is arguably the most dangerous of the three violations covered in Chapters 9–11, precisely because it silently inflates your confidence.

**An honest framing note for this chapter:** our running dataset is 5 students measured once each — there's no genuine time ordering, so a Durbin-Watson test here is really a demonstration of the *mechanics*, not a meaningful diagnostic (there's no reason student 3's error should relate to student 2's error just because they happen to be listed adjacently). We work the numbers anyway so the formula is concrete, but the right context for this test is genuine time-series or sequentially-collected data — e.g., quarterly sales data, repeated sensor readings — where adjacent-observation correlation is a real physical possibility.

---

## 11.2 The Durbin-Watson Test

**Test statistic:**

$$ DW = \frac{\sum_{i=2}^{n}(e_i-e_{i-1})^2}{\sum_{i=1}^{n}e_i^2} $$

**Plain-English reading:** the numerator measures how much consecutive residuals *differ* from each other; the denominator is just total residual variation (SSE). If consecutive residuals tend to be similar (positive autocorrelation — a positive residual tends to be followed by another positive one), the numerator shrinks relative to the denominator, and $DW$ drops well below 2. If consecutive residuals tend to be dissimilar (negative autocorrelation), $DW$ rises above 2. **$DW\approx2$ indicates no autocorrelation.**

**Worked example**, using Chapter 5's residuals in order: $e = 0.2,\ 0.6,\ -1,\ -0.6,\ 0.8$.

$$ \text{Differences: } (0.6-0.2),\ (-1-0.6),\ (-0.6-(-1)),\ (0.8-(-0.6)) = 0.4,\ -1.6,\ 0.4,\ 1.4 $$

$$ \sum(\text{differences})^2 = 0.16+2.56+0.16+1.96 = 4.84 $$

$$ DW = \frac{4.84}{2.4} \approx 2.02 $$

**Reading it:** $DW\approx2.02$ is almost exactly 2 — no evidence of autocorrelation. As flagged above, this is exactly what you'd expect from **cross-sectional** data where the observation order is arbitrary (alphabetical by student, say) rather than reflecting any real time sequence — there's no mechanism by which this result should show correlation, and it doesn't.

**Formal hypothesis testing caveat:** Durbin-Watson critical values ($d_L, d_U$) come from specialized tables indexed by $n$ and the number of predictors $k$, and those tables conventionally start around $n=15$ — this chapter's $n=5$ is too small for a formally valid critical-value lookup. The computation above is a legitimate illustration of the *statistic's mechanics*, not a valid hypothesis test at this sample size.

---

## 11.3 Connecting DW to the Autocorrelation Coefficient $\rho$

Under a simple **AR(1)** error structure ($\varepsilon_t = \rho\varepsilon_{t-1}+u_t$, where $u_t$ is genuinely uncorrelated noise), there's a direct approximate relationship:

$$ DW \approx 2(1-\hat{\rho}) \qquad\Rightarrow\qquad \hat{\rho} \approx 1-\frac{DW}{2} $$

**Worked check:** $\hat{\rho} \approx 1 - 2.02/2 = 1-1.01 = -0.01$ — essentially zero, consistent with the "no autocorrelation" reading above. $\hat{\rho}=1$ would indicate perfect positive autocorrelation ($DW\to0$); $\hat{\rho}=-1$ would indicate perfect negative autocorrelation ($DW\to4$); $\hat{\rho}=0$ (our case) gives $DW=2$ exactly.

---

## 11.4 Generalized Least Squares (GLS) — The General Remedy

Chapter 10's WLS handled unequal variances by weighting each observation individually. **GLS** generalizes this fully: instead of just a diagonal weight matrix, it allows the **entire error covariance structure** $\boldsymbol{\Sigma} = \text{Var}(\boldsymbol{\varepsilon})$ to be non-diagonal — capturing not just unequal variances but also *correlations between* different observations' errors.

$$ \hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\boldsymbol{\Sigma}^{-1}\mathbf{y} $$

**WLS is a special case of GLS** where $\boldsymbol{\Sigma}$ happens to be diagonal (unequal variances, but zero correlation between observations) — this is worth stating explicitly in an interview, since it ties Chapters 10 and 11 together as two instances of the same general fix, just with different assumed structures for $\boldsymbol{\Sigma}$.

**For AR(1) errors specifically**, $\boldsymbol{\Sigma}$ takes a specific banded (Toeplitz) form:

$$ \boldsymbol{\Sigma} = \frac{\sigma_u^2}{1-\rho^2}\begin{bmatrix}1&\rho&\rho^2&\cdots\\\rho&1&\rho&\cdots\\\rho^2&\rho&1&\cdots\\\vdots&&&\ddots\end{bmatrix} $$

---

## 11.5 The Cochrane-Orcutt / Prais-Winsten Transformation

Directly inverting $\boldsymbol{\Sigma}$ by hand is impractical even for moderate $n$. The standard practical shortcut transforms the data so that plain OLS on the *transformed* variables is equivalent to GLS on the original ones:

$$ y_t^* = y_t - \hat{\rho}\,y_{t-1}, \qquad x_t^* = x_t - \hat{\rho}\,x_{t-1} \qquad (\text{for } t=2,...,n) $$

Then run **ordinary** OLS of $y_t^*$ on $x_t^*$. The intuition: subtracting off $\hat{\rho}$ times the previous observation "removes" the correlated component, leaving behind approximately independent noise — the same trick, structurally, as differencing in ARIMA time-series modeling. (Prais-Winsten additionally recovers the first observation, which Cochrane-Orcutt simply discards, via a specific transformation for $t=1$ — a minor technical refinement.)

**Applied to this chapter's data:** since $\hat{\rho}\approx-0.01$ (essentially zero), this transformation would barely change $x_t$ or $y_t$ at all — correctly reflecting that GLS collapses back to ordinary OLS when $\rho=0$ (just as WLS collapses back to OLS when all weights are equal). This is a useful sanity check to state out loud in an interview: **GLS and WLS are both strict generalizations of OLS that reduce exactly to OLS when their respective assumed structures ($\rho=0$, or equal weights) hold.**

---

## 11.6 An Alternative Remedy: Newey-West (HAC) Standard Errors

Just as Chapter 10 offered "robust sandwich" standard errors as an alternative to WLS when the variance structure is unknown, **Newey-West standard errors** (also called **HAC** — Heteroscedasticity and Autocorrelation Consistent) serve the same role here: they keep the ordinary OLS point estimates unchanged, but correct the standard errors to remain valid under **both** heteroscedasticity and autocorrelation up to some specified lag, without requiring you to specify the exact AR structure. This is the most common practical choice in applied time-series work when the priority is trustworthy inference rather than fully modeling the error-generating process.

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

**Q: What does a Durbin-Watson statistic near 2 indicate? Near 0? Near 4?**
A: Near 2: no autocorrelation. Near 0: strong positive autocorrelation (consecutive errors tend to be similar). Near 4: strong negative autocorrelation (consecutive errors tend to alternate in sign).

**Q: Is WLS a special case of GLS, or the other way around?**
A: WLS is a special case of GLS — GLS allows the full error covariance matrix $\boldsymbol{\Sigma}$ to have any structure, including off-diagonal correlations; WLS restricts $\boldsymbol{\Sigma}$ to be diagonal (unequal variances only, no cross-observation correlation).

**Q: Why is autocorrelation often considered more dangerous in practice than heteroscedasticity?**
A: It tends to make OLS standard errors appear artificially small, inflating apparent statistical significance — this is especially common (and easy to miss) in time-series regressions with trending or seasonal data, where naive OLS can produce highly significant-looking but spurious relationships.

**Q: When would you use Newey-West standard errors instead of explicitly modeling the AR structure with GLS?**
A: When you want valid inference without committing to a specific correlation structure (exact AR order, specific $\rho$) — Newey-West corrects standard errors for autocorrelation up to a chosen lag length without requiring you to fully specify or estimate the error-generating process, at the cost of not improving the efficiency of the point estimates the way a correctly-specified GLS would.

---

*End of Chapter 11. Next: Chapter 12 — Transformations (Box-Cox, log transforms, and polynomial terms as remedies for the linearity violations first flagged back in Chapter 7's residuals-vs-fitted panel).*
