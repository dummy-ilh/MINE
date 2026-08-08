# Chapter 10 — Heteroscedasticity

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Uses Chapter 5's noisy dataset (residuals $e=0.2,0.6,-1,-0.6,0.8$; $SSE=2.4$) for the formal test, and a simple-regression version for the Weighted Least Squares illustration.*

---

## 10.1 The Motivating Question

Chapter 7's "scale-location" panel flagged a funnel-shaped spread of residuals as a visual warning sign. This chapter makes that warning rigorous: **is the assumption of constant error variance ($\text{Var}(\varepsilon_i)=\sigma^2$ for all $i$) actually violated, formally, or does it just look that way by eye?**

**Why it matters, precisely:** under heteroscedasticity, OLS coefficient estimates $\hat{\boldsymbol{\beta}}$ **remain unbiased** — Gauss-Markov's unbiasedness doesn't require homoscedasticity. What breaks is **BLUE-ness** (Chapter 6): OLS is no longer the *minimum-variance* linear unbiased estimator, and — the part that actually bites in practice — the standard errors, t-tests, and confidence intervals from Chapters 2 and 5 become **invalid**, because their formulas assumed $\text{Var}(\boldsymbol{\varepsilon})=\sigma^2\mathbf{I}$. You can still get a reasonable point estimate; you can no longer trust the p-value attached to it.

---

## 10.2 The Breusch-Pagan Test — Formalizing the Funnel Shape

**Core idea:** if error variance depends on the predictors, then squared residuals ($e_i^2$, our best available stand-in for the unobservable $\varepsilon_i^2$) should be *predictable* from those same predictors. So: **regress the squared residuals on the predictors**, and see if that auxiliary regression explains a significant amount of variation.

**Procedure:**
1. Fit the original model, obtain residuals $e_i$.
2. Fit an **auxiliary regression**: $e_i^2$ on the same predictors $x_1, ..., x_p$.
3. Compute the test statistic $BP = n\times R^2_{aux}$, which follows a $\chi^2_p$ distribution under $H_0:$ homoscedasticity.

**Worked example.** Using Chapter 5's residuals ($e=0.2,0.6,-1,-0.6,0.8$, so $e^2=0.04,0.36,1,0.36,0.64$), regress $e^2$ on $x_1,x_2$ using the same normal-equations machinery from Chapter 4 (the design matrix $\mathbf{X}$, and its $(\mathbf{X}^T\mathbf{X})^{-1}$, are unchanged from every prior chapter).

Solving gives the auxiliary fit: $\hat{z} = 0 - 0.08x_1 + 0.4x_2$ (where $z=e^2$), with:

$$ SSE_{aux} = 0.3264, \qquad SST_{aux} = 0.5184 \qquad\Rightarrow\qquad R^2_{aux} = 1-\frac{0.3264}{0.5184} \approx 0.370 $$

$$ BP = n\times R^2_{aux} = 5\times0.370 \approx 1.85 $$

Comparing to $\chi^2_2$ at $\alpha=0.05$ (critical value $\approx5.99$, with $df=p=2$ predictors): since $1.85 < 5.99$, **we fail to reject homoscedasticity** — no formal evidence of heteroscedasticity in this dataset. (As with earlier small-sample tests in this curriculum, a 5-observation dataset has very low power to detect anything — this result should be read as "no signal detected here," not "homoscedasticity is confirmed.")

---

## 10.3 White's Test — A More General Alternative

Breusch-Pagan assumes variance changes *linearly* with the predictors. **White's test** is more general: the auxiliary regression includes not just $x_1,x_2$ but also their **squares and cross-product** ($x_1^2, x_2^2, x_1x_2$), catching nonlinear or interactive variance patterns that Breusch-Pagan would miss.

$$ White = n\times R^2_{aux} \sim \chi^2_{df} $$

where $df$ equals the number of terms in the expanded auxiliary regression (here, 5: $x_1,x_2,x_1^2,x_2^2,x_1x_2$). **Trade-off:** White's test is more flexible but requires substantially more data relative to the number of auxiliary terms — with only 5 observations and 5 auxiliary predictors (plus intercept), this dataset has zero degrees of freedom left for White's test, illustrating concretely why White's test is a large-sample tool; it isn't attempted here for that reason.

---

## 10.4 Remedy 1 — Weighted Least Squares (WLS)

If you know (or can reasonably model) *how* variance changes across observations, **WLS** directly fixes the problem by down-weighting noisier observations and up-weighting more precise ones:

$$ \hat{\boldsymbol{\beta}}_{WLS} = (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{y} $$

where $\mathbf{W}$ is a diagonal matrix of weights, typically $w_i = 1/\hat{\sigma}_i^2$ — observations with larger assumed variance get **smaller** weight, correctly reflecting that they carry less reliable information. WLS is, in fact, the **new BLUE** estimator once homoscedasticity is violated but the weight structure is known — a direct instance of Chapter 6's closing point that a different linear unbiased estimator becomes optimal once Gauss-Markov's assumptions change.

**Worked illustration (simple regression, $y$ on $x_1$ alone, for tractability).** Suppose we assume $\text{Var}(\varepsilon_i)\propto x_{1i}$ — a common textbook setup (variance grows with the predictor) — so weights are $w_i=1/x_{1i}$: $w=1,\ 0.5,\ 0.333,\ 0.25,\ 0.2$.

Compute weighted means: $\bar{x}_w = \frac{\sum w_ix_{1i}}{\sum w_i} = \frac{5}{2.283}\approx2.190$, $\bar{y}_w=\frac{\sum w_iy_i}{\sum w_i}=\frac{133.27}{2.283}\approx58.37$.

Then, analogous to Chapter 1's $S_{xy}/S_{xx}$ but weighted:

$$ \hat{\beta}_{1,WLS} = \frac{\sum w_i(x_{1i}-\bar{x}_w)(y_i-\bar{y}_w)}{\sum w_i(x_{1i}-\bar{x}_w)^2} = \frac{31.17}{4.05} \approx 7.70 $$

$$ \hat{\beta}_{0,WLS} = \bar{y}_w - \hat{\beta}_{1,WLS}\bar{x}_w = 58.37-7.70(2.190) \approx 41.5 $$

Compare to the **unweighted** OLS simple-regression result from Chapter 5, §5.5: $\hat{\beta}_1=8.1$. WLS shifts the slope down to **7.70** — because student 5 (the highest-$x_1$ point, assumed to carry the most variance under our $\text{Var}\propto x_1$ assumption) is now down-weighted, reducing its pull on the fitted line. **This is the entire mechanism of WLS in one number:** it's OLS, but each point's "vote" in determining the line is scaled by how much you trust that point.

---

## 10.5 Remedy 2 — Robust (Sandwich) Standard Errors

WLS requires **knowing** (or modeling) the variance structure — sometimes you don't know it, but still want valid standard errors despite heteroscedasticity of unknown form. The **Huber-White sandwich estimator** achieves this without changing the point estimates at all — it keeps ordinary $\hat{\boldsymbol{\beta}}_{OLS}$, but replaces the standard-error formula:

$$ \widehat{\text{Var}}_{robust}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^T\mathbf{X})^{-1}\left(\mathbf{X}^T\text{diag}(e_i^2)\mathbf{X}\right)(\mathbf{X}^T\mathbf{X})^{-1} $$

**Why "sandwich":** the "bread" $(\mathbf{X}^T\mathbf{X})^{-1}$ appears on both outside layers, with the "filling" $\mathbf{X}^T\text{diag}(e_i^2)\mathbf{X}$ — built directly from the *observed*, possibly heteroscedastic squared residuals — in the middle. This gives asymptotically valid standard errors **without ever having to specify a weighting scheme** — the price is that it's a large-sample (asymptotic) fix, less reliable in very small samples like this chapter's 5-observation dataset.

**Practical choice between the two remedies:** use **WLS** when you have good reason to believe a specific variance structure (e.g., variance proportional to a known predictor, common in survey/aggregated data); use **robust standard errors** when you just want valid inference without committing to a specific weighting model — the modern default in most applied econometrics and observational-data settings, precisely because you rarely know the true variance structure with confidence.

---

## 10.6 Where the Textbooks Differ

- **Kutner** presents Breusch-Pagan with the fullest derivation tying the auxiliary regression back to the same normal-equations machinery used throughout the book.
- **Montgomery** emphasizes WLS heavily, reflecting its industrial-statistics/quality-control roots, where variance structures (e.g., measurement error scaling with quantity produced) are often known from the physical process itself.
- **Sheather** leans hardest into robust standard errors as the default modern remedy, consistent with its applied, software-output-driven approach — most real analyses don't know the true variance structure well enough to justify WLS confidently.
- **ESL/ISL** barely touch heteroscedasticity — it's a classical-inference concern, and their prediction-focused, cross-validation-driven framework is comparatively insensitive to it (out-of-sample predictive accuracy isn't corrupted by heteroscedasticity the way p-values and confidence intervals are).

---

## 10.7 Interview Q&A

**Q: Does heteroscedasticity bias OLS coefficient estimates?**
A: No — OLS remains unbiased. What's invalidated are the standard errors, and therefore the t-tests, F-tests, and confidence intervals built on the assumption of constant variance; OLS also stops being BLUE (a differently-weighted estimator becomes minimum-variance instead).

**Q: What's the difference between the Breusch-Pagan test and White's test?**
A: Breusch-Pagan regresses squared residuals on the original predictors only, assuming variance changes linearly with them. White's test adds squared and cross-product terms, catching more general (nonlinear, interactive) variance patterns — at the cost of needing more degrees of freedom.

**Q: When would you choose WLS over robust standard errors?**
A: WLS when you have good justification for a specific variance structure (e.g., known to scale with a particular predictor) — it improves both efficiency and inference. Robust standard errors when you want valid inference without committing to a specific weighting scheme, since they only correct the standard errors, not the (already unbiased) point estimates.

**Q: What does the "sandwich" in sandwich standard errors refer to?**
A: The formula $(\mathbf{X}^T\mathbf{X})^{-1}(\mathbf{X}^T\text{diag}(e_i^2)\mathbf{X})(\mathbf{X}^T\mathbf{X})^{-1}$ has the same matrix $(\mathbf{X}^T\mathbf{X})^{-1}$ on both outer sides ("bread"), with a middle term built from the observed squared residuals ("filling") that adapts to whatever heteroscedasticity pattern is actually present in the data.

**Q: If your Breusch-Pagan test fails to reject homoscedasticity, does that guarantee your errors have constant variance?**
A: No — failing to reject just means no significant evidence of heteroscedasticity was detected, which is especially uninformative in small samples (low statistical power). It doesn't positively confirm the null hypothesis is true.

---

*End of Chapter 10. Next: Chapter 11 — Autocorrelation (Durbin-Watson test, time-series residual patterns, and Generalized Least Squares as the appropriate remedy when errors are correlated across observations rather than just unequal in variance).*
