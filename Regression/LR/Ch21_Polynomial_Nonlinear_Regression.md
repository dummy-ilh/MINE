# Chapter 21 — Polynomial & Nonlinear Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Introduces a genuinely quadratic dataset to work through the full arc: misspecified linear fit → correctly specified polynomial fit → the collinearity problem this creates → the centering fix promised back in Chapter 12, §12.6.*

**New example dataset** — a quantity following an exact quadratic relationship, $y=2x^2+3x+1$:

| $x$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ | 6 | 15 | 28 | 45 | 66 |

---

## 21.1 The Motivating Question

Chapter 12 introduced polynomial terms briefly as an alternative to transformations for fixing curved residuals, promising a fuller treatment later — this is that chapter. The core idea, worth restating precisely: a polynomial model like $y=\beta_0+\beta_1x+\beta_2x^2$ is still **linear regression** — linear in the parameters $\beta_0,\beta_1,\beta_2$ — even though the fitted curve bends. This is a special case of the general idea of **basis expansion**: instead of restricting predictors to their raw form, feed the model transformed versions of them ($x^2$, $\ln x$, $\sin x$, indicator functions, etc.) as additional columns, and let ordinary linear regression machinery (Chapters 1–9, entirely unchanged) fit coefficients on this expanded basis.

---

## 21.2 Diagnosing the Missing Curvature

Fitting a straight line $y=\beta_0+\beta_1x$ to this exactly-quadratic data (same OLS mechanics as Chapter 1): $\bar{x}=3,\bar{y}=32,S_{xx}=10,S_{xy}=150$:

$$ \hat{\beta}_1=15, \qquad \hat{\beta}_0=32-15(3)=-13 $$

**Residuals:** $4,\ -2,\ -4,\ -2,\ 4$ — a textbook U-shape (positive at both ends, negative in the middle), exactly the linearity-violation signature from Chapter 7's residuals-vs-fitted panel.

---

## 21.3 Fitting the Correctly-Specified Polynomial Model

Adding an $x^2$ column: $y=\beta_0+\beta_1x+\beta_2x^2$. Since the data is exactly quadratic by construction, solving the normal equations (using $\sum x=15,\sum x^2=55,\sum x^3=225,\sum x^4=979$, $\sum y=160,\sum xy=630,\sum x^2y=2688$) recovers the true relationship **exactly**:

$$ \hat{\beta}_0=1, \qquad \hat{\beta}_1=3, \qquad \hat{\beta}_2=2 $$

**Verification** (row-by-row check of the normal equations, same style as every prior chapter): $5(1)+15(3)+55(2)=5+45+110=160$ ✓, matching $\sum y$; the other two rows check out identically. Zero residuals, as expected for data generated from exactly this functional form.

---

## 21.4 The Hidden Collinearity Problem

Here's the catch, previewed in Chapter 12, §12.6: $x$ and $x^2$ are **themselves highly correlated** whenever $x$ doesn't range symmetrically around zero. Computing the correlation between the raw $x$ and $x^2$ columns above ($\bar{x}=3,\bar{x^2}=11$): the cross-product works out to $S_{x,x^2}=60$, with $S_{xx}=10$ and $S_{x^2x^2}=374$, giving:

$$ \text{Corr}(x,x^2) = \frac{60}{\sqrt{10\times374}} = \frac{60}{61.16} \approx 0.981 $$

An implied $VIF = 1/(1-0.981^2) \approx 26.3$ — **severe** multicollinearity by Chapter 9's standard thresholds, arising purely from the modeling choice itself (adding $x^2$ as a raw column), not from anything about the underlying data-generating process. This would make the individual coefficients on $x$ and $x^2$ separately unstable and hard to interpret in a noisier, non-exact-fit dataset, exactly the symptom diagnosed back in Chapter 5, §5.4 and formalized in Chapter 9.

---

## 21.5 The Centering Fix

**The fix, as promised in Chapter 12:** center $x$ before squaring it. Let $x_c=x-\bar{x}$: values $-2,-1,0,1,2$. Then $x_c^2$: values $4,1,0,1,4$.

Computing the correlation between $x_c$ and $x_c^2$: $\bar{x_c}=0$, $\bar{x_c^2}=2$, and the cross-product $\sum(x_c-0)(x_c^2-2) = (-2)(2)+(-1)(-1)+(0)(-2)+(1)(-1)+(2)(2) = -4+1+0-1+4 = 0$.

$$ \text{Corr}(x_c,x_c^2) = 0 \quad\text{exactly} $$

**Centering completely eliminates the collinearity between the linear and quadratic terms** — not just reduces it, but drives it to exactly zero, for this symmetric spacing of $x$-values (a consequence of $x_c$ being an odd function of position around the mean, while $x_c^2$ is even — their product sums to zero by symmetry). Even with asymmetric spacing, centering substantially reduces (though may not fully zero out) this artificial collinearity in general. **This is why centering predictors before creating polynomial or interaction terms (Chapter 13) is standard practice**, not just a cosmetic convenience — it directly addresses a collinearity problem that the modeling choice itself introduces, entirely separate from any collinearity inherent in the original data.

**For polynomials of degree 3 or higher**, simple centering isn't fully sufficient to eliminate all cross-term correlations — the standard further refinement is to use **orthogonal polynomials** (e.g., via Gram-Schmidt or specialized recurrence relations), which construct a basis where every pair of polynomial-degree columns is exactly uncorrelated by construction, generalizing this section's centering trick to arbitrary polynomial degree.

---

## 21.6 The Risk of High-Degree Polynomials

Fitting a degree-$(n-1)$ polynomial to $n$ data points achieves a perfect fit (zero residuals) **trivially** — with enough parameters, any curve can be threaded exactly through any set of points, regardless of whether that curve reflects genuine structure or is simply overfitting noise. High-degree polynomials are also notorious for **Runge's phenomenon** — wild oscillations near the edges of the data range, even when the fit looks reasonable in the middle. **Choosing the polynomial degree is a direct instance of the model-selection problem from Chapter 14 and the overfitting problem from Chapter 15** — use cross-validation (or AIC/BIC/adjusted $R^2$) to select the degree that generalizes well, rather than always choosing the degree that maximizes in-sample fit.

---

## 21.7 A Brief Introduction to Splines

Polynomials have one significant limitation even after fixing the collinearity problem: a single global polynomial can be forced to fit unrelated curvature patterns in different regions of $x$ simultaneously, often at the cost of flexibility exactly where it's needed. **Splines** address this by fitting **different polynomial pieces in different regions**, joined together smoothly at chosen points called **knots**.

- **Piecewise polynomials**: fit a separate low-degree polynomial (often cubic) in each region between knots.
- **Continuity constraints**: require the pieces to join smoothly — matching value, first derivative, and (for cubic splines) second derivative at each knot, so the overall curve has no visible kinks.
- **Natural cubic splines**: additionally constrain the curve to be linear beyond the boundary knots, taming the wild edge behavior (Runge's phenomenon) that plain polynomials suffer from.
- **Smoothing splines**: rather than choosing knot locations manually, place a knot at every unique data point but add a **roughness penalty** (penalizing the integrated squared second derivative of the fitted curve) to prevent overfitting — this is structurally the same idea as ridge regression's L2 penalty (Chapter 16), just penalizing curvature instead of coefficient magnitude directly, with its own tuning parameter chosen via cross-validation (Chapter 15).

Splines remain, technically, an instance of basis expansion (§21.1) — a spline is fit via ordinary linear regression on a cleverly constructed set of basis-function columns — so everything from Chapters 1–9 (inference, diagnostics) still technically applies to a spline model's fitted coefficients, even though the basis functions themselves are more elaborate than a simple polynomial.

---

## 21.8 Where the Textbooks Differ

- **Kutner and Montgomery** cover polynomial regression thoroughly, including the centering fix and collinearity diagnostics from §21.4–21.5, but give splines only minimal treatment, if any — reflecting their more classical-statistics scope.
- **Sheather** covers both polynomials and a solid introduction to splines, with practical R-based fitting guidance (e.g., the `ns()`/`bs()` functions for natural/B-splines).
- **ESL/ISL** give splines — and basis expansion generally — their fullest and most central treatment among the four sources, including smoothing splines' explicit connection to ridge-style roughness penalties (§21.7) and the broader generalization to kernel methods, reflecting the more flexible, prediction-focused ML perspective that basis expansion methods sit naturally within.

---

## 21.9 Interview Q&A

**Q: Is polynomial regression "linear" or "nonlinear"?**
A: It's linear regression — linear in the parameters $\beta_0,\beta_1,\beta_2,...$ — even though the fitted curve is a nonlinear (curved) function of $x$. All of the OLS matrix machinery, inference, and diagnostics from earlier chapters apply unchanged; only the design matrix gains additional columns ($x^2, x^3,$ etc.).

**Q: Why does adding a raw $x^2$ term often cause severe multicollinearity, and how do you fix it?**
A: $x$ and $x^2$ are typically highly correlated whenever $x$ doesn't range symmetrically around zero — this is an artifact of the modeling choice, not the underlying data. Centering $x$ before squaring it (or using orthogonal polynomials for higher degrees) removes or substantially reduces this artificial collinearity.

**Q: How do you choose the right polynomial degree?**
A: Treat it as a model-selection problem (Chapter 14) — use cross-validation, AIC/BIC, or adjusted $R^2$ to compare degrees, since a higher degree always achieves an equal-or-better in-sample fit but risks severe overfitting (including Runge's phenomenon, wild oscillation near the data boundaries).

**Q: What's the difference between a polynomial fit and a spline?**
A: A single polynomial applies one global functional form across the entire range of $x$. A spline fits separate (typically cubic) polynomial pieces in different regions, joined smoothly at knots — allowing much more local flexibility without forcing distant regions of the data to share the same curvature pattern.

**Q: How does a smoothing spline relate to ridge regression?**
A: Both add a penalty to control model complexity as a tuning parameter, chosen via cross-validation — ridge penalizes the squared magnitude of the coefficients directly, while a smoothing spline penalizes the integrated squared second derivative (roughness/curvature) of the fitted curve. Structurally, they're the same regularization idea applied to different notions of "complexity."

---

*End of Chapter 21. Next: Chapter 22 — From Linear to Logistic (the GLM bridge connecting everything in this curriculum to binary-outcome modeling, linking forward to the existing logistic regression curriculum).*
