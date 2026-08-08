# Chapter 16 — Ridge Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Uses the centered version of Chapter 5's dataset ($x_1,x_2$ correlated with VIF≈9.33 from Chapter 9) to show ridge regression in action on exactly the instability that chapter diagnosed.*

---

## 16.1 The Motivating Question — Picking Up Chapter 6's Closing Thread

Chapter 6 proved OLS is BLUE — the minimum-variance option **among unbiased estimators** — and explicitly flagged that a **biased** estimator could still win on total error (bias² + variance) if it traded a little bias for a large variance reduction. Chapter 9 then showed exactly the scenario where OLS's variance becomes a real liability: under multicollinearity, $(\mathbf{X}^T\mathbf{X})^{-1}$'s diagonal entries blow up, making coefficient estimates wildly unstable (recall $VIF\approx9.33$ for both $x_1,x_2$).

**Ridge regression is the direct answer to that setup:** it deliberately introduces a small, controlled bias in exchange for a substantial reduction in variance — precisely by preventing $\mathbf{X}^T\mathbf{X}$ from ever being close to singular in the first place.

---

## 16.2 The Ridge Objective

Ordinary least squares minimizes $RSS(\boldsymbol{\beta})$ alone (Chapter 3). Ridge regression adds a penalty on the size of the coefficients:

$$ RSS_{ridge}(\boldsymbol{\beta}) = \sum_{i=1}^n(y_i-\mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^p\beta_j^2 = RSS(\boldsymbol{\beta}) + \lambda\|\boldsymbol{\beta}\|_2^2 $$

**Plain-English reading:** $\lambda\geq0$ is a **tuning parameter** controlling how much you penalize large coefficients. $\lambda=0$ recovers ordinary OLS exactly. As $\lambda\to\infty$, every coefficient is squeezed toward zero (an increasingly flat, "safe" model). **Critically, the intercept $\beta_0$ is never penalized** — only the slope coefficients — because the intercept simply reflects the overall level of $y$, which shouldn't be shrunk toward zero on principle; the standard way to handle this cleanly is to **center** $y$ and every predictor first (as we do below), so the intercept is just $\bar{y}$, estimated separately and left untouched by the penalty.

**Geometric picture:** minimizing $RSS_{ridge}$ is equivalent to minimizing plain $RSS$ subject to the constraint $\sum\beta_j^2\leq s$ for some $s$ that shrinks as $\lambda$ grows — i.e., constraining $\boldsymbol{\beta}$ to lie inside a **circular (L2) ball** around the origin. Contrast this with Chapter 17's lasso, whose constraint region is a diamond (L1 ball) instead — a shape difference with major practical consequences we'll return to there.

*(Diagram to visualize: elliptical RSS contour lines centered on the OLS solution, with a small circle centered at the origin representing the ridge constraint — the ridge solution sits at the point where the smallest ellipse touches the circle, generally pulled toward the origin along both axes simultaneously, in contrast to a diamond-shaped constraint which tends to touch an ellipse exactly at a corner, an axis, setting some coefficients to exactly zero.)*

---

## 16.3 Deriving the Closed-Form Solution

Following the same matrix-calculus approach as Chapter 3, §3.4:

$$ RSS_{ridge}(\boldsymbol{\beta}) = (\mathbf{y}-\mathbf{X}\boldsymbol{\beta})^T(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}) + \lambda\boldsymbol{\beta}^T\boldsymbol{\beta} $$

Taking the derivative with respect to $\boldsymbol{\beta}$ and setting to zero:

$$ -2\mathbf{X}^T\mathbf{y}+2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}+2\lambda\boldsymbol{\beta} = 0 $$

$$ (\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y} $$

$$ \boxed{\hat{\boldsymbol{\beta}}_{ridge} = (\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}} $$

**This is the single most important fact in the chapter, worth stating explicitly:** adding $\lambda\mathbf{I}$ to $\mathbf{X}^T\mathbf{X}$ before inverting is *exactly* what prevents the near-singularity problem from Chapter 9 — even if $\mathbf{X}^T\mathbf{X}$ is nearly singular (smallest eigenvalue close to 0), adding any $\lambda>0$ pushes every eigenvalue up by $\lambda$, guaranteeing $(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})$ is comfortably invertible. Ridge regression is, quite literally, a numerically stabilized version of the normal equations from Chapter 3.

---

## 16.4 Worked Example — Ridge on the Multicollinear Dataset from Chapter 9

Center all variables first ($x_1$ around $\bar{x}_1=3$, $x_2$ around $\bar{x}_2=1.8$, $y$ around $\bar{y}=64.6$). The centered cross-product matrix (already computed pieces from Chapters 5 and 9):

$$ \mathbf{X}_c^T\mathbf{X}_c = \begin{bmatrix}S_{x_1x_1}&S_{x_1x_2}\\S_{x_1x_2}&S_{x_2x_2}\end{bmatrix} = \begin{bmatrix}10&5\\5&2.8\end{bmatrix}, \qquad \mathbf{X}_c^T\mathbf{y}_c = \begin{bmatrix}81\\42.6\end{bmatrix} $$

**Check at $\lambda=0$ (should recover Chapter 5's OLS slopes exactly):** $\det=10(2.8)-5(5)=3$, giving $\hat{\beta}_1=4.6,\ \hat{\beta}_2=7$ — **exact match to Chapter 5.** Good confirmation the centered setup is equivalent to the original.

**At $\lambda=1$:**

$$ \mathbf{X}_c^T\mathbf{X}_c+\mathbf{I} = \begin{bmatrix}11&5\\5&3.8\end{bmatrix}, \qquad \det=11(3.8)-25=16.8 $$

$$ \hat{\beta}_{1,ridge} = \frac{3.8(81)-5(42.6)}{16.8} = \frac{307.8-213}{16.8} \approx 5.64 $$

$$ \hat{\beta}_{2,ridge} = \frac{-5(81)+11(42.6)}{16.8} = \frac{-405+468.6}{16.8} \approx 3.79 $$

**At $\lambda=5$:**

$$ \mathbf{X}_c^T\mathbf{X}_c+5\mathbf{I} = \begin{bmatrix}15&5\\5&7.8\end{bmatrix}, \qquad \det=15(7.8)-25=92 $$

$$ \hat{\beta}_{1,ridge}=\frac{7.8(81)-5(42.6)}{92}\approx4.55, \qquad \hat{\beta}_{2,ridge}=\frac{-5(81)+15(42.6)}{92}\approx2.54 $$

| $\lambda$ | $\hat{\beta}_1$ | $\hat{\beta}_2$ | $\|\hat{\boldsymbol{\beta}}\|_2$ |
|---|---|---|---|
| 0 (OLS) | 4.60 | 7.00 | 8.38 |
| 1 | 5.64 | 3.79 | 6.80 |
| 5 | 4.55 | 2.54 | 5.22 |

**Two things worth noticing.** First, the overall coefficient norm shrinks monotonically as $\lambda$ grows (8.38 → 6.80 → 5.22), exactly as ridge is designed to do. Second — and this is the more interesting, less-obvious result — $\hat{\beta}_2$ (the coefficient that was individually *insignificant* under OLS in Chapter 5, precisely because of multicollinearity) shrinks dramatically (7.00 → 3.79 → 2.54), while $\hat{\beta}_1$ barely moves and even ticks up slightly at $\lambda=1$ before settling down. **This is ridge doing exactly what Chapter 9 anticipated:** with two highly correlated predictors "fighting" over shared credit under OLS, the penalty resolves that instability by pulling the coefficients toward a more conservative, shared, and far less noise-sensitive allocation — rather than letting them swing to extreme, individually-uncertain values.

**Practical note on standardization:** because the penalty $\lambda\sum\beta_j^2$ treats every coefficient identically regardless of its predictor's scale, predictors should generally be **standardized** (mean 0, variance 1) before applying ridge — otherwise a predictor measured in different units would be penalized unfairly relative to another. We centered but didn't fully standardize the variances above for arithmetic simplicity ($S_{x_1x_1}=10$ vs. $S_{x_2x_2}=2.8$ remain on different scales); in a real analysis, dividing each predictor by its own standard deviation first is standard practice.

---

## 16.5 The Bias-Variance Tradeoff, Made Formal

$$ E[\hat{\boldsymbol{\beta}}_{ridge}] = (\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}\,\boldsymbol{\beta} \neq \boldsymbol{\beta} \quad\text{(for }\lambda>0\text{)} $$

This confirms ridge is **biased** whenever $\lambda>0$ — a direct, deliberate departure from Gauss-Markov's unbiasedness requirement (Chapter 6). In exchange:

$$ \text{Var}(\hat{\boldsymbol{\beta}}_{ridge}) = \sigma^2(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1} $$

which is **provably smaller** (in the same positive-semi-definite matrix sense as Chapter 6's proof) than OLS's variance $\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$, for any $\lambda>0$. The **total expected prediction error** decomposes as $\text{Bias}^2+\text{Variance}+\text{irreducible noise}$ — ridge accepts a bias penalty specifically because, under multicollinearity, the variance reduction it buys can be large enough to shrink this total, even though no single component alone tells the whole story.

---

## 16.6 Choosing $\lambda$

$\lambda$ is a **hyperparameter**, not something estimated by the same normal-equations machinery — it's chosen via **cross-validation** (a direct callback to Chapter 15): fit ridge across a grid of candidate $\lambda$ values, compute k-fold (or LOOCV) test error for each, and pick the $\lambda$ that minimizes estimated out-of-sample error. This is exactly the same LOOCV mechanism worked by hand in Chapter 15, just repeated across a grid of $\lambda$ rather than compared across a fixed pair of models.

---

## 16.7 Where the Textbooks Differ

- **Kutner** covers ridge only briefly, mostly as a remedy explicitly tied to the multicollinearity diagnostics from its VIF/condition-number chapter — practical rather than deeply theoretical.
- **Montgomery** gives ridge somewhat more room, including ridge trace plots (coefficients plotted as a function of $\lambda$) as a practical diagnostic tool for choosing a "stable-looking" $\lambda$ visually, historically predating formal cross-validation as the standard selection method.
- **ESL/ISL** give ridge its fullest theoretical treatment among the four sources — the bias-variance derivation, the connection to principal components (ridge shrinks more aggressively along low-variance principal component directions of $\mathbf{X}$, a fact worth knowing conceptually even without deriving it here), and the L2-ball geometric picture are all covered in depth.
- **Sheather** treats ridge primarily through its software implementation (`glmnet` in R), emphasizing the cross-validation curve for choosing $\lambda$ over the closed-form algebra.

---

## 16.8 Interview Q&A

**Q: Write the closed-form ridge regression estimator and explain why it's always well-defined, even under severe multicollinearity.**
A: $\hat{\boldsymbol{\beta}}_{ridge}=(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$. Adding $\lambda\mathbf{I}$ (for $\lambda>0$) shifts every eigenvalue of $\mathbf{X}^T\mathbf{X}$ up by $\lambda$, guaranteeing the matrix being inverted is nonsingular even if $\mathbf{X}^T\mathbf{X}$ itself is exactly or nearly singular.

**Q: Is ridge regression's estimator biased? Why would you ever want a biased estimator?**
A: Yes, biased for any $\lambda>0$ — this directly contradicts Gauss-Markov's unbiasedness requirement (Chapter 6). It's worth using because the resulting variance reduction can be large enough, especially under multicollinearity, to reduce *total* expected prediction error (bias²+variance) below what unbiased OLS achieves.

**Q: Why must predictors typically be standardized before applying ridge regression?**
A: The penalty $\lambda\sum\beta_j^2$ treats every coefficient identically regardless of scale — an unstandardized predictor with naturally larger raw values would have its coefficient penalized unfairly relative to a predictor on a smaller scale, distorting which variables get shrunk most.

**Q: How is $\lambda$ chosen in practice?**
A: Via cross-validation — evaluate out-of-sample error (e.g., k-fold or LOOCV, Chapter 15) across a grid of candidate $\lambda$ values and select the one minimizing estimated test error; it is a hyperparameter tuned externally, not estimated by the normal equations themselves.

**Q: What happens to ridge coefficients as $\lambda\to\infty$? As $\lambda\to0$?**
A: As $\lambda\to0$, ridge reduces exactly to OLS. As $\lambda\to\infty$, every slope coefficient is driven toward zero (though never exactly zero, in contrast to lasso — Chapter 17), converging toward an intercept-only model.

---

*End of Chapter 16. Next: Chapter 17 — Lasso Regression (the L1 penalty, why its diamond-shaped constraint region produces exact zeros/sparsity where ridge's circular region doesn't, and coordinate descent as the standard fitting algorithm).*
