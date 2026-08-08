# Chapter 17 — Lasso Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Continues Chapter 16's centered dataset ($x_1,x_2,y$; $S_{x_1x_1}=10, S_{x_2x_2}=2.8, S_{x_1x_2}=5, S_{x_1y}=81, S_{x_2y}=42.6$) so the contrast between ridge and lasso is directly numerical, not just conceptual.*

---

## 17.1 The Motivating Question — What Ridge Doesn't Do

Chapter 16 showed ridge shrinking both $\hat{\beta}_1$ and $\hat{\beta}_2$ toward more conservative values as $\lambda$ grew — but notice that at **no** value of $\lambda$ did $\hat{\beta}_2$ actually reach **exactly** zero; it kept shrinking asymptotically without ever fully dropping out. Sometimes you want more than shrinkage — you want the model to **actively select** which predictors matter and discard the rest entirely, producing a **sparse** model. **Lasso** (Least Absolute Shrinkage and Selection Operator) achieves exactly this, using a different penalty shape.

---

## 17.2 The Lasso Objective — L1 Instead of L2

$$ RSS_{lasso}(\boldsymbol{\beta}) = \sum_{i=1}^n(y_i-\mathbf{x}_i^T\boldsymbol{\beta})^2 + \lambda\sum_{j=1}^p|\beta_j| $$

The only change from ridge (Chapter 16, §16.2) is replacing the squared penalty $\sum\beta_j^2$ with an absolute-value penalty $\sum|\beta_j|$ — the **L1 norm** instead of the **L2 norm**. This single change has an outsized consequence, rooted entirely in geometry.

**Geometric picture:** the L1 constraint region $\sum|\beta_j|\leq s$ is a **diamond** (in 2D; a cross-polytope in higher dimensions) rather than ridge's circle. Diamonds have **corners that sit exactly on the coordinate axes** — and it's a general fact about constrained optimization that the solution tends to land where the elliptical RSS contours first touch the constraint region, which for a diamond is disproportionately likely to be exactly at a corner (where one or more coefficients are precisely zero), rather than at some generic interior-of-an-edge point. A circle, by contrast, has no corners — there's no geometric reason for an ellipse to touch it at a point where any coordinate is exactly zero.

*(Diagram to visualize: two side-by-side panels. Left: elliptical RSS contours touching a circular (L2) constraint region at a generic point off both axes — Chapter 16's ridge picture. Right: the same elliptical contours touching a diamond (L1) constraint region exactly at one of its corners, sitting on the vertical axis — meaning the horizontal-axis coefficient is exactly zero.)*

---

## 17.3 Why There's No Closed-Form Solution — And the Fix: Coordinate Descent

Unlike ridge, the L1 penalty $|\beta_j|$ is **not differentiable at $\beta_j=0$** — its derivative jumps discontinuously there. This means the clean matrix-calculus derivation from Chapter 16, §16.3 doesn't go through, and there's generally no closed-form $\hat{\boldsymbol{\beta}}_{lasso}$. Instead, lasso is fit using **coordinate descent**: optimize one coefficient at a time, holding all others fixed, cycling through all coefficients repeatedly until the values stop changing.

**The key building block — soft-thresholding.** Holding $\beta_2$ fixed and optimizing over $\beta_1$ alone, define the **partial residual** $r_i^{(1)} = y_i - x_{i2}\beta_2$ (what's left of $y$ after removing $x_2$'s current contribution). Minimizing $\sum(r_i^{(1)}-x_{i1}\beta_1)^2+\lambda|\beta_1|$ over $\beta_1$ gives the closed-form **soft-thresholding** update:

$$ \hat{\beta}_1 \leftarrow \text{sign}(z_1)\max\left(|z_1|-\frac{\lambda}{2S_{x_1x_1}},\ 0\right), \qquad z_1 = \frac{\sum x_{i1}r_i^{(1)}}{S_{x_1x_1}} $$

**Plain-English reading:** $z_1$ is simply "what the OLS coefficient on $x_1$ would be, given the current partial residual." Soft-thresholding then either shrinks that value toward zero by a fixed amount $\lambda/(2S_{x_1x_1})$, **or, if that shrinkage would push it past zero, snaps it to exactly zero instead.** That "snap to exactly zero" step is the entire mechanism behind lasso's sparsity — it's a direct consequence of the max$(\cdot,0)$ term, which has no analogue in ridge's smooth shrinkage formula.

---

## 17.4 Worked Example — Coordinate Descent to Convergence, $\lambda=10$

Initialize $\beta_1=\beta_2=0$.

**Update $\beta_1$** (holding $\beta_2=0$, so $r^{(1)}=y_c$ unchanged): $z_1 = S_{x_1y}/S_{x_1x_1} = 81/10 = 8.1$. Threshold $=\lambda/(2S_{x_1x_1})=10/20=0.5$.

$$ \beta_1 \leftarrow 8.1-0.5 = 7.6 $$

**Update $\beta_2$** (holding $\beta_1=7.6$): the partial residual's cross-product with $x_2$ is $S_{x_2y}-\beta_1S_{x_1x_2} = 42.6-7.6(5) = 42.6-38=4.6$, so $z_2=4.6/2.8\approx1.643$. Threshold $=\lambda/(2S_{x_2x_2})=10/5.6\approx1.786$.

Since $|z_2|=1.643 < 1.786$ (the threshold **exceeds** the OLS-implied value):

$$ \beta_2 \leftarrow 0 \quad\text{(exactly)} $$

**Second full pass — confirming convergence:** with $\beta_2=0$, updating $\beta_1$ reproduces the identical computation as before ($z_1=8.1$, threshold $0.5$, $\beta_1=7.6$ again) — unchanged. Updating $\beta_2$ with $\beta_1=7.6$ reproduces $z_2=1.643 < 1.786$ again — still thresholded to exactly zero. **The algorithm has converged:**

$$ \hat{\beta}_{1,lasso}=7.6, \qquad \hat{\beta}_{2,lasso}=0 $$

**Lasso has completely eliminated $x_2$ (practice tests) from the model** — a qualitatively different outcome from ridge, which at every $\lambda$ tried in Chapter 16 kept both coefficients nonzero (even at $\lambda=5$, ridge gave $\hat{\beta}_2\approx2.54$, still clearly nonzero). This is precisely the sparsity/feature-selection property that distinguishes lasso from ridge in practice.

---

## 17.5 Finding the Critical $\lambda$ — Where Sparsity Kicks In

At what $\lambda$ does $\beta_2$ first get thresholded to zero? Setting up the fixed-point condition at convergence (with $\beta_2=0$, so $\beta_1=8.1-\lambda/20$), the threshold condition for $\beta_2=0$ to be stable is:

$$ \frac{42.6-5\beta_1}{2.8} \leq \frac{\lambda}{5.6} \qquad\Rightarrow\qquad \lambda \geq 8.4 $$

**Check against our chapters so far:** at $\lambda=5$ (used for ridge in Chapter 16), lasso would **not** yet produce sparsity ($5<8.4$) — both coefficients would remain nonzero, similar in spirit to ridge's behavior at that same $\lambda$. Only once $\lambda$ crosses **8.4** does $x_2$ get dropped entirely. This threshold behavior — a specific critical value below which all predictors stay active, and above which some get zeroed out — has no analogue in ridge, where coefficients approach zero asymptotically but (mathematically) never exactly reach it for any finite $\lambda$.

**A substantive connection worth stating explicitly:** the surviving coefficient, $\hat{\beta}_{1,lasso}=7.6$, sits close to Chapter 5's simple-regression (reduced-model) slope of $8.1$ (§5.5) — sensible, since once $x_2$ is fully dropped, the remaining model is structurally similar to that earlier reduced model, with the small remaining gap (7.6 vs. 8.1) attributable to the residual shrinkage lasso still applies to $\beta_1$ itself.

---

## 17.6 Choosing $\lambda$, and Ridge vs. Lasso in Practice

As with ridge (Chapter 16, §16.6), $\lambda$ is selected via cross-validation (Chapter 15) — fit across a grid of $\lambda$ values, evaluate out-of-sample error for each, and pick the minimizer.

**When to prefer which, as a practical rule of thumb:**
- **Lasso** when you believe only a subset of predictors truly matter, and you want automatic feature selection built into the fitting process itself — the sparse result is also often easier to interpret and communicate.
- **Ridge** when you believe most/all predictors contribute at least a little, and you specifically want to tame instability among correlated predictors without discarding any of them outright — recall from Chapter 16 that ridge tends to shrink correlated predictors *together*, whereas lasso tends to arbitrarily pick one from a correlated group and zero out the rest, which can be a less stable/interpretable choice when the "winner" among correlated predictors is somewhat arbitrary.
- **Elastic Net** (Chapter 18) combines both penalties specifically to get some of lasso's sparsity while retaining more of ridge's stability under correlated predictors — directly motivated by this exact tension.

---

## 17.7 Where the Textbooks Differ

- **Kutner and Montgomery** (both pre-dating lasso's widespread adoption in classical statistics curricula) cover this topic only lightly, if at all, typically as a brief modern addendum rather than a core chapter — this material is comparatively recent relative to their historical core content.
- **ESL/ISL** give lasso its fullest theoretical treatment among the four sources — the geometric diamond-vs-circle argument in §17.2 is essentially their standard exposition, along with the connection to subset selection and the LARS (Least Angle Regression) algorithm as an alternative fitting approach to coordinate descent.
- **Sheather** covers lasso primarily through `glmnet` output in R, emphasizing the cross-validation curve and the resulting sparsity pattern (which coefficients survive at the chosen $\lambda$) over the coordinate-descent algebra worked through by hand above.

---

## 17.8 Interview Q&A

**Q: Why does lasso produce exact zeros while ridge doesn't?**
A: Geometrically, lasso's L1 constraint region is a diamond with corners on the coordinate axes, and optimal solutions tend to land at those corners (exact zeros) more often than at a generic point. Algebraically, the coordinate-descent update for lasso involves a $\max(\cdot,0)$ soft-thresholding step that can snap a coefficient to exactly zero; ridge's smooth quadratic penalty has no such discontinuity and only ever shrinks asymptotically toward, but never exactly to, zero.

**Q: Why doesn't lasso have a closed-form solution like ridge does?**
A: The L1 penalty $|\beta_j|$ isn't differentiable at $\beta_j=0$, so the standard matrix-calculus derivative-equals-zero approach used for OLS and ridge doesn't apply directly; lasso is instead fit iteratively via coordinate descent (or related algorithms like LARS).

**Q: If two predictors are highly correlated, how do ridge and lasso tend to behave differently?**
A: Ridge tends to shrink both coefficients together, keeping them similar in magnitude and both nonzero — spreading credit between them. Lasso tends to somewhat arbitrarily select one of the correlated predictors to keep (nonzero) and zero out the other(s) entirely, which can be less stable if the "choice" between near-equally-informative correlated predictors is sensitive to noise.

**Q: What does "soft-thresholding" mean, precisely?**
A: Given an unpenalized (OLS-like) coefficient value, subtract a fixed threshold amount toward zero; if that would cross zero, set the coefficient to exactly zero instead, rather than letting it flip sign. It's the core building block of the lasso coordinate-descent update.

**Q: When would you choose lasso over ridge, and vice versa?**
A: Lasso when you expect only a subset of predictors to be truly relevant and want built-in feature selection with an interpretable sparse result. Ridge when you expect most predictors to contribute something and primarily want to stabilize estimates under correlated predictors without dropping any entirely. Elastic Net (Chapter 18) is the common compromise when neither extreme fits cleanly.

---

*End of Chapter 17. Next: Chapter 18 — Elastic Net & Regularization Comparison (combining L1 and L2 penalties, the mixing parameter $\alpha$, and a side-by-side comparison of when each of the three regularization methods is the right default choice).*
