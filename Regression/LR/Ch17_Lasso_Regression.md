# Chapter 17 — Lasso Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations, a simplified formula cheat-sheet, and a full numerical lasso path. Continues Chapter 16's centered dataset ($x_1,x_2,y$; $S_{x_1x_1}=10,\ S_{x_2x_2}=2.8,\ S_{x_1x_2}=5,\ S_{x_1y}=81,\ S_{x_2y}=42.6$) so the contrast between ridge and lasso is directly numerical, not just conceptual.*

---

## 17.1 The Motivating Question — What Ridge Doesn't Do

Chapter 16 shrank both $\hat\beta_1,\hat\beta_2$ as $\lambda$ grew — but at **no** value of $\lambda$ did $\hat\beta_2$ reach **exactly** zero; it shrank asymptotically without ever dropping out. Sometimes you want more than shrinkage — you want the model to **actively select** which predictors matter and discard the rest, producing a **sparse** model. **Lasso** (Least Absolute Shrinkage and Selection Operator) does exactly this, via a different penalty shape.

---

## 17.2 The Lasso Objective — L1 Instead of L2

$$ RSS_{lasso}(\boldsymbol\beta) = \sum_{i=1}^n(y_i-\mathbf x_i^T\boldsymbol\beta)^2 + \lambda\sum_{j=1}^p|\beta_j| $$

**The only change from ridge:** $\sum\beta_j^2$ (L2 norm) becomes $\sum|\beta_j|$ (L1 norm) — one swapped exponent, with an outsized geometric consequence.

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Constraint region | Circle | Diamond |
| Corners on the axes? | No | Yes |
| Coefficients hit exactly zero? | Never (asymptotic only) | Yes, at a finite $\lambda$ |

**Plain-language geometry:** picture the same elliptical "fit quality" contours from Chapter 16. A circle has no corners, so there's no geometric reason for an ellipse to touch it exactly on an axis. A diamond *does* have corners sitting exactly on the axes — and it's a general fact of constrained optimization that solutions disproportionately land at corners. Landing at a corner means one or more coordinates are **exactly** zero. That's the entire mechanism, stated geometrically, for why lasso selects variables and ridge doesn't.

---

## 17.3 Why There's No Closed-Form Solution — And the Fix: Coordinate Descent

$|\beta_j|$ isn't differentiable at $\beta_j=0$ (its slope jumps from $-1$ to $+1$ there), so the clean derivative-equals-zero trick from Chapter 16, §16.3 doesn't work. Lasso is instead fit by **coordinate descent**: optimize one coefficient at a time, holding the others fixed, cycling until nothing changes.

### The building block — soft-thresholding, simplified

Holding $\beta_2$ fixed, define the partial residual's implied OLS coefficient on $x_1$:

$$ z_1 = \frac{S_{x_1y}-\beta_2 S_{x_1x_2}}{S_{x_1x_1}} \qquad\text{("what }\beta_1\text{ would be by itself, given }\beta_2\text{'s current value")} $$

The coordinate-descent update is then:

$$ \hat\beta_1 \leftarrow \text{sign}(z_1)\cdot\max\!\left(|z_1|-\underbrace{\frac{\lambda}{2S_{x_1x_1}}}_{\text{threshold}},\ 0\right) $$

**Stripped-down reading:** take the "as-if-alone" OLS-style value $z_1$, shrink it toward zero by a fixed amount, and **if that shrinkage would push it past zero, snap it to exactly zero instead of letting it flip sign.** That snap-to-zero (the $\max(\cdot,0)$) is the entire sparsity mechanism — ridge's smooth quadratic penalty has no equivalent step.

---

## 17.4 Worked Example — Coordinate Descent Actually Converging, $\lambda=5$

The original textbook example used $\lambda=10$, which turns out to zero out $\beta_2$ on the very first pass — a bit too fast to show the algorithm actually *iterating*. Using $\lambda=5$ instead (still below the sparsity threshold derived in §17.5) shows real iterative convergence to a **nonzero** fixed point.

Initialize $\beta_1=\beta_2=0$. Thresholds: $\lambda/(2S_{x_1x_1})=5/20=0.25$ and $\lambda/(2S_{x_2x_2})=5/5.6=0.893$.

| Pass | Update $\beta_1$ (holding $\beta_2$) | Update $\beta_2$ (holding $\beta_1$) |
|---|---|---|
| 1 | $z_1=81/10=8.10\Rightarrow\beta_1=8.10-0.25=7.850$ | $z_2=(42.6-7.85\cdot5)/2.8=3.35/2.8=1.196\Rightarrow\beta_2=1.196-0.893=0.304$ |
| 2 | $z_1=(81-0.304\cdot5)/10=7.948\Rightarrow\beta_1=7.948-0.25=7.698$ | $z_2=(42.6-7.698\cdot5)/2.8=4.109/2.8=1.467\Rightarrow\beta_2=1.467-0.893=0.574$ |
| 3 | $z_1=(81-0.574\cdot5)/10=7.813\Rightarrow\beta_1=7.813-0.25=7.563$ | $z_2=(42.6-7.563\cdot5)/2.8=4.782/2.8=1.708\Rightarrow\beta_2=1.708-0.893=0.815$ |
| $\vdots$ | converging... | converging... |
| **Fixed point** | $\hat\beta_1=6.433$ | $\hat\beta_2=2.833$ |

**Solving directly for the fixed point** (both coefficients positive throughout, so both stationarity conditions apply as equalities) gives a shortcut: at the fixed point, the system is exactly the ordinary normal equations with **both right-hand sides reduced by $\lambda/2$**:

$$ \begin{bmatrix}10&5\\5&2.8\end{bmatrix}\begin{bmatrix}\beta_1\\\beta_2\end{bmatrix} = \begin{bmatrix}81-\lambda/2\\42.6-\lambda/2\end{bmatrix} $$

Solving this $2\times2$ system (same $\det=3$ as always) at $\lambda=5$ ($81-2.5=78.5,\ 42.6-2.5=40.1$) gives $\hat\beta_1=6.433,\ \hat\beta_2=2.833$ directly — matching where the iteration above is heading, without needing to iterate by hand indefinitely. **Both coefficients survive at $\lambda=5$** — confirming this $\lambda$ sits below the sparsity threshold, worked out exactly in §17.5.

---

## 17.5 Finding the Critical $\lambda$ — Where Sparsity Kicks In

**A clean shortcut, generalized from §17.4:** as long as *both* coefficients stay positive (true throughout this dataset), the "both-active" solution is exactly the OLS system with each $y$-cross-product reduced by $\lambda/2$. Solving that $2\times2$ system symbolically as a function of $\lambda$ gives closed-form straight lines:

$$ \hat\beta_1(\lambda) = 4.6+\frac{11}{30}\lambda \approx 4.6+0.3667\lambda \qquad\qquad \hat\beta_2(\lambda) = 7-\frac{5}{6}\lambda \approx 7-0.8333\lambda $$

(Check at $\lambda=0$: $\hat\beta_1=4.6,\ \hat\beta_2=7$ — exact OLS match. Check at $\lambda=5$: $\hat\beta_1=6.433,\ \hat\beta_2=2.833$ — exact match to §17.4.)

**This "both-active" line for $\beta_2$ is only valid until it hits zero.** Setting $\hat\beta_2(\lambda)=0$:

$$ 7-\frac{5}{6}\lambda=0 \quad\Rightarrow\quad \lambda = 8.4 $$

**This is the critical $\lambda$:** below 8.4, both predictors survive; at or above 8.4, $x_2$ drops out entirely. At the boundary itself, both formulas agree exactly — $\hat\beta_1(8.4)=4.6+0.3667(8.4)=7.68$, which also matches the single-variable formula below, confirming a smooth (continuous, just non-differentiable) handoff between regimes.

**Beyond $\lambda=8.4$**, with $\beta_2$ permanently pinned at 0, only $\beta_1$'s own soft-threshold update matters ($z_1=8.1$ always, since $\beta_2=0$ removes its contribution):

$$ \hat\beta_1(\lambda) = 8.1-\frac{\lambda}{20} \qquad\text{valid for } 8.4\leq\lambda\leq162 $$

This itself hits zero at $\lambda=8.1\times20=162$ — beyond which **both** coefficients are zero (the null, intercept-only model).

### The full lasso path, computed end to end

| $\lambda$ | Regime | $\hat\beta_1$ | $\hat\beta_2$ | $\lVert\hat{\boldsymbol\beta}\rVert_1$ | Training SSE | Training $R^2$ |
|---|---|---|---|---|---|---|
| 0 (OLS) | both active | 4.600 | 7.000 | 11.600 | 2.40 | 0.9964 |
| 2 | both active | 5.333 | 5.333 | 10.667 | — | — |
| 4 | both active | 6.067 | 3.667 | 9.733 | — | — |
| 5 | both active | 6.433 | 2.833 | 9.267 | 8.24 | 0.9878 |
| 6 | both active | 6.800 | 2.000 | 8.800 | — | — |
| 8 | both active | 7.533 | 0.333 | 7.867 | — | — |
| **8.4** | **boundary — $x_2$ drops** | 7.680 | **0.000** | 7.680 | 18.86 | 0.9720 |
| 10 | $x_2$ inactive | 7.600 | 0.000 | 7.600 | 19.60 | 0.9709 |
| 20 | $x_2$ inactive | 7.100 | 0.000 | 7.100 | — | — |
| 50 | $x_2$ inactive | 5.600 | 0.000 | 5.600 | 79.60 | 0.8818 |
| 100 | $x_2$ inactive | 3.100 | 0.000 | 3.100 | 267.10 | 0.6032 |
| **162** | **both zero (null model)** | 0.000 | 0.000 | 0.000 | 673.20 | 0.0000 |

**Reading the path in plain words:** for small $\lambda$, both predictors share the burden and both shrink together (much like ridge). The instant $\lambda$ crosses **8.4**, $x_2$ is switched off entirely — not gradually, but as a genuine on/off event — and from there only $\hat\beta_1$ continues shrinking, linearly, until it too hits zero at $\lambda=\mathbf{162}$, leaving nothing but the intercept. Unlike ridge's smooth, always-curving decay, **the lasso path is piecewise-linear with sharp kinks exactly at the values where a coefficient switches off** ($\lambda=8.4$) or the model collapses entirely ($\lambda=162$) — a direct numeric signature of the diamond-shaped constraint's corners from §17.2.

**Contrast with Chapter 16's ridge trace on the identical data:** at $\lambda=5$, ridge gave $(\hat\beta_1,\hat\beta_2)=(4.552,2.543)$ — both still clearly nonzero and both still shrinking gently — while lasso at that same $\lambda=5$ gives $(6.433, 2.833)$, with $\hat\beta_2$ already visibly on its way toward the hard zero it will hit at $\lambda=8.4$. Same data, same $\lambda$, qualitatively different destinies for $\hat\beta_2$.

**A substantive connection worth stating explicitly:** the surviving coefficient at large $\lambda$, $\hat\beta_{1,lasso}\to7.6$ near the sparsity boundary, sits close to Chapter 5's simple-regression (reduced-model) slope of $8.1$ (§5.5) — sensible, since once $x_2$ is fully dropped, the remaining model is structurally similar to that earlier reduced model, with the small gap ($7.6$ vs. $8.1$ at $\lambda=10$) attributable to the residual shrinkage lasso still applies to $\beta_1$ itself even after $\beta_2$ is gone.

---

## 17.6 Choosing $\lambda$, and Ridge vs. Lasso in Practice

As with ridge, $\lambda$ is chosen via cross-validation (Chapter 15): fit across a grid, evaluate out-of-sample error, pick the minimizer. The same warning from Chapter 16, §16.6 applies identically here — training error (visible degrading steadily in the table above, from $R^2=0.996$ down to $0$) is minimized at $\lambda=0$ by construction, so it can never be the selection criterion.

**Practical rule of thumb:**

| Situation | Prefer |
|---|---|
| Only a subset of predictors truly matter; want automatic feature selection and an interpretable sparse result | **Lasso** |
| Most/all predictors contribute something; want to tame instability among correlated predictors without discarding any | **Ridge** |
| Want some of lasso's sparsity while keeping more of ridge's stability under correlation | **Elastic Net** (Chapter 18) |

**Why lasso can be less stable with correlated predictors, in plain words:** recall $x_1,x_2$ are correlated at $r\approx0.945$ (Chapter 9). Ridge spreads the burden between them roughly proportionally. Lasso, once past $\lambda=8.4$, picks $x_1$ to survive and zeros $x_2$ out completely — but that "choice" was driven by $x_1$ having a marginally larger $z$-value, not by $x_1$ being decisively more important. A slightly different sample could easily have flipped which of the two survived. Elastic Net (Chapter 18) exists specifically to soften this arbitrary-seeming winner-take-all behavior.

---

## 17.7 Where the Textbooks Differ

| Source | Distinctive contribution |
|---|---|
| **Kutner & Montgomery** | Both pre-date lasso's widespread classical-statistics adoption — light coverage, typically a brief modern addendum rather than a core chapter. |
| **ESL/ISL** | Fullest theoretical treatment — the geometric diamond-vs-circle argument in §17.2 is essentially their standard exposition, plus the connection to subset selection and LARS (Least Angle Regression) as an alternative fitting algorithm to coordinate descent. |
| **Sheather** | Primarily `glmnet` output in R — emphasizes the cross-validation curve and resulting sparsity pattern over the coordinate-descent algebra worked through by hand above. |

---

## 17.8 Formula Cheat-Sheet

| Quantity | Formula | Plain-English reading |
|---|---|---|
| Lasso objective | $RSS(\boldsymbol\beta)+\lambda\sum|\beta_j|$ | fit error + "stay small, on an absolute-value scale" penalty |
| No closed form because | $|\beta_j|$ not differentiable at 0 | the usual derivative trick breaks down at the one point that matters most |
| Coordinate-descent update | $\beta_j\leftarrow\text{sign}(z_j)\max(|z_j|-\text{threshold},0)$ | shrink the as-if-alone OLS value, snap to zero if shrinkage overshoots |
| "Both active" fixed point | ordinary normal equations, RHS reduced by $\lambda/2$ | same algebra as OLS, just with a flat penalty subtracted first |
| Critical $\lambda$ for a coefficient | where its "both-active" line hits zero | the exact point a predictor gets switched off |
| Path shape | piecewise-linear, kinks at each dropout | contrast with ridge's smooth, always-curving decay |

---

## 17.9 Interview Q&A

**Q: Why does lasso produce exact zeros while ridge doesn't?**
A: Geometrically, lasso's L1 constraint region is a diamond with corners on the coordinate axes, and solutions disproportionately land at those corners (exact zeros). Algebraically, the coordinate-descent update involves a $\max(\cdot,0)$ soft-thresholding step that can snap a coefficient to exactly zero; ridge's smooth quadratic penalty has no such discontinuity and only ever shrinks asymptotically toward, never exactly to, zero — visible directly by comparing the two traces on identical data (§17.5).

**Q: Why doesn't lasso have a closed-form solution like ridge does?**
A: $|\beta_j|$ isn't differentiable at $\beta_j=0$, so the standard derivative-equals-zero approach doesn't apply directly; lasso is fit iteratively via coordinate descent (or related algorithms like LARS). Away from zero, though, the "both-active" segments of the path *are* linear in $\lambda$ and solvable in closed form — it's specifically the kink points where a coefficient crosses zero that break the clean algebra, not the whole path.

**Q: If two predictors are highly correlated, how do ridge and lasso tend to behave differently?**
A: Ridge shrinks both together, keeping them similar in magnitude and both nonzero. Lasso tends to somewhat arbitrarily select one to keep and zero the other out entirely — in this chapter's data, $x_1$ survives and $x_2$ is switched off at $\lambda=8.4$, a choice driven by a small numerical edge rather than a decisive difference in importance.

**Q: What does "soft-thresholding" mean, precisely?**
A: Given an unpenalized (OLS-like) coefficient value, subtract a fixed threshold amount toward zero; if that would cross zero, set the coefficient to exactly zero instead of letting it flip sign. It's the core building block of the lasso coordinate-descent update.

**Q: Concretely, what does the "lasso path" look like as $\lambda$ increases, and how is that different from ridge's path?**
A: Piecewise-linear with sharp kinks, not a smooth curve. In this dataset: a straight-line segment where both coefficients shrink together (0 ≤ λ < 8.4), a kink where $x_2$ drops to exactly zero (λ = 8.4), a second straight-line segment where only $\hat\beta_1$ continues shrinking (8.4 ≤ λ < 162), and a final kink where the model collapses to intercept-only (λ = 162). Ridge, over the same range, never has a kink or an exact zero — every coefficient decays smoothly and asymptotically.

**Q: When would you choose lasso over ridge, and vice versa?**
A: Lasso when you expect only a subset of predictors to be truly relevant and want built-in feature selection with an interpretable sparse result. Ridge when you expect most predictors to contribute something and primarily want to stabilize estimates under correlated predictors without dropping any entirely. Elastic Net (Chapter 18) is the common compromise when neither extreme fits cleanly.

---

*End of Chapter 17. Next: Chapter 18 — Elastic Net & Regularization Comparison (combining L1 and L2 penalties, the mixing parameter $\alpha$, and a side-by-side comparison of when each of the three regularization methods is the right default choice).*
