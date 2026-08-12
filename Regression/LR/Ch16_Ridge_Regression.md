# Chapter 16 — Ridge Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — expanded with plain-language explanations, a simplified formula cheat-sheet, and a full end-to-end numerical walkthrough. Uses the centered version of Chapter 5's dataset ($x_1,x_2$ correlated with $VIF\approx9.33$ from Chapter 9) to show ridge regression in action on exactly the instability that chapter diagnosed.*

---

## 16.1 The Motivating Question — Picking Up Chapter 6's Closing Thread

Chapter 6 proved OLS is BLUE — minimum-variance **among unbiased estimators** — and flagged that a **biased** estimator could still win on total error (bias² + variance) if it traded a little bias for a large variance reduction. Chapter 9 showed exactly the scenario where this trade becomes attractive: under multicollinearity, $(\mathbf X^T\mathbf X)^{-1}$'s diagonal entries blow up ($VIF\approx9.33$ for both $x_1,x_2$), making coefficients wildly unstable.

**Ridge regression is the direct answer:** it deliberately introduces a small, controlled bias in exchange for a substantial variance reduction — by preventing $\mathbf X^T\mathbf X$ from ever being close to singular.

**One-sentence idea:** add a gentle "please don't get too extreme" leash on the coefficients, so the model isn't tempted toward wild, over-confident values just because two predictors are fighting over the same signal.

---

## 16.2 The Ridge Objective

$$ RSS_{ridge}(\boldsymbol\beta) = \underbrace{\sum_{i=1}^n(y_i-\mathbf x_i^T\boldsymbol\beta)^2}_{\text{fit the data (as before)}} + \underbrace{\lambda\sum_{j=1}^p\beta_j^2}_{\text{keep coefficients small}} = RSS(\boldsymbol\beta)+\lambda\|\boldsymbol\beta\|_2^2 $$

**Stripped-down reading:** ordinary regression cares about one thing (fit). Ridge cares about two things at once — fit **and** a "coefficients-stay-small" penalty — with $\lambda\geq0$ acting as a dial between them.

| $\lambda$ | Behavior |
|---|---|
| $\lambda=0$ | Exactly OLS — accuracy is the only concern |
| $\lambda\to\infty$ | Every slope squeezed toward 0 — an increasingly "safe," flat model |
| Intercept $\beta_0$ | **Never penalized** — center $y$ and every predictor first, so $\beta_0=\bar y$ is estimated separately, untouched by the penalty |

**Why the intercept gets a free pass:** it just represents "the average level of $y$" — there's no reason to punish a model for having an average score of 65 instead of 0. The penalty targets *extreme, over-confident slopes*, not the outcome's baseline level.

**Geometric picture:** minimizing $RSS_{ridge}$ is equivalent to minimizing plain $RSS$ subject to $\sum\beta_j^2\leq s$ — i.e., constraining $\boldsymbol\beta$ inside a **circular (L2) ball** around the origin. (Contrast Chapter 17's lasso, whose diamond-shaped L1 constraint tends to touch the fit contours exactly at a corner — setting some coefficients to exactly zero, which a circle essentially never does.)

**Plain-language geometry:** picture "best possible fit" (OLS) as a point on a map, surrounded by contour rings of "almost as good," "still pretty good," and so on. Ridge draws a circle around the origin and says "you can't leave this circle." The ridge answer is wherever the smallest ring first touches that circle — usually pulling *every* coefficient in a bit, since a circle restricts every direction equally.

---

## 16.3 Deriving the Closed-Form Solution

Same matrix-calculus approach as Chapter 3, §3.4:

$$ RSS_{ridge}(\boldsymbol\beta) = (\mathbf y-\mathbf X\boldsymbol\beta)^T(\mathbf y-\mathbf X\boldsymbol\beta)+\lambda\boldsymbol\beta^T\boldsymbol\beta $$

Differentiate and set to zero:

$$ -2\mathbf X^T\mathbf y + 2\mathbf X^T\mathbf X\boldsymbol\beta+2\lambda\boldsymbol\beta=0 \quad\Rightarrow\quad (\mathbf X^T\mathbf X+\lambda\mathbf I)\boldsymbol\beta=\mathbf X^T\mathbf y $$

$$ \boxed{\hat{\boldsymbol\beta}_{ridge}=(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T\mathbf y} $$

**The single most important fact in the chapter:** adding $\lambda\mathbf I$ before inverting is *exactly* what fixes Chapter 9's near-singularity problem — even if $\mathbf X^T\mathbf X$'s smallest eigenvalue is near 0, adding any $\lambda>0$ pushes **every** eigenvalue up by $\lambda$, guaranteeing $(\mathbf X^T\mathbf X+\lambda\mathbf I)$ is comfortably invertible. Ridge is, quite literally, a numerically stabilized version of the normal equations from Chapter 3.

**Plain words:** Chapter 9's trouble came specifically from $\mathbf X^T\mathbf X$'s eigenvalues getting dangerously close to zero. Ridge's fix is one tiny algebraic move — add a small positive number to the diagonal before inverting — and every eigenvalue, including the smallest, gets nudged safely away from zero. A small change to the formula buying a large amount of stability.

---

## 16.4 Worked Example — Ridge on the Multicollinear Dataset from Chapter 9

Center all variables ($x_1$ around $\bar x_1=3$, $x_2$ around $\bar x_2=1.8$, $y$ around $\bar y=64.6$):

$$ \mathbf X_c^T\mathbf X_c = \begin{bmatrix}S_{x_1x_1}&S_{x_1x_2}\\S_{x_1x_2}&S_{x_2x_2}\end{bmatrix}=\begin{bmatrix}10&5\\5&2.8\end{bmatrix},\qquad \mathbf X_c^T\mathbf y_c=\begin{bmatrix}81\\42.6\end{bmatrix} $$

For a $2\times2$ system $\begin{bmatrix}a&b\\b&d\end{bmatrix}\boldsymbol\beta=\begin{bmatrix}p\\q\end{bmatrix}$, the closed form is simple enough to hand-compute at every $\lambda$:

$$ \hat\beta_1=\frac{dp-bq}{ad-b^2},\qquad \hat\beta_2=\frac{-bp+aq}{ad-b^2},\qquad\text{with } a=10+\lambda,\ d=2.8+\lambda,\ b=5,\ p=81,\ q=42.6 $$

**Check at $\lambda=0$** (must recover Chapter 5's OLS slopes): $\det=10(2.8)-25=3\Rightarrow\hat\beta_1=4.60,\ \hat\beta_2=7.00$. ✅ Exact match.

### Full ridge trace — every quantity, across a grid of $\lambda$

Solving the same $2\times2$ system at eight values of $\lambda$, then computing training predictions $\hat y_{c,i}=\hat\beta_1 x_{1,i,c}+\hat\beta_2 x_{2,i,c}$ and training $SSE=\sum(y_{c,i}-\hat y_{c,i})^2$ at each:

| $\lambda$ | $\hat\beta_1$ | $\hat\beta_2$ | $\lVert\hat{\boldsymbol\beta}\rVert_2$ | Training SSE | Training $R^2$ |
|---|---|---|---|---|---|
| 0 (OLS) | 4.600 | 7.000 | 8.38 | 2.40 | 0.9964 |
| 0.5 | 5.627 | 4.383 | 7.13 | 5.25 | 0.9922 |
| 1 | 5.643 | 3.786 | 6.80 | 8.68 | 0.9871 |
| 2 | 5.393 | 3.258 | 6.30 | 18.22 | 0.9729 |
| 5 | 4.552 | 2.543 | 5.21 | 60.18 | 0.9106 |
| 10 | 3.567 | 1.935 | 4.06 | 137.24 | 0.7961 |
| 20 | 2.479 | 1.325 | 2.81 | 257.93 | 0.6168 |
| 50 | 1.293 | 0.684 | 1.46 | 432.34 | 0.3577 |

($R^2$ computed against $SST_c=\sum y_{c,i}^2=673.2$; as $\lambda\to\infty$, predictions shrink toward 0 and $SSE\to SST_c$, i.e. $R^2\to0$ — the "intercept-only" limit.)

**Two things worth noticing, made numerically explicit:**

1. **Coefficient norm shrinks monotonically** (8.38 → 1.46) as designed. But the shrinkage is **not symmetric**: $\hat\beta_2$ (the individually *insignificant* coefficient under OLS, precisely because of multicollinearity) collapses fast (7.00 → 0.68), while $\hat\beta_1$ shrinks far more gently (4.60 → 1.29) and even ticks up slightly at $\lambda\approx0.5$–$1$ before declining. Ridge isn't punishing every coefficient equally — it punishes the *unstable* one hardest, because its large OLS value was mostly an artifact of the multicollinearity, not well-supported signal.
2. **Training SSE rises monotonically and steeply** — this is the price of the bias, made completely concrete: fit degrades from $R^2=0.996$ at $\lambda=0$ to $R^2=0.358$ by $\lambda=50$. This table is also the answer to a common question in §16.6 below: *you cannot pick $\lambda$ by minimizing training error*, because training error is minimized, by construction, at $\lambda=0$ every single time.

---

## 16.5 The Bias-Variance Tradeoff, Made Formal — Then Verified Numerically

$$ E[\hat{\boldsymbol\beta}_{ridge}] = (\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T\mathbf X\,\boldsymbol\beta \neq \boldsymbol\beta \quad\text{for }\lambda>0 $$

Ridge is **biased** whenever $\lambda>0$ — a deliberate departure from Gauss-Markov (Chapter 6). In exchange:

$$ \text{Var}(\hat{\boldsymbol\beta}_{ridge}) = \sigma^2\underbrace{(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}}_{A}\ \mathbf X^T\mathbf X\ \underbrace{(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}}_{A} $$

**Stripped-down reading (the "sandwich"):** shrink ($A$) → rescale by the original information ($\mathbf X^T\mathbf X$) → shrink again ($A$). This is **provably smaller** (in the same positive-semi-definite sense as Chapter 6's OLS-variance proof) than OLS's $\sigma^2(\mathbf X^T\mathbf X)^{-1}$, for any $\lambda>0$.

### Numerically verifying the variance reduction — no assumptions needed

Using $\sigma^2\approx s^2=1.2$ (Chapter 9, §9.6) and the actual matrices above, at $\lambda=1$:

$$ A=(\mathbf X_c^T\mathbf X_c+\mathbf I)^{-1}=\frac{1}{16.8}\begin{bmatrix}3.8&-5\\-5&11\end{bmatrix} $$

Carrying out the sandwich product $\sigma^2\,A\,(\mathbf X_c^T\mathbf X_c)\,A$ gives:

| | OLS variance (Chapter 9) | Ridge variance ($\lambda=1$) | Ridge variance ($\lambda=5$) |
|---|---|---|---|
| $\text{Var}(\hat\beta_1)$ | 1.120 | 0.104 | 0.041 |
| $\text{Var}(\hat\beta_2)$ | 4.000 | 0.165 | 0.018 |
| $SE(\hat\beta_1)$ | 1.058 | 0.322 | 0.202 |
| $SE(\hat\beta_2)$ | 2.000 | 0.406 | 0.136 |

**This part is unambiguous fact, not approximation:** $\hat\beta_2$'s variance — the coefficient multicollinearity hit hardest — drops from 4.0 to 0.165 at $\lambda=1$ (a ~24× reduction) and to 0.018 at $\lambda=5$ (a ~217× reduction). This is the real, computable payoff ridge is built to deliver, and it requires nothing except the formula and the data — no knowledge of the true $\boldsymbol\beta$.

### An honest, caveated look at bias and total MSE

The *bias* side is harder to verify numerically, because $\text{Bias}=E[\hat{\boldsymbol\beta}_{ridge}]-\boldsymbol\beta$ requires knowing the **true** $\boldsymbol\beta$ — which, in any real dataset, we don't. As a common **illustrative-only** exercise, textbooks sometimes plug in the OLS estimate as a stand-in for "truth" ($\hat{\boldsymbol\beta}_{OLS}=(4.6,7.0)$) via $\text{Bias}\approx(A\,\mathbf X_c^T\mathbf X_c-\mathbf I)\,\hat{\boldsymbol\beta}_{OLS}$:

| | $\lambda=1$ | $\lambda=5$ |
|---|---|---|
| $\text{Bias}(\hat\beta_1)$ | +1.05 | −0.05 |
| $\text{Bias}(\hat\beta_2)$ | −3.21 | −4.46 |
| $\text{Bias}^2(\hat\beta_1)$ | 1.09 | 0.002 |
| $\text{Bias}^2(\hat\beta_2)$ | 10.33 | 19.86 |
| Implied $MSE(\hat\beta_1)=\text{Bias}^2+\text{Var}$ | 1.20 | 0.043 |
| Implied $MSE(\hat\beta_2)=\text{Bias}^2+\text{Var}$ | 10.49 | 19.88 |
| vs. OLS $MSE=\text{Var}$ only (unbiased) | $\beta_1$: 1.12 · $\beta_2$: 4.00 | same |

**Read this table carefully, because it teaches a genuinely important, easy-to-miss point.** By this plug-in calculation, ridge looks like a clear *win* for $\hat\beta_1$ (implied MSE drops well below OLS's) but a *loss* for $\hat\beta_2$ alone (implied MSE rises well above OLS's) at both $\lambda$ values shown. That is not a contradiction of ridge theory — it's a limitation of the plug-in method itself: using the noisy OLS estimate (recall $SE(\hat\beta_{2,OLS})=2.0$ — itself extremely uncertain) as if it were the fixed, known truth is circular, and can make ridge look worse than it is for exactly the coefficient whose OLS estimate was least trustworthy to begin with.

**What the rigorous theory (Hoerl–Kennard, 1970) actually guarantees** is narrower and more careful than the naive plug-in check: for any true $\boldsymbol\beta$ and true $\sigma^2$, there exists **some** $\lambda>0$ such that the *total* expected squared error, summed across all coefficients, $E\big[\lVert\hat{\boldsymbol\beta}_{ridge}-\boldsymbol\beta\rVert^2\big]$, is strictly less than OLS's. This is a statement about expectations over repeated sampling from the *true* generating process — not something you can confirm by plugging a single noisy estimate in for $\boldsymbol\beta$ and checking one sample's arithmetic. The variance table above is the part of the story you *can* verify directly from data; the bias/total-MSE comparison is conceptually correct in theory but not literally checkable from one dataset, and the table here is included specifically to make that distinction (verifiable vs. illustrative-only) concrete rather than to "prove" ridge won this particular case.

**Plain-English summary of the whole trade:** across many repeated samples, ridge's coefficients land slightly off from the true values on average — the price of bias. But each individual estimate is far less erratic — the variance payoff, and that part is not in question here; it's directly computed above. Whether the trade is *worth it* for any one coefficient in any one dataset is a subtler question than "is ridge biased" (yes, always, for $\lambda>0$) — and the honest answer requires either knowing the true $\boldsymbol\beta$ (which we never do in practice) or evaluating *predictive* performance via cross-validation instead, which is exactly what §16.6 does.

---

## 16.6 Choosing $\lambda$

$\lambda$ is a **hyperparameter**, not estimated by the normal-equations machinery — it's chosen via **cross-validation** (Chapter 15): fit ridge across a grid of candidate $\lambda$ values, compute k-fold (or LOOCV) *test* error for each, and pick the $\lambda$ minimizing estimated out-of-sample error.

**Why not just pick the $\lambda$ that minimizes training error?** §16.4's ridge-trace table already answers this directly: training SSE rose monotonically at *every* step (2.40 → 5.25 → 8.68 → … → 432.34) as $\lambda$ increased. Minimizing training error will therefore **always** select $\lambda=0$ (plain OLS) — that's mathematically guaranteed, not a coincidence of this dataset, because training error is exactly what OLS is built to minimize. The entire point of cross-validation is to instead measure performance on data the model *didn't* see, which is the only way a penalty term can ever look like a good idea numerically.

**In plain words:** there's no formula that hands you the "correct" $\lambda$ directly. Try a range of candidates (0.1, 0.5, 1, 5, 10, …), and for each, check how well that version predicts data it wasn't trained on (the cross-validation trick from Chapter 15). Whichever $\lambda$ predicts best out-of-sample is the one you keep.

---

## 16.7 Where the Textbooks Differ

| Source | Distinctive contribution |
|---|---|
| **Kutner** | Brief — mostly ties ridge to the multicollinearity diagnostics from its VIF/condition-number chapter; practical rather than theoretical. |
| **Montgomery** | Ridge trace plots (coefficients vs. $\lambda$, as in §16.4's table) as a practical, visual tool for picking a "stable-looking" $\lambda$ — historically predating cross-validation as the standard selection method. |
| **ESL/ISL** | Fullest theoretical treatment — the bias-variance derivation, the connection to principal components (ridge shrinks more aggressively along low-variance PC directions of $\mathbf X$), and the L2-ball geometric picture. |
| **Sheather** | Primarily software-driven (`glmnet` in R), emphasizing the cross-validation curve for choosing $\lambda$ over the closed-form algebra. |

---

## 16.8 Formula Cheat-Sheet

| Quantity | Formula | Plain-English reading |
|---|---|---|
| Ridge objective | $RSS(\boldsymbol\beta)+\lambda\lVert\boldsymbol\beta\rVert_2^2$ | fit error + "stay small" penalty |
| Closed-form solution | $\hat{\boldsymbol\beta}_{ridge}=(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T\mathbf y$ | add $\lambda$ to the diagonal before inverting — guarantees invertibility |
| Bias (conceptual) | $E[\hat{\boldsymbol\beta}_{ridge}]-\boldsymbol\beta\neq0$ for $\lambda>0$ | always biased when the penalty is on |
| Variance (the sandwich) | $\sigma^2 A\,\mathbf X^T\mathbf X\,A,\ \ A=(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}$ | shrink → rescale → shrink again; provably $\leq$ OLS variance |
| Choosing $\lambda$ | grid search + cross-validation | never minimize training error — that always picks $\lambda=0$ |

---

## 16.9 Interview Q&A

**Q: Write the closed-form ridge regression estimator and explain why it's always well-defined, even under severe multicollinearity.**
A: $\hat{\boldsymbol\beta}_{ridge}=(\mathbf X^T\mathbf X+\lambda\mathbf I)^{-1}\mathbf X^T\mathbf y$. Adding $\lambda\mathbf I$ ($\lambda>0$) shifts every eigenvalue of $\mathbf X^T\mathbf X$ up by $\lambda$, guaranteeing the matrix being inverted is nonsingular even if $\mathbf X^T\mathbf X$ itself is exactly or nearly singular.
*(Simple version: adding a small positive number to the diagonal before inverting nudges every "danger zone" eigenvalue safely away from zero.)*

**Q: Is ridge regression's estimator biased? Why would you ever want a biased estimator?**
A: Yes, for any $\lambda>0$ — directly contradicts Gauss-Markov's unbiasedness requirement. Worth using because the variance reduction can be large enough, especially under multicollinearity, to reduce *total* expected prediction error below unbiased OLS's. As computed in §16.5, the variance reduction is directly verifiable from data (e.g., $\hat\beta_2$'s variance drops ~24× at $\lambda=1$); the bias side is real but only checkable via cross-validated predictive performance, not by plugging a noisy point estimate in for the unknown truth.
*(Simple version: a reliably-slightly-off estimate often beats an unreliable, occasionally-wildly-wrong one.)*

**Q: Why must predictors typically be standardized before applying ridge regression?**
A: The penalty $\lambda\sum\beta_j^2$ treats every coefficient identically regardless of scale — an unstandardized predictor with naturally larger raw values would have its coefficient penalized unfairly relative to one on a smaller scale.
*(Simple version: comparing a dollars-scale coefficient to an inches-scale coefficient without standardizing first is apples to oranges.)*

**Q: How is $\lambda$ chosen in practice, and why can't you just pick whatever minimizes training error?**
A: Via cross-validation — evaluate out-of-sample error across a grid of $\lambda$ and pick the minimizer. Training error alone can't be used because it decreases monotonically as $\lambda\to0$ by construction (verified numerically in §16.4's table) — minimizing it always just recovers plain OLS, defeating the purpose of the penalty entirely.
*(Simple version: try a bunch of candidate values, keep whichever one predicts *new* data the best — not whichever fits the training data hardest.)*

**Q: What happens to ridge coefficients as $\lambda\to\infty$? As $\lambda\to0$?**
A: As $\lambda\to0$, ridge reduces exactly to OLS. As $\lambda\to\infty$, every slope is driven toward zero (never exactly zero, unlike lasso — Chapter 17), converging to an intercept-only model; correspondingly, training $R^2\to0$ (shown numerically in §16.4).
*(Simple version: dial the penalty to zero and you're back to OLS; dial it to the max and every slope gets squeezed toward — but never quite reaches — zero.)*

**Q: If ridge reduces variance, does that mean every individual coefficient's mean-squared error necessarily improves?**
A: Not necessarily, and not verifiably from a single dataset. What's rigorously guaranteed (Hoerl–Kennard) is that some $\lambda>0$ reduces the *total* squared error summed across coefficients, in expectation over the true data-generating process. Checking any one coefficient's MSE by plugging the OLS estimate in as "truth" is circular — as §16.5's worked example shows, that check can even suggest ridge made a particular coefficient worse, purely because the OLS estimate used as the truth proxy was itself noisy.

---

*End of Chapter 16. Next: Chapter 17 — Lasso Regression (the L1 penalty, why its diamond-shaped constraint region produces exact zeros/sparsity where ridge's circular region doesn't, and coordinate descent as the standard fitting algorithm).*
