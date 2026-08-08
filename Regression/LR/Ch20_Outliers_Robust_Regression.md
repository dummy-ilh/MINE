# Chapter 20 — Outliers & Robust Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Introduces a genuinely corrupted dataset — Chapter 1's clean data with one observation badly mis-recorded — since Chapter 8 only ever diagnosed influential points, without ever showing what happens when you actually fit through one, or how to fix it.*

**Corrupted dataset** — Chapter 1's original clean relationship ($x=1,2,3,4,5$; true $y=50,55,65,70,80$), but student 5's score is mis-recorded as **150** instead of 80 (imagine a data-entry error — an extra "1" typed in front):

| $x$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ | 50 | 55 | 65 | 70 | **150** |

---

## 20.1 The Motivating Question

Chapter 8 measured influence (Cook's distance, DFBETAS) as a **diagnostic** — a way to flag a point worth investigating. This chapter asks the natural next question: **once you know a point might be corrupted, or you simply want a fitting method less sensitive to such points in the first place, what alternative to OLS should you use?**

---

## 20.2 OLS's Sensitivity, Demonstrated Concretely

Fitting plain OLS to the corrupted data (same mechanics as Chapter 1): $\bar{x}=3$, $\bar{y}=78$, $S_{xx}=10$, $S_{xy}=215$:

$$ \hat{\beta}_1 = 215/10 = 21.5, \qquad \hat{\beta}_0 = 78-21.5(3) = 13.5 $$

**Compare to the true underlying relationship** (Chapter 1's clean data, slope $=7.5$): a **single** corrupted observation out of five has nearly **tripled** the estimated slope (7.5 → 21.5). Residuals under this fit: $15,\ -1.5,\ -13,\ -29.5,\ 29$ — every single point is now poorly fit, not just the corrupted one, because OLS's squared-error criterion lets one extreme point dominate the entire objective (recall: squaring a residual of 70+ contributes thousands to the RSS, dwarfing every other point's contribution).

---

## 20.3 M-Estimation and the Huber Loss

**The core idea:** replace OLS's squared-error loss with a loss function that behaves like squared error for small residuals (efficient, like OLS, when nothing unusual is happening) but transitions to something less explosive — like absolute-value loss — for large residuals, so no single point can dominate the objective.

**Huber loss:**

$$ L_\delta(e) = \begin{cases}\frac12e^2 & |e|\leq\delta \\ \delta\left(|e|-\frac12\delta\right) & |e|>\delta\end{cases} $$

**Plain-English reading:** within $\pm\delta$ of zero, residuals are treated exactly like OLS (squared) — full statistical efficiency for well-behaved data. Beyond $\pm\delta$, the loss grows only *linearly*, not quadratically — a residual of 70 no longer contributes catastrophically more than a residual of 30, capping any single point's leverage over the fit.

**Fitting via Iteratively Reweighted Least Squares (IRLS):** since Huber loss isn't a simple sum of squares, it's fit by repeatedly (1) computing weights based on current residuals, (2) refitting a **weighted** least squares model (Chapter 10's WLS machinery, directly reused) with those weights, and (3) recomputing residuals and weights, until convergence. The weight function:

$$ w(e) = \begin{cases}1 & |e|\leq\delta \\ \delta/|e| & |e|>\delta\end{cases} $$

**Worked example — one IRLS iteration**, using $\delta=10$ and OLS's residuals from §20.2 ($e=15,-1.5,-13,-29.5,29$):

$$ w_1=10/15\approx0.667,\ \ w_2=1,\ \ w_3=10/13\approx0.769,\ \ w_4=10/29.5\approx0.339,\ \ w_5=10/29\approx0.345 $$

Refitting via the same weighted-mean formulas from Chapter 10, §10.4 gives approximately $\hat{\beta}_1\approx19.6$, $\hat{\beta}_0\approx17.9$ after this first iteration — pulled somewhat back toward the true slope of 7.5, but **still substantially distorted.**

**An important, easy-to-miss limitation, worth stating explicitly:** Huber's weighting is based purely on the **size of the residual**, not on the point's **leverage** (Chapter 8). Student 5 sits at $x=5$ — already the highest-leverage predictor value in the dataset even before considering its corrupted $y$ — so it's what's called a **bad leverage point** (unusual in both $x$ and $y$ simultaneously). Down-weighting based on residual size alone only partially tames its pull, because its extreme $x$-position still gives it outsized influence on the fitted slope even at reduced weight. **M-estimators like Huber robustly handle vertical outliers (unusual $y$, typical $x$) well, but only partially handle bad leverage points** — this distinction is a frequently tested interview nuance.

---

## 20.4 RANSAC — A Fundamentally Different Strategy

**RANSAC (RANdom SAmple Consensus)** takes a completely different approach: rather than down-weighting outliers gradually, it repeatedly fits a model to small **random minimal subsets** of the data, and keeps whichever fit is agreed upon (has the most "inliers," points close to the fitted line) by the largest number of points overall.

**Algorithm:**
1. Randomly sample the minimum number of points needed to fit the model (2, for simple linear regression).
2. Fit the line through exactly those points.
3. Count **inliers** — how many of the *remaining* points fall within some threshold of this line.
4. Repeat many times; keep the fit with the most inliers.
5. **Refit** using ordinary OLS on just the inlier set from the winning iteration (a final polish step).

**Worked example — exhaustively enumerating candidate pairs** (feasible since $n=5$ gives only $\binom{5}{2}=10}$ possible pairs).

Try the pair $(1,50),(3,65)$: slope $=\frac{65-50}{3-1}=7.5$, intercept $=50-7.5(1)=42.5$. Line: $y=42.5+7.5x$.

Checking all 5 points against this line (threshold: residual $\leq5$): $x=1$: predicted 50, residual 0. $x=2$: predicted 57.5, residual $-2.5$. $x=3$: predicted 65, residual 0. $x=4$: predicted 72.5, residual $-2.5$. $x=5$: predicted 80, residual **70** — clearly outside the threshold.

**Result: 4 inliers (points 1–4), student 5 correctly excluded as an outlier.** Trying other pairs that exclude student 5 (e.g., $(2,55)$–$(4,70)$) consistently produces the same result: 4 inliers, student 5 excluded — because the 4 clean points genuinely agree with each other, while student 5 agrees with nothing.

**Final refit** (OLS on just the 4 inlier points, 1–4): $\bar{x}=2.5,\bar{y}=60,S_{xx}=5,S_{xy}=35$:

$$ \hat{\beta}_{1,RANSAC} = 35/5 = 7, \qquad \hat{\beta}_{0,RANSAC} = 60-7(2.5) = 42.5 $$

**This is extremely close to the true underlying slope of 7.5** (Chapter 1) — RANSAC has essentially fully recovered the clean relationship, completely discarding the corrupted point rather than merely down-weighting it.

---

## 20.5 Comparing All Three Approaches

| Method | $\hat{\beta}_1$ | How it handled the bad leverage point |
|---|---|---|
| True clean-data relationship (Chapter 1) | 7.5 | — |
| OLS (on corrupted data) | 21.5 | Fully dominated by the outlier |
| Huber M-estimator (1 IRLS iteration) | ≈19.6 | Partially down-weighted, but still leverage-vulnerable |
| RANSAC | 7.0 | Cleanly identified and excluded as a non-consensus point |

**The clear takeaway:** for this specific kind of problem — a single, severely bad leverage point — RANSAC's hard exclusion outperforms Huber's soft down-weighting, precisely because Huber's weighting scheme doesn't account for leverage at all. **In an interview, the right general answer is context-dependent:** Huber-type M-estimators are computationally cheaper, differentiable, and well-suited to data with many moderate vertical outliers; RANSAC is better suited to data with a smaller number of severe, structurally different bad points (as in computer vision applications, its original and still most common use case — e.g., fitting a line/plane to a point cloud contaminated with a distinct wrong-surface cluster).

---

## 20.6 A Brief Note on Bounded-Influence and Redescending M-Estimators

For completeness: some M-estimator variants specifically address the leverage limitation from §20.3 by incorporating leverage directly into the weighting scheme (**bounded-influence** or **GM-estimators**), or by using a **redescending** loss function (like Tukey's biweight) whose weight function actually goes to **zero** for sufficiently extreme residuals — effectively giving severely bad points zero influence, similar in spirit to RANSAC's hard exclusion, but arrived at through a smooth, differentiable weighting function rather than random sampling.

---

## 20.7 Where the Textbooks Differ

- **Kutner and Montgomery** cover Huber-type M-estimation as the primary robust-regression technique, consistent with their classical-statistics orientation, with comparatively little (if any) coverage of RANSAC.
- **Sheather** gives the most balanced treatment of both approaches, including practical guidance on choosing $\delta$ (often via a robust scale estimate like the Median Absolute Deviation, MAD, rather than an arbitrary round number as used for simplicity above).
- **ESL/ISL** discuss robust loss functions mainly in the broader context of loss-function choice for supervised learning generally (Huber loss appears prominently in gradient boosting, for instance), rather than as a dedicated "outliers in linear regression" topic — RANSAC, being a computer-vision-and-robotics-rooted algorithm, is typically outside their scope entirely.

---

## 20.8 Interview Q&A

**Q: Why does a single severe outlier distort an OLS fit so dramatically?**
A: OLS minimizes squared error, and squaring a large residual contributes disproportionately more to the objective than several small residuals combined — a single extreme point can dominate the entire fitting criterion.

**Q: What's the key limitation of Huber's M-estimator that RANSAC doesn't share?**
A: Huber's weighting is based purely on residual size, not on a point's leverage — a "bad leverage point" (unusual in both $x$ and $y$) retains outsized influence on the fitted slope even after down-weighting, since its extreme predictor value alone still pulls the line. RANSAC, by fitting on minimal random subsets and checking consensus, can fully exclude such points regardless of their leverage.

**Q: When would you choose RANSAC over a Huber M-estimator, or vice versa?**
A: RANSAC when you expect a smaller number of severely, structurally different bad points (e.g., a distinct wrong-surface cluster in a point cloud) — its hard exclusion handles this cleanly. Huber when outliers are more numerous but moderate in severity, and you want a smooth, differentiable, computationally cheaper method that doesn't require random sampling.

**Q: What is a "bad leverage point," and why is it more dangerous than a typical outlier?**
A: A point unusual in both its predictor value(s) (high leverage) and its response value (large residual) simultaneously — it combines the ability to pull the fitted line toward itself (leverage) with a genuinely wrong value to pull it toward, unlike a "good leverage point" (unusual $x$, but a $y$ consistent with the rest of the data) which causes little harm.

**Q: How is the Huber threshold $\delta$ typically chosen in practice?**
A: Often via a robust scale estimate like the Median Absolute Deviation (MAD) of the residuals, rather than an arbitrary fixed value — this adapts the threshold to the actual noise level of the specific dataset rather than assuming a fixed residual scale in advance.

---

*End of Chapter 20. Next: Chapter 21 — Polynomial & Nonlinear Regression (basis expansion, the centering fix for polynomial-term collinearity previewed in Chapter 12, and a first introduction to splines as a more flexible alternative).*
