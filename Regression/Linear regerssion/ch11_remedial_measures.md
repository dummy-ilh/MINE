# Chapter 11 — Building the Regression Model III: Remedial Measures
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Chapter 3 gave transformations as the first-line remedy for violated assumptions. This chapter covers what to do when transformations aren't enough or aren't appropriate: **Weighted Least Squares** (for non-constant variance), **Ridge Regression** (for multicollinearity), and brief treatments of **robust regression** and **bootstrapping**. These are the classical-statistics roots of ideas that now live at the center of modern ML (regularization, robust loss functions, resampling-based uncertainty quantification).

---

## 11.1 Unequal Error Variances Remedial Measures: Weighted Least Squares

### Why Ordinary Least Squares Struggles Under Heteroscedasticity

**Plain English.** Ordinary least squares treats every observation as equally trustworthy — it minimizes $\sum(Y_i-\hat Y_i)^2$, giving each squared residual equal weight regardless of how noisy that particular observation actually is. If some observations are inherently noisier than others (non-constant variance, Chapter 3), OLS is still **unbiased**, but it's no longer the *most efficient* (lowest-variance) estimator — it's being needlessly influenced by the noisiest, least trustworthy points just as much as the cleanest ones.

### The Weighted Least Squares Fix

**The idea:** if you know (or can estimate) each observation's error variance $\sigma_i^2$, give each observation a weight $w_i=1/\sigma_i^2$ — **inversely proportional to its noise level** — so noisier points contribute less to the fit.

$$
Q_W = \sum_i w_i(Y_i-b_0-b_1X_i)^2, \qquad w_i=\frac{1}{\sigma_i^2}
$$

**Matrix form:** $\mathbf{b}_{WLS}=(\mathbf{X}'\mathbf{W}\mathbf{X})^{-1}\mathbf{X}'\mathbf{W}\mathbf{Y}$, where $\mathbf{W}=\text{diag}(w_1,\ldots,w_n)$.

**Why this is the theoretically correct fix, not just an ad-hoc adjustment.** Weighted least squares is exactly ordinary least squares applied *after* dividing every observation by its own standard deviation $\sigma_i$ — a transformation that makes the rescaled errors have *constant* variance again, restoring OLS's optimality (Gauss-Markov) in the transformed space.

### Worked Example: Variance Growing with X

Suppose (from prior knowledge or a diagnostic like Chapter 3's residual-vs-X plot) we know $\text{Var}(\varepsilon_i) \propto X_i^2$ — a common pattern when the noise is roughly a fixed *percentage* of the signal rather than a fixed absolute amount (e.g., measurement error scaling with the magnitude being measured).

**Data** ($n=6$, true relationship $Y=5+2X$, with noise growing linearly in X):

| X | Y |
|---|---|
| 1 | 7.5 |
| 2 | 8.0 |
| 3 | 12.5 |
| 4 | 11.0 |
| 5 | 17.5 |
| 6 | 14.0 |

**OLS fit first** (for contrast), using the standard Chapter 1 method: $\bar X=3.5$, $\bar Y=11.75$, $S_{XX}=17.5$, $S_{XY}=29.75$:
$$
b_{1,OLS}=\frac{29.75}{17.5}=1.700, \qquad b_{0,OLS}=11.75-1.700(3.5)=5.800
$$
**Notice: the true slope is 2.0, and OLS gives 1.700** — pulled down by the high-variance, high-X observations (especially $X=6,Y=14$, a large negative deviation from the true line at a point where noise is largest).

**Now weighted least squares**, using $w_i=1/X_i^2$ (since $\text{Var}\propto X_i^2$):

$$
w_1=1,\ w_2=0.25,\ w_3=0.1111,\ w_4=0.0625,\ w_5=0.04,\ w_6=0.02778
$$

**The weighted normal equations:**
$$
b_0\sum w_i + b_1\sum w_iX_i = \sum w_iY_i \qquad b_0\sum w_iX_i+b_1\sum w_iX_i^2=\sum w_iX_iY_i
$$

Since $w_i=1/X_i^2$: note $\sum w_iX_i = \sum 1/X_i = 1+\frac12+\frac13+\frac14+\frac15+\frac16=2.45$ (the harmonic sum), and $\sum w_iX_i^2=\sum 1 = 6$ exactly — a nice simplification specific to this weighting scheme.

Computing $\sum w_i=1.4914$, $\sum w_iY_i=12.6653$, $\sum w_iX_iY_i=24.25$ (direct arithmetic — each term is $Y_i/X_i$ summed: $7.5+4.0+4.1667+2.75+3.5+2.3333=24.25$):

$$
1.4914\,b_0+2.45\,b_1=12.6653 \qquad 2.45\,b_0+6.0\,b_1=24.25
$$

Solving this 2×2 system (same elimination method as always):
$$
b_{1,WLS}=1.7435, \qquad b_{0,WLS}=5.628
$$

**Compare: $b_{1,OLS}=1.700$ vs. $b_{1,WLS}=1.7435$, vs. true value $2.0$.** WLS moves the estimate closer to the truth — modest in this small toy example, but the mechanism is exactly right: by downweighting the noisy, high-X observations (which OLS had been trusting just as much as the cleaner low-X ones), WLS lets the reliable, low-variance points drive the fit more.

**How weights are chosen in practice (since true $\sigma_i^2$ is essentially never known exactly):**
1. If theory specifies the variance function (e.g., Poisson-like data where $\text{Var}\approx\text{mean}$), use that directly.
2. Estimate variance empirically — e.g., group observations by X-range and compute each group's sample variance (similar in spirit to the Brown-Forsythe grouping from Chapter 3), then use the fitted variance-vs-X relationship to derive weights.
3. **Iteratively Reweighted Least Squares (IRLS):** fit OLS first, use the residuals to estimate a variance function, refit with those weights, re-estimate the variance function from the *new* residuals, and repeat until the weights stabilize — this iterative refinement is also exactly the algorithm underlying logistic regression's maximum likelihood fitting (foreshadowing Chapter 14).

**Interview question:** *"When would you use weighted least squares instead of transforming your outcome variable to fix heteroscedasticity?"*
**Ideal answer:** Transformations (Chapter 3's Box-Cox family) change *what* you're modeling — the transformed Y often has a less directly interpretable scale. WLS keeps Y on its original, interpretable scale and instead directly encodes your knowledge (or estimate) of how the noise varies across observations. WLS is preferable when you have a good handle on the variance structure itself (e.g., known measurement precision varying by observation, or a theoretically justified variance function), while transformations are often a more practical default when the variance structure is unknown but a standard pattern (like variance growing with the mean) is suspected from diagnostic plots.

---

## 11.2 Multicollinearity Remedial Measures: Ridge Regression

### Recap of the Problem, and Why Simple Fixes (Dropping a Variable) Aren't Always Acceptable

Chapter 7 showed severe multicollinearity ($VIF\approx17.5$) between hours-studied (X1) and practice-tests (X2) inflates coefficient variance without biasing point estimates. Dropping one variable "fixes" the instability but throws away a variable you might have good theoretical reasons to keep, and doesn't generalize to situations with many correlated predictors where you don't want to arbitrarily discard information.

### The Ridge Idea

**Plain English.** Ridge regression deliberately introduces a small amount of **bias** in exchange for a large reduction in coefficient **variance** — a direct, explicit bias-variance tradeoff, the same concept underlying regularization throughout modern ML.

$$
\mathbf{b}_{ridge} = (\mathbf{X}'\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}'\mathbf{Y}
$$

**Why adding $\lambda\mathbf{I}$ fixes the instability, mechanically.** Recall from Chapter 5: multicollinearity means $\mathbf{X}'\mathbf{X}$ is close to singular (its determinant near zero), which is exactly why $(\mathbf{X}'\mathbf{X})^{-1}$ blows up and coefficient variance ($\sigma^2(\mathbf{X}'\mathbf{X})^{-1}$) explodes. **Adding $\lambda\mathbf{I}$ (a positive constant on the diagonal) directly inflates the determinant away from zero**, making the matrix safely invertible and dramatically shrinking the resulting variance — at the cost of the resulting $\mathbf{b}_{ridge}$ no longer being unbiased.

**Practical convention:** ridge is typically applied to **standardized** predictors (Chapter 7.5's correlation transformation) so that the penalty $\lambda$ applies comparably across variables regardless of their original units, and the intercept is typically left unpenalized (since it just centers the model, and centering the data beforehand handles this automatically).

### Worked Example: Applying Ridge to the Chapter 7 Multicollinear Data

Recall from Chapter 7: $r_{12}=0.9711$ (correlation between X1 and X2), and the OLS **standardized** coefficients were $b_1^*=0.9103$, $b_2^*=0.0901$.

**In standardized/correlation-transformed form, the normal equations become:**
$$
\begin{bmatrix}1&r_{12}\\r_{12}&1\end{bmatrix}\begin{bmatrix}b_1^*\\b_2^*\end{bmatrix}=\begin{bmatrix}r_{1Y}\\r_{2Y}\end{bmatrix}
$$

We can back out $r_{1Y}, r_{2Y}$ from the known OLS solution: $r_{1Y}=b_1^*+b_2^*r_{12}=0.9103+0.0901(0.9711)=0.9978$; $r_{2Y}=b_1^*r_{12}+b_2^*=0.9103(0.9711)+0.0901=0.9741$. (Sanity check: $r_{1Y}^2=0.9956\approx R^2$ from regressing Y on X1 alone in Chapter 6 — matches exactly.)

**The ridge system, with penalty $\lambda$ added to the diagonal:**
$$
\begin{bmatrix}1+\lambda&r_{12}\\r_{12}&1+\lambda\end{bmatrix}\begin{bmatrix}b_1^*\\b_2^*\end{bmatrix}=\begin{bmatrix}r_{1Y}\\r_{2Y}\end{bmatrix}
$$

**At $\lambda=0.1$:**
$$
\det = (1.1)^2-(0.9711)^2 = 1.21-0.9430=0.2670
$$
$$
b_1^* = \frac{1.1(0.9978)-0.9711(0.9741)}{0.2670}=\frac{1.0976-0.9459}{0.2670}=\frac{0.1517}{0.2670}=0.568
$$
$$
b_2^* = \frac{1.1(0.9741)-0.9711(0.9978)}{0.2670}=\frac{1.0715-0.9690}{0.2670}=\frac{0.1025}{0.2670}=0.384
$$

**At $\lambda=1.0$ (a much stronger penalty):**
$$
\det=(2)^2-(0.9711)^2=4-0.9430=3.057
$$
$$
b_1^*=\frac{2(0.9978)-0.9711(0.9741)}{3.057}=\frac{1.9956-0.9459}{3.057}=\frac{1.0497}{3.057}=0.343
$$
$$
b_2^*=\frac{2(0.9741)-0.9711(0.9978)}{3.057}=\frac{1.9482-0.9690}{3.057}=\frac{0.9792}{3.057}=0.320
$$

### The Key Pattern: Ridge Redistributes Credit Between Correlated Variables

| $\lambda$ | $b_1^*$ | $b_2^*$ |
|---|---|---|
| 0 (OLS) | 0.910 | 0.090 |
| 0.1 | 0.568 | 0.384 |
| 1.0 | 0.343 | 0.320 |

**This is the single most important qualitative pattern to recognize about ridge under multicollinearity.** OLS assigned almost all the credit to X1 and almost none to X2 (0.910 vs. 0.090) — an artifact of which variable happened to align slightly better with Y in this particular sample, given how correlated they are with each other. As $\lambda$ increases, ridge pulls the two coefficients **toward each other** — by $\lambda=1.0$, they're almost equal (0.343 vs. 0.320). **Ridge doesn't "know" which of two highly correlated variables is the truly important one, so it hedges by splitting credit roughly evenly between them** — a direct, principled response to the ambiguity multicollinearity creates, rather than the somewhat arbitrary all-or-nothing allocation OLS produces.

**Interview question:** *"Why does ridge regression tend to shrink the coefficients of highly correlated predictors toward each other, specifically, rather than just shrinking every coefficient toward zero by the same amount?"*
**Ideal answer:** When two predictors are highly correlated, OLS can achieve almost the same fit with many different splits of the coefficient values between them (e.g., mostly on X1 and almost none on X2, or vice versa, or somewhere in between) — the data barely distinguishes these options, which is exactly why their individual variances are so inflated. Ridge's penalty term is minimized, for a given total fit quality, by the most "balanced" split of coefficients (since the penalty is on the sum of squared coefficients, and for a fixed sum, squared terms are minimized when the values are as equal as possible) — so among the many nearly-equally-good ways to fit the data, ridge selects the one that spreads credit evenly across the correlated variables, rather than arbitrarily favoring one.

**Choosing $\lambda$ in practice:** typically via cross-validation (holding out data and choosing $\lambda$ that minimizes out-of-sample prediction error) — the modern default — or via a "ridge trace" plot (coefficients plotted against a range of $\lambda$ values, choosing the point where they stabilize), which is Kutner's classical-era recommendation, predating cross-validation's practical dominance.

**Important honesty point for interviews:** ridge coefficients are **biased** — by design. The whole point is trading a controlled amount of bias for a large reduction in variance, typically improving *out-of-sample prediction accuracy* even though the in-sample coefficients are no longer unbiased estimates of the true $\beta$'s. This is worth stating explicitly if asked "does ridge regression give you unbiased estimates" — the answer is a clear, confident "no, intentionally not."

---

## 11.3 Robust Regression (Brief)

**Plain English.** OLS's squared-error objective is extremely sensitive to a small number of extreme outliers, since squaring amplifies large deviations disproportionately (recall Chapter 1's original justification for squaring — this cuts both ways). **Robust regression** replaces the squared-error loss with something less sensitive to extreme values.

**M-estimation (the classical approach Kutner covers):** instead of minimizing $\sum(Y_i-\hat Y_i)^2$, minimize $\sum \rho(e_i)$ for some function $\rho$ that grows more slowly than a square for large residuals — e.g., **Huber loss**, which behaves like squared error for small residuals (preserving efficiency when data is well-behaved) but switches to linear (absolute-value-like) growth beyond a threshold, capping the influence any single extreme point can have.

**Fitting procedure: Iteratively Reweighted Least Squares (IRLS)** — fit OLS, compute residuals, downweight observations with large residuals, refit, recompute weights, repeat until convergence. **Notice this is structurally the exact same iterative idea as WLS's variance-based reweighting in Section 11.1** — the difference is *why* you're reweighting (down-weighting suspected outliers vs. down-weighting known-noisy observations), but the computational machinery (iteratively reweighted least squares) is identical.

**Interview connection:** Huber loss is directly used as a loss function in modern ML regression tasks (e.g., available in scikit-learn's `HuberRegressor`, and as an option in gradient boosting objectives) for exactly the same reason Kutner introduces it here — robustness to outliers without fully discarding the efficiency benefits of squared-error loss for well-behaved data.

---

## 11.5 Bootstrapping for Inference in Nonstandard Situations (Brief)

**Plain English.** All the confidence intervals and hypothesis tests built in Chapters 2 and 4 relied on the Normal-errors assumption (Chapter 1.11) to derive exact t- and F-distributions. When that assumption is seriously in doubt (skewed or heavy-tailed errors) and the sample is too small for the Central Limit Theorem to rescue you, **bootstrapping** offers a way to estimate a coefficient's sampling distribution empirically rather than relying on a theoretical formula.

**The procedure:** repeatedly resample the observed data **with replacement** (each bootstrap sample the same size as the original), refit the regression on each resample, and collect the resulting distribution of $b_1$ (or any other statistic of interest) across thousands of resamples. **Use the spread of this empirical distribution directly as your confidence interval**, instead of $b_1\pm t\cdot s(b_1)$.

**Why this works, conceptually:** the observed sample is treated as a stand-in for the true population; resampling from it repeatedly simulates "what if we'd drawn a different sample from the same underlying population" — directly approximating the sampling variability that the t-distribution was only ever an analytical approximation of in the first place.

**Interview connection:** bootstrapping is now a completely standard tool across ML for estimating uncertainty in any statistic — model performance metrics, feature importance scores, prediction intervals for complex models where no clean analytical formula exists at all (e.g., random forests, neural networks) — a direct generalization of what Kutner introduces here narrowly for regression coefficients.

---

## Python Implementation

```python
import numpy as np

# --- Weighted Least Squares ---
X = np.array([1,2,3,4,5,6], dtype=float)
Y = np.array([7.5,8,12.5,11,17.5,14], dtype=float)
n = len(X)

w = 1 / X**2   # weights, since Var(e) is assumed proportional to X^2
W = np.diag(w)
X_design = np.column_stack([np.ones(n), X])

b_wls = np.linalg.inv(X_design.T @ W @ X_design) @ X_design.T @ W @ Y
print("WLS coefficients [b0, b1]:", b_wls)

b_ols = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y
print("OLS coefficients [b0, b1]:", b_ols)
```

```python
# --- Ridge Regression (on the Chapter 7 multicollinear dataset) ---
X1 = np.array([2,3,4,5,6,7], dtype=float)
X2 = np.array([1,1,2,3,3,4], dtype=float)
Y  = np.array([65,70,75,82,88,95], dtype=float)
n = len(Y)

# standardize predictors (correlation transformation)
def standardize(v):
    return (v - v.mean()) / v.std(ddof=1)

X1_s, X2_s, Y_s = standardize(X1), standardize(X2), standardize(Y)
X_s = np.column_stack([X1_s, X2_s])

for lam in [0, 0.1, 1.0, 5.0]:
    ridge_b = np.linalg.inv(X_s.T @ X_s + lam*np.eye(2)) @ X_s.T @ Y_s
    print(f"lambda={lam}: b1*={ridge_b[0]:.3f}, b2*={ridge_b[1]:.3f}")
```

```python
# --- scikit-learn equivalents ---
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler

X_raw = np.column_stack([X1, X2])
X_scaled = StandardScaler().fit_transform(X_raw)
Y_scaled = StandardScaler().fit_transform(Y.reshape(-1,1)).ravel()

for lam in [0.1, 1.0, 5.0]:
    ridge = Ridge(alpha=lam, fit_intercept=False).fit(X_scaled, Y_scaled)
    print(f"sklearn Ridge lambda={lam}: {ridge.coef_}")

huber = HuberRegressor().fit(X_raw, Y)
print("Huber (robust) coefficients:", huber.coef_)
```

```python
# --- Bootstrap confidence interval for a slope ---
import numpy as np

def bootstrap_slope(X, Y, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)  # resample with replacement
        Xb, Yb = X[idx], Y[idx]
        Xb_bar, Yb_bar = Xb.mean(), Yb.mean()
        slopes[b] = np.sum((Xb-Xb_bar)*(Yb-Yb_bar)) / np.sum((Xb-Xb_bar)**2)
    return slopes

slopes = bootstrap_slope(X, Y)
print("Bootstrap 95% CI for slope:", np.percentile(slopes, [2.5, 97.5]))
```

---

## Interview Question Bank — Chapter 11

**Conceptual:**
1. Why is OLS still unbiased under heteroscedasticity, but no longer the "best" (minimum variance) estimator?
2. What specific problem does adding $\lambda\mathbf{I}$ to $\mathbf{X}'\mathbf{X}$ solve, mechanically?
3. Why does ridge regression intentionally introduce bias, and why might that still improve real-world predictive performance?

**Derivation:**
4. Derive the weighted normal equations from the weighted objective function $Q_W=\sum w_i(Y_i-b_0-b_1X_i)^2$.
5. Show why, for two highly correlated standardized predictors, ridge regression's solution pushes their coefficients toward equality as $\lambda\to\infty$.

**ML/Statistics:**
6. Compare ridge regression's bias-variance tradeoff to WLS's efficiency argument — are they solving the same kind of problem?
7. When would you prefer Huber loss over squared-error loss in a regression model, and what does IRLS have to do with fitting it?
8. Explain how bootstrapping estimates a coefficient's sampling distribution without relying on the Normal-errors assumption.

**Coding:**
9. Implement weighted least squares from scratch in NumPy given a known or estimated per-observation variance.
10. Implement a "ridge trace" — coefficients across a grid of λ values — and identify where they stabilize.
11. Implement a bootstrap confidence interval for a regression slope from scratch.

**Traps:**
12. "Ridge regression gives more accurate (less biased) coefficient estimates than OLS." — what's wrong with this claim?
13. "Since OLS is unbiased even with heteroscedastic errors, there's no reason to bother with weighted least squares." — what's the flaw in this reasoning?
14. Someone bootstraps a regression slope and gets a much wider CI than the classical t-based formula gives. What might that discrepancy indicate about the data?

---

*This file covers Kutner Ch. 11 — weighted least squares as the principled fix for heteroscedasticity (worked by hand, showing the WLS slope moving closer to the true value than OLS), ridge regression as the fix for multicollinearity (worked by hand on the Chapter 7 dataset, showing the characteristic "shrink correlated coefficients toward each other" pattern across increasing λ), and brief treatments of robust regression (Huber loss, IRLS) and bootstrapping. This completes the "Building the Regression Model" three-chapter arc (9: selection, 10: diagnostics, 11: remedial measures) — together with Chapters 6–8, this is the complete applied multiple-regression toolkit. From here, Kutner continues into Chapter 12 (autocorrelation/time series), Chapter 13 (nonlinear regression), and Chapter 14 (logistic/Poisson/GLMs) — let me know if you'd like to continue into any of those, revisit anything, or shift to another part of your interview prep.*
