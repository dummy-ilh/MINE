# Chapter 6 — Multiple Regression I
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

This is the chapter everything so far has been building toward. The good news: **almost nothing new mathematically happens here.** Chapter 5's matrix machinery — $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$, the hat matrix, the ANOVA decomposition — was deliberately built in a form that doesn't care how many columns $\mathbf{X}$ has. Multiple regression is simple regression with more columns in $\mathbf{X}$. What's genuinely new is **interpretation**: what a coefficient means when there are other predictors in the room.

### A New Worked Dataset (2 predictors)

Predicting exam score (Y) from hours studied (X1) **and** number of practice tests taken (X2), n=6 students:

| i | X1 (hours) | X2 (practice tests) | Y (score) |
|---|---|---|---|
| 1 | 2 | 1 | 65 |
| 2 | 3 | 1 | 70 |
| 3 | 4 | 2 | 75 |
| 4 | 5 | 3 | 82 |
| 5 | 6 | 3 | 88 |
| 6 | 7 | 4 | 95 |

Notice X1 and X2 rise together (both track "how much this student prepared") — this is intentional; it'll let us demonstrate a critical multiple-regression phenomenon later in this chapter.

---

## 6.1 Multiple Regression Models — the Family of Models This Framework Covers

Kutner emphasizes that "multiple regression" is a much broader umbrella than just "several distinct measured predictors." The exact same matrix framework covers:

1. **First-order model with several predictors** (what we'll work through in detail):
$$
Y_i = \beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\varepsilon_i
$$
2. **Polynomial regression in one variable** — e.g., $Y_i=\beta_0+\beta_1X_i+\beta_2X_i^2+\varepsilon_i$. Note: this is *linear in the parameters* $\beta_0,\beta_1,\beta_2$ even though it's nonlinear in X — "linear regression" refers to linearity in the coefficients, not in the predictors. This is a very commonly confused point.
3. **Qualitative (categorical) predictors** via indicator/dummy variables — e.g., $X_2=1$ if "took prep course," 0 otherwise. Fully developed in Chapter 8.
4. **Interaction models** — e.g., $Y_i=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\beta_3X_{i1}X_{i2}+\varepsilon_i$, where the effect of X1 on Y depends on the level of X2 (also Chapter 8).

**Why this matters conceptually:** in ML terms, "multiple linear regression" is really "linear-in-parameters regression on an arbitrary feature matrix" — the same OLS machinery underlies plain multivariate regression, polynomial regression, and one-hot-encoded categorical features. This is worth stating explicitly in an interview to show you understand the framework's generality, not just its most literal use case.

---

## 6.2 The General Linear Regression Model in Matrix Terms

Identical to Chapter 5, just with more columns:
$$
\mathbf{Y}=\mathbf{X}\boldsymbol\beta+\boldsymbol\varepsilon, \qquad
\mathbf{X}=\begin{bmatrix}1&X_{11}&X_{12}\\1&X_{21}&X_{22}\\\vdots&\vdots&\vdots\\1&X_{n1}&X_{n2}\end{bmatrix},\qquad
\boldsymbol\beta=\begin{bmatrix}\beta_0\\\beta_1\\\beta_2\end{bmatrix}
$$

Same assumptions: $E[\boldsymbol\varepsilon]=\mathbf{0}$, $\text{Var}(\boldsymbol\varepsilon)=\sigma^2\mathbf{I}$.

**The only thing that changes conceptually:** $p$ now denotes the total number of parameters (including the intercept) — here $p=3$ ($\beta_0,\beta_1,\beta_2$). Degrees of freedom for error becomes $n-p$ (generalizing Chapter 1's $n-2$, which was really $n-p$ with $p=2$ all along).

---

## 6.3 Estimation of Regression Coefficients

Same formula, same derivation as Chapter 5:
$$
\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}
$$

### Worked Example: Building and Inverting a 3×3 $\mathbf{X}'\mathbf{X}$

**Necessary sums:**
$$
\sum X_1=27,\ \sum X_2=14,\ \sum Y=475
$$
$$
\sum X_1^2=139,\ \sum X_2^2=40,\ \sum X_1X_2=74
$$
$$
\sum X_1Y=2243,\ \sum X_2Y=1175
$$

$$
\mathbf{X}'\mathbf{X}=\begin{bmatrix}6&27&14\\27&139&74\\14&74&40\end{bmatrix},\qquad \mathbf{X}'\mathbf{Y}=\begin{bmatrix}475\\2243\\1175\end{bmatrix}
$$

**Determinant** (cofactor expansion along row 1):
$$
\det = 6(139\cdot40-74\cdot74) - 27(27\cdot40-74\cdot14)+14(27\cdot74-139\cdot14)
$$
$$
=6(5560-5476)-27(1080-1036)+14(1998-1946)=6(84)-27(44)+14(52)
$$
$$
=504-1188+728=44
$$

**Cofactor matrix** (computing each $2\times2$ minor):
$$
C=\begin{bmatrix}84&-44&52\\-44&44&-66\\52&-66&105\end{bmatrix}
$$
(Symmetric, as expected, since $\mathbf{X}'\mathbf{X}$ is symmetric.)

$$
(\mathbf{X}'\mathbf{X})^{-1}=\frac{1}{44}\begin{bmatrix}84&-44&52\\-44&44&-66\\52&-66&105\end{bmatrix}
$$

**Solving $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$:**

$$
b_0 = \frac{84(475)-44(2243)+52(1175)}{44}=\frac{39900-98692+61100}{44}=\frac{2308}{44}=52.4545
$$
$$
b_1 = \frac{-44(475)+44(2243)-66(1175)}{44}=\frac{-20900+98692-77550}{44}=\frac{242}{44}=5.5000
$$
$$
b_2 = \frac{52(475)-66(2243)+105(1175)}{44}=\frac{24700-148038+123375}{44}=\frac{37}{44}=0.8409
$$

**Fitted model:**
$$
\hat Y_i = 52.4545+5.5000X_{i1}+0.8409X_{i2}
$$

### Interpreting the Coefficients — the Genuinely New Idea in This Chapter

**Plain English.** $b_1=5.5$ means: *holding X2 (practice tests) constant*, each additional hour studied is associated with a 5.5-point increase in predicted score. $b_2=0.8409$ means: *holding X1 (hours studied) constant*, each additional practice test is associated with a 0.84-point increase.

**Why "holding constant" is the crucial phrase, and why it's mathematically forced, not just a verbal convention.** In multiple regression, $b_1$ is a **partial** coefficient — it represents the association between X1 and Y *after netting out* whatever part of that association is really attributable to X1's correlation with X2. This is fundamentally different from a simple regression slope, which captures X1's *total* association with Y, unadjusted for anything else.

### Direct Numerical Proof: Simple vs. Multiple Regression Slopes Differ

Let's fit **simple** regression of Y on X1 alone (ignoring X2 entirely), using the same data:

$$
\bar X_1=4.5,\ \bar Y=79.1667,\quad S_{X_1X_1}=\sum(X_1-4.5)^2=17.5
$$
$$
S_{X_1Y}=\sum(X_1-4.5)(Y-79.1667)=105.5
$$
$$
b_{1,\text{simple}} = \frac{105.5}{17.5}=6.0286
$$

**Compare: $b_{1,\text{simple}}=6.03$ vs. $b_{1,\text{multiple}}=5.50$.** These are genuinely different numbers, from the same data, for "the effect of X1 on Y." Neither is "wrong" — they answer different questions:
- $b_{1,\text{simple}}=6.03$: the *total* association between hours studied and score, including whatever part flows through the fact that students who study more also tend to take more practice tests.
- $b_{1,\text{multiple}}=5.50$: the association between hours studied and score, specifically *isolating* the hours-studied effect from the practice-test effect, since both are in the model together.

**This is one of the most important conceptual facts in all of applied regression, and an extremely common interview question.**

**Interview question:** *"Why might a variable's coefficient change — sometimes substantially, sometimes even flip sign — when you add another predictor to the model?"*
**Ideal answer:** Because a simple regression coefficient captures a predictor's *total* association with the outcome, while a multiple regression coefficient captures its *partial* association — the relationship remaining after accounting for its correlation with the other included predictors. If X1 and X2 are correlated, some of X1's apparent simple-regression effect is really "borrowed" from its relationship with X2; once X2 is explicitly included, that borrowed portion is reassigned, changing (or in extreme cases of confounding, even reversing the sign of) X1's coefficient. This is precisely the mechanism behind **Simpson's paradox** in regression contexts, and it's why omitting a correlated confounding variable can produce a badly misleading coefficient — this is the core justification for controlling for confounders in causal-inference-flavored regression work.

**Common misconception:** People often assume "the coefficient for X1 should be the same whether or not I include X2" — true *only* if X1 and X2 are completely uncorrelated (orthogonal) in the data, which almost never happens with real, non-experimentally-designed data.

---

## 6.4 Fitted Values and Residuals

Identical formulas to Chapter 5, generalized: $\hat{\mathbf{Y}}=\mathbf{H}\mathbf{Y}$ where $\mathbf{H}=\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ (now built from the 3-column X), $\mathbf{e}=(\mathbf{I}-\mathbf{H})\mathbf{Y}$.

**Worked fitted values and residuals:**

| i | X1 | X2 | Y | $\hat Y$ | $e_i$ |
|---|---|---|---|---|---|
| 1 | 2 | 1 | 65 | 64.296 | 0.704 |
| 2 | 3 | 1 | 70 | 69.795 | 0.205 |
| 3 | 4 | 2 | 75 | 76.136 | -1.136 |
| 4 | 5 | 3 | 82 | 82.477 | -0.477 |
| 5 | 6 | 3 | 88 | 87.977 | 0.023 |
| 6 | 7 | 4 | 95 | 94.318 | 0.682 |

Check: $\sum e_i \approx 0.704+0.205-1.136-0.477+0.023+0.682 \approx -0.0002 \approx 0$ ✓ (same algebraic guarantee as Chapter 1, now holding for every column of X, i.e. $\sum e_i=0$ **and** $\sum X_{i1}e_i=0$ **and** $\sum X_{i2}e_i=0$ simultaneously — three orthogonality conditions instead of two, one per column of X).

$$
SSE=\sum e_i^2 \approx 0.496+0.042+1.291+0.228+0.001+0.465=2.523
$$
$$
df_{error}=n-p=6-3=3, \qquad MSE=2.523/3=0.841
$$

---

## 6.5 Analysis of Variance Results

Same decomposition, $SSTO=SSR+SSE$, but now $df_{regression}=p-1$ (number of predictors, **excluding** the intercept) — here $p-1=2$.

$$
\bar Y = 79.1667
$$
$$
SSTO=\sum(Y_i-\bar Y)^2 = 200.694+84.028+17.361+8.028+78.028+250.694=638.833
$$
$$
SSR = SSTO-SSE = 638.833-2.523=636.310
$$

| Source | SS | df | MS | F* |
|---|---|---|---|---|
| Regression | 636.310 | 2 | 318.155 | 378.3 |
| Error | 2.523 | 3 | 0.841 | |
| Total | 638.833 | 5 | | |

$$
R^2 = \frac{SSR}{SSTO}=\frac{636.310}{638.833}=0.9961
$$

**The overall F-test** ($H_0:\beta_1=\beta_2=0$, "neither predictor matters"):
$$
F^*=\frac{MSR}{MSE}=\frac{318.155}{0.841}=378.3
$$
Compare to $F_{(0.95;\,2,3)}\approx9.55$ — since $378.3\gg9.55$, we reject $H_0$: **at least one** of the two predictors has a real linear association with Y. Crucially, this F-test does **not** tell you *which* predictor(s) matter, or whether *both* matter — that requires the individual t-tests below (Section 6.6), and as we're about to see, the two tests can tell surprisingly different stories.

**Interview trap, stated explicitly:** a highly significant overall F-test is very often misread as "every predictor in the model is important." It only tells you the *combination* of predictors explains significant variance — individual predictors can still be statistically weak or entirely redundant given the others, exactly as we're about to find with X2.

---

## 6.6 Inferences About the Individual Regression Coefficients

$$
s^2(\mathbf{b})=MSE\cdot(\mathbf{X}'\mathbf{X})^{-1}
$$

Diagonal entries give $s^2(b_0), s^2(b_1), s^2(b_2)$ directly. From our $(\mathbf{X}'\mathbf{X})^{-1}=\frac{1}{44}C$ above, the diagonal of $C$ is $[84, 44, 105]$:

$$
s^2(b_0)=0.841\times\frac{84}{44}=0.841(1.9091)=1.6058,\qquad s(b_0)=1.267
$$
$$
s^2(b_1)=0.841\times\frac{44}{44}=0.841,\qquad s(b_1)=0.917
$$
$$
s^2(b_2)=0.841\times\frac{105}{44}=0.841(2.3864)=2.007,\qquad s(b_2)=1.417
$$

**Individual t-tests** ($H_0:\beta_k=0$, $df=n-p=3$):
$$
t^*_{b_1}=\frac{5.5000}{0.917}=5.998, \qquad t^*_{b_2}=\frac{0.8409}{1.417}=0.594
$$
Critical value: $t_{(0.975;3)}=3.182$.

**Result: $b_1$ (hours studied) is statistically significant ($5.998>3.182$). $b_2$ (practice tests) is NOT significant ($0.594<3.182$), despite the overall model being extremely strong ($F^*=378.3$, $R^2=0.996$).**

**Why this happens — this is the single most important practical takeaway of Chapter 6.** X1 and X2 are highly correlated in this data (both track overall preparation). Once X1 is in the model, X2 adds very little *additional* explanatory power — most of what X2 could tell you about Y, X1 already tells you. This is an early, concrete preview of **multicollinearity**, formalized fully in Chapter 7 (extra sums of squares) and Chapter 10 (VIF, variance inflation factors).

**Interview question:** *"Your model has a highly significant F-test and high R², but one predictor's individual t-test is not significant. What's going on, and what would you check next?"*
**Ideal answer:** This is a classic sign of multicollinearity — the non-significant predictor is likely highly correlated with one or more other predictors already in the model, so it doesn't provide much *additional* explanatory power beyond what's already captured. The overall F-test asks "does this whole set of predictors jointly explain significant variance?" while the individual t-test asks "does this specific predictor add anything, given everything else already in the model?" These are different questions and can have different answers. I'd check pairwise correlations (or better, Variance Inflation Factors) among the predictors, and consider whether both variables are really needed, or whether one is redundant given the other.

---

## 6.7 Estimation of Mean Response and Prediction of a New Observation

Generalizing Chapter 2's formulas directly, for a new predictor vector $\mathbf{X}_h = [1, X_{h1}, X_{h2}]'$:

$$
\hat Y_h = \mathbf{X}_h'\mathbf{b}, \qquad s^2(\hat Y_h)=MSE\cdot\mathbf{X}_h'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}_h
$$
$$
s^2(pred) = MSE\left[1+\mathbf{X}_h'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}_h\right]
$$

**Same "+1" logic as Chapter 2.5** — mean-response intervals only account for uncertainty in estimating the line; prediction intervals add the new observation's own irreducible noise $\sigma^2$ on top.

### Worked Example: predict for a new student with $X_{h1}=5$ hours, $X_{h2}=2$ practice tests

$$
\mathbf{X}_h=\begin{bmatrix}1\\5\\2\end{bmatrix}
$$

**Compute $(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}_h = \frac{1}{44}C\mathbf{X}_h$:**
$$
C\mathbf{X}_h = \begin{bmatrix}84(1)-44(5)+52(2)\\-44(1)+44(5)-66(2)\\52(1)-66(5)+105(2)\end{bmatrix}=\begin{bmatrix}84-220+104\\-44+220-132\\52-330+210\end{bmatrix}=\begin{bmatrix}-32\\44\\-68\end{bmatrix}
$$
$$
(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}_h = \frac{1}{44}\begin{bmatrix}-32\\44\\-68\end{bmatrix}=\begin{bmatrix}-0.7273\\1.0000\\-1.5455\end{bmatrix}
$$

**Then $\mathbf{X}_h'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}_h$:**
$$
=1(-0.7273)+5(1.0)+2(-1.5455)=-0.7273+5.0-3.0909=1.1818
$$

**Point prediction:**
$$
\hat Y_h = 1(52.4545)+5(5.5000)+2(0.8409)=52.4545+27.5+1.6818=81.6364
$$

**Mean-response 95% CI:**
$$
s^2(\hat Y_h)=0.841(1.1818)=0.9939,\quad s(\hat Y_h)=0.9970
$$
$$
81.636\pm3.182(0.9970)=81.636\pm3.173\Rightarrow(78.463,\ 84.809)
$$

**New-observation 95% Prediction Interval:**
$$
s^2(pred)=0.841(1+1.1818)=0.841(2.1818)=1.8349,\quad s(pred)=1.3546
$$
$$
81.636\pm3.182(1.3546)=81.636\pm4.311\Rightarrow(77.325,\ 85.947)
$$

Same pattern as Chapter 2: the prediction interval $(77.3, 85.9)$ is noticeably wider than the confidence interval $(78.5, 84.8)$ for exactly the same reason — one extra irreducible $\sigma^2$ term.

---

## Python Implementation — From Scratch (NumPy) and statsmodels

```python
import numpy as np
import statsmodels.api as sm
from scipy import stats

X1 = np.array([2,3,4,5,6,7], dtype=float)
X2 = np.array([1,1,2,3,3,4], dtype=float)
Y  = np.array([65,70,75,82,88,95], dtype=float)
n = len(Y)

# --- From scratch ---
X = np.column_stack([np.ones(n), X1, X2])
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
b = XtX_inv @ X.T @ Y
print("b0, b1, b2:", b)

Y_hat = X @ b
resid = Y - Y_hat
SSE = np.sum(resid**2)
p = X.shape[1]
df_e = n - p
MSE = SSE / df_e

se_b = np.sqrt(MSE * np.diag(XtX_inv))
t_stats = b / se_b
print("Standard errors:", se_b)
print("t-statistics:", t_stats)

SSTO = np.sum((Y-Y.mean())**2)
SSR = SSTO - SSE
F_star = (SSR/(p-1)) / MSE
R2 = SSR/SSTO
print(f"F*={F_star:.2f}, R2={R2:.4f}")

# --- Prediction for a new student (X1=5, X2=2) ---
Xh = np.array([1, 5, 2])
Yhat_h = Xh @ b
leverage_h = Xh @ XtX_inv @ Xh
se_mean = np.sqrt(MSE * leverage_h)
se_pred = np.sqrt(MSE * (1 + leverage_h))
t_crit = stats.t.ppf(0.975, df_e)
print(f"Predicted Y: {Yhat_h:.3f}")
print(f"95% CI (mean): ({Yhat_h - t_crit*se_mean:.3f}, {Yhat_h + t_crit*se_mean:.3f})")
print(f"95% PI (new obs): ({Yhat_h - t_crit*se_pred:.3f}, {Yhat_h + t_crit*se_pred:.3f})")

# --- statsmodels equivalent ---
X_sm = sm.add_constant(np.column_stack([X1, X2]))
model = sm.OLS(Y, X_sm).fit()
print(model.summary())

pred = model.get_prediction([1, 5, 2])
print(pred.summary_frame(alpha=0.05))  # gives both CI and PI columns directly
```

## scikit-learn Equivalent (point estimates only — use statsmodels for inference)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.column_stack([X1, X2])
model = LinearRegression().fit(X, Y)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Prediction for (5,2):", model.predict([[5, 2]]))
```

---

## Interview Question Bank — Chapter 6

**Conceptual:**
1. What does a "partial regression coefficient" mean, precisely, and how is it different from a simple regression slope?
2. Why can a coefficient's sign flip when you add a correlated predictor to the model? Name the phenomenon this relates to.
3. What does the overall F-test tell you that the individual t-tests don't, and vice versa?

**Derivation:**
4. Set up and invert a 3×3 $\mathbf{X}'\mathbf{X}$ matrix by hand for a 2-predictor regression, given the raw sums.
5. Derive $s^2(\hat Y_h) = MSE\cdot\mathbf{X}_h'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}_h$ conceptually from the general variance-covariance result $\text{Var}(\mathbf{b})=\sigma^2(\mathbf{X}'\mathbf{X})^{-1}$.

**ML/Statistics:**
6. You add a new feature to a model and R² goes up, but the new feature's t-test isn't significant, and an existing feature's coefficient shrinks a lot. What's happening?
7. Why is "linear regression" called linear even when using polynomial terms like $X^2$?
8. Explain why controlling for a confounding variable can reverse the sign of another variable's coefficient (Simpson's Paradox), using the partial-vs-total association framework from this chapter.

**Coding:**
9. Implement multiple regression from scratch in NumPy, including standard errors and t-statistics for each coefficient.
10. Show, with code, that adding a highly-correlated redundant feature to a regression barely changes R² but inflates the standard errors of the correlated coefficients.

**Traps:**
11. "My model has F* = 378 and R²=0.996, so every predictor in it must be important." — what's the flaw?
12. "The coefficient for X1 should be the same in the simple regression and the multiple regression, since it's the same variable." — when is this actually true, and why does it usually fail with real data?
13. Someone reports a 95% "confidence interval" for a specific new prediction, using the mean-response formula instead of the prediction formula. What's wrong, and how much does it matter?

---

*This file covers Kutner Ch. 6 — the general multiple regression model and its matrix formulation, estimation via the identical $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$ formula (worked by hand with a 3×3 inversion), the crucial distinction between simple and partial regression coefficients (demonstrated numerically), the ANOVA/F-test machinery, individual coefficient t-tests (revealing an early multicollinearity symptom), and mean-response/prediction intervals via the quadratic form $\mathbf{X}_h'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}_h$. Chapter 7 (Multiple Regression II) is next — extra sums of squares, the sequential (Type I) vs. partial (Type III) decomposition of variance, coefficients of partial determination, and a full formal treatment of multicollinearity, picking up exactly where the X1/X2 example here left off.*
