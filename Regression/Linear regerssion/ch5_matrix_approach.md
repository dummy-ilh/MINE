# Chapter 5 — Matrix Approach to Simple Linear Regression
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

Everything in Chapters 1–4 was done with scalar sums ($\sum X_i$, $\sum X_iY_i$, etc.). That works fine with one predictor, but it becomes unmanageable with many predictors. Chapter 5 rebuilds the *exact same results* using matrix notation — not new statistics, just a new, more powerful language for expressing it, which is the language Chapters 6+ (multiple regression) depend on entirely.

We'll re-derive everything using the **Chapter 1 dataset** (hours studied X vs. exam score Y, n=5) so you can directly verify the matrix results match the scalar results we already computed by hand.

$$X: 2,3,5,7,8 \qquad Y: 65,70,78,85,92 \qquad b_0=56.654,\ b_1=4.269,\ MSE=1.3718$$

---

## 5.1–5.7 Matrix Algebra Essentials (moving briskly — you know this)

Since you're comfortable with linear algebra from ML, here's the compressed version of what Kutner spends several sections building up, with emphasis only on the pieces that are easy to forget or that regression uses in a specific way:

- **Matrix**: rectangular array of numbers. **Vector**: matrix with one column (or row).
- **Transpose** ($A'$ or $A^T$): flip rows and columns. Key regression fact: $(AB)' = B'A'$ (order reverses) — used constantly in the derivations below.
- **Matrix multiplication**: $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ — requires inner dimensions to match. Not commutative in general ($AB \ne BA$).
- **Identity matrix** $I$: diagonal matrix of 1's; acts like the number 1 ($IA=A$).
- **Inverse** $A^{-1}$: the matrix such that $AA^{-1}=I$. Only exists for **square, full-rank** matrices (rank = number of linearly independent rows/columns). **Why "full rank" matters for regression**: if predictor columns are linearly dependent (perfect multicollinearity), $(X'X)^{-1}$ doesn't exist at all — this is the formal, matrix-algebra version of "you can't fit a regression with redundant/perfectly collinear predictors," and it's the single most important fact from this section for the rest of the book (this is exactly why Chapter 7's discussion of multicollinearity matters practically).
- **Rank**: the number of linearly independent rows/columns. For an $n\times p$ predictor matrix X (n observations, p parameters including intercept), we need rank(X) = p (full column rank) for $(X'X)^{-1}$ to exist.

**Interview trap:** People sometimes think "singular $X'X$ matrix" is some exotic edge case. It happens routinely in practice — e.g., one-hot-encoding a categorical variable *without* dropping a reference category (the "dummy variable trap," covered fully in Chapter 8) creates exact linear dependence among columns, making $X'X$ singular.

- **Random vectors/matrices** (5.8): a vector of random variables has an associated **mean vector** and **variance-covariance matrix** — the multivariate generalization of "mean" and "variance" for a single random variable. This is the key conceptual bridge: just like a single random variable Y has $E[Y]$ and $\text{Var}(Y)$, a vector of random variables $\mathbf{Y}$ has a mean vector $E[\mathbf{Y}]$ and a variance-covariance matrix $\boldsymbol\Sigma$ — diagonal entries are individual variances, off-diagonal entries are covariances between pairs.

---

## 5.9 The Simple Linear Regression Model in Matrix Terms

### Setting Up the Matrices

We stack all $n$ observations into vectors/matrices at once, instead of writing $n$ separate scalar equations $Y_i = \beta_0+\beta_1X_i+\varepsilon_i$:

$$
\mathbf{Y} = \begin{bmatrix}Y_1\\Y_2\\ \vdots \\Y_n\end{bmatrix}, \qquad
\mathbf{X} = \begin{bmatrix}1 & X_1\\1& X_2\\ \vdots & \vdots\\1&X_n\end{bmatrix}, \qquad
\boldsymbol\beta = \begin{bmatrix}\beta_0\\\beta_1\end{bmatrix}, \qquad
\boldsymbol\varepsilon = \begin{bmatrix}\varepsilon_1\\ \vdots \\\varepsilon_n\end{bmatrix}
$$

The entire model, for all $n$ observations simultaneously, collapses into one compact equation:
$$
\mathbf{Y} = \mathbf{X}\boldsymbol\beta + \boldsymbol\varepsilon
$$

**Why the first column of X is all 1's — this trips people up the first time they see it.** Look at row $i$ of $\mathbf{X}\boldsymbol\beta$: it's $[1, X_i]\cdot[\beta_0,\beta_1]' = 1\cdot\beta_0 + X_i\cdot\beta_1 = \beta_0+\beta_1X_i$ — exactly the scalar model for observation $i$. The column of 1's is what "activates" the intercept term $\beta_0$ for every single row equally. **This is precisely why, in code, `sm.add_constant(X)` must be called before fitting** — it's literally inserting this column of 1's into the design matrix.

**Error assumptions in matrix form:**
$$
E[\boldsymbol\varepsilon] = \mathbf{0}, \qquad \text{Var}(\boldsymbol\varepsilon) = \sigma^2 \mathbf{I}
$$
The second statement is doing a lot of work in compact notation: $\sigma^2\mathbf{I}$ is a diagonal matrix with $\sigma^2$ on every diagonal entry and 0 everywhere off-diagonal. The diagonal entries being all equal to $\sigma^2$ **is** the constant-variance (homoscedasticity) assumption; the off-diagonal zeros **are** the uncorrelated-errors assumption. Two of Chapter 1's four assumptions, both captured in a single matrix equation.

**Data matrix for our worked example:**
$$
\mathbf{X} = \begin{bmatrix}1&2\\1&3\\1&5\\1&7\\1&8\end{bmatrix}, \qquad
\mathbf{Y}=\begin{bmatrix}65\\70\\78\\85\\92\end{bmatrix}
$$

---

## 5.10 Least Squares Estimation in Matrix Form

### Deriving $\mathbf{b} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$

**The objective, in matrix form**, is the same sum-of-squared-residuals idea from Chapter 1, now written compactly:
$$
Q = (\mathbf{Y}-\mathbf{X}\mathbf{b})'(\mathbf{Y}-\mathbf{X}\mathbf{b})
$$

(Check: this expands to $\sum_i (Y_i - b_0-b_1X_i)^2$ — exactly Chapter 1's Q, just written as a vector inner product with itself.)

**Taking the matrix derivative** (using standard matrix calculus identities — you don't need to re-derive these from scratch, but here's the shape of it): expand $Q = \mathbf{Y}'\mathbf{Y} - 2\mathbf{b}'\mathbf{X}'\mathbf{Y} + \mathbf{b}'\mathbf{X}'\mathbf{X}\mathbf{b}$, then differentiate with respect to the vector $\mathbf{b}$ and set to zero:
$$
\frac{\partial Q}{\partial \mathbf{b}} = -2\mathbf{X}'\mathbf{Y} + 2\mathbf{X}'\mathbf{X}\mathbf{b} = \mathbf{0}
$$
$$
\Rightarrow \quad \mathbf{X}'\mathbf{X}\,\mathbf{b} = \mathbf{X}'\mathbf{Y} \qquad \text{(the "Normal Equations," matrix form)}
$$

Solving for $\mathbf{b}$ (assuming $\mathbf{X}'\mathbf{X}$ is invertible, i.e., full rank — see Section 5.1–5.7 above):
$$
\boxed{\mathbf{b} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}}
$$

**Why this single equation replaces both scalar normal equations from Chapter 1 at once:** with $p=2$ parameters, $\mathbf{X}'\mathbf{X}\mathbf{b}=\mathbf{X}'\mathbf{Y}$ is actually a system of 2 equations (one per row) — and if you write them out entry by entry, they are *exactly* Chapter 1's two normal equations ($\sum Y_i = nb_0+b_1\sum X_i$ and $\sum X_iY_i = b_0\sum X_i + b_1\sum X_i^2$). Nothing new statistically — just compressed notation that will scale gracefully to $p$ predictors instead of just 2.

### Worked Numerical Example

$$
\mathbf{X}'\mathbf{X} = \begin{bmatrix}5 & 25\\25&151\end{bmatrix}
$$
(top-left = $n=5$; off-diagonal = $\sum X_i = 25$; bottom-right = $\sum X_i^2 = 4+9+25+49+64=151$.)

$$
\mathbf{X}'\mathbf{Y} = \begin{bmatrix}\sum Y_i \\ \sum X_iY_i\end{bmatrix} = \begin{bmatrix}390\\2061\end{bmatrix}
$$
($\sum X_iY_i = 2(65)+3(70)+5(78)+7(85)+8(92)=130+210+390+595+736=2061$, matching Chapter 4's calculation.)

**Inverting the 2×2 matrix** (using the standard formula $\begin{bmatrix}a&b\\c&d\end{bmatrix}^{-1}=\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$):
$$
\det(\mathbf{X}'\mathbf{X}) = 5(151)-25(25) = 755-625=130
$$
$$
(\mathbf{X}'\mathbf{X})^{-1} = \frac{1}{130}\begin{bmatrix}151&-25\\-25&5\end{bmatrix}
$$

**Multiplying through:**
$$
\mathbf{b} = \frac{1}{130}\begin{bmatrix}151&-25\\-25&5\end{bmatrix}\begin{bmatrix}390\\2061\end{bmatrix}
$$
Row 1: $151(390)+(-25)(2061) = 58890-51525=7365 \Rightarrow b_0 = 7365/130 = 56.654$ ✓
Row 2: $-25(390)+5(2061) = -9750+10305=555 \Rightarrow b_1 = 555/130 = 4.269$ ✓

**Exact match with Chapter 1's hand calculation** — confirming the matrix machinery reproduces the scalar results perfectly; it's the same math, different notation.

---

## The Hat Matrix: $\mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$

### Why This Matrix Deserves Its Own Name

**Plain English.** The fitted values are $\hat{\mathbf{Y}} = \mathbf{X}\mathbf{b} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$. Define $\mathbf{H} = \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ (an $n\times n$ matrix depending *only* on X, not on Y at all). Then:
$$
\hat{\mathbf{Y}} = \mathbf{H}\mathbf{Y}
$$
**"H" stands for "hat"** because it puts the hat on Y — it's the matrix that transforms observed Y directly into fitted $\hat Y$, without needing to separately compute $\mathbf{b}$ first.

**Why this matters — this is one of the single most important objects in the whole book, and it foreshadows Chapter 10 (influence/leverage) directly.** Each diagonal entry $h_{ii}$ of $\mathbf{H}$ measures the **leverage** of observation $i$ — literally, how much observation $i$'s own $Y_i$ value influences its own fitted value $\hat Y_i$. Points with $X_i$ far from $\bar X$ get *larger* $h_{ii}$ — they have more leverage, meaning they pull the fitted line toward themselves more strongly.

**Key algebraic facts about H** (all provable directly, and all frequently tested conceptually):
- $\mathbf{H}$ is **symmetric**: $\mathbf{H}'=\mathbf{H}$.
- $\mathbf{H}$ is **idempotent**: $\mathbf{H}\mathbf{H}=\mathbf{H}$ (applying it twice does nothing extra — makes sense, since projecting an already-projected point again changes nothing).
- $\text{trace}(\mathbf{H}) = p$ (the number of parameters, here $p=2$) — **the leverages always sum to exactly the number of parameters in the model**, regardless of the dataset.
- Residuals: $\mathbf{e} = \mathbf{Y}-\hat{\mathbf{Y}} = (\mathbf{I}-\mathbf{H})\mathbf{Y}$.

### Worked Example: Computing Leverage Values by Hand

For observation $i$ with predictor row $\mathbf{x}_i' = [1, X_i]$: $h_{ii} = \mathbf{x}_i'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_i$.

**For $X_1=2$:**
$$
(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_1 = \frac{1}{130}\begin{bmatrix}151&-25\\-25&5\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \frac{1}{130}\begin{bmatrix}151-50\\-25+10\end{bmatrix}=\frac{1}{130}\begin{bmatrix}101\\-15\end{bmatrix}
$$
$$
h_{11} = [1,2]\cdot\frac{1}{130}\begin{bmatrix}101\\-15\end{bmatrix} = \frac{101 + 2(-15)}{130}=\frac{101-30}{130}=\frac{71}{130}=0.5462
$$

**For $X_5=8$** (by symmetry — same distance 3 from $\bar X=5$ as $X_1=2$ — we should get the same leverage):
$$
(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_5 = \frac{1}{130}\begin{bmatrix}151-200\\-25+40\end{bmatrix}=\frac{1}{130}\begin{bmatrix}-49\\15\end{bmatrix}
$$
$$
h_{55}=\frac{-49+8(15)}{130}=\frac{-49+120}{130}=\frac{71}{130}=0.5462 \quad \checkmark \text{(matches, as expected)}
$$

**For $X_3=5=\bar X$** (the point exactly at the mean — should have the *smallest* leverage, since it's the "pivot" the line rotates around):
$$
(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_3 = \frac{1}{130}\begin{bmatrix}151-125\\-25+25\end{bmatrix}=\frac{1}{130}\begin{bmatrix}26\\0\end{bmatrix}
$$
$$
h_{33}=\frac{26+5(0)}{130}=\frac{26}{130}=0.2
$$

Computing the remaining two similarly gives $h_{22}=h_{44}=0.3538$ (for $X=3,7$, symmetric distance 2 from mean).

**Verify the trace fact:**
$$
\sum h_{ii} = 0.5462+0.3538+0.2+0.3538+0.5462 = 2.0000 = p \quad \checkmark
$$

**Beautiful confirmation of the theory**: leverages sum to exactly 2 (the number of parameters, $\beta_0$ and $\beta_1$), and they're smallest at the center of the X-distribution and largest at the extremes — the exact matrix-algebra confirmation of Chapter 2's intuition that predictions are least uncertain near $\bar X$ and most uncertain (and most influenced by that single point) at the extremes.

**Interview question:** *"What is the hat matrix, and what does its diagonal tell you?"*
**Ideal answer:** $\mathbf{H}=\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ is the matrix mapping observed Y directly to fitted $\hat Y$. Its diagonal entries $h_{ii}$ measure leverage — how strongly observation $i$'s own response value pulls its own fitted value toward it. Points with extreme predictor values have high leverage; the leverages always sum to exactly the number of model parameters, regardless of the data. High-leverage points deserve scrutiny because, combined with a large residual, they can indicate an influential outlier disproportionately steering the whole fitted line (formalized fully via Cook's distance in Chapter 10).

---

## Variance-Covariance Matrix of $\mathbf{b}$

$$
\text{Var}(\mathbf{b}) = \sigma^2(\mathbf{X}'\mathbf{X})^{-1}
$$

**Why this one compact formula replaces three separate Chapter 2 formulas at once:** the diagonal entries of $\sigma^2(\mathbf{X}'\mathbf{X})^{-1}$ are $\text{Var}(b_0)$ and $\text{Var}(b_1)$ (matching Chapter 2's formulas exactly), and the off-diagonal entry is $\text{Cov}(b_0,b_1)$ — new information we didn't compute explicitly before, though Chapter 2.3 warned us it was generally nonzero.

**Estimated version** (plugging in $MSE$ for unknown $\sigma^2$):
$$
s^2(\mathbf{b}) = MSE\cdot(\mathbf{X}'\mathbf{X})^{-1} = 1.3718\times\frac{1}{130}\begin{bmatrix}151&-25\\-25&5\end{bmatrix}
$$
$$
= \begin{bmatrix}1.5936 & -0.2638\\-0.2638 & 0.05276\end{bmatrix}
$$

**Reading off the diagonal:** $s^2(b_0)=1.5936$ (Chapter 2 got 1.594 — matches within rounding), $s^2(b_1)=0.05276$ (Chapter 2 got exactly 0.05276 — matches perfectly).

**Reading off the new off-diagonal entry:** $\text{Cov}(b_0,b_1) = -0.2638$. **This is genuinely new information** — it confirms Chapter 2.3's claim that $b_0$ and $b_1$ are *not* independent: here they're **negatively correlated**. Intuitively: if a particular sample happens to produce a steeper-than-true slope estimate ($b_1$ too high), the line has to compensate by having a lower intercept ($b_0$ too low) to still pass through $(\bar X,\bar Y)$ — hence the negative covariance. (The sign of this covariance is always $-\bar X$ times a positive quantity — so whenever $\bar X > 0$, expect negative covariance between $b_0$ and $b_1$, exactly as we see here with $\bar X=5$.)

---

## 5.12 ANOVA Results in Matrix Form

$$
SSTO = \mathbf{Y}'\mathbf{Y} - \frac{1}{n}(\mathbf{1}'\mathbf{Y})^2, \qquad SSE = \mathbf{Y}'\mathbf{Y}-\mathbf{b}'\mathbf{X}'\mathbf{Y}, \qquad SSR = SSTO - SSE
$$

**Worked check:**
$$
\mathbf{Y}'\mathbf{Y} = 65^2+70^2+78^2+85^2+92^2 = 4225+4900+6084+7225+8464=30898
$$
$$
\frac{(\mathbf{1}'\mathbf{Y})^2}{n} = \frac{390^2}{5}=\frac{152100}{5}=30420
$$
$$
SSTO = 30898-30420=478 \quad \checkmark \text{(matches Chapter 2 exactly)}
$$

$$
\mathbf{b}'\mathbf{X}'\mathbf{Y} = b_0(390)+b_1(2061) \approx 56.654(390)+4.269(2061) \approx 22095.0+8798.8=30893.8
$$
$$
SSE = 30898-30893.8 \approx 4.1 \quad \checkmark \text{(matches Chapter 1's SSE=4.1154 within rounding)}
$$

Nothing new statistically here either — just confirming that the ANOVA decomposition from Chapter 2.7 is exactly reproduced by this compact matrix formula, which will generalize directly to multiple regression's ANOVA table in Chapter 6.

---

## Python Implementation — From Scratch (NumPy, Matrix Form)

```python
import numpy as np

X_raw = np.array([2, 3, 5, 7, 8], dtype=float)
Y = np.array([65, 70, 78, 85, 92], dtype=float).reshape(-1, 1)
n = len(X_raw)

# Design matrix: column of 1's + X
X = np.column_stack([np.ones(n), X_raw])

# --- Normal equations, matrix form ---
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
XtY = X.T @ Y
b = XtX_inv @ XtY
print("b0, b1:", b.flatten())

# --- Hat matrix and leverages ---
H = X @ XtX_inv @ X.T
leverages = np.diag(H)
print("Leverages (h_ii):", np.round(leverages, 4))
print("Sum of leverages (should equal p=2):", leverages.sum())

# --- Fitted values and residuals via H ---
Y_hat = H @ Y
resid = (np.eye(n) - H) @ Y
print("Fitted values:", Y_hat.flatten())
print("Residuals:", resid.flatten())

# --- Variance-covariance matrix of b ---
SSE = float(resid.T @ resid)
MSE = SSE / (n - 2)
var_b = MSE * XtX_inv
print("Var-cov matrix of b:\n", var_b)

# --- ANOVA via matrices ---
ones = np.ones((n,1))
SSTO = float(Y.T @ Y - (ones.T @ Y)**2 / n)
SSE_check = float(Y.T @ Y - b.T @ XtY)
print(f"SSTO={SSTO:.3f}, SSE={SSE_check:.3f}, SSR={SSTO-SSE_check:.3f}")
```

## Equivalent via statsmodels (matrix machinery is what's running under the hood)

```python
import statsmodels.api as sm
import numpy as np

X_raw = np.array([2, 3, 5, 7, 8], dtype=float)
Y = np.array([65, 70, 78, 85, 92], dtype=float)
X = sm.add_constant(X_raw)  # inserts the column of 1's exactly as we did manually

model = sm.OLS(Y, X).fit()
H = model.get_influence().hat_matrix_diag  # leverage values directly
print("Leverages from statsmodels:", np.round(H, 4))
print("Cov matrix of coefficients:\n", model.cov_params())
```

---

## Interview Question Bank — Chapter 5

**Conceptual:**
1. Why must the design matrix X include a column of 1's to represent an intercept?
2. What does it mean, in matrix terms, for a regression model to be "unidentifiable," and how does that connect to invertibility of $(X'X)$?
3. What is the hat matrix, in one sentence, and why is it named that?

**Derivation:**
4. Derive $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$ from the matrix objective function $Q=(\mathbf{Y}-\mathbf{Xb})'(\mathbf{Y}-\mathbf{Xb})$.
5. Prove that $\mathbf{H}$ is idempotent ($\mathbf{HH}=\mathbf{H}$) and explain intuitively why that must be true.
6. Show why $\text{trace}(\mathbf{H})=p$ using the property $\text{trace}(AB)=\text{trace}(BA)$.

**ML/Statistics:**
7. Why does the "dummy variable trap" (not dropping a reference category in one-hot encoding) break the invertibility of $X'X$?
8. What practical modeling problem does a high-leverage point signal, and what additional information (not captured by leverage alone) do you need to know if it's actually problematic?
9. Why are $b_0$ and $b_1$ typically correlated with each other, and what does the sign of that covariance depend on?

**Coding:**
10. Implement the hat matrix and leverage calculation from scratch in NumPy for an arbitrary design matrix X.
11. Using statsmodels, extract the coefficient covariance matrix and verify it matches your from-scratch calculation.

**Traps:**
12. "Since $(X'X)^{-1}$ exists, my model must be well-specified." — what's wrong with this reasoning (invertibility says nothing about the four assumptions from Ch. 1)?
13. Someone claims "leverage measures how much an observation is an outlier." What's the precise distinction between leverage (unusualness in X) and a large residual (unusualness in Y given X)?

---

*This file covers Kutner Ch. 5 — matrix representation of the regression model, the least-squares solution $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$ derived and verified by hand, the hat matrix and leverage (with a full worked numerical example confirming the trace=p property), the variance-covariance matrix of $\mathbf{b}$ (revealing the b0/b1 covariance), and the matrix form of the ANOVA decomposition. This completes the simple-linear-regression arc of the book. Chapter 6 (Multiple Regression I) is next — and virtually everything here generalizes directly: the same $\mathbf{b}=(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$ formula, the same hat matrix, the same ANOVA structure — just with X now having more than 2 columns.*
