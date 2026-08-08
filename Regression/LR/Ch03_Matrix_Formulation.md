# Chapter 3 — Matrix Formulation of Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — still using the same 5-student dataset (x = hours studied, y = exam score) so the matrix arithmetic below produces the exact same $\hat{\beta}_0=41.5$, $\hat{\beta}_1=7.5$ you've already verified by hand twice.*

---

## 3.1 The Motivating Question

Everything in Chapters 1–2 was written using summation notation ($\sum$) because simple linear regression only has one predictor. The moment you have **two or more predictors**, summation notation becomes unmanageable — you'd need a different formula for every combination of predictors interacting with each other.

Matrix notation solves this by writing the *entire* regression problem — one predictor or a hundred — as a **single, unchanging equation**. Learn the matrix form once, and it works identically whether you have 1 predictor or 1,000. This is why every serious textbook (and every real software implementation — `lm()`, `statsmodels`, `sklearn`) works in matrix form internally, even for simple regression.

---

## 3.2 The Model in Matrix Form — Every Symbol Explained

$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon} $$

| Symbol | Shape | Plain-English meaning |
|---|---|---|
| $\mathbf{y}$ | $n \times 1$ | A column stacking all $n$ observed outcomes — one number per row |
| $\mathbf{X}$ | $n \times (p+1)$ | The **design matrix** — one row per observation, one column per predictor, **plus a first column of all 1's** for the intercept |
| $\boldsymbol{\beta}$ | $(p+1) \times 1$ | A column stacking every coefficient: $[\beta_0, \beta_1, ..., \beta_p]^T$ |
| $\boldsymbol{\varepsilon}$ | $n \times 1$ | A column stacking every unobserved error term |

The **column of 1's** is the detail everyone forgets to explain: it exists purely so that matrix multiplication produces $\beta_0 \times 1 = \beta_0$ for every row — it's a notational trick to fold the intercept into the same matrix multiplication as every other coefficient, rather than writing it as a separate "+$\beta_0$" term outside the matrix product.

**Our dataset in matrix form:**

$$ \mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \\ 1 & 5 \end{bmatrix} \qquad \mathbf{y} = \begin{bmatrix} 50 \\ 55 \\ 65 \\ 70 \\ 80 \end{bmatrix} \qquad \boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix} $$

---

## 3.3 The Least Squares Objective, Rewritten in Matrix Form

The RSS from Chapter 1 becomes:

$$ RSS(\boldsymbol{\beta}) = (\mathbf{y}-\mathbf{X}\boldsymbol{\beta})^T(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}) $$

**Why this is the same thing:** $(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})$ is just the column of all residuals stacked up. Multiplying a vector by its own transpose ($\mathbf{v}^T\mathbf{v}$) is exactly the sum of its squared entries — so this single matrix expression *is* $\sum e_i^2$, just written more compactly.

---

## 3.4 Deriving the Normal Equations (Matrix Calculus — Minimal, Necessary Steps)

Expand the objective:

$$ RSS = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} $$

Take the derivative with respect to the vector $\boldsymbol{\beta}$ (using the standard matrix-calculus identities $\frac{\partial}{\partial \boldsymbol{\beta}}(\boldsymbol{\beta}^T\mathbf{a}) = \mathbf{a}$ and $\frac{\partial}{\partial \boldsymbol{\beta}}(\boldsymbol{\beta}^T\mathbf{A}\boldsymbol{\beta}) = 2\mathbf{A}\boldsymbol{\beta}$ for symmetric $\mathbf{A}$), set to zero:

$$ -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = 0 $$

$$ \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y} \quad \text{(the "Normal Equations")} $$

Solving (assuming $\mathbf{X}^T\mathbf{X}$ is invertible — more on when it isn't in Chapter 9 on multicollinearity):

$$ \boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}} $$

**This single formula is the entire content of Chapter 1's $\hat{\beta}_0, \hat{\beta}_1$ derivation, generalized to any number of predictors.** This is arguably the single most important formula in classical statistics — interviewers routinely ask you to write it from memory and explain every piece.

---

## 3.5 Worked Numerical Example — Verifying It Matches Chapter 1

**Step 1 — compute $\mathbf{X}^T\mathbf{X}$:**

$$ \mathbf{X}^T\mathbf{X} = \begin{bmatrix} 1&1&1&1&1 \\ 1&2&3&4&5 \end{bmatrix} \begin{bmatrix} 1&1\\1&2\\1&3\\1&4\\1&5 \end{bmatrix} = \begin{bmatrix} 5 & 15 \\ 15 & 55 \end{bmatrix} $$

(Top-left = $n=5$; top-right/bottom-left = $\sum x_i = 15$; bottom-right = $\sum x_i^2 = 1+4+9+16+25=55$.)

**Step 2 — compute $\mathbf{X}^T\mathbf{y}$:**

$$ \mathbf{X}^T\mathbf{y} = \begin{bmatrix} \sum y_i \\ \sum x_i y_i \end{bmatrix} = \begin{bmatrix} 320 \\ 1035 \end{bmatrix} $$

($\sum y_i = 50+55+65+70+80=320$; $\sum x_iy_i = 50+110+195+280+400=1035$.)

**Step 3 — invert the $2\times2$ matrix $\mathbf{X}^T\mathbf{X}$:**

For a $2\times2$ matrix $\begin{bmatrix}a&b\\c&d\end{bmatrix}$, the inverse is $\frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$.

$$ \det = (5)(55)-(15)(15) = 275-225 = 50 $$

$$ (\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{50}\begin{bmatrix} 55 & -15 \\ -15 & 5 \end{bmatrix} = \begin{bmatrix} 1.1 & -0.3 \\ -0.3 & 0.1 \end{bmatrix} $$

**Step 4 — multiply through:**

$$ \hat{\boldsymbol{\beta}} = \begin{bmatrix} 1.1 & -0.3 \\ -0.3 & 0.1 \end{bmatrix}\begin{bmatrix} 320 \\ 1035 \end{bmatrix} = \begin{bmatrix} 1.1(320) + (-0.3)(1035) \\ -0.3(320)+0.1(1035) \end{bmatrix} = \begin{bmatrix} 352-310.5 \\ -96+103.5 \end{bmatrix} = \begin{bmatrix} 41.5 \\ 7.5 \end{bmatrix} $$

**Exact match to Chapter 1's hand-derived $\hat{\beta}_0=41.5, \hat{\beta}_1=7.5$.** This is the payoff of learning the matrix form: it's not a *different* answer, it's the *same* answer via a method that scales to any number of predictors without inventing new formulas.

---

## 3.6 The Hat Matrix

Once you have $\hat{\boldsymbol{\beta}}$, the fitted values are:

$$ \hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{H}\mathbf{y} $$

where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is called the **hat matrix**, because it "puts a hat on" $\mathbf{y}$ — it's the single matrix that transforms your raw observations directly into fitted values.

**Why this matters beyond notation — three properties every interviewer expects you to know:**

1. **$\mathbf{H}$ is symmetric and idempotent** ($\mathbf{H}\mathbf{H}=\mathbf{H}$) — applying it twice does nothing new, which makes sense: once you've projected onto the fitted-values plane, projecting again changes nothing.
2. **The diagonal entries $h_{ii}$ are called leverage values** — they measure how much influence observation $i$'s own $x_i$ has on its *own* fitted value. High-leverage points (unusual x-values) can disproportionately pull the line toward themselves — this becomes the entire subject of Chapter 8 (Leverage & Influence).
3. **$\text{trace}(\mathbf{H}) = p$**, the number of estimated parameters — always, exactly. (For our example, $p=2$, so the leverage values across all 5 students must sum to exactly 2.)

The residual vector can also be written compactly as $\mathbf{e} = (\mathbf{I}-\mathbf{H})\mathbf{y}$ — "everything left over after removing what the hat matrix explains."

---

## 3.7 Reconnecting to the Geometric Picture from Chapter 1

Chapter 1, §1.7 described OLS as an orthogonal projection without the matrix language to make it precise. Now it can be stated exactly: $\mathbf{H}$ **is** the projection matrix onto the column space of $\mathbf{X}$ (the subspace spanned by the intercept column and every predictor column). $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ is literally the orthogonal projection of $\mathbf{y}$ onto that subspace, and $\mathbf{e} = \mathbf{y}-\hat{\mathbf{y}}$ is, by the geometry of orthogonal projection, guaranteed perpendicular to every column of $\mathbf{X}$ — which is exactly why $\sum e_i = 0$ and $\sum e_i x_i = 0$ held automatically back in Chapter 1.

*(Diagram to visualize: a 3D sketch showing the vector $\mathbf{y}$ sticking up out of a 2D plane (the column space of $\mathbf{X}$), with a dashed perpendicular line dropping straight down from the tip of $\mathbf{y}$ to its "shadow" $\hat{\mathbf{y}}$ on the plane — that dashed segment is the residual vector $\mathbf{e}$.)*

---

## 3.8 Variance-Covariance Matrix of $\hat{\boldsymbol{\beta}}$

The matrix generalization of Chapter 2's $SE(\hat{\beta}_1) = \sqrt{\sigma^2/S_{xx}}$ is:

$$ \text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1} $$

This single $(p+1)\times(p+1)$ matrix contains **every** variance and covariance you'll ever need: the diagonal entries are $\text{Var}(\hat{\beta}_0), \text{Var}(\hat{\beta}_1), ...$ (their square roots are exactly the standard errors used in every t-test and CI from Chapter 2), and the off-diagonal entries tell you how correlated your coefficient estimates are with each other — a preview of why multicollinearity (Chapter 9) causes coefficient estimates to become unstable: it inflates these off-diagonal terms.

**Worked check:** $\text{Var}(\hat{\beta}_1) = \sigma^2 \times (\mathbf{X}^T\mathbf{X})^{-1}_{22} = \sigma^2 \times 0.1$. Using $\hat{\sigma}^2=MSE=2.5$ from Chapter 2: $\text{Var}(\hat{\beta}_1)=2.5\times0.1=0.25$, so $SE(\hat{\beta}_1)=\sqrt{0.25}=0.5$ — **exactly matching Chapter 2's hand-derived standard error.**

---

## 3.9 Where the Textbooks Differ

- **Kutner** introduces matrix notation relatively late and treats it as a compact restatement of results already derived with summation notation — a "translation," not new content.
- **Montgomery** leans on matrix form earlier and more heavily, especially once multiple regression and multicollinearity diagnostics (VIF, condition numbers — both matrix-derived quantities) enter the picture in later chapters.
- **ESL/ISL** assume matrix fluency from the start and barely pause to explain the normal equations derivation at all — they treat $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ as a known starting point, which is part of why this chapter slows down to derive it explicitly rather than assuming it.
- **Sheather** ties the matrix form directly to software output — e.g., how `lm()` internally computes and reports the variance-covariance matrix — bridging theory to the regression summary tables you'll actually see in practice.

---

## 3.10 Interview Q&A

**Q: Write the OLS estimator formula and explain every term.**
A: $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. $\mathbf{X}$ is the design matrix (with a leading column of 1's for the intercept), $\mathbf{y}$ the response vector, and this formula is the closed-form minimizer of the sum of squared residuals.

**Q: What happens if $\mathbf{X}^T\mathbf{X}$ is not invertible?**
A: This happens when predictor columns are perfectly collinear (one column is an exact linear combination of others) or when $n < p$ (fewer observations than parameters). No unique OLS solution exists; remedies include dropping redundant predictors, using regularization (ridge regression adds $\lambda I$ to force invertibility — Chapter 16), or using the Moore-Penrose pseudoinverse.

**Q: What is the hat matrix, and what does its diagonal represent?**
A: $\mathbf{H}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is the matrix that projects $\mathbf{y}$ onto its fitted values $\hat{\mathbf{y}}$. Its diagonal entries are leverage values — how much each observation's own predictor value influences its own fitted value; unusually high leverage flags a point that could disproportionately pull the fitted line.

**Q: Why is $\mathbf{H}$ idempotent, and why does that matter?**
A: $\mathbf{H}\mathbf{H}=\mathbf{H}$ because it's a projection — projecting an already-projected point changes nothing. This property is used to derive the exact distribution of SSE and to prove $\mathbf{H}$'s trace equals the number of parameters.

**Q: How does the variance-covariance matrix of $\hat{\boldsymbol{\beta}}$ relate to multicollinearity?**
A: $\text{Var}(\hat{\boldsymbol{\beta}})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$. When predictors are highly correlated, $\mathbf{X}^T\mathbf{X}$ becomes nearly singular (close to non-invertible), which inflates the diagonal entries of its inverse — meaning wildly unstable, high-variance coefficient estimates. This is the mathematical root of the multicollinearity problem covered in Chapter 9.

---

*End of Chapter 3. Next: Chapter 4 — Multiple Linear Regression (extending the design matrix to multiple predictors, interpreting coefficients as "partial effects holding other predictors constant," and the first real multi-predictor worked example).*
