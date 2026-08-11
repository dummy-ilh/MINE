# Chapter 3 — Matrix Formulation of Regression (Revised)

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL — same 5-student dataset (x = hours studied, y = exam score) as always, so every number below reproduces the $\hat{\beta}_0=41.5$, $\hat{\beta}_1=7.5$ you've already hand-derived twice.*

This version adds: a notation cheat-sheet with memory hooks up front, a fully worked numerical hat matrix (not just the abstract formula), a leverage worked example, and a second mini-example with 3 predictors so you see the machinery generalize before Chapter 4.

---

## 3.0 Notation Cheat-Sheet (read this first, refer back to it constantly)

Matrix regression notation is 90% of the battle. If you can hold this table in your head, the rest of the chapter is just arithmetic.

| Symbol | Say it as | Shape | Memory hook |
|---|---|---|---|
| $\mathbf{y}$ | "y" | $n\times1$ | The actual exam scores, stacked in a column. Lowercase bold = a column of numbers. |
| $\mathbf{X}$ | "X" (design matrix) | $n\times(p+1)$ | One **row per student**, one **column per predictor** (plus a leading column of 1's). Uppercase bold = a full table of numbers. |
| $\boldsymbol{\beta}$ | "beta vector" | $(p+1)\times1$ | The coefficients you're solving for, stacked: $[\beta_0,\beta_1,...]^T$. |
| $\boldsymbol{\varepsilon}$ | "epsilon vector" | $n\times1$ | The unobservable noise — one per student. You never see this, only estimate its size via residuals. |
| $\hat{\boldsymbol{\beta}}$ | "beta hat" | $(p+1)\times1$ | A **hat (^)** over any symbol always means "the estimate of," never the true value. |
| $\hat{\mathbf{y}}$ | "y hat" | $n\times1$ | The *fitted* values — what the line predicts, not what was observed. |
| $\mathbf{e}$ | "residual vector" | $n\times1$ | $\mathbf{y}-\hat{\mathbf{y}}$. Lowercase, no hat — it's a direct computation, not an estimate of something unknown. |
| $\mathbf{H}$ | "hat matrix" | $n\times n$ | The one matrix that turns $\mathbf{y}$ into $\hat{\mathbf{y}}$ — literally the machine that "puts the hat on." |
| $\mathbf{X}^T$ | "X transpose" | $(p+1)\times n$ | Flip rows and columns. Needed because you can't multiply $\mathbf{X}$ by itself — the shapes don't line up. |
| $(\mathbf{X}^T\mathbf{X})^{-1}$ | "X-transpose-X inverse" | $(p+1)\times(p+1)$ | The matrix "division" — matrices don't have division, so inverting is how you undo a multiplication. |

**Two memory tricks that make the big formula stick:**

1. **The OLS formula is a "sandwich":** $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. Notice $\mathbf{X}^T$ appears **twice**, $\mathbf{y}$ appears **once**, and there's exactly **one inverse**. If you ever forget the formula mid-interview, rebuild it from the shapes: you need something $(p+1)\times1$ out, you have $\mathbf{y}$ ($n\times1$) and $\mathbf{X}$ ($n\times(p+1)$) to work with — the only combination of transposes/inverses that produces the right shape *is* the sandwich. Shape-checking is a legitimate way to reconstruct a forgotten formula on a whiteboard.
2. **The column of 1's is an "always-on switch."** Every other column in $\mathbf{X}$ varies row to row (different $x_i$ per student). The 1's column never changes — it's on for every single row — which is exactly why multiplying it by $\beta_0$ gives every student the *same* baseline contribution, i.e., the intercept.

---

## 3.1 The Motivating Question

Chapters 1–2 used summation notation ($\sum$) because simple linear regression only has one predictor. The moment you have **two or more predictors**, summation notation becomes unmanageable — a different formula for every combination of predictors interacting.

Matrix notation solves this by writing the *entire* regression problem — 1 predictor or 1,000 — as a **single, unchanging equation**. Learn it once, and it's identical whether you have 1 predictor or 100. This is why every real implementation (`lm()`, `statsmodels`, `sklearn`) works in matrix form internally, even for simple regression.

---

## 3.2 The Model in Matrix Form

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

**Our dataset in matrix form:**

$$\mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \\ 1 & 5 \end{bmatrix} \qquad \mathbf{y} = \begin{bmatrix} 50 \\ 55 \\ 65 \\ 70 \\ 80 \end{bmatrix} \qquad \boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}$$

Read $\mathbf{X}$ row-by-row: row 3 is `[1, 3]` — student 3 studied 3 hours, and the leading 1 activates the intercept.

---

## 3.3 The Least Squares Objective, in Matrix Form

$$RSS(\boldsymbol{\beta}) = (\mathbf{y}-\mathbf{X}\boldsymbol{\beta})^T(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})$$

**Why this is the same thing:** $(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})$ is the column of all residuals stacked up. Multiplying a vector by its own transpose ($\mathbf{v}^T\mathbf{v}$) is exactly the sum of its squared entries — so this single expression *is* $\sum e_i^2$, just written compactly.

**Tiny concrete check with $2\times1$ vectors** (not our real data — just to see the mechanic): if $\mathbf{v}=[3,4]^T$, then $\mathbf{v}^T\mathbf{v} = 3\times3+4\times4=25 = 3^2+4^2$. That's all $\mathbf{v}^T\mathbf{v}$ ever does — "multiply matching entries, then add."

---

## 3.4 Deriving the Normal Equations

Expand the objective:

$$RSS = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

Take the derivative with respect to the vector $\boldsymbol{\beta}$ (using $\frac{\partial}{\partial \boldsymbol{\beta}}(\boldsymbol{\beta}^T\mathbf{a}) = \mathbf{a}$ and $\frac{\partial}{\partial \boldsymbol{\beta}}(\boldsymbol{\beta}^T\mathbf{A}\boldsymbol{\beta}) = 2\mathbf{A}\boldsymbol{\beta}$ for symmetric $\mathbf{A}$), set to zero:

$$-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = 0 \quad\Rightarrow\quad \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

Solving (assuming $\mathbf{X}^T\mathbf{X}$ is invertible — more in Chapter 9 on multicollinearity):

$$\boxed{\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

This single formula is Chapter 1's $\hat{\beta}_0, \hat{\beta}_1$ derivation, generalized to any number of predictors. Interviewers routinely ask you to write it from memory and explain every piece — that's exactly what §3.0's cheat-sheet is for.

---

## 3.5 Worked Numerical Example — Verifying It Matches Chapter 1

**Step 1 — compute $\mathbf{X}^T\mathbf{X}$:**

$$\mathbf{X}^T\mathbf{X} = \begin{bmatrix} 1&1&1&1&1 \\ 1&2&3&4&5 \end{bmatrix} \begin{bmatrix} 1&1\\1&2\\1&3\\1&4\\1&5 \end{bmatrix} = \begin{bmatrix} 5 & 15 \\ 15 & 55 \end{bmatrix}$$

Top-left = $n=5$; top-right/bottom-left = $\sum x_i = 15$; bottom-right = $\sum x_i^2 = 1+4+9+16+25=55$.

**Step 2 — compute $\mathbf{X}^T\mathbf{y}$:**

$$\mathbf{X}^T\mathbf{y} = \begin{bmatrix} \sum y_i \\ \sum x_i y_i \end{bmatrix} = \begin{bmatrix} 320 \\ 1035 \end{bmatrix}$$

**Step 3 — invert the $2\times2$ matrix:**

$$\det = (5)(55)-(15)(15) = 50 \qquad (\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{50}\begin{bmatrix} 55 & -15 \\ -15 & 5 \end{bmatrix} = \begin{bmatrix} 1.1 & -0.3 \\ -0.3 & 0.1 \end{bmatrix}$$

**Step 4 — multiply through:**

$$\hat{\boldsymbol{\beta}} = \begin{bmatrix} 1.1 & -0.3 \\ -0.3 & 0.1 \end{bmatrix}\begin{bmatrix} 320 \\ 1035 \end{bmatrix} = \begin{bmatrix} 41.5 \\ 7.5 \end{bmatrix}$$

**Exact match to Chapter 1.** Same answer, method that scales to any number of predictors.

---

## 3.6 The Hat Matrix — Fully Worked, Not Just the Formula

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = \mathbf{H}\mathbf{y}, \qquad \mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$

Last time this chapter stated $\mathbf{H}$'s properties abstractly. This time, let's actually **compute $\mathbf{H}$ for our 5 students** and watch the properties fall out of real numbers.

**Step 1 — compute $\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}$ (a $5\times2$ matrix).** For row $i$ with predictor value $x_i$, the row is $[1, x_i]\begin{bmatrix}1.1&-0.3\\-0.3&0.1\end{bmatrix} = [1.1-0.3x_i,\ -0.3+0.1x_i]$:

| Student | $x_i$ | Row of $\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}$ |
|---|---|---|
| 1 | 1 | $[0.8,\ -0.2]$ |
| 2 | 2 | $[0.5,\ -0.1]$ |
| 3 | 3 | $[0.2,\ \ 0.0]$ |
| 4 | 4 | $[-0.1,\ 0.1]$ |
| 5 | 5 | $[-0.4,\ 0.2]$ |

**Step 2 — multiply by $\mathbf{X}^T$ to get any entry $H_{ij}$.** The formula collapses to $H_{ij} = a_i + b_i x_j$, where $[a_i,b_i]$ is student $i$'s row above and $x_j$ is student $j$'s hours.

**The diagonal (leverage values), $H_{ii}$:**

$$H_{11}=0.8-0.2(1)=0.6 \quad H_{22}=0.5-0.1(2)=0.3 \quad H_{33}=0.2+0(3)=0.2$$
$$H_{44}=-0.1+0.1(4)=0.3 \quad H_{55}=-0.4+0.2(5)=0.6$$

Add them up: $0.6+0.3+0.2+0.3+0.6 = \mathbf{2.0}$ — exactly $p=2$ (intercept + 1 slope), confirming $\text{trace}(\mathbf{H})=p$ with real numbers instead of just asserting it.

**Notice the U-shape:** students 1 and 5 (the extremes — least and most hours studied) have the *highest* leverage (0.6), while student 3 (dead center, $x=3$ is the mean of $\{1,2,3,4,5\}$) has the *lowest* (0.2). This is the general pattern: **leverage grows the farther a predictor value sits from the mean of $x$.** A student who studied 1 hour or 5 hours has more power to tilt the fitted line than a student sitting right at the average — that's the intuition behind "leverage," made concrete.

**An off-diagonal entry, to see symmetry directly:**

$$H_{12} = a_1 + b_1 x_2 = 0.8 - 0.2(2) = 0.4 \qquad H_{21} = a_2 + b_2 x_1 = 0.5 - 0.1(1) = 0.4$$

$H_{12}=H_{21}=0.4$ — not a coincidence, this is $\mathbf{H}$'s symmetry property showing up in actual arithmetic.

**Using $\mathbf{H}$ to get a fitted value directly from row 1** ($H_{11}=0.6,H_{12}=0.4,H_{13}=0.2,H_{14}=0.0,H_{15}=-0.2$ — computed the same way):

$$\hat{y}_1 = \sum_j H_{1j}y_j = 0.6(50)+0.4(55)+0.2(65)+0.0(70)-0.2(80) = 30+22+13+0-16 = 49$$

Cross-check against the regular regression line: $\hat{y}_1 = 41.5+7.5(1) = 49$. **Same answer**, arrived at by a completely different route — this is what "$\hat{\mathbf{y}}=\mathbf{H}\mathbf{y}$" *means* in practice: $\mathbf{H}$ is a fixed weighting scheme, built once from $\mathbf{X}$ alone, that turns raw $y$-values into fitted values without ever re-solving for $\hat{\boldsymbol{\beta}}$.

**Three properties, now grounded in the numbers above:**

1. **Symmetric & idempotent** ($\mathbf{H}\mathbf{H}=\mathbf{H}$): $H_{12}=H_{21}=0.4$ demonstrated symmetry directly. Idempotence means applying $\mathbf{H}$ twice does nothing new — once $\mathbf{y}$ has been projected onto the fitted-values line, projecting *that* result again just gives the same line back.
2. **Diagonal = leverage**: computed above (0.6, 0.3, 0.2, 0.3, 0.6) — U-shaped, smallest at the mean of $x$, largest at the extremes.
3. **$\text{trace}(\mathbf{H})=p$**: verified as $0.6+0.3+0.2+0.3+0.6=2$, matching $p=2$ parameters exactly.

The residual vector is $\mathbf{e}=(\mathbf{I}-\mathbf{H})\mathbf{y}$ — "everything left over after removing what the hat matrix explains." Quick check for student 1: $e_1 = y_1-\hat{y}_1 = 50-49=1$.

---

## 3.7 Reconnecting to the Geometric Picture from Chapter 1

Chapter 1, §1.7 described OLS as an orthogonal projection without matrix language. Now it's exact: $\mathbf{H}$ **is** the projection matrix onto the column space of $\mathbf{X}$ (the subspace spanned by the intercept column and every predictor column). $\hat{\mathbf{y}}=\mathbf{H}\mathbf{y}$ is literally the orthogonal projection of $\mathbf{y}$ onto that subspace, and $\mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}$ is, by the geometry of orthogonal projection, guaranteed perpendicular to every column of $\mathbf{X}$ — exactly why $\sum e_i=0$ and $\sum e_i x_i=0$ held automatically back in Chapter 1.

```
        y  (actual scores — a point up in 5-D space)
        |\
        | \
        |  \  e  (residual — the dashed perpendicular drop)
        |   \
        |    \
--------+-----\----------------  <- column space of X
        yhat    (the "shadow" of y on the plane spanned
                 by the 1's column and the x column)
```

---

## 3.8 Variance-Covariance Matrix of $\hat{\boldsymbol{\beta}}$

The matrix generalization of Chapter 2's $SE(\hat{\beta}_1)=\sqrt{\sigma^2/S_{xx}}$:

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$$

This single $(p+1)\times(p+1)$ matrix contains every variance and covariance you'll need: diagonal entries are $\text{Var}(\hat{\beta}_0),\text{Var}(\hat{\beta}_1),...$ (their square roots are the standard errors used in every t-test/CI from Chapter 2); off-diagonal entries tell you how correlated your coefficient estimates are — a preview of why multicollinearity (Chapter 9) makes coefficients unstable: it inflates these off-diagonal terms.

**Worked check:** $\text{Var}(\hat{\beta}_1)=\sigma^2\times(\mathbf{X}^T\mathbf{X})^{-1}_{22}=\sigma^2\times0.1$. Using $\hat{\sigma}^2=MSE=2.5$ from Chapter 2: $\text{Var}(\hat{\beta}_1)=2.5\times0.1=0.25$, so $SE(\hat{\beta}_1)=\sqrt{0.25}=0.5$ — matching Chapter 2's hand-derived standard error exactly.

---

## 3.9 A Second Mini-Example — Adding a Predictor, So You See It Generalize

Everything above used $p=1$ predictor (hours studied). Here's the payoff of matrix notation: **add a second predictor and nothing about the method changes** — only $\mathbf{X}$ grows a column.

Suppose we also track "practice tests taken" ($x_2$) for the same 5 students: `[0, 1, 1, 2, 3]`.

$$\mathbf{X} = \begin{bmatrix} 1&1&0\\1&2&1\\1&3&1\\1&4&2\\1&5&3 \end{bmatrix}$$

Now $\mathbf{X}$ is $5\times3$ (intercept + 2 predictors), so $\mathbf{X}^T\mathbf{X}$ is $3\times3$, $\hat{\boldsymbol{\beta}}=[\beta_0,\beta_1,\beta_2]^T$, and $\mathbf{H}$ is still $5\times5$ but now with $\text{trace}(\mathbf{H})=p=3$. **The formula $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ is untouched** — you'd just be inverting a $3\times3$ instead of a $2\times2$. This is precisely the scaling property §3.1 promised, and it's the whole subject of Chapter 4 (Multiple Linear Regression), where this exact matrix gets solved and interpreted.

---

## 3.10 Where the Textbooks Differ

- **Kutner** introduces matrix notation relatively late, treating it as a compact restatement of results already derived with summation notation — a "translation," not new content.
- **Montgomery** leans on matrix form earlier and more heavily, especially once multicollinearity diagnostics (VIF, condition numbers — both matrix-derived) enter later chapters.
- **ESL/ISL** assume matrix fluency from the start and barely pause on the normal-equations derivation — treating $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ as a known starting point, which is why this chapter slows down to derive it explicitly.
- **Sheather** ties the matrix form directly to software output — e.g. how `lm()` internally computes and reports the variance-covariance matrix — bridging theory to the regression summary tables you'll actually see.

---

## 3.11 Interview Q&A

**Q: Write the OLS estimator formula and explain every term.**
A: $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$. $\mathbf{X}$ is the design matrix (leading column of 1's for the intercept), $\mathbf{y}$ the response vector — the closed-form minimizer of the sum of squared residuals. (Rebuild it under pressure using the "sandwich" shape-check from §3.0 if you blank.)

**Q: What happens if $\mathbf{X}^T\mathbf{X}$ is not invertible?**
A: Happens when predictor columns are perfectly collinear (one column is an exact linear combination of others) or $n<p$ (fewer observations than parameters). No unique OLS solution exists; remedies: drop redundant predictors, use regularization (ridge adds $\lambda I$ to force invertibility — Chapter 16), or the Moore-Penrose pseudoinverse.

**Q: What is the hat matrix, and what does its diagonal represent?**
A: $\mathbf{H}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$, projecting $\mathbf{y}$ onto its fitted values. Diagonal entries are leverage — how much an observation's own predictor value influences its own fitted value. Leverage rises the farther a point sits from the mean of $x$ (concretely shown in §3.6: the two extreme students had leverage 0.6, the centered student had 0.2).

**Q: Why is $\mathbf{H}$ idempotent, and why does that matter?**
A: $\mathbf{H}\mathbf{H}=\mathbf{H}$ because it's a projection — projecting an already-projected point changes nothing. Used to derive the exact distribution of SSE and to prove $\mathbf{H}$'s trace equals the number of parameters.

**Q: How does the variance-covariance matrix of $\hat{\boldsymbol{\beta}}$ relate to multicollinearity?**
A: $\text{Var}(\hat{\boldsymbol{\beta}})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$. Highly correlated predictors make $\mathbf{X}^T\mathbf{X}$ nearly singular, inflating the diagonal of its inverse — wildly unstable, high-variance coefficient estimates. The mathematical root of the multicollinearity problem in Chapter 9.

**Q: If you added a third predictor tomorrow, what would actually change in this chapter's formulas?**
A: Only the *shape* of $\mathbf{X}$ (one more column) and $\boldsymbol{\beta}$ (one more entry) — every formula ($\hat{\boldsymbol{\beta}}$, $\mathbf{H}$, $\text{Var}(\hat{\boldsymbol{\beta}})$) is unchanged. That's the entire point of matrix notation, demonstrated concretely in §3.9.

---

*End of Chapter 3 (revised). Next: Chapter 4 — Multiple Linear Regression (extending the design matrix to multiple predictors, interpreting coefficients as "partial effects holding other predictors constant," and the first real multi-predictor worked example — picking up exactly where §3.9 left off).*
