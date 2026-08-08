# Chapter 4 — Multiple Linear Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. We extend the running dataset with a second predictor so every idea below has a concrete, hand-checkable anchor.*

**Extended dataset** — same 5 students, now with a second predictor ($x_2$ = practice tests taken):

| Student | $x_1$ (hours studied) | $x_2$ (practice tests) | $y$ (exam score) |
|---|---|---|---|
| 1 | 1 | 1 | 50 |
| 2 | 2 | 1 | 55 |
| 3 | 3 | 2 | 65 |
| 4 | 4 | 2 | 70 |
| 5 | 5 | 3 | 80 |

---

## 4.1 The Motivating Question

Chapters 1–3 asked "how does $y$ change with $x$?" using a single predictor. But almost nothing interesting in the real world is caused by just one variable. Exam score plausibly depends on hours studied **and** practice tests taken **and** sleep **and** dozens of other things.

**The core question multiple regression answers:** "Holding everything else in the model fixed, what's the effect of *this one* predictor, isolated from the others?" That isolating power — separating out one variable's effect from the rest — is the entire reason multiple regression exists, and it's the source of almost every subtlety in this chapter.

---

## 4.2 The Model — Extending the Symbols You Already Know

$$ y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_p x_{ip} + \varepsilon_i $$

Nothing here is new notation — it's Chapter 1's model with more $\beta x$ terms added. In matrix form (Chapter 3), this is still exactly $\mathbf{y}=\mathbf{X}\boldsymbol{\beta}+\boldsymbol{\varepsilon}$, just with $\mathbf{X}$ now having 3 columns (intercept, $x_1$, $x_2$) instead of 2. **Everything from Chapter 3 — the normal equations, $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, the hat matrix — applies completely unchanged.** This is the payoff of having learned the matrix form first: multiple regression isn't new math, it's the same math with a wider matrix.

---

## 4.3 The Critical Shift in Interpretation — "Partial Effects"

This is the single most important conceptual idea in the entire chapter, and the one interviewers probe hardest.

In **simple** regression, $\hat{\beta}_1$ means: "for a 1-unit increase in $x$, $y$ changes by $\hat{\beta}_1$ — full stop, nothing else being accounted for."

In **multiple** regression, $\hat{\beta}_1$ means: "for a 1-unit increase in $x_1$, **holding $x_2$ (and every other predictor in the model) constant**, $y$ changes by $\hat{\beta}_1$." This is often called the **partial effect** or **ceteris paribus** ("all else equal") interpretation.

**Why this distinction matters enormously:** if $x_1$ and $x_2$ are correlated with each other (e.g., students who study more hours also tend to take more practice tests), the simple-regression coefficient on $x_1$ alone will be *contaminated* — it silently absorbs some of $x_2$'s effect too, because it has no way to isolate them. The multiple-regression coefficient strips that contamination out, *provided* $x_2$ is actually in the model. This single idea is the entire basis of **omitted variable bias**, which we return to properly in Chapter 24, but the intuition starts here.

---

## 4.4 Worked Example — Fitting by the Matrix Method

**Step 1 — build $\mathbf{X}^T\mathbf{X}$** (a $3\times3$ matrix now, since there are 3 parameters: $\beta_0,\beta_1,\beta_2$):

$$ \mathbf{X}^T\mathbf{X} = \begin{bmatrix} n & \sum x_1 & \sum x_2 \\ \sum x_1 & \sum x_1^2 & \sum x_1 x_2 \\ \sum x_2 & \sum x_1 x_2 & \sum x_2^2 \end{bmatrix} = \begin{bmatrix} 5 & 15 & 9 \\ 15 & 55 & 32 \\ 9 & 32 & 19 \end{bmatrix} $$

**Step 2 — build $\mathbf{X}^T\mathbf{y}$:**

$$ \mathbf{X}^T\mathbf{y} = \begin{bmatrix} \sum y \\ \sum x_1 y \\ \sum x_2 y \end{bmatrix} = \begin{bmatrix} 320 \\ 1035 \\ 615 \end{bmatrix} $$

**Step 3 — solve $\mathbf{X}^T\mathbf{X}\,\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$** (by substitution rather than inverting a $3\times3$ by hand — the same answer either way):

From row 1: $5\hat{\beta}_0+15\hat{\beta}_1+9\hat{\beta}_2=320 \Rightarrow \hat{\beta}_0 = 64-3\hat{\beta}_1-1.8\hat{\beta}_2$

Substituting into row 2 and row 3 and solving the resulting $2\times2$ system (algebra omitted — same substitution technique as Chapter 1) gives:

$$ \hat{\beta}_0 = 40, \qquad \hat{\beta}_1 = 5, \qquad \hat{\beta}_2 = 5 $$

**Fitted equation:** $\hat{y} = 40 + 5x_1 + 5x_2$

**Verification** (plug each row back in):

| Student | $40+5x_1+5x_2$ | Actual $y$ | Residual |
|---|---|---|---|
| 1 | $40+5+5=50$ | 50 | 0 |
| 2 | $40+10+5=55$ | 55 | 0 |
| 3 | $40+15+10=65$ | 65 | 0 |
| 4 | $40+20+10=70$ | 70 | 0 |
| 5 | $40+25+15=80$ | 80 | 0 |

**Important honesty note:** this toy dataset was constructed so the fit is exact (SSE = 0) — useful for cleanly verifying the matrix arithmetic, but real data essentially never fits this perfectly. Chapter 5 (inference in multiple regression) will use a version of this dataset with realistic noise added, since you need nonzero residuals to compute meaningful standard errors and F-tests.

---

## 4.5 Reading the Coefficients Correctly

$\hat{\beta}_1=5$: **holding practice tests fixed**, each additional hour studied is associated with a 5-point increase in exam score.

$\hat{\beta}_2=5$: **holding hours studied fixed**, each additional practice test taken is associated with a 5-point increase in exam score.

Notice these numbers differ from Chapter 1's simple-regression slope of $\hat{\beta}_1=7.5$ (studying alone, ignoring practice tests). That drop from 7.5 to 5 is exactly the phenomenon described in §4.3: some of the "credit" that simple regression gave entirely to hours-studied was actually attributable to practice tests, since the two predictors move together in this data ($x_1$ and $x_2$ are correlated — students who study more also tend to take more practice tests). Multiple regression correctly splits that credit between the two predictors; simple regression could not.

---

## 4.6 A Geometric Upgrade

Chapter 1's picture was a 2D line through a scatter of points. Chapter 3 upgraded this to an orthogonal projection. With two predictors, the fitted "line" becomes a **plane** slicing through 3D space (one axis for $x_1$, one for $x_2$, one for $y$) — and with $p$ predictors, a $p$-dimensional **hyperplane**. The hat matrix $\mathbf{H}$ from Chapter 3 still projects $\mathbf{y}$ orthogonally onto this hyperplane, completely unchanged in concept — only the dimension of the space it projects onto has grown.

*(Diagram to visualize: a 3D scatter plot with $x_1$ and $x_2$ as the two horizontal axes and $y$ as the vertical axis, with a tilted flat plane cutting through the cloud of points — that plane is the fitted multiple regression surface, replacing the single line from Chapter 1.)*

---

## 4.7 A Cautionary Note: Adding Predictors Isn't Free

It might seem like the fix for every regression problem is "just add more predictors." Two reasons this is wrong, previewed here and covered fully later:

- **$R^2$ mechanically never decreases** when you add any predictor — even pure random noise — so $R^2$ alone can never tell you whether a new predictor genuinely helps (Chapter 14, adjusted $R^2$).
- **Multicollinearity** — if $x_1$ and $x_2$ are too strongly correlated with each other, $\mathbf{X}^T\mathbf{X}$ becomes close to non-invertible (Chapter 3, §3.8), and coefficient estimates become unstable and hard to interpret individually, even if the model's overall predictions remain fine (Chapter 9).

---

## 4.8 Where the Textbooks Differ

- **Kutner** introduces multiple regression very formally, immediately generalizing the ANOVA-table framework from simple regression — heavy on notation, systematic, proof-first.
- **Montgomery** spends much more time here on the *practical meaning* of partial-effect coefficients and gives worked examples specifically designed to show how coefficients shift when predictors are added or removed — closest in spirit to §4.5 above.
- **ESL/ISL** treat multiple regression almost as a solved formality (a special, most-interpretable case of the broader linear-model framework) and rush toward variable selection and regularization — this is *their* chapter 3, essentially a bridge chapter, not a destination.
- **Sheather** emphasizes reading multiple-regression software output directly — showing how the coefficient table, individual t-tests, and overall F-test (Chapter 5 territory) all appear together in a single `lm()` summary.

---

## 4.9 Interview Q&A

**Q: What does a coefficient mean in multiple regression, precisely?**
A: The expected change in $y$ for a one-unit increase in that predictor, **holding all other predictors in the model constant** — the "partial effect," not the total unconditional effect you'd get from a simple regression on that variable alone.

**Q: Why can a coefficient's sign or magnitude change when you add a new predictor to the model?**
A: Because the original coefficient was previously absorbing some of the added predictor's effect (assuming the predictors are correlated). Once the new predictor is explicitly included, the earlier coefficient is "purified" to reflect only its own partial effect.

**Q: Does adding more predictors always improve the model?**
A: It always increases (or leaves unchanged) $R^2$ and never increases SSE, but that doesn't mean the new predictor is meaningful — it could be pure noise. Judging true improvement requires adjusted $R^2$, information criteria (AIC/BIC), or out-of-sample validation (Chapter 14).

**Q: What happens to the matrix formula $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ when you move from 1 to $p$ predictors?**
A: Nothing changes in the formula itself — only the shape of $\mathbf{X}$ grows (more columns). This is exactly why Chapter 3 was worth learning in matrix form before this chapter.

**Q: If two predictors are highly correlated with each other, what happens to their individual coefficients?**
A: Their estimates become unstable (high variance) — small changes in the data can swing the individual coefficients substantially, even though the *combined* predictive fit of the model stays reasonably stable. This is the multicollinearity problem, covered fully in Chapter 9.

---

*End of Chapter 4. Next: Chapter 5 — Inference in Multiple Regression (individual t-tests on each coefficient, the overall F-test for joint significance, partial F-tests for comparing nested models, and why these three tests answer three genuinely different questions).*
