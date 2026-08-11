# Chapter 4 — Multiple Linear Regression

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. We extend the running dataset with a second predictor so every idea below has a concrete, hand-checkable anchor.*
# Chapter 4 — Multiple Linear Regression (Revised)

*Synthesized from Kutner, Montgomery, Sheather, and ESL/ISL. Same running dataset, now with a second predictor.*

**Extended dataset** — 5 students, $x_1$ = hours studied, $x_2$ = practice tests taken, $y$ = exam score:

| Student | $x_1$ | $x_2$ | $y$ |
|---|---|---|---|
| 1 | 1 | 1 | 50 |
| 2 | 2 | 1 | 55 |
| 3 | 3 | 2 | 65 |
| 4 | 4 | 2 | 70 |
| 5 | 5 | 3 | 80 |

---

## 4.0 Quick Notation Refresher (carried over from Chapter 3, §3.0)

With $p=2$ predictors, $\mathbf{X}$ is $5\times3$: a leading column of 1's (intercept), then a column for $x_1$, then a column for $x_2$. $\boldsymbol{\beta}=[\beta_0,\beta_1,\beta_2]^T$ is now $3\times1$. Nothing else about the machinery changes — same sandwich formula $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, just wider matrices.

---

## 4.1 The Motivating Question

Chapters 1–3 asked "how does $y$ change with $x$?" using a single predictor. Almost nothing interesting in the real world is caused by just one variable. Exam score plausibly depends on hours studied **and** practice tests **and** sleep **and** dozens of other things.

**The core question multiple regression answers:** "Holding everything else in the model fixed, what's the effect of *this one* predictor, isolated from the others?" That isolating power is the entire reason multiple regression exists, and it's the source of almost every subtlety in this chapter.

---

## 4.2 The Model

$$ y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + ... + \beta_p x_{ip} + \varepsilon_i $$

Nothing new — it's Chapter 1's model with more $\beta x$ terms. In matrix form it's still $\mathbf{y}=\mathbf{X}\boldsymbol{\beta}+\boldsymbol{\varepsilon}$, just with $\mathbf{X}$ now having 3 columns instead of 2. **Everything from Chapter 3 — normal equations, $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, the hat matrix — applies completely unchanged.**

---

## 4.3 The Critical Shift in Interpretation — "Partial Effects"

The single most important conceptual idea in this chapter.

In **simple** regression, $\hat{\beta}_1$ means: "for a 1-unit increase in $x$, $y$ changes by $\hat{\beta}_1$ — nothing else accounted for."

In **multiple** regression, $\hat{\beta}_1$ means: "for a 1-unit increase in $x_1$, **holding $x_2$ constant**, $y$ changes by $\hat{\beta}_1$." This is the **partial effect** or **ceteris paribus** ("all else equal") interpretation.

**Why this matters enormously:** if $x_1$ and $x_2$ are correlated (students who study more also tend to take more practice tests), the simple-regression coefficient on $x_1$ alone is *contaminated* — it silently absorbs some of $x_2$'s effect too, with no way to isolate them. Multiple regression strips that contamination out, *provided* $x_2$ is actually in the model. This is the seed of **omitted variable bias** (Chapter 24) — we'll see it numerically in §4.6.

---

## 4.4 Building $\mathbf{X}^T\mathbf{X}$ From Scratch — Entry by Entry

This is the part last version glossed over. Here's exactly where every number in that $3\times3$ matrix comes from.

**Step 0 — write out $\mathbf{X}$ with its three columns labeled:**

$$\mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 1 \\ 1 & 3 & 2 \\ 1 & 4 & 2 \\ 1 & 5 & 3 \end{bmatrix} = \begin{bmatrix} | & | & | \\ \mathbf{c}_0 & \mathbf{c}_1 & \mathbf{c}_2 \\ | & | & | \end{bmatrix}, \quad \mathbf{c}_0=\begin{bmatrix}1\\1\\1\\1\\1\end{bmatrix},\ \mathbf{c}_1=\begin{bmatrix}1\\2\\3\\4\\5\end{bmatrix},\ \mathbf{c}_2=\begin{bmatrix}1\\1\\2\\2\\3\end{bmatrix}$$

**The one rule that explains the whole matrix:** for any matrix product $\mathbf{A}\mathbf{B}$, entry $(i,j)$ of the result is "row $i$ of $\mathbf{A}$" dotted with "column $j$ of $\mathbf{B}$." Here $\mathbf{A}=\mathbf{X}^T$ and $\mathbf{B}=\mathbf{X}$. But **row $i$ of $\mathbf{X}^T$ is exactly column $i$ of $\mathbf{X}$** (that's what transposing does). So:

$$\left(\mathbf{X}^T\mathbf{X}\right)_{ij} = \mathbf{c}_i \cdot \mathbf{c}_j$$

**In plain English: $\mathbf{X}^T\mathbf{X}$ is nothing but "every column of $\mathbf{X}$, dotted against every other column of $\mathbf{X}$."** A $3\times3$ matrix of all pairwise column dot-products. That's the entire mechanic — everything below is just doing this dot product 6 times (3 diagonal + 3 unique off-diagonal, mirrored).

**Entry $(0,0) = \mathbf{c}_0\cdot\mathbf{c}_0$:** dot the 1's column with itself — $1{\cdot}1+1{\cdot}1+1{\cdot}1+1{\cdot}1+1{\cdot}1 = 5$. This is just $n$ — the count of rows — for the same reason any all-1's-dot-itself is its own length.

**Entry $(0,1) = \mathbf{c}_0\cdot\mathbf{c}_1$:** dot 1's against $x_1$ — $1{\cdot}1+1{\cdot}2+1{\cdot}3+1{\cdot}4+1{\cdot}5 = 15$. Multiplying anything by 1 leaves it unchanged, so this dot product collapses to plain $\sum x_1 = 15$.

**Entry $(0,2) = \mathbf{c}_0\cdot\mathbf{c}_2$:** dot 1's against $x_2$ — same logic, $\sum x_2 = 1+1+2+2+3 = 9$.

**Entry $(1,1) = \mathbf{c}_1\cdot\mathbf{c}_1$:** dot $x_1$ against itself — $1^2+2^2+3^2+4^2+5^2 = 1+4+9+16+25 = 55 = \sum x_1^2$.

**Entry $(1,2) = \mathbf{c}_1\cdot\mathbf{c}_2$:** dot $x_1$ against $x_2$ — this is the one worth slowing down on, since it's the only entry mixing two *different* real predictors:

| Student | $x_{1i}$ | $x_{2i}$ | $x_{1i}\times x_{2i}$ |
|---|---|---|---|
| 1 | 1 | 1 | 1 |
| 2 | 2 | 1 | 2 |
| 3 | 3 | 2 | 6 |
| 4 | 4 | 2 | 8 |
| 5 | 5 | 3 | 15 |

Sum the last column: $1+2+6+8+15=32=\sum x_1x_2$.

**Entry $(2,2) = \mathbf{c}_2\cdot\mathbf{c}_2$:** dot $x_2$ against itself — $1^2+1^2+2^2+2^2+3^2 = 1+1+4+4+9=19=\sum x_2^2$.

**Because dot products are symmetric ($\mathbf{c}_i\cdot\mathbf{c}_j=\mathbf{c}_j\cdot\mathbf{c}_i$),** entries $(1,0)$, $(2,0)$, $(2,1)$ are just mirror copies of $(0,1)$, $(0,2)$, $(1,2)$ — this is *why* $\mathbf{X}^T\mathbf{X}$ is always symmetric, for any dataset, any number of predictors, no exceptions.

**Assembling the full matrix:**

$$ \mathbf{X}^T\mathbf{X} = \begin{bmatrix} 5 & 15 & 9 \\ 15 & 55 & 32 \\ 9 & 32 & 19 \end{bmatrix} $$

Reading it back: row/column 0 is "everything to do with the intercept" ($n$, $\sum x_1$, $\sum x_2$); row/column 1 is "everything to do with $x_1$" ($\sum x_1$, $\sum x_1^2$, $\sum x_1x_2$); row/column 2 is "everything to do with $x_2$." **This pattern generalizes exactly**: with $p$ predictors, $\mathbf{X}^T\mathbf{X}$ is always $(p+1)\times(p+1)$, always symmetric, and entry $(i,j)$ is always "column $i$ dotted with column $j$" — memorize the *rule*, not the specific numbers, and you can rebuild this matrix for any dataset from scratch.

**$\mathbf{X}^T\mathbf{y}$ by the identical logic** (dot each column of $\mathbf{X}$ against $\mathbf{y}$ instead of against another column of $\mathbf{X}$):

$$ \mathbf{X}^T\mathbf{y} = \begin{bmatrix} \mathbf{c}_0\cdot\mathbf{y} \\ \mathbf{c}_1\cdot\mathbf{y} \\ \mathbf{c}_2\cdot\mathbf{y} \end{bmatrix} = \begin{bmatrix} \sum y \\ \sum x_1 y \\ \sum x_2 y \end{bmatrix} = \begin{bmatrix} 320 \\ 1035 \\ 615 \end{bmatrix} $$

($\sum x_2 y = 1{\cdot}50+1{\cdot}55+2{\cdot}65+2{\cdot}70+3{\cdot}80 = 50+55+130+140+240=615$.)

---

## 4.5 Solving the System — Full Algebra, No Steps Skipped

We need $\hat{\boldsymbol{\beta}}$ solving $\mathbf{X}^T\mathbf{X}\,\hat{\boldsymbol{\beta}}=\mathbf{X}^T\mathbf{y}$:

$$5\beta_0+15\beta_1+9\beta_2=320 \quad(1)$$
$$15\beta_0+55\beta_1+32\beta_2=1035 \quad(2)$$
$$9\beta_0+32\beta1+19\beta_2=615 \quad(3)$$

**Eliminate $\beta_0$ using equation (1):**

$$\beta_0 = \frac{320-15\beta_1-9\beta_2}{5} = 64-3\beta_1-1.8\beta_2 \quad(1')$$

**Substitute $(1')$ into $(2)$:**

$$15(64-3\beta_1-1.8\beta_2)+55\beta_1+32\beta_2=1035$$
$$960-45\beta_1-27\beta_2+55\beta_1+32\beta_2=1035$$
$$960+10\beta_1+5\beta_2=1035$$
$$10\beta_1+5\beta_2=75 \quad\Rightarrow\quad 2\beta_1+\beta_2=15 \quad(2')$$

**Substitute $(1')$ into $(3)$:**

$$9(64-3\beta_1-1.8\beta_2)+32\beta_1+19\beta_2=615$$
$$576-27\beta_1-16.2\beta_2+32\beta_1+19\beta_2=615$$
$$576+5\beta_1+2.8\beta_2=615$$
$$5\beta_1+2.8\beta_2=39 \quad(3')$$

**Now solve $(2')$ and $(3')$ together — two equations, two unknowns.** From $(2')$: $\beta_2=15-2\beta_1$. Substitute into $(3')$:

$$5\beta_1+2.8(15-2\beta_1)=39$$
$$5\beta_1+42-5.6\beta_1=39$$
$$-0.6\beta_1=-3 \quad\Rightarrow\quad \beta_1=5$$

**Back-substitute to get $\beta_2$:** $\beta_2=15-2(5)=5$.

**Back-substitute both into $(1')$ to get $\beta_0$:** $\beta_0=64-3(5)-1.8(5)=64-15-9=40$.

$$\boxed{\hat{\beta}_0=40,\quad \hat{\beta}_1=5,\quad \hat{\beta}_2=5}$$

**Fitted equation:** $\hat{y}=40+5x_1+5x_2$.

**Verification (plug each row back in):**

| Student | $40+5x_1+5x_2$ | Actual $y$ | Residual |
|---|---|---|---|
| 1 | $40+5+5=50$ | 50 | 0 |
| 2 | $40+10+5=55$ | 55 | 0 |
| 3 | $40+15+10=65$ | 65 | 0 |
| 4 | $40+20+10=70$ | 70 | 0 |
| 5 | $40+25+15=80$ | 80 | 0 |

**Honesty note:** this toy dataset was built so the fit is exact (SSE = 0) — clean for verifying arithmetic, but real data essentially never fits this perfectly. Chapter 5 (inference) uses a noisy version of this dataset, since meaningful standard errors and F-tests need nonzero residuals.

---

## 4.6 Reading the Coefficients — and Seeing Omitted Variable Bias Numerically

$\hat{\beta}_1=5$: **holding practice tests fixed**, each additional hour studied is associated with a 5-point score increase.

$\hat{\beta}_2=5$: **holding hours studied fixed**, each additional practice test taken is associated with a 5-point score increase.

**Compare to Chapter 1's simple-regression slope of $\hat{\beta}_1=7.5$** (studying alone, ignoring practice tests entirely). The number *drops* from 7.5 to 5 once $x_2$ enters the model. Here's *why*, concretely:

$x_1$ and $x_2$ move together in this data — check it directly: as $x_1$ goes $1{,}2{,}3{,}4{,}5$, $x_2$ goes $1{,}1{,}2{,}2{,}3$ — every time $x_1$ rises, $x_2$ tends to rise too. When you regress $y$ on $x_1$ **alone**, $x_1$ gets "credit" not just for its own effect but for riding along with $x_2$'s effect too, since the model has no way to tell them apart. Once $x_2$ is explicitly added, the coefficient on $x_1$ is "purified" down to only its own partial contribution — the 2.5-point difference ($7.5-5$) is exactly the amount of credit that was misattributed. **This is the mechanism behind omitted-variable bias in one worked number**, not just an abstract warning.

---

## 4.7 A Geometric Upgrade

Chapter 1's picture was a 2D line through a scatter of points. Chapter 3 upgraded this to an orthogonal projection. With two predictors, the fitted "line" becomes a **plane** slicing through 3D space (one axis for $x_1$, one for $x_2$, one for $y$) — and with $p$ predictors, a $p$-dimensional **hyperplane**. The hat matrix $\mathbf{H}$ from Chapter 3 still projects $\mathbf{y}$ orthogonally onto this hyperplane, completely unchanged in concept — only the dimension of the space it projects onto has grown.

```
        y
        ^
        |     *  (actual data point, above the plane)
        |    /|
        |   / |  <- residual (vertical drop to the plane)
        |  /  |
        | /   *  <- fitted point, ON the tilted plane
        |/___________
       /            \
      x2              x1
   (the tilted plane is the fitted surface: yhat = 40 + 5x1 + 5x2)
```

---

## 4.8 A Cautionary Note: Adding Predictors Isn't Free

- **$R^2$ mechanically never decreases** when you add any predictor — even pure random noise — so $R^2$ alone can never tell you whether a new predictor genuinely helps (Chapter 14, adjusted $R^2$).
- **Multicollinearity** — if $x_1$ and $x_2$ are too strongly correlated, $\mathbf{X}^T\mathbf{X}$ becomes close to non-invertible (Chapter 3, §3.8), and coefficient estimates become unstable and hard to interpret individually, even if overall predictions stay fine (Chapter 9). In fact, our own $x_1,x_2$ in this dataset are fairly strongly correlated — a preview of exactly that concern, though 5 data points is too small a sample to diagnose it properly.

---

## 4.9 Where the Textbooks Differ

- **Kutner** introduces multiple regression very formally, immediately generalizing the ANOVA-table framework from simple regression — heavy on notation, systematic, proof-first.
- **Montgomery** spends much more time on the *practical meaning* of partial-effect coefficients, with worked examples designed to show how coefficients shift when predictors are added or removed — closest in spirit to §4.6 above.
- **ESL/ISL** treat multiple regression almost as a solved formality and rush toward variable selection and regularization — this is *their* Chapter 3, essentially a bridge, not a destination.
- **Sheather** emphasizes reading multiple-regression software output directly — the coefficient table, individual t-tests, and overall F-test (Chapter 5 territory) all appearing together in a single `lm()` summary.

---

## 4.10 Interview Q&A

**Q: What does a coefficient mean in multiple regression, precisely?**
A: The expected change in $y$ for a one-unit increase in that predictor, **holding all other predictors in the model constant** — the "partial effect," not the total unconditional effect from a simple regression on that variable alone.

**Q: How do you build $\mathbf{X}^T\mathbf{X}$ by hand for any dataset?**
A: It's the matrix of all pairwise dot products between the columns of $\mathbf{X}$ (intercept column included). Entry $(i,j)$ = column $i$ of $\mathbf{X}$ dotted with column $j$. Diagonal entries are each column dotted with itself (giving $n$ for the intercept row, $\sum x_k^2$ for predictor $k$); off-diagonal entries mix two different columns (giving $\sum x_k$ against the intercept, $\sum x_jx_k$ between two predictors). Symmetric by construction, always $(p+1)\times(p+1)$.

**Q: Why can a coefficient's sign or magnitude change when you add a new predictor?**
A: The original coefficient was previously absorbing some of the added predictor's effect (when the predictors are correlated). Once the new predictor is explicitly included, the earlier coefficient is "purified" to reflect only its own partial effect — demonstrated numerically in §4.6, where hours-studied's coefficient dropped from 7.5 to 5 once practice tests entered the model.

**Q: Does adding more predictors always improve the model?**
A: It always increases (or leaves unchanged) $R^2$ and never increases SSE, but that doesn't mean the new predictor is meaningful — it could be pure noise. True improvement requires adjusted $R^2$, information criteria (AIC/BIC), or out-of-sample validation (Chapter 14).

**Q: What happens to $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ when you move from 1 to $p$ predictors?**
A: Nothing changes in the formula — only the shape of $\mathbf{X}$ grows (more columns), so $\mathbf{X}^T\mathbf{X}$ grows to $(p+1)\times(p+1)$. Exactly why Chapter 3 was worth learning in matrix form first.

**Q: If two predictors are highly correlated, what happens to their individual coefficients?**
A: Their estimates become unstable (high variance) — small changes in the data can swing individual coefficients substantially, even though the model's *combined* predictive fit stays reasonably stable. The multicollinearity problem, covered fully in Chapter 9.

---

*End of Chapter 4 (revised). Next: Chapter 5 — Inference in Multiple Regression (individual t-tests on each coefficient, the overall F-test for joint significance, partial F-tests for comparing nested models, and why these three tests answer three genuinely different questions).*
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
