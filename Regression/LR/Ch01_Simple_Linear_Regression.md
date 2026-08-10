# Chapter 1 — Simple Linear Regression

*Synthesized from Kutner (Applied Linear Statistical Models), Montgomery (Intro to Linear Regression Analysis), Sheather (A Modern Approach to Regression with R), and Hastie/Tibshirani (ISL/ESL) — built for FAANG MLE/DS interview prep.*

---

## 1.1 The Motivating Question

Before any formula: what problem are we actually solving?

You have two things you can measure — say, **hours studied** (x) and **exam score** (y). You suspect one moves the other. You want a single, simple rule: *"if I know x, my best guess for y is ___."*

That's it. That's the entire ambition of simple linear regression: **draw the single straight line through a cloud of points that makes the smallest total error when used for prediction.**

Everything else in this chapter — the formulas, the derivations, the diagnostics — exists only to make that one sentence precise and trustworthy.

---

## 1.2 The Model — Every Symbol Explained Before Any Math

The model is written:

$$ y_i = \beta_0 + \beta_1 x_i + \varepsilon_i $$

Before touching this as an equation, here's what each piece *means*:

| Symbol | Plain-English meaning |
|---|---|
| $y_i$ | The actual observed outcome for data point $i$ (e.g., person $i$'s exam score) |
| $x_i$ | The input/predictor for data point $i$ (e.g., person $i$'s hours studied) |
| $\beta_0$ (**beta-nought**) | The **intercept** — the predicted value of $y$ when $x=0$. Think of it as "the starting point" of the line, not necessarily meaningful in real life (nobody studies "0 hours" meaningfully) but mathematically necessary to anchor the line |
| $\beta_1$ | The **slope** — for every 1-unit increase in $x$, how much do we expect $y$ to change, *on average* |
| $\varepsilon_i$ (**epsilon**) | The **error term** — everything that affects $y_i$ that isn't captured by $x_i$. Randomness, noise, unmeasured factors. This is *not* a mistake in your model — it's an admission that the world isn't perfectly deterministic |

Critically: $\beta_0$ and $\beta_1$ are **unknown, fixed, true population parameters**. We never observe them directly — we only ever *estimate* them from a sample, and we write those estimates as $\hat{\beta}_0$ and $\hat{\beta}_1$ (the "hats" mean "our best guess of").

So the **fitted line** — the one you actually compute and use for prediction — is:

$$ \hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i $$

with no $\varepsilon_i$ term, because $\hat{y}_i$ is a *prediction*, not an actual observation. The gap between the real $y_i$ and the predicted $\hat{y}_i$ is called the **residual**:

$$ e_i = y_i - \hat{y}_i $$

**Interview-critical distinction:** $\varepsilon_i$ (error) is a theoretical, unobservable quantity about the *true* population line. $e_i$ (residual) is a real, computable number about *your fitted* line. Interviewers love asking "what's the difference between error and residual" — this is it.

---

## 1.3 Why a Straight Line? Why Not Something Else?

Montgomery's framing is useful here: we're not claiming the *true* relationship between x and y is a perfect line. We're claiming it's a reasonable **local approximation**, and that linear models are:
- Easy to estimate (closed-form solution — no iterative optimization needed)
- Easy to interpret (a single slope number tells the whole story)
- Surprisingly hard to beat when data is limited or noisy (low variance, even if biased)

This bias-variance framing (borrowed from ESL/ISL) becomes central later when we compare linear regression to more flexible models — a simple model may have some bias (it doesn't capture curvature) but pays for it with much lower variance (it doesn't overfit noise).

---

## 1.4 Finding the Best Line — The Least Squares Criterion

"Best" needs a definition. We define best as: **the line that minimizes the total squared vertical distance between the actual points and the line.**

Why squared, and why vertical?

- **Vertical distance**, because we're predicting $y$ from $x$ — we care about error in the thing we're predicting, not error in $x$.
- **Squared**, for three reasons, all of which are fair game in interviews:
  1. It penalizes large errors disproportionately more than small ones (a residual of 4 contributes 16, not 4 — this discourages wildly bad predictions)
  2. It makes the math differentiable everywhere (unlike absolute value, which has a kink at 0) — this gives us a clean closed-form solution
  3. Squaring cancels sign, so positive and negative errors don't cancel each other out when summed

This gives the **objective function**, called the **Residual Sum of Squares (RSS)**, sometimes called SSE (Sum of Squared Errors):

$$ RSS(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2 $$

We want the $\beta_0, \beta_1$ that make this number as small as possible. This is officially called **Ordinary Least Squares (OLS)**.

---

## 1.5 Deriving the OLS Estimators (Necessary Derivation Only)

## 1.5 Deriving the OLS Estimators — Full Derivation, Simplified

**The goal:** find the line $\hat{y}_i = \beta_0 + \beta_1 x_i$ that makes the total squared error as small as possible:

$$ RSS(\beta_0, \beta_1) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2 $$

Think of RSS as a bowl-shaped surface sitting above the $(\beta_0, \beta_1)$ plane. The bottom of the bowl is the best-fit line. To find the bottom, we do what you'd do with any minimum: set the slope to zero in every direction — i.e., take the partial derivative with respect to each parameter and set it to 0.

---

### Step 1: Derivative with respect to $\beta_0$

$$ \frac{\partial RSS}{\partial \beta_0} = \sum_{i=1}^n 2(y_i - \beta_0 - \beta_1 x_i)(-1) = -2\sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) $$

Set it to 0 (drop the $-2$, it doesn't affect where the sum is zero):

$$ \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) = 0 $$

Split the sum term by term:

$$ \sum y_i - n\beta_0 - \beta_1 \sum x_i = 0 $$

Divide everything by $n$ (recall $\bar{y} = \frac{1}{n}\sum y_i$ and $\bar{x} = \frac{1}{n}\sum x_i$):

$$ \bar{y} - \beta_0 - \beta_1 \bar{x} = 0 \quad \Rightarrow \quad \boxed{\beta_0 = \bar{y} - \beta_1 \bar{x}} $$

**Plain-English takeaway:** whatever the slope turns out to be, the intercept is forced to be whatever makes the line pass through $(\bar{x}, \bar{y})$ — the "center of mass" of the data. That's just algebra falling out of the calculus, not an assumption.

---

### Step 2: Derivative with respect to $\beta_1$

$$ \frac{\partial RSS}{\partial \beta_1} = \sum_{i=1}^n 2(y_i - \beta_0 - \beta_1 x_i)(-x_i) = -2\sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1 x_i) $$

Set to 0, drop the $-2$:

$$ \sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1 x_i) = 0 $$

Now substitute Step 1's result, $\beta_0 = \bar{y} - \beta_1\bar{x}$:

$$ \sum_{i=1}^n x_i\big(y_i - (\bar{y} - \beta_1\bar{x}) - \beta_1 x_i\big) = 0 $$

$$ \sum_{i=1}^n x_i\big((y_i - \bar{y}) - \beta_1(x_i - \bar{x})\big) = 0 $$

Split into two sums:

$$ \sum_{i=1}^n x_i(y_i - \bar{y}) - \beta_1 \sum_{i=1}^n x_i(x_i - \bar{x}) = 0 $$

$$ \beta_1 = \frac{\sum_{i=1}^n x_i(y_i - \bar{y})}{\sum_{i=1}^n x_i(x_i - \bar{x})} $$

This is correct but ugly — there's a well-known trick to make it symmetric. Two small identities:

- $\sum x_i(y_i - \bar y) = \sum (x_i - \bar x)(y_i - \bar y)$ — true because $\sum \bar{x}(y_i - \bar y) = \bar{x}\sum(y_i-\bar y) = 0$ (deviations from the mean always sum to zero), so subtracting $\bar x$ inside the first factor changes nothing.
- Same logic: $\sum x_i(x_i - \bar x) = \sum (x_i - \bar x)^2$.

Applying both:

$$ \boxed{\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}} $$

**Plain-English reading:** numerator = "how much do x and y wiggle together" (covariance-flavored), denominator = "how much does x wiggle on its own" (variance). Slope = shared movement ÷ x's own spread.

---

### Bonus: why $\hat\beta_1 = r \cdot \dfrac{s_y}{s_x}$

Recall the definitions (with $n-1$ dividing both, which cancels anyway):

$$ r = \frac{S_{xy}/(n-1)}{s_x s_y}, \qquad s_x^2 = \frac{S_{xx}}{n-1}, \qquad s_y^2 = \frac{S_{yy}}{n-1} $$

From $r$'s definition: $S_{xy} = r \cdot s_x s_y \cdot (n-1)$. Plug into $\hat\beta_1 = S_{xy}/S_{xx}$, and use $S_{xx} = s_x^2(n-1)$:

$$ \hat\beta_1 = \frac{r \cdot s_x s_y \cdot (n-1)}{s_x^2 (n-1)} = r \cdot \frac{s_y}{s_x} $$

**Interview-ready one-liner:** correlation tells you the *strength and direction* of the linear relationship (unitless, between -1 and 1); the slope takes that same relationship and *rescales* it into the actual units of x and y.
---

## 1.6 Worked Numerical Example (By Hand)

Small dataset — hours studied (x) vs. exam score (y):

| Student | x (hours) | y (score) |
|---|---|---|
| 1 | 1 | 50 |
| 2 | 2 | 55 |
| 3 | 3 | 65 |
| 4 | 4 | 70 |
| 5 | 5 | 80 |

**Step 1 — means:**
$\bar{x} = (1+2+3+4+5)/5 = 3$
$\bar{y} = (50+55+65+70+80)/5 = 64$

**Step 2 — deviations and products:**

| x | y | $x-\bar{x}$ | $y-\bar{y}$ | product | $(x-\bar{x})^2$ |
|---|---|---|---|---|---|
| 1 | 50 | -2 | -14 | 28 | 4 |
| 2 | 55 | -1 | -9 | 9 | 1 |
| 3 | 65 | 0 | 1 | 0 | 0 |
| 4 | 70 | 1 | 6 | 6 | 1 |
| 5 | 80 | 2 | 16 | 32 | 4 |

$S_{xy} = 28+9+0+6+32 = 75$
$S_{xx} = 4+1+0+1+4 = 10$

**Step 3 — slope and intercept:**

$$ \hat{\beta}_1 = 75/10 = 7.5 $$
$$ \hat{\beta}_0 = 64 - 7.5(3) = 64 - 22.5 = 41.5 $$

**Fitted line:** $\hat{y} = 41.5 + 7.5x$

**Interpretation:** Every additional hour of studying is associated with a **7.5-point** increase in expected exam score. A student who studies 0 hours is predicted to score 41.5 (extrapolation — be careful, x=0 is outside our observed range of 1–5, so this intercept is a mathematical anchor, not necessarily a trustworthy prediction).

**Check:** predict for x=3 (a value we have): $\hat{y} = 41.5 + 7.5(3) = 64$. Matches $\bar{y}$ exactly — confirms the line passes through $(\bar{x},\bar{y})$.

**Residual for student 1:** actual y=50, predicted $\hat{y}=41.5+7.5(1)=49$. Residual $e_1 = 50-49 = 1$.

---

## 1.7 The Geometric Picture

Sheather's book leans on this visual heavily, and it's worth internalizing: think of OLS as an **orthogonal projection**. Your $n$-dimensional vector of observed y-values gets projected onto the 2-dimensional plane spanned by "the constant vector" (for $\beta_0$) and "the x-vector" (for $\beta_1$). The fitted values $\hat{y}$ are the *shadow* of $y$ on that plane, and the residual vector $e$ is exactly **perpendicular** to that plane.

This geometric fact — residuals are orthogonal to the fitted values and orthogonal to x — is why $\sum e_i = 0$ and $\sum e_i x_i = 0$ always hold exactly for OLS. Not approximately. Always, by construction, no matter the data.

*(Diagram to visualize: a scatter plot with the fitted line drawn through it, and dashed vertical segments from each point down to the line — those dashed segments are the residuals being minimized.)*

---

## 1.8 The Assumptions — "LINE"

For OLS estimates to have good statistical properties (unbiasedness, minimum variance — the **Gauss-Markov theorem**, which both Kutner and Montgomery prove in full), we assume:

- **L**inearity — the true relationship between x and y is linear
- **I**ndependence — errors $\varepsilon_i$ are independent of each other (no autocorrelation)
- **N**ormality — errors are normally distributed (needed for hypothesis tests/CIs, *not* needed just to compute $\hat{\beta}$)
- **E**qual variance (**homoscedasticity**) — the spread of errors is constant across all values of x

**Interview-critical nuance:** OLS point estimates ($\hat{\beta}_0, \hat{\beta}_1$) only require linearity + the errors having mean zero and being uncorrelated with x. Normality is needed only for valid t-tests/p-values/confidence intervals — a very commonly confused point that interviewers probe directly ("do you need normality to fit OLS?" — answer: no, you need it for inference, not estimation).




**BLUE = Best Linear Unbiased Estimator.** This is the punchline of the Gauss-Markov theorem, and it's worth knowing exactly what each letter means and which assumptions buy you which property.

**Gauss-Markov theorem, plain-English statement:** if the assumptions below hold, OLS isn't just *an* unbiased estimator — it's the *best* one, meaning it has the lowest variance among all linear unbiased estimators. You could invent some other linear-unbiased way to estimate the slope, but it would never beat OLS's variance. That's the whole content of the theorem.

**Breaking down "BLUE":**

- **B — Best** = minimum variance among all linear unbiased estimators
- **L — Linear** = the estimator is a linear function of the $y_i$'s (OLS is, by construction — $\hat\beta_1$ is just a weighted sum of the $y_i$'s)
- **U — Unbiased** = $E[\hat\beta_0] = \beta_0$ and $E[\hat\beta_1] = \beta_1$ — on average, across repeated samples, you hit the true value
- **E — Estimator** = it's a rule for estimating parameters, not a claim about any single sample

**What's actually required for each piece — this is the part interviewers drill on:**

| Property | What you need |
|---|---|
| **Unbiasedness** | Linearity of the true relationship + errors have mean zero ($E[\varepsilon_i]=0$) + errors uncorrelated with x |
| **Minimum variance (the "Best" in BLUE)** | All of the above **+ homoscedasticity** (constant error variance) **+ no autocorrelation** (independence) |
| **Valid t-tests, p-values, confidence intervals** | All of the above **+ normality of errors** |

**The key hierarchy to say out loud in an interview:**

1. To get *unbiased* estimates → you only need linearity + zero-mean, x-uncorrelated errors.
2. To get the *best* (lowest-variance) unbiased estimates → you additionally need homoscedasticity and independence. Drop either one and OLS is still unbiased, but it stops being "Best" — some other estimator (like Weighted Least Squares under heteroscedasticity, or GLS under autocorrelation) would have lower variance instead.
3. To do *inference* (hypothesis tests, CIs) → you additionally need normality. This is purely about the sampling distribution of $\hat\beta$ being normal so you can use t/F distributions — it has nothing to do with whether the point estimate itself is good.

**Common interview trap, restated precisely:** "Do you need normality to fit OLS?" — No. You need it to *trust your p-values and confidence intervals*, not to compute $\hat\beta_0, \hat\beta_1$, and not for those estimates to be unbiased or even best. By the Central Limit Theorem, with large enough $n$, your estimates are approximately normal anyway even if the errors aren't — so normality matters most in small samples.

**Quick mnemonic for what breaks what:**
- Break linearity → biased estimates, full stop.
- Break homoscedasticity → still unbiased, but no longer "Best" (standard errors are also wrong, so your CIs are wrong even though $\hat\beta$ itself is fine).
- Break independence (autocorrelation) → same story as homoscedasticity: unbiased but not best, and standard errors get distorted.
- Break normality → point estimates fine, inference (t-tests, CIs) unreliable in small samples.
---

## 1.9 Diagnostics — A First Look

You never trust a fitted line blindly. The primary tool: **the residual plot** (residuals $e_i$ on the y-axis, fitted values $\hat{y}_i$ or x on the x-axis).

What you're looking for:
- **Random scatter around zero** → good, assumptions look reasonable
- **A curved pattern (U-shape or arc)** → linearity is violated; you're missing a nonlinear term
- **A funnel/cone shape (spread increasing or decreasing)** → heteroscedasticity, equal-variance assumption violated
- **Clumping/trends over time/order** → independence violated (common in time series data)

*(Diagram to visualize: four small residual-plot panels side by side — one "healthy" random cloud, one U-shaped curve, one funnel/cone shape, one with a visible trend over time — this is the classic four-panel diagnostic image found in nearly every regression textbook.)*

We'll go much deeper into formal diagnostic tests (Breusch-Pagan, Durbin-Watson, leverage/Cook's distance) in a later chapter — this is just enough to build the instinct that **a fitted line is not the end of the analysis, it's the start of a diagnostic conversation.**

---

## 1.10 Where the Textbooks Differ (So You're Not Confused Later)

- **Notation**: Kutner tends to write $b_0, b_1$ for sample estimates instead of $\hat{\beta}_0, \hat{\beta}_1$; Montgomery and Sheather use hat notation. Same object, different symbol — don't let this trip you up switching between books.
- **Emphasis**: Kutner is derivation-heavy and proof-oriented (ANOVA-table framing throughout). Montgomery is more diagnostics/practitioner-oriented, especially strong on residual analysis and multicollinearity. Sheather leans computational (R-code-driven), showing you what a "bad" residual plot looks like in practice rather than just describing it abstractly.
- **ESL/ISL** barely dwells on simple linear regression at all — they treat it as a warm-up before jumping to multiple regression and regularization, which is why this synthesized chapter borrows more heavily from Kutner/Montgomery/Sheather at this stage.

---

## 1.11 Interview Q&A

**Q: Why do we minimize squared error instead of absolute error?**
A: Squared error is differentiable everywhere (giving a closed-form solution via calculus), penalizes large errors more heavily, and under Gaussian error assumptions, minimizing squared error is equivalent to maximum likelihood estimation. Absolute error minimization (which gives the median, not the mean) requires linear programming, not calculus.

**Q: What's the relationship between $\hat{\beta}_1$ and the correlation coefficient $r$?**
A: $\hat{\beta}_1 = r \cdot \frac{s_y}{s_x}$. The slope is the correlation rescaled by the ratio of standard deviations. If x and y are standardized (mean 0, variance 1), the slope literally *equals* the correlation.

**Q: Does the regression line always pass through a specific point?**
A: Yes — always through $(\bar{x}, \bar{y})$, a direct consequence of the first normal equation ($\partial RSS/\partial \beta_0 = 0$).

**Q: Do residuals sum to zero?**
A: Yes, always, by construction — this falls directly out of the normal equations, not an empirical coincidence.

**Q: Is normality of errors required to fit OLS?**
A: No. It's required for valid hypothesis tests, p-values, and confidence intervals — not for computing the point estimates themselves, which only need the errors to have mean zero.

**Q: What happens to $\hat{\beta}_1$ if you swap the roles of x and y (regress x on y instead of y on x)?**
A: You get a *different* line, not the reciprocal slope, unless $r = \pm 1$ exactly. This is a classic trap — regression is not symmetric like correlation is.

---

*End of Chapter 1. Next: Chapter 2 — Inference in Simple Linear Regression (standard errors, confidence intervals, hypothesis tests on $\beta_1$, the ANOVA decomposition, $R^2$).*
