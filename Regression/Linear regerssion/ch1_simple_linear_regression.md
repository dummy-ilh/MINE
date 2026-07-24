# Chapter 1 — Linear Regression with One Predictor Variable
### (Kutner, Nachtsheim, Neter, Li — *Applied Linear Statistical Models*)

---

## 1.1 Relations Between Variables

### Functional vs. Statistical Relations

**Plain English.** A *functional* relation: if you know X exactly, you know Y exactly (e.g., Area = πr²). A *statistical* relation: knowing X tells you the rough expected value of Y, but there's scatter around it (e.g., height vs. weight).

**Why it exists.** Regression exists because most real relationships are statistical, not functional. If the world were purely functional, we wouldn't need statistics — just algebra.

**Intuition.** A scatterplot of a statistical relation is a *cloud* of points with a trend, not a thin perfect line.

**Connection forward.** The scatter around the trend is exactly what the error term ε will capture in the formal model (Section 1.4).

**Misconception.** Scatter around a fitted line is not "model failure" — it's the reason probabilistic modeling is needed at all.

**Interview angle.** *Q: Why does the regression model include an error term instead of just Y = β0 + β1X?*
*A:* Because real relationships between variables are statistical — there is inherent variability in Y even for a fixed X (measurement error, omitted variables, natural randomness). The error term ε formally represents that variability so we can quantify uncertainty in our estimates, rather than pretending the relationship is deterministic.

---

## 1.2 Regression Models and Their Uses

Kutner identifies three purposes for building a regression model. This matters because *why* you're building the model changes how you evaluate it.

1. **Description** — summarizing/quantifying how Y relates to X. Example: "for every additional year of experience, salary increases by about $2,300 on average." You care about interpretability of coefficients.

2. **Control** — using the model to keep Y at a target level by controlling X. Example: a factory adjusts oven temperature (X) to keep product thickness (Y) within spec. Here you need the *causal* direction to be right — correlation alone can mislead you into "controlling" the wrong variable.

3. **Prediction** — using the model to forecast Y for new X values, without necessarily caring why the relationship holds. Example: predicting a house's sale price from square footage. Here interpretability matters less than predictive accuracy.

**Why this distinction matters for interviews.** A very common trap: someone builds a model that predicts well (good R², good test-set RMSE) and assumes the coefficients can be used for control/causal claims. Kutner's framework makes explicit that prediction and causal control are different goals, requiring different justifications (e.g., control requires that X actually causes changes in Y, not just correlates with it).

**Interview question:** *"You built a regression model with high R² predicting churn from customer support ticket count. Can you conclude that reducing ticket volume will reduce churn?"*
**Ideal answer:** No — this is a prediction-purpose model, not a control-purpose model. High predictive accuracy doesn't establish that ticket count *causes* churn; both could be driven by a third variable (e.g., product frustration). To use the model for control (intervening on X to change Y), you'd need a causal design — randomized experiment, natural experiment, or causal inference techniques (e.g., instrumental variables) — not just observational correlation.

---

## 1.4 Basic Simple Linear Regression Model

This is the core formal object of the whole chapter. Slow down here.

### The Model

$$
Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i
$$

**What every symbol means:**
- $Y_i$: the observed response for the $i$-th observation (e.g., the $i$-th person's weight).
- $X_i$: the observed value of the predictor for the $i$-th observation (known, fixed constant — not random, in Kutner's basic setup).
- $\beta_0$: the **true, unknown** intercept — the mean of Y when X = 0. A fixed population parameter, not something we observe directly.
- $\beta_1$: the **true, unknown** slope — the change in the mean of Y for a one-unit increase in X. Also a fixed population parameter.
- $\varepsilon_i$: the **random error term** for observation $i$ — the amount by which the actual $Y_i$ deviates from the true mean line $\beta_0 + \beta_1 X_i$.

**Why this form?** We split Y into two pieces: (1) a deterministic part $\beta_0 + \beta_1 X_i$ — "what we'd expect on average given X" — and (2) a random part $\varepsilon_i$ — "everything else we can't explain with X." This split is exactly the functional/statistical distinction from 1.1, formalized.

### The Four Core Assumptions on $\varepsilon_i$

1. $E[\varepsilon_i] = 0$ for all $i$. **Meaning:** the errors don't systematically push Y up or down — on average they cancel out. **Why it matters:** this is what makes $\beta_0 + \beta_1 X_i$ the *true mean* of Y at that X, i.e., $E[Y_i] = \beta_0 + \beta_1 X_i$. If errors had nonzero mean, that mean would just get absorbed into $\beta_0$, so this assumption is really more of a convention that makes the parameters identifiable and interpretable.

2. $\text{Var}(\varepsilon_i) = \sigma^2$ for all $i$ (constant variance — **homoscedasticity**). **Meaning:** the spread of the errors doesn't depend on X. **Why it matters:** if violated (heteroscedasticity), our formulas for standard errors of $\hat\beta_0, \hat\beta_1$ become wrong, and least squares is no longer the most efficient estimator (this is the setup covered in Ch. 3 — diagnostics).

3. $\varepsilon_i$ and $\varepsilon_j$ are uncorrelated for $i \ne j$. **Meaning:** knowing the error for observation 3 tells you nothing about the error for observation 7. **Why it matters:** violated constantly in time series data (autocorrelation, Ch. 12) — e.g., today's stock price error correlates with yesterday's.

4. (For inference, added later, Section 1.11) $\varepsilon_i \sim N(0, \sigma^2)$ — errors are **normally distributed**. **Why it matters:** needed for exact small-sample t-tests and F-tests, though by the Central Limit Theorem, inference is often approximately fine even if this is mildly violated, given large enough samples.

**Common misconception:** People often think "linear regression assumes X and Y are linearly related" is the *only* assumption. In fact there are four distinct, separable assumptions (mean-zero errors, constant variance, uncorrelated errors, normality), each of which can fail independently, and each of which is diagnosed differently (residual plots, Durbin-Watson, Q-Q plots respectively).

**Interview question:** *"What are the assumptions of linear regression, and how would you check each one?"*
**Ideal answer:** (1) Linearity of the mean response in X — checked via residual-vs-fitted plots (should show no pattern). (2) Constant error variance — checked via residual-vs-fitted plots (should show no "funnel" shape) or formal tests like Breusch-Pagan. (3) Uncorrelated errors — checked via residual-vs-time/order plots or Durbin-Watson statistic, especially relevant for time series. (4) Normality of errors — checked via Q-Q plots or Shapiro-Wilk, mainly needed for exact inference (t/F tests), less critical for point estimation itself since least squares estimators are unbiased regardless.

### Meaning of $\beta_0$ and $\beta_1$ Precisely

$$
E[Y_i] = \beta_0 + \beta_1 X_i
$$

This says: the **true regression function** is the line that traces the *mean* of Y as X varies — not any individual Y. $\beta_1$ is literally $\frac{\partial E[Y]}{\partial X}$: how much the mean of Y shifts per unit increase in X. $\beta_0$ is the mean of Y when X = 0 — sometimes meaningful (e.g., X = years of experience, so X=0 is a new hire), sometimes not physically meaningful (e.g., X = temperature in Celsius where 0 is outside the observed data range — extrapolation danger, covered later).

**Interview trap:** People say "the regression line predicts Y." More precise: the regression line estimates the *mean* of Y at a given X, not any single Y value. Any individual Y still has error $\varepsilon$ around that mean.

---

## 1.5 Data for Regression Analysis

Three types of studies produce regression data:

1. **Controlled experiments** — X is set/assigned by the experimenter (e.g., randomly assigning fertilizer amounts to plots). Strongest for causal claims.
2. **Controlled experiments without randomization** — X is set but not randomly assigned.
3. **Observational data** — X and Y are both just observed as they occur (e.g., survey data). Weakest for causal claims — confounding is a serious risk.

**Why this matters for interviews:** This is the data-generating process behind the "correlation vs. causation" warning. A Google/Meta DS interviewer loves asking this because production ML systems are almost always trained on observational data, so the interviewer wants to know you understand the limits of what the resulting coefficients can be used for.

---

## 1.7 Estimation of Regression Function: The Method of Least Squares

This is the mathematical heart of the chapter. We now derive $\hat\beta_0$ and $\hat\beta_1$ — the *estimates* of the true, unknown $\beta_0, \beta_1$ — from actual data.

### The Idea, in Plain English

We don't observe $\beta_0, \beta_1$. We only observe pairs $(X_i, Y_i)$. We want to draw the "best" line through the cloud of points. "Best" is defined as: the line that makes the total squared vertical distance between the observed points and the line as small as possible.

**Why squared distance, not just distance?** Three reasons:
- Squaring makes all deviations positive, so positive and negative errors don't cancel out and hide the true error magnitude.
- Squaring is differentiable everywhere (unlike absolute value), so we can use calculus to find the minimum in closed form.
- Squaring penalizes large errors disproportionately more than small ones, which turns out to have very clean statistical properties (it corresponds to Maximum Likelihood Estimation under Normal errors — proven in 1.11).

### The Objective Function

We define the **fitted line** as:
$$
\hat Y_i = b_0 + b_1 X_i
$$
where $b_0, b_1$ are our *estimates* (not the true unknown $\beta_0,\beta_1$ — careful, this notation distinction is a common interview trip-up: lowercase $b$ = estimate from data, Greek $\beta$ = true unknown population parameter).

The **residual** for observation $i$ is:
$$
e_i = Y_i - \hat Y_i
$$
(the gap between what was actually observed and what the fitted line predicts — an *estimate* of the unobservable true error $\varepsilon_i$).

We want to choose $b_0, b_1$ to minimize the **sum of squared residuals**:
$$
Q = \sum_{i=1}^n (Y_i - b_0 - b_1 X_i)^2
$$

### Deriving the Normal Equations (Calculus)

To minimize Q, take partial derivatives with respect to $b_0$ and $b_1$, set each to zero.

**Derivative w.r.t. $b_0$:**
$$
\frac{\partial Q}{\partial b_0} = -2\sum_{i=1}^n (Y_i - b_0 - b_1 X_i) = 0
$$
Divide by -2:
$$
\sum (Y_i - b_0 - b_1 X_i) = 0 \quad\Rightarrow\quad \sum Y_i = n b_0 + b_1 \sum X_i \quad \text{...(Normal Equation 1)}
$$

**Derivative w.r.t. $b_1$:**
$$
\frac{\partial Q}{\partial b_1} = -2\sum_{i=1}^n X_i(Y_i - b_0 - b_1 X_i) = 0
$$
$$
\sum X_i Y_i = b_0 \sum X_i + b_1 \sum X_i^2 \quad \text{...(Normal Equation 2)}
$$

Solving these two linear equations simultaneously (standard algebra — substitute Equation 1 into Equation 2) gives the **least squares estimators**:

$$
b_1 = \frac{\sum (X_i - \bar X)(Y_i - \bar Y)}{\sum (X_i - \bar X)^2} = \frac{S_{XY}}{S_{XX}}
$$

$$
b_0 = \bar Y - b_1 \bar X
$$

**What these formulas mean intuitively:**
- $b_1$ is the ratio of "how X and Y move together" ($S_{XY}$, the sum of cross-products of deviations) to "how much X varies by itself" ($S_{XX}$). If X and Y move together strongly relative to how much X spreads out, slope is large.
- $b_0$ is chosen so the fitted line passes exactly through the point $(\bar X, \bar Y)$ — the "center of mass" of the data. This is a direct algebraic consequence of Normal Equation 1 (divide it by n).

**Why this is worth remembering as a fact, not just a formula:** *the least squares regression line always passes through $(\bar X, \bar Y)$.* This is a common conceptual interview question, and it drops right out of Normal Equation 1 — it's not a separate assumption, it's forced by the calculus.

---

## Worked Numerical Example (by hand)

Let's use a tiny dataset: hours studied (X) vs. exam score (Y) for 5 students.

| Student | X (hours) | Y (score) |
|---|---|---|
| 1 | 2 | 65 |
| 2 | 3 | 70 |
| 3 | 5 | 78 |
| 4 | 7 | 85 |
| 5 | 8 | 92 |

**Step 1 — Compute means.**
$$
\bar X = \frac{2+3+5+7+8}{5} = \frac{25}{5} = 5
$$
$$
\bar Y = \frac{65+70+78+85+92}{5} = \frac{390}{5} = 78
$$

**Step 2 — Compute deviations from the mean, for each point.**

| $X_i$ | $Y_i$ | $X_i-\bar X$ | $Y_i - \bar Y$ | $(X_i-\bar X)(Y_i-\bar Y)$ | $(X_i - \bar X)^2$ |
|---|---|---|---|---|---|
| 2 | 65 | -3 | -13 | 39 | 9 |
| 3 | 70 | -2 | -8  | 16 | 4 |
| 5 | 78 | 0  | 0   | 0  | 0 |
| 7 | 85 | 2  | 7   | 14 | 4 |
| 8 | 92 | 3  | 14  | 42 | 9 |

**Step 3 — Sum the last two columns.**
$$
S_{XY} = \sum (X_i-\bar X)(Y_i-\bar Y) = 39+16+0+14+42 = 111
$$
$$
S_{XX} = \sum (X_i - \bar X)^2 = 9+4+0+4+9 = 26
$$

**Step 4 — Compute $b_1$ and $b_0$.**
$$
b_1 = \frac{S_{XY}}{S_{XX}} = \frac{111}{26} \approx 4.269
$$
$$
b_0 = \bar Y - b_1 \bar X = 78 - (4.269)(5) = 78 - 21.346 = 56.654
$$

**Fitted line:**
$$
\hat Y_i = 56.654 + 4.269 X_i
$$

**Interpretation:** For every additional hour studied, predicted exam score increases by about 4.27 points on average. A student who studies 0 hours is predicted to score about 56.65 (extrapolation caveat: X=0 is outside the observed range 2–8, so this intercept should be interpreted cautiously — classic Kutner point about extrapolation, covered further in Ch. 1.10 / Ch. 3).

**Step 5 — Compute fitted values and residuals** (needed for Section 1.9):

| $X_i$ | $Y_i$ | $\hat Y_i = 56.654+4.269X_i$ | $e_i = Y_i - \hat Y_i$ |
|---|---|---|---|
| 2 | 65 | 56.654+8.538=65.192 | 65-65.192 = -0.192 |
| 3 | 70 | 56.654+12.807=69.461 | 70-69.461 = 0.539 |
| 5 | 78 | 56.654+21.345=78.000 (=$\bar Y$, exact) | 78-78 = 0.000 |
| 7 | 85 | 56.654+29.883=86.538 | 85-86.538 = -1.538 |
| 8 | 92 | 56.654+34.152=90.807 | 92-90.807 = 1.193 |

Notice: at $X_i = \bar X = 5$, the fitted value equals $\bar Y$ exactly, and the residual is 0 — this is the direct visual confirmation that the line passes through $(\bar X, \bar Y)$.

---

## 1.9 Residuals — Properties

The residuals $e_i$ are our sample estimates of the unobservable true errors $\varepsilon_i$. Two algebraic facts (guaranteed by the normal equations, not assumptions):

1. $\sum e_i = 0$ always (direct consequence of Normal Equation 1). **Check on our data:** $-0.192+0.539+0.000-1.538+1.193 = 0.002 \approx 0$ ✓ (rounding).
2. $\sum X_i e_i = 0$ always (direct consequence of Normal Equation 2) — residuals are uncorrelated with X *by construction* in the fitted sample, regardless of whether the true model assumptions hold.

**Why this matters / common misconception:** People sometimes think "residuals summing to zero" or "residuals uncorrelated with X" is *evidence* the model assumptions are satisfied. It is **not** — these are mechanical, algebraic guarantees of the least squares procedure itself, true even for a badly misspecified model (e.g., fitting a straight line to clearly curved data). What you should check instead is whether residuals show a *pattern* when plotted against $\hat Y_i$ or $X_i$ (curvature, funnel shape, trends) — that's real diagnostic information (Chapter 3).

**Interview question:** *"If I fit OLS and the residuals sum to zero, does that mean my model assumptions are correct?"*
**Ideal answer:** No — residuals summing to zero (and being uncorrelated with X in-sample) is a mathematical guarantee of the least-squares normal equations, true regardless of whether the linearity, constant-variance, or other assumptions hold. It's not diagnostic evidence of a good fit. Real diagnostics come from examining the *pattern* of residuals (e.g., residual vs. fitted plots for non-linearity or heteroscedasticity).

---

## 1.10 Properties of the Fitted Regression Line

Beyond passing through $(\bar X, \bar Y)$:

- The sum of the fitted values equals the sum of observed values: $\sum \hat Y_i = \sum Y_i$. (Follows since $\sum e_i = 0$.)
- Estimating $\sigma^2$: since $\sigma^2 = \text{Var}(\varepsilon_i)$ is unknown, we estimate it from the residuals via the **Mean Squared Error (MSE)**:
$$
MSE = \frac{\sum e_i^2}{n-2}
$$
**Why divide by $n-2$, not $n$?** Because we "used up" 2 degrees of freedom estimating $b_0$ and $b_1$ from the data before we could compute residuals. This is the same logic as dividing by $n-1$ for sample variance (which uses up 1 df estimating $\bar X$). Each parameter estimated from the same data costs one degree of freedom.

**Compute for our example:**
$$
\sum e_i^2 = (-0.192)^2+(0.539)^2+(0)^2+(-1.538)^2+(1.193)^2
$$
$$
= 0.0369+0.2905+0+2.3654+1.4232 = 4.116
$$
$$
MSE = \frac{4.116}{5-2} = \frac{4.116}{3} \approx 1.372
$$

This $MSE$ estimates $\sigma^2$, the variance of the true errors — and it will be the key building block for standard errors and hypothesis tests in Chapter 2.

**Interview trap:** People confuse MSE here (mean squared *residual*, an unbiased estimator of $\sigma^2$ with $n-2$ denominator) with the ML/DS colloquial "MSE" used as a loss function (often divided by $n$, not $n-2$, and used purely for optimization/prediction accuracy, not as an unbiased variance estimator). Same name, different denominator, different purpose — worth explicitly distinguishing in an interview if asked.

---

## 1.11 Normal Error Regression Model — Why Least Squares = Maximum Likelihood

Adding the normality assumption ($\varepsilon_i \sim N(0,\sigma^2)$, independent), we can derive $b_0, b_1$ a completely different way: **Maximum Likelihood Estimation (MLE).**

**Setup:** Since $\varepsilon_i \sim N(0,\sigma^2)$ and $Y_i = \beta_0+\beta_1X_i+\varepsilon_i$, we have $Y_i \sim N(\beta_0+\beta_1X_i, \sigma^2)$. The likelihood of observing our data given parameters $\beta_0,\beta_1,\sigma^2$ is the product of each point's normal density:
$$
L(\beta_0,\beta_1,\sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(Y_i-\beta_0-\beta_1X_i)^2}{2\sigma^2}\right)
$$

Taking the log (log-likelihood, easier to maximize):
$$
\ell = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(Y_i-\beta_0-\beta_1X_i)^2
$$

**Key insight:** To maximize $\ell$ with respect to $\beta_0, \beta_1$, since the first term doesn't involve them, we just need to **minimize** $\sum(Y_i-\beta_0-\beta_1X_i)^2$ — which is *exactly* the least-squares objective Q from Section 1.7!

**This is a big deal, and a favorite interview fact:** Under the assumption of normally distributed errors, the least squares estimators and the maximum likelihood estimators of $\beta_0, \beta_1$ are **identical**. This is why least squares isn't an arbitrary choice — under normality, it's the estimator with the strongest theoretical justification (MLE has well-known optimality properties: asymptotic efficiency, consistency).

**Important nuance:** the MLE of $\sigma^2$ itself, however, is *not* identical to MSE — MLE gives $\hat\sigma^2_{MLE} = \frac{\sum e_i^2}{n}$ (divides by $n$, biased), while the unbiased estimator MSE divides by $n-2$. This is a classic interview "gotcha": least squares = MLE for the coefficients, but the natural unbiased variance estimator is not the MLE variance estimator.

**Interview question:** *"Why is least squares regression theoretically justified, rather than just a convenient computational trick?"*
**Ideal answer:** Under the assumption that errors are i.i.d. Normal(0, σ²), the least-squares estimators for β0 and β1 coincide exactly with the maximum likelihood estimators — because maximizing the Gaussian log-likelihood reduces to minimizing the sum of squared residuals. This gives OLS the full weight of MLE theory: consistency, asymptotic efficiency, and (by the Gauss-Markov theorem, covered in Ch. 5) it's the Best Linear Unbiased Estimator (BLUE) even without the normality assumption, needing only the first three assumptions (zero mean, constant variance, uncorrelated errors).

---

## Python Implementation — From Scratch (NumPy only)

```python
import numpy as np

# Data
X = np.array([2, 3, 5, 7, 8], dtype=float)
Y = np.array([65, 70, 78, 85, 92], dtype=float)
n = len(X)

# Step 1: means
X_bar = X.mean()
Y_bar = Y.mean()

# Step 2: deviations
dX = X - X_bar
dY = Y - Y_bar

# Step 3: S_xy and S_xx
S_xy = np.sum(dX * dY)
S_xx = np.sum(dX ** 2)

# Step 4: least squares estimates
b1 = S_xy / S_xx
b0 = Y_bar - b1 * X_bar

print(f"b0 = {b0:.4f}, b1 = {b1:.4f}")

# Step 5: fitted values and residuals
Y_hat = b0 + b1 * X
residuals = Y - Y_hat
print("Fitted values:", np.round(Y_hat, 3))
print("Residuals:", np.round(residuals, 3))

# Sanity checks (algebraic guarantees)
print("Sum of residuals (~0):", np.sum(residuals))
print("Sum of X_i * e_i (~0):", np.sum(X * residuals))

# MSE (estimate of sigma^2), df = n - 2
SSE = np.sum(residuals ** 2)
MSE = SSE / (n - 2)
print(f"SSE = {SSE:.4f}, MSE = {MSE:.4f}")
```

**Expected output** (matches our hand calculation):
```
b0 = 56.6538, b1 = 4.2692
Fitted values: [65.192 69.462 78.    86.538 90.808]
Residuals: [-0.192  0.538  0.    -1.538  1.192]
Sum of residuals (~0): ~0.0
Sum of X_i * e_i (~0): ~0.0
SSE = 4.1154, MSE = 1.3718
```

## Equivalent scikit-learn Implementation

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([2, 3, 5, 7, 8], dtype=float).reshape(-1, 1)  # sklearn wants 2D X
Y = np.array([65, 70, 78, 85, 92], dtype=float)

model = LinearRegression()
model.fit(X, Y)

print(f"Intercept (b0): {model.intercept_:.4f}")
print(f"Slope (b1): {model.coef_[0]:.4f}")

Y_hat = model.predict(X)
print("Fitted values:", np.round(Y_hat, 3))
```

Note: `sklearn` doesn't directly give you MSE-with-(n-2)-denominator or residual diagnostics out of the box — that's `statsmodels` territory, which we'll use once we get to Chapter 2 (inference), since statsmodels reports standard errors, t-tests, and confidence intervals natively.

---

## Interview Question Bank — Chapter 1

**Conceptual:**
1. What's the difference between a functional and a statistical relation? Why does regression need the latter?
2. What do β0 and β1 represent, precisely — of Y itself, or of the mean of Y?
3. Why do we need an error term at all in the regression model?

**Derivation:**
4. Derive the least-squares estimator $b_1$ from the objective function $Q = \sum(Y_i - b_0 - b_1X_i)^2$.
5. Prove that the least-squares fitted line always passes through $(\bar X, \bar Y)$.
6. Show that under normally distributed errors, the least squares estimator and the MLE for β0, β1 coincide.

**ML/Statistics:**
7. Why is MSE in regression theory divided by (n−2) rather than n? What's the general principle?
8. What's the difference between the MLE of σ² and the unbiased estimator (MSE)?
9. Under Gauss-Markov, what does "BLUE" mean, and which assumptions are needed to get there (versus needing full normality)?

**Coding:**
10. Implement simple linear regression from scratch using only NumPy (no sklearn/statsmodels).
11. Given fitted residuals, write code to verify the two algebraic guarantees (Σe_i = 0, ΣX_ie_i = 0).

**Traps:**
12. "My residuals sum to zero, so my model assumptions must be satisfied." — What's wrong with this reasoning?
13. "The regression line predicts the value of Y for a new observation." — What's the more precise statement?
14. If asked to extrapolate the fitted line to X=0 in our worked example, what's the danger?

**Ideal answers to traps 12–14** are given inline in the corresponding sections above (1.9, 1.4, worked example) — try answering from memory before re-reading them.

---

*This file covers Kutner Ch. 1, Sections 1.1–1.11 (relations between variables, purposes of regression, the simple linear regression model and its assumptions, data types, least-squares derivation, a full worked numerical example, residual properties, and the normal error/MLE equivalence). Chapter 2 (Inferences in Regression and Correlation Analysis) is next: sampling distributions of b0/b1, confidence intervals, and hypothesis tests — this is where the "slow down hard" pacing really kicks in.*
