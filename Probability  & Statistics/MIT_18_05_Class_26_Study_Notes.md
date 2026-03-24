# MIT 18.05 — Introduction to Probability and Statistics
## Complete Study Notes: Class 26
### Linear Regression — Least Squares, Polynomials, Overfitting, and Regression to the Mean

---

> **How to use these notes:** Every concept, example, and worked problem from the Class 26 PDFs is reproduced here with expanded explanations. You should never need to refer back to the original documents.

---

## Topic 1: Introduction to Linear Regression

### 1. Concept Overview

**Linear regression** is a method for modeling the relationship between two (or more) variables using a mathematical function. Given bivariate data $(x_i, y_i)$ for $i = 1, \ldots, n$, the goal is to find a function $y = f(x)$ that fits the data as closely as possible.

The modeling assumptions are:
- $x_i$ (the **predictor** or **independent variable**) is **not random** — it is a controlled or measured quantity.
- $y_i$ (the **response** or **dependent variable**) is a function of $x_i$ **plus random noise**.

### 2. Intuition

Think of trying to draw the "best" straight line through a cloud of data points. No single line will pass through all points (unless the data is perfectly linear). The question is: which line is closest to all the points simultaneously?

The **least squares** method answers this by minimizing the total squared vertical distance from each data point to the line. It is the most widely used fitting criterion in statistics and machine learning.

> **Why squared errors?** Squaring penalizes large deviations more than small ones, treats positive and negative errors symmetrically, and (critically) leads to a mathematically tractable optimization problem with a unique closed-form solution.

### 3. Terminology

| Term | Meaning |
|---|---|
| $x$ | Predictor, independent variable, explanatory variable |
| $y$ | Response, dependent variable |
| $\hat{y}$ | Predicted value of $y$ |
| $\epsilon_i$ | Residual (observed $y_i$ minus predicted $\hat{y}_i$) |
| $\hat{a}, \hat{b}$ | Least squares estimates of the slope and intercept |

### 4. What "Linear" Really Means

> **⚠️ Common Confusion:**
>
> The word **"linear"** in *linear regression* does **not** mean fitting a straight line. It refers to the fact that the model is **linear in the unknown parameters** (e.g., $a$ and $b$).
>
> A parabola $y = ax^2 + bx + c$ is also a linear regression model because it is linear in the parameters $a$, $b$, $c$.

When we specifically fit a straight line to bivariate data, we call it **simple linear regression**.

---

## Topic 2: Fitting a Line — Simple Linear Regression

### 1. The Model

For data $(x_1, y_1), \ldots, (x_n, y_n)$, the simple linear regression model is:

$$\boxed{y_i = a x_i + b + \epsilon_i}$$

where:
- $a$ = slope (unknown)
- $b$ = intercept (unknown)
- $\epsilon_i$ = random error at observation $i$

The **predicted value** for observation $i$ is:

$$\hat{y}_i = \hat{a} x_i + \hat{b}$$

The **residual** (leftover noise) is:

$$\epsilon_i = y_i - \hat{y}_i = y_i - \hat{a} x_i - \hat{b}$$

### 2. The Least Squares Criterion

We find $\hat{a}$ and $\hat{b}$ by minimizing the **total sum of squared errors (SSE)**:

$$S(a, b) = \sum_{i=1}^n \epsilon_i^2 = \sum_{i=1}^n (y_i - a x_i - b)^2$$

> **Intuition:** Each $(y_i - ax_i - b)$ is the vertical distance from the $i$-th data point to the line. We square these distances (so positive and negative errors don't cancel), then sum them. Minimizing this sum gives the "closest" line to all points simultaneously.

### 3. The Least Squares Solution

#### Key Formulas

$$\boxed{\hat{a} = \frac{s_{xy}}{s_{xx}}, \qquad \hat{b} = \bar{y} - \hat{a}\,\bar{x}}$$

where:

$$\bar{x} = \frac{1}{n}\sum x_i \qquad \bar{y} = \frac{1}{n}\sum y_i$$

$$s_{xx} = \frac{1}{n-1}\sum(x_i - \bar{x})^2 \qquad \text{(sample variance of } x\text{)}$$

$$s_{xy} = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y}) \qquad \text{(sample covariance of } x \text{ and } y\text{)}$$

#### Intuition for the Formulas

- $\hat{a} = s_{xy}/s_{xx}$: The slope equals the covariance between $x$ and $y$ divided by the variance of $x$. It tells us: for every one unit increase in $x$ (relative to its spread), how much does $y$ change (relative to its relationship with $x$)?
- $\hat{b} = \bar{y} - \hat{a}\bar{x}$: Once the slope is fixed, the intercept is determined by requiring the line to pass through the **point of means** $(\bar{x}, \bar{y})$.

#### Important Property: The Line Always Passes Through $(\bar{x}, \bar{y})$

Since $\hat{b} = \bar{y} - \hat{a}\bar{x}$, we have:

$$\hat{a}\bar{x} + \hat{b} = \hat{a}\bar{x} + \bar{y} - \hat{a}\bar{x} = \bar{y}$$

So $(\bar{x}, \bar{y})$ is always on the fitted regression line.

---

### 4. Derivation of the Least Squares Formula

Starting from:

$$S(a, b) = \sum_{i=1}^n (y_i - ax_i - b)^2$$

Take partial derivatives and set to zero:

**$\partial S / \partial b = 0$:**

$$\sum_{i=1}^n -2(y_i - ax_i - b) = 0 \quad \Rightarrow \quad \sum y_i = a\sum x_i + nb$$

Dividing by $n$:

$$\bar{y} = a\bar{x} + b \quad \Rightarrow \quad b = \bar{y} - a\bar{x} \tag{i}$$

**$\partial S / \partial a = 0$:**

$$\sum_{i=1}^n -2x_i(y_i - ax_i - b) = 0 \quad \Rightarrow \quad \sum x_i y_i = a\sum x_i^2 + b\sum x_i \tag{ii}$$

Substituting (i) into (ii) and simplifying using the identities for centered sums:

$$\sum x_i y_i - n\bar{x}\bar{y} = a\left(\sum x_i^2 - n\bar{x}^2\right)$$

$$\sum(x_i - \bar{x})(y_i - \bar{y}) = a \sum(x_i - \bar{x})^2$$

$$a = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{s_{xy}}{s_{xx}} \qquad \square$$

---

### 5. Worked Examples — Simple Linear Regression

#### Example 1 — US Postage Stamp Cost (Real Data)

**Problem:** Fit a line to stamp cost (in cents) vs. years since 1960.

Data includes 24 data points from 1963 to 2019.

**R output:**

$$\hat{y} = -0.21390 + 0.88203\,x$$

where $x$ = years since 1960, $y$ = cost in cents.

**Prediction for 2021** ($x = 61$):

$$\hat{y} = -0.21390 + 0.88203 \times 61 = -0.21390 + 53.804 = 53.59 \approx 53.6 \text{ cents}$$

**Actual cost in 2021:** 55 cents. The model is close but not perfect — regression predicts on average, not exactly.

**Interpretation:** The coefficient $\hat{a} = 0.88$ means stamp prices increased about 0.88 cents per year on average over this period.

---

#### Example 2 — Father-Son Heights

**Problem:** Predict adult son height from father's height.

**Setup:** $x_i$ = height of $i$-th father, $y_i$ = height of $i$-th adult son.

**Use:** Fit the least squares line $\hat{y} = \hat{a}x + \hat{b}$. For a new father with height $x_0$, predict his son's adult height as $\hat{y} = \hat{a}x_0 + \hat{b}$.

**Key insight:** This is the classic example that gave rise to the term "regression" (Galton, 1886). Tall fathers tend to have tall sons, but sons tend to be *closer to the average* than their fathers — this is "regression to the mean."

---

#### Example 5 — Least Squares Line (From Prep Notes)

**Data:** $(0, 1),\; (2, 1),\; (3, 4)$

**Step 1: Compute means**

$$\bar{x} = \frac{0+2+3}{3} = \frac{5}{3}, \qquad \bar{y} = \frac{1+1+4}{3} = 2$$

**Step 2: Compute $s_{xx}$**

$$s_{xx} = \frac{1}{n-1}\sum(x_i - \bar{x})^2 = \frac{1}{2}\left[\left(0-\frac{5}{3}\right)^2 + \left(2-\frac{5}{3}\right)^2 + \left(3-\frac{5}{3}\right)^2\right]$$

$$= \frac{1}{2}\left[\frac{25}{9} + \frac{1}{9} + \frac{16}{9}\right] = \frac{1}{2} \cdot \frac{42}{9} = \frac{42}{18} = \frac{7}{3}$$

**Step 3: Compute $s_{xy}$**

$$s_{xy} = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$$

$$= \frac{1}{2}\left[\left(-\frac{5}{3}\right)(1-2) + \left(\frac{1}{3}\right)(1-2) + \left(\frac{4}{3}\right)(4-2)\right]$$

$$= \frac{1}{2}\left[\frac{5}{3} - \frac{1}{3} + \frac{8}{3}\right] = \frac{1}{2} \cdot \frac{12}{3} = 2$$

**Step 4: Compute slope and intercept**

$$\hat{a} = \frac{s_{xy}}{s_{xx}} = \frac{2}{7/3} = \frac{6}{7}$$

$$\hat{b} = \bar{y} - \hat{a}\bar{x} = 2 - \frac{6}{7} \cdot \frac{5}{3} = 2 - \frac{10}{7} = \frac{4}{7}$$

**Final answer:** $\boxed{y = \frac{6}{7}x + \frac{4}{7}}$

**Interpretation:** For every unit increase in $x$, the predicted $y$ increases by $6/7 \approx 0.857$.

---

#### Problem 1 (Board) — Least Squares on Data (1,3), (2,1), (4,4)

**Part (a): Simple Linear Regression**

**Data:** $(x_1, y_1) = (1,3)$, $(x_2, y_2) = (2,1)$, $(x_3, y_3) = (4,4)$

**(a-i) Model:**

$$y_i = ax_i + b + e_i \qquad \text{prediction: } \hat{y}_i = ax_i + b$$

**(a-ii) Total squared error:**

$$T = \sum(y_i - ax_i - b)^2 = (3-a-b)^2 + (1-2a-b)^2 + (4-4a-b)^2$$

**(a-iii) Minimize by calculus:**

$$\frac{\partial T}{\partial a} = -2(3-a-b) - 4(1-2a-b) - 8(4-4a-b) = 0$$

$$\frac{\partial T}{\partial b} = -2(3-a-b) - 2(1-2a-b) - 2(4-4a-b) = 0$$

Expanding and collecting terms:

From $\partial T/\partial b = 0$:
$$-2(3-a-b) - 2(1-2a-b) - 2(4-4a-b) = 0$$
$$-6+2a+2b - 2+4a+2b - 8+8a+2b = 0$$
$$14a + 6b = 16 \implies 7a + 3b = 8 \tag{1}$$

From $\partial T/\partial a = 0$:
$$-2(3-a-b) - 4(1-2a-b) - 8(4-4a-b) = 0$$
$$-6+2a+2b - 4+8a+4b - 32+32a+8b = 0$$
$$42a + 14b = 42 \implies 3a + b = 3 \implies \text{full system: } 21a + 7b = 21 \tag{2}$$

**System of equations:**

$$21a + 7b = 21 \tag{2}$$
$$7a + 3b = 8 \tag{1}$$

**Solving:**

Multiply (1) by 3: $21a + 9b = 24$

Subtract (2): $2b = 3 \implies b = 3/2$

Substitute back: $7a + 9/2 = 8 \implies 7a = 7/2 \implies a = 1/2$

$$\boxed{\text{Best-fit line: } y = \frac{1}{2}x + \frac{3}{2}}$$

**Verification via formulas:**

$$\bar{x} = \frac{1+2+4}{3} = \frac{7}{3}, \qquad \bar{y} = \frac{3+1+4}{3} = \frac{8}{3}$$

$$s_{xx} = \frac{1}{2}\left[(1-\frac{7}{3})^2 + (2-\frac{7}{3})^2 + (4-\frac{7}{3})^2\right] = \frac{1}{2}\left[\frac{16}{9}+\frac{1}{9}+\frac{25}{9}\right] = \frac{7}{3}$$

$$s_{xy} = \frac{1}{2}\left[(1-\frac{7}{3})(3-\frac{8}{3}) + (2-\frac{7}{3})(1-\frac{8}{3}) + (4-\frac{7}{3})(4-\frac{8}{3})\right]$$

$$= \frac{1}{2}\left[\left(-\frac{4}{3}\right)\left(\frac{1}{3}\right) + \left(-\frac{1}{3}\right)\left(-\frac{5}{3}\right) + \left(\frac{5}{3}\right)\left(\frac{4}{3}\right)\right]$$

$$= \frac{1}{2}\left[-\frac{4}{9} + \frac{5}{9} + \frac{20}{9}\right] = \frac{1}{2} \cdot \frac{21}{9} = \frac{7}{6}$$

$$\hat{a} = \frac{s_{xy}}{s_{xx}} = \frac{7/6}{7/3} = \frac{1}{2}, \qquad \hat{b} = \frac{8}{3} - \frac{1}{2}\cdot\frac{7}{3} = \frac{8}{3} - \frac{7}{6} = \frac{16-7}{6} = \frac{3}{2} \checkmark$$

---

## Topic 3: Residuals and Homoscedasticity

### 1. Residuals

The **residual** for the $i$-th observation is:

$$e_i = y_i - \hat{y}_i = y_i - \hat{a}x_i - \hat{b}$$

Residuals represent the part of $y_i$ not explained by the regression line. Plotting residuals is an essential diagnostic:

- Residuals should appear **randomly scattered** around zero.
- Residuals should have **roughly constant spread** (vertical scatter) across all values of $x$.

### 2. Homoscedasticity

> **Definition — Homoscedasticity:** The residuals $\epsilon_i$ have the **same variance** for all $i$. Equivalently, the vertical spread of data around the regression line is approximately constant across all $x$.

**Why it matters:** The least squares method assumes homoscedasticity. If this assumption is violated, the estimates $\hat{a}$ and $\hat{b}$ may be inefficient or confidence intervals may be invalid.

> **Definition — Heteroscedasticity:** The residuals have **non-constant variance** — the spread of data increases (or decreases) as $x$ increases.

**What to do with heteroscedastic data:** Transform the data (e.g., take $\log(y)$) before applying least squares, or use weighted least squares.

**Visual check:**
- **Homoscedastic:** residual plot shows a horizontal band of roughly constant width.
- **Heteroscedastic:** residual plot shows a "fan" shape — the spread widens or narrows with $x$.

---

## Topic 4: Linear Regression for Polynomials

### 1. Concept Overview

We are not limited to fitting straight lines. The same least squares principle can fit **any polynomial** $y = a_d x^d + a_{d-1} x^{d-1} + \cdots + a_1 x + a_0$ to data.

The key insight: even though a parabola is not a "linear" shape, the error $S$ is still a quadratic function of the **parameters** $(a_0, a_1, \ldots, a_d)$ — which are what we optimize over. The optimality conditions are still a system of **linear algebraic equations** in the parameters.

### 2. Fitting a Parabola

**Model:** $y_i = ax_i^2 + bx_i + c + \epsilon_i$

**Prediction:** $\hat{y}_i = ax_i^2 + bx_i + c$

**Total squared error:**

$$S(a, b, c) = \sum_{i=1}^n (y_i - ax_i^2 - bx_i - c)^2$$

Setting $\partial S/\partial a = \partial S/\partial b = \partial S/\partial c = 0$ gives a **3×3 linear system** in $a$, $b$, $c$.

### 3. Worked Examples — Polynomial Regression

#### Example 6 — Best Fitting Parabola to (0,1), (2,1), (3,4)

**Setup:** Fit $y = ax^2 + bx + c$ to data $(0,1), (2,1), (3,4)$.

**Total squared error:**

$$S = (1 - 0 - 0 - c)^2 + (1 - 4a - 2b - c)^2 + (4 - 9a - 3b - c)^2$$

Setting partial derivatives to zero gives the system. The solution is:

$$\boxed{y = x^2 - 2x + 1}$$

**Verification:**
- At $x=0$: $y = 0 - 0 + 1 = 1$ ✓
- At $x=2$: $y = 4 - 4 + 1 = 1$ ✓
- At $x=3$: $y = 9 - 6 + 1 = 4$ ✓

> **Key insight:** With 3 data points and 3 parameters ($a$, $b$, $c$), the parabola passes through all 3 points exactly — zero residual error. This is because $n$ points with distinct $x$-values can always be fit exactly by a polynomial of degree $n-1$. This is the origin of **overfitting** with high-degree polynomials.

---

#### Problem 1(b) — Parabola for (1,3), (2,1), (4,4)

**Model:** $y_i = ax_i^2 + bx_i + c + e_i$

**Total squared error:**

$$T = (3-a-b-c)^2 + (1-4a-2b-c)^2 + (4-16a-4b-c)^2$$

**Partial derivatives set to zero:**

$$\frac{\partial T}{\partial a} = -2(3-a-b-c) - 8(1-4a-2b-c) - 32(4-16a-4b-c) = 0$$

$$\frac{\partial T}{\partial b} = -2(3-a-b-c) - 4(1-4a-2b-c) - 8(4-16a-4b-c) = 0$$

$$\frac{\partial T}{\partial c} = -2(3-a-b-c) - 2(1-4a-2b-c) - 2(4-16a-4b-c) = 0$$

After simplification, the linear system is:

$$273a + 73b + 21c = 71$$
$$73a + 21b + 7c = 21$$
$$21a + 7b + 3c = 8$$

**Solution (using R or Gaussian elimination):**

$$\hat{a} \approx 1.1667, \quad \hat{b} \approx -5.5, \quad \hat{c} \approx 7.3333$$

$$\boxed{y \approx 1.167x^2 - 5.5x + 7.333}$$

---

#### Problem 1(d) — Best Fitting Cubic (General Setup)

**Model:** $y_i = ax_i^3 + bx_i^2 + cx_i + d + e_i$

**Total squared error for general data $(x_1, y_1), \ldots, (x_n, y_n)$:**

$$T = \sum_{i=1}^n (y_i - ax_i^3 - bx_i^2 - cx_i - d)^2$$

Setting $\partial T/\partial a = \partial T/\partial b = \partial T/\partial c = \partial T/\partial d = 0$ yields a **4×4 linear system** in $(a, b, c, d)$. This system is solved numerically.

---

## Topic 5: Data Transformation — Fitting Non-Polynomial Curves

### 1. Concept Overview

Sometimes the relationship between $x$ and $y$ is not polynomial, but can be **linearized** by transforming the data. The most common case is exponential growth.

### 2. Exponential Fitting

**Suppose the true relationship is:** $y = ce^{ax}$

**Take the natural log:**

$$\ln(y) = ax + \ln(c)$$

Let $b = \ln(c)$. Then:

$$\ln(y) = ax + b$$

This is a **linear** relationship between $x$ and $\ln(y)$.

**Procedure:**
1. Compute $z_i = \ln(y_i)$ for each data point.
2. Apply simple linear regression to data $(x_i, z_i)$.
3. Get estimates $\hat{a}$ (slope) and $\hat{b}$ (intercept).
4. Back-transform: $\hat{c} = e^{\hat{b}}$, giving $\hat{y} = e^{\hat{b}} \cdot e^{\hat{a}x} = e^{\hat{a}x + \hat{b}}$.

#### Example 8 — Exponential Data Transformation

**Model:** $y = ce^{ax}$

**Transformation:** Let $z_i = \ln(y_i)$. Then $z_i = ax_i + b + \epsilon_i$ where $b = \ln(c)$.

**Apply linear regression to $(x_i, z_i)$** to get $\hat{a}$ and $\hat{b}$.

**Back-transform:** $\hat{y} = e^{\hat{b}} e^{\hat{a}x}$

---

#### Problem 1(c) — Exponential Fit to (1,3), (2,1), (4,4)

**Model:** $y = e^{ax+b}$, equivalently $\ln(y) = ax + b + e_i$.

**Transformation:** Let $z_i = \ln(y_i)$:

| $x_i$ | $y_i$ | $z_i = \ln(y_i)$ |
|---|---|---|
| 1 | 3 | $\ln(3) \approx 1.099$ |
| 2 | 1 | $\ln(1) = 0$ |
| 4 | 4 | $\ln(4) \approx 1.386$ |

**Total squared error in transformed space:**

$$T = (\ln 3 - a - b)^2 + (0 - 2a - b)^2 + (\ln 4 - 4a - b)^2$$

Minimize as before. Using R:

$$\hat{a} \approx 0.18, \quad \hat{b} \approx 0.41$$

**Final exponential model:**

$$\hat{y} = e^{0.41} \cdot e^{0.18x} \approx 1.507 \cdot e^{0.18x}$$

> **Why minimize in $\ln(y)$ space?** Because taking $\ln$ linearizes the exponential relationship. The squared errors are computed in the $\ln(y)$ space, which is not the same as minimizing errors in the original $y$ space. This is a modeling choice — it's appropriate when the noise is multiplicative (proportional to $y$).

---

## Topic 6: Overfitting

### 1. Concept Overview

Given $n$ data points with distinct $x$-values, a polynomial of degree $n-1$ can always be found that passes through ALL of them exactly (zero residual error). But this is usually a terrible idea.

**Overfitting** occurs when a model is so complex that it fits the random noise in the training data, rather than the true underlying relationship. An overfit model:
- Fits the current data very well (small training error).
- Generalizes poorly to new data (large prediction error on new observations).

### 2. Intuition

Imagine trying to predict tomorrow's temperature. If you fit a 365-degree polynomial to the last 365 days of temperatures, you'd get a perfect fit to historical data — but the resulting curve would wildly oscillate and be useless for predicting the next day.

### 3. The Bias-Variance Trade-off

| Model Complexity | Training Error | Prediction Error | Issue |
|---|---|---|---|
| Too simple (underfitting) | High | High | Misses true pattern (high bias) |
| Just right | Low | Low | Good generalization |
| Too complex (overfitting) | Zero | High | Fits noise (high variance) |

### 4. Worked Example — Overfitting (Example 9)

**Problem:** Fit polynomials of degree 1, 2, and 9 to 10 data points generated from a quadratic model.

**Results:**

| Degree | $R^2$ | Description |
|---|---|---|
| 1 (linear) | 0.3968 | Poor fit — misses the curvature |
| 2 (quadratic) | 0.9455 | Good fit — captures the true pattern |
| 9 (degree-9) | 1.0000 | Perfect fit to training data — overfits |

**Key insight:** The degree 2 model (which matches the true data-generating process) achieves the best balance. The degree 9 model perfectly interpolates all 10 training points but would make wild predictions on new data.

> **Practical guideline:** Prefer the simplest model that explains the data adequately. Adding complexity should be justified by a meaningful improvement in fit, not just for the sake of reducing training error.

---

## Topic 7: Multiple Linear Regression

### 1. Concept Overview

When the response variable depends on multiple predictors, we use **multiple linear regression**:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_m x_m$$

**Examples:**
- Fish population as a function of several pollutant levels.
- Son's height predicted from both mother's and father's height.
- House price predicted from size, age, number of rooms, location.

### 2. Model

For data tuples $(y_i, x_{1,i}, x_{2,i}, \ldots, x_{m,i})$:

$$y_i = \beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i} + \cdots + \beta_m x_{m,i} + \epsilon_i$$

The least squares criterion minimizes:

$$S(\beta_0, \ldots, \beta_m) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_{1,i} - \cdots - \beta_m x_{m,i})^2$$

This reduces to a linear system of $m+1$ equations in $m+1$ unknowns — solved by matrix algebra in R using `lm()`.

> **Scope note:** Multiple linear regression is not covered in detail in 18.05 but is foundational to machine learning and data science. The R function `lm()` handles it seamlessly.

---

## Topic 8: Linear Regression as a Statistical Model

### 1. The Full Statistical Model

The complete linear regression model for fitting a line treats each response $y_i$ as a **random variable**:

$$Y_i = ax_i + b + \varepsilon_i$$

where the **error terms** $\varepsilon_i$ are:
- Independent of each other.
- Each with mean 0 and standard deviation $\sigma$.
- Standard assumption: $\varepsilon_i \sim N(0, \sigma^2)$.

**Consequence:** Since $\varepsilon_i \sim N(0, \sigma^2)$ and $ax_i + b$ is a constant:

$$Y_i \sim N(ax_i + b, \sigma^2)$$

So $y_i$ has mean $ax_i + b$ (on the regression line) and constant variance $\sigma^2$ for all $i$.

### 2. Why the Mean of $Y_i$ Is on the Line

$$E[Y_i] = ax_i + b + E[\varepsilon_i] = ax_i + b + 0 = ax_i + b$$

The regression line describes the **conditional mean** of $Y$ given $x$:

$$E[Y \mid X = x] = ax + b$$

### 3. Connection: Least Squares = Maximum Likelihood

When $\varepsilon_i \sim N(0, \sigma^2)$, least squares and MLE are equivalent.

#### Proof (Problem 2d)

Since $Y_i \sim N(ax_i + b, \sigma^2)$, the likelihood of observation $y_i$ is:

$$f(y_i \mid x_i, a, b) = \frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(y_i - ax_i - b)^2}{2\sigma^2}\right)$$

Since the observations are independent, the joint likelihood is:

$$L(a, b) = \prod_{i=1}^n f(y_i \mid x_i, a, b) = \left(\frac{1}{\sqrt{2\pi}\sigma}\right)^n \exp\!\left(-\frac{\sum_{i=1}^n (y_i - ax_i - b)^2}{2\sigma^2}\right)$$

To maximize $L(a,b)$, we maximize the exponent, which means we **minimize**:

$$\sum_{i=1}^n (y_i - ax_i - b)^2$$

This is exactly the least squares criterion. $\square$

> **Key takeaway:** Under the normality assumption for errors, least squares estimates are also maximum likelihood estimates. This gives them strong theoretical justification.

---

## Topic 9: The Coefficient of Determination $R^2$

### 1. Measuring Goodness of Fit

Once the regression line is fit, we need to measure how well it fits. The most common measure is $R^2$.

### 2. Definitions

**Total Sum of Squares (TSS)** — total variability in $y$:

$$\text{TSS} = \sum_{i=1}^n (y_i - \bar{y})^2$$

**Residual Sum of Squares (RSS)** — variability NOT explained by the regression:

$$\text{RSS} = \sum_{i=1}^n (y_i - \hat{a}x_i - \hat{b})^2$$

**Explained variability:** $\text{TSS} - \text{RSS}$

**Coefficient of Determination:**

$$\boxed{R^2 = \frac{\text{TSS} - \text{RSS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}}$$

### 3. Interpretation

| $R^2$ value | Interpretation |
|---|---|
| Close to 1 | Good fit — regression explains most of the variability in $y$ |
| Close to 0 | Poor fit — regression explains little of the variability |
| Exactly 1 | Perfect fit — all points lie exactly on the line |
| Exactly 0 | No linear relationship — predicting $\bar{y}$ is as good as the regression |

> **For simple linear regression:** $R^2$ equals the **square of the correlation coefficient** $r$ between $x$ and $y$.

### 4. Warning: $R^2$ Increases with Polynomial Degree

$R^2$ always increases (or stays the same) when you add more parameters. This is why $R^2 = 1.0000$ for the degree-9 polynomial on 10 data points — but the model is overfit.

**Never use $R^2$ alone to choose model complexity.** Use cross-validation, AIC, BIC, or adjusted $R^2$ instead.

#### Example 12 — $R^2$ vs. Polynomial Degree

From the overfitting example (10 data points):

| Polynomial Degree | $R^2$ |
|---|---|
| 1 (linear) | 0.3968 |
| 2 (quadratic) | 0.9455 |
| 9 (degree-9) | 1.0000 |

The degree-9 fit achieves perfect $R^2$ on training data but is overfit and would perform poorly on new data. The degree-2 fit balances accuracy and parsimony best.

---

## Topic 10: Regression to the Mean

### 1. Concept Overview

The term "regression" was coined by Francis Galton in 1886 when he studied the relationship between the heights of fathers and their sons. He noticed that:

- Very tall fathers tend to have tall sons, but sons who are **shorter than their fathers** on average.
- Very short fathers tend to have short sons, but sons who are **taller than their fathers** on average.

In both cases, the son's height "regresses" toward the population mean. This is **regression to the mean**.

### 2. Formal Development

To see this mathematically, we **standardize** both variables:

$$u_i = \frac{x_i - \bar{x}}{\sqrt{s_{xx}}}, \qquad v_i = \frac{y_i - \bar{y}}{\sqrt{s_{yy}}}$$

After standardization:
$$\bar{u} = \bar{v} = 0, \qquad s_{uu} = s_{vv} = 1$$

The sample covariance of $u$ and $v$ is:

$$s_{uv} = \frac{s_{xy}}{\sqrt{s_{xx} s_{yy}}} = \rho$$

where $\rho$ is the **sample correlation coefficient** (between $-1$ and $1$).

**Regression line in standardized coordinates:**

$$\hat{a} = \frac{s_{uv}}{s_{uu}} = \rho, \qquad \hat{b} = \bar{v} - \hat{a}\bar{u} = 0$$

So the regression line is:

$$\hat{v} = \rho \cdot u$$

### 3. Why This Means Regression to the Mean

Suppose $\rho = 0.8$ (strong positive correlation, but not perfect). Then:
- If $u = 2$ (father is 2 standard deviations above average), the predicted son's $\hat{v} = 0.8 \times 2 = 1.6$ — only 1.6 standard deviations above average.
- If $u = -1.5$ (father is 1.5 std below average), the predicted $\hat{v} = -1.2$ — only 1.2 std below average.

In both cases, the son is predicted to be **closer to the mean** than the father. This "regression toward the mean" is not biology — it's pure mathematics. It happens for any $|\rho| < 1$.

**Extreme case — zero correlation:** If $\rho = 0$, then $\hat{v} = 0$ always. No matter how extreme the father's height, the best prediction for the son's height is always the mean $\bar{y}$.

### 4. Practical Consequences — Examples

#### Example 10 — IQ Tests and Useless Interventions

**Scenario:** Children take an IQ test at age 4, then again at age 5.

**Regression to the mean predicts:**
- Children scoring low at age 4 will, on average, score higher at age 5 (closer to the mean).
- Children scoring high at age 4 will, on average, score slightly lower at age 5.

**Danger:** If a "treatment" is applied to low scorers after the age 4 test, the improvement seen at age 5 may be entirely due to regression to the mean — not the treatment.

> **This is why randomized controlled experiments with a control group are essential.** Without a control group, regression to the mean can make a useless intervention appear effective.

#### Example 11 — Reward and Punishment in Schools

**Scenario:** A school rewards high exam scorers and punishes low scorers.

**What regression to the mean predicts:**
- High scorers will tend to do slightly worse next time (their high score partly reflected luck).
- Low scorers will tend to do slightly better next time (their low score partly reflected bad luck).

**The dangerous misinterpretation:** 
- Punishment appears to improve performance (low scorers improved).
- Reward appears to hurt performance (high scorers declined).

But this is entirely regression to the mean! The punishment and reward had no effect. Acting on this flawed interpretation — e.g., adopting harsher punishment regimes — has real and potentially harmful consequences.

> **Galton's original insight and its lessons:** Regression to the mean is one of the most commonly misunderstood phenomena in statistics. It underlies many fallacies in sports ("hot hand" misconceptions), medicine (why patients seek treatment when symptoms are worst), and education (why interventions for poor performers appear more effective than they are).

---

## Topic 11: Complete Worked Problem — Using Formulas

### Problem 2 (Board) — Full Analysis of (1,3), (2,1), (4,4)

#### Part (a) — Sample Means

$$\bar{x} = \frac{1+2+4}{3} = \frac{7}{3} \approx 2.33, \qquad \bar{y} = \frac{3+1+4}{3} = \frac{8}{3} \approx 2.67$$

#### Part (b) — Best-Fit Line Using Formulas

**Compute $s_{xx}$:**

$$s_{xx} = \frac{1}{2}\left[\left(1-\frac{7}{3}\right)^2 + \left(2-\frac{7}{3}\right)^2 + \left(4-\frac{7}{3}\right)^2\right]$$

$$= \frac{1}{2}\left[\left(-\frac{4}{3}\right)^2 + \left(-\frac{1}{3}\right)^2 + \left(\frac{5}{3}\right)^2\right] = \frac{1}{2}\cdot\frac{16+1+25}{9} = \frac{42}{18} = \frac{7}{3}$$

**Compute $s_{xy}$:**

$$s_{xy} = \frac{1}{2}\left[\left(-\frac{4}{3}\right)\left(3-\frac{8}{3}\right) + \left(-\frac{1}{3}\right)\left(1-\frac{8}{3}\right) + \left(\frac{5}{3}\right)\left(4-\frac{8}{3}\right)\right]$$

$$= \frac{1}{2}\left[\left(-\frac{4}{3}\right)\left(\frac{1}{3}\right) + \left(-\frac{1}{3}\right)\left(-\frac{5}{3}\right) + \left(\frac{5}{3}\right)\left(\frac{4}{3}\right)\right]$$

$$= \frac{1}{2}\left[-\frac{4}{9} + \frac{5}{9} + \frac{20}{9}\right] = \frac{1}{2}\cdot\frac{21}{9} = \frac{21}{18} = \frac{7}{6}$$

**Slope and intercept:**

$$\hat{a} = \frac{s_{xy}}{s_{xx}} = \frac{7/6}{7/3} = \frac{7}{6} \cdot \frac{3}{7} = \frac{1}{2}$$

$$\hat{b} = \bar{y} - \hat{a}\bar{x} = \frac{8}{3} - \frac{1}{2}\cdot\frac{7}{3} = \frac{8}{3} - \frac{7}{6} = \frac{16}{6} - \frac{7}{6} = \frac{9}{6} = \frac{3}{2}$$

$$\boxed{\hat{y} = \frac{1}{2}x + \frac{3}{2}}$$

(Same result as in Problem 1a — the two methods agree. ✓)

#### Part (c) — Point of Means Is Always on the Line

**Claim:** $(\bar{x}, \bar{y})$ is always on the fitted line.

**Proof:** From the formula $\hat{b} = \bar{y} - \hat{a}\bar{x}$:

$$\hat{a}\bar{x} + \hat{b} = \hat{a}\bar{x} + \bar{y} - \hat{a}\bar{x} = \bar{y} \qquad \square$$

**Numerical check:** $\hat{a}\bar{x} + \hat{b} = \frac{1}{2}\cdot\frac{7}{3} + \frac{3}{2} = \frac{7}{6} + \frac{9}{6} = \frac{16}{6} = \frac{8}{3} = \bar{y}$ ✓

#### Part (d) — Least Squares = MLE (Under Normal Errors)

**Model:** $y_i = ax_i + b + E_i$ where $E_i \sim N(0, \sigma^2)$ independently.

This means $y_i \sim N(ax_i + b, \sigma^2)$.

**Likelihood for a single observation:**

$$f(y_i \mid x_i, a, b) = \frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(y_i - ax_i - b)^2}{2\sigma^2}\right)$$

**Joint likelihood (independence → product of likelihoods → sum of exponents):**

$$L(a,b) = \left(\frac{1}{\sqrt{2\pi}\sigma}\right)^n \exp\!\left(-\frac{\sum_{i=1}^n (y_i - ax_i - b)^2}{2\sigma^2}\right)$$

**Maximizing $L(a,b)$:** Since the prefactor is a positive constant and the exponent is negative, $L$ is maximized when the exponent is **as close to 0 as possible** — i.e., when:

$$\sum_{i=1}^n (y_i - ax_i - b)^2 \text{ is minimized}$$

This is exactly the least squares criterion. $\square$

---

## Topic 12: The R Function `lm`

### 1. Overview

The R function `lm()` (linear model) handles all forms of linear regression without manual computation.

### 2. Basic Usage

```r
# Simple linear regression: y ~ x
data_x = c(0, 2, 3)
data_y = c(1, 1, 4)
fit = lm(data_y ~ data_x)
summary(fit)

# Extract coefficients
coef(fit)        # slope and intercept
fitted(fit)      # predicted values y_hat
residuals(fit)   # residuals e_i = y_i - y_hat_i

# Predict for a new x value
predict(fit, newdata = data.frame(data_x = 5))
```

### 3. Polynomial Regression in R

```r
# Fit a quadratic (degree 2) polynomial
fit_quad = lm(data_y ~ data_x + I(data_x^2))

# Fit a cubic (degree 3) polynomial
fit_cubic = lm(data_y ~ data_x + I(data_x^2) + I(data_x^3))

# Using poly() for cleaner syntax
fit_poly = lm(data_y ~ poly(data_x, degree=2, raw=TRUE))
```

### 4. Exponential Fit in R

```r
# For y = c * exp(a * x), take log first
z = log(data_y)   # natural log
fit_exp = lm(z ~ data_x)

# Extract parameters
a_hat = coef(fit_exp)[2]      # slope
b_hat = coef(fit_exp)[1]      # intercept = log(c)
c_hat = exp(b_hat)             # back-transform

# Predicted y: y_hat = c_hat * exp(a_hat * x)
```

### 5. Checking Model Fit

```r
# R-squared
summary(fit)$r.squared

# Residual plots
plot(data_x, residuals(fit))
abline(h = 0, col = "red")   # horizontal reference line

# Full diagnostic plots
plot(fit)
```

---

## Topic 13: Common Mistakes

| Mistake | Correct Understanding |
|---|---|
| "Linear regression can only fit lines" | "Linear" refers to linearity in parameters, not the shape of the curve; polynomials are also linear regression models |
| "A large $R^2$ means the model is good" | $R^2$ always increases with polynomial degree; high $R^2$ on training data can mean overfitting |
| "The point $(\bar{x}, \bar{y})$ might not be on the regression line" | It is always on the line — the formula $\hat{b} = \bar{y} - \hat{a}\bar{x}$ guarantees this |
| "Regression to the mean is a biological or causal phenomenon" | It is a pure mathematical consequence of imperfect correlation, $|\rho| < 1$ |
| "Using a high-degree polynomial gives the best predictions" | Overfitting — perfect fit on training data often means poor generalization |
| "Heteroscedastic data can be used with ordinary least squares" | Violation of the homoscedasticity assumption; transform or use weighted least squares |
| "Least squares minimizes absolute errors" | No — it minimizes the sum of SQUARED errors; this makes the optimization tractable |
| "Regression works with any data" | The model assumes: $x$ is non-random; errors are mean-zero; homoscedasticity; (for inference) normality of errors |

---

## Topic 14: Quick Reference — All Key Formulas

### Simple Linear Regression

$$\hat{y} = \hat{a}x + \hat{b}$$

$$\hat{a} = \frac{s_{xy}}{s_{xx}}, \qquad \hat{b} = \bar{y} - \hat{a}\bar{x}$$

$$\bar{x} = \frac{1}{n}\sum x_i, \quad \bar{y} = \frac{1}{n}\sum y_i$$

$$s_{xx} = \frac{1}{n-1}\sum(x_i-\bar{x})^2, \quad s_{xy} = \frac{1}{n-1}\sum(x_i-\bar{x})(y_i-\bar{y})$$

### Total Squared Error

$$\text{SSE} = S(a,b) = \sum_{i=1}^n (y_i - ax_i - b)^2$$

### Residuals

$$e_i = y_i - \hat{y}_i = y_i - \hat{a}x_i - \hat{b}$$

### Coefficient of Determination

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### Regression to the Mean (Standardized Coordinates)

$$\hat{v} = \rho \cdot u, \qquad \rho = \frac{s_{xy}}{\sqrt{s_{xx} s_{yy}}}$$

### Exponential Transformation

$$y = ce^{ax} \implies \ln(y) = ax + \ln(c) \implies \text{linear regression on } (x_i, \ln y_i)$$

---

## Topic 15: Quick Summary — Class 26 Linear Regression

- **Simple linear regression** fits $y = ax + b$ to bivariate data $(x_i, y_i)$ by minimizing the total squared error $\sum(y_i - ax_i - b)^2$.
- **Least squares estimates:** $\hat{a} = s_{xy}/s_{xx}$, $\hat{b} = \bar{y} - \hat{a}\bar{x}$.
- The fitted line always passes through the **point of means** $(\bar{x}, \bar{y})$.
- **"Linear" in linear regression** refers to linearity in parameters, not the shape of the curve.
- **Residuals** $e_i = y_i - \hat{y}_i$ should appear randomly scattered with constant variance (**homoscedasticity**).
- **Heteroscedastic** data has non-constant residual variance; transform data before applying OLS.
- **Polynomial regression** extends least squares to fit $y = a_d x^d + \cdots + a_0$ by minimizing the same SSE criterion.
- **Exponential fitting:** transform via $\ln(y)$ to linearize, then apply simple linear regression, then back-transform.
- **Overfitting:** a polynomial of degree $n-1$ always fits $n$ points exactly but generalizes poorly; prefer the simplest adequate model.
- **$R^2$** measures goodness of fit (fraction of variability explained). Increases with polynomial degree — use with caution.
- **Least squares = MLE** when errors are normal: $\varepsilon_i \sim N(0, \sigma^2)$.
- **Regression to the mean:** for $|\rho| < 1$, extreme values of $x$ predict $y$ values closer to $\bar{y}$; $\hat{v} = \rho u$ in standardized coordinates.
- **Practical dangers of regression to the mean:** useless interventions appear effective; punishment appears better than reward — all due to mathematical necessity, not causation.
- **R function:** `lm()` performs all forms of linear regression; `summary(fit)$r.squared` gives $R^2$.

---

## Appendix: The R Code Reference for Linear Regression

```r
# ===== SIMPLE LINEAR REGRESSION =====

# Data
x = c(1, 2, 4)
y = c(3, 1, 4)

# Compute manually
n = length(x)
xbar = mean(x)
ybar = mean(y)
sxx = var(x)                                  # = sum((x - xbar)^2)/(n-1)
sxy = sum((x - xbar) * (y - ybar)) / (n - 1) # sample covariance

a_hat = sxy / sxx
b_hat = ybar - a_hat * xbar

cat("Slope:", a_hat, "Intercept:", b_hat, "\n")

# Using lm()
fit = lm(y ~ x)
summary(fit)

# ===== POLYNOMIAL REGRESSION =====

fit2 = lm(y ~ x + I(x^2))           # quadratic
fit3 = lm(y ~ x + I(x^2) + I(x^3)) # cubic

# ===== EXPONENTIAL REGRESSION =====

z = log(y)
fit_exp = lm(z ~ x)
a = coef(fit_exp)["x"]
b = coef(fit_exp)["(Intercept)"]
cat("y = exp(", b, ") * exp(", a, "* x)\n")

# ===== GOODNESS OF FIT =====
summary(fit)$r.squared
plot(fit)            # 4 diagnostic plots

# ===== PREDICTIONS =====
predict(fit, newdata = data.frame(x = 5))

# ===== RESIDUALS =====
res = residuals(fit)
plot(x, res, main = "Residual plot")
abline(h = 0, col = "red")
```

---

*End of MIT 18.05 Study Notes — Class 26: Linear Regression.*

*Source: MIT OpenCourseWare, 18.05 Introduction to Probability and Statistics, Spring 2022. Jeremy Orloff and Jonathan Bloom.*

*These notes cover: Least Squares Fitting (lines, polynomials, exponentials), Residuals, Homoscedasticity, Overfitting, $R^2$, Regression to the Mean, and the MLE-Least Squares equivalence under normal errors.*
