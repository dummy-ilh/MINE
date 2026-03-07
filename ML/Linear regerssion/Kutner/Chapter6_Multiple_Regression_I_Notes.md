# Chapter 6: Multiple Regression I


---

## Table of Contents
1. [Why Multiple Regression?](#why-multiple-regression)
2. [Multiple Regression Models](#multiple-regression-models)
   - First-Order Model (Two Predictors)
   - First-Order Model (Many Predictors)
   - General Linear Regression Model
3. [Matrix Form of the Model](#matrix-form)
4. [Estimating Regression Coefficients](#estimating-coefficients)
5. [Fitted Values and Residuals](#fitted-values-and-residuals)
6. [Analysis of Variance](#analysis-of-variance)
7. [Inferences About Regression Parameters](#inferences)
8. [Estimating Mean Response & Predicting New Observations](#estimation-and-prediction)
9. [Diagnostics and Remedial Measures](#diagnostics)
10. [Full Worked Example: Dwaine Studios](#dwaine-studios-example)

---

## 1. Why Multiple Regression? {#why-multiple-regression}

### The Problem with One Predictor

Imagine trying to predict **how much a portrait studio will sell** using only the number of children in the city. You'd miss important information like local income levels. A single predictor model is often *too imprecise* to be useful because:

- Multiple variables genuinely affect the response
- Ignoring predictors inflates the error variance $\sigma^2$
- Predictions become wide and unreliable

### When Do We Use Multiple Regression?

**Observational settings:** The predictors aren't controlled — e.g., predicting hospital length-of-stay from patient demographics.

**Experimental settings:** The experimenter controls predictors simultaneously — e.g., testing both drug dose *and* administration method on patient outcomes.

> 🔑 **Key Insight:** Multiple regression doesn't just improve fit — it allows you to estimate the effect of one variable *while holding others constant*, which is the statistical analog of a controlled experiment.

### Classic Examples from the Textbook
- **Consumer finance chain:** Response = operating cost; Predictors = loans outstanding, new applications
- **Tractor sales:** Response = volume; Predictors = number of farms, crop production
- **Short children study:** Response = plasma growth hormone; Predictors = gender, age, body measurements (14 total!)

### General Linear Regression Model 
The term "General Linear Regression Model" is crucial. It describes any model that is **linear in its parameters**, even if it includes non-linear transformations of the predictor variables or interaction terms. This means it can represent a wide variety of relationships.
Examples of models falling under the general linear regression model:
* **Polynomial regression:** $Y_i = \beta_0 + \beta_1 X_i + \beta_2 X_i^2 + \epsilon_i$ (Here $X_1 = X$ and $X_2 = X^2$, still linear in $\beta$'s).
* **Interaction terms:** $Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \beta_3 X_{i1}X_{i2} + \epsilon_i$.
* **Dummy variables:** For categorical predictors (e.g., $Y_i = \beta_0 + \beta_1 \text{Gender}_i + \beta_2 \text{Age}_i + \epsilon_i$, where Gender is 0 or 1).
* **Transformed variables:** $Y_i = \beta_0 + \beta_1 \log(X_{i1}) + \beta_2 \sqrt{X_{i2}} + \epsilon_i$.

The key is that the $\beta$ coefficients are multiplied by known constants or functions of $X$, and the terms are added.
---

## 2. Multiple Regression Models {#multiple-regression-models}
This is the simplest form of multiple linear regression. A first-order model implies that the relationship between the response variable and each predictor variable is linear
### First-Order Model with Two Predictor Variables

When we have two predictors $X_1$ and $X_2$:

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \varepsilon_i \tag{6.1}$$
Where:
* $Y_i$: The dependent variable for the $i$-th observation.
* $X_{i1}$, $X_{i2}$: The two predictor (independent) variables for the $i$-th observation.
* $\beta_0$: The intercept (mean of $Y$ when $X_1=0$ and $X_2=0$).
* $\beta_1$: The partial regression coefficient for $X_1$. It represents the change in the mean of $Y$ for a one-unit increase in $X_1$, *holding $X_2$ constant*.
* $\beta_2$: The partial regression coefficient for $X_2$. It represents the change in the mean of $Y$ for a one-unit increase in $X_2$, *holding $X_1$ constant*.
* $\epsilon_i$: The random error term for the $i$-th observation.
* 
The **response function** (assuming $E\{\varepsilon_i\} = 0$) is:

$$E\{Y\} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 \tag{6.2}$$

This defines a **plane** in 3D space (not a line!).

📊 **Example:** A sales promotion model:

$$E\{Y\} = 10 + 2X_1 + 5X_2 \tag{6.3}$$

where $Y$ = test market sales (in \$10K), $X_1$ = point-of-sale expenditures (\$K), $X_2$ = TV expenditures (\$K).

![Response plane for sales promotion example](https://github.com/dummy-ilh/MINE/blob/main/ML/Linear%20regerssion/Kutner/images/61.PNG)

*Each point on this plane is the mean response $E\{Y\}$ at that combination of $X_1, X_2$. The vertical distance from a data point $Y_i$ to the plane is the error term $\varepsilon_i$.*

---

### Interpreting Regression Coefficients

| Parameter | Meaning |
|-----------|---------|
| $\beta_0$ | Mean of $Y$ when $X_1 = X_2 = 0$ (only meaningful if origin is in scope) |
| $\beta_1$ | Change in mean $Y$ per unit increase in $X_1$, **holding $X_2$ constant** |
| $\beta_2$ | Change in mean $Y$ per unit increase in $X_2$, **holding $X_1$ constant** |

These are called **partial regression coefficients** because they capture the partial effect of one variable when others are held fixed.

**Calculus confirmation:**
$$\frac{\partial E\{Y\}}{\partial X_1} = \beta_1 \qquad \frac{\partial E\{Y\}}{\partial X_2} = \beta_2$$

📊 **Example (continued):** In equation (6.3):
- $\beta_1 = 2$: If point-of-sale spending increases by \$1K while TV spending stays fixed, mean sales increase by \$2 (i.e., \$20K).
- $\beta_2 = 5$: If TV spending increases by \$1K while point-of-sale spending stays fixed, mean sales increase by \$5 (i.e., \$50K).

---

### Additive Effects vs. Interaction

In the first-order model, the effect of $X_1$ does **not depend** on the level of $X_2$ — this is called **additive effects** (or "no interaction").

> ⚠️ **Common Misconception:** "First-order" refers to the power of the predictors (all linear), not that there's only one predictor. A first-order model can have many predictors.

---

### First-Order Model with $p-1$ Predictor Variables

Generalizing to $p-1$ predictors:

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_{p-1} X_{i,p-1} + \varepsilon_i \tag{6.5}$$

Or more compactly:

$$Y_i = \beta_0 + \sum_{k=1}^{p-1} \beta_k X_{ik} + \varepsilon_i \tag{6.5a}$$

Setting $X_{i0} \equiv 1$:

$$Y_i = \sum_{k=0}^{p-1} \beta_k X_{ik} + \varepsilon_i \tag{6.5b}$$

The response function is a **hyperplane** — impossible to visualize for $p > 3$, but mathematically identical in spirit to the 2-predictor case.

The parameter $\beta_k$ still means: *change in mean $Y$ with a unit increase in $X_k$, with all other predictors held constant.*

> 💡 **Special case check:** When $p-1 = 1$, equation (6.5) reduces to simple linear regression $Y_i = \beta_0 + \beta_1 X_{i1} + \varepsilon_i$. ✓

---

### The General Linear Regression Model

The $X$ variables in a regression model don't have to represent *different physical predictors*. They can be:

#### (a) Multiple distinct predictors
Standard case — $X_1, X_2, \ldots$ are different variables.

#### (b) Qualitative (Categorical) Predictor Variables

Use **indicator (dummy) variables** that take values 0 or 1.

📊 **Example:** Predict hospital stay length ($Y$) from age ($X_1$) and gender ($X_2$):

$$X_{i2} = \begin{cases} 1 & \text{if patient female} \\ 0 & \text{if patient male} \end{cases}$$

Model:

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \varepsilon_i \tag{6.9}$$

This gives two **parallel regression lines**:

$$E\{Y\} = \beta_0 + \beta_1 X_1 \quad \text{(male patients, } X_2 = 0\text{)} \tag{6.10a}$$
$$E\{Y\} = (\beta_0 + \beta_2) + \beta_1 X_1 \quad \text{(female patients, } X_2 = 1\text{)} \tag{6.10b}$$

The lines have the **same slope** but different **intercepts** — $\beta_2$ represents the gender gap.

> 📌 **Rule for indicator variables:** A qualitative variable with $c$ classes needs $c-1$ indicator variables.

**Example with 3-class disability variable** (not disabled / partially disabled / fully disabled):

$$X_3 = \begin{cases}1 & \text{not disabled}\\ 0 & \text{otherwise}\end{cases} \qquad X_4 = \begin{cases}1 & \text{partially disabled}\\ 0 & \text{otherwise}\end{cases}$$

Fully disabled is the **reference category** (when $X_3 = X_4 = 0$). The full model becomes:

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \beta_3 X_{i3} + \beta_4 X_{i4} + \varepsilon_i \tag{6.11}$$

#### (c) Polynomial Regression

Include squared (or higher-power) terms of a predictor:

$$Y_i = \beta_0 + \beta_1 X_i + \beta_2 X_i^2 + \varepsilon_i \tag{6.12}$$

This creates a **curvilinear** response function, yet it's still in the general linear regression framework! Just let $X_{i1} = X_i$ and $X_{i2} = X_i^2$.

> 💡 **Why is this "linear"?** "Linear" refers to *linearity in the parameters* ($\beta$'s), not in the predictors. The model is linear in $\beta_0, \beta_1, \beta_2$ — that's what matters for OLS to work.

#### (d) Transformed Variables

$$\log Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \beta_3 X_{i3} + \varepsilon_i \tag{6.13}$$

Let $Y_i' = \log Y_i$ and this becomes a standard linear model with transformed response.

$$Y_i = \frac{1}{\beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \varepsilon_i} \tag{6.14}$$

Let $Y_i' = 1/Y_i$ → standard linear model.

#### (e) Interaction Effects

When effects are **not additive** — the impact of $X_1$ depends on the level of $X_2$:

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \beta_3 X_{i1} X_{i2} + \varepsilon_i \tag{6.15}$$

Let $X_{i3} = X_{i1} \cdot X_{i2}$ (cross-product term) and this is again a standard linear model.

#### (f) Combination Model

A full second-order model with two predictors (linear + quadratic + interaction):

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i1}^2 + \beta_3 X_{i2} + \beta_4 X_{i2}^2 + \beta_5 X_{i1} X_{i2} + \varepsilon_i \tag{6.16}$$

Define $Z_{i1} = X_{i1}$, $Z_{i2} = X_{i1}^2$, $Z_{i3} = X_{i2}$, $Z_{i4} = X_{i2}^2$, $Z_{i5} = X_{i1}X_{i2}$ and it reduces to the standard form.

---

### Formal Definition of the General Linear Regression Model

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \cdots + \beta_{p-1} X_{i,p-1} + \varepsilon_i \tag{6.7}$$

where:
- $\beta_0, \beta_1, \ldots, \beta_{p-1}$ are unknown **parameters**
- $X_{i1}, \ldots, X_{i,p-1}$ are **known constants** (observed values)
- $\varepsilon_i \sim \text{i.i.d. } N(0, \sigma^2)$
- $i = 1, \ldots, n$

The response function is:
$$E\{Y\} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_{p-1} X_{p-1} \tag{6.8}$$

> 🔑 **Key:** $p$ = total number of parameters (including $\beta_0$). So $p-1$ = number of predictors. Degrees of freedom for error = $n - p$.

### What Makes a Model Nonlinear?

A **nonlinear** regression model *cannot* be written as $Y_i = \sum c_{ik}\beta_k + \varepsilon_i$. Example:

$$Y_i = \beta_0 \exp(\beta_1 X_i) + \varepsilon_i$$

This is nonlinear in parameters. General linear models (even with polynomial/interaction terms) *are* linear in parameters — that's the key distinction.

---

## 3. General Linear Regression Model in Matrix Form {#matrix-form}

### Why Matrix Notation?

Matrix algebra lets us write results for a model with 2 predictors **identically** to a model with 100 predictors. The formulas look the same! The computer does the heavy lifting.

### Setting Up the Matrices

For the general model with $n$ observations and $p-1$ predictors:

$$\underset{n \times 1}{\mathbf{Y}} = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix} \qquad \underset{n \times p}{\mathbf{X}} = \begin{bmatrix} 1 & X_{11} & X_{12} & \cdots & X_{1,p-1} \\ 1 & X_{21} & X_{22} & \cdots & X_{2,p-1} \\ \vdots & \vdots & \vdots & & \vdots \\ 1 & X_{n1} & X_{n2} & \cdots & X_{n,p-1} \end{bmatrix} \tag{6.18}$$

$$\underset{p \times 1}{\boldsymbol{\beta}} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_{p-1} \end{bmatrix} \qquad \underset{n \times 1}{\boldsymbol{\varepsilon}} = \begin{bmatrix} \varepsilon_1 \\ \varepsilon_2 \\ \vdots \\ \varepsilon_n \end{bmatrix}$$

> 📌 **Remember the 1s:** The first column of $\mathbf{X}$ is all 1s — this corresponds to the intercept $\beta_0$, since $\beta_0 \cdot 1 = \beta_0$.

### The Matrix Model

$$\underset{n\times 1}{\mathbf{Y}} = \underset{n\times p}{\mathbf{X}} \underset{p\times 1}{\boldsymbol{\beta}} + \underset{n\times 1}{\boldsymbol{\varepsilon}} \tag{6.19}$$

### Error Term Properties

$$E\{\boldsymbol{\varepsilon}\} = \mathbf{0} \qquad \boldsymbol{\sigma}^2\{\boldsymbol{\varepsilon}\} = \sigma^2 \mathbf{I}_{n\times n} \tag{6.21}$$

The variance-covariance matrix is diagonal with $\sigma^2$ on the diagonal — errors are **independent** and have **equal variance**.

Consequently:
$$E\{\mathbf{Y}\} = \mathbf{X}\boldsymbol{\beta} \tag{6.20}$$
$$\boldsymbol{\sigma}^2\{\mathbf{Y}\} = \sigma^2 \mathbf{I} \tag{6.21}$$

---

## 4. Estimating Regression Coefficients {#estimating-coefficients}

### Least Squares Criterion

We minimize the sum of squared residuals:

$$Q = \sum_{i=1}^n \left(Y_i - \beta_0 - \beta_1 X_{i1} - \cdots - \beta_{p-1} X_{i,p-1}\right)^2 \tag{6.22}$$

### Normal Equations (Matrix Form)

Taking derivatives and setting equal to zero gives the **normal equations**:

$$\mathbf{X'Xb} = \mathbf{X'Y} \tag{6.24}$$

### Least Squares Estimator

$$\underset{p\times 1}{\mathbf{b}} = (\mathbf{X'X})^{-1} \mathbf{X'Y} \tag{6.25}$$

> 💡 **Analogy to Simple Regression:** In simple regression, $b_1 = S_{XY}/S_{XX}$. In matrix form, $\mathbf{b} = (\mathbf{X'X})^{-1}\mathbf{X'Y}$ — the same logic scaled up!

### Properties of $\mathbf{b}$

- **Unbiased:** $E\{\mathbf{b}\} = \boldsymbol{\beta}$ (equation 6.44)
- **Minimum variance** among all unbiased linear estimators (Gauss-Markov Theorem)
- Same as **Maximum Likelihood** estimators when errors are normal

### MLE Equivalence

The likelihood function generalizes directly:

$$L(\boldsymbol{\beta}, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{n/2}} \exp\left[-\frac{1}{2\sigma^2} \sum_{i=1}^n (Y_i - \beta_0 - \beta_1 X_{i1} - \cdots - \beta_{p-1} X_{i,p-1})^2\right] \tag{6.26}$$

Maximizing this yields the same $\mathbf{b}$ as OLS.

---

## 5. Fitted Values and Residuals {#fitted-values-and-residuals}

### Fitted Values

$$\hat{\mathbf{Y}} = \mathbf{Xb} \tag{6.28}$$

Or equivalently using the **Hat Matrix** $\mathbf{H}$:

$$\hat{\mathbf{Y}} = \mathbf{HY} \tag{6.30}$$

where:

$$\underset{n\times n}{\mathbf{H}} = \mathbf{X(X'X)^{-1}X'} \tag{6.30a}$$

> 💡 **Why "Hat Matrix"?** Because $\mathbf{H}$ puts the "hat" on $\mathbf{Y}$: it transforms $\mathbf{Y}$ into $\hat{\mathbf{Y}}$.

### Residuals

$$\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}} = \mathbf{Y} - \mathbf{Xb} \tag{6.29}$$

Or in terms of $\mathbf{H}$:

$$\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{Y} \tag{6.31}$$

### Variance-Covariance of Residuals

$$\boldsymbol{\sigma}^2\{\mathbf{e}\} = \sigma^2(\mathbf{I} - \mathbf{H}) \tag{6.32}$$

Estimated by:

$$\mathbf{s}^2\{\mathbf{e}\} = MSE(\mathbf{I} - \mathbf{H}) \tag{6.33}$$

> ⚠️ **Important:** Residuals are NOT independent! Their covariance matrix $\sigma^2(\mathbf{I}-\mathbf{H})$ is generally not diagonal.

---

## 6. Analysis of Variance {#analysis-of-variance}

### Sums of Squares

These have the exact same formulas as simple regression in matrix form:

| Source | Sum of Squares | df | Mean Square |
|--------|---------------|-----|-------------|
| Regression | $SSR = \mathbf{b'X'Y} - \frac{1}{n}\mathbf{Y'JY}$ | $p-1$ | $MSR = \frac{SSR}{p-1}$ |
| Error | $SSE = \mathbf{Y'Y} - \mathbf{b'X'Y}$ | $n-p$ | $MSE = \frac{SSE}{n-p}$ |
| Total | $SSTO = \mathbf{Y'Y} - \frac{1}{n}\mathbf{Y'JY}$ | $n-1$ | |

where $\mathbf{J}$ is an $n \times n$ matrix of all 1s.

> 🔑 **Degrees of Freedom:**
> - Error df = $n - p$ (because we estimate $p$ parameters: $\beta_0, \beta_1, \ldots, \beta_{p-1}$)
> - Regression df = $p - 1$ (number of $X$ variables)

### Expected Mean Squares

$$E\{MSE\} = \sigma^2 \quad \text{(always)}$$

$$E\{MSR\} = \sigma^2 + \text{nonnegative quantity} \quad \text{(equals } \sigma^2 \text{ iff all } \beta_k = 0\text{)}$$

---

### The F Test for Overall Regression Relation

**Question:** Is there any linear relationship between $Y$ and *any* of the predictors?

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_{p-1} = 0$$
$$H_a: \text{not all } \beta_k \text{ equal zero} \tag{6.39a}$$

**Test statistic:**

$$F^* = \frac{MSR}{MSE} \tag{6.39b}$$

**Decision rule** at level $\alpha$:

$$\text{If } F^* \leq F(1-\alpha;\, p-1,\, n-p): \text{ conclude } H_0$$
$$\text{If } F^* > F(1-\alpha;\, p-1,\, n-p): \text{ conclude } H_a \tag{6.39c}$$

> ⚠️ **Watch out:** Rejecting $H_0$ tells you the model is *better than nothing*, not that it's a *good* model. A significant F-test is necessary but not sufficient for a useful model.

---

### Coefficient of Multiple Determination ($R^2$)

$$R^2 = \frac{SSR}{SSTO} = 1 - \frac{SSE}{SSTO} \tag{6.40}$$

- Measures the **proportionate reduction in total variation** explained by all predictors together
- $0 \leq R^2 \leq 1$ (equation 6.41)
- $R^2 = 0$ when all $b_k = 0$; $R^2 = 1$ when all residuals are 0

> ⚠️ **Critical Warning: $R^2$ always increases (or stays the same) when you add more predictors**, even useless ones! This is because $SSE$ can only decrease (or stay equal) as more variables are added to a model.

### Adjusted $R^2$ (Adjusts for Number of Predictors)

$$R_a^2 = 1 - \frac{SSE/(n-p)}{SSTO/(n-1)} = 1 - \left(\frac{n-1}{n-p}\right)\frac{SSE}{SSTO} \tag{6.42}$$

- Uses mean squares instead of raw sums of squares
- **Can decrease** when a useless predictor is added (the penalty from losing a df can outweigh any tiny decrease in SSE)
- Better measure for model comparison than raw $R^2$

### Coefficient of Multiple Correlation ($R$)

$$R = \sqrt{R^2} \tag{6.43}$$

When $p - 1 = 1$ (simple regression), $R = |r|$ where $r$ is the simple correlation coefficient.

> 📌 **Caution about $R^2$:** A large $R^2$ doesn't mean:
> 1. The model is good for prediction (it might only fit within the observation region)
> 2. The right predictors were chosen
> 3. The model form is correct (could be curvature, etc.)

---

## 7. Inferences About Regression Parameters {#inferences}

### Variance-Covariance Matrix of $\mathbf{b}$

The true variance-covariance matrix:

$$\boldsymbol{\sigma}^2\{\mathbf{b}\} = \sigma^2(\mathbf{X'X})^{-1} \tag{6.46}$$

The **estimated** variance-covariance matrix:

$$\mathbf{s}^2\{\mathbf{b}\} = MSE(\mathbf{X'X})^{-1} \tag{6.48}$$

This matrix contains:
- **Diagonal elements:** $s^2\{b_k\}$ — estimated variance of each $b_k$
- **Off-diagonal elements:** $s\{b_k, b_{k'}\}$ — estimated covariances between pairs

So $s\{b_k\} = \sqrt{s^2\{b_k\}}$ is just the square root of the corresponding diagonal element.

---

### Interval Estimation of $\beta_k$

Sampling distribution:

$$\frac{b_k - \beta_k}{s\{b_k\}} \sim t(n-p) \tag{6.49}$$

**Confidence interval** for $\beta_k$ with coefficient $1-\alpha$:

$$b_k \pm t(1-\alpha/2;\, n-p)\, s\{b_k\} \tag{6.50}$$

---

### Tests for Individual $\beta_k$

$$H_0: \beta_k = 0 \qquad H_a: \beta_k \neq 0 \tag{6.51a}$$

**Test statistic:**

$$t^* = \frac{b_k}{s\{b_k\}} \tag{6.51b}$$

**Decision rule:**

$$\text{If } |t^*| \leq t(1-\alpha/2;\, n-p): \text{ conclude } H_0$$
$$\text{Otherwise: conclude } H_a \tag{6.51c}$$

> 💡 **Interpretation:** The t-test for $\beta_k = 0$ tests whether $X_k$ contributes to predicting $Y$ *given that all other predictors are already in the model*. This is different from asking whether $X_k$ is correlated with $Y$ in isolation!

---

### Joint Inferences (Bonferroni Method)

When estimating $g$ parameters simultaneously with family confidence coefficient $1-\alpha$:

$$b_k \pm B\, s\{b_k\} \tag{6.52}$$

where:

$$B = t(1-\alpha/2g;\, n-p) \tag{6.52a}$$

The Bonferroni correction widens each individual interval to maintain the family-wise confidence level.

---

## 8. Estimation of Mean Response & Prediction of New Observation {#estimation-and-prediction}

### Setup

For a specific combination of predictor values $X_{h1}, X_{h2}, \ldots, X_{h,p-1}$, define:

$$\mathbf{X}_h = \begin{bmatrix} 1 \\ X_{h1} \\ \vdots \\ X_{h,p-1} \end{bmatrix} \tag{6.53}$$

### Mean Response Estimation

The mean response to be estimated:

$$E\{Y_h\} = \mathbf{X}_h'\boldsymbol{\beta} \tag{6.54}$$

**Point estimate:**

$$\hat{Y}_h = \mathbf{X}_h'\mathbf{b} \tag{6.55}$$

**Variance of $\hat{Y}_h$:**

$$\sigma^2\{\hat{Y}_h\} = \sigma^2 \mathbf{X}_h'(\mathbf{X'X})^{-1}\mathbf{X}_h \tag{6.57}$$

Estimated by:

$$s^2\{\hat{Y}_h\} = MSE \cdot \mathbf{X}_h'(\mathbf{X'X})^{-1}\mathbf{X}_h \tag{6.58}$$

**Confidence interval for $E\{Y_h\}$:**

$$\hat{Y}_h \pm t(1-\alpha/2;\, n-p)\, s\{\hat{Y}_h\} \tag{6.59}$$

---

### Confidence Region for the Entire Regression Surface

The **Working-Hotelling** $1-\alpha$ confidence region covers all $\mathbf{X}_h$ combinations:

$$\hat{Y}_h \pm W s\{\hat{Y}_h\} \tag{6.60}$$

where:

$$W^2 = p \cdot F(1-\alpha;\, p,\, n-p) \tag{6.60a}$$

### Simultaneous Confidence Intervals for Multiple Mean Responses

**Method 1 — Working-Hotelling** (use when $\mathbf{X}_h$ vectors not specified in advance):

$$\hat{Y}_h \pm W s\{\hat{Y}_h\} \tag{6.61}$$

**Method 2 — Bonferroni** (use when $g$ specific vectors specified in advance):

$$\hat{Y}_h \pm B s\{\hat{Y}_h\}, \qquad B = t(1-\alpha/2g;\, n-p) \tag{6.62}$$

> 💡 **Which to use?** Compare $W$ and $B$ before running the analysis. The one giving narrower intervals wins. If $\mathbf{X}_h$ points aren't specified in advance, Working-Hotelling is safer.

---

### Prediction of a New Observation

**Prediction interval** for a single new observation $Y_{h(\text{new})}$:

$$\hat{Y}_h \pm t(1-\alpha/2;\, n-p)\, s\{\text{pred}\} \tag{6.63}$$

where:

$$s^2\{\text{pred}\} = MSE + s^2\{\hat{Y}_h\} = MSE(1 + \mathbf{X}_h'(\mathbf{X'X})^{-1}\mathbf{X}_h) \tag{6.63a}$$

### Prediction for Mean of $m$ New Observations

$$\hat{Y}_h \pm t(1-\alpha/2;\, n-p)\, s\{\text{predmean}\} \tag{6.64}$$

$$s^2\{\text{predmean}\} = \frac{MSE}{m} + s^2\{\hat{Y}_h\} = MSE\left(\frac{1}{m} + \mathbf{X}_h'(\mathbf{X'X})^{-1}\mathbf{X}_h\right) \tag{6.64a}$$

### Simultaneous Predictions for $g$ New Observations

**Scheffé method:**

$$\hat{Y}_h \pm S\, s\{\text{pred}\}, \qquad S^2 = g \cdot F(1-\alpha;\, g,\, n-p) \tag{6.65}$$

**Bonferroni method:**

$$\hat{Y}_h \pm B\, s\{\text{pred}\}, \qquad B = t(1-\alpha/2g;\, n-p) \tag{6.66}$$

---

### ⚠️ Caution About Hidden Extrapolations

> **This is a major pitfall unique to multiple regression!**

Even if a new $\mathbf{X}_h$ vector has each component within the observed range of each predictor *individually*, the combination $\mathbf{X}_h$ might fall **outside** the joint region of observation.

**Example (Figure 6.3):** A prediction point might have $X_1$ and $X_2$ each within their individual ranges, but the *combination* is far outside the ellipse of observed data. The model may not be appropriate there.

> 📌 **Rule:** Always check that $\mathbf{X}_h$ falls within the joint region of the data, not just within the marginal ranges of each predictor. (Chapter 10 discusses formal methods for this.)

---

## 9. Diagnostics and Remedial Measures {#diagnostics}

### Overview

Most diagnostic tools from simple regression (Chapter 3) carry over directly. We'll focus on what's new or different in the multiple regression setting.

---

### Scatter Plot Matrix

A **scatter plot matrix** (like Figure 6.4) shows all pairwise scatter plots in a grid:
- Upper/lower triangular entries: scatterplots of $Y$ vs each $X_k$, and $X_k$ vs $X_{k'}$
- Diagonal: variable name

**What to look for:**
- Linear vs. nonlinear relationships
- Heteroscedasticity (fan shapes)
- Outliers in any pair of variables
- Correlation between predictors (potential multicollinearity concern)

### Correlation Matrix

The **correlation matrix** complements the scatter plot matrix with numerical values:

$$\begin{bmatrix} 1 & r_{Y1} & r_{Y2} & \cdots & r_{Y,p-1} \\ r_{Y1} & 1 & r_{12} & \cdots & r_{1,p-1} \\ \vdots & \vdots & \vdots & & \vdots \\ r_{Y,p-1} & r_{1,p-1} & r_{2,p-1} & \cdots & 1 \end{bmatrix} \tag{6.67}$$

It's symmetric with 1s on the diagonal.

---

### Three-Dimensional Scatter Plots

For two predictors, a 3D point cloud (Figure 6.6) can reveal whether a response plane is appropriate. **Spinning** the plot to see from different perspectives can reveal patterns invisible from a single angle.

---

### Residual Plots

Plot residuals $e_i$ against:
1. **Fitted values $\hat{Y}_i$** — check for nonlinearity and nonconstant variance
2. **Each predictor $X_k$** — check adequacy of each predictor's role
3. **Omitted predictors** (variables not in model) — check if they should be included
4. **Interaction terms $X_j X_k$** — check for interaction effects not in the model
5. **Time sequence** (if applicable) — check for autocorrelation

> 💡 **Tip:** Also plot $|e_i|$ or $e_i^2$ against fitted values and predictors to check **variance constancy**.

---

### Correlation Test for Normality

Uses the correlation between ordered residuals and expected values under normality (equation 3.6). Compare to critical values in Table B.6.

---

### Brown-Forsythe Test for Constant Error Variance

- Divide data into two groups based on level of a predictor (low vs. high)
- Apply the BF test statistic (3.9) as in simple regression
- When variance depends on multiple predictors, regress squared residuals on all predictors and use the resulting $SSR^*$ with $SSE$ from the full model

---

### Breusch-Pagan Test for Constant Error Variance

- Regress squared residuals against the predictor variables
- Test statistic uses $SSR^*$ and $SSE$, now involving $q$ degrees of freedom where $q$ = number of predictors the variance might depend on

---

### F Test for Lack of Fit

When replicate observations are available (same $\mathbf{X}$ vector for multiple $Y$ values):

$$H_0: E\{Y\} = \beta_0 + \beta_1 X_1 + \cdots + \beta_{p-1} X_{p-1}$$
$$H_a: E\{Y\} \neq \beta_0 + \beta_1 X_1 + \cdots + \beta_{p-1} X_{p-1} \tag{6.68a}$$

$$F^* = \frac{SSLF/(c-p)}{SSPE/(n-c)} = \frac{MSLF}{MSPE} \tag{6.68b}$$

where $c$ = number of distinct $\mathbf{X}$ vectors.

**Decision:**
$$\text{If } F^* \leq F(1-\alpha;\, c-p,\, n-c): \text{ conclude } H_0$$
$$\text{If } F^* > F(1-\alpha;\, c-p,\, n-c): \text{ conclude } H_a \tag{6.68c}$$

> 📌 **No replicates?** Group cases with similar $\mathbf{X}_h$ vectors into pseudo-replicates and proceed similarly.

---

### Remedial Measures

When the model is inadequate:

1. **Add curvature terms:** Add $X_k^2$ to capture a quadratic effect
2. **Add interaction terms:** Add $X_j X_k$ if residuals vs. $X_j X_k$ show a pattern
3. **Transform $Y$:** Use Box-Cox procedure — minimize $SSE$ over different values of $\lambda$
4. **Transform $X_k$:** Use Box-Tidwell iterative approach for predictor transformations

---

## 10. Full Worked Example: Dwaine Studios {#dwaine-studios-example}

### Setting

Dwaine Studios operates portrait studios in 21 medium-sized cities. They want to predict **sales** ($Y$, in \$K) from:
- $X_1$ = TARGTPOP = number of persons 16 or younger (in thousands)
- $X_2$ = DISPOINC = per capita disposable income (in \$K)

**Model:**

$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \varepsilon_i \tag{6.69}$$

---

### Step 1: Exploratory Analysis

From Figure 6.4 (scatter plot matrix):
- Clear **positive linear relationship** between $Y$ and $X_1$ ($r = .945$)
- Moderate **positive linear relationship** between $Y$ and $X_2$ ($r = .836$)
- Some correlation between $X_1$ and $X_2$ ($r = .781$)

The 3D point cloud (Figure 6.6) supports a response plane as a reasonable fit.

---

### Step 2: Matrix Calculations

$$\mathbf{X} = \begin{bmatrix} 1 & 68.5 & 16.7 \\ 1 & 45.2 & 16.8 \\ \vdots & \vdots & \vdots \\ 1 & 52.3 & 16.0 \end{bmatrix}, \quad \mathbf{Y} = \begin{bmatrix} 174.4 \\ 164.4 \\ \vdots \\ 166.5 \end{bmatrix} \tag{6.70}$$

$$\mathbf{X'X} = \begin{bmatrix} 21.0 & 1,302.4 & 360.0 \\ 1,302.4 & 87,707.9 & 22,609.2 \\ 360.0 & 22,609.2 & 6,190.3 \end{bmatrix}, \quad \mathbf{X'Y} = \begin{bmatrix} 3,820 \\ 249,643 \\ 66,073 \end{bmatrix} \tag{6.71, 6.72}$$

$$(\mathbf{X'X})^{-1} = \begin{bmatrix} 29.7289 & .0722 & -1.9926 \\ .0722 & .00037 & -.0056 \\ -1.9926 & -.0056 & .1363 \end{bmatrix} \tag{6.73}$$

---

### Step 3: Estimated Regression Function

$$\mathbf{b} = (\mathbf{X'X})^{-1}\mathbf{X'Y} = \begin{bmatrix} -68.857 \\ 1.455 \\ 9.366 \end{bmatrix} \tag{6.76}$$

$$\boxed{\hat{Y} = -68.857 + 1.455 X_1 + 9.366 X_2}$$

**Interpretation:**
- $b_1 = 1.455$: Mean sales increase by \$1,455 for each additional thousand children in the community, holding income constant.
- $b_2 = 9.366$: Mean sales increase by \$9,366 for each additional \$1K in per-capita income, holding population constant.

---

### Step 4: Diagnostics 

**Residual plots show:**
- **(a) Residuals vs. $\hat{Y}$:** No systematic pattern → response plane is appropriate, variance appears constant
- **(b) Residuals vs. $X_1$:** Entirely consistent with good fit
- **(c) Residuals vs. $X_2$:** Consistent with good fit
- **(d) Residuals vs. $X_1 X_2$:** No systematic pattern → **no interaction effect** present

**Additional diagnostics (Figure 6.9):**
- **(a) Absolute residuals vs. $\hat{Y}$:** No indication of nonconstancy of error variance
- **(b) Normal probability plot:** Moderately linear; correlation coefficient of ordered residuals vs. expected = **.980** — exceeds the critical value of .9525 for $n=21, \alpha=.05$ → **normality assumption is reasonable**

✅ All diagnostics support using regression model (6.69).

---

### Step 5: ANOVA Table

| Source | SS | df | MS |
|--------|----|----|-----|
| Regression | 24,015.28 | 2 | 12,007.64 |
| Error | 2,180.93 | 18 | 121.1626 |
| Total | 26,196.21 | 20 | |

**F-test for overall regression:**

$$F^* = \frac{MSR}{MSE} = \frac{12,007.64}{121.1626} = 99.1$$

$F(.95; 2, 18) = 3.55$. Since $F^* = 99.1 > 3.55$, conclude $H_a$ → **sales are related to target population and income** ($p < .0001$).

**$R^2$ and $R_a^2$:**

$$R^2 = \frac{24,015.28}{26,196.21} = .917 \qquad R_a^2 = .907$$

The two predictors together explain **91.7%** of the variation in sales.

---

### Step 6: Estimation of Regression Parameters

Estimated variance-covariance matrix:

$$\mathbf{s}^2\{\mathbf{b}\} = 121.1626 \times (\mathbf{X'X})^{-1} = \begin{bmatrix} 3,602.0 & 8.748 & -241.43 \\ 8.748 & .0448 & -.679 \\ -241.43 & -.679 & 16.514 \end{bmatrix} \tag{6.78}$$

So:
- $s\{b_1\} = \sqrt{.0448} = .212$
- $s\{b_2\} = \sqrt{16.514} = 4.06$

**Joint 90% confidence intervals for $\beta_1$ and $\beta_2$** (Bonferroni, $g = 2$):

$$B = t(1 - .10/(2 \cdot 2); 18) = t(.975; 18) = 2.101$$

$$b_1 \pm B \cdot s\{b_1\}: \quad 1.455 \pm 2.101(0.212) \Rightarrow 1.01 \leq \beta_1 \leq 1.90$$
$$b_2 \pm B \cdot s\{b_2\}: \quad 9.366 \pm 2.101(4.06) \Rightarrow 0.84 \leq \beta_2 \leq 17.9$$

Both intervals are entirely positive — as expected from economic theory! ✓

---

### Step 7: Estimation of Mean Response

**Goal:** Estimate mean sales for cities with $X_{h1} = 65.4$ (thousands) and $X_{h2} = 17.6$ (\$K), with 95% CI.

$$\mathbf{X}_h = \begin{bmatrix} 1 \\ 65.4 \\ 17.6 \end{bmatrix}$$

**Point estimate:**

$$\hat{Y}_h = \mathbf{X}_h'\mathbf{b} = [1 \quad 65.4 \quad 17.6] \begin{bmatrix} -68.857 \\ 1.455 \\ 9.366 \end{bmatrix} = 191.10$$

**Estimated variance:**

$$s^2\{\hat{Y}_h\} = \mathbf{X}_h' \mathbf{s}^2\{\mathbf{b}\} \mathbf{X}_h = 7.656 \Rightarrow s\{\hat{Y}_h\} = 2.77$$

**95% CI:**

$$191.10 \pm t(.975; 18)(2.77) = 191.10 \pm 2.101(2.77) \Rightarrow 185.3 \leq E\{Y_h\} \leq 196.9$$

*We estimate with 95% confidence that mean sales in cities with these demographics are between \$185,300 and \$196,900.*

---

### Step 8: Prediction Limits for Two New Cities

| | City A | City B |
|--|--------|--------|
| $X_{h1}$ | 65.4 | 53.1 |
| $X_{h2}$ | 17.6 | 17.7 |

Both fall within the joint region of the 21 observed cities ✓

Using family confidence coefficient .90 (Bonferroni, $g=2$, $B = 2.101$):

$$s^2\{\text{pred}\} = MSE + s^2\{\hat{Y}_h\}$$

**City A:** $s^2\{\text{pred}\} = 121.1626 + 7.656 = 128.82 \Rightarrow s\{\text{pred}\} = 11.35$

$$191.10 \pm 2.101(11.35) \Rightarrow \boxed{167.3 \leq Y_{A(\text{new})} \leq 214.9}$$

**City B:** $\hat{Y}_h = 174.15$, $s\{\text{pred}\} = 11.93$

$$174.15 \pm 2.101(11.93) \Rightarrow \boxed{149.1 \leq Y_{B(\text{new})} \leq 199.2}$$

*Dwaine Studios considers these intervals useful for planning but would prefer tighter ones for individual city decisions — a consulting firm has been hired to find better predictor variables.*

---

## Key Formulas Reference Sheet

| Concept | Formula |
|---------|---------|
| Regression model | $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ |
| OLS estimator | $\mathbf{b} = (\mathbf{X'X})^{-1}\mathbf{X'Y}$ |
| Fitted values | $\hat{\mathbf{Y}} = \mathbf{Xb} = \mathbf{HY}$ |
| Hat matrix | $\mathbf{H} = \mathbf{X(X'X)^{-1}X'}$ |
| Residuals | $\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}} = (\mathbf{I}-\mathbf{H})\mathbf{Y}$ |
| MSE | $MSE = SSE/(n-p)$ |
| $\mathbf{s}^2\{\mathbf{b}\}$ | $MSE(\mathbf{X'X})^{-1}$ |
| CI for $\beta_k$ | $b_k \pm t(1-\alpha/2; n-p)\, s\{b_k\}$ |
| F test statistic | $F^* = MSR/MSE$ |
| $R^2$ | $SSR/SSTO$ |
| $R_a^2$ | $1 - (n-1)/(n-p) \cdot SSE/SSTO$ |
| CI for $E\{Y_h\}$ | $\hat{Y}_h \pm t(1-\alpha/2; n-p)\, s\{\hat{Y}_h\}$ |
| Prediction interval | $\hat{Y}_h \pm t(1-\alpha/2; n-p)\, s\{\text{pred}\}$ |

---

## Common Mistakes & Misconceptions

1. **"Adding predictors always helps"** → No! $R^2$ always increases, but $R_a^2$ may decrease. Each predictor costs a degree of freedom.

2. **"Significant t-test means the predictor is important"** → The t-test is conditional on all other predictors in the model. A predictor's significance can change dramatically when others are added or removed (this is the multicollinearity problem, covered in Chapter 10).

3. **"Prediction is safe anywhere in the marginal ranges of $X$s"** → False! Hidden extrapolation in the *joint* space is the danger. Always check the joint region.

4. **"Large $R^2$ means reliable predictions"** → Not if $MSE$ is still large in absolute terms, or if you're extrapolating.

5. **"The F-test and t-tests will always agree"** → Not necessarily! The F-test is for *all* parameters jointly; individual t-tests are marginal. They can give different answers.

6. **"A first-order model assumes a plane response surface"** → Only if there are exactly two predictors. With more, it's a hyperplane. With indicator or polynomial variables, the shape is more complex.

---

## Practice Problems

**Problem 1:** In a regression with $n=25$ and $p=4$ (intercept + 3 predictors), you find $SSE = 480$ and $SSTO = 1200$. 
- (a) Compute $R^2$, $R_a^2$, and $MSE$.
- (b) Test $H_0: \beta_1 = \beta_2 = \beta_3 = 0$ at $\alpha = .05$. Find the critical value.

**Problem 2:** Interpret $b_2 = 3.7$ in a model predicting salary ($Y$, \$K) from years of experience ($X_1$) and education level ($X_2$, years). What does $b_2 = 3.7$ mean in plain English?

**Problem 3:** You have a model with $X_1$ = age and $X_2$ = gender (1=female, 0=male). Write out the two response functions (one for males, one for females). What does $b_2$ represent?

**Problem 4:** A researcher adds 5 new predictors to a model. $R^2$ increases from .82 to .85. $n=50$, original $p=4$, new $p=9$. Did $R_a^2$ increase or decrease? Show the calculation.

---

The principles of diagnostics from Chapter 3 are equally, if not more, crucial in multiple regression. Violations of assumptions can be harder to spot and have more complex consequences.

* **Scatter Plot Matrix:** A matrix of scatter plots showing the relationship between each pair of predictor variables, and between each predictor and the response variable. Helps visualize pairwise correlations and detect gross non-linearity.
* **Three-Dimensional Scatter Plots:** Useful for visualizing relationships with two predictors and one response, but limited to three dimensions.
* **Residual Plots:**
    * **Residuals vs. Fitted Values ($\hat{Y}$):** Essential for checking constant variance and linearity.
    * **Residuals vs. Individual Predictors ($X_k$):** Helpful for detecting specific non-linearity or heteroscedasticity related to a single predictor.
    * **Residuals vs. Omitted Variables:** If you suspect an important variable is missing, plotting residuals against it can show if a pattern exists, indicating its importance.
    * **Residuals vs. Time Order:** For detecting autocorrelation.
* **Formal Tests:**
    * **Correlation Test for Normality (e.g., Shapiro-Wilk, Anderson-Darling):** Applied to the residuals to check the normality assumption.
    * **Brown-Forsythe Test for Constancy of Error Variance:** Robust test for heteroscedasticity across groups of residuals (e.g., grouped by predictor values).
    * **Breusch-Pagan Test for Constancy of Error Variance:** A more general test for heteroscedasticity, checking if variance is related to predictor variables.
    * **F Test for Lack of Fit:** Applicable if there are replicated observations in the multi-dimensional predictor space. Tests the appropriateness of the chosen functional form.
* **Remedial Measures:** The same remedial measures apply as in simple linear regression:
    * **Transformations:** Of $Y$ (for linearity, variance, normality) or $X$'s (for linearity).
    * **Weighted Least Squares (WLS):** To address heteroscedasticity.
    * **Robust Standard Errors:** To provide valid inferences in the presence of heteroscedasticity without changing point estimates.
    * **Dealing with Outliers and Influential Points:** Using measures like Cook's distance, DFFITS, DFBETAS to identify and potentially address unusual observations.
