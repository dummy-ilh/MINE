# Chapter 1: Introduction


## What is a Linear Model?

A **linear model** expresses a dependent variable `y` as a function of `k` independent variables and parameters:

```
y = β₁X₁ + β₂X₂ + ... + βₖXₖ + ε
```

- `y` — dependent (study) variable  
- `X₁, X₂, ..., Xₖ` — independent (explanatory) variables  
- `β₁, β₂, ..., βₖ` — parameters  
- `ε` — random error term (accounts for the stochastic nature of the relationship)

> **Key rule:** A model is *linear* if it is linear **in its parameters** — not necessarily in the variables. For example, `y = β₁logX₁ + β₂X₂² + ε` is still a linear model.

```md
# 📌 Why Linearity in Parameters Matters (Regression)

## 🔥 Core Idea
A regression model is **linear** if it is **linear in the parameters (β)** — not necessarily in the input variables (X).

---

## 🧠 Key Reason

### 1. What are we solving for?
- **X (features)** → known / observed  
- **β (parameters)** → unknown → must be estimated  

👉 The real problem:
> Can we solve for β efficiently?

---

## ⚡ Case 1: Linear in Parameters

### Model
\[
y = X\beta + \epsilon
\]

### Solution (Closed Form)
\[
\hat{\beta} = (X^T X)^{-1} X^T y
\]

### Properties
- ✅ Exact solution  
- ✅ Fast computation  
- ✅ No iteration needed  

👉 This is **Ordinary Least Squares (OLS)**

---

## ❌ Case 2: Nonlinear in Parameters

If β appears inside nonlinear operations:
- No closed-form solution  
- Requires iterative optimization:
  - Gradient Descent
  - Newton-Raphson  

### Issues
- ❌ Slow convergence  
- ❌ Possible instability  
- ❌ Local minima  

---

## ⚡ Intuition

Any linear model can be written as:

\[
y = \beta_1 f_1(X) + \beta_2 f_2(X) + \dots + \beta_k f_k(X) + \epsilon
\]

Where:
- \(f_i(X)\) = any transformation of X:
  - \(X\), \(X^2\), \(\log X\), \(\sin X\), etc.

👉 β’s act as **weights** over transformed features

---

## 🎯 Deep Insight

> Linear regression = **linear combination of features**, NOT linearity in X

---

## 🚀 Implications

You can freely use:
- Polynomial features → \(X^2, X^3\)
- Log transformations → \(\log X\)
- Interaction terms → \(X_1 X_2\)

👉 Still a **linear model** as long as β’s are linear

---

## 🌍 Real-World Power

This enables:
- Polynomial Regression  
- Basis Function Expansion  
- Kernel Methods (advanced)

👉 Linear models are far more expressive than they appear

---

## 🧠 Final Takeaway

A model is linear **iff**:

\[
y = \sum_{i=1}^{k} \beta_i \cdot f_i(X) + \epsilon
\]

### ✔️ Flexible in X  
### ❌ Strict in β  

👉 Linearity is about **parameters**, because that’s what we solve for
```

---

## What is Regression Analysis?

Regression analysis works **backwards** — the model exists in nature but is unknown. We collect data first, then use statistical techniques to recover the parameters.

> The word "regression" literally means *to move in the backward direction* — from data back to the model.


The term comes from statistics (by Francis Galton):

He observed that extreme values tend to move back toward the average
Called it “regression toward the mean”

👉 Name stuck, even though today it means much more


---

## Types of Regression

| Type | Description |
|---|---|
| **Simple** | One explanatory variable |
| **Multiple** | Two or more explanatory variables |
| **Univariate** | One response variable |
| **Multivariate** | Two or more response variables |
| **Linear** | Parameters enter linearly (after transformation if needed) |
| **Nonlinear** | Some parameters appear nonlinearly and cannot be transformed |
| **Logistic** | Response variable is qualitative (binary) |
| **Analysis of Variance (ANOVA)** | All explanatory variables are qualitative |
| **Analysis of Covariance (ANCOVA)** | Mix of qualitative and quantitative explanatory variables |

---

## Steps in Regression Analysis

1. **State the problem** — Define objectives clearly; misunderstanding here leads to wrong inferences.
2. **Choose relevant variables** — Identify which variables are response vs. explanatory.
3. **Collect data** — Decide on quantitative vs. qualitative measurement; note that converting quantitative → qualitative loses information.
4. **Specify the model** — Choose a linear or nonlinear form; many nonlinear models can be linearised via transformations.
5. **Choose a fitting method** — Most common: **least squares** (no distributional assumption needed). Others: maximum likelihood, ridge, principal components.
6. **Fit the model** — Substitute estimated parameters `β̂` to get the fitted equation; use it for prediction or forecasting.
7. **Validate and criticise** — Check assumptions iteratively; outputs diagnose whether inputs need revision.
8. **Apply the model** — Use for policy analysis, forecasting, or understanding variable relationships.

---

## Parameter Estimation Methods

| Method | Requires distribution of y? |
|---|---|
| Least Squares | No |
| Method of Moments | No |
| Maximum Likelihood | Yes |

---

## Key Distinctions

- **Fitted value** — `ŷ` computed for an observation already in the dataset.  
- **Predicted value** — `ŷ` computed for any new set of explanatory variable values.  
- **Forecasted value** — Predicted value where the explanatory variables are future values.  
- Avoid predictions **far outside the range** of the observed data.


