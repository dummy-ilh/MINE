# Regression Analysis — Chapter 1: Introduction
*Based on lecture notes by Shalabh, IIT Kanpur*

---

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

---

## What is Regression Analysis?

Regression analysis works **backwards** — the model exists in nature but is unknown. We collect data first, then use statistical techniques to recover the parameters.

> The word "regression" literally means *to move in the backward direction* — from data back to the model.

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

---

## The Iterative Process

```
Inputs (theory, data, assumptions)
        ↓
  Statistical methods
        ↓
Outputs (estimates, confidence regions, hypothesis tests)
        ↓
  Diagnosis & criticism
        ↓
  Revise inputs  →  repeat
```
