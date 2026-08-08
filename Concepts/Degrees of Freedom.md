# Degrees of Freedom — Master Notes (Interview-Ready)

## 1. The Core Intuition (Layman's Terms)

**Degrees of freedom = how many pieces of your data are actually "free to vary" once some constraints have been imposed.**

Think of it like a budget. You start with `n` data points — `n` independent pieces of information. But every time you *estimate a parameter from that same data*, you spend one unit of that freedom. The parameter estimate imposes a constraint that locks one data point's value in place (mathematically), even though you don't know in advance which one.

### The classic mean example
You have 5 numbers, and someone tells you their average is 10. How many of those 5 can you pick freely?

- You freely choose 4 of them.
- The 5th number is **forced** — it must be whatever value makes the average equal 10. You have zero choice left over it.

So you had 5 numbers, but only **4 degrees of freedom** once the mean was estimated from them. This is why sample variance uses `n − 1` in the denominator:

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2
$$

You "used up" 1 degree of freedom estimating $\bar{x}$ (the mean) before you could even compute a single deviation $(x_i - \bar{x})$.

## 2. Extending This to Linear Regression

In simple linear regression:

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i
$$

You are estimating **two** parameters from the data, not one:
- $\hat{\beta}_0$ (intercept)
- $\hat{\beta}_1$ (slope)

Each estimated parameter imposes one constraint on the residuals $(y_i - \hat{y}_i)$. Specifically, the least-squares normal equations guarantee:

$$
\sum_{i=1}^n (y_i - \hat{y}_i) = 0 \quad \text{and} \quad \sum_{i=1}^n x_i(y_i - \hat{y}_i) = 0
$$

These two equations mean the residuals aren't all free — if you know `n − 2` of them, the other 2 are mathematically determined by these constraints. That's exactly 2 degrees of freedom lost.

So when estimating the residual variance (mean squared error) to get an **unbiased** estimate of the true error variance $\sigma^2$:

$$
\hat{\sigma}^2 = \frac{1}{n-2}\sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{\text{RSS}}{n-2}
$$

You divide by `n − 2`, **not `n`**, because 2 parameters ($\beta_0$, $\beta_1$) were estimated from the same data before residuals could even be computed.

## 3. Why It Matters (Not Just Trivia)

If you divided by `n` instead of `n − 2`:
- You'd be treating the residuals as if all `n` of them carried independent information.
- But they don't — 2 "slots" were used up fitting the line.
- Result: $\hat{\sigma}^2$ would be **systematically biased downward** (too small), making your model look more precise/confident than it actually is.
- Downstream effects: your standard errors, t-statistics, confidence intervals, and p-values for $\hat{\beta}_0$, $\hat{\beta}_1$ would all be **wrong** (too narrow / too significant).

## 4. General Rule (Pattern to Remember)

> **Degrees of freedom = n − (number of parameters estimated from the data)**

| Scenario | # Parameters Estimated | Degrees of Freedom |
|---|---|---|
| Sample variance | 1 (the mean) | `n − 1` |
| Simple linear regression | 2 (intercept, slope) | `n − 2` |
| Multiple regression with `k` predictors | `k + 1` (k slopes + intercept) | `n − k − 1` |

This generalizes cleanly: multiple regression with `k` predictors estimates `k` slope coefficients plus 1 intercept = `k + 1` parameters, so you lose `k + 1` degrees of freedom.

## 5. Interview-Ready Answer (Memorize This)

> "Every parameter you estimate from your data imposes a constraint on the residuals, which costs you one degree of freedom. In simple linear regression we estimate 2 parameters — the intercept and slope — so we lose 2 degrees of freedom, and unbiased estimation of the error variance requires dividing by `n − 2`. This is the exact same logic as the `n − 1` correction in sample variance, which loses 1 degree of freedom estimating the mean. The general pattern is: degrees of freedom = n − (number of estimated parameters)."

## 6. Common Follow-Up Interview Questions

**Q: What happens if you use `n` instead of `n − 2`?**
A: You get a biased (too small) estimate of $\sigma^2$, which understates the true uncertainty in your model — standard errors, confidence intervals, and hypothesis tests all become artificially tight.

**Q: Why exactly 2 constraints and not more?**
A: The least-squares fitting process produces exactly 2 normal equations (one per parameter, from setting the partial derivatives of the loss to zero) — each is one independent constraint on the residual vector.

**Q: Does this generalize to any model, e.g., ridge/Lasso regression?**
A: The clean `n − p` formula strictly applies to OLS. For regularized models (ridge, Lasso), "effective degrees of freedom" is a more nuanced concept (often less than `p` due to shrinkage) and is computed differently — a good thing to mention if asked, to show depth.

**Q: Is this the same "degrees of freedom" as in a chi-square or t-distribution?**
A: Yes — it's the same underlying concept. The residual sum of squares, when scaled by $\sigma^2$, follows a chi-square distribution with `n − 2` degrees of freedom, which is exactly why `n − 2` shows up in the t-distribution used for hypothesis tests on $\hat{\beta}_0$ and $\hat{\beta}_1$.
