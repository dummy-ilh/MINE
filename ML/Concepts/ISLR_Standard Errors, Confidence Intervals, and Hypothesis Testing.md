
# 📘 ISLR — Linear Regression: Standard Errors, Confidence Intervals, and Hypothesis Testing

We continue our deep dive into **simple linear regression** using the Advertising dataset, focusing on **accuracy of estimates**, **confidence intervals**, and **hypothesis testing**.

---

## 1️⃣ Accuracy of an Estimate

### 🔹 Analogy: Estimating a Mean

- Suppose $Y$ has population mean $\mu$.
- Sample mean estimate:

$$
\hat{\mu} = \frac{1}{n} \sum_{i=1}^n y_i
$$

- **Unbiased:** $\mathbb{E}[\hat{\mu}] = \mu$
- But for a single sample, $\hat{\mu}$ may **overestimate or underestimate** $\mu$.

---

### 🔹 How to Quantify Accuracy?

- Compute **standard error (SE)**:

$$
\text{Var}(\hat{\mu}) = \text{SE}(\hat{\mu})^2 = \frac{\sigma^2}{n}
$$

- $\sigma^2$ = variance of $Y$
- Interpretation: **typical deviation of $\hat{\mu}$ from $\mu$**
- More observations → smaller SE → more accurate estimate

---

## 2️⃣ Standard Errors in Linear Regression

For simple linear regression:

$$
Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i, \quad \varepsilon_i \sim (0, \sigma^2)
$$

- Standard errors of coefficients:

$$
\text{SE}(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}, \quad
\text{SE}(\hat{\beta}_0)^2 = \sigma^2 \left[\frac{1}{n} + \frac{\bar{X}^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}\right]
$$

### 🔹 Intuition:

- $\text{SE}(\hat{\beta}_1)$ is smaller when $X$ values are **spread out** → easier to estimate slope
- $\text{SE}(\hat{\beta}_0)$ = similar to $\text{SE}(\hat{\mu})$ if $\bar{X} = 0$

### 🔹 Unknown $\sigma^2$:

- Usually unknown → estimated from data using **Residual Standard Error (RSE)**:

$$
\text{RSE} = \sqrt{\frac{\text{RSS}}{n-2}}, \quad \text{RSS = residual sum of squares}
$$

- Replace $\sigma$ with RSE in standard error formulas.

---

## 3️⃣ Confidence Intervals

- Standard errors allow **95% confidence intervals** for coefficients:

$$
\hat{\beta}_1 \pm 2 \cdot \text{SE}(\hat{\beta}_1), \quad
\hat{\beta}_0 \pm 2 \cdot \text{SE}(\hat{\beta}_0)
$$`

- Interpretation: Approx. 95% chance that interval contains true parameter.

### 🔹 Example (Advertising Data)

| Coefficient | 95% CI |
|-------------|--------|
| $\beta_0$ (intercept) | [6.130, 7.935] |
| $\beta_1$ (TV ads) | [0.042, 0.053] |

- Meaning:
  - Without TV ads, expected sales ≈ 6,130–7,940 units
  - Each \$1,000 increase in TV ads → sales increase by 42–53 units

---

## 4️⃣ Hypothesis Testing

### 🔹 Goal

Test if a predictor has **no effect** on response:

$$
H_0: \beta_1 = 0 \quad \text{(no relationship)} \\
H_a: \beta_1 \neq 0 \quad \text{(some relationship)}
$$

- If $\beta_1 = 0$, model reduces to $Y = \beta_0 + \varepsilon$, so $X$ does not explain $Y$.

---

### 🔹 Test Statistic: t-value

$$
t = \frac{\hat{\beta}_1 - 0}{\text{SE}(\hat{\beta}_1)}
$$

- Measures how many **standard errors** $\hat{\beta}_1$ is from 0
- If $H_0$ true → $t$ follows **t-distribution** with $n-2$ degrees of freedom
- Approximate **p-value**: probability of seeing $|t|$ or larger if $H_0$ true

---

### 🔹 Interpreting p-value

- Small p-value → unlikely that observed effect is due to chance → reject $H_0$
- Typical cutoffs: 0.05 (5%) or 0.01 (1%)

---

### 🔹 Advertising Example (Regression of Sales on TV ads)

| Coefficient | Estimate | Std. Error | t-statistic | p-value |
|-------------|---------|------------|-------------|---------|
| Intercept   | 7.0325  | 0.4578     | 15.36       | <0.0001 |
| TV          | 0.0475  | 0.0027     | 17.67       | <0.0001 |

- Interpretation:
  - Both coefficients highly significant
  - Very strong evidence that TV advertising is associated with sales
  - Null hypothesis $H_0: \beta_1 = 0$ is **rejected**

---

## 5️⃣ Summary: Standard Errors & Hypothesis Testing

1. **SE quantifies accuracy** of coefficient estimates
2. **Confidence intervals** provide a range likely to contain true parameters
3. **t-statistics and p-values** test whether predictors are associated with the response
4. **Larger spread in $X$** → smaller SE for slope → more precise estimate
5. **Advertising case:** TV ads significantly increase sales; estimate is precise

---

## ✅ Key Conceptual Flow

```text
Estimate coefficients (LS) → compute SE → construct CI → perform hypothesis test


* This forms the **core inference workflow** in linear regression.

```
```
