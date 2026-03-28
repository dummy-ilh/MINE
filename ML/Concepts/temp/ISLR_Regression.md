```md
# 📘 ISLR — Linear Regression Notes (Advertising Example)

## 1️⃣ Context
- Dataset: **Advertising** (TV, radio, newspaper budgets vs sales)
- Goal: Advise on marketing plan for next year to maximize sales
- Response variable: `Sales`
- Predictors: `TV`, `Radio`, `Newspaper` advertising budgets

---

## 2️⃣ Key Questions to Address Using Regression

### 1. Is there a relationship between advertising and sales?
- Test whether ad spend predicts sales
- Regression: test if coefficients $\beta_j \neq 0$

### 2. How strong is the relationship?
- Quantify predictive power
- Tools: $R^2$, residual standard error
- Strong relationship → can predict sales accurately

### 3. Which media contribute most?
- Separate effects of each medium using **multiple regression**
- Coefficients $\beta_j$ measure marginal effect controlling for other media

### 4. How accurately can we estimate effects?
- Measure uncertainty via standard errors and confidence intervals
- Example: effect of $1k on TV ads → increase in sales ± SE

### 5. How accurately can we predict future sales?
- Prediction intervals for new observations
- Includes model uncertainty + irreducible error

### 6. Is the relationship linear?
- Check if linear model is reasonable
- Diagnostics: residual plots, transformations, polynomial terms

### 7. Is there synergy among media? (interaction effect)
- Interaction terms: $\text{TV} \times \text{Radio}$, etc.
- Detect if combined spending yields extra effect

---

## 3️⃣ Why Linear Regression Fits

- Provides answers for all 7 questions
- Coefficients → inference about individual predictors
- Prediction → $\hat Y = \hat \beta_0 + \hat \beta_1 \text{TV} + \hat \beta_2 \text{Radio} + \hat \beta_3 \text{Newspaper}$
- Can include interactions → capture synergy
- Can handle transformations → capture non-linear effects

---

## 4️⃣ Summary Table

| Question | Regression Tool |
|----------|----------------|
| Relationship exists? | Hypothesis test ($\beta_j=0$) |
| Strength | $R^2$, residual error |
| Important media | $\hat \beta_j$ |
| Effect size | Coefficient estimate + CI |
| Predict future sales | Prediction interval |
| Linear? | Residual plots, transformations |
| Synergy | Interaction terms |

---

### ✅ Key Takeaway
> Linear regression is **not just fitting lines**; it is a **framework for answering multiple business and statistical questions** with quantified uncertainty.
```
````md
# 📘 ISLR — Linear Regression & Unbiasedness Notes

## 1️⃣ Analogy: Estimating a Mean

- Suppose we have a random variable $Y$ with mean $\mu$.
- Estimate $\mu$ using the **sample mean**:

$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^n y_i
$$

### 🔹 Key Concept: Unbiasedness

- $\hat{\mu}$ is **unbiased**:

$$
\mathbb{E}[\hat{\mu}] = \mu
$$

- Interpretation:
  - For one dataset, $\hat{\mu}$ might over- or under-estimate $\mu$
  - Across **many datasets**, the average $\hat{\mu}$ equals $\mu$
  - No systematic over- or under-estimation

---

## 2️⃣ Connection to Linear Regression

- Linear regression model:

$$
Y = \beta_0 + \beta_1 X + \varepsilon, \quad \mathbb{E}[\varepsilon] = 0
$$

- Least squares estimates:

$$
\hat{\beta}_1 = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2}, 
\quad 
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}
$$

### 🔹 Key Property: Unbiasedness

- $\hat{\beta}_0$ and $\hat{\beta}_1$ are **unbiased estimators**:

$$
\mathbb{E}[\hat{\beta}_0] = \beta_0, \quad \mathbb{E}[\hat{\beta}_1] = \beta_1
$$

- Meaning:
  - From one dataset, estimates may not exactly equal true $\beta$s
  - Across many datasets, the **average of the estimates equals the true parameters**
  - No systematic bias in the estimates

---

## 3️⃣ Intuition Using Multiple Datasets

- Imagine generating **many datasets** from the same population
- Fit a **least squares line** on each dataset
- If you average all these lines, you get the **true population regression line**

```text
Observed data → LS line → different per dataset
Average LS lines → true regression line
````

* Figure 3.3 (ISLR) demonstrates this visually:

  * Individual lines fluctuate around the true line
  * Average aligns closely with true regression line

---

## ✅ Key Takeaways

1. **Unbiasedness** ensures no systematic error in estimation
2. Least squares estimates behave like the **sample mean**, but in a multivariate context
3. Over multiple datasets, the **average LS estimate** recovers the true population regression
4. Helps build confidence in inference using regression coefficients

```
```
