Excellent — this passage is *rich* with fundamental ideas. Let’s unpack and reconstruct it into **comprehensive, professor-level notes** inspired by *Hosmer, Lemeshow, and Sturdivant (3rd ed.)*, Chapter 1 — rewritten in a clear, deeply explanatory style.

---

## 🔹 Introduction: Logistic Regression vs. Linear Regression

In any regression model, we seek to describe how an **outcome variable (Y)** depends on one or more **independent variables (x)**, also known as **covariates** or **predictors**.

In **linear regression**, the outcome is *continuous* — for example, height, income, or blood pressure.
In **logistic regression**, however, the outcome is **binary** (or *dichotomous*) — it takes on only two values such as:

* 0 = absence of a condition
* 1 = presence of a condition

In the example from the book, we study 100 individuals, each with:

* an identifier (ID)
* their **age (AGE)**
* whether they have **coronary heart disease (CHD)** or not (1 = present, 0 = absent)

This data set, referred to as **CHDAGE**, aims to explore the relationship between **AGE** and **CHD**.

---

## 🔹 Step 1: Understanding the Challenge of a Binary Outcome

If the outcome were continuous, our first step would be to **plot Y against X** (here, CHD vs. AGE) to visualize any trend or relationship.

However, when the outcome is **binary**, such a scatterplot becomes limited:

* All points fall on only two lines: y = 0 or y = 1.
* Although we might observe that older individuals tend to have CHD more frequently, the pattern is noisy.
* The variability in CHD at each age level is high — at any given age, some individuals have CHD (1), others don’t (0).

Thus, the scatterplot doesn’t provide a clear *functional form* for how CHD changes with age.

---

## 🔹 Step 2: Smoothing the Binary Outcome — Grouping by Age
Perfect — this table (Table 1.2 from *Applied Logistic Regression*, 3rd Edition) is central to understanding the **transition from raw binary data to the conceptual foundation of logistic regression**. Let’s integrate it seamlessly into our prior notes and interpret it in depth.

---

## 🔹 Step 2 (Expanded): Grouping AGE and Estimating the Conditional Mean — Table 1.2

When the outcome is **binary**, the raw scatterplot of CHD (0 = absent, 1 = present) versus AGE provides little visual clarity because all data points lie on two horizontal lines (y = 0 or y = 1).
To uncover any trend, Hosmer et al. propose summarizing the data by **age intervals**.

---

### 📊 Table 1.2 — Frequency Table of Age Group by CHD

| Age Group (years) | n       | CHD Absent | CHD Present | Mean ( = proportion with CHD ) |
| ----------------- | ------- | ---------- | ----------- | ------------------------------ |
| 20–29             | 10      | 9          | 1           | 0.100                          |
| 30–34             | 15      | 13         | 2           | 0.133                          |
| 35–39             | 12      | 9          | 3           | 0.250                          |
| 40–44             | 15      | 10         | 5           | 0.333                          |
| 45–49             | 13      | 7          | 6           | 0.462                          |
| 50–54             | 8       | 3          | 5           | 0.625                          |
| 55–59             | 17      | 4          | 13          | 0.765                          |
| 60–69             | 10      | 2          | 8           | 0.800                          |
| **Total**         | **100** | **57**     | **43**      | **0.430**                      |

---

### 🔹 Interpretation of the Table

Each age interval represents a **group** of individuals, and for each group we compute:

[
\text{Mean of Y} = \frac{\text{# with CHD (Y = 1)}}{n}
]

Because Y = 1 for “CHD present” and Y = 0 for “CHD absent,”
the mean is simply the **proportion of individuals with CHD** in that age group.

This mean serves as an empirical estimate of the **conditional expectation** of Y given x (AGE):

[
E(Y|x) \approx \text{Proportion with CHD in that age group}.
]

---

### 🔹 What the Table Shows

* There is a **clear increasing trend** in CHD prevalence with age:
  from **10%** in the 20–29 group up to **80%** in the 60–69 group.
* This pattern suggests that **age is positively associated** with the likelihood of CHD.
* The average across all subjects (0.43) reflects the **overall prevalence** of CHD in the sample.

If we plot the **mean CHD proportion (y-axis)** against the **midpoint of each age interval (x-axis)**, the resulting curve rises in an **S-shape**:
slow at first, then steeper in mid-ages, then leveling off at older ages — a hallmark of the **logistic relationship**.

---

### 🔹 Connecting to the Conditional Mean Concept

The **key regression quantity** is the *conditional mean* (E(Y|x)) — the expected value of Y given x.

In **linear regression**, we model this as:

[
E(Y|x) = \beta_0 + \beta_1 x
]

But, as discussed, this can take values below 0 or above 1 — invalid for probabilities.

In the **CHDAGE data**, the group means (0.10 → 0.80) stay within [0, 1], but the relationship between mean CHD and age is clearly **nonlinear**.
If we attempted to fit a straight line, it would fail to capture the curvature and might predict impossible probabilities (< 0 or > 1) for ages beyond the observed range.

---

### 🔹 Why This Table Matters

1. **Empirical motivation** — Table 1.2 provides real evidence that a simple linear model is inadequate.

2. **Transitional step** — These grouped means visually motivate the move to a **nonlinear probability model** (the logistic model).

3. **Bridge to π(x)** — Each mean value (0.10, 0.133, 0.25, …) is an estimate of:

   [
   \pi(x) = P(Y = 1 \mid x)
   ]

   which logistic regression models analytically as:

   [
   \pi(x) = \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}}.
   ]

4. **Shape validation** — When plotted, these eight mean points lie approximately on an **S-shaped curve**, confirming that the logistic function fits the data’s pattern much better than a straight line.

---

### 🔹 Visual Insight (conceptual sketch)

```
π(x)
 ↑
1|                                 ●
 |                              ●
 |                          ●
 |                      ●
 |                ●
 |          ●
 |     ●
 |  ●
0|____________________________________→ Age
```

* Each ● represents the group mean (from Table 1.2).
* The smooth curve passing through them would be the **fitted logistic curve**.
* The curvature visually demonstrates how the **probability of CHD increases with age** but cannot exceed 1.

---

### 🔹 Summary of the Role of Table 1.2

| Purpose                            | Explanation                                                          |                                                |
| ---------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------- |
| **Simplify the binary data**       | Grouping age removes individual-level noise and reveals structure.   |                                                |
| **Estimate the conditional mean**  | Group means approximate (E(Y                                         | x)), i.e., the probability of CHD at that age. |
| **Reveal nonlinearity**            | The pattern is S-shaped, not linear.                                 |                                                |
| **Motivate the logistic function** | Shows why a model bounded between 0 and 1 is needed.                 |                                                |
| **Bridge conceptually to π(x)**    | Helps transition from raw frequencies to a formal probability model. |                                                |

---

### 🔹 Final Conceptual Takeaway

The **CHDAGE example and Table 1.2** provide the empirical groundwork for logistic regression.
They show that:

* The relationship between age and CHD is *monotonic* (increasing) but *nonlinear*.
* A model that constrains probabilities within [0, 1] and captures an S-shaped pattern — the **logistic model** — is the natural solution.

---

Would you like me next to **illustrate how we estimate the coefficients (β₀, β₁)** from such data — showing how the logistic curve is fitted using **maximum likelihood** — or to focus first on **interpreting β₁ as an odds ratio** (the next conceptual step in the chapter)?

---

## 🔹 Step 3: Why Linear Regression Fails for Binary Outcomes

The expression (E(Y|x) = \beta_0 + \beta_1x) can take *any real value*, from (-\infty) to (+\infty).
However, for a binary outcome, the conditional mean (E(Y|x)) must be between **0 and 1**, because it represents a **probability**.

Therefore:

* A linear model can predict invalid probabilities (<0 or >1).
* The relationship between Y and x is rarely linear in such cases — it tends to be **S-shaped** (sigmoidal).

  * At low x-values, probabilities are near 0.
  * As x increases, the probability gradually rises.
  * It approaches 1 at large x-values.

This S-shape reflects the fact that **the change in E(Y|x)** per unit increase in x becomes **smaller** as we near 0 or 1.

Hence, we need a model that:

1. Keeps (E(Y|x)) within [0, 1].
2. Captures this nonlinear “S” behavior naturally.

---

## 🔹 Step 4: The Logistic Regression Model

The **logistic regression model** meets both requirements by expressing (E(Y|x)) — denoted (\pi(x)) — as:

[
\pi(x) = E(Y|x) = \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}}
]
*(Equation 1.1)*

This formulation ensures:

* (0 < \pi(x) < 1) for all values of x.
* The relationship between x and π(x) is S-shaped.

### Interpretation:

* (\pi(x)) represents the **probability** that Y = 1 given x.
* As x increases:

  * if (\beta_1 > 0), the probability of Y = 1 rises;
  * if (\beta_1 < 0), the probability falls.

---

## 🔹 Step 5: The Logit Transformation

To connect this nonlinear probability curve to a **linear model**, we apply the **logit transformation**, defined as:

[
g(x) = \ln\left(\frac{\pi(x)}{1 - \pi(x)}\right)
]

The term inside the log, (\frac{\pi(x)}{1 - \pi(x)}), is called the **odds** of the event (Y = 1) occurring.

Substituting Equation (1.1) into this expression, we find:

[
g(x) = \beta_0 + \beta_1 x
]

Thus, the **logit of the probability** is a *linear function* of x.

### Key properties:

* (g(x)) can take any real value (−∞ to +∞), just like the linear predictor in linear regression.
* The model is linear in its parameters ((\beta_0, \beta_1)).
* This linearity makes interpretation, estimation, and statistical inference much simpler.

---

## 🔹 Step 6: The Nature of the Error Term

In **linear regression**, the outcome is modeled as:

[
y = E(Y|x) + \varepsilon
]

where:

* (\varepsilon) is a normally distributed error term ((\varepsilon \sim N(0, \sigma^2))).
* Hence, Y|x follows a **normal distribution**.

But in **logistic regression**, Y can only be 0 or 1.
So we express it as:

[
y = \pi(x) + \varepsilon
]

where:

* If (y = 1), then (\varepsilon = 1 - \pi(x))
* If (y = 0), then (\varepsilon = 0 - \pi(x) = -\pi(x))

These two possible values show that ε does **not** follow a normal distribution.

Instead, the conditional distribution of Y|x is **binomial**:

* Y takes 1 with probability (\pi(x))
* Y takes 0 with probability (1 - \pi(x))

Hence, logistic regression is built on the **binomial distribution**, not the normal distribution.

---

## 🔹 Step 7: Key Distinctions Between Linear and Logistic Regression

| Concept                  | Linear Regression           | Logistic Regression                      |                                                                         |
| ------------------------ | --------------------------- | ---------------------------------------- | ----------------------------------------------------------------------- |
| **Outcome variable (Y)** | Continuous                  | Binary (0/1)                             |                                                                         |
| **Conditional mean**     | (E(Y                        | x) = \beta_0 + \beta_1 x)                | (\pi(x) = \dfrac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}}) |
| **Range of E(Y           | x)**                        | (-\infty) to (+\infty)                   | 0 to 1                                                                  |
| **Error distribution**   | Normal ((ε \sim N(0, σ^2))) | Binomial                                 |                                                                         |
| **Model scale**          | Linear in Y                 | Linear in log-odds ((g(x)))              |                                                                         |
| **Transformation**       | None                        | Logit: (g(x) = \ln(\pi(x)/(1 - \pi(x)))) |                                                                         |

---

## 🔹 Step 8: Conceptual Bridge Between the Two Models

Despite these differences, the **underlying principles** are similar:

* Both model the **expected value** of Y given x.
* Both assume a linear relationship — but in logistic regression, that linearity holds for the **logit** (log-odds), not for the probability itself.
* Both use maximum likelihood or least squares to estimate parameters that best fit the data.
* Both allow inference (testing, confidence intervals) on the relationship between predictors and the outcome.

Thus, logistic regression **extends the ideas of linear regression** to accommodate a binary outcome in a statistically coherent way.

---

## 🔹 Summary Takeaways

1. **Covariates** (independent variables) explain variation in a binary outcome.
2. Logistic regression models the **probability** that the event occurs, ensuring predictions stay between 0 and 1.
3. The **logit transformation** linearizes the relationship, making the model analytically tractable.
4. The **error distribution** is **binomial**, reflecting the discrete nature of the outcome.
5. Conceptually, logistic regression maintains the same structure and logic as linear regression — modeling a conditional mean — but it does so on the **log-odds scale**.

---

Excellent — you’ve reached one of the *most important conceptual pivots* in Chapter 1 of *Applied Logistic Regression* (Hosmer, Lemeshow, & Sturdivant, 3rd ed.): **how parameters are estimated in logistic regression**, and *why* we use **maximum likelihood (ML)** instead of **least squares (LS)**.

Let’s unpack this with deep, structured notes — preserving the spirit and reasoning of the book, while clarifying every detail.

---

## 🔹 1. Background: Parameter Estimation in Regression Models

In any regression model, our goal is to estimate unknown parameters (e.g., β₀, β₁) that describe how the outcome (Y) depends on predictor(s) (x).

For linear regression, we write:

[
E(Y|x) = \beta_0 + \beta_1 x
]

We want β₀ and β₁ such that the fitted line gives predicted values of (Y) that are as *close as possible* to the observed data points.

---

## 🔹 2. Least Squares Estimation in Linear Regression

The **method of least squares** finds parameter estimates that **minimize the sum of squared deviations** between observed and predicted outcomes:

[
\text{Minimize } ; S(\beta_0, \beta_1) = \sum_{i=1}^{n} [y_i - (\beta_0 + \beta_1 x_i)]^2
]

This is mathematically simple and statistically elegant **because of the normality assumption** for the error term ( \varepsilon_i ):

[
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad \varepsilon_i \sim N(0, \sigma^2)
]

Under these assumptions:

* The least squares estimates (LSEs) are *unbiased*, *efficient*, and *consistent*.
* They coincide with the **maximum likelihood estimates (MLEs)** under the normal model.

So in linear regression, least squares and maximum likelihood give the **same** results.

---

## 🔹 3. Why Least Squares Fails for a Dichotomous Outcome

In **logistic regression**, the outcome (Y_i) is **binary** — it takes values 0 or 1, and has no normal error structure.

The model is:

[
\pi_i = P(Y_i = 1 | x_i) = \frac{e^{\beta_0 + \beta_1 x_i}}{1 + e^{\beta_0 + \beta_1 x_i}}
]

and

[
Y_i \sim \text{Bernoulli}(\pi_i)
]

If we tried to apply least squares here — minimizing (\sum (y_i - \pi_i)^2) — we would run into problems:

* The variance of (Y_i) is not constant; it depends on (\pi_i) as (\pi_i(1 - \pi_i)).
* The relationship between (Y) and (x) is nonlinear.
* The error term cannot be assumed normally distributed — it takes only two discrete values.
* The least squares estimates can fall outside the valid probability range (0, 1).
* Statistical properties like unbiasedness and efficiency no longer hold.

Hence, **least squares estimation is inappropriate** for models with a binary outcome.

---

## 🔹 4. Maximum Likelihood: The General Estimation Principle

The **maximum likelihood (ML)** method is a general framework that works under many probability models — including the logistic model.

### 🔸 Conceptual Idea

ML estimation chooses parameter values that make the **observed data most probable** under the assumed model.

* Think of the observed dataset as a fixed outcome of a random process.
* For different candidate parameter values (β₀, β₁), the model assigns different probabilities (likelihoods) to seeing those data.
* The MLEs are the parameter values that **maximize** that probability.

Formally, if (L(\beta_0, \beta_1)) denotes the **likelihood function**, then the MLEs are:

[
(\hat{\beta}_0, \hat{\beta}*1) = \arg \max*{\beta_0, \beta_1} L(\beta_0, \beta_1)
]

---

## 🔹 5. Constructing the Likelihood Function for Logistic Regression

Because each observation (Y_i) follows a **Bernoulli distribution** with probability (\pi_i):

[
P(Y_i = y_i) = \pi_i^{y_i} (1 - \pi_i)^{1 - y_i}
]

and all observations are assumed independent, the joint probability (likelihood) for the full dataset is:

[
L(\beta_0, \beta_1)
= \prod_{i=1}^{n} \pi_i^{y_i} (1 - \pi_i)^{1 - y_i}
= \prod_{i=1}^{n} \left[ \frac{e^{\beta_0 + \beta_1 x_i}}{1 + e^{\beta_0 + \beta_1 x_i}} \right]^{y_i}
\left[ \frac{1}{1 + e^{\beta_0 + \beta_1 x_i}} \right]^{1 - y_i}
]

This is the **likelihood function**, expressing the probability of observing the data as a function of the unknown parameters β₀ and β₁.

---

## 🔹 6. The Log-Likelihood Function

Because products are hard to maximize directly, we take the natural logarithm to simplify (since log is a monotonic function).
The **log-likelihood** is:

[
\ell(\beta_0, \beta_1)
= \sum_{i=1}^{n} \left[
y_i \ln(\pi_i) + (1 - y_i)\ln(1 - \pi_i)
\right]
]

Substituting (\pi_i = \frac{e^{\beta_0 + \beta_1 x_i}}{1 + e^{\beta_0 + \beta_1 x_i}}), we get:

[
\ell(\beta_0, \beta_1)
= \sum_{i=1}^{n} \left[
y_i (\beta_0 + \beta_1 x_i) - \ln(1 + e^{\beta_0 + \beta_1 x_i})
\right]
]

This function is smooth, concave, and differentiable — ideal for numerical optimization.

---

## 🔹 7. Finding the MLEs

To find the estimates, we set the **score equations** (derivatives of the log-likelihood) to zero:

[
\frac{\partial \ell}{\partial \beta_j} = 0, \quad j = 0, 1
]

Because these equations are nonlinear in β₀ and β₁, **no closed-form solution exists**.
Instead, we solve them **iteratively** using numerical algorithms such as:

* **Newton–Raphson method**
* **Fisher scoring**
* or software routines implemented in packages like R, SAS, or Stata.

---

## 🔹 8. Why Maximum Likelihood Is Preferred

ML estimation has strong theoretical properties, especially as the sample size increases:

| Property                 | Description                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------- |
| **Consistency**          | Estimates converge to the true parameter values as (n \to \infty).                 |
| **Asymptotic Normality** | MLEs are approximately normally distributed for large samples.                     |
| **Efficiency**           | Among consistent estimators, MLEs have minimum possible variance (asymptotically). |
| **Invariance**           | Functions of MLEs (e.g., odds ratios = (e^{\beta_1})) are also MLEs.               |

These properties make MLE the standard approach for fitting logistic regression models.

---

## 🔹 9. Connecting Back to Linear Regression

Hosmer emphasizes that **maximum likelihood is the general method** from which least squares can be seen as a *special case*:

* In the linear regression model with normally distributed errors, maximizing the likelihood (of a normal distribution) leads *exactly* to the least squares criterion.
* In logistic regression, the outcome distribution is *Bernoulli (binomial)*, so ML gives a **different likelihood function** — but the logic is the same.

Hence, **maximum likelihood is the unifying estimation principle** across regression models.

---

## 🔹 10. Conceptual Summary

| Aspect             | Linear Regression                                | Logistic Regression                        |
| ------------------ | ------------------------------------------------ | ------------------------------------------ |
| Outcome            | Continuous                                       | Binary (0/1)                               |
| Error Distribution | Normal                                           | Bernoulli (Binomial)                       |
| Estimation Method  | Least Squares (equivalent to ML under normality) | Maximum Likelihood                         |
| Objective Function | Minimize sum of squared residuals                | Maximize probability of observed outcomes  |
| Solution Form      | Closed form                                      | Numerical (iterative)                      |
| Key Advantage      | Simplicity                                       | Generality and correctness for binary data |

---

### 🔸 Intuitive Analogy

Think of maximum likelihood as asking:

> “Given our model’s assumptions, what parameter values make the *observed pattern of 0s and 1s* most plausible?”

Least squares, by contrast, asks:

> “What line minimizes the average squared distance between observed and predicted Y values?”

For binary data, “distance” doesn’t make sense — probabilities do.
Therefore, **maximum likelihood** is the natural and statistically sound choice.

---

## 🔹 11. Final Takeaway

* Least squares works beautifully when (Y) is continuous and normal.
* For a binary outcome, the assumptions underlying least squares collapse.
* Maximum likelihood provides a principled alternative that:

  1. Matches the true (binomial) distribution of the data,
  2. Ensures predicted probabilities remain in [0, 1], and
  3. Produces estimators with strong statistical properties.

Thus, all logistic regression estimation throughout the book — and in modern practice — is based on **maximum likelihood**.

---
Nice — you’ve got the fitted logistic model from the software. I’ll (1) restate the model, (2) explain what the coefficients mean, (3) show worked numeric examples (step-by-step arithmetic) for several ages, (4) compute the implied odds ratio per year and per 10 years, and (5) compare fitted probabilities at the age-group midpoints to the observed group means in Table 1.2.

# Fitted model (from your output)

Estimated coefficients:

* (\hat\beta_0 = -5.309)
* (\hat\beta_1 = 0.111)

Fitted probability (predicted probability that CHD = 1 for a person of age AGE):
[
\hat\pi(\text{AGE}) = \frac{e^{-5.309 + 0.111\times\text{AGE}}}{1 + e^{-5.309 + 0.111\times\text{AGE}}}
]

Fitted logit (linear predictor):
[
\widehat{g}(\text{AGE}) = \ln!\frac{\hat\pi}{1-\hat\pi} = -5.309 + 0.111\times\text{AGE}.
]

---

# Interpretation of coefficients (conceptual)

* (\hat\beta_1 = 0.111): each additional **year of age** increases the **log-odds** of CHD by 0.111.
* On the **odds scale**, a 1-year increase multiplies the odds by (e^{0.111}\approx 1.1174) — i.e., about an **11.7% increase in the odds** of CHD per year of age (see numeric below).
* (\hat\beta_0 = -5.309) is the intercept: the log-odds of CHD when AGE = 0. It has no direct practical meaning here (AGE = 0 is out of range), but it is needed to position the curve.

---

# Worked numeric examples (digit-by-digit style)

We compute (\eta = -5.309 + 0.111\times\text{AGE}), then (\hat\pi = e^\eta/(1+e^\eta)).

Examples for AGE = 20, 30, 40, 50, 60:

1. AGE = 20

   * (0.111\times 20 = 2.220)
   * (\eta = -5.309 + 2.220 = -3.089)
   * (\mathrm{odds}=e^{-3.089}\approx 0.045547)
   * (\hat\pi = \dfrac{0.045547}{1+0.045547} \approx 0.043563) → **4.36%**

2. AGE = 30

   * (0.111\times30 = 3.330)
   * (\eta = -5.309 + 3.330 = -1.979)
   * (\mathrm{odds}=e^{-1.979}\approx 0.138207)
   * (\hat\pi \approx 0.121425) → **12.14%**

3. AGE = 40

   * (0.111\times40 = 4.440)
   * (\eta = -5.309 + 4.440 = -0.869) (rounded)
   * (\mathrm{odds}\approx 0.419371)
   * (\hat\pi \approx 0.295462) → **29.55%**

4. AGE = 50

   * (0.111\times50 = 5.550)
   * (\eta = -5.309 + 5.550 = 0.241)
   * (\mathrm{odds}\approx 1.272521)
   * (\hat\pi \approx 0.559960) → **56.00%**

5. AGE = 60

   * (0.111\times60 = 6.660)
   * (\eta = -5.309 + 6.660 = 1.351)
   * (\mathrm{odds}\approx 3.861285)
   * (\hat\pi \approx 0.794293) → **79.43%**

(rounded values shown; calculations done exactly as shown)

---

# Odds ratio interpretation

* **Per 1 year**: ( \text{OR}_{1\text{yr}} = e^{0.111} \approx 1.1174).
  → The odds of CHD increase by **~11.7% per additional year of age**.
* **Per 10 years**: ( \text{OR}_{10\text{yr}} = e^{0.111\times 10} = e^{1.11} \approx 3.0344).
  → The odds of CHD are about **3.03 times** as large for someone 10 years older (i.e., ~203% higher odds).

*(If you want confidence intervals for these ORs, I’ll need the standard error(s) from Table 1.3 — otherwise I can’t compute CIs.)*

---

# Comparison: fitted probabilities at age-group midpoints vs. observed group means (Table 1.2)

I computed fitted (\hat\pi) at the midpoints of the age intervals used in Table 1.2. Midpoints and results:

| Age group | Midpoint | Observed mean (Table 1.2) | Fitted (\hat\pi) (model) |
| --------- | -------: | ------------------------: | -----------------------: |
| 20–29     |     24.5 |                     0.100 |                   0.0698 |
| 30–34     |       32 |                     0.133 |                   0.1472 |
| 35–39     |       37 |                     0.250 |                   0.2311 |
| 40–44     |       42 |                     0.333 |                   0.3437 |
| 45–49     |       47 |                     0.462 |                   0.4770 |
| 50–54     |       52 |                     0.625 |                   0.6137 |
| 55–59     |       57 |                     0.765 |                   0.7346 |
| 60–69     |     64.5 |                     0.800 |                   0.8642 |

* The fitted probabilities follow the same increasing pattern and are quite close to the observed proportions in most groups (small deviations are expected).
* This demonstrates the model’s S-shaped probability curve fitting the grouped empirical proportions reasonably well.

---

Great! Let’s break this down and turn it into a **comprehensive, clear set of notes** on **testing the significance of coefficients in logistic regression**, based on what Hosmer describes in Section 1.3. I’ll make it standalone, thorough, and fully explained.

---

# **Testing for the Significance of Coefficients in Logistic Regression**

The main question in testing the significance of a coefficient is:

> *Does adding this independent variable to the model provide useful information for predicting the response variable?*

In other words, we compare two models:

1. **Reduced model**: excludes the variable in question.
2. **Full model**: includes the variable in question.

We evaluate whether the full model predicts the response significantly better than the reduced model.

---

## **1. Analogy to Linear Regression**

In **linear regression**, the standard method is to use an **Analysis of Variance (ANOVA) table**, which partitions variability:

* **Total Sum of Squares (SST)**: variability of the observed outcomes around their mean.
* **Regression Sum of Squares (SSR)**: variability explained by the model.
* **Residual Sum of Squares (SSE)**: variability unexplained (differences between observed and predicted).

Mathematically, for (n) observations:

[
SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
]

* If the independent variable is **not included**, the predicted value is the mean: (\hat{y}_i = \bar{y}). Then:

[
SSE_{\text{reduced}} = \sum_{i=1}^{n} (y_i - \bar{y})^2 = SST
]

* If the independent variable **is included**, the predicted values come from the regression line. Any **reduction in SSE** is due to the variable being meaningful.

[
SSR = SST - SSE
]

**Interpretation:**

* Large SSR → independent variable explains a lot → likely significant.
* Small SSR → independent variable contributes little → likely not significant.

---

## **2. Logistic Regression Approach**

The **conceptual principle** is the same: compare observed to predicted outcomes.

* **Key difference**: logistic regression uses **likelihood**, not squared differences.

### **2.1 Likelihood Function**

For logistic regression, the likelihood for (n) independent observations is:

[
L(\beta) = \prod_{i=1}^{n} \hat{\pi}_i^{y_i} (1 - \hat{\pi}_i)^{1 - y_i}
]

where:

* (y_i \in {0,1}) is the observed outcome.
* (\hat{\pi}_i = P(Y_i = 1 \mid X_i)) is the predicted probability from the model.

The **log-likelihood** is:

[
\ell(\beta) = \sum_{i=1}^{n} \big[ y_i \ln(\hat{\pi}_i) + (1 - y_i) \ln(1 - \hat{\pi}_i) \big]
]

---

### **2.2 Deviance**

To measure model fit, we define the **deviance**:

[
D = -2 \ln \left( \frac{L(\text{fitted model})}{L(\text{saturated model})} \right)
]

* **Saturated model**: a hypothetical model with as many parameters as observations. It perfectly predicts every outcome, so its likelihood = 1.0.
* In practice, this means:

[
D = -2 \ln \big( L(\text{fitted model}) \big)
]

**Key point:** The deviance (D) in logistic regression plays the **same role as SSE in linear regression**: smaller deviance = better fit.

---

### **2.3 Likelihood Ratio Test**

To test whether a variable significantly improves the model, compute:

[
G = D_{\text{reduced}} - D_{\text{full}}
]

* (D_{\text{reduced}}) = deviance of the model without the variable.
* (D_{\text{full}}) = deviance of the model with the variable.

**Interpretation:**

* Large (G) → adding the variable reduces deviance a lot → variable is significant.
* Small (G) → adding the variable doesn’t improve fit → variable likely not significant.

**Distribution for hypothesis testing:**

* Under the null hypothesis (H_0: \beta_j = 0), (G) approximately follows a **chi-square distribution** with degrees of freedom equal to the number of added parameters (usually 1 for a single variable).

[
G \sim \chi^2_{\text{df}}
]

---

## **3. Step-by-Step Summary**

1. **Fit the reduced model** (exclude the variable). Compute deviance (D_{\text{reduced}}).
2. **Fit the full model** (include the variable). Compute deviance (D_{\text{full}}).
3. Compute (G = D_{\text{reduced}} - D_{\text{full}}).
4. Compare (G) to the chi-square distribution with appropriate degrees of freedom.
5. Conclude: significant if (G) exceeds the critical chi-square value.

---

## **4. Example (Conceptual)**

Suppose we have a binary outcome (disease: yes/no) and a predictor (X) (risk factor):

1. **Reduced model:** just intercept. Deviance: (D_{\text{reduced}} = 120)
2. **Full model:** intercept + (X). Deviance: (D_{\text{full}} = 105)
3. Likelihood ratio statistic: (G = 120 - 105 = 15)
4. (G \sim \chi^2_1) → check chi-square table, critical value at 0.05 ≈ 3.841
5. Conclusion: (G = 15 > 3.841) → (X) is significant.

---

## **5. Key Takeaways**

* Logistic regression uses **likelihood** instead of squared errors to compare models.
* **Deviance** is the logistic regression analog of SSE.
* **Likelihood ratio test** compares models with and without a variable to assess significance.
* Large reduction in deviance → variable is significant.
* The test statistic (G) follows a chi-square distribution under (H_0).

---

## **6. Common Pitfalls**

* Forgetting the **degrees of freedom** for the chi-square test (1 per variable).
* Confusing deviance with log-likelihood itself; remember (D = -2 \ln(L_{\text{model}}/L_{\text{saturated}})).
* Assuming a large deviance automatically means poor fit; it’s **relative deviance reduction** that matters for significance testing.

---
Excellent! Let’s expand this section from **Hosmer (Sections 1.3–1.4)** into a **detailed, thorough set of notes**, covering the **likelihood ratio test, Wald test, score test, and Wald-based confidence intervals**, rewritten clearly and structured for deep understanding.

---

# **Testing Significance and Confidence Intervals in Logistic Regression**

In logistic regression, we assess whether a variable significantly improves the prediction of a binary outcome. Several statistical tests and confidence interval methods are used. We’ll go step by step.

---

## **1. Likelihood Ratio Test (LRT)**

The **likelihood ratio test** compares two nested models:

1. **Reduced model**: excludes the variable of interest.
2. **Full model**: includes the variable of interest.

The test statistic is:

[
G = -2 \ln \left( \frac{L_{\text{reduced}}}{L_{\text{full}}} \right)
]

where (L_{\text{reduced}}) and (L_{\text{full}}) are the likelihoods of the reduced and full models, respectively.

* For a **single binary variable**, the reduced model has only an intercept:

[
\hat{\beta}_0 = \ln\frac{n_1}{n_0}
]

where:

* (n_1 = \sum y_i) = number of “successes” (y = 1)
* (n_0 = \sum (1 - y_i)) = number of “failures” (y = 0)
* Predicted probability for all subjects: (\hat{\pi} = \frac{n_1}{n})

The **likelihood ratio statistic (G)** can then be computed as:

[
G = 2 \sum_{i=1}^n \Big[ y_i \ln \hat{\pi}_i + (1 - y_i) \ln(1 - \hat{\pi}_i) \Big] - 2 \big[n_1 \ln(n_1/n) + n_0 \ln(n_0/n)\big]
]

* First term → log-likelihood from the **full model**
* Second term → log-likelihood from the **reduced model**

**Interpretation:**

* Large (G) → variable significantly improves model fit.
* (G) is approximately (\chi^2) distributed with df = number of parameters added.

---

## **2. Wald Test**

The **Wald test** uses the estimated coefficient and its standard error:

[
W = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}
]

* For large samples, under (H_0: \beta_j = 0):

[
W \sim N(0,1)
]

* Example: coefficient for AGE:

[
\hat{\beta}_1 = 0.111, \quad \text{SE}(\hat{\beta}_1) = 0.024
]

[
W = \frac{0.111}{0.024} \approx 4.61
]

* Two-tailed p-value: (P(|z| > 4.61) < 0.001) → highly significant.
* Software sometimes reports (W^2 = z^2), which follows a (\chi^2_1) distribution.

**Notes:**

* Wald test relies on the normal approximation of (\hat{\beta}_j).
* Less robust in small samples or with extreme probabilities.

---

## **3. Score Test (Lagrange Multiplier Test)**

The **score test** evaluates significance **without fitting the full model**:

* Uses derivatives of the log-likelihood (the “score function”).
* In the **univariate case**, the statistic is:

[
ST = \frac{\sum_{i=1}^{n} x_i (y_i - \bar{y})}{\sqrt{\bar{y}(1-\bar{y}) \sum (x_i - \bar{x})^2}}
]

where:

* (\bar{y} = n_1 / n)

* (\bar{x} = \text{mean of predictor } x_i)

* Advantages: requires **less computation**, because you only evaluate the score at the null model ((\beta_1 = 0)).

* Limitations: less widely available in software; more complex for multivariate models.

---

## **4. Wald-Based Confidence Intervals**

Confidence intervals for logistic regression parameters are often derived from the **Wald test**:

### **4.1 For coefficients**

[
\text{CI for } \beta_j: \quad \hat{\beta}*j \pm z*{1-\alpha/2} , \text{SE}(\hat{\beta}_j)
]

* (z_{1-\alpha/2}) → critical value from standard normal distribution (e.g., 1.96 for 95% CI).

### **4.2 For logit (linear predictor)**

[
g(x) = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \dots
]

[
\text{CI for } g(x): \quad \hat{g}(x) \pm z_{1-\alpha/2} , \text{SE}[\hat{g}(x)]
]

### **4.3 For predicted probability**

Transform back to probability scale using logistic function:

[
\pi(x) = \frac{e^{\hat{g}(x)}}{1 + e^{\hat{g}(x)}}
]

* Lower limit:

[
\pi_{\text{lower}} = \frac{e^{\hat{g}(x) - z_{1-\alpha/2} SE[\hat{g}(x)]}}{1 + e^{\hat{g}(x) - z_{1-\alpha/2} SE[\hat{g}(x)]}}
]

* Upper limit:

[
\pi_{\text{upper}} = \frac{e^{\hat{g}(x) + z_{1-\alpha/2} SE[\hat{g}(x)]}}{1 + e^{\hat{g}(x) + z_{1-\alpha/2} SE[\hat{g}(x)]}}
]

**Example:**

* Age = 50, (\hat{g}(50) = 0.14), (SE[\hat{g}(50)] = 0.2)
* CI for probability: 0.435 – 0.677 → interprets as “mean probability of CHD at age 50 with 95% confidence.”

---

## **5. Practical Notes and Recommendations**

1. **Likelihood ratio test** is generally preferred over Wald test for significance because it is more robust, especially in small samples or extreme probabilities.
2. **Score test** is useful when computational simplicity is desired, though less commonly implemented.
3. **Wald-based confidence intervals** assume normality of (\hat{\beta}_j) — may be inaccurate for small samples or sparse data.
4. **Fitted probabilities** can be used for prediction or classification, linking logistic regression to discriminant analysis (discussed in Chapter 4).

---

## **6. Summary Table of Tests**

| Test             | Computation               | Key Statistic                              | Distribution                 | Notes                         |
| ---------------- | ------------------------- | ------------------------------------------ | ---------------------------- | ----------------------------- |
| Likelihood Ratio | Fit full & reduced models | (G = D_{\text{reduced}} - D_{\text{full}}) | (\chi^2_{df})                | Robust, preferred             |
| Wald             | Only full model           | (W = \hat{\beta}/SE(\hat{\beta}))          | Normal (or (\chi^2_1 = W^2)) | Sensitive to small samples    |
| Score            | Only reduced model        | Based on derivatives at null               | (\chi^2_{df})                | Less computation, rarely used |

---

Perfect! Let’s create a **conceptual diagram** that shows how the three main tests in logistic regression—**Likelihood Ratio Test (LRT), Wald Test, and Score Test**—relate to one another and to the deviance/log-likelihood. I’ll describe it in a clear, annotated ASCII-style flow for clarity.

---

# **Conceptual Diagram: Logistic Regression Significance Tests**

```
                ┌─────────────────────────────┐
                │    Logistic Regression      │
                │  (Binary outcome Y, n obs) │
                └─────────────┬──────────────┘
                              │
             ┌────────────────┴─────────────────┐
             │                                  │
     Reduced Model (Null)                 Full Model
   (Excludes variable of interest)      (Includes variable)
             │                                  │
             │                                  │
     Compute Log-Likelihood            Compute Log-Likelihood
       ℓ_reduced = log L(reduced)       ℓ_full = log L(full)
             │                                  │
             └──────────────┬──────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │ Compare models for significance│
            └───────────────┬───────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
      Likelihood Ratio      Wald Test       Score Test
          (LRT)               │                 │
G = -2 ln(L_reduced / L_full)  W = β̂ / SE(β̂)  ST = score^2 / var(score)
          │                 │                 │
χ² distribution            Normal or χ²        χ² distribution
(df = # of added params)   (df = 1)           (df = # of params)
          │                 │                 │
Interpret significance   Interpret significance  Interpret significance
by comparing G           by comparing W to      using score statistic
to χ² critical value     z (or W² to χ²)       (reduced model only)
```

---

### **Key Notes from Diagram**

1. **Likelihood Ratio Test (LRT):**

   * Compares full vs reduced models using **deviance/log-likelihood**.
   * Requires fitting **both models**.

2. **Wald Test:**

   * Uses the estimated coefficient and standard error from the **full model only**.
   * Easier to compute, but can be unstable in small samples or extreme probabilities.

3. **Score Test:**

   * Uses derivatives (score function) evaluated **at the null model**.
   * Does not require fitting the full model.

4. **Deviance Connection:**

   * LRT directly compares deviance (D = -2 \log L).
   * Wald and Score tests are approximations based on asymptotic theory.

---

If you like, I can also make a **second diagram showing the link between deviance, SSE (from linear regression), and reduction in deviance for a single predictor**, which makes the **analogy to linear regression** visually intuitive.

Do you want me to make that second diagram as well?

