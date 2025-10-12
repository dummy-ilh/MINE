

## 🔹 Introduction: Logistic Regression vs. Linear Regression

Sure — here’s your passage rewritten neatly and clearly in **Markdown format**, preserving structure, emphasis, and readability for notes or documentation.

---

# **Introduction to Logistic Regression**

The goal of modeling is to **find the best-fitting and most parsimonious, clinically interpretable model** to describe the relationship between an **outcome** (dependent or response) variable and a set of **independent** (predictor or explanatory) variables.
The independent variables are often called **covariates**.

The most common example of modeling — and one assumed to be familiar to readers — is the **usual linear regression model**, where the **outcome variable is continuous**.

---

## **Difference Between Linear and Logistic Regression**

What distinguishes a **logistic regression model** from the **linear regression model** is that the **outcome variable in logistic regression is binary or dichotomous**.

This difference between logistic and linear regression is reflected both in the **form of the model** and its **underlying assumptions**.
Once this difference is accounted for, the methods employed in a logistic regression analysis follow, more or less, the same general principles used in linear regression.

---

## **Example 1: The CHDAGE Data**

**Table 1.1** lists:

* The **age in years (AGE)**
* The **presence or absence** of evidence of significant **coronary heart disease (CHD)**
* An **identifier variable (ID)**
* An **age group variable (AGEGRP)**

for **100 subjects** in a hypothetical study of risk factors for heart disease.

The **outcome variable** is **CHD**, which is coded as:

* `0` → CHD absent
* `1` → CHD present

In general, any two values could be used, but it is most convenient to use **0 and 1**.

We refer to this data set as the **CHDAGE** data.

---

## **Exploring the Relationship Between AGE and CHD**

It is of interest to explore the relationship between **AGE** and the **presence or absence of CHD** in this group.

If our outcome variable were **continuous** rather than binary, we would begin by forming a **scatterplot** of the outcome versus the independent variable.
This scatterplot would help visualize the **nature and strength** of any relationship between the outcome and the independent variable.

A scatterplot of the data in **Table 1.1** shows some tendency for individuals **without CHD** to be **younger** than those **with CHD**.
While this plot depicts the **binary nature** of the outcome variable clearly, it does **not provide a clear picture** of the functional relationship between **CHD and AGE**.

---

## **Reducing Variability Using Age Groups**

The main problem with such a scatterplot is that the **variability in CHD** at all ages is large, making it difficult to visualize any relationship.

One common method to reduce some of this variation — while maintaining the relationship’s structure — is to **create intervals for the independent variable** and compute the **mean of the outcome variable within each group**.

We use this strategy by grouping **age** into categories (**AGEGRP**) defined in **Table 1.1**.
**Table 1.2** contains, for each age group:

* The **frequency** of each outcome
* The **percent** of individuals with CHD present

By examining this table, a clearer picture of the relationship emerges:

> As **age increases**, the **proportion (mean)** of individuals with evidence of **CHD** also **increases**.

**Figure 1.2** presents a plot of the **percent of individuals with CHD** versus the **midpoint** of each age interval.

---

## **Modeling the Relationship**

Let:

* ( Y ) denote the **outcome variable**, and
* ( x ) denote a specific value of the **independent variable**.

Then the expected value of ( Y ) given ( x ) is written as:

[
E(Y | x)
]

and is read as “the expected value of ( Y ), given the value ( x ).”

In **linear regression**, we assume that this **mean** can be expressed as an **equation** relating ( Y ) to ( x ):

[
E(Y | x) = \beta_0 + \beta_1 x
]




---

## 🔹  Smoothing the Binary Outcome — Grouping by Age


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

\[
\text{Mean of } Y = \frac{\text{# with CHD (Y = 1)}}{n}
\]


Because \(Y = 1\) for “CHD present” and \(Y = 0\) for “CHD absent,” the mean is simply the **proportion of individuals with CHD** in that age group.

This mean serves as an empirical estimate of the **conditional expectation** of \(Y\) given \(x\) (AGE):

$$
E(Y \mid x) \approx \text{Proportion with CHD in that age group}.
$$


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
$$
E(Y \mid x) = \beta_0 + \beta_1 x
$$


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
# Here’s your passage rewritten in **well-structured Markdown format**, preserving the technical details, equations, and flow for clarity and readability:

---

# 1.3 **Assessing Significance of Independent Variables**

When building a regression model, we often want to determine whether the **independent variables** are **significantly related** to the outcome variable.

This typically involves **formulating and testing a statistical hypothesis**:

> Does the model that includes a particular variable provide more information about the outcome than a model that does not include that variable?

The answer comes from **comparing the observed values of the response variable** to the **predicted values** under two models:

1. **Model including the variable**
2. **Model excluding the variable**

The mathematical function used for this comparison depends on the problem.

* If predictions **with the variable** are better or more accurate than **without the variable**, the variable is considered **significant**.
* Note: This is a **relative assessment**, not an absolute measure of goodness-of-fit (discussed in Chapter 5).

---

## **Linear Regression Approach**

In **linear regression**, significance of an independent variable is assessed via the **analysis of variance (ANOVA) table**, which partitions the total sum-of-squared deviations of observations about the mean:

1. **Residual sum-of-squares (SSE):** deviations of observations from the regression line
2. **Regression sum-of-squares (SSR):** deviations of predicted values from the mean

Mathematically, if ( y_i ) is the observed value and ( \hat{y}_i ) is the predicted value for the ( i )-th individual:

[
\text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
]

* **Model without the independent variable:**

  * Only parameter is ( \beta_0 )
  * Predicted values ( \hat{y}_i = \bar{y} ) (mean of the response)
  * SSE equals the **total sum-of-squares**

* **Model with the independent variable:**

  * Decrease in SSE is due to a **non-zero slope coefficient** for the variable
  * Change in SSE attributed to **SSR** (regression source of variability)

> A **large SSR** suggests the variable is important; a **small SSR** suggests it does not help predict the response.

---

## **Logistic Regression Approach**

The guiding principle is the **same as linear regression**: compare observed response values to predicted values from models **with and without the variable**.

* In logistic regression, comparison is based on the **log-likelihood function**.
* Conceptually, an observed response value can be thought of as a prediction from a **saturated model** (a model with as many parameters as data points).

---

### **Likelihood Ratio and Deviance**

The **likelihood ratio** compares models with and without a variable:

[
\Lambda = \frac{L(\text{model without variable})}{L(\text{model with variable})}
]

To use this for **hypothesis testing**, we take **minus twice the log**:

[
D = -2 \log \Lambda
]

* ( D ) is called the **deviance**
* For logistic regression, **deviance plays the same role as SSE in linear regression**
* When computed for linear regression, **deviance equals SSE exactly**

Using predicted probabilities ( \hat{\pi}_i = \hat{\pi}(x_i) ), the deviance can be expressed as:

[
D = -2 \sum_{i=1}^{n} \Big[ y_i \log(\hat{\pi}_i) + (1-y_i) \log(1-\hat{\pi}_i) \Big]
]

* This expression forms the basis for **likelihood ratio tests** in logistic regression

> In essence, **deviance measures how well the model predicts observed outcomes**, analogous to residual sum-of-squares in linear regression.

---

This Markdown version is now structured with:

* Headings for clarity
* Stepwise explanation
* Mathematical equations properly formatted
* Conceptual notes highlighting the connection between **linear and logistic regression**

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

