Alright, let's explore **Longitudinal Analysis / Repeated Measures**, which is a critical area when you have data collected from the *same subjects* (or units) over multiple time points. While our previous discussions focused on single time series, this topic deals with comparing *groups* based on their time series trajectories.

-----

## 10.1 Repeated Measures and Longitudinal Data

**Longitudinal Analysis** and **Repeated Measures Analysis** are terms used interchangeably (though "longitudinal" often implies more time points, and "repeated measures" might imply fewer, pre-defined time points) to describe the statistical analysis of data where the same outcome variable is measured on the same observational units (individuals, patients, plots, etc.) at multiple time points.

**Key Characteristic:** The defining feature is the **within-subject correlation**. Measurements taken on the same subject over time are likely to be correlated (e.g., a person's blood pressure today is highly related to their blood pressure yesterday). This violates the assumption of independence of observations, which is fundamental to standard statistical methods like Ordinary Least Squares (OLS) regression or standard ANOVA.

**Experimental Design for Repeated Measures Data:**

  * **Structure:** Typically involves one or more **between-subjects factors** (e.g., treatment group, gender, disease status) and one or more **within-subjects factors** (time).
  * **Example:** In your "Phelibitis Example," you might have:
      * **Between-subjects factor:** "Treatment Group" (e.g., Drug A, Drug B, Placebo).
      * **Within-subjects factor:** "Time" (e.g., measurements taken at Day 1, Day 3, Day 7, Day 14 post-treatment).
  * **Goal:** To compare how the outcome variable changes over time *between* different treatment groups or other between-subject factors.

### A Commonly Used Model (Repeated Measures ANOVA - RM-ANOVA)

Traditionally, Repeated Measures ANOVA (RM-ANOVA) is a common approach.

  * **Conceptual Model:**
    Outcome = Group Effect + Time Effect + Group x Time Interaction + Subject Effect + Error
  * **Purpose:**
      * Test for **main effect of time:** Does the outcome change significantly over time, across all groups?
      * Test for **main effect of group:** Are there overall differences between groups, averaged across time points?
      * Test for **Group x Time Interaction:** Is the pattern of change over time *different* for different groups? This is often the most important effect in these designs.

**Consequence of the Assumptions for This Model – Compound Symmetry:**
Standard RM-ANOVA (and especially older software implementations) often assumes a specific correlation structure for the within-subject errors, known as **Compound Symmetry**.

  * **Compound Symmetry:** Assumes that:

    1.  The **variance** of the response variable is constant at all time points.
    2.  The **correlation** between any two repeated measures on the *same subject* is constant, regardless of how far apart in time those measures are taken.

    <!-- end list -->

      * Example: Correlation between Day 1 and Day 2 is the same as between Day 1 and Day 10.

  * **Consequence of Violation:** If compound symmetry is violated (which is very common in real longitudinal data, as correlations usually decrease with increasing time lag), standard RM-ANOVA can lead to:

      * **Inflated Type I error rates:** You are more likely to find a significant effect when none truly exists (false positives), particularly for the within-subjects (time) effects and interaction terms.
      * **Inefficient estimates:** The parameter estimates might not be the most precise.

### Generalized Least Squares (GLS)

GLS is a more flexible and powerful method for analyzing repeated measures and longitudinal data because it can explicitly model various **correlation structures** within the errors, rather than strictly assuming compound symmetry.

  * **Concept:** GLS is an extension of OLS. While OLS assumes errors are independent and have constant variance (homoscedasticity), GLS accounts for situations where errors are correlated (autocorrelated) and/or have non-constant variance (heteroscedasticity).
  * **How it Works:** GLS uses a weighting matrix that takes into account the specified error correlation structure and heteroscedasticity. It transforms the data so that OLS assumptions are met on the transformed data, leading to more efficient and unbiased standard errors.
  * **Advantages over RM-ANOVA (under violation of assumptions):**
      * Provides valid standard errors and p-values even when compound symmetry is violated.
      * More flexible in modeling realistic within-subject correlation patterns.
      * Can handle unbalanced data (missing data points) more gracefully than traditional RM-ANOVA.

### Identifying and Interpreting Various Correlation Structures

When using GLS, you need to specify or estimate the covariance structure of the errors. Common structures include:

1.  **Compound Symmetry (CS):** (Already discussed) Assumes equal variance and equal correlation between any pair of repeated measures. Simplest, but often unrealistic.
2.  **Autoregressive of order 1 (AR(1)):**
      * **Concept:** Assumes that the correlation between two measures decreases exponentially as the time interval between them increases.
      * **Correlation between $Y\_t$ and $Y\_{t-k}$:** $\\rho^k$.
      * **Interpretation:** More realistic for many time series, where observations closer in time are more strongly correlated.
3.  **Compound Symmetry with Heterogeneous Variances (CSH):**
      * **Concept:** Allows the variance to differ at each time point but maintains constant correlation between any two measures.
4.  **Autoregressive of order 1 with Heterogeneous Variances (ARH(1)):**
      * **Concept:** Combines AR(1) correlation with differing variances at each time point.
5.  **Unstructured (UN):**
      * **Concept:** No assumptions are made about the variances or correlations. Each variance and each unique correlation is estimated separately.
      * **Interpretation:** The most flexible, but also requires the most parameters to be estimated. Can only be used with relatively few time points and many subjects, otherwise it runs out of degrees of freedom. Provides a baseline to see if simpler structures are sufficient.

### Comparing GLS Models with Different Correlation Structures

The choice of correlation structure is crucial. You typically compare models using:

1.  **Information Criteria (AIC, BIC):** Lower values indicate a better-fitting model while penalizing complexity. GLS models with different correlation structures can be compared using these.
2.  **Likelihood Ratio Test (LRT):** If one correlation structure is nested within another (e.g., CS is nested within UN), you can use an LRT to formally test if the more complex model provides a significantly better fit.
3.  **Visual Inspection of Residuals/ACF:** After fitting a GLS model, examine the residuals to ensure that the chosen correlation structure has successfully accounted for the within-subject dependencies. The residuals across subjects should ideally be uncorrelated.

### Estimating Polynomial Effects across Time

Instead of treating "Time" as a categorical factor (as in traditional RM-ANOVA), it can be treated as a continuous variable, and its relationship with the outcome can be modeled using polynomials (linear, quadratic, cubic, etc.).

  * **Concept:** This allows you to describe the "shape" of the trend over time.
      * **Linear Effect:** A constant rate of change over time ($\\beta\_1 \\times \\text{Time}\_t$).
      * **Quadratic Effect:** The rate of change itself changes over time (e.g., accelerating or decelerating growth, like a parabolic curve, $\\beta\_1 \\times \\text{Time}\_t + \\beta\_2 \\times \\text{Time}\_t^2$).
      * **Cubic Effect:** Allows for more complex S-shaped or undulating trends.
  * **Use in GLS:** You can include polynomial terms of time as continuous predictors in your GLS model, along with interaction terms between these polynomials and your group factors (e.g., Group \* Time, Group \* Time$^2$).
  * **Interpretation of Interaction Terms (Group x Time / Group x Polynomial of Time):**
      * These terms are critical. A significant `Group x Time` interaction (or `Group x Time^2`, etc.) indicates that the **trajectory of the outcome over time is significantly different across different groups.**
      * Example: Drug A might show a linear decrease in symptoms, while a placebo group shows no change, or Drug B shows an initial rapid decrease followed by a plateau (requiring a quadratic term).

### R Code for the Phlebitis Example (Conceptual using `nlme` package)

The `nlme` package in R is excellent for GLS modeling of repeated measures data.

```r
# install.packages("nlme")
library(nlme)

# Assume your data is in a long format dataframe 'df_phlebitis'
# with columns: SubjectID, Time (e.g., 1, 3, 7, 14), TreatmentGroup, Outcome

# 1. Visualize the data (crucial first step!)
# plot(Outcome ~ Time, data=df_phlebitis, groups=TreatmentGroup, type="l", col=1:nlevels(df_phlebitis$TreatmentGroup))
# This helps visually identify trends and group differences

# 2. Model Repeated Measures ANOVA (using lme for flexibility with correlations)
# Formula: Outcome ~ TreatmentGroup * Time (as a factor)
# random = ~1 | SubjectID (random intercept for each subject)
# correlation = corCompSymm() (specifies compound symmetry for within-subject errors)
model_cs <- lme(Outcome ~ TreatmentGroup * factor(Time),
                data = df_phlebitis,
                random = ~1 | SubjectID,
                correlation = corCompSymm())
summary(model_cs)
anova(model_cs) # Get ANOVA table with F-tests

# 3. Model with different correlation structures using GLS
# AR(1) structure:
model_ar1 <- lme(Outcome ~ TreatmentGroup * factor(Time),
                 data = df_phlebitis,
                 random = ~1 | SubjectID,
                 correlation = corAR1(form = ~ Time | SubjectID)) # Time needs to be numeric here
summary(model_ar1)

# Unstructured (UN) structure:
# Use corSymm() for general correlation matrix - 'unstructured' option in some software
# correlation = corSymm(form = ~ 1 | SubjectID, fixed = F, type = "unstructured") # This is for fixed correlation matrix

# For Unstructured in lme:
model_un <- lme(Outcome ~ TreatmentGroup * factor(Time),
                data = df_phlebitis,
                random = ~1 | SubjectID,
                correlation = corSymm(form = ~ 1 | SubjectID)) # This is the UN structure


# 4. Compare GLS models
AIC(model_cs, model_ar1, model_un)
BIC(model_cs, model_ar1, model_un)

# Likelihood Ratio Test (for nested models, e.g., CS vs AR(1))
anova(model_cs, model_ar1) # Tests if AR(1) is significantly better than CS

# 5. Estimate Polynomial Effects (Time as numeric)
# Outcome ~ TreatmentGroup * poly(Time, 2) for quadratic effect
model_poly <- lme(Outcome ~ TreatmentGroup * poly(Time, 2),
                  data = df_phlebitis,
                  random = ~1 | SubjectID,
                  correlation = corAR1(form = ~ Time | SubjectID)) # Or other structure
summary(model_poly) # Look at coefficients for poly(Time, 1) and poly(Time, 2)
```
Here's a structured explanation of **Lesson 10 – Longitudinal Analysis / Repeated Measures** (for interview prep), covering the key concepts that you might need to know:

---

## **1️⃣ Longitudinal Analysis / Repeated Measures Overview**

### Definition

* **Longitudinal Data**: Data collected from the same subjects repeatedly over time.
* **Repeated Measures**: A type of longitudinal data where each subject is measured under different conditions or at different time points.
* Goal: **Understand changes within subjects over time and compare differences across treatment groups**.

---

## **2️⃣ Experimental Design for Repeated Measures**

* **Subjects**: Same individuals measured multiple times.
* **Treatments**: Groups receiving different conditions/interventions.
* **Time Factor**: Measurements are not independent; correlations within subjects must be accounted for.

Example:

| Subject | Treatment | Time 1 | Time 2 | Time 3 |
| ------- | --------- | ------ | ------ | ------ |
| 1       | A         | 4.1    | 5.2    | 6.1    |
| 2       | B         | 3.9    | 4.5    | 4.7    |

---

## **3️⃣ Commonly Used Model (Linear Mixed / Repeated Measures ANOVA)**

A simple model:

$$
Y_{ij} = \mu + \tau_i + \beta_j + (\tau \beta)_{ij} + \epsilon_{ij}
$$

Where:

* $Y_{ij}$: Response for treatment $i$ at time $j$
* $\mu$: Overall mean
* $\tau_i$: Treatment effect
* $\beta_j$: Time effect
* $(\tau \beta)_{ij}$: Interaction effect (treatment × time)
* $\epsilon_{ij}$: Random error (correlated within subjects)

---

## **4️⃣ Compound Symmetry (CS) Assumption**

* **Definition**: Assumes:

  * Equal correlation between any two time points for the same subject
  * Equal variance across all time points
* **Covariance structure:**

$$
\Sigma = 
\begin{pmatrix}
\sigma^2 & \rho\sigma^2 & \rho\sigma^2 \\
\rho\sigma^2 & \sigma^2 & \rho\sigma^2 \\
\rho\sigma^2 & \rho\sigma^2 & \sigma^2
\end{pmatrix}
$$

* **Problem**: Often unrealistic because correlations usually decay as time difference increases.

---

## **5️⃣ Generalized Least Squares (GLS)**

When CS assumption is violated, **GLS** allows modeling with different covariance structures:

* **AR(1)**: Correlation decreases exponentially with time lag.
* **Unstructured (UN)**: Each pair of time points has its own correlation.
* **Toeplitz**: Correlations depend on distance between time points but are not necessarily exponential.

The GLS estimator:

$$
\hat{\beta}_{GLS} = (X' \Sigma^{-1} X)^{-1} X' \Sigma^{-1} Y
$$

Where:

* $\Sigma$: Covariance matrix accounting for correlations between repeated measures.

---

## **6️⃣ Interaction Terms**

* Interaction between **Treatment × Time** is crucial.
* If significant:

  * Treatment effect **changes over time**.
* If not significant:

  * Treatment effect is consistent across time.

---

## **7️⃣ Polynomial Effects Across Time**

* Sometimes, the effect of time is **nonlinear**.
* Polynomial terms (e.g., quadratic, cubic) can capture this:

$$
Y_{ij} = \beta_0 + \beta_1 t_j + \beta_2 t_j^2 + \dots + \epsilon_{ij}
$$

* Helps model curved growth patterns or delayed treatment effects.

---

## **8️⃣ R Code Example (Phlebitis Study)**

Suppose we have:

* Outcome = measure of phlebitis
* Groups = treatment groups
* Time = measurement days

```R
# Load data
data <- read.csv("phlebitis.csv")

# Fit repeated measures ANOVA
library(nlme)
model_gls <- gls(response ~ treatment * time,
                 data = data,
                 correlation = corCompSymm(form = ~ 1 | subject))

summary(model_gls)

# Fit AR(1) correlation structure
model_ar1 <- gls(response ~ treatment * time,
                 data = data,
                 correlation = corAR1(form = ~ 1 | subject))

anova(model_gls, model_ar1)  # Compare models

# Polynomial effects
model_poly <- gls(response ~ treatment * poly(time, 2),
                  data = data,
                  correlation = corAR1(form = ~ 1 | subject))
summary(model_poly)
```

---

## ✅ Key Takeaways for Interviews

* **Repeated measures ≠ independent observations** – correlation must be modeled.
* **Compound symmetry is a strong assumption**; often replaced by AR(1) or unstructured covariance.
* **GLS is used instead of OLS** when errors are correlated or heteroscedastic.
* **Interaction terms tell if treatment effect changes over time**.
* **Polynomial effects model nonlinear trends** in longitudinal data.
* **Model comparison via AIC or likelihood ratio tests** helps select the right covariance structure.

---

Would you like me to prepare a **one-page "Lesson 10 Interview Cheatsheet" (flowchart + equations + R commands)** like the one I made for Lesson 9?
