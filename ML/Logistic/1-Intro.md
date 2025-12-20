## David G. Kleinbaum Mitchel Klein Logistic Regression A Self‐Learning Text




### Logistic Regression and the Logistic Function

Logistic regression is a mathematical modeling approach used to describe the relationship between one or more independent variables (**Xs**) and a **dichotomous dependent variable** (e.g., presence or absence of a disease).

At the core of logistic regression lies the **logistic function**, which provides the mathematical foundation of the model. The function, denoted as **f(z)**, is expressed as:

[
f(z) = \frac{1}{1 + e^{-z}}
]

---

### Key Properties of the Logistic Function

1. **Range between 0 and 1**

   * When ( z \to -\infty ), ( f(z) \to 0 ).
   * When ( z \to +\infty ), ( f(z) \to 1 ).
   * Thus, ( f(z) ) always lies between **0 and 1**, making it ideal for modeling probabilities.

2. **S-Shaped Curve**

   * The logistic function has a characteristic **sigmoid (S-shaped) curve**.
   * Starting from very low values of ( z ), the function stays close to **0**.
   * As ( z ) increases, ( f(z) ) rises sharply in the middle range.
   * Finally, for large values of ( z ), the curve flattens and approaches **1**.

---

### Epidemiological Relevance

* In epidemiology, if **z** represents a combined index of several risk factors, then **f(z)** reflects the **probability (risk)** of disease for that value of ( z ).
* The S-shape illustrates a **threshold effect**:

  * For low values of ( z ), the risk remains minimal.
  * After crossing a threshold, risk increases rapidly.
  * Eventually, the risk saturates at a high level near 1.
* This makes the logistic model particularly attractive for studying disease conditions influenced by multiple risk factors.

---

### Why Logistic Regression is Popular

The logistic model is widely used because it provides:

* ✅ **Probability estimates constrained between 0 and 1**
* ✅ **An intuitive S-shaped curve** that effectively describes how multiple risk factors jointly influence disease risk

---

Got it 👍 — I’ll rewrite and **restructure your passage into a clean, well-formatted explanation** with equations, definitions, and the worked-out example clearly separated.

---

# Logistic Model from the Logistic Function

### 1. The Logistic Function

The logistic function is:

[
f(z) = \frac{1}{1 + e^{-z}}
]

To build a **logistic regression model**, we express ( z ) as a linear combination of independent variables:

[
z = a + b_1X_1 + b_2X_2 + \dots + b_kX_k
]

where:

* ( X_1, X_2, \dots, X_k ) are independent variables,
* ( a ) is the intercept,
* ( b_i ) are regression coefficients (parameters to be estimated).

Substituting this into the logistic function gives:

[
P(D = 1 \mid X_1, X_2, \dots, X_k) = \frac{1}{1 + e^{-\left(a + \sum_{i=1}^k b_iX_i\right)}}
]

This represents the **probability** of disease ( D ) given the values of risk factors ( X_1, \dots, X_k ).
For simplicity, we denote this probability as:

[
P(\mathbf{X}) = \frac{1}{1 + e^{-\left(a + b_1X_1 + b_2X_2 + \dots + b_kX_k\right)}}
]

---

### 2. Example: Coronary Heart Disease (CHD)

We consider a study of **609 white males** followed for 9 years. The outcome variable is:

[
D = \text{CHD (1 = disease, 0 = no disease)}
]

The independent variables are:

* ( X_1 = \text{CAT (Catecholamine level, 1 = high, 0 = low)} )
* ( X_2 = \text{AGE (continuous)} )
* ( X_3 = \text{ECG (Electrocardiogram, 1 = abnormal, 0 = normal)} )

The fitted logistic regression model is:

[
\hat{P}(\mathbf{X}) = \frac{1}{1 + e^{-\left(-3.911 + 0.652 \cdot CAT + 0.029 \cdot AGE + 0.342 \cdot ECG \right)}}
]

---

### 3. Predicted Risk for an Individual

Suppose we want the predicted risk for an individual with:

* ( CAT = 1 ) (high),
* ( AGE = 40 ),
* ( ECG = 0 ) (normal).

Plugging into the model:

[
\hat{P}(X) = \frac{1}{1 + e^{-\left(-3.911 + 0.652(1) + 0.029(40) + 0.342(0)\right)}}
]

Simplify step by step:
[
= \frac{1}{1 + e^{-\left(-3.911 + 0.652 + 1.16\right)}}
]
[
= \frac{1}{1 + e^{-(-2.101)}}
]
[
= \frac{1}{1 + e^{2.101}}
]
[
= \frac{1}{1 + 8.173} = 0.109
]

So, this person’s **predicted risk = 11%**.

---

### 4. Comparing Two Individuals

Now, compare with another person who has the same **AGE = 40** and **ECG = 0**, but **CAT = 0** (low catecholamine level).

* For **CAT = 0**:

[
\hat{P}(X) = 0.060 \quad (\text{≈ 6% risk})
]

* For **CAT = 1**:

[
\hat{P}(X) = 0.109 \quad (\text{≈ 11% risk})
]

---

### 5. Interpretation

* A high catecholamine level (( CAT = 1 )) increases the predicted risk of CHD from **6% → 11%** for a 40-year-old male with a normal ECG.
* This illustrates how logistic regression quantifies the effect of multiple risk factors on disease probability.

---

✅ **Final Fitted Model:**

[
\hat{P}(X) = \frac{1}{1 + e^{-(-3.911 + 0.652 \cdot CAT + 0.029 \cdot AGE + 0.342 \cdot ECG)}}
]

---
Perfect — thanks for sharing all that detail. Let me rewrite and **organize this into a clean, well-formatted, and clear explanation**. I’ll keep the technical precision but make the flow much easier to follow, with equations and sections.

---

# Risk Ratios, Odds Ratios, and Logistic Regression

### 1. Direct Estimation of Risk Ratios (RR)

From the fitted logistic regression model, we can calculate the **predicted risk** for different individuals by plugging their covariates into the model.

Example:

* Predicted risk for a 40-year-old with **CAT = 1**, **ECG = 0**:
  [
  \hat{P}_1(X) = 0.109 \quad (11%)
  ]

* Predicted risk for a 40-year-old with **CAT = 0**, **ECG = 0**:
  [
  \hat{P}_0(X) = 0.060 \quad (6%)
  ]

The **risk ratio (direct estimate)** is:

[
dRR = \frac{\hat{P}_1(X)}{\hat{P}_0(X)} = \frac{0.109}{0.060} = 1.82
]

**Interpretation:**
A person with high catecholamine levels has **1.82 times the risk** of coronary heart disease (CHD) compared to a similar person with low catecholamine levels, assuming both are 40 years old and have normal ECG readings.

---

### 2. Conditions for Direct Risk Estimation

Direct estimation of **RR** using a logistic model is possible only if:

1. The study is a **follow-up (cohort) study**, so that true risks can be estimated.
2. We specify values for **all independent variables** in the model to compute individual risk.

If either condition is not met (e.g., case-control study, missing covariates), then **RR cannot be estimated directly**.

---

### 3. Logistic Regression and Odds Ratios (OR)

* In reality, logistic regression naturally estimates **odds ratios (OR)**, not risk ratios.
* The odds ratio can be estimated regardless of study design: **follow-up, case-control, or cross-sectional**.
* This is because logistic regression models the **log-odds of disease**, not the direct probability.

Formally, the **odds ratio** from a 2×2 table is:

[
cOR = \frac{ad}{bc}
]

where (a, b, c, d) are cell counts.

In logistic regression, the odds ratio for a predictor (X_i) is:

[
OR = e^{b_i}
]

where (b_i) is the regression coefficient.

---

### 4. Why Case-Control and Cross-Sectional Studies Can’t Estimate Risk

* In **follow-up studies**, risk estimates are of the form:

[
P(D = 1 \mid E) \quad (\text{risk given exposure})
]

* In **case-control and cross-sectional studies**, the data structure only permits conditional probabilities of the form:

[
P(E \mid D) \quad (\text{exposure given disease})
]

Thus:

* **Follow-up → risk (RR) is valid**
* **Case-control / cross-sectional → only OR is valid**

This same distinction applies in logistic regression:

* In follow-up designs, both (a) (intercept) and (b_i) (coefficients) can be validly estimated → predicted risks are possible.
* In case-control designs, (a) cannot be validly estimated (because sampling fractions are unknown). Only the coefficients (b_i) can be estimated reliably → meaning we can estimate **ORs**, but not predicted probabilities.

---

### 5. Example: Case-Control Logistic Model

Suppose we fit a logistic model to case-control data with 3 predictors. A typical output might look like:

| Variable | Coefficient ((\hat{b})) |
| -------- | ----------------------- |
| Constant | -4.50 ((\hat{a}))       |
| (X_1)    | 0.70 ((\hat{b}_1))      |
| (X_2)    | 0.05 ((\hat{b}_2))      |
| (X_3)    | 0.42 ((\hat{b}_3))      |

Here:

* The constant ((\hat{a} = -4.5)) is **not validly interpretable** in a case-control study.
* But the coefficients ((\hat{b}_1, \hat{b}_2, \hat{b}_3)) are valid → they can be exponentiated to obtain **odds ratios**:

[
OR_i = e^{\hat{b}_i}
]

---

### 6. Key Takeaways

* ✅ Logistic regression is designed for modeling **probabilities of disease given covariates**.
* ✅ In **cohort (follow-up) studies**, it can provide **predicted risks and risk ratios**.
* ✅ In **case-control or cross-sectional studies**, it cannot provide risks, but it can provide **odds ratios**.
* ✅ The **intercept (a)** is unreliable in case-control studies, but the **slopes (b’s)** still yield valid measures of association (ORs).

---
Got it 👍 — what you’ve pasted is a full block of textbook-style notes that mix the **motivation for using risk ratios (RR) vs. odds ratios (OR)** with the **mathematical derivation of the logit form** and finally the **general and special cases of interpreting odds ratios in logistic regression**.

Let me clean this up into a **clear, structured, and easy-to-read explanation** while keeping all the details intact.

---

# Logistic Regression: Logit Transformation, Odds, and Odds Ratios

---

## 1. Preference for Risk Ratios in Follow-Up Studies

* In **follow-up (cohort) studies**, it is common to prefer a **risk ratio (RR)** rather than an **odds ratio (OR)**.
* This is because RR is more **directly interpretable** in terms of risk.
* Logistic regression can estimate RR **if and only if** values for **all independent variables (covariates)** are specified.

**Example:**

* Compare two persons (CAT = catecholamine, AGE = age, ECG = electrocardiogram status):

  * Group 1: CAT = 1, AGE = 40, ECG = 0
  * Group 0: CAT = 0, AGE = 40, ECG = 0

Here, we can compute risk ratio (RR) since we’ve fixed all covariates.

---

## 2. The Logit Transformation

The logistic regression model for probability is:

[
P(X) = \frac{1}{1 + e^{-(a + \sum b_i X_i)}}
]

The **logit** is defined as:

[
\text{logit } P(X) = \ln \left( \frac{P(X)}{1 - P(X)} \right)
]

---

### Key Steps:

1. Compute (P(X)).
2. Compute (1 - P(X)).
3. Form the odds:
   [
   \text{odds}(X) = \frac{P(X)}{1 - P(X)}
   ]
4. Take the natural log:
   [
   \text{logit } P(X) = \ln(\text{odds}(X))
   ]

---

### Simplification

If we substitute the logistic model for (P(X)):

[
\frac{P(X)}{1 - P(X)} = e^{(a + \sum b_i X_i)}
]

Then:

[
\text{logit } P(X) = a + \sum b_i X_i
]

This shows that the **logit is linear in the predictors**.

---

## 3. Interpretation of Model Parameters

* **Intercept ((a))**:

  * Represents the **log odds** of the event when all predictors (X_i = 0).
  * In practice, this may not be meaningful if no subject can have all predictors = 0.
  * A more general interpretation: (a) is the **log of the baseline odds** when predictors are not included.

* **Coefficients ((b_i))**:

  * Represent the **change in log odds** for a **one-unit increase** in (X_i), with other variables fixed.
  * Equivalently:
    [
    e^{b_i} = \text{odds ratio (OR) for a 1-unit change in } X_i
    ]

---

## 4. Odds Ratios in Logistic Regression

An **odds ratio (OR)** compares the odds of disease for two groups:

[
OR_{X_1, X_0} = \frac{\text{odds}(X_1)}{\text{odds}(X_0)}
]

* Example:

  * Group 1: CAT = 1, AGE = 40, ECG = 0
  * Group 0: CAT = 0, AGE = 40, ECG = 0

Substituting into the model:

[
OR_{X_1, X_0} = e^{b_1}
]

---

### General Formula for OR

For any two groups (X_1) and (X_0):

[
OR_{X_1, X_0} = e^{\sum b_i (X_{1i} - X_{0i})}
]

* Each predictor contributes **multiplicatively** to the odds ratio.
* Example: if one variable contributes a factor of 3 and another a factor of 4, the combined OR = (3 \times 4 = 12).

---

### Special Case: Binary (0/1) Variables

* If (X_i) is a binary variable (e.g., CAT, ECG), then:

[
\text{Adjusted OR} = e^{b_i}
]

* Interpretation: the OR for (X_i = 1) vs (X_i = 0), controlling for other covariates.

---

## 5. Summary

* Logistic regression models **probability via the logit transformation**.
* **Logit** = log odds = (a + \sum b_i X_i).
* **Intercept ((a))**: baseline log odds.
* **Coefficient ((b_i))**: change in log odds for a unit change in (X_i).
* **Odds ratio (OR)**: (e^{b_i}) for binary predictors, or more generally (e^{\sum b_i (X_{1i} - X_{0i})}).
* In **follow-up studies**, RR is usually preferred and can be computed if all covariates are specified.
* In **case-control or cross-sectional studies**, only OR is valid.

---


.
