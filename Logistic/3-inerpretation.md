Perfect! Let’s create a **comprehensive, structured set of notes on interpreting fitted logistic regression models**, based on your excerpt from Hosmer et al., covering **dichotomous, polychotomous, and continuous independent variables, interactions, and odds ratios**. I’ll make this fully standalone and intuitive.

---

# **Interpretation of the Fitted Logistic Regression Model**

The goal of interpreting a logistic regression model is to answer:

> *What do the estimated coefficients tell us about the relationship between the predictors and the outcome, in the context of the research question?*

---

## **1. General Principle**

Logistic regression models the **logit** (log-odds) of the outcome as a linear function of predictors:

[
g(x) = \text{logit}[\pi(x)] = \ln \frac{\pi(x)}{1 - \pi(x)} = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p
]

* The **slope coefficient (\beta_j)** represents the change in **logit** for a one-unit change in (X_j), holding other variables constant:

[
\beta_j = g(x_j + 1) - g(x_j)
]

* The **odds ratio (OR)** is:

[
OR_j = e^{\beta_j}
]

* Interpretation of (\beta_j) and (OR_j) depends on the type of predictor.

---

## **2. Dichotomous Independent Variable**

For a binary predictor (X \in {0,1}):

[
g(1) - g(0) = (\beta_0 + \beta_1) - \beta_0 = \beta_1
]

* Predicted probabilities:

[
\pi(1) = \frac{e^{\beta_0 + \beta_1}}{1 + e^{\beta_0 + \beta_1}}, \quad
\pi(0) = \frac{e^{\beta_0}}{1 + e^{\beta_0}}
]

* **Odds ratio**:

[
OR = \frac{\pi(1)/(1-\pi(1))}{\pi(0)/(1-\pi(0))} = e^{\beta_1}
]

**Interpretation:**

* (OR > 1): higher odds of outcome when (X = 1)
* (OR < 1): lower odds of outcome when (X = 1)

**Example:**

* If (\beta_1 = 0.55), (OR = e^{0.55} \approx 1.73) → 73% higher odds of outcome for (X = 1) vs (X = 0).

---

## **3. Polychotomous (Categorical) Independent Variable**

For predictors with >2 categories (e.g., risk levels: Low, Medium, High):

* Choose a **reference category** (e.g., “Low”)
* Include **dummy variables** for the remaining categories.

[
g(x) = \beta_0 + \beta_1 I(\text{Medium}) + \beta_2 I(\text{High})
]

* Odds ratios are **relative to the reference category**:

[
OR_{\text{Medium vs Low}} = e^{\beta_1}, \quad OR_{\text{High vs Low}} = e^{\beta_2}
]

**Example (GLOW study):**

| Category     | OR   | 95% CI       | ln(OR) |
| ------------ | ---- | ------------ | ------ |
| Same vs Less | 1.73 | (1.02, 2.91) | 0.55   |
| More vs Less | 2.48 | (1.46, 4.22) | 0.91   |

---

## **4. Continuous Independent Variable**

* The slope (\beta_j) represents the change in log-odds for a **one-unit increase** in the predictor.

[
g(x+1) - g(x) = \beta_j
]

* **Odds ratio**: (OR = e^{\beta_j}) per unit increase.
* **Effect modification / interaction**: If the relationship differs across levels of another variable, the effect is **not constant**.

**Example:**

* AGE and GENDER interaction: the logit slope for AGE differs between males and females.
* The odds ratio for GENDER depends on AGE:

[
OR_{\text{male vs female}}(age) = e^{\beta_{\text{gender}} + \beta_{\text{interaction}} \cdot \text{AGE}}
]

* Age is an **effect modifier** of gender.

---

## **5. Interaction Terms**

* Interaction term: (X_1 \cdot X_2)
* Interpretation: the effect of one predictor **depends on the level of another**.

[
g(x) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2
]

* (\beta_3) = additional change in log-odds when both (X_1) and (X_2) occur.
* Main effect coefficients are **conditional**: (\beta_1) = effect of (X_1) when (X_2 = 0).

> Important: after including interaction, there is **no single “adjusted” OR** applicable to all levels of the interacting variable. ORs must be estimated at specific values.

---

## **6. Presenting Fitted Values**

1. **Predicted logit:** (\hat{g}(x) = \hat{\beta}_0 + \sum \hat{\beta}_j X_j)
2. **Predicted probability:**

[
\hat{\pi}(x) = \frac{e^{\hat{g}(x)}}{1 + e^{\hat{g}(x)}}
]

3. **Confidence interval for probability:**

[
CI = \frac{e^{\hat{g}(x) \pm z SE[\hat{g}(x)]}}{1 + e^{\hat{g}(x) \pm z SE[\hat{g}(x)]]}}
]

**Example:**

* (\hat{g}(50\text{y}) = 0.14), SE = 0.2
* 95% CI probability: 0.435 – 0.677 → mean probability of outcome at age 50

4. Fitted probabilities can be used for **classification**, linking logistic regression to **discriminant analysis**.

---

## **7. Summary Guidelines**

| Predictor Type | Coefficient Interpretation                                | Odds Ratio Interpretation                    |
| -------------- | --------------------------------------------------------- | -------------------------------------------- |
| Dichotomous    | Change in logit when X=1 vs X=0                           | e^β1 → odds multiplier for X=1               |
| Polychotomous  | Change in logit vs reference category                     | e^βj → odds multiplier relative to reference |
| Continuous     | Change in logit per unit increase                         | e^βj → odds multiplier per unit              |
| Interaction    | Change in logit depends on levels of interacting variable | OR must be calculated at specific levels     |

**Key Takeaways:**

* Logistic regression coefficients quantify effects on the **logit (log-odds)**.
* Exponentiated coefficients → **odds ratios**, easier to interpret in context.
* Interaction terms require careful interpretation: effects are **conditional**.
* Fitted probabilities provide concrete predictions for individuals or groups.

---

If you want, I can **draw a diagram showing how dichotomous, continuous, and interaction terms affect the logit and predicted probabilities**, which makes the interpretation of coefficients visually intuitive.

Do you want me to make that diagram next?
