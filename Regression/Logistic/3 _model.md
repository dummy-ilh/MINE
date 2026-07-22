
# Module 3 — The Full Model Architecture & Weight Interpretation

## 1. WHY

We now have all the necessary components: a linear combination produces log-odds $z$, and the sigmoid function converts $z$ into a bounded probability $p$.

This module puts the full pipeline together: *"What does logistic regression compute feature-by-feature, and what does a learned weight $\beta_j$ actually mean?"*

Interpreting weights correctly is the single most tested conceptual checkpoint in senior machine learning interviews.

---

## 2. INTUITION: The Two-Stage Assembly Line

Logistic regression operates as a **two-stage pipeline**:

```text
FEATURES (X)  ---> [ STAGE 1: Linear Combination ] ---> LOG-ODDS (z) ---> [ STAGE 2: Sigmoid Activation ] ---> PROBABILITY (p)

```

1. **Stage 1 (Weighted Vote Aggregation):** Every feature casts a weighted vote based on its magnitude. Some features push toward Class 1 ($\beta_j > 0$), while others pull toward Class 0 ($\beta_j < 0$). We sum these weighted votes plus an intercept $\beta_0$. The resulting raw output $z \in (-\infty, +\infty)$ is the **log-odds score**.
2. **Stage 2 (Probability Mapping):** We pass $z$ through the sigmoid function $\sigma(z)$, which squashes the unbounded logit score into a valid probability $p \in (0, 1)$.

---

## 3. FORMULA

### Stage 1: Linear Combination (Log-Odds Space)

For $d$ input features $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$:

$$z = \text{logit}(p) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_d x_d = \boldsymbol{\beta}^T \mathbf{x} + \beta_0$$

Where:

* $z$ is the log-odds of the positive class ($Y=1$).
* $\beta_0$ (intercept) represents the baseline log-odds when all features $\mathbf{x} = \mathbf{0}$.
* $\beta_j$ (weight) represents the additive change in log-odds per unit change in $x_j$, holding all other features fixed.

### Stage 2: Sigmoid Transformation (Probability Space)

$$p = P(Y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-(\boldsymbol{\beta}^T \mathbf{x} + \beta_0)}}$$

---

## 4. WORKED NUMERIC EXAMPLE: Full Prediction Pipeline

Consider a customer churn model with two features:

* $x_1$: Number of complaints filed
* $x_2$: Months of account tenure

Suppose the trained model parameters are:


$$\beta_0 = -1.0, \quad \beta_1 = +0.8, \quad \beta_2 = -0.05$$

### Customer A ($x_1 = 3$ complaints, $x_2 = 6$ months tenure):

1. **Compute Logit $z$:**

$$z_A = -1.0 + (0.8 \times 3) + (-0.05 \times 6) = -1.0 + 2.4 - 0.3 = \mathbf{1.1}$$


2. **Compute Probability $p$:**

$$p_A = \sigma(1.1) = \frac{1}{1 + e^{-1.1}} \approx \frac{1}{1 + 0.3329} = \mathbf{0.7502} \quad (75.02\% \text{ churn probability})$$



### Customer B ($x_1 = 0$ complaints, $x_2 = 24$ months tenure):

1. **Compute Logit $z$:**

$$z_B = -1.0 + (0.8 \times 0) + (-0.05 \times 24) = -1.0 + 0 - 1.2 = \mathbf{-2.2}$$


2. **Compute Probability $p$:**

$$p_B = \sigma(-2.2) = \frac{1}{1 + e^{2.2}} \approx \frac{1}{1 + 9.025} = \mathbf{0.0998} \quad (9.98\% \text{ churn probability})$$



---

## 5. WEIGHT INTERPRETATION & THE ODDS RATIO

### The Common Candidate Trap

> *"If $\beta_1 = 0.8$, a 1-unit increase in complaints increases the probability of churn by $0.8$ ($80\%$)."* **(INCORRECT)**

Coefficients in logistic regression **do not operate linearly on probability space**. They operate **additively on log-odds space** and **multiplicatively on odds space**.

### Step-by-Step Derivation of the Odds Ratio (OR)

Let $x_j$ increase by $1$ unit while keeping all other features constant:

$$\text{Logit}_{\text{new}} = \text{Logit}_{\text{old}} + \beta_j$$

Exponentiating both sides converts the additive log-odds relation into a multiplicative odds relation:

$$\text{Odds}_{\text{new}} = e^{\text{Logit}_{\text{new}}} = e^{\text{Logit}_{\text{old}} + \beta_j} = e^{\text{Logit}_{\text{old}}} \times e^{\beta_j} = \text{Odds}_{\text{old}} \times e^{\beta_j}$$

$$\text{Odds Ratio (OR)} = \frac{\text{Odds}_{\text{new}}}{\text{Odds}_{\text{old}}} = e^{\beta_j}$$

### Interpreting Customer Complaints ($\beta_1 = +0.8$):

$$\text{OR}_1 = e^{0.8} \approx 2.2255$$

* **Plain-English Interpretation:** *"Holding tenure constant, each additional complaint **multiplies** the customer's odds of churning by approximately **$2.23\times$** (a $123\%$ increase in churn odds)."*

### Interpreting Tenure ($\beta_2 = -0.05$):

$$\text{OR}_2 = e^{-0.05} \approx 0.9512$$

* **Plain-English Interpretation:** *"Holding complaints constant, each additional month of tenure multiplies the customer's odds of churning by **$0.9512\times$** (a $4.88\%$ reduction in churn odds)."*

| Odds Ratio ($\text{OR} = e^{\beta_j}$) | Impact on Odds | Feature Type |
| --- | --- | --- |
| $\text{OR} > 1.0$ ($\beta_j > 0$) | Increases outcome odds | Risk factor / Positive correlation |
| $\text{OR} = 1.0$ ($\beta_j = 0$) | No effect on outcome odds | Neutral / Uninformative feature |
| $\text{OR} < 1.0$ ($\beta_j < 0$) | Decreases outcome odds | Protective factor / Negative correlation |

---

## 6. PROOF OF NON-LINEARITY ON PROBABILITY

Because the sigmoid curve flattens near $0$ and $1$, **the change in predicted probability per unit increase in $x_j$ depends entirely on the initial probability baseline**.

Let's test the effect of adding $+1.0$ unit to logit score $z$ ($\Delta z = 1.0$) across three different starting baselines:

```python
import numpy as np

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

baselines = [-3.0, 0.0, 3.0]
for z_start in baselines:
    z_end = z_start + 1.0
    p_start = sigmoid(z_start)
    p_end = sigmoid(z_end)
    delta_p = p_end - p_start
    print(f"z: {z_start:4.1f} -> {z_end:4.1f} | p: {p_start:6.4f} -> {p_end:6.4f} | Delta p = {delta_p:+.4f} ({delta_p*100:+.2f} pp)")

```

### Numerical Results:

* **Low Baseline ($z = -3.0$):** $p$ moves from $0.0474 \to 0.1192 \implies \mathbf{\Delta p = +7.18 \text{ percentage points}}$
* **Mid Baseline ($z = 0.0$):** $p$ moves from $0.5000 \to 0.7311 \implies \mathbf{\Delta p = +23.11 \text{ percentage points}}$
* **High Baseline ($z = 3.0$):** $p$ moves from $0.9526 \to 0.9820 \implies \mathbf{\Delta p = +2.94 \text{ percentage points}}$

> **Key Takeaway:** A weight $\beta_j$ exerts its maximum impact on probability near the decision boundary ($z \approx 0, p \approx 0.5$) and diminishes as $p$ approaches $0$ or $1$.

---

## 7. FAANG L5 ANGLE

### Common Interview Questions & Expert Responses

> **Q1: "In a logistic regression model, a feature has weight $\beta = 1.2$. What does this value mean?"**
> * **Answer:** A $1$-unit increase in this feature increases the **log-odds** of the outcome by $1.2$, holding all other features constant. Equivalently, taking $e^{1.2} \approx 3.32$, a $1$-unit increase **multiplies the odds** of the positive outcome by $3.32\times$ (a $232\%$ increase in odds).
> 
> 

> **Q2: "Can you state the effect of $\beta = 1.2$ directly in terms of probability change?"**
> * **Answer:** No, not as a single constant value. The effect on probability depends non-linearly on the baseline probability. The exact rate of change at a given point is determined by the partial derivative:
> 
> $$\frac{\partial p}{\partial x_j} = \beta_j \cdot p(1 - p)$$
> 
> 
> 
> The maximum marginal probability increase occurs when $p = 0.5$, yielding a maximum shift of $\frac{\beta_j}{4}$ (for $\beta_j = 1.2$, max probability shift is $+30\%$ percentage points per unit near $z=0$).
> 
> 

> **Q3: "Why do we say weights are additive in log-odds but multiplicative in odds?"**
> * **Answer:** Because $\text{logit}(p) = \ln(\text{odds}) = \boldsymbol{\beta}^T \mathbf{x}$. Adding $\beta_j$ to $\ln(\text{odds})$ corresponds to multiplying raw odds by $e^{\beta_j}$ due to the logarithmic identity $\ln(A) + B = \ln(A \cdot e^B)$.
> 
> 

---

## 8. PYTHON VERIFICATION

```python
import numpy as np

# Model Weights
beta_0 = -1.0
beta_1 = 0.8   # Complaints
beta_2 = -0.05 # Tenure

# Predict Function
def predict_churn(complaints: float, tenure: float) -> tuple[float, float]:
    z = beta_0 + beta_1 * complaints + beta_2 * tenure
    p = 1.0 / (1.0 + np.exp(-z))
    return z, p

# Evaluate Customers
for name, c, t in [("Customer A", 3, 6), ("Customer B", 0, 24)]:
    z, p = predict_churn(c, t)
    print(f"{name:10s} | Complaints: {c} | Tenure: {t:2d}m | Logit z: {z:6.3f} | Prob p: {p:.4f}")

# Odds Ratio Calculations
or_complaints = np.exp(beta_1)
or_tenure = np.exp(beta_2)
print(f"\nOdds Ratio (Complaints): {or_complaints:.4f}x per complaint")
print(f"Odds Ratio (Tenure)    : {or_tenure:.4f}x per month")

```

---

## 9. CONCEPT CHECK ANSWERS

1. **A model has weight $\beta = 1.2$ for number of prior purchases. What is the odds ratio and plain-English meaning?**
* *Answer:* $\text{OR} = e^{1.2} \approx 3.32$. Each additional prior purchase multiplies the odds of the outcome by approximately $3.32\times$, holding other features constant.


2. **True or False: "If $\beta_1 = 0.8$, a $1$-unit increase in $x_1$ increases probability by $80\%$."**
* *Answer:* **False**. Weights represent additive shifts in log-odds space, not linear percentage point shifts in probability space. The actual probability change depends on the baseline value $p(1-p)$.
