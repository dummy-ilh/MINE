
# Module 6 — Decision Boundaries & Thresholds

## 1. WHY: Converting Probabilities to Action

Logistic regression outputs a probability: $p = P(Y=1 \vert{} \mathbf{x}) \in [0, 1]$. However, production systems require discrete decisions (e.g., *Flag for fraud* vs. *Approve*, *Send email* vs. *Do not send*).

```
Model Output (Probability)  ──────►  Decision Rule (Threshold τ)  ──────►  Action
       p = 0.73                             If p ≥ 0.50                      CLASS 1

```

> **The Default 0.5 Trap:**
> The standard threshold $\tau = 0.50$ assumes that **False Positives (FP)** and **False Negatives (FN)** carry identical business costs. In real-world applications, error costs are rarely equal. Blindly defaulting to $\tau = 0.50$ without evaluating domain cost structures leads to sub-optimal business performance.

---

## 2. INTUITION: Threshold vs. Decision Boundary

The **Threshold** ($\tau$) and the **Decision Boundary** are two representations of the same decision mechanism:

| Concept | Domain | Definition |
| --- | --- | --- |
| **Threshold ($\tau$)** | Probability Space ($p \in [0, 1]$) | The numerical cutoff value above which an instance is classified as Positive. |
| **Decision Boundary** | Feature Space ($\mathbf{x} \in \mathbb{R}^d$) | The geometric surface where $P(Y=1 \vert{} \mathbf{x}) = \tau$. |

```
                       PROBABILITY SPACE (Sigmoid Curve)
                       
       1.0 ┼─────────────────────────────────● (Class 1)
           │                             ●●
       p   │                          ●●
       0.5 ┼───────────────────────● (Threshold = 0.5)
           │                    ●●
           │                 ●●
       0.0 ┼──────●───────────┴───────────────────────
           └──────┴───────────┼──────────────────────► z
                            z = 0
                            
                       FEATURE SPACE (Linear Cut)
                            
       x₂ ▲
          │          ● (Class 1)     ●
          │     ●           ●
          │───────────╲────────────────────── Decision Boundary
          │  ○         ╲        ●              b + w₁x₁ + w₂x₂ = 0
          │       ○     ╲   ●
          │   ○          ╲
          └───────────────┴──────────────────► x₁
             (Class 0)

```

---

## 3. MATHEMATICAL DERIVATION

### Step 1: Logistic Regression Output

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z$ is the log-odds (linear combination of features and weights):

$$z = b + \mathbf{w}^T \mathbf{x} = b + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$

---

### Step 2: Decision Rule with Default Threshold ($\tau = 0.5$)

$$\text{Predict } \hat{y} = \begin{cases} 1 & \text{if } p \ge 0.5 \\ 0 & \text{if } p < 0.5 \end{cases}$$

---

### Step 3: Solving for the Boundary in $z$-Space

To find the decision boundary, set $p = 0.5$:

$$\frac{1}{1 + e^{-z}} = 0.5 \implies 1 + e^{-z} = 2 \implies e^{-z} = 1 \implies z = 0$$

> **Core Insight:**
> In logistic regression with a default threshold of $\tau = 0.5$, the decision boundary corresponds precisely to the hyperplane where $z = 0$:
> $$b + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = 0$$
> 
> 

---

### Step 4: Decision Boundary Geometry in 2D

Rearranging $b + w_1 x_1 + w_2 x_2 = 0$ into slope-intercept form ($x_2 = m x_1 + c$):

$$x_2 = \left( -\frac{w_1}{w_2} \right) x_1 + \left( -\frac{b}{w_2} \right)$$

* **Slope ($m$):** $-\frac{w_1}{w_2}$
* **$x_2$-intercept ($c$):** $-\frac{b}{w_2}$

Because $z$ is a linear function of $\mathbf{x}$, the decision boundary of a standard logistic regression model is **always a linear hyperplane**.

---

### Step 5: Arbitrary Threshold Adjustments ($\tau \ne 0.5$)

For any general threshold $\tau \in (0, 1)$:

$$\sigma(z) = \tau \implies z = \ln \left( \frac{\tau}{1 - \tau} \right)$$

The decision boundary equation shifts parallel in feature space:

$$b + \mathbf{w}^T \mathbf{x} = \ln \left( \frac{\tau}{1 - \tau} \right)$$

* Increasing threshold ($\tau > 0.5 \implies z > 0$): Shifts boundary toward the Positive class region (makes predicting $Y=1$ more strict).
* Decreasing threshold ($\tau < 0.5 \implies z < 0$): Shifts boundary toward the Negative class region (makes predicting $Y=1$ more lenient).

---

## 4. WORKED NUMERIC EXAMPLE

**Scenario:** A customer churn model with two features:

* $x_1$: Months since last purchase
* $x_2$: Number of customer service calls
* Weights: $b = -2.5$, $w_1 = 0.3$, $w_2 = 0.8$

**Boundary Equation at $\tau = 0.5$ ($z = 0$):**

$$-2.5 + 0.3 x_1 + 0.8 x_2 = 0 \implies x_2 = \frac{2.5 - 0.3 x_1}{0.8}$$

```
Customer A: x₁ = 2, x₂ = 1
  z = -2.5 + 0.3(2) + 0.8(1) = -2.5 + 0.6 + 0.8 = -1.1
  p = σ(-1.1) ≈ 0.250  ──►  Predict No Churn (0)

Customer B: x₁ = 8, x₂ = 4
  z = -2.5 + 0.3(8) + 0.8(4) = -2.5 + 2.4 + 3.2 = 3.1
  p = σ(3.1) ≈ 0.957   ──►  Predict Churn (1)

```

---

## 5. BUSINESS-COST THRESHOLD OPTIMIZATION

Choosing a threshold is a business decision dictated by the asymmetry of error costs:

$$\text{Expected Cost} = (\text{Cost}_{\text{FP}} \times \text{FP}) + (\text{Cost}_{\text{FN}} \times \text{FN})$$

| Domain | High-Cost Error | Optimization Strategy | Threshold Adjustment |
| --- | --- | --- | --- |
| **Fraud Detection** | **False Negative** (Missed fraud = direct financial loss) | Minimize FN (maximize Recall) | **Lower Threshold** ($\tau \downarrow \approx 0.1 - 0.3$) |
| **Spam Filtering** | **False Positive** (Legitimate email sent to Spam) | Minimize FP (maximize Precision) | **Raise Threshold** ($\tau \uparrow \approx 0.7 - 0.9$) |
| **Cancer Screening** | **False Negative** (Undetected disease = risk to life) | Minimize FN (maximize Recall) | **Lower Threshold** ($\tau \downarrow \approx 0.05 - 0.2$) |

---

## 6. FAANG L5 INTERVIEW CHEAT SHEET

### Q1: "Is the decision boundary of logistic regression always linear?"

> *"Yes. The decision boundary is defined by $\sigma(z) = \tau$, which simplifies to $b + \mathbf{w}^T \mathbf{x} = C$ (where $C = \ln(\frac{\tau}{1-\tau})$). Since this is a degree-1 linear equation in feature space, the boundary is always a linear hyperplane. To capture non-linear decision boundaries with logistic regression, explicit non-linear feature engineering (e.g., polynomial features or interaction terms) must be applied beforehand."*

### Q2: "What happens to the decision boundary if you multiply all weights and the bias by a constant $k > 1$?"

> *"The geometric decision boundary at $\tau = 0.5$ remains unchanged, because $k(b + \mathbf{w}^T \mathbf{x}) = 0$ shares the exact same roots as $b + \mathbf{w}^T \mathbf{x} = 0$. However, the predicted probabilities will saturate faster toward 0 and 1, making the sigmoid transition steeper around the boundary."*

### Q3: "How do you select an optimal operational threshold in production?"

> *"Rather than relying on accuracy at default $\tau = 0.5$, we construct a cost function mapping the monetary or operational penalty of False Positives versus False Negatives. We sweep thresholds across the validation set to evaluate Precision-Recall curves or total cost curves, selecting the threshold that minimizes expected business cost."*

---

## 7. INTERACTIVE DECISION BOUNDARY & THRESHOLD SIMULATOR

Use the widget below to adjust feature weights ($w_1, w_2$), bias ($b$), and decision threshold ($\tau$) to visualize how the decision boundary shifts in feature space.

---

## 8. PYTHON IMPLEMENTATION

```python
import numpy as np

# Sample customer probabilities
probabilities = np.array([0.20, 0.40, 0.55, 0.65, 0.85])

# Threshold evaluation loop
thresholds = [0.3, 0.5, 0.7]

for tau in thresholds:
    # Compute z-score corresponding to threshold tau
    z_cutoff = np.log(tau / (1 - tau))
    
    # Apply decision rule
    decisions = (probabilities >= tau).astype(int)
    print(f"Threshold τ={tau:.1f} (z_cutoff={z_cutoff:+.3f}) -> Decisions: {decisions}")

```
