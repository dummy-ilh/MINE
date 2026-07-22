
# Module 9 — Multiclass Extension (Softmax)

## 1. WHY: Beyond Binary Classification

Standard logistic regression models binary outcomes ($Y \in \{0, 1\}$). However, real-world systems frequently operate across $K > 2$ discrete categories (e.g., *Document Classification*: `[Finance, HR, Engineering, Legal]`, *Medical Triage*: `[Low, Medium, High, Critical]`).

```
      BINARY LOGISTIC REGRESSION                   MULTICLASS (SOFTMAX)
┌─────────────────────────────────┐       ┌──────────────────────────────────┐
│ Single log-odds score z         │       │ K separate raw scores z_k        │
│ Single sigmoid output σ(z)      │  ──►  │ K-dimensional vector z_1 ... z_K │
│ Output: P(Y=1) ∈ [0, 1]         │       │ Normalized distribution Σ P_k=1  │
└─────────────────────────────────┘       └──────────────────────────────────┘

```

> **Why Sigmoid Fails for $K > 2$:**
> A single sigmoid output produces a single probability $p \in [0, 1]$, representing $P(Y=1)$. Extending this to $K$ classes requires predicting a probability vector $\mathbf{p} = [p_1, p_2, \dots, p_K]^T$ such that every $p_k \ge 0$ and the sum of all elements equals $1$:
> $$\sum_{k=1}^{K} p_k = 1$$
> 
> 

---

## 2. INTUITION: One-vs-Rest (OvR) vs. Softmax Regression

To generalize binary classification to $K$ classes, two primary architectural patterns exist:

```
                  1. ONE-VS-REST (OvR)                     2. SOFTMAX REGRESSION
            
               ┌──► Model 1 (Billing vs Rest) ──► p₁ ┐        ┌─────────────────────────┐
               │                                     │        │ Compute z_k for all K   │
   Input x ────┼──► Model 2 (Tech vs Rest)    ──► p₂ ┼──►     │ Apply Softmax Jointly   │
               │                                     │        │ Ensures Σ p_k = 1.0     │
               └──► Model 3 (Account vs Rest) ──► p₃ ┘        └───────────┬─────────────┘
                                                                          │
                   Uncoordinated Outputs (Σ p_k ≠ 1.0)           Valid Probability Vector

```

### Comparison

| Approach | Architecture | Output Property | Primary Disadvantage |
| --- | --- | --- | --- |
| **One-vs-Rest (OvR)** | $K$ independent binary models trained separately. | Uncalibrated scores ($p_1 + p_2 + \dots + p_K \neq 1.0$). | Scores are uncoordinated; independent models can yield total probabilities $> 1.0$ or $< 1.0$. |
| **Softmax Regression** | Single unified model producing $K$ linear logits jointly. | Coherent probability distribution ($\sum_{k=1}^K p_k = 1.0$). | Requires joint training across all class weight vectors simultaneously. |

---

## 3. FORMULA: The Softmax Function

Given $K$ classes, the model computes a raw linear logit $z_k$ for each class $k \in \{1, 2, \dots, K\}$ using class-specific weight vectors $\mathbf{w}_k$ and biases $b_k$:

$$z_k = \mathbf{w}_k^T \mathbf{x} + b_k$$

### The Softmax Transformation

To map the vector of raw logits $\mathbf{z} = [z_1, z_2, \dots, z_K]^T$ to a valid probability distribution $\mathbf{p}$, we apply the **Softmax function**:

$$P(Y = k \mid \mathbf{x}) = p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

### Mathematical Properties:

1. **Positivity Guarantee:** For any real-valued score $z_k \in (-\infty, \infty)$, $e^{z_k} > 0$.
2. **Normalization:** Dividing each exponentiated score by the denominator $\sum_{j=1}^K e^{z_j}$ enforces $\sum_{k=1}^K p_k = 1.0$.
3. **Monotonicity:** Larger raw logits $z_k$ strictly correspond to higher predicted probabilities $p_k$.

---

## 4. WORKED NUMERIC EXAMPLE

Consider a support-ticket classifier evaluating $K = 3$ categories: `[1: Billing, 2: Technical, 3: Account]`.

Given raw linear logits:

* $z_1 (\text{Billing}) = 1.2$
* $z_2 (\text{Technical}) = 0.5$
* $z_3 (\text{Account}) = -0.8$

---

### Step 1: Exponentiate Logits

$$e^{z_1} = e^{1.2} \approx 3.320$$

$$e^{z_2} = e^{0.5} \approx 1.649$$

$$e^{z_3} = e^{-0.8} \approx 0.449$$

---

### Step 2: Compute Normalization Term (Denominator)

$$\sum_{j=1}^{3} e^{z_j} = 3.320 + 1.649 + 0.449 = 5.418$$

---

### Step 3: Compute Final Class Probabilities

$$p_1 (\text{Billing}) = \frac{3.320}{5.418} \approx 0.6129 \quad (61.3\%)$$

$$p_2 (\text{Technical}) = \frac{1.649}{5.418} \approx 0.3044 \quad (30.4\%)$$

$$p_3 (\text{Account}) = \frac{0.449}{5.418} \approx 0.0827 \quad (8.3\%)$$

$$\text{Check: } 0.6129 + 0.3044 + 0.0827 = 1.0000 \quad \checkmark$$

---

## 5. PROOF: Sigmoid is a Special Case of Softmax ($K=2$)

For $K=2$ classes (`[Class 1, Class 2]`), evaluating Softmax for Class 1 yields:

$$P(Y=1 \mid \mathbf{x}) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}}$$

Divide numerator and denominator by $e^{z_1}$:

$$P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + \frac{e^{z_2}}{e^{z_1}}} = \frac{1}{1 + e^{-(z_1 - z_2)}}$$

Setting $\Delta z = z_1 - z_2$:

$$P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-\Delta z}} = \sigma(\Delta z)$$

> **Conclusion:**
> The binary Sigmoid function is mathematically identical to a 2-class Softmax function evaluated on the relative logit difference $\Delta z = z_1 - z_2$.

---

## 6. CATEGORICAL CROSS-ENTROPY LOSS

To train a Softmax model, we generalize binary log-loss to **Categorical Cross-Entropy Loss**. Given a one-hot encoded ground truth target vector $\mathbf{y} = [y_1, y_2, \dots, y_K]^T$ where $y_c = 1$ for the true class $c$ and $0$ elsewhere:

$$\mathcal{L}_{\text{CE}}(\mathbf{y}, \mathbf{p}) = -\sum_{k=1}^{K} y_k \ln(p_k) = -\ln(p_c)$$

> **Optimization Goal:**
> Minimizing cross-entropy loss directly maximizes the log-probability assigned to the correct class label $c$.

---

## 7. FAANG L5 INTERVIEW CHEAT SHEET

### Q1: "Why can't we normalize raw logits directly without exponentiating (i.e., $p_k = \frac{z_k}{\sum z_j}$)?"

> *"Raw logits $z_k$ can be negative ($z_k \in (-\infty, \infty)$). Direct linear normalization on negative values yields invalid negative probabilities. Furthermore, if the sum of logits equals zero ($\sum z_j = 0$), division by zero occurs. Exponentiation guarantees non-negative terms ($e^{z_k} > 0$), producing a valid, differentiable probability distribution."*

### Q2: "What is Softmax Numerical Instability (Overflow/Underflow), and how is it resolved in production?"

> *"Extremely large logits cause $e^{z_k}$ to overflow numerically in floating-point arithmetic (e.g., $e^{1000} \to \infty$). To make Softmax numerically stable, we subtract the maximum logit $C = \max(\mathbf{z})$ from all elements before exponentiating:*
> $$\text{Softmax}(\mathbf{z})_k = \frac{e^{z_k - C}}{\sum_{j=1}^K e^{z_j - C}}$$
> 
> 
> *This identity holds algebraically because $\frac{e^{z_k - C}}{\sum e^{z_j - C}} = \frac{e^{z_k} \cdot e^{-C}}{\sum e^{z_j} \cdot e^{-C}} = \frac{e^{z_k}}{\sum e^{z_j}}$."*

### Q3: "What is the difference between Softmax and Multi-Label Binary Classification?"

> *"Softmax assumes mutually exclusive classes ($\sum p_k = 1.0$; instance belongs to exactly one class). For multi-label classification (e.g., tagging an image as both `[Outdoor, Dog]`), we apply $K$ independent Sigmoid activation functions ($\sigma(z_k)$) instead of Softmax, allowing multiple classes to simultaneously have high predicted probabilities."*

---

## 8. INTERACTIVE SOFTMAX PROBABILITY SIMULATOR

Use the widget below to adjust 3 raw class logits ($z_1, z_2, z_3$) and observe real-time probability outputs and cross-entropy loss changes.

---

## 9. PYTHON IMPLEMENTATION

```python
import numpy as np

def softmax(z):
    # Subtract max logit for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def categorical_cross_entropy(probs, true_class_index):
    # Clip probabilities to prevent log(0) undefined errors
    p = np.clip(probs[true_class_index], 1e-15, 1 - 1e-15)
    return -np.log(p)

# Raw linear logits
logits = np.array([1.2, 0.5, -0.8])
true_class = 0  # Class 1 (Billing)

probs = softmax(logits)
loss = categorical_cross_entropy(probs, true_class)

print(f"Probabilities: {probs.round(4)}")
print(f"Sum Check:     {np.sum(probs):.4f}")
print(f"CE Loss:       {loss:.4f}")

```
