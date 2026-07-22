

# Module 7 — Regularization (L1/L2)

## 1. WHY

Imagine you train your churn model on 500 customers using 50 features (some genuinely predictive, others just noisy nonsense — like *"customer's favorite color"*). Without any safeguard, gradient descent will happily assign a large weight to **any** feature that happens to reduce training loss — even if that "pattern" is pure coincidence in your training data.

**What breaks without regularization:**

* The model **overfits** — memorizing quirks and noise specific to your sample (e.g., *"every customer named Dave happened to churn"*).
* It performs exceptionally well on training data but **falls apart on unseen validation data**.
* Regularization acts as a safeguard, actively discouraging the model from leaning too heavily on any single feature.

---

## 2. INTUITION

Think of training an unconstrained model like a student preparing for an exam where the only objective is *"get the highest score possible."* A student might resort to memorizing the exact answer key from a leaked copy of *one specific test* — total mastery of that single test, zero foundational understanding, and complete failure on any new exam.

> **Regularization adds a second objective:**
> *"Achieve a high test score, AND keep your total effort / model complexity minimal."*

It penalizes large weights directly during training: **"Every time you increase a weight, remember that large weights come with a cost — only do it if the improvement in predictions is worth the penalty."**

---

## 3. FORMULAS & MECHANICS

In Module 4, optimization focused strictly on minimizing the log-loss cost function $\mathcal{L}(\mathbf{w})$. Regularization adds an explicit penalty term based on weight magnitude:

$$\text{New Cost} = \text{Original Loss} + \left( \lambda \times \text{Weight Penalty} \right)$$

$$\mathcal{J}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \cdot \Omega(\mathbf{w})$$

Where:

* $\mathcal{L}(\mathbf{w})$ = Original log-loss (how wrong the predictions are).
* $\lambda$ ($\text{penalty\_strength}$) = Hyperparameter controlling penalty weight. Larger $\lambda \implies$ simpler model.
* $\Omega(\mathbf{w})$ = Function measuring total weight magnitude.

---

### L2 Regularization (Ridge)

> **Formula:** Sum of squared weights.

$$\Omega_{\text{L2}}(\mathbf{w}) = \Vert{}\mathbf{w}\Vert{}_2^2 = \sum_{j=1}^{p} w_j^2$$

$$\mathcal{J}_{\text{L2}}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \sum_{j=1}^{p} w_j^2$$

* **Effect:** Large weights get punished quadratically.
* **Result:** **Shrinkage** — pushes all weights to become smaller and smoothly distributed, but rarely forces a weight to *exactly* zero.

---

### L1 Regularization (Lasso)

> **Formula:** Sum of absolute weight magnitudes.

$$\Omega_{\text{L1}}(\mathbf{w}) = \Vert{}\mathbf{w}\Vert{}_1 = \sum_{j=1}^{p} \vert{}w_j\vert{}$$

$$\mathcal{J}_{\text{L1}}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \sum_{j=1}^{p} \vert{}w_j\vert{}$$

* **Effect:** Penalizes all non-zero weights at a constant rate, regardless of size.
* **Result:** **Sparsity / Feature Selection** — drives weak or redundant feature weights all the way to **exactly zero**, effectively removing them from the model.

---

## 4. WORKED NUMERIC EXAMPLE

Consider a model with 3 weights: $w_1 = 2.0$, $w_2 = 0.1$, $w_3 = -3.0$, an original log-loss $\mathcal{L} = 0.40$, and $\lambda = 0.1$.

### L2 Penalty Calculation

$$\Omega_{\text{L2}} = (2.0)^2 + (0.1)^2 + (-3.0)^2 = 4.0 + 0.01 + 9.0 = 13.01$$

$$\text{Cost}_{\text{L2}} = 0.40 + (0.1 \times 13.01) = 0.40 + 1.301 = 1.701$$

### L1 Penalty Calculation

$$\Omega_{\text{L1}} = \vert{}2.0\vert{} + \vert{}0.1\vert{} + \vert{}-3.0\vert{} = 2.0 + 0.1 + 3.0 = 5.10$$

$$\text{Cost}_{\text{L1}} = 0.40 + (0.1 \times 5.10) = 0.40 + 0.510 = 0.910$$

---

### Why L1 Drives Weights to Exactly Zero

Look at the derivative penalty contribution for a tiny weight like $w_2 = 0.1$:

* **Under L2:** The derivative of $w_2^2$ is $2 \cdot w_2 = 0.2$. As $w_2 \to 0$, the gradient update drops to near zero. **There is almost no pressure to shrink a tiny L2 weight further.**
* **Under L1:** The derivative of $\vert{}w_2\vert{}$ is $\text{sign}(w_2) = +1.0$. The penalty pressure remains constant ($1.0 \cdot \lambda$) regardless of how small $w_2$ gets. **If $w_2$ isn't actively reducing log-loss, L1 pushes it all the way to $0$.**

---

## 5. WHEN TO USE WHICH

| Scenario / Problem State | Preferred Regularization | Underlying Reason |
| --- | --- | --- |
| **High proportion of noisy or irrelevant features** | **L1 (Lasso)** | Zeroes out useless features automatically (built-in feature selection). |
| **Multicollinearity / Correlated features** | **L2 (Ridge)** or **Elastic Net** | L1 picks one correlated feature arbitrarily and zeroes the rest; L2 splits weight evenly across them. |
| **Desire for high interpretability / small model footprint** | **L1 (Lasso)** | Produces a sparse model with fewer active parameters. |
| **General variance reduction without feature dropping** | **L2 (Ridge)** | Smoothly shrinks all parameters while keeping all features in play. |
| **Need both feature selection and collinearity stability** | **Elastic Net** | Combines L1 and L2 penalties: $\lambda_1 \Vert{}\mathbf{w}\Vert{}_1 + \lambda_2 \Vert{}\mathbf{w}\Vert{}_2^2$. |

---

## 6. INTERPRETATION & PRACTICAL GUIDANCE

* Regularization directly controls the **bias-variance tradeoff**.
* **Zeroed-Out Weights (L1):** A zero weight means the feature failed to reduce loss enough to justify its parameter penalty. It can be safely removed from upstream data pipelines.
* **Overfitting Symptoms:** If training performance is high while validation performance lags, increase $\lambda$.
* **Underfitting Symptoms:** If both training and validation performance are poor, $\lambda$ is too high (crushing weights and over-simplifying the model).

---

## 7. FAANG L5 INTERVIEW CHEAT SHEET

### Q1: "What's the fundamental difference between L1 and L2 regularization?"

> *"Both add a weight penalty to the loss function to prevent overfitting. L2 (Ridge) penalizes squared weight magnitudes, resulting in smooth shrinkage toward zero. L1 (Lasso) penalizes absolute weight magnitudes, providing a constant push toward zero that forces uninformative feature weights to become strictly zero (sparsity)."*

### Q2: "Why does L1 produce sparse models while L2 does not?"

> *"Geometrically, L1 forms a diamond constraint boundary with sharp corners at the coordinate axes, making loss function contour intersections highly likely at points where weights equal zero. Mathematically, L1's derivative penalty rate is constant regardless of weight magnitude, whereas L2's derivative penalty shrinks linearly as weights approach zero."*

### Q3: "How do you determine the optimal penalty strength ($\lambda$)?"

> *"Through cross-validation across a grid of $\lambda$ values. We select the $\lambda$ that minimizes validation error (e.g., log-loss or PR-AUC), balancing variance reduction against underfitting risk."*

---

## 8. INTERACTIVE REGULARIZATION SIMULATOR

---

## 9. PYTHON IMPLEMENTATION

```python
import numpy as np

# Weights and setup
w = np.array([2.0, 0.1, -3.0])
original_log_loss = 0.40
lam = 0.1

# Penalties
l2_penalty = np.sum(w ** 2)
l1_penalty = np.sum(np.abs(w))

# Costs
new_cost_l2 = original_log_loss + lam * l2_penalty
new_cost_l1 = original_log_loss + lam * l1_penalty

print(f"L2 Penalty: {l2_penalty:.3f} | Total Cost (L2): {new_cost_l2:.3f}")
print(f"L1 Penalty: {l1_penalty:.3f} | Total Cost (L1): {new_cost_l1:.3f}")

```
