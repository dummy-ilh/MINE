
# Module 2 — The Sigmoid Function (The "Un-Translator")

## 1. WHY

In Module 1, we made a one-way trip: **Probability $\to$ Odds $\to$ Log-Odds (Logit)**. This allowed us to escape the bounded $[0, 1]$ probability space and land in "straight-line world" $(-\infty, +\infty)$, where a linear model $z = \mathbf{w}^T \mathbf{x} + b$ can predict freely without violating physical bounds.

However, once our linear model outputs a prediction, $z$ is a **log-odds value** (e.g., $-1.7$ or $3.2$). Business stakeholders and decision systems do not operate on log-odds; they require probabilities. Nobody wants to hear *"this user's churn log-odds is 1.1"*; they need *"this user has a 75% probability of churning."*

We need the **inverse mapping**: Log-Odds $\to$ Probability. Without this step, logistic regression outputs mathematically well-behaved numbers that are practically unusable.

---

## 2. INTUITION

Recall Module 1's framework:

```text
PROBABILITY WORLD  --[ link function: logit ]-->  STRAIGHT-LINE WORLD
     [0, 1]                                           (-∞, +∞)

```

The **sigmoid function ($\sigma$) is the inverse mapping** — the un-translator:

```text
STRAIGHT-LINE WORLD  --[ activation: sigmoid ]-->  PROBABILITY WORLD
     (-∞, +∞)                                          [0, 1]

```

Imagine log-odds as a position on an infinite real number line. The sigmoid function **smoothly compresses that infinite line into a bounded interval between 0 and 1**:

* **Large positive input ($z \to +\infty$):** Asymptotically approaches **$1.0$** (near-certain positive event).
* **Large negative input ($z \to -\infty$):** Asymptotically approaches **$0.0$** (near-certain negative event).
* **Zero input ($z = 0$):** Yields exactly **$0.5$** (complete maximum uncertainty / $50/50$ toss-up).

---

## 3. FORMULA

To find the inverse of $\text{logit}(p) = z$:

$$z = \ln\left(\frac{p}{1 - p}\right) \implies e^z = \frac{p}{1 - p} \implies p = e^z(1 - p) \implies p(1 + e^z) = e^z$$

$$p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}}$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:

* $z = \mathbf{w}^T \mathbf{x} + b$ is the raw linear output (also referred to as the **logit score**).
* $e \approx 2.71828$ (Euler's number).
* $\sigma(z) \in (0, 1)$ is the predicted probability $\hat{y} = P(Y=1 \mid \mathbf{x})$.

---

## 4. WORKED NUMERIC EXAMPLES

Let's compute probabilities for several raw $z$ scores:

### Example A: $z = 1.0986$ (From Module 1 where $p = 0.75$)

$$\sigma(1.0986) = \frac{1}{1 + e^{-1.0986}} = \frac{1}{1 + 0.3333} = \frac{1}{1.3333} = 0.75$$


*Logit and Sigmoid are exact mathematical inverses.*

### Example B: $z = 0.0$ (Boundary threshold)

$$\sigma(0) = \frac{1}{1 + e^0} = \frac{1}{1 + 1} = 0.50$$

### Example C: $z = 5.0$ (High confidence positive)

$$\sigma(5) = \frac{1}{1 + e^{-5}} = \frac{1}{1 + 0.0067} = 0.9933$$

### Example D: $z = -5.0$ (High confidence negative)

$$\sigma(-5) = \frac{1}{1 + e^5} = \frac{1}{1 + 148.41} = 0.0067$$

| Raw Logit ($z$) | Sigmoid $\sigma(z) = \hat{y}$ | Interpretation |
| --- | --- | --- |
| **$-5.00$** | **0.0067** | Highly likely Class 0 |
| **$-1.10$** | **0.2500** | Likely Class 0 |
| **$0.00$** | **0.5000** | Maximum Uncertainty ($50/50$) |
| **$+1.10$** | **0.7500** | Likely Class 1 |
| **$+5.00$** | **0.9933** | Highly likely Class 1 |

---

## 5. WHY THE "S" SHAPE MATTERS

Plotted against $z$, $\sigma(z)$ produces a characteristic S-shaped curve (sigmoid):

* **Center Region ($z \approx 0$):** High slope (gradient). Small changes in input features yield substantial shifts in output probability. This represents the **decision boundary boundary zone** where the model is uncertain.
* **Saturated Regions ($\vert{}z\vert{} \gg 0$):** Asymptotically flat (gradient approaches $0$). Large variations in $z$ barely change $\hat{y}$ because the model is already highly confident.

### Real-World Realism vs. Linear Models

Linear regression assumes a constant rate of change across the entire real line. Sigmoid models **diminishing marginal returns**: once a user is already at a 99.9% probability of churning, an additional complaint should not push confidence past 100%; it yields diminishing increases in risk score.

---

## 6. MATHEMATICAL DERIVATION: Derivative of the Sigmoid

In optimization (gradient descent), we frequently need $\frac{d\sigma}{dz}$. A unique property of the sigmoid function is that **its derivative can be expressed entirely as a function of its own output $p = \sigma(z)$**.

### Proof:

$$\sigma(z) = (1 + e^{-z})^{-1}$$

Apply the chain rule:

$$\frac{d}{dz}\sigma(z) = -1 \cdot (1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Decompose the fraction:

$$\frac{d}{dz}\sigma(z) = \left(\frac{1}{1 + e^{-z}}\right) \cdot \left(\frac{e^{-z}}{1 + e^{-z}}\right)$$

Notice that $\frac{e^{-z}}{1 + e^{-z}} = \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$.

Therefore:

$$\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z)) = p(1 - p)$$

### Numerical Interpretation of the Gradient:

* **At $z = 0$ ($p = 0.5$):** Gradient is maximized at $0.5 \times (1 - 0.5) = \mathbf{0.25}$.
* **At $z = 5$ ($p = 0.9933$):** Gradient shrinks to $0.9933 \times (1 - 0.9933) \approx \mathbf{0.0067}$ (**vanishing gradient region**).

---

## 7. DEEP INTUITION: "If $z \to \text{Sigmoid} \to p$, Why Study Link Functions at All?"

Mechanically, forward pass execution is just:

$$p = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

Why maintain the conceptual machinery of link functions?

1. **Coefficient Interpretability:** Linear coefficients $\beta_j$ operate linearly **only** in log-odds space ($\Delta \text{logit} = \beta_j$). They do not have a constant additive effect on probability space due to the non-linear S-curve. To state *"a 1-unit increase in feature $X$ multiplies odds by $e^{\beta_j}$"*, you must reason via log-odds.
2. **Canonical GLM Justification:** The sigmoid function is not an arbitrary squashing heuristic; it is the mathematical inverse of the **canonical link function** derived from the Bernoulli distribution's exponential family formulation.
3. **Generalization across ML Architecture:** Understanding the link function framework allows you to generalize from binary classification (Bernoulli $\to$ Logit $\to$ Sigmoid) to multi-class classification (Categorical $\to$ Log-partition $\to$ Softmax) and count modeling (Poisson $\to$ Log Link $\to$ Exponential).

---

## 8. CONNECTION TO NEURAL NETWORKS

A single artificial neuron computes a dot product followed by a non-linear activation function:

$$a = f(\mathbf{w}^T \mathbf{x} + b)$$

When $f(z) = \sigma(z)$, **a single neuron is identical to a Logistic Regression model**. Logistic Regression is functionally a single-layer Multi-Layer Perceptron (MLP) with a sigmoid output node.

---

## 9. FAANG L5 ANGLE

> **Q1: What is the derivative of the sigmoid function, and what issue does it cause in deep neural networks?**
> * **Answer:** The derivative is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. Because its maximum value is $0.25$ (at $z=0$) and approaches $0$ as $\vert{}z\vert{} \to \infty$, chaining multiple sigmoid layers in deep networks causes gradients to decay exponentially during backpropagation ($\prod \sigma'(z_l) \le 0.25^L$). This leads to the **vanishing gradient problem**, which is why hidden layers typically use activations like ReLU ($\text{derivative} = 1$ for $z > 0$) instead of Sigmoid.
> 
> 

> **Q2: What is the relationship between Sigmoid and Softmax?**
> * **Answer:** Sigmoid is a special case of Softmax for binary outcomes $K=2$. For two logits $z_1, z_2$, Softmax computes $P(Y=1) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \sigma(z_1 - z_2)$.
> 
> 

---

## 10. PYTHON VERIFICATION

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    p = sigmoid(z)
    return p * (1 - p)

z_vals = np.array([-5.0, -1.0986, 0.0, 1.0986, 5.0])
for z in z_vals:
    p = sigmoid(z)
    dp_dz = sigmoid_derivative(z)
    print(f"z = {z:7.4f} | Prob p = {p:.4f} | Derivative dp/dz = {dp_dz:.4f}")

```

---

## 11. CONCEPT CHECK ANSWERS

1. **Why does the sigmoid curve flatten near the extremes, and why is this desirable?**
* *Answer:* The flattening enforces asymptotic probability bounds $[0, 1]$ and models diminishing returns. Once confidence is near $100\%$ or $0\%$, additional feature magnitudes produce diminishing shifts in probability, reflecting real-world saturation.


2. **If $z = 0$, what is the output probability and model confidence?**
* *Answer:* $\sigma(0) = 0.5$. This represents maximum uncertainty ($50/50$ chance), sitting directly on the decision boundary.
