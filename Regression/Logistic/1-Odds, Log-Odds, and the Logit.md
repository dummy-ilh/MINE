# Module 1 — Odds, Log-Odds, and the Logit Link

## 1. WHY

In Module 0, we established that we cannot model probability directly using a straight line ($y = \mathbf{w}^T \mathbf{x} + b$) because probabilities are trapped inside $[0, 1]$, while linear equations naturally range across $(-\infty, +\infty)$.

Statisticians solved this with an elegant trick: **do not model probability directly. Model a transformation of probability that spans $(-\infty, +\infty)$, and then reverse the transformation at the very end.**

---

## 2. INTUITION — What Are "Odds"?

You are likely familiar with odds from sports betting: *"The odds of that team winning are 3 to 1."*

That means: **for every 1 time they lose, we expect them to win 3 times.** Winning is 3 times more likely than losing.

$$\text{Odds} = \frac{\text{Probability event happens}}{\text{Probability event does NOT happen}} = \frac{p}{1 - p}$$

Compare this to **probability**:

$$\text{Probability} = \frac{\text{Favorable outcomes}}{\text{Total possible outcomes}} = p$$

* **Probability** asks: *"Wins out of total races."* Bounds: $[0, 1]$
* **Odds** asks: *"Wins compared to losses."* Bounds: $[0, +\infty)$

By moving from probability to odds, we unlock the upper half of the number line — odds can shoot up to $+\infty$ when an event becomes nearly certain. However, odds still stop at $0$ when an event never happens. We are halfway to an unbounded scale.

---

## 3. FORMULA — Odds

$$\text{Odds}(p) = \frac{p}{1 - p}$$

Where:

* $p \in [0, 1]$ is the probability of the event occurring.
* $1 - p$ is the probability of the event **not** occurring.

---

## 4. WORKED NUMERIC EXAMPLE — Odds

If a customer has a probability of churn $p = 0.75$:

$$\text{Odds} = \frac{0.75}{1 - 0.75} = \frac{0.75}{0.25} = 3.0$$

In plain English: *"This customer is 3 times more likely to churn than to stay."*

| Probability ($p$) | $1 - p$ | $\text{Odds} = \frac{p}{1-p}$ | Plain English Interpretation |
| --- | --- | --- | --- |
| **0.50** | 0.50 | **1.0** | Equally likely to happen or not |
| **0.90** | 0.10 | **9.0** | 9x more likely to happen than not |
| **0.10** | 0.90 | **0.111** | 9x more likely **not** to happen |
| **0.99** | 0.01 | **99.0** | Extremely likely to happen |

---

## 5. LOG-ODDS (The "Logit")

To extend our domain down to $-\infty$, we apply the natural logarithm ($\ln$) to the odds:

$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right)$$

### Why the Logarithm Works:

* **Odds $> 1$** (more likely than not) become **positive numbers** under the log.
* **Odds $< 1$** (less likely than not) become **negative numbers** under the log.
* **Odds $= 1$** ($50/50$ chance) becomes **exactly $0$**.

This gives us a perfectly symmetric, completely unbounded scale from $-\infty$ to $+\infty$.

---

## 6. WORKED NUMERIC EXAMPLE — Log-Odds

Continuing our churn example where $p = 0.75$ and $\text{Odds} = 3.0$:

$$\text{logit}(0.75) = \ln(3.0) \approx 1.0986$$

Now test the complement: $p = 0.25$ (25% churn probability):

$$\text{Odds} = \frac{0.25}{0.75} = 0.3333 \implies \text{logit}(0.25) = \ln(0.3333) \approx -1.0986$$

### Full Transformation Mapping

| Probability ($p$) | Odds ($\frac{p}{1-p}$) | Logit ($\ln(\text{Odds})$) | State |
| --- | --- | --- | --- |
| **0.01** | 0.0101 | **-4.60** | Highly unlikely |
| **0.10** | 0.1111 | **-2.20** | Unlikely |
| **0.25** | 0.3333 | **-1.10** | Moderately unlikely |
| **0.50** | 1.0000 | **0.00** | $50/50$ decision boundary |
| **0.75** | 3.0000 | **+1.10** | Moderately likely |
| **0.90** | 9.0000 | **+2.20** | Likely |
| **0.99** | 99.0000 | **+4.60** | Highly likely |

```text
PROBABILITY WORLD           ODDS WORLD              LOG-ODDS (LOGIT) WORLD
    [0, 1]          --->     [0, +∞)       --->         (-∞, +∞)

```

---

## 7. WHAT IS A "LINK FUNCTION"?

In Generalized Linear Models (GLMs), a **link function** $g(\cdot)$ bridges the gap between the expected value of our bounded outcome and our linear combination of features:

$$g(p) = \mathbf{w}^T \mathbf{x} + b$$

For logistic regression, our link function is the **logit function**:

$$\text{logit}(p) = \ln\left(\frac{p}{1 - p}\right) = \mathbf{w}^T \mathbf{x} + b$$

Once we map probabilities into "straight-line world" via the logit link, standard linear operations work seamlessly without violating probability constraints.

---

## 8. MATHEMATICAL DEEP DIVE: Anti-Symmetry & Sigmoid Mirroring

### 1. Proof of Log-Odds Anti-Symmetry

Let us prove that $\text{logit}(1 - p) = -\text{logit}(p)$:

$$\text{logit}(1 - p) = \ln\left(\frac{1 - p}{p}\right) = \ln\left(\left(\frac{p}{1 - p}\right)^{-1}\right)$$

Using the natural logarithm identity $\ln(x^{-1}) = -\ln(x)$:

$$\text{logit}(1 - p) = -\ln\left(\frac{p}{1 - p}\right) = -\text{logit}(p) \quad \blacksquare$$

### 2. Sigmoid Symmetry Derivation

The **sigmoid function** $\sigma(z)$ is the inverse of the logit function, mapping linear outputs $z = \mathbf{w}^T \mathbf{x} + b$ back to probabilities:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Given $z = \text{logit}(p) \implies \sigma(z) = p$:

$$\sigma(-z) = \frac{1}{1 + e^z} = \frac{e^{-z}}{e^{-z}(1 + e^z)} = \frac{e^{-z}}{1 + e^{-z}} = \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$$

$$\sigma(-z) = 1 - \sigma(z)$$

This confirms that the sigmoid curve is point-symmetric around $(0, 0.5)$.

---

## 9. FAANG L5 ANGLE

### Common Interview Questions & High-Impact Answers

> **Q1: What is a link function, and why does logistic regression use the logit link?**
> * **Answer:** A link function transforms a bounded target space into an unbounded continuous space so it can be modeled via a linear combination of features. Logistic regression uses the logit link because it is the **natural (canonical) link function** for the Bernoulli/Binomial distribution in the Exponential Family. It directly maps log-odds to linear predictor space while ensuring natural mathematical properties (like concave cross-entropy loss optimization).
> 
> 

> **Q2: Why not use $\ln(p)$ directly as the link function instead of $\ln\left(\frac{p}{1-p}\right)$?**
> * **Answer:** $\ln(p)$ is only unbounded on one side. Since $p \le 1$, $\ln(p) \le 0$, ranging from $(-\infty, 0]$. This leaves the upper bound at $0$, meaning a large positive linear dot product $\mathbf{w}^T \mathbf{x} + b$ would still produce invalid probabilities. Taking the ratio $p / (1-p)$ first stretches the upper bound to $+\infty$, allowing $\ln$ to create a truly unbounded scale $(-\infty, +\infty)$.
> 
> 

---

## 10. PYTHON VERIFICATION

```python
import numpy as np

def odds(p: float | np.ndarray) -> float | np.ndarray:
    return p / (1 - p)

def logit(p: float | np.ndarray) -> float | np.ndarray:
    return np.log(odds(p))

def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-z))

probs = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
for p in probs:
    o = odds(p)
    l = logit(p)
    p_rec = sigmoid(l)
    print(f"p = {p:.2f} | Odds = {o:7.3f} | Logit (z) = {l:6.3f} | Sigmoid(z) = {p_rec:.2f}")

```

---

## 11. CONCEPT CHECK

1. **If a feature increases the odds of an event by a factor of 2.0 (Odds Ratio = 2.0), what does that mean?**
* *Answer:* The event is now twice as likely to happen relative to not happening compared to baseline.


2. **Why is raw odds alone insufficient as a target for linear regression?**
* *Answer:* Raw odds is bounded below by $0$. A standard linear regression line can still output negative values for small inputs, violating the domain of odds. We need the logarithm to map $[0, +\infty)$ to $(-\infty, +\infty)$.
