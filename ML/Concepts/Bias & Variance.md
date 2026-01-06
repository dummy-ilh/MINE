
# Bias–Variance Tradeoff (Exhaustive Master Note)

---

## 1. Formal Problem Setup (Statistical Learning Theory)

We observe data:
\[
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n
\]

Assume the data-generating process:
\[
y = f(x) + \epsilon
\]

Where:
- \( f(x) \): true (unknown) target function  
- \( \epsilon \): irreducible noise, with  
  \[
  \mathbb{E}[\epsilon] = 0,\quad \mathrm{Var}(\epsilon) = \sigma^2
  \]

Our model produces a predictor:
\[
\hat{f}(x; \mathcal{D})
\]

which depends on the **training dataset**.

---

## 2. What Bias and Variance Actually Measure

Bias and variance are defined **with respect to repeated sampling of datasets**.

### Expected Predictor
\[
\bar{f}(x) = \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]
\]

---

### Bias
\[
\text{Bias}(x) = \bar{f}(x) - f(x)
\]

**Interpretation**:
- Systematic error
- Error from incorrect assumptions
- Inability to represent the true function

---

### Variance
\[
\text{Var}(x) = \mathbb{E}_{\mathcal{D}}[(\hat{f}(x) - \bar{f}(x))^2]
\]

**Interpretation**:
- Sensitivity to data fluctuations
- Model instability

---

## 3. Bias–Variance Decomposition (Core Result)

For squared error loss:

\[
\mathbb{E}[(y - \hat{f}(x))^2] =
\underbrace{\text{Bias}^2}_{\text{approximation error}}
+
\underbrace{\text{Variance}}_{\text{estimation error}}
+
\underbrace{\sigma^2}_{\text{irreducible noise}}
\]

This decomposition holds **only for MSE**.

---

## 4. Irreducible Error (Often Ignored)

\[
\sigma^2 = \mathrm{Var}(\epsilon)
\]

- Comes from measurement noise, randomness, unobserved variables
- Cannot be reduced by **any** model
- Sets a **lower bound** on achievable error

---

## 5. Intuition via Repeated Training

Imagine:
- Fix a point \( x_0 \)
- Train the model on many datasets
- Plot predictions at \( x_0 \)

```text
Low Bias, High Variance:     High Bias, Low Variance:

 |   |                       |
 | o |                       |  o o o
 |o o|                       |   o
 | o |                       |
````

---

## 6. Model Complexity Perspective

| Model                 | Bias     | Variance |
| --------------------- | -------- | -------- |
| Linear regression     | High     | Low      |
| Polynomial (degree ↑) | ↓        | ↑        |
| k-NN (k ↑)            | ↑        | ↓        |
| Decision tree (deep)  | Low      | High     |
| Random forest         | Low      | Medium   |
| Boosting              | Very low | Medium   |

---

## 7. Bias–Variance Curve (Classic Visualization)

```text
Error
 ^
 | \        Variance
 |  \      /
 |   \    /
 |    \__/____  Total Error
 |     /\ 
 |    /  \ Bias
 |___/__________→ Model Complexity
```

* Increasing complexity ↓ bias but ↑ variance
* Optimal point minimizes **total error**

---

## 8. Algorithm-Specific Analysis

### Linear Regression

* High bias if true function is nonlinear
* Low variance due to closed-form solution

### k-NN

* k = 1 → low bias, high variance
* k → n → high bias, low variance

### Decision Trees

* Fully grown trees: near-zero bias, very high variance
* Pruning trades variance for bias

---

## 9. Regularization as Bias Injection

### Ridge Regression

[
\min_w |y - Xw|^2 + \lambda |w|^2
]

* Increases bias
* Reduces variance
* Improves generalization

### Lasso

* Same tradeoff + feature selection

---

## 10. Ensembles Through Bias–Variance Lens

### Bagging

* Reduces variance
* Does not reduce bias

### Random Forest

* Bagging + feature randomness
* Strong variance reduction

### Boosting

* Sequentially reduces bias
* Can increase variance if overfit

---

## 11. Bias–Variance vs Underfitting / Overfitting

| Concept      | Meaning       |
| ------------ | ------------- |
| Underfitting | High bias     |
| Overfitting  | High variance |
| Good fit     | Balanced      |

---

## 12. Loss-Function Dependence (Critical)

Bias–variance decomposition:

* ✅ Exists for **squared loss**
* ❌ Does NOT strictly exist for:

  * 0–1 loss
  * Log loss

This is why classification bias–variance is more nuanced.

---

## 13. Classification View (Advanced)

For classifiers:

* Bias relates to **systematic decision boundary error**
* Variance relates to **boundary instability**

No clean scalar decomposition — only conceptual.

---

## 14. Connection to VC Dimension & Capacity

* High-capacity models → low bias, high variance
* VC dimension controls generalization bounds

[
\text{Generalization Error} \le \text{Training Error} + \mathcal{O}\left(\sqrt{\frac{VC}{n}}\right)
]

---

## 15. Bias–Variance vs Data Size

* Small data → variance dominates
* Large data → bias dominates

Hence:

> With enough data, even complex models generalize.

---

## 16. Practical Diagnostics (FAANG-Style)

| Observation                      | Diagnosis     |
| -------------------------------- | ------------- |
| Train error high, val error high | High bias     |
| Train error low, val error high  | High variance |
| Both low                         | Good fit      |

---

## 17. How to Fix High Bias

* Increase model complexity
* Add nonlinear features
* Reduce regularization
* Switch algorithms

---

## 18. How to Fix High Variance

* More data
* Regularization
* Simpler model
* Bagging / ensembles
* Early stopping

---

## 19. Bias–Variance in Deep Learning

* Deep nets: **low bias**
* Regularization controls variance:

  * Dropout
  * Data augmentation
  * Weight decay
* Double descent phenomenon violates classical curve

---

## 20. Interview Questions That Matter

### Q1: Can you reduce both bias and variance?

**Answer:**
Yes, by increasing data or using better inductive bias.

---

### Q2: Why does bagging help trees?

**Answer:**
Trees are high-variance, low-bias learners.

---

### Q3: Why does boosting reduce bias?

**Answer:**
Sequentially corrects systematic errors.

---

## 21. What FAANG Actually Evaluates

* Can you **diagnose** bias vs variance from curves?
* Can you **act** (not just define)?
* Can you **connect math → algorithm → system behavior**?

---

## 22. One-Page Mental Model

* Bias = wrong assumptions
* Variance = sensitivity to data
* Noise = unavoidable
* Generalization = balance

```

---

### Next (strongly recommended order)
1. **Underfitting vs Overfitting (with learning curves)**
2. **Regularization (L1/L2/ElasticNet)**
3. **Cross-validation & model selection**
4. **Double descent (modern FAANG topic)**

Say which one — we’ll keep building this **properly**.
```
