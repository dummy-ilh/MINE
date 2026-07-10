md
# 📈 Moving Beyond Linearity — Structured, Exam-Ready Notes

Linear models are powerful because they are **simple, interpretable, and statistically well-understood**. However, the assumption  
\[
E(Y|X) = \beta_0 + \beta_1 X
\]
is almost always an **approximation**, and sometimes a poor one.

This chapter studies methods that **relax linearity** while **retaining interpretability**, sitting between classical linear regression and fully non-parametric methods like trees or KNN.

---

## 1️⃣ Why Move Beyond Linear Models?

### Strengths of Linear Models
- Easy to interpret coefficients
- Fast to fit
- Strong inferential framework (SEs, CI, hypothesis tests)
- Low variance when $n \gg p$

### Limitations
- High **bias** if the true relationship is nonlinear
- Even regularization (ridge, lasso, PCR) **reduces variance**, but:
  \[
  \text{Model is still linear in } X
  \]

📌 **Key idea**  
> Chapter 6 reduces variance **within linearity**.  
> Chapter 7 reduces bias by **relaxing linearity**.

---

## 2️⃣ Polynomial Regression

### Model
For degree $d$:
\[
Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \cdots + \beta_d X^d + \varepsilon
\]

- Still **linear in parameters**
- Nonlinear in $X$

### Properties
- Global fit (one polynomial over entire range)
- Simple extension of linear regression

### Pros
- Easy to implement
- Familiar inference tools apply
- Can model smooth curvature

### Cons
- High-degree polynomials are unstable
- Poor behavior near boundaries
- Global nature → local changes affect entire curve

📌 **Bias–Variance**
- Low degree → high bias
- High degree → high variance

---

## 3️⃣ Step Functions

### Idea
Divide range of $X$ into $K$ bins:
\[
X \in [c_{k-1}, c_k) \Rightarrow f(X) = \beta_k
\]

Equivalent to fitting:
\[
Y = \sum_{k=1}^K \beta_k \cdot \mathbf{1}(c_{k-1} \le X < c_k)
\]

### Properties
- Piecewise constant fit
- Discontinuous at cutpoints

### Pros
- Very interpretable
- Captures abrupt changes
- Simple to explain

### Cons
- Discontinuous
- High bias if true function is smooth

📌 Conceptually similar to **decision tree stumps**.

---

## 4️⃣ Regression Splines

### Idea
- Divide $X$ into $K$ regions using **knots**
- Fit a polynomial in each region
- Enforce smoothness at knots

For cubic splines:
- Function, first derivative, and second derivative continuous

### Model
\[
f(X) = \sum_{j=1}^M \beta_j b_j(X)
\]
where $b_j(X)$ are spline basis functions.

### Pros
- Much more flexible than polynomials
- Local control (changes in one region don’t affect others)
- Stable at boundaries

### Cons
- Choice of knots matters
- More complex interpretation

📌 **Regression splines generalize**
- Polynomials (few knots)
- Step functions (degree 0 splines)

---

## 5️⃣ Smoothing Splines

### Key Idea
Avoid explicit knot selection by solving:
\[
\min_f \sum_{i=1}^n (y_i - f(x_i))^2
\;+\;
\lambda \int (f''(t))^2 dt
\]

- First term: goodness of fit
- Second term: smoothness penalty
- $\lambda$ controls bias–variance tradeoff

### Properties
- Implicit knots at every data point
- Effective degrees of freedom controlled by $\lambda$

### Pros
- Automatic smoothness control
- Very flexible
- Elegant mathematical foundation

### Cons
- Less interpretable
- Computational cost $O(n^3)$ (without approximations)

---

## 6️⃣ Local Regression (LOESS / LOWESS)

### Idea
Estimate $f(x_0)$ using points **near $x_0$**:
\[
\min_{\beta_0,\beta_1} \sum_{i} K_\lambda(x_i - x_0)
(y_i - \beta_0 - \beta_1 x_i)^2
\]

- Window size (span) controls smoothness
- Overlapping local neighborhoods

### Pros
- Extremely flexible
- Excellent fit in low dimensions

### Cons
- No global model
- Poor extrapolation
- Computationally expensive
- Curse of dimensionality

📌 **Purely non-parametric** method.

---

## 7️⃣ Generalized Additive Models (GAMs)

### Model
\[
E(Y|X_1,\dots,X_p) = \beta_0 + f_1(X_1) + \cdots + f_p(X_p)
\]

- Each $f_j$ can be:
  - spline
  - polynomial
  - local smoother

### Properties
- Additive structure preserved
- Flexible yet interpretable

### Pros
- Handles multiple predictors
- Partial dependence interpretation
- Strong bias–variance compromise

### Cons
- No interactions (unless explicitly added)
- Less interpretable than linear regression

---

## 8️⃣ Big-Picture Comparison

| Method | Flexibility | Interpretability | Variance |
|----|----|----|----|
| Linear | Very low | Very high | Very low |
| Polynomial | Low–Medium | High | Medium |
| Step | Low | Very high | Low |
| Regression spline | Medium | Medium | Medium |
| Smoothing spline | High | Low | Medium–High |
| Local regression | Very high | Very low | High |
| GAM | High | Medium–High | Medium |

---

## 9️⃣ Bias–Variance Spectrum (Conceptual)


Linear → Polynomial → Splines → GAM → Local Regression
High bias                                  High variance


---

## 🔑 Final Insight

> *Chapter 6 controls variance without increasing flexibility.*  
> *Chapter 7 increases flexibility while controlling interpretability.*

These methods form the **bridge between linear models and fully non-parametric learning**.

---
Below are **concise but non-freshers, exam/interview-ready notes**, written in a **theoretical + mathematical style**, using **$…$ notation throughout**.

---

# 1. Entropy vs Gini Index (Mathematical Comparison)

Consider a node $m$ with class proportions
$$\hat p_{mk}, \quad k=1,\dots,K,\quad \sum_k \hat p_{mk}=1$$

## Gini Index

$$
G_m = \sum_{k=1}^K \hat p_{mk}(1-\hat p_{mk})
= 1 - \sum_{k=1}^K \hat p_{mk}^2
$$

## Cross-Entropy (Deviance)

$$
D_m = -\sum_{k=1}^K \hat p_{mk}\log \hat p_{mk}
$$

### Local behavior (binary case)

Let $\hat p = 0.5+\epsilon$.

* Gini:
  $$
  G(\hat p) = 2\hat p(1-\hat p) \approx 0.5 - 2\epsilon^2
  $$

* Entropy:
  $$
  H(\hat p) = -\hat p\log\hat p - (1-\hat p)\log(1-\hat p)
  \approx \log 2 - 2\epsilon^2
  $$

📌 **Key insight**: both penalize impurity quadratically near $0.5$.

### Practical difference

* **Entropy** penalizes extreme misclassification **more strongly**
* **Gini** is computationally simpler (no logs)
* Splits chosen by both are usually identical

---

# 2. Entropy vs Gini under Class Imbalance

Assume $\hat p_1=0.99,\ \hat p_2=0.01$.

* Gini:
  $$
  G = 1-(0.99^2+0.01^2)=0.0198
  $$

* Entropy:
  $$
  H = -0.99\log0.99 - 0.01\log0.01 \approx 0.056
  $$

📌 **Entropy is more sensitive to minority classes**, hence may favor splits improving rare-class purity.

---

# 3. Classification Trees vs Logistic Regression

## Model form

### Logistic Regression

$$
\log\frac{P(Y=1|X)}{1-P(Y=1|X)} = \beta_0 + \beta^T X
$$

* **Global linear decision boundary**
* Low variance, high bias if non-linear truth
* Interpretable coefficients, calibrated probabilities

### Classification Tree

$$
f(X)=\sum_{m=1}^M c_m \cdot \mathbf{1}(X\in R_m)
$$

* **Piecewise constant**, axis-aligned splits
* Captures interactions automatically
* High variance, unstable to perturbations

📌 Logistic regression estimates **parameters**; trees estimate **regions**.

---

# 4. Bias–Variance: Trees vs Linear Models

Let test MSE be:
$$
\mathbb{E}[(Y-\hat f(X))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2
$$

| Model             | Bias                       | Variance |
| ----------------- | -------------------------- | -------- |
| Linear regression | High (if non-linear truth) | Low      |
| Deep tree         | Low                        | High     |
| Pruned tree       | Moderate                   | Moderate |

📌 Trees reduce bias via flexibility, but inflate variance.

---

# 5. Why Ensembles Fix Tree Weaknesses

Trees suffer from:

* High variance
* Instability to data perturbations

Ensembles reduce variance by **averaging decorrelated estimators**.

---

# 6. Random Forest vs Boosting (Noise Perspective)

## Random Forest

$$
\hat f_{\text{RF}}(x) = \frac{1}{B}\sum_{b=1}^B T_b(x)
$$

* Bagging + feature subsampling
* Variance reduction:
  $$
  \text{Var}(\bar T) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
  $$
* Robust to label noise

## Boosting

Sequential:
$$
f_M(x)=\sum_{m=1}^M \gamma_m h_m(x)
$$

* Focuses on **hard residuals**
* Can **overfit noise**
* Low bias, potentially high variance

📌 **RF beats boosting on noisy data** because it *averages noise*, while boosting *chases it*.

---

# 7. Boosting as Functional Gradient Descent

We minimize empirical risk:
$$
\min_f \sum_{i=1}^n L(y_i,f(x_i))
$$

At iteration $m$:

1. Compute negative gradient:
   $$
   r_i^{(m)} = -\left.\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}\right|*{f=f*{m-1}}
   $$
2. Fit weak learner:
   $$
   h_m(x) \approx r^{(m)}
   $$
3. Update:
   $$
   f_m(x)=f_{m-1}(x)+\eta h_m(x)
   $$

📌 Boosting = **gradient descent in function space**

---

# 8. Trees vs Linear Models (Conceptual Summary)

| Aspect           | Linear Models     | Trees     |
| ---------------- | ----------------- | --------- |
| Structure        | Global            | Local     |
| Interpretability | Coefficients      | Rules     |
| Interactions     | Manual            | Automatic |
| Variance         | Low               | High      |
| Bias             | High if nonlinear | Low       |

---


Below is a **clean, structured, professor-level consolidation** of everything you pasted, focusing on **intuition → math → comparisons → trade-offs**. Think of this as **final notes for mastery**, not a rewrite.

---

# 7.4 Regression Splines

Regression splines address a core weakness of polynomial regression:

> **Global polynomials must use high degree to gain flexibility → instability, especially at boundaries**

Splines instead:

* Keep **low-degree polynomials**
* Gain flexibility by **localizing** them via **knots**

---

## 7.4.1 Piecewise Polynomials

### Idea

Partition the range of ( X ) using **knots**:
[
\xi_1 < \xi_2 < \dots < \xi_K
]

Fit **separate low-degree polynomials** in each interval.

### Example: Piecewise cubic with one knot ( c )

[
y_i =
\begin{cases}
\beta_{01} + \beta_{11}x_i + \beta_{21}x_i^2 + \beta_{31}x_i^3 + \varepsilon_i, & x_i < c \
\beta_{02} + \beta_{12}x_i + \beta_{22}x_i^2 + \beta_{32}x_i^3 + \varepsilon_i, & x_i \ge c
\end{cases}
]

### Problem

Without constraints:

* Function is **discontinuous**
* Derivatives can jump
* High variance, visually absurd fits

➡️ **Need smoothness constraints**

---

## 7.4.2 Constraints and Splines

A **regression spline** enforces:

* Continuity of the function
* Continuity of first ( d-1 ) derivatives (for degree-( d ) spline)

For **cubic splines**:

* Continuous function
* Continuous first and second derivatives
* Third derivative may jump at knots (imperceptible to humans)

---

## 7.4.3 Spline Basis Representation

Instead of explicitly fitting piecewise polynomials, we use a **basis expansion**.

### Cubic spline with ( K ) knots:

[
y_i = \beta_0 + \sum_{j=1}^{K+3} \beta_j b_j(x_i) + \varepsilon_i
]

This is just **linear regression in transformed features**.

---

### Truncated Power Basis

Start with cubic polynomial basis:
[
x,; x^2,; x^3
]

Add one basis per knot ( \xi_k ):

[
h(x,\xi_k) = (x-\xi_k)_+^3 =
\begin{cases}
(x-\xi_k)^3 & x>\xi_k \
0 & \text{otherwise}
\end{cases}
]

### Properties

* Guarantees smoothness up to second derivative
* Each knot adds **one degree of freedom**

### Degrees of Freedom

For cubic spline with ( K ) knots:
[
\text{df} = K + 4
]

---

## Natural Cubic Splines

### Problem with ordinary splines

At boundaries:

* High variance
* Wild extrapolation

### Natural spline constraint

Force linearity beyond boundary knots:
[
g''(x)=0 \quad \text{outside knot range}
]

### Effect

* Much more stable tails
* Narrower confidence intervals
* Preferred in practice

---

## 7.4.4 Choosing Knots

### Options

1. **Manual placement** (domain knowledge)
2. **Uniform quantiles** (default)
3. **Specify degrees of freedom**, software places knots automatically

### Model selection

Use **cross-validation**:
[
\text{CV-RSS}(K) = \sum (y_i - \hat y_i^{(-i)})^2
]

📌 Empirical finding:

* Performance plateaus quickly
* Small df (≈3–5) usually sufficient

---

## 7.4.5 Splines vs Polynomial Regression

| Aspect            | Polynomial   | Splines          |
| ----------------- | ------------ | ---------------- |
| Flexibility       | Global       | Local            |
| Stability         | Poor (tails) | Good             |
| Control           | Degree       | Knots / df       |
| Boundary behavior | Wild         | Stable (natural) |

📌 **Key insight**:
Splines add flexibility **locally**, not globally.

---

# 7.5 Smoothing Splines

Regression splines: **explicit knots**
Smoothing splines: **implicit knots at every data point**

---

## 7.5.1 Penalized Optimization View

Solve:
[
\min_g ;\sum_{i=1}^n (y_i-g(x_i))^2
;+;
\lambda \int (g''(t))^2 dt
]

### Interpretation

* First term: data fit
* Second term: smoothness penalty
* ( \lambda ): bias–variance knob

| ( \lambda )  | Result            |
| ------------ | ----------------- |
| ( 0 )        | Interpolation     |
| ( \infty )   | Linear regression |
| Intermediate | Smooth curve      |

---

### Key Result (Theorem)

The solution ( \hat g(x) ) is:

* A **natural cubic spline**
* Knots at all unique ( x_i )
* Shrunk by ( \lambda )

---

## 7.5.2 Effective Degrees of Freedom

Although smoothing splines have ( n ) parameters, they are **shrunk**.

Matrix form:
[
\hat g_\lambda = S_\lambda y
]

Define:
[
\text{df}*\lambda = \text{tr}(S*\lambda)
]

Properties:
[
\lambda \uparrow \quad \Rightarrow \quad \text{df}*\lambda \downarrow
]
[
n \ge \text{df}*\lambda \ge 2
]

📌 df measures **model flexibility**, not parameter count.

---

## Choosing ( \lambda ): LOOCV

Efficient formula:
[
\text{CV}(\lambda) =
\sum_{i=1}^n
\left(
\frac{y_i-\hat g_\lambda(x_i)}
{1-(S_\lambda)_{ii}}
\right)^2
]

➡️ Same cost as a single fit.

---

# 7.6 Local Regression (LOESS)

### Core idea

Estimate ( f(x_0) ) using **only nearby points**.

### Algorithm

1. Select fraction ( s ) of nearest neighbors
2. Weight by distance
3. Fit **weighted least squares**
4. Predict at ( x_0 )

### Tuning parameter

* **Span ( s )** controls smoothness
* Small ( s ): low bias, high variance
* Large ( s ): high bias, low variance

### Limitations

* Memory-based
* Computationally expensive
* Suffers from curse of dimensionality

---

# 7.7 Generalized Additive Models (GAMs)

## Model

Regression:
[
Y = \beta_0 + \sum_{j=1}^p f_j(X_j) + \varepsilon
]

Classification:
[
\log\frac{p(X)}{1-p(X)} = \beta_0 + \sum_{j=1}^p f_j(X_j)
]

---

## Why GAMs matter

### Strengths

✔ Non-linear effects
✔ Interpretability (partial plots)
✔ Automatic smoothness control
✔ Inference still possible

### Weakness

✖ Additivity assumption
✖ Misses high-order interactions

(Though interactions can be manually added)

---

## Estimation

* **Spline GAMs** → least squares
* **Smoothing spline GAMs** → backfitting
* Each update fits ( f_j ) on partial residuals:
  [
  r_i^{(j)} = y_i - \sum_{k\neq j} f_k(x_{ik})
  ]

---

# Big Picture Comparison

| Method            | Flexibility | Interpretability | Stability |
| ----------------- | ----------- | ---------------- | --------- |
| Polynomial        | Low         | Medium           | Poor      |
| Regression spline | Medium      | Medium           | Good      |
| Smoothing spline  | High        | Medium           | Very good |
| Local regression  | Very high   | Low              | Poor      |
| GAM               | High        | High             | Good      |
| Trees / RF        | Very high   | Low              | Strong    |

---

## Final Conceptual Hierarchy

[
\text{Linear} ;\subset;
\text{Splines} ;\subset;
\text{GAMs} ;\subset;
\text{Trees / Ensembles}
]

Each step trades **structure for flexibility**.

---




