# 📘 Logistic Regression — The Complete Reference

> **Ground-up. Every concept. Every question. Numerical examples throughout.**

---

## 🧠 What I Understood Before Writing This

Before diving in, here is a plain-English map of everything this document covers:

Logistic Regression is a **classification algorithm** (despite having "regression" in its name). It predicts the **probability** that a given input belongs to a class. The output is always between 0 and 1.

The core intellectual journey is:

```
Raw score (linear combo of features)
        ↓
We want to predict a probability (0 to 1)
        ↓
Probabilities are awkward to model directly (bounded)
        ↓
Convert to ODDS (0 to ∞) → less bounded
        ↓
Convert to LOG-ODDS / logit (−∞ to +∞) → now unbounded
        ↓
Model the log-odds as a linear function of inputs
        ↓
Invert back → get probabilities via the SIGMOID function
        ↓
Threshold to make a binary decision
```

This document builds each step from scratch, with numbers, derivations, and every possible Q&A.

---

## 📑 Table of Contents

1. [The Classification Problem](#1-the-classification-problem)
2. [Why Linear Regression Fails](#2-why-linear-regression-fails)
3. [Probability, Odds, and Log-Odds](#3-probability-odds-and-log-odds)
4. [From Log-Odds to the Logistic Function](#4-from-log-odds-to-the-logistic-function)
5. [The Logistic Regression Model](#5-the-logistic-regression-model)
6. [Decision Boundary](#6-decision-boundary)
7. [Logistic Regression as a GLM](#7-logistic-regression-as-a-glm)
8. [The Cost Function (Log-Loss)](#8-the-cost-function-log-loss)
9. [Maximum Likelihood Estimation](#9-maximum-likelihood-estimation)
10. [Gradient Descent & Parameter Updates](#10-gradient-descent--parameter-updates)
11. [Regularization](#11-regularization)
12. [Multiclass Logistic Regression](#12-multiclass-logistic-regression)
13. [Assumptions of Logistic Regression](#13-assumptions-of-logistic-regression)
14. [Model Evaluation Metrics](#14-model-evaluation-metrics)
15. [Numerical End-to-End Example](#15-numerical-end-to-end-example)
16. [Complete Q&A Reference](#16-complete-qa-reference)

---

## 1. The Classification Problem

### What Are We Trying to Do?

Given a set of input features **X**, predict a **discrete label y**.

**Binary Classification**: y ∈ {0, 1}

| x (hours studied) | y (pass/fail) |
|---|---|
| 1 | 0 (fail) |
| 2 | 0 |
| 3 | 0 |
| 5 | 1 (pass) |
| 6 | 1 |
| 7 | 1 |

We want a model that, given hours = 4.5, outputs something like "probability of passing = 0.72".

### Notation

| Symbol | Meaning |
|---|---|
| x | Input feature vector (n-dimensional) |
| y | True label (0 or 1) |
| ŷ | Predicted label |
| p or ŷ | Predicted probability P(y=1 \| x) |
| w or β | Weight vector (parameters) |
| b or β₀ | Bias / intercept |
| m | Number of training examples |
| n | Number of features |
| σ | Sigmoid function |

---

## 2. Why Linear Regression Fails

### The Naive Approach

You might think: "I'll fit a line to (x, y) and threshold at 0.5."

Let's try with a simple dataset:

```
x:  1   2   3   5   6   7
y:  0   0   0   1   1   1
```

A linear regression gives:  `ŷ = 0.167x − 0.333`

| x | Linear Prediction |
|---|---|
| 1 | −0.167 ← **NEGATIVE probability?** |
| 4 | 0.33 |
| 7 | 0.836 |
| 10 | 1.337 ← **GREATER THAN 1?** |

### Problem 1: Output is Unbounded

Linear regression predicts values in (−∞, +∞). Probabilities must live in [0, 1].

### Problem 2: The Boundary Shifts With Outliers

Add one outlier at x=50, y=1. The entire regression line tilts. A point at x=4 that should be classified as 1 might now be classified as 0 because the line moved.

### Problem 3: Constant Marginal Effect Doesn't Make Sense

Linear regression says: "each unit increase in x produces a constant increase in probability." But in reality, moving from p=0.01 to p=0.02 is very different from moving from p=0.49 to p=0.50. The relationship is inherently **nonlinear**.

### Problem 4: MSE Loss Is Wrong for Classification

MSE assumes residuals are normally distributed. For binary outcomes, residuals are bounded and follow a Bernoulli distribution, not Gaussian.

---

## 3. Probability, Odds, and Log-Odds

This is **the heart** of understanding logistic regression. We need to find a transformation of probability that:
- Is defined over the full real line (−∞, +∞)
- Is monotonically increasing (preserves order)
- Has a natural probabilistic interpretation

### 3.1 Probability

**p** = probability of the event (e.g., passing the exam)

Range: [0, 1]

Problem: Bounded. Cannot model as a linear function of inputs across the full range.

**Example**: p = 0.75 means "75% chance of passing."

### 3.2 Odds

**Odds** = ratio of the probability of the event occurring to it NOT occurring:

```
Odds = p / (1 − p)
```

Range: [0, +∞)

| Probability p | Odds | English |
|---|---|---|
| 0.10 | 0.10/0.90 = 0.111 | 1:9 against |
| 0.25 | 0.25/0.75 = 0.333 | 1:3 against |
| 0.50 | 0.50/0.50 = 1.000 | Even odds |
| 0.75 | 0.75/0.25 = 3.000 | 3:1 in favor |
| 0.90 | 0.90/0.10 = 9.000 | 9:1 in favor |
| 0.99 | 0.99/0.01 = 99.00 | 99:1 in favor |

**Interpretation**: Odds = 3 means "the event is 3 times more likely to happen than not happen."

Still a problem: The range is [0, +∞), still not symmetric.

### 3.3 Log-Odds (Logit)

Take the **natural logarithm** of the odds:

```
log-odds = logit(p) = ln(p / (1 − p))
```

Range: (−∞, +∞) ← **NOW UNBOUNDED IN BOTH DIRECTIONS!**

| Probability p | Odds | Log-Odds |
|---|---|---|
| 0.01 | 0.0101 | −4.595 |
| 0.10 | 0.111 | −2.197 |
| 0.25 | 0.333 | −1.099 |
| 0.50 | 1.000 | 0.000 |
| 0.75 | 3.000 | 1.099 |
| 0.90 | 9.000 | 2.197 |
| 0.99 | 99.00 | 4.595 |

**Key properties of the logit:**
- logit(0.5) = 0 → symmetric around 0
- logit(p) = −logit(1−p) → symmetric function
- Monotonically increasing (if p goes up, logit goes up)
- Defined on (−∞, +∞)

**Numerical example:**
```
p = 0.8
Odds = 0.8 / 0.2 = 4.0
Log-odds = ln(4.0) = 1.386

Now we can say:
"For every unit increase in x, the log-odds of passing increases by 1.386"
This is a statement we can model linearly!
```

---

## 4. From Log-Odds to the Logistic Function

### 4.1 The Key Modeling Assumption

We assume the **log-odds is a linear function of the input features**:

```
logit(p) = ln(p / (1 − p)) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Let's call the right-hand side `z`:

```
z = β₀ + β₁x₁ + ... + βₙxₙ    (the "linear score" or "logit score")
```

So our assumption is:

```
ln(p / (1 − p)) = z
```

### 4.2 Solving for p (Deriving the Sigmoid)

Now let's **algebraically invert** this to get p in terms of z:

**Step 1**: Exponentiate both sides:
```
p / (1 − p) = eᶻ
```

**Step 2**: Multiply both sides by (1 − p):
```
p = eᶻ(1 − p)
p = eᶻ − p·eᶻ
```

**Step 3**: Collect p terms on the left:
```
p + p·eᶻ = eᶻ
p(1 + eᶻ) = eᶻ
```

**Step 4**: Divide both sides by (1 + eᶻ):
```
p = eᶻ / (1 + eᶻ)
```

**Step 5**: (Optional simplification) Divide numerator and denominator by eᶻ:
```
p = 1 / (1 + e⁻ᶻ)
```

**This is the SIGMOID (logistic) function:**

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**And the full logistic regression model is:**

```
P(y=1 | x) = σ(β₀ + β₁x₁ + ... + βₙxₙ) = 1 / (1 + e⁻⁽ᵝ⁰ ⁺ ᵝ¹ˣ¹ ⁺ ··· ⁺ ᵝⁿˣⁿ⁾)
```

### 4.3 Properties of the Sigmoid

| z | σ(z) |
|---|---|
| −∞ | 0 |
| −5 | 0.0067 |
| −2 | 0.119 |
| −1 | 0.269 |
| 0 | 0.500 |
| 1 | 0.731 |
| 2 | 0.880 |
| 5 | 0.993 |
| +∞ | 1 |

**Key properties:**
- σ(z) ∈ (0, 1) always
- σ(0) = 0.5
- σ(−z) = 1 − σ(z)  (symmetry)
- σ'(z) = σ(z)(1 − σ(z))  (elegant derivative — crucial for backprop)
- Monotonically increasing

**Numerical example of the chain:**
```
Suppose we have a model:
z = −4 + 1.5x  (x = hours studied)

For x = 3 hours:
z = −4 + 1.5(3) = −4 + 4.5 = 0.5
p = 1 / (1 + e⁻⁰·⁵) = 1 / (1 + 0.6065) = 1 / 1.6065 = 0.6225

So a student who studies 3 hours has a 62.25% chance of passing.

For x = 2 hours:
z = −4 + 1.5(2) = −1.0
p = 1 / (1 + e¹) = 1 / (1 + 2.718) = 1 / 3.718 = 0.269

Only 26.9% chance.
```

---

## 5. The Logistic Regression Model

### 5.1 Full Formulation

For a binary classification with n features:

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

P(y=1 | x; β) = σ(z) = 1 / (1 + e⁻ᶻ)

P(y=0 | x; β) = 1 − σ(z) = e⁻ᶻ / (1 + e⁻ᶻ)
```

Or more compactly using vector notation:

```
z = βᵀx     (where x₀ = 1 for the bias term)

ŷ = σ(βᵀx)
```

### 5.2 Compact Form for Both Classes

Notice:
```
P(y=1 | x) = σ(z)
P(y=0 | x) = σ(−z) = 1 − σ(z)
```

This can be written in one elegant expression:

```
P(y | x) = σ(z)^y · (1 − σ(z))^(1−y)
```

This is the **Bernoulli PMF** with parameter σ(z). This will be crucial for deriving the loss function.

### 5.3 Interpreting Coefficients

**The coefficient βⱼ tells you:** A one-unit increase in xⱼ multiplies the odds by **e^βⱼ**, holding all other features constant.

**Why?**
```
log-odds = β₀ + β₁x₁ + ... + βⱼxⱼ + ...

Increase xⱼ by 1:
new log-odds = β₀ + β₁x₁ + ... + βⱼ(xⱼ+1) + ...
             = old log-odds + βⱼ

Change in log-odds = βⱼ
Change in odds = e^βⱼ   (the ODDS RATIO)
```

**Numerical example:**
```
Model for loan default:
log-odds = −3 + 0.05·(credit_score) + 0.8·(is_employed)

β for credit_score = 0.05
→ Each 1-point increase in credit score multiplies odds of default by e^0.05 = 1.051
→ i.e., 5.1% increase in odds

β for is_employed = 0.8
→ Being employed (vs not) multiplies odds of default by e^0.8 = 2.226
Wait — that's positive, meaning employment increases default risk.
If this were a real model, we'd expect a negative coefficient.

Let's say β = −1.5 for is_employed:
→ e^(−1.5) = 0.223
→ Being employed reduces odds of default by 77.7%
```

---

## 6. Decision Boundary

### 6.1 The Threshold Rule

To convert a probability to a class label, we apply a threshold t (default = 0.5):

```
ŷ = 1  if σ(z) ≥ 0.5
ŷ = 0  if σ(z) < 0.5
```

Since σ(z) ≥ 0.5 ↔ z ≥ 0, the decision rule simplifies to:

```
ŷ = 1  if βᵀx ≥ 0
ŷ = 0  if βᵀx < 0
```

### 6.2 Decision Boundary Is Linear

The decision boundary is where βᵀx = 0. This is a **hyperplane** in n-dimensional feature space.

**1D Example:**
```
z = −4 + 1.5x = 0
x = 4/1.5 = 2.667

So the boundary is at x = 2.67 hours.
Students studying > 2.67 hrs → predicted to pass.
Students studying ≤ 2.67 hrs → predicted to fail.
```

**2D Example:**
```
z = −3 + x₁ + 2x₂ = 0
x₁ + 2x₂ = 3  ← this is the decision boundary line

Point (1, 1): z = −3 + 1 + 2 = 0 → exactly on boundary
Point (2, 2): z = −3 + 2 + 4 = 3 > 0 → class 1
Point (0, 0): z = −3 + 0 + 0 = −3 < 0 → class 0
```

### 6.3 The Boundary Can Be Nonlinear With Feature Engineering

If we add x₁² and x₁x₂ as features, the decision boundary in original feature space becomes curved — but logistic regression itself is still linear in the **feature space**.

**Example:**
```
z = −1 + x₁² + x₂²

Boundary: x₁² + x₂² = 1  ← a CIRCLE in original feature space
```

### 6.4 Adjusting the Threshold

Changing the threshold from 0.5 changes the boundary location:

| Threshold | Effect |
|---|---|
| 0.5 (default) | Balanced precision/recall |
| 0.3 | More aggressive: predict more positives (higher recall, lower precision) |
| 0.7 | More conservative: predict fewer positives (lower recall, higher precision) |

**When to use a non-0.5 threshold:**
- Fraud detection → low threshold (don't miss fraud)
- Spam filter → high threshold (don't mark legitimate email as spam)

---

## 7. Logistic Regression as a GLM

### 7.1 What Is a Generalized Linear Model?

A **Generalized Linear Model (GLM)** is a flexible extension of ordinary linear regression that allows the response variable to have distributions other than normal.

A GLM has three components:

| Component | Description | Example (Logistic Reg.) |
|---|---|---|
| **Random Component** | Distribution of y | Bernoulli(p) |
| **Systematic Component** | Linear predictor | η = β₀ + β₁x₁ + ... |
| **Link Function** | Maps mean to linear predictor | logit(p) = ln(p/(1−p)) |

### 7.2 The GLM Framework in Detail

**General form:**
```
g(μ) = βᵀx

where:
μ = E[y|x]  (expected value / mean of y)
g = link function
```

**For logistic regression:**
```
μ = E[y|x] = P(y=1|x) = p
g(p) = logit(p) = ln(p/(1−p))

So: ln(p/(1−p)) = βᵀx
Inverting: p = σ(βᵀx)
```

### 7.3 Why the Logit Is the "Canonical Link" for Bernoulli

The exponential family form of the Bernoulli distribution is:

```
P(y; η) = exp(yη − log(1 + eη))

where η = log(p/(1−p))
```

The canonical parameter is **η = logit(p)**. Setting η = βᵀx gives the logistic regression model. The canonical link arises naturally from the distribution's exponential family form — it's not arbitrary.

### 7.4 Other Common GLMs

| Model | Distribution | Link Function | Use Case |
|---|---|---|---|
| Linear Regression | Normal | Identity: g(μ) = μ | Continuous outcome |
| Logistic Regression | Bernoulli | Logit: g(p) = log(p/(1-p)) | Binary outcome |
| Poisson Regression | Poisson | Log: g(μ) = log(μ) | Count data |
| Gamma Regression | Gamma | Log or Inverse | Positive continuous |
| Probit Model | Bernoulli | Probit: Φ⁻¹(p) | Binary (alternative to logit) |

### 7.5 Logit vs Probit

Both produce sigmoid-shaped curves, but:

```
Logit: uses logistic distribution → heavier tails
Probit: uses normal distribution → lighter tails

For most practical purposes, predictions are nearly identical.
Logit is preferred because coefficients have a natural odds-ratio interpretation.
```

---

## 8. The Cost Function (Log-Loss)

### 8.1 Why Not Use MSE?

For logistic regression, MSE creates a **non-convex** loss surface with many local minima. Gradient descent may get stuck.

```
MSE = (1/m) Σ (σ(z) − y)²

The composition of squared error with sigmoid is non-convex.
```

We need a loss function that:
1. Is convex (one global minimum)
2. Penalizes confident wrong predictions severely
3. Comes from a probabilistic foundation (MLE)

### 8.2 Deriving the Log-Loss from MLE

(Full MLE derivation in Section 9. Here we state the result.)

The **binary cross-entropy loss** for one sample:

```
L(y, ŷ) = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
```

Where ŷ = σ(βᵀx) = P(y=1|x)

**For y=1:**  `L = −log(ŷ)`    (we want ŷ close to 1)
**For y=0:**  `L = −log(1−ŷ)` (we want ŷ close to 0)

**Numerical examples:**

Case 1: y=1, ŷ=0.9 (correct, confident)
```
L = −log(0.9) = 0.105  ← small loss ✓
```

Case 2: y=1, ŷ=0.5 (correct, uncertain)
```
L = −log(0.5) = 0.693  ← moderate loss
```

Case 3: y=1, ŷ=0.1 (wrong, confident)
```
L = −log(0.1) = 2.303  ← large loss ✗
```

Case 4: y=0, ŷ=0.1 (correct, confident)
```
L = −log(1−0.1) = −log(0.9) = 0.105  ← small loss ✓
```

Case 5: y=0, ŷ=0.95 (wrong, very confident)
```
L = −log(1−0.95) = −log(0.05) = 2.996  ← huge penalty ✗✗
```

### 8.3 The Full Cost Function

Average over all m training examples:

```
J(β) = −(1/m) Σᵢ [yᵢ·log(σ(zᵢ)) + (1−yᵢ)·log(1−σ(zᵢ))]
```

Where zᵢ = βᵀxᵢ

This is the **binary cross-entropy** or **log-loss** cost function.

### 8.4 Why Is It Convex?

The Hessian of J(β) is **positive semi-definite**. This follows from:

```
∂²J/∂β² = (1/m) Xᵀ·diag(σ(z)·(1−σ(z)))·X

The diagonal matrix has non-negative entries (σ(z)(1−σ(z)) ≥ 0 always).
XᵀDX is always PSD for any X and non-negative D.
```

Therefore J(β) is convex → **gradient descent is guaranteed to find the global minimum**.

### 8.5 Comparing Log-Loss vs MSE Behavior

| Prediction ŷ | True y | MSE Loss | Log-Loss |
|---|---|---|---|
| 0.99 | 1 | 0.0001 | 0.010 |
| 0.90 | 1 | 0.0100 | 0.105 |
| 0.50 | 1 | 0.2500 | 0.693 |
| 0.10 | 1 | 0.8100 | 2.303 |
| 0.01 | 1 | 0.9801 | 4.605 |

Both increase as prediction gets worse, but log-loss **blows up faster** for overconfident wrong predictions → better signal for learning.

---

## 9. Maximum Likelihood Estimation

### 9.1 The Likelihood Function

We want to find the parameters β that make the observed data **most likely** under our model.

Assume each training example is i.i.d. The probability of a single label given the model:

```
P(yᵢ | xᵢ; β) = σ(zᵢ)^yᵢ · (1 − σ(zᵢ))^(1−yᵢ)
```

This is just saying:
- If yᵢ=1: P = σ(zᵢ)
- If yᵢ=0: P = 1−σ(zᵢ)

The **joint likelihood** of all m observations:

```
L(β) = ∏ᵢ₌₁ᵐ P(yᵢ | xᵢ; β)
     = ∏ᵢ₌₁ᵐ σ(zᵢ)^yᵢ · (1 − σ(zᵢ))^(1−yᵢ)
```

### 9.2 Log-Likelihood (Easier to Work With)

Products → sums under logarithm:

```
ℓ(β) = log L(β)
      = Σᵢ [yᵢ·log(σ(zᵢ)) + (1−yᵢ)·log(1−σ(zᵢ))]
```

**Maximizing ℓ(β) is equivalent to minimizing −ℓ(β)**, which gives the cross-entropy cost:

```
J(β) = −(1/m) ℓ(β) = −(1/m) Σᵢ [yᵢ·log(σ(zᵢ)) + (1−yᵢ)·log(1−σ(zᵢ))]
```

So the log-loss IS the negative log-likelihood. **Minimizing the cross-entropy is doing MLE.**

### 9.3 Numerical MLE Example

Dataset:
```
(x=1, y=0), (x=2, y=0), (x=5, y=1), (x=6, y=1)

Model: z = β₀ + β₁x

Try β₀ = −4, β₁ = 1:
```

| i | xᵢ | yᵢ | zᵢ | σ(zᵢ) | P(yᵢ\|xᵢ) | log P |
|---|---|---|---|---|---|---|
| 1 | 1 | 0 | −3 | 0.047 | 1−0.047=0.953 | −0.048 |
| 2 | 2 | 0 | −2 | 0.119 | 1−0.119=0.881 | −0.127 |
| 3 | 5 | 1 | 1 | 0.731 | 0.731 | −0.313 |
| 4 | 6 | 1 | 2 | 0.880 | 0.880 | −0.128 |

```
ℓ = −0.048 − 0.127 − 0.313 − 0.128 = −0.616
J = 0.616 / 4 = 0.154  ← this is our log-loss to minimize
```

Now try β₀ = −3, β₁ = 0.9 and compute J again. The β that gives the lowest J is the MLE.

---

## 10. Gradient Descent & Parameter Updates

### 10.1 Gradient of the Loss

The gradient of J(β) with respect to βⱼ:

```
∂J/∂βⱼ = (1/m) Σᵢ (σ(zᵢ) − yᵢ) · xᵢⱼ
```

In matrix form:

```
∇J = (1/m) Xᵀ(σ(Xβ) − y)
```

**Elegant result:** The gradient has the same form as linear regression — it's the "prediction error" times the feature value. The sigmoid nonlinearity fully disappears in the gradient formula.

### 10.2 Derivation of the Gradient

For one sample, differentiating L w.r.t. βⱼ:

```
∂L/∂βⱼ = ∂/∂βⱼ [−y·log(σ(z)) − (1−y)·log(1−σ(z))]

Using chain rule and the fact that σ'(z) = σ(z)(1−σ(z)):

∂L/∂βⱼ = [−y/σ(z) + (1−y)/(1−σ(z))] · σ(z)(1−σ(z)) · xⱼ

         = [−y(1−σ(z)) + (1−y)σ(z)] · xⱼ

         = [σ(z) − y] · xⱼ
```

Clean! The error is simply (predicted − actual).

### 10.3 Gradient Descent Update Rule

```
β := β − α · ∇J(β)

Componentwise:
βⱼ := βⱼ − α · (1/m) Σᵢ (σ(zᵢ) − yᵢ) · xᵢⱼ
```

Where α is the **learning rate**.

### 10.4 Numerical Gradient Descent Example

Dataset (simplified to 1 feature):
```
x = [1, 2, 5, 6],  y = [0, 0, 1, 1]

Initialize: β₀ = 0, β₁ = 0, learning rate α = 0.1
```

**Iteration 1:**

Step 1 — Compute z = β₀ + β₁x:
```
z = [0, 0, 0, 0]
```

Step 2 — Compute σ(z):
```
σ(z) = [0.5, 0.5, 0.5, 0.5]
```

Step 3 — Compute errors (σ − y):
```
errors = [0.5−0, 0.5−0, 0.5−1, 0.5−1]
       = [0.5,   0.5,  −0.5,  −0.5]
```

Step 4 — Compute gradients:
```
∂J/∂β₀ = (1/4)[0.5·1 + 0.5·1 + (−0.5)·1 + (−0.5)·1]
        = (1/4)[0.5 + 0.5 − 0.5 − 0.5]
        = 0

∂J/∂β₁ = (1/4)[0.5·1 + 0.5·2 + (−0.5)·5 + (−0.5)·6]
        = (1/4)[0.5 + 1.0 − 2.5 − 3.0]
        = (1/4)(−4.0)
        = −1.0
```

Step 5 — Update:
```
β₀ := 0 − 0.1·0 = 0
β₁ := 0 − 0.1·(−1.0) = 0.1
```

**Iteration 2:**

z = [0 + 0.1·1, 0 + 0.1·2, 0 + 0.1·5, 0 + 0.1·6]
  = [0.1, 0.2, 0.5, 0.6]

σ(z) = [0.525, 0.550, 0.622, 0.646]

errors = [0.525, 0.550, −0.378, −0.354]

∂J/∂β₁ = (1/4)[0.525·1 + 0.550·2 + (−0.378)·5 + (−0.354)·6]
        = (1/4)[0.525 + 1.1 − 1.89 − 2.124]
        = (1/4)(−2.389) = −0.597

β₁ := 0.1 − 0.1·(−0.597) = 0.1597

After many iterations, the model converges to approximately β₀ ≈ −4, β₁ ≈ 1.

### 10.5 Variants of Gradient Descent

| Variant | Uses | Update Frequency | Notes |
|---|---|---|---|
| Batch GD | All m samples | Per epoch | Stable, slow for large m |
| Stochastic GD (SGD) | 1 sample | Per sample | Fast, noisy |
| Mini-batch GD | Batch of k samples | Per batch | Best in practice |

### 10.6 Other Optimizers

Because J(β) is convex for logistic regression, **second-order methods** work well:

- **Newton-Raphson**: Uses Hessian → faster convergence, expensive for large n
- **L-BFGS**: Quasi-Newton, approximates Hessian → used in sklearn's default solver
- **IRLS (Iteratively Reweighted Least Squares)**: The classic statistical fitting method, equivalent to Newton-Raphson for GLMs

---

## 11. Regularization

### 11.1 Why Regularize?

Without regularization, logistic regression can **overfit**: it finds weights that perfectly separate the training data but fail on new data. If classes are perfectly linearly separable, weights grow to ±∞ (the model becomes infinitely confident).

### 11.2 L2 Regularization (Ridge)

Add a penalty proportional to the sum of squared weights:

```
J_ridge(β) = J(β) + (λ/2m) Σⱼ βⱼ²

Gradient update:
βⱼ := βⱼ − α · [(1/m)Σᵢ(σ(zᵢ)−yᵢ)xᵢⱼ + (λ/m)βⱼ]
```

Note: The bias β₀ is typically **NOT** regularized.

**Effect**: Shrinks weights toward zero, but doesn't make them exactly zero. Prefers many small weights. Handles correlated features well.

**In terms of C in sklearn**: λ = 1/C (so larger C = less regularization)

### 11.3 L1 Regularization (Lasso)

Add a penalty proportional to the sum of absolute values:

```
J_lasso(β) = J(β) + (λ/m) Σⱼ |βⱼ|
```

**Effect**: Produces **sparse** solutions — drives many weights exactly to zero. Effectively does feature selection.

**Why does L1 create sparsity?** The L1 ball (diamond shape) has corners at the axes. When the cost function's contours touch the L1 ball, they tend to hit at corners (where some βⱼ = 0).

### 11.4 Elastic Net

Combines L1 and L2:

```
J_elastic(β) = J(β) + λ[ρ·Σ|βⱼ| + (1−ρ)/2·Σβⱼ²]
```

Where ρ ∈ [0,1] controls the mix.

### 11.5 Numerical Regularization Example

```
Dataset: 2 features, both correlated with y
β* without regularization: β₁=5.2, β₂=4.8

With L2 regularization (λ=1):
β₁≈2.3, β₂≈2.1  (both shrunk, roughly equally)

With L1 regularization (λ=1):
β₁≈3.1, β₂≈0.0  (one feature selected, other zeroed out)
```

### 11.6 Choosing λ (or C)

Use **cross-validation**:
```
λ_values = [0.001, 0.01, 0.1, 1, 10, 100]
For each λ:
  Fit model on training folds
  Evaluate on validation fold
  Average accuracy/AUC
Pick λ with best validation performance
```

---

## 12. Multiclass Logistic Regression

### 12.1 One-vs-Rest (OvR) / One-vs-All

For K classes, train K binary classifiers:

- Classifier 1: Class 1 vs. {Class 2, 3, ..., K}
- Classifier 2: Class 2 vs. {Class 1, 3, ..., K}
- ...

At prediction, pick the class whose classifier outputs the highest probability.

**Drawback**: Probabilities from K classifiers don't sum to 1.

### 12.2 Softmax Regression (Multinomial Logistic Regression)

The natural generalization of logistic regression to K classes:

For each class k, compute a linear score:
```
zₖ = βₖᵀx    (separate weight vector per class)
```

Convert to probabilities using **softmax**:
```
P(y=k | x) = e^zₖ / Σⱼ e^zⱼ
```

**Properties:**
- All probabilities sum to 1
- Each probability ∈ (0, 1)
- Generalizes sigmoid (for K=2, softmax reduces to sigmoid)

**Numerical example:**
```
3 classes (cat, dog, bird), 1 feature (animal weight in kg)

Suppose for x = 5kg:
z_cat = 2.0,  z_dog = 1.5,  z_bird = −0.5

exp(z_cat) = 7.389
exp(z_dog) = 4.482
exp(z_bird) = 0.607
Sum = 12.478

P(cat)  = 7.389 / 12.478 = 0.592
P(dog)  = 4.482 / 12.478 = 0.359
P(bird) = 0.607 / 12.478 = 0.049
Sum = 1.000 ✓
```

### 12.3 Loss for Softmax

**Categorical cross-entropy:**
```
J = −(1/m) Σᵢ Σₖ yᵢₖ · log(P(y=k | xᵢ))
```

Where yᵢₖ is 1 if sample i belongs to class k (one-hot encoding).

---

## 13. Assumptions of Logistic Regression

### 13.1 The Six Key Assumptions

**1. Binary (or Categorical) Outcome**
The dependent variable must be binary (for standard logistic regression) or categorical (for multinomial).

**2. Independence of Observations**
Each data point is sampled independently. Violating this (e.g., time series, clustered data) invalidates standard errors.

**3. No Severe Multicollinearity**
Highly correlated predictors lead to unstable, inflated coefficient estimates. Check with VIF (Variance Inflation Factor). VIF > 5-10 is problematic.

**4. Linearity of Log-Odds**
The log-odds must be a **linear function of the continuous predictors**. This does NOT mean the relationship between x and p is linear. Check with Box-Tidwell test or by plotting log-odds vs. features.

**5. No Extreme Outliers**
Logistic regression is sensitive to outliers in continuous predictors. A single extreme point can shift the decision boundary significantly.

**6. Sufficient Sample Size**
Rule of thumb: at least **10-20 events per predictor variable** (EPP rule). With 5 predictors, you need ~50-100 positive cases minimum.

### 13.2 Assumptions NOT Required (Unlike Linear Regression)

- ❌ Normally distributed residuals (not required)
- ❌ Homoscedasticity (not required)
- ❌ Linear relationship between y and x (the relationship is sigmoid-shaped)

### 13.3 What Happens When Assumptions Are Violated

| Violation | Consequence | Fix |
|---|---|---|
| Multicollinearity | Unstable coefficients, wide CIs | Remove correlated features, PCA, regularization |
| Non-linear log-odds | Biased predictions | Add polynomial terms, splines |
| Small n / rare events | Biased β (inflated), convergence issues | Penalized regression (Firth's method), resampling |
| Perfect separation | Infinite MLE, non-convergence | L2 regularization |
| Class imbalance | Biased toward majority class | Oversample, class weights, threshold tuning |

### 13.4 Perfect Separation

This happens when a single feature or combination perfectly separates the classes.

**Example:**
```
x:   1  2  3  4  5  6
y:   0  0  0  1  1  1
```

If the boundary is known perfectly (e.g., x>3.5 → y=1), the MLE for β₁ is +∞. The likelihood can always be increased by making β₁ larger. Gradient descent never converges.

**Fix**: L2 regularization (adds curvature to the loss surface) or Firth's penalized likelihood.

---

## 14. Model Evaluation Metrics

### 14.1 Confusion Matrix

For a binary classifier with threshold t:

```
                    Predicted
                    0           1
Actual  0    TN (True Neg)   FP (False Pos)
        1    FN (False Neg)  TP (True Pos)
```

**Numerical Example:**
```
100 test samples: 50 positive (y=1), 50 negative (y=0)
Model predictions:
  TP = 45, FN = 5, FP = 8, TN = 42
```

### 14.2 Core Metrics

**Accuracy:**
```
(TP + TN) / (TP + TN + FP + FN) = (45 + 42) / 100 = 0.87
```

**Precision (Positive Predictive Value):**
```
TP / (TP + FP) = 45 / (45 + 8) = 45/53 = 0.849
"Of all predicted positives, how many are real?"
```

**Recall / Sensitivity / TPR:**
```
TP / (TP + FN) = 45 / (45 + 5) = 45/50 = 0.900
"Of all actual positives, how many did we catch?"
```

**Specificity / TNR:**
```
TN / (TN + FP) = 42 / (42 + 8) = 42/50 = 0.840
"Of all actual negatives, how many did we correctly identify?"
```

**F1 Score (harmonic mean of precision and recall):**
```
2 · (Precision · Recall) / (Precision + Recall)
= 2 · (0.849 · 0.900) / (0.849 + 0.900)
= 2 · 0.764 / 1.749
= 0.874
```

**F-beta Score (weighted):**
```
F_β = (1 + β²) · (Precision · Recall) / (β²·Precision + Recall)
β=2 → weighs recall twice as much as precision (use for fraud detection)
β=0.5 → weighs precision more (use for spam filtering)
```

### 14.3 ROC Curve and AUC

The **ROC curve** plots TPR (recall) vs. FPR (1−specificity) for all possible thresholds.

```
FPR = FP / (FP + TN) = 8/50 = 0.16
TPR = TP / (TP + FN) = 45/50 = 0.90
```

**AUC (Area Under the ROC Curve):**
- AUC = 1.0 → perfect model
- AUC = 0.5 → random classifier (diagonal line)
- AUC = 0.0 → perfectly wrong

**AUC interpretation**: The probability that a randomly chosen positive sample has a higher predicted probability than a randomly chosen negative sample.

### 14.4 Precision-Recall Curve

More useful than ROC when classes are **heavily imbalanced** (e.g., 99% negative, 1% positive).

### 14.5 Log-Loss (as an evaluation metric)

```
Log-loss = −(1/m) Σᵢ [yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
```

Lower is better. Penalizes confident wrong predictions heavily. Used in competitions.

### 14.6 Calibration

A model is **well-calibrated** if predicted probabilities match empirical frequencies.

If the model says "70% chance of rain" on 100 days, it should actually rain on approximately 70 of those days.

**Calibration plot**: x-axis = predicted probability bins, y-axis = actual fraction of positives.

Logistic regression is generally well-calibrated **if assumptions hold** and there's no severe imbalance.

### 14.7 Hosmer-Lemeshow Test

Statistical test for goodness-of-fit. Groups observations into deciles of predicted probability, then compares observed vs expected counts via chi-square.

H₀: The model fits well
p > 0.05 → fail to reject H₀ → good fit

### 14.8 McFadden's Pseudo-R²

Analogous to R² for OLS:

```
R²_McFadden = 1 − ℓ(β̂) / ℓ(β₀)
```

Where ℓ(β̂) = log-likelihood of fitted model, ℓ(β₀) = log-likelihood of null model (intercept only).

- Range: [0, 1)
- Values of 0.2–0.4 indicate excellent fit (not the same scale as OLS R²!)

---

## 15. Numerical End-to-End Example

### The Dataset

Predicting heart disease (0/1) from age and cholesterol:

| i | age (x₁) | cholesterol (x₂) | heart_disease (y) |
|---|---|---|---|
| 1 | 45 | 200 | 0 |
| 2 | 55 | 250 | 0 |
| 3 | 60 | 280 | 1 |
| 4 | 65 | 300 | 1 |
| 5 | 50 | 220 | 0 |
| 6 | 70 | 310 | 1 |

### Step 1: Feature Scaling (Standardization)

```
x₁_mean = (45+55+60+65+50+70)/6 = 57.5, std = 8.87
x₂_mean = (200+250+280+300+220+310)/6 = 260, std = 41.2

Scaled values (z-score = (x − mean)/std):
Age_scaled:  [−1.41, −0.28, 0.28, 0.85, −0.84, 1.41]
Chol_scaled: [−1.46, −0.24, 0.49, 0.97, −0.97, 1.21]
```

### Step 2: Initialize Parameters

```
β₀ = 0, β₁ = 0, β₂ = 0, α = 0.5
```

### Step 3: First Gradient Descent Step

All z = 0, all σ(z) = 0.5

Errors: [0.5, 0.5, −0.5, −0.5, 0.5, −0.5]

```
∂J/∂β₀ = (1/6)[0.5+0.5+(−0.5)+(−0.5)+0.5+(−0.5)] = (1/6)(0) = 0

∂J/∂β₁ = (1/6)[ 0.5·(−1.41) + 0.5·(−0.28) + (−0.5)·0.28 + 
                 (−0.5)·0.85 + 0.5·(−0.84) + (−0.5)·1.41 ]
        = (1/6)[−0.705 − 0.14 − 0.14 − 0.425 − 0.42 − 0.705]
        = (1/6)(−2.535) = −0.4225

∂J/∂β₂ = (1/6)[0.5·(−1.46) + 0.5·(−0.24) + (−0.5)·0.49 +
                (−0.5)·0.97 + 0.5·(−0.97) + (−0.5)·1.21]
        = (1/6)[−0.73 − 0.12 − 0.245 − 0.485 − 0.485 − 0.605]
        = (1/6)(−2.67) = −0.445

Update:
β₀ := 0 − 0.5·0 = 0
β₁ := 0 − 0.5·(−0.4225) = 0.211
β₂ := 0 − 0.5·(−0.445) = 0.223
```

### Step 4: After Convergence (hypothetical)

After many iterations:
```
β₀ = −6.2,  β₁ = 1.8,  β₂ = 2.1

Interpretation:
- A 1 SD increase in age multiplies odds of heart disease by e^1.8 = 6.05
- A 1 SD increase in cholesterol multiplies odds by e^2.1 = 8.17
```

### Step 5: Predict a New Patient

Patient: age=62, cholesterol=290

```
age_scaled = (62 − 57.5) / 8.87 = 0.507
chol_scaled = (290 − 260) / 41.2 = 0.728

z = −6.2 + 1.8·(0.507) + 2.1·(0.728)
  = −6.2 + 0.913 + 1.529
  = −3.758

Wait — this seems low. With β₀ = −6.2 and moderate positive features,
the result should be around −3.76, which gives:

P(heart disease) = 1 / (1 + e^3.758) = 1 / (1 + 42.87) = 0.023

This is a 2.3% predicted probability. Classification as 0 (no disease).
```

---

## 16. Complete Q&A Reference

### 🔵 Conceptual Questions

**Q: Why is it called "logistic" regression if it's a classification algorithm?**
A: Because it models the *log-odds* as a linear function (the "regression" part), and uses the *logistic function* (sigmoid) to convert that to a probability. The final decision (class label) is a threshold on the predicted probability. The name comes from the Belgian mathematician Pierre François Verhulst who introduced the "logistic" function in 1838.

---

**Q: Is logistic regression a linear or nonlinear model?**
A: Both, depending on what you mean. The **decision boundary** is linear (a hyperplane). The **probability output** is a nonlinear function of x (due to the sigmoid). The **log-odds** are linear in x. When people say "linear model," they usually mean the decision boundary is linear.

---

**Q: Why is log-loss preferred over MSE for logistic regression?**
A: Three reasons: (1) Log-loss is derived from the correct probabilistic model (MLE for Bernoulli), (2) It produces a convex optimization landscape — gradient descent is guaranteed to converge to the global optimum, (3) It penalizes overconfident wrong predictions much more harshly, which creates stronger learning signals.

---

**Q: What does the coefficient β₁ = 2.3 mean?**
A: A one-unit increase in x₁ increases the **log-odds** of y=1 by 2.3, which multiplies the **odds** by e^2.3 = 9.97. The predicted probability also increases, but not by a constant amount — how much it increases depends on the starting value of p (the effect is largest near p=0.5).

---

**Q: Can logistic regression output probabilities greater than 1 or less than 0?**
A: No. The sigmoid function is bounded in (0, 1) by construction. This is the whole point of using the logistic function — it squashes any real number into (0, 1).

---

**Q: How does logistic regression handle class imbalance?**
A: By default, it doesn't — the model will be biased toward the majority class. Common fixes:
- **Class weights**: `class_weight='balanced'` in sklearn penalizes errors on the minority class more
- **Threshold adjustment**: Use a lower threshold than 0.5 for predicting the minority class
- **Resampling**: Oversample (SMOTE) the minority class or undersample the majority
- **Use AUC-ROC instead of accuracy** as the evaluation metric

---

**Q: What is the null model / baseline?**
A: The model that predicts the same probability for every observation — specifically, the proportion of positive examples in the training data. If 30% of your data is class 1, the null model predicts p=0.3 for everyone. A good logistic regression model must substantially outperform this.

---

**Q: Does logistic regression require feature scaling?**
A: Technically, no — the math works regardless. But in practice: (1) Gradient descent **converges much faster** with scaled features (otherwise the loss surface is elongated), (2) Regularization terms affect all weights equally — without scaling, features on different scales are regularized differently. Always scale for gradient-based fitting with regularization.

---

**Q: What happens if two features are perfectly correlated?**
A: The weight matrix becomes singular (non-invertible). The coefficients are **not uniquely defined** — infinitely many β combinations give the same likelihood. Gradient descent may converge to different solutions depending on initialization, and coefficients are highly unstable. L2 regularization resolves this (adds λI to the Hessian, ensuring invertibility).

---

**Q: Why can't we use ordinary least squares (OLS) for logistic regression like we can for linear regression?**
A: Because the loss function (cross-entropy) doesn't have a closed-form solution. In linear regression, setting ∂J/∂β = 0 gives β = (XᵀX)⁻¹Xᵀy analytically. For logistic regression, the gradient equation is nonlinear in β (due to the sigmoid), so we must use iterative optimization (gradient descent, Newton-Raphson, IRLS).

---

**Q: Is the log-loss always convex for logistic regression?**
A: Yes, for the standard logistic regression model (without any non-convex regularization). The Hessian is positive semi-definite everywhere, so there's a unique global minimum. This is one of logistic regression's key advantages over more complex models.

---

### 🔴 Technical / Implementation Questions

**Q: What solvers does sklearn use and when should I choose which?**

| Solver | Best For | Notes |
|---|---|---|
| `lbfgs` | Small-medium datasets, L2 or no regularization | Default, good general choice |
| `liblinear` | Small datasets, L1 regularization, binary | Fast, coordinate descent |
| `sag` | Large datasets | Stochastic average gradient |
| `saga` | Large datasets, L1/ElasticNet | Faster than sag, handles L1 |
| `newton-cg` | Medium datasets, L2 | Full Newton step |

---

**Q: What does `max_iter` control in sklearn and what happens if I hit the limit?**
A: It controls the maximum number of optimization iterations. If the solver hasn't converged, sklearn throws a `ConvergenceWarning`. Fixes: increase `max_iter`, scale your features (faster convergence), increase regularization, or change the solver.

---

**Q: How do I interpret the output of `predict_proba` vs `predict` in sklearn?**
```python
model.predict_proba(X)  # Returns [[P(y=0), P(y=1)], ...] for each sample
model.predict(X)        # Returns [0, 1, 1, 0, ...] — applies 0.5 threshold
```

`predict_proba` gives you the raw probabilities; `predict` applies the default 0.5 threshold. For custom thresholds:
```python
probs = model.predict_proba(X)[:, 1]  # P(y=1)
preds = (probs >= 0.3).astype(int)    # Custom threshold of 0.3
```

---

**Q: What is the `C` parameter in sklearn's LogisticRegression?**
A: C = 1/λ, where λ is the regularization strength. Smaller C = more regularization (more penalty on large weights). Default is C=1.0. Tune via cross-validation.

---

**Q: What does `fit_intercept=False` do?**
A: Forces β₀ = 0, meaning the decision boundary passes through the origin. Use only if your data is already centered and you have a theoretical reason to exclude the intercept. Usually leave it as True.

---

**Q: Can logistic regression model nonlinear decision boundaries?**
A: Not directly, but yes indirectly via **feature engineering**. Add polynomial features (x², x₁·x₂, etc.) and logistic regression will find a nonlinear boundary in original space. For automated nonlinearity, use kernel logistic regression or switch to a neural network / tree-based model.

---

**Q: How does logistic regression compare to SVM?**

| Aspect | Logistic Regression | SVM |
|---|---|---|
| Output | Probabilities | Decision scores (need Platt scaling for probs) |
| Loss function | Log-loss | Hinge loss |
| Decision boundary | All points influence it | Only support vectors |
| Outlier sensitivity | More sensitive | Less sensitive (margin-based) |
| Kernel trick | Yes (kernel LR) | Yes (SVM is the standard) |
| Interpretability | High (coefficients = log-odds) | Lower |

---

**Q: How does logistic regression relate to neural networks?**
A: A logistic regression unit is literally a **single-layer neural network** with a sigmoid activation. Add more layers and neurons → deep neural network. The binary cross-entropy loss used to train logistic regression is the same loss used for the output layer of a neural network with binary output.

---

**Q: What is the relationship between logistic regression and information theory?**
A: The log-loss (cross-entropy) is the **expected cross-entropy** between the true distribution and the predicted distribution. Minimizing log-loss is equivalent to minimizing the KL divergence between the predicted Bernoulli(ŷ) and the true Bernoulli(y). It's the most "natural" loss for probability estimation in the information-theoretic sense.

---

### 🟡 Diagnostics and Troubleshooting

**Q: My model has 97% accuracy but terrible F1. What's happening?**
A: Classic class imbalance problem. If 97% of samples are class 0, a model that predicts class 0 for everything gets 97% accuracy. Use AUC-ROC, F1, or precision-recall instead. Apply class weighting or resampling.

---

**Q: My coefficients are extremely large (e.g., β = 500). Is something wrong?**
A: This typically means **perfect or near-perfect separation** — one or more features perfectly predict the outcome. The MLE doesn't exist (it's at infinity). Solutions: add L2 regularization (even small λ=0.01 is enough), use Firth's penalized regression, or remove the perfectly separating feature if it's data leakage.

---

**Q: Coefficients change drastically when I add a new feature. Why?**
A: Likely **multicollinearity** — the new feature is correlated with existing ones. The model redistributes weight across correlated features. Check VIF. Consider dimensionality reduction (PCA) or regularization (Ridge).

---

**Q: The model has high training accuracy but low test accuracy. What's wrong?**
A: **Overfitting**. The model memorized the training data. Solutions:
- Increase regularization (decrease C or increase λ)
- Remove features (feature selection)
- Get more training data
- Use cross-validation to properly tune hyperparameters

---

**Q: How do I test whether a feature significantly contributes to the model?**
A: Use the **Wald test**: for each coefficient βⱼ, compute:
```
z = βⱼ / SE(βⱼ)
```
Under H₀: βⱼ = 0, this follows a standard normal distribution. Large |z| → statistically significant. Equivalently, compare exp(βⱼ) (odds ratio) with its confidence interval — if CI excludes 1, the feature is significant.

---

**Q: What is deviance in logistic regression?**
A: Deviance is −2 × log-likelihood. It measures how far the model is from a perfect fit:
```
Null deviance: deviance of model with only intercept
Residual deviance: deviance of fitted model
```
A good model has much lower residual deviance than null deviance. The difference follows a chi-square distribution → use for model comparison (likelihood ratio test).

---

**Q: How do I compare two logistic regression models?**
A: Use the **Likelihood Ratio Test**:
```
LRT = 2·[ℓ(full model) − ℓ(reduced model)] ~ χ²(df = difference in params)
```
Also: AIC = −2ℓ + 2k (lower is better), BIC = −2ℓ + k·ln(m) (lower is better, penalizes complexity more than AIC)

---

### 🟢 Statistical Inference Questions

**Q: How do I get confidence intervals for the coefficients?**

From the information matrix (inverse Hessian of log-likelihood):
```
95% CI for βⱼ: βⱼ ± 1.96 · SE(βⱼ)
95% CI for odds ratio: [exp(βⱼ − 1.96·SE), exp(βⱼ + 1.96·SE)]
```
In sklearn, use `statsmodels.Logit` which provides full statistical output.

---

**Q: What are the standard errors of the logistic regression coefficients?**
A: They come from the **Fisher information matrix** (expected Hessian of the negative log-likelihood):
```
I(β) = Xᵀ · diag(σ(z)(1−σ(z))) · X

Cov(β̂) = I(β)⁻¹
SE(βⱼ) = √[I(β)⁻¹]ⱼⱼ
```

---

**Q: What is an odds ratio and how do I compute it from the model?**
A: The odds ratio (OR) for feature j is simply **e^βⱼ**:
```
OR = e^βⱼ

If βⱼ = 0.5:
OR = e^0.5 = 1.649
"A one-unit increase in xⱼ is associated with 64.9% higher odds of y=1"

If βⱼ = −0.3:
OR = e^(−0.3) = 0.741
"A one-unit increase in xⱼ is associated with 25.9% lower odds of y=1"
```

---

**Q: How do I compute a predicted probability with a confidence interval?**
A: Compute CI for z = βᵀx, then transform to probability scale:
```
SE(z) = √(xᵀ · Cov(β̂) · x)
95% CI for z: [z − 1.96·SE(z), z + 1.96·SE(z)]
95% CI for p: [σ(z − 1.96·SE(z)), σ(z + 1.96·SE(z))]
```

---

### 🔵 Advanced / Edge Case Questions

**Q: What is conditional logistic regression?**
A: Used when data has a **matched case-control** structure (e.g., each patient paired with a control). It accounts for the matching by conditioning on sufficient statistics, eliminating case-specific intercepts. Useful in epidemiology.

---

**Q: What is penalized (Firth's) logistic regression?**
A: Modifies the log-likelihood by adding a Jeffreys prior penalty:
```
ℓ_firth(β) = ℓ(β) + (1/2) log|I(β)|
```
This prevents infinite coefficient estimates under perfect separation and reduces small-sample bias. Widely used in medical studies with rare outcomes.

---

**Q: Can logistic regression be trained online (incremental learning)?**
A: Yes, using SGD variants. sklearn's `SGDClassifier(loss='log_loss')` supports `partial_fit()` for online learning. This is useful when data doesn't fit in memory.

---

**Q: What is the relationship between logistic regression and naive Bayes?**
A: When features are independent given the class (the Naive Bayes assumption), and features have Gaussian distributions, Gaussian Naive Bayes produces a decision boundary that is **equivalent to logistic regression**. They're "generative-discriminative pairs." Logistic regression (discriminative) typically outperforms Naive Bayes when the independence assumption is violated — which is usually.

---

**Q: What is a mixed-effects logistic regression?**
A: Extends standard logistic regression to handle **clustered or hierarchical data** (e.g., students within schools, patients within hospitals). Adds random effects β_group ~ N(0, σ²) to account for group-level variability.

---

**Q: How does logistic regression relate to maximum entropy models in NLP?**
A: Logistic regression and maximum entropy (MaxEnt) models are **identical**. MaxEnt is the name used in NLP/machine learning communities; logistic regression is the name in statistics. Both model P(y|x) as a log-linear model and use MLE.

---

## 📎 Summary Cheat Sheet

```
MODEL:
  z = β₀ + β₁x₁ + ... + βₙxₙ      (linear score)
  P(y=1|x) = σ(z) = 1/(1+e⁻ᶻ)     (sigmoid)
  logit(p) = z                       (log-odds = linear)

COEFFICIENTS:
  βⱼ > 0 → xⱼ increases P(y=1)
  βⱼ < 0 → xⱼ decreases P(y=1)
  Odds ratio = e^βⱼ

DECISION:
  ŷ = 1 if σ(z) ≥ 0.5  ↔  z ≥ 0

LOSS:
  J = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]   (cross-entropy)

GRADIENT:
  ∂J/∂βⱼ = (1/m) Σ (σ(z)−y)·xⱼ

UPDATE:
  βⱼ := βⱼ − α·∂J/∂βⱼ

REGULARIZED:
  Ridge: J + (λ/2m)Σβⱼ²    → shrinks all weights
  Lasso: J + (λ/m)Σ|βⱼ|   → sparse weights (feature selection)

EVALUATION:
  Accuracy, Precision, Recall, F1, AUC-ROC, Log-loss

ASSUMPTIONS:
  ✓ Binary outcome
  ✓ Independence of obs
  ✓ No severe multicollinearity
  ✓ Log-odds linear in features
  ✓ No extreme outliers
  ✓ Sufficient sample size (10+ events per predictor)
```

---

*Built ground-up: Probability → Odds → Log-Odds → Logit Model → Sigmoid → MLE → Gradient Descent → GLM → Full inference and evaluation.*
