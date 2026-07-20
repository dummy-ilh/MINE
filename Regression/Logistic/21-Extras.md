# Prior, Likelihood, Posterior & MLE

---

## 1. The three quantities

| Term | Notation | Question it answers |
|---|---|---|
| **Prior** | P(θ) | What do I believe about θ *before* seeing data? |
| **Likelihood** | P(X \| θ) | How well does this data fit, *given* θ? |
| **Posterior** | P(θ \| X) | What do I believe about θ *after* seeing data? |

---

## 2. Bayes' theorem

$$P(\theta \mid X) = \frac{P(X \mid \theta) \cdot P(\theta)}{P(X)}$$

Since P(X) is just a normalising constant (doesn't depend on θ):

```
posterior  ∝  likelihood × prior
```

The prior is your belief. The likelihood is the update signal from data. The posterior is the result.

---

## 3. What "likelihood" actually means

P(X | θ) looks like a conditional probability over X — but in likelihood thinking, **X is fixed** (you already observed it) and **θ varies**. You're asking: "for each possible θ, how probable is this exact dataset?"

- Not a distribution over X
- A scoring function over θ
- Written L(θ ; X) when you want to emphasise this

---

## 4. Concrete example — biased coin

You flip a coin 10 times and get 8 heads. θ = true probability of heads.

```
Prior:      P(θ) = Uniform[0,1]   # no prior knowledge
Likelihood: P(X | θ) = C(10,8) · θ⁸ · (1-θ)²
Posterior:  P(θ | X) ∝ θ⁸ · (1-θ)²   # peaks near θ = 0.8
```

The prior is flat, so the posterior just follows the likelihood here. If the prior had been peaked at 0.5 (strong belief in fairness), the posterior would land somewhere between 0.5 and 0.8.

---

## 5. MLE — Maximum Likelihood Estimation

**Throw the prior away. Find the θ that makes the data most probable.**

```
θ̂_MLE = argmax_θ  P(X | θ)
       = argmax_θ  L(θ ; X)
```

For the coin: MLE gives θ̂ = 0.8 exactly (8/10 heads).

### Why log-likelihood?

Likelihoods are products of many small probabilities → numerical underflow. Log turns products into sums, and argmax is preserved under monotone transforms.

```
ℓ(θ) = log L(θ ; X) = Σᵢ log P(xᵢ | θ)

θ̂_MLE = argmax_θ  ℓ(θ)
```

---

## 6. MAP — Maximum A Posteriori

**Keep the prior. Find the peak of the posterior instead.**

```
θ̂_MAP = argmax_θ  P(θ | X)
       = argmax_θ  log P(X | θ) + log P(θ)
       = MLE objective  +  log prior term
```

| Prior on θ | Equivalent regularisation |
|---|---|
| Gaussian N(0, σ²) | L2 (Ridge) |
| Laplace | L1 (Lasso) |
| Uniform | No regularisation = MLE |

Regularisation isn't a hack — it's the log prior in MAP estimation.

---

## 7. MLE for logistic regression — from scratch

### Setup

- Binary labels yᵢ ∈ {0, 1}, features xᵢ ∈ ℝᵈ, parameters w
- Model: output a probability via sigmoid

```
σ(z) = 1 / (1 + e^(−z))

p̂ᵢ = σ(wᵀxᵢ)
```

### Step 1 — write the likelihood

Each label is Bernoulli(p̂ᵢ). For one sample:

```
P(yᵢ | xᵢ, w) = p̂ᵢ^yᵢ · (1 − p̂ᵢ)^(1−yᵢ)
```

Joint likelihood over all n i.i.d. samples:

```
L(w) = ∏ᵢ  p̂ᵢ^yᵢ · (1 − p̂ᵢ)^(1−yᵢ)
```

### Step 2 — take the log-likelihood

```
ℓ(w) = Σᵢ [ yᵢ · log(p̂ᵢ)  +  (1−yᵢ) · log(1 − p̂ᵢ) ]
```

This is the **negative** of binary cross-entropy loss.  
Maximising ℓ(w) = minimising cross-entropy. They're the same thing.

### Step 3 — gradient

Using the fact that σ'(z) = σ(z)(1 − σ(z)):

```
∂ℓ/∂w = Σᵢ (yᵢ − p̂ᵢ) · xᵢ

In matrix form:
∇_w ℓ = Xᵀ(y − p̂)
```

Gradient = residuals (yᵢ − p̂ᵢ) weighted by features. Same structure as linear regression — both are GLMs.

### Step 4 — no closed form → gradient ascent

Setting ∇ℓ = 0 has no closed-form solution because p̂ᵢ = σ(wᵀxᵢ) is nonlinear in w.

```
w ← w + α · Xᵀ(y − p̂)    # gradient ascent on ℓ
w ← w − α · Xᵀ(p̂ − y)    # gradient descent on loss
```

### Step 5 — Hessian (why convergence is guaranteed)

```
H = −Xᵀ W X

where W = diag(p̂ᵢ(1 − p̂ᵢ))
```

H is negative semi-definite → ℓ is **concave** → unique global maximum → the solver always converges.

### Full derivation in one view

```
1.  Model:      p̂ᵢ = σ(wᵀxᵢ)
2.  Likelihood: L = ∏ᵢ p̂ᵢ^yᵢ (1−p̂ᵢ)^(1−yᵢ)
3.  Log-lik:    ℓ = Σᵢ [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]
4.  Gradient:   ∇_w ℓ = Xᵀ(y − p̂)
5.  No closed form → gradient ascent
6.  ℓ is concave → unique global max
```

---

## 8. Python

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, w):
    p = sigmoid(X @ w)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradient(X, y, w):
    p = sigmoid(X @ w)
    return X.T @ (y - p)   # shape (d,)

def fit(X, y, lr=0.1, iters=1000):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        w += lr * gradient(X, y, w)
    return w
```

---

## 9. Interview punchlines

**"Why cross-entropy loss?"**  
It's the negative log-likelihood under a Bernoulli model. Not arbitrary — it's the principled MLE objective.

**"What's the difference between MLE and MAP?"**  
MAP adds a log prior term. With a Gaussian prior this equals L2 regularisation; with Laplace, L1.


# Logistic Regression — Deep Topics

---

## 1. MLE Hand Calculation (tiny example)

Two training points, one feature, no bias for simplicity:

```
x₁ = 1,  y₁ = 1
x₂ = 2,  y₂ = 0
```

Model: p̂ᵢ = σ(w · xᵢ). We want to find w by MLE.

### Write the log-likelihood

```
ℓ(w) = y₁·log(p̂₁) + (1−y₁)·log(1−p̂₁)
      + y₂·log(p̂₂) + (1−y₂)·log(1−p̂₂)

     = 1·log(σ(w))  + 0
     + 0             + 1·log(1 − σ(2w))

     = log(σ(w))  +  log(σ(−2w))
```

Note: log(1 − σ(z)) = log(σ(−z)), a useful identity.

### Set up the gradient

```
dℓ/dw = (1 − σ(w))·1  +  (0 − σ(2w))·2
       = (y₁ − p̂₁)·x₁  +  (y₂ − p̂₂)·x₂
```

This is just ∇ℓ = Xᵀ(y − p̂) for our 2-point case.

### Iterate (gradient ascent, lr = 0.5)

**Init: w = 0**

```
p̂₁ = σ(0·1) = 0.5
p̂₂ = σ(0·2) = 0.5

grad = (1−0.5)·1 + (0−0.5)·2 = 0.5 − 1.0 = −0.5

w ← 0 + 0.5·(−0.5) = −0.25
```

**Iter 2: w = −0.25**

```
p̂₁ = σ(−0.25) ≈ 0.438
p̂₂ = σ(−0.50) ≈ 0.378

grad = (1−0.438)·1 + (0−0.378)·2
     = 0.562 − 0.756 = −0.194

w ← −0.25 + 0.5·(−0.194) = −0.347
```

w keeps moving negative — makes sense. x₂=2 has label 0, so the model learns to push high-x predictions toward 0, which means negative w.

As iterations → ∞, w → −∞ (perfect separation is achievable here, so MLE is unbounded — a sign you need regularisation).

### Why no closed form?

Setting dℓ/dw = 0:

```
(1 − σ(w)) − 2·σ(2w) = 0
```

σ is nonlinear — you can't isolate w algebraically. Hence: numerical optimisation only.

---

## 2. Why Sigmoid?

Three independent reasons, each sufficient on its own.

### Reason 1 — Probabilistic derivation (the real reason)

Assume the log-odds of the event is linear in x:

```
log(p / (1−p)) = wᵀx
```

Solve for p:

```
p / (1−p) = e^(wᵀx)
p = e^(wᵀx) · (1−p)
p = e^(wᵀx) − p·e^(wᵀx)
p(1 + e^(wᵀx)) = e^(wᵀx)
p = e^(wᵀx) / (1 + e^(wᵀx))
p = 1 / (1 + e^(−wᵀx))
```

That's σ. It didn't come from nowhere — it's the **inverse of the logit function**. If you believe log-odds is linear, sigmoid is the forced consequence.

### Reason 2 — Bernoulli exponential family

The Bernoulli distribution's natural parameter IS the log-odds. The sigmoid is the mean function of the Bernoulli in exponential family form. Logistic regression is a GLM with the canonical link — sigmoid emerges from the probability model, not from a design choice.

### Reason 3 — Output range and gradient

- Output ∈ (0, 1): valid probability, no clipping needed
- Differentiable everywhere
- σ'(z) = σ(z)(1 − σ(z)): derivative expressed in terms of itself → cheap to compute in backprop

---

## 3. Sigmoid vs Other Activation Functions

| Property | Sigmoid | Tanh | ReLU | Softmax |
|---|---|---|---|---|
| Output range | (0, 1) | (−1, 1) | [0, ∞) | (0,1), sums to 1 |
| Centred at 0? | No | Yes | No | No |
| Vanishing gradient? | Yes (tails) | Yes (tails) | No (positive side) | Yes (tails) |
| Use case | Binary output | Hidden layers (old) | Hidden layers (modern) | Multiclass output |

### Vanishing gradient problem with sigmoid

σ'(z) = σ(z)(1 − σ(z))

Maximum value is 0.25 (at z=0). In deep networks, backprop multiplies gradients through many layers — a chain of numbers ≤ 0.25 goes to zero exponentially fast. Tanh has the same problem (max derivative = 1.0, but still squashes).

ReLU fixes this: derivative is 1 for z > 0, so gradients don't shrink. But ReLU outputs aren't probabilities, so it can't be the final layer for classification.

### Why sigmoid for output, not hidden layers?

In hidden layers: sigmoid/tanh are mostly replaced by ReLU and variants (better gradient flow).  
At the output layer: sigmoid is correct for binary classification because we need P(y=1|x) ∈ (0,1).

### Sigmoid vs Softmax

Softmax is the multi-class generalisation of sigmoid. For K=2 classes, softmax reduces exactly to sigmoid. For K>2, softmax gives a probability distribution over all classes that sums to 1 and allows class scores to compete against each other.

```
softmax(zₖ) = e^zₖ / Σⱼ e^zⱼ

For K=2: softmax(z₁) = e^z₁ / (e^z₁ + e^z₂)
                      = 1 / (1 + e^(z₂−z₁))
                      = σ(z₁ − z₂)   ← sigmoid
```

---

## 4. Why Not MSE for Classification?

### Reason 1 — Wrong probability model

MSE assumes the noise around predictions is Gaussian:

```
y = wᵀx + ε,   ε ~ N(0, σ²)
```

Binary labels are Bernoulli — they can only be 0 or 1. Gaussian noise on a binary variable makes no probabilistic sense. Cross-entropy is the correct loss because it's the NLL of the Bernoulli model.

### Reason 2 — Non-convexity with sigmoid

With sigmoid outputs, the MSE loss is:

```
L = (1/n) Σᵢ (yᵢ − σ(wᵀxᵢ))²
```

The Hessian of this is not guaranteed to be positive semi-definite. The loss surface has local minima — gradient descent can get stuck. Cross-entropy + sigmoid gives a convex surface (proven in section 7).

### Reason 3 — Flat gradients (the practical killer)

MSE gradient w.r.t. output:

```
∂L/∂ŷ = 2(ŷ − y)
```

Chain rule through sigmoid:

```
∂L/∂w = 2(ŷ − y) · σ'(z) · x
       = 2(ŷ − y) · σ(z)(1−σ(z)) · x
```

When the model is very wrong — say y=1 but ŷ≈0 — σ(z)≈0 so σ'(z)≈0. The gradient vanishes precisely when it should be largest. Learning slows to a crawl when the model makes confident wrong predictions.

With cross-entropy:

```
∂L/∂w = (ŷ − y) · x
```

The σ'(z) term cancels algebraically. The gradient is just the residual — large when wrong, small when right. This is why cross-entropy trains so much faster.

### Reason 4 — Output interpretation

MSE doesn't push predictions toward 0 or 1. A prediction of 0.5 for a clear positive example gets penalised equally to 0.4 or 0.6. Cross-entropy is asymmetric: it heavily penalises confident wrong predictions (−log(0.01) = 4.6 vs −log(0.5) = 0.69).

### Summary

| Issue | MSE | Cross-entropy |
|---|---|---|
| Probability model | Gaussian (wrong) | Bernoulli (correct) |
| Loss surface | Non-convex | Convex |
| Gradient when confident & wrong | Near zero | Large |
| Gradient formula | (ŷ−y)·σ'(z)·x | (ŷ−y)·x |

---

## 5. Decision Boundary Explained

### What it is

The decision boundary is the set of points where the model is exactly 50/50:

```
p̂ = σ(wᵀx + b) = 0.5
```

σ(z) = 0.5 exactly when z = 0. So the boundary is:

```
wᵀx + b = 0
```

That's a hyperplane. In 2D it's a line. In 3D it's a plane.

### Intuition

w is a vector perpendicular to the boundary. Points on the side where wᵀx + b > 0 get p̂ > 0.5 (class 1). Points where wᵀx + b < 0 get p̂ < 0.5 (class 0).

The magnitude of wᵀx + b tells you how far from the boundary you are — the further away, the more confident the model.

### Visualised in 2D

```
Feature 2
    |     class 1 (p̂ > 0.5)
    |   /
    |  /  ← decision boundary: w₁x₁ + w₂x₂ + b = 0
    | /
    |/   class 0 (p̂ < 0.5)
    +------------- Feature 1
```

### Probability contours

Far from the boundary: p̂ → 0 or p̂ → 1 (confident)
At the boundary: p̂ = 0.5 (uncertain)
The sigmoid squashes the linear distance into a probability — the "confidence" is a function of how far a point is from the hyperplane.

---

## 6. Can Logistic Regression Do Nonlinear Boundaries? Why Not Use Linear Regression?

### Why not linear regression for classification?

Linear regression predicts ŷ = wᵀx ∈ (−∞, ∞). For binary classification:

**Problem 1 — outputs aren't probabilities.** Nothing stops predictions from being −3 or 7. You can't threshold meaningfully.

**Problem 2 — MSE loss treats all errors equally.** A prediction of 2.0 for y=1 is penalised (residual = 1.0), even though it's "more right" than 0.5. The loss pushes extreme correct predictions back toward the mean.

**Problem 3 — decision boundary gets distorted by outliers.** One extreme positive example pulls the regression line toward it, shifting the boundary in the wrong direction.

Practically: linear regression on binary labels sometimes works okay (it approximates a probability under specific conditions), but it breaks theoretically and fails on imbalanced or extreme data.

### Can logistic regression do nonlinear boundaries?

The raw model: wᵀx + b = 0 is always a hyperplane. Logistic regression is a **linear classifier** in feature space.

But: you can add nonlinear features manually.

```
Original:  x₁, x₂
Add:       x₁², x₂², x₁x₂

New model: w₁x₁ + w₂x₂ + w₃x₁² + w₄x₂² + w₅x₁x₂ + b = 0
```

This is still linear in the *expanded* feature space but produces a quadratic (ellipse, parabola, hyperbola) boundary in original x-space.

The same idea powers kernel methods: implicitly map to a very high-dimensional feature space without computing it explicitly. SVMs do this with kernels; logistic regression does it by explicit feature engineering.

### What logistic regression cannot do

Without feature engineering: it can't separate XOR-structured data.

```
Class 0: (0,0), (1,1)
Class 1: (0,1), (1,0)
```

No straight line separates these. A neural network with a hidden layer can — it learns the feature transformation automatically. That's the fundamental advantage of deep learning over logistic regression.

---

## 7. Proof: Cross-Entropy + Sigmoid is Convex

A function is convex if its Hessian is positive semi-definite (PSD) everywhere.

### The loss

For n samples, the negative log-likelihood (= cross-entropy loss):

```
L(w) = −Σᵢ [ yᵢ log(p̂ᵢ) + (1−yᵢ) log(1−p̂ᵢ) ]

where p̂ᵢ = σ(wᵀxᵢ)
```

### Step 1 — compute the gradient

We showed earlier:

```
∇_w L = Σᵢ (p̂ᵢ − yᵢ) xᵢ  =  Xᵀ(p̂ − y)
```

### Step 2 — compute the Hessian

Differentiate ∇_w L with respect to w:

```
∂/∂w [(p̂ᵢ − yᵢ) xᵢ] = (∂p̂ᵢ/∂w) xᵢ
                       = σ'(wᵀxᵢ) xᵢ xᵢᵀ
                       = p̂ᵢ(1−p̂ᵢ) xᵢ xᵢᵀ
```

Summing over all samples:

```
H = Σᵢ p̂ᵢ(1−p̂ᵢ) xᵢ xᵢᵀ

In matrix form:
H = Xᵀ W X

where W = diag(p̂ᵢ(1−p̂ᵢ))   [n×n diagonal matrix]
```

### Step 3 — show H is PSD

For any vector v ∈ ℝᵈ:

```
vᵀ H v = vᵀ (Xᵀ W X) v
        = (Xv)ᵀ W (Xv)
        = Σᵢ wᵢᵢ (xᵢᵀv)²
        = Σᵢ p̂ᵢ(1−p̂ᵢ) · (xᵢᵀv)²
```

Now:
- p̂ᵢ ∈ (0, 1) because σ outputs strictly between 0 and 1
- Therefore p̂ᵢ(1−p̂ᵢ) > 0 always
- (xᵢᵀv)² ≥ 0 always

So every term in the sum is ≥ 0, meaning:

```
vᵀ H v ≥ 0   for all v
```

**H is positive semi-definite. Therefore L(w) is convex. QED.**

### What this guarantees

- No local minima — any local minimum is the global minimum
- Gradient descent always converges to the same solution regardless of initialisation
- The solution is unique (assuming X has full column rank, in which case H is PD, not just PSD)

### Why MSE + sigmoid is NOT convex

With MSE:

```
L_MSE = Σᵢ (yᵢ − σ(wᵀxᵢ))²
```

The Hessian involves second derivatives of σ (the σ''(z) terms), which can be negative. You can construct cases where the Hessian has negative eigenvalues — i.e., the loss surface has saddle points and local minima. No such clean algebraic cancellation as in the cross-entropy case.

---

## Quick Reference

```
Model:         p̂ = σ(wᵀx + b) = 1 / (1 + e^(−wᵀx−b))
Loss:          L = −Σᵢ [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]
Gradient:      ∇L = Xᵀ(p̂ − y)
Hessian:       H = XᵀWX,  W = diag(p̂ᵢ(1−p̂ᵢ))
Update:        w ← w − α · Xᵀ(p̂ − y)
Boundary:      wᵀx + b = 0
Convexity:     vᵀHv = Σᵢ p̂ᵢ(1−p̂ᵢ)(xᵢᵀv)² ≥ 0  ✓
```

# Logistic Regression — Pitfalls & Comparisons

---

## 1. Common Mistakes & Pitfalls

### Pitfall 1 — Not scaling features

Logistic regression is sensitive to feature scale. If x₁ ∈ [0, 1] and x₂ ∈ [0, 1,000,000], the gradient updates for w₁ and w₂ are on completely different scales. Training is slow and unstable.

**Fix:** standardise features (zero mean, unit variance) before training. Always.

```
x_scaled = (x − mean(x)) / std(x)
```

### Pitfall 2 — Perfect separation (complete/quasi-complete)

If a feature perfectly separates the classes, MLE doesn't converge — the optimal w is ±∞. The likelihood keeps increasing as |w| grows. sklearn will converge numerically but the coefficients are meaningless.

```
x:  1  2  3  4  5  6
y:  0  0  0  1  1  1
```

x=3.5 perfectly separates. MLE: w → ∞.

**Signs:** very large coefficients, huge standard errors, warnings about convergence.  
**Fix:** L2 regularisation (always), or reduce the separating feature.

### Pitfall 3 — Multicollinearity

Highly correlated features make coefficient estimates unstable — small changes in data cause large swings in w. The model still predicts okay, but coefficients can't be interpreted as feature importances.

**Fix:** drop one of the correlated features, PCA, or L2 regularisation (shrinks correlated coefficients together).

### Pitfall 4 — Ignoring class imbalance

If 95% of labels are 0, predicting 0 always gives 95% accuracy. The model learns the majority class. Cross-entropy loss treats all samples equally by default.

**Fixes:**
- `class_weight='balanced'` in sklearn (upweights minority class in loss)
- Oversample minority (SMOTE) or undersample majority
- Evaluate with F1, AUC-ROC, precision-recall — not accuracy

### Pitfall 5 — Interpreting coefficients without odds ratios

Raw coefficients wⱼ are in log-odds space. A coefficient of 0.7 doesn't mean "70% more likely."

```
Correct interpretation:
  wⱼ = 0.7  →  odds ratio = e^0.7 ≈ 2.01
  → a 1-unit increase in xⱼ doubles the odds of the event
```

Saying "the coefficient is 0.7 so the feature increases probability by 70%" is wrong. The actual probability change depends on the base rate.

### Pitfall 6 — Not checking linearity in log-odds

Logistic regression assumes the log-odds is **linear** in each feature. If the true relationship is quadratic (e.g. risk is high for very low and very high values), the model silently fits the wrong shape.

**Check:** plot feature vs log-odds empirically (binning trick). Add polynomial features if needed.

### Pitfall 7 — Using predicted probabilities as calibrated without checking

Sklearn's logistic regression outputs probabilities, but they can be miscalibrated — especially with regularisation or imbalanced data. A predicted 0.8 might not mean 80% of events happen at that score.

**Fix:** plot calibration curves (reliability diagrams). Use `CalibratedClassifierCV` if needed.

### Pitfall 8 — Threshold at 0.5 by default

0.5 is only optimal when misclassification costs are equal and classes are balanced. In fraud detection, cancer screening, etc., you want to tune the threshold on a validation set to match the cost asymmetry.

**Fix:** plot the precision-recall curve, pick threshold based on business cost.

---

## 2. Logistic Regression vs Linear Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| Output | ŷ ∈ (−∞, ∞) | p̂ ∈ (0, 1) |
| Target variable | Continuous | Binary (or probability) |
| Link function | Identity | Logit |
| Loss | MSE | Cross-entropy (NLL) |
| Loss surface | Convex (always) | Convex (always) |
| Closed-form solution | Yes (normal equations) | No |
| Assumptions | Gaussian noise | Bernoulli noise |
| Coefficients mean | Unit change in y per unit xⱼ | Unit change in log-odds per unit xⱼ |

### They're the same model structure

Both are:

```
output = wᵀx + b   (linear combination of features)
```

The difference is what you do with that output:

```
Linear:   ŷ = wᵀx + b             (use directly)
Logistic: p̂ = σ(wᵀx + b)         (pass through sigmoid)
```

Both are **Generalised Linear Models (GLMs)**. Linear regression uses the identity link; logistic regression uses the logit link.

### Why linear regression fails on binary targets

**Unbounded outputs.** Nothing constrains predictions to [0,1]. A valid "probability" of −0.3 or 1.4 is meaningless.

**MSE distorts the boundary.** Outlier positive examples pull the regression line toward them, shifting the decision threshold incorrectly. Example:

```
Without outlier: threshold at x = 3
Add x = 100, y = 1: line tilts, threshold shifts to x = 5
Datapoints at x = 4 now misclassified
```

**Wrong noise model.** MSE derives from assuming Gaussian residuals. Binary labels are Bernoulli — the noise model is fundamentally different.

**When linear regression accidentally works:** with balanced classes, no extreme outliers, and predictions that happen to stay in [0,1], linear regression approximates logistic regression (the LPM — Linear Probability Model — in econometrics). But this is coincidence, not correctness.

---

## 3. Logistic Regression vs Naive Bayes

### The core difference — direction of modelling

```
Logistic Regression:   models P(y | x)   directly   [discriminative]
Naive Bayes:           models P(x | y)   then inverts via Bayes  [generative]
```

Logistic regression asks: "given these features, what's the label probability?"  
Naive Bayes asks: "given each class, how likely are these features?" — then uses Bayes to flip it.

### Naive Bayes assumption

Features are conditionally independent given the class:

```
P(x | y) = P(x₁ | y) · P(x₂ | y) · ... · P(xd | y)
```

This is almost always false in practice (words in text are not independent, vitals in medicine are not independent). Yet NB often works anyway.

### Why does NB work despite the wrong assumption?

The decision boundary only requires that the ranking of P(y=1|x) vs P(y=0|x) is correct — not the exact probability values. The independence assumption distorts probabilities but often preserves the correct ordering.

### Mathematical relationship

With Gaussian NB (features ~ Gaussian given class), the log-odds is:

```
log P(y=1|x) / P(y=0|x) = wᵀx + b
```

— which is exactly logistic regression's model form. Gaussian NB and logistic regression share the same decision boundary shape. The difference is how they estimate w: NB estimates class-conditional distributions separately; LR estimates w directly to maximise P(y|x).

### When to use which

| Situation | Prefer |
|---|---|
| Lots of data | Logistic Regression |
| Little data | Naive Bayes (fewer parameters, less overfitting) |
| Features truly independent | Naive Bayes |
| Need calibrated probabilities | Logistic Regression |
| Text classification (bag of words) | Naive Bayes (Multinomial NB) works well |
| Features highly correlated | Logistic Regression |
| Need to generate synthetic data | Naive Bayes (it's generative) |

### Key insight

Logistic regression is asymptotically better — with infinite data it will outperform NB because it makes fewer assumptions. But NB can win with small data because its strong priors (the independence assumption acts like regularisation) reduce variance. This is the **bias-variance tradeoff** made explicit.

---

## 4. Logistic Regression vs Decision Trees

### Fundamental difference

```
Logistic Regression:  finds a hyperplane in feature space
Decision Tree:        partitions feature space into axis-aligned rectangles
```

LR: one global linear boundary.  
Tree: many local rules, each applying to a region.

### Decision boundary shape

```
Logistic Regression:         Decision Tree:
                             
  x₂ |   /                    x₂ |  |     |
      |  / ← line             ----+--+-----+--
      | /                         |     |
      |/                    ------+--+--+
      +------ x₁                 |  |
                                  +------ x₁
```

LR draws one diagonal line. A tree draws horizontal/vertical cuts. Trees can approximate any boundary with enough depth but need exponentially many leaves for diagonal or curved boundaries. LR gets diagonal boundaries for free.

### Assumptions

**Logistic Regression:**
- Log-odds is linear in features
- Features should be scaled
- Sensitive to outliers
- Requires explicit feature engineering for nonlinear boundaries

**Decision Trees:**
- No distributional assumptions
- Invariant to monotone feature transforms (scaling irrelevant)
- Handles nonlinear boundaries natively
- Handles missing values natively (some implementations)

### Interpretability

Both are interpretable, but differently:

```
LR:   "a 1-unit increase in age multiplies the odds by e^0.3 = 1.35"
Tree: "if age > 40 AND income < 50k → predict churn"
```

Tree rules are easier for non-technical stakeholders. LR coefficients are better for statistical inference (confidence intervals, p-values, odds ratios).

### Overfitting

LR: regularise with L1/L2. Relatively stable — only d parameters.  
Tree: prone to overfitting (can memorise training data). Requires pruning or depth limit.


# Logistic Regression — sklearn, Imbalanced Data & Softmax

---

## 1. sklearn LogisticRegression — Every Parameter Explained

```python
from sklearn.linear_model import LogisticRegression

LogisticRegression(
    penalty='l2',
    dual=False,
    tol=1e-4,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver='lbfgs',
    max_iter=100,
    multi_class='auto',
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None
)
```

---

### `penalty` — what regularisation to apply

```
'l2'        Ridge: adds λ·Σwⱼ² to loss. Shrinks all weights, keeps all features.
'l1'        Lasso: adds λ·Σ|wⱼ| to loss. Drives some weights to exactly zero → feature selection.
'elasticnet' Combination of L1 and L2. Needs l1_ratio set.
None        No regularisation. MLE only. Will diverge on separable data.
```

**Which to use:**
- Default to `l2` unless you need sparsity
- `l1` when you have many features and suspect most are irrelevant
- `elasticnet` when you want sparsity but L1 alone is too aggressive

---

### `C` — regularisation strength (inverse of λ)

```
Loss = cross_entropy + (1/C) · penalty

Small C  →  strong regularisation  →  smaller weights  →  simpler model
Large C  →  weak regularisation   →  weights less constrained  →  closer to raw MLE
```

C = 1/λ. If you're used to λ notation: C = 0.01 means heavy regularisation, C = 100 means almost none.

**Default C=1.0 is arbitrary — always tune this with cross-validation.**

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LogisticRegression(), {'C': [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5)
```

---

### `solver` — optimisation algorithm

| Solver | Supports penalties | Best for |
|---|---|---|
| `lbfgs` | l2, None | Default. Good for most cases. Uses second-order info (quasi-Newton). |
| `liblinear` | l1, l2 | Small datasets. Only solver that supports l1 on small data well. |
| `saga` | l1, l2, elasticnet, None | Large datasets. Stochastic gradient. Only solver for elasticnet. |
| `sag` | l2, None | Large datasets. Faster than lbfgs on large n. |
| `newton-cg` | l2, None | Similar to lbfgs. |

**Rules:**
- Small data → `lbfgs` or `liblinear`
- Need L1 → `liblinear` (small) or `saga` (large)
- Need elasticnet → `saga` only
- Large dataset → `saga` or `sag`

---

### `max_iter` — maximum number of iterations

Default is 100. The most common warning in sklearn:

```
ConvergenceWarning: lbfgs failed to converge (status=1). 
Increase the number of iterations (max_iter).
```

This doesn't mean the model is wrong — it means the optimiser didn't reach `tol` within 100 steps.

**What to do:**
1. First, scale your features (unscaled features slow convergence dramatically)
2. Then increase max_iter: try 200, 500, 1000
3. If it still doesn't converge at 10000, check for perfect separation or extreme features

```python
# Step 1: always scale
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Step 2: increase iterations
lr = LogisticRegression(max_iter=1000)
```

---

### `tol` — convergence tolerance

The optimiser stops when the change in loss between iterations is less than `tol`. Default 1e-4.

- Smaller tol → more precise solution, more iterations
- Larger tol → faster, less precise
- Usually fine to leave at default. If convergence is slow, loosen to 1e-3 before increasing max_iter.

---

### `fit_intercept` — whether to include bias term b

```
True (default):  model = σ(wᵀx + b)    ← almost always correct
False:           model = σ(wᵀx)         ← only if data is already centred
```

Only set to False if you've manually centred your features (mean = 0 for all features). Otherwise always True.

---

### `class_weight` — handling imbalanced classes

```
None (default):    all samples weighted equally
'balanced':        weight class k by n_samples / (n_classes × n_k)
{0: 1, 1: 10}:    manual weights — class 1 counts 10× more in the loss
```

When `'balanced'`: minority class samples are upweighted so their loss contribution matches the majority class. Equivalent to oversampling the minority class.

Full explanation in section 2.

---

### `multi_class` — how to handle >2 classes

```
'auto':     uses 'ovr' for liblinear, 'multinomial' for others
'ovr':      One-vs-Rest. Trains K binary classifiers. Fast, less principled.
'multinomial': Softmax regression. Single model over all classes. More principled.
```

For multi-class, `'multinomial'` with `solver='lbfgs'` or `'saga'` is the right choice. Explained in section 3.

---

### `l1_ratio` — elasticnet mixing

Only used when `penalty='elasticnet'`:

```
Loss = cross_entropy 
     + (1/C) · [l1_ratio · Σ|wⱼ| + (1−l1_ratio) · Σwⱼ²]

l1_ratio = 0   →  pure L2
l1_ratio = 1   →  pure L1
l1_ratio = 0.5 →  equal mix
```

---

### `warm_start` — reuse previous fit

```
False (default): reinitialise weights each time fit() is called
True:            start from previous weights
```

Useful when tuning C across a grid — start the next C value from the previous solution (often nearby) instead of from scratch. Saves time.

---

### `random_state` — reproducibility

Only matters for `solver='sag'` and `solver='saga'` (stochastic solvers shuffle data). Set to any integer for reproducibility.

---

### `n_jobs` — parallelism

Only used in One-vs-Rest multi-class (trains K classifiers in parallel). `-1` = use all cores. No effect for binary classification.

---

### `dual` — dual vs primal formulation

Only for `liblinear` with L2. When n_samples > n_features, use `dual=False` (primal). When n_features > n_samples, `dual=True` can be faster. Almost always leave as default False.

---

## 2. Imbalanced Data — Full Process

### The problem

Suppose 95% of labels are 0 (not churn), 5% are 1 (churn).

The model minimises total cross-entropy loss. A sample from class 1 contributes the same to the loss as a sample from class 0. But there are 19× more class-0 samples, so the loss is dominated by getting class 0 right.

**Result:** the model learns to predict 0 almost always. Accuracy = 95%. But recall on class 1 = near 0. Useless for the actual task.

### What `class_weight='balanced'` does — the math

sklearn computes a weight for each sample:

```
wₖ = n_samples / (n_classes × nₖ)

where nₖ = number of samples in class k
```

For our 95/5 split with 1000 samples:

```
n_samples = 1000,  n_classes = 2
n₀ = 950,  n₁ = 50

w₀ = 1000 / (2 × 950) = 0.526
w₁ = 1000 / (2 × 50)  = 10.0
```

Each class-1 sample now contributes 10/0.526 ≈ 19× more to the loss than a class-0 sample. This exactly cancels the class imbalance.

**The modified loss:**

```
L = −Σᵢ  sampleweight(i) · [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]
```

Where sampleweight(i) = wₖ for sample i's class k.

**Gradient:**

```
∇_w L = Xᵀ D (y − p̂)

where D = diag(sampleweight(i))
```

The minority class residuals are amplified in the gradient → the decision boundary shifts to correctly classify minority class samples even at the cost of more class-0 errors.

### What happens to the decision boundary

Without weighting:

```
boundary sits close to the minority class
→ almost everything predicted as majority
```

With class_weight='balanced':

```
boundary shifts toward the majority class
→ more minority correctly classified
→ some majority now misclassified (acceptable tradeoff)
```

This is equivalent to moving the classification threshold below 0.5.

### The full toolkit for imbalanced data

**Level 1 — change the loss (class_weight):**

```python
lr = LogisticRegression(class_weight='balanced')
# or manual:
lr = LogisticRegression(class_weight={0: 1, 1: 19})
```

**Level 2 — resample the data:**

```python
# Oversample minority (SMOTE — creates synthetic minority samples)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X_train, y_train)

# Undersample majority (lose data)
from imblearn.under_sampling import RandomUnderSampler
X_res, y_res = RandomUnderSampler().fit_resample(X_train, y_train)
```

SMOTE creates new synthetic minority samples by interpolating between existing ones in feature space — it doesn't just duplicate.

**Level 3 — tune the threshold:**

Default threshold is 0.5. For imbalanced data this is almost always wrong.

```python
lr = LogisticRegression()
lr.fit(X_train, y_train)
probs = lr.predict_proba(X_val)[:, 1]

# Find threshold that maximises F1
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.01)
f1s = [f1_score(y_val, probs > t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1s)]

preds = (probs > best_threshold).astype(int)
```

**Level 4 — use the right metrics:**

```python
from sklearn.metrics import classification_report, roc_auc_score

# Never use accuracy alone
print(classification_report(y_test, preds))  # precision, recall, F1 per class
print("AUC-ROC:", roc_auc_score(y_test, probs))
# Also: average_precision_score (AUC-PR) is better than AUC-ROC for heavy imbalance
```

### When to use what

| Imbalance ratio | Recommended approach |
|---|---|
| 70/30 | Usually fine, maybe class_weight='balanced' |
| 90/10 | class_weight='balanced' + tune threshold |
| 95/5 | class_weight + SMOTE + tune threshold + AUC-PR metric |
| 99/1 | All of the above, consider anomaly detection instead |

---

## 3. Softmax — How It Works Under the Hood

### What problem softmax solves

Binary classification: one output → sigmoid → P(y=1|x).

Multi-class (K classes): K outputs. Need them to:
1. Each be in (0, 1)
2. Sum to exactly 1 (valid probability distribution)

Sigmoid on each independently fails condition 2 — K sigmoids don't sum to 1.

### The softmax function

Given K raw scores (logits) z₁, z₂, ..., zK:

```
softmax(zₖ) = e^zₖ / Σⱼ e^zⱼ

for k = 1, 2, ..., K
```

**Properties:**
- Each output ∈ (0, 1): e^z is always positive, divided by a larger positive number
- Outputs sum to 1: Σₖ e^zₖ / Σⱼ e^zⱼ = 1 by definition
- Monotone in zₖ: higher logit → higher probability
- Differentiable everywhere

### Worked example

Three classes, logits z = [2.0, 1.0, 0.5]:

```
e^2.0 = 7.389
e^1.0 = 2.718
e^0.5 = 1.649

Sum = 11.756

softmax = [7.389/11.756, 2.718/11.756, 1.649/11.756]
        = [0.629,        0.231,        0.140]

Check: 0.629 + 0.231 + 0.140 = 1.0  ✓
```

The highest logit (class 0, z=2.0) dominates. Softmax amplifies differences between logits via the exponential.

### The "soft" in softmax

Argmax is hard — it outputs 1 for the max, 0 for everything else. Softmax is a smooth, differentiable approximation:

```
logits: [2.0, 1.0, 0.5]

argmax: [1,   0,   0  ]   ← not differentiable
softmax: [0.629, 0.231, 0.140]  ← differentiable, usable in backprop
```

As temperature → 0 (or as the max logit dominates), softmax → argmax.

### Numerical stability — the log-sum-exp trick

e^z overflows for large z (e^1000 = inf in float32). Fix: subtract the max logit first.

```
softmax(z)ₖ = e^(zₖ − max(z)) / Σⱼ e^(zⱼ − max(z))
```

Mathematically identical (the max cancels in numerator and denominator), but numerically stable.

```python
def softmax_stable(z):
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    return e / e.sum()
```

### The model — softmax regression (multinomial logistic regression)

For K classes, learn K weight vectors w₁, ..., wK:

```
zₖ = wₖᵀx + bₖ          # logit for class k
p̂ₖ = softmax(z)ₖ         # probability for class k
     = e^(wₖᵀx + bₖ) / Σⱼ e^(wⱼᵀx + bⱼ)
```

This is what sklearn does with `multi_class='multinomial'`.

### Loss — categorical cross-entropy

```
L = −Σᵢ Σₖ yᵢₖ · log(p̂ᵢₖ)

where yᵢₖ = 1 if sample i belongs to class k, else 0   (one-hot)
```

For each sample, only the true class term survives (yᵢₖ=0 for all other k):

```
L = −Σᵢ log(p̂ᵢ, yᵢ)    # just the log-prob of the correct class
```

### Gradient — clean form

```
∂L/∂wₖ = Σᵢ (p̂ᵢₖ − yᵢₖ) · xᵢ

In matrix form:
∇_wₖ L = Xᵀ(p̂ₖ − yₖ)
```

Same structure as binary logistic regression — residuals times features. Elegant.

### Softmax gradient derivation (the key step)

The tricky part: ∂softmax(z)ₖ/∂zⱼ — the output of class k depends on ALL logits via the denominator.

```
If j = k:
  ∂p̂ₖ/∂zₖ = p̂ₖ(1 − p̂ₖ)      ← same as sigmoid!

If j ≠ k:
  ∂p̂ₖ/∂zⱼ = −p̂ₖ · p̂ⱼ         ← cross-class terms

Compact form using Kronecker delta δₖⱼ:
  ∂p̂ₖ/∂zⱼ = p̂ₖ(δₖⱼ − p̂ⱼ)
```

When composed with categorical cross-entropy, these cross-terms cancel beautifully and you get the clean gradient above.

### Softmax reduces to sigmoid for K=2

```
p̂₁ = e^z₁ / (e^z₁ + e^z₂)
    = 1 / (1 + e^(z₂−z₁))
    = σ(z₁ − z₂)
```

Binary logistic regression is a special case of softmax regression. The two-class softmax is exactly sigmoid applied to the difference of logits.

### OvR vs Multinomial in sklearn

**One-vs-Rest (`ovr`):**
- Trains K independent binary classifiers
- Classifier k: "is this class k vs everything else?"
- Probabilities from K classifiers don't naturally sum to 1 (sklearn normalises post-hoc)
- Faster, works with any binary solver

**Multinomial (true softmax):**
- One model, K output logits, softmax over all simultaneously
- Probabilities sum to 1 by construction
- Classes compete against each other in the loss
- More principled, usually better calibrated

```python
# OvR
lr_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs')

# Multinomial (softmax)
lr_mn = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

**Use multinomial** unless you have a specific reason not to (e.g. using liblinear which only supports OvR).

### When to use which

| Situation | Prefer |
|---|---|
| Linear or near-linear boundary | Logistic Regression |
| Complex nonlinear boundary | Decision Tree (or ensemble) |
| Need odds ratios / statistical inference | Logistic Regression |
| Mixed feature types, missing data | Decision Tree |
| Small number of features, large n | Logistic Regression |
| Need rule-based explanations | Decision Tree |
| Outliers in features | Decision Tree (invariant to scale) |
| Probability calibration matters | Logistic Regression |

### The ensemble angle

In practice, bare decision trees are rarely the answer. The real comparison is LR vs **Random Forests** or **Gradient Boosted Trees** (XGBoost, LightGBM). Those combine many trees and almost always beat bare LR on tabular data — unless you need interpretability, fast training, or the relationship really is linear.

---

## Quick Reference — When to Use Logistic Regression

**Use LR when:**
- Relationship between features and log-odds is approximately linear
- You need interpretable coefficients (odds ratios)
- You need well-calibrated probabilities
- Dataset is large and features are engineered
- Baseline model / benchmark

**Don't use LR when:**
- Boundary is highly nonlinear and you can't engineer features
- Heavy feature interactions that are hard to specify manually
- Tree ensembles are available and interpretability isn't critical

**"Why does logistic regression always converge?"**  
The log-likelihood is concave in w (negative semi-definite Hessian), so there's a unique global maximum.


# GLMs, Link Functions & Neural Nets vs Logistic Regression

---

## 1. Generalised Linear Models — The Full Picture

### The problem GLMs solve

Ordinary linear regression assumes:
- The target is continuous and unbounded
- The noise is Gaussian
- The mean response is linear in features

Real data violates all three constantly:
- Binary outcomes (churn, fraud, death)
- Count outcomes (number of clicks, accidents)
- Positive-only outcomes (income, time-to-event)

GLMs extend linear regression to handle all of these by changing two things: the **distribution** of the target, and the **link function** that connects the linear predictor to the mean.

---

### The three components of every GLM

Every GLM has exactly three parts:

```
1. Random component:    distribution of y  (Gaussian, Bernoulli, Poisson, ...)
2. Systematic component: linear predictor  η = wᵀx + b
3. Link function:        g(μ) = η          where μ = E[y|x]
```

The link function g maps the mean μ (which lives in a constrained space) to the linear predictor η (which lives on the real line).

Equivalently, the inverse link g⁻¹ maps η back to μ:

```
μ = g⁻¹(η) = g⁻¹(wᵀx + b)
```

This inverse is called the **mean function** or **response function**.

---

### The exponential family — where GLMs come from

All GLM distributions belong to the **exponential family**. The general form:

```
P(y | θ, φ) = exp[ (y·θ − b(θ)) / φ  +  c(y, φ) ]

θ  = natural parameter (canonical parameter)
b(θ) = log-partition function (ensures normalisation)
φ  = dispersion parameter (controls spread)
```

Two key facts fall out of this form automatically:

```
E[y]   = b'(θ)      (mean = derivative of log-partition)
Var[y] = φ · b''(θ) (variance = φ × second derivative)
```

This is why the distribution's mean and variance are linked — they both come from b(θ).

---

### The canonical link — the natural choice

The **canonical link** is the link function where g(μ) = θ, the natural parameter.

In other words: set the linear predictor η equal to the natural parameter θ directly.

```
η = wᵀx = θ  (canonical link)
```

Why is this special? Because when you use the canonical link:
- The sufficient statistic is Xᵀy (clean, simple)
- The score equations (gradient = 0) have a particularly clean form
- The Fisher information simplifies
- In practice: gradient has the form Xᵀ(y − μ) for every GLM — same structure always

The canonical link isn't always the best link for a problem, but it's the principled default derived from the probability model itself.

---

## 2. Every Major GLM — Distribution, Canonical Link, Mean Function

---

### GLM 1 — Linear Regression

```
Distribution:     Gaussian  y ~ N(μ, σ²)
Support of y:     (−∞, ∞)
Natural param θ:  μ  (the mean itself)
b(θ):             θ²/2

Canonical link:   g(μ) = μ        (identity)
Mean function:    μ = η = wᵀx

Loss:             MSE  ←  NLL of Gaussian
```

The identity link means "do nothing" — linear regression is the GLM where the mean IS the linear predictor. It's the base case.

**Variance:** Var[y] = σ² (constant — homoscedastic assumption)

---

### GLM 2 — Logistic Regression

```
Distribution:     Bernoulli  y ~ Bernoulli(p)
Support of y:     {0, 1}
Natural param θ:  log(p / (1−p))   ← the logit
b(θ):             log(1 + e^θ)

Canonical link:   g(μ) = log(μ / (1−μ))   (logit)
Mean function:    μ = σ(η) = 1/(1+e^{−η})  (sigmoid)

Loss:             Binary cross-entropy  ←  NLL of Bernoulli
```

The canonical link IS the logit. This is where the logit comes from — it's not invented, it's the natural parameter of the Bernoulli distribution. Sigmoid (the inverse) emerges as the mean function.

**Variance:** Var[y] = p(1−p)  — variance depends on the mean (heteroscedastic by nature)

---

### GLM 3 — Poisson Regression

```
Distribution:     Poisson  y ~ Poisson(λ)
Support of y:     {0, 1, 2, 3, ...}  (non-negative integers)
Natural param θ:  log(λ)
b(θ):             e^θ

Canonical link:   g(μ) = log(μ)    (log link)
Mean function:    μ = e^η = e^{wᵀx}

Loss:             Poisson NLL  ←  −Σᵢ [yᵢ·(wᵀxᵢ) − e^{wᵀxᵢ}]
```

Used for count data: website visits, insurance claims, number of accidents.

The log link ensures μ = e^η > 0 always (counts can't be negative). The exponential mean function means coefficients are multiplicative: a 1-unit increase in x multiplies the expected count by e^w.

**Variance:** Var[y] = λ  — variance equals the mean (Poisson assumption, often violated → use Negative Binomial instead)

---

### GLM 4 — Gamma Regression

```
Distribution:     Gamma  y ~ Gamma(α, β)
Support of y:     (0, ∞)  (positive continuous)
Natural param θ:  −1/μ
b(θ):             −log(−θ)

Canonical link:   g(μ) = 1/μ       (inverse link)
Mean function:    μ = 1/η

Common in practice: log link g(μ) = log(μ) is used instead
(inverse link is less numerically stable)

Loss:             Gamma NLL
```

Used for: insurance claim amounts, income, time-to-event, anything positive and right-skewed.

The Gamma distribution allows variance to scale with the mean squared: Var[y] = μ²/α — more variance for larger values (realistic for income data).

---

### GLM 5 — Inverse Gaussian Regression

```
Distribution:     Inverse Gaussian
Support of y:     (0, ∞)
Canonical link:   g(μ) = 1/μ²      (inverse squared)
Mean function:    μ = 1/√η

Variance:         Var[y] = μ³/λ
```

Used for: highly right-skewed positive data where variance grows faster than Gamma. Less common.

---

### Summary Table

| GLM | Distribution | Support | Canonical Link g(μ) | Mean function g⁻¹(η) | Variance |
|---|---|---|---|---|---|
| Linear | Gaussian | (−∞,∞) | Identity: μ | η | σ² (constant) |
| Logistic | Bernoulli | {0,1} | Logit: log(μ/1−μ) | Sigmoid: 1/(1+e^{−η}) | μ(1−μ) |
| Poisson | Poisson | {0,1,2,...} | Log: log(μ) | Exp: e^η | μ |
| Gamma | Gamma | (0,∞) | Inverse: 1/μ | 1/η | μ²/α |
| Inv. Gaussian | Inv. Gaussian | (0,∞) | Inv. squared: 1/μ² | 1/√η | μ³/λ |

---

### Non-canonical links — when you deviate

You can use a non-canonical link. Common reasons:

```
Binomial with log link    → estimates log-risk ratios (epidemiology prefers this)
Poisson with identity link → additive rather than multiplicative effects
Gamma with log link       → more stable numerics than canonical inverse link
```

Non-canonical links lose the clean sufficient statistic structure but are perfectly valid. The canonical link is the principled default, not a requirement.

---

### The unified gradient — why all GLMs share the same form

For ANY GLM with its canonical link, the gradient of the log-likelihood w.r.t. w is:

```
∇_w ℓ = Xᵀ(y − μ)
```

Always. The residuals times the features. Linear regression, logistic regression, Poisson regression — same gradient structure. This is the deep reason GLMs are a unified family, not a collection of ad-hoc models.

The only difference is what μ is:
```
Linear:    μ = wᵀx
Logistic:  μ = σ(wᵀx)
Poisson:   μ = e^{wᵀx}
```

---

## 3. Neural Networks vs Logistic Regression

### Logistic regression is a neural network

Literally. A single-layer neural network with sigmoid activation and binary cross-entropy loss IS logistic regression. Same model, same loss, same gradient.

```
Logistic Regression:

x₁ ─┐
x₂ ─┤─ [w₁x₁ + w₂x₂ + b] ─ σ(·) ─ p̂
x₃ ─┘

One layer. No hidden units. Linear transformation → sigmoid.
```

A neural network adds hidden layers between input and output:

```
Neural Network:

x₁ ─┐           ┌─ h₁ ─┐
x₂ ─┤─ W₁x+b₁ ─┤─ h₂ ─┤─ W₂h+b₂ ─ σ(·) ─ p̂
x₃ ─┘           └─ h₃ ─┘

Two layers. Hidden units. Each hidden unit applies a nonlinearity.
```

---

## 4. Why Neural Networks Are Nonlinear and Logistic Regression Is Linear

### Logistic regression is a linear classifier

The decision boundary is wᵀx + b = 0 — a hyperplane. The sigmoid is applied after the linear combination, it doesn't add nonlinearity to the decision boundary.

Why? Because the decision boundary is defined by:

```
p̂ = 0.5
σ(wᵀx + b) = 0.5
wᵀx + b = 0          ← still linear in x
```

Sigmoid is a monotone function. It rescales the output but doesn't change which side of zero the linear combination is on. The boundary is always a hyperplane.

### What happens with two layers

```
Layer 1:  h = σ(W₁x + b₁)     each hⱼ = σ(w₁ⱼᵀx + b₁ⱼ)
Layer 2:  p̂ = σ(W₂h + b₂)     = σ(Σⱼ w₂ⱼ·σ(w₁ⱼᵀx + b₁ⱼ) + b₂)
```

The output is a sigmoid of a **weighted sum of sigmoids of linear functions of x**. This is no longer linear in x — the inner sigmoids create nonlinear transformations of the features that the outer layer combines.

### Why composition of linear functions stays linear

If there were no activation function between layers:

```
Layer 1: h = W₁x + b₁
Layer 2: p̂ = W₂h + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```

This collapses to a single linear transformation. A 100-layer linear network is mathematically identical to a one-layer linear network. Depth without nonlinearity is useless.

The activation function (sigmoid, ReLU, tanh) between layers is what makes depth meaningful. Each layer applies a nonlinear transformation, and compositions of nonlinear functions can represent arbitrarily complex functions.

### The universal approximation theorem

A neural network with one hidden layer and enough hidden units can approximate any continuous function on a compact domain to arbitrary precision.

Logistic regression cannot. It's limited to linear decision boundaries (or whatever boundaries you can construct with explicit feature engineering).

### The feature learning perspective

This is the deepest difference.

**Logistic regression:** you engineer features manually, then fit a linear model on them.

```
Raw features → [YOU DO THIS PART] → engineered features → linear model → output
```

**Neural network:** the hidden layers learn the feature transformation automatically from data.

```
Raw features → [LEARNED BY HIDDEN LAYERS] → learned features → linear model → output
```

The last layer of a neural network IS logistic regression — it's a linear classifier on top of whatever the hidden layers learned. The hidden layers' job is to transform the raw features into a space where linear classification works.

```
Final layer of neural net:   W_final · h_last + b  → sigmoid → p̂
h_last:                      the learned feature representation
```

This is why "representation learning" is the core idea of deep learning.

### XOR — the canonical example

XOR cannot be linearly separated:

```
(0,0) → 0
(0,1) → 1
(1,0) → 1
(1,1) → 0
```

No straight line separates class 0 from class 1.

A neural network with 2 hidden units solves this by learning to compute:

```
h₁ = σ(x₁ + x₂ − 0.5)   ≈ "is at least one input on?"
h₂ = σ(x₁ + x₂ − 1.5)   ≈ "are both inputs on?"

output = σ(h₁ − 2·h₂)    ≈ "at least one but not both" = XOR
```

The hidden layer transformed (x₁, x₂) into (h₁, h₂) — a space where XOR IS linearly separable. Logistic regression in (h₁, h₂) space solves it.

### Summary

| | Logistic Regression | Neural Network |
|---|---|---|
| Decision boundary | Always a hyperplane (linear) | Arbitrary nonlinear |
| Feature engineering | Manual | Learned from data |
| Parameters | d + 1 (one per feature) | Can be millions |
| Training | Convex loss, guaranteed convergence | Non-convex, local minima |
| Interpretability | High (coefficients = odds ratios) | Low (black box) |
| Data needed | Works with small data | Needs lots of data |
| When it wins | Linear relationships, interpretability required | Complex patterns, raw inputs (images, text) |

### The relationship in one line

Logistic regression = neural network with zero hidden layers.  
Neural network = logistic regression on top of learned nonlinear features.

# Logistic Regression — Hypothesis Testing, GoF & Calibration

---

## 1. Hypothesis Testing in Logistic Regression

### What we're testing

After fitting logistic regression, we get coefficient estimates ŵⱼ. The natural question: is wⱼ actually non-zero, or did we get a non-zero estimate just by chance?

```
H₀: wⱼ = 0   (feature j has no effect on log-odds)
H₁: wⱼ ≠ 0
```

Three frameworks to answer this: Wald test, Likelihood Ratio Test (LRT), Score test (Rao test). All are asymptotically equivalent but differ in what they compute and when they're reliable.

---

## 2. Wald Test

### The idea

You have an estimate ŵⱼ and its standard error SE(ŵⱼ). If H₀: wⱼ=0 is true, then ŵⱼ/SE(ŵⱼ) should be close to zero. How far from zero is "significant"?

### The statistic

```
z = ŵⱼ / SE(ŵⱼ)

Under H₀:  z ~ N(0, 1)   asymptotically
```

Or equivalently the Wald chi-squared:

```
W = ŵⱼ² / Var(ŵⱼ) = (ŵⱼ / SE(ŵⱼ))²

Under H₀:  W ~ χ²(1)
```

For testing multiple coefficients simultaneously (H₀: w₁=w₂=0):

```
W = ŵᵀ [Var(ŵ)]⁻¹ ŵ  ~  χ²(k)

where k = number of constraints being tested
```

### Where SE(ŵⱼ) comes from

The variance of ŵ is the inverse of the Fisher Information Matrix (FIM):

```
Var(ŵ) = I(ŵ)⁻¹ = (XᵀWX)⁻¹

where W = diag(p̂ᵢ(1−p̂ᵢ))   [the Hessian we derived before]
```

So SE(ŵⱼ) = √[(XᵀWX)⁻¹]ⱼⱼ — the square root of the j-th diagonal element of the inverse Hessian.

### Hand calculation

Tiny example: n=6, one feature x, fitted model gives ŵ = 1.2.

```
Fitted probabilities: p̂ = [0.3, 0.4, 0.6, 0.7, 0.8, 0.5]
Features: x = [1, 2, 3, 4, 5, 6]  (plus intercept column of 1s)

W diagonal: p̂ᵢ(1−p̂ᵢ):
  0.3×0.7=0.21, 0.4×0.6=0.24, 0.6×0.4=0.24,
  0.7×0.3=0.21, 0.8×0.2=0.16, 0.5×0.5=0.25

XᵀWX for slope coefficient (simplified):
  Σᵢ wᵢ xᵢ² = 0.21×1 + 0.24×4 + 0.24×9 + 0.21×16 + 0.16×25 + 0.25×36
             = 0.21 + 0.96 + 2.16 + 3.36 + 4.00 + 9.00 = 19.69

Var(ŵ₁) ≈ 1 / 19.69 = 0.0508
SE(ŵ₁) = √0.0508 = 0.225

z = 1.2 / 0.225 = 5.33    (p-value ≈ 0.0001 → reject H₀)
W = 5.33² = 28.4  ~ χ²(1)
```

### Weakness of Wald test

Wald performs poorly when ŵⱼ is large. Why? SE(ŵⱼ) is estimated at ŵⱼ, not at 0. When there's perfect or near-perfect separation, ŵⱼ → ∞ and SE(ŵⱼ) → ∞ even faster, so the Wald statistic → 0. This is Hauck-Donner phenomenon: the Wald test says "not significant" precisely when the coefficient is most extreme.

```
Hauck-Donner:  ŵⱼ very large  →  W = ŵⱼ²/Var(ŵⱼ) → 0   (wrong answer)
```

**When Wald fails, use LRT.**

---

## 3. Likelihood Ratio Test (LRT)

### The idea

Compare the log-likelihood of two models:
- **Full model:** includes the feature(s) in question
- **Reduced model:** those features removed (set to 0)

If the feature matters, the full model should fit much better (higher log-likelihood).

### The statistic

```
G² = −2 · [ℓ(reduced) − ℓ(full)]
   =  2 · [ℓ(full) − ℓ(reduced)]

Under H₀:  G² ~ χ²(k)

where k = number of parameters removed (df of the test)
```

G² is also called the **deviance difference** or **likelihood ratio statistic**.

### Hand calculation

Two models on same data:

```
Full model (intercept + x):     ℓ_full    = −18.42
Reduced model (intercept only): ℓ_reduced = −24.31

G² = 2 × (−18.42 − (−24.31))
   = 2 × 5.89
   = 11.78

df = 1  (one parameter removed)

χ²(1) critical value at α=0.05: 3.84
11.78 > 3.84  →  reject H₀  →  x is a significant predictor
p-value = P(χ²(1) > 11.78) ≈ 0.0006
```

### Computing ℓ by hand

```
ℓ = Σᵢ [yᵢ log(p̂ᵢ) + (1−yᵢ) log(1−p̂ᵢ)]

For a single observation: y=1, p̂=0.7:
  contribution = 1×log(0.7) + 0×log(0.3) = log(0.7) = −0.357

For y=0, p̂=0.7:
  contribution = 0×log(0.7) + 1×log(0.3) = log(0.3) = −1.204

Null model (intercept only):
  p̂ = ȳ = proportion of 1s in data
  ℓ_null = n₁·log(ȳ) + n₀·log(1−ȳ)
```

### LRT for nested models — the general recipe

```
Step 1: fit full model,    get ℓ_full,    df_full
Step 2: fit reduced model, get ℓ_reduced, df_reduced

G² = 2(ℓ_full − ℓ_reduced)
df  = df_full − df_reduced   (number of extra params in full)

Compare G² to χ²(df) distribution
```

Models must be nested: reduced model is a special case of full model (some coefficients set to 0).

### LRT vs Wald — key differences

| | Wald | LRT |
|---|---|---|
| What it computes | Ratio of estimate to SE | Ratio of likelihoods |
| Models fit | One (full) | Two (full + reduced) |
| Computational cost | Cheaper | 2× the fitting |
| Accuracy in small samples | Worse | Better |
| Fails with separation | Yes (Hauck-Donner) | No |
| For large n | Equivalent to LRT | Equivalent to Wald |
| Preferred for | Quick inference, large n | Small n, large coefficients, separation |

**Rule of thumb:** use LRT when in doubt. Wald is an approximation to LRT that's convenient but less reliable.

---

## 4. Score Test (Rao Test)

### The idea

Only fit the reduced model. Check whether the gradient of the log-likelihood at the restricted estimate is "close enough to zero." If H₀ is true, the gradient at the restricted MLE should be near zero.

```
S = [∂ℓ/∂w]ₕ₀ᵀ · I(ŵ_H₀)⁻¹ · [∂ℓ/∂w]ₕ₀

Under H₀:  S ~ χ²(k)
```

The gradient ∂ℓ/∂w evaluated at the restricted (reduced) estimates.

### When to use score test

Only fit one model (the null). Useful when:
- The full model is very expensive to fit
- You want to screen many features quickly (test each without fitting full model)
- Lagrange multiplier test in econometrics

In practice, score tests are less common than Wald/LRT for logistic regression.

### The trinity — asymptotic equivalence

Under H₀ and as n → ∞:

```
Wald = LRT = Score   (asymptotically)
```

In finite samples they differ. LRT is generally most accurate. Wald is most convenient. Score is cheapest to compute.

---

## 5. All Logistic Regression Tests — Deep Dive

### 5.1 Testing the overall model — Omnibus test

Tests H₀: all coefficients (except intercept) = 0.

```
Full model:    intercept + all features,  ℓ_full
Null model:    intercept only,            ℓ_null = n₁·log(ȳ) + n₀·log(1−ȳ)

G²_model = 2(ℓ_full − ℓ_null)  ~  χ²(p)

where p = number of predictors (not counting intercept)
```

This is the logistic regression equivalent of the F-test in linear regression. If this is non-significant, no individual predictor is worth examining.

### 5.2 Testing individual coefficients

Use Wald or LRT for each wⱼ as above. In sklearn/statsmodels, these come from the summary table.

```python
import statsmodels.api as sm

model = sm.Logit(y, X).fit()
print(model.summary())

# Output includes:
# coef, std err, z (Wald), P>|z|, 95% CI
# Use .llr_pvalue for omnibus LRT p-value
```

### 5.3 Testing nested models (model comparison)

Did adding variables x₃, x₄ improve the model over x₁, x₂ alone?

```
Model A (reduced): w₀ + w₁x₁ + w₂x₂
Model B (full):    w₀ + w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄

G² = 2(ℓ_B − ℓ_A)  ~  χ²(2)   [2 extra params]
```

### 5.4 AIC and BIC — penalised likelihood for model selection

Not hypothesis tests, but used to compare non-nested models:

```
AIC = −2ℓ + 2k          (penalises complexity lightly)
BIC = −2ℓ + k·log(n)    (penalises complexity more heavily for large n)

k = number of parameters
n = sample size
```

Lower is better. BIC favours simpler models more strongly than AIC.

```
ℓ = −18.42,  k = 3,  n = 200

AIC = −2(−18.42) + 2(3) = 36.84 + 6  = 42.84
BIC = −2(−18.42) + 3×log(200) = 36.84 + 15.9 = 52.74
```

### 5.5 Confidence intervals for coefficients

Wald CI:

```
ŵⱼ ± z_{α/2} · SE(ŵⱼ)

95% CI:  ŵⱼ ± 1.96 · SE(ŵⱼ)
```

Profile likelihood CI (more accurate, based on LRT):

```
{w : 2(ℓ_full − ℓ(w)) ≤ χ²_{α}(1)}
```

Profile CIs are asymmetric and more reliable for small samples. Statsmodels computes them with `.conf_int(method='profile-likelihood')`.

For odds ratios, exponentiate the CI:

```
OR = e^ŵⱼ
95% CI for OR: [e^(ŵⱼ − 1.96·SE), e^(ŵⱼ + 1.96·SE)]
```

---

## 6. Goodness of Fit

### 6.1 Deviance

The deviance is the fundamental measure of fit for GLMs:

```
D = −2 · ℓ(fitted model)

Null deviance:     D_null = −2 · ℓ(null model)   [intercept only]
Residual deviance: D_res  = −2 · ℓ(fitted model)
```

The deviance difference = G² from the LRT:

```
G² = D_null − D_res  ~  χ²(p)
```

**Saturated model:** a model with one parameter per observation — fits perfectly. Deviance relative to saturated model measures how much information the fitted model loses.

### 6.2 Hosmer-Lemeshow Test

The most common GoF test for logistic regression.

**Procedure:**
1. Sort all observations by fitted probability p̂ᵢ
2. Divide into g groups (usually g=10 deciles)
3. In each group, compare observed events vs expected events

```
Expected events in group k:  Eₖ = Σᵢ∈k p̂ᵢ
Observed events in group k:  Oₖ = Σᵢ∈k yᵢ

HL statistic:
χ²_HL = Σₖ [ (Oₖ − Eₖ)² / (nₖ · p̄ₖ · (1−p̄ₖ)) ]  ~  χ²(g−2)

where p̄ₖ = mean predicted probability in group k
      nₖ  = number of observations in group k
      df  = g − 2  (subtract 2: one for each tail fixed)
```

**H₀:** model is well-calibrated (observed ≈ expected in each group)  
**Reject H₀:** model is poorly calibrated

### Hand calculation of Hosmer-Lemeshow

10 observations sorted by p̂, split into 2 groups of 5 (simplified, g=2 → df=0, so use g=4):

```
All 10 obs sorted by p̂:

Obs:  p̂:    y:
1     0.05   0
2     0.10   0
3     0.15   0
4     0.20   1
5     0.25   0
6     0.55   1
7     0.65   0
8     0.70   1
9     0.80   1
10    0.90   1

Group 1 (lowest 5, p̂ ≈ 0.05-0.25):
  n₁ = 5
  E₁ = 0.05+0.10+0.15+0.20+0.25 = 0.75
  O₁ = 0+0+0+1+0 = 1
  p̄₁ = 0.75/5 = 0.15

Group 2 (highest 5, p̂ ≈ 0.55-0.90):
  n₂ = 5
  E₂ = 0.55+0.65+0.70+0.80+0.90 = 3.60
  O₂ = 1+0+1+1+1 = 4
  p̄₂ = 3.60/5 = 0.72

χ²_HL = (1−0.75)² / (5×0.15×0.85)  +  (4−3.60)² / (5×0.72×0.28)
      = (0.0625) / (0.6375)          +  (0.16) / (1.008)
      = 0.098  +  0.159
      = 0.257

df = g−2 = 0  (degenerate for g=2; use at least g=5)
```

With g=10 (standard), df=8. If χ²_HL < χ²_{0.05}(8) = 15.51, fail to reject → good fit.

### 6.3 Pearson Chi-squared GoF

```
χ²_P = Σᵢ (yᵢ − p̂ᵢ)² / (p̂ᵢ(1−p̂ᵢ))   ~  χ²(n−p−1)
```

Works well for grouped data (multiple obs per covariate pattern). For individual binary outcomes with continuous predictors, this test has poor power — use HL instead.

### 6.4 Pseudo-R² measures

No single R² for logistic regression. Several approximations exist:

**McFadden's R²:**
```
R²_McF = 1 − ℓ_full / ℓ_null

Range: [0, 1). Values of 0.2−0.4 indicate excellent fit (much lower than linear R²).
```

**Cox-Snell R²:**
```
R²_CS = 1 − exp(2(ℓ_null − ℓ_full) / n)

Max value < 1 (annoying)
```

**Nagelkerke R²:**
```
R²_N = R²_CS / R²_CS_max

where R²_CS_max = 1 − exp(2ℓ_null/n)

Rescales Cox-Snell to have max = 1.
```

**Hand calculation:**

```
ℓ_null = −45.2,  ℓ_full = −28.7,  n = 100

McFadden: 1 − (−28.7)/(−45.2) = 1 − 0.635 = 0.365

Cox-Snell: 1 − exp(2(−45.2 − (−28.7))/100)
         = 1 − exp(2×(−16.5)/100)
         = 1 − exp(−0.33)
         = 1 − 0.719 = 0.281

R²_CS_max = 1 − exp(2×(−45.2)/100)
           = 1 − exp(−0.904) = 1 − 0.405 = 0.595

Nagelkerke: 0.281 / 0.595 = 0.472
```

---

## 7. Calibration — Full Treatment

### What calibration means

A model is **calibrated** if predicted probabilities match observed frequencies.

```
Among all observations where model predicts p̂ = 0.7,
approximately 70% should actually be positive.
```

A model can have high AUC (good discrimination) but be poorly calibrated — it ranks observations correctly but the actual probability values are wrong. For decision-making (setting thresholds, expected value calculations), calibration matters as much as discrimination.

### How to identify poor calibration

**Method 1 — Calibration curve (reliability diagram)**

```
1. Sort predictions by p̂
2. Bin into B buckets (e.g. 10 equal-width or equal-frequency bins)
3. In each bin: compute mean(p̂) and mean(y) = fraction of positives
4. Plot mean(p̂) on x-axis vs mean(y) on y-axis
5. Perfect calibration = diagonal line y = x
```

Patterns to look for:

```
S-shaped curve:    overconfident model (predicts too extreme, probabilities too close to 0/1)
Inverse S-shape:   underconfident (probabilities too close to 0.5)
Curve above diagonal: underpredicting (predicted p̂ too low vs actual rate)
Curve below diagonal: overpredicting (predicted p̂ too high vs actual rate)
```

**Method 2 — Hosmer-Lemeshow test** (section 6.2 above — it IS a calibration test)

**Method 3 — Expected Calibration Error (ECE)**

```
ECE = Σₖ (nₖ/n) · |mean(p̂)ₖ − mean(y)ₖ|

Weighted average of |predicted − actual| across bins.
Range: [0, 1]. Closer to 0 = better calibrated.
```

### Hand calibration calculation

8 observations:

```
p̂:   0.1  0.2  0.3  0.4  0.6  0.7  0.8  0.9
y:    0    0    1    0    1    1    0    1

Bin 1 (p̂ < 0.5): p̂=[0.1,0.2,0.3,0.4], y=[0,0,1,0]
  mean(p̂) = 0.25,  mean(y) = 0.25,  |diff| = 0.00  ← perfect

Bin 2 (p̂ ≥ 0.5): p̂=[0.6,0.7,0.8,0.9], y=[1,1,0,1]
  mean(p̂) = 0.75,  mean(y) = 0.75,  |diff| = 0.00  ← perfect

ECE = (4/8)×0.00 + (4/8)×0.00 = 0.00   (well calibrated)
```

Now a miscalibrated model on the same y:

```
p̂:   0.6  0.7  0.8  0.9  0.9  0.9  0.9  0.9   (overconfident)
y:    0    0    1    0    1    1    0    1

Bin 1 (p̂ < 0.85): p̂=[0.6,0.7,0.8], y=[0,0,1]
  mean(p̂) = 0.70,  mean(y) = 0.33,  |diff| = 0.37

Bin 2 (p̂ ≥ 0.85): p̂=[0.9,0.9,0.9,0.9,0.9], y=[0,1,1,0,1]
  mean(p̂) = 0.90,  mean(y) = 0.60,  |diff| = 0.30

ECE = (3/8)×0.37 + (5/8)×0.30 = 0.139 + 0.188 = 0.327  (poor calibration)
```

The overconfident model always predicts high but only 62.5% are actually positive.

### How to fix calibration

**Method 1 — Platt Scaling**

Fit a logistic regression on top of the model's raw scores:

```
Step 1: Train original model, get scores sᵢ (logits or probabilities)
Step 2: On held-out validation set, fit:
         p_calibrated = σ(a·sᵢ + b)
         where a, b are learned by MLE on (sᵢ, yᵢ) pairs

Platt scaling is logistic regression used as a calibration layer.
```

If a=1, b=0 → no correction needed (already calibrated).  
If a < 1 → the model was overconfident (Platt compresses scores toward 0.5).  
If b ≠ 0 → the model had a systematic bias.

**Hand calculation of Platt scaling:**

```
Raw scores (logits):  s = [−2, −1, 0, 1, 2]
True labels:          y = [0,   0,  1, 1, 1]

Fit σ(a·s + b) by MLE.
Initial: a=1, b=0  →  p̂ = σ(s) = [0.12, 0.27, 0.50, 0.73, 0.88]

Observed rates:
  s<0: 2 obs, 0 positives  → observed = 0.0, predicted = ~0.2  (too high)
  s≥0: 3 obs, 3 positives  → observed = 1.0, predicted = ~0.7  (too low)

After fitting a=0.8, b=−0.3:
  p̂ = σ(0.8s − 0.3) → better aligned with observations
```

**Method 2 — Isotonic Regression**

Non-parametric. Fits a monotone step function mapping raw scores to calibrated probabilities. More flexible than Platt but needs more data.

```
sklearn:
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling
cal_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)

# Isotonic regression  
cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)

cal_model.fit(X_train, y_train)
p_calibrated = cal_model.predict_proba(X_test)[:, 1]
```

**Method 3 — Temperature scaling** (neural nets / deep learning)

Scale all logits by a single temperature parameter T:

```
p_calibrated = softmax(z / T)

T > 1:  softens probabilities (reduces overconfidence)
T < 1:  sharpens probabilities (increases confidence)
T = 1:  no change
```

Find T by minimising NLL on a held-out validation set.

### Calibration vs discrimination

| | Calibration | Discrimination (AUC) |
|---|---|---|
| What it measures | Are probability values accurate? | Are observations correctly ranked? |
| Perfect score | ECE = 0, HL non-significant | AUC = 1.0 |
| Can you have one without the other? | Yes — high AUC, poor calibration (common) | Yes — perfect calibration, AUC=0.5 |
| Fixing one fixes the other? | No | No |
| When it matters most | Probability used directly (pricing, decisions) | Binary classification, ranking |

A model can rank patients perfectly by risk (AUC=1) but assign probabilities of 0.9 to everyone — not calibrated. A model can be perfectly calibrated (every 0.7 has 70% outcome rate) but fail to discriminate (AUC = 0.5). They measure orthogonal properties.

---

## Quick Reference

```
Test                Formula                       Dist under H₀    Use when
─────────────────────────────────────────────────────────────────────────────
Wald               W = (ŵ/SE)²                   χ²(1)            Quick, large n
LRT                G² = 2(ℓ_full − ℓ_red)        χ²(df)           Small n, separation
Score              S = gradient at H₀             χ²(k)            Only fit null model
Omnibus            G² = 2(ℓ_full − ℓ_null)       χ²(p)            Test whole model
Hosmer-Lemeshow    Σ(O−E)²/(n·p̄·(1−p̄))          χ²(g−2)          Calibration/GoF

Pseudo-R²          McFadden = 1 − ℓ_full/ℓ_null
                   Nagelkerke = rescaled Cox-Snell to [0,1]

Calibration        ECE = Σ(nₖ/n)|p̄ₖ − ȳₖ|       → 0 is perfect
Fix calibration    Platt scaling, isotonic regression, temperature scaling
```
# Logistic Regression — Hypothesis Testing, GoF & Calibration

---

## 1. Hypothesis Testing in Logistic Regression

### What we're testing

After fitting logistic regression, we get coefficient estimates ŵⱼ. The natural question: is wⱼ actually non-zero, or did we get a non-zero estimate just by chance?

```
H₀: wⱼ = 0   (feature j has no effect on log-odds)
H₁: wⱼ ≠ 0
```

Three frameworks to answer this: Wald test, Likelihood Ratio Test (LRT), Score test (Rao test). All are asymptotically equivalent but differ in what they compute and when they're reliable.

---

## 2. Wald Test

### The idea

You have an estimate ŵⱼ and its standard error SE(ŵⱼ). If H₀: wⱼ=0 is true, then ŵⱼ/SE(ŵⱼ) should be close to zero. How far from zero is "significant"?

### The statistic

```
z = ŵⱼ / SE(ŵⱼ)

Under H₀:  z ~ N(0, 1)   asymptotically
```

Or equivalently the Wald chi-squared:

```
W = ŵⱼ² / Var(ŵⱼ) = (ŵⱼ / SE(ŵⱼ))²

Under H₀:  W ~ χ²(1)
```

For testing multiple coefficients simultaneously (H₀: w₁=w₂=0):

```
W = ŵᵀ [Var(ŵ)]⁻¹ ŵ  ~  χ²(k)

where k = number of constraints being tested
```

### Where SE(ŵⱼ) comes from

The variance of ŵ is the inverse of the Fisher Information Matrix (FIM):

```
Var(ŵ) = I(ŵ)⁻¹ = (XᵀWX)⁻¹

where W = diag(p̂ᵢ(1−p̂ᵢ))   [the Hessian we derived before]
```

So SE(ŵⱼ) = √[(XᵀWX)⁻¹]ⱼⱼ — the square root of the j-th diagonal element of the inverse Hessian.

### Hand calculation

Tiny example: n=6, one feature x, fitted model gives ŵ = 1.2.

```
Fitted probabilities: p̂ = [0.3, 0.4, 0.6, 0.7, 0.8, 0.5]
Features: x = [1, 2, 3, 4, 5, 6]  (plus intercept column of 1s)

W diagonal: p̂ᵢ(1−p̂ᵢ):
  0.3×0.7=0.21, 0.4×0.6=0.24, 0.6×0.4=0.24,
  0.7×0.3=0.21, 0.8×0.2=0.16, 0.5×0.5=0.25

XᵀWX for slope coefficient (simplified):
  Σᵢ wᵢ xᵢ² = 0.21×1 + 0.24×4 + 0.24×9 + 0.21×16 + 0.16×25 + 0.25×36
             = 0.21 + 0.96 + 2.16 + 3.36 + 4.00 + 9.00 = 19.69

Var(ŵ₁) ≈ 1 / 19.69 = 0.0508
SE(ŵ₁) = √0.0508 = 0.225

z = 1.2 / 0.225 = 5.33    (p-value ≈ 0.0001 → reject H₀)
W = 5.33² = 28.4  ~ χ²(1)
```

### Weakness of Wald test

Wald performs poorly when ŵⱼ is large. Why? SE(ŵⱼ) is estimated at ŵⱼ, not at 0. When there's perfect or near-perfect separation, ŵⱼ → ∞ and SE(ŵⱼ) → ∞ even faster, so the Wald statistic → 0. This is Hauck-Donner phenomenon: the Wald test says "not significant" precisely when the coefficient is most extreme.

```
Hauck-Donner:  ŵⱼ very large  →  W = ŵⱼ²/Var(ŵⱼ) → 0   (wrong answer)
```

**When Wald fails, use LRT.**

---

## 3. Likelihood Ratio Test (LRT)

### The idea

Compare the log-likelihood of two models:
- **Full model:** includes the feature(s) in question
- **Reduced model:** those features removed (set to 0)

If the feature matters, the full model should fit much better (higher log-likelihood).

### The statistic

```
G² = −2 · [ℓ(reduced) − ℓ(full)]
   =  2 · [ℓ(full) − ℓ(reduced)]

Under H₀:  G² ~ χ²(k)

where k = number of parameters removed (df of the test)
```

G² is also called the **deviance difference** or **likelihood ratio statistic**.

### Hand calculation

Two models on same data:

```
Full model (intercept + x):     ℓ_full    = −18.42
Reduced model (intercept only): ℓ_reduced = −24.31

G² = 2 × (−18.42 − (−24.31))
   = 2 × 5.89
   = 11.78

df = 1  (one parameter removed)

χ²(1) critical value at α=0.05: 3.84
11.78 > 3.84  →  reject H₀  →  x is a significant predictor
p-value = P(χ²(1) > 11.78) ≈ 0.0006
```

### Computing ℓ by hand

```
ℓ = Σᵢ [yᵢ log(p̂ᵢ) + (1−yᵢ) log(1−p̂ᵢ)]

For a single observation: y=1, p̂=0.7:
  contribution = 1×log(0.7) + 0×log(0.3) = log(0.7) = −0.357

For y=0, p̂=0.7:
  contribution = 0×log(0.7) + 1×log(0.3) = log(0.3) = −1.204

Null model (intercept only):
  p̂ = ȳ = proportion of 1s in data
  ℓ_null = n₁·log(ȳ) + n₀·log(1−ȳ)
```

### LRT for nested models — the general recipe

```
Step 1: fit full model,    get ℓ_full,    df_full
Step 2: fit reduced model, get ℓ_reduced, df_reduced

G² = 2(ℓ_full − ℓ_reduced)
df  = df_full − df_reduced   (number of extra params in full)

Compare G² to χ²(df) distribution
```

Models must be nested: reduced model is a special case of full model (some coefficients set to 0).

### LRT vs Wald — key differences

| | Wald | LRT |
|---|---|---|
| What it computes | Ratio of estimate to SE | Ratio of likelihoods |
| Models fit | One (full) | Two (full + reduced) |
| Computational cost | Cheaper | 2× the fitting |
| Accuracy in small samples | Worse | Better |
| Fails with separation | Yes (Hauck-Donner) | No |
| For large n | Equivalent to LRT | Equivalent to Wald |
| Preferred for | Quick inference, large n | Small n, large coefficients, separation |

**Rule of thumb:** use LRT when in doubt. Wald is an approximation to LRT that's convenient but less reliable.

---

## 4. Score Test (Rao Test)

### The idea

Only fit the reduced model. Check whether the gradient of the log-likelihood at the restricted estimate is "close enough to zero." If H₀ is true, the gradient at the restricted MLE should be near zero.

```
S = [∂ℓ/∂w]ₕ₀ᵀ · I(ŵ_H₀)⁻¹ · [∂ℓ/∂w]ₕ₀

Under H₀:  S ~ χ²(k)
```

The gradient ∂ℓ/∂w evaluated at the restricted (reduced) estimates.

### When to use score test

Only fit one model (the null). Useful when:
- The full model is very expensive to fit
- You want to screen many features quickly (test each without fitting full model)
- Lagrange multiplier test in econometrics

In practice, score tests are less common than Wald/LRT for logistic regression.

### The trinity — asymptotic equivalence

Under H₀ and as n → ∞:

```
Wald = LRT = Score   (asymptotically)
```

In finite samples they differ. LRT is generally most accurate. Wald is most convenient. Score is cheapest to compute.

---

## 5. All Logistic Regression Tests — Deep Dive

### 5.1 Testing the overall model — Omnibus test

Tests H₀: all coefficients (except intercept) = 0.

```
Full model:    intercept + all features,  ℓ_full
Null model:    intercept only,            ℓ_null = n₁·log(ȳ) + n₀·log(1−ȳ)

G²_model = 2(ℓ_full − ℓ_null)  ~  χ²(p)

where p = number of predictors (not counting intercept)
```

This is the logistic regression equivalent of the F-test in linear regression. If this is non-significant, no individual predictor is worth examining.

### 5.2 Testing individual coefficients

Use Wald or LRT for each wⱼ as above. In sklearn/statsmodels, these come from the summary table.

```python
import statsmodels.api as sm

model = sm.Logit(y, X).fit()
print(model.summary())

# Output includes:
# coef, std err, z (Wald), P>|z|, 95% CI
# Use .llr_pvalue for omnibus LRT p-value
```

### 5.3 Testing nested models (model comparison)

Did adding variables x₃, x₄ improve the model over x₁, x₂ alone?

```
Model A (reduced): w₀ + w₁x₁ + w₂x₂
Model B (full):    w₀ + w₁x₁ + w₂x₂ + w₃x₃ + w₄x₄

G² = 2(ℓ_B − ℓ_A)  ~  χ²(2)   [2 extra params]
```

### 5.4 AIC and BIC — penalised likelihood for model selection

Not hypothesis tests, but used to compare non-nested models:

```
AIC = −2ℓ + 2k          (penalises complexity lightly)
BIC = −2ℓ + k·log(n)    (penalises complexity more heavily for large n)

k = number of parameters
n = sample size
```

Lower is better. BIC favours simpler models more strongly than AIC.

```
ℓ = −18.42,  k = 3,  n = 200

AIC = −2(−18.42) + 2(3) = 36.84 + 6  = 42.84
BIC = −2(−18.42) + 3×log(200) = 36.84 + 15.9 = 52.74
```

### 5.5 Confidence intervals for coefficients

Wald CI:

```
ŵⱼ ± z_{α/2} · SE(ŵⱼ)

95% CI:  ŵⱼ ± 1.96 · SE(ŵⱼ)
```

Profile likelihood CI (more accurate, based on LRT):

```
{w : 2(ℓ_full − ℓ(w)) ≤ χ²_{α}(1)}
```

Profile CIs are asymmetric and more reliable for small samples. Statsmodels computes them with `.conf_int(method='profile-likelihood')`.

For odds ratios, exponentiate the CI:

```
OR = e^ŵⱼ
95% CI for OR: [e^(ŵⱼ − 1.96·SE), e^(ŵⱼ + 1.96·SE)]
```

---

## 6. Goodness of Fit

### 6.1 Deviance

The deviance is the fundamental measure of fit for GLMs:

```
D = −2 · ℓ(fitted model)

Null deviance:     D_null = −2 · ℓ(null model)   [intercept only]
Residual deviance: D_res  = −2 · ℓ(fitted model)
```

The deviance difference = G² from the LRT:

```
G² = D_null − D_res  ~  χ²(p)
```

**Saturated model:** a model with one parameter per observation — fits perfectly. Deviance relative to saturated model measures how much information the fitted model loses.

### 6.2 Hosmer-Lemeshow Test

The most common GoF test for logistic regression.

**Procedure:**
1. Sort all observations by fitted probability p̂ᵢ
2. Divide into g groups (usually g=10 deciles)
3. In each group, compare observed events vs expected events

```
Expected events in group k:  Eₖ = Σᵢ∈k p̂ᵢ
Observed events in group k:  Oₖ = Σᵢ∈k yᵢ

HL statistic:
χ²_HL = Σₖ [ (Oₖ − Eₖ)² / (nₖ · p̄ₖ · (1−p̄ₖ)) ]  ~  χ²(g−2)

where p̄ₖ = mean predicted probability in group k
      nₖ  = number of observations in group k
      df  = g − 2  (subtract 2: one for each tail fixed)
```

**H₀:** model is well-calibrated (observed ≈ expected in each group)  
**Reject H₀:** model is poorly calibrated

### Hand calculation of Hosmer-Lemeshow

10 observations sorted by p̂, split into 2 groups of 5 (simplified, g=2 → df=0, so use g=4):

```
All 10 obs sorted by p̂:

Obs:  p̂:    y:
1     0.05   0
2     0.10   0
3     0.15   0
4     0.20   1
5     0.25   0
6     0.55   1
7     0.65   0
8     0.70   1
9     0.80   1
10    0.90   1

Group 1 (lowest 5, p̂ ≈ 0.05-0.25):
  n₁ = 5
  E₁ = 0.05+0.10+0.15+0.20+0.25 = 0.75
  O₁ = 0+0+0+1+0 = 1
  p̄₁ = 0.75/5 = 0.15

Group 2 (highest 5, p̂ ≈ 0.55-0.90):
  n₂ = 5
  E₂ = 0.55+0.65+0.70+0.80+0.90 = 3.60
  O₂ = 1+0+1+1+1 = 4
  p̄₂ = 3.60/5 = 0.72

χ²_HL = (1−0.75)² / (5×0.15×0.85)  +  (4−3.60)² / (5×0.72×0.28)
      = (0.0625) / (0.6375)          +  (0.16) / (1.008)
      = 0.098  +  0.159
      = 0.257

df = g−2 = 0  (degenerate for g=2; use at least g=5)
```

With g=10 (standard), df=8. If χ²_HL < χ²_{0.05}(8) = 15.51, fail to reject → good fit.

### 6.3 Pearson Chi-squared GoF

```
χ²_P = Σᵢ (yᵢ − p̂ᵢ)² / (p̂ᵢ(1−p̂ᵢ))   ~  χ²(n−p−1)
```

Works well for grouped data (multiple obs per covariate pattern). For individual binary outcomes with continuous predictors, this test has poor power — use HL instead.

### 6.4 Pseudo-R² measures

No single R² for logistic regression. Several approximations exist:

**McFadden's R²:**
```
R²_McF = 1 − ℓ_full / ℓ_null

Range: [0, 1). Values of 0.2−0.4 indicate excellent fit (much lower than linear R²).
```

**Cox-Snell R²:**
```
R²_CS = 1 − exp(2(ℓ_null − ℓ_full) / n)

Max value < 1 (annoying)
```

**Nagelkerke R²:**
```
R²_N = R²_CS / R²_CS_max

where R²_CS_max = 1 − exp(2ℓ_null/n)

Rescales Cox-Snell to have max = 1.
```

**Hand calculation:**

```
ℓ_null = −45.2,  ℓ_full = −28.7,  n = 100

McFadden: 1 − (−28.7)/(−45.2) = 1 − 0.635 = 0.365

Cox-Snell: 1 − exp(2(−45.2 − (−28.7))/100)
         = 1 − exp(2×(−16.5)/100)
         = 1 − exp(−0.33)
         = 1 − 0.719 = 0.281

R²_CS_max = 1 − exp(2×(−45.2)/100)
           = 1 − exp(−0.904) = 1 − 0.405 = 0.595

Nagelkerke: 0.281 / 0.595 = 0.472
```

---

## 7. Calibration — Full Treatment

### What calibration means

A model is **calibrated** if predicted probabilities match observed frequencies.

```
Among all observations where model predicts p̂ = 0.7,
approximately 70% should actually be positive.
```

A model can have high AUC (good discrimination) but be poorly calibrated — it ranks observations correctly but the actual probability values are wrong. For decision-making (setting thresholds, expected value calculations), calibration matters as much as discrimination.

### How to identify poor calibration

**Method 1 — Calibration curve (reliability diagram)**

```
1. Sort predictions by p̂
2. Bin into B buckets (e.g. 10 equal-width or equal-frequency bins)
3. In each bin: compute mean(p̂) and mean(y) = fraction of positives
4. Plot mean(p̂) on x-axis vs mean(y) on y-axis
5. Perfect calibration = diagonal line y = x
```

Patterns to look for:

```
S-shaped curve:    overconfident model (predicts too extreme, probabilities too close to 0/1)
Inverse S-shape:   underconfident (probabilities too close to 0.5)
Curve above diagonal: underpredicting (predicted p̂ too low vs actual rate)
Curve below diagonal: overpredicting (predicted p̂ too high vs actual rate)
```

**Method 2 — Hosmer-Lemeshow test** (section 6.2 above — it IS a calibration test)

**Method 3 — Expected Calibration Error (ECE)**

```
ECE = Σₖ (nₖ/n) · |mean(p̂)ₖ − mean(y)ₖ|

Weighted average of |predicted − actual| across bins.
Range: [0, 1]. Closer to 0 = better calibrated.
```

### Hand calibration calculation

8 observations:

```
p̂:   0.1  0.2  0.3  0.4  0.6  0.7  0.8  0.9
y:    0    0    1    0    1    1    0    1

Bin 1 (p̂ < 0.5): p̂=[0.1,0.2,0.3,0.4], y=[0,0,1,0]
  mean(p̂) = 0.25,  mean(y) = 0.25,  |diff| = 0.00  ← perfect

Bin 2 (p̂ ≥ 0.5): p̂=[0.6,0.7,0.8,0.9], y=[1,1,0,1]
  mean(p̂) = 0.75,  mean(y) = 0.75,  |diff| = 0.00  ← perfect

ECE = (4/8)×0.00 + (4/8)×0.00 = 0.00   (well calibrated)
```

Now a miscalibrated model on the same y:

```
p̂:   0.6  0.7  0.8  0.9  0.9  0.9  0.9  0.9   (overconfident)
y:    0    0    1    0    1    1    0    1

Bin 1 (p̂ < 0.85): p̂=[0.6,0.7,0.8], y=[0,0,1]
  mean(p̂) = 0.70,  mean(y) = 0.33,  |diff| = 0.37

Bin 2 (p̂ ≥ 0.85): p̂=[0.9,0.9,0.9,0.9,0.9], y=[0,1,1,0,1]
  mean(p̂) = 0.90,  mean(y) = 0.60,  |diff| = 0.30

ECE = (3/8)×0.37 + (5/8)×0.30 = 0.139 + 0.188 = 0.327  (poor calibration)
```

The overconfident model always predicts high but only 62.5% are actually positive.

### How to fix calibration

**Method 1 — Platt Scaling**

Fit a logistic regression on top of the model's raw scores:

```
Step 1: Train original model, get scores sᵢ (logits or probabilities)
Step 2: On held-out validation set, fit:
         p_calibrated = σ(a·sᵢ + b)
         where a, b are learned by MLE on (sᵢ, yᵢ) pairs

Platt scaling is logistic regression used as a calibration layer.
```

If a=1, b=0 → no correction needed (already calibrated).  
If a < 1 → the model was overconfident (Platt compresses scores toward 0.5).  
If b ≠ 0 → the model had a systematic bias.

**Hand calculation of Platt scaling:**

```
Raw scores (logits):  s = [−2, −1, 0, 1, 2]
True labels:          y = [0,   0,  1, 1, 1]

Fit σ(a·s + b) by MLE.
Initial: a=1, b=0  →  p̂ = σ(s) = [0.12, 0.27, 0.50, 0.73, 0.88]

Observed rates:
  s<0: 2 obs, 0 positives  → observed = 0.0, predicted = ~0.2  (too high)
  s≥0: 3 obs, 3 positives  → observed = 1.0, predicted = ~0.7  (too low)

After fitting a=0.8, b=−0.3:
  p̂ = σ(0.8s − 0.3) → better aligned with observations
```

**Method 2 — Isotonic Regression**

Non-parametric. Fits a monotone step function mapping raw scores to calibrated probabilities. More flexible than Platt but needs more data.

```
sklearn:
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling
cal_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)

# Isotonic regression  
cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)

cal_model.fit(X_train, y_train)
p_calibrated = cal_model.predict_proba(X_test)[:, 1]
```

**Method 3 — Temperature scaling** (neural nets / deep learning)

Scale all logits by a single temperature parameter T:

```
p_calibrated = softmax(z / T)

T > 1:  softens probabilities (reduces overconfidence)
T < 1:  sharpens probabilities (increases confidence)
T = 1:  no change
```

Find T by minimising NLL on a held-out validation set.

### Calibration vs discrimination

| | Calibration | Discrimination (AUC) |
|---|---|---|
| What it measures | Are probability values accurate? | Are observations correctly ranked? |
| Perfect score | ECE = 0, HL non-significant | AUC = 1.0 |
| Can you have one without the other? | Yes — high AUC, poor calibration (common) | Yes — perfect calibration, AUC=0.5 |
| Fixing one fixes the other? | No | No |
| When it matters most | Probability used directly (pricing, decisions) | Binary classification, ranking |

A model can rank patients perfectly by risk (AUC=1) but assign probabilities of 0.9 to everyone — not calibrated. A model can be perfectly calibrated (every 0.7 has 70% outcome rate) but fail to discriminate (AUC = 0.5). They measure orthogonal properties.

---

## Quick Reference

```
Test                Formula                       Dist under H₀    Use when
─────────────────────────────────────────────────────────────────────────────
Wald               W = (ŵ/SE)²                   χ²(1)            Quick, large n
LRT                G² = 2(ℓ_full − ℓ_red)        χ²(df)           Small n, separation
Score              S = gradient at H₀             χ²(k)            Only fit null model
Omnibus            G² = 2(ℓ_full − ℓ_null)       χ²(p)            Test whole model
Hosmer-Lemeshow    Σ(O−E)²/(n·p̄·(1−p̄))          χ²(g−2)          Calibration/GoF

Pseudo-R²          McFadden = 1 − ℓ_full/ℓ_null
                   Nagelkerke = rescaled Cox-Snell to [0,1]

Calibration        ECE = Σ(nₖ/n)|p̄ₖ − ȳₖ|       → 0 is perfect
Fix calibration    Platt scaling, isotonic regression, temperature scaling
```
