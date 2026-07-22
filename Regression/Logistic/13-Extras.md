# Logistic Regression — Extended Master Reference
### (Companion to Modules 0–13 — everything covered since, merged, deduplicated, and organized)

**Format key** used throughout every topic: WHY → INTUITION → FORMULA → WORKED EXAMPLE → INTERPRETATION → FAANG L5 ANGLE → CHECK (some foundational sections use a flatter derivation style since they were originally reference notes, not interview-drill topics — noted where that's the case).

---

## TABLE OF CONTENTS

**PART A — Foundations & First Principles**
1. Prior, Likelihood, Posterior, MLE, MAP
2. MLE Derivation for Logistic Regression (from scratch, plus two hand-worked numeric examples)
3. Why Sigmoid? (three independent derivations)
4. "Levels of Linearity" — disambiguating three senses of "linear"
5. Why Not MSE for Classification (full reasons + the algebraic gradient-cancellation proof)
6. Decision Boundary — what it is, why it's linear (geometric + algebraic proof), threshold-shift behavior
7. Modeling Non-Linearity — feature engineering techniques + what plain LR fundamentally cannot do (XOR)
8. Discriminative vs. Generative Models (Logistic Regression vs. Naive Bayes)

**PART B — Statistical Inference**
9. Standard Errors, the Wald Test, and the Hauck-Donner Failure Mode
10. The Likelihood Ratio Test (feature significance) and as a Goodness-of-Fit Test (vs. saturated model)
11. The Score (Rao) Test, and the Wald/LRT/Score Asymptotic Trinity
12. Confidence Intervals on Coefficients & Odds Ratios
13. Goodness of Fit: Deviance, Hosmer-Lemeshow, Pearson χ², Pseudo-R² (McFadden/Cox-Snell/Nagelkerke), AIC/BIC

**PART C — Calibration**
14. Calibration — Diagnosis (reliability diagrams, ECE, Hosmer-Lemeshow) and Fixes (Platt, Isotonic, Temperature Scaling)

**PART D — Model Building Practicalities**
15. Feature Importance (standardized coefficients vs. permutation importance)
16. Model Selection (train/val/test, cross-validation)
17. Feature Leakage — the four types, deep dive
18. Missing Data Handling
19. Outlier Handling
20. Feature Scaling and Gradient Descent Convergence
21. Separation / Quasi-Complete Separation (and its link to the Hauck-Donner phenomenon)

**PART E — Fairness & Responsible ML**
22. Fairness & Bias

**PART F — Modeling Variants & Multiclass**
23. Elastic Net
24. Weighted Logistic Regression (sample weights vs. class weights)
25. Multiclass: Softmax / Multinomial Logistic Regression (full derivation)
26. Ordinal Logistic Regression
27. Rare-Event Bias Correction (Firth's Correction)

**PART G — Systems & Scale**
28. Sparse / High-Cardinality Categorical Features (hashing trick, embeddings, Wide & Deep)
29. Real-Time vs. Batch Features & Feature Stores
30. Online / Incremental Learning

**PART H — GLMs & Neural Network Comparison**
31. Generalized Linear Models — the unifying framework
32. Neural Networks vs. Logistic Regression

**PART I — sklearn & Imbalanced Data**
33. `sklearn.LogisticRegression` — every parameter explained
34. Imbalanced Data — the full toolkit, with the `class_weight='balanced'` math

**PART J — Comparisons & Pitfalls**
35. Logistic Regression vs. Linear Regression
36. Logistic Regression vs. Decision Trees
37. Common Mistakes & Pitfalls — consolidated list

**PART K — Reference Tables & Checks**
38. Master Quick-Reference Tables
39. Consolidated Interview Check Questions (with model answers, where given)

---
---

# PART A — FOUNDATIONS & FIRST PRINCIPLES

## 1. Prior, Likelihood, Posterior, MLE, MAP

### 1.1 The three quantities

| Term | Notation | Question it answers |
|---|---|---|
| **Prior** | P(θ) | What do I believe about θ *before* seeing data? |
| **Likelihood** | P(X \| θ) | How well does this data fit, *given* θ? |
| **Posterior** | P(θ \| X) | What do I believe about θ *after* seeing data? |

### 1.2 Bayes' theorem

$$P(\theta \mid X) = \frac{P(X \mid \theta) \cdot P(\theta)}{P(X)}$$

Since P(X) is just a normalizing constant (doesn't depend on θ):

```
posterior  ∝  likelihood × prior
```

The prior is your belief. The likelihood is the update signal from data. The posterior is the result.

### 1.3 What "likelihood" actually means

P(X | θ) looks like a conditional probability over X — but in likelihood thinking, **X is fixed** (you already observed it) and **θ varies**. You're asking: "for each possible θ, how probable is this exact dataset?"

- Not a distribution over X
- A scoring function over θ
- Written L(θ ; X) when you want to emphasize this

### 1.4 Concrete example — biased coin

You flip a coin 10 times and get 8 heads. θ = true probability of heads.

```
Prior:      P(θ) = Uniform[0,1]   # no prior knowledge
Likelihood: P(X | θ) = C(10,8) · θ⁸ · (1-θ)²
Posterior:  P(θ | X) ∝ θ⁸ · (1-θ)²   # peaks near θ = 0.8
```

The prior is flat, so the posterior just follows the likelihood here. If the prior had been peaked at 0.5 (strong belief in fairness), the posterior would land somewhere between 0.5 and 0.8.

### 1.5 MLE — Maximum Likelihood Estimation

**Throw the prior away. Find the θ that makes the data most probable.**

```
θ̂_MLE = argmax_θ  P(X | θ) = argmax_θ  L(θ ; X)
```

For the coin: MLE gives θ̂ = 0.8 exactly (8/10 heads).

**Why log-likelihood?** Likelihoods are products of many small probabilities → numerical underflow. Log turns products into sums, and argmax is preserved under monotone transforms.

```
ℓ(θ) = log L(θ ; X) = Σᵢ log P(xᵢ | θ)
θ̂_MLE = argmax_θ  ℓ(θ)
```

### 1.6 MAP — Maximum A Posteriori

**Keep the prior. Find the peak of the posterior instead.**

```
θ̂_MAP = argmax_θ  P(θ | X) = argmax_θ  [log P(X | θ) + log P(θ)]
       = MLE objective  +  log prior term
```

| Prior on θ | Equivalent regularization |
|---|---|
| Gaussian N(0, σ²) | L2 (Ridge) |
| Laplace | L1 (Lasso) |
| Uniform | No regularization = MLE |

**Regularization isn't a hack — it's the log prior in MAP estimation.** This single fact is worth having ready verbatim in interviews: whenever asked "why does L2 regularization work," the deepest answer is "it's equivalent to placing a Gaussian prior on the weights and doing MAP instead of MLE."

### Interview punchlines
- **"Why cross-entropy loss?"** — It's the negative log-likelihood under a Bernoulli model. Not arbitrary — it's the principled MLE objective.
- **"What's the difference between MLE and MAP?"** — MAP adds a log prior term. With a Gaussian prior this equals L2 regularization; with Laplace, L1.

---

## 2. MLE Derivation for Logistic Regression (From Scratch)

### 2.1 Setup

- Binary labels yᵢ ∈ {0, 1}, features xᵢ ∈ ℝᵈ, parameters w
- Model: output a probability via sigmoid

```
σ(z) = 1 / (1 + e^(−z))
p̂ᵢ = σ(wᵀxᵢ)
```

### 2.2 Step 1 — write the likelihood of one observation

Each label is Bernoulli(p̂ᵢ). If they actually churned (y=1), the model explains that outcome with probability p̂. If not (y=0), with probability (1−p̂). Both cases are captured in one unified expression:

```
P(yᵢ | xᵢ, w) = p̂ᵢ^yᵢ · (1 − p̂ᵢ)^(1−yᵢ)
```

Check: y=1 → p̂¹(1−p̂)⁰ = p̂ ✓. y=0 → p̂⁰(1−p̂)¹ = (1−p̂) ✓.

### 2.3 Step 2 — the whole dataset's likelihood

Assuming i.i.d. rows (independence — see Module 10), the probability of the entire dataset is the **product** of each row's probability ("AND" logic for independent events):

```
L(w) = ∏ᵢ  p̂ᵢ^yᵢ · (1 − p̂ᵢ)^(1−yᵢ)
```

### 2.4 Step 3 — take the log (the key trick)

Multiplying thousands of small probabilities produces a tiny, numerically unstable number (underflow risk). `log(a×b) = log(a) + log(b)` turns the product into a sum — cheap and numerically stable, and argmax is preserved because log is monotonic.

```
ℓ(w) = Σᵢ [ yᵢ · log(p̂ᵢ)  +  (1−yᵢ) · log(1 − p̂ᵢ) ]
```

**This is exactly the negative of binary cross-entropy loss.** Maximizing ℓ(w) = minimizing cross-entropy — they are the same optimization problem, opposite sign.

### 2.5 Step 4 — gradient

Using σ'(z) = σ(z)(1−σ(z)):

```
∂ℓ/∂w = Σᵢ (yᵢ − p̂ᵢ) · xᵢ            [scalar-per-sample view]
∇_w ℓ = Xᵀ(y − p̂)                     [matrix form]
```

Gradient = residuals (yᵢ − p̂ᵢ) weighted by features — same structure as linear regression. Both are GLMs (Part H).

### 2.6 Step 5 — no closed form → gradient ascent/descent

Setting ∇ℓ = 0 has no closed-form solution because p̂ᵢ = σ(wᵀxᵢ) is nonlinear in w.

```
w ← w + α · Xᵀ(y − p̂)      # gradient ascent on ℓ (maximize likelihood)
w ← w − α · Xᵀ(p̂ − y)      # gradient descent on loss (minimize cross-entropy) — same update
```

### 2.7 Step 6 — Hessian (why convergence is guaranteed)

```
H = −Xᵀ W X,   where W = diag(p̂ᵢ(1 − p̂ᵢ))
```

H is negative semi-definite → ℓ is **concave** → unique global maximum → the solver always converges to the same answer regardless of initialization. (Full convexity proof for the *loss* — the mirror-image statement — is in §5.3.)

### 2.8 Full derivation in one view

```
1.  Model:      p̂ᵢ = σ(wᵀxᵢ)
2.  Likelihood: L = ∏ᵢ p̂ᵢ^yᵢ (1−p̂ᵢ)^(1−yᵢ)
3.  Log-lik:    ℓ = Σᵢ [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]
4.  Gradient:   ∇_w ℓ = Xᵀ(y − p̂)
5.  No closed form → gradient ascent
6.  ℓ is concave → unique global max
```

### 2.9 Worked numeric example #1 — deriving log-loss on a 4-row table

Reuse the standard 4-customer table:

| Customer | y | p̂ |
|---|---|---|
| 1 | 1 | 0.9 |
| 2 | 1 | 0.1 |
| 3 | 0 | 0.2 |
| 4 | 0 | 0.8 |

**Likelihood of the whole dataset (product of individual row probabilities):**
```
L = 0.9 × 0.1 × (1-0.2) × (1-0.8) = 0.9 × 0.1 × 0.8 × 0.2 = 0.0144
```
**Log-Likelihood (sum of logs):**
```
log(L) = log(0.9) + log(0.1) + log(0.8) + log(0.2)
       = -0.105 + (-2.303) + (-0.223) + (-1.609) = -4.240
```
**Negative log-likelihood, averaged over 4 rows:**
```
-log(L) / 4 = 4.240 / 4 = 1.06
```
This 1.06 matches the standard "average log-loss" computation exactly — same numbers, now understood as coming directly from the MLE derivation, not handed down as a formula.

### 2.10 Worked numeric example #2 — full gradient-ascent hand trace (tiny 2-point dataset)

Two training points, one feature, no bias for simplicity:
```
x₁ = 1,  y₁ = 1
x₂ = 2,  y₂ = 0
```
Model: p̂ᵢ = σ(w · xᵢ). Log-likelihood:
```
ℓ(w) = log(σ(w)) + log(σ(−2w))     [using log(1−σ(z)) = log(σ(−z))]
```
Gradient:
```
dℓ/dw = (1 − σ(w))·1 + (0 − σ(2w))·2 = (y₁−p̂₁)·x₁ + (y₂−p̂₂)·x₂
```
**Iterate (gradient ascent, lr = 0.5), init w=0:**
```
p̂₁ = σ(0) = 0.5,  p̂₂ = σ(0) = 0.5
grad = (1−0.5)·1 + (0−0.5)·2 = 0.5 − 1.0 = −0.5
w ← 0 + 0.5·(−0.5) = −0.25
```
**Iter 2, w = −0.25:**
```
p̂₁ = σ(−0.25) ≈ 0.438,  p̂₂ = σ(−0.50) ≈ 0.378
grad = (1−0.438)·1 + (0−0.378)·2 = 0.562 − 0.756 = −0.194
w ← −0.25 + 0.5·(−0.194) = −0.347
```
w keeps moving negative — sensible, since x₂=2 has label 0, so the model learns to push high-x predictions toward 0, meaning negative w. As iterations → ∞, **w → −∞** here: this tiny 2-point dataset is perfectly separable, so MLE is unbounded — a preview of the Separation topic (Part D §21) and a live demonstration of why regularization is needed in practice.

**Why no closed form?** Setting dℓ/dw = 0 gives `(1 − σ(w)) − 2·σ(2w) = 0` — σ is nonlinear, you can't isolate w algebraically. Numerical optimization only.

### 2.11 A third worked trace — the standard "one full training iteration" interview answer

Setup: 1 feature (complaints), 2 customers, starting weights b=0, w=0, learning rate 0.1.

| Customer | x (complaints) | y (actual churn) |
|---|---|---|
| 1 | 2 | 1 |
| 2 | 1 | 0 |

**Forward pass:**
```
Customer 1: z = 0 + (0×2) = 0 → p = σ(0) = 0.5
Customer 2: z = 0 + (0×1) = 0 → p = σ(0) = 0.5
```
**Loss:**
```
Customer 1: y=1, p=0.5 → loss = -log(0.5) = 0.693
Customer 2: y=0, p=0.5 → loss = -log(0.5) = 0.693
average loss = 0.693
```
**Gradient:**
```
(p−y) for Cust 1 = -0.5,  for Cust 2 = 0.5
gradient_w = [(-0.5×2) + (0.5×1)] / 2 = [-1.0+0.5]/2 = -0.25
gradient_b = [-0.5+0.5]/2 = 0
```
**Update:**
```
new_w = 0 - (0.1 × -0.25) = 0.025
new_b = 0 - (0.1 × 0) = 0
```
After one full pass: w=0.025, b=0 — moved positive, correctly, since the customer with more complaints (Cust 1) churned. This exact 4-step sequence (forward pass → loss → gradient → update) is what happens every iteration during `.fit()`, and is one of the highest-value things to be able to reproduce fluently, with real numbers, live in an interview.

> **Check:** If Customer 1 had 5 complaints instead of 2, would `gradient_w` become MORE negative or MORE positive (before the update)? Reason it through without recomputing everything. *(It becomes more negative — Customer 1's (p−y)×x term scales up in magnitude since x grew, and that term is already negative, pulling the average further negative — i.e., an even stronger positive push to w after the sign flip in the update rule.)*

> **Check:** Why do we take the PRODUCT of individual row probabilities (not the sum) to get the whole dataset's likelihood? *(Because independent events "all happening together" combine via multiplication — P(A and B) = P(A)×P(B) for independent A, B. Summing would answer a different question — "the probability that at least one of these specific outcomes happens" — not the joint probability of the observed dataset exactly as it occurred.)*

---

## 3. Why Sigmoid? (Three Independent Derivations)

Each of the three reasons below is independently sufficient — sigmoid isn't an arbitrary design choice, it falls out of the model's assumptions.

### 3.1 Reason 1 — Probabilistic derivation (the real reason)

Assume the log-odds of the event is linear in x (the foundational Module-1 assumption):
```
log(p / (1−p)) = wᵀx
```
Solve for p:
```
p / (1−p) = e^(wᵀx)
p = e^(wᵀx)·(1−p)
p = e^(wᵀx) − p·e^(wᵀx)
p(1 + e^(wᵀx)) = e^(wᵀx)
p = e^(wᵀx) / (1 + e^(wᵀx)) = 1 / (1 + e^(−wᵀx))
```
That's σ. It didn't come from nowhere — it's the **inverse of the logit function**. If you believe log-odds is linear, sigmoid is the forced mathematical consequence, not a free choice.

### 3.2 Reason 2 — Bernoulli exponential family

The Bernoulli distribution's natural parameter IS the log-odds. Sigmoid is the mean function of the Bernoulli in exponential-family form — logistic regression is a GLM with the *canonical link* (see Part H §31), so sigmoid emerges directly from the probability model, not from a design decision.

### 3.3 Reason 3 — Output range and gradient convenience

- Output ∈ (0,1): valid probability, no clipping needed
- Differentiable everywhere
- σ'(z) = σ(z)(1−σ(z)): derivative expressed in terms of itself → cheap to compute in backprop

### 3.4 Sigmoid vs. other activation functions

| Property | Sigmoid | Tanh | ReLU | Softmax |
|---|---|---|---|---|
| Output range | (0, 1) | (−1, 1) | [0, ∞) | (0,1), sums to 1 |
| Centred at 0? | No | Yes | No | No |
| Vanishing gradient? | Yes (tails) | Yes (tails) | No (positive side) | Yes (tails) |
| Use case | Binary output | Hidden layers (old) | Hidden layers (modern) | Multiclass output |

**Vanishing gradient with sigmoid:** σ'(z) = σ(z)(1−σ(z)) maxes out at 0.25 (at z=0). In deep networks, backprop multiplies gradients through many layers — a chain of numbers ≤0.25 shrinks to zero exponentially fast. Tanh has the same problem (max derivative 1.0, but still squashes). ReLU fixes this (derivative is 1 for z>0, gradients don't shrink) but isn't a valid probability, so it can't be the final classification layer.

**Sigmoid for output, not hidden layers:** in hidden layers, sigmoid/tanh have mostly been replaced by ReLU and variants (better gradient flow). At the output layer, sigmoid remains correct for binary classification because we need P(y=1|x) ∈ (0,1).

**Sigmoid vs. Softmax:** Softmax is the multiclass generalization of sigmoid. For K=2 classes, softmax reduces exactly to sigmoid:
```
softmax(z₁) = e^z₁/(e^z₁+e^z₂) = 1/(1+e^(z₂−z₁)) = σ(z₁ − z₂)
```
Full softmax derivation is in Part F §25.

---

## 4. "Levels of Linearity" — Disambiguating Three Senses of "Linear"

This phrase is worth unpacking because "linear" gets used in at least three different senses in this curriculum, and conflating them is a common, sneaky source of confusion — and a favorite trap question.

**Sense 1 — Linear in the PARAMETERS (always true, by definition).** `z = b + w1·x1 + w2·x2` is linear in `b, w1, w2` — this is what makes logistic regression a "Generalized LINEAR Model" (GLM, Part H). Always true, no matter what features you feed in (even x1² or interaction terms) — the weights themselves always combine additively.

**Sense 2 — Linear in the ORIGINAL features, w.r.t. log-odds (the assumption that CAN be violated).** Whether log-odds moves in a straight line as your raw, un-engineered feature changes. This is the assumption you check (binned log-odds plots) and can fix (polynomial features, binning — Part A §7).

**Sense 3 — NOT linear in probability (never true, and intentionally so).** The relationship between z and probability is always the sigmoid S-curve — never linear by design, and not something you'd want to "fix."

**FAANG trap question:** *"Is logistic regression a linear model?"* — Strong answer: "Yes, in the sense that it's linear in its parameters (that's the 'linear' in GLM) and it assumes linearity between features and log-odds — but the relationship with raw probability is explicitly non-linear (the sigmoid S-curve), and if the log-odds relationship with a raw feature is genuinely non-linear, we fix that through feature engineering, not by changing the model's fundamental linear-in-parameters structure." A one-word answer without this distinction is exactly the kind of imprecision interviewers flag.

---

## 5. Why Not MSE for Classification (Full Treatment + Proof)

### 5.1 Reason 1 — wrong probability model

MSE assumes Gaussian noise around predictions: `y = wᵀx + ε, ε ~ N(0,σ²)`. Binary labels are Bernoulli — they can only be 0 or 1. Gaussian noise on a binary variable makes no probabilistic sense. Cross-entropy is correct because it's the NLL of the Bernoulli model (derived from scratch in §2).

### 5.2 Reason 2 — non-convexity with sigmoid

```
L_MSE = (1/n) Σᵢ (yᵢ − σ(wᵀxᵢ))²
```
The Hessian of this is not guaranteed PSD — you can construct cases with negative eigenvalues, meaning the loss surface has saddle points and local minima. No clean algebraic cancellation exists here the way it does for cross-entropy (proof in §5.3 below shows the cross-entropy case IS convex).

### 5.3 Reason 3 — flat gradients (the practical killer)

MSE gradient w.r.t. output: `∂L/∂ŷ = 2(ŷ−y)`. Chain rule through sigmoid:
```
∂L/∂w = 2(ŷ−y) · σ'(z) · x = 2(ŷ−y) · σ(z)(1−σ(z)) · x
```
When the model is very wrong (e.g. y=1 but ŷ≈0), σ(z)≈0 so σ'(z)≈0 — **the gradient vanishes precisely when it should be largest.** Learning slows to a crawl exactly when the model is confidently wrong.

With cross-entropy:
```
∂L/∂w = (ŷ−y)·x
```
The σ'(z) term cancels algebraically. The gradient is just the residual — large when wrong, small when right. This is why cross-entropy trains dramatically faster.

### 5.4 Reason 4 — output interpretation / asymmetric penalty

MSE doesn't push predictions toward 0 or 1 — a prediction of 0.5 for a clear positive example is penalized the same as 0.4 or 0.6. Cross-entropy is asymmetric: it heavily penalizes confident wrong predictions (`−log(0.01) = 4.6` vs. `−log(0.5) = 0.69`).

### 5.5 Summary table

| Issue | MSE | Cross-entropy |
|---|---|---|
| Probability model | Gaussian (wrong) | Bernoulli (correct) |
| Loss surface | Non-convex (with sigmoid) | Convex |
| Gradient when confident & wrong | Near zero | Large |
| Gradient formula | (ŷ−y)·σ'(z)·x | (ŷ−y)·x |

**FAANG framing:** always cite BOTH the convexity argument (optimization) AND the confident-wrong-penalty shape (statistical behavior) — citing only one is a partial answer.

### 5.6 Full proof — cross-entropy + sigmoid IS convex

A function is convex if its Hessian is PSD everywhere.

**The loss:**
```
L(w) = −Σᵢ [ yᵢ log(p̂ᵢ) + (1−yᵢ) log(1−p̂ᵢ) ],  p̂ᵢ = σ(wᵀxᵢ)
```
**Gradient (derived in §2.5):** `∇_w L = Σᵢ (p̂ᵢ − yᵢ) xᵢ = Xᵀ(p̂ − y)`

**Hessian:** differentiate the gradient w.r.t. w:
```
∂/∂w [(p̂ᵢ − yᵢ)xᵢ] = (∂p̂ᵢ/∂w) xᵢ = σ'(wᵀxᵢ) xᵢxᵢᵀ = p̂ᵢ(1−p̂ᵢ) xᵢxᵢᵀ
```
Summing over all samples: `H = Σᵢ p̂ᵢ(1−p̂ᵢ) xᵢxᵢᵀ = XᵀWX`, where `W = diag(p̂ᵢ(1−p̂ᵢ))`.

**Show H is PSD:** for any vector v ∈ ℝᵈ:
```
vᵀHv = vᵀ(XᵀWX)v = (Xv)ᵀW(Xv) = Σᵢ wᵢᵢ(xᵢᵀv)² = Σᵢ p̂ᵢ(1−p̂ᵢ)·(xᵢᵀv)²
```
Since p̂ᵢ ∈ (0,1), `p̂ᵢ(1−p̂ᵢ) > 0` always, and `(xᵢᵀv)² ≥ 0` always. Every term is ≥0, so **vᵀHv ≥ 0 for all v → H is PSD → L(w) is convex. QED.**

**What this guarantees:** no local minima (any local min is the global min); gradient descent always converges to the same solution regardless of initialization; the solution is unique if X has full column rank (H becomes PD, not just PSD).

**Why MSE + sigmoid is NOT convex, algebraically:** the Hessian there involves second derivatives of σ (σ''(z) terms) which can be negative — no clean cancellation exists, and you can construct cases with negative eigenvalues (saddle points / local minima).

---

## 6. Decision Boundary — What It Is, Why It's Linear

### 6.1 Definition

The decision boundary is the set of points where the model sits exactly at the classification threshold:
```
p̂ = σ(wᵀx + b) = threshold
```
At the common default threshold=0.5: since σ(z)=0.5 exactly when z=0, the boundary is:
```
wᵀx + b = 0
```
That's a hyperplane — a line in 2D, a plane in 3D.

### 6.2 Intuition

`w` is a vector perpendicular to the boundary. Points where `wᵀx+b > 0` get p̂>0.5 (class 1); points where `wᵀx+b < 0` get p̂<0.5 (class 0). The *magnitude* of `wᵀx+b` tells you how far from the boundary a point is — further away = more confident. Far from the boundary, p̂ → 0 or 1 (confident); at the boundary, p̂=0.5 (uncertain). Sigmoid squashes linear distance-from-boundary into a probability.

```
Feature 2
    |     class 1 (p̂ > 0.5)
    |   /
    |  /  ← decision boundary: w₁x₁ + w₂x₂ + b = 0
    | /
    |/   class 0 (p̂ < 0.5)
    +------------- Feature 1
```

### 6.3 Different thresholds → parallel shifted lines (not curves)

If you pick a threshold other than 0.5 (say 0.7), the boundary shifts to wherever z equals `logit(0.7) ≈ 0.847` (solving `sigmoid(z)=0.7`). The decision boundary is literally the geometric line traced out by your threshold condition, plotted in feature space — different thresholds give different but still straight, parallel boundary lines, since they're all still just "z = some constant" equations.

> **Check:** moving the threshold from 0.5 to 0.9 — does the boundary move CLOSER to the positive cluster, or FARTHER from it? *(Closer to the positive cluster — customers now need to look MORE convincingly at-risk (higher z) to cross the higher bar, so the boundary shifts toward the region that used to be confidently "positive.")*

### 6.4 Formal proof that the boundary is always linear

**Given:** `z = b + w1x1 + w2x2` (definitional — always true for logistic regression).
**Decision rule:** classify positive when `p ≥ threshold`.

**Step 1:** Since sigmoid is strictly increasing (bigger z always gives bigger p, never reverses), `p ≥ threshold` is equivalent to `z ≥ some_constant`, where `some_constant = logit(threshold)`.

**Step 2:** Substitute: `b + w1x1 + w2x2 ≥ some_constant`.

**Step 3:** This is a linear inequality in x1, x2 — the boundary (where it's exactly equal) is:
```
b + w1x1 + w2x2 = some_constant
```

**Step 4:** Rearranged, this is the equation of a straight line (2D) or flat hyperplane (higher-D) — same form as `Ax+By=C`. **QED.**

> **Check:** if you added an x1² term (feature engineering, §7), would this proof still hold with x1² substituted in as if it were just another linear input? What would the boundary look like translated back into terms of the *original* x1? *(Yes — the proof holds identically in the expanded feature space {x1, x1²}; the boundary is linear in that expanded space, e.g. `b + w1·x1 + w2·x1² = const`. Translated back into raw x1 terms, this is a quadratic equation — a parabola-shaped boundary in the original 1-D feature, even though it's a straight line in the engineered 2-D feature space.)*

---

## 7. Modeling Non-Linearity — Feature Engineering Techniques (and What Plain LR Cannot Do)

### 7.1 WHY

Logistic regression assumes a straight-line relationship between features and log-odds (§4, Sense 2). Real relationships are often genuinely curved — e.g. churn risk might be high for brand-new customers, drop as they settle in, then rise again near contract renewal (a U-shape). A plain linear term can only draw one straight line, no matter how badly that fits a curved truth — this costs real accuracy exactly on the customers/ranges you may care about most (the extremes).

### 7.2 The core idea behind every technique below

Logistic regression can only ever draw straight lines (in log-odds space) — so if you want it to capture a curve, you must **feed it inputs that already contain the curve's shape**, rather than expecting the model to invent curvature on its own. You're doing the "curving" manually, before training, so a straight-line model on the NEW features produces curved behavior in the ORIGINAL feature.

### 7.3 Technique 1 — Polynomial Features

**In words:** alongside the original feature, create a new feature that's the original squared (or cubed). Feed both in.
```
z = b + w1·x + w2·x²
```
Even though the model is still doing simple weighted addition (linear in x and x²), the relationship between z and the *original* x becomes curved, because x² grows non-linearly as x changes.

**Worked example:** suppose `w1=-0.5, w2=0.02, b=2.0` for tenure (x):

| Tenure (x) | x² | z = 2.0 − 0.5x + 0.02x² | Interpretation |
|---|---|---|---|
| 0 | 0 | 2.0 | high log-odds of churn (new customer) |
| 12 | 144 | −1.12 | low log-odds (settled in) |
| 24 | 576 | 1.52 | log-odds rising again (renewal risk) |

A clean U-shape emerges directly from the numbers — a shape a plain linear term could never produce.

### 7.4 Technique 2 — Binning (Discretization)

**In words:** chop the raw continuous number into ranges/buckets (e.g. "0–6 months," "6–12 months"), and let the model learn a completely separate coefficient per bucket. Each bucket gets its own independent "vote," with no assumption that neighboring buckets follow any smooth mathematical pattern relative to each other — this can capture ANY shape (sharp jumps, U-shapes, anything), at the cost of losing the smooth continuous nature of the feature and needing enough data per bucket for reliable estimates.

### 7.5 Technique 3 — Interaction Terms

**WHY this is a different kind of non-linearity:** sometimes the issue isn't one feature having a curved effect alone — it's that TWO features' effects depend on each other. Example: "number of complaints" might matter a lot for NEW customers, but barely for long-tenured loyal customers.

```
z = b + w1·x1 + w2·x2 + w3·(x1×x2)
```
`w3` captures how the COMBINATION behaves, beyond what each contributes alone.

**Worked example:** `w1=0.6` (complaints alone), `w2=-0.05` (tenure alone), `w3=-0.03` (interaction). For a NEW customer (tenure=2) with 3 complaints: interaction term = 6 → contributes `-0.18` to z. For a LOYAL customer (tenure=30) with 3 complaints: interaction term = 90 → contributes `-2.7` to z. The **same 3 complaints** get treated very differently depending on tenure — capturing "loyal customers tolerate complaints better," a pattern neither feature alone could express.

### 7.6 Technique 4 — Feature Crosses (Categorical Version)

**WHY separately notable (Google-specific framing):** interaction terms above assumed numeric features. When BOTH features are categorical (device_type: mobile/desktop, user_segment: new/returning), you create a **feature cross**: a new categorical feature representing every COMBINATION (e.g. "mobile_new," "mobile_returning," "desktop_new," "desktop_returning" — 4 new binary indicators). This lets the model learn that "mobile users who are new" behave completely differently from "mobile users who are returning," rather than assuming device type and segment contribute independently/additively. Extremely common at scale in ad click-through-rate prediction — crossing "query type" × "ad category" is a textbook Google-style move.

### 7.7 When to use which

| Situation | Technique |
|---|---|
| One feature has a smooth curved (not wild) relationship with log-odds | Polynomial features (x², x³) |
| One feature has an unpredictable, sharp, or arbitrary-shaped relationship | Binning |
| Two numeric features' effects depend on each other | Interaction terms (numeric × numeric) |
| Two categorical features' effects depend on each other | Feature crosses (categorical × categorical) |

### 7.8 What logistic regression fundamentally CANNOT do: XOR

The raw model `wᵀx+b=0` is always a hyperplane — logistic regression is a **linear classifier** in feature space, full stop. Without feature engineering, it cannot separate XOR-structured data:
```
Class 0: (0,0), (1,1)
Class 1: (0,1), (1,0)
```
No straight line separates these. A neural network with a hidden layer solves this by *learning* the feature transformation automatically (see Part H §32) — that's the fundamental advantage of deep learning over logistic regression. The same idea (implicitly mapping to a higher-dimensional space) also powers kernel methods in SVMs; logistic regression instead does it via *explicit, manual* feature engineering (§7.3–7.6).

### 7.9 FAANG L5 angle

**Q: "Your logistic regression isn't capturing a non-linear relationship you suspect exists. What do you do?"**
Strong answer: check via a binned log-odds plot first, then propose the fix matching the SHAPE of non-linearity — polynomial for smooth curves, binning for arbitrary shapes, interaction terms/feature crosses if the issue is between two features rather than one.

**Q (Google-specific follow-up): "What's a feature cross, and why is it useful at scale?"**
A new categorical feature representing every combination of two original categorical features, letting the model learn combination-specific effects. Extremely common in ad ranking/CTR prediction.

**Common trap:** jumping straight to "switch to a more complex model" without recognizing manual feature engineering often lets you keep a simpler, cheaper, more interpretable model while still capturing meaningful non-linearity.

**Another trap:** adding polynomial/interaction terms indiscriminately without checking overfitting risk — every new engineered feature is another parameter that can memorize noise; add deliberately, guided by a diagnostic or domain reasoning.

```python
import numpy as np
tenure = np.array([0, 12, 24])
b, w1, w2 = 2.0, -0.5, 0.02
z = b + w1*tenure + w2*(tenure**2)

complaints = np.array([3, 3]); tenure2 = np.array([2, 30]); w3 = -0.03
interaction_contribution = w3 * (complaints * tenure2)
```

> **Check:** "Distance from nearest store" is suspected to have a U-shaped relationship with delivery-app usage (too close = don't need it, too far = out of service area, medium = highest usage). Which technique first, and why? *(Polynomial features — it's a single feature with a smooth, curved (not arbitrary/jagged) relationship, the textbook case for x².)*

> **Check:** Why does adding `x1×x2` NOT violate "linearity in log-odds," even though the resulting relationship with the ORIGINAL features is clearly non-linear? *(Because the model is still linear in its parameters and in the *engineered* feature x1×x2 — it's linear in the expanded feature space. "Linearity in log-odds" (Sense 2, §4) concerns the raw, un-engineered features; feature engineering is precisely the sanctioned way to introduce curvature without breaking the parameter-linearity that makes the model a GLM.)*

---

## 8. Discriminative vs. Generative Models (Logistic Regression vs. Naive Bayes)

### 8.1 WHY this matters

A favorite "does the candidate understand the landscape" question. Logistic regression is **discriminative**; Naive Bayes (and LDA) are **generative** — interviewers often ask you to contrast them to test whether you understand *why* you'd pick one over the other.

### 8.2 Intuition

**Discriminative models** learn the boundary between classes directly — "given these features, which side of the line are you on?" — without modeling how the data was generated. **Generative models** learn how EACH CLASS generates its data — "if this were a churner, what would their features typically look like?" — then flip that via Bayes' rule into a classification decision.

*Analogy:* Discriminative = a bouncer who's learned "people wearing X get turned away, Y get in" without knowing why. Generative = a bouncer who understands "regulars dress like THIS, troublemakers like THAT," and reasons from that fuller model.

### 8.3 What each actually models

**Discriminative (logistic regression):** directly models `P(y|x)`. That's the *entire* modeling target.

**Generative (e.g. Naive Bayes):** models `P(x|y)` per class and `P(y)` (base rate), combined via Bayes:
```
P(y|x) = [P(x|y)·P(y)] / P(x)
```

**Naive Bayes' core assumption:** features are conditionally independent given the class:
```
P(x|y) = P(x1|y)·P(x2|y)·...·P(xd|y)
```
Almost always false in practice (words in text aren't independent; vitals in medicine aren't independent) — yet NB often works anyway.

**Why NB works despite the wrong assumption:** the decision boundary only requires the *ranking* of P(y=1|x) vs P(y=0|x) to be correct — not the exact probability values. The independence assumption distorts probabilities but often preserves correct ordering.

**Mathematical relationship:** with Gaussian NB (features ~ Gaussian given class), the log-odds works out to `log[P(y=1|x)/P(y=0|x)] = wᵀx+b` — exactly logistic regression's model form. Gaussian NB and logistic regression **share the same decision-boundary shape**; the difference is *how* they estimate w — NB estimates class-conditional distributions separately, LR estimates w directly to maximize P(y|x).

### 8.4 Worked conceptual example — spam classification

**Discriminative (LR):** directly learns "if this email has word 'free' and 3 exclamation marks, what's P(spam)?" — a direct feature→outcome mapping, no modeling of "typical spam" as a whole.

**Generative (NB):** separately learns "typical word distribution in spam" and "typical word distribution in real email" and "fraction of all email that's spam" — then asks, for a new email, "which of these two generating processes more plausibly produced this exact email?"

### 8.5 Interpretation

Generative models can do things discriminative models can't: GENERATE new synthetic examples (they model the full data distribution), and often need LESS data when their generative assumptions roughly hold. Discriminative models typically achieve better classification accuracy asymptotically (with enough data), since they focus entirely on the boundary rather than the harder, more assumption-laden task of modeling each class's full distribution.

### 8.6 When to use which

| Situation | Prefer |
|---|---|
| Lots of data | Logistic Regression |
| Little data | Naive Bayes (fewer parameters, less overfitting) |
| Features truly independent | Naive Bayes |
| Need calibrated probabilities | Logistic Regression |
| Text classification (bag of words) | Naive Bayes (Multinomial NB works well) |
| Features highly correlated | Logistic Regression |
| Need to generate synthetic data | Naive Bayes (it's generative) |

**Key insight — bias-variance made explicit:** LR is asymptotically better (fewer assumptions, wins with infinite data), but NB can win with small data because its strong independence assumption acts like regularization, reducing variance at the cost of bias.

### 8.7 FAANG L5 angle

**Q: "When would you prefer Naive Bayes over logistic regression?"** — Smaller datasets, need to handle missing features gracefully (NB can more naturally marginalize over missing data), or you actually want to generate synthetic samples. LR preferred with more data and when raw classification accuracy is the goal.

**Common trap:** confusing "generative" (classical ML term, predates and is distinct from) with "generative AI" (GPT etc.) — worth clarifying if the term comes up ambiguously.

> **Check:** Naive Bayes assumes conditional independence (often unrealistic). Why might logistic regression, which makes NO such assumption, often outperform NB with enough data? *(LR directly optimizes P(y|x) without needing the independence assumption to hold — it can capture real feature correlations/interactions that NB's independence assumption forces it to ignore, and with enough data LR's lower bias dominates its (here, minimal) variance cost.)*

---
---

# PART B — STATISTICAL INFERENCE

## 9. Standard Errors, the Wald Test, and the Hauck-Donner Failure Mode

### 9.1 WHY

You've fit a model and have coefficient estimates — but is a coefficient's value a real effect, or noise from a small/unlucky sample? You don't want to refit an entire second model (that's the LRT, §10) just to check one coefficient — the Wald test gives a quick, one-shot answer using only the model you already have.

### 9.2 Intuition

Flip a coin 10 times, get 7 heads — is it biased? With only 10 flips, 70% isn't THAT unusual for a fair coin. Flip 10,000 times and get 7,000 heads (still 70%) — now you'd be much more confident, because that same ratio over a larger sample is far less likely by chance. **The same coefficient value can be rock-solid or pure noise depending on how much data (and variability) is behind the estimate.** The Wald test measures the coefficient relative to its own uncertainty (standard error).

### 9.3 What a standard error represents

How much a coefficient's estimated value would bounce around if you re-collected data and re-fit many times. Small SE = stable, precise estimate. Large SE = shaky, could easily have come out very different with a different sample. You don't need to hand-derive it (it comes from the curvature of the log-likelihood around the fitted weights — your software computes it), but you must know what it represents and how to use it.

### 9.4 Where SE(ŵⱼ) actually comes from

The variance of ŵ is the inverse Fisher Information Matrix:
```
Var(ŵ) = I(ŵ)⁻¹ = (XᵀWX)⁻¹,   W = diag(p̂ᵢ(1−p̂ᵢ))    [the same Hessian from §2.7/§5.6]
SE(ŵⱼ) = √[(XᵀWX)⁻¹]ⱼⱼ
```

### 9.5 The Wald statistic

```
z = ŵⱼ / SE(ŵⱼ)                    Under H₀ (wⱼ=0): z ~ N(0,1) asymptotically
W = z² = ŵⱼ²/Var(ŵⱼ)               Under H₀: W ~ χ²(1)
```
For testing multiple coefficients jointly (H₀: w1=w2=0):
```
W = ŵᵀ [Var(ŵ)]⁻¹ ŵ  ~  χ²(k),   k = number of constraints tested
```
**Turning z/W into a judgment call:** look up how likely a Wald statistic at least this extreme would be, purely by chance, if the true coefficient were 0. If that probability (p-value) is below a conventional threshold (5%), the coefficient is "statistically significant."

### 9.6 Worked numeric example — hand-calculating SE and the Wald statistic

n=6, one feature x, fitted ŵ=1.2.
```
Fitted p̂ = [0.3, 0.4, 0.6, 0.7, 0.8, 0.5]
x         = [1, 2, 3, 4, 5, 6]

W diagonal p̂ᵢ(1−p̂ᵢ): 0.21, 0.24, 0.24, 0.21, 0.16, 0.25

XᵀWX for the slope (simplified): Σᵢ wᵢxᵢ²
 = 0.21×1 + 0.24×4 + 0.24×9 + 0.21×16 + 0.16×25 + 0.25×36
 = 0.21+0.96+2.16+3.36+4.00+9.00 = 19.69

Var(ŵ) ≈ 1/19.69 = 0.0508
SE(ŵ) = √0.0508 = 0.225

z = 1.2/0.225 = 5.33   (p-value ≈ 0.0001 → reject H₀)
W = 5.33² = 28.4  ~ χ²(1)
```

### 9.7 A second worked contrast (well-supported vs. shaky coefficient)

**A — well-supported:** `coefficient=0.50, SE=0.08` → `z = 6.25`, far beyond ±1.96 → highly confident real effect.
**B — shaky:** `coefficient=0.30, SE=0.25` → `z = 1.20`, below ±1.96 → cannot confidently claim a real effect, even though 0.30 doesn't *look* tiny.

**The key lesson:** raw coefficient SIZE alone (0.50 vs 0.30, not wildly different) is misleading — it's the coefficient **relative to its own uncertainty** that tells you whether to trust it.

> **Check:** Two coefficients both equal 0.40. Coefficient A has SE=0.05, B has SE=0.60. Which do you trust more, and what are the Wald statistics? *(A: z=0.40/0.05=8.0. B: z=0.40/0.60=0.67. Trust A — same raw coefficient, but A's tiny SE means the estimate is precise; B's huge SE means the same-looking 0.40 could easily be noise.)*

> **Check:** A colleague says "my coefficient has a tiny p-value (0.001), so this feature must have a huge real-world effect." What's wrong? *(A tiny p-value means the effect is probably NOT zero (statistically significant) — not that it's LARGE. With enough data, even a trivially small real effect becomes statistically significant. Significance = confidence the effect exists; effect size (the coefficient/odds ratio) = how big it is. You need both.)*

### 9.8 The Wald test's known weakness — the Hauck-Donner phenomenon

Wald performs poorly when ŵⱼ is large. Why? SE(ŵⱼ) is estimated **at ŵⱼ**, not at 0. Under perfect or near-perfect separation (Part D §21), `ŵⱼ → ∞` and `SE(ŵⱼ) → ∞` **even faster**, so the Wald statistic `W = ŵⱼ²/Var(ŵⱼ) → 0` — the test says "not significant" *precisely when the coefficient is most extreme.* This is a real, named failure mode (the Hauck-Donner phenomenon), not a minor footnote — it's the direct statistical mechanism behind why huge coefficients paired with huge standard errors is a red flag for a broken fit (separation), not a strong signal of importance. **When Wald fails this way, use the LRT instead (§10) — it does not suffer from this problem.**

### 9.9 FAANG L5 angle

**Q: "How do you know if a coefficient is meaningful or just noise?"** — Compute SE and the Wald statistic (coefficient/SE); |z|>~1.96 (5% threshold) suggests non-zero. This is what software reports by default as each coefficient's p-value.

**Q: "What causes a large SE on a coefficient?"** — Small sample, high outcome variance, or — critically — **multicollinearity**: correlated features make the model unable to isolate each one's individual effect, inflating SEs even if the features ARE genuinely predictive as a group.

**Q: "When would you prefer Wald vs. LRT?"** — Wald: quick, cheap, per-coefficient, reading straight off software output. LRT: testing a GROUP of features together, or when the Wald test's known small-sample/large-coefficient unreliability is a concern.

**Common trap:** conflating "statistically significant" with "practically important" — always mention both when discussing a coefficient.

---

## 10. The Likelihood Ratio Test — Feature Significance, and as a Goodness-of-Fit Test

### 10.1 WHY (feature-significance framing)

Loss/likelihood ALWAYS improves at least slightly when you add more features — even random, useless ones — purely because more parameters give more flexibility to fit training quirks. Without a rigorous test you can't distinguish "genuinely useful feature" from "noise that happened to help a bit by chance."

### 10.2 Intuition — the courtroom analogy

A "simple story" (H₀: this feature doesn't matter, the smaller model is just as good) vs. a "complex story" (H₁: the feature genuinely helps). The LR test asks: how much MORE plausible does the data become under the complex story? Is that improvement big enough to be convincing, or the kind of small improvement you'd expect from chance alone?

### 10.3 The statistic

Recall log-loss is negative log-likelihood, so lower log-loss = higher likelihood.
```
G² = -2 × (ℓ_simple − ℓ_complex) = 2 × (ℓ_complex − ℓ_simple)
Under H₀: G² ~ χ²(k),  k = number of EXTRA parameters in the complex model
```
**Why -2 specifically?** Not arbitrary — multiplying by -2 makes the statistic follow the well-studied chi-squared distribution, for which lookup tables already exist telling you exactly how "surprising" a given value is.

**Models must be nested:** the simple model's features must be a strict SUBSET of the complex model's features (e.g. `{tenure, complaints} ⊂ {tenure, complaints, tickets}`). You cannot validly LR-test two models with completely different, non-overlapping feature sets.

### 10.4 Worked numeric example

Testing whether "support tickets filed" meaningfully improves a churn model that already has tenure+complaints.
```
Simple model (tenure+complaints):            ℓ_simple  = -150.2
Complex model (+support tickets):            ℓ_complex = -145.8

G² = -2×(-150.2 − (-145.8)) = -2×(-4.4) = 8.8
df = 1 (one new feature)
χ²(1) threshold at 5%: 3.84

8.8 > 3.84 → adding "support tickets" is a statistically significant improvement.
```
**Contrast — a useless feature:** if "favorite color" nudged ℓ from -150.2 to -150.0:
```
G² = -2×(-150.2 − (-150.0)) = -2×(-0.2) = 0.4  →  well below 3.84  →  not significant.
```

### 10.5 Interpretation

Especially useful for deciding whether to keep a GROUP of related features together (e.g. "should we keep all 4 geographic features, or drop them as a group?") — test the whole group's combined contribution at once, with df = number of features in the group.

### 10.6 FAANG L5 angle

**Q: "How would you decide whether a new feature is worth adding?"** — Mention the LR test explicitly (compare log-likelihoods of nested models, -2× the difference, check against χ² with df=#new params), and contrast with practical/business-driven validation-metric checks (Part D §16) — LR test is a formal statistical-significance check; validation performance is "does it help in the way we actually care about." Mention both.

**Q: "What does 'nested models' mean, and why does it matter?"** — Simpler model's features are a strict subset of the complex model's. The test specifically requires this structure.

**Q: "Is a statistically significant feature always worth keeping in production?"** — Not necessarily — significance tells you the improvement probably isn't chance, not that it's PRACTICALLY meaningful (a large dataset can make even a tiny, business-irrelevant effect "significant").

**Common trap:** confusing this with "did validation accuracy go up" — the LR test is a distinct, formal hypothesis test with its own machinery (χ², degrees of freedom).

```python
import numpy as np
from scipy.stats import chi2
ll_simple, ll_complex, extra_params = -150.2, -145.8, 1
lr_stat = -2 * (ll_simple - ll_complex)
p_value = 1 - chi2.cdf(lr_stat, df=extra_params)
```

> **Check:** Simple model ℓ=-200.0, complex model with 2 extra features ℓ=-197.5. Compute G² by hand. *(G² = -2×(-200.0 − (-197.5)) = -2×(-2.5) = 5.0, df=2. Compare to χ²(2) threshold ≈5.99 at 5% — here 5.0 < 5.99, so NOT significant at the 5% level.)*

> **Check:** Why can't you LR-test `{age, income}` against a completely different `{location, device_type}`? *(They aren't nested — neither feature set is a strict subset of the other, so there's no valid "simple vs. complex" relationship for the test's chi-squared machinery to apply to.)*

### 10.7 The SECOND use of the LR test — Goodness-of-Fit vs. the Saturated Model

**WHY this is a distinct use:** beyond comparing two of YOUR OWN nested models, the same LR-test machinery can ask a different question: **does my model fit the data well AT ALL**, compared to a hypothetical "perfect" model?

**Intuition — the saturated model:** imagine a model so flexible it perfectly predicts every training row (essentially one parameter per row — pure memorization). Not something you'd ever deploy — a theoretical best-case reference point for "the absolute best any model could conceivably do on this exact data."

**Deviance:**
```
Deviance = -2 × (ℓ_your_model − ℓ_saturated_model)
```
Smaller deviance = closer to the theoretical best fit. Like the earlier LR statistic, deviance follows a χ² distribution, letting you test "is my model's fit significantly worse than the best possible fit, more than noise alone would explain?" A LOW deviance (relative to the χ² threshold, roughly df = n − #parameters) means you fail to reject "my model fits reasonably well." A HIGH deviance suggests systematic lack of fit — perhaps a missing non-linear term (§7) or interaction.

**FAANG angle:** *"How would you assess whether your model fits the data well overall, not just whether one feature helps?"* — Mention deviance/GoF testing against the saturated model, alongside more practical day-to-day checks (calibration curves, Part C §14; validation metrics, Part D §16) — in practice most working DS lean more on those, but knowing the formal deviance-based test signals depth.

**Common trap:** confusing THIS use (GoF vs. a hypothetical saturated model) with the earlier use (comparing two of your own nested models to test one feature) — same math machinery, genuinely different purpose.

*(Full deviance/Hosmer-Lemeshow/Pseudo-R² treatment continues in §13.)*

---

## 11. The Score (Rao) Test, and the Wald / LRT / Score Asymptotic Trinity

### 11.1 The idea

Only fit the REDUCED (null) model. Check whether the gradient of the log-likelihood at the restricted estimate is "close enough to zero." If H₀ is true, the gradient at the restricted MLE should be near zero.
```
S = [∂ℓ/∂w]_{H₀}ᵀ · I(ŵ_{H₀})⁻¹ · [∂ℓ/∂w]_{H₀}
Under H₀: S ~ χ²(k)
```

### 11.2 When to use it

Only one model needs fitting (the null) — useful when the full model is very expensive to fit, when you want to screen many candidate features quickly without fitting the full model for each, or in econometrics (Lagrange multiplier test). Less common in practice for logistic regression than Wald/LRT.

### 11.3 The trinity — asymptotic equivalence

Under H₀, as n → ∞: **Wald = LRT = Score.** In finite samples they differ — LRT is generally most accurate, Wald most convenient, Score cheapest to compute.

### 11.4 Wald vs. LRT — the decisive comparison table

| | Wald | LRT |
|---|---|---|
| What it computes | Ratio of estimate to SE | Ratio of likelihoods |
| Models fit | One (full) | Two (full + reduced) |
| Computational cost | Cheaper | ~2× the fitting |
| Accuracy in small samples | Worse | Better |
| Fails with separation | Yes (Hauck-Donner, §9.8) | No |
| For large n | Equivalent to LRT | Equivalent to Wald |
| Preferred for | Quick inference, large n | Small n, large coefficients, separation |

**Rule of thumb: use LRT when in doubt.** Wald is a convenient approximation to LRT that's less reliable in edge cases.

### 11.5 Testing the whole model — the Omnibus test

Tests H₀: all coefficients (except intercept) = 0.
```
ℓ_null = n1·log(ȳ) + n0·log(1−ȳ)     [intercept-only model]
G²_model = 2(ℓ_full − ℓ_null)  ~  χ²(p),   p = number of predictors
```
This is the logistic-regression equivalent of the F-test in linear regression — if this isn't significant, no individual predictor is worth examining.

```python
import statsmodels.api as sm
model = sm.Logit(y, X).fit()
print(model.summary())   # coef, std err, z (Wald), P>|z|, 95% CI
# model.llr_pvalue        # omnibus LRT p-value
```

---

## 12. Confidence Intervals on Coefficients & Odds Ratios

### 12.1 WHY

The Wald test told you WHETHER a coefficient is significantly different from zero. A CI tells you a full RANGE of plausible true values — richer than a yes/no verdict.

### 12.2 Intuition

Instead of "is this probably not zero" (yes/no), a CI answers: "if I re-ran this data collection many times, what range of values would the estimated coefficient typically fall into?" Narrow CI = precise. Wide CI = a lot of remaining uncertainty, even with a confident-looking point estimate.

### 12.3 Formula — the (Wald-based) CI

```
CI = coefficient ± (1.96 × standard_error)      [for a 95% CI]
```

### 12.4 Worked numeric example

Reusing `coefficient=0.50, SE=0.08`:
```
lower = 0.50 − 1.96×0.08 = 0.343
upper = 0.50 + 1.96×0.08 = 0.657
95% CI for the coefficient: [0.343, 0.657]

Odds-ratio CI (exponentiate each endpoint):
lower OR = e^0.343 = 1.41,  upper OR = e^0.657 = 1.93
```
"We're 95% confident the true odds ratio lies between about 1.41× and 1.93×" — more informative than "the odds ratio is 1.65 and it's significant."

### 12.5 A more accurate alternative — the Profile-Likelihood CI

Wald CIs are symmetric by construction (`coef ± const`) and inherit the Hauck-Donner weakness for extreme coefficients. The **profile-likelihood CI**, based on the LRT rather than the Wald statistic, is asymmetric and more reliable for small samples or large coefficients:
```
{w : 2(ℓ_full − ℓ(w)) ≤ χ²_α(1)}
```
Statsmodels computes this via `.conf_int(method='profile-likelihood')`.

### 12.6 FAANG L5 angle — the classic misinterpretation trap

"There's a 95% probability the true value is in this interval" is technically WRONG frequentist phrasing. Correct: "if we repeated this sampling process many times, 95% of such intervals would contain the true value." A subtle distinction some interviewers specifically probe.

> **Check:** if your calibration curve showed predicted 20% but actual rate 35% (model *under*-confident), would Platt scaling still apply? What would the correction curve look like directionally? *(Yes, Platt scaling applies either direction — it's just fitting σ(a·s+b) to realign scores with reality. Here the correction would need to STRETCH predictions upward (increase them) rather than compress them, since the raw scores are systematically too low relative to actual outcomes — directionally the opposite adjustment from the overconfident case in §14.)*

---

## 13. Goodness of Fit: Deviance, Hosmer-Lemeshow, Pearson χ², Pseudo-R², AIC/BIC

### 13.1 Deviance

The fundamental GLM fit measure:
```
D = -2 · ℓ(fitted model)
Null deviance:     D_null = -2·ℓ(null model)     [intercept only]
Residual deviance: D_res  = -2·ℓ(fitted model)
Deviance difference = G² from the LRT: G² = D_null - D_res ~ χ²(p)
```
**Saturated model:** one parameter per observation, fits perfectly. Deviance relative to it measures how much information the fitted model loses (§10.7).

### 13.2 Hosmer-Lemeshow Test — the most common GoF test for logistic regression

**Procedure:** sort all observations by fitted p̂; split into g groups (usually g=10 deciles); in each group compare observed vs. expected event counts.
```
Eₖ = Σᵢ∈k p̂ᵢ    (expected events in group k)
Oₖ = Σᵢ∈k yᵢ    (observed events in group k)

χ²_HL = Σₖ [(Oₖ−Eₖ)² / (nₖ·p̄ₖ·(1−p̄ₖ))]  ~  χ²(g−2)
```
H₀: model is well-calibrated. Reject H₀ → poorly calibrated.

**Hand calculation** (10 obs, 2 groups of 5 for illustration — real practice uses g≥5, ideally g=10):
```
Group 1 (lowest 5, p̂≈0.05-0.25): E₁=0.75, O₁=1, p̄₁=0.15
Group 2 (highest 5, p̂≈0.55-0.90): E₂=3.60, O₂=4, p̄₂=0.72

χ²_HL = (1-0.75)²/(5×0.15×0.85) + (4-3.60)²/(5×0.72×0.28)
      = 0.0625/0.6375 + 0.16/1.008 = 0.098 + 0.159 = 0.257
```
With g=10 (standard), df=8; if χ²_HL < χ²_{0.05}(8)=15.51, fail to reject → good fit.

### 13.3 Pearson Chi-squared GoF

```
χ²_P = Σᵢ (yᵢ−p̂ᵢ)² / (p̂ᵢ(1−p̂ᵢ))   ~  χ²(n−p−1)
```
Works well for grouped data (multiple obs per covariate pattern). For individual binary outcomes with continuous predictors this test has poor power — prefer Hosmer-Lemeshow.

### 13.4 Pseudo-R² measures

No single R² exists for logistic regression; several approximations:

**McFadden's:** `R²_McF = 1 − ℓ_full/ℓ_null` — range [0,1); values of 0.2–0.4 indicate *excellent* fit (much lower bar than linear R²!).

**Cox-Snell:** `R²_CS = 1 − exp(2(ℓ_null−ℓ_full)/n)` — max value is annoyingly always <1.

**Nagelkerke:** `R²_N = R²_CS / R²_CS,max`, where `R²_CS,max = 1 − exp(2ℓ_null/n)` — rescales Cox-Snell to have a true max of 1.

**Hand calculation:**
```
ℓ_null=-45.2, ℓ_full=-28.7, n=100

McFadden:  1 − (-28.7)/(-45.2) = 1 − 0.635 = 0.365
Cox-Snell: 1 − exp(2(-45.2-(-28.7))/100) = 1 − exp(-0.33) = 1 − 0.719 = 0.281
R²_CS,max: 1 − exp(2(-45.2)/100) = 1 − exp(-0.904) = 1 − 0.405 = 0.595
Nagelkerke: 0.281/0.595 = 0.472
```

### 13.5 AIC and BIC — penalized likelihood for (non-nested) model comparison

Not hypothesis tests — used to compare models that AREN'T necessarily nested:
```
AIC = -2ℓ + 2k             (penalizes complexity lightly)
BIC = -2ℓ + k·log(n)       (penalizes complexity more heavily for large n)
```
Lower is better; BIC favors simpler models more strongly than AIC, especially as n grows.
```
ℓ=-18.42, k=3, n=200
AIC = 36.84 + 6 = 42.84
BIC = 36.84 + 3×log(200) = 36.84 + 15.9 = 52.74
```

### 13.6 Master quick-reference for Part B

```
Test                Formula                       Dist under H₀    Use when
─────────────────────────────────────────────────────────────────────────────
Wald               W = (ŵ/SE)²                   χ²(1)            Quick, large n
LRT                G² = 2(ℓ_full − ℓ_red)        χ²(df)           Small n, separation, group tests
Score              S = gradient at H₀             χ²(k)            Only fit null model
Omnibus            G² = 2(ℓ_full − ℓ_null)       χ²(p)            Test whole model
Hosmer-Lemeshow    Σ(O−E)²/(n·p̄·(1−p̄))          χ²(g−2)          Calibration/GoF

Pseudo-R²          McFadden = 1 − ℓ_full/ℓ_null
                   Nagelkerke = rescaled Cox-Snell to [0,1]
```

---
---

# PART C — CALIBRATION

## 14. Calibration — Diagnosis and Fixes (Full Treatment)

### 14.1 What calibration means

A model is calibrated if predicted probabilities match observed frequencies: among all rows where the model predicts p̂=0.7, roughly 70% should actually be positive. A model can have high AUC (good ranking/discrimination) but be poorly calibrated — it ranks correctly, but the actual probability *values* are wrong. For decisions that use the raw probability (thresholds, expected-value/cost calculations), calibration matters as much as discrimination.

### 14.2 Diagnosis Method 1 — the calibration curve (reliability diagram)

Sort predictions by p̂; bin into B buckets (equal-width or equal-frequency); in each bin compute mean(p̂) and mean(y); plot mean(p̂) (x) vs. mean(y) (y). Perfect calibration = the 45° diagonal.

**Patterns to recognize:**
```
S-shaped curve:        overconfident (predictions too extreme, too close to 0/1)
Inverse S-shape:       underconfident (predictions clustered too close to 0.5)
Curve above diagonal:  underpredicting (predicted p̂ too low vs actual rate)
Curve below diagonal:  overpredicting (predicted p̂ too high vs actual rate)
```

**Worked numeric example (overconfidence):**

| Predicted bucket | Customers | Actual churners | Actual rate |
|---|---|---|---|
| 60-70% | 100 | 45 | 45% |
| 70-80% | 100 | 55 | 55% |
| 80-90% | 100 | 60 | 60% |

The model predicts 60-70% but reality is only ~45-60% — systematically **overconfident** in this range.

### 14.3 Diagnosis Method 2 — Hosmer-Lemeshow Test

This test (fully derived in §13.2) IS itself a calibration test — same procedure, same statistic.

### 14.4 Diagnosis Method 3 — Expected Calibration Error (ECE)

```
ECE = Σₖ (nₖ/n) · |mean(p̂)ₖ − mean(y)ₖ|      range [0,1], closer to 0 = better
```

**Hand calculation — well-calibrated model (8 obs):**
```
p̂: 0.1 0.2 0.3 0.4 | 0.6 0.7 0.8 0.9
y:  0   0   1   0  |  1   1   0   1

Bin1 (p̂<0.5): mean(p̂)=0.25, mean(y)=0.25, |diff|=0.00
Bin2 (p̂≥0.5): mean(p̂)=0.75, mean(y)=0.75, |diff|=0.00
ECE = 0.00  (well calibrated)
```

**Hand calculation — overconfident model, same labels, different p̂:**
```
p̂: 0.6 0.7 0.8 0.9 0.9 0.9 0.9 0.9
y:  0   0   1   0   1   1   0   1

Bin1 (p̂<0.85): p̂=[0.6,0.7,0.8], y=[0,0,1] → mean(p̂)=0.70, mean(y)=0.33, |diff|=0.37
Bin2 (p̂≥0.85): p̂=[0.9×5],       y=[0,1,1,0,1] → mean(p̂)=0.90, mean(y)=0.60, |diff|=0.30

ECE = (3/8)×0.37 + (5/8)×0.30 = 0.139+0.188 = 0.327   (poor calibration)
```
This model always predicts high but only 62.5% of these rows are actually positive.

### 14.5 Fix Method 1 — Platt Scaling

Fit a SECOND, small logistic regression using the ORIGINAL model's output (score/logit) as the only input feature, predicting the true label:
```
p_calibrated = σ(a·s + b),   a,b learned by MLE on (sᵢ, yᵢ) pairs, on a HELD-OUT calibration set
```
`a=1,b=0` → no correction needed. `a<1` → the model was overconfident (Platt compresses scores toward 0.5). `b≠0` → systematic bias. Works well when miscalibration follows a roughly sigmoid-shaped distortion.

**Hand calculation:**
```
Raw logits s = [-2,-1,0,1,2],  y = [0,0,1,1,1]
Initial a=1,b=0 → p̂=σ(s) = [0.12,0.27,0.50,0.73,0.88]
Observed: s<0 → 0/2 positive (predicted ~0.2, too high); s≥0 → 3/3 positive (predicted ~0.7, too low)
After fitting a=0.8, b=-0.3: p̂=σ(0.8s−0.3) → better aligned with observations
```

### 14.6 Fix Method 2 — Isotonic Regression

Non-parametric: fits a flexible, monotone step-function correction (no assumed shape, just enforces higher raw score → higher-or-equal corrected score). More flexible than Platt scaling but needs more data to fit reliably without overfitting the correction itself.

### 14.7 Fix Method 3 — Temperature Scaling (neural nets / deep learning)

```
p_calibrated = softmax(z/T)
T>1: softens (reduces overconfidence)   T<1: sharpens   T=1: no change
```
T is found by minimizing NLL on a held-out validation set.

```python
from sklearn.calibration import CalibratedClassifierCV
cal_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)    # Platt
cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)   # Isotonic
cal_model.fit(X_train, y_train)
p_calibrated = cal_model.predict_proba(X_test)[:, 1]
```

### 14.8 Calibration vs. Discrimination — orthogonal properties

| | Calibration | Discrimination (AUC) |
|---|---|---|
| Measures | Are probability *values* accurate? | Are observations correctly *ranked*? |
| Perfect score | ECE=0, HL non-significant | AUC=1.0 |
| One without the other? | Yes — high AUC + poor calibration is common | Yes — perfect calibration + AUC=0.5 possible |
| Fixing one fixes the other? | No | No |
| Matters most when | Probability used directly (pricing, decisions) | Binary classification, ranking |

A model can rank patients perfectly by risk (AUC=1) but assign 0.9 to everyone (uncalibrated). A model can be perfectly calibrated (every 0.7-bucket has a 70% outcome rate) yet fail to discriminate (AUC=0.5). Genuinely independent axes of quality.

### 14.9 FAANG L5 angle

**Q: "Your model ranks well (AUC=0.85) but raw probabilities are unreliable — fix it without retraining."** — Platt scaling or isotonic regression, fit on a HELD-OUT calibration set (never the same data the original model trained on, or you won't reveal genuine miscalibration — the model has already adapted to that exact data).

**Common trap:** recalibrating using the same data the original model trained on.

---
---

# PART D — MODEL BUILDING PRACTICALITIES

## 15. Feature Importance

### 15.1 WHY

Once trained, a natural question: "which features actually matter most?" The most common mistake: comparing RAW coefficients directly across features on wildly different scales (`w_income=0.00003` vs `w_complaints=0.8` does NOT mean complaints matters 27,000× more) — a small-looking coefficient on a huge-scale feature can represent a bigger real-world effect than a large-looking coefficient on a tiny-scale feature.

### 15.2 Intuition

Employee A gets $0.10/email, Employee B gets $50/sale. Comparing "$0.10 vs $50" and concluding selling is 500× more valuable ignores that people send hundreds of emails but close very few sales — the SCALE of typical activity matters as much as the per-unit reward. A feature's importance depends on BOTH its coefficient AND how much that feature actually varies in real data.

### 15.3 Approach 1 — Standardized Coefficients

Rescale every feature to mean 0, std 1 BEFORE training; then all coefficients represent "effect per 1 standard deviation of typical variation" — a fair, apples-to-apples unit regardless of original scale.
```
standardized_x = (x − mean(x)) / std(x)
```
After training on standardized features, the coefficient with the largest |value| truly represents the biggest effect per typical unit of variation.

**Worked example:** raw `w_income=0.00002, w_complaints=0.80` (looks like complaints is 40,000× more important). Income: mean=$50k, std=$20k. Complaints: mean=1.2, std=1.5. After retraining on standardized features: `w_income=0.35, w_complaints=0.55` — complaints IS still more influential, but by roughly 1.5–2×, not 40,000×.

### 15.4 Approach 2 — Permutation Importance (model-agnostic, general)

**WHY it exists:** standardized coefficients are linear-model-specific. Permutation importance works for ANY model (trees, neural nets), giving one reusable method everywhere.

Take a trained model and a validation set; record baseline performance; for one feature at a time, randomly shuffle just THAT feature's values (destroying its real relationship with the outcome while keeping everything else intact); re-measure performance. The DROP tells you reliance on that feature.
```
importance(feature) = performance_before_shuffling − performance_after_shuffling
```

**Worked example:** baseline validation accuracy 85%.

| Feature | Accuracy after shuffling | Drop | Ranking |
|---|---|---|---|
| Complaints | 71% | 14pts | 1st |
| Tenure | 79% | 6pts | 2nd |
| Favorite color (junk) | 85% | 0pts | 3rd (irrelevant) |

```python
def permutation_importance(model_predict_fn, X, y, feature_idx, baseline_score, score_fn):
    X_shuffled = X.copy()
    np.random.shuffle(X_shuffled[:, feature_idx])
    shuffled_score = score_fn(y, model_predict_fn(X_shuffled))
    return baseline_score - shuffled_score
```

### 15.5 FAANG L5 angle

**Q: "How would you determine feature importance in a logistic regression model?"** — Mention BOTH raw-coefficient pitfalls (scale-dependence) and a fix: standardized coefficients (simple, LR-specific) or permutation importance (general, model-agnostic, more expensive since it re-scores repeatedly).

**Q: "Why can't you just compare raw coefficients?"** — Coefficient magnitude depends on feature scale — a dollar-scale feature naturally gets a small coefficient, a count-scale feature a larger one, even with similar real-world importance. Without standardizing, you're comparing apples to oranges.

**Q: "Is permutation importance affected by multicollinearity?"** — Yes — if two features are highly correlated, shuffling just ONE barely hurts performance (the model leans on the other correlated feature to compensate), making BOTH look artificially unimportant individually, even though together they're genuinely predictive.

**Common trap:** reciting "look at coefficients" without mentioning standardization.

**Another trap:** confusing feature importance (HOW MUCH a feature matters for predictions) with statistical significance/p-values (how confident you are the coefficient isn't zero). A feature can be significant but practically unimportant (tiny effect, huge dataset), or vice versa.

> **Check:** `w_age=1.2` (raw years), `w_income=0.00001` (raw dollars). Without standardizing, can you conclude age matters ~120,000× more? *(No — the two features are on wildly different scales; you must standardize both before any coefficient-magnitude comparison is meaningful.)*

> **Check:** if two features are highly correlated, what artifact appears in permutation importance? *(Both may show artificially low importance individually, since shuffling one leaves the model able to lean on the other correlated feature — masking their true combined predictive value.)*

---

## 16. Model Selection

### 16.1 WHY

You'll train many candidates (feature sets, regularization strengths, algorithms) before shipping one. Model selection is choosing which to ship, honestly. The most common mistake: evaluating on the SAME data used for training — a model can achieve near-perfect training performance by memorizing it (overfitting) while performing terribly on new data.

### 16.2 Intuition

Like studying for an exam using a practice test, then grading yourself on that SAME practice test — you'll look like a genius from memorization. The real test of learning is performance on questions never seen before. And if you keep peeking at your "unseen" data to pick a model, it stops being unseen — you need a THIRD, completely locked set for the final honest verdict.

### 16.3 The three-way split

```
Training set   (~60-70%) — fits model weights via gradient descent
Validation set (~15-20%) — compares candidate models/hyperparameters, picks the winner
Test set       (~15-20%) — used ONCE, at the very end, for the final honest performance number
```
Validation data influences your choice even though the model never directly trains on it. Peeking at test repeatedly and adjusting based on it silently turns it into a second validation set — invalidating its purpose (a subtle, real form of leakage — see §17).

### 16.4 Worked example — choosing λ via validation

| λ | Training log-loss | Validation log-loss |
|---|---|---|
| 0.001 (weak) | 0.22 | 0.51 |
| 0.1 (moderate) | 0.31 | 0.34 |
| 10 (strong) | 0.58 | 0.60 |

λ=0.001: big train/val gap → classic **overfitting**. λ=10: both high AND close → **underfitting**. λ=0.1: both low AND close → the sweet spot. **Pick λ=0.1**, then evaluate on test *once* (say, log-loss=0.35) — that's your final honest number.

### 16.5 Cross-Validation

**WHY:** a single validation split has a weakness — depending on which random rows land in it, you might get a lucky/unlucky read on true performance, especially with small datasets.

Divide training data into K folds; train K times, each time validating on a different fold; average the K scores.

**Worked mini-example (K=3, 300 rows / 100 per fold):**
```
Run 1: train B+C, validate A → 0.33
Run 2: train A+C, validate B → 0.29
Run 3: train A+B, validate C → 0.37
average = (0.33+0.29+0.37)/3 = 0.33
```
More trustworthy than any single split.

### 16.6 FAANG L5 angle

**Q: "Why do you need both a validation set and a test set?"** — Using one held-out set to BOTH pick the best model AND report final performance makes the reported number optimistically biased (you specifically chose the model that did well on that exact set — a subtle overfitting to validation itself). Test stays untouched for an honest final estimate.

**Q: "When would you use cross-validation instead of a simple split?"** — Especially with smaller datasets, where a single split might not be representative and you can't afford to "waste" a large chunk on validation. More computationally expensive (train K times) — with very large datasets a simple split is often sufficient and much cheaper.

**Common trap:** picking the model with lowest TRAINING loss — always wrong, since training loss can always be pushed lower via overfitting.

**Another trap:** "peeking" at test performance multiple times and adjusting — even "just once more" silently converts test into a second validation set.

> **Check:** Model A: train 0.15, val 0.45. Model B: train 0.30, val 0.32. Which would you pick, and why? *(Model B — the gap between train and val is small (0.30 vs 0.32), signaling good generalization; Model A's large gap (0.15 vs 0.45) is the classic overfitting signature, and its validation performance is worse anyway.)*

> **Check:** Why is cross-validation generally more expensive than a single split, and when would that cost NOT be worth paying? *(It requires training the model K separate times instead of once. Not worth paying with very large datasets, where a single split already gives a stable, representative estimate and the extra compute buys little additional reliability.)*

---

## 17. Feature Leakage — Deep Dive

### 17.1 WHY

Leakage is when information "sneaks" into training that wouldn't actually be available at prediction time — arguably the single most damaging bug in an ML pipeline because it's **invisible in your metrics.** Validation looks great (since the same leaked information contaminates validation too, coming from the same flawed pipeline) — then production, computed the "honest" way, doesn't have the shortcut, and performance collapses with no obvious error message.

### 17.2 Intuition

Studying with the answer key accidentally mixed into your practice questions — you ace every practice test by memorizing the shortcut, then walk into the real exam where the shortcut doesn't exist.

### 17.3 Type 1 — Target Leakage

A feature that's only known, or only meaningful, BECAUSE the outcome already happened. **Classic example:** "number of retention calls made" as a churn-prediction feature — retention calls only happen to customers already flagged as likely churners, so the feature just restates "someone already flagged this person," not a genuine predictor. Amazing validation metrics, useless in production (brand-new customers being scored have had no retention call yet — the feature is 0/missing for everyone who matters).

**How to catch it:** for every feature, ask "could this value have been computed at the EXACT moment I need to make a real prediction, using only information that existed then?" If no — because it depends on the outcome having already occurred — it's target leakage.

### 17.4 Type 2 — Temporal Leakage

Using data that chronologically comes from AFTER the moment you're predicting. **Classic example:** "total chargebacks on this account" computed using ALL historical data (past AND future) for a fraud model — at prediction time you can't know about chargebacks that haven't happened yet.

**How to catch it:** a random row shuffle-split can accidentally mix future rows into training and past rows into testing. The fix is often a **time-based split** — train on everything before a date, test on everything after, mimicking real deployment (always predicting forward, never backward).

### 17.5 Type 3 — Group/Entity Leakage

The same real-world entity appears in BOTH train and test (as different rows) — the model isn't really tested on truly unseen data. **Classic example:** 1,000 customers × 20 transactions each = 20,000 rows; a naive random 80/20 ROW split puts some of a customer's transactions in train and others in test, letting the model "recognize" that customer's quirks, inflating test performance in a way that won't hold for a genuinely new customer.

**How to catch it:** ask "what's the real-world UNIT I'm predicting for?" (usually a customer, not a transaction) — split by THAT unit.

**Worked example:**
```
WRONG (row-level split): Training: A(4),B(4),C(4),D(4). Test: A(1),B(1),C(1),D(1).
  → every test customer was ALSO in training — model may have learned customer-specific quirks.
CORRECT (customer-level split): Training: Customer A,B,C (15 rows). Test: Customer D (5 rows, never seen).
  → a fair test of "how will this model perform on a brand-new customer."
```

### 17.6 Type 4 — Preprocessing Leakage

Computing statistics (mean, std, encoding mappings) using the FULL dataset (train+val+test) before splitting. **Classic example:** standardizing `(x−mean)/std` using the whole dataset — test-set values technically contributed to that mean/std. Subtle, often a smaller-magnitude issue than Types 1-3, but a common, easy-to-miss "gotcha."

**Prevention:** always **split FIRST**, compute any statistics using ONLY the training set, then apply those training-derived numbers to transform validation and test.
```python
X_train, X_test = train_test_split(np.arange(100), test_size=0.2, random_state=42)
train_mean, train_std = X_train.mean(), X_train.std()
X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled  = (X_test  - train_mean) / train_std     # reuse TRAINING stats
```

### 17.7 FAANG L5 angle

**Q: "Your model has 98% validation AUC but performs poorly in production — what do you investigate?"** — Leakage should be one of your FIRST hypotheses (alongside training-serving skew, §29): target leakage? Temporal order respected in the split? Entities cleanly split, not spread across train/test? Any statistics computed on the full dataset before splitting?

**Q: "How would you split data for a fraud model with millions of transactions across thousands of customers over 2 years?"** — Needs BOTH temporal AND group thinking: split by time (train earlier period, test later, mimicking deployment); customer-specific aggregate features must use only data available up to each prediction's point in time, not the full history including future transactions. (Returning customers appearing in both periods is fine here — you'd expect that in production; the leakage risk is in the *feature computation*, not the customer overlap itself.)

**Common trap:** only mentioning "don't use future data" and forgetting entity/group leakage.

**Another trap:** treating preprocessing leakage as equally severe as target/group leakage — it's real, but typically much smaller-magnitude than leakage that can single-handedly invalidate an entire model's results.

> **Check:** predicting hospital readmission risk, feature = "length of the discharge summary note" (longer notes correlate with readmission). Leakage red flag? *(Potentially yes — ask whether longer notes are being WRITTEN because clinicians already suspect a complicated/high-risk case, meaning the note length partly reflects a judgment made *after* assessing risk, not an independent predictor available equally early for every patient. Investigate when in the patient's stay the note was finalized relative to when you'd need to make the prediction.)*

> **Check:** time-split by year (train 2023, test 2024), but "average monthly spend" is computed using EACH customer's ENTIRE history (2023+2024) even for 2023 training rows. What kind of leakage, and the fix? *(Temporal leakage — the 2023 training rows' feature values are contaminated with 2024 (future, from that row's perspective) information. Fix: compute each row's aggregate feature using only data available up to that row's own timestamp, not the customer's full history.)*

---

## 18. Missing Data Handling

### 18.1 WHY

The weighted-sum math (`z=b+w1x1+...`) has no built-in way to handle a missing x1. Libraries either crash, silently drop every row with any missing value (throwing away data if missingness is common), or treat missing as literally 0 — which might be wildly wrong ("0 complaints" vs. "we don't know" are very different things).

### 18.2 Intuition

Like a student skipping an exam question — you could give 0 (harsh, maybe wrong), estimate their likely answer from other answers (reasonable), or note "they skipped this" as its own signal. The right choice depends on WHY the data is missing.

### 18.3 Strategy 1 — Simple Imputation (mean/median/mode)

```
imputed_value = mean(non-missing values in this column)
```
Reasonable when missingness is roughly random. **Risk:** if 30% of a column is missing and you fill with the mean, you create an artificial spike at the mean, distorting the feature's true spread, and silently treats "missing" as "average" — possibly false.

### 18.4 Strategy 2 — Missingness Indicator ("was this missing" flag)

```
new_feature = 1 if original value was missing, else 0
original_feature = imputed value if missing, else the real value
```
**Key insight:** the fact that data is missing can itself be informative (e.g. customers who don't share income may be systematically different — more privacy-conscious, embarrassed about income level). The flag lets the model learn "does knowing-whether-missing help predict the outcome?" separately from "what was the likely value?"

### 18.5 Strategy 3 — Model-Based Imputation

Use OTHER features to predict the likely missing value (e.g. a small regression on age/tenure to predict likely income). Worth it when missingness is common and features are strongly related — at the cost of complexity and a real risk of "manufacturing" overly confident fake data if not done carefully.

### 18.6 Worked example — impute + flag

Income missing for 3/8 customers (45000, 62000, missing, 38000, missing, 71000, 55000, missing):
```
mean of non-missing = (45000+62000+38000+71000+55000)/5 = 54,200
```

| Customer | Income (imputed) | was_income_missing |
|---|---|---|
| 3 | 54,200 | 1 |
| 5 | 54,200 | 1 |
| 8 | 54,200 | 1 |
(others: real values, flag=0)

### 18.7 FAANG L5 angle

**Q: "How do you handle missing data in a logistic regression pipeline?"** — Strategy choice depends on WHY data is missing (MCAR/MAR/MNAR — Missing Completely At Random / At Random / Not At Random, the formal terms), and always consider a missingness-indicator flag alongside imputation, since missingness itself can be predictive.

**Common trap:** blindly dropping all rows with any missing value — can silently throw away a large, possibly non-random chunk of data, introducing bias.

> **Check:** "satisfaction survey score" missing for 40%, and you suspect about-to-churn customers are less likely to bother filling out the survey. Simple mean imputation alone, or something extra? *(Add a missingness-indicator flag — since you suspect the missingness itself is informative, a flat mean-fill alone hides that signal. The flag lets the model learn "not answering" as its own predictor, separate from the imputed value.)*

---

## 19. Outlier Handling

### 19.1 WHY

Gradient descent updates weights based on `(p−y)×x` — one row with a massive `x` (e.g. a typo like "age=950") can dominate the gradient, dragging weights around based on one broken point rather than genuine signal.

### 19.2 Intuition

One typo'd test score ("9500" instead of "95") massively distorts a class average, even though it says nothing real about the class's performance.

### 19.3 Detection — the IQR method

```
IQR = Q3 − Q1
lower_bound = Q1 − 1.5×IQR
upper_bound = Q3 + 1.5×IQR
```

### 19.4 Handling options

- **Capping/Winsorizing:** clip extreme values to the boundary — keeps the row, limits its influence.
- **Log/other transforms:** for right-skewed data (income, transaction amounts), compresses extremes without discarding them.
- **Removal:** only when confident the value is a genuine ERROR (age=950 is impossible), not just a legitimately unusual real value (a genuinely huge income isn't an error — cap/transform, don't delete real signal).

### 19.5 Worked example

Ages `[25,28,31,29,33,27,950,30,26]`: Q1≈27, Q3≈31, IQR=4, upper_bound=31+1.5×4=37. 950 is far beyond 37 — flagged, and since no human is 950, this is almost certainly a data-entry error → investigate and correct/remove.

### 19.6 FAANG L5 angle

**Q: "How would you detect and handle outliers before training?"** — IQR or z-score detection, then distinguish ERROR outliers (fix/remove) from LEGITIMATE extreme values (cap/transform — don't delete real signal).

**Common trap:** removing every statistical outlier automatically without checking if it's genuine — deleting real high-value customers (e.g. from a fraud or LTV model) because they're "outliers" biases the model against exactly the cases that matter most.

> **Check:** why might capping be preferable to deleting the row entirely? *(Capping preserves the row's OTHER feature values — deleting throws away all of a row's good information just because ONE feature was extreme. Capping also avoids shrinking dataset size, which matters more with smaller datasets.)*

---

## 20. Feature Scaling and Gradient Descent Convergence

### 20.1 WHY (distinct from the interpretability reason in §15)

Unscaled features cause slow/unstable gradient descent CONVERGENCE — a separate, real problem from coefficient interpretability. If one feature ranges 0-1 and another 0-100,000, the cost surface becomes extremely elongated/skewed. A single learning rate that suits the small-scale feature will be far too slow for the large-scale one (or vice versa), causing zig-zagging or many more iterations to converge.

### 20.2 Intuition

A foggy-hillside walker in a long narrow canyon (steep on the narrow sides, nearly flat along the long axis) bounces back and forth across the steep walls while making painfully slow progress toward the true bottom. Scaling reshapes the canyon back into something closer to a round bowl, where one learning rate makes even progress in every direction.

### 20.3 Formula (same standardization as §15.3, different justification)

```
standardized_x = (x − mean(x)) / std(x)     [applied to every numeric feature before training]
```

### 20.4 Worked example (conceptual)

`complaints` (0-10) and `income` (20,000-150,000): without scaling, a learning rate suited to `w_complaints` causes wildly oversized updates for `w_income` (15,000× larger scale) — likely overshooting/diverging. With scaling, both range roughly ±2 std devs — one learning rate behaves sensibly for both, converging much faster and more reliably.

### 20.5 FAANG L5 angle

**Q: "Why does feature scaling matter beyond coefficient interpretability?"** — Unscaled features create an elongated, skewed cost surface, making gradient descent slow/unstable with one shared learning rate; scaling reshapes it toward a round bowl. *(Bonus: this is LESS of an issue for tree-based models, which split on thresholds rather than using gradient-based weighted sums — a good contrast point.)*

**Common trap:** thinking scaling only matters for interpretation and forgetting the separate, real convergence-speed reason — interviewers specifically probe for this second reason.

> **Check:** True or false — feature scaling changes what a logistic regression model is *theoretically* capable of learning? *(False — scaling doesn't change what the model can represent; the same relationships can be captured either way since the model can always compensate with different-sized coefficients. It only affects HOW EASILY/QUICKLY gradient descent finds those coefficients, not the ceiling of what's learnable.)*

---

## 21. Separation / Quasi-Complete Separation

### 21.1 WHY

A real production bug, not a theory footnote. When a feature (or combination) perfectly, or almost perfectly, predicts the outcome in training data, it **breaks the training process entirely.** Gradient descent minimizes log-loss; if a feature perfectly separates the classes, the optimizer realizes it can keep shrinking log-loss FOREVER by pushing that coefficient toward infinity — an infinitely large coefficient makes predicted probability infinitely close to 0%/100% for those rows, exactly minimizing log-loss for a perfect predictor. **Weights never converge — they keep growing unboundedly**, and training crashes, times out, or silently returns nonsensical huge coefficients.

### 21.2 Intuition

A teacher grading purely on "did the student write the exact letter 'A' at the top?" — if every 'A'-writer got 100% and everyone else 0%, confidence in that rule would grow WITHOUT LIMIT the more students confirm it — no natural point to stop. The rule "explains" the data too perfectly, and the confidence (coefficient) has no natural ceiling.

### 21.3 Complete vs. Quasi-Complete Separation

**Complete:** a feature/combination perfectly divides the classes — zero overlap. **Quasi-complete:** almost perfect — a tiny bit of overlap, but still enough to cause a milder version of the same runaway-coefficient problem.

**Symptom you'll actually SEE:** absurdly large coefficients (e.g. w=847.3) and/or absurdly large standard errors on those same coefficients (connecting directly to §9.8's Hauck-Donner phenomenon — the Wald test technically breaks down here too, reporting "not significant" precisely because the coefficient AND its SE have both blown up), plus warnings like "Maximum Likelihood estimate does not exist" or convergence failures.

### 21.4 Worked example

Predicting loan default with credit score as the only feature:

| Credit Score | Defaulted? |
|---|---|
| 550,580,600 | Yes |
| 650,700,750 | No |

Every score below 620 defaulted, every score above didn't — a perfect dividing line (complete separation). Fitting logistic regression here, the optimizer keeps pushing the coefficient more extreme (-0.1 → -5 → -500...) because each larger magnitude makes it even MORE confident about a rule already perfectly true in this tiny sample, with no contradicting data point to push back.

### 21.5 Why this happens more often than you'd think

- **Small datasets:** with few rows, a feature can easily perfectly/near-perfectly separate classes by chance — common in small clinical trials, rare-event fraud data, small A/B tests.
- **Too many features relative to rows:** with enough features (especially one-hot categoricals with many rare categories), some combination becomes statistically likely to perfectly predict a handful of outcomes by chance.
- **A "too good to be true" feature:** sometimes a disguised leakage red flag (§17) — if a feature perfectly predicts the outcome, ask "is this secretly encoding the answer?" before assuming pure luck.

### 21.6 How to fix it

- **Regularization (most common fix):** L2 (or L1) explicitly penalizes large coefficients, directly counteracting the push toward infinity, forcing convergence to a large-but-finite answer.
- **Remove or investigate the separating feature:** consider whether it's leakage rather than a genuine, generalizable pattern.
- **Collect more data:** "perfect" separation is often an artifact of too-small a sample; more data often reveals overlap that breaks the artificial perfection.
- **Firth's correction:** a specialized bias-reduced technique designed for rare events/separation — worth name-dropping in interviews even without deriving it (full treatment in Part F §27).

### 21.7 FAANG L5 angle

**Q: "Your training produces coefficient=500, SE=300. What's going on?"** — Classic sign of separation — some feature is perfectly/near-perfectly predicting the outcome, pushing the coefficient toward infinity without converging. Investigate whether genuine (small-sample artifact) or leakage; apply regularization as an immediate fix.

**Q: "How does regularization specifically fix this?"** — Adds a penalty that GROWS as coefficients grow, directly opposing the unbounded-growth incentive, giving the optimizer a genuine stopping point (balancing "explain the data perfectly" against "keep weights reasonable") instead of an infinite one-directional incentive.

**Common trap:** assuming huge coefficients simply mean "this feature is very important" — huge coefficient WITH huge SE is a red flag for a broken fit, not a strong signal.

---
---

# PART E — FAIRNESS & RESPONSIBLE ML

## 22. Fairness & Bias

### 22.1 WHY

A model used for lending, hiring, or content moderation must not just be ACCURATE — it must not systematically disadvantage protected groups, both ethically and (increasingly) legally. A model can have excellent overall accuracy/AUC while producing dramatically different error/approval rates across subgroups — even when no one intended any bias.

### 22.2 Intuition

A hiring model trained mostly on historical resumes from one demographic (because that's who was historically hired regardless of true merit) — even with NO explicit race/gender feature, other features can act as **proxies** (zip code correlating with race; "employment gap" correlating with gender via historical caregiving patterns). The model faithfully encodes real-world historical bias — "who has historically been hired" rather than "who would actually be a great hire" — confidently, with no awareness it's doing so.

### 22.3 Key fairness metrics (vocabulary FAANG interviewers expect)

**Demographic Parity (Statistical Parity):** approve/flag positive outcomes at roughly EQUAL RATES across groups, regardless of true underlying qualification rates. *Limitation:* can force the model to ignore genuine, legitimate group differences, sometimes actively fighting overall accuracy.

**Equal Opportunity:** among people who ACTUALLY deserve the positive outcome, the model should have the SAME recall/TPR across groups. `recall_groupA ≈ recall_groupB`.

**Equalized Odds:** a stricter version — BOTH TPR and FPR should be roughly equal across groups.

**Key tension:** these definitions can mathematically CONFLICT — it's formally proven that you generally cannot simultaneously satisfy demographic parity, equal opportunity, AND perfect calibration (§14) across groups if true base rates differ between groups. Not a failure of effort — a genuine mathematical impossibility in most real cases. **Choosing a fairness metric is a value judgment/tradeoff, not a purely technical decision.**

### 22.4 Worked numeric example

| | Group A recall (TPR) | Group B recall (TPR) |
|---|---|---|
| Among people who WOULD repay | 85% approved | 60% approved |

Among genuinely repay-worthy people, Group A gets approved 85% of the time, Group B only 60% — an **Equal Opportunity violation**, even if overall AUC looks fine when averaged across groups.

### 22.5 Interpretation

A single "overall AUC=0.85" can completely hide a fairness problem — the aggregate number looks great while masking a serious subgroup disparity. Segment-level analysis isn't just a modeling nicety here — for models touching real people's lives, it's often a genuine legal/ethical requirement.

### 22.6 FAANG L5 angle

**Q: "How would you check if your model is fair across demographic groups?"** — Compute recall/precision/FPR separately per subgroup (not just aggregate), compare against fairness definitions, explicitly note they can conflict — a real product/policy decision, not a pure ML one.

**Q: "Can you have a model perfectly calibrated for both groups AND satisfying equal opportunity?"** — Generally no if true base rates differ between groups — a well-known impossibility result. Interviewers may not expect a derivation, but knowing it EXISTS signals depth.

**Q: "The model doesn't use race as a feature — how could it still be biased?"** — Proxy features: other variables (zip code, name, behavioral patterns) can correlate strongly with protected attributes even when the attribute itself is excluded. "Fairness through unawareness" is a common but often-ineffective naive fix.

**Common trap:** thinking removing the sensitive attribute is sufficient to guarantee fairness — it isn't, due to proxies.

> **Check:** coefficient=1,200, SE=900 for a feature — first hypothesis? *(Separation/quasi-complete separation — some feature is perfectly/near-perfectly predicting the outcome; check whether small sample size or leakage is the cause.)*

> **Check:** equal approval rates across groups (demographic parity satisfied), but among genuinely qualified candidates, one group gets approved far less often. What's violated, and why is "equal approval rates" alone misleading? *(Equal Opportunity is violated. Equal approval rates alone can mislead because a model could satisfy demographic parity by approving unqualified people from one group MORE often while approving qualified people from another group LESS often — aggregate rates match, but the underlying fairness (are equally-deserving people treated equally?) doesn't.)*

---
---

# PART F — MODELING VARIANTS & MULTICLASS

## 23. Elastic Net

### 23.1 WHY

L1 gives sparsity (automatic feature selection) but behaves erratically with correlated features (arbitrarily zeroes out one of a correlated pair). L2 handles correlated features gracefully but never produces true sparsity. **Elastic Net: why choose? Blend both.**

### 23.2 Formula

```
new_cost = original_log_loss + λ1×(Σ|weights|) + λ2×(Σweights²)
```
`λ1=0` recovers pure L2 (Ridge); `λ2=0` recovers pure L1 (Lasso) — Elastic Net strictly generalizes both.

### 23.3 Worked example

3 correlated "location" features (zip, city, region — encoding similar info) + 2 unrelated features (complaints, tenure). Elastic Net can zero out redundant location features (L1 behavior) while smoothly shrinking (not violently destabilizing) whichever location feature survives — the L2 component stabilizes coefficients among the correlated group rather than letting L1 arbitrarily pick one and zero the rest.

### 23.4 FAANG angle

**Q: "When would you use Elastic Net over plain L1 or L2?"** — When you suspect BOTH some features are truly irrelevant (want L1's selection) AND some remaining features are correlated (want L2's stability) — a very common real-world combination with large, messy feature sets.

> **Check:** 3 correlated "device" features + several unrelated features — L1, L2, or Elastic Net? *(Elastic Net — the correlated trio benefits from L2's smooth handling of correlated features (vs. L1 arbitrarily zeroing two of three), while you likely also want L1's sparsity pressure among the unrelated features if some turn out irrelevant.)*

---

## 24. Weighted Logistic Regression (Sample Weights vs. Class Weights)

### 24.1 WHY

**Class weights** (Part I §34) treat errors on the rare class as more costly, uniformly across ALL rows of that class. **Sample weights** are more general: a custom importance weight per INDIVIDUAL ROW, for any reason — not just class membership.

### 24.2 Formula

```
total_cost = average of [ sample_weight × (-[y×log(p) + (1-y)×log(1-p)]) ]
```

### 24.3 Distinction table

| | Class weights | Sample weights |
|---|---|---|
| Granularity | One weight per CLASS | One weight per ROW |
| Typical use | Fixing class imbalance | Weighting by recency/reliability/survey confidence, correcting biased sampling |

### 24.4 Worked example

Extra survey data collected from a specific city, oversampled relative to its true population share, for statistical power. When training a national model, apply a SAMPLE weight to DOWN-weight those over-sampled rows back to their true population proportion — a use case class weights can't handle, since it has nothing to do with the target class.

### 24.5 FAANG angle

**Q: "Training data was collected via a biased sampling process — how do you correct for it?"** — Sample weights, proportional to the inverse of each row's sampling probability ("inverse propensity weighting"), correcting the model's effective view of the data back toward the true population distribution.

---

## 25. Multiclass: Softmax / Multinomial Logistic Regression (Full Derivation)

### 25.1 What problem softmax solves

Binary classification: one output → sigmoid → P(y=1|x). Multi-class (K classes) needs K outputs that (1) each lie in (0,1), and (2) sum to exactly 1. Applying sigmoid to each independently fails condition (2).

### 25.2 The function

```
softmax(zₖ) = e^zₖ / Σⱼ e^zⱼ,   k = 1...K
```
**Properties:** each output ∈ (0,1); outputs sum to 1 by definition; monotone in zₖ (higher logit → higher probability); differentiable everywhere.

### 25.3 Worked numeric example

Logits `z = [2.0, 1.0, 0.5]`:
```
e^2.0=7.389, e^1.0=2.718, e^0.5=1.649, sum=11.756
softmax = [0.629, 0.231, 0.140]   (sums to 1.0 ✓)
```
The highest logit dominates — softmax amplifies differences via the exponential.

### 25.4 The "soft" in softmax

Argmax is hard (1 for the max, 0 elsewhere — not differentiable). Softmax is a smooth, differentiable approximation, usable in backprop. As the max logit dominates more strongly, softmax → argmax.

### 25.5 Numerical stability — log-sum-exp trick

e^z overflows for large z. Fix: subtract the max logit first (mathematically identical, numerically stable):
```
softmax(z)ₖ = e^(zₖ−max(z)) / Σⱼ e^(zⱼ−max(z))
```
```python
def softmax_stable(z):
    z_shifted = z - np.max(z)
    e = np.exp(z_shifted)
    return e / e.sum()
```

### 25.6 The model — softmax regression

For K classes, learn K weight vectors:
```
zₖ = wₖᵀx + bₖ
p̂ₖ = softmax(z)ₖ = e^(wₖᵀx+bₖ) / Σⱼ e^(wⱼᵀx+bⱼ)
```

### 25.7 Loss — categorical cross-entropy

```
L = -Σᵢ Σₖ yᵢₖ·log(p̂ᵢₖ),   yᵢₖ = one-hot indicator
```
Only the true-class term survives per sample: `L = -Σᵢ log(p̂ᵢ,yᵢ)` — just the log-prob of the correct class.

### 25.8 Gradient — clean form

```
∂L/∂wₖ = Σᵢ (p̂ᵢₖ − yᵢₖ)·xᵢ  →  ∇_wₖ L = Xᵀ(p̂ₖ − yₖ)
```
Same residual-times-features structure as binary logistic regression.

### 25.9 The softmax gradient derivation (why the cross-terms cancel)

The tricky part: `∂softmax(z)ₖ/∂zⱼ` — class k's output depends on ALL logits via the denominator.
```
j=k:  ∂p̂ₖ/∂zₖ = p̂ₖ(1−p̂ₖ)      ← same form as sigmoid!
j≠k:  ∂p̂ₖ/∂zⱼ = −p̂ₖ·p̂ⱼ         ← cross-class terms
Compact: ∂p̂ₖ/∂zⱼ = p̂ₖ(δₖⱼ − p̂ⱼ)    [δ = Kronecker delta]
```
Composed with categorical cross-entropy, these cross-terms cancel beautifully into the clean gradient of §25.8.

### 25.10 Softmax reduces to sigmoid for K=2

```
p̂₁ = e^z₁/(e^z₁+e^z₂) = 1/(1+e^(z₂−z₁)) = σ(z₁−z₂)
```
Binary logistic regression is a special case of softmax regression.

### 25.11 One-vs-Rest (OvR) vs. Multinomial

**OvR:** trains K independent binary classifiers ("is this class k vs. everything else?"). Probabilities don't naturally sum to 1 (normalized post-hoc). Faster, works with any binary solver, less principled.
**Multinomial (true softmax):** one model, K logits, softmax over all simultaneously. Probabilities sum to 1 by construction, classes compete in the loss, more principled, usually better calibrated. **Use multinomial unless there's a specific reason not to** (e.g. `liblinear` only supports OvR).

```python
lr_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs')
lr_mn  = LogisticRegression(multi_class='multinomial', solver='lbfgs')
```

---

## 26. Ordinal Logistic Regression

### 26.1 WHY

Softmax treats multiclass outcomes as fully UNORDERED — correctly so for "Billing/Technical/Account." But some targets have a natural order (star ratings 1-5, satisfaction levels, disease stages). Throwing away that order (treating a predicted-5-actual-4 the same as predicted-5-actual-1) wastes real information.

### 26.2 Intuition

Instead of learning N separate scores like softmax, ordinal LR learns ONE continuous "latent score" (an internal invisible scale) PLUS a set of **cutoff thresholds** that carve that scale into ordered categories — like a ruler with tick marks: a customer's true satisfaction might land at 6.2, and if tick marks are at 2, 5, 8, they land in the "3-star" bucket.

### 26.3 Formula (3-category example: Low/Medium/High)

```
z = b + w1x1 + w2x2                        [same linear part as always]
P(rating ≤ Low)    = sigmoid(threshold_1 − z)
P(rating ≤ Medium)  = sigmoid(threshold_2 − z)
```
threshold_1 < threshold_2. The model learns ONE shared weight set across all categories PLUS the thresholds — a key structural difference from softmax, which learns entirely separate weights per class.

### 26.4 Worked example

z=1.5 for a customer, thresholds=(0, 2). Since 0<1.5<2, predicted category = "Medium" — same core idea as a single threshold, just with multiple ordered cutoffs.

### 26.5 FAANG angle

**Q: "Predicting satisfaction on a 1-5 scale — softmax or something else?"** — Ordinal logistic regression, since 1-5 has genuine order softmax would discard (treating "predicted 1, actual 5" the same as "predicted 4, actual 5" — clearly wrong, the first error is far worse). Evaluate with order-aware metrics (ranked accuracy, MAE on the category number) rather than plain multiclass accuracy.

**Common trap:** defaulting to softmax for ANY multiclass problem without asking "is there a natural order here?" — a good clarifying question to ask an interviewer if the problem statement is ambiguous.

> **Check:** predicting movie review scores (1-5 stars) — softmax or ordinal, and why? *(Ordinal — a review predicted 4 when truth is 5 is a much smaller mistake than predicted 1 when truth is 5; softmax treats every wrong category equally, discarding real "how far off" information that ordinal's latent score + thresholds naturally captures.)*

---

## 27. Rare-Event Bias Correction (Firth's Correction)

### 27.1 WHY

When the positive class is EXTREMELY rare (not just imbalanced like 1-in-1,000, but more like 1-in-100,000), even standard MLE starts producing systematically biased estimates — coefficients tend to come out LARGER in magnitude than the true effect, purely as an artifact of having so few positive examples, not because the true effect is that large.

### 27.2 Intuition

Witnessing a rare event only 3 times, each time with a specific unusual circumstance present — you might wildly overestimate how strongly that circumstance "causes" the event. With so few observations, the estimate is extremely sensitive to the small sample, and this sensitivity has a known DIRECTIONAL bias (pushing estimates too extreme, not just noisy in both directions).

### 27.3 What Firth's correction does (conceptually)

Modifies the standard log-likelihood by adding a small penalty term specifically designed to counteract this small-sample/rare-event bias, pulling estimates back toward more realistic values — especially useful with separation-like symptoms (§21) driven by genuine rarity rather than a data error. You need to know: (1) it exists, (2) it solves rare-event/small-sample bias specifically, (3) it's closely related to the separation problem (both share the root cause: too little data relative to how "clean" a pattern appears) — not to derive the penalty term itself.

### 27.4 Worked example (conceptual)

Very rare disease: 40 cases / 200,000 patients. Standard LR on a key risk factor: coefficient=3.8 (odds ratio e^3.8≈45×). Firth's correction on the SAME data: coefficient=2.6 (OR≈13×) — still meaningfully elevated, but pulled back from an inflated, small-sample-biased extreme toward a more realistic, defensible value.

```python
# from firthlogist import FirthLogisticRegression
# model = FirthLogisticRegression().fit(X, y)
```

### 27.5 FAANG angle

**Q: "Rare disease dataset — only 40 positive cases out of 200,000. What should you be cautious about?"** — Standard MLE can be meaningfully biased with such a rare positive class — consider Firth's correction, and be alert to potential separation given how few positive examples constrain the fit.

**Common trap:** treating "more data always helps" as universal — for RARE events specifically, the relevant sample size is the COUNT OF POSITIVE EVENTS, not total rows — 200,000 rows sounds like plenty, but 40 events is genuinely small for estimating that effect.

---
---

# PART G — SYSTEMS & SCALE

## 28. Sparse / High-Cardinality Categorical Features

### 28.1 WHY

Production systems at scale deal with categorical features with MILLIONS of possible values (user ID, product ID, search query, URL). One-hot encoding creates one column PER category — with 10 million unique product IDs, that's 10 million columns: won't fit in memory or train in reasonable time.

### 28.2 Intuition

A filing cabinet with a separate labeled drawer for every customer who's ever shopped — with millions of customers, you'd need millions of mostly-empty drawers. Impractical; you need a smarter system that doesn't require a dedicated drawer per possible value.

### 28.3 Technique 1 — The Hashing Trick

Run each category value through a hash function; use the result modulo a FIXED number of buckets to decide which small, fixed column that category lands in. Multiple categories can collide into the same bucket — an accepted tradeoff.
```
column_index = hash(category_value) % num_buckets
```
Even with millions of raw categories colliding into thousands of buckets, the model can still learn useful patterns per bucket. The huge win: you fix feature dimensionality in advance regardless of how many categories exist, critical for handling brand-new category values appearing after training without retraining/resizing anything.

**Worked example:** `num_buckets=5`. `hash("shoe_042")%5=4 → bucket 4`, `hash("shirt_119")%5=1 → bucket 1`, `hash("hat_881")%5=0 → bucket 0`. A 4th product hashing into bucket 4 (colliding with "shoe_042") means the model can't perfectly distinguish those two specific products via this feature — an accepted tradeoff for massive space savings.

### 28.4 Technique 2 — Embeddings (bridge to deep learning)

Represent each category as a small vector of learned numbers (e.g. 8-16 dims) learned during training — similar categories end up with similar vectors. Usually learned via a neural embedding layer, not plain LR's scalar-weight-per-hashed-input — a genuine bridge point pushing high-cardinality handling toward the neural network side.

**Wide & Deep pattern (Google-scale):** "Wide" part uses simple sparse (hashed) features feeding into an LR-like component for memorization of specific combinations; "Deep" part uses embeddings feeding into a neural network for generalization to unseen combinations.

### 28.5 FAANG angle

**Q: "A categorical feature with 50 million unique values — how would you incorporate it?"** — One-hot is infeasible; use the hashing trick (e.g. 100K buckets), accepting collision risk, or a learned embedding in a neural architecture — mention Wide & Deep as a real-world combined example.

**Q: "Downside of hashing, and mitigation?"** — Collisions (unrelated categories sharing a bucket, "blurring" the model's ability to distinguish them). Mitigate with more buckets (memory/compute tradeoff) or a smarter hash minimizing collision-driven bias for known-important categories.

**Common trap:** suggesting one-hot encoding without acknowledging the scale problem.

---

## 29. Real-Time vs. Batch Features & Feature Stores

### 29.1 WHY

A trained model needs features AT prediction time. Some are cheap/instant (current hour of day); others need expensive historical aggregation (90-day average spend) — computing that fresh, on-the-fly, per request may be too slow for strict latency SLAs. Without a deliberate strategy: either stale features (hurting accuracy), painfully slow predictions (hurting UX), or the sneakiest failure — **training-serving skew** (feature computed differently for training vs. production), silently degrading performance with no obvious error.

### 29.2 Intuition

A restaurant kitchen: some ingredients (salt, oil) sit ready instantly (real-time/on-demand features); others (a 6-hour slow-simmered stock) get prepared in a big batch in advance and pulled from storage (batch-precomputed features). A well-run kitchen needs both, matched to each ingredient's actual constraints.

### 29.3 Batch Features

Compute periodically (e.g. daily), for every entity, store the result; at request time, LOOK UP the precomputed value instead of calculating fresh. Use when freshness-to-the-minute doesn't matter (e.g. 90-day average spend).

### 29.4 Real-Time (Streaming) Features

Compute on-the-fly, incorporating the latest events, right as the request arrives. Use when freshness genuinely matters (e.g. "clicks in the last 60 seconds" for fraud detection).

### 29.5 Feature Stores

A centralized system that computes, stores, and serves features consistently — critically, ensuring the EXACT SAME computation logic is used both during training (on historical data) and serving (live predictions), eliminating training-serving skew **by construction** rather than by careful manual discipline. This is the infrastructure-level solution to the training-serving skew problem: one feature definition, used consistently in both contexts, rather than two independently-written implementations hoping to stay in sync.

### 29.6 Worked example (conceptual)

Fraud model feature: "transaction count in last 5 minutes." **Without a feature store:** training pipeline uses one batch SQL definition; serving code (written separately, weeks later) uses a subtly different time window or "transaction" definition (does it include declined transactions?) — same feature name, two different meanings, undetected for months. **With a feature store:** one definition, pulled identically by both training and serving.

### 29.7 FAANG angle

**Q: "Design the feature pipeline for a real-time fraud detection model."** — Categorize each feature as batch (historical patterns, computed daily) vs. real-time (transaction velocity, computed on-the-fly); propose a feature store for training/serving consistency; mention latency budgets and millisecond-level SLAs for real-time features.

**Q: "What happens if a real-time feature service is down when a request comes in?"** — Graceful degradation: fall back to a slightly stale cached value rather than failing the prediction entirely, and/or design the model to be somewhat robust to a missing feature (ties to §18's missingness-flag idea).

**Common trap:** "just compute everything in real-time" — real-time computation for EVERY feature is often neither necessary nor affordable at scale.

> **Check:** "user's average rating over their lifetime" — batch or real-time? What about "movies clicked in the last 30 seconds of this session"? *(Lifetime average → batch, changes slowly, recomputing daily/weekly is far cheaper than live. Last-30-seconds clicks → real-time, useless if stale — a recommendation engine needs to react to what the user JUST did.)*

> **Check:** why does a feature store solve training-serving skew "by construction," not just reduce the CHANCE of it? *(Only ONE definition of each feature exists, used by both pipelines — there's no second, separately-written implementation that could drift out of sync. "Reducing the chance" implies two implementations that COULD still diverge; "by construction" means structurally only one implementation exists, so divergence isn't possible.)*

---

## 30. Online / Incremental Learning

### 30.1 WHY

What do you do when new data arrives CONTINUOUSLY and retraining the ENTIRE model from scratch every time is too slow/expensive? Retraining from scratch too infrequently lets the model go stale; too frequently burns enormous unnecessary compute reprocessing years of old data to incorporate one new day.

### 30.2 Intuition

Gradient descent's mini-batch/single-point (SGD) updates already don't require the full dataset at once. Online learning takes this to its natural conclusion: treat new data as a continuous, never-ending stream of mini-batches, updating the EXISTING model incrementally rather than restarting from zero. Analogy: re-reading an entire textbook from page 1 for every new fact (batch retraining) vs. adding new notes to your existing understanding as you learn things (online learning) — you keep prior knowledge (current weights) and nudge it, rather than throwing everything out.

### 30.3 Formula — warm-starting

```
new_weight = current_deployed_weight − (learning_rate × gradient_from_new_data_only)
```
Start from the CURRENTLY DEPLOYED weights (not zero/random — the "warm start"); run a few gradient steps using only the new batch; deploy the updated weights; repeat.

**Practical consideration:** online updates often use a SMALLER, more conservative learning rate than full initial training — a single unusual new batch (e.g. a genuine but atypical fraud pattern) could otherwise swing weights too aggressively based on limited fresh evidence, destabilizing a model well-calibrated on much more data.

### 30.4 FAANG angle

**Q: "How would you keep a fraud model up to date without daily full retraining?"** — Online/incremental learning: warm-start from deployed weights, gradient-update on new batches only, typically smaller learning rate; periodically (weekly/monthly) still do a full retrain as a "reset" to avoid compounding drift.

**Q: "Risk of ONLY doing online updates, never fully retraining?"** — Incremental updates can compound small biases/drift over time, and the model may never fully "unlearn" outdated patterns the way a fresh retrain would — most production systems use a HYBRID (frequent light incremental updates + periodic full retrains).

**Common trap:** "more data always helps" without recognizing that for rare events the relevant count is positive events, not total rows (ties directly to §27).

> **Check:** small hourly incremental updates for 6 months, never a full retrain — biggest risk? *(Compounding drift/bias accumulation — small errors or outdated patterns from very old data may never get "unlearned" the way a fresh full retrain would remove them; without a periodic reset, the model can drift further from what a from-scratch fit on current data would produce.)*

> **Check:** why does rare-event bias connect back to Separation — shared cause? *(Both stem from too little data — specifically too few positive examples — relative to how "clean"/extreme the apparent pattern looks; small samples let both MLE bias and separation-driven coefficient blow-up occur.)*

---
---

# PART H — GLMs & NEURAL NETWORK COMPARISON

## 31. Generalized Linear Models — The Unifying Framework

### 31.1 The problem GLMs solve

Ordinary linear regression assumes: continuous, unbounded target; Gaussian noise; mean response linear in features. Real data violates all three constantly — binary outcomes (churn, fraud), count outcomes (clicks, accidents), positive-only outcomes (income, time-to-event). GLMs extend linear regression by changing the target's **distribution** and the **link function** connecting the linear predictor to the mean.

### 31.2 The three components of every GLM

```
1. Random component:     distribution of y  (Gaussian, Bernoulli, Poisson, ...)
2. Systematic component: linear predictor  η = wᵀx + b
3. Link function:        g(μ) = η,   μ = E[y|x]
```
The link g maps the constrained mean μ to the unconstrained linear predictor η. The inverse `μ = g⁻¹(η)` is the **mean function**/response function.

### 31.3 The exponential family — where GLMs come from

```
P(y|θ,φ) = exp[(y·θ − b(θ))/φ + c(y,φ)]
θ = natural parameter, b(θ) = log-partition function, φ = dispersion parameter
E[y] = b'(θ),   Var[y] = φ·b''(θ)
```
This is WHY a distribution's mean and variance are linked — both derive from b(θ).

### 31.4 The canonical link

The link where `g(μ)=θ` — i.e. the linear predictor IS the natural parameter directly: `η = wᵀx = θ`. Special because it gives a clean sufficient statistic (Xᵀy), clean score equations, simplified Fisher information, and — critically — **the gradient always has the form Xᵀ(y−μ) for every GLM**, same structure every time. Not always the "best" link for a problem, but the principled default derived from the probability model itself.

### 31.5 Every major GLM

**Linear Regression:** Gaussian, support (−∞,∞), natural param μ, canonical link = Identity (g(μ)=μ), mean function μ=η, loss=MSE. The identity link means "do nothing" — linear regression is the base-case GLM. Var[y]=σ² (constant, homoscedastic).

**Logistic Regression:** Bernoulli, support {0,1}, natural param = logit `log(p/(1−p))`, canonical link = Logit, mean function = Sigmoid `1/(1+e⁻η)`, loss = binary cross-entropy. **The logit isn't invented — it's the natural parameter of the Bernoulli distribution; sigmoid is forced as the mean function.** Var[y]=p(1−p) (heteroscedastic by nature — variance depends on the mean).

**Poisson Regression:** Poisson, support {0,1,2,...}, natural param `log(λ)`, canonical link = Log (g(μ)=log μ), mean function `μ=e^η`, loss = Poisson NLL. Used for counts (website visits, insurance claims). Log link ensures μ>0 always; exponential mean function means coefficients are multiplicative. Var[y]=λ (variance=mean; often violated in practice → Negative Binomial instead).

**Gamma Regression:** Gamma, support (0,∞), canonical link = Inverse (g(μ)=1/μ), mean function μ=1/η (in practice, log link is often used instead for better numerical stability). Used for positive right-skewed data (insurance claims, income, time-to-event). Var[y]=μ²/α — variance scales with mean squared.

**Inverse Gaussian Regression:** support (0,∞), canonical link = inverse-squared (g(μ)=1/μ²). Used for even-more-right-skewed positive data than Gamma; less common.

### 31.6 Summary table

| GLM | Distribution | Support | Canonical Link g(μ) | Mean function g⁻¹(η) | Variance |
|---|---|---|---|---|---|
| Linear | Gaussian | (−∞,∞) | Identity: μ | η | σ² (constant) |
| Logistic | Bernoulli | {0,1} | Logit: log(μ/1−μ) | Sigmoid: 1/(1+e⁻η) | μ(1−μ) |
| Poisson | Poisson | {0,1,2,...} | Log: log(μ) | Exp: e^η | μ |
| Gamma | Gamma | (0,∞) | Inverse: 1/μ | 1/η | μ²/α |
| Inv. Gaussian | Inv. Gaussian | (0,∞) | Inv. squared: 1/μ² | 1/√η | μ³/λ |

### 31.7 Non-canonical links

You CAN use a non-canonical link. Common reasons: Binomial with log link (estimates log-risk ratios — preferred in epidemiology); Poisson with identity link (additive rather than multiplicative effects); Gamma with log link (more stable numerics than the canonical inverse link). Non-canonical links lose the clean sufficient-statistic structure but are perfectly valid — the canonical link is the principled default, not a requirement.

### 31.8 The unified gradient — why all GLMs share the same form

For ANY GLM with its canonical link: `∇_w ℓ = Xᵀ(y − μ)` — always. Residuals times features. Linear, logistic, Poisson regression — same gradient structure. This is the deep reason GLMs are a unified family, not ad-hoc models. The only difference is what μ is: Linear `μ=wᵀx`; Logistic `μ=σ(wᵀx)`; Poisson `μ=e^(wᵀx)`.

---

## 32. Neural Networks vs. Logistic Regression

### 32.1 Logistic regression IS a neural network

Literally — a single-layer neural network with sigmoid activation and binary cross-entropy loss IS logistic regression. Same model, same loss, same gradient.
```
Logistic Regression:  x → [w1x1+w2x2+b] → σ(·) → p̂     (one layer, no hidden units)
Neural Network:        x → W1x+b1 → [hidden h] → W2h+b2 → σ(·) → p̂   (hidden layer(s) added)
```

### 32.2 Why LR's boundary is linear and a NN's isn't

**LR:** the decision boundary is `wᵀx+b=0` — sigmoid is monotone, so it rescales the output but never changes which side of zero the linear combination is on. The boundary is ALWAYS a hyperplane (full proof in §6.4).

**Two-layer NN:** `h=σ(W1x+b1)`, `p̂=σ(W2h+b2) = σ(Σⱼw2ⱼ·σ(w1ⱼᵀx+b1ⱼ)+b2)` — a sigmoid of a weighted sum of sigmoids of linear functions of x. No longer linear in x — the inner sigmoids create nonlinear transformations the outer layer combines.

**Why composition of LINEAR functions alone stays linear:** without activation functions between layers, `h=W1x+b1` and `p̂=W2h+b2=(W2W1)x+(W2b1+b2)` collapses to a single linear transform — a 100-layer purely-linear network is mathematically identical to a one-layer one. **Depth without nonlinearity is useless** — the activation function between layers is what makes depth meaningful.

### 32.3 The Universal Approximation Theorem

A neural network with one hidden layer and enough hidden units can approximate any continuous function on a compact domain to arbitrary precision. Logistic regression cannot — it's limited to linear boundaries (or whatever you construct via explicit feature engineering, §7).

### 32.4 The feature-learning perspective (the deepest difference)

**LR:** you engineer features manually, then fit a linear model on them — `Raw features → [YOU DO THIS] → engineered features → linear model → output`.
**NN:** hidden layers learn the feature transformation automatically from data — `Raw features → [LEARNED BY HIDDEN LAYERS] → learned features → linear model → output`.

The FINAL layer of a neural network IS logistic regression — a linear classifier on top of whatever the hidden layers learned. The hidden layers' job is transforming raw features into a space where linear classification works. **"Representation learning" is the core idea of deep learning.**

### 32.5 XOR — the canonical worked example

```
(0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0
```
No straight line separates class 0 from class 1 (also shown in §7.8). A 2-hidden-unit NN solves it:
```
h1 = σ(x1+x2−0.5)   ≈ "is at least one input on?"
h2 = σ(x1+x2−1.5)   ≈ "are both inputs on?"
output = σ(h1 − 2·h2) ≈ "at least one but not both" = XOR
```
The hidden layer transformed (x1,x2) into (h1,h2) — a space where XOR IS linearly separable. Logistic regression in (h1,h2)-space solves it.

### 32.6 Summary table

| | Logistic Regression | Neural Network |
|---|---|---|
| Decision boundary | Always a hyperplane (linear) | Arbitrary nonlinear |
| Feature engineering | Manual | Learned from data |
| Parameters | d+1 (one per feature) | Can be millions |
| Training | Convex loss, guaranteed convergence | Non-convex, local minima |
| Interpretability | High (coefficients = odds ratios) | Low (black box) |
| Data needed | Works with small data | Needs lots of data |
| When it wins | Linear relationships, interpretability required | Complex patterns, raw inputs (images, text) |

**In one line:** Logistic regression = neural network with zero hidden layers. Neural network = logistic regression on top of learned nonlinear features.

---
---

# PART I — sklearn & IMBALANCED DATA

## 33. `sklearn.LogisticRegression` — Every Parameter Explained

```python
from sklearn.linear_model import LogisticRegression
LogisticRegression(
    penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True,
    intercept_scaling=1, class_weight=None, random_state=None,
    solver='lbfgs', max_iter=100, multi_class='auto', verbose=0,
    warm_start=False, n_jobs=None, l1_ratio=None
)
```

**`penalty`:** `'l2'` = Ridge (shrinks all weights, keeps all features). `'l1'` = Lasso (drives some weights exactly to zero → feature selection). `'elasticnet'` = combination, needs `l1_ratio` set. `None` = no regularization, MLE only, will diverge on separable data (§21). Default to `l2` unless you need sparsity; `l1` with many suspected-irrelevant features; `elasticnet` when you want sparsity but L1 alone is too aggressive.

**`C`** — regularization strength (inverse of λ):
```
Loss = cross_entropy + (1/C)·penalty
Small C → strong regularization → smaller weights → simpler model
Large C → weak regularization  → closer to raw MLE
```
`C=1/λ`. Default C=1.0 is arbitrary — always tune via cross-validation:
```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LogisticRegression(), {'C':[0.001,0.01,0.1,1,10,100]}, cv=5)
```

**`solver`:**

| Solver | Supports penalties | Best for |
|---|---|---|
| `lbfgs` | l2, None | Default; good for most cases, uses second-order (quasi-Newton) info |
| `liblinear` | l1, l2 | Small datasets; best small-data L1 support |
| `saga` | l1, l2, elasticnet, None | Large datasets; stochastic gradient; only solver for elasticnet |
| `sag` | l2, None | Large datasets; faster than lbfgs on large n |
| `newton-cg` | l2, None | Similar to lbfgs |

Rules: small data → lbfgs/liblinear; need L1 → liblinear (small) or saga (large); need elasticnet → saga only; large dataset → saga/sag.

**`max_iter`:** default 100 — the most common warning source (`ConvergenceWarning: lbfgs failed to converge`). This doesn't mean the model is wrong — the optimizer just hasn't reached `tol` in 100 steps. **Fix order:** (1) scale features first (unscaled features slow convergence dramatically, §20), (2) increase max_iter (200/500/1000), (3) if still failing at 10,000, check for separation (§21) or extreme features.

**`tol`:** convergence tolerance; optimizer stops when loss change between iterations is below this. Default 1e-4. Smaller → more precise, more iterations. Usually fine at default; loosen to 1e-3 before increasing max_iter if convergence is slow.

**`fit_intercept`:** `True` (default) = model includes bias `σ(wᵀx+b)` — almost always correct. `False` only if you've manually centered features (mean=0 for all).

**`class_weight`:** `None` = equal weighting. `'balanced'` = weight class k by `n_samples/(n_classes×n_k)`. `{0:1, 1:10}` = manual weights. Full math in §34.

**`multi_class`:** `'auto'` picks `'ovr'` for liblinear, `'multinomial'` otherwise. `'ovr'` = One-vs-Rest (K binary classifiers, fast, less principled). `'multinomial'` = true softmax (§25) — the right choice for multiclass with `lbfgs` or `saga`.

**`l1_ratio`:** only used with `penalty='elasticnet'`:
```
Loss = cross_entropy + (1/C)·[l1_ratio·Σ|wⱼ| + (1−l1_ratio)·Σwⱼ²]
l1_ratio=0 → pure L2, l1_ratio=1 → pure L1, l1_ratio=0.5 → equal mix
```

**`warm_start`:** `False` (default) reinitializes weights each `fit()` call; `True` starts from previous weights — useful when tuning C across a grid, starting each next value from the previous (often nearby) solution.

**`random_state`:** only matters for `sag`/`saga` (stochastic solvers shuffle data) — set for reproducibility.

**`n_jobs`:** only used in One-vs-Rest multiclass (parallel K-classifier training). `-1` = all cores. No effect for binary classification.

**`dual`:** only for `liblinear` with L2. `n_samples > n_features` → `dual=False` (primal, default). `n_features > n_samples` → `dual=True` can be faster. Almost always leave at default.

---

## 34. Imbalanced Data — The Full Toolkit

### 34.1 The problem

95% class 0, 5% class 1. Cross-entropy weights every sample equally, but there are 19× more class-0 samples, so the loss is dominated by getting class 0 right. **Result:** the model learns to predict 0 almost always. Accuracy=95%, but recall on class 1 ≈ 0 — useless for the actual task.

### 34.2 What `class_weight='balanced'` does — the exact math

```
wₖ = n_samples / (n_classes × nₖ)
```
For a 1000-sample, 950/50 split:
```
w₀ = 1000/(2×950) = 0.526
w₁ = 1000/(2×50)  = 10.0
```
Each class-1 sample now contributes `10/0.526 ≈ 19×` more to the loss than a class-0 sample — exactly canceling the imbalance.

**Modified loss:** `L = −Σᵢ sampleweight(i)·[yᵢlog(p̂ᵢ)+(1−yᵢ)log(1−p̂ᵢ)]`
**Modified gradient:** `∇_w L = Xᵀ D (y−p̂)`, `D = diag(sampleweight(i))` — the minority-class residuals are amplified, shifting the decision boundary to correctly classify minority samples even at the cost of more majority errors (equivalent to moving the classification threshold below 0.5).

### 34.3 The full toolkit, in order of typical escalation

**Level 1 — change the loss:**
```python
lr = LogisticRegression(class_weight='balanced')
# or: LogisticRegression(class_weight={0: 1, 1: 19})
```

**Level 2 — resample the data:**
```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X_train, y_train)     # synthetic minority interpolation

from imblearn.under_sampling import RandomUnderSampler
X_res, y_res = RandomUnderSampler().fit_resample(X_train, y_train)  # loses data
```
SMOTE creates new synthetic minority samples by interpolating between existing ones in feature space — not just duplication.

**Level 3 — tune the threshold** (default 0.5 is almost always wrong for imbalanced data):
```python
lr.fit(X_train, y_train)
probs = lr.predict_proba(X_val)[:, 1]
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.01)
f1s = [f1_score(y_val, probs > t) for t in thresholds]
best_threshold = thresholds[np.argmax(f1s)]
```

**Level 4 — use the right metrics:**
```python
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, preds))         # precision/recall/F1 per class — never accuracy alone
print("AUC-ROC:", roc_auc_score(y_test, probs))
# average_precision_score (AUC-PR) is better than AUC-ROC for heavy imbalance
```

### 34.4 When to use what

| Imbalance ratio | Recommended approach |
|---|---|
| 70/30 | Usually fine, maybe class_weight='balanced' |
| 90/10 | class_weight='balanced' + tune threshold |
| 95/5 | class_weight + SMOTE + tune threshold + AUC-PR metric |
| 99/1 | All of the above; consider anomaly detection instead |

---
---

# PART J — COMPARISONS & PITFALLS

## 35. Logistic Regression vs. Linear Regression

### 35.1 Side-by-side

| | Linear Regression | Logistic Regression |
|---|---|---|
| Output | ŷ ∈ (−∞,∞) | p̂ ∈ (0,1) |
| Target variable | Continuous | Binary (or probability) |
| Link function | Identity | Logit |
| Loss | MSE | Cross-entropy (NLL) |
| Loss surface | Convex | Convex |
| Closed-form solution | Yes (normal equations) | No |
| Assumptions | Gaussian noise | Bernoulli noise |
| Coefficients mean | Unit change in y per unit xⱼ | Unit change in log-odds per unit xⱼ |

### 35.2 They're the same underlying structure

Both compute `output = wᵀx+b`; the difference is what you do with it: Linear uses it directly (`ŷ=wᵀx+b`); Logistic passes it through sigmoid (`p̂=σ(wᵀx+b)`). Both are GLMs (Part H §31) — linear uses the identity link, logistic uses the logit link.

### 35.3 Why linear regression fails on binary targets

**Unbounded outputs:** nothing constrains predictions to [0,1] — a "probability" of −0.3 or 1.4 is meaningless. **MSE distorts the boundary:** an outlier positive example pulls the regression line toward it, shifting the decision threshold incorrectly (e.g. adding one extreme point tilts the line, shifting the threshold from x=3 to x=5, misclassifying points at x=4 that were previously fine). **Wrong noise model:** MSE assumes Gaussian residuals; binary labels are Bernoulli.

**When linear regression accidentally "works":** with balanced classes, no extreme outliers, and predictions that happen to stay in [0,1], linear regression approximates logistic regression (the Linear Probability Model in econometrics) — coincidence, not correctness.

---

## 36. Logistic Regression vs. Decision Trees

### 36.1 Fundamental difference

```
Logistic Regression:  finds a hyperplane in feature space
Decision Tree:        partitions feature space into axis-aligned rectangles
```
LR: one global linear boundary. Tree: many local rules, each applying to a region. LR gets diagonal boundaries for free; trees need exponentially many leaves to approximate a diagonal/curved boundary but handle any shape natively given enough depth.

### 36.2 Assumptions

**LR:** log-odds linear in features; features should be scaled (§20); sensitive to outliers (§19); requires explicit feature engineering for nonlinear boundaries (§7).
**Trees:** no distributional assumptions; invariant to monotone feature transforms (scaling irrelevant); handle nonlinear boundaries natively; some implementations handle missing values natively.

### 36.3 Interpretability, differently

```
LR:   "a 1-unit increase in age multiplies the odds by e^0.3 = 1.35"
Tree: "if age > 40 AND income < 50k → predict churn"
```
Tree rules are easier for non-technical stakeholders; LR coefficients are better for statistical inference (CIs, p-values, odds ratios, Part B).

### 36.4 Overfitting

LR: regularize with L1/L2, relatively stable (only d parameters). Trees: prone to overfitting (can memorize training data), need pruning/depth limits.

### 36.5 When to use which — combined table

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

**The ensemble angle:** in practice, bare decision trees are rarely the real answer — the true comparison is LR vs. Random Forests or Gradient Boosted Trees (XGBoost, LightGBM), which combine many trees and almost always beat bare LR on tabular data, unless interpretability, fast training, or a genuinely linear relationship favors LR.

---

## 37. Common Mistakes & Pitfalls — Consolidated List

1. **Not scaling features** — sensitive to scale differences; unscaled features make gradient updates for different weights operate on wildly different scales, slowing/destabilizing training (§20). Fix: standardize before training, always.
2. **Perfect separation (complete/quasi-complete)** — coefficient → ±∞, doesn't converge meaningfully (§21). Signs: very large coefficients, huge SEs, convergence warnings. Fix: L2 regularization (always), or reduce/investigate the separating feature.
3. **Multicollinearity** — correlated features make coefficient estimates unstable (small data changes → large w swings); predictions stay okay but coefficients can't be trusted as importances. Fix: drop one correlated feature, PCA, or L2 (shrinks correlated coefficients together).
4. **Ignoring class imbalance** — 95/5 split → predicting 0 always gives 95% accuracy but the model is useless (§34). Fix: `class_weight='balanced'`, SMOTE/undersampling, or evaluate with F1/AUC-ROC/AUC-PR, not accuracy.
5. **Interpreting coefficients without odds ratios** — `w=0.7` does NOT mean "70% more likely"; correct: `e^0.7≈2.01` → doubles the odds (§15, §35). Saying "increases probability by 70%" is wrong — the actual probability change depends on the base rate.
6. **Not checking linearity in log-odds** — if the true relationship is quadratic/U-shaped, the model silently fits the wrong shape (§7). Check via a binned log-odds plot; add polynomial/interaction features if needed.
7. **Assuming predicted probabilities are calibrated without checking** — probabilities can be miscalibrated, especially with regularization or imbalanced data (Part C, §14). Fix: plot calibration curves; use `CalibratedClassifierCV` if needed.
8. **Thresholding at 0.5 by default** — only optimal when misclassification costs are equal and classes are balanced. In fraud/cancer-screening contexts, tune the threshold on a validation set to match the cost asymmetry (precision-recall curve).

**"Why does logistic regression always converge?"** — The log-likelihood is concave in w (negative semi-definite Hessian, §2.7/§5.6), so there's a unique global maximum (assuming no separation, §21).

---
---

# PART K — MASTER REFERENCE TABLES & CONSOLIDATED CHECKS

## 38. Master Quick-Reference

```
Model:         p̂ = σ(wᵀx + b) = 1 / (1 + e^(−wᵀx−b))
Loss:          L = −Σᵢ [yᵢ log p̂ᵢ + (1−yᵢ) log(1−p̂ᵢ)]
Gradient:      ∇L = Xᵀ(p̂ − y)
Hessian:       H = XᵀWX,  W = diag(p̂ᵢ(1−p̂ᵢ))
Update:        w ← w − α · Xᵀ(p̂ − y)
Boundary:      wᵀx + b = 0
Convexity:     vᵀHv = Σᵢ p̂ᵢ(1−p̂ᵢ)(xᵢᵀv)² ≥ 0  ✓

Inference tests:
Test                Formula                       Dist under H₀    Use when
Wald               W = (ŵ/SE)²                   χ²(1)            Quick, large n
LRT                G² = 2(ℓ_full − ℓ_red)        χ²(df)           Small n, separation, group tests
Score              S = gradient at H₀             χ²(k)            Only fit null model
Omnibus            G² = 2(ℓ_full − ℓ_null)       χ²(p)            Test whole model
Hosmer-Lemeshow    Σ(O−E)²/(n·p̄·(1−p̄))          χ²(g−2)          Calibration/GoF

Pseudo-R²:  McFadden = 1 − ℓ_full/ℓ_null   |   Nagelkerke = rescaled Cox-Snell to [0,1]
Calibration: ECE = Σ(nₖ/n)|p̄ₖ − ȳₖ|  → 0 is perfect. Fix: Platt scaling, isotonic regression, temperature scaling.

GLM family:
GLM       | Link (canonical) | Mean function | Loss
Linear    | Identity         | η              | MSE
Logistic  | Logit            | Sigmoid        | Cross-entropy
Poisson   | Log              | e^η            | Poisson NLL
Gamma     | Inverse          | 1/η            | Gamma NLL

Non-linearity fixes:  smooth curve → polynomial (x²) | arbitrary shape → binning
                       2 numeric features interact → interaction term (x1×x2)
                       2 categorical features interact → feature cross

Imbalance escalation:  class_weight → resampling (SMOTE) → threshold tuning → AUC-PR/F1 metrics
Leakage types:         target | temporal | group/entity | preprocessing
```

## 39. Consolidated Interview Check Questions (with model answers where given in source material)

*(This section gathers every "CHECK" question from across the topics above into one drill list — model answers are inlined in italics where the original material supplied one; questions without an inlined answer are meant to be worked through live, using the relevant section above.)*

1. §2.10 — If Customer 1 had 5 complaints instead of 2, would `gradient_w` become more negative or more positive? *(More negative.)*
2. §2.10/§2.11 — Why take the PRODUCT of row probabilities, not the sum, for dataset likelihood? *(Independent events combine via multiplication.)*
3. §4 — "Is logistic regression a linear model?" — answer using all three senses of linearity.
4. §6.3 — Moving the threshold from 0.5 to 0.9 — does the boundary move closer to or farther from the positive cluster? *(Closer.)*
5. §6.4 — Does the linear-boundary proof still hold with x1² substituted in? What does the boundary look like in terms of raw x1? *(Yes in expanded space; quadratic/parabola in raw x1.)*
6. §7.9 — U-shaped "distance from store" relationship — which technique first? *(Polynomial features.)*
7. §7.9 — Why doesn't an interaction term violate "linearity in log-odds"? *(Linear in the expanded/engineered feature space.)*
8. §8.7 — Why might LR outperform Naive Bayes with enough data, despite NB's simpler assumptions? *(No independence assumption forcing it to ignore real correlations.)*
9. §9.7 — Two coefficients =0.40, SE=0.05 vs SE=0.60 — which do you trust, and what are the Wald stats? *(z=8.0 vs z=0.67; trust the smaller-SE one.)*
10. §9.7 — Tiny p-value (0.001) ⇒ "huge real-world effect"? What's wrong? *(Confuses significance with effect size.)*
11. §10.6 — G² by hand: ℓ_simple=-200.0, ℓ_complex(+2 params)=-197.5. *(G²=5.0, df=2, not significant at 5% vs. threshold ≈5.99.)*
12. §10.6 — Why can't you LR-test two completely disjoint feature-set models? *(Not nested.)*
13. §12.5 — Under-confident calibration curve (predicted 20%, actual 35%) — does Platt scaling still apply, and which direction? *(Yes; stretches predictions upward.)*
14. §15.5 — Raw `w_age=1.2` vs `w_income=0.00001` — can you conclude age matters 120,000× more? *(No — must standardize first.)*
15. §15.5 — Permutation importance artifact with two correlated features? *(Both appear artificially unimportant individually.)*
16. §16.6 — Model A (train 0.15/val 0.45) vs Model B (train 0.30/val 0.32) — which to pick? *(B — smaller generalization gap.)*
17. §16.6 — Why is cross-validation more expensive, and when is that cost not worth it? *(K trainings; not worth it on very large, already-stable datasets.)*
18. §17.7 — "Length of discharge summary note" predicting readmission — leakage red flag? *(Possibly — investigate timing of note relative to prediction point.)*
19. §17.7 — Time-split churn model but "avg monthly spend" uses full history including future rows — what leakage, and the fix? *(Temporal leakage; recompute using only data available up to each row's timestamp.)*
20. §18.7 — 40% missing satisfaction score, suspected non-random — mean impute alone, or more? *(Add a missingness flag.)*
21. §19.6 — Why prefer capping over deleting an outlier row? *(Preserves the row's other feature values; avoids shrinking the dataset.)*
22. §20.5 — Does feature scaling change what LR can theoretically learn? *(False — only affects convergence speed, not the ceiling of learnable relationships.)*
23. §22.6 (Fairness) — coefficient=1,200, SE=900 — first hypothesis? *(Separation.)*
24. §22.6 — Equal approval rates but unequal recall among qualified candidates — which fairness property is violated? *(Equal Opportunity.)*
25. §23.4 — 3 correlated device features + unrelated features — L1, L2, or Elastic Net? *(Elastic Net.)*
26. §26.5 — Movie review scores 1-5 — softmax or ordinal, and why? *(Ordinal — preserves "how far off" information.)*
27. §29.7 — "Lifetime average rating" vs "last-30-second clicks" — batch or real-time for each? *(Batch; real-time.)*
28. §29.7 — Why does a feature store solve skew "by construction"? *(Only one feature definition exists — no second implementation to drift.)*
29. §30.4 — 6 months of hourly online updates, never a full retrain — biggest risk? *(Compounding drift/bias.)*
30. §30.4 — Why does rare-event bias connect to Separation? *(Shared root cause: too little data, specifically too few positive events, relative to how clean the apparent pattern looks.)*

---

*End of merged reference. Cross-references throughout (e.g. "§21," "Part D") point to sections within this same document — all three source conversations have been consolidated here with duplicate treatments (the Wald/LRT/GoF/Calibration material that appeared more than once, and the MLE/decision-boundary/calibration content that appeared across multiple source documents) merged into single, most-complete versions rather than repeated.*
