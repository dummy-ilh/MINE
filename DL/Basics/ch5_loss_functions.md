# Chapter 5: Loss Functions

---

### 5.1 The Plain-English Picture

The network makes a prediction. How wrong is it? That question needs a precise, numerical answer — one that gradient descent can act on. The loss function provides that answer.

Think of the loss function as the network's conscience. After every prediction, it computes a single number — the loss — that quantifies how badly the network failed. A perfect prediction produces zero loss. A catastrophically wrong prediction produces a large loss. Gradient descent then asks: "which direction should I nudge the weights to make this number smaller?" The entire training process is the iterative minimization of this one number.

The choice of loss function is not aesthetic — it is a modeling decision with deep statistical meaning. Every loss function encodes an assumption about the probability distribution of your data. Use the wrong loss function and you are literally optimizing for the wrong thing: you might minimize the loss and still produce a useless model.

Here is the hierarchy of decisions:

```
What kind of output do you need?
  │
  ├── A continuous number (regression)
  │     ├── Errors should be penalized quadratically → MSE
  │     ├── Outliers should not dominate         → MAE or Huber
  │     └── You need a probability density        → NLL with Gaussian
  │
  └── A class label (classification)
        ├── Two classes                            → Binary Cross-Entropy
        ├── Multiple classes, one label            → Categorical Cross-Entropy
        └── Multiple labels simultaneously         → Binary CE per label
```

---

### 5.2 Mean Squared Error (MSE)

The workhorse of regression. Simple, differentiable, and statistically principled.

```
MSE FORMULA
===========

  L(ŷ, y) = (1/N) Σᵢ (ŷᵢ - yᵢ)²

  For a single example:
  l(ŷ, y) = (ŷ - y)²

Where:
  N    = number of training examples
  ŷᵢ   = predicted value for example i  (scalar for regression)
  yᵢ   = true target value for example i
  (ŷ - y)² = squared error (always non-negative)

Gradient w.r.t. ŷ (single example):
  ∂l/∂ŷ = 2(ŷ - y)

  → If ŷ > y (overestimate): gradient is positive → push ŷ down
  → If ŷ < y (underestimate): gradient is negative → push ŷ up
  → If ŷ = y (perfect): gradient is zero → no update

Shape of MSE loss surface:
  l
  │    ╲         ╱
  │      ╲     ╱
  │        ╲ ╱
  │─────────────────── ŷ
              ↑
              y (true value, minimum of the bowl)

  Convex bowl → gradient always points toward minimum.
  Single global minimum at ŷ = y.
```

**Statistical grounding:**

MSE is equivalent to Maximum Likelihood Estimation (MLE) under the assumption that errors are Gaussian:

```
Assume: y = f(x; θ) + ε,  where ε ~ N(0, σ²)

The likelihood of observing y given x:
  P(y | x; θ) = (1/√(2πσ²)) exp(-(y - f(x;θ))² / 2σ²)

Maximizing log-likelihood:
  log P(y | x; θ) = -log(√(2πσ²)) - (y - ŷ)² / 2σ²

  Maximizing this is equivalent to minimizing (y - ŷ)² = MSE

So: MSE training = assuming Gaussian noise. If your noise
is NOT Gaussian (e.g., heavy-tailed), MSE is suboptimal.
```

**The outlier problem:**

```
True targets: [1, 2, 3, 4, 100]   ← 100 is an outlier
Predictions:  [1, 2, 3, 4,   5]   ← predict 5 for the outlier

MSE = ((1-1)² + (2-2)² + (3-3)² + (4-4)² + (5-100)²) / 5
    = (0 + 0 + 0 + 0 + 9025) / 5
    = 1805

One outlier dominates the entire loss.
The network will distort ALL its weights to reduce this one
massive error, degrading predictions for everything else.

Squared penalty: errors grow quadratically.
  Error of 1 → loss of 1
  Error of 2 → loss of 4   (2× error → 4× loss)
  Error of 10 → loss of 100 (10× error → 100× loss)
  Error of 100 → loss of 10,000
```

**Use MSE when:** targets are continuous, errors are roughly Gaussian, and outliers are genuine data points (not mislabeled noise).

---

### 5.3 Mean Absolute Error (MAE)

```
MAE FORMULA
===========

  L(ŷ, y) = (1/N) Σᵢ |ŷᵢ - yᵢ|

  For a single example:
  l(ŷ, y) = |ŷ - y|

Gradient w.r.t. ŷ:
  ∂l/∂ŷ = +1   if ŷ > y
           -1   if ŷ < y
           undefined at ŷ = y  (use 0 in practice)

Shape:
  l
  │    ╲       ╱
  │      ╲   ╱
  │        ╲╱         ← V-shape (not smooth at minimum!)
  │─────────────────── ŷ
              ↑
              y

Statistical grounding:
  MAE = MLE under Laplace distribution assumption.
  Laplace has heavier tails than Gaussian → more robust to outliers.

Same example as before:
  MAE = (|1-1| + |2-2| + |3-3| + |4-4| + |5-100|) / 5
      = (0 + 0 + 0 + 0 + 95) / 5
      = 19    ← much smaller than MSE's 1805

Outlier contributes linearly, not quadratically.
```

**MSE vs MAE tradeoff:**

```
┌──────────────┬───────────────────────┬──────────────────────────┐
│              │ MSE                   │ MAE                      │
├──────────────┼───────────────────────┼──────────────────────────┤
│ Outliers     │ Dominated by them     │ Robust to them           │
│ Gradient     │ Smooth, proportional  │ Constant magnitude       │
│ Near minimum │ Gradient → 0 (smooth) │ Gradient stays ±1 (noisy)│
│ Solution     │ Mean of targets       │ Median of targets        │
│ Optimization │ Easier (smooth)       │ Harder (non-smooth)      │
└──────────────┴───────────────────────┴──────────────────────────┘

MAE solution = median: if you minimize MAE, the optimal constant
prediction is the median, not the mean. This makes MAE naturally
robust because the median is not affected by extreme outliers.
```

---

### 5.4 Huber Loss

Huber loss is the best of both worlds: MAE's robustness for large errors, MSE's smooth gradients for small errors.

```
HUBER LOSS
==========

          ⎧ (1/2)(ŷ - y)²           if |ŷ - y| ≤ δ
  Lδ =   ⎨
          ⎩ δ · |ŷ - y| - (1/2)δ²  if |ŷ - y| > δ

Where:
  δ = threshold hyperparameter (typically 1.0)
  Below δ: quadratic (like MSE) → smooth gradient near minimum
  Above δ: linear  (like MAE)  → robust to outliers

Gradient:
  ∂Lδ/∂ŷ = (ŷ - y)         if |ŷ - y| ≤ δ
            δ · sign(ŷ - y)  if |ŷ - y| > δ

Shape (δ=1):
  l
  │       ╲           ╱     ← linear for large errors (MAE-like)
  │         ╲       ╱
  │           ╲___╱         ← quadratic near minimum (MSE-like)
  │────────────────────────── ŷ
                ↑
                y

Continuity check at |ŷ - y| = δ:
  Quadratic: (1/2)δ²
  Linear:    δ·δ - (1/2)δ² = (1/2)δ²  ✓ (continuous at boundary)

Use Huber when: you have real-valued targets with occasional
outliers. Common in reinforcement learning (DQN) and robust
regression problems.
```

---

### 5.5 Binary Cross-Entropy (BCE)

The standard loss for binary classification. Derived from information theory and maximum likelihood.

```
BINARY CROSS-ENTROPY
====================

  L(ŷ, y) = -(1/N) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

  For a single example:
  l(ŷ, y) = -[y log(ŷ) + (1-y) log(1-ŷ)]

Where:
  y  ∈ {0, 1}     true binary label
  ŷ  ∈ (0, 1)     predicted probability of class 1
                   (output of sigmoid activation)
  log = natural logarithm (base e)

Case y = 1 (true label is positive):
  l = -log(ŷ)
  → If ŷ → 1 (correct, confident): -log(1) = 0       ✓ no loss
  → If ŷ = 0.5 (uncertain):        -log(0.5) = 0.693
  → If ŷ → 0 (wrong, confident):   -log(0) → ∞       ✗ infinite loss!

Case y = 0 (true label is negative):
  l = -log(1-ŷ)
  → If ŷ → 0 (correct, confident): -log(1) = 0       ✓ no loss
  → If ŷ = 0.5 (uncertain):        -log(0.5) = 0.693
  → If ŷ → 1 (wrong, confident):   -log(0) → ∞       ✗ infinite loss!

Loss surface visualization (y=1):
  l
  ∞│╲
   │  ╲
  2│    ╲
   │      ╲
  1│        ╲
   │          ╲___
  0│               ───────────
   └──────────────────────────── ŷ
    0   0.2  0.4  0.6  0.8  1.0

  The loss is not symmetric. Being confidently wrong is
  penalized infinitely. Being confidently right costs nothing.
  This asymmetry is exactly what we want.
```

**Statistical grounding: this IS maximum likelihood.**

```
Assume: y | x ~ Bernoulli(p), where p = σ(wᵀx + b)

Likelihood of observing y:
  P(y | x; θ) = ŷʸ · (1-ŷ)^(1-y)

Log-likelihood:
  log P(y | x; θ) = y·log(ŷ) + (1-y)·log(1-ŷ)

Minimizing negative log-likelihood = minimizing BCE.
BCE training ≡ MLE for Bernoulli outputs. Not a heuristic.
```

**Gradient (the beautiful result):**

```
l = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
ŷ = σ(z) = 1/(1+e^(-z))

∂l/∂z = ŷ - y

That's it. The gradient of BCE w.r.t. the pre-activation z
is simply (prediction - truth). Clean, simple, stable.

Derivation:
  ∂l/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
  ∂ŷ/∂z = ŷ(1-ŷ)     [sigmoid derivative]
  ∂l/∂z = ∂l/∂ŷ · ∂ŷ/∂z
         = [-y/ŷ + (1-y)/(1-ŷ)] · ŷ(1-ŷ)
         = -y(1-ŷ) + (1-y)ŷ
         = -y + yŷ + ŷ - yŷ
         = ŷ - y   ∎
```

**Why not MSE for classification?**

```
Suppose y=1, ŷ=0.01 (very wrong prediction).
  MSE loss:  (0.01 - 1)² = 0.9801  and  ∂MSE/∂z ≈ -0.02 (tiny!)
  BCE loss:  -log(0.01) = 4.605    and  ∂BCE/∂z = 0.01-1 = -0.99 (large)

With MSE + sigmoid, the gradient is tiny when the network is most
wrong. This is because σ'(z) → 0 when z is very negative (ŷ ≈ 0),
and MSE depends on σ'(z) through the chain rule.
With BCE, the gradient is large when the network is most wrong.
BCE was designed to cancel the sigmoid saturation exactly.
```

---

### 5.6 Categorical Cross-Entropy (CCE)

The generalization of BCE to K > 2 classes. Used with softmax output.

```
CATEGORICAL CROSS-ENTROPY
==========================

  L(ŷ, y) = -(1/N) Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)

  For a single example with K classes:
  l(ŷ, y) = -Σₖ yₖ · log(ŷₖ)

Where:
  K          = number of classes
  yₖ ∈ {0,1} = one-hot label (1 for the true class, 0 elsewhere)
  ŷₖ ∈ (0,1) = predicted probability for class k (softmax output)
  Σₖ ŷₖ = 1  (softmax constraint)

Since y is one-hot, only one term survives in the sum:
  Let c = true class index (where yc = 1)
  l(ŷ, y) = -log(ŷc)   ← just the log-prob of the correct class!

This is the negative log-likelihood of the correct class.

Example:
  True class: c = 2  (y = [0, 0, 1])
  Predictions: ŷ = [0.1, 0.2, 0.7]

  l = -log(ŷ₂) = -log(0.7) = 0.357   ← low loss, mostly correct

  If instead: ŷ = [0.6, 0.3, 0.1]   (network confident about wrong class)
  l = -log(0.1) = 2.303              ← high loss, confidently wrong
```

**Numerical stability (critical):**

```
Combining softmax + cross-entropy naively:
  ŷₖ = e^(zₖ) / Σⱼ e^(zⱼ)
  l = -log(ŷc) = -log(e^(zc) / Σⱼ e^(zⱼ))
               = -zc + log(Σⱼ e^(zⱼ))
               = log-sum-exp trick

In PyTorch:
  WRONG:  criterion = nn.CrossEntropyLoss()
          output = softmax(logits)       ← don't apply softmax first!
          loss = criterion(output, y)    ← applies log(softmax(x))
                                            to already-softmaxed values

  RIGHT:  criterion = nn.CrossEntropyLoss()  ← applies softmax internally
          loss = criterion(logits, y)         ← feed RAW LOGITS

nn.CrossEntropyLoss in PyTorch = Softmax + NLLLoss combined,
using the numerically stable log-sum-exp formulation.
Double-applying softmax is one of the most common DL bugs.
```

---

### 5.7 KL Divergence and Its Relation to Cross-Entropy

```
KL DIVERGENCE
=============

  KL(P || Q) = Σₖ P(k) · log(P(k) / Q(k))
             = Σₖ P(k) · log(P(k)) - Σₖ P(k) · log(Q(k))
             = -H(P)  +  H(P, Q)
               ↑           ↑
           negative      cross-entropy
           entropy       of Q relative to P
           of P

Where:
  P = true distribution (labels y)
  Q = predicted distribution (ŷ)
  H(P)    = entropy of P (constant w.r.t. model parameters)
  H(P, Q) = cross-entropy of P and Q

Since H(P) is constant during training:
  minimizing KL(P||Q) ≡ minimizing H(P, Q) ≡ minimizing CCE

Cross-entropy training IS minimizing KL divergence between
the true and predicted distributions. Information theory and
MLE are the same thing here.

Properties:
  KL(P||Q) ≥ 0         always non-negative (Gibbs inequality)
  KL(P||Q) = 0 iff P = Q   zero only when distributions match
  KL(P||Q) ≠ KL(Q||P)  not symmetric (not a true distance)
```

---

### 5.8 Worked Numerical Example: Computing and Comparing Losses

```
SCENARIO
========
Binary classification: spam detection
Batch of 4 emails

True labels:     y  = [1,    0,    1,    0   ]
Predictions:     ŷ  = [0.9,  0.2,  0.4,  0.8 ]
                       ↑     ↑     ↑     ↑
                      good  good  poor  poor

═══════════════════════════════════════════════════════════════
BINARY CROSS-ENTROPY
═══════════════════════════════════════════════════════════════

l₁ = -[1·log(0.9) + 0·log(0.1)] = -log(0.9)  = -(-0.105) = 0.105
l₂ = -[0·log(0.2) + 1·log(0.8)] = -log(0.8)  = -(-0.223) = 0.223
l₃ = -[1·log(0.4) + 0·log(0.6)] = -log(0.4)  = -(-0.916) = 0.916
l₄ = -[0·log(0.8) + 1·log(0.2)] = -log(0.2)  = -(-1.609) = 1.609

BCE = (0.105 + 0.223 + 0.916 + 1.609) / 4
    = 2.853 / 4
    = 0.713

Observation:
  Examples 3 and 4 are wrong predictions (3: predicted 0.4 for spam,
  4: predicted 0.8 for non-spam). Their losses (0.916, 1.609) dominate
  the total, as they should.

═══════════════════════════════════════════════════════════════
MSE (for comparison — WRONG choice for classification)
═══════════════════════════════════════════════════════════════

l₁ = (0.9 - 1)² = 0.01
l₂ = (0.2 - 0)² = 0.04
l₃ = (0.4 - 1)² = 0.36
l₄ = (0.8 - 0)² = 0.64

MSE = (0.01 + 0.04 + 0.36 + 0.64) / 4 = 1.05 / 4 = 0.2625

Gradient comparison for example 4 (ŷ=0.8, y=0, most wrong example):
  BCE gradient: ∂l/∂z = ŷ - y = 0.8 - 0 = 0.8   (large → fast learning)
  MSE gradient: ∂l/∂z = (ŷ-y)·ŷ·(1-ŷ)           (through sigmoid)
                       = (0.8)·(0.8)·(0.2) = 0.128 (6× smaller)

BCE pushes the weights 6× harder for this wrong prediction.
This is why BCE is correct for classification, not MSE.

═══════════════════════════════════════════════════════════════
MULTI-CLASS EXAMPLE: 3-class classification
═══════════════════════════════════════════════════════════════

True labels (one-hot):
  y⁽¹⁾ = [1, 0, 0]   (class 0)
  y⁽²⁾ = [0, 1, 0]   (class 1)
  y⁽³⁾ = [0, 0, 1]   (class 2)

Softmax predictions:
  ŷ⁽¹⁾ = [0.7, 0.2, 0.1]   → correct class prob: 0.7
  ŷ⁽²⁾ = [0.1, 0.6, 0.3]   → correct class prob: 0.6
  ŷ⁽³⁾ = [0.3, 0.5, 0.2]   → correct class prob: 0.2  ← wrong!

Cross-entropy per example:
  l⁽¹⁾ = -log(0.7) = 0.357
  l⁽²⁾ = -log(0.6) = 0.511
  l⁽³⁾ = -log(0.2) = 1.609   ← highest loss, wrongest prediction ✓

Average CCE = (0.357 + 0.511 + 1.609) / 3 = 0.826

Interpretation: a perfect model would have CCE = 0 (each correct
class gets probability 1.0, log(1.0) = 0). A random baseline with
3 classes would have CCE = -log(1/3) = 1.099. Our model (0.826)
is better than random but has room to improve on example 3.
```

---

### 5.9 Loss Functions for Special Cases

```
LOSS FUNCTION SELECTION GUIDE
==============================

Task                    │ Output Layer    │ Loss Function
────────────────────────┼─────────────────┼──────────────────────────
Regression              │ Linear (no act) │ MSE / Huber
Regression (robust)     │ Linear          │ MAE / Huber
Binary classification   │ Sigmoid         │ Binary Cross-Entropy
Multi-class (one label) │ Softmax         │ Categorical Cross-Entropy
Multi-label (K labels)  │ K Sigmoids      │ K binary CEs summed
Ranking / metric learn  │ Embedding       │ Triplet / Contrastive
Sequence generation     │ Softmax/step    │ CCE per token (summed)
Object detection        │ Mixed           │ CCE + regression losses
Generative (VAE)        │ Mixed           │ Reconstruction + KL
Reinforcement learning  │ Linear          │ Huber (TD error)


ADDITIONAL LOSS FUNCTIONS (brief):

Contrastive Loss (metric learning):
  L = (1/2N) Σ [y·d² + (1-y)·max(0, margin-d)²]
  Where d = distance between embeddings, y = 1 if same class.
  Pulls same-class embeddings together, pushes different apart.

Triplet Loss (FaceNet, 2015):
  L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + α)
  Where a=anchor, p=positive (same class), n=negative (different).
  α = margin (how much separation to enforce).

Focal Loss (RetinaNet, 2017 — for class imbalance):
  FL(ŷ, y) = -α(1-ŷ)^γ · log(ŷ)    when y=1
  The (1-ŷ)^γ factor down-weights easy examples.
  Hard examples (wrong predictions) dominate training.
  γ=2 is standard. Critical for object detection where
  background >> objects (extreme class imbalance).
```

---

### 5.10 Why This Matters — What Breaks If You Get This Wrong

1. **Using MSE for classification.** Your model will technically train — loss decreases — but more slowly and often to a worse optimum. The sigmoid saturation problem means gradients are tiny when the network is most wrong. In extreme cases (confident wrong predictions), the network barely updates. Use BCE.

2. **Applying softmax before `nn.CrossEntropyLoss`** in PyTorch. This is the most common PyTorch bug. `CrossEntropyLoss` applies log(softmax) internally. If you pre-apply softmax, you're feeding probabilities through softmax again — getting a different distribution with a different scale — and computing log of that. Your loss is numerically wrong and the model trains on incorrect gradients. The loss will be in an unexpected range and training will be slow or unstable.

3. **Using CE loss with hard labels for knowledge distillation.** Knowledge distillation requires soft labels (probability distributions, not one-hot). If you mistakenly use hard one-hot labels when the teacher produces soft probabilities, you throw away the inter-class similarity information that makes distillation work.

4. **Not accounting for class imbalance in the loss.** If your dataset is 99% class 0 and 1% class 1, a model that always predicts class 0 achieves 99% accuracy but is useless. BCE will still train, but the model will ignore the minority class. Solutions: weighted BCE (weight minority class by 99, majority by 1), or Focal Loss. Ignoring this produces a model that "works" on your training metric but fails in production.

5. **Using a bounded loss (BCE) for regression.** BCE assumes outputs are in (0,1) and labels are 0 or 1. If you accidentally apply it to regression targets (say, house prices), the gradients are meaningless and training diverges immediately.

---

### 5.11 Google/Apple-Level Interview Q&A

---

**Q1: "Cross-entropy loss is derived from maximum likelihood estimation. Walk me through the derivation from Bernoulli likelihood to binary cross-entropy. Then explain what assumption about the data distribution you are making when you use MSE instead, and when that assumption breaks."**

*Why this is asked:* This question separates engineers who use loss functions as black boxes from those who understand them as principled statistical estimators. At Google and Apple, ML models are deployed at scale — choosing the wrong loss function has real consequences. Understanding the MLE derivation proves you know *why* cross-entropy exists, not just how to call it.

**Answer:**

**Derivation of BCE from Bernoulli MLE:**

We model binary classification as: y | x ~ Bernoulli(p), where p = f(x; θ) is the network output (after sigmoid).

The probability of observing a single label y given input x:

```
P(y | x; θ) = p^y · (1-p)^(1-y)

  When y=1: P = p¹ · (1-p)⁰ = p       ← prob of positive class
  When y=0: P = p⁰ · (1-p)¹ = (1-p)   ← prob of negative class

For N i.i.d. examples, the joint likelihood:
  L(θ) = Πᵢ P(yᵢ | xᵢ; θ)
        = Πᵢ ŷᵢ^yᵢ · (1-ŷᵢ)^(1-yᵢ)

Log-likelihood (products → sums, easier to optimize):
  log L(θ) = Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

Maximizing log-likelihood = minimizing negative log-likelihood:
  NLL = -log L(θ) = -Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]

Dividing by N (to get per-example average):
  BCE = -(1/N) Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]   ∎
```

**Assumption underlying MSE:**

MSE arises from MLE under a Gaussian noise model:

```
Assume: y = f(x; θ) + ε,  ε ~ N(0, σ²)

P(y | x; θ) = (1/√2πσ²) · exp(-(y - f(x;θ))² / 2σ²)

NLL = (N/2)·log(2πσ²) + (1/2σ²)·Σᵢ(yᵢ - ŷᵢ)²

Minimizing NLL ≡ minimizing Σ(yᵢ - ŷᵢ)² = N · MSE   ∎
```

**When the Gaussian assumption breaks:**

```
1. Binary outputs (y ∈ {0,1}): Gaussian assumes y is continuous
   and unbounded. Predicting probabilities with MSE can produce
   values outside [0,1], and the gradient is poor near saturation.

2. Count data (y ∈ {0,1,2,...}): Poisson distribution is more
   appropriate → use Poisson NLL loss.

3. Heavy-tailed noise: Real-world errors often have outliers
   (measurement errors, mislabeling). Gaussian assigns tiny
   probability to large errors, so MSE massively penalizes them.
   → Use Laplace (MAE) or Student-t (Huber) assumptions.

4. Multimodal targets: If for a given x, y could reasonably be
   1.0 OR 5.0 (two valid answers), a Gaussian model will predict
   the mean (3.0) which is wrong for both. Need mixture models.

5. Skewed distributions (e.g., income, file sizes): Log-normal
   might be appropriate → model log(y) with MSE instead of y.
```

---

**Q2: "Your model is trained with cross-entropy loss and achieves 95% accuracy on the test set. A product manager says the model is ready to ship. You disagree. What could be wrong, and what metrics would you compute to make your case?"**

*Why this is asked:* This tests real-world ML judgment. Accuracy is almost always the wrong primary metric. Google and Apple ship products used by billions of people — a model that's wrong on a specific demographic, confidently wrong, or catastrophically wrong on edge cases causes real harm. This question probes whether the candidate thinks beyond the training objective.

**Answer:**

**95% accuracy can hide severe problems:**

```
Problem 1: Class Imbalance
  Dataset: 95% negative, 5% positive
  A model that ALWAYS predicts negative: 95% accuracy.
  This model has learned nothing.

  Better metrics:
    Precision = TP / (TP + FP)   — of predicted positives, how many right?
    Recall    = TP / (TP + FN)   — of actual positives, how many found?
    F1        = 2·P·R / (P+R)    — harmonic mean
    AUC-ROC   — ranking quality across all thresholds

Problem 2: Miscalibration
  The model is accurate (predicts correct class) but its probabilities
  are wrong. A model that says "99% confident" should be right 99%
  of the time. If it's only right 60% of the time when saying 99%,
  it's miscalibrated.

  Metric: Expected Calibration Error (ECE)
    Bin predictions by confidence [0-10%, 10-20%, ..., 90-100%]
    ECE = Σ_bins (|bin| / N) · |accuracy_in_bin - avg_confidence_in_bin|

Problem 3: Subgroup Disparities
  Overall accuracy 95%, but:
    Accuracy on Group A (80% of data): 98%
    Accuracy on Group B (20% of data): 75%
  
  A model shipped like this causes disparate impact.
  Metrics: per-subgroup accuracy, fairness metrics (equalized odds)

Problem 4: Catastrophic Failure Mode
  99% of examples: easy, 5% loss
  1% of examples: edge cases, model gives completely wrong answer
  Average loss looks fine, but shipping causes occasional disasters.
  
  Metric: tail loss (95th/99th percentile loss), worst-group accuracy

Problem 5: Distribution Shift
  Test set = held-out training distribution.
  Real users may be systematically different (different demographics,
  devices, time of day, etc.).
  
  Metric: test on multiple OOD (out-of-distribution) slices

What I would present to the PM:
  1. Confusion matrix (breakdown of error types)
  2. Precision/Recall curve and AUC
  3. Calibration curve (reliability diagram)
  4. Per-subgroup accuracy breakdown
  5. Failure case analysis (what do the wrong examples look like?)
  6. Comparison to a naive baseline (what does "always predict majority class" get?)
```

---

**Q3: "Explain why the cross-entropy loss function is convex with respect to the output probabilities ŷ, but non-convex with respect to the network weights θ. What are the practical implications of this distinction?"**

*Why this is asked:* Convexity is one of the most important properties in optimization. This question tests whether a candidate understands where the difficulty in training neural networks actually lies — and why we can't simply find the global optimum. It's a subtle question that reveals deep understanding of the optimization landscape.

**Answer:**

**Convexity w.r.t. ŷ (output probabilities):**

```
BCE loss as a function of ŷ (treating ŷ as the variable):
  l(ŷ) = -y·log(ŷ) - (1-y)·log(1-ŷ)

Second derivative:
  ∂l/∂ŷ  = -y/ŷ + (1-y)/(1-ŷ)
  ∂²l/∂ŷ² = y/ŷ² + (1-y)/(1-ŷ)² > 0   for all ŷ ∈ (0,1)

Positive second derivative everywhere → convex in ŷ.
The loss surface as a function of the output is a convex bowl.
If ŷ were a direct parameter, gradient descent would find the
global minimum (ŷ = y) guaranteed.

Similarly, CCE is convex in the softmax outputs ŷ.
Log is concave, negative log is convex, sum of convex = convex.
```

**Non-convexity w.r.t. θ (network weights):**

```
ŷ = f(x; θ) is a highly non-linear function of θ.
l(θ) = BCE(f(x; θ), y) = composition of convex BCE with
        non-linear f(·)

The composition of a convex function with a non-linear
function is generally NOT convex.

Concrete example (1 neuron, sigmoid):
  ŷ = σ(w·x + b)
  l(w) = -y·log(σ(wx+b)) - (1-y)·log(1-σ(wx+b))

  For a single neuron this is convex in w.
  For two neurons in sequence: l(W², W¹) is non-convex
  because ŷ = σ(W²·σ(W¹·x)) creates multiple local minima.

Loss landscape of a deep network:
  - Many local minima
  - Many saddle points (gradient=0, not a minimum)
  - Very few true global minima
  - Loss "valleys" (flat regions where gradient ≈ 0)
```

**Practical implications:**

```
1. No convergence guarantee.
   Gradient descent on a convex function always finds the global
   minimum (with appropriate learning rate). On a non-convex loss,
   it finds A minimum — which might be local, not global.

2. Initialization matters enormously.
   Different starting points θ⁰ can lead to very different minima.
   Random initialization + multiple restarts is often needed.
   Bad initialization (e.g., all zeros) can get stuck immediately.

3. Saddle points are the main problem (not local minima).
   Empirical research (Dauphin et al., 2014; Goodfellow et al., 2015)
   shows that in high dimensions, local minima are rare — most
   critical points are saddle points. Gradient descent escapes
   saddle points slowly (gradient is near zero there).
   Momentum and Adam help escape saddle points faster.

4. Overparameterization helps.
   Surprisingly: very large networks (more parameters than data)
   tend to find better minima than small networks. The loss
   landscape of overparameterized networks has more "valleys"
   connecting good solutions, making optimization easier.
   This is counter-intuitive but well-supported empirically.

5. The convexity of BCE in ŷ gives us the correct gradient signal
   once we've determined which direction to move the output.
   The non-convexity is purely in how we map θ → ŷ, which is
   the neural network's expressiveness — the cost of the power.
```

---

*End of Chapter 5. Chapter 6 (Backpropagation & Gradient Descent) coming next.*
