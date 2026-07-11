# Chapter 3: Activation Functions

---

### 3.1 The Plain-English Picture

A neuron computes a weighted sum of its inputs. That sum is a linear function. Stack a thousand linear functions and you still have a linear function — the entire network collapses to one matrix multiply (proved in Chapter 2). Activation functions are the cure: a non-linear transformation applied after each weighted sum, injecting the non-linearity that allows deep networks to represent complex patterns.

But not all non-linearities are created equal. The choice of activation function is one of the most consequential design decisions in a neural network. Use the wrong one and your network won't train at all. Use a mediocre one and it trains slowly. Use the right one and the same architecture converges faster, deeper, and better.

Think of the activation function as each neuron's "personality." The linear combination `z = wᵀx + b` is the evidence the neuron has collected. The activation function decides how the neuron *responds* to that evidence — does it fire proportionally? Abruptly? Only when evidence is strong? Only positively?

We'll cover six activation functions in this chapter. Each one was invented to solve a specific failure mode of its predecessor.

---

### 3.2 The Sigmoid Function

The original activation function — borrowed directly from logistic regression and neuroscience (the firing-rate model of biological neurons).

```
SIGMOID
=======

  σ(z) = 1 / (1 + e^(-z))

  Output range: (0, 1)
  Derivative:   σ'(z) = σ(z) · (1 - σ(z))

  Shape:
          1.0 ┤                  ╭──────────
          0.9 ┤               ╭──╯
          0.8 ┤            ╭──╯
          0.7 ┤          ╭─╯
          0.6 ┤        ╭─╯
          0.5 ┤      ──╯
          0.4 ┤    ╭─╯
          0.3 ┤  ╭─╯
          0.2 ┤╭─╯
          0.1 ┤╯
          0.0 ┤──────────╯
              └──────────────────────────────
             -6  -4  -2   0   2   4   6    z

  Derivative (maximum at z=0, value=0.25):
          0.25┤          ╭──╮
          0.20┤        ╭─╯  ╲─╮
          0.15┤      ╭─╯      ╲─╮
          0.10┤   ╭──╯          ╲──╮
          0.05┤╭──╯                ╲──╮
          0.00┤                        ╲────
              └──────────────────────────────
             -6  -4  -2   0   2   4   6    z
```

**Key equations:**

```
Forward:   σ(z) = 1 / (1 + e^(-z))

Derivative: dσ/dz = σ(z) · (1 - σ(z))

  Proof:
    Let f = 1 + e^(-z)
    σ = f^(-1)
    dσ/dz = -f^(-2) · (-e^(-z))
           = e^(-z) / (1 + e^(-z))²
           = [1/(1+e^(-z))] · [e^(-z)/(1+e^(-z))]
           = σ(z) · (1 - σ(z))    ∎

Where:
  z     = pre-activation (wᵀx + b)
  σ(z)  = output activation, interpreted as probability
  σ'(z) = gradient, used in backpropagation
```

**Strengths:**
- Output is a valid probability (bounded 0–1) → good for output layer of binary classifiers
- Smooth everywhere → differentiable everywhere → gradients always exist
- Interpretable: output = P(class=1 | input)

**Fatal weaknesses:**

1. **Vanishing gradients.** The maximum derivative is 0.25 (at z=0). For |z| > 4, the derivative is nearly zero. During backpropagation, gradients are multiplied through layers. In a 10-layer network: 0.25^10 ≈ 0.000001. Gradients vanish exponentially with depth. Early layers learn nothing.

2. **Not zero-centered.** Sigmoid outputs are always positive (0 to 1). This means gradients w.r.t. weights in the previous layer are always the same sign. This causes zig-zag updates — the optimizer can only move in all-positive or all-negative directions per step, slowing convergence.

3. **Expensive.** Computing `e^(-z)` is costly compared to ReLU.

**Use sigmoid:** Only at the output layer of binary classifiers. Never in hidden layers of deep networks.

---

### 3.3 The Tanh Function

Tanh was introduced to fix sigmoid's zero-centering problem.

```
TANH
====

  tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
           = 2σ(2z) - 1      ← just a rescaled sigmoid!

  Output range: (-1, 1)
  Derivative:   tanh'(z) = 1 - tanh²(z)

  Shape:
         1.0 ┤                  ╭──────────
         0.5 ┤           ╭──────╯
         0.0 ┤      ─────╯
        -0.5 ┤ ─────╮
        -1.0 ┤──────╯
             └──────────────────────────────
            -4  -3  -2  -1   0   1   2   3   4    z

  Derivative (maximum at z=0, value=1.0):
         1.0 ┤          ╭──╮
         0.8 ┤        ╭─╯  ╲─╮
         0.6 ┤      ╭─╯      ╲─╮
         0.4 ┤   ╭──╯          ╲──╮
         0.2 ┤╭──╯                ╲──╮
         0.0 ┤                        ╲────
             └──────────────────────────────
```

**Key equations:**

```
Forward:    tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

Derivative: d(tanh)/dz = 1 - tanh²(z)

  Maximum gradient = 1.0 at z = 0  (4× larger than sigmoid's 0.25)
  Still saturates to 0 for large |z|
```

**Improvements over sigmoid:**
- Zero-centered: outputs range (-1, 1), mean ≈ 0 → no zig-zag gradient problem
- Stronger gradient (max 1.0 vs. 0.25) → less severe vanishing gradient

**Remaining weakness:** Still saturates. For |z| > 2, gradient approaches zero. Still vanishes in deep networks, just more slowly.

**Use tanh:** Hidden layers of RNNs and LSTMs (Chapter 12-13) where zero-centered outputs matter for sequence dynamics. Rarely used in feedforward networks today — ReLU is almost always better.

---

### 3.4 ReLU: The Activation That Changed Everything

ReLU (Rectified Linear Unit) was popularized by Nair & Hinton (2010) and Krizhevsky et al. (AlexNet, 2012). It is the default activation function for hidden layers in almost every modern network.

```
RELU
====

  ReLU(z) = max(0, z)

  Output range: [0, ∞)
  Derivative:   ReLU'(z) = 1 if z > 0, else 0

  Shape:
          6 ┤                        ╱
          5 ┤                      ╱
          4 ┤                    ╱
          3 ┤                  ╱
          2 ┤                ╱
          1 ┤              ╱
          0 ┤────────────╱
            └──────────────────────────────
           -6  -4  -2   0   2   4   6    z

  Derivative:
          1 ┤              ┌───────────────
            ┤              │
          0 ┤──────────────┘
            └──────────────────────────────
           -6  -4  -2   0   2   4   6    z
```

**Key equations:**

```
Forward:    ReLU(z) = max(0, z)
                    = z   if z > 0
                    = 0   if z ≤ 0

Derivative: dReLU/dz = 1   if z > 0
                     = 0   if z < 0
                     = undefined at z = 0  (use 0 in practice)
```

**Why ReLU won:**

1. **No vanishing gradient for positive inputs.** Gradient is exactly 1 for z > 0. Signal passes through unchanged. A 100-layer network with ReLU can backpropagate gradients without exponential decay (for the active neurons).

2. **Sparse activation.** Negative inputs produce exactly 0. In a typical network, ~50% of neurons are inactive (output 0) at any given input. This sparsity is computationally efficient and acts as an implicit regularizer.

3. **Computationally trivial.** `max(0, z)` is a comparison and a threshold. No exponentiation. ~6× faster than sigmoid per operation.

4. **Linear for positive inputs.** Gradient is constant (1) — no saturation. Optimization is easier.

**The Dying ReLU problem:**

If a neuron's pre-activation z is consistently negative for all training examples, its gradient is permanently 0. The weight updates are zero. The neuron never recovers. It is "dead." This can happen when:
- Learning rate is too large (weights get large negative values)
- Weights are initialized poorly

In large networks, a significant fraction of neurons can die, reducing effective capacity.

---

### 3.5 Leaky ReLU and Parametric ReLU

Invented to fix the dying ReLU problem.

```
LEAKY RELU
==========

  LeakyReLU(z) = z         if z > 0
               = αz        if z ≤ 0   (α is small, e.g. 0.01)

  Derivative:  1    if z > 0
               α    if z ≤ 0   ← never exactly zero!

  Shape (α = 0.1 for visibility):
          4 ┤                    ╱
          2 ┤                  ╱
          0 ┤────────────────╱
         -0.3┤      ╱
         -0.6┤    ╱
             └──────────────────────────────
            -6  -4  -2   0   2   4   6    z

PARAMETRIC RELU (PReLU):
  Same as Leaky ReLU but α is a LEARNED parameter.
  Each neuron can learn its own α.
  Introduced by He et al. (2015).
```

**Key equation:**

```
LeakyReLU(z; α) = max(αz, z)

Where:
  α = leak coefficient (hyperparameter for LeakyReLU,
      learned parameter for PReLU)
  Typical α = 0.01 for LeakyReLU

Derivative:
  d/dz = 1   if z > 0
         α   if z ≤ 0   (small but nonzero — neuron never fully dies)
```

---

### 3.6 ELU: Exponential Linear Unit

```
ELU
===

  ELU(z; α) = z              if z > 0
            = α(e^z - 1)     if z ≤ 0

  Derivative: 1               if z > 0
              ELU(z) + α      if z ≤ 0

  Shape (α=1):
          4 ┤                    ╱
          2 ┤                  ╱
          0 ┤────────────────╱
         -0.5┤    ─────────╮
         -0.9┤ ────────────╯  (saturates at -α = -1)
             └──────────────────────────────

Key property: negative outputs are nonzero (unlike ReLU) and
smooth (unlike Leaky ReLU). Mean activation closer to zero.
Slower than ReLU due to exp() for negative inputs.
```

---

### 3.7 GELU: Gaussian Error Linear Unit

Used in BERT, GPT, and most modern transformers. The current state-of-the-art for language models.

```
GELU
====

  GELU(z) = z · Φ(z)

  Where Φ(z) is the CDF of the standard normal distribution:
  Φ(z) = P(X ≤ z) for X ~ N(0,1)

  Approximation (used in practice, faster):
  GELU(z) ≈ 0.5z · (1 + tanh[√(2/π) · (z + 0.044715z³)])

  Shape (similar to ReLU but smooth near zero):
          4 ┤                      ╱
          2 ┤                   ╱
          0 ┤──────────────╱╱
        -0.2┤        ╭─────
             └──────────────────────────────

  Key property: smooth everywhere (no kink at z=0 like ReLU).
  Slight negative dip near z=-0.5 before returning to 0.
  Empirically outperforms ReLU on language tasks.
```

**Key equation:**

```
GELU(z) = z · Φ(z)

Intuition: instead of hard-gating (ReLU: fire or don't fire),
GELU stochastically gates based on the probability that z is
positive under a Gaussian prior. The gate weight Φ(z) is:
  - Near 0 for very negative z  (neuron mostly silent)
  - Near 0.5 for z = 0          (neuron half-active)
  - Near 1 for very positive z  (neuron fully active)
  - Smooth everywhere            (better gradient flow)
```

---

### 3.8 Softmax: The Output Layer for Multi-Class Classification

Not used in hidden layers. Used exclusively at the output layer when you need a probability distribution over K classes.

```
SOFTMAX
=======

  Softmax(zₖ) = e^(zₖ) / Σⱼ e^(zⱼ)    for k = 1, ..., K

  Input:  z = [z₁, z₂, ..., z_K]   (raw scores/"logits")
  Output: p = [p₁, p₂, ..., p_K]   where pₖ = e^(zₖ) / Σⱼ e^(zⱼ)

  Properties:
    1. pₖ ∈ (0, 1) for all k       (valid probabilities)
    2. Σₖ pₖ = 1                    (sums to 1 — a distribution)
    3. Differentiable everywhere    (enables backpropagation)

  Example: 3-class classification
    z = [2.0, 1.0, 0.5]   (logits from final layer)

    e^z = [e^2.0, e^1.0, e^0.5]
        = [7.389, 2.718, 1.649]

    Σ = 7.389 + 2.718 + 1.649 = 11.756

    p = [7.389/11.756, 2.718/11.756, 1.649/11.756]
      = [0.629, 0.231, 0.140]

    Predict class 0 (highest probability: 62.9%).
    This is a valid probability distribution: 0.629+0.231+0.140 = 1.0 ✓
```

**Numerical stability trick (critical in practice):**

```
Naive softmax overflows for large z values:
  e^1000 = overflow in float32

Stable softmax: subtract the maximum before exponentiating
  Softmax(z)ₖ = e^(zₖ - max(z)) / Σⱼ e^(zⱼ - max(z))

This is mathematically identical (the max cancels in numerator and
denominator) but never overflows since the largest exponent is e^0 = 1.

Always implement stable softmax. Always.
```

---

### 3.9 Activation Function Comparison Table

```
┌─────────────┬───────────┬───────────┬──────────────┬──────────────────┐
│ Function    │ Range     │ Vanishing │ Zero-Centered│ Use Case         │
│             │           │ Gradient  │              │                  │
├─────────────┼───────────┼───────────┼──────────────┼──────────────────┤
│ Sigmoid     │ (0, 1)    │ Severe    │ No           │ Binary output    │
│ Tanh        │ (-1, 1)   │ Moderate  │ Yes          │ RNN hidden layer │
│ ReLU        │ [0, ∞)    │ None*     │ No           │ CNN/MLP hidden   │
│ Leaky ReLU  │ (-∞, ∞)   │ None      │ No           │ When ReLU dies   │
│ ELU         │ (-α, ∞)   │ None      │ Approx.      │ Deeper networks  │
│ GELU        │ (-0.17,∞) │ None      │ Approx.      │ Transformers/NLP │
│ Softmax     │ (0, 1)    │ N/A       │ N/A          │ Multi-class out  │
└─────────────┴───────────┴───────────┴──────────────┴──────────────────┘
* ReLU has no vanishing gradient for positive z, but dying ReLU for z<0
```

---

### 3.10 Worked Numerical Example: Forward Pass with Multiple Activations

```
NETWORK: 3 → 3 → 2
Hidden layer: ReLU
Output layer: Softmax

Input: x = [1.0, -0.5, 2.0]

Layer 1 weights and biases:
  W¹ = [[ 0.3, -0.2,  0.5],
         [-0.1,  0.4,  0.2],
         [ 0.6,  0.1, -0.3]]
  b¹ = [0.1, -0.1, 0.2]

STEP 1: Compute z¹ = W¹x + b¹
=====================================
  z¹₁ = (0.3)(1.0) + (-0.2)(-0.5) + (0.5)(2.0) + 0.1
       = 0.30 + 0.10 + 1.00 + 0.10
       = 1.50

  z¹₂ = (-0.1)(1.0) + (0.4)(-0.5) + (0.2)(2.0) + (-0.1)
       = -0.10 + (-0.20) + 0.40 + (-0.10)
       = 0.00

  z¹₃ = (0.6)(1.0) + (0.1)(-0.5) + (-0.3)(2.0) + 0.2
       = 0.60 + (-0.05) + (-0.60) + 0.20
       = 0.15

  z¹ = [1.50, 0.00, 0.15]

STEP 2: Apply ReLU → a¹ = max(0, z¹)
=====================================
  a¹₁ = max(0, 1.50) = 1.50   ← positive, passes through
  a¹₂ = max(0, 0.00) = 0.00   ← exactly zero (edge case)
  a¹₃ = max(0, 0.15) = 0.15   ← positive, passes through

  a¹ = [1.50, 0.00, 0.15]

  Note: neuron 2 outputs 0. If this happens consistently
  across all training examples, this neuron may be dying.

Layer 2 weights and biases:
  W² = [[0.4, 0.7, -0.2],
         [0.1, -0.5, 0.8]]
  b² = [0.05, -0.05]

STEP 3: Compute z² = W²a¹ + b²
=====================================
  z²₁ = (0.4)(1.50) + (0.7)(0.00) + (-0.2)(0.15) + 0.05
       = 0.60 + 0.00 + (-0.03) + 0.05
       = 0.62

  z²₂ = (0.1)(1.50) + (-0.5)(0.00) + (0.8)(0.15) + (-0.05)
       = 0.15 + 0.00 + 0.12 + (-0.05)
       = 0.22

  z² = [0.62, 0.22]   ← these are the logits

STEP 4: Apply Softmax → ŷ
=====================================
  e^0.62 = 1.859
  e^0.22 = 1.246
  Σ = 1.859 + 1.246 = 3.105

  ŷ₁ = 1.859 / 3.105 = 0.599
  ŷ₂ = 1.246 / 3.105 = 0.401

  ŷ = [0.599, 0.401]

INTERPRETATION:
  The network predicts class 0 with 59.9% probability
  and class 1 with 40.1% probability.
  Prediction: class 0.
  Check: 0.599 + 0.401 = 1.000 ✓
```

---

### 3.11 The Vanishing Gradient: Quantified

This is important enough to demonstrate numerically.

```
VANISHING GRADIENT IN A 5-LAYER SIGMOID NETWORK
================================================

During backpropagation, gradients are multiplied through layers.
The gradient at layer l depends on the product of derivatives
at all layers above it.

Sigmoid derivative at z=0 (best case): σ'(0) = 0.25

In a 5-layer network, gradient reaching layer 1:
  ∂L/∂W¹ ∝ σ'(z⁵) · σ'(z⁴) · σ'(z³) · σ'(z²) · σ'(z¹)

Best case (all z = 0):
  0.25 × 0.25 × 0.25 × 0.25 × 0.25 = 0.25⁵ = 0.000977

Typical case (z values spread, most derivatives ≈ 0.1):
  0.1⁵ = 0.00001

In a 10-layer network (typical case):
  0.1¹⁰ = 10⁻¹⁰

Gradients at early layers become so small that weights barely move.
Early layers learn astronomically slowly compared to late layers.
This is why deep sigmoid networks failed before ReLU.

WITH RELU (positive z):
  ReLU'(z) = 1 for z > 0

  5-layer network, all active neurons:
  1 × 1 × 1 × 1 × 1 = 1.0

  No decay. Gradient reaches layer 1 at full strength.
  This is why ReLU enabled deep networks.
```

---

### 3.12 Why This Matters — What Breaks If You Get This Wrong

1. **Using sigmoid in deep hidden layers.** Training stalls. Loss decreases for the first few epochs then flatlines. Early layers show near-zero gradient norms. The fix is mechanical: replace sigmoid with ReLU. This mistake was responsible for 10+ years of "neural networks don't scale deep" consensus before ReLU became standard.

2. **Using ReLU in the output layer for classification.** ReLU outputs are unbounded and non-probabilistic. You need sigmoid (binary) or softmax (multi-class) at the output. A ReLU output layer will produce outputs like 47.3 and 12.8 — you can't interpret these as probabilities or use cross-entropy loss on them directly.

3. **Forgetting numerical stability in softmax.** If logits are large (common in poorly initialized networks), `e^z` overflows to `inf`. The result is `nan`, and once you have a `nan` in the network, it propagates everywhere. The stable softmax (subtract max) costs nothing and prevents this entirely.

4. **Using tanh in very deep feedforward networks.** Tanh is better than sigmoid but still vanishes at depth. Beyond ~5 layers, use ReLU-family. Use tanh only where its zero-centered bounded output is specifically needed (RNN gates).

5. **Ignoring the dying ReLU problem.** If you see training loss plateau while validation loss is also high (both learning and performance stuck), and gradient norms in early layers are zero, you likely have dead neurons. Monitor the fraction of neurons outputting zero. If >50% are dead, reduce learning rate or switch to Leaky ReLU.

---

### 3.13 Google/Apple-Level Interview Q&A

---

**Q1: "ReLU is not differentiable at z=0. Why doesn't this break gradient descent in practice? What do modern frameworks actually do at that point?"**

*Why this is asked:* This tests mathematical rigor. Candidates who say "ReLU is differentiable" are wrong. Candidates who say "it breaks gradient descent" are also wrong. The correct answer requires understanding subgradients and the measure-theoretic reason why a single point of non-differentiability is harmless.

**Answer:**

ReLU is indeed non-differentiable at exactly z=0. The left-hand derivative is 0, the right-hand derivative is 1 — they disagree. In classical calculus, the derivative does not exist at this point.

However, this doesn't matter for two reasons:

**Reason 1: Measure zero.** In the real numbers, a single point (z=0) has measure zero. The probability that any neuron's pre-activation is *exactly* 0.000...000 during training is essentially zero (floating point numbers are continuous). In practice, you will essentially never hit z=0 exactly.

**Reason 2: Subgradients.** For convex functions (and piecewise-linear functions like ReLU), we can use the *subgradient* at non-differentiable points. The subdifferential of ReLU at z=0 is the interval [0, 1] — any value in [0,1] is a valid subgradient. Gradient descent with any valid subgradient still converges.

**What frameworks do:**

```python
# PyTorch (C++ backend) handles z=0 with a convention:
# ReLU'(0) = 0   ← use the left derivative

# This means at z=0, the neuron behaves as "inactive."
# Mathematically consistent, practically irrelevant.

# You can verify:
import torch
z = torch.tensor(0.0, requires_grad=True)
y = torch.relu(z)
y.backward()
print(z.grad)  # prints: tensor(0.)
```

The convention (0 or 1 at z=0) doesn't matter for training because:
- z=0 is a set of measure zero
- Even if it occurs, the weight update is either 0 or some small value — neither causes instability

---

**Q2: "Explain why the softmax function is called 'soft' max. What is 'hard' max? Under what temperature does softmax approach hardmax, and what happens when temperature → ∞?"**

*Why this is asked:* Tests depth of understanding beyond "softmax converts logits to probabilities." The temperature concept is critical for modern LLM inference (temperature sampling), knowledge distillation, and reinforcement learning. Apple/Google ask this because it distinguishes practitioners who use softmax from those who understand it.

**Answer:**

**Hardmax** (argmax one-hot):

```
hardmax([2.0, 1.0, 0.5]) = [1, 0, 0]

The maximum logit gets probability 1, all others get 0.
This is a "winner-takes-all" function.
Not differentiable → can't use in training.
```

**Softmax with temperature T:**

```
Softmax_T(zₖ) = e^(zₖ/T) / Σⱼ e^(zⱼ/T)

Standard softmax is T=1.

As T → 0  (cold):
  zₖ/T → ±∞ for any nonzero difference
  The largest logit dominates exponentially
  Softmax → hardmax (one-hot)
  Distribution becomes "peaky" / confident

  Example: z = [2.0, 1.0, 0.5], T = 0.1
  z/T = [20, 10, 5]
  e^z/T ≈ [485M, 22026, 148]
  p ≈ [0.9999, 0.00005, 0.0000003]  ← nearly one-hot

As T → ∞  (hot):
  zₖ/T → 0 for all k
  e^(zₖ/T) → 1 for all k
  Softmax → uniform distribution (1/K for each class)
  All classes equally likely — maximum uncertainty

  Example: z = [2.0, 1.0, 0.5], T = 100
  z/T = [0.02, 0.01, 0.005]
  e^z/T ≈ [1.020, 1.010, 1.005]
  p ≈ [0.337, 0.334, 0.330]  ← nearly uniform
```

**Why this matters in practice:**

1. **LLM inference temperature:** When sampling from GPT-4, temperature controls creativity. T=0 → deterministic (always pick the top token = greedy decoding). T=1 → standard sampling. T=1.5 → more random, creative, sometimes nonsensical.

2. **Knowledge distillation** (Hinton et al., 2015): Train a small student network to match a large teacher network's soft probabilities at high temperature T (e.g., T=5). The soft labels carry more information than hard one-hot labels — a "2" being 60% likely and a "7" being 30% likely is more informative than just "it's a 2." High temperature makes these soft targets more pronounced.

3. **The "soft" in softmax:** The function is a smooth, differentiable approximation to hardmax. It's "soft" because it doesn't make a hard decision — it distributes probability while favoring the maximum.

---

**Q3: "You're training a 20-layer network with ReLU activations and you observe that the training loss is stuck and gradients in the first 5 layers are effectively zero, but gradients in the last 5 layers are normal. What are the two most likely causes and how would you diagnose and fix each?"**

*Why this is asked:* This is a real debugging scenario that every ML engineer at Google/Apple faces. It tests the ability to translate theoretical knowledge (vanishing gradients, dying ReLU) into practical diagnosis and remediation. It also reveals whether the candidate knows what "effective" debugging looks like — monitoring the right metrics.

**Answer:**

**The symptom:** Gradient norms near zero in early layers, normal in late layers, with stuck training loss. This gradient imbalance means early layers are not learning.

**Cause 1: Dying ReLU (most likely with ReLU activations)**

Diagnosis:
```python
# During a training step, log the fraction of zero activations per layer
for name, module in model.named_modules():
    if isinstance(module, nn.ReLU):
        # Hook to capture output
        def hook(m, input, output):
            dead_fraction = (output == 0).float().mean().item()
            print(f"{name}: {dead_fraction:.1%} dead neurons")
        module.register_forward_hook(hook)

# If early layers show 70-100% dead neurons → dying ReLU confirmed
```

Fix options (in order of preference):
1. **Reduce learning rate.** Large learning rates push weights into large negative values, killing neurons. Try 10× smaller lr.
2. **Switch to Leaky ReLU / ELU.** Neurons can recover because gradient is nonzero for z < 0.
3. **Better weight initialization.** He initialization (Chapter 7) designed for ReLU prevents this.
4. **Gradient clipping.** Prevents large weight updates that kill neurons.

**Cause 2: Vanishing gradients from poor architecture choices**

If dying ReLU fraction is normal (say, 40-50%) but gradients still vanish:
- Could be a sigmoid/tanh layer accidentally left in the architecture
- Could be an unusually deep network where even ReLU gradients degrade due to other factors (poor init, very deep)

Diagnosis:
```python
# Log gradient norms by layer
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm = {grad_norm:.6f}")

# Early layers: near zero → vanishing
# Late layers: normal values → gradient doesn't propagate back
```

Fix options:
1. **Add residual connections** (Chapter 11). Residuals create a "gradient highway" that bypasses layers: gradient flows directly from output to any layer without passing through 20 matrix multiplies.
2. **Batch Normalization** (Chapter 9). Normalizes activations to have unit variance, preventing gradient magnitude from decaying or exploding.
3. **Better initialization.** He init sets weight variance = 2/nᵢₙ, specifically calibrated to keep gradient variance constant through ReLU layers.
4. **Gradient checkpointing + monitoring.** Use tensorboard to log per-layer gradient norms every N steps. You need to *see* the problem to solve it.

**General debugging protocol:**
```
Step 1: Log per-layer activation means and stds → detect dead neurons
Step 2: Log per-layer gradient norms → confirm early layer starvation
Step 3: Check for non-ReLU activations in early layers (architecture bug)
Step 4: Try: (a) lower LR, (b) Leaky ReLU, (c) add BatchNorm, (d) add residuals
Step 5: Verify fix by checking gradient norms normalize across layers
```

---

*End of Chapter 3. Chapter 4 (Forward Propagation) coming next.*
