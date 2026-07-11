# Chapter 6: Backpropagation & Gradient Descent

---

### 6.1 The Plain-English Picture

You've run a forward pass. The network made a prediction. The loss function measured how wrong it was. Now what?

Now you need to figure out: *which weights caused the error, and by how much?* That is the question backpropagation answers.

Here is the key insight. The loss L is a function of every weight in the network. If you increase weight w by a tiny amount ε, the loss changes by some amount ΔL. The ratio ΔL/ε is the gradient of L with respect to w — written ∂L/∂w. It tells you two things: the *sign* (should w increase or decrease to reduce loss?) and the *magnitude* (how sensitive is the loss to this particular weight?).

If ∂L/∂w is large and positive: increasing w increases loss → decrease w.
If ∂L/∂w is large and negative: increasing w decreases loss → increase w.
If ∂L/∂w ≈ 0: changing w barely affects loss → this weight doesn't matter much.

The problem: a neural network has millions of weights, and the loss is computed at the end after passing through all layers. How do you efficiently compute ∂L/∂w for every single weight?

The answer is backpropagation — an application of the chain rule from calculus, applied systematically backwards through the computational graph. It is the most important algorithm in deep learning. Every neural network that has ever been trained has used backpropagation (or something equivalent). Understanding it completely is non-negotiable.

The two-phase picture:

```
TRAINING LOOP
=============

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  1. FORWARD PASS                                    │
  │     Data flows: x → layer 1 → layer 2 → ... → ŷ   │
  │     Cache all zˡ and aˡ                             │
  │     Compute loss L = loss(ŷ, y)                     │
  │                                                     │
  │  2. BACKWARD PASS (Backpropagation)                 │
  │     Gradient flows: ∂L/∂aᴸ → ... → ∂L/∂W¹         │
  │     Compute ∂L/∂Wˡ and ∂L/∂bˡ for every layer     │
  │                                                     │
  │  3. WEIGHT UPDATE (Gradient Descent)                │
  │     Wˡ ← Wˡ - η · ∂L/∂Wˡ                         │
  │     bˡ ← bˡ - η · ∂L/∂bˡ                         │
  │                                                     │
  │  Repeat until loss is minimized                     │
  └─────────────────────────────────────────────────────┘
```

---

### 6.2 The Chain Rule: The Engine of Backpropagation

Backpropagation is the chain rule. That's it. Let's be precise about what that means.

**Univariate chain rule:**

```
If L depends on z, and z depends on w:
  L = f(z),  z = g(w)
  Then: dL/dw = (dL/dz) · (dz/dw)

Example:
  z = 3w²,   L = z³
  dz/dw = 6w
  dL/dz = 3z²
  dL/dw = 3z² · 6w = 18wz² = 18w(3w²)² = 162w⁵
```

**Multivariate chain rule (what we actually use):**

```
If L depends on z₁, z₂, ..., zₙ, and each zₖ depends on w:
  ∂L/∂w = Σₖ (∂L/∂zₖ) · (∂zₖ/∂w)

Example in a network:
  z² = W²a¹ + b²
  a² = σ(z²)
  z³ = W³a² + b³
  L = loss(z³, y)

  ∂L/∂W² = ?

  Chain rule:
  ∂L/∂W² = (∂L/∂z³) · (∂z³/∂a²) · (∂a²/∂z²) · (∂z²/∂W²)
             ↑             ↑            ↑             ↑
          from loss    = W³ (linear)  = σ'(z²)   = a¹ᵀ
          gradient

  This is the chain rule applied through 3 intermediate steps.
  In a 100-layer network, the chain has 100 terms.
  Backpropagation computes all of them SIMULTANEOUSLY and EFFICIENTLY
  by reusing intermediate gradients (dynamic programming).
```

---

### 6.3 The Delta (Error Signal): The Core Quantity

Define the **error signal** δˡ (delta) at layer l as the gradient of the loss with respect to the pre-activation:

```
δˡ = ∂L/∂zˡ    shape: [nˡ × 1]

This is the "error" that each neuron in layer l is responsible for.
It is the most important intermediate quantity in backprop.

Once you have δˡ, everything follows immediately:

  ∂L/∂Wˡ = δˡ · (aˡ⁻¹)ᵀ     gradient w.r.t. weights in layer l
  ∂L/∂bˡ = δˡ                 gradient w.r.t. biases in layer l
  δˡ⁻¹   = (Wˡ)ᵀ · δˡ ⊙ σ'(zˡ⁻¹)   propagate error to previous layer

Where:
  ⊙     = element-wise (Hadamard) product
  σ'(·) = derivative of the activation function
  (·)ᵀ  = transpose

The key recurrence:
  δᴸ   = ∂L/∂zᴸ              (gradient at output layer — depends on loss)
  δˡ   = (Wˡ⁺¹)ᵀ · δˡ⁺¹ ⊙ σ'(zˡ)  (propagate backward)
```

This recurrence is why it's called *back*propagation — the error signal propagates from the output layer (L) backward through each layer (L-1, L-2, ..., 1), accumulating gradient information as it goes.

---

### 6.4 The Four Equations of Backpropagation

These four equations completely characterize backpropagation for any feedforward network. Memorize them. Derive them. Understand them.

```
═══════════════════════════════════════════════════════════════
THE FOUR BACKPROP EQUATIONS
═══════════════════════════════════════════════════════════════

(BP1)  δᴸ = ∇ₐᴸ L ⊙ σ'(zᴸ)
       Error at output layer = loss gradient w.r.t. aᴸ,
       element-wise multiplied by activation derivative.

(BP2)  δˡ = ((Wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ σ'(zˡ)
       Error at layer l = (transposed next-layer weights times
       next-layer error), element-wise multiplied by local
       activation derivative.

(BP3)  ∂L/∂bˡ = δˡ
       Gradient of bias = error signal at that layer.

(BP4)  ∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ
       Gradient of weights = outer product of error signal
       and previous layer activations.

Where:
  ∇ₐᴸ L  = ∂L/∂aᴸ  (gradient of loss w.r.t. output activations)
  ⊙       = element-wise multiply
  σ'(zˡ) = derivative of activation applied to each element of zˡ
  (·)ᵀ   = matrix/vector transpose

These four equations are derived entirely from the chain rule.
There is no magic here — only careful bookkeeping of derivatives.
```

**Special case for Softmax + Cross-Entropy output:**

```
For the output layer with softmax activation and CCE loss:
  δᴸ = ŷ - y     (prediction minus true label)

This is the cleanest possible gradient. The complexity of
softmax and log cancel out perfectly:

  Proof sketch:
  L = -Σₖ yₖ log(ŷₖ)
  ŷₖ = e^(zₖ) / Σⱼ e^(zⱼ)

  ∂L/∂zₖ = ŷₖ - yₖ

  So δᴸ = ŷ - y  (vector of prediction minus one-hot label)
  e.g., ŷ = [0.7, 0.2, 0.1], y = [1, 0, 0]
        δᴸ = [-0.3, 0.2, 0.1]

  Interpretation: output neuron 0 needs to increase (it's 0.3 too low),
  neurons 1 and 2 need to decrease (they're 0.2 and 0.1 too high).
```

---

### 6.5 Gradient Descent: Using the Gradients

Once we have ∂L/∂Wˡ and ∂L/∂bˡ for all layers, gradient descent performs the weight update:

```
GRADIENT DESCENT UPDATE RULE
==============================

  Wˡ ← Wˡ - η · ∂L/∂Wˡ
  bˡ ← bˡ - η · ∂L/∂bˡ

Where:
  η (eta) = learning rate (a small positive scalar, e.g. 0.01)
  The minus sign: we move OPPOSITE to the gradient
  (gradient points uphill, we want to go downhill)

Geometric intuition:
  L
  │          ╲
  │            ╲      gradient > 0 here (slope upward to right)
  │              ╲    → decrease w (move left → down the slope)
  │                ╲
  │                  ╲___
  │                       ╲
  └─────────────────────────── w
                    ↑
                  minimum

Learning rate η controls step size:
  η too large:  overshoot minimum, diverge
  η too small:  converge very slowly, get stuck
  η just right: steady convergence to minimum

  Loss                         Loss                         Loss
  │  ↗ diverge                 │    converge slowly          │  converge well
  │ ↗                          │                  ──────     │        ──
  │↗                           │           ──────            │  ──────
  └────── epochs               └────── epochs               └────── epochs
     η too large                   η too small                  η just right
```

---

### 6.6 Variants of Gradient Descent

```
THREE VARIANTS
==============

1. BATCH GRADIENT DESCENT (BGD)
   Compute gradient over ALL N training examples, then update once.

   ∂L/∂W = (1/N) Σᵢ ∂Lᵢ/∂W

   Pros:  Exact gradient. Guaranteed to converge (convex case).
          Stable, smooth loss curve.
   Cons:  N can be millions → one update requires full dataset pass.
          Extremely slow. Doesn't fit in GPU memory.
   Use:   Almost never in deep learning. Classical ML only.

2. STOCHASTIC GRADIENT DESCENT (SGD)
   Compute gradient on ONE example at a time, update after each.

   ∂L/∂W ≈ ∂Lᵢ/∂W    for randomly selected example i

   Pros:  Very fast updates. Can escape local minima (noisy gradient).
          Memory: only one example at a time.
   Cons:  Noisy gradient → zigzag path → slow convergence.
          Can't use vectorized GPU operations efficiently.
   Use:   Rarely used alone. Historically important.

3. MINI-BATCH GRADIENT DESCENT (THE STANDARD)
   Compute gradient on a batch of m examples (m << N), then update.

   ∂L/∂W ≈ (1/m) Σᵢ∈batch ∂Lᵢ/∂W    m ∈ {32, 64, 128, 256}

   Pros:  Vectorized → fast GPU computation.
          Less noisy than SGD → smoother convergence.
          Noise helps escape saddle points (unlike BGD).
          Memory efficient → only m examples at a time.
   Cons:  Introduces hyperparameter m (batch size).
          Still noisy (unlike BGD).
   Use:   THE STANDARD. When people say "SGD" they usually mean this.

In practice: mini-batch GD with Adam optimizer (Chapter 8)
is what trains 99% of deep networks.
```

---

### 6.7 The Learning Rate: Most Important Hyperparameter

```
LEARNING RATE EFFECTS
=====================

Too large (η = 1.0):
  Loss
  10│╲
   8│  ╲  ╱╲
   6│    ╲╱  ╲
   4│         ╲╱
   2│           oscillates / diverges
    └──────── epochs

Too small (η = 0.00001):
  Loss
  10│──────────────────────────
   8│
   6│    barely moves
   4│
   2│
    └──────── epochs

Just right (η = 0.001):
  Loss
  10│╲
   8│  ╲
   6│    ╲──
   4│        ╲──
   2│             ──
   1│               ──────────
    └──────── epochs

LEARNING RATE SCHEDULES
========================
Static η is rarely optimal. Schedules adapt η over training:

  Step decay:      η ← η × 0.1  every 30 epochs
  Exponential:     η(t) = η₀ · e^(-kt)
  Cosine annealing: η(t) = η_min + (1/2)(η_max - η_min)(1 + cos(πt/T))
  Warmup + decay:  Linear increase for first W steps, then decay
                   (critical for transformers: avoids early instability)

  Loss with cosine annealing:
  10│╲
   8│  ╲           ╲
   6│    ╲         ╲  ←─ brief uptick when lr warms up
   4│      ╲────────
   2│              ╲──────
    └──────── epochs
    ↑ initial decay  ↑ second cycle
```

---

### 6.8 Worked Numerical Example: Full Backprop by Hand

A complete backward pass through a tiny network. Every number computed explicitly.

```
═══════════════════════════════════════════════════════════════
SETUP (from Chapter 4 forward pass)
═══════════════════════════════════════════════════════════════

Architecture:  2 → 2 → 1
Activation:    Sigmoid everywhere (for clean derivatives)
Loss:          Binary Cross-Entropy

Input:         x = a⁰ = [1.0, 0.5]
True label:    y = 1

Parameters:
  W¹ = [[0.3,  0.5],
         [0.2, -0.1]]      b¹ = [0.0, 0.0]

  W² = [[0.8, -0.3]]       b² = [0.0]

Learning rate: η = 0.5

═══════════════════════════════════════════════════════════════
FORWARD PASS (compute and cache everything)
═══════════════════════════════════════════════════════════════

Layer 1:
  z¹ = W¹ · a⁰ + b¹
  z¹₁ = (0.3)(1.0) + (0.5)(0.5) + 0 = 0.30 + 0.25 = 0.55
  z¹₂ = (0.2)(1.0) + (-0.1)(0.5) + 0 = 0.20 - 0.05 = 0.15

  z¹ = [0.55, 0.15]

  a¹ = σ(z¹)
  a¹₁ = σ(0.55) = 1/(1+e^(-0.55)) = 1/(1+0.5769) = 1/1.5769 = 0.6342
  a¹₂ = σ(0.15) = 1/(1+e^(-0.15)) = 1/(1+0.8607) = 1/1.8607 = 0.5374

  a¹ = [0.6342, 0.5374]

Layer 2:
  z² = W² · a¹ + b²
  z²₁ = (0.8)(0.6342) + (-0.3)(0.5374) + 0
       = 0.5074 - 0.1612 = 0.3462

  z² = [0.3462]

  ŷ = a² = σ(0.3462) = 1/(1+e^(-0.3462))
                      = 1/(1+0.7072)
                      = 1/1.7072
                      = 0.5857

LOSS:
  L = BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
          = -[1·log(0.5857) + 0·log(0.4143)]
          = -log(0.5857)
          = 0.5347

═══════════════════════════════════════════════════════════════
BACKWARD PASS
═══════════════════════════════════════════════════════════════

STEP 1: Gradient at output (BP1)
─────────────────────────────────
For BCE + sigmoid output, we showed that:
  δ² = ŷ - y = 0.5857 - 1 = -0.4143

  Interpretation: output is 0.4143 below target.
  Negative gradient → we need to increase z² → increase the output.

STEP 2: Gradients for Layer 2 weights (BP4)
────────────────────────────────────────────
  ∂L/∂W² = δ² · (a¹)ᵀ
           = [-0.4143] · [0.6342, 0.5374]
           = [(-0.4143)(0.6342), (-0.4143)(0.5374)]
           = [-0.2628, -0.2226]

  So: ∂L/∂W²₁₁ = -0.2628   (gradient for weight connecting h₁ to output)
      ∂L/∂W²₁₂ = -0.2226   (gradient for weight connecting h₂ to output)

  ∂L/∂b² = δ² = [-0.4143]

STEP 3: Propagate error to Layer 1 (BP2)
─────────────────────────────────────────
  Need σ'(z¹):
    σ'(z) = σ(z)(1 - σ(z))
    σ'(z¹₁) = a¹₁(1 - a¹₁) = 0.6342 × 0.3658 = 0.2319
    σ'(z¹₂) = a¹₂(1 - a¹₂) = 0.5374 × 0.4626 = 0.2486

  δ¹ = (W²)ᵀ · δ² ⊙ σ'(z¹)

  (W²)ᵀ = [[0.8], [-0.3]]   (transpose of [[0.8, -0.3]])

  (W²)ᵀ · δ² = [[0.8], [-0.3]] · [-0.4143]
              = [(0.8)(-0.4143), (-0.3)(-0.4143)]
              = [-0.3314, 0.1243]

  δ¹ = [-0.3314, 0.1243] ⊙ [0.2319, 0.2486]
     = [(-0.3314)(0.2319), (0.1243)(0.2486)]
     = [-0.0769, 0.0309]

STEP 4: Gradients for Layer 1 weights (BP4)
─────────────────────────────────────────────
  ∂L/∂W¹ = δ¹ · (a⁰)ᵀ

  a⁰ = [1.0, 0.5]

  ∂L/∂W¹ = [[-0.0769], [0.0309]] · [[1.0, 0.5]]

          = [(-0.0769)(1.0), (-0.0769)(0.5)]
            [(0.0309)(1.0),  (0.0309)(0.5) ]

          = [[-0.0769, -0.0385],
             [ 0.0309,  0.0155]]

  ∂L/∂b¹ = δ¹ = [-0.0769, 0.0309]

═══════════════════════════════════════════════════════════════
WEIGHT UPDATE (η = 0.5)
═══════════════════════════════════════════════════════════════

W² ← W² - η · ∂L/∂W²
   = [[0.8, -0.3]] - 0.5 · [[-0.2628, -0.2226]]
   = [[0.8 - (-0.1314), -0.3 - (-0.1113)]]
   = [[0.9314, -0.1887]]

b² ← b² - η · ∂L/∂b²
   = [0.0] - 0.5 · [-0.4143]
   = [0.2072]

W¹ ← W¹ - η · ∂L/∂W¹
   = [[0.3,  0.5 ],   -  0.5 · [[-0.0769, -0.0385],
      [0.2, -0.1 ]]              [ 0.0309,  0.0155]]

   = [[0.3-(-0.0385),  0.5-(-0.0193)],
      [0.2-(0.0155), -0.1-(0.0078) ]]

   = [[0.3385,  0.5193],
      [0.1845, -0.1078]]

b¹ ← b¹ - η · ∂L/∂b¹
   = [0.0, 0.0] - 0.5 · [-0.0769, 0.0309]
   = [0.0385, -0.0155]

═══════════════════════════════════════════════════════════════
VERIFICATION: run forward pass with new weights
═══════════════════════════════════════════════════════════════

z¹₁(new) = (0.3385)(1.0) + (0.5193)(0.5) + 0.0385
          = 0.3385 + 0.2597 + 0.0385 = 0.6367
a¹₁(new) = σ(0.6367) ≈ 0.6539

z¹₂(new) = (0.1845)(1.0) + (-0.1078)(0.5) + (-0.0155)
          = 0.1845 - 0.0539 - 0.0155 = 0.1151
a¹₂(new) = σ(0.1151) ≈ 0.5287

z²(new) = (0.9314)(0.6539) + (-0.1887)(0.5287) + 0.2072
         = 0.6089 - 0.0998 + 0.2072 = 0.7163

ŷ(new) = σ(0.7163) ≈ 0.6714

New loss = -log(0.6714) = 0.3984

OLD LOSS: 0.5347
NEW LOSS: 0.3984
REDUCTION: 0.1363 (25.5% improvement in one step!)

One gradient descent step moved us meaningfully toward the target.
Repeat thousands of times → network converges.
```

---

### 6.9 Computational Complexity of Backprop

```
FORWARD PASS:
  Each layer l: one matrix multiply [nˡ × nˡ⁻¹] · [nˡ⁻¹ × m]
  Cost: O(nˡ · nˡ⁻¹ · m) per layer
  Total: O(m · Σˡ nˡ · nˡ⁻¹)

BACKWARD PASS:
  Each layer l: one matrix multiply (W)ᵀ · δ: [nˡ⁻¹ × nˡ] · [nˡ × m]
                one outer product δ · aᵀ:     [nˡ × m] · [m × nˡ⁻¹]
  Cost: O(nˡ · nˡ⁻¹ · m) per layer  ← SAME as forward pass!
  Total: O(m · Σˡ nˡ · nˡ⁻¹)

KEY RESULT:
  Backprop costs ~2-3× forward pass. Not 10×, not N×.
  This is the miracle of backprop: computing gradients for
  ALL parameters is only a constant factor more expensive
  than a single forward pass.

  Before backprop, computing gradients required O(P) forward
  passes (one per parameter, finite differences). For a network
  with P = 10⁸ parameters, this is 10⁸× more expensive.
  Backprop reduces this to 1 backward pass. This is why
  deep learning is computationally feasible.

MEMORY:
  Must cache all {zˡ, aˡ} during forward pass for backprop.
  Memory ∝ L × m × max_layer_width
  For very deep networks or large batches: can run out of GPU memory.
  Solution: gradient checkpointing — cache only every k-th layer,
  recompute the rest on demand. Trades compute for memory.
```

---

### 6.10 Vanishing and Exploding Gradients (Revisited from Backprop Perspective)

```
VANISHING GRADIENTS
===================

From BP2: δˡ = (Wˡ⁺¹)ᵀ · δˡ⁺¹ ⊙ σ'(zˡ)

At each layer, gradient is multiplied by:
  1. Weights Wˡ⁺¹ (magnitude depends on initialization)
  2. σ'(zˡ)       (activation derivative)

For sigmoid: σ'(z) ≤ 0.25
For L layers: δ¹ ≈ δᴸ · Π_{l=2}^{L} (Wˡ)ᵀ · σ'(zˡ)

If each factor has magnitude < 1:
  ||δ¹|| ≈ ||δᴸ|| · c^(L-1)  where c < 1
  → exponential decay with depth
  → early layers receive near-zero gradients
  → early layers don't learn

EXPLODING GRADIENTS
===================

If each factor has magnitude > 1:
  ||δ¹|| ≈ ||δᴸ|| · c^(L-1)  where c > 1
  → exponential growth with depth
  → gradients become astronomically large
  → weight updates are massive → loss diverges → nan

Symptoms:
  - Loss suddenly jumps to nan
  - Weights become inf
  - Gradient norm explodes

Fix: GRADIENT CLIPPING
  if ||g|| > threshold:
    g ← g · (threshold / ||g||)

  Clips gradient VECTOR to have max norm = threshold.
  Preserves direction, limits magnitude.
  Standard threshold: 1.0 or 5.0
  Used in almost all RNN/LSTM training (Chapter 12-13).

  PyTorch: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 6.11 Automatic Differentiation (Autograd)

Modern frameworks don't implement backprop by hand — they use automatic differentiation.

```
THREE WAYS TO COMPUTE DERIVATIVES
===================================

1. NUMERICAL DIFFERENTIATION (finite differences)
   ∂L/∂w ≈ [L(w + ε) - L(w - ε)] / (2ε)
   
   Cost: one forward pass per parameter (O(P) passes)
   Error: O(ε²) approximation error
   Use:  gradient checking (debugging only)

2. SYMBOLIC DIFFERENTIATION (CAS like Mathematica)
   Build a symbolic expression for dL/dw.
   Cost: expression can grow exponentially ("expression swell")
   Use:  not practical for neural networks

3. AUTOMATIC DIFFERENTIATION (what PyTorch/JAX use)
   Record operations during forward pass (builds a tape/graph).
   Replay backward through the graph, applying chain rule at each node.
   Cost: O(1) backward pass (backpropagation!)
   Error: exact (to floating point precision)
   Use:  all modern deep learning

FORWARD MODE vs REVERSE MODE AD
================================

  Forward mode: propagate derivatives forward alongside values.
    Computes ∂output/∂wᵢ for ONE weight wᵢ in one pass.
    Cost: O(P) passes to get all P gradients.
    Efficient when: more outputs than inputs (rare in DL).

  Reverse mode: backpropagation.
    Computes ∂L/∂ALL weights in one backward pass.
    Cost: ~constant × forward pass (regardless of P!).
    Efficient when: scalar output (loss), many inputs (weights).
    → This is exactly our case. Reverse mode = backpropagation.

HOW PYTORCH AUTOGRAD WORKS
===========================

  x = torch.tensor([1.0], requires_grad=True)
  y = x ** 2 + 3 * x          # forward: builds graph
  y.backward()                 # backward: chain rule on graph
  print(x.grad)                # ∂y/∂x = 2x + 3 = 5.0

  Under the hood:
  - Every tensor operation creates a "grad_fn" node
  - grad_fn stores: (1) the operation type, (2) input tensors
  - backward() traverses these nodes in reverse, calling each
    node's VJP (vector-Jacobian product) function
  - Gradients accumulate in .grad attributes
```

---

### 6.12 Why This Matters — What Breaks If You Get This Wrong

1. **Incorrect gradient flow = incorrect training.** Any bug in backpropagation produces wrong gradients. The network may appear to train (loss decreases) but converges to a wrong solution. Gradient checking (finite differences) is the diagnostic: compare ∂L/∂w from backprop to [L(w+ε) - L(w-ε)] / 2ε. If they differ by more than ~10⁻⁵, there's a bug.

2. **Not zeroing gradients between batches.** PyTorch accumulates gradients by default (`.backward()` adds to `.grad`). If you forget `optimizer.zero_grad()` before each backward pass, gradients from previous batches contaminate the current update. Loss oscillates wildly. Always zero gradients before each batch.

3. **Gradient clipping ignored for RNNs.** RNNs backpropagate through time — effectively a very deep network (one layer per timestep). Exploding gradients are almost inevitable without clipping. Skipping clip_grad_norm on an RNN is like skipping a seat belt — fine until it isn't.

4. **Detaching tensors from the graph accidentally.** If you call `.detach()` on a tensor in the middle of your network (e.g., to convert to numpy for logging), and then accidentally use that detached tensor in subsequent operations, the gradient flow is severed. The layers before the detach point receive zero gradients. Subtle, silent, catastrophic.

5. **Double backward.** Some architectures need second-order gradients (∂²L/∂w²). Standard backprop only computes first-order. Calling `.backward()` twice doesn't compute second-order derivatives — it accumulates first-order gradients again. Use `torch.autograd.grad()` with `create_graph=True` for second-order.

---

### 6.13 Google/Apple-Level Interview Q&A

---

**Q1: "Derive the backpropagation equations from scratch for a two-layer network with sigmoid activations and MSE loss. Show every chain rule application explicitly."**

*Why this is asked:* This is the canonical "do you actually understand backprop or did you just use PyTorch?" question. Every serious ML engineering interview at top companies includes a backprop derivation. It reveals whether the candidate can think from first principles — essential for debugging, implementing custom layers, and understanding what goes wrong in training.

**Answer:**

```
SETUP
=====
Network:   x → [W¹,b¹] → z¹ → σ → a¹ → [W²,b²] → z² → σ → ŷ
Loss:      L = (1/2)(ŷ - y)²    (MSE, factor 1/2 for clean derivative)
Activation: σ(z) = 1/(1+e^(-z)), σ'(z) = σ(z)(1-σ(z))

FORWARD PASS (define all quantities):
  z¹ = W¹x + b¹
  a¹ = σ(z¹)
  z² = W²a¹ + b²
  ŷ = a² = σ(z²)
  L = (1/2)(ŷ - y)²

BACKWARD PASS — apply chain rule layer by layer:

∂L/∂ŷ:
  ∂L/∂ŷ = ŷ - y       [derivative of (1/2)(ŷ-y)²]

∂L/∂z² (= δ²):
  ∂L/∂z² = ∂L/∂ŷ · ∂ŷ/∂z²
           = (ŷ - y) · σ'(z²)
           = (ŷ - y) · ŷ(1 - ŷ)     [since ŷ = σ(z²)]

∂L/∂W²:
  ∂L/∂W² = ∂L/∂z² · ∂z²/∂W²
           = δ² · (a¹)ᵀ              [since z² = W²a¹ + b², ∂z²/∂W² = a¹ᵀ]

∂L/∂b²:
  ∂L/∂b² = ∂L/∂z² · ∂z²/∂b²
           = δ² · 1 = δ²

∂L/∂a¹ (gradient w.r.t. hidden activations):
  ∂L/∂a¹ = ∂L/∂z² · ∂z²/∂a¹
           = δ² · W²                  [since z² = W²a¹ + b², ∂z²/∂a¹ = W²]
           = (W²)ᵀ δ²                 [transpose for correct dimensions]

∂L/∂z¹ (= δ¹):
  ∂L/∂z¹ = ∂L/∂a¹ · ∂a¹/∂z¹
           = (W²)ᵀ δ² · σ'(z¹)
           = (W²)ᵀ δ² ⊙ a¹(1 - a¹)   [element-wise multiply]

∂L/∂W¹:
  ∂L/∂W¹ = δ¹ · (x)ᵀ                [since z¹ = W¹x + b¹, ∂z¹/∂W¹ = xᵀ]

∂L/∂b¹:
  ∂L/∂b¹ = δ¹

SUMMARY:
  δ²     = (ŷ - y) ⊙ ŷ(1-ŷ)
  δ¹     = (W²)ᵀ δ² ⊙ a¹(1-a¹)
  ∂L/∂W² = δ² (a¹)ᵀ
  ∂L/∂b² = δ²
  ∂L/∂W¹ = δ¹ (x)ᵀ
  ∂L/∂b¹ = δ¹

These are exactly BP1-BP4 instantiated for this specific network.
```

---

**Q2: "What is a vector-Jacobian product (VJP) and why is it the fundamental primitive of reverse-mode automatic differentiation? How does PyTorch use VJPs to implement backpropagation?"**

*Why this is asked:* This tests understanding at the framework level — crucial for implementing custom layers, debugging autograd, and understanding why backprop is efficient. It distinguishes ML engineers who write custom CUDA kernels from those who only call `model.fit()`.

**Answer:**

**The Jacobian:**

```
For a function f: ℝⁿ → ℝᵐ (n inputs, m outputs):
The Jacobian J ∈ ℝᵐˣⁿ contains all partial derivatives:
  Jᵢⱼ = ∂fᵢ/∂xⱼ

For a single layer: a = σ(Wx + b)
  a ∈ ℝⁿˡ, x ∈ ℝⁿˡ⁻¹
  Jacobian ∂a/∂x ∈ ℝ^(nˡ × nˡ⁻¹) — a potentially huge matrix

For a 1000→1000 layer: J is 1000×1000 = 10⁶ entries.
For a 10000→10000 layer: J is 10⁸ entries.
We NEVER explicitly form the full Jacobian.
```

**The Vector-Jacobian Product (VJP):**

```
Instead of forming J, we compute vᵀJ for a vector v ∈ ℝᵐ.
  VJP(f, x, v) = vᵀ · J   ∈ ℝⁿ

This is much cheaper: we never form the m×n matrix J.
We just compute the product of v with each column of J.
Cost: O(n·m) — same as one matrix-vector multiply.

In reverse mode AD:
  v = ∂L/∂a   (gradient flowing backward from downstream layers)
  VJP = vᵀ · (∂a/∂x) = ∂L/∂x   (gradient for upstream layers)

Each operation in the forward pass knows its VJP:
  Operation: z = Wx + b
  VJP w.r.t. x: vᵀ · (∂z/∂x) = vᵀ · W = Wᵀv
  VJP w.r.t. W: vᵀ · (∂z/∂W) = v · xᵀ  (outer product)
  VJP w.r.t. b: vᵀ · I = v
```

**How PyTorch uses VJPs:**

```python
# Every PyTorch operation registers a VJP function as its grad_fn:

# Example: matrix multiply
class MatMulBackward:
    def __init__(self, x, W):
        self.saved_x = x
        self.saved_W = W

    def __call__(self, v):  # v = upstream gradient (∂L/∂z)
        # VJP w.r.t. input x:  vᵀ · (∂z/∂x) = Wᵀv
        grad_x = self.saved_W.T @ v

        # VJP w.r.t. weight W:  vᵀ · (∂z/∂W) = v · xᵀ
        grad_W = v @ self.saved_x.T

        return grad_x, grad_W

# When you call z = W @ x:
# 1. PyTorch computes z (forward)
# 2. Attaches MatMulBackward(x, W) as z.grad_fn
# 3. When z.backward(v) is called:
#    - MatMulBackward(x, W)(v) computes grad_x and grad_W
#    - These are passed to x.grad_fn and W.grad_fn
#    - Recursion continues until leaf tensors

# This is exactly backpropagation, implemented as VJP compositions.
# Each node in the computation graph = one VJP.
# Backward pass = one VJP call per node, in reverse topological order.
```

**Why this is efficient:**

```
Forward pass: compute f₁, f₂, ..., fₙ (all operations)
Backward pass: compute VJP(fₙ), VJP(fₙ₋₁), ..., VJP(f₁)

Cost of each VJP ≈ cost of the corresponding forward operation.
Total backward cost ≈ total forward cost × constant.

Alternative (naive): compute full Jacobian at each layer.
Cost: O(n²) per layer instead of O(n).
For n=10⁶ weights: 10¹² operations vs 10⁶. Backprop wins by 10⁶×.
```

---

**Q3: "You're training a deep network and notice that the gradient norm for layer 1 is 10⁻⁸ while layer 10 is 1.0. Your loss is stuck. Walk me through three different interventions, explain the mechanism by which each one fixes the problem, and predict which will have the largest impact."**

*Why this is asked:* This is a real production debugging scenario. It tests the ability to connect theory (vanishing gradients) to root causes to solutions, and to reason about which solution is most effective. Companies like Google and Apple run teams that debug training at massive scale — this is a daily skill.

**Answer:**

**The problem:** Vanishing gradients — gradient signal decays by factor of ~10⁸ traversing from layer 10 to layer 1. Layer 1 is effectively not learning.

**Root cause analysis:**

```
From BP2: δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ σ'(zˡ)

Over 9 layers (10 → 1):
  ||δ¹|| = ||δ¹⁰|| · Π_{l=2}^{10} ||(Wˡ)ᵀ|| · |σ'(zˡ)|

If σ is sigmoid: each |σ'(z)| ≤ 0.25
With 9 multiplications: 0.25⁹ ≈ 3.8 × 10⁻⁶
Plus weight magnitudes < 1: easily reaches 10⁻⁸.
```

**Intervention 1: Replace sigmoid with ReLU (mechanism + impact)**

```
Mechanism: ReLU derivative = 1 for z > 0.
  The σ'(zˡ) term in BP2 becomes 1 for active neurons.
  Gradient no longer decays through the activation function.

  New gradient: ||δ¹|| ≈ ||δ¹⁰|| · Π_{l=2}^{10} ||(Wˡ)ᵀ||
  With Xavier init (||W|| ≈ 1): ||δ¹|| ≈ ||δ¹⁰||

Impact: LARGEST. This directly removes the exponential decay.
Gradient norm at layer 1 can jump from 10⁻⁸ to ~1.0.
Cost: change one argument in your layer definition. Free.
```

**Intervention 2: Add Batch Normalization after each layer (mechanism + impact)**

```
Mechanism: BatchNorm normalizes activations to zero mean,
unit variance after each layer. This has two effects on gradients:

  1. Keeps activation magnitudes from shrinking to zero.
     If a¹ has magnitude ~1 instead of ~10⁻⁴, the backprop
     signal through W² = δ² · (a¹)ᵀ also has reasonable magnitude.

  2. BatchNorm has a learnable scale γ. Its gradient has a
     direct path from loss to γ without passing through all the
     weight matrices — a form of shortcut for gradient flow.

Impact: LARGE. Often the second most impactful intervention.
BatchNorm is standard in almost all modern architectures.
Additional benefit: allows higher learning rates, stabilizes training.
```

**Intervention 3: Add Residual Connections (skip connections)**

```
Mechanism: Instead of aˡ = F(aˡ⁻¹), use aˡ = F(aˡ⁻¹) + aˡ⁻¹

During backprop through a residual block:
  ∂L/∂aˡ⁻¹ = ∂L/∂aˡ · (∂F/∂aˡ⁻¹ + I)
                                     ↑
                              identity term!

The identity adds 1 to the gradient at every residual block.
Even if ∂F/∂aˡ⁻¹ ≈ 0 (vanished), the I term preserves gradient:
  ∂L/∂aˡ⁻¹ ≈ ∂L/∂aˡ · 1 = ∂L/∂aˡ

Gradient flows through the identity path without decay.
In a 10-layer ResNet, layer 1 receives the same gradient
magnitude as layer 10 through the skip connection path.

Impact: LARGE. This is the key innovation of ResNet (Chapter 11).
Enables training of 100+ layer networks.
```

**Ranking by impact (for this scenario):**

```
1. ReLU activation:        largest impact, simplest fix
2. Residual connections:   large impact, requires architecture change
3. Batch Normalization:    large impact, adds parameters and compute

In practice: use all three together. Modern networks use
ReLU + ResNet + BatchNorm as a standard package.

Additional quick wins:
  - Better initialization (He init for ReLU) — free, immediate
  - Gradient clipping — prevents the opposite problem (exploding)
  - Lower learning rate — if exploding gradients are also occurring
```

---

*End of Chapter 6. Chapter 7 (Weight Initialization) coming next.*
