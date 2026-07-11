# Chapter 9: Regularization (Dropout, L1/L2, Batch Normalization)

---

### 9.1 The Plain-English Picture

A model that memorizes the training data is useless. It has learned the noise, the quirks, the accidental correlations specific to the examples it saw — not the underlying pattern that would let it generalize to new data. This is overfitting, and it is the central problem of machine learning.

Regularization is any technique that reduces overfitting by constraining the model — making it work harder to earn each unit of complexity it uses. The word comes from "regularize" in the mathematical sense: to make well-behaved, to impose smoothness, to prevent pathological solutions.

Think of it this way. A student can ace an exam by memorizing every practice problem, or by understanding the underlying concepts. The first student fails on new questions; the second doesn't. Regularization is what forces the student to learn concepts rather than memorize answers.

There are three main families:

```
REGULARIZATION FAMILIES
========================

1. PARAMETER CONSTRAINTS (L1/L2 weight decay)
   ─────────────────────────────────────────────
   Penalize large weights directly in the loss function.
   Large weights = complex functions = overfitting.
   Force weights to stay small → simpler models → better generalization.
   
   "You can use these neurons, but using them costs you."

2. STRUCTURAL NOISE (Dropout)
   ─────────────────────────────────────────────
   Randomly disable neurons during training.
   Forces the network to learn redundant representations.
   No single neuron can be relied upon → ensemble effect.
   
   "You can have these neurons, but you can't count on any of them."

3. NORMALIZATION (Batch Norm, Layer Norm, etc.)
   ─────────────────────────────────────────────
   Control the distribution of activations at each layer.
   Prevents internal covariate shift.
   Reduces sensitivity to initialization and learning rate.
   
   "Keep the signal well-behaved at every layer."
```

These three families are not redundant — they solve different aspects of the overfitting and training instability problem. Modern networks use all three simultaneously.

---

### 9.2 L2 Regularization (Weight Decay)

The most common parameter constraint. Add a penalty proportional to the sum of squared weights to the loss function.

```
L2 REGULARIZED LOSS
====================

  L_reg(θ) = L(θ) + (λ/2) · Σ wᵢ²
                              ↑
                         L2 penalty term

Where:
  L(θ)   = original loss (e.g., cross-entropy)
  λ       = regularization strength (hyperparameter, e.g., 1e-4)
  Σ wᵢ²  = sum of squared weights (L2 norm squared)
  (λ/2)   = the 1/2 is a convenience (cancels the 2 in the derivative)

GRADIENT:
  ∂L_reg/∂w = ∂L/∂w + λ·w

WEIGHT UPDATE (SGD):
  w ← w - η·(∂L/∂w + λ·w)
    = w - η·∂L/∂w - η·λ·w
    = w·(1 - η·λ) - η·∂L/∂w
         ↑
    "weight decay" — each step, weights shrink by factor (1-ηλ)
    This is why L2 regularization is called weight decay.

STATISTICAL INTERPRETATION:
  L2 regularization = MAP estimation with a Gaussian prior on weights:
    P(w) = N(0, 1/λ)    (prior: weights should be near zero)
    Minimizing L_reg = maximizing log P(data|θ) + log P(θ)
                     = MLE + log-prior
                     = MAP estimation

  Assumption encoded: weights are drawn from a Gaussian centered at 0.
  Large weights are unlikely under this prior → penalized.
```

**Effect on the loss landscape:**

```
WITHOUT L2:                    WITH L2:
  Elongated, ravine-like         More spherical bowl
  Loss                           Loss
  │  ╲                           │    ╲
  │   ╲─────────── w₁            │      ───────── w₁
  │                              │
  (anisotropic — different       (more isotropic — similar
   curvature in each direction)   curvature in each direction)

L2 adds λ to the curvature in every direction equally.
This "rounds out" the loss landscape, making optimization easier.
This is an additional benefit beyond regularization.

Typical values:
  λ = 1e-4 to 1e-2  (too large: underfitting; too small: no effect)
  For Adam: λ_AdamW = 0.01 to 0.1  (higher due to adaptive scaling)
```

---

### 9.3 L1 Regularization (Lasso)

Add a penalty proportional to the sum of absolute values of weights.

```
L1 REGULARIZED LOSS
====================

  L_reg(θ) = L(θ) + λ · Σ |wᵢ|

GRADIENT:
  ∂L_reg/∂w = ∂L/∂w + λ·sign(w)

  Where sign(w) = +1 if w > 0,  -1 if w < 0,  0 if w = 0

WEIGHT UPDATE (SGD):
  w ← w - η·∂L/∂w - η·λ·sign(w)

  For positive w: always subtract η·λ  → push toward zero
  For negative w: always add η·λ       → push toward zero

KEY PROPERTY: L1 produces SPARSE solutions.
  Weights are pushed all the way to exactly 0, not just near 0.
  With L2: weights shrink proportionally (large weights shrink more)
           → no weight ever reaches exactly 0
  With L1: constant force toward 0 regardless of weight magnitude
           → small weights get zeroed out completely

GEOMETRIC INTUITION:
  The constraint set of L1 (||w||₁ ≤ C) is a diamond (in 2D).
  The loss function's minimum is at some point in weight space.
  The constrained solution is where the loss ellipse first touches
  the diamond — and diamonds have CORNERS on the axes.

  L1 constraint (diamond):         L2 constraint (circle):
        w₂                               w₂
         │   ◆                            │   ○
         │  ◆ ◆                           │  ○ ○
         │ ◆   ◆                          │ ○   ○
    ─────◆─────◆───── w₁             ────○─────○──── w₁
         │ ◆   ◆                          │ ○   ○
         │  ◆ ◆                           │  ○ ○
         │   ◆                            │   ○

  Loss ellipse likely hits a corner of diamond → sparse solution.
  Loss ellipse hits circle anywhere → generally non-sparse solution.

STATISTICAL INTERPRETATION:
  L1 = MAP with a Laplace prior on weights:
    P(w) = (λ/2) · e^(-λ|w|)
  Laplace distribution has heavier tails → more weight on zero.

USE CASES:
  L1: feature selection (want to find which inputs matter), interpretable models
  L2: smooth shrinkage, default for neural networks
  Both: Elastic Net = αL1 + (1-α)L2  (combines both properties)
```

---

### 9.4 L1 vs L2: Numerical Comparison

```
EXPERIMENT: 4 weights, L = 0 (already at loss minimum), update weights
with regularization only. η = 0.1, λ = 0.5.

Initial weights: w = [2.0, 0.5, 0.1, -1.5]

─────────────────────────────────────────────────────────────────────
L2 UPDATE: w ← w - η·λ·w = w·(1 - 0.1×0.5) = w·0.95
─────────────────────────────────────────────────────────────────────

After 1 step:  [2.0×0.95, 0.5×0.95, 0.1×0.95, -1.5×0.95]
             = [1.900,    0.475,    0.095,    -1.425]

After 10 steps: w × 0.95¹⁰ = w × 0.5987
             = [1.197, 0.299, 0.060, -0.898]

After 100 steps: w × 0.95¹⁰⁰ = w × 0.00592
             = [0.012, 0.003, 0.001, -0.009]

Observation: ALL weights shrink proportionally toward zero.
None reach exactly zero. Large weights shrink faster in absolute
terms but same fraction. No sparsity.

─────────────────────────────────────────────────────────────────────
L1 UPDATE: w ← w - η·λ·sign(w)   (step = 0.1×0.5 = 0.05)
─────────────────────────────────────────────────────────────────────

After 1 step:  [2.0-0.05, 0.5-0.05, 0.1-0.05, -1.5+0.05]
             = [1.950,    0.450,    0.050,    -1.450]

After 10 steps:
  w₁ = 2.0 - 10×0.05 = 1.500   (still large, moving slowly)
  w₂ = 0.5 - 10×0.05 = 0.000   ← REACHED ZERO at step 10!
  w₃ = 0.1 - 2×0.05  = 0.000   ← REACHED ZERO at step 2!
  w₄ = -1.5 + 10×0.05 = -1.000

After more steps:
  w₁ still decreasing at constant rate 0.05/step
  w₂, w₃ stay at exactly 0  ← SPARSE

L1 produces exact zeros. Small weights are killed quickly.
Large weights survive longer. → Feature selection behavior.
```

---

### 9.5 Dropout

Dropout (Srivastava et al., 2014) is structurally the simplest and conceptually the most elegant regularizer. During training, randomly set each neuron's output to zero with probability p.

```
DROPOUT — FORWARD PASS
========================

Training mode:
  For each neuron j in a layer:
    mⱼ ~ Bernoulli(1-p)    (mask: 1 with prob 1-p, 0 with prob p)
    aⱼ_dropped = mⱼ · aⱼ  (zero out the neuron if mask=0)

  In practice, use "inverted dropout" (the standard):
    aⱼ_dropped = (mⱼ / (1-p)) · aⱼ

  The 1/(1-p) scaling ensures:
    E[aⱼ_dropped] = (1-p) · (aⱼ/(1-p)) + p·0 = aⱼ

  → Expected output during training = output without dropout.
  → No scaling adjustment needed at inference.

Inference mode:
  No dropout. Use all neurons.
  No scaling needed (inverted dropout already handles it).

DIAGRAM:
  Without dropout:        With dropout (p=0.5):
  
  x₁ ──► h₁ ──► ...      x₁ ──► h₁ ──✗ (zeroed)
  x₂ ──► h₂ ──► ...      x₂ ──► h₂ ──► ...
  x₃ ──► h₃ ──► ...      x₃ ──► h₃ ──✗ (zeroed)
  x₄ ──► h₄ ──► ...      x₄ ──► h₄ ──► ...

  Different neurons zeroed each forward pass (random each time).

Where:
  p     = dropout probability (probability of zeroing a neuron)
  1-p   = keep probability
  Typical p = 0.5 for fully connected layers
            = 0.1-0.3 for convolutional layers (less needed)
            = 0.1 for transformer attention (too aggressive otherwise)
```

**Why dropout works — three complementary views:**

```
VIEW 1: ENSEMBLE OF NETWORKS
==============================
With n neurons and dropout probability p, there are 2ⁿ possible
subnetworks (each neuron either present or absent).
Dropout trains all 2ⁿ subnetworks simultaneously, sharing weights.
At inference, using all neurons ≈ averaging the predictions of
all 2ⁿ subnetworks. Ensembles always outperform individuals.

For n=1000: 2¹⁰⁰⁰ subnetworks — an astronomical implicit ensemble.

VIEW 2: PREVENTING CO-ADAPTATION
==================================
Without dropout, neurons can co-adapt: neuron A learns to fix
the mistakes of neuron B, which learns to fix the mistakes of C.
This creates a fragile dependency chain.

With dropout: neuron A cannot rely on B or C (they might be zeroed).
A must learn a feature that is independently useful.
This forces diverse, redundant representations.

VIEW 3: NOISE INJECTION
=========================
Dropout adds multiplicative noise to activations.
Noise injection is a classical regularization technique.
It prevents the network from fitting noise in the training data
because the training signal itself is noisy.
Consistent patterns survive the noise. Spurious correlations don't.
```

**The train/inference discrepancy without inverted dropout:**

```
WITHOUT INVERTED DROPOUT (naive, do not use):
  Training:  E[output] = (1-p) · a   (only 1-p fraction of neurons active)
  Inference: output = a              (all neurons active)
  
  Inference output is 1/(1-p) LARGER than what the network trained on!
  With p=0.5: inference output is 2× training output → catastrophic.
  
  Old fix: scale by (1-p) at inference ("standard dropout").
  Requires changing code at inference time — error-prone.

WITH INVERTED DROPOUT (correct, use this):
  Training:  scale up by 1/(1-p) during training.
             E[output] = (1-p) · a/(1-p) = a  ← matches inference
  Inference: no change needed.
  
  This is what PyTorch's nn.Dropout implements.
  Always use inverted dropout.
```

---

### 9.6 Dropout Variants

```
SPATIAL DROPOUT (for CNNs)
===========================
Standard dropout zeros individual neurons.
For convolutional feature maps, nearby pixels are correlated —
zeroing one pixel doesn't help much, neighbors still carry the info.

Spatial Dropout: zero entire CHANNELS (feature maps).
  Standard: zero random pixels in feature map [H × W × C]
  Spatial:  zero random channels (entire [H × W] slices)
  
  Much more effective for CNNs because it forces the network
  to learn that no single feature map is indispensable.
  PyTorch: nn.Dropout2d

DROPCONNECT
============
Instead of zeroing neuron outputs, zero random WEIGHTS.
  Standard Dropout: aⱼ_dropped = mⱼ · aⱼ   (mask on activations)
  DropConnect:      z = (M ⊙ W) · x         (mask on weights)
  
  Slightly stronger regularization than dropout.
  More expensive (must sample mask for every weight, not neuron).
  Rarely used in practice — marginal gain over dropout.

VARIATIONAL DROPOUT / CONCRETE DROPOUT
========================================
Make the dropout probability p a learned parameter.
  - Network learns how much to drop each neuron
  - Different neurons can have different dropout rates
  - Particularly useful for Bayesian deep learning
  - Implemented as a differentiable relaxation of the Bernoulli mask

ALPHA DROPOUT (for SELU activations)
======================================
SELU (Scaled ELU) activation self-normalizes activations to N(0,1).
Standard dropout breaks this normalization.
Alpha Dropout: sets dropped neurons to a specific negative value
(not zero) that preserves the mean and variance of SELU outputs.
```

---

### 9.7 Batch Normalization

Batch Normalization (Ioffe & Szegedy, 2015) is one of the most impactful papers in deep learning. It solves training instability at its root, not as an afterthought.

```
THE PROBLEM: INTERNAL COVARIATE SHIFT
======================================
As weights update, the distribution of each layer's inputs changes.
Layer 3 must continuously adapt to shifting distributions from layer 2,
which itself is adapting to layer 1's shifting distributions.
This is "internal covariate shift."

Consequence:
  - Each layer must use conservative learning rates (to not overreact)
  - Training is slow, unstable, sensitive to initialization
  - Very deep networks are essentially untrainable without BN

BATCH NORMALIZATION — ALGORITHM
=================================

Given: mini-batch B = {x₁, ..., xₘ} of activations for one layer

Step 1 — Compute batch statistics:
  μ_B = (1/m) Σᵢ xᵢ              (batch mean)
  σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²   (batch variance)

Step 2 — Normalize:
  x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)   (zero mean, unit variance)

  Where ε = 1e-5 (prevents division by zero)

Step 3 — Scale and shift (learnable):
  yᵢ = γ · x̂ᵢ + β

Where:
  μ_B   = mean of the current mini-batch (computed, not learned)
  σ²_B  = variance of the current mini-batch (computed, not learned)
  γ     = learned scale parameter (initialized to 1)
  β     = learned shift parameter (initialized to 0)
  ε     = small constant for numerical stability
  x̂ᵢ   = normalized activation
  yᵢ   = final output (normalized then re-scaled)

The γ and β allow the network to UNDO the normalization if needed.
If γ=σ_B and β=μ_B, the output = original unnormalized input.
The network learns the optimal normalization level, not us.
```

**Why γ and β are necessary:**

```
WITHOUT γ and β:
  All activations are forced to N(0,1).
  This destroys information. For example:
  - Sigmoid works best near 0 (good)
  - But what if the task requires activations near 2.0?
  - BN without γ/β makes this impossible.

WITH γ and β:
  Network can represent any distribution (by learning γ,β).
  The normalization is "soft" — it provides a stable starting point
  but the network can deviate if the task requires it.
  After training, γ and β can encode σ≠1 and μ≠0 for each layer.

CRITICAL: γ and β are LEARNED PARAMETERS.
  They appear in backprop. Gradients flow through them.
  Each layer has 2×C parameters for BN (C = number of channels/neurons).
  These are typically far fewer parameters than the weight matrix.
```

**Training vs. inference behavior:**

```
TRAINING:
  Use batch statistics (μ_B, σ²_B) — computed from current mini-batch.
  These vary from batch to batch (stochastic).
  This stochasticity acts as a regularizer (similar to dropout).

INFERENCE:
  Cannot use batch statistics (batch size might be 1, or different distribution).
  Use RUNNING STATISTICS accumulated during training:
    μ_running ← α·μ_running + (1-α)·μ_B   (exponential moving average)
    σ²_running ← α·σ²_running + (1-α)·σ²_B

  α = momentum (typically 0.1 in PyTorch, meaning 10% weight on new batch)

  At inference: x̂ = (x - μ_running) / √(σ²_running + ε)

  CRITICAL BUG: If you forget model.eval() before inference,
  PyTorch uses batch statistics instead of running statistics.
  For batch size 1: σ²_B = 0 → division by zero → NaN.
  For small batches: statistics are noisy → wrong predictions.
  Always call model.eval() before inference.
```

**Where to place BatchNorm:**

```
ORIGINAL PAPER (Ioffe & Szegedy):
  CONV → BN → Activation
  
  x → Conv → BN → ReLU → next layer
  BN is applied to pre-activations z (before ReLU).

COMMON ALTERNATIVE (empirically often better):
  x → Conv → ReLU → BN → next layer
  BN applied to post-activations.

  Research is mixed on which is better. In practice:
  - Pre-activation BN (original): more common in older papers
  - Post-activation BN: more common in some modern architectures
  - The difference is usually small (<0.5% accuracy)

FOR RESIDUAL NETWORKS (Chapter 11):
  x ──────────────────────────────► +
  │                                 │
  └──► Conv → BN → ReLU → Conv → BN ┘

  BN is placed before the residual addition, NOT after.
  This is important: if BN is after the addition, the gradient
  highway through the skip connection passes through BN,
  which can impede gradient flow.
```

---

### 9.8 Batch Norm Variants: Layer Norm, Instance Norm, Group Norm

The batch dimension is not always the right one to normalize across.

```
NORMALIZATION VARIANTS — WHAT GETS NORMALIZED
===============================================

Consider a tensor: [Batch (N), Channels (C), Height (H), Width (W)]

Batch Norm:    normalize across N (batch dimension) for each C,H,W position
               Computes separate μ,σ for each channel across all batch examples.
               ● Good for: CNNs with large batches
               ✗ Bad for:  small batches (noisy stats), RNNs (variable length sequences)

Layer Norm:    normalize across C,H,W for each N (example) independently
               Each example has its own μ,σ computed across all features.
               ● Good for: Transformers, RNNs, NLP (batch-independent)
               ✗ Bad for:  CNNs (not proven better than BN)

Instance Norm: normalize across H,W for each N,C independently
               Each example AND each channel has its own μ,σ.
               ● Good for: style transfer (normalizes spatial statistics)
               ✗ Bad for:  tasks where channel correlations matter

Group Norm:    normalize across H,W and groups of C channels
               Channels split into G groups; normalize within each group.
               ● Good for: small batches, object detection (FPN)
               ✗ Bad for:  nothing specifically — a good general choice

VISUAL COMPARISON (each shaded block = one normalization group):

  N (batch) ──────────────────────────────────────────────────►
  
  Batch Norm     Layer Norm     Instance Norm    Group Norm
  ┌───┬───┐      ┌───────┐      ┌───┬───┐        ┌──┬──┬──┬──┐
  │▓▓▓│▓▓▓│      │▓▓▓▓▓▓▓│      │▓▓▓│   │        │▓▓│▓▓│  │  │
C ├───┼───┤      ├───────┤      ├───┼───┤        ├──┼──┼──┼──┤
  │▓▓▓│▓▓▓│      │▓▓▓▓▓▓▓│      │   │▓▓▓│        │  │  │▓▓│▓▓│
  └───┴───┘      └───────┘      └───┴───┘        └──┴──┴──┴──┘
  Same color = normalized together

FOR TRANSFORMERS:
  Layer Norm is standard. No dependence on batch size.
  Can normalize sequences of different lengths.
  Pre-LN (before attention/FFN) is more stable than Post-LN.
  GPT-2/3 uses Pre-LN: x → LN → Attention → + x
  BERT uses Post-LN: x → Attention → + x → LN
  Pre-LN trains more stably but Post-LN sometimes performs better.
```

---

### 9.9 Worked Numerical Example: Batch Normalization

```
BATCH NORMALIZATION — FORWARD PASS
=====================================

Mini-batch of 4 examples, 1 neuron (scalar activations for clarity):
  Pre-BN activations: x = [2.0, 4.0, 6.0, 8.0]

Learnable parameters: γ = 1.5, β = 0.5

STEP 1: Compute batch statistics
  μ_B = (2.0 + 4.0 + 6.0 + 8.0) / 4 = 20/4 = 5.0
  
  σ²_B = [(2-5)² + (4-5)² + (6-5)² + (8-5)²] / 4
       = [9 + 1 + 1 + 9] / 4
       = 20/4 = 5.0

STEP 2: Normalize
  ε = 1e-5 ≈ 0 (negligible)
  
  x̂₁ = (2.0 - 5.0) / √5.0 = -3.0 / 2.236 = -1.342
  x̂₂ = (4.0 - 5.0) / √5.0 = -1.0 / 2.236 = -0.447
  x̂₃ = (6.0 - 5.0) / √5.0 =  1.0 / 2.236 =  0.447
  x̂₄ = (8.0 - 5.0) / √5.0 =  3.0 / 2.236 =  1.342

  x̂ = [-1.342, -0.447, 0.447, 1.342]

  Check: mean(x̂) = (-1.342-0.447+0.447+1.342)/4 = 0/4 = 0 ✓
         var(x̂)  ≈ 1.0 ✓

STEP 3: Scale and shift
  y₁ = 1.5 × (-1.342) + 0.5 = -2.013 + 0.5 = -1.513
  y₂ = 1.5 × (-0.447) + 0.5 = -0.671 + 0.5 = -0.171
  y₃ = 1.5 ×  0.447  + 0.5 =  0.671 + 0.5 =  1.171
  y₄ = 1.5 ×  1.342  + 0.5 =  2.013 + 0.5 =  2.513

  y = [-1.513, -0.171, 1.171, 2.513]

  Check: mean(y) = (-1.513-0.171+1.171+2.513)/4 = 2.0/4 = 0.5 = β ✓
         var(y)  = γ² × var(x̂) = 1.5² × 1.0 = 2.25 ✓

BACKWARD PASS (BN gradient — the tricky part):
  During backprop, gradients must flow through the normalization.
  The gradient ∂L/∂γ and ∂L/∂β are straightforward:
    ∂L/∂γ = Σᵢ (∂L/∂yᵢ) · x̂ᵢ
    ∂L/∂β = Σᵢ (∂L/∂yᵢ)

  The gradient ∂L/∂xᵢ is more complex (involves batch statistics):
    ∂L/∂xᵢ = (γ/√σ²+ε) · [∂L/∂ŷᵢ - (1/m)Σⱼ∂L/∂ŷⱼ
                           - (x̂ᵢ/m)·Σⱼ(∂L/∂ŷⱼ·x̂ⱼ)]

  KEY INSIGHT: The gradient of each xᵢ depends on ALL other xᵢ
  in the batch! This is the coupling that makes BN batch-dependent
  and that creates the regularization effect (the network can't
  overfit to individual examples because their BN output
  depends on the rest of the batch).
```

---

### 9.10 Combining Regularizers

In practice, multiple regularizers are used simultaneously:

```
TYPICAL MODERN SETUP
=====================

ResNet for image classification:
  ✓ L2 weight decay (λ = 1e-4) on all conv/linear weights
  ✓ Batch Normalization after every conv layer
  ✓ Data augmentation (random crop, flip, color jitter)
  ✓ Label smoothing (soft targets instead of hard 0/1)
  ✗ Usually NO dropout (BN provides sufficient regularization)

Transformer for NLP:
  ✓ AdamW weight decay (λ = 0.01-0.1)
  ✓ Layer Normalization (pre-LN)
  ✓ Dropout on attention weights (p = 0.1)
  ✓ Dropout on feedforward activations (p = 0.1)
  ✓ Gradient clipping (norm ≤ 1.0)
  ✓ Label smoothing
  ✗ Usually NO Batch Normalization (Layer Norm instead)

Feedforward network (tabular data):
  ✓ L2 weight decay (λ = 1e-3 to 1e-2)
  ✓ Dropout (p = 0.3-0.5) in hidden layers
  ✓ Batch Normalization between layers

Rule of thumb:
  DO NOT tune every regularizer simultaneously.
  Priority order: 1. architecture, 2. data augmentation,
                  3. BN/LN, 4. weight decay, 5. dropout.
  Add one regularizer at a time and observe its effect.
```

---

### 9.11 Why This Matters — What Breaks If You Get This Wrong

1. **Dropout at inference time.** If you forget `model.eval()`, PyTorch keeps dropout active during inference. Predictions are random — the same input gives different outputs every time you call `model(x)`. Loss on the test set will be higher and non-deterministic. This is the most common production bug with dropout.

2. **L2 regularization with Adam (not AdamW).** As shown in Chapter 8, Adam with L2 in the gradient doesn't implement true weight decay — the regularization is modulated by the adaptive learning rate. The result is weaker regularization than intended, with overfit models. Use AdamW explicitly.

3. **Batch Normalization with very small batch sizes.** With batch size 2 or 4, the batch statistics (mean and variance) are computed from only 2-4 samples — extremely noisy estimates. BN works poorly and can hurt performance. Solutions: increase batch size, switch to Layer Norm or Group Norm, or use Ghost BatchNorm (accumulate statistics over multiple forward passes).

4. **Applying dropout to the wrong layers.** Dropout on the input layer (directly on features) works poorly — it discards raw information before the network can process it. Dropout on the output layer of a classifier interferes with softmax normalization. Dropout belongs in the hidden layers, with the exception of transformers where attention dropout is standard.

5. **Using BN with RNNs.** Batch Normalization across the time dimension doesn't work for variable-length sequences (different positions have wildly different statistics). Use Layer Norm for any sequence model. This was an early confusion when BN was first introduced — it explains why LSTMs used LayerNorm while CNNs used BatchNorm.

---

### 9.12 Google/Apple-Level Interview Q&A

---

**Q1: "Explain why Batch Normalization acts as a regularizer, even though it was designed for training stability. What is the mechanism, and why does this mean you often don't need dropout when using BN?"**

*Why this is asked:* BN's regularization effect is a side effect, not a design goal — understanding this requires deep knowledge of both batch normalization's mechanics and what regularization means statistically. It also tests whether a candidate understands the relationship between different regularization techniques and can reason about when they are redundant.

**Answer:**

```
BN WAS DESIGNED FOR: preventing internal covariate shift.
BN ALSO PROVIDES:    regularization. Why?

MECHANISM 1: STOCHASTIC BATCH STATISTICS
==========================================

During training, BN normalizes each example using the batch's
mean and variance, not the example's own statistics.

For example xᵢ:
  x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

μ_B and σ²_B are random variables — they depend on which other
examples happen to be in the same mini-batch.

So x̂ᵢ is a random variable even for a fixed xᵢ!

In equation:
  x̂ᵢ = (xᵢ - (1/m)Σⱼxⱼ) / √((1/m)Σⱼ(xⱼ-μ_B)² + ε)

The 1/m terms mean that xᵢ's normalized form depends on x₁,...,xₘ.
Every time xᵢ appears in a different batch, it's normalized differently.

This is equivalent to adding noise to each activation:
  x̂ᵢ ≈ (xᵢ - μ_pop)/σ_pop  +  noise(xᵢ, batch composition)

The network cannot overfit to a specific representation of xᵢ
because that representation changes with every batch.
This is exactly the same mechanism as dropout — stochastic noise
prevents co-adaptation and memorization.

MECHANISM 2: GRADIENT NOISE
=============================

BN's gradient (∂L/∂xᵢ) depends on all examples in the batch
(derived in Section 9.9). This creates a "gradient coupling"
where the update for one example is influenced by others.

This coupling means the gradient is an approximately unbiased
but noisy estimate — similar to the noise from mini-batch SGD.
Additional gradient noise improves generalization (Neelakantan 2015).

WHY BN OFTEN REPLACES DROPOUT IN CNNs:
=========================================

Both work via noise:
  BN:      stochastic normalization (noise in activation scale/shift)
  Dropout: stochastic zeroing (multiplicative Bernoulli noise)

For CNNs, BN's noise is typically sufficient:
  Activations are already constrained to N(0,1) per channel.
  The spatial structure of conv features makes them robust.
  Adding dropout on top provides diminishing returns.

Empirical evidence:
  ResNet-50 with BN + no dropout: ~77% ImageNet top-1
  ResNet-50 with BN + dropout:    ~77-77.5% (marginal or no gain)
  ResNet-50 with dropout, no BN:  ~75% (worse — needs BN for stability)

The regularization from BN's stochasticity is sufficient for CNNs.

CONTRAST WITH TRANSFORMERS:
  Transformers use Layer Norm, not Batch Norm.
  Layer Norm normalizes within each example (no cross-example coupling).
  Therefore Layer Norm provides NO stochastic regularization
  (the normalization for xᵢ doesn't depend on other examples).
  → Dropout is NECESSARY for transformers.
  BERT dropout rate = 0.1 everywhere.
```

---

**Q2: "You're training a network and the training loss is decreasing but validation loss increases after epoch 5. List at least 5 different interventions you would try, ordered from least to most invasive, explain the mechanism of each, and explain how you would know if each one worked."**

*Why this is asked:* This is the most common practical scenario in all of ML engineering — overfitting. The question tests systematic debugging ability, knowledge of regularization tools, and the ability to diagnose which intervention is actually working versus which is just adding complexity. Google and Apple want engineers who can fix problems methodically, not by throwing everything at the wall.

**Answer:**

```
DIAGNOSIS FIRST:
  Training loss: ↓ (decreasing)
  Validation loss: ↑ after epoch 5 (increasing)
  → Classic overfitting. Model is memorizing training data.
  → Gap = (val loss - train loss) is the overfitting signal.

ORDER: least invasive → most invasive
```

**Intervention 1: Early Stopping**

```
Mechanism: Stop training when validation loss stops improving.
  Save the model checkpoint at epoch 5 (the best validation point).
  Don't train further regardless of training loss.

Implementation:
  best_val_loss = infinity
  patience = 10  # wait 10 epochs for improvement
  for epoch in range(max_epochs):
      train(model)
      val_loss = evaluate(model, val_set)
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          save_checkpoint(model)
          patience_counter = 0
      else:
          patience_counter += 1
          if patience_counter >= patience:
              break
  load_checkpoint(model)  # restore best model

How to know it worked: use the model from the best checkpoint.
  Val loss at the checkpoint < val loss at epoch 20.
  Zero cost: no regularization, no architecture change.
  Risk: may stop before reaching the true optimum.
```

**Intervention 2: Increase L2 Weight Decay**

```
Mechanism: Penalize large weights. Larger λ → simpler function
  → less memorization capacity → less overfitting.
  The model is forced to distribute learned information
  across many small weights rather than concentrating it.

If current λ=1e-4, try: 1e-3, then 5e-3, then 1e-2.

How to know it worked:
  Train/val gap should shrink.
  If val loss at the gap's minimum improves → working.
  If both train and val loss increase (together) → λ too large (underfitting).
  Target: train loss and val loss track closer together.
```

**Intervention 3: Add/Increase Dropout**

```
Mechanism: Randomly zero neurons during training.
  Forces redundant representations.
  If no dropout currently: add p=0.2-0.3 after each dense layer.
  If dropout exists: increase p by 0.1.

Where to add (for a feedforward network):
  Input → Dense → ReLU → [Dropout(p=0.3)] → Dense → ReLU → [Dropout(p=0.3)] → Output

How to know it worked:
  Validation loss at epoch 5 should be lower/plateau later.
  Watch for: if training loss becomes very noisy → p too high.
  If no effect: dropout alone may not be enough, or p is too small.
  Verify: model.train() before training, model.eval() before eval.
```

**Intervention 4: Data Augmentation**

```
Mechanism: Artificially expand the training set by creating
  modified versions of existing examples.
  The model sees more variation → harder to memorize specific examples.
  
  For images: random flips, crops, rotations, color jitter, cutout.
  For text: back-translation, synonym replacement, random deletion.
  For tabular: mixup (interpolate between two examples).

Implementation (image example):
  transform = transforms.Compose([
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomCrop(32, padding=4),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
  ])

How to know it worked:
  Training loss may become slightly higher (harder to memorize augmented data).
  Validation loss should improve.
  The train/val gap should shrink.
  If training loss becomes too high → augmentation too aggressive.
```

**Intervention 5: Reduce Model Capacity**

```
Mechanism: If the model has too many parameters relative to data,
  it can memorize. Reduce model size.
  
  Options: fewer layers, fewer neurons per layer, smaller embedding dims.
  Example: if current network is [512, 256, 128], try [256, 128, 64].

How to know it worked:
  Training loss will decrease more slowly (less capacity to fit).
  Validation loss should be lower and track training loss more closely.
  Risk: if model becomes too small, both losses increase together → underfitting.
  
  The bias-variance tradeoff: you're explicitly trading variance for bias.
  Find the sweet spot with a capacity sweep.

DIAGNOSIS MATRIX:
  
  Intervention │ If it works (mechanism):      │ If it fails (signal):
  ─────────────┼───────────────────────────────┼─────────────────────
  Early stop   │ Best val loss improves         │ Best epoch = epoch 1
  L2 decay     │ Val-train gap narrows          │ Both losses increase
  Dropout      │ Val loss plateau extends       │ Training loss too noisy
  Augmentation │ Val loss improves, train ↑ sl  │ Train loss too high
  Reduce size  │ Losses track closer            │ Both losses high (underfit)
```

---

**Q3: "Layer Normalization and Batch Normalization normalize different dimensions of the data. Derive mathematically why Layer Norm is preferred for transformers over Batch Norm, considering variable sequence lengths and the attention mechanism."**

*Why this is asked:* This question tests whether a candidate understands the deep connection between normalization, architecture design, and the statistical assumptions each normalization makes. It requires combining knowledge of transformers (attention mechanism), normalization math, and practical constraints (variable-length sequences). This level of analysis is expected from ML scientists at companies building foundation models.

**Answer:**

**Mathematical definitions:**

```
Batch Norm for a layer with C features, batch size N:
  For feature c:
    μᶜ = (1/N) Σₙ xₙᶜ             (mean across batch dimension)
    σᶜ = √[(1/N)Σₙ(xₙᶜ - μᶜ)²]   (std across batch dimension)
    x̂ₙᶜ = (xₙᶜ - μᶜ) / (σᶜ + ε)  (normalized)

  BN depends on N other examples in the batch.

Layer Norm for one example with C features:
  μ = (1/C) Σc xc              (mean across feature dimension)
  σ = √[(1/C)Σc(xc - μ)²]     (std across feature dimension)
  x̂c = (xc - μ) / (σ + ε)    (normalized per example)

  LN depends only on the current example — no cross-example coupling.
```

**Why BN fails for transformers — 3 reasons:**

```
REASON 1: VARIABLE SEQUENCE LENGTHS
=====================================

Transformer input: X ∈ ℝ^(N × T × d)
  N = batch size, T = sequence length, d = model dimension

  Sequence 1: T₁ = 10 tokens
  Sequence 2: T₂ = 100 tokens
  Padded to same length: T_max = 100 (with zero padding for seq 1)

  With BN, statistics computed over all (N × T) positions:
    μ = mean over N=32 examples × T=100 positions = 3200 values

  PROBLEM: padding tokens (zeros) are included in the statistics!
    μ is pulled toward 0 by the padded positions.
    σ² is inflated by the variance of real vs. padded values.
    Normalization is incorrect for both real and padded positions.

  With LN, statistics computed independently per example:
    For each token position t, normalize over d features.
    Padding doesn't affect other positions' normalization.
    Works correctly regardless of sequence length.

REASON 2: AUTOREGRESSIVE GENERATION (BATCH SIZE = 1)
=====================================================

During generation (GPT decoding), we often run with batch size N=1.
  BN with N=1: μ = x₁, σ² = 0 → x̂ = (x-x)/0 → undefined!
  
  Even if we use running statistics:
    Running stats were computed during training with large batches.
    At inference with N=1, the distribution may differ.
    BN can give wildly wrong normalization.

  LN with N=1: normalizes over d=768 features → always well-defined.
  Works identically for N=1 and N=512.

REASON 3: ATTENTION MECHANISM COUPLING
========================================

Self-attention computes:
  Attention(Q,K,V) = softmax(QKᵀ/√d) · V

  Q = XW_Q,  K = XW_K,  V = XW_V  (projections of X)

If we apply BN to X (normalize across the batch):
  The normalized Q, K, V for token i in example n depend on
  ALL other examples in the batch through μ_B and σ_B.

  This creates an unintended cross-example dependency:
    "What does token i in example 1 attend to?" now depends on
    what tokens appear in examples 2, 3, ..., N.

  This breaks the independence of examples — the model at inference
  (where we run one example) behaves differently than at training
  (where statistics mix across examples).

  LN: each example is normalized independently.
    Q, K, V are computed from X normalized by that example's own stats.
    No cross-example contamination.
    Training and inference behavior are identical.

PROOF THAT LN IS INVARIANT TO BATCH COMPOSITION:
  Let f be a transformer with Layer Norm.
  f(x₁, batch={x₁,x₂}) = f(x₁, batch={x₁,x₃}) for any x₂,x₃.
  The output for x₁ is identical regardless of what else is in the batch.

  This is NOT true for Batch Norm:
  f_BN(x₁, batch={x₁,x₂}) ≠ f_BN(x₁, batch={x₁,x₃}) in general.
  BN creates a data-dependent coupling between examples.
  At inference with N=1, this coupling disappears → train/test mismatch.

CONCLUSION:
  For any architecture where:
    (a) batch size varies widely, OR
    (b) sequence lengths vary, OR
    (c) train/inference batch size differs significantly
  → Use Layer Norm.

  Batch Norm is ideal for CNNs processing fixed-size images in large batches,
  where the batch statistics are stable and meaningful.
```

---

*End of Chapter 9. Chapter 10 (CNNs) coming next.*
