# Chapter 4: Forward Propagation

---

### 4.1 The Plain-English Picture

Forward propagation is the act of passing data through a neural network from input to output — left to right, layer by layer — to produce a prediction. It is called "forward" because information flows in one direction: forward through the network. No feedback, no cycles, no looking back.

Every time you ask a neural network "what is this?" — whether that's a photo, a sentence, or a vector of numbers — it answers by running forward propagation. Forward propagation is inference. It is the thing the network actually *does* when deployed.

During training, forward propagation is only half the story (backpropagation follows, Chapter 6). But it is the foundational half. Everything in backpropagation depends on the values computed during the forward pass — they must be cached.

Think of forward propagation like an assembly line. Raw materials (input data) enter at one end. Each station (layer) performs a specific transformation — weighting, summing, applying a non-linearity. The finished product (a prediction) exits at the other end. Each station doesn't know or care what the final product will be used for; it just does its local transformation and passes the result to the next station.

The remarkable thing is how much expressiveness emerges from repeating this simple procedure — linear combination, then non-linearity — many times in sequence.

---

### 4.2 The Full Forward Pass: Notation

We define a network with L layers. Each layer l has:

```
NOTATION REFERENCE
==================

l           = layer index, l ∈ {1, 2, ..., L}
              (l=1 is first hidden layer, l=L is output layer)
              (l=0 denotes the input layer, no computation)

nˡ          = number of neurons in layer l

Wˡ          = weight matrix for layer l
              shape: [nˡ × nˡ⁻¹]
              Wˡᵢⱼ = weight from neuron j in layer (l-1)
                     to neuron i in layer l

bˡ          = bias vector for layer l
              shape: [nˡ × 1]

zˡ          = pre-activation vector for layer l
              shape: [nˡ × 1]
              zˡ = Wˡ aˡ⁻¹ + bˡ

aˡ          = post-activation (activation) vector for layer l
              shape: [nˡ × 1]
              aˡ = σˡ(zˡ)

σˡ(·)       = activation function at layer l
              (can differ per layer; often same for hidden layers)

a⁰ = x      = input vector (the data)
              shape: [n⁰ × 1] = [nᵢₙₚᵤₜ × 1]

ŷ = aᴸ      = output of the final layer (the prediction)
```

---

### 4.3 The Forward Pass Algorithm

```
FORWARD PROPAGATION ALGORITHM
==============================

Input:  x (one training example), parameters {Wˡ, bˡ} for l=1..L
Output: ŷ = aᴸ (prediction), cache of all {zˡ, aˡ}

Initialize:
  a⁰ ← x

For l = 1, 2, ..., L:
  Step 1 — Linear transform:
    zˡ = Wˡ · aˡ⁻¹ + bˡ

  Step 2 — Non-linear activation:
    aˡ = σˡ(zˡ)

  Step 3 — Cache:
    Store (zˡ, aˡ) for use in backpropagation

Return:
  ŷ = aᴸ                      ← the prediction
  cache = {z¹,a¹,...,zᴸ,aᴸ}  ← needed for backprop
```

**Why cache?** During backpropagation, the gradient of the loss with respect to Wˡ depends on aˡ⁻¹ (the activations of the previous layer). These values were computed during the forward pass. If you don't cache them, you have to recompute the entire forward pass for every layer during backprop — doubling computation. Caching trades memory for speed.

---

### 4.4 Full Network Diagram with Dimensions

```
EXAMPLE: 4 → 3 → 3 → 2 NETWORK
=================================

Layer:      Input (l=0)   Hidden1 (l=1)  Hidden2 (l=2)  Output (l=3)
Neurons:        4               3               3               2
Activation:    none           ReLU            ReLU          Softmax

  a⁰ ∈ ℝ⁴        a¹ ∈ ℝ³        a²  ∈ ℝ³       a³ ∈ ℝ²
  [x₁]           [h¹₁]          [h²₁]           [ŷ₁]
  [x₂]    W¹     [h¹₂]   W²     [h²₂]   W³      [ŷ₂]
  [x₃]  ──────►  [h¹₃]  ──────► [h²₃]  ──────►
  [x₄]

Weight matrix dimensions:
  W¹ ∈ ℝ³ˣ⁴    (3 neurons, each with 4 incoming weights)
  W² ∈ ℝ³ˣ³    (3 neurons, each with 3 incoming weights)
  W³ ∈ ℝ²ˣ³    (2 neurons, each with 3 incoming weights)

Bias vector dimensions:
  b¹ ∈ ℝ³
  b² ∈ ℝ³
  b³ ∈ ℝ²

Total parameters:
  W¹: 3×4 = 12    b¹: 3    → 15
  W²: 3×3 =  9    b²: 3    → 12
  W³: 2×3 =  6    b³: 2    →  8
  ─────────────────────────────
  Total:                      35
```

---

### 4.5 Vectorization: From One Example to a Batch

In practice, we never run forward propagation on one example at a time. We process a *batch* of m examples simultaneously. This is where the GPU earns its keep.

```
SINGLE EXAMPLE (vector form):
  zˡ = Wˡ aˡ⁻¹ + bˡ
  shape: [nˡ×1] = [nˡ×nˡ⁻¹] · [nˡ⁻¹×1] + [nˡ×1]

BATCH OF m EXAMPLES (matrix form):
  Stack all m input vectors as columns:
  A⁰ = [x⁽¹⁾ | x⁽²⁾ | ... | x⁽ᵐ⁾]   shape: [n⁰ × m]

  Then for each layer l:
  Zˡ = Wˡ · Aˡ⁻¹ + bˡ     ← bˡ broadcast across all m columns
  Aˡ = σˡ(Zˡ)              ← σ applied element-wise

  shape: [nˡ × m] = [nˡ × nˡ⁻¹] · [nˡ⁻¹ × m] + [nˡ × 1]
                                                    ↑
                                          broadcasts to [nˡ × m]

  This computes ALL m examples in ONE matrix multiply.
  A GPU can do this multiply in microseconds regardless of
  whether m=1 or m=512. This is why batch training is fast.

SHAPES AT EACH LAYER (example: 4→3→3→2, batch size m=32):
  A⁰: [4  × 32]
  Z¹: [3  × 32]    A¹: [3  × 32]
  Z²: [3  × 32]    A²: [3  × 32]
  Z³: [2  × 32]    A³: [2  × 32]   ← ŷ for all 32 examples
```

**Broadcasting explained:**

```
bˡ has shape [nˡ × 1].
Zˡ = Wˡ · Aˡ⁻¹ has shape [nˡ × m].

Adding [nˡ × 1] to [nˡ × m]:
  NumPy/PyTorch automatically replicates bˡ across all m columns.
  This is broadcasting — no actual memory copy occurs.
  Equivalent to: Zˡ = Wˡ · Aˡ⁻¹ + np.tile(bˡ, (1, m))
  but without the memory cost of tiling.
```

---

### 4.6 Computational Graph

The forward pass builds a **computational graph** — a directed acyclic graph (DAG) where nodes are operations and edges are tensors flowing between them. This graph is what automatic differentiation (autograd) traverses during backpropagation.

```
COMPUTATIONAL GRAPH for z² = W²·a¹ + b²,  a² = ReLU(z²)
==========================================================

  W²  ──────►┐
             ├──► [MatMul] ──► z²_pre ──►┐
  a¹  ──────►┘                           ├──► [Add] ──► z² ──► [ReLU] ──► a²
                                          │
  b²  ────────────────────────────────────┘

Each arrow is a tensor.
Each box is an operation node that:
  1. Computes its output during forward pass
  2. Knows how to compute its local gradient during backward pass

PyTorch builds this graph dynamically (define-by-run).
TensorFlow 1.x built it statically (define-then-run).
Modern TF/JAX support both.
```

**Why the graph matters:** Backpropagation is just the chain rule applied to this graph, in reverse. Every node stores its local Jacobian (or the information needed to compute it). During backprop, you traverse the graph backward, multiplying local gradients using the chain rule. If you understand the forward graph, you understand backprop (Chapter 6).

---

### 4.7 Worked Numerical Example: Complete Forward Pass

A complete, fully worked example with a 3-layer network (2→4→4→3), ReLU hidden layers, Softmax output.

```
═══════════════════════════════════════════════════════════════
NETWORK SPECIFICATION
═══════════════════════════════════════════════════════════════
Architecture:  2 → 4 → 4 → 3
Hidden layers: ReLU activation
Output layer:  Softmax (3-class classification)
Input:         x = [0.8, -1.2]
True label:    y = class 2  (one-hot: [0, 0, 1])

═══════════════════════════════════════════════════════════════
PARAMETERS (pretrained weights)
═══════════════════════════════════════════════════════════════

Layer 1 (2 → 4):
  W¹ = [[ 0.5,  0.3],
         [-0.4,  0.7],
         [ 0.2, -0.5],
         [ 0.8,  0.1]]     shape: [4 × 2]
  b¹ = [0.1, 0.0, -0.1, 0.2]

Layer 2 (4 → 4):
  W² = [[ 0.3, -0.2,  0.4,  0.1],
         [ 0.5,  0.3, -0.1,  0.2],
         [-0.2,  0.4,  0.3, -0.3],
         [ 0.1, -0.3,  0.2,  0.5]]   shape: [4 × 4]
  b² = [0.0, 0.1, -0.1, 0.0]

Layer 3 (4 → 3):
  W³ = [[ 0.4,  0.2, -0.3,  0.5],
         [-0.1,  0.5,  0.2, -0.4],
         [ 0.3, -0.2,  0.4,  0.1]]   shape: [3 × 4]
  b³ = [0.1, -0.1, 0.0]

═══════════════════════════════════════════════════════════════
LAYER 1 FORWARD PASS
═══════════════════════════════════════════════════════════════

a⁰ = x = [0.8, -1.2]

z¹ = W¹ · a⁰ + b¹

  z¹₁ = (0.5)(0.8) + (0.3)(-1.2) + 0.1
       = 0.40 - 0.36 + 0.10 = 0.14

  z¹₂ = (-0.4)(0.8) + (0.7)(-1.2) + 0.0
       = -0.32 - 0.84 + 0.00 = -1.16

  z¹₃ = (0.2)(0.8) + (-0.5)(-1.2) + (-0.1)
       = 0.16 + 0.60 - 0.10 = 0.66

  z¹₄ = (0.8)(0.8) + (0.1)(-1.2) + 0.2
       = 0.64 - 0.12 + 0.20 = 0.72

  z¹ = [0.14, -1.16, 0.66, 0.72]

Apply ReLU:  a¹ = max(0, z¹)

  a¹₁ = max(0,  0.14) =  0.14  ✓ active
  a¹₂ = max(0, -1.16) =  0.00  ✗ dead (negative → zeroed)
  a¹₃ = max(0,  0.66) =  0.66  ✓ active
  a¹₄ = max(0,  0.72) =  0.72  ✓ active

  a¹ = [0.14, 0.00, 0.66, 0.72]

  → 1 out of 4 neurons inactive (25% sparsity). Normal.

═══════════════════════════════════════════════════════════════
LAYER 2 FORWARD PASS
═══════════════════════════════════════════════════════════════

z² = W² · a¹ + b²

  z²₁ = (0.3)(0.14) + (-0.2)(0.00) + (0.4)(0.66) + (0.1)(0.72) + 0.0
       = 0.042 + 0.000 + 0.264 + 0.072 + 0.0
       = 0.378

  z²₂ = (0.5)(0.14) + (0.3)(0.00) + (-0.1)(0.66) + (0.2)(0.72) + 0.1
       = 0.070 + 0.000 - 0.066 + 0.144 + 0.1
       = 0.248

  z²₃ = (-0.2)(0.14) + (0.4)(0.00) + (0.3)(0.66) + (-0.3)(0.72) + (-0.1)
       = -0.028 + 0.000 + 0.198 - 0.216 - 0.1
       = -0.146

  z²₄ = (0.1)(0.14) + (-0.3)(0.00) + (0.2)(0.66) + (0.5)(0.72) + 0.0
       = 0.014 + 0.000 + 0.132 + 0.360 + 0.0
       = 0.506

  z² = [0.378, 0.248, -0.146, 0.506]

Apply ReLU: a² = max(0, z²)

  a²₁ = max(0,  0.378) = 0.378  ✓
  a²₂ = max(0,  0.248) = 0.248  ✓
  a²₃ = max(0, -0.146) = 0.000  ✗ dead
  a²₄ = max(0,  0.506) = 0.506  ✓

  a² = [0.378, 0.248, 0.000, 0.506]

═══════════════════════════════════════════════════════════════
LAYER 3 FORWARD PASS (Output)
═══════════════════════════════════════════════════════════════

z³ = W³ · a² + b³

  z³₁ = (0.4)(0.378) + (0.2)(0.248) + (-0.3)(0.000) + (0.5)(0.506) + 0.1
       = 0.1512 + 0.0496 + 0.0000 + 0.2530 + 0.1
       = 0.5538

  z³₂ = (-0.1)(0.378) + (0.5)(0.248) + (0.2)(0.000) + (-0.4)(0.506) + (-0.1)
       = -0.0378 + 0.1240 + 0.0000 - 0.2024 - 0.1
       = -0.2162

  z³₃ = (0.3)(0.378) + (-0.2)(0.248) + (0.4)(0.000) + (0.1)(0.506) + 0.0
       = 0.1134 - 0.0496 + 0.0000 + 0.0506 + 0.0
       = 0.1144

  z³ = [0.5538, -0.2162, 0.1144]   ← these are the logits

Apply Softmax: ŷ = softmax(z³)

  Stable softmax: subtract max first
  max(z³) = 0.5538

  z³ - max = [0.5538-0.5538, -0.2162-0.5538, 0.1144-0.5538]
           = [0.0000, -0.7700, -0.4394]

  Exponentiate:
    e^0.0000  = 1.0000
    e^-0.7700 = 0.4630
    e^-0.4394 = 0.6443

  Sum = 1.0000 + 0.4630 + 0.6443 = 2.1073

  Softmax:
    ŷ₁ = 1.0000 / 2.1073 = 0.4746
    ŷ₂ = 0.4630 / 2.1073 = 0.2197
    ŷ₃ = 0.6443 / 2.1073 = 0.3057

  ŷ = [0.4746, 0.2197, 0.3057]

═══════════════════════════════════════════════════════════════
RESULT
═══════════════════════════════════════════════════════════════

  Predicted probabilities:
    Class 0: 47.5%
    Class 1: 22.0%
    Class 2: 30.6%

  Prediction: Class 0 (highest probability)
  True label: Class 2

  → Prediction is WRONG. The network will receive a high loss
    and gradients will flow backward to update the weights
    (Chapter 6: Backpropagation).

  Check: 0.4746 + 0.2197 + 0.3057 = 1.0000 ✓
```

---

### 4.8 What Gets Cached and Why

```
CACHE CONTENTS AFTER FORWARD PASS
===================================

For each layer l, we store:
  ┌──────────┬───────────────────────────────────────────────┐
  │ Variable │ Needed for backprop because...                │
  ├──────────┼───────────────────────────────────────────────┤
  │ aˡ⁻¹    │ ∂L/∂Wˡ = δˡ · (aˡ⁻¹)ᵀ  — gradient of weights│
  │ zˡ      │ σ'(zˡ) needed for δˡ = δˡ⁺¹ · (Wˡ⁺¹)ᵀ ⊙ σ' │
  │ Wˡ      │ ∂L/∂aˡ⁻¹ = (Wˡ)ᵀ · δˡ — pass gradient back  │
  └──────────┴───────────────────────────────────────────────┘

From our example, the full cache is:
  a⁰ = [0.8, -1.2]             (the input itself)
  z¹ = [0.14, -1.16, 0.66, 0.72]
  a¹ = [0.14,  0.00, 0.66, 0.72]
  z² = [0.378, 0.248, -0.146, 0.506]
  a² = [0.378, 0.248,  0.000, 0.506]
  z³ = [0.5538, -0.2162, 0.1144]
  a³ = ŷ = [0.4746, 0.2197, 0.3057]

Memory cost: proportional to (network depth × batch size × layer width)
For large models this is significant. Gradient checkpointing trades
recomputation for memory: cache only every k-th layer, recompute
the rest during backprop.
```

---

### 4.9 Forward Pass in Code

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    # Numerically stable softmax
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_pass(X, parameters):
    """
    X:          input matrix, shape [n_input × m]
    parameters: dict with W1,b1,W2,b2,...,WL,bL
    Returns:    ŷ (predictions) and cache (all z's and a's)
    """
    cache = {}
    A = X                        # A⁰ = input
    cache['A0'] = X
    L = len(parameters) // 2    # number of layers (each layer has W and b)

    # Hidden layers: ReLU
    for l in range(1, L):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = W @ A + b            # linear combination
        A = relu(Z)              # ReLU activation
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

    # Output layer: Softmax
    W = parameters[f'W{L}']
    b = parameters[f'b{L}']
    Z = W @ A + b
    A = softmax(Z)
    cache[f'Z{L}'] = Z
    cache[f'A{L}'] = A

    y_hat = A
    return y_hat, cache

# Example usage:
parameters = {
    'W1': np.array([[ 0.5,  0.3],
                    [-0.4,  0.7],
                    [ 0.2, -0.5],
                    [ 0.8,  0.1]]),
    'b1': np.array([[0.1], [0.0], [-0.1], [0.2]]),
    # ... W2, b2, W3, b3 ...
}
X = np.array([[0.8], [-1.2]])   # single example, shape [2×1]
y_hat, cache = forward_pass(X, parameters)
# y_hat: predicted probabilities for each class
```

---

### 4.10 Depth vs. Width: A Forward Propagation Perspective

```
DEPTH vs. WIDTH TRADEOFF
=========================

Given a fixed parameter budget P, which is better?
  Option A: 1 wide hidden layer  (shallow & wide)
  Option B: many narrow layers   (deep & narrow)

Example: P ≈ 1000 parameters, 10 inputs, 1 output

Option A: 10 → 90 → 1
  W¹: 10×90 = 900    b¹: 90   W²: 90×1 = 90    b²: 1
  Total: 1081 params
  Depth: 2 layers

Option B: 10 → 10 → 10 → 10 → 1
  W¹: 100  W²: 100  W³: 100  W⁴: 10
  Biases: 10+10+10+1 = 31
  Total: 351 params
  Depth: 4 layers

Key results from the theory of circuit complexity:

1. Functions that require exponential width in a shallow network
   can be represented with polynomial depth in a deep network.
   Example: parity function on n bits requires O(2ⁿ) neurons
   in 1 hidden layer, but O(n) neurons in O(log n) layers.

2. Deep networks compose features hierarchically:
   Layer 1: edges
   Layer 2: corners (combinations of edges)
   Layer 3: shapes (combinations of corners)
   Layer 4: objects (combinations of shapes)
   A shallow network has to learn all of these simultaneously
   in one step — much harder.

3. In practice: go deeper before going wider.
   ResNet-50 (50 layers, 25M params) >> a single 25M-param layer.

INFORMATION FLOW VIEW:
  Each layer transforms the representation.
  Deep network: many small, composable transformations.
  Shallow network: one large transformation.
  The former generalizes better because simpler pieces
  are easier to learn and reuse.
```

---

### 4.11 Why This Matters — What Breaks If You Get This Wrong

1. **Shape mismatches.** The single most common error. If you confuse `[nˡ × nˡ⁻¹]` with `[nˡ⁻¹ × nˡ]`, your matrix multiply fails. Every layer's weight matrix must be `[output_size × input_size]`. Internalize this. The matrix `Wˡ` transforms a vector of size `nˡ⁻¹` to size `nˡ`, so it must have `nˡ` rows and `nˡ⁻¹` columns.

2. **Not caching intermediate values.** If you implement forward propagation without storing `zˡ` and `aˡ`, you cannot implement backpropagation. You'll have to recompute the entire forward pass for every layer during backprop — O(L) times slower. Always cache during the forward pass.

3. **Applying softmax in a hidden layer.** Softmax outputs sum to 1 across neurons. In a hidden layer this creates competition between neurons — one neuron activating forces others to suppress, destroying the ability to represent independent features. Softmax belongs only at the output of multi-class classifiers. Use ReLU/GELU in hidden layers.

4. **Forgetting to use stable softmax.** If logits are large (say, 100), `exp(100) = 2.7 × 10⁴³`, which overflows float32 (max ~3.4 × 10³⁸). You get `nan` in the output and training collapses silently. Always subtract the maximum logit before exponentiating. This is a one-line fix that prevents an insidious bug.

5. **Using Python loops over batch examples.** Running forward propagation with a loop `for x in batch: forward(x)` is 100–1000× slower than a batched matrix multiply. GPUs are built for large matrix operations, not sequential scalar operations. Vectorize everything.

---

### 4.12 Google/Apple-Level Interview Q&A

---

**Q1: "Walk me through the exact dimensions of every tensor in a forward pass for a fully connected network with input dimension 512, hidden layers of sizes [256, 128, 64], and 10 output classes, using a batch size of 32. Then tell me the total number of floating point multiplications required."**

*Why this is asked:* Dimension tracking is a daily skill for production ML engineers. Getting it wrong causes silent bugs (broadcasting can mask shape errors), inefficient memory allocation, and incorrect parameter counts. Apple and Google use this question to verify that a candidate can engineer reliably, not just conceptually understand networks.

**Answer:**

```
ARCHITECTURE: 512 → 256 → 128 → 64 → 10
BATCH SIZE: m = 32

TENSOR DIMENSIONS
=================

Layer 1 (512 → 256):
  W¹:  [256 × 512]     b¹: [256 × 1]
  Z¹ = W¹·A⁰ + b¹:   [256 × 32]   (256 × 512 matmul with 512 × 32)
  A¹ = ReLU(Z¹):      [256 × 32]

Layer 2 (256 → 128):
  W²:  [128 × 256]     b²: [128 × 1]
  Z²:  [128 × 32]
  A²:  [128 × 32]

Layer 3 (128 → 64):
  W³:  [64 × 128]      b³: [64 × 1]
  Z³:  [64 × 32]
  A³:  [64 × 32]

Layer 4 (64 → 10):
  W⁴:  [10 × 64]       b⁴: [10 × 1]
  Z⁴:  [10 × 32]
  A⁴:  [10 × 32]   ← softmax output, ŷ

PARAMETER COUNT
===============
  W¹: 256×512  = 131,072   b¹: 256   → 131,328
  W²: 128×256  =  32,768   b²: 128   →  32,896
  W³:  64×128  =   8,192   b³:  64   →   8,256
  W⁴:  10×64  =     640   b⁴:  10   →     650
  Total: 173,130 parameters

FLOP COUNT (multiply-accumulate operations)
============================================
A matrix multiply [A × B] · [B × C] requires A×B×C multiplications
and A×(B-1)×C additions ≈ 2·A·B·C FLOPs total.
For simplicity, count multiply-add pairs (MACs), each = 1 FLOP:

Layer 1: [256 × 512] · [512 × 32] = 256 × 512 × 32 = 4,194,304
Layer 2: [128 × 256] · [256 × 32] = 128 × 256 × 32 = 1,048,576
Layer 3: [ 64 × 128] · [128 × 32] =  64 × 128 × 32 =   262,144
Layer 4: [ 10 ×  64] · [ 64 × 32] =  10 ×  64 × 32 =    20,480

Total MACs per forward pass: 5,525,504 ≈ 5.5 million

Note: a modern GPU (A100) does ~312 trillion FLOPs/second.
This network's forward pass takes: 5.5M / 312T ≈ 0.018 microseconds.
(In practice, overhead makes it ~50–500 microseconds for a single batch.)
```

---

**Q2: "What is the difference between model inference and model training in terms of the forward pass? What computations can you skip during inference, and why does this matter for deployment?"**

*Why this is asked:* Production ML systems spend 99% of their compute budget on inference, not training. Understanding the inference/training distinction is critical for optimization, mobile deployment, and latency-sensitive applications. This question tests practical engineering judgment.

**Answer:**

**The core difference:**

During **training**, the forward pass must:
1. Cache all intermediate activations `{zˡ, aˡ}` — needed for backpropagation
2. Compute dropout masks and apply them (Chapter 9)
3. Compute batch normalization statistics (running mean/variance) and normalize
4. Build the computational graph (in PyTorch's autograd)

During **inference**, the forward pass needs only:
1. Compute the output `ŷ` — no caching needed
2. No dropout (all neurons active, weights scaled by keep probability)
3. Batch normalization uses *stored* running statistics, not batch statistics
4. No gradient tracking — the computational graph is not built

**What you can skip at inference:**

```python
# TRAINING
model.train()
with torch.enable_grad():
    y_hat = model(x)        # builds graph, caches activations
    loss = criterion(y_hat, y)
    loss.backward()          # uses cache
    optimizer.step()

# INFERENCE — skip all of that
model.eval()                 # switches BN and Dropout to inference mode
with torch.no_grad():        # disables graph construction (saves ~50% memory)
    y_hat = model(x)         # no cache, no graph, just the compute
```

**Concrete benefits for deployment:**

```
Memory:
  Training:   Need to store all activations for backprop
              For ResNet-50: ~1 GB per batch of 32
  Inference:  Only need current layer's activations
              For ResNet-50: ~10 MB (50× reduction)

Speed:
  No gradient computation: ~2× faster
  No autograd overhead: additional ~10-30% faster
  Batch size = 1 possible (no BN recomputation issues at eval)

Optimizations only valid at inference:
  - Quantization: replace float32 with int8 (4× smaller, 2-4× faster)
  - Pruning: remove near-zero weights (no impact on backward pass)
  - Layer fusion: fuse Conv+BN+ReLU into one kernel (GPU optimization)
  - TorchScript / ONNX export: remove Python overhead entirely
  - KV-caching in transformers: reuse attention computations
```

**The BN subtlety:** Batch Normalization (Chapter 9) during training normalizes using the *current batch's* mean and variance. During inference, you must use the *population* mean and variance (exponential moving averages accumulated during training). If you forget to call `model.eval()`, your inference uses batch statistics — which are wrong for batch size 1 (variance is undefined) and noisy for small batches. This is a real, common production bug.

---

**Q3: "Explain why forward propagation through a very deep network (say, 1000 layers) is numerically unstable with standard initialization, even before we talk about backpropagation. What specifically goes wrong with the activations?"**

*Why this is asked:* This probes understanding of signal propagation — a subtle but critical concept that motivates batch normalization, residual connections, and careful weight initialization. Many candidates understand vanishing/exploding *gradients* but don't think about vanishing/exploding *activations* in the forward pass. This distinguishes deep understanding from surface knowledge.

**Answer:**

The problem is **activation explosion or collapse** — the magnitude of activations either grows to infinity or shrinks to zero as signals pass through many layers.

**The math:**

```
Consider a deep network with L layers, all weights initialized
from a normal distribution with variance σ²_w.

At layer l, ignoring activation functions for clarity:
  aˡ = Wˡ aˡ⁻¹

Variance of aˡ given aˡ⁻¹ (assuming iid weights, iid inputs):
  Var(aˡᵢ) = nˡ⁻¹ · σ²_w · Var(aˡ⁻¹ⱼ)
              ↑
         (sum of nˡ⁻¹ independent products)

After L layers:
  Var(aᴸ) = (nˡ⁻¹ · σ²_w)ᴸ · Var(a⁰)

Case 1: nˡ⁻¹ · σ²_w > 1  (e.g., σ²_w = 0.1, n = 100 → product = 10)
  Var(aᴸ) = 10ᴸ · Var(a⁰)
  After 10 layers: 10¹⁰ × initial variance → EXPLOSION
  Activations become astronomically large → overflow → nan

Case 2: nˡ⁻¹ · σ²_w < 1  (e.g., σ²_w = 0.001, n = 100 → product = 0.1)
  Var(aᴸ) = 0.1ᴸ · Var(a⁰)
  After 10 layers: 10⁻¹⁰ × initial variance → COLLAPSE
  Activations become effectively zero → all predictions identical

Critical condition for stable propagation:
  nˡ⁻¹ · σ²_w = 1
  σ²_w = 1 / nˡ⁻¹

This is exactly Xavier initialization! (Chapter 7)
```

**What you observe in practice:**

```
Exploding activations (σ²_w too large):
  Layer 1:   activations ~ N(0, 1)
  Layer 5:   activations ~ N(0, 10⁴)
  Layer 10:  activations ~ N(0, 10⁸)  → float32 overflow at ~3×10³⁸
  Layer 40:  nan everywhere

Collapsing activations (σ²_w too small):
  Layer 1:   activations ~ N(0, 1)
  Layer 5:   activations ~ N(0, 10⁻⁴)
  Layer 10:  activations ~ N(0, 10⁻⁸)  → underflow, all ≈ 0
  Layer 40:  all neurons output ~0, softmax outputs 1/K (uniform)

In both cases: network cannot learn. Loss is stuck.
Backprop also fails: gradients are computed from activations,
so if activations are 0 or inf, gradients are too.
```

**The solutions (preview of Chapter 7 and 9):**

1. **Xavier/Glorot initialization:** Set `σ²_w = 2/(nˡ⁻¹ + nˡ)` for sigmoid/tanh. Keeps variance constant through layers.
2. **He initialization:** Set `σ²_w = 2/nˡ⁻¹` for ReLU. Accounts for the fact that ReLU kills ~50% of neurons, halving the effective variance.
3. **Batch Normalization:** Explicitly normalizes activations to zero mean and unit variance after every layer. Makes initialization less critical — you can use almost any initialization and BN will correct it layer by layer.
4. **Residual connections:** Add skip connections `aˡ = F(aˡ⁻¹) + aˡ⁻¹`. Even if `F(·)` collapses, the identity path preserves signal magnitude. This is the key innovation of ResNet (Chapter 11).

---

*End of Chapter 4. Chapter 5 (Loss Functions) coming next.*
