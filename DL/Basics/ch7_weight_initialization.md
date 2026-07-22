# Chapter 7: Weight Initialization — FAANG Interview Master Notes
### 🔥 Boosted Edition: Master Notes + Full Interview Q&A Bank

> **How to use this document:** Nothing from the original chapter has been removed. Every original section, diagram, formula, derivation, and Q&A is intact below. On top of that, this edition adds: (1) a rapid-review cheat sheet at the top, (2) an expanded interview Q&A bank at the end of the chapter, (3) "rapid-fire flashcards" for last-minute review, and (4) a combined formula/pitfall sheet at the very end. Look for the 🆕 marker to spot everything that's new.

---

## 🆕 MASTER CHEAT SHEET — Chapter 7 at a glance

| Method | Formula | Best for | Year |
|---|---|---|---|
| Zero init | W = 0 | Never — symmetry problem | — |
| Naive random | W ~ N(0, 0.01) | Shallow nets only | — |
| LeCun | σ² = 1/nᵢₙ | Sigmoid (linear approx near 0) | 1998 |
| Xavier/Glorot | σ² = 2/(nᵢₙ+nₒᵤₜ) | Tanh, Sigmoid | 2010 |
| He/Kaiming | σ² = 2/nᵢₙ | ReLU, Leaky ReLU | 2015 |
| Orthogonal | W = orthogonal matrix (via SVD) | RNNs, very deep nets | — |

| Key fact | Detail |
|---|---|
| Core goal of init | Preserve activation variance AND gradient variance across layers |
| Why not zero? | Symmetry problem — all neurons in a layer stay identical forever |
| Why He needs factor 2 vs LeCun's 1 | ReLU zeroes ~half the signal: E[ReLU(z)²] = ½Var(z) |
| Xavier's blind spot | Assumes symmetric activation around 0 — wrong for ReLU |
| Bias default | 0, except: LSTM forget gate (→1-2), imbalanced-class output layer (→ log-odds), regression output (→ mean(y)) |
| Does BatchNorm remove need for good init? | Reduces sensitivity, doesn't eliminate it — first forward pass still vulnerable |
| Orthogonal init property | ‖Wx‖₂ = ‖x‖₂ exactly — ideal for RNN recurrent weights across many timesteps |
| NaN on first forward pass (no gradient step yet) | Always an init/architecture/numerics issue, never an optimizer/LR issue |
| Output layer's special role | Controls the *initial prediction* (matches base rate) — separate from hidden-layer signal propagation |

---

<a name="chapter-7"></a>
## Chapter 7: Weight Initialization

---

### 7.1 The Plain-English Picture

Before training begins, every weight in the network needs a starting value. This seems like a boring bookkeeping detail — just pick some small numbers and get on with it. It is not. Weight initialization is one of the most consequential decisions in training a deep network, and getting it wrong can make training impossible regardless of how good your architecture, optimizer, or data are.

Here is the core tension: weights cannot all be zero (symmetry problem, Chapter 2), but they also cannot be arbitrarily large or small. Too large and activations explode through the forward pass. Too small and activations collapse to zero. Either way, gradients during backpropagation follow the same fate — exploding or vanishing — and training stalls.

The goal of weight initialization is to put the network in a state where signals can flow freely through it from day one. Specifically, we want:

```
1. Activations have reasonable magnitude across all layers
   (neither exploding nor collapsing)

2. Gradients have reasonable magnitude across all layers
   (neither exploding nor vanishing)

3. Symmetry is broken
   (each neuron starts different, so they specialize)
```

The solution is principled randomness: draw weights from a distribution whose variance is carefully calibrated to the network architecture. The question is: what variance?

Think of it like a telephone game. A message (signal) is passed from person to person (layer to layer). If each person randomly amplifies the message, the last person hears shouting. If each person randomly muffles it, the last person hears nothing. Good initialization ensures each person passes the message at roughly the same volume as they received it — signal preserved, neither amplified nor attenuated.

---

### 7.2 Why Initialization Matters: The Variance Analysis

Let's derive exactly what happens to signal variance as it passes through a layer. This derivation directly gives us the correct initialization.

```
SETUP
=====
Consider one layer: z = Wx + b
  x ∈ ℝⁿ       input vector, n = fan-in (number of inputs)
  W ∈ ℝᵐˣⁿ     weight matrix
  z ∈ ℝᵐ       pre-activation output

Assumptions (standard):
  - Inputs xᵢ are i.i.d. with E[xᵢ] = 0, Var(xᵢ) = σ²_x
  - Weights Wᵢⱼ are i.i.d. with E[Wᵢⱼ] = 0, Var(Wᵢⱼ) = σ²_w
  - Weights and inputs are independent of each other

Variance of one output neuron zₖ = Σⱼ Wₖⱼ xⱼ:

  Var(zₖ) = Σⱼ Var(Wₖⱼ xⱼ)            [independence]
           = Σⱼ [E[W²ₖⱼ] · E[x²ⱼ]]    [independence of W and x]
           = Σⱼ [Var(Wₖⱼ) · Var(xⱼ)]  [since means are zero]
           = n · σ²_w · σ²_x

CRITICAL RESULT:
  Var(z) = n · σ²_w · Var(x)

  For variance to be PRESERVED through this layer:
    Var(z) = Var(x)
    n · σ²_w · Var(x) = Var(x)
    n · σ²_w = 1
    σ²_w = 1/n

This is LeCun initialization (1998): σ²_w = 1/fan_in
The correct weight variance is 1 divided by the number of inputs.
```

---

### 7.3 Xavier / Glorot Initialization

Xavier Glorot and Yoshua Bengio (2010) extended the analysis to consider both the forward pass (activations) and the backward pass (gradients) simultaneously.

```
GLOROT ANALYSIS
===============

Forward pass requirement (preserve activation variance):
  σ²_w = 1/nᵢₙ    where nᵢₙ = fan-in

Backward pass requirement (preserve gradient variance):
  The gradient ∂L/∂xⱼ = Σₖ Wₖⱼ · δₖ
  By symmetric analysis:
  σ²_w = 1/nₒᵤₜ   where nₒᵤₜ = fan-out

These two conditions conflict (unless nᵢₙ = nₒᵤₜ).
Xavier compromise: average of the two conditions.

XAVIER / GLOROT INITIALIZATION:
  σ²_w = 2 / (nᵢₙ + nₒᵤₜ)

  Uniform version (equivalent, commonly used):
  W ~ Uniform(-a, a)  where a = √(6 / (nᵢₙ + nₒᵤₜ))

  Normal version:
  W ~ N(0, σ²)  where σ² = 2 / (nᵢₙ + nₒᵤₜ)

Where:
  nᵢₙ  = fan-in  = number of input connections to this layer
  nₒᵤₜ = fan-out = number of output connections from this layer

Example:
  Layer: 512 → 256
  nᵢₙ = 512, nₒᵤₜ = 256
  σ² = 2 / (512 + 256) = 2 / 768 = 0.0026
  σ  = 0.051

  Each weight drawn from N(0, 0.0026)
  Range: roughly [-0.15, 0.15] (within 3σ)

ASSUMPTION: works best with symmetric activations (tanh, sigmoid)
where the linear region approximation σ(z) ≈ z holds near z=0.
Does NOT account for ReLU's asymmetry (half of neurons zeroed).
```

---

### 7.4 He / Kaiming Initialization

Kaiming He et al. (2015) derived the correct initialization for ReLU networks.

```
HE ANALYSIS
===========

ReLU zeros out negative inputs: ReLU(z) = max(0, z)
This kills approximately half the neurons at any given input.
The effective variance passing through is halved:

  Var(a) = Var(ReLU(z)) ≈ (1/2) · Var(z)

(The factor 1/2 comes from the fact that only positive z values
survive. For a symmetric distribution around 0, half are positive.)

To compensate for this halving, we need to double σ²_w:

HE / KAIMING INITIALIZATION:
  σ²_w = 2 / nᵢₙ

  Normal version (most common):
  W ~ N(0, 2/nᵢₙ)

  Uniform version:
  W ~ Uniform(-a, a)  where a = √(6/nᵢₙ)

Example:
  Layer: 512 → 256, ReLU activation
  nᵢₙ = 512
  σ² = 2 / 512 = 0.00391
  σ  = 0.0625

  Each weight drawn from N(0, 0.00391)

WHY 2/nᵢₙ instead of 1/nᵢₙ (LeCun):
  Without compensation: Var(a^l) = (1/2) · Var(a^(l-1))
  In an L-layer network: Var(aᴸ) = (1/2)ᴸ · Var(a⁰)
  50-layer network: Var(a⁵⁰) = (1/2)⁵⁰ · Var(a⁰) ≈ 10⁻¹⁵ · Var(a⁰)
  → Complete collapse. Nothing reaches the output.

  With He init (2/nᵢₙ):
  Var(a^l) = (2/nᵢₙ) · nᵢₙ · (1/2) · Var(a^(l-1)) = Var(a^(l-1))
  → Variance preserved at every layer. Signal flows freely.

He init is the DEFAULT for any ReLU-family activation.
```

---

### 7.5 Comparison of Initialization Schemes

```
┌─────────────────┬──────────────────────────┬───────────────────────────┐
│ Method          │ Formula                  │ Best for                  │
├─────────────────┼──────────────────────────┼───────────────────────────┤
│ Zero            │ W = 0                    │ NEVER (symmetry problem)  │
│ Random (naive)  │ W ~ N(0, 0.01)           │ Shallow nets only         │
│ LeCun (1998)    │ σ² = 1/nᵢₙ              │ Sigmoid (linear approx)   │
│ Xavier (2010)   │ σ² = 2/(nᵢₙ+nₒᵤₜ)      │ Tanh, Sigmoid             │
│ He/Kaiming(2015)│ σ² = 2/nᵢₙ              │ ReLU, Leaky ReLU          │
│ Orthogonal      │ W = random orthog. matrix│ RNNs, very deep nets      │
│ MSRA            │ Same as He               │ Deep CNNs                 │
└─────────────────┴──────────────────────────┴───────────────────────────┘

PyTorch defaults:
  nn.Linear:  Kaiming uniform (He, uniform variant) — good for ReLU
  nn.Conv2d:  Kaiming uniform — good for ReLU

Override when using non-ReLU activations:
  nn.init.xavier_normal_(layer.weight)   # for tanh/sigmoid
  nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # for ReLU
```

---

### 7.6 Orthogonal Initialization

Used primarily for RNNs and very deep networks.

```
ORTHOGONAL INITIALIZATION
==========================

Procedure:
  1. Draw W ~ N(0, 1)  (random Gaussian matrix)
  2. Compute SVD: W = UΣVᵀ
  3. Use U (or V) as the initialization — it is orthogonal: UᵀU = I

Properties of orthogonal matrices:
  - All singular values = 1  (neither amplify nor attenuate)
  - ||Wx||₂ = ||x||₂        (preserve vector norm exactly)
  - Eigenvalues on unit circle (for square matrices)

Why this matters for RNNs:
  RNNs compute hₜ = σ(W·hₜ₋₁ + U·xₜ + b)
  The recurrent weight W is applied T times (one per timestep).
  If ||W|| > 1: hₜ explodes exponentially in T
  If ||W|| < 1: hₜ vanishes exponentially in T
  If W is orthogonal (||W|| = 1): hₜ stays bounded for any T

  Orthogonal init gives RNNs the best starting point for
  gradient flow across long sequences.

Limitation: only well-defined for square matrices.
For rectangular W ∈ ℝᵐˣⁿ, use U ∈ ℝᵐˣⁿ from the SVD.
```

---

### 7.7 Bias Initialization

```
BIAS INITIALIZATION
===================

Standard: b = 0  (always safe)

Reasoning: weights already break symmetry. Biases can start
at zero without causing the symmetry problem, because different
neurons receive different weight-scaled inputs.

EXCEPTIONS:

1. ReLU neurons: if W is initialized with zero mean, roughly
   half of neurons start dead (z < 0). Some practitioners
   initialize biases to small positive values (e.g., 0.01)
   to ensure neurons start active and receive gradients.
   In practice this rarely matters if weights are He-initialized.

2. Forget gate in LSTMs (Chapter 13): initialize to 1 or 2.
   This makes the LSTM remember by default at initialization,
   helping gradient flow through long sequences.

3. Output layer for class imbalance: if 90% of examples are
   class 0, initialize the output bias b₀ such that:
     σ(b₀) ≈ 0.9  →  b₀ = log(0.9/0.1) = log(9) ≈ 2.2
   This starts the network at the base rate, making early
   training faster (doesn't waste steps learning the prior).

4. Softmax output: initialize biases to 0. The network
   starts with uniform class predictions (1/K each), which
   is the right uninformative prior.
```

---

### 7.8 Worked Numerical Example: Initialization Impact

```
EXPERIMENT: 5-layer network, 100 neurons each, tanh activation
Compare: naive init (σ=1.0) vs Xavier init (σ=√(2/200)=0.1)

Input: x ~ N(0, 1), shape [100]

═══════════════════════════════════════════════════════════════
NAIVE INITIALIZATION (σ²_w = 1.0)
═══════════════════════════════════════════════════════════════

Layer 1:
  z¹ = W¹x,   each z¹ᵢ = Σⱼ W¹ᵢⱼxⱼ
  Var(z¹ᵢ) = 100 · 1.0 · 1.0 = 100
  Std(z¹) = 10

  a¹ = tanh(z¹):
    tanh(10) ≈ 1.0  (completely saturated!)
    tanh(-10) ≈ -1.0
    All activations ≈ ±1.0

Layer 2:
  z² = W²a¹,  but a¹ ≈ ±1 (binary-like)
  Var(z²) = 100 · 1.0 · Var(a¹) ≈ 100 · 1.0 · 1.0 = 100
  Std(z²) = 10 → a² ≈ ±1 (still saturated)

Layers 3, 4, 5: same story. All saturated.

Gradient analysis:
  tanh'(z) = 1 - tanh²(z)
  At z=10: tanh'(10) = 1 - (1.0)² = 0.0   ← zero gradient!
  Backprop: δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ tanh'(zˡ) ≈ 0

  ALL layers receive near-zero gradients from day 1.
  Network cannot train.

═══════════════════════════════════════════════════════════════
XAVIER INITIALIZATION (σ² = 2/(100+100) = 0.01, σ = 0.1)
═══════════════════════════════════════════════════════════════

Layer 1:
  Var(z¹) = 100 · 0.01 · 1.0 = 1.0
  Std(z¹) = 1.0

  a¹ = tanh(z¹):
    tanh is approximately linear for |z| ≤ 1
    tanh(1.0) = 0.762, tanh(-1.0) = -0.762
    Activations spread in [-0.8, 0.8] range — NOT saturated
    Var(a¹) ≈ Var(z¹) · [tanh'(0)]² ≈ 1.0 · 1.0 = 1.0

Layer 2:
  Var(z²) = 100 · 0.01 · 1.0 ≈ 1.0   ← preserved!
  Std(z²) = 1.0

Layers 3, 4, 5: Var ≈ 1.0 throughout.

Gradient analysis:
  tanh'(z) at z~N(0,1):
    E[tanh'(z)] ≈ 0.5  (reasonable, non-vanishing)
  Backprop gradients flow through all 5 layers at similar magnitude.
  Network can train effectively from step 1.

COMPARISON SUMMARY:
  Layer │ Naive std(a) │ Xavier std(a)
  ──────┼──────────────┼──────────────
    1   │   1.00       │   0.76
    2   │   1.00       │   0.72
    3   │   1.00       │   0.70
    4   │   1.00       │   0.69
    5   │   1.00       │   0.68
  Gradient │  ≈ 0      │  ≈ 0.3-0.5

  With naive init: std=1 looks fine but ALL neurons are
  saturated at ±1 (tanh(10)≈1). The variance is just the
  variance of {-1, +1}!

  With Xavier: std≈0.7 is in the active region of tanh.
  Gradients flow. Training works.
```

---

### 7.9 Modern Practices: Initialization + Batch Normalization

```
DOES BATCH NORM MAKE INITIALIZATION IRRELEVANT?
================================================

BatchNorm (Chapter 9) normalizes activations to N(0,1) after
each layer, regardless of weight values. This might suggest
initialization doesn't matter anymore.

Partially true, partially false:

TRUE:
  BatchNorm dramatically reduces sensitivity to initialization.
  A network with BatchNorm can train with naive σ=1.0 init
  that would fail without BatchNorm.
  BatchNorm is a form of "initialization rescue."

FALSE (initialization still matters because):

1. First forward pass before BN statistics are computed:
   If weights are astronomically large, first forward pass
   overflows before BN can normalize it.

2. Training stability in early steps:
   Even with BN, extremely bad init (e.g., σ=100) causes
   unstable loss in the first few steps before BN adapts.

3. Networks WITHOUT BatchNorm:
   Transformers (BERT, GPT) use LayerNorm which is less
   robust than BatchNorm. Initialization matters more.
   GPT uses σ = 0.02 (small constant) throughout.
   ResNets scale σ by 1/√(2L) for the last layer of each
   residual block, where L = number of residual blocks.

4. Convergence speed:
   Even if BN prevents catastrophic failure, good init
   converges faster (fewer steps to reach same performance).

PRACTICAL RULE:
  With BatchNorm:    Xavier or He init, either works fine.
  Without BatchNorm: He init for ReLU, Xavier for tanh/sigmoid.
  Always:            Never initialize all weights to the same value.
```

---

### 7.10 Why This Matters — What Breaks If You Get This Wrong

1. **Zero initialization.** Covered in Chapter 2 but worth repeating here: every neuron in a layer computes the same function forever. A 256-neuron layer with zero init is equivalent to a 1-neuron layer. Capacity is destroyed before training begins.

2. **Initialization too large (σ = 1.0 with ReLU/tanh).** Activations explode or saturate from the very first forward pass. Gradients are zero from step one. Loss stays at its initial value no matter how many epochs you train. This is frequently mistaken for a "learning rate problem" — but reducing the learning rate does nothing when gradients are already zero.

3. **Initialization too small (σ = 0.0001).** Activations collapse to zero. All neurons output nearly zero. Gradients are approximately zero (since ∂L/∂Wˡ = δˡ · (aˡ⁻¹)ᵀ and aˡ⁻¹ ≈ 0). Loss is stuck. Identical failure mode to initialization too large, different root cause.

4. **Using Xavier init with ReLU.** Xavier assumes activations are symmetric around zero (valid for tanh). ReLU kills all negative activations — the distribution is not symmetric. Xavier underestimates the variance needed to compensate for ReLU's zeroing. Result: activations gradually shrink through layers, gradient vanishes. Use He init for ReLU.

5. **Forgetting to initialize the output bias for imbalanced datasets.** If 99% of your data is class 0, the network must first learn the base rate (output 0.99 for class 0) before it can learn anything meaningful. This wastes potentially thousands of gradient steps. A properly initialized output bias starts the network at the base rate for free — training can immediately focus on the discriminative signal.

---

### 7.11 Google/Apple-Level Interview Q&A

---

**Q1: "Derive the He initialization formula from first principles, including the factor of 2 that distinguishes it from LeCun initialization. Why does that factor of 2 specifically arise from ReLU?"**

*Why this is asked:* Initialization derivations are a litmus test for mathematical depth. Anyone can look up "use He init for ReLU" — but explaining *why* the factor of 2 exists requires understanding the variance of a half-normal distribution, which is non-trivial. This level of derivation is expected from research engineers at Google Brain, DeepMind, or Apple ML Research.

**Answer:**

```
GOAL: find σ²_w such that Var(aˡ) = Var(aˡ⁻¹)
(signal variance preserved through one ReLU layer)

SETUP:
  zⱼ = Σᵢ Wⱼᵢ aᵢ   (pre-activation for neuron j)
  aⱼ = ReLU(zⱼ) = max(0, zⱼ)

STEP 1: Compute Var(zⱼ)
  aᵢ (previous layer output): E[aᵢ] = 0? 
  
  Wait — this is the subtlety. Previous layer also uses ReLU,
  so aᵢ = ReLU(zᵢ) ≥ 0. The mean is NOT zero!
  
  However, He et al. account for this by analyzing the second
  moment instead of the variance. Alternatively, note that if
  we initialize W symmetrically around 0, the distribution of
  Wⱼᵢ · aᵢ is symmetric around 0 in sign even if aᵢ > 0,
  because Wⱼᵢ can be positive or negative with equal probability.
  
  Under the assumption that Wⱼᵢ are symmetric and aᵢ is independent:
  E[Wⱼᵢ · aᵢ] = E[Wⱼᵢ] · E[aᵢ] = 0 · E[aᵢ] = 0
  
  Var(zⱼ) = nᵢₙ · E[W²ⱼᵢ] · E[a²ᵢ]
           = nᵢₙ · σ²_w · E[a²ᵢ]

STEP 2: Compute E[a²ᵢ] for ReLU
  aᵢ = ReLU(zᵢ) = max(0, zᵢ)
  
  Assume zᵢ ~ N(0, σ²_z) (symmetric around 0):
  
  E[a²ᵢ] = E[ReLU(zᵢ)²]
          = E[zᵢ² · 1(zᵢ > 0)]        [since ReLU²(z) = z² when z>0, 0 otherwise]
          = ∫₀^∞ z² · (1/√2πσ²_z) · e^(-z²/2σ²_z) dz
  
  By symmetry of the Gaussian:
  E[zᵢ²] = ∫_{-∞}^{∞} z² · p(z) dz = σ²_z
  
  The integrand z² · p(z) is symmetric (even function × even function):
  ∫₀^∞ z² · p(z) dz = (1/2) · ∫_{-∞}^{∞} z² · p(z) dz = (1/2) σ²_z
  
  Therefore:
  E[a²ᵢ] = (1/2) σ²_z = (1/2) Var(zᵢ)

  THE FACTOR OF 2: ReLU discards the negative half of the
  distribution, exactly halving the second moment. This is
  purely a consequence of the step-function nature of ReLU.

STEP 3: Propagate through the layer
  Var(z^l) = nᵢₙ · σ²_w · E[(a^(l-1))²]
            = nᵢₙ · σ²_w · (1/2) · Var(z^(l-1))

STEP 4: Set condition for variance preservation
  We want Var(z^l) = Var(z^(l-1)):
  nᵢₙ · σ²_w · (1/2) = 1
  σ²_w = 2/nᵢₙ   ∎

This is He initialization. The factor of 2 comes directly
from E[ReLU(z)²] = (1/2)E[z²] — ReLU kills half the signal,
so we compensate by doubling the initial weight variance.

For Leaky ReLU with negative slope α:
  E[a²] = (1/2)(1 + α²) · Var(z)
  σ²_w = 2 / ((1+α²) · nᵢₙ)
  
  When α=0: He init.
  When α=1: linear, gives σ²_w = 1/nᵢₙ (LeCun init).
```

---

**Q2: "You're training a 50-layer ResNet from scratch and you notice the loss is NaN after the first forward pass, before any gradient update. What went wrong, and walk through your exact diagnostic and fix process."**

*Why this is asked:* NaN on the first forward pass before any training is a pure initialization/architecture problem, not an optimization problem. This question tests ability to reason about numerical behavior in deep networks — a daily reality for engineers building large models at Apple or Google. The "before any gradient update" constraint is important: it eliminates learning rate, optimizer, and gradient issues.

**Answer:**

**Diagnosis — narrow down the cause:**

```
NaN on FIRST forward pass means:
  - NOT a gradient issue (no backward pass yet)
  - NOT a learning rate issue (no update yet)
  - MUST be one of:
    a) Weight overflow (too-large init → activations overflow float32)
    b) Numerical instability in an operation (log(0), 0/0, etc.)
    c) Input data contains NaN or inf
    d) Softmax overflow (unstable implementation)

STEP 1: Check input data
  assert not torch.isnan(X).any(), "Input contains NaN"
  assert not torch.isinf(X).any(), "Input contains inf"
  print(X.min(), X.max(), X.mean())   # sanity check range

STEP 2: Add NaN detection hooks to each layer
  def nan_hook(module, input, output):
      if torch.isnan(output).any():
          raise RuntimeError(f"NaN detected at {module}")
  
  for layer in model.modules():
      layer.register_forward_hook(nan_hook)
  
  model(X)  # run forward, hook will tell you EXACTLY which layer

STEP 3: Once the layer is identified, check its weights
  for name, param in model.named_parameters():
      print(f"{name}: min={param.min():.4f} max={param.max():.4f}")
      if torch.isnan(param).any():
          print(f"  !! NaN in weights: {name}")
```

**Most likely cause in a 50-layer ResNet:**

```
In a 50-layer network without BatchNorm, with naive init (σ=1.0):

Layer 1:  Var(z) = 512 × 1.0 × 1.0 = 512   → std ≈ 22.6
Layer 2:  Var(z) = 512 × 1.0 × 512 = 262K  → std ≈ 512
Layer 5:  Var(z) ≈ 512⁵ ≈ 3.4 × 10¹³
Layer 10: Var(z) ≈ 512¹⁰ ≈ 1.2 × 10²⁷
Layer 15: exceeds float32 max (3.4 × 10³⁸) → overflow → inf
Layer 16: inf × weight → NaN (∞ × 0 = NaN in IEEE 754)

First NaN typically appears around layer 15-20 with σ=1.0, n=512.
```

**Fix cascade (apply in order, stop when fixed):**

```
Fix 1: Use He initialization (immediate, no cost)
  torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
  
  With He: σ²_w = 2/512 ≈ 0.004, σ ≈ 0.063
  Layer 1:  Var(z) = 512 × 0.004 × 1.0 = 2.0 → std ≈ 1.4  ✓
  Layer 50: Var(z) ≈ 1.0 (preserved)  ✓

Fix 2: Add BatchNorm after each conv layer (if not already there)
  Normalizes activations to N(0,1) regardless of weight values.
  Acts as a safety net against any residual initialization issues.

Fix 3: Add residual connections (ResNet's core design)
  aˡ = F(aˡ⁻¹) + aˡ⁻¹
  Even if F(·) explodes slightly, the skip path aˡ⁻¹ is bounded.
  The sum is dominated by the stable term.
  Additional: He et al. (2015) recommend scaling the last layer
  of each residual block by 1/√(2L) for very deep networks.

Fix 4: Use gradient/activation clipping during first few steps
  with torch.no_grad():
      for p in model.parameters():
          p.clamp_(-1.0, 1.0)   # clip weights to reasonable range

Fix 5: Verify softmax is numerically stable
  Use F.cross_entropy(logits, y) NOT F.nll_loss(F.log_softmax(...))
  The fused version is more stable.

In practice: Fix 1 (He init) + Fix 2 (BatchNorm) solves 99% of
NaN-on-first-forward-pass issues in ResNets.
```

---

**Q3: "Explain why the output layer of a neural network sometimes requires special initialization different from the hidden layers, and give two concrete examples where this matters significantly."**

*Why this is asked:* This probes practical engineering knowledge beyond standard initialization recipes. Output layer initialization affects the initial loss value, training speed in early epochs, and convergence to good solutions — but it's rarely discussed in textbooks. Engineers who've trained production models know this matters; those who've only done tutorials don't.

**Answer:**

Hidden layer initialization controls signal propagation (Chapter 7's main topic). Output layer initialization controls the *starting prediction* — what the network outputs before seeing any data. The initial prediction determines the initial loss, which determines the initial gradient, which determines how fast the network learns from its first examples.

**Why the output layer is special:**

```
Hidden layers: we want σ²_w calibrated so variance is preserved.
               The "meaning" of activations doesn't matter much.

Output layer:  activations have SEMANTIC meaning:
               - Class probabilities (must sum to 1, be in [0,1])
               - Regression targets (should be in the range of y)
               - Log-probabilities (should be log(1/K) initially for K classes)

               A poorly initialized output layer makes the network
               start with a bad prior — it wastes training steps
               recovering to a sensible baseline instead of learning
               the true signal.
```

**Example 1: Multi-class classification with class imbalance**

```
Dataset: 95% class 0, 4% class 1, 1% class 2

With standard zero-bias init:
  Initial output logits ≈ [0, 0, 0] (after small random weights)
  Initial probabilities ≈ [0.333, 0.333, 0.333]  (uniform)
  Initial loss = -log(0.333) ≈ 1.099

But the optimal constant prediction (before seeing features) is:
  [0.95, 0.04, 0.01]   (the base rates)
  Optimal loss = -(0.95 · log(0.95) + 0.04 · log(0.04) + 0.01 · log(0.01))
               ≈ 0.20

Gap = 1.099 - 0.20 = 0.879 units of loss the network must close
BEFORE it can even start learning class-discriminative features.
This can take hundreds of gradient steps.

FIX: Initialize output biases to match log-odds of class frequencies.
  b₀ = log(0.95) = -0.051
  b₁ = log(0.04) = -3.219
  b₂ = log(0.01) = -4.605

  Initial logits ≈ b (since W ≈ 0 with small init)
  softmax([−0.051, −3.219, −4.605]) ≈ [0.95, 0.04, 0.01]
  Initial loss ≈ 0.20 → already at the base rate.

Network can immediately focus on discriminative features.
Early training is much faster.
```

**Example 2: Regression with non-zero mean targets**

```
Task: predict house prices, y ∈ [200K, 2M], mean ≈ $500K

With zero-bias init:
  Initial predictions ≈ 0 (zero weights + zero bias)
  Initial MSE = E[(0 - y)²] ≈ E[y²] ≈ (500K)² = 2.5 × 10¹¹

This enormous initial loss produces enormous initial gradients.
Even with gradient clipping, early updates are dominated by
the network learning "predict something near $500K" rather than
learning the features that distinguish $200K from $2M homes.

FIX: Initialize output bias to mean of targets.
  b_out = mean(y_train) ≈ 500,000

  Initial prediction ≈ 500K for all inputs (before learning features).
  Initial MSE = Var(y) ≈ (200K)²  (just the variance around the mean)
  This is the irreducible error from ignoring features — the best
  possible loss without looking at input at all.

  Network starts at the best constant predictor and immediately
  begins learning feature-discriminative adjustments.

Alternative: normalize targets to zero mean, unit variance before
training (y_normalized = (y - mean) / std), then denormalize at
inference. Then zero-bias init is correct, and loss scale is
always O(1). This is often cleaner than output bias initialization.
```

**The general principle:**

```
Initialize the output layer such that, at initialization, the
network's predictions match the empirical distribution of y in
the training set (ignoring all input features).

This gives the network a sensible "default answer" to fall back
on, and training focuses entirely on learning feature-conditional
deviations from this default.

Mathematically: at init (W ≈ 0), the output is dominated by b.
Set b such that σ(b) ≈ p(y) (the marginal target distribution).

For sigmoid output: b = logit(p(y=1)) = log(p/(1-p))
For softmax output: bₖ = log(p(y=k))  (up to a constant)
For linear output:  b = E[y]
```

---

## 🆕 7.12 EXPANDED INTERVIEW Q&A BANK — Chapter 7

**Q4 🆕: "Xavier initialization averages the fan-in and fan-out conditions (σ² = 2/(nᵢₙ+nₒᵤₜ)) rather than just picking one. What would go wrong if you used only the fan-in condition (σ²=1/nᵢₙ) throughout a very deep, non-square-layer network?"**

**Answer:** The fan-in condition (`σ²=1/nᵢₙ`, LeCun init) is derived purely to preserve *forward-pass activation variance* — it says nothing about what happens to gradients flowing backward. Backward-pass gradient variance is governed by the symmetric relation using `nₒᵤₜ` (since `∂L/∂x = Wᵀδ` sums over `nₒᵤₜ` terms, not `nᵢₙ`). If a network has layers where `nᵢₙ ≠ nₒᵤₜ` significantly (e.g., a bottleneck architecture: 1024 → 64 → 1024), using only the fan-in condition keeps forward activations well-scaled but leaves the backward pass unbalanced — gradients can systematically shrink or grow layer-by-layer purely due to the mismatched fan-out, even though the forward activations look perfectly healthy. This is exactly the discrepancy Xavier's compromise addresses: it accepts a slightly imperfect variance-preservation in *both* directions rather than a perfect one in only one direction, which empirically works better across typical architectures with asymmetric layer widths.

---

**Q5 🆕: "You initialize an RNN's recurrent weight matrix with He initialization (designed for ReLU feedforward nets) instead of orthogonal initialization. What specifically breaks, and why doesn't the 'preserve variance per layer' argument that works for feedforward nets transfer cleanly to RNNs?"**

**Answer:** He/Xavier-style init is calibrated so that, on average across many *independently drawn* weight matrices (one per feedforward layer), variance is preserved. An RNN instead applies the **same** weight matrix `W` repeatedly, once per timestep, so instead of "variance preserved on average across many different matrices," what matters is the actual spectral norm (largest singular value) of that one specific matrix `W`, raised to the power `T` (number of timesteps). A random Gaussian matrix scaled to have the "right" variance for a feedforward layer will typically still have some singular values above 1 and some below 1 — for a single layer this averages out fine, but composing the *same* matrix `T` times means any singular value `>1` grows exponentially (`σᵀ`) and any `<1` shrinks exponentially, regardless of how carefully the variance was tuned. Orthogonal initialization sidesteps this entirely by forcing **every** singular value to exactly 1, so `‖Wᵀx‖ = ‖x‖` exactly for any `T` — there's no exponential drift because there's no singular value to drift. This is why orthogonal (not He/Xavier) is the standard choice specifically for RNN recurrent weight matrices.

---

**Q6 🆕: "A teammate says: 'Since He init already prevents NaN, we don't need to worry about output bias initialization for our imbalanced classifier — it's a nice-to-have, not a correctness issue.' Do you agree?"**

**Answer:** Partially — they're conflating two different failure modes that this chapter treats separately. He init (§7.4) solves a **numerical stability** problem: preventing activation/gradient explosion or collapse through the *hidden* layers so the network doesn't produce `nan`/`inf` and can actually compute gradients at all. Output bias initialization (§7.11 Q3) solves a **training efficiency / convergence-speed** problem: without it, the network is numerically fine, trains normally, and will *eventually* converge — but it wastes potentially hundreds of gradient steps first learning the trivial base-rate prior before it can start learning genuinely discriminative features, as shown by the ~0.88 nats of avoidable initial loss in the worked example. So: He init is closer to a "correctness/does it train at all" issue, while output bias init is a "how many GPU-hours do you want to burn re-deriving something you already know (the class prior) from scratch" issue — both matter in production, but for different reasons, and calling one a strict subset of the other is not accurate.

---

**Q7 🆕: "Why does GPT-style initialization use a small fixed constant (σ=0.02) rather than a fan-in/fan-out-dependent formula like He or Xavier?"**

**Answer:** Transformers have two properties that break the assumptions behind He/Xavier's per-layer variance-preservation derivation: (1) they use **LayerNorm** (not BatchNorm) after most sub-blocks, which explicitly renormalizes activations regardless of what the raw pre-normalization variance was — so the precise fan-in-dependent scaling that matters most for *unnormalized* deep stacks (the He/Xavier setting) is far less load-bearing once LayerNorm is guaranteed to reset the distribution anyway; and (2) transformers rely heavily on **residual connections** around every sub-block (`x + Sublayer(x)`), and empirically, keeping the *initial* contribution of each sublayer's output small (via a small fixed σ) helps the residual stream start out dominated by the stable identity path rather than by a large, noisy transformation — similar in spirit to this chapter's Fix 3 for ResNets (scaling the last layer of each residual block by `1/√(2L)`). A small constant like 0.02 is essentially a simplified, empirically-tuned version of that same "keep each block's initial perturbation small relative to the residual stream" principle, calibrated once for the specific depth/width regime GPT models are trained in, rather than re-derived per-layer from fan-in/fan-out.

---

## 🆕 7.13 RAPID-FIRE FLASHCARDS — Chapter 7

| Prompt | Answer |
|---|---|
| LeCun init? | σ² = 1/nᵢₙ |
| Xavier/Glorot init? | σ² = 2/(nᵢₙ+nₒᵤₜ) |
| He/Kaiming init? | σ² = 2/nᵢₙ |
| Why does He need 2× LeCun's variance? | ReLU halves the second moment: E[ReLU(z)²] = ½Var(z) |
| Xavier's hidden assumption? | Activation symmetric around 0 (breaks for ReLU) |
| Default bias value? | 0 |
| LSTM forget gate bias exception? | Init to 1–2 to encourage remembering by default |
| Imbalanced-classifier output bias? | b = log-odds of class frequency, e.g. log(p/(1-p)) |
| Regression output bias? | b = mean(y_train) |
| Orthogonal init guarantees? | ‖Wx‖₂ = ‖x‖₂ exactly (all singular values = 1) |
| Why orthogonal for RNNs specifically? | Same W applied T times — any singular value ≠1 compounds exponentially |
| Does BatchNorm make init irrelevant? | No — reduces sensitivity but first forward pass & non-BN nets (transformers) still care |
| NaN on first forward pass (before any update) implicates? | Init/architecture/numerics — never LR or optimizer |
| GPT init strategy? | Small fixed σ=0.02, relies on LayerNorm + residual stream stability |
| ResNet deep-net trick beyond He init? | Scale last layer of each residual block by 1/√(2L) |

---

*End of Chapter 7. Chapter 8 (Optimizers) coming next.*

---

## 🆕 CHAPTER 7 FORMULA SHEET

```
Variance propagation (general):    Var(z) = n · σ²_w · Var(x)
LeCun:                              σ²_w = 1/nᵢₙ
Xavier/Glorot:                      σ²_w = 2/(nᵢₙ+nₒᵤₜ)
He/Kaiming:                         σ²_w = 2/nᵢₙ
Leaky ReLU generalized He:          σ²_w = 2/((1+α²)·nᵢₙ)
Orthogonal property:                ‖Wx‖₂ = ‖x‖₂  (all singular values = 1)

Output bias — sigmoid:              b = log(p/(1-p))
Output bias — softmax:              bₖ = log(p(y=k))
Output bias — linear/regression:    b = E[y]
```

## 🆕 "TOP 5 THINGS THAT TRIP PEOPLE UP" — Chapter 7

1. Using Xavier init with ReLU networks "because it's the classic default" — Xavier's symmetric-activation assumption is specifically violated by ReLU.
2. Assuming BatchNorm makes initialization a non-issue — it helps a lot, but the very first forward pass (before any BN statistics exist) is still exposed.
3. Applying feedforward-style (He/Xavier) init to an RNN's recurrent weights — the *same* matrix applied across many timesteps needs orthogonal init, not a per-layer variance formula.
4. Treating output-bias initialization as cosmetic — on an imbalanced dataset it can cost hundreds of wasted gradient steps re-deriving the class prior.
5. Diagnosing a NaN-on-first-forward-pass as a learning-rate problem — if no gradient step has happened yet, the learning rate cannot possibly be the cause.

---

*This document preserves 100% of the original Chapter 7 content and adds interview-focused expansions marked with 🆕. Ready for Chapter 8 (Optimizers) whenever you want it boosted the same way.*
