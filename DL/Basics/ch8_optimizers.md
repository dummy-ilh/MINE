# Chapter 8: Optimizers (SGD, Momentum, RMSProp, Adam)

---

### 8.1 The Plain-English Picture

Gradient descent (Chapter 6) tells us the direction to move weights to reduce loss. But it answers only one question: which direction? It says nothing about how far to move, how to handle directions where the loss changes very slowly versus very quickly, or how to avoid oscillating in ravines while crawling toward a distant minimum.

Optimizers are the algorithms that turn raw gradients into weight updates. They all start from the same gradient signal and apply different strategies to make training faster, more stable, and better at finding good minima.

Think of the loss landscape as a hilly terrain and the optimizer as a ball rolling downhill:

```
LOSS LANDSCAPE — top view
==========================

  ┌─────────────────────────────────────────────┐
  │         ~~~~high loss~~~~                   │
  │    ~~~~~~~                ~~~~~~~           │
  │  ~~~  saddle point here  ~~~~               │
  │  ~~  ──────────────────  ~~~                │
  │  ~~ │   narrow ravine  │ ~~~                │
  │  ~~ │                  │ ~~~                │
  │  ~~ │     minimum ●    │ ~~~                │
  │  ~~ └──────────────────┘ ~~~                │
  │    ~~~~~~~         ~~~~~~~                  │
  │         ~~~~high loss~~~~                   │
  └─────────────────────────────────────────────┘

Vanilla SGD: bounces back and forth across the ravine,
             crawls slowly toward the minimum.

Momentum:    builds up speed along the ravine floor,
             reduces side-to-side oscillation.

RMSProp:     automatically scales steps per dimension —
             small steps across the ravine (steep),
             large steps along the ravine (flat).

Adam:        momentum + adaptive scaling. The fastest,
             most widely used optimizer in deep learning.
```

The history of optimizers is a history of solving specific failure modes of the simpler algorithms. Understanding each failure mode is understanding why each optimizer exists.

---

### 8.2 Vanilla SGD (Stochastic Gradient Descent)

The baseline. Everything else is built on top of this.

```
SGD UPDATE RULE
===============

  θₜ₊₁ = θₜ - η · gₜ

Where:
  θₜ   = parameters at step t (all weights and biases)
  gₜ   = gradient at step t: gₜ = ∂L/∂θ  evaluated at θₜ
  η    = learning rate (scalar hyperparameter)
  t    = step index

In component form, for one weight w:
  wₜ₊₁ = wₜ - η · (∂L/∂w)

PROPERTIES:
  + Simple, low memory (only store θ, no extra state)
  + Unbiased gradient estimate (with mini-batches)
  - Same learning rate for ALL parameters
  - No memory of past gradients (each step independent)
  - Sensitive to learning rate choice
  - Slow convergence in ravines (oscillates across, crawls along)
```

**The ravine problem — quantified:**

```
Imagine a loss function shaped like an elongated bowl:
  L(w₁, w₂) = 100·w₁² + w₂²

Gradient: [∂L/∂w₁, ∂L/∂w₂] = [200·w₁, 2·w₂]

Starting at (0.5, 5.0):
  Gradient = [100, 10]
  Step = η × gradient

With η = 0.01:
  w₁: 0.5 - 0.01×100 = 0.5 - 1.0 = -0.5  ← OVERSHOOT! w₁ flips sign
  w₂: 5.0 - 0.01×10  = 5.0 - 0.1  = 4.9  ← tiny progress

The loss is 100× more curved in w₁ direction.
SGD oscillates wildly in w₁ while crawling in w₂.
To avoid oscillating in w₁, we'd need η ≤ 0.005.
But then w₂ progress is even slower.
This is the fundamental tension SGD cannot resolve.
```

---

### 8.3 SGD with Momentum

Momentum borrows physics intuition: a ball rolling downhill accumulates velocity. It doesn't just respond to the current slope — it carries momentum from previous steps.

```
MOMENTUM UPDATE RULE
====================

  vₜ = β · vₜ₋₁ + gₜ          (velocity update)
  θₜ₊₁ = θₜ - η · vₜ          (parameter update)

Alternative (equivalent) formulation:
  vₜ = β · vₜ₋₁ + (1-β) · gₜ  (exponential moving average of g)
  θₜ₊₁ = θₜ - η/(1-β) · vₜ

Where:
  vₜ   = velocity (momentum vector), same shape as θ
  β    = momentum coefficient (hyperparameter, typically 0.9)
  gₜ   = current gradient
  v₀   = 0  (initialized to zero)

INTUITION:
  v accumulates gradients over time.
  If gradients consistently point in the same direction:
    v grows → larger steps → faster progress (acceleration)
  If gradients oscillate in sign:
    v cancels out → smaller steps → less oscillation (damping)

EXPANDING THE RECURRENCE:
  v₁ = g₁
  v₂ = β·g₁ + g₂
  v₃ = β²·g₁ + β·g₂ + g₃
  vₜ = Σₖ₌₁ᵗ βᵗ⁻ᵏ · gₖ

  → vₜ is an exponentially weighted sum of all past gradients.
  Recent gradients get weight 1 (β⁰ = 1).
  Gradients from k steps ago get weight βᵏ.
  With β=0.9: gradient from 10 steps ago has weight 0.9¹⁰ ≈ 0.35.
              gradient from 50 steps ago has weight 0.9⁵⁰ ≈ 0.005.

EFFECTIVE LOOK-BACK WINDOW:
  The effective number of steps contributing to v:
    1 / (1-β)
  β=0.9:  window ≈ 10 steps
  β=0.99: window ≈ 100 steps
```

**Resolving the ravine problem:**

```
Same ravine: L(w₁,w₂) = 100w₁² + w₂²
β = 0.9, η = 0.01

w₁ direction (steep): gradient oscillates +100, -100, +100, ...
  v(w₁) = 0.9 × (−100) + 100 = 10   ← mostly cancels!
  Effective step in w₁ ≈ small → oscillation damped ✓

w₂ direction (flat): gradient consistently +10, +10, +10, ...
  v(w₂) = 0.9×10 + 10 = 19   → step = 0.01×19 = 0.19
  Next: v(w₂) = 0.9×19 + 10 = 27.1 → step = 0.271
  Accelerating toward minimum in w₂ ✓

Momentum achieves: damp oscillations, accelerate in consistent directions.
```

**Nesterov Momentum (NAG):**

```
NESTEROV ACCELERATED GRADIENT
==============================

Standard momentum: compute gradient at current position, then update.
Nesterov:          compute gradient at the "lookahead" position first.

  θ̃ₜ = θₜ - η·β·vₜ₋₁        (lookahead step using old velocity)
  gₜ = ∂L/∂θ at θ̃ₜ          (gradient at lookahead position)
  vₜ = β·vₜ₋₁ + gₜ          (velocity update)
  θₜ₊₁ = θₜ - η·vₜ          (parameter update)

INTUITION:
  Standard momentum: use gradient from where you ARE, then apply momentum.
  Nesterov: apply momentum first (where you're GOING), then correct.

  It's like a skier who looks ahead before turning,
  vs one who turns based on where they are right now.

  Nesterov converges faster than standard momentum in theory
  (O(1/t²) vs O(1/t) for convex problems).
  Small but consistent improvement in practice.
  Used in some production systems (pytorch SGD has nesterov=True).
```

---

### 8.4 AdaGrad

The first adaptive learning rate optimizer. Different parameters get different learning rates, automatically calibrated to their history.

```
ADAGRAD UPDATE RULE
===================

  Gₜ = Gₜ₋₁ + gₜ²             (accumulate squared gradients)
  θₜ₊₁ = θₜ - (η / √(Gₜ + ε)) · gₜ

Where:
  Gₜ   = accumulated sum of squared gradients (same shape as θ)
  gₜ²  = element-wise square of gradient
  ε    = small constant for numerical stability (e.g., 1e-8)
  η    = global learning rate (typically larger, e.g. 0.01)
  G₀   = 0

ADAPTIVE EFFECT:
  Parameters with large historical gradients:
    Gₜ is large → η/√Gₜ is small → small steps (reduced lr)
    These are parameters that are already changing a lot.

  Parameters with small historical gradients:
    Gₜ is small → η/√Gₜ is large → large steps (increased lr)
    These are parameters that rarely update (sparse features).

WHY THIS HELPS:
  NLP example: word embeddings.
  "the" appears in every sentence → gradient always large → needs small lr
  "xylophone" appears rarely → gradient often 0 → needs large lr when it appears

  AdaGrad automatically gives "the" a small lr and "xylophone" a large lr.

FATAL FLAW:
  Gₜ = Σₖ₌₁ᵗ gₖ²  is monotonically increasing — it NEVER decreases.

  After many steps, Gₜ → ∞ → η/√Gₜ → 0 → learning rate → 0.

  In long training runs, ALL learning stops, even if the model
  hasn't converged. AdaGrad "dies" in non-convex settings.
  Great for convex optimization (proved optimal for sparse problems).
  Bad for deep learning (training runs too long).
```

---

### 8.5 RMSProp

Geoff Hinton introduced RMSProp in a 2012 Coursera lecture (never formally published!) specifically to fix AdaGrad's dying learning rate.

```
RMSPROP UPDATE RULE
===================

  Sₜ = ρ·Sₜ₋₁ + (1-ρ)·gₜ²    (exponential moving avg of g²)
  θₜ₊₁ = θₜ - (η / √(Sₜ + ε)) · gₜ

Where:
  Sₜ   = exponential moving average of squared gradients
  ρ    = decay rate (hyperparameter, typically 0.9 or 0.99)
  gₜ²  = element-wise square of gradient
  ε    = 1e-8 (numerical stability)
  η    = learning rate (typically 0.001)
  S₀   = 0

KEY DIFFERENCE FROM ADAGRAD:
  AdaGrad:  Gₜ = Σₖ₌₁ᵗ gₖ²           (sum: grows forever)
  RMSProp:  Sₜ = ρ·Sₜ₋₁ + (1-ρ)·gₜ²  (EMA: stays bounded)

  Sₜ is an exponential moving average — it "forgets" old gradients.
  Old gradient magnitudes decay at rate ρ per step.
  Learning rate adapts to RECENT gradient history, not all history.

EXPANDING THE RECURRENCE:
  Sₜ = (1-ρ) · Σₖ₌₀ᵗ ρᵏ · gₜ₋ₖ²

  This is a weighted average where recent g² values get more weight.
  The denominator √Sₜ approximates the RMS (root mean square) of
  recent gradients — hence the name: Root Mean Square Propagation.

EFFECTIVE WINDOW:
  Same as momentum: 1/(1-ρ)
  ρ=0.9:  window ≈ 10 steps
  ρ=0.99: window ≈ 100 steps

INTUITION: RMSProp normalizes the gradient by its recent magnitude.
  Large recent gradients → small step (parameter is changing a lot)
  Small recent gradients → large step (parameter needs a push)

This is direction-independent rescaling: it doesn't care about
the sign of g, only its magnitude (via g²).
```

---

### 8.6 Adam: Adaptive Moment Estimation

Adam (Kingma & Ba, 2014) combines the best of momentum (first moment: direction smoothing) and RMSProp (second moment: magnitude scaling). It is the default optimizer for the majority of deep learning work.

```
ADAM UPDATE RULE
================

Initialize: m₀ = 0, v₀ = 0, t = 0

At each step t:
  t ← t + 1

  1. Compute gradient:
     gₜ = ∂L/∂θ at θₜ

  2. Update biased first moment estimate (mean of gradients):
     mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ

  3. Update biased second moment estimate (mean of g²):
     vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²

  4. Bias correction:
     m̂ₜ = mₜ / (1 - β₁ᵗ)
     v̂ₜ = vₜ / (1 - β₂ᵗ)

  5. Update parameters:
     θₜ₊₁ = θₜ - η · m̂ₜ / (√v̂ₜ + ε)

Where:
  mₜ   = first moment (exponential moving average of gradients)
  vₜ   = second moment (exponential moving average of g²)
  β₁   = first moment decay (default: 0.9)
  β₂   = second moment decay (default: 0.999)
  η    = learning rate (default: 0.001)
  ε    = 1e-8 (numerical stability)
  β₁ᵗ  = β₁ raised to the power t (exponentially decaying)

DEFAULT HYPERPARAMETERS:
  β₁ = 0.9    (momentum — smooth gradient direction)
  β₂ = 0.999  (RMSProp — track gradient magnitude)
  η  = 0.001  (learning rate — usually works without tuning)
  ε  = 1e-8   (stability)

These defaults work well across a remarkable range of problems.
```

**The bias correction — why it exists:**

```
WHY BIAS CORRECTION?
====================

At t=1, with β₁=0.9:
  m₁ = β₁·m₀ + (1-β₁)·g₁ = 0.9×0 + 0.1×g₁ = 0.1·g₁

  m₁ is only 0.1 of the true gradient! It's heavily biased
  toward zero because we initialized m₀=0.

  m̂₁ = m₁ / (1 - β₁¹) = 0.1·g₁ / (1 - 0.9) = 0.1·g₁ / 0.1 = g₁ ✓

Bias correction recovers the true estimate in early steps.

At t=1000, β₁¹⁰⁰⁰ ≈ 0:
  m̂₁₀₀₀ = m₁₀₀₀ / (1 - 0) ≈ m₁₀₀₀  (no correction needed)

Bias correction matters most in the FIRST FEW STEPS of training.
After ~100 steps, it has essentially no effect (for β₁=0.9).
For β₂=0.999, it matters for the first ~1000 steps.
```

**Adam's effective learning rate per parameter:**

```
Effective lr for parameter i:
  η_eff = η · m̂ᵢ / (√v̂ᵢ + ε)

The update is approximately:
  η_eff ≈ η · sign(m̂ᵢ)   when |m̂ᵢ| << √v̂ᵢ (most cases)

This means Adam's effective step size is roughly:
  ≈ η in all directions (very consistent step sizes)

Unlike SGD where step size = η × |gradient| (can be huge or tiny),
Adam normalizes the update so all parameters take similar-sized steps.

This is why Adam is less sensitive to learning rate:
  The normalization by √v̂ keeps updates bounded regardless of
  gradient magnitude. η=0.001 works for wildly different problems.
```

---

### 8.7 AdamW: Adam with Decoupled Weight Decay

```
THE PROBLEM WITH L2 IN ADAM
=============================

L2 regularization adds λ·θ to the gradient: g̃ₜ = gₜ + λ·θₜ

In SGD: weight decay and L2 are equivalent.
  Δθ = -η·(g + λθ) = -η·g - η·λ·θ

In Adam: weight decay and L2 are NOT equivalent.
  Adam normalizes by √v̂, so:
  Δθ = -η·(g + λθ) / √v̂

  The regularization term (λθ) is also divided by √v̂.
  Parameters with large gradients (large v̂) receive LESS
  regularization than parameters with small gradients.
  This breaks the intended regularization behavior.

ADAMW FIX (Loshchilov & Hutter, 2017):
  Decouple weight decay from the gradient update.

  mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ         (gradient only, no λθ)
  vₜ = β₂·vₜ₋₁ + (1-β₂)·gₜ²        (gradient only)
  θₜ₊₁ = θₜ - η·(m̂ₜ/√v̂ₜ + ε) - η·λ·θₜ   (weight decay applied directly)

  Weight decay now shrinks θ by η·λ regardless of gradient history.
  This is the mathematically correct form of weight decay for Adam.

AdamW is now the DEFAULT for transformer training.
GPT-3, BERT, LLaMA all use AdamW.
Standard hyperparameters:
  β₁=0.9, β₂=0.95 (not 0.999!), η=3e-4, λ=0.1
```

---

### 8.8 Optimizer Comparison

```
┌─────────────┬──────────┬──────────┬────────────┬────────────────────────┐
│ Optimizer   │ Memory   │ Per-param│ Robust to  │ Best for               │
│             │ overhead │ lr adapt │ lr choice  │                        │
├─────────────┼──────────┼──────────┼────────────┼────────────────────────┤
│ SGD         │ 0        │ No       │ Very low   │ CNNs (with tuning)     │
│ SGD+Momentum│ 1× θ     │ No       │ Low        │ CNNs (SotA with tuning)│
│ AdaGrad     │ 1× θ     │ Yes      │ Medium     │ Sparse data (NLP old)  │
│ RMSProp     │ 1× θ     │ Yes      │ High       │ RNNs, non-stationary   │
│ Adam        │ 2× θ     │ Yes      │ Very high  │ Most DL problems       │
│ AdamW       │ 2× θ     │ Yes      │ Very high  │ Transformers (default) │
│ Nadam       │ 2× θ     │ Yes      │ Very high  │ Adam + Nesterov look   │
└─────────────┴──────────┴──────────┴────────────┴────────────────────────┘

MEMORY: θ = number of parameters
  SGD:        store only θ            (1× parameter count)
  Momentum:   store θ + v            (2× parameter count)
  Adam:       store θ + m + v        (3× parameter count)
  For GPT-3 (175B params): Adam needs 175B×3×4 bytes ≈ 2.1 TB just for optimizer state!
  This is why large model training uses memory-efficient variants:
  Adafactor, 8-bit Adam (bitsandbytes), ZeRO optimizer (DeepSpeed).
```

---

### 8.9 Worked Numerical Example: 5 Steps of Adam

```
SETUP
=====
Single parameter θ (scalar for clarity)
True gradient at each step: g = [0.5, 0.8, -0.3, 0.6, 0.4]

Hyperparameters:
  η   = 0.001
  β₁  = 0.9
  β₂  = 0.999
  ε   = 1e-8

Initial state: θ₀ = 1.0, m₀ = 0, v₀ = 0

═══════════════════════════════════════════════════════════════
STEP t=1, g₁=0.5
═══════════════════════════════════════════════════════════════
m₁ = 0.9×0    + 0.1×0.5   = 0.0500
v₁ = 0.999×0  + 0.001×0.25 = 0.000250

Bias correction:
  m̂₁ = 0.0500 / (1 - 0.9¹)   = 0.0500 / 0.1    = 0.5000
  v̂₁ = 0.000250 / (1 - 0.999¹) = 0.000250 / 0.001 = 0.2500

Update:
  θ₁ = 1.0 - 0.001 × 0.5000 / (√0.2500 + 1e-8)
     = 1.0 - 0.001 × 0.5000 / 0.5000
     = 1.0 - 0.001 × 1.0000
     = 1.0 - 0.001000
     = 0.999000

═══════════════════════════════════════════════════════════════
STEP t=2, g₂=0.8
═══════════════════════════════════════════════════════════════
m₂ = 0.9×0.0500  + 0.1×0.8   = 0.0450 + 0.0800 = 0.1250
v₂ = 0.999×0.000250 + 0.001×0.64 = 0.000250 + 0.000640 = 0.000890

Bias correction:
  m̂₂ = 0.1250 / (1 - 0.9²)   = 0.1250 / 0.19   = 0.6579
  v̂₂ = 0.000890 / (1 - 0.999²) = 0.000890 / 0.001999 = 0.4452

Update:
  θ₂ = 0.999000 - 0.001 × 0.6579 / (√0.4452 + 1e-8)
     = 0.999000 - 0.001 × 0.6579 / 0.6672
     = 0.999000 - 0.001 × 0.9861
     = 0.999000 - 0.000986
     = 0.998014

═══════════════════════════════════════════════════════════════
STEP t=3, g₃=-0.3
═══════════════════════════════════════════════════════════════
m₃ = 0.9×0.1250  + 0.1×(-0.3) = 0.1125 - 0.0300 = 0.0825
v₃ = 0.999×0.000890 + 0.001×0.09 = 0.000889 + 0.000090 = 0.000979

Bias correction:
  m̂₃ = 0.0825 / (1 - 0.9³)   = 0.0825 / 0.271  = 0.3044
  v̂₃ = 0.000979 / (1 - 0.999³) = 0.000979 / 0.002997 = 0.3267

Update:
  θ₃ = 0.998014 - 0.001 × 0.3044 / (√0.3267 + 1e-8)
     = 0.998014 - 0.001 × 0.3044 / 0.5716
     = 0.998014 - 0.001 × 0.5325
     = 0.998014 - 0.000533
     = 0.997482

═══════════════════════════════════════════════════════════════
STEPS 4 AND 5 (summarized)
═══════════════════════════════════════════════════════════════

Step 4 (g₄=0.6):
  m₄ = 0.9×0.0825 + 0.1×0.6  = 0.1343
  v₄ = 0.999×0.000979 + 0.001×0.36 = 0.001339
  m̂₄ = 0.1343/0.3439 = 0.3905
  v̂₄ = 0.001339/0.003994 = 0.3353
  θ₄ = 0.997482 - 0.001 × 0.3905/0.5790 = 0.997482 - 0.000675 = 0.996808

Step 5 (g₅=0.4):
  m₅ = 0.9×0.1343 + 0.1×0.4  = 0.1609
  v₅ = 0.999×0.001339 + 0.001×0.16 = 0.001499
  m̂₅ = 0.1609/0.4095 = 0.3929
  v̂₅ = 0.001499/0.004990 = 0.3004
  θ₅ = 0.996808 - 0.001 × 0.3929/0.5481 = 0.996808 - 0.000717 = 0.996091

═══════════════════════════════════════════════════════════════
COMPARISON WITH VANILLA SGD (η=0.001)
═══════════════════════════════════════════════════════════════

Step │ Gradient │ SGD θ         │ Adam θ       │ Adam step size
─────┼──────────┼───────────────┼──────────────┼──────────────
  1  │  0.5     │ 1.000-0.00050 │ 1.000-0.00100│   0.00100
  2  │  0.8     │ 0.9995-0.00080│ 0.999-0.00099│   0.00099
  3  │ -0.3     │ 0.9987+0.00030│ 0.998-0.00053│   0.00053  ← reversed!
  4  │  0.6     │ 0.9990-0.00060│ 0.997-0.00068│   0.00068
  5  │  0.4     │ 0.9984-0.00040│ 0.997-0.00072│   0.00072

Key observations:
  1. Adam's step sizes are more consistent (~0.001) vs SGD which
     scales with gradient magnitude (0.0003 to 0.0008).
  2. At step 3, gradient reversed sign. Adam's momentum (m₃=0.0825)
     still points positive, so it takes a small POSITIVE step
     even though the raw gradient is negative. This is momentum
     "continuing" in the accumulated direction.
  3. After 5 steps, Adam has moved θ: 1.000 → 0.996 (move of 0.004)
     SGD has moved θ:  1.000 → 0.998 (move of 0.002)
     Adam moved faster despite the same η, because of momentum.
```

---

### 8.10 Learning Rate Schedules with Adam

```
WARMUP + COSINE DECAY (transformer standard)
=============================================

  Phase 1 — Warmup (first W steps):
    η(t) = η_max × (t / W)    linear increase from 0 to η_max

  Phase 2 — Cosine decay (remaining T-W steps):
    η(t) = η_min + (1/2)(η_max - η_min)(1 + cos(π(t-W)/(T-W)))

  Typical values:
    η_max = 3e-4   (peak learning rate)
    η_min = 3e-5   (minimum learning rate, 10% of peak)
    W     = 4000   (warmup steps for BERT-scale models)

  Loss during training:
    │╲                            ← high initial loss
    │  ╲   (warmup)
    │    ╲────
    │        ╲────               ← cosine decay
    │              ╲────
    │                    ────    ← converged
    └────────────────────────────

WHY WARMUP?
  At initialization, gradient estimates are noisy and biased.
  Adam's second moment v̂ is near zero (bias-corrected helps
  but doesn't fully fix the issue for very deep networks).
  Large initial learning rates cause instability.
  Warmup gives Adam time to build reliable moment estimates
  before taking large steps.
  Without warmup: common to see loss spike and NaN in first ~100 steps.

ONE CYCLE POLICY (Leslie Smith, 2018):
  Start low, ramp up to max, ramp down past original low.
  Also includes cyclical momentum (high β₁ when lr low, vice versa).
  Can train in 1/5th the steps of constant-lr SGD.
  Used in fast.ai, popular for image classification.
```

---

### 8.11 Why This Matters — What Breaks If You Get This Wrong

1. **Using vanilla SGD on a transformer.** Transformers have highly non-uniform gradient scales across layers (attention parameters vs. embedding parameters). SGD treats all parameters identically. Training either diverges (lr too high for some params) or stalls (lr too low for others). Adam's per-parameter adaptation is not optional for transformers — it's required.

2. **Forgetting bias correction in Adam implementation.** If you implement Adam without the (1-β₁ᵗ) and (1-β₂ᵗ) corrections, the first few hundred steps use effectively zero learning rate (m and v are near zero, unbiased). Loss barely moves. Training looks broken. The fix is two division operations — trivial to add, catastrophic to omit.

3. **Using Adam when SGD generalizes better.** A well-known empirical finding (Wilson et al., 2017): Adam often converges to sharper minima that generalize worse than SGD's flatter minima. For image classification (ResNets on ImageNet), SGD + momentum with cosine learning rate decay often outperforms Adam by 1-2% top-1 accuracy. Use Adam when training is hard (transformers, RNNs). Use SGD+momentum when you care about squeezing out maximum generalization (CNNs).

4. **Not decoupling weight decay (using Adam instead of AdamW).** Standard Adam applies weight decay through L2 in the gradient, which gets rescaled by the adaptive learning rate. Parameters with large gradients get less regularization than intended. AdamW decouples the two. For transformers, this is the difference between a model that overfit and one that generalizes — often 1-3% on downstream tasks.

5. **Learning rate too high without warmup.** On large models (>100M parameters), starting Adam at lr=3e-4 from step 1 causes loss spikes or NaN in the first ~50 steps, before moment estimates stabilize. The solution is trivial (linear warmup over 1000-4000 steps) but easy to overlook. A training run that crashes at step 47 after 2 hours of compute is an expensive lesson.

---

### 8.12 Google/Apple-Level Interview Q&A

---

**Q1: "Explain Adam's bias correction mathematically. Why is it necessary at the beginning of training but not later? What would happen if you removed it?"**

*Why this is asked:* Bias correction is one of Adam's most misunderstood components. Many practitioners use Adam without understanding why m and v are divided by (1-β^t). This question tests mathematical depth — the ability to reason about initialization of recursive sequences — and reveals whether the candidate understands Adam from first principles or just the API.

**Answer:**

```
THE PROBLEM: INITIALIZATION BIAS
==================================

m₀ = 0, v₀ = 0  (initialized to zero vectors)

First moment recurrence:
  mₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ

Expanding:
  m₁ = (1-β₁)·g₁
  m₂ = β₁(1-β₁)·g₁ + (1-β₁)·g₂  = (1-β₁)[β₁·g₁ + g₂]
  mₜ = (1-β₁)·Σₖ₌₁ᵗ β₁ᵗ⁻ᵏ·gₖ

If gradients are drawn from a stationary distribution
with E[gₖ] = μ (true mean gradient):

  E[mₜ] = (1-β₁)·Σₖ₌₁ᵗ β₁ᵗ⁻ᵏ·μ
         = μ · (1-β₁) · (1 - β₁ᵗ)/(1-β₁)   [geometric series]
         = μ · (1 - β₁ᵗ)

So: E[mₜ] = μ · (1 - β₁ᵗ)  ≠  μ  (biased toward zero!)

The bias factor is (1 - β₁ᵗ):
  t=1:   1 - 0.9¹  = 0.10  → mₜ is 10% of the true mean
  t=5:   1 - 0.9⁵  = 0.41  → mₜ is 41% of the true mean
  t=10:  1 - 0.9¹⁰ = 0.65  → mₜ is 65% of the true mean
  t=50:  1 - 0.9⁵⁰ = 0.995 → mₜ is 99.5% of the true mean
  t=100: 1 - 0.9¹⁰⁰ ≈ 1.0  → fully unbiased

BIAS CORRECTION:
  m̂ₜ = mₜ / (1-β₁ᵗ)
  E[m̂ₜ] = E[mₜ] / (1-β₁ᵗ) = μ·(1-β₁ᵗ)/(1-β₁ᵗ) = μ   ✓

Same analysis for second moment vₜ with β₂=0.999:
  t=1:    1-0.999¹  = 0.001  → v₁ is 0.1% of true E[g²]
  t=100:  1-0.999¹⁰⁰ ≈ 0.095 → v₁₀₀ is 9.5% of true E[g²]
  t=1000: 1-0.999¹⁰⁰⁰ ≈ 0.63 → still only 63%!

CRITICAL INSIGHT: Because β₂=0.999 is so close to 1,
  the second moment takes ~1000 steps to become unbiased.
  Without correction: steps 1-1000 use a v̂ that's too small
  → effective lr = η/√v̂ is much LARGER than intended
  → risk of instability in early training.

WHAT HAPPENS WITHOUT BIAS CORRECTION:
  Without correction, the effective learning rate in early steps:
    η_eff = η / √vₜ  where vₜ << true E[g²]
  
  At t=1: v₁ = (1-β₂)·g₁² = 0.001·g₁²
    η_eff = η/√(0.001·g₁²) = η/(0.0316·|g₁|) = 31.6 × (η/|g₁|)

  With bias correction: v̂₁ = v₁/0.001 = g₁²
    η_eff = η/√g₁² = η/|g₁|

  Without correction, the effective lr is 31.6× larger in step 1.
  This can cause the infamous NaN/divergence in early Adam training.
  Bias correction is not cosmetic — it's a numerical safety mechanism.
```

---

**Q2: "You're asked to train a large language model (say, 7B parameters) and your GPU memory budget can only fit the model weights plus one extra copy. Adam requires 2× extra copies (m and v). How do you solve this? Name at least two approaches and explain the memory math."**

*Why this is asked:* Large-scale training is the frontier where theory meets engineering constraints. Google and Apple train massive models under severe memory pressure. This question tests awareness of production-scale optimization challenges and the memory-efficient optimizer landscape — knowledge that separates ML engineers from ML researchers.

**Answer:**

```
MEMORY MATH
============

7B parameter model in float32:
  Model weights:   7 × 10⁹ × 4 bytes = 28 GB
  Adam m (float32): 28 GB
  Adam v (float32): 28 GB
  Total:           84 GB

GPU budget: 28 GB (weights) + 28 GB (one extra copy) = 56 GB
Shortfall: 84 GB - 56 GB = 28 GB over budget.

We need to eliminate one full copy of optimizer state (28 GB).
```

**Solution 1: Adafactor (Shazeer & Stern, 2018)**

```
ADAFACTOR
=========
Instead of storing the full m and v matrices (one float per param),
factor the second moment matrix into low-rank approximations.

For a weight matrix W ∈ ℝᵐˣⁿ:
  Standard Adam: store v ∈ ℝᵐˣⁿ  (m×n floats)
  Adafactor: approximate v ≈ rˢ · cˢ  (m + n floats, rank-1)
  
  rˢ ∈ ℝᵐ  (row factors)
  cˢ ∈ ℝⁿ  (column factors)

Memory for v: m×n → m+n  (massive reduction for large matrices)

For W ∈ ℝ⁴⁰⁹⁶ˣ⁴⁰⁹⁶:
  Full v:      4096 × 4096 = 16.7M floats = 67 MB
  Adafactor v: 4096 + 4096 = 8192 floats  = 33 KB  (2000× smaller!)

Adafactor also optionally removes the first moment (m),
replacing it with gradient clipping, saving another 28 GB.

Total memory: model (28 GB) + adafactor state (~2-3 GB) ≈ 31 GB ✓

Used by: T5, PaLM (Google), many production-scale models.
Downside: slightly slower convergence than Adam in practice.
```

**Solution 2: 8-bit Adam (Dettmers et al., 2022)**

```
8-BIT ADAM
==========

Store m and v in 8-bit integers instead of 32-bit floats.

Memory reduction: 4 bytes → 1 byte per element = 4× reduction
  Adam m (int8): 7 GB (was 28 GB)
  Adam v (int8): 7 GB (was 28 GB)
  Total: 28 + 7 + 7 = 42 GB  (vs. 84 GB)

The trick: dynamic quantization
  Before each update, dequantize m and v to float32 for the update computation.
  After the update, re-quantize back to int8 for storage.
  
  The loss in precision is tiny: int8 has 256 levels.
  But: the optimizer state changes slowly (smooth EMA), so
  8-bit is sufficient to represent the direction accurately.

Empirical result: matches full-precision Adam to within
0.1-0.3% on downstream tasks, at 4× lower memory.

Implementation: bitsandbytes library (works as drop-in replacement)
  import bitsandbytes as bnb
  optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)

Used by: many open-source LLM fine-tuning workflows.
```

**Solution 3: ZeRO Optimizer (DeepSpeed)**

```
ZERO (Zero Redundancy Optimizer)
=================================
Shard the optimizer state across multiple GPUs.

With N GPUs:
  Each GPU stores only (1/N) of m and v.
  During the update, GPUs communicate to share necessary shards.

Memory per GPU (N=8 GPUs):
  Weights: 28 GB (replicated or sharded — ZeRO stages 1/2/3)
  m (sharded): 28/8 = 3.5 GB
  v (sharded): 28/8 = 3.5 GB
  Total per GPU: ~35 GB  ✓

ZeRO stages:
  Stage 1: shard optimizer state only
  Stage 2: shard gradients + optimizer state
  Stage 3: shard weights + gradients + optimizer state

ZeRO-3 allows training models larger than any single GPU can hold.
Used by: Microsoft's Megatron-LLM, OpenAI's training infrastructure.
```

---

**Q3: "In practice, SGD with momentum often generalizes better than Adam for image classification, but Adam is preferred for NLP. Explain the theoretical and empirical reasons for this difference."**

*Why this is asked:* This is a nuanced question with no clean textbook answer — it requires synthesizing theory, empirical results, and architectural understanding. It distinguishes ML scientists from ML engineers, and reveals whether a candidate actually thinks about why optimizers work rather than just using them as black boxes.

**Answer:**

**The empirical observation:**

```
ImageNet top-1 accuracy (ResNet-50):
  SGD + momentum + cosine lr: ~77-78%
  Adam:                       ~75-76%
  Gap: ~1-2% favoring SGD

GLUE benchmark (BERT-base):
  Adam / AdamW:               ~84-85%
  SGD + momentum:             ~75-80% (highly sensitive to lr)
  Gap: ~5-10% favoring Adam
```

**Theoretical reason 1: Sharpness of minima**

```
Wilson et al. (2017) showed that adaptive optimizers tend to
converge to SHARPER minima (higher curvature) than SGD.

Sharp minimum:                  Flat minimum:
  Loss                            Loss
  │   ╱╲                          │      ╭─────╮
  │  ╱  ╲                         │    ╭─╯     ╰─╮
  │ ╱    ╲                        │  ──╯         ╰──
  └────────                       └────────────────
  
  Distribution shift: if test set is slightly different from
  train, the optimal θ shifts slightly.
  
  In a sharp minimum: small shift in θ → large loss increase
  In a flat minimum:  small shift in θ → small loss increase
  
  Flat minima generalize better.
  SGD's noisy, less precise gradient estimates navigate toward
  flatter regions of the loss landscape.
  Adam's adaptive scaling is more precise, which helps it find
  minima faster — but those minima tend to be sharper.
```

**Theoretical reason 2: The geometry of image vs. language tasks**

```
IMAGE CLASSIFICATION (ResNets):
  - Gradients are relatively homogeneous across parameters
  - Learning rates for different layers don't need to vary by orders of magnitude
  - The loss landscape is relatively smooth and well-conditioned
  - SGD's uniform treatment of all parameters is fine here
  - The noise in SGD provides beneficial regularization (implicit regularization)

LANGUAGE MODELS (Transformers, BERT, GPT):
  - Massive dynamic range in gradient scales:
    * Embedding layers: sparse gradients (most tokens not in this batch)
    * Attention weights: dense but highly variable magnitude
    * LayerNorm parameters: very different scale from weight matrices
  - AdaGrad-like adaptation is NECESSARY:
    * Rare token embeddings need large updates when they appear
    * Frequent token embeddings need small updates (already learned)
    * SGD would either ignore rare tokens or destabilize frequent ones
  - Transformer loss landscapes have extremely high curvature in
    some directions (attention head competition) → adaptive lr critical

EMPIRICAL SUPPORT:
  You can think of Adam as SGD with implicit per-parameter lr tuning.
  For ResNets: the manually tuned SGD lr schedule (cosine, warmup)
  can match what Adam finds automatically, given enough tuning effort.
  For Transformers: no manually designed SGD schedule reliably matches
  Adam, because the scale diversity is too extreme to handle uniformly.
```

**Practical implication:**

```
Guideline (backed by evidence):
  Task with homogeneous gradient scales + time to tune lr: SGD+momentum
  Task with heterogeneous gradient scales or NLP: Adam/AdamW

  BOTH with:
    - Learning rate warmup (first 1000-4000 steps)
    - Learning rate decay (cosine or linear)
    - Gradient clipping (norm ≤ 1.0)
    - Weight decay (AdamW for Adam, explicit L2 for SGD)

The "Adam generalizes worse" finding is specifically about
image classification with well-tuned hyperparameters.
In practice, most teams use Adam for everything because:
  1. Less hyperparameter tuning required
  2. The generalization gap is small enough to accept
  3. Faster to reach good performance even if peak is slightly lower

For maximum generalization on CNNs: SGD+momentum+cosine+long training.
For everything else: AdamW.
```

---

*End of Chapter 8. Chapter 9 (Regularization) coming next.*
