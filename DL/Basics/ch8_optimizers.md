# Chapter 8: Optimizers (SGD, Momentum, RMSProp, Adam) — FAANG Master Interview Notes

---

> **How to use this document:** Every section you already knew is now interview-hardened. All original content is preserved and expanded with conceptual Q&A, trap questions, memory math, and the "why behind the why" that FAANG interviewers probe for.

---

## 8.1 The Plain-English Picture

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

### 📌 Conceptual Q&A — Section 8.1

**Q: What is the difference between an optimizer and gradient descent? Are they the same thing?**

A: Gradient descent is a special case of an optimizer — specifically, the simplest one. All optimizers use gradients, but they differ in how they transform gradients into weight updates. Gradient descent uses the raw gradient scaled by a fixed learning rate. Optimizers like Adam transform that gradient using momentum, adaptive scaling, bias correction, and other mechanisms. "Gradient descent" describes the core principle; "optimizer" describes the specific algorithm used to implement it.

**Q: Why can't we just use a very small learning rate to solve the ravine oscillation problem in SGD?**

A: It "solves" oscillation by making all steps tiny — including progress in the right direction. The fundamental problem is that the loss landscape has wildly different curvatures in different directions (anisotropy). A single scalar learning rate cannot simultaneously be small enough to avoid overshooting in steep directions AND large enough to make meaningful progress in flat directions. Adaptive optimizers solve this by using a per-parameter learning rate — not a single scalar.

**Q: What exactly is a "saddle point" and why do optimizers care about it?**

A: A saddle point is a location where the gradient is zero in all directions, but it's neither a minimum nor a maximum — it's a minimum in some directions and a maximum in others (like the center of a saddle). In high-dimensional spaces (millions of parameters), saddle points are far more common than local minima. SGD with noise can escape saddle points by accident. Momentum helps explicitly — it builds up velocity in consistent directions, letting the optimizer coast through flat saddle regions. Adam's adaptive scaling gives large steps in directions with small recent gradients, helping escape flat saddle plateaus.

**Q: Does the choice of optimizer affect the final model quality or just the speed of training?**

A: Both. Speed is the obvious effect: Adam converges in fewer steps than SGD. But the optimizer also determines *which minimum* the network converges to. SGD tends to find flatter minima (better generalization). Adam tends to find sharper minima (slightly worse generalization but faster convergence). The optimizer is part of the implicit regularization of the training procedure — not just a hyperparameter for speed.

---

## 8.2 Vanilla SGD (Stochastic Gradient Descent)

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

### 📌 Conceptual Q&A — Section 8.2

**Q: What exactly is "stochastic" about SGD? If we use a full batch, is it still SGD?**

A: "Stochastic" refers to using a random mini-batch (or single sample) to estimate the gradient, rather than computing the exact gradient over the full dataset. The exact gradient would require a full pass through all N training examples — prohibitively expensive at N=1 billion. Full-batch gradient descent is theoretically cleaner but practically impossible at scale. Mini-batch SGD is the compromise: use a subset (e.g., 32 or 256 examples), compute the gradient, update, repeat. With a full batch, it's technically "gradient descent," not "stochastic gradient descent," but in deep learning literature the term SGD almost always means mini-batch SGD.

**Q: Why is the mini-batch gradient an unbiased estimate of the full-batch gradient?**

A: If examples are drawn uniformly at random, each mini-batch is an independent random sample from the data distribution. The expected value of the mini-batch gradient equals the full-batch gradient: E[g_mini-batch] = g_full-batch. Unbiasedness means the mini-batch gradient doesn't systematically point in the wrong direction — just a noisier estimate of the right direction. This is why SGD can converge despite noisy gradients.

**Q: If SGD is so limited, why do some production systems (e.g., Google's image classification training) still use it?**

A: For well-conditioned problems where you have time to tune the learning rate schedule, SGD + momentum achieves better generalization than Adam. The anisotropy problem (ravine oscillation) is less severe in CNNs on image data than in transformers on text. And SGD has 0 extra memory overhead — for a 100B parameter model, that's a huge practical advantage. The choice is: Adam (fast convergence, slightly worse final accuracy, no tuning needed) vs. SGD+momentum (slower convergence, better final accuracy, requires careful lr schedule). For ImageNet competitions and research benchmarks, SGD wins. For rapid prototyping and transformers, Adam wins.

**Q: What is the condition on learning rate η for SGD to converge? What breaks if η is too large?**

A: For a convex function with Lipschitz-continuous gradients (gradient changes by at most L per unit step), SGD converges when η < 2/L. In the ravine example, L = 200 (the largest eigenvalue of the Hessian), so η < 0.01 is required. If η > 2/L: the step in the steep direction overshoots the minimum and flips to the other side, then overshoots again — oscillation diverges. More precisely, the weight update is multiplied by |1 - η·λ| each step where λ is the curvature. For convergence: |1 - η·λ| < 1, i.e., η < 2/λ.

**Q: What is gradient clipping and when is it used with SGD?**

A: Gradient clipping scales the gradient down if its norm exceeds a threshold: if ‖g‖ > clip_value, then g ← g × (clip_value / ‖g‖). It prevents gradient explosion — when the loss surface is very steep and a single step would be catastrophically large. Used primarily in RNNs (where gradients through long sequences can explode exponentially) and in large transformer training. It's a safety valve, not a precision tool. Typical clip_value: 1.0.

---

## 8.3 SGD with Momentum

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

### 📌 Conceptual Q&A — Section 8.3

**Q: Why is the effective look-back window for momentum equal to 1/(1-β)?**

A: The velocity is an exponentially weighted sum: vₜ = Σₖ βᵏ · gₜ₋ₖ. The weights sum to Σₖ₌₀^∞ βᵏ = 1/(1-β). So the "effective number of past steps" contributing roughly equal weight is approximately 1/(1-β) — the number of steps at which the cumulative weight reaches ~63% of the total. For β=0.9: 1/0.1 = 10 steps. For β=0.99: 1/0.01 = 100 steps. This is the same formula as the "halflife" of an EMA.

**Q: Can momentum overshoot a minimum? What happens if β is too large?**

A: Yes. Momentum can carry the parameter past a minimum due to accumulated velocity. If β is too large (close to 1), the effective window is very long — the optimizer "remembers" many past gradients and accelerates too much. In a tight bowl-shaped minimum, momentum causes the optimizer to spiral or oscillate around the minimum instead of settling. This is why β=0.9 is the standard: it's empirically balanced between acceleration and stability. Higher β values require smaller learning rates to compensate.

**Q: What does Nesterov momentum solve specifically?**

A: Standard momentum computes the gradient at the current position and then takes a combined step (gradient step + momentum). By the time the gradient is computed, you're about to move — so you're computing the gradient from the wrong place. Nesterov first takes the momentum step to the "expected" future position, then computes the gradient there, then corrects. The gradient is now evaluated at a more relevant position. Formally, this improves the convergence rate from O(1/t) to O(1/t²) for convex problems — the same asymptotic rate as Conjugate Gradient methods. Practically: slightly faster early convergence, especially in ravine-like geometries.

**Q: Derive the steady-state velocity in a constant-gradient scenario.**

A: If g is constant at every step, the velocity converges to a fixed point where vₜ = vₜ₋₁ = v*:
```
v* = β · v* + g
v* - β·v* = g
v*(1 - β) = g
v* = g / (1 - β)

Parameter update per step = η · v* = η · g / (1-β)
Effective learning rate = η / (1-β)

For β=0.9:  effective lr = η / 0.1 = 10η
For β=0.99: effective lr = η / 0.01 = 100η
```
Momentum effectively multiplies the learning rate by 1/(1-β) in constant-gradient directions. This is the acceleration effect — it's not just damping oscillations, it's also amplifying consistent gradients.

**Q: How does momentum interact with batch size? If you double the batch size, what happens to the effective velocity?**

A: Doubling the batch size reduces gradient noise but doesn't change the gradient magnitude in expectation. The velocity v accumulates the same expected gradient, just with less variance. The practical implication: with larger batches, you can use larger β (longer effective window) without instability, because the gradients are less noisy. Facebook's linear scaling rule for ResNet training: when doubling batch size, scale the learning rate linearly (η → 2η) and linearly scale warmup steps. The momentum term itself is typically left unchanged.

---

## 8.4 AdaGrad

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

### 📌 Conceptual Q&A — Section 8.4

**Q: Why is dividing by √G instead of G? What is the geometric intuition?**

A: The denominator √Gₜ is the root mean square (RMS) of historical gradients. Dividing by g instead of √g would make the update scale as 1/g² — an overly aggressive correction that would collapse learning rates too quickly for moderately large gradients. Dividing by √g creates a scale-invariant update: if you multiply all gradients by a constant c, Gₜ scales by c², √Gₜ scales by c, and the update η/√Gₜ · g scales by (1/c) · c = 1 — unchanged. This makes AdaGrad invariant to the scale of the gradients, which is the desired property.

**Q: AdaGrad was designed for sparse gradients. What does "sparse gradient" mean in this context?**

A: A sparse gradient is one where most components are zero at any given step. This happens naturally in NLP word embeddings: each mini-batch contains only a subset of vocabulary tokens, so the gradient for most token embeddings is exactly 0 most of the time. For token "xylophone," Gₜ grows slowly (only the rare steps when xylophone appears in the batch add to G). So η/√Gₜ stays large, giving xylophone a large effective learning rate exactly when it does appear. For "the," G grows fast (every batch), so the effective lr is small. This is precisely the right adaptive behavior for sparse data.

**Q: If AdaGrad "dies," why was it ever used in production? What was it good for?**

A: AdaGrad was state-of-the-art for training word embeddings (2013-2015) because: (1) Training word embeddings converges quickly — not many steps needed before G→∞ becomes a problem. (2) For convex problems, AdaGrad has provably optimal convergence for sparse gradients. (3) It was the first algorithm to automatically solve the sparse-feature learning rate problem. Its death sentence for deep learning came when training became multi-epoch over massive datasets with non-convex loss — the decaying lr prevented reaching good solutions. RMSProp was created specifically to fix this.

---

## 8.5 RMSProp

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

### 📌 Conceptual Q&A — Section 8.5

**Q: What is the difference between AdaGrad and RMSProp in one sentence?**

A: AdaGrad accumulates all past squared gradients (sum), causing the learning rate to monotonically shrink to zero; RMSProp uses an exponential moving average of squared gradients, so the denominator reflects only *recent* gradient magnitude, keeping the effective learning rate non-zero throughout training.

**Q: RMSProp has no bias correction. Does its S term have the same initialization bias as Adam's v term?**

A: Yes — S₀=0 means S₁ = (1-ρ)·g₁², which understates the true E[g²] by factor (1-ρ). But RMSProp doesn't correct for this. The consequence: early in training, S is too small → effective lr is too large → potential instability in the first ~1/(1-ρ) steps. This is one reason Adam improved on RMSProp: adding bias correction makes the first steps well-calibrated. In practice, RMSProp is often used with a smaller global lr to compensate.

**Q: Why is RMSProp particularly effective for RNNs?**

A: RNNs process sequences with variable-length dependencies. The gradient landscape changes dramatically over the course of processing a long sequence — gradients at different time steps have wildly different magnitudes (vanishing/exploding through many time steps). RMSProp's per-parameter adaptive scaling handles this naturally: parameters receiving large gradients (recent time steps) get small lr; parameters receiving tiny gradients (distant time steps) get large lr. This automatic adjustment was what made RNNs trainable in the pre-LSTM era and is why Hinton introduced it in the context of his RNN lecture.

**Q: RMSProp uses only the second moment (g²). What information does it lack compared to Adam?**

A: RMSProp lacks the first moment (the gradient mean/direction). It adapts step sizes per parameter but doesn't smooth the gradient direction. Concretely: if gradients alternate sign (+g, -g, +g, -g), RMSProp sees S → g² (large), takes small steps — good for damping oscillations. But it doesn't accumulate directional momentum — it takes a small step in the current gradient direction regardless of history. Adam adds the first moment, so even when the current gradient points in a noisy direction, the accumulated mean (momentum) provides a cleaner direction estimate. RMSProp scales; Adam scales and directs.

---

## 8.6 Adam: Adaptive Moment Estimation

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

### 📌 Conceptual Q&A — Section 8.6

**Q: Adam is called "Adaptive Moment Estimation." What are the "moments" and what does "adaptive" refer to?**

A: In statistics, the k-th moment of a random variable X is E[Xᵏ]. For the gradient distribution:
- First moment: E[g] = mean of gradients → estimated by mₜ
- Second moment: E[g²] = mean of squared gradients → estimated by vₜ

"Adaptive" means the learning rate is adapted per-parameter based on these moment estimates: large second moment → large historical gradient magnitude → small effective lr. "Estimation" means m and v are running estimates of the true moments of the gradient distribution.

**Q: Why is β₂=0.999 (not 0.9) for the second moment? Shouldn't they match?**

A: They serve different purposes and operate on different timescales. The first moment (β₁=0.9) is a direction smoother — you want it to respond quickly to direction changes (10-step window). The second moment (β₂=0.999) is a gradient magnitude estimator — you want a stable, low-variance estimate of the typical gradient scale. Variance estimates require more data to be reliable than mean estimates. Using a 1000-step window for the second moment makes it stable, preventing the adaptive lr from fluctuating wildly with noisy gradient magnitudes.

**Q: What happens to Adam when all gradients are zero (e.g., after convergence)?**

A: Both m and v decay exponentially toward zero. v → 0 means √v̂ → 0, and the parameter update η · m̂/(√v̂ + ε) → η · 0/ε = 0. So Adam naturally stops updating once gradients consistently vanish. The ε term prevents division by zero. In contrast, without ε, 0/0 would be undefined. This is also why ε matters: if gradients are tiny but nonzero and v̂ is near zero, ε prevents the effective lr from becoming infinitely large.

**Q: Can you implement Adam from scratch in PyTorch?**

A:
```python
def adam_step(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """
    params: list of parameter tensors
    grads:  list of gradient tensors (same shape as params)
    m, v:   lists of moment tensors (initialized to zeros)
    t:      step count (integer, starts at 1)
    """
    for i, (p, g) in enumerate(zip(params, grads)):
        # Update moments
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g ** 2
        
        # Bias correction
        m_hat = m[i] / (1 - b1 ** t)
        v_hat = v[i] / (1 - b2 ** t)
        
        # Parameter update
        p -= lr * m_hat / (v_hat.sqrt() + eps)
    
    return params, m, v

# Usage:
m = [torch.zeros_like(p) for p in model.parameters()]
v = [torch.zeros_like(p) for p in model.parameters()]
for t in range(1, num_steps + 1):
    grads = [p.grad for p in model.parameters()]
    params = [p.data for p in model.parameters()]
    params, m, v = adam_step(params, grads, m, v, t)
```

**Q: What is Adam's convergence guarantee? Does it provably converge?**

A: The original Adam paper proved convergence for convex objectives under standard assumptions. However, Reddi et al. (2018) showed a counterexample where Adam can fail to converge on non-convex problems — it can cycle without converging. They proposed AMSGrad as a fix: track the maximum of all past v̂ values instead of the current one. In practice: Adam converges reliably on deep learning problems despite this theoretical caveat, because practical loss landscapes are not adversarially constructed. The convergence guarantee failure is a pathological edge case, not a real training concern.

**Q: How does Adam interact with batch normalization?**

A: Batch normalization normalizes pre-activations to zero mean and unit variance before the activation function. This has an important implication for Adam: BN makes the loss landscape more isotropic (more similar curvature in all directions) by preventing activation scale explosions. Less anisotropy → less need for adaptive lr → Adam's per-parameter scaling is less critical but still helpful. Conversely, without BN (e.g., in transformers using LayerNorm), the gradient scales across parameters can vary by orders of magnitude, making Adam's adaptive scaling essential.

---

## 8.7 AdamW: Adam with Decoupled Weight Decay

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

### 📌 Conceptual Q&A — Section 8.7

**Q: Prove concisely that L2 regularization and weight decay are equivalent in SGD but not in Adam.**

A: In SGD, L2 regularization adds λθ to the loss gradient, giving update: `θ ← θ - η(g + λθ) = θ(1 - ηλ) - ηg`. Weight decay directly multiplies θ by (1-ηλ): same result. In Adam, L2 modifies the gradient fed into the moment estimates: `m ← β₁m + (1-β₁)(g + λθ)`, then `θ ← θ - η·m̂/(√v̂+ε)`. The weight decay term λθ gets processed through the adaptive scaling denominator √v̂. Parameters with large v̂ (historically large gradients) have their regularization reduced by √v̂ — they receive less shrinkage than intended. AdamW bypasses this by applying `- η·λ·θ` as a separate additive term, independent of the adaptive lr scaling.

**Q: Why does AdamW use β₂=0.95 instead of 0.999 in large language model training?**

A: This is a hyperparameter that the LLM community empirically found works better for long training runs. β₂=0.999 has an effective window of 1000 steps — it's a very slow EMA. Early in training this is necessary for stability (slow update to v). But in very long training runs (millions of steps), a 1000-step window means v is tracking average gradient magnitude over a very long recent history. β₂=0.95 (window ≈ 20 steps) makes v more responsive to current gradient statistics, which improves convergence at large scale. Chinchilla scaling laws paper and LLaMA both report β₂=0.95 as better for billion-parameter models.

**Q: Does AdamW change the gradient computation or only the weight update?**

A: Only the weight update. The gradient gₜ = ∂L/∂θ is computed from the loss function alone — weight decay is not added to the loss. The weight decay term `- η·λ·θₜ` is added directly to the parameter update step, after the Adam moment update and bias correction. This is the "decoupling" — weight decay and gradient learning are now two separate, independent operations. Consequence: the second moment v doesn't "see" the weight decay term, so large-gradient parameters still have their weight decay applied at full strength.

**Q: When should you use Adam vs. AdamW? Is there ever a reason to prefer vanilla Adam?**

A: Use AdamW as the default for any problem where you want weight regularization. Use vanilla Adam only: (1) when you have no regularization (λ=0, making them identical), (2) for extremely short training runs where regularization doesn't matter, or (3) when reproducing older results that used Adam specifically. In all transformer-based work (NLP, vision transformers, multimodal models), AdamW is the standard. The improvement over Adam is significant enough (1-3% on downstream benchmarks) that there's no reason to use Adam with weight decay instead of AdamW.

---

## 8.8 Optimizer Comparison

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

### 📌 Conceptual Q&A — Section 8.8

**Q: What is Nadam and when is it useful?**

A: Nadam = Nesterov + Adam. It replaces Adam's standard momentum first moment with the Nesterov lookahead variant. The update is:
```
θₜ₊₁ = θₜ - η/(√v̂ₜ + ε) · [β₁·m̂ₜ + (1-β₁)·gₜ/(1-β₁ᵗ)]
```
The gradient contribution is added directly using the current gradient (Nesterov correction) rather than going through the stale moment. Nadam tends to converge slightly faster than Adam in the early steps of training. It's used when you want the best of both worlds: Nesterov's lookahead accuracy plus Adam's adaptive scaling. In practice, the difference from Adam is small — most practitioners don't bother unless squeezing out every bit of performance.

**Q: How does memory overhead translate to real GPU VRAM in a practical example?**

A:
```
BERT-Large (340M parameters):
  Model weights (float32):     340M × 4 bytes = 1.36 GB
  Gradients:                   1.36 GB
  Adam m:                      1.36 GB
  Adam v:                      1.36 GB
  Total:                       5.44 GB
  GPU: fits on a single 8GB GPU (barely)

LLaMA-7B (7B parameters):
  Model weights (bfloat16):    7B × 2 bytes = 14 GB
  Adam m (float32):            7B × 4 bytes = 28 GB
  Adam v (float32):            7B × 4 bytes = 28 GB
  Gradients (float32):         28 GB
  Total:                       98 GB
  GPU: requires 8× A100 80GB GPUs just for this
  → 8-bit Adam or ZeRO are not optional at this scale
```

Note: In mixed-precision training, model weights are bfloat16 (2 bytes) but optimizer states are kept in float32 (4 bytes) for numerical stability. This asymmetry is intentional and standard.

**Q: What is Lion optimizer and how does it differ from Adam?**

A: Lion (EvoLved Sign Momentum, Chen et al. 2023) is a memory-efficient optimizer discovered via evolutionary search. Update rule:
```
uₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ
θₜ₊₁ = θₜ - η·(sign(uₜ) + λ·θₜ)   (weight decay decoupled)
mₜ = β₂·mₜ₋₁ + (1-β₂)·gₜ
```
Key differences from Adam: (1) Uses sign(u) instead of m̂/(√v̂+ε) — every parameter gets the same step magnitude η, only direction varies. (2) Needs only one moment vector (m), not two (m and v) → 1.5× memory vs Adam's 3× (model + m vs model + m + v). (3) Requires smaller learning rate (typically 3-10× smaller than Adam) because all steps are ±η regardless of gradient magnitude. Reported to match or beat AdamW on image and language tasks with less memory. Not yet widely adopted in production but gaining traction.

---

## 8.9 Worked Numerical Example: 5 Steps of Adam

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

### 📌 Conceptual Q&A — Worked Example

**Q: In step 3, Adam continues moving in the positive direction despite a negative gradient. Is this correct behavior or a bug?**

A: It's correct and intentional — this is momentum in action. The accumulated first moment m₃ = 0.0825 still points positive (net of all past gradients). The current gradient of -0.3 is a single step that could be noise. Adam's momentum says: "we've been consistently going positive for 2 steps, one negative gradient doesn't override that." The step is reduced (0.00053 instead of 0.001) reflecting the weakened first moment, but direction is maintained. This is the feature: momentum smooths direction changes, making training more robust to noisy gradients.

**Q: At step 1, bias-corrected m̂₁ = g₁ exactly. Is this a coincidence?**

A: No — it's by design. At t=1, m₁ = (1-β₁)·g₁. Bias correction: m̂₁ = m₁/(1-β₁¹) = (1-β₁)·g₁/(1-β₁) = g₁. So at step 1, Adam's first moment estimate exactly equals the current gradient — the bias correction makes it as if there's no history (because there isn't). Similarly, v̂₁ = g₁² exactly. So at step 1, Adam behaves like SGD with lr=η/|g₁| — a normalized gradient step. The history-building happens from step 2 onward.

---

## 8.10 Learning Rate Schedules with Adam

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

### 📌 Conceptual Q&A — Section 8.10

**Q: Why does cosine decay outperform linear decay or constant lr?**

A: Cosine decay has a specific property: it decays slowly at first and quickly in the middle, then slowly again at the end. The slow start allows exploration in a wider neighborhood after initial convergence. The slow end allows fine-grained refinement near the final minimum. Linear decay decays at constant rate — it's too fast initially (preventing exploration) and too slow at the end (not refining enough). Constant lr never refines. Empirically, cosine decay consistently beats both by 0.5-2% on image and language benchmarks with the same total training budget.

**Q: Why do some papers use "cosine annealing with warm restarts" (SGDR)?**

A: SGDR (Loshchilov & Hutter, 2016) periodically restarts the cosine schedule, resetting lr from η_max, then decaying again. This allows the optimizer to escape sharp minima it may have converged to: the high lr after each restart lets it jump out. After the restart, it converges to a potentially flatter (better-generalizing) minimum. It's like escaping local optima through controlled perturbation. Used primarily in image classification where many sharp local minima exist. Less common in NLP where the loss landscape is smoother and restarts disrupt learned representations.

**Q: How do you determine the peak learning rate η_max? Is there a systematic way?**

A: The learning rate range test (Leslie Smith, 2017): run training for a few hundred steps, increasing lr exponentially from very small to very large. Plot loss vs. lr. Loss initially decreases (lr is productive), then plateaus, then diverges (lr too large). Choose η_max at the steepest downward slope — just before the plateau. For Adam, η_max ≈ 1e-3 to 3e-4 is typical and often works without explicit range testing. For SGD+momentum, the optimal lr varies more widely and range testing is more valuable.

**Q: What is the relationship between warmup steps W and model size?**

A: Larger models require more warmup steps because: (1) More parameters → more gradient statistics to accumulate for reliable Adam moment estimates. (2) Deeper models → gradients at lower layers are noisier early in training. (3) Higher training throughput → more data seen per step, but the *number of steps* for warmup scales with model depth, not data volume. A rule of thumb: W ≈ 2000-4000 steps for BERT-base (110M params), W ≈ 4000-8000 for BERT-large (340M), W ≈ larger still for billion-scale models. Some LLM recipes use W = 2% of total training steps.

---

## 8.11 Why This Matters — What Breaks If You Get This Wrong

1. **Using vanilla SGD on a transformer.** Transformers have highly non-uniform gradient scales across layers (attention parameters vs. embedding parameters). SGD treats all parameters identically. Training either diverges (lr too high for some params) or stalls (lr too low for others). Adam's per-parameter adaptation is not optional for transformers — it's required.

2. **Forgetting bias correction in Adam implementation.** If you implement Adam without the (1-β₁ᵗ) and (1-β₂ᵗ) corrections, the first few hundred steps use effectively zero learning rate (m and v are near zero, unbiased). Loss barely moves. Training looks broken. The fix is two division operations — trivial to add, catastrophic to omit.

3. **Using Adam when SGD generalizes better.** A well-known empirical finding (Wilson et al., 2017): Adam often converges to sharper minima that generalize worse than SGD's flatter minima. For image classification (ResNets on ImageNet), SGD + momentum with cosine learning rate decay often outperforms Adam by 1-2% top-1 accuracy. Use Adam when training is hard (transformers, RNNs). Use SGD+momentum when you care about squeezing out maximum generalization (CNNs).

4. **Not decoupling weight decay (using Adam instead of AdamW).** Standard Adam applies weight decay through L2 in the gradient, which gets rescaled by the adaptive learning rate. Parameters with large gradients get less regularization than intended. AdamW decouples the two. For transformers, this is the difference between a model that overfit and one that generalizes — often 1-3% on downstream tasks.

5. **Learning rate too high without warmup.** On large models (>100M parameters), starting Adam at lr=3e-4 from step 1 causes loss spikes or NaN in the first ~50 steps, before moment estimates stabilize. The solution is trivial (linear warmup over 1000-4000 steps) but easy to overlook. A training run that crashes at step 47 after 2 hours of compute is an expensive lesson.

---

### 📌 Additional Failure Mode Q&A

**Q: My Adam training loss is decreasing but validation loss is increasing from step 1 (immediate overfitting). The optimizer seems fine. What's wrong?**

A: If Adam is working but generalization fails immediately, the issue is not the optimizer but one of: (1) Data leakage — train/val split is contaminated. (2) Model too large for the dataset — reduce capacity or add regularization (weight decay λ=0.01-0.1 in AdamW). (3) lr too high — even with warmup, if η_max is too large, Adam takes overshooting steps that find sharp memorizing minima. Try 3-10× lower lr. (4) No dropout or insufficient dropout — add dropout between layers.

**Q: What does a "loss spike" during Adam training look like and how do you debug it?**

A: A loss spike is a sudden large increase in training loss (e.g., loss jumps from 2.1 to 15.7 in one step), often followed by partial or full recovery. Causes: (1) Bad batch with outlier gradients that exceed Adam's moment adaptation speed. (2) Learning rate too high at that point in training. (3) Gradient explosion in a particular layer (e.g., attention weights). Diagnosis: log per-layer gradient norms. Debug: add gradient clipping (clip_grad_norm=1.0), reduce lr, or increase warmup steps. Prevention: gradient clipping is standard practice for transformers precisely because attention softmax can produce sudden large gradients.

---

## 8.12 Google/Apple-Level Interview Q&A

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

## 8.13 Additional FAANG Interview Questions

---

**Q4: "What is gradient clipping and how does it interact with Adam? Does Adam 'need' gradient clipping less than SGD?"**

A: Gradient clipping scales the gradient vector if its global norm exceeds a threshold:
```
if ‖g‖ > clip_max:
    g ← g × (clip_max / ‖g‖)

PyTorch: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Adam does NOT make gradient clipping unnecessary. Here's why: Adam normalizes each parameter's update by its individual gradient history (via √v̂). But if a single step has an extremely large gradient (e.g., 100× the typical magnitude), Adam's second moment v is a slow EMA — it hasn't "caught up" yet, so the denominator is too small and the step is enormous.

```
Example: typical g ≈ 0.1, sudden spike g = 50.0
  v (EMA with β₂=0.999) after spike: v ≈ 0.001×50² ≈ 2.5
  (still mostly tracking old, small gradients)
  
  Update = η × m̂/√v̂ ≈ η × 50/√2.5 ≈ η × 31.6
  
  This is a 316× larger step than intended.
  
  With clipping at max_norm=1.0: g → 0.02 (clipped)
  Adam sees a small gradient → small, controlled update.
```

Standard practice: use gradient clipping AND Adam together. For transformers: `clip_grad_norm=1.0` is essentially universal.

---

**Q5: "What is the difference between online learning (SGD with batch size 1) and mini-batch SGD? When would you prefer each?"**

A:
```
Online learning (batch size = 1):
  Each weight update uses exactly one training example.
  
  Pros:
  + Maximum update frequency — weights updated after every example
  + Can adapt to non-stationary data distributions (streaming data)
  + No need to store the full dataset (works with infinite data streams)
  + Naturally escapes local minima via high variance
  
  Cons:
  - Extremely noisy gradient estimates (variance = N × mini-batch variance)
  - Cannot parallelize across examples (one example at a time)
  - Very sensitive to learning rate — high variance needs very small lr
  - Slow per-step training on modern GPU hardware (underutilizes parallelism)

Mini-batch SGD (batch size 32-512):
  Each update uses a subset of training examples.
  
  Pros:
  + Parallelizes over GPU cores — 32× more efficient than batch size 1
  + Lower variance gradient estimates → more stable training
  + Enables BatchNorm (requires statistics over multiple examples)
  + Modern optimizers (Adam) assume ~IID gradient noise; mini-batch satisfies this
  
  Cons:
  - Requires loading batch into memory
  - More hyperparameter choices (batch size)

When to prefer online learning:
  1. Streaming data (real-time recommendation systems, online RL)
  2. Data that is non-stationary (distribution shifts over time)
  3. Extremely large datasets where full passes are expensive
  
When to prefer mini-batch:
  Almost always for deep learning training on fixed datasets.
  Batch size 32-256 is the sweet spot for GPU utilization and gradient quality.
```

---

**Q6: "Explain the relationship between learning rate, batch size, and gradient noise. What is the 'linear scaling rule' and when does it break down?"**

A: Mini-batch gradient is the mean gradient over B examples: g_batch = (1/B) Σᵢ gᵢ. The variance of the mean scales as Var[g_batch] = σ²/B where σ² is the per-example gradient variance. So doubling B halves gradient variance — the gradient estimate is more accurate.

The linear scaling rule (Goyal et al., Facebook, 2017): when multiplying batch size by k, multiply learning rate by k.

```
Intuition:
  With B examples and lr η: gradient = g, step = η·g
  With 2B examples and lr 2η: gradient = g (same direction, less noisy),
                               step = 2η·g
  
  To make the SAME expected parameter update per training example:
  Original: η·g per B examples → η·g/B per example
  Scaled:   2η·g per 2B examples → 2η·g/(2B) = η·g/B per example ✓
  
  Same update per example → same training dynamics.
```

When it breaks down:
1. **Very large batch sizes (B > 8192):** The linear scaling rule assumes gradients are in the "noise-dominated" regime. For very large B, gradients become nearly deterministic (noise → 0) and the rule breaks. Training dynamics change qualitatively.
2. **Early training:** High noise, large gradient magnitudes — warmup overrides the linear scaling. Always combine the rule with extended warmup steps proportional to batch size increase.
3. **Very small batch sizes:** The rule assumes the variance reduction approximation holds, which requires B >> 1.

Practical: use the linear scaling rule for 2-8× batch size increases from a known baseline. For larger increases, use the square-root scaling rule: scale lr by √k instead of k.

---

**Q7: "How does optimizer choice interact with mixed-precision training (FP16/BF16)?"**

A: Mixed-precision training uses 16-bit floats for the forward pass (weights, activations, gradients) but keeps a full 32-bit "master copy" of weights for optimizer state.

```
MEMORY LAYOUT (mixed-precision):
  Model weights (inference copy): bf16   — 2 bytes/param
  Model weights (master copy):    fp32   — 4 bytes/param (for optimizer update)
  Adam m:                         fp32   — 4 bytes/param
  Adam v:                         fp32   — 4 bytes/param
  Gradients:                      fp16   — 2 bytes/param (accumulated in fp32)

Total: ~16 bytes/param (vs 12 bytes/param pure fp32: weights + m + v)

WHY KEEP OPTIMIZER STATE IN FP32?
  Adam's moment updates require fine-grained precision:
  mₜ = 0.9 × mₜ₋₁ + 0.1 × gₜ
  
  If m is near zero (small weight update), fp16 underflows:
  fp16 minimum positive value: ~6×10⁻⁵
  A learning rate update of η × m̂/(√v̂) = 0.001 × 0.001 = 10⁻⁶ underflows fp16.
  
  fp32 (minimum ~10⁻³⁸) handles this without underflow.
  Optimizer states in fp32 are non-negotiable for stable training.

BF16 vs FP16:
  BF16 has same dynamic range as FP32 (8 exponent bits) but less precision (7 mantissa).
  FP16 has less dynamic range but more precision (5 exponent, 10 mantissa).
  
  For large model training: BF16 is preferred because gradient magnitudes
  can vary by many orders of magnitude (dynamic range matters more than precision).
  FP16 risks overflow in attention logits; BF16 does not.
  A100/H100 GPUs have native BF16 support — this is why modern LLM training uses BF16.
```

---

## 8.14 Quick-Fire Recall: Things Interviewers Expect You to Know Cold

```
OPTIMIZER FORMULAS (memorize for whiteboard):

SGD:
  θ ← θ - η·g

SGD + Momentum:
  v ← β·v + g
  θ ← θ - η·v
  Steady-state effective lr: η/(1-β)

RMSProp:
  S ← ρ·S + (1-ρ)·g²
  θ ← θ - η·g/√(S + ε)

Adam (5 steps):
  1. g = ∂L/∂θ
  2. m ← β₁·m + (1-β₁)·g
  3. v ← β₂·v + (1-β₂)·g²
  4. m̂ = m/(1-β₁ᵗ),  v̂ = v/(1-β₂ᵗ)
  5. θ ← θ - η·m̂/(√v̂ + ε)

AdamW (add one term):
  5. θ ← θ - η·m̂/(√v̂ + ε) - η·λ·θ

DEFAULT HYPERPARAMETERS:
  Adam:  β₁=0.9, β₂=0.999, η=0.001, ε=1e-8
  AdamW: β₁=0.9, β₂=0.95,  η=3e-4,  ε=1e-8, λ=0.1

MEMORY OVERHEAD:
  SGD:          0 extra buffers
  SGD+Momentum: 1× θ (velocity v)
  Adam/AdamW:   2× θ (m and v)

BIAS CORRECTION WINDOW (when it matters):
  β₁=0.9:   matters for first ~100 steps
  β₂=0.999: matters for first ~1000 steps

EFFECTIVE LOOK-BACK WINDOW:
  Formula: 1/(1-β)
  β=0.9  → 10 steps
  β=0.99 → 100 steps
  β=0.999 → 1000 steps

OPTIMIZER SELECTION GUIDE:
  CNN image classification:    SGD + Momentum + cosine lr (best generalization)
  Transformer / NLP:           AdamW (required)
  RNN / LSTM:                  RMSProp or Adam
  Sparse data / NLP embeddings: Adam/AdamW (handles sparse gradients)
  Memory constrained (>7B):    Adafactor or 8-bit Adam
  Multi-GPU training:          ZeRO + Adam

WHEN SGD > ADAM:
  - Well-conditioned loss landscape (CNNs on images)
  - Time available for lr tuning
  - Maximum generalization is the goal
  - Homogeneous gradient scales

WHEN ADAM > SGD:
  - Heterogeneous gradient scales (transformers, embeddings)
  - No time for lr tuning (Adam's defaults work)
  - Deep/complex models where SGD diverges without careful scheduling
  - Sparse gradients

LEARNING RATE SCHEDULE STANDARD:
  Warmup steps: ~1-5% of total steps (min 1000, up to 4000 for BERT-scale)
  Decay: cosine from η_max to η_min = η_max/10
  Gradient clipping: max_norm=1.0 (always for transformers)

ADAM STABILITY: what causes early divergence
  1. No bias correction → effective lr 31× too large at step 1
  2. No warmup → same issue for deep networks
  3. No gradient clipping → single bad batch destroys moment estimates
  4. Weight decay via L2 (not AdamW) → over/under regularized parameters
```

---

*End of Chapter 8 — Master Notes Edition. Chapter 9 (Regularization) coming next.*
