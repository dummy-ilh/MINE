# Chapter 7: Why LSTM Gating Fixes Vanishing Gradients — Derived

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapters 4 (vanishing gradients), 6 (LSTM forward numbers we reuse here)

---

## 7.1 The Computational Graph: Two Paths from `C_{t-1}` to `C_t`

```
C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ C̃_t
```

`C_{t-1}` influences `C_t` through **two distinct paths**:

1. **Direct path:** the `f_t ⊙ C_{t-1}` term itself — `C_{t-1}` multiplied by gate `f_t`.
2. **Indirect path:** `C_{t-1}` also influenced `h_{t-1}` (via `h_{t-1} = o_{t-1}⊙tanh(C_{t-1})`), and `h_{t-1}` feeds into computing `f_t, i_t, C̃_t` themselves at the next step.

**The direct path is the dominant "gradient superhighway"** and is what nearly all textbook explanations (and interviews) focus on. The indirect path is real, handled correctly by autodiff in practice, but is a second-order effect on top of the main story — we'll note it but not belabor it.

## 7.2 Direct-Path Derivative: the Key Formula

Treating gates as locally constant with respect to this specific edge (the standard "immediate partial derivative" used for this argument):

```
∂C_t/∂C_{t-1} = f_t
```

**This is the entire fix, in one line.** Compare:

| | Vanilla RNN | LSTM (cell-state path) |
|---|---|---|
| Recurrence | `h_t = tanh(W_hh h_{t-1} + ...)` | `C_t = f_t ⊙ C_{t-1} + ...` |
| Local derivative | `W_hhᵀ · diag(1-h²)` — **fixed matrix**, same every step regardless of input | `f_t` — **elementwise, learned, content-dependent scalar per unit** |
| Structure | Multiplicative through a shared weight matrix | *Affine* in `C_{t-1}` — like `y = mx + b`, whose derivative w.r.t. `x` is just the coefficient `m` |

The critical shift: in vanilla RNN, the "multiplier" is baked into fixed weights and applies uniformly at every timestep no matter what the data says. In LSTM, the multiplier **is a gate the network computes fresh at every timestep based on context** — and critically, the network **can learn to push `f_t → 1`** whenever it needs to preserve information over a long horizon, decoupling gradient preservation from any fixed spectral radius.

## 7.3 Chaining Across `k` Steps

```
∂C_t/∂C_{t-k}  ≈  Πⱼ₌₀ᵏ⁻¹ f_{t-j}     (elementwise product of forget gates, NOT a matrix product)
```

Two structural differences from the vanilla RNN case (Ch. 4) matter enormously:
- This is a product of **scalars in (0,1)** per unit (elementwise), not a matrix product — no eigenvalue/spectral-radius analysis needed; each unit's memory decay is independent and directly interpretable.
- Each `f_{t-j}` is **learned and content-dependent** — the network can and does learn different forget-gate behavior for different features (some units specialize in short-term info with low `f`, others in long-term info with `f≈1`).

## 7.4 Numerical Demo, Part 1: Our Actual Chapter 6 Forget Gates

Recall the forget gate values we hand-computed: `f_1=0.6900, f_2=0.6055, f_3=0.3164`.

```
∂C_3/∂C_0 ≈ f_1 · f_2 · f_3 = 0.6900 × 0.6055 × 0.3164 ≈ 0.1322
```

**Honest observation:** this is actually *similar in magnitude* to the vanilla RNN's 3-step decay from Chapter 4 (`0.51³ ≈ 0.133`)! This is an important, easily-missed nuance:

> **LSTM does not automatically guarantee better gradient flow — it only makes strong gradient flow *achievable* when the model learns to push forget gates near 1.** With these particular (untrained, illustrative) weights, the forget gates aren't saturated near 1, so we don't yet see the benefit.

## 7.5 Numerical Demo, Part 2: What Happens When Forget Gates Learn to Saturate

Now suppose training has pushed a given unit's forget gate to consistently sit around `f_t ≈ 0.95` (this is realistic — trained LSTMs do learn near-saturated forget gates for units carrying long-range information):

| Steps back (k) | LSTM: `0.95^k` | Vanilla RNN: `0.51^k` (from Ch. 4) |
|---|---|---|
| 5 | 0.774 | 0.0345 |
| 10 | 0.599 | 0.00119 |
| 20 | 0.358 | 0.00000142 |
| 50 | 0.0769 | ~0 (astronomically small) |

**At 20 steps, LSTM retains 35.8% of the gradient signal vs. vanilla RNN's 0.00014%** — a difference of roughly **250,000×**. This is the number to have ready in an interview: it's not that LSTM "solves" vanishing gradients as a mathematical guarantee, it's that LSTM gives the network **the mechanism to choose to preserve gradient** when the task needs it, whereas vanilla RNN has no such lever at all — its decay rate is baked into `W_hh` and applies indiscriminately.

## 7.6 The Hidden-State Gradient Path (Secondary, briefly)

`h_t = o_t ⊙ tanh(C_t)` also creates a gradient path back through `o_t` and the `tanh(C_t)` nonlinearity — this path *does* still have a `tanh'` shrinkage factor, similar in spirit to vanilla RNN. But since `h_t` isn't the primary long-term memory carrier (`C_t` is), this path mattering less for long-range dependencies is by design — it's meant to affect the *immediate* output, not to be the long-haul memory highway.

## 7.7 Interview Talking Points (L5 Signal)

- "LSTM converts a **fixed multiplicative decay** into a **learned, content-dependent, elementwise multiplicative decay** — the mathematical form of the recurrence is still a product across time, but the *base* of that product is no longer fixed by initialization/architecture; it's something gradient descent can push toward 1 for units that need to remember."
- "This is why interview answers like 'LSTM prevents vanishing gradients' are subtly wrong — the more precise statement is 'LSTM makes long-range gradient preservation *learnable*, rather than architecturally impossible.'"
- "Even with LSTM, if forget gates are initialized or trained to be small, you can still get vanishing gradients on that specific unit — which is exactly why practitioners often **initialize the forget gate bias to a positive value (e.g., 1 or 2)** at the start of training, biasing `f_t` toward "remember by default" and only learning to forget when the data justifies it."

## 7.8 Sample Interview Q&A

**Q: Does LSTM completely solve the vanishing gradient problem?**
A: Not as an absolute guarantee — it solves the *architectural* obstruction, by replacing a fixed-matrix multiplicative recurrence with a content-dependent, elementwise gated one. Whether gradients actually flow well over long distances still depends on what the forget gates learn; a poorly-trained or poorly-initialized LSTM can still exhibit substantial decay. What LSTM guarantees is that strong gradient flow is *achievable* by the model, which vanilla RNN structurally cannot achieve regardless of training.

**Q: Why do practitioners often initialize the forget gate bias to a positive constant?**
A: `σ(z)` is close to 1 for `z` moderately positive, so a positive bias (e.g., `+1` or `+2`) makes `f_t ≈ 0.73–0.88` right from initialization, biasing the network toward preserving memory by default. This avoids the early-training pathology where a near-zero-initialized forget gate causes the model to erase its cell state constantly before it's learned anything worth remembering, which can stall learning of long-range dependencies from the start.

**Q: Contrast the gradient path through `C_t` vs. through `h_t` in an LSTM.**
A: The `C_t` path is (locally) affine in `C_{t-1}` with coefficient `f_t` — a clean, controllable gradient highway. The `h_t` path additionally passes through a `tanh` nonlinearity (`h_t = o_t⊙tanh(C_t)`) and the output gate, which reintroduces some saturation-based shrinkage — but this path is intentionally the "read-out" mechanism for the *current* timestep, not the long-term memory carrier, so its shrinkage matters less for long-range dependency learning.

## 7.9 Comprehension Check

1. Why is `∂C_t/∂C_{t-1} = f_t` and not something involving a weight matrix, given the cell-state update equation?
2. Using our actual Chapter 6 forget gates (0.69, 0.6055, 0.3164), is this particular untrained LSTM example actually better than the vanilla RNN at preserving 3-step gradient? Why might a trained model behave differently?
3. Why does initializing the forget-gate bias to a positive value help early training?
4. Is it accurate to say "LSTMs solve vanishing gradients"? Give the more precise one-sentence version.

---
**Next:** Chapter 8 — GRU architecture: how merging LSTM's forget+input gates into one "update gate," and dropping the separate cell state, gives 25% fewer parameters with (often) comparable performance.
