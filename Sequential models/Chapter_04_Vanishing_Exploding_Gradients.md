# Chapter 4: Vanishing & Exploding Gradients — Proven Numerically

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 3 (BPTT)

---

## 4.1 Recap: the Recursive Gradient

From Chapter 3, propagating a hidden-state gradient back `k` steps is:

```
dL/dh_{t-k} = [ Πᵢ₌₀ᵏ⁻¹  W_hhᵀ · diag(1 - h_{t-i}²) ]  ·  dL/dh_t
```

Two things multiply together at **every single step** of this product:
1. **`W_hhᵀ`** — the same matrix, over and over
2. **`diag(1 - h²)`** — the tanh derivative, which is `≤ 1` always, and `→ 0` whenever `h` saturates near `±1`

This is a "double whammy" — two independent sources of shrinkage (or one source of shrinkage plus a source of growth), compounding multiplicatively.

## 4.2 Ingredient 1: Spectral Radius of `W_hh`

The long-run behavior of repeatedly multiplying by a matrix `M` is governed by its **spectral radius** `ρ(M)` = the magnitude of its largest eigenvalue.

- If `ρ(M) < 1`: `Mᵏ → 0` as `k → ∞` (vanishing)
- If `ρ(M) > 1`: `Mᵏ → ∞` as `k → ∞` (exploding)
- If `ρ(M) = 1`: stable, neither — the "sweet spot" (motivates orthogonal initialization of `W_hh` in practice)

**Let's compute this for our Chapter 2/3 `W_hh`:**
```
W_hh = [[0.2, 0.4], [-0.5, 0.3]]

trace = 0.2 + 0.3 = 0.5
det   = (0.2)(0.3) - (0.4)(-0.5) = 0.06 + 0.20 = 0.26

Eigenvalues λ satisfy: λ² - (trace)λ + det = 0
λ² - 0.5λ + 0.26 = 0
discriminant = 0.5² - 4(0.26) = 0.25 - 1.04 = -0.79   (negative → complex eigenvalues)

|λ| = √det = √0.26 ≈ 0.510
```

**Spectral radius ≈ 0.51.** This is `< 1` — so this `W_hh` is in the *vanishing* regime, before we even account for the tanh-derivative shrinkage.

## 4.3 Numerical Demo: Vanishing Gradient Over a Longer Sequence

Ignoring the tanh factor for a moment (i.e., an upper bound — the real decay is at least this fast, usually faster), the gradient magnitude ratio after `k` steps is approximately `ρ(W_hh)^k = 0.51^k`:

| Steps back (k) | Gradient magnitude retained (≈ 0.51^k) | Interpretation |
|---|---|---|
| 1 | 0.510 | 51% of signal remains |
| 5 | 0.0345 | 3.5% remains |
| 10 | 0.00119 | 0.1% remains |
| 15 | 0.0000410 | 0.004% remains |
| 20 | 0.00000142 | Effectively zero |

**By 20 timesteps back, the gradient signal is ~1.4 millionths of its original size.** Any dependency on a token 20+ steps in the past is, for practical purposes, **untrainable** — the weight updates carrying that information are indistinguishable from numerical noise.

**Now add the tanh factor.** If hidden units are even moderately saturated (say `h ≈ 0.8`, giving `1-h² = 0.36`), each step multiplies in an *additional* 0.36 shrinkage on top of the 0.51 from `W_hh`:
```
effective decay per step ≈ 0.51 × 0.36 ≈ 0.184
0.184^10 ≈ 0.00000135   (even faster vanishing than the W_hh-only estimate)
```

**Interview point:** this is exactly why vanilla RNNs are described as struggling with dependencies beyond ~10-20 tokens in practice, even though there's no hard architectural limit — it's a *gradient signal-to-noise* problem, not a memory-capacity problem per se.

## 4.4 Numerical Demo: Exploding Gradient

Now consider a differently-initialized `W_hh'`:
```
W_hh' = [[1.5, 0.4], [-0.5, 1.3]]

trace = 2.8
det   = (1.5)(1.3) - (0.4)(-0.5) = 1.95 + 0.20 = 2.15

discriminant = 2.8² - 4(2.15) = 7.84 - 8.60 = -0.76  (complex again)
|λ| = √2.15 ≈ 1.466
```

**Spectral radius ≈ 1.47 — greater than 1.** Even though `tanh'≤1` provides *some* damping, if the weight growth outpaces it (common early in training or with poor initialization), gradients explode:

| Steps back (k) | Gradient magnitude (≈ 1.466^k) |
|---|---|
| 1 | 1.47 |
| 5 | 6.85 |
| 10 | 46.9 |
| 20 | 2,199 |

By 20 steps, the gradient has grown **~2,200×**. In practice this manifests as the loss suddenly spiking to `NaN` or `Inf` mid-training — often described by practitioners as the model "blowing up."

## 4.5 Why Both Failure Modes Matter for Interviews

| | Vanishing | Exploding |
|---|---|---|
| **Symptom** | Loss plateaus; model ignores long-range context; training is just... slow and stuck | Loss spikes to NaN; training destabilizes suddenly |
| **Root cause** | `ρ(W_hh) < 1` and/or saturated tanh | `ρ(W_hh) > 1` |
| **Detection** | Monitor gradient norm — stays tiny for early-timestep weights | Monitor gradient norm — spikes enormous |
| **Architecture-level fix** | LSTM/GRU gating (Ch. 5-9) — additive, not purely multiplicative, memory update | Gating helps some, but... |
| **Training-level fix** | Better init (orthogonal `W_hh`), shorter truncation window, skip/residual-style connections | **Gradient clipping** (rescale grad if norm > threshold) — the standard fix |

**Important nuance for L5-level depth:** LSTM/GRU do **not** eliminate exploding gradients — clipping is still standard practice with LSTMs/GRUs in production. What gating does is fix *vanishing* by providing an (approximately) *additive* path for gradient flow across time — decoupling gradient preservation from the multiplicative `W_hh` chain. We derive exactly how in Chapter 7.

## 4.6 Detecting This in Practice

- Log the **gradient norm** per layer/per timestep during training.
- A clean early-training diagnostic: plot `||dL/dh_t||` as a function of how far `t` is from the loss-contributing output. A vanishing RNN shows this decaying to ~0 within a handful of steps; a healthy model (or LSTM) shows a much flatter/slower decay.
- Anthropic-style interview framing: this is analogous to plotting per-layer gradient norms in a very deep feedforward net — same underlying phenomenon (repeated multiplication through a deep computational graph), just deep-in-**time** instead of deep-in-**layers**.

## 4.7 Interview Talking Points (L5 Signal)

- "Vanishing and exploding gradients are really the *same* phenomenon (repeated multiplication of Jacobians) with different spectral radii — not two separate bugs."
- "Depth in an RNN over time is mathematically the same problem as depth in a very deep feedforward network — an RNN unrolled over 100 timesteps *is* effectively a 100-layer network with tied weights. That's why the same tools (residual/skip-style connections, careful initialization, normalization) that helped deep ANNs also inform RNN design — and why LSTM's gating can be viewed as an early precursor to the ResNet skip-connection idea, just for the time dimension instead of the depth dimension."
- "Clipping fixes the symptom (numerical blow-up); it doesn't fix the underlying inability to learn long-range dependencies. Gating (LSTM/GRU) fixes the actual root cause for vanishing."

## 4.8 Sample Interview Q&A

**Q: Does using ReLU instead of tanh in an RNN solve vanishing gradients?**
A: Partially — ReLU's derivative is 1 (not `<1`) when active, removing the tanh-saturation shrinkage factor. But you still have the `W_hhᵀ` multiplicative chain, so if `ρ(W_hh) < 1` you still vanish; and ReLU RNNs are also *more* prone to exploding since there's no upper bound squashing activations. In practice this needs careful initialization (e.g., identity-initialized `W_hh`, as in the IRNN paper) to work at all — it's not a drop-in fix.

**Q: Why does gradient clipping only address exploding, not vanishing gradients?**
A: Clipping rescales the gradient vector *if its norm exceeds a threshold* — it does nothing when the norm is already small. It's a ceiling, not a floor. Vanishing gradients need an architectural fix that changes how gradient flows structurally (additive cell state in LSTM/GRU), not just a magnitude cap.

**Q: If I told you a specific RNN's `W_hh` had spectral radius exactly 1.0, would you say the vanishing/exploding problem is solved?**
A: Not entirely — that only addresses the `W_hhᵀ` factor of the product. The `diag(1-h²)` tanh-derivative factor is still `≤ 1` at every step (and `=1` only when `h=0`, i.e., no activation, which is degenerate). So the model can still vanish due to activation saturation even with a perfectly-conditioned `W_hh`. This is exactly why orthogonal initialization alone isn't a full fix — it just delays the problem, and gating (LSTM/GRU) is still needed for genuinely long sequences.

## 4.9 Comprehension Check

1. Compute the spectral radius (or just verify `ρ(M) = √det(M)` when eigenvalues are complex — why does that identity hold?) for `W_hh'' = [[0.9, 0.1],[-0.2, 0.7]]`. Is this RNN closer to vanishing or exploding?
2. Why is an RNN unrolled over T timesteps analogous to a T-layer deep feedforward network?
3. Explain why gradient clipping is still used even after you switch to LSTM/GRU.
4. In your own words, what are the "two ingredients" that multiply together in the BPTT gradient chain, and why does saturating activations make vanishing *worse* regardless of `W_hh`?

---
**Next:** Chapter 5 — LSTM architecture and gate intuition: the additive cell-state pathway that directly targets the vanishing-gradient problem you just proved.
