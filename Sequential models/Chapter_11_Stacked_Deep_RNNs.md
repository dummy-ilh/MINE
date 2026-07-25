# Chapter 11: Stacked (Deep) RNNs

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 2 (we reuse layer-1 outputs as layer-2 inputs)

---

## 11.1 The Idea: Depth in the "Feature" Dimension

Everything so far has one recurrent layer: input `x_t → h_t → output`. Just as a single-hidden-layer ANN is often less expressive than a deep one, a **single-layer RNN's hidden state may not have enough representational capacity** to capture complex hierarchical structure in a sequence (e.g., low-level syntax vs. higher-level semantics in language).

**The fix: stack multiple recurrent layers.** The hidden-state *output* of layer `l` at every timestep becomes the *input* to layer `l+1` at that same timestep:

```
h_t^(1) = Cell^(1)(x_t,        h_{t-1}^(1))     — layer 1, same as before
h_t^(2) = Cell^(2)(h_t^(1),    h_{t-1}^(2))     — layer 2 takes layer 1's output as its "input"
h_t^(3) = Cell^(3)(h_t^(2),    h_{t-1}^(3))     — and so on
```

Each layer has **its own independent set of weights** and **its own recurrent connection through time** (`h_{t-1}^(l) → h_t^(l)`, within that layer only). Depth here is orthogonal to the time dimension — you now have two separate axes of "depth": across time (T steps) and across layers (L layers).

## 11.2 Numerical Illustration: One Step of Layer 2

Reusing Chapter 2's layer-1 output: `h_1^(1) = [0.4219, 0.3799]` (that was layer 1's hidden state at t=1). Now feed this *as the input* to a second layer with its own weights:

```
W_xh^(2) = [[0.3, 0.1], [0.2, -0.4]]
W_hh^(2) = [[0.1, 0.2], [0.3, -0.1]]
b_h^(2)  = [0, 0.05]
h_0^(2)  = [0, 0]
```

**Step A:** `W_xh^(2) · h_1^(1)`
```
row1: 0.3(0.4219) + 0.1(0.3799) = 0.1266 + 0.0380 = 0.1646
row2: 0.2(0.4219) + (-0.4)(0.3799) = 0.0844 - 0.1520 = -0.0676
→ [0.1646, -0.0676]
```

**Step B:** `W_hh^(2) · h_0^(2) = [0, 0]` (since `h_0^(2) = [0,0]`)

**Step C:** add bias and apply tanh
```
a = [0.1646+0, -0.0676+0.05] = [0.1646, -0.0176]
h_1^(2) = [tanh(0.1646), tanh(-0.0176)] = [0.1632, -0.0176]
```

`h_1^(2)` is now the *layer-2* hidden state at t=1 — a "higher-level" representation built on top of layer 1's output. The exact same recipe continues for `t=2, 3` (feeding `h_2^(1), h_3^(1)` as layer-2 inputs, with layer-2's own recurrence through `h_1^(2), h_2^(2)`), following the identical pattern from Chapter 2.

## 11.3 Why This Is Trickier Than It Looks (Interview-Relevant)

Stacking compounds the vanishing-gradient concern from Chapter 4 along a **second axis**. Gradient now has to flow back through both:
- **Time**, within each layer (as in Ch. 3-4)
- **Layers**, at each timestep (an additional chain of Jacobians per layer, similar in flavor to a deep feedforward net)

This is exactly why:
- Stacked RNNs in practice are usually shallow (**2-4 layers** is typical; going much deeper without mitigation rarely helps and often hurts optimization).
- **Inter-layer residual/skip connections** are common in deeper stacked RNN/LSTM architectures (e.g., adding `h_t^(l-1)` directly to `h_t^(l)`) — the same ResNet-style fix used in deep ANNs, applied across the layer axis here.
- **Dropout between layers** (specifically *variational* dropout — using the same dropout mask at every timestep within a layer, rather than resampling every step) is standard regularization for stacked RNNs; naive per-timestep-resampled dropout tends to interfere destructively with the recurrent memory.

## 11.4 Interview Talking Points (L5 Signal)

- "Stacking adds a second, independent 'depth' axis (layers) on top of the existing time-depth axis — gradient health has to be reasoned about along both, not just the time dimension covered in Ch. 3-4."
- "This is precisely analogous to why very deep feedforward/CNN architectures needed residual connections (ResNet) to train well past a certain depth — stacked RNNs face the same wall, and the same class of fix (skip connections across layers) applies."
- "Practical depth for RNN-family models tends to plateau much earlier (2-4 layers) than for CNNs/Transformers (dozens to hundreds of layers) — largely because the *time* axis is already contributing significant depth-like optimization difficulty on its own, so architects are usually conservative about compounding it further with many stacked layers."

## 11.5 Sample Interview Q&A

**Q: If a single-layer LSTM with hidden size 512 underperforms, would you first try doubling the hidden size, or stacking a second layer of size 512?**
A: Depends on the failure mode — if the model seems to lack raw *capacity* (underfitting even on training data), widening (bigger hidden size) is often the simpler, more stable first move. If the model fits training data reasonably but seems to miss *hierarchical* structure (e.g., needs to compose low-level and high-level patterns), stacking is more targeted, but comes with the added optimization difficulty discussed above — I'd validate both empirically rather than assume either wins by default.

**Q: Why would you use variational dropout instead of standard dropout in a stacked LSTM?**
A: Standard dropout, resampled independently at every timestep, effectively corrupts the recurrent hidden-state pathway with a different random mask at every step, which can severely disrupt the model's ability to maintain consistent long-term memory. Variational dropout fixes the same dropout mask across all timesteps within a given layer/sequence, so the *recurrent* connections see a consistent regularization pattern rather than incoherent per-step noise.

**Q: Does stacking help with the vanishing gradient problem from Chapter 4?**
A: Not directly, and in fact it can make optimization harder by adding another axis gradients must flow through cleanly. Stacking is about representational *capacity/hierarchy*, not about fixing gradient flow — that's LSTM/GRU's job (Ch. 5-9). The two are complementary but solve different problems, and it's a common interview mistake to conflate "deeper model" with "better long-range gradient flow."

## 11.6 Comprehension Check

1. What becomes the "input" to layer 2 at each timestep, in a stacked RNN?
2. Name the two independent "depth" axes present in a stacked RNN, and explain briefly why both matter for gradient flow.
3. Why is variational dropout (fixed mask across time) preferred over standard per-step dropout in stacked RNNs/LSTMs?
4. True/False: stacking more layers directly addresses vanishing gradients over long time horizons. Justify your answer.

---
**Next:** Chapter 12 — Sequence-to-Sequence (Encoder-Decoder): how to handle tasks where input and output lengths differ, hand-computed for one decoding step.
