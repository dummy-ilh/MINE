# Chapter 5: LSTM — Architecture & Gate Intuition

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 4 (you now know *exactly* what problem this chapter solves)

---

## 5.1 The One-Sentence Fix

Vanilla RNN's problem (Ch. 4): gradient flow through time is a **repeated matrix multiplication** (`W_hhᵀ`), which geometrically vanishes or explodes.

LSTM's fix: introduce a second recurrent pathway — the **cell state `C_t`** — whose update across time is dominated by **elementwise multiplication by a gate**, not a shared weight matrix. This makes the gradient path `dC_t/dC_{t-1} ≈ f_t` (a number between 0 and 1, chosen by the *data*, not a fixed matrix reused blindly every step).

This is the single idea to hold onto through all the equations below.

## 5.2 Two State Vectors Now, Not One

| | Vanilla RNN | LSTM |
|---|---|---|
| Recurrent state(s) | `h_t` only | `C_t` (cell state — long-term memory) **and** `h_t` (hidden state — short-term/output-facing) |
| Update mechanism | `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)` — one non-linear blend | `C_t` updated mostly *additively*; `h_t` derived from `C_t` |

Think of `C_t` as a **conveyor belt** running through the whole sequence, and the gates as **valves** that control what gets dropped onto the belt, what gets removed from it, and how much of it is read off into the visible hidden state at this timestep.

## 5.3 The Three Gates (each is just a sigmoid-activated mini-layer)

All three gates have the *identical* mathematical form — a linear layer followed by sigmoid, producing values in `(0,1)` that act as "how much to let through":

```
f_t = σ(W_xf · x_t + W_hf · h_{t-1} + b_f)     — FORGET gate
i_t = σ(W_xi · x_t + W_hi · h_{t-1} + b_i)     — INPUT gate
o_t = σ(W_xo · x_t + W_ho · h_{t-1} + b_o)     — OUTPUT gate
```

**Forget gate (`f_t`):** "How much of the old cell state should I keep?" `f_t ≈ 1` → keep everything; `f_t ≈ 0` → erase it. (Example: seeing a new sentence subject might trigger forgetting the old subject's gender/number info.)

**Input gate (`i_t`):** "How much of the new candidate information should I write in?" Paired with a **candidate cell value**:
```
C̃_t = tanh(W_xc · x_t + W_hc · h_{t-1} + b_c)     — candidate (proposed new content, in [-1,1])
```
`i_t` scales how much of `C̃_t` actually gets added.

**Output gate (`o_t`):** "How much of the (updated) cell state should I expose as the hidden state this timestep?" Not everything in long-term memory needs to be relevant to the *immediate* output.

## 5.4 The Cell State Update — the Critical Equation

```
C_t = f_t ⊙ C_{t-1}  +  i_t ⊙ C̃_t
```

Read this as: **"keep some fraction of the old memory, and add in some fraction of new information."** Both operations are **elementwise** (`⊙`), not matrix multiplication. This is the additive superhighway.

## 5.5 The Hidden State (Output) Update

```
h_t = o_t ⊙ tanh(C_t)
```

`tanh(C_t)` squashes the cell state to `(-1,1)` for output purposes (the cell state itself is *not* squashed — it can grow, which is intentional, it's meant to accumulate), and `o_t` decides how much of that to reveal as this timestep's hidden state.

## 5.6 Why This Fixes Vanishing Gradients (Preview — full derivation in Ch. 7)

Differentiate the cell-state recurrence:
```
∂C_t/∂C_{t-1} = f_t     (+ smaller terms from f_t, i_t, C̃_t depending on C_{t-1} through h_{t-1} — secondary path)
```

Compare to vanilla RNN's `∂h_t/∂h_{t-1} = W_hh · diag(1-h²)` — a **fixed matrix** applied identically regardless of content.

Here, `∂C_t/∂C_{t-1} = f_t` is a **gate value the network learns to set close to 1** whenever it wants to preserve information over long distances. If the model learns "keep forgetting-gate near 1 for this information," gradient flows through with *almost no decay*, no matter how many timesteps pass — because you're now chaining multiplications by numbers the network can deliberately keep near 1, rather than being at the mercy of a fixed matrix's spectral radius.

**This is the core interview insight**: LSTM doesn't eliminate the multiplicative chain — it makes the multiplier *learnable and content-dependent* rather than *fixed and content-independent*.

## 5.7 Parameter Count (Interview-Relevant)

For `d_x`-dim input and `d_h`-dim hidden state, LSTM has **4 sets** of `(W_x, W_h, b)` — one each for forget, input, output, candidate:
```
Total params ≈ 4 × (d_h·d_x + d_h² + d_h)
```
vs. vanilla RNN's single set: `d_h·d_x + d_h² + d_h`. **LSTM has ~4× the parameters of a vanilla RNN at the same hidden size** — a common interview gotcha when comparing model capacity/latency at "equivalent" hidden dimensions (relevant to Apple's on-device latency/size constraints).

## 5.8 Interview Talking Points (L5 Signal)

- "The forget gate is the single most important LSTM component — the original 1997 LSTM didn't have one (added in 2000 by Gers et al.) and couldn't reset its memory, which caused unbounded cell-state growth on continual streams."
- "LSTM's gates are *content-dependent* gradient highways — they let the network learn *when* to preserve gradient, rather than the vanilla RNN's *fixed, uniform* treatment of every timestep."
- "The 4× parameter overhead vs. vanilla RNN is a real production consideration — this is exactly the trade-off GRU (Ch. 8) addresses by merging gates."

## 5.9 Sample Interview Q&A

**Q: What happens if the forget gate saturates at exactly 1.0 and the input gate at exactly 0.0 for a sustained period?**
A: The cell state becomes literally constant (`C_t = C_{t-1}`), providing perfect gradient flow (`∂C_t/∂C_{t-1} = 1`) for as long as that holds — this is the mechanism that lets LSTMs bridge dependencies over hundreds of timesteps, something vanilla RNNs structurally cannot do regardless of training.

**Q: Is the cell state `C_t` bounded like the hidden state `h_t`?**
A: No — `C_t` is not passed through a squashing nonlinearity in its update (only `tanh(C_t)` is used downstream to produce `h_t`). This is deliberate: it lets `C_t` act as an accumulator that can hold large magnitude information, while `h_t` (which feeds into the next gates and the output layer) stays bounded for numerical stability.

**Q: Why do we need both a hidden state `h_t` and cell state `C_t` — why not just use `C_t` everywhere?**
A: Separating them lets the network decouple "what to remember long-term" (`C_t`) from "what's relevant to expose right now" (`h_t`). Without this separation, you'd be forced to expose your entire long-term memory at every timestep even when only a small part of it is relevant to the immediate output — the output gate gives the model selective "read access."

## 5.10 Comprehension Check

1. Write out, from memory, all 4 LSTM equations (3 gates + candidate) and the 2 update equations (cell state, hidden state).
2. Why is the cell-state gradient path `f_t` described as "content-dependent" rather than fixed, and why does that matter for long sequences?
3. Roughly how many more parameters does an LSTM have than a vanilla RNN at the same hidden size, and why?
4. What would happen (structurally) if you removed the output gate and just set `h_t = tanh(C_t)` always?

---
**Next:** Chapter 6 — LSTM forward pass, fully hand-computed on a toy scalar example across 3 timesteps.
