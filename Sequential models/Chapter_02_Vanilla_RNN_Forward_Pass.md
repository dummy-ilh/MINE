# Chapter 2: Vanilla RNN — Forward Pass, Hand-Computed

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 1 (motivation)

---

## 2.1 The Equations

At every timestep `t`, a vanilla RNN cell does:

```
a_t = W_xh · x_t + W_hh · h_{t-1} + b_h        (pre-activation)
h_t = tanh(a_t)                                 (new hidden state)
y_t = W_hy · h_t + b_y                          (output, optional at every step)
```

- `x_t`: input vector at time t (dim = `d_x`)
- `h_t`: hidden state at time t (dim = `d_h`) — this is the "memory"
- `h_0`: initialized to zeros (standard default)
- `W_xh` (shape `d_h × d_x`), `W_hh` (shape `d_h × d_h`), `W_hy` (shape `d_y × d_h`) — **same three matrices reused at every timestep**
- `b_h`, `b_y`: biases

**Key thing to internalize:** `h_t` depends on `h_{t-1}`, which depended on `h_{t-2}`, all the way back to `h_0`. This is what gives the RNN its "memory" — and also what will cause vanishing gradients in Chapter 3/4.

## 2.2 Toy Setup

Let's use concrete numbers you can re-derive by hand. Say we're processing a 3-token sequence (e.g. embeddings for "not", "very", "good") with:

- Input dim `d_x = 2`
- Hidden dim `d_h = 2`
- Output dim `d_y = 1` (e.g., a sentiment score)

**Inputs:**
```
x1 = [1.0,  0.5]
x2 = [0.0,  1.0]
x3 = [-1.0, 0.5]
```

**Weights (fixed, given — as if already trained):**
```
W_xh = [[0.5, -0.3],
        [0.1,  0.8]]

W_hh = [[ 0.2, 0.4],
        [-0.5, 0.3]]

b_h  = [0.1, -0.1]

W_hy = [0.6, -0.2]      (row vector, since d_y=1)
b_y  = 0.05

h_0  = [0.0, 0.0]        (standard initialization)
```

## 2.3 Timestep 1 — Full Hand Computation

**Step A: `W_xh · x1`**
```
row1: 0.5(1.0) + (-0.3)(0.5) = 0.5 - 0.15 = 0.35
row2: 0.1(1.0) + 0.8(0.5)    = 0.1 + 0.4  = 0.50
→ [0.35, 0.50]
```

**Step B: `W_hh · h_0`** → since `h_0 = [0,0]`, this is `[0, 0]`.

**Step C: add + bias**
```
a_1 = [0.35+0+0.1, 0.50+0-0.1] = [0.45, 0.40]
```

**Step D: apply tanh**
```
h_1 = [tanh(0.45), tanh(0.40)] = [0.4219, 0.3799]
```

**Step E: output**
```
y_1 = 0.6(0.4219) + (-0.2)(0.3799) + 0.05
    = 0.2531 - 0.0760 + 0.05
    = 0.2272
```

## 2.4 Timestep 2 — Full Hand Computation

Now `h_1 = [0.4219, 0.3799]` feeds forward as memory.

**Step A: `W_xh · x2`**, `x2 = [0, 1.0]`
```
row1: 0.5(0) + (-0.3)(1.0) = -0.30
row2: 0.1(0) + 0.8(1.0)    =  0.80
→ [-0.30, 0.80]
```

**Step B: `W_hh · h_1`**
```
row1: 0.2(0.4219) + 0.4(0.3799)  = 0.0844 + 0.1520 = 0.2363
row2: -0.5(0.4219) + 0.3(0.3799) = -0.2110 + 0.1140 = -0.0970
→ [0.2363, -0.0970]
```

**Step C: add + bias**
```
a_2 = [-0.30+0.2363+0.1, 0.80-0.0970-0.1] = [0.0363, 0.6030]
```

**Step D: tanh**
```
h_2 = [tanh(0.0363), tanh(0.6030)] = [0.0363, 0.5397]
```

**Step E: output**
```
y_2 = 0.6(0.0363) + (-0.2)(0.5397) + 0.05 = 0.0218 - 0.1079 + 0.05 = -0.0361
```

**Notice:** `h_2`'s first component (0.0363) is much smaller than `h_1`'s (0.4219) — the memory of timestep 1 got partially "washed out" combining with `x2`. This is the beginning of the vanishing-gradient story we formalize in Ch. 4.

## 2.5 Timestep 3 — Full Hand Computation

`h_2 = [0.0363, 0.5397]`, `x3 = [-1.0, 0.5]`

**Step A: `W_xh · x3`**
```
row1: 0.5(-1.0) + (-0.3)(0.5) = -0.5 - 0.15 = -0.65
row2: 0.1(-1.0) + 0.8(0.5)    = -0.1 + 0.4  =  0.30
→ [-0.65, 0.30]
```

**Step B: `W_hh · h_2`**
```
row1: 0.2(0.0363) + 0.4(0.5397)  = 0.0073 + 0.2159 = 0.2232
row2: -0.5(0.0363) + 0.3(0.5397) = -0.0182 + 0.1619 = 0.1437
→ [0.2232, 0.1437]
```

**Step C: add + bias**
```
a_3 = [-0.65+0.2232+0.1, 0.30+0.1437-0.1] = [-0.3268, 0.3437]
```

**Step D: tanh**
```
h_3 = [tanh(-0.3268), tanh(0.3437)] = [-0.3157, 0.3305]
```

**Step E: output**
```
y_3 = 0.6(-0.3157) + (-0.2)(0.3305) + 0.05 = -0.1894 - 0.0661 + 0.05 = -0.2055
```

## 2.6 Summary Table

| t | x_t | h_t | y_t |
|---|---|---|---|
| 1 | [1.0, 0.5] | [0.4219, 0.3799] | 0.2272 |
| 2 | [0.0, 1.0] | [0.0363, 0.5397] | -0.0361 |
| 3 | [-1.0, 0.5] | [-0.3157, 0.3305] | -0.2055 |

**Do this yourself:** re-derive `h_2` and `h_3` from scratch without looking. If your numbers match to 2 decimal places, you've internalized the mechanics. This is exactly the kind of "can you trace a forward pass by hand" question that separates candidates who've memorized the equation from those who understand it.

## 2.7 Where `y_t` Comes From Matters (Interview Trap #2)

Depending on the task, you only take specific `y_t`'s:

- **Many-to-one** (e.g., sentiment classification of a whole sentence): only use `y_3` (the last timestep's output) — you discard `y_1, y_2`.
- **Many-to-many, aligned** (e.g., POS tagging — one label per word): use `y_1, y_2, y_3` all.
- **Many-to-many, unaligned** (e.g., translation — different input/output lengths): needs an encoder-decoder architecture (Chapter 12) — you can't just read off `y_t` directly.

An L5 answer proactively names which of these three regimes a given problem falls into, since it changes both architecture and loss computation.

## 2.8 Complexity / Production Note

- Per-timestep compute: `O(d_h² + d_h·d_x)` (dominated by the `W_hh` and `W_xh` matmuls).
- Total forward pass for sequence length `T`: `O(T·(d_h² + d_h·d_x))` — and critically, this is **inherently sequential in T** (can't parallelize across timesteps) because `h_t` needs `h_{t-1}`. This is the single biggest practical drawback vs. Transformers at training time, but is actually an *advantage* at inference time for streaming/online use cases (you only need `O(d_h)` state carried forward, not the whole sequence) — relevant if this ever comes up in an Apple on-device/latency-sensitive context.

## 2.9 Sample Interview Q&A

**Q: Walk me through what happens if you set `W_hh = 0`.**
A: The recurrence collapses — `h_t` would depend only on `x_t` and `b_h`, with zero contribution from history. This degenerates into applying the same feedforward layer independently at each timestep — i.e., no memory at all. It's a good way to sanity-check that you understand `W_hh` is specifically the "memory" pathway.

**Q: Why `tanh` and not `ReLU` in vanilla RNNs?**
A: `tanh` bounds the hidden state to `(-1, 1)`, which helps stabilize the recurrent multiplication over many timesteps (unbounded activations under repeated multiplication by `W_hh` can blow up). ReLU RNNs are used sometimes (e.g. IRNN with identity-initialized `W_hh`) but are more prone to exploding activations without careful initialization. This tension between saturation (vanishing gradient) and boundedness (stability) is exactly what motivates LSTM/GRU gating later.

**Q: Is `h_t` an embedding?**
A: It's a *learned representation*, similar in spirit to an embedding, but it's a *contextual, time-conditioned* representation — its value depends on everything seen up to time t, not a fixed lookup. This distinction (static vs. contextual representation) is a good thing to articulate cleanly since it also explains why contextual embeddings (ELMo, and later Transformers) were a breakthrough over static word embeddings.

## 2.10 Comprehension Check

1. Recompute `h_2` by hand from `h_1` and `x2` given above — do you get `[0.0363, 0.5397]`?
2. Which weight matrix would you inspect if you suspected the model wasn't using any history at all?
3. For a sentence-level sentiment classifier processing a 10-word sentence, which `y_t` do you use as the prediction?
4. Why is the vanilla RNN forward pass hard to parallelize across timesteps, but easy to parallelize across the *batch* dimension?

---
**Next:** Chapter 3 — Backpropagation Through Time (BPTT), hand-derived gradient for this exact example.
