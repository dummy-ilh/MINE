# Chapter 3: Backpropagation Through Time (BPTT) — Hand-Derived

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 2 (forward pass — we reuse those exact numbers)

---

## 3.1 The Big Idea

BPTT is just standard backprop applied to the **unrolled** computation graph. The RNN reused the same `W_xh, W_hh, b_h` at every timestep, so when we backprop:

> **Each shared weight receives a gradient contribution from every timestep it was used in, and these contributions are summed.**

This is the single most important structural fact about BPTT — everything else is "regular backprop, done repeatedly and added up."

## 3.2 Setup: Loss

We'll treat this as a **many-to-one** task (Ch. 2.7) — only `y_3` feeds the loss. Say the target is `t = 1.0` and we use squared error:

```
L = 0.5 · (y_3 - target)²
```

Recall from Chapter 2: `y_3 = -0.2055`. So:

```
dL/dy_3 = (y_3 - target) = -0.2055 - 1.0 = -1.2055
```

## 3.3 The Chain Rule Skeleton

We need gradients for `W_xh, W_hh, b_h, W_hy, b_y`. The path from loss to each early-timestep weight goes **through every later hidden state**:

```
L → y_3 → h_3 → a_3 → h_2 → a_2 → h_1 → a_1 → (h_0, fixed)
                  ↑             ↑
            W_hh, W_xh, b_h used HERE at every arrow labeled a_t
```

So `dL/dh_1` isn't computed directly — it flows backward through `h_3` and `h_2` first. This backward flow through hidden states, timestep by timestep, **is** BPTT.

## 3.4 Output Layer Gradients (straightforward)

```
dL/dW_hy = dL/dy_3 · h_3ᵀ = -1.2055 · [-0.3157, 0.3305] = [0.3806, -0.3984]
dL/db_y  = dL/dy_3 = -1.2055
dL/dh_3  = dL/dy_3 · W_hy = -1.2055 · [0.6, -0.2] = [-0.7233, 0.2411]
```

`dL/dh_3` is the seed that now propagates backward through time.

## 3.5 Timestep 3 → 2 (the core recurrent step)

**Step 1 — through tanh.** Recall `tanh'(a) = 1 - tanh(a)² = 1 - h²`.
```
h_3 = [-0.3157, 0.3305]  →  1 - h_3² = [0.9003, 0.8908]
dL/da_3 = dL/dh_3 ⊙ (1 - h_3²) = [-0.7233·0.9003, 0.2411·0.8908] = [-0.6513, 0.2148]
```
(`⊙` = elementwise product)

**Step 2 — gradients w.r.t. weights used at t=3:**
```
dL/dW_hh (from t=3) = dL/da_3 ⊗ h_2ᵀ    (outer product, h_2 = [0.0363, 0.5397])
                     = [[-0.0236, -0.3515],
                        [ 0.0078,  0.1159]]

dL/dW_xh (from t=3) = dL/da_3 ⊗ x_3ᵀ    (x_3 = [-1.0, 0.5])
                     = [[ 0.6513, -0.3257],
                        [-0.2148,  0.1074]]

dL/db_h (from t=3)  = dL/da_3 = [-0.6513, 0.2148]
```

**Step 3 — propagate to `h_2`:** since `a_3 = W_xh·x_3 + W_hh·h_2 + b_h`, we have `∂a_3/∂h_2 = W_hh`, so:
```
dL/dh_2 = W_hhᵀ · dL/da_3
        = [[0.2, -0.5], [0.4, 0.3]] · [-0.6513, 0.2148]
        = [-0.2377, -0.1961]
```

**This is the single most important line in the whole chapter.** Notice we multiplied by `W_hhᵀ`. Every step further back multiplies by `W_hhᵀ` (and a tanh-derivative) *again*. That repeated multiplication is exactly what we'll show explodes or vanishes in Chapter 4.

## 3.6 Timestep 2 → 1 (repeat the same recipe)

**Through tanh:**
```
h_2 = [0.0363, 0.5397] → 1 - h_2² = [0.9987, 0.7087]
dL/da_2 = dL/dh_2 ⊙ (1-h_2²) = [-0.2377·0.9987, -0.1961·0.7087] = [-0.2374, -0.1390]
```

**Weight gradients from t=2:**
```
dL/dW_hh (from t=2) = dL/da_2 ⊗ h_1ᵀ   (h_1 = [0.4219, 0.3799])
                     = [[-0.1001, -0.0902],
                        [-0.0586, -0.0528]]

dL/dW_xh (from t=2) = dL/da_2 ⊗ x_2ᵀ   (x_2 = [0, 1.0])
                     = [[0, -0.2374],
                        [0, -0.1390]]

dL/db_h (from t=2)  = [-0.2374, -0.1390]
```

**Propagate to `h_1`:**
```
dL/dh_1 = W_hhᵀ · dL/da_2 = [[0.2,-0.5],[0.4,0.3]] · [-0.2374,-0.1390]
        = [0.0220, -0.1367]
```

## 3.7 Timestep 1 (final step — stops at `h_0`)

```
h_1 = [0.4219, 0.3799] → 1 - h_1² = [0.8220, 0.8557]
dL/da_1 = dL/dh_1 ⊙ (1-h_1²) = [0.0220·0.8220, -0.1367·0.8557] = [0.0181, -0.1169]

dL/dW_hh (from t=1) = dL/da_1 ⊗ h_0ᵀ = 0   (since h_0 = [0,0] — no contribution)
dL/dW_xh (from t=1) = dL/da_1 ⊗ x_1ᵀ  (x_1 = [1.0, 0.5])
                     = [[0.0181, 0.0091],
                        [-0.1169, -0.0585]]
dL/db_h (from t=1)  = [0.0181, -0.1169]
```

We stop here since `h_0` is a fixed constant, not a learned parameter (standard convention).

## 3.8 Summing Across Time — the Defining Step of BPTT

```
dL/dW_xh (TOTAL) = (contribution t=1) + (contribution t=2) + (contribution t=3)
                  ≈ [[0.669, -0.554],
                     [-0.332, -0.090]]

dL/dW_hh (TOTAL) = (t=1) + (t=2) + (t=3)
                  ≈ [[-0.124, -0.442],
                     [-0.051,  0.063]]

dL/db_h  (TOTAL) = (t=1) + (t=2) + (t=3)
                  ≈ [-0.871, -0.041]
```

**This is the crux of BPTT**: `W_hh` (and `W_xh`, `b_h`) get **one gradient contribution per timestep they were used**, and training sums all of them before the weight update. A 100-timestep sequence means 100 additive contributions to `dL/dW_hh` in a single backward pass.

## 3.9 The Multiplicative Chain (bridge to Chapter 4)

Look at what happened to get from `dL/dh_3` to `dL/dh_1`:

```
dL/dh_1 = W_hhᵀ · diag(1-h_1²) · [ W_hhᵀ · diag(1-h_2²) · dL/dh_3 ]
```

Generalizing to a gap of `k` timesteps:

```
dL/dh_{t-k} = [ Πᵢ  W_hhᵀ · diag(1 - h_i²) ]  ·  dL/dh_t
```

This is a **product of k Jacobian matrices**. If the dominant eigenvalue of `W_hhᵀ·diag(1-h²)` is consistently `< 1`, this product shrinks geometrically toward zero as `k` grows → **vanishing gradient**. If it's consistently `> 1`, it grows geometrically → **exploding gradient**. Chapter 4 makes this rigorous and shows it numerically with a longer sequence.

## 3.10 Interview Talking Points (L5 Signal)

- "BPTT isn't a different algorithm from backprop — it's ordinary backprop on the unrolled graph, where shared weights simply accumulate gradients from every timestep."
- "The recurrent gradient path is a **repeated matrix product**, not a repeated *sum* — that's precisely why RNNs are numerically fragile over long sequences, in a way that, say, ResNet skip-connections (additive) are not."
- "In practice, **truncated BPTT** (Ch. 14) caps how far back you propagate — you deliberately introduce bias to control this multiplicative blowup/vanish and keep training tractable."

## 3.11 Sample Interview Q&A

**Q: If your sequence has 200 timesteps, do you really backprop through all 200?**
A: In theory yes (full BPTT), but in practice this is expensive and numerically unstable, so **truncated BPTT** is standard: forward pass runs the full sequence, but gradients are only propagated back a fixed number of steps (e.g., 20–35), treating the hidden state beyond that as a constant. This trades gradient accuracy for tractability and stability.

**Q: Why do we sum gradient contributions across timesteps rather than averaging?**
A: Because it's a direct consequence of the chain rule for a variable reused multiple times in a computation graph (`dL/dW = Σₜ dL/dW|_t`) — not a design choice. If you average instead, you'd be computing a different (incorrect) gradient of the true loss.

**Q: What's the practical symptom of exploding gradients during training, and the standard fix?**
A: Loss suddenly spikes to NaN/very large values, often after training seemed stable. Standard fix: **gradient clipping** — rescale the gradient vector if its norm exceeds a threshold (Ch. 14 covers this in production detail).

## 3.12 Comprehension Check

1. Why does `dL/dW_hh` receive a separate contribution at every timestep, but `dL/dW_hh (from t=1)` turned out to be exactly zero in our example?
2. What matrix gets multiplied repeatedly as you propagate the hidden-state gradient further back in time?
3. In your own words: what's the difference between full BPTT and truncated BPTT, and why would you choose the latter in production?
4. If `W_hh` had much larger singular values (e.g., all eigenvalues > 2), would you expect vanishing or exploding gradients over a long sequence?

---
**Next:** Chapter 4 — Vanishing/Exploding Gradients, proven numerically on a longer sequence, and why this single issue justifies the entire existence of LSTM/GRU.
