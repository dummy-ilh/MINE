# Chapter 9: GRU Forward Pass — Hand-Computed, Side-by-Side with LSTM

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 6 (LSTM numbers — we deliberately reuse the same inputs and matching weights for direct comparison), Chapter 8 (GRU equations)

---

## 9.1 Setup (same inputs as Chapter 6, weights chosen to mirror LSTM's gates)

```
Inputs:  x1 = 1.0,  x2 = 0.5,  x3 = -1.0
Initial: h_0 = 0

Weights (scalar, biases = 0):
  Update:    w_xz=0.5, w_hz=0.2     (deliberately same numbers as LSTM's input gate, Ch.6)
  Reset:     w_xr=0.6, w_hr=0.3
  Candidate: w_xh=1.0, w_hh=-0.1    (same as LSTM's candidate weights, Ch.6)
```

## 9.2 Timestep 1

```
a_z = 0.5(1.0) + 0.2(0) = 0.500     →  z_1 = σ(0.500) = 0.6225
a_r = 0.6(1.0) + 0.3(0) = 0.600     →  r_1 = σ(0.600) = 0.6457
a_h̃ = 1.0(1.0) + (-0.1)(r_1·h_0) = 1.0 + (-0.1)(0.6457·0) = 1.000  →  h̃_1 = tanh(1.000) = 0.7616

h_1 = (1-z_1)·h_0 + z_1·h̃_1 = (1-0.6225)(0) + 0.6225(0.7616) = 0.4741
```

**Notice:** `h_1 = 0.4741` — identical to LSTM's `C_1` from Chapter 6! This isn't a coincidence: with `h_0=0`, the reset gate has nothing to act on yet, and we deliberately matched `w_xz,w_hz` to LSTM's input-gate weights and the candidate weights exactly. This lets you see clearly that **GRU's `h_t` plays the role of LSTM's `C_t` (accumulator), not LSTM's `h_t` (filtered output)** — GRU has no separate output-gate filtering step.

## 9.3 Timestep 2

```
a_z = 0.5(0.5) + 0.2(0.4741) = 0.2500 + 0.0948 = 0.3448   →  z_2 = σ(0.3448) = 0.5854
a_r = 0.6(0.5) + 0.3(0.4741) = 0.3000 + 0.1422 = 0.4422   →  r_2 = σ(0.4422) = 0.6088

a_h̃ = 1.0(0.5) + (-0.1)(r_2 · h_1) = 0.5 + (-0.1)(0.6088 × 0.4741)
     = 0.5 - 0.1(0.2886) = 0.5 - 0.0289 = 0.4711          →  h̃_2 = tanh(0.4711) = 0.4392

h_2 = (1-z_2)·h_1 + z_2·h̃_2 = (0.4146)(0.4741) + (0.5854)(0.4392)
    = 0.1966 + 0.2572 = 0.4537
```

**Notice the reset gate in action:** `r_2 = 0.6088` scaled down `h_1`'s contribution to the candidate computation (`r_2·h_1 = 0.2886` vs. `h_1 = 0.4741` on its own) — the candidate is computed with partially-suppressed history, exactly the mechanism described in Chapter 8.

## 9.4 Timestep 3

```
a_z = 0.5(-1.0) + 0.2(0.4537) = -0.5000 + 0.0907 = -0.4093  →  z_3 = σ(-0.4093) = 0.3991
a_r = 0.6(-1.0) + 0.3(0.4537) = -0.6000 + 0.1361 = -0.4639  →  r_3 = σ(-0.4639) = 0.3861

a_h̃ = 1.0(-1.0) + (-0.1)(r_3 · h_2) = -1.0 + (-0.1)(0.3861 × 0.4537)
     = -1.0 - 0.1(0.1752) = -1.0 - 0.0175 = -1.0175         →  h̃_3 = tanh(-1.0175) = -0.7697

h_3 = (1-z_3)·h_2 + z_3·h̃_3 = (0.6009)(0.4537) + (0.3991)(-0.7697)
    = 0.2726 - 0.3072 = -0.0346
```

## 9.5 Summary Table (GRU)

| t | x_t | z_t | r_t | h̃_t | h_t |
|---|---|---|---|---|---|
| 1 | 1.0 | 0.6225 | 0.6457 | 0.7616 | 0.4741 |
| 2 | 0.5 | 0.5854 | 0.6088 | 0.4392 | 0.4537 |
| 3 | -1.0 | 0.3991 | 0.3861 | -0.7697 | -0.0346 |

## 9.6 Direct Side-by-Side: LSTM (Ch. 6) vs. GRU (this chapter)

| t | LSTM `h_t` | GRU `h_t` | LSTM `C_t` |
|---|---|---|---|
| 1 | 0.2852 | 0.4741 | 0.4741 |
| 2 | 0.2937 | 0.4537 | 0.5403 |
| 3 | -0.0493 | -0.0346 | -0.1323 |

**Key visual takeaway:** GRU's `h_t` tracks much closer to LSTM's `C_t` (the unfiltered accumulator) than to LSTM's `h_t` (which is damped by the output gate — LSTM's `o_t` values were 0.65, 0.60, 0.37, all `<1`, shrinking the exposed state). This is the concrete, numerical version of the architectural point from Chapter 8: **GRU always exposes its full state; LSTM can choose to hide part of it.**

## 9.7 Parameter/Compute Comparison, Concretely

For this toy scalar example (`d_x=d_h=1`): LSTM used **4** weight pairs `(w_x, w_h)` = 8 scalar weights; GRU used **3** pairs = 6 scalar weights. At `d_h=1` this looks small, but recall from Chapter 8 that at realistic scale (`d_h=256`, say) this 25% difference translates into real parameter-count and matmul-FLOPs savings — directly relevant when discussing on-device/latency-constrained deployment (Apple context) in an interview.

## 9.8 Interview Talking Points (L5 Signal)

- "If you hand-trace both architectures on identical inputs, you see GRU's hidden state numerically resembles LSTM's *cell* state more than its *hidden* state — because GRU has no output-gate filtering step. This is a clean way to demonstrate genuine structural understanding beyond memorized equations."
- "The reset gate's effect is visible directly in the arithmetic: it's the coefficient scaling `h_{t-1}` inside the candidate computation, distinct from the update gate's role of blending old vs. new *after* the candidate is computed."
- "Empirically (e.g. Chung et al. 2014, Jozefowicz et al. 2015), GRU and LSTM are close in performance across many tasks with no universal winner — so a rigorous answer names the structural trade-off (coupled gating, no output gate, fewer params) rather than claiming one is definitively 'better.'"

## 9.9 Sample Interview Q&A

**Q: If you had to guess, without running numbers, which would have a *smoother* trajectory of hidden state values across time — LSTM's `h_t` or GRU's `h_t` — and why?**
A: LSTM's `h_t`, generally — because the output gate provides an additional filtering/damping step that can smooth out swings in the underlying cell state before exposing it. GRU exposes its accumulator state directly, so its `h_t` can show larger swings, as seen numerically above (GRU's `h_t` values are notably larger in magnitude than LSTM's at t=1,2).

**Q: In production, would you expect GRU or LSTM to converge faster in wall-clock training time, at matched hidden size?**
A: GRU, generally — fewer parameters means less compute per step (roughly 25% fewer matmul FLOPs in the recurrent cell) and a smaller parameter search space, both of which typically translate to faster wall-clock training and inference, though this needs to be weighed against any task-specific accuracy difference found via validation.

**Q: Could you get GRU-like behavior out of an LSTM by tying `i_t = 1 - f_t` and fixing `o_t = 1`?**
A: Yes, approximately, as discussed in Chapter 8 — this collapses LSTM's independent forget/input gating into GRU's coupled update-gate behavior and removes LSTM's output filtering, though LSTM would still lack GRU's reset-gate mechanism inside candidate computation, so it's an approximation, not an exact equivalence.

## 9.10 Comprehension Check

1. Recompute `h_2` for GRU from scratch — does it match `0.4537`?
2. Why does GRU's `h_1` numerically equal LSTM's `C_1` in this specific setup, and what does that reveal architecturally?
3. Which GRU gate is responsible for scaling down `h_1`'s contribution when computing `h̃_2`, and is this the same job as the update gate?
4. Name one scenario where you'd choose LSTM over GRU despite the parameter savings, and justify it structurally (not just "it might perform better").

---
**Next:** Chapter 10 — Bidirectional RNN/LSTM: architecture, when forward-only context isn't enough, and why you can never use bidirectional models for real-time streaming.
