# Chapter 6: LSTM — Forward Pass, Hand-Computed

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapter 5 (gate equations)

---

## 6.1 Why We Use Scalars Here (Important Note)

LSTM gates operate **elementwise** — every unit of the hidden/cell state has its own independent gate values. This means the full mechanics of an LSTM are completely captured by walking through a **single scalar unit** (`d_h = 1`); a real `d_h = 128` LSTM is just this exact same computation done 128 times in parallel, one per unit, with no cross-unit interaction inside the gating math (cross-unit mixing happens only through the weight matrices producing the gate pre-activations). Using scalars here keeps every number traceable by hand while losing zero conceptual content.

## 6.2 The Equations (scalar form)

```
f_t = σ(w_xf·x_t + w_hf·h_{t-1} + b_f)
i_t = σ(w_xi·x_t + w_hi·h_{t-1} + b_i)
o_t = σ(w_xo·x_t + w_ho·h_{t-1} + b_o)
C̃_t = tanh(w_xc·x_t + w_hc·h_{t-1} + b_c)

C_t = f_t·C_{t-1} + i_t·C̃_t
h_t = o_t·tanh(C_t)
```

## 6.3 Toy Setup

```
Inputs:  x1 = 1.0,  x2 = 0.5,  x3 = -1.0
Initial: C_0 = 0,   h_0 = 0

Weights (all biases = 0 for simplicity):
  Forget:    w_xf=0.8, w_hf=0.1
  Input:     w_xi=0.5, w_hi=0.2
  Output:    w_xo=0.6, w_ho=0.3
  Candidate: w_xc=1.0, w_hc=-0.1
```

(Recall `σ(z) = 1/(1+e^-z)`; useful reference values: `σ(0.5)=0.6225`, `σ(0.6)=0.6457`, `σ(0.8)=0.6900`.)

## 6.4 Timestep 1

```
a_f = 0.8(1.0) + 0.1(0) = 0.800   →  f_1 = σ(0.800) = 0.6900
a_i = 0.5(1.0) + 0.2(0) = 0.500   →  i_1 = σ(0.500) = 0.6225
a_o = 0.6(1.0) + 0.3(0) = 0.600   →  o_1 = σ(0.600) = 0.6457
a_c = 1.0(1.0) + (-0.1)(0) = 1.000 →  C̃_1 = tanh(1.000) = 0.7616

C_1 = f_1·C_0 + i_1·C̃_1 = 0.6900(0) + 0.6225(0.7616) = 0.4741
h_1 = o_1·tanh(C_1) = 0.6457·tanh(0.4741) = 0.6457(0.4417) = 0.2852
```

**Read this in words:** at t=1, there's no prior memory (`C_0=0`), so `C_1` is built entirely from the input gate admitting 62% of the new candidate value. The output gate then reveals 65% of that (squashed) into `h_1`.

## 6.5 Timestep 2

```
a_f = 0.8(0.5) + 0.1(0.2852) = 0.4000 + 0.0285 = 0.4285  →  f_2 = σ(0.4285) = 0.6055
a_i = 0.5(0.5) + 0.2(0.2852) = 0.2500 + 0.0570 = 0.3070  →  i_2 = σ(0.3070) = 0.5762
a_o = 0.6(0.5) + 0.3(0.2852) = 0.3000 + 0.0856 = 0.3856  →  o_2 = σ(0.3856) = 0.5952
a_c = 1.0(0.5) + (-0.1)(0.2852) = 0.5000 - 0.0285 = 0.4715 →  C̃_2 = tanh(0.4715) = 0.4394

C_2 = f_2·C_1 + i_2·C̃_2 = 0.6055(0.4741) + 0.5762(0.4394) = 0.2871 + 0.2532 = 0.5403
h_2 = o_2·tanh(C_2) = 0.5952·tanh(0.5403) = 0.5952(0.4934) = 0.2937
```

**Read this in words:** the forget gate now keeps ~61% of the accumulated `C_1`, while the input gate admits ~58% of the new candidate. `C_2 = 0.5403` is *larger* than `C_1 = 0.4741` — the cell state is accumulating, as designed.

## 6.6 Timestep 3

```
a_f = 0.8(-1.0) + 0.1(0.2937) = -0.8000 + 0.0294 = -0.7706  →  f_3 = σ(-0.7706) = 0.3164
a_i = 0.5(-1.0) + 0.2(0.2937) = -0.5000 + 0.0587 = -0.4413  →  i_3 = σ(-0.4413) = 0.3915
a_o = 0.6(-1.0) + 0.3(0.2937) = -0.6000 + 0.0881 = -0.5119  →  o_3 = σ(-0.5119) = 0.3748
a_c = 1.0(-1.0) + (-0.1)(0.2937) = -1.0000 - 0.0294 = -1.0294 → C̃_3 = tanh(-1.0294) = -0.7745

C_3 = f_3·C_2 + i_3·C̃_3 = 0.3164(0.5403) + 0.3915(-0.7745) = 0.1709 - 0.3032 = -0.1323
h_3 = o_3·tanh(C_3) = 0.3748·tanh(-0.1323) = 0.3748(-0.1316) = -0.0493
```

**Read this in words:** `x_3 = -1.0` strongly negative input flips the forget gate low (0.32 — actively erasing most of the old cell state) and pushes a strongly negative candidate (-0.77) which the input gate lets through significantly (0.39) — net effect, the cell state flips sign from positive (+0.54) to negative (-0.13).

## 6.7 Summary Table

| t | x_t | f_t | i_t | o_t | C̃_t | C_t | h_t |
|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 0.6900 | 0.6225 | 0.6457 | 0.7616 | 0.4741 | 0.2852 |
| 2 | 0.5 | 0.6055 | 0.5762 | 0.5952 | 0.4394 | 0.5403 | 0.2937 |
| 3 | -1.0 | 0.3164 | 0.3915 | 0.3748 | -0.7745 | -0.1323 | -0.0493 |

**Do this yourself:** recompute `C_2` and `h_2` from `C_1, h_1, x_2` without looking — if you match to 3 decimal places, you've internalized the mechanics.

## 6.8 Comparing to Vanilla RNN (Same Inputs, Side-by-Side Intuition)

In Chapter 2's vanilla RNN, `h_t` was a single blended `tanh` of everything at once — no separate mechanism for "keep old" vs. "add new." Here, notice how explicit and inspectable the LSTM's bookkeeping is: you can literally read off "the model kept 61% of old memory and added 58% of new information" at t=2. This interpretability of gate values (even if the *learned* weights end up less clean in a real trained model) is a genuine practical advantage when debugging production sequence models — you can log gate activations and directly diagnose whether a model is "forgetting too aggressively" or "never forgetting."

## 6.9 Sample Interview Q&A

**Q: At t=3 in this example, the forget gate dropped to 0.32. What real-world linguistic scenario would cause this in a trained language model?**
A: A likely trigger is a strong topic/context shift — e.g., end of a sentence/clause, a conjunction like "but" that signals contrasting information is coming, or a new sentence subject that makes prior grammatical-agreement information irrelevant. In practice, trained LSTMs' forget gates have been shown to specialize this way (e.g., resetting more at sentence boundaries).

**Q: Given the equations, could `C_t` in principle grow unboundedly over a very long sequence?**
A: Yes — since `C_t = f_t·C_{t-1} + i_t·C̃_t` has no squashing on `C_t` itself, if `f_t` stays near 1 and `i_t·C̃_t` keeps adding same-signed increments, `C_t` can grow large. In practice this is rarely catastrophic because `tanh(C_t)` in the `h_t` computation saturates regardless of how large `C_t` gets, and gradient clipping (Ch.14) guards training stability — but it's a real and sometimes-tested interview edge case.

**Q: How would you modify this walkthrough for a real `d_h = 128` LSTM — does the math change?**
A: No — every equation above is applied independently, elementwise, to each of the 128 hidden units; the only added complexity is that `w_xf` etc. become `128×d_x` matrices producing 128 gate pre-activations at once, and `C_t, h_t` become 128-dim vectors. The scalar mechanics per-unit are identical to what we just hand-computed.

## 6.10 Comprehension Check

1. Recompute `C_2` and `h_2` from scratch — did you match Section 6.5?
2. In this example, why did the cell state flip from positive to negative at t=3, mechanically (name the two gate/candidate values responsible)?
3. Why is a "scalar" (d_h=1) LSTM walkthrough sufficient to understand a full-scale LSTM's mechanics?
4. If you saw `i_t ≈ 0` and `f_t ≈ 1` for many consecutive timesteps in a trained model's gate logs, what would you conclude the model is doing?

---
**Next:** Chapter 7 — Why LSTM gating fixes vanishing gradients: the full gradient derivation through the cell-state pathway, hand-computed on this exact example.
