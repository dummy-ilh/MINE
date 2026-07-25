# Chapter 18: Master Cheat Sheet

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN) — Final Reference
**Purpose:** Self-contained pre-interview review. Every equation, every key number, every comprehension-check answer.

---

## 18.1 Equation Reference (All Architectures)

**Vanilla RNN (Ch. 2):**
```
h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)
y_t = W_hy·h_t + b_y
```

**LSTM (Ch. 5-6):**
```
f_t = σ(W_xf·x_t + W_hf·h_{t-1} + b_f)     — forget gate
i_t = σ(W_xi·x_t + W_hi·h_{t-1} + b_i)     — input gate
o_t = σ(W_xo·x_t + W_ho·h_{t-1} + b_o)     — output gate
C̃_t = tanh(W_xc·x_t + W_hc·h_{t-1} + b_c)  — candidate
C_t = f_t⊙C_{t-1} + i_t⊙C̃_t                — cell state (additive!)
h_t = o_t⊙tanh(C_t)                         — hidden state (filtered read)
```

**GRU (Ch. 8-9):**
```
z_t = σ(W_xz·x_t + W_hz·h_{t-1} + b_z)              — update gate
r_t = σ(W_xr·x_t + W_hr·h_{t-1} + b_r)              — reset gate
h̃_t = tanh(W_xh·x_t + W_hh·(r_t⊙h_{t-1}) + b_h)     — candidate
h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t                      — hidden state (coupled blend)
```

**Bidirectional (Ch. 10):** `h_t^bi = [h_t^fwd ; h_t^bwd]` — two independent networks, concatenated.

**Attention (Ch. 13):**
```
score(s_{t-1}, h_i) = vᵀ·tanh(W_s·s_{t-1} + W_h·h_i)   — additive/Bahdanau
α_{t,i} = softmax_i(score(s_{t-1}, h_i))
c_t = Σᵢ α_{t,i}·h_i
```

**BPTT gradient chain (Ch. 3-4, 7):**
```
Vanilla RNN: ∂h_t/∂h_{t-k} ≈ Πᵢ W_hhᵀ·diag(1-h_i²)          — FIXED matrix, repeated
LSTM:        ∂C_t/∂C_{t-k} ≈ Πᵢ f_{t-i}                       — LEARNED scalar, repeated (elementwise, no matrix)
GRU:         ∂h_t/∂h_{t-k} ≈ Πᵢ (1-z_{t-i})                   — same idea, one gate
```

## 18.2 Key Numbers to Have Ready

| Fact | Number | Source |
|---|---|---|
| Vanilla RNN gradient retained after 20 steps (ρ≈0.51) | ~0.00014% | Ch. 4.3 |
| Same, exploding regime (ρ≈1.47) | ~2,200× growth | Ch. 4.4 |
| LSTM gradient retained after 20 steps, forget gate ≈0.95 | ~35.8% | Ch. 7.5 |
| LSTM vs. vanilla RNN gradient retention ratio at k=20 | ~250,000× | Ch. 7.5 |
| LSTM param count vs. vanilla RNN | ~4× | Ch. 5.7 |
| GRU param count vs. LSTM | ~25% fewer (3 gates vs. 4) | Ch. 8.4, 9.7 |
| GRU vs. LSTM FLOPs/token at d_h=256 | ~0.79M vs. ~1.05M | Ch. 15.3 |
| Typical truncated-BPTT window | k≈20-35 | Ch. 14.2 |
| Typical stacked-RNN depth | 2-4 layers | Ch. 11.3 |
| RNN inference memory per token (streaming) | O(d_h), constant | Ch. 15.1 |
| Transformer decoder inference memory (KV-cache) | O(T·d), grows with length | Ch. 15.1 |

## 18.3 Decision Framework

| Question | Answer |
|---|---|
| Vanilla RNN vs. LSTM/GRU? | Vanilla RNN essentially never used in practice beyond teaching — always start with LSTM/GRU baseline |
| LSTM vs. GRU? | Default GRU for latency/size-constrained (on-device); LSTM if decoupled forget/input control matters empirically; **always validate on your task** — no universal winner |
| Unidirectional vs. bidirectional? | Bidirectional ONLY if full input available upfront (no streaming/generation constraint) — else unidirectional or windowed/chunked-bidirectional compromise |
| Single-layer vs. stacked? | Start single-layer; stack 2-4 layers if capacity-limited (underfitting), not for gradient-flow issues (that's LSTM/GRU's job) |
| Fixed context vs. attention? | Attention essentially always better once you're past a trivial seq2seq baseline — removes the fixed-capacity bottleneck |
| Full BPTT vs. truncated? | Truncated for any long sequence in practice — tractability and stability, at the cost of a hard dependency-length ceiling |

## 18.4 Full Comprehension-Check Answer Key (Chapters 1-14)

**Ch.1** — (1) Fixed input size + no parameter sharing across positions + no memory mechanism. (2) `x_t` and `h_{t-1}`. (3) `W_hh`. (4) False — the hidden state is a fixed-size vector, so it must lossily compress unboundedly long history; capacity is bounded even though updates happen every step.

**Ch.2** — (1) Self-verify against §2.4-2.5. (2) `W_hh` (the only matrix connecting `h_{t-1}` to the current computation). (3) `y_T` (the final timestep's output). (4) `h_t` depends sequentially on `h_{t-1}`, but different *batch examples* are fully independent of each other, so the batch dimension parallelizes freely.

**Ch.3** — (1) `h_0 = [0,0]`, and `dL/dW_hh` from t=1 is an outer product with `h_0`, which is zero. (2) `W_hhᵀ` (multiplied at every step, alongside the tanh-derivative diagonal). (3) Full BPTT backprops through the entire sequence; truncated BPTT caps the backward window at `k` steps for tractability/stability, introducing a hard ceiling on learnable dependency length. (4) Exploding — eigenvalue magnitudes consistently `>1` cause geometric growth.

**Ch.4** — (1) For `W_hh''=[[0.9,0.1],[-0.2,0.7]]`: trace=1.6, det=0.9(0.7)-0.1(-0.2)=0.65, discriminant=1.6²-4(0.65)=-0.04 (complex), `|λ|=√0.65≈0.806 <1` → closer to vanishing. (2) An unrolled T-step RNN is structurally a T-layer feedforward network with tied weights — same repeated-Jacobian-multiplication gradient issue, just across time instead of depth. (3) Clipping only bounds gradient *magnitude* from above (fixes exploding); it does nothing to reintroduce vanished (too-small) gradients, and LSTM/GRU's `h_t`-path (via `tanh`) and early-training dynamics can still explode. (4) Spectral radius of `W_hh`, and tanh-saturation (`1-h²→0` near `|h|≈1`) — the latter shrinks gradients regardless of how well-conditioned `W_hh` is.

**Ch.5** — (1) See §5.3-5.5 equation block. (2) `f_t` is a *learned, content-dependent* output of the network (via gradient descent), not a fixed weight — the network can push it toward 1 specifically for units/timesteps where long-range memory matters. (3) ~4× (four independent gate weight-sets vs. vanilla RNN's one). (4) `h_t` would just equal `tanh(C_t)` always — the model would lose the ability to selectively hide part of long-term memory from the immediate output.

**Ch.6** — (1) Self-verify against §6.5. (2) `f_3` dropped to 0.3164 (actively erasing prior memory) while `C̃_3=-0.7745` was strongly negative and let through at `i_3=0.3915` — together flipping `C_t`'s sign. (3) Gates operate elementwise/independently per hidden unit — a `d_h`-dim LSTM is `d_h` independent copies of this exact scalar computation, with mixing happening only in the weight-matrix matmuls producing pre-activations. (4) The model has learned to treat that unit as stable long-term memory — writing almost nothing new, forgetting almost nothing old.

**Ch.7** — (1) `C_t = f_t⊙C_{t-1} + i_t⊙C̃_t` is *affine* in `C_{t-1}` (like `y=mx+b`); the derivative of an affine function w.r.t. its variable is just its coefficient, here `f_t`. (2) Similar magnitude in this specific untrained example (~0.13 both) because these particular forget gates aren't saturated near 1; a *trained* model can push `f_t→1` for units needing long retention, which vanilla RNN's fixed `W_hh` can never do regardless of training. (3) Biases the network toward "remember by default" from initialization, avoiding erasing the cell state before anything worth keeping has been learned. (4) Not quite — the precise statement is "LSTM makes long-range gradient preservation *learnable/achievable*, not automatically guaranteed."

**Ch.8** — (1) See §8.2 equation block. (2) `(1-z_t)` and `z_t` are forced to sum to 1, so GRU cannot independently set "how much to keep" and "how much to add" the way LSTM's separate `f_t, i_t` can — a real, nameable expressiveness reduction. (3) Reset gate controls how much past hidden state feeds into *computing the candidate*; update gate controls the *final blend* of old state vs. that candidate. (4) ~25% fewer parameters/FLOPs — directly reduces on-device latency, memory bandwidth, and battery cost.

**Ch.9** — (1) Self-verify against §9.3. (2) GRU has no output-gate filtering, so its `h_t` is the raw accumulator (like LSTM's `C_t`), not a filtered read (like LSTM's `h_t`) — with matching weights and `h_0=0`, they coincide exactly at t=1 since the reset gate has nothing to act on yet. (3) The reset gate (`r_2`) — it scales `h_1`'s contribution inside the candidate computation; this is distinct from the update gate's job of blending old state vs. the (already-computed) candidate. (4) A task needing genuinely decoupled forget/input control, or simply empirical validation favoring LSTM on your specific data/metric.

**Ch.10** — (1) Self-verify against §10.3 Step 2. (2) Generation produces tokens sequentially; future output tokens don't exist yet at generation time, so there's no "future" for a backward pass to consume. (3) Encoder is typically bidirectional (full input available, no streaming constraint); decoder must remain unidirectional/autoregressive (future outputs don't exist yet). (4) A windowed/chunked bidirectional model — buffer a small, bounded look-ahead window rather than requiring the entire sequence.

**Ch.11** — (1) The previous layer's hidden-state output at that same timestep, `h_t^(l-1)`. (2) Time (within each layer, as in Ch.3-4) and layers (across the stack) — gradient must flow cleanly through both axes for effective training. (3) Standard per-step-resampled dropout injects a different random mask at every timestep, corrupting the recurrent memory pathway with incoherent noise; variational dropout uses one fixed mask across all timesteps within a layer, preserving a consistent regularization signal along the recurrence. (4) False — stacking addresses representational capacity/hierarchy, not gradient decay over time; that's specifically what LSTM/GRU gating (Ch.5-9) addresses.

**Ch.12** — (1) Future output tokens don't exist yet during generation — nothing for a backward pass to see. (2) Teacher forcing feeds ground-truth previous tokens during training; inference feeds the model's own (possibly wrong) previous predictions — this train/inference mismatch is exposure bias, where early errors can compound. (3) The fixed-size context vector must compress the entire source sequence regardless of length; Chapter 13's attention mechanism directly addresses it. (4) Greedy decoding picks the single highest-probability token at each step; beam search keeps the top-k partial sequences at each step for better global sequence quality, at k× the compute.

**Ch.13** — (1) Self-verify against §13.3 Step C. (2) `c_t` is now a fresh, differentiable weighted combination of *all* encoder states at every decoding step, so information doesn't have to be pre-compressed into one fixed-size vector regardless of source length. (3) The underlying recurrent encoder/decoder itself — Transformers compute attention directly over all position-pairs in parallel, with no recurrent hidden-state pathway at all. (4) Additive (Bahdanau: `vᵀtanh(W_s·s+W_h·h)`) vs. multiplicative (Luong: dot-product-style); the multiplicative form scales more directly into Transformer's `QKᵀ`.

**Ch.14** — (1) Trade-off: tractable, numerically stable training in exchange for a hard ceiling on learnable dependency length (`≤k` steps); a bigger `k` only pushes the ceiling higher, it doesn't remove the fundamental trade-off (and reintroduces the Ch.4 instability risk as `k` grows). (2) `g=[6,8]`, `‖g‖=10`, `threshold=3` → `scale=0.3` → `g_clipped=[1.8, 2.4]`, `‖g_clipped‖=3.0`. (3) Masking must happen *before* softmax (via a large negative value) so the padded position's contribution vanishes during exponentiation; zeroing weights *after* softmax would leave the distribution un-renormalized, incorrectly stealing probability mass from real positions. (4) Bucketing groups sequences of similar length into the same batch, minimizing wasted compute on padding — this matters most when length distributions are highly skewed (heavy padding waste otherwise, as shown numerically in Ch.15.2).

## 18.5 The Ten Sentences to Have Instantly Ready

1. Vanilla RNNs fail because gradient flow through time is a fixed, repeated matrix multiplication that geometrically vanishes or explodes.
2. LSTM fixes this by making the multiplicative gradient decay a *learned, content-dependent* gate value instead of a fixed matrix.
3. GRU is LSTM simplified: one merged update gate instead of separate forget/input, no separate cell state, ~25% fewer parameters, at the cost of coupled keep/add fractions.
4. LSTM doesn't guarantee good gradient flow — it makes it *achievable*, contingent on what the forget gates learn.
5. Bidirectionality and gradient-flow architecture are orthogonal axes — solving one doesn't solve the other.
6. Bidirectional models cannot stream or generate autoregressively — the backward pass needs the full sequence upfront.
7. Stacking adds a second depth axis (layers × time) and addresses capacity/hierarchy, not gradient decay.
8. The seq2seq fixed-context-vector bottleneck is fixed by attention: a fresh, full-access weighted combination of encoder states at every decoding step.
9. RNN-family models retain a genuine constant-memory inference advantage over Transformer decoders for long-context streaming — directly relevant on-device.
10. Training-serving skew for sequence models has specific failure modes beyond generic ML skew: history-availability mismatch, bidirectionality mismatch, and chunking/tokenization consistency.

---

**Curriculum complete.** You've now covered: motivation (Ch.1) → vanilla RNN mechanics and failure modes (Ch.2-4) → LSTM (Ch.5-7) → GRU (Ch.8-9) → bidirectionality (Ch.10) → depth (Ch.11) → seq2seq and attention (Ch.12-13) → training mechanics and production (Ch.14-15) → interview Q&A, system design, and coding (Ch.16-17) → this consolidated reference (Ch.18).

Good luck with the Apple interviews — you've hand-derived every core equation in this space at least once, which is well beyond what most candidates bring to the table.
