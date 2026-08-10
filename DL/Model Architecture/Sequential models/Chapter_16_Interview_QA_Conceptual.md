# Chapter 16: Apple/Google Interview Q&A — Conceptual Rapid Review

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Format:** Rapid-fire Q&A pulling together Chapters 1-15. Cover the answer, self-test, then check.

---

### Foundations (Ch. 1-4)

**Q1. Why can't a plain feedforward ANN handle variable-length sequential data well?**
A: Fixed input size forces a hand-picked window; no parameter sharing across positions means relearning the same pattern at every position separately; no explicit memory mechanism to carry information across arbitrary distances.

**Q2. What's the fundamental capacity limitation of a vanilla RNN's hidden state?**
A: It's a fixed-dimensional vector that must lossily compress arbitrarily long history — a hard capacity bottleneck, independent of the vanishing-gradient issue (though the two compound in practice).

**Q3. What causes vanishing/exploding gradients in vanilla RNNs, precisely?**
A: BPTT's gradient w.r.t. an early hidden state is a product of `k` Jacobians (`W_hhᵀ·diag(1-h²)`) — if the dominant eigenvalue of this product is consistently `<1`, gradients vanish geometrically with `k`; if `>1`, they explode.

**Q4. Does using ReLU instead of tanh solve vanishing gradients in an RNN?**
A: Partially — removes the tanh-saturation shrinkage factor, but the `W_hhᵀ` multiplicative chain remains, so vanishing/exploding still depends on `W_hh`'s spectral radius; ReLU RNNs also risk exploding more without careful (e.g., identity) initialization.

### LSTM (Ch. 5-7)

**Q5. What's the single architectural idea that lets LSTM address vanishing gradients?**
A: An additive, elementwise-gated cell-state recurrence (`C_t = f_t⊙C_{t-1} + i_t⊙C̃_t`) whose local derivative `∂C_t/∂C_{t-1} = f_t` is a learned, content-dependent scalar per unit, rather than a fixed shared matrix applied uniformly every step.

**Q6. Does LSTM guarantee good gradient flow over long sequences?**
A: No — only makes it *achievable*. If forget gates aren't trained/initialized to be near 1 where needed, decay still occurs. This is why forget-gate bias is often initialized positively (e.g., +1 or +2) to bias toward "remember by default" early in training.

**Q7. What are the three gates in an LSTM, and their individual jobs?**
A: Forget gate — how much old cell state to keep; input gate — how much new candidate info to write in; output gate — how much of the (updated) cell state to expose as the hidden state.

**Q8. Is the LSTM cell state bounded like the hidden state?**
A: No — `C_t` has no squashing nonlinearity in its own update; only `tanh(C_t)` is used downstream when producing `h_t`. This lets `C_t` act as an unbounded accumulator.

### GRU (Ch. 8-9)

**Q9. What two simplifications does GRU make relative to LSTM?**
A: (1) Merges forget+input into one update gate (`z_t`/`1-z_t`, forced complementary — a real expressiveness reduction vs. LSTM's independent gates), and (2) drops the separate cell state, using only `h_t`, with no output gate filtering.

**Q10. Is GRU always better than LSTM because of fewer parameters?**
A: No — GRU's coupled gating is a genuine capability reduction; empirical performance is task-dependent with no universal winner (e.g., Chung et al. 2014). The right framing is: GRU trades some expressiveness for ~25% fewer parameters and faster training/inference — validate empirically per task, and default to GRU when latency/size constraints are binding.

**Q11. Structurally, does GRU's `h_t` more closely resemble LSTM's `h_t` or `C_t`?**
A: LSTM's `C_t` — since GRU has no output-gate filtering step, its `h_t` is the direct, unfiltered accumulator, unlike LSTM's `h_t` which is a gated/filtered *read* of `C_t`.

### Bidirectional (Ch. 10)

**Q12. Why can't you use a bidirectional model for real-time streaming or autoregressive generation?**
A: The backward pass requires the entire sequence upfront (it starts at the last token) — fundamentally incompatible with generating/receiving tokens one at a time before the sequence is complete.

**Q13. In an encoder-decoder, why is the encoder often bidirectional while the decoder never is?**
A: The encoder sees the complete, fixed input — no streaming constraint. The decoder generates output tokens sequentially; future output tokens don't exist yet during generation, so bidirectionality is structurally impossible there.

**Q14. Does bidirectionality address vanishing gradients?**
A: No — orthogonal concerns. Bidirectionality is about *what context is available* (past vs. past+future); vanishing gradients are about *gradient flow quality* within a single directional pass. A bidirectional vanilla RNN can still vanish badly in both directions.

### Stacking, Seq2Seq, Attention (Ch. 11-13)

**Q15. What's the practical depth limit for stacked RNNs, and why?**
A: Typically 2-4 layers — stacking compounds gradient-flow difficulty along a second axis (layers) on top of the existing time axis, so optimization gets harder faster than in CNNs/Transformers, which tolerate much greater depth (often with residual connections specifically enabling that depth).

**Q16. What's the core bottleneck in a vanilla (non-attention) seq2seq encoder-decoder?**
A: The single fixed-size context vector must summarize the entire source sequence regardless of length — quality degrades as source length grows, since more information must be compressed into the same fixed capacity.

**Q17. How does attention fix this bottleneck, mechanically?**
A: Instead of one fixed `c`, compute a fresh, learned weighted combination (`c_t = Σᵢ α_{t,i}·h_i`) of *all* encoder hidden states at every decoding step, with weights (`α`) determined by a learned relevance/alignment score between the current decoder state and each encoder position.

**Q18. What's exposure bias, and how is it typically mitigated?**
A: A train/inference mismatch: training uses teacher-forced ground-truth history; inference uses the model's own (possibly imperfect) generated history, letting errors compound. Mitigated via scheduled sampling (gradually mixing in the model's own predictions during training) or via decoding strategies like beam search that better match production usage.

### Training Mechanics & Production (Ch. 14-15)

**Q19. Why clip gradients by global norm rather than element-wise?**
A: Element-wise clipping distorts the gradient's direction (not just magnitude), effectively changing the optimization target; global-norm clipping preserves direction while capping magnitude.

**Q20. What's the systematic cost of truncated BPTT?**
A: A hard ceiling on the maximum learnable dependency length (bounded by the truncation window `k`), traded deliberately for tractable, numerically stable training on long sequences.

**Q21. Why must attention scores (not just the loss) be masked for padded positions?**
A: Unmasked padded positions would receive nonzero attention weight and influence the context vector with meaningless content — the model would effectively learn to "attend to" padding as if it were real signal.

**Q22. What structural inference-time advantage do RNN-family models retain over Transformer decoders, even today?**
A: Constant-size (`O(d_h)`) memory per generated token, vs. Transformer's linearly-growing KV-cache (`O(T·d)`) — genuinely relevant for long-context, memory-constrained, on-device serving, and part of the motivation behind recent State Space Model (Mamba/S4-style) research.

---

## 16.1 Self-Test Protocol

Cover the "A:" lines and answer each question aloud, in your own words, in under 30 seconds — that's roughly the real-time budget in a live interview. If you find yourself needing to recite an equation to answer a conceptual question, that's a signal to revisit the corresponding chapter's *intuition* sections (5.1-5.3, 8.1-8.3, etc.), not just the math.

## 16.2 Common Failure Patterns to Watch For (Self-Diagnosis)

- Saying "LSTM solves vanishing gradients" without the nuance that it makes it *learnable*, not guaranteed (Q6) — this is the single most common oversimplification interviewers probe for.
- Treating bidirectionality and gradient-flow architecture (LSTM/GRU) as the same axis of improvement — they're orthogonal (Q14).
- Recommending BiLSTM for a streaming/generative system design question without immediately flagging the incompatibility (Q12) — a very common trap question.
- Defaulting to "GRU is just better/faster" without naming the specific expressiveness trade-off (coupled gating) it makes (Q10).

---
**Next:** Chapter 17 — Apple/Google System Design & Coding Q&A: longer-form questions (design a streaming transcription system, implement an LSTM cell from scratch, etc.) with full worked solutions.
