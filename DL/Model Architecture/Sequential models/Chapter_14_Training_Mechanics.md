# Chapter 14: Training Mechanics — Truncated BPTT, Gradient Clipping, Padding & Masking

**Curriculum:** End-to-End Sequential Modeling (RNN → LSTM → GRU → BiRNN)
**Prerequisite:** Chapters 3-4 (BPTT, vanishing/exploding), 12 (teacher forcing intro)

---

## 14.1 Teacher Forcing — Quick Recap + One Addition

Covered in Ch. 12.4: feed ground-truth previous tokens during training, model's own predictions at inference. **One addition:** teacher forcing ratio can be **annealed** — e.g., start training at 100% teacher forcing, gradually lower toward 0% (feeding the model's own predictions increasingly often) as training progresses. This is the essence of **scheduled sampling**, directly narrowing the train/inference distribution mismatch (exposure bias) mentioned in Ch. 12.4.

## 14.2 Truncated BPTT — The Practical Recipe

Full BPTT (Ch. 3) backprops through the *entire* sequence — for a 10,000-timestep sequence (e.g., a full document, or a long audio stream), this is both computationally prohibitive and numerically dangerous (recall Ch. 4's exploding-gradient risk compounds with sequence length).

**Truncated BPTT recipe:**
1. Split the sequence into chunks of length `k` (e.g., `k=35`, a common historical default from language modeling literature).
2. Run the forward pass chunk-by-chunk, **carrying the hidden state forward** between chunks (so the model still has continuity of memory).
3. **Backpropagate only within each chunk** — when starting the next chunk's backward pass, treat the incoming hidden state as a constant (`.detach()` in PyTorch terms), stopping gradient flow into the previous chunk.

**This introduces a systematic bias**: the model can never learn a dependency spanning more than `k` timesteps, *from the gradient's perspective* — even though the forward-pass hidden state does carry information further back. This is a deliberate, explicit trade-off: tractability and stability, purchased with a hard ceiling on learnable dependency length.

**Interview-relevant nuance:** this ceiling is exactly the kind of limitation that motivated architectures avoiding recurrence-through-time gradient flow altogether (Transformers use attention with direct, unbounded-in-principle access to all positions, no truncation needed) — worth naming as a genuine, still-relevant reason for that architectural shift, beyond just training-speed parallelism.

## 14.3 Gradient Clipping — the Standard Exploding-Gradient Fix

Recall Ch. 4: exploding gradients cause loss to spike toward `NaN`. **Gradient norm clipping** rescales the entire gradient vector if its norm exceeds a threshold, preserving direction but capping magnitude:

```
if ||g|| > threshold:
    g ← g · (threshold / ||g||)
```

**Numerical example:** suppose a gradient vector (flattened across all parameters) is `g = [3, 4]` (a toy 2-dim stand-in), so `||g|| = √(3²+4²) = √25 = 5`. With `threshold = 2`:
```
g_clipped = [3,4] · (2/5) = [1.2, 1.6]
||g_clipped|| = √(1.2²+1.6²) = √(1.44+2.56) = √4 = 2.0   ✓ (exactly at threshold, direction unchanged)
```

**Typical thresholds** in practice: values like 1, 5, or 10 are common starting points, tuned per task — this is a real hyperparameter, not a fixed universal constant. **Important distinction:** clip by *global norm* (across all parameters, as above) rather than clipping each parameter/gradient element independently — clipping element-wise would distort the gradient's direction, not just its magnitude, which can actively hurt optimization.

## 14.4 Padding & Masking for Variable-Length Sequences (Batching Requirement)

RNN/LSTM/GRU cells process one sequence at a time conceptually, but training in **batches** requires all sequences in a batch to have the same length (for the underlying tensor operations to work). Real sequences vary in length — the standard fix:

1. **Pad** shorter sequences with a special `<PAD>` token up to the batch's max length.
2. **Mask** the loss computation so `<PAD>` positions don't contribute to the gradient (you don't want the model penalized — or rewarded — for "predicting" padding).
3. If using attention (Ch. 13), **mask the attention scores** at padded positions too — set them to a very large negative number (e.g., `-1e9`) *before* the softmax, so they receive essentially zero attention weight.

**Numerical masking example:** suppose raw attention scores for a length-3 real sequence plus 1 padded position are `[2.0, 1.0, 0.5, ???]`. Set the padded position's score to `-1e9`:
```
scores = [2.0, 1.0, 0.5, -1e9]
softmax: e^2.0=7.389, e^1.0=2.718, e^0.5=1.649, e^-1e9 ≈ 0 (underflows to exactly 0 in floating point)
sum ≈ 7.389+2.718+1.649+0 = 11.756
weights ≈ [0.6285, 0.2312, 0.1403, 0.0000]
```
The padded position receives (effectively) exactly zero attention weight — it cannot influence the context vector at all, which is the correct behavior.

## 14.5 Batching Efficiency: Bucketing

Padding every sequence in a batch to the length of the *longest* sequence wastes compute on `<PAD>` positions that contribute nothing useful. **Bucketing** (a.k.a. length-based batching) groups sequences of similar length into the same batch, minimizing wasted padding — a standard production data-loading optimization, especially relevant when sequence-length distributions are highly skewed (e.g., search queries: mostly short, occasional very long ones).

## 14.6 Interview Talking Points (L5 Signal)

- "Truncated BPTT is a deliberate bias-variance-style trade-off — you're explicitly capping the *learnable* dependency length in exchange for tractable, stable training. It's worth stating the trade-off explicitly rather than presenting truncation as a free optimization."
- "Gradient clipping is clipped by *global norm*, not per-element — clipping per-element would distort gradient direction, effectively changing what the optimizer is optimizing towards, not just how far it steps."
- "Masking has to be applied consistently everywhere padding could leak influence — the loss, and separately, attention scores if attention is used. Forgetting to mask attention (even if the loss is correctly masked) is a subtle, common production bug — the model would learn to attend to `<PAD>` tokens as if they were real content."

## 14.7 Sample Interview Q&A

**Q: You notice your model's loss is fine for most batches but occasionally spikes to NaN. What's your first diagnostic step, and likely fix?**
A: First, log gradient norms per batch to confirm this is an exploding-gradient event (vs., say, a data issue or numerical instability elsewhere, like a bad log(0) in the loss). If gradient norms spike correspondingly, gradient clipping (by global norm) is the standard first fix; if it persists, also check learning rate and initialization, and consider whether a specific rare, unusually long sequence in that batch is contributing an outsized gradient.

**Q: Why not just backprop through the full sequence and skip truncation, given modern hardware?**
A: Even with sufficient memory, full BPTT accumulates the same multiplicative gradient risk from Chapter 4 across the entire sequence length, so numerical instability (exploding or vanishing) generally gets worse, not just more expensive, as sequence length grows unbounded. Truncation is as much a stability tool as a compute-saving one.

**Q: If you forget to mask padded positions in the loss function, what specifically goes wrong?**
A: The model receives gradient signal trying to make it accurately predict the arbitrary `<PAD>` token at padded positions, which is meaningless noise — at best this wastes some training signal/capacity, at worst (especially with many padded positions relative to real content) it can meaningfully bias what the model learns, particularly if padding is heavily concentrated at certain positions (e.g., always at the end) that the model then over-indexes on.

## 14.8 Comprehension Check

1. What's the essential trade-off truncated BPTT makes, and why can't you avoid it just by using a bigger `k`?
2. Redo the gradient-clipping numerical example with `g = [6, 8]` and `threshold = 3` — what's `g_clipped`, and what's its norm?
3. Why must attention scores be masked with a large *negative* value before softmax, rather than just zeroing out the attention weight afterward?
4. What production problem does bucketing/length-based batching solve, and why does it matter more when sequence lengths are highly skewed?

---
**Next:** Chapter 15 — Production considerations: batching at scale, serving latency, training-serving skew (directly connects to your recommendation-systems curriculum).
