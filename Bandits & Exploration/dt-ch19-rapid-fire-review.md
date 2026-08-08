# Chapter 19 — Rapid-Fire Review & L5-vs-L6 Differentiators

*(Final chapter — condensed review of Chapters 1-18.)*

---

## 19.1 Master comparison table: parallelism strategies

| Strategy | What's split | Fixes which wall | Communication frequency | Needs fast interconnect? |
|---|---|---|---|---|
| Data parallelism | Batch of data | Speed | Once per step (gradient all-reduce) | No, tolerates cross-node |
| Tensor parallelism | Inside one layer's math | Memory | Multiple times per forward/backward pass | Yes, must stay in-node (NVLink) |
| Pipeline parallelism | Layers (between them) | Memory | Once per stage boundary, per micro-batch | Moderate, can span nodes |
| ZeRO Stage 1 | Optimizer states | Memory | Minimal extra beyond ordinary all-reduce | No |
| ZeRO Stage 2 | + Gradients | Memory | Slightly more | No |
| ZeRO Stage 3 / FSDP | + Parameters | Memory | Substantial — all-gather per layer | Somewhat, benefits from decent bandwidth |

---

## 19.2 Master comparison table: precision formats

| Format | Bits | Exponent bits (range) | Mantissa bits (precision) | Main risk |
|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | None — reference precision |
| fp16 | 16 | 5 | 10 | Underflow (needs loss scaling) |
| bf16 | 16 | 8 | 7 | Lower precision, but same range as fp32 |

---

## 19.3 Master formulas worth having cold

- **Memory rule of thumb**: parameters + gradients + Adam optimizer states ≈ 4× raw parameter memory (before activations), in fp32.
- **Effective batch size**: micro-batch size × number of GPUs × gradient accumulation steps.
- **Ring all-reduce per-device cost**: $\frac{2(N-1)}{N} \times$ full gradient size → converges to ~2× as $N$ grows.
- **Linear Scaling Rule**: multiply learning rate by the same factor you multiply batch size by; always paired with warmup.
- **Failure probability at scale**: $1-(1-p)^N$ for $N$ devices each with per-device failure probability $p$ — grows fast even for small $p$.

---

## 19.4 The 12 most likely follow-up questions, with short model answers

**1. "Why not just train in fp32 everywhere?"**
→ Memory and speed cost. bf16 halves memory and runs faster on tensor cores, with minimal precision downside for most operations.

**2. "Why does tensor parallelism need to stay within one server?"**
→ It communicates multiple times per single forward/backward pass — far more chatty than data or pipeline parallelism — so it needs NVLink-level bandwidth, not cross-node networking.

**3. "What's the difference between ZeRO Stage 3 and tensor parallelism — aren't both 'splitting parameters'?"**
→ ZeRO-3 temporarily all-gathers a layer's full parameters right before computing, then discards them. Tensor parallelism computes jointly, permanently, on its own slice — never reconstructing the full layer.

**4. "Why does the pipeline bubble happen, and how do you fix it?"**
→ Naive pipelining processes one batch at a time, leaving most devices idle while data is elsewhere in the pipeline. Micro-batching (GPipe) keeps the pipeline full; 1F1B/PipeDream scheduling reduces the resulting activation-memory cost further.

**5. "Why can't you just keep adding data-parallel GPUs forever?"**
→ Gradient noise reduction from larger batches has diminishing returns — beyond a point, you're paying more compute per step without a proportionally better step.

**6. "What actually needs to be in a checkpoint?"**
→ Parameters, optimizer state, data-loader position, RNG state. Missing any one silently produces a subtly different resumed run, not an obvious crash.

**7. "How do you handle GPU failures at scale?"**
→ Expect them — at 1,000 GPUs even a low per-device failure rate gives a high daily probability of some failure. Frequent (ideally async) checkpointing plus elastic training to avoid full-job restarts.

**8. "What's a straggler, and why is it different from a failure?"**
→ A device that's still running but abnormally slow. In synchronous data parallelism, one straggler bottlenecks the entire group's all-reduce, without ever registering as a technical failure.

**9. "Walk me through ring all-reduce."**
→ Two phases, each $N-1$ steps: reduce-scatter (chunks circulate and accumulate until each device holds one fully-summed chunk) then all-gather (completed chunks circulate until every device has all of them). Per-device cost converges to ~2× gradient size regardless of $N$.

**10. "Why does mixed precision speed things up, not just save memory?"**
→ Tensor cores are dedicated hardware built to run fp16/bf16 matrix multiplications faster than fp32 ones — a real hardware speed benefit, not just a memory-size benefit.

**11. "How would you lay out parallelism for N GPUs?"**
→ Tensor-parallel degree from hardware (GPUs per fast-interconnect node), pipeline-parallel degree from per-stage memory needs, data-parallel degree from whatever's left over.

**12. "Why pair the Linear Scaling Rule with warmup?"**
→ The scaled-up learning rate is only safe once training has moved past the unstable early phase; applying it from step 1 risks divergence.

---

## 19.5 Full traps checklist

- Optimizer states, not just parameters, dominate memory — Adam alone adds 2× parameter size, usually kept in fp32 even during bf16/fp16 training.
- Column-parallel splitting needs concatenation; row-parallel splitting needs an all-reduce — don't flip these.
- Parameter server bandwidth grows linearly with worker count; naive all-to-all grows quadratically; ring all-reduce's per-device cost converges to a constant.
- fp16 sacrifices range for precision (risks underflow); bf16 sacrifices precision for range (avoids most underflow).
- Data parallelism alone never fixes a "model doesn't fit" problem — every device still needs the whole model.
- The pipeline bubble never fully disappears (ramp-up/ramp-down are fixed costs) — more micro-batches only shrinks its proportion.
- ZeRO Stage 3 ≠ tensor parallelism — temporary reconstruction vs. permanent joint computation.
- More micro-batches or more gradient accumulation steps both cost real wall-clock time or memory — neither is a free lunch.
- Activation checkpointing is a good deal only when memory-bound, not when compute-bound.
- Offloading is a last resort after ZeRO sharding and activation checkpointing, not a first-line optimization.
- A complete checkpoint needs parameters + optimizer state + data-loader position + RNG state.
- Bigger batch size without a corresponding learning rate increase (and warmup) often hurts training quality.

---

## 19.6 Closing note

This closes the 19-chapter Distributed Training syllabus — from the two walls (memory, speed) in Chapter 1, through data/tensor/pipeline parallelism and their combination into 3D layouts, ZeRO/FSDP and activation checkpointing, GPU/TPU hardware realities, and the operational material (checkpointing, fault tolerance, scaling laws) that tends to separate textbook familiarity from real deployment judgment.

The single habit worth carrying forward: before proposing any distributed training design, compute the actual memory budget first. Nearly every mock and worked example in this course started there — it's the fastest way to make an interview answer concrete instead of buzzword-shaped.
