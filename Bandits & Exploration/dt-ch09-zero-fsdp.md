# Chapter 9 — ZeRO and Fully Sharded Data Parallelism (FSDP)

*(Plain language first, building the memory savings one stage at a time with real numbers.)*

---

## 9.1 Reopening Chapter 1's memory table — the waste ZeRO targets

Recall Chapter 1's memory breakdown for a 7B model in fp32 with Adam: parameters (28 GB) + gradients (28 GB) + optimizer states (56 GB) = 112 GB total. Now recall Chapter 3's plain data parallelism: **every single GPU keeps a full, identical copy of all of this** — if you have 8 GPUs doing plain data parallelism, you have **8 separate, fully redundant copies** of that entire 112 GB sitting in memory across your cluster, even though, in principle, only *one* full copy's worth of information actually exists (they're all identical!). **ZeRO's entire idea is to eliminate this redundancy** — instead of every GPU storing everything, **split (shard) the parameters/gradients/optimizer states across the GPUs**, so collectively the whole group stores exactly one full copy, not $N$ redundant copies.

---

## 9.2 ZeRO Stage 1: shard the optimizer states

**The idea, in plain words**: optimizer states (56 GB in our example — the single biggest chunk!) are only actually *needed*, in full, at the moment the optimizer applies its update step. So instead of every GPU keeping its own full 56 GB copy, **split the optimizer states into $N$ pieces (one piece per GPU)** — each GPU only ever stores $1/N$ of the total optimizer state.

**Worked example with $N=8$ GPUs**: instead of every GPU storing the full 56 GB of optimizer state, each GPU stores only $56/8 = 7$ GB. Parameters and gradients are still fully replicated on every GPU at this stage (that's what "Stage 1" means — only the optimizer states are sharded so far). **New per-GPU memory total: $28 + 28 + 7 = 63$ GB** — down from 112 GB, a substantial savings, just from sharding the single biggest piece.

---

## 9.3 ZeRO Stage 2: also shard the gradients

**The idea, in plain words**: gradients, similarly, don't need to be kept in their full, replicated form on every device for very long — Stage 2 shards the gradients the same way Stage 1 sharded the optimizer states.

**Worked example, same $N=8$**: gradients go from $28$ GB (full copy) down to $28/8 = 3.5$ GB per GPU. **New per-GPU memory total: $28 \text{ (params, still full)} + 3.5 \text{ (gradients, sharded)} + 7 \text{ (optimizer states, sharded)} = 38.5$ GB** — continuing to drop, now with two of the three big pieces sharded.

---

## 9.4 ZeRO Stage 3: also shard the parameters themselves

**The idea, in plain words**: go all the way — shard the **parameters** too, the last remaining fully-replicated piece.

**Worked example, same $N=8$**: parameters go from $28$ GB down to $28/8 = 3.5$ GB per GPU. **New per-GPU memory total: $3.5 + 3.5 + 7 = 14$ GB** — an enormous reduction from the original 112 GB (roughly 8× smaller, matching our $N=8$ GPU count almost exactly, since we've now sharded essentially everything).

---

## 9.5 The catch: Stage 3 needs the *full* parameters back, temporarily, to actually compute

Here's the crucial mechanical detail that makes ZeRO Stage 3 genuinely different from simple model parallelism (tensor/pipeline), and worth being precise about: **during the actual forward and backward pass computation, a given layer's math still needs its *full*, complete weight matrix** — you can't do a real matrix multiplication with only 1/8th of the weights sitting on your GPU and the rest missing.

**So what actually happens**: right before a given layer needs to compute, ZeRO Stage 3 does a quick **all-gather** (recall this exact operation from Chapter 4's ring all-reduce discussion — an all-gather is literally half of that same mechanism) to temporarily reconstruct that one layer's full parameters, just long enough to use them, and then **discards them again** immediately afterward, freeing the memory back up. This happens **layer by layer**, continuously, throughout the forward and backward pass — a lot of extra communication compared to Stage 1 or Stage 2, in exchange for the maximal memory savings.

**Why this is a genuinely different mechanism from tensor parallelism (Chapter 6), even though both involve splitting up parameters**: tensor parallelism computes **jointly**, with each device permanently holding and permanently computing on its own slice, communicating partial *results*. ZeRO Stage 3 computes **one full layer at a time**, temporarily reassembling the *full* parameters on every device just before use — a fundamentally different strategy (temporarily "borrow" the full picture vs. permanently work with a slice) that happens to also reduce memory, via a different mechanism entirely.

---

## 9.6 The general pattern across all 3 stages

| Stage | What's sharded | What's still fully replicated | Extra communication introduced |
|---|---|---|---|
| Stage 1 | Optimizer states | Parameters, gradients | Minimal beyond ordinary data-parallel all-reduce |
| Stage 2 | Optimizer states + gradients | Parameters only | A bit more, to reduce-scatter gradients instead of fully all-reducing them |
| Stage 3 | Optimizer states + gradients + parameters | Nothing | Substantial — an all-gather per layer, every forward and backward pass |

**The consistent, simple pattern worth internalizing**: each stage trades **more communication** for **less memory**, one more piece at a time — exactly the same kind of tradeoff spirit as Chapter 7's pipeline bubble-vs-memory tension, just applied to a different mechanism.

---

## 9.7 FSDP — PyTorch's version of this same idea

**FSDP (Fully Sharded Data Parallelism)** is PyTorch's own built-in implementation of essentially the ZeRO Stage 3 idea — sharding parameters, gradients, and optimizer states across data-parallel workers, with the same "temporarily all-gather a layer's full parameters right before using them, then discard" mechanism described in Section 9.5. **Practically, if someone says "we used FSDP to train this model," they mean something very close to "we used ZeRO Stage 3"** — same underlying idea, different name/implementation, and it's worth knowing both terms since papers and job descriptions use them somewhat interchangeably depending on which specific framework (DeepSpeed vs. native PyTorch) was used.

---

## 9.8 Production considerations

- **ZeRO/FSDP is often the *first* thing engineers reach for when a model doesn't fit in plain data parallelism**, before reaching for tensor or pipeline parallelism (Chapters 6–7) — it requires far less code restructuring (you don't need to manually rewrite every layer's forward pass the way tensor parallelism requires), while still capturing much of the memory benefit, which is a major reason for its popularity.
- **ZeRO/FSDP and tensor/pipeline parallelism are not mutually exclusive** — real large-scale training setups often combine ZeRO-style sharding *within* the data-parallel dimension of a full 3D-parallel layout (exactly the "4th dimension" hinted at in Chapter 8, Section 8.6), getting benefits from both approaches simultaneously.
- **The extra communication introduced by Stage 3/FSDP (the frequent all-gathers) needs reasonably fast interconnects to not become a bottleneck** — similar in spirit to tensor parallelism's fast-interconnect requirement from Chapter 6, though generally somewhat more tolerant of slower connections than tensor parallelism specifically requires.

---

## 9.9 Interview traps

- **Not knowing what specifically gets sharded at each ZeRO stage.** Being able to precisely state "Stage 1: optimizer states; Stage 2: + gradients; Stage 3: + parameters" — in that specific order, cumulatively — is a very checkable, specific fact worth having exactly right.
- **Confusing ZeRO Stage 3 with tensor parallelism** because both involve "splitting parameters across devices." The key distinguishing mechanical fact (Section 9.5): ZeRO Stage 3 temporarily reconstructs full parameters via all-gather right before computing, then discards them — it doesn't compute jointly on permanent slices the way tensor parallelism does.
- **Forgetting that each stage's memory savings comes with a real communication cost increase** — presenting ZeRO Stage 3 as a strictly free, unlimited memory-reduction trick misses the genuine tradeoff involved.

---

## 9.10 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describes the general idea of sharding optimizer states/gradients/parameters across data-parallel workers, and knows FSDP is PyTorch's implementation of this idea.
- **L6 bar**:
  - Can walk through the full worked memory-reduction numbers across all three stages (like Sections 9.2–9.4), for a given model size and GPU count, live.
  - Precisely explains *why* ZeRO Stage 3 is mechanically different from tensor parallelism (temporary all-gather-then-discard vs. permanent joint computation on slices) rather than treating them as interchangeable "ways to split parameters."
  - Proactively notes that ZeRO/FSDP is often combined with, not a replacement for, tensor/pipeline parallelism in real large-scale 3D-parallel setups, connecting this chapter back to Chapter 8's overall layout picture.

---

## 9.11 Comprehension checks

1. Why does plain data parallelism waste so much memory across multiple GPUs, and what is ZeRO's core fix for this waste?
2. Walk through the memory numbers for a 7B-parameter model (fp32, Adam) across ZeRO Stages 1, 2, and 3, with $N=8$ GPUs.
3. What specifically happens, mechanically, right before a layer computes under ZeRO Stage 3 — and why is this necessary?
4. How is ZeRO Stage 3 mechanically different from tensor parallelism, even though both involve splitting parameters across devices?
5. What is FSDP, and how does it relate to ZeRO Stage 3?

---

*Next: Chapter 10 — Activation Checkpointing (Gradient Recomputation), covering the other major memory-optimization tool — trading extra compute for reduced activation memory.*
