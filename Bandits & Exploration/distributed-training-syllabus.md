# Distributed Training — Interview Mastery Syllabus
### Target: Google / Apple L5 MLE & Applied Scientist interviews
### Depth and breadth: every topic explained with "why" and "how," not just "what"
### Prerequisites assumed: basic deep learning (backprop, SGD, what a GPU roughly does) — no prior distributed-systems experience assumed

---

## How this curriculum is structured

Same format as your other curricula — each chapter (or small cluster) becomes its own self-contained markdown file, built to be simple and plain-language first, with worked/numerical grounding wherever the math would otherwise feel abstract:

1. **Intuition first** — plain-English mental model before notation
2. **How it actually works** — mechanics, with simple worked numbers where useful
3. **Why it exists** — what problem it solves, what breaks without it
4. **Production/hardware reality** — how this plays out on real GPUs/TPUs at Google/Apple scale
5. **Interview traps**
6. **L5-vs-L6 differentiating talking points**
7. **Comprehension checks**

---

## Phase 0 — Foundations (Ch. 1–2)
*Goal: understand why distributed training is necessary at all, and the vocabulary for talking about it.*

**Ch 1 — Why We Need Distributed Training**
- The two walls: models too big for one GPU's memory; datasets/compute too slow on one GPU
- GPU memory anatomy (parameters, gradients, optimizer states, activations) — where the memory actually goes, with a worked memory-budget example for a concrete model size
- FLOPs vs memory-bandwidth-bound — why more compute alone doesn't fix everything
- The interconnect hierarchy: within-GPU, NVLink between GPUs, network between nodes — and why bandwidth drops by orders of magnitude at each hop

**Ch 2 — The Parallelism Taxonomy**
- Data parallelism, model parallelism (tensor + pipeline), and how they differ in *what* gets split
- A simple mental model: split the data, split the layers, or split each layer itself
- Why real large-scale training uses combinations of all three ("3D parallelism") — previewed here, built up properly by Chapter 8

---

## Phase 1 — Data Parallelism & Communication (Ch. 3–5)
*Goal: the most common, most interview-tested form of parallelism, done right.*

**Ch 3 — Data Parallelism and Gradient Accumulation**
- Basic data-parallel training loop: same model copied across devices, different data shards, gradients combined
- Why gradients need to be combined (averaged) before the optimizer step, and what breaks if you don't
- Gradient accumulation: simulating a larger batch size than fits in memory, worked through with a simple numeric example
- Effective batch size arithmetic — a very common interview calculation

**Ch 4 — Communication Primitives: All-Reduce vs. Parameter Servers**
- Parameter server architecture: how it works, why it was the original approach, its bottleneck (central server traffic)
- All-reduce, and specifically **ring all-reduce**: step-by-step mechanics, why its communication cost doesn't scale with the number of GPUs the way a naive approach would
- Worked numerical trace of ring all-reduce on a small number of GPUs
- When parameter servers still make sense today (highly asynchronous, elastic, heterogeneous settings) vs. why all-reduce dominates modern large-scale training

**Ch 5 — Mixed Precision Training**
- fp32 vs fp16 vs bf16 — what the bits actually represent, and why range vs. precision is the core tradeoff
- Why naive fp16 training diverges: gradient underflow, and the fix (loss scaling), worked through numerically
- Why bf16 sidesteps much of this (same exponent range as fp32) and why newer large-model training often prefers it outright
- Automatic Mixed Precision (AMP) — what it actually does under the hood

---

## Phase 2 — Model Parallelism (Ch. 6–8)
*Goal: what to do when the model itself doesn't fit on one device.*

**Ch 6 — Tensor Parallelism**
- The core idea: split an individual layer's matrix multiplication across devices, not just the data
- Megatron-LM-style row/column parallelism for transformer layers, explained with a simple small-matrix worked example
- The communication cost this introduces (needing an all-reduce *within* a single forward/backward pass, not just once per step) — why this makes tensor parallelism sensitive to interconnect speed, and why it's typically kept within a single high-bandwidth node

**Ch 7 — Pipeline Parallelism**
- The core idea: split the model's layers across devices, each device owning a contiguous chunk
- The naive version's big problem: the "bubble" (idle time while waiting for activations to flow through the pipeline), explained visually/numerically
- Micro-batching as the fix: GPipe-style scheduling, worked through with a simple timeline diagram in words
- PipeDream and interleaved/1F1B scheduling — reducing the bubble further, and the memory-vs-bubble tradeoff this introduces

**Ch 8 — Combining Strategies: 3D Parallelism**
- Why real frontier-model training combines data + tensor + pipeline parallelism simultaneously
- A worked "how would you lay out 512 GPUs" example, reasoning through the layout step by step
- Communication locality principle: keep the most communication-heavy parallelism (tensor) within a node, the least (data) across nodes

---

## Phase 3 — Memory Optimization (Ch. 9–11)
*Goal: how modern systems train huge models without simply buying infinite memory.*

**Ch 9 — ZeRO and Fully Sharded Data Parallelism (FSDP)**
- Revisiting the Chapter 1 memory budget: why optimizer states and gradients, not just parameters, dominate memory
- ZeRO Stage 1 (shard optimizer states), Stage 2 (+ shard gradients), Stage 3 (+ shard parameters themselves) — built up one stage at a time, with the memory savings computed explicitly at each stage
- FSDP as PyTorch's realization of the ZeRO-3 idea — what it changes operationally vs. plain data parallelism

**Ch 10 — Activation Checkpointing (Gradient Recomputation)**
- The core tradeoff: recompute activations during the backward pass instead of storing them, trading compute for memory
- Worked numerical example: memory saved vs. extra forward-pass compute incurred
- Where to place checkpoints (which layers) — the practical rule of thumb and why it matters

**Ch 11 — Offloading: CPU and NVMe**
- ZeRO-Offload / ZeRO-Infinity: pushing optimizer states or even parameters to CPU RAM or disk, and paying a bandwidth cost to bring them back for compute
- When offloading is worth it (very large models, limited GPU count) vs. when it just adds latency without buying much

---

## Phase 4 — Hardware Realities (Ch. 12–13)
*Goal: connect all of the above to the actual physical hardware Google/Apple interviewers expect you to reason about.*

**Ch 12 — GPU Architecture, NVLink, and Interconnects**
- SMs, tensor cores, HBM — what actually limits GPU throughput for training workloads
- NVLink/NVSwitch bandwidth vs. PCIe vs. Ethernet/InfiniBand between nodes — the bandwidth cliff that shapes every parallelism decision made in Chapters 3–8
- Why interconnect topology (not just raw GPU count) determines which parallelism strategies are viable at a given scale

**Ch 13 — TPU Architecture and Pods**
- How TPUs differ structurally from GPUs (systolic arrays, matrix-multiply-first design)
- TPU pods and the interconnect that links them — why Google's own large-model training leans on this
- Practical implications for how you'd design a training job differently on TPU vs. GPU

---

## Phase 5 — Systems, Fault Tolerance, and Scaling (Ch. 14–16)
*Goal: the operational realities of running training jobs that last days to weeks on hundreds/thousands of devices.*

**Ch 14 — Checkpointing Strategies**
- Why checkpointing is harder than "just save the weights" at scale (synchronization, storage bandwidth, sharded state)
- Synchronous vs. asynchronous checkpointing, checkpoint frequency tradeoffs
- Recovering from a crash: what needs to be restored (model, optimizer state, data-loader position, RNG state) and why missing any of these silently corrupts a resumed run

**Ch 15 — Fault Tolerance and Elastic Training**
- Why failures are a certainty, not an edge case, at large enough scale (expected time-to-failure math, worked through simply)
- Elastic training: adding/removing workers mid-job without restarting from scratch
- Straggler mitigation — what happens when one device is just slow, not failed

**Ch 16 — Scaling Laws and Batch Size Scaling**
- Why bigger batch sizes eventually stop helping (gradient noise scale, diminishing returns) — the practical ceiling on data parallelism
- Learning rate scaling rules as batch size grows (linear scaling rule, warmup) — why this isn't optional at scale
- A brief, practical connection to Chinchilla/scaling-law style reasoning about compute-optimal model and data size, kept at interview-appropriate depth

---

## Phase 6 — Interview Mastery (Ch. 17–19)
*Goal: convert all of the above into fast, confident interview performance.*

**Ch 17 — Whiteboard Problem Bank**
- Compute a model's memory budget across parameters/gradients/optimizer states/activations for a given model size and precision
- Derive ring all-reduce's communication cost and compare it to a naive all-to-all
- Work out an effective-batch-size and learning-rate-scaling calculation from given constraints

**Ch 18 — System Design Case Studies (dialogue-format mocks)**
- "Design the distributed training setup for a large language model on 1,000 GPUs, given a memory/interconnect budget"
- "Your training job keeps crashing every 6 hours — diagnose and fix"
- Full L5-vs-L6 answer breakdowns, same format as your other system-design mocks

**Ch 19 — Rapid-Fire Review & L5-vs-L6 Differentiators**
- Consolidated comparison tables (data vs. tensor vs. pipeline parallelism; ZeRO stages; precision formats)
- The most likely follow-up questions with model answers
- Full traps checklist compiled from every chapter

---

## Suggested pacing

19 chapters — a similar density to your bandits curriculum. Phase 2 (model parallelism) and Phase 3 (memory optimization) are the densest, most mechanically-involved material and are worth extra time; Phase 5 (systems/fault tolerance) is where L5-vs-L6 answers are most often won or lost, since it's the material candidates are least likely to have hands-on experience with.

Ready to start with **Chapter 1 — Why We Need Distributed Training** whenever you'd like to begin.
