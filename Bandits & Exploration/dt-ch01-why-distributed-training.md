# Chapter 1 — Why We Need Distributed Training

*(Plain language first, numbers introduced slowly and explained as we go — same style as the later chapters of your bandits course.)*

---

## 1.1 The one-sentence idea

We use multiple GPUs/TPUs instead of one for exactly two reasons: **either the model doesn't fit in one device's memory, or training on one device would simply take too long.** Everything in this entire course is a strategy for dealing with one or both of these two walls.

---

## 1.2 Wall #1: the model doesn't fit

Let's make "doesn't fit" concrete with real numbers, because this is a very common interview calculation, and it's worth being able to do it fluently, from memory.

**Step 1 — how much memory do the parameters themselves take?**

Say we have a model with 7 billion parameters (a common "small-ish large language model" size, like a 7B model). If we store each parameter in **fp32** (32-bit / 4-byte floating point — we'll explain exactly what this means in Chapter 5), that's:

$$7{,}000{,}000{,}000 \text{ parameters} \times 4 \text{ bytes} = 28{,}000{,}000{,}000 \text{ bytes} = 28 \text{ GB}$$

Just to *store the weights*, in fp32, a 7B model needs 28 GB. A single high-end GPU (say, an 80GB A100/H100) can technically hold this — but we're not done. Training needs much more than just the weights.

**Step 2 — gradients.** During training, you also need to store one gradient value per parameter (the "which direction to nudge this weight" number from backpropagation). That's **another 28 GB**, same size as the parameters themselves, since there's exactly one gradient per parameter.

**Step 3 — optimizer states.** This is the part people most often forget, and it's usually the *biggest* piece. If you're using Adam (the most common optimizer for large models), Adam keeps **two** extra numbers per parameter (a running average of the gradient, and a running average of the squared gradient — you don't need the exact Adam formula memorized for this course, just the fact that it stores 2 extra numbers per parameter). That's **another 56 GB** (2 × 28 GB).

**Running total so far**: $28 + 28 + 56 = 112$ GB — and this is *before* we've stored a single activation (the intermediate values computed during the forward pass, needed for backpropagation) or done a single step of actual training. **A 7B-parameter model, in fp32, with Adam, needs well over 100 GB just for parameters + gradients + optimizer states** — more than a single 80GB GPU can hold, and that's for what's often considered a *small* large language model by current standards. This is Wall #1, made completely concrete: **for many real models, one device simply cannot hold everything needed to train it.**

---

## 1.3 A simple table to keep this memory breakdown straight

| Component | Size (relative to parameter count) | For our 7B example, fp32 |
|---|---|---|
| Parameters | 1× | 28 GB |
| Gradients | 1× | 28 GB |
| Optimizer states (Adam) | 2× | 56 GB |
| **Total (before activations)** | **4×** | **112 GB** |

**The simple rule of thumb worth memorizing**: with Adam and fp32, you need roughly **4 times** the size of the raw parameters just for parameters + gradients + optimizer state, before activations even enter the picture. This single "4×" number is one of the most useful, fast, interview-ready facts in this entire course — being able to produce this table from memory, for any stated model size, is a strong first impression.

---

## 1.4 What about activations?

Activations are the intermediate outputs computed at each layer during the forward pass — you need to keep them around because backpropagation needs them to compute gradients. Unlike parameters/gradients/optimizer states (which depend only on model size), **activation memory depends on batch size and sequence length too** — bigger batches or longer sequences mean more activation memory, on top of everything in the table above. This is exactly why activation memory becomes its own whole topic later (Chapter 10, activation checkpointing) — it's a genuinely separate, and often large, contributor to the total memory picture, not a minor detail.

---

## 1.5 Wall #2: training would simply take too long

Even when a model *does* technically fit on one device, there's a separate problem: modern models are often trained on **trillions** of tokens of data. Even a very fast single GPU, doing one forward+backward pass at a time, would take an impractically long time — months or years — to get through that much data. **This is a pure speed problem, separate from the memory problem in Section 1.2** — the fix here is to have many devices process different chunks of data *simultaneously*, which is exactly what data parallelism (Chapter 3) is for.

**Why it's useful to keep these two walls mentally separate**: memory problems are solved by splitting the *model* (model parallelism, Chapters 6–7) or by being cleverer about what you store (Chapters 9–11). Speed problems are solved by splitting the *data* (data parallelism, Chapter 3) and running more copies in parallel. Real large-scale training combines fixes for both walls at once (Chapter 8) — but understanding them as two separate underlying problems, each with its own family of solutions, is the single most useful organizing idea for this entire course.

---

## 1.6 FLOPs-bound vs. memory-bandwidth-bound — a quick, important distinction

Here's a subtlety worth having ready: **"the GPU is slow" can mean two genuinely different things**, and mixing them up is a common, checkable interview mistake.

- **Compute-bound (FLOPs-bound)**: the GPU is busy doing actual arithmetic (matrix multiplications) essentially the whole time — its arithmetic units are the bottleneck. Buying a GPU with more FLOPs (raw compute capability) directly helps here.
- **Memory-bandwidth-bound**: the GPU's arithmetic units are actually sitting idle a lot of the time, waiting for data to be fetched from memory fast enough to feed them. Here, buying a faster-computing GPU **doesn't help much** — you need faster memory (higher bandwidth), not more raw arithmetic power.

**A simple way to reason about which one you're facing**: big matrix multiplications (like the core operations in a transformer's feedforward layers) tend to be compute-bound — lots of arithmetic per byte of data moved. Operations like fetching embeddings or doing simple elementwise operations on huge tensors tend to be memory-bandwidth-bound — not much arithmetic per byte moved. **This distinction matters throughout the whole course** — for instance, it's part of why activation checkpointing (trading compute for memory, Chapter 10) is often a genuinely good deal: if you have compute to spare but not memory, recomputing something is closer to "free" than it sounds.

---

## 1.7 The interconnect hierarchy — why "just add more GPUs" isn't simple

Here's the last foundational idea for this chapter, and it's the one that shapes almost every design decision in Phases 1–2: **not all connections between devices are equally fast, and the difference is enormous — often 10-100× at each step up the hierarchy.**

Think of it as a set of concentric circles, fastest in the middle:

1. **Within a single GPU**: moving data between the GPU's own compute units and its own memory (HBM) — extremely fast.
2. **Between GPUs in the same server, via NVLink/NVSwitch**: still fast, but a real step down from within-GPU speed.
3. **Between servers in the same data center, via a network (InfiniBand/Ethernet)**: another big step down — often 10× slower or more than NVLink.

**Why this matters, concretely**: any parallelism strategy that requires devices to constantly exchange information (like tensor parallelism, Chapter 6) needs to happen over the *fastest* available connection — which is why, as we'll see in Chapter 8, tensor parallelism is typically kept *within* a single server (over NVLink), while strategies that communicate less often (like data parallelism) can be spread *across* servers without as much penalty. **This one idea — "keep the chattiest communication on the fastest connection" — is the single organizing principle behind how real 3D-parallelism layouts get designed**, and we'll return to it explicitly in Chapter 8.

---

## 1.8 Production considerations

- **The 4× memory rule of thumb (Section 1.3) is something you should be able to apply instantly to any model size an interviewer names** — "how much memory would a 70B parameter model need to train in fp32 with Adam" should immediately trigger "70B × 4 bytes × 4 = 1,120 GB, before activations" as a fluent, fast mental calculation.
- **Real systems almost never train in pure fp32 anymore** — mixed precision (Chapter 5) cuts much of this memory roughly in half, which is a large part of why mixed precision is close to universal in practice, not just a nice-to-have optimization.
- **The interconnect hierarchy (Section 1.7) is why cluster topology, not just GPU count, is often the real constraint** on what's achievable — a company with 1,000 GPUs spread across many separate, poorly-interconnected racks can be meaningfully worse off than a company with fewer GPUs but excellent interconnects between them.

---

## 1.9 Interview traps

- **Forgetting optimizer states when computing memory requirements.** A very common mistake is computing "model size" as just parameters, or parameters + gradients, and forgetting Adam's extra 2× — this alone can be off by nearly 2× on the real answer.
- **Treating "the model is slow" as one single problem.** Not distinguishing compute-bound from memory-bandwidth-bound (Section 1.6) is a real, checkable gap — a strong candidate asks or clarifies which regime is relevant before proposing a fix.
- **Assuming more GPUs always means proportionally faster training**, without accounting for the interconnect hierarchy (Section 1.7) — communication overhead, not just raw device count, often determines real-world scaling efficiency, a theme that will recur constantly through this course.

---

## 1.10 L5-vs-L6 differentiating talking points

- **L5 bar**: can name the two "walls" (memory, speed), and can roughly compute parameter memory for a stated model size.
- **L6 bar**:
  - Produces the full 4× memory breakdown (Section 1.3) fluently and immediately, including correctly explaining *why* Adam specifically contributes 2× (two running-average buffers per parameter), not just citing "optimizer overhead" vaguely.
  - Distinguishes compute-bound from memory-bandwidth-bound unprompted when discussing why something is slow, rather than treating all slowness as the same kind of problem.
  - Connects the interconnect hierarchy (Section 1.7) forward to parallelism-strategy placement (tensor parallelism within a node, data parallelism across nodes) even before that material is formally covered in Chapter 8 — showing they already see the throughline of the whole course.

---

## 1.11 Comprehension checks

1. What are the two "walls" that make distributed training necessary, and are they the same problem or two different problems?
2. Walk through the memory breakdown for a 3B-parameter model in fp32 with Adam — how much memory for parameters, gradients, optimizer states, and the total (before activations)?
3. Why does activation memory depend on batch size and sequence length, unlike parameter/gradient/optimizer-state memory?
4. What's the difference between being compute-bound and being memory-bandwidth-bound, and why does buying a faster GPU only help with one of them?
5. Why does it matter, when designing a parallelism strategy, that NVLink is much faster than the network between servers?

---

*Next: Chapter 2 — The Parallelism Taxonomy, where we build the simple mental map (split the data, split the layers, or split each layer itself) that every subsequent chapter in Phases 1–2 fills in.*
