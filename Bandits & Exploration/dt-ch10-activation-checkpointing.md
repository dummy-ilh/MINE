# Chapter 10 — Activation Checkpointing (Gradient Recomputation)

*(Plain language first, with a worked numerical tradeoff calculation.)*

---

## 10.1 Reopening the activation-memory problem from Chapter 1

Recall Chapter 1, Section 1.4: activations (the intermediate outputs computed during the forward pass, at every layer) need to be kept around because backpropagation needs them to compute gradients — and, unlike parameters/gradients/optimizer states, **activation memory grows with batch size and sequence length**, not just model size. For long sequences or large batches, activation memory can become the single dominant memory consumer, even after ZeRO/FSDP (Chapter 9) has already sharded everything else. **Activation checkpointing is the standard tool for addressing this specific piece of the memory puzzle.**

---

## 10.2 The core idea, in plain words

**Normally**: during the forward pass, every layer's activations get stored in memory, all the way through the network, so that when the backward pass runs (in reverse order), each layer already has the activation values it needs, sitting ready in memory.

**With activation checkpointing**: instead of storing *every* layer's activations, **only store a small subset of "checkpoint" layers' activations** — throw away everything in between. Then, during the backward pass, when you reach a point where you need an activation you didn't save, **just recompute it** — rerun the small forward-pass segment between the nearest saved checkpoint and the layer you actually need, right there in the middle of the backward pass, to regenerate the needed values on the spot.

**The trade being made, stated plainly**: you're trading **extra compute** (redoing parts of the forward pass a second time) for **less memory** (not storing every single layer's activations the whole time) — this is exactly the same "spend one resource to save another" flavor of tradeoff you've now seen repeatedly (gradient accumulation trading time for memory in Chapter 3, ZeRO trading communication for memory in Chapter 9) — activation checkpointing is simply the version of this idea applied specifically to activations.

---

## 10.3 A worked numerical example

Say a model has 24 layers, and storing every layer's activations (no checkpointing) takes, say, 24 GB total (1 GB per layer, for simplicity).

**Without checkpointing**: 24 GB of activation memory, but the backward pass needs zero extra recomputation — every activation it needs is already sitting in memory.

**With checkpointing, saving every 4th layer** (checkpoints at layers 4, 8, 12, 16, 20, 24 — 6 checkpoints total): you now only store 6 GB of activations (6 checkpoint layers × 1 GB each) instead of 24 GB — a **4× reduction in activation memory**. But during the backward pass, whenever you need an activation from, say, layer 10 (which wasn't saved), you have to **recompute** the forward pass from the nearest earlier checkpoint (layer 8) back up to layer 10 — redoing roughly 2 layers' worth of forward computation, on average, for each "gap" between checkpoints.

**The general pattern, stated as a simple rule of thumb**: if you save 1 out of every $k$ layers' activations, you roughly cut activation memory by a factor of $k$, at the cost of roughly re-doing $1/k$ of an *additional* forward pass's worth of total compute (since you're redoing, on average, about half of each $k$-layer gap, across all the gaps) — **a genuinely tunable knob**, where $k$ controls exactly how aggressively you trade compute for memory.

---

## 10.4 Why this trade is often an excellent deal in practice

Recall Chapter 1, Section 1.6's compute-bound vs. memory-bandwidth-bound distinction. **The forward pass recomputation activation checkpointing introduces is pure extra compute** — and if your training run is currently **memory-constrained rather than compute-constrained** (a very common situation for large models with long sequences), you likely have some "spare" compute capacity sitting around anyway, simply because memory ran out before compute did. **In that situation, trading some of that spare compute for a large memory reduction is close to a free win** — you weren't fully using your compute budget anyway, so redoing some forward-pass work costs you comparatively little in wall-clock time, while potentially letting you fit a much larger batch size or sequence length that wouldn't have been possible otherwise.

**When it's a worse deal**: if your training is already tightly compute-bound (every bit of GPU compute capacity is already being fully used), adding recomputation directly slows down wall-clock training time, with no "spare capacity" to absorb the cost — in this regime, activation checkpointing is a much more real tradeoff, and you'd only reach for it if the memory savings were genuinely necessary to fit the model/batch at all (rather than just a nice-to-have).

---

## 10.5 Where to place checkpoints — the practical rule of thumb

**The simple, commonly-used heuristic**: checkpoint at the boundaries of **major repeated blocks** in the architecture — e.g., for a transformer, checkpoint once per transformer layer (saving the activation right after each full transformer block, discarding everything computed *inside* that block) rather than checkpointing at some arbitrary, finer-grained point in the middle of an attention computation. **Why this specific placement is sensible**: transformer layers are architecturally repeated, roughly-uniform-cost blocks — checkpointing at these natural boundaries gives you a predictable, easy-to-reason-about memory-vs-compute tradeoff (each "gap" between checkpoints costs roughly the same amount to recompute), rather than an uneven, harder-to-predict tradeoff you'd get from checkpointing at arbitrary, irregular points.

---

## 10.6 Production considerations

- **Activation checkpointing is close to a standard, default-on setting in most large-model training frameworks** (PyTorch's `torch.utils.checkpoint`, DeepSpeed's activation checkpointing options) precisely because the memory savings are usually large and the compute cost, in a typically memory-constrained regime, is usually modest — it's a very commonly "just turned on" optimization, not an exotic, rarely-used trick.
- **Activation checkpointing composes naturally with everything else in this course** — it works alongside data parallelism, tensor/pipeline parallelism, and ZeRO/FSDP, since it's purely about *how* one device manages its own activation memory, independent of how devices are otherwise coordinating with each other.
- **Sequence length is often the single biggest driver of activation memory in modern large language models** (since activation memory tends to scale with sequence length, often more than linearly for attention specifically) — this is exactly why activation checkpointing has become especially important as models have moved toward much longer context windows.

---

## 10.7 Interview traps

- **Describing activation checkpointing as simply "not storing some activations," without mentioning that they get *recomputed* during the backward pass.** The recomputation step is the entire mechanism — without it, you'd simply be missing information the backward pass needs, not making a deliberate tradeoff.
- **Treating activation checkpointing as a strictly free memory-reduction trick.** As shown in Section 10.4, whether it's a "good deal" genuinely depends on whether your training run is currently compute-bound or memory-bound — a strong answer names this dependency explicitly rather than presenting checkpointing as universally costless.
- **Not having a concrete answer for *where* to place checkpoints** (Section 10.5) — a candidate who understands the general idea but can't say "at natural architectural block boundaries, like once per transformer layer" is missing a genuinely practical, checkable detail.

---

## 10.8 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describes activation checkpointing as trading compute for memory via selective recomputation.
- **L6 bar**:
  - Can walk through a worked numerical tradeoff like Section 10.3 (memory savings and recomputation cost as a function of checkpoint spacing $k$), rather than describing the idea only qualitatively.
  - Explicitly connects the "is this a good deal" question back to Chapter 1's compute-bound vs. memory-bandwidth-bound distinction (Section 10.4), reasoning about when checkpointing is close to free versus when it's a real cost.
  - Names the natural-block-boundary checkpoint placement heuristic (Section 10.5) and explains *why* it produces a more predictable tradeoff than arbitrary placement.

---

## 10.9 Comprehension checks

1. In your own words, what does activation checkpointing actually do differently from normal training, and what happens during the backward pass when a needed activation wasn't saved?
2. Using the worked example in Section 10.3, why does checkpointing every 4th layer roughly quarter the activation memory, and roughly what extra recomputation cost does this introduce?
3. Why is activation checkpointing often "close to free" when training is memory-bound, but a real cost when training is compute-bound?
4. What's the practical rule of thumb for where to place checkpoints, and why does this placement produce a more predictable tradeoff than arbitrary placement?
5. Why has activation checkpointing become especially important as models have moved toward much longer context windows?

---

*Next: Chapter 11 — Offloading: CPU and NVMe, the last of the three memory-optimization tools, covering ZeRO-Offload and ZeRO-Infinity — pushing optimizer states or parameters off the GPU entirely.*
