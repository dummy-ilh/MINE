# Chapter 6 — Tensor Parallelism

*(Plain language first, with a small worked matrix example to make the mechanics concrete.)*

---

## 6.1 Recap: what tensor parallelism is, in one sentence

From Chapter 2: tensor parallelism splits **the math inside a single layer** across devices — not different layers (that's pipeline parallelism, Chapter 7), but the *same* one operation, computed jointly by multiple devices at once, like several people lifting different corners of one heavy object.

---

## 6.2 Why you'd want to split a single matrix multiplication at all

Recall Chapter 1: some models have individual layers so large that even *one layer's* weight matrix doesn't comfortably fit (or compute efficiently) on a single device — this is especially true for the very large feedforward and attention-projection matrices inside transformer layers at frontier model scale. Tensor parallelism's job is specifically to let several devices **jointly hold and jointly compute** one such oversized matrix multiplication.

---

## 6.3 The core building block: splitting a matrix multiplication

Let's build the intuition with genuinely small numbers, small enough to trace by hand. Say we want to compute $y = xW$, where $x$ is our input (a row vector, say 1×4) and $W$ is a weight matrix (4×4, in our tiny example — real weight matrices are, of course, vastly larger, but the splitting logic is identical at any size).

There are two natural ways to split $W$ across, say, 2 devices — and this distinction (row-split vs. column-split) is the single most important mechanical idea in this chapter.

### Column-parallel split

Split $W$'s **columns** across the 2 devices: Device 1 gets $W$'s first 2 columns, Device 2 gets the last 2 columns. Each device computes $x$ times its own slice of columns — producing *part* of the final output vector $y$ (Device 1 produces $y$'s first 2 entries, Device 2 produces $y$'s last 2 entries). **No communication is needed *during* this particular matrix multiply** — each device has everything it needs (the full input $x$, and its own slice of columns) to compute its piece of the output completely independently. The two output pieces just need to be **concatenated** together afterward to form the full $y$ — a cheap operation, not a real communication-heavy step.

### Row-parallel split

Split $W$'s **rows** across the 2 devices instead: Device 1 gets $W$'s first 2 rows, Device 2 gets the last 2 rows. But now here's the catch: each device needs only the *matching part* of the input $x$ (Device 1 needs $x$'s first 2 entries to multiply against its 2 rows, Device 2 needs $x$'s last 2 entries) — **but you can't just concatenate the two partial results the way you could with column-parallel**. Row-parallel splitting produces two **partial sums** that need to be **added together** (not concatenated) to get the true final $y$ — this addition step **does require real communication between the two devices** (specifically, an all-reduce, exactly the mechanism from Chapter 4).

**The key mechanical takeaway to keep straight**: **column-parallel splitting needs a cheap concatenation at the end; row-parallel splitting needs a real all-reduce at the end.** This single distinction — which operation you need afterward — is exactly why real systems (like Megatron-LM) carefully alternate between column-parallel and row-parallel splits across consecutive layers, specifically to minimize how often the expensive all-reduce step is actually needed.

---

## 6.4 How Megatron-LM applies this to a real transformer layer

Real transformer layers have (at least) two major matrix-multiplication-heavy pieces: the **attention** block and the **feedforward (MLP)** block. Megatron-LM's specific, well-known trick: **use column-parallel splitting for the first matrix multiplication in each block, and row-parallel splitting for the second matrix multiplication in the same block** — arranged so that the cheap-concatenation output of the column-parallel step feeds directly into the row-parallel step's input requirement, and the row-parallel step's necessary all-reduce only needs to happen **once per block** (once for attention, once for the MLP), rather than after every single individual matrix multiplication.

**Why this specific arrangement is clever, restated plainly**: instead of paying the expensive all-reduce cost after *every* matrix multiply inside a transformer layer, careful column-then-row pairing means you only pay it **twice per transformer layer** (once after attention, once after the MLP) — a real, deliberate optimization that directly follows from understanding the column-vs-row distinction in Section 6.3.

---

## 6.5 The communication cost this introduces, and why it must stay on fast interconnect

Here's the critical point that connects directly back to Chapter 1's interconnect hierarchy (Section 1.7): **tensor parallelism's all-reduce steps happen *during* a single forward (and backward) pass — potentially many times per training step**, not once per step the way data parallelism's gradient-averaging all-reduce does (Chapter 3). This means tensor parallelism is **far more communication-hungry, far more often**, than data parallelism.

**The direct consequence**: tensor parallelism is only practical when the devices involved are connected by very fast, very low-latency links — in practice, this almost always means **keeping tensor-parallel groups within a single server node**, connected via NVLink/NVSwitch (Chapter 1's fastest tier), never spread across the much slower network between separate servers. This is exactly the "keep the chattiest communication on the fastest connection" principle previewed at the end of Chapter 1, now made fully concrete: **tensor parallelism is the chattiest communication pattern in this whole course, so it gets the fastest connection.**

---

## 6.6 Production considerations

- **Tensor parallelism's degree (how many devices you split a single layer across) is typically capped by how many GPUs share fast NVLink connectivity within one server** (often 8 GPUs per server in common configurations) — this is a hard practical ceiling, not an arbitrary choice, directly following from Section 6.5's reasoning.
- **Real transformer implementations (Megatron-LM, and its descendants used inside most major LLM training stacks) implement the column-then-row pairing from Section 6.4 as a standard, well-tested pattern** — this isn't a niche academic trick, it's close to the default way large transformer layers get tensor-parallelized in practice.
- **Tensor parallelism adds real engineering complexity** (every layer's forward and backward pass needs to be rewritten to handle the split-and-combine logic) compared to data parallelism's relative simplicity — this is part of why it's reached for specifically when memory constraints force the issue (Chapter 2's "which wall" framing), not used by default.

---

## 6.7 Interview traps

- **Confusing which split (row vs. column) requires communication.** The precise, checkable fact: column-parallel needs only a cheap concatenation; row-parallel needs a real all-reduce. Getting this backward is a common and serious mistake.
- **Not knowing why Megatron specifically alternates column-then-row** — a candidate who can name "column-parallel" and "row-parallel" but can't explain *why* they're paired in that specific order (to minimize all-reduce frequency, Section 6.4) is at a shallower level of understanding.
- **Forgetting that tensor parallelism's all-reduce happens multiple times per step, not once** — conflating its communication pattern with data parallelism's (once-per-step) communication pattern misses the entire reason tensor parallelism needs to stay within a fast-interconnect node.

---

## 6.8 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describes tensor parallelism as splitting a single layer's computation across devices, and knows it requires more frequent communication than data parallelism.
- **L6 bar**:
  - Can walk through the row-vs-column split distinction with a small concrete matrix example (Section 6.3), correctly identifying which one needs concatenation vs. all-reduce.
  - Explains Megatron's column-then-row pairing (Section 6.4) as a deliberate optimization to minimize all-reduce frequency, not just a naming convention.
  - Explicitly connects tensor parallelism's communication frequency back to Chapter 1's interconnect hierarchy, correctly explaining *why* it must stay within a single fast-interconnect node — and can state the practical consequence (tensor-parallel degree capped by GPUs-per-server) as a concrete design constraint.

---

## 6.9 Comprehension checks

1. In your own words, what's the difference between column-parallel and row-parallel splitting of a matrix multiplication?
2. Why does row-parallel splitting require an all-reduce, while column-parallel splitting only requires a cheap concatenation?
3. Why does Megatron-LM specifically pair a column-parallel step followed by a row-parallel step within each transformer block?
4. Why must tensor parallelism's communication happen over the fastest available interconnect, unlike data parallelism's gradient-averaging communication?
5. What's a practical, hardware-driven reason tensor-parallel degree is often capped at the number of GPUs in a single server?

---

*Next: Chapter 7 — Pipeline Parallelism, where we cover splitting the model's layers across devices, the "bubble" problem this creates, and how micro-batching (GPipe) and interleaved scheduling (PipeDream) fix it.*
