# Chapter 3 — Data Parallelism and Gradient Accumulation

*(Same plain-language style as prior chapters.)*

---

## 3.1 The basic data-parallel training loop, step by step

Recall from Chapter 2: every device gets a full copy of the model, and a different slice of data. Let's walk through exactly what happens in one training step, with a small concrete example — say, 4 GPUs, and a batch of 32 examples we want to process together.

**Step 1 — split the batch.** Divide the 32 examples into 4 chunks of 8 examples each (one chunk per GPU). This per-GPU chunk size (8, here) is often called the **local batch size** or **micro-batch size** — worth knowing both terms, since papers use them somewhat interchangeably.

**Step 2 — each GPU does its own forward and backward pass, independently.** GPU 1 processes its 8 examples through the (identical) model, computes a loss, and backpropagates to get gradients — entirely on its own, using only its own 8 examples. Same for GPUs 2, 3, and 4, all happening at the same time, completely independently at this stage.

**Step 3 — combine (average) the gradients across all 4 GPUs.** This is the crucial synchronization step — without it, each GPU would end up with a *different* model after updating, since each computed gradients from different data. This combining step is done via **all-reduce** (Chapter 4 is entirely about how this works) — for now, just know that after this step, **every GPU ends up with the exact same, averaged gradient.**

**Step 4 — every GPU applies the same optimizer update using the same averaged gradient.** Since every GPU started with an identical model copy and now has an identical averaged gradient, every GPU ends this step with, once again, an identical model — ready for the next step.

**Why this whole loop is mathematically equivalent to training on one giant GPU with the full batch of 32**: averaging 4 gradients, each computed from 8 examples, is the same computation as computing one gradient from all 32 examples directly (basic property of how gradients of an average loss work) — **this equivalence is worth being able to state explicitly**, since it's exactly why data parallelism doesn't change your model's mathematical training behavior, only how fast you get there.

---

## 3.2 Gradient accumulation — simulating a bigger batch than actually fits

Here's a related but genuinely different problem: what if you want an even bigger *effective* batch size than your hardware can physically handle at once — even with data parallelism spread across your GPUs?

**The idea, in plain words**: instead of updating the model's weights after every single forward+backward pass, **keep adding up ("accumulating") gradients across several forward+backward passes**, and only actually update the weights once you've accumulated gradients from as many examples as your target batch size calls for.

### A simple worked example

Say each GPU can only fit 8 examples in memory at once (a **micro-batch** of 8), but you want an **effective batch size** of 32 examples per GPU (maybe because you've found, or a paper reports, that this larger batch size trains better).

- **Micro-step 1**: run forward+backward on 8 examples, computing gradient $g_1$. Don't update weights yet — just store/add $g_1$ into a running total.
- **Micro-step 2**: run forward+backward on the *next* 8 examples, computing gradient $g_2$. Add it to the running total: now you have $g_1 + g_2$.
- **Micro-step 3 and 4**: same idea, ending with a running total of $g_1 + g_2 + g_3 + g_4$.
- **Now, and only now, update the weights** — using the *averaged* accumulated gradient, $(g_1+g_2+g_3+g_4)/4$, which is mathematically equivalent to having computed the gradient from all 32 examples directly (same equivalence idea as Section 3.1, just accumulated over time instead of across devices).

**Why this is useful**: it decouples "how big a batch fits in memory at once" from "how big a batch size you actually want to train with" — you can get the training benefits of a large effective batch size even on hardware that could only ever fit a much smaller micro-batch at a time. The cost is purely **time**: 4 micro-steps take roughly 4× as long as 1 step would, since you're doing the same total amount of forward/backward computation, just spread across more, smaller steps instead of one big one.

---

## 3.3 Effective batch size arithmetic — the calculation interviewers love to ask

This is one of the most common, most concretely-checkable calculations in this entire course, so let's nail the formula, in plain words first:

$$\text{Effective batch size} = (\text{micro-batch size per GPU}) \times (\text{number of GPUs}) \times (\text{gradient accumulation steps})$$

**Worked example**: micro-batch size of 8, 4 GPUs (data parallelism), and 4 gradient accumulation steps:

$$\text{Effective batch size} = 8 \times 4 \times 4 = 128$$

**Walk through why each factor belongs in the formula, in plain words**: the micro-batch size is how many examples each GPU actually processes per forward/backward pass. Multiplying by the number of GPUs accounts for the fact that all GPUs are doing this *simultaneously*, each on their own different data (Section 3.1's data-parallelism idea). Multiplying by the accumulation steps accounts for the fact that you're *also* summing up gradients across several sequential passes before actually updating (Section 3.2's idea) — these two multiplications are stacking two genuinely different mechanisms (across-devices, and across-time) that both contribute to the final effective batch size.

**Why this calculation matters so much in interviews**: being handed a target effective batch size (e.g., "we want an effective batch size of 4 million tokens, like some published large-model training runs") and being asked to reason backward — "given this many GPUs and this much memory per GPU, how many gradient accumulation steps do we need" — is an extremely natural, very commonly-asked whiteboard question, and fluency with this formula is exactly what's needed to answer it quickly and correctly.

---

## 3.4 Production considerations

- **Gradient accumulation is used constantly in real large-model training**, not just as a theoretical trick — it's one of the most common practical tools for hitting a target effective batch size when GPU memory is the binding constraint, and virtually every major training framework (PyTorch, DeepSpeed, Megatron) has built-in support for it.
- **There's a real time cost to gradient accumulation that's easy to forget**: more accumulation steps means more sequential forward/backward passes before each actual weight update, which means the wall-clock time per "effective step" goes up proportionally — so gradient accumulation trades memory for extra time, similar in spirit to activation checkpointing's compute-for-memory tradeoff (foreshadowing Chapter 10).
- **The all-reduce step (Section 3.1, Step 3) is not free** — it's real communication that takes real time, and its cost is a central topic of the very next chapter; data parallelism's efficiency at scale depends heavily on how well this communication overlaps with computation, not just on how fast the raw compute is.

---

## 3.5 Interview traps

- **Forgetting that data parallelism and gradient accumulation are two separate, stackable mechanisms**, and conflating them into one idea. Data parallelism spreads a batch *across devices simultaneously*; gradient accumulation spreads a batch *across time on the same device(s)*. Both increase the effective batch size, but via genuinely different mechanisms, and the effective-batch-size formula (Section 3.3) needs both factors accounted for separately.
- **Forgetting to actually zero out or reset the accumulated gradient after the weight update.** A common, very real implementation bug: if you don't clear the accumulated gradient buffer after stepping the optimizer, the next round of accumulation starts from a non-zero, stale value — a small but very checkable detail.
- **Not being able to reason backward through the effective-batch-size formula** (i.e., given a target effective batch size and known constraints, solving for the missing piece, like number of accumulation steps needed) — being able to only compute forward (given all three factors, find the product) is a weaker, less flexible level of fluency than being able to solve for any one of the three given the other two.

---

## 3.6 L5-vs-L6 differentiating talking points

- **L5 bar**: can describe the basic data-parallel loop (Section 3.1) and the basic gradient accumulation idea (Section 3.2) correctly.
- **L6 bar**:
  - Can state, and briefly justify, the mathematical equivalence between "average gradients across devices" and "compute the gradient from the full combined batch directly" (Section 3.1) — showing understanding of *why* data parallelism doesn't change training dynamics, not just that it doesn't.
  - Fluently solves the effective-batch-size formula in *either* direction — both computing the effective batch size given the three factors, and solving backward for a missing factor given a target effective batch size — live, on request.
  - Proactively raises the wall-clock time cost of gradient accumulation (Section 3.4) as an explicit tradeoff, rather than presenting it as a purely beneficial trick with no downside.

---

## 3.7 Comprehension checks

1. Walk through the four steps of the basic data-parallel training loop, in your own words.
2. Why is averaging gradients across 4 GPUs, each with 8 examples, mathematically equivalent to computing one gradient from all 32 examples at once?
3. In your own words, what problem does gradient accumulation solve, and what's the real cost of using it?
4. Write the effective batch size formula from memory, and compute it for: micro-batch size 4, 8 GPUs, 2 accumulation steps.
5. If you're told the target effective batch size is 512, you have 8 GPUs, and your micro-batch size (memory-limited) is 16, how many gradient accumulation steps do you need?

---

*Next: Chapter 4 — Communication Primitives: All-Reduce vs. Parameter Servers, where we open up the "combine the gradients" step from Section 3.1 and see exactly how it works, and why ring all-reduce in particular scales so well.*
