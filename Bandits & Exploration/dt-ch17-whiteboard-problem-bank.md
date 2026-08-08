# Chapter 17 — Whiteboard Problem Bank

*(Opening Phase 6 — hands-on practice, not new theory.)*

---

## 17.1 What this chapter is for

Same purpose as your bandits course's equivalent chapter: pure practice, the kind of thing you'd actually be asked to do live. Three exercises: memory-budget calculations, a communication-cost derivation, and batch-size/learning-rate scaling problems.

---

## 17.2 Exercise 1: Full memory-budget calculation, from scratch

**Prompt**: "We want to train a 13-billion-parameter model using bf16 for the forward/backward pass, with Adam, using ZeRO Stage 2, across 16 GPUs. Estimate the per-GPU memory needed for parameters, gradients, and optimizer states (ignore activations for now)."

**Walking through it, step by step, the way you'd talk through it live**:

**Step 1 — parameters, in bf16.** $13 \times 10^9$ parameters × 2 bytes (bf16 is 2 bytes, half of fp32's 4) = 26 GB. Under ZeRO Stage 2, parameters are *not* sharded (recall Chapter 9 — only gradients and optimizer states are sharded at Stage 2), so **every GPU still holds the full 26 GB of parameters.**

**Step 2 — gradients, in bf16, sharded across 16 GPUs.** Full gradient size: $13\times10^9 \times 2$ bytes = 26 GB. Sharded across 16 GPUs: $26 / 16 = 1.625$ GB per GPU.

**Step 3 — optimizer states.** Here's a genuinely important, easy-to-miss detail: **optimizer states are almost always kept in fp32, even in an otherwise-bf16 training setup** (recall Chapter 5, Section 5.7's "fp32 master weights" point) — this is specifically to preserve numerical precision for the optimizer's own accumulated statistics over many steps. So: Adam's 2 buffers per parameter, in fp32 (4 bytes each): $13\times10^9 \times 4 \text{ bytes} \times 2 = 104$ GB full size. Sharded across 16 GPUs (ZeRO Stage 2 shards this): $104/16 = 6.5$ GB per GPU.

**Step 4 — sum it up.** $26 \text{ (params)} + 1.625 \text{ (gradients)} + 6.5 \text{ (optimizer states)} = 34.125$ GB per GPU, before activations.

**The key teaching point of this exercise**: notice that **optimizer states, even after sharding across 16 GPUs, are still the single largest contributor** (6.5 GB, versus 1.625 GB for sharded gradients) — precisely because their *unsharded* size (104 GB, due to fp32 + Adam's 2 buffers) was so much bigger to begin with. This is exactly the kind of multi-step reasoning (get the pre-sharding size right first, in the right precision, *then* apply the sharding factor) that separates a fluent answer from a fumbled one.

---

## 17.3 Exercise 2: Deriving ring all-reduce's cost formula, live

**Prompt**: "Explain, from first principles, why ring all-reduce's per-device communication cost is roughly $2\times$ the gradient size, regardless of the number of devices."

**Walking through it, step by step**:

"Ring all-reduce has two phases — reduce-scatter and all-gather — each taking $N-1$ steps for $N$ devices, so $2(N-1)$ steps total. Before starting, we split the gradient into $N$ equal chunks, so each chunk is $1/N$ of the full gradient size. In each of the $2(N-1)$ steps, a device sends and receives exactly one chunk. So the total data moved per device is $2(N-1)$ chunks, each of size (full gradient / $N$) — giving a total of $\frac{2(N-1)}{N} \times$ (full gradient size). As $N$ gets large, $(N-1)/N$ approaches 1, so the whole expression approaches $2 \times 1 \times$ (full gradient size) = twice the gradient size — and this holds regardless of how large $N$ actually is, which is exactly why ring all-reduce scales so well: your per-device communication cost doesn't keep growing as you add more devices to the ring."

**This is worth being able to reproduce close to verbatim, live, since it's one of the highest-value derivations in the whole course** — both because it's genuinely likely to come up, and because walking through it cleanly signals real understanding of Chapter 4's material, not just memorized vocabulary.

---

## 17.4 Exercise 3: Batch size and learning rate scaling, worked backward

**Prompt**: "You're currently training with a micro-batch size of 4 per GPU, 8 GPUs, no gradient accumulation, and a learning rate of 0.0002. You want to scale up to 64 GPUs while keeping the same micro-batch size per GPU and no gradient accumulation. What's your new effective batch size, and what learning rate would the Linear Scaling Rule suggest?"

**Walking through it, step by step**:

**Step 1 — current effective batch size.** Using Chapter 3's formula: micro-batch × GPUs × accumulation steps = $4 \times 8 \times 1 = 32$.

**Step 2 — new effective batch size.** Same formula, new GPU count: $4 \times 64 \times 1 = 256$.

**Step 3 — the scaling factor.** New batch size / old batch size = $256 / 32 = 8\times$.

**Step 4 — apply the Linear Scaling Rule (Chapter 16, Section 16.3).** New learning rate = old learning rate × scaling factor = $0.0002 \times 8 = 0.0016$.

**Step 5 — don't forget warmup (Chapter 16, Section 16.4).** A complete answer notes that this new, larger learning rate (0.0016) shouldn't be applied from step 1 — it should be reached gradually via a warmup schedule over some number of initial steps, to avoid early-training instability.

**Why Step 5 matters for a complete answer**: a candidate who correctly computes 0.0016 but stops there has given a *mechanically correct but incomplete* answer — explicitly adding the warmup caveat, unprompted, is exactly the kind of small addition that signals genuine practical fluency rather than just formula recall.

---

## 17.5 Production considerations

- **These three exercise types (memory budgeting, communication-cost derivation, batch/LR scaling) cover the large majority of "quantitative whiteboard" questions** that come up in distributed-training-focused interviews — being fluent in all three, in either direction (forward calculation or backward-solving for a missing variable, as in Exercise 3), is disproportionately high-value prep.
- **Real engineers doing capacity planning for a training run perform exactly these calculations**, often before writing a single line of training code — this isn't just interview theater, it's a genuine, common part of the job.

---

## 17.6 Interview traps

- **In memory-budget calculations, forgetting that optimizer states are usually kept in fp32 even during bf16/fp16 training** (Exercise 1, Step 3) — this is a specific, easy-to-miss detail that changes the answer meaningfully, and interviewers who know this material will specifically probe for it.
- **In the ring all-reduce derivation, forgetting the factor of 2** (i.e., forgetting that there are *two* phases — reduce-scatter *and* all-gather — each contributing $(N-1)$ steps) and instead only accounting for one phase.
- **In batch-size/LR scaling problems, computing the new learning rate correctly but forgetting to mention warmup** — a mechanically correct but practically incomplete answer.

---

## 17.7 L5-vs-L6 differentiating talking points

- **L5 bar**: can work through all three exercise types with some guidance or a bit of hesitation, arriving at roughly correct answers.
- **L6 bar**:
  - Works through all three exercises fluently and quickly, with minimal hesitation, correctly handling the "easy to miss" details (fp32 optimizer states, the factor of 2, warmup) proactively rather than needing to be prompted.
  - Can adapt each exercise to different numbers on the fly, live, rather than only being able to reproduce one memorized worked example.
  - Explains the *reasoning* behind each step while computing (as modeled in the walkthroughs above), not just the final numeric answer — showing the calculation is understood, not memorized.

---

## 17.8 Comprehension checks

1. Redo Exercise 1 for a 7B-parameter model, fp16 (not bf16), ZeRO Stage 3, across 8 GPUs — what's the per-GPU memory for parameters, gradients, and optimizer states?
2. Reproduce the ring all-reduce cost derivation from Exercise 2 in your own words, without looking, for $N=6$ devices — what fraction of "twice the gradient size" does this represent?
3. Redo Exercise 3 with a starting setup of micro-batch 8, 4 GPUs, learning rate 0.0001, scaling up to 32 GPUs — what's the new effective batch size and new learning rate?
4. Why does ZeRO Stage 2 leave parameters unsharded but shard gradients and optimizer states — how does this affect a memory-budget calculation compared to Stage 3?
5. Why is "forgetting warmup" considered an incomplete answer, even when the learning rate scaling arithmetic itself is correct?

---

*Next: Chapter 18 — System Design Case Studies, running full dialogue-format mock interviews for designing a large-scale training system and diagnosing a recurring production failure.*
