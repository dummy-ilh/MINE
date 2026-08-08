# Chapter 8 — Combining Strategies: 3D Parallelism

*(Plain language first, with a full worked GPU-layout example — this chapter is the payoff for everything built up in Chapters 2–7.)*

---

## 8.1 Why we combine three strategies instead of picking one

Recall Chapter 2's core framing: tensor and pipeline parallelism fix the **memory wall** (fitting a huge model across devices); data parallelism fixes the **speed wall** (processing enough data fast enough). A frontier-scale model training job typically needs **both** fixed simultaneously — the model is too big for any single device, *and* there's far too much data to process on just a few devices in reasonable time. **3D parallelism is simply using all three strategies together, each handling the specific problem it's best suited for.**

---

## 8.2 The organizing principle: communication locality

Before the worked example, let's state the single rule that drives every layout decision in this chapter — we've been building toward this since Chapter 1:

**Rule: put the most communication-heavy (most frequent, most bandwidth-hungry) type of parallelism on the fastest available connection, and the least communication-heavy type on the slowest available connection.**

Recall the ranking from earlier chapters:
- **Tensor parallelism** (Chapter 6): communicates multiple times *within* every single forward/backward pass — the most chatty by far. → Needs the **fastest** connection (NVLink, within one server node).
- **Pipeline parallelism** (Chapter 7): communicates once between each pair of adjacent pipeline stages, per micro-batch — moderately chatty. → Can tolerate a **somewhat slower** connection (can span across nodes, though ideally still well-connected ones).
- **Data parallelism** (Chapter 3–4): communicates once per full training step (the gradient all-reduce) — the least chatty. → Can tolerate the **slowest** available connection (across racks, across the whole cluster).

This ranking — tensor tightest, pipeline in the middle, data loosest — is the single most important thing to internalize in this entire chapter, and it directly determines the physical layout in the worked example below.

---

## 8.3 A worked example: laying out 512 GPUs

Let's actually do this, step by step, the way an interviewer might ask you to reason through it live.

**Given**: 512 GPUs total, arranged as 64 servers × 8 GPUs per server (a very standard real-world configuration — 8 GPUs per server connected via NVLink is common).

**Step 1 — decide the tensor-parallel degree.** Recall from Chapter 6 (Section 6.6): tensor parallelism should stay within one server's fast NVLink domain. With 8 GPUs per server, a natural choice is **tensor-parallel degree = 8** — using all 8 GPUs within a single server purely for tensor parallelism, splitting each layer's matrix multiplications across all 8.

**Step 2 — decide the pipeline-parallel degree.** Now we need to decide how many "tensor-parallel groups of 8" to chain together into a pipeline. Say our model has enough layers that we want, for example, **pipeline-parallel degree = 8** — meaning we chain together 8 of these "8-GPU tensor-parallel groups," one holding each pipeline stage's layers. That's $8 \text{ (tensor)} \times 8 \text{ (pipeline)} = 64$ GPUs, forming one complete "copy" of the whole model, split across both tensor and pipeline dimensions.

**Step 3 — decide the data-parallel degree.** We have 512 total GPUs, and each complete model copy (tensor + pipeline combined) uses 64 GPUs — so we can fit $512 / 64 = 8$ complete, independent copies of the whole model side by side. **Data-parallel degree = 8** — 8 independent replicas, each internally split via tensor+pipeline parallelism, all training on different data shards simultaneously, synchronized via the (relatively infrequent) gradient all-reduce from Chapter 4.

**The final layout, summarized**: $8 \text{ (tensor)} \times 8 \text{ (pipeline)} \times 8 \text{ (data)} = 512$ GPUs total — exactly matching our given GPU count. This is exactly what "3D parallelism" means concretely: three separate parallelism "dimensions," multiplied together to use all your available devices.

---

## 8.4 Checking the layout against the communication-locality principle

Let's verify Step 1–3's layout actually respects Section 8.2's rule:

- **Tensor-parallel groups (the chattiest)**: each group is exactly the 8 GPUs within one physical server, connected via NVLink — the fastest connection, exactly as required.
- **Pipeline-parallel groups (moderately chatty)**: each pipeline stage boundary crosses *between* servers (since each server is fully consumed by one tensor-parallel group) — using the network between servers, a real step down from NVLink, but this is tolerable since pipeline parallelism communicates far less frequently than tensor parallelism (once per micro-batch boundary, not multiple times per single layer).
- **Data-parallel groups (least chatty)**: the 8 independent model replicas can be spread as widely as necessary across the cluster — even in different racks or (in some very large-scale setups) different data center zones — since data parallelism's gradient all-reduce happens only once per full training step, the least frequently of all three.

**This is the entire point of the worked example**: the layout isn't arbitrary — it's a direct, mechanical consequence of matching each parallelism strategy's communication frequency to an appropriately-fast (or appropriately-tolerant-of-slowness) physical connection.

---

## 8.5 A simple way to reason through a *new* layout problem, live

If an interviewer changes the numbers (different GPU count, different per-server GPU count, different model size), here's the repeatable reasoning process, in order:

1. **Tensor-parallel degree**: set this to (at most) the number of GPUs that share the fastest interconnect within one server — this is usually a hardware-fixed number (like 8), not something you'd typically want to exceed.
2. **Pipeline-parallel degree**: determined by how many stages you need to make the model's layers fit comfortably in memory per stage, given Chapter 1's memory-budget reasoning and Chapter 7's bubble/memory tradeoffs — more stages means a deeper pipeline (potentially worse bubble, per Chapter 7) but less memory needed per stage.
3. **Data-parallel degree**: simply whatever's left over — total GPUs divided by (tensor-parallel degree × pipeline-parallel degree) — used to scale up throughput on however many complete model replicas you can afford.

**This three-step process (tensor degree from hardware, pipeline degree from memory needs, data degree from what's left) is exactly the kind of structured, step-by-step reasoning an interviewer wants to see**, rather than a memorized single "correct" layout — the actual numbers change based on the specific model and cluster, but the reasoning process stays the same.

---

## 8.6 Production considerations

- **Real frameworks (Megatron-LM, DeepSpeed) have built-in support for specifying exactly this kind of 3D layout** (tensor-parallel size, pipeline-parallel size, data-parallel size as separate configuration parameters) — this isn't just a theoretical exercise, it's literally how these systems are configured in practice.
- **The "8 GPUs per server" assumption in the worked example is a real, common hardware configuration** (e.g., NVIDIA DGX-style servers), which is exactly why tensor-parallel degree of 8 shows up so often in practice — it's not an arbitrary round number, it's tied to real, common hardware topology.
- **Real large-scale training runs sometimes add a "4th dimension"** beyond the three covered here — e.g., ZeRO-style sharding (Chapter 9) applied *within* the data-parallel dimension, effectively getting some of model-parallelism's memory benefits *combined with* data parallelism's simplicity — worth flagging as a preview of where Chapter 9 fits into this overall picture.

---

## 8.7 Interview traps

- **Picking parallelism degrees arbitrarily rather than reasoning from the communication-locality principle.** A candidate who proposes a 3D layout without justifying *why* tensor parallelism gets the smallest, fastest-interconnected grouping is missing the core organizing idea of this entire chapter.
- **Forgetting to sanity-check that the three degrees multiply to the total GPU count.** This is a simple, checkable arithmetic step (Section 8.3) that's easy to skip under interview pressure but signals careful, rigorous thinking when done explicitly.
- **Treating pipeline-parallel degree as a free parameter without connecting it back to the memory/bubble tradeoff from Chapter 7.** The pipeline-parallel degree isn't picked in isolation — it's a real tradeoff between per-stage memory (more stages, less memory per stage) and pipeline bubble severity (more stages, generally a bigger bubble to manage).

---

## 8.8 L5-vs-L6 differentiating talking points

- **L5 bar**: knows that 3D parallelism combines all three strategies, and can roughly describe why.
- **L6 bar**:
  - Can walk through the full worked layout example (Section 8.3) live, reasoning step by step (tensor degree from hardware → pipeline degree from memory needs → data degree from what's left), rather than reciting a memorized configuration.
  - Explicitly justifies each placement decision using the communication-locality principle (Section 8.2), connecting tensor/pipeline/data parallelism's differing communication frequencies back to the interconnect hierarchy from Chapter 1.
  - When given new, different numbers by the interviewer, can adapt the same three-step reasoning process (Section 8.5) live, rather than only being able to reproduce one memorized example.

---

## 8.9 Comprehension checks

1. State the communication-locality principle in your own words — which parallelism strategy gets the fastest connection, and why?
2. Walk through the worked example from Section 8.3 — for 512 GPUs (64 servers × 8 GPUs), what are the tensor, pipeline, and data-parallel degrees, and why do they multiply to 512?
3. Why does tensor parallelism get placed within a single server, while data parallelism can be spread across the whole cluster?
4. Using the three-step reasoning process from Section 8.5, how would you lay out parallelism degrees for 256 GPUs, 4 GPUs per server, and a model that needs 4 pipeline stages to fit comfortably in memory?
5. What's one reason real systems sometimes add a "4th dimension" (like ZeRO-style sharding) on top of the three covered in this chapter?

---

*This closes out Phase 2 (Model Parallelism). Next: Chapter 9 — ZeRO and Fully Sharded Data Parallelism, opening Phase 3 by showing how to get much of model-parallelism's memory benefit while keeping data-parallelism's relative simplicity.*
