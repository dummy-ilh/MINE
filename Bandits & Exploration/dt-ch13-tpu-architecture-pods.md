# Chapter 13 — TPU Architecture and Pods

*(Plain language first — closing out Phase 4.)*

---

## 13.1 The one-sentence idea

TPUs (Tensor Processing Units) are chips Google designed **specifically and only** for the matrix-multiplication-heavy workloads that dominate deep learning — unlike GPUs, which started life as general-purpose graphics chips and were later adapted for this workload. This "built for one job from the start" design philosophy explains almost every structural difference covered in this chapter.

---

## 13.2 The systolic array — TPU's core computational idea

**GPUs (Chapter 12)** are built from many separate, relatively independent SMs, each capable of running its own threads, which then need to coordinate to jointly perform a large matrix multiplication.

**TPUs** are built around a fundamentally different structure called a **systolic array** — a large, tightly-packed grid of simple multiply-and-add units, wired directly to their immediate neighbors, through which data **flows rhythmically, in a wave-like pattern**, with each unit doing one small piece of a matrix multiplication and passing its partial result directly to the next unit in the grid, without needing to go back out to memory in between.

**Why this matters, in plain words**: for the specific operation deep learning cares most about — large matrix multiplications — a systolic array can be dramatically more efficient than a general-purpose collection of independent processing units, precisely because the data flow pattern of a matrix multiplication maps almost perfectly onto the systolic array's own rhythmic, neighbor-to-neighbor data flow. **The tradeoff**: this specialization makes TPUs excellent at matrix-multiplication-heavy workloads specifically, but much less flexible for arbitrary, general-purpose computation than a GPU — which is exactly the "built for one job" tradeoff mentioned in Section 13.1.

---

## 13.3 TPU Pods — the interconnect story

Recall Chapter 12's interconnect hierarchy for GPUs (within-GPU → NVLink → cross-server network). **TPUs have their own analogous hierarchy, but built around a custom, dedicated high-speed interconnect that directly links many TPU chips together into what Google calls a "pod"** — a single pod can contain a very large number of TPU chips (this number has grown across TPU generations), all connected via this dedicated interconnect, specifically designed to support exactly the kind of communication-heavy parallelism strategies covered in Chapters 3–9 (all-reduce, all-gather, and so on) at very large scale, within one tightly-integrated system.

**Why this is a meaningfully different design point from GPU clusters**: a GPU cluster's "fast interconnect domain" (NVLink) is typically limited to the GPUs within one physical server (commonly 8, per Chapter 8's worked example) — beyond that, you're on a comparatively slower network. **A TPU pod's fast interconnect domain can span a much larger number of chips**, meaning strategies that would need to "step down" to a slower connection on a GPU cluster (like tensor parallelism spanning beyond one server) can potentially stay on TPU's faster, pod-wide interconnect for longer, changing some of the tradeoffs from Chapter 8's communication-locality reasoning.

---

## 13.4 Practical implications for how you'd design a training job differently

**On GPU**, the 3D-parallelism layout reasoning from Chapter 8 is heavily shaped by the sharp bandwidth cliff between "within one server" (NVLink) and "across servers" (network) — this cliff is the main reason tensor-parallel degree gets capped at the per-server GPU count.

**On TPU**, because the pod's fast interconnect spans a much larger number of chips than a single GPU server does, **the same sharp cliff doesn't appear at nearly as small a scale** — meaning some of the same parallelism strategies can potentially be applied more broadly across more chips before hitting a comparable bandwidth wall. **The core reasoning principles from Chapters 2–9 (memory wall vs. speed wall, communication-locality, ZeRO-style sharding) all still apply on TPU** — what changes is *where* the practical bandwidth cliffs sit, which shifts the specific numbers in a layout calculation like Chapter 8's, without changing the underlying logic.

**A simple, honest interview-level summary**: *"The conceptual framework — data vs. tensor vs. pipeline parallelism, memory vs. speed tradeoffs, communication locality — transfers directly from GPU to TPU. What changes is the specific hardware topology and bandwidth numbers feeding into that same framework, since TPU pods offer a larger fast-interconnect domain than a typical GPU server does."*

---

## 13.5 Production considerations

- **Google's own large-model training (and much of its internal ML infrastructure) is built around TPU pods** — given this course is aimed partly at Google MLE/Applied Scientist interviews, having at least this level of TPU fluency (systolic arrays, pod-scale interconnects, and how they shift the Chapter 8 layout reasoning) is a genuinely relevant, non-generic piece of interview preparation.
- **Apple, by contrast, more commonly trains on GPU-based infrastructure** (and its own specialized chips for on-device inference, a different topic from large-scale training) — worth being aware that TPU-specific depth is more directly relevant to Google-flavored interviews than Apple-flavored ones, though the underlying parallelism concepts transfer either way.
- **Frameworks like JAX are closely associated with TPU training** in practice (having been designed with TPU's execution model closely in mind) — worth knowing this association exists, even without deep JAX-specific expertise, as a concrete, current, real-ecosystem detail.

---

## 13.6 Interview traps

- **Describing TPUs as "just Google's version of a GPU."** The systolic-array architecture (Section 13.2) is a genuinely different computational structure, not merely a rebranded GPU — being able to name and briefly explain this structural difference is a meaningfully stronger answer than treating the two as interchangeable.
- **Not being able to say anything about why TPU pods change the Chapter 8 layout reasoning.** A strong answer connects the pod's larger fast-interconnect domain directly back to where the tensor-parallel-degree ceiling would sit, rather than treating TPU and GPU parallelism strategy as identical in every practical respect.
- **Overclaiming deep TPU hardware expertise you don't have.** For most interviews, the conceptual fluency in this chapter (systolic arrays exist and why, pods have a larger fast-interconnect domain, the core parallelism framework still applies) is the appropriate depth — you're not expected to reproduce exact TPU chip specifications from memory.

---

## 13.7 L5-vs-L6 differentiating talking points

- **L5 bar**: knows TPUs are Google's custom deep-learning-specific chips and that they're organized into pods with a fast interconnect.
- **L6 bar**:
  - Can explain the systolic array idea at a genuine mechanical level (data flowing rhythmically between neighboring multiply-add units) rather than just naming the term.
  - Explicitly reasons about how a TPU pod's larger fast-interconnect domain would shift a Chapter 8-style layout calculation, rather than treating GPU and TPU parallelism design as identical.
  - Correctly calibrates confidence — clearly stating the conceptual framework transfers directly while being upfront that exact TPU specifications aren't necessary to reproduce from memory, showing good judgment about where genuine expertise is expected versus where conceptual fluency suffices.

---

## 13.8 Comprehension checks

1. In your own words, what is a systolic array, and why does it suit matrix-multiplication-heavy workloads particularly well?
2. What's the key structural difference between how GPUs and TPUs are organized internally?
3. Why does a TPU pod's interconnect topology potentially change the tensor-parallel-degree ceiling compared to a typical GPU server?
4. What parts of the parallelism framework from Chapters 2–9 stay exactly the same when moving from GPU to TPU, and what specifically changes?
5. Why might TPU-specific fluency matter somewhat more for a Google interview than an Apple one, based on this chapter?

---

*This closes out Phase 4 (Hardware Realities). Next: Chapter 14 — Checkpointing Strategies, opening Phase 5 with the operational realities of running training jobs that last days to weeks — starting with why checkpointing is harder than "just save the weights" at scale.*
