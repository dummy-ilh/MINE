# Chapter 6 — Latency vs. Accuracy Tradeoffs

*(Module 6 of the syllabus)*

---

## 1. Why this tradeoff exists at all

Up to now, we've mostly talked about *correctness* — is the model right, is it deployed safely, is it seeing the right data. This chapter introduces a second axis that's just as important in production: **speed**. A model that's more accurate but too slow to return an answer within the required time budget can be worse, in a real system, than a slightly less accurate model that responds instantly.

The core tension: **generally, the techniques that make a model more accurate (bigger models, more computation, ensembling multiple models) also make it slower and more expensive to run.** Production systems almost always operate under a hard or soft latency budget (an SLA — service level agreement), so you're constantly trading some accuracy for speed, or vice versa, rather than just chasing the highest possible accuracy in isolation.

---

## 2. Why latency is a first-class constraint, not an afterthought

Three distinct reasons latency matters, each worth being able to name separately:

- **SLAs (service level agreements)** — many production systems have contractual or product-defined limits on response time (e.g., "must respond within 100ms"). Exceeding this isn't just "a bit slow," it can be an outright system failure, regardless of how accurate the answer was.
- **User experience** — for anything user-facing (search, recommendations, autocomplete), delay is directly felt by the user and measurably affects engagement, even if the eventual answer is high quality.
- **Cost** — slower models generally mean more compute time per request, which at scale (millions/billions of requests) directly translates to real infrastructure cost. A model that's 1% more accurate but 5x more expensive to run may not be worth it, and interviewers want to see you weigh this explicitly rather than assume "more accurate = better" unconditionally.

**The framing to always lead with in an interview:** before answering *any* latency-vs-accuracy question, state the product constraint first. A fraud-detection model blocking a real-time transaction has a completely different latency budget than a model doing overnight batch scoring of all customers for a marketing campaign. The "right" tradeoff only makes sense relative to a stated use case — jumping straight to "use quantization" without first asking or stating the latency requirement is a weak answer.

---

## 3. Model compression techniques

These are techniques that reduce a model's size and/or computation cost, generally at some (hopefully small) accuracy cost.

### Quantization
**What it is:** reducing the numerical precision used to store and compute the model's weights — for example, converting weights from 32-bit floating point numbers down to 8-bit integers.

**Why it helps:** lower-precision numbers take less memory to store and less compute to process, so inference becomes faster and the model becomes smaller (important for memory-constrained environments like mobile devices, which is directly relevant if you're interviewing at Apple, given how much on-device inference matters there).

**The tradeoff:** you lose some numerical precision, which can slightly reduce accuracy — the model's weights are now coarser approximations of what was learned during full-precision training. In practice, well-executed quantization often costs surprisingly little accuracy relative to the speed/size gains, which is exactly why it's so widely used, but it's not free and needs to be validated, not assumed safe.

### Pruning
**What it is:** removing weights (or entire neurons/connections) from the trained model that contribute little to its output — essentially, cutting out the parts of the network that turned out to matter least.

**Why it helps:** a smaller network (fewer weights to compute through) means faster inference and a smaller memory footprint.

**The tradeoff:** prune too aggressively and you start removing weights that *do* matter, degrading accuracy. The general theme in this whole chapter: nearly every efficiency technique is really "find how much you can cut before accuracy meaningfully suffers," not a free lunch.

### Distillation
**What it is:** train a smaller "student" model to mimic the behavior of a larger, more accurate "teacher" model — rather than training the small model from scratch on the original labels alone. The student learns from the teacher's outputs (which carry richer information than just the raw correct answer, since they reflect the teacher's learned confidence/uncertainty across all possible answers, not just the single correct one).

**Why it helps:** you end up with a small, fast model that has absorbed much of the larger model's learned behavior — often significantly more accurate than a same-sized model trained from scratch on raw labels alone.

**The tradeoff:** you still generally can't fully match the teacher's accuracy with a meaningfully smaller student — you're compressing, and some information/capability is lost in that compression. Also worth noting: distillation requires you to already *have* a strong teacher model and the compute budget to train the student against it, so it's a heavier upfront investment than quantization or pruning.

**Quick way to keep these three straight:** quantization = same architecture, coarser numbers. Pruning = same architecture (mostly), fewer connections. Distillation = an entirely different, smaller architecture, taught to imitate a bigger one.

---

## 4. Batching — throughput vs. per-request latency

We touched on this in Chapter 3 (dynamic batching in model servers) — here's the fuller framing.

**The core tradeoff:** running the model once per individual incoming request is often computationally inefficient, especially on GPU hardware, which is designed to do a lot of parallel computation at once. Grouping several requests into a single batch and running them through the model together dramatically improves **throughput** (total requests processed per second, system-wide) — but it does so by adding a small delay to any individual request, since the system has to wait briefly to accumulate a batch before running it.

**Framed as a direct tradeoff:** larger batch sizes → better throughput, worse per-request latency (each request waits longer to be batched, and waits for the whole batch to finish). Smaller batch sizes (or no batching) → worse throughput, better per-request latency.

**This is a genuinely tunable knob**, not a one-time architectural decision — batch size, and the maximum wait time before running a partially-filled batch, are parameters you actively tune based on your specific SLA and traffic volume.

---

## 5. Caching predictions

**What it is:** instead of always running the model fresh for every request, store (cache) previous predictions and, when a repeat or sufficiently-similar request arrives, return the cached prediction instead of recomputing it.

**When this genuinely helps:** works well when a meaningful fraction of requests are exact repeats or fall into a small set of common cases (e.g., predicting on a fixed catalog of products, where the same product gets scored repeatedly).

**When it doesn't help / can actively hurt:** if inputs are highly unique per request (e.g., personalized predictions for individual users based on real-time context), caching offers little benefit since cache hits are rare — and worse, a stale cached prediction can silently serve *outdated* results, reintroducing a version of the "world changed but nobody noticed" problem from Chapter 4, just at the caching layer instead of the training layer. Always pair caching with a sensible expiration/invalidation policy if you propose it.

---

## 6. Hardware tradeoffs

- **CPU** — cheaper per unit, widely available, generally fine for smaller models or lower request volumes; not efficient for large, computation-heavy models at scale.
- **GPU** — excels at the kind of massively parallel computation deep learning models require; the natural choice once model size/traffic makes CPU inference too slow or too expensive per request, but GPUs cost significantly more and are a scarcer resource to provision.
- **Specialized accelerators** (e.g., hardware built specifically for ML inference) — can offer better efficiency than general-purpose GPUs for specific, well-matched workloads, at the cost of being less flexible and often requiring more specialized deployment tooling.

**The interview-safe framing:** "it depends on model size and traffic volume" is the honest answer, but don't stop there — name the actual variables that decide it (request volume, model size/complexity, latency budget, cost budget) so the interviewer sees you reasoning about the decision, not just naming the three hardware options.

---

## 7. Common pitfall interviewers listen for

**"Just use the biggest GPU"** as a default answer to a latency problem is a red flag response — it ignores cost entirely, and it skips past all the cheaper, often more appropriate techniques above (quantization, pruning, distillation, batching tuning) that should typically be considered *before* reaching for more expensive hardware. A strong answer works through the cheaper software-level levers first, and only escalates to bigger/different hardware once those are exhausted or clearly insufficient for the stated constraint.

---

## Comprehension check

1. In your own words, why does nearly every model compression technique in this chapter involve *some* accuracy cost — what's the underlying reason none of them are free?
2. Explain the batching tradeoff in your own words: why does improving throughput generally come at the cost of individual-request latency?
3. You're asked: "how would you serve a recommendation model that needs sub-50ms latency at very high request volume, but accuracy is currently mediocre and you have budget to improve it?" Sketch, in a few sentences, what factors and techniques from this chapter you'd bring into that answer, and in what order you'd consider them.

Say "c7" when ready for **Chapter 7: Monitoring & Observability** (data drift vs. concept drift lives here).
