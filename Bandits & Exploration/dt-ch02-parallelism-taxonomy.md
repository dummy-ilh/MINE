# Chapter 2 — The Parallelism Taxonomy

*(Same plain-language style as Chapter 1.)*

---

## 2.1 The one-sentence idea

There are exactly three basic things you can split across multiple devices: **the data, the layers, or the inside of a single layer.** Everything else in Phases 1–2 is just a detailed version of one of these three ideas (or a combination of them).

---

## 2.2 Split #1: Data Parallelism — "everyone has the whole model, different data"

**The idea, in plain words**: give every device a **complete copy** of the entire model. Each device processes a **different slice** of the current batch of data, computes its own gradients, and then all the devices **combine (average) their gradients** before updating — so every copy of the model stays identical after each step.

**Simple analogy**: imagine 8 people, each with an identical copy of the same textbook, each independently grading a different stack of 100 student essays using the same rubric, then averaging their scores together at the end. Everyone has the same "model" (the rubric), just different data (different essays) to process.

**What this fixes**: Wall #2 from Chapter 1 (training would take too long) — you're processing more data per unit time, since multiple devices work simultaneously.

**What this does NOT fix**: Wall #1 (the model doesn't fit) — every single device still needs to hold the *entire* model, gradients, and optimizer state. If the model itself is too big for one GPU, data parallelism alone doesn't help at all — this is exactly why Chapter 9 (ZeRO/FSDP) exists: it's a clever way to get some of model-parallelism's memory benefits while keeping data-parallelism's simplicity.

---

## 2.3 Split #2: Model Parallelism — "split the model itself across devices"

This comes in two genuinely different flavors, worth telling apart clearly from the very start:

**Pipeline parallelism — split *between* layers**: Device 1 holds layers 1–10, Device 2 holds layers 11–20, Device 3 holds layers 21–30, and so on. Data flows through the devices in sequence, like an assembly line — Device 1 finishes its chunk of layers and passes the result to Device 2, and so on.

**Simple analogy**: an assembly line building a car — station 1 attaches the frame, station 2 attaches the engine, station 3 paints it. Each station does a *different, complete* task, and the (partially-built) car moves between stations.

**Tensor parallelism — split *within* a single layer**: instead of giving different devices different *whole* layers, you split one giant matrix multiplication (say, inside one single layer) so that each device computes only *part* of that one operation, and the results get stitched back together.

**Simple analogy**: instead of an assembly line (different stations doing different tasks), imagine 4 people jointly lifting one single, very heavy object together — each person handles one corner of the *same* object, at the *same* time, not different objects in sequence.

**What both flavors fix**: Wall #1 (the model doesn't fit) — no single device needs to hold the entire model; each holds only its assigned piece.

**What both flavors cost**: devices now need to communicate with each other *during* a single forward/backward pass (not just once per step, the way data parallelism does) — this is a meaningfully different, and often more expensive, communication pattern, which is exactly why Chapter 1's interconnect hierarchy (Section 1.7) matters so much for these two strategies specifically.

---

## 2.4 A simple table to keep the three ideas straight

| Strategy | What gets split | What's identical across devices | What each device needs to fit | Fixes which wall? |
|---|---|---|---|---|
| Data parallelism | The batch of data | The entire model | The **entire** model | Speed (Wall #2) |
| Pipeline parallelism | The layers (between layers) | Nothing — each device has different layers | Only its assigned **layers** | Memory (Wall #1) |
| Tensor parallelism | Inside one layer's math | Nothing — each device has a slice of the same layer | Only its **slice** of each layer | Memory (Wall #1) |

This table is worth being able to reproduce from memory — it's the map for the entire rest of Phases 1–2.

---

## 2.5 Why real systems combine all three ("3D parallelism," previewed)

Here's the natural question this taxonomy raises immediately: **if pipeline and tensor parallelism fix the memory wall, why do we still need data parallelism at all?**

Answer: because pipeline and tensor parallelism only let you fit a *bigger model* — they don't, by themselves, make training faster in the sense of processing more data per unit time (Wall #2 is still there). So the real answer, used by essentially every frontier large-model training run, is: **use tensor + pipeline parallelism to make the model fit across a set of devices, and then use data parallelism on top, running many identical copies of that whole tensor+pipeline setup side by side, to get through the data fast enough.** This combination is called **3D parallelism**, and it's the subject of Chapter 8, once Chapters 3–7 have built up each individual piece properly.

---

## 2.6 Production considerations

- **Data parallelism is almost always the "default" first choice** whenever a model actually fits on one device — it's the simplest to implement, the most well-supported by frameworks (PyTorch's DistributedDataParallel, for instance), and only becomes insufficient once the model itself is too large for a single device's memory.
- **Model parallelism (either flavor) is reached for specifically because the model doesn't fit** — it's a more complex, more communication-heavy strategy, and real systems avoid it unless memory constraints force the issue.
- **The "which combination should I use" decision is fundamentally driven by two numbers**: how big is the model (relative to one device's memory) and how much data/compute do you need to get through (relative to one device's speed) — this is the same "diagnose the actual bottleneck first" instinct from Chapter 1's compute-bound vs. memory-bandwidth-bound distinction, just one level up, applied to whole-system design instead of single-operation performance.

---

## 2.7 Interview traps

- **Confusing pipeline parallelism with tensor parallelism** — these are genuinely different ("different layers on different devices" vs. "one layer's math split across devices"), and conflating them is one of the most common, most easily-avoided mistakes in this whole topic area. The assembly-line vs. lifting-one-heavy-object analogies from Section 2.3 are worth keeping handy specifically to avoid this mix-up under interview pressure.
- **Thinking data parallelism alone can solve a "model doesn't fit" problem.** It cannot — every device still needs the whole model. This is a very common, very checkable misunderstanding.
- **Not being able to explain, from first principles, *why* real systems combine all three strategies** rather than picking just one — Section 2.5's reasoning (model parallelism fixes memory, data parallelism fixes speed, you often need both) should be produced fluently, not just asserted as "it's common practice."

---

## 2.8 L5-vs-L6 differentiating talking points

- **L5 bar**: can correctly define all three strategies and correctly fill in the comparison table from Section 2.4.
- **L6 bar**:
  - Can explain, unprompted, *why* combining strategies is necessary (Section 2.5) rather than just naming "3D parallelism" as a buzzword.
  - Uses the "which wall does this fix" framing (memory vs. speed) as a genuine diagnostic tool when given a hypothetical scenario — e.g., correctly reasoning that a scenario described as "the model itself barely fits on one GPU, but we have plenty of GPUs and time" calls for model parallelism first, data parallelism second (or not at all), rather than reflexively reaching for whichever strategy is most familiar.
  - Explicitly connects the tensor-parallelism communication cost back to Chapter 1's interconnect hierarchy, correctly predicting that tensor parallelism will want to stay within a single fast-interconnect node — before this is formally stated in Chapter 8.

---

## 2.9 Comprehension checks

1. In one sentence each, what gets split in data parallelism, pipeline parallelism, and tensor parallelism?
2. Why doesn't data parallelism help when the model itself is too big to fit on one device?
3. Using the assembly-line vs. lifting-one-object analogies, explain the difference between pipeline and tensor parallelism in your own words.
4. Why do real large-scale training systems typically combine all three strategies rather than picking just one?
5. If someone told you "our model fits comfortably on one GPU, but we have a huge amount of data and want to train faster," which strategy would you reach for first, and why?

---

*Next: Chapter 3 — Data Parallelism and Gradient Accumulation, opening Phase 1 with the mechanics of the most common and most interview-tested parallelism strategy.*
