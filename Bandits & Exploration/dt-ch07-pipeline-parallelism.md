# Chapter 7 — Pipeline Parallelism

*(Plain language first, with a worked timeline example to make the "bubble" problem and its fix concrete.)*

---

## 7.1 Recap: what pipeline parallelism is, in one sentence

From Chapter 2: pipeline parallelism splits the model's **layers** across devices — Device 1 holds layers 1–10, Device 2 holds layers 11–20, and so on — like an assembly line, where each station does a different, complete task on the (partially processed) item as it moves through.

---

## 7.2 The naive version, and its big problem: the bubble

**The naive setup**: send one batch of data through the pipeline. Device 1 processes it through layers 1–10, then hands its output to Device 2, which processes layers 11–20, then hands off to Device 3, and so on — a full forward pass, then a full backward pass flowing back the other way.

**Here's the problem, made concrete with a small timeline.** Say we have 4 devices in our pipeline (Device 1 through Device 4), and processing one batch through one device's chunk of layers takes 1 unit of time.

| Time step | Device 1 | Device 2 | Device 3 | Device 4 |
|---|---|---|---|---|
| 1 | Working (layers 1-10) | idle | idle | idle |
| 2 | idle | Working (layers 11-20) | idle | idle |
| 3 | idle | idle | Working (layers 21-30) | idle |
| 4 | idle | idle | idle | Working (layers 31-40) |

**Look at how much idle time there is.** At time step 1, only Device 1 is doing anything — Devices 2, 3, and 4 are sitting completely idle, waiting for data that hasn't reached them yet. By time step 4, only Device 4 is working — Devices 1, 2, and 3 are now idle, having already finished their part for this one batch and having nothing else to do yet. **Out of 16 total device-time-slots in this table (4 devices × 4 time steps), only 4 are actually doing useful work — a 75% idle rate!** This wasted idle time is called the **pipeline bubble**, and it's the single biggest problem with the naive version of pipeline parallelism — you've split the model successfully (fixing the memory wall), but you've introduced enormous, wasted idle time doing it (badly hurting the speed side of things).

---

## 7.3 The fix: micro-batching (GPipe's approach)

**The idea, in plain words**: instead of sending one *big* batch through the pipeline all at once, split it into several smaller **micro-batches**, and feed them into the pipeline **one right after another**, so that by the time Device 1 finishes micro-batch 1 and hands it to Device 2, Device 1 can immediately start working on micro-batch 2 — instead of sitting idle waiting for the whole first batch to finish its entire journey through the pipeline.

### The same timeline, now with micro-batching (4 micro-batches, labeled a, b, c, d)

| Time step | Device 1 | Device 2 | Device 3 | Device 4 |
|---|---|---|---|---|
| 1 | a | idle | idle | idle |
| 2 | b | a | idle | idle |
| 3 | c | b | a | idle |
| 4 | d | c | b | a |
| 5 | idle | d | c | b |
| 6 | idle | idle | d | c |
| 7 | idle | idle | idle | d |

**Notice the difference immediately**: from time step 4 onward, **every single device is busy at the same time** — this is the "filled" middle section of the pipeline, where all the idle-time waste from the naive version has been eliminated. The bubble hasn't disappeared entirely — there's still unavoidable idle time at the very start (the "ramp-up," time steps 1-3, where later devices are still waiting for the first micro-batch to arrive) and at the very end (the "ramp-down," time steps 5-7, where earlier devices have run out of new micro-batches to process) — but the **middle** of the pipeline, once it's "full," achieves genuinely parallel, non-idle utilization across all devices.

**Why more micro-batches make the bubble proportionally smaller**: the ramp-up and ramp-down periods are a **fixed cost**, roughly proportional to the number of devices (more devices in the pipeline means a longer ramp-up/ramp-down) — but they don't grow if you add *more* micro-batches. So the more micro-batches you split your batch into, the smaller the bubble's *proportion* of the total time becomes, since you're spreading that same fixed ramp-up/ramp-down cost over more total useful working time. **This is exactly analogous to Chapter 3's gradient accumulation** — splitting a batch into smaller pieces processed sequentially — just now the reason for splitting is to fill a pipeline rather than to fit memory.

---

## 7.4 The real tradeoff: more micro-batches helps the bubble, but costs memory

Here's the catch that keeps this from being a free lunch: **each in-flight micro-batch needs its own activations stored** (remember, activations need to be kept around for the backward pass, per Chapter 1, Section 1.4) — and with more micro-batches simultaneously "in flight" in the pipeline at once (as the timeline table shows — at time step 4, micro-batches a, b, c, and d are ALL simultaneously somewhere in the pipeline), you need memory to hold **all of their** activations at once, not just one batch's worth.

**The core tradeoff, stated plainly**: more micro-batches → smaller proportional bubble (better speed) → but more simultaneously-stored activations → more memory used. This is a genuine, real tension that system designers have to balance — not a "more is always better" situation.

---

## 7.5 PipeDream and interleaved (1F1B) scheduling — reducing the bubble further

GPipe's schedule (Section 7.3) does **all the forward passes for all micro-batches first, then all the backward passes** — meaning activations for *every* in-flight micro-batch need to be held in memory simultaneously until their corresponding backward pass finally happens.

**PipeDream's key idea: interleave forward and backward passes**, often called **1F1B scheduling** ("one-forward-one-backward") — as soon as a micro-batch's backward pass becomes available to run, prioritize running it (freeing up its activation memory) rather than continuing to push more forward passes first. This means, at any given moment, **fewer micro-batches' activations need to be simultaneously held in memory**, compared to GPipe's "all forwards, then all backwards" approach — directly attacking the memory side of the Section 7.4 tradeoff, while achieving similar (or, with further refinements, even better) bubble reduction.

**The simple summary worth remembering**: *"GPipe reduces the bubble with micro-batching, but holds all micro-batches' activations simultaneously until the backward pass sweep begins. PipeDream's 1F1B scheduling interleaves forward and backward passes so activation memory gets freed sooner, reducing the memory cost of pipelining without sacrificing much of the bubble-reduction benefit."*

---

## 7.6 Production considerations

- **The number of micro-batches is a real, tunable hyperparameter** in practice — too few, and the bubble wastes a lot of time; too many, and activation memory becomes the binding constraint (Section 7.4) — real systems tune this based on the specific pipeline depth (number of pipeline stages) and available memory per device.
- **1F1B-style interleaved scheduling (Section 7.5) is standard in modern large-scale training frameworks** (Megatron-LM's pipeline parallelism implementation, for instance, uses interleaved scheduling by default in many configurations) — knowing this name and roughly what problem it solves is a genuinely useful, current, real-systems detail.
- **Pipeline parallelism's bubble is a form of pure wasted compute time** — unlike, say, activation checkpointing's compute-for-memory tradeoff (a deliberate, useful tradeoff), the pipeline bubble is closer to pure overhead that good scheduling tries to minimize, not a tradeoff you'd ever want more of.

---

## 7.7 Interview traps

- **Not being able to explain, concretely, why the naive pipeline approach wastes so much time** — a strong answer should be able to sketch or describe a timeline table like Section 7.2's, showing the idle "diagonal" pattern explicitly, not just assert "there's a bubble."
- **Thinking more micro-batches is a strictly free improvement.** As shown in Section 7.4, more micro-batches directly costs more simultaneously-held activation memory — this is a genuine tradeoff, not a one-directional win.
- **Confusing GPipe's "all forwards then all backwards" schedule with PipeDream's interleaved 1F1B schedule** — these are genuinely different scheduling strategies with different memory implications, and conflating them misses the specific improvement 1F1B provides.

---

## 7.8 L5-vs-L6 differentiating talking points

- **L5 bar**: correctly describes the pipeline bubble problem and knows micro-batching is the standard fix.
- **L6 bar**:
  - Can sketch or describe the actual timeline table (Section 7.2 vs. 7.3), showing the idle-slot pattern concretely and explaining why the "filled middle" of the pipeline achieves full utilization while ramp-up/ramp-down don't.
  - Explicitly connects the "more micro-batches shrinks the bubble's proportion" reasoning to the fixed-vs-scaling cost structure (ramp-up/down is fixed, useful work scales with micro-batch count) rather than just asserting "more micro-batches is better."
  - Names PipeDream/1F1B scheduling specifically as the fix for GPipe's activation-memory cost, and can state the tradeoff precisely: less simultaneous activation memory, achieved by interleaving backward passes sooner rather than batching all forwards first.

---

## 7.9 Comprehension checks

1. Using the timeline table from Section 7.2, explain in your own words why the naive pipeline approach has so much idle time.
2. How does micro-batching (Section 7.3) reduce this idle time, and why does the "bubble" never disappear completely (ramp-up/ramp-down)?
3. Why does adding more micro-batches make the bubble's *proportion* of total time smaller, even though the ramp-up/ramp-down cost itself doesn't shrink?
4. What's the real cost of using more micro-batches, and why does this create a genuine tradeoff rather than a free improvement?
5. In one sentence, what does PipeDream's 1F1B scheduling change compared to GPipe, and what problem does that change solve?

---

*Next: Chapter 8 — Combining Strategies: 3D Parallelism, where we bring data, tensor, and pipeline parallelism together and work through how you'd actually lay out a large training job across hundreds of GPUs.*
