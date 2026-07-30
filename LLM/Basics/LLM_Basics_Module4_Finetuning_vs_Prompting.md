# Module 4 — Fine-tuning vs Prompting vs In-Context Learning (Master Notes, Expanded)

> **Note on this version:** This file preserves 100% of your original notes, in their original order and wording. Every addition is clearly tagged with one of: `📌 Added Explanation`, `🧮 Numerical Example`, `❓ Interview Q&A`, or `🔎 Accuracy Flag`. Nothing original was deleted or shortened.

## 0. The big picture — three ways to adapt a pretrained model to a task

A pretrained LLM knows "language + world knowledge" broadly, but you usually want it to do something specific (classify support tickets, follow instructions, answer in a certain style). There are three fundamentally different ways to get there, differing in **whether you touch the model's weights at all**:

| Approach | Touches weights? | Cost | Persistence |
|---|---|---|---|
| Prompting / in-context learning | No | Cheap, instant, per-request | Not learned — must repeat every time |
| Full fine-tuning | Yes, all weights | Expensive (full backprop through all params) | Permanently baked into the model |
| Parameter-efficient fine-tuning (LoRA etc.) | Yes, small added subset | Cheap-ish (small fraction of full fine-tune cost) | Permanently baked into a small add-on |

### 📌 Added Explanation — "in simple terms" framing for Section 0

Think of the pretrained model as a very well-read employee who has read the entire internet but has never worked at *your* company. You have three ways to get them productive on a specific task:

- **Prompting** = handing them a sticky note with instructions every single time they do the task. Fast, free, but you have to hand them the note again for the next task — they don't remember it.
- **Full fine-tuning** = sending them through a full retraining program that touches literally everything they know, updating their entire brain. Extremely thorough, extremely expensive, and you'd need a separate "clone" of the employee for every different task you retrain them on.
- **PEFT / LoRA** = giving them a small removable notebook of task-specific notes that clips onto their existing brain. Cheap to produce, swappable, and you only need one "base employee" plus many small notebooks.

The key axis distinguishing all three is **whether the model's weights themselves change**, because that determines cost, persistence, and how many "copies" of the model you end up needing.

---

## 1. Zero-shot / Few-shot Prompting and In-Context Learning (ICL)

### Core idea, in plain words
Just describe the task (zero-shot) or show a few examples (few-shot) **directly in the input text**, at inference time, with no weight updates at all. The model "figures out" the task pattern purely from the prompt.

### 📌 Added Explanation — why this is even possible

It seems almost magical that a model with completely frozen weights can "learn" a brand-new task just by reading a few examples in its input. The resolution is that the model isn't learning in the traditional sense (no parameter changes happen) — it is doing **pattern recognition over its own input sequence**, using capabilities it already acquired during pretraining. The few-shot examples don't teach the model new skills; they tell the *already-capable* model which skill, among the huge repertoire it learned during pretraining, to activate right now. This distinction ("selecting a latent skill" vs. "learning a new skill") is the crux of why ICL works and is worth saying explicitly in an interview.

### Numerical example
**Zero-shot** prompt: `"Classify sentiment: 'This movie was fantastic!' → "` — model must infer what "sentiment" means and the expected output format from just the instruction.

**Few-shot (3-shot)** prompt:
```
Review: "Terrible film, waste of time." → Negative
Review: "Loved every minute." → Positive
Review: "Mediocre, forgettable." → Neutral
Review: "This movie was fantastic!" → 
```
The model has now seen 3 input→output examples directly in-context and typically completes with "Positive" — despite **no gradient update ever happening**. This is In-Context Learning: the model adapts its behavior purely by conditioning on the examples present in its input window, using the same frozen weights it always has.

### 🧮 Numerical Example — walking through the few-shot prediction step by step

Let's make the mechanism concrete instead of just asserting "the model predicts Positive."

1. **Tokenize the whole prompt** (all 3 examples + the query) into a single sequence of tokens — say this comes out to roughly 45 tokens total (the exact count depends on the tokenizer, but the point is it's *one* sequence, not 4 separate calls).
2. **Single forward pass**: the entire 45-token sequence is passed through the Transformer *once*. At every layer, self-attention lets the token positions corresponding to `"This movie was fantastic!" →` attend back over all previous tokens — including the three `Review: ... → Label` pairs.
3. **Pattern extraction via attention**: attention heads (including induction heads, explained just below) notice the repeating template `Review: "..." → Label` and, specifically, notice that the current query token position looks structurally identical to the start of the three previous "Review:" blocks.
4. **Next-token prediction**: the final hidden state at the last position is projected through the vocabulary head (the same frozen softmax output layer used for ordinary next-token prediction) to produce a probability distribution over the vocabulary. Because "fantastic" is semantically close to "Loved every minute" (mapped to Positive) and far from "Terrible"/"Mediocre" (mapped to Negative/Neutral), the token "Positive" ends up with the highest probability — say, hypothetically, `P(Positive) = 0.82`, `P(Negative) = 0.05`, `P(Neutral) = 0.10`, with the remainder spread over other vocabulary tokens.
5. **Decode**: greedy decoding (or sampling) picks "Positive" as the output.

The crucial point demonstrated by this walkthrough: **all of this happens in one forward pass with fixed weights** — steps 2–4 are just attention and matrix multiplications with the same `W_Q, W_K, W_V, W_O` the model always has. Nothing is "trained" during this process.

### Why ICL works at all — the mechanistic explanations (interview-level, cite both)
1. **Induction heads** (Anthropic's interpretability research finding): certain attention heads, discovered via mechanistic interpretability, specifically implement a "if I've seen pattern [A][B] before, and I now see [A] again, predict [B]" copying behavior. Few-shot examples give the model exactly this kind of repeatable pattern to latch onto — literally attending back to earlier occurrences of a similar structure and copying the completion pattern forward.
2. **Meta-gradient / implicit fine-tuning view**: some theoretical work argues that the forward pass through a Transformer, when given in-context examples, is mathematically analogous to performing an implicit gradient-descent-like update using the in-context examples as if they were training data — the attention mechanism's output can be shown (under simplifying assumptions) to resemble a gradient step, essentially "fine-tuning within the forward pass," without ever touching stored weights.

### 📌 Added Explanation — unpacking "induction heads" with a toy mechanism

An induction head operates, informally, in two steps across two different attention heads working together (sometimes called a "previous-token head" + "induction head" circuit):

- Step 1 (a "previous-token head"): at each position, attend to the *previous* token and copy information about it forward.
- Step 2 (the induction head itself): look back through the sequence for a position where the *current* token previously appeared, then attend to *whatever token came right after that previous occurrence*, and boost the probability of predicting that same token now.

**In simple terms**: "I've seen the pattern `[A][B]` earlier in this text. I'm looking at `[A]` again right now. So my best guess for what comes next is `[B]`." Applied to few-shot prompting, `[A]` = "Review: ... →" and `[B]` = the label that followed it last time, so the pattern repeats naturally.

### 📌 Added Explanation — unpacking the "meta-gradient" / implicit fine-tuning view, with the intuition behind the math

The claim sounds exotic, so here is the intuition without the full derivation (the original papers, e.g. Dai et al. 2022/von Oswald et al. 2022, work through the linear-attention special case in detail):

- In ordinary gradient descent fine-tuning, you'd compute a gradient of the loss with respect to a weight matrix `W` using your labeled examples, and update `W ← W - η∇L`.
- The "meta-gradient" argument shows that, for a simplified (linear) attention mechanism, the *output* of attending over in-context example tokens can be written in a form that looks algebraically identical to "the original weight matrix `W`, plus a correction term built out of the in-context examples" — i.e., structurally the same shape as `W + ΔW` from an implicit fine-tuning step, even though no actual gradient was computed or applied to `W`.
- The correction term is constructed purely from attention operations over the in-context tokens (keys/values built from the example tokens), so it happens entirely within the forward pass.

**In simple terms**: "It's *as if* the model quietly did one step of fine-tuning using your few-shot examples as training data — except that 'fine-tuning' happened transiently, inside the computation for this one prompt, and vanishes the moment the prompt is gone. No stored weight actually changed."

### 🔎 Accuracy Flag
The "meta-gradient"/implicit-fine-tuning equivalence has been rigorously shown only under simplifying assumptions (e.g., linear or linearized attention, specific loss functions) — it is a compelling theoretical *lens*, not a proven universal mechanism for how every real, nonlinear, multi-layer Transformer performs ICL in practice. In an interview, it's safest to present it as "one influential theoretical account, with induction heads as complementary empirical/mechanistic evidence" rather than as settled fact.

**Practical framing to say out loud**: "ICL isn't the model learning new facts on the fly — it's the frozen model recognizing 'oh, this is the pattern-completion task,' using in-context examples the same way induction heads use repeated sequences, or equivalently, approximating what a gradient update would have done, all within a single forward pass."

---

## 2. Full Fine-tuning vs Instruction Tuning

### Full fine-tuning
Take the pretrained model, continue training (standard backprop, all parameters updated) on a **labeled, task-specific dataset**. Example: take a base CLM-pretrained model and continue training it purely on medical-question-answer pairs to specialize it.

**Cost problem**: full fine-tuning requires storing gradients and optimizer states (e.g. Adam needs 2 extra copies of every parameter — first and second moment estimates) for **every single parameter** in the model. For a 70B-parameter model, this means memory footprint several times the base model size just for training state — completely impractical on typical hardware, and you end up with a full separate copy of a 70B-parameter model per fine-tuned task.

### 📌 Added Explanation — deriving the "several times the base model size" claim precisely

This is a very common interview follow-up ("why exactly is fine-tuning so expensive — give me the memory breakdown"), so let's derive it term by term for a model with `N` parameters, assuming mixed-precision training (a very standard real-world setup):

| Component | Precision (typical) | Bytes per parameter | Purpose |
|---|---|---|---|
| Model weights (master copy) | fp32 | 4 bytes | The parameters themselves, kept in high precision for stable updates |
| Model weights (compute copy) | fp16/bf16 | 2 bytes | Used for the actual forward/backward matmuls (faster, less memory bandwidth) |
| Gradients | fp16/bf16 | 2 bytes | `∂Loss/∂W`, one per parameter, needed for the update step |
| Adam optimizer state — first moment `m` | fp32 | 4 bytes | Exponential moving average of past gradients (momentum term) |
| Adam optimizer state — second moment `v` | fp32 | 4 bytes | Exponential moving average of past *squared* gradients (adaptive learning rate term) |

Total ≈ `4 + 2 + 2 + 4 + 4 = 16 bytes per parameter`, i.e., roughly **4x the footprint of just storing the fp32 weights alone (4 bytes/param)**, or **8x the footprint of storing fp16 weights alone (2 bytes/param)** — this is exactly where "several times the base model size" comes from. (Activations for backprop add further memory on top of this, scaling with batch size and sequence length, which is why fine-tuning large models often also requires activation checkpointing.)

### 🧮 Numerical Example — 70B model full fine-tuning memory, in GB

Using the ~16 bytes/parameter figure above:

```
N = 70,000,000,000 parameters (70B)
Memory ≈ 70B × 16 bytes = 1,120,000,000,000 bytes
        = 1,120 GB  (≈ 1.12 TB)
```

Compare that to just storing the model for *inference* (fp16, no training state): `70B × 2 bytes = 140 GB`. So full fine-tuning training-state memory (~1.12 TB) is roughly **8x** the inference-only footprint (140 GB) — and this is before counting activation memory, which scales further with batch size × sequence length × number of layers. This single comparison ("140 GB to just *run* it, over 1 TB to *fine-tune* it") is a great one-liner to have ready in an interview to justify why nobody full-fine-tunes 70B models on a single GPU, and why PEFT methods (Section 3) exist.

### Instruction tuning (FLAN, T0 — a specific *type* of full fine-tuning)
Instead of fine-tuning on one narrow task, fine-tune on a **large, diverse mixture of tasks, each phrased as a natural-language instruction** ("Summarize this text:", "Translate to French:", "Answer this question:" ...) with many different task types combined in one training set.

**The key finding (FLAN paper)**: this doesn't just make the model better at the specific tasks it was instruction-tuned on — it improves **zero-shot generalization to entirely new, unseen tasks/instructions**, because the model learns the general skill of "parse an instruction and follow it," not just memorize answers to specific task formats. This is precisely the step that turns a raw pretrained CLM model into something that behaves like "ChatGPT-style, follows your instructions" rather than just "autocomplete."

### 📌 Added Explanation — why diversity of tasks, specifically, causes generalization

It's worth being able to explain *why* mixing many task types (rather than just training on more examples of one task) is what unlocks generalization to brand-new instructions:

- If you fine-tune on only "summarization" examples, the model can very plausibly just learn "summarization" as a narrow skill, without learning anything general about *following an instruction it's never seen phrased that way before*.
- If you fine-tune on hundreds of different task types — summarization, translation, sentiment classification, question answering, etc. — all phrased through the same "instruction → output" template, the *one thing all these training examples have in common* is the meta-pattern "read the instruction text, then do what it says." Because that's the only consistent signal across such a diverse dataset, gradient descent is pushed toward learning that general meta-skill rather than memorizing any single task's surface form.
- **In simple terms**: training on one task teaches the model that task. Training on hundreds of wildly different tasks, all wrapped in "here's an instruction, follow it," is the only way for the model to notice that the *real* underlying regularity is "instructions get followed," which then transfers to instructions it has never literally seen before.

### Where instruction tuning is used standalone in practice
**FLAN-T5, T0** (academic instruction-tuning research models), and it's a core ingredient (alongside RLHF, covered in Module 5) in essentially every production assistant model (GPT-3.5/4, Claude, Llama-Chat).

---

## 3. Parameter-Efficient Fine-Tuning (PEFT)

### The motivating problem
Full fine-tuning a 70B model costs enormous memory/compute, and if you want 10 different task-specialized versions, you'd need 10 full copies of a 70B model. PEFT methods fine-tune **only a small number of new/added parameters**, while **freezing almost all of the original pretrained weights**.

### LoRA (Low-Rank Adaptation) — the one to know cold, math included

**Core idea**: instead of updating a weight matrix `W` (say, a `d × d` attention projection matrix) directly, freeze `W` entirely, and add a small trainable "delta" expressed as a **low-rank decomposition**:
```
W_new = W_frozen + ΔW,  where  ΔW = B × A
```
- `A` has shape `(r × d)`, `B` has shape `(d × r)` — where `r` (the "rank") is a small number, e.g. r=8 or r=16, **much smaller than d** (which might be 4096 or larger for a big model).
- Only `A` and `B` are trained; `W_frozen` never gets a gradient update.

### 📌 Added Explanation — deriving/explaining the LoRA equation term by term

Let's break down `W_new = W_frozen + ΔW = W_frozen + BA` completely:

- **`W_frozen` (shape `d × d`)**: the original pretrained weight matrix — e.g., a query, key, value, or output projection matrix inside a self-attention block. It is loaded from the pretrained checkpoint and its gradient is disabled (`requires_grad = False` in PyTorch terms), so it literally never moves during LoRA training.
- **`ΔW` (shape `d × d`)**: the "correction" that fine-tuning *would* have applied to `W` if you'd done full fine-tuning. LoRA's whole bet is that this correction, despite living in a `d × d`-sized matrix, doesn't need `d × d` independent degrees of freedom to represent well.
- **`A` (shape `r × d`)** and **`B` (shape `d × r`)**: two small matrices whose product `BA` reconstructs a `d × d`-shaped matrix, but by construction `BA` can only have **rank at most `r`** (rank of a matrix product is bounded by the smaller of the two factors' ranks/inner dimension). This is the "low-rank" in Low-Rank Adaptation — you are deliberately restricting `ΔW` to live in a `d²`-dimensional space but only along `r × 2d` learnable directions.
- **Why add rather than replace**: because `W_new` is a simple matrix sum, at inference time you can precompute `W_new = W_frozen + BA` once and store just `W_new` — the model's forward pass is then *identical in structure* to the original model, just with different numbers in `W`. This is exactly why LoRA adds zero inference latency (explained further below).
- **Initialization detail** (from the original paper): `B` is typically initialized to all zeros and `A` is initialized randomly (e.g., Gaussian) — this guarantees `BA = 0` at the very start of training, so `W_new = W_frozen` exactly at step 0, meaning the fine-tuned model starts out **behaviorally identical** to the base pretrained model and only diverges as training proceeds. This is a nice safety/stability property worth mentioning if asked.

### Why this drastically cuts trainable parameters — the actual numbers
Say `d = 4096` (a realistic hidden dimension) and rank `r = 8`.

**Full fine-tuning** this one weight matrix: `d × d = 4096 × 4096 = 16,777,216` trainable parameters.

**LoRA** with rank 8: `A` is `(8 × 4096) = 32,768` params, `B` is `(4096 × 8) = 32,768` params → total = `65,536` trainable parameters.

```
Reduction factor = 16,777,216 / 65,536 = 256x fewer trainable parameters, for this one matrix.
```
This is the number to have ready in an interview: **LoRA can cut trainable parameters by roughly 2-3 orders of magnitude** depending on rank choice, while empirically retaining most of full-fine-tuning's task performance for many tasks.

### 🧮 Numerical Example — extending the LoRA calculation across a whole model, and across different ranks

**(a) Scaling up to a whole model.** Suppose your model has 32 Transformer layers, and in each layer you apply LoRA to 4 projection matrices (`W_Q, W_K, W_V, W_O`), each `d × d = 4096 × 4096`.

```
Matrices touched by LoRA = 32 layers × 4 matrices = 128 matrices
Trainable params per matrix (r=8) = 65,536  (from above)
Total LoRA trainable params = 128 × 65,536 = 8,388,608 ≈ 8.4M parameters
```

If the full model has, say, 7B parameters total, then LoRA is training:
```
8.4M / 7,000M ≈ 0.12% of the total model's parameters
```
— i.e., you are updating roughly **one-tenth of one percent** of the model to adapt it to a new task, while 99.88% of the weights stay completely frozen.

**(b) Effect of changing the rank `r`.** Holding `d = 4096` fixed, trainable params per matrix `= 2 × r × d`:

| Rank `r` | Trainable params per matrix (`2 × r × d`) | Reduction vs full FT (16,777,216) |
|---|---|---|
| 4 | 2 × 4 × 4096 = 32,768 | ≈ 512x |
| 8 | 2 × 8 × 4096 = 65,536 | ≈ 256x |
| 16 | 2 × 16 × 4096 = 131,072 | ≈ 128x |
| 64 | 2 × 64 × 4096 = 524,288 | ≈ 32x |

**In simple terms**: doubling the rank exactly doubles the trainable parameter count for a given matrix (the relationship is linear in `r`), so rank is a direct, simple dial for trading off "how expressive is my adapter" against "how many parameters am I training" — small `r` (4–16) is enough for most practical tasks, which is why LoRA is so cheap in practice.

### Why low-rank works at all (the intuition, not just the mechanics)
The hypothesis behind LoRA (supported empirically in the paper) is that the *change* a model needs during task-adaptation (the difference between pretrained weights and ideally fine-tuned weights) has a naturally **low "intrinsic rank"** — i.e., the useful update direction lives in a small subspace, even though the full weight matrix is huge. You don't need to move every one of 16M+ directions; you need to move a much smaller number of the *right* directions, and `B×A` (rank r) is exactly a compact way to parameterize "a rank-r update in the full d×d space."

### 📌 Added Explanation — an analogy for "low intrinsic rank"

Imagine `W_frozen` as a giant, extremely detailed 3D terrain map (millions of independently-adjustable elevation points). Full fine-tuning lets you individually re-sculpt every single elevation point. The "low intrinsic rank" hypothesis says: *for most task-adaptation purposes, you don't actually need to independently reshape every point — a much simpler transformation, like "tilt this whole region a bit, and stretch that whole region a bit" (a handful of broad, sweeping adjustments), captures almost all of the useful change.* A rank-`r` matrix `BA` is mathematically exactly this kind of "few broad sweeping adjustments" object — it can only express changes built out of `r` independent directions, no matter how big `d` is, which is precisely why it's compact yet still effective for many tasks.

### Practical deployment benefit (a favorite follow-up question)
Because `ΔW = BA` is small, you can store many different task-specific `(A, B)` pairs (a few MB each) alongside a single shared frozen base model (many GB), and **swap adapters at inference time** without duplicating the base model — this is why LoRA became the dominant way to serve many customized model variants cheaply.

### 🧮 Numerical Example — adapter storage size vs. a full model copy

Using the earlier 8.4M-parameter LoRA adapter (whole 7B model, rank 8) at fp16 (2 bytes/param):

```
Adapter size ≈ 8.4M × 2 bytes = 16.8 MB
```
Compare to storing a *second full copy* of the 7B model for a second task, at fp16:
```
Full model copy ≈ 7,000M × 2 bytes = 14,000 MB ≈ 14 GB
```
So one LoRA adapter (~17 MB) is roughly **1/800th the size** of a full duplicated model copy (~14 GB). If you need 50 different task-specialized variants, storing 50 LoRA adapters costs `50 × 17 MB ≈ 850 MB` total on top of one shared 14 GB base model, versus `50 × 14 GB = 700 GB` for 50 full model copies — this is the concrete number behind "LoRA is the dominant way to serve many customized variants cheaply."

### QLoRA — the natural follow-up extension
QLoRA combines LoRA with **quantizing the frozen base model** (typically to 4-bit precision, using a technique called NF4 — "4-bit NormalFloat," designed to match the actual distribution of pretrained weights better than naive uniform 4-bit quantization) — so the large frozen weights sit in GPU memory at 4-bit precision (huge memory savings), while the small LoRA adapter matrices `A, B` are still trained in higher precision (e.g. bf16). This is what makes it feasible to fine-tune a 65B+ parameter model on a **single consumer GPU** — a concrete, quotable practical result from the QLoRA paper.

### 🧮 Numerical Example — QLoRA memory savings for a 65B model

Base model weight storage only (ignoring optimizer state, since only the tiny LoRA adapter needs gradients/optimizer state under QLoRA):

```
fp16 storage:  65B × 2 bytes = 130 GB
4-bit (NF4) storage: 65B × 0.5 bytes = 32.5 GB
```
`0.5 bytes` because 4 bits = 0.5 bytes per parameter. That's a **4x reduction** (130 GB → 32.5 GB) just from quantizing the frozen base weights — and because the LoRA adapter itself is tiny (tens of MB, as computed above), the *total* memory footprint (≈32.5 GB base + a small LoRA/optimizer overhead + activations) can fit within a single high-end consumer/prosumer GPU (e.g., a 48 GB card), whereas the fp16 base model alone (130 GB) could never fit on a single consumer GPU, let alone with full fine-tuning's ~16 bytes/param optimizer overhead (65B × 16 bytes = 1,040 GB). This chain of numbers — 1,040 GB (full FT) → 130 GB (fp16 inference-only) → 32.5 GB (QLoRA 4-bit) — is a great one to narrate live in an interview.

### Other PEFT methods (briefly, know they exist + one differentiator each)
- **Adapters**: insert small new trainable feed-forward "bottleneck" layers *between* existing frozen Transformer layers (rather than modifying existing weight matrices in place like LoRA does) — adds a small amount of inference latency since it's literally extra layers in the forward pass, whereas LoRA's `BA` can be mathematically merged back into `W` after training, adding **zero** extra inference latency.
- **Prefix-tuning / Prompt-tuning**: instead of modifying any weights, prepend a small number of **trainable "virtual token" embeddings** to the input at every layer (prefix-tuning) or just the input layer (prompt-tuning) — the base model and all its real weights stay completely frozen; only these virtual embeddings are learned. Cheapest of all in trainable-parameter count, but generally the weakest in task performance among PEFT methods for harder tasks.

### 📌 Added Explanation — "in simple terms" comparison of the three PEFT families

- **LoRA**: modifies *existing* weight matrices via a small additive rank-`r` patch. Mergeable → zero extra latency.
- **Adapters**: inserts brand-new small layers *between* existing frozen layers. Not mergeable (they're extra computation, not a patch to existing weights) → adds latency.
- **Prefix/Prompt-tuning**: doesn't touch weights at all, just prepends learned "virtual tokens" to the input so the frozen model's own attention does the adapting. Cheapest, but weakest, because you're not giving the model any new computational capacity — you're only ever nudging what it attends to.

An easy memory hook: **LoRA patches weights, Adapters add layers, Prefix-tuning adds fake input tokens.** Only Adapters add inference latency, because only Adapters are literally extra layers you must compute through every forward pass.

---

## 4. Fine-tune vs Prompt vs RAG — the decision framework (common system-design question)

This is less a formula and more a framework interviewers want you to reason through out loud:

**Use prompting/few-shot when**: the task is simple, you need to iterate fast, you don't have labeled training data, or the behavior you want changes frequently (prompts are instant to edit; fine-tunes require retraining).

**Use RAG (retrieval-augmented generation) when**: the model needs access to information that's *frequently changing* or *too large to bake into weights* (a live product catalog, a company's internal docs) — RAG keeps the knowledge external and retrievable, so updating the knowledge base doesn't require retraining anything.

**Use fine-tuning (full or PEFT) when**: you need to change the model's **behavior/style/format** persistently and consistently (e.g., always respond in a specific JSON schema, always adopt a specific tone, or perform a narrow specialized task extremely reliably) — fine-tuning bakes in a behavior pattern more reliably and with less per-request prompt-length overhead than repeatedly stuffing instructions/examples into every prompt.

**Practical combined pattern in real systems**: it's common to use **all three together** — a fine-tuned/instruction-tuned base model, augmented with RAG for up-to-date factual grounding, further steered per-request with prompting/few-shot examples for the specific immediate task. Say this explicitly if asked "which one should I use" — the honest answer in production is usually "not mutually exclusive."

### 📌 Added Explanation — a decision-tree phrasing for interviews

If you're asked this live, a clean way to narrate it out loud:

1. "First I'd ask: does the model need facts that change often or are too big to memorize? If yes → RAG handles the knowledge layer."
2. "Then I'd ask: do I need the model's *behavior* (format, tone, task specialization) to be reliable without me re-explaining it every single call? If yes → fine-tuning (PEFT if I want it cheap/swappable, full FT if I have huge budget and need maximum reliability)."
3. "Finally, whatever's left — task-specific nuances, edge cases, the immediate ask — I'd handle with prompting/few-shot on top of whichever of the above I've already set up."

This framing signals to an interviewer that you see these as complementary layers of a system, not competing techniques you must pick exactly one of.

---

## 5. Side-by-side summary table (memorize this cold)

| | Prompting / ICL | Full Fine-tuning | LoRA (PEFT) |
|---|---|---|---|
| Weights updated? | None | All | Small added low-rank matrices only |
| Persistence | None — must repeat every call | Permanent, baked in | Permanent, baked into small adapter |
| Cost | Cheapest, instant | Most expensive (full grad+optimizer state) | Cheap (often 100-1000x fewer trainable params) |
| Storage per task | Zero extra | Full model copy per task | A few MB per task (adapter only) |
| Inference latency added | None (just longer prompt) | None | None (if merged back into W after training) |
| Adds new knowledge well? | Limited — bounded by context window | Yes | Yes, though less capacity than full fine-tune for large domain shifts |

---

## 6. Quick-fire Q&A (self-test)

**Q: What are the two leading mechanistic explanations for why in-context learning works?**
A: Induction heads (attention heads that implement "copy the completion of a pattern seen earlier"), and the meta-gradient view (the forward pass over in-context examples behaves mathematically similar to an implicit gradient-descent update, without touching stored weights).

#### 📌 Added Explanation — fuller answer with reasoning
Both explanations answer the same puzzle from different angles — one **mechanistic/empirical**, one **theoretical/mathematical**:
- *Induction heads* is a bottom-up, "we looked inside real trained Transformers with interpretability tools and found specific attention heads doing this specific copying computation" answer. It's grounded in actual observed circuits.
- *Meta-gradient* is a top-down, "under simplifying mathematical assumptions, we can show the forward computation is algebraically equivalent to a gradient step" answer. It's grounded in theory, not direct observation of real production models.
Together they suggest ICL isn't one single trick — it's consistent with the model having learned, during pretraining, general "pattern-continuation" machinery that can be *interpreted* both as literal pattern-copying (induction heads) and as *approximating* what a gradient update would do (meta-gradient view). Reasoning: because both accounts converge on the same functional behavior (use recent context to predict analogous continuations) from independent directions, they reinforce rather than contradict each other, even though neither alone is a fully complete proof for large-scale nonlinear models (see 🔎 Accuracy Flag above).

**Q: What's the key finding of instruction tuning (FLAN) beyond just "fine-tuning on more tasks"?**
A: Training on a diverse mixture of instruction-phrased tasks improves zero-shot generalization to entirely new, unseen tasks/instructions — the model learns the general skill of following instructions, not just memorizing specific task formats.

#### 📌 Added Explanation — fuller answer with reasoning
The reasoning hinges on what gradient descent can "notice" as a consistent signal across the training set. If every training example were from the same task, the model's easiest path to lowering the loss is to fit that one task's surface patterns. Once you mix in hundreds of *different* tasks, all phrased as "instruction → do the thing," the only feature *common to all of them* is "follow whatever the instruction says." Since gradient descent tends to find the simplest consistent explanation for the training signal, and "follow instructions in general" is a simpler, more consistent explanation across a highly diverse task mixture than memorizing hundreds of unrelated task-specific mappings, the model is pushed toward learning the general instruction-following skill. That general skill then transfers to brand-new instructions never seen in training, which is exactly the "zero-shot generalization to unseen tasks" result FLAN reports.

**Q: Write the LoRA weight update formula and explain each symbol.**
A: `W_new = W_frozen + BA`, where `W_frozen` is the original pretrained weight matrix (never updated), and `B` (d×r) and `A` (r×d) are small trainable low-rank matrices whose product approximates the needed weight change, with rank r ≪ d.

#### 📌 Added Explanation — fuller answer with reasoning
See the full derivation in Section 3 above, but condensed: `W_frozen` is `d×d` and fixed; the "ideal" fine-tuned matrix would be `W_frozen + ΔW_ideal` where `ΔW_ideal` is also `d×d` and, under full fine-tuning, has up to `d²` independently-trainable entries. LoRA replaces `ΔW_ideal` with the constrained form `BA`, where `A` is `r×d` and `B` is `d×r`. Because matrix multiplication of an `d×r` matrix by an `r×d` matrix can produce a `d×d` result with **rank at most `r`** (a linear-algebra fact: rank of a product is bounded by the minimum rank of its factors, and each factor has rank at most `r` since `r < d`), `BA` can only express `d×d`-shaped updates that live in an `r`-dimensional "direction budget," dramatically cutting the number of independently trainable numbers from `d²` down to `2rd`, while (empirically) still capturing most of the useful adaptation because real task-adaptation updates tend to be low-rank in practice.

**Q: For d=4096 and rank r=8, how many trainable parameters does LoRA use for one weight matrix, vs full fine-tuning, and what's the reduction factor?**
A: Full fine-tuning: 4096×4096 ≈ 16.78M params. LoRA: (8×4096)+(4096×8) = 65,536 params. Reduction ≈ 256x fewer trainable parameters.

#### 📌 Added Explanation — fuller answer with reasoning
Full fine-tuning trains every entry of the `d×d` matrix independently: `4096 × 4096 = 16,777,216`. LoRA instead trains only the entries of `A` (`8 × 4096 = 32,768`) and `B` (`4096 × 8 = 32,768`), summing to `65,536`. Dividing: `16,777,216 / 65,536 = 256`. The reasoning behind *why* this specific ratio falls out: the parameter count for LoRA scales as `2rd` (linear in both `r` and `d`), while full fine-tuning scales as `d²` (quadratic in `d`). The ratio `d² / 2rd = d / 2r`. Plugging in `d=4096, r=8`: `4096 / 16 = 256` — matching the computed reduction factor exactly, and showing algebraically that the reduction factor grows *linearly with `d`* (bigger models benefit even more from LoRA) and shrinks linearly as you increase `r` (higher-rank adapters cost more, as tabulated earlier).

**Q: Why does LoRA add zero inference latency, while Adapter layers do add latency?**
A: LoRA's BA update can be mathematically merged directly back into the frozen weight matrix W after training (W_new is just a single matrix again), so inference uses the exact same architecture as the base model. Adapters insert genuinely new layers into the forward pass, so every inference call must compute through those extra layers.

#### 📌 Added Explanation — fuller answer with reasoning
The reasoning comes down to *where* each method's new parameters live relative to the existing computational graph. LoRA's new parameters (`A, B`) are combined with `W_frozen` via simple matrix addition (`W_new = W_frozen + BA`) — addition is commutative/associative and can be done once, offline, after training finishes, producing a single ordinary weight matrix indistinguishable in shape or computational role from any other weight matrix the base model already had. So at inference time, the forward pass literally does not know LoRA was ever used — it's just matrix `W_new` where `W_frozen` used to be. Adapters, by contrast, insert an entirely new sub-network (extra linear layers + nonlinearity) as an additional *step* in the computational graph, between existing frozen layers — this is not something you can "fold into" the surrounding frozen weights, because it's a nonlinear function applied sequentially, not a simple additive correction to an existing linear map. Therefore every inference call must literally execute those extra matrix multiplications and nonlinearities, which is where the added latency comes from.

**Q: What does QLoRA add on top of LoRA, and what practical result does it enable?**
A: QLoRA quantizes the frozen base model weights to 4-bit precision (via NF4) while keeping the small trainable LoRA matrices in higher precision — this cuts base-model memory footprint enough to fine-tune 65B+ parameter models on a single consumer GPU.

#### 📌 Added Explanation — fuller answer with reasoning
The reasoning for why this combination works (rather than just quantizing everything, or just using LoRA alone) is that the two techniques target *different* sources of memory cost and are compatible because they touch different objects: LoRA already ensures that the *trainable* parameters (and their gradients/optimizer state) are tiny (Section 3's numbers), so the dominant remaining memory cost is simply *storing* the huge frozen base model, which never needs gradients or optimizer state at all under LoRA. Since the frozen base model is never updated, there's no correctness reason it needs to be kept in a high-precision format for training purposes — it only needs to be precise enough that forward-pass computations (used to produce activations that the LoRA layers then adapt) remain numerically reasonable. NF4 is designed specifically to represent typical pretrained-weight distributions (which tend to be roughly Gaussian-ish) more accurately at 4 bits than a naive uniform 4-bit quantization would, minimizing the accuracy lost from quantization. So QLoRA reasoning is: "the part that must stay precise and trainable (LoRA adapters) is already tiny; the part that's huge (frozen base weights) doesn't need training precision, only inference precision, so aggressively quantize just that huge part." That's what allows a 65B-parameter model, which would need ~130GB in fp16 just to store, to instead fit in roughly a quarter of that (Section 3's numerical example: ~32.5GB), small enough for a single consumer GPU.

**Q: When would you choose RAG over fine-tuning?**
A: When the needed information changes frequently or is too large to bake into weights (e.g. a live product catalog or internal docs) — RAG keeps knowledge external and updatable without retraining, whereas fine-tuning is better for baking in a persistent behavior/style/format rather than volatile facts.

#### 📌 Added Explanation — fuller answer with reasoning
The reasoning is about **where information should physically live relative to how often it changes**. Fine-tuning bakes information into the model's weights, which is a slow, batch, offline process (you must re-run training every time you want to update what's baked in) — that's a poor fit for information that changes daily/hourly (stock prices, current inventory, this week's internal announcements), because you'd be perpetually retraining just to stay current, and there is no guarantee the model won't also "forget" or blend outdated and updated facts inconsistently (a known failure mode called catastrophic forgetting/knowledge conflict). RAG instead keeps facts in an external, directly-editable store (a database/vector index) and retrieves the relevant facts at query time to include in the prompt — updating a fact is then just a database write, taking effect immediately on the very next query, with no retraining at all. Fine-tuning remains the better tool for changing *how* the model behaves (format, tone, task-specific reasoning patterns) because behavior is not "a fact to look up" — it's a persistent disposition you want the model to have every time, which is exactly what baking a pattern into weights (rather than retrieving it) is good for.

---

## ❓ Interview Q&A — Apple / Google ML Engineer style questions

*(These go beyond the "quick-fire" self-test above — they're phrased the way an interviewer would actually ask them in a live loop, often as follow-ups probing depth, trade-offs, and system-design judgment.)*

**Q1. "Walk me through what happens, mechanically, inside the model when I give it a few-shot prompt — don't just say 'it learns from examples,' actually explain the computation."**
A: Nothing is learned in the weight-update sense — there is exactly one forward pass over the entire concatenated sequence (instructions + examples + query), with the same frozen `Q/K/V` projection matrices used for ordinary next-token prediction. Self-attention lets the final query tokens attend back over the earlier example tokens; specific attention circuits (induction heads) detect the repeating template and effectively "copy forward" the label that followed the most similar earlier example. The output is just an ordinary softmax over the vocabulary at the final position, conditioned on unusually informative context — no gradients, no optimizer, no parameter changes anywhere. This is worth emphasizing explicitly because interviewers are checking whether you conflate "the model appears to learn" with "the model's parameters change" — they don't.

**Q2. "If LoRA only touches 0.1% of parameters, why doesn't it perform much worse than full fine-tuning? Isn't it obviously less expressive?"**
A: It is strictly less expressive in a mathematical sense (any full-rank update is not exactly representable by a rank-`r` matrix if `r < d`), but empirically the *useful* updates needed for most downstream task adaptation appear to have low intrinsic rank — meaning the actual difference between "pretrained weights" and "ideally task-adapted weights" doesn't require full-rank expressiveness to capture the bulk of the performance gain. This is an empirical claim from the LoRA paper's ablations, not a theorem, so the honest answer includes: "for tasks requiring large domain shifts or entirely new knowledge, LoRA (or PEFT generally) tends to underperform full fine-tuning, which is consistent with the low-rank hypothesis being a good-but-not-universal approximation."

**Q3. "You're asked to deploy a model that must (a) always answer in a fixed JSON schema, (b) reference our constantly-updated internal wiki, and (c) handle a handful of ad hoc one-off requests from different teams. Design the system — which of prompting/fine-tuning/RAG do you use where, and why?"**
A: (a) Fixed JSON schema output is a **persistent behavioral constraint** → best solved with fine-tuning (likely LoRA for cost reasons) so the model reliably emits the schema without needing the schema re-specified in every prompt, saving prompt length and improving reliability. (b) A constantly-updated internal wiki is **volatile, large, external knowledge** → RAG, so wiki edits take effect immediately without retraining. (c) Ad hoc, one-off, per-team requests are exactly the low-effort, fast-iteration, no-training-data-available case → prompting/few-shot on top of the already fine-tuned + RAG-augmented base. This mirrors the "combined pattern" called out in Section 4 and demonstrates you're reasoning about these as complementary layers of one system, which is what system-design interviewers are actually screening for.

**Q4. "What's a failure mode of choosing fine-tuning when you should have used RAG (or vice versa)?"**
A: Fine-tuning-when-you-needed-RAG: you bake in a fact (e.g., "the current CEO is X") that later becomes false, and now every future output confidently repeats the stale fact until you retrain — worse, retraining on updated facts can cause catastrophic forgetting of other things the model previously knew, or produce inconsistent blends of old/new facts. RAG-when-you-needed-fine-tuning: you try to control persistent behavior/format purely by stuffing formatting instructions into every prompt via retrieved context, but the model still drifts from the desired format under distribution shift or long conversations, because "please always respond in JSON" sitting in a retrieved context chunk is just another instruction competing with everything else in the prompt, not a baked-in disposition — it's inherently less reliable than a fine-tuned behavioral prior.

**Q5. "Derive, from first principles, why LoRA's parameter count scales as `2rd` and not `rd` or `r²`."**
A: `A` has shape `(r × d)`, contributing `r × d` independent trainable entries. `B` has shape `(d × r)`, contributing `d × r` independent trainable entries. These are two separate matrices with no shared entries, so total trainable parameters = `(r×d) + (d×r) = 2rd`. It is not `rd` because there are genuinely two matrices, not one, and it is not `r²` because neither matrix is `r × r` — the whole point of the low-rank factorization is that the *large* dimension `d` appears once in each factor (so the parameter count still scales linearly with `d`, not quadratically like full fine-tuning's `d²`), while `r` is the "bottleneck" dimension shared between the two factors that keeps the overall count small.

**Q6. "Under what conditions might full fine-tuning still be the right choice over PEFT, despite the cost?"**
A: When you have (a) a very large labeled dataset specific to a domain far from the pretraining distribution (e.g., adapting a general LLM into a highly specialized scientific/legal model with substantial new terminology/knowledge), where the low-intrinsic-rank hypothesis is less likely to hold because the needed change is large and diffuse rather than a small correction; (b) enough compute budget that the ~8-16x memory overhead versus inference-only footprint (see Section 2's numerical derivation) is not prohibitive; and (c) you only need to serve one or a small number of specialized model variants, so you don't need PEFT's "many cheap swappable adapters on one shared base" advantage. In short: PEFT wins on cost and flexibility for many small-to-moderate adaptations; full fine-tuning can still win on raw capability ceiling for a small number of large, high-budget, high-value specializations.

---
*End of Module 4 (expanded). Next: Module 5 — Alignment: RLHF & Alternatives (SFT → reward model → PPO, DPO derivation, RLAIF/Constitutional AI/KTO).*
