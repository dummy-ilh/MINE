# Module 4 — Fine-tuning vs Prompting vs In-Context Learning (Master Notes, Expanded)

## 0. The big picture — three ways to adapt a pretrained model to a task

A pretrained LLM knows "language + world knowledge" broadly, but you usually want it to do something specific (classify support tickets, follow instructions, answer in a certain style). There are three fundamentally different ways to get there, differing in **whether you touch the model's weights at all**:

| Approach | Touches weights? | Cost | Persistence |
|---|---|---|---|
| Prompting / in-context learning | No | Cheap, instant, per-request | Not learned — must repeat every time |
| Full fine-tuning | Yes, all weights | Expensive (full backprop through all params) | Permanently baked into the model |
| Parameter-efficient fine-tuning (LoRA etc.) | Yes, small added subset | Cheap-ish (small fraction of full fine-tune cost) | Permanently baked into a small add-on |

---

## 1. Zero-shot / Few-shot Prompting and In-Context Learning (ICL)

### Core idea, in plain words
Just describe the task (zero-shot) or show a few examples (few-shot) **directly in the input text**, at inference time, with no weight updates at all. The model "figures out" the task pattern purely from the prompt.

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

### Why ICL works at all — the mechanistic explanations (interview-level, cite both)
1. **Induction heads** (Anthropic's interpretability research finding): certain attention heads, discovered via mechanistic interpretability, specifically implement a "if I've seen pattern [A][B] before, and I now see [A] again, predict [B]" copying behavior. Few-shot examples give the model exactly this kind of repeatable pattern to latch onto — literally attending back to earlier occurrences of a similar structure and copying the completion pattern forward.
2. **Meta-gradient / implicit fine-tuning view**: some theoretical work argues that the forward pass through a Transformer, when given in-context examples, is mathematically analogous to performing an implicit gradient-descent-like update using the in-context examples as if they were training data — the attention mechanism's output can be shown (under simplifying assumptions) to resemble a gradient step, essentially "fine-tuning within the forward pass," without ever touching stored weights.

**Practical framing to say out loud**: "ICL isn't the model learning new facts on the fly — it's the frozen model recognizing 'oh, this is the pattern-completion task,' using in-context examples the same way induction heads use repeated sequences, or equivalently, approximating what a gradient update would have done, all within a single forward pass."

---

## 2. Full Fine-tuning vs Instruction Tuning

### Full fine-tuning
Take the pretrained model, continue training (standard backprop, all parameters updated) on a **labeled, task-specific dataset**. Example: take a base CLM-pretrained model and continue training it purely on medical-question-answer pairs to specialize it.

**Cost problem**: full fine-tuning requires storing gradients and optimizer states (e.g. Adam needs 2 extra copies of every parameter — first and second moment estimates) for **every single parameter** in the model. For a 70B-parameter model, this means memory footprint several times the base model size just for training state — completely impractical on typical hardware, and you end up with a full separate copy of a 70B-parameter model per fine-tuned task.

### Instruction tuning (FLAN, T0 — a specific *type* of full fine-tuning)
Instead of fine-tuning on one narrow task, fine-tune on a **large, diverse mixture of tasks, each phrased as a natural-language instruction** ("Summarize this text:", "Translate to French:", "Answer this question:" ...) with many different task types combined in one training set.

**The key finding (FLAN paper)**: this doesn't just make the model better at the specific tasks it was instruction-tuned on — it improves **zero-shot generalization to entirely new, unseen tasks/instructions**, because the model learns the general skill of "parse an instruction and follow it," not just memorize answers to specific task formats. This is precisely the step that turns a raw pretrained CLM model into something that behaves like "ChatGPT-style, follows your instructions" rather than just "autocomplete."

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

### Why this drastically cuts trainable parameters — the actual numbers
Say `d = 4096` (a realistic hidden dimension) and rank `r = 8`.

**Full fine-tuning** this one weight matrix: `d × d = 4096 × 4096 = 16,777,216` trainable parameters.

**LoRA** with rank 8: `A` is `(8 × 4096) = 32,768` params, `B` is `(4096 × 8) = 32,768` params → total = `65,536` trainable parameters.

```
Reduction factor = 16,777,216 / 65,536 = 256x fewer trainable parameters, for this one matrix.
```
This is the number to have ready in an interview: **LoRA can cut trainable parameters by roughly 2-3 orders of magnitude** depending on rank choice, while empirically retaining most of full-fine-tuning's task performance for many tasks.

### Why low-rank works at all (the intuition, not just the mechanics)
The hypothesis behind LoRA (supported empirically in the paper) is that the *change* a model needs during task-adaptation (the difference between pretrained weights and ideally fine-tuned weights) has a naturally **low "intrinsic rank"** — i.e., the useful update direction lives in a small subspace, even though the full weight matrix is huge. You don't need to move every one of 16M+ directions; you need to move a much smaller number of the *right* directions, and `B×A` (rank r) is exactly a compact way to parameterize "a rank-r update in the full d×d space."

### Practical deployment benefit (a favorite follow-up question)
Because `ΔW = BA` is small, you can store many different task-specific `(A, B)` pairs (a few MB each) alongside a single shared frozen base model (many GB), and **swap adapters at inference time** without duplicating the base model — this is why LoRA became the dominant way to serve many customized model variants cheaply.

### QLoRA — the natural follow-up extension
QLoRA combines LoRA with **quantizing the frozen base model** (typically to 4-bit precision, using a technique called NF4 — "4-bit NormalFloat," designed to match the actual distribution of pretrained weights better than naive uniform 4-bit quantization) — so the large frozen weights sit in GPU memory at 4-bit precision (huge memory savings), while the small LoRA adapter matrices `A, B` are still trained in higher precision (e.g. bf16). This is what makes it feasible to fine-tune a 65B+ parameter model on a **single consumer GPU** — a concrete, quotable practical result from the QLoRA paper.

### Other PEFT methods (briefly, know they exist + one differentiator each)
- **Adapters**: insert small new trainable feed-forward "bottleneck" layers *between* existing frozen Transformer layers (rather than modifying existing weight matrices in place like LoRA does) — adds a small amount of inference latency since it's literally extra layers in the forward pass, whereas LoRA's `BA` can be mathematically merged back into `W` after training, adding **zero** extra inference latency.
- **Prefix-tuning / Prompt-tuning**: instead of modifying any weights, prepend a small number of **trainable "virtual token" embeddings** to the input at every layer (prefix-tuning) or just the input layer (prompt-tuning) — the base model and all its real weights stay completely frozen; only these virtual embeddings are learned. Cheapest of all in trainable-parameter count, but generally the weakest in task performance among PEFT methods for harder tasks.

---

## 4. Fine-tune vs Prompt vs RAG — the decision framework (common system-design question)

This is less a formula and more a framework interviewers want you to reason through out loud:

**Use prompting/few-shot when**: the task is simple, you need to iterate fast, you don't have labeled training data, or the behavior you want changes frequently (prompts are instant to edit; fine-tunes require retraining).

**Use RAG (retrieval-augmented generation) when**: the model needs access to information that's *frequently changing* or *too large to bake into weights* (a live product catalog, a company's internal docs) — RAG keeps the knowledge external and retrievable, so updating the knowledge base doesn't require retraining anything.

**Use fine-tuning (full or PEFT) when**: you need to change the model's **behavior/style/format** persistently and consistently (e.g., always respond in a specific JSON schema, always adopt a specific tone, or perform a narrow specialized task extremely reliably) — fine-tuning bakes in a behavior pattern more reliably and with less per-request prompt-length overhead than repeatedly stuffing instructions/examples into every prompt.

**Practical combined pattern in real systems**: it's common to use **all three together** — a fine-tuned/instruction-tuned base model, augmented with RAG for up-to-date factual grounding, further steered per-request with prompting/few-shot examples for the specific immediate task. Say this explicitly if asked "which one should I use" — the honest answer in production is usually "not mutually exclusive."

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

**Q: What's the key finding of instruction tuning (FLAN) beyond just "fine-tuning on more tasks"?**
A: Training on a diverse mixture of instruction-phrased tasks improves zero-shot generalization to entirely new, unseen tasks/instructions — the model learns the general skill of following instructions, not just memorizing specific task formats.

**Q: Write the LoRA weight update formula and explain each symbol.**
A: `W_new = W_frozen + BA`, where `W_frozen` is the original pretrained weight matrix (never updated), and `B` (d×r) and `A` (r×d) are small trainable low-rank matrices whose product approximates the needed weight change, with rank r ≪ d.

**Q: For d=4096 and rank r=8, how many trainable parameters does LoRA use for one weight matrix, vs full fine-tuning, and what's the reduction factor?**
A: Full fine-tuning: 4096×4096 ≈ 16.78M params. LoRA: (8×4096)+(4096×8) = 65,536 params. Reduction ≈ 256x fewer trainable parameters.

**Q: Why does LoRA add zero inference latency, while Adapter layers do add latency?**
A: LoRA's BA update can be mathematically merged directly back into the frozen weight matrix W after training (W_new is just a single matrix again), so inference uses the exact same architecture as the base model. Adapters insert genuinely new layers into the forward pass, so every inference call must compute through those extra layers.

**Q: What does QLoRA add on top of LoRA, and what practical result does it enable?**
A: QLoRA quantizes the frozen base model weights to 4-bit precision (via NF4) while keeping the small trainable LoRA matrices in higher precision — this cuts base-model memory footprint enough to fine-tune 65B+ parameter models on a single consumer GPU.

**Q: When would you choose RAG over fine-tuning?**
A: When the needed information changes frequently or is too large to bake into weights (e.g. a live product catalog or internal docs) — RAG keeps knowledge external and updatable without retraining, whereas fine-tuning is better for baking in a persistent behavior/style/format rather than volatile facts.

---
*End of Module 4 (expanded). Next: Module 5 — Alignment: RLHF & Alternatives (SFT → reward model → PPO, DPO derivation, RLAIF/Constitutional AI/KTO).*
