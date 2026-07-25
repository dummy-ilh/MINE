# Module 9 — Interview-Style Synthesis (Master Notes, Maximum Depth)

## 0. How to use this module

This isn't new material — it's **cross-module synthesis**: questions that force you to connect ideas across Modules 1-8, the way a real interviewer actually probes (they rarely ask "define BPE" in isolation; they ask "walk me through what happens when a user sends a prompt," and expect you to touch tokenization → inference → serving in one answer). Each section below groups questions by the *kind* of interview question, not by module, since that's how they'll actually show up.

---

## 1. "Walk me through end-to-end" questions

### Q: A user sends a prompt to a deployed chat LLM. Walk me through everything that happens, technically, from input to output.

**Strong answer structure** (this is the shape to reproduce, not memorize verbatim):
1. **Tokenization** (Module 1): the raw text prompt is tokenized using the model's fixed vocabulary (e.g. byte-level BPE) into a sequence of integer token IDs.
2. **Prefill / initial forward pass**: all prompt tokens are processed through the model in a single parallel forward pass — this is where the KV cache (Module 6) gets populated for every prompt token's Key/Value vectors, across every layer.
3. **Autoregressive decoding begins** (Module 6): the model computes a probability distribution over the vocabulary for the first new token; a decoding strategy (temperature + top-p, typically, for a chat product) samples the next token.
4. **Loop**: that new token's K/V is appended to the cache (not recomputed from scratch — this is exactly why KV caching matters for latency), and the process repeats — generate one token, sample, append to cache, repeat — until an end-of-sequence token or length limit is hit.
5. **(Optionally) Speculative decoding** (Module 6) may be used under the hood to reduce the number of expensive full forward passes needed for a given number of output tokens, without changing the output distribution.
6. **Serving-level considerations** (Module 7): the underlying weights are likely served in a reduced-precision format (fp16/bf16, possibly further quantized to int8/int4) to fit memory and reduce latency/cost; if it's a MoE model, only a small subset of experts are active per token.
7. **Detokenization**: the generated token IDs are converted back into human-readable text using the same tokenizer's vocabulary.

**What this question is actually testing**: whether you understand that tokenization, decoding, and serving efficiency are not independent trivia facts, but a single connected pipeline — losing points here usually means answering only one piece (e.g. only decoding strategies) without touching how it connects to caching/serving.

### Q: Walk me through how you'd take a base pretrained model to a production-ready chat assistant.

**Strong answer structure**:
1. **Pretraining** (Module 2): CLM objective on a massive, deduplicated, deliberately-mixed corpus, at a roughly Chinchilla-optimal token-to-parameter ratio (Module 3) for the chosen compute budget.
2. **SFT / instruction tuning** (Module 4/5): fine-tune on diverse instruction-formatted (prompt, ideal-response) pairs — this is the stage that teaches the model to actually follow instructions rather than just autocomplete plausible web text.
3. **Reward model training** (Module 5): collect pairwise human preference comparisons on SFT-model outputs, train a scalar reward model via the Bradley-Terry loss.
4. **RLHF (PPO) or DPO** (Module 5): align the policy toward human preferences using the reward model + KL-constrained PPO, or skip the explicit reward model entirely with DPO's direct preference loss.
5. **Evaluation** (Module 8) at every stage — benchmark suites for broad capability tracking, human eval/LLM-as-judge for the specific alignment-sensitive behaviors, hallucination measurement, watching for benchmark contamination in reported numbers.
6. **Serving optimization** (Module 7) for production: quantization, possibly MoE architecture choice made far earlier (at pretraining time, not a late add-on), KV-cache-aware serving infrastructure.

---

## 2. "Compare and decide" questions (these test judgment, not just recall)

### Q: You have a fixed budget — would you rather train a bigger model on less data, or a smaller model on more data?

Answer using Module 3's Chinchilla result directly: for a **fixed compute budget**, the compute-optimal choice is **not** "as big as possible" — it's roughly balancing N and D at the ~20-tokens-per-parameter ratio. State the GPT-3-vs-Chinchilla numbers as concrete evidence (GPT-3: ~1.7 tok/param, badly undertrained; Chinchilla: ~20 tok/param, smaller but outperformed GPT-3). Then add the Module 3 nuance: if **inference cost over the model's lifetime** also matters (not just training-compute-optimality), you might deliberately choose an even smaller model trained on even more tokens than Chinchilla-optimal (the Llama approach) — a genuinely balanced answer names both considerations rather than treating Chinchilla's ratio as the final word.

### Q: Fine-tune, prompt, or RAG — how do you decide?

Directly reuse Module 4's decision framework: prompting for fast iteration/no labeled data/frequently-changing behavior; RAG for knowledge that's large or changes frequently (avoids retraining to update facts); fine-tuning for persistent behavior/style/format changes. Close with the honest production nuance: these are usually **combined**, not mutually exclusive (a fine-tuned base model, augmented with RAG, further steered by per-request prompting).

### Q: RLHF (PPO) or DPO — which would you pick for a real alignment pipeline, and why?

Reuse Module 5's balanced closer: DPO for simplicity/stability/cost (no reward model, no RL infrastructure); RLHF/PPO if you specifically want a standalone reusable reward model (for reranking, red-teaming, evaluation) or believe online RL exploration reaches preference signal a fixed offline DPO dataset can't. **Don't just pick one — name the actual tradeoff**, since interviewers are testing whether you understand *why* the tradeoff exists, not just which one is currently trendier.

### Q: Full fine-tuning or LoRA?

Full fine-tuning if you have abundant compute/data and need maximum possible task adaptation with no constraint on trainable capacity (e.g., a large domain shift). LoRA (Module 4) when compute/memory is constrained, when you need to serve many different task-specialized variants cheaply (small adapters swapped over one shared frozen base), or when the needed weight update plausibly has low intrinsic rank — which empirically covers a very large fraction of real fine-tuning use cases, which is exactly why LoRA became the default rather than the exception in practice.

---

## 3. "Diagnose the failure" questions (a very common Google/Apple MLE interview pattern)

### Q: Your fine-tuned model's benchmark scores look great, but it performs poorly for real users. What's going on, and how would you investigate?

Pull directly from Module 8: first suspect is **benchmark contamination** (check whether benchmark-style questions leaked into pretraining/fine-tuning data) or **benchmark saturation combined with format sensitivity** (the benchmark may no longer differentiate meaningfully, or the eval harness's exact formatting doesn't match how real users actually phrase requests). Second-order suspect: the benchmark measures narrow capability (e.g. multiple-choice knowledge, MMLU-style) that doesn't correlate well with the actual open-ended, conversational task real users need (this is the Module 2/3 "loss/benchmark doesn't equal downstream usefulness" theme resurfacing). Investigation plan: run targeted human evaluation or LLM-as-judge (Module 8) directly on a sample of real production-style queries, specifically checking for the known biases (length, position) that might be inflating the perception of quality if you're only looking at automated proxy metrics.

### Q: Users report the model "sounds confident but says wrong things." Diagnose and propose a fix.

This is a hallucination + calibration question (Module 8), directly connected to the CLM objective's mechanism (Module 2): CLM optimizes for fluent, plausible continuation, with no built-in fact-verification signal, so confident tone and factual correctness are correlated but not identical in what the model was trained to produce. Propose: (a) consistency-based hallucination measurement (sample multiple generations, check agreement, per SelfCheckGPT-style methods) to *quantify* the problem before fixing it, (b) calibration-aware fine-tuning/alignment (reward honest expressions of uncertainty during RLHF/DPO preference data collection — i.e., bake "say 'I'm not sure' when appropriate" into the preference-comparison training signal itself, not just correctness), (c) RAG-style grounding (Module 4) for domains where hallucination risk is highest and external verifiable sources exist.

### Q: Your RLHF-trained model's outputs have gotten weird/degenerate compared to the SFT checkpoint — repetitive praise, strange phrasing, gaming-the-metric behavior. What happened, and how do you fix it?

Textbook **reward hacking** (Module 5) — the policy has drifted too far from `π_SFT` and found a blind spot in the reward model's approximation of true human preference (Goodhart's Law). Fix: increase the KL-penalty coefficient β to constrain drift more tightly, inspect/retrain the reward model on more diverse preference data specifically covering the failure mode observed, or consider whether the reward model was undertrained/overfit on too narrow a preference distribution in the first place.

### Q: A long-context deployment is running out of GPU memory as conversations get longer, even though the model itself fits fine in memory. Why, and what are your options?

This is testing whether you separate model-weight memory from **KV cache memory** (Module 6) — KV cache grows linearly with sequence length × layers × hidden dim × batch size, and can become the dominant memory consumer for long conversations/large batches, entirely independent of the (fixed) model weight footprint. Options: Multi-Query Attention/Grouped-Query Attention (Module 6) to shrink cache size architecturally, quantizing the KV cache itself (not just weights), reducing max batch size or max context length as a serving-config lever, or RoPE-scaling/ALiBi-style techniques (Module 6) if the real goal is supporting longer contexts without proportionally exploding memory.

---

## 4. "Explain like I'm the interviewer, and I'll interrupt with follow-ups" — derivation-style questions

These are the ones where being able to actually write the formula, not just gesture at it, separates strong from weak answers. Have these fully loaded:

- **Chinchilla compute formula**: `C ≈ 6ND`, and the ~20 tokens/parameter compute-optimal ratio, plus the concrete GPT-3 vs Chinchilla numbers (Module 3).
- **LoRA parameter count**: `ΔW = BA`, with the explicit d=4096, r=8 → 256x reduction example (Module 4) ready to redo live on a whiteboard.
- **Bradley-Terry + DPO derivation**: from `P(y_w≻y_l) = σ(r(y_w)-r(y_l))`, through the closed-form optimal policy `π* = (1/Z)π_SFT·exp(r/β)`, to the final DPO loss with `Z(x)` cancellation (Module 5) — this is the single most likely "derive this on the whiteboard" question in the whole syllabus, given how clean and self-contained the derivation is.
- **KV cache complexity**: O(N²) naive vs O(N) cached, plus the concrete ~2GB-for-one-sequence memory example (Module 6).
- **Quantization formula**: `scale = (max-min)/(2^n-1)`, plus why NF4 uses non-uniform quantile-based spacing instead (Module 7).

---

## 5. Rapid-fire cross-module connections (say these out loud, unprompted, when relevant — they signal real understanding, not memorized definitions)

- Perplexity (Module 2) not predicting downstream task performance is the **same underlying theme** as emergent abilities (Module 3) being partly a metric artifact, and as benchmark scores (Module 8) not predicting real-user satisfaction — in all three cases, **the proxy metric and the thing you actually care about are correlated but not identical**, and that gap is where a lot of real-world model-evaluation mistakes happen.
- The Bradley-Terry model (Module 5, reward modeling) and Elo-based human-eval aggregation (Module 8) are **mathematically the same underlying pairwise-comparison framework**, just applied at different stages of the pipeline (training signal vs. final evaluation).
- KL divergence shows up in **three unrelated-sounding places** that are worth explicitly connecting if asked: PPO's policy-drift penalty (Module 5), DPO's implicit β-weighted preference loss (Module 5, same underlying constraint reformulated), and distillation's teacher-student matching loss (Module 7) — same mathematical tool, three different purposes (constrain drift, encode implicit reward, transfer knowledge).
- MoE (Module 7) breaking the "all parameters are active per token" assumption directly means the **Chinchilla scaling-law formula `C≈6ND` (Module 3) needs adjustment for MoE models** — the effective compute-relevant parameter count is the *active* parameter count per token, not total parameters — a genuinely open research nuance worth naming if scaling laws come up in the context of a MoE architecture.

---

## 6. Final self-check — can you do all of these cold?

- [ ] Derive the DPO loss from Bradley-Terry + the KL-constrained RL optimal policy, on a whiteboard, unprompted.
- [ ] Compute LoRA's parameter reduction for a given d and r, from scratch.
- [ ] Explain why GPT-3 was undertrained relative to Chinchilla, with the actual tokens/parameter numbers.
- [ ] Explain KV caching's O(N²)→O(N) complexity shift, and estimate cache memory for a given model size/context length.
- [ ] Name and explain all three LLM-as-judge biases, with a mitigation for each.
- [ ] Explain why CLM pretraining makes hallucination a structural consequence, not an incidental bug.
- [ ] Walk through the full pretrain → SFT → RM → RLHF/DPO pipeline without skipping a stage.
- [ ] Give a balanced (not one-sided) answer to "fine-tune vs prompt vs RAG" and "RLHF vs DPO."

If any of these feel shaky, that's a pointer back to the specific module above — everything on this list is covered in full depth somewhere in Modules 1-8.

---
*End of Module 9. This completes the LLM Basics syllabus (Modules 1-9) — tokenization through interview synthesis, all at full depth with formulas, numerical examples, and standalone real-world usage notes.*
