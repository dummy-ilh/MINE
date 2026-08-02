# BERT & DistilBERT — Full Interview Question Bank (Google / Apple MLE style)

Eighth companion doc. A consolidated question bank spanning everything in the earlier docs plus DistilBERT specifically, organized the way these two companies tend to lean: Google interviews skew toward research depth and "why did the original authors make this choice," Apple interviews skew toward production/on-device efficiency (latency, memory, quantization) given Siri and on-device ML. Sections are ordered roughly fundamentals → depth → DistilBERT → production, so you can stop wherever your prep is already solid.

---

## Section 1 — BERT Fundamentals

**Q: What does BERT stand for and what problem does it solve?**
A: Bidirectional Encoder Representations from Transformers. It solves the problem that prior contextual embedding methods (GPT-1, ELMo) were either strictly unidirectional or only shallowly bidirectional (concatenated separate left-to-right and right-to-left models) — BERT lets every token attend to full left and right context simultaneously, via masked language modeling.

**Q: Why can't you train a genuinely bidirectional model with standard left-to-right language modeling?**
A: If every token could see the whole sequence including itself while trying to predict itself, the task is trivial — the answer is literally in the input. Masking removes the target token so bidirectional context has to be used to infer it, not read it directly.

**Q: What are the two pre-training objectives and why both?**
A: Masked Language Modeling (MLM) for token-level bidirectional understanding, and Next Sentence Prediction (NSP) for cross-sentence relationship understanding needed by tasks like QA and NLI. (Caveat worth stating: RoBERTa later showed NSP contributes little — see Section 3.)

**Q: Walk me through the 80/10/10 masking rule.**
A: Of the 15% of tokens selected for masking: 80% replaced with `[MASK]`, 10% replaced with a random token, 10% left unchanged. This closes the train/inference mismatch — `[MASK]` never appears at inference time, so the model must learn good representations for every token, not just ones it knows are corrupted.

**Q: What is WordPiece tokenization and why not word-level or character-level?**
A: Subword tokenization that splits rare words into meaningful pieces while keeping common words whole. Word-level has an unbounded vocabulary / OOV problem; character-level avoids OOV but produces very long sequences and loses semantic chunking. WordPiece is the practical middle ground.

**Q: What is `[CLS]` and why does it work as a sequence representation?**
A: A special token prepended to every input; during pretraining its final hidden state is trained (via NSP) to summarize the whole sequence, so it transfers naturally to classification tasks at fine-tuning time.

---

## Section 2 — Architecture Depth

**Q: What are the three embedding types and why sum them instead of concatenating?**
A: Token, segment, and position embeddings. They're summed (not concatenated) so the output stays at `hidden_size` (768) rather than growing — the model has enough capacity to disentangle the combined signal, and this keeps every layer at a uniform width.

**Q: Why self-attention over RNNs for this task?**
A: Self-attention gives O(1) path length between any two tokens regardless of distance, versus O(n) sequential steps in an RNN — better for long-range dependencies and fully parallelizable during training (no sequential recurrence).

**Q: Why multiple attention heads instead of one larger head?**
A: Multiple heads let the model represent several distinct relational patterns in parallel (e.g. one head tracking coreference, another tracking local syntax); a single head of the same total width tends to collapse toward one dominant pattern rather than cleanly separating concerns.

**Q: Why is there a `1/sqrt(d_k)` scaling factor in attention?**
A: Dot products grow in magnitude with dimensionality; without scaling, softmax saturates into a near-one-hot distribution with vanishing gradients almost everywhere except the top score. Scaling keeps the score variance roughly constant regardless of head dimension.

**Q: Why do you need a feed-forward sublayer at all if you already have attention?**
A: Attention is a weighted average — a linear operation. Stacked attention alone never introduces a nonlinear transformation of the content; the FFN is where per-token nonlinear processing actually happens. Attention routes information, FFN transforms it.

**Q: Why residual connections?**
A: They fix the degradation problem in deep networks (adding depth makes optimization harder, separate from vanishing gradients) by giving each sublayer an easier job — learn a correction to the input, not a full reconstruction — and give gradients an unobstructed path back to earlier layers.

**Q: Why LayerNorm instead of BatchNorm?**
A: LayerNorm normalizes per-token across features, independent of batch composition — robust to variable sequence length and small/uneven batch sizes, unlike BatchNorm which relies on stable cross-example batch statistics.

**Q: Post-norm vs pre-norm — which does BERT use, and what's the trade-off?**
A: BERT uses post-norm (`LayerNorm(x + Sublayer(x))`). It works at BERT's 12-24 layer depth but becomes harder to train stably at much greater depth; later architectures moved to pre-norm (`x + Sublayer(LayerNorm(x))`) for more stable very-deep training.

**Q: Where do most of BERT's parameters actually live?**
A: The feed-forward sublayers — roughly 2x the parameter count of the attention projections per layer, since FFN scales with `d_model × d_ff` (4x expansion) while attention scales with `d_model²`. Commonly misjudged since attention gets the architectural spotlight.

**Q: Why is BERT capped at 512 tokens?**
A: Quadratic O(n²) attention cost, plus learned (non-extrapolating) position embeddings only trained up to position 512 — the model has no representation for positions beyond that without retraining.

---

## Section 3 — Known Limitations & What Later Work Fixed (strong signal in a Google-style interview)

**Q: What did RoBERTa change, and why?**
A: Removed NSP (found to add little to nothing), trained on longer contiguous text spans, used dynamic masking (different mask pattern each epoch instead of static), larger batches, more data, longer training. Demonstrates BERT was undertrained relative to its own architecture's capacity.

**Q: What did ALBERT change, and why?**
A: Factorized embedding parameterization (decouples vocabulary embedding size from hidden size) and cross-layer parameter sharing, both aimed at reducing parameter count without proportionally hurting performance — addresses the "parameters mostly go to redundant per-layer weights" observation.

**Q: What did ELECTRA change, and why?**
A: Replaced MLM with "replaced token detection" — a small generator proposes plausible token replacements, and the main model learns to detect which tokens were replaced. This uses *every* token as a training signal (not just the 15% masked), making pretraining far more sample-efficient.

**Q: If asked "what would you change about BERT today," what's a strong answer?**
A: Pre-norm instead of post-norm for deeper/more stable training, drop NSP per RoBERTa's findings, and use a linear/sparse attention variant to remove the quadratic sequence-length ceiling — shows you know the specific, documented follow-up work rather than giving a vague "make it bigger" answer.

---

## Section 4 — Fine-Tuning Practice

**Q: What learning rate range do you fine-tune BERT with, and why so small relative to training from scratch?**
A: `2e-5`–`5e-5`. Pretrained weights already encode useful structure; large steps risk catastrophic forgetting of that structure before task-specific adaptation happens.

**Q: Why so few epochs (2-4) compared to training from scratch?**
A: BERT-base has ~110M parameters — more than enough capacity to memorize a small fine-tuning set within a couple epochs, after which further training just overfits. Watch validation loss, use early stopping.

**Q: How do you decide how many layers to unfreeze?**
A: Primarily driven by dataset size — freeze more on small data (head-only or top few layers for <1k examples), full fine-tune on large data (>100k examples), with progressive unfreezing as an empirical middle-ground approach that stops as soon as validation loss stops improving after an unfreeze step.

**Q: How do you handle class imbalance?**
A: Inverse-frequency class weighting in the loss function, or oversampling/undersampling, or focal loss for extreme imbalance — and report F1/precision-recall rather than accuracy, since accuracy hides majority-class-only prediction on imbalanced data.

**Q: `[CLS]` vs mean pooling for classification — which and why?**
A: `[CLS]` is the standard default (pretrained via NSP to summarize the sequence); mean pooling over all tokens is a reasonable alternative/ablation, sometimes performing better for similarity-style tasks (cf. Sentence-BERT).

---

## Section 5 — DistilBERT: What It Is and Why It Exists

**Q: What is DistilBERT, in one sentence?**
A: A smaller, faster version of BERT produced via knowledge distillation — a 6-layer student model trained to mimic a 12-layer BERT teacher's output distribution, retaining ~97% of BERT's performance at roughly 60% of the size and 60% faster inference.

**Q: What problem does DistilBERT solve that BERT itself doesn't?**
A: Deployment cost. BERT-base at 110M parameters is expensive to run at low latency or on resource-constrained devices (mobile, edge, high-QPS production services). DistilBERT targets the latency/memory/cost side of the trade-off, accepting a small accuracy hit.

**Q: What's removed/changed structurally from BERT to get DistilBERT?**
A: Layer count halved (6 instead of 12) — width (hidden_size=768) and number of attention heads (12) stay the same as BERT-base. Token-type embeddings are removed (found to contribute little). The pooler layer is removed. NSP is dropped entirely from pretraining.

**Q: Why halve depth specifically, rather than reduce width (hidden_size) instead?**
A: The DistilBERT authors found reducing depth hurts performance less than reducing width for a given parameter budget — layers appear to have more redundancy than the hidden dimension does, so removing every other layer (initializing the student from alternating teacher layers) preserves more capability per parameter removed.

**Q: How is DistilBERT initialized before distillation training starts?**
A: From every other layer of the pretrained BERT teacher (e.g. teacher layers 1, 3, 5, 7, 9, 11 seed the 6 student layers) rather than random initialization — gives the student a strong starting point instead of learning the distillation task from scratch.

---

## Section 6 — Knowledge Distillation Mechanics (the part people fumble)

**Q: What is the distillation loss, conceptually?**
A: Instead of (or in addition to) training the student against hard ground-truth labels, you train it to match the teacher's **full output probability distribution** ("soft labels") — not just the teacher's top prediction, but the relative probabilities it assigns to every class/token.

**Q: Why match the full distribution instead of just the teacher's top prediction?**
A: The relative probabilities across non-top classes carry information Hinton's original distillation paper calls "dark knowledge" — e.g. a teacher predicting "cat" but assigning meaningfully more probability to "dog" than to "car" is expressing a similarity structure between classes that a single hard label throws away. Training the student to match this full distribution transfers more of what the teacher learned.

**Q: What is temperature in distillation, and what does it do numerically?**
A: A scalar `T` used to soften the softmax (`softmax(logits / T)`) before computing the distillation loss (typically KL divergence between teacher and student softened distributions). Higher `T` flattens the distribution, revealing more of the "dark knowledge" in the smaller probabilities; `T=1` is standard softmax.

Worked example — teacher logits `[4.0, 1.0, 0.2]`, student logits `[3.0, 1.5, 0.5]`:

| Temperature | Teacher probs | Student probs | KL divergence |
|---|---|---|---|
| T=1 | [0.933, 0.046, 0.021] | [0.766, 0.171, 0.063] | 0.0999 |
| T=2 | [0.728, 0.163, 0.109] | [0.569, 0.269, 0.163] | 0.0552 |
| T=4 | [0.538, 0.254, 0.208] | [0.450, 0.309, 0.241] | 0.0157 |

Reading this: at `T=1`, the teacher's distribution is sharply peaked (93% on the top class) and the small gap between classes 2 and 3 is barely visible. At `T=4`, both distributions flatten out and the *relative ordering and ratios* between all three classes become much more visible to compare — this is what "temperature reveals dark knowledge" looks like numerically. Note KL divergence also shrinks as T grows here because both distributions flatten toward uniform, converging toward each other — in practice the distillation loss is typically rescaled by `T²` to keep gradient magnitudes comparable across different temperature choices.

**Q: What is DistilBERT's actual training loss — is it purely distillation?**
A: No — a **triple loss**: (1) the distillation loss (KL divergence between softened teacher/student output distributions, on the MLM task), (2) the standard supervised MLM loss against true masked-token labels (hard labels), and (3) a cosine embedding loss that aligns the direction of the student's and teacher's hidden state vectors. Combining these was found to outperform distillation loss alone.

**Q: Why include the hard-label MLM loss at all if you're already distilling from the teacher?**
A: The teacher isn't perfect — it can be confidently wrong on some examples. Blending in the ground-truth hard-label loss anchors the student to actual correct answers rather than purely mimicking the teacher's imperfections, which softens the "student inherits the teacher's mistakes" risk.

**Q: Why the cosine embedding loss specifically, on top of the other two?**
A: It directly encourages the student's internal representations to point in the same *direction* in vector space as the teacher's, not just produce similar final output probabilities — a more direct alignment of internal representations, found empirically to improve downstream transfer.

---

## Section 7 — BERT vs DistilBERT, Side by Side

| Property | BERT-base | DistilBERT |
|---|---|---|
| Layers | 12 | 6 |
| Hidden size | 768 | 768 (unchanged) |
| Attention heads | 12 | 12 (unchanged) |
| Parameters | ~110M | ~66M (~40% smaller) |
| Token-type embeddings | Yes | Removed |
| Pooler layer | Yes | Removed |
| Pretraining objective | MLM + NSP | MLM (distillation) + hard-label MLM + cosine loss; no NSP |
| Inference speed | Baseline | ~60% faster |
| Benchmark performance (GLUE, roughly) | Baseline | ~97% of BERT's score |

**Q: If DistilBERT is only 40% smaller but 60% faster, why the mismatch?**
A: Depth reduction disproportionately helps latency because inference cost is roughly linear in layer count for sequential layer-by-layer computation — halving layers roughly halves the sequential compute path, while parameter count reduction also includes non-depth-related removals (pooler, token-type embeddings) that don't scale the same way.

**Q: When would you choose DistilBERT over BERT, and when would you NOT?**
A: Choose DistilBERT when latency/memory/cost constraints are real and binding (mobile/edge deployment, high-QPS services, tight SLAs) and the ~3% performance gap is acceptable for the task. Stick with full BERT (or a larger model) when the task is performance-critical and compute isn't the bottleneck, or when the task specifically benefits from NSP-style cross-sentence reasoning that DistilBERT's pretraining doesn't include.

---

## Section 8 — Production / On-Device Angle (Apple-flavored)

**Q: Beyond distillation, what other techniques reduce BERT's deployment footprint?**
A: **Quantization** (reducing weight precision, e.g. fp32 → int8, shrinking memory footprint and often speeding up inference on supporting hardware with minor accuracy loss), **pruning** (removing individual weights or whole attention heads found to contribute little), and **ONNX/CoreML conversion** for optimized on-device runtime execution — these are often combined with distillation rather than used as alternatives to it.

**Q: How would you decide between quantizing BERT-base vs. using DistilBERT vs. doing both, for an on-device deployment?**
A: Depends on the specific latency/memory budget and where the bottleneck actually is — quantization mainly shrinks memory/bandwidth and can speed up inference on hardware with efficient low-precision kernels but doesn't reduce sequential compute depth; distillation reduces depth and thus the sequential compute path itself. In practice, combining both (a distilled model, then quantized) is common when the budget is tight, since the two techniques target different bottlenecks and compound rather than overlap.

**Q: What's a concrete risk of deploying a distilled/quantized model that a team might miss?**
A: Aggregate benchmark performance (e.g. overall GLUE score, or overall accuracy) can look acceptable while performance degrades unevenly across subgroups or edge cases the compressed model handles worse than the original — always validate the compressed model's per-class/per-subgroup metrics, not just the aggregate number, especially for anything user-facing.

**Q: Why might on-device inference favor a fixed, short max sequence length even more strongly than server-side deployment?**
A: Attention's quadratic cost in sequence length hits memory-constrained mobile hardware harder — a server can often afford headroom that a phone's available RAM and thermal/battery budget can't, making the "pick max_length from your data's actual distribution, not BERT's max of 512" practice (from the hyperparameters doc) even more important on-device.

---

## Section 9 — Rapid-Fire / Gotcha Round

**Q: Can BERT generate text?**
A: Not natively — no causal mask, so there's no valid left-to-right generation procedure without leaking future tokens the model was trained to see.

**Q: Is DistilBERT trained from scratch or distilled from a specific BERT checkpoint?**
A: Distilled from a specific pretrained BERT-base teacher, with the student initialized from every other layer of that teacher rather than trained from random init.

**Q: True or false: DistilBERT uses NSP during its training.**
A: False — NSP is dropped entirely; DistilBERT's pretraining loss is MLM-distillation + hard-label MLM + cosine embedding loss.

**Q: Does distillation temperature affect inference, or only training?**
A: Only training (specifically, only the distillation loss computation) — at inference/deployment time, the student runs standard `T=1` softmax like any other classifier.

**Q: What's the single biggest lever for reducing BERT's inference latency: quantization, distillation, or reducing max sequence length?**
A: Depends on where your actual bottleneck is — if you're padding short text to a needlessly long max_length, that's often the cheapest, biggest win (quadratic savings) before reaching for a different model. If sequence length is already tight and appropriate, distillation (fewer sequential layers) and quantization (lower-precision compute/memory) become the next levers, often combined.

**Q: In one sentence, why does DistilBERT keep hidden_size and head count the same as BERT-base but cut layers in half?**
A: Because the authors found depth is more redundant than width for a fixed parameter budget — cutting width hurt performance more than cutting layers, for the same overall size reduction.
