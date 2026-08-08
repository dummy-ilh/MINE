# LLM Terminology: End-to-End Reference

A self-contained glossary and tutorial covering every stage of the LLM lifecycle — from raw text to a deployed, aligned model. Organized so you can read top to bottom (the order things actually happen) or jump to a section for interview prep.

---

## 1. Data & Tokenization

**Corpus** — The raw text dataset used for training (e.g., Common Crawl, books, code, Wikipedia). Modern frontier models train on trillions of tokens.

**Token** — The atomic unit a model reads/writes. Not a word — usually a sub-word chunk. "unbelievable" might split into `un`, `believ`, `able`.

**Tokenizer** — The algorithm that converts text ↔ tokens. Common types:
- **BPE (Byte-Pair Encoding)** — starts with individual characters/bytes, iteratively merges the most frequent adjacent pair into a new token, until a target vocabulary size is reached.
- **WordPiece** — like BPE but picks merges that maximize likelihood of the training data, not just frequency (used in BERT).
- **SentencePiece / Unigram** — treats tokenization as a probabilistic segmentation problem; works directly on raw text (no pre-split on whitespace), so it handles languages without spaces well.

**Worked example (BPE, tiny toy corpus):**
Corpus: `low, low, lower, lowest, newer, newer`
Start: characters only → `l o w`, `l o w e r`, etc.
Step 1: most frequent adjacent pair is `(l,o)` → merge into `lo`. Now `lo w`, `lo w e r`...
Step 2: next most frequent is `(lo,w)` → merge into `low`.
Repeat until vocab size target is hit. This is exactly how GPT-style tokenizers were built, just at a scale of billions of characters.

**Vocabulary size** — Number of unique tokens the tokenizer can produce. GPT-4-class models: ~100k–200k tokens. Larger vocab = shorter sequences per sentence, but a bigger embedding table.

**Context window / context length** — Maximum number of tokens the model can attend to at once (input + output combined). E.g., 8k, 32k, 128k, 1M tokens. This is a hard architectural limit, not a soft guideline.

**Embedding** — A learned vector (e.g., 4096 numbers) representing a token's meaning. Similar tokens end up with similar vectors (cosine similarity high).

**Positional encoding** — Since attention has no inherent sense of order, position must be injected separately.
- **Sinusoidal** (original Transformer) — fixed sine/cosine functions of position.
- **Learned absolute** — a trainable embedding per position index.
- **RoPE (Rotary Position Embedding)** — rotates query/key vectors by an angle proportional to position; used in LLaMA, GPT-NeoX, most modern LLMs because it generalizes better to longer sequences than trained.
- **ALiBi** — adds a distance-proportional penalty directly to attention scores instead of modifying embeddings.

---

## 2. Core Architecture (Transformer)

**Transformer** — The architecture underlying essentially all modern LLMs (Vaswani et al., 2017). Built entirely from attention + feed-forward blocks, no recurrence.

**Self-attention** — Each token computes a weighted average of *all other tokens'* representations, where the weights are learned based on relevance.

**Query, Key, Value (Q, K, V)** — Three learned linear projections of each token's embedding.
- Query: "what am I looking for?"
- Key: "what do I contain?"
- Value: "what do I actually pass along if picked?"

**Attention score formula:**
```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) V
```
- `QKᵀ` — dot product of every query with every key → raw relevance scores.
- `/√d_k` — scaling factor (d_k = dimension of key vectors) to keep gradients stable; without it, large dot products push softmax into near-one-hot regions and gradients vanish.
- `softmax` — converts scores to a probability distribution over positions.
- multiply by `V` — weighted sum of value vectors = the output.

**Worked numerical mini-example:**
Say we have 2 tokens, d_k = 2.
Q = [[1,0],[0,1]], K = [[1,0],[0,1]], V = [[10,0],[0,20]]
QKᵀ = [[1,0],[0,1]] (identity — each token perfectly matches itself)
Scale by √2 ≈ 1.41 → [[0.71,0],[0,0.71]]
Softmax per row → row 1: softmax([0.71,0]) ≈ [0.67, 0.33]
Output row 1 = 0.67×[10,0] + 0.33×[0,20] = [6.7, 6.6]
So token 1's new representation blends mostly its own value but picks up a little from token 2 — this is attention "mixing information."

**Multi-head attention** — Instead of one attention computation, split Q/K/V into `h` heads (e.g., 32 heads of dimension 128 each, if model dim = 4096). Each head can learn a different kind of relationship (one head might track syntax, another long-range coreference). Outputs are concatenated and projected back.

**Causal mask** — In decoder-only LLMs, each token is only allowed to attend to itself and earlier tokens (upper-triangular mask set to −∞ before softmax). This enforces autoregressive generation (no peeking at the future).

**KV cache** — During generation, the Key/Value vectors for all previous tokens are cached so they don't need to be recomputed at every new step. This is *the* reason inference is fast — without it, generating token N would require reprocessing all N-1 prior tokens from scratch.

**Feed-forward network (FFN) / MLP block** — After attention, each token independently passes through a 2-layer MLP (expand to ~4x dimension, activation, project back). This is where a large fraction of a model's parameters and "knowledge storage" actually lives.

**Activation functions** — ReLU (older), GELU (BERT/GPT-2/3), SwiGLU (LLaMA, PaLM — a gated variant that empirically improves quality per parameter).

**Layer normalization (LayerNorm) / RMSNorm** — Normalizes activations to stabilize training. RMSNorm (used in LLaMA) is a simplified, cheaper variant that skips mean-centering.

**Residual connection (skip connection)** — Output of each block is `x + Sublayer(x)`, not just `Sublayer(x)`. Lets gradients flow directly through many layers, which is what makes 100+ layer networks trainable at all.

**Pre-norm vs post-norm** — Whether LayerNorm is applied before or after the sublayer. Pre-norm (now standard) gives more stable training at scale.

**Encoder vs decoder vs encoder-decoder:**
- **Encoder-only** (BERT) — bidirectional attention, good for understanding/classification, not generation.
- **Decoder-only** (GPT, LLaMA, Claude) — causal attention, generates text left-to-right. This is the dominant LLM architecture today.
- **Encoder-decoder** (T5, BART) — encoder reads full input, decoder generates output attending back to it (cross-attention). Common for translation/summarization.

**Cross-attention** — Decoder attends to encoder outputs (Q from decoder, K/V from encoder). Absent in decoder-only models.

**Mixture of Experts (MoE)** — Instead of one dense FFN per layer, have several "expert" FFNs and a router that sends each token to only a few (e.g., 2 of 8). Increases total parameter count without proportionally increasing compute per token (sparse activation). Used in Mixtral, GPT-4 (reportedly), Switch Transformer.

**Parameters** — The learned weights of the model. "7B model" = 7 billion parameters. Roughly correlates with capability but not perfectly (data quality and architecture matter too).

---

## 3. Training: Pretraining

**Pretraining** — Training the base model from scratch (random init) on massive unlabeled text with a self-supervised objective. This is where "world knowledge" and language ability come from.

**Next-token prediction / causal language modeling (CLM)** — The pretraining objective for decoder-only models: predict token t+1 given tokens 1..t. Loss = cross-entropy between predicted distribution and actual next token.

**Masked language modeling (MLM)** — BERT's objective: randomly mask ~15% of tokens, predict them using bidirectional context. Good for understanding tasks, not for generation.

**Cross-entropy loss:**
```
L = − Σ y_i · log(ŷ_i)
```
where y is the true one-hot next-token distribution and ŷ is the model's predicted probability distribution. Since y is one-hot, this simplifies to `−log(ŷ_correct_token)` — you're just penalizing low probability assigned to the actual next word.

**Perplexity** — `exp(cross-entropy loss)`. Interpretable as "the effective number of equally-likely choices the model was confused between." Perplexity 1 = perfect prediction; perplexity of 50 means roughly as confused as choosing uniformly among 50 options. Lower is better.

**Scaling laws** — Empirical relationships (Kaplan et al. 2020, Chinchilla/Hoffmann et al. 2022) showing loss decreases as a power law with more parameters, more data, and more compute — *if* they're scaled together correctly.

**Chinchilla-optimal** — The Hoffmann et al. finding that most pre-2022 LLMs were "over-parameterized, under-trained" — for a fixed compute budget, you get a better model by training a smaller model on more tokens. Rule of thumb: ~20 tokens per parameter.

**Compute (FLOPs)** — Total floating-point operations used for training. Roughly `FLOPs ≈ 6 × params × tokens` for training a dense transformer.

**Emergent abilities** — Capabilities (e.g., multi-step arithmetic, chain-of-thought reasoning) that appear abruptly at certain scale thresholds rather than improving smoothly — a debated but widely observed phenomenon.

---

## 4. Training: Adapting the Base Model

**Base model** — The raw pretrained model. Completes text but isn't tuned to follow instructions or converse — e.g., asked "What's the capital of France?" it might continue with more trivia questions rather than answering.

**Fine-tuning** — Continuing training on a smaller, task/domain-specific dataset, updating some or all weights.

**Supervised fine-tuning (SFT)** — Fine-tuning on (prompt, ideal response) pairs written/curated by humans. Turns a base model into an "instruction-following" model.

**Instruction tuning** — SFT specifically on diverse (instruction, response) pairs across many task types, so the model generalizes to following novel instructions, not just memorizing specific tasks.

**RLHF (Reinforcement Learning from Human Feedback)** — 3-stage pipeline:
1. SFT model as starting point.
2. Train a **reward model**: humans rank multiple model outputs for the same prompt; reward model learns to predict human preference scores.
3. Use RL (typically **PPO**, Proximal Policy Optimization) to fine-tune the SFT model to maximize reward-model score, with a KL-divergence penalty against the original SFT model so it doesn't drift too far or "reward hack."

**Reward model** — A separate model (often initialized from the SFT model) trained to output a scalar score for how good a response is, from pairwise human comparisons (Bradley-Terry model is the typical loss formulation).

**PPO (Proximal Policy Optimization)** — The RL algorithm typically used in RLHF; clips the policy update so it doesn't change too drastically in one step, keeping training stable.

**DPO (Direct Preference Optimization)** — A newer, simpler alternative to RLHF that skips training a separate reward model and skips RL entirely — it directly optimizes the policy on preference pairs using a closed-form loss derived to be mathematically equivalent to the RLHF objective under mild assumptions. Much cheaper and more stable to train than PPO-based RLHF.

**RLAIF (RL from AI Feedback)** — Same idea as RLHF, but an AI model generates the preference labels instead of humans (used e.g. in Constitutional AI).

**Constitutional AI** — A technique (Anthropic) where a model critiques and revises its own outputs against a written set of principles ("constitution"), generating training data for itself, reducing reliance on large-scale human labeling for harmlessness.

**Catastrophic forgetting** — When fine-tuning on new data causes a model to lose previously learned capabilities. Mitigated with replay buffers, low learning rates, or regularization toward the original weights.

**PEFT (Parameter-Efficient Fine-Tuning)** — Family of methods that fine-tune a small fraction of parameters instead of the whole model.
- **LoRA (Low-Rank Adaptation)** — Freezes original weight matrix `W`, learns a low-rank update `ΔW = A·B` where A is (d×r) and B is (r×d), r << d (e.g., r=8 vs d=4096). Only A and B are trained — orders of magnitude fewer parameters. At inference, `W' = W + A·B` (can be merged, adding zero latency).
- **QLoRA** — LoRA on top of a 4-bit quantized frozen base model — lets you fine-tune a 65B model on a single consumer GPU.
- **Adapters** — Small bottleneck FFN layers inserted between transformer blocks; only these are trained.
- **Prefix/Prompt tuning** — Learn a small set of continuous "virtual tokens" prepended to the input; the rest of the model is frozen.
- **BitFit** — Only fine-tune the bias terms of the network — extremely parameter-efficient, weaker than LoRA generally.
- **IA3** — Learns per-channel rescaling vectors applied to activations; fewer parameters than LoRA, competitive on some tasks.

**LoRA numerical example:** For a weight matrix W of shape 4096×4096 (16.7M params), full fine-tuning updates all 16.7M values. LoRA with rank r=8: A is 4096×8, B is 8×4096 → 2×(4096×8) = 65,536 trainable params — about **0.4%** of the original.

**Model merging / task arithmetic** — Combining multiple fine-tuned models (or their weight deltas, "task vectors") via arithmetic (e.g., averaging, or adding a "task vector" to add a capability, subtracting one to remove it) without further training.

**Domain adaptation** — Adjusting a model trained on one distribution to perform well on another, related distribution. Techniques: DANN (adversarial domain-invariant feature learning), CORAL (aligning feature covariances between domains), pseudo-labeling (using the model's own confident predictions on target-domain data as extra training labels).

**Transfer learning** — General principle underlying all of the above: knowledge learned on one task/domain is reused to speed up or improve learning on another.

**Multi-task learning** — Training one model jointly on multiple tasks/objectives simultaneously (shared backbone, task-specific heads), as opposed to sequential transfer.

---

## 5. Prompting & In-Context Behavior

**Prompt** — The input text given to the model.

**Zero-shot** — Asking the model to do a task with no examples, just an instruction.

**Few-shot / in-context learning (ICL)** — Providing a handful of (input, output) examples in the prompt itself; the model infers the task pattern without any weight updates. Remarkable because *no gradient step happens* — the "learning" is purely a function of what's in the context window at inference time.

**Chain-of-thought (CoT) prompting** — Asking the model to produce intermediate reasoning steps before the final answer ("Let's think step by step"), which measurably improves accuracy on multi-step reasoning tasks.

**Self-consistency** — Sampling multiple CoT reasoning paths and taking a majority vote on the final answer — improves reliability over a single greedy chain.

**System prompt** — A special instruction block (not visible to the end user, or shown separately) that sets persona, rules, or context for the whole conversation.

**Prompt injection** — An attack where malicious instructions are embedded in content the model processes (a webpage, document, tool result), attempting to override the original instructions.

**Jailbreak** — A prompt engineered to bypass a model's safety training and elicit disallowed behavior.

---

## 6. Inference & Decoding

**Autoregressive generation** — Producing tokens one at a time, each conditioned on all previous tokens (including ones just generated).

**Logits** — The raw, unnormalized scores the model outputs for each vocabulary token before softmax.

**Softmax with temperature:**
```
P(token_i) = exp(logit_i / T) / Σ_j exp(logit_j / T)
```
- **T = 1** — standard softmax.
- **T < 1** (e.g., 0.3) — sharpens the distribution, makes output more deterministic/repetitive.
- **T → 0** — approaches greedy decoding (always pick the single highest-logit token).
- **T > 1** — flattens distribution, more random/diverse (and more error-prone) output.

**Greedy decoding** — Always pick the single highest-probability next token. Fast, deterministic, but often produces bland or repetitive text.

**Beam search** — Keep the top-k partial sequences ("beams") at each step instead of just one, expanding each and pruning back to k. Better for tasks with one "correct" answer (translation); tends to produce generic/repetitive text for open-ended generation.

**Top-k sampling** — Restrict sampling to the k highest-probability tokens, renormalize, then sample. E.g., k=40.

**Top-p (nucleus) sampling** — Restrict sampling to the smallest set of tokens whose cumulative probability ≥ p (e.g., p=0.9), then renormalize and sample. Adapts the candidate pool size to the model's confidence (narrow when confident, wide when uncertain) — generally preferred over fixed top-k.

**Repetition penalty / frequency penalty** — Downweights logits of tokens that have already appeared, to reduce loops/repetition.

**Speculative decoding** — Use a small, fast "draft" model to propose several tokens ahead, then have the large target model verify them all in a single forward pass, accepting the matching prefix. Speeds up generation without changing the output distribution, because verification is exact.

**Stop sequence / EOS token** — A special token or string signaling generation should end.

**Streaming** — Returning tokens to the user as they're generated rather than waiting for the full response.

---

## 7. Efficiency & Deployment

**Quantization** — Reducing the numerical precision of weights/activations (FP32 → FP16/BF16 → INT8 → INT4) to shrink memory footprint and speed up inference, at some accuracy cost.
- **Post-training quantization (PTQ)** — quantize an already-trained model, no retraining.
- **Quantization-aware training (QAT)** — simulate quantization during training so the model adapts to the lower precision.

**Distillation (knowledge distillation)** — Train a smaller "student" model to mimic a larger "teacher" model's output distribution (soft labels), often recovering much of the teacher's quality at a fraction of the size/cost.

**Pruning** — Removing weights, neurons, or attention heads that contribute little to output quality, to shrink the model.

**Small Language Model (SLM)** — A model (roughly <10B params) designed to run efficiently on-device or at low cost, often built via distillation or aggressive pretraining-data curation (e.g., Phi, Gemma-2B) rather than sheer scale.

**Batching** — Processing multiple requests together to better utilize GPU parallelism; **continuous batching** dynamically adds/removes sequences from a running batch as they finish, rather than waiting for a fixed batch to fully complete.

**Model parallelism / tensor parallelism / pipeline parallelism** — Ways to split a model too large for one GPU across multiple GPUs (splitting individual weight matrices vs. splitting layers across devices, respectively).

**Latency vs. throughput** — Latency = time to get one response; throughput = total tokens/sec served across all users. Optimizing one often trades off against the other.

---

## 8. Evaluation

**Benchmark** — A standardized test set used to compare models, e.g., MMLU (broad knowledge), HumanEval (code), GSM8K (grade-school math), HellaSwag (commonsense).

**Hallucination** — When a model generates fluent but factually incorrect or unsupported content, stated with unwarranted confidence.

**Calibration** — Whether a model's stated confidence matches its actual accuracy (a well-calibrated model that says "90% confident" is right about 90% of the time).

**BLEU / ROUGE** — N-gram overlap metrics comparing generated text to reference text; common for translation (BLEU) and summarization (ROUGE), though poorly correlated with human judgment for open-ended generation.

**LLM-as-judge** — Using a strong LLM to score/compare outputs of other models, as a cheaper proxy for human evaluation.

**Red-teaming** — Deliberately probing a model for harmful, biased, or unsafe outputs before deployment.

---

## 9. Retrieval, Agents & Tool Use

**RAG (Retrieval-Augmented Generation)** — Retrieve relevant documents from an external knowledge base (via embedding similarity search) and insert them into the prompt, so the model can ground its answer in up-to-date or proprietary information it wasn't trained on.

**Vector database** — A database (e.g., Pinecone, FAISS, Weaviate) optimized for storing embeddings and doing fast approximate nearest-neighbor search over them.

**Function calling / tool use** — The model outputs a structured request (function name + arguments) instead of natural language, which the calling application executes and feeds results back — the mechanism behind agents, calculators, search integration, etc.

**Agent / agentic loop** — A model that iteratively plans, calls tools, observes results, and decides on next actions/termination, rather than producing one single response.

**Context stuffing / long-context** — Simply putting a large amount of relevant material directly in the prompt (as context windows have grown to 100k–1M tokens), sometimes as an alternative to RAG.

---

## 10. Multimodality & Beyond

**Multimodal model** — Processes/generates more than one modality (text, image, audio, video) within one architecture, typically by projecting non-text inputs into the same embedding space the transformer operates on (e.g., a vision encoder's patch embeddings fed in alongside text token embeddings).

**Vision encoder (e.g., ViT)** — Splits an image into patches, embeds each patch like a "token," so a transformer can attend over image content the same way it attends over text.

**Diffusion model** — A different generative paradigm (used for images, and increasingly explored for text) that learns to reverse a gradual noising process — distinct from autoregressive LLMs, though the two are increasingly combined/compared in research.

---

## Quick Interview Cheat-Sheet (rapid recall)

| Ask | One-line answer |
|---|---|
| Why scale QKᵀ by √d_k? | Keeps dot products from growing with dimension, avoiding vanishing gradients in softmax |
| Why causal mask? | Enforces autoregressive property — no attending to future tokens |
| KV cache purpose? | Avoid recomputing past K/V at every generation step — makes inference tractable |
| RLHF vs DPO? | RLHF trains a reward model + PPO; DPO optimizes preferences directly, no RL loop |
| LoRA idea in one line? | Freeze W, learn a low-rank update ΔW = AB, train only A and B |
| Top-p vs top-k? | Top-p adapts candidate pool size to model confidence; top-k is a fixed cutoff |
| Chinchilla finding? | Most early LLMs were under-trained relative to their size; ~20 tokens/parameter is compute-optimal |
| Why RMSNorm over LayerNorm? | Cheaper (skips mean-centering), similar stability benefit |
| MoE benefit? | More total parameters/capacity without proportional increase in per-token compute |

---

*This document is designed to be self-contained — re-read top to bottom for full-picture understanding, or use the cheat-sheet for rapid pre-interview review.*
