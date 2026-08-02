# BERT, Phase by Phase — MLE Interview Guide

## The one-sentence pitch

BERT (Bidirectional Encoder Representations from Transformers) learns deep, **bidirectional** language representations by pre-training on unlabeled text with two self-supervised objectives, then gets **fine-tuned** cheaply on downstream tasks. The core insight that made it a big deal in 2018: previous models (ELMo, GPT-1) were either shallow-bidirectional (concatenating two unidirectional LSTMs) or strictly left-to-right. BERT is the first to let every token attend to every other token, in both directions, during pre-training — because it uses **masking** instead of next-token prediction.

Keep that contrast (unidirectional vs. bidirectional) in your back pocket — it's the answer to "why does BERT exist" in almost any interview.

---

## Phase 1 — Input Representation (Tokenization)

**What:** Raw text → WordPiece subword tokens → special tokens added (`[CLS]`, `[SEP]`).

**Why subwords, not words or characters:**
- Word-level vocab explodes (OOV problem — "unfriendliness" might never appear in training).
- Character-level is robust to OOV but loses semantic chunking and makes sequences very long (more compute for the same information).
- Subwords (WordPiece, ~30k vocab) are the middle ground: common words stay whole, rare words split into meaningful pieces (`"unaffordable"` → `un`, `##afford`, `##able`). Handles OOV gracefully at a fixed, manageable vocab size.

**Why `[CLS]` and `[SEP]`:**
- `[CLS]` is prepended to every input. Its final hidden state is trained to be a pooled, whole-sequence representation — used directly for classification tasks. It exists *because* Transformers have no inherent notion of "the sentence as a whole" the way an RNN's last hidden state does; you need a dedicated token whose job is to aggregate.
- `[SEP]` separates sentence A from sentence B (or marks end of a single sentence). Needed because BERT can take **sentence pairs** as input (for tasks like NLI or QA), and the model needs an explicit boundary marker since there's no sequential "end" signal like in an RNN.

**Interview trap:** People forget *why* `[CLS]` works — it's not magic, it's a consequence of training. During pre-training, `[CLS]`'s output is literally used for the NSP objective, so it's forced to learn a sentence-pair-relationship representation, which transfers well to classification.

---

## Phase 2 — Embeddings

**What:** Each input token's vector = **Token Embedding + Segment Embedding + Position Embedding** (summed, then LayerNorm + dropout).

**Why three separate embeddings:**

| Embedding | What it encodes | Why it's needed |
|---|---|---|
| Token | Identity of the subword | Basic lexical meaning |
| Segment (A/B) | Which sentence a token belongs to | Self-attention has no notion of "sentence membership" — without this, the model can't tell sentence-A tokens from sentence-B tokens once they're mixed in one sequence |
| Position | Where the token sits in the sequence | Self-attention is **permutation-invariant** by construction (it's just weighted sums over a set) — without position info, "dog bites man" and "man bites dog" would look identical to the attention mechanism |

**Why learned (not sinusoidal) position embeddings:** Unlike the original Transformer paper, BERT learns position embeddings from data rather than using fixed sinusoids. Trade-off: simpler and works well in practice, but caps max sequence length at whatever was trained (512 for base BERT) since it can't extrapolate to unseen positions.

---

## Phase 3 — The Architecture (Encoder Stack)

**What:** A stack of Transformer **encoder** blocks (12 layers/12 heads for BERT-base, 24/16 for BERT-large), each with multi-head self-attention + feed-forward, residual connections, and LayerNorm.

**Why encoder-only (not encoder-decoder, not decoder-only):**
- A decoder (like GPT) uses **causal masking** — token *i* can only attend to tokens `< i`. That's necessary for autoregressive generation but means each token's representation is starved of right-side context.
- BERT doesn't need to *generate* text, it needs to *understand* it — so there's no reason to restrict attention direction. Full bidirectional self-attention gives every token the richest possible context: past AND future words.
- This is the core architectural bet, and it's exactly why BERT can't be used directly for text generation (no causal mask = no valid way to generate left-to-right without seeing "future" tokens it's not supposed to have yet).

**Why multi-head attention specifically:** Each head can specialize in a different type of relationship (e.g., one head tracks syntactic dependencies, another tracks coreference). A single attention head averages all of this into one pattern; multiple heads let the model represent several relational patterns in parallel, then the outputs are concatenated and projected back down.

---

## Phase 4 — Pre-training Objectives

This is the heart of BERT and the most-asked interview section. Two objectives, trained **jointly**:

### 4a. Masked Language Modeling (MLM)

**What:** Randomly mask 15% of input tokens; model predicts the original token from context on both sides.

**Why masking instead of standard left-to-right LM:** This is *the* trick that enables bidirectionality. If you tried to just predict every token from its full bidirectional context, the model would trivially "see" the answer (the target token is present in the input on both sides — it's cheating, not learning). Masking removes the token so the model is forced to genuinely infer it from surrounding context.

**Why the 80/10/10 split** (of the 15% selected: 80% replaced with `[MASK]`, 10% replaced with a random token, 10% left unchanged):
- If you *always* used `[MASK]`, the model would only ever learn to produce good representations for `[MASK]` tokens — but `[MASK]` never appears at fine-tuning/inference time. That's a train/inference mismatch.
- The 10% random + 10% unchanged force the model to keep building good contextual representations for *every* token, not just masked ones, since it can never be sure which tokens are "real" vs corrupted. This closes the pre-train/fine-tune gap.

**Why only 15%, not more/less:** Too few masked tokens → training signal per pass is weak, slow convergence. Too many → not enough surrounding context left to make correct inference possible, and pre-training starts to look less like the real (mostly-unmasked) distribution the model will see at fine-tuning time.

### 4b. Next Sentence Prediction (NSP)

**What:** Feed sentence pairs (A, B); 50% of the time B truly follows A, 50% of the time B is a random sentence. Model (via `[CLS]`) predicts IsNext / NotNext.

**Why it was included:** Many downstream tasks (QA, NLI, sentence-pair classification) require understanding the *relationship between two sentences*, not just token-level understanding. MLM alone never trains the model on cross-sentence reasoning — NSP was added to explicitly bake that in, and it's also what makes the `[CLS]` embedding meaningful.

**Important interview nuance:** Later work (RoBERTa, 2019) found NSP contributes little to nothing, and removing it (while training on longer, contiguous text spans instead) actually *improved* downstream performance. Knowing this is a strong signal in an interview — it shows you understand BERT wasn't a perfectly-designed final answer, just a well-motivated first attempt, and that ablation studies are how the field actually progresses.

---

## Phase 5 — Fine-tuning

**What:** Take the pre-trained weights, add a small task-specific head (usually just one linear layer), and train the *whole* model end-to-end on labeled task data with a small learning rate for a few epochs.

**Why fine-tune the whole model (not just the new head), unlike classic transfer learning in vision:**
- Freezing the backbone and only training a linear probe works far worse for BERT than full fine-tuning, because the pre-trained representations are general-purpose, not task-specialized. Letting gradients flow through the whole network lets it *adapt* its representations to the task's specific notion of "similarity" or "relevance."
- It's cheap to do this per-task because you're not training from scratch — you're nudging an already-good representation, so it converges in a few epochs on comparatively tiny labeled datasets. This is the entire economic argument for pre-train/fine-tune as a paradigm: expensive unsupervised pre-training happens once, cheap supervised fine-tuning happens per task.

**Why minimal architecture changes per task:**
- Classification (sentiment, NLI): use `[CLS]`'s final vector → linear layer → softmax.
- Token-level tasks (NER, POS tagging): use *every* token's final vector → linear layer per token.
- QA (SQuAD-style): predict start/end position of the answer span by learning two vectors (start, end) dotted against every token's representation.
- The point being tested by interviewers here: BERT's pre-training produces a **general-purpose contextual representation**; the fine-tuning head is intentionally tiny because almost all the "knowledge" should already be in the backbone, not learned fresh per task.

---

## Why the phases exist together: the summary logic chain

1. **Tokenization** solves vocabulary/OOV problems → gives you a manageable, meaningful input unit.
2. **Embeddings** inject the information self-attention structurally lacks (order, sentence membership).
3. **Bidirectional encoder architecture** removes the directionality constraint that limited prior models.
4. **MLM** is the trick that makes bidirectional *training* possible without the model cheating.
5. **NSP** (originally) tries to add cross-sentence reasoning that MLM alone can't teach.
6. **Fine-tuning** is what makes one expensive pre-training run reusable across hundreds of cheap downstream tasks.

Each phase exists to patch a specific limitation of the phase before it (or of prior architectures). That causal chain — "X was a problem, so we did Y" — is what interviewers are actually listening for, more than the trivia itself.

---

## Rapid-fire interview Q&A

**Q: Why can't you use BERT for text generation out of the box?**
A: No causal mask — every token was trained seeing full bidirectional context, so there's no valid left-to-right generation procedure without leaking future tokens.

**Q: Why not just mask 100% and predict everything?**
A: Then every input token is `[MASK]`, so there's zero context left to predict from — you need most tokens intact so the masked ones are inferable.

**Q: What replaced NSP in later models and why?**
A: RoBERTa dropped NSP and trained on longer contiguous spans instead, finding NSP added little value and full-length inputs helped more.

**Q: Why does fine-tuning use a small learning rate?**
A: The pre-trained weights already encode good general representations; large updates would catastrophically overwrite that knowledge (catastrophic forgetting) instead of adapting it.

**Q: Why 12/24 layers specifically, why not more?**
A: Empirical compute/performance trade-off from the original paper's ablations — deeper generally helps up to a point, then returns diminish relative to compute cost (also why BERT-large exists as the "more capacity" variant).
