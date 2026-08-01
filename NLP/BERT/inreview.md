# BERT Cheatsheet — Apple MLE Interview, Last-Mile Review

*Companion to the full Chapter 1-9 master notes. This is the compressed version — three sections: (1) every layer's purpose + failure mode, (2) a pure dimension trace, (3) a few-page pre-interview speed-read.*

---

# Part 1 — Every Layer: Purpose, and What Breaks If You Remove It

## 1.0 Pre-model components (not "layers," but load-bearing)

| Component | Purpose | If removed / skipped |
|---|---|---|
| **WordPiece tokenizer** | Splits text into ~30,522 subword units; balances vocab size vs. sequence length; handles OOV via decomposition | Word-level → unbounded vocab, OOV fails hard. Char-level → sequence length inflates 3-4x, attention cost inflates ~10-16x (quadratic) |
| **`[CLS]` token** | Prepended slot with no inherent meaning; becomes the sentence-level summary after 12 layers of attention | No dedicated, purpose-trained aggregation point; would need ad hoc pooling (e.g. mean over tokens), not specifically optimized for it |
| **`[SEP]` token** | Marks segment boundaries between sentence A/B | Model must infer boundaries from content alone — strictly harder, no explicit signal |
| **`[MASK]` token (pre-train only)** | Placeholder for hidden tokens during MLM | No mechanism to create a "predict this" target while keeping the input non-trivial |
| **`[PAD]` + attention mask** | Batches need equal-length sequences; the *mask* (not the pad ID itself) excludes padding from softmax | Without the mask specifically: real tokens attend into meaningless padding, corrupting `[CLS]` and all representations |

## 1.1 The three embedding tables (Chapter 3)

| Layer | Shape | Purpose | If removed |
|---|---|---|---|
| Token embedding | `[30,522 × 768]` | Word/subword identity — the "default meaning" before context | No word meaning at all — model has nothing to look up |
| Position embedding | `[512 × 768]` | Injects order — self-attention alone is permutation-invariant | "the cat sat" = "sat cat the" — Bag-of-Words-level blindness to order |
| Segment embedding | `[2 × 768]` | Marks which of two sentences a token belongs to | Model loses explicit "which sentence" signal (redundant with `[SEP]`, but still a real loss) |
| **Sum + LayerNorm** | `[n × 768]` | Combines all three into one vector per token, keeps scale stable | Concatenating instead would ~triple every downstream layer's width/cost; skipping LayerNorm here risks scale drift into block 1 |

## 1.2 Inside each of the 12 Transformer blocks (Chapters 4-7)

| Sub-layer | Shape (BERT-base) | Purpose | If removed |
|---|---|---|---|
| **Q/K/V projections** (×12 heads) | `X[n×768]·W[768×64]` → `[n×64]` each | Project into query ("what am I seeking"), key ("what I broadcast"), value ("what I hand over") | Tying Q=K forces a *symmetric* attention matrix — can't represent directional relationships (e.g. verb→subject ≠ subject→verb) |
| **Self-attention** (per head) | `softmax(QKᵀ/√d_k)·V` → `[n×64]` | Every token computes a weighted blend of every other token's Value, weighted by Query·Key relevance | No context mixing at all — every token stays its static, identity-only embedding forever |
| **√d_k scaling** | scalar divide | Cancels dot-product variance growth (∝ d_k) so softmax doesn't saturate | Raw scores grow with dimension → softmax → near one-hot → gradient ≈ 0, learning stalls |
| **Concat heads + W_O** | `[n×768]·[768×768]` → `[n×768]` | Preserves each head's undiluted signal, then learns to mix across heads | Averaging instead of concat forces an immediate, fixed, undifferentiated blend before any learning can weigh heads differently |
| **Residual 1** | `x + Attention(x)` | Guarantees `∂output/∂x ≥ 1` — unshrinking gradient path back to early layers | Vanishing gradients — 0.7 local grad × 24 sub-layers ≈ 0.02% surviving signal, early layers stop learning |
| **LayerNorm 1** | per-token mean-0/std-1 + learned γ,β | Resets scale after the residual add, bounding forward-pass growth | Nothing caps activation magnitude — grows ~2.5x (uncorrelated case) to ~12x+ (correlated case) over 24 additions |
| **FFN** | `768→3072→768` (GELU in between) | Per-token nonlinear transformation; **holds ~52% of BERT's total parameters** (more than attention's ~26%) | Model loses most of its actual representational capacity — attention alone just does weighted averaging, no real "thinking" per token |
| **Residual 2 + LayerNorm 2** | same pattern as 1 | Same reasons, around the FFN sub-layer | Same failure modes, around the FFN instead |

**Repeat all of the above ×12** — each block has its own independently-learned weights (nothing is shared block-to-block except the original embedding tables, which are only used once at the bottom).

## 1.3 After the stack (Chapters 8-10)

| Layer | Shape | Purpose | If removed |
|---|---|---|---|
| **Pooler** | `Linear(768→768) + tanh` on `[CLS]` only | Extra transform specifically for sentence-level classification use of `[CLS]` | Downstream classifiers use raw `[CLS]` hidden state directly — usually fine, pooler is a minor refinement, not load-bearing |
| **MLM head** (pretrain only) | `Linear(768→768)→GELU→LayerNorm→[tied decoder]→30,522` | Predicts masked tokens; decoder is weight-tied to the input embedding table (saves ~23.4M params) | No pre-training signal at all for token-level understanding |
| **NSP head** (pretrain only, later dropped by RoBERTa) | `Linear(768→2)` on `[CLS]` | Binary sentence-relationship classification | RoBERTa showed removing this and training MLM-only on longer sequences *improves* most benchmarks — NSP's easy topic-shift shortcut wasn't teaching much |
| **Task-specific head** (fine-tune) | varies: `768→C` (classification), `768→2` (QA start/end), `768→tags` (NER) | Adapts the general-purpose 768-d representations to your specific task | Without any head, you have contextual embeddings but no task output — this is the whole point of fine-tuning (Chapter 10) |

---

# Part 2 — Dimension Trace (the "new sheet")

**Notation used throughout:** `n` = sequence length, `d` = 768 (hidden size), `h` = 12 (heads), `d_k` = 64 (per-head dim), `d_ff` = 3072, `V` = 30,522 (vocab), `B` = batch size.

## 2.1 General shape formulas (memorize this table)

| Stage | Shape (general) | Params at this stage |
|---|---|---|
| Token IDs | `[B, n]` (integers) | — |
| Token / Position / Segment embeddings, each | `[B, n, d]` | 30,522×768 / 512×768 / 2×768 |
| Summed + normalized embedding | `[B, n, d]` | — |
| Q, K, V (per head) | `[B, n, d_k]` | `d×d_k` each, ×3, ×h heads = `d×d` total per Q/K/V |
| Attention scores (`QKᵀ`), per head | `[B, n, n]` | 0 (no params — just a matmul) |
| Attention output, per head | `[B, n, d_k]` | — |
| Concat all heads | `[B, n, d]` (= `h × d_k`) | — |
| After `W_O` | `[B, n, d]` | `d×d` = 589,824 |
| After residual 1 + LN 1 | `[B, n, d]` | — |
| FFN hidden | `[B, n, d_ff]` | `d×d_ff` = 2,359,296 |
| FFN output | `[B, n, d]` | `d_ff×d` = 2,359,296 |
| After residual 2 + LN 2 (= one block's output) | `[B, n, d]` | — |
| **After all 12 blocks** | `[B, n, d]` — **same shape as input to block 1** | — |
| Pooler output (from `[CLS]` only) | `[B, d]` | `d×d` = 589,824 |
| MLM logits (per masked position) | `[B, n_masked, V]` | 0 extra (tied to embedding) |
| NSP / classification logits | `[B, C]` (C = num classes, 2 for NSP) | `d×C` |

**The one thing to say out loud**: *"Shape in = shape out for every single Transformer block — `[n×768]` the entire way through all 12 blocks. Only the attention-score intermediate `[n×n]` and the FFN's internal `[n×3072]` ever leave that width, and both get projected straight back down to 768 before the block ends."*

## 2.2 Concrete numeric trace — `"[CLS] the cat sat [SEP]"`, n=5, batch=1

| Stage | Shape | Notes |
|---|---|---|
| Token IDs | `[1, 5]` | `[101, 1996, 4937, 2938, 102]` |
| Token embeddings | `[1, 5, 768]` | lookup, per Chapter 3 |
| Position embeddings | `[1, 5, 768]` | positions 0-4 |
| Segment embeddings | `[1, 5, 768]` | all Segment A (single sentence) |
| Summed + LayerNorm | `[1, 5, 768]` | input to Block 1 |
| Per-head Q, K, V (×12 heads) | `[1, 5, 64]` each | 12 independent sets |
| Attention scores (per head) | `[1, 5, 5]` | 25 pairwise scores per head |
| Attention output (per head) | `[1, 5, 64]` | |
| Concat 12 heads | `[1, 5, 768]` | `64 × 12 = 768` |
| After `W_O` | `[1, 5, 768]` | |
| Block 1 output (after both residual+LN) | `[1, 5, 768]` | feeds Block 2 |
| ... (×12, identical shapes throughout) | `[1, 5, 768]` | only content changes, never shape |
| Final hidden states (`X_12`) | `[1, 5, 768]` | one 768-d vector per token |
| `[CLS]` vector specifically | `[1, 768]` | `X_12[:, 0, :]` |
| Pooler output | `[1, 768]` | `tanh(Linear(X_12[:,0,:]))` |
| Binary classification head (e.g. sentiment) | `[1, 2]` | logits |

## 2.3 Where the O(n²) and O(n·d²) costs live, by shape

| Operation | Shape driving cost | Scaling |
|---|---|---|
| `QKᵀ` (attention scores) | `[n,d_k]·[d_k,n] → [n,n]` | **O(n²·d)** total across heads |
| `softmax(...)·V` | `[n,n]·[n,d_k] → [n,d_k]` | O(n²·d_k) |
| Q/K/V/O projections | `[n,d]·[d,d]` | O(n·d²) |
| FFN | `[n,d]·[d,d_ff]` then back | O(n·d·d_ff) ≈ O(n·d²) since d_ff=4d |
| **Crossover point** (attention = other) | — | n = 6d = **4,608 tokens** for BERT-base — below this, FFN+projections dominate; BERT's 512-token ceiling sits well *under* this crossover |

---

# Part 3 — The Few-Pager: Speed-Read Before Walking In

## 3.1 One-liner per chapter (say these out loud, in order, as a warm-up)

1. **The problem**: static embeddings (Word2Vec) → no context; RNNs/GPT → one direction only; BERT = bidirectional Transformer encoder via masking.
2. **Tokenization**: WordPiece scores merges by likelihood gain `count(a,b)/(count(a)·count(b))`, not raw frequency (that's BPE) — 30,522 vocab balances OOV-robustness against sequence length.
3. **Embeddings**: token (identity) + position (order, learned not sinusoidal) + segment (which sentence), **summed** (not concatenated — avoids ~3x parameter blowup), then LayerNorm'd.
4. **Self-attention**: `softmax(QKᵀ/√d_k)·V` — separate Q/K/V projections (not tied) so attention can be directional; √d_k scaling cancels variance growth so softmax doesn't saturate.
5. **Multi-head**: 12 heads × 64-d costs the *same* total Q/K/V params as one 768-d head (just re-partitioned) — the only new cost is `W_O` (589,824 params). Concat, not average, preserves each head's signal undiluted.
6. **FFN**: `768→3072→768`, holds ~52% of all parameters — more than attention (~26%).
7. **Residuals + LayerNorm**: residual guarantees `∂out/∂x ≥ 1` (fixes backward-pass vanishing gradients); LayerNorm resets scale every sub-layer (fixes forward-pass growth). Two different problems, two different passes — you need both.
8. **Stacking 12 blocks**: shape never changes (`[n×768]` throughout); "receptive field" doesn't really apply (every token reaches every token from layer 1) — what grows with depth is compositional richness, not reachability.
9. **Pre-training**: MLM (mask target, forces bidirectionality since there's no answer to copy) + NSP (later shown mostly unnecessary by RoBERTa). Weight-tied MLM output saves ~23.4M params.

## 3.2 Numbers to have cold

| Fact | Number |
|---|---|
| Vocab size | 30,522 |
| Hidden size (`d`) | 768 |
| Heads | 12 (× 64-d each) |
| Layers | 12 (BERT-base) / 24 (BERT-large) |
| FFN hidden | 3,072 (= 4×768) |
| Max sequence length | 512 (hard limit — positional table has exactly 512 rows) |
| Total params (BERT-base) | ~110M (109,482,240 exactly) |
| Param split | Embeddings 21.8% / Attention 25.9% / **FFN 51.8%** / Pooler 0.5% |
| MLM masking | 15% of tokens selected → 80% `[MASK]` / 10% random / 10% unchanged |
| Attention-vs-FFN compute crossover | n = 6d = 4,608 tokens (below this, FFN dominates FLOPs — includes BERT's entire 512-token operating range) |
| Weight tying savings | 23,440,896 params (MLM decoder = embedding table transposed) |
| BERT-base vs large on SQuAD F1 | 88.5 vs 90.9 |
| DistilBERT | 6 layers, ~97% of BERT-base performance, 40% of the size — via distillation, not from-scratch training |
| Pre-training corpus | 3.3B words (800M BooksCorpus + 2,500M Wikipedia) |
| Pre-training compute | 1M steps, BERT-base: 4 TPUs × 4 days |

## 3.3 Equations, simplified to one line each

| Concept | Equation | Say this in words |
|---|---|---|
| Self-attention | `softmax(QKᵀ/√d_k)·V` | scores → scale → weights → blend |
| √d_k scaling | var(dot product) ∝ d_k → std ∝ √d_k | dividing by √d_k cancels dimension-driven variance growth |
| Residual | `∂(x+F(x))/∂x = 1 + ∂F/∂x` | the `+1` guarantees gradient never fully vanishes on that path |
| LayerNorm | `γ·(x-μ)/σ + β`, μ/σ **per token** | center, scale, learned rescale, learned reshift — never across the batch |
| WordPiece score | `count(a,b) / (count(a)·count(b))` | favors merging pairs that are wasteful to keep separate, not just frequent pairs |
| Sinusoidal PE (not used by BERT, but know it) | `sin(pos/10000^(2i/d))`, `cos(...)` | different dimension-pairs oscillate at different speeds → multi-resolution position signal |
| MLM loss | `-Σ log P(x_i \| context)`, masked positions only | standard cross-entropy, just computed only where masked |

## 3.4 The five interview traps most likely to come up

1. **"Where do most of BERT's parameters live?"** → FFN (52%), not attention (26%). Most people guess wrong here.
2. **"Isn't multi-head attention way more expensive than one big head?"** → No — same total Q/K/V params either way (12×[768×64] = 1×[768×768]). Only `W_O` is new cost.
3. **"WordPiece and BPE are basically the same, right?"** → No — WordPiece scores by likelihood gain (`count(a,b)/(count(a)·count(b))`), BPE by raw frequency. They can pick different merges.
4. **"Doesn't attention's O(n²) mean it's the compute bottleneck?"** → Only past ~4,608 tokens for BERT-base; at n=512, the FFN dominates both params and FLOPs.
5. **"Does receptive field grow with depth like in a CNN?"** → No — every token already reaches every token at layer 1 via full self-attention. Depth grows compositional richness/abstraction, not reachability.

## 3.5 If asked to whiteboard something, be ready for:

- Full attention formula on a 3-token toy example (scores → scale → softmax → weighted V sum).
- LayerNorm by hand on a 4-d vector (mean, variance, normalize).
- Why `Q=K` forces symmetric attention (dot product commutativity) — and why that's bad for language (subject→verb ≠ verb→subject strength).
- The residual derivative in one line: `1 + ∂F/∂x`.
- Sketching the full block diagram: `x → Attn → +x → LN → FFN → +← → LN → output`, same shape in and out.

---

*Pair this with the Chapter 1-9 master notes for the full derivations and worked numbers behind every line above.*
