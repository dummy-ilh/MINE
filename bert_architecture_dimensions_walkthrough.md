# BERT Architecture — Step-by-Step Walkthrough With Dimensions Carried Through

Fifth companion doc. This is the "walk me through it end to end" version — every stage of a forward pass, what it does, and the exact tensor shape at that point. Use BERT-base's real numbers throughout:

```
vocab_size      = 30,522
hidden_size (H) = 768
num_layers (L)  = 12
num_heads (A)   = 12
head_dim (d_k)  = H / A = 64
intermediate    = 3072   (4 x H, the FFN expansion)
max_position    = 512
```

Worked example dimensions used throughout: **batch size B = 8, sequence length T = 128** (a batch of 8 sentences, each padded/truncated to 128 tokens).

---

## Step 0 — Raw text → token IDs

**What it does:** WordPiece tokenizer splits text into subword tokens, adds `[CLS]` at the start and `[SEP]` at the end/between sentences, then maps every token to an integer ID from the 30,522-token vocabulary. Sequences shorter than T are padded; longer ones are truncated.

```
Input:  "I like NLP"
Tokens: [CLS] I like NL ##P [SEP]
IDs:    [101, 146, 1176, 21239, 2101, 102]  (example values)
```

**Dimensions:**
```
input_ids:      (B, T)        = (8, 128)      dtype: int
attention_mask: (B, T)        = (8, 128)      dtype: int (1 = real token, 0 = padding)
token_type_ids: (B, T)        = (8, 128)      dtype: int (0 = sentence A, 1 = sentence B)
```
Nothing here is a "vector" yet — these are just integer index arrays. No hidden_size dimension exists until the embedding lookup happens.

---

## Step 1 — Embedding layer

**What it does:** Three separate lookup tables convert the integer IDs into dense vectors, which are then **summed** element-wise: token identity, position in sequence, and segment (sentence A/B) — followed by LayerNorm and dropout. (Full rationale for why three separate embeddings is in the phases doc — short version: self-attention has no built-in notion of order or sentence membership, so both have to be injected explicitly.)

```
token_embeddings    = TokenEmbedTable[input_ids]        (30,522 x 768) lookup
position_embeddings = PosEmbedTable[0..T-1]              (512 x 768) lookup
segment_embeddings  = SegEmbedTable[token_type_ids]       (2 x 768) lookup

embeddings = LayerNorm(token_emb + position_emb + segment_emb)
```

**Dimensions:**
```
input_ids            (8, 128)
   │  lookup (30522 x 768 table)
   ▼
token_embeddings     (8, 128, 768)
   +
position_embeddings  (8, 128, 768)   ← broadcast from (128, 768), same for every item in batch
   +
segment_embeddings   (8, 128, 768)
   ▼
embeddings           (8, 128, 768)   ← this is the input to encoder layer 1
```

This is the first point where the `768` (hidden_size) dimension appears — from here on, every intermediate representation stays anchored to 768 in its last dimension (with one temporary exception inside the FFN — see Step 6).

---

## Step 2 — Linear projections: Q, K, V

**What it does:** Three separate learned weight matrices (`W_Q, W_K, W_V`, each 768×768) project the same input into Query, Key, and Value spaces. (Why three separate matrices, not one: covered in the numeric-walkthrough doc — prevents a token's self-similarity from trivially dominating attention.)

```
Q = X @ W_Q + b_Q
K = X @ W_K + b_K
V = X @ W_V + b_V
```

**Dimensions:**
```
X (block input)   (8, 128, 768)
W_Q, W_K, W_V     (768, 768)   each
   ▼  matrix multiply per token, applied across the whole batch and sequence
Q, K, V           (8, 128, 768)   each — same shape as input, projected into a new space
```

---

## Step 3 — Split into heads

**What it does:** Reshape the 768-dim Q/K/V vectors into 12 separate 64-dim chunks — one per attention head — so each head can compute its own independent attention pattern (specialization mechanism explained + demonstrated numerically in the earlier walkthrough doc).

```
Q (8, 128, 768) → reshape → (8, 128, 12, 64) → transpose → (8, 12, 128, 64)
```
Same reshape applied to K and V.

**Dimensions:**
```
Q, K, V before split:  (8, 128, 768)
   ▼  reshape (768 = 12 heads x 64 dims) + transpose head dim forward
Q, K, V after split:   (8, 12, 128, 64)
                         │   │    │   └─ per-token vector dim within this head
                         │   │    └───── sequence length (unchanged)
                         │   └────────── number of heads (new axis)
                         └────────────── batch (unchanged)
```

---

## Step 4 — Scaled dot-product attention (per head, all heads in parallel)

**What it does:** For each head independently: compute similarity scores between every pair of tokens (`Q @ Kᵀ`), scale by `1/sqrt(64)` to keep softmax well-behaved, mask out padding tokens using `attention_mask`, softmax each row into a probability distribution, then take a weighted sum of V using those probabilities.

```
scores = (Q @ K^T) / sqrt(64)        # similarity between every token pair, per head
scores = scores.masked_fill(padding, -inf)   # padding tokens get ~0 attention weight after softmax
attn   = softmax(scores, dim=-1)
head_output = attn @ V
```

**Dimensions:**
```
Q            (8, 12, 128, 64)
K^T          (8, 12, 64, 128)          ← last two dims transposed
   ▼  Q @ K^T
scores       (8, 12, 128, 128)         ← n x n attention matrix, per head, per batch item
                                          this is the O(T^2) cost: 128 x 128 per head x 12 heads x 8 batch
   ▼  softmax over last dim
attn_weights (8, 12, 128, 128)         ← same shape, now each row sums to 1
   ▼  attn_weights @ V   where V is (8, 12, 128, 64)
head_output  (8, 12, 128, 64)          ← back to per-head vector size
```

This `(8, 12, 128, 128)` attention matrix is the single largest intermediate tensor in the whole block — it's why attention memory cost grows quadratically with T (doubling T quadruples this tensor's size, as covered in the architecture-components doc).

---

## Step 5 — Concatenate heads + output projection

**What it does:** Merge the 12 heads' 64-dim outputs back into one 768-dim vector per token, then apply one more learned matrix `W_O` (768×768) that lets the model mix information across heads before it re-enters the residual stream.

```
concat = reshape heads back together
attn_output = concat @ W_O + b_O
```

**Dimensions:**
```
head_output       (8, 12, 128, 64)
   ▼  transpose heads back + reshape (12 x 64 = 768)
concat            (8, 128, 768)
   ▼  @ W_O (768, 768)
attn_output       (8, 128, 768)      ← back to hidden_size, ready for residual add
```

---

## Step 6 — Residual add + LayerNorm (post-attention)

**What it does:** Add the attention sublayer's output back to its own input (`X`), so the sublayer only had to learn a correction rather than reconstruct everything; then LayerNorm to keep every token's activation scale consistent before the next sublayer.

```
residual_out = attn_output + X
normed_out   = LayerNorm(residual_out)
```

**Dimensions:**
```
attn_output   (8, 128, 768)
   +
X (original)  (8, 128, 768)
   ▼
residual_out  (8, 128, 768)
   ▼  LayerNorm over the last dim (768) independently per token
normed_out    (8, 128, 768)     ← unchanged shape, this feeds the FFN sublayer
```

**Note the shape never changes here** — residual connections and LayerNorm are shape-preserving by design, which is exactly what lets them be inserted anywhere without disrupting the rest of the architecture.

---

## Step 7 — Feed-forward sublayer (the one place the dimension temporarily changes)

**What it does:** Two linear layers per token with GELU in between, expanding to 3072 dims and back down to 768. This is where nonlinear, per-token processing happens — attention only ever mixes/routes information, it never applies a nonlinearity (full rationale in the components doc).

```
h1 = GELU(normed_out @ W1 + b1)      # 768 -> 3072
ffn_out = h1 @ W2 + b2                # 3072 -> 768
```

**Dimensions:**
```
normed_out   (8, 128, 768)
   ▼  @ W1 (768, 3072)
h1           (8, 128, 3072)     ← the ONLY point in the whole block where the last dim isn't 768
   ▼  GELU (elementwise, shape unchanged)
h1_activated (8, 128, 3072)
   ▼  @ W2 (3072, 768)
ffn_out      (8, 128, 768)      ← back to hidden_size
```

---

## Step 8 — Residual add + LayerNorm (post-FFN)

**What it does:** Same mechanism as Step 6, applied to the FFN sublayer's output instead of attention's.

```
residual_out2 = ffn_out + normed_out
block_output  = LayerNorm(residual_out2)
```

**Dimensions:**
```
ffn_out        (8, 128, 768)
   +
normed_out     (8, 128, 768)
   ▼
residual_out2  (8, 128, 768)
   ▼  LayerNorm
block_output   (8, 128, 768)     ← this is the final output of ONE encoder block
```

**This `block_output` becomes the input `X` to the next encoder block.** Steps 2 through 8 repeat identically 12 times (BERT-base) — same shapes at every layer, only the weight matrices differ per layer. The `(8, 128, 768)` shape never changes across all 12 layers; only the values inside it do, becoming progressively more contextualized/abstract with each layer (lower layers → surface/syntax, upper layers → semantics, as discussed in the phases doc).

---

## Step 9 — Final output → task head (classification example)

**What it does:** After the 12th block, take the final `[CLS]` token's vector (position 0 in the sequence) as the pooled sequence representation, optionally pass it through one more `Linear + Tanh` "pooler" layer (BERT's original design), then a task-specific classification head.

```
last_hidden_state = output of encoder layer 12          (8, 128, 768)
cls_vector = last_hidden_state[:, 0, :]                   # take position 0 for every batch item
pooled = tanh(cls_vector @ W_pool + b_pool)                # optional BERT pooler
logits = pooled @ W_cls + b_cls                             # task-specific head
```

**Dimensions:**
```
last_hidden_state  (8, 128, 768)
   ▼  slice out position 0 (the [CLS] token) for every item in the batch
cls_vector         (8, 768)              ← sequence dimension (128) is GONE — one vector per example now
   ▼  @ W_pool (768, 768), tanh
pooled             (8, 768)
   ▼  @ W_cls (768, num_labels)   e.g. num_labels = 2 for binary sentiment
logits             (8, 2)                ← final output: one score per class, per example in the batch
```

`softmax(logits)` at inference, or `CrossEntropyLoss(logits, labels)` at training time, from here.

---

## The full shape journey, in one table

| Step | Operation | Output shape |
|---|---|---|
| 0 | Tokenize | `(8, 128)` — int IDs, no hidden dim yet |
| 1 | Embedding lookup + sum | `(8, 128, 768)` |
| 2 | Q/K/V projection | `(8, 128, 768)` x3 |
| 3 | Split into 12 heads | `(8, 12, 128, 64)` x3 |
| 4 | Attention scores | `(8, 12, 128, 128)` |
| 4 | Attention output (per head) | `(8, 12, 128, 64)` |
| 5 | Concat heads + output proj | `(8, 128, 768)` |
| 6 | Residual + LayerNorm | `(8, 128, 768)` |
| 7 | FFN expand (768→3072) | `(8, 128, 3072)` |
| 7 | FFN project back (3072→768) | `(8, 128, 768)` |
| 8 | Residual + LayerNorm | `(8, 128, 768)` — **repeat steps 2-8 x12 layers** |
| 9 | Take `[CLS]`, pool | `(8, 768)` |
| 9 | Classification head | `(8, num_labels)` |

**The one thing to say if asked to summarize this in one sentence:** *the hidden dimension (768) stays fixed through the entire stack — the only shape changes happen inside the FFN's temporary 4x expansion, and in the attention step where a new `(T x T)` matrix is created and then collapsed back down — everything else is either a reshape (heads) or a shape-preserving operation (residual, LayerNorm).*
