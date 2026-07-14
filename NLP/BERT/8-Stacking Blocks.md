# Chapter 8: Stacking Transformer Blocks

You now know everything that happens inside one block. Let's zoom out and ask: **what happens when you run 12 of them in sequence?**

---

## 8.1 The Full Architecture — Top to Bottom

```
Input: "the cat sat"

         [CLS]  the   cat   sat  [SEP]
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Embedding Layer             │
    │  Token + Position + Segment     │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Transformer Block 1         │
    │  MultiHeadAttn → FFN            │
    │  + Residuals + LayerNorm        │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Transformer Block 2         │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓
          ...   ...   ...   ...   ...
           ↓     ↓     ↓     ↓     ↓
    ┌─────────────────────────────────┐
    │     Transformer Block 12        │
    └─────────────────────────────────┘
           ↓     ↓     ↓     ↓     ↓

    Final hidden states: one 768-d vector per token
```

**The shape never changes:**
```
After embedding:   [5 × 768]
After block 1:     [5 × 768]
After block 2:     [5 × 768]
...
After block 12:    [5 × 768]
```

Every block takes a `[seq_len × 768]` matrix and outputs a `[seq_len × 768]` matrix. What changes is what those 768 numbers **mean** — they get progressively richer.

---

## 8.2 What's Actually Shared vs Separate Across Blocks

A common confusion. Let's be explicit:

```
SEPARATE per block (each block has its own):
  W_Q, W_K, W_V for all 12 heads    → 12 × 3 × (768×64)
  W_O                                → 768×768
  W1, b1 (FFN expand)                → 768×3072
  W2, b2 (FFN compress)              → 3072×768
  γ, β for both LayerNorms           → 4 × 768

SHARED across all blocks:
  Token Embedding Table              → 30522×768
  Position Embedding Table           → 512×768
  Segment Embedding Table            → 2×768
```

This means every block is learning **different transformations** of the same representational space. Block 1's W_Q has nothing to do with Block 7's W_Q.

---

## 8.3 How Information Flows Through 12 Blocks

Let's trace the word **"bank"** through all 12 blocks in:

```
"I went to the river bank to fish"
```

### Block 1 — Raw Signal, Local Patterns

At the embedding layer, "bank" starts with its default token embedding — a static vector that's an average of all its meanings (financial, river, etc.).

Block 1 attention patterns tend to focus on **local neighbors** and **syntactic structure**. "bank" looks at "river" next to it, and "the" before it.

```
"bank" after Block 1:
  Still mostly its default embedding
  Slight pull toward "river" (adjacent)
  Basic syntactic position noted
```

### Blocks 2-4 — Syntax Solidifies

These layers handle grammatical structure:
- Subject-verb relationships
- Noun phrase boundaries
- Prepositional phrase attachments

```
"bank" after Block 4:
  Understands it's a noun
  Knows it's the object of "to"
  Knows "river" modifies it
  Starts pulling away from financial meaning
```

### Blocks 5-8 — Semantics Emerge

Longer range dependencies. "fish" at the end of the sentence is 4 positions away — attention heads in these layers reach further.

```
"bank" after Block 8:
  "fish" has now influenced it strongly
  "river" + "fish" together = strong nature/water signal
  Financial meaning is largely suppressed
  Co-reference relationships resolved
```

### Blocks 9-12 — Task-Specific Refinement

The highest layers produce the most abstract, task-relevant representations.

```
"bank" after Block 12:
  Fully contextualized toward river/nature meaning
  Its 768-d vector is now close in embedding space to
  "shore", "riverbank", "stream" — not "finance", "deposit"
  Ready for downstream task
```

---

## 8.4 Numerical Trace — One Token Across 3 Blocks

Let's track "cat" numerically through 3 simplified blocks (d=4).

**Starting vector after embedding:**
```
cat_0 = [0.70, 0.33, -0.31, 0.73]
```

We'll use simplified transformations to show the progression.

### After Block 1:

Attention output adds context from "the" and "sat":
```
Attention(cat_0) = [0.451, 0.170, 0.160, 0.300]   (from Chapter 4)

Residual:  cat_0 + Attention = [1.151, 0.500, -0.150, 1.030]
LayerNorm: cat_1 = [1.008, -0.259, -1.524, 0.772]  (from Chapter 7)

FFN transforms:
FFN(cat_1) = [0.291, 0.070, 0.394, -0.006]   (from Chapter 6)

Residual:  cat_1 + FFN = [1.299, -0.189, -1.130, 0.766]
LayerNorm: cat_1_final = [1.201, -0.301, -1.187, 0.701]
```

### After Block 2:

Block 2 has completely different W_Q, W_K, W_V, W1, W2. It sees cat_1_final as input and runs the same process with different weights.

```
cat_1_final = [1.201, -0.301, -1.187, 0.701]

(Block 2 attention + FFN + residuals + layernorm)

cat_2_final = [0.890, 0.124, -0.956, 1.102]   ← shifted again
```

### After Block 3:

```
cat_2_final = [0.890, 0.124, -0.956, 1.102]

(Block 3 attention + FFN + residuals + layernorm)

cat_3_final = [0.654, 0.441, -0.723, 0.987]   ← shifted again
```

**The progression:**
```
Block 0 (embedding): [0.700,  0.330, -0.310,  0.730]
Block 1:             [1.201, -0.301, -1.187,  0.701]
Block 2:             [0.890,  0.124, -0.956,  1.102]
Block 3:             [0.654,  0.441, -0.723,  0.987]
```

Each block reshapes the vector. The numbers aren't random drift — each transformation is learned to make the final layer 12 output maximally useful for predicting masked tokens during pre-training.

---

## 8.5 The [CLS] Token's Journey — Most Important Token

[CLS] is the most interesting token to trace because it has a special job: **by layer 12, it must represent the entire sentence.**

How does it achieve this? Through attention.

### Layer 1:
```
[CLS] attends to all tokens, but weights are fairly uniform
[CLS]_1 = weak blend of all token embeddings
```

### Layers 2-6:
```
[CLS] starts attending more selectively
It learns which tokens carry the most sentence-level information
[CLS]_6 = richer blend, starting to encode sentence meaning
```

### Layers 7-11:
```
[CLS] heavily attends to semantically important tokens
For "the cat sat on the mat":
  [CLS] → "cat" (0.31), "sat" (0.28), "mat" (0.22), others lower
[CLS]_11 = strong sentence-level representation
```

### Layer 12:
```
[CLS]_12 = final sentence embedding
Used directly for classification tasks
Its 768 dimensions encode the full meaning of the input
```

**This is why classification tasks use [CLS]** — it's the only token specifically designed, through 12 layers of attention over all other tokens, to aggregate global sentence meaning.

---

## 8.6 Probing Experiments — What Each Layer Actually Learns

Researchers have run **probing classifiers** on BERT — training a simple linear classifier on each layer's hidden states to test what linguistic information is encoded.

Results:

```
Layer  | Best captures
───────|──────────────────────────────────────────
1-2    | Basic token features, subword information
3-4    | POS tags (noun, verb, adjective)
5-6    | Syntactic chunking (noun phrases, verb phrases)
7-8    | Syntactic dependencies (subject, object)
9-10   | Semantic roles (who did what to whom)
11-12  | Coreference, long-range semantic relations
```

This layered hierarchy mirrors how linguists describe language:

```
Surface form → POS → Phrase structure → Dependencies → Semantics
     ↓           ↓          ↓                ↓             ↓
  Layers 1-2   3-4        5-6             7-8           9-12
```

BERT **rediscovered the structure of linguistics** purely from predicting masked words. Nobody told it that layer 4 should learn POS tags.

---

## 8.7 Why 12 Layers? Why Not 6 or 24?

BERT was released in two sizes:

```
BERT-base:  12 layers, 768 hidden, 12 heads → 110M params
BERT-large: 24 layers, 1024 hidden, 16 heads → 340M params
```

**BERT-base vs BERT-large on SQuAD (QA benchmark):**
```
BERT-base:  88.5 F1
BERT-large: 90.9 F1
```

Large wins, but costs 3× the compute and memory. The tradeoff:

```
More layers → captures more abstract relationships → better accuracy
More layers → more parameters → more data needed, more compute
```

6 layers (DistilBERT) gets ~97% of BERT-base performance at 40% the size — because many of BERT's 12 layers are partially redundant.

---

## 8.8 Parameter Count — Full BERT-base

Let's add everything up:

```
EMBEDDING LAYERS:
  Token embeddings:    30,522 × 768    =  23,440,896
  Position embeddings: 512 × 768       =     393,216
  Segment embeddings:  2 × 768         =       1,536
  LayerNorm (emb):     2 × 768         =       1,536
                                         ───────────
  Embedding total:                       23,837,184

PER TRANSFORMER BLOCK (× 12):
  Attention:
    W_Q, W_K, W_V:    3 × (768×768)   =   1,769,472
    W_O:               768×768         =     589,824
    Attention biases:  3×768 + 768     =       3,072
  LayerNorm 1:         2×768           =       1,536
  FFN:
    W1:                768×3072        =   2,359,296
    b1:                3072            =       3,072
    W2:                3072×768        =   2,359,296
    b2:                768             =         768
  LayerNorm 2:         2×768           =       1,536
                                         ───────────
  Per block:                             7,087,872
  × 12 blocks:                          85,054,464

POOLER (for [CLS] output):
  Linear:              768×768         =     589,824
  Bias:                768             =         768
                                         ───────────
  Pooler total:                            590,592

─────────────────────────────────────────────────────
TOTAL:                                   109,482,240
                                       ≈ 110M parameters
```

---

## 8.9 Receptive Field — How Far Can Each Layer See?

In CNNs, a layer can only "see" a local window. In Transformers, **every layer can see every token** — but the effective receptive field still grows with depth.

Here's why:

```
Block 1:  "bank" directly attends to "river" → direct connection

Block 2:  "bank" attends to Block 1's output of "river"
          Block 1's "river" already incorporated "I went to the"
          So "bank" now indirectly sees "I went to the river"

Block 3:  "bank" attends to Block 2 representations
          Which already have 2 layers of context baked in
          Effectively seeing the full sentence multiple times over
```

By block 12, every token has integrated context from every other token **through 12 rounds of attention**. The representations are extraordinarily rich.

---

## 8.10 The Complete Forward Pass — Everything Together

Let's write out the complete forward pass of BERT from raw text to final hidden states:

```
INPUT: "the cat sat"

Step 1 — Tokenize:
  ["[CLS]", "the", "cat", "sat", "[SEP]"]
  IDs: [101, 1996, 4937, 2938, 102]

Step 2 — Three Embeddings + Sum + LayerNorm:
  X_0 = LayerNorm(TokenEmb + PosEmb + SegEmb)
  Shape: [5 × 768]

Step 3 — Block 1:
  A_1 = LayerNorm(X_0 + MultiHeadAttention(X_0))
  X_1 = LayerNorm(A_1 + FFN(A_1))
  Shape: [5 × 768]

Step 4 — Block 2:
  A_2 = LayerNorm(X_1 + MultiHeadAttention(X_1))
  X_2 = LayerNorm(A_2 + FFN(A_2))
  Shape: [5 × 768]

  ... (repeat for blocks 3-11)

Step 5 — Block 12:
  A_12 = LayerNorm(X_11 + MultiHeadAttention(X_11))
  X_12 = LayerNorm(A_12 + FFN(A_12))
  Shape: [5 × 768]

OUTPUT:
  X_12[0] = [CLS] vector  → sentence-level representation
  X_12[1] = "the" vector  → contextual embedding of "the"
  X_12[2] = "cat" vector  → contextual embedding of "cat"
  X_12[3] = "sat" vector  → contextual embedding of "sat"
  X_12[4] = [SEP] vector  → end marker
```

---

## Chapter 8 Summary

```
12 blocks, each identical in structure, different in weights
Shape stays [seq_len × 768] throughout — only content changes

Layers 1-4:   syntax, POS, local patterns
Layers 5-8:   semantics, longer range
Layers 9-12:  abstract, task-relevant representations

[CLS] aggregates full sentence meaning across all 12 layers
Each block adds one round of context refinement
Residuals ensure gradients flow all the way back to layer 1
```

| | BERT-base | BERT-large |
|---|---|---|
| Blocks | 12 | 24 |
| Hidden dim | 768 | 1024 |
| Attention heads | 12 | 16 |
| Parameters | 110M | 340M |

---

Now you understand BERT's full architecture from tokens to final representations. But we haven't answered: **how did BERT learn all of this?** Nobody labeled 3.3 billion words with syntax trees or semantic roles. BERT learned everything from raw text alone.

That's Chapter 9 — Pre-training. The two ingenious self-supervised tasks that taught BERT everything it knows. Ready?
