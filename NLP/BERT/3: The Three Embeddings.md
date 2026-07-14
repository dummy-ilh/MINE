# Chapter 3: The Three Embeddings

At the end of Chapter 2, you had this for `"the cat sat"`:

```
Token IDs:   [101,  1996, 4937, 2938,  102]
Positions:   [  0,     1,    2,    3,    4]
Segments:    [  0,     0,    0,    0,    0]
```

Three integer arrays. BERT can't do math on integers directly — they need to become **dense vectors**. This chapter is about exactly how that happens, and why three separate embeddings are needed.

---

## The Big Picture First

```
Token IDs  → Token Embedding Table  → 768-d vector per token
Positions  → Position Embedding Table → 768-d vector per position  
Segments   → Segment Embedding Table → 768-d vector per segment
                                         ↓
                              SUM all three element-wise
                                         ↓
                              LayerNorm + Dropout
                                         ↓
                     One 768-d vector per token → into Transformer Block 1
```

Each embedding table is **learned during pre-training** — they are not hand-crafted. They are just weight matrices that get updated by backprop like everything else.

---

## 3.1 Token Embeddings

### What it is

A lookup table of shape:
```
[vocab_size × hidden_dim] = [30,522 × 768]
```

Each row is the embedding vector for one token in the vocabulary. Token ID 4937 ("cat") → row 4937 → a 768-d vector.

### Parameters

```
30,522 × 768 = 23,440,896 parameters
```

That's ~23M parameters just in this one table. It's the largest single weight matrix in BERT.

### What it captures

Pure **identity** of the token — what word/subword this is, divorced from position or context. Think of it as the "default meaning" before any context is applied.

### Numerical Example (d=4 for clarity)

```
Vocabulary:
  [CLS] (ID 101)  → row 101  → [ 0.12, -0.08,  0.31,  0.05]
  the   (ID 1996) → row 1996 → [ 0.21, -0.45,  0.88,  0.12]
  cat   (ID 4937) → row 4937 → [ 0.55,  0.31, -0.22,  0.67]
  sat   (ID 2938) → row 2938 → [-0.10,  0.44,  0.19, -0.33]
  [SEP] (ID 102)  → row 102  → [-0.05,  0.02, -0.11,  0.08]
```

These values are **random at initialization** and learned during pre-training.

---

## 3.2 Positional Embeddings

### The Problem They Solve

Self-attention (Chapter 4) looks at all tokens simultaneously. There is no inherent left-to-right order — mathematically, shuffling the tokens produces the same attention scores (just in different order).

**Without positional embeddings:**
```
"the cat sat"   →  same representation as  →  "sat cat the"
```

That's catastrophic. Word order is meaning.

### What it is

Another lookup table of shape:
```
[max_seq_len × hidden_dim] = [512 × 768]
```

Position 0 has its own 768-d vector. Position 1 has its own. Up to position 511.

### BERT vs Original Transformer

The original 2017 Transformer paper used **fixed sinusoidal functions** for position:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

BERT said: **don't hand-craft this, learn it.** BERT's positional embeddings are fully learned parameters, initialized randomly and trained end-to-end.

**Why learned beats fixed here:**
The sinusoidal approach was designed so the model could extrapolate to longer sequences than seen in training. But BERT has a hard 512 limit anyway — so there's no need to extrapolate. Learned embeddings can adapt to the actual patterns in the data.

### What it captures

**Order and distance.** After training, nearby positions tend to have similar embeddings. The model learns that position 3 is "close to" position 4 and "far from" position 200.

### The 512 Limit Explained

This table has exactly 512 rows. If your sequence is 513 tokens — there is no row 512 in the table. You literally cannot represent it. This is why 512 is a hard architectural limit, not just a recommendation.

### Numerical Example (d=4)

```
Position 0 ([CLS]) → [ 0.01,  0.03, -0.02,  0.04]
Position 1 (the)   → [ 0.10, -0.05,  0.08, -0.01]
Position 2 (cat)   → [ 0.15,  0.02, -0.09,  0.06]
Position 3 (sat)   → [ 0.20,  0.07,  0.11, -0.03]
Position 4 ([SEP]) → [ 0.25,  0.09,  0.14, -0.05]
```

Notice positions 3 and 4 are more similar to each other than positions 0 and 4 — the model learns proximity.

---

## 3.3 Segment Embeddings

### The Problem They Solve

Many NLP tasks involve **two sentences**:

```
QA:  Question: "Where did the cat sit?"
     Passage:  "The cat sat on the mat."

NLI: Premise:    "The cat sat on the mat."
     Hypothesis: "An animal was on the mat."
```

BERT concatenates both into one sequence. But how does it know which tokens belong to which sentence? Token embeddings don't carry that info. Position embeddings don't either.

Enter segment embeddings.

### What it is

The simplest of the three tables:
```
[2 × hidden_dim] = [2 × 768]
```

Just **two rows.** Row 0 = Segment A embedding. Row 1 = Segment B embedding.

Every token gets either the Segment A vector or Segment B vector added to it, depending on which sentence it belongs to.

### What it captures

**Which sentence am I in?** This gives the model a way to compare and contrast the two segments, which is essential for tasks like:
- Does sentence B follow from sentence A? (NSP)
- Where in the passage does the answer to the question appear? (QA)

### For Single Sentence Tasks

All tokens get Segment A (row 0). Segment B is simply not used.

### Numerical Example (d=4)

```
Segment A → [ 0.00,  0.00,  0.00,  0.00]  (often initializes near zero)
Segment B → [ 0.10, -0.10,  0.20,  0.05]  (learned to be different from A)
```

---

## 3.4 Putting It All Together — Full Numerical Example

Input: `"the cat sat"` — single sentence, d=4

### The Three Tables

```
TOKEN EMBEDDINGS:
  [CLS] → [ 0.12, -0.08,  0.31,  0.05]
  the   → [ 0.21, -0.45,  0.88,  0.12]
  cat   → [ 0.55,  0.31, -0.22,  0.67]
  sat   → [-0.10,  0.44,  0.19, -0.33]
  [SEP] → [-0.05,  0.02, -0.11,  0.08]

POSITION EMBEDDINGS:
  pos 0 → [ 0.01,  0.03, -0.02,  0.04]
  pos 1 → [ 0.10, -0.05,  0.08, -0.01]
  pos 2 → [ 0.15,  0.02, -0.09,  0.06]
  pos 3 → [ 0.20,  0.07,  0.11, -0.03]
  pos 4 → [ 0.25,  0.09,  0.14, -0.05]

SEGMENT EMBEDDINGS (all Segment A, single sentence):
  all   → [ 0.00,  0.00,  0.00,  0.00]
```

### Element-wise Sum for Each Token

```
[CLS] (pos 0, seg A):
  Token    [ 0.12, -0.08,  0.31,  0.05]
  Position [ 0.01,  0.03, -0.02,  0.04]
  Segment  [ 0.00,  0.00,  0.00,  0.00]
  ────────────────────────────────────
  Sum      [ 0.13, -0.05,  0.29,  0.09]

the (pos 1, seg A):
  Token    [ 0.21, -0.45,  0.88,  0.12]
  Position [ 0.10, -0.05,  0.08, -0.01]
  Segment  [ 0.00,  0.00,  0.00,  0.00]
  ────────────────────────────────────
  Sum      [ 0.31, -0.50,  0.96,  0.11]

cat (pos 2, seg A):
  Token    [ 0.55,  0.31, -0.22,  0.67]
  Position [ 0.15,  0.02, -0.09,  0.06]
  Segment  [ 0.00,  0.00,  0.00,  0.00]
  ────────────────────────────────────
  Sum      [ 0.70,  0.33, -0.31,  0.73]

sat (pos 3, seg A):
  Token    [-0.10,  0.44,  0.19, -0.33]
  Position [ 0.20,  0.07,  0.11, -0.03]
  Segment  [ 0.00,  0.00,  0.00,  0.00]
  ────────────────────────────────────
  Sum      [ 0.10,  0.51,  0.30, -0.36]

[SEP] (pos 4, seg A):
  Token    [-0.05,  0.02, -0.11,  0.08]
  Position [ 0.25,  0.09,  0.14, -0.05]
  Segment  [ 0.00,  0.00,  0.00,  0.00]
  ────────────────────────────────────
  Sum      [ 0.20,  0.11,  0.03,  0.03]
```

### What Goes Into Transformer Block 1

```
Shape: [5 tokens × 4 dims]   (in reality [5 × 768])

Input matrix X =
  [CLS] → [ 0.13, -0.05,  0.29,  0.09]
  the   → [ 0.31, -0.50,  0.96,  0.11]
  cat   → [ 0.70,  0.33, -0.31,  0.73]
  sat   → [ 0.10,  0.51,  0.30, -0.36]
  [SEP] → [ 0.20,  0.11,  0.03,  0.03]
```

This matrix X is the input to every attention head in Block 1.

---

## 3.5 Why Sum and Not Concatenate?

You might ask — why add the three embeddings instead of concatenating them into a 768×3 = 2304-d vector?

**Concatenation** would triple the model size and every downstream weight matrix. Computationally expensive.

**Summation** keeps the dimension at 768 throughout. The model learns to encode all three signals (identity + position + segment) into the same 768-d space, with different dimensions specializing for different signals. Empirically it works just as well.

---

## 3.6 What Happens After Summing?

Two more steps before Block 1:

**LayerNorm:** Stabilizes the scale of the summed vectors (covered deeply in Chapter 7).
```
X = LayerNorm(Token_Emb + Position_Emb + Segment_Emb)
```

**Dropout:** Randomly zeros some values during training to prevent overfitting.
```
X = Dropout(X, p=0.1)
```

Then X flows into Transformer Block 1.

---

## Chapter 3 Summary

| Embedding | Table Shape | Captures | Without It |
|---|---|---|---|
| Token | [30522 × 768] | Word identity | No word meaning at all |
| Position | [512 × 768] | Word order | "cat sat" = "sat cat" |
| Segment | [2 × 768] | Which sentence | Can't handle two-sentence tasks |

Three separate learned tables. Summed element-wise. LayerNormed. That 768-d result per token is what the Transformer sees.

---

Now here's what's beautiful — after Chapter 2 and 3, you have X: a matrix of shape `[seq_len × 768]`. Every token is a vector. But those vectors don't yet "know about" each other. Token "cat" has no idea "sat" exists next to it.

Yes, exactly. Both are learned during **pre-training** and then **frozen** (or fine-tuned) after.

Let me make the lifecycle crystal clear:

---

## During Pre-Training (Google's job)

```
Token Embedding Table    → randomly initialized → updated by backprop
Position Embedding Table → randomly initialized → updated by backprop
Segment Embedding Table  → randomly initialized → updated by backprop
+ All Transformer weights
```

All three tables are just weight matrices. Backprop flows through them exactly like any other layer. After training on 3.3 billion words, they settle into meaningful values.

---

## During Fine-Tuning (Your job)

You download the pre-trained weights — including all three embedding tables — and continue training on your task data.

Two common choices:

```
Option 1: Fine-tune everything
  → All three tables keep updating on your task data
  → Better if you have enough data

Option 2: Freeze lower layers, update upper layers only
  → Token/Position embeddings often frozen
  → Upper Transformer blocks update
  → Better if your dataset is small
```

---

## During Inference (Production)

```
All weights frozen. No updates.
Token ID 4937 → always the same row → same vector.
Position 2    → always the same row → same vector.
```

---

## The Subtle but Important Point

The **token embedding** encodes what the word *is*.
The **position embedding** encodes *where* it is.

But here's what's key — after attention, the token's vector gets **transformed by context**. So even though "cat" always starts with the same token embedding, by layer 12 it has a completely different vector depending on the sentence around it.

```
Pre-training learns:  good default embeddings
Attention learns:     how to transform them using context
Fine-tuning adjusts: both, for your specific task
```

The embeddings are the **starting point**. Attention does the contextual work on top.

Ready for Chapter 4?
