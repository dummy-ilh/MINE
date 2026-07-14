# Chapter 4: Self-Attention — The Heart of BERT

At the end of Chapter 3, you have this matrix entering Block 1:

```
X = [seq_len × 768]

  [CLS] → [ 0.13, -0.05,  0.29,  0.09]
  the   → [ 0.31, -0.50,  0.96,  0.11]
  cat   → [ 0.70,  0.33, -0.31,  0.73]
  sat   → [ 0.10,  0.51,  0.30, -0.36]
  [SEP] → [ 0.20,  0.11,  0.03,  0.03]
```

Every token has a vector. But each vector only knows about **itself** — "cat" has no idea "sat" is next to it. "the" has no idea what noun it's modifying.

Self-attention fixes this. It lets every token **look at every other token** and update itself based on what it sees.

---

## 4.1 The Core Intuition

Imagine you're the word "bank" in this sentence:

```
"I went to the bank to fish"
```

You need to figure out what you mean. So you ask every other word:

```
"I"     → are you relevant to my meaning? → a little
"went"  → are you relevant?               → not much  
"to"    → relevant?                       → not much
"the"   → relevant?                       → not much
"fish"  → relevant?                       → YES, a lot
```

You collect answers, weight them by relevance, and update your own representation. Now "bank" leans toward "river/nature" meaning because "fish" pulled it strongly.

That relevance-checking process — that's self-attention.

---

## 4.2 The Three Vectors: Q, K, V

Every token produces three vectors from its embedding. These are learned projections:

```
Q (Query)  → "What am I looking for?"
K (Key)    → "What do I advertise to others?"
V (Value)  → "What do I give if someone attends to me?"
```

Think of it like a **search engine**:
- Your Query is your search term
- Keys are the index entries of all documents
- Values are the actual document contents
- Attention score = how well your query matches each key
- Output = weighted mix of values based on match scores

### The Projection Matrices

Three learned weight matrices — same for all tokens, applied per token:

```
W_Q: [768 × 64]
W_K: [768 × 64]
W_V: [768 × 64]
```

Why 64? Because BERT-base has 12 attention heads, and 768/12 = 64. Each head works in a 64-d subspace. (More on this in Chapter 5.)

### Computing Q, K, V

```
Q = X · W_Q    [seq_len × 768] · [768 × 64] = [seq_len × 64]
K = X · W_K    [seq_len × 768] · [768 × 64] = [seq_len × 64]
V = X · W_V    [seq_len × 768] · [768 × 64] = [seq_len × 64]
```

Every token now has three 64-d vectors. Let's work with numbers.

---

## 4.3 Full Numerical Example — Step by Step

We'll use 3 tokens to keep it tractable. d=4 (instead of 64).

```
Tokens: [the] [cat] [sat]

Input X (3×4):
  the → [0.31, -0.50,  0.96,  0.11]
  cat → [0.70,  0.33, -0.31,  0.73]
  sat → [0.10,  0.51,  0.30, -0.36]
```

### Projection Matrices (4×4, learned)

```
W_Q = [[ 1.0,  0.0,  0.0,  0.0],
       [ 0.0,  1.0,  0.0,  0.0],
       [ 0.0,  0.0,  1.0,  0.0],
       [ 0.0,  0.0,  0.0,  1.0]]   ← identity, for clarity

W_K = same as W_Q
W_V = same as W_Q
```

So Q = K = V = X here (since W is identity). In reality these matrices are learned and very different from each other.

```
Q = K = V =
  the → [0.31, -0.50,  0.96,  0.11]
  cat → [0.70,  0.33, -0.31,  0.73]
  sat → [0.10,  0.51,  0.30, -0.36]
```

---

### STEP 1: Raw Attention Scores = Q · Kᵀ

"How much does each token's Query match each token's Key?"

Kᵀ (4×3):
```
     the    cat    sat
  [ 0.31,  0.70,  0.10]
  [-0.50,  0.33,  0.51]
  [ 0.96, -0.31,  0.30]
  [ 0.11,  0.73, -0.36]
```

Score matrix = Q · Kᵀ (3×3):

Each cell = dot product of one row of Q with one column of Kᵀ.

```
Score[the→the] = 0.31×0.31 + (-0.50×-0.50) + 0.96×0.96 + 0.11×0.11
               = 0.096 + 0.250 + 0.922 + 0.012
               = 1.280

Score[the→cat] = 0.31×0.70 + (-0.50×0.33) + 0.96×(-0.31) + 0.11×0.73
               = 0.217 - 0.165 - 0.298 + 0.080
               = -0.166

Score[the→sat] = 0.31×0.10 + (-0.50×0.51) + 0.96×0.30 + 0.11×(-0.36)
               = 0.031 - 0.255 + 0.288 - 0.040
               = 0.024

Score[cat→the] = 0.70×0.31 + 0.33×(-0.50) + (-0.31)×0.96 + 0.73×0.11
               = 0.217 - 0.165 - 0.298 + 0.080
               = -0.166

Score[cat→cat] = 0.70×0.70 + 0.33×0.33 + (-0.31)×(-0.31) + 0.73×0.73
               = 0.490 + 0.109 + 0.096 + 0.533
               = 1.228

Score[cat→sat] = 0.70×0.10 + 0.33×0.51 + (-0.31)×0.30 + 0.73×(-0.36)
               = 0.070 + 0.168 - 0.093 - 0.263
               = -0.118

Score[sat→the] = 0.10×0.31 + 0.51×(-0.50) + 0.30×0.96 + (-0.36)×0.11
               = 0.031 - 0.255 + 0.288 - 0.040
               = 0.024

Score[sat→cat] = 0.10×0.70 + 0.51×0.33 + 0.30×(-0.31) + (-0.36)×0.73
               = 0.070 + 0.168 - 0.093 - 0.263
               = -0.118

Score[sat→sat] = 0.10×0.10 + 0.51×0.51 + 0.30×0.30 + (-0.36)×(-0.36)
               = 0.010 + 0.260 + 0.090 + 0.130
               = 0.490
```

**Raw Score Matrix (3×3):**
```
              the      cat      sat
  the  →  [ 1.280,  -0.166,   0.024]
  cat  →  [-0.166,   1.228,  -0.118]
  sat  →  [ 0.024,  -0.118,   0.490]
```

Each row = one token's attention scores toward every other token.

---

### STEP 2: Scale by √d_k

d_k = 4, so √4 = 2.0

```
Scaled =
              the      cat      sat
  the  →  [ 0.640,  -0.083,   0.012]
  cat  →  [-0.083,   0.614,  -0.059]
  sat  →  [ 0.012,  -0.059,   0.245]
```

**Why does this matter? Let's see what happens without it:**

```
Without scaling (d_k=64, realistic case):
  Scores could be: [38.4, -4.98, 0.72]
  Softmax → [≈1.00, ≈0.00, ≈0.00]  ← one-hot, no gradient

With √64=8 scaling:
  Scores become: [4.80, -0.62, 0.09]
  Softmax → [0.91, 0.04, 0.05]  ← softer, gradients flow
```

Scaling keeps the softmax in a regime where gradients are meaningful.

---

### STEP 3: Softmax Row by Row

Softmax converts raw scores into **probabilities that sum to 1** — the attention weights.

```
softmax([z1, z2, z3]) = [e^z1, e^z2, e^z3] / (e^z1 + e^z2 + e^z3)
```

**Row 1 — "the" attending to [the, cat, sat]:**
```
Scores: [0.640, -0.083, 0.012]

e^0.640  = 1.896
e^-0.083 = 0.920
e^0.012  = 1.012
Sum      = 3.828

Weights: [1.896/3.828, 0.920/3.828, 1.012/3.828]
       = [0.495,       0.240,        0.264]
```

"the" attends to itself most (0.495), then "sat" (0.264), then "cat" (0.240).

**Row 2 — "cat" attending to [the, cat, sat]:**
```
Scores: [-0.083, 0.614, -0.059]

e^-0.083 = 0.920
e^0.614  = 1.848
e^-0.059 = 0.943
Sum      = 3.711

Weights: [0.248, 0.498, 0.254]
```

"cat" attends to itself most (0.498), roughly equally to "the" and "sat".

**Row 3 — "sat" attending to [the, cat, sat]:**
```
Scores: [0.012, -0.059, 0.245]

e^0.012  = 1.012
e^-0.059 = 0.943
e^0.245  = 1.278
Sum      = 3.233

Weights: [0.313, 0.292, 0.395]
```

"sat" attends to itself most (0.395), then "the" (0.313), then "cat" (0.292).

**Full Attention Weight Matrix A (3×3):**
```
              the    cat    sat
  the  →  [0.495, 0.240, 0.264]   ← sums to 1.0
  cat  →  [0.248, 0.498, 0.254]   ← sums to 1.0
  sat  →  [0.313, 0.292, 0.395]   ← sums to 1.0
```

---

### STEP 4: Weighted Sum of Values

Now use those weights to blend the Value vectors:

```
Output[token] = sum of (attention_weight × V[each_token])
```

**Output for "the":**
```
= 0.495 × V[the] + 0.240 × V[cat] + 0.264 × V[sat]

= 0.495 × [ 0.31, -0.50,  0.96,  0.11]
+ 0.240 × [ 0.70,  0.33, -0.31,  0.73]
+ 0.264 × [ 0.10,  0.51,  0.30, -0.36]

= [ 0.153, -0.248,  0.475,  0.054]   (0.495 × V[the])
+ [ 0.168,  0.079, -0.074,  0.175]   (0.240 × V[cat])
+ [ 0.026,  0.135,  0.079, -0.095]   (0.264 × V[sat])
─────────────────────────────────────
= [ 0.348, -0.033,  0.481,  0.135]
```

**Output for "cat":**
```
= 0.248 × [ 0.31, -0.50,  0.96,  0.11]
+ 0.498 × [ 0.70,  0.33, -0.31,  0.73]
+ 0.254 × [ 0.10,  0.51,  0.30, -0.36]

= [ 0.077, -0.124,  0.238,  0.027]
+ [ 0.349,  0.164, -0.154,  0.364]
+ [ 0.025,  0.130,  0.076, -0.091]
─────────────────────────────────────
= [ 0.451,  0.170,  0.160,  0.300]
```

**Output for "sat":**
```
= 0.313 × [ 0.31, -0.50,  0.96,  0.11]
+ 0.292 × [ 0.70,  0.33, -0.31,  0.73]
+ 0.395 × [ 0.10,  0.51,  0.30, -0.36]

= [ 0.097, -0.157,  0.300,  0.034]
+ [ 0.204,  0.096, -0.091,  0.213]
+ [ 0.040,  0.201,  0.119, -0.142]
─────────────────────────────────────
= [ 0.341,  0.141,  0.329,  0.105]
```

---

### Final Output of Self-Attention (3×4):

```
              dim1    dim2    dim3    dim4
  the  →  [ 0.348, -0.033,  0.481,  0.135]
  cat  →  [ 0.451,  0.170,  0.160,  0.300]
  sat  →  [ 0.341,  0.141,  0.329,  0.105]
```

**Compare to what went IN:**
```
  the  →  [ 0.31, -0.50,  0.96,  0.11]   ← before attention
  cat  →  [ 0.70,  0.33, -0.31,  0.73]   ← before attention
  sat  →  [ 0.10,  0.51,  0.30, -0.36]   ← before attention
```

Every token's vector has **changed**. "cat" is no longer just its own embedding — it now contains a weighted blend of information from "the" and "sat" too. It has **context**.

---

## 4.4 The Full Formula in One Line

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

That's it. Four steps collapsed into one line:
- `Q·Kᵀ` → raw scores
- `/ √d_k` → scale
- `softmax(...)` → weights
- `· V` → weighted blend

---

## 4.5 What the Attention Matrix Tells You

```
              the    cat    sat
  the  →  [0.495, 0.240, 0.264]
  cat  →  [0.248, 0.498, 0.254]
  sat  →  [0.313, 0.292, 0.395]
```

Read each **row** as: "Token X distributes its attention across all tokens as follows."

Right now with identity weight matrices, tokens attend mostly to themselves. In a **real trained BERT**, these weights become linguistically meaningful:

```
"The cat that chased the mouse sat"

  sat  →  cat: 0.52  (subject-verb: sat's subject is cat)
           the: 0.18
           chased: 0.12
           mouse: 0.08
           that: 0.06
           The: 0.04
```

The model has learned that "sat" should heavily attend to its subject "cat" — even though they are far apart with a relative clause between them. **This is what RNNs couldn't do.**

---

## 4.6 Complexity: The O(n²) Problem

The score matrix is `[seq_len × seq_len]`. For every pair of tokens, you compute one dot product.

```
Sequence length 512:   512 × 512 = 262,144 dot products
Sequence length 1024:  1024 × 1024 = 1,048,576 dot products
Sequence length 4096:  4096 × 4096 = 16,777,216 dot products
```

Attention is **O(n²)** in sequence length. This is why:
- BERT caps at 512 tokens
- Long-document models (Longformer, BigBird) replace full attention with sparse attention patterns
- This is one of the most common Google interview discussion points

---

## Chapter 4 Summary

| Step | Operation | What it does |
|---|---|---|
| 1 | Q = X·W_Q, K = X·W_K, V = X·W_V | Project each token into query/key/value spaces |
| 2 | Scores = Q·Kᵀ | How much does each token's query match each key |
| 3 | Scale by √d_k | Prevent softmax saturation |
| 4 | Softmax row-wise | Convert scores to probability weights |
| 5 | Output = Weights · V | Blend values by attention weights |

The output is the same shape as the input `[seq_len × d_k]` — but every vector is now contextually aware of every other token.

---

One head doing this is powerful. But BERT runs **12 of these in parallel**, each with different W_Q, W_K, W_V matrices, each learning to attend to different types of relationships.

That's Chapter 5 — and it's where the real richness of BERT emerges. Ready?
