# Chapter 5: Multi-Head Attention

In Chapter 4, one attention head took X and produced a context-aware output. Let's start with a question before diving in.

---

## 5.1 What's Wrong With One Head?

Consider this sentence:

```
"The trophy didn't fit in the suitcase because it was too big"
```

To fully understand this sentence, you simultaneously need to track:

```
Question 1: What does "it" refer to? → "trophy" (co-reference)
Question 2: What didn't fit where?   → trophy didn't fit in suitcase (syntactic subject-object)
Question 3: Why didn't it fit?       → because it was too big (causal relationship)
Question 4: What was too big?        → the trophy (adjectival modification)
```

These are **four different types of relationships**, all needed at the same time.

A single attention head produces one attention weight matrix — one "view" of the sentence. It has to somehow capture all relationship types simultaneously in that one view. That creates competition. The head can't strongly attend to both the co-reference signal AND the syntactic signal at the same time — they pull in different directions.

**Multi-head attention runs multiple heads in parallel**, each free to specialize in a different relationship type.

---

## 5.2 The Architecture

BERT-base uses **h = 12 heads**, each operating in **d_k = 64 dimensions** (since 768/12 = 64).

Each head has its **own** projection matrices:

```
Head 1:  W_Q1 [768×64],  W_K1 [768×64],  W_V1 [768×64]
Head 2:  W_Q2 [768×64],  W_K2 [768×64],  W_V2 [768×64]
...
Head 12: W_Q12[768×64],  W_K12[768×64],  W_V12[768×64]
```

Each head independently runs the full attention mechanism from Chapter 4:

```
head_i = Attention(X·W_Qi, X·W_Ki, X·W_Vi)    output: [seq_len × 64]
```

---

## 5.3 Concatenation + Final Projection

After all 12 heads run:

```
head_1  output: [seq_len × 64]
head_2  output: [seq_len × 64]
...
head_12 output: [seq_len × 64]
```

Concatenate along the feature dimension:

```
Concat(head_1, ..., head_12) → [seq_len × 768]   (64 × 12 = 768)
```

Then multiply by a final learned projection matrix W_O:

```
W_O: [768 × 768]

MultiHead output = Concat(head_1,...,head_12) · W_O → [seq_len × 768]
```

**Why W_O?**
Concatenation just stacks the 12 subspaces side by side. W_O **mixes** information across all heads — it learns which combinations of head outputs are most useful, and projects back into a unified 768-d space.

---

## 5.4 Full Numerical Example — 2 Heads, d=4

Let's use 2 heads (instead of 12) and d=4 (instead of 768), with d_k=2 per head.

Input X (3 tokens × 4 dims):
```
  the → [0.31, -0.50,  0.96,  0.11]
  cat → [0.70,  0.33, -0.31,  0.73]
  sat → [0.10,  0.51,  0.30, -0.36]
```

### Head 1 — Learns Syntactic Relationships

**Projection matrices W_Q1, W_K1, W_V1 (4×2):**
```
W_Q1 = [[1.0, 0.0],    W_K1 = [[1.0, 0.0],    W_V1 = [[1.0, 0.0],
         [0.0, 1.0],            [0.0, 1.0],            [0.0, 1.0],
         [0.0, 0.0],            [0.0, 0.0],            [0.0, 0.0],
         [0.0, 0.0]]            [0.0, 0.0]]            [0.0, 0.0]]
```
(Takes first 2 dimensions of X — simplified)

```
Q1 = K1 = V1 =
  the → [0.31, -0.50]
  cat → [0.70,  0.33]
  sat → [0.10,  0.51]
```

**Scores = Q1 · K1ᵀ, scaled by √2 = 1.414:**

```
Raw scores:
  the→the: 0.31×0.31 + (-0.50×-0.50) = 0.096+0.250 = 0.346
  the→cat: 0.31×0.70 + (-0.50×0.33)  = 0.217-0.165 = 0.052
  the→sat: 0.31×0.10 + (-0.50×0.51)  = 0.031-0.255 = -0.224

  cat→the: 0.70×0.31 + 0.33×(-0.50)  = 0.217-0.165 = 0.052
  cat→cat: 0.70×0.70 + 0.33×0.33     = 0.490+0.109 = 0.599
  cat→sat: 0.70×0.10 + 0.33×0.51     = 0.070+0.168 = 0.238

  sat→the: 0.10×0.31 + 0.51×(-0.50)  = 0.031-0.255 = -0.224
  sat→cat: 0.10×0.70 + 0.51×0.33     = 0.070+0.168 = 0.238
  sat→sat: 0.10×0.10 + 0.51×0.51     = 0.010+0.260 = 0.270

Scaled (÷1.414):
  the → [0.245,  0.037, -0.158]
  cat → [0.037,  0.424,  0.168]
  sat → [-0.158, 0.168,  0.191]
```

**Softmax:**
```
  the → [0.390, 0.318, 0.292]
  cat → [0.263, 0.435, 0.302]
  sat → [0.251, 0.356, 0.393]
```

**Output Head 1 = Weights · V1:**
```
  the → 0.390×[0.31,-0.50] + 0.318×[0.70,0.33] + 0.292×[0.10,0.51]
      = [0.121,-0.195] + [0.223,0.105] + [0.029,0.149]
      = [0.373, 0.059]

  cat → 0.263×[0.31,-0.50] + 0.435×[0.70,0.33] + 0.302×[0.10,0.51]
      = [0.082,-0.132] + [0.305,0.144] + [0.030,0.154]
      = [0.417, 0.166]

  sat → 0.251×[0.31,-0.50] + 0.356×[0.70,0.33] + 0.393×[0.10,0.51]
      = [0.078,-0.126] + [0.249,0.118] + [0.039,0.200]
      = [0.366, 0.192]
```

**Head 1 output (3×2):**
```
  the → [0.373,  0.059]
  cat → [0.417,  0.166]
  sat → [0.366,  0.192]
```

---

### Head 2 — Learns Semantic Relationships

**Projection matrices take last 2 dimensions of X:**
```
W_Q2 = [[0.0, 0.0],    (takes dims 3 and 4)
         [0.0, 0.0],
         [1.0, 0.0],
         [0.0, 1.0]]
```

```
Q2 = K2 = V2 =
  the → [ 0.96,  0.11]
  cat → [-0.31,  0.73]
  sat → [ 0.30, -0.36]
```

**Scores = Q2 · K2ᵀ, scaled by √2:**

```
Raw scores:
  the→the:  0.96×0.96 + 0.11×0.11  = 0.922+0.012 = 0.934
  the→cat:  0.96×(-0.31)+0.11×0.73 = -0.298+0.080 = -0.218
  the→sat:  0.96×0.30 + 0.11×(-0.36)= 0.288-0.040 = 0.248

  cat→the: -0.31×0.96 + 0.73×0.11  = -0.298+0.080 = -0.218
  cat→cat: -0.31×(-0.31)+0.73×0.73 = 0.096+0.533  = 0.629
  cat→sat: -0.31×0.30 + 0.73×(-0.36)= -0.093-0.263 = -0.356

  sat→the:  0.30×0.96 +(-0.36)×0.11 = 0.288-0.040 = 0.248
  sat→cat:  0.30×(-0.31)+(-0.36)×0.73= -0.093-0.263 = -0.356
  sat→sat:  0.30×0.30 +(-0.36)×(-0.36)= 0.090+0.130 = 0.220

Scaled (÷1.414):
  the → [ 0.660, -0.154,  0.175]
  cat → [-0.154,  0.445, -0.252]
  sat → [ 0.175, -0.252,  0.156]
```

**Softmax:**
```
  the → [0.522, 0.232, 0.246]   ← "the" strongly attends to itself
  cat → [0.234, 0.435, 0.221]   ← "cat" attends mostly to itself
  sat → [0.362, 0.235, 0.353]   ← "sat" split between itself and "the"
```

**Output Head 2 = Weights · V2:**
```
  the → 0.522×[0.96,0.11] + 0.232×[-0.31,0.73] + 0.246×[0.30,-0.36]
      = [0.501,0.057] + [-0.072,0.169] + [0.074,-0.089]
      = [0.503, 0.138]

  cat → 0.234×[0.96,0.11] + 0.435×[-0.31,0.73] + 0.221×[0.30,-0.36]
      = [0.225,0.026] + [-0.135,0.318] + [0.066,-0.080]
      = [0.156, 0.264]

  sat → 0.362×[0.96,0.11] + 0.235×[-0.31,0.73] + 0.353×[0.30,-0.36]
      = [0.347,0.040] + [-0.073,0.172] + [0.106,-0.127]
      = [0.380, 0.085]
```

**Head 2 output (3×2):**
```
  the → [0.503,  0.138]
  cat → [0.156,  0.264]
  sat → [0.380,  0.085]
```

---

### Concatenate Both Heads

```
Head 1 output    Head 2 output    Concatenated (3×4)
the: [0.373, 0.059]  +  [0.503, 0.138]  =  [0.373, 0.059, 0.503, 0.138]
cat: [0.417, 0.166]  +  [0.156, 0.264]  =  [0.417, 0.166, 0.156, 0.264]
sat: [0.366, 0.192]  +  [0.380, 0.085]  =  [0.366, 0.192, 0.380, 0.085]
```

---

### Apply W_O (4×4) → Final Output

```
W_O = [[0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5]]   ← simplified mixing matrix
```

**Final MultiHead output:**
```
  the → [0.373×0.5+0.059×0.0+0.503×0.5+0.138×0.0,  ...]
      = [0.438, 0.099, 0.438, 0.099]

  cat → [0.417×0.5+0.166×0.0+0.156×0.5+0.264×0.0,  ...]
      = [0.287, 0.215, 0.287, 0.215]

  sat → [0.366×0.5+0.192×0.0+0.380×0.5+0.085×0.0,  ...]
      = [0.373, 0.139, 0.373, 0.139]
```

**Final output shape: [3 tokens × 4 dims] — same as input X.**

---

## 5.5 What Each Head Learns in Real BERT

Research has probed BERT's 12 heads and found consistent specializations:

```
Head Type          What it attends to
──────────────────────────────────────────────────────
Syntactic heads    Subject → verb, verb → object
Co-reference heads "it", "they", "he" → the noun they refer to
Positional heads   Always attend to next token, or previous token
Separator heads    Everything attends heavily to [CLS] or [SEP]
Rare/noisy heads   No clear pattern — may be redundant
```

This is why **attention head pruning** works — you can remove 30-40% of heads with minimal accuracy loss. Some heads are redundant.

---

## 5.6 The Parameter Count

For BERT-base (12 heads, d_model=768, d_k=64):

```
Per head:    W_Q + W_K + W_V = 3 × (768×64) = 147,456 params
12 heads:    12 × 147,456    = 1,769,472 params
W_O:         768 × 768       = 589,824 params
─────────────────────────────────────────────────
Total per layer:              = 2,359,296 params (~2.4M)
× 12 layers:                  = 28,311,552 params (~28M)
```

Just for attention. The FFN adds roughly the same again. Total BERT-base: 110M params.

---

## 5.7 Why This Design Is Elegant

```
Single head attention:
  One 768-d space trying to capture everything
  → Competition between relationship types
  → Weaker representations

Multi-head attention:
  12 independent 64-d subspaces
  Each free to specialize
  Concatenated and mixed by W_O
  → Rich, multi-faceted representations
```

The beauty is that **no one tells head 3 to learn co-reference or head 7 to learn syntax**. They specialize purely through gradient descent, because specialization produces better predictions during pre-training.

---

## Chapter 5 Summary

```
Input X [seq_len × 768]
    ↓
Split into 12 heads, each projects to 64-d Q, K, V
    ↓
Each head runs full attention independently  → [seq_len × 64] each
    ↓
Concatenate all 12 heads                    → [seq_len × 768]
    ↓
Multiply by W_O [768 × 768]                 → [seq_len × 768]
    ↓
Output: same shape as input, but richer
```

| Design Choice | Why |
|---|---|
| 12 heads | Multiple relationship types simultaneously |
| 64-d per head | 768/12 — keeps total computation same as one 768-d head |
| Separate W_Q/K/V per head | Each head sees different projections, learns different patterns |
| W_O at end | Mixes across heads, projects back to unified space |

---

Now you have the output of multi-head attention — shape `[seq_len × 768]`. But this is only **half** of one Transformer block. Before passing to the next block, this output goes through two more operations: a residual connection, a layer norm, and then a Feed-Forward Network.

That's Chapter 6 and 7 — and they're shorter but critical. Ready?
