# Chapter 6: The Feed-Forward Network

You're now halfway through one Transformer block. Here's where you are:

```
Input X [seq_len × 768]
        ↓
  Multi-Head Attention    ← done (Chapters 4 & 5)
        ↓
  Output [seq_len × 768]  ← you are here
        ↓
  Feed-Forward Network    ← this chapter
        ↓
  Output [seq_len × 768]  → goes to next block
```

---

## 6.1 The Core Intuition

Before explaining the math, understand **what problem the FFN solves.**

Attention answers: *"which tokens should I look at?"*

But attention is fundamentally a **weighted average** of Value vectors. Weighted averages are linear operations. And linear operations stacked on top of each other are still just one linear operation — no matter how many layers deep.

**Without the FFN, 12 Transformer blocks collapse into one.**

The FFN introduces **non-linearity** after every attention step. This is what makes depth meaningful — each layer can learn genuinely more complex transformations than the last.

Think of it this way:

```
Attention  → figures out WHICH information to gather from context
FFN        → figures out WHAT TO DO with that gathered information
```

Attention is the gathering step. FFN is the thinking step.

---

## 6.2 The Structure

The FFN is applied **independently to each token**. It does not mix information across tokens — that's attention's job. It transforms each token's vector in place.

```
FFN(x) = GELU( x · W1 + b1 ) · W2 + b2
```

Two linear layers with a non-linearity in between.

**Dimensions in BERT-base:**

```
Input x:          [768]
W1:               [768 × 3072]    ← expand to 4× hidden size
b1:               [3072]
After GELU:       [3072]
W2:               [3072 × 768]    ← compress back
b2:               [768]
Output:           [768]
```

For the full sequence: input `[seq_len × 768]` → output `[seq_len × 768]`.

---

## 6.3 Why 4× Expansion?

This is one of the most common interview questions about BERT.

When you expand to 3072 dimensions, you're creating a **high-dimensional thinking space**. In this space, features that were entangled in 768 dimensions can be **pulled apart, operated on, and recombined**.

An analogy: imagine trying to sort objects in a tiny room (768-d). Hard — everything is crammed together. Move them to a large hall (3072-d), sort them properly, then pack them back into the small room (768-d). The final packing is much more organized.

**Why exactly 4×?** Empirically chosen in the original Transformer paper and it stuck. Later research (GPT-3, PaLM) has tried other ratios, but 4× remains a strong default.

**Parameter count for FFN per layer:**
```
W1:  768 × 3072  = 2,359,296
W2:  3072 × 768  = 2,359,296
b1:  3072         = 3,072
b2:  768          = 768
─────────────────────────────
Total per layer:  ~4.7M params
× 12 layers:      ~56.6M params
```

More than half of BERT's 110M parameters live in the FFN layers.

---

## 6.4 GELU — Why Not ReLU?

The activation function between W1 and W2 is **GELU** (Gaussian Error Linear Unit), not the more familiar ReLU.

### ReLU:
```
ReLU(x) = max(0, x)

  x = -2.0 → 0.000   (hard zero)
  x = -0.5 → 0.000   (hard zero)
  x =  0.0 → 0.000
  x =  0.5 → 0.500
  x =  2.0 → 2.000
```

**Problem:** Any negative input produces exactly zero output AND zero gradient. That neuron is dead for that input — no learning signal flows back.

### GELU:
```
GELU(x) ≈ x · Φ(x)

where Φ(x) is the cumulative distribution function of the standard normal.

Approximation used in practice:
GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
```

**In plain English:** GELU smoothly gates the input. Large positive values pass through almost unchanged. Large negative values are almost zeroed. But near zero, it's a smooth curve — not a hard cutoff.

```
  x = -2.0 → -0.045   (small negative, not hard zero)
  x = -1.0 → -0.159   (some signal preserved)
  x = -0.5 → -0.154
  x =  0.0 →  0.000
  x =  0.5 →  0.346
  x =  1.0 →  0.841
  x =  2.0 →  1.955
```

### Side by Side:

```
x      ReLU(x)    GELU(x)    Difference
─────────────────────────────────────────
-2.0    0.000     -0.045     GELU passes small signal
-1.0    0.000     -0.159     GELU passes small signal
-0.5    0.000     -0.154     GELU passes small signal
 0.0    0.000      0.000     Same
 0.5    0.500      0.346     GELU slightly lower
 1.0    1.000      0.841     GELU slightly lower
 2.0    2.000      1.955     Almost identical
```

**Why does this matter for training?**

ReLU's hard zero means the gradient is also exactly zero for all negative inputs — the network gets no feedback about whether being "more negative" or "less negative" was better.

GELU's smooth curve means gradients flow even for slightly negative values — the network can learn from them.

**The intuition:** GELU says "probably let this through" for positive values and "probably suppress this" for negative values — based on a probabilistic interpretation. It's a softer, more informative gate.

---

## 6.5 Full Numerical Example

Let's run one token through the FFN. We'll use d=4 (instead of 768) and expand to d_ff=8 (instead of 3072).

**Input — "cat" vector after attention (from Chapter 4):**
```
x = [0.451, 0.170, 0.160, 0.300]
```

### Step 1: First Linear Layer — x · W1 + b1

W1 (4×8) — learned weights (simplified example):
```
W1 = [[ 0.5,  0.3, -0.2,  0.8,  0.1, -0.4,  0.6,  0.2],
      [ 0.1, -0.5,  0.7,  0.2, -0.3,  0.9, -0.1,  0.4],
      [-0.3,  0.6,  0.4, -0.1,  0.8, -0.2,  0.3, -0.5],
      [ 0.7, -0.2,  0.1,  0.5, -0.6,  0.3,  0.8, -0.1]]

b1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   (zeros for simplicity)
```

Computing x · W1:
```
dim1: 0.451×0.5 + 0.170×0.1 + 0.160×(-0.3) + 0.300×0.7
    = 0.226 + 0.017 - 0.048 + 0.210 = 0.405

dim2: 0.451×0.3 + 0.170×(-0.5) + 0.160×0.6 + 0.300×(-0.2)
    = 0.135 - 0.085 + 0.096 - 0.060 = 0.086

dim3: 0.451×(-0.2) + 0.170×0.7 + 0.160×0.4 + 0.300×0.1
    = -0.090 + 0.119 + 0.064 + 0.030 = 0.123

dim4: 0.451×0.8 + 0.170×0.2 + 0.160×(-0.1) + 0.300×0.5
    = 0.361 + 0.034 - 0.016 + 0.150 = 0.529

dim5: 0.451×0.1 + 0.170×(-0.3) + 0.160×0.8 + 0.300×(-0.6)
    = 0.045 - 0.051 + 0.128 - 0.180 = -0.058

dim6: 0.451×(-0.4) + 0.170×0.9 + 0.160×(-0.2) + 0.300×0.3
    = -0.180 + 0.153 - 0.032 + 0.090 = 0.031

dim7: 0.451×0.6 + 0.170×(-0.1) + 0.160×0.3 + 0.300×0.8
    = 0.271 - 0.017 + 0.048 + 0.240 = 0.542

dim8: 0.451×0.2 + 0.170×0.4 + 0.160×(-0.5) + 0.300×(-0.1)
    = 0.090 + 0.068 - 0.080 - 0.030 = 0.048
```

**After W1 (pre-activation):**
```
h = [0.405, 0.086, 0.123, 0.529, -0.058, 0.031, 0.542, 0.048]
```

### Step 2: Apply GELU

```
GELU(0.405)  ≈ 0.405 × 0.657 = 0.266   (moderately positive → mostly passes)
GELU(0.086)  ≈ 0.086 × 0.524 = 0.045   (small positive → half passes)
GELU(0.123)  ≈ 0.123 × 0.549 = 0.068
GELU(0.529)  ≈ 0.529 × 0.701 = 0.371   (larger positive → mostly passes)
GELU(-0.058) ≈ -0.058 × 0.477 = -0.028  (small negative → mostly suppressed but not zero)
GELU(0.031)  ≈ 0.031 × 0.508 = 0.016
GELU(0.542)  ≈ 0.542 × 0.706 = 0.383   (large positive → mostly passes)
GELU(0.048)  ≈ 0.048 × 0.512 = 0.025
```

**After GELU:**
```
h_activated = [0.266, 0.045, 0.068, 0.371, -0.028, 0.016, 0.383, 0.025]
```

Notice dim5 = -0.028: **not zero**. With ReLU it would be exactly 0. GELU kept a small signal.

### Step 3: Second Linear Layer — h_activated · W2 + b2

W2 (8×4) — projects back to original dimension:
```
W2 = [[ 0.4, -0.1,  0.3,  0.2],
      [-0.2,  0.6,  0.1, -0.3],
      [ 0.5,  0.2, -0.4,  0.1],
      [ 0.1,  0.3,  0.5, -0.2],
      [-0.3, -0.1,  0.2,  0.4],
      [ 0.2,  0.4, -0.1,  0.3],
      [ 0.3, -0.2,  0.4,  0.1],
      [-0.1,  0.5,  0.2, -0.4]]

b2 = [0.0, 0.0, 0.0, 0.0]
```

**Computing h_activated · W2:**
```
dim1: 0.266×0.4 + 0.045×(-0.2) + 0.068×0.5 + 0.371×0.1
    + (-0.028)×(-0.3) + 0.016×0.2 + 0.383×0.3 + 0.025×(-0.1)
    = 0.106 - 0.009 + 0.034 + 0.037 + 0.008 + 0.003 + 0.115 - 0.003
    = 0.291

dim2: 0.266×(-0.1) + 0.045×0.6 + 0.068×0.2 + 0.371×0.3
    + (-0.028)×(-0.1) + 0.016×0.4 + 0.383×(-0.2) + 0.025×0.5
    = -0.027 + 0.027 + 0.014 + 0.111 + 0.003 + 0.006 - 0.077 + 0.013
    = 0.070

dim3: 0.266×0.3 + 0.045×0.1 + 0.068×(-0.4) + 0.371×0.5
    + (-0.028)×0.2 + 0.016×(-0.1) + 0.383×0.4 + 0.025×0.2
    = 0.080 + 0.005 - 0.027 + 0.186 - 0.006 - 0.002 + 0.153 + 0.005
    = 0.394

dim4: 0.266×0.2 + 0.045×(-0.3) + 0.068×0.1 + 0.371×(-0.2)
    + (-0.028)×0.4 + 0.016×0.3 + 0.383×0.1 + 0.025×(-0.4)
    = 0.053 - 0.014 + 0.007 - 0.074 - 0.011 + 0.005 + 0.038 - 0.010
    = -0.006
```

**FFN Output for "cat":**
```
FFN(cat) = [0.291, 0.070, 0.394, -0.006]
```

**Compare to what went in:**
```
Input to FFN:   [0.451, 0.170, 0.160,  0.300]
Output of FFN:  [0.291, 0.070, 0.394, -0.006]
```

The vector has been **non-linearly transformed**. Information has been reorganized across dimensions. This is what the FFN adds that attention alone cannot.

---

## 6.6 The FFN Is Per-Token, Not Cross-Token

This is a subtle but important point.

**Attention:** "cat" at position 2 looks at "the" at position 1 and "sat" at position 3. Cross-token by design.

**FFN:** "cat" at position 2 goes through FFN independently. The FFN at position 2 receives no information from positions 1 or 3.

```
Attention:   token_2 ←→ token_1, token_3    (cross-token)
FFN:         token_2 only                   (per-token)
```

So what is the FFN actually doing if not mixing tokens?

It's acting as a **memory** — storing factual knowledge learned during pre-training in its weights. Research has shown that:

```
"The capital of France is ___"
```

The FFN layers are where BERT "recalls" that the answer is "Paris." This knowledge is encoded in the W1 and W2 weights, not in the attention patterns.

This is why the FFN is sometimes called the **"key-value memory"** of the Transformer.

---

## 6.7 What Happens Without the FFN?

```
Without FFN:
  Block = Attention only
  Stacking 12 attention layers = still a linear operation
  (linear × linear × linear = linear)
  Model capacity collapses
  Performance degrades severely

With FFN:
  Non-linearity after every attention step
  Each layer genuinely more expressive than the last
  Depth becomes meaningful
```

Ablation studies show removing FFN layers hurts more than removing attention heads — the FFN is arguably the more critical component for raw capacity.

---

## Chapter 6 Summary

```
FFN(x) = GELU(x · W1 + b1) · W2 + b2

Input:       [seq_len × 768]
Expand:      [seq_len × 3072]   via W1
Activate:    GELU (smooth, non-zero for negatives)
Compress:    [seq_len × 768]    via W2
Output:      [seq_len × 768]
```

| Design Choice | Why |
|---|---|
| 4× expansion | Creates high-dimensional thinking space |
| GELU not ReLU | Smooth gradient flow, small signal preserved for negatives |
| Per-token (not cross-token) | Token mixing is attention's job; FFN does per-token transformation |
| After every attention block | Ensures depth is non-trivially expressive |
| Large param count (~56M) | Stores factual world knowledge learned during pre-training |

---

Now you have both halves of a Transformer block:
1. Multi-Head Attention — gathers context across tokens
2. FFN — transforms each token non-linearly

But these two operations alone would make training 12 layers deep nearly impossible. There are two more components — **residual connections** and **layer normalization** — that are the reason deep BERT actually trains at all.

That's Chapter 7. Ready?
