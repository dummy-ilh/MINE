# Chapter 7: Residual Connections & Layer Normalization

Let's start with a brutal fact.

Before residual connections were invented (2015, ResNet paper), training networks deeper than ~20 layers was essentially **impossible**. Gradients vanished. Accuracy got **worse** as you added more layers, not better.

BERT has 12-24 layers. Without the two techniques in this chapter, it would never have trained at all.

---

## 7.1 The Problem: Vanishing Gradients in Deep Networks

To understand why these techniques exist, you need to feel the problem first.

During backpropagation, gradients flow **backwards** through every layer. At each layer, the gradient gets multiplied by the layer's local gradient (via chain rule).

```
∂Loss/∂Layer_1 = ∂Loss/∂Layer_12 × ∂Layer_12/∂Layer_11 × ... × ∂Layer_2/∂Layer_1
```

If each local gradient is slightly less than 1 — say 0.9 — then:

```
Layer 12 gradient:  1.000
Layer 11 gradient:  0.900
Layer 10 gradient:  0.810
Layer  9 gradient:  0.729
...
Layer  1 gradient:  0.9^12 = 0.282
```

By the time you reach layer 1, the gradient is 28% of what it was at layer 12. That's survivable.

But with slightly worse local gradients — say 0.7:

```
Layer 12 gradient:  1.000
Layer  6 gradient:  0.7^6  = 0.118
Layer  1 gradient:  0.7^12 = 0.014
```

Layer 1 gets 1.4% of the gradient signal. It barely learns. Early layers effectively stop updating — and in a deep network, the early layers are where basic linguistic features are learned. The whole model degrades.

**This is vanishing gradients.** And it gets exponentially worse with depth.

---

## 7.2 Residual Connections — The Fix

The insight from the 2015 ResNet paper was elegant:

**Instead of learning a transformation F(x), learn the residual F(x) - x.**

In practice, this means adding the input directly to the output:

```
output = x + F(x)
```

Where F(x) is whatever sub-layer you're applying (attention or FFN).

In BERT, this appears twice per block:

```
After Attention:  a = LayerNorm( x + MultiHeadAttention(x) )
After FFN:        output = LayerNorm( a + FFN(a) )
```

### Why Does This Fix Vanishing Gradients?

The gradient of the residual output with respect to the input:

```
∂(x + F(x))/∂x = 1 + ∂F(x)/∂x
```

There's always a **+1 term**. No matter how small ∂F(x)/∂x becomes, the gradient is at least 1. It can never vanish.

Think of it as a **gradient highway** — a direct path from the loss all the way back to layer 1, bypassing all the intermediate transformations.

```
Without residuals:
  Loss → Layer 12 → Layer 11 → ... → Layer 1
  Gradient shrinks at every step

With residuals:
  Loss ──────────────────────────────→ Layer 1  (direct highway)
       ↘ Layer 12 ↘ Layer 11 ↘ ...  ↗          (also flows normally)
```

### The Identity Mapping Insight

There's a second benefit. If a layer is not useful, the model can learn to set F(x) ≈ 0, making:

```
output = x + F(x) ≈ x + 0 = x
```

The layer becomes an **identity function** — it passes input through unchanged. Without residuals, a useless layer would distort the signal. With residuals, it gracefully does nothing.

This is why BERT can have 12 layers without every layer being forced to contribute — some layers can specialize while others act as near-identity mappings.

---

## 7.3 Numerical Example — Residual Connection

Continuing from Chapter 6. Before attention, "cat" had:

```
x_cat (input to block) = [0.70,  0.33, -0.31,  0.73]
```

After multi-head attention, "cat" became:

```
Attention(x_cat) = [0.451, 0.170,  0.160,  0.300]
```

**Apply residual — add input to attention output:**

```
x_cat           = [ 0.700,  0.330, -0.310,  0.730]
Attention output = [ 0.451,  0.170,  0.160,  0.300]
                   ────────────────────────────────
x + Attention   = [ 1.151,  0.500, -0.150,  1.030]
```

Notice: dim3 was -0.31 (negative). After adding residual it's -0.150 — the original signal is preserved and mixed with the attention output. The attention didn't overwrite the original information — it **added to it**.

Then LayerNorm is applied to this sum. (We'll compute that next.)

Then after FFN:

```
a (after first LayerNorm) = [...normalized version of above...]

FFN(a) = [0.291, 0.070, 0.394, -0.006]   (from Chapter 6)

Second residual:
a + FFN(a) = a + [0.291, 0.070, 0.394, -0.006]
```

Again — FFN output is **added** to what went in, not replacing it.

---

## 7.4 Layer Normalization

Residual connections solve vanishing gradients. But they introduce a new problem: as you add x + F(x) repeatedly across 12 layers, the **scale of vectors can explode or shift dramatically**.

Layer normalization keeps every vector on a stable scale throughout the network.

### The Formula

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
```

Where:
- **μ** = mean of x across the feature dimension
- **σ** = standard deviation of x across the feature dimension  
- **ε** = tiny constant (1e-8) to prevent division by zero
- **γ, β** = learned scale and shift parameters (same size as x)

### What It Does Step by Step

1. **Subtract mean** → centers the vector around 0
2. **Divide by std** → normalizes scale to ~1
3. **Scale by γ** → learned rescaling
4. **Shift by β** → learned recentering

Steps 1-2 are fixed normalization. Steps 3-4 let the model undo the normalization if needed — the network can learn "actually for this layer, a mean of 2.0 is better."

### Why Across Features, Not Batch?

This is the key difference from **BatchNorm** (common in CNNs):

```
BatchNorm:  normalize across the batch dimension
            → statistics depend on other samples in the batch
            → unstable for variable-length sequences
            → different behavior at train vs inference

LayerNorm:  normalize across the feature dimension (768 dims)
            → statistics computed per token, per sample
            → no dependence on batch size or other samples
            → identical behavior at train and inference
            → works perfectly for NLP
```

---

## 7.5 Full Numerical Example — LayerNorm

**Input vector** (result of x + Attention(x) from above):

```
v = [1.151, 0.500, -0.150, 1.030]    (d=4)
```

### Step 1: Compute Mean

```
μ = (1.151 + 0.500 + (-0.150) + 1.030) / 4
  = 2.531 / 4
  = 0.633
```

### Step 2: Compute Standard Deviation

```
Deviations from mean:
  1.151 - 0.633 =  0.518
  0.500 - 0.633 = -0.133
 -0.150 - 0.633 = -0.783
  1.030 - 0.633 =  0.397

Squared deviations:
   0.518² = 0.268
  -0.133² = 0.018
  -0.783² = 0.613
   0.397² = 0.158

Variance = (0.268 + 0.018 + 0.613 + 0.158) / 4
         = 1.057 / 4
         = 0.264

σ = √0.264 = 0.514
```

### Step 3: Normalize

```
x_norm = (v - μ) / (σ + ε)

dim1: (1.151 - 0.633) / 0.514 =  0.518 / 0.514 =  1.008
dim2: (0.500 - 0.633) / 0.514 = -0.133 / 0.514 = -0.259
dim3:(-0.150 - 0.633) / 0.514 = -0.783 / 0.514 = -1.524
dim4: (1.030 - 0.633) / 0.514 =  0.397 / 0.514 =  0.772

x_norm = [1.008, -0.259, -1.524, 0.772]
```

**Check:** mean ≈ 0, std ≈ 1. ✓

### Step 4: Scale and Shift

With learned γ and β (assume γ=[1,1,1,1], β=[0,0,0,0] for simplicity — identity transform):

```
LayerNorm output = γ · x_norm + β
                 = [1.008, -0.259, -1.524, 0.772]
```

In a trained model, γ and β are non-trivial values learned to rescale each dimension optimally for that layer.

---

## 7.6 Where Exactly Do These Appear in a Block?

BERT uses **Post-LayerNorm** (original Transformer) arrangement:

```
┌─────────────────────────────────────┐
│           TRANSFORMER BLOCK          │
│                                      │
│  x ──────────────────────┐          │
│  ↓                        ↓          │
│  MultiHeadAttention(x)    │          │
│  ↓                        │          │
│  + ←──────────────────────┘  (residual 1)
│  ↓                                   │
│  LayerNorm                           │
│  ↓                                   │
│  a ──────────────────────┐          │
│  ↓                        ↓          │
│  FFN(a)                   │          │
│  ↓                        │          │
│  + ←──────────────────────┘  (residual 2)
│  ↓                                   │
│  LayerNorm                           │
│  ↓                                   │
│  output                              │
└─────────────────────────────────────┘
```

Two residuals. Two LayerNorms. Per block. 12 blocks = 24 residuals, 24 LayerNorms.

---

## 7.7 What Happens Without Each?

### Without Residual Connections:

```
Gradient at layer 1 after 12 layers: ≈ 0
Early layers don't update
Model learns only shallow features
Performance collapses with depth
BERT-12 would perform like a 2-layer model
```

### Without Layer Normalization:

```
After each residual addition x + F(x):
  Vector scales grow: [0.7] → [1.5] → [3.1] → [7.2] → ...
  
By layer 12: vectors have exploded in magnitude
Softmax in attention saturates → one-hot attention weights
FFN inputs are huge → GELU output is saturated
Gradients explode in the other direction
Training diverges
```

### Without Both:

```
Training a 12-layer Transformer is essentially impossible.
This isn't theoretical — it was empirically verified before
these techniques existed.
```

---

## 7.8 The Interaction Between Residuals and LayerNorm

They work together as a system:

```
Residual:   ensures gradient flows back unimpeded
LayerNorm:  ensures forward activations stay at reasonable scale

Without residual → gradients die going backward
Without LayerNorm → activations explode going forward
Together → stable training at any depth
```

This pairing is one of the most important engineering insights in modern deep learning. It's what made transformers with 12, 24, 96, and even 1000+ layers feasible.

---

## Chapter 7 Summary

### Residual Connections
```
output = x + SubLayer(x)

Effect on gradient:   ∂output/∂x = 1 + ∂F/∂x   (never vanishes)
Effect on signal:     input is preserved, SubLayer adds to it
Effect on depth:      makes 12-24 layers trainable
```

### Layer Normalization
```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

Normalizes:   across 768 feature dimensions, per token
Stabilizes:   forward pass scale throughout all 12 layers
vs BatchNorm: per-sample (no batch dependence), same at train/inference
```

### One Complete Transformer Block

```
Input x
  ↓
  x + MultiHeadAttention(x) → LayerNorm  →  a
                                              ↓
                              a + FFN(a) → LayerNorm  →  output
```

**Output shape = Input shape = [seq_len × 768]**

---

You now have every single component of one Transformer block. You know exactly what happens inside it, why each piece exists, and what breaks without it.

Chapter 8 is where we **stack 12 of these blocks** and ask: what changes as you go deeper? What does layer 1 see that layer 12 doesn't? How does the [CLS] token accumulate meaning across 12 layers?

Ready?
