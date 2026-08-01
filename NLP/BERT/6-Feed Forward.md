# Chapter 6: The Feed-Forward Network (FFN) — Master Notes

*Apple MLE interview prep — self-contained, boosted version*

---

## 0. Where This Fits (recap)

```
Input X [seq_len × 768]
        ↓
  Multi-Head Attention    ← done (Ch 4 & 5) — "which tokens matter?"
        ↓
  Output [seq_len × 768]
        ↓
  Feed-Forward Network    ← THIS CHAPTER — "what do I do with what I gathered?"
        ↓
  Output [seq_len × 768]  → next block
```

One-line summary you should be able to say out loud in an interview:

> "Attention moves information *between* tokens. The FFN transforms information *within* each token, and it's the part of the Transformer that adds non-linearity and stores most of the learned knowledge."

---

## 1. Why the FFN Exists At All (the "what if we don't" question)

This is the single most important interview question in this chapter, so let's over-explain it.

**Claim:** Attention alone, no matter how many layers you stack, is mathematically still just *one* linear transformation.

**Why?** Attention output = a weighted average of Value vectors. A weighted average is a linear combination. And here's the key fact from linear algebra:

> Linear function ∘ Linear function ∘ Linear function ... = still just ONE linear function.

Concretely: if `f(x) = Ax` and `g(x) = Bx`, then `g(f(x)) = B(Ax) = (BA)x` — which is *still* just some matrix `C = BA` times `x`. Stack 12 of these and you still get `Cx` for some single matrix `C`.

**So what happens if you delete the FFN from every block?**

- 12 attention-only layers mathematically collapse into behavior equivalent to roughly *one* linear layer.
- The model can no longer represent curved decision boundaries, XOR-like logic, or any function that isn't a straight-line transformation of the input.
- Empirically (ablation studies), removing FFN layers hurts performance **more** than removing attention heads — meaning the FFN carries more of the "raw modeling capacity" than people usually assume.

**Analogy:** Attention is like a research assistant who runs around the library and brings back all the relevant books to your desk (gathering). The FFN is you actually sitting down, reading, and reasoning about what those books mean (thinking/transforming). If you only ever gathered books and never read/thought about them, more research assistants wouldn't help you write a better essay.

---

## 2. The Structure — Simplified

### 2.1 The equation, in three readable pieces

Instead of memorizing one dense line, break it into 3 steps:

```
Step 1 (expand):    h = x · W1 + b1        [768 → 3072]
Step 2 (activate):  a = GELU(h)            [3072 → 3072, elementwise]
Step 3 (compress):  y = a · W2 + b2        [3072 → 768]
```

Full equation (same thing, one line):
```
FFN(x) = GELU(x·W1 + b1)·W2 + b2
```

If someone asks you to "write the FFN equation" in an interview, write it as the 3 steps above first — it shows you understand *why* it's shaped that way, not just that you memorized a formula.

### 2.2 Dimensions (BERT-base)

| Component | Shape | Role |
|---|---|---|
| x | [768] | one token's vector |
| W1 | [768 × 3072] | expand ("open up" the representation) |
| b1 | [3072] | bias for expansion |
| GELU | [3072] → [3072] | non-linearity, elementwise |
| W2 | [3072 × 768] | compress back down |
| b2 | [768] | bias for compression |
| output | [768] | same shape as input |

**Critical property:** input shape = output shape. This is *required* because the FFN output gets added back to the input via a residual connection (Chapter 7) — you can't add two vectors of different sizes.

### 2.3 Parameter count — why this matters for interviews

```
W1:  768 × 3072  = 2,359,296
W2:  3072 × 768  = 2,359,296
b1:               3,072
b2:                 768
─────────────────────────
≈ 4.7M params per layer × 12 layers ≈ 56.6M params
```

**Interview-relevant fact:** this is **more than half** of BERT-base's ~110M total parameters. If someone asks "where do most of a Transformer's parameters live?" — the answer is the FFN, not attention. This surprises a lot of people, and interviewers like testing it.

---

## 3. Why Expand to 4× and Not Just Transform In-Place?

**What if you skipped the expansion and did `GELU(x·W + b)` at 768 dimensions the whole way through?**

You *could*, technically — but you'd lose modeling capacity. Here's the intuition:

- In 768-d space, features are packed tightly together and entangled (e.g., "is this token a verb," "is this token negated," "is this the subject" might all be smeared across overlapping directions).
- Projecting up to 3072-d gives the network **more independent directions** to temporarily separate those entangled features, apply a non-linear operation to each, and then recombine them intelligently when compressing back down.

**Analogy (kept from original, it's a good one):** sorting objects in a small room is hard because everything overlaps; move them to a large hall, sort properly, then repack into the small room — the final packing is more organized than if you'd tried to sort in the cramped room the whole time.

**Why 4× specifically and not 2× or 8×?**
- It's an empirical choice from the original "Attention Is All You Need" paper — not derived from theory.
- It stuck because it's a good capacity/cost tradeoff. Going higher (e.g., 8×) roughly doubles FFN compute and params for diminishing returns; going lower (2×) measurably hurts performance on many benchmarks.
- Later architectures experimented with different ratios (e.g., some LLaMA-style models use ~2.7× with gated variants like SwiGLU, because the gating mechanism itself adds effective capacity, letting them use a smaller expansion ratio for similar performance).

**Interview answer template:** *"4× is an empirically chosen expansion factor that gives the FFN a higher-dimensional space to disentangle features before compressing back to the residual stream dimension. It's not derived from a proof — it's a capacity/efficiency tradeoff that has held up empirically across many model families, though modern architectures sometimes deviate from it when combined with gating mechanisms."*

---

## 4. GELU vs ReLU — Simplified and Deepened

### 4.1 The core difference in one sentence

> ReLU makes a **hard binary decision** (pass or kill) at zero. GELU makes a **soft, probabilistic decision** that gradually shifts from "mostly kill" to "mostly pass."

### 4.2 The equations, simplified

**ReLU** — dead simple:
```
ReLU(x) = max(0, x)
```
Just: if negative, output 0. If positive, output the value unchanged.

**GELU** — think of it as "ReLU with a soft, probability-based gate" instead of a hard cutoff:
```
GELU(x) ≈ x · Φ(x)
```
Where `Φ(x)` is just "the probability that a standard normal random variable is less than x." You don't need to memorize the CDF formula — just remember:

- `Φ(x)` is close to **0** when x is very negative → gate mostly closed → output ≈ 0
- `Φ(x)` is close to **1** when x is very positive → gate mostly open → output ≈ x
- `Φ(x) = 0.5` exactly at x = 0 → output = 0.5·0 = 0

The practical approximation formula used in real implementations:
```
GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
```
You should recognize this exists and know it's an approximation of `x·Φ(x)` — you do not need to derive it from scratch in an interview. What you *do* need to explain is the conceptual gate idea above.

### 4.3 Numbers side by side (from the original chapter, kept — this table is genuinely useful to have memorized in rough shape)

| x | ReLU(x) | GELU(x) | What's happening |
|---|---|---|---|
| -2.0 | 0.000 | -0.045 | GELU lets a tiny negative signal through |
| -1.0 | 0.000 | -0.159 | still small negative, not dead |
| -0.5 | 0.000 | -0.154 | still nonzero |
| 0.0 | 0.000 | 0.000 | identical |
| 0.5 | 0.500 | 0.346 | GELU is slightly more conservative near zero |
| 1.0 | 1.000 | 0.841 | GELU still slightly damps it |
| 2.0 | 2.000 | 1.955 | at large x, GELU ≈ ReLU (both just pass the value) |

**Key visual takeaway:** far from zero, GELU and ReLU nearly agree. The entire difference lives in a narrow band near zero — but that narrow band is exactly where most pre-activation values land during training, so the difference matters a lot in aggregate.

### 4.4 Why does the "dead zone" of ReLU actually hurt training?

This is the "what if we don't" for GELU.

**What happens if you use ReLU and a token's pre-activation value is negative?**
1. Output = exactly 0.
2. **Gradient = exactly 0** (the slope of `max(0,x)` for x<0 is 0).
3. During backpropagation, **zero gradient means zero learning signal** flows back through that neuron for that example.
4. If a neuron gets consistently negative inputs across most training examples, its weights never update → **"dying ReLU"** — a neuron that's permanently stuck outputting 0 and contributes nothing to the model, forever.

**What does GELU fix?**
- Since GELU is smooth and never has a perfectly flat zero-gradient region (except in the limit as x → -∞), even negative pre-activations still carry *some* gradient.
- The network gets a nonzero learning signal even for "probably-suppress-this" inputs, so it can still adjust and recover if a neuron starts getting mostly-negative inputs.
- This is a big part of why GELU (and similar smooth activations like Swish/SiLU) became standard in Transformers — dying neurons are a real, practical training failure mode.

**Interview one-liner:** *"ReLU's hard zero means dead neurons get zero gradient and can never recover. GELU's smooth curve keeps a small gradient flowing everywhere, so the network can still adjust neurons that received negative pre-activations."*

---

## 5. Full Numerical Walkthrough (small example, d=4 → d_ff=8)

This example (with real numbers) is the best thing to be able to reproduce by hand or explain step-by-step in an interview — it proves you actually understand the mechanics, not just the words.

**Setup:** token "cat" after attention:
```
x = [0.451, 0.170, 0.160, 0.300]
```

### Step 1 — Expand: `h = x · W1 + b1`

With the given W1 (4×8, b1 = 0):
```
h = [0.405, 0.086, 0.123, 0.529, -0.058, 0.031, 0.542, 0.048]
```

*How each number is computed (dim1 shown as an example — same pattern for all 8):*
```
h[1] = x[1]·W1[1,1] + x[2]·W1[2,1] + x[3]·W1[3,1] + x[4]·W1[4,1]
     = 0.451×0.5 + 0.170×0.1 + 0.160×(-0.3) + 0.300×0.7
     = 0.226 + 0.017 - 0.048 + 0.210 = 0.405
```
Each output dimension is just a dot product of the input vector with one column of W1. This is the "each output neuron looks at the whole input and forms its own opinion" pattern — the same pattern used in every fully-connected layer you'll ever see.

### Step 2 — Activate: `a = GELU(h)`

```
a = [0.266, 0.045, 0.068, 0.371, -0.028, 0.016, 0.383, 0.025]
```

Notice `h[5] = -0.058` (negative) becomes `a[5] = -0.028` — **not exactly zero**. Contrast with ReLU, which would force this to be exactly `0.000` and kill the gradient there. This single number is the clearest possible illustration of "GELU preserves a small signal that ReLU would destroy."

### Step 3 — Compress: `y = a · W2 + b2`

```
FFN(cat) = [0.291, 0.070, 0.394, -0.006]
```

### Before/after comparison

```
Input to FFN:   [0.451, 0.170, 0.160,  0.300]
Output of FFN:  [0.291, 0.070, 0.394, -0.006]
```

Same shape, but the *content* has been reorganized — information that lived mostly in dimension 1 and 4 of the input has been redistributed differently across the output dimensions. This reorganization is what "the FFN reasons about the gathered context" means concretely: it's a learned nonlinear remapping of the vector, not a copy.

---

## 6. Per-Token, Not Cross-Token — Why This Distinction Matters

```
Attention:   token_2 ←→ token_1, token_3     (mixes across positions)
FFN:         token_2 only                    (never sees token_1 or token_3)
```

**What if the FFN mixed tokens too?** Then you'd be duplicating attention's job with a much less controlled mechanism (no learned attention weights, just a fixed dense matrix applied identically everywhere) — and you'd lose the clean separation of concerns:

- Attention = **routing** (decide what information travels where)
- FFN = **computation** (decide what to do with the information once it's arrived)

This clean separation is part of why Transformers are so trainable and interpretable relative to earlier architectures.

### 6.1 The FFN as "key-value memory"

Because the same W1/W2 weights are applied identically to every token position, and because they're the largest parameter block in the model, research (e.g., work by Geva et al. on Transformer FFN layers as key-value memories) has shown the FFN behaves like an associative memory:

- Rows of W1 act like **keys** that detect certain input patterns (e.g., "this token relates to European capitals").
- Rows of W2 act like **values** that get activated and added to the output when the corresponding key fires (e.g., injecting information related to "Paris").

**Interview framing:** *"If attention decides which tokens are relevant, the FFN is closer to where the model's factual/world knowledge is actually stored — it behaves like a big lookup table implemented as two matrix multiplications with a nonlinearity in between."*

---

## 7. What If We Remove Each Piece? (Consolidated "what if we don't" table)

| Remove this | What breaks | Why |
|---|---|---|
| **FFN entirely** | Model collapses toward a single linear transformation across all 12 layers | Attention alone = weighted averaging = linear; stacking linear layers stays linear |
| **The 4× expansion** (use 1× instead) | Reduced capacity to disentangle features; measurable performance drop | Less room to separate entangled features before recompressing |
| **GELU → ReLU** | More dead neurons, noisier/less stable training, slightly worse final performance in most benchmarks | Hard zero-gradient region for x<0 blocks learning signal |
| **The second linear layer (W2)**, i.e. leave output at 3072-d | Breaks the residual connection (shape mismatch: 3072 ≠ 768) | Residual stream requires input/output same shape |
| **Bias terms (b1, b2)** | Small, usually minor performance loss; model gets less flexibility per neuron | Bias lets each neuron shift its activation threshold independently of the input |

---

## 8. Interview Q&A Bank

**Q1: What problem does the FFN solve that attention alone cannot?**
A: Attention is a weighted average of Value vectors, which is a linear operation. Stacking linear operations (across 12 layers) still collapses into one linear transformation. The FFN introduces non-linearity (via GELU) after every attention step, which is what makes stacking multiple layers meaningfully more expressive than one layer.

**Q2: Why does the FFN expand to 4× the hidden size instead of keeping the same dimension throughout?**
A: The higher-dimensional space (3072 vs 768 in BERT-base) gives the network more independent directions to separate entangled features, apply nonlinear transformations to them, and then recombine them into a better-organized 768-d representation when compressing back down. 4× is an empirical choice from the original Transformer paper, not a theoretically derived optimum.

**Q3: Why GELU instead of ReLU?**
A: ReLU has a hard cutoff at zero — any negative input produces exactly zero output *and* exactly zero gradient, so the network gets no learning signal for that neuron on that example (the "dying ReLU" problem). GELU is a smooth, probabilistic gate (`x·Φ(x)`) that lets a small amount of signal and gradient through even for negative inputs, giving the model more chances to keep learning.

**Q4: Is the FFN applied across the whole sequence at once, or per token?**
A: Per token, independently. The exact same W1, b1, W2, b2 are applied to every token position separately — there is no mixing of information across positions in the FFN. Cross-token mixing is exclusively attention's job.

**Q5: Where do most of a Transformer's parameters live — attention or FFN?**
A: The FFN. In BERT-base, the FFN accounts for roughly 56M of the ~110M total parameters — over half — because W1 and W2 are each 768×3072 matrices (~2.36M params each), while attention's Q/K/V/output projections are comparatively smaller.

**Q6: What is the FFN sometimes called, and why?**
A: A "key-value memory." Because the FFN weights are applied identically everywhere and store the bulk of learned parameters, research has shown that rows of W1 behave like pattern-detecting "keys" and rows of W2 behave like "values" that get added to the output when their corresponding key fires — functioning like a large associative lookup table for factual/world knowledge learned during pretraining.

**Q7: What would happen if you removed the FFN layers entirely and only kept attention?**
A: Empirically, performance degrades severely — more than removing an equivalent number of attention heads in ablation studies. Theoretically, because attention is linear, a stack of attention-only layers behaves close to a single linear layer, drastically limiting what functions the model can represent.

**Q8: Why must the FFN's output dimension match its input dimension?**
A: Because the FFN's output is added to its input via a residual connection (covered in Chapter 7) before being passed to the next block. You cannot elementwise-add two vectors of different shapes, so output shape must equal input shape (768 in BERT-base).

**Q9: If GELU is smoother and "better" than ReLU, why did earlier networks use ReLU at all?**
A: ReLU is computationally cheaper (a simple max operation vs. GELU's CDF/tanh-based approximation) and was sufficient for many earlier, shallower architectures. As Transformers scaled to many more layers, the dying-neuron problem became more costly, and the extra compute cost of GELU was judged worth the improved gradient flow and slightly better empirical performance.

**Q10: Are 4× expansion and GELU still used in modern LLMs (GPT-4, LLaMA, etc.)?**
A: The general expand-activate-compress pattern is universal, but details vary. Many modern models (e.g., LLaMA family) use gated variants like SwiGLU instead of plain GELU, and sometimes use a smaller expansion ratio (e.g., ~2.7×) because the gating mechanism adds capacity in a different way. The core intuition — non-linear per-token transformation with a temporary higher-dimensional space — carries through even where exact hyperparameters differ.

---

## 9. Chapter 6 Summary (boosted)

```
FFN(x) = GELU(x·W1 + b1)·W2 + b2

Step 1 — Expand:    x [768] → h [3072]        via W1, b1
Step 2 — Activate:  h [3072] → a [3072]       via GELU (smooth, non-zero for negatives)
Step 3 — Compress:  a [3072] → y [768]        via W2, b2
```

| Design choice | Why | What breaks if removed |
|---|---|---|
| 4× expansion | High-dimensional space to disentangle features | Less capacity; measurable perf drop |
| GELU not ReLU | Smooth gradient flow, avoids dying neurons | Dead neurons, noisier training |
| Per-token (not cross-token) | Clean separation: attention routes, FFN computes | Duplicates/muddies attention's role |
| Present after every attention block | Prevents 12 linear layers from collapsing into 1 | Model capacity collapses to ~linear |
| ~56M params (>50% of BERT) | Where most factual/world knowledge is stored | Model loses most of its "memory" |

**One sentence to remember everything:** *The FFN takes each token's context-enriched vector, temporarily blows it up into a bigger space where entangled features can be separated and nonlinearly reshaped, then compresses it back down — and this per-token nonlinear "thinking" step is what prevents a deep Transformer from mathematically degenerating into a single linear layer.*

---

*Next: Chapter 7 — Residual Connections & Layer Normalization: the two components that make training a 12-layer-deep stack of these blocks actually feasible.*
