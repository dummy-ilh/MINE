# BERT Architecture — Component by Component, With Numbers

Third companion doc. The first two explained *what* each piece is and *why* it exists conceptually. This one runs an actual mini forward pass through a toy encoder block — real numbers, computed and verified with code — so you can see exactly what each component *does* to the numbers, not just why it's there in theory.

**Toy setup:** 3 tokens ("I", "like", "NLP"), embedding dim = 4 (real BERT: 768), 2 attention heads of dim 2 each (real BERT: 12 heads of dim 64), feed-forward hidden dim = 8 (real BERT: 3072). Small enough to read every number, structurally identical to the real thing.

Input embeddings (3 tokens × 4 dims):

```
X = [[1, 0, 1, 0],    # "I"
     [0, 2, 0, 2],    # "like"
     [1, 1, 1, 1]]    # "NLP"
```

---

## 1. Q, K, V projections — why you need three separate matrices, not one

**What it does:** Three learned weight matrices (`Wq, Wk, Wv`, each 4×4 here / 768×768 in real BERT) project the same input `X` into three different vectors per token.

**Why three, not one shared projection:** If Q and K were the same vector, every token's attention score with *itself* would automatically be the maximum possible (dot product of a vector with itself is its squared norm — always the largest possible score against that vector). Every token would over-attend to itself by construction, which defeats the point of gathering information from *other* tokens. Splitting into separate Q ("what am I looking for") and K ("what do I offer, for others to find") breaks that symmetry — a token can now offer something under K without necessarily seeking that same thing under Q. V is separate again because "what I use to search/be found" (Q/K) shouldn't have to be the same thing as "what information I actually hand over" (V) — you want the freedom to search using one representation and retrieve another.

Computed values (X @ Wq, X @ Wk, X @ Wv):

```
Q = [[1.0, 0.5, 0.0, 0.5],
     [0.0, 1.0, 2.0, 1.0],
     [1.0, 1.0, 1.0, 1.0]]

K = [[0.5, 0.5, 0.0, 1.0],
     [1.0, 1.0, 2.0, 0.0],
     [1.0, 1.0, 1.0, 1.0]]

V = [[1.0, 0.5, 0.5, 0.0],
     [0.0, 1.0, 1.0, 2.0],
     [1.0, 1.0, 1.0, 1.0]]
```

---

## 2. Splitting into heads — the multi-head mechanism, numerically

**What it does:** The 4-dim Q/K/V vectors are split into 2 chunks of 2 dims each — one chunk per head. No extra computation here, just a reshape: head 1 uses columns [0,1], head 2 uses columns [2,3].

```
Head 1: Q1 = [[1.0, 0.5], [0.0, 1.0], [1.0, 1.0]]
Head 2: Q2 = [[0.0, 0.5], [2.0, 1.0], [1.0, 1.0]]
```

**Why this matters (the point from the previous message, now visible in numbers):** Head 1 and Head 2 are about to compute completely independent attention patterns from these different slices — see step 3. That's the mechanism, concretely: "multiple heads" literally means "run attention separately on different learned slices of the vector, then recombine."

---

## 3. Scaled dot-product attention — per head, worked in full

**What it does, per head:** `scores = (Q @ Kᵀ) / sqrt(d_head)` → softmax across each row → `output = softmax(scores) @ V`.

**Why divide by `sqrt(d_head)` (here `sqrt(2) ≈ 1.41`):** Dot products grow in magnitude as dimensionality increases (more terms summed). Without scaling, scores can get large, pushing softmax into a saturated regime where gradients vanish almost everywhere except the single largest score — the model would learn to attend almost entirely to one token and get poor gradient signal for everything else. Dividing by `sqrt(d_head)` keeps the variance of the scores roughly constant regardless of dimension, keeping softmax in a well-behaved range.

**Head 1** — scaled scores (row = query token, col = key token):

```
        "I"     "like"  "NLP"
"I"    [0.530,  1.061,  1.061]
"like" [0.354,  0.707,  0.707]
"NLP"  [0.707,  1.414,  1.414]
```

After softmax (each row now sums to 1 — this is the actual attention distribution):

```
        "I"     "like"  "NLP"
"I"    [0.227,  0.386,  0.386]
"like" [0.260,  0.370,  0.370]
"NLP"  [0.198,  0.401,  0.401]
```

Reading this: for query token "I", head 1 puts 38.6% of its attention on "like", 38.6% on "NLP", and only 22.7% on itself — this head is pulling in outside context, not self-focused.

Head 1 output (`attn @ V1`):

```
[[0.614, 0.886],
 [0.630, 0.870],
 [0.599, 0.901]]
```

**Head 2** — scaled scores and softmax come out *differently* from head 1, because Q2/K2 are a different slice of the projection:

```
Softmax weights, head 2:
        "I"     "like"  "NLP"
"I"    [0.370,  0.260,  0.370]
"like" [0.074,  0.620,  0.306]
"NLP"  [0.198,  0.401,  0.401]
```

Note query "like" in head 2 puts 62% of its attention on *itself* — a completely different pattern from head 1's diffuse attention. **This is multi-head specialization made concrete**: same input, same query token, two heads producing two different attention distributions, because each is looking at a different learned subspace of the representation.

---

## 4. Concatenation + output projection

**What it does:** Stack the two heads' outputs back into one 4-dim vector per token, then pass through one more learned matrix `Wo` (here identity, for readability) to let the model mix information across heads before it re-enters the residual stream.

```
Concatenated (head1 | head2):
[[0.614, 0.886, 0.815, 0.890],
 [0.630, 0.870, 0.963, 1.546],
 [0.599, 0.901, 0.901, 1.203]]
```

**Why `Wo` is needed, not just concatenation:** Without it, each head's output dimensions stay siloed in fixed positions of the final vector forever — nothing lets information from head 1 combine with information from head 2. `Wo` is a learned linear mix across all heads' outputs, giving the model one more chance to combine the specialized views into something more useful before it's added back to the residual stream.

---

## 5. Residual connection — numerically, why it's not optional

**What it does:** `attn_output + X` (element-wise add, same shape).

```
Residual result:
[[1.614, 0.886, 1.815, 0.890],
 [0.630, 2.870, 0.963, 3.546],
 [1.599, 1.901, 1.901, 2.203]]
```

**Concretely what this buys you:** notice the output is *still recognizably related to* the original input `X = [[1,0,1,0],[0,2,0,2],[1,1,1,1]]` — the attention sublayer only had to learn a *correction* on top of the input, not reconstruct the input's information from scratch. If you removed the `+ X` term, the sublayer would need to perfectly re-derive and re-encode everything already present in `X` — much harder to learn, and if attention weights start near-random (as at initialization), you'd be adding pure noise on top of nothing rather than pure noise on top of a stable base signal.

---

## 6. LayerNorm — before and after, numerically

**What it does:** Per token (per row), subtract the mean and divide by the standard deviation across the 4 features.

Before norm — the three tokens have wildly different scales:

```
Token means:     [1.301,  2.002,  1.901]
Token variances: [0.176,  1.525,  0.046]
```

Token "like" has variance 1.525, token "NLP" has variance 0.046 — a >30x difference in scale between tokens at this point. If this fed directly into the next layer, that layer would see wildly inconsistent input scales depending on which token it's processing.

After LayerNorm:

```
[[ 0.745, -0.990,  1.226, -0.981],
 [-1.111,  0.703, -0.842,  1.250],
 [-1.414,  0.000,  0.000,  1.414]]

Token means (now ~0):     [-0.0, 0.0, 0.0]
Token variances (now ~1): [1.0, 1.0, 1.0]
```

**What this concretely buys you:** every token, regardless of what happened to it in attention, now enters the next sublayer on the same scale. The FFN's weights only ever have to deal with inputs in a consistent range — this is what "stabilizes training" means in numbers, not just in theory.

---

## 7. Feed-forward sublayer — expand, GELU, project back

**What it does:** `4 → 8 → 4`, with GELU in between. Real BERT: `768 → 3072 → 768`.

Token "I" through the first linear layer (pre-activation):

```
[0.038, -0.109, 0.287, -0.151, 0.592, 0.519, 1.067, -0.731]
```

After GELU:

```
[0.020, -0.050, 0.176, -0.067, 0.428, 0.362, 0.914, -0.170]
```

Notice GELU doesn't hard-zero the negative values (compare `-0.109 → -0.050`) the way ReLU would (`-0.109 → 0`) — small negative signal survives, just shrunk, which is the smooth-gradient property from the previous doc made visible.

Final FFN output for all 3 tokens (back down to dim 4):

```
[[-0.170, -0.189,  0.329,  0.333],
 [-0.033,  0.031, -0.276, -0.211],
 [ 0.026,  0.068, -0.162, -0.132]]
```

This then goes through **another** residual-add + LayerNorm (same mechanism as steps 5-6) to produce the final output of this one encoder block — which then becomes the input `X` to the next block, and the whole thing repeats 12 (or 24) times.

---

## 8. Parameter count — toy vs. real BERT-base, same ratio

Toy block (this walkthrough):

| Component | Params |
|---|---|
| Attention (Wq,Wk,Wv,Wo, 4×4 each) | 64 |
| FFN (4→8→4) | 76 |
| **FFN / Attention ratio** | **1.19x** |

Real BERT-base (d_model=768, d_ff=3072), per layer:

| Component | Params |
|---|---|
| Attention (Wq,Wk,Wv,Wo, 768×768 each) | 2,359,296 |
| FFN (768→3072→768) | 4,718,592 |
| **FFN / Attention ratio** | **2.0x** |

The toy version's ratio is close but not identical (small-scale rounding from the toy's arbitrary weights) — the structural point holds either way: **the FFN sublayer consistently has more parameters than attention**, because attention's parameter count only scales with `d_model²` while the FFN scales with `d_model × d_ff` where `d_ff` is 4x larger.

---

## 9. The quadratic attention cost — numerically, why 512 tokens is where it starts to hurt

**What it is:** Computing `Q @ Kᵀ` costs roughly `n² × d` multiply-adds, where `n` = sequence length, `d` = model dimension. This is per layer, per head-group (heads split `d` but the total work across all heads is still `n² × d`).

At real BERT scale (`d = 768`):

| Sequence length | QK^T cost (n² × d) | Growth vs. previous row |
|---|---|---|
| 128 | 12,582,912 | — |
| 256 | 50,331,648 | 4x |
| 512 (BERT's max) | 201,326,592 | 4x |
| 1024 | 805,306,368 | 4x |
| 2048 | 3,221,225,472 | 4x |

**Read this table literally:** every time you double the sequence length, the attention cost quadruples — not doubles. Going from 512 → 1024 tokens (2x the text) costs 4x the compute for the attention step alone, and 2048 tokens costs *16x* what 512 tokens costs. This is the numeric face of "O(n²) doesn't scale" — it's not a vague concern, it's a hard quadratic wall, which is exactly why BERT was capped at 512 and why later long-context architectures had to specifically redesign the attention computation (sparse patterns, linear attention approximations, etc.) rather than just "training BERT on longer sequences."

---

## Takeaways to say out loud in an interview

- Q/K/V are separate specifically so a token's "self-similarity" doesn't dominate attention by construction.
- Multi-head isn't a hyperparameter tweak — running the walkthrough shows two heads producing genuinely different attention distributions over the *same* tokens from the *same* input.
- The `1/sqrt(d_head)` scaling exists to keep softmax out of a saturated, low-gradient regime.
- Residuals mean each sublayer only learns a *correction*, not a full reconstruction — visible in how the residual output stays close to the original input.
- LayerNorm's job, concretely, is making sure every token's activation scale is comparable before the next sublayer sees it — the before/after variance numbers show exactly why that matters.
- FFN has roughly 2x the parameters of attention in real BERT — attention gets the spotlight, FFN does more of the heavy lifting.
- Attention cost is quadratic in sequence length — a fact you can point to a number for (4x cost per doubling), not just cite as "it's expensive."
