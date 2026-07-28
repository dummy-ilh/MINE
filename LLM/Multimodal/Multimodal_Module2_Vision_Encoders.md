# Multimodal Module 2 — Vision Encoders (Master Notes, Expanded)

## 0. The task this module solves

Module 1 established the problem: images are continuous and spatially-structured, not sequential/discrete like text. A **vision encoder** is the component that converts a raw image into a sequence of vectors a transformer can process — this module covers how that conversion actually works, both the ViT approach (which reuses your existing transformer knowledge almost directly) and the CNN alternative it largely displaced in modern VLMs.

---

## 1. Vision Transformer (ViT) — applying the transformer you already know to images

### The core idea, in plain words
Instead of feeding a transformer individual pixels (which would make sequences enormous — a modest 224×224 image has 50,176 pixels), **chop the image into fixed-size square patches** (commonly 16×16 pixels), **flatten and linearly project each patch into an embedding vector** — and now you have a sequence of patch embeddings, exactly analogous to a sequence of token embeddings in text, which the standard transformer encoder architecture (self-attention, feedforward layers — everything you already know) can process directly.

### The patchification math, worked numerically
Take a 224×224×3 (RGB) image, patch size 16×16:
```
Number of patches = (224/16) × (224/16) = 14 × 14 = 196 patches
```
Each patch is 16×16×3 = 768 raw values (flattened). A **learned linear projection** (a single dense layer, weight matrix shape `768 × d_model`) maps each flattened patch to the model's hidden dimension `d_model` (e.g., 768 for ViT-Base, coincidentally the same number here but not fundamentally related — just how ViT-Base was configured):
```
Patch embedding = Linear(flatten(patch))  →  a 196-token sequence of d_model-dimensional vectors
```
This is directly, mechanically analogous to LLM Basics Module 1's tokenization + embedding lookup — the difference is text tokenization uses a *discrete lookup table* (a fixed vocabulary), while ViT's "tokenization" is a *continuous linear projection* of raw patch pixel values (there's no fixed "vocabulary of patches" — every possible 16×16 pixel patch maps through the same learned linear function).

### Position encoding for 2D images
Text's positional encoding (RoPE, ALiBi, or learned/sinusoidal, LLM Basics Module 6) encodes a 1D sequence position. Images are naturally 2D, so ViT needs position information that captures **row and column**, not just a single linear index. The original ViT paper found that simply **learning a separate 1D positional embedding per patch position (treating the 14×14 grid as a flattened 196-length sequence in raster-scan order — left to right, top to bottom)** works well in practice, despite discarding explicit 2D structure — the position embeddings are simply learned parameters, one per patch slot, added to each patch's content embedding, and the model empirically learns to recover the useful 2D spatial relationships from this flattened 1D positional signal during training. Some later variants use explicit 2D-aware positional schemes (separate learned/sinusoidal embeddings for row and column, added together), but the simpler flattened 1D-learned approach remains a strong, common baseline.

### The [CLS] token
Directly borrowed from BERT (LLM Basics Module 2) — a special learnable embedding is **prepended** to the patch-embedding sequence before the first transformer layer. After the full sequence passes through all the transformer encoder layers (standard bidirectional self-attention — critically, **ViT's attention is not causally masked**, since there's no autoregressive generation happening here; every patch can attend to every other patch, both directions, from the very first layer, exactly like BERT's MLM encoder), the [CLS] token's final-layer output vector is treated as an **aggregate representation of the whole image**, used for whole-image tasks like classification — the same role it played for whole-sequence classification in BERT.

### Numerical parameter comparison (why patch size is a real design tradeoff)
Smaller patches → more patches → longer sequence → more attention compute (O(n²) per LLM Basics Module 1's exact reasoning, now applied to patch-sequence length rather than text-token length) but finer-grained spatial detail captured per patch. Larger patches → shorter sequence, cheaper compute, but coarser detail (a 32×32 patch blurs together more visual information into one token than a 16×16 patch does).
```
16×16 patches on a 224×224 image: 196 patches
32×32 patches on the same image: 49 patches (4x fewer, roughly 16x less attention compute given O(n²))
```
This is the exact same sequence-length-vs-granularity tradeoff logic from LLM Basics Module 1's tokenization tradeoffs (bigger "vocabulary unit" = shorter sequence = less compute but coarser signal) — worth naming this parallel explicitly if asked why patch size is a meaningful design choice, not an arbitrary hyperparameter.

---

## 2. CNN-based Encoders (ResNet-style) — the pre-ViT default, still relevant to know

### The core idea
Convolutional layers apply small, spatially-local learned filters that slide across the image, building up a hierarchy of features — early layers detect simple local patterns (edges, colors), progressively deeper layers combine these into increasingly complex, larger-receptive-field features (textures, then object parts, then whole objects) — this hierarchical, spatially-local feature-building is architecturally very different from ViT's approach, where every patch can directly attend to every other patch from the very first layer, with no inherent spatial locality bias built into the architecture itself.

### The key architectural bias difference — a favorite interview comparison point
CNNs have a strong **built-in inductive bias for spatial locality and translation invariance** (a filter that detects an edge works the same way regardless of where in the image that edge appears, since the same filter weights slide across the whole image) — this bias is architecturally baked in, not learned. ViT has **no such built-in spatial bias** — full self-attention treats all patch positions symmetrically at initialization, and the model must *learn* useful spatial relationships purely from data (aided only by the positional embeddings, which are just additive learned vectors, not an architectural constraint forcing locality).

### The practical consequence — data scale dependence
This difference in built-in inductive bias explains a well-documented empirical finding: **ViT tends to underperform comparable CNNs when trained on smaller datasets**, because the CNN's built-in spatial-locality bias acts as a helpful prior that reduces how much the model needs to learn purely from data — but **ViT tends to match or exceed CNN performance once trained on sufficiently large datasets**, because at large enough scale, the model has enough data to learn useful spatial relationships on its own, and the *lack* of a hard-coded locality constraint becomes an advantage rather than a disadvantage (the model can learn genuinely non-local, long-range spatial relationships that a CNN's local-filter structure would need many stacked layers to approximate, if it can capture them at all). This exact "less inductive bias needs more data, but scales better" pattern is conceptually the same theme as LLM Basics Module 3's scaling-laws material — a good cross-topic connection to draw if asked to compare ViT and CNN scaling behavior.

---

## 3. Side-by-side summary table (memorize this cold)

| | Vision Transformer (ViT) | CNN (ResNet-style) |
|---|---|---|
| How it processes the image | Patchify → linear projection → self-attention over all patches | Local convolutional filters, hierarchical feature building |
| Built-in inductive bias | Minimal — must learn spatial relationships from data + position embeddings | Strong — spatial locality and translation invariance are architecturally baked in |
| Data-scale behavior | Underperforms CNNs on small data; matches/exceeds at large scale | Performs relatively well even on smaller datasets, due to helpful built-in bias |
| Attention pattern | Full bidirectional self-attention, every patch attends to every other patch from layer 1 | N/A — local receptive fields, growing with depth |
| Whole-image representation | [CLS] token's final output (borrowed from BERT) | Global average pooling over final feature maps (a different, non-transformer mechanism) |
| Dominant choice in modern VLMs | Yes — CLIP, LLaVA, most current VLMs use ViT-family encoders | Largely legacy at this point for new VLM designs, though still used/studied |

---

## 4. Quick-fire Q&A (self-test)

**Q: Walk through the patchification math for a 224×224 image with 16×16 patches — how many patches, and what's the raw flattened size of each before projection?**
A: (224/16)×(224/16) = 14×14 = 196 patches; each patch is 16×16×3 (RGB) = 768 raw values before the learned linear projection maps it to the model's hidden dimension.

**Q: How does ViT handle positional information for a fundamentally 2D input, and what's a common practical approach?**
A: A common and effective approach is simply learning a separate 1D positional embedding per patch slot, treating the 2D patch grid as a flattened raster-scan sequence — despite discarding explicit 2D structure, the model empirically learns to recover useful spatial relationships from this signal during training; some variants use explicit 2D-aware (row+column) positional schemes instead.

**Q: What role does the [CLS] token play in ViT, and where is this idea borrowed from?**
A: It's a learnable embedding prepended to the patch sequence; after passing through all transformer layers, its final-layer output serves as an aggregate whole-image representation, used for tasks like classification — directly borrowed from BERT's use of a [CLS] token for whole-sequence representations.

**Q: What is the key architectural bias difference between ViT and CNN-based encoders?**
A: CNNs have spatial locality and translation invariance architecturally baked in via local sliding filters; ViT has no such built-in spatial bias — full self-attention treats all patch positions symmetrically at initialization, and any useful spatial relationships must be learned purely from data (aided only by additive positional embeddings, not a hard architectural constraint).

**Q: Explain the empirical data-scale-dependent performance difference between ViT and CNNs, and connect it to a concept from LLM Basics.**
A: ViT underperforms CNNs on smaller datasets (lacking a helpful built-in locality prior that reduces the data needed to learn useful spatial structure) but matches or exceeds CNN performance at large data scale (where it can learn spatial relationships, including long-range ones a CNN's local filters would need many layers to approximate, purely from data). This mirrors LLM Basics Module 3's scaling-laws theme: less built-in inductive bias generally requires more data, but tends to scale better once enough data is available.

**Q: Why is ViT's self-attention over patches described as having "no causal masking," and why does that make sense for this use case?**
A: Every patch can attend to every other patch in both directions from the first layer, unlike a decoder-only CLM model's triangular causal mask — this makes sense because there's no autoregressive generation happening in a vision encoder; the goal is building a rich bidirectional representation of the whole image, directly analogous to BERT's bidirectional MLM encoder rather than GPT's causal decoder.

---
*End of Multimodal Module 2 (expanded). Next: Module 3 — Contrastive Learning & CLIP (InfoNCE loss, zero-shot classification, shared embedding space construction).*
