# Multimodal Module 4 — Cross-Modal Fusion Architectures (Master Notes, Expanded)

## 0. Why this module exists — picking up where CLIP left off

Module 3 ended on CLIP's key limitation: dual-encoder architectures never let image and text attend to each other during encoding, so they capture holistic alignment but not fine-grained, compositional cross-modal reasoning. This module covers the architectural techniques designed specifically to fix that — letting text and image representations genuinely interact via attention, not just get compared after the fact.

---

## 1. The fusion taxonomy — early, late, and cross-attention fusion

### Late fusion (what CLIP does)
Each modality is encoded **completely independently** through its own full encoder stack, and the two final representations are only combined/compared at the very end (CLIP's similarity comparison is the simplest possible late-fusion case). Cheapest computationally, most limited in cross-modal interaction depth — this is the baseline Module 3 already covered in detail.

### Early fusion
Combine raw or lightly-processed representations of both modalities **before** most of the network's processing — e.g., concatenating patch embeddings and text token embeddings into a single sequence right at the input, then running the *entire* combined sequence through one shared transformer stack (self-attention over both modalities jointly, from layer 1). This gives the model maximal opportunity for cross-modal interaction (every layer can mix information across modalities), at the cost of significantly higher compute (attention is O(n²) over the *combined* sequence length, so mixing modalities early makes every subsequent layer more expensive) and a harder optimization problem (the model has to learn to handle both modalities' very different statistical properties within the same shared weights from the very first layer).

### Cross-attention fusion (the dominant middle-ground approach)
Keep **separate** encoder stacks for each modality (so each modality gets to build up its own specialized internal representations first, unlike early fusion), but insert **cross-attention layers** at specific points that let one modality's representations attend to the other's — most commonly, text tokens (as Queries) attend to image patch embeddings (as Keys and Values), pulling in relevant visual information at that specific point in processing, while each modality otherwise continues to use its own specialized self-attention layers.

---

## 2. Cross-attention mechanics, worked through step by step

### The mechanism, reusing exactly what you know from transformer self-attention
Standard self-attention computes Query, Key, and Value all from the **same** input sequence. **Cross-attention** computes the Query from **one** sequence and the Key/Value from a **different** sequence:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
Same formula you already know — the only change is **where Q comes from vs. where K and V come from**. In a typical vision-language cross-attention layer: `Q` comes from the text sequence's current hidden states (what the text is "looking for"), `K` and `V` come from the image patch embeddings (what visual information is available to look at) — the output is a new representation for each text token, now informed by whichever image patches it attended to most strongly, added (typically via a residual connection) back into the text stream's ongoing representation.

### Concrete worked walkthrough
Suppose you have a text token embedding for the word "cat" (as the Query) and 196 image patch embeddings (as Keys/Values, from Module 2's ViT patchification). The cross-attention layer computes a similarity score between the "cat" query and **all 196** patch keys (a dot product each, scaled by `√d_k` exactly as in standard self-attention), softmaxes those 196 scores into an attention-weight distribution, and produces a weighted sum of the 196 patch **values** using those weights. If the image genuinely contains a cat in a specific region, the patches covering that region should (after training) receive the highest attention weights — the "cat" token's representation gets updated to be a weighted blend dominated by the visual information from exactly the patches depicting the cat, not the whole image indiscriminately. This is the concrete mechanism that gives fine-grained, spatially-localized cross-modal grounding — exactly what Module 3 flagged as missing from CLIP's late-fusion approach.

### Where cross-attention layers typically get inserted
Rather than replacing every self-attention layer, cross-attention layers are commonly **interleaved** with regular self-attention layers within a decoder stack (e.g., self-attention layer → cross-attention layer → feedforward → repeat) — this pattern is directly inherited from the **original encoder-decoder Transformer architecture** (which you should already know from your transformer course: the decoder's cross-attention layers attend to the encoder's output, exactly the same mechanical pattern, just with "encoder output" replaced by "image patch embeddings" here).

---

## 3. Q-Former (BLIP-2) — learned query tokens as an information bottleneck

### The core idea
Rather than letting text directly cross-attend to potentially hundreds of raw image patch embeddings (expensive, and somewhat unstructured), Q-Former introduces a **small, fixed number of learnable "query" embeddings** (e.g., 32 queries, regardless of the underlying image's resolution or patch count) that are trained to **extract the most relevant visual information from the (frozen) image encoder's patch embeddings via cross-attention**, compressing potentially hundreds of patches down into a small, fixed-size set of highly-informative visual tokens — these compressed query outputs are what actually get passed on to the language model, not the raw patch embeddings themselves.

### Why this compression matters practically
A fixed, small number of output tokens (e.g., 32) regardless of image resolution/patch count directly controls the cost of feeding visual information into the downstream LLM — since LLM Basics Module 6 established that longer context means more attention compute and more KV-cache memory, capping the "cost" of an image at a fixed, small token budget (rather than letting it scale with resolution — a 196-patch ViT output would otherwise cost nearly 200 tokens of LLM context per image) is a deliberate, important efficiency design choice, not an incidental detail.

### The two-stage training approach (worth knowing at a high level)
BLIP-2's Q-Former is trained in stages designed to bootstrap useful query behavior: an initial stage trains the queries to extract vision-language-relevant information using objectives similar in spirit to CLIP's contrastive alignment (Module 3) plus additional objectives (like image-grounded text generation), before a second stage connects the now-competent Q-Former to an actual frozen LLM for the full generative vision-language task — the detail worth retaining for an interview is less the exact staged-objective specifics and more the **overall strategy**: train the compression/extraction mechanism (the queries) to be genuinely useful *before* asking the full pipeline to do end-to-end generation, rather than training everything jointly from a random initialization.

---

## 4. Perceiver Resampler (Flamingo-style) — the same compression idea, different lineage

### The core idea
Conceptually very similar to Q-Former's goal (compress a variable/large number of visual inputs into a small, fixed number of output tokens via learned queries cross-attending to the raw visual features), inherited from the general **Perceiver architecture** (originally designed as a general-purpose way to handle very large/variable-length inputs of any modality by cross-attending them into a small fixed-size latent array, rather than being vision-language-specific from the start). In the Flamingo model specifically, the Perceiver Resampler compresses visual features (potentially from **multiple frames of a video**, not just a single image — a detail worth knowing, since it highlights the technique's generality beyond single still images) into a small, fixed set of visual tokens, which are then made available to the language model via interleaved cross-attention layers (Section 2's mechanism) inserted throughout the LLM's decoder stack.

### The key distinguishing detail vs. Q-Former, worth being precise about
Both techniques solve the same core problem (compress variable-size visual input into a small fixed-size token set via learned-query cross-attention) — the differences are more about specific training recipe and integration details (how the compressed tokens are fed into the LLM — Flamingo's cross-attention layers interleaved throughout the LLM decoder vs. BLIP-2's approach of feeding Q-Former's output tokens more directly as part of the input sequence) than a fundamentally different mechanism. **The one-sentence framing to have ready**: "Q-Former and the Perceiver Resampler are both instances of the same general pattern — a small set of learned query tokens cross-attending into a large/variable visual feature set to produce a compact, fixed-size visual representation — they differ mainly in training recipe and exactly how the compressed tokens get integrated into the downstream language model."

---

## 5. Side-by-side summary table (memorize this cold)

| | Late Fusion (CLIP) | Early Fusion | Cross-Attention Fusion | Q-Former / Perceiver Resampler |
|---|---|---|---|---|
| When modalities interact | Only after full independent encoding | From the very first layer, jointly | At specific inserted layers, via cross-attention | Via a compression step, using learned queries cross-attending to raw visual features |
| Compute cost | Cheapest | Most expensive (O(n²) over combined sequence from layer 1) | Moderate | Moderate — compression step is added cost, but downstream LLM cost is reduced via fixed small token count |
| Fine-grained cross-modal reasoning | Weak (holistic only) | Strong | Strong | Strong, and compressed to a fixed small token budget |
| Visual token cost to downstream LLM | N/A (no shared downstream sequence) | High (all patches feed into one shared sequence) | Depends on how many raw patches text attends to | Fixed and small, regardless of image resolution/patch count |

---

## 6. Quick-fire Q&A (self-test)

**Q: What's the core tradeoff between early fusion and late fusion, stated precisely?**
A: Early fusion combines modalities from the very first layer via one shared transformer stack, maximizing cross-modal interaction depth but at significantly higher compute cost (O(n²) attention over the combined sequence from layer 1) and a harder joint-optimization problem. Late fusion keeps modalities fully separate until a final comparison step, which is cheap but limited to holistic, non-fine-grained cross-modal interaction.

**Q: In a typical vision-language cross-attention layer, where does the Query come from, and where do the Key and Value come from?**
A: Query typically comes from the text sequence's current hidden states; Key and Value come from the image patch embeddings — the text is "looking at" the image, pulling in relevant visual information at that point in processing, using the exact same attention formula as standard self-attention, just with Q sourced from a different sequence than K/V.

**Q: What specific architectural pattern is vision-language cross-attention insertion directly inherited from?**
A: The original encoder-decoder Transformer architecture — the decoder's cross-attention layers attending to the encoder's output is mechanically identical to text cross-attending to image patch embeddings here, just with "encoder output" replaced by visual features.

**Q: What problem does Q-Former's fixed small number of learned query tokens solve, and why does that matter for downstream LLM cost?**
A: It compresses a potentially large, resolution-dependent number of raw image patch embeddings (e.g., 196 for a single ViT pass) down into a small, fixed number of highly-informative visual tokens (e.g., 32), regardless of image resolution. This matters because feeding raw uncompressed patches into an LLM's context would scale token cost (and thus attention compute and KV-cache memory, per LLM Basics Module 6) with image resolution — a fixed small token budget makes visual input cost predictable and controlled.

**Q: What's the core similarity and the core difference between Q-Former and the Perceiver Resampler?**
A: Both use the same general pattern — a small set of learned query tokens cross-attending into a large/variable visual feature set to produce a compact, fixed-size visual representation. They differ mainly in training recipe and integration details: how the compressed visual tokens are actually fed into the downstream language model (interleaved cross-attention layers throughout the LLM decoder, Flamingo-style, vs. feeding compressed tokens more directly into the input sequence, BLIP-2-style), rather than in the fundamental compression mechanism itself.

**Q: Why is the Perceiver Resampler described as more general than a vision-language-specific technique?**
A: It's inherited from the general Perceiver architecture, originally designed to handle very large or variable-length inputs of any modality by cross-attending them into a small fixed-size latent array — in Flamingo specifically, it's applied to visual features potentially spanning multiple video frames, not just a single still image, illustrating that the underlying compression mechanism isn't inherently tied to single-image vision-language fusion.

---
*End of Multimodal Module 4 (expanded). Next: Module 5 — Vision-Language Model Architectures (LLaVA, Gemini-style native multimodal, image tokenization for LLMs).*
