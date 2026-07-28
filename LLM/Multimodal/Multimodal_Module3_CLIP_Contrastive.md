# Multimodal Module 3 — Contrastive Learning & CLIP (Master Notes, Expanded)

## 0. What problem CLIP actually solves

Module 1 named the core problem: independently-trained image and text embeddings live in unrelated vector spaces. Module 2 gave you a vision encoder that turns an image into a sequence of vectors — but says nothing about whether those vectors relate meaningfully to text. **CLIP (Contrastive Language-Image Pretraining)** is the training recipe that explicitly forces image and text representations into a **shared, comparable embedding space**, by training on large-scale image-text pairs with a specific objective designed exactly for this purpose.

---

## 1. The CLIP architecture, at a glance

Two separate encoders, trained jointly:
- **Image encoder**: a ViT (Module 2) or CNN, mapping an image to a single vector (e.g., the [CLS] token output, or a pooled representation).
- **Text encoder**: a standard transformer text encoder (architecturally similar to what you know from LLM Basics — commonly a smaller transformer, not necessarily a full LLM), mapping a text caption to a single vector.

Each encoder's output is passed through a final **linear projection layer** mapping it into a shared embedding dimension (so image and text vectors end up the same size, a prerequisite for comparing them directly via dot product/cosine similarity) — this projection layer is itself learned, part of what training adjusts.

**Critical framing**: CLIP does **not** use cross-attention between the two modalities during encoding — the image and text are encoded **completely independently** of each other, and only *compared* after both are fully encoded into their final vectors. This "two independent towers, compared only at the end" structure (sometimes called a **dual-encoder** or **two-tower** architecture) is a deliberate design choice, with real consequences covered in Section 4.

---

## 2. The contrastive objective — InfoNCE loss, full derivation

### The setup
Take a training batch of `N` image-text pairs (image₁-caption₁, image₂-caption₂, ..., imageₙ-captionₙ) — critically, these `N` pairs are **genuinely matching, correct pairs** as scraped/curated from the training data. Encode all `N` images and all `N` captions through their respective encoders, producing `N` image embeddings and `N` text embeddings.

### The key trick — using the rest of the batch as negatives, for free
For any given image `i`, its **true matching caption** is caption `i` — but every *other* caption in the batch (caption `j` for `j≠i`) serves as a **negative example** (a non-matching, "wrong" caption) for image `i`, entirely for free, with no separate negative-sampling step needed — this is the practical trick that makes contrastive training scale efficiently: a batch of size `N` gives you `N` positive pairs and `N×(N-1)` negative pairs, all constructed automatically just from batch composition.

### The InfoNCE loss formula, explained term by term
For image `i`, compute the similarity (typically cosine similarity, i.e., normalized dot product) between its embedding and **every** text embedding in the batch, then apply a softmax over those similarities (scaled by a temperature `τ`) to get a probability distribution over "which caption in this batch matches this image":
```
P(caption_i matches image_i) = exp(sim(img_i, txt_i)/τ) / Σ_j exp(sim(img_i, txt_j)/τ)
```
The loss for image `i` is the **negative log** of the probability assigned to the true matching caption:
```
L_i = -log[ exp(sim(img_i, txt_i)/τ) / Σ_j exp(sim(img_i, txt_j)/τ) ]
```
This is **exactly the same cross-entropy loss structure** from LLM Basics Module 2's CLM formula — the "classes" here are just "which caption in the batch is the right match" instead of "which vocabulary token comes next." The full CLIP loss symmetrizes this in **both directions** (image-to-text matching, as above, AND text-to-image matching, doing the same softmax-over-similarities computation the other way, treating each caption as the anchor and images as the candidates) and averages the two — this symmetric double direction is a specific, worth-remembering detail of CLIP's exact loss (not just a single-direction contrastive loss).

### Role of the temperature `τ`
Exactly the same mechanism as LLM Basics Module 6's decoding temperature, but applied to *training* rather than sampling: dividing similarities by a small `τ` **sharpens** the softmax distribution (makes the model's implicit "confidence" more peaked, pushing harder to separate the true match from negatives), while a larger `τ` **flattens** it. In CLIP, `τ` is typically a **learned parameter** (not a fixed hyperparameter) — the model learns the optimal sharpness for its own embedding scale during training, rather than requiring manual tuning.

### Full numerical worked example
Take a tiny batch of N=3 image-text pairs. Suppose the raw cosine similarities (already computed) between image 1 and all 3 captions are:
```
sim(img_1, txt_1) = 0.8   (the true match)
sim(img_1, txt_2) = 0.3
sim(img_1, txt_3) = 0.1
```
With temperature τ=0.1 (a realistic small value, sharpening the distribution — CLIP's learned τ often ends up small):
```
Scaled: [0.8/0.1, 0.3/0.1, 0.1/0.1] = [8.0, 3.0, 1.0]
exp: [2980.96, 20.09, 2.72]
sum = 3003.77
P(txt_1) = 2980.96/3003.77 ≈ 0.9924
P(txt_2) = 20.09/3003.77 ≈ 0.0067
P(txt_3) = 2.72/3003.77 ≈ 0.0009
```
Loss for image 1 = `-log(0.9924) ≈ 0.0076` — very low loss, since the model already assigns very high probability (99.2%) to the correct caption, and the low temperature has sharply amplified an already-decent similarity gap (0.8 vs 0.3) into an even more confident probability separation. Compare this to what would happen **without** temperature scaling (τ=1, no sharpening): `exp([0.8,0.3,0.1])=[2.226,1.350,1.105]`, sum=4.681, `P(txt_1)=2.226/4.681≈0.476` — under 50% probability on the correct match despite it having the clearly highest raw similarity — this concretely shows why the learned temperature matters: it lets the model calibrate how aggressively small similarity differences should translate into confident predictions, rather than being stuck with whatever raw cosine-similarity scale the embeddings happen to have.

---

## 3. Zero-Shot Classification via CLIP — the practical payoff

### The mechanism
Once CLIP is trained, you can perform image classification on an **entirely new set of classes never seen during training**, with no additional fine-tuning, using a clever trick: construct a text caption for each candidate class (e.g., "a photo of a {class}" filled in with "dog," "cat," "airplane," etc. — this templating is itself a real, documented detail worth knowing, sometimes called **prompt engineering for CLIP**, and different phrasings measurably affect zero-shot accuracy), encode all these candidate-class captions with the text encoder, encode the actual query image with the image encoder, and then **pick the class whose caption embedding has the highest similarity to the image embedding** — the "classification head" is entirely implicit, just a similarity comparison against a set of class-description text embeddings, no classification-specific weights ever trained.

### Why this is a genuinely notable capability, not just a cute trick
Traditional image classifiers require a fixed, predetermined set of output classes baked into the model's final layer at training time (Module 2's CNN/ViT encoders, used alone for classification, would need a specific classification head trained on specific labeled classes). CLIP's zero-shot approach means **the "classes" are just arbitrary text you supply at inference time** — you can add, remove, or completely redefine the classification task without retraining anything, purely by changing the text prompts you compare the image against. This directly reuses In-Context-Learning-style flexibility (LLM Basics Module 4) but for classification rather than generation — the frozen model adapts its "behavior" purely through what's provided at inference time, no weight updates needed.

### Numerical framing of why more/better templates help
CLIP's original paper found that averaging embeddings across **multiple prompt templates** per class (e.g., "a photo of a {class}", "a blurry photo of a {class}", "an image containing a {class}" — many templates, embeddings averaged) improved zero-shot accuracy over using a single fixed template — a form of ensembling over prompt phrasing variance, directly analogous to LLM Basics Module 3's discussion of prompt/format sensitivity affecting benchmark scores; CLIP's zero-shot accuracy is measurably sensitive to exactly how you phrase the class-description text, which is worth naming explicitly as a real, practical gotcha.

---

## 4. Why the "dual-encoder, no cross-attention" design is a deliberate tradeoff

### The advantage — retrieval-scale efficiency
Because image and text are encoded **completely independently**, you can **precompute and cache embeddings for a huge database of images (or text) once**, then at query time only need to encode the *new* query (a single image or single piece of text) and compare it via fast vector similarity search (e.g., nearest-neighbor lookup, LLM Basics Module 6's embedding-similarity retrieval mechanism reused here) against the precomputed database — this is exactly what makes CLIP-style models practical for large-scale image search/retrieval systems: you never need to re-run a joint image+text forward pass for every possible pairing at query time.

### The disadvantage — no fine-grained cross-modal reasoning at encoding time
Because the two modalities never attend to each other during encoding, CLIP's embeddings capture **holistic, whole-image/whole-caption alignment**, not fine-grained interaction between specific image regions and specific words (e.g., CLIP is good at "does this image roughly depict a red car on a street," but structurally weaker at fine-grained compositional questions like "is the red object to the left or right of the blue object," since that kind of question benefits from genuine cross-attention between specific visual regions and specific text tokens — exactly the capability the dual-encoder architecture deliberately forgoes for retrieval efficiency). This is precisely the motivation for Module 4's cross-attention fusion architectures, which sacrifice the retrieval-efficiency advantage in exchange for genuine fine-grained cross-modal interaction — a direct architectural tradeoff worth stating explicitly if asked "why not just always use cross-attention instead of CLIP's dual-encoder approach."

---

## 5. Side-by-side summary table (memorize this cold)

| | CLIP (dual-encoder, contrastive) |
|---|---|
| How image/text interact during encoding | Not at all — fully independent encoders, compared only after full encoding |
| Training objective | Symmetric InfoNCE contrastive loss over batch-constructed positive/negative pairs |
| Negative examples | Free — every other pair in the same training batch |
| Key hyperparameter | Temperature τ (learned, controls softmax sharpness) |
| Headline practical capability | Zero-shot classification via text-prompt-as-class-label similarity comparison |
| Main structural limitation | No fine-grained cross-modal (region-to-word) reasoning — only holistic whole-image/whole-caption alignment |
| Main structural advantage | Precomputable, cacheable embeddings — efficient for large-scale retrieval |

---

## 6. Quick-fire Q&A (self-test)

**Q: Where do CLIP's negative examples for the contrastive loss come from, and why is this an efficient design?**
A: Every other image-text pair in the same training batch serves as a negative example for any given pair — a batch of size N gives N positive pairs and N×(N-1) negative pairs automatically from batch composition, with no separate negative-sampling step needed.

**Q: Write the InfoNCE loss formula for one image and explain the role of the temperature term.**
A: `L_i = -log[exp(sim(img_i,txt_i)/τ) / Σ_j exp(sim(img_i,txt_j)/τ)]` — a softmax over similarities to every caption in the batch, with negative log-probability assigned to the true match as the loss (structurally identical to standard cross-entropy). Temperature τ controls how sharply similarities are converted into a peaked probability distribution — smaller τ sharpens/amplifies similarity differences into more confident predictions; in CLIP, τ is typically a learned parameter rather than a fixed hyperparameter.

**Q: Why is CLIP's loss described as "symmetric," and what does that mean concretely?**
A: The full loss computes the contrastive softmax in both directions — image-to-text (treating each image as the anchor, captions as candidates) and text-to-image (treating each caption as the anchor, images as candidates) — and averages both, rather than only optimizing one direction of the matching task.

**Q: Explain how CLIP performs zero-shot classification on classes never seen during training, with no fine-tuning.**
A: Construct a text caption for each candidate class (e.g., "a photo of a {class}"), encode all candidate-class captions and the query image, and pick the class whose text embedding has the highest similarity to the image embedding — the classification head is implicit, just a similarity comparison, so arbitrary new classes can be introduced purely by supplying new text prompts at inference time, no retraining needed.

**Q: What's the core structural tradeoff of CLIP's dual-encoder (no cross-attention) design, stated as an explicit advantage/disadvantage pair?**
A: Advantage: image and text embeddings can be precomputed and cached independently, enabling efficient large-scale retrieval via fast similarity search without re-running a joint forward pass per query. Disadvantage: because the modalities never attend to each other during encoding, CLIP captures holistic whole-image/whole-caption alignment but is structurally weaker at fine-grained compositional cross-modal reasoning (e.g., spatial relationships between specific objects), which is exactly what motivates cross-attention fusion architectures instead.

**Q: Why does prompt template phrasing measurably affect CLIP's zero-shot accuracy, and what's a known mitigation?**
A: Because the "class label" at inference time is just an arbitrary text string compared via embedding similarity, and different phrasings of the same underlying class concept produce different text embeddings with different similarity to the true image content — a known mitigation is averaging embeddings across multiple diverse prompt templates per class, a form of ensembling over prompt-phrasing variance that CLIP's original paper found improves zero-shot accuracy over a single fixed template.

---
*End of Multimodal Module 3 (expanded). Next: Module 4 — Cross-Modal Fusion Architectures (early/late fusion, cross-attention, Q-Former, Perceiver Resampler).*
