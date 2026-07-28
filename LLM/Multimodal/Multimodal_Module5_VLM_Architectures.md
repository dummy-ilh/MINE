# Multimodal Module 5 — Vision-Language Model Architectures: LLaVA, Gemini-style (Master Notes, Expanded)

## 0. Where this module sits — assembling the pieces

You now have all the building blocks: a vision encoder (Module 2) that turns images into patch embeddings, contrastive pretraining (Module 3) that can align embedding spaces, and fusion mechanisms (Module 4) that let modalities interact. This module covers how real, production-style VLMs actually **assemble** these pieces into an end-to-end system that takes an image and text in, and produces text out — specifically the two dominant architectural philosophies: LLaVA's simple-adapter approach and Gemini-style native multimodal processing.

---

## 1. The LLaVA approach — simple linear projection, frozen components

### The core idea
LLaVA's architecture is deliberately minimal: take a **frozen, pretrained vision encoder** (typically CLIP's ViT, Modules 2-3 — already knows how to produce meaningful image representations from contrastive pretraining), take a **frozen (or later, fine-tuned) pretrained LLM** (already knows language, per LLM Basics Modules 2-5), and connect them with just a **single trainable linear layer** (or in later versions, a small 2-layer MLP) that projects the vision encoder's patch embedding outputs directly into the LLM's token embedding space.

### The mechanism, step by step
1. Image → frozen ViT (CLIP's vision encoder) → a sequence of patch embeddings (Module 2's output, e.g., 576 patches for a common LLaVA configuration).
2. Each patch embedding → the trainable linear projection layer → now each patch embedding has the **same dimensionality as the LLM's own token embeddings**, and — critically — the projection is trained specifically so these projected vectors are **meaningful inputs to the LLM's existing attention mechanism**, not just dimensionally compatible.
3. These projected "visual tokens" are **concatenated directly into the LLM's input sequence** alongside the actual text tokens (e.g., `[image tokens] [text: "what is in this image?"]`) — from this point on, **the LLM processes the combined sequence using its completely standard, unmodified self-attention** — there's no special cross-attention architecture; the visual tokens are just... more tokens in the sequence, attended to via the exact same mechanism the LLM already uses for text-to-text attention.

### Why this radically simple approach works at all — the key insight
This directly reuses LLM Basics Module 4's LoRA/PEFT philosophy: **you don't need to retrain everything from scratch if you can find a small, targeted intervention that bridges two already-competent frozen components.** The vision encoder already produces meaningful visual representations (from CLIP's contrastive pretraining, Module 3); the LLM already knows how to process token sequences and reason over context (from its own pretraining/alignment, LLM Basics Modules 2-5) — the **only** genuinely new thing that needs to be learned is the **mapping between the two representation spaces**, which turns out to be learnable with a comparatively tiny, cheap-to-train linear/MLP projection layer, not a full joint retraining of either large component.

### Training stages (worth knowing at a high level)
1. **Feature alignment pretraining**: train *only* the projection layer (both vision encoder and LLM stay frozen) on image-caption pairs, teaching the projection to produce visual tokens the frozen LLM can meaningfully use — directly analogous to CLIP's alignment goal, but now aligning into an *existing* LLM's specific embedding space rather than training a wholly new joint space from scratch.
2. **Visual instruction tuning**: fine-tune the projection layer **and** the LLM itself (the vision encoder commonly stays frozen even at this stage) on instruction-following data that involves images — directly LLM Basics Module 4/5's instruction-tuning concept, now extended to multimodal instructions ("describe this image," "what's unusual about this picture," etc.) rather than text-only instructions.

### The known limitation of this simple approach
Because the LLM sees visual tokens as "just more tokens" with no architecturally-special treatment, and the projection is a comparatively simple/shallow transformation, this approach can be **less capable at genuinely fine-grained, spatially-precise visual reasoning** than architectures with dedicated cross-attention fusion (Module 4) — the LLM has to learn to extract fine-grained spatial relationships purely through its standard self-attention over the projected tokens, without any architectural bias toward doing so, similar in spirit to Module 2's ViT-vs-CNN inductive-bias discussion (less specialized structure, more reliance on scale/data to compensate).

---

## 2. Gemini-style native multimodal architectures

### The core philosophical difference from LLaVA
Rather than bolting a vision encoder onto a pretrained text-only LLM via an adapter (LLaVA's approach — text-first, vision added on), a **natively multimodal** model is designed and **pretrained from the start** to jointly process interleaved sequences of text, image, (and often audio/video) tokens within a single unified transformer — there's no "frozen LLM that vision gets projected into" distinction; every modality's tokens are, from the very beginning of training, just part of the same sequence type the model has always processed.

### What "interleaved token sequences" means concretely
Training data isn't structured as separate (image, caption) pairs fed through distinct pathways — it's structured as **long documents/sequences where text and image tokens are interspersed in their natural order** (e.g., a webpage or document with paragraphs of text and embedded images/diagrams, tokenized and concatenated in reading order) — the model learns to predict/process the *next* token regardless of whether that token happens to be a text token or an image-patch-derived token, using essentially the same underlying autoregressive/unified objective throughout (LLM Basics Module 2's CLM framing, now generalized across modalities rather than being text-specific).

### Why this is architecturally more ambitious than LLaVA's approach
This requires solving Module 1's core embedding-space-alignment problem **from scratch, jointly, across the entire pretraining process**, rather than leveraging a pre-solved alignment (CLIP) and bridging it into an already-fully-trained separate LLM after the fact — a genuinely harder training problem (more moving pieces jointly optimized from the start), but with the potential upside of a model whose internal representations were **never modality-siloed in the first place** — every layer of the network, not just a final projection point, has the opportunity to develop genuinely joint, modality-agnostic reasoning patterns, since it was never trained purely on one modality before being introduced to the other.

### The practical tradeoff to name explicitly
Native multimodal pretraining is dramatically more expensive and complex (requires interleaved multimodal pretraining data at the necessary scale, and joint optimization of the entire model rather than reusing an already-pretrained frozen LLM) — this is precisely why the LLaVA-style adapter approach remains extremely popular and practical for building capable VLMs *without* needing to pretrain a foundation model from scratch (a much more accessible path for most practitioners/researchers), while native multimodal pretraining (Gemini-style) is the domain of the largest labs with the resources to pretrain foundation models from the ground up, generally yielding a more deeply, natively integrated multimodal capability as a result.

---

## 3. Image Tokenization for LLMs — how an image literally becomes "tokens"

### Recap and extension of Module 2's patchification
The mechanical process (however the specific architecture wires it in) always starts with Module 2's patchification: image → fixed-size patches → linear projection → a sequence of continuous-valued vectors. The key distinction from text tokenization worth stating precisely (a common point of confusion): **there is no discrete "vocabulary" of image tokens the way there's a discrete vocabulary of text subword tokens** (LLM Basics Module 1) — each image patch's embedding is a **continuous vector** produced by a learned linear/nonlinear function of that specific patch's actual pixel content, not a lookup into a fixed, finite table of possible "image token IDs." This is sometimes described as image tokens being **"soft" tokens** (continuous-valued) as opposed to text's **"hard" tokens** (discrete IDs from a fixed vocabulary) — a precise, useful distinction to draw if asked "is an image token the same kind of thing as a text token."

### Resolution and the "how many tokens does an image cost" question (previewed here, expanded in Module 9)
Since patch count scales with image resolution (Module 2's math: more pixels ÷ fixed patch size = more patches), higher-resolution images produce **more visual tokens**, directly increasing the LLM's effective context length consumption for that single image — this is exactly why compression techniques (Q-Former, Perceiver Resampler, Module 4) that cap visual token count at a small fixed number are such a practically important design lever, and why some production VLMs use **tiling** strategies for high-resolution images (splitting a large image into multiple lower-resolution tiles, each separately patchified, rather than naively patchifying one enormous image into a huge number of patches) — a detail this syllabus returns to in Module 9's practical/deployment coverage.

---

## 4. Side-by-side summary table (memorize this cold)

| | LLaVA-style (adapter) | Gemini-style (native multimodal) |
|---|---|---|
| Vision encoder | Frozen, pretrained separately (e.g., CLIP ViT) | Trained jointly as part of the unified model from the start |
| LLM | Frozen or lightly fine-tuned, pretrained separately on text first | No separate "text-only" pretraining phase — multimodal from the start |
| Bridge mechanism | Small trainable linear/MLP projection layer | No explicit "bridge" — unified architecture processes all modality tokens the same way throughout |
| Training cost/complexity | Lower — reuses two already-competent frozen/near-frozen components | Much higher — requires interleaved multimodal data at scale and full joint optimization |
| Accessibility | Practical for most practitioners/researchers to build on top of existing models | Effectively limited to large labs with foundation-model pretraining resources |
| Depth of cross-modal integration | Good, but bounded by the simplicity of the projection bridge | Potentially deeper — no layer of the network was ever purely single-modality |

---

## 5. Quick-fire Q&A (self-test)

**Q: In LLaVA's architecture, what exactly does the trainable projection layer do, and why is such a simple component sufficient?**
A: It maps the frozen vision encoder's patch embeddings into the same dimensional space as the frozen/fine-tuned LLM's token embeddings, producing "visual tokens" the LLM's existing, unmodified self-attention can process directly alongside text tokens. It's sufficient because both the vision encoder (via CLIP's contrastive pretraining) and the LLM (via its own text pretraining/alignment) are already highly competent — the only genuinely new thing to learn is the mapping between their two representation spaces, a comparatively small, targeted problem.

**Q: What is the core philosophical/architectural difference between LLaVA-style and Gemini-style (native multimodal) VLMs?**
A: LLaVA bridges two separately-pretrained, largely frozen components (a vision encoder and an LLM) via a small adapter layer — text-first, vision added afterward. Native multimodal models are pretrained from the start on interleaved multimodal sequences, with no separate text-only pretraining phase and no explicit "bridge" component — every layer processes all modality tokens using the same unified mechanism throughout training.

**Q: Why is "interleaved token sequences" a meaningfully different training data structure than CLIP's (image, caption) pairs?**
A: Interleaved sequences reflect the natural order text and images appear together in real documents (paragraphs and embedded images in reading order), letting the model learn to predict/process the next token regardless of its modality using one unified objective — rather than CLIP's structure of discrete, separately-encoded (image, caption) pairs compared only after full independent encoding.

**Q: What does it mean to say image tokens are "soft" while text tokens are "hard," and why does this distinction matter?**
A: Text tokens are discrete IDs looked up from a fixed, finite vocabulary (LLM Basics Module 1's BPE/WordPiece output); image tokens are continuous-valued vectors produced by a learned function of each patch's actual pixel content, with no equivalent fixed, finite "vocabulary of possible image tokens." This matters because it means there's no notion of image-token "identity" the way there's a stable, reusable text-token identity — every image patch produces its own unique continuous embedding based on its specific content.

**Q: Why does the LLaVA approach remain popular despite Gemini-style native multimodal potentially achieving deeper integration?**
A: Native multimodal pretraining requires interleaved multimodal data at massive scale and full joint optimization of the entire model from scratch — dramatically more expensive and complex, and effectively limited to large labs with foundation-model pretraining resources. LLaVA's adapter approach reuses two already-competent, separately-pretrained frozen components, making it a far more accessible and practical path for most practitioners building capable VLMs without needing to pretrain a foundation model from the ground up.

**Q: Why does image resolution directly affect an LLM's effective context length consumption in a VLM, and what's one practical mitigation?**
A: Patch count scales with resolution (more pixels ÷ fixed patch size = more patches), so higher-resolution images produce more visual tokens, consuming more of the LLM's context budget per image. Mitigations include compression techniques that cap visual token count at a small fixed number regardless of resolution (Q-Former, Perceiver Resampler, Module 4), or tiling strategies that split large images into multiple separately-processed lower-resolution tiles.

---
*End of Multimodal Module 5 (expanded). Next: Module 6 — Training Objectives & Data (image-text pair pretraining, visual instruction tuning, captioning and VQA as tasks).*
