# Multimodal Module 10 — Interview Synthesis (Master Notes, Maximum Depth)

## 0. How to use this module

Same purpose as the final synthesis modules in your other syllabi: cross-module questions organized by type, since that's how interviewers actually probe. Assumes Modules 1-9 as background — each answer names the specific module to revisit if anything feels shaky.

---

## 1. "Walk me through end-to-end" questions

### Q: A user uploads a photo and asks "what's happening in this image?" — walk through everything that happens, technically, from upload to answer.

**Strong answer structure**:
1. **Preprocessing** (Module 9): resize/rescale to the vision encoder's expected input resolution, normalize using the correct training-time statistics, handle aspect ratio (crop or pad).
2. **Patchification** (Module 2): the image is split into fixed-size patches, each linearly projected into an embedding vector — a sequence of "soft," continuous-valued visual tokens (Module 5's distinction from discrete text tokens).
3. **Vision encoding**: the patch sequence passes through the ViT's bidirectional self-attention layers (no causal masking — Module 2), producing contextualized patch representations.
4. **Bridging into the LLM** — depends on architecture: LLaVA-style (Module 5) applies a small linear/MLP projection directly into the LLM's token embedding space and concatenates as more tokens; a Q-Former/Perceiver-style system (Module 4) first compresses the patch sequence into a small fixed number of tokens via learned-query cross-attention, then feeds those into the LLM.
5. **Joint processing**: the LLM processes the combined visual+text token sequence using its standard (LLaVA-style) or interleaved-cross-attention (Flamingo-style) mechanism, generating a text answer autoregressively (LLM Basics Module 2's CLM, now conditioned on visual tokens as part of the context).
6. **Cost consideration** (Module 9): note explicitly how many tokens the image consumed, and whether compression was used — a strong answer flags this unprompted.

**What this tests**: whether you connect preprocessing → encoding → bridging → generation as one pipeline, not just recite architecture facts in isolation.

### Q: Walk through how you'd train a VLM from an existing pretrained LLM and vision encoder, LLaVA-style.

Reuse Module 5/6 directly: Stage 1 — freeze both the vision encoder and LLM, train only the projection layer on image-caption pairs (Module 6's web-scraped data, filtered per Module 6's quality discussion) to align visual tokens into the LLM's embedding space, conceptually similar in spirit to CLIP's alignment goal (Module 3) but now targeting an *existing* LLM's specific space. Stage 2 — visual instruction tuning: fine-tune the projection layer and the LLM (vision encoder typically still frozen) on multimodal instruction-response data, generated via the LLaVA bootstrapping trick (Module 6: a strong text-only LLM given structured text image annotations, no actual image input to the generator).

---

## 2. "Compare and decide" questions

### Q: CLIP's dual-encoder approach vs. cross-attention fusion — when would you use each?

Reuse Module 3/4's tradeoff directly: dual-encoder (CLIP) when you need efficient, large-scale retrieval (precomputed, cacheable embeddings, fast similarity search) and the task is more holistic (does this image roughly match this text) than fine-grained/compositional. Cross-attention fusion when the task needs genuine fine-grained, spatially-localized cross-modal reasoning (specific object relationships, detailed grounding) — at the cost of losing the precompute/cache efficiency, since encoding now depends on the specific pairing being evaluated.

### Q: LLaVA-style adapter approach vs. Gemini-style native multimodal — which would you recommend for a new product, and why?

Reuse Module 5's honest tradeoff: LLaVA-style for almost any practical, resource-constrained scenario — reuses already-competent frozen components, dramatically cheaper and faster to build, and is the practically dominant real-world approach (Module 9). Native multimodal only makes sense if you have foundation-model-scale pretraining resources and need the deepest possible cross-modal integration (e.g., genuinely fluid interleaved multimodal reasoning across many modalities) — for a product team without that scale, recommending native-from-scratch pretraining would be a red flag, not a sophisticated answer.

### Q: Q-Former/Perceiver compression vs. raw uncompressed patch tokens — when do you accept the higher token cost?

Reuse Module 9's decision framework: accept uncompressed/tiled higher token cost specifically when fine-grained visual fidelity is required (document/chart reading, small text, detailed inspection); default to compression when holistic understanding suffices and cost/latency matters more (general photo description, casual VQA) — name the specific use case rather than picking one universally.

### Q: Diffusion-based image generation vs. autoregressive token-by-token generation — why has diffusion become dominant?

Reuse Module 7 directly: images lack a natural sequential reading order the way text does, making a strictly sequential autoregressive generation order an awkward architectural fit; diffusion's iterative denoising process doesn't impose that same artificial ordering constraint, and has generally produced higher quality results for this reason — while still being honest that autoregressive image-token approaches exist and are used in some systems, not claiming diffusion is the only valid approach.

---

## 3. "Diagnose the failure" questions

### Q: Your VLM confidently describes an object that isn't actually present in the image. Diagnose and propose fixes.

This is Module 8's VLM-specific hallucination failure mode. Diagnosis: distinguish whether this is a general CLM-fluency-without-grounding issue (LLM Basics Module 8's mechanism, recurring here) or specifically a language-prior-driven hallucination (the model defaulting to a commonly-co-occurring object from its learned caption statistics rather than actually attending to this specific image's content) — use object-presence probing (POPE-style, Module 8) to specifically test whether the model over-predicts commonly-co-occurring-but-absent objects, which would confirm the language-prior mechanism. Fixes: instruction-tuning data specifically including negative examples (explicitly confirming an object's absence, not just describing present objects), and/or stronger visual-grounding training signal (more targeted VQA-style data requiring precise attention to specific image regions, per Module 6's captioning-vs-VQA complementary-signal point).

### Q: Your VLM performs well on holistic image description but fails badly on questions like "is the red object to the left or right of the blue object." Diagnose.

This is Module 3's exact CLIP dual-encoder limitation, if the underlying architecture relies primarily on late-fusion/holistic alignment rather than genuine cross-attention fusion. Diagnosis: check whether the vision-language bridge uses fine-grained cross-attention (Module 4) or a simpler holistic projection without spatial-reasoning-specific training signal. Fix: incorporate cross-attention fusion (Module 4) if not already present, and/or add training data specifically targeting compositional/spatial-relationship VQA (GQA-style data, Module 8), since this is precisely the capability gap those benchmarks were designed to probe and that CLIP-style holistic alignment structurally underserves.

### Q: A document-understanding VLM performs well on natural photos but poorly on scanned documents with small text. Diagnose.

This connects Module 9's resolution/tiling material directly: small text in a document requires fine-grained visual detail that a single fixed-low-resolution encoder pass will lose (Module 9's downsampling-loses-detail point). Diagnosis: check the effective resolution the vision encoder is actually processing the document at, and whether tiling is in use. Fix: implement tiling (Module 9) to preserve fine-grained detail at higher effective resolution, likely combined with compression (Module 4) to keep the resulting higher patch count from exploding total token cost, and consider whether document-specific training data (structured text/chart understanding, Module 8's benchmark category) was adequately represented in instruction tuning.

### Q: Your text-to-image model generates images that don't actually match the text prompt's specific details (e.g., asks for "a red car," gets a blue car).

This points to a text-conditioning failure in the diffusion cross-attention mechanism (Module 7). Diagnosis: check whether the text encoder used for conditioning genuinely captures the relevant semantic detail (e.g., is it a weak/undertrained text encoder, or a strong CLIP-style encoder — Module 3), and whether the cross-attention layers are adequately weighted/positioned within the U-Net/DiT (Module 7) to let that conditioning genuinely influence generation throughout the denoising process, not just weakly at a few layers. Fix: stronger/better-aligned text encoder, and/or classifier-free guidance strength tuning (a diffusion-specific technique for controlling how strongly the text conditioning influences generation relative to unconditional generation — worth naming if this comes up, even though it's beyond this module's core depth).

---

## 4. "Explain/derive the mechanism" questions

Have these fully loaded:

- **InfoNCE/CLIP loss derivation** (Module 3): the full formula, the temperature's role, and the numerical worked example showing how low temperature amplifies a similarity gap into a confident probability.
- **Patchification math** (Module 2): patch count for a given image size/patch size, and the sequence-length-vs-granularity tradeoff.
- **Cross-attention mechanics** (Module 4): the Q-from-one-sequence, K/V-from-another-sequence pattern, and a worked example (a text token attending to image patches).
- **Diffusion forward/reverse process** (Module 7): the noising formula, the simple MSE loss, and why the forward process needs no learning at all.
- **VQA accuracy formula** (Module 8): the min(1, matches/3) formula and why it exists.

---

## 5. Rapid-fire cross-module connections (say these unprompted when relevant)

- Cross-attention (Module 4) shows up in **three different places** across this syllabus, worth naming as one mechanism, not three: vision-language fusion for understanding (Module 4), Q-Former/Perceiver compression (Module 4), and text-conditioning in diffusion generation (Module 7) — same Q-from-one-sequence/K-V-from-another formula, three different applications.
- The ViT-vs-CNN inductive-bias tradeoff (Module 2) **recurs identically** in DiT-vs-U-Net for diffusion generation networks (Module 7) — less built-in spatial bias, more reliance on data/compute scale, same underlying pattern as LLM Basics Module 3's scaling-laws theme.
- LLaVA's "reuse frozen competent components, learn only a small bridge" philosophy (Module 5) is the **same underlying principle** as LLM Basics Module 4's LoRA (don't retrain everything, find a small targeted intervention) — worth naming this parallel explicitly if asked why the adapter approach works so well with so little new training.
- The proxy-metric gap theme — perplexity not predicting downstream task performance (LLM Basics Module 2/8), benchmark contamination/saturation (LLM Basics Module 8, recurring in MMMU per Module 8 here), and captioning metrics not directly verifying factual accuracy (Module 8) — is the **same structural issue appearing a fourth time**, worth naming as a throughline if the interviewer probes evaluation philosophy broadly.
- CLIP's contrastive alignment (Module 3) and RLHF's reward model (LLM Basics Module 5) are structurally different (contrastive vs. Bradley-Terry preference modeling) but serve an analogous *purpose* — both are training signals designed to align one representation/behavior against a target derived from paired/comparative data, worth drawing as a loose but real conceptual parallel if pressed on "what other training paradigms rely on paired comparison data."

---

## 6. System Design: "Design a VLM for document understanding" — fully worked

### Requirements/stakes
Task: extract and reason over information from scanned documents (contracts, forms, receipts, technical manuals with embedded charts/diagrams) — includes both natural free-text and structured/tabular content, often with small text requiring high effective resolution. Stakes vary by use case (informational Q&A vs. a downstream automated decision like contract compliance checking — flag this distinction, echoing the Agents System Design notes' "stakes determine architecture" principle).

### Vision encoder choice
A ViT-family encoder (Module 2), likely CLIP-pretrained-and-then-further-fine-tuned or trained further on document-specific data, since natural-photo CLIP pretraining alone underrepresents dense-text/structured-document visual statistics (Module 6's data-source-matters point, applied here).

### Resolution strategy — the central design decision for this specific use case
This is precisely Module 9's tiling scenario: naive downsampling to a small fixed resolution will destroy small text legibility. Use tiling — split the document image into multiple standard-resolution tiles, process each through the vision encoder, plus one global low-resolution view for overall document layout context (Module 9's "local detail + global context" combination).

### Bridging architecture
Given the need for fine-grained detail (individual words/numbers in specific document regions must be preserved, not just holistically summarized), lean toward Q-Former/Perceiver-style compression (Module 4) **per tile** rather than raw uncompressed concatenation — this keeps total token cost bounded even with many tiles, while still preserving per-tile fine-grained information via the learned-query cross-attention extraction (Module 4), rather than either losing detail (naive downsampling) or exploding token count (raw uncompressed tiling).

### Training data
Beyond generic web image-text pairs (Module 6), specifically needs: OCR-paired data (document images with ground-truth extracted text) and document-specific VQA/instruction data (Module 6/8's document-understanding benchmark category) — generic natural-photo captioning/VQA data alone (Module 6) will not adequately cover structured-document understanding, echoing Module 6's point that captioning and VQA are complementary but both still need to be *domain-matched* to the deployment use case.

### Guardrails and evaluation
Object/text-presence hallucination checking specifically matters here (Module 8's POPE-style probing, applied to document content — e.g., does the model hallucinate a contract clause that isn't actually present) given the potentially high-stakes downstream use (compliance/legal document review) — directly reusing the reward-hacking/hallucination caution from the Agents Issues notes' broader theme of not trusting a model's self-reported confidence for high-stakes claims. Evaluation should include both document-specific benchmark suites (Module 8) and, for any high-stakes automated-decision use case, human review gates before acting on extracted information (directly the Agents System Design pattern of confirmation gates before consequential actions, applied here to "acting on extracted document content" as the consequential step).

---

## 7. Final self-check — can you do all of these cold?

- [ ] Walk through the full image-in, text-out pipeline for a VLM, naming every module's contribution in order.
- [ ] Derive the InfoNCE/CLIP loss and reproduce the temperature numerical example.
- [ ] Give a balanced (not one-sided) answer to "LLaVA-style vs. Gemini-style" and "CLIP dual-encoder vs. cross-attention fusion."
- [ ] Diagnose VLM object hallucination, spatial-reasoning failure, and small-text/document-resolution failure, each with the specific correct fix.
- [ ] Explain why cross-attention is the same mechanism across vision-language fusion, Q-Former/Perceiver compression, and diffusion text-conditioning.
- [ ] Fully design a document-understanding VLM end-to-end, unprompted naming tiling, compression, domain-matched training data, and hallucination guardrails.
- [ ] Explain precisely why diffusion models don't need a "learned" forward process, and what the reverse process's network actually predicts.

If anything here feels shaky, that's a direct pointer back to the specific module above — everything on this list is covered in full depth somewhere in Multimodal Modules 1-9.

---
*End of Multimodal Module 10. This completes the Multimodal Models syllabus (Modules 1-10) — foundations through interview synthesis and full system design, all at full depth with mechanisms, numerical examples, and standalone real-world reference points.*
