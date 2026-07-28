# Multimodal Module 6 — Training Objectives & Data (Master Notes, Expanded)

## 0. Why this module is separate from the architecture modules

Modules 2-5 covered *what* the components are (encoders, fusion mechanisms, full VLM assemblies). This module covers *what data and objectives actually train them* — a genuinely distinct question, since the same architecture can be pretrained very differently depending on data source and objective, with real consequences for what the resulting model is actually good at.

---

## 1. Pretraining Data: Image-Text Pairs

### The dominant source — web-scraped alt-text
The large-scale data underlying CLIP-style contrastive pretraining (Module 3) and vision-encoder pretraining generally comes overwhelmingly from **web-scraped images paired with their alt-text or surrounding captions** — HTML `alt` attributes, image captions on web pages, and similar naturally-occurring text-near-image associations, collected at massive scale (hundreds of millions to billions of pairs for the largest models).

### The known data-quality problem this creates
Alt-text is written by web page authors for accessibility/SEO purposes, not as careful, accurate image descriptions — this means the raw scraped data is **noisy**: captions can be irrelevant, overly generic ("image1.jpg" or a page's unrelated title text used as filler alt-text), or only loosely related to the actual visual content. This directly connects to LLM Basics Module 2's pretraining-data-quality discussion (deduplication, deliberate mixture ratios) — multimodal pretraining data curation is, if anything, a **harder** filtering problem, since assessing "does this caption actually, accurately describe this image" requires either an existing capable model to do the filtering (a bootstrapping problem) or expensive human review at a scale that's impractical for billion-pair datasets.

### Filtering approaches worth knowing
- **CLIP-score filtering**: use an already-trained CLIP model to score candidate image-caption pairs by their embedding similarity, discarding low-scoring (likely mismatched/noisy) pairs before training a *new* model — a bootstrapping approach where an earlier model's judgment filters data for training a better one.
- **Caption rewriting/regeneration**: use a capable existing captioning model to generate a **new**, more accurate/descriptive caption for each image, replacing or supplementing the original noisy web alt-text — this is a documented technique behind some notable dataset improvements (higher-quality synthetic captions measurably improving downstream model quality vs. training on raw noisy web alt-text alone).

---

## 2. Interleaved Document Data

### What it is, and why it's a distinct data type from simple pairs
Beyond isolated (image, caption) pairs, native multimodal pretraining (Module 5's Gemini-style approach) additionally requires **long documents where text and images naturally co-occur in sequence** (web pages, PDFs, articles with embedded figures) — preserving the natural reading-order interleaving, rather than extracting isolated image-caption pairs out of their surrounding context.

### Why this data type specifically matters for native multimodal training
Interleaved documents let the model learn **longer-range, contextual relationships** between images and surrounding text that go beyond simple "this specific caption describes this specific image" (e.g., a diagram referenced and explained across several preceding and following paragraphs, not immediately adjacent to it) — this is a genuinely different, richer training signal than isolated pairs provide, and is specifically necessary (not just helpful) for training a model to handle real-world interleaved multimodal input at inference time (Module 5's "reading a document with embedded images" use case) — a model trained purely on isolated pairs has no direct training signal preparing it for that longer-range, document-level interleaved structure.

---

## 3. Visual Instruction Tuning (LLaVA-style data generation)

### The problem this solves
Pretraining (CLIP-style contrastive alignment, or native multimodal next-token prediction) teaches a model to produce meaningful joint representations and/or predict plausible next tokens — but, exactly parallel to LLM Basics Module 4's point about text-only pretraining, this doesn't teach the model to **follow instructions** about images ("describe what's happening in this image," "what's the person in the red shirt doing," "is this image safe for a child to see") — that requires a distinct instruction-tuning stage, now extended to multimodal instructions.

### The data-generation trick — bootstrapping instruction data with a text-only LLM
A genuinely clever, worth-knowing technique from the original LLaVA paper: to generate large-scale visual instruction-tuning data **without expensive human annotation**, they took existing image datasets with structured annotations (bounding boxes, existing captions, object labels) and **fed those structured text annotations (describing the image's content in text form, without the model ever seeing the actual image) to a strong text-only LLM (GPT-4 at the time), prompting it to generate plausible multi-turn conversational instruction-response pairs** (questions a user might ask about the image, and appropriate detailed answers) — entirely from the structured text description, no image input to the generating LLM required. This is a specific, real, and interview-relevant instance of using a strong existing model to bootstrap training data for a new capability (directly analogous in spirit to LLM Basics Module 5's RLAIF, using AI-generated rather than human-generated supervision, though for instruction data generation here rather than preference labeling).

### Why this bootstrapping approach is clever, stated precisely
It sidesteps the need for either expensive human annotators writing multimodal conversations from scratch, or an already-existing capable *multimodal* model to generate the data (a circular bootstrapping problem, since you're trying to train the first capable multimodal instruction-following model) — by using a strong **text-only** model conditioned on **structured text descriptions** of images (not the images themselves), you can generate plausible, diverse, image-grounded conversational data using a capability (strong text-only instruction generation) that already existed independently of the multimodal capability you're trying to build.

---

## 4. Captioning and VQA as Training/Evaluation Tasks

### Image Captioning
The task of generating a natural-language description of an image's content — used both as a **training objective** (generate the correct caption given the image, a direct generative/CLM-style loss over caption tokens conditioned on image tokens — LLM Basics Module 2's CLM formula, now with image tokens prepended as additional conditioning context) and as an **evaluation task** (Module 8 covers the specific metrics like CIDEr in more depth).

### Visual Question Answering (VQA)
The task of answering a natural-language question about an image's content — a genuinely distinct task shape from captioning (captioning is open-ended "describe this," VQA is targeted "answer this specific question about this") — used extensively both as a training signal (a large fraction of visual instruction-tuning data, Section 3, is effectively VQA-shaped: question about an image, appropriate answer) and as a standard evaluation benchmark task.

### Why both matter as distinct training signals, not redundant
Captioning trains **broad, holistic scene understanding and description** (what's generally present, described comprehensively); VQA trains **targeted, query-specific attention and reasoning** (find and reason about the specific detail the question asks about, ignoring irrelevant parts of the image) — a model trained only on captioning data might be weaker at answering specific, narrow questions requiring focused attention to one image region/detail, while a model trained only on narrow VQA-style data might be weaker at producing comprehensive, well-organized holistic descriptions — a genuinely complementary pair of training signals, worth naming both explicitly if asked "what data do you need to train a good VLM," rather than treating them as interchangeable.

---

## 5. Side-by-side summary table (memorize this cold)

| | Web-scraped image-text pairs | Interleaved documents | Visual instruction-tuning data |
|---|---|---|---|
| Structure | Isolated (image, caption) pairs | Long documents with natural text-image interleaving | Multi-turn conversational (image, instruction, response) |
| Primary use | CLIP-style contrastive pretraining (Module 3), basic vision-language alignment | Native multimodal pretraining (Gemini-style, Module 5) | Instruction-tuning stage (post-pretraining) |
| Key quality challenge | Noisy/inaccurate alt-text | Requires preserving natural document structure at scale | Requires diverse, high-quality instruction-response pairs |
| Common fix/technique | CLIP-score filtering, caption regeneration | Careful large-scale document-source curation | Bootstrap via text-only LLM conditioned on structured text annotations (LLaVA's approach) |

---

## 6. Quick-fire Q&A (self-test)

**Q: What's the core data-quality problem with web-scraped image-text pairs, and why is it a harder filtering problem than text-only pretraining data curation?**
A: Alt-text/captions are written for accessibility/SEO, not accurate description, so raw pairs are often noisy, generic, or loosely related to actual image content. It's a harder filtering problem than text-only curation because assessing "does this caption accurately describe this image" requires either an existing capable model to judge the match (a bootstrapping problem) or expensive human review at a scale impractical for billion-pair datasets.

**Q: What specific capability does interleaved document data provide that isolated image-caption pairs cannot?**
A: It provides longer-range, contextual image-text relationships (e.g., a diagram discussed across several surrounding paragraphs, not immediately adjacent) — training signal specifically necessary for a model to handle real-world interleaved multimodal input (documents, web pages) at inference time, which isolated-pair training provides no direct preparation for.

**Q: Explain LLaVA's bootstrapping technique for generating visual instruction-tuning data, and why it avoids a circularity problem.**
A: It feeds existing structured text annotations of images (captions, bounding boxes, object labels — text only, no actual image input) to a strong text-only LLM, prompting it to generate plausible multi-turn instruction-response conversations about that image's content. This avoids circularity because it uses a capability that already existed independently (strong text-only instruction generation) rather than requiring an already-capable multimodal model to generate multimodal training data for itself.

**Q: Why are captioning and VQA considered complementary rather than redundant training signals?**
A: Captioning trains broad, holistic scene understanding and comprehensive description; VQA trains targeted, query-specific attention and reasoning toward a particular detail while ignoring irrelevant parts of the image. A model trained only on one tends to be weaker at the other's specific skill — comprehensive description vs. focused, query-driven reasoning are genuinely different capabilities.

**Q: What is CLIP-score filtering, and what kind of technique does it represent more generally?**
A: Using an already-trained CLIP model to score candidate image-caption pairs by embedding similarity and discarding low-scoring, likely-mismatched pairs before training a new model on the filtered data — a bootstrapping technique where an earlier model's judgment is used to curate/filter training data for a subsequent, ideally better, model.

---
*End of Multimodal Module 6 (expanded). Next: Module 7 — Generation: Text-to-Image Basics (diffusion mechanics, text conditioning via cross-attention).*
