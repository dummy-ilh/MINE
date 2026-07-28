# Multimodal Module 8 — Evaluation of Multimodal Models (Master Notes, Expanded)

## 0. How this builds on LLM Basics Module 8

Everything from LLM Basics Module 8 (benchmark contamination/saturation, human eval, LLM-as-judge biases, hallucination and calibration) applies to multimodal models too — the underlying evaluation *principles* don't change. This module covers what's **genuinely new**: metrics specific to captioning/VQA, multimodal benchmark suites, and — critically — a visually-grounded hallucination failure mode that doesn't have a direct text-only analog.

---

## 1. Captioning Metrics

### The core evaluation problem
Given a generated caption and one or more human-written reference captions for the same image, how do you score similarity — accounting for the fact that there are many valid, differently-worded ways to correctly describe the same image (unlike, say, exact-match benchmark questions from LLM Basics Module 8, where there's typically one correct answer)?

### CIDEr (Consensus-based Image Description Evaluation)
The dominant classical captioning metric. Core idea: weight n-gram matches between the generated caption and reference captions by their **TF-IDF-style informativeness** — n-grams that appear across *many different* reference captions for *many different* images (generic phrases like "a photo of" or "in the image") get **down-weighted**, while n-grams that are more distinctive/specific to this particular image's actual content get **up-weighted**. **In plain words**: CIDEr specifically rewards matching the *distinctive, informative* content of a caption, not just any word overlap — a generated caption that's technically similar to the reference only via generic filler phrases scores lower than one that captures the reference's specific, informative details, even if raw word-overlap counts were similar.

### Other metrics worth knowing by name (brief)
- **BLEU**: originally a machine-translation metric (n-gram precision-based), adapted for captioning — generally considered a weaker fit than CIDEr for captioning specifically, since it doesn't have CIDEr's informativeness-weighting refinement.
- **METEOR**: incorporates synonym matching and stemming (not just exact n-gram overlap), addressing some of BLEU's rigidity, though still less commonly the headline metric than CIDEr for captioning leaderboards.
- **SPICE**: parses captions into a structured scene-graph representation (objects, attributes, relationships) and compares scene graphs rather than raw text overlap — closer to genuinely comparing *semantic content* rather than surface-level word overlap, at the cost of being more complex to compute.

### The shared limitation across all of these (an important interview-level caveat)
All of these are **automated, reference-based, n-gram-or-structure-overlap metrics** — none of them directly measure whether a caption is factually accurate about the actual image content the way a human judge would; they measure similarity to human-written references, which is a real but imperfect proxy (exactly the same proxy-metric gap theme from LLM Basics Module 2/3/8, now recurring here). A caption could score well by superficially resembling reference phrasing while still containing a subtle factual error the metric has no mechanism to catch.

---

## 2. VQA Accuracy

### The metric, and a specific wrinkle worth knowing
Standard VQA benchmarks (like the VQA dataset itself) typically collect **multiple human answers per question** (often 10), since many visual questions have some genuine ambiguity or multiple acceptable phrasings of a correct answer ("red" vs. "it's red" vs. "the color is red"). The standard VQA accuracy formula for a given model answer is:
```
accuracy = min(1, (number of humans who gave this exact answer) / 3)
```
**In plain words**: if at least 3 of the (typically 10) human annotators gave the exact same answer the model gave, the model gets full credit (accuracy=1) for that question; fewer matching humans gives partial credit, scaled proportionally. This specific "at least 3 humans agree" threshold is a real, concrete detail worth having ready — it's a deliberate design choice acknowledging genuine answer ambiguity, rather than requiring a single rigid ground-truth string match the way a simpler benchmark might (LLM Basics Module 8's MMLU, by contrast, has one clearly correct multiple-choice answer per question, a meaningfully different and simpler evaluation shape).

---

## 3. Multimodal Benchmark Suites (brief — know the categories and what they target)

- **MMMU (Massive Multi-discipline Multimodal Understanding)**: broad, MMLU-style (LLM Basics Module 8) breadth-of-knowledge benchmark, but requiring genuine image understanding across many academic/professional disciplines — directly the multimodal analog of MMLU, inheriting the same general strengths (broad coverage) and weaknesses (contamination risk, saturation as models improve, LLM Basics Module 8's exact concerns, now applied to a multimodal dataset).
- **VQAv2 / GQA-style benchmarks**: targeted visual question-answering benchmarks, some (like GQA) specifically designed with compositional, multi-step reasoning questions about object relationships (echoing Module 3's point about CLIP's dual-encoder weakness at fine-grained compositional reasoning — these benchmarks are, not coincidentally, good tests for exactly that capability gap).
- **Document/chart understanding benchmarks**: test structured-visual-content understanding (reading charts, tables, documents with embedded figures) — a distinct, practically important capability from natural-photo understanding, directly relevant to Module 9's document-understanding practical use case.

**Interview-level point, reusing LLM Basics Module 8's framing directly**: name the *category* and what capability gap it targets, rather than trying to recite exact leaderboard numbers — the same advice given for agent benchmarks (Agents Module 8) and LLM benchmarks (LLM Basics Module 8) applies identically here.

---

## 4. Hallucination in VLMs — a distinct, visually-grounded failure mode

### The core phenomenon
LLM Basics Module 8 covered hallucination as fabricated/unsupported *factual* claims in text generation. VLM hallucination has a specific, additional flavor: the model describes **objects, attributes, or relationships that are not actually present in the image** — e.g., confidently stating "there's a dog in the image" when no dog is present, or describing an object's color/position incorrectly — a **visually-grounded** factual error, checkable directly against the actual image content (unlike open-domain text hallucination, which often requires external world-knowledge verification, this is checkable against the specific input right in front of the model).

### Why this happens — mechanistically, connecting back to earlier modules
Two contributing mechanisms worth naming: (1) the same CLM-style generative training objective (LLM Basics Module 2) that produces fluent, plausible-sounding text has no built-in mechanism forcing strict grounding in the actual visual input, just as it has no built-in mechanism forcing strict grounding in factual truth generally (LLM Basics Module 8's hallucination mechanism, directly recurring here); (2) **language priors/bias** — VLMs trained on large text-heavy corpora can develop strong *statistical* priors about what objects/attributes commonly co-occur in captions (e.g., "grass" and "tree" frequently appear together in training captions), and can sometimes hallucinate a commonly-co-occurring object into a description even when it's not actually present in this specific image, essentially defaulting to a strong learned text-side prior over careful visual verification — a failure mode with no direct text-only-LLM analog, since it specifically stems from the tension between the model's *visual* input and its *learned linguistic* co-occurrence patterns.

### Measurement approaches specific to VLM hallucination
- **Object-presence probing (e.g., POPE-style benchmarks)**: directly ask the model yes/no questions about whether specific objects are present in an image (including questions about objects that are plausible-sounding but actually absent, specifically probing the language-prior-driven hallucination mechanism above) — a clean, directly-checkable-against-ground-truth evaluation, unlike open-domain text hallucination's harder external-verification requirement (LLM Basics Module 8).
- **Caption-based fact extraction and verification**: decompose a generated caption into individual claimed objects/attributes/relationships, and check each against the image's actual annotated ground truth (when available) — the multimodal-specific version of LLM Basics Module 8's "decompose into claims, verify each" fact-verification approach, made more tractable here since the "source of truth" is the specific input image itself, not an open-domain external knowledge base.

### Interview-ready synthesis
"VLM hallucination inherits the general CLM-objective-doesn't-guarantee-truthfulness mechanism from text-only hallucination, but adds a visually-specific failure mode: strong learned language co-occurrence priors can override careful visual grounding, causing the model to describe plausible-sounding but actually-absent content. The upside, evaluation-wise, is that this is often more directly and cheaply checkable than open-domain text hallucination, since the ground truth is right there in the input image rather than requiring external world-knowledge verification."

---

## 5. Side-by-side summary table (memorize this cold)

| | CIDEr (captioning) | VQA Accuracy | POPE-style object-presence probing |
|---|---|---|---|
| What it measures | Weighted n-gram overlap with references, favoring distinctive/informative content | Agreement with human answers (≥3 of ~10 annotators for full credit) | Yes/no accuracy on object-presence questions, including plausible-but-absent probes |
| Reference-based? | Yes | Yes | Yes (ground-truth image annotations) |
| Directly checks factual/visual accuracy? | No — measures reference similarity, not independent factual verification | Partially — measures agreement with human-given answers | Yes — directly probes visually-grounded hallucination specifically |

---

## 6. Quick-fire Q&A (self-test)

**Q: What does CIDEr specifically reward that a simpler raw n-gram-overlap metric would not?**
A: It weights n-gram matches by TF-IDF-style informativeness, down-weighting generic phrases common across many captions/images ("a photo of") and up-weighting distinctive, image-specific content — rewarding captures of a reference's specific, informative details rather than any word overlap, including generic filler.

**Q: Write the standard VQA accuracy formula and explain the reasoning behind the "3" threshold.**
A: `accuracy = min(1, (number of humans who gave this exact answer)/3)`. The threshold acknowledges genuine answer ambiguity in visual questions (multiple correct phrasings/answers are common) — requiring only 3 of typically 10 annotators to agree for full credit, rather than a single rigid ground-truth match, is a deliberate design choice reflecting that visual questions often don't have one uniquely "correct" string answer.

**Q: What's the core limitation shared by all reference-based captioning metrics (CIDEr, BLEU, METEOR, SPICE)?**
A: They measure similarity to human-written reference captions, not independent, direct verification of factual accuracy against the actual image content — a caption could score well by superficially resembling reference phrasing while still containing a factual error the metric has no mechanism to catch.

**Q: Describe the "language prior" mechanism behind a specific kind of VLM hallucination, and why it has no direct text-only-LLM analog.**
A: VLMs trained on caption-heavy text corpora develop strong statistical priors about commonly co-occurring objects/attributes (e.g., "grass" and "tree" frequently appearing together), and can hallucinate a commonly-co-occurring object into a description even when it's actually absent from the specific image — defaulting to a learned linguistic co-occurrence pattern over careful visual verification. This has no direct text-only analog because it specifically arises from tension between visual input and learned text-side co-occurrence statistics, a distinctly multimodal failure mode.

**Q: Why is VLM hallucination measurement often more tractable than open-domain text hallucination measurement (LLM Basics Module 8)?**
A: The "ground truth" for verification is the specific input image itself, directly available and checkable, rather than requiring external world-knowledge sources or fact-checking against a comprehensive open-domain reference — object-presence probing (POPE-style) can directly and cheaply verify claims against known image annotations, unlike open-domain text claims which often require harder, less directly verifiable fact-checking.

---
*End of Multimodal Module 8 (expanded). Next: Module 9 — Practical/Deployment Aspects (preprocessing, resolution/tiling, visual token cost, tooling).*
