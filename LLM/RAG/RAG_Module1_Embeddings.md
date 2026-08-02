# RAG Interview Master Notes — Module 1: Embeddings

> **How to use these notes:** Read the Quick Summary first, then the Deep Dives before an interview. The Q&A section at the end is your drill material — cover the answers and recite them cold.

---

## Quick Summary

An embedding is a function that converts text (or any discrete object) into a fixed-length list of numbers, arranged so that **similar meanings live close together in that numerical space**. RAG systems use embeddings to find relevant documents at speed: you pre-convert every document into a vector once, then at query time you convert the question into a vector and find the nearest document vectors — no expensive model call per document needed. The two core architectural choices are *how* you train embeddings (contrastive loss, hard negatives) and *what architecture* you use at retrieval time (bi-encoder vs cross-encoder).

---

## 1. What an Embedding Actually Is

### Key Concept

> **Think of it like this:** Imagine a city map where semantically related words live in the same neighbourhood. "Dog" and "puppy" are on the same block. "Dog" and "automobile" are across town. An embedding model learns to draw this map — not by hand, but from millions of examples of what humans consider "similar."

Formally, an embedding is a learned function:

```
f: X → ℝ^d
```

- **X** = your input space (words, sentences, documents, code, images)
- **ℝ^d** = d-dimensional real-valued space (a list of d floating-point numbers)
- **d** = the embedding dimension, typically 128 to 4096

The model is trained so that **geometric distance in ℝ^d ≈ semantic similarity in X**.

### Three Properties Every Good Embedding Has

**Locality** — semantically similar inputs produce nearby vectors. "What is the capital of France?" and "France's capital city?" should land close together, even though they share no words.

**Isotropy** — the vectors are spread out across all directions in the space, not squashed into a narrow cone. This matters because if all your vectors point in roughly the same direction, cosine similarity stops being informative (everything looks ~similar to everything else). Raw transformer outputs (plain BERT `[CLS]` tokens) famously suffer from this — they cluster in a narrow cone, which is why contrastive fine-tuning was invented.

**Fixed dimensionality** — regardless of whether the input is 3 words or 3,000, the output is always exactly d numbers. This requires a *pooling* step: the model reads all the tokens, then collapses them into one vector. Three common strategies:
- **CLS token pooling**: use the special `[CLS]` token's final hidden state
- **Mean pooling**: average all token hidden states (most common in modern models)
- **Last-token pooling**: use the last token (common in decoder-only LLMs)

### The Isotropy Problem — Why Raw BERT Doesn't Work

> **Interview gotcha:** Raw BERT `[CLS]` embeddings are *not* good similarity embeddings. Many candidates miss this.

Transformer training objectives (masked language modeling, next-token prediction) optimise for *predicting tokens*, not for *geometric similarity*. The resulting representation space has two problems:

1. **Anisotropy**: vectors cluster in a narrow cone. Cosine similarity between any two random sentences is surprisingly high — you get poor discrimination.
2. **Frequency/norm artifacts**: high-frequency words end up with large magnitude vectors, biasing dot-product similarity.

This is exactly *why* Sentence-BERT (SBERT, 2019) was a big deal — it was the paper that said "fine-tune with a contrastive objective specifically designed to fix the geometry," not just "use any transformer."

---

## 2. Training Objectives

### 2.1 InfoNCE / Contrastive Loss (The Workhorse)

**Plain English:** "Pull similar pairs close together, push dissimilar pairs apart — and do it for a whole batch at once."

#### The Formula

```
L = -log( exp(sim(a, p) / τ) / Σᵢ exp(sim(a, nᵢ) / τ) )
```

#### Breaking It Down, Symbol by Symbol

| Symbol | What it is | Intuition |
|--------|-----------|-----------|
| `a` | **anchor** | Your query or reference sentence |
| `p` | **positive** | A semantically matching sentence for `a` |
| `nᵢ` | **negatives** | Sentences that do *not* match `a` |
| `sim(·,·)` | **similarity function** | Usually cosine similarity or dot product |
| `τ` (tau) | **temperature** | Controls how "sharp" or "soft" the distribution is |
| `exp(·)` | e raised to a power | Converts similarity scores into positive numbers |
| `Σᵢ` | sum over all negatives | The denominator normalises across the whole batch |

#### Step-by-Step Numerical Example

Suppose you have:
- Anchor (a): "How do I train a neural network?"
- Positive (p): "Steps to train a deep learning model"
- Negative 1 (n₁): "Best pasta recipes for dinner"
- Negative 2 (n₂): "What is the boiling point of water?"

After encoding, imagine these similarity scores (cosine, ranging −1 to 1):

```
sim(a, p)  = 0.85   ← high: semantically related
sim(a, n₁) = 0.10   ← low: totally unrelated
sim(a, n₂) = 0.20   ← low: unrelated
```

With temperature τ = 0.07 (a typical value):

```
exp(sim(a,p)  / τ) = exp(0.85 / 0.07) = exp(12.14) ≈ 188,247
exp(sim(a,n₁) / τ) = exp(0.10 / 0.07) = exp(1.43)  ≈ 4.18
exp(sim(a,n₂) / τ) = exp(0.20 / 0.07) = exp(2.86)  ≈ 17.46
```

```
Denominator = 188,247 + 4.18 + 17.46 = 188,268.64

L = -log(188,247 / 188,268.64) = -log(0.99988) ≈ 0.00012
```

Low loss! The model is already separating positive from negatives well. Early in training, these numbers would be much closer, giving higher loss and larger gradients.

#### What Happens When You Change τ?

| τ value | Effect | Use case |
|---------|--------|----------|
| Very small (0.01–0.05) | Extremely sharp — near-miss negatives get harshly penalised | Hard-negative heavy training, forces fine-grained distinctions |
| Medium (0.05–0.1) | Standard — good balance | Most production embedding training |
| Large (0.5–1.0) | Soft — model barely discriminates | Not useful for retrieval; used sometimes in classification |

#### Why This Formula Matters in Practice

This is **the engine** behind every major embedding model (OpenAI ada, Cohere embed, BGE, E5). It's a softmax cross-entropy loss — the same mathematics as multi-class classification, except the "classes" are "which item is the positive." The temperature τ is the single most important hyperparameter to tune.

**Business example:** Spotify uses contrastive loss to train song embeddings — the anchor is a song, the positive is another song the same user listened to in the same session, and negatives are random other songs. The resulting embedding space clusters songs by listening context (acoustic indie folk ends up near lo-fi study beats, not because the audio is similar but because people listen to them together).

---

### 2.2 Triplet Loss (The Predecessor)

```
L = max(0, sim(a, n) - sim(a, p) + margin)
```

**Plain English:** "Make the positive similarity at least `margin` higher than the negative similarity. If it already is, loss is zero."

#### Breaking It Down

| Symbol | Meaning |
|--------|---------|
| `sim(a,p)` | similarity between anchor and positive |
| `sim(a,n)` | similarity between anchor and one negative |
| `margin` | minimum gap you require (e.g. 0.3) |
| `max(0, ...)` | if gap is already ≥ margin, no loss (don't disturb what's working) |

#### Numerical Example

```
sim(a, p) = 0.6, sim(a, n) = 0.5, margin = 0.3

L = max(0, 0.5 - 0.6 + 0.3) = max(0, 0.2) = 0.2  ← still has loss; model must push further apart
```

```
sim(a, p) = 0.8, sim(a, n) = 0.4, margin = 0.3

L = max(0, 0.4 - 0.8 + 0.3) = max(0, -0.1) = 0   ← no loss; gap is already > 0.3
```

**Why InfoNCE won over Triplet Loss:** Triplet loss only considers one negative at a time. InfoNCE considers the entire batch simultaneously, which gives a much richer gradient signal. With batch size 256, InfoNCE effectively trains with 255 negatives per anchor per step. Triplet loss trains with 1. Scale wins.

---

### 2.3 In-Batch Negatives

> **Think of it like this:** You're in a library with 256 people all holding books. You pick up *your* book (the positive). Every other person's book becomes a negative — for free, no extra work.

In a training batch of B (anchor, positive) pairs:
- For anchor `aᵢ`, the positive is `pᵢ`
- All other `pⱼ` (j ≠ i) are treated as negatives for `aᵢ`
- This gives B−1 negatives per anchor at zero extra compute cost

**The catch:** negative quality is bounded by what's in your batch. If your batch is 256 random examples, most negatives are trivially easy (totally unrelated to the anchor). The model learns fast early, then plateaus — it never has to make hard distinctions.

---

### 2.4 Hard Negative Mining

> **Think of it like this:** Easy negatives are like asking a student to distinguish a dog from a refrigerator. Hard negatives are like asking them to distinguish a Labrador from a Golden Retriever. The hard ones force real learning.

**Hard negatives** are examples that are lexically or topically close to the anchor but semantically wrong.

Example:
- Anchor: "Python tutorial for beginners"
- Easy negative: "Recipe for chocolate cake" (clearly unrelated)
- Hard negative: "Advanced Python memory management" (same topic, wrong level/intent)

#### Two Flavours of Hard Negative Mining

**Static hard negative mining** (offline):
1. For each anchor, use BM25 (keyword search) or a weak embedding model to retrieve top-k candidates
2. Filter out true positives
3. The survivors are your hard negatives — baked into the dataset before training starts

**Dynamic/online hard negative mining** (during training):
1. Every N steps, re-embed the corpus with the *current* model checkpoint
2. Mine new hard negatives based on what the current model finds confusingly similar
3. Hardness evolves as the model improves — you always train on the hardest examples the current model struggles with

This is what state-of-the-art models (BGE, E5, GTE) do and is a key reason they outperform SBERT on benchmarks.

#### The False Negative Trap

> **Gotcha for interviews:** "More negatives is always better" — FALSE.

If your mining is imperfect, some "hard negatives" are actually valid positives that were mislabelled. Training on these actively hurts performance — the model is penalised for getting the right answer. Past a certain point, negative quality matters more than negative count.

---

## 3. Bi-Encoders vs Cross-Encoders

> This is the single most-tested concept in RAG interviews. Know it cold.

### The Core Trade-off

| | Bi-encoder | Cross-encoder |
|---|---|---|
| **Architecture** | Encode query and document *separately* → vectors → cosine similarity | Concatenate query + document → joint model → single relevance score |
| **Cross-attention** | None between query and doc tokens | Full cross-attention between all query and doc tokens |
| **Pre-computation** | ✅ Documents can be embedded once, stored, retrieved instantly | ❌ Must run full model for every (query, doc) pair — no pre-computation |
| **Speed** | O(1) per query after indexing | O(n) — n forward passes per query |
| **Accuracy** | Lower | Higher |
| **Use case** | First-stage retrieval at scale (millions of docs) | Second-stage reranking of a small shortlist (top 50–100) |

### Why Bi-Encoders Are Fast But Imprecise

> **Think of it like this:** A bi-encoder is like two people in separate rooms writing down their key points on index cards, then someone outside compares the cards. They can't interrupt each other or ask follow-up questions — they just compare summaries.

Because query and document are encoded *independently*, there's zero interaction between their tokens during encoding. The model can't notice that "Python" in the query means the programming language while "Python" in the document refers to a snake. It compresses all meaning into a single vector before comparison.

### Why Cross-Encoders Are Accurate But Slow

> **Think of it like this:** A cross-encoder is like two people in the same room, reading each other's text together and having a full conversation. They can catch every nuance — but you can only afford this for a few candidates.

Cross-encoders read the query and document tokens together with full self-attention. The word "bank" in the query can attend to "river" in the document and correctly disambiguate the meaning. But this requires a full forward pass per pair — with 10M documents, that's 10M forward passes per query, typically 10–50ms each. Completely infeasible.

### Late Interaction: ColBERT (The Middle Ground — Name This in Interviews)

ColBERT is worth naming proactively — it signals depth beyond the binary framing.

**Idea:** Encode query tokens and document tokens separately (so doc embeddings are precomputable, like a bi-encoder) but compute a fine-grained **MaxSim** interaction at query time:

```
score(q, d) = Σⱼ max_i (Eqⱼ · Edᵢ)
```

For each query token j, find the most similar document token i. Sum these per-token maximum similarities. This gives token-level matching quality approaching cross-encoder accuracy, while still allowing document pre-computation.

**Trade-off:** Storage cost. Instead of one d-dim vector per document, you store one vector per *token* per document. A 200-token document at 128-dim needs 200×128=25,600 floats instead of 128. Index size goes up 100–200×.

### The Two-Stage Pipeline (Why RAG Works This Way)

```
Query
  │
  ▼
[Bi-encoder retrieval]   ← Fast, approximate, recall-focused
  │  Returns top-k docs (e.g. k=100)
  ▼
[Cross-encoder reranking]  ← Slow, precise, precision-focused
  │  Returns reranked top-m docs (e.g. m=5)
  ▼
LLM generation
```

This architecture exists because the two stages have complementary failure modes:
- Bi-encoder: high recall, lower precision (returns relevant docs but also irrelevant ones)
- Cross-encoder: high precision, but can't scale to full corpus

Together they give you scalability *and* quality.

---

## 4. Model Families

### The Landscape

**Sentence-BERT (SBERT, 2019)** — The paper that started it all. Fine-tuned BERT with siamese/triplet network structure. Now a baseline, but historically important because it demonstrated the geometry-fixing power of contrastive fine-tuning. Still used in lightweight/embedded applications.

**OpenAI `text-embedding-3-small` / `3-large`** — Proprietary. Strong general-purpose performance. Key feature: Matryoshka training (see section 6), which lets you shrink dimension at inference time without retraining. The "-3" generation was a significant quality jump over ada-002.

**Cohere `embed-v3`** — Strong multilingual support (100+ languages). Distinctive feature: explicit `input_type` parameter (`search_document` vs `search_query` vs `classification`). This is a productised implementation of asymmetric encoding — see below.

**BGE (BAAI General Embedding) / E5 (Microsoft) / GTE (Alibaba)** — Open-source, consistently near the top of MTEB (Massive Text Embedding Benchmark). All use the same recipe: large-scale weakly-supervised contrastive pretraining on web-scale text pairs + supervised fine-tuning on labelled retrieval datasets + dynamic hard negative mining. Interchangeable in many pipelines.

**Instructor** — Prompt-based embeddings. You prepend a natural-language task description before encoding: `"Represent this sentence for retrieval:"` or `"Represent this sentence for clustering:"`. One model, multiple geometry regimes. Useful when you don't want to maintain multiple models.

### Asymmetric Encoding — A Subtlety Worth Raising Proactively

Queries and documents are structurally different:
- A query: short (5–15 words), vague, underspecified, often phrased as a question
- A document: long (hundreds of words), information-dense, declarative

Forcing both through the same encoder with the same representation regime is suboptimal. Asymmetric encoding addresses this in two ways:

1. **Separate encoders**: train a query encoder and a document encoder independently (like DPR — Dense Passage Retrieval)
2. **Instruction prefixes**: use different prefixes for query vs document encoding with a prompted model ("Represent this question for retrieval:" vs "Represent this document for retrieval:")

The intuition: what makes a good query representation (capturing *intent*) is different from what makes a good document representation (capturing *content*).

---

## 5. Similarity Metrics

### The Three Metrics and When to Use Each

| Metric | Formula | Magnitude sensitive? | When to use |
|--------|---------|----------------------|-------------|
| Cosine similarity | `(a·b) / (‖a‖ · ‖b‖)` | No — divides by both magnitudes | Default for text; direction is what matters |
| Dot product | `a·b` | Yes | Only if vectors are pre-normalised to unit length |
| Euclidean / L2 | `‖a−b‖` | Yes | Metric learning, image embeddings, rarely for text |

### The Critical Relationship Between Them

For unit-norm vectors (‖a‖ = ‖b‖ = 1):

```
dot product = cosine similarity     (since denominator = 1×1 = 1)

L2 distance = √(2 − 2·cosine)      (algebraically: ‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b = 2 − 2cos)
```

This means: if your vectors are unit-normalised, all three metrics give equivalent rankings. The differences only matter when vectors have different magnitudes.

### The Production Bug Class: Magnitude Bias

> **This is a real bug that ships in production. Mention it unprompted.**

**Scenario:** You build a FAISS index with `IndexFlatIP` (inner product / dot product search). You embed documents with an embedding model that does NOT output unit-norm vectors. You query it.

**What happens:** Dot product = `‖a‖ × ‖b‖ × cos(angle)`. Documents with large vector magnitudes (often longer, information-dense documents) systematically score higher than shorter documents — regardless of actual relevance. You've built a popularity bias into your retrieval system without intending to.

**Fix:** Either normalise all vectors before indexing (call `faiss.normalize_L2(vectors)`) or use `IndexFlatL2` or cosine-equivalent indexing. Always check whether your embedding model's output is pre-normalised.

### Numerical Example of Magnitude Bias

Suppose:
- Query vector q = [0.6, 0.8] (unit norm: √(0.36+0.64) = 1.0)
- Doc A: a = [0.6, 0.8] (unit norm, perfect directional match)
- Doc B: b = [1.2, 1.6] (same direction as A, but 2× magnitude)

```
Cosine(q, a) = 1.0   Cosine(q, b) = 1.0   → correctly tied
Dot(q, a)    = 0.48+0.64 = 1.12
Dot(q, b)    = 0.72+1.28 = 2.00            → Doc B ranked higher! Same direction but penalised
```

Doc B would be retrieved over Doc A purely due to magnitude, not relevance.

---

## 6. Matryoshka Embeddings

### The Intuition

> **Think of it like this:** Russian Matryoshka dolls nest inside each other — the smallest is complete, the medium contains the small one, the largest contains all of them. A Matryoshka embedding works the same way: the first 64 dimensions are a valid embedding on their own. Dimensions 1–128 are a better valid embedding. All 768 are the best.

### How It's Trained

Standard embeddings are trained so all d dimensions are equally important — you can't drop any without significant quality loss. Matryoshka training adds a loss that sums contrastive loss over multiple prefix lengths simultaneously:

```
L_total = L(full, d=768) + L(prefix, d=256) + L(prefix, d=128) + L(prefix, d=64)
```

This forces the model to pack the most important information into the first dimensions, because the loss at d=64 will penalise it if the first 64 dimensions aren't informative on their own.

### Why It Matters for RAG

| Use case | Dimension | Trade-off |
|----------|-----------|-----------|
| First-pass ANN retrieval (millions of docs) | 128 | Faster, smaller index, slight recall drop |
| Reranking top-100 | 768 | Slower, larger, best quality |
| Storage-constrained edge deployment | 64 | Smallest, still meaningful |

A single embedding model can now serve multiple speed/quality operating points **without retraining or re-embedding your corpus**. This is a significant engineering win.

### The Practical Payoff

OpenAI's `text-embedding-3-small` is Matryoshka-trained. If you're using it and storing 1536-dim vectors, you could truncate to 256 dims and likely see only 2–5% quality drop with 6× the index compression and search speedup.

---

## 7. Domain Adaptation

### Why Off-the-Shelf Embeddings Underperform in Production

General-purpose embedding models are trained on web-scale text (Common Crawl, Wikipedia, MS MARCO, Stack Overflow). They work well for general queries. They systematically fail on:

**Specialised vocabulary:** "MI" in a general corpus means Michigan (common) or Mission Impossible. In a cardiology codebase, it means myocardial infarction. The model assigns the wrong cluster, so cardiology documents never retrieve correctly for cardiology queries.

**Short, acronym-heavy text:** Internal company communication ("Q3 OKR sync re: GTM motion for SMB segment") is almost absent from web pretraining data. The model produces garbage embeddings for these.

**Domain-specific similarity:** Two legal contracts might be "similar" because they share a clause structure around limitation-of-liability, even if they're about completely different industries. A general embedder has no signal for structural similarity; it optimises for topical similarity.

### Fixes (Ordered by Effort and Payoff)

**Option 1: Instruction prefixes (zero effort)**

If using Instructor or similar:
```
"Represent this medical record for retrieval of similar diagnoses: " + document
```
Can shift quality meaningfully for moderate domain gaps. Free. Try this first.

**Option 2: Fine-tune on in-domain pairs (moderate effort, high payoff)**

Collect even 500–5,000 labelled (query, relevant document) pairs from:
- Real user search logs + click data
- Manual annotation
- Synthetic pairs generated by an LLM

Fine-tune with contrastive loss on top of a general-purpose backbone. Even a small in-domain dataset consistently outperforms a large general model in the target domain. The backbone already "knows" language; you're just adjusting the geometric space to match domain similarity.

**Option 3: Continued pretraining + fine-tune (high effort, maximum payoff)**

For extreme domain gaps (biomedical, legal, financial code):
1. Take a general backbone
2. Continue MLM/CLM pretraining on a large in-domain corpus (acquire vocabulary)
3. Fine-tune with contrastive loss on in-domain pairs

This is how BioBERT, LegalBERT, and CodeBERT were created. Significant compute investment; justified when the vocabulary gap is severe.

**Option 4: Hybrid retrieval as a cheap mitigation**

If you can't fine-tune, combine BM25 (keyword search) with dense retrieval and merge results. BM25 is inherently good at rare/exact term matching because it's purely lexical. Dense retrieval handles semantic variation. Together they cover each other's weaknesses. This is the RRF (Reciprocal Rank Fusion) pattern common in production RAG stacks.

---

## Interview Q&A Drill

Work through these without looking at the answers. Cover the right column, answer out loud, then check.

---

**Q: Why not just use a cross-encoder for everything if it's more accurate?**

A: Cost. A cross-encoder requires a full forward pass per (query, document) pair with no pre-computation possible. For a 10M-document corpus, that's 10M forward passes per query — at even 5ms per pass, that's 14 hours of compute per query. Bi-encoders allow pre-computing and storing all document embeddings; at query time you only embed the query once and run an ANN search, which takes milliseconds. Cross-encoders are reserved for reranking a small shortlist (top 50–100) that bi-encoder already narrowed down.

---

**Q: Your retrieval quality is bad on queries with rare technical acronyms. What's your first hypothesis?**

A: Dense embedding coverage gap. The hypothesis: the embedding model was trained on web-scale general text; rare domain acronyms appear rarely or with different meanings, so their embeddings are poorly positioned in the vector space.

Verification steps:
1. Separate your eval set into acronym-heavy queries vs natural-language queries. Compare hit rate. If acronym queries consistently underperform, confirms the hypothesis.
2. Run BM25 (pure keyword) on the same queries. If BM25 outperforms dense retrieval on acronym queries, confirms a vocabulary gap, not a chunking or indexing issue.

Fix: either hybrid retrieval (BM25 + dense) or fine-tune embeddings on in-domain data with hard negatives that include domain terminology.

---

**Q: What's the difference between in-batch negatives and hard negative mining, and why use both?**

A: In-batch negatives are free (other examples in the same batch serve as negatives) and give broad coverage — the model learns to separate obviously unrelated content. But they're usually "easy" — the model plateaus after learning basic distinctions. Hard negatives are topically/lexically close but semantically wrong, forcing finer-grained discrimination near the decision boundary where real retrieval errors concentrate. Using both: in-batch negatives give the base training signal efficiently; hard negatives (mined offline via BM25 or the model's own previous checkpoint) sharpen quality on the hard cases that matter most in production.

---

**Q: Cosine similarity and dot product gave different top-k rankings on the same index. What happened?**

A: The embeddings are not unit-normalised. Dot product = cosine × ‖a‖ × ‖b‖. When vectors have different magnitudes, dot product biases retrieval toward high-magnitude documents (typically longer, information-dense documents) regardless of actual directional similarity. Fix: normalise all vectors to unit length before indexing (making dot product and cosine equivalent), or explicitly use a cosine-similarity index type. Always verify whether your embedding model outputs pre-normalised vectors.

---

**Q: What is the temperature τ in InfoNCE loss and what happens if you set it too high or too low?**

A: τ (temperature) controls the sharpness of the softmax distribution over negatives. Low τ (e.g. 0.01): the loss extremely harshly penalises near-miss negatives, demanding very sharp separation — harder training signal, can cause instability or over-fitting to the hardest negatives. High τ (e.g. 0.5): the loss treats all negatives roughly equally, producing a softer, weaker gradient signal — the model learns more slowly and may never learn fine-grained distinctions. Standard practice: start around τ = 0.05–0.07 and tune empirically. Many state-of-the-art models learn τ as a trainable parameter.

---

**Q: What is ColBERT and when would you use it over a standard bi-encoder?**

A: ColBERT is a late-interaction model that encodes query and document tokens separately (enabling document pre-computation, like a bi-encoder) but scores relevance via per-token MaxSim at query time — for each query token, find the most similar document token and sum these scores. This gives token-level interaction quality approaching cross-encoders while retaining the pre-computation benefit of bi-encoders. Use ColBERT when: bi-encoder quality is insufficient but cross-encoder latency is unacceptable, and you can afford the storage overhead (roughly 100–200× more storage per document vs a standard bi-encoder). ColBERT suits high-stakes retrieval pipelines (legal search, medical) where per-token matching precision matters.

---

**Q: How does Matryoshka training work and what's the practical benefit?**

A: Matryoshka training adds loss terms for multiple prefix lengths of the embedding vector during training (e.g. dimensions 1–64, 1–128, 1–256, full). This forces the model to pack the most information into the first dimensions. At inference time, you can truncate to any trained prefix length without retraining or re-embedding your corpus. Practical benefit: one model, multiple operating points. Use 128-dim for fast first-pass ANN retrieval and 768-dim for reranking, from the same model checkpoint. Reduces index size 4–12× with minimal quality loss at the lower dimensions.

---

## Key Gotchas Summary

| Gotcha | Correct understanding |
|--------|----------------------|
| "Raw BERT embeddings are good for retrieval" | False — anisotropy makes cosine similarity uninformative without contrastive fine-tuning |
| "More negatives always help" | False — past a point, false negatives (mislabelled positives) and easy negatives dilute signal |
| "Cosine and dot product are the same" | Only when vectors are unit-normalised; otherwise dot product adds magnitude bias |
| "Cross-encoders are always better" | Better quality but O(n) cost — infeasible at corpus scale |
| "One embedding model fits all domains" | Fails on specialised vocabulary and domain-specific similarity; fine-tuning usually required |
| "Matryoshka = just truncate any embedding" | No — standard embeddings degrade badly when truncated; only Matryoshka-trained models handle it well |

---

*Next: Module 2 — Chunking Strategies*
