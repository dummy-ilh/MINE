# RAG Module 1 — Embeddings

---

## 1.1 What an embedding actually is

An embedding is a learned mapping `f: X → R^d` from a discrete object (text, image, code) to a dense, fixed-length vector, such that **geometric distance in R^d approximates semantic similarity in the original space**.

Key properties an embedding model is trained to have:
- **Locality**: semantically similar inputs → nearby vectors
- **Isotropy** (ideally): the vector space isn't collapsed into a narrow cone — otherwise cosine similarity becomes uninformative (a real, common failure mode in raw transformer embeddings before contrastive fine-tuning)
- **Fixed dimensionality**: regardless of input length, output is a single d-dim vector (usually via pooling — mean pooling, CLS token, or last-token pooling depending on model)

**Pitfall to flag in interviews**: raw last-hidden-state outputs of an LM (e.g. plain BERT `[CLS]`) are *not* good similarity embeddings out of the box — anisotropy in transformer representation space means untrained cosine similarity is dominated by frequency/norm artifacts, not meaning. This is *why* contrastive fine-tuning (Sentence-BERT and onward) was necessary — it's not just "use any transformer."

---

## 1.2 Training objectives

### Contrastive loss (the workhorse)
Given an anchor `a`, a positive `p` (semantically matching), and negatives `n_1...n_k`:

```
L = -log( exp(sim(a,p)/τ) / Σ_i exp(sim(a,n_i)/τ) )
```

This is **InfoNCE** — softmax over similarity scores, τ is a temperature that controls how sharply the loss penalizes near-miss negatives. Lower τ → sharper distinctions required → harder training signal.

### Triplet loss (predecessor to InfoNCE-style contrastive)
```
L = max(0, sim(a,n) - sim(a,p) + margin)
```
Pushes negative similarity below positive similarity by at least `margin`. Weaker signal than InfoNCE because it only considers one negative at a time rather than a full softmax over a batch — InfoNCE became dominant because it scales better with in-batch negatives.

### In-batch negatives
Cheapest and most common trick: within a training batch of (anchor, positive) pairs, treat every *other* example's positive as a negative for this anchor. Free negatives, no extra sampling — but negative "hardness" is bounded by whatever happens to be in the batch (random negatives are mostly easy negatives).

### Hard negative mining
Random in-batch negatives are usually too easy (embedding model learns to separate obviously-unrelated text, plateaus early). Hard negatives — text that's lexically/topically close but semantically wrong — force the model to learn finer distinctions.
- **Static hard negative mining**: precompute with BM25 or a weaker embedding model, mine top-k non-matching but lexically similar docs
- **Dynamic/online hard negative mining**: mine using the *model's own current embeddings* during training, refreshed periodically — this is what most modern top-performing embedders (BGE, E5, GTE) do
- **Cross-batch negatives / memory bank**: maintain a queue of embeddings from previous batches to increase the negative pool without blowing up batch size (borrowed from MoCo in vision contrastive learning)

**Interview trap**: "more negatives is always better" — not quite true past a point; too many easy negatives dilute signal, and mined negatives that are actually false negatives (near-duplicate positives mislabeled as negative) actively hurt training. Data quality of the negative set matters more than raw count past a threshold.

---

## 1.3 Bi-encoders vs cross-encoders

This distinction is **the single most-tested concept in RAG interviews** — know it cold.

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Architecture | Encode query and doc **separately** into vectors, compare via cosine/dot product | Concatenate query+doc, pass **jointly** through the model, output a single relevance score |
| Interaction | No cross-attention between query and doc tokens | Full cross-attention between query and doc tokens |
| Speed | O(1) per query after docs are pre-embedded — enables ANN search over millions of docs | O(n) — must run the full model for every candidate doc, no pre-computation possible |
| Accuracy | Lower — misses fine-grained token-level interaction | Higher — captures nuanced relevance signals |
| Use case | **First-stage retrieval** at scale (search a large corpus) | **Second-stage reranking** of a small candidate set (top-k from bi-encoder) |

This is *why* RAG pipelines are two-stage: bi-encoder for cheap large-scale recall, cross-encoder for expensive small-scale precision (this connects directly into Module 5 — Reranking).

**Late-interaction models (ColBERT)** are the middle ground: encode query and doc tokens separately (so doc embeddings are precomputable, like bi-encoders) but compute a fine-grained token-to-token MaxSim interaction at query time (closer to cross-encoder quality). Worth naming as the "third option" in interviews — shows depth beyond the binary framing.

---

## 1.4 Model families (know the landscape, not just names)

- **Sentence-BERT (SBERT)** — the original bi-encoder fine-tuning recipe (siamese/triplet network on top of BERT). Historically important, now a baseline.
- **OpenAI `text-embedding-3-*`** — proprietary, strong general-purpose, supports variable output dims via Matryoshka-style training (see 1.6).
- **Cohere `embed-v3`** — strong multilingual + supports "input type" flag (`search_document` vs `search_query`) — a good example of **asymmetric encoding** (see below).
- **BGE (BAAI General Embedding)**, **E5 (Microsoft)**, **GTE (Alibaba)** — open-source, top of MTEB leaderboard historically, trained with large-scale weakly-supervised contrastive pretraining + supervised fine-tuning + hard negative mining.
- **Instructor / task-specific prompted embeddings** — prepend a natural-language instruction ("Represent this sentence for retrieval:") before encoding, letting one model serve multiple embedding tasks (retrieval, clustering, classification) with different geometries.

**Asymmetric encoding** — a subtlety worth surfacing proactively: queries and documents are often *structurally different* (a query is short and underspecified, a doc is long and information-dense). Some models use **separate encoders (or separate instruction prefixes) for query vs document** rather than a single symmetric encoder — improves retrieval because it stops forcing "what is the capital of France?" and a 3-paragraph Wikipedia passage into the same representation regime.

---

## 1.5 Similarity metrics

| Metric | Formula | Notes |
|---|---|---|
| Cosine similarity | `(a·b)/(‖a‖‖b‖)` | Magnitude-invariant — only measures direction. Standard default. |
| Dot product | `a·b` | Magnitude-*sensitive*. Equivalent to cosine **only if vectors are pre-normalized** to unit length. |
| Euclidean (L2) distance | `‖a-b‖` | Sensitive to magnitude; monotonically related to cosine *only* for normalized vectors (`‖a-b‖² = 2 - 2·cos(a,b)` when ‖a‖=‖b‖=1). |

**Common bug to flag in interviews**: mixing an index built for dot-product ANN search (e.g. FAISS `IndexFlatIP`) with unnormalized embeddings silently biases retrieval toward *longer or higher-magnitude* documents, since dot product conflates "similar direction" with "large magnitude." Always know whether your embedding model's vectors are pretrained to unit norm, and match your index's metric accordingly. This is a real production bug class, not just theory — good to mention unprompted.

---

## 1.6 Matryoshka embeddings (variable-dimension)

Trained so that **truncating the vector to its first k dimensions still yields a valid, well-formed embedding** (rather than embeddings where all dimensions are equally load-bearing and truncation destroys the space).

- Trained via a loss that sums contrastive loss over multiple prefix lengths (e.g. 64, 128, 256, 768 dims) simultaneously, nested like Russian dolls (hence "Matryoshka")
- Lets you trade off storage/speed vs accuracy *at inference time* without retraining or re-embedding — e.g. use 128-dim for a fast first-pass ANN search, 768-dim for reranking
- Directly relevant to Module 3 (indexing) — smaller dims = smaller index = faster ANN search = the practical payoff

---

## 1.7 Domain adaptation — why off-the-shelf embeddings underperform

General-purpose embedders (trained on web-scale weakly supervised pairs — titles/bodies, Q&A forums, etc.) systematically underperform on:
- **Specialized vocabulary** (legal, medical, internal company jargon) — the model has never learned that "MI" means "myocardial infarction" not "Michigan" in this corpus
- **Short, acronym-heavy, or code-like text** — poor coverage in general pretraining data
- **Domain-specific notions of similarity** — e.g. in legal search, two contracts might be "similar" because they share a clause structure, not because they're topically about the same subject; a general embedder has no signal for that

**Fixes, in increasing order of effort:**
1. **Prompt/instruction engineering** with instruction-tuned embedders (cheapest, no training)
2. **Fine-tune on in-domain contrastive pairs** — even a few thousand labeled (query, relevant doc) pairs from real user logs or click data can meaningfully shift retrieval quality
3. **Continued pretraining** on domain corpus before contrastive fine-tuning (most expensive, used when domain vocabulary is extremely different, e.g. biomedical/code)
4. **Hybrid retrieval as a cheap mitigation** — lean on BM25/sparse retrieval to compensate for dense embedding weakness on rare/exact terms (foreshadows Module 4)

---

## Interview Q&A drill

**Q: Why not just use a cross-encoder for everything if it's more accurate?**
A: Cost — cross-encoders require a full forward pass per (query, doc) pair with no precomputation possible. For a corpus of 10M docs, that's 10M forward passes per query, infeasible at any real latency budget. Bi-encoders let you precompute doc embeddings once and do fast ANN lookup at query time; cross-encoders are reserved for reranking a small shortlist (top 50–100) that a bi-encoder already narrowed down.

**Q: Your retrieval quality is bad on queries with rare technical acronyms. What's your first hypothesis and how do you verify it?**
A: First hypothesis: dense embedding model has poor coverage of domain-specific rare tokens (acronyms get embedded close to unrelated common words due to weak/no signal during pretraining). Verify by comparing retrieval hit rate on acronym-heavy queries specifically vs. natural-language queries in your eval set, and/or checking if BM25 alone outperforms the dense retriever on that query subset — if sparse retrieval wins on that slice, it confirms a dense-embedding coverage gap rather than a chunking or indexing bug.

**Q: What's the difference between in-batch negatives and hard negative mining, and why would you use both?**
A: In-batch negatives are free (any other example's positive in the batch), but tend to be "easy" — trivially unrelated. Hard negatives are lexically/topically close but wrong, forcing finer-grained discrimination. Using both: in-batch negatives give broad coverage cheaply, hard negatives (mined via BM25 or the model's own prior checkpoint) sharpen the decision boundary near the hardest cases, which is usually where real-world retrieval errors concentrate.

**Q: Cosine similarity and dot product gave different top-k rankings on the same index. What happened?**
A: The embeddings are not unit-normalized, so dot product is conflating vector magnitude with directional similarity — likely biasing toward longer or more "information-dense" documents that happen to have larger vector norms. Fix: either normalize all vectors to unit length before indexing (making dot product and cosine equivalent) or explicitly use a cosine-similarity index type.

---

**Next up: Module 2 — Chunking strategies.** Say the word when ready.
