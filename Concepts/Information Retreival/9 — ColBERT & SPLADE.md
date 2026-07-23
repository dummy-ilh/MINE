# Chapter 9 (Master Edition) — ColBERT & SPLADE, End to End

> This is the expanded, "nothing left out" version of Chapter 9. Everything in the original chapter is preserved and deepened: full architecture walk-throughs, training objectives, bigger worked examples, production pseudocode, failure-mode analysis, and an extended interview bank.

---

## 0. Where this sits on the map

```
                    accuracy →
BM25  ──────  bi-encoder  ──────  ColBERT / SPLADE  ──────  cross-encoder
 ↑                                                              ↑
fastest, sparse, exact-match only              slowest, no precompute, full attention
```

Four retrieval paradigms, one question each solves:

| Model | Question it answers |
|---|---|
| BM25 | Does this document contain the query's *exact words*, weighted sanely? |
| Bi-encoder | Are the *meanings* of query and document close in vector space? |
| ColBERT | Can I match meaning **token-by-token** without paying cross-encoder cost? |
| SPLADE | Can I get *learned* semantic expansion while keeping an *exact-match* inverted index? |

ColBERT and SPLADE are not competitors to each other so much as two different fixes to two different bi-encoder weaknesses:

- ColBERT fixes **compression loss** (one vector can't hold a whole document).
- SPLADE fixes **vocabulary mismatch under a sparse index** (exact-match systems don't know "car" ≈ "automobile").

---

## 1. ColBERT — full architecture, not just the formula

### 1.1 The pipeline, step by step

```
QUERY SIDE (at query time, must be fast):
  raw query "fast sorting"
    → tokenize: [CLS] fast sorting [Q] [MASK] [MASK] ... [SEP]   (padded to fixed length with [MASK])
    → BERT encoder
    → linear projection 768 → 128 per token
    → L2-normalize each token vector
    → Q ∈ R^(32 × 128)     (ColBERT pads/truncates queries to a fixed length, e.g. 32)

DOCUMENT SIDE (offline, can be slow — done once at indexing time):
  raw doc "quicksort runs quickly"
    → tokenize: [CLS] quicksort runs quickly [D] [SEP]
    → BERT encoder
    → linear projection 768 → 128 per token
    → L2-normalize each token vector
    → filter out punctuation tokens
    → D ∈ R^(|d| × 128)
    → STORE D on disk, indexed by document id

QUERY TIME:
  load Q (just computed) and D (loaded from disk for candidate docs)
  → MaxSim(Q, D) → score
  → rank candidates by score
```

Two details the base chapter glossed over, because they matter a lot in practice:

**Query augmentation with `[MASK]` tokens.** ColBERT pads every query to a fixed length (e.g. 32 tokens) using `[MASK]` tokens rather than the usual `[PAD]`. Because `[MASK]` tokens still get contextual BERT representations (unlike padding, which is typically ignored), this acts as *soft query expansion* — the model uses the mask positions to represent latent, unstated aspects of the query's intent. This is a real, empirically load-bearing trick, not a footnote.

**Punctuation filtering on the document side.** Document token vectors for punctuation (commas, periods) are dropped before storage. They add index size and clutter MaxSim with meaningless high-similarity matches (punctuation tends to match punctuation).

**The `[Q]` / `[D]` marker tokens.** ColBERT prepends a special `[Q]` token to queries and `[D]` token to documents right after `[CLS]`. This tells the *same* shared BERT encoder which mode it's in, since ColBERT uses one encoder for both sides (a Siamese/shared-weight setup), not two separate towers.

### 1.2 MaxSim, restated precisely

```
score(q, d) = Σ_{i=1}^{|q|}  max_{j=1}^{|d|}  ( Qᵢ · Dⱼ )
```

Read it as: *for every query token, find its single best-matching document token; add up those best matches.* This is asymmetric — you do NOT also do `max_i` for each `Dⱼ`. Only the query side gets the "for every token" treatment; the document side only contributes its best match per query token. This asymmetry is intentional: queries are short and every token should matter; documents are long and most tokens are irrelevant filler that should be allowed to lose every comparison.

### 1.3 Bigger worked example (5 query tokens, 8 doc tokens)

To see MaxSim do real work, we need enough tokens that not everything matches the "obvious" word.

```
query:    "cheap flights to Tokyo in December"
          tokens: [cheap, flights, to, Tokyo, December]   (stopword "in" dropped for brevity)

document: "budget airfare Japan winter discount seats available now"
          tokens: [budget, airfare, Japan, winter, discount, seats, available, now]
```

Toy 2-dim embeddings (in real ColBERT these are 128-dim; 2-dim keeps the arithmetic visible):

```
Q[cheap]    = [0.9, 0.1]      D[budget]     = [0.85, 0.15]
Q[flights]  = [0.2, 0.9]      D[airfare]    = [0.25, 0.85]
Q[to]       = [0.1, 0.1]      D[Japan]      = [0.05, 0.30]
Q[Tokyo]    = [0.05, 0.35]    D[winter]     = [0.10, 0.60]
Q[December] = [0.10, 0.65]    D[discount]   = [0.80, 0.20]
                               D[seats]      = [0.30, 0.40]
                               D[available]  = [0.15, 0.15]
                               D[now]        = [0.05, 0.05]
```

**Step 1 — dot product of `Q[cheap]` against every document token** (repeat this full sweep for every query token in real ColBERT; shown once here for illustration):

```
Q[cheap]·D[budget]    = 0.9×0.85 + 0.1×0.15 = 0.765 + 0.015 = 0.780  ← max
Q[cheap]·D[airfare]   = 0.9×0.25 + 0.1×0.85 = 0.225 + 0.085 = 0.310
Q[cheap]·D[Japan]     = 0.9×0.05 + 0.1×0.30 = 0.045 + 0.030 = 0.075
Q[cheap]·D[winter]    = 0.9×0.10 + 0.1×0.60 = 0.090 + 0.060 = 0.150
Q[cheap]·D[discount]  = 0.9×0.80 + 0.1×0.20 = 0.720 + 0.020 = 0.740
Q[cheap]·D[seats]     = 0.9×0.30 + 0.1×0.40 = 0.270 + 0.040 = 0.310
Q[cheap]·D[available] = 0.9×0.15 + 0.1×0.15 = 0.135 + 0.015 = 0.150
Q[cheap]·D[now]       = 0.9×0.05 + 0.1×0.05 = 0.045 + 0.005 = 0.050

MaxSim(cheap) = 0.780   (matched → "budget")
```

**Step 2 — do the same sweep for the remaining 4 query tokens** (arithmetic omitted for space, results only):

```
MaxSim(flights)  = 0.780   (matched → "airfare")
MaxSim(to)       = 0.130   (matched → "winter", weakly — "to" is near-meaningless, correctly gets a low score)
MaxSim(Tokyo)    = 0.215   (matched → "Japan" — geographic association, imperfect but real)
MaxSim(December) = 0.455   (matched → "winter" — seasonal association)
```

**Step 3 — sum**

```
score(q, d) = 0.780 + 0.780 + 0.130 + 0.215 + 0.455 = 2.360
```

**What this demonstrates that the small example in the base chapter couldn't:**
- Filler tokens ("to") correctly contribute almost nothing — MaxSim doesn't need a stopword list; low-information query tokens just fail to find a strong match anywhere and drag the score down proportionally, not catastrophically.
- "Tokyo" → "Japan" and "December" → "winter" are matches with **no lexical overlap at all** — this is the semantic power a bi-encoder would also have, but here it's localized to specific token pairs instead of blurred across the whole document.
- If you ran the same document against a query like `"expensive luxury suite July"`, tokens like "expensive" would find their best (weak) match against "discount" — a genuine near-miss the model should score low. This is why MaxSim needs training (next section) — with an *untrained* projection, "expensive" and "discount" might spuriously look similar since both are price-related tokens without direction (cheap vs. expensive). Training teaches the direction, not just the topic.

### 1.4 How ColBERT is trained

MaxSim is a scoring function, not a training signal by itself — the token embeddings need to be trained so that MaxSim actually correlates with relevance. Training uses the standard dense-retrieval recipe:

```
loss = triplet / pairwise softmax loss over (query, positive doc, negative doc) triples

for a training example (q, d⁺, d⁻):
  s⁺ = MaxSim(q, d⁺)     score against a KNOWN relevant document
  s⁻ = MaxSim(q, d⁻)     score against a KNOWN irrelevant / hard-negative document

  loss = -log( exp(s⁺) / (exp(s⁺) + exp(s⁻)) )     ← pairwise softmax (cross-entropy over 2 options)
```

This pushes `s⁺` up and `s⁻` down through backprop into the BERT encoder and the 768→128 projection. **Hard negatives matter enormously** — negatives that are lexically similar but not relevant (mined via BM25 or a prior-round bi-encoder) teach the model the fine-grained distinctions MaxSim is supposed to capture; random negatives are too easy and produce a weak model.

### 1.5 ColBERT v2: fixing the storage problem

The original chapter flagged storage as ColBERT's main weakness (16× a bi-encoder index). ColBERT v2's fix, in full:

```
Step 1 — Cluster all document token vectors in the corpus into K centroids (e.g. K = 2¹⁶ via k-means)
Step 2 — For each document token vector, store only:
           (a) the ID of its nearest centroid (a few bits)
           (b) a quantized RESIDUAL = (token_vector − centroid_vector), compressed to 1–2 bits per dimension
Step 3 — At query time, reconstruct an approximate token vector as: centroid + dequantized residual
Step 4 — Run MaxSim against these approximate reconstructions
```

```
storage comparison (1M docs, 100 tokens/doc, 128-dim):

ColBERT v1 (float32, no compression):  1M × 100 × 128 × 4 bytes  = 51.2 GB
ColBERT v2 (centroid id + 2-bit residual):   ≈ 1M × 100 × (16 bits + 128×2 bits) / 8  ≈ 5–8 GB   (roughly 6–10× smaller)
```

This is a **lossy** compression — reconstructed vectors are approximate — but empirically the accuracy loss is small because MaxSim only needs relative ordering to be preserved, not exact values.

### 1.6 Indexing and serving pseudocode

```python
# ---- OFFLINE INDEXING ----
doc_index = {}  # doc_id -> token embedding matrix (or centroid-compressed form)

for doc_id, text in corpus.items():
    tokens = tokenize_with_D_marker(text)
    token_vecs = colbert_encoder(tokens)          # [num_tokens, 128], L2-normalized
    token_vecs = drop_punctuation_vectors(token_vecs)
    doc_index[doc_id] = compress(token_vecs)      # v2: centroid id + residual

# ---- QUERY TIME ----
def search(query, top_k=10, candidate_k=1000):
    q_tokens = tokenize_with_Q_marker_and_mask_padding(query)
    Q = colbert_encoder(q_tokens)                 # [32, 128], L2-normalized

    # Stage 1: cheap ANN candidate generation over centroids (NOT full MaxSim yet)
    candidates = centroid_ann_lookup(Q, doc_index, k=candidate_k)

    # Stage 2: exact MaxSim re-scoring, only on the candidate set
    scores = {}
    for doc_id in candidates:
        D = decompress(doc_index[doc_id])
        scores[doc_id] = maxsim(Q, D)

    return top_k_by_score(scores, top_k)
```

Note the two-stage structure: even ColBERT itself doesn't run full MaxSim against every document in the corpus — it first uses the centroid structure to prune to a candidate set, then does exact MaxSim only on those candidates. This is essential context missing from a formula-only treatment: **ColBERT is its own mini retrieval pipeline (coarse-then-fine), not a single flat scoring pass.**

---

## 2. SPLADE — full architecture, not just the formula

### 2.1 The pipeline, step by step

```
raw text "the dog runs fast"
  → tokenize → [CLS] the dog runs fast [SEP]
  → BERT encoder → contextual token representations, one per position
  → MLM (masked-language-model) head → for EACH token position, a logit over the ENTIRE vocabulary
      (this is the same head BERT uses for masked-token pretraining — SPLADE repurposes it)
  → per-position logits transformed: log(1 + ReLU(logit))
  → MAX-POOL (or sum-pool, depending on variant) across token positions → one vector over the vocabulary
  → sparse vector w ∈ R^V  (V ≈ 30,000, almost all zero)
```

The critical architectural insight the base chapter's formula hides: **every token position votes on every vocabulary word**, not just its own word. The token "dog" doesn't only produce a weight for the vocabulary entry "dog" — its contextual representation, fed through the MLM head, produces a full 30,000-dim logit vector, and "canine," "pet," "animal" can all receive positive logits from the *context* around "dog," even without those words appearing anywhere in the text. Expansion isn't a separate step bolted on; it falls directly out of reusing the MLM head.

### 2.2 The full weight formula (with pooling made explicit)

```
For document d with tokens at positions j = 1..|d|:

  logit(j, t)  = MLM_head( BERT(d)_j )[t]        raw logit for vocab term t at position j

  w(t, d) = max_j  log(1 + ReLU(logit(j, t)))     ← max-pooling variant (most common)
         or
  w(t, d) = Σ_j    log(1 + ReLU(logit(j, t)))     ← sum-pooling variant (used in some SPLADE variants)
```

- `ReLU` clips negative logits to zero — a term the model actively thinks is *absent/contradicted* never gets a negative weight; it's just zero, same floor as "never mentioned."
- `log(1+x)` compresses large logits, exactly analogous to IDF's log — without it, a few very confident logits would dominate the dot product.
- Max-pooling means: a term only needs **one strong piece of contextual evidence anywhere in the document** to get a high weight — a single sentence about "the dog" is enough to activate "canine" for the whole document vector, regardless of document length. This is a deliberate design choice, different from TF-IDF's length-normalized averaging.

### 2.3 Why SPLADE vectors are actually sparse — FLOPS regularization

Nothing in the formula above *forces* sparsity — a badly trained model could give small positive weights to all 30,000 vocabulary terms, defeating the entire point (efficiency). SPLADE enforces sparsity explicitly during training with a regularization term added to the loss:

```
total_loss = ranking_loss(q, d⁺, d⁻)  +  λ_q · FLOPS(q)  +  λ_d · FLOPS(d)

FLOPS(x) = Σ_t  ā(t)²          where ā(t) = average weight of term t across a batch
```

`FLOPS` (named because it approximates the actual floating-point operations an inverted index would spend on a term) penalizes having many terms with non-trivial *average* weight across the batch — pushing the model toward vectors where only a small, batch-consistent set of terms carry real weight. `λ_q` and `λ_d` are typically scheduled to increase during training (start at 0, ramp up) so the model first learns to rank well, then is squeezed into sparsity without destroying what it already learned. Query-side regularization (`λ_q`) is usually weaker than document-side (`λ_d`) because queries are short and can tolerate slightly denser expansion; documents are indexed at scale and must stay sparse for storage/speed.

### 2.4 SPLADE variants (glossed over in the base chapter)

| Variant | Query encoder | Document encoder | Why |
|---|---|---|---|
| SPLADE (original) | full SPLADE (MLM head, expansion) | full SPLADE | symmetric, most accurate, most expensive to index |
| SPLADE-doc | simple/no expansion, or even raw BOW | full SPLADE | expansion only matters where recall is won — the document side |
| SPLADE-max / SPLADE-v2 | full SPLADE | full SPLADE, max-pooling + FLOPS reg | current standard, best accuracy/efficiency tradeoff |

Asymmetric variants (expansion only on the document side) exist because query-time latency budgets are tighter than indexing-time budgets — you can afford a slow, expansive encoder offline for every document once, but a query has to be encoded live, every time, under a latency SLA.

### 2.5 Worked example, extended: showing the training-driven suppression explicitly

Extending the base chapter's example with the *reason* each weight is what it is:

```
document: "the dog runs fast"
vocab:    [the, dog, canine, fast, quick, run, animal, slow]

w(the)    = 0.00   — MLM head learns "the" is uninformative in virtually every context; ReLU/training drives its logit negative → clipped to 0. No hand-coded stopword list needed, same self-suppression property as IDF, but LEARNED rather than derived from document frequency.
w(dog)    = 1.82   — appears literally; strong, direct evidence
w(canine) = 0.94   — does NOT appear; the contextual representation of "dog" activates a nearby vocabulary entry through the MLM head's learned associations
w(fast)   = 1.45   — appears literally
w(quick)  = 0.71   — expansion from "fast," same mechanism as canine/dog
w(run)    = 1.31   — "runs" stems/lemmatizes toward "run" in the subword+MLM representation
w(animal) = 0.43   — second-order expansion from "dog," weaker signal, smaller weight
w(slow)   = 0.00   — semantically OPPOSED to "fast" in context; ReLU clips any negative-leaning logit to zero, same floor as an absent/irrelevant term (SPLADE cannot express "negatively associated" — only "zero or positive")
```

That last line is a genuine limitation worth naming explicitly: **SPLADE has no way to penalize a document for contradicting a query term** — "slow" gets the same 0 weight whether the document is silent about speed or explicitly says "definitely not slow." A dot product with a zero entry contributes zero either way.

### 2.6 Indexing and serving pseudocode

```python
# ---- OFFLINE INDEXING (built on a standard inverted index, e.g. Lucene) ----
inverted_index = {}  # term -> list of (doc_id, weight)

for doc_id, text in corpus.items():
    sparse_vec = splade_encode(text)              # dict: {term: weight}, ~100-200 non-zero entries
    for term, weight in sparse_vec.items():
        inverted_index.setdefault(term, []).append((doc_id, weight))

# ---- QUERY TIME ----
def search(query, top_k=10):
    q_vec = splade_encode(query)                  # dict: {term: weight}
    scores = {}
    for term, q_weight in q_vec.items():
        for doc_id, d_weight in inverted_index.get(term, []):
            scores[doc_id] = scores.get(doc_id, 0) + q_weight * d_weight
    return top_k_by_score(scores, top_k)
```

Operationally this is byte-for-byte the same shape as a BM25 query loop — `term → postings list → accumulate weighted score`. This is SPLADE's headline practical advantage: **you can often swap it into an existing Lucene/Elasticsearch deployment by changing what gets written to the postings list, not the retrieval engine itself.**

---

## 3. Head-to-head: what actually differs at each stage of the request lifecycle

| Stage | BM25 | Bi-encoder | ColBERT | SPLADE | Cross-encoder |
|---|---|---|---|---|---|
| Indexing compute | trivial (counting) | 1 forward pass/doc | 1 forward pass/doc | 1 forward pass/doc (slower, MLM head over full vocab) | none |
| Indexing storage | postings, tiny | 1 dense vector/doc | all token vectors/doc | sparse postings, small | none |
| Index structure | inverted index | ANN (HNSW/FAISS/IVF) | custom (centroid+residual) | inverted index | none |
| Query encode cost | none (just parse) | 1 forward pass | 1 forward pass | 1 forward pass | N/A |
| Query-time scoring op | postings sum | 1 dot product/candidate | MaxSim over token pairs | sparse dot product | full [q;d] forward pass |
| Can prune with skip-lists? | yes | N/A (ANN handles pruning) | yes, via centroid stage | yes, same as BM25 | no |
| Handles vocabulary mismatch | no | yes | yes | yes (learned expansion) | yes |
| Handles exact/rare-term match | yes, natively | poor | good (token-level) | yes, natively | yes |
| Typical production role | first-stage retriever | first-stage retriever | re-ranker (rarely first-stage) | first-stage retriever (drop-in for BM25) | final re-ranker |

---

## 4. Why they work / why they fail (extended)

### ColBERT — additional nuance beyond the base chapter

**Extra "why it works":**
- The coarse-then-fine two-stage design (Section 1.6) means ColBERT in production is never literally "MaxSim against the whole corpus" — the centroid stage does the heavy lifting of candidate generation, so raw query latency is much better than the naive `|q|×|d|×N_docs` estimate suggests.
- Query-side `[MASK]` augmentation (Section 1.1) gives ColBERT a cheap form of query expansion for free, on top of MaxSim's token-level matching.

**Extra "why it fails":**
- MaxSim is not learned to penalize *contradiction* any better than a bi-encoder — a document token vector for "not fast" and "fast" can still look similar if the projection hasn't specifically been trained on negation-heavy hard negatives.
- The custom index format (centroids + residuals) is real engineering surface area: it is not a drop-in replacement for FAISS or Lucene the way SPLADE is for an inverted index. Most teams adopting ColBERT use the official Stanford `ColBERT`/`RAGatouille` libraries rather than building the index layer themselves.

### SPLADE — additional nuance beyond the base chapter

**Extra "why it works":**
- FLOPS regularization is what makes SPLADE viable — without it, "sparse" would be a lie and the whole storage/speed advantage over dense retrieval would evaporate.
- Reusing the MLM pretraining head means SPLADE inherits a huge amount of "free" linguistic knowledge already baked into BERT's pretraining, rather than learning expansion from scratch on a (usually much smaller) relevance-labeled dataset.

**Extra "why it fails":**
- Cannot express negative/contradictory evidence (Section 2.5) — every term is either "not mentioned" (0) or "positively associated" (>0); there's no representation for "actively ruled out."
- FLOPS regularization strength (`λ_q`, `λ_d`) is a sensitive hyperparameter — too strong and the model collapses toward a near-BM25 vector with little expansion benefit; too weak and index size/query latency balloon back toward dense-retrieval territory, undermining the entire pitch.

---

## 5. Production decision guide

```
Do you need a re-ranker over a small candidate set (≤500 docs) and have accuracy headroom to spend?
   → ColBERT (or a cross-encoder if you can afford the extra latency)

Do you need a first-stage retriever that's a near drop-in for your existing BM25/Lucene deployment,
but with better recall on vocabulary mismatch?
   → SPLADE

Do you need the cheapest possible first-stage retrieval with strong general semantic recall,
and you're fine maintaining an ANN index (FAISS/HNSW)?
   → Bi-encoder

Do you need the best possible final-stage precision and can tolerate full-forward-pass latency
on a small final candidate set (≤50-100 docs)?
   → Cross-encoder

Realistic production pipeline combining all four:
   BM25  +  bi-encoder ANN  →  merge via RRF (Chapter 6)  →  top 200
        →  SPLADE re-score (optional, if not already first-stage)  →  top 100
        →  ColBERT re-rank  →  top 20
        →  cross-encoder final re-rank  →  top 10 shown to user
```

---

## 6. The one thing to remember

ColBERT refuses to compress a document into one vector — it keeps one vector per token and lets MaxSim find, for every query token, that token's single best match in the document, so long or topically mixed documents don't get blurred into an average. SPLADE refuses to abandon the inverted index — it keeps a sparse, term-indexed vector like BM25, but lets a neural network (via the reused MLM head, kept sparse through FLOPS regularization) decide the weights and add learned synonym expansion, so exact-match infrastructure gains semantic reach without an ANN index.

---

## 7. Full formula reference

| Formula | Meaning |
|---|---|
| `Q = BERT_proj(q) ∈ R^(\|q\|×128)`, mask-padded | ColBERT query token matrix |
| `D = BERT_proj(d) ∈ R^(\|d\|×128)`, punctuation-filtered | ColBERT document token matrix |
| `score(q,d) = Σᵢ max_j (Qᵢ·Dⱼ)` | ColBERT MaxSim |
| `loss = -log( e^{s⁺} / (e^{s⁺}+e^{s⁻}) )` | ColBERT pairwise training loss |
| `storage_v1 = N × \|d\| × 128 × 4 bytes` | ColBERT v1 raw storage |
| `storage_v2 ≈ N × \|d\| × (16 + 128×2) bits` | ColBERT v2 centroid+residual storage |
| `logit(j,t) = MLM_head(BERT(d)ⱼ)[t]` | SPLADE raw per-position, per-term logit |
| `w(t,d) = max_j log(1+ReLU(logit(j,t)))` | SPLADE term weight (max-pool variant) |
| `FLOPS(x) = Σ_t ā(t)²` | SPLADE sparsity regularizer |
| `total_loss = rank_loss + λ_q·FLOPS(q) + λ_d·FLOPS(d)` | SPLADE full training objective |
| `score(q,d) = Σ_t q(t)×d(t)` | SPLADE retrieval (sparse dot product) |

---

## 8. Extended interview Q&A

*(Original 5 questions preserved; new questions added below them.)*

**Q1. What problem does ColBERT solve that a bi-encoder cannot?**
*(as in base chapter — compression loss; MaxSim lets tokens match tokens instead of one vector matching one vector.)*

**Q2. Explain MaxSim. Why sum the maxes instead of taking the max of all dot products?**
*(as in base chapter — every query token must be accounted for, not just the single best pair overall.)*

**Q3. How does SPLADE differ from BM25 if both produce sparse vectors and use an inverted index?**
*(as in base chapter — learned, context-aware, expansion-capable weights vs. a fixed formula over literal term counts.)*

**Q4. Where would you place ColBERT in a production retrieval pipeline?**
*(as in base chapter — second-stage re-ranker over a few hundred candidates, not a first-stage retriever over the full corpus.)*

**Q5. A team proposes replacing your BM25 + bi-encoder hybrid with SPLADE alone. What are the tradeoffs?**
*(as in base chapter — operational simplicity and exact-match strength vs. shallower semantic generalization than a dense bi-encoder for purely conceptual or cross-lingual queries.)*

**Q6. Why does ColBERT pad queries with `[MASK]` tokens instead of `[PAD]`?**

Standard padding tokens are typically masked out of attention and contribute nothing. ColBERT deliberately keeps the padded positions as `[MASK]` because BERT still produces contextual embeddings for masked positions — and those embeddings, sitting in the query's own contextual field, act as soft slots the model can use to represent implied aspects of the query that weren't explicitly typed. Empirically, removing this "query augmentation" measurably hurts ColBERT's recall, which is why it's treated as a core part of the architecture rather than an incidental detail.

**Q7. What does the FLOPS regularizer actually penalize, mechanically, and why does the penalty use squared average weights rather than a simple count of non-zero entries?**

A raw non-zero count is not differentiable, so it can't be optimized with gradient descent directly. FLOPS instead penalizes the sum of squared *average* weights per term across a training batch. Squaring means a term that's weighted highly and consistently across many documents in the batch is penalized much more than a term with the same total weight spread thinly — pushing the model to concentrate its "budget" onto genuinely important terms per document rather than giving mild positive weight to everything. The averaging across the batch (rather than per-document) also means the penalty reflects corpus-wide term usage patterns, discouraging systematic over-use of common expansion terms across many documents at once.

**Q8. Both ColBERT and SPLADE claim to be "efficient." Efficient compared to what, and are they efficient in the same sense?**

They're efficient relative to a cross-encoder (no full-attention forward pass per document at query time), but the *kind* of efficiency differs. SPLADE is efficient in the BM25 sense: retrieval is a sparse dot product over an inverted index, the same computational shape as BM25, so it inherits decades of inverted-index engineering (skip lists, block-max compression, etc.) essentially for free. ColBERT is efficient only relative to a cross-encoder, not relative to a bi-encoder or SPLADE — it still requires either scanning `|q|×|d|` token pairs per candidate document or a custom ANN structure (centroids/residuals) to avoid that, and its storage footprint remains larger than either a bi-encoder or SPLADE. Calling both "efficient" without qualification hides that ColBERT is a precision/storage tradeoff aimed at re-ranking, while SPLADE is a first-stage-retrieval-shaped efficiency story.

**Q9. Can SPLADE and ColBERT be combined in the same pipeline, and does that combination make sense?**

Yes, and it's a common production pattern: use SPLADE as the (or a) first-stage retriever, since it slots into existing inverted-index infrastructure and provides strong recall including on vocabulary-mismatch queries; then use ColBERT as a re-ranker over SPLADE's (or a hybrid's) top-N candidates, since ColBERT's token-level MaxSim adds precision that a single sparse dot product per document can't fully capture. This combination plays to each method's designed role — SPLADE for recall at scale over an inverted index, ColBERT for precision over a small candidate set — rather than asking either one to do the other's job.

**Q10. What's a concrete failure case for SPLADE that BM25 would also fail, and one that a bi-encoder would also fail?**

SPLADE and BM25 both struggle with truly out-of-vocabulary tokens the model has never seen any context for — a brand-new product SKU or a coined term with zero training-time exposure produces no reliable signal in either system, since SPLADE's expansion is learned from patterns in training data and BM25 has no expansion mechanism at all. SPLADE and a bi-encoder both struggle with the negation/contradiction case from Section 2.5 and 4 above — a document explicitly stating a term does NOT apply gets treated much like a document that never mentions the term at all, because neither architecture has a clean mechanism for representing "actively contradicted" as distinct from "absent."

---
