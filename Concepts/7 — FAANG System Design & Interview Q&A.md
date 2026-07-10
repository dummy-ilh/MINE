# Chapter 7 — FAANG System Design & Interview Q&A

## What is it?

This is the capstone chapter. Everything in Chapters 1–6 was building blocks. Now we put them together the way a FAANG interviewer expects — as a **system you design, justify, and defend under pressure.**

FAANG IR interviews come in three flavors:

1. **Conceptual** — "explain TF-IDF" or "what is NDCG" (Chapters 1–5 cover these)
2. **Comparative** — "when would you use BM25 over dense retrieval" (Chapter 6 covers this)
3. **System design** — "design a semantic search system for a 10 billion document corpus" (this chapter)

System design questions are the hardest because there is no single right answer. The interviewer is evaluating your **reasoning process** — how you break down ambiguity, identify tradeoffs, justify decisions, and handle follow-up challenges. This chapter gives you the framework and worked examples.

---

## The system design framework for IR

Every IR system design answer should walk through these six stages in order. Skipping stages is the most common mistake candidates make.

```
1. Clarify requirements       → what are we actually building?
2. Define the data            → corpus size, document type, update frequency
3. Choose retrieval strategy  → sparse, dense, hybrid — justify the choice
4. Design the index           → inverted index, ANN type, storage
5. Design the query pipeline  → latency budget, re-ranking, caching
6. Define success metrics     → how do you know it's working?
```

Interviewers will interrupt at any stage with follow-up challenges. Knowing the full framework means you can always orient yourself — "I've covered stages 1–3, let me move to the index design."

---

## Stage 1 — Clarify requirements

Never start designing immediately. Ask these questions first:

```
scale:        How many documents? How many queries per second (QPS)?
latency:      What is the acceptable response time? (p50, p99)
freshness:    How quickly must new documents be searchable?
query type:   Keyword search, semantic search, or both?
relevance:    Binary (relevant/not) or graded? Who labels it?
language:     Single language or multilingual?
explainability: Does the system need to justify its results?
```

**Why this matters in an interview:** It signals seniority. A junior engineer starts building immediately. A senior engineer spends 5 minutes asking the right questions because the answers change every design decision downstream.

---

## Stage 2 — Define the data

```
document type:    short (tweets, product titles) vs long (papers, legal docs)
corpus size:      thousands vs millions vs billions
update frequency: static vs append-only vs full CRUD
domain:           general web vs specialized (medical, legal, code)
language:         monolingual vs multilingual
```

These constrain your choices:

```
short docs + high QPS           → BM25 baseline, k₁ low
long docs + semantic queries    → dense retrieval, cross-encoder re-rank
billions of docs                → FAISS IVF over HNSW (memory constraint)
frequent updates                → segment-based inverted index, append-friendly ANN
specialized domain              → fine-tuned domain-specific embeddings
```

---

## Stage 3 — Choose retrieval strategy

Use this decision tree:

```
do you have labeled query-document pairs?
├── no  → start with BM25, collect labels, iterate
└── yes → how important is exact match?
          ├── critical (product IDs, names, codes) → hybrid (BM25 + dense)
          └── not critical (conceptual queries)    → dense + BM25 fallback

is latency under 100ms required?
├── yes → bi-encoder ANN only (no cross-encoder, or very small re-rank set)
└── no  → add cross-encoder re-ranker for top-k

is the domain specialized?
├── yes → fine-tune bi-encoder on domain data before deploying
└── no  → off-the-shelf embedding model (e5-large, BGE, OpenAI ada-002)
```

---

## Stage 4 — Design the index

### Inverted index (for BM25 component)

```
technology:    Elasticsearch / Lucene / custom
indexing:      segment-based (Chapter 1) — write to memory buffer,
               flush to immutable segments, merge periodically
tokenization:  language-specific tokenizer + stemmer + stopword list
storage:       SSD for segments, RAM for hot posting lists
update:        tombstone deletes + new segment inserts
```

### ANN index (for dense component)

```
< 10M docs:    HNSW (high recall, fits in RAM)
10M–1B docs:   FAISS IVF + PQ (memory-efficient, tunable nprobe)
> 1B docs:     FAISS IVF + PQ with sharding across multiple machines
               or ScaNN (Google's production system)

quantization:  float32 (full precision) → int8 or PQ (4–16× compression)
               recall@10 drops ~1–3% but memory drops 4–16×
```

### Storage sizing example

```
corpus: 100M documents
embedding dim: 768
float32:  100M × 768 × 4 bytes = 307 GB  ← needs ~300GB RAM for HNSW
int8:     100M × 768 × 1 byte  =  77 GB  ← 4× smaller, ~1% recall loss
PQ (64 bytes per vector): 100M × 64 = 6.4 GB  ← 48× smaller, ~3% recall loss
```

This is why quantization is not optional at scale — 300GB of RAM per machine is expensive. PQ at 6.4GB is deployable on a standard server.

---

## Stage 5 — Design the query pipeline

```
user query
    ↓
query processing (tokenize, normalize)          ~1ms
    ↓
[parallel]
BM25 lookup → top 1000                         ~5ms
dense encode → ANN search → top 1000           ~50ms
    ↓
RRF fusion → top 200                           ~2ms
    ↓
cross-encoder re-rank → top 10                 ~200ms
    ↓
business rules (freshness, diversity, A/B)     ~5ms
    ↓
return results                                 total: ~260ms
```

### Caching

```
query cache:   exact query string → cached result (TTL: minutes to hours)
               hit rate typically 20–40% for navigational queries

embedding cache: query → cached embedding vector
                 useful for repeated or near-duplicate queries

when NOT to cache: personalized results, breaking news, inventory-sensitive
                   (e-commerce stock levels change, cached results go stale)
```

### Handling QPS at scale

```
100 QPS:    single machine, BM25 + small HNSW
1,000 QPS:  horizontal scaling, load balancer → N retrieval servers
10,000 QPS: sharded index (each shard holds fraction of corpus),
            scatter-gather pattern (query all shards in parallel, merge results)
```

---

## Stage 6 — Define success metrics

Never end a system design answer without metrics. This is where many candidates drop points.

```
offline metrics (before deployment):
  NDCG@10     → overall ranking quality (graded relevance)
  MAP         → recall-aware ranking quality (binary relevance)
  MRR         → first-result quality (QA-style queries)
  Recall@1000 → first-stage retrieval ceiling
               (if true answer isn't in top 1000, re-ranker can't save you)

online metrics (after deployment, A/B test):
  CTR@1       → did users click the first result?
  Session abandonment rate → did users leave without clicking anything?
  Dwell time  → did users spend time on the result (signal of quality)?
  Reformulation rate → did users immediately rephrase their query?
                       (signal that the result was wrong)

the gap between offline and online:
  offline NDCG can improve while online CTR drops — this happens when
  the model retrieves "objectively" better documents that don't match
  user expectation or presentation format. Always A/B test.
```

---

## Worked system design — semantic search for a code repository (GitHub-style)

**Interviewer prompt:** "Design a search system for a code repository hosting 500 million code files. Users search with natural language queries like 'function that sorts a list in Python' or exact queries like 'def quicksort'."

### Step 1 — clarify requirements

```
scale:     500M code files, growing ~10M/month
QPS:       10,000 queries/second at peak
latency:   p99 < 500ms
freshness: new commits searchable within 5 minutes
query:     both natural language AND exact keyword/symbol
language:  code in 50+ programming languages
```

### Step 2 — define the data

```
document:  a code file or function-level chunk (~50–200 tokens)
           chunking at function level (not file level) improves precision
           metadata: language, repo, stars, last updated, license
short docs → lower k₁ in BM25 (~0.9), function-level chunking
updates:   10M new files/month = ~4 files/second → near-real-time indexing needed
```

### Step 3 — choose retrieval strategy

```
exact queries ("def quicksort", "import torch"):        BM25 essential
natural language ("function that sorts a list"):        dense essential
→ hybrid is mandatory, not optional

embedding model: fine-tune on code
  base model: CodeBERT or UniXcoder (pre-trained on code)
  fine-tune:  on (natural language query, code snippet) pairs
              using existing GitHub issues + linked code as training data
```

### Step 4 — design the index

```
BM25 component (Elasticsearch):
  tokenizer:  code-aware (split on camelCase, snake_case, operators)
              "quickSort" → ["quick", "sort", "quicksort"]
  fields:     function name (boosted 3×), comments (boosted 2×), body (1×)
  sharding:   50 shards across 50 machines (10M files per shard)
  updates:    new commits → Kafka queue → indexing workers → Elasticsearch

dense component (FAISS IVF + PQ):
  embeddings: 768-dim from fine-tuned CodeBERT
  storage:    500M × 64 bytes (PQ) = 32 GB  ← fits on one machine
              500M × 768 × 4 bytes (float32) = 1.5 TB  ← needs sharding
  → use PQ compression: 32GB fits in RAM, ~2% recall loss acceptable
  updates:    encode new files → append to FAISS IVF cluster
              full reindex monthly (distribution shift from new repos)
```

### Step 5 — design the query pipeline

```
user query: "function that merges two sorted lists in Python"
    ↓
query processing:
  detect language hint ("Python") → add to filter metadata
  tokenize for BM25: ["function", "merge", "sort", "list", "python"]
  encode for dense:  CodeBERT forward pass → 768-dim vector
    ↓ (parallel, 50ms)
BM25 on Elasticsearch → top 500 candidates per shard → scatter-gather → top 1000
dense ANN on FAISS    → top 1000 candidates
    ↓
RRF fusion → top 100
    ↓
metadata filter: language = Python (applied post-fusion)
    ↓
cross-encoder re-rank top 50  (~100ms with batched GPU inference)
    ↓
return top 10 with code highlighting
```

### Step 6 — success metrics

```
offline: NDCG@10 on labeled code search benchmark (CodeSearchNet)
         MRR on exact symbol lookup queries

online:  CTR@1 (did user click first result?)
         copy-to-clipboard rate (stronger signal than click)
         query reformulation rate (user rephrased → result was wrong)
         session depth (how far down the list did users go?)
```

---

## Worked system design — RAG pipeline for an enterprise Q&A system

**Interviewer prompt:** "Design a question-answering system over a company's internal documentation — 2 million documents including PDFs, wikis, Slack messages, and email threads."

### Step 1 — clarify requirements

```
scale:     2M documents, ~500K new documents/month
QPS:       500 queries/second (internal tool, not consumer scale)
latency:   p99 < 3 seconds (LLM generation adds ~2s, retrieval must be <1s)
freshness: new docs searchable within 1 hour
query:     natural language questions ("how do I request PTO?")
output:    not just retrieved docs — a generated answer with citations
```

### Step 2 — define the data

```
document types:  PDFs (parse with pdfplumber), wikis (HTML), 
                 Slack (JSON API), email (MIME parse)
chunking strategy:
  PDFs:    chunk by paragraph, max 512 tokens, overlap 50 tokens
  wikis:   chunk by section header
  Slack:   thread as one chunk (usually short enough)
  email:   subject + body as one chunk

metadata per chunk:
  source type, author, date, access permissions
  → critical: access control — user should only see docs they're allowed to see
```

### Step 3 — choose retrieval strategy

```
query type: natural language questions → dense retrieval primary
exact match less critical (no product IDs or code symbols)
but: employee names, project codenames → BM25 fallback important
→ hybrid (BM25 + dense), with dense weighted higher (α=0.3 BM25, α=0.7 dense)
   or RRF with slight dense preference

embedding model:
  start: off-the-shelf (e5-large-v2 or OpenAI ada-002)
  later: fine-tune on company-specific (query, document) pairs
         collected from user feedback (clicks, thumbs up/down on answers)
```

### Step 4 — design the index

```
chunking pipeline:
  raw doc → parser → chunker → embedder → dual write:
                                          → Elasticsearch (BM25)
                                          → FAISS IVF (dense)

access control:
  each chunk tagged with permission_groups = ["eng", "hr", "all"]
  at query time: filter by user's groups BEFORE returning results
  never retrieve chunks the user isn't allowed to see

storage:
  2M chunks × 768 × 4 bytes = 6.1 GB float32 → fits in RAM comfortably
  → HNSW is fine at this scale (2M docs << 10M threshold)
```

### Step 5 — query pipeline (RAG-specific)

```
user query: "what is our policy on parental leave?"
    ↓
hybrid retrieval → RRF → top 20 chunks           ~100ms
    ↓
access control filter (remove unauthorized chunks)
    ↓
cross-encoder re-rank → top 5 chunks              ~50ms
    ↓
prompt construction:
  system: "Answer using only the provided context. Cite sources."
  context: [chunk 1] [chunk 2] [chunk 3] [chunk 4] [chunk 5]
  query:   "what is our policy on parental leave?"
    ↓
LLM generation (claude-sonnet / GPT-4)            ~2000ms
    ↓
return: generated answer + source citations + links to original docs
```

### Step 6 — success metrics

```
offline: NDCG@5 on labeled internal query set
         answer correctness (LLM-as-judge: does the answer match ground truth?)
         citation accuracy (do citations actually support the answer?)

online:  thumbs up/down on generated answers
         follow-up question rate (user asked again → answer was incomplete)
         answer acceptance rate (user copied the answer → it was useful)
         hallucination rate (answer contradicts source chunks — monitored by LLM judge)
```

---

## Common FAANG follow-up challenges — and how to answer them

These are the curveball questions interviewers throw after you present a design.

---

**"Your retrieval latency is 50ms but we need 10ms. What do you do?"**

```
option 1: reduce ANN search scope
  HNSW: lower ef_search parameter → faster traversal, lower recall
  FAISS: lower nprobe → search fewer clusters

option 2: quantize embeddings more aggressively
  float32 → int8 → PQ (each step faster, each step slightly lower recall)

option 3: reduce embedding dimension
  768-dim → 256-dim via PCA or MRL (Matryoshka Representation Learning)
  dot product on 256-dim is 3× faster than 768-dim

option 4: cache aggressively
  common queries have cached results, bypass retrieval entirely

option 5: drop the cross-encoder re-ranker
  saves 200ms at the cost of precision — acceptable for some use cases

option 6: move BM25 and ANN search to dedicated hardware
  ANN on GPU (cuVS/FAISS-GPU) cuts search time 5-10×
```

---

**"How do you handle a query where BM25 and dense retrieval return completely different results?"**

```
this is actually useful signal, not a problem.

if BM25 top result and dense top result have RRF scores very close:
  → the query is ambiguous or lies at the boundary of both paradigms
  → log these queries for analysis — they're training data gold

if they diverge consistently on a query type:
  → examine why. usually: exact vs semantic query types are separable
  → build a query classifier: "is this query keyword-style or semantic-style?"
  → route keyword queries to BM25-dominant fusion (high α)
  → route semantic queries to dense-dominant fusion (low α)
  → this query routing pattern is used at Bing, Google, and Elasticsearch
```

---

**"How do you evaluate retrieval quality when you have no labeled data?"**

```
option 1: click-through as implicit relevance
  user clicked rank 3 but not rank 1 → rank 3 was more relevant
  use clicks to build noisy but scalable relevance labels

option 2: LLM-as-judge
  for each (query, retrieved document) pair, prompt an LLM:
  "on a scale 0-3, how relevant is this document to this query?"
  use LLM scores as graded relevance labels for NDCG computation
  fast, scalable, surprisingly accurate (correlates well with human labels)

option 3: generate synthetic queries
  for each document, prompt an LLM: "write 3 questions this document answers"
  → creates (synthetic query, document) pairs
  → evaluate: does your retrieval system retrieve the document for its own synthetic query?
  this is a self-consistency check, not a true evaluation, but useful for zero-label scenarios

option 4: user study
  recruit 10–20 internal users, give them 50 queries, have them rate results
  expensive but ground truth for calibrating the above proxies
```

---

**"What happens if the embedding model gets updated — do you have to re-index everything?"**

```
yes, in general: yes. a new embedding model produces vectors in a different
space — old and new vectors are not comparable. you cannot mix them.

in practice:
  dual-write period: new docs get both old and new embeddings
                     query uses new model against new-doc index + old model against old-doc index
                     merge results with RRF during transition

  background reindex: batch job re-encodes all documents with new model
                      can take days to weeks for billion-doc corpora
                      run in background while old index serves traffic

  Matryoshka embeddings (MRL): models trained with MRL produce nested embeddings —
                                the first 256 dims of a 768-dim vector are
                                themselves a valid 256-dim embedding
                                → you can truncate to match old dimensions,
                                  easing transition between model versions
```

---

## The mental model for any IR system design question

```
1. what type of queries? (keyword / semantic / both)
   → determines sparse vs dense vs hybrid

2. what scale? (docs × QPS)
   → determines index type (HNSW vs FAISS IVF), sharding, caching

3. what latency budget?
   → determines whether cross-encoder re-ranking is possible

4. what update frequency?
   → determines indexing architecture (batch vs streaming)

5. what success metric?
   → determines evaluation setup and what you optimize

answer these 5 questions explicitly and you've covered 90% of what
the interviewer wants to see before they start asking follow-ups.
```

---

## Full formula reference — all 7 chapters

| Chapter | Formula | Meaning |
|---------|---------|---------|
| 1 | `posting(t) = [d : t ∈ d]` | Posting list for term t |
| 1 | `Boolean AND = ∩ posting lists` | Exact intersection retrieval |
| 2 | `TF(t,d) = count(t,d) / \|d\|` | Term frequency normalized by doc length |
| 2 | `IDF(t) = log(N / df(t))` | Inverse document frequency |
| 2 | `TFIDF(t,d) = TF × IDF` | TF-IDF score |
| 2 | `score(q,d) = Σ TFIDF(t,d)` | Query score = sum over query terms |
| 3 | `BM25 = Σ IDF × [TF(k₁+1)] / [TF + k₁(1-b+b\|d\|/avgdl)]` | Full BM25 formula |
| 3 | `TF ceiling = k₁ + 1` | BM25 TF saturation cap |
| 4 | `cosine_sim(q,d) = (q·d) / (\|q\|\|d\|)` | Cosine similarity |
| 4 | `score(q,d) = q·d` | Dot product for unit-normalized vectors |
| 4 | `L2(q,d) = √(Σ(qᵢ-dᵢ)²)` | Euclidean distance |
| 4 | `HNSW: O(log N)` | ANN query complexity |
| 5 | `P@k = relevant_in_top_k / k` | Precision at k |
| 5 | `R@k = relevant_in_top_k / total_relevant` | Recall at k |
| 5 | `AP = (1/R) × Σ P@k × rel(k)` | Average precision |
| 5 | `MAP = (1/\|Q\|) × Σ AP(q)` | Mean average precision |
| 5 | `DCG@k = Σ (2^relᵢ-1) / log₂(i+1)` | Discounted cumulative gain |
| 5 | `NDCG@k = DCG@k / IDCG@k` | Normalized DCG |
| 5 | `MRR = (1/\|Q\|) × Σ 1/rank_first_relevant` | Mean reciprocal rank |
| 6 | `RRF(d) = Σ 1/(k + rank_i(d))` | Reciprocal rank fusion |
| 6 | `hybrid = α×BM25 + (1-α)×dense` | Linear hybrid (requires normalization) |

---

## Final interview checklist

Before walking into an IR system design interview, make sure you can answer every one of these cold:

```
foundations:
  □ explain the inverted index and why it's fast
  □ derive TF-IDF from first principles (two observations → two terms)
  □ explain what BM25 fixes over TF-IDF and why k₁ and b matter
  □ explain bi-encoder vs cross-encoder with latency tradeoffs

retrieval:
  □ walk through HNSW search step by step
  □ explain FAISS IVF nprobe tradeoff
  □ explain why RRF uses k=60 and what happens without it
  □ give three scenarios where you'd choose BM25 over dense

evaluation:
  □ compute AP by hand for a 5-result ranked list
  □ compute NDCG@3 for a graded relevance list
  □ explain when to use MRR vs MAP vs NDCG
  □ explain the gap between offline metrics and online A/B results

system design:
  □ draw the full two-stage pipeline (retrieval → re-rank)
  □ size an embedding index for 100M documents
  □ explain how to handle index updates at scale
  □ explain how to evaluate when you have no labeled data
```

---

That's all 7 chapters. You now have a complete IR curriculum for FAANG ML/DL interviews — from the inverted index to production system design. Good luck.
