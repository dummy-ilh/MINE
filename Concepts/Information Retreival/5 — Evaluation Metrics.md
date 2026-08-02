# Information Retrieval — Interview Question Bank
### Google · Apple · Cracking the IR Loop

---

## How to Use This Document

Questions are organized by **type**, then by **company flavor**. Google leans into web-scale search and ML system design. Apple leans into on-device retrieval, App Store/Siri ranking, and privacy constraints.

For each question, the answer structure is: **Lead → Core idea → Tradeoffs/Edge cases.** That's the format that lands at L5+.

---

## Part 1 — Fundamentals (asked everywhere, both companies)

These are the building-block questions. If you can't answer these cold, nothing else matters.

---

### Q1. What is the difference between precision and recall in a retrieval system? What is the tradeoff?

**Lead:** Precision = of what we returned, how much is relevant. Recall = of all relevant things, how much did we return.

```
Precision@k = relevant docs in top-k / k
Recall@k    = relevant docs in top-k / total relevant docs in corpus

Example:
  10 relevant docs exist in corpus
  System returns 8 docs, 6 of which are relevant

  Precision@8 = 6/8 = 0.75
  Recall@8    = 6/10 = 0.60
```

**Tradeoff:** Increasing recall typically hurts precision — you retrieve more documents, including irrelevant ones. A system that retrieves everything has recall=1.0 but terrible precision.

**In practice:** Stage 1 (retrieval) optimizes recall — you want every relevant document in your candidate set. Stage 2 (ranking/re-ranking) optimizes precision — you want only the best at the top.

---

### Q2. What is NDCG? Why is it preferred over MAP or MRR?

**Lead:** NDCG (Normalized Discounted Cumulative Gain) measures ranking quality with position-weighted, graded relevance.

```
DCG@k = Σᵢ₌₁ᵏ (2^rel(i) - 1) / log₂(i+1)
NDCG@k = DCG@k / IDCG@k   (normalized by perfect ranking)

Position weighting: rank 1 gets full credit, rank 10 gets ~1/3 credit
Graded relevance: label=3 contributes (2³-1)=7, label=1 contributes (2¹-1)=1
```

**Why NDCG over alternatives:**

| Metric | Graded relevance? | Position-weighted? | Normalized? |
|---|---|---|---|
| MRR | No (binary) | Yes (first relevant) | No |
| MAP | No (binary) | Yes | Yes |
| NDCG | Yes | Yes | Yes |

NDCG handles the reality that "very relevant" and "slightly relevant" aren't the same. It's the standard metric at Google, Bing, and in all major IR benchmarks (TREC, MS MARCO).

---

### Q3. Explain BM25 and why it outperforms TF-IDF.

**BM25 improves on TF-IDF in two ways:**

```
TF-IDF problem 1: Unbounded TF
  A document with "diabetes" 100 times scores 10× higher than one with 10 times.
  In practice, 10 mentions is already a strong signal — the marginal value of 100 is near zero.

BM25 fix: Saturating TF
  TF_BM25(t,d) = TF(t,d) × (k1+1) / (TF(t,d) + k1)
  With k1=1.2: TF=1 → 1.83, TF=10 → 1.98, TF=100 → 2.00
  Diminishing returns kick in immediately. ✓

TF-IDF problem 2: No document length normalization
  A 10,000-word article mentioning "diabetes" 10 times is no more relevant
  than a 500-word article mentioning it 10 times — but TF-IDF scores them equally.

BM25 fix: Length normalization
  Denominator includes (1 - b + b×|d|/avgdl)
  Longer documents get penalized; b=0.75 is the standard balance.
```

Result: BM25 is the industry standard sparse retrieval baseline. Still competitive on many benchmarks even against fine-tuned neural retrievers.

---

### Q4. What is an inverted index? How is it built and queried?

**Build:**
```
corpus: ["diabetes treatment options", "metformin for diabetes", "insulin resistance"]

tokenize and build posting lists:
  "diabetes"   → [doc0, doc1]
  "treatment"  → [doc0]
  "options"    → [doc0]
  "metformin"  → [doc1]
  "for"        → [doc1]
  "insulin"    → [doc2]
  "resistance" → [doc2]

Each entry stores: (doc_id, term_frequency, positions)
```

**Query:**
```
query: "diabetes treatment"
  lookup "diabetes"  → [doc0, doc1]
  lookup "treatment" → [doc0]
  intersect          → [doc0]  (contains both terms)
  score doc0 with BM25
```

**Scalability:** In production (Google), inverted indexes are sharded across thousands of machines. Each shard holds a portion of the document space. Query fanout hits all shards in parallel, results merged.

---

### Q5. What is the difference between a bi-encoder and a cross-encoder? When do you use each?

```
Bi-encoder:
  Query  → encoder → q_vector
  Doc    → encoder → d_vector  (done offline, stored)
  Score  = cosine(q_vector, d_vector)

  Advantage: doc encoding is precomputed — O(1) at query time
  Disadvantage: query and doc never "see" each other — limited interaction

Cross-encoder:
  [Query; Doc] → encoder → relevance_score
  
  Advantage: full attention between query and doc — much richer signal
  Disadvantage: doc cannot be pre-encoded — must run at query time for every doc
```

**Usage pattern:**
- Bi-encoder: Stage 1 retrieval over millions of documents
- Cross-encoder: Stage 2 re-ranking over top 50-200 candidates

At 20ms per cross-encoder call, you can handle ~25 docs in 500ms. So you use bi-encoder to narrow from millions to ~200, then cross-encoder to pick the best 10.

---

## Part 2 — Google-Specific Questions

Google's focus: web-scale architecture, ranking quality, query understanding, LTR, distributed systems.

---

### G1. [System Design] Design Google Search. Walk me through the full architecture.

**What they're looking for:** Can you decompose a massive system into stages and explain each one?

```
Stage 1 — Web Crawling
  Distributed crawler: billions of URLs
  Politeness constraints (robots.txt, crawl rate limits)
  Duplicate detection (SimHash/MinHash for near-duplicate pages)
  Priority: PageRank estimate determines crawl frequency

Stage 2 — Indexing
  Parse HTML → extract text, links, metadata
  Tokenize, stem, remove stopwords
  Build inverted index (sharded, replicated)
  Compute doc-level features: PageRank, spam score, freshness
  Index updates: batch (weekly full rebuild) + incremental (real-time for news)

Stage 3 — Query Understanding
  Spell correction (noisy channel model)
  Query segmentation: "new york times" → ["New York Times"] not ["new", "york", "times"]
  Entity recognition: "Paris Hilton" → person, not location + hotel
  Query expansion: "MI" → "myocardial infarction" for medical queries
  Intent classification: navigational vs informational vs transactional

Stage 4 — Retrieval
  BM25 + dense ANN in parallel → top 1,000 candidates per shard
  Fan out across all index shards
  Merge results via RRF or score normalization

Stage 5 — Ranking (LambdaMART or neural ranker)
  200+ features: BM25, PageRank, freshness, CTR, dwell time, dense_sim, ...
  Two-stage: lightweight ranker → top 200 → heavy ranker → top 10

Stage 6 — Serving
  Result diversification (don't show 10 pages from the same domain)
  Feature enrichment (snippets, knowledge graph boxes, featured answers)
  Personalization (logged-in users get results weighted by search history)
  A/B testing framework: 1% of traffic to experimental rankers
```

**Latency budget for a 200ms SLA:**
- Crawl/index: offline, not on critical path
- Query understanding: ~10ms
- Retrieval (parallel shards): ~30ms
- Ranking: ~50ms
- Serving/formatting: ~10ms
- Total: ~100ms (leaves headroom)

---

### G2. How would you improve search quality after a bad metric regression?

This is a debugging/investigation question. Walk through it systematically:

```
Step 1 — Characterize the regression
  Which metric dropped? NDCG@10, CTR, long-click rate, user satisfaction?
  Which query categories? Head (high volume) vs tail (rare)? Navigational vs informational?
  Which time range? Sudden drop (code change) vs gradual (data drift)?

Step 2 — Isolate the cause
  Did a feature distribution shift? (CTR feature from stale click logs?)
  Did a model change deploy? (New LambdaMART version with different features?)
  Did the index change? (New corpus shard, updated crawl)?
  Did user behavior change? (New query patterns the model hasn't seen)

Step 3 — Reproduce in offline evaluation
  Run the old vs new model on held-out query set with editorial labels
  If offline NDCG also regressed → model/feature issue
  If offline NDCG is fine but online CTR dropped → presentation issue or position bias shift

Step 4 — Fix and re-evaluate
  Roll back if critical
  If data drift: retrain with recent data
  If feature issue: audit feature pipeline
  Always re-evaluate on both offline (NDCG) and online (A/B test)
```

---

### G3. How does PageRank work, and what are its failure modes?

**The algorithm:**
```
PR(d) = (1-d)/N + d × Σ_{v→d} PR(v) / out_degree(v)

d = 0.85 (damping factor — probability of following a link vs. jumping randomly)
N = total pages
v = pages that link to d

Interpretation: rank is proportional to how much "voting weight" flows into you
via links from other highly-ranked pages.

Solved iteratively: initialize PR = 1/N for all, update until convergence (~50 iterations).
```

**Failure modes:**
- **Link farms:** networks of fake pages linking to each other to inflate PageRank
- **Dangling nodes:** pages with no outbound links accumulate rank but distribute nothing (handled by teleportation)
- **Topic insensitivity:** PageRank is query-independent — a page about plumbing can outrank a specialist page just by having more inbound links
- **Temporal staleness:** PageRank is computed offline; newly published viral content won't have accumulated links yet

**Google's response:** PageRank is one of 200+ signals. Trust rank (seed set of trusted domains), spam classifiers, and freshness signals counteract manipulation.

---

### G4. A query returns great results for head queries but fails on tail queries. What do you do?

**Why this happens:**
- Head queries have abundant training data (clicks, dwell time, editorial labels)
- Tail queries (<10 searches/day each) have sparse or zero behavioral data
- LambdaMART overfits to head query patterns; features like CTR are meaningless for tail

**Fixes:**

```
1. Query expansion for tail queries
   Expand "BRCA1 c.5266dupC" with synonyms from medical ontologies (UMLS, MeSH)
   Map rare query to nearby head query clusters

2. Zero-shot dense retrieval
   Dense models generalize better to unseen query patterns than LTR models that
   need behavioral features. Use dense retrieval as a stronger component for tail.

3. Synthetic query generation
   For tail query domains: generate synthetic (query, document) pairs using an LLM
   Fine-tune bi-encoder on synthetic data to improve tail coverage

4. Separate models for head vs tail
   Classifier routes query to head model (CTR + behavioral features) or
   tail model (more weight on BM25 + dense, less on CTR which is unreliable)

5. Human evaluation on tail queries
   Head queries are self-monitoring (lots of clicks).
   Tail quality only surfaces via human raters — sample tail queries regularly.
```

---

### G5. How do you handle position bias in click-through data for LTR training?

**The problem:**
```
rank 1 click rate: ~34%
rank 10 click rate: ~2%

Raw CTR at rank 1 looks 17× better than rank 10,
but the document at rank 1 might be worse — it just benefited from position.
Training on raw CTR amplifies this: model puts rank-1 docs at rank 1 again.
Feedback loop → ranking ossifies around initial (possibly wrong) choices.
```

**Solutions:**

```
Option 1: Inverse Propensity Scoring (IPS)
  Estimate propensity P(click | position k, not relevant)
  Adjust: corrected_CTR(d) = raw_CTR(d, k) / propensity(k)
  
  Propensity estimation: swap rank 1 and rank 2 results for 1% of queries,
  compare click rates to infer position effect.

Option 2: Counterfactual / Randomized display
  For 5% of queries, shuffle the top results randomly.
  Click data from randomized results is unbiased — use for training.
  User experience cost: ~5% of users get a slightly worse experience.

Option 3: Unbiased LambdaMART
  Incorporate propensity weights directly into the lambda gradient:
  λᵢⱼ = (-σ̄ᵢⱼ × |ΔNDCGᵢⱼ|) / (propensity(i) × propensity(j))
  Higher-position documents get downweighted in the gradient.

Option 4: Dwell time as a debiased signal
  A user who clicks rank 1 and immediately bounces → negative signal
  A user who clicks rank 5 and stays 4 minutes → strong positive signal
  Dwell time is less position-biased than raw CTR.
```

---

### G6. Walk me through how you'd evaluate a new ranking model before shipping.

```
Layer 1 — Offline evaluation (before any users see it)
  Dataset: held-out query set with editorial labels (human-judged qrels)
  Metric: NDCG@10, MAP, MRR
  Baseline: current production model
  Pass criteria: NDCG@10 improvement ≥ 0.5% on test set (statistically significant)
  
  Also check: model latency (p50, p99), memory footprint, feature availability in prod

Layer 2 — Shadow mode (model runs but doesn't affect users)
  New model scores every query in parallel with production
  Log scores, compare against production rankings
  Check for distribution anomalies: are scores sane? Any NaN or extreme values?

Layer 3 — A/B test (small traffic slice)
  5% of users → new model
  5% of users → old model (holdout)
  Run for 2–4 weeks (enough statistical power)
  Primary metrics: CTR on top results, long-click rate (dwell > 30s), task abandonment
  Guardrail metrics: query latency (p99), error rate, revenue per query
  
  Watch for novelty effect: users click new results just because they're different

Layer 4 — Gradual rollout
  10% → 25% → 50% → 100% with monitoring at each stage
  Automated rollback if guardrail metrics degrade
```

---

### G7. [LTR Theory] Why does LambdaMART outperform a pointwise regression model?

**Pointwise failure:** A regression model minimizes prediction error on absolute relevance scores. A document scoring 2.1 instead of 2.0 has low loss even if it's ranked below a 0.9-scoring document that should be ranked much lower. The loss doesn't see the ranking — it sees individual prediction errors.

**LambdaMART's advantage:** The lambda gradient directly encodes the ranking objective. For each pair (Di, Dj) where Di should rank above Dj:

```
λᵢⱼ = -σ̄ᵢⱼ × |ΔNDCGᵢⱼ|

The gradient is:
  Large when the ordering is wrong (σ̄ᵢⱼ large)  AND
  Large when fixing the ordering would improve NDCG a lot (|ΔNDCGᵢⱼ| large)

This means:
  - Errors at rank 1-3 get 10-20× larger gradients than errors at rank 50-100
  - The model is explicitly trained to fix the mistakes that hurt users most
  - NDCG improves on each boosting iteration by construction
```

The combination with GBDT (gradient boosted trees) adds: fast inference, handles missing features natively, doesn't require feature normalization, and rarely overfits on datasets of typical IR size (millions of query-doc pairs).

---

## Part 3 — Apple-Specific Questions

Apple's focus: on-device constraints, privacy-preserving retrieval, App Store ranking, Siri/Apple Intelligence, latency under hardware limits.

---

### A1. [System Design] Design the App Store search and ranking system.

**What makes this distinctly Apple:** on-device privacy, no user-level tracking across apps, device-local signals.

```
Stage 1 — Query Understanding
  Spell correction (especially for app names with unusual spellings)
  Category classification: "fitness tracker" → Health & Fitness
  Intent: brand query ("Instagram") vs category query ("photo editor") vs feature query ("dark mode")
  Language detection + localization

Stage 2 — Candidate Retrieval
  BM25 on: app name (high weight), subtitle, keywords field, developer name
  Dense retrieval: app description embedding vs query embedding
  Category filter: narrow by detected category
  Candidates: top 500 apps

Stage 3 — Ranking Features
  Relevance: BM25 score, dense similarity, exact name match (binary), keyword coverage
  Quality: avg rating, rating count, crash rate, app size, update recency
  Popularity: install count, category rank, revenue rank
  Behavioral: CTR on this query (aggregated, not individual), conversion rate (click → install)
  Freshness: days since last update, whether app supports latest iOS version
  Privacy label: apps with minimal data collection may get a small boost (Apple's values)
  Personalization: device language, previously installed categories, country

Stage 4 — Ranking Model
  LambdaMART or two-tower neural ranker
  Training data: editorial quality labels + install conversion as implicit signal
  Key challenge: install → use → keep is the real metric, not just install
    (some apps have high install rate but high uninstall rate within 7 days)
  Multi-objective: optimize install rate + 30-day retention jointly

Stage 5 — Privacy Constraints (Apple-specific)
  No user-level behavioral tracking across apps
  Aggregated click signals only (differential privacy applied)
  On-device personalization: device stores local interest vector, 
    never sent to Apple servers in identifiable form
  Search ads clearly labeled, separated from organic results
```

---

### A2. How would you build an on-device semantic search for Spotlight/Siri with strict latency and memory constraints?

**Constraints:**
- Latency: <50ms on older iPhone hardware (A14 chip)
- Memory: <200MB for the entire search index
- No network calls (works offline)
- Battery: can't drain CPU/GPU continuously

**Architecture:**

```
Offline (server-side, pushed to device):
  1. Build compressed embedding index for on-device content
     (Notes, Messages, Files, Contacts, Mail subject lines)
  2. Use quantized embeddings: float32 → int8 (4× size reduction)
     768-dim float32 vector = 3KB → int8 = 768 bytes
  3. HNSW index structure for ANN search (hierarchical navigable small world)
     Allows fast approximate search with 10-30ms query time
  4. Total index: 1M documents × 768 bytes + HNSW overhead ≈ ~1GB
     Too large → use PQ (Product Quantization) to compress to ~100MB

On-device query serving:
  1. Encode query: CoreML model (MobileBERT or Apple's custom small model)
     Runs on Neural Engine → ~5ms
  2. ANN search in compressed HNSW index → top 50 candidates → ~10ms
  3. Re-score top 50 with lightweight cross-encoder (on-device) → ~20ms
  4. Return top 10 → ~5ms formatting
  Total: ~40ms ✓

Privacy guarantees:
  All computation on device
  Query never leaves device
  No click signal sent to Apple servers
  Index built from device content → no server knows what's on device
```

---

### A3. Apple News ranking: how do you personalize without user-level tracking?

**The core tension:** Personalization requires knowing user preferences. Apple's privacy model forbids sending individual reading history to servers.

**Solution: On-device personalization model**

```
1. Local interest model (runs on device)
   Tracks: articles read, dwell time, shares, topics engaged with
   Builds: local interest vector [sports=0.8, tech=0.6, politics=0.2, ...]
   Stored: only on device, never transmitted

2. Server-side: train a general ranker (no personalization)
   Features: article quality, publisher reputation, freshness, topic trend score
   Train with: editorial labels, aggregated (not individual) engagement signals
   Output: base_score(article)

3. On-device reranking
   final_score(article) = base_score(article) × personalization(article, local_interest_vector)
   
   personalization = dot_product(article_topic_vector, local_interest_vector)
   
   This multiplication happens entirely on-device: server never sees what the user likes

4. Aggregated signals with differential privacy
   Apple can collect: "what fraction of users who read tech articles also read finance articles"
   This trains the base ranker's co-engagement features
   Individual user's behavior: never transmitted
   DP noise added before aggregation: individual contributions unrecoverable
```

---

### A4. How would you detect and handle query drift in a production Apple Search ranking model?

**Query drift:** User search patterns change over time. A model trained on last year's data may perform poorly on today's queries.

```
Detection:
  Monitor: distribution of query topics (KL divergence vs training distribution)
  Monitor: new n-grams not seen in training (vocabulary OOV rate)
  Monitor: model confidence scores — if confidence drops, distribution shift likely
  Monitor: CTR on served results — unexplained drops signal model mismatch
  Alert: if any metric moves >2σ from 30-day moving average

Types of drift:
  Concept drift:  "iPhone" used to mean iPhone 14 searches → now iPhone 16
  Feature drift:  CTR feature collected under old ranking → biased for new ranking
  Seasonal drift: holiday queries, new product launches, iOS release cycles

Response:
  Fast path (< 1 day): 
    Re-weight recent training data more heavily in next model refresh
    Boost freshness feature weight for topics with detected drift

  Medium path (1 week):
    Trigger full retraining with sliding window of recent data
    Add new vocabulary to embedding model via continual fine-tuning

  Slow path (model architecture):
    If drift is persistent, evaluate whether model architecture needs updating
    (e.g., add a topic-freshness feature that wasn't in original design)

Apple-specific: model is also deployed on-device.
  On-device model update cadence: pushed via software update or background download
  Can't update on-device model on same day as server-side detection
  → server-side fallback must handle drift period while device update propagates
```

---

### A5. [Apple Intelligence] Design a RAG system for Siri that answers questions from personal device content.

This is the 2025-2026 question Apple is actively building toward with Apple Intelligence.

```
What we're building:
  User: "Siri, what did John say about the meeting venue?"
  Siri: searches iMessages, Calendar, Notes, Mail → retrieves relevant context → answers

Architecture:

Indexing (on-device, runs in background when charging + on Wi-Fi):
  Sources: Messages, Mail, Notes, Calendar, Files, Photos metadata, Safari history
  Chunking strategy:
    Messages: conversation threads as chunks (10-message windows with overlap)
    Notes: paragraph-level chunks (150-200 tokens with 20 token overlap)
    Mail: subject + first paragraph as chunk (body too long; truncate)
    Files: PDF/doc text extraction → sentence-level chunks
  Embedding: on-device CoreML model (small: 256-dim, quantized to int8)
  Storage: HNSW index per source type (separate indexes for Messages, Mail, etc.)

Query serving:
  Query → encode → ANN search across all source indexes → top 20 per source
  Metadata filter: "last week" → filter by date before ANN search
  Source-aware RRF: fuse results across sources
  Re-rank top 20 with lightweight cross-encoder

Context assembly:
  Top 5 retrieved chunks → injected into Siri's prompt as context
  Query + context → on-device LLM (Apple Intelligence model) → natural language answer

Privacy:
  Everything on-device: query encoding, index search, LLM inference
  No personal content sent to server
  iCloud Private Relay used if any server call needed (metadata only)

Key challenges:
  Index freshness: new messages arrive continuously → incremental index updates
  Multi-modal: user photo has a person's name detected by Vision → index that too
  Memory: full personal index must fit in ~500MB
  Authorization: Siri must only access data the user has granted access to
```

---

## Part 4 — Cross-Company Deep Questions

These appear at both Google and Apple (and Meta, Bing, Amazon). Master them.

---

### X1. What is the curse of dimensionality and how does it affect vector search?

```
In high-dimensional space, all points become approximately equidistant.

Intuition:
  In 2D: a query vector has a clear "nearest neighbor"
  In 768D: the ratio (max_dist - min_dist) / min_dist → 0
  All documents cluster at roughly the same cosine distance from the query
  True nearest neighbor search becomes meaningless

In practice for retrieval:
  768-dim embeddings: still useful — enough signal survives
  But exact nearest neighbor search is O(N) — you must compare every vector
  → Use Approximate Nearest Neighbor (ANN) search

ANN algorithms:
  HNSW (Hierarchical Navigable Small World):
    Build a graph where nearby vectors are connected
    Query: greedy graph traversal → O(log N) average
    Tradeoff: build time is slow, recall is 95-99% (not 100%)
  
  IVF (Inverted File Index, used by FAISS):
    Cluster vectors into k centroids (via k-means)
    At query time: compare to centroids → search only in nearby clusters
    Tradeoff: recall depends on how many clusters you search (nprobe parameter)
  
  Product Quantization (PQ):
    Compress vectors to save memory
    Tradeoff: some precision lost, but 8-32× smaller index

Production choice:
  Google/Meta scale: IVF-PQ (FAISS) for memory efficiency
  Apple on-device: HNSW with aggressive quantization for speed+size
```

---

### X2. What is query expansion and when does it help vs hurt?

```
Query expansion: add related terms to the original query before retrieval
Goal: increase recall for queries with synonym/vocabulary mismatch

Methods:

1. Pseudo-Relevance Feedback (PRF)
   Run original query → take top-k results → extract frequent terms → 
   re-run query with those terms added
   
   Risk: if top-k results are wrong, expansion amplifies the error
   ("query drift")

2. Thesaurus/Ontology expansion
   "diabetes" → add "T2DM", "type 2 diabetes", "hyperglycemia"
   Using UMLS (medical), WordNet (general)
   
   Risk: "bank" → "river bank", "financial institution" — wrong sense expansion

3. LLM-based expansion (2024+ trend)
   Prompt an LLM: "List 5 alternative phrasings for: [query]"
   Use generated queries as parallel BM25 searches, fuse with RRF
   
   Risk: LLM may generate hallucinated terms that match irrelevant docs
   Latency: adds ~50ms for LLM call

When it helps:
  Rare medical/legal/scientific terminology
  Voice queries (informal language matching formal documents)
  Cross-lingual queries

When it hurts:
  Precise queries: "Python 3.11 asyncio bug" → expansion may return Python 2 docs
  Named entity queries: "Apple stock" → expansion may add fruit-related terms
  Short queries where intent is already clear
```

---

### X3. How do you build a search system for a new domain with no training data?

This is a classic cold-start problem. Step-by-step answer:

```
Step 1 — Deploy BM25 as immediate baseline
  No training data needed. Works on any text corpus.
  Benchmark: measure NDCG@10 with a small labeled query set (50-100 queries,
  hand-labeled by domain experts)

Step 2 — Add a general-purpose dense model
  Start with a pre-trained model (e.g., sentence-transformers/all-mpnet-base-v2)
  This gives you semantic search immediately, even without domain fine-tuning
  Combine with BM25 via RRF
  Re-measure NDCG@10: hybrid usually beats BM25 alone by 5-15%

Step 3 — Synthetic query generation (if still insufficient)
  For each document, prompt an LLM: "Write 3 questions that this document answers"
  You now have (query, document) pairs for fine-tuning
  Fine-tune bi-encoder on synthetic pairs (GPL method)
  Re-measure: typically +5-15% more NDCG

Step 4 — Collect real feedback
  Deploy the Step 2-3 system to real users
  Log: queries, results shown, clicks, dwell time
  This is your first real training signal for LTR
  Build first LambdaMART model after 2-4 weeks of data collection

Step 5 — Human evaluation on a sample
  Sample 500 queries from logs (stratified: head/torso/tail)
  Have domain experts label relevance
  These become your ground-truth qrels for offline evaluation going forward
```

---

## Part 5 — Quick-Fire Definitions (common in phone screens)

**What is TF-IDF?** Term Frequency × Inverse Document Frequency. Rewards terms that appear frequently in a document but rarely across the corpus. Rare terms get high scores.

**What is cosine similarity?** Dot product of two unit vectors. Measures the angle between them regardless of magnitude. Standard similarity metric for embedding vectors.

**What is ANN search?** Approximate Nearest Neighbor — finds vectors close to the query without checking every vector in the index. Trades small recall loss for large speed gain. HNSW and IVF-FAISS are the standard implementations.

**What is chunking in RAG?** Splitting long documents into smaller segments for embedding and retrieval. Chunk size tradeoff: small chunks (precise retrieval, lose context) vs large chunks (more context, less precise). Typical: 256-512 tokens with 10-20% overlap.

**What is a posting list?** In an inverted index, the list of documents containing a given term. Lookup: `"diabetes" → [(doc3, tf=2, pos=[14,87]), (doc7, tf=1, pos=[3]), ...]`

**What is the two-tower model?** A neural architecture where query and document are encoded by separate towers (encoders) into the same vector space. Equivalent to bi-encoder. Standard for dense retrieval at scale.

**What is MRR?** Mean Reciprocal Rank — average of 1/rank_of_first_relevant_document across queries. MRR=1.0 means the first result is always relevant. Good for navigational queries where there's one right answer.

**What is HNSW?** Hierarchical Navigable Small World. A graph-based ANN index where each node connects to nearby nodes at multiple resolution layers. Query: start at top layer → greedy descent → find approximate nearest neighbors in O(log N). Fast, high recall, but memory-intensive.

**What is the vocabulary mismatch problem?** When a query uses different words than the relevant documents (synonyms, paraphrases). BM25 fails completely; dense retrieval solves it. The core motivation for hybrid search.

---

## Master Summary — What Interviewers Are Listening For

| What they ask | What they're really testing |
|---|---|
| "Design Google Search" | Can you decompose and prioritize a massive system? |
| "How does LambdaMART work?" | Do you understand *why* LTR beats pointwise approaches? |
| "Handle position bias in click data" | Do you know that naive click training creates feedback loops? |
| "Design App Store ranking" | Can you apply IR principles to a specific Apple product context? |
| "On-device search constraints" | Do you understand Apple's privacy model and latency/memory tradeoffs? |
| "Cold-start domain with no data" | Can you build incrementally: BM25 → hybrid → synthetic → LTR? |
| "NDCG vs MRR vs MAP" | Do you know which metric fits which use case? |
| "Bi-encoder vs cross-encoder" | Do you understand the recall-precision pipeline architecture? |

**The meta-skill they're hiring for:** You can take a vague product requirement ("users can't find what they're looking for"), decompose it into an IR problem, pick the right retrieval and ranking approach for the constraints, and iterate from baseline to production with clear evaluation at each step.
