# RAG Interview Prep — Day 28
## Final Master Cheat Sheet — Days 1–24 Consolidated

*Built for a single fast skim the morning of the interview. Organized by week. Each block is deliberately compressed — if anything here doesn't immediately make sense, that's your signal for one last targeted look at that specific day's file tonight, not a new gap to panic about.*

---

## WEEK 1 — FOUNDATIONS

**Day 1 — RAG vs. Fine-Tuning vs. Long-Context**
Fine-tuning = fix *behavior* (tone, format, skill), bakes into weights, stale unless retrained, no natural citations. Long-context = fresh but expensive/slow, subject to lost-in-the-middle. RAG = fix *knowledge*, cheap per-query at scale (~300-800x cheaper than long-context in worked examples), natural citations + access control. **Golden line:** rarely either/or — fine-tune the skill of using context well, RAG supplies the knowledge.

**Day 2 — Embeddings**
Pipeline: tokenize → encoder → pooling (mean or CLS) → fixed vector. Cosine similarity = `(A·B)/(‖A‖‖B‖)`, magnitude-invariant, the default. Dot product = cosine-equivalent only if pre-normalized (cheaper then). Model choice: dimensionality vs. cost, general vs. domain-specific (often bigger lever than tuning k), MTEB as filter not final answer. **Gotchas:** embedding drift (never mix vector spaces across model versions — full re-embed on upgrade), quantization (float32→int8 ≈ 4x savings).

**Day 3 — Chunking**
Fixed-size (fast, cuts sentences) → sliding window w/ overlap (`stride = chunk_size − overlap`) → recursive (production default) → semantic (embeds sentences, splits at similarity drops, priciest) → structure-aware (tables/code/headers). **Small-to-big retrieval:** search small chunks for precision, return large parent chunks for generation completeness. Chunk size = never a fixed default — sweep against golden eval set.

**Day 4 — Vector DB & Indexing**
Flat (exact, O(N×d), unusable past ~1M) → HNSW (graph, `M`/`ef_construction`/`ef_search`, great incremental updates, more memory) → IVF (clusters, `nlist`/`nprobe`, faster build, worse at updates — centroid drift) → PQ (orthogonal compression, ~100x+ via sub-vector codebooks, approximate distances) → IVF-PQ (billion-scale combo). Capacity planning: raw vector memory → index-overhead multiplier → shard decision → per-replica QPS → replica count.

**Day 5 — Metadata Filtering & Multi-Tenancy**
Post-filter only for low-selectivity filters; pre-filter/filter-aware ANN for high-selectivity. Multi-tenancy: shared+filter (cheap, filter-bug=data leak, noisy neighbors) → namespace/partition (good middle ground) → fully separate index (strongest isolation, linear cost). **Access control MUST be enforced at retrieval layer, not UI** — else unauthorized content leaks into generated answers silently. Graph-based ANN indexes (HNSW) don't support cheap clean deletion — compliance/GDPR implication.

---

## WEEK 2 — RETRIEVAL

**Day 7 — Sparse Retrieval (BM25/TF-IDF)**
`TF-IDF = TF(t,d) × IDF(t)`, `IDF = log(N/n(t))`. BM25 fixes TF-IDF via saturation (`f(f+k1)`, diminishing returns) + tunable length norm (`b`, relative to corpus avg length). Typical `k1≈1.2-2.0`, `b≈0.75`. Sparse still wins: exact IDs (SKUs/error codes), rare/OOV terms, phrasing-sensitive domains. Inverted index = term→doc list, avoids brute force.

**Day 8 — Dense Retrieval (Bi- vs. Cross-Encoder)**
Bi-encoder: independent encoding, precomputable doc vectors, fast (feeds ANN search). Cross-encoder: joint attention, NOT precomputable, more accurate, infeasible past small candidate sets (5M docs × 15ms ≈ 20+ hrs). Asymmetric encoders (DPR) for structurally different query/doc pairs. Training: InfoNCE contrastive loss, hard negatives >> in-batch negatives for quality.

**Day 9 — Hybrid Search & RRF**
Can't add BM25 + cosine directly (incompatible scales). **RRF:** `Σ 1/(k+rank)`, `k≈60`, uses only rank not raw score — sidesteps scale problem entirely, robust to outliers, the production default. Score-level fusion (min-max normalize + weighted `α`) is more tunable but fragile to per-query outliers.

**Day 10 — Reranking**
2nd stage: reorders candidates for top-of-list precision, doesn't discover new docs (measure impact via **nDCG, not Recall@k**). Cross-encoder = standard. ColBERT/late-interaction (MaxSim: sum of each query token's best doc-token match) = precomputable middle ground. LLM-based rerank = most flexible/expensive, reserve for small shortlists. Reranker scores aren't calibrated across queries — no fixed thresholds.

**Day 11 — Query Transformation**
Multi-query (N phrasings, N retrievals, fuse via RRF) vs. HyDE (generate hypothetical answer, embed THAT, 1 retrieval — hypothetical doc discarded after embedding, factual accuracy irrelevant) vs. decomposition (multi-part/comparative questions, no single chunk could answer the whole thing) vs. step-back (broader query alongside specific one). All add real latency — use query-aware routing, not blanket application.

---

## WEEK 3 — GENERATION

**Day 13 — Context Construction & Lost-in-the-Middle**
U-shaped attention curve — start/end = high attention, middle = risk zone, even if content is technically present. **Sandwiching:** best chunks at both start AND end. More chunks (k) ≠ better — tune against generation metrics, not Recall@k. Reserve generation headroom FIRST in budget allocation. Query near the end of prompt (recency effect).

**Day 14 — Compression & Caching**
Dedup (free, always first) → extractive (moderate risk, worked ex: 8x) → abstractive (highest faithfulness risk — it's itself a generation step) → token-level algorithmic. Prefix/KV caching = exact-match, risk-free, reuses attention states for identical prefixes. Semantic caching = similarity-based, needs threshold tuning AND cache invalidation tied to knowledge-base updates (staleness risk is real).

**Day 15 — Citation & Faithfulness Enforcement**
Measurement (Module 7, aggregate) ≠ enforcement (runtime, per-response). Inline citations = cheap, fabrication risk. Post-hoc attribution = verify-then-cite, more reliable. Constrained decoding = format guarantee only, not accuracy. Runtime groundedness guardrail = live gate, ~200-400ms typical cost, justified for high-stakes domains. **Refusal calibration = highest-leverage single intervention** — needs a two-sided golden eval set (answerable + unanswerable) to tune correctly.

**Day 16 — Multi-Hop & Agentic RAG**
Static decomposition (Day 11) plans upfront; dynamic multi-hop (ReAct: Thought→Action→Observation) plans as it goes, needed for genuinely sequential dependencies. Self-RAG/reflection adds a quality gate per hop. Stopping criteria: max-hop limit + LLM self-assessment, combine don't rely on one. **Error propagation** = the signature unique risk — an early wrong fact silently corrupts everything downstream.

**Day 17 — Failure Modes Catalog**
Taxonomy: Retrieval (found it?) → Context assembly (presented well?) → Generation (used correctly?) → Cross-cutting (drift/staleness). **Over-reliance on parametric knowledge** = correct context present, model still answers from a strong pretrained default — needs deliberately-constructed counterfactual eval slices to detect. **Refusal miscalibration** = two-sided classification problem (false refusal vs. false answer), costs differ by domain.

---

## EVALUATION & DIAGNOSIS

**Module 7 — Evaluation**
Retrieval metrics: Recall@k (`|relevant∩top-k|/|relevant|`, most fundamental), Precision@k, MRR (`1/rank`, single best hit), nDCG (`DCG/IDCG`, graded relevance + position, most complete). RAG triad: Faithfulness (grounded in context? ≠ correct), Answer Relevance (addresses the question?), Context Relevance (was fetched content useful?) — fail independently, triangulate together. LLM-judge biases: position, verbosity, self-preference — always calibrate against humans. Golden set needs: easy, multi-hop, no-answer-exists, paraphrased slices. Offline = cheap pre-filter; A/B test = final confirmation.

**Day 20 — Diagnosis & Debugging**
Workflow order: (1) Was the right evidence retrieved? (raw top-k) → (2) Was it presented well? (actual constructed prompt) → (3) Did generation use it correctly? → (4) Is it actually a data/corpus problem? Required logging: query+transforms, raw retrieval+scores, post-rerank order, THE LITERAL PROMPT, answer+citations, guardrail results. Real bugs often span multiple stages — don't stop at first plausible explanation. Always regression-test the full eval set after a fix, not just the original complaint.

---

## SYSTEMS & APPLE-SPECIFIC

**Day 22 — System Design Methodology**
4 phases: Clarify (scale/latency/freshness/consistency/tenancy/budget/stakes) → High-level architecture (one full pipeline diagram) → Deep dives (interviewer-directed, 2-3 areas) → Trade-offs. Don't over-engineer for scale you don't have. Cost model: account for embedding + **reranking (often the surprise dominant cost)** + generation + infra separately. Monitoring: split retrieval-stage vs. generation-stage metrics on dashboards; sample faithfulness checks, don't run on 100% of traffic.

**Day 23 — Apple-Specific**
Three-tier: on-device (AFM 3 Core, ~3B params, 2-bit quantized, ~4K context, offline, no limits) → Private Cloud Compute (~32K context, reasoning, connectivity required, daily limits, non-retention privacy) → AFM 3 Cloud Pro (Google Cloud/NVIDIA, most demanding). RAG matters MORE for small on-device models (less parametric knowledge to fall back on). Index compression = mandatory at far smaller scale on-device (absolute memory ceiling, not relative). Privacy = architectural (data never leaves device) not just filtering. Routing = core architecture decision, not optimization.

---

## THE FIVE ANSWERS TO HAVE INSTANTLY READY

These come up in some form in nearly every RAG interview — have them at true reflex speed:

1. **"Why RAG over long-context?"** → Cost (pay for full context every query), latency, lost-in-the-middle, access control. Not mutually exclusive with fine-tuning/long-context.

2. **"Why two-stage retrieve-then-rerank?"** → Bi-encoders are precomputable/fast but miss fine-grained interactions; cross-encoders are accurate but can't scale to a full corpus (architectural, not just slow). Combination gets both.

3. **"How do you combine sparse and dense?"** → RRF — fuse ranks not scores, sidesteps incompatible scales, robust default.

4. **"Why does a wrong answer happen even with good retrieval?"** → Could be lost-in-the-middle, dilution, pure hallucination, OR over-reliance on parametric knowledge (correct context present, model ignores it) — work the diagnostic workflow stage by stage, don't assume.

5. **"How do you evaluate a RAG system?"** → Separate retrieval metrics from generation metrics, always — a single end-to-end score conflates independently-failing stages and can't tell you what to fix.

---
# RAG Interview Cheat Sheet

Quick-scan reference — pairs with your detailed RAG Q&A doc and the 500M-PDF system design doc.

---

## 1. The Pipeline (memorize this order)

```
INDEXING (offline):  Parse → Chunk → Embed → Store (vector index + doc store)
QUERY (online):      Query → [rewrite] → Embed → Retrieve (dense+sparse) → Rerank → Prompt → Generate → [verify]
```

---

## 2. Core Term Sheet

| Term | One-liner |
|---|---|
| Bi-encoder | Encodes query/doc independently → fast, used for first-stage retrieval |
| Cross-encoder | Encodes query+doc jointly → accurate, slow, used for reranking only |
| Dense retrieval | Embedding similarity search — good at semantic/paraphrase matches |
| Sparse retrieval (BM25) | Term-frequency matching — good at exact terms, IDs, rare words |
| Hybrid search | Dense + sparse fused (usually via RRF) — covers both failure modes |
| HNSW | Graph-based ANN index; default choice, fast, memory-hungry |
| IVF / IVF-PQ | Cluster-based ANN index; lower memory (PQ = compressed vectors) |
| RRF (Reciprocal Rank Fusion) | Standard way to merge ranked lists from dense + sparse search |
| HyDE | Embed a *generated hypothetical answer* instead of the raw query |
| Lost in the middle | LLMs underuse info placed mid-context vs. start/end |
| Groundedness / faithfulness | Does the answer only state what's in retrieved context? |
| RAGAS | LLM-as-judge framework for faithfulness/relevance/context metrics |
| Parent-document retrieval | Retrieve small chunks, but feed the full parent section to the LLM |

---

## 3. Retrieval Evaluation Metrics

| Metric | Measures | Use when |
|---|---|---|
| Recall@k | Is the right doc in top-k? | Standard baseline check |
| Precision@k | How many of top-k are relevant? | Care about noise in context |
| MRR | 1/rank of first relevant hit | Usually one right answer |
| nDCG | Rewards relevant docs ranked earlier, handles graded relevance | Multiple relevance levels |

---

## 4. Chunking Decision Table

| Doc type | Strategy |
|---|---|
| Unstructured prose | Semantic chunking (split where embedding similarity drops) or recursive splitting |
| Structured docs (Markdown/HTML/contracts) | Structure-aware (split on headers/sections) |
| Tables | Serialize rows with header context — don't chunk mid-table |
| General default | Recursive char splitting, ~256–512 tokens, 10–20% overlap |

**Tradeoff:** too small → loses context; too large → embedding gets "blurry," wastes context budget.

---

## 5. RAG vs. Alternatives

| | RAG | Fine-tuning | Long-context |
|---|---|---|---|
| Best for | Dynamic facts, citations | Style/format/behavior | Small static corpora |
| Update cost | Cheap (re-index) | Expensive (retrain) | N/A |
| Hallucination control | Good | Weak on facts | Good if relevant, but lost-in-middle risk |

---

## 6. Debugging Hallucination (despite correct retrieval) — checklist

1. Is the right chunk buried mid-context? → reorder / rerank
2. Does the prompt enforce "only use provided context"? → add explicit grounding instruction
3. Is the fact split across chunk boundaries? → increase overlap or use parent-doc retrieval
4. Too much noise in context? → add cross-encoder reranking, reduce k
5. Model ignoring context in favor of pretraining knowledge? → stronger grounding + post-hoc entailment check

---

## 7. Scaling Cheat Sheet (500M+ docs)

- **Storage estimate:** `chunks × embedding_dim × bytes_per_dim` (e.g., 5B chunks × 768 × 4B ≈ 15TB before compression)
- **Sharding:** hash-based, topic-cluster, or time-based — needed once index exceeds single-node RAM
- **Compression:** PQ trades some recall for large memory savings
- **Metadata filtering:** pre-filter (filter then search) beats post-filter at scale
- **Freshness:** incremental upserts + tombstoning for deletes, not full re-index

---

## 8. Latency Budget (typical interactive target ~800ms–2s)

| Stage | Rough cost |
|---|---|
| Query embedding | ~20ms |
| ANN search | ~50–100ms |
| Cross-encoder rerank | ~100–300ms (scales with candidates) |
| LLM generation | ~500ms–2s+ |

**Fastest levers:** cache embeddings/responses, shrink k before reranking, stream generation output.

---

## 9. Clarifying-Question Dimensions (system design opener)

Scale · Latency · Freshness · Consistency · Multi-tenancy · Budget · Accuracy/stakes

**Method:** for each — extract the number → identify the mechanism it stresses (memory? sync call? blast radius?) → pick the technique that relieves that specific stress. State the mechanism out loud, not just the technique.

---

## 10. Apple-Specific Quick Hits

- **On-device-first:** privacy + latency + cost-at-scale (billion+ devices) all point the same direction.
- **Private Cloud Compute (PCC):** stateless, no privileged access, cryptographically attested — cloud fallback isn't "call an API," it's this specific pattern.
- **Core ML / ANE constraints:** model size, memory footprint, op coverage, battery/thermal — quantization is often not optional.
- **Federated learning + differential privacy:** train from device data without centralizing raw data; DP adds noise so aggregates can't be reverse-engineered to an individual.
- **Red flag answer:** "just send everything to a cloud LLM and cache aggressively" — ignores on-device-first / privacy-preserving-escalation expectations.

---

## 11. Rapid-Fire One-Liners

- Cosine vs. dot product: same ranking if vectors are normalized; dot product is faster but magnitude-sensitive.
- Reranking cost: O(query×doc) per pair — only feasible on ~20–100 candidates, never the full corpus.
- Multi-hop questions: decompose into sub-queries, iterate (ReAct-style), or use graph-based retrieval.
- Why not just use a huge context window instead of RAG: cost scales per call, lost-in-the-middle, doesn't scale past corpus > context size.
