# RAG Module 9 — System Design & Interview Synthesis

---

## 9.1 End-to-end system design walkthrough

A worked example of how to structure a live "design RAG for X" interview answer — the *shape* of a strong answer, using "design RAG for internal company documentation" as the running example.

### Step 1: Clarify scope before designing (always do this first, out loud)
- Corpus size and update frequency (10K static docs vs 10M docs updated hourly are completely different systems)
- Query volume and latency requirements (batch analysis tool vs interactive chat with sub-second SLA)
- Query type distribution (short factoid lookups vs complex multi-hop synthesis)
- Access control requirements (does every user see the same corpus, or is content permissioned per user/team?)
- Freshness requirements (can answers be a day stale, or must they reflect the latest edit within minutes?)

**Why this matters as an interview move**: jumping straight to "I'll use Pinecone with HNSW and a cross-encoder reranker" without asking these questions signals memorized-architecture thinking rather than design thinking. Every module in this syllabus has tradeoffs that are *scope-dependent* — stating that explicitly is itself a signal of depth.

### Step 2: Ingestion pipeline
- Source connectors (Confluence, Google Docs, Slack, PDFs — each needs different extraction handling, Module 2.5)
- Chunking strategy choice, justified by corpus type (structure-aware for docs with clear headings, semantic for unstructured prose, table-aware for data-heavy content)
- Embedding model choice (general-purpose vs domain-fine-tuned, Module 1.7) — justify based on how specialized the internal vocabulary is
- Metadata extraction (source, timestamp, author, permissions/ACL tags — directly feeds Step 5)

### Step 3: Indexing
- Index type choice (HNSW vs IVF-PQ) justified by corpus size and update frequency (Module 3.2/3.6) — e.g. "given hourly updates, I'd lean HNSW for native incremental insertion rather than IVF's periodic retraining need"
- Managed vs self-hosted vector DB, justified by existing infra (Module 3.4) — e.g. "if the team already runs Postgres, pgvector avoids a new operational dependency at this scale"

### Step 4: Retrieval + reranking
- Hybrid retrieval (dense + BM25 via RRF, Module 4.3) as a strong default, justified by internal docs typically mixing natural-language content with exact identifiers (ticket numbers, project codenames)
- Reranking stage sized against the latency budget from Step 1 (Module 5.6) — e.g. "given a 1-second SLA, I'd keep k=30 into a lightweight cross-encoder rather than k=100"

### Step 5: Access-control-aware retrieval (expanded in 9.3 below)

### Step 6: Generation + citation
- Context ordering to mitigate lost-in-the-middle (Module 6.1), citation strategy for user trust (Module 6.2)

### Step 7: Evaluation and monitoring
- Golden eval set construction (Module 7.6), online monitoring signals (Module 8.4)

**Interview signal**: walking through steps in this order — scope, ingestion, index, retrieval, generation, eval — *and explicitly connecting each choice back to the scoping constraints from Step 1* is what separates a strong system-design answer from a list of RAG buzzwords. Every choice should have a one-sentence "because..." tied to a stated constraint.

---

## 9.2 Scaling considerations

### Ingestion pipeline scaling
- Batch vs streaming ingestion: batch (nightly re-embed/reindex) is simpler but introduces up-to-a-day staleness; streaming (embed and upsert on document change events) supports near-real-time freshness but requires the index to support incremental updates well (HNSW, Module 3.6) and adds pipeline complexity (event-driven architecture, backpressure handling if embedding throughput can't keep up with document change rate)
- Incremental indexing at scale: re-embedding the *entire* corpus on every content update doesn't scale — need change-detection (only re-embed documents that actually changed, via content hashing) rather than blind full reprocessing

### Multi-tenancy
- Shared index with metadata-based tenant filtering (Module 3.5's filtering mechanisms) vs fully separate indexes per tenant — shared index is more resource-efficient (better utilization, one index to operate) but requires the filtering mechanism to be airtight (a filtering bug leaks one tenant's data into another's results — a severe failure, not a minor bug); separate indexes give hard isolation at the cost of operational overhead multiplying with tenant count and worse resource utilization for small tenants

### Cost modeling — the actual line items to name in an interview
- **Embedding calls**: cost scales with ingestion volume (one-time per document, plus re-embedding on updates) and query volume (one embedding call per query, plus per query-transformation call if using HyDE/multi-query, Module 4.4)
- **Storage**: vector storage scales with corpus size × embedding dimension (Module 3.7's napkin math) — quantization (PQ) or Matryoshka truncation (Module 1.6) are the direct cost levers here
- **Reranker calls**: scale with query volume × k (candidates reranked per query, Module 5.6) — often the largest per-query cost after generation itself
- **Generation calls**: dominant cost per query in most systems, scales with total context tokens (retrieved chunks + prompt overhead) × query volume — this is why context compression (Module 6.4) and tight top-n tuning aren't just latency optimizations, they're direct cost levers

**Interview-ready framing**: cost isn't one number, it's a per-query sum of several independently-scaling line items, and naming which stage dominates cost at a given scale (usually generation, sometimes reranking at very high k) shows you understand the pipeline as a system rather than a single black box.

---

## 9.3 Security / access-control-aware retrieval

A RAG system over permissioned content (internal docs, customer support tickets with PII, multi-tenant SaaS data) must guarantee that **retrieval never surfaces a chunk the requesting user isn't authorized to see** — this is a hard security requirement, not a quality/relevance nice-to-have, and needs to be treated with that level of rigor in a system-design answer.

**Implementation approaches**:
- **Metadata-based filtering at retrieval time** (Module 3.5/4.6): tag every chunk with ACL metadata at ingestion (which users/groups/roles can see it), and enforce that filter as part of every retrieval query — never as a post-hoc filter on the LLM's output, since by then the sensitive content has already been exposed to the model's context window (and potentially logged, cached, or leaked through generation).
- **Pre-filter vs integrated filtering tradeoff, security-flavored**: naive post-filtering (Module 3.5) is not just a recall-degradation risk here, it's a **security bug class** — briefly having unauthorized content pass through any part of the pipeline (even if filtered before being shown to the user) can violate compliance requirements (e.g. data residency, need-to-know access controls) depending on the domain. This is a case where the "correct" engineering answer (integrated/pre-filtering) is also the only acceptable *security* answer, not just a performance optimization.
- **Row-level security / ACL sync**: permissions change over time (a user leaves a team, a document's sharing settings change) — the ACL metadata in the vector index must stay in sync with the source-of-truth permission system (e.g. the underlying document platform's actual ACLs), which is itself a real-time sync problem, not a one-time ingestion-time tag.

**Interview signal**: proactively raising this topic (rather than waiting to be asked) in any system-design question involving internal/enterprise/multi-tenant data is a strong differentiator — it shows awareness that RAG systems are not just retrieval-quality problems, they're data-governance problems.

---

## 9.4 Advanced architectures

### GraphRAG
Instead of (or alongside) a flat vector index, build a **knowledge graph** from the corpus (entities and relations extracted via an LLM or NLP pipeline), and retrieve by graph traversal rather than pure vector similarity.
- **Where it wins**: questions requiring explicit relational reasoning across entities ("what companies did this person's former colleagues go on to found?") where the answer depends on *structured relationships*, not just topical similarity — vector similarity alone struggles here because the relevant entities might not be textually similar to the query at all, only *connected* to it through relationships.
- **Where it loses**: significant upfront cost to build and maintain the graph (entity/relation extraction is itself an error-prone LLM pipeline), and graph construction quality directly bounds retrieval quality (a graph missing an edge is a retrieval-miss failure mode unique to this architecture, on top of everything in Module 8's taxonomy).
- **Practical positioning**: usually deployed as a complement to vector search, not a full replacement — route relationally-structured queries to the graph, route topical/semantic queries to the standard vector pipeline (a routing decision similar in spirit to agentic RAG's retrieval-strategy selection, Module 6.5).

### Multi-index / federated RAG
Query across multiple distinct indexes (e.g. separate indexes per data source — Confluence, Slack, ticketing system — or per data freshness tier) rather than one monolithic index.
- **Why**: different sources often warrant different chunking/embedding strategies (Slack messages are short and conversational, formal docs are long and structured) — forcing them into one index with one chunking scheme is a lossy compromise; federation lets each source use its optimal pipeline.
- **Cost**: query-time fan-out to multiple indexes, then a merge/fusion step across heterogeneous result sets (same fusion problem as hybrid dense+sparse retrieval, Module 4.3 — RRF-style rank fusion generalizes naturally to N systems, not just two) — added latency and engineering complexity relative to a single index.

### Agentic / self-correcting RAG
Covered in depth in Module 6.5 (tool-calling retrieval, Self-RAG, Corrective RAG) — the system-design framing here is *when* to reach for this added complexity: when query difficulty is genuinely heterogeneous (a mix of trivial and multi-hop/ambiguous queries) and a fixed single-shot pipeline would either over-serve simple queries (unnecessary latency/cost) or under-serve complex ones (insufficient iteration to actually answer correctly).

---

## 9.5 Practice question bank

**Conceptual / whiteboard**
1. Walk me through what happens end-to-end when a user submits a query to a RAG system, from query to answer.
2. Explain the bi-encoder vs cross-encoder tradeoff and where each is used in a RAG pipeline.
3. Why does hybrid retrieval outperform dense-only retrieval, and when might dense-only still be the right call?
4. What is "lost in the middle" and how does it change how you construct prompts for RAG?
5. Explain RRF and why rank fusion is preferred over raw score fusion.
6. What's the difference between faithfulness and answer relevance as evaluation metrics, and why measure both?
7. Walk through the four-part RAG failure taxonomy and how you'd distinguish between them for a given bad answer.

**System design**
8. Design a RAG system for customer support over a company's product documentation, targeting sub-second response latency.
9. Design a RAG system for enterprise search over internal documents with per-user access control.
10. Design a RAG system that needs to answer both simple factoid questions and complex multi-hop questions efficiently — how would you avoid paying multi-hop cost on every query?
11. How would you scale a RAG system from 100K to 100M documents? What breaks first, and what would you change?
12. How would you migrate a live RAG system to a new embedding model with zero downtime?

**Debugging / diagnosis**
13. A user reports a hallucinated answer that includes a citation to a real document. How do you determine if the citation is genuine or fabricated-but-plausible?
14. Retrieval quality was fine last quarter and has degraded since, with no code changes — what are your hypotheses and how do you test them?
15. Your eval set shows strong metrics but production user satisfaction is dropping — what's your first hypothesis?

**Tradeoffs / judgment**
16. When would you choose ColBERT-style late interaction over a standard bi-encoder + cross-encoder pipeline?
17. When is GraphRAG worth its added complexity, and when is it overkill?
18. Your reranker is the single largest latency contributor in your pipeline. What are your options for reducing it, and what does each cost you?
19. When would you use Corrective RAG (CRAG) versus a simpler fixed retrieval pipeline?
20. How would you decide between pgvector, a self-hosted FAISS index, and a managed vector DB like Pinecone for a given project?

---

This closes the core RAG syllabus (Modules 0-9, plus 4B on multi-hop). You've now got dense reference notes across embeddings, chunking, indexing, retrieval, multi-hop, reranking, augmentation/generation, evaluation, diagnosis, and system design/synthesis — matching the format and depth of your Agents and LLM Basics tracks.

Want a standalone practice-question bank as a separate file (matching the pattern from your Agents syllabus), or is the question bank in 9.5 sufficient? Also happy to go deeper on any single module, or move on to another topic area.
