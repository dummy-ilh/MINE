# RAG Module 2 — Chunking Strategies

---

## 2.1 Why chunking is the highest-leverage knob in RAG

Before anything gets embedded or indexed, a decision is made about *what unit of text becomes retrievable*. This decision constrains everything downstream:

- Too large → chunks contain multiple topics, embedding becomes a diluted "average" that matches nothing precisely (low precision), and you waste context-window budget on irrelevant text packed alongside the relevant part
- Too small → chunks lose surrounding context needed to be understood standalone (a sentence like "It increased by 40%" is useless without knowing what "it" is), and you fragment a single coherent fact across multiple disconnected chunks (hurts recall — the retriever might grab one fragment but not the others)

**Framing for interviews**: chunking is fundamentally a **precision/recall and context-completeness tradeoff**, and unlike model selection, it's nearly free to iterate on — which is why it's usually the first thing to tune when diagnosing bad RAG output (ties into Module 8).

---

## 2.2 Fixed-size chunking

Split text into chunks of N tokens/characters, typically with an overlap window (e.g. 512 tokens with 50-token overlap).

**Why overlap exists**: without it, a sentence spanning a chunk boundary gets split mid-thought, and the fact/entity reference on one side of the boundary loses its context on the other side. Overlap creates redundancy so boundary-straddling information appears intact in at least one chunk.

**Pros**: trivial to implement, predictable chunk sizes (predictable embedding cost, predictable token budget), works reasonably as a default baseline.

**Cons**: completely ignores document structure — can split mid-sentence, mid-table-row, mid-code-function. Purely mechanical, no semantic awareness.

**Interview trap**: overlap is not free — it multiplies your index size (redundant content stored multiple times) and can cause the *same underlying fact* to be retrieved as several near-duplicate chunks, wasting top-k slots that could've gone to distinct information. Typical overlap is 10–20% of chunk size, not a large fraction.

---

## 2.3 Recursive / structure-aware chunking

Instead of blindly cutting at N tokens, split hierarchically along natural document boundaries — try splitting on `\n\n` (paragraphs) first; if a paragraph is still too large, fall back to splitting on `\n` (lines); if still too large, fall back to sentences; if still too large, fall back to fixed-size as a last resort.

This is the actual default behavior of most production chunkers (e.g. LangChain's `RecursiveCharacterTextSplitter`) — worth naming as the practical industry default, not just fixed-size.

**Why it's better than naive fixed-size**: respects the author's own structural signal (paragraph breaks usually indicate topic shifts) rather than imposing an arbitrary token boundary that ignores meaning entirely.

**Still a limitation**: structure ≠ semantics. A document can have long paragraphs that still cover multiple ideas, or short paragraphs that are only meaningful together (a list item + its intro sentence).

---

## 2.4 Semantic chunking

Instead of splitting on structure, split on **meaning shifts**:
1. Split document into small units (sentences)
2. Embed each sentence
3. Compute similarity between consecutive sentence embeddings
4. Where similarity drops below a threshold (a "semantic breakpoint"), insert a chunk boundary

**Why this exists**: structure-aware chunking can still lump together a paragraph that drifts across two topics, or split apart two short paragraphs that are actually one continuous thought. Semantic chunking directly targets topic coherence rather than proxying it via formatting.

**Cost**: requires embedding every sentence just to *decide* chunk boundaries, before you even embed the final chunks — meaningfully more expensive at ingestion time. Usually only worth it for high-value corpora where retrieval quality matters more than ingestion cost (legal, medical), not for bulk low-stakes content.

**Threshold sensitivity**: too aggressive a similarity threshold → chunks fragment into near-single-sentence units (loses context); too lenient → barely different from paragraph-based chunking. This threshold is usually tuned empirically per corpus, not a universal constant.

---

## 2.5 Document-specific strategies

Generic chunkers break on structured content types. Know the specific failure modes:

- **Tables**: naive chunking can split a table mid-row, or separate the header row from data rows entirely — destroying the ability to interpret any cell. Fix: chunk tables as atomic units when small, or repeat the header row into every chunk when a table must be split (so each chunk is self-describing).
- **Code**: splitting mid-function breaks syntactic and semantic coherence (a function body without its signature is nearly meaningless). Fix: use AST-aware/language-aware splitters that respect function/class boundaries (e.g. tree-sitter-based chunkers).
- **PDFs with layout**: naive text extraction from PDFs often interleaves multi-column text incorrectly (reading left column line 1, right column line 1, left column line 2... in wrong order) or loses table structure entirely. Fix: layout-aware extraction (e.g. detecting columns, using PDF structure/bounding boxes) before chunking — a garbage extraction makes any chunking strategy moot ("garbage in, garbage out" applies at the extraction stage, before chunking even begins).
- **Markdown/structured docs**: chunk along heading hierarchy (H1/H2/H3) so each chunk inherits its section context — directly enables the metadata enrichment technique below.

---

## 2.6 Small-to-big / parent-child chunking

A specific and very commonly tested pattern: **decouple the unit used for retrieval matching from the unit fed to the generator.**

- Index small chunks (e.g. single sentences or small paragraphs) for embedding/retrieval — small chunks give more *precise* similarity matches, since the embedding isn't diluted by unrelated surrounding text
- But when a small chunk is retrieved, **expand and return its parent** (the full section/paragraph/page it belongs to) as the actual context fed to the LLM — giving the generator enough surrounding context to answer well

This solves the core tension in 2.1 directly: precision at retrieval time (small chunks), completeness at generation time (large parent context), without forcing one chunk size to serve both jobs.

**Variants**:
- **Sentence-window retrieval**: retrieve on a single sentence, expand to a fixed window of ±k surrounding sentences at generation time
- **Hierarchical indexing**: multiple levels of chunk granularity indexed simultaneously (e.g. section summaries AND paragraph chunks), letting retrieval match at whichever granularity best fits the query

---

## 2.7 Metadata enrichment per chunk

Attach structured metadata to each chunk beyond raw text, used for filtering and/or improving retrievability:

- **Section titles / breadcrumbs** ("Chapter 3 > Refund Policy > International Orders") — gives the chunk context even when read standalone, and enables metadata-filtered retrieval (Module 3/4)
- **Auto-generated chunk summaries** — a short LLM-generated summary embedded *alongside or instead of* the raw chunk, useful when raw chunk text is noisy (e.g. dense tables, boilerplate-heavy text) and a clean summary embeds better
- **Hypothetical questions** — generate synthetic questions that this chunk would answer, embed those questions instead of (or alongside) the raw chunk. This directly narrows the query-document "asymmetry gap" (recall 1.4's asymmetric encoding note) — since real user queries are questions, embedding synthetic questions puts the index in the *same distributional space* as what it'll be searched with. This is conceptually the ingestion-time cousin of HyDE (previewed here, covered fully in Module 4).
- **Source/timestamp/access metadata** — not for retrieval quality directly, but for filtering (e.g. "only search docs updated in the last year," or permission-based filtering per user — foreshadows Module 9's access-control-aware retrieval)

---

## 2.8 Chunk size vs recall/precision — the empirical tuning loop

There is no universally correct chunk size — it depends on:
- **Query type**: fact-lookup queries (short, specific answers) favor smaller chunks; synthesis/summary queries ("summarize the company's Q3 strategy") favor larger chunks with more surrounding context
- **Embedding model's effective context**: many embedding models degrade in representation quality well before their stated max token limit — stuffing a chunk to the model's absolute max often produces a worse embedding than a more moderate chunk size
- **Downstream LLM context budget**: if you retrieve top-k=10 chunks, total injected context = k × chunk_size — larger chunks force smaller k (or blow the context window), trading breadth of retrieved evidence for depth per chunk

**Practical tuning approach** (good to state explicitly in interviews — shows you don't treat this as guesswork):
1. Build a small labeled eval set (query → known-relevant chunk/passage)
2. Sweep chunk size (and overlap) as a grid, measure Recall@k on the eval set for each configuration
3. Pick the knee of the curve — usually recall improves sharply then plateaus; going past the plateau just wastes context budget
4. Re-validate after any embedding model change — the "best" chunk size is *model-dependent*, not a fixed universal constant

---

## Interview Q&A drill

**Q: You increased chunk size and retrieval recall went up, but answer quality went down. Explain.**
A: Larger chunks capture more complete context per chunk (higher chance the retrieved chunk *contains* the answer, improving recall), but they also dilute the embedding — the chunk-level vector represents an average over more content, blurring precision, and more irrelevant text gets fed to the generator alongside the relevant part, giving it more opportunity to be distracted or to synthesize incorrectly from unrelated content packed into the same context. This is the classic precision/recall tradeoff of chunk sizing, and it's why "just make chunks bigger" isn't a fix — the parent-child pattern (2.6) is often the actual fix, since it decouples retrieval precision from generation completeness.

**Q: When would you choose semantic chunking over recursive structure-aware chunking, given it's more expensive?**
A: When the corpus has high-value, information-dense documents where topic-boundary precision materially affects downstream decisions (legal contracts, medical records, financial filings) and where the extra embedding-time cost is justified relative to the cost of a bad retrieval. For high-volume, lower-stakes content (e.g. general web docs, FAQs with naturally short/structured entries), recursive structure-aware chunking is usually sufficient and much cheaper to run at ingestion scale.

**Q: How do you chunk a 50-page PDF containing both prose and financial tables?**
A: Don't treat it as one chunking problem — split by content type first. Use layout-aware extraction to separate prose regions from table regions. Chunk prose with recursive/structure-aware splitting as normal. Treat tables as atomic units where feasible (or split by logical row groups with the header row repeated into each chunk) so each table chunk stays self-interpretable in isolation, since a table row without its header is not embeddable into anything meaningful.

**Q: What's the actual difference between "chunking" and "indexing," and why do candidates conflate them?**
A: Chunking decides *what text units exist* (a preprocessing/data decision); indexing decides *how those units are stored and searched* (an infrastructure/algorithm decision — ANN structure, index type, sharding). They're conflated because both happen in the ingestion pipeline back-to-back, but chunking quality is corpus/domain-driven and mostly reasoned about with eval sets, while indexing is a systems/scale problem reasoned about with latency and memory tradeoffs (Module 3). A good chunking strategy on a poorly chosen index still retrieves slowly at scale; a great index over badly chunked data still retrieves the wrong content quickly.

---

**Next up: Module 3 — Indexing & vector databases.** Say the word when ready.
