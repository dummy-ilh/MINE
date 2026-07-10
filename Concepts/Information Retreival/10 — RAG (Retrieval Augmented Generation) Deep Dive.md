# Chapter 10 — RAG (Retrieval Augmented Generation) Deep Dive

## What is it?

Retrieval Augmented Generation (RAG) is an architecture that combines an IR retrieval system with a large language model (LLM). Instead of asking an LLM to answer from its parametric memory alone — knowledge baked into its weights during training — RAG first retrieves relevant documents from an external corpus, then feeds those documents as context to the LLM to generate the answer.

RAG is the most interview-relevant IR topic right now. Nearly every FAANG team building on top of LLMs uses some form of it. Understanding RAG deeply — not just "retrieve then generate" but chunking strategy, context management, hallucination control, and evaluation — is what separates a strong candidate from a weak one.

The core problem RAG solves:

```
LLM alone:
  ✗ knowledge cutoff — doesn't know recent events
  ✗ hallucination — confidently states wrong facts
  ✗ no citations — can't point to sources
  ✗ private data — can't access your company's internal docs

RAG:
  ✓ retrieves fresh documents at query time — no cutoff
  ✓ answer grounded in retrieved text — reduces hallucination
  ✓ citations come from retrieved chunks — auditable
  ✓ works on any corpus — private, proprietary, real-time
```

---

## The intuition

Think of RAG as an open-book exam versus a closed-book exam.

A closed-book LLM (no retrieval) must rely entirely on what it memorized during training. It's impressive for general knowledge but fails on specifics, recent events, and private information. It also hallucinates — when it doesn't know something, it often generates a plausible-sounding wrong answer instead of saying "I don't know."

A RAG system gives the LLM the relevant pages from the textbook before it answers. The LLM's job shifts from "remember the answer" to "read these passages and synthesize a response." This is fundamentally more reliable for factual questions — the answer is right there in the context, not reconstructed from compressed weights.

The IR system's job in RAG is not just to find relevant documents — it is to find the **right chunks at the right granularity** so the LLM has exactly what it needs without being overwhelmed by irrelevant context.

---

## The RAG pipeline — full anatomy

```
offline (indexing):
  raw documents
      ↓
  parsing          → extract clean text from PDFs, HTML, DOCX
      ↓
  chunking         → split into retrievable pieces
      ↓
  embedding        → encode each chunk with bi-encoder
      ↓
  dual write       → inverted index (BM25) + ANN index (dense)

online (query):
  user question
      ↓
  query processing → rewrite, expand, decompose
      ↓
  retrieval        → hybrid BM25 + dense → RRF → top-k chunks
      ↓
  re-ranking       → cross-encoder or ColBERT → top-n chunks
      ↓
  context assembly → order chunks, add metadata, trim to token budget
      ↓
  prompt construction → system prompt + context + question
      ↓
  LLM generation   → answer + citations
      ↓
  post-processing  → verify citations, check faithfulness, format
```

Every stage has failure modes. This chapter covers each one.

---

## Stage 1 — Chunking strategies

Chunking is the most underestimated part of RAG. The wrong chunking strategy ruins retrieval regardless of how good your embedding model is.

### Why chunking matters

```
problem 1 — too large:
  chunk = entire 50-page document
  embedding = blurry average of all 50 pages
  query about one specific paragraph → embedding similarity is diluted
  retrieval misses the relevant chunk entirely

problem 2 — too small:
  chunk = one sentence
  "The treatment requires careful monitoring."
  no context: monitoring of what? what treatment?
  embedding has no meaning without surrounding context
  LLM receives decontextualized fragments → poor answer
```

The right chunk size depends on your document type and query style. There is no universal answer — but there are principled strategies.

### Strategy 1 — Fixed size chunking

```
chunk_size = 512 tokens, overlap = 50 tokens

document tokens: [1....512][463....974][925....1436]...
                     ↑ overlap ↑

overlap: ensures a sentence split across a boundary appears fully in at least one chunk
```

Simple, fast, works as a baseline. Fails when semantic units (paragraphs, sections) don't align with token boundaries — you split a sentence in half and lose coherence.

### Strategy 2 — Semantic / structural chunking

Split on natural document boundaries:

```
PDFs:      split on paragraph breaks, section headers
HTML/wiki: split on <h2>, <h3> tags → each section is one chunk
code:      split on function/class boundaries (not line count)
email:     one thread = one chunk (usually fits in context window)
legal:     split on clause numbers (§ 1.1, § 1.2)
```

Better coherence than fixed size but chunks vary wildly in length — a section could be 10 tokens or 5,000 tokens. Need a max-length backstop.

### Strategy 3 — Hierarchical chunking (parent-child)

```
index two granularities simultaneously:

parent chunk: full section (~1000 tokens) — for context
child chunk:  paragraph (~100 tokens) — for retrieval precision

retrieval: search over child chunks (small → precise embedding)
context:   return parent chunk to LLM (large → full context)
```

This solves the core tension: small chunks for precise retrieval, large chunks for rich context. A common pattern in LangChain's "ParentDocumentRetriever."

### Worked numeric example — chunk size effect on retrieval

```
document: "Section 3: Optimization Methods
  3.1 Gradient Descent: Updates weights in direction of steepest descent.
      Requires careful learning rate selection. Too high → divergence.
  3.2 Adam: Adaptive learning rate optimizer. Combines momentum and RMSprop.
      Default learning rate 0.001. Robust to hyperparameter choice.
  3.3 LBFGS: Quasi-Newton method. Effective for small datasets and full-batch.
      Memory-efficient approximation of Hessian matrix."

query: "which optimizer is robust to hyperparameter choice?"

chunk strategy A — fixed 30 tokens:
  chunk 1: "Section 3: Optimization Methods 3.1 Gradient Descent: Updates"
  chunk 2: "weights in direction of steepest descent. Requires careful learning"
  chunk 3: "rate selection. Too high → divergence. 3.2 Adam: Adaptive learning"
  chunk 4: "rate optimizer. Combines momentum and RMSprop. Default learning rate"
  chunk 5: "0.001. Robust to hyperparameter choice. 3.3 LBFGS: Quasi-Newton"

  best matching chunk: chunk 4 or 5 (contains "robust to hyperparameter")
  but: neither chunk contains "Adam" — the answer is split across chunks 3/4/5
  LLM receives chunk 4: "rate optimizer. Combines momentum and RMSprop..."
  LLM cannot answer "which optimizer" — "Adam" is in chunk 3, not returned

chunk strategy B — structural (one chunk per subsection):
  chunk 1: "3.1 Gradient Descent: Updates weights in direction of steepest descent.
            Requires careful learning rate selection. Too high → divergence."
  chunk 2: "3.2 Adam: Adaptive learning rate optimizer. Combines momentum and RMSprop.
            Default learning rate 0.001. Robust to hyperparameter choice."
  chunk 3: "3.3 LBFGS: Quasi-Newton method. Effective for small datasets..."

  best matching chunk: chunk 2
  LLM receives: full Adam description including the name
  LLM answer: "Adam is robust to hyperparameter choice"  ✓
```

The chunking strategy determined whether the question was answerable at all.

---

## Stage 2 — Query processing

Raw user queries are often bad retrieval queries. Three transformations help:

### Query rewriting

```
user query:   "what did they say about the deadline?"
problem:      "they" and "deadline" are vague — who? which deadline?
rewritten:    "project deadline policy engineering team Q3"
```

Use an LLM to rewrite the query into a more explicit retrieval-friendly form before sending to the retrieval system.

### HyDE — Hypothetical Document Embeddings

This is a powerful technique that comes up frequently in interviews:

```
standard:  embed(query) → search doc embeddings
HyDE:      LLM(query) → hypothetical answer → embed(answer) → search doc embeddings
```

**Why this works:** The query "what causes type 2 diabetes?" is a short question. The documents are long descriptive passages. Their embeddings live in different parts of the vector space — there's an inherent asymmetry between question embeddings and answer embeddings.

HyDE asks the LLM to generate what the answer *might* look like — "Type 2 diabetes is caused by insulin resistance, where cells fail to respond to insulin..." — then embeds that hypothetical answer. A hypothetical answer embedding is much closer to a real answer document embedding than a question embedding is.

```
worked example:

query: "how does attention mechanism work in transformers?"

HyDE generates:
  "The attention mechanism in transformers computes queries, keys, and values.
   For each query vector, dot products with all key vectors are computed,
   scaled by √d_k, passed through softmax to get attention weights,
   then used to weight sum the value vectors..."

embed(HyDE answer) → much closer to actual transformer paper passages
                   → better retrieval than embed(original query)
```

The risk: if the LLM hallucinates in the hypothetical answer, the embedding drifts toward wrong documents. HyDE works well when the LLM has reasonable prior knowledge of the domain.

### Query decomposition

```
complex query: "compare the side effects of metformin and insulin for type 2 diabetes"

decompose into:
  sub-query 1: "metformin side effects type 2 diabetes"
  sub-query 2: "insulin side effects type 2 diabetes"

retrieve separately → merge results → LLM synthesizes comparison
```

Complex multi-part questions retrieve better when broken into focused sub-queries. Each sub-query retrieves the most relevant chunks for its specific aspect.

---

## Stage 3 — Context assembly

You have retrieved the top-k chunks. Now you must fit them into the LLM's context window intelligently.

### The lost in the middle problem

Research (Liu et al. 2023) showed that LLMs perform significantly worse when the relevant information is in the middle of a long context:

```
context order: [chunk1, chunk2, chunk3, chunk4, chunk5]
                  ↑                                ↑
              LLM pays most attention here    and here

relevant chunk at position 1 or 5: LLM finds it easily
relevant chunk at position 3:      LLM often misses it
```

**Implication for RAG:** Put the most relevant chunks at the beginning and end of the context, not in the middle. If you have 5 chunks and chunk 3 is most relevant, reorder to put it first.

### Token budget management

```
LLM context window: 128,000 tokens (Claude), 8,192 tokens (GPT-3.5)
token budget allocation:

  system prompt:    ~500 tokens   (instructions, persona, format)
  retrieved chunks: ~4,000 tokens (the actual content)
  query:            ~100 tokens
  answer buffer:    ~1,000 tokens (space for generation)
  safety margin:    ~500 tokens
  ─────────────────────────────
  total budget:     ~6,100 tokens for GPT-3.5

if top-5 chunks × 300 tokens each = 1,500 tokens → fits easily
if top-5 chunks × 1,500 tokens each = 7,500 tokens → exceeds budget
→ truncate chunks, reduce k, or summarize chunks before inserting
```

### Chunk deduplication

Multiple retrieved chunks often overlap — especially with sliding window chunking:

```
chunk 3: "...Adam optimizer uses adaptive learning rates. It combines momentum..."
chunk 7: "...combines momentum and RMSprop. Default learning rate 0.001..."

overlap: "combines momentum" appears in both

solution: deduplicate by content hash or semantic similarity before
          assembling context — don't waste tokens on repeated content
```

---

## Stage 4 — Prompt construction

The prompt is the interface between retrieval and generation. A poorly constructed prompt wastes good retrieval.

### Standard RAG prompt template

```
system:
  You are a helpful assistant. Answer the question using ONLY the provided
  context. If the answer is not in the context, say "I don't have enough
  information to answer this." Do not use prior knowledge. Cite the source
  chunk number for each claim you make.

context:
  [CHUNK 1 — source: HR Policy Manual, section 4.2]
  Employees are entitled to 20 days of annual leave per year...

  [CHUNK 2 — source: HR Policy Manual, section 4.5]
  Parental leave is granted for 16 weeks at full pay...

  [CHUNK 3 — source: Employee FAQ, updated 2024-01]
  Leave requests must be submitted at least 2 weeks in advance...

user:
  How much parental leave am I entitled to and how do I request it?
```

### Why "only use the context" matters

Without this instruction, the LLM blends retrieved context with its parametric knowledge — producing answers that are half-grounded, half-hallucinated with no way to tell which is which. The instruction forces the model to stay within the retrieved evidence, making hallucination detectable (the answer contradicts or goes beyond the chunks).

---

## Stage 5 — Hallucination control

Hallucination in RAG takes two forms:

```
type 1 — faithful but wrong retrieval:
  retrieval returned the wrong chunks
  LLM faithfully generates from wrong chunks
  answer is wrong but grounded in context
  → fix: improve retrieval quality (better chunking, hybrid search, re-ranking)

type 2 — unfaithful generation:
  retrieval returned correct chunks
  LLM generates claims not supported by the chunks
  → fix: better prompt constraints, faithfulness checking
```

### Faithfulness checking — worked example

After generation, run a faithfulness check:

```
generated answer: "Employees get 20 days annual leave and parental leave
                   is 16 weeks at 80% pay. Requests need 2 weeks notice."

check each claim against retrieved chunks:
  "20 days annual leave"    → found in chunk 1 ✓
  "parental leave 16 weeks" → found in chunk 2 ✓
  "80% pay"                 → NOT in any chunk ✗  ← hallucination
  "2 weeks notice"          → found in chunk 3 ✓

action: flag "80% pay" as unsupported, regenerate or caveat
```

Faithfulness checking can be done by:
- Rule-based NLI (Natural Language Inference) — does chunk entail the claim?
- LLM-as-judge — prompt a second LLM to verify each claim against the chunks
- RAGAS framework — automated RAG evaluation suite (common in production)

---

## Evaluation — RAGAS framework

RAGAS (RAG Assessment) is the standard evaluation framework. Four metrics:

### Metric 1 — Context Precision

```
of the retrieved chunks, what fraction were actually relevant?

context_precision = relevant_retrieved_chunks / total_retrieved_chunks
```

Measures retrieval precision — are you giving the LLM useful context or polluting it with noise?

### Metric 2 — Context Recall

```
of all information needed to answer the question, what fraction
is covered by the retrieved chunks?

context_recall = information_covered_by_chunks / total_needed_information
```

Measures retrieval recall — did you retrieve everything the LLM needs?

### Metric 3 — Faithfulness

```
of all claims in the generated answer, what fraction are
supported by the retrieved context?

faithfulness = supported_claims / total_claims_in_answer
```

Measures generation quality — is the answer grounded in the retrieved evidence?

### Metric 4 — Answer Relevance

```
how well does the generated answer address the original question?
(regardless of whether it's faithful to context)

measured by: embed(answer) similarity to embed(question)
             or LLM-as-judge scoring
```

### Worked RAGAS example

```
query: "what are the side effects of aspirin?"

retrieved chunks: 3 chunks
  chunk 1: "aspirin can cause stomach bleeding in some patients"  → relevant ✓
  chunk 2: "aspirin is derived from salicylic acid"               → not relevant ✗
  chunk 3: "common side effects include nausea and tinnitus"      → relevant ✓

context_precision = 2/3 = 0.667

information needed:
  - stomach bleeding ✓ (in chunk 1)
  - nausea ✓ (in chunk 3)
  - tinnitus ✓ (in chunk 3)
  - Reye's syndrome ✗ (not in any chunk)

context_recall = 3/4 = 0.750

generated answer: "aspirin can cause stomach bleeding, nausea, tinnitus,
                   and should not be taken with blood thinners"

claims:
  "stomach bleeding" → in chunk 1 ✓
  "nausea"           → in chunk 3 ✓
  "tinnitus"         → in chunk 3 ✓
  "not with blood thinners" → NOT in any chunk ✗ ← hallucination

faithfulness = 3/4 = 0.750
```

---

## Advanced RAG patterns

### Iterative / multi-hop retrieval

```
query: "what is the capital of the country that won the 2022 World Cup?"

naive RAG:
  retrieve for full query → may not find answer directly

iterative RAG:
  step 1: retrieve for "2022 World Cup winner" → Argentina
  step 2: retrieve for "capital of Argentina"  → Buenos Aires
  step 3: answer: Buenos Aires
```

Multi-hop questions require chaining retrievals — the output of one retrieval informs the next query. Used in systems like ReAct (Reason + Act) where the LLM decides when to retrieve and what to retrieve next.

### FLARE — Forward-Looking Active Retrieval

```
LLM generates token by token
when the LLM's confidence drops below a threshold:
  → pause generation
  → use the partial generation as a query
  → retrieve more context
  → continue generation with new context
```

Instead of retrieving once upfront, FLARE retrieves on demand as the LLM discovers it needs more information mid-generation.

### Corrective RAG (CRAG)

```
after retrieval, before generation:
  evaluate each retrieved chunk's relevance score
  if all chunks score below threshold:
    → fall back to web search (retrieval correction)
    → or signal "I don't have enough information"
  if some chunks are relevant:
    → filter to relevant chunks only
    → proceed with generation
```

Prevents the LLM from generating confidently wrong answers when retrieval fails.

---

## Why it works / why it fails

**Why it works:**
- Grounds LLM generation in retrieved evidence — dramatically reduces hallucination on factual queries
- No knowledge cutoff — retrieves fresh documents at query time
- Works on private data — no fine-tuning required, just index and retrieve
- Citations are natural — the retrieved chunks are the sources
- Scales to any corpus size independently of LLM context window

**Why it fails:**
- **Retrieval is the bottleneck** — if the right chunk isn't retrieved, the LLM cannot compensate. Garbage in, garbage out.
- **Chunking errors are silent** — a bad chunking strategy degrades quality with no obvious error signal
- **Lost in the middle** — LLMs don't attend uniformly to long contexts, so chunk ordering matters
- **Faithfulness is not guaranteed** — LLMs still hallucinate even with good context, especially when context is ambiguous or contradictory
- **Latency** — retrieval + re-ranking + LLM generation adds up. A typical RAG response takes 1–5 seconds vs. ~500ms for retrieval alone
- **Context window is finite** — can't retrieve unlimited chunks. Forces tradeoffs between breadth (more chunks, broader coverage) and depth (fewer chunks, more context per chunk)

---

## The one thing to remember

RAG is not just "retrieve then generate" — it is a pipeline with failure modes at every stage: chunking determines what can be retrieved, query processing determines what is retrieved, context assembly determines what the LLM sees, and faithfulness checking determines whether the answer is trustworthy. Getting all stages right is what makes a RAG system production-ready.

---

## Formulas used in this chapter

| Formula | Meaning |
|---------|---------|
| `context_precision = relevant_chunks / retrieved_chunks` | RAGAS: fraction of retrieved chunks that are relevant |
| `context_recall = covered_information / needed_information` | RAGAS: fraction of needed info present in retrieved chunks |
| `faithfulness = supported_claims / total_claims` | RAGAS: fraction of answer claims grounded in context |
| `HyDE: embed(LLM(q)) → ANN search` | Embed hypothetical answer instead of raw query |
| `token_budget = context_window - system_prompt - query - answer_buffer` | Available tokens for retrieved chunks |

---

## Interview Q&A

**Q1. What is the lost in the middle problem and how do you mitigate it?**

Research showed that LLMs pay most attention to content at the beginning and end of their context window, and significantly less to content in the middle. In a RAG context with 10 retrieved chunks, if the most relevant chunk is placed at position 5 (middle), the LLM is statistically less likely to use it correctly than if it were at position 1 or 10. Mitigation strategies: reorder retrieved chunks so the highest-scoring chunks appear at the beginning and end, not the middle. Use a shorter context with fewer, higher-quality chunks rather than a long context with many mediocre ones. Use a model with stronger long-context attention (Claude's attention is more uniform across context than earlier models). Evaluate context length sensitivity empirically on your specific use case.

**Q2. When would HyDE hurt retrieval quality?**

HyDE generates a hypothetical answer using the LLM's parametric knowledge before retrieval. If the LLM has incorrect or outdated knowledge about the domain, the hypothetical answer will be wrong — and its embedding will be biased toward documents supporting the wrong answer. For example, if the LLM "knows" an outdated drug dosage recommendation, HyDE will retrieve documents supporting the old dosage rather than the current guidelines. HyDE is also risky for queries about private or proprietary information the LLM has never seen — the hypothetical answer will be generic, producing generic embeddings that don't retrieve the specific private document. HyDE is best for domains where the LLM has solid prior knowledge and the vocabulary gap between queries and documents is the main problem.

**Q3. Walk through how you'd debug a RAG system that's giving wrong answers.**

Start by decomposing the pipeline into retrieval quality and generation quality separately. First, check retrieval: for a sample of failing queries, manually inspect the top-5 retrieved chunks. Are the right chunks being retrieved? If not, the problem is upstream — chunking (are semantically coherent passages split across chunks?), embedding quality (is the model domain-appropriate?), or hybrid search balance (are keyword queries missing because BM25 is underweighted?). Second, if the right chunks are retrieved but the answer is wrong, the problem is in generation. Check faithfulness: does the answer contain claims not supported by the chunks? If yes, strengthen the prompt constraints ("only use the context"). Check context assembly: is the most relevant chunk being lost in the middle? Is the context window being exceeded and chunks truncated? Third, measure RAGAS metrics on a labeled eval set to quantify which component is the bottleneck — a low context_recall score points to retrieval, a low faithfulness score points to generation.

**Q4. How does multi-hop retrieval work and when do you need it?**

Multi-hop retrieval is needed when answering a question requires information from multiple documents that aren't individually sufficient. A single retrieval step retrieves documents relevant to the surface query but may miss the intermediate steps needed to reach the answer. In multi-hop retrieval, the system iterates: retrieve documents for the initial query, read the retrieved context, formulate a follow-up query based on what was learned, retrieve again, and repeat until the answer can be synthesized. This requires an LLM in the loop to generate follow-up queries — the IR system alone cannot decompose the reasoning chain. The ReAct pattern implements this by interleaving LLM reasoning steps with retrieval actions. The cost is latency — each hop adds a retrieval round-trip and LLM call, so multi-hop RAG typically takes 3–10 seconds end-to-end. Use it only when the query complexity demands it, not as a default.

**Q5. How do you evaluate a RAG system when you have no labeled question-answer pairs?**

Several approaches. First, generate synthetic QA pairs: for each document chunk, prompt an LLM to generate 3–5 questions that the chunk answers. You now have (question, chunk, answer) triples. Use these to measure retrieval recall (does your system retrieve the source chunk for its own generated question?) and answer accuracy (does your system generate the same answer?). Second, use LLM-as-judge: for a sample of real user queries, have a strong LLM (GPT-4, Claude Opus) score the retrieved chunks for relevance and the generated answers for faithfulness and correctness on a 1–5 scale. Correlates well with human judgment at a fraction of the cost. Third, track implicit signals in production: answer acceptance rate (did the user act on the answer?), follow-up question rate (did the user ask for clarification, suggesting the answer was incomplete?), thumbs up/down if you expose that UI. Bootstrap with synthetic data, validate with LLM-as-judge, and refine with production signals.

---

Ready for your comments — what stays, what changes, what's missing?
