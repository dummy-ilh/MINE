# RAG Interview Master Notes — Module 2: Chunking Strategies

> **How to use these notes:** Chunking is the highest-leverage, lowest-cost knob in a RAG pipeline — interviewers love it because it tests whether you think about systems holistically, not just model selection. Read the Quick Summary, then the Q&A drill at the end cold.

---

## Quick Summary

Chunking is the decision of what unit of text becomes retrievable — it happens before embedding, before indexing, before retrieval. Everything downstream is constrained by it. Too large and your embeddings become diluted averages that match nothing precisely; too small and your chunks lose the context needed to be understood in isolation. The art of chunking is navigating this precision/recall tradeoff for your specific query type and corpus structure, and unlike model selection, it's cheap to iterate on — making it the right first place to look when RAG output quality is poor.

---

## 1. Why Chunking Is the Highest-Leverage Knob

### The Core Tension

> **Think of it like this:** Imagine you're building a filing system for a law library. If each "file" is an entire book, the librarian (retriever) will keep pulling out entire books when you ask one specific question — accurate occasionally, but mostly wasteful. If each "file" is a single word, the librarian will pull out cards that say "the" and "a" with no surrounding context. The right granularity is somewhere in between — and it depends on what questions people typically ask.

Every chunking decision forces a tradeoff across four dimensions:

| Too Large | Too Small |
|-----------|-----------|
| Embedding is a diluted average of multiple topics | Chunk loses surrounding context needed for standalone interpretation |
| Low retrieval precision (wrong content retrieved alongside correct) | Low recall (single coherent fact fragmented across multiple chunks) |
| Wastes LLM context window on irrelevant text | Sentence like "It increased by 40%" is useless without the referent of "it" |
| Fewer unique chunks in top-k slots | Same fact retrieved as multiple disconnected fragments |

### Why It's the First Thing to Tune

Unlike model swaps (expensive, slow, risky), chunking changes are:
- Free to experiment on (no retraining)
- Measurable with a small labelled eval set (query → known-relevant passage)
- High-impact — misconfigured chunking can make a great embedding model look terrible

**Interview framing:** "Before I touch the embedding model or the retriever, I'd audit the chunking strategy — it's the cheapest, highest-leverage variable to change."

---

## 2. Fixed-Size Chunking

### What It Is

Split text into chunks of exactly N tokens (or characters), with an overlap window of M tokens between consecutive chunks.

**Example configuration:** 512-token chunks, 50-token overlap.

```
[Token 1 ........... Token 512]
                  [Token 462 ........... Token 974]
                                     [Token 924 ........... Token 1436]
                   ←50 tok→         ←50 tok→
                   (overlap)         (overlap)
```

### Why Overlap Exists — With a Concrete Example

Imagine this sentence spans a chunk boundary:

> *"The policy, which was first introduced in the 2021 fiscal year, increased premiums by 40%."*

Without overlap:
- Chunk A ends at: `"The policy, which was first introduced in the 2021 fiscal year"`
- Chunk B starts at: `"increased premiums by 40%."`

Chunk B is now uninterpretable — "increased" has no subject. Overlap ensures this sentence appears intact in at least one chunk.

### The Hidden Cost of Overlap

> **Interview gotcha:** Many candidates mention overlap as purely beneficial. It has real costs.

**Numerical example of index bloat:**

Corpus: 1,000,000 tokens  
Chunk size: 512 tokens, Overlap: 50 tokens  
Effective stride: 512 − 50 = 462 tokens per chunk  
Number of chunks ≈ 1,000,000 / 462 ≈ **2,165 chunks**

Without overlap (stride = 512):  
Number of chunks ≈ 1,000,000 / 512 ≈ **1,953 chunks**

That's ~11% more chunks — but the real cost is that the same underlying fact now appears in up to 2 adjacent chunks. If your retriever returns top-k=5, two of those five slots might contain near-duplicate information from overlapping chunks, wasting retrieval diversity.

### When to Use Fixed-Size

- Quick prototypes and baselines
- Homogeneous, prose-heavy corpora with no special structure (articles, books)
- When ingestion simplicity matters more than retrieval precision

### Pros and Cons

| Pros | Cons |
|------|------|
| Trivial to implement | Ignores document structure entirely |
| Predictable chunk sizes | Can split mid-sentence, mid-table, mid-function |
| Predictable embedding cost | No semantic awareness |
| Good baseline to beat | Overlap inflates index size |

---

## 3. Recursive / Structure-Aware Chunking

### What It Is

> **Think of it like this:** Instead of using a ruler to cut a document at fixed intervals, you use the document's own natural seams — like tearing bread along its scored lines rather than cutting straight across the loaf.

Split hierarchically along natural document boundaries, in priority order:

```
Try splitting on \n\n (paragraph breaks) first
  ↓ If any result is still > max_size
Try splitting on \n (line breaks)
  ↓ If still > max_size
Try splitting on ". " (sentence boundaries)
  ↓ If still > max_size
Fall back to fixed-size token splitting
```

This is the actual behaviour of LangChain's `RecursiveCharacterTextSplitter` — **the production default in most real RAG stacks**. Worth naming explicitly in interviews.

### Why It's Better Than Fixed-Size

Document authors encode topic shifts through formatting. A double newline (paragraph break) usually signals a topic change. A heading signals an even bigger shift. Respecting these signals improves chunk coherence — each chunk is more likely to be about one thing.

**Example:**

Document paragraph 1: "Our refund policy for domestic orders allows returns within 30 days..."  
Document paragraph 2: "For international customers, customs duties are non-refundable..."

Fixed-size might lump both paragraphs into one chunk, creating an embedding that's an average of "domestic refunds" and "international customs." A query about domestic refunds retrieves this chunk but the LLM then has to wade through the international customs text too.

Recursive splitting respects the paragraph break and creates two clean, focused chunks.

### Still a Limitation

Structure ≠ semantics. A single long paragraph can drift across two topics. Two short paragraphs might be one continuous thought. Structure is a *proxy* for semantic coherence, not a guarantee.

---

## 4. Semantic Chunking

### What It Is

> **Think of it like this:** Instead of splitting at formatting cues, you're splitting where the *conversation changes subject* — like a skilled editor who can feel when a document transitions from one idea to the next, regardless of how it's formatted.

**Algorithm:**

1. Split document into minimal units (sentences)
2. Embed each sentence individually
3. Compute cosine similarity between each consecutive sentence pair
4. Where similarity drops sharply below a threshold → insert a chunk boundary
5. Merge sentences between boundaries into chunks

```
Sentence 1: "The kidney filters blood." 
Sentence 2: "It removes waste products via urine."       sim(1,2) = 0.82 → same topic, no break
Sentence 3: "Hypertension is a leading cause of failure." sim(2,3) = 0.71 → same topic, no break  
Sentence 4: "The liver produces bile for digestion."      sim(3,4) = 0.23 → TOPIC SHIFT → chunk boundary
```

### The Cost Model — When Semantic Chunking Is Worth It

**At ingestion, you pay for two rounds of embedding:**
- Round 1: embed every sentence to detect breakpoints (throw these away after)
- Round 2: embed the final chunks for the index

**Example cost comparison for a 1,000-page legal corpus:**

Assume 500 sentences per page, 1,000 pages = 500,000 sentences for breakpoint detection.  
At $0.00002 per 1K tokens (approx. small embedding model API cost), 500,000 × ~15 tokens avg ≈ 7.5M tokens ≈ **$0.15 extra for breakpoint detection**.

For a legal firm where a single missed clause costs thousands of dollars, this is a trivial cost. For a startup indexing Wikipedia for a chatbot, it's probably overkill.

### Threshold Sensitivity

| Threshold setting | Effect | Failure mode |
|-------------------|--------|--------------|
| Too strict (e.g. require sim > 0.9 to stay in same chunk) | Splits at any phrasing variation | Near-single-sentence chunks; loses surrounding context |
| Too lenient (e.g. require sim > 0.3 to split) | Almost never splits | Barely better than paragraph chunking |
| Well-tuned | Splits at genuine topic shifts | Requires empirical tuning per corpus |

**There is no universal threshold.** Tune empirically per corpus using a small labelled eval set.

### When to Use Semantic Chunking

Use it when:
- High-value corpus where retrieval errors are costly (legal, medical, financial)
- Documents have long, flowing paragraphs with embedded topic shifts
- Ingestion is infrequent (you don't re-chunk constantly)

Skip it when:
- High-volume, low-stakes content (FAQs, product descriptions, web articles)
- Documents already have clean structural formatting (Markdown headings, numbered sections)
- Ingestion speed is a constraint

---

## 5. Document-Specific Strategies

### The Golden Rule

> Generic chunkers applied to structured content types produce garbage. Always match the chunker to the content type.

### Tables

**Problem:** Naive chunking splits a table mid-row, or worse, separates the header row from all data rows. A cell value like "32.4%" is meaningless without the column header ("Q3 Revenue Growth") and row label ("EMEA").

**Fix A — Atomic table chunks (small tables):** Treat the entire table as one chunk. Serialize it to markdown or a structured text format. Keep it together.

**Fix B — Header-repeated chunks (large tables):** If the table is too large for one chunk, split by logical row groups and **repeat the header row into every chunk**:

```
Chunk 1:
  | Region | Q3 Revenue | YoY Growth |
  | EMEA   | $1.2B      | 32.4%      |
  | APAC   | $0.8B      | 18.1%      |

Chunk 2:
  | Region | Q3 Revenue | YoY Growth |   ← header repeated
  | AMER   | $2.1B      | 11.3%      |
  | LATAM  | $0.3B      | 41.2%      |
```

Each chunk is now self-interpreting.

### Code

**Problem:** Splitting mid-function is catastrophic. A function body without its signature (the `def my_function(arg1, arg2):` line) is nearly un-embeddable — the signature carries the semantics, not the body alone.

**Fix:** Use AST-aware (Abstract Syntax Tree) splitters that respect language structure. Libraries like `tree-sitter` parse code into a syntax tree; you chunk at function, class, or module boundaries, not at character positions.

```python
# WRONG: fixed-size splits here
def calculate_revenue(units, price_per_unit,  # ← chunk 1 ends here
    discount_rate):                             # ← chunk 2 starts mid-signature
    return units * price_per_unit * (1 - discount_rate)

# RIGHT: chunk at function boundary
# Chunk = entire function (signature + body + docstring)
def calculate_revenue(units, price_per_unit, discount_rate):
    """Compute net revenue after discount."""
    return units * price_per_unit * (1 - discount_rate)
```

### PDFs with Multi-Column Layout

**Problem:** Naive PDF text extraction reads left-to-right, top-to-bottom across columns — producing this interleaving for a two-column article:

```
WRONG extraction:
"Column A line 1 Column B line 1 Column A line 2 Column B line 2..."

Correct:
"Column A line 1 Column A line 2 ... [column break] Column B line 1 Column B line 2..."
```

Any chunking strategy applied after garbage extraction is wasted effort. The fix is **before** chunking: use layout-aware PDF extraction (PDFPlumber, AWS Textract, Azure Document Intelligence) that detects column bounding boxes before assembling text.

> **Principle:** Garbage extraction makes any chunking strategy moot. Fix the extraction layer first.

### Markdown / Structured Docs

**Strategy:** Chunk along heading hierarchy. Each chunk inherits its document location:

```
Chunk metadata:
  text: "For international customers, customs duties are non-refundable..."
  breadcrumb: "Returns Policy > International Orders > Customs & Duties"
```

This inherited breadcrumb enables both:
1. Metadata-filtered retrieval ("only return chunks from the Returns Policy section")
2. Better standalone interpretability (the chunk's heading gives context without needing surrounding text)

---

## 6. Small-to-Big / Parent-Child Chunking

### The Core Insight

> **Think of it like this:** When you search a book index, you look up the fine-grained term (small granularity for matching). But when you sit down to read, you read the full section, not just the index entry (large granularity for understanding). Parent-child chunking builds exactly this two-level system.

**The fundamental tension in chunking:**
- Small chunks = precise retrieval matches (embedding isn't diluted)
- Large chunks = enough context for the LLM to generate a good answer

These two needs conflict if you have only one chunk size. Parent-child decouples them.

### How It Works

**Ingestion:**
```
Document
  ├── Section 1 (parent chunk — large, ~500 tokens)
  │     ├── Sentence 1 (child chunk — small, ~30 tokens) → embedded & indexed
  │     ├── Sentence 2 (child chunk)                     → embedded & indexed
  │     └── Sentence 3 (child chunk)                     → embedded & indexed
  └── Section 2 (parent chunk)
        ├── Sentence 4 (child chunk)                     → embedded & indexed
        └── ...
```

**Retrieval:**
```
Query: "What is the refund timeline for international orders?"
  ↓
Retriever finds: Child chunk (sentence 3) — high similarity match
  ↓
System fetches: Parent chunk (Section 1) — the full section containing sentence 3
  ↓
LLM receives: Full section context (not just the matched sentence)
```

### The Two Variants

**Sentence-window retrieval:** Index individual sentences. When a sentence is retrieved, expand to ±k surrounding sentences (e.g. ±2 sentences, giving a 5-sentence window). Simpler to implement than full parent-child, nearly as effective for many corpora.

**Hierarchical indexing:** Index multiple levels simultaneously — section summaries AND paragraph chunks AND sentence chunks. Route different query types to different levels:
- "Summarise the company's Q3 strategy" → section-level chunks (broad context)
- "What was the EMEA revenue in Q3?" → sentence-level chunks (specific fact)

### Why This Pattern Keeps Coming Up in Interviews

It directly solves the core tension stated in section 2.1 — and it's a concrete, implementable answer, not a theoretical statement about tradeoffs. When an interviewer asks "how do you balance retrieval precision with generation context quality?", parent-child is the answer.

---

## 7. Metadata Enrichment Per Chunk

### What It Is

Beyond storing raw chunk text, attach structured fields that can be used for filtering, re-ranking, or improving retrievability.

### The Four Types Worth Knowing

**Type 1: Breadcrumb / Section Hierarchy**

```json
{
  "text": "International returns must be initiated within 14 days of delivery.",
  "source": "policy_manual_v3.pdf",
  "section": "Returns Policy > International Orders",
  "page": 47
}
```

Enables filtered retrieval: "search only within the Returns Policy section." Without this, a query about returns might retrieve unrelated content that happens to use the word "return."

**Type 2: Auto-Generated Chunk Summaries**

For noisy chunks (dense tables, boilerplate-heavy regulatory text), generate a clean LLM summary and embed the summary instead of (or alongside) the raw text.

*Why:* A dense earnings table embedded as raw text ("Revenue 1,234,567 1,189,234 Cost of Goods 456,789...") produces a poor embedding. A summary ("Q3 2024 revenue grew 3.8% YoY to $1.23B, driven by APAC expansion") embeds far better because it's natural language with semantic content.

**Type 3: Hypothetical Questions (HyQ)**

Generate 3–5 synthetic questions that this chunk would answer, embed the questions (not the chunk text) in the index.

```
Chunk text: "For international customers, customs duties are non-refundable and will not be credited."

Synthetic questions embedded:
- "Are customs fees refundable for international orders?"
- "What happens to import duties if I return an item from abroad?"
- "Can I get a refund on customs charges?"
```

*Why this works:* Real user queries are questions. Chunk text is declarative prose. The distributional gap between query space and document space is the core challenge of asymmetric retrieval (Module 1, section 1.4). Embedding synthetic questions instead of raw text closes this gap at ingestion time. This is the ingestion-time cousin of HyDE (Hypothetical Document Embeddings), which is the query-time version of the same idea — previewed here, covered in Module 4.

**Type 4: Source / Temporal / Access Metadata**

```json
{
  "last_updated": "2024-09-15",
  "access_level": "internal",
  "document_type": "policy",
  "jurisdiction": "EU"
}
```

Not for retrieval quality directly, but for filtering:
- "Only search documents updated in the last 6 months" (recency filtering)
- "Only return results this user has permission to see" (access control — Module 9)
- "Only return EU-jurisdiction policy documents" (namespace filtering)

---

## 8. The Empirical Tuning Loop

### There Is No Universal Correct Chunk Size

The right chunk size depends on:

| Factor | Effect on optimal chunk size |
|--------|------------------------------|
| **Query type** | Fact-lookup ("what is X?") → smaller chunks. Synthesis ("summarise Q3 strategy") → larger chunks |
| **Embedding model's effective context** | Most models degrade well before their stated max — e.g. a model with a 512-token limit may produce better embeddings at 200–300 tokens |
| **LLM context budget** | If top-k=10 and chunk size=512, you're consuming 5,120 tokens of context per query. Larger chunks force smaller k or bust the context window |
| **Corpus structure** | Short FAQ entries → small chunks natural. Long regulatory documents → larger chunks needed for coherence |

### How to Tune Systematically (State This in Interviews)

> Saying "I'd tune chunk size empirically with an eval set" distinguishes you from candidates who treat it as guesswork.

**Step 1:** Assemble a small labelled eval set — 50–200 (query, relevant passage) pairs. Can be sampled from user logs, manually annotated, or synthetically generated.

**Step 2:** Define your metric. For retrieval: Recall@k (what fraction of relevant passages appear in the top-k retrieved chunks). For end-to-end: answer correctness on the eval set.

**Step 3:** Grid search over (chunk_size, overlap) pairs:

```
chunk_sizes = [128, 256, 512, 1024]
overlaps    = [0, 32, 64, 128]

for chunk_size in chunk_sizes:
    for overlap in overlaps:
        rebuild_index(chunk_size, overlap)
        recall = evaluate_recall_at_k(eval_set, k=5)
        log(chunk_size, overlap, recall)
```

**Step 4:** Plot Recall@k vs chunk_size. Find the knee of the curve — recall usually improves steeply then plateaus. The knee is your sweet spot; going past it just wastes context budget.

**Step 5:** Re-validate after any embedding model change. The optimal chunk size is model-dependent — a model with better long-context representation may benefit from larger chunks than one that degrades past 200 tokens.

---

## The Chunking Strategy Decision Tree

Use this to pick a strategy in an interview when given a specific scenario:

```
What is the primary content type?
│
├─ Code → AST-aware chunking (tree-sitter, function/class boundaries)
│
├─ Tables → Atomic table chunks OR header-repeated fixed splits
│
├─ PDFs with layout complexity → Layout-aware extraction FIRST, then...
│     └─ proceed to prose/table branches
│
└─ Prose / documents
      │
      ├─ Well-structured (Markdown headings, clear sections)?
      │     └─ Recursive / structure-aware chunking
      │
      ├─ High-value corpus, long flowing paragraphs, budget for extra compute?
      │     └─ Semantic chunking
      │
      └─ Need both retrieval precision AND generation context?
            └─ Parent-child / small-to-big chunking
                 (always a valid answer for complex corpora)
```

---

## Interview Q&A Drill

---

**Q: You increased chunk size and retrieval recall went up, but answer quality went down. Explain.**

A: Two things happen when chunk size increases. First, each chunk is more likely to *contain* the answer (higher recall — the target text is less likely to be split across chunks). Second, the embedding of a larger chunk is a vector average over more content, which dilutes precision — the embedding doesn't pinpoint the specific relevant section as sharply, and more irrelevant text gets bundled into the retrieved context. The LLM then receives both the relevant passage and surrounding noise, increasing the chance it synthesises incorrectly from the irrelevant content.

The fix is usually parent-child chunking: index small chunks for retrieval precision, expand to the parent context for generation — getting high recall from the small-chunk match without the diluted-embedding problem, and giving the LLM sufficient context without forcing it to wade through noise.

---

**Q: When would you choose semantic chunking over recursive structure-aware chunking?**

A: When retrieval precision materially affects real-world outcomes and the corpus warrants the extra ingestion cost. Concrete cases: legal contracts where a missed clause costs money, medical records where misattributing a symptom to the wrong condition causes harm, financial filings where one number being in the wrong context changes an investment decision.

For high-volume, lower-stakes content — general-purpose Q&A chatbots, product documentation, FAQs — recursive structure-aware chunking is sufficient and avoids the two-round embedding cost at ingestion.

---

**Q: How do you chunk a 50-page PDF containing both prose and financial tables?**

A: Treat it as two separate problems. First, use layout-aware PDF extraction (e.g. PDFPlumber or a document intelligence service) to separate prose regions from table regions before chunking — naive extraction will interleave multi-column text incorrectly and destroy table structure. Then: chunk prose with recursive structure-aware splitting. Chunk tables as atomic units where feasible (full table as one chunk); where tables span multiple pages, split by logical row groups and repeat the header row into each chunk so every chunk is self-interpreting. Never apply a single generic text splitter to a mixed-content PDF.

---

**Q: What's the actual difference between "chunking" and "indexing," and why do candidates conflate them?**

A: Chunking is a *data* decision — what units of text to create, resolved by understanding the corpus, query types, and retrieval precision/recall tradeoffs. Indexing is a *systems* decision — how to store and search those units at scale, resolved by understanding query latency, memory budget, and update frequency.

They're conflated because they happen back-to-back in the ingestion pipeline. But the reasoning is completely different: chunking is tuned with labelled eval sets and corpus analysis; indexing is tuned with benchmark latency tests and memory profiling. A great chunking strategy on a poorly-chosen index retrieves the right content slowly; a great index over badly-chunked data retrieves the wrong content quickly.

---

**Q: Why does parent-child chunking help, and when would it fail?**

A: It helps because it decouples the retrieval unit (small, precise) from the generation unit (large, contextually complete), solving the fundamental chunk-size tradeoff. It fails when: (1) parent chunks are themselves too large or too noisy, so fetching the parent gives the LLM too much irrelevant content alongside the relevant sentence; (2) the document structure doesn't support clear parent-child relationships (e.g. continuous prose with no section boundaries — hard to define what the "parent" of a sentence is); (3) the parent contains multiple semantically distinct topics, defeating the purpose of expanding for context. In those cases, sentence-window retrieval (expand to ±k sentences around the matched sentence) is a simpler and more robust alternative.

---

**Q: A user asks a synthesis question: "Summarise our company's Q3 strategy across all business units." How should your chunking strategy handle this?**

A: Synthesis queries are inherently multi-chunk — the answer requires assembling information from many different parts of the corpus. Small chunks hurt here because the answer isn't in one place; you need broad coverage across documents. Two approaches: (1) use larger chunks or hierarchical indexing with section-level summaries specifically for synthesis queries — if you can classify the query as synthesis vs fact-lookup at query time, route to the appropriate index level; (2) generate and index document-level or section-level summaries as metadata at ingestion time, then retrieve on those summaries for synthesis queries and on fine-grained chunks for fact-lookup queries. The underlying principle: match the retrieval granularity to the query type.

---

## Key Gotchas Summary

| Gotcha | Correct understanding |
|--------|----------------------|
| "Overlap is always beneficial" | Overlap inflates index size and wastes top-k slots with near-duplicate chunks |
| "Bigger chunks always improve context" | Bigger chunks dilute embeddings and pack irrelevant noise into LLM context |
| "Fixed-size chunking is fine for everything" | Catastrophically wrong for tables, code, and multi-column PDFs |
| "Semantic chunking is always worth it" | Only for high-value corpora; expensive at ingestion scale |
| "Chunk size is a fixed constant" | Model-dependent and query-type-dependent; tune empirically per corpus |
| "Better extraction doesn't matter once you have chunking" | Garbage extraction makes any chunking strategy irrelevant — fix extraction first |
| "Chunking and indexing are the same thing" | Different problem types: chunking is a data/eval problem, indexing is a systems/scale problem |

---

*Next: Module 3 — Indexing & Vector Databases*
