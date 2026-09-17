# RAG Interview Prep — Day 3
## Chunking Strategies

---

## 🚀 Quick Summary

Chunking is the process of splitting long documents into smaller retrievable units *before* they get embedded and indexed (Day 2), and it is arguably the highest-leverage, most-underrated decision in the entire RAG pipeline — a bad chunking strategy silently caps retrieval quality no matter how good your embedding model or search algorithm is. Unlike swapping the embedding model or the LLM (expensive, slow, sometimes risky), chunking changes are **free to experiment with** (no retraining), **measurable** with a small labelled eval set, and **high-impact** — misconfigured chunking can make a great embedding model look terrible. In production, teams often find that switching chunking strategy improves retrieval accuracy more than switching embedding models.

**Think of it like cutting a cake for a buffet.** Serve the whole cake as one slab and nobody can grab a manageable piece — that's "no chunking," where a whole document gets embedded as one blurry-average vector. Cut it into crumbs and each piece is meaningless alone — that's chunks too small, stripped of context. The goal is slices: big enough to be a complete, coherent idea, small enough that grabbing one or two actually answers the question.

**Interview framing to use out loud:** *"Before I touch the embedding model or the retriever, I'd audit the chunking strategy — it's the cheapest, highest-leverage variable to change."*

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Chunk** | A retrievable unit of text — the thing that actually gets embedded, indexed, and returned by search |
| **Chunk size** | The target length of a chunk, in tokens (not characters — token count is what the embedding model and LLM context window actually consume) |
| **Overlap** | Shared content between consecutive chunks, meant to reduce the chance of splitting an idea across a chunk boundary |
| **Stride** | How far forward the next chunk starts = chunk_size − overlap |
| **Recursive splitting** | Trying natural boundaries first (paragraph → sentence → word), falling back to smaller units only when needed. Implemented in production tools like LangChain's `RecursiveCharacterTextSplitter` |
| **Semantic chunking** | Using embeddings to detect topic-shift boundaries and splitting there, instead of at a fixed size |
| **Structure-aware chunking** | Respecting a document's native structure (markdown headers, code blocks, table rows) as chunk boundaries |
| **Small-to-big / parent-child retrieval** | Searching over small chunks for precision, but returning their larger parent context to the generator for completeness |
| **AST-aware chunking** | Parsing code into an Abstract Syntax Tree (e.g., via `tree-sitter`) and chunking at function/class boundaries rather than raw character counts |
| **HyQ (Hypothetical Questions)** | Generating synthetic questions a chunk would answer, and embedding those questions instead of (or alongside) the raw chunk text, to close the query-vs-document distributional gap at ingestion time |

---

# PHASE 1 — Intuition & Visual Map

## Why chunking is the single highest-leverage lever in the whole pipeline

Everything downstream — embedding quality, retrieval precision, generation faithfulness — operates *on* chunks, not on raw documents. If a chunk contains one clean idea, its embedding is a sharp, accurate point on the meaning-map from Day 2. If a chunk crams three unrelated ideas together, its embedding becomes a blurred average of all three — sitting in a mediocre, ambiguous location that isn't a great match for a query about *any* of the three ideas individually.

```
   GOOD CHUNK (one clean idea)              BAD CHUNK (three ideas mashed together)

   "Return window: 14 days from             "Return window: 14 days from purchase.
    purchase for AirPods Pro."                Battery lasts 6 hours. Case is IPX4
                                               water resistant."
        │                                            │
        ▼                                            ▼
   ┌─────────┐                              ┌─────────┐
   │  sharp   │ ← clear point on             │ blurry   │ ← smeared point,
   │  vector  │    the meaning map            │ vector   │    ambiguous location
   └─────────┘                              └─────────┘
```

**Another framing:** think of an LLM as a brilliant expert with a *very short memory span* — you can't hand it a whole 400-page report, you have to hand it the right few paragraphs. Or think of it as a library filing system: file by whole book, and the librarian (retriever) keeps pulling out entire books for one specific question; file by single word, and the librarian pulls out index cards with no context. The right granularity sits in between, and it depends on what questions people actually ask.

## The Core Tension — Four Dimensions

| Too Large | Too Small |
|---|---|
| Embedding is a diluted average of multiple topics | Chunk loses surrounding context needed for standalone interpretation |
| Low retrieval precision (wrong content retrieved alongside correct) | Low recall (a coherent fact fragmented across multiple chunks) |
| Wastes LLM context window on irrelevant text | A sentence like "It increased by 40%" is useless without the referent of "it" |
| Fewer unique chunks fit in top-k slots | Same fact retrieved as multiple disconnected fragments |

## When to use each strategy

- ✅ **Fixed-size** — fast prototyping, homogeneous unstructured text with no meaningful structure (transcripts with no paragraph breaks). ❌ Not for legal clauses, step-by-step instructions, tables, or code, where cutting mid-unit is costly.
- ✅ **Recursive** — the sensible production default for most real-world prose; respects structure without the cost of semantic chunking.
- ✅ **Semantic** — high-value corpora where topical coherence really matters (legal, medical, financial) and you can afford extra embedding calls at ingestion. ❌ Not worth it for huge, low-stakes corpora, or documents that already have clean structural formatting (structure-aware is free and just as good there).
- ✅ **Structure-aware** — anything with native structure: markdown docs, code, tables, API references, contracts. Ignoring structure here is a very avoidable mistake.

### 🎯 Interview Gotcha
> "Isn't chunking a solved problem — just split every 500 tokens?"

No. Fixed-size splitting is a *baseline*, not a *solution*. Interviewers want to hear that chunking is content-dependent and query-type-dependent, and requires evaluation — not a fire-and-forget default.

> "Which single strategy would you pick for a production RAG system?"

The trap is picking just one. The strong answer is a **hybrid**: structure-aware chunking as the outer layer, recursive chunking as the fallback for oversized sections, semantic chunking reserved for high-value unstructured corpora where the ingestion cost is justified — and parent-child chunking layered on top to resolve the precision-vs-context tension no single chunk size can solve alone.

---

# PHASE 2 — Deep Dive: Strategies, Math, and Production Patterns

## Notation table

| Symbol | Meaning |
|---|---|
| `L` | Document length (tokens) |
| `C` | Chunk size (tokens) |
| `O` | Overlap (tokens) |
| `S` | Stride = C − O |
| `N` | Number of resulting chunks |

## Stride, Chunk Count, and the Cost of Overlap

```
S = C - O
N ≈ ⌈(L - O) / S⌉
```

**Plain English:** Stride is how far forward the "window" moves for each new chunk. If chunk size is 300 and overlap is 50, each new chunk only advances 250 tokens past where the previous one started — the 50-token overlap region gets indexed twice.

**Worked example — three overlap settings on the same document (L = 3000, C = 300):**

| Overlap setting | Stride | Chunks (N) | Total indexed tokens | Redundancy | Trade-off |
|---|---|---|---|---|---|
| 0 tokens | 300 | 10 | 3000 | 0% | Cheapest, highest risk of boundary splits |
| 50 tokens (~17%) | 250 | 12 | 3600 | +20% | Good default — most boundary-split risk removed at modest cost |
| 150 tokens (50%) | 150 | 19 | 5700 | +90% | Rarely worth it — diminishing returns past a certain overlap ratio |

**Why overlap exists — a concrete failure mode without it.** Take a sentence spanning a chunk boundary: *"The policy, which was first introduced in the 2021 fiscal year, increased premiums by 40%."* Without overlap, chunk A might end at "...first introduced in the 2021 fiscal year" and chunk B starts at "increased premiums by 40%." — now uninterpretable, since "increased" has no subject. Overlap ensures the full sentence survives intact in at least one chunk.

**Why more overlap isn't free — the index-bloat math.** For a 1,000,000-token corpus at chunk size 512 with overlap 50: stride = 462, so you get ≈2,165 chunks, vs. ≈1,953 chunks with zero overlap — about **11% more chunks**. The real cost isn't just storage: the *same underlying fact* can now appear in two adjacent chunks. If top-k = 5, two of those five retrieval slots can be near-duplicates from overlapping chunks, wasting retrieval diversity instead of surfacing five distinct pieces of evidence. This is the concrete number to have ready when someone claims "overlap is purely beneficial."

**Why 10–20% overlap is the standard range:** below ~10%, facts straddling a boundary are frequently lost from both resulting chunks; above ~20%, index size and cost grow roughly linearly while recall gains flatten — you're mostly paying for near-duplicates occupying top-k slots. 10–20% is the empirical "knee of the curve" most teams converge on.

---

## Strategy 1 — Fixed-Size Chunking

**Think of it like** slicing a loaf of bread with a ruler, every 2 centimeters, regardless of whether you cut through the crust, a raisin, or a slice of cheese sitting on top.

**How it works:** pick a chunk size and overlap in tokens, slide a window across the text, cut a chunk every `chunk_size - overlap` tokens, store each chunk with start/end offsets as metadata.

```python
def fixed_size_chunk(text, chunk_size=512, overlap=50, tokenizer=None):
    tokens = tokenizer.encode(text) if tokenizer else text.split()
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + chunk_size]
        if not chunk_tokens:
            break
        chunk_text = (tokenizer.decode(chunk_tokens)
                      if tokenizer else " ".join(chunk_tokens))
        chunks.append({
            "text": chunk_text,
            "start_token": start,
            "end_token": start + len(chunk_tokens)
        })
    return chunks
```

**When to use / not:** homogeneous prose-heavy corpora, quick baselines, when ingestion simplicity matters more than precision. Avoid for structured documents (contracts, tables, code) where mid-sentence or mid-row cuts destroy meaning.

**Common mistakes:** chunking by raw character count instead of tokens; assuming more overlap is strictly better (see index-bloat math above).

---

## Strategy 2 — Recursive / Sentence-Based Chunking

**Think of it like** tearing bread along its scored lines rather than cutting straight across with a ruler — using the document's own natural seams instead of a fixed interval.

**How it works:** define an ordered list of separators, most preferred first (`["\n\n", "\n", ". ", " ", ""]`); try splitting on paragraph breaks first; recursively fall back to the next separator only for pieces still too large; recombine small adjacent pieces up to `chunk_size` with overlap. This is the actual behavior of LangChain's `RecursiveCharacterTextSplitter` — **the production default in most real RAG stacks**, worth naming explicitly in interviews.

```python
def recursive_chunk(text, chunk_size=512, overlap=50,
                     separators=("\n\n", "\n", ". ", " ", "")):
    def split_text(text, seps):
        if not seps:
            return [text]
        sep, rest = seps[0], seps[1:]
        pieces = text.split(sep) if sep else list(text)
        result = []
        for piece in pieces:
            if len(piece) > chunk_size and rest:
                result.extend(split_text(piece, rest))
            else:
                result.append(piece)
        return result

    raw_pieces = split_text(text, separators)
    chunks, current = [], ""
    for piece in raw_pieces:
        candidate = (current + " " + piece).strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    overlapped = []
    for i, c in enumerate(chunks):
        prefix = chunks[i - 1][-overlap:] if i > 0 else ""
        overlapped.append((prefix + " " + c).strip())
    return overlapped
```

**Why it beats fixed-size:** document authors encode topic shifts through formatting — a double newline usually signals a topic change. Fixed-size chunking might lump two unrelated paragraphs (e.g., "domestic refunds" and "international customs") into one chunk, producing an embedding that's an average of both topics. Recursive splitting respects the paragraph break instead.

**Still a limitation:** structure ≠ semantics. A single long paragraph can drift across two topics; two short paragraphs might be one continuous thought. Structure is a *proxy* for semantic coherence, not a guarantee — the gap that semantic chunking exists to close.

---

## Strategy 3 — Semantic Chunking

**Think of it like** reading with a highlighter and starting a new color every time the *topic* shifts, not every time you hit a word-count limit.

**How it works:** split into sentences → embed each sentence individually → compute cosine similarity between each consecutive sentence pair → where similarity drops sharply below a threshold, insert a chunk boundary → merge sentences between boundaries into a chunk.

```
Sentence 1: "AirPods Pro have active noise cancellation."
Sentence 2: "The noise cancellation adapts in real time to your environment."
   → cosine similarity(S1, S2) = 0.91  (high — same topic, no break)

Sentence 3: "Return window for AirPods Pro is 14 days from purchase."
   → cosine similarity(S2, S3) = 0.38  (low — topic shifted, INSERT BOUNDARY HERE)
```

```python
import numpy as np

def semantic_chunk(sentences, embed_fn, threshold_percentile=90, max_chunk_sentences=10):
    embeddings = [embed_fn(s) for s in sentences]
    sims = [
        np.dot(embeddings[i], embeddings[i + 1]) /
        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]
    distances = [1 - s for s in sims]  # a LOW similarity marks a breakpoint
    breakpoint_threshold = np.percentile(distances, threshold_percentile)

    chunks, current = [], [sentences[0]]
    for i, d in enumerate(distances):
        if d > breakpoint_threshold or len(current) >= max_chunk_sentences:
            chunks.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunks.append(" ".join(current))
    return chunks
```

**Cost model — is it worth it?** At ingestion you pay for two rounds of embedding: one to embed every sentence for breakpoint detection (thrown away afterward), one to embed the final chunks for the index. Worked example: a 1,000-page legal corpus at ~500 sentences/page = 500,000 sentences; at a small embedding model's rough pricing, that breakpoint-detection pass costs roughly **$0.15 extra** — trivial for a legal firm where a missed clause costs thousands of dollars, probably overkill for a startup indexing Wikipedia for a low-stakes chatbot. Naming this cost/benefit judgment explicitly is what separates a strong interview answer.

**Threshold sensitivity:** there is no universal similarity threshold — too strict (e.g., require sim > 0.9 to stay together) splits at any phrasing variation, producing near-single-sentence chunks; too lenient almost never splits, barely beating paragraph chunking. In practice the threshold is set via a **percentile** of the observed similarity/distance distribution *within a document* (e.g., "the steepest 10% of similarity drops are breakpoints"), tuned empirically per corpus, because absolute similarity scales differ across embedding models and domains.

---

## Strategy 4 — Structure-Aware Chunking

**Think of it like** cutting a cake along the lines already drawn on it — headings, sections, table boundaries are natural divisions the author already made.

**How it works:** parse the document into its structural tree (headings, paragraphs, lists, tables, code blocks — via `unstructured`, `LlamaParse`, HTML/Markdown parsers, or a PDF layout model) → treat each structural unit as a candidate chunk → if too large, recursively split *within* that unit's own boundaries (never merge across sibling sections) → if too small, merge with an adjacent sibling under the same parent heading → attach structural metadata to every chunk (`{"h1": "...", "h2": "...", "section_path": "..."}`) — gold for retrieval and citations.

```python
import re

def structure_aware_chunk(markdown_text, max_chunk_size=800):
    pattern = r'(?=^#{1,3}\s)'
    sections = re.split(pattern, markdown_text, flags=re.MULTILINE)

    chunks = []
    heading_stack = {}
    for section in sections:
        if not section.strip():
            continue
        heading_match = re.match(r'^(#{1,3})\s+(.*)', section)
        level, title = (len(heading_match.group(1)), heading_match.group(2)) \
            if heading_match else (None, None)
        if level:
            heading_stack[level] = title
            for deeper in [l for l in heading_stack if l > level]:
                del heading_stack[deeper]

        if len(section) <= max_chunk_size:
            chunks.append({"text": section, "path": dict(heading_stack)})
        else:
            for i in range(0, len(section), max_chunk_size):
                sub = section[i:i + max_chunk_size]
                chunks.append({"text": sub, "path": dict(heading_stack)})
    return chunks
```

**What goes wrong if ignored, by content type:**

| Content type | Structural signal to respect | What goes wrong if ignored |
|---|---|---|
| Markdown / docs | Headers, bullet lists | A boundary lands mid-list, separating a heading from its content |
| Code | Function/class boundaries | Splitting a function produces unmatched brackets, no coherent meaning |
| Tables | Row/column structure | A mid-table split produces header-less numbers — uninterpretable |
| Legal contracts | Clause/section numbering | Splitting mid-clause separates an obligation from its conditions |

**Practical technique:** prepend the relevant header path (e.g., `"Support > AirPods > Returns"`) to every chunk under that header, even if the header text is physically far from the chunk — gives every chunk standalone context even after being pulled out of its original document.

---

## Head-to-Head Comparison

| Strategy | Coherence | Cost | Speed | Handles Structure | Typical Use |
|---|---|---|---|---|---|
| Fixed-size | ❌ Low | 💰 Free | ⚡⚡⚡ Fastest | ❌ No | Quick baseline, uniform text |
| Recursive/sentence | ✅ Good | 💰 Free | ⚡⚡ Fast | 🟡 Partial | Default production choice |
| Semantic | ✅✅ Best | 💰💰 Higher (2 embed passes) | 🐢 Slow | 🟡 Partial | High-value prose, topic-shift-heavy docs |
| Structure-aware | ✅✅ Best (if structure exists) | 💰 Free–low | 🟡 Medium | ✅✅ Yes | Contracts, manuals, reports, textbooks |

---

## Small-to-Big (Parent-Child) Retrieval — Resolving the Core Chunking Tension

**Think of it like** a library's card catalog: when you *search*, you look up a fine-grained index card (small granularity, precise matching); when you *read*, you pull the full book off the shelf, not just the card (large granularity, full context).

**The tension:** small chunks are better for search precision (a sharp, focused embedding matches a specific query well), but worse for generation (too little surrounding context for a complete answer). Large chunks are the opposite. Picking one chunk size forces you to accept both sides of that trade-off.

**The pattern:**
```
Document
  ├── Section 1 (parent chunk — large, ~500 tokens)
  │     ├── Sentence 1 (child chunk — small) → embedded & indexed
  │     ├── Sentence 2 (child chunk)         → embedded & indexed
  │     └── Sentence 3 (child chunk)         → embedded & indexed
  └── Section 2 (parent chunk)
        └── ...
```
At query time: search over small child chunks (sharp matching) → once you know which matched, fetch their larger parent chunks → feed the parent chunks to the generator (full context).

**Two variants worth naming:**
- **Sentence-window retrieval** — index individual sentences; when one is retrieved, expand to ±k surrounding sentences (e.g., ±2, a 5-sentence window). Simpler than full parent-child, nearly as effective for many corpora.
- **Hierarchical indexing** — index multiple levels simultaneously (section summaries + paragraph chunks + sentence chunks), then route different query types to different levels: "Summarize the company's Q3 strategy" → section-level chunks; "What was EMEA revenue in Q3?" → sentence-level chunks.

**When parent-child fails:** parent chunks are themselves too large/noisy (fetching the parent buries the relevant sentence in irrelevant content); the document has no clear section boundaries to define a "parent" (continuous prose); the parent spans multiple distinct topics, defeating the purpose of expanding for context. Sentence-window retrieval is the simpler, more robust fallback in those cases.

**Why this keeps coming up in interviews:** it's a concrete, implementable answer — not a theoretical statement about trade-offs — to "how do you balance retrieval precision with generation context quality?"

---

## Industry-Standard Chunk Sizes

**Think of chunk size like a camera's zoom level.** Zoom in too far (tiny chunks) and you see one pixel with no context. Zoom out too far (huge chunks) and the one detail you needed is buried in a landscape shot.

| Use Case | Typical Chunk Size | Why |
|---|---|---|
| Q&A over short FAQs / support docs | 128–256 tokens | Answers are short and localized |
| General knowledge base RAG | 256–512 tokens | Balances context vs. precision — most common default |
| Long-form technical/legal docs | 512–1024 tokens | Needs more surrounding context to preserve meaning |
| Code retrieval | 1 function/class (AST-bounded) | Natural structural unit is the function, not a token count |
| Conversational/chat memory | 1 turn or a small window of turns | Turns are the natural semantic unit |

**What actually determines optimal chunk size:**

| Factor | Effect on optimal chunk size |
|---|---|
| Query type | Fact-lookup ("what is X?") → smaller chunks. Synthesis ("summarize Q3 strategy") → larger chunks or hierarchical indexing |
| Embedding model's effective context | Most models degrade well before their stated max — a 512-token-limit model may embed best at 200–300 tokens |
| LLM context budget | top-k=10 at chunk_size=512 consumes 5,120 tokens/query — larger chunks force a smaller k or risk busting the context window |
| Corpus structure | Short FAQ entries → small chunks natural; long regulatory documents → larger chunks needed |

256–512 tokens with 10–20% overlap is the most common production **starting point**, but should always be treated as a hyperparameter to tune (see Evaluation section below), never copied from a blog post.

**Common production patterns:** parent-child chunking as the default answer to the precision-vs-context tension; metadata-enriched chunks (source, section path, page number, timestamp); re-chunking from raw source (never from already-chunked text) when hyperparameters change, to avoid compounding information loss.

**Common anti-patterns:** chunking after lossy text extraction (table structure destroyed before chunking starts); one-size-fits-all chunk size across wildly different document types; treating chunk size as a fixed constant instead of a tunable hyperparameter; never re-evaluating chunk size after launch or after an embedding model swap.

---

## Multi-Modal & Document-Specific Chunking

**Think of a PDF like a stage play, not a script.** A script just gives you words in order; a stage play has a visual layout — positions, structure — that carries meaning. Naive text extraction reads only the dialogue and loses the blocking. **Generic chunkers applied to structured content types produce garbage — always match the chunker to the content type.**

**The golden workflow:**
```
PDF Input
   │
   ▼
1) LAYOUT ANALYSIS — detect text blocks, tables, images, headers/footers, columns
   (tools: LayoutParser, Unstructured, Azure Document Intelligence, AWS Textract,
    Google Document AI, PDFPlumber, LlamaParse, or OCR + layout heuristics for scans)
   │
   ▼
2) CONTENT EXTRACTION (per element type)
   - Text blocks   → plain text, preserving true reading order (not raster order)
   - Tables        → structured extraction (rows/cols), NOT flattened text
   - Images/charts → caption/describe with a vision model, don't embed raw pixels
   - Headers/footers/page numbers → strip from text, store separately as metadata
   │
   ▼
3) INTELLIGENT CHUNKING (type-aware)
   - Text   → structure-aware / recursive chunking
   - Tables → table-specific chunking (below)
   - Code   → AST-aware chunking
   - Images → chunk the generated description text, link back to the image asset
   │
   ▼
Chunks + rich metadata (page #, bbox, element type, section path)
```

**Multi-column PDFs — a common, underrated failure mode.** Naive extraction reads left-to-right, top-to-bottom *across* columns, interleaving two unrelated columns into garbage text. The fix is layout-aware extraction that detects column bounding boxes *before* assembling text — however smart the chunker downstream, it inherits lost structure permanently if extraction is wrong first.

**Code chunking (AST-aware).** Splitting mid-function is catastrophic — a function body without its signature is nearly un-embeddable, since the signature carries most of the semantics. Fix: use AST-aware splitters like `tree-sitter` that parse code into a syntax tree and chunk at function, class, or module boundaries, never at raw character positions.

**Charts, diagrams, images.** Generate a textual description/caption via a vision-language model, chunk and embed *that description* with a pointer back to the original image asset. Don't embed raw pixels into a text-based index unless specifically using a multi-modal embedding model.

**Headers, footers, page numbers.** Usually noise for retrieval (repeated on every page, polluting embeddings). Strip during layout analysis, but keep the page number as metadata — valuable for citations.

**Markdown / already-structured docs.** Chunk along heading hierarchy; each chunk inherits a breadcrumb: `"Returns Policy > International Orders > Customs & Duties"`. Enables metadata-filtered retrieval and better standalone interpretability.

**Production best practice:** build a **type-aware chunking pipeline**, not a single chunker — route each detected element (paragraph, table, image, code block, header) to a different chunking function, merge into one unified index with a `content_type` metadata field so retrieval can be boosted or filtered by type.

---

## Numerical & Tabular Data Chunking

**Think of a table like a family** — every cell's meaning depends on its relatives: the header above it, the row label beside it. `32.4%` is meaningless without "Q3 Revenue Growth" (column) and "EMEA" (row). Chunk a table the way you'd cut a family photo — keep people with the people they belong with.

**Fix A — atomic table chunks (small tables):** treat the entire table as one chunk, serialize to markdown or structured text, don't split.

**Fix B — header-repeated chunks (large tables):** split by logical row groups and repeat the header row into every chunk, so each chunk is self-interpreting independent of neighbors.

```python
def chunk_table_by_row(headers, rows, table_title=""):
    chunks = []
    for row in rows:
        row_desc = ", ".join(f"{h}: {v}" for h, v in zip(headers, row))
        chunk_text = f"Table: {table_title}\n{row_desc}"
        chunks.append(chunk_text)
    return chunks
```

**Other techniques:** column-based chunking when queries tend to ask about one metric across all rows; semantic/cell-relationship chunking for dense tables (group all "Revenue" line items separately from "Expenses"); multi-page tables must be stitched into one logical table during layout analysis (detect repeated header rows across pages as the continuation signal) before chunking — never treat each page as an independent table.

**Numerical precision:** keep full precision for exact compliance/audit figures; rounding to 1–2 significant figures is fine (and improves LLM reasoning reliability) for trend/comparison queries; keep full precision in the chunk for scientific measurements, rounding only in the displayed answer if at all.

**Why this matters:** LLMs are known to struggle with precise multi-digit arithmetic read directly from context. The safer production pattern for financial/scientific use cases: chunk with full precision preserved, but pair retrieval with a **calculator/code-execution tool** for any arithmetic the question requires, rather than trusting the LLM to "read" and compute from the table.

**Table formats for LLM consumption:**

| Format | Pros | Cons |
|---|---|---|
| Markdown table | Compact, LLMs heavily trained on it, readable | Breaks on very wide tables |
| HTML table | Preserves merged cells/complex structure | Verbose, more tokens |
| Structured text ("Row: X, Col: Y, Value: Z") | Most robust for retrieval — each fact explicit | Most verbose, more chunks |

Production best practice: store the table in structured text/JSON form for retrieval/chunking, but render it as a markdown table in the final prompt shown to the LLM.

### 🎯 Interview Gotcha
> "How would you answer 'What was the ROI for Q3 2024?' if the table cell doesn't literally say 'ROI'?"

This tests whether you understand that retrieval needs *derived context*, not just literal keyword match. If ROI must be computed from Revenue and Cost columns, the chunk must include both related columns together (not split across separate row-only chunks), and the pipeline should route to a calculation step rather than expecting the raw chunk to contain a pre-computed value. See Scenario F below for the full worked version.

---

## Metadata Enrichment Per Chunk

Four types worth knowing cold:

**1. Breadcrumb / section hierarchy** — `{"section": "Returns Policy > International Orders", "page": 47}`. Enables filtered retrieval ("search only within Returns Policy"). Without it, a query about returns might retrieve unrelated content that happens to use the word "return."

**2. Auto-generated chunk summaries** — for noisy chunks (dense tables, boilerplate regulatory text), generate a clean LLM summary and embed the summary instead of (or alongside) the raw text. A dense earnings table embedded as raw numbers produces a poor embedding; a natural-language summary ("Q3 2024 revenue grew 3.8% YoY to $1.23B") embeds far better.

**3. Hypothetical Questions (HyQ)** — generate 3–5 synthetic questions a chunk would answer, and embed the *questions* instead of the declarative chunk text. Real user queries are questions; chunk text is declarative prose — this distributional gap is the core challenge of asymmetric retrieval. Embedding synthetic questions closes the gap at *ingestion* time; it's the ingestion-time cousin of HyDE (Hypothetical Document Embeddings), the *query-time* version of the same idea.

**4. Source / temporal / access metadata** — `{"last_updated": "2024-09-15", "access_level": "internal", "jurisdiction": "EU"}`. Not for retrieval quality directly, but for filtering: recency, per-user access control, namespace scoping.

---

## Evaluation & the Empirical Tuning Loop

**Think of it like taste-testing a recipe before a dinner party** — you don't guess "more salt is better," you make small batches and measure against what guests actually want.

**Key metrics:**

| Metric | What It Measures |
|---|---|
| Recall@k | Did the correct chunk appear in the top-k retrieved results? |
| MRR (Mean Reciprocal Rank) | How high up was the first relevant chunk ranked? |
| Context relevance | Of the chunks retrieved, how much is actually relevant? |
| Answer correctness | Does the final generated answer match ground truth? |
| Faithfulness / groundedness | Is the answer actually supported by the retrieved chunks (not hallucinated)? |

**Tuning methodology (say this out loud in interviews — it's what separates a strong answer from guesswork):**
1. **Assemble a golden eval set** — 50–200+ (query, relevant passage) pairs, from user logs, manual annotation, or synthetic generation.
2. **Define the metric** — Recall@k for retrieval in isolation; answer correctness for end-to-end.
3. **Grid search over (chunk_size, overlap):**
```python
chunk_sizes = [128, 256, 512, 1024]
overlaps    = [0, 32, 64, 128]

for chunk_size in chunk_sizes:
    for overlap in overlaps:
        rebuild_index(chunk_size, overlap)
        recall = evaluate_recall_at_k(eval_set, k=5)
        log(chunk_size, overlap, recall)
```
4. **Find the knee of the curve** — recall usually improves steeply then plateaus; going past the knee just wastes context budget.
5. **Re-validate after any embedding model change** — the optimal chunk size is model-dependent.
6. **Segment results by document/query type and roll out gradually** (shadow-test before fully switching the index) — an aggregate win can hide a regression on, say, table-heavy queries specifically.

**Common pitfalls:** evaluating only end-to-end answer quality conflates chunking, retrieval, and generation quality — always measure retrieval metrics in isolation too; small unrepresentative golden sets that happen to favor one chunk size by luck; ignoring latency/cost in the trade-off (a 2% recall gain that triples ingestion cost may not be worth shipping); testing only easy factoid queries, missing multi-hop or table-lookup queries that are far more chunking-sensitive.

### 🎯 Interview Gotcha
> "Your retrieval Recall@5 looks great (95%) but users still complain answers are wrong. What's going on?"

High recall means the *right chunk is being retrieved* — so the bug is downstream: maybe the chunk is retrieved but truncated before the LLM sees it, maybe the LLM is ignoring the context (need a faithfulness/groundedness check), or maybe the "correct chunk" in the golden set isn't actually sufficient to answer the question on its own — a chunking *granularity* problem, even with correct retrieval.

---

## Worked Scenarios

- **A — PDF with text + tables + images:** layout analysis first detects the three element types; route text through structure-aware chunking, tables through header-aware row chunking, images through a vision-model caption step whose output is chunked normally. All share page number and section-path metadata.
- **B — Multi-page table spanning 3 pages:** detect the matching header row across pages as the continuation signal, stitch rows into one logical table before chunking, apply header-aware row chunking, store page number per row so citations stay accurate.
- **C — Dense numerical spreadsheets:** avoid one giant flattened chunk; chunk by logical sub-table (Revenue block separate from Expenses block), with sheet name and label cells as metadata.
- **D — Legal documents with nested clauses:** structure-aware chunking is essential; chunk at the clause level, always including the full heading path, and consider a brief parent-section summary for context when a sub-clause is retrieved in isolation.
- **E — Technical papers with equations:** never chunk an equation in isolation — keep it with the sentence(s) that define its variables.
- **F — "What's the ROI for Q3 2024?" (fully worked):** ROI is a *derived* metric, not a literal cell value. (1) At chunking time, Revenue and Cost for Q3 2024 must live in chunks retrievable *together* — a header-aware chunk spanning the full Q3 2024 column, not split per-metric. (2) At retrieval time, the query matches chunks containing "Q3 2024" + financial terms, narrowed by metadata filtering on `period: "Q3 2024"`. (3) At generation time, the LLM either computes ROI directly or — more robustly — routes through a code-execution/calculator tool to compute `(Revenue - Cost) / Cost` reliably. (4) Key takeaway: the chunking decision (keeping Revenue and Cost together) is what makes step 3 even *possible* at all.

## Decision Tree

```
What is the primary content type?
│
├─ Code → AST-aware chunking (tree-sitter, function/class boundaries)
│
├─ Tables → Atomic table chunks OR header-repeated row-group splits
│
├─ PDFs with layout complexity → Layout-aware extraction FIRST, then...
│     └─ proceed to prose/table branches below
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

# PHASE 3 — Interview Q&A Practice Set

**Q1 (Easy — conceptual).** Why is chunking considered one of the highest-leverage decisions in a RAG pipeline?

**A1.** Every downstream step — embedding, retrieval, generation — operates on chunks, not raw documents. A chunk containing one coherent idea produces a sharp, accurate embedding; a chunk crammed with multiple unrelated ideas produces a blurred, ambiguous embedding that matches poorly against any single-topic query. Because this happens at the very first stage of the pipeline, a bad chunking decision caps the ceiling of everything built on top of it, regardless of how good the embedding model, retrieval algorithm, or generator are. Unlike swapping the embedding model or LLM, it's also cheap and fast to iterate on — no retraining required.

---

**Q2 (Easy — calculation).** A 4000-token document is chunked with chunk size 400 and overlap 100. Compute the stride, approximate number of chunks, and the redundancy percentage.

**A2.**
```
S = 400 - 100 = 300
N ≈ ⌈(4000-100)/300⌉ = ⌈13⌉ = 13 chunks
total indexed tokens = 13 × 400 = 5200
redundancy = (5200 - 4000)/4000 = 30%
```

---

**Q3 (Medium — conceptual).** How does semantic chunking decide where to place a chunk boundary, and what's the main cost trade-off compared to fixed-size or recursive chunking?

**A3.** Semantic chunking embeds small units (typically individual sentences) and computes similarity between consecutive sentence embeddings as it walks through the document. When similarity drops sharply below a threshold — usually set as a percentile of the observed similarity/distance distribution within that document, since absolute similarity scales vary by embedding model and domain — a chunk boundary is placed there. The main cost trade-off is that this requires an embedding call per sentence just to find good boundaries, on top of embedding the final chunks themselves — meaningfully more expensive at ingestion time than fixed-size or recursive chunking, so it's typically reserved for high-value corpora where topical coherence materially affects downstream quality.

---

**Q4 (Medium — conceptual).** What problem does small-to-big (parent-document) retrieval solve, and how does it work?

**A4.** It resolves the tension between search precision (favoring small, focused chunks with sharp embeddings) and generation completeness (favoring larger chunks with enough surrounding context for a coherent answer). Instead of picking one chunk size and accepting both sides of that trade-off, small chunks are indexed and searched for precise matching, but each small chunk stores a pointer to a larger "parent" chunk. At query time, search happens over the small chunks, but the larger parent chunks are what actually get retrieved and passed to the generator — decoupling the unit you search over from the unit you generate from. Two lighter-weight variants: sentence-window retrieval (expand ±k sentences around a match) and hierarchical indexing (index multiple granularities and route by query type).

---

**Q5 (Medium — conceptual).** A team indexes API documentation and technical tables using fixed-size 200-token chunking with no structural awareness. What specifically goes wrong, and what would you recommend instead?

**A5.** Fixed-size chunking with no structural awareness will frequently split tables mid-row or mid-column and split code blocks or function definitions across chunk boundaries, producing fragments that are uninterpretable out of context (numbers with no header, code with unmatched brackets and no coherent standalone meaning — a function body without its signature is nearly un-embeddable). I'd recommend structure-aware/AST-aware chunking that respects the document's native boundaries: treat whole tables (or logical row groups, with the header row repeated into each) and whole functions as atomic units, and where a document has header hierarchy, keep the header path attached to each chunk so it retains standalone context.

---

**Q6 (Hard — synthesis / trade-off reasoning).** How would you actually determine the right chunk size and overlap for a brand-new RAG system, rather than guessing a default? Walk through your methodology.

**A6.** I'd treat chunk size and overlap as tunable hyperparameters and validate them empirically rather than picking a fixed default from general advice. Methodology: (1) build or bootstrap a golden eval set (50–200+ query → relevant-passage pairs) representative of real query patterns for this corpus; (2) grid-search a set of candidate chunk sizes and overlap ratios; (3) re-index the corpus at each configuration; (4) run the golden eval set against each, measuring not just Recall@k in isolation but also downstream generation metrics like faithfulness and answer correctness, since chunk size affects both stages; (5) find the "knee of the curve" and select the configuration that best balances quality against indexing/storage cost, rather than chasing the single highest metric regardless of cost; (6) segment results by document/query type before rolling out fully, since an aggregate win can hide a regression on a specific query type (e.g., table lookups); (7) re-validate this whole sweep whenever the embedding model changes, since optimal chunk size is model-dependent.

---

**Q7 (Medium — practical).** You increased chunk size and retrieval recall went up, but answer quality went down. Explain what happened.

**A7.** Larger chunks are more likely to *contain* the answer (higher recall — the target text is less likely to be split across a boundary). But the embedding of a larger chunk is a vector average over more content, which dilutes precision — it doesn't pinpoint the relevant section as sharply, and more irrelevant text gets bundled into the retrieved context. The LLM then receives the relevant passage plus surrounding noise, increasing the chance of incorrect synthesis or the model latching onto the wrong detail. The fix is usually parent-child chunking: keep small chunks for retrieval precision, and expand to the parent chunk only after retrieval, for generation.

---

**Q8 (Medium — multi-modal).** How would you chunk a 50-page PDF containing both prose and financial tables?

**A8.** Treat it as two problems in sequence. First, layout-aware extraction (a document-intelligence tool or layout parser) separates prose from table regions *before* any chunking happens — naive text extraction interleaves multi-column text and destroys table structure, and no chunking strategy can recover information that's already lost at extraction. Then apply type-specific chunking: recursive/structure-aware chunking for the prose sections; atomic or header-repeated chunking for the tables (whole table as one chunk where feasible, header row repeated into every chunk when a table must be split across chunks). I'd never apply a single generic text splitter across the whole mixed-content document.

---

**Q9 (Hard — derived-metric scenario).** A user asks "What was the ROI for Q3 2024?" but no cell in the source table literally says "ROI." How does your chunking strategy need to account for this?

**A9.** ROI is a derived metric — it has to be computed from Revenue and Cost, not looked up directly — so the chunking decision that makes this answerable at all is keeping the Revenue and Cost figures for Q3 2024 *together* in the same retrievable chunk (e.g., a header-aware chunk spanning the full Q3 2024 column), rather than splitting metrics into separate per-row or per-metric chunks. At retrieval time, metadata filtering on `period: "Q3 2024"` helps narrow candidates. At generation time, rather than trusting the LLM to compute the ratio correctly from raw numbers in context, the more robust pattern routes the actual arithmetic through a calculator/code-execution tool. The broader principle: retrieval needs to preserve *relationships* between related figures, not just the literal keyword the user happened to use.

---

**Q10 (Systems distinction).** What's the actual difference between "chunking" and "indexing," and why do candidates conflate them?

**A10.** Chunking is a *data* decision — what units of text to create — resolved through labelled eval sets and corpus analysis. Indexing is a *systems* decision — how to store and search those units at scale (Day 4 territory: vector databases, ANN structures) — resolved through latency benchmarks and memory profiling. They're conflated because they happen back-to-back in the ingestion pipeline, but a great chunking strategy on a poorly-chosen index retrieves the right content slowly; a great index over badly-chunked data retrieves the wrong content quickly. They're independent axes of quality.

---

# 🧠 Gotchas — Common Mistakes Recap

| Gotcha | Correct Understanding |
|---|---|
| "Overlap is always beneficial" | Overlap inflates index size and can waste top-k slots on near-duplicate chunks — diminishing returns past ~20% |
| "Bigger chunks always improve context" | Bigger chunks dilute embeddings and pack irrelevant noise into LLM context, hurting precision even as recall rises |
| "Fixed-size chunking is fine for everything" | Catastrophically wrong for tables, code, and multi-column PDFs |
| "Semantic chunking is always worth it" | Only for high-value corpora — meaningfully more expensive at ingestion scale |
| "Chunk size is a fixed constant" | Model-dependent and query-type-dependent — tune empirically per corpus, and re-tune after any embedding model swap |
| "Better extraction doesn't matter once you have chunking" | Garbage extraction (e.g., naive multi-column PDF text) makes any downstream chunking strategy irrelevant — fix extraction first |
| "Chunking and indexing are the same thing" | Different problem types: chunking is a data/eval problem, indexing is a systems/scale problem |
| "You should always maximize retrieval recall" | Recall must be balanced against precision and latency/cost — retrieving 50 chunks to be safe buries the LLM in noise and increases hallucination risk; the goal is the smallest *sufficient* context |

---

# 📌 Cheat Sheet (Day 3)

**Strategies:** Fixed-size (fast, cuts sentences) → recursive (respects structure, solid production default, e.g. LangChain's `RecursiveCharacterTextSplitter`) → semantic (embeds sentences, splits at similarity drops, most coherent but two embedding passes at ingestion) → structure-aware (respects tables/code/headers explicitly, free once you have layout parsing).

**Overlap trade-off:** modest overlap (~10–20%) removes most boundary-split risk cheaply; heavy overlap (~50%+) rarely worth the near-doubling of storage/embedding cost and wastes top-k slots on near-duplicates.

**Small-to-big retrieval:** search small chunks for precision, return large parent chunks for generation completeness — decouples the two competing needs instead of compromising on one chunk size. Sentence-window and hierarchical indexing are lighter-weight variants.

**Multi-modal / structured content:** always match the chunker to the content type — layout-aware extraction before chunking for PDFs, AST-aware chunking for code, header-repeated row chunks (with Revenue/Cost kept together for derived metrics) for tables, vision-model captions for images.

**Metadata:** breadcrumbs/section paths, auto-generated summaries for noisy chunks, hypothetical questions (HyQ) to close the query-vs-document gap, and source/temporal/access fields for filtering.

**Chunk size selection:** never a fixed default — grid-search chunk size × overlap against a golden eval set, measure retrieval *and* generation metrics, find the knee of the curve, balance quality against cost, re-validate after any embedding model change.

---

*End of Day 3. Next up — Day 4: Vector Databases & Indexing.*
