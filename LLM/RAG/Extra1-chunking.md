# 🧩 Chunking Strategies for RAG Systems — The Complete Interview Guide

*Taught the way Andrew Ng would: intuition first, math and code second, "why it matters" always.*

---

## 📚 Table of Contents

1. [The Big Picture: Why Chunking Even Matters](#1-the-big-picture)
2. [Fundamental Chunking Strategies](#2-fundamental-chunking-strategies)
   - 2.1 [Fixed-Size Chunking](#21-fixed-size-chunking)
   - 2.2 [Recursive / Sentence-Based Chunking](#22-recursive--sentence-based-chunking)
   - 2.3 [Semantic Chunking](#23-semantic-chunking)
   - 2.4 [Document Structure-Aware Chunking](#24-document-structure-aware-chunking)
   - 2.5 [Head-to-Head Comparison](#25-head-to-head-comparison)
3. [Industry Standards & Best Practices](#3-industry-standards--best-practices)
4. [Multi-Modal Document Chunking (PDF Focus)](#4-multi-modal-document-chunking-pdf-focus)
5. [Numerical & Tabular Data Chunking](#5-numerical--tabular-data-chunking)
6. [Evaluating Chunking Strategies](#6-evaluating-chunking-strategies)
7. [Worked Scenarios](#7-worked-scenarios)
8. [Final Interview Cram Sheet](#8-final-interview-cram-sheet)

---

## 1. The Big Picture

Think of an LLM like a brilliant expert with a **very short memory span**. You can't hand them your entire 400-page annual report and say "answer this question." You have to hand them just the *right few paragraphs*.

**Chunking is the process of cutting your documents into those "right few paragraphs"** — small enough to fit in the context window and be retrieved precisely, but large enough to still make sense on their own.

### Why This Matters 💡
Chunking is arguably the highest-leverage, most under-appreciated design decision in a RAG pipeline. You can have the best embedding model and the best LLM in the world — if your chunks are bad (too small, cutting off mid-sentence; too large, drowning the answer in noise), your retrieval will fail and the LLM will hallucinate or miss the answer entirely. In production, teams often find that **switching chunking strategy improves retrieval accuracy more than switching embedding models.**

Think of it like packing for a trip. If you pack every item loose in the suitcase (no chunking — feed the whole doc), you can't find your passport when you need it. If you pack one item per suitcase (chunks that are too small — a single sentence), you need 40 suitcases and lose the outfit that only makes sense as a whole. Good chunking is packing complete outfits into labeled bags — self-contained, retrievable, and useful the moment you open them.

### 🎯 Interview Gotcha
> "Isn't chunking a solved problem — just split every 500 tokens?"

No. This is a classic trap. Fixed-size splitting is a *baseline*, not a *solution*. Interviewers want to hear that you know chunking is content-dependent, retrieval-task-dependent, and requires evaluation — not a fire-and-forget default.

---

## 2. Fundamental Chunking Strategies

### 2.1 Fixed-Size Chunking

**Think of it like** slicing a loaf of bread with a ruler, every 2 centimeters, regardless of whether you cut through the crust, a raisin, or a slice of cheese sitting on top. Fast, predictable, but occasionally you slice right through the good part.

**How it works (step-by-step):**
1. Pick a chunk size in tokens (or characters) — e.g., 512 tokens.
2. Pick an overlap size — e.g., 50 tokens (10%).
3. Slide a window across the raw text, cutting a chunk every `chunk_size - overlap` tokens.
4. Store each chunk with its start/end offsets as metadata.

**When to use:**
- Homogeneous, unstructured text (chat logs, transcripts).
- You need a fast baseline to ship v1 of a RAG system.
- Very large corpora where per-chunk semantic analysis is too expensive.

**When NOT to use:**
- Structured documents (contracts, tables, code) where mid-sentence or mid-row cuts destroy meaning.
- When precision on "which exact fact" matters more than throughput.

**Advantages / Disadvantages**

| Aspect | Fixed-Size Chunking |
|---|---|
| Speed | ⚡ Very fast, O(n), no model calls |
| Cost | 💰 Free (no embedding calls needed to *decide* chunk boundaries) |
| Semantic coherence | ❌ Poor — can cut mid-sentence or mid-idea |
| Predictability | ✅ Chunk count and size are fully deterministic |
| Best for | Long, uniform prose; quick prototypes |

**Code Example:**
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

### ⚠️ Common Mistakes
- Chunking by raw character count instead of tokens — token counts are what the embedding model and LLM context window actually care about.
- Forgetting overlap entirely, which silently drops facts that straddle a chunk boundary.

### 🎯 Interview Gotcha
> "Why not just set overlap to 50%?"

Diminishing returns: high overlap balloons your index size and retrieval cost roughly linearly, while marginal recall gains flatten out fast after ~10-20%. Be ready to reason about this cost/recall tradeoff explicitly, not just quote "10-20% is standard."

---

### 2.2 Recursive / Sentence-Based Chunking

**Think of it like** a smarter version of the bread-slicer above: instead of a ruler, you use a set of *preferred* cutting lines — first try to cut along paragraph breaks, then sentence breaks, then word breaks, only falling back to a hard character cut as a last resort.

**How it works (step-by-step):**
1. Define an ordered list of separators, from "most preferred" to "least preferred": `["\n\n", "\n", ". ", " ", ""]`.
2. Try splitting on the first separator (paragraphs).
3. For any resulting piece still bigger than `chunk_size`, recursively split it using the *next* separator down the list.
4. Recombine small adjacent pieces up to `chunk_size` with overlap, so you don't end up with tiny orphan chunks.

**When to use:**
- General-purpose default for most production RAG systems — this is the most widely used strategy in practice (e.g., LangChain's `RecursiveCharacterTextSplitter`).
- Mixed prose documents: articles, docs, wikis, emails.

**When NOT to use:**
- Highly structured data (tables, code, JSON) where "sentence" isn't a meaningful unit.
- Documents where meaning depends on long-range structure (e.g., a legal clause referencing "Section 4.2" three pages later) — structure-aware chunking is safer there.

**Advantages / Disadvantages**

| Aspect | Recursive / Sentence-Based |
|---|---|
| Semantic coherence | ✅ Good — respects natural language boundaries |
| Speed | ⚡ Fast — no embedding model needed |
| Complexity to implement | 🟡 Medium |
| Handles structure (tables, code) | ❌ No |
| Best for | General-purpose production default |

**Code Example:**
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

    # merge small pieces up to chunk_size, with overlap
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

    # add simple trailing overlap
    overlapped = []
    for i, c in enumerate(chunks):
        prefix = chunks[i - 1][-overlap:] if i > 0 else ""
        overlapped.append((prefix + " " + c).strip())
    return overlapped
```

### 💡 Why This Matters
Recursive chunking is the workhorse of production RAG because it's the best "80/20" strategy: it costs nothing extra to compute (no model calls), yet respects natural language structure far better than fixed-size cuts. Most teams start here and only reach for semantic or structure-aware chunking once evaluation reveals it isn't enough for their specific documents.

---

### 2.3 Semantic Chunking

**Think of it like** reading a document with a highlighter, and starting a new highlight color every time the *topic* shifts — not every time you hit an arbitrary word count. You group sentences by meaning, not by length.

**How it works (step-by-step):**
1. Split the document into sentences.
2. Embed each sentence (or a small sliding window of sentences) with an embedding model.
3. Compute the cosine similarity (or distance) between consecutive sentence embeddings.
4. Wherever similarity drops below a threshold (a "semantic breakpoint" — meaning the topic just changed), insert a chunk boundary.
5. Merge the sentences between breakpoints into a chunk; optionally cap chunk size as a safety limit.

**When to use:**
- Documents with clear topic shifts but no reliable formatting cues (e.g., a raw transcript, an FAQ dump, a long-form narrative).
- When retrieval precision matters more than latency/cost — you're willing to pay for embedding calls at ingestion time.

**When NOT to use:**
- Very large corpora where the embedding cost at ingestion time is prohibitive.
- Real-time ingestion pipelines needing sub-second processing.
- Highly structured documents where structure already tells you the boundaries (use 2.4 instead).

**Advantages / Disadvantages**

| Aspect | Semantic Chunking |
|---|---|
| Semantic coherence | ✅✅ Best — chunks are topically self-contained |
| Speed | 🐢 Slow — requires embedding every sentence at ingestion |
| Cost | 💰💰 Higher — extra embedding calls per document |
| Threshold tuning | 🟡 Requires calibration per domain |
| Best for | High-value, well-written prose corpora (policies, wikis, research) |

**Code Example:**
```python
import numpy as np

def semantic_chunk(sentences, embed_fn, threshold_percentile=90, max_chunk_sentences=10):
    embeddings = [embed_fn(s) for s in sentences]
    sims = [
        np.dot(embeddings[i], embeddings[i + 1]) /
        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]
    # a *low* similarity marks a breakpoint -> use distance
    distances = [1 - s for s in sims]
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

### 🎯 Interview Gotcha
> "How do you pick the similarity threshold?"

There's no universal number — it's typically set via a **percentile** of the observed distance distribution within a document (e.g., "top 10% biggest jumps in distance are breakpoints") rather than an absolute cosine value, because absolute similarity scales differ across embedding models and domains. Say this explicitly; it signals real experience versus memorized trivia.

---

### 2.4 Document Structure-Aware Chunking

**Think of it like** cutting a cake along the lines already drawn on it — the document itself tells you where the natural divisions are: headings, sections, bullet lists, table boundaries. You're not inventing boundaries; you're respecting the ones the author already made.

**How it works (step-by-step):**
1. Parse the document into its structural tree — headings (H1/H2/H3), paragraphs, lists, tables, code blocks (using tools like `unstructured`, `LlamaParse`, HTML/Markdown parsers, or a PDF layout model).
2. Treat each structural unit (a section, a list item block, a table) as a candidate chunk.
3. If a unit is too large, recursively split it *within its own boundaries* (e.g., split a huge section by its sub-headings or by sentences, but never merge it with a sibling section).
4. If a unit is too small, merge it with an adjacent sibling under the same parent heading.
5. Attach structural metadata to every chunk: `{"h1": "...", "h2": "...", "section_path": "..."}`. This metadata is gold for retrieval and for citations.

**When to use:**
- Any document with real structure: technical docs, legal contracts, textbooks, API references, financial reports.
- When you need to preserve hierarchy (e.g., "this clause only makes sense under Section 4: Termination").

**When NOT to use:**
- Unstructured raw text with no headings (chat logs, plain narrative) — there's no structure to exploit, so recursive chunking is simpler and just as effective.

**Advantages / Disadvantages**

| Aspect | Structure-Aware Chunking |
|---|---|
| Semantic coherence | ✅✅ Excellent when structure is reliable |
| Metadata richness | ✅✅ Best — headings become filterable/citable metadata |
| Speed | 🟡 Medium — parsing overhead, but no embedding calls needed |
| Fragility | ❌ Breaks on malformed/inconsistent structure (e.g., messy scanned PDFs) |
| Best for | Contracts, manuals, textbooks, structured reports |

**Code Example:**
```python
import re

def structure_aware_chunk(markdown_text, max_chunk_size=800):
    # split on markdown headings, keep the heading with its content
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
            # fall back to recursive split within this section only
            for i in range(0, len(section), max_chunk_size):
                sub = section[i:i + max_chunk_size]
                chunks.append({"text": sub, "path": dict(heading_stack)})
    return chunks
```

### 💡 Why This Matters
The `path` / heading metadata attached here isn't a nice-to-have — it's what lets you (a) show users *where* an answer came from ("Section 4.2: Termination Clauses"), and (b) apply metadata filters at query time (e.g., "only search within the 'Pricing' section"). This is a favorite interview follow-up: *"How would you support the query 'What does the pricing section say about refunds?'"* — the answer is structure-aware metadata + filtered retrieval, not smarter embeddings.

---

### 2.5 Head-to-Head Comparison

| Strategy | Coherence | Cost | Speed | Handles Structure | Typical Use |
|---|---|---|---|---|---|
| Fixed-size | ❌ Low | 💰 Free | ⚡⚡⚡ Fastest | ❌ No | Quick baseline, uniform text |
| Recursive/sentence | ✅ Good | 💰 Free | ⚡⚡ Fast | 🟡 Partial | Default production choice |
| Semantic | ✅✅ Best | 💰💰 Higher (embed at ingest) | 🐢 Slow | 🟡 Partial | High-value prose, topic-shift-heavy docs |
| Structure-aware | ✅✅ Best (if structure exists) | 💰 Free–low | 🟡 Medium | ✅✅ Yes | Contracts, manuals, reports, textbooks |

### 🎯 Interview Gotcha
> "Which one strategy would you pick for a production RAG system?"

The trap is picking just one. The correct answer is a **hybrid**: structure-aware chunking as the outer layer (respect document hierarchy), recursive chunking as the fallback for oversized sections, and semantic chunking reserved for high-value unstructured corpora where the ingestion cost is justified. Naming this hybrid approach is what separates a strong answer from a memorized one.

---

## 3. Industry Standards & Best Practices

**Think of chunk size like a camera's zoom level.** Zoom in too far (tiny chunks) and you see a single pixel with no context. Zoom out too far (huge chunks) and the one detail you needed is buried in a landscape shot. Production teams tune this "zoom level" per use case.

### Typical Production Chunk Sizes

| Use Case | Typical Chunk Size | Why |
|---|---|---|
| Q&A over short FAQs / support docs | 128–256 tokens | Answers are short and localized |
| General knowledge base RAG | 256–512 tokens | Balances context vs precision — most common default |
| Long-form technical/legal docs | 512–1024 tokens | Need more surrounding context to preserve meaning |
| Code retrieval | 1 function/class, or ~300–500 tokens | Natural structural unit is the function |
| Conversational/chat memory | 1 turn or a small window of turns | Turns are the natural semantic unit |

### 💡 Why This Matters
There is no single "correct" chunk size — it's a hyperparameter, and like any hyperparameter it should be *tuned against your evaluation set*, not copy-pasted from a blog post. That said, **256–512 tokens with 10–20% overlap is the most common production starting point** across the industry, because it roughly matches the length of a self-contained paragraph while staying comfortably inside typical embedding model context limits (e.g., many embedding models cap around 512 tokens natively).

### Why 10–20% Overlap Is Standard
- Below ~10%: facts that straddle a boundary (a sentence split across two chunks) are frequently lost entirely from both chunks' embeddings.
- Above ~20%: index size and retrieval cost grow roughly linearly with overlap, but recall gains flatten — you're mostly paying for duplicate storage.
- 10–20% is the empirical "knee of the curve" most teams converge on.

### How Major Players Approach It (General Patterns)
- **Retrieval-focused vector DB providers** (e.g., Pinecone) generally publish guidance favoring recursive/structure-aware chunking with metadata-rich chunks over naive fixed-size splitting, and emphasize evaluating chunk size empirically per use case rather than using a universal default.
- **LLM providers** building RAG reference architectures typically emphasize combining chunking with strong metadata and re-ranking, treating chunking as one stage in a larger retrieval pipeline rather than the sole lever for quality.
- **Open-source frameworks** (LangChain, LlamaIndex) ship `RecursiveCharacterTextSplitter`-style splitters as the default precisely because it's the best general-purpose tradeoff discussed in Section 2.5.

*(Exact numbers change over time — if this comes up in a live interview, anchor on the reasoning above rather than quoting a specific vendor's current documentation, since these guidelines are periodically updated.)*

### Common Production Patterns ✅
- **Parent-child chunking**: embed small chunks for precise retrieval, but return the larger "parent" chunk (or the full section) to the LLM for context — best of both worlds.
- **Metadata-enriched chunks**: every chunk carries source, section path, page number, and timestamp for filtering and citation.
- **Sliding-window re-chunking on re-index**: as chunk-size hyperparameters get tuned, re-chunk from raw source rather than re-chunking already-chunked text (avoids compounding information loss).

### Common Anti-Patterns ❌
- Chunking after lossy text extraction (e.g., extracting a PDF to plain text and losing table structure *before* chunking — the damage is already done).
- One-size-fits-all chunk size across wildly different document types (contracts and Slack messages should not share a chunking config).
- Never re-evaluating chunk size after launch, even as the document corpus evolves.

### 🎯 Interview Gotcha
> "What chunk size should I always use?"

There is no "always." The correct interview answer names the tradeoff (recall vs precision vs cost) and the tuning process (evaluate on a held-out Q&A set), not a memorized number.

---

## 4. Multi-Modal Document Chunking (PDF Focus)

**Think of a PDF like a stage play, not a script.** A script (plain text) just gives you the words in order. A stage play (PDF) has actors in specific positions, props, lighting cues, a set — that is, a *visual layout* that carries meaning. Naively extracting "just the text" from a PDF is like reading only the play's dialogue and losing all the blocking — you'll misread which line belongs to which character, or miss that two "lines" were actually a table's rows and columns.

### The Golden Workflow: Layout Analysis → Content Extraction → Intelligent Chunking

```
PDF Input
   │
   ▼
1) LAYOUT ANALYSIS
   - Detect page structure: text blocks, tables, images, headers/footers
   - Tools: layout models (e.g., LayoutParser, Unstructured, Azure Doc
     Intelligence, LlamaParse), or OCR + layout heuristics for scans
   │
   ▼
2) CONTENT EXTRACTION (per element type)
   - Text blocks  -> plain text, preserving reading order
   - Tables       -> structured extraction (rows/cols), NOT flattened text
   - Images/charts-> caption/describe with a vision model, don't embed pixels
   - Headers/footers/page numbers -> strip or store separately as metadata
   │
   ▼
3) INTELLIGENT CHUNKING (type-aware)
   - Text   -> structure-aware / recursive chunking (Section 2.2, 2.4)
   - Tables -> table-specific chunking (Section 5)
   - Images -> chunk the generated description text, link back to image asset
   │
   ▼
Chunks + rich metadata (page #, bbox, element type, section path)
```

### Handling Each Content Type

**📄 Text (body paragraphs):**
Use structure-aware chunking anchored on detected headings; fall back to recursive chunking within long sections.

**📊 Tables:**
Never flatten a table into a stream of text and chunk it like prose — you'll lose which value belongs to which row/column. See Section 5 for dedicated strategies.

**🖼️ Charts, diagrams, images:**
Don't try to "embed the pixels" into a text-based retrieval index (unless you're specifically using a multi-modal embedding model). Instead:
1. Generate a textual description/caption using a vision-language model ("Bar chart showing Q1–Q4 2024 revenue by region, Q3 South America is the highest at $4.2M").
2. Chunk and embed *that description*, with a pointer/link back to the original image asset.
3. At answer time, the LLM can be shown both the retrieved description and (if multi-modal) the original image.

**📑 Headers, footers, page numbers:**
These are usually noise for retrieval (repeated on every page) and can pollute embeddings with irrelevant repeated tokens. Detect and strip them during layout analysis, but **keep page number as metadata** — it's valuable for citations ("see page 14").

### ⚠️ Common Mistakes
- Running a generic PDF-to-text library (that ignores layout) and *then* chunking — this bakes table/column-order errors into every downstream chunk.
- Treating a 2-column academic paper as one linear text stream — this interleaves the left and right columns into nonsense unless the layout model handles multi-column reading order.
- Embedding a chart's raw image as if it were a text chunk in a text-only vector index — most text embedding models cannot meaningfully embed images.

### 🎯 Production Best Practice
Build a **type-aware chunking pipeline**, not a single chunker. A PDF ingestion pipeline typically routes each detected element (paragraph, table, image, header) to a *different* chunking function, then merges the resulting chunks into one unified index with a `content_type` metadata field. This lets you apply different retrieval strategies later (e.g., "if the query looks numerical, boost table-type chunks").

### 🎯 Interview Gotcha
> "Why not just OCR the whole PDF to text and chunk that?"

OCR-to-text throws away layout (column order, table grid, image regions) *before* chunking, so any downstream chunker — however smart — inherits that lost structure permanently. Layout analysis has to happen *first*, and chunking has to be type-aware, not just text-aware.

---

## 5. Numerical & Tabular Data Chunking

**Think of a table like a family** — every cell's meaning depends on its relatives: the header above it (what column) and the row label beside it (what row/entity). Chunk a table like you'd cut up a family photo — keep people with the people they belong with, or the picture makes no sense.

### The Core Problem
If you naively chunk a table by raw character count, you might end up with a chunk like:

```
Q3    Q4    2023    2024
1.2M  1.5M  Revenue Growth
```

— numbers completely divorced from their row/column headers. The LLM has no way to know `1.2M` means "Q3 2024 Revenue."

### Techniques

**1. Header-Aware Chunking**
Every chunk derived from a table repeats the relevant column headers (and ideally the table title/caption) at the top of the chunk, even if that means some redundancy across chunks.

**2. Row-with-Context Chunking**
Chunk one (or a few) rows at a time, but *prepend* the table's headers and title to every chunk:
```
Table: Quarterly Revenue by Region (in $M)
Headers: Region | Q1 2024 | Q2 2024 | Q3 2024 | Q4 2024
Row: South America | 3.1 | 3.4 | 4.2 | 3.9
```
This keeps each row self-contained and retrievable on its own.

**3. Column-Based Chunking**
Useful when queries tend to ask about a single metric across all rows (e.g., "show revenue trend for all regions"). Chunk by column instead of by row, again prepending the row labels as context.

**4. Semantic/Cell-Relationship Chunking**
For dense tables (e.g., a 50x50 financial model), chunk by *logical sub-table* — group related rows/columns (e.g., all "Revenue" line items together, separate from all "Expense" line items), preserving the sub-header for each group.

**5. Multi-Page Table Handling**
A table spanning pages 4–6 must be **stitched back into one logical table** during layout analysis *before* chunking — detect repeated header rows across pages (a strong signal of continuation) and merge, rather than treating each page as an independent table.

### Numerical Precision: Round or Keep Full?

| Situation | Recommendation |
|---|---|
| Query needs exact compliance/audit figures (financial statements, legal filings) | Keep full precision, exactly as reported |
| Query needs trend/comparison ("did revenue grow?") | Rounding to 1-2 significant figures is fine and improves LLM reasoning reliability |
| Very long decimal chains (e.g., scientific measurements) | Keep full precision in the chunk, but consider a rounded version in the *displayed* answer |

### 💡 Why This Matters
LLMs are known to struggle with precise multi-digit arithmetic directly from context. For financial/scientific use cases, the safer production pattern is: chunk with full precision preserved, but pair retrieval with a **calculator/code-execution tool** for any arithmetic the user's question requires, rather than trusting the LLM to compute `4.2M - 3.1M` correctly by "reading" the table.

### Formatting Tables for LLM Consumption

| Format | Pros | Cons |
|---|---|---|
| Markdown table | Compact, LLMs are heavily trained on it, human-readable | Breaks on very wide tables |
| HTML table | Preserves merged cells/complex structure | Verbose, more tokens |
| Structured text ("Row: X, Col: Y, Value: Z") | Most robust for retrieval — each fact is explicit and self-contained | Most verbose, more chunks |

**Production best practice:** store the table in **structured text or JSON form** for retrieval/chunking (so each fact is unambiguous and independently retrievable), but render it as a **markdown table** in the final prompt shown to the LLM (compact and well-understood by the model).

### Code Example: Header-Aware Row Chunking
```python
def chunk_table_by_row(headers, rows, table_title=""):
    chunks = []
    for row in rows:
        row_desc = ", ".join(f"{h}: {v}" for h, v in zip(headers, row))
        chunk_text = f"Table: {table_title}\n{row_desc}"
        chunks.append(chunk_text)
    return chunks

# Example
headers = ["Region", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
rows = [
    ["South America", "3.1", "3.4", "4.2", "3.9"],
    ["North America", "8.5", "8.9", "9.1", "9.6"],
]
chunks = chunk_table_by_row(headers, rows, "Quarterly Revenue ($M)")
# chunks[0] -> "Table: Quarterly Revenue ($M)\nRegion: South America, Q1 2024: 3.1, ..."
```

### Handling Financial Reports, Scientific Papers, Data Sheets
- **Financial reports**: always retain the reporting period and currency/unit as chunk metadata (`unit: "$M"`, `period: "Q3 2024"`) — a number without units is a bug waiting to happen.
- **Scientific papers**: equations should be chunked with the surrounding sentence that defines each variable, not extracted as bare LaTeX — an equation without variable definitions is unusable for retrieval.
- **Data sheets** (spec sheets): chunk by product/spec-group with the product name repeated in every chunk (e.g., every chunk about "Model X-200" restates "Model X-200" so it stays retrievable even if the model name only appears once at the top of the original sheet).

### 🎯 Interview Gotcha
> "How would you answer 'What was the ROI for Q3 2024?' if the table cell doesn't literally say 'ROI'?"

This tests whether you understand that retrieval needs *derived context*, not just literal keyword match. If ROI must be computed from Revenue and Cost columns, the chunk needs to include both related columns together (not split across separate row-only chunks), and ideally the pipeline should route to a calculation step rather than expecting the raw chunk to contain a pre-computed "ROI" value. See Section 7 for the full worked scenario.

---

## 6. Evaluating Chunking Strategies

**Think of evaluation like taste-testing a recipe before serving it at a dinner party.** You don't just guess that "more salt is better" — you make small batches, taste them side by side, and measure against what your guests actually want. Chunking strategy is the "recipe"; evaluation is the taste test.

### Key Metrics

| Metric | What It Measures | How to Compute |
|---|---|---|
| **Retrieval accuracy (Recall@k / Precision@k)** | Did the *correct* chunk appear in the top-k retrieved results? | Curate a labeled Q&A set with known "gold" source chunks; check overlap |
| **Context relevance** | Of the chunks retrieved, how much of their content is actually relevant to the query? | LLM-as-judge scoring, or human annotation, on retrieved chunks |
| **Answer correctness / response quality** | Does the final generated answer match ground truth? | Exact match / semantic similarity / LLM-as-judge against a reference answer |
| **Faithfulness / groundedness** | Is the answer actually supported by the retrieved chunks (not hallucinated)? | LLM-as-judge checking claim-by-claim support |
| **MRR (Mean Reciprocal Rank)** | How high up was the first relevant chunk ranked? | Standard IR metric over the labeled set |

### A/B Testing Framework for Chunking

1. **Build a golden evaluation set**: 50–200+ representative (question, expected source chunk / expected answer) pairs, covering easy, hard, and edge-case queries.
2. **Fix everything except chunking**: same embedding model, same retriever top-k, same LLM — vary only chunk size/overlap/strategy between variants.
3. **Run each variant** against the golden set and compute Recall@k, MRR, and answer correctness.
4. **Statistical significance**: with small eval sets, a 2-point difference in accuracy can be noise — use bootstrapped confidence intervals or paired significance tests before declaring a winner.
5. **Segment results** by document type/query type — a strategy that wins overall might lose specifically on table-heavy queries; aggregate numbers can hide this.
6. **Roll out gradually**: shadow-test the new chunking strategy in production before fully switching the index.

### Common Pitfalls in Evaluation ⚠️
- **Evaluating only end-to-end answer quality**, which conflates chunking quality with retrieval quality and generation quality — you can't tell *which stage* is failing. Always also measure retrieval metrics in isolation.
- **Small, unrepresentative golden sets** that happen to favor one chunk size by luck. Cover diverse query types deliberately.
- **Ignoring latency/cost** in the tradeoff — a strategy that improves recall by 2% but triples ingestion cost may not be worth shipping.
- **Testing only "easy" factoid queries**, missing multi-hop or table-lookup queries that are far more sensitive to chunking choices.
- **Not re-evaluating after the underlying embedding model changes** — chunk size sweet spots can shift when you swap embedding models.

### 💡 Why This Matters
"Just try a few chunk sizes and eyeball the results" is not an evaluation strategy an interviewer wants to hear. The strong answer treats chunking as a hyperparameter with a proper train/eval loop: golden set → controlled A/B → statistically-aware comparison → segmented analysis → gradual rollout.

### 🎯 Interview Gotcha
> "Your retrieval Recall@5 looks great (95%) but users still complain answers are wrong. What's going on?"

This is testing whether you understand the pipeline has multiple failure points. High recall means the *right chunk is being retrieved* — so the bug is downstream: maybe the chunk is retrieved but truncated before the LLM, maybe the LLM is ignoring the context (need a faithfulness/groundedness check), or maybe the golden set's "correct chunk" isn't actually sufficient to answer the question (a chunking granularity problem, even with correct retrieval).

---

## 7. Worked Scenarios

### Scenario A: PDF with Text + Tables + Images
**Approach:** Layout analysis first (Section 4) to detect three distinct element types on the page. Route text through structure-aware chunking, tables through header-aware row chunking (Section 5), and images through a vision-model caption step whose output text gets chunked normally. All three chunk types share the same page number and section-path metadata so a query can retrieve across types and the LLM can be told "here's the relevant paragraph, table row, and chart description from page 12."

### Scenario B: Multi-Page Table Spanning 3 Pages
**Approach:** During layout analysis, detect that the header row on page 5 matches the header row on page 4 (same column labels) — this is the signal that it's a continuation, not a new table. Stitch all rows from pages 4–6 into one logical table object before chunking. Then apply header-aware row chunking as normal, with page number stored per-row so citations remain accurate even though rows are logically merged.

### Scenario C: Dense Numerical Spreadsheets
**Approach:** Avoid one giant "flatten the sheet to text" chunk. Instead, chunk by logical sub-table/section within the sheet (e.g., "Revenue" block separate from "Expenses" block), using header-aware row chunking within each block, with the sheet name and any surrounding label cells captured as metadata.

### Scenario D: Legal Documents with Complex Structure
**Approach:** Structure-aware chunking is essential here — legal documents are built from nested numbered clauses (e.g., "Section 4 > 4.2 > 4.2.1") where meaning is highly dependent on hierarchy ("this obligation applies only if Section 3 conditions are met"). Chunk at the clause level, always including the clause's full heading path in metadata, and consider including a brief summary of the parent section for context when a sub-clause is retrieved in isolation.

### Scenario E: Technical Papers with Equations
**Approach:** Never chunk an equation in isolation. Include the sentence(s) immediately before/after that define the variables, and prefer keeping an equation with its surrounding paragraph as one chunk (structure-aware, section-scoped) rather than splitting equation from explanation.

### Scenario F: "What's the ROI for Q3 2024?" (Table + Context Retrieval) — Fully Worked
This is a favorite interview scenario because "ROI" is a **derived** metric, not a literal cell value.

**Step-by-step solution:**
1. **Chunking time:** the Revenue and Cost rows/columns for Q3 2024 must live in chunks that are retrievable *together* — e.g., a header-aware chunk containing the full Q3 2024 column across all relevant line items (Revenue, Cost, Net Income), not split into separate per-metric chunks.
2. **Retrieval time:** the query "ROI for Q3 2024" is embedded and should match chunks containing "Q3 2024" + financial terms; metadata filtering on `period: "Q3 2024"` further narrows candidates.
3. **Generation time:** the LLM receives the retrieved Revenue and Cost figures for Q3 2024 and either (a) computes ROI directly if the numbers are simple, or (b) — in a robust production system — the answer is routed through a **code-execution/calculator tool** to compute `(Revenue - Cost) / Cost` reliably, rather than trusting free-form LLM arithmetic.
4. **Key takeaway for interviews:** the chunking decision (keeping Revenue and Cost together) is what makes step 3 even *possible* — this is the clearest illustration of why "preserving relationships" in tabular chunking directly determines whether a derived-metric question can be answered at all.

---

## 8. Final Interview Cram Sheet

### Rapid-Fire Q&A

**Q: What's the single most important tradeoff in chunk size selection?**
A: Recall vs. precision — smaller chunks improve precision (less noise per chunk) but hurt recall/context (facts split across chunks); larger chunks do the opposite. Overlap partially mitigates the recall loss from small chunks.

**Q: Why is fixed-size chunking still used in production despite being "worse" semantically?**
A: It's essentially free (no model calls), fully predictable, and for homogeneous unstructured text the semantic loss is often small — it's a legitimate choice when speed/cost dominate and content is uniform.

**Q: What's the biggest mistake teams make with PDF chunking?**
A: Extracting to plain text *before* preserving layout, which permanently destroys table structure and reading order before chunking even begins.

**Q: How do you evaluate whether your chunking strategy is good?**
A: A golden Q&A eval set with known source chunks, measuring Recall@k/MRR for retrieval in isolation (not just end-to-end answer quality), A/B'd across chunking variants with everything else held constant.

**Q: Give an example of when semantic chunking is worth its extra cost.**
A: A long-form policy document or FAQ dump with no reliable headings but frequent topic shifts — structure-aware chunking has nothing to anchor on, and fixed-size chunking would cut across topics arbitrarily.

**Q: What's "parent-child chunking" and why does it matter?**
A: Embedding small, precise child chunks for retrieval accuracy, but returning the larger parent chunk/section to the LLM for full context — combines retrieval precision with generation-time context richness.

**Q: How should you chunk a table so the LLM can still answer questions correctly?**
A: Header-aware, so every chunk restates the relevant column/row labels — never let a numeric value get separated from the header/label that gives it meaning.

**Q: Trick question — should you always maximize retrieval recall?**
A: No — recall must be balanced against precision and latency/cost; retrieving 50 chunks to guarantee the answer is present buries the LLM in noise and increases hallucination risk, plus cost. The goal is the *smallest sufficient context*, not the largest possible one.

---

### 🗺️ One-Page Mental Model

```
                     ┌─────────────────────────┐
                     │   Does structure exist?  │
                     └────────────┬────────────┘
                       Yes ◄──────┴──────► No
                        │                  │
         ┌──────────────▼───────┐   ┌──────▼───────────────┐
         │ Structure-aware       │   │ Topic shifts within   │
         │ chunking (headings,   │   │ unstructured text?    │
         │ tables, clauses)      │   └──────┬────────────────┘
         └──────────────┬────────┘     Yes │      No
                         │                  │       │
                Oversized section?  ┌───────▼──┐  ┌─▼──────────┐
                    │                │Semantic  │  │ Recursive/  │
              ┌─────▼──────┐         │chunking  │  │sentence-    │
              │Recursive    │         └──────────┘  │based chunk  │
              │split within │                        │ (default)  │
              │boundary     │                        └────────────┘
              └─────────────┘
```

---

*Good luck with your interviews! Remember: the strongest answers always connect the "why" (the tradeoff being solved) to the "how" (the specific technique) — an interviewer would rather hear you reason through a novel document type live than recite a memorized strategy name.*
