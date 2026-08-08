# 🧩 Chunking Strategies for RAG Systems — The Complete Interview Guide

*Taught the way Andrew Ng would: intuition first, math and code second, "why it matters" always.*

---

## 📚 Table of Contents

1. [The Big Picture: Why Chunking Is the Highest-Leverage Knob](#1-the-big-picture)
2. [Fundamental Chunking Strategies](#2-fundamental-chunking-strategies)
   - 2.1 [Fixed-Size Chunking](#21-fixed-size-chunking)
   - 2.2 [Recursive / Sentence-Based Chunking](#22-recursive--sentence-based-chunking)
   - 2.3 [Semantic Chunking](#23-semantic-chunking)
   - 2.4 [Document Structure-Aware Chunking](#24-document-structure-aware-chunking)
   - 2.5 [Head-to-Head Comparison](#25-head-to-head-comparison)
3. [Small-to-Big / Parent-Child Chunking](#3-small-to-big--parent-child-chunking)
4. [Industry Standards & Best Practices](#4-industry-standards--best-practices)
5. [Multi-Modal & Document-Specific Chunking (PDFs, Tables, Code)](#5-multi-modal--document-specific-chunking)
6. [Numerical & Tabular Data Chunking](#6-numerical--tabular-data-chunking)
7. [Metadata Enrichment Per Chunk](#7-metadata-enrichment-per-chunk)
8. [Evaluation & the Empirical Tuning Loop](#8-evaluation--the-empirical-tuning-loop)
9. [Worked Scenarios](#9-worked-scenarios)
10. [Decision Tree](#10-decision-tree)
11. [Final Interview Cram Sheet & Q&A Drill](#11-final-interview-cram-sheet--qa-drill)

---

## 1. The Big Picture

Think of an LLM like a brilliant expert with a **very short memory span**. You can't hand them your entire 400-page annual report and say "answer this question." You have to hand them just the *right few paragraphs*. **Chunking is the decision of what unit of text becomes retrievable** — it happens before embedding, before indexing, before retrieval, and everything downstream is constrained by it.

**Another way to picture it:** you're building a filing system for a law library. If each "file" is an entire book, the librarian (retriever) keeps pulling out whole books when you ask one specific question — accurate occasionally, but mostly wasteful. If each "file" is a single word, the librarian pulls out cards that say "the" and "a" with no surrounding context. The right granularity sits in between, and it depends on what questions people typically ask.

### The Core Tension — Four Dimensions

| Too Large | Too Small |
|---|---|
| Embedding is a diluted average of multiple topics | Chunk loses surrounding context needed for standalone interpretation |
| Low retrieval precision (wrong content retrieved alongside correct) | Low recall (a coherent fact fragmented across multiple chunks) |
| Wastes LLM context window on irrelevant text | A sentence like "It increased by 40%" is useless without the referent of "it" |
| Fewer unique chunks fit in top-k slots | Same fact retrieved as multiple disconnected fragments |

### Why This Matters 💡
Chunking is arguably the highest-leverage, most under-appreciated design decision in a RAG pipeline — and, unlike swapping the embedding model or the LLM (expensive, slow, sometimes risky), chunking changes are **free to experiment with** (no retraining), **measurable** with a small labelled eval set (query → known-relevant passage), and **high-impact** — misconfigured chunking can make a great embedding model look terrible. In production, teams often find that switching chunking strategy improves retrieval accuracy more than switching embedding models.

**Interview framing to use out loud:** *"Before I touch the embedding model or the retriever, I'd audit the chunking strategy — it's the cheapest, highest-leverage variable to change."*

### 🎯 Interview Gotcha
> "Isn't chunking a solved problem — just split every 500 tokens?"

No. Fixed-size splitting is a *baseline*, not a *solution*. Interviewers want to hear that you know chunking is content-dependent, query-type-dependent, and requires evaluation — not a fire-and-forget default.

---

## 2. Fundamental Chunking Strategies

### 2.1 Fixed-Size Chunking

**Think of it like** slicing a loaf of bread with a ruler, every 2 centimeters, regardless of whether you cut through the crust, a raisin, or a slice of cheese sitting on top. Fast, predictable, but occasionally you slice right through the good part.

**How it works (step-by-step):**
1. Pick a chunk size in tokens — e.g., 512 tokens.
2. Pick an overlap size — e.g., 50 tokens (~10%).
3. Slide a window across the raw text, cutting a chunk every `chunk_size - overlap` tokens.
4. Store each chunk with its start/end offsets as metadata.

```
[Token 1 ........... Token 512]
                  [Token 462 ........... Token 974]
                                     [Token 924 ........... Token 1436]
                   ←50 tok→         ←50 tok→
                   (overlap)         (overlap)
```

**Why overlap exists — a concrete example.** Imagine this sentence spans a chunk boundary:

> *"The policy, which was first introduced in the 2021 fiscal year, increased premiums by 40%."*

Without overlap:
- Chunk A ends at: `"...first introduced in the 2021 fiscal year"`
- Chunk B starts at: `"increased premiums by 40%."`

Chunk B is now uninterpretable — "increased" has no subject. Overlap ensures the full sentence survives intact in at least one chunk.

### ⚠️ The Hidden Cost of Overlap (a numbers-based Interview Gotcha)
Many candidates describe overlap as purely beneficial — it isn't free.

**Worked example — index bloat:**
- Corpus: 1,000,000 tokens
- Chunk size: 512 tokens, overlap: 50 tokens → effective stride = 512 − 50 = 462 tokens/chunk
- Number of chunks ≈ 1,000,000 / 462 ≈ **2,165 chunks**
- Without overlap (stride = 512): ≈ 1,000,000 / 512 ≈ **1,953 chunks**

That's ~11% more chunks — and the real cost isn't just storage: the *same underlying fact* now appears in up to two adjacent chunks. If your retriever returns top-k = 5, two of those five slots can be near-duplicates from overlapping chunks, wasting retrieval diversity instead of surfacing five distinct pieces of evidence.

**When to use:** homogeneous, prose-heavy corpora with no special structure; quick prototypes/baselines; when ingestion simplicity matters more than retrieval precision.

**When NOT to use:** structured documents (contracts, tables, code) where mid-sentence or mid-row cuts destroy meaning; when precision on "which exact fact" matters more than throughput.

**Advantages / Disadvantages**

| Aspect | Fixed-Size Chunking |
|---|---|
| Speed / cost | ⚡ Trivial to implement, free (no model calls) |
| Predictability | ✅ Deterministic chunk sizes and embedding cost |
| Semantic coherence | ❌ Ignores document structure entirely — can split mid-sentence, mid-table, mid-function |
| Index size | ❌ Overlap inflates index size and can waste top-k slots on near-duplicates |
| Best for | Quick baselines; uniform, unstructured prose |

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
- Assuming more overlap is strictly better — see the index-bloat math above.

---

### 2.2 Recursive / Sentence-Based Chunking

**Think of it like** tearing bread along its scored lines rather than cutting straight across the loaf with a ruler: instead of a fixed interval, you use the document's own natural seams.

**How it works (step-by-step):**
1. Define an ordered list of separators, most preferred first: `["\n\n", "\n", ". ", " ", ""]`.
2. Try splitting on the first separator (paragraph breaks).
3. For any resulting piece still bigger than `chunk_size`, recursively split it using the *next* separator down the list.
4. Recombine small adjacent pieces up to `chunk_size` with overlap, so you don't end up with tiny orphan chunks.

This is the actual behavior of LangChain's `RecursiveCharacterTextSplitter` — **the production default in most real RAG stacks.** Worth naming explicitly in interviews.

**Why it's better than fixed-size — a concrete example.** Document authors encode topic shifts through formatting: a double newline usually signals a topic change.

> Paragraph 1: *"Our refund policy for domestic orders allows returns within 30 days..."*
> Paragraph 2: *"For international customers, customs duties are non-refundable..."*

Fixed-size chunking might lump both paragraphs into one chunk, producing an embedding that's an *average* of "domestic refunds" and "international customs" — a query about domestic refunds retrieves this chunk, but the LLM then has to wade through irrelevant international-customs text too. Recursive splitting respects the paragraph break and creates two clean, focused chunks instead.

**Still a limitation:** structure ≠ semantics. A single long paragraph can drift across two topics; two short paragraphs might be one continuous thought. Structure is a *proxy* for semantic coherence, not a guarantee — that gap is exactly what semantic chunking (2.3) exists to close.

**When to use:** general-purpose default for most production RAG systems; mixed prose documents (articles, docs, wikis, emails).

**When NOT to use:** highly structured data (tables, code, JSON) where "sentence" isn't a meaningful unit; documents where meaning depends on long-range hierarchical structure (use 2.4 instead).

**Advantages / Disadvantages**

| Aspect | Recursive / Sentence-Based |
|---|---|
| Semantic coherence | ✅ Good — respects natural language boundaries |
| Speed / cost | ⚡ Fast, free — no embedding model needed at chunk time |
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

---

### 2.3 Semantic Chunking

**Think of it like** reading with a highlighter and starting a new color every time the *topic* shifts, not every time you hit a word-count limit — like a skilled editor who can feel when a document transitions from one idea to the next, regardless of formatting.

**How it works (step-by-step):**
1. Split the document into sentences.
2. Embed each sentence individually.
3. Compute cosine similarity between each consecutive sentence pair.
4. Where similarity drops sharply below a threshold → insert a chunk boundary.
5. Merge sentences between boundaries into a chunk.

```
Sentence 1: "The kidney filters blood."
Sentence 2: "It removes waste products via urine."       sim(1,2) = 0.82 → same topic, no break
Sentence 3: "Hypertension is a leading cause of failure." sim(2,3) = 0.71 → same topic, no break
Sentence 4: "The liver produces bile for digestion."      sim(3,4) = 0.23 → TOPIC SHIFT → boundary
```

### 💰 The Cost Model — When Is Semantic Chunking Worth It?
At ingestion you effectively pay for **two rounds of embedding**: round 1 embeds every sentence to detect breakpoints (thrown away afterward), round 2 embeds the final chunks for the index.

**Worked example — 1,000-page legal corpus:**
- Assume 500 sentences/page × 1,000 pages = 500,000 sentences for breakpoint detection.
- At roughly $0.00002 per 1K tokens (small embedding model), 500,000 sentences × ~15 tokens avg ≈ 7.5M tokens ≈ **$0.15 extra** for breakpoint detection.

For a legal firm where a single missed clause costs thousands of dollars, that's trivial. For a startup indexing Wikipedia for a chatbot, it's probably overkill relative to the value gained — this is the cost/benefit judgment call to name explicitly in an interview.

### Threshold Sensitivity

| Threshold setting | Effect | Failure mode |
|---|---|---|
| Too strict (e.g., require sim > 0.9 to stay together) | Splits at any phrasing variation | Near-single-sentence chunks; loses context |
| Too lenient (e.g., require sim < 0.3 to split) | Almost never splits | Barely better than paragraph chunking |
| Well-tuned | Splits at genuine topic shifts | Requires empirical tuning per corpus |

There is **no universal threshold** — in practice it's set via a **percentile** of the observed similarity/distance distribution within a document (e.g., "the steepest 10% of similarity drops are breakpoints"), tuned empirically per corpus, because absolute similarity scales differ across embedding models and domains.

**When to use:** high-value corpora where retrieval errors are costly (legal, medical, financial); long flowing paragraphs with embedded topic shifts and no reliable headings; infrequent ingestion.

**When NOT to use:** high-volume, low-stakes content (FAQs, product descriptions); documents that already have clean structural formatting (use 2.4 — it's free); real-time/latency-sensitive ingestion.

**Advantages / Disadvantages**

| Aspect | Semantic Chunking |
|---|---|
| Semantic coherence | ✅✅ Best — chunks are topically self-contained |
| Speed / cost | 🐢💰💰 Slow, two embedding passes at ingestion |
| Threshold tuning | 🟡 Requires per-domain calibration |
| Best for | High-value, well-written prose corpora (policy, legal, medical, financial) |

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

---

### 2.4 Document Structure-Aware Chunking

**Think of it like** cutting a cake along the lines already drawn on it — headings, sections, bullet lists, table boundaries are natural divisions the author already made; you're respecting them, not inventing new ones.

**How it works (step-by-step):**
1. Parse the document into its structural tree — headings (H1/H2/H3), paragraphs, lists, tables, code blocks (using tools like `unstructured`, `LlamaParse`, HTML/Markdown parsers, or a PDF layout model).
2. Treat each structural unit as a candidate chunk.
3. If a unit is too large, recursively split it *within its own boundaries* (never merge across sibling sections).
4. If a unit is too small, merge it with an adjacent sibling under the same parent heading.
5. Attach structural metadata to every chunk: `{"h1": "...", "h2": "...", "section_path": "..."}` — this metadata is gold for retrieval and citations (see Section 7).

**When to use:** any document with real structure — technical docs, legal contracts, textbooks, API references, financial reports; when you need to preserve hierarchy ("this clause only makes sense under Section 4: Termination").

**When NOT to use:** unstructured raw text with no headings — recursive chunking is simpler and just as effective.

**Advantages / Disadvantages**

| Aspect | Structure-Aware Chunking |
|---|---|
| Semantic coherence | ✅✅ Excellent when structure is reliable |
| Metadata richness | ✅✅ Best — headings become filterable/citable metadata |
| Speed / cost | 🟡 Medium parsing overhead, but no embedding calls needed |
| Fragility | ❌ Breaks on malformed/inconsistent structure (messy scanned PDFs) |
| Best for | Contracts, manuals, textbooks, structured reports |

**Code Example:**
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

---

### 2.5 Head-to-Head Comparison

| Strategy | Coherence | Cost | Speed | Handles Structure | Typical Use |
|---|---|---|---|---|---|
| Fixed-size | ❌ Low | 💰 Free | ⚡⚡⚡ Fastest | ❌ No | Quick baseline, uniform text |
| Recursive/sentence | ✅ Good | 💰 Free | ⚡⚡ Fast | 🟡 Partial | Default production choice |
| Semantic | ✅✅ Best | 💰💰 Higher (2 embed passes) | 🐢 Slow | 🟡 Partial | High-value prose, topic-shift-heavy docs |
| Structure-aware | ✅✅ Best (if structure exists) | 💰 Free–low | 🟡 Medium | ✅✅ Yes | Contracts, manuals, reports, textbooks |

### 🎯 Interview Gotcha
> "Which single strategy would you pick for a production RAG system?"

The trap is picking just one. The strong answer is a **hybrid**: structure-aware chunking as the outer layer, recursive chunking as the fallback for oversized sections, semantic chunking reserved for high-value unstructured corpora where the ingestion cost is justified — and parent-child chunking (Section 3) layered on top to resolve the precision-vs-context tension no single chunk size can solve alone.

---

## 3. Small-to-Big / Parent-Child Chunking

**Think of it like** using a library's card catalog. When you *search*, you look up a fine-grained index card (small granularity, precise matching). When you *read*, you pull the full book off the shelf, not just the index card (large granularity, full context). Parent-child chunking builds exactly this two-level system for retrieval.

### The Core Insight
The fundamental tension from Section 1 — small chunks give precise retrieval matches, large chunks give the LLM enough context to generate a good answer — cannot be resolved by picking one chunk size. Parent-child **decouples retrieval granularity from generation granularity.**

### How It Works

**At ingestion:**
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

**At retrieval:**
```
Query: "What is the refund timeline for international orders?"
   ↓
Retriever finds: Child chunk (sentence 3) — high similarity match
   ↓
System fetches: Parent chunk (Section 1) — the full section containing sentence 3
   ↓
LLM receives: Full section context, not just the matched sentence
```

### Two Variants Worth Naming in an Interview

**Sentence-window retrieval:** index individual sentences; when a sentence is retrieved, expand to ±k surrounding sentences (e.g., ±2, giving a 5-sentence window). Simpler to implement than full parent-child, nearly as effective for many corpora.

**Hierarchical indexing:** index multiple levels simultaneously — section summaries *and* paragraph chunks *and* sentence chunks — then route different query types to different levels: "Summarize the company's Q3 strategy" → section-level chunks (broad context); "What was EMEA revenue in Q3?" → sentence-level chunks (specific fact).

### 💡 Why This Pattern Keeps Coming Up in Interviews
It directly solves the core tension from Section 1 with a concrete, implementable answer rather than a theoretical statement about tradeoffs. When an interviewer asks *"how do you balance retrieval precision with generation context quality?"* — parent-child is the answer.

### ⚠️ When Parent-Child Fails
- Parent chunks are themselves too large or too noisy, so fetching the parent gives the LLM too much irrelevant content alongside the relevant sentence.
- The document structure doesn't support clear parent-child relationships (continuous prose with no section boundaries — what's the "parent" of a sentence?).
- The parent spans multiple semantically distinct topics, defeating the purpose of expanding for context.

In those cases, sentence-window retrieval is a simpler, more robust fallback.

---

## 4. Industry Standards & Best Practices

**Think of chunk size like a camera's zoom level.** Zoom in too far (tiny chunks) and you see a single pixel with no context. Zoom out too far (huge chunks) and the one detail you needed is buried in a landscape shot.

### Typical Production Chunk Sizes

| Use Case | Typical Chunk Size | Why |
|---|---|---|
| Q&A over short FAQs / support docs | 128–256 tokens | Answers are short and localized |
| General knowledge base RAG | 256–512 tokens | Balances context vs. precision — most common default |
| Long-form technical/legal docs | 512–1024 tokens | Needs more surrounding context to preserve meaning |
| Code retrieval | 1 function/class (AST-bounded) | Natural structural unit is the function, not a token count |
| Conversational/chat memory | 1 turn or a small window of turns | Turns are the natural semantic unit |

### 💡 Why This Matters
There is **no universal correct chunk size** — it depends on:

| Factor | Effect on optimal chunk size |
|---|---|
| Query type | Fact-lookup ("what is X?") → smaller chunks. Synthesis ("summarize Q3 strategy") → larger chunks or hierarchical indexing |
| Embedding model's effective context | Most models degrade well before their stated max — a 512-token-limit model may embed best at 200–300 tokens |
| LLM context budget | top-k=10 at chunk_size=512 consumes 5,120 tokens/query — larger chunks force a smaller k or risk busting the context window |
| Corpus structure | Short FAQ entries → small chunks natural. Long regulatory documents → larger chunks needed for coherence |

256–512 tokens with ~10–20% overlap remains the most common production **starting point**, because it roughly matches a self-contained paragraph while staying inside typical embedding model limits — but it should always be treated as a hyperparameter to tune (Section 8), not copied from a blog post.

### Why 10–20% Overlap Is the Standard Range
- Below ~10%: facts straddling a boundary are frequently lost from both resulting chunks.
- Above ~20%: index size and retrieval cost grow roughly linearly (see the worked bloat example in 2.1), but recall gains flatten — you're mostly paying for near-duplicates occupying top-k slots.
- 10–20% is the empirical "knee of the curve" most teams converge on.

### Common Production Patterns ✅
- **Parent-child chunking** (Section 3) as the default answer to the precision-vs-context tension.
- **Metadata-enriched chunks** (Section 7): every chunk carries source, section path, page number, and timestamp.
- **Re-chunk from raw source**, never from already-chunked text, when hyperparameters change — avoids compounding information loss.

### Common Anti-Patterns ❌
- Chunking after lossy text extraction (table structure destroyed *before* chunking even starts).
- One-size-fits-all chunk size across wildly different document types.
- Treating chunk size as a fixed constant instead of a model- and query-type-dependent hyperparameter.
- Never re-evaluating chunk size after launch or after an embedding model swap.

### 🎯 Interview Gotcha
> "What chunk size should I always use?"

There is no "always." The strong answer names the tradeoff (recall vs. precision vs. cost) and the tuning process (Section 8), not a memorized number.

---

## 5. Multi-Modal & Document-Specific Chunking

**Think of a PDF like a stage play, not a script.** A script (plain text) just gives you words in order. A stage play (PDF) has actors in specific positions, props, a set — a *visual layout* that carries meaning. Naive text extraction is like reading only the dialogue and losing the blocking. **Generic chunkers applied to structured content types produce garbage — always match the chunker to the content type.**

### The Golden Workflow: Layout Analysis → Content Extraction → Intelligent Chunking

```
PDF Input
   │
   ▼
1) LAYOUT ANALYSIS
   - Detect page structure: text blocks, tables, images, headers/footers, columns
   - Tools: LayoutParser, Unstructured, Azure Document Intelligence, AWS Textract,
     PDFPlumber, LlamaParse (or OCR + layout heuristics for scans)
   │
   ▼
2) CONTENT EXTRACTION (per element type)
   - Text blocks  -> plain text, preserving true reading order (not raster order)
   - Tables       -> structured extraction (rows/cols), NOT flattened text
   - Images/charts-> caption/describe with a vision model, don't embed raw pixels
   - Headers/footers/page numbers -> strip from text, store separately as metadata
   │
   ▼
3) INTELLIGENT CHUNKING (type-aware)
   - Text   -> structure-aware / recursive chunking (Section 2.2, 2.4)
   - Tables -> table-specific chunking (Section 6)
   - Code   -> AST-aware chunking (below)
   - Images -> chunk the generated description text, link back to the image asset
   │
   ▼
Chunks + rich metadata (page #, bbox, element type, section path)
```

### 🖼️ Multi-Column PDFs — a Common, Underrated Failure Mode
Naive PDF text extraction reads left-to-right, top-to-bottom *across* columns, producing garbage interleaving for a two-column article:

```
WRONG extraction:
"Column A line 1  Column B line 1  Column A line 2  Column B line 2 ..."

CORRECT extraction:
"Column A line 1  Column A line 2 ... [column break] Column B line 1  Column B line 2 ..."
```

> **Principle:** garbage extraction makes any chunking strategy moot — however smart the chunker, it inherits the lost structure permanently. The fix is *before* chunking: use layout-aware extraction that detects column bounding boxes before assembling text.

### 💻 Code Chunking (AST-Aware)
**Problem:** splitting mid-function is catastrophic — a function body without its signature (`def my_function(arg1, arg2):`) is nearly un-embeddable, because the signature carries most of the semantics.

```python
# WRONG: fixed-size splits here
def calculate_revenue(units, price_per_unit,  # ← chunk 1 ends here
    discount_rate):                             # ← chunk 2 starts mid-signature
    return units * price_per_unit * (1 - discount_rate)

# RIGHT: chunk at function boundary
# Chunk = entire function (signature + body + docstring), never split
def calculate_revenue(units, price_per_unit, discount_rate):
    """Compute net revenue after discount."""
    return units * price_per_unit * (1 - discount_rate)
```

**Fix:** use AST-aware (Abstract Syntax Tree) splitters like `tree-sitter` that parse code into a syntax tree, and chunk at function, class, or module boundaries — never at raw character positions.

### 🖼️ Charts, Diagrams, Images
1. Generate a textual description/caption using a vision-language model ("Bar chart showing Q1–Q4 2024 revenue by region, Q3 South America is the highest at $4.2M").
2. Chunk and embed *that description*, with a pointer back to the original image asset.
3. At answer time, show the LLM both the retrieved description and (if multi-modal) the original image.

Don't embed raw chart/image pixels into a text-based index unless you're specifically using a multi-modal embedding model — most text embedding models can't meaningfully embed images.

### 📑 Headers, Footers, Page Numbers
Usually noise for retrieval (repeated on every page, polluting embeddings with irrelevant repeated tokens). Detect and strip during layout analysis, but **keep the page number as metadata** — valuable for citations ("see page 14").

### 📝 Markdown / Already-Structured Docs
Chunk along the heading hierarchy; each chunk inherits its document location as a breadcrumb:
```json
{
  "text": "For international customers, customs duties are non-refundable...",
  "breadcrumb": "Returns Policy > International Orders > Customs & Duties"
}
```
This breadcrumb enables both metadata-filtered retrieval ("only return chunks from the Returns Policy section") and better standalone interpretability.

### ⚠️ Common Mistakes
- Running a generic PDF-to-text library (that ignores layout) and *then* chunking — bakes column/table errors into every downstream chunk.
- Treating a 2-column paper as one linear text stream.
- Splitting code at arbitrary character positions instead of AST boundaries.

### 🎯 Production Best Practice
Build a **type-aware chunking pipeline**, not a single chunker: route each detected element (paragraph, table, image, code block, header) to a *different* chunking function, then merge into one unified index with a `content_type` metadata field so retrieval can later be boosted or filtered by type (e.g., "if the query looks numerical, boost table-type chunks").

---

## 6. Numerical & Tabular Data Chunking

**Think of a table like a family** — every cell's meaning depends on its relatives: the header above it (what column) and the row label beside it (what row/entity). A cell value like `32.4%` is meaningless without the column header ("Q3 Revenue Growth") and the row label ("EMEA"). Chunk a table the way you'd cut up a family photo — keep people with the people they belong with, or the picture makes no sense.

### The Core Problem
Naively chunking a table by raw character count can split it mid-row, or worse, separate the header row from all data rows:
```
Q3    Q4    2023    2024
1.2M  1.5M  Revenue Growth
```
— numbers completely divorced from what they mean.

### Fix A — Atomic Table Chunks (Small Tables)
Treat the entire table as one chunk. Serialize to markdown or a structured text format; keep it together, no splitting.

### Fix B — Header-Repeated Chunks (Large Tables)
If the table is too large for one chunk, split by logical row groups and **repeat the header row into every chunk**:
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
Each chunk is now self-interpreting, independent of neighboring chunks.

### Other Techniques
- **Column-based chunking:** useful when queries tend to ask about a single metric across all rows ("show revenue trend for all regions") — chunk by column, prepending row labels as context.
- **Semantic/cell-relationship chunking:** for dense tables (e.g., a 50×50 financial model), chunk by *logical sub-table* (all "Revenue" line items together, separate from "Expenses"), preserving each sub-header.
- **Multi-page tables:** a table spanning pages 4–6 must be **stitched into one logical table** during layout analysis — detect repeated header rows across pages as the signal of continuation, then chunk the merged table as usual (never treat each page as an independent table).

### Numerical Precision: Round or Keep Full?

| Situation | Recommendation |
|---|---|
| Exact compliance/audit figures (financial statements, legal filings) | Keep full precision, exactly as reported |
| Trend/comparison queries ("did revenue grow?") | Rounding to 1–2 significant figures is fine and improves LLM reasoning reliability |
| Long decimal chains (scientific measurements) | Keep full precision in the chunk; consider rounding only in the *displayed* answer |

### 💡 Why This Matters
LLMs are known to struggle with precise multi-digit arithmetic read directly from context. For financial/scientific use cases, the safer production pattern is: chunk with full precision preserved, but pair retrieval with a **calculator/code-execution tool** for any arithmetic the question requires, rather than trusting the LLM to compute `4.2M - 3.1M` correctly by "reading" the table.

### Formatting Tables for LLM Consumption

| Format | Pros | Cons |
|---|---|---|
| Markdown table | Compact, LLMs are heavily trained on it, human-readable | Breaks on very wide tables |
| HTML table | Preserves merged cells/complex structure | Verbose, more tokens |
| Structured text ("Row: X, Col: Y, Value: Z") | Most robust for retrieval — each fact explicit and independently retrievable | Most verbose, more chunks |

**Production best practice:** store the table in **structured text or JSON form** for retrieval/chunking, but render it as a **markdown table** in the final prompt shown to the LLM.

### Code Example: Header-Aware Row Chunking
```python
def chunk_table_by_row(headers, rows, table_title=""):
    chunks = []
    for row in rows:
        row_desc = ", ".join(f"{h}: {v}" for h, v in zip(headers, row))
        chunk_text = f"Table: {table_title}\n{row_desc}"
        chunks.append(chunk_text)
    return chunks

headers = ["Region", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
rows = [
    ["South America", "3.1", "3.4", "4.2", "3.9"],
    ["North America", "8.5", "8.9", "9.1", "9.6"],
]
chunks = chunk_table_by_row(headers, rows, "Quarterly Revenue ($M)")
# chunks[0] -> "Table: Quarterly Revenue ($M)\nRegion: South America, Q1 2024: 3.1, ..."
```

### Financial Reports, Scientific Papers, Data Sheets
- **Financial reports:** always retain reporting period and currency/unit as chunk metadata (`unit: "$M"`, `period: "Q3 2024"`) — a number without units is a bug waiting to happen.
- **Scientific papers:** chunk equations with the surrounding sentence that defines each variable — an equation without variable definitions is unusable for retrieval.
- **Data sheets:** chunk by product/spec-group with the product name repeated in every chunk, so it stays retrievable even if the name only appears once at the top of the original sheet.

### 🎯 Interview Gotcha
> "How would you answer 'What was the ROI for Q3 2024?' if the table cell doesn't literally say 'ROI'?"

Tests whether you understand retrieval needs *derived context*, not just literal keyword match. If ROI must be computed from Revenue and Cost columns, the chunk must include both related columns together (not split across separate row-only chunks), and ideally the pipeline routes to a calculation step rather than expecting the raw chunk to contain a pre-computed value. Fully worked in Section 9, Scenario F.

---

## 7. Metadata Enrichment Per Chunk

Beyond storing raw chunk text, attach structured fields usable for filtering, re-ranking, or improving retrievability. Four types worth knowing cold:

### Type 1 — Breadcrumb / Section Hierarchy
```json
{
  "text": "International returns must be initiated within 14 days of delivery.",
  "source": "policy_manual_v3.pdf",
  "section": "Returns Policy > International Orders",
  "page": 47
}
```
Enables filtered retrieval ("search only within the Returns Policy section"). Without this, a query about returns might retrieve unrelated content that happens to use the word "return."

### Type 2 — Auto-Generated Chunk Summaries
For noisy chunks (dense tables, boilerplate-heavy regulatory text), generate a clean LLM summary and embed the summary instead of (or alongside) the raw text.

*Why:* a dense earnings table embedded as raw text ("Revenue 1,234,567 1,189,234 Cost of Goods 456,789...") produces a poor embedding. A summary ("Q3 2024 revenue grew 3.8% YoY to $1.23B, driven by APAC expansion") embeds far better — it's natural language with real semantic content.

### Type 3 — Hypothetical Questions (HyQ)
Generate 3–5 synthetic questions that a chunk would answer, and embed the *questions* (not the chunk text) in the index.

```
Chunk text: "For international customers, customs duties are non-refundable and will not be credited."

Synthetic questions embedded:
- "Are customs fees refundable for international orders?"
- "What happens to import duties if I return an item from abroad?"
- "Can I get a refund on customs charges?"
```

*Why this works:* real user queries are questions; chunk text is declarative prose. The distributional gap between query space and document space is the core challenge of asymmetric retrieval. Embedding synthetic questions instead of raw text closes this gap at *ingestion* time — this is the ingestion-time cousin of HyDE (Hypothetical Document Embeddings), which is the *query-time* version of the same idea.

### Type 4 — Source / Temporal / Access Metadata
```json
{
  "last_updated": "2024-09-15",
  "access_level": "internal",
  "document_type": "policy",
  "jurisdiction": "EU"
}
```
Not for retrieval quality directly, but for filtering: recency filtering ("only search documents updated in the last 6 months"), access control ("only return results this user is permitted to see"), and namespace filtering ("only return EU-jurisdiction policy documents").

---

## 8. Evaluation & the Empirical Tuning Loop

**Think of evaluation like taste-testing a recipe before serving it at a dinner party.** You don't just guess that "more salt is better" — you make small batches, taste them side by side, and measure against what your guests actually want.

### Key Metrics

| Metric | What It Measures | How to Compute |
|---|---|---|
| **Recall@k** | Did the correct chunk appear in the top-k retrieved results? | Labelled eval set (query → known-relevant passage); check overlap |
| **MRR (Mean Reciprocal Rank)** | How high up was the first relevant chunk ranked? | Standard IR metric over the labelled set |
| **Context relevance** | Of the chunks retrieved, how much is actually relevant? | LLM-as-judge or human annotation on retrieved chunks |
| **Answer correctness** | Does the final generated answer match ground truth? | Exact match / semantic similarity / LLM-as-judge |
| **Faithfulness / groundedness** | Is the answer actually supported by the retrieved chunks (not hallucinated)? | LLM-as-judge, claim-by-claim support check |

### There Is No Universal Correct Chunk Size — Tune It

| Factor | Effect on optimal chunk size |
|---|---|
| Query type | Fact-lookup → smaller chunks; synthesis → larger chunks or hierarchical indexing |
| Embedding model's effective context | Degrades well before the stated max in many models |
| LLM context budget | top-k × chunk_size must fit comfortably in the context window |
| Corpus structure | FAQ entries → small chunks natural; regulatory text → larger chunks needed |

### How to Tune Systematically (Say This Out Loud in Interviews)
> "I'd tune chunk size empirically with an eval set" is what separates a strong answer from guesswork.

**Step 1 — Assemble a golden eval set:** 50–200+ (query, relevant passage) pairs — sampled from user logs, manually annotated, or synthetically generated.

**Step 2 — Define the metric:** Recall@k for retrieval in isolation; answer correctness for end-to-end.

**Step 3 — Grid search over (chunk_size, overlap):**
```python
chunk_sizes = [128, 256, 512, 1024]
overlaps    = [0, 32, 64, 128]

for chunk_size in chunk_sizes:
    for overlap in overlaps:
        rebuild_index(chunk_size, overlap)
        recall = evaluate_recall_at_k(eval_set, k=5)
        log(chunk_size, overlap, recall)
```

**Step 4 — Find the knee of the curve:** plot Recall@k vs. chunk_size — recall usually improves steeply, then plateaus. The knee is the sweet spot; going past it just wastes context budget.

**Step 5 — Re-validate after any embedding model change:** the optimal chunk size is model-dependent — a model with stronger long-context representation may benefit from larger chunks than one that degrades past ~200 tokens.

**Step 6 — Segment results by document/query type** and **roll out gradually** (shadow-test before fully switching the index) — an aggregate win can hide a regression on, say, table-heavy queries specifically.

### Common Pitfalls in Evaluation ⚠️
- Evaluating only end-to-end answer quality conflates chunking, retrieval, and generation quality — always measure retrieval metrics in isolation too.
- Small, unrepresentative golden sets that happen to favor one chunk size by luck.
- Ignoring latency/cost in the tradeoff — a 2% recall gain that triples ingestion cost may not be worth shipping.
- Testing only easy factoid queries, missing multi-hop or table-lookup queries that are far more chunking-sensitive.

### 🎯 Interview Gotcha
> "Your retrieval Recall@5 looks great (95%) but users still complain answers are wrong. What's going on?"

High recall means the *right chunk is being retrieved* — so the bug is downstream: maybe the chunk is retrieved but truncated before the LLM sees it, maybe the LLM is ignoring the context (need a faithfulness/groundedness check), or maybe the "correct chunk" in the golden set isn't actually sufficient to answer the question on its own — a chunking *granularity* problem, even with correct retrieval.

---

## 9. Worked Scenarios

### Scenario A: PDF with Text + Tables + Images
Layout analysis first (Section 5) detects three distinct element types on the page. Route text through structure-aware chunking, tables through header-aware row chunking (Section 6), images through a vision-model caption step whose output text is chunked normally. All three share page number and section-path metadata so a query can retrieve across types and the LLM can be told "here's the relevant paragraph, table row, and chart description from page 12."

### Scenario B: Multi-Page Table Spanning 3 Pages
During layout analysis, detect that the header row on page 5 matches the header row on page 4 — the signal of continuation, not a new table. Stitch all rows from pages 4–6 into one logical table before chunking, then apply header-aware row chunking, storing page number per-row so citations stay accurate even though rows are logically merged.

### Scenario C: Dense Numerical Spreadsheets
Avoid one giant "flatten the sheet to text" chunk. Chunk by logical sub-table/section within the sheet (e.g., "Revenue" block separate from "Expenses" block), using header-aware row chunking within each block, with the sheet name and surrounding label cells captured as metadata.

### Scenario D: Legal Documents with Complex Structure
Structure-aware chunking is essential — legal documents are nested numbered clauses ("Section 4 > 4.2 > 4.2.1") where meaning depends heavily on hierarchy. Chunk at the clause level, always including the full heading path in metadata, and consider including a brief parent-section summary for context when a sub-clause is retrieved in isolation (a natural application of Section 3's parent-child pattern).

### Scenario E: Technical Papers with Equations
Never chunk an equation in isolation — include the sentence(s) immediately before/after that define the variables, keeping an equation with its surrounding paragraph as one chunk rather than splitting equation from explanation.

### Scenario F: "What's the ROI for Q3 2024?" (Table + Context Retrieval) — Fully Worked
"ROI" is a **derived** metric, not a literal cell value — a favorite interview scenario for exactly that reason.

1. **Chunking time:** Revenue and Cost rows/columns for Q3 2024 must live in chunks retrievable *together* — a header-aware chunk containing the full Q3 2024 column across Revenue, Cost, and Net Income, not split into separate per-metric chunks.
2. **Retrieval time:** the query embeds and should match chunks containing "Q3 2024" + financial terms; metadata filtering on `period: "Q3 2024"` further narrows candidates.
3. **Generation time:** the LLM receives Revenue and Cost for Q3 2024 and either computes ROI directly (simple case) or — more robustly — the answer routes through a **code-execution/calculator tool** to compute `(Revenue - Cost) / Cost` reliably, rather than trusting free-form LLM arithmetic.
4. **Key takeaway:** the chunking decision (keeping Revenue and Cost together) is what makes step 3 even *possible* — the clearest illustration of why "preserving relationships" in tabular chunking determines whether a derived-metric question can be answered at all.

---

## 10. Decision Tree

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

## 11. Final Interview Cram Sheet & Q&A Drill

### Rapid-Fire

**Q: What's the single most important tradeoff in chunk size selection?**
A: Recall vs. precision — smaller chunks improve precision but hurt recall/context (facts split across chunks); larger chunks do the opposite. Overlap partially mitigates the recall loss from small chunks, at the cost of index bloat.

**Q: Why is fixed-size chunking still used in production despite being "worse" semantically?**
A: It's essentially free, fully predictable, and for homogeneous unstructured text the semantic loss is often small — legitimate when speed/cost dominate and content is uniform.

**Q: What's the biggest mistake teams make with PDF chunking?**
A: Extracting to plain text *before* preserving layout, permanently destroying table structure and column reading order before chunking even begins.

**Q: How do you evaluate whether your chunking strategy is good?**
A: A golden Q&A eval set with known source chunks, measuring Recall@k/MRR for retrieval in isolation (not just end-to-end answer quality), A/B'd across chunking variants with everything else held constant.

**Q: What's "parent-child chunking" and why does it matter?**
A: Embedding small, precise child chunks for retrieval accuracy, but returning the larger parent chunk/section to the LLM for full context — combines retrieval precision with generation-time context richness; the concrete answer to the core precision/recall tension.

**Q: Trick question — should you always maximize retrieval recall?**
A: No — recall must be balanced against precision and latency/cost; retrieving 50 chunks to guarantee the answer is present buries the LLM in noise and increases hallucination risk. The goal is the *smallest sufficient context*, not the largest possible one.

### Deep-Dive Q&A Drill

---

**Q: You increased chunk size and retrieval recall went up, but answer quality went down. Explain.**

A: Larger chunks are more likely to *contain* the answer (higher recall — the target text is less likely to be split across chunks). But the embedding of a larger chunk is a vector average over more content, which dilutes precision — it doesn't pinpoint the relevant section as sharply, and more irrelevant text gets bundled into the retrieved context. The LLM then receives the relevant passage plus surrounding noise, increasing the chance of incorrect synthesis. The fix is usually parent-child chunking: small chunks for retrieval precision, expand to the parent for generation.

---

**Q: When would you choose semantic chunking over recursive structure-aware chunking?**

A: When retrieval precision materially affects real-world outcomes and the corpus warrants the extra ingestion cost — legal contracts where a missed clause costs money, medical records where misattributing a symptom causes harm, financial filings where one number in the wrong context changes a decision. For high-volume, lower-stakes content, recursive structure-aware chunking is sufficient and avoids the two-pass embedding cost.

---

**Q: How do you chunk a 50-page PDF containing both prose and financial tables?**

A: Treat it as two problems. First, layout-aware extraction (e.g., PDFPlumber or a document intelligence service) separates prose from table regions *before* chunking — naive extraction interleaves multi-column text and destroys table structure. Then: recursive/structure-aware chunking for prose; atomic or header-repeated chunking for tables (full table as one chunk where feasible; header row repeated into every chunk when tables must be split). Never apply a single generic text splitter to mixed-content PDFs.

---

**Q: What's the actual difference between "chunking" and "indexing," and why do candidates conflate them?**

A: Chunking is a *data* decision — what units of text to create, resolved via labelled eval sets and corpus analysis. Indexing is a *systems* decision — how to store and search those units at scale, resolved via latency benchmarks and memory profiling. They're conflated because they happen back-to-back in the ingestion pipeline, but a great chunking strategy on a poorly-chosen index retrieves the right content slowly; a great index over badly-chunked data retrieves the wrong content quickly.

---

**Q: Why does parent-child chunking help, and when would it fail?**

A: It helps by decoupling the retrieval unit (small, precise) from the generation unit (large, contextually complete). It fails when parent chunks are themselves too large/noisy, when the document has no clear section boundaries to define a "parent," or when the parent spans multiple distinct topics — defeating the purpose of expanding for context. Sentence-window retrieval (±k sentences) is the simpler, more robust fallback in those cases.

---

**Q: A user asks: "Summarize our company's Q3 strategy across all business units." How should your chunking strategy handle this?**

A: Synthesis queries are inherently multi-chunk — the answer requires assembling information from many parts of the corpus, so small chunks hurt (the answer isn't in one place). Two approaches: (1) classify the query as synthesis vs. fact-lookup at query time and route to larger chunks or section-level summaries for synthesis, fine-grained chunks for fact-lookup; (2) generate and index document-/section-level summaries as metadata at ingestion time specifically for this routing. The underlying principle: match retrieval granularity to query type.

---

### Key Gotchas Summary Table

| Gotcha | Correct Understanding |
|---|---|
| "Overlap is always beneficial" | Overlap inflates index size and can waste top-k slots on near-duplicate chunks |
| "Bigger chunks always improve context" | Bigger chunks dilute embeddings and pack irrelevant noise into LLM context |
| "Fixed-size chunking is fine for everything" | Catastrophically wrong for tables, code, and multi-column PDFs |
| "Semantic chunking is always worth it" | Only for high-value corpora — expensive at ingestion scale |
| "Chunk size is a fixed constant" | Model-dependent and query-type-dependent — tune empirically per corpus |
| "Better extraction doesn't matter once you have chunking" | Garbage extraction makes any chunking strategy irrelevant — fix extraction first |
| "Chunking and indexing are the same thing" | Different problem types: chunking is a data/eval problem, indexing is a systems/scale problem |

---

*Good luck with your interviews. The strongest answers always connect the "why" (the tradeoff being solved) to the "how" (the specific technique) — an interviewer would rather hear you reason through a novel document type live than recite a memorized strategy name.*
