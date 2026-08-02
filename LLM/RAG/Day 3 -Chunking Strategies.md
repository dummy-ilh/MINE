# RAG Interview Prep — Day 3
## Chunking Strategies

---

## 🚀 Quick Summary

Chunking is the process of splitting long documents into smaller retrievable units *before* they get embedded and indexed (Day 2), and it is arguably the highest-leverage, most-underrated decision in the entire RAG pipeline — a bad chunking strategy silently caps retrieval quality no matter how good your embedding model or search algorithm is. Today covers the four main strategies (fixed-size, sliding window, recursive, semantic), the quantifiable trade-offs between chunk size and retrieval quality, structure-aware chunking for real-world documents (tables, code, markdown), and the "small-to-big" pattern that resolves the core tension between search precision and generation context.

**Think of it like cutting a cake for a buffet.** Serve the whole cake as one slab and nobody can grab a manageable piece — that's "no chunking," where a whole document gets embedded as one blurry-average vector. Cut it into crumbs and each piece is meaningless alone — that's chunks too small, stripped of context. The goal is slices: big enough to be a complete, coherent idea, small enough that grabbing one or two actually answers the question.

---

## 🔑 Key Concepts

| Term | One-line definition |
|---|---|
| **Chunk** | A retrievable unit of text — the thing that actually gets embedded, indexed, and returned by search |
| **Chunk size** | The target length of a chunk, in tokens or characters |
| **Overlap** | Shared content between consecutive chunks, meant to reduce the chance of splitting an idea across a chunk boundary |
| **Stride** | How far forward the next chunk starts = chunk_size − overlap |
| **Recursive splitting** | Trying natural boundaries first (paragraph → sentence → word), falling back to smaller units only when needed |
| **Semantic chunking** | Using embeddings to detect topic-shift boundaries and splitting there, instead of at a fixed size |
| **Structure-aware chunking** | Respecting a document's native structure (markdown headers, code blocks, table rows) as chunk boundaries |
| **Small-to-big retrieval (parent-document retrieval)** | Searching over small chunks for precision, but returning their larger parent context to the generator for completeness |

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

## When to use each strategy / when not to

- ✅ **Fixed-size** — fast prototyping, homogeneous unstructured text where structure doesn't carry meaning (e.g., transcripts with no natural paragraph breaks)
- ❌ Not for anything where cutting mid-sentence or mid-instruction is costly (legal clauses, step-by-step instructions)
- ✅ **Recursive** — the sensible production default for most real-world documents; respects structure without the cost of semantic chunking
- ✅ **Semantic** — high-value corpora where topical coherence really matters (legal, medical) and you can afford extra embedding calls at ingestion time
- ❌ Not worth it for huge low-value corpora where the extra embedding-call cost at ingestion doesn't pay for itself
- ✅ **Structure-aware** — anything with native structure: markdown docs, code, tables, API references — ignoring structure here is a very avoidable mistake

---

# PHASE 2 — Math & Formulas

## Notation table

| Symbol | Meaning |
|---|---|
| `L` | Document length (tokens) |
| `C` | Chunk size (tokens) |
| `O` | Overlap (tokens) |
| `S` | Stride = C − O |
| `N` | Number of resulting chunks |

---

### Stride and Chunk Count

```
S = C - O
N ≈ ⌈(L - O) / S⌉
```

**Plain English:** Stride is how far forward the "window" moves for each new chunk. If chunk size is 200 and overlap is 50, each new chunk only advances 150 tokens past where the previous one started — the 50-token overlap region gets indexed twice (once at the end of chunk *i*, once at the start of chunk *i+1*).

**Worked numerical example — three overlap settings on the same document, so you can see the trade-off directly:**

Document length `L = 3000` tokens, chunk size `C = 300` tokens. Compare **no overlap**, **modest overlap (50)**, and **heavy overlap (150)**.

**No overlap (O = 0):**
```
S = 300 - 0 = 300
N ≈ ⌈(3000 - 0)/300⌉ = ⌈10⌉ = 10 chunks
total indexed tokens = 10 × 300 = 3000  (0% redundancy)
```

**Modest overlap (O = 50):**
```
S = 300 - 50 = 250
N ≈ ⌈(3000 - 50)/250⌉ = ⌈11.8⌉ = 12 chunks
total indexed tokens = 12 × 300 = 3600  (20% redundancy over the original 3000)
```

**Heavy overlap (O = 150, i.e. 50% overlap):**
```
S = 300 - 150 = 150
N ≈ ⌈(3000 - 150)/150⌉ = ⌈19⌉ = 19 chunks
total indexed tokens = 19 × 300 = 5700  (90% redundancy over the original 3000)
```

**Why this matters in practice:** Overlap is a direct, quantifiable dial between **boundary-safety** and **cost** — more overlap means fewer ideas get awkwardly split across chunk boundaries, but you pay for it in embedding calls (at ingestion), storage, and index size, roughly linearly. Going from 0% to 20% redundancy (modest overlap) is usually a very good trade; going to 90% redundancy (heavy overlap) rarely buys enough additional boundary-safety to justify nearly doubling your storage and embedding cost — this is the kind of trade-off table an interviewer wants you to reason through live, not just recite.

| Overlap setting | Chunks (N) | Total indexed tokens | Redundancy | Trade-off |
|---|---|---|---|---|
| 0 tokens | 10 | 3000 | 0% | Cheapest, highest risk of boundary splits |
| 50 tokens (~17%) | 12 | 3600 | +20% | Good default — most boundary-split risk removed at modest cost |
| 150 tokens (50%) | 19 | 5700 | +90% | Rarely worth it — diminishing returns past a certain overlap ratio |

---

## Semantic Chunking — How the Breakpoint Detection Actually Works

**Plain English mechanism:** Instead of a fixed size, semantic chunking embeds small units (e.g., individual sentences), then walks through the document comparing the similarity between consecutive sentence embeddings. When similarity **drops below a threshold** (i.e., the topic shifts), that's marked as a chunk boundary.

**Worked conceptual example:**
```
Sentence 1: "AirPods Pro have active noise cancellation."
Sentence 2: "The noise cancellation adapts in real time to your environment."
   → cosine similarity(S1, S2) = 0.91  (high — same topic, no break)

Sentence 3: "Return window for AirPods Pro is 14 days from purchase."
   → cosine similarity(S2, S3) = 0.38  (low — topic shifted, INSERT BOUNDARY HERE)
```
The chunk boundary is placed between sentence 2 and sentence 3, because that's where the semantic "distance" spikes — grouping S1+S2 into one coherent chunk about noise cancellation, and starting a new chunk at S3 about the return policy.

**Cost trade-off to state explicitly:** This requires an embedding call *per sentence* just to find good boundaries, on top of the embedding calls you'll make for the final chunks themselves — meaningfully more expensive at ingestion time than fixed-size or recursive splitting, which is exactly why it's usually reserved for corpora where coherence quality matters enough to justify it.

---

## Structure-Aware Chunking

**The idea:** Real documents already contain structural signals that tell you where natural, meaning-preserving boundaries are — don't throw this information away by treating every document as a flat wall of text.

| Content type | Structural signal to respect | What goes wrong if ignored |
|---|---|---|
| **Markdown / docs** | Headers (`#`, `##`), bullet lists | A chunk boundary lands mid-list, separating a heading from its own content, losing the context of what the list is even about |
| **Code** | Function/class boundaries, indentation blocks | Splitting a function in half produces a chunk with unmatched brackets and no coherent standalone meaning |
| **Tables** | Row/column structure | A fixed-size split mid-table produces a chunk with header-less numbers — completely uninterpretable out of context |
| **Legal contracts** | Clause/section numbering | Splitting mid-clause can separate an obligation from its conditions, changing the effective meaning of the retrieved fragment |

**Practical technique — keep headers attached to their content:** A common production pattern is to prepend the relevant markdown header path (e.g., `"Support > AirPods > Returns"`) to every chunk under that header, even if the header text itself is physically far from the chunk in the original document. This gives every chunk enough standalone context to be embedded meaningfully, even after being pulled out of its original surrounding document.

---

## Small-to-Big (Parent-Document) Retrieval — Resolving the Core Chunking Tension

**The tension:** Small chunks are *better for search precision* (a sharp, focused embedding matches a specific query well), but *worse for generation* (too little surrounding context for the LLM to produce a complete, well-grounded answer). Large chunks are the opposite. Fixed-size/recursive/semantic chunking alone forces you to pick one size and live with both sides of that trade-off.

**The pattern:** Index **small** chunks for search (e.g., single sentences or short 100-token pieces) — but store a pointer from each small chunk back to a **larger parent chunk** (e.g., the full paragraph or section it came from). At query time:
1. Search over the small chunks (sharp, precise matching)
2. Once you know *which* small chunks matched, retrieve their **larger parent chunks** instead
3. Feed those larger parent chunks to the generator (full context for a complete, coherent answer)

```
   SEARCH INDEX (small chunks)         RETURNED TO GENERATOR (parent chunks)

   [sentence-level chunk] ──matches──▶  [full paragraph/section
    "14-day return window"               containing that sentence,
                                          plus surrounding context]
```

**Why it matters in practice:** This pattern is a direct, elegant answer to "how do you balance retrieval precision against generation completeness" — a very common system-design-style interview question. It says: don't pick one chunk size and accept the trade-off; decouple the unit you *search* over from the unit you *generate* from.

---

## Chunk Size Selection — The Evaluation-Driven Approach

**The honest answer to "what chunk size should I use":** There's no universally correct number — it should be determined empirically via a sweep against your golden eval set (Module 7, §7.6), not picked from a blog-post default.

**Practical sweep methodology:**
1. Pick a small set of candidate chunk sizes (e.g., 128, 256, 512, 1024 tokens) and overlap ratios (e.g., 0%, 15%, 30%)
2. Re-index the corpus at each configuration
3. Run the golden eval set against each configuration, measuring Recall@k and, ideally, downstream faithfulness/answer relevance too — not just retrieval metrics in isolation, since chunk size affects both stages
4. Pick the configuration that best balances retrieval quality against indexing/storage cost, rather than chasing the single highest metric regardless of cost

> **Why This Matters callout:** If asked "how would you pick a chunk size for a new RAG system," the weak answer is a fixed number ("I'd use 512 tokens"). The strong answer describes the *sweep methodology* — because the right chunk size is corpus- and query-distribution-dependent (a legal-contract corpus with dense, precise clauses behaves very differently from a casual FAQ corpus), and stating that you'd validate empirically against a golden eval set shows you understand chunk size as a tunable hyperparameter, not a fixed constant.

---

# PHASE 3 — Interview Q&A Practice Set

*(Answers are separated below each question — cover them and self-test first.)*

---

**Q1 (Easy — conceptual).** Why is chunking considered one of the highest-leverage decisions in a RAG pipeline?

<details>
<summary>Show answer</summary>

Every downstream step — embedding, retrieval, generation — operates on chunks, not raw documents. A chunk containing one coherent idea produces a sharp, accurate embedding; a chunk crammed with multiple unrelated ideas produces a blurred, ambiguous embedding that matches poorly against any single-topic query. Because this happens at the very first stage of the pipeline, a bad chunking decision caps the ceiling of everything built on top of it, regardless of how good the embedding model, retrieval algorithm, or generator are.
</details>

---

**Q2 (Easy — calculation).** A 4000-token document is chunked with chunk size 400 and overlap 100. Compute the stride, approximate number of chunks, and the redundancy percentage.

<details>
<summary>Show answer</summary>

```
S = 400 - 100 = 300
N ≈ ⌈(4000-100)/300⌉ = ⌈13⌉ = 13 chunks
total indexed tokens = 13 × 400 = 5200
redundancy = (5200 - 4000)/4000 = 30%
```
</details>

---

**Q3 (Medium — conceptual).** How does semantic chunking decide where to place a chunk boundary, and what's the main cost trade-off compared to fixed-size or recursive chunking?

<details>
<summary>Show answer</summary>

Semantic chunking embeds small units (typically individual sentences) and computes similarity between consecutive sentence embeddings as it walks through the document. When similarity drops sharply (indicating a topic shift), a chunk boundary is placed there, grouping high-similarity consecutive sentences into the same chunk. The main cost trade-off is that this requires an embedding call per sentence just to find good boundaries — meaningfully more expensive at ingestion time than fixed-size (no embedding calls needed for splitting) or recursive (uses structural heuristics, not embeddings) chunking — so it's typically reserved for high-value corpora where topical coherence materially affects downstream quality.
</details>

---

**Q4 (Medium — conceptual).** What problem does small-to-big (parent-document) retrieval solve, and how does it work?

<details>
<summary>Show answer</summary>

It resolves the tension between search precision (favoring small, focused chunks with sharp embeddings) and generation completeness (favoring larger chunks with enough surrounding context for a coherent answer). Instead of picking one chunk size and accepting both sides of that trade-off, small chunks are indexed and searched for precise matching, but each small chunk stores a pointer to a larger "parent" chunk (e.g., the full paragraph or section). At query time, search happens over the small chunks, but the larger parent chunks are what actually get retrieved and passed to the generator — decoupling the unit you search over from the unit you generate from.
</details>

---

**Q5 (Medium — conceptual).** A team indexes API documentation and technical tables using fixed-size 200-token chunking with no structural awareness. What specifically goes wrong, and what would you recommend instead?

<details>
<summary>Show answer</summary>

Fixed-size chunking with no structural awareness will frequently split tables mid-row or mid-column and split code blocks or function definitions across chunk boundaries, producing fragments that are uninterpretable out of context (numbers with no header, code with unmatched brackets and no coherent standalone meaning). I'd recommend structure-aware chunking that respects the document's native boundaries — treat whole tables (or logical row groups) and whole functions/code blocks as atomic units where possible, and where a document has header hierarchy, keep the header path attached to (e.g., prepended to) each chunk so it retains standalone context even after being separated from its original document position.
</details>

---

**Q6 (Hard — synthesis / trade-off reasoning).** How would you actually determine the right chunk size and overlap for a brand-new RAG system, rather than guessing a default? Walk through your methodology.

<details>
<summary>Show answer</summary>

I'd treat chunk size and overlap as tunable hyperparameters and validate them empirically rather than picking a fixed default from general advice. Methodology: (1) build or bootstrap a golden eval set (Module 7, §7.6) representative of real query patterns for this corpus; (2) pick a small grid of candidate chunk sizes and overlap ratios; (3) re-index the corpus at each configuration; (4) run the golden eval set against each, measuring not just Recall@k in isolation but also downstream generation metrics like faithfulness and context relevance, since chunk size affects both the retrieval stage and how cleanly the generator can use what's retrieved; (5) select the configuration that best balances retrieval/generation quality against indexing and storage cost, rather than chasing the single highest metric regardless of cost — since very fine chunking or heavy overlap both increase cost with diminishing quality returns past a point.
</details>

---

# 🧠 Gotchas — Common Mistakes Recap

- ❌ Chunking with zero overlap and no respect for natural boundaries — the single most common beginner mistake, silently splits critical sentences/instructions.
- ❌ Assuming heavier overlap is always better — redundancy (and cost) grows fast while boundary-safety gains diminish past a modest overlap ratio.
- ❌ Ignoring native document structure (tables, code, markdown headers) and treating every corpus as flat unstructured text.
- ❌ Picking one chunk size and accepting the full precision-vs-completeness trade-off, instead of considering small-to-big retrieval to decouple search granularity from generation context.
- ❌ Choosing a chunk size from a blog-post default instead of sweeping against a golden eval set for your specific corpus and query distribution.
- ❌ Forgetting to store chunk metadata (source doc, position, header path, parent-chunk pointer) at chunking time — expensive to reconstruct later.

---

# 📌 Cheat Sheet (Day 3)

**Strategies:** Fixed-size (fast, cuts sentences) → sliding window w/ overlap (`stride = chunk_size − overlap`, quantifiable redundancy cost) → recursive (respects structure, solid production default) → semantic (embeds sentences, splits at similarity drops, most coherent but most expensive at ingestion) → structure-aware (respects tables/code/headers explicitly).

**Overlap trade-off:** modest overlap (~15-20%) removes most boundary-split risk cheaply; heavy overlap (~50%+) rarely worth the near-doubling of storage/embedding cost.

**Small-to-big retrieval:** search small chunks for precision, return large parent chunks for generation completeness — decouples the two competing needs instead of compromising on one chunk size.

**Chunk size selection:** never a fixed default — sweep chunk size × overlap against a golden eval set, measure retrieval *and* generation metrics, balance quality against cost.

---

*End of Day 3. Next up — Day 4: Vector Databases & Indexing.*
