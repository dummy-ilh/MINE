

# 🧩 Context Optimization: Chunking, Windowing & Compression

By now you’ve built:

```
Rewrite → Hybrid Retrieve → Rerank → Generate
```

But here’s a brutal truth:

> **Most RAG systems fail because of bad chunking.**

Even perfect retrieval cannot fix poorly structured context.

Today we optimize the *atomic unit* of RAG:
👉 **The chunk**

---

# 1️⃣ Why Chunking Is Critical

Imagine a 20-page document.

If you:

* Embed the whole thing → semantic dilution
* Split randomly every 500 tokens → broken meaning
* Split too small → context fragmentation

Chunking directly affects:

* Retrieval recall
* Ranking precision
* Hallucination risk
* Token cost

---

# 🧠 Core Principle

> A chunk should represent one coherent idea.

Not:

* Half a paragraph
* Two unrelated sections
* An entire chapter

Think in terms of **semantic atomicity**.

---

# 2️⃣ Fixed-Size Chunking (Baseline)

Example:

```
Split every 512 tokens with 50-token overlap
```

Pros:

* Simple
* Fast
* Works okay for uniform text

Cons:

* Cuts meaning arbitrarily
* Breaks tables and lists
* Hurts legal/technical docs

Good for:

* Blog content
* FAQs
* Short documents

---

# 3️⃣ Semantic Chunking (Better)

Instead of token length, split by:

* Paragraph boundaries
* Headings
* Section markers
* Sentence clustering

Approach:

1. Split by paragraphs
2. Merge small ones
3. Keep chunks 300–800 tokens

This preserves meaning structure.

---

# 4️⃣ Overlap Strategy

Overlap prevents boundary information loss.

Example:

```
Chunk 1: Tokens 0–500
Chunk 2: Tokens 450–950
```

Why overlap works:

* Questions often span chunk boundaries
* It increases recall
* Slightly increases storage cost

Typical overlap:

* 10–20%

Too much overlap → duplication noise
Too little → recall drops

---

# 5️⃣ Sliding Window Retrieval

Instead of static chunks:

1. Retrieve candidate doc
2. Slide window across doc
3. Select best matching window

This reduces semantic dilution.

Used in advanced search systems like:

* Google search
* Microsoft enterprise search

---

# 6️⃣ Hierarchical Retrieval

Very powerful pattern.

```
Level 1: Retrieve section
Level 2: Retrieve chunk inside section
```

Instead of indexing tiny chunks directly:

* Index sections
* Then refine inside

This reduces false positives.

---

# 7️⃣ Context Compression (Advanced)

Even after reranking, context can be noisy.

We compress before sending to LLM.

## Method A: Extractive Compression

Use smaller model to extract:

* Only relevant sentences
* Remove boilerplate

## Method B: LLM Compression

Prompt:

> “Extract only information relevant to the query.”

This reduces:

* Token cost
* Attention dilution
* Hallucination risk

---

# 8️⃣ Lost-in-the-Middle Problem

Large context windows suffer from:

> Middle content gets less attention.

Even long-context models struggle here.

Solution:

* Put most relevant chunk at top
* Order by relevance score
* Keep context small and precise

---

# 9️⃣ Chunk Size Tradeoff

| Small Chunks            | Large Chunks     |
| ----------------------- | ---------------- |
| High precision          | High recall      |
| Lower semantic dilution | More noise       |
| More embeddings         | Fewer embeddings |

Empirically:

* 300–800 tokens often best
* Domain dependent

---

# 🔬 Experimental Insight

Research shows:

Better chunking can improve retrieval more than switching embedding models.

Yes — chunking matters more than model choice sometimes.

---

# 🏗 Ideal Production Pattern

```
Ingestion:
- Semantic split
- 500 tokens
- 15% overlap
- Metadata tagging

Retrieval:
- Hybrid search
- Rerank

Pre-Generation:
- Compress to relevant spans
- Order by relevance

Generation:
- Strong grounding prompt
```

This is near state-of-the-art practical RAG.

---

# 🧠 Deep Insight

Think of chunking like:

> Database schema design for vector search.

If your schema is bad, queries will suffer.

---

# 🧪 Today’s Exercise

1. Compare:

   * 200-token chunks
   * 500-token chunks
   * 1000-token chunks
2. Measure:

   * Recall@5
   * MRR
   * Latency
3. Try semantic splitting instead of fixed-size.
4. Observe which failure cases disappear.

---

# 🔥 Critical Thinking

Why do extremely large context windows (like 100k tokens) NOT eliminate the need for good chunking?

Think carefully — this is subtle and important.

---


