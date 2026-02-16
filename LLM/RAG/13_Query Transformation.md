# 🚀 RAG Mastery – Day 13

## 🔎 Query Transformation & Multi-Hop Retrieval

Today we move from **basic retrieval** → **intelligent retrieval**.

Up to now, your RAG pipeline probably looks like:

```
User Query → Embed → Retrieve → Rerank → Generate
```

But here’s the harsh truth:

> Most RAG failures are NOT because of bad embeddings.
> They happen because the query itself is bad.

Day 13 is about fixing that.

---

# 🧠 Why Query Transformation Matters

Real user queries are:

* Ambiguous
* Underspecified
* Conversational
* Multi-hop
* Noisy

Example:

> “Why did revenue drop after the compliance update?”

This actually requires:

1. Retrieve info about revenue
2. Retrieve info about compliance update
3. Retrieve timeline correlation
4. Combine reasoning

A single vector search won’t cut it.

---

# 🔁 1️⃣ Query Rewriting

### Problem

User queries are often poorly phrased for retrieval.

Example:

> “What did they change in the security thing last quarter?”

Embedding this directly → poor recall.

---

### Solution: LLM-based Rewrite

Rewrite query into a retrieval-optimized form.

**Original:**

> What did they change in the security thing last quarter?

**Rewritten:**

> What security policy updates were implemented in Q4 2025?

Much better semantic alignment.

---

### Architecture

```
User Query
    ↓
LLM Rewriter
    ↓
Optimized Query
    ↓
Retriever
```

---

### Code Example

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI()

rewrite_prompt = PromptTemplate.from_template("""
Rewrite the query for optimal document retrieval.
Query: {query}
Optimized:
""")

def rewrite_query(query):
    return llm.invoke(rewrite_prompt.format(query=query)).content
```

---

# 🔍 2️⃣ Multi-Query Retrieval (Improves Recall)

Instead of 1 embedding → generate 3–5 variations.

Example:

User:

> “How does rate limiting affect API latency?”

Generate:

1. Impact of rate limiting on response time
2. API throttling and latency relationship
3. Performance implications of request limiting

Now retrieve for all → merge results.

---

### Why This Works

Vector search recall improves dramatically because:

* Embedding space is imperfect
* Different phrasing lands in different regions

This is especially powerful in technical corpora.

---

### Implementation Concept

```
LLM → Generate N queries
For each:
    retrieve top-k
Merge & deduplicate
```

---

# 🧩 3️⃣ Multi-Hop Retrieval

Now we go deeper.

Some questions require sequential retrieval.

Example:

> Which papers cited the work that introduced Transformers?

This requires:

1. Retrieve paper introducing Transformers
   → Attention Is All You Need
2. Extract citation info
3. Retrieve citing papers

That’s multi-hop reasoning.

---

### Strategy: Iterative RAG

```
Query → Retrieve
     ↓
Extract intermediate entity
     ↓
New query
     ↓
Retrieve again
     ↓
Combine evidence
```

---

### Example (Enterprise Setting)

User:

> Which customers complained after the pricing policy change?

Hop 1:
Retrieve pricing policy change date.

Hop 2:
Retrieve complaints after that date.

Hop 3:
Join results.

This is basically building a retrieval agent.

---

# 🧠 4️⃣ Self-Ask Pattern

LLM decomposes complex question into subquestions.

Example:

> Did revenue increase after we launched feature X in Europe?

Self-Ask:

1. When was feature X launched in Europe?
2. What was revenue before?
3. What was revenue after?
4. Compare.

This is structured reasoning before retrieval.

---

# 📊 5️⃣ Tradeoffs

| Method      | Pros                      | Cons                |
| ----------- | ------------------------- | ------------------- |
| Rewrite     | High precision            | Slight latency      |
| Multi-query | High recall               | More vector cost    |
| Multi-hop   | Handles complex reasoning | Pipeline complexity |
| Self-ask    | Transparent reasoning     | LLM tokens ↑        |

In production, you combine them.

---

# 🏗 Real-World Systems Using This

* Perplexity AI → heavy query rewriting
* Google search → multi-stage retrieval
* OpenAI → query planning in deep research mode

Modern RAG ≠ single vector search.

---

# 🧪 Practical Exercise (Critical)

Build this today:

1. Add query rewriting
2. Add multi-query generation (3 variations)
3. Merge retrieval results
4. Compare against baseline

Measure:

* Recall@k
* Answer faithfulness
* Latency impact

---

# 🧠 Deep Insight

Think of RAG as:

> Retrieval = Database
> LLM = Query Planner

The more complex the question → the more it should behave like SQL planning.

---

# 🎯 Tomorrow (Day 14)

We go into:

> 🧬 Hybrid Retrieval (BM25 + Dense + Sparse + Metadata Filters)

This is where most production RAG systems start looking serious.

---

If you want, tomorrow we can:

* Implement full multi-hop agent
* Or run a mini research-style RAG experiment

Your call.
