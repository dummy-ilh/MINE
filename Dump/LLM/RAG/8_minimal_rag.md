

> **Day 8 — Build a Minimal RAG From Scratch (No Framework Magic)**

No LangChain abstractions.
No “vector store wrappers.”
Just the core pipeline so you actually understand what’s happening.

---

# 📘 Day 8 — Minimal RAG (Raw, Clean Architecture)

---

# 1️⃣ The Full Pipeline (What We’re Building)

```
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Index (FAISS)
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Prompt Construction
   ↓
LLM Answer
```

That’s it.

If you master this, frameworks become optional tools — not crutches.

---

# 2️⃣ Step 1 — Install Dependencies

```bash
pip install openai faiss-cpu tiktoken numpy
```

(You can swap embedding model later.)

---

# 3️⃣ Step 2 — Prepare Documents

Let’s simulate:

```python
documents = [
    "Refunds requested after 30 days require manual approval.",
    "Customers in Germany are subject to additional VAT penalties.",
    "Refunds are processed within 5–7 business days."
]
```

In production, these come from:

* PDFs
* Markdown
* HTML
* Databases

We keep it simple today.

---

# 4️⃣ Step 3 — Chunking (Minimal Version)

For now, assume:

* Each document = one chunk

Later (Day 9+), we’ll build a real ingestion pipeline.

---

# 5️⃣ Step 4 — Create Embeddings

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([e.embedding for e in response.data])
```

Generate document embeddings:

```python
doc_embeddings = embed(documents)
```

Shape:

```
(num_docs, embedding_dim)
```

---

# 6️⃣ Step 5 — Build Vector Index (FAISS)

```python
import faiss

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(doc_embeddings)
```

Now FAISS holds your semantic space.

---

# 7️⃣ Step 6 — Query-Time Retrieval

```python
query = "What happens if refund is late in Germany?"

query_embedding = embed([query])

k = 2
distances, indices = index.search(query_embedding, k)

retrieved_docs = [documents[i] for i in indices[0]]
```

Now you have top-k relevant chunks.

---

# 8️⃣ Step 7 — Construct the Prompt

```python
context = "\n\n".join(retrieved_docs)

prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}
"""
```

---

# 9️⃣ Step 8 — Generate Answer

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print(response.choices[0].message.content)
```

Temperature = 0
→ deterministic
→ better for grounded answers

---

# 🔟 What You Just Built

You built:

* Embedding model
* Vector index
* Similarity search
* Prompt grounding
* Generation

This is 90% of every RAG system.

Frameworks just add:

* Orchestration
* Memory
* Observability
* Tools

---

# 🧠 Important Observations

### 1️⃣ Retrieval quality matters more than generation

Even GPT-4 cannot fix wrong retrieval.

### 2️⃣ You didn’t normalize embeddings

If using cosine similarity:

* Normalize vectors first

FAISS L2 ≠ cosine unless normalized.

### 3️⃣ No metadata yet

Production systems need:

* IDs
* Versioning
* Filtering

We’ll add that soon.

---

# 11️⃣ Where This Minimal System Breaks

* No chunking strategy
* No hybrid search
* No re-ranking
* No metadata filtering
* No evaluation metrics
* No scaling

But this skeleton is correct.

---

# 🎯 Interview Angle

If asked:

> “Can you build RAG without LangChain?”

You should confidently say:

> “Yes, it’s just embeddings, vector search, and prompt construction. Frameworks add orchestration, but the core logic is straightforward.”

That shows true understanding.

---
\
