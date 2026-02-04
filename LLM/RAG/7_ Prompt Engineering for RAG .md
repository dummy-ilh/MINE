---

# 📘 RAG Daily Tutorial

## **Day 6 — Prompt Engineering for RAG (Grounding Without Hallucination)**

---

# 1️⃣ The Core Problem

You retrieved:

* The correct chunks
* High-quality sources
* Properly ranked context

But the LLM:

* Hallucinates
* Over-generalizes
* Ignores context
* Mixes prior knowledge with retrieved text

Why?

Because **the model is trained to be helpful, not faithful.**

Prompting determines whether it:

* Treats retrieved context as truth
* Or treats it as optional inspiration

---

# 2️⃣ The Fundamental Rule of RAG Prompting

You must explicitly instruct the model to:

1. Use only the provided context
2. Admit uncertainty
3. Cite sources
4. Avoid external knowledge

If you don’t say this clearly, it will improvise.

---

# 3️⃣ Basic RAG Prompt Template (Naïve)

```
Answer the question using the context below.

Context:
{retrieved_chunks}

Question:
{user_query}
```

This works — but it’s fragile.

---

# 4️⃣ Strong Grounded Prompt (Production-Level)

Better structure:

```
You are a system that answers questions strictly using the provided context.
If the answer is not contained in the context, say:
"I don't have enough information from the provided documents."

Use direct quotes where appropriate.
Cite the source section after each claim.

Context:
-----------------
{retrieved_chunks}
-----------------

Question:
{user_query}

Answer:
```

This reduces hallucination significantly.

---

# 5️⃣ Why Citation Prompts Work

When you ask the model to:

> “Cite the section for each claim”

You force:

* Context alignment
* Claim-by-claim grounding
* Reduced fabrication

It creates a **faithfulness constraint**.

---

# 6️⃣ Context Formatting Matters

Bad:

```
Chunk1 text Chunk2 text Chunk3 text
```

Better:

```
[Document: Refund_Policy.md | Section: Late Refunds]

Refunds requested after 30 days...

---

[Document: Germany_Addendum.md | Section: Penalties]

For Germany customers...
```

Why?

* Structure improves attention
* Attribution improves precision
* LLM reasons per document

---

# 7️⃣ Guarding Against Context Overload

LLMs suffer from:

* Recency bias
* Lost-in-the-middle problem

If you send:

* 10 long chunks
* Mixed relevance

The model may:

* Focus on the first
* Ignore the most relevant

Solutions:

* Re-rank aggressively
* Put most relevant chunk first
* Use summary compression

---

# 8️⃣ Anti-Hallucination Techniques

## 🔹 Explicit refusal clause

“If the context does not contain the answer, say so.”

## 🔹 Extract-then-answer

Step 1:

* Extract relevant passages

Step 2:

* Generate final answer only from extracted passages

This reduces drift.

## 🔹 Structured output

Force JSON:

```
{
  "answer": "...",
  "sources": ["doc1", "doc2"],
  "confidence": "high/low"
}
```

Structured prompts reduce creative deviation.

---

# 9️⃣ Common Prompting Mistakes

❌ Allowing “general knowledge”
❌ Mixing system instructions with context
❌ Overly verbose system prompts
❌ No refusal instruction
❌ No citation requirement

Prompt design = alignment engineering.

---

# 🔟 Interview-Level Answer

If asked:

> “How do you reduce hallucinations in RAG?”

Strong answer:

> “I enforce strict context grounding through explicit refusal instructions, citation requirements, structured output formats, and careful context ordering to mitigate recency and attention biases.”

That shows system-level thinking.

---

# 🧠 Mental Model

Retrieval gives facts.
Prompting enforces discipline.

Without disciplined prompting:

RAG = “LLM with vibes”.

---

