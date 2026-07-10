
# 🧠 30-Day RAG Mastery Agenda

*(From fundamentals → production → research-level)*

---

## **WEEK 1 — Foundations (No Code Yet, Pure Understanding)**

> Goal: Build *mental models* so you don’t cargo-cult LangChain.

### **Day 1** – What RAG Really Is

* Parametric vs external memory
* Why LLMs hallucinate
* Why fine-tuning ≠ knowledge
* RAG vs prompt stuffing
  ✅ *Completed*

### **Day 2** – Embeddings Deep Dive

* What embeddings actually encode
* Cosine vs dot vs L2
* Dense vs sparse retrieval
* When BM25 beats embeddings
* Curse of dimensionality

### **Day 3** – Chunking Is the Hidden Boss

* Why chunk size matters
* Sliding windows vs semantic chunks
* Overlap tradeoffs
* Header-aware chunking
* PDF horror stories

### **Day 4** – Vector Databases Internals

* What a vector DB actually stores
* HNSW, IVF, PQ (intuition, not math spam)
* Recall vs latency tradeoffs
* Metadata filtering

### **Day 5** – Retrieval Strategies

* Top-k vs threshold
* Hybrid search (BM25 + vectors)
* Re-ranking (cross-encoders)
* Query expansion

### **Day 6** – Prompt Engineering for RAG

* Context injection patterns
* Citation-aware prompting
* Guardrails against hallucination
* System vs user prompts

### **Day 7** – Failure Modes & Debugging

* Empty retrieval
* Context overload
* Wrong chunk retrieved
* Over-trusting the LLM
  📌 *Week-1 checkpoint*

---

## **WEEK 2 — Hands-On RAG (Real Code, No Toy Examples)**

> Goal: Build a **working RAG pipeline from scratch**

### **Day 8** – Minimal RAG (From Scratch)

* Raw Python
* No LangChain
* Manual embeddings + FAISS

### **Day 9** – Document Ingestion Pipeline

* PDFs, markdown, HTML
* Cleaning, normalization
* Deduplication

### **Day 10** – Embedding Models Comparison

* OpenAI vs open-source
* SentenceTransformers
* Domain-specific embeddings

### **Day 11** – Vector Stores in Practice

* FAISS vs Pinecone vs Weaviate vs Chroma
* Cost, scaling, persistence

### **Day 12** – Retrieval Evaluation

* Precision@k
* Recall@k
* MRR
* Human eval pitfalls

### **Day 13** – RAG with LangChain / LlamaIndex

* What abstractions help
* What abstractions hide
* When to avoid them

### **Day 14** – End-to-End RAG App

* Query → Answer → Sources
* Logging + observability
  📌 *Week-2 checkpoint*

---

## **WEEK 3 — Advanced RAG (This Is Where People Drop Off)**

> Goal: Build **robust, scalable, intelligent** RAG

### **Day 15** – Query Understanding

* Query rewriting
* Multi-query retrieval
* Intent detection

### **Day 16** – Re-Ranking & Compression

* Cross-encoders
* Contextual compression
* Passage selection

### **Day 17** – Multi-Hop RAG

* Question decomposition
* Iterative retrieval
* Graph-style RAG

### **Day 18** – Structured + Unstructured RAG

* SQL + vector hybrid
* Knowledge graphs
* Tabular grounding

### **Day 19** – Long Context Models vs RAG

* When RAG still wins
* Context window economics
* Token budgeting

### **Day 20** – Security & Safety

* Prompt injection attacks
* Data leakage
* Access control per document

### **Day 21** – RAG at Scale

* Caching strategies
* Async retrieval
* Latency budgets
  📌 *Week-3 checkpoint*

---

## **WEEK 4 — Production, Research & Interview Mastery**

### **Day 22** – RAG Evaluation (Hard Problem)

* Faithfulness
* Groundedness
* Answer relevance
* RAGAS, G-Eval

### **Day 23** – Hallucination Reduction Techniques

* Answer verification
* Self-consistency
* Refusal policies

### **Day 24** – RAG vs Fine-Tuning vs Tool-Use

* Decision framework
* Hybrid systems

### **Day 25** – Domain-Specific RAG

* Legal
* Medical
* Finance
* Code RAG

### **Day 26** – Research Frontiers

* Agentic RAG
* Memory-augmented transformers
* RAPTOR, HyDE, FLARE

### **Day 27** – RAG System Design Interviews

* Whiteboard architecture
* Tradeoff questions
* Failure analysis

### **Day 28** – Build Your Own RAG Framework

* Custom retriever
* Custom ranker
* Observability hooks

### **Day 29** – Capstone Project

* Real dataset
* Measured improvements
* Written design doc

### **Day 30** – Mastery Check

* Explain RAG to:

  * CEO
  * ML engineer
  * Researcher
* Mock interview Q&A
* Final system review

---


