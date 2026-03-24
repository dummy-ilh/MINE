## Day 17 – Multi-Hop RAG


Multi-hop RAG is required when a question **cannot be answered from a single document chunk**. The model must retrieve → reason → retrieve again → aggregate → answer.

This is fundamentally different from single-hop RAG.

---

# 1️⃣ What is Multi-Hop Reasoning?

A **multi-hop question** requires chaining multiple facts.

Example:

> "Which university did the CEO of the company that acquired Instagram attend?"

To answer:

1. Who acquired Instagram? → Facebook
2. Who was CEO of Facebook? → Mark Zuckerberg
3. Where did Mark Zuckerberg study? → Harvard

That’s **3 hops**.

---

# Why Vanilla RAG Fails

Standard RAG:

```
Query → Embed → Retrieve top-k → LLM answers
```

Failure modes:

* Retrieves partial context
* Misses bridging entity
* Hallucinates intermediate step
* Context window gets cluttered

Multi-hop RAG fixes this via structured retrieval strategies.

---

# 2️⃣ Question Decomposition

## Core Idea

Break a complex question into smaller atomic sub-questions.

### Pipeline

```
User Question
      ↓
LLM Decomposition
      ↓
Sub-question 1 → Retrieve → Answer
      ↓
Sub-question 2 (conditioned on 1)
      ↓
Aggregate
```

---

### Example

Original:

> "What awards has the director of Inception won?"

Step 1:
Who directed Inception?

Step 2:
What awards has Christopher Nolan won?

---

### Implementation Sketch

```python
def decompose(question):
    return llm("Break this into atomic factual sub-questions: " + question)

def retrieve_answer(q):
    docs = retriever.search(q)
    return llm(context=docs, question=q)

def multi_hop(question):
    subqs = decompose(question)
    answers = []
    for q in subqs:
        ans = retrieve_answer(q)
        answers.append(ans)
    return aggregate(answers)
```

---

### Design Considerations

| Issue             | Solution              |
| ----------------- | --------------------- |
| Bad decomposition | Fine-tune decomposer  |
| Propagation error | Use verification step |
| Long chains       | Limit hop count       |

---

# 3️⃣ Iterative Retrieval (Self-Ask Style)

Instead of decomposing upfront, the model retrieves → reasons → decides next query.

This mimics how humans search.

### Architecture

```
Query
  ↓
Retrieve
  ↓
LLM reasoning
  ↓
Need more info?
  ↓ Yes → New query
  ↓ No → Final answer
```

This is dynamic multi-hop.

---

### Pseudo-Code

```python
state = question
for step in range(max_hops):
    docs = retriever.search(state)
    reasoning = llm(context=docs, question=question)
    
    if reasoning.contains("Need more info"):
        state = reasoning.next_query
    else:
        return reasoning.answer
```

---

### Advantages

* Adaptive
* Handles unknown hop counts
* More human-like

### Risks

* Query drift
* Infinite loops
* Cost explosion

Mitigation:

* Hop budget
* Confidence thresholds
* Query similarity checks

---

# 4️⃣ Graph-Style RAG

Now we enter structured retrieval.

Instead of vector chunks only, we use a **knowledge graph**.

Nodes = Entities
Edges = Relationships

This is ideal for multi-hop reasoning.

---

## Example Graph

Let’s use the film **Inception** directed by **Christopher Nolan**.

![Image](https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw%40%40._V1_FMjpg_UX1000_.jpg)

![Image](https://collectionimages.npg.org.uk/large/mw201514/Christopher-Nolan.jpg)

![Image](https://solutionsreview.com/data-management/files/2022/08/G5.jpg)

![Image](https://www.researchgate.net/publication/352188493/figure/fig1/AS%3A1032114115969024%401623086589280/Knowledge-graph-example-Nodes-represent-entities-edge-labels-represent-types-of.png)

Graph edges:

```
(Inception) --directed_by--> (Christopher Nolan)
(Christopher Nolan) --won--> (Academy Award)
```

Multi-hop becomes graph traversal.

---

### Graph Query Strategy

```
Start Node: Inception
Edge: directed_by
Node: Christopher Nolan
Edge: won
Node: Awards
```

This is much more reliable than semantic guessing.

---

### Graph RAG Architecture

```
User Query
    ↓
Entity Linking
    ↓
Graph Traversal (k hops)
    ↓
Fetch supporting documents
    ↓
LLM synthesis
```

---

### Why Graph RAG Is Powerful

| Vector RAG             | Graph RAG               |
| ---------------------- | ----------------------- |
| Semantic similarity    | Structural reasoning    |
| May miss bridge entity | Explicit relation edges |
| No path awareness      | Path-traceable answers  |

---

# 5️⃣ Comparing the Three Approaches

| Method                 | Best For             | Weakness                    |
| ---------------------- | -------------------- | --------------------------- |
| Question Decomposition | Clear logical chains | Depends on good splitting   |
| Iterative Retrieval    | Unknown hop count    | Query drift                 |
| Graph RAG              | Entity-heavy domains | Requires graph construction |

---

# 6️⃣ Real-World Applications

Multi-hop RAG is critical in:

* Legal research (case precedents → laws → amendments)
* Biomedical research (drug → gene → pathway → disease)
* Financial analysis (company → subsidiary → executive → regulatory issue)
* Academic QA systems
* Enterprise knowledge assistants

---

# 7️⃣ Advanced Techniques

## A. Hybrid Graph + Vector RAG

1. Use vector search for candidate entities
2. Use graph traversal for reasoning
3. Fetch supporting text
4. LLM synthesizes

This balances recall + precision.

---

## B. Reinforcement Learning for Hop Control

Train agent to:

* Minimize hops
* Maximize factual grounding
* Penalize hallucination

---

## C. Verification Loop

After final answer:

```
Answer → Retrieve evidence → Cross-check
```

This reduces hallucination in long chains.

---

# 8️⃣ Evaluation Metrics for Multi-Hop RAG

Single-hop metrics are insufficient.

You must evaluate:

* **Answer EM / F1**
* **Supporting fact accuracy**
* **Path correctness**
* **Hop efficiency**
* **Retrieval recall per hop**

Datasets:

* HotpotQA
* MuSiQue
* 2WikiMultiHopQA

---

# 9️⃣ Common Pitfalls

1. Hop explosion (too many queries)
2. Error compounding
3. Retrieval noise amplification
4. Graph sparsity
5. Entity linking mistakes

---

# 10️⃣ Mental Model

Think of multi-hop RAG as:

> RAG + Planning + Memory + Control Flow

It’s closer to an agent than a simple retriever.

---

Excellent. Let’s push this to **production-grade depth** and then sharpen it with **serious interview questions** (the kind that distinguish surface knowledge from architectural mastery).

---

# 🔷 Production Multi-Hop RAG Architecture

A robust system combines:

* Planner (decomposition / reasoning control)
* Retriever (vector + keyword)
* Optional Knowledge Graph
* Memory (state across hops)
* Verifier
* Answer synthesizer

---

## 1️⃣ High-Level Architecture

```
User Question
      ↓
Query Planner (LLM)
      ↓
Hop Controller
      ↓
Retriever (Vector / Hybrid / Graph)
      ↓
Evidence Store (Memory Buffer)
      ↓
LLM Reasoning
      ↓
Verifier
      ↓
Final Answer
```

This is now closer to an **agent system** than plain RAG.

---

# 🔷 Example Walkthrough (Concrete Multi-Hop)

Question:

> "Which awards has the director of the film Interstellar won?"

Let’s anchor this to real entities:

* Film: **Interstellar**
* Director: **Christopher Nolan**

![Image](https://m.media-amazon.com/images/I/81v1NMJBrlL._AC_UF894%2C1000_QL80_.jpg)

![Image](https://collectionimages.npg.org.uk/large/mw201514/Christopher-Nolan.jpg)

![Image](https://www.researchgate.net/publication/330027047/figure/fig2/AS%3A728302734045187%401550652311143/A-toy-example-of-movieentitys-knowledge-graph-The-latent-relationship-similarity.png)

![Image](https://www.researchgate.net/publication/371009440/figure/fig1/AS%3A11431281161270366%401684984489787/An-example-of-a-movie-knowledge-graph-with-multiple-types-of-entities-In-our-work-text.png)

### Step-by-step reasoning:

**Hop 1**
Query: Who directed Interstellar?
Retrieve: Christopher Nolan

**Hop 2**
Query: What awards has Christopher Nolan won?
Retrieve: Academy Award nominations, BAFTA awards, etc.

**Aggregation**
LLM synthesizes final answer.

---

# 🔷 Minimal Multi-Hop Agent (Clean Architecture)

Here’s a simplified but production-oriented structure.

```python
class MultiHopAgent:

    def __init__(self, retriever, llm, max_hops=3):
        self.retriever = retriever
        self.llm = llm
        self.max_hops = max_hops
        self.memory = []

    def plan(self, question):
        prompt = f"Decompose into stepwise sub-questions:\n{question}"
        return self.llm(prompt)

    def retrieve(self, query):
        docs = self.retriever.search(query, top_k=5)
        return docs

    def reason(self, question, context):
        prompt = f"Answer using evidence:\nQuestion:{question}\nContext:{context}"
        return self.llm(prompt)

    def verify(self, answer):
        prompt = f"Verify factual consistency:\n{answer}"
        return self.llm(prompt)

    def run(self, question):
        subqs = self.plan(question)
        
        for hop, q in enumerate(subqs[:self.max_hops]):
            docs = self.retrieve(q)
            self.memory.append((q, docs))

        aggregated_context = "\n".join(str(m) for m in self.memory)
        answer = self.reason(question, aggregated_context)

        return self.verify(answer)
```

This is simplistic — but architecturally correct.

---

# 🔷 Advanced Design Considerations (Senior-Level Thinking)

## 1. Hop Budgeting

Let:

* C_r = retrieval cost
* C_l = LLM cost per hop
* H = number of hops

Total cost ≈ H(C_r + C_l)

Optimization objective:

```
Minimize H subject to Answer Accuracy ≥ threshold
```

This becomes a reinforcement learning or bandit problem.

---

## 2. Query Drift Detection

At each hop:

```
sim(original_question, new_query)
```

If similarity drops too much → likely drift.

Mitigation:

* Constrain query generation
* Use entity grounding
* Enforce structured reasoning

---

## 3. Hybrid Retrieval Strategy

Best practice in real systems:

```
Step 1: BM25 (keyword precision)
Step 2: Dense retrieval (semantic recall)
Step 3: Reranker (cross-encoder)
Step 4: Graph expansion (if entity detected)
```

This reduces compounding noise.

---

# 🔷 Graph + Vector Hybrid (Enterprise Pattern)

Example in biomedical domain:

Drug → Gene → Pathway → Disease

Vector search finds candidate drug info.
Graph traversal finds gene interaction edges.
LLM synthesizes.

This architecture is used in:

* Legal research systems
* Pharma intelligence
* Financial risk analysis
* Enterprise knowledge assistants

---

# 🔷 Verification Loop (Critical for Interviews)

After final answer:

```
Answer → Generate supporting statements → Retrieve again → Cross-check
```

This is called:

* Retrieval-augmented verification
* Self-consistency checking

Reduces hallucination significantly in multi-hop settings.

---

# 🔷 Evaluation (Interview-Level Detail)

Standard RAG metrics are insufficient.

You need:

### 1. Answer Metrics

* Exact Match (EM)
* F1 Score

### 2. Retrieval Metrics

* Recall@k per hop
* MRR

### 3. Reasoning Metrics

* Supporting fact accuracy
* Path correctness

Datasets:

* HotpotQA
* MuSiQue
* 2WikiMultiHopQA

---

# 🔷 Failure Modes (Very Important for Interviews)

1. Error propagation across hops
2. Retrieval noise amplification
3. Entity linking failure
4. Over-decomposition
5. Hop explosion
6. Context window saturation

A strong candidate talks about mitigation strategies.

---

# 🔥 Now — Interview Questions (Tiered)

---

## 🟢 Beginner-Level

1. What is multi-hop RAG and how does it differ from standard RAG?
2. Why does single-shot retrieval fail for multi-hop questions?
3. What is question decomposition?

---

## 🟡 Intermediate-Level

4. How would you design an iterative retrieval loop?
5. What are the trade-offs between decomposition vs iterative retrieval?
6. How do you prevent query drift?
7. How do you evaluate multi-hop RAG systems?
8. When would you choose Graph RAG over vector RAG?

---

## 🔴 Senior-Level / FAANG-Level

9. Suppose hop count is unknown. How would you dynamically control retrieval depth?
10. How would you reduce hallucination in multi-hop pipelines?
11. Design a cost-aware multi-hop RAG for production.
12. How would you debug performance degradation in a 3-hop RAG?
13. How would you train a model to decide when to stop retrieving?
14. Explain how reinforcement learning could optimize hop strategy.
15. How would you design multi-hop RAG for legal or biomedical domain?

---

## 🧠 Deep Theoretical Question

16. Can multi-hop reasoning be reduced to graph traversal?

* When yes?
* When no?
* What are the limitations?

---

# 🔥 Hard Whiteboard Question

Design a multi-hop RAG system that:

* Works with 10M+ documents
* Has latency < 2 seconds
* Maintains 90% answer accuracy
* Minimizes cost

Explain:

* Indexing strategy
* Retrieval pipeline
* Caching
* Failure recovery
* Monitoring metrics

If someone can answer this cleanly, they understand the field deeply.

---

Perfect. Now we move from **knowing** to **defending in an interview**.

I’ll answer the interview questions at a senior level — structured, precise, and technically deep.

---

# 🟢 Beginner-Level Answers

---

### 1️⃣ What is Multi-Hop RAG and how does it differ from standard RAG?

**Standard RAG**

```
Query → Retrieve once → LLM answers
```

Assumption: All required evidence exists in a single retrieval batch.

**Multi-Hop RAG**

```
Query → Retrieve → Reason → Retrieve again → Aggregate → Answer
```

Used when:

* Evidence is distributed
* There are bridging entities
* Logical chaining is required

Key difference:

* Single-hop retrieves evidence
* Multi-hop retrieves evidence *and discovers new queries dynamically*

---

### 2️⃣ Why does single-shot retrieval fail?

Because:

* Embedding similarity cannot infer intermediate reasoning.
* Bridging entities may not appear in the original question.
* Retrieval noise compounds.
* Context window gets cluttered.

Example:
“Where did the CEO of the company that acquired Instagram study?”

The phrase *“Mark Zuckerberg”* is not in the question — so single retrieval may fail.

---

### 3️⃣ What is question decomposition?

Breaking a complex query into atomic sub-questions:

Example:

```
Q: Awards won by director of Interstellar?
→ Who directed Interstellar?
→ What awards has that person won?
```

It introduces **planning before retrieval**.

---

# 🟡 Intermediate-Level Answers

---

### 4️⃣ How would you design an iterative retrieval loop?

Core loop:

```
state = question
for hop in max_hops:
    retrieve(state)
    reason()
    if sufficient info:
        stop
    else:
        generate next query
```

Key components:

* Hop controller
* Query generator
* Stop criterion (confidence score or verifier)

Add safeguards:

* Hop limit
* Similarity constraint to original question
* Retrieval confidence threshold

---

### 5️⃣ Trade-offs: Decomposition vs Iterative Retrieval

| Decomposition             | Iterative          |
| ------------------------- | ------------------ |
| Pre-planned               | Adaptive           |
| Deterministic             | Dynamic            |
| Easier to debug           | More flexible      |
| Sensitive to bad planning | Sensitive to drift |

Decomposition works well when logical chain is clear.
Iterative works when hop count is unknown.

---

### 6️⃣ How do you prevent query drift?

Drift = model starts retrieving irrelevant branches.

Mitigation:

1. Similarity constraint:

   ```
   sim(new_query, original_query) > threshold
   ```

2. Entity grounding:
   Force queries to include entities found earlier.

3. Structured generation:
   Use JSON schema for next query.

4. Penalize entropy in query expansion.

---

### 7️⃣ How do you evaluate multi-hop RAG?

You must evaluate:

1. **Answer accuracy**

   * EM
   * F1

2. **Retrieval quality**

   * Recall@k per hop
   * MRR

3. **Reasoning correctness**

   * Supporting fact accuracy
   * Path accuracy

Datasets:

* HotpotQA
* MuSiQue
* 2WikiMultiHopQA

A senior answer mentions **supporting fact supervision**, not just final EM.

---

### 8️⃣ When choose Graph RAG over Vector RAG?

Choose Graph RAG when:

* Data is highly relational
* Queries involve explicit relationships
* Entities matter more than semantic similarity

Example domains:

* Biomedical networks
* Legal citation graphs
* Financial ownership structures

Vector RAG is better when:

* Documents are long-form narrative
* Relations are implicit

Hybrid is best in production.

---

# 🔴 Senior-Level Answers

---

### 9️⃣ Unknown hop count — dynamic depth control?

Use:

1. Confidence-based stopping
2. Entropy threshold
3. Verifier score
4. Reinforcement learning policy

Formal framing:

Let:

* R(h) = reward at hop h
* C(h) = cost

Optimize:

```
max_h R(h) - λC(h)
```

This becomes a sequential decision problem.

---

### 🔟 Reduce hallucination in multi-hop?

Techniques:

1. Retrieval-constrained generation
2. Evidence citation enforcement
3. Verification loop
4. Self-consistency sampling
5. Structured reasoning prompts
6. Graph grounding

Critical insight:
Hallucination probability increases with hop count due to error propagation.

---

### 1️⃣1️⃣ Cost-aware production multi-hop RAG

Components:

* Hybrid index (BM25 + dense)
* Reranker
* Caching intermediate results
* Hop budget (2–3 max)
* Confidence-based early stop
* Async retrieval

Cost formula:

```
Total cost ≈ H × (LLM + Retrieval + Rerank)
```

Reduce H aggressively.

---

### 1️⃣2️⃣ Debug performance degradation?

Break pipeline into stages:

1. Retrieval recall dropping?
2. Reranker misranking?
3. Planner generating bad sub-questions?
4. Context overflow?
5. Entity linker failing?

Use ablation testing:

* Replace planner with oracle
* Replace retriever with ground-truth
* Measure stage-wise degradation

---

### 1️⃣3️⃣ Train model to decide when to stop?

Three methods:

1. Supervised learning on optimal hop count
2. RL with cost penalty
3. Confidence calibration head

Stop when:

```
P(answer_correct | evidence) > threshold
```

---

### 1️⃣4️⃣ RL for hop optimization?

State:

* Current evidence
* Retrieval score
* Hop number

Action:

* Retrieve again
* Stop

Reward:

* +1 correct answer
* -λ per hop

This is a Markov Decision Process.

---

### 1️⃣5️⃣ Legal/Biomedical multi-hop design?

Must use:

* Knowledge graph backbone
* Entity normalization
* Citation grounding
* Strict verification

Example:
Drug → Gene → Pathway → Disease

Graph traversal + text evidence retrieval.

---

# 🧠 Deep Theory: Is Multi-Hop = Graph Traversal?

Sometimes yes.

If:

* Data is structured
* Relations explicit

But not always.

Natural language:

* Implicit relations
* Temporal logic
* Causal reasoning
* Multi-document aggregation

Graph traversal alone cannot handle fuzzy reasoning.

Thus:
Multi-hop RAG = Graph traversal + probabilistic reasoning + semantic retrieval.

---

# 🔥 Hard Whiteboard System Design (2-second latency, 10M docs)

Architecture:

1. Sharded vector index (FAISS/HNSW)
2. BM25 inverted index
3. Hybrid retrieval
4. Cross-encoder reranker (lightweight)
5. Hop cap = 2
6. Cached embeddings
7. Parallel retrieval calls
8. Verifier only if low confidence

Monitoring:

* Latency per stage
* Recall@k
* Hop distribution
* Hallucination rate
* Cost per query

Use async + batching aggressively.

---

# 🏁 Final Insight

Multi-hop RAG is no longer just “retrieval + LLM.”

It is:

> Planning + Memory + Control Flow + Retrieval + Verification

At this point, you’re designing agents — not pipelines.

---


