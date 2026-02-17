## Day 17 – Multi-Hop RAG

(*Master-level deep dive: architecture, reasoning patterns, and failure modes*)

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

If you'd like next, I can:

* Show a full production-ready architecture diagram
* Build a minimal multi-hop RAG agent in code
* Explain how this integrates with LLM agents
* Or dive into evaluation + benchmarking methodology

Your move.
