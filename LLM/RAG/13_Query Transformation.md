## 🔎 Query Transformation & Multi-Hop Retrieval

We move from **basic retrieval** → **intelligent retrieval**.

Up to now, your RAG pipeline probably looks like:

```
User Query → Embed → Retrieve → Rerank → Generate
```

But here’s the harsh truth:

> Most RAG failures are NOT because of bad embeddings.
> They happen because the query itself is bad.


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

When building retrieval systems (RAG, search engines, agents), the *surface form* of a query is often not sufficient to retrieve the right information. Query transformation reformulates the user input into something the system can actually reason over.

Let’s go deeper.

---

#  Multi-Hop Queries (Deep Explanation)

### 🔹 What is Multi-Hop?

A **multi-hop query** is a question that requires **chaining multiple pieces of information together** before producing the final answer.

Instead of:

> Single lookup → answer

You need:

> Retrieve A → Use A to retrieve B → Combine → Answer

This is essentially **compositional reasoning over multiple documents or facts**.

---

## 🔍 Example 1

> "Who is the spouse of the CEO of Tesla?"

Step-by-step reasoning:

1. Identify CEO of Tesla → Tesla
2. CEO = Elon Musk
3. Retrieve spouse of Elon Musk
4. Answer

That’s **two hops**:

* Hop 1: Tesla → CEO
* Hop 2: CEO → Spouse

---

## 🔍 Example 2 (Harder)

> "Which university did the author of The Hobbit attend?"

Steps:

1. Identify author of The Hobbit
2. Author = J. R. R. Tolkien
3. Retrieve Tolkien’s university
4. Answer = University of Oxford

Again: multi-hop reasoning.

---

## 🧠 Why Multi-Hop Is Hard

### 1. Retrieval Challenge

Embedding similarity may retrieve:

* Docs about Tesla
* Docs about Elon Musk
* Docs about spouses

But not necessarily in the right order.

### 2. Context Explosion

Each hop expands search space.

### 3. Query Decomposition Required

You often need to transform:

> “Which university did the author of The Hobbit attend?”

Into:

* Subquery 1: Who wrote The Hobbit?
* Subquery 2: Where did Tolkien study?

---

## 🏗 How Systems Handle Multi-Hop

### Approach 1: Query Decomposition

Break into smaller questions.

### Approach 2: Iterative Retrieval (Agent-style)

Retrieve → Update query → Retrieve again.

### Approach 3: Graph-Based Retrieval

Use knowledge graphs to traverse relations.

---

## 🎯 Real-World Applications

* Legal reasoning
* Financial due diligence
* Medical diagnosis chains
* Research assistants
* Complex analytics queries

---

# Ambiguous Queries (Brief)

### Definition:

Query has multiple interpretations.

Example:

> “Apple revenue”

Could mean:

* Apple Inc. revenue
* Apple (fruit) industry revenue

Query transformation may expand to:

> “Apple Inc. annual revenue 2025”

---

#  Underspecified Queries (Brief)

### Definition:

Missing necessary constraints.

Example:

> “Best laptop”

Missing:

* Budget?
* Gaming?
* Coding?
* Lightweight?

Transformation might add inferred context:

> “Best lightweight laptop under $1000 for programming”

---

#  Conversational Queries (Brief)

Context-dependent follow-ups.

Example:

User:

> Who is the CEO of Tesla?

System:

> Elon Musk.

User:

> Where did he study?

"He" must resolve to Elon Musk.

This requires **coreference resolution + context memory**.

---

#  Noisy Queries (Brief)

Contain:

* Typos
* Slang
* Speech-to-text errors
* Broken grammar

Example:

> “whats da ceo tesla study?”

Needs normalization before retrieval.

---

# 🔥 Big Picture

Multi-hop queries are fundamentally different because:

| Type           | Main Challenge                      |
| -------------- | ----------------------------------- |
| Ambiguous      | Disambiguation                      |
| Underspecified | Add constraints                     |
| Conversational | Context tracking                    |
| Noisy          | Cleaning                            |
| **Multi-hop**  | **Reasoning across multiple facts** |

Multi-hop is the most structurally complex because it requires **composition of knowledge**, not just better retrieval.

---





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
You want to **master Query Transformation** — so we’ll treat this like a serious systems-design + ML interview prep.

This is heavily relevant for:

* RAG systems
* Search infra
* LLM agents
* Retrieval optimization
* Applied ML roles

We’ll structure this as:

1. Core Concepts Refresher
2. Medium Interview Q&A (deep but manageable)
3. Hard Interview Q&A (research-level thinking)
4. System Design Challenges
5. Evaluation + Tradeoffs
6. Real-world pitfalls

---

# 🔹 Part 1 — Core Mental Model

### What is Query Transformation?

Query transformation is the process of converting a **raw user query** into one or more improved queries that maximize retrieval accuracy and reasoning correctness.

It may involve:

* Rewriting
* Expanding
* Decomposing
* Disambiguating
* Context injecting
* Multi-hop breaking
* Structured conversion (SQL/DSL)

---

# 🔶 MEDIUM INTERVIEW QUESTIONS

---

## 1️⃣ What is query rewriting and why is it needed in RAG?

### Answer Structure

**Problem:**
Raw queries are:

* Ambiguous
* Underspecified
* Conversational
* Poorly optimized for vector similarity

Example:

> “Where did he study?”

The retriever cannot resolve “he”.

**Solution:**
Rewrite into:

> “Where did Elon Musk study?”

This improves embedding alignment and retrieval relevance.

---

### Key Insight

Embedding models work better with:

* Fully qualified entities
* Clear intent
* Specific nouns

Query rewriting improves:

* Recall
* Precision
* Faithfulness

---

## 2️⃣ Explain Query Expansion vs Query Rewriting

### Query Expansion

Add semantically related terms.

Example:

> “Best budget laptop”

Expanded:

> “Best affordable low-cost laptop under $1000”

Improves recall.

---

### Query Rewriting

Reformulate for clarity.

Example:

> “Who is Tesla CEO?”

Rewritten:

> “Who is the CEO of Tesla?”

Improves precision.

---

### Interview Insight

Expansion = broader retrieval
Rewriting = clearer intent

Too much expansion → noise
Too much rewriting → risk of wrong assumption

---

## 3️⃣ How would you handle multi-hop queries?

Example:

> “Which university did the author of The Hobbit attend?”

Entities involved:

* The Hobbit
* J. R. R. Tolkien
* University of Oxford

### Proper Answer

Approaches:

1. Query decomposition

   * Subquery 1: Who wrote The Hobbit?
   * Subquery 2: Where did Tolkien study?

2. Iterative retrieval

   * Retrieve author
   * Inject result into next query

3. Graph traversal

   * Use structured knowledge graph

Explain tradeoffs:

* Decomposition increases latency
* Single-shot retrieval may fail
* Agents handle dynamic branching better

---

## 4️⃣ What are risks of aggressive query transformation?

Expected points:

* Hallucinated constraints
* Over-specification
* Query drift
* Latency increase
* Loss of original intent

Strong answer includes:

> Always preserve original query alongside transformed version.

---

## 5️⃣ How would you evaluate query transformation quality?

Key metrics:

### Offline

* Recall@k
* MRR
* nDCG
* Answer EM/F1

### Online

* CTR
* User satisfaction
* Follow-up correction rate

### Advanced

Measure delta:

* Retrieval before transform
* Retrieval after transform

---

# 🔷 HARD INTERVIEW QUESTIONS

Now we go deeper.

---

## 6️⃣ Design a Query Transformation Pipeline for a Production RAG System

### Expected structured answer:

Layered pipeline:

1. Query normalization

   * Lowercasing
   * Typo correction
   * Entity recognition

2. Intent classification

   * Informational
   * Comparison
   * Multi-hop
   * SQL-type

3. Conditional transformation:

   * If ambiguous → disambiguation rewrite
   * If multi-hop → decomposition
   * If underspecified → constraint injection

4. Generate multiple candidates

   * Top-k rewrites

5. Retrieve per candidate

6. Merge results

7. Rerank

Advanced insight:

> Use LLM for transformation but lightweight models for classification.

---

## 7️⃣ When does query transformation hurt performance?

Deep answer expected:

* High-quality short factual queries (“capital of France”)
* Structured domain queries
* If embedding model already robust
* If transformation introduces bias

Also:
Transformation can shift semantic vector location too far.

---

## 8️⃣ How would you train a model to perform query rewriting?

Expected layers:

### Supervised

Dataset:

* (Original query, improved query)

### Self-supervised

* Use retrieval success signal
* Reinforcement learning from retrieval quality

### Feedback-driven

* Optimize rewrite for downstream answer correctness

Advanced answer:

> Use retrieval-aware training objective.

---

## 9️⃣ Explain the connection between Query Transformation and Agents

Agents perform dynamic transformation.

Example:

User:

> “Compare Tesla revenue with its biggest competitor.”

Needs:

* Identify competitor of Tesla
* Retrieve competitor revenue
* Retrieve Tesla revenue
* Compare

Agent loop:

* Plan
* Decompose
* Retrieve
* Update query
* Repeat

Vanilla RAG fails here.

---

## 🔟 Hardest Conceptual Question

“Is query transformation always better than improving embeddings?”

Master-level answer:

No.

Because:

* Strong embeddings reduce need for rewriting.
* Hybrid search (BM25 + vector) reduces transformation need.
* Over-transformation increases system complexity.
* Sometimes better chunking improves retrieval more.

The correct strategy is:

> Improve retrieval robustness first, then use transformation selectively.

---

# 🔥 Advanced Mastery Concepts

---

## 1️⃣ Multi-Query Retrieval

Generate 5 rewrites:

* Different phrasings
* Different semantic focuses

Merge results.

Improves recall significantly.

---

## 2️⃣ Retrieval-Conditioned Rewriting

Rewrite based on:

* What failed to retrieve
* What top-k contained

Closed-loop system.

---

## 3️⃣ Structured Query Conversion

Natural language → SQL / DSL

Example:

> “Top 5 companies by revenue in 2024”

→ SQL

This is query transformation at its strongest form.

---

## 🎯 Common Interview Trap

Interviewer asks:

> Why not just increase top-k retrieval?

Strong answer:

* More documents increase noise.
* Reranker cost grows.
* LLM context window wasted.
* Latency increases.

Transformation is smarter than brute-force recall.

---

# 🚀 Real-World Production Challenges

1. Latency constraints
2. Caching transformed queries
3. Monitoring drift
4. Preventing over-aggressive rewriting
5. Multi-lingual transformation
6. Cost control (LLM rewrites expensive)

---

# 🧠 Final Mastery Summary

If you truly understand Query Transformation, you should be able to explain:

* When to use it
* When NOT to use it
* How to evaluate it
* How to integrate it in RAG
* How agents generalize it
* How it interacts with embeddings
* How it affects latency and cost

---
Query transformation is the "secret sauce" of advanced RAG (Retrieval-Augmented Generation) systems. It’s the process of re-writing or decomposing a user's input to ensure the retriever finds the best possible context.

Here is a curated list of medium and hard interview questions designed to test your architectural intuition and technical depth.

---

## Medium: The "How It Works" Level

### 1. What is "HyDE" (Hypothetical Document Embeddings), and why is it useful?

**The Answer:** HyDE uses an LLM to generate a "fake" or hypothetical answer to a user's query first. We then embed that *fake answer* and use it to search the vector database instead of the original question.

* **Why it works:** Dense retrieval relies on semantic similarity. A question and its answer are often semantically different, but two answers (even if one is hallucinated) are usually very similar in vector space.

### 2. Explain the difference between "Multi-Query" and "RAG-Fusion."

**The Answer:** * **Multi-Query:** The LLM generates 3–5 variations of the same question from different perspectives. We run retrieval for all of them and take the union of the results.

* **RAG-Fusion:** It goes a step further. After retrieving documents for multiple queries, it uses **Reciprocal Rank Fusion (RRF)** to re-rank the documents based on their position across all result sets, prioritizing docs that appear consistently high across different variations.

### 3. When would you use "Query Decomposition" over a simple rewrite?

**The Answer:** You use decomposition for **complex, multi-hop queries**.

* *Example:* "Who is older, the CEO of Apple or the CEO of Microsoft?"
* A simple rewrite fails because there is no single document with that answer. Decomposition breaks it into:
1. "Who is the CEO of Apple and what is their age?"
2. "Who is the CEO of Microsoft and what is their age?"
3. Compare results.



---

## Hard: The "Architect" Level

### 4. How do you handle "Keyword vs. Semantic" mismatch in transformations?

**The Answer:** Sometimes a user uses specific jargon or acronyms that embeddings might miss. To solve this, you implement a **Step-Back Prompting** transformation.

* The LLM generates a more "generic" or high-level version of the query (the "step-back").
* You then perform a hybrid search: search for the specific jargon (Keyword/BM25) and the high-level concept (Dense/Vector). This ensures you don't lose the forest for the trees.

### 5. In a conversation, how do you handle "Query Condensation"?

**The Answer:** This addresses **Anaphora Resolution** (when a user says "it" or "he" referring to a previous turn).

* **The Technical Fix:** You pass the entire chat history and the latest query to an LLM with the instruction: "Given the conversation below, re-write the last user input to be a standalone question."
* *Hard follow-up:* "How do you do this without doubling your latency?"
* *Pro-Tip:* Use a smaller, faster model (like a 7B parameter model or a specialized distil-model) strictly for the rewrite, or only trigger the rewrite if the current query contains pronouns.

### 6. What are the risks of over-transforming a query?

**The Answer:**

* **Drift:** The LLM might rewrite a specific technical question into something too broad, leading to "noisy" retrieval.
* **Latency:** Every transformation is an LLM call. In a production system, adding 500ms for a rewrite can break the UX.
* **Cost:** If you are using Multi-Query (5 variations), you are 5x-ing your embedding costs and retrieval time.

---

## Comparison Table: Transformation Strategies

| Strategy | Best For... | Main Downside |
| --- | --- | --- |
| **Rewrite-Retrieve-Read** | Vague or poorly phrased queries. | LLM might change the intent. |
| **HyDE** | Questions with zero-shot difficulty. | Can retrieve "confidently wrong" docs. |
| **Sub-Query Decomposition** | Comparing things or multi-part questions. | High latency (multiple retrievals). |
| **Step-Back** | Complex technical reasoning. | Requires very precise prompting. |

---

## Mastering the Topic

To truly master this, you need to understand the math behind **Reciprocal Rank Fusion (RRF)**, as it’s the standard way to merge transformed queries.

The formula for the score of a document  is:

Score(d)=∑  1/(​k+rank(d,q))   where q∈Queries​
(Where k is a constant, usually 60, used to mitigate the impact of low-ranked results.)

Good. Now we go into **real interview territory**.

I’ll structure this exactly like a FAANG whiteboard round:

1. 🔥 System Design (architecture depth)
2. 🔥 Coding-style implementation (algorithm + complexity)
3. 🔥 Research-level adaptive rewriting problem

This will push you to mastery level.

---

# 🔥 PART 1 — FAANG-STYLE SYSTEM DESIGN

## 🧠 Problem Statement

Design a **Query Transformation Layer** for a production RAG system used by millions of users.

Constraints:

* < 500ms p95 latency
* Supports conversational context
* Handles multi-hop queries
* Must reduce hallucinations
* Works across enterprise + web documents

---

## Step 1: Clarify Requirements

### Functional

* Rewrite ambiguous queries
* Handle coreference
* Detect multi-hop
* Generate sub-queries
* Improve retrieval recall
* Avoid query drift

### Non-Functional

* Low latency
* Scalable
* Observable
* Safe (no hallucinated assumptions)

---

## Step 2: High-Level Architecture

```
User Query
   ↓
Query Preprocessor
   ↓
Intent & Complexity Classifier
   ↓
Transformation Engine
   ↓
Multi-Query Generator (optional)
   ↓
Retriever (Hybrid)
   ↓
Reranker
   ↓
LLM Answer
```

---

## Step 3: Core Modules

---

### 1️⃣ Query Preprocessing

* Typo correction
* Normalization
* Entity recognition

Example:

“tesla ceo wife?”

Recognize:

* Tesla

---

### 2️⃣ Intent & Complexity Classifier

Classify into:

* Simple factual
* Ambiguous
* Conversational
* Multi-hop
* Analytical / comparison

Use:

* Lightweight transformer
* Few-shot LLM
* Rule-based heuristics

Key Insight:

> Do NOT rewrite simple queries unnecessarily.

---

### 3️⃣ Transformation Engine

#### Case A: Ambiguous

Add qualifiers.

#### Case B: Conversational

Resolve coreference.

Example:

User: Who is CEO of Tesla?
User: Where did he study?

Rewrite:

> Where did Elon Musk study?

---

#### Case C: Multi-Hop

Example:

“Which university did the author of The Hobbit attend?”

Entities:

* The Hobbit
* J. R. R. Tolkien

Decompose:

1. Who wrote The Hobbit?
2. Where did Tolkien study?

---

### 4️⃣ Multi-Query Generator

Generate top-k rewrites:

* Semantic variant
* Keyword-heavy
* Short factual

Retrieve for all → Merge → Rerank.

Improves recall.

---

## Step 4: Latency Strategy

* Cache rewritten queries
* Use lightweight classifier
* Only use LLM rewrite when confidence < threshold
* Parallel retrieval for multi-query

---

## Step 5: Evaluation

Offline:

* Recall@k
* nDCG
* EM/F1

Online:

* CTR
* User correction rate
* Follow-up clarification rate

---

## Step 6: Failure Modes

* Over-specification
* Wrong entity resolution
* Query drift
* Latency blowup

---

### 🔥 Interview-Level Insight

The strongest answer:

> Query transformation should be conditional, retrieval-aware, and reversible.

Always preserve original query for fallback.

---

# 🔥 PART 2 — CODING-STYLE IMPLEMENTATION

## 🧠 Problem

Implement a simplified query transformation engine that:

1. Detects multi-hop patterns
2. Decomposes into subqueries
3. Returns execution plan

---

### Input

```
"Which university did the author of The Hobbit attend?"
```

### Output

```
[
  "Who is the author of The Hobbit?",
  "Which university did J. R. R. Tolkien attend?"
]
```

---

## Step 1 — Multi-Hop Detection Heuristic

Patterns:

* "author of"
* "CEO of"
* "spouse of"
* "capital of"
* Nested noun phrases

---

## Step 2 — Pseudo-Code

```python
def detect_multihop(query):
    triggers = ["author of", "CEO of", "spouse of", "founder of"]
    for t in triggers:
        if t in query:
            return True
    return False
```

Time complexity: O(n)

---

## Step 3 — Decomposition Strategy

1. Extract relation phrase
2. Form intermediate question
3. Replace phrase with entity placeholder

---

## Full Implementation Sketch

```python
def decompose_query(query):
    if "author of" in query:
        book = extract_entity_after(query, "author of")
        
        subq1 = f"Who is the author of {book}?"
        
        author = lookup_entity(book)  # retrieval call
        
        subq2 = query.replace(
            f"author of {book}",
            author
        )
        
        return [subq1, subq2]
    
    return [query]
```

---

### Complexity

Detection: O(n)
Rewrite: O(n)
Retrieval calls: k hops

Overall complexity:
O(k * retrieval_cost)

---

### Advanced Version

Replace `lookup_entity()` with:

* Vector retrieval
* Knowledge graph traversal
* Agent call

---

# 🔥 PART 3 — RESEARCH-LEVEL OPEN QUESTION

## 🧠 Adaptive Query Rewriting

Problem:

Not all queries need rewriting.

So:

> How do we dynamically decide when to rewrite?

---

## Research Question

Can we build a system that:

* Observes retrieval quality
* Learns from failures
* Rewrites only when necessary

---

## Proposed Architecture

1. Retrieve with original query
2. Evaluate retrieval quality score

   * Embedding similarity distribution
   * Reranker confidence
   * Entropy of top-k
3. If confidence low → trigger rewrite

---

## Advanced Idea

Train a **Rewrite Policy Network**

State:

* Query
* Retrieval scores
* Conversation history

Action:

* Rewrite
* Expand
* Decompose
* Do nothing

Reward:

* Answer accuracy
* Latency penalty

This becomes a reinforcement learning problem.

---

## 🔥 Hardest Open Problem

How to prevent **query drift** in adaptive rewriting?

Because:
Small rewrite errors compound in multi-hop loops.

Potential solutions:

* KL-divergence constraint between original and rewritten embedding
* Semantic similarity threshold
* Constrained decoding

---

# 🚀 Mastery-Level Insight

The future of query transformation is:

* Retrieval-aware
* Confidence-triggered
* Multi-query parallelized
* Policy-learned
* Cost-aware

And eventually merged into:

> Agentic Retrieval Systems

---

# 📘 Day 15 — Query Understanding

**Theme:** Make retrieval smarter than the user.

Most RAG systems fail because they treat the user query as perfect.

It’s not.

Users are vague, underspecified, ambiguous, multi-intent, or incorrectly phrased.

If retrieval depends on the raw query, recall suffers.

Today we fix that.

---

# 🧠 Why Query Understanding Matters

Retrieval systems depend on **semantic similarity**.

But embeddings only capture what is *expressed*, not what is *implied*.

Example:

> “How does it scale?”

Scale what? Database? Model? Architecture?

If your retriever uses this directly, it fails.

So we introduce:

1. Query rewriting
2. Multi-query retrieval
3. Intent detection

These are pre-retrieval intelligence layers.

---

# 🧩 1️⃣ Query Rewriting

## Problem

User query is:

* Too short
* Too vague
* Missing keywords
* Ambiguous

Example:

> “What are the issues?”

Rewrite into:

> “What are the common failure modes in production retrieval augmented generation systems?”

Now retrieval works better.

---

## How to Implement (Minimal Version)

```python
def rewrite_query(client, query):
    prompt = f"""
Rewrite the query to make it more specific and retrieval-friendly.

Original Query:
{query}

Rewritten Query:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
```

---

## When It Helps

* Enterprise document search
* Technical corpora
* Multi-domain knowledge bases
* Sparse user queries

---

## Hidden Cost

⚠️ Adds 1 LLM call → latency + cost.

This is why observability from Day 14 matters.

---

# 🧠 2️⃣ Multi-Query Retrieval

Single query → single embedding → single semantic projection.

But meaning is multi-dimensional.

So we generate multiple query variants.

---

## Example

Original:

> “How does RAG handle scaling?”

Variants:

* “Scaling vector databases in RAG systems”
* “Performance bottlenecks in large-scale retrieval”
* “Latency issues in production RAG pipelines”

Each hits different embedding neighborhoods.

---

## Minimal Implementation

```python
def generate_query_variants(client, query, n=3):
    prompt = f"""
Generate {n} alternative versions of this query for better document retrieval.

Query:
{query}

Return them as separate lines.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    variants = response.choices[0].message.content.split("\n")
    return [v.strip() for v in variants if v.strip()]
```

Then:

```python
all_results = []
for q in variants:
    results, _ = retriever.retrieve(q)
    all_results.extend(results)
```

Then deduplicate + rerank.

---

## Why This Works

Embedding space is nonlinear.

Different phrasing → different vector direction.

You increase recall without increasing k blindly.

---

## When To Use

* Complex domains
* Legal / medical corpora
* When recall matters more than latency

---

# 🧠 3️⃣ Intent Detection

Not all queries are retrieval queries.

Some are:

* Conversational
* Meta questions
* Follow-ups
* Clarifications
* Summaries

If you retrieve for everything, you waste latency.

---

## Simple Intent Classifier

```python
def detect_intent(client, query):
    prompt = f"""
Classify the intent of this query into one of:
- retrieval
- clarification
- conversational
- summarization

Query:
{query}

Return only the label.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip().lower()
```

---

## Example Behavior

If intent = conversational → skip retrieval.

If intent = summarization → retrieve larger context.

If intent = clarification → maybe ask user question.

Now your pipeline is conditional.

That’s architectural thinking.

---

# 🔄 Updated Pipeline (Day 15 Version)

```
User Query
   ↓
Intent Detection
   ↓
Query Rewriting (if needed)
   ↓
Multi-Query Generation
   ↓
Retrieve for each variant
   ↓
Merge + Deduplicate
   ↓
Rerank (Day 16)
   ↓
Context Build
   ↓
LLM Answer
```

Retrieval is no longer passive.

It is strategic.

---

# ⚠️ Tradeoffs

| Technique        | Recall ↑ | Latency ↑    | Cost ↑ | Complexity ↑ |
| ---------------- | -------- | ------------ | ------ | ------------ |
| Rewrite          | Medium   | +1 call      | +      | Low          |
| Multi-query      | High     | +n retrieval | ++     | Medium       |
| Intent detection | Medium   | +1 call      | +      | Low          |

Architects measure tradeoffs.

---

# 🎯 Failure Modes

1. Rewriting drifts meaning.
2. Multi-query returns too many duplicates.
3. Intent classifier misclassifies.
4. Latency explodes.
5. Token budget exceeded.

Every intelligence layer must be measured.

---

# 🧠 Deep Insight

Retrieval quality =

```
f(query_quality, embedding_model, index_quality)
```

Most people only tune embeddings.

Elite engineers tune query quality.

---

# 📊 Interview-Level Answers

## Why does query rewriting improve recall?

Because embeddings represent semantic direction.
If a query lacks domain-specific keywords, its vector won’t align with relevant documents.

Rewriting increases semantic alignment.

---

## Why not always use multi-query?

Because it multiplies:

* vector searches
* deduplication complexity
* reranking cost

Use when recall is critical.

---

## How would you evaluate rewrite quality?

* Compare recall@k before/after.
* Track if relevant document appears in top_k.
* Manual inspection of semantic drift.

---

# 🧠 Architect-Level Takeaway

Week 1 RAG:

```
embed(query) → retrieve
```

Week 3 RAG:

```
understand(query) → reformulate(query) → strategically retrieve
```

This is the beginning of intelligent retrieval.

---



