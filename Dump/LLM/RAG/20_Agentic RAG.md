Good.

Before we move to **Day 20 (Agentic RAG)**, I want you to think through yesterday’s critical question carefully:


* Precision loss in numeric filtering
* Aggregation impossibility in vector space
* Inefficiency for exact-match constraints
* Update consistency issues
* Logical operations (GROUP BY, SUM, JOIN) not being semantic similarity problems

Vector search is great for meaning.

SQL is great for logic.

Confusing the two is a common architectural mistake.

---

Now let’s level up.

# 🚀 Day 20 — Agentic RAG

## 🧠 Planning, Tool Use & Multi-Step Reasoning

So far, your pipeline is reactive:

```
Query → Retrieve → Generate
```

Agentic RAG introduces **planning**.

Instead of answering immediately, the system:

1. Understands the goal
2. Breaks it into sub-tasks
3. Calls tools
4. Iterates
5. Synthesizes final answer

This is a major jump in capability.

---

# 1️⃣ What Makes a System “Agentic”?

An agentic system can:

* Decide which tool to call
* Decide how many times to retrieve
* Use intermediate results
* Refine its own queries
* Stop when sufficient information is gathered

This is not just RAG.

This is reasoning + action.

---

# 2️⃣ Example: Non-Agentic vs Agentic

### Non-Agentic RAG

User:

> Why did churn increase after pricing changes?

Pipeline:

* Retrieve top 5 docs
* Generate answer

May miss:

* Time segmentation
* Segment breakdown
* Numeric comparison

---

### Agentic RAG

Plan:

1. Retrieve pricing change date
2. Query churn before/after via SQL
3. Retrieve customer feedback docs
4. Synthesize explanation

Now the system behaves like an analyst.

---

# 3️⃣ Core Components of Agentic RAG

```
Planner
  ↓
Tool Selector
  ↓
Tool Executor
  ↓
Memory Store
  ↓
Final Synthesizer
```

Tools may include:

* Vector retriever
* SQL executor
* Web search
* Calculator
* Code interpreter

---

# 4️⃣ Planning Strategies

## A) ReAct Pattern (Reason + Act)

Thought:

> I need churn data before/after pricing change.

Action:

> Call SQL tool

Observation:

> Got churn metrics.

Thought:

> Need qualitative reasons.

Action:

> Call vector retriever

Observation:

> Retrieved customer complaints.

Final Answer:

> Combined reasoning.

This loop continues until sufficient info is gathered.

---

## B) Self-Ask with Tool Use

LLM decomposes into sub-questions.

For each sub-question:

* Retrieve
* Answer
* Store result
* Continue

---

# 5️⃣ When Agentic RAG Is Necessary

Use agentic approach when:

* Multi-hop reasoning required
* Structured + unstructured data combined
* Conditional branching needed
* Iterative refinement improves accuracy

Don’t use it for simple FAQ bots.

---

# 6️⃣ Risk: Infinite Loops

Agents can:

* Keep calling tools endlessly
* Re-query same info
* Overthink

Solutions:

* Max step limit
* Cost budget
* Confidence threshold
* Stop token

---

# 7️⃣ Evaluation Changes for Agents

Now you must evaluate:

* Tool selection accuracy
* Planning correctness
* Step efficiency
* Final correctness

Not just answer quality.

---

# 8️⃣ Production Considerations

Agentic systems:

* Cost more
* Are slower
* Are harder to debug
* Need strong guardrails

But:

They solve harder problems.

Used in:

* OpenAI deep research modes
* Microsoft Copilot advanced workflows
* Google AI search agents

---

# 🧠 Deep Insight

RAG = Retrieval + Generation
Agentic RAG = Retrieval + Tools + Planning + Memory + Control

This is closer to autonomous systems than search.

---

# 🧪 Exercise

Take a complex question:

> Why did churn increase in Europe after Q2 pricing change, and which customer segments were most affected?

Design:

1. Planner steps
2. Tool calls
3. Intermediate results
4. Final synthesis prompt

Design it as if you were building production system.

---

# 🔥 Critical Thinking

When does adding an agent actually make the system worse?

Think about:

* Latency
* Reliability
* Over-engineering
* User expectations

Answer this carefully.

---

