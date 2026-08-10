# Day 9: Agentic RAG — Retrieval as a Tool Call, Query Rewriting, Self-Correction

## 1. The Intuition First

Recall Day 1's exact distinction: is retrieval a fixed pipeline step, or a decision the model makes? Classic RAG and Agentic RAG are literally the two ends of that spectrum, applied to one specific capability: retrieval.

Think about looking something up in a library.

- **Classic RAG** is like a librarian who, no matter what you ask, always walks to the same shelf, grabs the first 5 books whose titles vaguely match your words, and hands them to you — once, unconditionally, whether or not they actually answer your question.
- **Agentic RAG** is like a research librarian who reads your question, decides *whether* a lookup is even needed, picks *which* section to search, reads what comes back, notices it's not quite right, reformulates the search ("oh, you mean the 2024 edition, not 2019"), tries again, and only stops when they've actually found something that answers you.

The librarian analogy makes the core claim precise: **classic RAG treats retrieval as a mandatory, single-shot preprocessing step; agentic RAG treats retrieval as a tool the model can choose to invoke, zero or many times, based on its own judgment of whether the results are sufficient.** This is Day 1's "workflow vs. agent" distinction, and Day 3's "tool schema" mechanics, applied to exactly one function: `search(query)`.

---

## 2. Formalizing It

### 2.1 Classic RAG — Fixed Pipeline (Day 1, Level 1)

```python
def classic_rag(user_query):
    docs = vector_search(embed(user_query), top_k=5)     # ALWAYS runs, unconditionally
    prompt = f"Context: {docs}\n\nQuestion: {user_query}"
    return llm_call(prompt)                                # single generation, no loop
```
- Retrieval happens exactly once, always, regardless of whether the query actually needs it.
- The query used for retrieval is exactly the user's raw query — no reformulation.
- There's no mechanism to notice "these 5 docs don't actually answer the question" and try again.

### 2.2 Agentic RAG — Retrieval as a Tool (Day 1, Level 3)

```python
tools = [search(query)]
messages = [{"role": "user", "content": user_query}]

while True:
    response = llm_call(messages, tools=tools)
    if response.tool_calls:
        for call in response.tool_calls:
            result = vector_search(embed(call.arguments["query"]), top_k=5)
            messages.append(tool_result(call, result))
    else:
        return response.content
```
Structurally, this is **identical to the Day 3 execution loop** — because it IS that loop, with `search` as just one more tool. The model now decides:
- **Whether to search at all** (a question like "what's 2+2" needs no retrieval).
- **What query to search with** (can be reformulated, not just the raw user question).
- **Whether to search again** (if the first results are insufficient).
- **When to stop and answer.**

This is the entire conceptual content of "agentic RAG" — there is no separate special mechanism, it's Day 3's tool-use loop with retrieval as the tool, and Day 2's ReAct pattern governing when to invoke it. If you understand Days 2 and 3 cold, "agentic RAG" is not new machinery, just a specific, very common application of it.

---

## 3. Worked Example: Query Rewriting in Action

**Task**: "How did the change we shipped last sprint affect latency?"

### 3.1 Classic RAG (fails silently)

```
Raw query embedded and searched as-is: "How did the change we shipped last sprint affect latency?"
Top-5 results: generic docs about "latency" and "sprint" that happen to share vocabulary,
but nothing about "the change" (too vague/context-dependent a phrase to embed meaningfully)
→ LLM generates an answer ungrounded in anything useful, or hedges vaguely.
```
The raw query has "last sprint" and "the change" — both context-dependent phrases that mean nothing to a vector search unless resolved to something concrete. Classic RAG has no mechanism to notice this and fix it — it embeds exactly what it's given, once.

### 3.2 Agentic RAG (reformulates)

```
Thought: "Last sprint" and "the change" are ambiguous for a search — I should first
find out what sprint and what specific change is being referenced, likely from recent
commit/deploy history, before searching latency data.
Action: search("most recent deployed change, last sprint, commit log")
Observation: "Deploy on [date]: migrated user-session lookups from Postgres to Redis cache."

Thought: Now I have a concrete change to search latency data against.
Action: search("p50 p99 latency before after Redis session cache migration")
Observation: "p99 latency dropped from 340ms to 85ms after the migration; p50 unchanged."

Thought: I have a concrete, grounded answer now.
Final Answer: "The Redis session-cache migration deployed last sprint reduced p99 latency
from 340ms to 85ms, with no significant change to p50."
```

Notice: **the model rewrote the query TWICE**, neither time using the user's literal words — first to resolve "the change" into something concrete, then to search specifically for latency data tied to that concrete change. This reformulation step — turning a vague, context-dependent user question into a concrete, retrievable query — is the single highest-leverage difference agentic RAG offers over classic RAG, and it's exactly the kind of multi-hop dependency chain from Day 2's Oppenheimer/Nolan/London example, just applied to internal company data instead of general knowledge.

---

## 4. Worked Example: Self-Correction on Insufficient Results

**Task**: "What's our refund policy for enterprise customers?"

```
Thought: I should search our policy docs for this.
Action: search("refund policy enterprise customers")
Observation: "Refund Policy (Consumer): Refunds available within 30 days of purchase for individual plans..."

Thought: This result is about the CONSUMER refund policy, not enterprise — it doesn't
actually answer the question. I should search more specifically.
Action: search("enterprise contract refund terms SLA")
Observation: "Enterprise Master Service Agreement, Section 8: Refunds for enterprise
contracts are governed by the specific SLA terms negotiated per contract; no blanket
refund window applies."

Thought: This is a better, more specific and accurate answer. I have what I need.
Final Answer: "Enterprise refunds aren't governed by a blanket policy like the consumer
30-day window — they're determined by SLA terms in each customer's specific contract
(per Section 8 of the enterprise MSA)."
```

This is the **exact same self-correction mechanism from Day 2's CEO example** — the model judges an observation as insufficient/off-target ("this is about consumers, not enterprise") and issues a refined action, rather than confidently answering with a wrong or irrelevant retrieved document, which is precisely what classic RAG would have done (it retrieves once, and the LLM would likely just generate an answer from the consumer policy doc, either hedging or — worse — confidently misapplying it).

---

## 5. Production Considerations

### 5.1 Retrieval Isn't Free — The Cost/Latency Tradeoff Is Real Here Too

Every agentic RAG turn that decides to search is a full extra round trip: embed query → vector search → inject results → another LLM call. A classic RAG pipeline is 1 embedding + 1 search + 1 generation, always. An agentic RAG trajectory that searches 3 times before answering is 3x that cost, plus the reasoning tokens for each Thought. This is Day 2 §5's runaway-loop risk, specifically manifested as "the agent keeps re-searching because it's not satisfied with results" — same mitigation: iteration caps, explicit stopping criteria ("if you've found a directly relevant passage, stop searching").

### 5.2 When Classic RAG Is Actually the Better Engineering Choice

Directly extending Day 1's "least agentic design that solves the problem" and Day 8's "don't reach for multi-agent reflexively": **if your queries are consistently well-formed and single-hop** (e.g., a narrow customer-support bot answering FAQ-style questions against a small, well-curated doc set), classic RAG's fixed single retrieval is faster, cheaper, and just as accurate — agentic RAG's reformulation/retry machinery buys you nothing if the first search was already going to succeed nearly every time. **Agentic RAG earns its cost specifically when queries are ambiguous, multi-hop, or context-dependent** — exactly the "last sprint" example above — where a single fixed retrieval structurally cannot succeed.

### 5.3 Retrieval-Augmented Generation's Classic Failure Modes Still Apply, Now Compounded

Agentic RAG doesn't eliminate RAG's usual problems (chunking quality, embedding model mismatch, stale index) — it can actually compound them if not handled carefully: a model that retries a bad query 3 times against a poorly-chunked index just burns 3x the cost for the same bad results. **Agentic RAG is not a substitute for good retrieval infrastructure — it's a layer that helps route around occasional retrieval misses, not a fix for systematically bad retrieval.** This is a common interview trap: candidates sometimes imply agentic RAG "solves" RAG's hard problems; the correct framing is it adds resilience to individual query misses, while the underlying index/chunking/embedding quality still needs to be solid on its own.

### 5.4 Deciding NOT to Retrieve — An Underrated Capability

A subtle but real production win: agentic RAG lets the model recognize when retrieval isn't needed at all ("what's 15% of 200?" needs no document lookup) and skip straight to answering — avoiding wasted latency/cost on unnecessary searches. Classic RAG has no such judgment; it retrieves unconditionally, every time, even for queries a search index can't possibly help with. This sounds minor but at scale (millions of queries) is a meaningful cost lever, and it's a good concrete example to cite when asked "what's a benefit of agentic RAG beyond query rewriting."

### 5.5 Grounding and Citation in Multi-Hop Agentic RAG

As the number of retrieval turns grows, tracking **which specific retrieved passage supports which specific claim** in the final answer gets harder — by turn 3, you have 3 separate observation blocks in context, and the final synthesis needs to correctly attribute claims back to the right source, not blend them ungrounded. Production systems often require the model to explicitly cite which observation/turn a claim came from (e.g., "[Source: Turn 2 result]") specifically to make this traceable — both for user trust and for debugging when the final answer is wrong (was it a bad retrieval, or a bad synthesis of a good retrieval?).

---

## 6. Interview Q&A

**Q1: What's the fundamental difference between classic RAG and agentic RAG?**
A: Classic RAG treats retrieval as a fixed, unconditional pipeline step — always runs once, with the raw user query, regardless of whether it's actually needed or well-formed. Agentic RAG treats retrieval as a tool the model can choose to invoke, with the query it chooses, zero or multiple times, based on its own judgment of whether results are sufficient — mechanically, it's just Day 3's tool-use loop with `search` as one of the tools, governed by Day 2's ReAct pattern.

**Q2: Give an example query where classic RAG would fail but agentic RAG succeeds, and explain precisely why.**
A: [Use the "last sprint" latency example.] A query with context-dependent phrases like "last sprint" or "the change" embeds poorly for vector search as-is — classic RAG searches the raw query once and gets irrelevant results with no recovery mechanism. Agentic RAG can reason "this phrase is ambiguous for search" and first resolve it to something concrete (e.g., find what the specific change was) before issuing a second, well-formed search — a multi-hop reformulation classic RAG structurally cannot do.

**Q3: Does agentic RAG fix bad retrieval infrastructure (poor chunking, stale index, wrong embedding model)?**
A: No — it adds resilience to individual query misses (via retry/reformulation) but doesn't fix systematically bad retrieval; if the underlying index is poorly chunked or stale, retrying a bad query against it 3 times just triples the cost for similarly bad results. Agentic RAG is a layer on top of solid retrieval infrastructure, not a substitute for it.

**Q4: When would you choose classic RAG over agentic RAG in production?**
A: When queries are consistently well-formed, single-hop, and drawn from a narrow, well-curated document set — e.g., an FAQ bot — where a single fixed retrieval already succeeds nearly every time. In that case agentic RAG's reformulation/retry machinery adds latency and cost without meaningfully improving accuracy; this follows the same "least agentic design that solves the problem" principle from Day 1.

**Q5: What's a benefit of agentic RAG beyond query reformulation that's easy to overlook?**
A: The ability to decide NOT to retrieve at all when a question doesn't need document lookup (e.g., simple arithmetic or general reasoning) — classic RAG retrieves unconditionally every time, wasting latency and cost on searches that can't possibly help. Agentic RAG's tool-use framing means retrieval is only invoked when the model judges it's actually useful.

**Q6: In a multi-turn agentic RAG trajectory, why does citation/grounding get harder as the number of retrieval turns increases, and how do you address it?**
A: With multiple retrieval turns, the final synthesis has to correctly attribute each claim back to the specific observation that supports it, rather than blending several retrieved passages together ungrounded — the more turns, the easier it is to lose track of which passage supports which claim. Production systems typically require the model to explicitly cite which turn/observation each claim is drawn from, both for user-facing trust and for debugging (distinguishing a bad retrieval from a bad synthesis of good retrieval).

---

## 7. Summary Card

- **Agentic RAG = Day 3's tool-use loop, with `search` as a tool, governed by Day 2's ReAct pattern.** No new machinery — a specific application of what you already know.
- Core capabilities classic RAG lacks: **query reformulation** (resolving ambiguous/context-dependent queries before searching), **self-correction** (retrying when results are off-target), and **deciding not to retrieve at all**.
- It does NOT fix underlying retrieval infrastructure problems (chunking, staleness, embedding quality) — it adds resilience around individual query misses, on top of infrastructure that still needs to be good.
- Choose classic RAG for narrow, well-formed, single-hop query patterns; agentic RAG earns its cost specifically for ambiguous, multi-hop, or context-dependent queries.
- Same runaway-cost risk as any agentic loop — cap retrieval retries, give explicit stopping criteria.

---
*Next: Day 10 — Tree of Thought / Search-Based Planning (when reasoning needs branching, not just a single linear trajectory).*
