# Day 5: Memory Systems (Short-Term, Long-Term, Episodic)

## 1. The Intuition First

Think about the difference between three things you do as a human:

- **Holding a phone number in your head** while you walk across the room to dial it — gone the moment you're distracted. This is **working memory**: small, fast, temporary, and vanishes when the "session" ends.
- **Remembering your best friend's birthday** without re-learning it every time you see them — this fact persists indefinitely, is retrieved when relevant, and doesn't depend on the current conversation. This is **long-term memory**: durable, retrievable across sessions.
- **Remembering that last time you suggested a specific restaurant to a friend, they said they're allergic to shellfish** — a memory of a *specific past experience* that changes your future behavior. This is **episodic memory**: memory of *events*, used to inform future decisions, distinct from just "facts."

LLM agents have an analogous but fundamentally different problem: **the model itself has zero persistent memory.** Every single API call is stateless — the model has no idea anything happened before unless you explicitly put it back in the prompt. Everything we call "agent memory" is actually **engineering built entirely outside the model**, whose job is to decide what to re-inject into the context window on the next call.

> **The one-line mental model for all of agent memory**: The model has no memory. "Memory" is just *your system deciding what old information to paste back into a fresh, stateless prompt.*

---

## 2. Formalizing the Three Types

### 2.1 Short-Term / Working Memory — The Context Window Itself

This is simply the conversation history (and tool observations) accumulated *within the current session*, sitting in the context window right now. Nothing fancy — it's the `messages` list from Day 3's execution loop.

- **Capacity**: bounded by the model's context window (e.g., 200K tokens) — but practically much smaller, because quality often degrades well before the hard limit ("lost in the middle" effects — models attend less reliably to information buried deep in a very long context).
- **Lifespan**: exists only for the current session/thread. Gone the instant the session ends, unless explicitly persisted elsewhere.
- **This is what Days 2-4's entire Thought/Action/Observation loop lives inside.**

### 2.2 Long-Term Memory — Persisted, Retrievable Across Sessions

Facts, preferences, or knowledge that should survive *beyond* the current conversation. Mechanically, this is almost always implemented as:

```
1. Extract a fact worth remembering (from conversation, or from a document corpus)
2. Embed it into a vector (via an embedding model)
3. Store the vector + original text in a vector database
4. On a NEW session: embed the current query, retrieve the top-k most similar stored memories
5. Inject those retrieved memories into the new session's context window
```

This is literally the RAG pattern (retrieval-augmented generation) applied to *memories about the user/agent* instead of a static document corpus. The mechanism is identical; only the *content being retrieved* differs — facts about this specific user/agent history vs. a general knowledge base.

**Key distinction from short-term memory**: long-term memory requires an explicit **write** step (deciding what's worth remembering) and an explicit **retrieval** step (deciding what's relevant right now) — it's never "just there" the way conversation history is.

### 2.3 Episodic Memory — Memory of Specific Past Experiences/Trajectories

A special case of long-term memory, but specifically storing **records of past task attempts**, including what was tried and what the outcome was — not general facts, but *"here's what happened last time I did something like this."*

This is exactly what Reflexion (Day 4 §4) was doing — the reflection text ("my implementation didn't normalize the string...") is an episodic memory. The distinguishing feature: episodic memories are indexed by *situation similarity* ("have I faced a similar task before?") and retrieved to inform *strategy*, not just facts.

---

## 3. Worked Example: All Three Types in One System

**Scenario**: A coding assistant agent used across many sessions by the same engineer.

**Session 1 (Monday)**:
```
User: "Refactor this function to use async/await."
[... agent does the work, uses short-term memory (this session's context) throughout ...]
User: "By the way, I prefer type hints on every function, even trivial ones."
```
At the end of the session, a background process extracts: *"User prefers type hints on every function, including trivial ones."* → embeds it → stores it in a long-term memory vector store, tagged with a timestamp and topic ("coding preferences").

**Session 2 (Wednesday, new session, fresh context window)**:
```
User: "Write a helper function to parse this CSV."
```
Before generating a response, the system:
1. Embeds the current query.
2. Retrieves top-k similar long-term memories → finds Monday's stored preference (semantically close: "coding preferences" relates to "write a function").
3. Injects it into the fresh context: `[Relevant memory: User prefers type hints on every function, even trivial ones.]`
4. The model, now aware of this *without the user repeating it*, writes the function with full type hints unprompted.

**Session 5 (later, a task attempt fails)**:
```
Agent attempts to write a recursive function, hits a stack overflow on large inputs.
Reflection generated: "My recursive approach failed on inputs >10K items due to stack depth.
For large-input recursive-shaped problems with this user's codebase, I should default to
an iterative approach or explicit tail-call trampolining."
```
This is stored as an **episodic memory**, tagged by *situation type* ("large-input recursive problems"), not just as a generic fact. Session 9, when a similar-shaped task appears, this episodic memory is retrieved — informing *strategy* ("use iterative"), not just a preference.

### 3.1 Why This Distinction Actually Matters (not just taxonomy trivia)

Notice the **retrieval trigger** is different for each:
- Short-term: always present, no retrieval needed, it's just "what's currently in context."
- Long-term (preference-style): retrieved by **topical/semantic similarity** to the current query.
- Episodic: retrieved by **situational/structural similarity** to the current *task shape*, often needing a different embedding strategy — you're matching "what kind of problem is this" rather than "what fact is this about."

Conflating these in a single vector store with one embedding strategy is a common real design mistake — a system tuned to retrieve topical facts well often retrieves task-strategy episodes poorly, because the similarity signal that matters is different (semantic topic vs. structural/situational similarity).

---

## 4. Worked Example: What Goes Wrong Without Explicit Memory Management

**Failure mode — everything crammed into short-term memory, no long-term store**:

A user has a 3-hour debugging session. By hour 2, the context window contains 150K tokens of tool outputs, stack traces, and back-and-forth. The user asks a question referencing something said 2 hours ago.

- If it's still within the context window: the model *might* get it right, but "lost in the middle" effects mean information buried in the middle of a huge context is frequently under-attended to, even though it's technically present — this is a real, measured phenomenon, not a hypothetical.
- If the session ever needs to restart (browser refresh, timeout, context limit hit and truncation kicks in), that fact is **gone entirely**, because it only ever existed in short-term memory with no long-term extraction step.

**The production fix**: don't rely on "keep everything in the giant context forever." Instead:
1. **Summarize and compress** older parts of the conversation periodically (a distinct LLM call: "summarize the key facts/decisions from this conversation so far into a compact form").
2. **Extract durable facts** into long-term memory as they occur, not just at session end.
3. Keep only recent turns + retrieved relevant memories + a running summary in the active context — not the full raw history.

This directly previews Day 19 (State/Context Management at Scale), where we go deep on summarization strategies — Day 5 is the conceptual foundation; Day 19 is the production-scale engineering of it.

---

## 5. Production Considerations

### 5.1 What to Write to Long-Term Memory — The Hardest Part

The mechanics of *storing* a memory (embed + insert into vector DB) are trivial. The actually hard problem is **deciding what's worth writing in the first place**. Naive approaches:

- **Write everything**: floods the store with noise, degrades retrieval precision (top-k results become dominated by irrelevant trivia), and costs money (embedding + storage at scale).
- **Write nothing automatically, only on explicit user request** ("remember that..."): simple and predictable, but misses implicit signals users don't think to flag (like the type-hints preference above, stated in passing).

**Common production approach**: a dedicated "memory extraction" LLM call, run periodically or at session end, specifically prompted to identify durable, generalizable facts/preferences worth persisting — distinct from one-off conversational content. This is itself a small classification/extraction task, often with its own eval set (precision/recall on "should this have been remembered?").

### 5.2 Retrieval Quality — Same Problems as Any RAG System

Because long-term agent memory *is* RAG under the hood, it inherits all of RAG's classic failure modes (many of which get deeper coverage in Day 9, Agentic RAG):
- **Irrelevant retrieval**: top-k memories that are semantically similar but not actually useful right now, adding noise to context.
- **Stale memory**: a preference stored 6 months ago may no longer be true; without expiry/versioning, agents can confidently apply outdated facts.
- **Conflicting memories**: "User prefers concise answers" (stored in March) vs. "User asked for more detailed explanations" (stated in July) — retrieval alone doesn't resolve contradictions; you need explicit conflict resolution (e.g., recency-weighting, or surfacing the conflict rather than silently picking one).

### 5.3 Memory Write Race Conditions in Multi-Session/Multi-Agent Systems

If multiple agent sessions (or multiple sub-agents in a multi-agent system, Day 8) can write to the same long-term memory store concurrently, you get real distributed-systems problems: two sessions might both extract and write near-duplicate memories, or one session's write could be based on stale context that a concurrent session has already superseded. Production systems typically need deduplication (e.g., a similarity threshold check before writing — "is this materially different from an existing memory?") and sometimes explicit locking/versioning for user-profile-style memory stores.

### 5.4 Cost and Latency of the Retrieval Step

Every long-term memory retrieval is: embed the query (one model call) + vector search (fast, but not free at scale) + inject results into context (more prompt tokens on every call). For latency-sensitive paths, this adds a real round trip before the "real" LLM call even starts. Common mitigation: cache embeddings for repeated/similar queries, and only trigger memory retrieval when there's a heuristic signal it's likely to matter (e.g., not for every single turn of a fast back-and-forth, but on session start or topic shift).

---

## 6. Interview Q&A

**Q1: Does the LLM itself "remember" anything between calls? If not, what does "agent memory" actually mean?**
A: No — every LLM call is stateless; the model has no persistent state between invocations. "Agent memory" is entirely an engineering layer outside the model: a system that decides what old information (conversation history, extracted facts, past task outcomes) to retrieve and re-inject into a fresh prompt on each new call. All three memory types (short-term, long-term, episodic) are really just different strategies for *what to paste back into context and when*.

**Q2: What's the difference between long-term memory and episodic memory, mechanically?**
A: Both are persisted outside the current session and retrieved via similarity search, so structurally they're similar (embed → store → retrieve top-k). The difference is what's stored and how it's retrieved: long-term memory typically stores general facts/preferences retrieved by topical/semantic similarity to the current query, while episodic memory stores records of specific past task attempts (what was tried, what happened) retrieved by situational/structural similarity to the current task shape — you're matching "have I faced a problem like this before," which often needs a different embedding/indexing strategy than fact retrieval.

**Q3: What's the hardest part of building a long-term memory system — and it's not the vector database?**
A: Deciding what's actually worth writing. Writing everything floods the store with noise and hurts retrieval precision; writing only on explicit user request misses important implicit signals. Production systems typically use a dedicated extraction step (often its own LLM call, evaluated for precision/recall) to identify durable, generalizable facts worth persisting, distinct from one-off conversational content.

**Q4: A user says "you used to give me shorter answers, now you don't" — what memory system failure does this point to, and how do you fix it?**
A: This points to conflicting or stale memories — an old preference and a newer, contradicting one both exist in the store, and retrieval alone doesn't resolve the contradiction (it might return both, or return whichever scores marginally higher on similarity, essentially at random). Fix with explicit conflict handling: recency-weighting retrieval scores, versioning/expiring older preference memories when a newer one on the same topic is written, or in ambiguous cases, surfacing the conflict back to the user rather than silently picking one.

**Q5: Why can't you just always keep the full conversation history in context instead of building a separate long-term memory system?**
A: Two reasons: (1) hard limits — context windows are bounded, and very long sessions eventually exceed them regardless; (2) even before hitting the limit, "lost in the middle" effects mean models attend less reliably to information buried in a very long context, so a fact is technically present but practically unreliable to retrieve. Long-term memory with explicit extraction and targeted retrieval keeps the active context smaller and denser with what's actually relevant right now, rather than relying on the model to find a needle in a token haystack.

---

## 7. Summary Card

- **The model has zero memory.** All "agent memory" is external engineering deciding what to re-inject into a stateless prompt.
- **Short-term** = current context window (bounded, session-only, subject to "lost in the middle").
- **Long-term** = persisted facts/preferences, retrieved via embedding similarity (RAG pattern applied to memories).
- **Episodic** = persisted records of past task attempts, retrieved by situational similarity, used to inform *strategy* (this is what Reflexion, Day 4, actually stores).
- Hardest production problem isn't storage mechanics — it's **deciding what to write**, resolving **conflicts/staleness**, and controlling **retrieval cost/precision**.

---
*Next: Day 6 — Review + Interview Q&A for Phase 1 (Foundations: agentic spectrum, ReAct, tool use, planning, memory) — consolidation day.*
