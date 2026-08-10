# Day 12: Frameworks Landscape — LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, and Raw Loops

## 1. The Intuition First

Every framework you'll be asked about is solving **the exact same problem you've been hand-building since Day 3**: the THINKING → ACTING → OBSERVING loop, tool schemas, state persistence, multi-agent coordination. None of them invent new capability — they package the patterns from Days 1-11 into reusable abstractions, each with a different opinion about which parts should be explicit vs. hidden from you.

Think of it like choosing between building a house with raw lumber, a prefab kit, or hiring a general contractor who also happens to sub in specialists. All three build the same house. The differences are: how much control you have over unusual requirements, how fast you can get started, how easy it is to debug when something's wrong, and how much you're locked into someone else's assumptions about how houses should be built.

**The single most important interview instinct for this topic**: never say "I'd use LangGraph" as if that's a complete answer. Say what property of the task drove the choice — because that's what's actually being tested.

---

## 2. The Landscape, Mapped to What You Already Know

### 2.1 Raw Loops (Days 2-7, hand-built)

Exactly what you've built by hand in this curriculum — a `while True` loop, explicit tool registry, explicit state handling (Day 7).

- **What you get**: total control, zero abstraction overhead, full visibility into every mechanism (Day 3's error handling, Day 7's state machine, Day 11's approval gates — all exactly as much or as little as you build).
- **What you pay**: you build and maintain everything yourself — retries, state persistence, tracing, multi-agent coordination primitives, all from scratch.
- **When it's the right choice**: novel architectures that don't fit existing frameworks' assumptions, performance-critical paths where framework overhead matters, or genuinely simple single-agent tasks where a framework's abstraction is pure overhead for a 20-line loop.

### 2.2 LangGraph — Explicit Graph/State-Machine Framework

This maps almost one-to-one onto Day 7's state machine formalization — LangGraph makes you **explicitly define nodes (states) and edges (transitions)** as a graph, with a shared state object passed between nodes.

```python
# Conceptual shape — not exact API
graph = StateGraph(AgentState)
graph.add_node("thinking", think_node)
graph.add_node("acting", act_node)
graph.add_node("observing", observe_node)
graph.add_conditional_edges("thinking", route_based_on_tool_calls, {
    "tool_call": "acting",
    "no_tool_call": END
})
graph.add_edge("acting", "observing")
graph.add_edge("observing", "thinking")
```

- **What you get**: Day 7's state machine, but with the framework handling persistence/checkpointing (Day 7 §5.2), branching, and cycles for you — plus built-in support for the AWAITING_CONFIRMATION-style interrupt points from Day 11.
- **What you pay**: you're writing to LangGraph's execution model — debugging requires understanding its abstractions, not just your own code, and the graph formalism adds real boilerplate for genuinely simple linear tasks.
- **Best fit**: exactly the tasks Day 7-11 discussed needing explicit state — long-running agents needing checkpointing, human-in-the-loop gates, complex branching logic — where you want the state machine formalized and battle-tested rather than hand-rolled.

### 2.3 AutoGen — Conversation-Centric Multi-Agent Framework

Models multi-agent systems (Day 8) as agents having a **conversation with each other** — each agent is a participant that can send/receive messages, and coordination emerges from the conversation structure (who talks to whom, in what order).

- **What you get**: Day 8's orchestrator/worker and debate patterns, but expressed natively as "agents talking to agents" rather than you manually managing dispatch/synthesis logic — good fit specifically for Day 8.2's debate pattern, since that's fundamentally agents critiquing each other's messages.
- **What you pay**: the conversational abstraction can feel like a poor fit for orchestrator/worker patterns (Day 8.1) where the "conversation" isn't natural — you're really doing task dispatch and result synthesis, and forcing that into a chat metaphor can be awkward.
- **Best fit**: multi-agent patterns where the coordination genuinely resembles a conversation/negotiation — debate, brainstorming, adversarial review — less natural for strict hierarchical dispatch.

### 2.4 CrewAI — Role-Based Multi-Agent Framework

Also targets Day 8's multi-agent patterns, but organizes around **defining agents by role** (e.g., "Researcher," "Writer," "Editor"), each with a persona, goal, and tool set, then assembling them into a "crew" with a defined process (sequential or hierarchical — literally Day 8.1's orchestrator/worker and Day 8.3's hierarchical pattern, with different naming).

- **What you get**: fast setup for orchestrator/worker-style systems (Day 8.1) via role definitions rather than hand-writing dispatch logic — opinionated defaults get you to a working multi-agent system quickly.
- **What you pay**: the role/persona abstraction can obscure what's actually happening mechanically (which context each agent sees, exactly when handoff occurs) — Day 8 §5.2's "hidden worker errors" problem is easier to accidentally overlook when the framework's role abstraction makes coordination feel automatic rather than something you're explicitly managing.
- **Best fit**: rapid prototyping of orchestrator/worker-shaped multi-agent systems where the roles map naturally onto real job functions, less suited when you need fine-grained control over context isolation (Day 8 §3.1) or custom coordination logic.

### 2.5 OpenAI Agents SDK — Lightweight, Provider-Native Orchestration

A thinner layer specifically for the Day 2/3 loop (ReAct + tool use) plus lightweight multi-agent handoffs, staying close to the raw API rather than introducing a heavy new abstraction (like LangGraph's graph model or CrewAI's role model).

- **What you get**: less abstraction distance from what you'd hand-build (Days 2-3), making it easier to reason about and debug, with built-in tracing/handoff primitives.
- **What you pay**: fewer built-in patterns for the more complex cases (Day 7's explicit state persistence, Day 10's ToT-style branching) — you're closer to raw loops with some conveniences, not a full framework for every pattern in this curriculum.
- **Best fit**: teams that want Day 2/3's core loop plus simple multi-agent handoffs without buying into a heavyweight framework's full opinionated model.

---

## 3. Worked Example: Same Task, Framework Choice Actually Matters

**Task**: "Build a customer-support agent that classifies incoming tickets, routes to a specialist sub-agent (billing, technical, or account), and requires human approval before issuing any refund over $50."

Let's reason through why each framework's fit differs — this is the actual interview skill, not memorizing feature lists.

- **Raw loop**: Fully buildable (you've done everything harder than this since Day 8 and Day 11), but you're hand-building the state persistence for the approval gate (Day 7 §5.2 — what if the approval takes 2 hours to arrive?) and the routing/dispatch logic. Reasonable if this is a small, contained system with no other framework dependencies already in your stack.

- **LangGraph**: **Strong fit.** This task is exactly a state machine with branching (Day 7) — classify → route (conditional edge) → specialist subgraph → conditional approval-gate node (Day 11) → act. LangGraph's checkpointing directly solves the "approval might take hours" persistence problem without you hand-rolling it.

- **AutoGen**: **Weaker fit.** The routing/approval structure here isn't naturally a multi-agent conversation — it's closer to Day 8.1's orchestrator/worker with a Day 11 gate bolted on. You *could* force it into AutoGen's conversational model, but you'd be fighting the abstraction rather than using it naturally.

- **CrewAI**: **Reasonable fit for the multi-agent routing part** (billing/technical/account as roles is a natural mapping), but the approval-gate requirement is less central to CrewAI's role-based model — you'd likely need to drop into custom logic for that piece anyway, partially defeating the "fast setup" benefit.

- **OpenAI Agents SDK**: **Reasonable fit** for the classify-and-handoff part (this is close to its core use case), but you'd likely be hand-building the approval-gate persistence yourself, similar to the raw-loop option, just with less boilerplate for the routing piece.

**The interview-ready synthesis**: *"For this task, I'd lean toward LangGraph specifically because the approval gate with unpredictable wait time needs real state persistence, not just a design pattern — that's a capability the framework provides natively rather than something I'd want to hand-roll. If the approval-gate requirement weren't there, and this were pure multi-agent routing, CrewAI's role-based model would get me there faster."*

---

## 4. Production Considerations

### 4.1 Framework Lock-In and Debugging Cost

Every framework abstraction — LangGraph's graph, AutoGen's conversation, CrewAI's roles — means that when something goes wrong in production, you're debugging **through** the framework's execution model, not your own code directly. This is a real, underrated cost: a raw loop's stack trace points straight at your logic; a framework's stack trace often points into the framework's internals first, and you need framework-specific knowledge to interpret it. This is the direct cost side of the abstraction-vs-control tradeoff, and it scales with how "opinionated" the framework is — LangGraph and CrewAI's stronger abstractions cost more debugging distance than the OpenAI Agents SDK's thinner layer.

### 4.2 Don't Let the Framework Choice Precede the Architecture Choice

This is the single most common interview trap on this topic: candidates jump straight to "I'd use LangGraph" without first establishing (per Days 1, 8, 10) whether the task needs single-agent, multi-agent, explicit state persistence, or branching search at all. **The correct order is always: determine the architecture from the task's actual requirements (Days 1-11), THEN pick the framework whose abstraction matches that architecture** — not the reverse. A framework choice made before the architecture decision is a sign of pattern-matching to a familiar tool rather than reasoning about the problem, which is exactly what a strong interview answer needs to avoid signaling.

### 4.3 Frameworks Version Fast — Know the Landscape, Not Exact Syntax

This entire space changes rapidly; the specific APIs shown here are illustrative, not something to memorize verbatim for an interview. What's durable and worth actually knowing cold: **the underlying pattern each framework is packaging** (state machine, conversation, role-based crew, thin native loop) — because that's what lets you reason about fit for a novel task, even against a framework you've never used, versus reciting memorized syntax that goes stale.

### 4.4 Hybrid Approaches Are Common and Legitimate

Production systems frequently mix a framework for the parts it's genuinely good at with raw/custom logic for the parts it isn't — e.g., using LangGraph for the overall state machine but writing custom tool-execution and error-handling logic (Day 3) rather than relying entirely on the framework's built-in tool-calling if your error-recovery needs are unusual. Presenting frameworks as mutually exclusive, all-or-nothing choices is a weaker answer than acknowledging that most real systems are a considered mix.

---

## 5. Interview Q&A

**Q1: How would you choose between LangGraph, AutoGen, CrewAI, and a raw loop for a given task?**
A: First determine the architecture the task actually needs — single vs. multi-agent (Day 8), whether explicit state persistence/checkpointing matters (Day 7), whether human-in-the-loop gates with unpredictable wait times are involved (Day 11) — independent of any framework. Then match: LangGraph fits when you need an explicit, checkpointable state machine with branching; AutoGen fits multi-agent patterns that are naturally conversational, like debate; CrewAI fits fast setup of role-based orchestrator/worker systems; a raw loop fits novel architectures, performance-critical paths, or genuinely simple tasks where any framework's abstraction is pure overhead.

**Q2: What's the real cost of using a heavier framework like LangGraph or CrewAI, beyond just learning curve?**
A: Debugging distance — when something breaks in production, you're now debugging through the framework's execution model (its graph engine, its role/conversation abstraction) rather than your own code directly, requiring framework-specific knowledge to interpret failures. This cost scales with how opinionated the framework is; a thinner layer like the OpenAI Agents SDK stays closer to what you'd hand-build, trading fewer built-in patterns for less debugging distance.

**Q3: A candidate says "I'd just use LangGraph for this" as their first response to a system design question. What's missing from that answer?**
A: The architectural reasoning that should come first — whether the task actually needs single vs. multi-agent, explicit state persistence, branching, or human-in-the-loop gates. Naming a framework before establishing the architecture signals pattern-matching to a familiar tool rather than reasoning from the task's requirements; the framework choice should be a consequence of the architecture decision, not a substitute for it.

**Q4: Why might AutoGen be a poor fit for a strict orchestrator/worker dispatch pattern, even though it handles multi-agent coordination?**
A: AutoGen's core abstraction models multi-agent coordination as agents having a conversation with each other, which fits patterns like debate naturally. A strict orchestrator/worker pattern (Day 8.1) is really task dispatch and result synthesis with isolated worker contexts — forcing that into a conversational metaphor can be awkward and doesn't naturally match the framework's assumptions, even though it's technically possible.

**Q5: Do you need to memorize each framework's exact API to answer questions about this topic well in an interview?**
A: No — this space changes quickly, so exact syntax goes stale. What's durable is understanding the underlying pattern each framework packages (explicit state machine, agent conversation, role-based crew, thin native loop) well enough to reason about fit for a task you've never seen a framework used on, rather than reciting memorized API calls for tasks you have seen before.

---

## 6. Summary Card

- Every framework packages patterns from Days 1-11 — **none introduce new fundamental capability**, only different abstractions over the same THINKING/ACTING/OBSERVING loop, state persistence, and multi-agent coordination.
- **LangGraph** ≈ Day 7's state machine, formalized, with built-in checkpointing/branching — best when explicit state persistence and human-in-the-loop gates (Day 11) matter.
- **AutoGen** ≈ Day 8's multi-agent patterns, modeled as conversation — best fit for debate-style coordination, weaker fit for strict dispatch.
- **CrewAI** ≈ Day 8's orchestrator/worker, modeled as roles — fast setup, less fine-grained control over context isolation.
- **OpenAI Agents SDK** ≈ Days 2-3's core loop, lightly packaged — closest to raw loops, least abstraction distance.
- **Raw loop** — full control, full maintenance burden — right when the task is novel, performance-critical, or too simple to need a framework at all.
- **Always derive architecture from the task first (Days 1-11), then pick the framework that matches it** — never the reverse.

---
*Next: Day 13 — Phase 2 Review + Interview Q&A (Architectures: single-agent, multi-agent, agentic RAG, ToT, HITL, frameworks) — consolidation day.*
