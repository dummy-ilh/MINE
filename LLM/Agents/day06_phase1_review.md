# Day 6: Phase 1 Review — Foundations Consolidation

Today isn't new material — it's forcing the five days into one coherent mental model, the way you'd actually need to hold it during a 45-minute interview. Andrew Ng style: if you can't redraw the whole picture from memory, you don't own it yet.

---

## 1. The One-Page Mental Model

```
                    ┌─────────────────────────────────────┐
                    │   Is this even an agent? (Day 1)     │
                    │   Levels 0-4: who controls the       │
                    │   next step — code or the model?     │
                    └──────────────┬────────────────────────┘
                                   │ Level 3+: model controls flow
                                   ▼
                    ┌─────────────────────────────────────┐
                    │   HOW does it reason+act? (Day 2)    │
                    │   ReAct: Thought → Action →           │
                    │   Observation, looped                │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │   WHAT is "Action," mechanically?    │
                    │   (Day 3) Model emits structured      │
                    │   text; YOUR code parses + executes  │
                    │   + feeds result back as observation │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │   HOW is the sequence of actions      │
                    │   organized? (Day 4) Implicit          │
                    │   (ReAct) vs explicit upfront          │
                    │   (Plan-and-Execute) vs self-critique  │
                    │   loop on failure (Reflexion)          │
                    └──────────────┬────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │   WHAT persists across turns/         │
                    │   sessions? (Day 5) Short-term         │
                    │   (context window) vs long-term        │
                    │   (facts, RAG-style) vs episodic       │
                    │   (past attempt outcomes)              │
                    └─────────────────────────────────────┘
```

**The thread connecting all five days**: the model is a stateless text generator. Everything that makes it "agentic" — autonomy, tool use, planning, memory — is external scaffolding *you* build around repeated calls to that stateless generator. There is no magic inside the model; there's a loop, a parser, an executor, and a retrieval system, all sitting outside it.

---

## 2. Cross-Day Connections You Should Be Able to Draw Unprompted

These are the links interviewers listen for — showing you see the system as one thing, not five separate flashcards.

1. **Day 2's "Thought" and Day 4's "Reflection" are the same mechanism at different timescales.** A Thought conditions the *next single action*; a Reflection conditions the *next entire attempt*. Both work because generated reasoning text changes the distribution over subsequent tokens — reasoning isn't decorative in either case.

2. **Day 3's error-as-observation pattern is what makes Day 2's self-correction possible.** ReAct's "notice the observation was wrong, try again" only works if your execution layer (Day 3) actually surfaces failures as text instead of crashing or swallowing them. Bad Day 3 engineering silently breaks Day 2's core value proposition.

3. **Day 4's Plan-and-Execute is Day 1's Level 2 (router) generalized and made recursive.** A router picks one of N fixed branches once; a planner picks a full ordered sequence of steps once — same "decide upfront, then execute" philosophy, just at greater granularity, and each step can itself drop back down into a Day 2 ReAct loop.

4. **Day 5's episodic memory is literally Day 4's Reflexion output, persisted.** Reflexion without storage = the lesson is forgotten the moment the session ends (it's short-term/in-context only). Reflexion + episodic memory = the lesson survives and gets retrieved next time a structurally similar task appears. This is the difference between "learning within a task" and "learning across tasks" — and neither involves updating model weights.

5. **Day 1's cost/latency/predictability tradeoff table reappears in every subsequent day.** ReAct trades predictability for adaptiveness (Day 2). More tools traded for more capability costs context and accuracy (Day 3). Reflexion trades cost for reliability (Day 4). Long-term memory retrieval trades latency for personalization (Day 5). If you can name the *specific* tradeoff at each layer, that's a strong systems-design signal.

---

## 3. Rebuild-From-Memory Exercise

Before checking the answer, try to answer each without looking back at prior days.

**Try it**: "Design, end to end, an agent that can answer 'What's the weather-adjusted commute time to my office tomorrow, and should I leave earlier?' Name every component from Days 1-5 you'd use and why."

<details>
<summary>Reference answer (check after attempting)</summary>

- **Day 1 — level check**: this needs real branching (weather varies, commute varies, "office" needs to be known) — not enumerable ahead of time cleanly, so Level 3 (autonomous tool-use loop) is justified over a fixed pipeline.
- **Day 5 — memory**: retrieve long-term memory for "office address" and "usual commute mode" (stored from a past session, not re-asked every time).
- **Day 4 — planning**: this is a short, mostly-sequential task (get weather → get traffic → combine → decide) — ReAct is sufficient; Plan-and-Execute would be overkill for a 3-4 step task with no independent parallel branches worth pre-planning.
- **Day 2 — ReAct loop**: Thought ("I need tomorrow's weather at the office location") → Action (`get_weather(location, date=tomorrow)`) → Observation → Thought ("now I need current traffic estimates, weather may affect it") → Action (`get_traffic(origin, dest)`) → Observation → Thought ("I have what I need") → Final Answer with a recommendation.
- **Day 3 — tool mechanics**: two read-only tools (`get_weather`, `get_traffic`) — safe to retry freely, no idempotency concerns, could even be called in parallel since they're independent of each other (both only depend on the retrieved office address, not on each other's output) — a latency win worth calling out.
- **Day 5 — write-back**: if the user corrects something ("actually I take the train, not drive"), that's worth extracting as a new long-term memory for next time.

This answer touches every day, names the *specific* mechanism (not just the label), and justifies each choice against the alternative — exactly the shape of a strong interview answer.
</details>

---

## 4. Rapid-Fire Q&A Bank (Phase 1, Cumulative)

**Q1.** What's the single test for whether a system is "agentic"?
*A: Does the model or the code control the next step? If the model chooses the action from a non-trivial space, interacts with something external, and does so in a loop driven by its own prior outputs — it's agentic. Missing any one of those three, it's a workflow.*

**Q2.** Why does ReAct outperform plain Chain-of-Thought on tasks requiring current information?
*A: CoT reasons only from parametric knowledge with no way to verify or refresh it; ReAct grounds each reasoning step in a real observation from a tool call, so it can detect and correct for stale or wrong assumptions mid-trajectory.*

**Q3.** When the model "calls a tool," what part of that is the model actually responsible for?
*A: Only producing the structured text describing the call (name + arguments), constrained by the schema you provided. Execution, error handling, and feeding results back are entirely your application code's responsibility.*

**Q4.** Give one reason to prefer Plan-and-Execute over pure ReAct, and one reason to prefer the reverse.
*A: Prefer Plan-and-Execute when the task is long-horizon with independent sub-goals you want to inspect, parallelize, or cost-estimate before executing. Prefer pure ReAct when the task is short and highly unpredictable, where committing to an upfront plan would just mean paying for a plan you'll immediately have to throw away.*

**Q5.** What's the actual mechanism by which Reflexion improves a second attempt — not "it learns," but specifically how?
*A: A dedicated reflection step generates natural-language text articulating the specific failure; that text is injected into the next attempt's prompt, conditioning the model's next generation toward addressing that specific issue — no weight updates occur, it's entirely in-context.*

**Q6.** Your agent has been running for 3 hours in one session. What's likely degrading, and what's the fix?
*A: "Lost in the middle" effects — information technically still in the context window becomes unreliably retrieved as the context grows very long. Fix: periodic summarization/compression of older turns, plus extraction of durable facts into long-term memory rather than relying on the raw history staying usable indefinitely.*

**Q7.** Name three distinct places in this system where uncontrolled cost/latency growth is a real production risk, and the day each comes from.
*A: (Day 2) unbounded ReAct loops that never satisfy their own stopping condition; (Day 3) tool schema bloat sent on every call regardless of relevance; (Day 4) unbounded Reflexion retry cycles. All three need explicit caps in production — none of them self-limit by default.*

**Q8.** Why is "write everything to long-term memory" a bad default?
*A: It floods the store with noise, which directly degrades retrieval precision (top-k results get crowded out by irrelevant trivia) and adds unnecessary embedding/storage cost at scale — the hard part of memory systems is deciding what's durable and generalizable enough to persist, not the storage mechanics themselves.*

---

## 5. Self-Check Before Moving to Phase 2

You should be able to, without notes:
- [ ] Draw the ReAct loop and label exactly what's model-generated vs. externally-executed at each arrow.
- [ ] Explain why a router with 2 fixed branches is NOT a full agent, precisely.
- [ ] Write a tool schema from scratch and explain why the description field matters more than the type constraints.
- [ ] Name the specific production risk (not just "cost") that ReAct, Reflexion, and tool schema bloat each introduce.
- [ ] Explain, mechanically, why episodic memory is Reflexion's output made persistent.

If any of these feel shaky, that's the one to re-open before Phase 2 — Phase 2 (Days 7-13) builds architectures directly on top of all five of these primitives, so gaps compound.

---
*Next: Day 7 — Single-Agent Architecture Deep Dive (the loop in depth, state machines) — start of Phase 2: Core Architectures.*
