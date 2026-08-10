# Day 13: Phase 2 Review — Architectures Consolidation

Same format as Day 6: force Days 7-12 into one coherent picture you can redraw and defend, not six separate flashcards.

---

## 1. The One-Page Mental Model

```
                ┌───────────────────────────────────┐
                │  Day 7: How is ONE agent           │
                │  structured internally?            │
                │  THINKING → ACTING → OBSERVING →   │
                │  COMPLETE, explicit state machine  │
                └──────────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │ one agent enough?                │ needs more?
              ▼                                   ▼
    ┌───────────────────┐              ┌───────────────────────┐
    │ stay single-agent,  │              │ Day 8: MULTIPLE agents │
    │ deepen with Day 9/10│              │ orchestrator/worker,   │
    └─────────┬───────────┘              │ debate, hierarchical   │
              │                          └───────────┬────────────┘
              │                                       │
    ┌─────────▼──────────┐              ┌─────────────▼─────────────┐
    │ Day 9: retrieval as  │              │ Day 8 §5.2: context        │
    │ a TOOL the agent      │              │ isolation hides worker      │
    │ chooses (agentic RAG) │              │ errors from orchestrator    │
    └─────────┬─────────────┘              └─────────────────────────────┘
              │
    ┌─────────▼──────────┐
    │ Day 10: when linear   │
    │ reasoning isn't enough,│
    │ branch explicitly (ToT)│
    └─────────┬───────────────┘
              │
    ┌─────────▼──────────────────────────┐
    │ Day 11: WHERE does a human sit in    │
    │ any of the above? Approval gates,     │
    │ escalation, interrupts                │
    └─────────┬────────────────────────────┘
              │
    ┌─────────▼──────────────────────────┐
    │ Day 12: which framework packages      │
    │ whichever of the above you chose?     │
    │ (architecture decision FIRST,         │
    │ framework SECOND)                      │
    └────────────────────────────────────────┘
```

**The thread through all six days**: Day 7 gives you the atomic unit (one agent's internal structure). Days 8-10 are all variations on "how much reasoning structure does this task need" — more agents (Day 8), tool-mediated retrieval (Day 9), or explicit branching (Day 10) — each answering a different axis of "the naive single ReAct loop isn't enough, what do I add." Day 11 asks a completely orthogonal question — not "how does the agent reason" but "where does a human need to sit in this" — which applies on top of ANY of Days 7-10's architecture. Day 12 is purely an implementation-layer question that should always come last.

---

## 2. Cross-Day Connections You Should Be Able to Draw Unprompted

1. **Day 8's orchestrator IS Day 7's state machine, one level up.** The orchestrator has its own THINKING/ACTING/OBSERVING loop; "ACTING" for an orchestrator just happens to mean "dispatch to a worker" instead of "call a tool." Same state machine, different granularity of what an "action" is.

2. **Day 9 is not a new architecture — it's Day 3's tool-use loop with one specific tool.** If you can already answer "how does an agent decide to call a tool, retry on failure, and know when to stop," you already know agentic RAG; the content is entirely in recognizing that retrieval fits the exact same mold.

3. **Day 10's evaluator problem is the same shape as Day 8's "hidden worker errors" problem.** Both are fundamentally: *something is judging quality/promise of a partial result, and that judgment can itself be wrong or unreliable* — ToT's evaluator scoring a partial reasoning path, and an orchestrator trusting a worker's summary, are the same underlying risk (a judgment step with no ground truth to check against) in two different architectures.

4. **Day 11's approval gate is literally an extra state in Day 7's state machine** (AWAITING_CONFIRMATION), which means it composes cleanly with Day 8 (gate a worker's proposed action), Day 9 (gate before executing a write-tool discovered via agentic RAG — e.g., "update this document"), and Day 10 (gate before committing to executing the winning branch of a ToT search on a high-stakes plan). Human-in-the-loop isn't a separate architecture — it's a state you can insert into any of them.

5. **Day 12's frameworks are literally named after which of Days 7-11 they formalize**: LangGraph → Day 7 (state machine) + Day 11 (checkpointed gates); AutoGen → Day 8 debate; CrewAI → Day 8 orchestrator/worker; OpenAI Agents SDK → Days 2-3's core loop. If you understand Days 7-11 deeply, Day 12 is just relabeling, not new content.

6. **The cost/complexity tradeoff from Day 1 reappears at every single layer, in the same shape each time**: single→multi-agent costs more calls for context isolation/parallelism (Day 8); ReAct→agentic RAG costs more calls for query correction (Day 9); linear→ToT costs combinatorially more for branch exploration (Day 10); no-gate→gated costs latency for safety (Day 11); raw loop→framework costs abstraction/debugging distance for built-in machinery (Day 12). **Every escalation in sophistication in this entire curriculum trades some form of cost for some form of reliability or capability — naming the specific trade at each layer, not just "it costs more," is the actual interview signal.**

---

## 3. Rebuild-From-Memory Exercise

**Try it before checking the answer**: "Design an agentic system for automated code review on pull requests: it should read the diff, check it against team style guidelines (stored in a large internal wiki), flag potential bugs, and — for anything touching payment/billing code — require a senior engineer's sign-off before posting review comments. Walk through the full architecture."

<details>
<summary>Reference answer</summary>

- **Day 1 level check**: genuinely needs autonomy — style-guideline relevance and bug-worthiness aren't enumerable branches — Level 3 (tool-using loop) justified, not a fixed pipeline.
- **Day 9 — agentic RAG**: the "large internal wiki" of style guidelines is a retrieval problem, and it should be agentic rather than classic RAG, because "does this diff violate a guideline" often needs reformulated, multi-hop queries (e.g., resolve which subsystem the diff touches, THEN search guidelines specific to that subsystem) — exactly Day 9's "last sprint" pattern.
- **Day 8 — multi-agent, orchestrator/worker**: reasonable to split into a style-checker worker (uses Day 9's agentic RAG against the wiki) and a bug-pattern worker (uses static-analysis-style tools), each with isolated context, dispatched in parallel by an orchestrator, since these are genuinely independent workstreams (Day 8 §3.1's justification) — NOT debate, since these aren't producing competing answers to the same question, they're covering different concerns.
- **Day 7 — state machine**: each worker internally runs THINKING/ACTING/OBSERVING; the orchestrator's own loop treats "dispatch to workers" as its ACTING state.
- **Day 11 — approval gate**: the orchestrator's synthesis step, before posting any comment, checks whether the diff touches payment/billing code (a specific, rule-based trigger condition, per Day 11 §6.1 — this should be explicit policy, not model judgment) and if so transitions to AWAITING_CONFIRMATION rather than directly posting — needs Day 7 §5.2-style state persistence, since a senior engineer's sign-off could take hours.
- **Day 10 — probably NOT needed**: this task doesn't have the "early choice is hard to evaluate and matters a lot for the final outcome" shape that justifies ToT's cost — style/bug checking is closer to independent evaluation than combinatorial branching search, so bringing in ToT here would be over-engineering per Day 10 §5.1's guidance.
- **Day 12 — framework choice, last**: given the approval gate with unpredictable wait time (senior engineer sign-off) and the orchestrator/worker structure, LangGraph is a strong fit for the state persistence, and CrewAI could handle the worker role definitions — but this choice comes only after all the above architecture is decided, not before.

This answer explicitly justifies inclusion (Day 8, Day 9, Day 11) AND exclusion (Day 10) of specific patterns — that's the mark of a strong systems-design answer, not just listing every pattern you know.
</details>

---

## 4. Rapid-Fire Q&A Bank (Phase 2, Cumulative)

**Q1.** What's the one thing an orchestrator's context should NOT contain, and why?
*A: A worker's full internal reasoning trace/tool-call history — only the worker's final summary. This keeps the orchestrator's own context small and synthesis-focused, but it's also exactly why hidden worker errors (Day 8 §5.2) are a real risk: the orchestrator has no visibility into how a worker reached its conclusion.*

**Q2.** How is agentic RAG's "decide not to retrieve" capability connected to Day 1's core framing?
*A: It's the model exercising autonomy over whether an action (retrieval) is needed at all — exactly Day 1's definition of what makes something agentic (the model, not fixed code, controls whether and when the action happens), applied to one specific tool.*

**Q3.** Why is ToT's evaluator described as "the real bottleneck," and what's the parallel risk in Day 8?
*A: Because generating diverse candidates is easy but judging which partial path is actually promising is hard and can be unreliable — the same underlying risk as an orchestrator trusting a worker's confidently-stated but possibly-wrong summary: a judgment step with no independent ground truth to verify it.*

**Q4.** Where exactly does a Day 11 approval gate fit into a Day 7 state machine?
*A: As an additional named state (AWAITING_CONFIRMATION) inserted on the transition edge between THINKING (which proposed an action) and ACTING (which would execute it) — it's a local, well-scoped addition to the state machine, not a redesign of it.*

**Q5.** You're asked to design a system and you're not sure if it needs multi-agent. What's the test?
*A: Per Day 8 §5.4 — does the task genuinely benefit from parallelism (independent workstreams), context isolation (avoiding cross-contamination/"lost in the middle" from Day 5), or specialization (different tool sets/expertise per sub-task)? If none apply, a single well-designed agent is the right call, and multi-agent just adds cost and new failure modes (hidden worker errors) for no benefit.*

**Q6.** Name the specific cost each of these trades for reliability/capability: multi-agent (Day 8), agentic RAG (Day 9), ToT (Day 10), approval gates (Day 11).
*A: Multi-agent trades extra LLM calls (roughly Nx) for parallelism/context isolation/specialization. Agentic RAG trades extra round trips (search-reformulate-search) for handling ambiguous/multi-hop queries classic RAG can't. ToT trades combinatorial (O(k^d)) cost for exploring branches before committing, avoiding wasted full-trajectory dead ends. Approval gates trade latency (an unpredictable human-response wait) for safety on irreversible/costly actions.*

---

## 5. Self-Check Before Moving to Phase 3

You should be able to, without notes:
- [ ] Draw Day 7's state machine AND show exactly where a Day 11 approval gate slots in.
- [ ] Explain, in one sentence, why agentic RAG is "nothing new" mechanically, and what specifically it adds over classic RAG.
- [ ] State the precise difference between an approval gate and escalation, with an example of each.
- [ ] Given a novel task description, justify BOTH which Day 7-11 patterns you'd include AND which you'd explicitly exclude (per the code-review exercise above) — not just list everything you know.
- [ ] Name which framework (Day 12) maps to which architectural pattern (Days 7-11), and articulate why architecture must be decided before framework.

Phase 3 (Days 14-20) shifts from "how is this system architected" to "how does this system survive contact with real production traffic" — reliability, observability, cost, guardrails, evaluation. Everything in Phase 3 assumes you can already correctly architect the system; it's now about making that architecture actually work at scale, under failure, and under adversarial conditions.

---
*Next: Day 14 — Reliability (retries, timeouts, fallback models, circuit breakers) — start of Phase 3: Production Engineering.*
