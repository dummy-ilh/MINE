# Agentic Development — Interview Cheatsheet

## Core Definitions

| Term | One-line definition |
|---|---|
| **Agent** | Model controls the next step (autonomy + environment interaction + loop driven by its own outputs) |
| **Workflow** | Code controls the next step — fixed pipeline, LLM just fills in content |
| **ReAct** | Thought → Action → Observation, looped until model decides it's done |
| **Tool call** | Model emits structured text only; YOUR code parses, executes, feeds result back |
| **Plan-and-Execute** | Upfront explicit plan, then execute each step (often ReAct internally per step) |
| **Reflexion** | Self-critique after failure → reflection text injected into next attempt (in-context, no weight updates) |
| **Tree of Thought** | Generate k candidates, evaluate partial progress, prune weak branches, before committing |
| **Agentic RAG** | Retrieval as a tool the model chooses to invoke — reformulate, retry, or skip retrieval entirely |
| **Multi-agent** | Justified only by parallelism, context isolation, or specialization — not new capability |

---

## The Agent Spectrum (Day 1)
`Level 0` static pipeline → `1` LLM fills content → `2` LLM picks fixed branch (router) → `3` LLM chooses tool sequence/order, loops → `4` multi-agent, LLM delegates to sub-agents.
**Rule**: use the least agentic design that solves the problem.

## Agent as State Machine (Day 7)
`THINKING → ACTING → OBSERVING → (loop) → COMPLETE`. Failed tool calls flow through OBSERVING normally (errors are observations, not exceptions). Approval gates = one added state (`AWAITING_CONFIRMATION`) on the THINKING→ACTING edge.

## Multi-Agent Patterns (Day 8)
- **Orchestrator/Worker**: decompose → dispatch to isolated-context workers → synthesize final summaries only. Risk: orchestrator can't see worker's reasoning, only conclusion (hidden wrong-but-confident errors).
- **Debate**: independent answers → cross-critique → revise. Catches errors a single agent's self-review misses (different reasoning paths, not the same blind spots).
- **Hierarchical**: orchestrator/worker, recursively.
- Cost: orchestrator/worker parallelizes latency (not cost); debate multiplies both cost AND latency (sequential rounds).

## Planning Decision Tree (Day 4)
- Short, unpredictable task → **ReAct**
- Long-horizon, independent sub-goals, want auditability/parallelism → **Plan-and-Execute**
- Verifiable success signal, can afford retries → **Reflexion** (layer on either)
- ToT specifically: only when early choices are hard to locally evaluate AND a reliable evaluator exists (self-critique or programmatic) AND stakes justify combinatorial cost (O(k^d) worst case)

## Human-in-the-Loop (Day 11)
| | Who initiates | Trigger | 
|---|---|---|
| Approval gate | Agent | Specific action matches risk rule (policy config, not model judgment) |
| Escalation | Agent | Whole situation outside competence — hands off task, not just an action |
| Interrupt | Human | Any time — hardest to build (can't preempt in-flight actions, checked only at state boundaries) |

## Frameworks (Day 12) — always pick AFTER architecture, never before
LangGraph ≈ Day 7 state machine + checkpointing/gates. AutoGen ≈ Day 8 conversational (debate). CrewAI ≈ Day 8 role-based (orchestrator/worker). OpenAI Agents SDK ≈ thin Day 2/3 loop. Raw loop ≈ full control, full maintenance.

---

## Production Engineering Quick Reference (Phase 3)

**Reliability (14)**: retries w/ exponential backoff (transient only — malformed calls need correction, not blind retry) → timeouts (per-tool-type) → fallback models (abstracted model layer) → circuit breakers (closed/open/half-open, per-dependency). Retry safety requires idempotency for write tools.

**Observability (15)**: Trace (one run, full detail) → Log (structured, searchable across runs) → Metric (aggregated, alertable). Debug flow: metric → log → trace. Redact sensitive fields before persisting. Multi-agent needs a shared/propagated trace ID.

**Cost & Latency (16)**: token budgets (hard cap + inject remaining budget into model's own context) · caching (exact-match / prompt-prefix / semantic — each riskier than the last) · model routing (big model only where capability changes the outcome). Parallel independent tool calls = free latency win. Measure (via observability) before optimizing.

**Guardrails (17)**: model can't structurally separate instructions from data → layered defense: structural tagging (weakest alone) + least-privilege tool scoping (strongest single layer) + output validation (backstop regardless of reasoning). Sandboxing (containers, resource limits, no persistent creds, network allowlist) protects against injection AND honest mistakes.

**Evaluation (18)**: outcome eval (did it work) ≠ trajectory eval (was the process sound). Most dangerous quadrant: correct outcome + bad trajectory ("right for wrong reasons" — outcome eval can't catch it). LLM-as-judge pitfalls: position/verbosity bias, shared blind spots with generator model, vague rubrics → use specific per-criterion checks, different/stronger judge model.

**State at Scale (19)**: trigger compression before hard limit (~75%, not 100%). Narrative summary = cheap/lossy; structured extraction = higher-fidelity, needs known schema. Checkpoint the SAME compressed state as live context, not a separate raw backup. Checkpoint granularity matches stakes (fine for pending approvals).

---

## Failure Modes Quick-Recognition (Day 23)

| Failure | Signature | Fix |
|---|---|---|
| **Infinite loop** | Repeated near-identical tool calls, no state progress | Hard cap + explicit stop criteria + budget visibility + repetition detection |
| **Tool misuse** | Call succeeds but: wrong tool / wrong args / wrong timing | Sharper tool descriptions / disambiguate before acting / trajectory eval |
| **Hallucinated action** | Final answer claims an action/verification with no matching trace entry | Programmatically diff claims vs. actual trace |
| **Cascading error** | Each step looks locally reasonable; early unverified claim is the flawed foundation | Re-verify at decision points, propagate uncertainty, independent cross-check |

**The one universal check**: does every claim trace back to an actual observation in the trace? Catches ungrounded RAG, hallucinated actions, and cascading errors at once.

---

## System Design Answer Template (use for any prompt)

1. **Clarify scope** — esp. "can the agent take actions, or only answer/recommend?" (drives how much of HITL + guardrails becomes load-bearing)
2. **Level check** (Day 1) — does this need real autonomy, or is a fixed pipeline enough?
3. **Single vs. multi-agent** (Day 8) — name which of parallelism/isolation/specialization applies; if none, stay single-agent
4. **Planning strategy** (Day 4) — ReAct default; justify Plan-and-Execute or ToT only if the task's shape earns it
5. **Retrieval** (Day 9) — agentic RAG if queries are ambiguous/multi-hop; classic RAG if narrow/well-formed
6. **HITL** (Day 11) — where do gates/escalation go, and is it policy-driven?
7. **Framework** (Day 12) — pick last, justified by 1-6
8. **Phase 3 pass** — explicitly state which of reliability/observability/cost/guardrails/eval/state-mgmt are HIGH vs. LOW relevance for THIS task — don't apply all uniformly
9. **Name a tradeoff you're NOT taking** — e.g., "I'm not using ToT here because X" — exclusions are as important as inclusions

---

## Five Ideas to Remember Above All Else

1. **Least complex design that solves the problem** — every escalation in sophistication trades cost for capability; never default to maximal.
2. **The model only emits text; your code does everything else** — this is why errors become observations, why guardrails work, why state is checkpointable.
3. **Does every claim trace to an actual observation?** — the single highest-leverage diagnostic question in the whole space.
4. **No mechanism is a single point of defense** — reliability, security, and loop-prevention all need layered, redundant safeguards.
5. **Justify inclusion AND exclusion** — never list every pattern you know; explain what fits this task and what you deliberately left out.
