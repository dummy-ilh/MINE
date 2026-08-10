# Day 8: Multi-Agent Systems — Orchestrator/Worker, Debate, Hierarchical Patterns

## 1. The Intuition First

Think about how a consulting firm handles a big client project versus how a single freelance consultant does.

- **The freelancer**: one person, one context, holds everything in their head, does research, writes the report, checks their own work. This is Day 7's single agent — one THINKING loop, however sophisticated.
- **The consulting firm**: a partner (orchestrator) breaks the engagement into workstreams, assigns a market-research analyst, a financial modeler, and a writer (workers) — each with their own focused context, expertise, and tools — then the partner synthesizes their outputs into one deliverable.

Why does the firm do this instead of having one super-competent generalist do everything? Not because any single workstream is impossible for one person — it's because **specialization, parallelism, and bounded context per person produce a better and faster result than one person context-switching between market research, financial modeling, and writing.**

That is the entire justification for multi-agent systems: **you're not adding capability the underlying model doesn't have — you're improving how that capability is organized: parallelized, context-isolated, and specialized.** If you're not getting one of those three benefits, adding more agents is pure overhead, not architecture.

---

## 2. Formalizing the Three Core Patterns

### 2.1 Orchestrator/Worker (a.k.a. Supervisor Pattern)

One agent (the **orchestrator**) has NO domain tools itself — its only job is to decompose the task and dispatch sub-tasks to specialized **worker** agents, then synthesize their results.

```
                    ┌──────────────┐
                    │ Orchestrator  │  (decomposes, dispatches, synthesizes)
                    └──────┬────────┘
              ┌────────────┼────────────┐
              ▼             ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker A │ │ Worker B │ │ Worker C │  (each: own context, own tools, own ReAct loop)
        │(research)│ │(finance) │ │(writing) │
        └──────────┘ └──────────┘ └──────────┘
```

Each worker is, internally, exactly the Day 7 single-agent state machine — its own THINKING/ACTING/OBSERVING loop, with its OWN tool set and its OWN context window, isolated from the other workers. The orchestrator's context contains only: the original task, the sub-task assignments, and the workers' final outputs — **not** each worker's full internal reasoning trace. This context isolation is the single most important mechanical detail of this pattern.

### 2.2 Debate (a.k.a. Multi-Agent Deliberation)

Multiple agents, each with the SAME task and typically similar capabilities, independently produce an answer, then critique each other's answers, iterating toward consensus or a judged best answer.

```
Agent A: produces Answer_A ──┐
Agent B: produces Answer_B ──┼──▶ Each agent sees the OTHERS' answers ──▶ 
Agent C: produces Answer_C ──┘     Each revises given the critique ──▶ repeat N rounds ──▶ Judge/vote on final
```

Unlike orchestrator/worker (division of labor), debate is about **using disagreement to surface errors** — an answer that survives multiple independent agents' critique is more likely to be correct than a single agent's first-pass answer, similar in spirit to ensemble methods in classical ML, but the "ensembling" happens via natural-language critique, not just output averaging.

### 2.3 Hierarchical (Multi-Level Orchestration)

Orchestrator/worker, but the "workers" can themselves be orchestrators of their own sub-workers — recursion.

```
                         ┌──────────────────┐
                         │ Top Orchestrator   │
                         └─────────┬───────────┘
                    ┌──────────────┼──────────────┐
                    ▼                              ▼
          ┌──────────────────┐          ┌──────────────────┐
          │ Sub-Orchestrator A │          │  Worker B          │
          └─────────┬───────────┘          └──────────────────┘
              ┌──────┼──────┐
              ▼             ▼
        ┌──────────┐ ┌──────────┐
        │ Worker A1│ │ Worker A2│
        └──────────┘ └──────────┘
```

Used when a sub-task is itself complex enough to warrant its own decomposition — e.g., "market research" workstream is itself broken into "competitor analysis" and "customer survey analysis" sub-workers, each reporting up to Sub-Orchestrator A, which reports up to the Top Orchestrator.

---

## 3. Worked Example: Orchestrator/Worker End to End

**Task**: "Write a due-diligence brief on Company X for a potential acquisition."

**Orchestrator's THINKING**:
```
Thought: This task has 3 clearly separable workstreams: (1) financial health,
(2) legal/regulatory risk, (3) market position. These are largely independent —
I can dispatch them in parallel and synthesize once all three return.

Action: dispatch_worker("financial_analyst", task="Assess Company X's financial health from public filings")
Action: dispatch_worker("legal_analyst", task="Identify any pending litigation or regulatory issues for Company X")
Action: dispatch_worker("market_analyst", task="Assess Company X's competitive position and market share trends")
```

**Worker A (financial_analyst) — runs its OWN full Day 7 state machine, invisible to the orchestrator**:
```
[Worker A's private context — orchestrator never sees this level of detail]
Thought: I need Company X's recent 10-K filing.
Action: search_filings("Company X 10-K")
Observation: [filing found]
Thought: Let me check revenue trend and debt levels.
Action: extract_financials(filing_url)
Observation: {revenue_growth: "8% YoY", debt_to_equity: 1.4}
Thought: I have enough to summarize.
Final Answer (returned to orchestrator ONLY): "Company X shows healthy 8% YoY revenue growth
but a debt-to-equity ratio of 1.4, above industry median of 0.9 — moderate leverage risk."
```

**Worker B and Worker C** run in parallel, similarly, each returning only a final summary.

**Orchestrator's THINKING (after all 3 return)**:
```
Observation: [3 worker summaries received]
Thought: I now have financial, legal, and market findings. I'll synthesize into one brief.
Final Answer: "Due diligence summary for Company X: Financially, [financial_analyst's summary].
Legally, [legal_analyst's summary]. Competitively, [market_analyst's summary].
Overall recommendation: proceed with caution given the leverage risk noted above."
```

### 3.1 Why Context Isolation Is the Real Point (Not Just Division of Labor)

If you ran this as a SINGLE agent instead, its context window would accumulate: every filing search, every extraction call, every legal-database query, every market-report snippet — potentially tens of thousands of tokens of intermediate tool traffic, all in ONE context, before it ever gets to synthesize. Two concrete costs of that:

1. **"Lost in the middle" (Day 5)** — by the time the single agent reaches the synthesis step, early findings (financial data) are buried under later ones (market data), and synthesis quality measurably degrades.
2. **Cross-contamination of reasoning** — a single agent bouncing between "am I checking legal risk or financial risk right now" repeatedly loses focus, compared to a worker whose ENTIRE context is scoped to one workstream and nothing else.

**The interview-ready framing**: *"Multi-agent isn't about giving the system new capabilities — the same underlying model could theoretically do all three workstreams sequentially in one context. It's about context isolation: each worker's context stays small, focused, and uncontaminated by irrelevant tool traffic from other workstreams, which measurably improves per-workstream quality and enables parallel execution."*

---

## 4. Worked Example: Debate Pattern Catching an Error

**Task**: "Is it safe to deploy this database migration script during business hours?"

**Round 1 — independent answers**:
```
Agent A: "Yes, the script only adds a new nullable column, which is a non-blocking
operation in this database engine — safe during business hours."

Agent B: "No — the script includes a CREATE INDEX statement, which takes a table-level
lock on this database engine version and would cause a production outage during business hours."

Agent C: "Yes, nullable column additions are safe."
```

**Round 2 — each agent sees the others' answers, revises**:
```
Agent A: "I missed the CREATE INDEX statement further down in the script — Agent B is
correct, that operation does lock the table on this engine. Revising my answer to: NOT safe
during business hours without using CONCURRENTLY/online index creation."

Agent C: "Agreed with B and A's revision — I only evaluated the column addition and
missed the index statement entirely."
```

**Judge/aggregation**: Majority-revised consensus → "NOT safe as written; requires online index creation to be safe during business hours."

### 4.1 Why This Is Genuinely Different From Just Running One Agent Twice

The critical mechanical detail: **Agent A did not catch its own mistake by re-reading the script a second time** (that's just retrying — Day 4 §4.1 covered exactly why that's weak) — it caught the mistake because **Agent B's independent, differently-focused pass surfaced something Agent A's reasoning path skipped entirely.** This is the actual value of debate: different agents, run independently, don't share the same reasoning blind spots on a given pass, so cross-critique surfaces errors that repeated self-review from a single reasoning path tends to miss.

**Where debate is NOT worth the cost**: for tasks with a single obviously correct, easily-verifiable answer (Day 18's eval concepts apply here), debate's extra cost buys little — 3x the LLM calls for a question a single agent already answers correctly and confidently. Debate earns its cost specifically on judgment-heavy, error-prone, high-stakes tasks — like the production-safety question above — where independent perspectives genuinely change the outcome.

---

## 5. Production Considerations

### 5.1 Cost Multiplication — This Is the Central Tradeoff of the Entire Topic

- Orchestrator/worker with 3 workers: roughly 4x the LLM calls of a single agent (orchestrator + 3 workers), though workers run in **parallel**, so wall-clock latency can be close to a single worker's latency, not 4x — cost and latency scale differently here, and you should be explicit about which one you're optimizing when discussing this pattern.
- Debate with 3 agents over 2 rounds: roughly 6x the LLM calls (3 agents × 2 rounds) of a single agent, and this is largely **sequential-ish** (round 2 needs round 1's outputs from all agents), so both cost AND latency multiply — debate is expensive on both axes, which is exactly why it's reserved for high-stakes judgment calls.

### 5.2 Failure Propagation — A New Failure Mode Multi-Agent Introduces

In a single agent, one bad reasoning step is visible and correctable in the next Thought (Day 2). In orchestrator/worker, if Worker A returns a **confidently wrong** summary (not an error — a plausible-sounding but incorrect conclusion), the orchestrator has NO way to catch this, because it never sees Worker A's underlying reasoning trace, only the final summary. **This is a genuine, non-obvious production risk**: context isolation, the very thing that makes multi-agent valuable (§3.1), is the same thing that hides a worker's bad reasoning from the layer that could otherwise catch it.

**Mitigations**:
- Require workers to return not just a conclusion but a brief justification/evidence trail, giving the orchestrator (or a human reviewer) something to sanity-check against.
- For high-stakes workstreams, pair a worker with a second worker doing the same task (essentially debate, scoped to one workstream) rather than trusting a single worker's output blindly.
- Log full worker traces to an observability system (Day 15) even though the orchestrator itself doesn't see them — so a human can audit after the fact if the final output seems wrong.

### 5.3 Orchestrator as a Single Point of Failure / Bottleneck

If the orchestrator's decomposition is wrong (e.g., it fails to recognize that "legal risk" and "financial risk" actually have an important interaction that needs to be evaluated jointly, not separately), no amount of worker quality fixes that — the workers are only ever as good as the decomposition they were handed. This mirrors Day 4 §5.3's plan-granularity problem exactly, but now the "plan" is a task decomposition across separate contexts instead of separate steps within one context — same failure shape, higher stakes because workers can't easily "see" what they're missing from siblings' context.

### 5.4 When NOT to Use Multi-Agent (the interview trap)

Given how much attention multi-agent architectures get, interviewers specifically probe whether you'll reach for it reflexively. The correct instinct, directly extending Day 1's "least agentic design that solves the problem": **multi-agent adds real cost and real new failure modes (§5.2) — only justify it when you can point to a specific one of the three benefits (parallelism, context isolation, specialization) that a single well-designed agent genuinely can't get.** A task that's sequential, fits comfortably in one context window, and doesn't need genuinely different tool/expertise scopes gains nothing from being split into multiple agents — it just gets slower and more expensive for no benefit.

---

## 6. Interview Q&A

**Q1: What's the actual justification for using multiple agents instead of one more capable single agent?**
A: It's not about unlocking new capability — a sufficiently large context window could theoretically let one agent do everything. The real benefits are parallelism (independent workstreams execute concurrently), context isolation (each agent's context stays small and focused, avoiding "lost in the middle" degradation and cross-workstream reasoning contamination), and specialization (each agent can have a narrower, more precisely-scoped tool set and system prompt). If a task doesn't clearly benefit from at least one of these three, splitting it into multiple agents adds cost and failure surface without benefit.

**Q2: Walk through the orchestrator/worker pattern and explain what context the orchestrator does and doesn't see.**
A: [Use the due-diligence example above.] The orchestrator decomposes the task into independent sub-tasks and dispatches them to workers; each worker runs its own full internal reasoning loop (its own THINKING/ACTING/OBSERVING cycle) in an isolated context with its own tools. The orchestrator only receives each worker's final summary/output — it never sees the worker's intermediate tool calls or reasoning trace, which is what keeps the orchestrator's own context small enough to synthesize effectively, but also means it has no way to catch a worker that's confidently wrong.

**Q3: How does the debate pattern actually catch errors that a single agent re-checking its own work would miss?**
A: A single agent re-reading its own output tends to repeat the same reasoning blind spots that produced the error in the first place — it's the same reasoning path run twice. In debate, multiple agents independently produce answers along different reasoning paths, then see each other's answers; an error one agent's path missed is often caught by another agent's differently-focused pass, and revision happens because of that external critique, not from re-inspection alone.

**Q4: What's a failure mode that's specific to multi-agent systems and doesn't really exist in a well-designed single agent?**
A: A worker returning a confidently wrong (not erroring, just incorrect) conclusion that the orchestrator has no way to catch, because context isolation means the orchestrator never sees the worker's underlying reasoning — only its final summary. In a single agent, a bad reasoning step is visible in the same context and can be caught by a subsequent Thought; in orchestrator/worker, the very isolation that makes the pattern valuable also hides bad reasoning from the layer that could otherwise catch it.

**Q5: When would you explicitly recommend AGAINST using a multi-agent architecture?**
A: When the task is sequential (no independent workstreams to parallelize), fits comfortably in a single context window without degradation, and doesn't require meaningfully different tool sets or expertise per sub-task — i.e., none of the three core benefits (parallelism, context isolation, specialization) actually apply. In that case, multi-agent just multiplies cost (roughly Nx LLM calls) and introduces new failure modes (like hidden worker errors) without buying anything a single well-designed agent doesn't already provide.

**Q6: Compare the cost/latency profile of orchestrator/worker vs. debate.**
A: Orchestrator/worker with N workers costs roughly (N+1)x the LLM calls of a single agent, but because workers run in parallel, wall-clock latency stays close to a single worker's latency, not Nx — you're paying more in cost but not necessarily in time. Debate with N agents over R rounds costs roughly N×R calls, and because later rounds depend on earlier rounds' outputs across all agents, it's largely sequential — so debate multiplies both cost AND latency, which is why it's reserved for judgment-heavy, high-stakes tasks rather than used by default.

---

## 7. Summary Card

- Multi-agent's justification is **parallelism, context isolation, or specialization** — not new capability. If none apply, don't use it.
- **Orchestrator/worker**: decompose → dispatch to isolated-context specialists → synthesize final summaries only (orchestrator never sees worker internals).
- **Debate**: independent answers → cross-critique → revise → consensus/judge — catches errors a single agent's self-review tends to miss, because different agents don't share the same reasoning blind spots.
- **Hierarchical**: orchestrator/worker, recursively, when a sub-task itself needs decomposition.
- New production risk introduced: **hidden worker errors** — context isolation that makes the pattern valuable also hides bad reasoning from the orchestrator. Mitigate with justification trails, paired workers on high-stakes streams, and full trace logging for audit.
- Cost/latency multiply differently per pattern: orchestrator/worker parallelizes latency but not cost; debate multiplies both.

---
*Next: Day 9 — Agentic RAG (retrieval as a tool call, query rewriting, self-correction).*
