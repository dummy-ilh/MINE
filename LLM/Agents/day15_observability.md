# Day 15: Observability — Tracing Agent Steps, Logging Tool Calls, Debugging Failures

## 1. The Intuition First

Imagine you come back to your delivery driver from Day 14, and they say "yeah, deliveries have been rough this week." That's a mood report, not information you can act on. What you actually need: *which* deliveries failed, at *which* step (couldn't find the address? customer not home? item damaged?), *how often* this is happening, and *when* it started. Without that, you can't tell if it's one bad route, a systemic problem, or nothing at all.

That's exactly the gap between "the agent seems to be having issues" and actually being able to fix anything. **Observability is the discipline of making an agent's internal behavior — every decision, every tool call, every failure — inspectable after the fact, without having to re-run the task and hope you catch it happening live.** Days 7-14 gave you a system with named states, explicit tool calls, and reliability mechanisms — today is about making all of that VISIBLE, because a well-architected system you can't see into is still undebuggable in practice.

---

## 2. Formalizing the Three Pillars

### 2.1 Traces — The Full Shape of One Execution

A **trace** is the complete record of one agent run: every state transition (Day 7), every THINKING/ACTING/OBSERVING cycle, in order, with timing. Think of it as the flight recorder for a single trajectory.

```
Trace ID: run_8f3a2
├── [0.0s] THINKING (span: 1.2s) → model decided: call search_transcripts
├── [1.2s] ACTING (span: 0.4s) → search_transcripts("Company X Q3 earnings")
├── [1.6s] OBSERVING (span: 0.01s) → result appended (240 tokens)
├── [1.6s] THINKING (span: 0.9s) → model decided: call fetch_document
├── [2.5s] ACTING (span: 3.1s) → fetch_document(url) → ERROR: 404
├── [5.6s] OBSERVING (span: 0.01s) → error appended
├── [5.6s] THINKING (span: 1.1s) → model decided: retry search with different query
├── ...
└── [12.3s] COMPLETE → final answer returned
```

Each individual step in the trace is called a **span** — a named unit of work with a start time, duration, and outcome. Traces are hierarchical: a top-level trace for the whole agent run contains child spans for each state transition, and (in multi-agent systems, Day 8) an orchestrator's trace contains nested child traces for each worker's own full trajectory.

### 2.2 Logs — Structured Records of Specific Events

Where a trace shows the *shape* of a full execution, **logs** are individual structured records of specific events, usually queryable/searchable across many runs, not just one. The key word is **structured** — not free-text print statements, but consistent, machine-parseable fields:

```json
{"timestamp": "2026-08-09T14:32:01Z", "trace_id": "run_8f3a2", "event": "tool_call",
 "tool_name": "fetch_document", "arguments": {"url": "..."}, "duration_ms": 3100,
 "status": "error", "error_type": "HTTPError", "error_detail": "404 Not Found"}
```

Structured logging is what lets you ask cross-run questions a single trace can't answer: *"across the last 10,000 runs, what's the failure rate of `fetch_document` specifically, and has it changed in the last hour?"* — a single trace only tells you about ONE run; aggregated structured logs tell you about the SYSTEM.

### 2.3 Metrics — Aggregated Numbers Over Time

**Metrics** are the numerical rollups computed FROM traces and logs: tool call success rate, p50/p95/p99 latency per state, average iterations-to-completion, cost per trajectory, circuit-breaker open events per hour (directly feeding Day 14 §4.4's requirement that reliability events surface to observability). Metrics are what populate dashboards and trigger alerts — you don't want to read raw traces to notice a problem, you want a metric to cross a threshold and page someone.

### 2.4 How the Three Fit Together

| | Question it answers | Granularity |
|---|---|---|
| **Trace** | "What exactly happened in THIS run?" | One execution, full detail |
| **Log** | "Did this specific type of event happen, when, with what details?" | One event, searchable across runs |
| **Metric** | "How is the system doing overall, and is it getting worse?" | Aggregated across many runs |

**The typical debugging workflow moves from metric → log → trace**: a metric alert fires ("tool error rate spiked"), you query logs to find which specific runs/tools were affected, then you pull the full trace of one representative failing run to see the exact sequence of reasoning and actions that led to the failure. This top-down flow is worth stating explicitly in an interview — it shows you understand these aren't three redundant systems, they're three different zoom levels of the same underlying data.

---

## 3. Worked Example: Debugging a Real Failure Using All Three Layers

**Scenario**: A support-ticket-routing agent (from Day 11's example lineage) has started producing noticeably worse ticket categorizations over the past day, and users are complaining.

**Step 1 — Metrics catch the signal**:
```
Dashboard alert: "ticket_categorization_confidence" metric (average model-reported
confidence per categorization) dropped from 0.89 to 0.61 over the last 18 hours.
```
This is the FIRST signal — nobody read a single trace yet, a metric crossed a threshold. This is exactly why you instrument metrics: so you find out about a problem from a dashboard, not from an angry customer email three days later.

**Step 2 — Logs narrow down where**:
```sql
-- Query: which tool/step correlates with the low-confidence categorizations?
SELECT tool_name, AVG(duration_ms), COUNT(*) 
FROM tool_call_logs 
WHERE trace_id IN (SELECT trace_id FROM traces WHERE confidence < 0.7 AND timestamp > NOW() - INTERVAL '18 hours')
GROUP BY tool_name;

Result: search_knowledge_base → avg duration 8500ms (was ~400ms baseline), 340 calls
```
The structured logs reveal something specific and actionable: one particular tool, `search_knowledge_base`, has gotten dramatically slower over this exact window — a concrete lead, not just "something's wrong somewhere."

**Step 3 — Pull one full trace to see the actual mechanism**:
```
Trace ID: run_c91f0 (one representative low-confidence run)
├── [0.0s] THINKING → decided: call search_knowledge_base("refund policy category")
├── [0.1s] ACTING → search_knowledge_base(...) — duration: 9.2s (way above the ~0.4s baseline)
├── [9.3s] OBSERVING → result: EMPTY (0 documents returned, despite a 9.2s search)
├── [9.3s] THINKING → Thought: "No knowledge base results found — I'll categorize
│          based on the ticket text alone, though I'm less confident without
│          reference documentation."
├── [9.5s] ACTING → categorize(ticket, confidence=0.58)
└── [9.6s] COMPLETE
```

**Root cause now visible**: the knowledge base search is taking 20x longer than baseline AND returning empty results — almost certainly an infrastructure issue with the search index itself (maybe it's reindexing, or a node is degraded), not a model reasoning problem at all. The model is actually behaving completely correctly here — it's honestly reporting lower confidence given a genuine lack of retrieved context (exactly the "self-correction"/honest-uncertainty behavior you'd WANT, per Day 9's agentic RAG principles) — the real bug is entirely in the retrieval infrastructure, which you'd never have found by staring at the model's prompt or trying to "fix the categorization logic."

### 3.1 Why This Example Matters for the Interview

This is a deliberately realistic trap: a naive response to "categorization quality dropped" is to assume it's a *model/prompt* problem and start tweaking the categorization prompt. **The actual root cause was three layers away from the symptom** — an infrastructure degradation in a retrieval tool, surfaced only by following metric → log → trace all the way down. This is precisely the kind of debugging path a strong observability setup makes possible and a weak one makes nearly impossible — without traces showing the empty search result and its abnormal duration, you'd be guessing.

---

## 4. Production Considerations

### 4.1 What to Instrument — Every State Transition From Day 7, at Minimum

Given Day 7's explicit state machine, instrumentation is almost mechanical: emit a span at every state transition (THINKING start/end, ACTING start/end per tool call, OBSERVING), tagged with the trace ID, timestamps, and outcome. This is a direct, concrete payoff of Day 7's formalization that was flagged at the time (§5.3): *"you cannot build a trace/dashboard that says 'this agent spent 80% of its time in tool execution' unless your states are named and instrumented."* Today is that promise cashed in.

### 4.2 Cost and Latency of Observability Itself

Emitting traces/logs/metrics is not free — writing structured logs, especially at high volume (large tool inputs/outputs, long reasoning traces), adds real latency and storage cost. Common production tradeoffs: **sample** full-detail traces (e.g., log 100% of errors, but only 5% of successful runs in full detail, to control volume while still catching every failure), and **truncate** large payloads in logs (log the first/last N tokens of a huge tool result, not the entire multi-thousand-token blob, unless specifically flagged for deep debugging).

### 4.3 Sensitive Data in Traces — A Real, Concrete Risk

Agent traces routinely contain full conversation content, tool arguments, and tool results — which means they routinely contain PII, credentials, or other sensitive data flowing through the system. Logging raw tool arguments unconditionally (e.g., a `charge_credit_card` call's arguments, or a customer's full support message) can create a compliance/security liability: your observability system becomes a second, less-guarded copy of sensitive data. Production systems need explicit **redaction/scrubbing** rules for known-sensitive fields before data is persisted to logs/traces — this needs to be designed in from the start, since retrofitting redaction onto years of already-logged sensitive data is a much harder problem than preventing it from being logged in the first place.

### 4.4 Distributed Tracing Across Multi-Agent Systems (Direct Extension of Day 8)

In an orchestrator/worker system (Day 8), a single logical task spans MULTIPLE agents, each potentially with their own trace. Without a shared trace ID propagated from the orchestrator down through every worker dispatch, you end up with disconnected fragments — you can see the orchestrator's trace and each worker's trace separately, but can't easily reconstruct "here's everything that happened, across all agents, for this one user request." Production systems propagate a single top-level trace ID (or a parent-span-ID linking structure) through every dispatched worker call, so the full multi-agent trajectory can be reassembled and viewed as one coherent tree, directly enabling the kind of debugging Day 8 §5.2 flagged as otherwise impossible ("the orchestrator never sees the worker's reasoning trace" — full tracing is exactly what makes that trace available for a human debugging after the fact, even though the orchestrator itself doesn't consume it at run-time).

### 4.5 Real-Time Observability Enables Day 11's Interrupts

Recall Day 11 §5's interrupt example: a human watching a live dashboard mid-execution redirected the agent. That's only possible if traces are streamed/visible in near-real-time, not just written to logs for later batch analysis. This is a concrete reason observability and human-in-the-loop are coupled concerns: **a human can't meaningfully interrupt or approve what they can't currently see.**

---

## 5. Interview Q&A

**Q1: What's the difference between a trace, a log, and a metric, and how do they typically get used together when debugging?**
A: A trace is the full, ordered record of one execution — every state transition and tool call, in order, with timing, answering "what happened in this specific run." A log is a structured, searchable record of individual events across many runs, answering "did this type of event happen, and with what details, across the system." A metric is an aggregated number over time (success rate, latency percentiles), answering "how is the system doing overall." Debugging typically flows top-down: a metric alert signals something's wrong, logs narrow down which component/tool/time-window is affected, and a full trace of one representative failing run shows the exact mechanism.

**Q2: A categorization agent's output quality drops. Walk through how you'd use observability to find the root cause, without assuming it's a prompt problem.**
A: [Use the worked example above.] Start from the metric that caught the signal (confidence score dropped), query structured logs to correlate the drop with a specific tool or step (a particular tool's latency spiked in that window), then pull a full trace of a representative affected run to see the actual mechanism — in this case, a knowledge-base search taking 20x longer and returning empty, an infrastructure issue, not a reasoning or prompt problem. The key lesson: don't assume where the bug is; let metric → log → trace narrow it down before changing anything.

**Q3: Why can't you just log everything at full detail for every single run?**
A: Cost and latency — writing structured logs at high volume, especially with large tool inputs/outputs and long reasoning traces, adds real overhead and storage cost. Production systems typically sample (log 100% of errors but a smaller percentage of successful runs in full detail) and truncate large payloads, balancing the ability to debug failures against the cost of logging everything unconditionally.

**Q4: What's a security/compliance risk specific to agent observability, and how do you mitigate it?**
A: Traces and logs routinely capture full tool arguments and conversation content, which often includes PII, credentials, or other sensitive data — meaning your observability system can become an under-guarded second copy of sensitive information. Mitigate with explicit redaction/scrubbing rules for known-sensitive fields, designed in from the start, since retroactively cleaning already-logged sensitive data is much harder than preventing it from being logged in the first place.

**Q5: In a multi-agent orchestrator/worker system, why is a shared trace ID across all agents important?**
A: Without a trace ID propagated from the orchestrator through every worker dispatch, each agent's execution is only visible as a disconnected fragment — you can't reconstruct the full picture of what happened across all agents for one logical user request. Propagating a shared trace ID (or parent-span linking) lets you reassemble the complete multi-agent trajectory as one coherent tree, which is also what makes a worker's hidden reasoning (Day 8 §5.2's blind spot for the orchestrator at run-time) still auditable by a human debugging after the fact.

**Q6: How does observability relate to the human-in-the-loop interrupt pattern from Day 11?**
A: A human can only meaningfully send a mid-execution interrupt if they can see what the agent is currently doing — which requires traces to be visible in near-real-time, not just written to logs for later batch analysis. Real-time observability and human-in-the-loop interrupts are coupled concerns: you can't build one without the other actually being useful in practice.

---

## 6. Summary Card

- **Traces** = full detail of ONE run; **Logs** = structured, searchable events across MANY runs; **Metrics** = aggregated numbers that trigger alerts. Debugging flows metric → log → trace, top-down.
- Instrument at every Day 7 state transition — this is the direct, concrete payoff of having named states in the first place.
- Observability isn't free: sample and truncate to control cost/latency; redact sensitive fields before persisting, designed in from the start.
- Multi-agent systems (Day 8) need a shared/propagated trace ID across all agents, or the full trajectory can't be reconstructed — this is also what makes a worker's otherwise-hidden reasoning auditable after the fact.
- Real-time observability is a prerequisite for Day 11's human interrupts — you can't interrupt what you can't currently see.

---
*Next: Day 16 — Cost & Latency (token budgets, caching, model routing).*
