# Day 7: Single-Agent Architecture — The Loop in Depth, State Machines

## 1. The Intuition First

Days 1-6 treated "the agent loop" as a black box: `while True: think, act, observe`. Today we open that box fully and ask the question a systems-design interview actually cares about: **what is the loop's internal state, precisely, and how does it transition?**

Think about a thermostat. It's not "smart," but it *is* a loop with state:
```
State: {current_temp, target_temp, mode: [idle, heating, cooling]}
Loop: read current_temp → compare to target → transition mode if needed → act (turn on/off heater)
```
It has exactly one thing it's ever "doing" — a **current mode** — and clear rules for when that mode changes. This is a **state machine**: a finite set of named states, plus rules for transitioning between them based on inputs.

An LLM agent, underneath all the "reasoning" language, is exactly this — just with far more states and much fuzzier (model-decided, not hardcoded) transition rules. Making this explicit is what separates a toy `while True` loop that breaks in production from an architecture you can actually debug, test, and reason about.

---

## 2. Formalizing It: The Agent as an Explicit State Machine

### 2.1 The Naive Loop (what we've shown through Day 6)

```python
while True:
    response = llm_call(messages, tools=tools)
    if response.tool_calls:
        execute_and_append(response.tool_calls)
    else:
        return response.content
```

This has exactly **2 implicit states**: "waiting for next model decision" and "done." That's fine for toy examples, but production agents need more granularity to be debuggable, testable, and controllable. Here's the state machine most real single-agent systems actually implement, even if it's not always drawn explicitly:

```
┌───────────┐     needs tool call      ┌────────────┐
│  THINKING │ ────────────────────────▶│  ACTING     │
│ (LLM call)│                          │ (execute)   │
└─────┬─────┘                          └──────┬──────┘
      │                                        │
      │ no tool call = done                    │ result ready
      ▼                                        ▼
┌───────────┐                          ┌────────────┐
│ COMPLETE  │◀─────── max iters ───────│ OBSERVING   │
│  (return) │        or error cap      │ (append)    │
└───────────┘                          └──────┬──────┘
      ▲                                        │
      │                                        │
      └────────────────────────────────────────┘
                    loop back to THINKING
```

Five explicit states: **THINKING → ACTING → OBSERVING → (loop back to THINKING) → COMPLETE**, plus an escape hatch to COMPLETE from anywhere via iteration/error caps.

### 2.2 Why Making States Explicit Actually Matters

This isn't academic formalism for its own sake — each explicit state is a place you can:
- **Attach a timeout** (ACTING taking too long? Kill it, transition to an ERROR state, not an infinite hang).
- **Attach logging/tracing** (Day 15 builds directly on this — you can't trace what you haven't named).
- **Attach a guard/validation check** (before transitioning OBSERVING → THINKING, validate the tool result isn't malformed).
- **Test in isolation** (you can unit-test "given this OBSERVING state, does the state machine correctly transition to THINKING with the right context appended?" without running a full live LLM call).

A `while True` loop with no named states can only be tested end-to-end, black-box. A state machine can be tested state-by-state, transition-by-transition — this is a real, practical software-engineering difference, not just terminology.

---

## 3. Worked Example: Tracing State Transitions Explicitly

**Task**: "Summarize the latest quarterly earnings call for Company X."

Let's trace the *state*, not just the conversation, at each step:

```
State: THINKING
  messages = [user: "Summarize the latest earnings call for Company X"]
  LLM call → response has tool_call: search_transcripts("Company X Q3 earnings call")
  Transition: THINKING → ACTING (tool call present)

State: ACTING
  Execute search_transcripts("Company X Q3 earnings call")
  Result: transcript_url found successfully
  Transition: ACTING → OBSERVING (execution completed, no error)

State: OBSERVING
  Append tool result to messages
  messages = [..., tool_result: "transcript found at url X"]
  Transition: OBSERVING → THINKING (loop continues, more work likely needed)

State: THINKING
  LLM call → response has tool_call: fetch_document(transcript_url)
  Transition: THINKING → ACTING

State: ACTING
  Execute fetch_document(...)
  Result: ERROR — 404, transcript URL expired
  Transition: ACTING → OBSERVING (execution completed, WITH an error — still transitions normally, error becomes content)

State: OBSERVING
  Append error as tool result: "Error: 404, URL expired"
  Transition: OBSERVING → THINKING

State: THINKING
  LLM call → response reasons: "URL expired, let me search again with a fresher query"
  → tool_call: search_transcripts("Company X Q3 2026 earnings call transcript")
  Transition: THINKING → ACTING

  [... continues similarly ...]

State: THINKING (final)
  LLM call → response has NO tool_call, just text: "Here's the summary: ..."
  Transition: THINKING → COMPLETE

State: COMPLETE
  Return response.content to caller.
```

### 3.1 What This Trace Makes Visible That the Naive Loop Hides

Notice: **the ACTING → OBSERVING transition happens identically whether the tool call succeeded or failed.** This is the same "errors are observations, not exceptions" principle from Day 3 — but now it's visible as a structural fact about the state machine, not just a coding convention: **there is no separate ERROR state for tool failures** — a failed tool call is just an OBSERVING state with error content, which flows back into THINKING normally. This is a deliberate design choice you should be able to articulate: *"tool failures are recoverable by design — they become part of the reasoning context, not a special-cased crash path."*

Compare this to where a **real** ERROR/terminal state *is* needed: if `ACTING` itself hangs (network never returns) or the iteration cap is hit — those need an explicit exit that bypasses the normal loop, because there's no observation to reason about yet.

---

## 4. Worked Example: Adding a Confirmation Gate as a New State

This is where explicit state machines earn their keep for production requirements — **adding a human-in-the-loop approval step (Day 11 preview) is just adding one new state**, not rewriting the loop:

```
┌───────────┐     tool call is         ┌──────────────┐
│  THINKING │ ─── side-effecting ─────▶│ AWAITING_     │
│           │     (e.g. send_email)    │ CONFIRMATION  │
└─────┬─────┘                          └──────┬─────────┘
      │ tool call is                          │ user approves
      │ read-only                             ▼
      ▼                                ┌────────────┐
┌───────────┐                          │  ACTING     │
│  ACTING    │◀─────────────────────────┘            │
└─────┬─────┘         user rejects → back to THINKING with rejection noted
      ▼
  (rest of loop unchanged)
```

The key insight for an interview: **because the architecture is already a named state machine, inserting a new required state (AWAITING_CONFIRMATION) is a local, well-scoped change** — you add one state, one new transition condition ("is this tool call side-effecting?" from Day 3 §5.3), and the rest of the machine is untouched. If your original implementation was an unstructured `while True` loop with tool execution inlined, adding this same gate means threading conditional logic through the whole function, much more error-prone and harder to review.

---

## 5. Production Considerations

### 5.1 The Iteration Cap Isn't Just "a number" — It's a State Transition Guard

Recall Day 2's runaway-loop mitigation ("max iteration cap"). Made explicit in the state machine: **every transition INTO THINKING checks a guard condition** — `if iteration_count >= MAX_ITERATIONS: transition to COMPLETE (best-effort answer) instead`. This is cleaner to reason about and test than a bare `for i in range(MAX_ITERATIONS)` wrapped around opaque logic, because the guard is attached to a specific, named transition, not buried in loop bookkeeping.

```python
MAX_ITERATIONS = 10

def transition_from_observing(state, iteration_count):
    if iteration_count >= MAX_ITERATIONS:
        return State.COMPLETE, generate_best_effort_answer(state)
    return State.THINKING, state
```

### 5.2 Persisting State for Long-Running or Interruptible Agents

For agents that might run for minutes/hours (deep research agents, multi-day workflows), or that need to survive a server restart mid-task, the state machine formalization becomes **essential**, not just nice-to-have: if state is just local Python variables inside a `while True` loop, a crash loses everything. If state is an explicit, serializable object (`{current_state: "ACTING", messages: [...], iteration_count: 4}`), you can:
- **Checkpoint it** to a database after every transition.
- **Resume** an interrupted agent by loading the last checkpoint and re-entering the state machine at the correct state — this is exactly what Day 19 (state/context management at scale) builds on.
- **Pause for asynchronous human input** (the AWAITING_CONFIRMATION state above) — the process can literally exit and restart later when the human responds, because state was persisted, not held in memory.

### 5.3 Observability Requires Named States

You cannot build a trace/dashboard (Day 15) that says "this agent spent 80% of its time in tool execution, 15% waiting on the model, 5% idle" unless your states are named and instrumented. This is the direct payoff of today's formalization for a topic three days from now — it's worth explicitly connecting this in an interview: *"I'd instrument each state transition with a timestamp and duration, which gives me per-state latency breakdowns for free, without needing separate ad hoc logging."*

### 5.4 Concurrency Within a Single Agent's Loop

Even within one "single agent," the ACTING state can itself fan out (recall Day 3 §5.2's parallel tool calls) — meaning ACTING isn't always a single atomic step, it can be "N parallel sub-actions, wait for all to complete, then transition to OBSERVING once as a batch." This is still a single-agent architecture (one LLM making decisions) — don't confuse this with multi-agent (Day 8), where *multiple independent LLM decision-makers* exist. The distinguishing question: **how many entities are making autonomous decisions?** One entity dispatching parallel tool calls = still single-agent. Multiple LLMs each with their own THINKING state, coordinating = multi-agent.

---

## 6. Interview Q&A

**Q1: Why model an agent as an explicit state machine instead of just a `while True` loop?**
A: An unstructured loop only supports black-box, end-to-end testing and makes it hard to attach guards, timeouts, logging, or new required steps (like a confirmation gate) without threading conditional logic through the whole function. An explicit state machine names each phase (THINKING, ACTING, OBSERVING, COMPLETE), so you can test transitions in isolation, attach instrumentation per state for observability, insert new states as local well-scoped changes, and persist/checkpoint state for long-running or resumable agents.

**Q2: In your state machine, does a failed tool call go to a special ERROR state?**
A: No — a failed tool call still transitions ACTING → OBSERVING normally, with the error captured as the observation's content, consistent with the "errors are observations, not exceptions" principle. A genuinely separate terminal/error exit is only needed when there's no observation to reason about at all — e.g., ACTING hangs indefinitely, or an iteration/cost cap is hit — those bypass the normal loop rather than flowing through it.

**Q3: How would you add a human-approval step before any side-effecting tool call, given this architecture?**
A: Add one new state, AWAITING_CONFIRMATION, with a transition guard on the THINKING → ACTING edge: if the proposed tool call is side-effecting (per Day 3's read/write tool distinction), route to AWAITING_CONFIRMATION instead of directly to ACTING; on approval, proceed to ACTING; on rejection, return to THINKING with the rejection noted as context. Because the architecture is already state-based, this is a local addition, not a rewrite of the control flow.

**Q4: Your agent needs to survive a server restart mid-task. What does the state machine formalization buy you here?**
A: If state is explicit and serializable (current state name, message history, iteration count) rather than held in local variables inside a loop, you can checkpoint it after every transition and resume by reloading the last checkpoint and re-entering the state machine at the correct state — a crash doesn't lose the task. This also naturally supports pausing for asynchronous human input, since the process can exit entirely and restart later from a persisted state.

**Q5: Is an agent that dispatches 5 parallel tool calls in one turn a "multi-agent" system?**
A: No — it's still a single agent, because there's still exactly one entity (one LLM, one THINKING state) making autonomous decisions; the parallel tool calls are just a fan-out within the ACTING state, waited on as a batch before transitioning to OBSERVING. Multi-agent specifically means multiple independent decision-making entities, each with their own reasoning loop, coordinating with each other — a distinction Day 8 covers in depth.

---

## 7. Summary Card

- Underneath the "reasoning" language, an agent is a **state machine**: named states (THINKING, ACTING, OBSERVING, COMPLETE) with explicit transition rules.
- Failed tool calls are **not** a special error state — they flow through OBSERVING normally, as content, per Day 3's principle.
- Explicit states are what make **guards** (iteration caps), **new required steps** (confirmation gates), **checkpointing/resumability**, and **observability** all tractable — an unstructured loop makes each of these a bolt-on hack.
- Parallel tool dispatch within ACTING is still single-agent; multi-agent means multiple independent decision-making loops (Day 8).

---
*Next: Day 8 — Multi-Agent Systems (orchestrator/worker, debate, hierarchical patterns).*
