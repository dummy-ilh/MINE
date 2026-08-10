# Practical Library Walkthrough: LangGraph — Every Function Explained

Picking LangGraph because it maps directly onto Day 7's state machine and Day 11's HITL gates — you already know the concepts, this is just seeing them as real, runnable API calls. Each function below: what it does, why it exists, and which Day-N concept it implements.

---

## 1. Setup

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
import operator
```

- `StateGraph` — the class you instantiate to build your agent's state machine (Day 7). This IS the THINKING/ACTING/OBSERVING diagram, made into actual code.
- `END` — a special sentinel node meaning "terminate the graph here" — the COMPLETE state from Day 7.
- `MemorySaver` — a checkpointer implementation (Day 7 §5.2, Day 19's checkpointing) that persists graph state in memory (swap for a database-backed one in real production).

---

## 2. Defining State — The Shared Object Passed Between Nodes

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # conversation history
    iteration_count: int
    remaining_budget: int
```

**What this does**: defines the SCHEMA of the state object that flows through every node — this is literally Day 7's "explicit, serializable state" made concrete. Every node function receives this dict and returns updates to it.

**`Annotated[list, operator.add]`** — this is the part worth understanding precisely: it tells LangGraph HOW to merge a node's returned value into the existing state for that key. `operator.add` means "when a node returns a new `messages` list, APPEND it to the existing one" (list concatenation) rather than overwrite it. Without this annotation, LangGraph's default behavior is to REPLACE the field entirely — which would silently wipe your conversation history every time any node touched `messages`. This single line is the most commonly misunderstood part of the library, and it's the direct mechanism behind Day 5's "context accumulates across the loop."

---

## 3. Defining Nodes — Each One Is a Day 7 State

```python
def thinking_node(state: AgentState) -> dict:
    """Corresponds to Day 7's THINKING state."""
    response = llm_call(state["messages"], tools=available_tools)
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1
    }

def acting_node(state: AgentState) -> dict:
    """Corresponds to Day 7's ACTING state."""
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    result = execute_tool(tool_call.name, tool_call.arguments)   # Day 3's execution layer
    return {"messages": [tool_result_message(tool_call.id, result)]}
```

**What each node function does, mechanically**: takes the CURRENT state dict, does its work (an LLM call, a tool execution — whatever that state represents), and returns a PARTIAL dict of updates. LangGraph merges this partial dict into the full state (using the merge rules from §2) before passing the updated state to whichever node runs next. **This is the direct, literal implementation of Day 7's "named states, each doing one well-scoped job."**

---

## 4. Building the Graph — Wiring States Together

```python
graph = StateGraph(AgentState)

graph.add_node("thinking", thinking_node)
graph.add_node("acting", acting_node)

graph.set_entry_point("thinking")
```

- **`StateGraph(AgentState)`** — instantiates the graph, telling it what schema (§2) every node will read/write.
- **`add_node(name, function)`** — registers a node under a string name; this name is how you reference it in edges below. Each `add_node` call is adding one box to Day 7's diagram.
- **`set_entry_point("thinking")`** — declares which node runs FIRST when the graph is invoked — Day 7's initial transition into THINKING.

---

## 5. Edges — The Transition Rules

### 5.1 Unconditional edge

```python
graph.add_edge("acting", "observing")
```
**What it does**: after `acting` finishes, ALWAYS go to `observing` next, no branching logic. Direct implementation of a fixed Day 7 transition arrow (ACTING → OBSERVING always happens, regardless of outcome — recall Day 7 §3.1: even a failed tool call flows through this same edge, since the error becomes the observation's content).

### 5.2 Conditional edge — Where the Model's Decision Actually Branches the Graph

```python
def route_after_thinking(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "acting"
    else:
        return END

graph.add_conditional_edges(
    "thinking",
    route_after_thinking,
    {"acting": "acting", END: END}
)
```

**What this does, precisely**: `add_conditional_edges` takes three things — the SOURCE node (`"thinking"`), a ROUTING FUNCTION (`route_after_thinking`) that inspects the current state and returns a string label, and a MAPPING from that label to the actual next node. This is the literal code for Day 7's diagram branch: *"if the model's response has a tool call, go to ACTING; if not, go to COMPLETE."* The routing function is pure Python — you're not asking the LLM to decide the graph structure, you're inspecting what the LLM already decided (via `tool_calls` being present or not) and translating that into a graph transition. **This is precisely the "code controls the transition based on the model's decision" split from Day 3** — the model decided WHAT action to take; your routing function decides WHERE that leads in the graph.

### 5.3 The loop-back edge

```python
graph.add_edge("observing", "thinking")
```
**What it does**: closes the loop — after OBSERVING, go back to THINKING. This single line, combined with §5.2's conditional edge, is the entire ReAct loop (Day 2) expressed as graph structure: THINKING routes to either ACTING (continue looping) or END (done), and ACTING always flows through OBSERVING back to THINKING.

---

## 6. Compiling — Turning the Definition Into a Runnable Object

```python
app = graph.compile(checkpointer=MemorySaver())
```

**What `compile()` does**: validates the graph (checks all referenced nodes exist, there's a valid entry point, no orphaned nodes) and returns an executable object. **The `checkpointer` argument is the direct implementation of Day 7 §5.2 / Day 19's checkpointing discussion** — passing one here means LangGraph automatically persists the full state after every node execution, keyed by a thread ID (below), without you writing any manual `save_state()` calls. This is the single biggest reason to reach for LangGraph over a raw loop, per Day 12's comparison: this exact capability is otherwise something you'd hand-build.

---

## 7. Running the Graph

```python
config = {"configurable": {"thread_id": "conversation-42"}}

result = app.invoke(
    {"messages": [user_message], "iteration_count": 0, "remaining_budget": 50000},
    config=config
)
```

- **`app.invoke(initial_state, config)`** — runs the graph from the entry point, following edges/nodes until it hits `END`, and returns the FINAL state. This single call is doing everything Days 2-3's `while True` loop did manually — the entry point starts at THINKING, and the graph mechanically follows your conditional edges (§5.2) until routing hits `END`.
- **`thread_id`** — this is what the checkpointer uses to know WHICH conversation's state to save/restore. Different `thread_id` values are completely independent conversations with independent checkpointed histories — this is the mechanism that makes Day 19's "resume after a server restart" concrete: call `invoke` again with the SAME `thread_id`, and LangGraph automatically loads the last checkpoint rather than starting from scratch.

**Streaming variant** — for watching state transitions as they happen (feeding Day 15's real-time observability, and Day 11's interrupt-visibility requirement):
```python
for event in app.stream(initial_state, config=config):
    print(event)   # yields the state update after EACH node, not just the final result
```
`stream()` does the same graph execution as `invoke()`, but yields an event after every single node completes, rather than blocking until the whole graph finishes — this is what a live dashboard (Day 15) or a human watching for an interrupt opportunity (Day 11 §5) would actually consume.

---

## 8. Human-in-the-Loop — `interrupt_before` / `interrupt_after`

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["acting"]   # pause BEFORE this node runs
)
```

**What this does**: tells the graph to STOP and return control to your code right before the `acting` node would execute, rather than running it automatically. This is the direct, built-in implementation of Day 11's `AWAITING_CONFIRMATION` state — instead of you manually coding a new node and conditional edge for the approval gate (Day 7 §4's manual version), LangGraph gives you this as a compile-time flag on an EXISTING node.

**Resuming after the interrupt**:
```python
# First call — runs until it hits the interrupt, then returns
result = app.invoke(initial_state, config=config)
# result now reflects state AS OF right before "acting" — nothing has executed yet

# ... human reviews, decides to approve ...

# Resume by invoking again with the SAME thread_id and NO new input (None) —
# this tells LangGraph "continue from where you paused"
result = app.invoke(None, config=config)
```
**Why passing `None` resumes rather than restarts**: because the checkpointer (§6) already has the paused state saved under this `thread_id` — invoking with `None` as input means "don't add new input, just continue execution from the last checkpoint," which is exactly Day 11 §6.2's requirement that an approval gate needs to survive an arbitrarily long wait without losing state, since the wait could be milliseconds (an automated check) or hours (an actual human).

**If the human wants to modify state before resuming** (e.g., Day 11 §3.1's "approved, but change the email tone" example):
```python
app.update_state(config, {"messages": [human_feedback_message]})
result = app.invoke(None, config=config)
```
`update_state()` lets you inject new information into the checkpointed state BEFORE resuming — this is the literal mechanism for feeding the human's modification back in as new context (Day 11 §3.1's point that approval isn't just binary), rather than the graph blindly re-running the exact same proposed action.

---

## 9. Multi-Agent — Nesting a Compiled Graph as a Node

```python
worker_graph = build_worker_graph().compile()   # a fully separate StateGraph, its own THINKING/ACTING/OBSERVING

def orchestrator_dispatch_node(state: OrchestratorState) -> dict:
    """The orchestrator's ACTING state — dispatching to a worker."""
    worker_result = worker_graph.invoke({"messages": [state["subtask"]]})
    return {"worker_summaries": [worker_result["final_summary"]]}

orchestrator_graph.add_node("dispatch_to_worker", orchestrator_dispatch_node)
```

**What this does**: a compiled graph (`worker_graph`) is just a callable — you can invoke it FROM INSIDE another graph's node function. This is the literal code-level realization of Day 8's "the orchestrator's ACTING state means dispatch to a worker" and Day 7 §5.4's "the orchestrator's own state machine treats worker dispatch as one action" — there's no special "multi-agent mode" in LangGraph, it's just graphs calling other compiled graphs as a regular function call inside a node.

**For running workers in parallel** (Day 8's parallelism benefit, Day 3 §5.2's parallel-call pattern): dispatch multiple worker `.invoke()` calls using `asyncio.gather` (or LangGraph's built-in fan-out via multiple edges from one node, depending on version) rather than sequential calls — the mechanism is the same async/concurrent-execution principle from Day 3, just applied to whole sub-graphs instead of individual tool calls.

---

## 10. Full Picture — What Maps to What

| LangGraph API | Curriculum concept |
|---|---|
| `StateGraph(Schema)` | Day 7's state machine, instantiated |
| `add_node(name, fn)` | One named state (THINKING/ACTING/OBSERVING) |
| `add_edge(a, b)` | An unconditional transition arrow |
| `add_conditional_edges(...)` | The branch where the model's decision routes the graph (Day 3's "code decides where, model decided what") |
| `Annotated[list, operator.add]` | How new observations ACCUMULATE into context (Day 5's short-term memory) |
| `compile(checkpointer=...)` | Day 7 §5.2 / Day 19's checkpointing, given for free |
| `thread_id` | Which conversation's state to save/resume (Day 19's resumability) |
| `interrupt_before=[...]` | Day 11's AWAITING_CONFIRMATION approval gate, built-in |
| `update_state()` | Feeding human feedback into context before resuming (Day 11 §3.1) |
| `stream()` | Real-time visibility for observability (Day 15) and interrupts (Day 11 §5) |
| nested `compiled_graph.invoke()` inside a node | Multi-agent orchestrator/worker dispatch (Day 8) |

**The single takeaway**: nothing in this library is new machinery — every function is a direct, named implementation of a concept you already derived from first principles across Days 1-19. Learning the API is just learning WHERE each concept you already understand lives in this specific library's naming conventions.
