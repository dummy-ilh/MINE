# Day 1: What Makes Something "Agentic"?

## 1. The Intuition First

Before any formalism — think about the difference between a **vending machine** and a **personal assistant**.

- **Vending machine**: You press B4. It always dispenses the item in slot B4. Fixed input → fixed output. No matter what's happening around it, it does the exact same mechanical thing every time.
- **Personal assistant**: You say "get me a coffee." The assistant decides *how* — checks if the coffee shop is open, picks a route, notices it's raining and grabs an umbrella, adapts if the shop is closed by going to a different one. The **goal** is fixed, but the **path to achieve it is decided by the assistant, in real time, based on what it observes**.

That's the entire difference between a **workflow** and an **agent**:

> **Workflow** = the *path* is decided by the engineer, ahead of time, in code.
> **Agent** = the *path* is decided by the model, at run-time, based on what it observes.

Everything else in agentic development is commentary on this one idea.

---

## 2. Formalizing It

### 2.1 The Spectrum, Not a Binary

Most real systems aren't purely "workflow" or purely "agent" — they sit on a spectrum of **how much control is handed to the LLM**.

| Level | Name | Who decides the next step? | Example |
|---|---|---|---|
| 0 | Static pipeline | Engineer (100% fixed code) | `extract_text() → summarize() → save()` |
| 1 | LLM call inside a pipeline | Engineer decides *when* to call the LLM; LLM only fills in content | Fixed prompt-chain: summarize → translate → classify |
| 2 | LLM chooses between fixed branches | Engineer defines branches; LLM picks which one | Router: "is this a billing or technical question?" → route to one of 2 fixed chains |
| 3 | LLM chooses tools + order dynamically, loops until done | LLM decides sequence, can call same tool multiple times, decides when to stop | ReAct agent solving a multi-step research question |
| 4 | Multi-agent, LLM spawns/delegates to other LLMs | LLM decides not just actions but *who* should act | Orchestrator agent that creates sub-agents for sub-tasks |

**Interview framing**: When asked "is this agentic?", don't answer yes/no. Answer: *"It's agentic to the degree that control flow is determined by the model rather than by code. This looks like level 2/3 because..."* — this signals systems maturity.

### 2.2 The Formal Definition (what most papers/interviewers use)

An **agent** is a system where an LLM operates in a loop:

```
observe(state) → LLM decides action → execute action → new observation → repeat until goal met or stop condition
```

The three necessary ingredients:
1. **Autonomy over the next step** — the LLM, not the code, picks the next action from a set of possibilities.
2. **Environment interaction** — the agent can *act* on something outside itself (call an API, run code, search, write a file) and *observe* the result.
3. **Persistence across steps** — there is a loop, i.e., more than one LLM call chained by the *results of the model's own decisions*, not by fixed code logic.

If a system is missing any of these three, it's a workflow, not an agent — no matter how "smart" the single LLM call is.

---

## 3. Worked Example #1: Same Task, Two Implementations

**Task**: "Find the current weather in the user's city and tell them if they need an umbrella."

### Implementation A — Workflow (Level 1)
```python
def handle_request(city):
    weather = call_weather_api(city)                     # fixed step 1
    prompt = f"Weather is {weather}. Write a friendly reply about whether to bring an umbrella."
    reply = llm_call(prompt)                              # fixed step 2
    return reply
```
- The LLM never decides *to* call the weather API. The engineer already decided that step exists and runs first, always.
- The LLM's only job: turn structured data into friendly text.
- **Zero autonomy.** This is NOT an agent, even though it uses an LLM.

### Implementation B — Agent (Level 3)
```python
tools = [get_weather(city), search_web(query), get_user_location()]

messages = [{"role": "user", "content": "Do I need an umbrella today?"}]

while True:
    response = llm_call(messages, tools=tools)
    if response.tool_calls:
        for call in response.tool_calls:
            result = execute(call)                        # e.g. get_user_location() -> "Austin"
            messages.append(tool_result(call, result))
    else:
        return response.content                            # LLM decided it's done, gives final answer
```
- Here the LLM sees "Do I need an umbrella today?" and has to **decide**: *I don't know the city yet → call get_user_location() → now I know it's Austin → call get_weather("Austin") → now I have data → I can answer.*
- Every one of those decisions — which tool, in what order, when to stop — was made by the model at run-time, not hardcoded.
- **This is agentic**, per all 3 ingredients: autonomy (chose the tool sequence), environment interaction (tool calls), persistence (loop driven by its own outputs).

**The interview-ready one-liner**: *"Implementation A calls an LLM. Implementation B is an LLM that calls things."* Notice who's the subject of the sentence — that's the tell.

---

## 4. Worked Example #2: The Ambiguous Case (this is what interviewers actually probe)

**Task**: A support bot classifies a ticket as "billing" or "technical," then always calls a corresponding fixed sub-chain.

Is this agentic?

- The LLM *chooses* between 2 paths → looks agentic (Level 2).
- But: there's no loop, no tool use beyond classification, and it never revisits its own decision. It's a **router**, not an agent.
- **Correct interview answer**: "This is a conditional workflow with an LLM-based router. It exhibits *decision-making* but not *autonomy over an open-ended action space*, and there's no loop — so I wouldn't call it a full agent. It's Level 2 on the spectrum: useful, cheaper, more predictable, but not what most interviewers mean by 'agent.'"

This distinction matters immensely in system design interviews — **naming the level precisely, instead of saying "yes it's an agent," is what separates L4 from L5+ answers.**

---

## 5. Why This Distinction Matters in Production (not just semantics)

This isn't pedantry — the level of "agentic-ness" you choose is a **direct engineering tradeoff**, and knowing this tradeoff cold is what gets asked in interviews.

| Property | Low agentic (workflow) | High agentic (autonomous loop) |
|---|---|---|
| **Predictability** | High — same input, same path, every time | Low — model may take a different path each run |
| **Latency** | Low — fixed number of LLM calls | Variable, often high — loop can run N steps |
| **Cost** | Predictable, bounded | Unbounded unless you cap steps (real incidents: agents looping 40+ times burning $100s) |
| **Debuggability** | Easy — deterministic trace | Hard — need full observability/tracing infra |
| **Flexibility to novel inputs** | Low — breaks on cases outside the fixed paths | High — model improvises for unseen scenarios |
| **Failure mode** | Wrong output for edge cases | Can spiral: wrong action → wrong observation → compounding wrong actions |

**The production rule of thumb** (this is the single most interview-relevant takeaway of Day 1):

> **Use the least agentic design that solves the problem.** Escalate up the spectrum (workflow → router → bounded loop → full autonomous agent) only when the task's branching is too combinatorially large to hand-code.

A staff-level answer to "would you build this as an agent?" almost always starts with: *"What's the actual variability in the task? If I can enumerate the paths, I hardcode them — more reliable, cheaper, easier to eval. I only reach for an autonomous loop when the action sequence genuinely can't be predicted ahead of time."*

---

## 6. Interview Q&A

**Q1: What's the difference between a chain and an agent?**
A: A chain is a fixed sequence of LLM/tool calls determined by the developer at write-time — same steps every run. An agent has the LLM decide, at run-time, which action to take next based on the current state, typically inside a loop that continues until the model itself decides it's done. The defining test: *does the model choose the control flow, or does the code?*

**Q2: Give an example of a system that looks agentic but isn't, and explain why.**
A: A ticket router that picks between 2 fixed sub-chains based on LLM classification. It has decision-making but no loop and no open-ended action space — it's a conditional workflow (Level 2), not an autonomous agent (Level 3+). The tell: it never revisits or corrects its own decision, and the action space is a discrete, enumerable branch, not open-ended tool composition.

**Q3: Why wouldn't you always build the most agentic version of a system?**
A: Cost and latency are unbounded/variable in a loop-based agent (each step = an LLM call, and a bad loop can run dozens of times). Predictability and debuggability drop sharply — the same input can produce a different trajectory each run, which is hard to test and hard to guarantee correctness for. Production rule: use the least agentic solution that satisfies the task's actual branching complexity, and only escalate when hardcoding paths becomes intractable.

**Q4: What are the 3 necessary ingredients for a system to be called "agentic"?**
A: (1) Autonomy — the model picks the next action from a non-trivial action space, not the code. (2) Environment interaction — the model can act on and observe something outside itself (tool calls, code execution, search). (3) Persistence — a loop where subsequent steps are driven by the model's own prior outputs, not fixed control flow.

**Q5: A candidate says "my RAG system is an agent because it retrieves documents based on the query." Is that accurate?**
A: Generally no, if retrieval is a single fixed step (retrieve → generate) triggered unconditionally — that's Level 1, an LLM call inside a pipeline, not an agent. It becomes "agentic RAG" only if the model can *decide* whether to retrieve, reformulate the query, retrieve again, or stop — i.e., retrieval becomes a tool the model chooses to invoke inside a loop, not a hardcoded first step. (This exact distinction is Day 9.)

---

## 7. Summary Card

- **Agent = model controls the control flow.** Workflow = code controls the control flow.
- Think in **levels (0-4)**, not a binary yes/no — this is the interview-signal move.
- The 3 tests: autonomy, environment interaction, persistence via a loop.
- Production tradeoff: agentic-ness buys flexibility, costs predictability/latency/cost-boundedness.
- Golden rule: **least agentic design that solves the problem.**

---
*Next: Day 2 — The ReAct Pattern (Reasoning + Acting), with a full worked trace of an agent solving a multi-hop question step by step.*
