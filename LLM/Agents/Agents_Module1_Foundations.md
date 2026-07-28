# Agents Module 1 — Agent Foundations (Master Notes, Expanded)

## 0. What "agentic" actually means — the core distinction

A plain LLM call is: **input text in → output text out, once, done.** No matter how good the CoT reasoning inside that single generation is, it's still one shot — the model never gets to check its own work against the real world, gather new information mid-task, or take an action that changes state outside the conversation.

An **agent** wraps that single call in a **loop** that can run multiple times, where each iteration can:
1. **Perceive** — observe the current state (the original task, plus anything learned so far, plus results of any actions taken).
2. **Reason** — decide what to do next, using the LLM as the "brain" of this decision.
3. **Act** — actually do something that has an effect outside the model itself (call an API, run code, search the web, write a file).
4. **Observe** — get the real-world result of that action back, and feed it into the next iteration's perception step.

**The one-sentence definition to have ready**: "An agent is an LLM placed inside a perceive-reason-act-observe loop, where the loop continues until the task is solved or some stopping condition is hit — the key architectural addition over a single LLM call is the feedback loop with the environment."

### Why plain prompting breaks down for multi-step tasks — a concrete numerical example
Suppose a task requires 5 sequential steps, and the model's single-shot success probability for correctly reasoning through a task **without ever being able to verify intermediate results** is 90% per step, but errors from earlier steps don't get corrected because there's no observation/feedback:
```
P(all 5 steps correct, no correction possible) = 0.9^5 ≈ 0.59
```
Only ~59% end-to-end success — and this drops fast as step count grows (Module 3's chain-of-thought material revisits this exact compounding-error math). An agentic loop with observation/correction at each step doesn't eliminate individual-step error, but it lets the model **detect and recover from a wrong step** before compounding further (e.g., if step 3's tool call returns an error or an unexpected result, the agent can re-plan step 3 rather than blindly continuing with a broken foundation) — this recoverability, not perfect single-step accuracy, is the actual practical value agentic loops add over single-shot generation.

---

## 1. The agent-environment interaction cycle, formalized

This is directly borrowed from classical **reinforcement learning** framing (deliberately — most agent architectures are, at a conceptual level, applying RL-style agent/environment/action/observation vocabulary to LLMs, even when no actual RL training is happening):

- **State (s)**: everything the agent currently knows — the original task, conversation history, results of prior tool calls, any retrieved memory.
- **Action (a)**: what the agent decides to do — could be a tool call, a message to the user, or a "final answer" action that ends the loop.
- **Environment**: everything outside the LLM itself that actions affect and that generates observations — a code execution sandbox, a web search API, a file system, another agent.
- **Observation (o)**: what the environment returns after an action — a tool's return value, an error message, search results, a file's contents.
- **Policy**: in classical RL this is a learned function mapping state→action; in most current LLM agents, **the "policy" is just the LLM itself being prompted with the current state and asked to decide the next action** — there is usually no separate RL training loop updating weights based on task success (though RL-trained agent policies do exist as an active research direction; the vast majority of production agent systems today use a frozen, already-aligned LLM as the decision-maker, not one specifically RL-trained for the agent task itself).

**Important interview distinction to make explicit**: don't confuse "the agent loop resembles RL's state/action/observation framing" with "the agent was trained using RL" — most agents you'll be asked about (ReAct, ToT, tool-use systems) use a standard pretrained+RLHF'd chat model (Module 5 of LLM Basics) as a frozen decision-maker inside a scripted or lightly-orchestrated loop; the RL vocabulary is a conceptual framing tool, not necessarily a training methodology being used here.

---

## 2. The stopping condition — a detail interviewers specifically probe

An agent loop needs an explicit way to know **when to stop**. Common mechanisms:
- **Explicit "final answer" action**: the model itself emits a specific signal (e.g., a `finish` or `submit` tool call, or a specially-formatted "Final Answer:" string) indicating it believes the task is complete — the loop-controller code watches for this signal.
- **Max iteration count**: a hard cap on the number of loop iterations, purely as a safety/cost bound, regardless of whether the model believes it's done — critical in production, since a poorly-designed agent can otherwise loop indefinitely (e.g., repeatedly calling a tool that keeps returning an unhelpful result, with the model never recognizing it should give up or try a different approach).
- **External verification**: for some tasks (e.g., code that must pass a test suite, or a math problem with a checkable final answer), the environment itself can signal task completion independent of the model's own self-assessment — this is generally more reliable than trusting the model's own "I'm done" judgment, since models can be wrong about their own success (directly connects to Module 8's evaluation-of-agents material, and to LLM Basics Module 8's calibration discussion — a model's stated confidence that it succeeded is not the same as it actually having succeeded).

---

## 3. Why this matters architecturally — where the LLM's role narrows

A critical, often-overlooked point: in most agent frameworks, the **LLM itself is stateless between calls** — it has no persistent memory of its own across loop iterations by default. Everything the agent "remembers" (prior actions, prior observations, task history) has to be **explicitly re-included in the prompt text at every single loop iteration**, or retrieved from an external memory store (Module 6) and re-injected into context. This is a direct consequence of the LLM being a pure function of its input context (no hidden internal state persists across separate API calls) — **the agent's apparent "memory" and "persistence" across a multi-step task is entirely an engineering construct built around the LLM, not a property of the LLM itself.**

**Interview-ready framing**: "People sometimes talk about agents as if the model itself is remembering and planning across steps in some deep way — architecturally, that's not quite right. The LLM is called repeatedly, stateless each time, and the agent framework's job is to reconstruct exactly the right context (history, tool results, retrieved memory) at each call so the frozen, stateless model can make a good next-step decision as if it remembered everything. All the 'agentic' behavior comes from how well that context-reconstruction is engineered, not from any change to the underlying model."

---

## 4. Side-by-side: Single-shot LLM call vs. Agent

| | Single-shot LLM call | Agent |
|---|---|---|
| Number of model invocations | Exactly 1 | Multiple, in a loop |
| Can take real-world actions? | No (text out only, unless the caller manually acts on it afterward) | Yes, actions are part of the loop itself |
| Can correct mid-task errors? | No | Yes, via observation feedback |
| State across steps | N/A | Must be explicitly reconstructed in context at every call (LLM itself is stateless) |
| Stopping condition | Ends after one generation | Requires an explicit mechanism (final-answer signal, max iterations, external verification) |
| Underlying "decision-maker" | The LLM's single forward generation | Usually still just the same frozen LLM, called repeatedly inside a loop |

---

## 5. Quick-fire Q&A (self-test)

**Q: Give the one-sentence definition of an agent that distinguishes it from a plain LLM call.**
A: An agent places the LLM inside a perceive-reason-act-observe loop with real-world actions and feedback, continuing until a stopping condition, whereas a plain LLM call is a single generation with no ability to act, observe results, or correct mid-task.

**Q: Why does an agentic loop help with multi-step task success rate, if it doesn't actually make the model more accurate on any individual step?**
A: Because it enables error *recovery* — a wrong intermediate step (e.g., a failed tool call or unexpected result) can be detected via observation and corrected before compounding into further errors, rather than being blindly built upon as in single-shot generation with no feedback.

**Q: What's the important distinction between "the agent loop resembles RL" and "the agent was RL-trained"?**
A: The state/action/observation vocabulary is borrowed conceptually from RL to describe the loop structure, but most production agents use a standard pretrained+RLHF'd chat model as a frozen decision-maker inside a scripted loop — there's typically no RL training happening specifically for the agentic task itself, even though RL-trained agent policies are an active research direction.

**Q: Name three mechanisms for an agent to know when to stop, and which is generally most reliable and why.**
A: An explicit final-answer signal from the model itself, a hard max-iteration cap (safety/cost bound), and external/environment-based verification (e.g., passing a test suite). External verification is generally most reliable because it doesn't rely on the model's own self-assessment of success, which can be miscalibrated — the model believing it succeeded isn't the same as it having actually succeeded.

**Q: Why is it inaccurate to say the LLM itself "remembers" prior steps across an agent's loop iterations?**
A: The LLM is a stateless function of its input context — it has no persistent internal state across separate calls. Any apparent memory of prior actions/observations is an engineering construct: the agent framework must explicitly reconstruct and re-inject the relevant history into the prompt (or retrieve it from external memory) at every single loop iteration for the frozen model to act as if it remembers.

---
*End of Agents Module 1. Next: Module 2 — Tool Use & Function Calling (schema design, the execution loop, parallel vs sequential calls, structured output).*
