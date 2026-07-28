# Agents Module 7 — Multi-Agent Systems & Orchestration (Master Notes, Expanded)

## 0. Why use multiple agents at all — the motivating problem

A single agent (Modules 1-6) running one long Thought-Action-Observation loop has to hold the **entire task's context** in its working memory at once, and must be simultaneously good at every sub-skill the task requires (planning, specific domain reasoning, writing code, critiquing its own output, etc.) using one generic prompt/role. Multi-agent systems split a complex task across **multiple agent instances, each with a narrower role, own context window, and possibly a different prompt/persona (or even a different underlying model)** — trading added coordination complexity for better per-role specialization and smaller, more focused context per agent.

**Interview framing to have ready**: "Multi-agent systems aren't a fundamentally different capability — they're an orchestration pattern for splitting one large context/task into multiple smaller ones, each handled by a separately-prompted (and possibly separately-modeled) agent instance, with an explicit mechanism for them to communicate. The underlying LLM call mechanism is identical to a single agent; what's different is how many separate loops are running and how their contexts get combined."

---

## 1. Common Multi-Agent Architectures

### Orchestrator-Worker (a.k.a. Manager-Worker / Hierarchical)
A single **orchestrator agent** breaks the overall task into subtasks, dispatches each subtask to a specialized **worker agent** (each worker often has a narrower role/prompt, e.g. "research worker," "code-writing worker," "summarization worker"), collects their results, and either synthesizes a final answer itself or dispatches further subtasks based on what the workers returned.

**Direct structural analogy**: this is architecturally very similar to Module 2's tool-use pattern, just with each "tool" being an entire other agent (with its own reasoning loop) instead of a simple deterministic function — the orchestrator's Action is "dispatch subtask to worker X," and the "Observation" it receives back is that worker's full completed output, not a simple API return value.

### Debate / Adversarial
Multiple agents independently attempt the same task (or take explicitly opposing positions on a question), then **see each other's responses and are prompted to critique, defend, or revise their own position** across one or more rounds, with a final answer determined either by consensus, a separate judge agent, or majority agreement. The motivating idea: an agent explicitly forced to defend its reasoning against a genuine counter-argument (rather than just self-reflecting in isolation, as in Module 5's Reflexion) may surface errors that a single agent's own self-critique would miss, because the counter-argument comes from a genuinely different context/sampling path, not the same agent re-examining its own already-anchored reasoning.

### Sequential Pipeline
Agents are arranged in a fixed sequence, each one's output becoming the next one's input — no back-and-forth communication, no orchestrator dynamically deciding routing; just a fixed hand-off chain (e.g., "research agent" → "drafting agent" → "editing agent" → "fact-checking agent"). Simplest to reason about and debug, but the least flexible — if a later-stage agent discovers the earlier stage's output was actually flawed, there's no built-in mechanism to loop back and re-invoke an earlier stage (unless explicitly engineered in as a special case).

---

## 2. Communication Protocols Between Agents

### The core mechanism — still just text in context
Regardless of architecture, agent-to-agent "communication" ultimately means: **one agent's generated text output becomes part of another agent's input context** — there's no separate, richer channel; it's the same context-construction mechanism from Module 1/6, just with the source of the injected text being another agent's output rather than a tool's return value or a retrieved memory.

### Structured vs. free-form communication
- **Free-form**: agents pass natural-language messages to each other directly (e.g., a worker's full natural-language explanation gets included verbatim in the orchestrator's next context) — flexible, but can be verbose/ambiguous, and the receiving agent has to correctly interpret unstructured prose.
- **Structured**: agents communicate via a defined schema (similar to Module 2's function-calling JSON schemas) — e.g., a worker must return `{"status": "success", "result": ..., "confidence": ...}` rather than free-flowing prose — more reliable for downstream automated processing/routing decisions (the orchestrator can programmatically check `status` and `confidence` fields rather than needing to re-parse/interpret prose), at the cost of losing some nuance a free-form explanation might have conveyed.

### Shared vs. private context
A key design decision: does every agent see the **full transcript of everyone else's reasoning** (shared/global context — maximizes information available to each agent, but grows context length quickly and can dilute a worker's focus with irrelevant other-agent chatter), or does each agent only see a **filtered/summarized subset** relevant to its specific role (private/local context — keeps each agent's context focused and smaller, but risks losing information one agent had that another genuinely needed and didn't receive) — directly analogous to the working-memory-size tradeoff from Module 6, now playing out across multiple agents instead of within one agent's growing single transcript.

---

## 3. Common Failure Modes

### Compounding errors across agents
If Agent A produces a subtly wrong intermediate result and Agent B trusts it uncritically (rather than verifying it), the error propagates and can compound further through Agent C, and so on — this is the multi-agent version of Module 1's single-agent compounding-error math, except now the "steps" are entire other agents' outputs rather than individual reasoning steps within one agent's loop. **A worker agent's confident, fluent-sounding wrong answer is just as capable of misleading a downstream orchestrator/agent as a human's confidently wrong statement would be** — nothing about inter-agent communication makes agents inherently better at catching each other's mistakes unless the architecture specifically builds in verification/critique (as debate architectures explicitly attempt to do).

### Coordination overhead
More agents means more total LLM calls, more context to construct/pass around, and more points where miscommunication (a worker misunderstanding what the orchestrator actually wanted, or an orchestrator misinterpreting a worker's structured-or-not output) can derail the task — the coordination machinery itself has a real cost and failure surface, not just the underlying reasoning tasks each agent performs individually. A concrete practical question worth having an answer ready for: "when does splitting into multiple agents actually help vs. just adding overhead?" — the honest answer is roughly: multi-agent architectures pay off when subtasks genuinely benefit from **different specialized prompts/roles/contexts** that would conflict or dilute each other if crammed into one agent's single context/persona, and the added coordination cost is justified by that specialization gain; for tasks that are naturally linear/simple enough for one well-designed single-agent ReAct loop, adding multiple agents is often pure overhead with no compensating benefit.

### Redundant/wasted work
Without careful task decomposition, multiple agents can end up **duplicating effort** (two workers independently researching the same sub-question because the orchestrator's task-splitting wasn't clean/non-overlapping) — a pure orchestration-design failure, not a reasoning failure of any individual agent.

### Deadlock / non-termination in interactive architectures
In debate or iterative-critique architectures specifically, agents can get stuck in an unproductive back-and-forth (each repeatedly restating a position without genuine convergence) — directly analogous to Module 4's ReAct "loop divergence" failure mode, requiring the same kind of mitigation (a max-rounds cap, explicit detection of repeated/non-converging exchanges, or a designated tie-breaking judge agent to force resolution rather than relying on the debating agents to naturally converge).

---

## 4. Side-by-side summary table (memorize this cold)

| | Orchestrator-Worker | Debate/Adversarial | Sequential Pipeline |
|---|---|---|---|
| Structure | Hierarchical — orchestrator dispatches and synthesizes | Multiple agents critique/challenge each other | Fixed linear hand-off chain |
| Best suited for | Tasks decomposable into distinct, largely-independent subtasks | Tasks where surfacing errors via genuine counter-argument adds value | Tasks with a natural, fixed stage-by-stage structure |
| Flexibility to loop back on discovered errors | Yes, if orchestrator is designed to re-dispatch | Yes, inherently (multiple rounds) | No, unless explicitly engineered as a special case |
| Main failure risk | Orchestrator misinterpreting worker outputs, redundant dispatch | Deadlock / non-convergence | Compounding errors with no recovery path |

---

## 5. Quick-fire Q&A (self-test)

**Q: What's the fundamental mechanism underlying "communication" between agents, regardless of architecture?**
A: One agent's generated text output becomes part of another agent's input context — there's no separate richer channel; it's the same context-construction mechanism used for tool observations or retrieved memory, just with another agent's output as the source text.

**Q: How is the Orchestrator-Worker pattern structurally similar to Module 2's tool-use mechanism?**
A: The orchestrator's dispatch-to-worker action is analogous to a tool call, and the worker's completed response returned to the orchestrator is analogous to a tool's observation/return value — the difference is that the "tool" here is itself a full agent with its own reasoning loop, rather than a simple deterministic function.

**Q: What's the core tradeoff between shared/global context and private/filtered context across agents?**
A: Shared context maximizes information available to every agent but grows context length quickly and can dilute focus with irrelevant other-agent content; private/filtered context keeps each agent's context smaller and focused but risks losing information one agent had that another genuinely needed and didn't receive — directly analogous to the single-agent working-memory-size tradeoff, now applied across agents.

**Q: Why doesn't inter-agent communication automatically make a multi-agent system better at catching individual agents' mistakes?**
A: A worker's confidently wrong output is just as capable of misleading a downstream agent as any other agent's confident wrong output — nothing about passing text between agents inherently adds verification. Catching mistakes requires the architecture to specifically build in critique/verification (as debate architectures explicitly attempt), not just multi-agent communication itself.

**Q: Give the honest decision criterion for when multi-agent architectures are worth their coordination overhead vs. just using one well-designed single agent.**
A: They pay off when subtasks genuinely benefit from different, specialized prompts/roles/contexts that would conflict or dilute each other if combined into a single agent's context/persona, and that specialization gain outweighs the added coordination cost — for naturally linear or simple-enough tasks, a single well-designed ReAct loop is often both cheaper and sufficient, and multi-agent overhead is largely wasted.

**Q: What failure mode is specific to debate/adversarial architectures, and what's a direct mitigation?**
A: Deadlock/non-convergence — agents repeatedly restating positions without genuine resolution, analogous to Module 4's ReAct loop-divergence failure. Mitigations include a max-rounds cap, explicit detection of non-converging repeated exchanges, or a designated judge agent to force a final resolution rather than relying on the debating agents to naturally converge.

---
*End of Agents Module 7 (expanded). Next: Module 8 — Evaluating Agents (why it's harder than single-turn eval, task success rate, step efficiency, tool-call accuracy, benchmark suites).*
