# Agents Module 2 — Tool Use & Function Calling (Master Notes, Expanded)

## 0. Why tool use exists — the capability gap it fills

An LLM's weights encode a frozen snapshot of training-time knowledge, and it cannot natively: get current information (today's weather, a live stock price), do precise/reliable arithmetic on large numbers, execute code, or affect anything outside its own text output. **Tool use is the mechanism that lets the model delegate a sub-task to an external, deterministic system**, and incorporate that system's result back into its reasoning — trading "the model must know or compute this itself" for "the model must know *when* and *how* to ask something else to do it."

**Interview framing to have ready**: "Tool use doesn't make the model smarter in the sense of adding knowledge to its weights — it makes the model more *capable* by letting it defer to specialized, deterministic, up-to-date systems for exactly the sub-tasks LLMs are structurally bad at (precise math, live data, side effects), while the LLM's own strength (language understanding, reasoning about *what* to do) handles the orchestration."

---

## 1. Function-calling schema — how a tool is described to the model

### The core mechanism
The model is not given the tool's actual code — it's given a **structured, natural-language-adjacent description** of the tool: its name, a plain-English description of what it does, and its expected input parameters (name, type, description, whether required) — almost universally expressed as **JSON Schema**, since that's a well-established, machine-parseable format for describing structured data shapes.

**Concrete example schema** (the shape you should be able to sketch on a whiteboard):
```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "The city name, e.g. 'Boston'"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}
```

### What actually happens mechanically at inference time
The full list of available tool schemas is included as part of the model's input context (alongside the conversation/task), essentially as additional structured text the model conditions on. When the model decides a tool call is the right next action, it doesn't call anything itself — **it generates structured text (typically JSON) matching one of the provided schemas**, specifying which tool and what argument values. This generated JSON is then **parsed and executed by the surrounding system** (not the model) — the model's entire contribution is emitting the *intent* and *arguments* as structured text; a separate, deterministic piece of code (the "agent harness" or "orchestrator") is responsible for actually invoking the real function/API with those arguments.

**Why this distinction matters, stated explicitly**: "The model never directly executes code or calls an API — it only ever generates text. Tool use works because that generated text is *structured enough* (schema-constrained JSON) for a deterministic wrapper program to reliably parse and act on. This is a critical security/reliability point: you should never treat model-generated tool-call arguments as automatically safe to execute without validation, since the model can hallucinate malformed or unsafe arguments just like it can hallucinate any other text."

### Why structured output (JSON mode / constrained decoding) matters here
For tool calling to be reliable, the model's output needs to be **valid, parseable JSON matching the schema**, not just JSON-flavored prose. Production systems achieve this via **constrained decoding** — at each generation step, the decoding process (Module 6 of LLM Basics) restricts the next-token sampling to only tokens that would keep the output consistent with the target JSON schema's grammar (e.g., after an opening `{`, only tokens that could validly start a JSON key are allowed) — this is a decoding-time constraint layered on top of everything from LLM Basics Module 6 (temperature, top-p, etc. still apply *within* the allowed token set at each step), not a separate model capability. Some model providers instead rely purely on strong instruction-following from alignment training (Module 5 of LLM Basics) to produce well-formed JSON without hard constrained decoding — a real design tradeoff between reliability (constrained decoding guarantees valid syntax) and flexibility/latency overhead.

---

## 2. The tool-call execution loop, step by step

1. Agent framework sends the model: the task/conversation so far + the list of available tool schemas.
2. Model generates either (a) a direct text response (no tool needed), or (b) a structured tool-call request (tool name + arguments).
3. If (b): the framework parses the tool-call JSON, validates the arguments against the schema, and **executes the actual underlying function/API** — this step happens entirely outside the model, in regular deterministic code.
4. The tool's return value (or an error, if execution failed) is packaged as an **observation** and appended back into the conversation context, typically as a distinct message type (e.g., a "tool" or "function" role message, alongside "user"/"assistant" roles).
5. The updated context (now including the tool's result) is sent back to the model for the **next** generation — the model now reasons over the real tool output, not a guess.
6. Loop continues (more tool calls, or a final text answer) until a stopping condition (Module 1) is met.

### Numerical illustration of why this loop, not a single call, is necessary
Suppose a task requires calling a currency-conversion tool, then a tax-calculation tool using that converted amount, then formatting a final summary. **A single LLM call cannot do this** — it doesn't know the real conversion rate or tax result, only what it might guess/hallucinate from training data (likely wrong, and definitely not current). The loop is exactly what lets each step's *real* output become the *input* to the next step's reasoning — this is the same "why plain prompting breaks down for multi-step tasks" point from Module 1, now made concrete with an actual tool-dependency chain.

---

## 3. Parallel vs. sequential tool calls

### Sequential (the default, simplest case)
Each tool call's result is needed before the next call/decision can be made (as in the currency-then-tax example above) — the loop must proceed one call at a time, since step N+1 genuinely depends on step N's real output.

### Parallel tool calls
When multiple tool calls are **independent of each other** (e.g., "get the weather in Boston AND get the weather in Tokyo" — neither call's arguments or purpose depends on the other's result), a well-designed agent framework can dispatch both calls **simultaneously** rather than waiting for one to finish before starting the next — pure latency optimization, no change in what's logically being computed. Some modern model APIs support the model emitting **multiple tool-call requests in a single generation turn** specifically to enable this (the model signals "these N calls are all needed and independent," and the framework executes them concurrently, then feeds all N results back together before the next generation call).

**Interview-level distinction to state clearly**: "Whether calls can run in parallel is a property of the *task's dependency structure*, not something the model decides arbitrarily — the framework (or the model's own reasoning about the task) needs to correctly identify which calls are truly independent before parallelizing, since incorrectly parallelizing dependent calls would mean running a later step before the data it actually needs exists."

---

## 4. Error handling and retries — the part real production systems spend the most engineering effort on

### Why this matters more than it might seem
Tools fail for entirely mundane reasons unrelated to the model's reasoning quality — network timeouts, invalid arguments the model hallucinated, rate limits, the external API being temporarily down, or the tool returning a result in an unexpected format. A robust agent framework must handle these failures **without the whole task collapsing.**

### Common patterns
- **Error observation as feedback, not a crash**: rather than the framework crashing the whole loop on a tool error, the error message itself is packaged as an *observation* (same mechanism as a successful result) and fed back to the model — letting the model reason about the failure and decide whether to retry with corrected arguments, try a different tool, or report the failure to the user. This directly reuses the same loop mechanism from Section 2 — errors are just another kind of observation.
- **Argument validation before execution**: validate the model's generated arguments against the tool's schema (types, required fields, value ranges) *before* actually executing the tool — catching a hallucinated malformed argument early (e.g., the model passing a string where an integer was required) avoids wasting a real API call or, worse, causing an unintended side effect with bad data.
- **Bounded retries**: cap the number of retry attempts for a failing tool call (directly analogous to Module 1's max-iteration stopping condition) — an agent that keeps retrying a fundamentally broken call indefinitely is both a cost problem and a reliability problem.
- **Idempotency awareness for side-effecting tools**: for tools with real-world side effects (sending an email, making a purchase, deleting a file), naive automatic retries are dangerous — a retry after an ambiguous timeout (where you don't actually know if the first attempt succeeded or not) could cause the action to happen twice. Production systems handle this via idempotency keys (a unique identifier per logical action, so the underlying system can recognize and ignore a duplicate retry) or by requiring explicit confirmation before retrying side-effecting actions — a genuinely important practical point to raise if asked about deploying agents that take real-world actions, not just read-only/informational tools.

---

## 5. Side-by-side summary table (memorize this cold)

| | Sequential tool calls | Parallel tool calls |
|---|---|---|
| When used | Later call depends on an earlier call's real result | Calls are independent of each other |
| Latency | Sum of all call latencies | Roughly the max of individual call latencies |
| Risk if misapplied | N/A (default-safe) | Executing a dependent call before its real input exists |

| | Read-only tools (search, lookup) | Side-effecting tools (send email, purchase, delete) |
|---|---|---|
| Retry safety | Generally safe to retry freely | Requires idempotency keys or explicit confirmation before retry |
| Error handling stakes | Wasted call, minor cost | Risk of duplicate real-world action |

---

## 6. Quick-fire Q&A (self-test)

**Q: What does the model actually generate when it "calls a tool," and who actually executes the tool?**
A: The model only ever generates structured text (typically JSON matching a provided schema) specifying the tool name and arguments — it never executes anything itself. A separate, deterministic piece of surrounding code (the agent framework/orchestrator) parses that generated text and actually invokes the real function or API.

**Q: What is constrained decoding, and why does it matter specifically for tool calling?**
A: Constrained decoding restricts next-token sampling at each generation step to only tokens consistent with a target grammar/schema (e.g., valid JSON matching the tool's parameter schema) — it matters for tool calling because reliable execution requires the model's output to be syntactically valid and schema-conformant, not just JSON-flavored prose that a parser might fail on.

**Q: Give a concrete example of why a single LLM call can't replace a multi-step tool-use loop.**
A: A task requiring currency conversion followed by tax calculation on the converted amount — the model doesn't know the real, current conversion rate, so it cannot correctly compute the tax step in a single generation; the loop is required so the real tool output from step 1 becomes genuine input to step 2's reasoning.

**Q: What determines whether tool calls can be run in parallel, and what's the risk of getting this wrong?**
A: Whether the calls are independent of each other in the task's dependency structure — not an arbitrary choice. Incorrectly parallelizing calls that are actually dependent means executing a later step before the real data it needs actually exists, producing a wrong or nonsensical result.

**Q: Why is a tool-call error typically fed back to the model as an observation, rather than crashing the loop?**
A: Because it reuses the same feedback mechanism as a successful result — feeding the error back lets the model reason about the failure and decide how to respond (retry with corrected arguments, try a different tool, or report the failure), rather than the entire task collapsing on any single tool failure.

**Q: Why are side-effecting tools (like sending an email or making a purchase) more dangerous to naively retry than read-only tools?**
A: If a call times out ambiguously (unclear whether it actually succeeded), a naive retry risks performing the real-world action twice — read-only tools have no such risk since repeating a lookup causes no harm. Idempotency keys or explicit confirmation before retrying are the standard mitigations for side-effecting actions.

---
*End of Agents Module 2 (expanded). Next: Module 3 — Chain-of-Thought & Reasoning Prompting (standard CoT, self-consistency, least-to-most decomposition).*
