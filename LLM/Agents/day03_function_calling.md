# Day 3: Function Calling / Tool Use Mechanics

## 1. The Intuition First

Yesterday we treated `Action: search("...")` as if it just magically happens. Today we open the hood: **how does a language model — a thing that only outputs text — actually "call a function"?**

Here's the key mental unlock: **the model never executes anything.** It is a text generator. All it ever does is produce a specific, structured piece of *text* that *looks like* a function call — something like `{"name": "search", "arguments": {"query": "OpenAI CEO"}}`. That's it. That's the entire "action."

Everything else — actually running Python code, actually hitting an API, actually returning a result — happens in **your code**, sitting outside the model, which:
1. notices the model produced that structured text instead of a normal reply,
2. parses it,
3. actually executes the real function,
4. takes the real result and stuffs it back into the conversation as if it were a new message,
5. calls the model again with that appended context.

So "tool use" is really: **a text-generation contract + an execution loop you write.** The model's job is to be very good at producing correctly-formatted call requests. Your job is to actually run them safely.

---

## 2. Formalizing It

### 2.1 The Full Round Trip

```
1. You send: [system prompt, user message, tool schemas]
2. Model returns: either (a) normal text, or (b) a "tool_use" block: {name, arguments}
3. Your code: if (b), execute the real function with those arguments
4. Your code: append a "tool_result" message with the output
5. You send the updated message list back to the model
6. Repeat from step 2 until model returns (a) normal text
```

This is exactly the ReAct loop from Day 2, but now we're looking at what's inside the "Action" box mechanically.

### 2.2 The Tool Schema — What the Model Actually Sees

The model doesn't "know" your Python function exists. You describe it in a schema (JSON Schema), and that description is injected into the model's context alongside the conversation. A typical schema:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given city. Use this whenever the user asks about weather, temperature, or whether they need an umbrella.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "The city name, e.g. 'Austin' or 'London'"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature units to return"
      }
    },
    "required": ["city"]
  }
}
```

Three things matter enormously here, and this is where most real-world tool-use bugs come from:

1. **The `description` field is doing the actual "programming."** The model decides *whether* and *when* to call this tool almost entirely based on how well this description matches the user's intent. A vague description ("gets weather") leads to the model either never calling it or calling it at the wrong times. A precise description with trigger conditions ("use this whenever the user asks about weather...") dramatically improves correct invocation.
2. **`required` fields matter for parsing correctness** — if `city` isn't marked required, the model may generate a malformed call missing it, and your code needs to handle that gracefully (see §4).
3. **This schema costs context-window tokens on every single call.** If you have 40 tools, all 40 schemas are sent on *every* turn, not just when relevant — this is a real cost lever (see §5).

### 2.3 What the Model Literally Outputs

For the schema above, given "What's the weather in Austin?", the model's raw output (simplified) looks like:

```json
{
  "type": "tool_use",
  "id": "call_abc123",
  "name": "get_weather",
  "input": {"city": "Austin", "units": "fahrenheit"}
}
```

Nothing has happened yet. No API has been hit. This is just the model's best guess at structured text, generated token-by-token exactly like any other text — the model is literally predicting `{`, then `"`, then `t`, then `y`, then `p`, ... one token at a time, constrained (in modern implementations) by a grammar/schema validator that only allows tokens that keep the JSON valid.

---

## 3. Worked Example: Building the Execution Loop by Hand

Let's write the *actual* loop, not a framework abstraction, so the mechanics are unambiguous.

```python
def get_weather(city, units="fahrenheit"):
    # the REAL function — this is what actually runs
    response = requests.get(f"https://api.weather.example/{city}?units={units}")
    return response.json()  # e.g. {"temp": 71, "condition": "clear"}

tools_schema = [{
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    }
}]

# maps the string name the model outputs -> the real Python function
tool_registry = {"get_weather": get_weather}

messages = [{"role": "user", "content": "Do I need an umbrella in Austin?"}]

while True:
    response = llm_call(messages=messages, tools=tools_schema)

    if response.stop_reason == "tool_use":
        # STEP 1: model produced a structured call request (just text/JSON, nothing executed yet)
        tool_call = response.tool_use_block
        fn_name = tool_call.name              # "get_weather"
        fn_args = tool_call.input             # {"city": "Austin", "units": "fahrenheit"}

        # STEP 2: YOUR code looks up and actually executes the real function
        fn = tool_registry[fn_name]
        try:
            result = fn(**fn_args)             # <-- the actual side-effecting call happens HERE
            result_content = json.dumps(result)
        except Exception as e:
            result_content = f"Error: {str(e)}" # tool errors get FED BACK to the model as text!

        # STEP 3: append both the model's call and the real result to the conversation
        messages.append({"role": "assistant", "content": [tool_call]})
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_content})

        # loop continues — model will see this result on the next call

    else:
        # model decided it has enough info and returned plain text
        return response.content
```

### 3.1 The Non-Obvious Part: Errors Become Text

Look closely at the `except` block. If `get_weather()` throws (network timeout, city not found, API key expired), **you don't crash the loop** — you convert the exception into a text observation and feed it back to the model, exactly like a successful result. The model then *reasons about the error* the same way it reasoned about the Turn 2 ambiguous CEO result in Day 2:

```
Observation: Error: city 'Austn' not found (404)
Thought: I think I misspelled the city name. Let me retry with the correct spelling.
Action: get_weather(city="Austin", units="fahrenheit")
```

This is a critical, non-obvious production pattern: **tool errors are not exceptions to be caught and hidden — they are observations to be reasoned about.** Silently swallowing an error (returning nothing, or crashing the whole loop) removes the model's ability to self-correct.

---

## 4. Worked Example: Handling Malformed Calls

Despite JSON-schema-constrained decoding, malformed or semantically-wrong calls still happen — e.g., the model hallucinates a tool name that doesn't exist, or gets an argument type wrong.

```python
fn_name = "get_wether"   # model typo'd the tool name (rare with constrained decoding, but happens with certain models/setups)

if fn_name not in tool_registry:
    result_content = f"Error: tool '{fn_name}' does not exist. Available tools: {list(tool_registry.keys())}"
```

Again — **feed the error back as an observation**, don't crash. The model, on seeing "does not exist, available tools are: [...]", will typically self-correct on the next turn by picking the right name from the list you gave it. This is a real, common, cheap trick: **when in doubt, tell the model what its valid options actually are, inside the error message itself.**

---

## 5. Production Considerations

### 5.1 Tool Schema Bloat — The Hidden Context Tax

Every tool schema is sent on **every single call**, whether or not it's used that turn. If you have 50 tools each with a ~150-token schema (name + description + parameters), that's **7,500 tokens of pure overhead, per LLM call, forever**, before the actual conversation even starts.

**Production mitigations**:
- **Tool subsetting / retrieval**: dynamically select only the top-k most relevant tools for the current query (using embedding similarity of the query against tool descriptions) instead of sending all tools every time. This is essentially "RAG for tools."
- **Tool namespacing/grouping**: group related tools under a single higher-level tool that then routes internally, reducing the top-level schema count the model has to reason over.
- Real reported pattern at scale: agents with 100+ available tools see measurably *worse* tool-selection accuracy when all schemas are dumped in at once — too many similar options increases confusion, not just cost. This is a genuine accuracy problem, not just a token-cost problem.

### 5.2 Parallel vs Sequential Tool Calls

Modern models can emit **multiple tool_use blocks in a single response** — e.g., "call `get_weather("Austin")` AND `get_weather("Dallas")`" in one turn, if both are independent.

```
Thought: I need weather for both cities the user mentioned, and these calls are independent, so I can do them in parallel.
Action: [get_weather("Austin"), get_weather("Dallas")]
```

**Production win**: if your execution layer detects independent calls and runs them concurrently (e.g., `asyncio.gather`), you cut latency roughly in half (or more) versus looping through them one at a time, each requiring a full model round-trip. This is a real, high-leverage optimization — sequential tool execution when calls are independent is one of the most common unnecessary-latency bugs in production agents.

### 5.3 Idempotency and Side Effects

This is the single most safety-critical production concern in tool use. Consider a `send_email` or `charge_credit_card` tool. If:
- the model's stop-reason parsing has a bug and the loop retries the same tool call,
- or a network timeout causes your code to resend a request whose first attempt actually succeeded server-side,

...you can get **duplicate real-world side effects** — double-charged customers, duplicate emails sent. Standard mitigations:
- **Idempotency keys**: generate a unique key per logical action (not per retry) and have the downstream system dedupe on it.
- **Read vs. write tool separation**: read-only tools (search, get_weather) can be retried freely with no harm; write/side-effecting tools (send_email, charge_card, delete_file) should require explicit confirmation gates or human-in-the-loop approval (Day 11) before execution.

### 5.4 Sandboxing

If one of your "tools" is `execute_code(code: str)` (very common for coding/data-analysis agents), you are letting the model's output directly drive code execution on your infrastructure. This must run in an isolated sandbox (container, VM, restricted runtime) with no access to secrets, the host filesystem, or the network beyond what's explicitly allowlisted — because a prompt-injected or simply mistaken model can generate destructive code, and you must assume it eventually will.

---

## 6. Interview Q&A

**Q1: Walk me through, mechanically, what happens when a model "calls a function."**
A: The model doesn't execute anything — it's a text generator. It produces a structured JSON object naming a tool and arguments, constrained by a schema you provided. Your application code detects this structured output, looks up the real function by name in a registry, executes it with the given arguments, and appends the real result back into the conversation as a new message before calling the model again. All actual execution happens outside the model, in code you control.

**Q2: What's in a tool schema, and which part matters most for correct tool selection?**
A: Name, description, and a JSON-schema-defined set of parameters (with types and which are required). The `description` field matters most — it's effectively the only signal the model has for deciding *whether* and *when* to invoke that tool, so vague descriptions cause missed or wrongly-timed calls, while precise descriptions with explicit trigger conditions ("use this when...") significantly improve invocation accuracy.

**Q3: A tool call throws an exception. What should your code do?**
A: Never let it crash the agent loop or silently swallow it — convert the exception into a text observation (e.g., "Error: city not found") and feed it back into the conversation like any other tool result. This lets the model reason about the failure and self-correct (retry with fixed arguments, try a different tool, or inform the user), the same way it reasons about any other observation.

**Q4: How would you reduce latency for an agent that needs to call 5 independent tools to answer one question?**
A: Check whether the model emitted multiple tool_use blocks in a single turn (most modern APIs support this for independent actions), and execute them concurrently (e.g., asyncio.gather) rather than looping through them sequentially with a full model round-trip between each. This can cut end-to-end latency dramatically since you're paying for one round of parallel tool execution instead of N sequential round trips.

**Q5: You have 80 tools registered. What production problems does this cause, and how do you fix them?**
A: Two problems: (1) token cost — every schema is sent on every call regardless of relevance, and (2) accuracy — too many similar tool options measurably degrades correct tool selection, it's not just a cost issue. Fix with dynamic tool subsetting (retrieve only the top-k relevant tools per query via embedding similarity against tool descriptions, essentially RAG over your tool catalog) or by grouping related tools behind a smaller number of higher-level router tools.

**Q6: Why is a `charge_credit_card` tool fundamentally different, from a systems-design perspective, than a `search` tool?**
A: `search` is read-only and idempotent — retrying it has no harmful side effect, so it can be safely re-invoked on ambiguity or failure. `charge_credit_card` is a side-effecting write — a duplicate call from a retry or a parsing bug causes a real-world harm (double charge). Side-effecting tools need idempotency keys, explicit confirmation/human-in-the-loop gates before execution, and much stricter error handling than read-only tools.

---

## 7. Summary Card

- The model **only ever generates structured text** describing a call — it never executes anything. Your code parses it, runs the real function, and injects the real result back as a new message.
- The `description` field in a tool schema is doing the real "programming" — it's the model's only signal for when to use a tool.
- **Errors are observations, not exceptions to hide** — always feed failures back to the model as text so it can self-correct.
- Production levers: tool subsetting (cost + accuracy), parallel execution of independent calls (latency), idempotency keys + confirmation gates for side-effecting tools (safety), sandboxing for code-execution tools (security).

---
*Next: Day 4 — Planning Strategies (task decomposition, ReAct vs. Plan-and-Execute vs. Reflexion).*
