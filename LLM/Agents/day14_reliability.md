# Day 14: Reliability — Retries, Timeouts, Fallback Models, Circuit Breakers

## 1. The Intuition First

Everything through Day 13 assumed the world cooperates: tool calls succeed, models respond in reasonable time, APIs are up. Production doesn't cooperate. Networks drop packets. Model providers have outages. A tool's downstream API rate-limits you at 2am. **Reliability engineering is the discipline of assuming all of this WILL happen, regularly, and designing so the agent degrades gracefully instead of falling over.**

Think about a delivery driver's job. If a customer doesn't answer the door, they don't just stand there forever (no timeout = infinite hang) or give up after one knock and drive away forever (no retry = fragile). They knock, wait a reasonable amount, try again, maybe call the customer, and if it's genuinely not working, they follow a fallback procedure (leave it with a neighbor, return it to the depot) rather than blocking every other delivery on their route behind this one stuck delivery. That instinct — bounded patience, then a defined fallback, without blocking everything else — is the entirety of reliability engineering, applied to agent systems instead of package delivery.

---

## 2. Formalizing the Four Core Mechanisms

### 2.1 Retries — For Transient Failures

Not every failure means "this will never work" — many are transient (a momentary network blip, a rate limit that clears in a second). Retrying is appropriate specifically for **failures where trying again, unchanged or slightly modified, has a real chance of succeeding.**

```python
def call_with_retry(fn, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return fn()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)  # exponential backoff
            time.sleep(delay)
```

**Exponential backoff** matters specifically because a fixed retry delay (retry every 1 second) can make things WORSE during an actual outage — if a downstream service is struggling under load, a wave of clients all retrying every 1 second adds sustained pressure exactly when the service needs load to drop. Exponential backoff (1s, 2s, 4s, 8s...) gives the struggling service room to recover, and spreads out retry pressure over time rather than hammering it at a constant rate.

**Not everything should be retried.** A malformed tool call (Day 3 §4 — hallucinated tool name, wrong argument types) retried unchanged will fail identically every time — that's not a transient failure, it needs correction (feed the error back as an observation, per Day 3, so the MODEL retries with different arguments), not blind re-execution of the same call.

### 2.2 Timeouts — Bounding How Long You'll Wait

Every external call — LLM inference, tool execution, a downstream API — needs an explicit maximum wait time, or a single hung call can block the entire agent loop indefinitely (this is the ACTING-state hang scenario mentioned in Day 7 §2.2's escape-hatch discussion, now made concrete).

```python
def call_with_timeout(fn, timeout_seconds=30):
    try:
        return fn(timeout=timeout_seconds)
    except TimeoutError:
        return {"error": f"Operation timed out after {timeout_seconds}s"}
```

The result of a timeout should be treated exactly like any other tool error (Day 3 §3.1) — fed back as an observation the model can reason about ("the weather API timed out, let me try a different source"), not a silent crash.

**Setting the right timeout value is itself a real engineering decision**: too short, and you abort operations that would have succeeded given a bit more time (wasted retry cost, worse user experience); too long, and a single slow call can dominate the perceived latency of the whole agent trajectory. Production systems often use **different timeouts for different tool types** — a quick lookup tool might get 5 seconds, a code-execution tool doing real computation might reasonably get 60.

### 2.3 Fallback Models — Degrading Gracefully When the Primary Model Is Unavailable

If your primary model provider has an outage or is rate-limiting you, the agent shouldn't simply fail — it can fall back to an alternate model (a different provider, or a smaller/older model version) to keep functioning, even at reduced quality, rather than not functioning at all.

```python
def llm_call_with_fallback(messages, tools, primary="model-a", fallback="model-b"):
    try:
        return call_model(primary, messages, tools)
    except ProviderUnavailableError:
        log_warning(f"{primary} unavailable, falling back to {fallback}")
        return call_model(fallback, messages, tools)
```

**This connects directly to Day 12's framework discussion**: whichever framework/raw-loop approach you use, the model-calling layer needs to be abstracted enough that swapping the underlying model doesn't require rewriting the agent's logic — this is a concrete reason "avoid framework/vendor lock-in in your core loop" matters practically, not just architecturally.

### 2.4 Circuit Breakers — Stop Calling Something That's Clearly Broken

Retries handle occasional transient failures. But if a downstream tool/API is **consistently** failing (not a blip — actually down), continuing to retry every single call against it wastes time, adds latency to every request, and can make the outage worse (retry storms, Day 2's/Day 9's runaway-cost pattern applied to infrastructure instead of reasoning loops). A **circuit breaker** tracks the failure rate of a given dependency and, once it crosses a threshold, "opens" — stops attempting calls to that dependency entirely for a cooldown period, failing fast instead of waiting through a timeout each time.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_seconds=60):
        self.failure_count = 0
        self.state = "closed"  # closed = normal, open = blocking, half_open = testing
        self.opened_at = None
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

    def call(self, fn):
        if self.state == "open":
            if time.time() - self.opened_at > self.cooldown_seconds:
                self.state = "half_open"  # allow one test call through
            else:
                raise CircuitOpenError("Circuit breaker open — failing fast")

        try:
            result = fn()
            if self.state == "half_open":
                self.state = "closed"  # test call succeeded, fully close again
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.opened_at = time.time()
            raise
```

Three states: **closed** (normal operation), **open** (failing fast, not even attempting the call), **half-open** (after the cooldown, cautiously let one call through to test if the dependency has recovered). This is the standard circuit-breaker state machine, and it's worth explicitly noting the parallel to Day 7: **this is itself a small state machine, the same formalization principle applied to a reliability concern instead of the agent's reasoning loop.**

---

## 3. Worked Example: All Four Mechanisms in One Trajectory

**Task**: Agent needs to call an internal `get_inventory_status(sku)` tool as part of answering a customer question, and this particular downstream inventory service is having a bad day.

```
State: ACTING
  Attempt 1: get_inventory_status("SKU-123")
  → Network timeout after 5s (timeout mechanism fires)
  → Observation fed back: "Error: request timed out after 5s"

State: THINKING
  Thought: Timeout could be transient — worth retrying with backoff before giving up.
  [Retry logic, NOT model-driven this time — this is infrastructure-level retry,
   distinct from the model deciding to re-issue a DIFFERENT call]

State: ACTING (retry #1, after 1s backoff)
  → Times out again.

State: ACTING (retry #2, after 2s backoff)
  → Times out again. This is now the 3rd consecutive failure for this tool.

  [Circuit breaker for get_inventory_status trips: failure_count reaches threshold,
   circuit opens. Next 60 seconds, calls to this tool fail IMMEDIATELY without
   even attempting the network call.]

State: OBSERVING
  Observation: "Error: inventory service is currently unavailable (circuit open).
  Unable to retrieve real-time stock status."

State: THINKING
  Thought: The inventory service is down and retrying won't help right now (circuit
  is open). I should let the customer know I can't confirm live stock status, rather
  than keep trying or guessing.
  Final Answer: "I'm unable to check real-time inventory right now due to a system
  issue on our end — I'd recommend checking the product page directly, or I can
  follow up once the service is back."
```

Note what did NOT happen: the agent did not hang indefinitely (timeout), did not retry forever burning cost/latency on a service that's genuinely down (circuit breaker), and did not silently fail with no explanation (error fed back as an observation, per Day 3, letting the model produce an honest, helpful response instead of a crash or a hallucinated inventory number).

### 3.1 The Fallback-Model Case, Separately

**Same task, different failure**: the primary LLM provider itself has an outage (not the tool, the reasoning model).

```
[Orchestration layer attempts to call primary model for the next THINKING step]
→ ProviderUnavailableError (503, primary provider down)
→ Fallback triggers: switch to secondary model provider for this call
→ Secondary model completes the THINKING step normally, agent continues

[A log/alert fires noting the fallback occurred — this is NOT invisible; Day 15's
observability needs to surface "we're running degraded on the fallback model" so
someone investigates the primary outage]
```

**Important distinction**: retries/timeouts/circuit breakers are about *tool calls and downstream dependencies* (Day 3's execution layer); fallback models are about *the reasoning engine itself* being unavailable — a different, higher-severity failure category, since without ANY model, the agent literally cannot think, only degrade to whatever fallback is available.

---

## 4. Production Considerations

### 4.1 Idempotency Is a Prerequisite for Safe Retries (Direct Callback to Day 3 §5.3)

Retrying a read-only tool (`get_inventory_status`) is always safe. Retrying a side-effecting tool (`charge_credit_card`, `send_email`) without an idempotency key can cause duplicate real-world effects if the first attempt actually succeeded server-side but the response was lost before your code saw it (a very real failure mode — success on the server, timeout on the client). **This means your retry policy needs to be tool-aware**: read-only tools get straightforward retry-with-backoff; write/side-effecting tools need idempotency keys (Day 3 §5.3) before retries are safe at all, or need to skip automatic retry entirely and surface the ambiguous state for human review (Day 11's escalation, applied to an infrastructure failure rather than a reasoning judgment call).

### 4.2 Retry Budgets Interact With Iteration Caps (Direct Callback to Day 2 §5.2 / Day 4 §5.2)

If each individual tool call can retry up to 3 times, and the agent's overall iteration cap is 10 THINKING/ACTING cycles, a tool that's consistently failing (but not yet tripped a circuit breaker) can silently consume 3x the "budget" per attempt — meaning your agent might exhaust its intended reasoning budget almost entirely on retries for one bad tool, rather than genuine reasoning steps. **Total cost/latency budgets need to account for retry multiplication at every layer, not just count top-level loop iterations** — this is a common gap between a design that looks bounded on paper and one that's actually bounded in practice.

### 4.3 Circuit Breakers Need Per-Dependency Granularity

A single global circuit breaker for "the agent" doesn't make sense — if `get_inventory_status` is down but `get_weather` is fine, you want the inventory circuit open while weather calls proceed normally. This means production systems maintain **one circuit breaker instance per distinct external dependency** (per tool, or per downstream service if multiple tools share one backend), so a failure in one dependency doesn't inappropriately block unrelated, healthy dependencies.

### 4.4 Silent Degradation Is a Real Risk — Reliability Mechanisms Need Their Own Observability

If fallback models, retries, and circuit breakers all work exactly as designed, the AGENT keeps functioning — but silently, at reduced quality or reduced capability, with no visible signal that anything is wrong. This is a genuine production risk: **reliability mechanisms that succeed at their job can mask an ongoing incident from the people who need to know about it.** Every fallback trigger, every circuit-breaker open event, every exhausted retry needs to emit a signal to your observability layer (Day 15) — reliability and observability are not separate concerns, they're tightly coupled: the entire point of graceful degradation is staying up WHILE someone gets paged to fix the underlying issue, not staying up instead of anyone finding out.

---

## 5. Interview Q&A

**Q1: Why is exponential backoff preferred over a fixed retry delay?**
A: A fixed short delay means all failing clients retry at the same constant rate, which can add sustained pressure to a downstream service exactly when it's struggling and needs load to decrease — potentially worsening an outage. Exponential backoff (doubling the delay each attempt) spreads retry pressure out over time and gives the struggling dependency room to recover, rather than hammering it at a constant rate.

**Q2: Should every tool failure be retried automatically? Why or why not?**
A: No — only transient failures (network blips, momentary rate limits) benefit from blind retry. A malformed tool call, like a hallucinated tool name or wrong argument type, will fail identically on retry since nothing about the call changed; that needs to be fed back as an observation so the model can retry with a corrected call, not automatically re-executed unchanged.

**Q3: What's the difference between a timeout and a circuit breaker, and why do you need both?**
A: A timeout bounds how long you'll wait for a single call before giving up. A circuit breaker tracks the failure rate across many calls to a dependency and, once a threshold is crossed, stops attempting calls entirely for a cooldown period, failing fast rather than waiting through a timeout on every single subsequent request. You need both: timeouts prevent one call from hanging indefinitely; circuit breakers prevent repeatedly paying the full timeout cost, request after request, against a dependency you already know is down.

**Q4: Why can't you safely retry every tool call the same way?**
A: Retry safety depends on idempotency. Read-only tools can be retried freely since repeating them has no side effect. Side-effecting tools (charging a payment, sending an email) risk duplicate real-world effects if a retry re-executes an action whose first attempt actually succeeded server-side but whose response was lost — those need idempotency keys before automated retry is safe, or should skip automatic retry and surface the ambiguous state for human review instead.

**Q5: Your agent has a 10-iteration cap, but each tool call can retry up to 3 times. What's the hidden risk in this design?**
A: A single tool that's consistently failing (without yet tripping a circuit breaker) can consume up to 3x its apparent "cost" per attempt in retries, meaning the agent could exhaust most of its intended 10-iteration reasoning budget on retries against one bad tool rather than genuine progress. Cost/latency budgets need to account for retry multiplication at every layer, not just count top-level loop iterations, or a design that looks bounded on paper isn't actually bounded in practice.

**Q6: Your reliability mechanisms (retries, fallback models, circuit breakers) are all working perfectly and the agent stays fully functional during an incident. Is there still a problem?**
A: Potentially yes — if none of these events (a fallback trigger, a circuit opening, retries being exhausted) are surfaced to an observability system, the incident is invisible even though the agent keeps working, possibly at silently reduced quality. Reliability and observability need to be coupled: the goal of graceful degradation is staying functional while someone gets alerted to fix the root cause, not staying functional as a substitute for anyone knowing there's an ongoing issue.

---

## 6. Summary Card

- **Retries**: for transient failures only, with exponential backoff to avoid worsening real outages; malformed calls need correction (fed back as observations), not blind retry.
- **Timeouts**: bound every external call's max wait; tool-type-specific durations; timeout results feed back as observations like any other error.
- **Fallback models**: a higher-severity failure category (the reasoning engine itself unavailable) — requires an abstracted model-calling layer to swap providers without rewriting agent logic.
- **Circuit breakers**: closed/open/half-open state machine per dependency; stop wasting time/cost on a dependency that's genuinely down rather than transiently flaky; needs per-dependency granularity, not one global breaker.
- Cross-cutting risks: retry-safety depends on idempotency (Day 3 §5.3); retry budgets compound with iteration caps (Days 2, 4); and every one of these mechanisms needs to emit signal to observability (Day 15) — silent graceful degradation can hide a real incident.

---
*Next: Day 15 — Observability (tracing agent steps, logging tool calls, debugging failures).*
