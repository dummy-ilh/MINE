# Day 16: Cost & Latency — Token Budgets, Caching, Model Routing

## 1. The Intuition First

Every single mechanism you've built through Day 15 — ReAct loops, multi-agent dispatch, agentic RAG, ToT branching, retries — has one thing in common: **it's an LLM call, and LLM calls cost money and take time, every single one.** Days 1-15 kept flagging this ("this triples cost," "this adds a round trip") as a side note on each pattern. Today those side notes become the main event — the specific, quantifiable engineering discipline of making an agent affordable and fast enough to actually ship.

Think about a contractor billing by the hour who also has to hit a deadline. Every unnecessary phone call, every redundant trip to the hardware store for the same materials, every time they call in a specialist for a job the apprentice could handle — all of that is pure waste against both the budget and the clock. Cost and latency optimization for agents is exactly this instinct, formalized: **don't do more expensive work than the task requires, don't repeat work you've already done, and don't use a bigger tool than the job needs.**

---

## 2. Formalizing the Three Levers

### 2.1 Token Budgets — Bounding What You Spend, Explicitly

Every mechanism from Days 1-15 that "loops" (ReAct, Reflexion, ToT, multi-agent dispatch) has an implicit cost multiplier. A **token budget** makes that cost explicit and enforced, not just hoped-for:

```python
class TokenBudget:
    def __init__(self, max_tokens=50000):
        self.max_tokens = max_tokens
        self.used = 0

    def charge(self, tokens_used):
        self.used += tokens_used
        if self.used >= self.max_tokens:
            raise BudgetExceededError(f"Used {self.used}/{self.max_tokens} tokens")

    def remaining_pct(self):
        return 1 - (self.used / self.max_tokens)
```

**The genuinely interesting production technique**: inject the remaining budget INTO the model's context, so its own reasoning (Day 2's Thought step) can factor it in — directly extending Day 2 §5.2's mitigation for runaway loops:

```
[System context: You have used 38,000 of 50,000 available tokens (76%). Budget carefully.]

Thought: I'm at 76% of my budget. I've confirmed the core fact I need; rather than
doing one more verification search, I should finalize my answer now with what I have.
```
This is the SAME principle as Day 4's Reflexion or Day 10's ToT evaluator — giving the model information it can reason about, rather than a purely external kill switch it has no visibility into. Both matter: the external hard cap (never actually exceed the budget) AND the soft internal signal (let the model's own judgment steer toward efficiency before the hard cap forces a worse outcome).

### 2.2 Caching — Never Pay Twice for the Same Work

If the same (or a near-identical) input produces the same output, computing it again is pure waste. Three distinct levels of caching apply to agents, and interviewers expect you to distinguish them:

**a) Exact-match response caching**: identical tool call with identical arguments → return the cached result instead of re-executing.
```python
cache_key = hash((tool_name, frozenset(arguments.items())))
if cache_key in cache and not is_stale(cache[cache_key]):
    return cache[cache_key]
result = execute_tool(tool_name, arguments)
cache[cache_key] = result
```

**b) Prompt/prefix caching**: many LLM providers can cache the KV-state of a repeated prompt prefix (e.g., your system prompt + tool schemas, which are IDENTICAL on every single turn of a conversation) so the model doesn't re-process those same tokens from scratch on every call — this is a provider-level mechanism you enable, not something you build yourself, but knowing it exists and WHY it works (the prefix genuinely doesn't change turn to turn) is a real, current, interview-relevant fact.

**c) Semantic caching**: NOT an exact match, but a semantically similar query returns a cached result — e.g., "what's our refund policy" and "how do refunds work" might hit the same cached answer, using embedding similarity to detect the match rather than exact string equality. More powerful, but riskier — a near-miss on similarity can return a subtly wrong cached answer for a query that only LOOKS similar (directly analogous to Day 5 §5.2's RAG retrieval-precision problem, now applied to caching instead of memory retrieval).

### 2.3 Model Routing — Match Model Size to Task Difficulty

Not every step in an agent's trajectory needs your most capable (and most expensive/slowest) model. **Model routing** dynamically picks which model handles a given call based on the complexity of that specific step.

```python
def route_model(step_type, complexity_estimate):
    if step_type == "simple_classification" or complexity_estimate == "low":
        return "small-fast-model"
    elif step_type == "final_synthesis" or complexity_estimate == "high":
        return "large-capable-model"
    else:
        return "medium-model"
```

**Concrete example**: in the Day 8 orchestrator/worker due-diligence example, the ORCHESTRATOR's decomposition and final synthesis probably benefit from your most capable model (getting the plan and synthesis right matters most for overall quality) — but a WORKER doing a narrow, well-defined sub-task (e.g., "extract revenue figures from this filing") might perform just as well on a smaller, faster, much cheaper model, since the task is well-scoped and doesn't need the largest model's full reasoning capacity.

---

## 3. Worked Example: All Three Levers on One System

**Task**: A customer-support agent handling ~10,000 tickets/day, using the orchestrator/worker pattern (Day 8) with an agentic-RAG-based knowledge lookup (Day 9).

**Without optimization** (naive baseline):
```
Every ticket: 
  1 large-model call to classify + route (system prompt + tool schemas re-sent every time)
  1-3 large-model calls for knowledge-base search + reformulation (Day 9's agentic RAG loop)
  1 large-model call to draft the response

Estimated: ~5 large-model calls/ticket × 10,000 tickets/day × large-model pricing
= expensive, and each ticket takes the full latency of 5 sequential large-model calls
```

**With all three levers applied**:

```
1. MODEL ROUTING:
   - Classification/routing step → small-fast-model (this is a well-scoped,
     narrow task — doesn't need large-model reasoning capacity)
   - Knowledge-base search reformulation → small-fast-model (also narrow)
   - Final response drafting → large-model (this is the step where quality
     genuinely matters most to the customer-facing outcome)
   Net effect: 3 of the ~5 calls per ticket move to a cheaper, faster tier.

2. CACHING:
   - Prompt-prefix caching enabled → system prompt + tool schemas (identical
     every call) aren't re-processed from scratch each time — pure savings
     on the LARGE, FIXED part of every prompt (recall Day 3 §5.1: schema
     tokens are sent on every call regardless of relevance — prefix caching
     is a direct mitigation for exactly that cost).
   - Exact-match tool-result caching on knowledge-base search → many tickets
     ask semantically identical common questions ("how do I reset my password"),
     so a meaningful fraction of searches hit a cache instead of re-querying.
   - Semantic caching on the FINAL drafted response for near-duplicate common
     questions → riskier (per §2.2c), so gated to only apply for high-confidence
     similarity matches on a small set of pre-vetted, template-like FAQ responses,
     not applied broadly to novel customer issues.

3. TOKEN BUDGET:
   - Each ticket's agentic RAG loop (Day 9) gets a budget of 3 search attempts
     max, with the remaining-budget signal injected into context so the model
     stops searching once it judges it has enough, rather than defaulting to
     always using all 3 attempts.

Net effect: most tickets now use 1-2 large-model calls (routing + drafting) instead
of ~5, a meaningful fraction of searches/routing hit cache or run on the cheap
model tier, and the RAG loop is capped rather than open-ended.
```

### 3.1 Why This Isn't Just "Use a Cheaper Model Everywhere"

The naive alternative — just use the small model for everything — would save more money but would genuinely hurt quality on the step that matters most (final response drafting, the thing the customer actually reads). **The actual skill being tested here is NOT "minimize cost" — it's "allocate your most expensive resource (the large model) specifically to the steps where its extra capability changes the outcome, and use cheaper resources everywhere else."** This is the same principle as Day 8's specialization benefit, applied to model selection instead of agent role — put the expensive expert exactly where expertise matters, not uniformly everywhere.

---

## 4. Production Considerations

### 4.1 Cache Invalidation — The Classic Hard Problem, Now in an Agent Context

Caching is easy to build and easy to get subtly wrong: a cached tool result (e.g., inventory stock levels, Day 14's example) can go **stale** — correct when cached, wrong now. This directly echoes Day 5 §5.2's "stale memory" problem, and the same category of fix applies: **cache entries need explicit TTLs (time-to-live) appropriate to how fast the underlying data changes** — a company's refund POLICY might be cacheable for days, but live inventory stock should be cached for seconds at most, if at all. Treating all cached data with one uniform TTL is a common, avoidable production bug.

### 4.2 The Cost/Quality/Latency Triangle — You Can't Fully Optimize All Three

This is worth being explicit about in an interview, because it's a genuine, unavoidable tradeoff, not a solvable puzzle: **cheaper models are typically faster too (a real win on two axes at once), but they generally trade away some quality/capability.** Aggressive caching improves both cost and latency, but only where cache-hit rates are genuinely high and staleness risk is genuinely low — it doesn't help novel, first-time queries at all. Token budgets protect cost predictability but can force a worse-quality "best effort" answer if the cap is hit before the task is genuinely done. **There's no configuration that maximizes all three simultaneously — the right answer is always "which one matters most for THIS specific task," stated explicitly**, not a generic "optimize everything" hand-wave.

### 4.3 Parallel Execution as a Latency Lever (Direct Callback to Day 3 §5.2 and Day 8 §5.1)

Worth explicitly re-surfacing here as a cost/latency-specific point: independent tool calls (Day 3) and independent worker dispatch (Day 8) both parallelize well, meaning **latency can often be reduced without any reduction in the actual amount of work done** — you're not doing less, you're doing the same total work concurrently instead of sequentially. This is a genuinely "free" latency win (no quality tradeoff) wherever true independence exists, which is why identifying independent vs. dependent steps in a trajectory (something the model itself can reason about, as shown in Day 3's parallel weather-lookup example) is one of the highest-value, lowest-cost optimizations available.

### 4.4 Measuring Before Optimizing — Don't Guess Where the Cost Actually Is

This directly depends on Day 15's observability layer: **you cannot correctly apply model routing, caching, or budget limits without first knowing, from real trace/log data, where cost and latency are actually concentrated.** A common mistake is intuiting "the LLM calls must be the expensive part" and optimizing those first, when in some systems a slow, un-cached tool call (like Day 15's degraded knowledge-base search) dominates total latency far more than any model choice does. **The correct order is: instrument (Day 15) → measure → THEN optimize the actual bottleneck**, not the assumed one.

---

## 5. Interview Q&A

**Q1: Why inject the remaining token budget into the model's own context, rather than just enforcing a hard cutoff externally?**
A: A purely external hard cap only prevents catastrophic overrun — it doesn't help the model make better decisions along the way. Injecting the remaining budget as context lets the model's own reasoning (the Thought step) factor in urgency, e.g., choosing to finalize an answer with what it has rather than doing one more speculative search — the same principle as giving a model visibility into any other constraint it should reason about, rather than only discovering the constraint when it's abruptly cut off.

**Q2: Explain the three levels of caching relevant to agent systems and the risk profile of each.**
A: Exact-match caching returns a cached result for an identical tool call with identical arguments — safe, but only helps on literal repeats. Prompt-prefix caching lets a provider avoid reprocessing an identical prompt prefix (system prompt, tool schemas) across calls in the same conversation — a provider-level mechanism, essentially free savings on fixed, repeated content. Semantic caching matches near-similar (not identical) queries via embedding similarity — more powerful because it catches paraphrased repeats, but riskier, since a near-miss on similarity can return a subtly wrong cached answer for a query that only superficially resembles the cached one.

**Q3: Would you use the same, cheapest available model for every step of an agent's trajectory to minimize cost? Why or why not?**
A: No — that would save the most money but hurt quality on whichever step actually determines the outcome's quality (e.g., final response drafting in a customer-support flow). The right approach is model routing: match model capability to each specific step's actual difficulty, reserving the most capable (and expensive) model for steps where its extra capability changes the outcome, and using cheaper/faster models for narrow, well-scoped sub-tasks like classification.

**Q4: What's a common mistake in setting cache TTLs for an agentic system with multiple tools?**
A: Applying one uniform TTL across all cached data, when different tools' underlying data changes at very different rates — a policy document might be safely cacheable for days, while live inventory or pricing data could be stale within seconds. TTLs need to be set per data type based on how quickly that specific underlying data actually changes, not applied as one blanket setting.

**Q5: Can you fully optimize cost, latency, and quality simultaneously? Why or why not?**
A: Generally no — they trade off against each other in real ways: cheaper/faster models generally sacrifice some capability; aggressive caching helps cost and latency but only on cache hits, doing nothing for novel queries; tight token budgets protect cost predictability but risk forcing a worse "best effort" answer if hit before the task is genuinely complete. The right framing in an interview is naming which of the three matters most for the specific task at hand, rather than claiming a configuration that maximizes all three at once.

**Q6: Before applying any cost/latency optimization, what should you do first?**
A: Instrument the system (Day 15's traces/logs/metrics) and measure where cost and latency are actually concentrated, rather than assuming — a common mistake is optimizing LLM call costs first on the assumption that's the expensive part, when in practice a slow or un-cached tool call can dominate total latency far more than model choice does. Optimization should target the bottleneck the data actually shows, not the one that seems intuitively likely.

---

## 6. Summary Card

- **Token budgets**: enforce a hard external cap, AND inject the remaining budget into the model's own context so its reasoning can steer toward efficiency before the cap forces a worse outcome.
- **Caching**: three distinct levels — exact-match (safe, narrow), prompt-prefix (provider-level, near-free for repeated fixed content), semantic (powerful but riskier, same precision tradeoff as RAG retrieval) — each needs TTLs matched to how fast the underlying data actually changes.
- **Model routing**: match model capability to each step's actual difficulty; concentrate your most expensive model on the steps where its capability genuinely changes the outcome, not uniformly everywhere.
- Parallel execution (Days 3, 8) is a "free" latency win wherever true task independence exists — no quality tradeoff.
- Cost/latency/quality is a genuine three-way tradeoff with no universal optimum — always optimize toward the metric that matters most for the specific task, and always measure (Day 15) before optimizing.

---
*Next: Day 17 — Guardrails & Safety (input/output validation, prompt injection defense, sandboxing tool execution).*
