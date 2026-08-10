# Day 4: Planning Strategies — ReAct vs. Plan-and-Execute vs. Reflexion

## 1. The Intuition First

Imagine three different people asked to plan a birthday party:

- **Person A (ReAct-style)**: Doesn't write anything down ahead of time. Thinks "I should book a venue" → books it → then thinks "now I should order a cake" → orders it → then thinks "now I need to send invites" → sends them. Every decision is made one at a time, informed by what just happened.
- **Person B (Plan-and-Execute)**: Sits down first and writes a full plan: "1) Book venue, 2) Order cake, 3) Send invites, 4) Buy decorations, 5) Confirm headcount." *Then* executes each step in order, only replanning if something breaks (venue is booked out).
- **Person C (Reflexion-style)**: Does the whole party planning once, then afterward reviews what went wrong ("the cake arrived late because I ordered too close to the date"), writes that lesson down, and uses it to do better next time — or even mid-task, catches a mistake and revises the remaining plan based on the reflection.

These are exactly the three dominant planning strategies in agentic systems, and interviewers expect you to know precisely when each is appropriate — this is one of the highest-yield systems-design topics because it's a genuine engineering tradeoff, not just trivia.

---

## 2. Formalizing Each Strategy

### 2.1 ReAct (Recap from Day 2) — Implicit, Interleaved Planning

No explicit upfront plan. Plan emerges one step at a time:
```
Thought → Action → Observation → Thought → Action → Observation → ... → Answer
```
**Planning horizon: 1 step at a time.** The "plan" only exists implicitly, inside each Thought, and is fully revised after every observation.

### 2.2 Plan-and-Execute — Explicit, Upfront Decomposition

Two distinct phases, often literally two different LLM calls with different prompts:

**Phase 1 — Planner**: Given the task, produce a full ordered list of sub-tasks *before* executing anything.
```
Task: "Research the top 3 EV makers by 2025 sales and write a comparison summary."

Plan:
1. Search for 2025 EV sales rankings.
2. Identify top 3 companies by sales volume.
3. Search for key specs/pricing of each of the 3.
4. Compose comparison summary.
```

**Phase 2 — Executor**: Execute each plan step, in order, typically using a ReAct-style loop *within* each step (so it's not mutually exclusive with ReAct — it's ReAct nested inside a higher-level plan).

**Replanning**: If step 2 reveals something that invalidates step 3 or 4 (e.g., a company merger happened, changing "top 3"), a **replanner** component can regenerate the remaining plan given the new information, rather than blindly executing the stale plan.

### 2.3 Reflexion — Self-Critique and Iterative Improvement

Adds a **verbal self-reflection loop on top of either of the above**. After a full attempt (or a sub-task) completes:
```
Attempt → Evaluate outcome (success/fail, and why) → Generate a natural-language reflection ("I failed because...") → Store reflection in memory → Retry the task, with the reflection injected into context
```

Critically, Reflexion doesn't update model weights — the "learning" is entirely **in-context**: the reflection text itself becomes part of the prompt on the next attempt, acting as an episodic memory of past mistakes (this connects directly to Day 5, Memory Systems).

---

## 3. Worked Example: Same Task, Three Strategies

**Task**: "Find a 3-star-or-better rated Italian restaurant within 2 miles that's open right now, and book a table for 2 at 7pm."

### 3.1 Pure ReAct
```
Thought: I need to search for Italian restaurants nearby first.
Action: search_restaurants(cuisine="Italian", radius_miles=2)
Observation: [5 results, with ratings and hours]

Thought: I need to filter for 3+ stars and check which are open now. Result 3 (Mario's, 4.2 stars) and Result 5 (Luigi's, 3.5 stars) qualify; let me check Mario's hours.
Action: check_hours("Mario's")
Observation: "Open until 10pm today."

Thought: Mario's works. I'll try to book there.
Action: book_table("Mario's", party_size=2, time="19:00")
Observation: "Booking confirmed."

Thought: Task complete.
Final Answer: Booked a table for 2 at Mario's (4.2 stars) at 7pm.
```
Notice: the model never explicitly says "my plan is: search, filter, check hours, book" — it just does the next obviously-needed thing each time, informed only by what it's seen so far.

### 3.2 Plan-and-Execute
```
PLAN (generated upfront, before any tool call):
1. Search Italian restaurants within 2 miles.
2. Filter to those rated 3+ stars.
3. Check open-now status for filtered candidates.
4. Book a table for 2 at 7pm at the first available match.

EXECUTING STEP 1:
Action: search_restaurants(cuisine="Italian", radius_miles=2)
Observation: [5 results]

EXECUTING STEP 2:
[filters in-context, no tool call needed] → Mario's (4.2★), Luigi's (3.5★) qualify.

EXECUTING STEP 3:
Action: check_hours("Mario's")
Observation: "Open until 10pm."

EXECUTING STEP 4:
Action: book_table("Mario's", party_size=2, time="19:00")
Observation: "Booking confirmed."

Final Answer: Booked Mario's, 7pm, party of 2.
```
Looks almost identical in outcome here — **but the difference matters when things go wrong**, which is the next example.

### 3.3 Where Plan-and-Execute Actually Wins: Replanning

Same task, but now: Step 3 reveals Mario's is closed.

**Pure ReAct** naturally adapts step-by-step anyway (this is its strength) — it would just think "Mario's is closed, let me check Luigi's" and continue. So for *this particular failure*, ReAct handles it fine too.

Where Plan-and-Execute's explicit plan actually earns its keep is **long-horizon tasks with many independent sub-goals**, where you want:
- **Partial parallelism**: since the full plan is known upfront, independent steps (e.g., "check hours for Mario's" AND "check hours for Luigi's" AND "check hours for a 3rd candidate") can be dispatched concurrently — ReAct, being strictly sequential thought-by-thought, cannot naturally parallelize this because it doesn't know it'll need Luigi's info until it decides to.
- **Auditability**: a human (or an eval harness) can read the plan *before* execution and catch bad plans early ("step 4 books before confirming hours — reorder"), rather than discovering a flawed strategy only after tokens/tool-calls have already been spent.
- **Cost control on long tasks**: for a 20-step research task, generating the full plan upfront lets you estimate total cost/tool-calls in advance and reject or simplify plans that are too expensive, before executing a single step.

**The precise interview answer**: *"ReAct is naturally adaptive because replanning is implicit and continuous, but it's strictly sequential and can't be inspected before execution starts. Plan-and-Execute trades some adaptiveness for auditability, parallelizability of independent sub-steps, and upfront cost estimation — which matters much more as task horizon grows. Most production systems use Plan-and-Execute for the top-level structure and ReAct within each individual step."*

---

## 4. Worked Example: Reflexion in Action

**Task**: A coding agent asked to write a function that passes a given test suite.

**Attempt 1**:
```
Action: write_code("def is_palindrome(s): return s == s[::-1]")
Action: run_tests()
Observation: "3/5 tests passed. Failed: is_palindrome('A man a plan a canal Panama') expected True, got False."
```

**Reflection step** (a distinct LLM call, prompted specifically to critique the failure):
```
Reflection: "My implementation does a naive string reversal comparison, but it doesn't
account for case and spacing/punctuation — the failing test has spaces and mixed case.
I need to normalize the string (lowercase, strip non-alphanumeric characters) before
comparing it to its reverse."
```

This reflection is then **injected into the context of Attempt 2**:

**Attempt 2** (same task, but now with the reflection text prepended to the prompt):
```
[Context includes: "Reflection from previous attempt: ...normalize the string..."]

Action: write_code("""
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
""")
Action: run_tests()
Observation: "5/5 tests passed."

Final Answer: Task complete.
```

### 4.1 Why This Isn't Just "Retry"

The critical mechanical detail: a plain retry (just running the same prompt again, maybe with higher temperature/sampling) has **no memory of what specifically went wrong** — it might stumble onto the fix by luck, or repeat the exact same mistake. Reflexion forces an intermediate step whose *entire job* is to articulate the failure in natural language, which then becomes grounding context for the next attempt — turning "try again randomly" into "try again, informed."

This matters a lot in practice: **the reflection text is doing the same job as the "Thought" in ReAct** — it's not decorative, it's an intermediate reasoning artifact that measurably changes what the next generation conditions on.

---

## 5. Production Considerations

### 5.1 Choosing a Strategy — The Real Decision Tree

| Situation | Best fit |
|---|---|
| Short task (1-5 steps), high need for adaptiveness to unpredictable observations | ReAct |
| Long-horizon task (10+ steps), sub-goals mostly independent, want auditability/parallelism/cost estimation | Plan-and-Execute |
| Task has a verifiable success signal (tests pass, output matches schema, score threshold) and can afford multiple attempts | Reflexion (layered on either of the above) |
| Latency-critical, single-shot-feasible task | Neither — don't over-engineer; a workflow (Day 1, Level 0-1) may be correct |

**This decision-tree framing is exactly what senior candidates are expected to produce unprompted** in a system design interview — not just define the three patterns.

### 5.2 Reflexion's Real Cost: It's Not Free Improvement

Every reflection cycle is a full extra attempt (extra tool calls, extra tokens, extra latency) — Reflexion effectively multiplies your worst-case cost by the number of retry cycles you allow. In production, this is almost always **bounded** (e.g., max 2-3 reflection cycles), because:
- diminishing returns kick in fast (the failure mode you can articulate and fix in one reflection is usually the "easy" one — repeated failures often need a different tool, not another reflection),
- and unbounded reflection loops have the exact same runaway-cost risk as the unbounded ReAct loops discussed in Day 2 §5.2.

### 5.3 Plan Quality Depends Entirely on Decomposition Granularity

A subtle production failure mode: if the planner produces steps that are **too coarse** ("1. Do the research. 2. Write the summary"), the executor is left doing all the real ReAct-style improvisation *inside* step 1 anyway, and you've gained none of Plan-and-Execute's benefits (auditability, parallelism) — you've just added an extra LLM call for a plan that doesn't actually constrain anything.

Conversely, if steps are **too fine-grained** ("1. Open browser. 2. Type search query. 3. Press enter..."), you lose flexibility — a single unexpected observation ("no results found") can invalidate 5 downstream steps, forcing constant replanning overhead that erodes the benefit of planning upfront at all.

**Practical rule of thumb used in production planning prompts**: plan at the granularity of *independently verifiable sub-goals* — each step should be something you could hand to a separate ReAct-loop worker and get back a clear success/fail signal for, no finer.

---

## 6. Interview Q&A

**Q1: Compare ReAct and Plan-and-Execute. When would you choose one over the other?**
A: ReAct interleaves reasoning and acting one step at a time with no upfront plan — highly adaptive, but strictly sequential and not inspectable before execution. Plan-and-Execute generates a full ordered plan upfront, then executes each step (often using ReAct internally per step) — this trades some adaptiveness for auditability (a human/eval can review the plan before spending tool calls), parallelizability of independent steps, and upfront cost estimation. Choose ReAct for short, unpredictable tasks; choose Plan-and-Execute for long-horizon tasks with mostly independent sub-goals where you want to inspect or parallelize the plan.

**Q2: What does Reflexion add on top of ReAct or Plan-and-Execute, and how is it different from just retrying?**
A: Reflexion adds an explicit self-critique step after a failed attempt — the model articulates in natural language *why* it failed, and that reflection text is injected into the context of the next attempt. This differs from a plain retry (rerunning the same prompt, possibly with different sampling) because a plain retry carries no memory of the specific failure mode, while Reflexion's articulated reflection actively conditions the next generation toward fixing that specific issue.

**Q3: What's the main production risk with Reflexion, and how do you bound it?**
A: Each reflection cycle is a full extra attempt — extra tokens, tool calls, and latency — so unbounded reflection has the same runaway-cost risk as an unbounded ReAct loop. Production systems cap the number of reflection cycles (commonly 2-3), since returns diminish quickly: a failure that resists a couple of targeted fixes usually needs a different tool or human intervention, not more reflection.

**Q4: You're building a Plan-and-Execute agent and the planner keeps producing 2-step plans like "1. Do the research 2. Write the report." What's wrong and how do you fix it?**
A: The plan is too coarse — all the real work and adaptiveness happens inside a single step, so you get none of Plan-and-Execute's benefits (auditability, parallelism, upfront cost estimate) while still paying for an extra planning LLM call. Fix by prompting the planner to decompose into independently verifiable sub-goals — steps granular enough that each could be handed to a separate worker with a clear success/fail signal, but not so fine-grained (e.g., "click button") that a single unexpected observation invalidates the whole plan.

**Q5: Can ReAct and Plan-and-Execute be combined? How?**
A: Yes, and this is the common production pattern — Plan-and-Execute provides the top-level structure (an upfront, inspectable, potentially parallelizable list of sub-goals), while each individual step is executed using an internal ReAct loop (Thought→Action→Observation) to handle the local adaptiveness needed within that step. This gets you auditability and parallelism at the macro level, and adaptive error recovery at the micro level.

---

## 7. Summary Card

- **ReAct**: plan emerges one step at a time, fully adaptive, sequential, not inspectable upfront.
- **Plan-and-Execute**: explicit upfront plan, enables auditability/parallelism/cost estimation; often runs ReAct internally per step.
- **Reflexion**: adds self-critique after failure; reflection text becomes grounding context for the next attempt — not the same as a blind retry.
- Decision hinges on task horizon, need for adaptiveness vs. inspectability, and whether a verifiable success signal exists to make retries worthwhile.
- All three share the same runaway-cost risk pattern from Day 2/3 — always bound iteration/reflection counts in production.

---
*Next: Day 5 — Memory Systems (short-term, long-term/vector store, episodic memory).*
