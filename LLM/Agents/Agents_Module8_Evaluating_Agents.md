# Agents Module 8 — Evaluating Agents (Master Notes, Expanded)

## 0. Why this is a genuinely harder problem than single-turn LLM evaluation

LLM Basics Module 8 covered evaluating a single generation (benchmarks, human eval, LLM-as-judge, hallucination) — a well-defined, bounded problem: one input, one output, compare against a reference or a preference judgment. Agent evaluation inherits every one of those challenges **and adds new ones specific to multi-step, environment-interacting behavior**: the output isn't one generation, it's an entire **trajectory** (a sequence of Thoughts, Actions, Observations spanning many turns), success can depend on the specific state of an external environment (which may itself be stochastic or change between runs), and a single wrong step deep in a long trajectory can invalidate an otherwise-good overall approach — evaluation has to account for the *process*, not just the final output, in a way single-turn evaluation never had to.

---

## 1. Why long-horizon, compounding-error tasks are specifically hard to evaluate

### The core statistical problem, connecting directly back to Module 1's foundational math
Recall Module 1's compounding-error framing: if each step in an N-step task has independent success probability `p`, overall task success is roughly `p^N` — for long trajectories (large N), even a high per-step accuracy `p` yields a much lower overall success rate, and **small changes in per-step accuracy produce large, nonlinear changes in overall success rate** (exactly the same mathematical shape as LLM Basics Module 3's emergent-abilities compounding-metric discussion — worth explicitly naming that parallel if asked, since it's the same underlying mechanism: an all-or-nothing outcome metric built on top of a chain of probabilistic steps).

**Numerical example of why this makes evaluation noisy and hard to interpret**: suppose two candidate agent designs have true per-step accuracies of 88% and 91% respectively, on a 10-step task:
```
Design A: 0.88^10 ≈ 27.9% task success rate
Design B: 0.91^10 ≈ 38.9% task success rate
```
A genuinely meaningful 3-percentage-point per-step accuracy improvement translates into an 11-point overall success rate difference — but if you only ran, say, 20 evaluation trials of each design, the resulting observed success counts (roughly 5-6 successes for A, 7-8 for B out of 20) would have **enough sampling noise that the difference might not even look statistically significant** despite reflecting a real, meaningful underlying capability gap — this is a concrete, quantifiable reason why agent evaluation typically requires **many more trial runs** than single-turn evaluation to reach reliable conclusions, and why reported agent benchmark numbers from small trial counts deserve real skepticism.

### Environment variability and non-determinism
Unlike a fixed, static single-turn benchmark question, an agent's environment can itself be **stochastic or change between runs** (a live web search returning different results at different times, a simulated environment with randomized initial conditions, another agent's non-deterministic responses in a multi-agent setup) — meaning the *same* agent design can genuinely succeed on one run and fail on an identically-intentioned rerun, purely due to environment variability, not a change in the agent's actual capability. This means agent evaluation needs **either a controlled/deterministic environment for fair comparison** (a fixed, replayable environment/sandbox), or enough repeated trials to average out genuine environment-driven variance from actual capability differences — conflating the two (attributing environment-driven variance to a capability difference, or vice versa) is a common, real evaluation mistake.

---

## 2. Task Success Rate

### The metric
The most direct measure: fraction of evaluation episodes where the agent achieved the correct/desired final outcome, as judged by some ground-truth criterion (could be exact-match on a known-correct final answer, a programmatic check like "did the code pass the test suite," or a human/LLM-judge assessment for more open-ended tasks — directly reusing LLM Basics Module 8's human-eval and LLM-as-judge machinery, just applied to a full trajectory's final outcome rather than a single generation).

### The key limitation — success rate alone hides *why* an agent fails
A raw success rate number tells you nothing about **where in the trajectory things typically go wrong** — an agent that fails because of poor initial planning (bad Module 5 decomposition) needs a very different fix than an agent that plans well but fails due to unreliable tool-call argument formatting (Module 2) or getting stuck in unproductive loops (Module 4's loop-divergence failure mode). This is exactly why practical agent evaluation almost never stops at a single aggregate success-rate number — it's paired with more diagnostic, step-level metrics.

---

## 3. Step Efficiency

### The metric
Given that a task was completed successfully, **how many steps/tool calls/tokens did it take** relative to some baseline or the theoretical minimum needed — a real practical cost dimension distinct from raw success/failure, since two agent designs might have identical success rates but very different real-world serving costs if one takes far more loop iterations (and thus far more LLM calls, each with real latency and compute cost, per LLM Basics Module 7's serving-cost material) to reach the same successful outcome.

### Why this matters even when success rate is the primary headline metric
An agent design that succeeds 90% of the time but averages 15 tool calls per task is meaningfully worse in practice (higher latency, higher API/compute cost per completed task) than one that also succeeds 90% of the time but averages 5 tool calls — this is a genuine engineering tradeoff worth explicitly raising in an interview answer about agent evaluation, since "just maximize success rate" is an incomplete optimization target for a real production system where cost and latency matter alongside correctness.

---

## 4. Tool-Call Accuracy

### The metric
Specifically measures whether the agent's **individual tool-call decisions** were correct — did it call the right tool for the situation, with correctly-formatted and semantically-correct arguments (directly connects to Module 2's schema-validation and error-handling material) — independent of whether the overall task ultimately succeeded or failed. This is a genuinely useful **diagnostic** metric precisely because it isolates one specific failure surface (tool selection/argument correctness) from the broader, harder-to-decompose question of overall task success, letting you determine whether a low overall success rate is being driven specifically by unreliable tool use vs. some other stage of the pipeline (planning, reasoning over observations, etc.).

### Concrete numerical framing
If tool-call accuracy is measured at, say, 95% per individual call, but overall task success (requiring, say, 8 correct tool calls in a sequence, ignoring other potential failure points for simplicity) is only:
```
0.95^8 ≈ 66.3%
```
This immediately tells you: **even a quite reliable 95%-accurate tool-calling agent will still fail roughly a third of longer multi-tool tasks purely from tool-call error compounding** — a genuinely important, easy-to-underestimate number to have ready, since it demonstrates why "our tool-calling accuracy looks great in isolation" doesn't automatically translate into a correspondingly great end-to-end task success rate on longer tasks, directly reinforcing this module's opening point about compounding-error math driving the whole evaluation-difficulty problem.

---

## 5. Benchmark Suites for Agents (brief — know these exist and their general shape)

A few commonly-referenced categories worth naming if asked, without needing deep familiarity with any single one:
- **Tool-use/API benchmarks** (e.g., ToolBench-style benchmarks): test whether an agent correctly selects and calls tools (often many available tools, some deliberately irrelevant/distractor tools included) to complete a task — directly targets Section 4's tool-call-accuracy dimension in a controlled, scored setting.
- **Web/software-interaction benchmarks** (e.g., WebArena-style, SWE-bench-style): place the agent in a realistic, often sandboxed environment (a simulated website, an actual code repository with a real bug to fix) and measure task completion against a ground-truth outcome (did the agent successfully complete the web task, does the code pass the held-out test suite) — these directly address the "environment variability, need for a controlled/replayable environment" point from Section 1, since a good agent benchmark specifically constructs a *deterministic, replayable* environment precisely to make fair comparison possible despite the general problem of environment non-determinism.
- **Multi-step reasoning/planning benchmarks**: puzzle-style or planning-style tasks (games, logic puzzles) specifically designed to have genuine branching/backtracking structure, directly relevant for evaluating Module 5's planning/search techniques (ToT, MCTS) rather than simpler linear ReAct-sufficient tasks.

**Interview-level point to make if this comes up**: name the *category* of benchmark and what specific failure surface it targets (tool accuracy vs. environment task completion vs. planning under branching structure) rather than trying to recite exact benchmark names/numbers from memory — the categorical understanding is what's actually being tested, and it directly maps back onto this module's own success-rate/step-efficiency/tool-accuracy metric breakdown.

---

## 6. Side-by-side summary table (memorize this cold)

| | Task Success Rate | Step Efficiency | Tool-Call Accuracy |
|---|---|---|---|
| What it measures | Did the agent achieve the correct final outcome | Cost (steps/calls/tokens) to reach a successful outcome | Correctness of individual tool-call decisions, independent of overall outcome |
| Diagnostic value | Low alone — doesn't reveal *why* failures happen | Reveals real-world cost/latency tradeoffs between designs with similar success rates | Isolates one specific failure surface (tool selection/formatting) from others (planning, reasoning) |
| Sensitive to compounding-error math | Yes — the headline number most affected by per-step error compounding over long trajectories | Indirectly (more steps = more compounding-error exposure) | Yes at the individual-call level; also compounds across a multi-tool trajectory |

---

## 7. Quick-fire Q&A (self-test)

**Q: Using the compounding-error framing, explain why agent evaluation needs more trial runs than single-turn LLM evaluation to detect a real, meaningful capability difference.**
A: Small per-step accuracy differences translate into large, nonlinear differences in overall trajectory success rate (per-step accuracy raised to the power of the number of steps) — but with few evaluation trials, sampling noise in the observed success counts can obscure even a real, meaningful underlying per-step capability gap, requiring many more trials to reach statistically reliable conclusions than a single-turn benchmark question would.

**Q: Why does environment non-determinism specifically complicate agent evaluation in a way that doesn't apply to standard single-turn benchmarks?**
A: An agent's environment (live web results, stochastic simulations, other non-deterministic agents) can itself vary between runs, meaning the same agent design can genuinely succeed on one run and fail on an identically-intentioned rerun purely from environment variability — requiring either a controlled/replayable environment for fair comparison, or enough repeated trials to separate genuine capability differences from environment-driven noise.

**Q: Why is task success rate alone considered an insufficient agent evaluation metric, even though it's the most direct measure of "did it work"?**
A: It reveals nothing about *where* in a trajectory failures typically occur — an agent failing from poor initial planning needs a very different fix than one failing from unreliable tool-call formatting or getting stuck in loops, so success rate is typically paired with more diagnostic step-level metrics (step efficiency, tool-call accuracy) to actually locate the failure surface.

**Q: Give a concrete numerical example of why high individual tool-call accuracy doesn't guarantee a correspondingly high end-to-end multi-tool task success rate.**
A: A 95%-per-call tool-calling accuracy still yields only about 0.95^8 ≈ 66.3% end-to-end success for a task requiring 8 correct sequential tool calls — individually-high per-call accuracy still compounds into a substantially lower overall trajectory success rate over a longer tool-call sequence.

**Q: Why does step efficiency matter as a metric even when comparing two agent designs with identical success rates?**
A: Two designs with the same success rate can have very different real-world serving costs if one requires substantially more loop iterations/tool calls (and thus more LLM calls, latency, and compute cost) to reach the same successful outcome — "maximize success rate alone" is an incomplete optimization target for a real production system where cost and latency matter alongside raw correctness.

**Q: What should you name if asked to categorize agent benchmark types, rather than trying to recall specific benchmark details?**
A: The general categories and what failure surface each targets — tool-use/API benchmarks (tool-call accuracy, often with distractor tools), web/software-interaction benchmarks in controlled/replayable sandboxed environments (end-to-end task completion despite general environment non-determinism), and multi-step reasoning/planning benchmarks with genuine branching structure (relevant specifically to evaluating Tree-of-Thought/MCTS-style techniques rather than simpler linear tasks).

---
*End of Agents Module 8 (expanded). Next: Module 9 — Interview Synthesis (cross-module Q&A and system-design-style questions spanning the full Agents syllabus).*
