# Day 23: Failure Modes Catalog — Infinite Loops, Tool Misuse, Hallucinated Actions, Cascading Errors

## 1. The Intuition First

Days 21-22 showed HOW to build these systems. Today is the deliberate inversion: **given a working-looking system, what specifically breaks, and how do you recognize it fast?** This is a distinct interview skill from system design — "design X" tests synthesis; "here's a system, what could go wrong, and how would you detect/prevent it" tests diagnostic instinct, and interviewers often probe both in the same conversation. Today catalogs the four failure families that recur across virtually every agentic system, each one a named pattern you should recognize instantly, not re-derive from scratch under interview pressure.

Think of this like a pilot's pre-flight failure checklist — not "here's how planes work" (you already know that), but "here are the specific things that go wrong, their warning signs, and the standard response to each." Fluency here is what separates "I could probably debug this eventually" from "I immediately recognize this pattern and know the fix."

---

## 2. Failure Mode 1: Infinite / Runaway Loops

**What it is**: the agent's own "am I done" judgment (Day 2's stopping condition) never triggers, and the ReAct/Reflexion/ToT loop continues indefinitely (or until an external cap forces termination) — first introduced in Day 2 §5.2, and recurring at every subsequent layer (Day 4's unbounded Reflexion, Day 9's unbounded search-retry, Day 16's token-budget necessity).

**Recognizable signatures**:
- Near-identical or exactly-repeated tool calls across consecutive turns (the agent re-searching the same query, re-checking the same fact).
- Thought text that verbalizes uncertainty about stopping ("let me verify this once more...", "just to be extra sure...") repeated across many turns without new information being gathered.
- Monotonically increasing iteration count with no corresponding progress in the underlying task state (Day 19's structured state tracking makes this MUCH easier to detect than narrative-only context — "hypotheses_tested" not growing across 10 turns is a clear, structured signal).

**Standard fixes, layered (echoing Day 14's defense-in-depth structure)**:
1. Hard iteration cap (Day 2 §5.2) — the non-negotiable backstop.
2. Explicit stopping criteria in the system prompt ("if you've confirmed a fact from one reliable source, that's sufficient").
3. Remaining-budget injection into context (Day 16 §2.1) — let the model's own reasoning account for the constraint.
4. Repetition/loop detection — programmatically compare the last K actions for near-duplication and force termination or escalate (Day 11) if detected, rather than waiting for the hard cap.

**The interview-precise framing**: *"This isn't one bug to fix — it's a failure mode every agentic loop is structurally exposed to, so the fix is always layered: a hard cap that can never be exceeded, plus softer signals (explicit criteria, budget visibility) that ideally prevent hitting the cap in the first place."*

---

## 3. Failure Mode 2: Tool Misuse

**What it is**: the agent calls a real, correctly-defined tool, but with the WRONG arguments, at the WRONG time, or in a way that technically executes successfully but doesn't serve the actual task — distinct from Day 3 §4's malformed calls (which fail to parse); tool misuse calls succeed mechanically but are semantically wrong.

**Three common sub-patterns, each worth naming individually**:

**a) Wrong tool selected for the task** (a description/schema problem, Day 3 §2.2): the agent calls `search_web` when it should have called `search_internal_docs`, because the tool descriptions weren't distinctive enough for the model to reliably choose correctly. Fix: sharpen tool descriptions with explicit trigger conditions, and/or reduce the number of superficially-similar tools available at once (Day 3 §5.1's tool-subsetting, now serving an accuracy goal, not just a cost goal).

**b) Right tool, wrong arguments** — e.g., calling `get_weather(city="Paris")` when the user meant Paris, Texas, not Paris, France, because the agent didn't disambiguate before acting. Fix: for genuinely ambiguous arguments, the agent should either ask a clarifying question (if a human is present to ask) or explicitly reason through disambiguation using available context BEFORE calling the tool, not after — this is a prompt/reasoning-quality fix, not an infrastructure fix.

**c) Right tool, right arguments, wrong TIMING** — e.g., calling `issue_refund` before actually verifying the customer's complaint is valid (skipping a verification step in the reasoning chain, even though each individual step in isolation looks correct). This is the hardest sub-pattern to catch mechanically, because nothing about the individual tool call LOOKS wrong — it's the SEQUENCE that's flawed. This is exactly why Day 18's trajectory evaluation exists: outcome-only checking often can't catch this at all if the premature refund happens to have been for a valid complaint anyway (the "right for the wrong reasons" pattern from Day 18 §2.3, now instantiated as a specific tool-misuse case).

**Worked example distinguishing (b) and (c)**:
```
Ticket: "I was charged twice for my subscription last month."

MISUSE (b) — wrong arguments: Action: issue_refund(order_id=<this month's order>, amount=$50)
  [wrong order — should have checked LAST month's charges, not this month's]

MISUSE (c) — wrong timing: Action: issue_refund(order_id=<correct order from last month>, amount=$50)
  [correct order and amount, but called BEFORE checking get_billing_history to confirm
  a double-charge actually occurred — got lucky that it happened to be true this time]
```

---

## 4. Failure Mode 3: Hallucinated Actions

**What it is**: the agent's reasoning (Thought text) confidently asserts that an action was taken, or that a fact was verified, when it WASN'T — distinct from a hallucinated FACT (a classic LLM issue), this is specifically hallucinating about the agent's OWN behavior/state.

**Two distinct sub-cases**:

**a) Hallucinated tool call** — the model's Final Answer describes having done something ("I've processed your refund") when no corresponding tool call actually appears in the trajectory. This is a genuinely dangerous failure mode because it produces a CONFIDENT, well-formed, false claim about system state — a customer told "your refund has been processed" when it hasn't been is worse than an honest "I wasn't able to process this" in almost every real scenario.

**b) Hallucinated verification** — the model's Thought claims to have confirmed something ("I've checked the knowledge base and confirmed this is correct") without a corresponding OBSERVING step that actually returned that confirmation — i.e., the model asserts grounding that doesn't exist in the trace. This connects directly to Day 18's trajectory evaluation: "does every claim trace back to an actual observation" (Day 18 §2.2's exact check) is PRECISELY the mechanism that catches this — a system with only outcome evaluation would have no way to detect it at all if the ungrounded claim happened to be true.

**Detection, concretely**: programmatic trace validation — for every claim in a Final Answer that references an action or a verification, check the trace for a CORRESPONDING tool_call/observation pair. This is buildable as an automated check (not just a manual eval process), directly extending Day 15's tracing infrastructure: if every ACTING state is logged with its result, you can programmatically diff "claims made" against "actions actually taken," rather than relying on a human or an LLM judge to notice the discrepancy after the fact.

**Why this is worse than a normal hallucination**: a hallucinated FACT ("the capital of X is Y" when it's wrong) is bad but often correctable by the user checking elsewhere. A hallucinated ACTION ("I've refunded you") creates a false belief about SYSTEM STATE that the user has no independent way to verify except by later discovering it's false — which is exactly the shape of failure that erodes trust in an agentic system fastest and most durably.

---

## 5. Failure Mode 4: Cascading Errors

**What it is**: an early, small error compounds through subsequent reasoning steps, each one building on the (wrong) output of the previous one, producing a final result that's dramatically wrong even though each INDIVIDUAL step, viewed in isolation, seems like a locally reasonable continuation of what came before.

**Where this specifically shows up, connecting to earlier days**:
- **Within a single agent's trajectory**: Day 2's multi-hop chaining (Oppenheimer → Nolan → London) is a FEATURE when each hop is correct — the same structure becomes a cascading-error liability the moment ONE hop is wrong, since every subsequent hop builds on it without re-verifying the foundation.
- **In multi-agent systems (Day 8 §5.2, revisited)**: a worker's confidently-wrong summary gets treated as ground truth by the orchestrator, which then synthesizes a final answer that inherits and often AMPLIFIES the original error (the orchestrator may add its own confident framing on top of an already-wrong input, making the final error even more convincing-sounding than the original mistake).
- **In long-running state (Day 19, revisited)**: an error in an early summarization step (a fact dropped or misrepresented during compression) propagates forward, since later reasoning steps trust the summary as ground truth without re-verifying against the (now-compressed-away) original source.

**Worked example — cascading error in a research agent (Day 22-style)**:
```
Turn 3: Thought: "The config file shows timeout=30, so the issue must be a timeout
  during high load." [FLAWED: misread the config, actual value was 300, not 30]

Turn 4: Thought: "Given the confirmed timeout misconfiguration, I'll focus my fix
  on the timeout handling logic." [Built entirely on Turn 3's wrong premise]

Turn 5: Thought: "Since we've established this is a timeout issue, the fix should
  increase the timeout value and add retry logic." [Compounding further — now
  proposing a WRONG fix with increasing confidence, purely from internal consistency
  with its own earlier, unverified claim]
```
Notice: turns 4 and 5 are each individually "reasonable continuations" of turn 3 — the cascading failure isn't visible by looking at any single turn's LOCAL reasoning quality, only by checking turn 3's claim against the actual observation (Day 18's trajectory-eval principle again: "does every claim trace to an actual observation" — turn 3's claim of `timeout=30` should be checked against the literal tool output, not taken at face value by subsequent turns).

**Mitigation approaches**:
- **Re-verification at decision points**: before committing significant effort based on an early finding, especially one that becomes load-bearing for subsequent steps, explicitly re-check it against the source rather than trusting the earlier Thought's summary of it — costs some efficiency but bounds cascading risk.
- **Confidence-weighted propagation**: if an early claim was itself uncertain/low-confidence, subsequent reasoning should carry that uncertainty forward explicitly ("assuming the timeout hypothesis is correct...") rather than treating it as settled fact — makes the cascade's foundation visible rather than silently hardening into certainty.
- **Multi-agent cross-checking (Day 8's debate pattern, applied defensively)**: for especially consequential trajectories, having a second, independent pass re-derive the same early finding catches exactly this kind of single-path compounding error, the same mechanism Day 8 §4 showed catching the database-migration mistake.

---

## 6. Interview Q&A

**Q1: What's the difference between a malformed tool call (Day 3) and tool misuse?**
A: A malformed call fails to parse or execute at all — wrong argument types, a hallucinated tool name — and the failure is immediately visible as an execution error. Tool misuse is a call that executes successfully and mechanically correctly, but is semantically wrong: the wrong tool was chosen for the task, the right tool was called with wrong arguments due to unresolved ambiguity, or the right tool/arguments were used at the wrong point in the reasoning sequence (e.g., skipping a verification step). Tool misuse is harder to catch because nothing about the call itself looks broken.

**Q2: Why is a hallucinated action worse than a hallucinated fact?**
A: A hallucinated fact is often independently checkable by the user and, while bad, doesn't necessarily create a false belief about something only the system can verify. A hallucinated action — the agent confidently claiming it processed a refund or verified something when no corresponding tool call or observation exists in the trace — creates a false belief about actual system state that the user typically has no way to detect except by later discovering it's false, which is a much more durable trust failure.

**Q3: How would you detect hallucinated verification claims programmatically, not just through manual review?**
A: By validating every claim in a Final Answer that references a verification or action against the actual trace — checking that a corresponding tool_call and OBSERVING result genuinely exists for each claim made. This is a direct extension of Day 15's tracing infrastructure and Day 18's trajectory evaluation principle ("does every claim trace to an actual observation") — buildable as an automated check that diffs claims against actions actually logged, rather than relying on manual or LLM-judge review to happen to notice the gap.

**Q4: Explain cascading error and why it's hard to catch by reviewing individual reasoning steps in isolation.**
A: An early error becomes the unverified foundation for subsequent reasoning steps, each of which is a locally reasonable continuation of what came before — meaning each individual step, viewed on its own, looks like sound reasoning. The actual flaw is only visible by checking the EARLY claim against its source observation, not by evaluating any single downstream step's internal logic, since downstream steps are correctly reasoning FROM a premise that was never actually verified.

**Q5: What's a mitigation for cascading errors that doesn't require catching the error at its original source?**
A: Multi-agent cross-checking, applied defensively — having a second, independent reasoning path re-derive the same early finding rather than relying on a single trajectory's self-consistency. Since a single agent's cascading error is often internally consistent (each step logically follows from the last), an independent second pass that doesn't share the same reasoning path is more likely to catch a discrepancy at the source, the same mechanism Day 8's debate pattern used to catch the database-migration mistake.

**Q6: An agent produces a correct final outcome despite an internally cascading reasoning error partway through (the error happened to cancel out). Would outcome evaluation catch this?**
A: No — this is exactly Day 18's "right for the wrong reasons" quadrant, and it's a real risk specifically because outcome evaluation reports a pass, hiding a process that will very likely fail on the next case where the error doesn't happen to cancel out. Only trajectory evaluation, checking whether each claim actually traces to its supporting observation, would surface this.

---

## 7. Summary Card

- **Infinite/runaway loops**: the stopping condition never triggers; layered defense (hard cap, explicit criteria, budget visibility, repetition detection) — never rely on one mechanism alone.
- **Tool misuse**: mechanically successful calls that are semantically wrong — wrong tool (fix: sharper descriptions), wrong arguments (fix: disambiguate before acting), or wrong timing (fix: trajectory eval, since individual steps look fine in isolation).
- **Hallucinated actions**: the agent claims to have done/verified something with no corresponding trace evidence — worse than a hallucinated fact because it creates an unverifiable false belief about system state; detectable via programmatic claim-vs-trace validation.
- **Cascading errors**: an early unverified claim becomes the silently-trusted foundation for locally-reasonable-looking subsequent steps; hard to catch by reviewing any single downstream step, only by re-checking the early claim against its source; mitigated by re-verification at decision points, explicit uncertainty propagation, or independent cross-checking.
- All four connect back to the same root tool: **Day 18's trajectory evaluation** — "does every claim trace to an actual observation" is the single check that catches tool-misuse-by-timing, hallucinated verification, AND cascading errors simultaneously.

---
*Next: Day 24 — Mock System Design Questions (timed practice across the patterns from Days 1-23).*
