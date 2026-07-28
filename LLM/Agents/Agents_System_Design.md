# Agents — System Design (Master Notes, Maximum Depth)

## 0. Why agent system design is a distinct interview skill

Everything in Modules 1-9 gives you the vocabulary and mechanisms. System design tests whether you can **assemble them under real constraints** — ambiguous requirements, cost/latency/safety tradeoffs, and the judgment to know which pieces of the toolkit a specific problem actually needs (not "use everything you know"). This module gives you a reusable framework to structure any agent system design answer, then applies it fully to several realistic prompts.

---

## 1. The reusable framework — apply this structure to any agent design question

### Step 1: Clarify scope and requirements (2-3 min, don't skip this)
- What's the actual task boundary? (e.g., "book a flight" — does this include payment? Multi-passenger? International?)
- What are the stakes of a mistake? (Informational answer vs. an irreversible side effect like a purchase or a sent email — this single question should shape almost everything downstream, especially confirmation/guardrail design.)
- Latency/cost constraints? (Real-time chat vs. an async batch job change which techniques from Module 5 are affordable.)
- What's already available? (Existing APIs/tools, existing user data/memory, or building from scratch?)

**Why this step matters for signal**: jumping straight to architecture without clarifying stakes is the single most common way strong candidates lose points — a design that's fine for "summarize this document" is dangerously incomplete for "issue a refund."

### Step 2: Define the core loop
- Single agent (ReAct, Module 4) or does the task decompose into genuinely distinct roles (multi-agent, Module 7)?
- Does the task have real branching/backtracking structure requiring Tree-of-Thought/MCTS (Module 5), or is it linear enough for plain ReAct?
- What's the stopping condition (Module 1)? Explicit final-answer signal, external verification, or both?

### Step 3: Define tools
- What actions does the agent need? For each: is it read-only or side-effecting (Module 2's idempotency/confirmation distinction — flag every side-effecting tool explicitly)?
- Schema design — keep descriptions clear enough to minimize Module 2's schema-mismatch failure mode.

### Step 4: Define memory
- Does this task need memory across sessions at all, or is working memory (Module 6) within a single task sufficient?
- If cross-session: what's genuinely episodic (specific past interactions) vs. semantic (durable facts/preferences) — don't default to "just store everything," name the distillation/consolidation step (Module 6, Section 5).

### Step 5: Define guardrails
- Confirmation steps before side-effecting actions.
- Max-iteration/timeout caps (Module 1).
- Escalation path to a human when the agent is uncertain or stuck (Module 1's stopping condition, Section 1's loop-divergence fix).
- Any prompt-injection-relevant surface (does the agent process untrusted external content? Flag it, per the Issues/Diagnosis notes).

### Step 6: Define evaluation
- What's the success metric, and what's the risk of it being gamed (Module 8, and the reward-hacking entry in Issues/Diagnosis)?
- Step-level diagnostics (tool-call accuracy, step efficiency) in addition to end-to-end success rate (Module 8).
- How many trials needed given compounding-error noise, and does the environment need to be made deterministic/replayable for fair testing (Module 8, Section 1)?

**The one-sentence framing to open any system design answer with**: "I'll walk through requirements/stakes, the core loop structure, tools, memory, guardrails, and evaluation, in that order — flagging tradeoffs as I go rather than assuming one 'correct' architecture."

---

## 2. Worked Example 1 — Customer Support Agent (refunds, account lookups, escalation)

### Requirements/stakes
Mixed stakes: account lookups are read-only/low-risk; refunds are side-effecting and financially consequential; some queries need human escalation (angry customers, ambiguous policy edge cases). Latency matters (real-time chat), moderate cost sensitivity (high volume).

### Core loop
Single ReAct agent for the common case — the task is largely linear (look up account → understand issue → decide action → act), not branching enough to justify ToT. **Explicit escalation as a first-class "action"**, not an afterthought: the agent should be able to choose "escalate to human" as a valid Action at any point, particularly if uncertainty is high or the situation matches known escalation triggers (angry sentiment, policy ambiguity, repeated failed resolution attempts — directly reusing the loop-divergence detection logic from the Issues notes).

### Tools
- `lookup_account(account_id)` — read-only.
- `get_order_history(account_id)` — read-only.
- `issue_refund(order_id, amount, reason)` — **side-effecting, high stakes**. Requires: idempotency key, a confirmation step (either explicit user confirmation, or a policy-based auto-approval threshold — e.g., auto-approve refunds under $50 matching a clear policy match, escalate/require human sign-off above that or for ambiguous cases).
- `escalate_to_human(summary, reason)` — hands off with full context, not a bare "I can't help."

### Memory
Semantic memory: durable account-level facts (subscription tier, known prior issues) retrieved at conversation start. Episodic: log of past support interactions, retrievable if the user references "like last time." No need for procedural memory here — this isn't a skill-acquisition task.

### Guardrails
Refund amount thresholds gating auto-approval vs. escalation (a deterministic, non-model-reasoning-dependent policy check layered on top of the agent's own judgment — never trust the agent's own sense of "this seems fine" as the sole gate for money movement). Idempotency keys on `issue_refund`. Max-iteration cap with automatic escalation if exceeded (directly the loop-divergence fix from the Issues notes, applied as a designed safety net rather than left to chance).

### Evaluation
Task success rate (issue actually resolved, verified via follow-up or ticket-closure-with-no-reopen, *not* just "ticket marked closed" alone — directly avoiding the reward-hacking failure mode named in the Issues notes). Tool-call accuracy on refund amount/reason correctness specifically (highest-stakes tool). Escalation-appropriateness as a separate metric (both over-escalation, which is a cost/UX problem, and under-escalation, which is a risk problem, tracked separately).

---

## 3. Worked Example 2 — Coding Agent (fixes GitHub issues, SWE-bench-style)

### Requirements/stakes
Low real-world side-effect risk if properly sandboxed (code changes proposed as a PR, not auto-merged) — this framing choice (propose vs. auto-execute) is itself a key design decision worth stating explicitly. Latency less critical than correctness (can afford more compute per task, e.g., ToT-style exploration of fix approaches). External verification available (test suite) — a genuinely favorable case for Module 5/8's "external verification is more reliable than self-assessed success" principle.

### Core loop
ReAct as the base, but specifically **worth escalating to Tree-of-Thought or Reflexion** given (a) genuine branching structure (multiple plausible fix approaches for a given bug, some leading to dead ends when they don't actually fix the failing test) and (b) a clean external verifier (the test suite) making Reflexion's failure-diagnosis step unusually reliable — this is a strong, concrete case for combining both techniques rather than picking one, and stating that combination explicitly (per Module 5's closing point) is a strong signal.

### Tools
- `read_file(path)`, `search_codebase(query)` — read-only.
- `run_tests(test_path)` — read-only in effect (doesn't modify state), but the critical **evaluation** tool — this is the external verification signal driving both the ReAct loop's stopping condition and any Reflexion retry.
- `write_file(path, content)` / `apply_patch(diff)` — side-effecting *within the sandbox*, but not side-effecting in the real-world-consequence sense if properly isolated (a sandboxed repo clone, not the live repository) — flag this distinction explicitly, since it changes how aggressively you can allow retries/exploration compared to Example 1's refund tool.
- `open_pull_request(...)` — the actual real-world side-effecting action; this is where confirmation/review gates belong, not on the intermediate file-editing tools.

### Memory
Procedural memory is genuinely relevant here (Module 6, Section 4) — a growing library of previously-successful fix patterns/utility functions for this specific codebase, directly reusable across future issues, rather than every issue starting from a blank slate. Episodic/semantic less central than in Example 1 — this task is more about within-task iteration than cross-session personalization.

### Guardrails
Sandbox isolation (never operate directly on the production repository). Max-iteration cap on the fix-attempt loop (a coding agent looping indefinitely on a hard bug is a real, common failure). PR-based human review gate before any real merge, regardless of how confident the agent's own self-assessment is (directly reusing Module 8's calibration point — the agent believing it succeeded isn't the same as it having succeeded, and external human review is a stronger check than trusting a test suite alone, especially for tests the agent itself could have gamed, per the reward-hacking entry).

### Evaluation
Task success = tests pass **and** the fix doesn't game the test suite in a hollow way (the same reward-hacking check named in the Issues notes — spot-check that fixes are substantively correct, not just technically test-passing via a shortcut like hard-coding expected output). Step efficiency matters (fewer fix-attempt iterations = lower compute cost per resolved issue). This is a clean case for the "controlled/replayable environment" point from Module 8 — a sandboxed repo state gives you exactly the deterministic environment needed for fair, repeatable trial-based evaluation.

---

## 4. Worked Example 3 — Multi-Source Research Assistant (gathers info, writes synthesized report)

### Requirements/stakes
Informational stakes (no side effects), but **accuracy/hallucination risk is the primary concern** (directly LLM Basics Module 8's hallucination material, now at the agent level) — a fabricated-but-fluent-sounding synthesized claim is the main failure to design against, not an irreversible action.

### Core loop
This is the clearest case for **multi-agent orchestrator-worker** (Module 7): an orchestrator decomposes the research question into sub-questions, dispatches each to a worker agent (each doing its own search/retrieval ReAct loop), then synthesizes — directly because the sub-questions genuinely benefit from separate, focused contexts (each worker doesn't need the others' full search history cluttering its context) rather than one agent juggling everything in one long transcript.

### Tools
- `web_search(query)`, `fetch_page(url)` — read-only, available to worker agents.
- No side-effecting tools needed for this task shape.

### Memory
Minimal need for persistent cross-session memory unless this is a recurring research assistant for the same user/topic over time (in which case semantic memory of established, previously-verified facts would avoid redundant re-research). Working memory is the dominant concern here instead — each worker's context should stay focused (Module 6's context-management, Module 7's private-context pattern) rather than accumulating the entire research session's raw search history.

### Guardrails
**This is the design where the Issues notes' multi-agent synthesis and hallucination entries matter most**: require the orchestrator's synthesis to cite which worker/source each claim came from (directly the "explicit attribution" fix from the Issues notes' multi-agent-miscommunication entry) — this both reduces misattribution/conflation risk and gives a human reviewer a way to spot-check the final report against sources. Consider a dedicated fact-checking/critique pass (a debate-style or Reflexion-style verification agent, Module 5/7) specifically checking the synthesized report's claims against the original worker outputs before finalizing, given how consequential unflagged hallucination is for this specific task.

### Evaluation
Primarily human eval or LLM-as-judge (LLM Basics Module 8) on factual accuracy of the final synthesized report against verifiable sources — this is a case where automated benchmark-style success metrics are weak (open-ended synthesis quality is hard to score with exact-match methods), so the evaluation design itself needs to lean on the qualitative methods from LLM Basics Module 8, explicitly acknowledging their known biases (length/verbosity bias would be a real risk here, since a longer, more detailed-looking synthesized report could be over-rewarded regardless of actual accuracy).

---

## 5. Cross-example patterns worth naming explicitly in any design interview

- **Stakes determine architecture more than task complexity does** — the refund tool in Example 1 gets far more guardrail attention than the more "interesting" reasoning-heavy coding task in Example 2, precisely because side-effecting, hard-to-reverse actions warrant it regardless of how technically sophisticated the surrounding reasoning is.
- **External verification, when available, should be leaned on hard** — Example 2's test suite is the strongest evaluation/stopping-condition signal across all three examples, precisely because it doesn't depend on the model's own self-assessment (Module 8's calibration point). When a design question doesn't obviously have one, it's worth explicitly asking/noting whether one could be constructed, rather than defaulting straight to LLM-as-judge.
- **Multi-agent is a response to context/role separation needs, not a default "more sophisticated" choice** — Example 3 needs it because sub-questions genuinely benefit from separated contexts; Example 1 explicitly does *not* use multi-agent, because the support-conversation task doesn't have that same separation benefit — naming why you're *not* using a technique is as valuable a signal as correctly using one.
- **Every side-effecting tool needs an explicit sentence about idempotency/confirmation** — this is the single most consistently-tested guardrail detail across realistic agent system design questions, and it's cheap to always mention.

---

## 6. Quick-fire Q&A (self-test)

**Q: What's the single most important clarifying question to ask at the start of any agent system design interview, and why?**
A: What are the stakes of a mistake — specifically, does the task involve irreversible, real-world side effects (payments, sent messages, deletions) versus purely informational output. This single distinction shapes nearly every downstream architecture decision, especially guardrail and confirmation design, more than almost any other requirement.

**Q: Why is the coding-agent example ("Worked Example 2") an unusually favorable case for combining Tree-of-Thought with Reflexion?**
A: It has both genuine branching structure (multiple plausible fix approaches, some leading to dead ends) and a clean, reliable external verifier (the test suite) — making Reflexion's failure-diagnosis step unusually trustworthy (not dependent on the model's own possibly-miscalibrated self-assessment) while ToT's branching handles exploring alternative fix approaches within or across attempts.

**Q: In the research-assistant example, why is requiring explicit source attribution in the orchestrator's synthesis step specifically important?**
A: It directly mitigates the multi-agent-synthesis-error and hallucination risks named in the Issues/Diagnosis notes — forcing the orchestrator to cite which worker/source each claim came from reduces misattribution/conflation risk and gives a human reviewer a concrete way to spot-check the final report against original sources, rather than trusting an unattributed synthesized narrative.

**Q: Why does the customer-support example explicitly avoid a multi-agent architecture, when the coding and research examples use one?**
A: Multi-agent architectures pay off when subtasks genuinely benefit from separated contexts/roles that would conflict or dilute each other in one agent's context — the support conversation is a largely linear, single-context task (look up account, understand issue, act) without that separation benefit, so a single ReAct agent with explicit escalation-as-an-action is both sufficient and avoids unnecessary multi-agent coordination overhead.

**Q: Across all three worked examples, what's the one guardrail detail that should always be mentioned for any side-effecting tool, regardless of the specific task?**
A: An explicit statement of idempotency-key usage and/or a confirmation step before execution — this is the single most consistently-relevant guardrail detail in realistic agent system design, and omitting it is a common way to lose points even when the rest of the architecture is well-reasoned.

---
*End of Agents System Design notes.*
