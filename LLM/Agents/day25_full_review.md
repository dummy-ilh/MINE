# Day 25: Full Review — Rapid-Fire Q&A Bank Across All 25 Days

## 1. The Whole Curriculum, One Diagram

```
PHASE 1 (Days 1-6): FOUNDATIONS
  Day 1: agent vs. workflow (who controls the next step?)
  Day 2: ReAct — Thought/Action/Observation, looped
  Day 3: tool calls — model emits text, YOUR code executes
  Day 4: planning — ReAct / Plan-and-Execute / Reflexion
  Day 5: memory — short-term / long-term / episodic
  Day 6: consolidation

PHASE 2 (Days 7-13): ARCHITECTURES
  Day 7: single agent = explicit state machine
  Day 8: multi-agent = parallelism / isolation / specialization, or don't bother
  Day 9: agentic RAG = Day 3's loop, applied to `search`
  Day 10: Tree of Thought = explicit branching when linear reasoning isn't enough
  Day 11: human-in-the-loop = approval gates / escalation / interrupts
  Day 12: frameworks = which of Days 7-11 each one packages
  Day 13: consolidation

PHASE 3 (Days 14-20): PRODUCTION ENGINEERING
  Day 14: reliability = retries / timeouts / fallback models / circuit breakers
  Day 15: observability = traces / logs / metrics
  Day 16: cost & latency = token budgets / caching / model routing
  Day 17: guardrails = injection defense / sandboxing (data risk vs. execution risk)
  Day 18: evaluation = outcome vs. trajectory, LLM-as-judge pitfalls
  Day 19: state at scale = context pressure / summarization / checkpointing
  Day 20: consolidation

PHASE 4 (Days 21-25): SYSTEMS DESIGN & PRACTICE
  Day 21: case study (support agent) — fine-grained HITL, injection-heavy
  Day 22: case study (coding agent) — long-horizon, ToT-justified, sandboxing-heavy
  Day 23: failure modes — loops / tool misuse / hallucinated actions / cascading errors
  Day 24: mock practice — ambiguity, follow-ups, comparisons, monitoring
  Day 25: this file
```

**The single sentence that summarizes all 25 days**: an agent is a stateless model wrapped in a loop you architect (Phase 1), organized into a specific structure matched to the task's actual shape (Phase 2), hardened to survive real traffic, cost pressure, and adversaries (Phase 3), and the whole discipline is choosing which pieces a given task actually needs — never applying all of it uniformly (Phase 4).

---

## 2. The Five Ideas That Recur Most — If You Remember Nothing Else

1. **"Least complex design that solves the problem"** (Day 1, reinforced in Days 8, 10, 12, 21, 22): every escalation in sophistication — workflow→agent, single→multi-agent, ReAct→ToT, raw loop→framework — trades cost/complexity for capability/reliability, and the right amount is exactly what the task needs, never maximal by default.

2. **"The model only emits text; your code does everything else"** (Day 3, underlying Days 9, 14, 17): tool calls, retrieval, reliability mechanisms — none of it is the model "doing" anything, all of it is your architecture around a stateless text generator. This is WHY errors can be fed back as observations, why guardrails work, why state can be checkpointed.

3. **"Does every claim trace back to an actual observation?"** (Day 18, resurfacing across Days 9, 19, 23): this single check catches ungrounded RAG answers, hallucinated actions, and cascading errors simultaneously — probably the highest-leverage single diagnostic question in the entire curriculum.

4. **"No mechanism is a single point of defense"** (Day 14's retry+timeout+circuit-breaker stack, Day 17's layered injection defense, Day 23's layered loop-prevention): assume any one layer fails and design so the others still hold — this defense-in-depth instinct appears in reliability, security, and failure-prevention alike.

5. **"Justify inclusion AND exclusion"** (explicit in Days 13, 20, 21, 22, 24): a strong system-design or debugging answer never lists every pattern you know — it explains why specific patterns fit THIS task and, just as importantly, why others were deliberately left out.

---

## 3. Cumulative Rapid-Fire Q&A — 25 Questions, One Per Day

**Day 1.** What's the one-sentence test for whether a system is agentic?
*Does the model or the code control the next step — autonomy, environment interaction, and a loop driven by the model's own outputs, all three required.*

**Day 2.** Why does ReAct beat plain Chain-of-Thought on tasks needing current information?
*CoT reasons only from parametric knowledge with no way to verify it; ReAct grounds every step in a real observation, enabling mid-trajectory self-correction.*

**Day 3.** What's the model's actual responsibility when it "calls a tool"?
*Only producing structured text naming the tool and arguments — execution, error handling, and feeding results back are entirely your code's job.*

**Day 4.** When would you choose Plan-and-Execute over pure ReAct?
*Long-horizon tasks with independent sub-goals you want to inspect, parallelize, or cost-estimate before executing — not short, unpredictable tasks.*

**Day 5.** Does the model remember anything between calls?
*No — every call is stateless; "memory" is entirely external engineering deciding what to re-inject into a fresh prompt.*

**Day 6.** What's the thread connecting Days 1-5?
*The model is a stateless text generator; autonomy, tool use, planning, and memory are all external scaffolding built around repeated calls to it.*

**Day 7.** Why model an agent as an explicit state machine instead of a bare `while True` loop?
*Named states enable isolated testing, guards/timeouts, checkpointing, observability, and clean insertion of new required steps like approval gates.*

**Day 8.** What's the actual justification for multi-agent over a better single agent?
*Parallelism, context isolation, or specialization — not new capability. Absent all three, it just adds cost and hidden-worker-error risk.*

**Day 9.** What does agentic RAG add over classic RAG?
*Query reformulation for ambiguous/multi-hop queries, self-correction on insufficient results, and the ability to decide not to retrieve at all.*

**Day 10.** Why is ToT more expensive than even Reflexion?
*Reflexion pays for sequential full attempts; ToT pays for k generations AND k evaluations at every branch point, scaling combinatorially in the worst case.*

**Day 11.** Approval gate vs. escalation — what's the real difference?
*An approval gate asks permission for a specific already-decided action; escalation hands off the whole situation because the agent doesn't trust its own framing of the problem.*

**Day 12.** Should architecture or framework be decided first?
*Architecture always first — pick the framework whose abstraction matches the already-determined architecture, never the reverse.*

**Day 13.** How does Day 11's approval gate relate to Day 7's state machine?
*It's just one additional named state (AWAITING_CONFIRMATION) inserted on an existing transition edge — a local addition, not a redesign.*

**Day 14.** Why prefer exponential backoff over a fixed retry delay?
*A fixed delay adds sustained, constant pressure to a struggling service exactly when it needs load to drop; backoff spreads retry pressure out and gives it room to recover.*

**Day 15.** What's the typical debugging flow across observability's three pillars?
*Metric alerts you something's wrong → logs narrow down which component/time window → a full trace of one representative run shows the exact mechanism.*

**Day 16.** Why not use the cheapest model for every step to minimize cost?
*It saves money uniformly but hurts quality on whichever step actually determines the outcome — route capability to where it changes the result, not everywhere equally.*

**Day 17.** Why is "tell the model to ignore injected instructions" alone insufficient defense?
*It's the single weakest layer — bypassable by a sufficiently crafted injection — so you need independent layers (tool scoping, output validation) that hold even when it fails.*

**Day 18.** What's the most dangerous quadrant in outcome-vs-trajectory evaluation?
*Correct outcome, bad trajectory — "right for the wrong reasons" — because outcome-only eval reports it as a full pass, hiding a process likely to fail differently elsewhere.*

**Day 19.** Should a checkpoint store full raw history or the same compressed state as live context?
*The same compressed state — a separate full-fidelity backup defeats the point of having deliberately summarized in the first place.*

**Day 20.** What's true of every mechanism across Days 14-19, without exception?
*None has one universally correct setting — every one calibrates to the specific task's stakes, volatility, or trust level.*

**Day 21.** What's the single highest-leverage clarifying question for a support-style agent?
*Can it actually take actions (refunds, account changes), or only answer questions — this one answer determines how much of Day 11 and Day 17 becomes load-bearing.*

**Day 22.** Why does Tree of Thought finally earn its cost in a coding-agent case study when it didn't in the support-agent one?
*Bug investigation genuinely has multiple plausible root-cause hypotheses worth shallow-evaluating before committing, AND has a programmatic verifier (tests) making evaluation reliable — support tickets rarely have either property.*

**Day 23.** What single trajectory-eval check catches tool-misuse-by-timing, hallucinated verification, and cascading errors all at once?
*Does every claim in the reasoning trace back to an actual observation in the trace — the same check, three different failure modes.*

**Day 24.** In the stale-flight-pricing follow-up question, what's the correct diagnosis, and what's the tempting wrong one?
*Correct: a staleness/re-verification problem, fixed by checking availability immediately before the consequential action. Tempting-wrong: reflexively reaching for "add a retry," which doesn't address data going stale between check and use.*

**Day 25.** What's the one sentence that summarizes the whole curriculum?
*An agent is a stateless model wrapped in an architected loop, structured to match the task's actual shape, hardened for production reality, with judgment about which pieces a given task needs — never applied uniformly.*

---

## 4. Final Self-Assessment Before an Interview

Walk through this list. Any unchecked box is worth 10-15 minutes revisiting the relevant day before you feel done:

- [ ] I can draw Day 7's state machine from memory and correctly place a Day 11 approval gate on it.
- [ ] I can explain why agentic RAG (Day 9) is "nothing new" mechanically, in one sentence.
- [ ] I can name three independent, specific layers of prompt-injection defense (Day 17), not just "tell the model not to."
- [ ] I can distinguish outcome eval from trajectory eval (Day 18) and give an example where they'd disagree.
- [ ] Given ANY new system-design prompt, my first move is a clarifying question, not a framework name.
- [ ] I can justify EXCLUDING multi-agent, ToT, or a framework as readily as I can justify including them.
- [ ] I know which Phase 3 topic (Days 14-19) is highest-relevance for a GIVEN task, not just what each topic covers in the abstract.
- [ ] I can recognize Day 23's four failure modes from a trace/description, not just recite their definitions.

You've now built and reviewed 25 days of material covering the full span from "what is an agent" through shipping, debugging, and defending one in a live interview. The material is complete — from here, the highest-value next step is timed mock interviews against novel prompts (not from this curriculum), since recognizing patterns you've studied is a different skill from generating them fresh under pressure, and that gap is what remaining practice time should close.

---
*Curriculum complete. All 25 days delivered as standalone markdown files.*
