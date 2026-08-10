# Day 22: Case Study — Coding/Research Agent (Full System Design Walkthrough)

## 1. The Setup

**Prompt**: "Design an AI agent that can autonomously investigate and fix bugs in a large codebase — given a bug report, it should reproduce the issue, locate the relevant code, propose and implement a fix, run tests, and open a pull request."

This case study is deliberately chosen to be architecturally DIFFERENT from Day 21's support agent in specific, instructive ways — a strong interview candidate recognizes that "coding agent" and "support agent" aren't solved with the same template, even though both are agentic systems. The differences you should be actively listening for and calling out: **longer-running trajectories (Day 19 becomes central, not peripheral), code execution as a core tool (Day 17's sandboxing becomes central, not optional), and a fundamentally different risk profile (Day 17's guardrails matter for different reasons than Day 21's).**

---

## 2. Step 1 — Clarify Scope

**Questions worth asking**:
- "Does the agent commit directly, or always open a PR for human review?" (drives Day 11's approval-gate design — this answer changes almost everything downstream)
- "How large is the codebase — can relevant context fit in one context window, or does it need retrieval over the repo?" (drives Day 9's agentic RAG design, applied to code instead of documents)
- "Is there an existing test suite, and how reliable is it?" (drives Day 18's evaluation approach — a good test suite is a rare gift: an EXTERNAL, PROGRAMMATIC verifier, directly connecting to Day 10 §5.2's point about ToT/verification quality)
- "What's an acceptable trajectory length — minutes, or could this reasonably run for hours on a hard bug?" (drives Day 19's urgency directly)

**Assume**: large codebase (doesn't fit in one context window), decent test coverage, agent always opens a PR (never commits directly) — and bugs can reasonably take up to 30-45 minutes of agent time to resolve.

---

## 3. Step 2 — Architecture Decisions, Layer by Layer (Days 1-13)

### 3.1 Retrieval Over the Codebase — Agentic RAG, Code-Specific Considerations (Day 9)
Unlike Day 21's KB search, "finding the relevant code" is a genuinely hard multi-hop retrieval problem: a bug report says "checkout fails for international addresses," but the actual relevant code might be 3 files removed from anything mentioning "checkout" or "international" in an obvious way (e.g., it's actually a validation regex in a shared utility file). This is Day 9's query-reformulation pattern, but the search tool itself is different — typically a mix of semantic code search (embedding-based) AND structural search (find references to a specific function, trace call graphs) — worth explicitly naming that code retrieval often needs BOTH search types, not just embedding similarity, unlike Day 21's mostly-prose knowledge base.

### 3.2 Reproduction Step — A Distinct Phase This Task Has and Day 21 Didn't
Before any fix can be attempted, the agent needs to CONFIRM it can reproduce the reported bug — this is functionally a verification step (directly connecting to Day 10 §5.2's discussion of programmatic verifiers): write/run a test that currently FAILS in the way the bug report describes, before touching any fix logic. This matters because it converts "does the eventual fix actually work" from a fuzzy judgment call into a checkable, binary signal (does the reproduction test now pass) — exactly the kind of external verifier that makes rigorous evaluation (Day 18) and even Tree-of-Thought-style exploration (Day 10) genuinely tractable for this domain in a way that's much harder for open-ended tasks like Day 21's support responses.

### 3.3 Planning Strategy — Where This Case Study Actually Uses Day 10 (ToT)
This is the FIRST case study in this curriculum where Tree of Thought (Day 10) is genuinely justified, not over-engineering: a non-trivial bug often has multiple plausible root-cause hypotheses (e.g., "is this a validation bug, a data-formatting bug, or a race condition in the async handler?"), and committing early to investigating ONE hypothesis fully, only to discover 20 minutes later it was wrong, is exactly the wasted-reasoning-budget problem Day 10 exists to address. Concretely: generate 2-3 candidate root-cause hypotheses, do a SHALLOW investigation of each (just enough to gather evidence, not a full fix attempt), evaluate which hypothesis the evidence actually supports (here, evaluation can be partially PROGRAMMATIC — does inserting a debug print at hypothesis A's suspected location actually show the bad value — a more reliable evaluator than Day 10 §5.2's harder case of pure LLM self-critique), then commit fully to investigating the best-supported hypothesis.

**Why Day 21 never needed this and Day 22 does**: support tickets rarely have this "multiple plausible root causes requiring investigation before you can tell which is right" shape — most tickets map fairly directly to a known issue type. Bug investigation frequently does have exactly this shape. This is a good, concrete example of Day 10 §5.1's guidance ("reserved for a specific, narrow class of problems") being satisfied by one case study and not the other.

### 3.4 Reflexion — Central to the Fix-and-Test Loop (Day 4)
Once a fix is implemented, running the test suite is a natural, CLEAN Reflexion trigger (directly matching Day 4 §4's worked example almost exactly): tests fail → generate a reflection on WHY (not just "tests failed" but "the fix didn't handle the empty-string case, which test_empty_address specifically checks") → retry the fix informed by that reflection. This is a much cleaner, more reliable Reflexion loop than most domains get, PRECISELY because the test suite is a rigorous programmatic verifier (Day 18's distinction between reliable external verification vs. noisy LLM self-judgment) — worth explicitly contrasting with Day 21's support agent, where "was this response good" has no equivalent clean pass/fail signal.

### 3.5 Human-in-the-Loop (Day 11) — Different Shape Than Day 21
The "always opens a PR, never commits directly" scoping decision means the ENTIRE final output is already gated by human review — this is actually a much simpler HITL design than Day 21's, in one sense: there's no need for fine-grained per-action approval gates on individual tool calls (running tests, reading files are all low-risk/reversible), because the ONE consequential action (merging code) is already structurally gated by the PR review process itself, outside the agent's control entirely. Escalation is still relevant, though: if the agent's reproduction step fails repeatedly (can't even confirm the bug exists) or the investigation reveals the bug is far more architecturally significant than the report suggested, escalating BEFORE spending a full 45-minute budget on a fix attempt that's unlikely to succeed is the right call (Day 11 §2.2's "situation itself is outside scope," not just "gate this action").

---

## 4. Step 3 — Production Engineering Layer (Days 14-20), With Explicit Emphasis Shifts

### Day 19 (State & Context) — HIGH Relevance Here, Unlike Day 21
This is the most significant emphasis shift versus Day 21. A 30-45 minute bug investigation, potentially exploring multiple ToT branches (§3.3), reading many files, running many test iterations, WILL hit meaningful context pressure (Day 19 §2.1). Structured extraction (Day 19 §2.2c) is a strong fit here specifically because the facts worth preserving are highly predictable in this domain: `{files_examined: [...], hypotheses_tested: [...], current_fix_attempt: ..., test_results: [...]}` — much more tractable to define a good structured schema for a coding task than for Day 21's more open-ended support conversations, directly illustrating Day 19 §2.2's tradeoff point (structured extraction requires knowing the schema in advance; here, that's genuinely feasible). Checkpointing matters concretely: if this runs for 45 minutes and something interrupts it, resuming from "I was mid-investigation of hypothesis B, having ruled out hypothesis A" (not from scratch) is a real, meaningful production requirement.

### Day 17 (Guardrails) — Different Risk Profile Than Day 21
Day 21's injection risk was primarily about untrusted TEXT (ticket content) manipulating actions. Here, the code-execution tool itself (running tests, potentially running the reproduction script) is the primary risk surface — directly Day 17 §4's sandboxing discussion, now load-bearing rather than a brief mention: test execution MUST run in an isolated environment with no access to production credentials/data, resource limits (a buggy or maliciously-crafted fix attempt could contain an infinite loop or resource exhaustion), and no network access beyond what's explicitly needed. This is a genuinely different emphasis: Day 21 needed injection defense as the primary guardrail; Day 22 needs sandboxing as the primary guardrail — both are "Day 17," but which SUB-topic dominates is domain-specific, worth explicitly calling out rather than treating Day 17 as one undifferentiated checklist item.

### Day 18 (Evaluation) — Easier Here Than Day 21, Worth Explicitly Noting
Because a real test suite exists, MUCH of this system's evaluation can be outcome-based and fully objective (does the reproduction test now pass, do all existing tests still pass) rather than relying on LLM-as-judge's pitfalls (Day 18 §4) — a genuine, notable advantage of this domain versus Day 21's support responses, where "was this a good answer" has no equivalent objective check. Trajectory eval is still valuable (did the agent actually investigate before proposing a fix, or pattern-match to a superficially similar past bug without verifying), but the OUTCOME signal here is far more trustworthy than in most other agentic domains — worth explicitly stating this contrast if asked to compare the two case studies.

### Day 16 (Cost & Latency) — Different Shape
Given 30-45 minute trajectories are expected and accepted (per the scoping assumption), LATENCY pressure is much lower here than Day 21's ticket system (users aren't waiting live for a chat response) — this shifts the cost/latency tradeoff (Day 16 §4.2) toward "optimize cost, tolerate latency" rather than Day 21's "both cost AND latency matter, given live user-facing traffic." Model routing still applies (simpler file-reads/searches on a cheaper model, the actual fix-generation reasoning on the most capable model) but there's less urgency around aggressive caching/parallelism than a high-volume, latency-sensitive system would need.

---

## 5. Step 4 — Whole-System Diagram

```
Bug report arrives
     │
     ▼
[THINKING] search codebase (agentic RAG, semantic + structural, Day 9) for likely locations
     │
     ▼
[ACTING] write + run a reproduction test (sandboxed, Day 17) → confirm bug reproduces
     │
     ├─── can't reproduce after reasonable attempts ──► [ESCALATED] (Day 11) — don't burn budget
     │
     ▼ (reproduced)
[THINKING] generate 2-3 root-cause hypotheses (Day 10, ToT) → shallow investigation of each
     │        (structured state tracking which hypotheses tested/ruled out, Day 19)
     ▼
[THINKING] commit to best-supported hypothesis → implement fix
     │
     ▼
[ACTING] run full test suite (sandboxed, Day 17)
     │
     ├─── tests fail ──► [Reflexion, Day 4] generate reflection on WHY → retry fix
     │                    (bounded retry count, Day 4 §5.2)
     │
     ▼ (tests pass)
[ACTING] open PR (the one consequential action — inherently gated by human review, Day 11)
     │
COMPLETE

[Cross-cutting: Day 15 observability on hypothesis-testing trace; Day 19 checkpointing every
 ~5-10 min given long trajectory; Day 16 model routing on cheap-vs-expensive steps]
```

---

## 6. Explicit Comparison — What This Case Study Teaches That Day 21 Didn't

This is worth internalizing as its own interview-ready synthesis, since "compare and contrast two system designs" is itself a common interview question:

| Dimension | Day 21 (Support Agent) | Day 22 (Coding Agent) |
|---|---|---|
| Trajectory length | Short (seconds-minutes) | Long (30-45 min) |
| Day 19 relevance | Low | High |
| Primary Day 17 concern | Injection defense (untrusted text) | Sandboxing (code execution) |
| Day 18 evaluation basis | Mostly LLM-as-judge (no ground truth) | Mostly objective (test suite = verifier) |
| Day 10 (ToT) relevance | Low (most tickets don't branch) | High (root-cause hypotheses genuinely branch) |
| HITL shape (Day 11) | Fine-grained per-action gates | Coarse (single PR-review gate) |
| Cost/latency priority | Both matter (live user-facing) | Cost matters more than latency |

**The single most important takeaway from doing both case studies back to back**: the SAME 20 days of concepts get assembled completely differently depending on the task's actual shape — long vs. short horizon, objective vs. subjective success signal, text-injection vs. code-execution risk, fine-grained vs. coarse-grained human oversight. **A senior candidate doesn't have one template they force onto every system design question — they re-derive the architecture from the task's specific properties every time**, which is exactly the discipline both walkthroughs modeled explicitly (asking scope questions, justifying inclusion AND exclusion, explaining relevance shifts) rather than reciting.

---
*Next: Day 23 — Failure Modes Catalog (infinite loops, tool misuse, hallucinated actions, cascading errors) — the "what goes wrong" companion to Days 21-22's "how it's built."*
