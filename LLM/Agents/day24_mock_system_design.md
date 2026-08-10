# Day 24: Mock System Design Questions — Timed Practice

## 1. How to Use This Day

Four prompts, each formatted the way an interviewer would actually give it — short, deliberately underspecified. **The instruction: read one prompt, set a timer for 15 minutes, and actually produce your answer (out loud or written) before reading the reference approach below it.** The reference approach is not "the correct answer" — system design has no single correct answer — it's a model of the REASONING PROCESS worth comparing yourself against: did you ask clarifying questions, did you justify inclusions and exclusions, did you connect to Phase 3 concerns, did you avoid leading with a framework name.

If you only read the reference answers without attempting your own first, you'll recognize good reasoning without training yourself to produce it under pressure — the entire value of this day is in the attempt, not the answer key.

---

## 2. Question 1: The Ambiguous One-Liner

**Prompt**: *"Design an agent that helps employees book travel for business trips."*

<details>
<summary>Reference approach (read only after attempting)</summary>

**Clarify first**: Can the agent actually BOOK (charge a corporate card, commit to a reservation) or only search/recommend? This is the single highest-leverage question (mirrors Day 21 §2's exact framing) — it determines whether Day 11's approval gates and Day 17's guardrails are central or peripheral. Assume: agent CAN book, up to a per-trip budget policy, above which it needs approval.

**Architecture**: Level 3 agentic (Day 1) — flight/hotel options and policy constraints aren't enumerable branches. Core loop: ReAct (Day 2) with tools for flight search, hotel search, calendar lookup (to avoid double-booking travel over existing meetings), and a `book_travel` action. Agentic RAG (Day 9) over the company's travel policy doc (e.g., "economy only for trips under 6 hours, manager approval needed for trips over $2000"), since policy applicability is often multi-hop ("what's the policy for THIS specific route/hotel" requires resolving destination and cost before the right policy clause is even findable). Single-agent is sufficient (Day 8 §5.4) — this doesn't have Day 8's clear independent-workstream shape; it's one coherent reasoning chain (find options → check policy → check calendar → book). Reflexion (Day 4) is a natural fit if a booking attempt fails (e.g., flight sold out between search and booking) — retry informed by that specific failure, not blind re-search.

**HITL (Day 11)**: approval gate for any booking above the policy threshold, exactly mirroring Day 21's refund-threshold pattern; escalation if the agent can't find ANY compliant option within policy (e.g., all reasonable flights exceed the trip's budget) — a situation-level problem, not a single-action approval question.

**Phase 3 emphasis**: Day 17 is genuinely central here in an underrated way — a `book_travel` tool is a REAL financial commitment, so output validation checking the booking amount/dates against the approved policy programmatically (not trusting the model's stated reasoning alone) is essential, directly mirroring Day 17's refund example. Day 16's cost/latency matters less than Day 21's ticket system (booking a trip is not a live-chat-latency-sensitive interaction) — worth explicitly noting this relevance shift, exactly the instinct Day 22 modeled.

**What a weak answer misses**: jumping straight to "I'd use a multi-agent system with a flight agent and hotel agent" without asking whether the task actually has the independent-workstream shape that would justify it (it doesn't, particularly — calendar/policy/booking are sequentially dependent, not parallel).
</details>

---

## 3. Question 2: The "Add a Constraint" Follow-Up

**Prompt**: *"Now: your travel-booking agent's flight-search tool sometimes returns stale pricing — a price shown as available is actually sold out by the time booking is attempted 30 seconds later. How does this change your design?"*

<details>
<summary>Reference approach</summary>

This is testing whether you can adapt an existing design to a NEW specific failure, not redesign from scratch — a very common interview follow-up pattern.

**Immediate diagnosis**: this is a TOCTOU-style (time-of-check-to-time-of-use) reliability problem, not a reasoning problem — the agent's DECISION to book was correct given what it observed; the underlying data went stale between observation and action. This connects to Day 14's reliability concepts, but note it's NOT quite the same as Day 14's transient-failure retry case — retrying the exact same booking attempt would just hit the same staleness issue again.

**Fix**: re-verify price/availability immediately before the ACTUAL booking call, not just relying on the earlier search result (essentially, add a fresh `check_availability` call as the LAST step before `book_travel`, even though this feels redundant with the search that already happened) — directly extending Day 14 §4.1's idempotency discussion in a new direction: the issue here isn't retry-safety, it's STALENESS of the data a decision was based on, and the fix is re-verification immediately before a consequential action, not a caching or retry mechanism.

**Failure mode this connects to (Day 23)**: if the agent's Final Answer says "I've booked your flight" based on the ORIGINAL stale search result without confirming the booking call actually succeeded, that's Day 23's hallucinated-action pattern — the fix requires the Final Answer to be gated on an actual successful `book_travel` observation, not on the earlier search having looked promising.

**What this question is really testing**: whether you reflexively reach for "add a retry" (wrong fix for THIS specific failure shape) versus correctly diagnosing that the issue is staleness of a premise, requiring re-verification at the point of action, not retry of the action itself.
</details>

---

## 4. Question 3: The Comparison Question

**Prompt**: *"Would you use a multi-agent architecture or a single agent with a large context window for a system that needs to analyze a 500-page legal contract and flag risky clauses?"*

<details>
<summary>Reference approach</summary>

This is testing Day 8's core justification test directly — apply it explicitly rather than picking a side reflexively.

**Walk the three justifications (Day 8 §5.4)**:
- **Parallelism**: does clause analysis benefit from parallel processing? Plausibly yes — different SECTIONS of the contract (indemnification, termination, liability) can genuinely be analyzed independently without needing each other's context, unlike Day 21's sequential ticket-resolution reasoning.
- **Context isolation**: a 500-page document may genuinely exceed practical context limits even if it technically fits — and even where it fits, Day 5's "lost in the middle" risk is real at this length; splitting into worker-per-section keeps each worker's context tightly scoped to what it's actually analyzing.
- **Specialization**: if certain clause types need different analysis approaches (e.g., a specialized check specifically for indemnification-clause red flags vs. general risk flagging), per-section workers with tailored prompts/tools is a genuine specialization benefit.

**Conclusion**: multi-agent (orchestrator/worker, Day 8.1) is justified here — genuinely different from Question 1's travel-booking case, where none of these three applied. Explicit design: orchestrator splits the document by section, dispatches each section to a worker with a risk-flagging prompt, workers return flagged clauses + brief justification (directly addressing Day 8 §5.2's hidden-worker-error risk — requiring a justification trail is exactly Day 8's own recommended mitigation), orchestrator synthesizes into one consolidated risk report.

**The counterpoint worth raising explicitly**: a single agent with a sufficiently large context window COULD handle this without splitting, and might catch CROSS-SECTION risk interactions that isolated per-section workers would miss (e.g., a definition in section 2 that changes how a clause in section 40 should be interpreted — a genuine downside of context isolation, worth naming rather than ignoring). **Balanced answer**: multi-agent for the initial pass (leveraging parallelism/isolation benefits), with a final single-agent synthesis pass that has access to ALL workers' flagged clauses together, specifically to catch cross-section interactions the isolated workers structurally can't see — a hybrid that takes the benefit of both approaches while mitigating each one's specific weakness.

**What a weak answer misses**: picking one option and defending it without acknowledging the genuine downside of the choice (context isolation's cross-section blind spot) — a strong answer shows awareness of the tradeoff, not just the winning side of it.
</details>

---

## 5. Question 4: The "What Would You Monitor" Question

**Prompt**: *"You've shipped an agentic system that automatically categorizes and prioritizes incoming IT support tickets. Three weeks in, what would you be watching to know if it's working well?"*

<details>
<summary>Reference approach</summary>

This is a pure Phase 3 question (Days 14-20), specifically testing Day 15's observability instincts and Day 18's evaluation instincts together — a common interview shape once the "how would you design it" question has already been asked in an earlier round.

**Leading metrics (Day 15 §2.3)**: categorization confidence distribution over time (a downward drift is the EARLY warning signal, mirroring Day 15's exact worked example); escalation/override rate (how often does a human manually recategorize what the agent assigned — a rising override rate is a direct proxy for declining accuracy, often available faster than a full eval run); tool error rates for whatever ticketing-system API the agent calls (Day 14's reliability metrics).

**Evaluation, ongoing (Day 18 §5.2)**: this can't be a one-time pre-launch check — set up continuous sampling of agent categorizations against what a human reviewer would have assigned, building the eval set FROM real disagreements (Day 18 §5.1) rather than a fixed pre-launch test set that goes stale as ticket patterns shift over time.

**Trajectory-level check, not just outcome (Day 18 §2.2)**: even where the categorization happens to be correct, check whether the agent's stated reasoning actually grounds in the ticket content, or whether it's pattern-matching to superficial keyword similarity without real understanding — this catches the Day 23 cascading/hallucination risk before it manifests as a visibly wrong categorization on some future ticket.

**A specific thing many candidates miss**: watching for DISTRIBUTION SHIFT in ticket types themselves, not just agent accuracy — if a new category of IT issue starts appearing (say, a new software rollout generates an unfamiliar ticket pattern), the agent's performance on THAT SPECIFIC SLICE needs to be watched separately from overall aggregate accuracy, since an aggregate metric can look fine while a specific, growing subpopulation is being handled poorly — this is a subtle but real production monitoring instinct (essentially: don't just watch the average, watch the worst-performing meaningful slice, since averages can hide it).

**What this question is really testing**: whether "shipped" means "done" to you, or whether you understand production ML/agentic systems require ongoing operational vigilance — Day 20's entire Phase 3 consolidation point, applied as a live, unprompted answer rather than recited.
</details>

---

## 6. Self-Assessment After Completing All Four

Score yourself honestly against these, not against getting "the right answer" (there isn't one):

- [ ] Did you ask at least one clarifying question before designing, on questions where scope was genuinely ambiguous (Q1)?
- [ ] Did you explicitly justify EXCLUDING a pattern you know about, not just including ones that fit (e.g., correctly NOT reaching for multi-agent in Q1, or correctly reaching for it in Q3)?
- [ ] On the follow-up/constraint question (Q2), did you correctly diagnose the SPECIFIC failure shape (staleness, not generic "add retries") rather than pattern-matching to the nearest memorized fix?
- [ ] On the comparison question (Q3), did you name the genuine DOWNSIDE of your chosen approach, not just its benefits?
- [ ] On the monitoring question (Q4), did you go beyond "I'd track accuracy" to name SPECIFIC, actionable metrics and explain what each one would tell you?

If most of these are checked, you're reasoning from the underlying principles rather than pattern-matching to memorized templates — which is precisely what 24 days of this curriculum was building toward.

---
*Next: Day 25 — Full Review + Rapid-Fire Q&A Bank (final consolidation across all 25 days).*
