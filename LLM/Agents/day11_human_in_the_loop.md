# Day 11: Human-in-the-Loop Patterns — Approval Gates, Escalation, Interrupts

## 1. The Intuition First

Think about a junior employee you're mentoring. For routine, low-stakes decisions — checking a public webpage, drafting an email — you let them act independently and review the outcome afterward. For anything consequential — sending a company-wide announcement, approving a large expense, deleting a customer's data — you want to see it **before** it happens, not after. And if they hit something genuinely outside their judgment (an angry customer threatening legal action), you want them to stop and escalate to you immediately, rather than improvising.

That's the entire design space of human-in-the-loop (HITL) for agents: **not "should a human ever be involved" but "at which specific points, and in what mode, should the human be involved."** This directly extends Day 7's AWAITING_CONFIRMATION state — today we go deep on exactly when to insert that state, what shape it takes, and the production engineering required to make it work reliably (not just draw the arrow on a diagram).

---

## 2. Formalizing the Three Patterns

### 2.1 Approval Gates — Synchronous, Pre-Action Confirmation

The agent proposes an action, halts, and waits for explicit human approval **before** executing it. This is exactly Day 7's AWAITING_CONFIRMATION state:

```
THINKING → (proposes side-effecting action) → AWAITING_CONFIRMATION → [human approves] → ACTING
                                                        │
                                                  [human rejects] → THINKING (with rejection as new context)
```

**When to use it**: irreversible or costly actions — sending external communications, financial transactions, deleting data, deploying code to production. The defining trait: **the action does not happen until a human explicitly says yes**, full stop.

### 2.2 Escalation — Agent Recognizes Its Own Limits and Hands Off

Different from an approval gate: the agent isn't asking permission for a specific action it already decided on — it's recognizing that **the situation itself is outside its competence or authority**, and handing the entire task (or remainder of it) to a human.

```
THINKING → (recognizes: confidence too low / policy violation / genuinely ambiguous situation)
        → ESCALATED (task handed to human, agent stops autonomous action on this task)
```

**When to use it**: situations where continuing to act — even with individual approval gates — isn't appropriate, e.g., a legal threat, a request that might violate policy in a way the agent isn't equipped to judge, or a confidence score below a defined threshold on a high-stakes classification.

### 2.3 Interrupts — Human Can Intervene Mid-Execution, Asynchronously

Different again: this isn't the agent stopping to ask — it's the human being able to **inject input, redirect, or halt the agent while it's actively running**, without the agent needing to have specifically paused for that purpose.

```
ACTING/THINKING (agent running autonomously) ⟵ [human sends interrupt: "stop" / "actually, focus on X instead"]
                                              → agent's next loop iteration incorporates the interrupt
```

**When to use it**: long-running autonomous tasks (a research agent running for 20 minutes) where a human observing progress in real time might want to redirect without waiting for the agent to hit a natural pause point.

### 2.4 The Key Distinction Interviewers Test For

All three insert a human into the loop, but at fundamentally different **triggers** and **initiators**:

| Pattern | Who initiates the pause? | What triggers it? | Agent's state while waiting |
|---|---|---|---|
| Approval gate | The agent (built into the flow) | A specific proposed action matching a rule (e.g., side-effecting) | Fully halted, cannot proceed without a yes/no |
| Escalation | The agent | Self-assessed inability/inappropriateness to continue | Task handed off; agent may not resume autonomously at all |
| Interrupt | The human | Anything, anytime, at the human's discretion | Was running autonomously; must gracefully incorporate an unplanned signal |

**The interview-precise way to say this**: *"Approval gates and escalation are both agent-initiated pauses, triggered by conditions built into the agent's own logic. Interrupts are human-initiated, and the agent has no way to predict when one will arrive — which makes them the hardest to engineer, because the agent's loop has to be built to gracefully accept an out-of-band signal at any point, not just at a designed pause."*

---

## 3. Worked Example: Approval Gate on a Side-Effecting Tool

**Task**: "Refund the customer's last order and send them an apology email."

```
State: THINKING
  Thought: I need to process a refund and send an email — both are side-effecting
  actions (Day 3 §5.3's write-tool classification). Per policy, refunds over $0 always
  require approval.
  Action (proposed, not yet executed): process_refund(order_id="A123", amount=45.00)

State: AWAITING_CONFIRMATION
  Message to human: "Proposing: refund $45.00 to order A123 (reason: customer complaint
  about late delivery). Approve?"
  [Agent halts here. No refund has happened. No email has been sent.]

  Human: "Approved, but make the email more apologetic — this is their 3rd late delivery."

State: ACTING
  Execute process_refund(order_id="A123", amount=45.00)
  Observation: "Refund processed successfully."

State: THINKING
  Thought: Refund done. Now for the email — the human specifically asked for a more
  apologetic tone given repeat lateness, I should incorporate that context.
  Action (proposed): send_email(to=customer, subject="Our apologies", body=<updated tone>)

State: AWAITING_CONFIRMATION
  Message to human: "Draft email: [shows draft]. Send?"
  Human: "Send it."

State: ACTING
  Execute send_email(...)
  Observation: "Email sent."

State: COMPLETE
```

### 3.1 The Non-Obvious Detail: The Human's Approval Isn't Just Yes/No

Notice the human didn't just say "approved" — they added context ("make the email more apologetic") that **changed a downstream action**. A well-designed approval gate doesn't just gate a binary decision — it's a checkpoint where the human's response becomes new context fed back into THINKING, exactly like any other observation (Day 2's core principle again: everything that happens gets reasoned about, not silently applied). If your implementation only supports approve/reject and can't incorporate free-form human feedback into subsequent reasoning, you've built a much weaker gate than what's actually needed in practice.

---

## 4. Worked Example: Escalation vs. an Approval Gate (the distinction that trips people up)

**Same overall task, different trigger**:

```
State: THINKING
  Thought: I need to process a refund for this customer complaint.
  Action: search_order_history(customer_id)
  Observation: "This customer has filed 6 refund requests in 30 days, and the current
  message contains language threatening legal action over 'systemic billing fraud.'"

  Thought: This isn't a normal refund-approval situation anymore — the legal-threat
  language and unusual request pattern are outside what I'm equipped to judge or act
  on. This needs a human to handle directly, not just approve an action I've already decided on.

State: ESCALATED
  Message to human: "Escalating: customer with 6 refund requests in 30 days, current
  message contains legal-threat language re: 'systemic billing fraud.' Recommend
  legal/senior support review — not routing through standard refund flow."
  [Agent does NOT propose a specific action for approval. It hands off the whole
  situation, because the situation itself, not just one action within it, is outside scope.]
```

**Why this is escalation, not an approval gate**: an approval gate assumes the agent has already correctly judged what the right action is and just needs a human's sign-off before executing it. Here, the agent is explicitly saying **it doesn't trust its own judgment about what the right action even is** — the situation itself needs human judgment, not just human sign-off on the agent's proposed next step. Conflating these in production is a real design mistake: **routing an escalation-worthy situation through a normal approval gate implicitly asks the human to approve or reject the agent's framing of the problem, when the actual issue is that the framing itself may be wrong.**

---

## 5. Worked Example: An Interrupt Mid-Execution

**Task**: A research agent has been running autonomously for 12 minutes, gathering sources for a competitive analysis report (a long ReAct/Plan-and-Execute trajectory, Day 4).

```
[Agent is in ACTING state, 8 tool calls in, still gathering sources]

[Human, watching a live trace/dashboard (Day 15 preview), sends an interrupt:
"Stop researching Competitor C, they're not relevant anymore, focus remaining
time on Competitor A and B instead."]

State: (interrupt received — must be checked for at the NEXT loop boundary,
        since the agent can't be interrupted mid-token-generation)

State: OBSERVING (next natural transition point)
  The interrupt is injected as a new message, exactly like a tool observation:
  "[Human interrupt]: Stop researching Competitor C, focus remaining time on A and B."

State: THINKING
  Thought: I've received new direction — I should drop my remaining planned research
  on Competitor C and reallocate effort to A and B.
  [Revises remaining plan accordingly, continues.]
```

### 5.1 Why Interrupts Are Mechanically the Hardest of the Three

Approval gates and escalation are **designed pause points** — you control exactly where in the code they occur. An interrupt can arrive **at any arbitrary moment**, including mid-tool-execution. This creates a real engineering problem: **you cannot interrupt an LLM generation or a tool call that's already in flight** — the practical implementation is that the interrupt signal gets queued, and the agent's loop checks for pending interrupts at defined boundaries (typically: after each OBSERVING state, before the next THINKING call starts, echoing Day 7's state machine). This means there's an unavoidable, bounded delay between when a human sends an interrupt and when the agent actually incorporates it — usually the time to finish the current in-flight action. **This delay is a real, quotable production constraint, not a bug** — and worth naming explicitly if asked to design this: *"the interrupt can't preempt an in-flight action, so I'd check for it at the state machine's natural transition points, accepting a bounded delay of one tool-call's duration."*

---

## 6. Production Considerations

### 6.1 Deciding Where Gates Go — This Is a Policy Decision, Not Just an Engineering One

Directly extending Day 3 §5.3 (read vs. write tool distinction): the most common production rule is **gate all side-effecting/write actions above some risk threshold, never gate read-only actions**. But "risk threshold" is genuinely a product/policy decision, not a purely technical one — should a $5 refund require approval, or only refunds over $500? This needs to be an explicit, reviewable policy (often literally a config: `{tool: "process_refund", requires_approval_above: 100}`), not a judgment call left to the model's own discretion each time — because leaving it to the model means the threshold isn't consistent or auditable, and a model that's slightly miscalibrated could silently expand or shrink the effective threshold over time.

### 6.2 Latency Cost of Synchronous Gates

An approval gate makes the agent **stop and wait for a human**, which could be seconds (a human watching a live dashboard) or hours (an email approval sent to someone who's in a meeting). For any task with a deadline or user-facing latency expectation, this needs to be designed for explicitly: does the task time out? Does it queue and resume later (this is exactly Day 7 §5.2's checkpointing — the agent's state needs to be persistable precisely because it might be waiting on a human for an unpredictable amount of time)? A common mistake is building an approval gate assuming synchronous, fast human response, then discovering in production that approvals often sit for hours, during which the agent's in-memory state would have been lost without persistence.

### 6.3 Escalation Needs a Real Destination, Not Just a Flag

An escalation that just sets a flag or writes a log line nobody monitors is not actually human-in-the-loop — it's silently dropping the task. Production escalation paths need: a real notification channel (paging, ticket creation, dashboard alert) and a defined SLA for response, or the "human in the loop" is theoretical. This is a common gap between a system design interview's whiteboard version ("if uncertain, escalate to a human") and what actually ships — worth explicitly acknowledging if asked to go deep: *"the escalation path is only as good as the human response process behind it; I'd want a defined on-call/SLA structure, not just a fire-and-forget flag."*

### 6.4 Over-Gating Destroys the Value of Agentic Automation

This is the counterbalancing production risk, and a good interview answer shows you see both sides: gate too aggressively (every single action needs approval) and you've built an expensive, slow approval-request generator, not an agent — you've paid all the engineering cost of agentic automation while getting none of the autonomy benefit. This connects directly back to Day 1's core tradeoff table: **approval gates are a direct lever trading autonomy/latency for safety/predictability**, and the right amount is exactly as much as the specific action's risk justifies, not maximal caution applied uniformly.

---

## 7. Interview Q&A

**Q1: What's the difference between an approval gate and escalation? Give an example of each.**
A: An approval gate is agent-initiated, pausing before a *specific proposed action* that matches a rule (e.g., any refund over $100) — the agent has already decided what to do and is asking permission to execute it. Escalation is also agent-initiated, but triggers when the agent judges the *entire situation* is outside its competence or authority to handle at all — e.g., a customer's message contains a legal threat — and it hands off the whole task rather than proposing a specific action for sign-off. Routing an escalation-worthy situation through a normal approval gate is a common mistake, since it implicitly asks the human to approve the agent's framing of the problem when the actual issue is that the framing itself may be wrong.

**Q2: Why are interrupts mechanically harder to implement than approval gates or escalation?**
A: Approval gates and escalation are designed pause points, built into the agent's own logic at known locations in the code. An interrupt can arrive from a human at any arbitrary moment, including while a tool call or LLM generation is already in flight, and you cannot preempt those mid-execution — so the practical implementation queues the interrupt and checks for it only at defined state-machine boundaries (e.g., after OBSERVING, before the next THINKING call), which means there's an unavoidable, bounded delay before the agent incorporates it.

**Q3: A human approves a proposed action but adds extra context or a modification request. How should that be handled?**
A: The human's response shouldn't just be treated as a binary unlock for the exact proposed action — it should be fed back into the agent's context (THINKING state) like any other observation, so the agent can incorporate the added context into subsequent actions. A gate that only supports strict approve/reject and can't incorporate free-form feedback is weaker than what's actually needed in most real approval workflows.

**Q4: How do you decide which actions require an approval gate?**
A: This is fundamentally a policy/product decision, not something left to the model's own discretion at run-time — typically implemented as explicit, reviewable configuration (e.g., a specific tool requires approval above a defined dollar/risk threshold). Leaving the threshold to the model's judgment each time makes it inconsistent and unauditable, and a slightly miscalibrated model could silently drift the effective threshold over time.

**Q5: What's a common production gap between a whiteboard "escalate to a human" design and what actually needs to ship?**
A: An escalation path that just sets a flag or logs an event with no real notification mechanism or defined response SLA is effectively dropping the task, not actually looping in a human. Production escalation needs a genuine destination — paging, ticket creation, a monitored queue — and a defined response expectation, or "human in the loop" is theoretical rather than real.

**Q6: What's the risk of over-using approval gates, and how would you push back on a design that gates every single action?**
A: Gating too aggressively means every action pauses for human sign-off, which erodes the core value proposition of agentic automation — you're paying the full engineering and latency cost of building an agent while getting essentially none of the autonomy benefit, effectively building an expensive approval-request generator instead. The right amount of gating should scale with each specific action's actual risk (irreversibility, cost, scope of impact) — same autonomy-vs-predictability tradeoff introduced in Day 1, applied specifically to where human checkpoints go.

---

## 8. Summary Card

- **Approval gate**: agent-initiated, pauses before a specific proposed action matching a risk rule; human's response should feed back as context, not just approve/reject.
- **Escalation**: agent-initiated, triggers when the whole *situation* (not just one action) is outside the agent's competence; hands off the task, doesn't propose an action for sign-off.
- **Interrupt**: human-initiated, can arrive at any moment; mechanically hardest because in-flight actions can't be preempted — checked only at state-machine boundaries, with a bounded, unavoidable delay.
- Gate thresholds are a **policy decision** (explicit config), not model discretion.
- Real production risks: synchronous gates need state persistence for unpredictable wait times (Day 7's checkpointing); escalation needs a real notification/SLA destination, not a silent flag; over-gating destroys the value of automation entirely.

---
*Next: Day 12 — Frameworks Landscape (LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, and when raw loops beat all of them).*
