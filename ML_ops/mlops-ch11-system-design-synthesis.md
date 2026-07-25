# Chapter 11 — System Design Synthesis (Capstone)

*(Module 11 of the syllabus — ties Chapters 1–10 together)*

---

## 1. Why this chapter is different

Every chapter so far taught you one *piece* of the puzzle in isolation. In a real interview, you're rarely asked "explain training-serving skew" in a vacuum — you're asked to **design a system**, and a strong answer weaves together versioning, packaging, deployment strategy, latency, monitoring, governance, and retraining into one coherent whole, prioritized around the specific constraints of the scenario.

This chapter gives you a repeatable framework, then walks through four worked examples using it.

---

## 2. The framework — five steps, in order

Use this structure for *any* "design the deployment/system for X" prompt. The order matters — jumping straight to "I'd use a canary release" without first establishing requirements is the single most common way candidates lose points on these questions.

### Step 1: Requirements
Before naming a single tool or strategy, pin down: What does this model do? What's the latency budget? What's the traffic volume? What happens if it's wrong (low-stakes recommendation vs. high-stakes fraud/lending decision)? Is this real-time or batch?

### Step 2: Constraints
What's fixed and non-negotiable versus what's flexible? Common ones: a hard SLA, a compute/cost budget, a regulatory requirement (Chapter 9), how fast the domain drifts (Chapter 7/10).

### Step 3: Data flow
Sketch, even just verbally, how data moves: where features come from, whether training and serving share a feature store (Chapter 4), how predictions and outcomes get logged (Chapter 7).

### Step 4: Deployment strategy
Given the risk tolerance from Step 1/2, choose and sequence your rollout approach (Chapter 5) — shadow → canary → A/B → full, or a subset, justified by the stakes.

### Step 5: Monitoring & failure modes
What could go wrong, how would you detect it (Chapter 7), what's the automatic response (rollback, Chapter 5; retrain trigger, Chapter 10), and what's the audit trail (Chapter 9).

**Say this structure out loud as you answer** — literally narrating "let me first establish requirements, then constraints, then walk through data flow, deployment, and monitoring" is itself a strong signal to an interviewer that you have a repeatable process, not just scattered knowledge.

---

## 3. Worked example 1: Fraud detection model needing sub-100ms latency

**Step 1 — Requirements:** Real-time decision blocking or allowing a transaction. Sub-100ms is a hard SLA — this decision happens inline, blocking the user's checkout flow. Being wrong is costly in both directions (false positive blocks a legitimate purchase; false negative lets fraud through) — this is genuinely high-stakes.

**Step 2 — Constraints:** Latency is non-negotiable (Ch6). Fraud patterns shift adversarially and can shift *suddenly*, not just gradually (Ch7 — concept drift, not just data drift) — fraudsters actively adapt to evade detection.

**Step 3 — Data flow:** A feature store (Ch4) with a real-time online store is essential here specifically because of the latency requirement — features need to be pre-computed/fresh and instantly retrievable, not computed on the fly during the request. Point-in-time correctness in the offline store prevents leakage during training.

**Step 4 — Deployment strategy:** Given how high-stakes a wrong decision is, sequence matters a lot here: shadow first (validate predictions against the current model with zero user risk), then a small canary with tight, automatic rollback thresholds (Ch5) — not a large canary, given the cost of being wrong at scale. A/B testing on business metrics (actual fraud caught vs. false-positive rate impact on legitimate customers) before full rollout.

**Step 5 — Monitoring & failure modes:** Given the sub-100ms SLA, latency monitoring at p99 (Ch7) is critical — a slow model here doesn't just degrade UX, it can literally block checkout. Given adversarial concept drift, performance-triggered retraining (Ch10) is essential — a fixed weekly schedule alone would be too slow to respond to a sudden new fraud pattern. Given the stakes, governance (Ch9) — full auditability of which model version made which call — is a hard requirement, not a nice-to-have, since fraud decisions get disputed and reviewed.

**Why this is a strong answer:** notice how the *latency requirement* and the *adversarial nature of fraud* — both established in Step 1/2 — directly drove specific downstream choices (real-time feature store, tight canary, performance-triggered retraining) rather than defaulting to generic best practices disconnected from the scenario.

---

## 4. Worked example 2: A/B testing framework for a recommendation model

**Step 1 — Requirements:** This prompt is specifically about *measuring* which of two models is better for the business, not primarily about safe rollout mechanics — so this pulls mainly from Chapter 5's A/B testing section rather than the full deployment-strategy sequence.

**Step 2 — Constraints:** Need a large enough, representative enough user population split to reach statistical significance (Ch5) in a reasonable timeframe. Need to define the primary metric (e.g., engagement or conversion) *and* guardrail metrics (Ch5 — e.g., don't let click-through improve while user-reported satisfaction quietly drops) before starting, not after seeing results.

**Step 3 — Data flow:** Both model versions need to be served from the same infrastructure with a consistent traffic-splitting mechanism (e.g., a stable hash of user ID, so the same user consistently lands in the same group for the test's duration — otherwise you get noisy, inconsistent measurement).

**Step 4 — Deployment strategy:** This *is* the deployment strategy, but note it should still be preceded by a smaller-scale shadow/canary check (Ch5) to confirm the new model isn't obviously broken before committing meaningful traffic to a rigorous, longer-running A/B test — no point running a weeks-long statistical experiment on a model that has a basic bug.

**Step 5 — Monitoring & failure modes:** Watch for the novelty effect (Ch5) — a short test window can show an artificial lift just because the recommendations are different, not better; running long enough to see if the effect persists is a specific, concrete design decision here. Guardrail metric breaches should be able to stop the test early, similar in spirit to a canary rollback trigger.

---

## 5. Worked example 3: Detecting and responding to a model silently degrading in production

This prompt is almost entirely Chapters 7 and 10, applied together as a response plan rather than isolated definitions.

**Detection:** Layer the four monitoring levels from Ch7 — infrastructure health (rule out an outright outage first, it's the fastest check), then feature/input distribution (early warning, catches skew per Ch4 and data drift), then output/business metrics (slower but most directly meaningful). State explicitly: silent degradation is exactly the case where infrastructure metrics look fine, so the *feature and output distribution layers* are where this actually gets caught — naming this explicitly shows you understood Ch7's central point.

**Diagnosis:** Once flagged, follow Ch4's debugging order — check for an offline/online feature computation mismatch first (fixable immediately, a bug), then leakage in the original training pipeline, then genuine drift (Ch7 — distinguish data drift from concept drift, since the fix differs).

**Response:** If it's a genuine bug (skew/leakage), that's a rollback situation (Ch5) while the bug gets fixed properly. If it's real drift, that's a retraining trigger (Ch10) — performance-triggered specifically, since it's reacting to a detected problem rather than waiting on a calendar.

**Prevention going forward:** This is where you'd propose tightening the actual gap that let this go undetected for a while — e.g., if it took too long to notice, that's an alerting threshold problem (Ch7); if it took too long to diagnose, a feature store (Ch4) would eliminate the offline/online mismatch risk going forward.

---

## 6. Worked example 4: Rollback within 60 seconds of a bad metric

**Step 1 — Requirements:** The 60-second number is the whole ballgame here — it explicitly rules out anything involving a human in the loop for the *decision* to roll back (a human noticing an alert, evaluating it, and manually triggering a rollback essentially never happens in under 60 seconds reliably). This has to be automated.

**Step 2 — Constraints:** The rollback mechanism itself must be fast — this pushes toward blue-green (Ch5), since it offers instant traffic redirection back to a fully-running previous environment, versus canary's gradual percentage-reduction approach, which is comparatively slower to fully reverse.

**Step 3 — Data flow / detection:** The triggering metric needs to be computable in near-real-time, not something that requires waiting on delayed ground truth (Ch7's Layer 4 business metrics are often too slow for this — a 60-second requirement likely needs to key off a faster signal, like an immediate output-distribution anomaly or an error-rate spike, rather than waiting for actual outcome data).

**Step 4 — Deployment strategy:** Pre-defined, automatic thresholds (Ch5's rollback principle — "decide before, not after") that, when breached, trigger the rollback without waiting for human confirmation. This is a direct, concrete application of that principle from Chapter 5, now with a hard number attached.

**Step 5 — Monitoring & failure modes:** Worth flagging a genuine risk here: an automatic, fast rollback mechanism could itself misfire on a noisy metric spike that isn't a real problem, causing unnecessary churn. A strong answer acknowledges this tension (speed vs. false-positive rollback) rather than presenting automatic fast rollback as purely upside.

---

## 7. Common pitfall interviewers listen for, across all of these

Jumping straight into naming tools/strategies before establishing requirements is the single biggest way candidates lose points on system design questions — it produces a technically correct-sounding answer that isn't actually justified by, or connected to, the specific scenario asked about. Every worked example above deliberately shows requirements/constraints (Steps 1-2) driving the *specific* choices in Steps 3-5, not the other way around. If you notice yourself naming a strategy before you've stated the latency budget, stakes, or traffic pattern, back up.

---

## Final comprehension check — no more chapters after this, so make this one count

1. Walk through the five-step framework from memory, in order, without looking back at Section 2.
2. Pick one worked example above and, without re-reading it, redo Steps 1-2 (requirements and constraints) in your own words — then check how closely your reasoning matches what drove the later steps.
3. Design your own: "A voice assistant's speech-to-text model needs to run on-device (Apple-relevant), with limited compute and battery budget, and needs to keep working even with no internet connection." Walk through all five steps. This one specifically should pull hard on Chapter 6 (compression techniques — quantization/pruning/distillation are especially relevant for on-device) and should make you think about what changes when there's no live serving infrastructure to canary against at all.

---

## You've completed the syllabus

All 11 modules are done. If you want, tell me which chapters felt shakiest and I can build a focused review — or we can turn any of the worked examples above into a mock interview where you talk through the five steps out loud and I push back like an interviewer would.
