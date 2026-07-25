# Chapter 5 — Deployment Strategies

*(Module 5 of the syllabus)*

---

## 1. The core question every strategy answers

You have a new model, validated offline (Ch1-2), packaged and served correctly (Ch3), and you've checked it isn't going to suffer from an obvious skew problem (Ch4). Now: **how do you actually let it start touching real users, without betting the whole system on it being right?**

Every deployment strategy in this chapter is really just a different answer to one question: **how much real traffic do you expose to the new model, how gradually, and how do you decide to keep going or pull back?** Notice this is fundamentally a *risk management* problem, not a technical one — which is why interviewers expect you to justify your choice based on stated constraints (risk tolerance, traffic volume, how fast you need a signal), not just recite definitions.

---

## 2. Shadow Deployment

**What it is:** the new model runs in production, receiving a copy of real live traffic, and makes real predictions — but those predictions are only **logged**, never actually shown to users or acted upon. The old model continues to be the one actually serving users, unaffected.

**Why you'd use it:** it's the lowest-risk way to answer "how would this new model actually perform on real, live traffic?" — with zero chance of a bad prediction ever reaching a real user, because its output goes nowhere except a log.

**What you learn from it:** you can compare the shadow model's predictions against the current production model's predictions on the *exact same* live requests, and (once ground truth becomes available, e.g. did the transaction turn out to be fraud) evaluate real-world accuracy before ever risking exposure.

**What it can't tell you:** shadow deployment only tells you how the model *predicts* — it can't tell you how the model *changes user behavior*, because users never actually see its outputs. This limitation is the natural setup for A/B testing later in this chapter.

**Cost consideration worth mentioning:** running a shadow model means running double the inference compute (both models processing every request), so it's not free — this is a legitimate tradeoff to flag if asked "any downsides?"

---

## 3. Canary Release

**What it is:** the new model is given a small percentage of real live traffic (say, 1-5%) — and this traffic is genuinely served the new model's actual predictions, not just logged. The rest of traffic continues on the old model. If the canary traffic looks healthy after monitoring, the percentage is gradually increased (5% → 25% → 100%) over time.

**Why the name "canary":** it's a reference to canaries historically used in coal mines to detect dangerous gas early — a small amount of exposure gives you an early warning before a large-scale problem.

**Why you'd use it:** unlike shadow deployment, canary release actually tests the model's real impact on real users — including things shadow deployment can't catch, like actual latency under real production load, or real user reactions/behavior changes. But by limiting the blast radius to a small percentage, a bad model only affects a small slice of users, and you can catch problems before rolling out further.

**Key design decision — what triggers a rollback/pause during canary:** you need pre-defined thresholds (e.g., error rate, latency, key business metric) that, if breached during the canary phase, automatically halt or reverse the rollout. This connects directly to the "decide before deploying" principle covered later in this chapter.

---

## 4. Blue-Green Deployment

**What it is:** you maintain **two full, complete production environments** — call them "blue" (currently live, old model) and "green" (fully set up with the new model, but not yet receiving traffic). When you're ready to switch, you redirect *all* traffic from blue to green **instantly** — not gradually.

**Why you'd use it:** the main benefit is **instant rollback** — if something goes wrong with green, you just redirect traffic back to blue immediately, since blue was never torn down and is still fully running. There's no "in-between" state where some traffic is on old and some is on new (unlike canary), which can simplify certain kinds of debugging and avoid inconsistent user experiences.

**Tradeoff vs. canary:** blue-green gives you instant, clean rollback, but it does *not* give you gradual, limited-exposure risk testing — you either fully commit to green or you don't, there's no "let's see how 5% does first." It also costs more infrastructure, since you're running two complete environments simultaneously, at least during the transition window.

**When to reach for blue-green over canary in an interview answer:** situations where you need a clean, instant cutover — e.g., changes where gradually mixing old and new behavior would itself cause problems (like a change to something that must be globally consistent) — versus canary being better when you specifically want to *learn* from limited real-world exposure before committing further.

---

## 5. A/B Testing

**What it is:** distinct from the strategies above in an important way — A/B testing isn't primarily about *safely rolling out* a model, it's about **rigorously measuring which of two models is actually better**, using real user-facing outcomes, not just offline ML metrics.

**Why this is a separate concept from canary, even though both involve splitting traffic:** a canary release's traffic split is temporary and directional (working toward 100% on the new model, assuming it's healthy). An A/B test's traffic split is often a genuinely fixed, controlled experiment (e.g., a stable 50/50 split held for a set period), specifically designed to produce a statistically valid comparison — the goal is *measurement*, not *rollout*.

**Why offline ML metrics aren't the end of the story:** a model might have better offline accuracy but somehow produce worse *business* outcomes (e.g., a recommendation model that's more "accurate" by some offline metric but actually reduces user engagement or revenue). A/B testing measures the metric that actually matters to the product — often a business metric — not just the ML metric.

**Concepts you should be able to define if asked:**
- **Statistical significance** — the change you're seeing between groups A and B is unlikely to be due to random chance alone, evaluated against a pre-defined confidence threshold.
- **Guardrail metrics** — secondary metrics you monitor to make sure the new model isn't quietly harming something else, even if the primary metric improves (e.g., a recommendation model that boosts click-through but tanks user-reported satisfaction).
- **Novelty effect** — users may temporarily react differently to *anything* new, simply because it's new/different, independent of whether it's actually better — meaning short A/B tests can be misleading, and effects sometimes need to be measured over a longer window to confirm they're real and lasting.

---

## 6. Putting it together — which strategy, when?

| Strategy | Real users affected? | Primary purpose | Rollback speed |
|---|---|---|---|
| Shadow | No (logged only) | Validate predictions on real traffic, zero risk | N/A — never live |
| Canary | Yes, small % | Gradual, risk-limited rollout with real impact | Fast (reduce %) |
| Blue-Green | Yes, all-or-nothing | Instant, clean cutover | Instant (flip back) |
| A/B Test | Yes, controlled split | Rigorous comparison of business impact | Depends on design |

A well-structured interview answer to "design a rollout for X" often *combines* these in sequence: **shadow first** (validate predictions are sane, zero risk) → **canary** (confirm real-world health at small scale) → **A/B test** (rigorously confirm it's actually better for the business, not just non-broken) → **full rollout**. Being able to sequence these, rather than picking just one, is what separates a strong answer from a list-the-definitions answer.

---

## 7. Rollback strategy — decide *before*, not *after*

A critical principle, worth stating explicitly in any deployment design answer: **the conditions that trigger a rollback, and the mechanism to execute it, must be decided and built *before* the deployment happens — not improvised in the moment during an incident.**

Why this matters: during an actual production incident, you're under time pressure, possibly paged at an inconvenient hour, and reasoning clearly about thresholds is harder than it sounds. If "what counts as bad enough to roll back" isn't already a pre-agreed, automatable threshold, you get slow, inconsistent, high-stress decisions exactly when you can least afford them. This is why canary releases pair with pre-set automatic thresholds, and why blue-green's core value proposition *is* the pre-built instant-rollback path.

---

## Comprehension check

1. Explain why shadow deployment, despite testing "real" traffic, still can't tell you whether a model actually improves user experience.
2. A teammate proposes skipping canary and going straight to a 50/50 A/B test for a brand-new model nobody has validated on live traffic yet. What risk are they introducing, and what would you suggest instead?
3. Describe a rollout plan (using two or more of these strategies in sequence) for a new fraud-detection model at a payments company. Justify your ordering.

Say "c6" when ready for **Chapter 6: Latency vs. Accuracy Tradeoffs**.
