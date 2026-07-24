# Chapter 1: From Business Objective to ML Task

## 1. Why This Chapter Exists

Every ML interview question starts with a sentence like "we want to increase user engagement" or "reduce fraud losses" or "help users find what they're looking for faster." None of these is an ML problem yet. An ML problem needs:

- A **prediction target** (what specific quantity are we estimating?)
- A **unit of prediction** (per user? per session? per item? per (user, item) pair?)
- A **decision** that consumes the prediction (what does the system *do* with the output?)

The single most common failure mode at the L5 bar isn't picking the wrong algorithm — it's skipping straight to "I'll build a gradient boosted model" without ever pinning down what the model is predicting and why that prediction, if perfect, would actually move the business metric. Interviewers are explicitly listening for whether you do this translation step, out loud, before touching architecture.

## 2. The Translation Framework

Think of the translation as a chain with four links, each of which can break:

```
Business Objective  →  Decision  →  Prediction Target  →  Ranking/Model Task
   (why we care)        (what we do)   (what we estimate)    (how we estimate it)
```

**Step 1 — Business Objective.** State it as a metric the company actually tracks: revenue, DAU, churn rate, support cost, fraud loss in dollars. If you can't name the metric, you don't have an objective yet, you have a slogan.

**Step 2 — Decision.** Identify the concrete action a system takes as a result of the model's output. A model that predicts something nobody acts on is worthless regardless of its accuracy. Examples: show this ad, rank these ten items, block this transaction, send this notification, route this ticket to a human.

**Step 3 — Prediction Target.** Given the decision, work backward to the minimal thing you need to predict to make that decision well. This is where most of the design freedom — and most of the interview signal — lives, because usually there are several plausible targets and picking well requires judgement about data availability, latency, and how directly the target relates to the objective.

**Step 4 — Model Task.** Only now do you pick classification/regression/ranking/etc. (Chapter 2 covers this mapping in depth.)

### Worked Numerical Example: Video Platform "Increase Watch Time"

Business objective: increase total watch-hours per DAU (currently 42 minutes/day average, target +10%).

Naively, someone jumps to "predict watch time per video, rank by it." Let's stress-test that with numbers.

Suppose we have candidate videos $A, B, C$ for a user, with true (unknown) watch probabilities and durations:

| Video | P(click) | E[watch time \| click] | Video length | E[watch time] (unconditional) |
|---|---|---|---|---|
| A | 0.30 | 25 min | 28 min | 7.5 min |
| B | 0.10 | 55 min | 60 min | 5.5 min |
| C | 0.60 | 4 min | 5 min | 2.4 min |

If the decision is "rank by predicted unconditional watch time" ($P(\text{click}) \times E[\text{watch} \mid \text{click}]$), we'd rank A > B > C, surfacing the video that actually maximizes expected watch-hours. If instead someone had (incorrectly) formulated the target as "predict P(click)" alone — because click data is easy to log and label — we'd rank C > A > B, which optimizes for clickbait, not watch time. This is a concrete illustration of Step 3 breaking: the *proxy* target (click) diverges from the *true* target (watch time) implied by the business objective.

This is exactly the kind of arithmetic an interviewer wants to see you produce unprompted — not just "click prediction is a bad proxy" as an assertion, but a small table showing *how* it's a bad proxy.

## 3. A Formal Way to State the Chain

You can write the expected business metric as an expectation over the decision policy $\pi$ and the true outcome distribution:

$$
\mathbb{E}[\text{Business Metric}] = \mathbb{E}_{x \sim \text{traffic}}\Big[ \, \sum_{a \in \mathcal{A}} \pi(a \mid x) \cdot R(x, a) \, \Big]
$$

where $x$ is the context (user, session, candidate set), $\mathcal{A}$ is the set of possible decisions, $\pi(a\mid x)$ is the policy induced by your ranking/model, and $R(x,a)$ is the true reward (e.g., watch time) if decision $a$ is taken in context $x$.

Your model doesn't get to observe $R(x,a)$ directly at serving time — it produces an estimate $\hat{R}(x,a)$, and $\pi$ is typically "pick $\arg\max_a \hat{R}(x,a)$" or a softmax over it. The entire problem-formulation exercise is: **choose a prediction target $\hat{R}$ such that optimizing it under realistic data constraints gets you as close as possible to optimizing the true $R$, using data you can actually collect.**

This formula is worth memorizing not to recite verbatim, but because it gives you a checklist: for any target you propose, ask (a) does $\hat R \approx R$, and (b) is $\pi$ built from $\hat R$ actually implementable (latency, candidate set size, data freshness)?

## 4. Diagnostic Questions to Ask Out Loud in an Interview

When given a vague prompt, work through these before saying the word "model":

1. **What decision is downstream of this prediction?** If the interviewer can't answer, that's a signal the problem is underspecified — surface it explicitly rather than guessing.
2. **What's the unit of prediction?** Per-user? Per-(user,item)? Per-session? This changes your entire data pipeline and label definition.
3. **What would a perfect oracle predictor let us do, and would that actually move the business metric?** This is the check against building an accurate model that's aimed at the wrong target (like the click-vs-watch-time example above).
4. **What's the feedback latency?** Fraud labels might arrive in 60 days (chargeback window); churn labels in 30 days; click labels in milliseconds. This affects whether your target is even trainable at the cadence the business wants.
5. **Is there a simpler non-ML decision rule that already captures most of the value?** (Chapter 10 covers baselines, but flagging this early shows maturity — sometimes the answer to "how do we increase watch time" is a UI change, not a model.)

## 5. Production Considerations

- **Target drift from business-metric drift.** Business objectives get renegotiated (e.g., "watch time" quietly becomes "watch time excluding autoplay" after a policy change). Your prediction target needs an owner and a versioned definition, or your offline metrics silently stop meaning what stakeholders think they mean.
- **Multiple decisions, one model.** Often several different downstream decisions (ranking, filtering, budget allocation) all want to consume the "same" prediction. Resist building one entangled target for all of them — Chapter 8 covers multi-objective formulations for this.
- **Instrumentation precedes modeling.** If the event that defines your prediction target isn't logged yet (e.g., "did the user actually read the article" vs. "did the page load"), the correct next step is a logging/instrumentation project, not a model — say this explicitly if it applies.
- **Reward misspecification compounds under optimization.** Because the model actively selects which items users see, a target that's even slightly misaligned with the true objective gets *amplified* by feedback loops (preview of Chapter 11) — small proxy errors become large systemic biases over time as the model's own outputs shape future training data.

## 6. Common Interview Traps

- **Jumping to algorithm choice before objective is stated.** ("I'd use XGBoost" in response to "how would you increase engagement" — with nothing in between — reads as pattern-matching, not reasoning.)
- **Conflating "what's easy to label" with "what we should predict."** Click data is abundant and cheap; that doesn't make click-through the right target if the objective is watch time, retention, or revenue.
- **Ignoring the decision entirely.** Some candidates describe a model that predicts something reasonable but never state what action the system takes with the prediction — this is the single most common gap between L4 and L5 answers on this topic.
- **Treating the business objective as fixed and unquestionable.** Senior candidates probe whether the stated objective is even the right one to optimize (e.g., watch time vs. long-term retention can trade off against each other — optimizing pure watch time can promote addictive-but-low-satisfaction content).

## 7. L5-Differentiating Talking Points

- Explicitly write out (even just verbally) the four-link chain — objective, decision, target, task — before proposing any model. This signals structured thinking rather than pattern-matched recall.
- Proactively raise the proxy-metric gap (Section 2's worked example) unprompted, and quantify it with a small example rather than asserting it abstractly.
- Mention that the prediction target and the online business metric should be periodically reconciled (e.g., a quarterly review comparing model gains offline to actual metric movement online) — this shows awareness that the mapping in Step 3 isn't "solved once," it needs governance.
- Bring up short-term vs. long-term objective tension (e.g., engagement vs. well-being, revenue vs. trust) as a natural extension of "is this the right business objective" — L5 candidates are expected to push back on ill-posed objectives, not just accept them.

## 8. Comprehension Checks

1. A fintech company says "reduce fraud losses." Walk through the four-link chain: what's a plausible decision, prediction target, and model task? What data-latency issue would you flag immediately?
2. Why is "predict whether the user clicks" often a worse target than "predict expected watch time," even though click labels are far easier to obtain? Use the reward-decomposition formula from Section 3 to justify your answer.
3. Give an example (different from the video platform one) where optimizing a well-chosen proxy metric offline produced a *worse* outcome on the true business metric online. What does this imply about how you should treat offline metric improvements during model development?
4. A stakeholder asks you to "just build a model to predict user churn." List three clarifying questions from Section 4 you'd ask before writing a single line of feature engineering code, and explain what a bad answer to each would derail.

---

*End of Chapter 1. Chapter 2 will cover the task taxonomy (classification vs. regression vs. ranking vs. generative) and the decision criteria for choosing among them given a fixed prediction target.*
