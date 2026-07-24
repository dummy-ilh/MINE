# Chapter 12: Case Studies — End-to-End Formulations

## 1. Why This Chapter Exists

Chapters 1–11 each isolated one link in the formulation chain. Real interview questions ("design a system to do X") demand you walk the *entire* chain fluently, under time pressure, while an interviewer probes gaps. This chapter works three case studies end-to-end, explicitly citing which chapter's tool is being applied at each step, and contrasts a typical L4-level answer against an L5-differentiated one for each.

## 2. Case Study A: News Feed Ranking Objective

**Prompt**: "Design the ranking objective for a social media news feed."

**Step 1 — Business objective → decision (Ch. 1).** State the metric explicitly: e.g., long-term user retention and healthy engagement, *not* just raw time-on-app (a known Ch. 1 proxy trap — pure time-on-app can be driven by outrage/addictive content that damages long-term retention, echoing Ch. 6's Mechanism B). Decision: which ~20 of a user's ~2,000 available candidate posts to show, in what order, per app session.

**Step 2 — Task taxonomy (Ch. 2).** This is retrieval (candidate generation from ~thousands to a few hundred) + ranking (order the reduced set), not a single end-to-end classifier over the full candidate pool — necessary given catalog scale.

**Step 3 — Prediction target (Ch. 3).** Rather than one scalar, decompose into several sub-predictions per (user, post) pair over a fixed short window (e.g., next-session): $P(\text{like})$, $P(\text{comment})$, $P(\text{share})$, $P(\text{report/hide})$, $P(\text{extended dwell time} > \tau)$ — a multi-task setup, since a single proxy (e.g., just $P(\text{click})$) reintroduces Ch. 1's clickbait problem.

**Step 4 — Label design (Ch. 4).** Comments/shares are abundant implicit signals; "report/hide" is a rarer but critical negative signal (needs Ch. 9's rare-event handling — likely trained with class weighting, Ch. 7 Section 3). Watch for delayed labels (a post "aging poorly" — getting reported hours after initial positive engagement).

**Step 5 — Objective function (Ch. 7).** Combine sub-predictions into a single ranking score via a weighted sum:
$$
\text{score} = w_1 \hat P(\text{like}) + w_2 \hat P(\text{comment}) + w_3 \hat P(\text{share}) - w_4 \hat P(\text{report})
$$
with $w_4$ set large relative to the others (Ch. 7's asymmetric-cost reasoning — a single report is far costlier to the platform's health than a missed like).

**Step 6 — Multi-objective/constraints (Ch. 8).** Add a *constraint*, not just a weighted term, for content diversity (e.g., no more than 2 posts from the same source in the top 10) and for "healthy engagement" guardrails (e.g., a floor on the fraction of feed from accounts the user explicitly follows, vs. purely algorithmic discovery) — framed as hard constraints because they're easier to audit/communicate (Ch. 8 Section 3's exact argument) than an implicit weighted tradeoff.

**Step 7 — Offline & online metrics (Ch. 5–6).** Offline: weighted combination of per-task AUC/logloss, evaluated via IPS-corrected offline replay (Ch. 11) given the feed is a closed feedback loop. Online: primary metric close to long-term retention proxy (Ch. 6's proxy-validation method), with guardrails on report-rate, session length distribution, and creator-diversity of impressions (fairness angle from Ch. 11).

**Step 8 — Feedback loop awareness (Ch. 11).** Explicitly flag that engagement-optimized feeds create exactly the exposure-amplification dynamic from Ch. 11 Section 3 — content from smaller/newer creators risks systemic suppression absent a deliberate exploration budget.

**L4 vs. L5 contrast**: an L4 answer typically stops at "predict engagement probability and rank by it." An L5 answer produces the full 8-step decomposition above, explicitly naming the proxy-metric risk, the multi-task decomposition, the asymmetric weighting of negative signals, the diversity constraint, and the feedback-loop/fairness implication — unprompted.

## 3. Case Study B: Churn Prediction

**Prompt**: "Formulate a churn prediction problem for a subscription business."

**Step 1 — Objective → decision (Ch. 1).** Objective: reduce revenue lost to cancellations. Decision: which at-risk users to target with a retention intervention (discount, outreach, feature highlight) — critically, the model must identify users where an *intervention would actually change the outcome*, not just users who happen to be leaving regardless (a subtlety many candidates miss: predicting churn and predicting "save-ability" are different targets).

**Step 2 — Task taxonomy (Ch. 2).** Binary classification (or a survival-analysis / time-to-event formulation, which is often more rigorous since it naturally handles varying observation windows and censored users who haven't churned yet by the data cutoff).

**Step 3 — Prediction target & windows (Ch. 3).** Explicitly define the label window (e.g., 30-day churn) and feature window (e.g., trailing 60 days of usage), and flag the label-maturity/freshness tradeoff (Ch. 3 Section 4) if leadership wants near-real-time flagging.

**Step 4 — Label design (Ch. 4).** Explicit cancellation events are clean labels; watch for "silent churners" who stop using the product but haven't formally cancelled (a selection-bias-adjacent nuance — Ch. 4 Section 5) — the true label of interest may need to blend usage-based and formal-cancellation signals.

**Step 5 — The uplift-modeling subtlety.** Because the decision is "who to intervene on," the *ideal* target isn't $P(\text{churn})$ at all, but the **uplift**: $P(\text{retain} \mid \text{treated}) - P(\text{retain} \mid \text{untreated})$ — a causal quantity, not a simple predictive one. A user with high churn probability but who wouldn't respond to any intervention ("lost cause") shouldn't be targeted; nor should a user with low churn probability who'd churn if annoyed by an unnecessary offer ("sleeping dog"). This requires either a randomized holdout of past interventions (to estimate uplift directly) or a two-model approach as an approximation.

**Step 6 — Objective/metrics (Ch. 5, 7).** Given intervention costs are known (e.g., a $10 discount), a cost-weighted framing (Ch. 7 Section 3) directly compares intervention cost against expected retained-revenue value — an explicit expected-value calculation, not just an AUC number.

**Step 7 — Baseline (Ch. 10).** Baseline: current heuristic (e.g., "target anyone with >30% usage drop last month") — the ML system needs to beat this *specific* baseline, cost-adjusted, not a trivial "predict majority class" baseline.

**L4 vs. L5 contrast**: an L4 answer formulates straightforward churn classification. An L5 answer recognizes the uplift-modeling subtlety in Step 5 — that the actionable target is causal, not merely predictive — which is the single highest-signal insight interviewers listen for on this exact prompt.

## 4. Case Study C: Fraud Detection

**Prompt**: "Formulate a real-time transaction fraud detection system."

**Step 1 — Objective/decision (Ch. 1).** Objective: minimize (fraud losses + false-decline customer friction costs) jointly, not fraud losses alone. Decision: real-time block / hold-for-review / allow, under a strict latency budget (often under 100ms).

**Step 2 — Task taxonomy (Ch. 2).** A layered approach: expected-loss regression (Ch. 2 Section 3, Formulation B) for known fraud patterns, plus anomaly detection (Formulation C) as a safety net for novel attack patterns — explicitly justified by data availability (Ch. 9 Section 5's rare-event reformulation options).

**Step 3 — Label design/delay (Ch. 4).** Chargebacks mature over 60–120 days; propose the dual-cadence strategy (fast rules-engine/merchant-report proxy label + slow confirmed-chargeback ground truth) from Ch. 4 Section 3, explicitly estimating the proxy's $\alpha,\beta$ noise rates via periodic human audit (Ch. 4 Section 4).

**Step 4 — Objective function (Ch. 7).** Weighted/asymmetric loss reflecting the true FN/FP dollar cost ratio (Ch. 7 Section 3's worked 100:1 example applies directly here), balanced against training-stability concerns (moderate the weight, split correction with threshold tuning).

**Step 5 — Metrics (Ch. 5).** AUC-PR (not AUC-ROC, given severe class imbalance — Ch. 5 Section 3), reported alongside a cost-weighted confusion matrix at the actual deployed threshold, not just an aggregate metric.

**Step 6 — Feedback loop (Ch. 11).** Fraudsters actively adapt to the model's decisions (an adversarial, non-stationary feedback loop distinct from Ch. 11's passive exposure-bias case) — this requires continuous retraining cadence and monitoring for concept drift specifically, beyond the standard feedback-loop concerns.

**Step 7 — Baseline (Ch. 10).** The relevant baseline is the current production rules-engine or prior model version, cost-adjusted (Ch. 10 Section 3) — not a trivial always-allow baseline.

**L4 vs. L5 contrast**: an L4 answer proposes a single binary classifier with cross-entropy loss and reports AUC-ROC. An L5 answer proposes the layered known-pattern/novel-pattern architecture, names the specific asymmetric cost structure and encodes it directly into the loss, flags the delayed-label problem with a concrete dual-cadence fix, uses AUC-PR given imbalance, and raises the adversarial (not just passive) feedback-loop concern specific to fraud.

## 5. A General-Purpose Checklist for Any Formulation Prompt

When facing a new, unseen formulation prompt in an interview, work through these in order, out loud:

1. What's the business metric, and what decision is downstream of the prediction? (Ch. 1)
2. What's the right task type given the decision and candidate-set scale? (Ch. 2)
3. What's the exact prediction target — unit, feature window, label window? (Ch. 3)
4. Where do labels come from, are they delayed, and how noisy/biased might they be? (Ch. 4)
5. What offline metric matches the real cost structure, and is the problem imbalanced? (Ch. 5)
6. What online guardrails would catch this offline metric being gamed or diverging from true impact? (Ch. 6)
7. Does the loss function itself need to encode a cost asymmetry, not just the eval threshold? (Ch. 7)
8. Are there multiple competing objectives — and should any be a hard constraint rather than a weighted term? (Ch. 8)
9. Is there a cold-start or rare-event data problem requiring a reformulation, not just "collect more data"? (Ch. 9)
10. What's the actually-relevant baseline, and by how much (cost-adjusted, significance-tested) does the proposal need to beat it? (Ch. 10)
11. Does deploying this model create a feedback loop that biases its own future training data, and does the evaluation need a counterfactual correction? (Ch. 11)

## 6. Comprehension Checks

1. Walk through the 11-point checklist in Section 5 for a new prompt: "formulate a system to detect and de-rank low-quality product listings on an e-commerce marketplace." Give at least a one-sentence answer for each point.
2. In Case Study B, explain precisely why "predict churn probability" and "predict retention uplift from intervention" are different targets, and construct a numeric example (two hypothetical users) where targeting by raw churn probability would waste an intervention budget that targeting by uplift would not.
3. In Case Study C, explain why the fraud feedback loop is described as "adversarial" rather than merely "passive" (as in Chapter 11's general treatment), and why this distinction changes the recommended retraining/monitoring cadence.
4. For Case Study A (news feed), identify which single design choice most directly prevents the Ch. 1 clickbait failure mode from re-emerging, and explain the causal chain from that design choice back to the original problem.

---

*This concludes the 12-chapter ML Problem Formulation curriculum. Together, Chapters 1–11 built the individual tools (objective translation, task taxonomy, target definition, labeling, metrics, objectives, multi-objective tradeoffs, data-scarcity reformulation, baselines, feedback loops), and this chapter exercised them end-to-end against realistic interview prompts, with explicit L4-vs-L5 contrasts throughout.*
