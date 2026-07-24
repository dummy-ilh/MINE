# Chapter 11: Feedback Loops & Selection Bias

## 1. Why This Chapter Exists

This chapter formalizes something earlier chapters kept gesturing at: once a model is deployed, it doesn't just make predictions — it *shapes the data it will be trained on next*. This is fundamentally different from the standard supervised-learning assumption (train and test data are i.i.d. from the same distribution). Recognizing and correcting for this is one of the clearest markers of senior ML thinking, because it means an offline-clean-looking pipeline can be quietly self-reinforcing a bias nobody chose deliberately.

## 2. The Core Mechanism: Logged Data Is Policy-Dependent

Suppose a ranking model $\pi_{\text{old}}$ decides what gets shown. Users can only click, purchase, or rate items they were *shown*. So your training data — clicks, purchases, ratings — is drawn from:

$$
P_{\text{observed}}(x, y) = P(x) \cdot \pi_{\text{old}}(\text{shown} \mid x) \cdot P(y \mid x, \text{shown})
$$

Items your old policy rarely or never showed are underrepresented (or entirely absent) in the training data for the *next* model, regardless of their true quality — this is exactly the "rich-get-richer" dynamic previewed in Chapter 9's item cold-start discussion, but now framed as a general property of any closed-loop, self-training system, not just brand-new items.

## 3. Worked Numerical Example: A Feedback Loop Amplifying an Initial Bias

Suppose two items, X and Y, both have true relevance 0.6 (equally good), but due to an early arbitrary tie-break, the first model shows X 90% of the time and Y 10% of the time to a fixed audience of 10,000 users/day.

**Day 1 (using the initial model)**: X gets 9,000 impressions, generating (at true 0.6 relevance, with some click noise) roughly 5,400 clicks (n=9,000). Y gets 1,000 impressions, roughly 600 clicks (n=1,000). Both have an *observed* click rate near 0.60 — consistent with true relevance — but note the **sample size disparity**: your estimate of Y's relevance is statistically noisier (n=1,000) than X's (n=9,000).

**Day 2 (retrain on Day 1 data, model overfits slightly to noise given the imbalanced sample sizes)**: due to sampling noise, Y's estimated relevance comes back at 0.57 (a random miss, not a true difference) while X's comes back precisely at 0.60 (more data, less noise). The retrained model, trusting its (noisier but nominally lower) estimate for Y, now shows X 95% of the time and Y 5% of the time.

**Day N (repeated over many cycles)**: Y's impression share keeps shrinking, which keeps its relevance-estimate noise high, which keeps producing occasional low-relevance-estimate draws purely by chance, which keeps suppressing its impression share further — a self-reinforcing spiral where an item that is *truly* just as good as X can be driven toward near-total suppression, **not because it's actually worse, but because early noise plus a closed feedback loop compounds small estimation errors into large exposure differences.** This is the concrete mechanism behind why naive retraining on logged, policy-biased data without correction can systematically and unfairly suppress content/candidates that never got a fair, sufficiently-sized initial trial.

## 4. Counterfactual Correction: Inverse Propensity Scoring (IPS)

The standard correction technique re-weights logged examples by the *inverse* of the probability they were shown under the logging policy, producing an unbiased estimate of what would have happened under a *different* (e.g., uniform-random) policy:

$$
\hat{V}_{\text{IPS}}(\pi_{\text{new}}) = \frac{1}{n}\sum_{i=1}^n \frac{\pi_{\text{new}}(a_i \mid x_i)}{\pi_{\text{old}}(a_i \mid x_i)} \cdot r_i
$$

where $r_i$ is the observed reward (e.g., click) for the action actually logged, and the ratio $\pi_{\text{new}}/\pi_{\text{old}}$ re-weights each observed example by how much more (or less) likely the new policy would have been to take that same action.

**Worked numeric example**: item Y was shown with probability $\pi_{\text{old}}(Y) = 0.05$ under the logging policy, and it received a click ($r=1$). If your proposed new policy $\pi_{\text{new}}$ would show Y with probability 0.30 (much more often — say, because you're evaluating a "more exploratory" candidate policy), this single observation gets re-weighted by $0.30 / 0.05 = 6$, i.e., counted as if it were 6 independent observations, since a rare event under the old policy that turned out positive is extra-informative about how a policy that shows it more often would perform. This directly counteracts the sample-size disparity from Section 3 — items that were shown rarely get their few observations *upweighted*, rather than simply having a noisier, unweighted estimate silently used as if it were reliable.

**Practical caveat**: IPS estimates have high variance when $\pi_{\text{old}}(a_i \mid x_i)$ is very small (dividing by a tiny number), so real systems often use variance-reduction techniques (e.g., clipped/capped propensity weights, or doubly-robust estimators that combine IPS with a separate reward model) rather than raw IPS.

## 5. Structural Fixes: Deliberate Exploration and Logging Policy Design

Beyond correcting after the fact (Section 4), a more robust fix is designing the *logging policy itself* to avoid creating severe selection bias in the first place:

- **Randomization/exploration budget**: reserve a small fraction of traffic (e.g., 2–5%) for near-uniform-random exposure across candidates, specifically to generate an unbiased sample for future model training — distinct from Chapter 9's UCB-style exploration (which optimizes user experience *during* exploration) though related in spirit; here the emphasis is on guaranteeing enough unbiased data exists at all.
- **Propensity logging**: log $\pi_{\text{old}}(a_i \mid x_i)$ itself alongside every served impression — without this logged value, IPS correction (Section 4) is impossible to apply retroactively, so this needs to be built into the serving/logging pipeline from day one, not added after the fact once a bias is discovered.
- **Periodic full re-evaluation on randomized data**: even a small held-out slice of near-random exposure, evaluated periodically, serves as a bias-free sanity check against the main, policy-biased training pipeline drifting in an unrecognized direction.

## 6. Production Considerations

- **Propensity logging (Section 5) has a real infrastructure cost** and must be planned into the serving architecture from the start — retrofitting it after a feedback-loop problem is discovered means you have no way to correct the *historical* data, only future data going forward.
- **Exploration traffic (Section 5) has a real short-term cost** (worse experience for the users in the exploration bucket) that should be sized and monitored as its own guardrail metric, exactly mirroring Chapter 8's Pareto-tradeoff framing — a business decision, not a purely technical one.
- **Feedback-loop bias tends to be invisible in standard offline evaluation** because the offline test set is drawn from the *same* biased logging policy as the training set (an extension of Chapter 6's Mechanism A) — a held-out randomized-exposure slice (Section 5) is often the only way to detect it at all.
- **IPS/doubly-robust corrections should be validated periodically**, not applied once and trusted forever — as the logging policy itself changes over model iterations, the propensity estimates used for correction need to stay synchronized with the actual policy that generated the data.

## 7. Common Interview Traps

- **Not recognizing that a deployed model changes the distribution of its own future training data** — treating logged production data as if it were an i.i.d. sample from some fixed, policy-independent distribution is the single deepest conceptual gap this chapter targets.
- **Proposing to "just collect more data" without recognizing that more data collected under the same biased policy doesn't fix the bias** — it can, as Section 3 shows, actually *compound* it over successive retraining cycles.
- **Not connecting item/content suppression dynamics back to fairness implications** — a feedback loop that suppresses content from newer or smaller creators (who start with an inherent exposure disadvantage) is a concrete equity concern, not just a technical curiosity.
- **Proposing IPS without acknowledging its variance problem** (Section 4) under small propensities — a giveaway of only surface familiarity with the technique.

## 8. L5-Differentiating Talking Points

- Explicitly state that logged production data is generated by a policy, and that this makes naive retraining on it structurally different from standard i.i.d. supervised learning — framing this as a first-class concern before discussing any specific correction technique.
- Propose propensity logging and a dedicated exploration budget (Section 5) as *upfront infrastructure decisions*, not retrofits — showing awareness that feedback-loop bias is best prevented by design, not corrected after discovery.
- Bring up IPS/doubly-robust estimation by name when discussing how to fairly evaluate a new policy using old, biased logged data, and note the variance tradeoff of raw IPS.
- Connect feedback-loop amplification explicitly to fairness/equity concerns for historically under-exposed candidates (new creators, minority-language content, small sellers), broadening the technical discussion to its real-world stakes.

## 9. Comprehension Checks

1. Using Section 3's numeric spiral, explain in your own words why more data collected under the same biased logging policy doesn't fix the underlying bias, and can even make it worse over successive retraining rounds.
2. Using the IPS formula in Section 4, compute the re-weighting factor for an observation where $\pi_{\text{old}}(a_i\mid x_i) = 0.02$ and $\pi_{\text{new}}(a_i \mid x_i) = 0.10$, and explain qualitatively why raw IPS becomes high-variance as $\pi_{\text{old}}$ shrinks toward zero.
3. A company discovers, three years into running a recommendation system, that certain content categories have been persistently under-shown despite no evidence they're actually lower quality. Using Section 5, explain what upfront infrastructure decision, if made three years earlier, would have allowed them to definitively test this hypothesis today — and why they can't fully answer it retroactively without that infrastructure.
4. Explain why a held-out slice of near-random exposure traffic (Section 5) can catch a feedback-loop bias that standard offline evaluation (train/test split from the same logged data) would miss entirely.

---

*End of Chapter 11. Chapter 12 will present worked case-study formulations end-to-end (news feed ranking objective, churn prediction, fraud detection), each contrasted at L5-vs-lower-level answer depth, tying together all preceding chapters.*
