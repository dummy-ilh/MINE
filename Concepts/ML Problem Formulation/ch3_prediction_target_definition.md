# Chapter 3: Defining the Prediction Target — Granularity, Time Windows, and Label Construction

## 1. Why This Chapter Exists

Chapters 1–2 got you to "predict expected watch time, as a ranking task." That still isn't a specification you can hand to a data engineer. You need to answer: watch time predicted *for what exact unit*? Over *what window*? Measured *how*, exactly, in the event logs? This chapter is about closing the gap between a conceptual target and an operational one — the step where most of the subtle bugs in real ML systems are born (label leakage, mismatched windows, silently wrong ground truth).

## 2. Three Axes That Define a Target Precisely

**Axis 1 — Unit of prediction (the "row" in your training set).** Is a training example a user? A (user, day)? A (user, item)? A (user, item, session)? Getting this wrong either makes the problem intractable (predicting something per-user when the true phenomenon varies per-session) or throws away signal (aggregating away a real (user,item) interaction effect into a per-user average).

**Axis 2 — Time window (both feature window and label window).** Two separate windows matter and are frequently confused:
- *Feature window*: how far back do we look to build features (last 7 days of activity? last 30?).
- *Label window*: how far forward do we look to decide the outcome (did the user churn *within 30 days* of this prediction? did the transaction turn out to be fraud *within the 60-day chargeback window*?).

**Axis 3 — Ground-truth operationalization.** The conceptual label ("did the user churn") must be translated into an exact, queryable definition ("zero sessions in the 30 days following day $t$, for a user with at least 1 session in the preceding 30 days"). Ambiguity here (what counts as a "session"? does a push-notification-driven app-open count?) causes label noise that no model architecture can fix.

## 3. Worked Numerical Example: Defining "Churn" Precisely

Business objective: reduce churn. Naïve target: "predict if user churns." Let's operationalize it with actual numbers and see how the definition changes the label for the same user.

Say we define churn as **"zero logins in the 30 days after the prediction date $t$."** Consider a user with login timestamps (days since account creation): $\{5, 12, 40, 41, 42, 100\}$. Let $t = 45$.

- Feature window: e.g., last 30 days before $t$, i.e., days 15–45 → this user has logins on day 40, 41, 42 → 3 logins in feature window, recency = 3 days before $t$.
- Label window: 30 days after $t$, i.e., days 45–75 → no logins in that range → **label = churned (1)**.
- But the user *does* return on day 100 — 55 days after $t$, outside our 30-day label window.

This is not a bug — it's a definitional choice, but it means our label is "did not return within 30 days," not "never returned," and these produce a different label for the exact same user. If the business actually cares about 90-day retention, this 30-day-window target *systematically mislabels* users who take long breaks and return, and the model faithfully learns to predict "quiet for 30 days" — a real but possibly business-irrelevant phenomenon if what leadership actually means by "churn" is "gone for good."

**Numeric consequence:** if 8% of "30-day churners" in your labeled data actually return within 90 days (as in the example above), your model is effectively 8 percentage points "wrong" relative to a 90-day-window definition of the same word "churn" — not because of model error, but because of a target-definition mismatch that never gets surfaced unless you explicitly state which window you used.

## 4. Formalizing the Label Window Tradeoff

For a label window of length $W$, define the *label maturity delay* as $W$ itself — you cannot know the true label until $W$ days after the prediction point. This creates a direct tradeoff:

$$
\text{Usable training data volume}(W) \;\propto\; \big(T_{\text{now}} - W\big)
$$

where $T_{\text{now}}$ is the current date, since you can only form a *known* label for examples where the full window has already elapsed. Larger $W$ (e.g., 90-day churn vs. 30-day churn) gives a more business-faithful label but leaves you with less usable labeled history and staler training data (your most recent 90 days of predictions have no label yet at all). This is a real tradeoff to state explicitly, not an afterthought: **label fidelity vs. label freshness/volume.**

## 5. Label Leakage: The Silent Killer of Target Definitions

A feature "leaks" the label when it encodes information that could only exist *because* the label's future outcome already happened. Two numeric examples:

**Example A (obvious once stated).** Predicting 30-day churn, and one of your "features" is "number of logins in the 30 days following $t$" — accidentally computed with the wrong window boundary in a pipeline bug. This feature has near-perfect correlation with the label ($r \approx -0.98$ in a quick sanity check) because it's numerically almost the label's complement. A suspiciously high offline AUC (e.g., 0.99 on a task where reasonable models get 0.75–0.80) is the standard tell.

**Example B (subtler).** Predicting fraud at transaction time, but a feature is "was this transaction ever refunded" — refunds for confirmed fraud often happen *after* the fraud investigation concludes, meaning this feature is only populated (non-null) for transactions that have already been adjudicated, which correlates strongly with the label by construction, not by any predictive relationship available at serving time.

**The interview-relevant check:** for every feature, ask "would this value be available, in this exact form, at the moment we need to serve a prediction — before the label window has elapsed?" If not, it's leakage regardless of how much it improves offline metrics.

## 6. Production Considerations

- **Training/serving skew from window misalignment.** If your training pipeline computes the feature window as "30 days ending at label-creation time" but your serving pipeline computes it as "30 days ending at request time," these are subtly different reference points and will silently degrade live performance relative to offline metrics.
- **Label window length directly gates model refresh cadence.** A 90-day label window structurally limits you to retraining on labels that are, at best, 90 days stale — this needs to be an explicit constraint communicated to stakeholders expecting "real-time learning."
- **Backfilling label definitions is expensive and often incomplete.** If the business redefines "churn" from 30 to 60 days after your system is live, historical labels may need full reconstruction from raw event logs, not a quick metric recomputation — flag this cost early rather than after the redefinition request arrives.
- **Unit-of-prediction changes are effectively new problems.** Moving from per-user churn to per-subscription churn (a user might have multiple subscriptions) is not a metric tweak; it's a full label/feature-table redesign.

## 7. Common Interview Traps

- **Stating a target without a time window at all** ("predict churn") — an L5 interviewer will immediately probe "over what window?" and a candidate without an answer loses significant credibility.
- **Not distinguishing feature window from label window** — conflating them (e.g., using data from *inside* the label window as a feature) is exactly how leakage (Section 5) sneaks in.
- **Treating a too-good-to-be-true offline AUC as a win rather than a leakage red flag.** Section 5's Example A is a classic trap: candidates who don't flag anomalously high offline metrics as suspicious miss an easy signal of rigor.
- **Ignoring the label-maturity/data-freshness tradeoff** and proposing a long label window with no acknowledgment that it costs you retraining freshness (Section 4).

## 8. L5-Differentiating Talking Points

- Proactively separate "feature window" from "label window" in your very first pass at defining a target, using those exact terms, before an interviewer has to ask.
- Bring up the label-maturity-vs-freshness tradeoff (Section 4) unprompted when proposing any target with a forward-looking window (churn, LTV, fraud with a chargeback window).
- When quoting an offline metric in a hypothetical, explicitly note what magnitude would make you suspicious of leakage (e.g., "if I saw 0.99 AUC on a churn task, I'd suspect leakage before celebrating").
- Note that target *redefinition* (e.g., stakeholders changing what "churn" means) is a recurring, expected event in production ML, not a one-time setup decision — and that this has real backfilling cost.

## 9. Comprehension Checks

1. For a subscription business, define "churn" three different ways using different label windows (e.g., 7-day, 30-day, "cancels subscription explicitly"). For each, state one business decision it would and would not be well-suited to support.
2. In the churn example in Section 3, if the label window were extended to 60 days instead of 30, would this specific user (logins at days 5, 12, 40, 41, 42, 100, with $t=45$) be labeled churned or not? Show the reasoning.
3. Explain, using the "would this feature be available at serving time" test from Section 5, why "total refunds issued for this transaction" is a leakage risk for a fraud model but "average refund rate for this merchant over the last 6 months" is not.
4. A stakeholder wants near-real-time retraining (daily) but also wants labels defined over a 45-day forward window. Explain, using Section 4's relationship, why these two requirements are in tension, and propose one way to reconcile them (e.g., a fast-but-noisier proxy label plus a slower, high-fidelity label for periodic recalibration).

---

*End of Chapter 3. Chapter 4 will cover label design and ground truth in more depth — implicit vs. explicit labels, delayed feedback, and noisy-label mitigation strategies.*
