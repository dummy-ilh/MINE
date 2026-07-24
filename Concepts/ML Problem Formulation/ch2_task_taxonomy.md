# Chapter 2: Task Taxonomy — Choosing the Right Model Task

## 1. Why This Chapter Exists

Chapter 1 got you from a business objective to a well-defined prediction target. This chapter covers the last link in the chain: given that target, which *kind* of model task should estimate it? This sounds trivial ("just pick classification or regression") but the taxonomy has more branches than most candidates realize, and picking the wrong branch is a common way to lose points even when Chapter 1's reasoning was flawless — you can correctly identify "predict expected watch time" as the target and still mis-formulate it as a poorly-suited classification task.

## 2. The Taxonomy

| Task type | What it estimates | Canonical output | Typical loss |
|---|---|---|---|
| Binary/multi-class classification | $P(y = k \mid x)$ over a discrete label set | Class probabilities | Cross-entropy |
| Regression | $\mathbb{E}[y \mid x]$ for continuous $y$ | Real number | MSE / MAE / Huber |
| Ranking (learning-to-rank) | Relative order over a candidate set for a given context | Ordered list / scores | Pairwise/listwise (e.g., LambdaRank) |
| Recommendation (retrieval + ranking) | Which items from a huge catalog are relevant to a user | Top-K item list | Contrastive / softmax over catalog |
| Clustering | Latent grouping structure, no labels | Cluster assignment | Within-cluster variance (e.g., k-means objective) |
| Anomaly / outlier detection | $P(x)$ is low, or $x$ deviates from a learned normal-behavior model | Anomaly score / flag | Reconstruction error, density estimate |
| Generative / sequence generation | $P(\text{next token} \mid \text{context})$ or full joint $P(x)$ | Sampled sequence/object | Negative log-likelihood |

The table is a lookup aid, not the reasoning itself. The reasoning is: **what does the decision (from Chapter 1) actually require — a label, a number, an order, a set, a group, a flag, or a novel artifact?**

### Decision Criteria, One Level Deeper

- **Classification vs. regression**: is the target intrinsically discrete (fraud/not-fraud) or continuous (watch time, price)? Careful — some continuous targets get *discretized* into buckets for business reasons (e.g., risk tiers: low/medium/high), which turns a regression problem into ordinal classification. That's a legitimate design choice but should be justified (e.g., the downstream decision only consumes tiers, not exact scores, so a calibrated tier is enough and easier to make robust than an exact real-valued estimate).
- **Regression vs. ranking**: if the decision is "pick the single best value" (e.g., "what price will this house sell for"), regression is natural. If the decision is "order N candidates relative to each other" (e.g., "which 10 of these 500 videos to show"), you rarely need calibrated absolute values — only correct *relative* order — which is exactly what ranking losses optimize for and regression losses don't guarantee.
- **Ranking vs. full recommendation**: ranking assumes you already have a manageable candidate set (dozens to low-hundreds). Recommendation at catalog scale (millions of items) needs a **retrieval** stage (Chapter-9-adjacent: approximate nearest neighbor / two-tower models) to cut the candidate set down before a ranker can be applied — you cannot run a heavy pairwise ranker over a 10-million-item catalog per request.
- **Supervised vs. clustering/anomaly**: do you have (or can you cheaply obtain) labels? If the definition of "positive" is itself unclear or events are too rare/novel to have been labeled (e.g., a brand-new fraud pattern), you're often in unsupervised/anomaly territory, at least until enough labeled examples accumulate to bootstrap a supervised model.
- **Discriminative vs. generative**: does the decision need to *pick among existing options* (discriminative: classification, ranking) or *produce a new artifact* (generative: writing a summary, generating an image, autocompleting a query)?

## 3. Worked Numerical Example: Same Business Objective, Three Valid Task Formulations

Objective: reduce fraudulent transactions (business metric: fraud loss in dollars per month). Decision: block, hold-for-review, or allow a transaction in real time.

**Formulation A — Binary Classification.** Target: $P(\text{fraud} \mid x)$. Suppose the model outputs $\hat p = 0.83$ for a transaction. Policy: block if $\hat p > 0.9$, review if $0.3 < \hat p \le 0.9$, allow otherwise. This transaction goes to review.

**Formulation B — Regression on Expected Loss.** Instead of $P(\text{fraud})$, predict $\mathbb{E}[\text{loss} \mid x] = P(\text{fraud}\mid x) \times \text{transaction amount}$. Consider two transactions both with $\hat p = 0.83$: one for \$20, one for \$20{,}000. Expected loss is \$16.60 vs. \$16{,}600 — wildly different business risk, identical fraud probability. If the true objective is dollars saved rather than fraud-event-count, Formulation B is the better target, since the classification-only formulation A treats a $20 and $20,000 transaction identically once probability is equal, even though blocking the latter mistakenly (a false positive) or missing it (a false negative) has 1000x the dollar consequence.

**Formulation C — Anomaly Detection.** For genuinely novel fraud patterns not yet represented in historical labels, predict an anomaly score based on deviation from the learned distribution of "normal" transactions (e.g., reconstruction error from an autoencoder trained only on legitimate transactions). This doesn't require fraud labels at all, trading off precision for coverage of unseen attack patterns.

All three are defensible; an L5 answer states which one you'd start with and *why*, tying the choice back to what's actually available (labels? dollar amounts? known vs. novel fraud patterns?) — and often proposes layering B for known patterns with C as a safety net for novel ones, rather than picking just one.

## 4. A Formal Note: Why Ranking Losses Differ from Regression Losses

If you regress on a per-item relevance score $\hat y_i = f(x_i)$ and minimize $\sum_i (\hat y_i - y_i)^2$, you are penalizing *absolute* error. But two rankings can have wildly different pointwise error and still induce the *same* correct order, or have near-zero pointwise error and still induce the *wrong* order near the top (which matters most for a Top-K UI). Ranking losses instead directly optimize pairwise or listwise order, e.g. a pairwise logistic loss over pairs $(i,j)$ where $i$ is truly preferred over $j$:

$$
\mathcal{L}_{\text{pairwise}} = \sum_{(i,j):\, y_i > y_j} \log\left(1 + e^{-(\hat y_i - \hat y_j)}\right)
$$

This only penalizes *inversions* (wrong relative order), which is exactly what the ranking decision cares about, and ignores whether the absolute scores are calibrated — a property regression loss cannot give you directly.

**Numerical illustration:** true relevances $y = [3, 2, 1]$ for items $A, B, C$. Model 1 predicts $\hat y = [10, 9, 8]$ (huge absolute error, correct order). Model 2 predicts $\hat y = [3.1, 2.9, 2.8]$ (tiny absolute error on $A$ vs $B$'s true gap, but note $B$'s prediction (2.9) is very close to $C$... let's make it concrete): Model 2 predicts $\hat y = [2.0, 2.1, 1.9]$ (small absolute errors on each point individually, but $B$ and $A$ are inverted: $2.1 > 2.0$ implies $B \succ A$, which is wrong). Squared error for Model 1 is large (≈ 49+49+49), squared error for Model 2 is tiny (≈1+0.01+0.01), yet Model 1 gets the ranking fully correct and Model 2 gets the top-2 order wrong. If your decision is "show items in order," Model 1 is what you want, and only a ranking-aware loss reliably selects for it — pointwise regression loss would favor Model 2 despite its inverted top ranking.

## 5. Production Considerations

- **Task choice constrains your serving architecture.** Retrieval-then-rank (recommendation at scale) needs two separate services with different latency budgets (ANN lookup in single-digit ms, ranker in tens of ms); a single end-to-end classifier over the full catalog is often not servable at all at low latency.
- **Label availability often decides the task for you, not the ideal target.** You may *want* regression on dollar loss (Formulation B above) but only have binary fraud/not-fraud labels historically — in which case you either invest in richer labeling or start with classification as a pragmatic first version, explicitly flagging the target gap this creates versus the true objective.
- **Calibration matters differently per task.** A classifier used purely for ranking doesn't need calibrated probabilities (order is enough). A classifier whose output feeds a downstream dollar-value decision (e.g., expected-loss-weighted blocking threshold) absolutely does need calibration, which may require a post-hoc calibration step (Platt scaling / isotonic regression) if the base model produces well-ranked but poorly-calibrated scores.
- **Task mismatches often first appear as "the offline metric looks great but the product doesn't feel right."** E.g., low RMSE on regression targets but a ranking UI still feels wrong at the top-of-list — this is the Section 4 phenomenon showing up in production, and is a sign the task type itself, not just the model, needs revisiting.

## 6. Common Interview Traps

- **Defaulting to classification for everything.** Some candidates reach for classification even when the decision clearly requires an order (ranking) or a scale (regression), because classification is the most-drilled task in coursework.
- **Ignoring candidate-set scale.** Proposing a ranking model directly over a multi-million-item catalog without a retrieval stage is a red flag — this is one of the most common gaps between an L4 and L5 recommendation-system answer.
- **Assuming regression and ranking are interchangeable.** As Section 4 shows, low pointwise error does not imply correct order, and vice versa; conflating them under "I'll just use MSE for the ranking model" signals a gap in understanding what the downstream decision needs.
- **Forgetting the discretization tradeoff.** Turning a continuous target into buckets (e.g., risk tiers) throws away information at the tier boundaries; not acknowledging this tradeoff when proposing it is a missed opportunity to show judgement.

## 7. L5-Differentiating Talking Points

- Explicitly name the *candidate-set scale* as a first-class factor in task choice (ranking vs. retrieval+ranking), not just the abstract "which loss function" question.
- Distinguish calibration needs from ordering needs early, and state which one the downstream decision actually requires — this single distinction is often the fastest way to demonstrate you understand Section 4's core point.
- Propose layered formulations when appropriate (e.g., classification/regression for known patterns + anomaly detection as a safety net for novel ones) rather than insisting on one clean task type — real systems are rarely single-task.
- When labels don't yet support the "ideal" target (e.g., dollar-loss regression but only binary labels exist historically), say so explicitly and propose a path (labeling investment, or a pragmatic classification-first MVP with a stated migration plan) rather than silently downgrading the formulation without comment.

## 8. Comprehension Checks

1. A ride-sharing app wants to formulate "estimated time of arrival" (ETA) prediction. Is this classification, regression, or ranking? Justify using the decision the ETA feeds into.
2. Using the pairwise loss formula in Section 4, compute $\mathcal{L}_{\text{pairwise}}$ for the pair $(A, B)$ under Model 1's predictions ($\hat y_A = 10, \hat y_B = 9$) and Model 2's predictions ($\hat y_A = 2.0, \hat y_B = 2.1$), given true relevances $y_A = 3 > y_B = 2$. Which model incurs loss on this pair, and why does that match intuition?
3. A search engine wants to go from "10 billion web pages" to "10 blue links shown to the user." Describe the two-stage architecture this implies and why a single end-to-end ranking model over all 10 billion pages is not a viable formulation.
4. Give an example of a target that's naturally continuous but where you'd deliberately choose to formulate it as classification anyway, and explain the production or business reason that justifies the information loss.

---

*End of Chapter 2. Chapter 3 will cover defining the prediction target itself — granularity (per-user/per-session/per-item-pair), time windows, and how target definition choices ripple into feature engineering and label construction.*
