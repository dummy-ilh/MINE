# Chapter 1: Metric Selection Fundamentals

> *"Choosing the wrong metric is worse than having no metric at all — you'll optimize hard in the wrong direction."*
> — Andrew Ng (paraphrased)

---

## 1.1 Why Metric Selection Is a Design Decision

When you build an ML system, you are making two decisions simultaneously:

1. What to **optimize** (the loss function during training)
2. What to **measure** (the evaluation metric after training)

These are almost never the same thing — and the gap between them is where most real-world ML systems fail silently.

> **Key insight:** Your model will do exactly what you measure. Not what you intend. Not what you hope. What you measure.

**Why the loss function and the evaluation metric are different animals.** The loss function has to be differentiable (or at least optimizable) so gradient descent can walk downhill. Cross-entropy, MSE, hinge loss — these exist because they have nice gradients, not because a human stakeholder cares about them. The evaluation metric, by contrast, is chosen to reflect what a human *actually* cares about — F1, NDCG, BLEU, revenue lift. A model can drive cross-entropy down for years and still be terrible on the metric a product manager reads in a dashboard, because "lower loss" and "better product" are correlated, not identical.

A simple mental model: think of the loss function as the **compass** telling the optimizer which direction is "downhill locally," and the evaluation metric as the **map** telling you whether you've actually arrived somewhere useful. You can walk downhill forever and still end up in the wrong town if the compass and the map don't agree.

---

## 1.2 The Task-to-Metric Map

Different task types have natural metric families. Start here before customizing. Below, each family also gets a plain-language "what is this actually measuring" note, since memorizing metric names without the intuition is a common interview trap.

### Binary Classification

| Situation | Recommended Metric | What it's really asking |
|---|---|---|
| Balanced classes, costs equal | Accuracy, F1 | "Overall, how often am I right?" |
| Imbalanced classes | PR-AUC, F1, MCC | "Am I actually catching the rare class, not just the common one?" |
| Ranking/threshold matters | ROC-AUC | "If I compare a random positive and a random negative, does my model rank the positive higher?" |
| Probability output matters | Log-loss, Brier Score | "Are my confidence numbers themselves trustworthy, not just my final yes/no?" |

**Worked numerical example — why accuracy vs. F1 diverge.**
Suppose a test set has 100 examples: 90 negative, 10 positive. A model predicts "negative" for everything except 4 true positives it catches (and 2 false positives along the way):

- TP = 4, FN = 6, FP = 2, TN = 88
- Accuracy = (TP + TN) / Total = (4 + 88) / 100 = **0.92**
- Precision = TP / (TP + FP) = 4 / 6 ≈ **0.67**
- Recall = TP / (TP + FN) = 4 / 10 = **0.40**
- F1 = 2 · (Precision · Recall) / (Precision + Recall) = 2 · (0.67 · 0.40) / (0.67 + 0.40) ≈ **0.50**

Accuracy (0.92) looks great. F1 (0.50) tells the truth: the model is missing 60% of the positives it's supposed to catch. This gap is exactly why accuracy is dangerous on imbalanced data.

### Multi-class Classification

| Situation | Recommended Metric | What it's really asking |
|---|---|---|
| All classes equal importance | Macro F1 | "Treat every class as equally important, even rare ones — average their F1s unweighted." |
| Classes weighted by frequency | Weighted F1 | "Let common classes count more, since they occur more often in practice." |
| Strict correctness required | Top-1 Accuracy | "Did I get the single most likely class exactly right?" |

**Worked numerical example — macro vs. weighted F1.**
Three classes: A (F1 = 0.90, 100 samples), B (F1 = 0.60, 50 samples), C (F1 = 0.20, 10 samples).

- Macro F1 = (0.90 + 0.60 + 0.20) / 3 = **0.567** — every class counts the same, so the weak class C drags the average down hard.
- Weighted F1 = (0.90·100 + 0.60·50 + 0.20·10) / 160 = (90 + 30 + 2) / 160 = **0.763** — because C is rare, its poor performance barely moves the number.

If C is a rare-but-critical class (e.g., a safety failure mode), weighted F1 will *hide* the problem. This is a favorite interview gotcha.

### Regression

| Situation | Recommended Metric | What it's really asking |
|---|---|---|
| Outliers are real signal | RMSE | "Punish big misses much harder than small ones (error is squared)." |
| Outliers should be downweighted | MAE | "Treat every unit of error the same, regardless of size." |
| Relative error matters | MAPE | "How far off am I as a *percentage* of the true value?" |
| Robust to both | Huber Loss | "Act like MAE for small errors, like RMSE-style squaring for large errors, with a tunable cutoff." |

**Worked numerical example — RMSE vs. MAE sensitivity to outliers.**
True values: [10, 12, 11, 9]. Predictions: [11, 13, 10, 30] (last prediction is a bad outlier).

- Errors: [1, 1, 1, 21]
- MAE = (1 + 1 + 1 + 21) / 4 = **6.0**
- RMSE = sqrt((1² + 1² + 1² + 21²) / 4) = sqrt((1+1+1+441)/4) = sqrt(444/4) = sqrt(111) ≈ **10.5**

MAE (6.0) treats the outlier as "just another error of size 21, averaged in." RMSE (10.5) nearly doubles the reported error because squaring a large error inflates it disproportionately. If your business genuinely loses money quadratically with bigger mistakes (e.g., large forecasting misses cascade badly), RMSE is honest. If one weird data point shouldn't dominate the whole evaluation, MAE is the safer choice.

### Ranking / Retrieval

| Situation | Recommended Metric | What it's really asking |
|---|---|---|
| Position matters, graded relevance | NDCG | "Give partial credit for relevance, and penalize putting good results lower in the list." |
| Position matters, binary relevance | MAP | "Across all relevant items, how good was my precision at each point I found one?" |
| Only first result matters | MRR, Precision@1 | "How far down the list did I have to go before hitting something useful?" |

**Worked numerical example — MRR.**
Three queries. The rank of the first relevant result is 1, 3, and 2 respectively.

- Reciprocal ranks: 1/1 = 1.0, 1/3 ≈ 0.333, 1/2 = 0.5
- MRR = (1.0 + 0.333 + 0.5) / 3 ≈ **0.611**

A single query where the right answer was buried at rank 10 (reciprocal rank 0.1) would drag this down fast — MRR punishes "the answer was technically in there somewhere" almost as much as "the answer wasn't found at all," which is why it's best reserved for tasks where users truly only look at the first result (e.g., "I'm feeling lucky" search).

### Generation

| Situation | Recommended Metric | What it's really asking |
|---|---|---|
| Translation / summarization | BLEU, ROUGE | "How much word/n-gram overlap is there with a reference text?" |
| Semantic fidelity | BERTScore | "Do these two texts mean the same thing, even with different wording?" |
| Open-ended generation | Human eval, LLM-as-judge | "Does a competent judge consider this good, helpful, or correct?" |

---

## 1.3 Proxy Metrics vs. Target Metrics

This is the single most important distinction in applied ML.

```
Target metric:   What actually matters to the business
                 (e.g., user retention, revenue, patient survival)

Proxy metric:    What we can compute from model outputs
                 (e.g., click-through rate, AUC, accuracy)
```

**Why can't we just optimize the target directly?**

- Target metrics are often **delayed** (retention takes months to measure)
- They're **sparse** (rare events like churn)
- They're **confounded** (many factors beyond the model)
- They're **not differentiable** (can't backprop through "user retained")

So we use proxies — but we must constantly ask: *does improving this proxy actually move the target?*

**How to actually check proxy-target alignment (this is the part interviewers probe on).** You don't just assume the proxy works — you validate it, typically by:

1. **Historical correlation studies** — look at past A/B tests or cohorts and check whether movements in the proxy actually correlated with movements in the target over time.
2. **Long-running holdout experiments** — keep a small population on an old model/proxy indefinitely and watch whether the "proxy-optimized" group's target metric diverges over months.
3. **Periodic re-validation** — proxy-target relationships decay as user behavior, product surface, and competitive landscape shift; a proxy validated two years ago isn't guaranteed valid today.

### Example: YouTube Recommendation

| Layer | Metric |
|---|---|
| Model trains on | Watch probability (cross-entropy) |
| Team measures | Watch time, satisfaction survey |
| Business cares about | Long-term user engagement, advertiser revenue |

Each layer has a proxy relationship to the one below it. If watch time goes up but satisfaction goes down, you have a misaligned proxy — this is the textbook "clickbait/engagement trap": a model can learn that outrage or cliffhanger thumbnails increase watch time in the short run while quietly eroding trust and long-term retention.

**Worked numerical example — proxy divergence.**
Suppose over 6 months, watch time per session rose from 20 to 24 minutes (+20%), while a monthly satisfaction survey score fell from 4.2/5 to 3.6/5 (a 14% drop). If historically these two moved together (correlation ≈ +0.8), this sudden divergence is itself a signal the recommendation model has started exploiting the proxy rather than serving the target — exactly the kind of numeric pattern that should trigger a metric audit.

---

## 1.4 Single-Number Metrics vs. Multiple Metrics

Andrew Ng's advice: **collapse your evaluation to a single number** — but do it deliberately.

**Why one number, when reality is multi-dimensional?** Because decision-making stalls when you compare two models on 5 metrics and Model A wins on 3 while Model B wins on 2. Someone has to break the tie — and if that tie-breaking logic isn't decided *in advance*, it gets decided informally, inconsistently, and politically after the fact. A single number (or a clearly defined optimizing/satisficing split) forces that judgment call to happen up front, transparently.

### The Satisficing / Optimizing Framework

Divide your metrics into two buckets:

- **Optimizing metric**: The one thing you maximize (e.g., F1 score)
- **Satisficing metrics**: Constraints that must be met (e.g., latency < 100ms, accuracy on protected groups > 85%)

> Example: *"Maximize NDCG@10, subject to: P(demographic A) ≥ 0.9 × P(demographic B), and p99 latency ≤ 200ms."*

This is far more actionable than comparing models on 5 metrics simultaneously and arguing about which one wins.

**Worked numerical example.** Three candidate models:

| Model | NDCG@10 | p99 latency | Fairness ratio (A vs B) |
|---|---|---|---|
| M1 | 0.82 | 180ms | 0.95 |
| M2 | 0.85 | 250ms | 0.92 |
| M3 | 0.79 | 150ms | 0.88 |

With the constraint "latency ≤ 200ms, fairness ratio ≥ 0.9": M2 is eliminated (latency 250ms fails the satisficing bound), M3 is eliminated (fairness ratio 0.88 fails), leaving **M1** as the only model that clears both constraints — even though M2 had the best raw NDCG. Without the satisficing framework, a naive "pick the highest NDCG" rule would have shipped a model that violates a hard latency SLA.

### When multiple metrics are unavoidable

Use a **weighted composite**:

```
Score = α × Precision + β × Recall + γ × Latency_penalty
```

But be explicit about the weights — they encode your value judgments. Hidden weights are a source of silent disagreement on teams.

**Worked numerical example.** With α = 0.5, β = 0.4, γ = 0.1, Precision = 0.80, Recall = 0.70, Latency_penalty = 0.30 (already normalized so higher = worse, subtracted):

```
Score = 0.5(0.80) + 0.4(0.70) - 0.1(0.30) = 0.40 + 0.28 - 0.03 = 0.65
```

If a colleague instead believes recall matters more (β = 0.6, α = 0.3, γ = 0.1), the same model scores:

```
Score = 0.3(0.80) + 0.6(0.70) - 0.1(0.30) = 0.24 + 0.42 - 0.03 = 0.63
```

Nearly identical here, but as precision and recall diverge further apart across models, these weight choices can flip which model "wins" — which is exactly why the weights need to be agreed on *before* looking at results, not tuned afterward to justify a preferred model.

---

## 1.5 Cost-Sensitive Evaluation

Accuracy treats all errors equally. Real problems rarely do.

### The cost matrix

For binary classification:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) — missed! |
| **Actual Negative** | False Positive (FP) — false alarm | True Negative (TN) |

**Cost-weighted accuracy:**

```
Cost = C_FN × FN + C_FP × FP
```

**How this changes your threshold, not just your reporting.** Cost-sensitivity isn't only about how you *report* results — it should change *where you set the decision threshold*. If C_FN is 10× larger than C_FP, you want to lower your classification threshold (be more willing to flag positives) even if that increases false positives, because each missed positive is far more expensive than each false alarm.

### Classic examples

| Domain | Costly Error | Why |
|---|---|---|
| Medical diagnosis | False Negative | Missing cancer is worse than extra biopsy |
| Spam filter | False Positive | Blocking real email is worse than spam |
| Fraud detection | False Negative | Missed fraud costs more than blocking legit txn |
| Autonomous driving | False Negative | Missing a pedestrian is catastrophic |

> **Practical tip:** Before choosing your threshold, draw out the cost matrix with your product/business stakeholders. Engineers rarely know the true cost asymmetry. Stakeholders rarely know it's a design choice. Bring them together.

**Worked numerical example.** A fraud model on 1,000 transactions: FN = 15 (missed fraud), FP = 40 (blocked legitimate transactions). Say each missed fraud costs $500 on average, and each wrongly-blocked transaction costs $20 in customer friction/support:

```
Cost = ($500 × 15) + ($20 × 40) = $7,500 + $800 = $8,300
```

Now suppose you lower the threshold, catching more fraud: FN drops to 5, but FP rises to 120:

```
Cost = ($500 × 5) + ($20 × 120) = $2,500 + $2,400 = $4,900
```

Even though total *errors* went from 55 to 125, total *cost* dropped from $8,300 to $4,900 — because the errors shifted from the expensive kind to the cheap kind. This is the concrete argument for why "fewer total errors" and "lower cost" are not the same optimization target.

---

## 1.6 When Metrics Mislead

### The Accuracy Paradox

Dataset: 99% of transactions are non-fraudulent.

A model that predicts "not fraud" for everything achieves **99% accuracy** — but is completely useless. Precision and recall on the fraud class would be 0/0 (undefined, since the model makes zero positive predictions) — in practice reported as 0.

**Numerically:** 990 TN, 0 TP, 0 FP, 10 FN. Accuracy = 990/1000 = **0.99**. Recall = 0/(0+10) = **0**. A single number (accuracy) says "great," while the metric that actually matters (recall on fraud) says "total failure."

### Simpson's Paradox

Aggregated metrics can reverse when you disaggregate. A model might look better overall but worse on every individual subgroup due to distribution shifts between groups.

**Worked numerical example.** Model A vs. Model B, split by user segment:

| Segment | Model A | Model B |
|---|---|---|
| New users (n=1000) | 90% accuracy | 95% accuracy |
| Existing users (n=100) | 70% accuracy | 75% accuracy |
| **Weighted overall** | (0.9·1000+0.7·100)/1100 = **88.2%** | (0.95·1000+0.75·100)/1100 = **93.2%** |

Here B actually wins in aggregate *and* on every subgroup, so this particular table doesn't paradox — but flip the segment *sizes* (e.g., if Model A were tested mostly on the segment it's weak on, and Model B mostly on the segment it's strong on) and the weighted averages can cross over and reverse which model "wins" overall, even though B is better in every individual row. The lesson: always check that subgroup sample sizes are comparable across models before trusting an aggregate comparison.

Always **slice your metrics** by:
- Input distribution segments
- Time windows (metrics decay)
- User cohorts
- Geographic regions

### Metric-Task Mismatch

| Mistake | What Happened |
|---|---|
| Using accuracy for imbalanced data | Ignored the minority class |
| Using BLEU for dialogue | Penalized valid paraphrases |
| Using RMSE when MAE was needed | Outliers dominated the signal |
| Using offline AUC for a ranking product | Didn't account for position bias |

---

## 1.7 Establishing Your Baseline

A metric only means something relative to a baseline. Always report:

1. **Random baseline** — predicting the prior distribution
2. **Human-level performance** — where applicable
3. **Previous model / rule-based system** — the production baseline
4. **State-of-the-art** — from literature (for calibration)

```
Improvement = (Your metric - Baseline metric) / (Ceiling - Baseline metric)
```

This "headroom" framing tells you how much of the possible improvement you've captured.

**Worked numerical example.** Suppose:
- Baseline (current production rule-based system): F1 = 0.60
- Your new model: F1 = 0.72
- Ceiling (human-level performance): F1 = 0.90

```
Improvement = (0.72 - 0.60) / (0.90 - 0.60) = 0.12 / 0.30 = 0.40 → 40% of available headroom captured
```

Reporting "F1 improved by 0.12" sounds modest. Reporting "we captured 40% of the remaining gap to human performance" reframes the same result in a way that's far more meaningful to stakeholders, and also flags that 60% of the achievable improvement is still on the table.

---

## 1.8 Checklist: Before Finalizing Your Metric

Before locking in a metric for a project, go through this:

- [ ] Does improving this metric improve the actual user/business outcome?
- [ ] Is the class distribution accounted for?
- [ ] Are error costs symmetric, or do you need a cost matrix?
- [ ] Have you defined satisficing constraints separately from the optimizing metric?
- [ ] Is the metric computable at the frequency you need (real-time? daily? monthly)?
- [ ] Will this metric still be valid after the model goes to production (distribution shift)?
- [ ] Have you stress-tested it on edge case slices?

---

## Q&A

**Q1: Why can accuracy be a dangerous default metric?**
A: Because it treats every class and every error type as equally important. On imbalanced data, a model can score very high accuracy while completely failing on the minority class — the "accuracy paradox" in §1.6 shows a 99%-accurate model with 0 recall on the class that actually mattered.

**Q2: What's the practical difference between a proxy metric and a target metric?**
A: The target metric is the real-world outcome you care about (revenue, retention, patient survival) — but it's usually slow, sparse, confounded, or non-differentiable, so you can't optimize it directly. The proxy metric is a fast, dense, differentiable stand-in (CTR, accuracy, AUC) that you optimize instead, on the assumption — which must be periodically re-validated — that moving the proxy moves the target.

**Q3: Why does Andrew Ng recommend a single-number evaluation metric, and doesn't that oversimplify things?**
A: It doesn't oversimplify if you're deliberate about it. The point isn't to pretend only one thing matters — it's to force the "how do we trade off these different concerns" decision to happen explicitly and in advance (via the optimizing/satisficing split or explicit composite weights), rather than being decided informally and inconsistently after seeing results.

**Q4: When should you use RMSE instead of MAE, and vice versa?**
A: Use RMSE when large errors are genuinely more costly than the sum of many small ones (e.g., squared penalties reflect real-world risk) and outliers are meaningful signal you want to be sensitive to. Use MAE when you want every unit of error weighted equally and don't want a few outliers to dominate the reported number. Huber loss is the middle ground: MAE-like for small errors, RMSE-like for large ones, with a tunable cutoff.

**Q5: What's an example of a cost matrix changing a real decision, not just a report?**
A: In fraud detection, if missing fraud (FN) costs far more than a false alarm (FP), the right move is to lower the classification threshold — accepting more false positives to catch more true fraud — because the total dollar cost drops even though the total error *count* rises (see the worked example in §1.5, where cost fell from $8,300 to $4,900 despite total errors more than doubling).

**Q6: How would you explain Simpson's Paradox to a non-technical stakeholder?**
A: A model can appear better on every individual subgroup of your data, yet look worse (or the ranking can flip) once you combine those subgroups into one aggregate number — if the *mix* of subgroups differs between what's being compared. The fix is to always check that you're comparing like-for-like segment sizes, and to look at sliced metrics, not just the headline aggregate.

**Q7: Why is "headroom" a more useful framing than raw metric improvement?**
A: A raw jump like "F1 went from 0.60 to 0.72" doesn't tell you whether that's a huge win or a drop in the bucket. Dividing by the gap to a ceiling (human-level or theoretical best) — e.g., "we closed 40% of the remaining gap to human performance" — tells stakeholders both how much was achieved and how much opportunity remains.

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Task-metric map | Start with the right family, then customize |
| Proxy vs. target | Know what you're measuring vs. what you care about |
| Single-number | Optimizing metric + satisficing constraints |
| Cost asymmetry | Accuracy lies when mistakes have unequal costs |
| Baselines | A metric without a baseline is meaningless |
| Slicing | Always disaggregate; aggregates can hide failure |

---

## Further Reading

- Ng, A. — *Machine Learning Yearning*, Chapters 7–11 (metric selection)
- Sculley et al. — *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015)
- Recht et al. — *Do ImageNet Classifiers Generalize to ImageNet?* (on metric stability)

---

*Next: Chapter 2 — Offline vs. Online Evaluation*
