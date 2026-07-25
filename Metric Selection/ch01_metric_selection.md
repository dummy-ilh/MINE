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

---

## 1.2 The Task-to-Metric Map

Different task types have natural metric families. Start here before customizing.

### Binary Classification

| Situation | Recommended Metric |
|---|---|
| Balanced classes, costs equal | Accuracy, F1 |
| Imbalanced classes | PR-AUC, F1, MCC |
| Ranking/threshold matters | ROC-AUC |
| Probability output matters | Log-loss, Brier Score |

### Multi-class Classification

| Situation | Recommended Metric |
|---|---|
| All classes equal importance | Macro F1 |
| Classes weighted by frequency | Weighted F1 |
| Strict correctness required | Top-1 Accuracy |

### Regression

| Situation | Recommended Metric |
|---|---|
| Outliers are real signal | RMSE |
| Outliers should be downweighted | MAE |
| Relative error matters | MAPE |
| Robust to both | Huber Loss |

### Ranking / Retrieval

| Situation | Recommended Metric |
|---|---|
| Position matters, graded relevance | NDCG |
| Position matters, binary relevance | MAP |
| Only first result matters | MRR, Precision@1 |

### Generation

| Situation | Recommended Metric |
|---|---|
| Translation / summarization | BLEU, ROUGE |
| Semantic fidelity | BERTScore |
| Open-ended generation | Human eval, LLM-as-judge |

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

### Example: YouTube Recommendation

| Layer | Metric |
|---|---|
| Model trains on | Watch probability (cross-entropy) |
| Team measures | Watch time, satisfaction survey |
| Business cares about | Long-term user engagement, advertiser revenue |

Each layer has a proxy relationship to the one below it. If watch time goes up but satisfaction goes down, you have a misaligned proxy.

---

## 1.4 Single-Number Metrics vs. Multiple Metrics

Andrew Ng's advice: **collapse your evaluation to a single number** — but do it deliberately.

### The Satisficing / Optimizing Framework

Divide your metrics into two buckets:

- **Optimizing metric**: The one thing you maximize (e.g., F1 score)
- **Satisficing metrics**: Constraints that must be met (e.g., latency < 100ms, accuracy on protected groups > 85%)

> Example: *"Maximize NDCG@10, subject to: P(demographic A) ≥ 0.9 × P(demographic B), and p99 latency ≤ 200ms."*

This is far more actionable than comparing models on 5 metrics simultaneously and arguing about which one wins.

### When multiple metrics are unavoidable

Use a **weighted composite**:

```
Score = α × Precision + β × Recall + γ × Latency_penalty
```

But be explicit about the weights — they encode your value judgments. Hidden weights are a source of silent disagreement on teams.

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

### Classic examples

| Domain | Costly Error | Why |
|---|---|---|
| Medical diagnosis | False Negative | Missing cancer is worse than extra biopsy |
| Spam filter | False Positive | Blocking real email is worse than spam |
| Fraud detection | False Negative | Missed fraud costs more than blocking legit txn |
| Autonomous driving | False Negative | Missing a pedestrian is catastrophic |

> **Practical tip:** Before choosing your threshold, draw out the cost matrix with your product/business stakeholders. Engineers rarely know the true cost asymmetry. Stakeholders rarely know it's a design choice. Bring them together.

---

## 1.6 When Metrics Mislead

### The Accuracy Paradox

Dataset: 99% of transactions are non-fraudulent.

A model that predicts "not fraud" for everything achieves **99% accuracy** — but is completely useless. Precision and recall on the fraud class would be 0.

### Simpson's Paradox

Aggregated metrics can reverse when you disaggregate. A model might look better overall but worse on every individual subgroup due to distribution shifts between groups.

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
