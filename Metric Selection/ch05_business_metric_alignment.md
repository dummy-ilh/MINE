# Chapter 5: Business Metric Alignment

> *"The most dangerous phrase in ML is 'the model is performing well.' Performing well on what? For whom? Toward what end? Until you can answer those questions, you don't know if your model is helping or hurting."*

---

## 5.1 The Translation Problem

There is always a gap between what ML can measure and what the business actually cares about. Bridging this gap is not a technical problem — it is a **communication, prioritization, and alignment problem** that requires working across engineering, product, and business teams.

```
What the business cares about:
  Revenue, retention, customer satisfaction, market share, cost reduction

What ML teams measure:
  AUC, F1, NDCG, log-loss, RMSE

The gap between these is where ML projects succeed or fail.
```

Most ML projects that get cancelled or deprioritized aren't cancelled because the model was bad. They're cancelled because the team couldn't demonstrate that a better model moved a metric the business cared about.

---

## 5.2 The Metric Stack

Think of metrics as a stack with four layers. Every ML system lives somewhere in this stack, and your job is to draw explicit lines between layers.

```
┌─────────────────────────────────────────────────────┐
│  Layer 4: Business Outcomes                         │
│  Revenue, retention, NPS, market share, cost        │
│  → Measured over months; many confounders           │
├─────────────────────────────────────────────────────┤
│  Layer 3: Product Metrics                           │
│  DAU, session length, conversion rate, CSAT         │
│  → Measured over days/weeks; A/B testable           │
├─────────────────────────────────────────────────────┤
│  Layer 2: Behavioral Metrics                        │
│  CTR, dwell time, task completion, error rate       │
│  → Directly influenced by the model                 │
├─────────────────────────────────────────────────────┤
│  Layer 1: Model Metrics                             │
│  AUC, F1, NDCG, calibration error, latency         │
│  → Computed from model outputs on held-out data     │
└─────────────────────────────────────────────────────┘
```

**The alignment question at each layer:** Does improving Layer N reliably improve Layer N+1?

If you can't answer that with evidence or a credible causal argument, your metric stack has a broken link — and you're flying blind.

---

## 5.3 Building the Causal Chain

The most rigorous way to align metrics is to build an explicit **causal chain** from model metric to business outcome.

### Example: Search Engine

```
Model metric:    NDCG@10 improves by 0.02
        ↓
Behavioral:      Users find relevant results faster
                 → Session success rate increases
        ↓
Product metric:  DAU increases (users return more)
                 Query volume increases
        ↓
Business:        Ad revenue increases
                 Subscription retention improves
```

Each arrow is a hypothesis that must be validated — ideally with an A/B test or historical correlation analysis.

### Example: Credit Risk Model

```
Model metric:    AUC improves from 0.78 to 0.83
        ↓
Behavioral:      Better separation of good/bad borrowers
                 → Default rate drops on approved loans
        ↓
Product metric:  Portfolio loss rate decreases
                 Approval rate maintained or improved
        ↓
Business:        Net interest margin improves
                 Regulatory capital requirements met
```

### Example: Content Recommendation

```
Model metric:    Recall@50 improves (better candidate retrieval)
        ↓
Behavioral:      Users encounter more relevant content
                 → Dwell time per session increases
        ↓
Product metric:  D7 retention improves
        ↓
Business:        LTV per user increases
                 Churn rate decreases
```

---

## 5.4 Surrogate Metrics

Because business outcomes are slow to measure and confounded by many factors, we use **surrogate metrics** — faster, cleaner proxies that correlate with business outcomes.

### Properties of a Good Surrogate

| Property | Description |
|---|---|
| **Sensitive** | Moves when the model improves |
| **Specific** | Doesn't move for unrelated reasons |
| **Fast** | Observable within the A/B test window |
| **Causal** | Improving it actually causes business outcome to improve |
| **Stable** | Doesn't oscillate; can be trusted as a trend signal |

### Validating Surrogates

Before trusting a surrogate, validate it historically:

1. Find past A/B tests where you measured both the surrogate and the business outcome
2. Check correlation: do experiments that improved the surrogate also improve the business outcome?
3. Check directionality: have there been cases where the surrogate improved but business outcome declined?

> A surrogate that has failed even once in a known direction is dangerous to use without understanding why.

### Common Surrogate Failures

| Surrogate | Business Metric | What Went Wrong |
|---|---|---|
| CTR | Revenue | Clickbait improved CTR, users didn't convert |
| Session length | Retention | Frustration (infinite scroll, bugs) increased session length |
| Watch time | Satisfaction | Autoplay drove watch time; user satisfaction dropped |
| Model accuracy | Customer churn | Model was accurate on majority class; minority class churned |
| AUC | Default rate | AUC improved but threshold was set wrong; actual decisions didn't improve |

---

## 5.5 North Star Metrics

Many product teams define a single **North Star Metric** — the one number that best captures whether the product is delivering value.

| Company (illustrative) | North Star Metric |
|---|---|
| Spotify | Time spent listening |
| Airbnb | Nights booked |
| Facebook | Daily active users |
| Duolingo | Daily active learners |
| Uber | Trips completed |

Every ML system in the company should be able to draw a line — however indirect — to the North Star. If it can't, the project needs to justify its existence on other grounds (infrastructure, compliance, enabling future work).

### Dangers of North Star Metrics

- **Tunnel vision**: teams optimize North Star at the expense of everything else
- **Gaming**: if everyone is measured on DAU, teams find ways to inflate it artificially
- **Lagging**: North Star often moves slowly; hard to use for day-to-day decisions
- **Conflation**: one number can't capture all dimensions of health (engagement vs. satisfaction vs. revenue)

The North Star is a compass, not a scoreboard. Use it to orient, not to rank teams against each other.

---

## 5.6 Cost-Benefit Analysis for ML Metrics

When a model improves, the improvement has a dollar value. Making this explicit changes conversations.

### Framework

```
Business value of improvement = 
    (Change in metric) × (Volume affected) × (Value per unit) × (Attribution fraction)
```

**Example: Fraud Detection**

- Model AUC improves → false negative rate drops by 0.5%
- Volume: 10M transactions/month
- Average fraud loss: $200
- Attribution: model catches 70% of the improvement (30% from rules)

```
Value = 0.005 × 10,000,000 × $200 × 0.70
      = $7,000,000 / month
```

Now your model improvement has a dollar figure. This is the language of business conversations.

### Don't Forget Costs

Business value must be weighed against:

| Cost Type | Examples |
|---|---|
| Compute cost | Inference at scale, training cost |
| Engineering cost | Building, maintaining, monitoring the system |
| False positive cost | Blocking legitimate transactions, bad user experience |
| Opportunity cost | What else could the team build? |
| Risk cost | Regulatory, reputational, safety exposure |

---

## 5.7 Metric Tensions and Trade-offs

Real products involve multiple metrics that trade off against each other. Pretending otherwise is naive.

### Common Tensions

**Precision vs. Recall**
More aggressive spam filtering catches more spam (recall) but blocks more legitimate email (precision). Where you set the threshold is a business decision, not a model decision.

**Engagement vs. Wellbeing**
Optimizing for session length can increase anxiety, FOMO, and addictive behavior. Short-term engagement metrics can conflict with long-term user health and retention.

**Individual relevance vs. Ecosystem health**
A recommendation system optimizing per-user relevance may crowd out smaller creators, reduce diversity, and harm the long-term health of the content ecosystem.

**Revenue vs. Fairness**
A credit model that maximizes profit may systematically disadvantage certain demographic groups. Fairness constraints reduce profit but are ethically and legally required.

### How to Handle Tensions Explicitly

Use the **satisficing/optimizing** framework from Chapter 1:

```
Optimize:    Revenue per session
Subject to:  User satisfaction score ≥ 4.2/5.0
             Diversity index ≥ 0.6
             Demographic parity ratio ≥ 0.8
             p99 latency ≤ 150ms
```

Make the constraints explicit, quantified, and agreed upon with stakeholders before launch. Implicit constraints become arguments later.

---

## 5.8 Involving Stakeholders in Metric Design

This is where most ML teams underinvest. Metric design is not a solo engineering activity.

### The Metric Design Workshop

Run this before starting any significant ML project:

**Step 1: Outcome mapping**
Ask stakeholders: "If this system worked perfectly, what would be different in 6 months?" Write down the answers literally.

**Step 2: Metric brainstorm**
For each outcome, brainstorm measurable proxies. Don't filter yet.

**Step 3: Feasibility filter**
Which metrics can be measured? How quickly? At what cost?

**Step 4: Prioritization**
Use the satisficing/optimizing framework to agree on one optimizing metric and a set of constraints.

**Step 5: Failure modes**
Ask: "How could a model score well on these metrics but still fail the business?" Document the answers as monitoring requirements.

**Step 6: Sign-off**
Get explicit sign-off from product, business, and engineering leads. Put it in writing. Revisit quarterly.

### Stakeholder Questions to Ask

- What decision will this model's output feed into?
- Who is harmed if the model is wrong in each direction?
- What's the acceptable false positive rate? False negative rate?
- Is there a cost asymmetry between error types?
- What would you want to know if the model degrades after launch?
- Are there subpopulations that must not be disadvantaged?

---

## 5.9 Metrics That Destroy Value

Some metrics, when optimized directly, predictably destroy the thing they were meant to measure. 

### Click-Through Rate (CTR)

Pure CTR optimization produces clickbait. The click happens; the value doesn't. This is why platforms moved to post-click metrics: dwell time, saves, shares, return visits.

### Time on Site

Optimizing time on site can trap users in frustrating experiences (infinite scroll, broken flows, impossible-to-cancel subscriptions). Time on site went up; NPS went down.

### Number of Features Shipped

Incentivizes shipping features nobody uses. Better metric: feature adoption rate, or features that reach X% of users within 30 days.

### Model Accuracy on the Training Distribution

Maximizing accuracy on a fixed benchmark encourages overfitting to the benchmark. The model performs well on the test set; it performs poorly on real users. Benchmark saturation is a real phenomenon.

---

## 5.10 Reporting Metrics to Leadership

Leadership doesn't want to hear about AUC. Here's how to translate:

### The One-Page Summary Structure

```
1. What we built (one sentence)
2. What we measured (metric + baseline + improvement)
3. What it means in business terms ($ or %, customer impact)
4. What we're not sure about (honest uncertainty)
5. What we're monitoring going forward (guardrails)
```

### Framing Examples

| Don't say | Say instead |
|---|---|
| "AUC improved from 0.81 to 0.86" | "The model is 12% better at separating high-risk from low-risk customers" |
| "Recall improved by 8%" | "We now catch 8% more fraudulent transactions before they process" |
| "NDCG@10 is 0.74" | "Users find what they're looking for in the first 3 results 74% of the time" |
| "Log-loss decreased by 0.04" | "Model confidence is now better aligned with actual outcomes" |

---

## Summary

| Concept | One-line takeaway |
|---|---|
| Metric stack | Business outcomes → product metrics → behavioral metrics → model metrics |
| Causal chain | Explicitly map how model improvement drives business improvement |
| Surrogate metrics | Fast proxies for slow business outcomes; must be validated |
| North Star | Single orientation metric; compass, not scoreboard |
| Cost-benefit | Quantify model improvements in dollar terms |
| Metric tensions | Make trade-offs explicit with satisficing constraints |
| Stakeholder alignment | Metric design is a cross-functional activity, not a solo one |
| Bad metrics | Some metrics destroy what they were meant to measure |

---

## Further Reading

- Hubbard, D. — *How to Measure Anything* (quantifying business value of information)
- Kohavi et al. — *Online Controlled Experiments at Large Scale* (surrogate metric validation)
- Croll & Yoskovitz — *Lean Analytics* (North Star metrics in practice)
- Sculley et al. — *Hidden Technical Debt in ML Systems* (metric-system coupling)

---

*Next: Chapter 6 — Goodhart's Law & Metric Gaming*
