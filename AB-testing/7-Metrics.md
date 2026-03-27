Below is a **master, interview-ready sheet** that ties together:

> **Metric → Unit of analysis → Trade-offs → Pitfalls**

Use it to **answer any product / A/B / system design question** with structure and depth.

---

# 🧠 1. Acquisition & Growth Metrics

| Metric                     | Unit              | Trade-offs                   | Pitfalls                           |
| -------------------------- | ----------------- | ---------------------------- | ---------------------------------- |
| CTR (Click Through Rate)   | Impression / User | ↑CTR vs ↓quality (clickbait) | Vanity metric, doesn’t imply value |
| Signup Conversion Rate     | User              | Growth vs user quality       | Fake/low-intent users              |
| Cost per Acquisition (CPA) | User              | Cost vs scale                | Cheap users ≠ valuable users       |
| Install Rate               | Device/User       | Reach vs retention           | High installs, low usage           |

---

# 🧠 2. Engagement Metrics

| Metric            | Unit           | Trade-offs              | Pitfalls                     |
| ----------------- | -------------- | ----------------------- | ---------------------------- |
| DAU / MAU         | User           | Frequency vs depth      | Doesn’t capture satisfaction |
| Session Time      | Session/User   | Time vs efficiency      | Addictive design             |
| Pages per Session | Session        | Exploration vs friction | Artificial inflation         |
| Likes / Shares    | User / Content | Virality vs quality     | Spam / bot activity          |
| Active Days       | User           | Habit vs burnout        | Doesn’t measure value        |

---

# 🧠 3. Retention Metrics

| Metric               | Unit        | Trade-offs             | Pitfalls                       |
| -------------------- | ----------- | ---------------------- | ------------------------------ |
| Day N Retention      | User cohort | Short vs long-term     | Misleading if not cohort-based |
| Rolling Retention    | User        | Flexibility vs clarity | Overestimates retention        |
| Churn Rate           | User        | Simplicity vs nuance   | Doesn’t show why users leave   |
| Stickiness (DAU/MAU) | User        | Frequency vs depth     | Can be gamed                   |

---

# 🧠 4. Monetization Metrics

| Metric                  | Unit        | Trade-offs               | Pitfalls             |
| ----------------------- | ----------- | ------------------------ | -------------------- |
| Revenue per User (ARPU) | User        | Revenue vs retention     | Over-monetization    |
| Lifetime Value (LTV)    | User        | Long-term vs short-term  | Hard to estimate     |
| Conversion to Paid      | User        | Revenue vs user trust    | Aggressive upselling |
| Average Order Value     | Transaction | Basket size vs frequency | Inflated purchases   |

---

# 🧠 5. Quality / Satisfaction Metrics

| Metric                   | Unit         | Trade-offs                  | Pitfalls               |
| ------------------------ | ------------ | --------------------------- | ---------------------- |
| NPS (Net Promoter Score) | User         | Simplicity vs accuracy      | Biased responses       |
| CSAT                     | User/session | Immediate vs long-term      | Survey fatigue         |
| Complaint Rate           | User         | Detection vs reporting bias | Silent dissatisfaction |
| Content Quality Score    | Content/User | Quality vs engagement       | Hard to define         |

---

# 🧠 6. System / ML Metrics (MLE Focus)

| Metric             | Unit       | Trade-offs                         | Pitfalls             |
| ------------------ | ---------- | ---------------------------------- | -------------------- |
| Accuracy           | Prediction | Accuracy vs latency                | Ignores imbalance    |
| Precision / Recall | Prediction | False positives vs false negatives | Depends on threshold |
| AUC                | Model      | Ranking vs calibration             | Hard to interpret    |
| Latency            | Request    | Speed vs accuracy                  | Slow → bad UX        |
| Throughput         | System     | Scale vs cost                      | Over-optimization    |
| Cache Hit Rate     | Request    | Speed vs freshness                 | Stale results        |

---

# 🧠 7. Marketplace Metrics (Two-sided platforms)

| Metric              | Unit        | Trade-offs            | Pitfalls               |
| ------------------- | ----------- | --------------------- | ---------------------- |
| Supply-Demand Ratio | Region/User | Balance vs efficiency | Local imbalance hidden |
| Fill Rate           | Request     | Matching vs quality   | Low-quality matches    |
| Time to Match       | Request     | Speed vs relevance    | Poor experience        |

---

# 🧠 8. Experimentation Metrics

| Metric              | Unit       | Trade-offs            | Pitfalls                 |
| ------------------- | ---------- | --------------------- | ------------------------ |
| Lift (%)            | User       | Sensitivity vs noise  | Misleading small effects |
| P-value             | Experiment | Simplicity vs misuse  | Misinterpretation        |
| Confidence Interval | Experiment | Clarity vs complexity | Ignored in decisions     |
| Power               | Experiment | Detection vs cost     | Underpowered tests       |

---

# 🧠 9. Guardrail Metrics (Cross-cutting)

| Metric          | Unit         | Trade-offs               | Pitfalls           |
| --------------- | ------------ | ------------------------ | ------------------ |
| Crash Rate      | Session/User | Stability vs speed       | Rare but critical  |
| Latency         | Request      | Speed vs accuracy        | Impacts engagement |
| Bounce Rate     | Session      | Quick exit vs efficiency | Misinterpreted     |
| Abuse/Spam Rate | Content/User | Growth vs safety         | Under-detection    |

---

# 🔥 Cross-Metric Trade-off Patterns (VERY IMPORTANT)

---

## ⚖️ 1. Growth vs Quality

* CTR ↑ → Quality ↓
* Signups ↑ → Retention ↓

---

## ⚖️ 2. Engagement vs Well-being

* Time spent ↑ → Addiction risk
* Notifications ↑ → Fatigue

---

## ⚖️ 3. Monetization vs Retention

* Ads ↑ → Revenue ↑ but churn ↑
* Pricing ↑ → Short-term gain, long-term loss

---

## ⚖️ 4. Accuracy vs Latency (MLE CORE)

* Complex model → better predictions but slower
* Simple model → fast but less accurate

---

## ⚖️ 5. Exploration vs Exploitation

* Show new content → discovery
* Show known content → engagement

---

# 🚨 Universal Pitfalls (Across ALL Metrics)

---

## ❌ 1. Metric Misalignment

* Optimizing wrong thing

---

## ❌ 2. Proxy Metric Failure

* CTR ≠ user satisfaction

---

## ❌ 3. Gaming the Metric

* Clickbait, spam

---

## ❌ 4. Ignoring Segments

* Simpson’s paradox

---

## ❌ 5. Short-term Optimization

* Hurts long-term retention

---

## ❌ 6. Not Defining Unit Properly

* User vs session mismatch

---

# 🧠 Final Mental Model

---

```id="metricmaster1"
Metric = What you measure

Unit = What you measure it on

Trade-off = What you sacrifice

Pitfall = How it can mislead you
```

---

# 🚀 How to Use This in Interviews

When asked ANY product question:

---

### ✅ Say:

> “I’d define a primary metric at the user level, evaluate trade-offs such as X vs Y, and include guardrails to prevent pitfalls like metric gaming or short-term optimization.”

---

---

