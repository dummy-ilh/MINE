Here’s a **high-signal cheat table + flow** that maps **common FAANG A/B questions → what YOU should define** (metric, guardrails, unit, sample size, duration, pitfalls).

Think of this as your **“instant answer builder” under pressure**.

---

# 🧠 🔥 MASTER FLOW (Use for ANY question)

```id="abflow1"
1. What is the PRODUCT GOAL?
2. What is the PRIMARY METRIC?
3. What could go WRONG? → Guardrails
4. What is the RANDOMIZATION UNIT?
5. Do we have enough SAMPLE SIZE?
6. How long should we RUN?
7. Any PITFALLS? (network, novelty, bias)
```

---

# 📊 Common FAANG Questions → Structured Answers

---

## 1. 🎯 “We changed button color. CTR increased. Ship it?”

| Component          | What to Say                          |
| ------------------ | ------------------------------------ |
| Goal               | Increase engagement                  |
| Primary Metric     | CTR                                  |
| Guardrails         | Bounce rate, session time, retention |
| Randomization Unit | User-level                           |
| Sample Size        | Based on baseline CTR + MDE          |
| Duration           | ≥ 1 week                             |
| Pitfalls           | Novelty effect, clickbait behavior   |

---

## 2. 💰 “New pricing increases revenue. Good?”

| Component          | What to Say                     |
| ------------------ | ------------------------------- |
| Goal               | Monetization                    |
| Primary Metric     | Revenue per user                |
| Guardrails         | Conversion rate, churn, refunds |
| Randomization Unit | User-level or region-level      |
| Sample Size        | High variance → larger sample   |
| Duration           | Longer (behavioral change)      |
| Pitfalls           | Long-term churn, selection bias |

---

## 3. 🧠 “New recommendation system”

| Component          | What to Say                        |
| ------------------ | ---------------------------------- |
| Goal               | Engagement                         |
| Primary Metric     | Watch time / CTR                   |
| Guardrails         | Diversity, retention, satisfaction |
| Randomization Unit | User-level                         |
| Sample Size        | Medium–large                       |
| Duration           | ≥ 2 weeks                          |
| Pitfalls           | Network effects, Simpson’s paradox |

---

## 4. 🛒 “Checkout flow redesign”

| Component          | What to Say                   |
| ------------------ | ----------------------------- |
| Goal               | Conversion                    |
| Primary Metric     | Purchase rate                 |
| Guardrails         | Drop-off rate, latency        |
| Randomization Unit | User-level (critical)         |
| Sample Size        | Funnel variance matters       |
| Duration           | ≥ 1–2 weeks                   |
| Pitfalls           | Contamination, selection bias |

---

## 5. 📱 “Notification strategy change”

| Component          | What to Say              |
| ------------------ | ------------------------ |
| Goal               | Engagement               |
| Primary Metric     | Open rate / DAU          |
| Guardrails         | Uninstall rate, churn    |
| Randomization Unit | User-level               |
| Sample Size        | Large (noisy metric)     |
| Duration           | Longer (habit formation) |
| Pitfalls           | Novelty effect, fatigue  |

---

## 6. 👥 “Social feature (likes/comments)”

| Component          | What to Say                |
| ------------------ | -------------------------- |
| Goal               | Engagement                 |
| Primary Metric     | Interactions per user      |
| Guardrails         | Spam, user satisfaction    |
| Randomization Unit | **Cluster-level**          |
| Sample Size        | Complex (network variance) |
| Duration           | Longer                     |
| Pitfalls           | 🚨 Network effects         |

---

## 7. 🔍 “Search ranking change”

| Component          | What to Say          |
| ------------------ | -------------------- |
| Goal               | Relevance            |
| Primary Metric     | CTR / success rate   |
| Guardrails         | Latency, abandonment |
| Randomization Unit | Query or user-level  |
| Sample Size        | Large (high traffic) |
| Duration           | Short–medium         |
| Pitfalls           | Query heterogeneity  |

---

# ⚡ Patterns You MUST Recognize

---

## 🧠 1. Metric ↔ Guardrail Pairing

| Primary Metric | Common Guardrails      |
| -------------- | ---------------------- |
| CTR            | Bounce rate, retention |
| Revenue        | Conversion, churn      |
| Engagement     | Satisfaction, crashes  |
| Time spent     | Quality, retention     |

---

## 🧠 2. Randomization Rules

| Situation                 | Unit          |
| ------------------------- | ------------- |
| Same user multiple visits | User          |
| Independent events        | Session/query |
| Social interaction        | Cluster       |

---

## 🧠 3. Sample Size Heuristic

| Scenario             | Sample Need    |
| -------------------- | -------------- |
| Small effect         | HUGE sample    |
| High variance metric | Large sample   |
| Stable metric        | Smaller sample |

---

## 🧠 4. Duration Heuristic

| Case            | Duration         |
| --------------- | ---------------- |
| UI change       | 1–2 weeks        |
| Behavior change | 2–4 weeks        |
| Network effect  | Longer           |
| High traffic    | Shorter possible |

---

# 🚨 Common Interview Mistakes (Avoid These)

---

## ❌ Mistake 1:

> Only talking about p-value

---

## ❌ Mistake 2:

> No guardrails

---

## ❌ Mistake 3:

> Ignoring randomization unit

---

## ❌ Mistake 4:

> “We stop when significant”

---

## ❌ Mistake 5:

> No discussion of pitfalls

---

# 🎯 FAANG-Level Answer Template

Use this almost verbatim:

---

> “First, I’d clarify the product goal and define a primary metric aligned with it.
> Then I’d add guardrail metrics to ensure we’re not harming other aspects of the user experience.
> I’d choose an appropriate randomization unit—typically user-level unless there are network effects.
> Next, I’d estimate sample size based on expected effect size and variance, and determine experiment duration based on traffic.
> Finally, I’d check for potential pitfalls like novelty effects, selection bias, or interference before making a decision.”


