
---

# 🧠 A/B Testing Metrics Flowchart

```
                 🎯 What is the PRODUCT GOAL?
                             │
     ┌───────────────────────┼────────────────────────┐
     │                       │                        │
 📈 Growth / Acquisition   💰 Monetization       🧠 Engagement / Retention
     │                       │                        │
     │                       │                        │
 PRIMARY METRIC         PRIMARY METRIC           PRIMARY METRIC
     │                       │                        │
 CTR / Signups           Revenue/User           DAU / Session Time
 Conversion Rate         ARPU / LTV             Retention Rate
     │                       │                        │
     │                       │                        │
 ────────────────     ────────────────      ────────────────
     │                       │                        │
 ⚠️ GUARDRAILS           ⚠️ GUARDRAILS          ⚠️ GUARDRAILS
     │                       │                        │
 - Bounce Rate           - Conversion Rate       - App Crashes
 - Session Time          - User Drop-off         - Latency
 - Retention             - Refund Rate           - User Satisfaction
```

---

# 🔥 How to Use This in Interview (Very Important)

Instead of memorizing, **derive it live**:

---

## Step 1: Identify Goal

Ask:

> “What is the product trying to optimize?”

---

## Step 2: Pick Primary Metric

Rule:

* **One main metric ONLY**

| Goal         | Primary Metric    |
| ------------ | ----------------- |
| Growth       | CTR, Signups      |
| Monetization | Revenue per user  |
| Engagement   | DAU, session time |

---

## Step 3: Add Guardrails (THIS IS WHERE YOU STAND OUT)

👉 Golden rule:

> “What could go wrong if we optimize this?”

---

### Examples:

#### 📈 Growth (CTR ↑)

* Users might click more but hate experience

Guardrails:

* Bounce rate
* Session time
* Retention

---

#### 💰 Monetization (Revenue ↑)

* You might exploit users

Guardrails:

* Conversion rate
* Refund rate
* User churn

---

#### 🧠 Engagement (Time ↑)

* Addictive but harmful UX

Guardrails:

* User satisfaction
* Crash rate
* Long-term retention

---

# ⚡ Super High-Signal Insight

### 💡 Metric Anti-Pattern:

| Bad Primary Metric | Why It's Dangerous      |
| ------------------ | ----------------------- |
| Clicks only        | Clickbait               |
| Time spent only    | Addiction / low quality |
| Revenue only       | User churn              |

👉 Always say:

> “We need guardrails to prevent optimizing the wrong behavior.”

---

# 🧠 Mental Model (Memorize This)

```
Primary Metric = What we WANT

Guardrails = What we DON'T want to break
```

---

# 🎯 FAANG-Level Answer Example

If asked:

**“We changed button color → CTR increased. Ship it?”**

You say:

> “CTR is our primary metric, but I’d check guardrails like bounce rate, session duration, and retention to ensure users aren’t clicking but leaving immediately. If those are stable or improved, then I’d consider shipping.”

---

