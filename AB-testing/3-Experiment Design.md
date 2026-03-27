

# 🧪 4.1 Randomization Unit Selection

---

## 🧠 Core Idea

> **Randomization unit = entity you assign to control vs treatment**

Wrong choice ⇒ **biased results / invalid experiment**

---

# 📊 Common Randomization Units (With Examples)

| Scenario               | Randomization Unit   | Why?                                 | Pitfall if Wrong                           |
| ---------------------- | -------------------- | ------------------------------------ | ------------------------------------------ |
| Website UI change      | **User ID**          | Same user sees consistent experience | If session-based → user sees both variants |
| Search ranking         | **Query**            | Each search independent              | User-level → noisy (user intent varies)    |
| Ads experiment         | **User ID / Cookie** | Tracks exposure & conversion         | Impression-level → double counting         |
| Ride-sharing pricing   | **City / Region**    | Avoid cross-market contamination     | User-level → users compare prices          |
| Social network feature | **Cluster (groups)** | Network effects                      | User-level → spillover bias                |
| Email campaign         | **User ID**          | Clear attribution                    | Session-level → inconsistent exposure      |
| Checkout flow          | **User ID**          | Multi-step journey                   | Page-level → inconsistent UX               |
| Video recommendations  | **User ID**          | Personalized system                  | Session-level → unstable signals           |

---

# ⚠️ Golden Rule (Interview Must-Say)

> “I’d choose the **highest level unit that avoids contamination but preserves statistical power**.”

---

## 🚨 Key Pitfalls

### 1. Contamination (MOST IMPORTANT)

* User exposed to both A and B

Example:

* Session-level randomization → same user sees both designs

---

### 2. Network Effects

* One user affects another

Example:

* Social media → likes/comments influence others

👉 Fix:

* Cluster randomization

---

### 3. Mismatch with Metric

Example:

* Metric = user retention
* Randomization = session ❌

---

# 📏 Sample Size Selection

---

## 🧠 What Determines Sample Size?

| Factor              | Effect        |
| ------------------- | ------------- |
| Effect size (MDE) ↓ | Sample size ↑ |
| Variance ↑          | Sample size ↑ |
| Confidence level ↑  | Sample size ↑ |
| Power ↑             | Sample size ↑ |

---

## 🔥 Key Formula (Conceptual)

Sample size grows as:

n \propto \frac{\sigma^2}{\delta^2}

Where:

* ( \sigma^2 ): variance
* ( \delta ): effect size

---

## 🎯 Interview Insight

> Smaller effect you want to detect → exponentially more data needed

---

## 💡 Typical Defaults (Say This)

* α = 0.05
* Power = 80%
* MDE = 1–5%

---

# ⏱️ Experiment Duration

---

## 🧠 How to Decide Duration?

### Formula (conceptually):

[
\text{Duration} = \frac{\text{Required Sample Size}}{\text{Traffic per day}}
]

---

## 🎯 Rules of Thumb

| Rule                 | Why                                |
| -------------------- | ---------------------------------- |
| Run ≥ 1 week         | Capture weekday/weekend patterns   |
| Avoid early stopping | Prevent false positives            |
| Watch novelty effect | Users behave differently initially |

---

## 🚨 Interview Traps

### ❌ “We stop when p < 0.05”

👉 Wrong → **peeking problem**

---

### ✅ Correct Answer

> “I’d fix duration upfront based on sample size and traffic, and avoid peeking unless using sequential testing.”

---

# 🧪 A/A Test (Highly Underrated Signal)

---

## 🧠 What is A/A Test?

* Both groups get SAME experience
* Expected: **no difference**

---

## 🎯 Why Run A/A Test?

| Purpose                | What it Detects    |
| ---------------------- | ------------------ |
| Validate randomization | Bias               |
| Check metric pipeline  | Logging errors     |
| Estimate variance      | Better sample size |
| Detect false positives | System issues      |

---

## 🚨 Red Flag

If A/A shows significant difference:

👉 Something is broken:

* Tracking issue
* Sampling bias
* Randomization bug

---

# 🔥 Putting It All Together (Interview Answer Template)

---

### If asked:

**“How would you design an A/B test?”**

---

### ✅ Strong Answer:

> “First, I’d choose the appropriate randomization unit—typically user-level—to avoid contamination. If network effects are present, I’d consider cluster-level randomization.
>
> Then I’d estimate sample size based on the minimum detectable effect, variance, significance level, and power.
>
> Based on traffic, I’d determine experiment duration, ensuring it runs long enough to capture periodic patterns and avoid novelty effects.
>
> I’d also run an A/A test beforehand to validate the experiment setup and ensure no bias or logging issues exist.”

---
Yes — and this is where most candidates **fail silently**.
FAANG doesn’t ask “what is randomization unit?” — they embed it inside **messy product questions**.

Let’s go through **real-style questions + what they’re actually testing + how to answer**.

---

# 🧠 1. Classic FAANG Question

### ❓ “We changed the checkout page UI. How would you design an experiment?”

---

### 🎯 What they’re testing:

* Randomization unit
* Metric choice
* Funnel thinking

---

### ✅ Strong Answer:

> “Since checkout is a multi-step flow, I’d randomize at the **user level** to ensure a consistent experience across steps.
> If we randomize at page/session level, users might see mixed versions, leading to contamination.”

---

### 🚨 Weak Answer:

* “I’ll randomly assign traffic” ❌ (too vague)

---

# 🧠 2. Trick Question (VERY COMMON)

### ❓ “Should we randomize at user or session level?”

---

### 🎯 What they want:

* Do you understand **contamination**

---

### ✅ Strong Answer:

| Case                                | Choose                  |
| ----------------------------------- | ----------------------- |
| Same user can revisit               | **User-level**          |
| Independent events (search queries) | **Session/query-level** |

---

### 💡 Gold Line:

> “If the same user can experience both variants, I’d avoid session-level randomization.”

---

# 🧠 3. Network Effect Question (Meta/LinkedIn)

### ❓ “We introduce a new feature showing friends’ likes. How do you A/B test?”

---

### 🎯 What they’re testing:

* Network effects awareness

---

### ❌ Naive Answer:

* “Randomize users” → WRONG

---

### ✅ Strong Answer:

> “Since user behavior is influenced by their friends, this introduces network effects.
> I’d consider **cluster-level randomization** (e.g., friend groups or communities) to avoid interference.”

---

# 🧠 4. Sample Size Question

### ❓ “How would you determine sample size?”

---

### 🎯 What they want:

* Conceptual understanding (NOT exact formula)

---

### ✅ Strong Answer:

> “Sample size depends on the minimum detectable effect, variance, desired significance level, and power.
> If we want to detect smaller effects, we need larger samples.”

---

### 🔥 Bonus (High Signal):

> “We can use historical data to estimate variance and baseline rates.”

---

# 🧠 5. Duration Trap Question

### ❓ “When do you stop the experiment?”

---

### ❌ Wrong:

* “When p < 0.05” 🚫

---

### ✅ Correct:

> “I’d predefine duration based on sample size and traffic.
> Stopping early without correction leads to inflated false positives.”

---

### 🔥 Bonus:

> “Unless we use sequential testing methods.”

---

# 🧠 6. A/A Test Question

### ❓ “Why would you run an A/A test?”

---

### 🎯 What they want:

* Systems thinking

---

### ✅ Strong Answer:

> “To validate the experiment pipeline.
> If A/A shows a significant difference, it indicates issues like bias, logging errors, or incorrect randomization.”

---

# 🧠 7. Real FAANG Scenario

### ❓ “You ran an experiment, but results are inconsistent across users. Why?”

---

### 🎯 What they’re testing:

* Randomization mistakes
* Heterogeneity

---

### ✅ Possible Reasons:

* Wrong randomization unit
* Segment differences
* Network effects
* Small sample size

---

# 🧠 8. High-Level Design Question

### ❓ “Design an experiment for a new recommendation system”

---

### 🎯 Hidden checks:

* Unit = user-level ✅
* Metrics = engagement + guardrails ✅
* Duration + sample size ✅

---

### 🚨 Hidden Trap:

If you say:

* “Randomize per request” ❌

---

# ⚡ Pattern You Should Memorize

---

## 🎯 Every Question = Hidden Version of This:

```id="j82q0h"
1. What is the unit?
2. Can contamination happen?
3. Are there network effects?
4. Is the metric aligned with the unit?
5. Do we have enough data?
```

---

# 🧠 What Separates Strong Candidates

---

### Weak:

* Talks about p-values only

---

### Strong:

* Talks about:

  * contamination
  * network effects
  * consistency of experience
  * metric alignment

---


