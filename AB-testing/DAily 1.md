

# 🧪 **Day 1: A/B Testing Question**

## 📌 Problem

You are working at a product company.

The team changes the **“Sign Up” button color** from blue → green.

After running an A/B test:

* Control (Blue): **10% conversion rate**
* Treatment (Green): **11% conversion rate**
* p-value = **0.04**

---

## ❓ Question

**Do you ship the change?**

---

# ✋ STOP — Your Turn First

Answer like you're in an interview.

Structure your answer:

1. What does the result mean?
2. Is this statistically significant?
3. Is that enough to decide?
4. What else would you check?

---


# 🧠 Step 1: What You Got Right

### ✅ You said:

* It’s statistically significant ✔️
* Think about engagement/other effects ✔️


---

# ❌ Where You Lost Points (Important)

### 1. “New button → more revenue”

🚫 **Wrong assumption**

* We only know **conversion rate increased**
* We do **NOT know revenue increased**

👉 Example:

* Maybe users sign up more…
* But they churn faster / don’t pay

---

# 🧠 Now Let’s Build the PERFECT Answer

## ✅ Step 1: Interpret the Result

* Conversion increased from **10% → 11%**
* That’s a **+1% absolute**, **+10% relative lift**
* p-value = 0.04 → statistically significant at 5% level

✔️ So yes, the effect is **likely real (not random)**

---

## ⚠️ Step 2: Is That Enough?

👉 **NO. Never ship based only on p-value.**

This is the biggest lesson of Day 1.

---

# 🔥 Step 3: What Else Must You Check?

## 1. 💰 Business Impact (CRITICAL)

Ask:

* Does this increase **revenue**?
* Or just **low-quality signups**?

Example risk:

* Users accidentally clicking green button
* Leads to fake / low-intent users

---

## 2. 🛡️ Guardrail Metrics

These protect you from hidden damage.

Check:

* Retention (do users stay?)
* Engagement (DAU, session time)
* Bounce rate
* Downstream conversion (purchase, subscription)

---

## 3. 📊 Practical Significance

Even if statistically significant:

👉 Is **+1% worth it?**

Depends on:

* Scale (millions of users? HUGE impact)
* Engineering cost (low here → good)

---

## 4. ⚠️ Experiment Validity

Check for:

* Sample size sufficient?
* Ran long enough? (weekend vs weekday bias)
* Randomization correct?

---

## 5. 🧠 Behavioral Reasoning

Ask:

👉 *Why did this work?*

* Green = more visible?
* Better UX?
* Or accidental clicks?

---

# 🏆 Final FAANG-Level Answer (What You Should Say)

> The experiment shows a statistically significant increase in conversion from 10% to 11% (p = 0.04), so the effect is likely real.
>
> However, I would not immediately ship it.
>
> First, I would evaluate whether this leads to meaningful business impact, such as improved revenue or high-quality user acquisition.
>
> Second, I would check guardrail metrics like retention, engagement, and downstream conversions to ensure we are not introducing low-quality signups or harming user experience.
>
> Third, I would assess practical significance — whether a 1% absolute lift is meaningful at our scale.
>
> Finally, I would validate experiment design (duration, sample size, randomization) and try to understand the behavioral reason behind the change.
>
> If all checks pass, I would recommend shipping.

---

# 🧠 Core Lesson (MEMORIZE THIS)

👉 **Statistical significance ≠ Business success**

---

# 🔥 Interview Trick (High ROI)

Always say this line:

> “I would also check guardrail metrics to ensure no negative side effects.”

This alone boosts your answer level massively.

---
Perfect! Day 2 is where we **start introducing subtle traps**—the kind that will make or break you in FAANG A/B rounds.

---

# 🧪 **Day 2: A/B Testing Question (Trap Edition)**

## 📌 Problem

You are working on a **streaming platform**.

* You run an experiment on a new **“Skip Intro”** feature.
* Results after 1 week:

| Metric               | Control | Treatment |
| -------------------- | ------- | --------- |
| Watch Time per User  | 120 min | 125 min   |
| Subscription Renewal | 60%     | 58%       |

* p-value for Watch Time = 0.03
* p-value for Renewal = 0.07

---

## ❓ Question

**Do you ship the new “Skip Intro” feature?**

Think carefully—this is a **Simpson’s paradox / multi-metric trap** style question.

---

### ✋ STOP — Your Turn First

Structure your answer like yesterday:

1. Interpret the metrics
2. Consider statistical significance
3. Check for business impact
4. Identify pitfalls / hidden traps

Good instincts again—you’re **spotting the key conflict**: watch time ↑ but subscription renewal ↓. That’s exactly the kind of scenario FAANG loves to test. But your answer is **too short** and misses several critical layers. Let’s break it down.

---

# 🧠 Step 1: What You Got Right

### ✅ You noticed:

* **Conflicting metrics:** watch time ↑, renewal ↓ ✔️
* **Business impact matters:** you’re thinking beyond p-values ✔️

This is the **high-level intuition** they want.

---

# ❌ Where Your Answer Falls Short

1. **You didn’t mention significance clearly:**

* Watch time p = 0.03 → statistically significant
* Renewal p = 0.07 → not significant (but still a red flag)

> Even non-significant metrics can signal negative trends. FAANG candidates **always flag these**.

2. **You didn’t discuss guardrail metrics / user segments:**

* Maybe **the increase in watch time is from heavy users**, while light users churn more → **Simpson’s paradox**
* You need to check **segments**, not just overall averages.

3. **No experiment design sanity check:**

* Was randomization correct?
* Was the experiment long enough?
* Were all relevant metrics measured?

---

# 🧠 Step 2: Perfect FAANG-Level Reasoning

### 1️⃣ Interpret Results

* Watch time ↑ (120 → 125 min, p = 0.03) → users are watching more
* Subscription ↓ (60 → 58%, p = 0.07) → possibly harmful trend, even if not significant

### 2️⃣ Business Judgment

* **Renewal is primary metric** → revenue impact matters more than watch time
* Watch time increase is nice-to-have, but negative trend in revenue is **red flag**

### 3️⃣ Check Hidden Traps

* **Simpson’s paradox:** Maybe heavy users drive watch time increase, but light users churn → overall effect hides subgroup problems
* **Multi-metric tradeoff:** Don’t optimize one metric blindly

### 4️⃣ Next Steps

* Analyze **user segments**: heavy vs light users
* Check **guardrail metrics**: churn, engagement, complaints
* Decide if the feature should be **rolled back, modified, or rolled out gradually**

---

# 🏆 FAANG-Level Answer (Example)

> The experiment shows that watch time per user increased significantly (p = 0.03), but subscription renewal decreased slightly (p = 0.07).
>
> Although renewal is not statistically significant, it is a key business metric, so we must be cautious.
>
> Before shipping, I would analyze user segments to check for Simpson’s paradox—perhaps heavy users watch more while light users churn. I would also monitor guardrail metrics like complaints, retention, and engagement.
>
> Based on this, I would **not ship immediately**. Instead, I’d either refine the feature, run a longer test, or roll it out to a subset of users while monitoring key metrics.

---

# 🔥 Core Lesson (Day 2)

> **Optimizing a secondary metric (watch time) while a primary metric (renewal/revenue) trends down is dangerous.**
>
> FAANG interviews often test **multi-metric tradeoffs** and **hidden segment effects**.

---



