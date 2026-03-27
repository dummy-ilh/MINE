# 🧠 A/B Testing Cheat Sheet (DS / MLE Interviews)

### 1. 🔥 Core Framework (Always Start Here)
When an interviewer asks **ANY** A/B testing question, follow this structured approach:

**✅ Step-by-step thinking:**
1.  **Define Goal**
2.  **Define Metrics** (Primary + Guardrails)
3.  **Form Hypothesis**
4.  **Experiment Design**
5.  **Statistical Test**
6.  **Decision + Trade-offs**

---

### 2. 🎯 Metrics (Most Important Part)
#### Types of Metrics:
* **Primary Metric (North Star):** What you are specifically optimizing.
    * *Examples:* CTR (Click Through Rate), Conversion Rate, Revenue per user.
* **Guardrail Metrics (VERY IMPORTANT — FAANG loves this):** Used to prevent damage elsewhere in the ecosystem.
    * *Examples:* User engagement ↓, Session time ↓, Retention ↓.
* **Diagnostic Metrics:** Used to help explain *why* a change happened.

> 💡 **Interview Trap:** If you ignore guardrails, it is often a **reject signal**.

---

### 3. 📊 Hypothesis
**Structure:**
* **Null Hypothesis ($H_0$):** No effect (the change did nothing).
* **Alternative Hypothesis ($H_1$):** The change has a significant effect.

*Example:*
* $H_0$: New button color = same CTR.
* $H_1$: New button color increases CTR.

---

### 4. ⚙️ Experiment Design
#### 4.1 Randomization Unit
* **User-level:** Most common.
* **Session-level**
* **Device-level**

> 💡 **Interview Trick:** If users can switch devices, use **user-level** randomization to avoid "pollution."

#### 4.2 Sample Size (Power Analysis)
Factors required to calculate sample size:
* **Effect size:** Minimum Detectable Effect (MDE).
* **Variance** of the metric.
* **Significance level ($\alpha$):** Usually 0.05.
* **Power ($1 - \beta$):** Usually 0.80.

#### 4.3 Duration
Run long enough to:
* Capture **weekly cycles** (weekdays vs. weekends).
* Avoid **novelty effects** (users clicking just because it's new).

#### 4.4 A/A Test
Used to validate experiment setup and detect underlying bias or system errors.

---

### 5. 📐 Statistical Testing
#### 5.1 Common Tests
| Metric Type | Test |
| :--- | :--- |
| **Conversion (binary)** | Z-test / Chi-square |
| **Mean (continuous)** | t-test |
| **Non-normal Data** | Mann-Whitney U test |

#### 5.2 Key Formula (Conversion Rate)
$$Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{p(1-p)(\frac{1}{n_1}+\frac{1}{n_2})}}$$
*Where:*
* $\hat{p}_1, \hat{p}_2$: Sample conversion rates.
* $p$: Pooled probability.

#### 5.3 P-value
The probability that the observed result occurred by chance.
* If **p < 0.05** → Reject $H_0$.

#### 5.4 Confidence Interval (CI)
Shows the range of the effect. **FAANG prefers CI interpretation** over p-values because it provides the magnitude of the change.

---

### 6. ⚠️ Common Pitfalls (VERY HIGH SIGNAL)
1.  **🚫 Peeking Problem:** Checking results early leads to inflated false positives.
2.  **🚫 Multiple Testing:** Running many experiments simultaneously increases Type I error.
    * *Fix:* **Bonferroni correction** or **False Discovery Rate (FDR)**.
3.  **🚫 Selection Bias:** Non-random assignment of users.
4.  **🚫 Simpson’s Paradox:** When aggregated data provides a misleading trend that disappears when data is segmented.
5.  **🚫 Novelty Effect:** Users behave differently initially because the feature is new, but return to baseline later.
6.  **🚫 Network Effects:** One user's behavior affects another (common in social apps/marketplaces).

---

### 7. 🧪 Advanced Topics (FAANG Differentiator)
* **7.1 CUPED (Variance Reduction):** Uses pre-experiment data to reduce variance, allowing for faster experiments with smaller samples.
* **7.2 Sequential Testing:** A methodology to avoid peeking issues; used in real-time monitoring systems.
* **7.3 Bayesian A/B Testing:** Provides a direct probability interpretation (e.g., "There is a 95% probability that variant B is better than A").
* **7.4 Heterogeneous Treatment Effects:** Analyzing how the effect differs across segments (e.g., New users ✅ vs. Old users ❌).

---

### 8. 💡 Interpretation (Most Important in Interview)
* **❌ Weak Answer:** "p < 0.05, so we should launch."
* **✅ Strong Answer:** * Is the effect **practically significant** (ROI)?
    * Are there negative impacts on **guardrail metrics**?
    * Is the result **consistent across segments**?
    * Are there any **long-term risks**?

---

### 9. 🧠 Standard Interview Questions
* **Q1: New feature increases CTR. Ship it?**
    * *Structure:* Check stat-sig → Check guardrails → Check long-term impact → Check segments.
* **Q2: No significant result. What do you do?**
    * Increase sample size, check power, check metric sensitivity, or look at specific segments.
* **Q3: Metric improves but revenue drops.**
    * 🔥 **GOLD Answer:** Perform a trade-off analysis; the primary metric might be misaligned with the business objective.
* **Q4: How to reduce experiment time?**
    * Use CUPED, choose a more sensitive metric, or increase traffic.

---

### 10. ⚡ Golden Answer Template (Memorize This)
> "First, I’d define the **goal** and **primary metric**, along with **guardrail metrics**. Then I’d design a **randomized experiment** at the user level, ensuring sufficient **sample size and duration**. Next, I’d run the appropriate **statistical test** and evaluate both **statistical and practical significance**. Finally, I’d check for pitfalls like **bias, novelty effects, and segment-level differences** before making a decision."

---

### 11. 🚀 FAANG-Level Insights (What Gets You Hired)
* Talk about **trade-offs**, not just results.
* Mention **guardrails** without being prompted.
* Bring up **real-world constraints** (engineering cost, latency).
* **Question the metric** itself—is it actually measuring what matters?
* Think like a **product + data + system owner**.
