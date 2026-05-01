# 📐 01 — Foundations of Hypothesis Testing

> *"The null hypothesis is never proven or established, but is possibly disproved, in the course of experimentation."* — Karl Popper

---

## 🧠 What is Hypothesis Testing?

Hypothesis testing is a **formal statistical framework** for making decisions about populations based on sample data. Rather than claiming certainty, it quantifies the *strength of evidence* against a default assumption.

### Core Idea (Intuition First)

Think of it like a **court trial**:
- The defendant is **innocent until proven guilty** → The null hypothesis is assumed true until data proves otherwise.
- You need *evidence beyond reasonable doubt* → You need a *statistically significant* result.
- You can only **"fail to convict"** — you can't prove innocence → You can only *fail to reject* H₀, not *accept* it.

### The Scientific Process Mapped

```
Real World Question
       ↓
Formulate H₀ and H₁
       ↓
Collect Sample Data
       ↓
Compute Test Statistic
       ↓
Compare to Null Distribution
       ↓
Make Decision (Reject / Fail to Reject H₀)
       ↓
Draw Conclusion in Context
```

---

## 📌 Null Hypothesis (H₀)

### Definition

The **null hypothesis** is the default assumption — the "nothing interesting is happening" statement. It represents:
- No effect
- No difference
- No relationship
- Status quo

### Mathematical Form

$$H_0: \theta = \theta_0$$

Where $\theta$ is the population parameter of interest (mean, proportion, variance, etc.) and $\theta_0$ is a specific claimed value.

### Common Forms

| Context | H₀ |
|---|---|
| One-sample mean | $H_0: \mu = \mu_0$ |
| Two-sample comparison | $H_0: \mu_1 = \mu_2$ |
| Proportion | $H_0: p = p_0$ |
| Correlation | $H_0: \rho = 0$ |
| Regression coefficient | $H_0: \beta_1 = 0$ |

### Real-World Examples

- "The new drug has **no effect** on blood pressure compared to placebo."
- "Clicking the blue button **does not change** conversion rate vs. the green button."
- "Customer age is **not correlated** with purchase value."

### ⚠️ Critical Nuance: H₀ is Falsifiable, Not Proven

You **never prove H₀ is true**. A failure to reject simply means *you don't have enough evidence to disprove it*. This is a crucial distinction — see Section 15 on interview traps.

---

## 📌 Alternative Hypothesis (H₁ or Hₐ)

### Definition

The **alternative hypothesis** is what you're trying to find evidence *for* — the research claim. It represents the presence of an effect, difference, or relationship.

### Mathematical Form

$$H_1: \theta \neq \theta_0 \quad \text{(two-tailed)}$$
$$H_1: \theta > \theta_0 \quad \text{(right-tailed)}$$
$$H_1: \theta < \theta_0 \quad \text{(left-tailed)}$$

### Key Properties
- Must be mutually exclusive with H₀
- Should be specified *before* seeing the data (pre-registration)
- Drives the directionality of your test

---

## 🔁 One-Tailed vs Two-Tailed Tests

### Visual Intuition

![One-tailed vs Two-tailed rejection regions](https://statisticsbyjim.com/wp-content/uploads/2017/08/one_two_tailed_test.png)

### Two-Tailed Test

**Use when:** You care about deviation in *either direction*.

$$H_0: \mu = 50 \quad \text{vs} \quad H_1: \mu \neq 50$$

Rejection region: both tails of the distribution.

$$\text{Reject } H_0 \text{ if } |Z| > z_{\alpha/2}$$

**Example:** Testing whether a coin is *biased at all* (could be heads-biased or tails-biased).

**Alpha split:** Each tail gets $\alpha/2$. If $\alpha = 0.05$, critical values are $\pm 1.96$.

---

### Right-Tailed (Upper-Tailed) Test

**Use when:** You only care if the parameter is *greater than* the null value.

$$H_0: \mu \leq \mu_0 \quad \text{vs} \quad H_1: \mu > \mu_0$$

Rejection region: right tail only.

$$\text{Reject } H_0 \text{ if } Z > z_{\alpha}$$

**Example:** Does the new drug *increase* mean survival time?

---

### Left-Tailed (Lower-Tailed) Test

**Use when:** You only care if the parameter is *less than* the null value.

$$H_0: \mu \geq \mu_0 \quad \text{vs} \quad H_1: \mu < \mu_0$$

Rejection region: left tail only.

$$\text{Reject } H_0 \text{ if } Z < -z_{\alpha}$$

**Example:** Is the new manufacturing process *reducing* defect rate?

---

### Decision Guide: One vs Two-Tailed

| Situation | Test Type |
|---|---|
| "Is there any difference?" | Two-tailed |
| "Is A better than B?" | One-tailed |
| "Did the metric improve?" | One-tailed |
| "Did anything change?" | Two-tailed |
| A/B test: "Does variant beat control?" | One-tailed (directional) |
| A/B test: "Is there any effect?" | Two-tailed |

### ⚠️ Pitfall: Choosing Tails After Seeing Data

**Never** switch from two-tailed to one-tailed after seeing the direction of the result. This is a form of p-hacking (covered in Section 13).

> Rule of Thumb: **When in doubt, use two-tailed.** It's more conservative and more defensible.

---

## 📊 Step-by-Step Framework (Universal)

```
Step 1: State H₀ and H₁ clearly (before data collection)
Step 2: Choose significance level α (typically 0.05)
Step 3: Choose the appropriate test (Z, t, χ², etc.)
Step 4: Collect data / compute test statistic
Step 5: Compute p-value (or find critical value)
Step 6: Decision rule:
        - If p-value < α → Reject H₀
        - If p-value ≥ α → Fail to Reject H₀
Step 7: State conclusion in the context of the problem
```

---

## 💬 Interview Questions & Answers

**Q1: What is the difference between H₀ and H₁? Who decides which is which?**

> A: H₀ is the default, conservative assumption — the "no effect" or "status quo" claim. H₁ is what the researcher wants to detect. You assign the claim you're testing *against* as H₀ because statistical tests are designed to quantify evidence *against* it. H₁ is what you'd conclude if you reject H₀.

**Q2: Why can't we just "accept the null hypothesis"?**

> A: Because failure to reject H₀ doesn't mean it's true — it just means we don't have sufficient evidence to disprove it. The absence of evidence is not evidence of absence. With a small sample, we may fail to detect a real effect (Type II error). Saying "accept H₀" implies we've proven it, which is logically incorrect.

**Q3: When would you use a one-tailed test in a product context?**

> A: When you have a strong, directional prior hypothesis and changing direction would require a separate investigation. For example: "We're launching a new recommendation algorithm — we'd only ship it if it *increases* click-through rate." If it decreases CTR, we'd investigate why, so we don't need to test that direction here.

**Q4: How do you define the null hypothesis for an A/B test?**

> A: H₀: There is no difference in the metric of interest (e.g., conversion rate) between the control and treatment groups. Formally: H₀: p_control = p_treatment, or equivalently H₀: Δp = 0.

---

## 🚨 Common Pitfalls

1. **Setting up H₁ as the null**: Always put "no effect" in H₀.
2. **Post-hoc directionality**: Deciding one-tailed vs. two-tailed after seeing the result.
3. **Conflating statistical and practical significance**: A significant result isn't necessarily meaningful.
4. **Collecting data without specifying H₀ first**: Leads to data dredging.
5. **Testing the wrong parameter**: Make sure H₀ and H₁ match what you actually want to know.

---

*Next: [02 — Type I & Type II Errors →](02_errors.md)*
