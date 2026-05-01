# ⚠️ 02 — Errors in Hypothesis Testing & Statistical Power

> *"All models are wrong, but some are useful — and all tests make errors, but some make fewer."*

---

## The Decision Matrix

Every hypothesis test results in one of four outcomes:

|  | **H₀ is Actually True** | **H₀ is Actually False** |
|---|---|---|
| **Reject H₀** | ❌ Type I Error (False Positive) | ✅ Correct Rejection (Power) |
| **Fail to Reject H₀** | ✅ Correct Acceptance | ❌ Type II Error (False Negative) |

![Type I vs Type II Error diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Type_I_and_type_II_errors.svg/800px-Type_I_and_type_II_errors.svg.png)

---

## 🔴 Type I Error (α — False Positive)

### Definition

A **Type I error** occurs when you **reject a true null hypothesis** — you conclude an effect exists when it doesn't.

$$P(\text{Type I Error}) = P(\text{Reject } H_0 \mid H_0 \text{ is True}) = \alpha$$

### Intuition

- **Medical analogy**: Diagnosing a healthy patient as sick.
- **Court analogy**: Convicting an innocent person.
- **A/B test analogy**: Shipping a feature that actually has no effect (wasted engineering resources, potentially harmful).

### Key Properties

- **You control this directly** by choosing $\alpha$ (significance level).
- Common choices: $\alpha = 0.05$, $0.01$, $0.001$
- At $\alpha = 0.05$: If H₀ is true and you run 100 experiments, ~5 will produce false positives by chance.

### Numeric Example

> A company runs 20 A/B tests per month. All have no real effect (H₀ true). At $\alpha = 0.05$:
>
> Expected false positives = $20 \times 0.05 = 1$ spurious "significant" result per month.

---

## 🔵 Type II Error (β — False Negative)

### Definition

A **Type II error** occurs when you **fail to reject a false null hypothesis** — you miss a real effect.

$$P(\text{Type II Error}) = P(\text{Fail to Reject } H_0 \mid H_0 \text{ is False}) = \beta$$

### Intuition

- **Medical analogy**: Missing a sick patient's diagnosis (they're actually ill but you say they're fine).
- **Court analogy**: Acquitting a guilty person.
- **A/B test analogy**: Not shipping a genuinely better feature (lost revenue opportunity).

### What Increases Type II Error?
- Small sample size $n$
- Tiny true effect size (hard to detect)
- High variability in data ($\sigma^2$ large)
- Strict $\alpha$ (more conservative → harder to reject → more $\beta$)

---

## ⚡ Statistical Power (1 − β)

### Definition

**Power** is the probability of correctly rejecting a false null hypothesis — the ability to *detect a real effect*.

$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 \mid H_0 \text{ is False})$$

### Visual Intuition

![Power visualization](https://www.statisticshowto.com/wp-content/uploads/2013/09/statistical-power.png)

Two overlapping distributions:
- **Left (H₀ distribution)**: Centered at $\mu_0$
- **Right (H₁ distribution)**: Centered at $\mu_1 = \mu_0 + \delta$ (true effect)
- The rejection region starts at the critical value
- **Power = area of H₁ distribution that falls in the rejection region**

### What Increases Power?

$$\text{Power} \uparrow \text{ when:}$$

| Factor | Direction | Why |
|---|---|---|
| Sample size $n$ | Increase | Narrower sampling distribution → easier to separate |
| Effect size $\delta = \mu_1 - \mu_0$ | Increase | Distributions further apart |
| Significance level $\alpha$ | Increase | Rejection region widens |
| Data variability $\sigma$ | Decrease | Less noise → cleaner signal |

### Power Formula (Z-test case)

$$\text{Power} = \Phi\left(\frac{|\mu_1 - \mu_0|}{\sigma/\sqrt{n}} - z_{\alpha/2}\right)$$

Where $\Phi$ is the standard normal CDF.

### Sample Size from Power Analysis

Rearranging:

$$n = \left(\frac{(z_{\alpha/2} + z_{\beta}) \cdot \sigma}{\mu_1 - \mu_0}\right)^2$$

**Example:**
- Want to detect a 5-unit shift in mean
- $\sigma = 20$, $\alpha = 0.05$, $\beta = 0.20$ (80% power)
- $z_{0.025} = 1.96$, $z_{0.20} = 0.842$

$$n = \left(\frac{(1.96 + 0.842) \times 20}{5}\right)^2 = \left(\frac{56.04}{5}\right)^2 = (11.21)^2 \approx 126 \text{ per group}$$

### Standard Power Convention

> In practice: **Power ≥ 80%** ($\beta \leq 0.20$) is the widely accepted minimum. High-stakes experiments (medical trials) use **90–95%**.

---

## ⚖️ The Tradeoff: α vs β

### Core Tension

For a **fixed sample size**, reducing $\alpha$ (being more strict about false positives) **automatically increases** $\beta$ (more false negatives), and vice versa. The only way to reduce *both* simultaneously is to **increase sample size**.

```
         Strict α (0.01)          Lenient α (0.10)
         ┌──────────────┐         ┌──────────────┐
         │  Few FP      │         │  Many FP     │
         │  Many FN     │         │  Few FN      │
         │  Low Power   │         │  High Power  │
         └──────────────┘         └──────────────┘
```

### Business Decision Framework

| Domain | Priority | Reasoning |
|---|---|---|
| Drug safety | Minimize α | A false positive → patients get useless/harmful drug |
| Cancer screening | Minimize β | A false negative → missed cancer → death |
| A/B testing (early stage) | Balance | Both matter; use standard α=0.05, power=80% |
| Feature kill decisions | Minimize β | Don't kill a good feature |

---

## 📐 Effect Size

Effect size quantifies *how large* the true effect is — independent of sample size.

### Cohen's d (for means)

$$d = \frac{\mu_1 - \mu_0}{\sigma}$$

| d value | Interpretation |
|---|---|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

### Why Effect Size Matters

With $n = 1{,}000{,}000$, even a $d = 0.001$ effect becomes statistically significant. This doesn't mean it's *practically* meaningful. Effect size bridges statistical and practical significance (see Section 14).

---

## 💬 Interview Questions & Answers

**Q1: What's the difference between Type I and Type II errors? Which is worse?**

> A: Type I (α) = false positive — concluding an effect exists when it doesn't. Type II (β) = false negative — missing a real effect. Which is worse depends on context. In drug testing, a Type I error could mean approving an ineffective (or harmful) drug — very costly. In cancer screening, a Type II error means missing a patient who has cancer — also very costly. In product analytics, a Type I error means shipping a feature with no real benefit (wasted engineering), while a Type II error means not shipping something genuinely good (missed revenue).

**Q2: How do you increase statistical power in an A/B test?**

> A: Four levers: (1) Increase sample size — the most effective. (2) Increase the minimum detectable effect (MDE) — accept you'll only detect larger changes. (3) Increase α — more lenient, but more false positives. (4) Reduce noise — better targeting, variance reduction techniques like CUPED.

**Q3: If a test fails to reach significance, is the null hypothesis true?**

> A: No. Failure to reject H₀ only means we lack sufficient evidence to disprove it. It could mean: the true effect is smaller than our MDE, our sample was too small, or there truly is no effect. We cannot distinguish these from a failed test alone. Always look at confidence intervals alongside the p-value.

**Q4: A PM wants to run an experiment but has only 2 weeks of data, giving 60% power. How do you advise?**

> A: I'd explain that at 60% power, we have a 40% chance of missing a real effect if it exists. That's very high. Options: (1) Run the experiment longer to hit 80% power. (2) Restrict the test to a segment with less variance. (3) If speed is critical, acknowledge the elevated Type II risk and plan to re-test if results are inconclusive. Never ship based on an underpowered nonsignificant result, but also don't dismiss a significant result from an underpowered test — it just means we got lucky to detect it.

---

## 🚨 Common Pitfalls

1. **Equating "not significant" with "no effect"** — could be underpowered.
2. **Forgetting to do power analysis before the experiment** — retroactive power analysis is uninformative.
3. **Using post-hoc power** (computing power after seeing results) — this is circular and misleading.
4. **Not considering effect size** — a tiny effect can be statistically significant with large n.
5. **One-size α** — don't blindly use 0.05; match to the cost of each error type.

---

*← [01 — Foundations](01_foundations.md) | [03 — Statistical Concepts →](03_statistical_concepts.md)*
