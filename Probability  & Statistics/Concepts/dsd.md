

## Types of t-Test — When to Use What

---

### Type 1 — One-Sample t-Test

**Use when:** You have one group and want to compare its mean against a known or hypothesized value.

**Formula:**
```
t = (X̄ - μ₀) / (s / √n)       df = n - 1
```

**Conditions:**
- σ is unknown (you only have your sample's `s`)
- n < 30 (or any size if population is normal)
- Population is approximately normal

**Real-world examples:**

| Domain | Scenario |
|--------|----------|
| Manufacturing | A factory claims its bolts are 50mm long. You sample 20 bolts (X̄ = 49.3mm). Is this significantly different from 50mm? |
| Healthcare | A hospital claims average patient wait time is 30 min. You sample 25 visits. Did they meet the target? |
| Education | A school tests whether its students' average score (X̄ = 74) differs from the national average (μ₀ = 70). |
| Tech/Eng | Your API has a target latency of 200ms. You sample 15 requests. Is actual latency significantly above target? |

---

### Type 2 — Two-Sample t-Test (Welch's)

**Use when:** You have two independent groups and want to compare their means.

**Formula:**
```
t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)

df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

> Always default to Welch's over Student's t-test. It works equally well when variances are equal, and is more robust when they aren't.

**Conditions:**
- Two independent, unrelated groups
- σ unknown in both groups
- Both groups approximately normal (or n ≥ 30 each)

**Real-world examples:**

| Domain | Scenario |
|--------|----------|
| A/B Testing (small scale) | 40 users see old UI, 40 see new UI. Compare average session duration. |
| Drug trials | Treatment group (n=25) vs placebo group (n=25). Compare blood pressure reduction. |
| HR/People Analytics | Do engineers in team A have higher productivity scores than team B? |
| Education | Do students taught with Method A score higher than those with Method B? |
| E-commerce | Does a new checkout flow reduce cart abandonment time vs old flow? (small pilot) |

---

### Type 3 — Paired t-Test

**Use when:** The same subjects are measured twice — before/after, or matched in pairs.

**Formula:**
```
d_i  = X_after,i  -  X_before,i     (difference per subject)
t    = d̄ / (s_d / √n)               df = n - 1
```

**Conditions:**
- Same subjects measured under two conditions (or naturally matched pairs)
- The *differences* d_i are approximately normally distributed
- More powerful than two-sample t-test when applicable — removes between-subject noise

**Real-world examples:**

| Domain | Scenario |
|--------|----------|
| Product/UX | Same 30 users complete a task on old UI, then new UI. Compare task completion times. |
| Healthcare | Same 20 patients measured before and after taking a drug. Compare cholesterol levels. |
| Sports science | Athletes' sprint times before vs after a 6-week training programme. |
| Finance | Same portfolio's return before vs after a strategy change. |
| ML | Same test set evaluated by Model A and Model B — paired accuracy per sample (McNemar's is related). |

---

### At a Glance — Which t-Test?

| Situation | Test | Key signal |
|-----------|------|-----------|
| 1 group vs a fixed number | One-sample t | "Is our mean = target?" |
| 2 separate, unrelated groups | Two-sample (Welch's) | "Do these two groups differ?" |
| Same group measured twice | Paired t | "Did this group change?" |
| Large n (≥30), proportions, σ known | z-test | "Scale data, binary metric" |

---

## When Is z-Test Applicable? — The Full Checklist

This is one of the most common interview questions. The z-test is valid under **any of these three conditions**:

### Condition 1 — σ (population standard deviation) is truly known

This is rarer in practice than textbooks imply. Real scenarios:

- **Standardized tests** — SAT, GRE, IQ tests have known population σ from decades of data
- **Manufacturing processes** — a calibrated machine produces parts where σ is known from historical spec sheets
- **Quality control baselines** — a production line has a known defect rate σ established over millions of runs
- **Psychometric instruments** — validated scales with known population parameters

```
z = (X̄ - μ₀) / (σ / √n)       ← use true σ here
```

### Condition 2 — Sample size is large (n ≥ 30)

When n ≥ 30, by the Central Limit Theorem:
- The sampling distribution of X̄ is approximately normal regardless of the population shape
- The sample standard deviation `s` becomes a reliable estimate of σ
- The t-distribution with df ≥ 29 is nearly identical to the standard normal (z)

So you can substitute `s` for `σ` and use z-tables:

```
z ≈ (X̄ - μ₀) / (s / √n)       ← s substitutes for σ when n ≥ 30
```

> At FAANG scale (millions of users), n >> 30 is always true. z-test is the default.

### Condition 3 — Testing proportions (binary metrics)

For binary outcomes (clicked/not, converted/not, churned/not), the proportion `p̂` follows a binomial distribution. When:

```
n × p ≥ 10    AND    n × (1 - p) ≥ 10
```

The binomial is well-approximated by the normal, and you use the proportion z-test:

```
z = (p̂₁ - p̂₂) / √[ p̂(1-p̂)(1/n₁ + 1/n₂) ]

where p̂ = (x₁ + x₂) / (n₁ + n₂)    ← pooled proportion
```

This is the most common test in online A/B experimentation at tech companies.

---

### z-Test vs t-Test — The Convergence

As degrees of freedom (df) grow, t → z:

| df (≈ n) | t critical value (α=0.05, two-tail) | z critical value |
|----------|--------------------------------------|-----------------|
| 5 | ±2.571 | ±1.960 |
| 10 | ±2.228 | ±1.960 |
| 20 | ±2.086 | ±1.960 |
| 30 | ±2.042 | ±1.960 |
| 100 | ±1.984 | ±1.960 |
| ∞ | ±1.960 | ±1.960 |

By df = 30, the difference is tiny. By df = 100, it's negligible. This is the mathematical reason "n ≥ 30 → use z-test" works.

---

### Summary Decision Rule

```
Do you know σ?
  YES → z-test

Is n ≥ 30?
  YES → z-test (s reliably estimates σ)

Is your metric a proportion (binary)?
  YES → proportion z-test (if np ≥ 10 and n(1-p) ≥ 10)

Are you comparing 1 group to a known value, with small n and unknown σ?
  → One-sample t-test

Are you comparing 2 independent groups, small n, unknown σ?
  → Two-sample Welch's t-test

Are you comparing the same subjects measured twice?
  → Paired t-test
```
