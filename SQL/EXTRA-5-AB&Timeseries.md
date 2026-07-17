# Day 13 + Day 16 — A/B Testing & Time Series Analysis in SQL
**FAANG SQL 30-Day Prep (Expanded Edition)**

---

## How to use this guide

Same format as the Day 20 fraud-detection rewrite: every pattern shows **the idea → the SQL → sample input → sample output → why → gotcha**, so you can trace the logic by hand instead of just reading a query and hoping it clicks. Two topics, two shared mini-datasets — one for the A/B testing half, one for the time series half.

---

# PART 1 — A/B Testing & Statistical Analysis

## The shared sample dataset

`ab_assignments(user_id, variant, assigned_date)`

| user_id | variant | assigned_date |
|---|---|---|
| U1 | control | 2026-06-01 |
| U2 | treatment | 2026-06-01 |
| U3 | control | 2026-06-01 |
| U4 | treatment | 2026-06-01 |
| U5 | control | 2026-06-02 |
| U6 | treatment | 2026-06-02 |

`orders(order_id, user_id, amount, order_date)`

| order_id | user_id | amount | order_date |
|---|---|---|---|
| O1 | U2 | 50 | 2026-06-03 |
| O2 | U4 | 80 | 2026-06-05 |
| O3 | U1 | 20 | 2026-06-10 |
| O4 | U6 | 100 | 2026-06-04 |

**What this means in plain terms:** 3 users in **control** (U1, U3, U5), 3 in **treatment** (U2, U4, U6). Only U1 converts in control ($20). All three treatment users convert ($50 + $80 + $100). This tiny dataset is deliberately small so you can hand-verify every formula below — real experiments need thousands of users per arm before a z-test means anything (see the Gotcha in section 4).

## Concepts Covered

1. A/B Test Setup & Sanity Checks
2. Core Metrics Computation
3. Lift & Relative Uplift
4. Z-Test for Proportions (CVR)
5. Confidence Intervals
6. T-Test for Continuous Metrics
7. Novelty Effect & Time-Series of Metrics
8. Segmented Analysis
9. Power Analysis — Minimum Sample Size
10. **NEW** — Multiple Testing Correction
11. **NEW** — Peeking Problem / Sequential Testing
12. **NEW** — Guardrail Metrics

---

## 1. A/B Test Setup & Sanity Checks

**The idea:** before you trust *any* result, verify the experiment itself was run correctly. Two failure modes wreck experiments silently: a broken randomizer (Sample Ratio Mismatch) and a bias that existed *before* the experiment even started (checked via an AA-style comparison of pre-experiment history).

```sql
-- Step 1: sample sizes per variant
SELECT variant, COUNT(DISTINCT user_id) AS users,
  MIN(assigned_date) AS start_date, MAX(assigned_date) AS end_date
FROM ab_assignments
GROUP BY variant;
```

**Output (from our data):**

| variant | users | start_date | end_date |
|---|---|---|---|
| control | 3 | 2026-06-01 | 2026-06-02 |
| treatment | 3 | 2026-06-01 | 2026-06-02 |

```sql
-- Step 2: Sample Ratio Mismatch (SRM) check
WITH counts AS (SELECT variant, COUNT(*) AS n FROM ab_assignments GROUP BY variant),
total AS (SELECT SUM(n) AS total FROM counts)
SELECT variant, n,
  ROUND(n * 100.0 / total.total, 2) AS actual_pct,
  50.0 AS expected_pct,
  ABS(n * 100.0 / total.total - 50) AS deviation_pct
FROM counts CROSS JOIN total;
```

**Output:**

| variant | n | actual_pct | expected_pct | deviation_pct |
|---|---|---|---|---|
| control | 3 | 50.00 | 50.0 | 0.00 |
| treatment | 3 | 50.00 | 50.0 | 0.00 |

**Why this matters:** a perfect 50/50 split here means the randomizer is behaving. In practice you'll see something like 49.3% / 50.7% — the rule of thumb is that a deviation over roughly 1% (or, more rigorously, a chi-square test p-value under 0.001) means **SRM detected**, and you should stop and debug the assignment logic *before* reading any lift numbers, because a broken randomizer can fabricate a "significant" result out of nothing but a selection bias.

```sql
-- Step 3: pre-experiment balance (AA test)
SELECT a.variant, AVG(prior.amount) AS avg_prior_spend, COUNT(DISTINCT a.user_id) AS users
FROM ab_assignments a
LEFT JOIN orders prior
  ON prior.user_id = a.user_id AND prior.order_date < a.assigned_date
GROUP BY a.variant;
```

**Gotcha:** people often skip Step 3 because there's no "prior" data in a brand-new product — but for any experiment on an *existing* user base, skipping the AA check is how you end up attributing a pre-existing group difference (e.g., treatment happened to get slightly higher-value users by chance) to your treatment itself.

---

## 2. Core Metrics Computation

**The idea:** every A/B report needs the same three numbers per variant — conversion rate (did they buy at all), revenue per user (how much did they spend on average, including zeros for non-converters), and average order value (how much *converters* spent per order). These are computed by first building one row per user (with a `LEFT JOIN` so non-converters show up as `NULL`/`0`, not missing rows).

```sql
WITH experiment_orders AS (
  SELECT a.user_id, a.variant,
    COUNT(o.order_id) AS orders,
    COALESCE(SUM(o.amount), 0) AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  LEFT JOIN orders o
    ON o.user_id = a.user_id
    AND o.order_date BETWEEN a.assigned_date AND a.assigned_date + INTERVAL 14 DAY
  GROUP BY a.user_id, a.variant
)
SELECT variant, COUNT(*) AS total_users, SUM(converted) AS conversions,
  ROUND(SUM(converted) * 100.0 / COUNT(*), 2) AS conversion_rate,
  ROUND(SUM(revenue) / COUNT(*), 2) AS revenue_per_user
FROM experiment_orders
GROUP BY variant;
```

**Output:**

| variant | total_users | conversions | conversion_rate | revenue_per_user |
|---|---|---|---|---|
| control | 3 | 1 | 33.33 | 6.67 |
| treatment | 3 | 3 | 100.00 | 76.67 |

**Why this output:** the `LEFT JOIN ... BETWEEN assigned_date AND assigned_date + 14 DAY` window means U5 and U3 (no orders at all) correctly contribute a `0` to the sum via `COALESCE`, rather than disappearing from the denominator — this is the single most common bug in A/B SQL: an `INNER JOIN` here would silently drop non-converters and inflate the conversion rate.

**Gotcha:** always fix the attribution window (14 days here) *before* looking at the data, and apply it identically to both variants. Cherry-picking a window after peeking at results is a form of p-hacking.

---

## 3. Lift & Relative Uplift

**The idea:** the raw difference in conversion rate (absolute lift) is useful for math, but stakeholders think in percentages (relative lift) — "conversion went up 66.7 percentage points" sounds insane; "conversion tripled" is the same fact stated the way a PM actually needs to hear it.

```sql
WITH metrics AS (
  SELECT variant, SUM(converted) * 1.0 / COUNT(*) AS cvr, AVG(revenue) AS avg_rev
  FROM experiment_orders GROUP BY variant
),
control AS (SELECT cvr AS ctrl_cvr, avg_rev AS ctrl_rev FROM metrics WHERE variant = 'control'),
treatment AS (SELECT cvr AS trt_cvr, avg_rev AS trt_rev FROM metrics WHERE variant = 'treatment')
SELECT
  ROUND(t.trt_cvr - c.ctrl_cvr, 4) AS absolute_lift_cvr,
  ROUND((t.trt_cvr - c.ctrl_cvr) * 100.0 / NULLIF(c.ctrl_cvr, 0), 2) AS relative_lift_cvr_pct
FROM treatment t CROSS JOIN control c;
```

**Output:**

| absolute_lift_cvr | relative_lift_cvr_pct |
|---|---|
| 0.6667 | 200.00 |

**Why this output:** treatment's CVR (1.00) minus control's CVR (0.333) is an absolute lift of 0.667 (66.7 percentage points); dividing that by control's own rate gives 200% — treatment converted at **three times** control's rate. Both numbers describe the exact same underlying fact, but relative lift is what goes in a slide deck.

**Gotcha:** relative lift explodes when the control rate is tiny — a control CVR of 0.1% moving to 0.2% is a "100% lift" that sounds dramatic but represents one extra conversion per thousand users. Always report the absolute lift alongside the relative one so nobody gets misled by a big-sounding percentage on a tiny base rate.

---

## 4. Z-Test for Proportions (Conversion Rate)

**The idea:** is the CVR difference we just measured *real*, or could it plausibly be random noise given how few users we tested? The two-proportion z-test answers this by pooling both groups' conversion counts into a single "expected" rate, then measuring how many standard errors apart the two observed rates are from each other.

```sql
WITH stats AS (
  SELECT variant, COUNT(*) AS n, SUM(converted) AS x, SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders GROUP BY variant
),
pooled AS (SELECT SUM(x) * 1.0 / SUM(n) AS p_pool FROM stats),
z_calc AS (
  SELECT
    MAX(CASE WHEN variant = 'treatment' THEN p END) AS p_t,
    MAX(CASE WHEN variant = 'control'   THEN p END) AS p_c,
    MAX(CASE WHEN variant = 'treatment' THEN n END) AS n_t,
    MAX(CASE WHEN variant = 'control'   THEN n END) AS n_c,
    po.p_pool
  FROM stats CROSS JOIN pooled po GROUP BY po.p_pool
)
SELECT p_t, p_c, ROUND(p_t - p_c, 4) AS diff, ROUND(p_pool, 4) AS p_pool,
  ROUND((p_t - p_c) / SQRT(p_pool * (1 - p_pool) * (1.0/n_t + 1.0/n_c)), 4) AS z_score,
  CASE WHEN ABS((p_t - p_c) / SQRT(p_pool * (1 - p_pool) * (1.0/n_t + 1.0/n_c))) > 1.96
       THEN 'Significant (95%)' ELSE 'Not Significant' END AS significance
FROM z_calc;
```

**Output:**

| p_t | p_c | diff | p_pool | z_score | significance |
|---|---|---|---|---|---|
| 1.00 | 0.333 | 0.667 | 0.667 | 1.63 | Not Significant |

**Why this output:** even though treatment's rate looks 3x better, the z-score only reaches 1.63 — below the 1.96 threshold for 95% confidence — because `n=3` per group is far too small for the standard error term (`1.0/n_t + 1.0/n_c`) to shrink enough to make the difference statistically distinguishable from noise. A 100%-vs-33% split on 3 users each is exactly the kind of result that *looks* dramatic but isn't proof of anything yet.

> 💡 **Z-score thresholds:** |z| > 1.645 → 90% confidence · |z| > 1.960 → 95% confidence · |z| > 2.576 → 99% confidence

**Gotcha:** this is the whole reason **power analysis (section 9)** exists — you should compute the required sample size *before* running the test, not run a test on whatever traffic you happened to get and hope the z-score clears 1.96.

---

## 5. Confidence Intervals

**The idea:** a point estimate like "CVR = 33.3%" hides how uncertain that number is with only 3 users. A 95% confidence interval expresses that uncertainty directly as a range you're 95% confident contains the true population rate.

```sql
WITH stats AS (
  SELECT variant, COUNT(*) AS n, SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders GROUP BY variant
)
SELECT variant, n, ROUND(p, 4) AS conversion_rate,
  ROUND(p - 1.96 * SQRT(p * (1-p) / n), 4) AS ci_lower,
  ROUND(p + 1.96 * SQRT(p * (1-p) / n), 4) AS ci_upper
FROM stats;
```

**Output:**

| variant | n | conversion_rate | ci_lower | ci_upper |
|---|---|---|---|---|
| control | 3 | 0.3333 | -0.2007 | 0.8673 |
| treatment | 3 | 1.0000 | 1.0000 | 1.0000 |

**Why this output:** with only 3 users, control's interval is enormous (-20% to 87% — nonsensical below zero, a known weakness of this normal-approximation formula at small n) and treatment's interval collapses to a single point because `p=1` makes the `p*(1-p)` term zero. Both are symptoms of the same underlying problem: **the sample is too small for this formula to behave sensibly.**

**Gotcha:** when `p` is at or near 0 or 1, or `n` is small, use the **Wilson score interval** instead of this normal-approximation formula — it doesn't produce impossible negative bounds and is the standard replacement in production experimentation platforms.

---

## 6. T-Test for Continuous Metrics

**The idea:** conversion rate is a yes/no metric; revenue per user is continuous (any dollar amount), so it needs a different test — Welch's t-test, which (unlike a basic two-sample t-test) doesn't assume both groups have equal variance, which is realistic for revenue data where a few big spenders in one arm can massively inflate that arm's variance.

```sql
WITH stats AS (
  SELECT variant, COUNT(*) AS n, AVG(revenue) AS mean_r, VARIANCE(revenue) AS var_r
  FROM experiment_orders GROUP BY variant
),
t_calc AS (
  SELECT
    MAX(CASE WHEN variant='treatment' THEN mean_r END) AS mean_t,
    MAX(CASE WHEN variant='control'   THEN mean_r END) AS mean_c,
    MAX(CASE WHEN variant='treatment' THEN var_r  END) AS var_t,
    MAX(CASE WHEN variant='control'   THEN var_r  END) AS var_c,
    MAX(CASE WHEN variant='treatment' THEN n END) AS n_t,
    MAX(CASE WHEN variant='control'   THEN n END) AS n_c
  FROM stats
)
SELECT ROUND(mean_t - mean_c, 2) AS mean_diff,
  ROUND((mean_t - mean_c) / SQRT(var_t/n_t + var_c/n_c), 4) AS t_statistic,
  CASE WHEN ABS((mean_t - mean_c) / SQRT(var_t/n_t + var_c/n_c)) > 1.96
       THEN 'Significant (95%)' ELSE 'Not Significant' END AS significance
FROM t_calc;
```

**Output:**

| mean_diff | t_statistic | significance |
|---|---|---|
| 70.00 | 3.24 | Significant (95%) |

**Why this output:** treatment's mean revenue ($76.67) minus control's ($6.67) is $70; dividing by the pooled standard error produces a t-statistic of 3.24, which *does* clear 1.96. Notice this contradicts the CVR z-test in section 4, which wasn't significant — that's a realistic and important lesson: **different metrics on the same experiment can disagree**, especially at small sample sizes, because revenue captures both "did they convert" and "how much did they spend," compounding the signal.

**Gotcha:** with `n=3` per group this t-statistic is not trustworthy in a real analysis either — you'd want to use the actual Student's t-distribution critical value for very small degrees of freedom (much higher than 1.96) rather than the large-sample normal approximation used here. This example is for tracing the *mechanics* of the formula, not for treating n=3 as adequate power.

---

## 7. Novelty Effect & Time-Series of Metrics

**The idea:** some treatments spike in performance for the first few days simply because it's *new and different*, then fade back to baseline (or below) as the novelty wears off. Tracking daily CVR — instead of only the pooled total — reveals this pattern, which a single aggregate number would hide completely.

```sql
WITH daily AS (
  SELECT a.variant, DATE(o.order_date) AS day,
    COUNT(DISTINCT a.user_id) AS users, COUNT(DISTINCT o.user_id) AS converters
  FROM ab_assignments a
  LEFT JOIN orders o ON o.user_id = a.user_id AND DATE(o.order_date) = DATE(a.assigned_date)
  GROUP BY a.variant, DATE(o.order_date)
)
SELECT variant, day, ROUND(converters * 100.0 / users, 2) AS daily_cvr,
  AVG(converters * 1.0 / users) OVER (
    PARTITION BY variant ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d_cvr
FROM daily ORDER BY variant, day;
```

**Why you'd want this even with a "flat" pooled result:** imagine treatment's pooled 14-day CVR is roughly equal to control's — a single aggregate would say "no effect." But the *daily* breakdown might show treatment converting at 2x control on days 1–3, then dropping below control by day 10. That's a novelty effect, and the correct read is "the feature works initially but has a retention problem," which is a completely different product decision than "the feature has no effect at all."

**Gotcha:** don't conclude "no novelty effect" from a short experiment — novelty effects can take 1–2 weeks to fully decay, so a 3-day experiment can't distinguish a real lift from a fading one. This is a common reason experienced experimenters insist on a minimum 2-week runtime even when the sample size math alone would allow stopping sooner.

---

## 8. Segmented Analysis

**The idea:** an overall "flat" result can hide the fact that the treatment helped one group and hurt another, canceling out in the aggregate (a form of Simpson's paradox). Breaking results down by segment (country, plan type, device, etc.) surfaces these **heterogeneous treatment effects**.

```sql
WITH experiment_orders AS (
  SELECT a.user_id, a.variant, u.country, u.plan_type,
    COALESCE(SUM(o.amount), 0) AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  JOIN users u ON a.user_id = u.user_id
  LEFT JOIN orders o ON o.user_id = a.user_id AND o.order_date >= a.assigned_date
  GROUP BY a.user_id, a.variant, u.country, u.plan_type
)
SELECT country, plan_type, variant, COUNT(*) AS users,
  ROUND(SUM(converted) * 100.0 / COUNT(*), 2) AS cvr
FROM experiment_orders
GROUP BY country, plan_type, variant
ORDER BY country, plan_type, variant;
```

**Gotcha (this is the important one):** segmented analysis is exploratory, not confirmatory — the more segments you slice by, the more likely you are to find *some* segment where the difference looks "significant" purely by chance (this is exactly the multiple-testing problem covered in section 10). Treat interesting segment findings as hypotheses to test in a *follow-up* experiment, not as conclusions from this one.

---

## 9. Power Analysis — Minimum Sample Size

**The idea:** run this calculation **before** launching an experiment, not after. It answers: "given my current baseline conversion rate and the smallest lift I actually care about detecting, how many users do I need per arm before a significant result is even possible?"

```sql
WITH params AS (
  SELECT 0.10 AS baseline_cvr, 0.005 AS min_detectable_effect, 1.96 AS z_alpha, 0.84 AS z_beta
)
SELECT baseline_cvr, min_detectable_effect,
  CEIL(2 * POW(z_alpha + z_beta, 2) * baseline_cvr * (1 - baseline_cvr) / POW(min_detectable_effect, 2))
    AS required_sample_size_per_variant
FROM params;
```

**Output:**

| baseline_cvr | min_detectable_effect | required_sample_size_per_variant |
|---|---|---|
| 0.10 | 0.005 | 27,681 |

**Why this output:** to detect a 0.5-percentage-point absolute lift on a 10% baseline (a 5% relative lift) with 95% confidence and 80% power, you need roughly **27,700 users per arm** — nearly 20,000x more than the toy 3-per-arm dataset used throughout this section. This is exactly why the earlier z-test and t-test results above shouldn't be taken as real conclusions: they illustrate the *formula*, not an adequately powered experiment.

**Gotcha:** the smaller the minimum detectable effect you want to catch, the sample size grows with the *square* of that effect (`min_detectable_effect` is squared in the denominator) — halving the effect size you want to detect roughly **quadruples** the required sample size. This is why teams often accept a coarser minimum detectable effect rather than run experiments for months.

---

## 10. NEW — Multiple Testing Correction

**The idea:** if you check 20 different metrics (or 20 segments) for "significance" at the 95% confidence level, you'd expect **about 1 of them to look significant purely by chance**, even if the treatment does absolutely nothing. The Bonferroni correction tightens your significance threshold in proportion to how many tests you're running, so your overall false-positive rate across all tests stays at 5%.

```sql
-- Table: experiment_metrics(metric_name, variant, p_value)  -- one p-value per metric tested

WITH metric_count AS (
  SELECT COUNT(DISTINCT metric_name) AS num_tests FROM experiment_metrics
)
SELECT em.metric_name, em.p_value,
  ROUND(0.05 / mc.num_tests, 5) AS bonferroni_threshold,
  CASE WHEN em.p_value < 0.05 / mc.num_tests
       THEN '✅ Significant after correction'
       ELSE '❌ Not significant after correction'
  END AS corrected_result
FROM experiment_metrics em CROSS JOIN metric_count mc
ORDER BY em.p_value;
```

**Sample input:** 10 metrics tested; one metric ("checkout_cvr") comes back with p = 0.03 — normally "significant" at the standard 0.05 cutoff.

**Sample output:**

| metric_name | p_value | bonferroni_threshold | corrected_result |
|---|---|---|---|
| checkout_cvr | 0.03 | 0.005 | ❌ Not significant after correction |

**Why this output:** with 10 metrics tested, the corrected threshold becomes `0.05 / 10 = 0.005`. A p-value of 0.03 clears the *uncorrected* 0.05 bar but not the corrected 0.005 bar — meaning this "win" is exactly the kind of false positive multiple testing correction exists to catch.

**Gotcha:** Bonferroni is deliberately conservative (it can cause real effects to be missed, i.e. more false negatives) — many teams instead designate **one primary metric decided before the experiment starts** and treat everything else as directional/exploratory, sidestepping the correction problem entirely rather than solving it statistically.

---

## 11. NEW — Peeking Problem / Sequential Testing

**The idea:** checking your z-score every day and stopping "as soon as it crosses 1.96" inflates your false-positive rate dramatically — sometimes to 20–30% instead of the intended 5% — because you're effectively running many sequential tests and grabbing the first one that happens to look good by chance. SQL can at least make the *daily peeking behavior* visible so a reviewer can catch it.

```sql
WITH daily_cumulative AS (
  SELECT a.variant, DATE(o.order_date) AS day,
    COUNT(DISTINCT a.user_id) OVER (PARTITION BY a.variant ORDER BY DATE(o.order_date)) AS cum_users,
    COUNT(DISTINCT o.user_id) OVER (PARTITION BY a.variant ORDER BY DATE(o.order_date)) AS cum_converters
  FROM ab_assignments a
  LEFT JOIN orders o ON o.user_id = a.user_id
)
SELECT variant, day, cum_users, cum_converters,
  ROUND(cum_converters * 1.0 / cum_users, 4) AS cumulative_cvr
FROM daily_cumulative
GROUP BY variant, day, cum_users, cum_converters
ORDER BY variant, day;
```

**Why this matters more than the query itself:** the query just tracks cumulative CVR day by day per variant — the real point is *how it's used*. If someone re-runs a z-test on this cumulative CVR every single day and stops the moment it exceeds 1.96, they are p-hacking, even with entirely correct SQL. The fix isn't a different query — it's a different *process*: pre-register a fixed sample size or duration (section 9), or use a formal sequential testing method (e.g., Sequential Probability Ratio Test / always-valid p-values) that's specifically designed to be checked repeatedly without inflating the false-positive rate.

**Gotcha:** "we'll just check it every day and stop when it looks good" is one of the most common real-world mistakes in industry — even senior engineers do this. If asked about it in an interview, naming the **peeking problem** and a mitigation (fixed horizon, or a sequential-testing method) is a strong signal.

---

## 12. NEW — Guardrail Metrics

**The idea:** an experiment can "win" on its primary metric while quietly damaging something else important (page load time, unsubscribe rate, customer support tickets). Guardrail metrics are checked in every experiment regardless of the hypothesis being tested, as a safety net.

```sql
-- Table: page_performance(user_id, variant, page_load_ms)
-- Table: support_tickets(user_id, variant, ticket_opened_date)

SELECT variant,
  ROUND(AVG(page_load_ms), 0) AS avg_page_load_ms,
  ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY page_load_ms), 0) AS p95_page_load_ms
FROM page_performance
GROUP BY variant;
```

**Sample output:**

| variant | avg_page_load_ms | p95_page_load_ms |
|---|---|---|
| control | 420 | 890 |
| treatment | 610 | 1,450 |

**Why this matters:** even if `treatment` wins decisively on conversion rate, a jump from 890ms to 1,450ms at the 95th percentile in page load time is a real cost — slow pages quietly erode long-term retention in ways a 14-day conversion metric won't capture. A launch decision has to weigh the primary metric win against guardrail damage, not treat the primary metric as the only thing that matters.

**Gotcha:** guardrails should be decided and documented *before* the experiment launches, exactly like the primary metric — deciding after the fact which "guardrail" mattered is just multiple testing wearing a different hat.

---

## Part 1 Summary Cheatsheet

| Check | What to Look For |
|---|---|
| SRM | Actual split within ~1% of expected |
| Pre-experiment balance | Similar prior behavior across variants |
| Z-test (CVR) | \|z\| > 1.96 for 95% significance |
| T-test (revenue) | \|t\| > 1.96 for 95% significance (large n) |
| Confidence interval | Does the CI exclude 0 (or use Wilson interval at small n)? |
| Novelty effect | Daily CVR — does treatment fade over time? |
| Segmented analysis | Heterogeneous effects — but exploratory, not confirmatory |
| Power analysis | Required n *before* running the experiment |
| Multiple testing correction | Tighten threshold (e.g. Bonferroni) when testing many metrics |
| Peeking problem | Don't stop early just because today's z-score looks good |
| Guardrail metrics | Primary metric win must be weighed against secondary harm |

---

## Part 1 Practice Questions

### 🟢 Q1 — Easy
Compute conversion rate, revenue per user, and stddev of revenue per variant.
<details><summary>Solution</summary>

```sql
WITH experiment_orders AS (
  SELECT a.user_id, a.variant,
    COALESCE(SUM(o.amount), 0) AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  LEFT JOIN orders o ON o.user_id = a.user_id AND o.order_date >= a.assigned_date
  GROUP BY a.user_id, a.variant
)
SELECT variant, COUNT(*) AS total_users, SUM(converted) AS conversions,
  ROUND(SUM(converted) * 100.0 / COUNT(*), 4) AS conversion_rate,
  ROUND(AVG(revenue), 4) AS avg_revenue_per_user,
  ROUND(STDDEV(revenue), 4) AS stddev_revenue
FROM experiment_orders GROUP BY variant;
```
</details>

### 🟡 Q2 — Medium
Compute z-score, absolute/relative lift, and 95% CI for CVR.
<details><summary>Solution</summary>

See **Section 4** and **Section 5** above — combine the pooled z-test with the per-variant CI formulas.
</details>

### 🔴 Q3 — Hard
Full report: SRM check + metrics + significance + segmentation in one result set (via `UNION ALL` across sections).
<details><summary>Solution</summary>

See the original multi-section `UNION ALL` pattern — SRM check rows, then overall significance rows, stacked into one report table with matching column shapes.
</details>

### 🟡 Q4 — Medium (NEW)
Given `experiment_metrics(metric_name, variant, p_value)` for 8 metrics, apply a Bonferroni correction and return which metrics remain significant.
<details><summary>Solution</summary>

See **Section 10** above — divide 0.05 by `COUNT(DISTINCT metric_name)` and compare each p-value against that corrected threshold.
</details>

---

# PART 2 — Time Series Analysis & Forecasting

## The shared sample dataset

Daily revenue, `2025-01-01` through `2025-01-10` (from a `daily_revenue(dt, revenue)` view built off `orders`):

| dt | revenue |
|---|---|
| 2025-01-01 | 100 |
| 2025-01-02 | 120 |
| 2025-01-03 | 110 |
| 2025-01-04 | 130 |
| 2025-01-05 | 125 |
| 2025-01-06 | 900 |
| 2025-01-07 | 115 |
| 2025-01-08 | 140 |
| 2025-01-09 | 135 |
| 2025-01-10 | 150 |

**What this means:** revenue drifts gently upward around $100–140/day — except **Jan 6, which spikes to $900**, roughly 7x the surrounding days. That single spike is what most of the anomaly-detection queries below will surface. This is intentionally a tiny window (10 days) so you can hand-check the moving-average math; real dashboards use months of history.

## Concepts Covered

1. Time Series Fundamentals (Date Spine)
2. Trend Detection with Moving Averages
3. Seasonality Detection
4. Anomaly Detection (Z-Score)
5. WoW / YoY Comparisons
6. Forecasting with Moving Averages
7. Trend Decomposition
8. Event Impact Analysis
9. FAANG Patterns (Spike Detection, MAPE)
10. **NEW** — Change Point Detection
11. **NEW** — Exponential Smoothing (Holt-Winters style)
12. **NEW** — Cohort Retention Curves

---

## 1. Date Spine (Zero-Fill Gaps)

**The idea:** if a day has zero orders, it simply won't appear in a `GROUP BY` on the orders table — silently *missing* a day is very different from *correctly showing* $0 revenue that day, and every downstream moving average / anomaly check will be wrong if gaps aren't filled. A **date spine** (a complete calendar generated independently of the data) fixes this by `LEFT JOIN`-ing real data onto guaranteed-complete dates.

```sql
WITH RECURSIVE dates AS (
  SELECT DATE'2025-01-01' AS dt
  UNION ALL
  SELECT dt + INTERVAL 1 DAY FROM dates WHERE dt < '2025-01-10'
)
SELECT d.dt, COALESCE(r.revenue, 0) AS revenue
FROM dates d
LEFT JOIN daily_revenue r ON d.dt = r.dt
ORDER BY d.dt;
```

**Input:** suppose the underlying `orders` table happens to have zero orders on Jan 3 (no row at all for that date in a naive `GROUP BY`).

**Output:**

| dt | revenue |
|---|---|
| 2025-01-01 | 100 |
| 2025-01-02 | 120 |
| 2025-01-03 | 0 |
| 2025-01-04 | 130 |
| ... | ... |

**Why this output:** the recursive CTE generates all 10 calendar dates regardless of what's in `orders`; the `LEFT JOIN` then attaches real revenue where it exists and `COALESCE` turns the resulting `NULL` (for Jan 3, where no matching row exists) into an explicit `0`. Every moving average below assumes this spine has already been applied — skipping it silently compresses a 10-day window into 9 real data points without you noticing.

**Gotcha:** always build the date spine as its own independent CTE, never derived from `MIN`/`MAX` of the data itself if there's any chance the *first* or *last* days in the data range also have gaps.

---

## 2. Trend Detection

**The idea:** daily revenue bounces around too much to eyeball a trend directly. A moving average (MA) smooths short-term noise so the underlying direction becomes visible; comparing a *short* MA (7-day) to a *longer* MA (30-day) gives a simple, robust "are we trending up or down right now" signal — the same logic behind a stock market moving-average crossover.

```sql
SELECT dt, revenue,
  AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d,
  AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma_3d
FROM daily_revenue ORDER BY dt;
```

**Output (using a 3-day MA in place of 30-day, since our sample is only 10 days):**

| dt | revenue | ma_3d |
|---|---|---|
| 2025-01-01 | 100 | 100.0 |
| 2025-01-02 | 120 | 110.0 |
| 2025-01-03 | 110 | 110.0 |
| 2025-01-04 | 130 | 120.0 |
| 2025-01-05 | 125 | 121.7 |
| 2025-01-06 | 900 | 385.0 |
| 2025-01-07 | 115 | 380.0 |
| 2025-01-08 | 140 | 385.0 |
| 2025-01-09 | 135 | 130.0 |
| 2025-01-10 | 150 | 141.7 |

**Why this output:** each row's `ma_3d` averages itself plus the two preceding rows (`ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`) — notice how the Jan 6 spike drags the 3-day average up to 385 for *three consecutive rows* (Jan 6, 7, and 8) even though revenue on Jan 7 and 8 is completely normal. This is the classic tradeoff of moving averages: they smooth noise, but a single extreme outlier "bleeds" into every window it's a part of.

**Gotcha:** for exactly this reason, always run anomaly detection (section 4) on the *raw* series, not the smoothed moving average — smoothing an outlier into a trend line can mask the very thing you're trying to detect.

---

## 3. Seasonality Detection

**The idea:** revenue naturally varies by day-of-week (weekends often differ from weekdays) and by month (holiday shopping spikes). A **seasonality index** — a period's average divided by the grand average — tells you which periods run above (index > 1) or below (index < 1) the overall baseline, so you can separate "this Tuesday is just a normal slow Tuesday" from "this Tuesday is genuinely underperforming."

```sql
WITH overall_avg AS (SELECT AVG(revenue) AS grand_avg FROM daily_revenue)
SELECT DAYNAME(dt) AS day_of_week, DAYOFWEEK(dt) AS dow_num,
  ROUND(AVG(revenue), 2) AS avg_revenue,
  ROUND(AVG(revenue) / o.grand_avg, 3) AS seasonality_index
FROM daily_revenue CROSS JOIN overall_avg o
GROUP BY DAYNAME(dt), DAYOFWEEK(dt), o.grand_avg
ORDER BY dow_num;
```

**Why this output would matter at scale:** with only one week of data here, each day-of-week bucket has just one observation, so the index is trivial (equal to that single day's revenue over the grand average). The pattern only becomes meaningful with several weeks of history — e.g., if every Saturday across 8 weeks averages 1.4x the grand mean, that's a real, repeatable seasonal effect you can plan staffing and inventory around, not noise.

**Gotcha:** don't compute a seasonality index from a short window like this sample and treat it as reliable — you need enough repeated cycles (weeks, months, years) for the "average" per period to actually average out random daily noise.

---

## 4. Anomaly Detection (Z-Score)

**The idea:** the same z-score logic from Day 20's fraud detection — how many standard deviations is today's value from what's "normal"? — applied to a metric over time instead of a per-user baseline. Two flavors: a **global** z-score (using the whole series' mean/stddev) is simple but gets distorted by the very outliers it's trying to detect; a **rolling** z-score (using only the trailing N days, excluding "today") adapts to a shifting baseline and doesn't get contaminated by the anomaly itself.

```sql
-- Global z-score
WITH stats AS (SELECT AVG(revenue) AS mean_rev, STDDEV(revenue) AS std_rev FROM daily_revenue)
SELECT d.dt, d.revenue,
  ROUND((d.revenue - s.mean_rev) / NULLIF(s.std_rev, 0), 2) AS z_score,
  CASE WHEN ABS((d.revenue - s.mean_rev) / NULLIF(s.std_rev, 0)) > 2 THEN 'Anomaly' ELSE 'Normal' END AS anomaly_flag
FROM daily_revenue d CROSS JOIN stats s
ORDER BY ABS(z_score) DESC;
```

**Output:**

| dt | revenue | z_score | anomaly_flag |
|---|---|---|---|
| 2025-01-06 | 900 | 2.87 | Anomaly |
| 2025-01-01 | 100 | -0.75 | Normal |
| 2025-01-04 | 130 | -0.53 | Normal |
| ... | ... | ... | ... |

**Why this output:** the mean across all 10 days (≈203, dragged upward by the $900 spike) and stddev (≈245, also inflated by that same spike) are computed *including* the anomaly itself — this is the key weakness of the global method. The spike is still large enough to clear the z > 2 threshold here, but in a longer series with more "normal" days diluting the mean, a single outlier can actually shrink its own z-score below the detection threshold by inflating the very std-dev used to measure it.

```sql
-- Rolling z-score (local context — the more robust version)
WITH rolling_stats AS (
  SELECT dt, revenue,
    AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_mean,
    STDDEV(revenue) OVER (ORDER BY dt ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_std
  FROM daily_revenue
)
SELECT dt, revenue, ROUND(rolling_mean, 2) AS rolling_mean,
  ROUND((revenue - rolling_mean) / NULLIF(rolling_std, 0), 2) AS local_z_score,
  CASE WHEN ABS((revenue - rolling_mean) / NULLIF(rolling_std, 0)) > 2 THEN 'Anomaly' ELSE 'Normal' END AS anomaly_flag
FROM rolling_stats ORDER BY dt;
```

**Output for Jan 6:**

| dt | revenue | rolling_mean | local_z_score | anomaly_flag |
|---|---|---|---|---|
| 2025-01-06 | 900 | 117.0 | huge (>10) | Anomaly |

**Why this output is better:** critically, the window `ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING` **excludes the current row** — so Jan 6's own $900 value never contaminates the mean/stddev it's being compared against. The rolling mean of ~117 (based only on Jan 1–5) makes the deviation far more extreme and unambiguous than the global method's diluted comparison.

**Gotcha:** always exclude the current row from its own baseline window (`1 PRECEDING`, not `CURRENT ROW`) — this is the time-series equivalent of the "exclude today from the baseline" rule from Day 20's fraud z-score section, and it's just as easy to get backwards.

---

## 5. WoW / YoY Comparisons

**The idea:** `LAG()` lets you compare today directly to the same day last week (7 rows back) or the same day last year (364 rows back — note: 364, not 365, so the day-of-week lines up, since 364 = 52×7).

```sql
SELECT dt, revenue,
  LAG(revenue, 1) OVER (ORDER BY dt) AS prev_day,
  ROUND((revenue - LAG(revenue,1) OVER (ORDER BY dt)) * 100.0 / NULLIF(LAG(revenue,1) OVER (ORDER BY dt), 0), 2) AS dod_pct,
  LAG(revenue, 7) OVER (ORDER BY dt) AS prev_week,
  ROUND((revenue - LAG(revenue,7) OVER (ORDER BY dt)) * 100.0 / NULLIF(LAG(revenue,7) OVER (ORDER BY dt), 0), 2) AS wow_pct
FROM daily_revenue ORDER BY dt;
```

**Output (partial):**

| dt | revenue | prev_day | dod_pct | prev_week | wow_pct |
|---|---|---|---|---|---|
| 2025-01-06 | 900 | 125 | 620.00 | NULL | NULL |
| 2025-01-08 | 140 | 115 | 21.74 | NULL | NULL |
| 2025-01-09 | 135 | 140 | -3.57 | NULL | NULL |

**Why this output:** `dod_pct` on Jan 6 explodes to 620% because `LAG(revenue,1)` correctly pulls the *previous calendar day's* value (Jan 5 = 125) via window ordering — not row position, which matters once the date spine is applied and there might be gaps. `wow_pct` is `NULL` for early dates simply because there's no data 7 days earlier in this 10-day sample; in a longer series it would show whether this week's performance actually differs from the same weekday last week, which controls for day-of-week seasonality automatically (comparing a Monday to the previous Monday, not to Sunday).

**Gotcha:** `LAG(revenue, 7)` assumes **no gaps** in the underlying rows — if you didn't build a full date spine (section 1) first, a missing day silently shifts every subsequent `LAG` by one day, quietly corrupting every WoW/YoY comparison after the gap.

---

## 6. Forecasting with Moving Averages

**The idea:** a simple moving average treats every day in the window equally; an exponentially weighted moving average (EWMA) instead weights *recent* days more heavily than older ones, which reacts faster to genuine trend changes while still smoothing noise.

```sql
SELECT dt, revenue,
  (revenue * 4 + LAG(revenue,1) OVER (ORDER BY dt) * 3
   + LAG(revenue,2) OVER (ORDER BY dt) * 2 + LAG(revenue,3) OVER (ORDER BY dt) * 1) / 10.0 AS ewma_forecast
FROM daily_revenue ORDER BY dt;
```

**Output for Jan 7 (the day right after the spike):**

| dt | revenue | ewma_forecast |
|---|---|---|
| 2025-01-07 | 115 | (115×4 + 900×3 + 125×2 + 130×1) / 10 = 331.5 |

**Why this output:** the weights (4, 3, 2, 1) favor the most recent days but still let the Jan 6 spike (weighted ×3, two days back) pull the forecast for Jan 7 up to 331.5 — nearly 3x the actual value that day. This demonstrates a real limitation: EWMA still gets dragged around by a single extreme outlier for several subsequent periods, just less severely than a plain moving average would.

**Gotcha:** for a series with occasional real spikes (promotions, viral events) rather than only noise, consider excluding known one-off events from the training window entirely (see section 8, Event Impact Analysis) rather than trying to smooth through them.

---

## 7. Trend Decomposition

**The idea:** any time series can be conceptually split into three additive/multiplicative parts: **trend** (the slow underlying direction), **seasonality** (the repeating day/week/month pattern), and **residual** (whatever's left over — including anomalies). Decomposing lets you ask "is this specific day high because of trend, because of seasonality, or because something unusual happened?" instead of lumping all three causes together.

```sql
WITH trend AS (
  SELECT dt, revenue,
    AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) AS trend_component
  FROM daily_revenue
),
overall_avg AS (SELECT AVG(revenue) AS grand_avg FROM daily_revenue)
SELECT t.dt, t.revenue, ROUND(t.trend_component, 2) AS trend,
  ROUND(t.revenue / NULLIF(t.trend_component, 0), 3) AS residual_ratio
FROM trend t CROSS JOIN overall_avg o
ORDER BY t.dt;
```

**Output for Jan 6:**

| dt | revenue | trend | residual_ratio |
|---|---|---|---|
| 2025-01-06 | 900 | 202.9 | 4.44 |

**Why this output:** the centered 7-day trend window (3 days before *and* after) smooths right through the spike, producing a "what this day should look like given the surrounding trend" estimate of ~203. Dividing actual revenue by that trend gives a residual ratio of 4.44 — i.e., Jan 6 was 4.44x higher than the trend alone would predict, isolating the "something unusual happened" component cleanly from the underlying gradual growth.

**Gotcha:** a *centered* window (`3 PRECEDING AND 3 FOLLOWING`) can't be computed for the most recent days in your data (there's no "future" data yet) — decomposition is inherently a look-back / historical-analysis tool, not something you can run in real time on today's data.

---

## 8. Event Impact Analysis

**The idea:** for planned events (a marketing campaign, a price change, a feature launch), compare a defined "pre" window to a defined "post" window — and ideally also to the same calendar period *last year* — to isolate the event's effect from both ordinary trend and ordinary seasonality.

```sql
WITH periods AS (
  SELECT dt, revenue,
    CASE
      WHEN dt BETWEEN '2025-01-01' AND '2025-01-05' THEN 'pre_event'
      WHEN dt BETWEEN '2025-01-06' AND '2025-01-10' THEN 'post_event'
    END AS period
  FROM daily_revenue
)
SELECT period, COUNT(*) AS days, ROUND(AVG(revenue), 2) AS avg_daily_revenue
FROM periods WHERE period IS NOT NULL
GROUP BY period;
```

**Output:**

| period | days | avg_daily_revenue |
|---|---|---|
| pre_event | 5 | 117.0 |
| post_event | 5 | 288.0 |

**Why this output:** treating Jan 6 as the "event day" (say, a flash sale), the post-event average ($288) is more than double the pre-event average ($117) — but notice that single $900 day is doing almost all of that work; without it, the post-event days (115, 140, 135, 150) look completely ordinary. This is exactly why you should also check the *median*, not just the mean, when one day dominates a short window — the mean here overstates the event's sustained impact.

**Gotcha:** always add a `baseline_last_year` period (the same calendar dates one year earlier) alongside pre/post — without it, you can't tell whether the "lift" is due to the event or to typical growth/seasonality that would have happened anyway.

---

## 9. FAANG Patterns

**Spike detection (search/traffic style):** compares each hour's volume to that same hour's historical average, flagging ratios far above normal — this is the general form the Jan 6 anomaly detection above is a specific instance of, just bucketed by hour-of-day instead of by absolute date.

```sql
WITH baseline AS (
  SELECT HOUR(event_time) AS hr, AVG(cnt) AS avg_searches
  FROM (SELECT event_time, COUNT(*) OVER (PARTITION BY DATE(event_time), HOUR(event_time)) AS cnt FROM search_events) t
  GROUP BY HOUR(event_time)
)
SELECT h.hr, h.searches, ROUND(h.searches / NULLIF(b.avg_searches, 0), 2) AS vs_avg_ratio,
  CASE WHEN h.searches > 3 * b.avg_searches THEN 'Spike' ELSE 'Normal' END AS status
FROM hourly h JOIN baseline b ON h.hr = b.hr
ORDER BY vs_avg_ratio DESC;
```

**Forecast accuracy (MAPE/MAE/RMSE):** once you have a forecast, you need to score how good it actually was.

```sql
SELECT product_id,
  ROUND(AVG(ABS(actual_demand - forecasted_demand) / NULLIF(actual_demand, 0)) * 100, 2) AS mape_pct,
  ROUND(AVG(ABS(actual_demand - forecasted_demand)), 2) AS mae,
  ROUND(SQRT(AVG(POW(actual_demand - forecasted_demand, 2))), 2) AS rmse
FROM demand_forecasts
GROUP BY product_id ORDER BY mape_pct;
```

**Sample output:**

| product_id | mape_pct | mae | rmse |
|---|---|---|---|
| P100 | 8.4 | 12.1 | 18.7 |

**Why MAPE is usually the headline number:** an 8.4% MAPE means "on average, the forecast was off by 8.4% of actual demand" — a business stakeholder can act on that directly, whereas MAE ($12.1 units) and RMSE ($18.7, which penalizes big misses harder than small ones because it squares errors) are more useful for comparing models to each other than for explaining forecast quality to a non-technical audience.

**Gotcha:** MAPE breaks down (divides by near-zero) for products with very low or zero actual demand — a product that sold 1 unit but was forecast at 3 shows a 200% error that dominates the average even though the absolute miss (2 units) is trivial. Use a weighted variant (WAPE, weighting by actual volume) when your data includes low-volume items.

---

## 10. NEW — Change Point Detection

**The idea:** anomaly detection (section 4) finds a single day that's unusual; **change point detection** finds the day the *underlying level itself permanently shifted* — e.g., revenue was steady around $120/day, then a pricing change on a specific date pushed it to a new steady-state around $180/day forever after. A simple SQL approach compares the average of a trailing window to the average of a leading window at every point and flags where the gap between them is largest.

```sql
WITH windows AS (
  SELECT dt, revenue,
    AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS avg_before,
    AVG(revenue) OVER (ORDER BY dt ROWS BETWEEN 1 FOLLOWING AND 4 FOLLOWING) AS avg_after
  FROM daily_revenue
)
SELECT dt, revenue, ROUND(avg_before, 2) AS avg_before, ROUND(avg_after, 2) AS avg_after,
  ROUND(avg_after - avg_before, 2) AS level_shift
FROM windows
WHERE avg_before IS NOT NULL AND avg_after IS NOT NULL
ORDER BY ABS(avg_after - avg_before) DESC;
```

**Output (top row):**

| dt | revenue | avg_before | avg_after | level_shift |
|---|---|---|---|---|
| 2025-01-06 | 900 | 116.25 | 135.0 | 18.75 |

**Why this output:** in our sample, revenue reverts *back* to roughly its old level right after Jan 6, so the "before" and "after" window averages end up close together (the spike is a one-off anomaly, not a permanent level shift) — a small `level_shift` correctly tells you this isn't a genuine change point. Contrast this with a real change point, where you'd see `avg_before ≈ 120` and `avg_after ≈ 180` sustained for many days afterward, producing a large and *durable* shift rather than a single-day blip.

**Gotcha:** this simple before/after window comparison confuses one-off spikes with genuine level shifts unless you also check that the "after" average *stays* elevated over many subsequent points, not just the four days immediately following — production change-point algorithms (e.g., PELT, Bayesian change point detection) formalize this durability check statistically; the SQL version here is a reasonable first-pass heuristic, not a replacement.

---

## 11. NEW — Exponential Smoothing (Holt-Winters style)

**The idea:** section 6's EWMA only smooths the *level* of the series. Holt-Winters-style triple exponential smoothing extends that idea to also track a **trend** component (is the level itself rising or falling) and a **seasonal** component (a repeating multiplier), each smoothed with its own weight. A full implementation typically needs a procedural loop, but the core update step can be expressed in SQL using window functions if you're willing to approximate it recursively.

```sql
-- Simplified level + trend smoothing (alpha=0.3 for level, beta=0.1 for trend)
WITH RECURSIVE smoothed AS (
  SELECT dt, revenue,
    revenue AS level,
    0 AS trend,
    ROW_NUMBER() OVER (ORDER BY dt) AS rn
  FROM daily_revenue WHERE dt = (SELECT MIN(dt) FROM daily_revenue)

  UNION ALL

  SELECT d.dt, d.revenue,
    0.3 * d.revenue + 0.7 * (s.level + s.trend)              AS level,
    0.1 * (0.3 * d.revenue + 0.7 * (s.level + s.trend) - s.level) + 0.9 * s.trend AS trend,
    s.rn + 1
  FROM smoothed s
  JOIN daily_revenue d ON d.dt = (SELECT dt FROM daily_revenue WHERE dt > s.dt ORDER BY dt LIMIT 1)
)
SELECT dt, revenue, ROUND(level, 2) AS smoothed_level, ROUND(trend, 2) AS smoothed_trend,
  ROUND(level + trend, 2) AS one_step_forecast
FROM smoothed ORDER BY dt;
```

**Why this is worth knowing rather than memorizing:** the recursive structure here — each row's smoothed level depends on the *previous row's* smoothed level, not the previous row's raw value — is exactly why this needs a recursive CTE instead of a plain window function; window functions can't reference their own previously-computed output. In a real interview, it's more valuable to say "this needs recursion because each smoothed value depends on the prior smoothed value, not the prior raw value" than to get the exact alpha/beta constants right.

**Gotcha:** most production forecasting is done in Python/R (`statsmodels`, `prophet`) rather than hand-rolled in SQL specifically because of this recursive dependency — SQL is the right tool for computing the *inputs* to a forecasting model (clean, gap-filled daily series) and for evaluating forecast accuracy after the fact (MAPE/MAE/RMSE), but rarely for the smoothing/fitting step itself.

---

## 12. NEW — Cohort Retention Curves

**The idea:** rather than tracking one aggregate metric over calendar time, track behavior over time-*since-signup* for each signup cohort — this answers "do users who joined in January stick around better than users who joined in February," which a single revenue-over-time chart can't show, because it blends users at every stage of their lifecycle together.

```sql
-- Table: users(user_id, signup_date)
-- Table: activity(user_id, activity_date)

WITH cohorts AS (
  SELECT user_id, DATE_FORMAT(signup_date, '%Y-%m') AS cohort_month, signup_date
  FROM users
),
activity_with_cohort AS (
  SELECT c.user_id, c.cohort_month,
    TIMESTAMPDIFF(DAY, c.signup_date, a.activity_date) AS days_since_signup
  FROM cohorts c
  JOIN activity a ON a.user_id = c.user_id
)
SELECT cohort_month, days_since_signup,
  COUNT(DISTINCT user_id) AS active_users,
  ROUND(COUNT(DISTINCT user_id) * 100.0 /
    MAX(COUNT(DISTINCT user_id)) OVER (PARTITION BY cohort_month), 2) AS pct_of_day0
FROM activity_with_cohort
WHERE days_since_signup IN (0, 1, 7, 14, 30)
GROUP BY cohort_month, days_since_signup
ORDER BY cohort_month, days_since_signup;
```

**Sample output:**

| cohort_month | days_since_signup | active_users | pct_of_day0 |
|---|---|---|---|
| 2025-01 | 0 | 500 | 100.00 |
| 2025-01 | 1 | 340 | 68.00 |
| 2025-01 | 7 | 210 | 42.00 |
| 2025-01 | 30 | 95 | 19.00 |

**Why this output:** `days_since_signup` re-anchors every user's timeline to "day 0 = their own signup," so a user who joined January 5th and one who joined January 20th both contribute to the "day 7" bucket, just measured 7 days after *their own* start rather than on the same calendar date — that's what makes cohorts comparable to each other regardless of when they joined. The steep drop from 100% (day 0) to 42% (day 7) is a classic retention curve shape — most attrition happens fast, then the curve flattens.

**Gotcha:** `MAX(...) OVER (PARTITION BY cohort_month)` assumes day 0 always has the highest active-user count for that cohort (true for signup-based cohorts, since everyone is "active" the day they sign up) — don't reuse this exact pattern for a metric where the peak isn't guaranteed to be on day 0.

---

## Part 2 Summary Cheatsheet

| Pattern | SQL Technique |
|---|---|
| Date spine | Recursive CTE calendar + `LEFT JOIN` + `COALESCE` |
| Trend (MA) | `AVG() OVER (ROWS BETWEEN N PRECEDING AND CURRENT ROW)` |
| Seasonality index | `period_avg / grand_avg` — needs several repeated cycles to be reliable |
| Global z-score anomaly | `(value - overall_mean) / overall_stddev` — distorted by its own outliers |
| Rolling z-score anomaly | Exclude current row: `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` |
| WoW / YoY | `LAG(value, 7)` / `LAG(value, 364)` — 364 keeps weekday aligned |
| EWMA forecast | Weighted sum of `LAG()` values, heavier weight on recent days |
| Decomposition | Centered window trend × seasonal index × residual |
| Event impact | Pre vs. post window + same-period-last-year baseline |
| MAPE/MAE/RMSE | Forecast accuracy — MAPE breaks down near zero-demand items |
| Change point | Compare trailing vs. leading window average at every point |
| Exponential smoothing | Recursive CTE — each smoothed value depends on the *prior smoothed* value |
| Cohort retention | Re-anchor time to `days_since_signup`, not calendar date |

---

## Part 2 Practice Questions

### 🟢 Q1 — Easy
Compute a 7-day moving average per product and flag days where actual revenue exceeds it.
<details><summary>Solution</summary>

```sql
WITH daily AS (
  SELECT sale_date, product_id, revenue,
    AVG(revenue) OVER (PARTITION BY product_id ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d
  FROM daily_sales
)
SELECT sale_date, product_id, revenue, ROUND(ma_7d, 2) AS ma_7d,
  CASE WHEN revenue > ma_7d THEN 1 ELSE 0 END AS above_ma
FROM daily ORDER BY product_id, sale_date;
```
</details>

### 🟡 Q2 — Medium
Flag anomalies using a rolling 30-day z-score (excluding the current day) per product.
<details><summary>Solution</summary>

See **Part 2, Section 4** (rolling z-score query) — same pattern, partitioned by `product_id`.
</details>

### 🔴 Q3 — Hard
For each marketing event, compute pre-vs-post average daily revenue in the same product category and flag "High Impact" if lift exceeds 10%.
<details><summary>Solution</summary>

See **Part 2, Section 8** — extend to a `JOIN` against `marketing_events` filtered to the matching `category`, with `CASE WHEN pct_lift > 10 THEN 'High Impact' ELSE 'Low Impact' END`.
</details>

### 🟡 Q4 — Medium (NEW)
Build a day-0/day-1/day-7/day-30 retention curve per signup cohort month.
<details><summary>Solution</summary>

See **Part 2, Section 12** above — `TIMESTAMPDIFF(DAY, signup_date, activity_date)` re-anchored per user, then `pct_of_day0` normalized against each cohort's own day-0 count.
</details>

---

## Overall Key Takeaways

**A/B Testing:**
- Always run SRM + AA checks *before* trusting any lift number
- Non-converters must appear as `0`, never disappear via an `INNER JOIN`
- Z-test for proportions, Welch's t-test for continuous metrics — different metrics can disagree, especially at small n
- Power analysis comes *before* the experiment, not after a disappointing z-score
- More metrics/segments tested = higher chance of a false positive — correct for it or pre-designate a primary metric
- Never stop early just because today's cumulative z-score looks good (the peeking problem)
- A primary-metric win still has to clear guardrail metrics before shipping

**Time Series:**
- Always zero-fill gaps with a date spine before computing anything else
- Smoothing (moving averages, EWMA) can mask the very anomalies you're hunting for — check the raw series too
- Exclude the current point from its own rolling baseline, exactly like the fraud z-score pattern
- 364-day lag (not 365) keeps day-of-week aligned for YoY comparisons
- A one-day spike and a permanent change point look similar at first glance — check whether the new level *persists*
- Recursive smoothing (Holt-Winters style) needs a recursive CTE because each value depends on the previous *smoothed* value, not the previous raw value
- Cohort curves re-anchor time to "days since signup" so cohorts from different calendar periods become comparable

---

*Day 13 + Day 16 combined rewrite complete 🚀*
