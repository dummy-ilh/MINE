# Day 13 — A/B Testing & Statistical Analysis in SQL
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. A/B Test Setup & Sanity Checks
2. Core Metrics Computation
3. Lift & Relative Uplift
4. Z-Test for Proportions (CVR)
5. Confidence Intervals
6. T-Test for Continuous Metrics
7. Novelty Effect & Time-Series
8. Segmented Analysis
9. Power Analysis — Minimum Sample Size

---


## 1. A/B Test Setup & Sanity Checks

Before analyzing results, always validate the experiment.

```sql
-- Table: ab_assignments(user_id, variant, assigned_date)
-- Table: orders(order_id, user_id, amount, order_date)

-- Step 1: Check sample sizes per variant
SELECT variant,
  COUNT(DISTINCT user_id)    AS users,
  MIN(assigned_date)         AS start_date,
  MAX(assigned_date)         AS end_date
FROM ab_assignments
GROUP BY variant;
```

```sql
-- Step 2: Sample Ratio Mismatch (SRM) check
-- If you expect 50/50 split, actual should be close to 50/50
WITH counts AS (
  SELECT variant, COUNT(*) AS n
  FROM ab_assignments
  GROUP BY variant
),
total AS (SELECT SUM(n) AS total FROM counts)
SELECT variant, n,
  ROUND(n * 100.0 / total.total, 2) AS actual_pct,
  50.0                              AS expected_pct,
  ABS(n * 100.0 / total.total - 50) AS deviation_pct
FROM counts CROSS JOIN total;
-- If deviation > 1% → SRM detected → experiment is invalid
```

```sql
-- Step 3: Check pre-experiment balance (AA test)
-- Users in both groups should have similar prior behavior
SELECT
  a.variant,
  AVG(prior.amount)          AS avg_prior_spend,
  COUNT(DISTINCT a.user_id)  AS users
FROM ab_assignments a
LEFT JOIN orders prior
  ON  prior.user_id    = a.user_id
  AND prior.order_date < a.assigned_date   -- only pre-experiment
GROUP BY a.variant;
-- Both variants should have similar avg_prior_spend
```

---

## 2. Core Metrics Computation

```sql
-- Primary metrics: conversion rate, revenue per user, avg order value
WITH experiment_orders AS (
  SELECT a.user_id, a.variant,
    COUNT(o.order_id)    AS orders,
    SUM(o.amount)        AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  LEFT JOIN orders o
    ON  o.user_id    = a.user_id
    AND o.order_date BETWEEN a.assigned_date
                         AND a.assigned_date + INTERVAL 14 DAY
  GROUP BY a.user_id, a.variant
)
SELECT variant,
  COUNT(*)                                   AS total_users,
  SUM(converted)                             AS conversions,
  ROUND(SUM(converted) * 100.0 /
        COUNT(*), 4)                         AS conversion_rate,
  ROUND(SUM(revenue) / COUNT(*), 4)          AS revenue_per_user,
  ROUND(SUM(revenue) /
        NULLIF(SUM(orders), 0), 4)           AS avg_order_value,
  ROUND(AVG(revenue), 4)                     AS avg_revenue,
  ROUND(STDDEV(revenue), 4)                  AS stddev_revenue
FROM experiment_orders
GROUP BY variant;
```

---

## 3. Lift & Relative Uplift

```sql
WITH metrics AS (
  SELECT variant,
    COUNT(*)                          AS n,
    SUM(converted)                    AS conversions,
    SUM(converted) * 1.0 / COUNT(*)  AS cvr,
    AVG(revenue)                      AS avg_rev,
    STDDEV(revenue)                   AS std_rev
  FROM experiment_orders
  GROUP BY variant
),
control AS (
  SELECT cvr AS ctrl_cvr, avg_rev AS ctrl_rev
  FROM metrics WHERE variant = 'control'
),
treatment AS (
  SELECT cvr AS trt_cvr, avg_rev AS trt_rev
  FROM metrics WHERE variant = 'treatment'
)
SELECT
  t.trt_cvr - c.ctrl_cvr                              AS absolute_lift_cvr,
  ROUND((t.trt_cvr - c.ctrl_cvr) * 100.0 /
        NULLIF(c.ctrl_cvr, 0), 4)                     AS relative_lift_cvr_pct,
  t.trt_rev - c.ctrl_rev                              AS absolute_lift_rev,
  ROUND((t.trt_rev - c.ctrl_rev) * 100.0 /
        NULLIF(c.ctrl_rev, 0), 4)                     AS relative_lift_rev_pct
FROM treatment t CROSS JOIN control c;
```

---

## 4. Z-Test for Proportions (Conversion Rate)

The standard test for A/B conversion rate significance.

```sql
-- Z-score for difference in proportions
WITH stats AS (
  SELECT variant,
    COUNT(*)                         AS n,
    SUM(converted)                   AS x,
    SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders
  GROUP BY variant
),
pooled AS (
  SELECT
    SUM(x) * 1.0 / SUM(n) AS p_pool
  FROM stats
),
z_calc AS (
  SELECT
    MAX(CASE WHEN variant = 'treatment' THEN p END) AS p_t,
    MAX(CASE WHEN variant = 'control'   THEN p END) AS p_c,
    MAX(CASE WHEN variant = 'treatment' THEN n END) AS n_t,
    MAX(CASE WHEN variant = 'control'   THEN n END) AS n_c,
    po.p_pool
  FROM stats CROSS JOIN pooled po
  GROUP BY po.p_pool
)
SELECT
  p_t, p_c,
  p_t - p_c                                            AS diff,
  p_pool,
  ROUND(
    (p_t - p_c) /
    SQRT(p_pool * (1 - p_pool) * (1.0/n_t + 1.0/n_c))
  , 4)                                                  AS z_score,
  -- |z| > 1.96 → statistically significant at 95% confidence
  CASE
    WHEN ABS((p_t - p_c) /
      SQRT(p_pool * (1 - p_pool) * (1.0/n_t + 1.0/n_c))) > 1.96
    THEN 'Significant (95%)'
    ELSE 'Not Significant'
  END AS significance
FROM z_calc;
```

> 💡 **Z-score thresholds:**
> - |z| > 1.645 → 90% confidence
> - |z| > 1.960 → 95% confidence
> - |z| > 2.576 → 99% confidence

---

## 5. Confidence Intervals

```sql
-- 95% CI for conversion rate per variant
WITH stats AS (
  SELECT variant,
    COUNT(*)                          AS n,
    SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders
  GROUP BY variant
)
SELECT variant, n, ROUND(p, 4) AS conversion_rate,
  -- 95% CI: p ± 1.96 * sqrt(p(1-p)/n)
  ROUND(p - 1.96 * SQRT(p * (1-p) / n), 4) AS ci_lower,
  ROUND(p + 1.96 * SQRT(p * (1-p) / n), 4) AS ci_upper
FROM stats;
```

```sql
-- 95% CI for mean revenue per user
WITH stats AS (
  SELECT variant,
    COUNT(*)        AS n,
    AVG(revenue)    AS mean_rev,
    STDDEV(revenue) AS std_rev
  FROM experiment_orders
  GROUP BY variant
)
SELECT variant,
  ROUND(mean_rev, 2)                                   AS mean_revenue,
  ROUND(mean_rev - 1.96 * std_rev / SQRT(n), 2)       AS ci_lower,
  ROUND(mean_rev + 1.96 * std_rev / SQRT(n), 2)       AS ci_upper
FROM stats;
```

---

## 6. T-Test for Continuous Metrics

```sql
-- Welch's t-test for difference in means (revenue)
WITH stats AS (
  SELECT variant,
    COUNT(*)        AS n,
    AVG(revenue)    AS mean_r,
    STDDEV(revenue) AS std_r,
    VARIANCE(revenue) AS var_r
  FROM experiment_orders
  GROUP BY variant
),
t_calc AS (
  SELECT
    MAX(CASE WHEN variant = 'treatment' THEN mean_r  END) AS mean_t,
    MAX(CASE WHEN variant = 'control'   THEN mean_r  END) AS mean_c,
    MAX(CASE WHEN variant = 'treatment' THEN var_r   END) AS var_t,
    MAX(CASE WHEN variant = 'control'   THEN var_r   END) AS var_c,
    MAX(CASE WHEN variant = 'treatment' THEN n       END) AS n_t,
    MAX(CASE WHEN variant = 'control'   THEN n       END) AS n_c
  FROM stats
)
SELECT
  mean_t - mean_c                                      AS mean_diff,
  ROUND(
    (mean_t - mean_c) /
    SQRT(var_t / n_t + var_c / n_c)
  , 4)                                                  AS t_statistic,
  CASE
    WHEN ABS((mean_t - mean_c) /
      SQRT(var_t / n_t + var_c / n_c)) > 1.96
    THEN 'Significant (95%)'
    ELSE 'Not Significant'
  END AS significance
FROM t_calc;
```

---

## 7. Novelty Effect & Time-Series of Metrics

```sql
-- Track daily conversion rate per variant
-- to detect novelty effect (treatment spikes then fades)
WITH daily AS (
  SELECT
    a.variant,
    DATE(o.order_date)                  AS day,
    COUNT(DISTINCT a.user_id)           AS users,
    COUNT(DISTINCT o.user_id)           AS converters
  FROM ab_assignments a
  LEFT JOIN orders o
    ON  o.user_id = a.user_id
    AND DATE(o.order_date) = DATE(a.assigned_date)
  GROUP BY a.variant, DATE(o.order_date)
)
SELECT variant, day,
  ROUND(converters * 100.0 / users, 2)     AS daily_cvr,
  AVG(converters * 1.0 / users) OVER (
    PARTITION BY variant
    ORDER BY day
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  )                                         AS rolling_7d_cvr
FROM daily
ORDER BY variant, day;
```

---

## 8. Segmented Analysis

```sql
-- Break results down by user segment
-- Did the treatment work better for certain users?
WITH experiment_orders AS (
  SELECT a.user_id, a.variant, u.country, u.plan_type,
    COALESCE(SUM(o.amount), 0) AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  JOIN users u ON a.user_id = u.user_id
  LEFT JOIN orders o
    ON  o.user_id    = a.user_id
    AND o.order_date >= a.assigned_date
  GROUP BY a.user_id, a.variant, u.country, u.plan_type
)
SELECT country, plan_type, variant,
  COUNT(*)                                    AS users,
  ROUND(SUM(converted) * 100.0 / COUNT(*), 2) AS cvr,
  ROUND(AVG(revenue), 2)                      AS avg_revenue
FROM experiment_orders
GROUP BY country, plan_type, variant
ORDER BY country, plan_type, variant;
```

---

## 9. Power Analysis — Minimum Sample Size

```sql
-- How many users do you need to detect a 5% lift in CVR?
-- Formula: n = 2 * (z_alpha + z_beta)^2 * p(1-p) / delta^2
-- z_alpha = 1.96 (95% CI), z_beta = 0.84 (80% power)

WITH params AS (
  SELECT
    0.10  AS baseline_cvr,  -- current conversion rate
    0.005 AS min_detectable_effect,  -- 5% relative = 0.5% absolute
    1.96  AS z_alpha,  -- 95% confidence
    0.84  AS z_beta    -- 80% power
)
SELECT
  baseline_cvr,
  min_detectable_effect,
  CEIL(
    2 * POW(z_alpha + z_beta, 2) *
    baseline_cvr * (1 - baseline_cvr) /
    POW(min_detectable_effect, 2)
  ) AS required_sample_size_per_variant
FROM params;
```

---

## Summary Cheatsheet

| Check | What to Look For |
|---|---|
| SRM | Actual split within 1% of expected |
| Pre-experiment balance | Similar prior behavior across variants |
| Z-test (CVR) | z > 1.96 for 95% significance |
| T-test (revenue) | t > 1.96 for 95% significance |
| Confidence interval | Does CI exclude 0? |
| Novelty effect | Daily CVR — does treatment fade over time? |
| Segmented analysis | Heterogeneous treatment effects by group |
| Power analysis | Required n before running experiment |

---

### Q1 — Easy ✅
```sql
-- Conversion rate, revenue per user per variant
WITH experiment_orders AS (
  SELECT a.user_id, a.variant,
    COALESCE(SUM(o.amount), 0) AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL
             THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  LEFT JOIN orders o
    ON  o.user_id    = a.user_id
    AND o.order_date >= a.assigned_date
  GROUP BY a.user_id, a.variant
)
SELECT variant,
  COUNT(*)                                    AS total_users,
  SUM(converted)                              AS conversions,
  ROUND(SUM(converted) * 100.0 /
        COUNT(*), 4)                          AS conversion_rate,
  ROUND(AVG(revenue), 4)                      AS avg_revenue_per_user,
  ROUND(STDDEV(revenue), 4)                   AS stddev_revenue
FROM experiment_orders
GROUP BY variant;
```

---

### Q2 — Medium ✅
```sql
-- Z-score + 95% CI for conversion rate
WITH stats AS (
  SELECT variant,
    COUNT(*)                          AS n,
    SUM(converted)                    AS x,
    SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders
  GROUP BY variant
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
SELECT
  ROUND(p_t - p_c, 4)                                AS absolute_lift,
  ROUND((p_t - p_c) * 100.0 /
        NULLIF(p_c, 0), 2)                           AS relative_lift_pct,
  ROUND((p_t - p_c) /
    SQRT(p_pool * (1-p_pool) *
         (1.0/n_t + 1.0/n_c)), 4)                   AS z_score,
  ROUND(p_t - 1.96 * SQRT(p_t*(1-p_t)/n_t), 4)     AS trt_ci_lower,
  ROUND(p_t + 1.96 * SQRT(p_t*(1-p_t)/n_t), 4)     AS trt_ci_upper,
  CASE
    WHEN ABS((p_t - p_c) /
      SQRT(p_pool * (1-p_pool) *
           (1.0/n_t + 1.0/n_c))) > 1.96
    THEN 'Significant (95%)'
    ELSE 'Not Significant'
  END AS result
FROM z_calc;
```

---

### Q3 — Hard ✅
```sql
-- Full A/B report: SRM check + metrics + significance + segmentation
WITH sample_check AS (
  SELECT variant, COUNT(*) AS n,
    ROUND(COUNT(*) * 100.0 /
      SUM(COUNT(*)) OVER (), 2) AS actual_pct
  FROM ab_assignments GROUP BY variant
),
experiment_orders AS (
  SELECT a.user_id, a.variant, u.country,
    COALESCE(SUM(o.amount), 0) AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL
             THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  JOIN users u ON a.user_id = u.user_id
  LEFT JOIN orders o
    ON  o.user_id    = a.user_id
    AND o.order_date >= a.assigned_date
    AND o.order_date <= a.assigned_date + INTERVAL 14 DAY
  GROUP BY a.user_id, a.variant, u.country
),
overall AS (
  SELECT variant,
    COUNT(*)                          AS n,
    SUM(converted) * 1.0 / COUNT(*) AS cvr,
    AVG(revenue)                      AS avg_rev,
    STDDEV(revenue)                   AS std_rev
  FROM experiment_orders GROUP BY variant
),
pooled AS (
  SELECT SUM(converted) * 1.0 / COUNT(*) AS p_pool
  FROM experiment_orders
)
-- SRM check
SELECT 'SRM Check' AS section,
  variant, n, actual_pct, NULL AS cvr,
  NULL AS z_score, NULL AS significance
FROM sample_check

UNION ALL

-- Overall significance
SELECT 'Overall Results',
  o.variant, o.n,
  NULL,
  ROUND(o.cvr, 4),
  ROUND(
    (MAX(CASE WHEN variant='treatment' THEN cvr END) OVER () -
     MAX(CASE WHEN variant='control'   THEN cvr END) OVER ()) /
    SQRT(p.p_pool * (1-p.p_pool) *
      (1.0/MAX(CASE WHEN variant='treatment' THEN n END) OVER () +
       1.0/MAX(CASE WHEN variant='control'   THEN n END) OVER ())), 4),
  CASE WHEN ABS(
    (MAX(CASE WHEN variant='treatment' THEN cvr END) OVER () -
     MAX(CASE WHEN variant='control'   THEN cvr END) OVER ()) /
    SQRT(p.p_pool * (1-p.p_pool) *
      (1.0/MAX(CASE WHEN variant='treatment' THEN n END) OVER () +
       1.0/MAX(CASE WHEN variant='control'   THEN n END) OVER ()))) > 1.96
  THEN 'Significant (95%)' ELSE 'Not Significant' END
FROM overall o CROSS JOIN pooled p;
```Day 13 saved ⬆️

---

**Week 2 almost done! Here's where you stand:**

| Day | Topic |
|---|---|
| 8 | Data Cleaning |
| 9 | Product Analytics |
| 10 | Advanced Window Functions |
| 11 | Recursive CTEs |
| 12 | ML Feature Engineering |
| 13 | A/B Testing & Stats ✅ |

---

**Day 14 — Query Optimization & Performance** is next. Index usage, query plans, avoiding full table scans, partitioning — what FAANG engineers ask when they want to know if you can write SQL that actually works at scale (billions of rows).

Type **"Day 14"** when ready 👇
