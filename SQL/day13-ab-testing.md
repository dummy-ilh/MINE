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

```sql
-- Sample sizes per variant
SELECT variant,
  COUNT(DISTINCT user_id) AS users,
  MIN(assigned_date)      AS start_date,
  MAX(assigned_date)      AS end_date
FROM ab_assignments
GROUP BY variant;

-- Sample Ratio Mismatch (SRM) check
WITH counts AS (
  SELECT variant, COUNT(*) AS n FROM ab_assignments GROUP BY variant
),
total AS (SELECT SUM(n) AS total FROM counts)
SELECT variant, n,
  ROUND(n * 100.0 / total.total, 2) AS actual_pct,
  50.0                              AS expected_pct,
  ABS(n * 100.0 / total.total - 50) AS deviation_pct
FROM counts CROSS JOIN total;
-- deviation > 1% → SRM → experiment invalid

-- Pre-experiment balance check
SELECT a.variant,
  AVG(prior.amount)         AS avg_prior_spend,
  COUNT(DISTINCT a.user_id) AS users
FROM ab_assignments a
LEFT JOIN orders prior
  ON  prior.user_id    = a.user_id
  AND prior.order_date < a.assigned_date
GROUP BY a.variant;
```

---

## 2. Core Metrics Computation

```sql
WITH experiment_orders AS (
  SELECT a.user_id, a.variant,
    COUNT(o.order_id)   AS orders,
    SUM(o.amount)       AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  LEFT JOIN orders o
    ON  o.user_id    = a.user_id
    AND o.order_date BETWEEN a.assigned_date
                         AND a.assigned_date + INTERVAL 14 DAY
  GROUP BY a.user_id, a.variant
)
SELECT variant,
  COUNT(*)                                  AS total_users,
  SUM(converted)                            AS conversions,
  ROUND(SUM(converted) * 100.0 / COUNT(*), 4) AS conversion_rate,
  ROUND(SUM(revenue) / COUNT(*), 4)         AS revenue_per_user,
  ROUND(AVG(revenue), 4)                    AS avg_revenue,
  ROUND(STDDEV(revenue), 4)                 AS stddev_revenue
FROM experiment_orders
GROUP BY variant;
```

---

## 3. Lift & Relative Uplift

```sql
WITH metrics AS (
  SELECT variant,
    SUM(converted) * 1.0 / COUNT(*) AS cvr,
    AVG(revenue) AS avg_rev
  FROM experiment_orders GROUP BY variant
),
ctrl AS (SELECT cvr AS c_cvr, avg_rev AS c_rev FROM metrics WHERE variant='control'),
trt  AS (SELECT cvr AS t_cvr, avg_rev AS t_rev FROM metrics WHERE variant='treatment')
SELECT
  t.t_cvr - c.c_cvr                              AS absolute_lift_cvr,
  ROUND((t.t_cvr - c.c_cvr) * 100.0 /
        NULLIF(c.c_cvr, 0), 4)                   AS relative_lift_cvr_pct,
  t.t_rev - c.c_rev                              AS absolute_lift_rev,
  ROUND((t.t_rev - c.c_rev) * 100.0 /
        NULLIF(c.c_rev, 0), 4)                   AS relative_lift_rev_pct
FROM trt t CROSS JOIN ctrl c;
```

---

## 4. Z-Test for Proportions (CVR)

```sql
WITH stats AS (
  SELECT variant,
    COUNT(*) AS n, SUM(converted) AS x,
    SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders GROUP BY variant
),
pooled AS (SELECT SUM(x) * 1.0 / SUM(n) AS p_pool FROM stats),
z_calc AS (
  SELECT
    MAX(CASE WHEN variant='treatment' THEN p END) AS p_t,
    MAX(CASE WHEN variant='control'   THEN p END) AS p_c,
    MAX(CASE WHEN variant='treatment' THEN n END) AS n_t,
    MAX(CASE WHEN variant='control'   THEN n END) AS n_c,
    po.p_pool
  FROM stats CROSS JOIN pooled po GROUP BY po.p_pool
)
SELECT
  ROUND((p_t - p_c) /
    SQRT(p_pool*(1-p_pool)*(1.0/n_t+1.0/n_c)), 4) AS z_score,
  CASE
    WHEN ABS((p_t - p_c) /
      SQRT(p_pool*(1-p_pool)*(1.0/n_t+1.0/n_c))) > 1.96
    THEN 'Significant (95%)' ELSE 'Not Significant'
  END AS result
FROM z_calc;
```

**Z-score thresholds:**
- |z| > 1.645 → 90% confidence
- |z| > 1.960 → 95% confidence
- |z| > 2.576 → 99% confidence

---

## 5. Confidence Intervals

```sql
-- 95% CI for conversion rate
WITH stats AS (
  SELECT variant, COUNT(*) AS n,
    SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders GROUP BY variant
)
SELECT variant, ROUND(p, 4) AS cvr,
  ROUND(p - 1.96 * SQRT(p*(1-p)/n), 4) AS ci_lower,
  ROUND(p + 1.96 * SQRT(p*(1-p)/n), 4) AS ci_upper
FROM stats;

-- 95% CI for mean revenue
WITH stats AS (
  SELECT variant, COUNT(*) AS n,
    AVG(revenue) AS mean_r, STDDEV(revenue) AS std_r
  FROM experiment_orders GROUP BY variant
)
SELECT variant,
  ROUND(mean_r, 2)                               AS mean_revenue,
  ROUND(mean_r - 1.96 * std_r / SQRT(n), 2)     AS ci_lower,
  ROUND(mean_r + 1.96 * std_r / SQRT(n), 2)     AS ci_upper
FROM stats;
```

---

## 6. T-Test for Continuous Metrics

```sql
WITH stats AS (
  SELECT variant, COUNT(*) AS n,
    AVG(revenue) AS mean_r,
    VARIANCE(revenue) AS var_r
  FROM experiment_orders GROUP BY variant
),
t_calc AS (
  SELECT
    MAX(CASE WHEN variant='treatment' THEN mean_r END) AS mean_t,
    MAX(CASE WHEN variant='control'   THEN mean_r END) AS mean_c,
    MAX(CASE WHEN variant='treatment' THEN var_r  END) AS var_t,
    MAX(CASE WHEN variant='control'   THEN var_r  END) AS var_c,
    MAX(CASE WHEN variant='treatment' THEN n      END) AS n_t,
    MAX(CASE WHEN variant='control'   THEN n      END) AS n_c
  FROM stats
)
SELECT
  ROUND((mean_t - mean_c) /
    SQRT(var_t/n_t + var_c/n_c), 4) AS t_statistic,
  CASE
    WHEN ABS((mean_t - mean_c) /
      SQRT(var_t/n_t + var_c/n_c)) > 1.96
    THEN 'Significant (95%)' ELSE 'Not Significant'
  END AS result
FROM t_calc;
```

---

## 7. Novelty Effect Detection

```sql
WITH daily AS (
  SELECT a.variant, DATE(o.order_date) AS day,
    COUNT(DISTINCT a.user_id) AS users,
    COUNT(DISTINCT o.user_id) AS converters
  FROM ab_assignments a
  LEFT JOIN orders o ON o.user_id = a.user_id
    AND DATE(o.order_date) = DATE(a.assigned_date)
  GROUP BY a.variant, DATE(o.order_date)
)
SELECT variant, day,
  ROUND(converters * 100.0 / users, 2) AS daily_cvr,
  AVG(converters * 1.0 / users) OVER (
    PARTITION BY variant ORDER BY day
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS rolling_7d_cvr
FROM daily ORDER BY variant, day;
```

---

## 8. Segmented Analysis

```sql
SELECT country, plan_type, variant,
  COUNT(*)                                     AS users,
  ROUND(SUM(converted) * 100.0 / COUNT(*), 2) AS cvr,
  ROUND(AVG(revenue), 2)                       AS avg_revenue
FROM experiment_orders
GROUP BY country, plan_type, variant
ORDER BY country, plan_type, variant;
```

---

## 9. Power Analysis

```sql
-- Minimum sample size per variant
WITH params AS (
  SELECT
    0.10  AS baseline_cvr,
    0.005 AS min_detectable_effect,
    1.96  AS z_alpha,
    0.84  AS z_beta
)
SELECT
  CEIL(
    2 * POW(z_alpha + z_beta, 2) *
    baseline_cvr * (1 - baseline_cvr) /
    POW(min_detectable_effect, 2)
  ) AS required_n_per_variant
FROM params;
```

---

## Practice Questions

### Q1 — Easy ✅
```sql
WITH experiment_orders AS (
  SELECT a.user_id, a.variant,
    COALESCE(SUM(o.amount), 0) AS revenue,
    MAX(CASE WHEN o.order_id IS NOT NULL THEN 1 ELSE 0 END) AS converted
  FROM ab_assignments a
  LEFT JOIN orders o
    ON  o.user_id    = a.user_id
    AND o.order_date >= a.assigned_date
  GROUP BY a.user_id, a.variant
)
SELECT variant,
  COUNT(*)                                     AS total_users,
  SUM(converted)                               AS conversions,
  ROUND(SUM(converted) * 100.0 / COUNT(*), 4) AS conversion_rate,
  ROUND(AVG(revenue), 4)                       AS avg_revenue_per_user,
  ROUND(STDDEV(revenue), 4)                    AS stddev_revenue
FROM experiment_orders GROUP BY variant;
```

### Q2 — Medium ✅
```sql
WITH stats AS (
  SELECT variant, COUNT(*) AS n,
    SUM(converted) AS x,
    SUM(converted) * 1.0 / COUNT(*) AS p
  FROM experiment_orders GROUP BY variant
),
pooled AS (SELECT SUM(x) * 1.0 / SUM(n) AS p_pool FROM stats),
z_calc AS (
  SELECT
    MAX(CASE WHEN variant='treatment' THEN p END) AS p_t,
    MAX(CASE WHEN variant='control'   THEN p END) AS p_c,
    MAX(CASE WHEN variant='treatment' THEN n END) AS n_t,
    MAX(CASE WHEN variant='control'   THEN n END) AS n_c,
    po.p_pool
  FROM stats CROSS JOIN pooled po GROUP BY po.p_pool
)
SELECT
  ROUND(p_t - p_c, 4)                              AS absolute_lift,
  ROUND((p_t-p_c)*100.0/NULLIF(p_c,0), 2)         AS relative_lift_pct,
  ROUND((p_t-p_c)/SQRT(p_pool*(1-p_pool)*
        (1.0/n_t+1.0/n_c)), 4)                     AS z_score,
  ROUND(p_t - 1.96*SQRT(p_t*(1-p_t)/n_t), 4)      AS trt_ci_lower,
  ROUND(p_t + 1.96*SQRT(p_t*(1-p_t)/n_t), 4)      AS trt_ci_upper,
  CASE WHEN ABS((p_t-p_c)/SQRT(p_pool*(1-p_pool)*
        (1.0/n_t+1.0/n_c))) > 1.96
       THEN 'Significant (95%)' ELSE 'Not Significant'
  END AS result
FROM z_calc;
```

### Q3 — Hard ✅
Full A/B report with SRM + metrics + significance + segmentation (see full query in Day 13 notes above).

---

## Key Takeaways

- **Always run SRM check first** — invalid experiment if split is off by >1%
- **Pre-experiment balance** — both variants should have similar prior behavior
- **Z-test** → for conversion rates (proportions)
- **T-test** → for continuous metrics (revenue, session duration)
- **|z| > 1.96** → statistically significant at 95% confidence
- **Confidence interval** → does it exclude 0? If yes → significant
- **Novelty effect** → plot daily CVR; if treatment fades → novelty not real lift
- **Segmented analysis** → always check for heterogeneous treatment effects
- **Power analysis** → run before the experiment, not after
- **Point-in-time window** → always join orders within experiment window only

---

*Day 13 complete — 17 days to go 🚀*
