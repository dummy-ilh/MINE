# SQL for ML Interviews — Applied & Special Topics

> **Module 5** — Real-world ML data patterns. Every topic shows the exact query shape interviewers expect, with ML-relevant tables, full input/output, and pitfalls. No other source needed.

---

## Table of Contents

30. [Funnel Analysis](#30-funnel-analysis)
31. [Retention Cohorts](#31-retention-cohorts)
32. [Sessionization](#32-sessionization)
33. [Time Series and Rolling Features](#33-time-series-and-rolling-features)
34. [A/B Testing and Experimentation SQL](#34-ab-testing-and-experimentation-sql)
35. [Feature Engineering in SQL for ML](#35-feature-engineering-in-sql-for-ml)
36. [Data Quality and Anomaly Detection](#36-data-quality-and-anomaly-detection)
37. [Metric Definitions: DAU/WAU/MAU, Stickiness](#37-metric-definitions-dauwaumau-stickiness)
38. [Market Basket / Co-occurrence Analysis](#38-market-basket--co-occurrence-analysis)
39. [Attribution Modeling — First-touch / Last-touch](#39-attribution-modeling--first-touch--last-touch)
40. [Sampling in SQL — TABLESAMPLE, Stratified Sampling](#40-sampling-in-sql--tablesample-stratified-sampling)
41. [Fuzzy / Near-duplicate Matching](#41-fuzzy--near-duplicate-matching)

---

## 30. Funnel Analysis

### Concept

A funnel tracks users through an **ordered sequence of steps** and measures drop-off at each stage. This is one of the most common SQL interview questions at product and ML companies.

The core challenge: a user must complete steps **in order** — you can't credit step 3 if step 2 never happened. Two approaches:

| Approach | When to use |
|----------|-------------|
| `COUNT(DISTINCT user_id)` per step | When order doesn't strictly matter (e.g., any signup before purchase) |
| `MAX(CASE WHEN step THEN ts END)` with ordering check | When strict ordering is required (step 2 must happen after step 1) |

---

### Sample Table: `user_events`

| event_id | user_id | event_type    | event_ts            |
|----------|---------|---------------|---------------------|
| 1        | u1      | page_view     | 2024-01-01 10:00:00 |
| 2        | u1      | signup        | 2024-01-01 10:05:00 |
| 3        | u1      | add_to_cart   | 2024-01-01 10:10:00 |
| 4        | u1      | purchase      | 2024-01-01 10:15:00 |
| 5        | u2      | page_view     | 2024-01-01 11:00:00 |
| 6        | u2      | signup        | 2024-01-01 11:10:00 |
| 7        | u2      | add_to_cart   | 2024-01-01 11:20:00 |
| 8        | u3      | page_view     | 2024-01-01 12:00:00 |
| 9        | u3      | signup        | 2024-01-01 12:05:00 |
| 10       | u4      | page_view     | 2024-01-01 13:00:00 |

---

### Variant 1 — Simple funnel (count per step)

```sql
SELECT
    'page_view'   AS step, 1 AS step_order,
    COUNT(DISTINCT user_id) AS users
FROM user_events WHERE event_type = 'page_view'
UNION ALL
SELECT 'signup',      2, COUNT(DISTINCT user_id) FROM user_events WHERE event_type = 'signup'
UNION ALL
SELECT 'add_to_cart', 3, COUNT(DISTINCT user_id) FROM user_events WHERE event_type = 'add_to_cart'
UNION ALL
SELECT 'purchase',    4, COUNT(DISTINCT user_id) FROM user_events WHERE event_type = 'purchase'
ORDER BY step_order;
```

**Output:**

| step         | step_order | users |
|--------------|------------|-------|
| page_view    | 1          | 4     |
| signup       | 2          | 3     |
| add_to_cart  | 3          | 2     |
| purchase     | 4          | 1     |

---

### Variant 2 — Funnel with conversion rates and drop-off

```sql
WITH steps AS (
    SELECT
        COUNT(DISTINCT CASE WHEN event_type = 'page_view'   THEN user_id END) AS step1,
        COUNT(DISTINCT CASE WHEN event_type = 'signup'      THEN user_id END) AS step2,
        COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN user_id END) AS step3,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase'    THEN user_id END) AS step4
    FROM user_events
)
SELECT
    'page_view'   AS step, step1 AS users,
    100.0                  AS pct_of_top,
    NULL                   AS step_conversion
FROM steps
UNION ALL
SELECT 'signup',      step2, ROUND(step2*100.0/NULLIF(step1,0),1), ROUND(step2*100.0/NULLIF(step1,0),1) FROM steps
UNION ALL
SELECT 'add_to_cart', step3, ROUND(step3*100.0/NULLIF(step1,0),1), ROUND(step3*100.0/NULLIF(step2,0),1) FROM steps
UNION ALL
SELECT 'purchase',    step4, ROUND(step4*100.0/NULLIF(step1,0),1), ROUND(step4*100.0/NULLIF(step3,0),1) FROM steps;
```

**Output:**

| step         | users | pct_of_top | step_conversion |
|--------------|-------|------------|-----------------|
| page_view    | 4     | 100.0      | NULL            |
| signup       | 3     | 75.0       | 75.0            |
| add_to_cart  | 2     | 50.0       | 66.7            |
| purchase     | 1     | 25.0       | 50.0            |

---

### Variant 3 — Ordered funnel (step 2 must happen AFTER step 1)

```sql
WITH user_step_times AS (
    SELECT
        user_id,
        MIN(CASE WHEN event_type = 'page_view'   THEN event_ts END) AS t1,
        MIN(CASE WHEN event_type = 'signup'      THEN event_ts END) AS t2,
        MIN(CASE WHEN event_type = 'add_to_cart' THEN event_ts END) AS t3,
        MIN(CASE WHEN event_type = 'purchase'    THEN event_ts END) AS t4
    FROM user_events
    GROUP BY user_id
)
SELECT
    COUNT(*)                                            AS reached_step1,
    COUNT(CASE WHEN t2 > t1 THEN 1 END)                AS reached_step2,
    COUNT(CASE WHEN t3 > t2 AND t2 > t1 THEN 1 END)   AS reached_step3,
    COUNT(CASE WHEN t4 > t3 AND t3 > t2 THEN 1 END)   AS reached_step4
FROM user_step_times
WHERE t1 IS NOT NULL;
```

**Output:**

| reached_step1 | reached_step2 | reached_step3 | reached_step4 |
|---------------|---------------|---------------|---------------|
| 4             | 3             | 2             | 1             |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Not enforcing order | A user who purchased before signing up shouldn't count | Use ordered variant with `t2 > t1` checks |
| Counting events not users | `COUNT(*)` counts clicks, not unique users | Always `COUNT(DISTINCT user_id)` in funnels |
| Funnel window too wide | User signs up in Jan, purchases in Dec — misleading conversion | Add `WHERE event_ts BETWEEN start AND start + INTERVAL '7 days'` |
| Missing steps return NULL not 0 | CASE WHEN returns NULL for no match | `COALESCE(COUNT(...), 0)` |

---

## 31. Retention Cohorts

### Concept

**Retention** measures what fraction of users from a starting cohort return on subsequent days/weeks. The output is a **cohort retention matrix** — rows are cohorts (signup week), columns are days/weeks since signup.

**Day-N retention** = users who were active on exactly day N / users in the cohort.
**Week-N retention** = users active in week N / cohort size.

---

### Sample Tables

**`signups`**

| user_id | signup_date |
|---------|-------------|
| u1      | 2024-01-01  |
| u2      | 2024-01-01  |
| u3      | 2024-01-08  |
| u4      | 2024-01-08  |
| u5      | 2024-01-08  |

**`activity`**

| user_id | activity_date |
|---------|---------------|
| u1      | 2024-01-01    |
| u1      | 2024-01-08    |
| u1      | 2024-01-15    |
| u2      | 2024-01-01    |
| u2      | 2024-01-08    |
| u3      | 2024-01-08    |
| u3      | 2024-01-15    |
| u4      | 2024-01-08    |
| u5      | 2024-01-08    |

---

### Variant 1 — Day-N retention (weeks as columns)

```sql
WITH cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('week', signup_date)::DATE AS cohort_week
    FROM signups
),
activity_with_cohort AS (
    SELECT
        c.user_id,
        c.cohort_week,
        a.activity_date,
        -- weeks since signup
        ((a.activity_date - c.cohort_week) / 7)::INT AS weeks_since_signup
    FROM cohorts c
    JOIN activity a ON c.user_id = a.user_id
    WHERE a.activity_date >= c.cohort_week
),
cohort_sizes AS (
    SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size
    FROM cohorts
    GROUP BY cohort_week
)
SELECT
    a.cohort_week,
    cs.cohort_size,
    COUNT(DISTINCT CASE WHEN weeks_since_signup = 0 THEN a.user_id END) AS week_0,
    COUNT(DISTINCT CASE WHEN weeks_since_signup = 1 THEN a.user_id END) AS week_1,
    COUNT(DISTINCT CASE WHEN weeks_since_signup = 2 THEN a.user_id END) AS week_2,
    -- As percentages
    ROUND(COUNT(DISTINCT CASE WHEN weeks_since_signup = 1 THEN a.user_id END)
          * 100.0 / cs.cohort_size, 1)                                  AS week_1_pct,
    ROUND(COUNT(DISTINCT CASE WHEN weeks_since_signup = 2 THEN a.user_id END)
          * 100.0 / cs.cohort_size, 1)                                  AS week_2_pct
FROM activity_with_cohort a
JOIN cohort_sizes cs ON a.cohort_week = cs.cohort_week
GROUP BY a.cohort_week, cs.cohort_size
ORDER BY a.cohort_week;
```

**Output:**

| cohort_week | cohort_size | week_0 | week_1 | week_2 | week_1_pct | week_2_pct |
|-------------|-------------|--------|--------|--------|------------|------------|
| 2024-01-01  | 2           | 2      | 2      | 1      | 100.0      | 50.0       |
| 2024-01-08  | 3           | 3      | 1      | 1      | 33.3       | 33.3       |

---

### Variant 2 — Long-format retention (easier to chart)

```sql
WITH cohorts AS (
    SELECT user_id, DATE_TRUNC('week', signup_date)::DATE AS cohort_week
    FROM signups
),
retention AS (
    SELECT
        c.cohort_week,
        ((a.activity_date - c.cohort_week) / 7)::INT AS week_num,
        COUNT(DISTINCT a.user_id)                     AS active_users
    FROM cohorts c
    JOIN activity a ON c.user_id = a.user_id
    WHERE a.activity_date >= c.cohort_week
    GROUP BY c.cohort_week, week_num
),
sizes AS (
    SELECT DATE_TRUNC('week', signup_date)::DATE AS cohort_week,
           COUNT(*) AS cohort_size
    FROM signups GROUP BY 1
)
SELECT
    r.cohort_week,
    s.cohort_size,
    r.week_num,
    r.active_users,
    ROUND(r.active_users * 100.0 / s.cohort_size, 1) AS retention_pct
FROM retention r
JOIN sizes s ON r.cohort_week = s.cohort_week
ORDER BY r.cohort_week, r.week_num;
```

**Output:**

| cohort_week | cohort_size | week_num | active_users | retention_pct |
|-------------|-------------|----------|--------------|---------------|
| 2024-01-01  | 2           | 0        | 2            | 100.0         |
| 2024-01-01  | 2           | 1        | 2            | 100.0         |
| 2024-01-01  | 2           | 2        | 1            | 50.0          |
| 2024-01-08  | 3           | 0        | 3            | 100.0         |
| 2024-01-08  | 3           | 1        | 1            | 33.3          |
| 2024-01-08  | 3           | 2        | 1            | 33.3          |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using calendar date instead of days-since-signup | Jan 8 means different things for different cohorts | Always compute `activity_date - cohort_date` as the X axis |
| Counting events not users | Double-counting users who were active multiple times in week N | `COUNT(DISTINCT user_id)` within each week bucket |
| Week 0 retention isn't always 100% | If signup event != first activity, cohort_size > week_0 | Define cohort and activity consistently (both from same event source) |
| Cohort week vs cohort month boundary | `DATE_TRUNC('week', ...)` starts on Monday in PostgreSQL | Verify first day of week; add `SET datestyle` if needed |

---

## 32. Sessionization

### Concept

Sessionization assigns a **session ID** to groups of events where the gap between consecutive events is below a threshold (typically 30 minutes). Covered in detail in Module 3 (Topic 36). Here we focus on the **ML-specific extensions**: per-session feature extraction for training data.

---

### Sample Table: `user_events` (with session IDs already assigned)

| event_id | user_id | session_id | event_ts            | event_type | page        |
|----------|---------|------------|---------------------|------------|-------------|
| 1        | u1      | 1          | 2024-01-01 10:00:00 | view       | home        |
| 2        | u1      | 1          | 2024-01-01 10:03:00 | view       | product_A   |
| 3        | u1      | 1          | 2024-01-01 10:07:00 | click      | product_A   |
| 4        | u1      | 1          | 2024-01-01 10:12:00 | purchase   | checkout    |
| 5        | u1      | 2          | 2024-01-01 14:00:00 | view       | home        |
| 6        | u1      | 2          | 2024-01-01 14:05:00 | view       | product_B   |
| 7        | u2      | 1          | 2024-01-01 09:00:00 | view       | home        |
| 8        | u2      | 1          | 2024-01-01 09:04:00 | click      | product_C   |

---

### Variant 1 — Per-session feature extraction (ML training features)

```sql
SELECT
    user_id,
    session_id,
    -- Time features
    MIN(event_ts)                                                    AS session_start,
    MAX(event_ts)                                                    AS session_end,
    EXTRACT(EPOCH FROM (MAX(event_ts) - MIN(event_ts)))             AS duration_seconds,
    EXTRACT(HOUR FROM MIN(event_ts))                                AS start_hour,
    -- Volume features
    COUNT(*)                                                         AS total_events,
    COUNT(DISTINCT page)                                             AS unique_pages,
    COUNT(CASE WHEN event_type = 'view'     THEN 1 END)             AS views,
    COUNT(CASE WHEN event_type = 'click'    THEN 1 END)             AS clicks,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END)             AS purchases,
    -- Engagement ratios
    ROUND(COUNT(CASE WHEN event_type = 'click' THEN 1 END)
          * 1.0 / NULLIF(COUNT(CASE WHEN event_type = 'view' THEN 1 END), 0), 3)
                                                                     AS click_through_rate,
    -- Entry and exit pages
    FIRST_VALUE(page) OVER (
        PARTITION BY user_id, session_id ORDER BY event_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                                                AS entry_page,
    LAST_VALUE(page) OVER (
        PARTITION BY user_id, session_id ORDER BY event_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                                                AS exit_page,
    -- Label for training
    MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END)        AS converted
FROM user_events
GROUP BY user_id, session_id
ORDER BY user_id, session_id;
```

**Output:**

| user_id | session_id | duration_s | total_events | clicks | purchases | ctr   | entry_page | exit_page | converted |
|---------|------------|------------|--------------|--------|-----------|-------|------------|-----------|-----------|
| u1      | 1          | 720        | 4            | 1      | 1         | 0.500 | home       | checkout  | 1         |
| u1      | 2          | 300        | 2            | 0      | 0         | 0.000 | home       | product_B | 0         |
| u2      | 1          | 240        | 2            | 1      | 0         | 1.000 | home       | product_C | 0         |

---

### Variant 2 — Inter-session features (recency, frequency for next-session prediction)

```sql
WITH session_summary AS (
    SELECT
        user_id,
        session_id,
        MIN(event_ts) AS session_start,
        MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS converted
    FROM user_events
    GROUP BY user_id, session_id
)
SELECT
    user_id,
    session_id,
    session_start,
    converted,
    LAG(session_start) OVER (PARTITION BY user_id ORDER BY session_start) AS prev_session_start,
    EXTRACT(EPOCH FROM (
        session_start -
        LAG(session_start) OVER (PARTITION BY user_id ORDER BY session_start)
    )) / 3600                                                              AS hours_since_last_session,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY session_start)       AS session_number,
    SUM(converted) OVER (
        PARTITION BY user_id ORDER BY session_start
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )                                                                      AS prior_conversions
FROM session_summary
ORDER BY user_id, session_start;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| FIRST_VALUE / LAST_VALUE frame issues | Default frame returns wrong values for entry/exit pages | Always use `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` |
| Sessions spanning midnight | Day-of-week feature for session is ambiguous | Use session_start timestamp, not date |
| Bot sessions | Extremely short duration, very high event counts | Filter: `duration_seconds > 5 AND total_events < 500` |

---

## 33. Time Series and Rolling Features

### Concept

For ML models, raw event data must be aggregated into **fixed-width feature windows** per entity (user/model/item) at a specific point in time. This is the core of **feature engineering for time-series ML**.

Key patterns:
- **Point-in-time correctness** — features must only use data available *before* the label timestamp (no data leakage)
- **Multiple window sizes** — compute 1d, 7d, 30d features simultaneously
- **Date spine** — generate a row for every entity × date combination before joining events

---

### Sample Table: `user_purchases`

| user_id | purchase_date | amount |
|---------|---------------|--------|
| u1      | 2024-01-01    | 50     |
| u1      | 2024-01-03    | 30     |
| u1      | 2024-01-10    | 100    |
| u2      | 2024-01-05    | 80     |
| u2      | 2024-01-12    | 60     |

---

### Variant 1 — Multi-window rolling features (point-in-time safe)

```sql
-- For each user, compute features as of 2024-01-15
-- (everything before this date is known; this date is the prediction point)

WITH prediction_date AS (SELECT DATE '2024-01-15' AS pred_dt)
SELECT
    u.user_id,
    pd.pred_dt,
    -- 7-day window
    COUNT(CASE WHEN p.purchase_date >= pd.pred_dt - 7  AND p.purchase_date < pd.pred_dt
               THEN 1 END)                                           AS purchases_7d,
    COALESCE(SUM(CASE WHEN p.purchase_date >= pd.pred_dt - 7
                       AND p.purchase_date < pd.pred_dt
                      THEN p.amount END), 0)                         AS spend_7d,
    -- 30-day window
    COUNT(CASE WHEN p.purchase_date >= pd.pred_dt - 30 AND p.purchase_date < pd.pred_dt
               THEN 1 END)                                           AS purchases_30d,
    COALESCE(SUM(CASE WHEN p.purchase_date >= pd.pred_dt - 30
                       AND p.purchase_date < pd.pred_dt
                      THEN p.amount END), 0)                         AS spend_30d,
    -- All-time recency
    pd.pred_dt - MAX(p.purchase_date)                                AS days_since_last_purchase,
    COUNT(p.purchase_date)                                           AS total_purchases_ever
FROM (SELECT DISTINCT user_id FROM user_purchases) u
CROSS JOIN prediction_date pd
LEFT JOIN user_purchases p
    ON u.user_id = p.user_id
    AND p.purchase_date < pd.pred_dt       -- point-in-time: only past data
GROUP BY u.user_id, pd.pred_dt
ORDER BY u.user_id;
```

**Output:**

| user_id | pred_dt    | purchases_7d | spend_7d | purchases_30d | spend_30d | days_since_last_purchase | total_purchases |
|---------|------------|--------------|----------|---------------|-----------|--------------------------|-----------------|
| u1      | 2024-01-15 | 1            | 100      | 3             | 180       | 5                        | 3               |
| u2      | 2024-01-15 | 1            | 60       | 2             | 140       | 3                        | 2               |

---

### Variant 2 — Date spine: one row per user per day

```sql
WITH date_spine AS (
    SELECT GENERATE_SERIES(
        '2024-01-01'::DATE,
        '2024-01-15'::DATE,
        '1 day'::INTERVAL
    )::DATE AS dt
),
users AS (SELECT DISTINCT user_id FROM user_purchases),
spine AS (
    SELECT u.user_id, d.dt
    FROM users u CROSS JOIN date_spine d
)
SELECT
    s.user_id,
    s.dt,
    COALESCE(SUM(p.amount), 0)   AS daily_spend,
    COALESCE(COUNT(p.purchase_date), 0) AS daily_purchases,
    -- 7-day rolling spend up to this day
    SUM(COALESCE(p.amount, 0)) OVER (
        PARTITION BY s.user_id
        ORDER BY s.dt
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    )                             AS rolling_7d_spend
FROM spine s
LEFT JOIN user_purchases p
    ON s.user_id = p.user_id AND p.purchase_date = s.dt
GROUP BY s.user_id, s.dt
ORDER BY s.user_id, s.dt;
```

---

### Variant 3 — Lag features for sequential models (RNN/LSTM feature prep)

```sql
WITH daily AS (
    SELECT
        user_id,
        purchase_date,
        SUM(amount) AS daily_spend
    FROM user_purchases
    GROUP BY user_id, purchase_date
)
SELECT
    user_id,
    purchase_date,
    daily_spend,
    LAG(daily_spend, 1) OVER (PARTITION BY user_id ORDER BY purchase_date) AS spend_t_minus_1,
    LAG(daily_spend, 2) OVER (PARTITION BY user_id ORDER BY purchase_date) AS spend_t_minus_2,
    LAG(daily_spend, 3) OVER (PARTITION BY user_id ORDER BY purchase_date) AS spend_t_minus_3,
    AVG(daily_spend) OVER (
        PARTITION BY user_id ORDER BY purchase_date
        ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
    )                                                                        AS spend_ma7
FROM daily
ORDER BY user_id, purchase_date;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Data leakage | Using `purchase_date <= pred_dt` instead of `< pred_dt` includes same-day events | Always use strict `< pred_dt` for point-in-time correctness |
| Missing users with no events | LEFT JOIN to a user spine or users table | Use a separate user list, CROSS JOIN with dates, then LEFT JOIN events |
| GENERATE_SERIES is PostgreSQL only | BigQuery: `UNNEST(GENERATE_DATE_ARRAY(...))` | Know your DB syntax |
| Window function on pre-aggregated data | Rolling sum over daily aggregates isn't same as rolling sum over raw events if multiple events per day | Aggregate to daily first, then apply window |

---

## 34. A/B Testing and Experimentation SQL

### Concept

A/B testing SQL covers: assignment validation, metric computation, significance testing setup, and guardrail metric checks. Interviewers test whether you can write the full pipeline, not just the average.

**Key checks before computing results:**
1. Sample Ratio Mismatch (SRM) — are group sizes what we expect?
2. Pre-experiment equivalence — were groups equal before the test?
3. Novelty effects — do early results differ from late results?

---

### Sample Tables

**`assignments`**

| user_id | variant   | assigned_at        |
|---------|-----------|--------------------|
| u1      | control   | 2024-01-01         |
| u2      | treatment | 2024-01-01         |
| u3      | control   | 2024-01-01         |
| u4      | treatment | 2024-01-01         |
| u5      | control   | 2024-01-01         |
| u6      | treatment | 2024-01-01         |

**`conversions`**

| user_id | converted | revenue | event_date |
|---------|-----------|---------|------------|
| u1      | 0         | 0       | 2024-01-05 |
| u2      | 1         | 49.99   | 2024-01-05 |
| u3      | 1         | 29.99   | 2024-01-06 |
| u4      | 1         | 59.99   | 2024-01-06 |
| u5      | 0         | 0       | 2024-01-07 |
| u6      | 0         | 0       | 2024-01-07 |

---

### Variant 1 — Core metric computation per variant

```sql
SELECT
    a.variant,
    COUNT(DISTINCT a.user_id)                        AS users,
    SUM(c.converted)                                 AS conversions,
    ROUND(AVG(c.converted), 4)                       AS conversion_rate,
    ROUND(AVG(c.revenue), 2)                         AS avg_revenue_per_user,
    ROUND(SUM(c.revenue), 2)                         AS total_revenue,
    -- Standard deviation for significance test
    ROUND(STDDEV(c.converted), 4)                    AS conversion_stddev,
    ROUND(STDDEV(c.revenue),   2)                    AS revenue_stddev
FROM assignments a
LEFT JOIN conversions c ON a.user_id = c.user_id
GROUP BY a.variant
ORDER BY a.variant;
```

**Output:**

| variant   | users | conversions | conversion_rate | avg_revenue | total_revenue | conversion_stddev |
|-----------|-------|-------------|-----------------|-------------|---------------|-------------------|
| control   | 3     | 1           | 0.3333          | 9.99        | 29.99         | 0.5774            |
| treatment | 3     | 2           | 0.6667          | 36.66       | 109.98        | 0.5774            |

---

### Variant 2 — Sample Ratio Mismatch (SRM) check

```sql
-- Expect 50/50 split; detect if assignment is broken
WITH counts AS (
    SELECT
        variant,
        COUNT(*) AS n
    FROM assignments
    GROUP BY variant
),
total AS (SELECT SUM(n) AS total_n FROM counts)
SELECT
    c.variant,
    c.n                                                         AS observed,
    t.total_n / 2                                               AS expected,
    ROUND(c.n * 100.0 / t.total_n, 1)                         AS observed_pct,
    ABS(c.n - t.total_n / 2)                                   AS abs_deviation,
    -- Chi-square contribution: (O-E)^2 / E
    ROUND(POWER(c.n - t.total_n / 2.0, 2) / (t.total_n / 2.0), 4) AS chi_sq_contribution
FROM counts c CROSS JOIN total t;
```

**Output:**

| variant   | observed | expected | observed_pct | abs_deviation | chi_sq_contribution |
|-----------|----------|----------|--------------|---------------|---------------------|
| control   | 3        | 3        | 50.0         | 0             | 0.0000              |
| treatment | 3        | 3        | 50.0         | 0             | 0.0000              |

> Sum of chi_sq_contribution > 3.84 → SRM detected at p < 0.05 (1 degree of freedom). Stop the experiment.

---

### Variant 3 — Novelty effect check (early vs late results)

```sql
WITH weekly AS (
    SELECT
        a.variant,
        DATE_TRUNC('week', c.event_date)::DATE AS week,
        COUNT(DISTINCT a.user_id)              AS users,
        ROUND(AVG(c.converted), 4)             AS conv_rate
    FROM assignments a
    JOIN conversions c ON a.user_id = c.user_id
    GROUP BY a.variant, DATE_TRUNC('week', c.event_date)
)
SELECT
    week,
    MAX(CASE WHEN variant = 'control'   THEN conv_rate END) AS control_rate,
    MAX(CASE WHEN variant = 'treatment' THEN conv_rate END) AS treatment_rate,
    MAX(CASE WHEN variant = 'treatment' THEN conv_rate END) -
    MAX(CASE WHEN variant = 'control'   THEN conv_rate END) AS lift
FROM weekly
GROUP BY week
ORDER BY week;
```

---

### Variant 4 — CUPED: pre-experiment covariate adjustment

```sql
-- CUPED reduces variance by subtracting pre-experiment covariate
-- pre_revenue = revenue in the week BEFORE the experiment
WITH pre AS (
    SELECT user_id, SUM(revenue) AS pre_revenue
    FROM conversions
    WHERE event_date < '2024-01-01'   -- pre-experiment period
    GROUP BY user_id
),
experiment AS (
    SELECT
        a.user_id,
        a.variant,
        c.revenue,
        COALESCE(p.pre_revenue, 0) AS pre_revenue
    FROM assignments a
    LEFT JOIN conversions c ON a.user_id = c.user_id
    LEFT JOIN pre p ON a.user_id = p.user_id
),
theta AS (
    -- theta = Cov(Y, X) / Var(X) — computed globally
    SELECT
        (AVG(revenue * pre_revenue) - AVG(revenue) * AVG(pre_revenue))
        / NULLIF(VARIANCE(pre_revenue), 0) AS theta
    FROM experiment
)
SELECT
    e.variant,
    AVG(e.revenue - t.theta * (e.pre_revenue - AVG(e.pre_revenue) OVER ())) AS cuped_mean,
    COUNT(*) AS n
FROM experiment e CROSS JOIN theta t
GROUP BY e.variant;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Users in multiple variants | Dilutes the experiment — treatment users see control content too | Check: `HAVING COUNT(DISTINCT variant) > 1` and exclude |
| Including pre-assignment events | User bought before being assigned — inflates treatment | Filter: `c.event_date >= a.assigned_at` |
| Peeking (stopping early on significance) | False positive rate balloons with multiple looks | Use sequential testing or pre-register a fixed end date |
| Ignoring network effects | Treatment user influences control user's behaviour | Use cluster randomisation or holdout cells |

---

## 35. Feature Engineering in SQL for ML

### Concept

SQL is often used to generate the training feature table for ML models. The key principles:

1. **One row per training example** (user, item pair, or event)
2. **All features must be computable at prediction time** (no future data)
3. **Ratio features need `NULLIF` guards**
4. **Categorical features need frequency or target encoding**

---

### Sample Tables

**`users`**: user_id, age, country, signup_date
**`events`**: user_id, event_date, event_type, item_id
**`items`**: item_id, category, price

---

### Variant 1 — User-level feature table

```sql
WITH base_date AS (SELECT DATE '2024-01-15' AS dt),
user_stats AS (
    SELECT
        e.user_id,
        -- Recency
        (SELECT dt FROM base_date) - MAX(e.event_date)                        AS days_since_last_event,
        -- Frequency
        COUNT(DISTINCT e.event_date)                                           AS active_days,
        COUNT(*)                                                               AS total_events,
        COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END)                 AS total_purchases,
        COUNT(CASE WHEN e.event_type = 'view'     THEN 1 END)                 AS total_views,
        -- Monetary
        COUNT(DISTINCT e.item_id)                                              AS unique_items_seen,
        -- Ratios
        ROUND(
            COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) * 1.0
            / NULLIF(COUNT(CASE WHEN e.event_type = 'view' THEN 1 END), 0),
            4
        )                                                                      AS purchase_to_view_ratio,
        -- Recency-weighted (recent activity counts more)
        SUM(CASE WHEN e.event_date >= (SELECT dt FROM base_date) - 7 THEN 1 ELSE 0 END)  AS events_7d,
        SUM(CASE WHEN e.event_date >= (SELECT dt FROM base_date) - 30 THEN 1 ELSE 0 END) AS events_30d
    FROM events e
    WHERE e.event_date < (SELECT dt FROM base_date)
    GROUP BY e.user_id
)
SELECT
    u.user_id,
    -- User profile
    DATE_PART('year', AGE((SELECT dt FROM base_date), u.signup_date)) AS account_age_years,
    u.country,
    u.age,
    -- Behavioural features
    us.days_since_last_event,
    us.active_days,
    us.total_purchases,
    us.total_views,
    us.purchase_to_view_ratio,
    us.events_7d,
    us.events_30d,
    -- Derived
    ROUND(us.events_7d * 1.0 / NULLIF(us.events_30d, 0), 3) AS recency_ratio   -- 7d/30d activity ratio
FROM users u
JOIN user_stats us ON u.user_id = us.user_id;
```

---

### Variant 2 — Item-level features + user-item interaction features

```sql
WITH user_item_pairs AS (
    -- One row per (user, item) — the training examples
    SELECT DISTINCT user_id, item_id FROM events WHERE event_type = 'view'
),
item_stats AS (
    SELECT
        item_id,
        COUNT(DISTINCT user_id)                             AS item_view_count,
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS item_purchase_count,
        ROUND(
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) * 1.0
            / NULLIF(COUNT(DISTINCT user_id), 0), 4
        )                                                   AS item_conversion_rate
    FROM events
    GROUP BY item_id
)
SELECT
    uip.user_id,
    uip.item_id,
    i.category,
    i.price,
    ist.item_view_count,
    ist.item_conversion_rate,
    -- Has this user bought this item before?
    MAX(CASE WHEN e.event_type = 'purchase' AND e.item_id = uip.item_id THEN 1 ELSE 0 END) AS user_bought_item,
    -- How many times did user view this item?
    COUNT(CASE WHEN e.item_id = uip.item_id AND e.event_type = 'view' THEN 1 END)         AS user_item_views
FROM user_item_pairs uip
JOIN items i ON uip.item_id = i.item_id
JOIN item_stats ist ON uip.item_id = ist.item_id
LEFT JOIN events e ON uip.user_id = e.user_id
GROUP BY uip.user_id, uip.item_id, i.category, i.price,
         ist.item_view_count, ist.item_conversion_rate;
```

---

### Variant 3 — Target encoding (replace category with its average target)

```sql
-- Target encoding: replace category with its avg conversion rate
-- Must use leave-one-out or cross-validated to avoid leakage in training

WITH category_rates AS (
    SELECT
        i.category,
        AVG(CASE WHEN e.event_type = 'purchase' THEN 1.0 ELSE 0 END) AS category_cvr,
        COUNT(DISTINCT e.user_id)                                      AS category_sample_size
    FROM events e
    JOIN items i ON e.item_id = i.item_id
    GROUP BY i.category
)
SELECT
    e.user_id,
    e.item_id,
    i.category,
    cr.category_cvr   AS category_target_encoded,
    cr.category_sample_size
FROM events e
JOIN items i ON e.item_id = i.item_id
JOIN category_rates cr ON i.category = cr.category
WHERE e.event_type = 'view';
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Data leakage from future events | Using events after the label timestamp | Always filter `WHERE event_date < label_date` |
| Target encoding leakage | Computing category CVR from full dataset including the row itself | Use cross-validation folds or leave-one-out encoding |
| NULL features break tree models differently | NULL in XGBoost vs NULL in sklearn behave differently | Decide on imputation strategy (median, mode, or leave NULL) consistently |
| Ratio features with zero denominator | `views = 0` → `NULLIF` guard needed | `ROUND(... / NULLIF(views, 0), 4)` |
| Not grouping at the right grain | One row per (user, day) vs one row per user — wrong grain for the model | Confirm the grain of the training table before writing the query |

---

## 36. Data Quality and Anomaly Detection

### Concept

Data quality checks and anomaly detection are core to ML pipelines — garbage in, garbage out. SQL is the primary tool for:
- Schema validation (unexpected NULLs, wrong types)
- Statistical anomalies (metric spikes, sudden drops)
- Distribution shift detection

---

### Sample Table: `daily_model_metrics`

| date       | model_name | accuracy | requests | error_rate |
|------------|------------|----------|----------|------------|
| 2024-01-10 | XGBoost    | 0.910    | 1200     | 0.010      |
| 2024-01-11 | XGBoost    | 0.912    | 1250     | 0.009      |
| 2024-01-12 | XGBoost    | 0.908    | 1180     | 0.011      |
| 2024-01-13 | XGBoost    | 0.550    | 50       | 0.450      |  ← anomaly
| 2024-01-14 | XGBoost    | 0.911    | 1230     | 0.010      |

---

### Variant 1 — Completeness and validity checks

```sql
SELECT
    COUNT(*)                                                       AS total_rows,
    COUNT(accuracy)                                                AS non_null_accuracy,
    COUNT(*) - COUNT(accuracy)                                     AS null_accuracy,
    SUM(CASE WHEN accuracy < 0 OR accuracy > 1 THEN 1 END)        AS out_of_range_accuracy,
    SUM(CASE WHEN requests < 0 THEN 1 END)                        AS negative_requests,
    SUM(CASE WHEN error_rate < 0 OR error_rate > 1 THEN 1 END)    AS invalid_error_rate,
    COUNT(DISTINCT date)                                           AS distinct_dates,
    MAX(date) - MIN(date) + 1                                     AS expected_days,
    (MAX(date) - MIN(date) + 1) - COUNT(DISTINCT date)            AS missing_days
FROM daily_model_metrics
WHERE model_name = 'XGBoost';
```

**Output:**

| total_rows | null_accuracy | out_of_range | missing_days |
|------------|---------------|--------------|--------------|
| 5          | 0             | 0            | 0            |

---

### Variant 2 — Z-score based anomaly detection

```sql
WITH stats AS (
    SELECT
        model_name,
        AVG(accuracy)    AS mean_acc,
        STDDEV(accuracy) AS std_acc,
        AVG(requests)    AS mean_req,
        STDDEV(requests) AS std_req
    FROM daily_model_metrics
    GROUP BY model_name
)
SELECT
    m.date,
    m.model_name,
    m.accuracy,
    m.requests,
    ROUND((m.accuracy - s.mean_acc) / NULLIF(s.std_acc, 0), 2)  AS accuracy_zscore,
    ROUND((m.requests - s.mean_req) / NULLIF(s.std_req, 0), 2)  AS requests_zscore,
    CASE
        WHEN ABS((m.accuracy - s.mean_acc) / NULLIF(s.std_acc, 0)) > 2 THEN 'ANOMALY'
        ELSE 'normal'
    END                                                           AS accuracy_status
FROM daily_model_metrics m
JOIN stats s ON m.model_name = s.model_name
ORDER BY m.date;
```

**Output:**

| date       | model_name | accuracy | accuracy_zscore | accuracy_status |
|------------|------------|----------|-----------------|-----------------|
| 2024-01-10 | XGBoost    | 0.910    | 0.21            | normal          |
| 2024-01-11 | XGBoost    | 0.912    | 0.36            | normal          |
| 2024-01-12 | XGBoost    | 0.908    | 0.06            | normal          |
| 2024-01-13 | XGBoost    | 0.550    | -3.92           | ANOMALY         |
| 2024-01-14 | XGBoost    | 0.911    | 0.28            | normal          |

---

### Variant 3 — Rolling baseline anomaly (compare today to prior 7 days)

```sql
WITH rolling AS (
    SELECT
        date,
        model_name,
        accuracy,
        AVG(accuracy) OVER (
            PARTITION BY model_name
            ORDER BY date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS rolling_mean,
        STDDEV(accuracy) OVER (
            PARTITION BY model_name
            ORDER BY date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) AS rolling_std
    FROM daily_model_metrics
)
SELECT
    date,
    model_name,
    accuracy,
    ROUND(rolling_mean, 4) AS baseline,
    ROUND((accuracy - rolling_mean) / NULLIF(rolling_std, 0), 2) AS zscore,
    CASE
        WHEN ABS((accuracy - rolling_mean) / NULLIF(rolling_std, 0)) > 2
        THEN 'ALERT'
        ELSE 'OK'
    END AS alert
FROM rolling
WHERE rolling_mean IS NOT NULL
ORDER BY date;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Z-score inflated by the anomaly itself | Computing stddev over a window that includes the anomaly makes z-score smaller | Use rolling baseline that excludes current row |
| Zero stddev when all values identical | Division by zero | `NULLIF(stddev, 0)` — returns NULL for z-score |
| Missing dates look like OK days | If Jan 13 was missing instead of anomalous, the check passes | Separately check for date gaps (Topic 35 gap detection) |
| Checking NULLs with = NULL | `WHERE accuracy = NULL` returns 0 rows | `WHERE accuracy IS NULL` |

---

## 37. Metric Definitions: DAU/WAU/MAU, Stickiness

### Concept

These are the most commonly asked product metric questions in ML interviews, especially for recommendation, ranking, and personalization roles.

| Metric | Definition |
|--------|------------|
| **DAU** | Distinct users active on a given calendar day |
| **WAU** | Distinct users active in a 7-day rolling window |
| **MAU** | Distinct users active in a 28- or 30-day rolling window |
| **DAU/MAU (Stickiness)** | How often monthly users come back daily; range [0,1] |
| **L28** | Count of days (out of 28) a user was active — Google's metric |

---

### Sample Table: `daily_activity`

| user_id | activity_date |
|---------|---------------|
| u1      | 2024-01-01    |
| u1      | 2024-01-02    |
| u1      | 2024-01-08    |
| u2      | 2024-01-01    |
| u2      | 2024-01-15    |
| u3      | 2024-01-05    |
| u3      | 2024-01-06    |
| u3      | 2024-01-07    |

---

### Variant 1 — DAU, WAU, MAU for each date

```sql
SELECT
    d.activity_date,
    -- DAU: active on this exact day
    COUNT(DISTINCT CASE WHEN a.activity_date = d.activity_date
                        THEN a.user_id END)                    AS dau,
    -- WAU: active in prior 7 days (rolling)
    COUNT(DISTINCT CASE WHEN a.activity_date BETWEEN d.activity_date - 6
                                                  AND d.activity_date
                        THEN a.user_id END)                    AS wau,
    -- MAU: active in prior 28 days (rolling)
    COUNT(DISTINCT CASE WHEN a.activity_date BETWEEN d.activity_date - 27
                                                  AND d.activity_date
                        THEN a.user_id END)                    AS mau
FROM (SELECT DISTINCT activity_date FROM daily_activity) d
CROSS JOIN daily_activity a
WHERE a.activity_date BETWEEN d.activity_date - 27 AND d.activity_date
GROUP BY d.activity_date
ORDER BY d.activity_date;
```

**Output:**

| activity_date | dau | wau | mau |
|---------------|-----|-----|-----|
| 2024-01-01    | 2   | 2   | 2   |
| 2024-01-02    | 1   | 2   | 3   |
| 2024-01-05    | 1   | 2   | 3   |
| 2024-01-06    | 1   | 2   | 3   |
| 2024-01-07    | 1   | 2   | 3   |
| 2024-01-08    | 1   | 2   | 3   |
| 2024-01-15    | 1   | 1   | 3   |

---

### Variant 2 — Stickiness ratio (DAU/MAU) per day

```sql
WITH metrics AS (
    -- (use query from Variant 1 as a CTE here)
    ...
)
SELECT
    activity_date,
    dau,
    mau,
    ROUND(dau * 1.0 / NULLIF(mau, 0), 3) AS stickiness
FROM metrics
ORDER BY activity_date;
```

---

### Variant 3 — L28: per-user days active in last 28 days

```sql
-- As of 2024-01-28
SELECT
    user_id,
    COUNT(DISTINCT activity_date)        AS l28_days_active,
    COUNT(DISTINCT activity_date) / 28.0 AS l28_ratio
FROM daily_activity
WHERE activity_date BETWEEN '2024-01-01' AND '2024-01-28'
GROUP BY user_id
ORDER BY l28_days_active DESC;
```

**Output:**

| user_id | l28_days_active | l28_ratio |
|---------|-----------------|-----------|
| u1      | 3               | 0.107     |
| u3      | 3               | 0.107     |
| u2      | 2               | 0.071     |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| MAU with calendar month vs 28-day rolling | Calendar month changes length; 28-day rolling is comparable across months | Use rolling window; specify which definition in the interview |
| Counting sessions not users | Events table may have many rows per user per day | `COUNT(DISTINCT user_id)` always |
| Stickiness > 1 | Impossible with correct DAU/MAU — means your DAU > MAU, which is a bug | Verify DAU window is ≤ MAU window |
| Timezone: same event in two days | UTC midnight can split a user's session into two calendar days | Standardise timezone before computing DAU |

---

## 38. Market Basket / Co-occurrence Analysis

### Concept

Market basket analysis finds **which items are frequently purchased/viewed together**. The SQL pattern uses a **self-join on the same basket (order/session)** to generate all item pairs, then counts co-occurrences.

Key metrics:
- **Support** = P(A and B) = baskets with both / total baskets
- **Confidence** = P(B | A) = baskets with both / baskets with A
- **Lift** = Confidence(A→B) / Support(B) — > 1 means positive association

---

### Sample Table: `orders`

| order_id | user_id | item_id | category |
|----------|---------|---------|----------|
| o1       | u1      | A       | shoes    |
| o1       | u1      | B       | socks    |
| o1       | u1      | C       | bag      |
| o2       | u2      | A       | shoes    |
| o2       | u2      | B       | socks    |
| o3       | u3      | B       | socks    |
| o3       | u3      | D       | hat      |
| o4       | u4      | A       | shoes    |
| o4       | u4      | C       | bag      |

---

### Variant 1 — Item pair co-occurrence counts

```sql
WITH pairs AS (
    SELECT
        a.order_id,
        a.item_id AS item_a,
        b.item_id AS item_b
    FROM orders a
    JOIN orders b
        ON a.order_id = b.order_id
        AND a.item_id < b.item_id   -- avoid duplicates: (A,B) and (B,A)
),
total_orders AS (SELECT COUNT(DISTINCT order_id) AS n FROM orders)
SELECT
    p.item_a,
    p.item_b,
    COUNT(*)                                                  AS co_occurrences,
    ROUND(COUNT(*) * 1.0 / t.n, 3)                          AS support
FROM pairs p
CROSS JOIN total_orders t
GROUP BY p.item_a, p.item_b, t.n
ORDER BY co_occurrences DESC;
```

**Output:**

| item_a | item_b | co_occurrences | support |
|--------|--------|----------------|---------|
| A      | B      | 2              | 0.500   |
| A      | C      | 2              | 0.500   |
| B      | C      | 1              | 0.250   |
| B      | D      | 1              | 0.250   |

---

### Variant 2 — Confidence and Lift

```sql
WITH pairs AS (
    SELECT a.order_id, a.item_id AS item_a, b.item_id AS item_b
    FROM orders a JOIN orders b
        ON a.order_id = b.order_id AND a.item_id != b.item_id   -- both directions
),
item_counts AS (
    SELECT item_id, COUNT(DISTINCT order_id) AS item_orders
    FROM orders
    GROUP BY item_id
),
total AS (SELECT COUNT(DISTINCT order_id) AS n FROM orders),
pair_counts AS (
    SELECT item_a, item_b, COUNT(DISTINCT order_id) AS pair_orders
    FROM pairs
    GROUP BY item_a, item_b
)
SELECT
    pc.item_a,
    pc.item_b,
    pc.pair_orders,
    ia.item_orders                                                      AS item_a_orders,
    ib.item_orders                                                      AS item_b_orders,
    ROUND(pc.pair_orders * 1.0 / t.n, 3)                              AS support,
    ROUND(pc.pair_orders * 1.0 / NULLIF(ia.item_orders, 0), 3)        AS confidence,   -- P(B|A)
    ROUND(
        (pc.pair_orders * 1.0 / NULLIF(ia.item_orders, 0))
        / (ib.item_orders * 1.0 / t.n),
        3
    )                                                                   AS lift
FROM pair_counts pc
JOIN item_counts ia ON pc.item_a = ia.item_id
JOIN item_counts ib ON pc.item_b = ib.item_id
CROSS JOIN total t
ORDER BY lift DESC;
```

**Output:**

| item_a | item_b | support | confidence | lift  |
|--------|--------|---------|------------|-------|
| A      | B      | 0.500   | 0.667      | 0.889 |
| A      | C      | 0.500   | 0.667      | 1.333 |
| C      | A      | 0.500   | 1.000      | 1.333 |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Self-join explosion | n items per basket → n² pairs; 1000-item baskets create 1M rows | Cap basket size or use approximate methods |
| item_a < item_b vs != | `<` gives unique pairs; `!=` gives directed pairs (A→B and B→A) | Use `<` for support; `!=` for directional confidence/lift |
| Lift < 1 misread | Lift < 1 = negative association (items repel each other) | Note this in output; it's useful information |
| Cold start: rare items have unreliable lift | One basket with A and B inflates lift if B is rare | Filter `WHERE item_a_orders >= 5 AND item_b_orders >= 5` |

---

## 39. Attribution Modeling — First-touch / Last-touch

### Concept

Attribution assigns **credit for a conversion** to the marketing touchpoints that preceded it. SQL implements the most common models:

| Model | Logic |
|-------|-------|
| **First-touch** | 100% credit to the first touchpoint |
| **Last-touch** | 100% credit to the touchpoint immediately before conversion |
| **Linear** | Equal credit split across all touchpoints |
| **Time-decay** | More credit to touchpoints closer to conversion |
| **Position-based** | 40% first, 40% last, 20% split across middle |

---

### Sample Table: `touchpoints`

| user_id | channel  | touchpoint_ts       | converted | conversion_ts       |
|---------|----------|---------------------|-----------|---------------------|
| u1      | email    | 2024-01-01 09:00:00 | 1         | 2024-01-05 14:00:00 |
| u1      | social   | 2024-01-03 11:00:00 | 1         | 2024-01-05 14:00:00 |
| u1      | search   | 2024-01-05 13:00:00 | 1         | 2024-01-05 14:00:00 |
| u2      | email    | 2024-01-02 10:00:00 | 0         | NULL                |
| u2      | display  | 2024-01-04 12:00:00 | 0         | NULL                |

---

### Variant 1 — First-touch and Last-touch attribution

```sql
WITH ranked AS (
    SELECT
        user_id,
        channel,
        touchpoint_ts,
        converted,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY touchpoint_ts ASC)  AS first_rank,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY touchpoint_ts DESC) AS last_rank,
        COUNT(*) OVER (PARTITION BY user_id)                                 AS total_touches
    FROM touchpoints
    WHERE converted = 1   -- only converting users
)
SELECT
    channel,
    SUM(CASE WHEN first_rank = 1 THEN 1 ELSE 0 END)   AS first_touch_conversions,
    SUM(CASE WHEN last_rank  = 1 THEN 1 ELSE 0 END)   AS last_touch_conversions
FROM ranked
GROUP BY channel
ORDER BY last_touch_conversions DESC;
```

**Output:**

| channel | first_touch_conversions | last_touch_conversions |
|---------|-------------------------|------------------------|
| email   | 1                       | 0                      |
| search  | 0                       | 1                      |
| social  | 0                       | 0                      |

---

### Variant 2 — Linear attribution (equal split)

```sql
WITH converting_touches AS (
    SELECT
        user_id,
        channel,
        touchpoint_ts,
        COUNT(*) OVER (PARTITION BY user_id) AS total_touches
    FROM touchpoints
    WHERE converted = 1
)
SELECT
    channel,
    COUNT(*)                                              AS touchpoint_count,
    ROUND(SUM(1.0 / total_touches), 2)                   AS linear_credit
FROM converting_touches
GROUP BY channel
ORDER BY linear_credit DESC;
```

**Output:**

| channel | touchpoint_count | linear_credit |
|---------|------------------|---------------|
| email   | 1                | 0.33          |
| search  | 1                | 0.33          |
| social  | 1                | 0.33          |

---

### Variant 3 — Time-decay attribution

```sql
WITH converting AS (
    SELECT
        user_id,
        channel,
        touchpoint_ts,
        conversion_ts,
        -- Days before conversion (more recent = smaller gap = more weight)
        EXTRACT(EPOCH FROM (conversion_ts - touchpoint_ts)) / 86400 AS days_before_conversion
    FROM touchpoints
    WHERE converted = 1
),
with_weights AS (
    SELECT
        user_id,
        channel,
        -- Exponential decay: weight = 0.5 ^ (days / 7) — halves every 7 days
        POWER(0.5, days_before_conversion / 7.0) AS raw_weight
    FROM converting
),
normalised AS (
    SELECT
        user_id,
        channel,
        raw_weight,
        raw_weight / SUM(raw_weight) OVER (PARTITION BY user_id) AS normalised_weight
    FROM with_weights
)
SELECT
    channel,
    ROUND(SUM(normalised_weight), 3) AS time_decay_credit
FROM normalised
GROUP BY channel
ORDER BY time_decay_credit DESC;
```

**Output:**

| channel | time_decay_credit |
|---------|-------------------|
| search  | 0.571             |
| social  | 0.286             |
| email   | 0.143             |

> Search gets the most credit — it was closest to conversion. Email was 4 days before, so least credit.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Including touchpoints after conversion | Post-conversion ads get credit they don't deserve | `WHERE touchpoint_ts < conversion_ts` |
| Cross-device / cross-session matching | Same user on mobile and desktop = two user_ids | Requires identity resolution before attribution |
| Self-attribution bias | Email channel sends to existing buyers → high last-touch attribution | Use incrementality testing to validate |
| Credit summing to != 1 | Floating point errors in normalisation | Round and verify: `SUM(credit) = 1.0 per user` |

---

## 40. Sampling in SQL — TABLESAMPLE, Stratified Sampling

### Concept

For large datasets, you need SQL-level sampling for:
- Quick exploratory queries
- Creating train/test splits
- Balanced class sampling for imbalanced classification

| Method | DB | Notes |
|--------|----|-------|
| `TABLESAMPLE BERNOULLI(p)` | PostgreSQL, SQL Standard | Each row independently included with probability p% — exact percentage varies |
| `TABLESAMPLE SYSTEM(p)` | PostgreSQL | Block-level sampling — faster but less random |
| `WHERE RANDOM() < 0.1` | PostgreSQL | Row-by-row random filter — slowest but most flexible |
| `MOD(user_id::INT, 10) = 0` | All DBs | Deterministic 10% sample — reproducible |
| `RAND() < 0.1` | MySQL/BigQuery | Same as RANDOM() |

---

### Sample Table: `model_training_data`

| row_id | user_id | features | label |
|--------|---------|----------|-------|
| ...    | ...     | ...      | 0/1   |

---

### Variant 1 — Simple random sample (two methods)

```sql
-- Method 1: TABLESAMPLE (fast, approximate)
SELECT * FROM model_training_data TABLESAMPLE BERNOULLI(10);   -- ~10% of rows

-- Method 2: RANDOM() filter (slower but works anywhere)
SELECT * FROM model_training_data WHERE RANDOM() < 0.10;

-- Method 3: Deterministic (same result every run — good for reproducibility)
SELECT * FROM model_training_data WHERE MOD(row_id, 10) = 0;
```

---

### Variant 2 — Stratified sampling (preserve class balance)

```sql
-- Sample 10% from each class independently to maintain class distribution
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY label
            ORDER BY RANDOM()    -- shuffle within each class
        ) AS rn,
        COUNT(*) OVER (PARTITION BY label) AS class_size
    FROM model_training_data
)
SELECT row_id, user_id, features, label
FROM ranked
WHERE rn <= CEIL(class_size * 0.10)   -- keep 10% from each class
ORDER BY label, rn;
```

---

### Variant 3 — Train/val/test split (80/10/10)

```sql
WITH shuffled AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn,
        COUNT(*) OVER ()                      AS total
    FROM model_training_data
)
SELECT
    row_id,
    features,
    label,
    CASE
        WHEN rn <= total * 0.80 THEN 'train'
        WHEN rn <= total * 0.90 THEN 'val'
        ELSE 'test'
    END AS split
FROM shuffled;
```

---

### Variant 4 — Stratified train/val/test split (preserve class balance in each split)

```sql
WITH class_shuffled AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY label ORDER BY RANDOM()) AS rn_in_class,
        COUNT(*) OVER (PARTITION BY label)                       AS class_size
    FROM model_training_data
)
SELECT
    row_id, features, label,
    CASE
        WHEN rn_in_class <= class_size * 0.80 THEN 'train'
        WHEN rn_in_class <= class_size * 0.90 THEN 'val'
        ELSE 'test'
    END AS split
FROM class_shuffled;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `RANDOM()` is non-reproducible | Different result every run | Use `SETSEED(0.42)` before query (PostgreSQL) or MOD-based deterministic sample |
| `TABLESAMPLE SYSTEM` is block-level | May return 0 rows for small tables or over-represent some pages | Use BERNOULLI for correctness; SYSTEM only for very large tables |
| Stratified split with float CEIL | CEIL(100 * 0.10) = 10 but CEIL(99 * 0.10) = 10 too — slight imbalance | Acceptable; just document the exact split sizes |
| Train/test leakage at user level | Same user appears in train and test splits | Split on `user_id` not `row_id`: `MOD(user_id::INT, 10) < 8` for train |

---

## 41. Fuzzy / Near-duplicate Matching

### Concept

Exact string matching (`=`) fails on messy data: `"XGBoost"` vs `"xgboost"` vs `"XG Boost"`. Fuzzy matching handles:
- **Case/whitespace differences** → LOWER + TRIM (exact, already covered)
- **Typos / edit distance** → Levenshtein distance (pg_trgm extension)
- **Phonetic similarity** → Soundex, Metaphone
- **Near-duplicate records** → trigram similarity

---

### PostgreSQL Extensions Required

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- trigram similarity
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;  -- Levenshtein, Soundex, Metaphone
```

---

### Sample Table: `raw_model_names`

| id | model_name       |
|----|------------------|
| 1  | XGBoost          |
| 2  | xgboost          |
| 3  | XG Boost         |
| 4  | XGBooost         |
| 5  | LightGBM         |
| 6  | Light GBM        |
| 7  | lightgbm         |
| 8  | BERT             |
| 9  | Bert             |
| 10 | bERT             |

---

### Variant 1 — Levenshtein distance (edit distance)

```sql
SELECT
    a.id AS id_a,
    a.model_name AS name_a,
    b.id AS id_b,
    b.model_name AS name_b,
    LEVENSHTEIN(LOWER(a.model_name), LOWER(b.model_name)) AS edit_distance
FROM raw_model_names a
JOIN raw_model_names b ON a.id < b.id
WHERE LEVENSHTEIN(LOWER(a.model_name), LOWER(b.model_name)) <= 2  -- within 2 edits
ORDER BY edit_distance, a.model_name;
```

**Output:**

| id_a | name_a   | id_b | name_b   | edit_distance |
|------|----------|------|----------|---------------|
| 1    | XGBoost  | 2    | xgboost  | 0             |
| 8    | BERT     | 9    | Bert     | 0             |
| 8    | BERT     | 10   | bERT     | 0             |
| 1    | XGBoost  | 4    | XGBooost | 1             |
| 5    | LightGBM | 7    | lightgbm | 0             |

> Edit distance = 0 after LOWER means only case differences. Edit distance = 1 means one character added/removed/changed.

---

### Variant 2 — Trigram similarity (pg_trgm)

```sql
SELECT
    a.model_name AS name_a,
    b.model_name AS name_b,
    ROUND(SIMILARITY(a.model_name, b.model_name)::NUMERIC, 3) AS trgm_similarity
FROM raw_model_names a
JOIN raw_model_names b ON a.id < b.id
WHERE SIMILARITY(a.model_name, b.model_name) > 0.4
ORDER BY trgm_similarity DESC;
```

**Output:**

| name_a   | name_b   | trgm_similarity |
|----------|----------|-----------------|
| XGBoost  | XGBooost | 0.800           |
| LightGBM | Light GBM| 0.727           |
| XGBoost  | XG Boost | 0.700           |
| BERT     | Bert     | 0.500           |

---

### Variant 3 — Cluster near-duplicates (canonical form assignment)

```sql
-- Assign a canonical name: lowest id in the near-duplicate cluster
WITH similarity_pairs AS (
    SELECT
        a.id AS id_a,
        b.id AS id_b,
        SIMILARITY(LOWER(a.model_name), LOWER(b.model_name)) AS sim
    FROM raw_model_names a
    JOIN raw_model_names b ON a.id < b.id
    WHERE SIMILARITY(LOWER(a.model_name), LOWER(b.model_name)) > 0.5
),
-- Each id maps to the lowest id in its cluster (simplified single-link clustering)
cluster_map AS (
    SELECT id_b AS id, MIN(id_a) AS canonical_id FROM similarity_pairs GROUP BY id_b
    UNION ALL
    SELECT id_a, id_a FROM (SELECT DISTINCT id_a FROM similarity_pairs) t
)
SELECT
    r.id,
    r.model_name,
    COALESCE(cm.canonical_id, r.id) AS canonical_id,
    c.model_name                    AS canonical_name
FROM raw_model_names r
LEFT JOIN cluster_map cm ON r.id = cm.id
LEFT JOIN raw_model_names c ON COALESCE(cm.canonical_id, r.id) = c.id
ORDER BY canonical_id, r.id;
```

**Output (simplified):**

| id | model_name | canonical_id | canonical_name |
|----|------------|--------------|----------------|
| 1  | XGBoost    | 1            | XGBoost        |
| 2  | xgboost    | 1            | XGBoost        |
| 3  | XG Boost   | 1            | XGBoost        |
| 4  | XGBooost   | 1            | XGBoost        |
| 5  | LightGBM   | 5            | LightGBM       |
| 6  | Light GBM  | 5            | LightGBM       |
| 7  | lightgbm   | 5            | LightGBM       |
| 8  | BERT       | 8            | BERT           |
| 9  | Bert       | 8            | BERT           |
| 10 | bERT       | 8            | BERT           |

---

### Variant 4 — Soundex for phonetic matching

```sql
-- Soundex: groups words that sound the same
SELECT
    model_name,
    SOUNDEX(model_name) AS soundex_code
FROM raw_model_names
ORDER BY soundex_code, model_name;
```

**Output:**

| model_name | soundex_code |
|------------|--------------|
| BERT       | B630         |
| Bert       | B630         |
| bERT       | B630         |
| LightGBM   | L323         |
| Light GBM  | L323         |
| XGBoost    | X213         |
| XGBooost   | X213         |

> Soundex groups BERT/Bert/bERT correctly. Good for name deduplication in entity matching tasks.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Levenshtein on long strings is slow | O(n*m) per pair; n² pairs in a cross join = very slow on large tables | Filter candidates with trigram index first; use Levenshtein only on shortlist |
| Trigram similarity threshold tuning | 0.3 is very loose; 0.8 is very strict — no universal threshold | Test on labelled pairs; plot precision/recall curve for your data |
| Soundex too aggressive | "BERT" and "Birte" have same Soundex | Use Metaphone (more accurate) or trigrams for technical strings |
| Transitive clusters not handled | A≈B and B≈C but A≉C — naive approach puts all in one cluster | Use Union-Find algorithm (implement in Python after SQL identifies pairs) |
| Cross join on large tables | 100k names → 10B pairs | Block on first character or first trigram before cross joining |

---

## Master Cheat Sheet — Applied Topics

### Pattern → Query Shape

| Interview question | Core query pattern |
|-------------------|--------------------|
| "Build a funnel" | `COUNT(DISTINCT user_id) WHERE event_type = X` per step + UNION ALL |
| "Compute Day-7 retention" | `signup cohort JOIN activity WHERE days_since_signup = 7` |
| "Assign session IDs" | `LAG(ts) + SUM(is_new_session) OVER (PARTITION BY user ORDER BY ts)` |
| "Create rolling features for ML" | `CROSS JOIN dates + LEFT JOIN events WHERE event_date < pred_date + GROUP BY user, date` |
| "Analyse A/B test" | Per-variant `AVG(converted)` + SRM check + STDDEV |
| "Build a feature table" | `users CROSS JOIN pred_date LEFT JOIN events WHERE ts < pred_date` |
| "Find anomalous days" | Z-score: `(value - rolling_avg) / rolling_std > 2` with `ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING` |
| "Compute DAU/MAU stickiness" | `COUNT(DISTINCT user_id)` over different rolling windows, divide |
| "Market basket / 'customers also bought'" | Self-join orders on same order_id + count pair co-occurrences |
| "Attribution modelling" | `ROW_NUMBER() OVER (PARTITION BY user ORDER BY ts)` for first/last; SUM(1/n) for linear |
| "Sample 10% for training" | `WHERE MOD(user_id, 10) = 0` (deterministic) or `TABLESAMPLE BERNOULLI(10)` |
| "Deduplicate model names" | `SIMILARITY(LOWER(a), LOWER(b)) > 0.5` + cluster via canonical ID |

---

*End of Module 5 — Applied & Special Topics.*
*You now have the complete SQL for ML Interview guide across all 5 modules.*
