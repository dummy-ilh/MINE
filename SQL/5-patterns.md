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
42. [Histogram & Bucketing](#42-histogram-buckets)
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
## 42. Histogram & Bucketing


---

## Table of Contents

1. [The Bucketing Mindset](#1-the-bucketing-mindset)
2. [The Three Bucketing Methods](#2-the-three-bucketing-methods)
3. [Fixed-Width Buckets — CASE WHEN](#3-fixed-width-buckets--case-when)
4. [Dynamic-Width Buckets — FLOOR + Division](#4-dynamic-width-buckets--floor--division)
5. [Percentile Buckets — NTILE](#5-percentile-buckets--ntile)
6. [Width-Bucket — Auto Histogram](#6-width-bucket--auto-histogram)
7. [Frequency Distribution — The True Histogram](#7-frequency-distribution--the-true-histogram)
8. [Cumulative Distribution (CDF)](#8-cumulative-distribution-cdf)
9. [Revenue Bucketing by Order Size](#9-revenue-bucketing-by-order-size)
10. [Age Cohort Analysis](#10-age-cohort-analysis)
11. [Spend Tier Segmentation (RFM-style)](#11-spend-tier-segmentation-rfm-style)
12. [Bucket + Funnel (Conversion by Tier)](#12-bucket--funnel-conversion-by-tier)
13. [Session Duration Distribution](#13-session-duration-distribution)
14. [Percentile Computation (P50, P90, P99)](#14-percentile-computation-p50-p90-p99)
15. [Outlier Detection via Buckets](#15-outlier-detection-via-buckets)
16. [Key Takeaways Cheatsheet](#16-key-takeaways-cheatsheet)

---

## 1. The Bucketing Mindset

Raw continuous values — age, revenue, session duration, order amount — are useless in GROUP BY. You can't group by `age = 27` and learn anything. Bucketing converts a continuous axis into **discrete, comparable groups**.

```
Raw column:  23, 27, 31, 45, 52, 67, 71, 84
                         │
                    BUCKET BY 20s
                         │
             20-29 → 2 users
             30-39 → 1 user
             40-49 → 1 user
             50-59 → 1 user
             60-69 → 1 user
             70-79 → 1 user
             80-89 → 1 user
```

**When to bucket:**
- Visualizing distributions ("how are our users spread across age groups?")
- Comparing metrics across tiers ("do high spenders churn less?")
- Cohort analysis ("how does conversion differ by order size?")
- Detecting skew / outliers ("is revenue concentrated in a few users?")

**The three questions before writing any bucket query:**
1. **Fixed or dynamic width?** — Do you want equal-sized ranges (0-10, 10-20) or equal-sized populations (each bucket has same number of users)?
2. **Business-driven or data-driven?** — Business says "under 25, 25-34, 35-44, 45+" OR you let the data decide with NTILE/percentiles.
3. **What to aggregate inside each bucket?** — Count, sum, average, conversion rate, retention rate?

---

## 2. The Three Bucketing Methods

| Method | What It Does | Use When |
|--------|-------------|----------|
| `CASE WHEN` | Hand-written ranges, any width | Business-defined tiers, irregular breakpoints |
| `FLOOR(value / width) * width` | Auto equal-width buckets | Quick histograms, any numeric column |
| `NTILE(n)` | Equal-population buckets (quantiles) | Quartiles, deciles, percentile tiers |
| `WIDTH_BUCKET` | Auto histogram between min/max | PostgreSQL/Snowflake; cleanest syntax |

Each method solves a different problem. You'll mix them in real queries.

---

## 3. Fixed-Width Buckets — CASE WHEN

**The most readable, most interview-friendly method. Use when a business person defined the tiers.**

### Pattern

```sql
CASE
  WHEN value <  10          THEN '0-9'
  WHEN value <  20          THEN '10-19'
  WHEN value <  30          THEN '20-29'
  WHEN value >= 30          THEN '30+'
  ELSE 'Unknown'
END AS bucket
```

SQL evaluates CASE top-to-bottom and stops at the first TRUE condition. So `WHEN value < 20` implicitly means `>= 10 AND < 20` because the `< 10` case already caught anything below 10.

### Full Query — Age Distribution

```sql
WITH age_buckets AS (
  SELECT
    user_id,
    age,
    CASE
      WHEN age < 18           THEN 'Under 18'
      WHEN age BETWEEN 18 AND 24 THEN '18-24'
      WHEN age BETWEEN 25 AND 34 THEN '25-34'
      WHEN age BETWEEN 35 AND 44 THEN '35-44'
      WHEN age BETWEEN 45 AND 54 THEN '45-54'
      WHEN age >= 55          THEN '55+'
      ELSE 'Unknown'
    END AS age_group
  FROM users
)
SELECT
  age_group,
  COUNT(*)                                   AS user_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM age_buckets
GROUP BY age_group
ORDER BY MIN(age);   -- order by actual age, not alphabetically
```

### CTE-Level I/O Trace

**Input `users`:**
```
user_id | age
U1      | 22
U2      | 29
U3      | 33
U4      | 41
U5      | 17
U6      | 28
U7      | 60
```

**`age_buckets`:**
```
user_id | age | age_group
U1      | 22  | 18-24
U2      | 29  | 25-34
U3      | 33  | 25-34
U4      | 41  | 35-44
U5      | 17  | Under 18
U6      | 28  | 25-34
U7      | 60  | 55+
```

**GROUP BY age_group + COUNT:**
```
age_group  | user_count | pct_of_total
Under 18   | 1          | 14.29
18-24      | 1          | 14.29
25-34      | 3          | 42.86
35-44      | 1          | 14.29
55+        | 1          | 14.29
```

**Why `ORDER BY MIN(age)` not `ORDER BY age_group`?** Alphabetical order gives: "18-24, 25-34, 35-44, 55+, Under 18" — wrong. `MIN(age)` inside each group returns the smallest real age value per bucket, so the order is chronological.

**Why `SUM(COUNT(*)) OVER ()`?** This is a window function with no PARTITION BY — it sums across ALL rows of the result set, giving total users. Each row divides its own count by that total for the percentage.

---

## 4. Dynamic-Width Buckets — FLOOR + Division

**When you don't want to write every bucket by hand — especially useful for revenue, price, or any column with a wide range.**

### The Formula

```sql
FLOOR(value / bucket_width) * bucket_width AS bucket_floor
```

This maps any value to the lower bound of its bucket:
- `FLOOR(23 / 10) * 10 = FLOOR(2.3) * 10 = 2 * 10 = 20`  → bucket "20-29"
- `FLOOR(57 / 10) * 10 = FLOOR(5.7) * 10 = 5 * 10 = 50`  → bucket "50-59"
- `FLOOR(100 / 10) * 10 = 100`                              → bucket "100-109"

### Full Query — Order Amount Distribution

```sql
WITH bucketed AS (
  SELECT
    order_id,
    amount,
    FLOOR(amount / 50) * 50 AS bucket_floor     -- $50 bucket width
  FROM orders
)
SELECT
  bucket_floor                                        AS bucket_start,
  bucket_floor + 49                                   AS bucket_end,
  CONCAT('$', bucket_floor, '-$', bucket_floor + 49) AS bucket_label,
  COUNT(*)                                            AS order_count,
  SUM(amount)                                         AS total_revenue,
  ROUND(AVG(amount), 2)                               AS avg_order_value,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_orders
FROM bucketed
GROUP BY bucket_floor
ORDER BY bucket_floor;
```

### CTE-Level I/O Trace

**Input `orders`:**
```
order_id | amount
O1       | 12.50
O2       | 47.00
O3       | 53.00
O4       | 98.00
O5       | 105.00
O6       | 23.00
O7       | 76.00
```

**`bucketed` — FLOOR(amount / 50) * 50:**
```
order_id | amount | FLOOR(amount/50) | bucket_floor
O1       | 12.50  | FLOOR(0.25) = 0  | 0
O2       | 47.00  | FLOOR(0.94) = 0  | 0
O3       | 53.00  | FLOOR(1.06) = 1  | 50
O4       | 98.00  | FLOOR(1.96) = 1  | 50
O5       | 105.00 | FLOOR(2.10) = 2  | 100
O6       | 23.00  | FLOOR(0.46) = 0  | 0
O7       | 76.00  | FLOOR(1.52) = 1  | 50
```

**GROUP BY bucket_floor:**
```
bucket_start | bucket_end | label       | order_count | total_revenue | avg_order
0            | 49         | $0-$49      | 3           | 82.50         | 27.50
50           | 99         | $50-$99     | 3           | 227.00        | 75.67
100          | 149        | $100-$149   | 1           | 105.00        | 105.00
```

**Changing bucket width is one number change:** Switch `50` to `25` for finer granularity, or `100` for coarser. No rewriting of CASE WHEN.

---

## 5. Percentile Buckets — NTILE

**Equal-population buckets. Instead of equal-width ranges, every bucket gets the same number of rows.**

### When to Use NTILE vs FLOOR

| FLOOR bucketing | NTILE bucketing |
|----------------|-----------------|
| Equal-width ranges | Equal-population groups |
| Most users in $0-$50, few in $500+ | Each quartile has exactly 25% of users |
| Good for showing distribution shape | Good for "top 25%" analysis |
| Sensitive to outliers | Robust to outliers |

### Full Query — Customer Spend Quartiles

```sql
WITH spend_per_user AS (
  SELECT
    user_id,
    SUM(amount) AS total_spend
  FROM orders
  GROUP BY user_id
),
quartiles AS (
  SELECT
    user_id,
    total_spend,
    NTILE(4) OVER (ORDER BY total_spend ASC) AS quartile
  FROM spend_per_user
)
SELECT
  quartile,
  CASE quartile
    WHEN 1 THEN 'Q1 — Bottom 25%'
    WHEN 2 THEN 'Q2 — Lower Mid'
    WHEN 3 THEN 'Q3 — Upper Mid'
    WHEN 4 THEN 'Q4 — Top 25%'
  END AS quartile_label,
  COUNT(*)                AS user_count,
  MIN(total_spend)        AS min_spend,
  MAX(total_spend)        AS max_spend,
  ROUND(AVG(total_spend), 2) AS avg_spend,
  SUM(total_spend)        AS total_revenue_from_tier
FROM quartiles
GROUP BY quartile
ORDER BY quartile;
```

### CTE-Level I/O Trace

**Input `orders` → `spend_per_user`:**
```
user_id | total_spend
U1      | 20
U2      | 45
U3      | 80
U4      | 150
U5      | 200
U6      | 320
U7      | 500
U8      | 950
```

**`quartiles` — NTILE(4) ORDER BY total_spend ASC:**

8 users, 4 buckets → 2 users per bucket.
```
user_id | total_spend | quartile
U1      | 20          | 1
U2      | 45          | 1
U3      | 80          | 2
U4      | 150         | 2
U5      | 200         | 3
U6      | 320         | 3
U7      | 500         | 4
U8      | 950         | 4
```

**GROUP BY quartile:**
```
quartile | label          | count | min  | max  | avg    | total_revenue
1        | Q1 Bottom 25%  | 2     | 20   | 45   | 32.50  | 65
2        | Q2 Lower Mid   | 2     | 80   | 150  | 115.00 | 230
3        | Q3 Upper Mid   | 2     | 200  | 320  | 260.00 | 520
4        | Q4 Top 25%     | 2     | 500  | 950  | 725.00 | 1450
```

**The key insight:** Q4 (top 25% of users) generates 1450/2265 = **64% of total revenue**. This is the kind of finding that drives business decisions — protect your top quartile at all costs.

### Deciles — Same Pattern, N=10

```sql
NTILE(10) OVER (ORDER BY total_spend ASC) AS decile
-- Decile 10 = top 10%, decile 1 = bottom 10%
```

---

## 6. WIDTH_BUCKET — Auto Histogram

**PostgreSQL / Snowflake / BigQuery only. Cleanest syntax for equal-width histograms.**

```sql
-- WIDTH_BUCKET(value, min, max, num_buckets)
-- Returns bucket number 1 through num_buckets
-- Values below min → 0, above max → num_buckets + 1

SELECT
  WIDTH_BUCKET(amount, 0, 1000, 10) AS bucket_num,
  -- Translate bucket number back to range label
  (WIDTH_BUCKET(amount, 0, 1000, 10) - 1) * 100 AS range_start,
  WIDTH_BUCKET(amount, 0, 1000, 10) * 100        AS range_end,
  COUNT(*)                                        AS order_count,
  SUM(amount)                                     AS revenue
FROM orders
WHERE amount BETWEEN 0 AND 1000
GROUP BY WIDTH_BUCKET(amount, 0, 1000, 10)
ORDER BY bucket_num;
```

### I/O Trace

`WIDTH_BUCKET(amount, 0, 1000, 10)` creates 10 equal buckets of width 100:
```
amount  | bucket_num | range
55      | 1          | $0–$100
150     | 2          | $100–$200
375     | 4          | $300–$400
999     | 10         | $900–$1000
1001    | 11         | overflow (> max)
```

**Output:**
```
bucket_num | range_start | range_end | order_count | revenue
1          | 0           | 100       | 142         | 8,234
2          | 100         | 200       | 87          | 13,050
3          | 200         | 300       | 61          | 15,300
...
```

---

## 7. Frequency Distribution — The True Histogram

**"How many users have exactly 1 order? 2 orders? 3-5 orders? 6+?"**

This is bucketing applied to a *computed* column (order count), not a raw column.

```sql
WITH order_counts AS (
  SELECT
    user_id,
    COUNT(DISTINCT order_id) AS num_orders
  FROM orders
  GROUP BY user_id
),
frequency_buckets AS (
  SELECT
    user_id,
    num_orders,
    CASE
      WHEN num_orders = 1  THEN '1 order'
      WHEN num_orders = 2  THEN '2 orders'
      WHEN num_orders BETWEEN 3 AND 5   THEN '3-5 orders'
      WHEN num_orders BETWEEN 6 AND 10  THEN '6-10 orders'
      WHEN num_orders > 10 THEN '10+ orders'
    END AS frequency_bucket
  FROM order_counts
)
SELECT
  frequency_bucket,
  COUNT(*)                                            AS user_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_users,
  SUM(COUNT(*)) OVER (
    ORDER BY MIN(num_orders)
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )                                                   AS cumulative_users
FROM frequency_buckets
GROUP BY frequency_bucket
ORDER BY MIN(num_orders);
```

### CTE-Level I/O Trace

**`order_counts`:**
```
user_id | num_orders
U1      | 1
U2      | 1
U3      | 2
U4      | 4
U5      | 4
U6      | 7
U7      | 1
U8      | 12
```

**`frequency_buckets`:**
```
user_id | num_orders | frequency_bucket
U1      | 1          | 1 order
U2      | 1          | 1 order
U3      | 2          | 2 orders
U4      | 4          | 3-5 orders
U5      | 4          | 3-5 orders
U6      | 7          | 6-10 orders
U7      | 1          | 1 order
U8      | 12         | 10+ orders
```

**Final output:**
```
frequency_bucket | user_count | pct_users | cumulative_users
1 order          | 3          | 37.50     | 3
2 orders         | 1          | 12.50     | 4
3-5 orders       | 2          | 25.00     | 6
6-10 orders      | 1          | 12.50     | 7
10+ orders       | 1          | 12.50     | 8
```

**Reading this:** 37.5% of users are one-and-done. Only 12.5% are highly engaged (10+ orders). Classic long-tail distribution you see in every e-commerce dataset.

---

## 8. Cumulative Distribution (CDF)

**"What percentage of orders are under $100? Under $200? Under $500?"**

The CDF answers: for a given value X, what fraction of the data falls at or below X.

```sql
WITH bucketed AS (
  SELECT
    order_id,
    amount,
    FLOOR(amount / 100) * 100 AS bucket_floor
  FROM orders
),
bucket_counts AS (
  SELECT
    bucket_floor,
    COUNT(*) AS bucket_count
  FROM bucketed
  GROUP BY bucket_floor
)
SELECT
  bucket_floor                              AS bucket_start,
  bucket_floor + 99                         AS bucket_end,
  bucket_count,
  SUM(bucket_count) OVER (
    ORDER BY bucket_floor
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )                                         AS cumulative_count,
  ROUND(
    SUM(bucket_count) OVER (
      ORDER BY bucket_floor
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) * 100.0 / SUM(bucket_count) OVER ()
  , 2)                                      AS cumulative_pct
FROM bucket_counts
ORDER BY bucket_floor;
```

### CTE-Level I/O Trace

**`bucket_counts`:**
```
bucket_floor | bucket_count
0            | 400
100          | 250
200          | 150
300          | 100
400          | 60
500          | 40
```

Total = 1000 orders.

**After window functions:**
```
bucket_start | bucket_end | bucket_count | cumulative_count | cumulative_pct
0            | 99         | 400          | 400              | 40.00
100          | 199        | 250          | 650              | 65.00
200          | 299        | 150          | 800              | 80.00
300          | 399        | 100          | 900              | 90.00
400          | 499        | 60           | 960              | 96.00
500          | 599        | 40           | 1000             | 100.00
```

**Reading this:** 80% of orders are under $300. 96% are under $500. The $500+ tail is only 4% of order volume — but may be disproportionate revenue. Pair this with a `SUM(amount)` cumulative to see revenue CDF.

---

## 9. Revenue Bucketing by Order Size

**"Do small, medium, or large orders drive most of our revenue?"**

```sql
WITH order_buckets AS (
  SELECT
    order_id,
    user_id,
    amount,
    CASE
      WHEN amount <  50   THEN 'Small  (<$50)'
      WHEN amount <  200  THEN 'Medium ($50-$199)'
      WHEN amount <  500  THEN 'Large  ($200-$499)'
      WHEN amount >= 500  THEN 'XL     ($500+)'
    END AS order_size
  FROM orders
)
SELECT
  order_size,
  COUNT(*)                                           AS order_count,
  COUNT(DISTINCT user_id)                            AS unique_buyers,
  ROUND(SUM(amount), 2)                              AS total_revenue,
  ROUND(AVG(amount), 2)                              AS avg_order_value,
  -- % of order volume
  ROUND(COUNT(*) * 100.0        / SUM(COUNT(*))        OVER (), 2) AS pct_orders,
  -- % of revenue
  ROUND(SUM(amount) * 100.0     / SUM(SUM(amount))     OVER (), 2) AS pct_revenue,
  -- avg orders per buyer in this tier
  ROUND(COUNT(*) * 1.0          / COUNT(DISTINCT user_id), 2)      AS orders_per_buyer
FROM order_buckets
GROUP BY order_size
ORDER BY MIN(amount);
```

### I/O Trace

**`order_buckets`:**
```
order_size     | order_count | total_revenue
Small (<$50)   | 600         | 15,000
Medium ($50-$) | 300         | 36,000
Large ($200-)  | 80          | 24,000
XL ($500+)     | 20          | 18,000
```

Total orders = 1000, total revenue = 93,000.

```
order_size     | order_count | revenue  | pct_orders | pct_revenue
Small          | 600         | 15,000   | 60.00%     | 16.13%
Medium         | 300         | 36,000   | 30.00%     | 38.71%
Large          | 80          | 24,000   | 8.00%      | 25.81%
XL ($500+)     | 20          | 18,000   | 2.00%      | 19.35%
```

**The "aha":** 2% of orders (XL) generate nearly 20% of revenue. Small orders are 60% of volume but only 16% of revenue. Classic power law. This drives decisions: optimize checkout for large orders, use promotions to convert medium buyers to large.

---

## 10. Age Cohort Analysis

**"Do younger or older users buy more, spend more, and retain better?"**

```sql
WITH user_stats AS (
  SELECT
    u.user_id,
    u.age,
    u.signup_date,
    COUNT(DISTINCT o.order_id)   AS total_orders,
    COALESCE(SUM(o.amount), 0)   AS total_spend,
    MAX(o.order_date)            AS last_order_date
  FROM users u
  LEFT JOIN orders o ON u.user_id = o.user_id
  GROUP BY u.user_id, u.age, u.signup_date
),
age_cohorts AS (
  SELECT
    user_id, age, total_orders, total_spend, last_order_date,
    CASE
      WHEN age < 25           THEN 'Gen Z    (<25)'
      WHEN age BETWEEN 25 AND 34 THEN 'Millennial (25-34)'
      WHEN age BETWEEN 35 AND 44 THEN 'Gen X Early (35-44)'
      WHEN age BETWEEN 45 AND 54 THEN 'Gen X Late (45-54)'
      WHEN age >= 55          THEN 'Boomer+ (55+)'
    END AS age_cohort,
    -- Is the user "active" — ordered in last 90 days?
    CASE WHEN last_order_date >= CURRENT_DATE - INTERVAL 90 DAY
         THEN 1 ELSE 0 END AS is_active
  FROM user_stats
)
SELECT
  age_cohort,
  COUNT(*)                                           AS users,
  ROUND(AVG(total_orders), 2)                        AS avg_orders,
  ROUND(AVG(total_spend), 2)                         AS avg_ltv,
  ROUND(SUM(total_spend), 2)                         AS total_revenue,
  ROUND(SUM(is_active) * 100.0 / COUNT(*), 2)        AS active_pct,
  ROUND(AVG(CASE WHEN total_orders > 0
            THEN total_spend / total_orders END), 2)  AS avg_order_value
FROM age_cohorts
GROUP BY age_cohort
ORDER BY MIN(age);
```

### CTE-Level I/O Trace

**`user_stats` (abbreviated):**
```
user_id | age | total_orders | total_spend | last_order_date
U1      | 22  | 3            | 85.00       | 2024-11-01
U2      | 29  | 8            | 340.00      | 2025-01-10
U3      | 44  | 15           | 1200.00     | 2025-01-15
U4      | 51  | 12           | 980.00      | 2024-08-01
U5      | 23  | 1            | 25.00       | 2024-06-01
```

**`age_cohorts`:**
```
user_id | age_cohort          | is_active
U1      | Gen Z (<25)         | 1   (ordered Nov 2024, within 90d of Jan 2025)
U2      | Millennial (25-34)  | 1
U3      | Gen X Early (35-44) | 1
U4      | Gen X Late (45-54)  | 0   (Aug 2024, > 90d ago)
U5      | Gen Z (<25)         | 0
```

**Final output:**
```
age_cohort           | users | avg_orders | avg_ltv | active_pct | avg_order_value
Gen Z (<25)          | 2     | 2.00       | 55.00   | 50.00      | 28.33
Millennial (25-34)   | 1     | 8.00       | 340.00  | 100.00     | 42.50
Gen X Early (35-44)  | 1     | 15.00      | 1200.00 | 100.00     | 80.00
Gen X Late (45-54)   | 1     | 12.00      | 980.00  | 0.00       | 81.67
```

---

## 11. Spend Tier Segmentation (RFM-style)

**Bucket users by total spend, then label them as Bronze/Silver/Gold/Platinum.**

```sql
WITH user_spend AS (
  SELECT
    user_id,
    SUM(amount)               AS total_spend,
    COUNT(DISTINCT order_id)  AS order_count,
    MAX(order_date)           AS last_order_date,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS days_since_last_order
  FROM orders
  GROUP BY user_id
),
spend_percentiles AS (
  SELECT
    user_id,
    total_spend,
    order_count,
    days_since_last_order,
    NTILE(100) OVER (ORDER BY total_spend ASC) AS spend_percentile
  FROM user_spend
),
tiers AS (
  SELECT
    user_id,
    total_spend,
    order_count,
    days_since_last_order,
    spend_percentile,
    CASE
      WHEN spend_percentile >= 95 THEN 'Platinum (Top 5%)'
      WHEN spend_percentile >= 75 THEN 'Gold (75th-95th)'
      WHEN spend_percentile >= 40 THEN 'Silver (40th-75th)'
      ELSE                             'Bronze (Bottom 40%)'
    END AS spend_tier
  FROM spend_percentiles
)
SELECT
  spend_tier,
  COUNT(*)                                             AS user_count,
  ROUND(MIN(total_spend), 2)                           AS min_spend,
  ROUND(MAX(total_spend), 2)                           AS max_spend,
  ROUND(AVG(total_spend), 2)                           AS avg_spend,
  ROUND(SUM(total_spend), 2)                           AS tier_revenue,
  ROUND(SUM(total_spend) * 100.0 /
        SUM(SUM(total_spend)) OVER (), 2)              AS pct_of_revenue,
  ROUND(AVG(days_since_last_order), 1)                 AS avg_days_since_order
FROM tiers
GROUP BY spend_tier
ORDER BY MIN(spend_percentile) DESC;
```

### CTE-Level I/O Trace (8 users)

**`user_spend`:**
```
user_id | total_spend | order_count | days_since_last_order
U1      | 20          | 1           | 200
U2      | 45          | 2           | 150
U3      | 80          | 3           | 90
U4      | 150         | 5           | 60
U5      | 200         | 6           | 45
U6      | 320         | 9           | 30
U7      | 500         | 12          | 15
U8      | 950         | 20          | 5
```

**`spend_percentiles`** — NTILE(100) on 8 users distributes percentiles:
```
U1 → percentile ~6
U2 → percentile ~19
U3 → percentile ~31
U4 → percentile ~44
U5 → percentile ~56
U6 → percentile ~69
U7 → percentile ~81
U8 → percentile ~94
```

**`tiers`:**
```
user_id | spend_tier
U1      | Bronze
U2      | Bronze
U3      | Bronze
U4      | Silver
U5      | Silver
U6      | Silver
U7      | Gold
U8      | Gold
```

> Note: With only 8 users, Platinum (top 5%) has no members. In real data with thousands of users NTILE(100) is precise.

**Final output:**
```
spend_tier  | users | min  | max  | avg    | tier_revenue | pct_revenue | avg_days
Gold        | 2     | 500  | 950  | 725    | 1,450        | 62.5%       | 10
Silver      | 3     | 150  | 320  | 223    | 670          | 28.9%       | 45
Bronze      | 3     | 20   | 80   | 48     | 145          | 6.3%        | 147
```

**The business reading:** Gold is 25% of users, 62.5% of revenue. Bronze is 37.5% of users, 6.3% of revenue. Gold users also ordered recently (avg 10 days ago) — they're your loyal high-value segment.

---

## 12. Bucket + Funnel (Conversion by Tier)

**"Do high spenders convert from trial to paid at higher rates?"**

This combines bucketing with funnel analysis — measuring a rate metric *inside* each bucket.

```sql
WITH user_spend AS (
  SELECT user_id, SUM(amount) AS total_spend
  FROM orders GROUP BY user_id
),
spend_tiers AS (
  SELECT
    user_id,
    total_spend,
    CASE
      WHEN total_spend <  100  THEN '1. Low (<$100)'
      WHEN total_spend <  500  THEN '2. Mid ($100-$499)'
      WHEN total_spend <  1000 THEN '3. High ($500-$999)'
      WHEN total_spend >= 1000 THEN '4. VIP ($1000+)'
    END AS tier
  FROM user_spend
),
funnel_events AS (
  SELECT
    st.tier,
    st.user_id,
    -- Binary flags per user per funnel stage
    MAX(CASE WHEN e.event_type = 'trial_start'   THEN 1 ELSE 0 END) AS started_trial,
    MAX(CASE WHEN e.event_type = 'trial_end'     THEN 1 ELSE 0 END) AS completed_trial,
    MAX(CASE WHEN e.event_type = 'subscription'  THEN 1 ELSE 0 END) AS converted
  FROM spend_tiers st
  LEFT JOIN events e ON st.user_id = e.user_id
  GROUP BY st.tier, st.user_id
)
SELECT
  tier,
  COUNT(*)                                              AS total_users,
  SUM(started_trial)                                    AS trialists,
  SUM(converted)                                        AS converted,
  -- Trial start rate
  ROUND(SUM(started_trial) * 100.0  / COUNT(*), 2)     AS trial_start_rate,
  -- Conversion rate among trialists
  ROUND(SUM(converted) * 100.0 /
        NULLIF(SUM(started_trial), 0), 2)               AS trial_to_paid_cvr
FROM funnel_events
GROUP BY tier
ORDER BY tier;
```

### I/O Trace

**`funnel_events` (per user, aggregated):**
```
tier       | user_id | started_trial | completed_trial | converted
1. Low     | U1      | 1             | 0               | 0
1. Low     | U2      | 1             | 1               | 0
2. Mid     | U3      | 1             | 1               | 1
2. Mid     | U4      | 1             | 1               | 1
3. High    | U5      | 1             | 1               | 1
```

**Final output:**
```
tier   | total_users | trialists | converted | trial_rate | cvr
1. Low | 2           | 2         | 0         | 100%       | 0.00%
2. Mid | 2           | 2         | 2         | 100%       | 100.00%
3. High| 1           | 1         | 1         | 100%       | 100.00%
```

**The pattern this reveals:** Higher-spend users convert at higher rates. This is common — high-spend users are already engaged. Hypothesis: nudge Mid-tier users who haven't converted with targeted offers.

---

## 13. Session Duration Distribution

**"How long are users spending per session, and where does engagement fall off?"**

```sql
WITH session_durations AS (
  SELECT
    session_id,
    user_id,
    TIMESTAMPDIFF(MINUTE, session_start, session_end) AS duration_minutes
  FROM sessions
  WHERE session_end IS NOT NULL     -- exclude abandoned/open sessions
),
bucketed AS (
  SELECT
    session_id,
    user_id,
    duration_minutes,
    CASE
      WHEN duration_minutes <   1 THEN '0. Bounce (<1m)'
      WHEN duration_minutes <   5 THEN '1. Short  (1-4m)'
      WHEN duration_minutes <  15 THEN '2. Medium (5-14m)'
      WHEN duration_minutes <  30 THEN '3. Long   (15-29m)'
      WHEN duration_minutes >= 30 THEN '4. Power  (30m+)'
    END AS duration_bucket
  FROM session_durations
)
SELECT
  duration_bucket,
  COUNT(*)                                             AS session_count,
  COUNT(DISTINCT user_id)                              AS unique_users,
  ROUND(AVG(duration_minutes), 1)                      AS avg_duration,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)  AS pct_sessions,
  -- Cumulative: what % of sessions are AT MOST this long?
  ROUND(SUM(COUNT(*)) OVER (
    ORDER BY MIN(duration_minutes)
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) * 100.0 / SUM(COUNT(*)) OVER (), 2)               AS cumulative_pct
FROM bucketed
GROUP BY duration_bucket
ORDER BY duration_bucket;
```

### I/O Trace

**`bucketed` (abbreviated):**
```
duration_bucket   | session_count
0. Bounce (<1m)   | 350
1. Short  (1-4m)  | 280
2. Medium (5-14m) | 200
3. Long  (15-29m) | 100
4. Power  (30m+)  | 70
```

Total = 1000 sessions.

**Final output:**
```
duration_bucket   | sessions | pct   | cumulative_pct
0. Bounce (<1m)   | 350      | 35.0% | 35.0%
1. Short  (1-4m)  | 280      | 28.0% | 63.0%
2. Medium (5-14m) | 200      | 20.0% | 83.0%
3. Long  (15-29m) | 100      | 10.0% | 93.0%
4. Power  (30m+)  | 70       | 7.0%  | 100.0%
```

**Reading:** 35% of sessions bounce immediately. 83% of sessions are under 15 minutes. Only 7% are power sessions — but these users are your most valuable. Cross this with conversion: do power-session users buy at higher rates?

---

## 14. Percentile Computation (P50, P90, P99)

**"What's the median order value? What's the 90th percentile? What does our worst 1% look like?"**

### Method 1 — PERCENTILE_CONT (Standard SQL)

```sql
SELECT
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount) AS p50_median,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS p75,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount) AS p90,
  PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) AS p95,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) AS p99
FROM orders;
```

### Method 2 — NTILE Approximation (when PERCENTILE_CONT unavailable)

```sql
WITH ranked AS (
  SELECT
    amount,
    NTILE(100) OVER (ORDER BY amount) AS percentile
  FROM orders
)
SELECT
  MAX(CASE WHEN percentile = 50 THEN amount END) AS p50,
  MAX(CASE WHEN percentile = 75 THEN amount END) AS p75,
  MAX(CASE WHEN percentile = 90 THEN amount END) AS p90,
  MAX(CASE WHEN percentile = 99 THEN amount END) AS p99
FROM ranked;
```

### Method 3 — Per-Bucket Percentiles (Most Useful for Analysis)

```sql
WITH age_groups AS (
  SELECT
    o.user_id,
    o.amount,
    CASE
      WHEN u.age < 25  THEN 'Gen Z'
      WHEN u.age < 35  THEN 'Millennial'
      WHEN u.age < 45  THEN 'Gen X'
      ELSE                  'Boomer+'
    END AS age_group
  FROM orders o
  JOIN users u ON o.user_id = u.user_id
)
SELECT
  age_group,
  COUNT(*)                                                          AS orders,
  ROUND(AVG(amount), 2)                                            AS avg_amount,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount)             AS p50,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount)             AS p90,
  PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount)             AS p99,
  MAX(amount)                                                       AS max_amount
FROM age_groups
GROUP BY age_group
ORDER BY age_group;
```

### I/O Trace

**Input amounts for Gen Z users:** 10, 15, 20, 25, 30, 50, 200

```
p50 = 25    (middle value)
p90 = 50    (90% of orders are ≤ 50)
p99 = 200   (99% of orders are ≤ 200 — the $200 is an outlier)
avg = 50    (mean is pulled up by the $200 outlier)
```

**Why P50 > AVG is a red flag:**  
If average = $50 but P50 = $25, the median buyer spends $25 but a few very large orders pull the average up. Reporting "average order value = $50" is misleading. Always report median alongside mean.

---

## 15. Outlier Detection via Buckets

**"Which users have unusually high session counts or spend? Flag them for review."**

```sql
WITH user_metrics AS (
  SELECT
    user_id,
    COUNT(DISTINCT order_id)  AS order_count,
    SUM(amount)               AS total_spend,
    AVG(amount)               AS avg_order
  FROM orders
  GROUP BY user_id
),
with_stats AS (
  SELECT
    user_id, order_count, total_spend, avg_order,
    AVG(total_spend)    OVER () AS mean_spend,
    STDDEV(total_spend) OVER () AS stddev_spend,
    NTILE(100)          OVER (ORDER BY total_spend ASC) AS spend_percentile
  FROM user_metrics
)
SELECT
  user_id,
  order_count,
  ROUND(total_spend, 2)      AS total_spend,
  ROUND(mean_spend, 2)       AS mean_spend,
  ROUND(stddev_spend, 2)     AS stddev_spend,
  spend_percentile,
  -- Z-score: how many std deviations from mean?
  ROUND((total_spend - mean_spend) / NULLIF(stddev_spend, 0), 2) AS z_score,
  CASE
    WHEN (total_spend - mean_spend) / NULLIF(stddev_spend, 0) >  3 THEN 'Outlier High'
    WHEN (total_spend - mean_spend) / NULLIF(stddev_spend, 0) < -3 THEN 'Outlier Low'
    WHEN spend_percentile >= 95 THEN 'Top 5%'
    WHEN spend_percentile <= 5  THEN 'Bottom 5%'
    ELSE 'Normal'
  END AS flag
FROM with_stats
ORDER BY z_score DESC;
```

### I/O Trace

Mean spend = $200, Stddev = $150.

```
user_id | total_spend | z_score | flag
U8      | 1200        | 6.67    | Outlier High  ← (1200-200)/150 = 6.67
U7      | 500         | 2.00    | Top 5%
U3      | 80          | -0.80   | Normal
U1      | 5           | -1.30   | Normal
```

**Z-score > 3:** Statistically unusual. Could be a fraudulent account, a B2B buyer accidentally in consumer dataset, or a data error. Flag for investigation.

---

## 16. Key Takeaways Cheatsheet

### Method Selection Guide

```
What are you trying to do?
│
├── Business-defined tiers ("under 25", "25-34", "premium vs standard")
│   └── CASE WHEN — readable, explicit, handles irregular breakpoints
│
├── Quick histogram, width is a round number you choose
│   └── FLOOR(value / width) * width — one-line, no CASE needed
│
├── Equal-population groups (top 25%, quartiles, deciles)
│   └── NTILE(n) OVER (ORDER BY value) — ensures balanced group sizes
│
├── Auto histogram between min/max (PostgreSQL/Snowflake/BigQuery)
│   └── WIDTH_BUCKET(value, min, max, n_buckets) — cleanest syntax
│
└── Exact percentile values (median, P90, P99)
    └── PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY value)
```

### The Universal Bucket Query Shape

```sql
WITH bucketed AS (
  SELECT
    id_column,
    value_column,
    <BUCKET EXPRESSION> AS bucket   -- CASE WHEN or FLOOR or NTILE
  FROM source_table
)
SELECT
  bucket,
  COUNT(*)                                             AS count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)  AS pct,
  MIN(value_column)                                    AS min_val,
  MAX(value_column)                                    AS max_val,
  ROUND(AVG(value_column), 2)                          AS avg_val,
  SUM(value_column)                                    AS total,
  -- Optional CDF
  SUM(COUNT(*)) OVER (
    ORDER BY MIN(value_column)
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  )                                                    AS cumulative_count
FROM bucketed
GROUP BY bucket
ORDER BY MIN(value_column);   -- never ORDER BY bucket label alphabetically
```

### Window Functions Used in Bucketing

| Function | What It Computes | Used For |
|----------|-----------------|----------|
| `SUM(COUNT(*)) OVER ()` | Grand total across all buckets | Percentage of total |
| `SUM(SUM(amount)) OVER ()` | Grand revenue total | Revenue percentage |
| `SUM(COUNT(*)) OVER (ORDER BY ... ROWS UNBOUNDED PRECEDING)` | Running total | CDF / cumulative % |
| `NTILE(n) OVER (ORDER BY value)` | Equal-population bucket number | Quartiles / deciles |
| `PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY value)` | Exact percentile value | P50, P90, P99 |
| `STDDEV(value) OVER ()` | Standard deviation across all rows | Z-score outlier detection |
| `AVG(value) OVER ()` | Grand mean | Z-score outlier detection |

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| `ORDER BY bucket_label` alphabetically | Use `ORDER BY MIN(value_column)` — sorts by actual data |
| `NTILE(n)` gives unequal bucket sizes | Expected when `COUNT(*) % n != 0`; SQL distributes remainder to first buckets |
| `FLOOR` on negative values | `FLOOR(-0.3) = -1` → bucket -10 to -1, not 0-9. Add check or shift values |
| NULL values landing in no bucket | Add `ELSE 'Unknown'` in CASE WHEN; filter NULLs or handle separately |
| Mean reported without median | Always pair AVG with P50; skewed distributions make mean misleading |
| `BETWEEN` inclusive on both ends | `BETWEEN 25 AND 34` includes 25 AND 34. Use `< 35` if you want <35 |
| Overlapping CASE WHEN ranges | SQL takes FIRST match; order from smallest to largest to avoid gaps |
| Dividing by zero in percentages | `NULLIF(SUM(COUNT(*)) OVER (), 0)` — rare but safe habit |

---


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
