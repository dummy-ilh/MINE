# Day 12 — ML Feature Engineering in SQL

---

## 1. Why SQL for Feature Engineering?

At FAANG, most ML pipelines start with SQL — features are computed in BigQuery/Hive/Spark SQL before being fed into models. Interviewers expect you to write production-grade feature pipelines directly in SQL.

```
Raw Tables → SQL Feature Engineering → Feature Store → Model Training
```

**Common asks in MLE interviews:**
- Compute lag features for time series
- Rolling aggregates as model inputs
- User behavior features (recency, frequency, diversity)
- Target encoding
- Train/test split logic in SQL

---

## 2. Lag Features

```sql
-- Table: user_purchases(user_id, purchase_date, amount, category)
-- Feature: amount spent in previous 1, 2, 3 purchases

SELECT
  user_id, purchase_date, amount,
  LAG(amount, 1) OVER (PARTITION BY user_id ORDER BY purchase_date)
    AS prev_purchase_1,
  LAG(amount, 2) OVER (PARTITION BY user_id ORDER BY purchase_date)
    AS prev_purchase_2,
  LAG(amount, 3) OVER (PARTITION BY user_id ORDER BY purchase_date)
    AS prev_purchase_3,
  -- Days since last purchase
  DATEDIFF(purchase_date,
    LAG(purchase_date, 1) OVER (PARTITION BY user_id ORDER BY purchase_date)
  ) AS days_since_last_purchase
FROM user_purchases;
```

```sql
-- Category of last 3 purchases (sequence features)
SELECT user_id, purchase_date, category,
  LAG(category, 1) OVER (PARTITION BY user_id ORDER BY purchase_date)
    AS prev_cat_1,
  LAG(category, 2) OVER (PARTITION BY user_id ORDER BY purchase_date)
    AS prev_cat_2,
  LAG(category, 3) OVER (PARTITION BY user_id ORDER BY purchase_date)
    AS prev_cat_3
FROM user_purchases;
```

---

## 3. Rolling Aggregates as Features

```sql
-- Rolling window features: last 7, 30, 90 days per user
-- Table: daily_user_activity(user_id, activity_date, events, revenue)

SELECT
  user_id, activity_date, events, revenue,

  -- 7-day rolling features
  SUM(events) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS events_last_7d,

  AVG(revenue) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS avg_revenue_7d,

  -- 30-day rolling features
  SUM(events) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ) AS events_last_30d,

  -- Trend: 7d vs 30d ratio (are they more active recently?)
  SUM(events) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) * 1.0 /
  NULLIF(SUM(events) OVER (
    PARTITION BY user_id ORDER BY activity_date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
  ), 0) AS recent_activity_ratio
FROM daily_user_activity;
```

---

## 4. User Behavior Features

```sql
-- Rich user feature vector for churn prediction model
WITH order_stats AS (
  SELECT
    user_id,
    COUNT(*)                                    AS total_orders,
    SUM(amount)                                 AS total_revenue,
    AVG(amount)                                 AS avg_order_value,
    MIN(amount)                                 AS min_order_value,
    MAX(amount)                                 AS max_order_value,
    STDDEV(amount)                              AS stddev_order_value,
    COUNT(DISTINCT category)                    AS category_diversity,
    COUNT(DISTINCT DATE_FORMAT(order_date,'%Y-%m')) AS active_months,
    DATEDIFF(MAX(order_date), MIN(order_date))  AS customer_lifespan_days,
    DATEDIFF(CURRENT_DATE, MAX(order_date))     AS recency_days,
    COUNT(*) / NULLIF(
      DATEDIFF(MAX(order_date), MIN(order_date)), 0
    )                                           AS order_frequency_per_day
  FROM orders
  GROUP BY user_id
),
session_stats AS (
  SELECT user_id,
    COUNT(DISTINCT session_id)                  AS total_sessions,
    AVG(session_duration_mins)                  AS avg_session_duration,
    AVG(pages_per_session)                      AS avg_pages_per_session
  FROM sessions
  GROUP BY user_id
)
SELECT
  o.*,
  s.total_sessions,
  s.avg_session_duration,
  s.avg_pages_per_session,
  -- Engagement ratio
  o.total_orders * 1.0 /
    NULLIF(s.total_sessions, 0)                 AS orders_per_session
FROM order_stats o
LEFT JOIN session_stats s ON o.user_id = s.user_id;
```

---

## 5. Target Encoding

Replace a categorical variable with the **mean of the target** for that category. Common for high-cardinality categoricals.

```sql
-- Encode 'city' with mean purchase amount (target = amount)
-- Use leave-one-out to prevent data leakage

WITH city_stats AS (
  SELECT city,
    COUNT(*)    AS city_count,
    SUM(amount) AS city_total
  FROM orders
  GROUP BY city
)
SELECT o.order_id, o.user_id, o.city, o.amount,
  -- Global mean (smoothing fallback)
  AVG(o.amount) OVER () AS global_mean,
  -- Leave-one-out target encoding
  (cs.city_total - o.amount) /
    NULLIF(cs.city_count - 1, 0)               AS city_loo_encoding,
  -- Smoothed target encoding (blend city mean with global mean)
  (cs.city_total + 10 * AVG(o.amount) OVER ()) /
    (cs.city_count + 10)                        AS city_smoothed_encoding
FROM orders o
JOIN city_stats cs ON o.city = cs.city;
```

> 💡 **Leave-one-out encoding** removes the current row from the group mean — prevents the model from seeing the target during training. Always use LOO for training data.

---

## 6. Train / Test Split in SQL

```sql
-- Method 1: Random split using RAND()
SELECT user_id, features,
  CASE WHEN RAND() < 0.8 THEN 'train' ELSE 'test' END AS split
FROM feature_table;
-- ⚠️ Not reproducible — RAND() changes each run

-- Method 2: Deterministic split using hash (reproducible)
SELECT user_id, features,
  CASE
    WHEN MOD(ABS(FARM_FINGERPRINT(CAST(user_id AS STRING))), 10) < 8
    THEN 'train' ELSE 'test'
  END AS split
FROM feature_table;
-- FARM_FINGERPRINT (BigQuery) — same user always gets same split

-- MySQL equivalent
SELECT user_id, features,
  CASE
    WHEN MOD(CRC32(user_id), 10) < 8 THEN 'train'
    ELSE 'test'
  END AS split
FROM feature_table;
```

```sql
-- Temporal split (most correct for time series)
-- Train: before cutoff | Test: after cutoff
SELECT user_id, feature_date, features,
  CASE
    WHEN feature_date < '2025-10-01' THEN 'train'
    WHEN feature_date < '2025-11-01' THEN 'validation'
    ELSE 'test'
  END AS split
FROM feature_table;
```

> 💡 **Always use temporal splits for time-series data** — random splits cause data leakage when future data ends up in training set.

---

## 7. Feature Interaction & Derived Features

```sql
-- Interaction features (combine two features into one)
SELECT user_id,
  -- Price sensitivity: how much does user deviate from avg?
  amount / NULLIF(AVG(amount) OVER (PARTITION BY user_id), 0)
    AS amount_vs_personal_avg,

  -- Cross features
  CONCAT(device_type, '_', country)
    AS device_country,

  -- Binned features (continuous → categorical)
  CASE
    WHEN age BETWEEN 18 AND 24 THEN '18-24'
    WHEN age BETWEEN 25 AND 34 THEN '25-34'
    WHEN age BETWEEN 35 AND 44 THEN '35-44'
    ELSE '45+'
  END AS age_bucket,

  -- Boolean flags
  CASE WHEN recency_days <= 7  THEN 1 ELSE 0 END AS active_last_7d,
  CASE WHEN recency_days <= 30 THEN 1 ELSE 0 END AS active_last_30d,
  CASE WHEN total_orders >= 10 THEN 1 ELSE 0 END AS is_power_user
FROM user_features;
```

---

## 8. Point-in-Time Correct Features (No Data Leakage)

The hardest part of ML feature engineering — only use data available **at prediction time.**

```sql
-- For each order, compute features using ONLY past data
-- (no future information leaks into features)

SELECT
  o.order_id,
  o.user_id,
  o.order_date,
  o.amount AS target,

  -- Features: only look at PRIOR orders (exclude current)
  COUNT(prev.order_id)              AS prior_order_count,
  AVG(prev.amount)                  AS prior_avg_amount,
  SUM(prev.amount)                  AS prior_total_spend,
  DATEDIFF(o.order_date, MAX(prev.order_date)) AS days_since_last_order,
  MAX(prev.amount)                  AS prior_max_amount

FROM orders o
LEFT JOIN orders prev
  ON  prev.user_id   = o.user_id
  AND prev.order_date < o.order_date   -- ← only PRIOR orders
GROUP BY o.order_id, o.user_id, o.order_date, o.amount;
```

> ⚠️ This is the most common **data leakage mistake** in MLE interviews. Always use `< order_date` not `<= order_date` for point-in-time features.

---

## 9. Full Feature Pipeline Example

```sql
-- Complete churn prediction feature set
WITH base AS (
  SELECT user_id, signup_date, plan_type, country
  FROM users
  WHERE is_active = 1
),
order_features AS (
  SELECT user_id,
    COUNT(*)                                    AS total_orders,
    SUM(amount)                                 AS ltv,
    AVG(amount)                                 AS avg_order_value,
    DATEDIFF(CURRENT_DATE, MAX(order_date))     AS recency_days,
    DATEDIFF(MAX(order_date), MIN(order_date))  AS order_span_days,
    COUNT(DISTINCT category)                    AS category_diversity,
    SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 30 DAY
             THEN amount ELSE 0 END)            AS revenue_last_30d,
    SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 90 DAY
             THEN amount ELSE 0 END)            AS revenue_last_90d
  FROM orders GROUP BY user_id
),
event_features AS (
  SELECT user_id,
    COUNT(*)                                    AS total_events,
    COUNT(DISTINCT event_type)                  AS event_diversity,
    DATEDIFF(CURRENT_DATE, MAX(event_date))     AS days_since_last_event,
    SUM(CASE WHEN event_date >= CURRENT_DATE - INTERVAL 7 DAY
             THEN 1 ELSE 0 END)                 AS events_last_7d
  FROM events GROUP BY user_id
),
support_features AS (
  SELECT user_id,
    COUNT(*)                                    AS support_tickets,
    AVG(CASE WHEN resolved = 1 THEN 1.0 ELSE 0 END) AS resolution_rate
  FROM support_tickets GROUP BY user_id
)
SELECT
  b.user_id, b.plan_type, b.country,
  DATEDIFF(CURRENT_DATE, b.signup_date)         AS account_age_days,
  o.total_orders, o.ltv, o.avg_order_value,
  o.recency_days, o.category_diversity,
  o.revenue_last_30d, o.revenue_last_90d,
  -- Trend feature
  o.revenue_last_30d / NULLIF(o.revenue_last_90d, 0) AS revenue_trend,
  e.total_events, e.event_diversity,
  e.days_since_last_event, e.events_last_7d,
  COALESCE(s.support_tickets, 0)                AS support_tickets,
  COALESCE(s.resolution_rate, 1.0)              AS resolution_rate
FROM base b
LEFT JOIN order_features  o ON b.user_id = o.user_id
LEFT JOIN event_features  e ON b.user_id = e.user_id
LEFT JOIN support_features s ON b.user_id = s.user_id;
```

---

## Summary Cheatsheet

| Feature Type | SQL Pattern |
|---|---|
| Lag features | `LAG(col, n) OVER (PARTITION BY user ORDER BY date)` |
| Rolling aggregates | `SUM/AVG OVER (ROWS BETWEEN N PRECEDING AND CURRENT ROW)` |
| Recency | `DATEDIFF(CURRENT_DATE, MAX(event_date))` |
| Frequency | `COUNT(*) GROUP BY user_id` |
| Diversity | `COUNT(DISTINCT category)` |
| Target encoding | `(group_sum - row_val) / (group_count - 1)` |
| Train/test split | `MOD(CRC32(user_id), 10) < 8` |
| Point-in-time | `JOIN ON prior.date < current.date` |
| Trend feature | `recent_window / longer_window` |
| Boolean flag | `CASE WHEN condition THEN 1 ELSE 0 END` |

---

### 🟢 Q1 — Easy
> Table: `orders(order_id, user_id, amount, category, order_date)`
>
> For each order compute: **previous order amount**, **days since previous order**, and a flag **`is_repeat_category`** (1 if same category as previous order, else 0).

---

### 🟡 Q2 — Medium
> Same table plus `users(user_id, signup_date, country, plan_type)`
>
> Build a **user feature vector** for a propensity-to-upgrade model. Include: account age, total orders, avg order value, recency, revenue last 30d vs 90d trend, category diversity, and a **deterministic train/test split** (80/20 using CRC32).

---

### 🔴 Q3 — Hard
> Tables: `orders(order_id, user_id, amount, order_date)`, `users(user_id, city, signup_date)`
>
> Build a **point-in-time correct feature table** for predicting next order amount. For each order, features must use **only data available before that order date**. Include: prior order count, prior avg amount, prior max amount, days since last order, and **smoothed city-level target encoding** of avg amount (using only prior orders).
### Q1 ✅
```sql
SELECT
  order_id, user_id, amount, category, order_date,
  LAG(amount) OVER (
    PARTITION BY user_id ORDER BY order_date
  ) AS prev_order_amount,
  DATEDIFF(order_date,
    LAG(order_date) OVER (
      PARTITION BY user_id ORDER BY order_date
    )
  ) AS days_since_prev_order,
  CASE
    WHEN category = LAG(category) OVER (
      PARTITION BY user_id ORDER BY order_date
    ) THEN 1 ELSE 0
  END AS is_repeat_category
FROM orders;
```

---

### Q2 ✅
```sql
WITH order_features AS (
  SELECT user_id,
    COUNT(*)                                     AS total_orders,
    AVG(amount)                                  AS avg_order_value,
    DATEDIFF(CURRENT_DATE, MAX(order_date))      AS recency_days,
    COUNT(DISTINCT category)                     AS category_diversity,
    SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 30 DAY
             THEN amount ELSE 0 END)             AS revenue_last_30d,
    SUM(CASE WHEN order_date >= CURRENT_DATE - INTERVAL 90 DAY
             THEN amount ELSE 0 END)             AS revenue_last_90d
  FROM orders
  GROUP BY user_id
)
SELECT
  u.user_id, u.country, u.plan_type,
  DATEDIFF(CURRENT_DATE, u.signup_date)          AS account_age_days,
  o.total_orders, o.avg_order_value,
  o.recency_days, o.category_diversity,
  o.revenue_last_30d, o.revenue_last_90d,
  ROUND(o.revenue_last_30d /
    NULLIF(o.revenue_last_90d, 0), 4)            AS revenue_trend,
  CASE
    WHEN MOD(CRC32(u.user_id), 10) < 8
    THEN 'train' ELSE 'test'
  END AS split
FROM users u
LEFT JOIN order_features o ON u.user_id = o.user_id;
```

---

### Q3 ✅
```sql
WITH city_prior AS (
  -- City-level stats using ONLY prior orders (point-in-time correct)
  SELECT
    o.order_id,
    o.user_id,
    o.amount,
    o.order_date,
    u.city,
    -- Prior city stats excluding current order
    SUM(prev.amount)   AS city_prior_total,
    COUNT(prev.amount) AS city_prior_count
  FROM orders o
  JOIN users u ON o.user_id = u.user_id
  LEFT JOIN orders prev
    JOIN users pu ON prev.user_id = pu.user_id
    ON  pu.city        = u.city
    AND prev.order_date < o.order_date   -- only prior orders
  GROUP BY o.order_id, o.user_id, o.amount, o.order_date, u.city
),
global_mean AS (
  SELECT AVG(amount) AS global_avg FROM orders
),
prior_user AS (
  SELECT
    o.order_id,
    COUNT(prev.order_id)              AS prior_order_count,
    AVG(prev.amount)                  AS prior_avg_amount,
    MAX(prev.amount)                  AS prior_max_amount,
    DATEDIFF(o.order_date,
      MAX(prev.order_date))           AS days_since_last_order
  FROM orders o
  LEFT JOIN orders prev
    ON  prev.user_id    = o.user_id
    AND prev.order_date < o.order_date
  GROUP BY o.order_id, o.order_date
)
SELECT
  cp.order_id, cp.user_id, cp.order_date,
  cp.amount                           AS target,
  pu.prior_order_count,
  ROUND(pu.prior_avg_amount, 2)       AS prior_avg_amount,
  ROUND(pu.prior_max_amount, 2)       AS prior_max_amount,
  pu.days_since_last_order,
  -- Smoothed city encoding (k=10 smoothing)
  ROUND(
    (COALESCE(cp.city_prior_total, 0) + 10 * gm.global_avg) /
    (COALESCE(cp.city_prior_count, 0) + 10), 2
  ) AS city_smoothed_encoding
FROM city_prior cp
JOIN prior_user pu ON cp.order_id = pu.order_id
CROSS JOIN global_mean gm
ORDER BY cp.user_id, cp.order_date;
```

> 💡 The `k=10` smoothing constant blends the city mean toward the global mean when city sample size is small — standard Bayesian smoothing for target encoding.
