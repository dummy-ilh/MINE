# Day 17 — Cohort Analysis & Retention in SQL
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. What is Cohort Analysis
2. Building the Cohort Base
3. Classic N-Month Retention
4. Retention Heatmap (Pivot Format)
5. N-Day Retention (Daily Granularity)
6. Rolling vs Exact Retention
7. Churn Analysis
8. LTV by Cohort
9. Retention Curve Smoothing
10. FAANG Weekly Cohort Pattern

---

## 1. What is Cohort Analysis

```
Cohort = users who share a common starting event in the same time period

Jan cohort: users who signed up in January
→ Track: of Jan cohort, how many returned in month 1? month 2? month 6?
```

---

## 2. Building the Cohort Base

```sql
WITH first_activity AS (
  SELECT user_id,
    MIN(activity_date)                       AS first_date,
    DATE_FORMAT(MIN(activity_date), '%Y-%m') AS cohort_month
  FROM user_activity GROUP BY user_id
)
SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
FROM first_activity
GROUP BY cohort_month ORDER BY cohort_month;
```

---

## 3. Classic N-Month Retention

```sql
WITH first_activity AS (
  SELECT user_id,
    MIN(DATE_FORMAT(activity_date, '%Y-%m-01')) AS cohort_date
  FROM user_activity GROUP BY user_id
),
cohort_activity AS (
  SELECT f.user_id, f.cohort_date,
    PERIOD_DIFF(
      DATE_FORMAT(a.activity_date, '%Y%m'),
      DATE_FORMAT(f.cohort_date,   '%Y%m')
    ) AS month_number
  FROM first_activity f
  JOIN user_activity a ON f.user_id = a.user_id
),
cohort_size AS (
  SELECT cohort_date, COUNT(DISTINCT user_id) AS total_users
  FROM first_activity GROUP BY cohort_date
)
SELECT ca.cohort_date, cs.total_users AS cohort_size,
  ca.month_number,
  COUNT(DISTINCT ca.user_id) AS retained_users,
  ROUND(COUNT(DISTINCT ca.user_id) * 100.0 /
        cs.total_users, 2)   AS retention_rate
FROM cohort_activity ca
JOIN cohort_size cs ON ca.cohort_date = cs.cohort_date
GROUP BY ca.cohort_date, cs.total_users, ca.month_number
ORDER BY ca.cohort_date, ca.month_number;
```

---

## 4. Retention Heatmap (Pivot)

```sql
-- One row per cohort, columns = months 0-6
WITH first_activity AS (
  SELECT user_id,
    MIN(DATE_FORMAT(activity_date, '%Y-%m-01')) AS cohort_date
  FROM user_activity GROUP BY user_id
),
cohort_activity AS (
  SELECT f.user_id, f.cohort_date,
    PERIOD_DIFF(
      DATE_FORMAT(a.activity_date, '%Y%m'),
      DATE_FORMAT(f.cohort_date,   '%Y%m')
    ) AS month_num
  FROM first_activity f JOIN user_activity a ON f.user_id = a.user_id
),
cohort_size AS (
  SELECT cohort_date, COUNT(DISTINCT user_id) AS total_users
  FROM first_activity GROUP BY cohort_date
),
retention AS (
  SELECT ca.cohort_date, ca.month_num,
    COUNT(DISTINCT ca.user_id) AS retained
  FROM cohort_activity ca GROUP BY ca.cohort_date, ca.month_num
)
SELECT r.cohort_date, cs.total_users AS cohort_size,
  MAX(CASE WHEN month_num=0 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m0,
  MAX(CASE WHEN month_num=1 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m1,
  MAX(CASE WHEN month_num=2 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m2,
  MAX(CASE WHEN month_num=3 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m3,
  MAX(CASE WHEN month_num=4 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m4,
  MAX(CASE WHEN month_num=5 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m5,
  MAX(CASE WHEN month_num=6 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m6
FROM retention r
JOIN cohort_size cs ON r.cohort_date = cs.cohort_date
GROUP BY r.cohort_date, cs.total_users
ORDER BY r.cohort_date;
-- M0 always = 100% by definition
```

---

## 5. N-Day Retention (Daily Granularity)

```sql
WITH first_seen AS (
  SELECT user_id, MIN(event_date) AS first_date
  FROM events GROUP BY user_id
)
SELECT f.first_date, COUNT(DISTINCT f.user_id) AS cohort_size,
  ROUND(COUNT(DISTINCT CASE WHEN DATEDIFF(a.event_date, f.first_date) = 1
    THEN a.user_id END) * 100.0 / COUNT(DISTINCT f.user_id), 2) AS d1_retention,
  ROUND(COUNT(DISTINCT CASE WHEN DATEDIFF(a.event_date, f.first_date) = 7
    THEN a.user_id END) * 100.0 / COUNT(DISTINCT f.user_id), 2) AS d7_retention,
  ROUND(COUNT(DISTINCT CASE WHEN DATEDIFF(a.event_date, f.first_date) = 14
    THEN a.user_id END) * 100.0 / COUNT(DISTINCT f.user_id), 2) AS d14_retention,
  ROUND(COUNT(DISTINCT CASE WHEN DATEDIFF(a.event_date, f.first_date) = 30
    THEN a.user_id END) * 100.0 / COUNT(DISTINCT f.user_id), 2) AS d30_retention
FROM first_seen f
JOIN events a ON f.user_id = a.user_id
GROUP BY f.first_date ORDER BY f.first_date;
```

---

## 6. Rolling vs Exact Retention

```sql
WITH first_seen AS (
  SELECT user_id, MIN(event_date) AS first_date FROM events GROUP BY user_id
)
SELECT f.first_date, COUNT(DISTINCT f.user_id) AS cohort_size,
  -- Exact: active on precisely day 7
  COUNT(DISTINCT CASE WHEN DATEDIFF(a.event_date, f.first_date) = 7
    THEN a.user_id END) AS exact_d7,
  -- Rolling: active at any point within days 1-7
  COUNT(DISTINCT CASE WHEN DATEDIFF(a.event_date, f.first_date) BETWEEN 1 AND 7
    THEN a.user_id END) AS rolling_d7,
  -- Rolling 30
  COUNT(DISTINCT CASE WHEN DATEDIFF(a.event_date, f.first_date) BETWEEN 1 AND 30
    THEN a.user_id END) AS rolling_d30
FROM first_seen f
JOIN events a ON f.user_id = a.user_id
GROUP BY f.first_date ORDER BY f.first_date;
```

> 💡 Rolling retention is always >= exact retention. Use exact for engagement metrics, rolling for churn risk.

---

## 7. Churn Analysis

```sql
WITH last_activity AS (
  SELECT user_id,
    MAX(activity_date) AS last_active,
    DATEDIFF(CURRENT_DATE, MAX(activity_date)) AS days_inactive
  FROM user_activity GROUP BY user_id
)
SELECT
  CASE
    WHEN days_inactive <= 7  THEN 'Active (0-7d)'
    WHEN days_inactive <= 30 THEN 'At Risk (8-30d)'
    WHEN days_inactive <= 90 THEN 'Churning (31-90d)'
    ELSE 'Churned (90d+)'
  END AS user_status,
  COUNT(*) AS users,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM last_activity
GROUP BY 1 ORDER BY MIN(days_inactive);
```

---

## 8. LTV by Cohort

```sql
WITH first_order AS (
  SELECT user_id,
    MIN(DATE_FORMAT(order_date, '%Y-%m-01')) AS cohort_date
  FROM orders GROUP BY user_id
),
cohort_revenue AS (
  SELECT f.user_id, f.cohort_date,
    PERIOD_DIFF(
      DATE_FORMAT(o.order_date, '%Y%m'),
      DATE_FORMAT(f.cohort_date, '%Y%m')
    ) AS month_number, o.amount
  FROM first_order f JOIN orders o ON f.user_id = o.user_id
),
cohort_size AS (
  SELECT cohort_date, COUNT(DISTINCT user_id) AS users
  FROM first_order GROUP BY cohort_date
),
monthly_ltv AS (
  SELECT cohort_date, month_number, SUM(amount) AS monthly_revenue
  FROM cohort_revenue GROUP BY cohort_date, month_number
)
SELECT m.cohort_date, cs.users AS cohort_size, m.month_number,
  ROUND(SUM(m.monthly_revenue) OVER (
    PARTITION BY m.cohort_date ORDER BY m.month_number
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) / cs.users, 2) AS cumulative_ltv_per_user
FROM monthly_ltv m
JOIN cohort_size cs ON m.cohort_date = cs.cohort_date
ORDER BY m.cohort_date, m.month_number;
```

---

## 9. Average Retention Curve

```sql
-- Average retention across all cohorts
WITH first_activity AS (
  SELECT user_id,
    MIN(DATE_FORMAT(activity_date, '%Y-%m-01')) AS cohort_date
  FROM user_activity GROUP BY user_id
),
retention AS (
  SELECT ca.cohort_date, ca.month_num,
    COUNT(DISTINCT ca.user_id) * 1.0 / cs.total_users AS retention_rate
  FROM (
    SELECT f.user_id, f.cohort_date,
      PERIOD_DIFF(DATE_FORMAT(a.activity_date,'%Y%m'),
                  DATE_FORMAT(f.cohort_date,'%Y%m')) AS month_num
    FROM first_activity f JOIN user_activity a ON f.user_id = a.user_id
  ) ca
  JOIN (SELECT cohort_date, COUNT(DISTINCT user_id) AS total_users
        FROM first_activity GROUP BY cohort_date) cs
    ON ca.cohort_date = cs.cohort_date
  GROUP BY ca.cohort_date, ca.month_num, cs.total_users
)
SELECT month_num,
  ROUND(AVG(retention_rate) * 100, 2) AS avg_retention_pct,
  ROUND(MIN(retention_rate) * 100, 2) AS min_retention_pct,
  ROUND(MAX(retention_rate) * 100, 2) AS max_retention_pct
FROM retention GROUP BY month_num ORDER BY month_num;
```

---

## 10. Weekly Cohort (Meta/Facebook style)

```sql
WITH first_week AS (
  SELECT user_id,
    DATE_SUB(MIN(event_date),
      INTERVAL DAYOFWEEK(MIN(event_date))-1 DAY) AS cohort_week
  FROM events GROUP BY user_id
),
weekly_activity AS (
  SELECT f.user_id, f.cohort_week,
    DATE_SUB(a.event_date,
      INTERVAL DAYOFWEEK(a.event_date)-1 DAY) AS activity_week
  FROM first_week f JOIN events a ON f.user_id = a.user_id
),
cohort_size AS (
  SELECT cohort_week, COUNT(DISTINCT user_id) AS users
  FROM first_week GROUP BY cohort_week
)
SELECT wa.cohort_week, cs.users AS cohort_size,
  DATEDIFF(wa.activity_week, wa.cohort_week) / 7 AS week_number,
  COUNT(DISTINCT wa.user_id) AS retained_users,
  ROUND(COUNT(DISTINCT wa.user_id) * 100.0 / cs.users, 2) AS retention_pct
FROM weekly_activity wa
JOIN cohort_size cs ON wa.cohort_week = cs.cohort_week
GROUP BY wa.cohort_week, cs.users,
         DATEDIFF(wa.activity_week, wa.cohort_week) / 7
ORDER BY wa.cohort_week, week_number;
```

---

## Practice Questions
🟢 Q1 — Easy

Table: user_logins(user_id, login_date)
Assign each user to their signup cohort month (month of first login). Return cohort_month and cohort_size, sorted chronologically.


🟡 Q2 — Medium

Same table.
Build a monthly retention table: for each cohort month, show retention rate at months 0, 1, 2, and 3. Output one row per cohort with columns cohort_month, cohort_size, m0, m1, m2, m3.


🔴 Q3 — Hard

Tables: users(user_id, signup_date), orders(order_id, user_id, amount, order_date)
For each monthly signup cohort, compute:

Cohort size
Month 1, 2, 3 purchase retention (% who made any purchase that month)
Cumulative LTV per user at months 1, 2, 3

Return one row per cohort.
### Q1 — Easy ✅
Signup cohort month + cohort size.

```sql
WITH first_login AS (
  SELECT user_id,
    DATE_FORMAT(MIN(login_date), '%Y-%m') AS cohort_month
  FROM user_logins GROUP BY user_id
)
SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
FROM first_login
GROUP BY cohort_month ORDER BY cohort_month;
```

### Q2 — Medium ✅
Monthly retention pivot: m0, m1, m2, m3.

```sql
WITH first_login AS (
  SELECT user_id,
    MIN(DATE_FORMAT(login_date, '%Y-%m-01')) AS cohort_date
  FROM user_logins GROUP BY user_id
),
cohort_activity AS (
  SELECT f.user_id, f.cohort_date,
    PERIOD_DIFF(
      DATE_FORMAT(l.login_date, '%Y%m'),
      DATE_FORMAT(f.cohort_date, '%Y%m')
    ) AS month_num
  FROM first_login f JOIN user_logins l ON f.user_id = l.user_id
),
cohort_size AS (
  SELECT cohort_date, COUNT(DISTINCT user_id) AS total_users
  FROM first_login GROUP BY cohort_date
),
retention AS (
  SELECT cohort_date, month_num,
    COUNT(DISTINCT user_id) AS retained
  FROM cohort_activity GROUP BY cohort_date, month_num
)
SELECT DATE_FORMAT(r.cohort_date, '%Y-%m') AS cohort_month,
  cs.total_users AS cohort_size,
  MAX(CASE WHEN month_num=0 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m0,
  MAX(CASE WHEN month_num=1 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m1,
  MAX(CASE WHEN month_num=2 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m2,
  MAX(CASE WHEN month_num=3 THEN ROUND(retained*100.0/cs.total_users,1) END) AS m3
FROM retention r
JOIN cohort_size cs ON r.cohort_date = cs.cohort_date
GROUP BY r.cohort_date, cs.total_users ORDER BY r.cohort_date;
```

### Q3 — Hard ✅
Purchase retention % + cumulative LTV per user at months 1–3.

```sql
WITH cohort_base AS (
  SELECT user_id,
    MIN(DATE_FORMAT(signup_date, '%Y-%m-01')) AS cohort_date
  FROM users GROUP BY user_id
),
cohort_orders AS (
  SELECT c.user_id, c.cohort_date,
    PERIOD_DIFF(
      DATE_FORMAT(o.order_date, '%Y%m'),
      DATE_FORMAT(c.cohort_date, '%Y%m')
    ) AS month_num, o.amount
  FROM cohort_base c JOIN orders o ON c.user_id = o.user_id
),
cohort_size AS (
  SELECT cohort_date, COUNT(DISTINCT user_id) AS total_users
  FROM cohort_base GROUP BY cohort_date
),
monthly_stats AS (
  SELECT cohort_date, month_num,
    COUNT(DISTINCT user_id) AS retained_buyers,
    SUM(amount)             AS monthly_revenue
  FROM cohort_orders GROUP BY cohort_date, month_num
),
cumulative AS (
  SELECT cohort_date, month_num, retained_buyers, monthly_revenue,
    SUM(monthly_revenue) OVER (
      PARTITION BY cohort_date ORDER BY month_num
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_revenue
  FROM monthly_stats
)
SELECT DATE_FORMAT(c.cohort_date, '%Y-%m') AS cohort_month,
  cs.total_users AS cohort_size,
  MAX(CASE WHEN month_num=1 THEN ROUND(retained_buyers*100.0/cs.total_users,2) END) AS m1_retention,
  MAX(CASE WHEN month_num=2 THEN ROUND(retained_buyers*100.0/cs.total_users,2) END) AS m2_retention,
  MAX(CASE WHEN month_num=3 THEN ROUND(retained_buyers*100.0/cs.total_users,2) END) AS m3_retention,
  MAX(CASE WHEN month_num=1 THEN ROUND(cumulative_revenue/cs.total_users,2) END)    AS ltv_m1,
  MAX(CASE WHEN month_num=2 THEN ROUND(cumulative_revenue/cs.total_users,2) END)    AS ltv_m2,
  MAX(CASE WHEN month_num=3 THEN ROUND(cumulative_revenue/cs.total_users,2) END)    AS ltv_m3
FROM cumulative c
JOIN cohort_size cs ON c.cohort_date = cs.cohort_date
GROUP BY c.cohort_date, cs.total_users ORDER BY c.cohort_date;
```

---

## Key Takeaways

- **Cohort base** → `MIN(event_date)` per user, truncate to month/week
- **Month number** → `PERIOD_DIFF(activity_ym, cohort_ym)`
- **Retention pivot** → `MAX(CASE WHEN month_num = N THEN rate END)`
- **M0 = 100%** always — users are active in their own signup period
- **Exact D-N** → `DATEDIFF = N` | **Rolling D-N** → `DATEDIFF BETWEEN 1 AND N`
- **LTV** → cumulative revenue SUM OVER window / cohort size
- **Churn buckets** → segment by days_inactive thresholds
- **Weekly cohorts** → truncate date to Monday with `DATE_SUB(date, INTERVAL DAYOFWEEK-1 DAY)`
- **Avg retention curve** → `AVG(retention_rate)` grouped by month_num across cohorts

---

*Day 17 complete — 13 days to go 🚀*
