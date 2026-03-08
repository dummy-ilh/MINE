# Day 22 — Complex Multi-Step Queries
**FAANG SQL 30-Day Prep**

---

## The Multi-Step Mindset

```
Complex FAANG questions combine 3–5 techniques in one query.

Approach:
1. Understand the output — what does the final table look like?
2. Work backwards — what intermediate tables do you need?
3. Build CTEs bottom-up — one CTE per logical step
4. Validate at each step — would this CTE produce the right shape?
```

---

## Pattern 1 — Rank Within Group + Compare to Aggregate

```sql
WITH dept_stats AS (
  SELECT department,
    AVG(salary) AS dept_avg, STDDEV(salary) AS dept_std
  FROM employees GROUP BY department
),
ranked AS (
  SELECT e.*,
    ds.dept_avg,
    ROUND((e.salary - ds.dept_avg)*100.0/NULLIF(ds.dept_avg,0),2) AS pct_above_avg,
    RANK() OVER (PARTITION BY e.department ORDER BY e.salary DESC) AS dept_rank,
    ROUND((e.salary - ds.dept_avg)/NULLIF(ds.dept_std,0),2)       AS z_score
  FROM employees e JOIN dept_stats ds ON e.department = ds.department
)
SELECT name, department, salary, dept_avg, pct_above_avg, dept_rank, z_score
FROM ranked WHERE salary > dept_avg
ORDER BY department, dept_rank;
```

---

## Pattern 2 — First + Last + Change

```sql
WITH order_stats AS (
  SELECT user_id, COUNT(*) AS total_orders,
    SUM(amount) AS total_spend, AVG(amount) AS avg_order_value
  FROM orders GROUP BY user_id
),
first_last AS (
  SELECT DISTINCT user_id,
    FIRST_VALUE(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_amount,
    LAST_VALUE(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_amount
  FROM orders
)
SELECT os.user_id, os.total_orders, os.total_spend,
  fl.first_amount, fl.last_amount,
  ROUND((fl.last_amount - fl.first_amount)*100.0/
        NULLIF(fl.first_amount,0),2) AS spend_change_pct,
  CASE
    WHEN fl.last_amount > fl.first_amount THEN 'Increasing'
    WHEN fl.last_amount < fl.first_amount THEN 'Decreasing'
    ELSE 'Flat'
  END AS spend_trend
FROM order_stats os JOIN first_last fl ON os.user_id = fl.user_id
ORDER BY os.total_spend DESC;
```

---

## Pattern 3 — Session Boundaries (Running State Machine)

```sql
-- Gap > 30 min = new session
WITH event_gaps AS (
  SELECT user_id, event_time, event_type,
    TIMESTAMPDIFF(MINUTE,
      LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time),
      event_time) AS gap_minutes
  FROM events
),
session_flags AS (
  SELECT *,
    CASE WHEN gap_minutes > 30 OR gap_minutes IS NULL
         THEN 1 ELSE 0 END AS is_new_session
  FROM event_gaps
),
with_session_id AS (
  SELECT *,
    SUM(is_new_session) OVER (
      PARTITION BY user_id ORDER BY event_time
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS session_num
  FROM session_flags
)
SELECT user_id, session_num,
  MIN(event_time)                               AS session_start,
  MAX(event_time)                               AS session_end,
  TIMESTAMPDIFF(MINUTE, MIN(event_time), MAX(event_time)) AS duration_mins,
  COUNT(*)                                      AS event_count,
  MAX(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS converted
FROM with_session_id
GROUP BY user_id, session_num
HAVING COUNT(*) >= 2
ORDER BY user_id, session_num;
```

> 💡 **Session ID trick**: cumulative SUM of `is_new_session` flags = unique session counter per user.

---

## Pattern 4 — Multi-Cohort Funnel

```sql
WITH cohorts AS (
  SELECT user_id, DATE_FORMAT(signup_date,'%Y-%u') AS signup_week
  FROM users
),
funnel AS (
  SELECT c.signup_week,
    COUNT(DISTINCT c.user_id)                                  AS signed_up,
    COUNT(DISTINCT CASE WHEN u.verified_at IS NOT NULL
          THEN c.user_id END)                                  AS verified,
    COUNT(DISTINCT o1.user_id)                                 AS first_purchase,
    COUNT(DISTINCT CASE WHEN o2.cnt >= 2 THEN c.user_id END)   AS repeat_purchase
  FROM cohorts c
  JOIN users u ON c.user_id = u.user_id
  LEFT JOIN (SELECT DISTINCT user_id FROM orders) o1 ON c.user_id = o1.user_id
  LEFT JOIN (SELECT user_id, COUNT(*) AS cnt FROM orders GROUP BY user_id) o2
    ON c.user_id = o2.user_id
  GROUP BY c.signup_week
)
SELECT signup_week, signed_up, verified,
  ROUND(verified*100.0/NULLIF(signed_up,0),2)      AS verification_rate,
  first_purchase,
  ROUND(first_purchase*100.0/NULLIF(signed_up,0),2) AS purchase_cvr,
  repeat_purchase,
  ROUND(repeat_purchase*100.0/NULLIF(first_purchase,0),2) AS repeat_rate
FROM funnel ORDER BY signup_week;
```

---

## Pattern 5 — 3-Consecutive-Month Trend

```sql
WITH monthly_sales AS (
  SELECT product_id,
    DATE_FORMAT(order_date,'%Y-%m-01') AS month,
    SUM(amount) AS revenue
  FROM orders GROUP BY product_id, DATE_FORMAT(order_date,'%Y-%m-01')
),
with_prev AS (
  SELECT product_id, month, revenue,
    LAG(revenue,1) OVER (PARTITION BY product_id ORDER BY month) AS prev_1,
    LAG(revenue,2) OVER (PARTITION BY product_id ORDER BY month) AS prev_2
  FROM monthly_sales
)
SELECT product_id, month, revenue, prev_1, prev_2
FROM with_prev
WHERE revenue > prev_1 AND prev_1 > prev_2 AND prev_2 IS NOT NULL
ORDER BY product_id, month;
```

---

## Pattern 6 — Multi-Touch Attribution

```sql
WITH touchpoints AS (
  SELECT user_id, channel, touched_at,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY touched_at) AS touch_num,
    COUNT(*) OVER (PARTITION BY user_id) AS total_touches
  FROM marketing_touches
  WHERE user_id IN (SELECT DISTINCT user_id FROM orders)
),
conversions AS (
  SELECT user_id, SUM(amount) AS revenue FROM orders GROUP BY user_id
)
SELECT t.channel,
  ROUND(SUM(c.revenue / t.total_touches), 2)  AS linear_revenue,
  ROUND(SUM(CASE WHEN t.touch_num = 1
       THEN c.revenue ELSE 0 END), 2)          AS first_touch_revenue,
  ROUND(SUM(CASE WHEN t.touch_num = t.total_touches
       THEN c.revenue ELSE 0 END), 2)          AS last_touch_revenue
FROM touchpoints t JOIN conversions c ON t.user_id = c.user_id
GROUP BY t.channel ORDER BY linear_revenue DESC;
```

---

## Pattern 7 — Retention / Churn / Reactivation

```sql
WITH monthly_activity AS (
  SELECT DISTINCT user_id,
    DATE_FORMAT(activity_date,'%Y-%m-01') AS active_month
  FROM user_activity
),
with_prev AS (
  SELECT user_id, active_month,
    LAG(active_month) OVER (PARTITION BY user_id ORDER BY active_month) AS prev_month
  FROM monthly_activity
),
first_month AS (
  SELECT user_id, MIN(active_month) AS first_month FROM monthly_activity GROUP BY user_id
)
SELECT m.active_month,
  COUNT(DISTINCT CASE WHEN m.active_month = fm.first_month
        THEN m.user_id END) AS new_users,
  COUNT(DISTINCT CASE
        WHEN m.active_month != fm.first_month
         AND PERIOD_DIFF(DATE_FORMAT(m.active_month,'%Y%m'),
                         DATE_FORMAT(m.prev_month,'%Y%m')) = 1
        THEN m.user_id END) AS retained,
  COUNT(DISTINCT CASE
        WHEN m.active_month != fm.first_month
         AND PERIOD_DIFF(DATE_FORMAT(m.active_month,'%Y%m'),
                         DATE_FORMAT(m.prev_month,'%Y%m')) > 1
        THEN m.user_id END) AS reactivated
FROM with_prev m JOIN first_month fm ON m.user_id = fm.user_id
GROUP BY m.active_month ORDER BY m.active_month;
```

---

## Practice Questions

### Q1 — Medium ✅
First/last order amount + spend trend (users with 3+ orders).

```sql
WITH order_stats AS (
  SELECT user_id, COUNT(*) AS total_orders,
    SUM(amount) AS total_spend, ROUND(AVG(amount),2) AS avg_order_value
  FROM orders GROUP BY user_id HAVING COUNT(*) >= 3
),
first_last AS (
  SELECT DISTINCT user_id,
    FIRST_VALUE(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_amount,
    LAST_VALUE(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_amount
  FROM orders WHERE user_id IN (SELECT user_id FROM order_stats)
)
SELECT u.user_id, u.country, os.total_orders,
  os.total_spend, os.avg_order_value,
  fl.first_amount, fl.last_amount,
  ROUND((fl.last_amount-fl.first_amount)*100.0/NULLIF(fl.first_amount,0),2) AS spend_change_pct,
  CASE WHEN fl.last_amount > fl.first_amount THEN 'Increasing'
       WHEN fl.last_amount < fl.first_amount THEN 'Decreasing'
       ELSE 'Flat' END AS spend_trend
FROM order_stats os
JOIN first_last fl ON os.user_id = fl.user_id
JOIN users u       ON os.user_id = u.user_id
ORDER BY os.total_spend DESC;
```

### Q2 — Hard ✅
Session boundaries (30-min gap) with conversion flag.

```sql
WITH event_gaps AS (
  SELECT user_id, event_type, event_time,
    TIMESTAMPDIFF(MINUTE,
      LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time),
      event_time) AS gap_minutes
  FROM events
),
session_flags AS (
  SELECT *,
    CASE WHEN gap_minutes > 30 OR gap_minutes IS NULL
         THEN 1 ELSE 0 END AS is_new_session
  FROM event_gaps
),
with_session_id AS (
  SELECT *,
    SUM(is_new_session) OVER (
      PARTITION BY user_id ORDER BY event_time
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS session_num
  FROM session_flags
)
SELECT user_id, session_num,
  MIN(event_time) AS session_start, MAX(event_time) AS session_end,
  TIMESTAMPDIFF(MINUTE, MIN(event_time), MAX(event_time)) AS duration_mins,
  COUNT(*) AS event_count,
  SUM(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS purchases,
  CASE WHEN SUM(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) > 0
       THEN 1 ELSE 0 END AS converted
FROM with_session_id
GROUP BY user_id, session_num
HAVING COUNT(*) >= 2
ORDER BY user_id, session_num;
```

### Q3 — Very Hard ✅
Monthly cohort health: new / retained / reactivated / churned / net change.

```sql
WITH cohort_base AS (
  SELECT user_id, DATE_FORMAT(signup_date,'%Y-%m-01') AS cohort_month
  FROM users WHERE YEAR(signup_date) = 2025
),
monthly_activity AS (
  SELECT DISTINCT user_id, DATE_FORMAT(order_date,'%Y-%m-01') AS active_month
  FROM orders
),
cohort_monthly AS (
  SELECT cb.user_id, cb.cohort_month,
    DATE_ADD(cb.cohort_month, INTERVAL n.n MONTH) AS report_month
  FROM cohort_base cb
  CROSS JOIN (SELECT 1 AS n UNION SELECT 2 UNION SELECT 3
              UNION SELECT 4 UNION SELECT 5 UNION SELECT 6) n
),
activity_flags AS (
  SELECT cm.cohort_month, cm.report_month, cm.user_id,
    MAX(CASE WHEN ma.active_month = cm.report_month
        THEN 1 ELSE 0 END) AS active_this_month,
    MAX(CASE WHEN ma.active_month = DATE_SUB(cm.report_month, INTERVAL 1 MONTH)
        THEN 1 ELSE 0 END) AS active_last_month,
    MAX(CASE WHEN ma.active_month < DATE_SUB(cm.report_month, INTERVAL 1 MONTH)
        THEN 1 ELSE 0 END) AS active_before_last
  FROM cohort_monthly cm
  LEFT JOIN monthly_activity ma ON cm.user_id = ma.user_id
  GROUP BY cm.cohort_month, cm.report_month, cm.user_id
),
cohort_size AS (
  SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
  FROM cohort_base GROUP BY cohort_month
)
SELECT af.cohort_month, cs.cohort_size, af.report_month,
  PERIOD_DIFF(DATE_FORMAT(af.report_month,'%Y%m'),
              DATE_FORMAT(af.cohort_month,'%Y%m')) AS months_since_signup,
  COUNT(DISTINCT CASE WHEN active_this_month=1 AND active_last_month=0
    AND active_before_last=0 THEN af.user_id END)  AS new_users,
  COUNT(DISTINCT CASE WHEN active_this_month=1
    AND active_last_month=1  THEN af.user_id END)  AS retained,
  COUNT(DISTINCT CASE WHEN active_this_month=1 AND active_last_month=0
    AND active_before_last=1 THEN af.user_id END)  AS reactivated,
  COUNT(DISTINCT CASE WHEN active_this_month=0
    AND active_last_month=1  THEN af.user_id END)  AS churned,
  COUNT(DISTINCT CASE WHEN active_this_month=1
    THEN af.user_id END) -
  COUNT(DISTINCT CASE WHEN active_last_month=1
    THEN af.user_id END)                           AS net_change
FROM activity_flags af
JOIN cohort_size cs ON af.cohort_month = cs.cohort_month
GROUP BY af.cohort_month, cs.cohort_size, af.report_month
ORDER BY af.cohort_month, af.report_month;
```

---

## Key Takeaways

- **Session ID** → cumulative SUM of `is_new_session` flags per user
- **FIRST/LAST VALUE** → always use `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`
- **Retention/churn/reactivation** → classify by LAG month gap: =1 retained, >1 reactivated, NULL churned
- **Multi-touch attribution** → `revenue / total_touches` for linear, touch_num=1 for first-touch
- **Funnel cohorts** → COUNT DISTINCT with LEFT JOINs per funnel step
- **3-month trend** → LAG(1) and LAG(2), WHERE current > prev_1 AND prev_1 > prev_2
- **Build bottom-up** → simplest CTE first, add complexity step by step
- **One CTE = one logical concept** → keeps queries readable and debuggable

---

*Day 22 complete — 8 days to go 🚀*
