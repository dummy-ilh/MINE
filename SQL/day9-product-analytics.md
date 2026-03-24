# Day 9 — Product Analytics & Funnel Analysis
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Conversion Funnel Basics
2. Ordered Funnel (Sequential Steps)
3. Retention Analysis (Day N)
4. DAU / WAU / MAU + Stickiness
5. RFM Segmentation
6. Session Analysis
7. A/B Test Results
8. Feature Adoption & Power Users

---

## 1. Conversion Funnel Basics
A funnel tracks users moving through sequential steps. The key metric is drop-off rate at each step.
```sql
-- Table: events(user_id, event_type, event_date)
-- Funnel: signup → onboarding → first_purchase → repeat_purchase

WITH funnel AS (
  SELECT
    COUNT(DISTINCT CASE WHEN event_type = 'signup'
          THEN user_id END)           AS step1_signup,
    COUNT(DISTINCT CASE WHEN event_type = 'onboarding'
          THEN user_id END)           AS step2_onboarding,
    COUNT(DISTINCT CASE WHEN event_type = 'first_purchase'
          THEN user_id END)           AS step3_purchase,
    COUNT(DISTINCT CASE WHEN event_type = 'repeat_purchase'
          THEN user_id END)           AS step4_repeat
  FROM events
)
SELECT
  step1_signup,
  step2_onboarding,
  step3_purchase,
  step4_repeat,
  ROUND(step2_onboarding  * 100.0 / step1_signup,    2) AS signup_to_onboard_pct,
  ROUND(step3_purchase    * 100.0 / step2_onboarding, 2) AS onboard_to_purchase_pct,
  ROUND(step4_repeat      * 100.0 / step3_purchase,   2) AS purchase_to_repeat_pct,
  ROUND(step4_repeat      * 100.0 / step1_signup,     2) AS overall_conversion_pct
FROM funnel;
```

---

## 2. Ordered Funnel (Steps Must Be Sequential)
Real funnels require steps to happen in order. This is harder.
```sql
-- User must: view_product → add_to_cart → checkout → purchase
-- Each step must come AFTER the previous one

WITH step1 AS (
  SELECT DISTINCT user_id, MIN(event_date) AS s1_date
  FROM events WHERE event_type = 'view_product'
  GROUP BY user_id
),
step2 AS (
  SELECT DISTINCT e.user_id, MIN(e.event_date) AS s2_date
  FROM events e
  JOIN step1 s ON e.user_id = s.user_id
  WHERE e.event_type = 'add_to_cart'
    AND e.event_date > s.s1_date   -- must happen AFTER step 1
  GROUP BY e.user_id
),
step3 AS (
  SELECT DISTINCT e.user_id, MIN(e.event_date) AS s3_date
  FROM events e
  JOIN step2 s ON e.user_id = s.user_id
  WHERE e.event_type = 'checkout'
    AND e.event_date > s.s2_date
  GROUP BY e.user_id
),
step4 AS (
  SELECT DISTINCT e.user_id, MIN(e.event_date) AS s4_date
  FROM events e
  JOIN step3 s ON e.user_id = s.user_id
  WHERE e.event_type = 'purchase'
    AND e.event_date > s.s3_date
  GROUP BY e.user_id
)
SELECT
  COUNT(DISTINCT step1.user_id) AS viewed,
  COUNT(DISTINCT step2.user_id) AS added_to_cart,
  COUNT(DISTINCT step3.user_id) AS checked_out,
  COUNT(DISTINCT step4.user_id) AS purchased,
  ROUND(COUNT(DISTINCT step2.user_id) * 100.0 /
        COUNT(DISTINCT step1.user_id), 2) AS view_to_cart_pct,
  ROUND(COUNT(DISTINCT step4.user_id) * 100.0 /
        COUNT(DISTINCT step1.user_id), 2) AS overall_pct
FROM step1
LEFT JOIN step2 USING (user_id)
LEFT JOIN step3 USING (user_id)
LEFT JOIN step4 USING (user_id);
```

> 💡 Each CTE filters `event_date > prev step date` — enforces ordering.

---

## 3. Retention Analysis
Day N retention = % of users who come back N days after their first event.
```sql
-- Day 1, Day 7, Day 30 retention
WITH first_seen AS (
  SELECT user_id, MIN(event_date) AS first_date
  FROM events
  GROUP BY user_id
),
activity AS (
  SELECT DISTINCT user_id, event_date
  FROM events
)
SELECT
  COUNT(DISTINCT f.user_id)                          AS total_users,
  -- Day 1
  COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 1
    THEN a.user_id END)                              AS day1_retained,
  -- Day 7
  COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 7
    THEN a.user_id END)                              AS day7_retained,
  -- Day 30
  COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 30
    THEN a.user_id END)                              AS day30_retained,
  ROUND(COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 1
    THEN a.user_id END) * 100.0 /
    COUNT(DISTINCT f.user_id), 2)                    AS day1_retention_pct,
  ROUND(COUNT(DISTINCT CASE
    WHEN DATEDIFF(a.event_date, f.first_date) = 7
    THEN a.user_id END) * 100.0 /
    COUNT(DISTINCT f.user_id), 2)                    AS day7_retention_pct
FROM first_seen f
LEFT JOIN activity a ON f.user_id = a.user_id;
```

---

## 4. DAU / WAU / MAU + Stickiness

```sql
-- DAU — Daily Active Users
SELECT event_date, COUNT(DISTINCT user_id) AS DAU
FROM events
GROUP BY event_date
ORDER BY event_date;

-- WAU — Weekly Active Users
SELECT
  YEAR(event_date)  AS yr,
  WEEK(event_date)  AS wk,
  COUNT(DISTINCT user_id) AS WAU
FROM events
GROUP BY YEAR(event_date), WEEK(event_date)
ORDER BY yr, wk;

-- MAU — Monthly Active Users
SELECT
  DATE_FORMAT(event_date, '%Y-%m') AS month,
  COUNT(DISTINCT user_id) AS MAU
FROM events
GROUP BY DATE_FORMAT(event_date, '%Y-%m')
ORDER BY month;

-- DAU/MAU ratio — measures stickiness (higher = more sticky)
WITH dau AS (
  SELECT event_date, COUNT(DISTINCT user_id) AS daily_users
  FROM events GROUP BY event_date
),
mau AS (
  SELECT DATE_FORMAT(event_date, '%Y-%m') AS month,
    COUNT(DISTINCT user_id) AS monthly_users
  FROM events GROUP BY DATE_FORMAT(event_date, '%Y-%m')
)
SELECT
  DATE_FORMAT(d.event_date, '%Y-%m') AS month,
  AVG(d.daily_users)                 AS avg_dau,
  m.monthly_users                    AS mau,
  ROUND(AVG(d.daily_users) * 100.0 / m.monthly_users, 2) AS dau_mau_ratio
FROM dau d
JOIN mau m ON DATE_FORMAT(d.event_date, '%Y-%m') = m.month
GROUP BY DATE_FORMAT(d.event_date, '%Y-%m'), m.monthly_users
ORDER BY month;
```

> 💡 DAU/MAU ratio is a key product health metric. WhatsApp ~70%, Facebook ~50%, most apps < 20%. Interviewers love asking you to compute and interpret this.

---

## 5. RFM Segmentation
RFM = Recency, Frequency, Monetary — the classic DS segmentation framework.
```sql
-- Table: orders(order_id, user_id, amount, order_date)
WITH rfm_raw AS (
  SELECT user_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    COUNT(*)                                AS frequency,
    SUM(amount)                             AS monetary
  FROM orders GROUP BY user_id
),
rfm_scored AS (
  SELECT user_id, recency_days, frequency, monetary,
    NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
    NTILE(5) OVER (ORDER BY frequency DESC)    AS f_score,
    NTILE(5) OVER (ORDER BY monetary DESC)     AS m_score
  FROM rfm_raw
)
SELECT user_id, r_score, f_score, m_score,
  r_score + f_score + m_score AS total_rfm,
  CASE
    WHEN r_score >= 4 AND f_score >= 4 THEN 'Champion'
    WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
    WHEN r_score >= 4 AND f_score <= 2 THEN 'New Customer'
    WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
    WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
    ELSE 'Potential'
  END AS segment
FROM rfm_scored
ORDER BY total_rfm DESC;
```

---

## 6. Session Analysis

```sql
-- Table: page_views(user_id, page, view_time)
-- Define session: gap > 30 mins = new session
WITH gaps AS (
  SELECT user_id, page, view_time,
    LAG(view_time) OVER (
      PARTITION BY user_id ORDER BY view_time
    ) AS prev_time
  FROM page_views
),
sessions AS (
  SELECT user_id, page, view_time,
    SUM(CASE
      WHEN prev_time IS NULL OR
           TIMESTAMPDIFF(MINUTE, prev_time, view_time) > 30
      THEN 1 ELSE 0
    END) OVER (PARTITION BY user_id ORDER BY view_time) AS session_id
  FROM gaps
)
SELECT user_id, session_id,
  COUNT(*)                                              AS pages_per_session,
  MIN(view_time)                                        AS session_start,
  MAX(view_time)                                        AS session_end,
  TIMESTAMPDIFF(MINUTE, MIN(view_time), MAX(view_time)) AS duration_mins
FROM sessions
GROUP BY user_id, session_id;
```

---

## 7. A/B Test Results

```sql
-- Tables: ab_assignments(user_id, variant), orders(user_id, amount, order_date)
-- Question: Did the treatment group convert better?
WITH experiment AS (
  SELECT a.variant,
    COUNT(DISTINCT a.user_id)  AS total_users,
    COUNT(DISTINCT o.user_id)  AS converted_users,
    SUM(o.amount)              AS total_revenue,
    AVG(o.amount)              AS avg_order_value
  FROM ab_assignments a
  LEFT JOIN orders o
    ON  a.user_id = o.user_id
    AND o.order_date BETWEEN '2025-01-01' AND '2025-02-28'
  GROUP BY a.variant
)
SELECT variant, total_users, converted_users,
  ROUND(converted_users * 100.0 / total_users, 2) AS conversion_rate,
  ROUND(total_revenue / total_users, 2)           AS revenue_per_user,
  ROUND(avg_order_value, 2)                       AS avg_order_value
FROM experiment ORDER BY variant;
```

---

## 8. Power Users & Feature Adoption

```sql
-- What % of users use each feature?
-- Feature adoption rate
SELECT feature_name,
  COUNT(DISTINCT user_id) AS users_used,
  ROUND(COUNT(DISTINCT user_id) * 100.0 /
    (SELECT COUNT(DISTINCT user_id) FROM users), 2) AS adoption_pct
FROM feature_events
GROUP BY feature_name
ORDER BY adoption_pct DESC;

-- Top 10% power users
-- Power users: top 10% by event count
WITH activity AS (
  SELECT user_id, COUNT(*) AS event_count
  FROM events GROUP BY user_id
)
SELECT user_id, event_count,
  NTILE(10) OVER (ORDER BY event_count DESC) AS decile
FROM activity
HAVING decile = 1;
```

---

## Practice Questions
Table: events(user_id, event_type, event_date)
Events: 'signup', 'first_login', 'first_purchase'
Build a simple 3-step funnel showing user count and conversion rate at each step.


🟡 Q2 — Medium

Table: orders(order_id, user_id, amount, order_date)
Compute RFM scores for each user. Recency = days since last order, Frequency = total orders, Monetary = total spend. Use NTILE(4) to score each dimension 1–4.


🔴 Q3 — Hard

Tables: users(user_id, signup_date), events(user_id, event_type, event_date)
For each weekly signup cohort in 2025, calculate Week 0, Week 1, Week 2, Week 4 retention rates (% of cohort who had any event that week). Return cohort week, cohort size, and retention % for each week.
### Q1 — Easy ✅
Simple 3-step funnel.
Table: events(user_id, event_type, event_date)
Events: 'signup', 'first_login', 'first_purchase'
Build a simple 3-step funnel showing user count and conversion rate at each step.
```sql
WITH funnel AS (
  SELECT
    COUNT(DISTINCT CASE WHEN event_type = 'signup'
          THEN user_id END)         AS step1_signup,
    COUNT(DISTINCT CASE WHEN event_type = 'first_login'
          THEN user_id END)         AS step2_login,
    COUNT(DISTINCT CASE WHEN event_type = 'first_purchase'
          THEN user_id END)         AS step3_purchase
  FROM events
)
SELECT
  step1_signup, step2_login, step3_purchase,
  ROUND(step2_login    * 100.0 / step1_signup, 2) AS signup_to_login_pct,
  ROUND(step3_purchase * 100.0 / step2_login,  2) AS login_to_purchase_pct,
  ROUND(step3_purchase * 100.0 / step1_signup, 2) AS overall_pct
FROM funnel;
```

### Q2 — Medium ✅
Table: orders(order_id, user_id, amount, order_date)
Compute RFM scores for each user. Recency = days since last order, Frequency = total orders, Monetary = total spend. Use NTILE(4) to score each dimension 1–4.
RFM scores with NTILE(4).

```sql
WITH rfm_raw AS (
  SELECT user_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) AS recency_days,
    COUNT(*)                                AS frequency,
    SUM(amount)                             AS monetary
  FROM orders GROUP BY user_id
)
SELECT user_id, recency_days, frequency, monetary,
  NTILE(4) OVER (ORDER BY recency_days ASC)  AS r_score,
  NTILE(4) OVER (ORDER BY frequency DESC)    AS f_score,
  NTILE(4) OVER (ORDER BY monetary DESC)     AS m_score
FROM rfm_raw;
```

### Q3 — Hard ✅
Weekly cohort retention — Week 0 through Week 4.
Tables: users(user_id, signup_date), events(user_id, event_type, event_date)
For each weekly signup cohort in 2025, calculate Week 0, Week 1, Week 2, Week 4 retention rates (% of cohort who had any event that week). Return cohort week, cohort size, and retention % for each week.
```sql
WITH cohorts AS (
  SELECT user_id,
    DATE_TRUNC('week', signup_date) AS cohort_week
  FROM users WHERE YEAR(signup_date) = 2025
),
activity AS (
  SELECT DISTINCT user_id,
    DATE_TRUNC('week', event_date) AS active_week
  FROM events
),
combined AS (
  SELECT c.cohort_week, c.user_id,
    DATEDIFF(a.active_week, c.cohort_week) / 7 AS week_number
  FROM cohorts c
  LEFT JOIN activity a ON c.user_id = a.user_id
),
cohort_sizes AS (
  SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size
  FROM cohorts GROUP BY cohort_week
)
SELECT c.cohort_week, cs.cohort_size,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 0
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week0_pct,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 1
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week1_pct,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 2
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week2_pct,
  ROUND(COUNT(DISTINCT CASE WHEN week_number = 4
        THEN c.user_id END) * 100.0 / cs.cohort_size, 2) AS week4_pct
FROM combined c
JOIN cohort_sizes cs ON c.cohort_week = cs.cohort_week
GROUP BY c.cohort_week, cs.cohort_size
ORDER BY c.cohort_week;
```

---

## Key Takeaways

- **Funnel** → COUNT DISTINCT per event type / first step count
- **Ordered funnel** → chain CTEs, each filtering `event_date > prev step date`
- **Day N retention** → DATEDIFF(event_date, first_date) = N
- **DAU/MAU ratio** → stickiness metric, higher = better engagement
- **RFM** → NTILE on recency (ASC), frequency (DESC), monetary (DESC)
- **Session detection** → LAG + TIMESTAMPDIFF > 30 min threshold
- **A/B test** → LEFT JOIN on variant + date window + conversion rate
- **Cohort pivot** → CASE WHEN week_number = N inside COUNT DISTINCT
- **Power users**→NTILE(10) on event count

---

*Day 9 complete — 21 days to go 🚀*
