# Day 26 — Hard Mixed Problems I
**FAANG SQL 30-Day Prep**

---

## Problem 1 — Consecutive Active Days (7+ Streak)

```sql
WITH daily AS (
  SELECT DISTINCT user_id, activity_date
  FROM user_activity WHERE YEAR(activity_date) = 2025
),
gaps AS (
  SELECT user_id, activity_date,
    DATEDIFF(activity_date,
      LAG(activity_date) OVER (PARTITION BY user_id ORDER BY activity_date)) AS day_gap
  FROM daily
),
streak_groups AS (
  SELECT user_id, activity_date,
    SUM(CASE WHEN day_gap != 1 OR day_gap IS NULL THEN 1 ELSE 0 END)
      OVER (PARTITION BY user_id ORDER BY activity_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS streak_id
  FROM gaps
),
streaks AS (
  SELECT user_id, streak_id,
    MIN(activity_date) AS streak_start,
    MAX(activity_date) AS streak_end,
    COUNT(*) AS streak_length
  FROM streak_groups GROUP BY user_id, streak_id
)
SELECT user_id, streak_start, streak_end, streak_length
FROM streaks WHERE streak_length >= 7
ORDER BY streak_length DESC;
```

---

## Problem 2 — MRR Waterfall (New / Expansion / Contraction / Churn)

```sql
WITH prev AS (
  SELECT user_id, month, revenue,
    LAG(revenue) OVER (PARTITION BY user_id ORDER BY month) AS prev_revenue,
    LAG(month)   OVER (PARTITION BY user_id ORDER BY month) AS prev_month
  FROM subscriptions
),
classified AS (
  SELECT month, user_id, revenue, prev_revenue,
    CASE WHEN prev_revenue IS NULL  THEN 'new'
         WHEN revenue > prev_revenue THEN 'expansion'
         WHEN revenue < prev_revenue THEN 'contraction'
         ELSE 'flat' END AS mrr_type
  FROM prev
),
churned AS (
  SELECT DATE_ADD(month, INTERVAL 1 MONTH) AS month,
    SUM(revenue) AS churned_mrr
  FROM subscriptions s
  WHERE NOT EXISTS (
    SELECT 1 FROM subscriptions s2
    WHERE s2.user_id = s.user_id
      AND s2.month = DATE_ADD(s.month, INTERVAL 1 MONTH)
  )
  GROUP BY DATE_ADD(month, INTERVAL 1 MONTH)
)
SELECT c.month,
  SUM(CASE WHEN c.mrr_type='new' THEN c.revenue ELSE 0 END) AS new_mrr,
  SUM(CASE WHEN c.mrr_type='expansion' THEN c.revenue-c.prev_revenue ELSE 0 END) AS expansion_mrr,
  -SUM(CASE WHEN c.mrr_type='contraction' THEN c.prev_revenue-c.revenue ELSE 0 END) AS contraction_mrr,
  -COALESCE(ch.churned_mrr,0) AS churned_mrr,
  SUM(CASE WHEN c.mrr_type='new' THEN c.revenue ELSE 0 END)
  + SUM(CASE WHEN c.mrr_type='expansion' THEN c.revenue-c.prev_revenue ELSE 0 END)
  - SUM(CASE WHEN c.mrr_type='contraction' THEN c.prev_revenue-c.revenue ELSE 0 END)
  - COALESCE(ch.churned_mrr,0) AS net_new_mrr
FROM classified c LEFT JOIN churned ch ON c.month = ch.month
GROUP BY c.month, ch.churned_mrr ORDER BY c.month;
```

> 💡 Churn is a separate CTE using NOT EXISTS — users who appeared last month but not this month.

---

## Problem 3 — Median Without MEDIAN()

```sql
WITH monthly AS (
  SELECT DATE_FORMAT(order_date,'%Y-%m') AS month, amount,
    ROW_NUMBER() OVER (PARTITION BY DATE_FORMAT(order_date,'%Y-%m') ORDER BY amount) AS rn,
    COUNT(*) OVER (PARTITION BY DATE_FORMAT(order_date,'%Y-%m')) AS cnt
  FROM orders
)
SELECT month, AVG(amount) AS median_amount
FROM monthly
WHERE rn IN (FLOOR((cnt+1)/2), CEIL((cnt+1)/2))
GROUP BY month ORDER BY month;
-- FLOOR/CEIL handles both odd (same row) and even (avg of two middle rows)
```

---

## Problem 4 — Minimum Products for 50% Revenue

```sql
WITH rev AS (
  SELECT product_id, SUM(amount) AS revenue
  FROM orders GROUP BY product_id
),
ranked AS (
  SELECT product_id, revenue,
    ROUND(revenue*100.0/SUM(revenue) OVER (),4) AS pct,
    ROUND(SUM(revenue) OVER (ORDER BY revenue DESC
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)*100.0/SUM(revenue) OVER (),4) AS cumulative_pct,
    ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rank_num
  FROM rev
)
SELECT product_id, revenue, pct, cumulative_pct, rank_num
FROM ranked
WHERE cumulative_pct - pct < 50
   OR rank_num = (SELECT MIN(rank_num) FROM ranked WHERE cumulative_pct >= 50)
ORDER BY rank_num;
```

---

## Problem 5 — User Journey + Bottleneck Detection

```sql
WITH journey AS (
  SELECT user_id,
    MIN(CASE WHEN event_type='signup'          THEN event_time END) AS t_signup,
    MIN(CASE WHEN event_type='onboarding'      THEN event_time END) AS t_onboard,
    MIN(CASE WHEN event_type='first_purchase'  THEN event_time END) AS t_first,
    MIN(CASE WHEN event_type='second_purchase' THEN event_time END) AS t_second
  FROM events GROUP BY user_id
),
durations AS (
  SELECT user_id,
    TIMESTAMPDIFF(HOUR, t_signup,  t_onboard) AS s_to_o,
    TIMESTAMPDIFF(HOUR, t_onboard, t_first)   AS o_to_f,
    TIMESTAMPDIFF(HOUR, t_first,   t_second)  AS f_to_s
  FROM journey
  WHERE t_signup IS NOT NULL AND t_onboard IS NOT NULL
    AND t_first  IS NOT NULL AND t_second  IS NOT NULL
),
medians AS (
  SELECT
    AVG(CASE WHEN rn1 IN (FLOOR((cnt+1)/2),CEIL((cnt+1)/2)) THEN s_to_o END) AS med_s_o,
    AVG(CASE WHEN rn2 IN (FLOOR((cnt+1)/2),CEIL((cnt+1)/2)) THEN o_to_f END) AS med_o_f,
    AVG(CASE WHEN rn3 IN (FLOOR((cnt+1)/2),CEIL((cnt+1)/2)) THEN f_to_s END) AS med_f_s
  FROM (
    SELECT *, COUNT(*) OVER () AS cnt,
      ROW_NUMBER() OVER (ORDER BY s_to_o) AS rn1,
      ROW_NUMBER() OVER (ORDER BY o_to_f) AS rn2,
      ROW_NUMBER() OVER (ORDER BY f_to_s) AS rn3
    FROM durations
  ) t
)
SELECT d.user_id, d.s_to_o, d.o_to_f, d.f_to_s,
  m.med_s_o, m.med_o_f, m.med_f_s,
  CASE
    WHEN d.s_to_o > 2*m.med_s_o THEN 'Slow: signup→onboard'
    WHEN d.o_to_f > 2*m.med_o_f THEN 'Slow: onboard→first'
    WHEN d.f_to_s > 2*m.med_f_s THEN 'Slow: first→second'
    ELSE '✅ Normal'
  END AS bottleneck
FROM durations d CROSS JOIN medians m
ORDER BY d.f_to_s DESC;
```

---

## Problem 6 — Recursive Org Chart Salary Rollup

```sql
WITH RECURSIVE org AS (
  SELECT emp_id, name, manager_id, salary, department,
    1 AS depth, emp_id AS subtree_root
  FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.emp_id, e.name, e.manager_id, e.salary, e.department,
    o.depth+1, o.subtree_root
  FROM employees e JOIN org o ON e.manager_id = o.emp_id
),
subtree AS (
  SELECT subtree_root,
    COUNT(*) AS headcount, SUM(salary) AS total_salary,
    ROUND(AVG(salary),2) AS avg_salary
  FROM org GROUP BY subtree_root
)
SELECT e.emp_id, e.name, e.department, e.salary,
  o.depth,
  s.headcount-1 AS reports_count,
  s.total_salary-e.salary AS reports_budget,
  s.avg_salary
FROM employees e
JOIN (SELECT DISTINCT emp_id, depth FROM org WHERE emp_id=subtree_root) o ON e.emp_id=o.emp_id
JOIN subtree s ON e.emp_id=s.subtree_root
ORDER BY o.depth, e.salary DESC;
```

---

## Problem 7 — N-Day Retention Matrix

```sql
WITH cohorts AS (
  SELECT user_id,
    DATE_SUB(signup_date, INTERVAL DAYOFWEEK(signup_date)-1 DAY) AS cohort_week
  FROM users
),
sizes AS (SELECT cohort_week, COUNT(*) AS sz FROM cohorts GROUP BY cohort_week),
ret AS (
  SELECT c.cohort_week,
    DATEDIFF(s.session_date, u.signup_date) AS day_num,
    COUNT(DISTINCT c.user_id) AS retained
  FROM cohorts c JOIN users u ON c.user_id=u.user_id
  JOIN sessions s ON c.user_id=s.user_id
  WHERE DATEDIFF(s.session_date, u.signup_date) IN (1,3,7,14,30)
  GROUP BY c.cohort_week, day_num
)
SELECT sz.cohort_week, sz.sz AS cohort_size,
  ROUND(MAX(CASE WHEN r.day_num=1  THEN r.retained END)*100.0/sz.sz,2) AS d1,
  ROUND(MAX(CASE WHEN r.day_num=3  THEN r.retained END)*100.0/sz.sz,2) AS d3,
  ROUND(MAX(CASE WHEN r.day_num=7  THEN r.retained END)*100.0/sz.sz,2) AS d7,
  ROUND(MAX(CASE WHEN r.day_num=14 THEN r.retained END)*100.0/sz.sz,2) AS d14,
  ROUND(MAX(CASE WHEN r.day_num=30 THEN r.retained END)*100.0/sz.sz,2) AS d30
FROM sizes sz LEFT JOIN ret r ON sz.cohort_week=r.cohort_week
GROUP BY sz.cohort_week, sz.sz ORDER BY sz.cohort_week;
```

---

## Problem 8 — Linear Attribution (30-Day Window)

```sql
WITH touches AS (
  SELECT t.user_id, t.channel, t.touch_time,
    c.conversion_time, c.revenue,
    COUNT(*) OVER (PARTITION BY t.user_id, c.conversion_time) AS total_touches
  FROM ad_touches t JOIN conversions c ON t.user_id=c.user_id
    AND t.touch_time <= c.conversion_time
    AND t.touch_time >= c.conversion_time - INTERVAL 30 DAY
)
SELECT channel,
  COUNT(DISTINCT user_id) AS users_touched,
  ROUND(SUM(revenue/total_touches),2) AS linear_revenue,
  COUNT(*) AS touch_credits
FROM touches GROUP BY channel ORDER BY linear_revenue DESC;
```

---

## Practice Questions & Answers

### Q1 ✅ — Logged in every day of January 2025

```sql
SELECT user_id, COUNT(*) AS total_logins
FROM logins
WHERE login_date BETWEEN '2025-01-01' AND '2025-01-31'
GROUP BY user_id
HAVING COUNT(DISTINCT login_date) = 31
ORDER BY user_id;
```

> 💡 `COUNT(DISTINCT login_date) = 31` handles duplicate login rows per day cleanly.

### Q2 ✅ — Weekly net revenue + rolling avg + alert

```sql
WITH wo AS (
  SELECT DATE_SUB(DATE(order_date), INTERVAL DAYOFWEEK(DATE(order_date))-1 DAY) AS ws,
    COUNT(*) AS total_orders, SUM(amount) AS order_revenue
  FROM orders GROUP BY ws
),
wr AS (
  SELECT DATE_SUB(DATE(refund_date), INTERVAL DAYOFWEEK(DATE(refund_date))-1 DAY) AS ws,
    COUNT(*) AS refunded_orders, SUM(refund_amount) AS refund_total
  FROM refunds GROUP BY ws
),
combined AS (
  SELECT wo.ws AS week_start, wo.total_orders,
    wo.order_revenue, COALESCE(wr.refunded_orders,0) AS refunded_orders,
    COALESCE(wr.refund_total,0) AS refund_total,
    wo.order_revenue - COALESCE(wr.refund_total,0) AS net_revenue
  FROM wo LEFT JOIN wr ON wo.ws = wr.ws
),
with_rolling AS (
  SELECT *,
    ROUND(refunded_orders*100.0/NULLIF(total_orders,0),2) AS refund_rate,
    AVG(net_revenue) OVER (ORDER BY week_start ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING) AS rolling_4wk
  FROM combined
)
SELECT week_start, total_orders, order_revenue, refunded_orders,
  refund_total, net_revenue, refund_rate, ROUND(rolling_4wk,2) AS rolling_4wk_avg,
  ROUND((net_revenue-rolling_4wk)*100.0/NULLIF(rolling_4wk,0),2) AS vs_rolling_pct,
  CASE WHEN (net_revenue-rolling_4wk)*100.0/NULLIF(rolling_4wk,0) < -15
       THEN '🔴 REVENUE DROP' ELSE '✅ NORMAL' END AS alert
FROM with_rolling ORDER BY week_start;
```

### Q3 ✅ — First overdraft transaction per account

```sql
WITH running AS (
  SELECT txn_id, account_id, txn_type, amount, txn_date,
    SUM(CASE WHEN txn_type='credit' THEN amount
             WHEN txn_type='debit'  THEN -amount ELSE 0 END)
      OVER (PARTITION BY account_id ORDER BY txn_date, txn_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_balance,
    SUM(CASE WHEN txn_type='credit' THEN amount
             WHEN txn_type='debit'  THEN -amount ELSE 0 END)
      OVER (PARTITION BY account_id ORDER BY txn_date, txn_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_balance
  FROM transactions
),
overdrafts AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY txn_date, txn_id) AS rn
  FROM running
  WHERE running_balance < 0 AND (prev_balance >= 0 OR prev_balance IS NULL)
)
SELECT txn_id, account_id, txn_type, amount, txn_date,
  ROUND(prev_balance,2) AS balance_before,
  ROUND(running_balance,2) AS balance_after,
  '🔴 FIRST OVERDRAFT' AS flag
FROM overdrafts WHERE rn = 1 ORDER BY txn_date;
```

> 💡 Two window frames in same CTE: `CURRENT ROW` = balance after, `1 PRECEDING` = balance before. Filter `running_balance < 0 AND prev_balance >= 0` = exact crossing transaction.

---

## Key Patterns This Day

| Problem | Core Technique |
|---|---|
| 7+ day streak | Gaps-and-islands → cumulative SUM of break flags |
| MRR waterfall | LAG revenue + NOT EXISTS for churn |
| Median (MySQL) | ROW_NUMBER + FLOOR/CEIL for middle row(s) |
| Pareto 50% | Cumulative SUM, filter where prev_cumulative < 50 |
| User journey | MIN per event_type → TIMESTAMPDIFF → CROSS JOIN medians |
| Org chart rollup | Recursive CTE with subtree_root propagation |
| Retention matrix | DATEDIFF + CASE WHEN pivot |
| Attribution window | JOIN on time range + COUNT OVER partition |
| Full-month login | COUNT(DISTINCT login_date) = N |
| First overdraft | Dual window frames (CURRENT ROW vs 1 PRECEDING) |

---

*Day 26 complete — 4 days to go 🚀*
