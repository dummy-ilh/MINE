# Day 10 — Advanced Window Functions & Analytics
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. PERCENTILE Functions
2. Running Totals by Group with Reset
3. Gaps & Islands (Consecutive Streaks)
4. Multi-Layer Window CTEs
5. Comparing Rows to Group Statistics
6. Year-over-Year & Period Comparisons
7. Weighted Averages
8. FAANG Hard Patterns

---

## 1. PERCENTILE Functions

```sql
-- Postgres / BigQuery
SELECT
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) AS median_salary,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS p75_salary,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary) AS p90_salary
FROM employees;

-- Per department
SELECT department,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) AS median_sal,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary) AS p90_sal
FROM employees
GROUP BY department;

-- MySQL simulation (no PERCENTILE_CONT)
WITH ranked AS (
  SELECT salary,
    ROW_NUMBER() OVER (ORDER BY salary) AS rn,
    COUNT(*) OVER ()                    AS total
  FROM employees
)
SELECT AVG(salary) AS median
FROM ranked
WHERE rn IN (FLOOR((total + 1) / 2), CEIL((total + 1) / 2));
```

> 💡 Median salary per group is one of the most asked FAANG window questions.

---

## 2. Running Totals by Group with Reset

```sql
-- Monthly running total per sales rep (resets each month)
SELECT rep_id, sale_date, amount,
  SUM(amount) OVER (
    PARTITION BY rep_id, DATE_FORMAT(sale_date, '%Y-%m')
    ORDER BY sale_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS monthly_running_total
FROM sales;

-- Running order count per customer per year
SELECT customer_id, order_date, order_id,
  COUNT(*) OVER (
    PARTITION BY customer_id, YEAR(order_date)
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS order_num_this_year
FROM orders;
```

---

## 3. Gaps & Islands — Consecutive Streaks

**Key insight:** For consecutive dates, `date - ROW_NUMBER = constant` within each streak.

```sql
-- All streaks per user
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY login_date
    ) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
islands AS (
  SELECT user_id, login_date,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp
  FROM numbered
)
SELECT user_id, grp,
  MIN(login_date) AS streak_start,
  MAX(login_date) AS streak_end,
  COUNT(*)        AS streak_length
FROM islands
GROUP BY user_id, grp
ORDER BY user_id, streak_start;

-- Longest streak per user
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
islands AS (
  SELECT user_id,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp
  FROM numbered
),
streaks AS (
  SELECT user_id, COUNT(*) AS streak_len
  FROM islands GROUP BY user_id, grp
)
SELECT user_id, MAX(streak_len) AS longest_streak
FROM streaks
GROUP BY user_id
ORDER BY longest_streak DESC;
```

---

## 4. Multi-Layer Window CTEs

```sql
-- Days where revenue exceeded product's own 30-day moving avg
WITH moving_avg AS (
  SELECT product_id, sale_date, revenue,
    AVG(revenue) OVER (
      PARTITION BY product_id
      ORDER BY sale_date
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS ma_30d
  FROM daily_sales
),
above_avg_days AS (
  SELECT product_id, sale_date, revenue, ma_30d,
    CASE WHEN revenue > ma_30d THEN 1 ELSE 0 END AS above_avg
  FROM moving_avg
)
SELECT product_id,
  COUNT(*)                             AS total_days,
  SUM(above_avg)                       AS days_above_avg,
  ROUND(SUM(above_avg) * 100.0 / COUNT(*), 2) AS pct_above_avg
FROM above_avg_days
GROUP BY product_id;
```

---

## 5. Comparing Rows to Group Statistics

```sql
WITH dept_stats AS (
  SELECT department,
    AVG(salary)                                          AS dept_avg,
    MAX(salary)                                          AS dept_max,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS dept_median
  FROM employees GROUP BY department
)
SELECT e.name, e.department, e.salary,
  ROUND(d.dept_avg, 0)                    AS dept_avg,
  ROUND(e.salary - d.dept_avg, 0)         AS diff_from_avg,
  ROUND(e.salary * 100.0 / d.dept_max, 1) AS pct_of_dept_max,
  RANK() OVER (PARTITION BY e.department
               ORDER BY e.salary DESC)    AS dept_rank
FROM employees e
JOIN dept_stats d ON e.department = d.department;
```

---

## 6. Year-over-Year & Period Comparisons

```sql
-- YoY revenue growth
WITH yearly AS (
  SELECT YEAR(order_date) AS yr, SUM(amount) AS revenue
  FROM orders GROUP BY YEAR(order_date)
)
SELECT yr, revenue,
  LAG(revenue) OVER (ORDER BY yr) AS prev_year,
  ROUND(
    (revenue - LAG(revenue) OVER (ORDER BY yr)) * 100.0 /
    NULLIF(LAG(revenue) OVER (ORDER BY yr), 0), 2
  ) AS yoy_growth_pct
FROM yearly;

-- Same week last year
WITH weekly AS (
  SELECT YEAR(sale_date) AS yr, WEEK(sale_date) AS wk,
    SUM(revenue) AS weekly_rev
  FROM daily_sales GROUP BY YEAR(sale_date), WEEK(sale_date)
)
SELECT yr, wk, weekly_rev,
  LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr) AS same_wk_last_yr,
  ROUND(
    (weekly_rev - LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr))
    * 100.0 /
    NULLIF(LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr), 0), 2
  ) AS yoy_pct
FROM weekly ORDER BY yr, wk;
```

---

## 7. Weighted Averages

```sql
-- Weighted avg order value (avoids Simpson's paradox)
SELECT category,
  SUM(amount) / COUNT(*) AS weighted_avg,
  AVG(amount)            AS simple_avg
FROM orders GROUP BY category;

-- 3-month weighted moving average
SELECT month, revenue,
  ROUND(
    (revenue * 3
     + LAG(revenue, 1, 0) OVER (ORDER BY month) * 2
     + LAG(revenue, 2, 0) OVER (ORDER BY month) * 1
    ) / 6, 2
  ) AS weighted_3mo_avg
FROM monthly_revenue;
```

---

## 8. FAANG Hard Patterns

```sql
-- First purchase channel attribution
WITH ranked AS (
  SELECT user_id, channel, purchase_date,
    ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY purchase_date
    ) AS rn
  FROM purchases
)
SELECT channel,
  COUNT(*) AS first_purchases,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_attribution
FROM ranked WHERE rn = 1
GROUP BY channel;
```

```sql
-- Rolling 7-day unique users
SELECT DISTINCT a.event_date,
  COUNT(DISTINCT b.user_id) AS rolling_7d_users
FROM events a
JOIN events b
  ON b.event_date BETWEEN
     a.event_date - INTERVAL 6 DAY AND a.event_date
GROUP BY a.event_date;
```

```sql
-- Percentile rank per department
SELECT name, department, salary,
  ROUND(PERCENT_RANK() OVER (
    PARTITION BY department ORDER BY salary
  ) * 100, 1) AS dept_percentile,
  CUME_DIST() OVER (
    PARTITION BY department ORDER BY salary
  )           AS cumulative_dist
FROM employees;
```

```sql
-- Find order that crossed $1000 cumulative spend
WITH running AS (
  SELECT user_id, order_id, amount, order_date,
    SUM(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_spend,
    SUM(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prev_cumulative
  FROM orders
)
SELECT user_id, order_id, amount, order_date, cumulative_spend
FROM running
WHERE prev_cumulative < 1000
  AND cumulative_spend >= 1000;
```

---

## Practice Questions

### Q1 — Easy ✅
Dept median, p90 salary, and each employee's percentile rank.

```sql
WITH dept_stats AS (
  SELECT department,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) AS median_sal,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary) AS p90_sal
  FROM employees GROUP BY department
)
SELECT e.name, e.department, e.salary,
  ROUND(d.median_sal, 0) AS dept_median,
  ROUND(d.p90_sal, 0)    AS dept_p90,
  ROUND(PERCENT_RANK() OVER (
    PARTITION BY e.department ORDER BY e.salary
  ) * 100, 1)            AS dept_percentile_rank
FROM employees e
JOIN dept_stats d ON e.department = d.department
ORDER BY e.department, e.salary DESC;
```

### Q2 — Medium ✅
Longest consecutive login streak per user.

```sql
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
islands AS (
  SELECT user_id,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp
  FROM numbered
),
streaks AS (
  SELECT user_id, COUNT(*) AS streak_len
  FROM islands GROUP BY user_id, grp
)
SELECT user_id, MAX(streak_len) AS longest_streak
FROM streaks
GROUP BY user_id
ORDER BY longest_streak DESC;
```

### Q3 — Hard ✅
Channel attribution + MoM growth in first purchases.

```sql
WITH first_orders AS (
  SELECT user_id, channel, amount, order_date,
    ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY order_date
    ) AS rn
  FROM orders
),
first_only AS (
  SELECT channel, amount, order_date
  FROM first_orders WHERE rn = 1
),
channel_summary AS (
  SELECT channel,
    COUNT(*)              AS first_purchases,
    ROUND(AVG(amount), 2) AS avg_first_order_value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_attribution
  FROM first_only GROUP BY channel
),
monthly AS (
  SELECT channel,
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    COUNT(*) AS monthly_first_purchases
  FROM first_only
  WHERE YEAR(order_date) = 2025
  GROUP BY channel, DATE_FORMAT(order_date, '%Y-%m')
),
monthly_growth AS (
  SELECT channel, month, monthly_first_purchases,
    ROUND(
      (monthly_first_purchases -
       LAG(monthly_first_purchases) OVER (
         PARTITION BY channel ORDER BY month)) * 100.0 /
      NULLIF(LAG(monthly_first_purchases) OVER (
        PARTITION BY channel ORDER BY month), 0), 2
    ) AS mom_growth_pct
  FROM monthly
)
SELECT cs.channel, cs.first_purchases, cs.pct_attribution,
  cs.avg_first_order_value, mg.month,
  mg.monthly_first_purchases, mg.mom_growth_pct
FROM channel_summary cs
JOIN monthly_growth mg ON cs.channel = mg.channel
ORDER BY cs.first_purchases DESC, mg.month;
```

---

## Key Takeaways

- **PERCENTILE_CONT** → Postgres/BQ; simulate in MySQL with ROW_NUMBER + middle row
- **Gaps & Islands** → DATE - ROW_NUMBER = constant for consecutive dates
- **Running total reset** → PARTITION BY group AND period together
- **YoY** → LAG over years | **Same period last year** → LAG PARTITION BY period
- **PERCENT_RANK()** → 0 to 1 percentile within window
- **CUME_DIST()** → cumulative distribution (what % are at or below this value)
- **Threshold crossing** → running SUM where prev < X AND curr >= X
- **First touch attribution** → ROW_NUMBER PARTITION BY user ORDER BY date ASC, WHERE rn=1

---

*Day 10 complete — 20 days to go 🚀*
