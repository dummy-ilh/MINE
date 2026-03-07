# Day 10 — Advanced Window Functions & Analytics
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. PERCENTILE Functions
2. Running Totals by Group with Reset
3. Gaps & Islands
4. Window Functions Inside CTEs
5. Comparing Rows to Group Statistics
6. Year-over-Year & Period Comparisons
7. Weighted Averages
8. FAANG Hard Patterns

---

## 1. PERCENTILE Functions

```sql
-- Postgres / BigQuery
SELECT department,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) AS median_sal,
  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS p75_sal,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary) AS p90_sal
FROM employees
GROUP BY department;

-- MySQL — simulate median with ROW_NUMBER
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

> 💡 Median salary per group = one of the most asked FAANG window questions.

---

## 2. Running Totals by Group with Reset

```sql
-- Running revenue per rep, resets each month
SELECT rep_id, sale_date, amount,
  SUM(amount) OVER (
    PARTITION BY rep_id, DATE_FORMAT(sale_date, '%Y-%m')
    ORDER BY sale_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS monthly_running_total
FROM sales;

-- Running order count per customer per year
SELECT customer_id, order_date,
  COUNT(*) OVER (
    PARTITION BY customer_id, YEAR(order_date)
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS order_num_this_year
FROM orders;
```

---

## 3. Gaps & Islands

```sql
-- KEY INSIGHT: date - row_number = constant for consecutive dates

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
```

```sql
-- Longest streak per user
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
islands AS (
  SELECT user_id,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp,
    COUNT(*) AS streak_len
  FROM numbered
  GROUP BY user_id, DATE_SUB(login_date, INTERVAL rn DAY)
)
SELECT user_id, MAX(streak_len) AS longest_streak
FROM islands
GROUP BY user_id
ORDER BY longest_streak DESC;
```

> 💡 DATE - ROW_NUMBER = constant for consecutive dates. The island trick. Memorise it.

---

## 4. Window Functions Inside CTEs

```sql
-- Days where revenue > 30-day moving average
WITH moving_avg AS (
  SELECT product_id, sale_date, revenue,
    AVG(revenue) OVER (
      PARTITION BY product_id
      ORDER BY sale_date
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS ma_30d
  FROM daily_sales
),
above_avg AS (
  SELECT product_id, sale_date, revenue, ma_30d,
    CASE WHEN revenue > ma_30d THEN 1 ELSE 0 END AS is_above
  FROM moving_avg
)
SELECT product_id,
  COUNT(*)                                   AS total_days,
  SUM(is_above)                              AS days_above_avg,
  ROUND(SUM(is_above) * 100.0 / COUNT(*), 2) AS pct_above_avg
FROM above_avg
GROUP BY product_id;
```

---

## 5. Comparing Rows to Group Statistics

```sql
SELECT e.name, e.department, e.salary,
  ROUND(AVG(e.salary) OVER (PARTITION BY e.department), 0) AS dept_avg,
  ROUND(e.salary - AVG(e.salary) OVER (
        PARTITION BY e.department), 0)                     AS diff_from_avg,
  RANK() OVER (PARTITION BY e.department
               ORDER BY e.salary DESC)                     AS dept_rank
FROM employees e
ORDER BY e.department, dept_rank;
```

---

## 6. Year-over-Year & Period Comparisons

```sql
-- YoY revenue
WITH yearly AS (
  SELECT YEAR(order_date) AS yr, SUM(amount) AS revenue
  FROM orders GROUP BY YEAR(order_date)
)
SELECT yr, revenue,
  LAG(revenue) OVER (ORDER BY yr) AS prev_year,
  ROUND(
    (revenue - LAG(revenue) OVER (ORDER BY yr)) * 100.0 /
    NULLIF(LAG(revenue) OVER (ORDER BY yr), 0), 2
  ) AS yoy_pct
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
    (weekly_rev - LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr)) * 100.0 /
    NULLIF(LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr), 0), 2
  ) AS yoy_pct
FROM weekly ORDER BY yr, wk;
```

---

## 7. Weighted Averages

```sql
-- Weighted avg (avoids Simpson's paradox)
SELECT category,
  SUM(amount) / COUNT(*) AS weighted_avg,
  AVG(amount)            AS simple_avg
FROM orders GROUP BY category;

-- Rolling 3-month weighted average (recent = higher weight)
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
GROUP BY channel ORDER BY first_purchases DESC;
```

```sql
-- Rolling 7-day unique users
SELECT DISTINCT a.event_date,
  COUNT(DISTINCT b.user_id) AS rolling_7d_users
FROM events a
JOIN events b
  ON b.event_date BETWEEN
     a.event_date - INTERVAL 6 DAY AND a.event_date
GROUP BY a.event_date
ORDER BY a.event_date;
```

```sql
-- Order that pushed cumulative spend over $1000
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
```sql
SELECT name, department, salary,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary)
    OVER (PARTITION BY department) AS median_sal,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary)
    OVER (PARTITION BY department) AS p90_sal,
  ROUND(PERCENT_RANK() OVER (
    PARTITION BY department ORDER BY salary
  ) * 100, 1)                      AS dept_percentile_rank
FROM employees
ORDER BY department, salary;
```

### Q2 — Medium ✅
```sql
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY login_date
    ) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
islands AS (
  SELECT user_id,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp,
    COUNT(*) AS streak_len
  FROM numbered
  GROUP BY user_id, DATE_SUB(login_date, INTERVAL rn DAY)
)
SELECT user_id, MAX(streak_len) AS longest_streak
FROM islands
GROUP BY user_id
ORDER BY longest_streak DESC;
```

### Q3 — Hard ✅
```sql
WITH first_purchases AS (
  SELECT user_id, channel, amount, order_date,
    ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY order_date
    ) AS rn
  FROM orders
),
first_only AS (SELECT * FROM first_purchases WHERE rn = 1),
monthly AS (
  SELECT channel,
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    COUNT(*)    AS first_purchases,
    AVG(amount) AS avg_order_value
  FROM first_only
  WHERE YEAR(order_date) = 2025
  GROUP BY channel, DATE_FORMAT(order_date, '%Y-%m')
)
SELECT channel, month, first_purchases,
  ROUND(first_purchases * 100.0 /
    SUM(first_purchases) OVER (PARTITION BY month), 2) AS pct_of_month,
  ROUND(avg_order_value, 2)                            AS avg_order_value,
  ROUND(
    (first_purchases - LAG(first_purchases) OVER (
      PARTITION BY channel ORDER BY month)) * 100.0 /
    NULLIF(LAG(first_purchases) OVER (
      PARTITION BY channel ORDER BY month), 0), 2
  ) AS mom_growth_pct
FROM monthly
ORDER BY channel, month;
```

---

## Key Takeaways

- **Median in MySQL** → ROW_NUMBER + WHERE rn IN (floor, ceil of middle)
- **Gaps & Islands** → DATE - ROW_NUMBER = constant for consecutive rows
- **Running total reset** → PARTITION BY group + time period together
- **YoY** → LAG over years | **Same week last year** → LAG PARTITION BY week
- **PERCENT_RANK()** → 0 to 1 percentile rank within window
- **CUME_DIST()** → fraction of rows ≤ current row
- **Threshold crossing** → running SUM WHERE prev < X AND curr >= X
- **First touch attribution** → ROW_NUMBER PARTITION BY user ORDER BY date ASC

---

*Day 10 complete — 20 days to go 🚀*
