# Day 5 — Window Functions
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. What is a Window Function
2. PARTITION BY vs ORDER BY in OVER()
3. ROW_NUMBER, RANK, DENSE_RANK
4. LAG and LEAD
5. Running Totals & Moving Averages (Window Frames)
6. NTILE
7. FIRST_VALUE, LAST_VALUE, NTH_VALUE
8. FAANG Favourite Patterns

---

## 1. What is a Window Function?
Performs a calculation across a set of rows related to the current row — without collapsing them like GROUP BY does.

```sql
-- GROUP BY collapses rows — you lose individual row detail
SELECT department, AVG(salary) FROM employees GROUP BY department;
-- Result: 1 row per department

-- Window function keeps ALL rows + adds the aggregate alongside
SELECT name, department, salary,
  AVG(salary) OVER (PARTITION BY department) AS dept_avg
FROM employees;
-- Result: every employee row + their dept avg on the same line
```

**Syntax:**
```sql
function_name() OVER (
  PARTITION BY col   -- defines the group/window
  ORDER BY col       -- defines row order within window
  ROWS/RANGE BETWEEN -- defines window frame (optional)
)
```

---

## 2. PARTITION BY vs ORDER BY in OVER()

```sql
-- PARTITION BY only — aggregate per group, all rows kept
SELECT name, department, salary,
  AVG(salary)  OVER (PARTITION BY department) AS dept_avg,
  SUM(salary)  OVER (PARTITION BY department) AS dept_total,
  COUNT(*)     OVER (PARTITION BY department) AS dept_headcount
FROM employees;
```

```sql
-- ORDER BY only — running/cumulative across all rows
SELECT name, order_date, amount,
  SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders;
```

```sql
-- PARTITION BY + ORDER BY — running calculation per group
SELECT name, department, salary, hire_date,
  SUM(salary) OVER (
    PARTITION BY department
    ORDER BY hire_date
  ) AS running_dept_payroll
FROM employees;
```

---

## 3. ROW_NUMBER, RANK, DENSE_RANK

```sql
-- employees: Alice=100k, Bob=100k, Carol=90k, Dave=80k

SELECT name, salary,
  ROW_NUMBER()  OVER (ORDER BY salary DESC) AS row_num,
  RANK()        OVER (ORDER BY salary DESC) AS rnk,
  DENSE_RANK()  OVER (ORDER BY salary DESC) AS dense_rnk
FROM employees;
```

| name | salary | ROW_NUMBER | RANK | DENSE_RANK |
|---|---|---|---|---|
| Alice | 100k | 1 | 1 | 1 |
| Bob | 100k | 2 | 1 | 1 |
| Carol | 90k | 3 | 3 | 2 |
| Dave | 80k | 4 | 4 | 3 |

- **ROW_NUMBER** — always unique, arbitrary tiebreak
- **RANK** — ties same rank, skips next (1,1,3,4)
- **DENSE_RANK** — ties same rank, no skipping (1,1,2,3)

```sql
-- Top earner per department
WITH ranked AS (
  SELECT name, department, salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn
  FROM employees
)
SELECT name, department, salary FROM ranked WHERE rn = 1;
```

```sql
-- Top 3 earners per department (ties included fairly)
WITH ranked AS (
  SELECT name, department, salary,
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dr
  FROM employees
)
SELECT name, department, salary FROM ranked WHERE dr <= 3;
```

---

## 4. LAG and LEAD
Access a value from a previous or next row without a self join.
```sql
-- LAG: value from N rows before | LEAD: value from N rows ahead
LAG(col, n, default)  OVER (PARTITION BY ... ORDER BY ...)
LEAD(col, n, default) OVER (PARTITION BY ... ORDER BY ...)
```

```sql
-- Month-over-month revenue change
WITH monthly AS (
  SELECT YEAR(order_date) AS yr, MONTH(order_date) AS mo,
    SUM(amount) AS revenue
  FROM orders
  GROUP BY YEAR(order_date), MONTH(order_date)
)
SELECT yr, mo, revenue,
  LAG(revenue, 1, 0) OVER (ORDER BY yr, mo) AS prev_month_rev,
  revenue - LAG(revenue, 1, 0) OVER (ORDER BY yr, mo) AS mom_change,
  ROUND(
    (revenue - LAG(revenue, 1, 0) OVER (ORDER BY yr, mo)) * 100.0 /
    NULLIF(LAG(revenue, 1, 0) OVER (ORDER BY yr, mo), 0), 2
  ) AS mom_pct_change
FROM monthly;
```

```sql
-- Days between orders per customer
SELECT customer_id, order_date, amount,
  LEAD(order_date) OVER (
    PARTITION BY customer_id ORDER BY order_date
  ) AS next_order_date,
  DATEDIFF(
    LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
    order_date
  ) AS days_until_next_order
FROM orders;
```

```sql
-- Detect price movement
SELECT product_id, price_date, price,
  LAG(price) OVER (PARTITION BY product_id ORDER BY price_date) AS prev_price,
  CASE
    WHEN price > LAG(price) OVER (PARTITION BY product_id ORDER BY price_date)
    THEN 'Increased'
    WHEN price < LAG(price) OVER (PARTITION BY product_id ORDER BY price_date)
    THEN 'Decreased'
    ELSE 'No Change'
  END AS price_movement
FROM product_prices;
```

---

## 5. Running Totals & Moving Averages
The frame clause controls exactly which rows are included in the window.
### Frame Clause Options
```sql
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW          -- all rows up to now
ROWS BETWEEN 2 PRECEDING AND CURRENT ROW                   -- last 3 rows
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING                   -- 3-row centered
ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING   -- entire partition
```

```sql
-- Running total
SELECT order_date, amount,
  SUM(amount) OVER (
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_total
FROM orders;
```

```sql
-- 7-day moving average
SELECT metric_date, daily_value,
  ROUND(AVG(daily_value) OVER (
    ORDER BY metric_date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ), 2) AS moving_avg_7d
FROM daily_metrics;
```

```sql
-- Cumulative % of total revenue
SELECT order_date, amount,
  ROUND(
    SUM(amount) OVER (
      ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) * 100.0 / SUM(amount) OVER (), 2
  ) AS cumulative_pct
FROM orders;
-- SUM() OVER () with no args = grand total
```

---

## 6. NTILE

```sql
-- Salary quartiles
SELECT name, salary,
  NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;
-- 1 = top 25%, 4 = bottom 25%

-- With labels
WITH bucketed AS (
  SELECT name, salary,
    NTILE(4) OVER (ORDER BY salary DESC) AS quartile
  FROM employees
)
SELECT name, salary,
  CASE quartile
    WHEN 1 THEN 'Top 25%'
    WHEN 2 THEN 'Upper Mid'
    WHEN 3 THEN 'Lower Mid'
    WHEN 4 THEN 'Bottom 25%'
  END AS salary_band
FROM bucketed;
```

---

## 7. FIRST_VALUE & LAST_VALUE

```sql
-- Top earner in dept on every row
SELECT name, department, salary,
  FIRST_VALUE(name) OVER (
    PARTITION BY department ORDER BY salary DESC
  ) AS top_earner_in_dept
FROM employees;
```

```sql
-- LAST_VALUE — always use explicit frame!
SELECT name, department, salary,
  LAST_VALUE(salary) OVER (
    PARTITION BY department
    ORDER BY salary DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS dept_min_salary
FROM employees;
```

> ⚠️ Without explicit frame, LAST_VALUE only looks up to current row — very common bug.

---

## 8. FAANG Favourite Patterns

```sql
-- Consecutive 7-day active users (island-gap trick)
WITH daily AS (
  SELECT DISTINCT user_id, DATE(order_date) AS day FROM orders
),
with_grp AS (
  SELECT user_id, day,
    DATE_SUB(day, INTERVAL ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY day) DAY) AS grp
  FROM daily
)
SELECT user_id FROM with_grp
GROUP BY user_id, grp
HAVING COUNT(*) >= 7;
```

```sql
-- Session analysis (30-min gap = new session)
WITH gaps AS (
  SELECT user_id, event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_time
  FROM events
),
sessions AS (
  SELECT user_id, event_time,
    SUM(CASE
      WHEN TIMESTAMPDIFF(MINUTE, prev_time, event_time) > 30
        OR prev_time IS NULL THEN 1 ELSE 0
    END) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
  FROM gaps
)
SELECT user_id, session_id, COUNT(*) AS events_in_session
FROM sessions
GROUP BY user_id, session_id;
```

---

## Practice Questions

### Q1 — Easy ✅
```sql
SELECT name, department, salary,
  DENSE_RANK() OVER (
    PARTITION BY department
    ORDER BY salary DESC
  ) AS dense_rnk
FROM employees;
```

### Q2 — Medium ✅
```sql
SELECT customer_id, order_id, amount, order_date,
  LAG(amount, 1) OVER (
    PARTITION BY customer_id ORDER BY order_date
  ) AS prev_order_amount,
  amount - LAG(amount, 1) OVER (
    PARTITION BY customer_id ORDER BY order_date
  ) AS amount_diff
FROM orders;
```

### Q3 — Hard ✅
```sql
WITH moving_avg AS (
  SELECT product_id, sale_date, revenue,
    AVG(revenue) OVER (
      PARTITION BY product_id
      ORDER BY sale_date
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
  FROM daily_sales
),
latest_avg AS (
  SELECT product_id,
    LAST_VALUE(moving_avg_7d) OVER (
      PARTITION BY product_id
      ORDER BY sale_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS latest_7d_avg
  FROM moving_avg
)
SELECT DISTINCT m.product_id, m.sale_date, m.revenue, m.moving_avg_7d
FROM moving_avg m
JOIN latest_avg l ON m.product_id = l.product_id
WHERE l.latest_7d_avg > 500;
```

---

## Key Takeaways

- **Window vs GROUP BY** — window keeps all rows, GROUP BY collapses them
- **ROW_NUMBER** → always unique | **RANK** → skips after tie | **DENSE_RANK** → no skip
- **LAG/LEAD** → previous/next row value without a self join
- **Frame clause** → controls which rows are in the window calculation
- **LAST_VALUE** → always add `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`
- **SUM() OVER ()** with no args = grand total of entire result set
- **Island-gap trick** → ROW_NUMBER + DATE_SUB = detect consecutive sequences

---

FunctionPurposeROW_NUMBER()Unique sequential number — no tiesRANK()Ties same rank, skips next (1,1,3)DENSE_RANK()Ties same rank, no skip (1,1,2)LAG(col, n)Value from N rows beforeLEAD(col, n)Value from N rows afterNTILE(n)Split into N equal bucketsFIRST_VALUE()First value in windowLAST_VALUE()Last value — always use explicit frameSUM/AVG OVER()Running total / moving average
