# Day 2 — GROUP BY & Aggregations
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. COUNT(*) vs COUNT(col)
2. GROUP BY
3. HAVING vs WHERE
4. GROUP BY with CASE WHEN
5. DISTINCT in Aggregates
6. ROLLUP

---

## 1. COUNT(*) vs COUNT(col)

```sql
-- Table: orders
-- order_id | customer_id | coupon_code | amount
--    1      |     101     |   'SAVE10'  |  500
--    2      |     102     |    NULL     |  300
--    3      |     103     |    NULL     |  200
--    4      |     104     |   'FLAT50'  |  450

SELECT
  COUNT(*)                    AS total_orders,        -- 4 (counts every row)
  COUNT(coupon_code)          AS orders_with_coupon,  -- 2 (skips NULLs)
  COUNT(DISTINCT coupon_code) AS unique_coupons       -- 2
FROM orders;

-- How many orders had NO coupon?
SELECT COUNT(*) - COUNT(coupon_code) AS orders_without_coupon
FROM orders;
-- 4 - 2 = 2
```

> ⚠️ `COUNT(*)` counts all rows. `COUNT(col)` skips NULLs. Extremely common FAANG trap.

```sql
SELECT
  COUNT(*)          AS total_rows,       -- counts everything incl. NULLs
  COUNT(manager_id) AS non_null_managers,-- counts only non-NULL values
  SUM(salary)       AS total_salary,
  AVG(salary)       AS avg_salary,
  MIN(salary)       AS lowest,
  MAX(salary)       AS highest
FROM employees;

-- employees table has 100 rows, 20 have NULL manager_id
SELECT COUNT(*)          -- returns 100
SELECT COUNT(manager_id) -- returns 80
```
---

## 2. GROUP BY

Every column in SELECT that is **not inside an aggregate** must appear in GROUP BY. Splits rows into groups, then aggregates each group.

```sql

sql-- How many employees per department?
SELECT department, COUNT(*) AS headcount
FROM employees
GROUP BY department;
sql-- Avg salary per department per job title
SELECT department, job_title, AVG(salary) AS avg_sal
FROM employees
GROUP BY department, job_title
ORDER BY avg_sal DESC;


-- Revenue by region
SELECT region, SUM(amount) AS revenue
FROM sales
GROUP BY region;

-- Best selling product per region
SELECT region, product, SUM(amount) AS revenue
FROM sales
GROUP BY region, product
ORDER BY region, revenue DESC;

-- Monthly sales trend
SELECT
  YEAR(sale_date)  AS yr,
  MONTH(sale_date) AS mo,
  SUM(amount)      AS monthly_revenue
FROM sales
GROUP BY YEAR(sale_date), MONTH(sale_date)
ORDER BY yr, mo;
```

```sql

-- ❌ FAILS — name is not aggregated or grouped
SELECT department, name, COUNT(*)
FROM employees
GROUP BY department;

-- ✅ WORKS
SELECT department, COUNT(*)
FROM employees
GROUP BY department;

-- ❌ FAILS — can't GROUP BY alias in most DBs
SELECT YEAR(sale_date) AS yr, SUM(amount)
FROM sales
GROUP BY yr;

-- ✅ Repeat the expression
SELECT YEAR(sale_date) AS yr, SUM(amount)
FROM sales
GROUP BY YEAR(sale_date);
```

---

## 3. HAVING vs WHERE

| | WHERE | HAVING |
|---|---|---|
| Runs at | Before grouping | After grouping |
| Filters | Individual rows | Groups |
| Can use aggregates? | ❌ No | ✅ Yes |

```sql
-- ❌ FAILS — can't use COUNT() in WHERE
SELECT department, COUNT(*) AS headcount
FROM employees
WHERE COUNT(*) > 10
GROUP BY department;

-- ✅ Use HAVING for aggregate filters
SELECT department, COUNT(*) AS headcount
FROM employees
GROUP BY department
HAVING COUNT(*) > 10;
```

```sql
-- Departments with pay gap (MAX > 3x MIN)
SELECT department, MIN(salary) AS lowest, MAX(salary) AS highest
FROM employees
GROUP BY department
HAVING MAX(salary) > 3 * MIN(salary);
```

```sql
-- Customers who ONLY placed cancelled orders
SELECT customer_id
FROM orders
GROUP BY customer_id
HAVING COUNT(*) = SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END);
```

```sql
-- Departments above company average salary
SELECT department, AVG(salary) AS avg_sal
FROM employees
GROUP BY department
HAVING AVG(salary) > (SELECT AVG(salary) FROM employees);
```

```sql
-- Classic combo: WHERE first, then HAVING
SELECT department, AVG(salary) AS avg_sal
FROM employees
WHERE is_active = 1           -- raw column filter
GROUP BY department
HAVING AVG(salary) > 90000    -- aggregate filter
ORDER BY avg_sal DESC;

-- Combining WHERE and HAVING (very common in interviews)
SELECT department, AVG(salary) AS avg_sal
FROM employees
WHERE is_active = 1              -- filter rows FIRST
GROUP BY department
HAVING AVG(salary) > 100000      -- then filter groups
ORDER BY avg_sal DESC;

```

> 💡 **Rule:** Filtering raw column? → WHERE. Filtering COUNT/SUM/AVG/MIN/MAX? → HAVING.

---

## 4. GROUP BY with CASE WHEN

```sql

-- Count employees per salary band
SELECT
  CASE
    WHEN salary >= 150000 THEN 'High'
    WHEN salary >= 100000 THEN 'Medium'
    ELSE 'Low'
  END AS salary_band,
  COUNT(*) AS headcount
FROM employees
GROUP BY
  CASE
    WHEN salary >= 150000 THEN 'High'
    WHEN salary >= 100000 THEN 'Medium'
    ELSE 'Low'
  END;
```
```sql
-- Bucket users by activity level
SELECT
  CASE
    WHEN login_count >= 20 THEN 'Power User'
    WHEN login_count >= 10 THEN 'Regular'
    WHEN login_count >= 1  THEN 'Casual'
    ELSE 'Inactive'
  END AS user_segment,
  COUNT(*)   AS user_count,
  AVG(spend) AS avg_spend
FROM users
GROUP BY
  CASE
    WHEN login_count >= 20 THEN 'Power User'
    WHEN login_count >= 10 THEN 'Regular'
    WHEN login_count >= 1  THEN 'Casual'
    ELSE 'Inactive'
  END
ORDER BY avg_spend DESC;
```

```sql
-- Revenue split by order size
SELECT
  CASE
    WHEN amount >= 1000 THEN 'Large'
    WHEN amount >= 500  THEN 'Medium'
    ELSE 'Small'
  END AS order_size,
  COUNT(*)    AS num_orders,
  SUM(amount) AS total_revenue
FROM orders
GROUP BY
  CASE
    WHEN amount >= 1000 THEN 'Large'
    WHEN amount >= 500  THEN 'Medium'
    ELSE 'Small'
  END;
```

> ⚠️ The full CASE WHEN expression must be repeated in GROUP BY — you cannot use the alias.

---

## 5. DISTINCT in Aggregates
```sql
-- How many unique departments?
SELECT COUNT(DISTINCT department) FROM employees;

-- Total salary paid to unique job titles only (rare but asked)
SELECT SUM(DISTINCT salary) FROM employees;
```

```sql
-- Total events vs unique users
SELECT
  COUNT(*)                AS total_events,
  COUNT(DISTINCT user_id) AS unique_users
FROM user_events;

-- Unique users per event type
SELECT event_type,
  COUNT(DISTINCT user_id) AS unique_users
FROM user_events
GROUP BY event_type;

-- % of users who did each event type (classic FAANG pattern)
SELECT event_type,
  COUNT(DISTINCT user_id) AS users_did_this,
  ROUND(
    COUNT(DISTINCT user_id) * 100.0 /
    (SELECT COUNT(DISTINCT user_id) FROM user_events), 2
  ) AS pct_of_users
FROM user_events
GROUP BY event_type;
```

---

## 6. ROLLUP

Automatically adds subtotal and grand total rows.

```sql
SELECT region, product, SUM(amount) AS revenue
FROM sales
GROUP BY ROLLUP(region, product);
```

**Output pattern:**
| region | product | revenue |
|---|---|---|
| North | Laptop | 50000 |
| North | Phone | 30000 |
| North | NULL | 80000 ← subtotal |
| South | Laptop | 40000 |
| South | NULL | 60000 ← subtotal |
| NULL | NULL | 140000 ← grand total |

```sql
-- Clean NULL labels with COALESCE + GROUPING()
SELECT
  COALESCE(region, 'ALL REGIONS')   AS region,
  COALESCE(product, 'ALL PRODUCTS') AS product,
  SUM(amount)       AS revenue,
  GROUPING(region)  AS is_region_rollup,
  GROUPING(product) AS is_product_rollup
FROM sales
GROUP BY ROLLUP(region, product);
```

```sql
-- 3-level ROLLUP: year → quarter → month
SELECT
  YEAR(sale_date)    AS yr,
  QUARTER(sale_date) AS qtr,
  MONTH(sale_date)   AS mo,
  SUM(amount)        AS revenue
FROM sales
GROUP BY ROLLUP(
  YEAR(sale_date),
  QUARTER(sale_date),
  MONTH(sale_date)
);
```
```sql
-- Subtotals + grand total automatically
SELECT department, job_title, SUM(salary)
FROM employees
GROUP BY ROLLUP(department, job_title);
-- Adds a subtotal row per department + a grand total row at end
> 💡 In FAANG interviews — if you see subtotal NULLs in a result set, think ROLLUP.
```
---

## Practice Questions

### Q1 — Easy ✅
**Table:** `orders(order_id, customer_id, amount, status, order_date)`
Total revenue and number of orders per status, sorted by revenue descending.

```sql
SELECT status,
  SUM(amount) AS total_revenue,
  COUNT(*)    AS num_orders
FROM orders
GROUP BY status
ORDER BY total_revenue DESC;
```

---

### Q2 — Medium ✅
Customers who placed more than 3 orders AND spent over $500 total.

```sql
SELECT customer_id,
  SUM(amount) AS total_spend,
  COUNT(*)    AS order_count
FROM orders
GROUP BY customer_id
HAVING COUNT(*) > 3
  AND SUM(amount) > 500;
```

---

### Q3 — Hard ✅
Per month in 2023, departments with average salary above the company average.

```sql
SELECT MONTH(hire_date) AS month,
  department,
  AVG(salary) AS avg_sal
FROM employees
WHERE YEAR(hire_date) = 2023
GROUP BY MONTH(hire_date), department
HAVING AVG(salary) > (SELECT AVG(salary) FROM employees)
ORDER BY month, avg_sal DESC;
```

---

## Key Takeaways

- **COUNT(\*)** counts all rows; **COUNT(col)** skips NULLs
- **GROUP BY** — every non-aggregated SELECT column must be here
- **WHERE** filters rows before grouping; **HAVING** filters groups after aggregation
- **CASE WHEN in GROUP BY** — must repeat the full expression, not the alias
- **COUNT(DISTINCT col)** — unique non-null values only
- **ROLLUP** — auto subtotals + grand total; NULL in result = rollup row

---

## 1. GROUP BY with CASE WHEN

**The core idea:** You're creating a new "virtual column" on the fly, then grouping by it. SQL evaluates the CASE WHEN *before* grouping happens.

**Why you can't use the alias in GROUP BY:**
SQL processes clauses in this order: `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY`

GROUP BY runs *before* SELECT, so the alias doesn't exist yet when GROUP BY executes. The alias is only born in the SELECT phase.

```sql
-- ❌ This fails — alias doesn't exist at GROUP BY time
SELECT
  CASE WHEN salary >= 100000 THEN 'High' ELSE 'Low' END AS band,
  COUNT(*)
FROM employees
GROUP BY band;  -- "band" is unknown here

-- ✅ Option 1: Repeat the full expression (standard SQL, works everywhere)
GROUP BY CASE WHEN salary >= 100000 THEN 'High' ELSE 'Low' END

-- ✅ Option 2: Use a subquery / CTE — write it once, reuse cleanly
WITH bucketed AS (
  SELECT
    CASE WHEN salary >= 100000 THEN 'High' ELSE 'Low' END AS band,
    salary
  FROM employees
)
SELECT band, COUNT(*) FROM bucketed GROUP BY band;

-- ✅ Option 3: PostgreSQL / MySQL allow alias in GROUP BY (non-standard)
GROUP BY band;  -- works in Postgres/MySQL, NOT in SQL Server/Oracle
```

**Mental model — what SQL actually does:**
```
Row 1: salary=120000 → CASE evaluates → 'High'   ──┐
Row 2: salary=80000  → CASE evaluates → 'Low'    ──┤ grouped into buckets
Row 3: salary=95000  → CASE evaluates → 'Low'    ──┤ then COUNT(*) per bucket
Row 4: salary=160000 → CASE evaluates → 'High'   ──┘
```

---

## 2. DISTINCT in Aggregates

**The core idea:** `DISTINCT` inside an aggregate function deduplicates *before* the aggregation math runs.

```
COUNT(*)           → counts every row, including duplicates
COUNT(user_id)     → counts every non-NULL user_id, including duplicates  
COUNT(DISTINCT user_id) → deduplicates user_ids first, then counts
```

**The classic FAANG pattern — % of users who did X:**
```sql
SELECT
  event_type,
  COUNT(DISTINCT user_id) AS did_this,
  ROUND(
    COUNT(DISTINCT user_id) * 100.0 /
    (SELECT COUNT(DISTINCT user_id) FROM user_events),  -- total unique users
  2) AS pct
FROM user_events
GROUP BY event_type;
```

Why `* 100.0` and not `* 100`? Integer division truncates in most databases:
```
5 / 20     = 0       ❌  (integer division)
5 * 100.0 / 20 = 25.0  ✅  (forced float division)
```

**`SUM(DISTINCT ...)` — the weird one:**
```sql
-- salary table: [50000, 50000, 80000, 80000, 100000]
SUM(salary)          → 380000  (adds everything)
SUM(DISTINCT salary) → 230000  (adds 50000 + 80000 + 100000, ignores dupes)
```
Rarely useful for business logic — usually a mistake. The main valid use case is avoiding double-counting in joins where rows get duplicated.

**Common trap — COUNT(*) vs COUNT(col) vs COUNT(DISTINCT col):**
```sql
-- users table: [1, 2, 2, 3, NULL, NULL]
COUNT(*)              → 6  (every row)
COUNT(user_id)        → 4  (excludes NULLs)
COUNT(DISTINCT user_id) → 3  (excludes NULLs + dedupes: 1,2,3)
```

---

## 3. ROLLUP

**The core idea:** ROLLUP automatically adds subtotal rows and a grand total row to your GROUP BY results — without writing UNION ALL manually.

**Basic example:**
```sql
SELECT department, job_title, SUM(salary) AS total
FROM employees
GROUP BY ROLLUP(department, job_title);
```

This produces:
```
department   | job_title  | total
-------------|------------|--------
Engineering  | Engineer   | 300000   ← normal group
Engineering  | Manager    | 150000   ← normal group
Engineering  | NULL       | 450000   ← subtotal for Engineering dept
Sales        | Rep        | 200000   ← normal group
Sales        | NULL       | 200000   ← subtotal for Sales dept
NULL         | NULL       | 650000   ← grand total
```

The **NULL values are the subtotals** — that's how ROLLUP signals "this row is a rollup/summary row."

**Hierarchy matters:** ROLLUP rolls up from *right to left*. The last column collapses first.
```sql
GROUP BY ROLLUP(a, b, c)
-- generates groupings: (a,b,c), (a,b), (a), ()
-- NOT: (a,c) or (b,c) — it's strictly hierarchical
```

**Distinguishing real NULLs from rollup NULLs — use GROUPING():**
```sql
SELECT
  CASE WHEN GROUPING(department) = 1 THEN 'ALL DEPTS' ELSE department END AS dept,
  CASE WHEN GROUPING(job_title) = 1  THEN 'ALL TITLES' ELSE job_title END AS title,
  SUM(salary)
FROM employees
GROUP BY ROLLUP(department, job_title);
```
`GROUPING(col)` returns `1` if that NULL was introduced by ROLLUP, `0` if it's a real NULL from data.

**ROLLUP vs CUBE vs GROUPING SETS — the family:**
```sql
-- ROLLUP: hierarchical subtotals only (most common)
GROUP BY ROLLUP(a, b)
-- generates: (a,b), (a), ()

-- CUBE: every possible combination of subtotals
GROUP BY CUBE(a, b)
-- generates: (a,b), (a), (b), ()

-- GROUPING SETS: you pick exactly which groupings you want
GROUP BY GROUPING SETS((a,b), (a), ())
-- generates: exactly what you listed
```

**Real-world use — sales dashboard:**
```sql
SELECT
  COALESCE(region, 'TOTAL')       AS region,
  COALESCE(product, 'ALL PRODUCTS') AS product,
  SUM(revenue)   AS revenue,
  COUNT(*)       AS orders
FROM sales
GROUP BY ROLLUP(region, product)
ORDER BY region, product;
```

---

**Quick cheat sheet:**

| Concept | Key rule | Common trap |
|---|---|---|
| GROUP BY CASE WHEN | Repeat full expression (or use CTE) | Alias doesn't exist at GROUP BY time |
| COUNT(DISTINCT x) | Dedupes before counting | NULL is excluded silently |
| ROLLUP | NULL = subtotal row, right-to-left collapse | Use GROUPING() to tell real NULLs apart |
