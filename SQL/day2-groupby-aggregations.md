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

---

## 2. GROUP BY

Every column in SELECT that is **not inside an aggregate** must appear in GROUP BY.

```sql
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
```

> 💡 **Rule:** Filtering raw column? → WHERE. Filtering COUNT/SUM/AVG/MIN/MAX? → HAVING.

---

## 4. GROUP BY with CASE WHEN

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

> 💡 In FAANG interviews — if you see subtotal NULLs in a result set, think ROLLUP.

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

*Day 2 complete — 28 days to go 🚀*
