# Day 4 — Subqueries & CTEs
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Subquery Basics (WHERE / FROM / SELECT)
2. Correlated vs Non-Correlated Subqueries
3. EXISTS vs IN (and NOT EXISTS vs NOT IN)
4. CTEs — WITH Clause
5. Multiple CTEs
6. CTE vs Subquery
7. Scalar Subqueries in SELECT
8. Classic FAANG Subquery Patterns

---

## 1. Subquery Basics

A query inside another query. Can live in SELECT, FROM, or WHERE.

```sql
-- In WHERE
SELECT name FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- In FROM (derived table)
SELECT dept, avg_sal
FROM (
  SELECT department AS dept, AVG(salary) AS avg_sal
  FROM employees
  GROUP BY department
) dept_summary
WHERE avg_sal > 90000;

-- In SELECT (scalar subquery)
SELECT name, salary,
  (SELECT AVG(salary) FROM employees) AS company_avg
FROM employees;
```

### More WHERE Examples

```sql
-- Employees in top 10% salary bracket
SELECT name, salary
FROM employees
WHERE salary >= (
  SELECT PERCENTILE_CONT(0.90)
  WITHIN GROUP (ORDER BY salary)
  FROM employees
);

-- Products never ordered in last 30 days
SELECT product_id, product_name
FROM products
WHERE product_id NOT IN (
  SELECT DISTINCT product_id FROM orders
  WHERE order_date >= CURRENT_DATE - INTERVAL 30 DAY
);

-- Churned users (no order in 90 days)
SELECT user_id, name
FROM users
WHERE user_id IN (
  SELECT user_id FROM orders
  GROUP BY user_id
  HAVING MAX(order_date) < CURRENT_DATE - INTERVAL 90 DAY
);
```

---

## 2. Correlated vs Non-Correlated Subqueries

### Non-Correlated — runs ONCE
```sql
-- Inner query has no reference to outer — runs independently
SELECT name FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
--              ↑ runs once, returns one number
```

### Correlated — runs for EVERY row
```sql
-- Inner query references outer query's current row
SELECT e.name, e.salary
FROM employees e
WHERE e.salary > (
  SELECT AVG(salary) FROM employees
  WHERE department = e.department  -- ← references outer row
);
```

### More Correlated Examples

```sql
-- How many people in same dept earn more than each employee?
SELECT
  name, department, salary,
  (SELECT COUNT(*) FROM employees e2
   WHERE e2.department = e1.department
   AND e2.salary > e1.salary) AS people_earning_more
FROM employees e1
ORDER BY department, salary DESC;
```

```sql
-- Most recent order per customer (correlated)
SELECT o1.customer_id, o1.order_id, o1.amount, o1.order_date
FROM orders o1
WHERE o1.order_date = (
  SELECT MAX(order_date) FROM orders o2
  WHERE o2.customer_id = o1.customer_id
);
```

```sql
-- Employees earning above avg of their own job title
SELECT name, job_title, salary
FROM employees e1
WHERE salary > (
  SELECT AVG(salary) FROM employees e2
  WHERE e2.job_title = e1.job_title
);
```

> ⚠️ Correlated subqueries run once per row — slow at scale. Optimize by rewriting as JOIN or CTE.

---

## 3. EXISTS vs IN

```sql
-- IN: pulls all values then checks membership
SELECT name FROM employees
WHERE dept_id IN (
  SELECT dept_id FROM departments WHERE location = 'NYC'
);

-- EXISTS: short-circuits on first match — faster, NULL-safe
SELECT name FROM employees e
WHERE EXISTS (
  SELECT 1 FROM departments d
  WHERE d.dept_id = e.dept_id
  AND d.location = 'NYC'
);
```

### NOT EXISTS vs NOT IN — Critical Difference

```sql
-- ❌ NOT IN breaks if subquery has any NULLs — returns empty
SELECT name FROM employees
WHERE dept_id NOT IN (
  SELECT dept_id FROM departments WHERE location = 'NYC'
);

-- ✅ NOT EXISTS — always safe
SELECT name FROM employees e
WHERE NOT EXISTS (
  SELECT 1 FROM departments d
  WHERE d.dept_id = e.dept_id
  AND d.location = 'NYC'
);
```

### More EXISTS Examples

```sql
-- Customers with at least one order over $500
SELECT name FROM customers c
WHERE EXISTS (
  SELECT 1 FROM orders o
  WHERE o.customer_id = c.customer_id AND o.amount > 500
);

-- Products never returned
SELECT product_id, product_name FROM products p
WHERE NOT EXISTS (
  SELECT 1 FROM returns r WHERE r.product_id = p.product_id
);

-- Users who signed up but never logged in
SELECT user_id, name FROM users u
WHERE NOT EXISTS (
  SELECT 1 FROM login_events l WHERE l.user_id = u.user_id
);

-- Managers with at least one report earning over 150k
SELECT DISTINCT m.name AS manager
FROM employees m
WHERE EXISTS (
  SELECT 1 FROM employees e
  WHERE e.manager_id = m.emp_id AND e.salary > 150000
);
```

> 💡 Rule: Use EXISTS to check if something exists. Use NOT EXISTS over NOT IN — always NULL-safe.

---

## 4. CTEs — WITH Clause

Named temporary result set. Defined at top, reused below.

```sql
WITH dept_avg AS (
  SELECT department, AVG(salary) AS avg_sal
  FROM employees
  GROUP BY department
)
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN dept_avg d ON e.department = d.department
WHERE e.salary > d.avg_sal;
```

### CTE vs Nested Subquery

```sql
-- ❌ Nested subquery hell
SELECT name FROM employees
WHERE dept_id IN (
  SELECT dept_id FROM departments
  WHERE location IN (
    SELECT location FROM offices WHERE country = 'India'
  )
);

-- ✅ CTE version — readable and debuggable
WITH india_offices AS (
  SELECT location FROM offices WHERE country = 'India'
),
india_depts AS (
  SELECT dept_id FROM departments
  WHERE location IN (SELECT location FROM india_offices)
)
SELECT name FROM employees
WHERE dept_id IN (SELECT dept_id FROM india_depts);
```

---

## 5. Multiple CTEs

Each CTE can reference the ones defined before it.

```sql
-- Full funnel: signups → activated → paid
WITH
signups AS (
  SELECT user_id FROM users WHERE YEAR(created_at) = 2023
),
activated AS (
  SELECT DISTINCT user_id FROM events WHERE event_type = 'activation'
),
paid AS (
  SELECT DISTINCT user_id FROM orders WHERE amount > 0
)
SELECT
  COUNT(DISTINCT s.user_id)                       AS total_signups,
  COUNT(DISTINCT a.user_id)                       AS activated,
  COUNT(DISTINCT p.user_id)                       AS paid,
  ROUND(COUNT(DISTINCT a.user_id) * 100.0 /
        COUNT(DISTINCT s.user_id), 2)             AS activation_rate,
  ROUND(COUNT(DISTINCT p.user_id) * 100.0 /
        COUNT(DISTINCT s.user_id), 2)             AS conversion_rate
FROM signups s
LEFT JOIN activated a ON s.user_id = a.user_id
LEFT JOIN paid p      ON s.user_id = p.user_id;
```

```sql
-- Revenue per user segment
WITH
order_counts AS (
  SELECT user_id, COUNT(*) AS total_orders
  FROM orders GROUP BY user_id
),
user_segments AS (
  SELECT user_id,
    CASE
      WHEN total_orders >= 10 THEN 'VIP'
      WHEN total_orders >= 3  THEN 'Regular'
      ELSE 'Occasional'
    END AS segment
  FROM order_counts
),
segment_revenue AS (
  SELECT s.segment, SUM(o.amount) AS revenue, COUNT(*) AS orders
  FROM orders o
  JOIN user_segments s ON o.user_id = s.user_id
  GROUP BY s.segment
)
SELECT segment, revenue, orders,
  ROUND(revenue / orders, 2) AS avg_order_value
FROM segment_revenue
ORDER BY revenue DESC;
```

---

## 6. CTE vs Subquery

| | Subquery | CTE |
|---|---|---|
| Readability | Messy when nested | Clean named blocks |
| Reuse | ❌ Must repeat | ✅ Define once |
| Debugging | Hard | Easy — test each block |
| Recursion | ❌ No | ✅ Yes |
| Performance | Same in most DBs | Same in most DBs |

---

## 7. Scalar Subqueries in SELECT

```sql
-- Salary vs company avg + % of payroll
SELECT
  name, salary,
  ROUND(salary - (SELECT AVG(salary) FROM employees), 2) AS diff_from_avg,
  ROUND(salary * 100.0 / (SELECT SUM(salary) FROM employees), 2) AS pct_of_payroll
FROM employees
ORDER BY salary DESC;
```

```sql
-- Each employee's salary percentile in their department
SELECT
  name, department, salary,
  ROUND(
    (SELECT COUNT(*) FROM employees e2
     WHERE e2.department = e1.department AND e2.salary <= e1.salary) * 100.0 /
    (SELECT COUNT(*) FROM employees e3
     WHERE e3.department = e1.department), 1
  ) AS dept_percentile
FROM employees e1;
```

```sql
-- Each order's % of that customer's total spend
SELECT
  order_id, customer_id, amount,
  ROUND(amount * 100.0 / (
    SELECT SUM(amount) FROM orders o2
    WHERE o2.customer_id = o1.customer_id
  ), 2) AS pct_of_customer_total
FROM orders o1
ORDER BY customer_id, amount DESC;
```

---

## 8. Classic FAANG Subquery Patterns

```sql
-- Nth highest salary
SELECT DISTINCT salary FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 2;  -- 3rd highest (OFFSET = N-1)
```

```sql
-- Find gaps in sequential IDs
SELECT id + 1 AS gap_start
FROM orders o1
WHERE NOT EXISTS (
  SELECT 1 FROM orders o2 WHERE o2.id = o1.id + 1
)
AND id < (SELECT MAX(id) FROM orders);
```

```sql
-- Customers who ordered EVERY product (relational division)
SELECT customer_id
FROM orders
GROUP BY customer_id
HAVING COUNT(DISTINCT product_id) = (SELECT COUNT(*) FROM products);
```

```sql
-- Products bought together frequently (market basket)
SELECT
  a.product_id AS product_1,
  b.product_id AS product_2,
  COUNT(*) AS times_bought_together
FROM orders a
JOIN orders b
  ON  a.order_id   = b.order_id
  AND a.product_id < b.product_id
GROUP BY a.product_id, b.product_id
HAVING COUNT(*) > 10
ORDER BY times_bought_together DESC;
```

```sql
-- Delete duplicates keeping only latest
DELETE FROM orders
WHERE order_id NOT IN (
  SELECT MAX(order_id)
  FROM orders
  GROUP BY customer_id, product_id, order_date
);
```

---

## Practice Questions

### Q1 — Easy ✅
Find all employees earning above company average salary.

```sql
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

### Q2 — Medium ✅
Using a CTE, find employees earning above their department average.

```sql
WITH dept_avg AS (
  SELECT department, AVG(salary) AS avg_sal
  FROM employees
  GROUP BY department
)
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN dept_avg d ON e.department = d.department
WHERE e.salary > d.avg_sal;
```

### Q3 — Hard ✅
Find customers who ordered every product in the products table.

```sql
SELECT c.customer_id, c.name
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name
HAVING COUNT(DISTINCT o.product_id) = (SELECT COUNT(*) FROM products);
```

---

## Key Takeaways

- **Subquery in WHERE** → filter on aggregated or external condition
- **Subquery in FROM** → treat result as a derived table
- **Subquery in SELECT** → scalar value per row (use sparingly — can be slow)
- **Correlated subquery** → runs once per row, references outer query
- **EXISTS** → fast, NULL-safe, stops at first match
- **NOT EXISTS** → always use over NOT IN (NOT IN breaks on NULLs)
- **CTE** → cleaner than nested subqueries, reusable, debuggable
- **Multiple CTEs** → build complex logic step by step — the FAANG standard

---

*Day 4 complete — 26 days to go 🚀*
