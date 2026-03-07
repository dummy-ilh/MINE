# Day 3 — JOINs (Core + Tricky FAANG Patterns)
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. JOIN Types Overview
2. INNER JOIN
3. LEFT JOIN + Anti-Join
4. RIGHT JOIN
5. FULL OUTER JOIN
6. SELF JOIN
7. CROSS JOIN
8. JOIN Traps
9. Tricky FAANG JOIN Patterns

---

## 1. JOIN Types Overview

| JOIN Type | What you get |
|---|---|
| INNER JOIN | Only matching rows from both tables |
| LEFT JOIN | All of left + matches from right (NULL if no match) |
| RIGHT JOIN | All of right + matches from left (NULL if no match) |
| FULL OUTER JOIN | Everything from both tables |
| CROSS JOIN | Every row of A × every row of B |
| SELF JOIN | Table joined with itself |

---

## 2. INNER JOIN

```sql
SELECT e.name, e.salary, d.dept_name
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;
-- Excludes employees with no dept and depts with no employees
```

```sql
-- Multi-table INNER JOIN
SELECT c.name, p.product_name, o.amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN products p  ON o.product_id  = p.product_id;
```

---

## 3. LEFT JOIN + Anti-Join

```sql
-- All employees, with their department if it exists
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;
-- employees with no department → dept_name = NULL
```

```sql
-- Classic FAANG pattern: find rows with NO match (anti-join)
-- "Find employees who are NOT assigned to any department"
SELECT e.name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
WHERE d.dept_id IS NULL;  -- ← the key: filter where right side is NULL
```

> ⚠️ **LEFT JOIN + WHERE right.col IS NULL = Anti-join.** Finds rows that DON'T exist in the other table. Asked constantly at Meta/Google.

---

## 4. RIGHT JOIN

```sql
-- All departments, even with no employees
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;

-- Tip: Rewrite as LEFT JOIN by swapping tables (cleaner)
SELECT e.name, d.dept_name
FROM departments d
LEFT JOIN employees e ON e.dept_id = d.dept_id;
```
```
-- These two are identical:
SELECT * FROM A RIGHT JOIN B ON A.id = B.id;
SELECT * FROM B LEFT JOIN A  ON A.id = B.id;```
---
```
## 5. FULL OUTER JOIN
Returns everything from both tables. NULLs fill in where there's no match.
```sql
-- Everything from both tables
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- Find unmatched on BOTH sides
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id
WHERE e.emp_id IS NULL OR d.dept_id IS NULL;
```

```sql
-- MySQL workaround (no FULL OUTER JOIN support)
SELECT e.name, d.dept_name
FROM employees e LEFT JOIN departments d ON e.dept_id = d.dept_id
UNION
SELECT e.name, d.dept_name
FROM employees e RIGHT JOIN departments d ON e.dept_id = d.dept_id;
```

---

## 6. SELF JOIN
A table joined with itself. Used for hierarchies, comparisons within same table.
```sql
-- employees(emp_id, name, manager_id)
-- manager_id references emp_id in the same table

-- Find each employee and their manager's name
SELECT
  e.name        AS employee,
  m.name        AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;
-- LEFT JOIN so employees with no manager (CEO) still appear
```

```sql
-- Pairs of employees in same dept within $5000 salary of each other
SELECT a.name AS emp1, b.name AS emp2, a.salary, b.salary
FROM employees a
JOIN employees b
  ON  a.dept_id = b.dept_id
  AND a.emp_id  < b.emp_id   -- avoid (A,B) and (B,A) duplicates
  AND ABS(a.salary - b.salary) <= 5000;
```

---

## 7. CROSS JOIN
Every row of A combined with every row of B. M × N rows.
```sql
-- All size-color combinations (M × N rows)
SELECT s.size, c.color
FROM sizes s
CROSS JOIN colors c;
```

```sql
-- Common FAANG use: generate a date spine
-- dates table has one row per day
-- combine with all users to get user × day grid for activity analysis
SELECT u.user_id, d.date
FROM users u
CROSS JOIN dates d
WHERE d.date BETWEEN '2023-01-01' AND '2023-12-31';
```

---

## 8. JOIN Traps

### Trap 1: Duplicates from JOIN
```sql
-- If join key has duplicates in either table, rows multiply
-- Always verify uniqueness of join key before joining
-- If departments has duplicate dept_ids, your JOIN multiplies rows
-- Always check: SELECT COUNT(*) FROM departments WHERE dept_id IN (SELECT dept_id FROM departments GROUP BY dept_id HAVING COUNT(*) > 1)
SELECT COUNT(*), COUNT(DISTINCT dept_id) FROM departments;
```

### Trap 2: Filtering in ON vs WHERE (Critical!)
```sql
-- ❌ WHERE filter turns LEFT JOIN into INNER JOIN
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
WHERE d.location = 'NYC';  -- loses employees with no dept

-- ✅ Filter in ON preserves LEFT JOIN behavior
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d
  ON  e.dept_id  = d.dept_id
  AND d.location = 'NYC';  -- unmatched employees still appear
```

### Trap 3: NULL keys never match
```sql
-- NULL = NULL is FALSE in SQL
-- Rows with NULL join keys always produce NULL on the other side
```

---

## 9. Tricky FAANG JOIN Patterns

### Pattern 1: Range / Inequality JOIN
```sql
-- Assign discount tier based on order amount range
SELECT o.order_id, o.amount, t.tier_name, t.discount_pct
FROM orders o
JOIN discount_tiers t
  ON o.amount BETWEEN t.min_amount AND t.max_amount;
```

### Pattern 2: Time-Window JOIN (Meta/Google Ads)
```sql
-- Clicks within 1 hour of ad impression
SELECT i.user_id, i.imp_time, c.click_time
FROM impressions i
JOIN clicks c
  ON  i.user_id   = c.user_id
  AND c.click_time BETWEEN i.imp_time AND i.imp_time + INTERVAL 1 HOUR;
```

### Pattern 3: JOIN on Subquery Aggregate (Top 5 FAANG Pattern)
```sql
-- Employee with max salary per department
SELECT e.name, e.department, e.salary
FROM employees e
JOIN (
  SELECT department, MAX(salary) AS max_sal
  FROM employees
  GROUP BY department
) dept_max
  ON  e.department = dept_max.department
  AND e.salary     = dept_max.max_sal;
```

```sql
-- Most recent order per customer
SELECT o.customer_id, o.order_id, o.amount, o.order_date
FROM orders o
JOIN (
  SELECT customer_id, MAX(order_date) AS latest
  FROM orders
  GROUP BY customer_id
) last_order
  ON  o.customer_id = last_order.customer_id
  AND o.order_date  = last_order.latest;
```

### Pattern 4: CROSS JOIN + Anti-Join (Recommendations)
```sql
-- User-product combinations never purchased (Netflix/Amazon pattern)
SELECT u.user_id, p.product_id
FROM users u
CROSS JOIN products p
LEFT JOIN purchases pur
  ON  u.user_id    = pur.user_id
  AND p.product_id = pur.product_id
WHERE pur.purchase_id IS NULL;
```

### Pattern 5: Chained LEFT JOINs — Filter in ON
```sql
-- ❌ Wrong — WHERE breaks the LEFT JOIN chain
SELECT e.name, d.dept_name, p.project_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
LEFT JOIN projects p    ON d.dept_id = p.dept_id
WHERE d.location = 'NYC';

-- ✅ Correct — filter inside ON
SELECT e.name, d.dept_name, p.project_name
FROM employees e
LEFT JOIN departments d
  ON  e.dept_id  = d.dept_id
  AND d.location = 'NYC'
LEFT JOIN projects p ON d.dept_id = p.dept_id;
```

### Pattern 6: Consecutive Events (Retention/Fraud)
```sql
-- Users who logged in on two consecutive days
SELECT DISTINCT a.user_id
FROM logins a
JOIN logins b
  ON  a.user_id   = b.user_id
  AND b.login_date = a.login_date + INTERVAL 1 DAY;
```

```sql
-- Two events within 5 minutes for same user
SELECT a.user_id, a.event_type AS event1, b.event_type AS event2
FROM events a
JOIN events b
  ON  a.user_id   = b.user_id
  AND b.event_time BETWEEN a.event_time AND a.event_time + INTERVAL 5 MINUTE
  AND a.event_time < b.event_time;
```

### Pattern 7: USING vs ON
```sql
-- ON (always works)
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- USING (cleaner, only when column names match exactly)
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d USING (dept_id);
-- dept_id appears only once in result set
```

---

## Practice Questions

### Q1 — Easy ✅
Get each employee's name and department name. Include employees with no department.

```sql
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;
```

### Q2 — Medium ✅
Find departments with no projects assigned.

```sql
SELECT d.dept_id, d.dept_name
FROM departments d
LEFT JOIN projects p ON d.dept_id = p.dept_id
WHERE p.dept_id IS NULL;
```

### Q3 — Hard ✅
Find employees who earn more than their direct manager.

```sql
SELECT
  e.name   AS employee,
  e.salary AS emp_salary,
  m.name   AS manager,
  m.salary AS mgr_salary
FROM employees e
JOIN employees m ON e.manager_id = m.emp_id
WHERE e.salary > m.salary;
```

---

## FAANG JOIN Cheatsheet

| Pattern | Use Case |
|---|---|
| `LEFT JOIN + IS NULL` | Anti-join — find missing/unmatched rows |
| `SELF JOIN` | Hierarchies, consecutive events, within-table comparisons |
| `JOIN on subquery` | Match against group-level aggregates (max, latest) |
| `CROSS JOIN + LEFT JOIN IS NULL` | Find all missing combinations |
| `JOIN with BETWEEN` | Range joins — pricing tiers, time windows |
| `Filter in ON not WHERE` | Preserve LEFT JOIN behavior in chains |
| `SELF JOIN + INTERVAL` | Consecutive day / session analysis |

---

## Key Takeaways

- **Which table goes LEFT?** → The one you want ALL rows from
- **Anti-join** = LEFT JOIN + WHERE right side IS NULL
- **NULL keys never match** — NULL = NULL is FALSE in SQL
- **Filter in ON vs WHERE** — WHERE turns LEFT JOIN into INNER JOIN
- **SELF JOIN** → always use different aliases + `a.id < b.id` to avoid duplicates
- **Time-window JOINs** → massive in ads, ML features, fraud detection
- **JOIN on subquery** → top 5 most asked FAANG pattern

---

Tricky JOINs — FAANG Special

Trick 1: JOIN on Non-Equality Conditions
Most people only know ON a.id = b.id. FAANG tests range and inequality joins.
sql-- Find all employees who earn more than ANY engineer
-- (not just their own department)
SELECT e.name, e.salary, e2.name AS engineer, e2.salary AS eng_salary
FROM employees e
JOIN employees e2
  ON  e2.department = 'Engineering'
  AND e.salary > e2.salary
  AND e.department != 'Engineering';
sql-- Assign discount tiers based on order amount ranges
-- discount_tiers(tier_name, min_amount, max_amount, discount_pct)
SELECT o.order_id, o.amount, t.tier_name, t.discount_pct
FROM orders o
JOIN discount_tiers t
  ON o.amount BETWEEN t.min_amount AND t.max_amount;
-- No equality — pure range join. Very common in pricing/ML feature tables.

Trick 2: Multiple JOIN Conditions
sql-- Match on TWO columns — both must match
-- Prevents wrong matches when IDs repeat across regions
SELECT *
FROM orders_us o
JOIN returns_us r
  ON  o.order_id   = r.order_id
  AND o.customer_id = r.customer_id;  -- extra safety condition
sql-- Time-based JOIN — match events within a time window
-- "Find clicks that happened within 1 hour after an ad impression"
-- impressions(user_id, imp_time), clicks(user_id, click_time)
SELECT i.user_id, i.imp_time, c.click_time
FROM impressions i
JOIN clicks c
  ON  i.user_id   = c.user_id
  AND c.click_time BETWEEN i.imp_time AND i.imp_time + INTERVAL 1 HOUR;

💡 Time-window JOINs are extremely common in Meta/Google ads & ML feature engineering interviews.


Trick 3: Joining on Aggregates (Derived Tables)
sql-- Find employees who earn the MAX salary in their department
SELECT e.name, e.department, e.salary
FROM employees e
JOIN (
  SELECT department, MAX(salary) AS max_sal
  FROM employees
  GROUP BY department
) dept_max
  ON  e.department = dept_max.department
  AND e.salary     = dept_max.max_sal;
sql-- Find the most recent order per customer
SELECT o.customer_id, o.order_id, o.amount, o.order_date
FROM orders o
JOIN (
  SELECT customer_id, MAX(order_date) AS latest
  FROM orders
  GROUP BY customer_id
) last_order
  ON  o.customer_id = last_order.customer_id
  AND o.order_date  = last_order.latest;

💡 This pattern (JOIN on subquery aggregate) is one of the top 5 FAANG SQL patterns. Comes up in almost every DS phone screen.


Trick 4: CROSS JOIN for Gap Detection
sql-- "Which user-product combinations have NEVER been purchased?"
-- Generate all possible pairs, then anti-join against actual purchases

SELECT u.user_id, p.product_id
FROM users u
CROSS JOIN products p

EXCEPT  -- or use LEFT JOIN + IS NULL

SELECT user_id, product_id
FROM purchases;
sql-- Same with LEFT JOIN (works in MySQL too)
SELECT u.user_id, p.product_id
FROM users u
CROSS JOIN products p
LEFT JOIN purchases pur
  ON  u.user_id   = pur.user_id
  AND p.product_id = pur.product_id
WHERE pur.purchase_id IS NULL;

💡 This is the foundation of recommendation system queries at Netflix/Amazon — find what users haven't seen/bought yet.


Trick 5: Chained LEFT JOINs — NULL Propagation Trap
sql-- ❌ Trap: filtering middle table in WHERE breaks outer joins
SELECT e.name, d.dept_name, p.project_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
LEFT JOIN projects p    ON d.dept_id = p.dept_id
WHERE d.location = 'NYC';  -- ❌ kills LEFT JOIN, excludes employees with no dept
sql-- ✅ Filter in ON clause to preserve LEFT JOIN behavior
SELECT e.name, d.dept_name, p.project_name
FROM employees e
LEFT JOIN departments d
  ON  e.dept_id    = d.dept_id
  AND d.location   = 'NYC'       -- ✅ filter stays in ON
LEFT JOIN projects p ON d.dept_id = p.dept_id;

Trick 6: SELF JOIN for Consecutive Events
sql-- "Find users who logged in on two consecutive days"
-- logins(user_id, login_date)

SELECT DISTINCT a.user_id
FROM logins a
JOIN logins b
  ON  a.user_id   = b.user_id
  AND b.login_date = a.login_date + INTERVAL 1 DAY;
sql-- "Find sessions where a user had two events within 5 minutes"
-- events(user_id, event_type, event_time)

SELECT a.user_id, a.event_type AS event1, b.event_type AS event2
FROM events a
JOIN events b
  ON  a.user_id   = b.user_id
  AND b.event_time BETWEEN a.event_time AND a.event_time + INTERVAL 5 MINUTE
  AND a.event_time < b.event_time;  -- avoid self-matching same row

💡 Consecutive event patterns come up in retention, funnel analysis, and fraud detection — all massive DS/MLE interview topics.


Trick 7: USING vs ON
sql-- When both tables have identically named join column
-- ON version
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- USING version (cleaner, but only when column names match exactly)
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d USING (dept_id);
-- dept_id appears only once in result (not duplicated like with ON)
