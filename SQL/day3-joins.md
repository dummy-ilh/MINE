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
🟢 Q1 — Easy

Tables: employees(emp_id, name, dept_id, salary) and departments(dept_id, dept_name)
Get each employee's name and their department name. Include employees with no department.


🟡 Q2 — Medium

Same tables plus projects(project_id, dept_id, budget)
Find departments that have no projects assigned. Return dept_id, dept_name.


🔴 Q3 — Hard

Table: employees(emp_id, name, manager_id, salary)
Find all employees who earn more than their direct manager. Return employee name, employee salary, manager name, manager salary.
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
# SQL Joins — Complete Guide

---

## What is a JOIN?

A JOIN combines rows from two tables based on a shared column (usually a foreign key).
The type of JOIN controls what happens when a row in one table has **no match** in the other.

---

## 1. INNER JOIN

**Returns:** Only rows that have a match in **both** tables. Unmatched rows are silently dropped.

**Use when:**
- You only want complete data — no gaps
- You're reporting on things that definitely exist on both sides
- Example: Show all customers who have placed at least one order

**Do NOT use when:**
- You need to see customers who haven't ordered yet (they'll vanish from results)
- You're auditing for missing data

**Real-world examples:**
- Show all employees and their assigned department names
- List all products that have been ordered at least once
- Get all students who have enrolled in a course

```sql
SELECT c.name, o.total
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id;
```

---

## 2. LEFT JOIN (LEFT OUTER JOIN)

**Returns:** All rows from the **left** table + matching rows from the right. If no match, right-side columns are `NULL`.

**Use when:**
- The left table is your main focus and the right table is optional extra info
- You want to include records even if they have no related data
- You want to find rows with NO match (add `WHERE right.id IS NULL`)

**Do NOT use when:**
- You strictly need both sides to have data (use INNER JOIN instead)

**Real-world examples:**
- List all customers, showing their last order (or NULL if they've never ordered)
- Show all employees and their manager's name (new hires may have no manager yet)
- Find all products that have **never** been ordered: `WHERE o.id IS NULL`
- List all blog posts and their comments (posts with no comments still appear)

```sql
-- All customers, with or without orders
SELECT c.name, o.total
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;

-- Find customers with NO orders at all
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;
```

---

## 3. RIGHT JOIN (RIGHT OUTER JOIN)

**Returns:** All rows from the **right** table + matching rows from the left. Left-side columns are `NULL` if no match.

**Use when:**
- The right table is your main focus
- You inherited a query where the table order is fixed

**Do NOT use when:**
- You can just swap the table order and use a LEFT JOIN — which is almost always clearer to read

**Real-world examples:**
- List all departments, even those with no employees assigned yet
- Show all job positions, with applicants if any exist

```sql
SELECT c.name, o.total
FROM customers c
RIGHT JOIN orders o ON c.id = o.customer_id;
-- Same result as: orders LEFT JOIN customers
```

> 💡 **Tip:** Most developers avoid RIGHT JOIN entirely — just swap the table order and use LEFT JOIN for consistency.

---

## 4. FULL OUTER JOIN

**Returns:** All rows from **both** tables. Unmatched rows on either side show `NULL` for the other table's columns.

**Use when:**
- No table is more important than the other
- You're reconciling two data sources and need to see what's missing on either side
- You're merging datasets and want to catch gaps

**Do NOT use when:**
- You only need matched rows (use INNER JOIN — FULL OUTER is expensive on large tables)
- Your database is MySQL — it doesn't support FULL OUTER JOIN natively (use UNION of LEFT + RIGHT)

**Real-world examples:**
- Compare this month's sales vs last month's — see products that appeared in one but not the other
- Sync two systems: find records in System A missing from System B and vice versa
- Audit employee records across two HR databases

```sql
SELECT c.name, o.total
FROM customers c
FULL OUTER JOIN orders o ON c.id = o.customer_id;
```

---

## 5. CROSS JOIN

**Returns:** Every possible combination of rows from both tables. If Table A has 4 rows and Table B has 3, you get 4 × 3 = 12 rows.

**Use when:**
- You intentionally need every combination
- Generating test/seed data
- Building combination matrices

**Do NOT use when:**
- You forgot to write an ON clause — an accidental CROSS JOIN on large tables can crash your database
- Tables are large (1000 × 1000 = 1,000,000 rows)

**Real-world examples:**
- Generate all size + color combinations for a clothing product (S/M/L × Red/Blue/Green)
- Create a schedule grid: every employee × every shift slot
- Build a multiplication table or probability matrix

```sql
SELECT s.size, c.color
FROM sizes s
CROSS JOIN colors c;
```

---

## 6. SELF JOIN

**Returns:** Rows from a table joined against itself, using two aliases to tell them apart.

**Use when:**
- Rows in a table have a relationship with other rows in the **same** table
- Working with hierarchical or peer data

**Do NOT use when:**
- The relationship spans two different tables (use a normal JOIN)
- You haven't aliased both sides — it will error

**Real-world examples:**
- Show each employee alongside their manager's name (both in the `employees` table)
- Find pairs of products in the same category
- Detect duplicate rows: join a table to itself on identical columns
- Find all flights departing from and arriving at the same city on the same day

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
JOIN employees m ON e.manager_id = m.id;
```

---

## Quick Decision Cheatsheet

| Scenario | Use |
|---|---|
| Only want complete matched data | `INNER JOIN` |
| Left table is main, right is optional | `LEFT JOIN` |
| Find rows with NO match in another table | `LEFT JOIN ... WHERE right.id IS NULL` |
| Right table is main, left is optional | `RIGHT JOIN` (or flip + LEFT JOIN) |
| Need everything from both, gaps included | `FULL OUTER JOIN` |
| Every combination of two tables | `CROSS JOIN` |
| Table relates to itself (hierarchy, peers) | `SELF JOIN` |

---

## Common Mistakes

- **Accidentally getting a CROSS JOIN** — forgetting the `ON` clause in old-style SQL
- **Using INNER JOIN when you meant LEFT JOIN** — silently losing rows with no match
- **FULL OUTER JOIN in MySQL** — not supported; use `LEFT JOIN UNION RIGHT JOIN` instead
- **Forgetting aliases in SELF JOIN** — both sides must have different names

---


## 9. Tricky FAANG JOIN Patterns

---

### Pattern 1: Range / Inequality JOIN

- Join on a range instead of a single value
- Useful for tier assignment, pricing bands, date buckets

```sql
-- Assign discount tier based on order amount range
SELECT o.order_id, o.amount, t.tier_name, t.discount_pct
FROM orders o
JOIN discount_tiers t
  ON o.amount BETWEEN t.min_amount AND t.max_amount;
```

---

### Pattern 2: Time-Window JOIN (Meta / Google Ads)

- Join rows only if they fall within a time gap of each other
- Common in ad attribution, session analysis, funnel tracking

```sql
-- Clicks within 1 hour of ad impression
SELECT i.user_id, i.imp_time, c.click_time
FROM impressions i
JOIN clicks c
  ON  i.user_id   = c.user_id
  AND c.click_time BETWEEN i.imp_time AND i.imp_time + INTERVAL 1 HOUR;
```

---

### Pattern 3: JOIN on Subquery Aggregate

- Subquery computes a group-level stat (MAX, MIN, COUNT), then JOIN filters to only matching rows
- Classic for "top N per group" problems

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

---

### Pattern 4: CROSS JOIN + Anti-JOIN (Recommendations)

- CROSS JOIN generates all possible pairs
- LEFT JOIN + `WHERE NULL` filters out pairs that already exist
- Used for "what hasn't this user seen/bought?" (Netflix, Amazon)

```sql
-- User-product combinations never purchased
SELECT u.user_id, p.product_id
FROM users u
CROSS JOIN products p
LEFT JOIN purchases pur
  ON  u.user_id    = pur.user_id
  AND p.product_id = pur.product_id
WHERE pur.purchase_id IS NULL;
```

---

### Pattern 5: Chained LEFT JOINs — Filter in ON vs WHERE

- Putting a filter in `WHERE` on a LEFT JOIN silently converts it to an INNER JOIN
- Always move optional filters into the `ON` clause to preserve nulls

```sql
-- ❌ Wrong — WHERE kills rows where dept is NULL (acts like INNER JOIN)
SELECT e.name, d.dept_name, p.project_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
LEFT JOIN projects p    ON d.dept_id = p.dept_id
WHERE d.location = 'NYC';

-- ✅ Correct — filter inside ON keeps all employees
SELECT e.name, d.dept_name, p.project_name
FROM employees e
LEFT JOIN departments d
  ON  e.dept_id  = d.dept_id
  AND d.location = 'NYC'
LEFT JOIN projects p ON d.dept_id = p.dept_id;
```

---

### Pattern 6: Consecutive Events (Retention / Fraud)

- Self-join on the same table with a time offset condition
- Used for streak detection, fraud signals, funnel step sequencing

```sql
-- Users who logged in on two consecutive days
SELECT DISTINCT a.user_id
FROM logins a
JOIN logins b
  ON  a.user_id    = b.user_id
  AND b.login_date = a.login_date + INTERVAL 1 DAY;
```

```sql
-- Two events within 5 minutes for the same user
SELECT a.user_id, a.event_type AS event1, b.event_type AS event2
FROM events a
JOIN events b
  ON  a.user_id    = b.user_id
  AND b.event_time BETWEEN a.event_time AND a.event_time + INTERVAL 5 MINUTE
  AND a.event_time < b.event_time;
```

---

### Pattern 7: USING vs ON

- `ON` works everywhere, even when column names differ
- `USING` is cleaner shorthand when the column name is identical in both tables — and deduplicates it in the result set

```sql
-- ON (always works)
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;

-- USING (cleaner when column names match exactly)
-- dept_id appears only once in the result set
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d USING (dept_id);
```

---


# SQL Joins — Complete Guide

---

## What is a JOIN?

A JOIN combines rows from two tables based on a shared column (usually a foreign key).
The type of JOIN controls what happens when a row in one table has **no match** in the other.

---

## 1. INNER JOIN

**Returns:** Only rows that have a match in **both** tables. Unmatched rows are silently dropped.

**Use when:**
- You only want complete data — no gaps
- You're reporting on things that definitely exist on both sides
- Example: Show all customers who have placed at least one order

**Do NOT use when:**
- You need to see customers who haven't ordered yet (they'll vanish from results)
- You're auditing for missing data

**Real-world examples:**
- Show all employees and their assigned department names
- List all products that have been ordered at least once
- Get all students who have enrolled in a course

```sql
SELECT c.name, o.total
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id;
```

---

## 2. LEFT JOIN (LEFT OUTER JOIN)

**Returns:** All rows from the **left** table + matching rows from the right. If no match, right-side columns are `NULL`.

**Use when:**
- The left table is your main focus and the right table is optional extra info
- You want to include records even if they have no related data
- You want to find rows with NO match (add `WHERE right.id IS NULL`)

**Do NOT use when:**
- You strictly need both sides to have data (use INNER JOIN instead)

**Real-world examples:**
- List all customers, showing their last order (or NULL if they've never ordered)
- Show all employees and their manager's name (new hires may have no manager yet)
- Find all products that have **never** been ordered: `WHERE o.id IS NULL`
- List all blog posts and their comments (posts with no comments still appear)

```sql
-- All customers, with or without orders
SELECT c.name, o.total
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;

-- Find customers with NO orders at all
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;
```

---

## 3. RIGHT JOIN (RIGHT OUTER JOIN)

**Returns:** All rows from the **right** table + matching rows from the left. Left-side columns are `NULL` if no match.

**Use when:**
- The right table is your main focus
- You inherited a query where the table order is fixed

**Do NOT use when:**
- You can just swap the table order and use a LEFT JOIN — which is almost always clearer to read

**Real-world examples:**
- List all departments, even those with no employees assigned yet
- Show all job positions, with applicants if any exist

```sql
SELECT c.name, o.total
FROM customers c
RIGHT JOIN orders o ON c.id = o.customer_id;
-- Same result as: orders LEFT JOIN customers
```

> 💡 **Tip:** Most developers avoid RIGHT JOIN entirely — just swap the table order and use LEFT JOIN for consistency.

---

## 4. FULL OUTER JOIN

**Returns:** All rows from **both** tables. Unmatched rows on either side show `NULL` for the other table's columns.

**Use when:**
- No table is more important than the other
- You're reconciling two data sources and need to see what's missing on either side
- You're merging datasets and want to catch gaps

**Do NOT use when:**
- You only need matched rows (use INNER JOIN — FULL OUTER is expensive on large tables)
- Your database is MySQL — it doesn't support FULL OUTER JOIN natively (use UNION of LEFT + RIGHT)

**Real-world examples:**
- Compare this month's sales vs last month's — see products that appeared in one but not the other
- Sync two systems: find records in System A missing from System B and vice versa
- Audit employee records across two HR databases

```sql
SELECT c.name, o.total
FROM customers c
FULL OUTER JOIN orders o ON c.id = o.customer_id;
```

---

## 5. CROSS JOIN

**Returns:** Every possible combination of rows from both tables. If Table A has 4 rows and Table B has 3, you get 4 × 3 = 12 rows.

**Use when:**
- You intentionally need every combination
- Generating test/seed data
- Building combination matrices

**Do NOT use when:**
- You forgot to write an ON clause — an accidental CROSS JOIN on large tables can crash your database
- Tables are large (1000 × 1000 = 1,000,000 rows)

**Real-world examples:**
- Generate all size + color combinations for a clothing product (S/M/L × Red/Blue/Green)
- Create a schedule grid: every employee × every shift slot
- Build a multiplication table or probability matrix

```sql
SELECT s.size, c.color
FROM sizes s
CROSS JOIN colors c;
```

---

## 6. SELF JOIN

**Returns:** Rows from a table joined against itself, using two aliases to tell them apart.

**Use when:**
- Rows in a table have a relationship with other rows in the **same** table
- Working with hierarchical or peer data

**Do NOT use when:**
- The relationship spans two different tables (use a normal JOIN)
- You haven't aliased both sides — it will error

**Real-world examples:**
- Show each employee alongside their manager's name (both in the `employees` table)
- Find pairs of products in the same category
- Detect duplicate rows: join a table to itself on identical columns
- Find all flights departing from and arriving at the same city on the same day

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
JOIN employees m ON e.manager_id = m.id;
```

---

## Quick Decision Cheatsheet

| Scenario | Use |
|---|---|
| Only want complete matched data | `INNER JOIN` |
| Left table is main, right is optional | `LEFT JOIN` |
| Find rows with NO match in another table | `LEFT JOIN ... WHERE right.id IS NULL` |
| Right table is main, left is optional | `RIGHT JOIN` (or flip + LEFT JOIN) |
| Need everything from both, gaps included | `FULL OUTER JOIN` |
| Every combination of two tables | `CROSS JOIN` |
| Table relates to itself (hierarchy, peers) | `SELF JOIN` |

---

## Common Mistakes

- **Accidentally getting a CROSS JOIN** — forgetting the `ON` clause in old-style SQL
- **Using INNER JOIN when you meant LEFT JOIN** — silently losing rows with no match
- **FULL OUTER JOIN in MySQL** — not supported; use `LEFT JOIN UNION RIGHT JOIN` instead
- **Forgetting aliases in SELF JOIN** — both sides must have different names

---

## 9. Tricky FAANG JOIN Patterns

---

### Pattern 1: Range / Inequality JOIN

`orders`
| order_id | amount |
|----------|--------|
| 1001 | 45 |
| 1002 | 130 |
| 1003 | 280 |

`discount_tiers`
| tier_name | min_amount | max_amount | discount_pct |
|-----------|------------|------------|--------------|
| Bronze | 0 | 99 | 5% |
| Silver | 100 | 199 | 10% |
| Gold | 200 | 999 | 15% |

**Result:**
| order_id | amount | tier_name | discount_pct |
|----------|--------|-----------|--------------|
| 1001 | 45 | Bronze | 5% |
| 1002 | 130 | Silver | 10% |
| 1003 | 280 | Gold | 15% |

---

### Pattern 2: Time-Window JOIN

`impressions`
| user_id | imp_time |
|---------|----------|
| u1 | 10:00 |
| u2 | 10:05 |

`clicks`
| user_id | click_time |
|---------|------------|
| u1 | 10:42 |
| u1 | 11:30 |
| u2 | 12:00 |

**Result** (within 1 hour of impression):
| user_id | imp_time | click_time |
|---------|----------|------------|
| u1 | 10:00 | 10:42 |

> u1's 11:30 click excluded (>1hr). u2 excluded (no match in window).

---

### Pattern 3: JOIN on Subquery Aggregate

`employees`
| name | department | salary |
|------|------------|--------|
| Alice | Eng | 120k |
| Bob | Eng | 95k |
| Carol | HR | 80k |
| Dave | HR | 72k |

**Subquery intermediate:**
| department | max_sal |
|------------|---------|
| Eng | 120k |
| HR | 80k |

**Result:**
| name | department | salary |
|------|------------|--------|
| Alice | Eng | 120k |
| Carol | HR | 80k |

---

### Pattern 4: CROSS JOIN + Anti-JOIN

`users`
| user_id |
|---------|
| u1 |
| u2 |

`products`
| product_id |
|------------|
| p1 |
| p2 |
| p3 |

`purchases` (already bought)
| user_id | product_id |
|---------|------------|
| u1 | p1 |
| u2 | p2 |

**Result** (2 × 3 = 6 combos minus 2 existing):
| user_id | product_id |
|---------|------------|
| u1 | p2 |
| u1 | p3 |
| u2 | p1 |
| u2 | p3 |

---

### Pattern 5: Chained LEFT JOINs — ON vs WHERE

`employees`
| name | dept_id |
|------|---------|
| Alice | 1 |
| Bob | 2 |
| Carol | NULL |

`departments`
| dept_id | dept_name | location |
|---------|-----------|----------|
| 1 | Engineering | NYC |
| 2 | HR | LA |

**❌ WHERE result** (Carol silently dropped):
| name | dept_name |
|------|-----------|
| Alice | Engineering |

**✅ ON result** (Carol preserved):
| name | dept_name |
|------|-----------|
| Alice | Engineering |
| Bob | NULL |
| Carol | NULL |

---

### Pattern 6: Consecutive Events

`logins` (self-joined)
| user_id | login_date |
|---------|------------|
| u1 | Mar 1 |
| u1 | Mar 2 |
| u1 | Mar 5 |
| u2 | Mar 1 |
| u2 | Mar 3 |

**Result:**
| user_id |
|---------|
| u1 |

> u1 matched Mar 1 → Mar 2. u2 skipped Mar 2, no consecutive pair.

---

### Pattern 7: USING vs ON

`employees`
| name | dept_id |
|------|---------|
| Alice | 1 |
| Bob | 2 |

`departments`
| dept_id | dept_name |
|---------|-----------|
| 1 | Engineering |
| 2 | HR |

**ON result** (dept_id appears twice):
| name | e.dept_id | d.dept_id | dept_name |
|------|-----------|-----------|-----------|
| Alice | 1 | 1 | Engineering |
| Bob | 2 | 2 | HR |

**USING result** (dept_id appears once):
| name | dept_id | dept_name |
|------|---------|-----------|
| Alice | 1 | Engineering |
| Bob | 2 | HR |

---
