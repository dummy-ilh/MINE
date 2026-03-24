

# Day 11 — Recursive CTEs & Hierarchical Data

This is **Week 2's hardest topic** — org charts, network graphs, bill of materials, path finding. Asked at Google, Meta, Amazon for senior DS/MLE roles.

---

## 1. What is a Recursive CTE?

A CTE that **references itself** — keeps running until no new rows are produced.

```sql
WITH RECURSIVE cte_name AS (
  -- Anchor member: starting point (runs once)
  SELECT ...

  UNION ALL

  -- Recursive member: references cte_name (runs repeatedly)
  SELECT ... FROM cte_name WHERE <stop condition>
)
SELECT * FROM cte_name;
```

> ⚠️ Always include a **stop condition** in WHERE — otherwise it runs forever.

---

## 2. Number Series Generator

```sql
-- Generate numbers 1 to 100
WITH RECURSIVE numbers AS (
  SELECT 1 AS n          -- anchor

  UNION ALL

  SELECT n + 1           -- recursive step
  FROM numbers
  WHERE n < 100          -- stop condition
)
SELECT * FROM numbers;
```

```sql
-- Generate a date range (extremely useful for date spines)
WITH RECURSIVE date_series AS (
  SELECT '2025-01-01' AS dt   -- anchor

  UNION ALL

  SELECT dt + INTERVAL 1 DAY  -- recursive step
  FROM date_series
  WHERE dt < '2025-12-31'     -- stop condition
)
SELECT dt FROM date_series;
```

> 💡 **Date spine** — cross join this with your data to fill gaps (zero revenue days, missing cohort weeks etc.)

---

## 3. Org Chart / Hierarchy Traversal

```sql
-- employees(emp_id, name, manager_id, department)
-- Find all reports under a given manager (any depth)

WITH RECURSIVE org_chart AS (
  -- Anchor: start with the top manager (e.g. CEO, emp_id = 1)
  SELECT emp_id, name, manager_id, 0 AS depth
  FROM employees
  WHERE emp_id = 1

  UNION ALL

  -- Recursive: find direct reports of current level
  SELECT e.emp_id, e.name, e.manager_id, oc.depth + 1
  FROM employees e
  JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT emp_id, name, depth,
  REPEAT('  ', depth) AS indent  -- visual indentation
FROM org_chart
ORDER BY depth, name;
```

```sql
-- Find full reporting chain for a specific employee (bottom up)
WITH RECURSIVE chain AS (
  -- Anchor: start with the employee
  SELECT emp_id, name, manager_id, 0 AS level
  FROM employees
  WHERE emp_id = 42   -- target employee

  UNION ALL

  -- Recursive: walk up to each manager
  SELECT e.emp_id, e.name, e.manager_id, c.level + 1
  FROM employees e
  JOIN chain c ON e.emp_id = c.manager_id
)
SELECT level, name FROM chain ORDER BY level;
-- level 0 = employee, level 1 = direct manager, level 2 = skip-level, etc.
```

---

## 4. Path Building

```sql
-- Build the full path string from root to each node
WITH RECURSIVE org_path AS (
  -- Anchor: root node (no manager)
  SELECT emp_id, name, manager_id,
    CAST(name AS CHAR(500)) AS path
  FROM employees
  WHERE manager_id IS NULL

  UNION ALL

  -- Recursive: append current name to path
  SELECT e.emp_id, e.name, e.manager_id,
    CONCAT(op.path, ' → ', e.name)
  FROM employees e
  JOIN org_path op ON e.manager_id = op.emp_id
)
SELECT emp_id, name, path FROM org_path ORDER BY path;
-- Output: 'CEO → VP Eng → Director → Alice'
```

---

## 5. Depth-Limited Traversal

```sql
-- Only go 3 levels deep (prevents infinite loops on bad data)
WITH RECURSIVE limited AS (
  SELECT emp_id, name, manager_id, 1 AS depth
  FROM employees
  WHERE manager_id IS NULL

  UNION ALL

  SELECT e.emp_id, e.name, e.manager_id, l.depth + 1
  FROM employees e
  JOIN limited l ON e.manager_id = l.emp_id
  WHERE l.depth < 3    -- ← depth limit stop condition
)
SELECT * FROM limited;
```

---

## 6. Bill of Materials (BOM)

Classic recursive problem — product made of components, each component made of sub-components.

```sql
-- components(component_id, name, parent_id, quantity)

WITH RECURSIVE bom AS (
  -- Anchor: top-level product
  SELECT component_id, name, parent_id, quantity, 0 AS level,
    CAST(quantity AS DECIMAL(10,2)) AS total_quantity
  FROM components
  WHERE parent_id IS NULL

  UNION ALL

  -- Recursive: multiply quantities down the tree
  SELECT c.component_id, c.name, c.parent_id, c.quantity,
    b.level + 1,
    b.total_quantity * c.quantity AS total_quantity
  FROM components c
  JOIN bom b ON c.parent_id = b.component_id
)
SELECT level, name, quantity, total_quantity
FROM bom
ORDER BY level, name;
```

---

## 7. Finding Cycles (Graph Safety)

```sql
-- Detect if there are cycles in a hierarchy (bad data)
WITH RECURSIVE cycle_check AS (
  SELECT emp_id, manager_id,
    CAST(emp_id AS CHAR(1000)) AS visited_path,
    0 AS is_cycle
  FROM employees WHERE manager_id IS NULL

  UNION ALL

  SELECT e.emp_id, e.manager_id,
    CONCAT(cc.visited_path, ',', e.emp_id),
    CASE WHEN FIND_IN_SET(e.emp_id, cc.visited_path) > 0
         THEN 1 ELSE 0 END
  FROM employees e
  JOIN cycle_check cc ON e.manager_id = cc.emp_id
  WHERE cc.is_cycle = 0   -- stop if cycle detected
)
SELECT * FROM cycle_check WHERE is_cycle = 1;
```

---

## 8. Recursive CTE for Cumulative Calculations

```sql
-- Running compound interest calculation
WITH RECURSIVE compound AS (
  SELECT 1 AS month, 10000.00 AS balance   -- anchor: initial deposit

  UNION ALL

  SELECT month + 1,
    ROUND(balance * 1.005, 2)              -- 0.5% monthly interest
  FROM compound
  WHERE month < 60                         -- 5 years = 60 months
)
SELECT month, balance,
  balance - 10000 AS total_interest
FROM compound;
```

```sql
-- Fibonacci sequence (common interview question)
WITH RECURSIVE fib AS (
  SELECT 1 AS n, 0 AS a, 1 AS b   -- anchor

  UNION ALL

  SELECT n + 1, b, a + b           -- recursive step
  FROM fib
  WHERE n < 20
)
SELECT n, a AS fibonacci_number FROM fib;
```

---

## Summary Cheatsheet

| Pattern | Use Case |
|---|---|
| Number/date generator | Fill gaps, date spines, test data |
| Top-down traversal | Org chart, all reports under manager |
| Bottom-up traversal | Full chain up to root |
| Path building | CONCAT path string at each level |
| Depth limit | WHERE depth < N — prevents infinite loops |
| BOM | Multiply quantities down component tree |
| Cycle detection | Track visited nodes in path string |

---

### 🟢 Q1 — Easy
> Generate a **date series for all of 2025** (Jan 1 to Dec 31). Then cross join with a `products` table to create a **product × date grid** showing 0 for days with no sales.

---

### 🟡 Q2 — Medium
> Table: `employees(emp_id, name, manager_id, salary)`
>
> For **manager emp_id = 5**, find all direct and indirect reports at any depth. Return `emp_id`, `name`, `depth` (level below manager 5), and their `salary`.

---

### 🔴 Q3 — Hard
> Table: `employees(emp_id, name, manager_id, salary)`
>
> For each employee, find the **total salary cost of their entire team** (all direct and indirect reports). Return `emp_id`, `name`, `team_size`, `total_team_salary`. Sort by total_team_salary descending.

### Q1 ✅
```sql
WITH RECURSIVE date_series AS (
  SELECT '2025-01-01' AS dt
  UNION ALL
  SELECT dt + INTERVAL 1 DAY
  FROM date_series
  WHERE dt < '2025-12-31'
)
SELECT p.product_id, p.product_name, d.dt AS sale_date,
  COALESCE(SUM(s.revenue), 0) AS daily_revenue
FROM date_series d
CROSS JOIN products p
LEFT JOIN daily_sales s
  ON  s.product_id = p.product_id
  AND s.sale_date  = d.dt
GROUP BY p.product_id, p.product_name, d.dt
ORDER BY p.product_id, d.dt;
```

---

### Q2 ✅
```sql
WITH RECURSIVE reports AS (
  SELECT emp_id, name, manager_id, salary, 0 AS depth
  FROM employees
  WHERE emp_id = 5

  UNION ALL

  SELECT e.emp_id, e.name, e.manager_id, e.salary, r.depth + 1
  FROM employees e
  JOIN reports r ON e.manager_id = r.emp_id
)
SELECT emp_id, name, depth, salary
FROM reports
WHERE emp_id != 5
ORDER BY depth, name;
```

---

### Q3 ✅
```sql
WITH RECURSIVE all_reports AS (
  SELECT
    emp_id          AS root_id,
    emp_id          AS member_id,
    salary          AS member_salary
  FROM employees

  UNION ALL

  SELECT
    ar.root_id,
    e.emp_id,
    e.salary
  FROM employees e
  JOIN all_reports ar ON e.manager_id = ar.member_id
),
team_stats AS (
  SELECT
    root_id,
    COUNT(*)      - 1 AS team_size,       -- exclude self
    SUM(member_salary) - MAX(CASE
      WHEN root_id = member_id
      THEN member_salary END)             AS total_team_salary
  FROM all_reports
  GROUP BY root_id
)
SELECT e.emp_id, e.name, t.team_size, t.total_team_salary
FROM team_stats t
JOIN employees e ON t.root_id = e.emp_id
WHERE t.team_size > 0
ORDER BY t.total_team_salary DESC;
```Day 11 saved ⬆️

---

