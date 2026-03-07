# Day 11 — Recursive CTEs & Hierarchical Data
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. What is a Recursive CTE
2. Number & Date Series Generator
3. Org Chart / Top-Down Traversal
4. Bottom-Up Chain Traversal
5. Path Building
6. Depth-Limited Traversal
7. Bill of Materials (BOM)
8. Cycle Detection
9. Recursive Cumulative Calculations

---

## 1. What is a Recursive CTE?

```sql
WITH RECURSIVE cte_name AS (
  -- Anchor: starting point (runs ONCE)
  SELECT ...

  UNION ALL

  -- Recursive: references cte_name (runs repeatedly until no new rows)
  SELECT ... FROM cte_name WHERE <stop condition>
)
SELECT * FROM cte_name;
```

> ⚠️ Always include a stop condition — otherwise runs forever.

---

## 2. Number & Date Series Generator

```sql
-- Numbers 1 to 100
WITH RECURSIVE numbers AS (
  SELECT 1 AS n
  UNION ALL
  SELECT n + 1 FROM numbers WHERE n < 100
)
SELECT * FROM numbers;

-- Date spine: all days in 2025
WITH RECURSIVE date_series AS (
  SELECT '2025-01-01' AS dt
  UNION ALL
  SELECT dt + INTERVAL 1 DAY
  FROM date_series
  WHERE dt < '2025-12-31'
)
SELECT dt FROM date_series;
```

> 💡 Cross join date spine with products/users to fill zero-revenue gaps.

---

## 3. Org Chart — Top-Down Traversal

```sql
-- All reports under manager emp_id = 1, any depth
WITH RECURSIVE org_chart AS (
  SELECT emp_id, name, manager_id, 0 AS depth
  FROM employees WHERE emp_id = 1

  UNION ALL

  SELECT e.emp_id, e.name, e.manager_id, oc.depth + 1
  FROM employees e
  JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT emp_id, name, depth,
  REPEAT('  ', depth) AS indent
FROM org_chart
ORDER BY depth, name;
```

---

## 4. Bottom-Up Chain Traversal

```sql
-- Full reporting chain from employee up to root
WITH RECURSIVE chain AS (
  SELECT emp_id, name, manager_id, 0 AS level
  FROM employees WHERE emp_id = 42

  UNION ALL

  SELECT e.emp_id, e.name, e.manager_id, c.level + 1
  FROM employees e
  JOIN chain c ON e.emp_id = c.manager_id
)
SELECT level, name FROM chain ORDER BY level;
-- level 0 = employee, 1 = manager, 2 = skip-level...
```

---

## 5. Path Building

```sql
-- Build 'CEO → VP → Director → Alice' path string
WITH RECURSIVE org_path AS (
  SELECT emp_id, name, manager_id,
    CAST(name AS CHAR(500)) AS path
  FROM employees WHERE manager_id IS NULL

  UNION ALL

  SELECT e.emp_id, e.name, e.manager_id,
    CONCAT(op.path, ' → ', e.name)
  FROM employees e
  JOIN org_path op ON e.manager_id = op.emp_id
)
SELECT emp_id, name, path FROM org_path ORDER BY path;
```

---

## 6. Depth-Limited Traversal

```sql
WITH RECURSIVE limited AS (
  SELECT emp_id, name, manager_id, 1 AS depth
  FROM employees WHERE manager_id IS NULL

  UNION ALL

  SELECT e.emp_id, e.name, e.manager_id, l.depth + 1
  FROM employees e
  JOIN limited l ON e.manager_id = l.emp_id
  WHERE l.depth < 3    -- max 3 levels deep
)
SELECT * FROM limited;
```

---

## 7. Bill of Materials (BOM)

```sql
-- Multiply quantities down the component tree
WITH RECURSIVE bom AS (
  SELECT component_id, name, parent_id, quantity,
    0 AS level,
    CAST(quantity AS DECIMAL(10,2)) AS total_quantity
  FROM components WHERE parent_id IS NULL

  UNION ALL

  SELECT c.component_id, c.name, c.parent_id, c.quantity,
    b.level + 1,
    b.total_quantity * c.quantity
  FROM components c
  JOIN bom b ON c.parent_id = b.component_id
)
SELECT level, name, quantity, total_quantity
FROM bom ORDER BY level, name;
```

---

## 8. Cycle Detection

```sql
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
  WHERE cc.is_cycle = 0
)
SELECT * FROM cycle_check WHERE is_cycle = 1;
```

---

## 9. Recursive Cumulative Calculations

```sql
-- Compound interest over 60 months
WITH RECURSIVE compound AS (
  SELECT 1 AS month, 10000.00 AS balance
  UNION ALL
  SELECT month + 1, ROUND(balance * 1.005, 2)
  FROM compound WHERE month < 60
)
SELECT month, balance, balance - 10000 AS total_interest
FROM compound;

-- Fibonacci sequence
WITH RECURSIVE fib AS (
  SELECT 1 AS n, 0 AS a, 1 AS b
  UNION ALL
  SELECT n + 1, b, a + b FROM fib WHERE n < 20
)
SELECT n, a AS fibonacci_number FROM fib;
```

---

## Practice Questions

### Q1 — Easy ✅
Date spine × products grid with zero-fill.

```sql
WITH RECURSIVE date_series AS (
  SELECT '2025-01-01' AS dt
  UNION ALL
  SELECT dt + INTERVAL 1 DAY
  FROM date_series WHERE dt < '2025-12-31'
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

### Q2 — Medium ✅
All reports under manager 5 with depth and salary.

```sql
WITH RECURSIVE reports AS (
  SELECT emp_id, name, manager_id, salary, 0 AS depth
  FROM employees WHERE emp_id = 5

  UNION ALL

  SELECT e.emp_id, e.name, e.manager_id, e.salary, r.depth + 1
  FROM employees e
  JOIN reports r ON e.manager_id = r.emp_id
)
SELECT emp_id, name, depth, salary
FROM reports WHERE emp_id != 5
ORDER BY depth, name;
```

### Q3 — Hard ✅
Total team size and salary cost per manager.

```sql
WITH RECURSIVE all_reports AS (
  SELECT emp_id AS root_id, emp_id AS member_id, salary AS member_salary
  FROM employees

  UNION ALL

  SELECT ar.root_id, e.emp_id, e.salary
  FROM employees e
  JOIN all_reports ar ON e.manager_id = ar.member_id
),
team_stats AS (
  SELECT root_id,
    COUNT(*) - 1 AS team_size,
    SUM(member_salary) - MAX(CASE
      WHEN root_id = member_id THEN member_salary END) AS total_team_salary
  FROM all_reports
  GROUP BY root_id
)
SELECT e.emp_id, e.name, t.team_size, t.total_team_salary
FROM team_stats t
JOIN employees e ON t.root_id = e.emp_id
WHERE t.team_size > 0
ORDER BY t.total_team_salary DESC;
```

---

## Key Takeaways

- **Recursive CTE structure** → anchor UNION ALL recursive member + stop condition
- **Date spine** → generate with recursive CTE, CROSS JOIN to fill gaps
- **Top-down** → JOIN on `e.manager_id = oc.emp_id`
- **Bottom-up** → JOIN on `e.emp_id = c.manager_id`
- **Path string** → CONCAT at each level to build full hierarchy path
- **Depth limit** → WHERE depth < N in recursive member — always add for safety
- **BOM** → multiply quantities at each level `parent_qty * child_qty`
- **Cycle detection** → track visited IDs in a string, FIND_IN_SET to check

---

*Day 11 complete — 19 days to go 🚀*
