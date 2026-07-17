# Day 14 — Query Optimization & Performance
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. How SQL Engines Execute Queries
2. Indexes
3. Avoiding Full Table Scans
4. JOIN Optimization
5. Subquery vs JOIN Performance
6. Partitioning
7. CTEs vs Temp Tables
8. Aggregation Optimization
9. FAANG Scale Patterns
10. Anti-Patterns to Avoid

---

## 1. How SQL Engines Execute Queries

```
Query → Parser → Optimizer → Execution Plan → Result

Optimizer decides:
- Which indexes to use
- Join order
- Whether to scan or seek
- How to aggregate
```

```sql
-- Always check execution plan
EXPLAIN SELECT * FROM orders WHERE user_id = 123;
EXPLAIN ANALYZE SELECT * FROM orders WHERE user_id = 123; -- Postgres
-- Look for: full table scans, missing indexes, nested loops on large tables
```

---

## 2. Indexes

```sql
-- Single column index
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Composite index — column order matters!
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);
-- Good for: WHERE user_id = X AND order_date > Y
-- Bad for:  WHERE order_date > Y alone (user_id must lead)
```

**Index killer patterns:**
```sql
-- ❌ Function on indexed column
SELECT * FROM orders WHERE YEAR(order_date) = 2025;

-- ✅ Rewrite as range
SELECT * FROM orders
WHERE order_date >= '2025-01-01'
  AND order_date <  '2026-01-01';

-- ❌ Expression on column
WHERE user_id + 1 = 124;

-- ✅
WHERE user_id = 123;
```

---

## 3. Avoiding Full Table Scans

```sql
-- ❌ Leading wildcard kills index
SELECT * FROM users WHERE email LIKE '%gmail.com';

-- ✅ Trailing wildcard uses index
SELECT * FROM users WHERE email LIKE 'john%';

-- ❌ OR on different columns
SELECT * FROM users WHERE email = 'a@b.com' OR phone = '1234567890';

-- ✅ UNION — each branch uses own index
SELECT * FROM users WHERE email = 'a@b.com'
UNION
SELECT * FROM users WHERE phone = '1234567890';

-- ❌ SELECT * — excess I/O
SELECT * FROM orders WHERE user_id = 123;

-- ✅ Only needed columns
SELECT order_id, amount, order_date
FROM orders WHERE user_id = 123;
```

---

## 4. JOIN Optimization

```sql
-- Always join on indexed columns
-- ✅
SELECT o.*, u.name
FROM orders o JOIN users u ON o.user_id = u.user_id;

-- Pre-filter before joining
SELECT o.*, u.country
FROM (SELECT * FROM orders WHERE order_date >= '2025-01-01') o
JOIN (SELECT user_id, country FROM users WHERE country = 'India') u
  ON o.user_id = u.user_id;
```

---

## 5. Subquery vs JOIN Performance

```sql
-- ❌ Correlated subquery — O(n²)
SELECT name FROM employees e
WHERE salary > (
  SELECT AVG(salary) FROM employees
  WHERE department = e.department
);

-- ✅ CTE + JOIN — O(n)
WITH dept_avg AS (
  SELECT department, AVG(salary) AS avg_sal
  FROM employees GROUP BY department
)
SELECT e.name FROM employees e
JOIN dept_avg d ON e.department = d.department
WHERE e.salary > d.avg_sal;

-- ❌ IN with large subquery
SELECT * FROM orders
WHERE user_id IN (SELECT user_id FROM users WHERE country = 'India');

-- ✅ EXISTS or JOIN — faster
SELECT o.* FROM orders o
JOIN users u ON o.user_id = u.user_id
WHERE u.country = 'India';
```

---

## 6. Partitioning

```sql
-- Range partition by year
CREATE TABLE orders_partitioned (
  order_id INT, user_id INT,
  amount DECIMAL(10,2), order_date DATE
)
PARTITION BY RANGE (YEAR(order_date)) (
  PARTITION p2023 VALUES LESS THAN (2024),
  PARTITION p2024 VALUES LESS THAN (2025),
  PARTITION p2025 VALUES LESS THAN (2026),
  PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- Query hits only relevant partition
SELECT * FROM orders_partitioned
WHERE order_date >= '2025-01-01';  -- only scans p2025
```

```sql
-- BigQuery — always filter on partition column
SELECT user_id, SUM(amount) AS revenue
FROM `project.dataset.orders`
WHERE DATE(order_date) >= '2025-01-01'
  AND DATE(order_date) <  '2025-04-01'
GROUP BY user_id;
```

---

## 7. CTEs vs Temp Tables

```sql
-- CTE: inline, may re-evaluate if referenced multiple times
WITH stats AS (SELECT user_id, COUNT(*) AS n FROM orders GROUP BY user_id)
SELECT * FROM stats WHERE n > 10;

-- Temp table: materialized once — better for complex multi-step
CREATE TEMPORARY TABLE user_stats AS
SELECT user_id, COUNT(*) AS orders, SUM(amount) AS revenue
FROM orders GROUP BY user_id;

SELECT * FROM user_stats WHERE orders > 10;
DROP TEMPORARY TABLE user_stats;
```

> 💡 Use temp tables when the same CTE is referenced 2+ times in a complex query.

---

## 8. Aggregation Optimization

```sql
-- Filter before aggregating
SELECT user_id, COUNT(*) AS orders
FROM orders
WHERE order_date >= '2025-01-01'  -- filter first
GROUP BY user_id
HAVING COUNT(*) > 5;

-- Approximate COUNT DISTINCT (BigQuery — ~1% error, much faster)
SELECT APPROX_COUNT_DISTINCT(user_id) FROM events;
```

---

## 9. FAANG Scale Patterns

```sql
-- Incremental processing — only new data
INSERT INTO daily_user_stats
SELECT user_id, DATE(event_date), COUNT(*)
FROM events
WHERE DATE(event_date) = CURRENT_DATE - INTERVAL 1 DAY
GROUP BY user_id, DATE(event_date);

-- Aggregate before joining (reduce join size)
WITH user_agg AS (
  SELECT user_id, COUNT(*) AS events, AVG(duration) AS avg_dur
  FROM events
  WHERE event_date >= '2025-01-01'
  GROUP BY user_id
)
SELECT u.country, SUM(a.events), AVG(a.avg_dur)
FROM user_agg a
JOIN users u ON a.user_id = u.user_id
GROUP BY u.country;
```

---

## 10. Anti-Patterns to Avoid

```sql
-- ❌ SELECT * in production
-- ❌ DISTINCT as a JOIN bug fix
-- ❌ ORDER BY inside subquery (no effect)
-- ❌ NOT IN with NULLs → returns empty
-- ❌ Implicit type conversion → kills index
SELECT * FROM orders WHERE user_id = '123';  -- user_id is INT

-- ❌ HAVING for non-aggregate filter
SELECT user_id, COUNT(*) FROM orders
GROUP BY user_id HAVING user_id > 1000;  -- should be WHERE

-- ✅ Correct
SELECT user_id, COUNT(*) FROM orders
WHERE user_id > 1000 GROUP BY user_id;
```

---

## Practice Questions

### Q1 — Easy ✅
Rewrite to use index (avoid function on column).

```sql
-- ❌
SELECT * FROM orders WHERE YEAR(order_date) = 2025 AND MONTH(order_date) = 3;

-- ✅
SELECT order_id, user_id, amount, order_date
FROM orders
WHERE order_date >= '2025-03-01'
  AND order_date <  '2025-04-01';
```

### Q2 — Medium ✅
Rewrite correlated subquery as CTE + JOIN.

```sql
WITH dept_avg AS (
  SELECT department, AVG(salary) AS avg_sal
  FROM employees GROUP BY department
)
SELECT e.name, e.salary
FROM employees e
JOIN dept_avg d ON e.department = d.department
WHERE e.salary > d.avg_sal;
```

### Q3 — Hard ✅
Optimize 10B row events table query.

```sql
WITH filtered_events AS (
  SELECT user_id, event_date, session_duration
  FROM events
  WHERE event_date >= '2025-01-01'
    AND event_date <  '2026-01-01'
),
aggregated AS (
  SELECT user_id,
    COUNT(*)              AS event_count,
    AVG(session_duration) AS avg_session
  FROM filtered_events
  GROUP BY user_id
)
SELECT
  u.country,
  COUNT(DISTINCT a.user_id)    AS unique_users,
  SUM(a.event_count)           AS total_events,
  ROUND(AVG(a.avg_session), 2) AS avg_session
FROM aggregated a
JOIN users u ON a.user_id = u.user_id
GROUP BY u.country
ORDER BY unique_users DESC;
```

---

## Key Takeaways

- **EXPLAIN first** — always check execution plan before optimizing
- **Index killers** → functions, expressions, leading LIKE wildcards on indexed columns
- **Range > function** → `date >= X AND date < Y` beats `YEAR(date) = X`
- **Correlated subquery** → always rewrite as CTE + JOIN
- **IN → EXISTS or JOIN** → especially for large subqueries
- **Filter early** → reduce rows before JOINs and aggregations
- **Partition pruning** → always filter on partition column in BigQuery/Hive
- **HAVING vs WHERE** → use WHERE for non-aggregate filters
- **NOT IN + NULLs** → always use NOT EXISTS instead
- **Aggregate before JOIN** → reduces join size dramatically at scale

---

*Day 14 complete — Week 2 done! 16 days to go 🚀*
