# CTEs and Window Functions — Logic & Mental Breakdown

---

## Part 1: CTEs (Common Table Expressions)

---

### What is a CTE?

A CTE is a **named, temporary result set** you define at the top of a query and use like a table below it.

Think of it as writing a subquery, but giving it a name and pulling it out of the mess.

**Mental model:**
> "Let me define what X is first. Then I'll query it."

---

### Anatomy of a CTE

```sql
WITH cte_name AS (
  -- this runs first, produces a virtual table
  SELECT ...
  FROM ...
)
-- this runs second, uses the virtual table
SELECT *
FROM cte_name;
```

The `WITH` block runs first, builds a named result set in memory, and then the outer query uses it.

---

### Why use a CTE instead of a subquery?

**Subquery version — hard to read:**
```sql
SELECT name, salary
FROM (
  SELECT name, salary, AVG(salary) OVER () AS avg_sal
  FROM employees
) sub
WHERE salary > avg_sal;
```

**CTE version — reads top to bottom like English:**
```sql
WITH avg_salaries AS (
  SELECT name, salary, AVG(salary) OVER () AS avg_sal
  FROM employees
)
SELECT name, salary
FROM avg_salaries
WHERE salary > avg_sal;
```

Same result. The CTE version tells a story: "First, compute average salaries. Then, filter."

---

### Chaining multiple CTEs

You can chain CTEs by separating them with commas. Each one can reference the one above it.

```sql
WITH
step1 AS (
  SELECT customer_id, SUM(amount) AS total_spent
  FROM orders
  GROUP BY customer_id
),
step2 AS (
  SELECT customer_id, total_spent,
         RANK() OVER (ORDER BY total_spent DESC) AS spend_rank
  FROM step1
)
SELECT *
FROM step2
WHERE spend_rank <= 10;
```

**Mental model — read it as a pipeline:**
1. `step1` → aggregate orders into one row per customer
2. `step2` → rank those customers by spend
3. Final query → keep only the top 10

Each CTE is one clean step. No nesting, no confusion.

---

### Recursive CTEs

A recursive CTE references itself. Used for hierarchical data (org charts, file trees, graph traversal).

**Structure:**
```sql
WITH RECURSIVE cte AS (
  -- anchor: the starting point
  SELECT id, name, manager_id, 0 AS level
  FROM employees
  WHERE manager_id IS NULL        -- the CEO

  UNION ALL

  -- recursive part: each iteration adds the next level
  SELECT e.id, e.name, e.manager_id, cte.level + 1
  FROM employees e
  JOIN cte ON e.manager_id = cte.id
)
SELECT * FROM cte ORDER BY level;
```

**Input table:**
| id | name | manager_id |
|----|------|------------|
| 1 | CEO | NULL |
| 2 | VP | 1 |
| 3 | Manager | 2 |
| 4 | Engineer | 3 |

**Result:**
| id | name | manager_id | level |
|----|------|------------|-------|
| 1 | CEO | NULL | 0 |
| 2 | VP | 1 | 1 |
| 3 | Manager | 2 | 2 |
| 4 | Engineer | 3 | 3 |

**Mental model:**
> Start at the root. Each iteration finds all children of the previous level. Stop when no more rows match.

---

### When to use a CTE

- Query needs more than 2 levels of logic — break it into named steps
- You reference the same subquery more than once
- You need to build a hierarchy or walk a graph (recursive CTE)
- You want another human to understand your query tomorrow

### When NOT to use a CTE

- Simple one-level filter — a WHERE clause is enough
- Performance is critical — some databases don't optimize CTEs as well as subqueries (test both)
- You only need the result once and it's a single simple SELECT

---

## Part 2: Window Functions

---

### What is a window function?

A window function computes a value **across a set of rows related to the current row** — without collapsing them into one row like `GROUP BY` does.

**Mental model:**
> "`GROUP BY` crushes rows together. A window function looks sideways at neighboring rows while keeping every row intact."

---

### The key difference: GROUP BY vs Window

**GROUP BY** — 4 rows collapse into 2:

`employees`
| name | dept | salary |
|------|------|--------|
| Alice | Eng | 120k |
| Bob | Eng | 95k |
| Carol | HR | 80k |
| Dave | HR | 72k |

```sql
SELECT dept, AVG(salary) FROM employees GROUP BY dept;
```

| dept | avg_salary |
|------|------------|
| Eng | 107.5k |
| HR | 76k |

**Window function** — 4 rows stay as 4, avg added as a new column:

```sql
SELECT name, dept, salary, AVG(salary) OVER (PARTITION BY dept) AS dept_avg
FROM employees;
```

| name | dept | salary | dept_avg |
|------|------|--------|----------|
| Alice | Eng | 120k | 107.5k |
| Bob | Eng | 95k | 107.5k |
| Carol | HR | 80k | 76k |
| Dave | HR | 72k | 76k |

Every row stays. The avg is computed per-group and stamped on each row.

---

### Anatomy of a window function

```sql
FUNCTION_NAME() OVER (
  PARTITION BY col     -- define the group (optional)
  ORDER BY col         -- define the order within the group (optional)
  ROWS/RANGE BETWEEN   -- define the frame (optional)
)
```

Each clause narrows the "window" the function looks through:

| Clause | What it does | Mental model |
|--------|--------------|--------------|
| `PARTITION BY` | Splits rows into groups | "Do this separately for each dept" |
| `ORDER BY` | Orders rows within each group | "In what order do I walk through rows?" |
| `ROWS BETWEEN` | Defines a sliding frame | "How many rows back/ahead do I look?" |

You can omit any of them:
- No `PARTITION BY` → the whole table is one group
- No `ORDER BY` → no ordering (only makes sense for pure aggregates)
- No frame → defaults vary by function

---

### The window frame

The frame defines exactly which rows the function can "see" from the current row's position.

```sql
ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
-- looks at: 2 rows before me + me = 3 rows total
```

```sql
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- looks at: all rows from the start up to me (running total)
```

```sql
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
-- looks at: me + 1 before + 1 after = 3 rows (rolling average)
```

**Input:**
| day | sales |
|-----|-------|
| Mon | 100 |
| Tue | 200 |
| Wed | 150 |
| Thu | 300 |
| Fri | 250 |

**3-day rolling average** (`ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`):
| day | sales | rolling_avg |
|-----|-------|-------------|
| Mon | 100 | 100 |
| Tue | 200 | 150 |
| Wed | 150 | 150 |
| Thu | 300 | 217 |
| Fri | 250 | 233 |

---

## Part 3: Window Function Types

---

### Type 1: Ranking functions

These assign a position number to each row within its partition.

| Function | Behaviour | Tie handling |
|----------|-----------|--------------|
| `ROW_NUMBER()` | Always unique 1,2,3,4... | Ties get arbitrary order |
| `RANK()` | Gaps after ties | 1,2,2,4 |
| `DENSE_RANK()` | No gaps after ties | 1,2,2,3 |
| `NTILE(n)` | Splits into n equal buckets | — |

**Input:**
| name | score |
|------|-------|
| Alice | 90 |
| Bob | 85 |
| Carol | 85 |
| Dave | 70 |

```sql
SELECT name, score,
  ROW_NUMBER() OVER (ORDER BY score DESC) AS row_num,
  RANK()       OVER (ORDER BY score DESC) AS rnk,
  DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rnk
FROM scores;
```

| name | score | row_num | rnk | dense_rnk |
|------|-------|---------|-----|-----------|
| Alice | 90 | 1 | 1 | 1 |
| Bob | 85 | 2 | 2 | 2 |
| Carol | 85 | 3 | 2 | 2 |
| Dave | 70 | 4 | 4 | 3 |

**When to use which:**
- `ROW_NUMBER()` — de-duplication, picking exactly 1 row per group
- `RANK()` — leaderboards where ties share a rank and the next rank skips
- `DENSE_RANK()` — leaderboards where ties share a rank but nothing is skipped
- `NTILE(4)` — split users into quartiles (top 25%, next 25%, etc.)

---

### Type 2: Aggregate window functions

Standard aggregates (`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`) used with `OVER()` — without collapsing rows.

**Running total:**
```sql
SELECT order_date, amount,
  SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM orders;
```

| order_date | amount | running_total |
|------------|--------|---------------|
| Jan 1 | 100 | 100 |
| Jan 2 | 200 | 300 |
| Jan 3 | 150 | 450 |
| Jan 4 | 300 | 750 |

**Percentage of group total:**
```sql
SELECT name, dept, salary,
  ROUND(salary / SUM(salary) OVER (PARTITION BY dept) * 100, 1) AS pct_of_dept
FROM employees;
```

| name | dept | salary | pct_of_dept |
|------|------|--------|-------------|
| Alice | Eng | 120k | 55.8% |
| Bob | Eng | 95k | 44.2% |
| Carol | HR | 80k | 52.6% |
| Dave | HR | 72k | 47.4% |

---

### Type 3: Offset functions — LAG and LEAD

These look at a row that comes before (`LAG`) or after (`LEAD`) the current row.

```sql
LAG(col, n, default)   -- value from n rows BEFORE current
LEAD(col, n, default)  -- value from n rows AFTER current
```

**Use case — month-over-month change:**
```sql
SELECT month, revenue,
  LAG(revenue, 1) OVER (ORDER BY month) AS prev_month,
  revenue - LAG(revenue, 1) OVER (ORDER BY month) AS change
FROM monthly_sales;
```

| month | revenue | prev_month | change |
|-------|---------|------------|--------|
| Jan | 1000 | NULL | NULL |
| Feb | 1200 | 1000 | +200 |
| Mar | 1100 | 1200 | -100 |
| Apr | 1500 | 1100 | +400 |

**Mental model:**
> `LAG` reaches back in time. `LEAD` peeks ahead. Both are just "give me the value from a different row in the same window."

---

### Type 4: FIRST_VALUE and LAST_VALUE

Returns the first or last value in the window frame.

```sql
SELECT name, dept, salary,
  FIRST_VALUE(name) OVER (PARTITION BY dept ORDER BY salary DESC) AS top_earner
FROM employees;
```

| name | dept | salary | top_earner |
|------|------|--------|------------|
| Alice | Eng | 120k | Alice |
| Bob | Eng | 95k | Alice |
| Carol | HR | 80k | Carol |
| Dave | HR | 72k | Carol |

> `LAST_VALUE` needs `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` or it only sees up to the current row by default.

---

## Part 4: Combining CTEs + Window Functions

This is where real FAANG-level SQL is written.

---

### Pattern: Top N per group

```sql
WITH ranked AS (
  SELECT
    name, dept, salary,
    ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn
  FROM employees
)
SELECT name, dept, salary
FROM ranked
WHERE rn <= 2;
```

**Why this works:**
- The CTE assigns a rank within each department
- The outer query simply filters `rn <= 2`
- You can't use window functions in a `WHERE` clause directly — the CTE wraps it so you can filter on it

**Input:**
| name | dept | salary |
|------|------|--------|
| Alice | Eng | 120k |
| Bob | Eng | 95k |
| Eve | Eng | 88k |
| Carol | HR | 80k |
| Dave | HR | 72k |

**Result:**
| name | dept | salary |
|------|------|--------|
| Alice | Eng | 120k |
| Bob | Eng | 95k |
| Carol | HR | 80k |
| Dave | HR | 72k |

---

### Pattern: Running total then filter

```sql
WITH running AS (
  SELECT
    order_date, amount,
    SUM(amount) OVER (ORDER BY order_date) AS running_total
  FROM orders
)
SELECT *
FROM running
WHERE running_total <= 1000;
```

---

### Pattern: Compare each row to its group average

```sql
WITH dept_stats AS (
  SELECT
    name, dept, salary,
    AVG(salary) OVER (PARTITION BY dept) AS dept_avg,
    salary - AVG(salary) OVER (PARTITION BY dept) AS diff_from_avg
  FROM employees
)
SELECT *
FROM dept_stats
WHERE diff_from_avg > 0;   -- only above-average earners
```

---

## Part 5: Mental Checklists

---

### When writing a window function, ask:

1. **What am I computing?** → Pick the function (`SUM`, `RANK`, `LAG`, etc.)
2. **Per what group?** → `PARTITION BY dept` / `PARTITION BY user_id` / nothing (whole table)
3. **In what order?** → `ORDER BY date DESC` / `ORDER BY salary`
4. **How far back/ahead?** → Frame clause, or leave default

---

### CTE vs Subquery decision:

| Situation | Use |
|-----------|-----|
| Logic has 1 level | Subquery or WHERE |
| Logic has 2+ levels | CTE |
| Same subquery used twice | CTE |
| Hierarchical data | Recursive CTE |
| Readability matters | Always CTE |

---

### Window function vs GROUP BY decision:

| You want | Use |
|----------|-----|
| One summary row per group | `GROUP BY` |
| Summary value on every row | Window function |
| Compare a row to its group | Window function |
| Rank rows within a group | Window function |
| Running total / rolling avg | Window function |

---

### Common mistakes

- **Using a window function in `WHERE`** — not allowed. Wrap in a CTE or subquery first, then filter.
- **`LAST_VALUE` returning wrong result** — default frame stops at current row. Use `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`.
- **`RANK()` when you meant `ROW_NUMBER()`** — if you need exactly one row per group, use `ROW_NUMBER()`, not `RANK()` (ties give multiple rows with rank = 1).
- **Forgetting `ORDER BY` inside `OVER()`** — `LAG`, `LEAD`, `FIRST_VALUE`, running totals all require it. Without it, results are undefined.
- **Assuming CTEs are always materialized** — in PostgreSQL they are (each CTE runs once). In other databases (MySQL 8+, SQL Server) they may be inlined. This affects performance.

---
