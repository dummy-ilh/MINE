# Grokking the SQL Interview — Supplement
> Covers all topics missing from the original guide (Q71–Q120)
> Same reference tables (employees, departments, orders, products) apply throughout.

---

# Section 15 — Transactions and Concurrency

---

## Q71 — ACID Properties

**Question:**  
What are ACID properties? Explain each with a practical example.

**Answer:**

| Property    | Meaning | Example |
|-------------|---------|---------|
| **Atomicity**   | All steps in a transaction succeed or all are rolled back — no partial state | Transferring money: debit AND credit must both succeed, or neither happens |
| **Consistency** | A transaction moves the database from one valid state to another; all rules/constraints are respected | A transfer cannot leave an account balance negative if a CHECK constraint forbids it |
| **Isolation**   | Concurrent transactions do not interfere with each other | Two users buying the last item simultaneously — only one succeeds |
| **Durability**  | Once committed, data survives crashes, power loss, etc. | After `COMMIT`, the record persists even if the server crashes immediately after |

```sql
-- Classic money transfer demonstrating Atomicity
BEGIN;

UPDATE accounts SET balance = balance - 500 WHERE account_id = 1;  -- debit
UPDATE accounts SET balance = balance + 500 WHERE account_id = 2;  -- credit

-- Only write if both succeed
COMMIT;

-- If anything goes wrong, undo everything
ROLLBACK;
```

---

## Q72 — BEGIN, COMMIT, ROLLBACK, SAVEPOINT

**Question:**  
Show how transactions work in practice, including partial rollbacks using `SAVEPOINT`.

**Answer:**
```sql
BEGIN;

INSERT INTO employees (emp_id, name, dept_id, salary, hire_date, job_title)
VALUES (10, 'Jake', 10, 80000, '2024-01-15', 'Engineer');

SAVEPOINT after_insert;  -- mark this point

UPDATE employees SET salary = 999999 WHERE emp_id = 10;  -- accidental bad update

ROLLBACK TO SAVEPOINT after_insert;  -- undo only the UPDATE, keep the INSERT

COMMIT;  -- commit the INSERT with the original salary
```

> **Interview tip:** `ROLLBACK TO SAVEPOINT` undoes changes back to the savepoint but keeps the transaction open. A full `ROLLBACK` undoes the entire transaction.

---

## Q73 — Transaction Isolation Levels

**Question:**  
What are the four transaction isolation levels and what concurrency problems does each prevent?

**Answer:**

| Isolation Level    | Dirty Read | Non-Repeatable Read | Phantom Read |
|--------------------|------------|---------------------|--------------|
| READ UNCOMMITTED   | ✅ possible | ✅ possible          | ✅ possible  |
| READ COMMITTED     | ❌ prevented| ✅ possible          | ✅ possible  |
| REPEATABLE READ    | ❌ prevented| ❌ prevented         | ✅ possible  |
| SERIALIZABLE       | ❌ prevented| ❌ prevented         | ❌ prevented |

```sql
-- Set isolation level (MySQL)
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT salary FROM employees WHERE emp_id = 1;
-- ... other work ...
SELECT salary FROM employees WHERE emp_id = 1;  -- guaranteed same result
COMMIT;
```

**Concurrency problems defined:**
- **Dirty Read:** Reading uncommitted data from another transaction that may yet be rolled back
- **Non-Repeatable Read:** Reading the same row twice and getting different values because another transaction updated it between reads
- **Phantom Read:** A re-executed range query returns different rows because another transaction inserted/deleted rows

---

# Section 16 — Views

---

## Q74 — Creating and Using Views

**Question:**  
Create a view called `high_earners` that returns all employees with salary > $100,000, then query it.

**Answer:**
```sql
-- Create the view
CREATE VIEW high_earners AS
SELECT emp_id, name, dept_id, salary, job_title
FROM employees
WHERE salary > 100000;

-- Query the view just like a table
SELECT name, salary
FROM high_earners
ORDER BY salary DESC;
```

**Sample Output:**

| name  | salary |
|-------|--------|
| Frank | 130000 |
| Alice | 120000 |
| Dave  | 110000 |

> **Interview tip:** Views do not store data — they are saved `SELECT` statements that execute fresh on every query. They are useful for security (expose only certain columns), simplicity (hide complex joins), and consistency (one definition used everywhere).

---

## Q75 — Updatable Views

**Question:**  
When can you INSERT or UPDATE through a view? What makes a view non-updatable?

**Answer:**
```sql
-- This view IS updatable (simple, single-table, no aggregation)
CREATE VIEW dept10_employees AS
SELECT emp_id, name, salary
FROM employees
WHERE dept_id = 10;

-- This UPDATE works — it modifies the base table
UPDATE dept10_employees
SET salary = 100000
WHERE emp_id = 2;

-- Views are NOT updatable when they contain:
-- GROUP BY / HAVING / aggregates / DISTINCT / UNION / subqueries in SELECT
CREATE VIEW dept_summary AS
SELECT dept_id, COUNT(*) AS cnt, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id;
-- UPDATE dept_summary SET avg_sal = 90000; -- ERROR: not updatable
```

---

## Q76 — Materialized Views

**Question:**  
What is a materialized view and how does it differ from a regular view?

**Answer:**

| Feature           | Regular View       | Materialized View          |
|-------------------|--------------------|----------------------------|
| Data stored?      | No — re-runs query | Yes — snapshot on disk      |
| Speed             | Depends on query   | Fast (pre-computed)         |
| Data freshness    | Always current     | Stale until refreshed       |
| Use case          | Real-time accuracy | Heavy aggregation, reporting|

```sql
-- PostgreSQL materialized view
CREATE MATERIALIZED VIEW dept_salary_summary AS
SELECT
  d.dept_name,
  COUNT(e.emp_id)  AS headcount,
  AVG(e.salary)    AS avg_salary,
  SUM(e.salary)    AS total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name;

-- Refresh when underlying data changes
REFRESH MATERIALIZED VIEW dept_salary_summary;

-- Query it (fast — reads from stored data)
SELECT * FROM dept_salary_summary ORDER BY total_salary DESC;
```

**Sample Output:**

| dept_name   | headcount | avg_salary | total_salary |
|-------------|-----------|------------|--------------|
| Engineering | 3         | 100000     | 300000       |
| Sales       | 3         | 98000      | 294000       |
| Marketing   | 2         | 91000      | 182000       |

---

# Section 17 — DML: INSERT, UPDATE, DELETE

---

## Q77 — INSERT INTO … SELECT (Bulk Insert)

**Question:**  
Copy all Sales department employees into an `archive_employees` table.

**Answer:**
```sql
-- First create the archive table (if not exists)
CREATE TABLE archive_employees AS
SELECT * FROM employees WHERE 1 = 0;  -- copies structure, no rows

-- Then bulk-insert matching rows
INSERT INTO archive_employees (emp_id, name, dept_id, salary, hire_date, job_title)
SELECT emp_id, name, dept_id, salary, hire_date, job_title
FROM employees
WHERE dept_id = 30;

SELECT * FROM archive_employees;
```

**Sample Output:**

| emp_id | name  | dept_id | salary | hire_date  | job_title  |
|--------|-------|---------|--------|------------|------------|
| 6      | Frank | 30      | 130000 | 2016-05-11 | VP Sales   |
| 7      | Grace | 30      | 88000  | 2020-09-01 | Sales Lead |
| 8      | Heidi | 30      | 76000  | 2022-01-15 | Sales Rep  |

---

## Q78 — UPDATE with JOIN

**Question:**  
Give every employee in the San Francisco office a 10% raise (location is stored in the `departments` table, not `employees`).

**Answer:**
```sql
-- MySQL
UPDATE employees e
JOIN departments d ON e.dept_id = d.dept_id
SET e.salary = e.salary * 1.10
WHERE d.location = 'San Francisco';

-- PostgreSQL / SQL Server (UPDATE ... FROM)
UPDATE employees
SET salary = salary * 1.10
FROM departments
WHERE employees.dept_id = departments.dept_id
  AND departments.location = 'San Francisco';
```

**Sample Output (Engineering dept before and after):**

| name  | old_salary | new_salary |
|-------|------------|------------|
| Alice | 120000     | 132000     |
| Bob   | 95000      | 104500     |
| Carol | 85000      | 93500      |

---

## Q79 — UPDATE with CASE

**Question:**  
Give different raises to different salary bands in one statement: Entry (<75k) gets 15%, Mid (75–100k) gets 10%, Senior (>100k) gets 5%.

**Answer:**
```sql
UPDATE employees
SET salary = salary * CASE
  WHEN salary < 75000  THEN 1.15
  WHEN salary <= 100000 THEN 1.10
  ELSE 1.05
END;
```

**Sample Output:**

| name  | old_salary | new_salary |
|-------|------------|------------|
| Ivan  | 60000      | 69000      |
| Eve   | 72000      | 82800      |
| Heidi | 76000      | 83600      |
| Carol | 85000      | 93500      |
| Bob   | 95000      | 104500     |
| Alice | 120000     | 126000     |
| Frank | 130000     | 136500     |

---

## Q80 — UPSERT (INSERT … ON CONFLICT / MERGE)

**Question:**  
Insert a new department, but if it already exists (same `dept_id`), update the location instead.

**Answer:**
```sql
-- PostgreSQL (ON CONFLICT)
INSERT INTO departments (dept_id, dept_name, location)
VALUES (10, 'Engineering', 'Remote')
ON CONFLICT (dept_id)
DO UPDATE SET location = EXCLUDED.location;

-- MySQL (ON DUPLICATE KEY UPDATE)
INSERT INTO departments (dept_id, dept_name, location)
VALUES (10, 'Engineering', 'Remote')
ON DUPLICATE KEY UPDATE location = VALUES(location);

-- SQL Server / Oracle (MERGE)
MERGE INTO departments AS target
USING (SELECT 10 AS dept_id, 'Engineering' AS dept_name, 'Remote' AS location) AS src
  ON target.dept_id = src.dept_id
WHEN MATCHED THEN
  UPDATE SET location = src.location
WHEN NOT MATCHED THEN
  INSERT (dept_id, dept_name, location)
  VALUES (src.dept_id, src.dept_name, src.location);
```

---

# Section 18 — Advanced Window Functions

---

## Q81 — FIRST_VALUE and LAST_VALUE

**Question:**  
For each employee, show the name and salary of the highest-paid and lowest-paid person in their department.

**Answer:**
```sql
SELECT
  name,
  dept_id,
  salary,
  FIRST_VALUE(name) OVER (
    PARTITION BY dept_id
    ORDER BY salary DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS dept_highest_earner,
  LAST_VALUE(name) OVER (
    PARTITION BY dept_id
    ORDER BY salary DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS dept_lowest_earner
FROM employees
WHERE dept_id IS NOT NULL
ORDER BY dept_id, salary DESC;
```

**Sample Output:**

| name  | dept_id | salary | dept_highest_earner | dept_lowest_earner |
|-------|---------|--------|---------------------|-------------------|
| Alice | 10      | 120000 | Alice               | Carol             |
| Bob   | 10      | 95000  | Alice               | Carol             |
| Carol | 10      | 85000  | Alice               | Carol             |
| Dave  | 20      | 110000 | Dave                | Eve               |
| Eve   | 20      | 72000  | Dave                | Eve               |

> **Interview tip:** `LAST_VALUE` requires `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` — without it, the default frame ends at the current row, making `LAST_VALUE` equal to the current row's value.

---

## Q82 — ROWS vs RANGE Window Frames

**Question:**  
Explain the difference between `ROWS` and `RANGE` in a window frame clause. When do they produce different results?

**Answer:**
```sql
-- Setup: two employees hired on the same date (ties)
-- hire_date: 2020-01-10 → Carol (85000)
-- hire_date: 2020-01-10 → (imagine a second person with same date, 90000)

-- ROWS: counts physical rows — precise, never affected by ties
SELECT name, hire_date, salary,
       SUM(salary) OVER (
         ORDER BY hire_date
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_total_rows
FROM employees ORDER BY hire_date;

-- RANGE: counts all rows with the same ORDER BY value as "current"
-- If two employees share a hire_date, RANGE includes BOTH in the frame
SELECT name, hire_date, salary,
       SUM(salary) OVER (
         ORDER BY hire_date
         RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_total_range
FROM employees ORDER BY hire_date;
```

**When they differ — example with ties:**

| name  | hire_date  | salary | ROWS total | RANGE total |
|-------|------------|--------|------------|-------------|
| Carol | 2020-01-10 | 85000  | 360000     | 448000      |
| NewEmp| 2020-01-10 | 88000  | 448000     | 448000      |

> `ROWS` counts exactly up to this physical row. `RANGE` includes all rows with the same sort key, so both tied rows get the combined total. **Use `ROWS` for running totals** — it's precise and predictable.

---

## Q83 — STRING_AGG / GROUP_CONCAT / LISTAGG

**Question:**  
For each department, produce a comma-separated list of employee names.

**Answer:**
```sql
-- MySQL
SELECT
  dept_id,
  GROUP_CONCAT(name ORDER BY name SEPARATOR ', ') AS employee_list
FROM employees
WHERE dept_id IS NOT NULL
GROUP BY dept_id;

-- PostgreSQL / SQL Server 2017+
SELECT
  dept_id,
  STRING_AGG(name, ', ' ORDER BY name) AS employee_list
FROM employees
WHERE dept_id IS NOT NULL
GROUP BY dept_id;

-- Oracle
SELECT
  dept_id,
  LISTAGG(name, ', ') WITHIN GROUP (ORDER BY name) AS employee_list
FROM employees
WHERE dept_id IS NOT NULL
GROUP BY dept_id;
```

**Sample Output:**

| dept_id | employee_list       |
|---------|---------------------|
| 10      | Alice, Bob, Carol   |
| 20      | Dave, Eve           |
| 30      | Frank, Grace, Heidi |

---

# Section 19 — Advanced Grouping

---

## Q84 — GROUPING SETS

**Question:**  
Report total salary grouped by department only, by job title only, and the grand total — all in one query.

**Answer:**
```sql
SELECT
  dept_id,
  job_title,
  SUM(salary) AS total_salary
FROM employees
GROUP BY GROUPING SETS (
  (dept_id),      -- subtotal per department
  (job_title),    -- subtotal per job title
  ()              -- grand total
)
ORDER BY dept_id, job_title;
```

**Sample Output:**

| dept_id | job_title         | total_salary |
|---------|-------------------|--------------|
| 10      | NULL              | 300000       |
| 20      | NULL              | 182000       |
| 30      | NULL              | 294000       |
| NULL    | Contractor        | 60000        |
| NULL    | Engineer          | 85000        |
| NULL    | VP Engineering    | 120000       |
| NULL    | VP Marketing      | 110000       |
| NULL    | VP Sales          | 130000       |
| NULL    | NULL              | 836000       |

---

## Q85 — ROLLUP

**Question:**  
Show total salary grouped by department and job title, with subtotals per department and a grand total.

**Answer:**
```sql
SELECT
  d.dept_name,
  e.job_title,
  SUM(e.salary) AS total_salary
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
GROUP BY ROLLUP(d.dept_name, e.job_title)
ORDER BY d.dept_name, e.job_title;
```

**Sample Output:**

| dept_name   | job_title       | total_salary |
|-------------|-----------------|--------------|
| Engineering | Engineer        | 85000        |
| Engineering | Senior Engineer | 95000        |
| Engineering | VP Engineering  | 120000       |
| Engineering | NULL            | 300000       |  ← dept subtotal
| Marketing   | Marketing Analyst | 72000      |
| Marketing   | VP Marketing    | 110000       |
| Marketing   | NULL            | 182000       |  ← dept subtotal
| NULL        | NULL            | 836000       |  ← grand total

---

## Q86 — CUBE

**Question:**  
Generate every possible combination of subtotals for department and job title.

**Answer:**
```sql
SELECT
  dept_id,
  job_title,
  SUM(salary)  AS total_salary,
  COUNT(*)     AS headcount
FROM employees
GROUP BY CUBE(dept_id, job_title)
ORDER BY dept_id, job_title;
```

> `CUBE(A, B)` is equivalent to `GROUPING SETS((A,B), (A), (B), ())` — it produces all 4 combinations. For 3 dimensions, it produces 8 combinations (2³).

---

# Section 20 — Classic Interview Problems (Additional)

---

## Q87 — Swap Column Values

**Question:**  
In a table with columns `col_a` and `col_b`, swap their values for all rows — without a temporary column.

**Answer:**
```sql
-- Works in most databases
UPDATE swap_table
SET col_a = col_b,
    col_b = col_a;

-- MySQL requires a workaround (evaluates SET clauses left-to-right)
UPDATE swap_table
SET col_a = col_a + col_b,
    col_b = col_a - col_b,
    col_a = col_a - col_b;

-- Or use a CASE expression (universally safe)
UPDATE swap_table
SET col_a = CASE WHEN col_a = col_a THEN col_b ELSE col_a END,
    col_b = CASE WHEN col_b = col_b THEN col_a ELSE col_b END;
```

**Sample Input:**

| id | col_a | col_b |
|----|-------|-------|
| 1  | 10    | 20    |
| 2  | 30    | 40    |

**Sample Output:**

| id | col_a | col_b |
|----|-------|-------|
| 1  | 20    | 10    |
| 2  | 40    | 30    |

---

## Q88 — Find Missing Numbers in a Sequence

**Question:**  
You have a table of IDs from 1 to 10 but some are missing. Find the gaps.

**Answer:**
```sql
-- Generate 1–10, left join to find which are absent
WITH RECURSIVE seq AS (
  SELECT 1 AS n
  UNION ALL
  SELECT n + 1 FROM seq WHERE n < 10
)
SELECT seq.n AS missing_id
FROM seq
LEFT JOIN your_table t ON seq.n = t.id
WHERE t.id IS NULL
ORDER BY seq.n;

-- SQL Server / PostgreSQL alternative using a numbers table
SELECT n AS missing_id
FROM (VALUES (1),(2),(3),(4),(5),(6),(7),(8),(9),(10)) AS nums(n)
WHERE n NOT IN (SELECT id FROM your_table);
```

**Sample Input (your_table):**

| id |
|----|
| 1  |
| 2  |
| 4  |
| 7  |
| 10 |

**Sample Output:**

| missing_id |
|------------|
| 3          |
| 5          |
| 6          |
| 8          |
| 9          |

---

## Q89 — Year-over-Year Comparison

**Question:**  
Calculate total revenue per year and the year-over-year growth amount and percentage.

**Answer:**
```sql
WITH yearly AS (
  SELECT
    YEAR(order_date)   AS yr,
    SUM(amount)        AS revenue
  FROM orders
  GROUP BY YEAR(order_date)
)
SELECT
  yr,
  revenue,
  LAG(revenue) OVER (ORDER BY yr)                               AS prev_year_revenue,
  revenue - LAG(revenue) OVER (ORDER BY yr)                     AS yoy_change,
  ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY yr))
        / NULLIF(LAG(revenue) OVER (ORDER BY yr), 0), 2)        AS yoy_pct
FROM yearly
ORDER BY yr;
```

**Sample Output:**

| yr   | revenue  | prev_year_revenue | yoy_change | yoy_pct |
|------|----------|-------------------|------------|---------|
| 2022 | 450000   | NULL              | NULL       | NULL    |
| 2023 | 520000   | 450000            | 70000      | 15.56   |
| 2024 | 610000   | 520000            | 90000      | 17.31   |

---

## Q90 — Most Recent Order per Customer

**Question:**  
Return each customer's most recent order (order ID, date, and amount). Demonstrate two methods.

**Answer:**
```sql
-- Method 1: Window function (cleanest)
SELECT customer_id, order_id, order_date, amount
FROM (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
  FROM orders
) t
WHERE rn = 1;

-- Method 2: Correlated subquery
SELECT order_id, customer_id, order_date, amount
FROM orders o
WHERE order_date = (
  SELECT MAX(order_date)
  FROM orders
  WHERE customer_id = o.customer_id
);

-- Method 3: Self-join anti-join pattern
SELECT o.*
FROM orders o
LEFT JOIN orders o2
  ON o.customer_id = o2.customer_id AND o.order_date < o2.order_date
WHERE o2.order_id IS NULL;
```

**Sample Output:**

| customer_id | order_id | order_date | amount |
|-------------|----------|------------|--------|
| 101         | 5042     | 2024-03-15 | 250.00 |
| 102         | 5089     | 2024-04-01 | 180.00 |
| 103         | 5103     | 2024-04-10 | 320.00 |

---

## Q91 — Employees with the Same Salary

**Question:**  
Find all employees who share the same salary as at least one other employee.

**Answer:**
```sql
-- Method 1: Self-join
SELECT DISTINCT e1.name, e1.salary
FROM employees e1
JOIN employees e2
  ON e1.salary = e2.salary
 AND e1.emp_id <> e2.emp_id
ORDER BY e1.salary;

-- Method 2: Window function
SELECT name, salary
FROM (
  SELECT name, salary,
         COUNT(*) OVER (PARTITION BY salary) AS cnt
  FROM employees
) t
WHERE cnt > 1
ORDER BY salary;
```

**Sample Input (add two employees with matching salary):**

| name  | salary |
|-------|--------|
| Alice | 120000 |
| Bob   | 95000  |
| Carol | 85000  |
| NewA  | 85000  |  ← same as Carol

**Sample Output:**

| name  | salary |
|-------|--------|
| Carol | 85000  |
| NewA  | 85000  |

---

## Q92 — Multiple-Column Subquery

**Question:**  
Find employees who have the same (dept_id, salary) combination as any manager. Managers are employees whose `emp_id` appears as someone else's `manager_id`.

**Answer:**
```sql
-- Multiple-column IN subquery
SELECT name, dept_id, salary
FROM employees
WHERE (dept_id, salary) IN (
  SELECT dept_id, salary
  FROM employees
  WHERE emp_id IN (SELECT DISTINCT manager_id FROM employees WHERE manager_id IS NOT NULL)
);

-- Equivalent using EXISTS
SELECT e.name, e.dept_id, e.salary
FROM employees e
WHERE EXISTS (
  SELECT 1
  FROM employees mgr
  WHERE mgr.emp_id IN (SELECT DISTINCT manager_id FROM employees WHERE manager_id IS NOT NULL)
    AND mgr.dept_id = e.dept_id
    AND mgr.salary  = e.salary
);
```

---

## Q93 — NOT IN with NULL Gotcha

**Question:**  
Why does `NOT IN` return no rows when the subquery contains a NULL? Demonstrate the problem and the fix.

**Answer:**
```sql
-- Setup
CREATE TABLE a (val INT);
CREATE TABLE b (val INT);
INSERT INTO a VALUES (1),(2),(3),(4);
INSERT INTO b VALUES (1),(2),(NULL);  -- NOTE: NULL in b

-- PROBLEM: returns NO rows even though 3 and 4 are not in b
SELECT val FROM a WHERE val NOT IN (SELECT val FROM b);
-- Result: (empty) ← WRONG

-- Why: NOT IN (1, 2, NULL) → NOT (val=1 OR val=2 OR val=NULL)
-- val=3: NOT (FALSE OR FALSE OR NULL) = NOT NULL = NULL (not TRUE!)
-- NULL condition is never TRUE, so no rows pass

-- FIX 1: Use NOT EXISTS (NULL-safe)
SELECT a.val
FROM a
WHERE NOT EXISTS (
  SELECT 1 FROM b WHERE b.val = a.val
);

-- FIX 2: Filter NULLs from subquery
SELECT val FROM a
WHERE val NOT IN (SELECT val FROM b WHERE val IS NOT NULL);
```

**Output comparison:**

| Query             | Result        |
|-------------------|---------------|
| NOT IN (with NULL)| (empty — wrong)|
| NOT EXISTS        | 3, 4          |
| NOT IN + IS NOT NULL | 3, 4       |

---

## Q94 — Temporary Tables vs CTEs

**Question:**  
What is the difference between a CTE and a temporary table? When should you use each?

**Answer:**
```sql
-- CTE: exists only for the duration of one query
WITH dept_avg AS (
  SELECT dept_id, AVG(salary) AS avg_sal
  FROM employees GROUP BY dept_id
)
SELECT e.name, e.salary, da.avg_sal
FROM employees e
JOIN dept_avg da ON e.dept_id = da.dept_id
WHERE e.salary > da.avg_sal;
-- dept_avg is gone after this query

-- Temporary Table: persists for the session, can be indexed and reused
CREATE TEMPORARY TABLE tmp_dept_avg AS
SELECT dept_id, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id;

-- Can be queried multiple times in the same session
SELECT * FROM tmp_dept_avg;  -- first use
-- ... other queries ...
SELECT * FROM tmp_dept_avg;  -- reused without recomputing

-- Add an index for performance
CREATE INDEX idx_tmp_dept ON tmp_dept_avg(dept_id);

DROP TEMPORARY TABLE tmp_dept_avg;  -- clean up
```

| Feature          | CTE                        | Temporary Table             |
|------------------|----------------------------|-----------------------------|
| Scope            | Single query               | Entire session              |
| Can be reused?   | No (recalculated each use) | Yes                         |
| Can be indexed?  | No                         | Yes                         |
| Materialised?    | Sometimes (optimizer decides) | Always                   |
| Best for         | Readability, one-off logic | Heavy intermediate results  |

---

# Section 21 — COALESCE / IFNULL / NVL Variants

---

## Q95 — NULL-Replacement Functions Across Databases

**Question:**  
What are the vendor-specific alternatives to `COALESCE`? Show examples of each.

**Answer:**
```sql
-- COALESCE: ANSI standard — works everywhere, accepts multiple arguments
SELECT COALESCE(NULL, NULL, 'fallback') AS result;  -- 'fallback'
SELECT COALESCE(comm, bonus, 0) AS total_extra FROM employees;

-- IFNULL: MySQL / SQLite — only two arguments
SELECT IFNULL(NULL, 'fallback');  -- 'fallback'
SELECT IFNULL(comm, 0) AS comm FROM employees;

-- ISNULL: SQL Server — only two arguments
SELECT ISNULL(NULL, 'fallback');  -- 'fallback'

-- NVL: Oracle — only two arguments (like IFNULL)
SELECT NVL(NULL, 'fallback') FROM dual;

-- NVL2: Oracle — three arguments (if_not_null, if_null)
SELECT NVL2(comm, 'Has Commission', 'No Commission') FROM employees;

-- NULLIF: returns NULL if two values are equal (reverse of COALESCE)
SELECT NULLIF(dept_id, 0) FROM employees;  -- turns 0 into NULL
-- Common use: prevent division by zero
SELECT salary / NULLIF(dept_id, 0) FROM employees;
```

| Function  | Vendor     | Args | Purpose                          |
|-----------|------------|------|----------------------------------|
| COALESCE  | All (ANSI) | 2+   | Return first non-NULL            |
| IFNULL    | MySQL      | 2    | Replace NULL with default        |
| ISNULL    | SQL Server | 2    | Replace NULL with default        |
| NVL       | Oracle     | 2    | Replace NULL with default        |
| NVL2      | Oracle     | 3    | Different value for NULL / not-NULL |
| NULLIF    | All (ANSI) | 2    | Return NULL if values are equal  |

---

# Section 22 — String Functions (Additional)

---

## Q96 — LPAD and RPAD

**Question:**  
Format employee IDs as zero-padded 5-character strings, and right-pad names to 15 characters with dots for a report.

**Answer:**
```sql
SELECT
  LPAD(CAST(emp_id AS VARCHAR), 5, '0') AS formatted_id,
  RPAD(name, 15, '.')                    AS formatted_name,
  salary
FROM employees
ORDER BY emp_id;
```

**Sample Output:**

| formatted_id | formatted_name  | salary |
|--------------|-----------------|--------|
| 00001        | Alice........   | 120000 |
| 00002        | Bob..........   | 95000  |
| 00003        | Carol........   | 85000  |

---

## Q97 — CONCAT_WS (Concatenate with Separator)

**Question:**  
Build a formatted address string from separate columns using a single separator.

**Answer:**
```sql
-- CONCAT_WS ignores NULL arguments (unlike CONCAT which would return NULL)
SELECT
  CONCAT_WS(', ',
    name,
    job_title,
    CAST(salary AS VARCHAR)
  ) AS employee_summary
FROM employees;

-- vs CONCAT which propagates NULLs
SELECT CONCAT('Hello', NULL, 'World');    -- NULL (bad)
SELECT CONCAT_WS(' ', 'Hello', NULL, 'World'); -- 'Hello World' (good)
```

**Sample Output:**

| employee_summary                   |
|------------------------------------|
| Alice, VP Engineering, 120000      |
| Bob, Senior Engineer, 95000        |
| Ivan, Contractor, 60000            |

---

# Section 23 — Date Functions (Additional)

---

## Q98 — DATE_TRUNC / Truncating Dates

**Question:**  
Group orders by week, month, and quarter — truncating order dates to the start of each period.

**Answer:**
```sql
-- PostgreSQL (DATE_TRUNC)
SELECT
  DATE_TRUNC('month',  order_date) AS month_start,
  DATE_TRUNC('week',   order_date) AS week_start,
  DATE_TRUNC('quarter',order_date) AS quarter_start,
  SUM(amount)                       AS revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month_start;

-- MySQL equivalent
SELECT
  DATE_FORMAT(order_date, '%Y-%m-01')  AS month_start,
  SUM(amount)                           AS revenue
FROM orders
GROUP BY DATE_FORMAT(order_date, '%Y-%m-01')
ORDER BY month_start;

-- SQL Server
SELECT
  DATEFROMPARTS(YEAR(order_date), MONTH(order_date), 1) AS month_start,
  SUM(amount) AS revenue
FROM orders
GROUP BY DATEFROMPARTS(YEAR(order_date), MONTH(order_date), 1);
```

**Sample Output:**

| month_start | revenue  |
|-------------|----------|
| 2024-01-01  | 125000   |
| 2024-02-01  | 148000   |
| 2024-03-01  | 162000   |

---

## Q99 — Working Day Calculations

**Question:**  
Find orders that were placed on a weekend. Also compute the number of calendar days between order date and today.

**Answer:**
```sql
-- MySQL: DAYOFWEEK returns 1=Sunday, 7=Saturday
SELECT order_id, order_date, amount
FROM orders
WHERE DAYOFWEEK(order_date) IN (1, 7);  -- weekend orders

-- PostgreSQL: EXTRACT(DOW) returns 0=Sunday, 6=Saturday
SELECT order_id, order_date, amount
FROM orders
WHERE EXTRACT(DOW FROM order_date) IN (0, 6);

-- Days since order
SELECT order_id, order_date,
       DATEDIFF(CURRENT_DATE, order_date) AS days_since_order
FROM orders
ORDER BY days_since_order DESC;
```

**Sample Output (weekend orders):**

| order_id | order_date | amount |
|----------|------------|--------|
| 1023     | 2024-02-10 | 350.00 |
| 1087     | 2024-03-16 | 220.00 |

---

# Section 24 — Stored Procedures and Functions

---

## Q100 — Stored Procedure vs Function

**Question:**  
What is the difference between a stored procedure and a function? Write one of each.

**Answer:**

| Feature            | Stored Procedure           | Function                      |
|--------------------|----------------------------|-------------------------------|
| Returns value?     | Optional (via OUT params)  | Must return exactly one value |
| Can use in SELECT? | No                         | Yes                           |
| Can modify data?   | Yes                        | Generally No (read-only)      |
| Called with        | CALL / EXEC                | SELECT / inside expressions   |

```sql
-- Stored Procedure: gives a raise and logs it
DELIMITER $$
CREATE PROCEDURE give_raise(IN p_emp_id INT, IN p_pct DECIMAL(5,2))
BEGIN
  UPDATE employees
  SET salary = salary * (1 + p_pct / 100)
  WHERE emp_id = p_emp_id;

  INSERT INTO salary_log (emp_id, changed_at, pct_increase)
  VALUES (p_emp_id, NOW(), p_pct);
END$$
DELIMITER ;

-- Call it
CALL give_raise(2, 10);  -- give Bob a 10% raise

-- Function: compute annual bonus (read-only, returns a value)
DELIMITER $$
CREATE FUNCTION calc_bonus(p_salary DECIMAL(10,2), p_rating INT)
RETURNS DECIMAL(10,2)
DETERMINISTIC
BEGIN
  RETURN p_salary * CASE p_rating
    WHEN 5 THEN 0.20
    WHEN 4 THEN 0.15
    WHEN 3 THEN 0.10
    ELSE 0.05
  END;
END$$
DELIMITER ;

-- Use in a SELECT
SELECT name, salary, calc_bonus(salary, 4) AS bonus
FROM employees;
```

**Sample Output:**

| name  | salary | bonus   |
|-------|--------|---------|
| Alice | 120000 | 18000   |
| Bob   | 95000  | 14250   |
| Carol | 85000  | 12750   |

---

## Q101 — Triggers

**Question:**  
Write a trigger that automatically logs every salary change in an audit table.

**Answer:**
```sql
-- Create audit table
CREATE TABLE salary_audit (
  audit_id    INT AUTO_INCREMENT PRIMARY KEY,
  emp_id      INT,
  old_salary  DECIMAL(10,2),
  new_salary  DECIMAL(10,2),
  changed_at  DATETIME,
  changed_by  VARCHAR(50)
);

-- Create AFTER UPDATE trigger
DELIMITER $$
CREATE TRIGGER trg_salary_audit
AFTER UPDATE ON employees
FOR EACH ROW
BEGIN
  IF NEW.salary <> OLD.salary THEN
    INSERT INTO salary_audit (emp_id, old_salary, new_salary, changed_at, changed_by)
    VALUES (NEW.emp_id, OLD.salary, NEW.salary, NOW(), USER());
  END IF;
END$$
DELIMITER ;

-- Now any salary update is automatically logged
UPDATE employees SET salary = 100000 WHERE emp_id = 3;

-- Check the audit trail
SELECT * FROM salary_audit;
```

**Sample Output (salary_audit after update):**

| audit_id | emp_id | old_salary | new_salary | changed_at          | changed_by |
|----------|--------|------------|------------|---------------------|------------|
| 1        | 3      | 85000      | 100000     | 2024-04-10 14:32:00 | root@localhost |

---

# Section 25 — Indexes (Advanced)

---

## Q102 — Clustered vs Non-Clustered Index

**Question:**  
What is the difference between a clustered and a non-clustered index?

**Answer:**

| Feature              | Clustered Index             | Non-Clustered Index          |
|----------------------|-----------------------------|------------------------------|
| Data storage         | Table rows stored in index order | Separate structure from rows |
| Count per table      | Only 1 allowed              | Many allowed (up to 999 in SQL Server) |
| Row lookup           | Direct — index IS the data  | Indirect — index → pointer → row |
| Speed for range scans| Very fast                   | Slower (follow pointers)     |
| Default on           | Primary key (usually)       | Any other columns            |

```sql
-- Clustered index (SQL Server)
CREATE CLUSTERED INDEX idx_cl_emp_id ON employees(emp_id);
-- Rows on disk are physically ordered by emp_id

-- Non-clustered index
CREATE NONCLUSTERED INDEX idx_nc_salary ON employees(salary);
-- Separate B-tree with pointers back to data rows
CREATE INDEX idx_dept_salary ON employees(dept_id, salary);  -- MySQL equivalent
```

---

## Q103 — Covering Index

**Question:**  
What is a covering index and how does it eliminate table lookups?

**Answer:**
```sql
-- Query we want to optimise
SELECT name, salary FROM employees WHERE dept_id = 10;

-- Without covering index: engine finds dept_id=10 rows in index,
-- then fetches full rows from table to get 'name' and 'salary' (extra I/O)
CREATE INDEX idx_dept ON employees(dept_id);

-- WITH covering index: all needed columns are IN the index itself
-- No table lookup needed at all — "index-only scan"
CREATE INDEX idx_dept_covering ON employees(dept_id, name, salary);
-- Now dept_id, name, salary are all in the index — query satisfied entirely from index
```

> **Interview tip:** A covering index includes all columns referenced in `SELECT`, `WHERE`, and `ORDER BY`. The query is answered entirely from the index without touching the main table — dramatically faster for read-heavy queries.

---

## Q104 — Index Best Practices and Anti-Patterns

**Question:**  
What are the most common indexing mistakes in SQL, and how do you fix them?

**Answer:**

**1. Wrapping indexed columns in functions (defeats the index)**
```sql
-- BAD: function on indexed column → full table scan
SELECT * FROM orders WHERE YEAR(order_date) = 2024;
SELECT * FROM employees WHERE UPPER(name) = 'ALICE';

-- GOOD: rewrite without function on the indexed column
SELECT * FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';
SELECT * FROM employees WHERE name = 'Alice';  -- or store names in consistent case
```

**2. Leading wildcard LIKE (can't use B-tree index)**
```sql
-- BAD: leading % means full scan
SELECT * FROM employees WHERE name LIKE '%son';

-- GOOD: trailing wildcard uses the index
SELECT * FROM employees WHERE name LIKE 'Alice%';

-- Alternative for full-text search: use FULLTEXT index
ALTER TABLE employees ADD FULLTEXT idx_ft_name (name);
SELECT * FROM employees WHERE MATCH(name) AGAINST ('son');
```

**3. Wrong column order in composite index**
```sql
-- Index on (dept_id, salary) — the "leftmost prefix rule"
CREATE INDEX idx_dept_sal ON employees(dept_id, salary);

SELECT * FROM employees WHERE dept_id = 10;               -- ✅ uses index
SELECT * FROM employees WHERE dept_id = 10 AND salary > 90000; -- ✅ uses index
SELECT * FROM employees WHERE salary > 90000;             -- ❌ cannot use index (skips dept_id)
```

---

# Section 26 — Database Design Concepts

---

## Q105 — OLAP vs OLTP

**Question:**  
What is the difference between OLTP and OLAP databases? How does this affect schema design?

**Answer:**

| Feature          | OLTP                          | OLAP                          |
|------------------|-------------------------------|-------------------------------|
| Purpose          | Day-to-day transactions       | Analytics and reporting       |
| Query type       | Simple, single-row reads/writes | Complex, multi-table aggregates |
| Data volume      | Current data, GBs             | Historical data, TBs–PBs      |
| Schema style     | Normalized (3NF)              | Denormalized (Star/Snowflake) |
| Optimized for    | Write speed, concurrency      | Read/scan speed               |
| Examples         | E-commerce orders, banking    | Data warehouses, dashboards   |
| Index strategy   | Primary + foreign keys        | Columnar, bitmap indexes      |

```sql
-- OLTP schema (normalized — no redundancy)
orders(order_id, customer_id, product_id, order_date, quantity)
customers(customer_id, name, email)
products(product_id, name, category, price)

-- OLAP schema (denormalized fact table — everything in one wide table)
fact_sales(date_key, customer_name, product_name, category, region,
           quantity, unit_price, total_revenue)
-- Redundant but enables fast aggregation without joins
```

---

## Q106 — Star Schema vs Snowflake Schema

**Question:**  
Explain star schema and snowflake schema with examples.

**Answer:**

**Star Schema:** One central fact table connected to flat (denormalized) dimension tables. Simple, fast, widely used.

```
                    dim_date
                       |
dim_customer ─── fact_sales ─── dim_product
                       |
                   dim_store
```

```sql
-- Fact table (measurable events)
CREATE TABLE fact_sales (
  sale_id      INT,
  date_key     INT,     -- FK to dim_date
  customer_key INT,     -- FK to dim_customer
  product_key  INT,     -- FK to dim_product
  quantity     INT,
  revenue      DECIMAL(12,2)
);

-- Dimension table (flat, denormalized)
CREATE TABLE dim_product (
  product_key INT PRIMARY KEY,
  product_name VARCHAR(100),
  category     VARCHAR(50),
  subcategory  VARCHAR(50),   -- repeated/denormalized here
  brand        VARCHAR(50)
);
```

**Snowflake Schema:** Dimension tables are normalized further into sub-dimension tables. Less redundancy, more joins.

```sql
-- Dimension table (normalized — split subcategory out)
CREATE TABLE dim_product (
  product_key    INT PRIMARY KEY,
  product_name   VARCHAR(100),
  subcategory_key INT    -- FK to dim_subcategory
);

CREATE TABLE dim_subcategory (
  subcategory_key INT PRIMARY KEY,
  subcategory     VARCHAR(50),
  category_key    INT    -- FK to dim_category
);

CREATE TABLE dim_category (
  category_key INT PRIMARY KEY,
  category     VARCHAR(50)
);
```

| Feature         | Star Schema            | Snowflake Schema          |
|-----------------|------------------------|---------------------------|
| Complexity      | Simple                 | More complex              |
| Query speed     | Faster (fewer joins)   | Slower (more joins)       |
| Storage         | More (redundant data)  | Less (normalized)         |
| ETL complexity  | Simpler                | More complex              |
| Common in       | Most data warehouses   | When storage is a concern |

---

## Q107 — BCNF (Boyce-Codd Normal Form)

**Question:**  
What is BCNF and how does it differ from 3NF?

**Answer:**

**3NF allows** a non-trivial functional dependency `X → Y` where X is not a superkey, *as long as Y is part of a candidate key*. BCNF is stricter — it requires that for every non-trivial functional dependency `X → Y`, X must be a superkey.

```sql
-- Example violating BCNF (but satisfying 3NF)
-- Table: course_teacher(student, course, teacher)
-- A teacher teaches only one course: teacher → course
-- A student takes a course with one teacher: (student, course) → teacher
-- Candidate keys: (student, course) and (student, teacher)
-- teacher → course: teacher is NOT a superkey → violates BCNF

-- BCNF fix: decompose
CREATE TABLE teacher_course (teacher VARCHAR(50), course VARCHAR(50));
CREATE TABLE student_teacher (student VARCHAR(50), teacher VARCHAR(50));
```

> **Interview tip:** Most practical databases aim for 3NF. BCNF is mentioned in interviews to test depth of understanding — the difference matters mainly for theoretical database design.

---

# Section 27 — Advanced Patterns (Additional)

---

## Q108 — Running Rank without Window Functions

**Question:**  
Rank employees by salary without using window functions (for databases that don't support them or as an interview variant).

**Answer:**
```sql
-- Correlated subquery approach (works everywhere)
SELECT
  e1.name,
  e1.salary,
  (SELECT COUNT(DISTINCT e2.salary)
   FROM employees e2
   WHERE e2.salary >= e1.salary) AS dense_rank_equiv
FROM employees e1
ORDER BY e1.salary DESC;
```

**Sample Output:**

| name  | salary | dense_rank_equiv |
|-------|--------|-----------------|
| Frank | 130000 | 1               |
| Alice | 120000 | 2               |
| Dave  | 110000 | 3               |
| Bob   | 95000  | 4               |
| Grace | 88000  | 5               |

---

## Q109 — Cohort Analysis

**Question:**  
Group customers by their first-purchase month (cohort) and track how many placed a second order in each subsequent month.

**Answer:**
```sql
WITH first_orders AS (
  SELECT customer_id,
         DATE_FORMAT(MIN(order_date), '%Y-%m') AS cohort_month
  FROM orders
  GROUP BY customer_id
),
subsequent AS (
  SELECT
    fo.customer_id,
    fo.cohort_month,
    DATE_FORMAT(o.order_date, '%Y-%m') AS order_month,
    PERIOD_DIFF(
      DATE_FORMAT(o.order_date, '%Y%m'),
      DATE_FORMAT(STR_TO_DATE(CONCAT(fo.cohort_month, '-01'), '%Y-%m-%d'), '%Y%m')
    ) AS months_since_cohort
  FROM first_orders fo
  JOIN orders o ON fo.customer_id = o.customer_id
)
SELECT
  cohort_month,
  months_since_cohort,
  COUNT(DISTINCT customer_id) AS active_customers
FROM subsequent
GROUP BY cohort_month, months_since_cohort
ORDER BY cohort_month, months_since_cohort;
```

**Sample Output:**

| cohort_month | months_since_cohort | active_customers |
|--------------|---------------------|-----------------|
| 2024-01      | 0                   | 200             |
| 2024-01      | 1                   | 134             |
| 2024-01      | 2                   | 98              |
| 2024-01      | 3                   | 75              |

---

## Q110 — Detecting Outliers with Standard Deviation

**Question:**  
Find employees whose salary is more than 2 standard deviations away from the company mean (statistical outliers).

**Answer:**
```sql
WITH stats AS (
  SELECT
    AVG(salary)    AS mean_sal,
    STDDEV(salary) AS stddev_sal
  FROM employees
)
SELECT
  e.name,
  e.salary,
  ROUND((e.salary - s.mean_sal) / s.stddev_sal, 2) AS z_score
FROM employees e, stats s
WHERE ABS(e.salary - s.mean_sal) > 2 * s.stddev_sal
ORDER BY ABS(e.salary - s.mean_sal) DESC;
```

**Sample Output:**

| name  | salary | z_score |
|-------|--------|---------|
| Frank | 130000 | 2.31    |
| Ivan  | 60000  | -2.14   |

---

## Q111 — Finding the First Non-NULL Value Across Columns

**Question:**  
A table has three bonus columns (bonus_q1, bonus_q2, bonus_q3). Return the first non-NULL bonus for each employee.

**Answer:**
```sql
SELECT
  name,
  COALESCE(bonus_q1, bonus_q2, bonus_q3) AS first_bonus
FROM employee_bonuses;

-- If you also want to know WHICH quarter it came from
SELECT
  name,
  CASE
    WHEN bonus_q1 IS NOT NULL THEN CONCAT('Q1: ', bonus_q1)
    WHEN bonus_q2 IS NOT NULL THEN CONCAT('Q2: ', bonus_q2)
    WHEN bonus_q3 IS NOT NULL THEN CONCAT('Q3: ', bonus_q3)
    ELSE 'No bonus'
  END AS first_bonus_source
FROM employee_bonuses;
```

**Sample Input:**

| name  | bonus_q1 | bonus_q2 | bonus_q3 |
|-------|----------|----------|----------|
| Alice | 5000     | 6000     | 4500     |
| Bob   | NULL     | 3000     | 2000     |
| Carol | NULL     | NULL     | 1500     |
| Dave  | NULL     | NULL     | NULL     |

**Sample Output:**

| name  | first_bonus |
|-------|-------------|
| Alice | 5000        |
| Bob   | 3000        |
| Carol | 1500        |
| Dave  | NULL        |

---

## Q112 — Comparing Across Rows: Previous vs Current Value

**Question:**  
Find all orders where the amount increased compared to the previous order for the same customer.

**Answer:**
```sql
SELECT customer_id, order_id, order_date, amount, prev_amount
FROM (
  SELECT
    customer_id,
    order_id,
    order_date,
    amount,
    LAG(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_amount
  FROM orders
) t
WHERE amount > prev_amount
ORDER BY customer_id, order_date;
```

**Sample Input:**

| customer_id | order_id | order_date | amount |
|-------------|----------|------------|--------|
| 101         | 1        | 2024-01-10 | 100    |
| 101         | 2        | 2024-02-15 | 80     |
| 101         | 3        | 2024-03-20 | 150    |

**Sample Output:**

| customer_id | order_id | amount | prev_amount |
|-------------|----------|--------|-------------|
| 101         | 3        | 150    | 80          |

---

# Section 28 — Quick Reference Additions

---

## Q113 — SQL Execution Order

**Question:**  
In what order does SQL actually evaluate the clauses of a SELECT statement?

**Answer:**

```sql
SELECT   dept_id, AVG(salary) AS avg_sal   -- 7. Evaluated last (aliases defined here)
FROM     employees                          -- 1. Which tables to use
JOIN     departments ON ...                -- 2. Combine tables
WHERE    salary > 50000                    -- 3. Filter individual rows (before grouping)
GROUP BY dept_id                           -- 4. Group the filtered rows
HAVING   AVG(salary) > 80000              -- 5. Filter groups
ORDER BY avg_sal DESC                      -- 6. Sort results
LIMIT    5;                                -- 8. Restrict row count
```

**Execution order:**

| Step | Clause   | What Happens                                      |
|------|----------|---------------------------------------------------|
| 1    | FROM     | Identify source tables                            |
| 2    | JOIN     | Combine tables                                    |
| 3    | WHERE    | Filter rows (aliases NOT yet available here)      |
| 4    | GROUP BY | Group filtered rows                               |
| 5    | HAVING   | Filter groups (aggregates available)              |
| 6    | SELECT   | Compute expressions and aliases                   |
| 7    | DISTINCT | Remove duplicates (if specified)                  |
| 8    | ORDER BY | Sort (aliases NOW available — evaluated after SELECT) |
| 9    | LIMIT    | Restrict final row count                          |

> **Why this matters in interviews:** It explains why you can't use a SELECT alias in a WHERE clause (alias doesn't exist yet at step 3), but you CAN use it in ORDER BY (evaluated after SELECT).

---

## Q114 — Data Types: Choosing the Right One

**Question:**  
What key data type decisions come up most in SQL interviews?

**Answer:**

```sql
-- INT vs BIGINT: use BIGINT for IDs on large tables (INT max = 2.1B)
user_id BIGINT AUTO_INCREMENT PRIMARY KEY,  -- safe for scale

-- CHAR vs VARCHAR
country_code CHAR(2),       -- CHAR: fixed length, faster comparisons
description  VARCHAR(500),  -- VARCHAR: variable length, saves space

-- DECIMAL vs FLOAT
price    DECIMAL(10, 2),  -- exact — always for money/finance
rating   FLOAT,           -- approximate — ok for scientific data
-- NEVER use FLOAT for money: 0.1 + 0.2 = 0.30000000000000004

-- DATE vs DATETIME vs TIMESTAMP
hire_date   DATE,          -- date only (no time)
created_at  DATETIME,      -- date + time, not timezone-aware
updated_at  TIMESTAMP,     -- date + time, auto-converts to UTC

-- TEXT vs VARCHAR
bio         TEXT,          -- for large, unindexable blobs of text
name        VARCHAR(100),  -- for shorter, indexable strings
```

| Type         | Use When                            | Avoid When              |
|--------------|-------------------------------------|-------------------------|
| TINYINT      | 0-255 flags/booleans               | IDs that may grow       |
| INT          | IDs, counts (<2.1B)                | Very large tables       |
| BIGINT       | IDs on big tables                  | Unnecessary (wastes space) |
| DECIMAL(p,s) | Money, exact arithmetic            | Approximate scientific data |
| FLOAT/DOUBLE | Scientific calculations            | Financial data          |
| CHAR(n)      | Fixed-length codes (ISO codes)     | Variable-length text    |
| VARCHAR(n)   | Names, titles, most strings        | Very large text         |
| TEXT         | Long descriptions, JSON blobs      | Things you need to index |
| DATE         | Birth dates, event dates (no time) | When time matters       |
| TIMESTAMP    | Created/updated audit columns      | Historical dates (1970+) only |

---

## Q115 — Window Function Framing Cheat Sheet

**Question:**  
What are the standard window frame options and what does each do?

**Answer:**

```sql
-- Frame syntax
SUM(salary) OVER (
  PARTITION BY dept_id           -- optional: reset per group
  ORDER BY hire_date             -- required for frames
  ROWS | RANGE                   -- ROWS = physical rows, RANGE = by value
  BETWEEN <start> AND <end>      -- the frame bounds
)

-- Common frame patterns:

-- Running total (accumulate from beginning to current row)
SUM(sal) OVER (ORDER BY hire_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

-- Entire partition (use for window aggregates without running logic)
AVG(sal) OVER (PARTITION BY dept_id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)

-- Moving average of last 3 rows (including current)
AVG(sal) OVER (ORDER BY hire_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)

-- Look-ahead: next 2 rows
SUM(sal) OVER (ORDER BY hire_date ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING)

-- Centered: 1 row before and 1 row after
AVG(sal) OVER (ORDER BY hire_date ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
```

| Frame Bound         | Meaning                                        |
|---------------------|------------------------------------------------|
| UNBOUNDED PRECEDING | From the very first row in the partition       |
| n PRECEDING         | n rows before the current row                  |
| CURRENT ROW         | The current row itself                         |
| n FOLLOWING         | n rows after the current row                   |
| UNBOUNDED FOLLOWING | To the very last row in the partition          |

---

## Q116 — Cheat Sheet: Aggregate vs Window Functions

**Question:**  
When should you use an aggregate function vs a window function?

**Answer:**

```sql
-- Aggregate: collapses rows — you LOSE individual row detail
SELECT dept_id, AVG(salary) AS avg_sal
FROM employees
GROUP BY dept_id;
-- 3 rows output (one per dept)

-- Window: keeps all rows AND adds aggregate alongside
SELECT name, dept_id, salary,
       AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
-- 9 rows output (all rows preserved)

-- Rule of thumb:
-- Need 1 row per group?  → GROUP BY + aggregate
-- Need detail + summary? → Window function
-- Need row-to-row comparison? → LAG / LEAD
-- Need rank within group? → RANK / DENSE_RANK / ROW_NUMBER
-- Need running total?  → SUM OVER (ORDER BY ... ROWS UNBOUNDED PRECEDING)
```

---

## Q117 — Deadlocks

**Question:**  
What is a deadlock in a database, and how do you prevent or resolve it?

**Answer:**

> A **deadlock** occurs when two or more transactions each hold a lock that the other needs, creating a circular wait — neither can proceed.

```
Transaction A:           Transaction B:
LOCK table accounts      LOCK table orders
   (waiting for orders)     (waiting for accounts)
       ← DEADLOCK →
```

**Prevention strategies:**
```sql
-- 1. Always access tables in the same order across all transactions
-- Transaction A and B both: lock accounts THEN orders (never reverse)

-- 2. Keep transactions short — acquire locks late, release early
BEGIN;
SELECT ... FOR UPDATE;  -- acquire lock
-- do minimal work
UPDATE ...;
COMMIT;  -- release immediately

-- 3. Use lower isolation levels where possible (fewer locks)
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- 4. Use SELECT ... FOR UPDATE only when you will modify the row
-- Avoid SELECT ... FOR UPDATE for read-only operations
```

**When a deadlock occurs:**
- The database automatically detects it and kills one transaction (the "victim")
- The killed transaction receives an error and must be retried by the application

---

## Q118 — Partitioning

**Question:**  
What is table partitioning and when should you use it?

**Answer:**

> **Partitioning** splits one large table into smaller physical pieces (partitions) based on a column's value. Queries that filter on the partition key only scan relevant partitions (**partition pruning**).

```sql
-- Range partitioning by year (MySQL)
CREATE TABLE orders (
  order_id   INT,
  order_date DATE,
  amount     DECIMAL(10,2)
)
PARTITION BY RANGE (YEAR(order_date)) (
  PARTITION p2022 VALUES LESS THAN (2023),
  PARTITION p2023 VALUES LESS THAN (2024),
  PARTITION p2024 VALUES LESS THAN (2025),
  PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- This query only scans partition p2024 (partition pruning)
SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';

-- List partitioning (by category)
PARTITION BY LIST (dept_id) (
  PARTITION eng   VALUES IN (10),
  PARTITION mktg  VALUES IN (20),
  PARTITION sales VALUES IN (30)
);
```

| Partition Type | Use Case                                  |
|----------------|-------------------------------------------|
| RANGE          | Dates, IDs (most common for time-series) |
| LIST           | Discrete values (regions, categories)    |
| HASH           | Evenly distribute rows, no obvious key   |
| KEY            | Similar to HASH, uses MySQL's own hash   |

---

## Q119 — Five Hardest SQL Interview Questions Summary

**Question:**  
What are the most commonly asked "hard" SQL interview questions? Provide a quick-reference summary.

**Answer:**

**1. Nth highest salary**
```sql
SELECT salary FROM (
  SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS dr FROM employees
) t WHERE dr = N;
```

**2. Consecutive rows / streaks**
```sql
-- Row number minus date = same value for consecutive days
DATE_SUB(login_date, INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) DAY) AS grp
```

**3. Employees earning more than their manager**
```sql
SELECT e.name FROM employees e JOIN employees m ON e.manager_id = m.emp_id WHERE e.salary > m.salary;
```

**4. Running total**
```sql
SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
```

**5. Duplicate detection and deletion**
```sql
DELETE FROM t WHERE id NOT IN (SELECT MIN(id) FROM t GROUP BY name, dept_id);
-- or CTE + ROW_NUMBER WHERE rn > 1
```

---

## Q120 — Interview Tips: Common Mistakes to Avoid

**Question:**  
What are the most common SQL mistakes candidates make in interviews?

**Answer:**

| Mistake | Problem | Fix |
|---------|---------|-----|
| `WHERE` on aggregate | `WHERE AVG(salary) > 80000` fails | Use `HAVING` |
| `SELECT *` | Fetches unnecessary data, brittle | List columns explicitly |
| `NOT IN` with NULLs | Returns 0 rows silently | Use `NOT EXISTS` or filter NULLs |
| `= NULL` | Always returns 0 rows | Use `IS NULL` |
| Missing `DISTINCT` in subquery | Duplicate joins inflate results | Add `DISTINCT` or check join logic |
| Forgetting NULL in `GROUP BY` | NULLs form their own group | Use `COALESCE` in GROUP BY if needed |
| `HAVING` without `GROUP BY` | Treats whole table as one group | Usually correct but rarely intended |
| Alias in `WHERE` | Alias not yet defined at WHERE stage | Use subquery / CTE |
| Division by zero | Crashes query | Wrap divisor with `NULLIF(col, 0)` |
| FLOAT for money | Rounding errors (0.1+0.2≠0.3) | Use `DECIMAL(p,s)` |

```sql
-- Most common: alias in WHERE (fails)
SELECT salary * 12 AS annual FROM employees WHERE annual > 100000;  -- ERROR

-- Fix: subquery
SELECT annual FROM (
  SELECT salary * 12 AS annual FROM employees
) t WHERE annual > 100000;

-- Or CTE
WITH t AS (SELECT salary * 12 AS annual FROM employees)
SELECT annual FROM t WHERE annual > 100000;
```

---

*End of Supplement — Q71 through Q120*  
*Together with the original guide (Q1–Q70), this covers the complete SQL interview curriculum.*
