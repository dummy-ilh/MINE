# Grokking the SQL Interview — Study Guide

> Covers all core SQL interview topics: retrieval, joins, aggregation, subqueries,
> window functions, string/date manipulation, NULL handling, set operations,
> schema design, indexing, and advanced patterns.

---

## Reference Tables Used Throughout

```sql
-- Employees
CREATE TABLE employees (
  emp_id     INT PRIMARY KEY,
  name       VARCHAR(50),
  dept_id    INT,
  manager_id INT,
  salary     DECIMAL(10,2),
  hire_date  DATE,
  job_title  VARCHAR(50)
);

-- Departments
CREATE TABLE departments (
  dept_id   INT PRIMARY KEY,
  dept_name VARCHAR(50),
  location  VARCHAR(50)
);

-- Orders
CREATE TABLE orders (
  order_id    INT PRIMARY KEY,
  customer_id INT,
  order_date  DATE,
  amount      DECIMAL(10,2),
  status      VARCHAR(20)
);

-- Products
CREATE TABLE products (
  product_id  INT PRIMARY KEY,
  name        VARCHAR(50),
  category    VARCHAR(30),
  price       DECIMAL(10,2)
);
```

**employees sample data:**

| emp_id | name    | dept_id | manager_id | salary   | hire_date  | job_title         |
|--------|---------|---------|------------|----------|------------|-------------------|
| 1      | Alice   | 10      | NULL       | 120000   | 2018-03-01 | VP Engineering    |
| 2      | Bob     | 10      | 1          | 95000    | 2019-06-15 | Senior Engineer   |
| 3      | Carol   | 10      | 1          | 85000    | 2020-01-10 | Engineer          |
| 4      | Dave    | 20      | NULL       | 110000   | 2017-08-20 | VP Marketing      |
| 5      | Eve     | 20      | 4          | 72000    | 2021-03-05 | Marketing Analyst |
| 6      | Frank   | 30      | NULL       | 130000   | 2016-05-11 | VP Sales          |
| 7      | Grace   | 30      | 6          | 88000    | 2020-09-01 | Sales Lead        |
| 8      | Heidi   | 30      | 6          | 76000    | 2022-01-15 | Sales Rep         |
| 9      | Ivan    | NULL    | NULL       | 60000    | 2023-07-01 | Contractor        |

**departments sample data:**

| dept_id | dept_name   | location     |
|---------|-------------|--------------|
| 10      | Engineering | San Francisco|
| 20      | Marketing   | New York     |
| 30      | Sales       | Chicago      |
| 40      | HR          | Austin       |

---

# Section 1 — Basic Retrieval

---

## Q1 — Retrieve All Columns and Rows

**Question:**  
Write a query to fetch every row and column from the `employees` table.

**Answer:**
```sql
SELECT * FROM employees;
```

**Sample Output:** *(all 9 rows from the reference table above)*

> **Interview tip:** In production code, always list columns explicitly rather than using `*` — it prevents bugs when columns are added or reordered.

---

## Q2 — Filter Rows with WHERE

**Question:**  
Find all employees in department 10 who earn more than $90,000.

**Answer:**
```sql
SELECT name, salary, dept_id
FROM employees
WHERE dept_id = 10
  AND salary > 90000;
```

**Sample Output:**

| name  | salary | dept_id |
|-------|--------|---------|
| Alice | 120000 | 10      |
| Bob   | 95000  | 10      |

---

## Q3 — Column Aliases

**Question:**  
Return each employee's name and annual salary, but label the salary column `annual_salary`. Also show a computed monthly salary.

**Answer:**
```sql
SELECT
  name,
  salary         AS annual_salary,
  salary / 12    AS monthly_salary
FROM employees;
```

**Sample Output:**

| name  | annual_salary | monthly_salary |
|-------|---------------|----------------|
| Alice | 120000        | 10000.00       |
| Bob   | 95000         | 7916.67        |
| Carol | 85000         | 7083.33        |

---

## Q4 — DISTINCT Values

**Question:**  
Return a unique list of job titles from the `employees` table.

**Answer:**
```sql
SELECT DISTINCT job_title
FROM employees
ORDER BY job_title;
```

**Sample Output:**

| job_title         |
|-------------------|
| Contractor        |
| Engineer          |
| Marketing Analyst |
| Sales Lead        |
| Sales Rep         |
| Senior Engineer   |
| VP Engineering    |
| VP Marketing      |
| VP Sales          |

---

## Q5 — Sorting Results

**Question:**  
List all employees ordered by department (ascending) and then by salary (descending) within each department.

**Answer:**
```sql
SELECT name, dept_id, salary
FROM employees
ORDER BY dept_id ASC, salary DESC;
```

**Sample Output:**

| name  | dept_id | salary |
|-------|---------|--------|
| Alice | 10      | 120000 |
| Bob   | 10      | 95000  |
| Carol | 10      | 85000  |
| Dave  | 20      | 110000 |
| Eve   | 20      | 72000  |
| Frank | 30      | 130000 |
| Grace | 30      | 88000  |
| Heidi | 30      | 76000  |
| Ivan  | NULL    | 60000  |

---

## Q6 — LIMIT / TOP / FETCH FIRST

**Question:**  
Return only the top 3 highest-paid employees.

**Answer:**
```sql
-- MySQL / PostgreSQL / SQLite
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 3;

-- SQL Server
SELECT TOP 3 name, salary
FROM employees
ORDER BY salary DESC;

-- Oracle
SELECT name, salary
FROM employees
ORDER BY salary DESC
FETCH FIRST 3 ROWS ONLY;
```

**Sample Output:**

| name  | salary |
|-------|--------|
| Frank | 130000 |
| Alice | 120000 |
| Dave  | 110000 |

---

## Q7 — Pattern Matching with LIKE

**Question:**  
Find all employees whose name starts with a vowel (A, E, I, O, or U).

**Answer:**
```sql
SELECT name
FROM employees
WHERE name LIKE 'A%'
   OR name LIKE 'E%'
   OR name LIKE 'I%'
   OR name LIKE 'O%'
   OR name LIKE 'U%';
```

**Sample Output:**

| name  |
|-------|
| Alice |
| Eve   |
| Ivan  |

---

## Q8 — IN and BETWEEN

**Question:**  
Find employees in departments 10 or 30, with salaries between $80,000 and $130,000.

**Answer:**
```sql
SELECT name, dept_id, salary
FROM employees
WHERE dept_id IN (10, 30)
  AND salary BETWEEN 80000 AND 130000;
```

**Sample Output:**

| name  | dept_id | salary |
|-------|---------|--------|
| Alice | 10      | 120000 |
| Bob   | 10      | 95000  |
| Carol | 10      | 85000  |
| Frank | 30      | 130000 |
| Grace | 30      | 88000  |

---

## Q9 — Handling NULL Values

**Question:**  
Find all employees who have no manager (i.e., `manager_id` is NULL). Also write a version that replaces NULL manager IDs with the string `'No Manager'`.

**Answer:**
```sql
-- Find employees with no manager
SELECT name, manager_id
FROM employees
WHERE manager_id IS NULL;

-- Replace NULL with a label
SELECT name,
       COALESCE(CAST(manager_id AS VARCHAR), 'No Manager') AS manager
FROM employees;
```

**Sample Output (IS NULL query):**

| name  | manager_id |
|-------|------------|
| Alice | NULL       |
| Dave  | NULL       |
| Frank | NULL       |
| Ivan  | NULL       |

---

## Q10 — CASE Expressions

**Question:**  
Classify each employee's salary into a band: `'Entry'` (< 75k), `'Mid'` (75k–100k), `'Senior'` (> 100k).

**Answer:**
```sql
SELECT name, salary,
  CASE
    WHEN salary < 75000  THEN 'Entry'
    WHEN salary <= 100000 THEN 'Mid'
    ELSE 'Senior'
  END AS salary_band
FROM employees
ORDER BY salary;
```

**Sample Output:**

| name  | salary | salary_band |
|-------|--------|-------------|
| Ivan  | 60000  | Entry       |
| Eve   | 72000  | Entry       |
| Heidi | 76000  | Mid         |
| Carol | 85000  | Mid         |
| Grace | 88000  | Mid         |
| Bob   | 95000  | Mid         |
| Dave  | 110000 | Senior      |
| Alice | 120000 | Senior      |
| Frank | 130000 | Senior      |

---

# Section 2 — Aggregations and Grouping

---

## Q11 — Basic Aggregation Functions

**Question:**  
Find the total payroll, average salary, minimum salary, maximum salary, and number of employees company-wide.

**Answer:**
```sql
SELECT
  COUNT(*)       AS total_employees,
  SUM(salary)    AS total_payroll,
  AVG(salary)    AS avg_salary,
  MIN(salary)    AS min_salary,
  MAX(salary)    AS max_salary
FROM employees;
```

**Sample Output:**

| total_employees | total_payroll | avg_salary | min_salary | max_salary |
|-----------------|---------------|------------|------------|------------|
| 9               | 836000        | 92888.89   | 60000      | 130000     |

---

## Q12 — GROUP BY

**Question:**  
Show the number of employees and average salary for each department.

**Answer:**
```sql
SELECT
  dept_id,
  COUNT(*)    AS headcount,
  AVG(salary) AS avg_salary
FROM employees
GROUP BY dept_id
ORDER BY dept_id;
```

**Sample Output:**

| dept_id | headcount | avg_salary |
|---------|-----------|------------|
| 10      | 3         | 100000.00  |
| 20      | 2         | 91000.00   |
| 30      | 3         | 98000.00   |
| NULL    | 1         | 60000.00   |

---

## Q13 — HAVING vs WHERE

**Question:**  
Return only departments that have more than 2 employees AND an average salary above $90,000.

**Answer:**
```sql
SELECT
  dept_id,
  COUNT(*)    AS headcount,
  AVG(salary) AS avg_salary
FROM employees
GROUP BY dept_id
HAVING COUNT(*) > 2
   AND AVG(salary) > 90000;
```

**Sample Output:**

| dept_id | headcount | avg_salary |
|---------|-----------|------------|
| 10      | 3         | 100000.00  |
| 30      | 3         | 98000.00   |

> **Interview tip:** `WHERE` filters rows **before** grouping. `HAVING` filters **after** grouping. You cannot use aggregate functions in a `WHERE` clause.

---

## Q14 — COUNT(*) vs COUNT(column)

**Question:**  
What is the difference between `COUNT(*)` and `COUNT(dept_id)` in the employees table?

**Answer:**
```sql
SELECT
  COUNT(*)       AS count_all_rows,
  COUNT(dept_id) AS count_non_null_dept
FROM employees;
```

**Sample Output:**

| count_all_rows | count_non_null_dept |
|----------------|---------------------|
| 9              | 8                   |

> `COUNT(*)` counts every row. `COUNT(column)` ignores rows where that column is NULL. Ivan has NULL dept_id, so `COUNT(dept_id)` = 8.

---

## Q15 — Aggregation with CASE (Conditional Counting)

**Question:**  
Count how many employees are VPs versus non-VPs in each department.

**Answer:**
```sql
SELECT
  dept_id,
  SUM(CASE WHEN job_title LIKE 'VP%' THEN 1 ELSE 0 END) AS vp_count,
  SUM(CASE WHEN job_title NOT LIKE 'VP%' THEN 1 ELSE 0 END) AS non_vp_count
FROM employees
WHERE dept_id IS NOT NULL
GROUP BY dept_id
ORDER BY dept_id;
```

**Sample Output:**

| dept_id | vp_count | non_vp_count |
|---------|----------|--------------|
| 10      | 1        | 2            |
| 20      | 1        | 1            |
| 30      | 1        | 2            |

---

# Section 3 — JOINs

---

## Q16 — INNER JOIN

**Question:**  
Return each employee's name alongside their department name. Only include employees who belong to a department.

**Answer:**
```sql
SELECT e.name, d.dept_name, d.location
FROM employees e
INNER JOIN departments d ON e.dept_id = d.dept_id;
```

**Sample Output:**

| name  | dept_name   | location      |
|-------|-------------|---------------|
| Alice | Engineering | San Francisco |
| Bob   | Engineering | San Francisco |
| Carol | Engineering | San Francisco |
| Dave  | Marketing   | New York      |
| Eve   | Marketing   | New York      |
| Frank | Sales       | Chicago       |
| Grace | Sales       | Chicago       |
| Heidi | Sales       | Chicago       |

> Ivan is excluded because his `dept_id` is NULL and doesn't match any department.

---

## Q17 — LEFT JOIN

**Question:**  
Return ALL employees, including those with no department, along with their department name (show NULL if no department).

**Answer:**
```sql
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
ORDER BY e.emp_id;
```

**Sample Output:**

| name  | dept_name   |
|-------|-------------|
| Alice | Engineering |
| Bob   | Engineering |
| Carol | Engineering |
| Dave  | Marketing   |
| Eve   | Marketing   |
| Frank | Sales       |
| Grace | Sales       |
| Heidi | Sales       |
| Ivan  | NULL        |

---

## Q18 — RIGHT JOIN

**Question:**  
Return ALL departments, including those with no employees, along with employee names.

**Answer:**
```sql
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id
ORDER BY d.dept_id;
```

**Sample Output:**

| name  | dept_name   |
|-------|-------------|
| Alice | Engineering |
| Bob   | Engineering |
| Carol | Engineering |
| Dave  | Marketing   |
| Eve   | Marketing   |
| Frank | Sales       |
| Grace | Sales       |
| Heidi | Sales       |
| NULL  | HR          |

> HR has no employees, so it appears with a NULL name.

---

## Q19 — FULL OUTER JOIN

**Question:**  
Return all employees and all departments, even if there is no match on either side.

**Answer:**
```sql
-- PostgreSQL, SQL Server, DB2
SELECT e.name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;

-- MySQL workaround (UNION of LEFT + RIGHT)
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
UNION
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;
```

**Sample Output:**

| name  | dept_name   |
|-------|-------------|
| Alice | Engineering |
| ...   | ...         |
| Heidi | Sales       |
| Ivan  | NULL        |
| NULL  | HR          |

---

## Q20 — SELF JOIN

**Question:**  
List each employee alongside their manager's name.

**Answer:**
```sql
SELECT
  e.name        AS employee,
  m.name        AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id
ORDER BY e.emp_id;
```

**Sample Output:**

| employee | manager |
|----------|---------|
| Alice    | NULL    |
| Bob      | Alice   |
| Carol    | Alice   |
| Dave     | NULL    |
| Eve      | Dave    |
| Frank    | NULL    |
| Grace    | Frank   |
| Heidi    | Frank   |
| Ivan     | NULL    |

---

## Q21 — CROSS JOIN

**Question:**  
Generate every possible pairing of departments. (Used for generating combinations.)

**Answer:**
```sql
SELECT
  a.dept_name AS dept_1,
  b.dept_name AS dept_2
FROM departments a
CROSS JOIN departments b
WHERE a.dept_id < b.dept_id
ORDER BY a.dept_id, b.dept_id;
```

**Sample Output:**

| dept_1      | dept_2      |
|-------------|-------------|
| Engineering | Marketing   |
| Engineering | Sales       |
| Engineering | HR          |
| Marketing   | Sales       |
| Marketing   | HR          |
| Sales       | HR          |

---

## Q22 — Finding Unmatched Rows (Anti-Join)

**Question:**  
Find all departments that currently have no employees.

**Answer:**
```sql
-- Using LEFT JOIN + IS NULL
SELECT d.dept_name
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
WHERE e.emp_id IS NULL;

-- Using NOT EXISTS
SELECT d.dept_name
FROM departments d
WHERE NOT EXISTS (
  SELECT 1 FROM employees e WHERE e.dept_id = d.dept_id
);

-- Using NOT IN (caution with NULLs)
SELECT dept_name
FROM departments
WHERE dept_id NOT IN (
  SELECT dept_id FROM employees WHERE dept_id IS NOT NULL
);
```

**Sample Output:**

| dept_name |
|-----------|
| HR        |

---

## Q23 — Multi-Table JOIN

**Question:**  
Show each employee's name, department name, and manager's name in one result set.

**Answer:**
```sql
SELECT
  e.name       AS employee,
  d.dept_name,
  m.name       AS manager
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id
LEFT JOIN employees m  ON e.manager_id = m.emp_id
ORDER BY d.dept_name, e.name;
```

**Sample Output:**

| employee | dept_name   | manager |
|----------|-------------|---------|
| Alice    | Engineering | NULL    |
| Bob      | Engineering | Alice   |
| Carol    | Engineering | Alice   |
| Dave     | Marketing   | NULL    |
| Eve      | Marketing   | Dave    |
| Frank    | Sales       | NULL    |
| Grace    | Sales       | Frank   |
| Heidi    | Sales       | Frank   |
| Ivan     | NULL        | NULL    |

---

# Section 4 — Subqueries

---

## Q24 — Scalar Subquery

**Question:**  
Return each employee's name, salary, and the difference between their salary and the company-wide average.

**Answer:**
```sql
SELECT
  name,
  salary,
  salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees
ORDER BY diff_from_avg DESC;
```

**Sample Output:**

| name  | salary | diff_from_avg |
|-------|--------|---------------|
| Frank | 130000 | 37111.11      |
| Alice | 120000 | 27111.11      |
| Dave  | 110000 | 17111.11      |
| Bob   | 95000  | 2111.11       |
| Grace | 88000  | -4888.89      |
| Carol | 85000  | -7888.89      |
| Heidi | 76000  | -16888.89     |
| Eve   | 72000  | -20888.89     |
| Ivan  | 60000  | -32888.89     |

---

## Q25 — Subquery in WHERE

**Question:**  
Find all employees who earn more than the average salary of their own department.

**Answer:**
```sql
SELECT e.name, e.salary, e.dept_id
FROM employees e
WHERE e.salary > (
  SELECT AVG(salary)
  FROM employees
  WHERE dept_id = e.dept_id
)
ORDER BY e.dept_id;
```

**Sample Output:**

| name  | salary | dept_id |
|-------|--------|---------|
| Alice | 120000 | 10      |
| Dave  | 110000 | 20      |
| Frank | 130000 | 30      |
| Grace | 88000  | 30      |

---

## Q26 — Subquery in FROM (Derived Table / Inline View)

**Question:**  
Find departments where the average salary is above the company-wide average.

**Answer:**
```sql
SELECT dept_id, avg_salary
FROM (
  SELECT dept_id, AVG(salary) AS avg_salary
  FROM employees
  WHERE dept_id IS NOT NULL
  GROUP BY dept_id
) dept_averages
WHERE avg_salary > (SELECT AVG(salary) FROM employees)
ORDER BY avg_salary DESC;
```

**Sample Output:**

| dept_id | avg_salary |
|---------|------------|
| 30      | 98000.00   |
| 10      | 100000.00  |

---

## Q27 — EXISTS vs IN

**Question:**  
Find all departments that have at least one employee earning over $100,000. Write it two ways: using `IN` and using `EXISTS`.

**Answer:**
```sql
-- Using IN
SELECT dept_name
FROM departments
WHERE dept_id IN (
  SELECT dept_id FROM employees WHERE salary > 100000
);

-- Using EXISTS (preferred for large datasets)
SELECT d.dept_name
FROM departments d
WHERE EXISTS (
  SELECT 1
  FROM employees e
  WHERE e.dept_id = d.dept_id
    AND e.salary > 100000
);
```

**Sample Output:**

| dept_name   |
|-------------|
| Engineering |
| Marketing   |
| Sales       |

> **Interview tip:** `EXISTS` stops scanning as soon as one match is found, making it more efficient than `IN` when the subquery result is large.

---

## Q28 — Correlated Subquery

**Question:**  
For each employee, find the number of colleagues in the same department who earn more than them.

**Answer:**
```sql
SELECT
  e.name,
  e.salary,
  e.dept_id,
  (SELECT COUNT(*)
   FROM employees e2
   WHERE e2.dept_id = e.dept_id
     AND e2.salary > e.salary) AS higher_earners
FROM employees e
WHERE e.dept_id IS NOT NULL
ORDER BY e.dept_id, e.salary DESC;
```

**Sample Output:**

| name  | salary | dept_id | higher_earners |
|-------|--------|---------|----------------|
| Alice | 120000 | 10      | 0              |
| Bob   | 95000  | 10      | 1              |
| Carol | 85000  | 10      | 2              |
| Dave  | 110000 | 20      | 0              |
| Eve   | 72000  | 20      | 1              |
| Frank | 130000 | 30      | 0              |
| Grace | 88000  | 30      | 1              |
| Heidi | 76000  | 30      | 2              |

---

## Q29 — Common Table Expressions (CTEs)

**Question:**  
Using a CTE, find the top earner in each department.

**Answer:**
```sql
WITH dept_max AS (
  SELECT dept_id, MAX(salary) AS max_salary
  FROM employees
  WHERE dept_id IS NOT NULL
  GROUP BY dept_id
)
SELECT e.name, e.salary, e.dept_id
FROM employees e
JOIN dept_max dm
  ON e.dept_id = dm.dept_id
 AND e.salary  = dm.max_salary
ORDER BY e.dept_id;
```

**Sample Output:**

| name  | salary | dept_id |
|-------|--------|---------|
| Alice | 120000 | 10      |
| Dave  | 110000 | 20      |
| Frank | 130000 | 30      |

---

## Q30 — Multiple CTEs

**Question:**  
Calculate the department average salary, then find employees who earn above their department average, and classify them by how far above average they are.

**Answer:**
```sql
WITH dept_avg AS (
  SELECT dept_id, AVG(salary) AS avg_sal
  FROM employees
  WHERE dept_id IS NOT NULL
  GROUP BY dept_id
),
above_avg AS (
  SELECT e.name, e.salary, e.dept_id, da.avg_sal,
         e.salary - da.avg_sal AS diff
  FROM employees e
  JOIN dept_avg da ON e.dept_id = da.dept_id
  WHERE e.salary > da.avg_sal
)
SELECT name, salary, dept_id,
       ROUND(diff, 2) AS above_avg_by,
       CASE WHEN diff > 20000 THEN 'Significantly above'
            ELSE 'Moderately above' END AS classification
FROM above_avg
ORDER BY diff DESC;
```

**Sample Output:**

| name  | salary | dept_id | above_avg_by | classification      |
|-------|--------|---------|--------------|---------------------|
| Frank | 130000 | 30      | 32000.00     | Significantly above |
| Alice | 120000 | 10      | 20000.00     | Moderately above    |
| Dave  | 110000 | 20      | 19000.00     | Moderately above    |
| Grace | 88000  | 30      | -10000.00... |                     |

---

# Section 5 — Window Functions

---

## Q31 — ROW_NUMBER

**Question:**  
Assign a sequential row number to each employee within their department, ordered by salary descending.

**Answer:**
```sql
SELECT
  name, dept_id, salary,
  ROW_NUMBER() OVER (
    PARTITION BY dept_id
    ORDER BY salary DESC
  ) AS rn
FROM employees
WHERE dept_id IS NOT NULL
ORDER BY dept_id, rn;
```

**Sample Output:**

| name  | dept_id | salary | rn |
|-------|---------|--------|----|
| Alice | 10      | 120000 | 1  |
| Bob   | 10      | 95000  | 2  |
| Carol | 10      | 85000  | 3  |
| Dave  | 20      | 110000 | 1  |
| Eve   | 20      | 72000  | 2  |
| Frank | 30      | 130000 | 1  |
| Grace | 30      | 88000  | 2  |
| Heidi | 30      | 76000  | 3  |

---

## Q32 — RANK vs DENSE_RANK vs ROW_NUMBER

**Question:**  
Explain the difference between `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` with an example where two employees have the same salary.

**Answer:**
```sql
SELECT
  name, salary,
  ROW_NUMBER()  OVER (ORDER BY salary DESC) AS row_num,
  RANK()        OVER (ORDER BY salary DESC) AS rnk,
  DENSE_RANK()  OVER (ORDER BY salary DESC) AS dense_rnk
FROM employees
ORDER BY salary DESC;
```

**Sample Output:**

| name  | salary | row_num | rnk | dense_rnk |
|-------|--------|---------|-----|-----------|
| Frank | 130000 | 1       | 1   | 1         |
| Alice | 120000 | 2       | 2   | 2         |
| Dave  | 110000 | 3       | 3   | 3         |
| Bob   | 95000  | 4       | 4   | 4         |
| Grace | 88000  | 5       | 5   | 5         |
| Carol | 85000  | 6       | 6   | 6         |
| Heidi | 76000  | 7       | 7   | 7         |
| Eve   | 72000  | 8       | 8   | 8         |
| Ivan  | 60000  | 9       | 9   | 9         |

> **Key differences:**
> - `ROW_NUMBER()` — always unique, no ties possible (1, 2, 3, 4)
> - `RANK()` — ties share a rank, next rank skips (1, 2, 2, 4)
> - `DENSE_RANK()` — ties share a rank, next rank does NOT skip (1, 2, 2, 3)

---

## Q33 — Nth Highest Salary (Classic Interview Question)

**Question:**  
Find the 2nd highest salary in the company. What about the Nth highest?

**Answer:**
```sql
-- 2nd highest using DENSE_RANK
SELECT salary
FROM (
  SELECT salary,
         DENSE_RANK() OVER (ORDER BY salary DESC) AS dr
  FROM employees
) ranked
WHERE dr = 2;

-- Parameterized Nth highest (replace 2 with N)
SELECT DISTINCT salary
FROM (
  SELECT salary,
         DENSE_RANK() OVER (ORDER BY salary DESC) AS dr
  FROM employees
) ranked
WHERE dr = 2;

-- Alternative using subquery
SELECT MAX(salary) AS second_highest
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

**Sample Output:**

| salary |
|--------|
| 120000 |

---

## Q34 — Top N Per Group

**Question:**  
Return the top 2 highest-paid employees from each department.

**Answer:**
```sql
SELECT name, dept_id, salary
FROM (
  SELECT name, dept_id, salary,
         DENSE_RANK() OVER (
           PARTITION BY dept_id
           ORDER BY salary DESC
         ) AS dr
  FROM employees
  WHERE dept_id IS NOT NULL
) ranked
WHERE dr <= 2
ORDER BY dept_id, salary DESC;
```

**Sample Output:**

| name  | dept_id | salary |
|-------|---------|--------|
| Alice | 10      | 120000 |
| Bob   | 10      | 95000  |
| Dave  | 20      | 110000 |
| Eve   | 20      | 72000  |
| Frank | 30      | 130000 |
| Grace | 30      | 88000  |

---

## Q35 — LAG and LEAD

**Question:**  
For each employee ordered by hire date, show the salary of the previous and next hired employee.

**Answer:**
```sql
SELECT
  name,
  hire_date,
  salary,
  LAG(salary)  OVER (ORDER BY hire_date) AS prev_hired_salary,
  LEAD(salary) OVER (ORDER BY hire_date) AS next_hired_salary
FROM employees
ORDER BY hire_date;
```

**Sample Output:**

| name  | hire_date  | salary | prev_hired_salary | next_hired_salary |
|-------|------------|--------|-------------------|-------------------|
| Frank | 2016-05-11 | 130000 | NULL              | 110000            |
| Dave  | 2017-08-20 | 110000 | 130000            | 120000            |
| Alice | 2018-03-01 | 120000 | 110000            | 95000             |
| Bob   | 2019-06-15 | 95000  | 120000            | 60000             |
| Carol | 2020-01-10 | 85000  | 95000             | 88000             |
| Grace | 2020-09-01 | 88000  | 85000             | 72000             |
| Eve   | 2021-03-05 | 72000  | 88000             | 76000             |
| Heidi | 2022-01-15 | 76000  | 72000             | 60000             |
| Ivan  | 2023-07-01 | 60000  | 76000             | NULL              |

---

## Q36 — Running Total

**Question:**  
Show a running total of salary as employees are ordered by hire date (cumulative payroll over time).

**Answer:**
```sql
SELECT
  name,
  hire_date,
  salary,
  SUM(salary) OVER (
    ORDER BY hire_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS cumulative_payroll
FROM employees
ORDER BY hire_date;
```

**Sample Output:**

| name  | hire_date  | salary | cumulative_payroll |
|-------|------------|--------|--------------------|
| Frank | 2016-05-11 | 130000 | 130000             |
| Dave  | 2017-08-20 | 110000 | 240000             |
| Alice | 2018-03-01 | 120000 | 360000             |
| Bob   | 2019-06-15 | 95000  | 455000             |
| Carol | 2020-01-10 | 85000  | 540000             |
| Grace | 2020-09-01 | 88000  | 628000             |
| Eve   | 2021-03-05 | 72000  | 700000             |
| Heidi | 2022-01-15 | 76000  | 776000             |
| Ivan  | 2023-07-01 | 60000  | 836000             |

---

## Q37 — Moving Average

**Question:**  
Calculate a 3-row moving average of salary ordered by hire date.

**Answer:**
```sql
SELECT
  name,
  hire_date,
  salary,
  AVG(salary) OVER (
    ORDER BY hire_date
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS moving_avg_3
FROM employees
ORDER BY hire_date;
```

**Sample Output:**

| name  | salary | moving_avg_3 |
|-------|--------|--------------|
| Frank | 130000 | 130000.00    |
| Dave  | 110000 | 120000.00    |
| Alice | 120000 | 120000.00    |
| Bob   | 95000  | 108333.33    |
| Carol | 85000  | 100000.00    |

---

## Q38 — NTILE

**Question:**  
Divide employees into 3 equal salary buckets (terciles).

**Answer:**
```sql
SELECT
  name,
  salary,
  NTILE(3) OVER (ORDER BY salary DESC) AS salary_tier
FROM employees
ORDER BY salary DESC;
```

**Sample Output:**

| name  | salary | salary_tier |
|-------|--------|-------------|
| Frank | 130000 | 1           |
| Alice | 120000 | 1           |
| Dave  | 110000 | 1           |
| Bob   | 95000  | 2           |
| Grace | 88000  | 2           |
| Carol | 85000  | 2           |
| Heidi | 76000  | 3           |
| Eve   | 72000  | 3           |
| Ivan  | 60000  | 3           |

---

## Q39 — PERCENT_RANK and CUME_DIST

**Question:**  
For each employee, compute their salary percentile rank and cumulative distribution within the company.

**Answer:**
```sql
SELECT
  name,
  salary,
  ROUND(PERCENT_RANK() OVER (ORDER BY salary) * 100, 1) AS pct_rank,
  ROUND(CUME_DIST()    OVER (ORDER BY salary) * 100, 1) AS cume_dist_pct
FROM employees
ORDER BY salary;
```

**Sample Output:**

| name  | salary | pct_rank | cume_dist_pct |
|-------|--------|----------|---------------|
| Ivan  | 60000  | 0.0      | 11.1          |
| Eve   | 72000  | 12.5     | 22.2          |
| Heidi | 76000  | 25.0     | 33.3          |
| Carol | 85000  | 37.5     | 44.4          |
| Grace | 88000  | 50.0     | 55.6          |
| Bob   | 95000  | 62.5     | 66.7          |
| Dave  | 110000 | 75.0     | 77.8          |
| Alice | 120000 | 87.5     | 88.9          |
| Frank | 130000 | 100.0    | 100.0         |

---

# Section 6 — NULL Handling

---

## Q40 — COALESCE and NULLIF

**Question:**  
Explain `COALESCE` and `NULLIF` with practical examples.

**Answer:**
```sql
-- COALESCE: returns the first non-NULL value
SELECT
  name,
  dept_id,
  COALESCE(dept_id, -1) AS dept_id_safe  -- replace NULL with -1
FROM employees;

-- NULLIF: returns NULL if two values are equal, otherwise returns the first
-- Use case: avoid division by zero
SELECT
  name,
  salary,
  10000 / NULLIF(dept_id, 0) AS ratio  -- if dept_id=0, returns NULL not error
FROM employees
WHERE dept_id IS NOT NULL;

-- Combining: replace NULL dept_id label
SELECT
  name,
  COALESCE(d.dept_name, 'Unassigned') AS department
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;
```

**Sample Output (COALESCE dept_name):**

| name  | department  |
|-------|-------------|
| Alice | Engineering |
| Ivan  | Unassigned  |

---

## Q41 — NULL in Aggregations

**Question:**  
How does NULL affect COUNT, SUM, AVG, and comparisons? Demonstrate with examples.

**Answer:**
```sql
-- NULL is ignored by aggregate functions
SELECT
  COUNT(*)       AS total_rows,
  COUNT(dept_id) AS non_null_dept,
  SUM(dept_id)   AS sum_dept,      -- NULL dept_id ignored
  AVG(dept_id)   AS avg_dept       -- NULL dept_id ignored
FROM employees;

-- NULL comparisons always return UNKNOWN (not TRUE or FALSE)
SELECT COUNT(*) FROM employees WHERE dept_id  = NULL;   -- returns 0 (wrong!)
SELECT COUNT(*) FROM employees WHERE dept_id IS NULL;   -- returns 1 (correct)
SELECT COUNT(*) FROM employees WHERE dept_id != NULL;   -- returns 0 (wrong!)
SELECT COUNT(*) FROM employees WHERE dept_id IS NOT NULL; -- returns 8 (correct)
```

**Key outputs:**

| total_rows | non_null_dept | sum_dept | avg_dept |
|------------|---------------|----------|----------|
| 9          | 8             | 170      | 21.25    |

---

# Section 7 — Set Operations

---

## Q42 — UNION vs UNION ALL

**Question:**  
You have two tables: `current_employees` and `former_employees`. Return a combined list of all names. Then explain when to use `UNION` vs `UNION ALL`.

**Answer:**
```sql
-- UNION (removes duplicates — slower)
SELECT name FROM current_employees
UNION
SELECT name FROM former_employees;

-- UNION ALL (keeps duplicates — faster)
SELECT name FROM current_employees
UNION ALL
SELECT name FROM former_employees;
```

> **Interview tip:**
> - `UNION` performs a sort to eliminate duplicates — O(n log n)
> - `UNION ALL` simply concatenates — O(n)
> - Use `UNION ALL` unless you explicitly need de-duplication

---

## Q43 — INTERSECT

**Question:**  
Find names that appear in both the `engineering_team` and `on_call_rotation` tables.

**Answer:**
```sql
SELECT name FROM engineering_team
INTERSECT
SELECT name FROM on_call_rotation;
```

> Returns only rows that exist in **both** result sets. Equivalent to an INNER JOIN on the same column, but cleaner for set logic.

---

## Q44 — EXCEPT / MINUS

**Question:**  
Find all engineering team members who are NOT in the on-call rotation.

**Answer:**
```sql
-- PostgreSQL, SQL Server, DB2
SELECT name FROM engineering_team
EXCEPT
SELECT name FROM on_call_rotation;

-- Oracle
SELECT name FROM engineering_team
MINUS
SELECT name FROM on_call_rotation;
```

> Returns rows from the **first** result set that don't appear in the **second**. Order matters — `A EXCEPT B` ≠ `B EXCEPT A`.

---

# Section 8 — String Functions

---

## Q45 — Common String Functions

**Question:**  
Demonstrate the most commonly tested string functions: `LENGTH`, `UPPER`, `LOWER`, `TRIM`, `SUBSTRING`, `CONCAT`, `REPLACE`, `INSTR/CHARINDEX`.

**Answer:**
```sql
SELECT
  name,
  LENGTH(name)                          AS name_length,
  UPPER(name)                           AS upper_name,
  LOWER(name)                           AS lower_name,
  TRIM('  Alice  ')                     AS trimmed,
  SUBSTRING(name, 1, 3)                 AS first_3_chars,
  CONCAT(name, ' (', job_title, ')')    AS name_and_title,
  REPLACE(job_title, 'VP', 'Director')  AS updated_title
FROM employees
WHERE dept_id = 10;
```

**Sample Output:**

| name  | name_length | upper_name | first_3_chars | name_and_title                  |
|-------|-------------|------------|---------------|---------------------------------|
| Alice | 5           | ALICE      | Ali           | Alice (VP Engineering)          |
| Bob   | 3           | BOB        | Bob           | Bob (Senior Engineer)           |
| Carol | 5           | CAROL      | Car           | Carol (Engineer)                |

---

## Q46 — Extracting Parts of a String

**Question:**  
Given email addresses stored as `'alice@company.com'`, extract the username (before `@`) and domain (after `@`).

**Answer:**
```sql
-- MySQL / PostgreSQL
SELECT
  email,
  SUBSTRING(email, 1, INSTR(email, '@') - 1)     AS username,
  SUBSTRING(email, INSTR(email, '@') + 1)         AS domain
FROM user_emails;

-- SQL Server
SELECT
  email,
  LEFT(email, CHARINDEX('@', email) - 1)          AS username,
  RIGHT(email, LEN(email) - CHARINDEX('@', email)) AS domain
FROM user_emails;
```

**Sample Input:**

| email              |
|--------------------|
| alice@company.com  |
| bob@gmail.com      |

**Sample Output:**

| email             | username | domain      |
|-------------------|----------|-------------|
| alice@company.com | alice    | company.com |
| bob@gmail.com     | bob      | gmail.com   |

---

## Q47 — Counting Character Occurrences

**Question:**  
Count how many times the letter `'e'` appears in each employee's name.

**Answer:**
```sql
SELECT
  name,
  LENGTH(name) - LENGTH(REPLACE(LOWER(name), 'e', '')) AS count_e
FROM employees;
```

**Sample Output:**

| name  | count_e |
|-------|---------|
| Alice | 1       |
| Bob   | 0       |
| Carol | 0       |
| Dave  | 1       |
| Eve   | 2       |
| Grace | 1       |
| Heidi | 1       |

---

# Section 9 — Date Functions

---

## Q48 — Current Date and Common Date Functions

**Question:**  
Show the current date, extract year/month/day from hire dates, and compute each employee's tenure in years.

**Answer:**
```sql
-- MySQL
SELECT
  name,
  hire_date,
  YEAR(hire_date)                              AS hire_year,
  MONTH(hire_date)                             AS hire_month,
  DAY(hire_date)                               AS hire_day,
  DATEDIFF(CURDATE(), hire_date) / 365         AS tenure_years,
  TIMESTAMPDIFF(YEAR, hire_date, CURDATE())    AS tenure_years_exact
FROM employees
ORDER BY hire_date;

-- PostgreSQL
SELECT
  name,
  hire_date,
  EXTRACT(YEAR  FROM hire_date)                AS hire_year,
  EXTRACT(MONTH FROM hire_date)                AS hire_month,
  DATE_PART('year', AGE(CURRENT_DATE, hire_date)) AS tenure_years
FROM employees;
```

**Sample Output:**

| name  | hire_date  | hire_year | hire_month | tenure_years |
|-------|------------|-----------|------------|--------------|
| Frank | 2016-05-11 | 2016      | 5          | 8            |
| Dave  | 2017-08-20 | 2017      | 8          | 7            |
| Alice | 2018-03-01 | 2018      | 3          | 6            |

---

## Q49 — Date Arithmetic

**Question:**  
Find all employees hired in the last 5 years. Also compute their next work anniversary date.

**Answer:**
```sql
-- MySQL
SELECT
  name,
  hire_date,
  DATE_ADD(hire_date, INTERVAL TIMESTAMPDIFF(YEAR, hire_date, CURDATE()) + 1 YEAR)
    AS next_anniversary
FROM employees
WHERE hire_date >= DATE_SUB(CURDATE(), INTERVAL 5 YEAR)
ORDER BY hire_date;

-- PostgreSQL
SELECT
  name,
  hire_date
FROM employees
WHERE hire_date >= CURRENT_DATE - INTERVAL '5 years';
```

**Sample Output** *(assuming current date is 2026-03-20):*

| name  | hire_date  | next_anniversary |
|-------|------------|------------------|
| Eve   | 2021-03-05 | 2027-03-05       |
| Heidi | 2022-01-15 | 2027-01-15       |
| Ivan  | 2023-07-01 | 2027-07-01       |

---

## Q50 — Finding Gaps in Date Sequences

**Question:**  
Given a table of daily order counts, find dates where no orders were placed (gaps in the calendar).

**Answer:**
```sql
-- Generate a calendar CTE, then left-join to find missing dates
WITH RECURSIVE calendar AS (
  SELECT '2024-01-01' AS dt
  UNION ALL
  SELECT DATE_ADD(dt, INTERVAL 1 DAY)
  FROM calendar
  WHERE dt < '2024-01-31'
)
SELECT c.dt AS missing_date
FROM calendar c
LEFT JOIN orders o ON DATE(o.order_date) = c.dt
WHERE o.order_id IS NULL
ORDER BY c.dt;
```

**Sample Output:**

| missing_date |
|--------------|
| 2024-01-03   |
| 2024-01-07   |
| 2024-01-14   |

---

# Section 10 — Advanced SQL Patterns

---

## Q51 — Duplicate Detection and Removal

**Question:**  
Find duplicate employees (same name and department). Then write a query to delete all but one copy.

**Answer:**
```sql
-- Step 1: Find duplicates
SELECT name, dept_id, COUNT(*) AS cnt
FROM employees
GROUP BY name, dept_id
HAVING COUNT(*) > 1;

-- Step 2: Keep only the row with the lowest emp_id (delete the rest)
DELETE FROM employees
WHERE emp_id NOT IN (
  SELECT MIN(emp_id)
  FROM employees
  GROUP BY name, dept_id
);

-- PostgreSQL / CTE-based delete (cleaner)
WITH dupes AS (
  SELECT emp_id,
         ROW_NUMBER() OVER (PARTITION BY name, dept_id ORDER BY emp_id) AS rn
  FROM employees
)
DELETE FROM employees
WHERE emp_id IN (SELECT emp_id FROM dupes WHERE rn > 1);
```

---

## Q52 — Gaps and Islands

**Question:**  
Given a table of consecutive login dates per user, find the "islands" — contiguous sequences of login days — and report each island's start date, end date, and duration.

**Answer:**
```sql
-- Classic gaps-and-islands using row_number difference trick
WITH numbered AS (
  SELECT
    user_id,
    login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM user_logins
),
grouped AS (
  SELECT
    user_id,
    login_date,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp  -- same for consecutive dates
  FROM numbered
)
SELECT
  user_id,
  MIN(login_date)                              AS island_start,
  MAX(login_date)                              AS island_end,
  DATEDIFF(MAX(login_date), MIN(login_date))+1 AS duration_days
FROM grouped
GROUP BY user_id, grp
ORDER BY user_id, island_start;
```

**Sample Input:**

| user_id | login_date |
|---------|------------|
| 1       | 2024-01-01 |
| 1       | 2024-01-02 |
| 1       | 2024-01-03 |
| 1       | 2024-01-07 |
| 1       | 2024-01-08 |

**Sample Output:**

| user_id | island_start | island_end | duration_days |
|---------|--------------|------------|---------------|
| 1       | 2024-01-01   | 2024-01-03 | 3             |
| 1       | 2024-01-07   | 2024-01-08 | 2             |

---

## Q53 — Recursive CTEs (Hierarchical Data)

**Question:**  
Using the manager–employee hierarchy in the `employees` table, list the entire reporting chain under Alice (emp_id = 1), showing each person's depth in the hierarchy.

**Answer:**
```sql
WITH RECURSIVE org_chart AS (
  -- Anchor: start with Alice
  SELECT emp_id, name, manager_id, 0 AS depth
  FROM employees
  WHERE emp_id = 1

  UNION ALL

  -- Recursive: find direct reports of each found employee
  SELECT e.emp_id, e.name, e.manager_id, oc.depth + 1
  FROM employees e
  JOIN org_chart oc ON e.manager_id = oc.emp_id
)
SELECT
  LPAD(' ', depth * 4, ' ') || name AS org_tree,
  depth
FROM org_chart
ORDER BY depth, name;
```

**Sample Output:**

| org_tree        | depth |
|-----------------|-------|
| Alice           | 0     |
|     Bob         | 1     |
|     Carol       | 1     |

---

## Q54 — Pivot / Cross-Tab

**Question:**  
Show the count of employees in each department broken out by salary band (Entry / Mid / Senior) as columns.

**Answer:**
```sql
SELECT
  d.dept_name,
  SUM(CASE WHEN e.salary < 75000  THEN 1 ELSE 0 END) AS entry_count,
  SUM(CASE WHEN e.salary BETWEEN 75000 AND 100000 THEN 1 ELSE 0 END) AS mid_count,
  SUM(CASE WHEN e.salary > 100000 THEN 1 ELSE 0 END) AS senior_count
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name
ORDER BY d.dept_name;
```

**Sample Output:**

| dept_name   | entry_count | mid_count | senior_count |
|-------------|-------------|-----------|--------------|
| Engineering | 0           | 2         | 1            |
| Marketing   | 1           | 0         | 1            |
| Sales       | 0           | 2         | 1            |

---

## Q55 — Running Percentage of Total

**Question:**  
For each employee sorted by salary, show their salary, the running total, and what percentage of the total payroll that running total represents.

**Answer:**
```sql
SELECT
  name,
  salary,
  SUM(salary) OVER (ORDER BY salary
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
  ROUND(
    100.0 * SUM(salary) OVER (ORDER BY salary
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    / SUM(salary) OVER (), 1
  ) AS running_pct
FROM employees
ORDER BY salary;
```

**Sample Output:**

| name  | salary | running_total | running_pct |
|-------|--------|---------------|-------------|
| Ivan  | 60000  | 60000         | 7.2         |
| Eve   | 72000  | 132000        | 15.8        |
| Heidi | 76000  | 208000        | 24.9        |
| Carol | 85000  | 293000        | 35.1        |
| Grace | 88000  | 381000        | 45.6        |
| Bob   | 95000  | 476000        | 57.0        |
| Dave  | 110000 | 586000        | 70.1        |
| Alice | 120000 | 706000        | 84.5        |
| Frank | 130000 | 836000        | 100.0       |

---

## Q56 — Sessionization

**Question:**  
Group user events into sessions, where a new session starts if more than 30 minutes have passed since the last event.

**Answer:**
```sql
WITH lagged AS (
  SELECT
    user_id,
    event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event
  FROM user_events
),
session_flags AS (
  SELECT
    user_id,
    event_time,
    CASE
      WHEN prev_event IS NULL
        OR TIMESTAMPDIFF(MINUTE, prev_event, event_time) > 30
      THEN 1 ELSE 0
    END AS new_session
  FROM lagged
),
sessions AS (
  SELECT
    user_id,
    event_time,
    SUM(new_session) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
  FROM session_flags
)
SELECT user_id, session_id,
       MIN(event_time) AS session_start,
       MAX(event_time) AS session_end,
       COUNT(*)        AS event_count
FROM sessions
GROUP BY user_id, session_id
ORDER BY user_id, session_id;
```

---

# Section 11 — Schema Design and Normalization

---

## Q57 — First, Second, Third Normal Form

**Question:**  
Explain 1NF, 2NF, and 3NF with examples of violations and how to fix them.

**Answer:**

**1NF (First Normal Form):** Each column must contain atomic (indivisible) values; no repeating groups.

```sql
-- VIOLATES 1NF (multiple values in one column)
| emp_id | name  | skills              |
|--------|-------|---------------------|
| 1      | Alice | Python, SQL, Java   |

-- FIXED (separate table)
| emp_id | skill  |
|--------|--------|
| 1      | Python |
| 1      | SQL    |
| 1      | Java   |
```

**2NF (Second Normal Form):** Must be in 1NF + every non-key attribute must depend on the ENTIRE primary key (no partial dependencies).

```sql
-- VIOLATES 2NF (dept_name depends only on dept_id, not the full PK)
| emp_id | dept_id | dept_name   | salary |
-- dept_name is a partial dependency on dept_id alone

-- FIXED: split into two tables
employees(emp_id, dept_id, salary)
departments(dept_id, dept_name)
```

**3NF (Third Normal Form):** Must be in 2NF + no transitive dependencies (non-key column depending on another non-key column).

```sql
-- VIOLATES 3NF (zip_code → city is a transitive dependency)
| emp_id | zip_code | city     |

-- FIXED: split out the transitive dependency
employees(emp_id, zip_code)
zip_codes(zip_code, city)
```

---

## Q58 — Primary Keys, Foreign Keys, and Constraints

**Question:**  
What are the key database constraints? Give DDL examples for each.

**Answer:**
```sql
CREATE TABLE departments (
  dept_id   INT         PRIMARY KEY,        -- uniquely identifies each row
  dept_name VARCHAR(50) NOT NULL,           -- cannot be NULL
  location  VARCHAR(50) UNIQUE              -- no two depts in same location
);

CREATE TABLE employees (
  emp_id     INT          PRIMARY KEY,
  name       VARCHAR(50)  NOT NULL,
  dept_id    INT          REFERENCES departments(dept_id),  -- FK
  salary     DECIMAL(10,2) CHECK (salary > 0),              -- must be positive
  hire_date  DATE         DEFAULT CURRENT_DATE              -- auto-populated
);
```

| Constraint  | Purpose                                    |
|-------------|--------------------------------------------|
| PRIMARY KEY | Uniquely identifies each row; no NULLs     |
| FOREIGN KEY | Enforces referential integrity between tables |
| NOT NULL    | Column must have a value                   |
| UNIQUE      | No duplicate values (NULLs may be allowed) |
| CHECK       | Value must satisfy a boolean expression    |
| DEFAULT     | Auto-fill value when none is provided      |

---

## Q59 — Indexes: When and Why

**Question:**  
When should you add an index? What are the tradeoffs?

**Answer:**

```sql
-- Create a basic index on a frequently filtered column
CREATE INDEX idx_employees_dept_id ON employees(dept_id);

-- Composite index (column order matters — leftmost prefix rule)
CREATE INDEX idx_emp_dept_salary ON employees(dept_id, salary);

-- This query CAN use idx_emp_dept_salary (uses leftmost prefix):
SELECT * FROM employees WHERE dept_id = 10 AND salary > 90000;

-- This query CANNOT efficiently use it (skips dept_id):
SELECT * FROM employees WHERE salary > 90000;
```

| Scenario | Add Index? |
|----------|-----------|
| Column appears in WHERE / JOIN / ORDER BY frequently | ✅ Yes |
| Table is read-heavy (OLAP) | ✅ Yes |
| Column has high cardinality (many distinct values) | ✅ Yes |
| Table is write-heavy (many INSERTs/UPDATEs) | ⚠️ Caution — indexes slow writes |
| Very small table (< 1,000 rows) | ❌ No — full scan is faster |
| Low-cardinality column (e.g., boolean) | ❌ Usually not worth it |

---

# Section 12 — Classic Interview Questions

---

## Q60 — Employees Earning More Than Their Manager

**Question:**  
Find all employees who earn more than their direct manager.

**Answer:**
```sql
SELECT e.name AS employee, e.salary AS emp_salary,
       m.name AS manager,  m.salary AS mgr_salary
FROM employees e
JOIN employees m ON e.manager_id = m.emp_id
WHERE e.salary > m.salary;
```

**Sample Output:** *(none in reference data — Bob: 95k < Alice: 120k)*  
*(Would show results if, say, Grace earned more than Frank)*

---

## Q61 — Department with the Highest Total Salary

**Question:**  
Which department has the highest total salary bill?

**Answer:**
```sql
SELECT d.dept_name, SUM(e.salary) AS total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name
ORDER BY total_salary DESC
LIMIT 1;

-- Handles ties (returns all tied departments)
SELECT d.dept_name, SUM(e.salary) AS total_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name
HAVING SUM(e.salary) = (
  SELECT MAX(total)
  FROM (
    SELECT SUM(salary) AS total
    FROM employees
    WHERE dept_id IS NOT NULL
    GROUP BY dept_id
  ) t
);
```

**Sample Output:**

| dept_name   | total_salary |
|-------------|--------------|
| Engineering | 300000       |

---

## Q62 — Consecutive Login Days (Streak)

**Question:**  
Find the longest consecutive login streak for each user.

**Answer:**
```sql
WITH numbered AS (
  SELECT user_id, login_date,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
grp AS (
  SELECT user_id, login_date,
         DATE_SUB(login_date, INTERVAL rn DAY) AS grp_key
  FROM numbered
),
streaks AS (
  SELECT user_id, grp_key,
         COUNT(*) AS streak_length
  FROM grp
  GROUP BY user_id, grp_key
)
SELECT user_id, MAX(streak_length) AS longest_streak
FROM streaks
GROUP BY user_id
ORDER BY longest_streak DESC;
```

**Sample Output:**

| user_id | longest_streak |
|---------|----------------|
| 3       | 14             |
| 1       | 7              |
| 2       | 3              |

---

## Q63 — Month-over-Month Growth Rate

**Question:**  
Calculate the month-over-month revenue growth rate from an `orders` table.

**Answer:**
```sql
WITH monthly AS (
  SELECT
    DATE_FORMAT(order_date, '%Y-%m')      AS month,
    SUM(amount)                           AS revenue
  FROM orders
  GROUP BY DATE_FORMAT(order_date, '%Y-%m')
),
with_lag AS (
  SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_revenue
  FROM monthly
)
SELECT
  month,
  revenue,
  prev_revenue,
  ROUND(100.0 * (revenue - prev_revenue) / prev_revenue, 2) AS mom_growth_pct
FROM with_lag
ORDER BY month;
```

**Sample Output:**

| month   | revenue | prev_revenue | mom_growth_pct |
|---------|---------|--------------|----------------|
| 2024-01 | 50000   | NULL         | NULL           |
| 2024-02 | 62000   | 50000        | 24.00          |
| 2024-03 | 58000   | 62000        | -6.45          |
| 2024-04 | 71000   | 58000        | 22.41          |

---

## Q64 — Median Salary (No Built-in Function)

**Question:**  
Calculate the median salary without using a built-in `MEDIAN()` function.

**Answer:**
```sql
-- Works in MySQL, PostgreSQL, SQL Server
SELECT AVG(salary) AS median_salary
FROM (
  SELECT salary,
         ROW_NUMBER() OVER (ORDER BY salary)       AS rn,
         COUNT(*) OVER ()                           AS total
  FROM employees
) t
WHERE rn IN (FLOOR((total + 1) / 2.0), CEIL((total + 1) / 2.0));
```

**Sample Output:**

| median_salary |
|---------------|
| 88000.00      |

---

## Q65 — Users Active in Both Period A and Period B

**Question:**  
Find users who placed orders in both January 2024 and February 2024.

**Answer:**
```sql
-- Using INTERSECT
SELECT DISTINCT customer_id FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31'
INTERSECT
SELECT DISTINCT customer_id FROM orders WHERE order_date BETWEEN '2024-02-01' AND '2024-02-29';

-- Using EXISTS (works in MySQL which lacks INTERSECT)
SELECT DISTINCT customer_id
FROM orders o1
WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31'
  AND EXISTS (
    SELECT 1 FROM orders o2
    WHERE o2.customer_id = o1.customer_id
      AND o2.order_date BETWEEN '2024-02-01' AND '2024-02-29'
  );

-- Using GROUP BY + HAVING
SELECT customer_id
FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-02-29'
GROUP BY customer_id
HAVING SUM(CASE WHEN MONTH(order_date) = 1 THEN 1 ELSE 0 END) > 0
   AND SUM(CASE WHEN MONTH(order_date) = 2 THEN 1 ELSE 0 END) > 0;
```

---

## Q66 — Retention Rate

**Question:**  
Calculate the 1-month retention rate: what percentage of users who signed up in January 2024 also placed an order in February 2024?

**Answer:**
```sql
WITH jan_users AS (
  SELECT DISTINCT customer_id
  FROM orders
  WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31'
),
feb_users AS (
  SELECT DISTINCT customer_id
  FROM orders
  WHERE order_date BETWEEN '2024-02-01' AND '2024-02-29'
)
SELECT
  COUNT(DISTINCT ju.customer_id)                    AS jan_cohort_size,
  COUNT(DISTINCT fu.customer_id)                    AS retained_in_feb,
  ROUND(100.0 * COUNT(DISTINCT fu.customer_id)
        / COUNT(DISTINCT ju.customer_id), 1)        AS retention_rate_pct
FROM jan_users ju
LEFT JOIN feb_users fu ON ju.customer_id = fu.customer_id;
```

**Sample Output:**

| jan_cohort_size | retained_in_feb | retention_rate_pct |
|-----------------|-----------------|--------------------|
| 200             | 134             | 67.0               |

---

## Q67 — Cumulative Distinct Users

**Question:**  
For each day, how many total unique users have ever placed an order up to and including that date?

**Answer:**
```sql
WITH daily_new_users AS (
  SELECT
    DATE(order_date) AS order_day,
    customer_id,
    MIN(DATE(order_date)) OVER (PARTITION BY customer_id) AS first_order_date
  FROM orders
),
first_appearances AS (
  SELECT DISTINCT first_order_date AS dt, customer_id
  FROM daily_new_users
)
SELECT
  dt,
  COUNT(*) OVER (ORDER BY dt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    AS cumulative_unique_users
FROM (
  SELECT dt, COUNT(*) AS new_users
  FROM first_appearances
  GROUP BY dt
) t
ORDER BY dt;
```

---

## Q68 — Product Funnel Analysis

**Question:**  
Given a `user_events` table with event types (`view`, `cart`, `purchase`), compute a conversion funnel showing the count and drop-off at each stage.

**Answer:**
```sql
SELECT
  SUM(CASE WHEN event_type = 'view'     THEN 1 ELSE 0 END) AS views,
  SUM(CASE WHEN event_type = 'cart'     THEN 1 ELSE 0 END) AS carts,
  SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchases,
  ROUND(100.0 *
    SUM(CASE WHEN event_type = 'cart'     THEN 1 ELSE 0 END) /
    NULLIF(SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END), 0),
    1) AS view_to_cart_pct,
  ROUND(100.0 *
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) /
    NULLIF(SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END), 0),
    1) AS cart_to_purchase_pct
FROM user_events;
```

**Sample Output:**

| views | carts | purchases | view_to_cart_pct | cart_to_purchase_pct |
|-------|-------|-----------|------------------|----------------------|
| 10000 | 3500  | 980       | 35.0             | 28.0                 |

---

# Section 13 — Performance and Optimization

---

## Q69 — Query Optimization Checklist

**Question:**  
What are the most impactful ways to optimize a slow SQL query?

**Answer:**

**1. Use indexes on WHERE / JOIN / ORDER BY columns**
```sql
-- Slow (full table scan)
SELECT * FROM orders WHERE YEAR(order_date) = 2024;

-- Fast (index on order_date can be used)
SELECT * FROM orders
WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';
```

**2. Avoid SELECT \* — fetch only needed columns**
```sql
-- Bad: fetches all columns including large BLOBs
SELECT * FROM products WHERE category = 'Electronics';

-- Good
SELECT product_id, name, price FROM products WHERE category = 'Electronics';
```

**3. Filter early — push WHERE into subqueries**
```sql
-- Bad: joins first, then filters
SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE c.country = 'US';

-- Good: filter first, then join
SELECT * FROM orders o
JOIN (SELECT * FROM customers WHERE country = 'US') c
  ON o.customer_id = c.customer_id;
```

**4. Avoid functions on indexed columns in WHERE**
```sql
-- Bad: index on hire_date cannot be used
WHERE YEAR(hire_date) = 2020

-- Good: range query uses index
WHERE hire_date BETWEEN '2020-01-01' AND '2020-12-31'
```

**5. Use EXISTS instead of IN for large subqueries**
```sql
-- Potentially slow for large subquery
WHERE dept_id IN (SELECT dept_id FROM departments WHERE location = 'NYC')

-- More efficient
WHERE EXISTS (SELECT 1 FROM departments d
              WHERE d.dept_id = e.dept_id AND d.location = 'NYC')
```

---

## Q70 — EXPLAIN / Query Execution Plans

**Question:**  
How do you analyze query performance using EXPLAIN?

**Answer:**
```sql
-- MySQL
EXPLAIN SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE e.salary > 90000;

-- PostgreSQL
EXPLAIN ANALYZE SELECT ...;

-- SQL Server
SET STATISTICS IO ON;
-- or use graphical execution plan in SSMS
```

**Key columns to look at in EXPLAIN output:**

| Column      | What to look for                                        |
|-------------|---------------------------------------------------------|
| `type`      | `ALL` = full scan (bad), `ref`/`eq_ref` = index (good) |
| `rows`      | Estimated rows scanned — lower is better                |
| `key`       | Which index is being used (NULL = no index)             |
| `Extra`     | `Using filesort` / `Using temporary` = potential issues |

---

# Section 14 — Quick Reference: Key Differences

---

## Comparison Table — Window Function Ranking

| Function     | Ties | Gap after tie | Example output |
|--------------|------|---------------|----------------|
| ROW_NUMBER() | No   | N/A           | 1, 2, 3, 4     |
| RANK()       | Yes  | Yes           | 1, 2, 2, 4     |
| DENSE_RANK() | Yes  | No            | 1, 2, 2, 3     |
| NTILE(n)     | N/A  | N/A           | Bucket numbers |

## Comparison Table — JOIN Types

| JOIN Type        | Includes unmatched rows from... |
|------------------|---------------------------------|
| INNER JOIN       | Neither table                   |
| LEFT JOIN        | Left table only                 |
| RIGHT JOIN       | Right table only                |
| FULL OUTER JOIN  | Both tables                     |
| CROSS JOIN       | N/A — Cartesian product         |

## Comparison Table — WHERE vs HAVING

|              | WHERE              | HAVING             |
|--------------|--------------------|--------------------|
| Filters      | Individual rows    | Grouped results    |
| Timing       | Before GROUP BY    | After GROUP BY     |
| Can use aggregates? | No          | Yes                |

## Comparison Table — DELETE vs TRUNCATE vs DROP

| Command  | Removes rows | Keeps structure | Can rollback | Resets identity |
|----------|-------------|-----------------|--------------|-----------------|
| DELETE   | ✅ (with WHERE) | ✅           | ✅           | ❌              |
| TRUNCATE | ✅ (all rows)   | ✅           | ⚠️ Usually  | ✅              |
| DROP     | ✅ (all rows)   | ❌ (table gone) | ❌        | N/A             |

---

*End of Study Guide — Grokking the SQL Interview*  
*Practice each query on a live database. The best way to retain SQL is to run it.*
