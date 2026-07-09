# SQL FAANG Interview Questions — 50 Problems with Sample Data & Solutions

> Covers: Window Functions, CTEs, Joins, Aggregations, Subqueries, String ops, Date logic, Gap-and-island problems, and more.

---

## TABLE OF CONTENTS

| # | Topic | Difficulty |
|---|-------|------------|
| 1–10 | Core Aggregations & Grouping | 🟢 Easy |
| 11–20 | Joins & Subqueries | 🟡 Medium |
| 21–35 | Window Functions | 🟡 Medium – 🔴 Hard |
| 36–45 | CTEs, Recursion & Advanced Logic | 🔴 Hard |
| 46–50 | Real FAANG Classics | 🔴 Hard |

---

# SECTION 1 — Core Aggregations & Grouping (Easy)

---

## Q1. Second Highest Salary

**Table: `employees`**

| emp_id | name    | salary |
|--------|---------|--------|
| 1      | Alice   | 90000  |
| 2      | Bob     | 75000  |
| 3      | Carol   | 90000  |
| 4      | Dave    | 60000  |

**Expected Output:**

| second_highest_salary |
|-----------------------|
| 75000                 |

```sql
SELECT MAX(salary) AS second_highest_salary
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

**Approach:** Exclude the global max salary, then take the max of what remains. Handles duplicates at the top naturally.

---

## Q2. Department with Highest Average Salary

**Table: `employees`**

| emp_id | name  | dept    | salary |
|--------|-------|---------|--------|
| 1      | Alice | Eng     | 100000 |
| 2      | Bob   | Eng     | 90000  |
| 3      | Carol | Sales   | 60000  |
| 4      | Dave  | Sales   | 70000  |
| 5      | Eve   | HR      | 50000  |

**Expected Output:**

| dept | avg_salary |
|------|------------|
| Eng  | 95000      |

```sql
SELECT dept, AVG(salary) AS avg_salary
FROM employees
GROUP BY dept
ORDER BY avg_salary DESC
LIMIT 1;
```

---

## Q3. Count of Employees per Department

**Same `employees` table as Q2.**

**Expected Output:**

| dept  | emp_count |
|-------|-----------|
| Eng   | 2         |
| Sales | 2         |
| HR    | 1         |

```sql
SELECT dept, COUNT(*) AS emp_count
FROM employees
GROUP BY dept
ORDER BY emp_count DESC;
```

---

## Q4. Find Duplicate Emails

**Table: `users`**

| id | email             |
|----|-------------------|
| 1  | alice@example.com |
| 2  | bob@example.com   |
| 3  | alice@example.com |
| 4  | carol@example.com |

**Expected Output:**

| email             |
|-------------------|
| alice@example.com |

```sql
SELECT email
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

---

## Q5. Employees Earning More than Their Manager

**Table: `employees`**

| emp_id | name  | salary | manager_id |
|--------|-------|--------|------------|
| 1      | Alice | 90000  | NULL       |
| 2      | Bob   | 95000  | 1          |
| 3      | Carol | 80000  | 1          |
| 4      | Dave  | 70000  | 2          |

**Expected Output:**

| name |
|------|
| Bob  |

```sql
SELECT e.name
FROM employees e
JOIN employees m ON e.manager_id = m.emp_id
WHERE e.salary > m.salary;
```

---

## Q6. Find Customers Who Never Ordered

**Tables: `customers`, `orders`**

| customer_id | name  |
|-------------|-------|
| 1           | Alice |
| 2           | Bob   |
| 3           | Carol |

| order_id | customer_id | amount |
|----------|-------------|--------|
| 101      | 1           | 200    |
| 102      | 1           | 300    |
| 103      | 2           | 150    |

**Expected Output:**

| name  |
|-------|
| Carol |

```sql
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

**Approach:** LEFT JOIN keeps all customers; filtering on NULL in the right table keeps only those with no match.

---

## Q7. Total Revenue per Product per Month

**Table: `sales`**

| sale_id | product | sale_date  | amount |
|---------|---------|------------|--------|
| 1       | A       | 2024-01-05 | 100    |
| 2       | A       | 2024-01-20 | 200    |
| 3       | B       | 2024-01-15 | 150    |
| 4       | A       | 2024-02-10 | 300    |

**Expected Output:**

| product | month   | total_revenue |
|---------|---------|---------------|
| A       | 2024-01 | 300           |
| B       | 2024-01 | 150           |
| A       | 2024-02 | 300           |

```sql
SELECT
    product,
    DATE_FORMAT(sale_date, '%Y-%m') AS month,
    SUM(amount) AS total_revenue
FROM sales
GROUP BY product, DATE_FORMAT(sale_date, '%Y-%m')
ORDER BY month, product;
```

---

## Q8. Top 3 Products by Total Sales

**Same `sales` table as Q7.**

**Expected Output:**

| product | total_revenue |
|---------|---------------|
| A       | 600           |
| B       | 150           |

```sql
SELECT product, SUM(amount) AS total_revenue
FROM sales
GROUP BY product
ORDER BY total_revenue DESC
LIMIT 3;
```

---

## Q9. Employees Hired in the Last 30 Days

**Table: `employees`**

| emp_id | name  | hire_date  |
|--------|-------|------------|
| 1      | Alice | 2024-05-01 |
| 2      | Bob   | 2024-04-01 |
| 3      | Carol | 2024-03-01 |

*(Assume today is 2024-05-10)*

**Expected Output:**

| name  | hire_date  |
|-------|------------|
| Alice | 2024-05-01 |

```sql
SELECT name, hire_date
FROM employees
WHERE hire_date >= CURDATE() - INTERVAL 30 DAY;
```

---

## Q10. Department Headcount and Salary Budget

**Same `employees` table as Q2.**

**Expected Output:**

| dept  | headcount | salary_budget |
|-------|-----------|---------------|
| Eng   | 2         | 190000        |
| Sales | 2         | 130000        |
| HR    | 1         | 50000         |

```sql
SELECT
    dept,
    COUNT(*) AS headcount,
    SUM(salary) AS salary_budget
FROM employees
GROUP BY dept
ORDER BY salary_budget DESC;
```

---

# SECTION 2 — Joins & Subqueries (Medium)

---

## Q11. Highest Paid Employee per Department

**Same `employees` table as Q2.**

**Expected Output:**

| dept  | name  | salary |
|-------|-------|--------|
| Eng   | Alice | 100000 |
| HR    | Eve   | 50000  |
| Sales | Dave  | 70000  |

```sql
SELECT dept, name, salary
FROM (
    SELECT dept, name, salary,
           RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk
    FROM employees
) ranked
WHERE rnk = 1;
```

**Approach:** Use `RANK()` instead of `ROW_NUMBER()` to handle ties — if two people in the same dept have the same top salary, both are returned.

---

## Q12. Products Never Sold

**Tables: `products`, `order_items`**

| product_id | product_name |
|------------|--------------|
| 1          | Widget       |
| 2          | Gadget       |
| 3          | Doohickey    |

| order_id | product_id | qty |
|----------|------------|-----|
| 1        | 1          | 5   |
| 2        | 2          | 3   |

**Expected Output:**

| product_name |
|--------------|
| Doohickey    |

```sql
SELECT p.product_name
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.product_id IS NULL;
```

---

## Q13. Self Join — Find Pairs of Employees in the Same Dept

**Same `employees` table as Q2.**

**Expected Output:**

| emp1  | emp2  | dept  |
|-------|-------|-------|
| Alice | Bob   | Eng   |
| Carol | Dave  | Sales |

```sql
SELECT e1.name AS emp1, e2.name AS emp2, e1.dept
FROM employees e1
JOIN employees e2
  ON e1.dept = e2.dept AND e1.emp_id < e2.emp_id
ORDER BY e1.dept;
```

**Approach:** `e1.emp_id < e2.emp_id` prevents duplicates like (Bob, Alice) and self-pairs.

---

## Q14. Orders with No Returns

**Tables: `orders`, `returns`**

| order_id | customer_id | amount |
|----------|-------------|--------|
| 1        | 101         | 200    |
| 2        | 102         | 300    |
| 3        | 103         | 150    |

| return_id | order_id |
|-----------|----------|
| 1         | 2        |

**Expected Output:**

| order_id | amount |
|----------|--------|
| 1        | 200    |
| 3        | 150    |

```sql
SELECT o.order_id, o.amount
FROM orders o
WHERE o.order_id NOT IN (SELECT order_id FROM returns);
```

*Alternative (safer with NULLs):*
```sql
SELECT o.order_id, o.amount
FROM orders o
LEFT JOIN returns r ON o.order_id = r.order_id
WHERE r.return_id IS NULL;
```

---

## Q15. Employees with Above-Average Salary in Their Department

**Same `employees` table as Q2.**

**Expected Output:**

| name  | dept  | salary |
|-------|-------|--------|
| Alice | Eng   | 100000 |
| Dave  | Sales | 70000  |
| Eve   | HR    | 50000  |

```sql
SELECT e.name, e.dept, e.salary
FROM employees e
JOIN (
    SELECT dept, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept
) dept_avg ON e.dept = dept_avg.dept
WHERE e.salary > dept_avg.avg_sal;
```

---

## Q16. Cumulative Revenue over Time

**Table: `daily_sales`**

| sale_date  | revenue |
|------------|---------|
| 2024-01-01 | 100     |
| 2024-01-02 | 200     |
| 2024-01-03 | 150     |
| 2024-01-04 | 300     |

**Expected Output:**

| sale_date  | revenue | cumulative_revenue |
|------------|---------|--------------------|
| 2024-01-01 | 100     | 100                |
| 2024-01-02 | 200     | 300                |
| 2024-01-03 | 150     | 450                |
| 2024-01-04 | 300     | 750                |

```sql
SELECT
    sale_date,
    revenue,
    SUM(revenue) OVER (ORDER BY sale_date) AS cumulative_revenue
FROM daily_sales;
```

---

## Q17. Nth Highest Salary (Generalised)

Find the **3rd highest** distinct salary.

**Same `employees` table as Q1.**

```sql
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 2;   -- OFFSET = N-1
```

*Or using a correlated subquery:*
```sql
SELECT DISTINCT salary
FROM employees e1
WHERE 2 = (
    SELECT COUNT(DISTINCT salary)
    FROM employees e2
    WHERE e2.salary > e1.salary
);
```

---

## Q18. Users Active on Consecutive Days

**Table: `user_activity`**

| user_id | activity_date |
|---------|---------------|
| 1       | 2024-01-01    |
| 1       | 2024-01-02    |
| 1       | 2024-01-04    |
| 2       | 2024-01-01    |
| 2       | 2024-01-02    |
| 2       | 2024-01-03    |

**Expected Output (users with at least 2 consecutive days):**

| user_id |
|---------|
| 1       |
| 2       |

```sql
SELECT DISTINCT a.user_id
FROM user_activity a
JOIN user_activity b
  ON a.user_id = b.user_id
 AND DATEDIFF(b.activity_date, a.activity_date) = 1;
```

---

## Q19. Customers with Orders in Both Jan and Feb

**Table: `orders`**

| order_id | customer_id | order_date |
|----------|-------------|------------|
| 1        | 101         | 2024-01-10 |
| 2        | 101         | 2024-02-15 |
| 3        | 102         | 2024-01-20 |
| 4        | 103         | 2024-02-05 |

**Expected Output:**

| customer_id |
|-------------|
| 101         |

```sql
SELECT customer_id
FROM orders
WHERE MONTH(order_date) IN (1, 2) AND YEAR(order_date) = 2024
GROUP BY customer_id
HAVING COUNT(DISTINCT MONTH(order_date)) = 2;
```

---

## Q20. Rank Products by Revenue within Each Category

**Table: `product_sales`**

| product  | category | revenue |
|----------|----------|---------|
| A        | Tech     | 5000    |
| B        | Tech     | 8000    |
| C        | Home     | 3000    |
| D        | Home     | 4500    |
| E        | Tech     | 8000    |

**Expected Output:**

| product | category | revenue | rank |
|---------|----------|---------|------|
| B       | Tech     | 8000    | 1    |
| E       | Tech     | 8000    | 1    |
| A       | Tech     | 5000    | 3    |
| D       | Home     | 4500    | 1    |
| C       | Home     | 3000    | 2    |

```sql
SELECT
    product,
    category,
    revenue,
    RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS rank
FROM product_sales;
```

---

# SECTION 3 — Window Functions (Medium–Hard)

---

## Q21. Running Total with PARTITION

**Table: `transactions`**

| txn_id | user_id | txn_date   | amount |
|--------|---------|------------|--------|
| 1      | A       | 2024-01-01 | 100    |
| 2      | A       | 2024-01-03 | 200    |
| 3      | B       | 2024-01-02 | 50     |
| 4      | B       | 2024-01-05 | 150    |

**Expected Output:**

| user_id | txn_date   | amount | running_total |
|---------|------------|--------|---------------|
| A       | 2024-01-01 | 100    | 100           |
| A       | 2024-01-03 | 200    | 300           |
| B       | 2024-01-02 | 50     | 50            |
| B       | 2024-01-05 | 150    | 200           |

```sql
SELECT
    user_id,
    txn_date,
    amount,
    SUM(amount) OVER (PARTITION BY user_id ORDER BY txn_date) AS running_total
FROM transactions;
```

---

## Q22. Month-over-Month Revenue Growth

**Table: `monthly_revenue`**

| month   | revenue |
|---------|---------|
| 2024-01 | 10000   |
| 2024-02 | 12000   |
| 2024-03 | 11000   |
| 2024-04 | 15000   |

**Expected Output:**

| month   | revenue | prev_revenue | growth_pct |
|---------|---------|--------------|------------|
| 2024-01 | 10000   | NULL         | NULL       |
| 2024-02 | 12000   | 10000        | 20.00      |
| 2024-03 | 11000   | 12000        | -8.33      |
| 2024-04 | 15000   | 11000        | 36.36      |

```sql
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
              / LAG(revenue) OVER (ORDER BY month),
        2
    ) AS growth_pct
FROM monthly_revenue;
```

**Approach:** `LAG()` fetches the previous row's value. Divide the delta by the prior value for percentage growth.

---

## Q23. Moving 3-Day Average

**Table: `daily_sales`** (same as Q16)

**Expected Output:**

| sale_date  | revenue | moving_avg_3d |
|------------|---------|---------------|
| 2024-01-01 | 100     | 100.00        |
| 2024-01-02 | 200     | 150.00        |
| 2024-01-03 | 150     | 150.00        |
| 2024-01-04 | 300     | 216.67        |

```sql
SELECT
    sale_date,
    revenue,
    ROUND(
        AVG(revenue) OVER (
            ORDER BY sale_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2
    ) AS moving_avg_3d
FROM daily_sales;
```

**Approach:** `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` defines a sliding 3-row window.

---

## Q24. Dense Rank vs Rank vs Row Number

**Table: `scores`**

| player | score |
|--------|-------|
| A      | 90    |
| B      | 85    |
| C      | 90    |
| D      | 80    |

**Expected Output:**

| player | score | row_num | rank | dense_rank |
|--------|-------|---------|------|------------|
| A      | 90    | 1       | 1    | 1          |
| C      | 90    | 2       | 1    | 1          |
| B      | 85    | 3       | 3    | 2          |
| D      | 80    | 4       | 4    | 3          |

```sql
SELECT
    player,
    score,
    ROW_NUMBER() OVER (ORDER BY score DESC) AS row_num,
    RANK()       OVER (ORDER BY score DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM scores;
```

**Key Difference:**
- `ROW_NUMBER` — always unique, sequential
- `RANK` — ties get the same rank, next rank skips (1,1,3)
- `DENSE_RANK` — ties get the same rank, next rank doesn't skip (1,1,2)

---

## Q25. First and Last Purchase per Customer

**Table: `orders`**

| order_id | customer_id | order_date | amount |
|----------|-------------|------------|--------|
| 1        | 101         | 2024-01-05 | 200    |
| 2        | 101         | 2024-02-10 | 300    |
| 3        | 101         | 2024-03-15 | 100    |
| 4        | 102         | 2024-01-20 | 450    |

**Expected Output:**

| customer_id | first_order | last_order | total_orders |
|-------------|-------------|------------|--------------|
| 101         | 2024-01-05  | 2024-03-15 | 3            |
| 102         | 2024-01-20  | 2024-01-20 | 1            |

```sql
SELECT
    customer_id,
    MIN(order_date) AS first_order,
    MAX(order_date) AS last_order,
    COUNT(*) AS total_orders
FROM orders
GROUP BY customer_id;
```

---

## Q26. Median Salary

**Same `employees` table as Q2.**

```sql
-- MySQL approach using PERCENTILE_CONT (supported in MySQL 8+)
SELECT
    AVG(salary) AS median_salary
FROM (
    SELECT salary,
           ROW_NUMBER() OVER (ORDER BY salary) AS rn,
           COUNT(*) OVER () AS cnt
    FROM employees
) t
WHERE rn IN (FLOOR((cnt + 1) / 2), CEIL((cnt + 1) / 2));
```

**Approach:** Find the middle row(s). For odd count, one middle row; for even count, average the two middle rows.

---

## Q27. Percentage of Total per Category

**Same `product_sales` table as Q20.**

**Expected Output:**

| product | category | revenue | pct_of_category |
|---------|----------|---------|-----------------|
| B       | Tech     | 8000    | 38.10           |
| E       | Tech     | 8000    | 38.10           |
| A       | Tech     | 5000    | 23.81           |
| D       | Home     | 4500    | 60.00           |
| C       | Home     | 3000    | 40.00           |

```sql
SELECT
    product,
    category,
    revenue,
    ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY category), 2) AS pct_of_category
FROM product_sales;
```

---

## Q28. Lead and Lag — Next Purchase Gap

**Same `orders` table as Q25.**

**Expected Output:**

| customer_id | order_date | next_order_date | days_between |
|-------------|------------|-----------------|--------------|
| 101         | 2024-01-05 | 2024-02-10      | 36           |
| 101         | 2024-02-10 | 2024-03-15      | 34           |
| 101         | 2024-03-15 | NULL            | NULL         |
| 102         | 2024-01-20 | NULL            | NULL         |

```sql
SELECT
    customer_id,
    order_date,
    LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_date,
    DATEDIFF(
        LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
        order_date
    ) AS days_between
FROM orders;
```

---

## Q29. Top 2 Products per Category

**Same `product_sales` table as Q20.**

**Expected Output:**

| category | product | revenue |
|----------|---------|---------|
| Tech     | B       | 8000    |
| Tech     | E       | 8000    |
| Home     | D       | 4500    |
| Home     | C       | 3000    |

```sql
SELECT category, product, revenue
FROM (
    SELECT category, product, revenue,
           DENSE_RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS dr
    FROM product_sales
) t
WHERE dr <= 2;
```

---

## Q30. Salary Percentile per Department

**Same `employees` table as Q2.**

```sql
SELECT
    name,
    dept,
    salary,
    ROUND(
        PERCENT_RANK() OVER (PARTITION BY dept ORDER BY salary) * 100,
        2
    ) AS percentile
FROM employees;
```

**Approach:** `PERCENT_RANK()` = (rank - 1) / (total rows - 1). Returns 0 to 1.

---

## Q31. Identify Sessions from Events (Session Gap = 30 min)

**Table: `page_events`**

| user_id | event_time          |
|---------|---------------------|
| 1       | 2024-01-01 10:00:00 |
| 1       | 2024-01-01 10:10:00 |
| 1       | 2024-01-01 10:45:00 |
| 1       | 2024-01-01 11:30:00 |

**Expected Output:** Assign a session ID (new session if gap > 30 min)

| user_id | event_time          | session_id |
|---------|---------------------|------------|
| 1       | 2024-01-01 10:00:00 | 1          |
| 1       | 2024-01-01 10:10:00 | 1          |
| 1       | 2024-01-01 10:45:00 | 2          |
| 1       | 2024-01-01 11:30:00 | 3          |

```sql
SELECT
    user_id,
    event_time,
    SUM(new_session) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
FROM (
    SELECT
        user_id,
        event_time,
        CASE
            WHEN TIMESTAMPDIFF(MINUTE,
                LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time),
                event_time) > 30 OR
                LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) IS NULL
            THEN 1
            ELSE 0
        END AS new_session
    FROM page_events
) flagged;
```

**Approach (Gap-and-Island):** Flag rows where the gap from the prior event exceeds 30 minutes. Then cumulative sum of flags = session ID.

---

## Q32. Find the Longest Streak of Daily Logins

**Table: `logins`**

| user_id | login_date |
|---------|------------|
| 1       | 2024-01-01 |
| 1       | 2024-01-02 |
| 1       | 2024-01-03 |
| 1       | 2024-01-05 |
| 1       | 2024-01-06 |

**Expected Output:**

| user_id | longest_streak |
|---------|----------------|
| 1       | 3              |

```sql
WITH groups AS (
    SELECT
        user_id,
        login_date,
        DATEADD(DAY, -ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date), login_date) AS grp
    FROM logins
),
streaks AS (
    SELECT user_id, grp, COUNT(*) AS streak_len
    FROM groups
    GROUP BY user_id, grp
)
SELECT user_id, MAX(streak_len) AS longest_streak
FROM streaks
GROUP BY user_id;
```

**Approach (Classic Island):** Subtracting the row number from the date creates identical "group" dates for consecutive days. Group by this pseudo-date to find each streak length.

---

## Q33. Pivot — Monthly Sales by Product

**Table: `sales`** (same as Q7)

**Expected Output:**

| product | Jan | Feb |
|---------|-----|-----|
| A       | 300 | 300 |
| B       | 150 | 0   |

```sql
SELECT
    product,
    SUM(CASE WHEN MONTH(sale_date) = 1 THEN amount ELSE 0 END) AS Jan,
    SUM(CASE WHEN MONTH(sale_date) = 2 THEN amount ELSE 0 END) AS Feb
FROM sales
GROUP BY product;
```

---

## Q34. Rolling 7-Day Active Users (DAU)

**Table: `user_activity`** (same as Q18)

```sql
SELECT
    activity_date,
    COUNT(DISTINCT user_id) AS dau,
    COUNT(DISTINCT user_id) OVER (
        ORDER BY activity_date
        RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW
    ) AS rolling_7d_users
FROM user_activity
GROUP BY activity_date;
```

---

## Q35. Year-over-Year Comparison

**Table: `annual_revenue`**

| year | revenue |
|------|---------|
| 2022 | 100000  |
| 2023 | 120000  |
| 2024 | 115000  |

**Expected Output:**

| year | revenue | prev_year_revenue | yoy_growth_pct |
|------|---------|-------------------|----------------|
| 2022 | 100000  | NULL              | NULL           |
| 2023 | 120000  | 100000            | 20.00          |
| 2024 | 115000  | 120000            | -4.17          |

```sql
SELECT
    year,
    revenue,
    LAG(revenue) OVER (ORDER BY year) AS prev_year_revenue,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY year))
              / LAG(revenue) OVER (ORDER BY year),
        2
    ) AS yoy_growth_pct
FROM annual_revenue;
```

---

# SECTION 4 — CTEs, Recursion & Advanced Logic (Hard)

---

## Q36. Recursive CTE — Org Hierarchy

**Table: `employees`**

| emp_id | name    | manager_id |
|--------|---------|------------|
| 1      | CEO     | NULL       |
| 2      | VP Eng  | 1          |
| 3      | Manager | 2          |
| 4      | Dev     | 3          |

**Expected Output:** Full reporting chain from CEO down

| emp_id | name    | level |
|--------|---------|-------|
| 1      | CEO     | 0     |
| 2      | VP Eng  | 1     |
| 3      | Manager | 2     |
| 4      | Dev     | 3     |

```sql
WITH RECURSIVE org AS (
    -- Anchor: start from the top
    SELECT emp_id, name, manager_id, 0 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive: join children
    SELECT e.emp_id, e.name, e.manager_id, org.level + 1
    FROM employees e
    JOIN org ON e.manager_id = org.emp_id
)
SELECT emp_id, name, level
FROM org
ORDER BY level;
```

---

## Q37. Customer Retention — Month 1 Cohorts

**Table: `orders`**

| customer_id | order_date |
|-------------|------------|
| 1           | 2024-01-05 |
| 1           | 2024-02-10 |
| 2           | 2024-01-20 |
| 3           | 2024-02-01 |

**Expected Output:** For Jan cohort — what % returned in Feb?

| cohort_month | return_month | cohort_size | retained | retention_rate |
|--------------|--------------|-------------|----------|----------------|
| 2024-01      | 2024-02      | 2           | 1        | 50.00          |

```sql
WITH cohort AS (
    SELECT customer_id,
           DATE_FORMAT(MIN(order_date), '%Y-%m') AS cohort_month
    FROM orders
    GROUP BY customer_id
),
activity AS (
    SELECT o.customer_id,
           c.cohort_month,
           DATE_FORMAT(o.order_date, '%Y-%m') AS order_month
    FROM orders o
    JOIN cohort c ON o.customer_id = c.customer_id
)
SELECT
    cohort_month,
    order_month AS return_month,
    COUNT(DISTINCT a2.customer_id) AS cohort_size,
    COUNT(DISTINCT a1.customer_id) AS retained,
    ROUND(100.0 * COUNT(DISTINCT a1.customer_id) / COUNT(DISTINCT a2.customer_id), 2) AS retention_rate
FROM activity a1
RIGHT JOIN (SELECT customer_id, cohort_month FROM cohort WHERE cohort_month = '2024-01') a2
    ON a1.customer_id = a2.customer_id AND a1.order_month = '2024-02'
GROUP BY cohort_month, return_month;
```

---

## Q38. Find Missing Dates in a Series

**Table: `daily_sales`**

| sale_date  | revenue |
|------------|---------|
| 2024-01-01 | 100     |
| 2024-01-02 | 200     |
| 2024-01-04 | 150     |  ← Jan 3 is missing

**Expected Output:**

| missing_date |
|--------------|
| 2024-01-03   |

```sql
WITH RECURSIVE date_series AS (
    SELECT MIN(sale_date) AS d FROM daily_sales
    UNION ALL
    SELECT d + INTERVAL 1 DAY FROM date_series WHERE d < (SELECT MAX(sale_date) FROM daily_sales)
)
SELECT d AS missing_date
FROM date_series
WHERE d NOT IN (SELECT sale_date FROM daily_sales);
```

---

## Q39. Multi-Level Aggregation — Sales Subtotals with ROLLUP

**Same `sales` table as Q7.**

```sql
SELECT
    COALESCE(product, 'ALL PRODUCTS') AS product,
    COALESCE(DATE_FORMAT(sale_date, '%Y-%m'), 'ALL MONTHS') AS month,
    SUM(amount) AS total
FROM sales
GROUP BY product, DATE_FORMAT(sale_date, '%Y-%m') WITH ROLLUP;
```

**Approach:** `WITH ROLLUP` generates subtotal rows automatically. `COALESCE` replaces NULLs in rollup rows with readable labels.

---

## Q40. Identify Accounts with Suspicious Activity (> 3 txns in 1 hour)

**Table: `transactions`** (same as Q21)

```sql
SELECT DISTINCT user_id
FROM (
    SELECT
        user_id,
        txn_date,
        COUNT(*) OVER (
            PARTITION BY user_id
            ORDER BY txn_date
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
        ) AS txns_in_hour
    FROM transactions
) t
WHERE txns_in_hour > 3;
```

---

## Q41. Friend Recommendation (Mutual Friends)

**Table: `friendships`**

| user_a | user_b |
|--------|--------|
| 1      | 2      |
| 1      | 3      |
| 2      | 4      |
| 3      | 4      |

*Find users not yet friends who share at least 1 mutual friend.*

**Expected Output:**

| user_a | user_b | mutual_friends |
|--------|--------|----------------|
| 1      | 4      | 2              |

```sql
WITH all_friends AS (
    SELECT user_a AS user1, user_b AS user2 FROM friendships
    UNION ALL
    SELECT user_b, user_a FROM friendships
)
SELECT a.user1, b.user1 AS user_b, COUNT(*) AS mutual_friends
FROM all_friends a
JOIN all_friends b ON a.user2 = b.user2 AND a.user1 < b.user1
WHERE NOT EXISTS (
    SELECT 1 FROM all_friends f
    WHERE f.user1 = a.user1 AND f.user2 = b.user1
)
GROUP BY a.user1, b.user1
ORDER BY mutual_friends DESC;
```

---

## Q42. Detect Island — Consecutive Same-Status Records

**Table: `server_status`**

| check_time          | status |
|---------------------|--------|
| 2024-01-01 10:00:00 | UP     |
| 2024-01-01 10:05:00 | UP     |
| 2024-01-01 10:10:00 | DOWN   |
| 2024-01-01 10:15:00 | DOWN   |
| 2024-01-01 10:20:00 | UP     |

**Expected Output:** Start and end of each status period

| status | period_start        | period_end          |
|--------|---------------------|---------------------|
| UP     | 2024-01-01 10:00:00 | 2024-01-01 10:05:00 |
| DOWN   | 2024-01-01 10:10:00 | 2024-01-01 10:15:00 |
| UP     | 2024-01-01 10:20:00 | 2024-01-01 10:20:00 |

```sql
WITH flagged AS (
    SELECT check_time, status,
           CASE WHEN status = LAG(status) OVER (ORDER BY check_time)
                THEN 0 ELSE 1 END AS is_new_group
    FROM server_status
),
grouped AS (
    SELECT check_time, status,
           SUM(is_new_group) OVER (ORDER BY check_time) AS grp
    FROM flagged
)
SELECT status,
       MIN(check_time) AS period_start,
       MAX(check_time) AS period_end
FROM grouped
GROUP BY grp, status
ORDER BY period_start;
```

---

## Q43. Calculate Shopping Cart Abandonment Rate

**Table: `funnel_events`**

| event_id | user_id | event_type    | event_time          |
|----------|---------|---------------|---------------------|
| 1        | A       | add_to_cart   | 2024-01-01 10:00:00 |
| 2        | A       | checkout      | 2024-01-01 10:05:00 |
| 3        | B       | add_to_cart   | 2024-01-01 11:00:00 |
| 4        | C       | add_to_cart   | 2024-01-01 12:00:00 |
| 5        | C       | checkout      | 2024-01-01 12:10:00 |
| 6        | C       | purchase      | 2024-01-01 12:15:00 |

**Expected Output:**

| added_to_cart | checked_out | purchased | abandonment_rate |
|---------------|-------------|-----------|------------------|
| 3             | 2           | 1         | 33.33            |

```sql
SELECT
    COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN user_id END) AS added_to_cart,
    COUNT(DISTINCT CASE WHEN event_type = 'checkout' THEN user_id END) AS checked_out,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) AS purchased,
    ROUND(100.0 * (
        COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN user_id END) -
        COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END)
    ) / NULLIF(COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN user_id END), 0), 2) AS abandonment_rate
FROM funnel_events;
```

---

## Q44. Median without PERCENTILE_CONT (MySQL 5.x)

**Table: `scores`** (same as Q24)

```sql
SELECT AVG(score) AS median
FROM (
    SELECT score
    FROM (
        SELECT score, @row := @row + 1 AS rn, @total AS total
        FROM scores, (SELECT @row := 0, @total := (SELECT COUNT(*) FROM scores)) init
        ORDER BY score
    ) ranked
    WHERE rn BETWEEN total / 2.0 AND total / 2.0 + 1
) mid;
```

---

## Q45. Find Employees with No Reports (Leaf Nodes)

**Same org `employees` table as Q36.**

**Expected Output:**

| emp_id | name |
|--------|------|
| 4      | Dev  |

```sql
SELECT emp_id, name
FROM employees
WHERE emp_id NOT IN (
    SELECT DISTINCT manager_id
    FROM employees
    WHERE manager_id IS NOT NULL
);
```

---

# SECTION 5 — Real FAANG Classics (Hard)

---

## Q46. [Meta/Facebook] Friend Count & Mutual Friends

**Table: `friendship`**

| user1 | user2 |
|-------|-------|
| 1     | 2     |
| 1     | 3     |
| 2     | 3     |
| 3     | 4     |

**Problem:** For each user, find their friend count.

```sql
WITH all_edges AS (
    SELECT user1 AS user_id, user2 AS friend_id FROM friendship
    UNION ALL
    SELECT user2, user1 FROM friendship
)
SELECT user_id, COUNT(*) AS friend_count
FROM all_edges
GROUP BY user_id
ORDER BY user_id;
```

---

## Q47. [Amazon] Repeat Purchasers within 30 Days

**Table: `orders`**

| order_id | customer_id | order_date | product_id |
|----------|-------------|------------|------------|
| 1        | 101         | 2024-01-01 | A          |
| 2        | 101         | 2024-01-20 | A          |
| 3        | 102         | 2024-01-05 | B          |
| 4        | 102         | 2024-02-20 | B          |

**Expected Output:** Customers who bought the same product again within 30 days

| customer_id | product_id |
|-------------|------------|
| 101         | A          |

```sql
SELECT DISTINCT o1.customer_id, o1.product_id
FROM orders o1
JOIN orders o2
  ON o1.customer_id = o2.customer_id
 AND o1.product_id = o2.product_id
 AND o2.order_date > o1.order_date
 AND DATEDIFF(o2.order_date, o1.order_date) <= 30;
```

---

## Q48. [Google] Active Users Definition — DAU/MAU Ratio

**Table: `user_activity`**

| user_id | activity_date |
|---------|---------------|
| 1       | 2024-01-28    |
| 1       | 2024-01-29    |
| 2       | 2024-01-29    |
| 3       | 2024-01-01    |

**Expected Output:** DAU/MAU stickiness ratio for Jan 29, 2024

| date       | dau | mau | stickiness |
|------------|-----|-----|------------|
| 2024-01-29 | 2   | 3   | 66.67      |

```sql
WITH target_date AS (SELECT DATE '2024-01-29' AS d),
dau AS (
    SELECT COUNT(DISTINCT user_id) AS dau_count
    FROM user_activity, target_date
    WHERE activity_date = d
),
mau AS (
    SELECT COUNT(DISTINCT user_id) AS mau_count
    FROM user_activity, target_date
    WHERE activity_date BETWEEN DATE_SUB(d, INTERVAL 29 DAY) AND d
)
SELECT
    (SELECT d FROM target_date) AS date,
    dau_count AS dau,
    mau_count AS mau,
    ROUND(100.0 * dau_count / mau_count, 2) AS stickiness
FROM dau, mau;
```

---

## Q49. [Netflix] Content Watch Completion Rate

**Table: `watch_events`**

| user_id | content_id | watch_pct |
|---------|------------|-----------|
| 1       | M1         | 100       |
| 2       | M1         | 45        |
| 3       | M1         | 80        |
| 1       | M2         | 30        |
| 2       | M2         | 100       |

**Expected Output:** % of users who completed (>= 80%) each content

| content_id | total_viewers | completions | completion_rate |
|------------|---------------|-------------|-----------------|
| M1         | 3             | 2           | 66.67           |
| M2         | 2             | 1           | 50.00           |

```sql
SELECT
    content_id,
    COUNT(*) AS total_viewers,
    SUM(CASE WHEN watch_pct >= 80 THEN 1 ELSE 0 END) AS completions,
    ROUND(100.0 * SUM(CASE WHEN watch_pct >= 80 THEN 1 ELSE 0 END) / COUNT(*), 2) AS completion_rate
FROM watch_events
GROUP BY content_id
ORDER BY content_id;
```

---

## Q50. [Uber/Lyft] Driver Earnings vs Average — with Percentile

**Table: `driver_trips`**

| driver_id | trip_date  | earnings |
|-----------|------------|----------|
| D1        | 2024-01-01 | 200      |
| D1        | 2024-01-02 | 300      |
| D2        | 2024-01-01 | 150      |
| D2        | 2024-01-02 | 400      |
| D3        | 2024-01-01 | 100      |

**Expected Output:** Per driver — total, vs global average, and their percentile

| driver_id | total_earnings | global_avg | diff_from_avg | percentile |
|-----------|----------------|------------|---------------|------------|
| D2        | 550            | 383.33     | 166.67        | 100.00     |
| D1        | 500            | 383.33     | 116.67        | 50.00      |
| D3        | 100            | 383.33     | -283.33       | 0.00       |

```sql
WITH driver_totals AS (
    SELECT driver_id, SUM(earnings) AS total_earnings
    FROM driver_trips
    GROUP BY driver_id
)
SELECT
    driver_id,
    total_earnings,
    ROUND(AVG(total_earnings) OVER (), 2) AS global_avg,
    ROUND(total_earnings - AVG(total_earnings) OVER (), 2) AS diff_from_avg,
    ROUND(PERCENT_RANK() OVER (ORDER BY total_earnings) * 100, 2) AS percentile
FROM driver_totals
ORDER BY total_earnings DESC;
```

---

# QUICK REFERENCE — Key Concepts

## Window Function Frames

```sql
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  -- cumulative from start
ROWS BETWEEN 2 PRECEDING AND CURRENT ROW           -- rolling 3-row window
RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW  -- rolling 7 days
```

## Ranking Functions Cheat Sheet

| Function | Ties | Gaps |
|----------|------|------|
| `ROW_NUMBER()` | Arbitrary order | N/A |
| `RANK()` | Same rank | Yes (1,1,3) |
| `DENSE_RANK()` | Same rank | No (1,1,2) |
| `PERCENT_RANK()` | 0.0 to 1.0 | — |
| `NTILE(n)` | Split into n buckets | — |

## NULL-Safe Patterns

```sql
-- Avoid divide-by-zero
revenue / NULLIF(total, 0)

-- Replace NULLs in ROLLUP
COALESCE(dept, 'All Depts')

-- NOT IN fails with NULLs — prefer NOT EXISTS or LEFT JOIN IS NULL
```

## Gap-and-Island Template

```sql
-- Consecutive groups: subtract row number from date → same value for consecutive rows
date_col - ROW_NUMBER() OVER (PARTITION BY grp_col ORDER BY date_col) AS island_group
```

---

*End of 50 SQL FAANG Questions*
