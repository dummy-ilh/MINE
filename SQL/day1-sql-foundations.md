# Day 1 — SELECT Fundamentals
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Execution Order
2. SELECT & FROM
3. WHERE with AND / OR / IN / BETWEEN / LIKE
4. NULL Handling & COALESCE
5. ORDER BY & LIMIT
6. DISTINCT
7. CASE WHEN

---

## 1. Execution Order

You **write** SQL in this order:
```
SELECT → FROM → WHERE → GROUP BY → HAVING → ORDER BY → LIMIT
```
SQL **runs** it in this order:
```
FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```
```sql
-- ❌ FAILS — alias doesn't exist at WHERE stage
SELECT age AS senior FROM users WHERE senior > 60;

-- ✅ WORKS
SELECT age AS senior FROM users WHERE age > 60;
```

---

## 2. SELECT & FROM

```sql
-- All columns
SELECT * FROM employees;

-- Specific columns
SELECT emp_id, name, salary FROM employees;

-- With alias
SELECT name AS employee_name, salary * 12 AS annual_salary
FROM employees;
```

---

## 3. WHERE — Filtering Rows

```sql
-- Single condition
SELECT * FROM employees WHERE department = 'Engineering';

-- AND / OR
SELECT * FROM employees
WHERE department = 'Engineering'
  AND salary > 100000;

-- Always wrap OR groups in parentheses
SELECT * FROM employees
WHERE YEAR(hire_date) = 2023
  AND (department = 'DS' OR department = 'MLE');
```

---

## 4. IN, BETWEEN, LIKE

```sql
-- IN
SELECT * FROM employees
WHERE department IN ('DS', 'MLE', 'Engineering');

-- BETWEEN (inclusive)
SELECT * FROM employees
WHERE salary BETWEEN 80000 AND 120000;

-- LIKE
SELECT * FROM employees WHERE name LIKE 'A%';      -- starts with A
SELECT * FROM employees WHERE email LIKE '%@gmail.com'; -- ends with
SELECT * FROM employees WHERE name LIKE '_ohn';    -- John, Rohn etc.
```

---

## 5. NULL Handling

```sql
-- ❌ Wrong
SELECT * FROM employees WHERE manager_id = NULL;

-- ✅ Correct
SELECT * FROM employees WHERE manager_id IS NULL;
SELECT * FROM employees WHERE manager_id IS NOT NULL;
```

### COALESCE
Returns the **first non-null value** from the list, left to right.

```sql
SELECT user_id,
  COALESCE(phone, work_email, personal_email, 'no contact') AS best_contact
FROM contacts;
```

```sql
-- NULL in math = NULL. Fix with COALESCE
SELECT 100 + COALESCE(bonus, 0) AS total
FROM employees;
```

---

## 6. ORDER BY & LIMIT

```sql
-- Sort descending
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees ORDER BY department ASC, salary DESC;

-- Top 5
SELECT * FROM employees ORDER BY salary DESC LIMIT 5;

-- Pagination
SELECT * FROM employees ORDER BY emp_id LIMIT 10 OFFSET 20;
```

---

## 7. DISTINCT

```sql
SELECT DISTINCT department FROM employees;
SELECT DISTINCT department, job_title FROM employees;
```

---

## 8. CASE WHEN

```sql
SELECT name, salary,
  CASE
    WHEN salary >= 150000 THEN 'Senior'
    WHEN salary >= 100000 THEN 'Mid'
    ELSE 'Junior'
  END AS level
FROM employees;
```
> ⚠️ CASE evaluates **top to bottom, stops at first match.** Order your conditions from most specific to least specific.

---

## Practice Questions

### Q1 — Easy ✅
**Table:** `users(user_id, name, country, signup_date, is_active)`
Get names of all active users from India, sorted by signup date newest first.

```sql
SELECT name
FROM users
WHERE country = 'India'
  AND is_active = 1
ORDER BY signup_date DESC;
```

---

### Q2 — Medium ✅
**Table:** `users(user_id, name, country, signup_date, is_active)`
Find users who signed up in 2023 and whose name starts with 'A' or 'S'.

```sql
SELECT user_id, name, signup_date
FROM users
WHERE YEAR(signup_date) = 2023
  AND (name LIKE 'A%' OR name LIKE 'S%');
```
> ⚠️ Always wrap OR in parentheses when mixing with AND.

---

### Q3 — Hard ✅
**Table:** `employees(emp_id, name, department, salary, manager_id, hire_date)`
Return name, salary, salary_band for employees hired after 2020, not in HR.

```sql
SELECT name, salary,
  CASE
    WHEN salary >= 150000 THEN 'High'
    WHEN salary >= 100000 THEN 'Medium'
    ELSE 'Low'
  END AS salary_band
FROM employees
WHERE YEAR(hire_date) > 2020
  AND department != 'HR'
ORDER BY salary DESC;
```

---

## Key Takeaways

- **Execution order** — SQL runs FROM first, SELECT last. Aliases don't exist in WHERE.
- **Quotes** — Always single quotes `'India'` for strings, never double.
- **Dates** — Use `YEAR()` or `BETWEEN` for date filters, never `LIKE`.
- **OR + AND** — Always wrap OR groups in parentheses.
- **NULL** — Never use `= NULL`, always `IS NULL`.
- **COALESCE** — First non-null wins. Great for fallback values and math.

---

*Day 1 complete — 29 days to go 🚀*
