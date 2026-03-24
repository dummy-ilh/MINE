# Learning SQL — Study Guide
> Based on *Learning SQL, 2nd Edition* by Alan Beaulieu (O'Reilly, 2009)  
> Database: MySQL **bank** schema  
> All examples use the bank sample database included with the book.

---

## Reference: The Bank Schema (Core Tables)

```sql
-- Key tables used throughout all examples
employee   (emp_id, fname, lname, title, start_date, dept_id,
            superior_emp_id, assigned_branch_id)
department (dept_id, name)
branch     (branch_id, name, address, city, state, zip)
account    (account_id, product_cd, cust_id, open_date, status,
            open_emp_id, open_branch_id, avail_balance, pending_balance)
customer   (cust_id, fed_id, cust_type_cd, address, city, state)
individual (cust_id, fname, lname, birth_date)
business   (cust_id, name, state_id, incorp_date)
product    (product_cd, name, product_type_cd, date_offered, date_retired)
product_type (product_type_cd, name)
transaction (txn_id, txn_date, account_id, txn_type_cd, amount, teller_emp_id)
```

**Sample employee rows:**

| emp_id | fname   | lname    | title              | dept_id |
|--------|---------|----------|--------------------|---------|
| 1      | Michael | Smith    | President          | 3       |
| 2      | Susan   | Barker   | Vice President     | 3       |
| 3      | Robert  | Tyler    | Treasurer          | 3       |
| 4      | Susan   | Hawthorne| Operations Manager | 1       |
| 6      | Helen   | Fleming  | Head Teller        | 1       |
| 10     | Paula   | Roberts  | Head Teller        | 1       |

**Sample account rows:**

| account_id | product_cd | cust_id | avail_balance | status |
|------------|------------|---------|---------------|--------|
| 1          | CHK        | 1       | 1057.75       | ACTIVE |
| 2          | SAV        | 1       | 500.00        | ACTIVE |
| 3          | CD         | 1       | 3000.00       | ACTIVE |
| 7          | CHK        | 3       | 1057.75       | ACTIVE |
| 10         | CHK        | 4       | 534.12        | ACTIVE |
| 15         | CD         | 6       | 10000.00      | ACTIVE |
| 29         | SBL        | 13      | 50000.00      | ACTIVE |

---

# Chapter 1 — A Little Background

---

## Q1 — Relational Model vs Hierarchical Model

**Question:**  
What problem does the relational model solve that hierarchical and network database systems struggled with?

**Answer:**  
Hierarchical systems store data as one-or-more tree structures — each node has one parent (single-parent hierarchy). To find an account's transactions you must navigate from customer → account → transactions in order. Network systems added multi-parent links but still required navigating pre-defined paths.

The relational model (E.F. Codd, 1970) replaces navigation-by-pointer with **redundant data** linking rows across flat tables. You can reach any data from any starting point using a query, without knowing the physical storage structure.

```sql
-- In relational model: link customer to account via shared cust_id column
SELECT c.fname, c.lname, a.account_id, a.avail_balance
FROM customer c
INNER JOIN individual i  ON c.cust_id = i.cust_id
INNER JOIN account   a  ON c.cust_id = a.cust_id
WHERE i.lname = 'Smith';
```

**Sample Output:**

| fname  | lname | account_id | avail_balance |
|--------|-------|------------|---------------|
| George | Smith | 103        | 75.00         |

---

## Q2 — SQL Statement Classes

**Question:**  
What are the three main classes of SQL statements and what does each do?

**Answer:**

| Class | Purpose | Examples |
|-------|---------|---------|
| **SQL schema statements** | Define database objects | `CREATE TABLE`, `ALTER TABLE`, `DROP INDEX`, `CREATE VIEW` |
| **SQL data statements** | Create, manipulate, and retrieve data | `SELECT`, `INSERT`, `UPDATE`, `DELETE` |
| **SQL transaction statements** | Begin, end, and roll back transactions | `START TRANSACTION`, `COMMIT`, `ROLLBACK`, `SAVEPOINT` |

```sql
-- Schema statement: define a table
CREATE TABLE branch (
  branch_id   SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  name        VARCHAR(20) NOT NULL,
  city        VARCHAR(20),
  PRIMARY KEY (branch_id)
);

-- Data statement: insert a row
INSERT INTO branch (name, city) VALUES ('Headquarters', 'Waltham');

-- Data statement: retrieve rows
SELECT branch_id, name, city FROM branch;
```

---

# Chapter 2 — Creating and Populating a Database

---

## Q3 — MySQL Data Types: Character

**Question:**  
What is the difference between `CHAR` and `VARCHAR` in MySQL? When should you use each?

**Answer:**
```sql
-- CHAR(n): fixed-length, always stores exactly n bytes (right-padded with spaces)
-- VARCHAR(n): variable-length, stores only as many bytes as needed (+ 1 or 2 overhead bytes)

CREATE TABLE example (
  code      CHAR(5),        -- always 5 bytes; good for fixed-length codes like 'US   '
  name      VARCHAR(100),   -- up to 100 bytes; only stores what's needed
  notes     TEXT            -- up to 65,535 bytes; not indexable
);
```

| Type       | Storage        | Use When                                 |
|------------|----------------|------------------------------------------|
| `CHAR(n)`  | Always n bytes | Fixed-length: state codes, zip codes, flags |
| `VARCHAR(n)`| 1–n+2 bytes   | Variable text: names, addresses, titles  |
| `TEXT`     | Up to 65KB     | Long descriptions (cannot be fully indexed) |

---

## Q4 — MySQL Data Types: Numeric

**Question:**  
Which numeric types are available in MySQL and when should you choose `FLOAT` vs `DECIMAL`?

**Answer:**
```sql
-- Integer types (exact)
tinyint_col   TINYINT,            -- 1 byte,  -128 to 127
smallint_col  SMALLINT,           -- 2 bytes, -32768 to 32767
int_col       INT,                -- 4 bytes, -2.1B to 2.1B
bigint_col    BIGINT,             -- 8 bytes, very large integers

-- Floating-point (approximate — avoid for money!)
float_col     FLOAT(p,s),         -- 4 bytes,  approximate

-- Fixed-point (exact — use for money/finance)
balance       DECIMAL(10,2),      -- up to 10 digits, 2 after decimal
```

**Practical example from bank schema:**
```sql
CREATE TABLE account (
  account_id      INT UNSIGNED NOT NULL AUTO_INCREMENT,
  avail_balance   FLOAT(10,2),   -- book uses FLOAT; DECIMAL is safer for money
  pending_balance FLOAT(10,2),
  PRIMARY KEY (account_id)
);
```

> **Key rule:** Never use `FLOAT` or `DOUBLE` for monetary values — floating-point arithmetic produces rounding errors (0.1 + 0.2 ≠ 0.3). Use `DECIMAL(p,s)` instead.

---

## Q5 — MySQL Data Types: Temporal

**Question:**  
Describe the temporal data types in MySQL. What are the differences between `DATE`, `DATETIME`, and `TIMESTAMP`?

**Answer:**

| Type        | Format                  | Range                          | Timezone-aware? |
|-------------|-------------------------|--------------------------------|-----------------|
| `DATE`      | YYYY-MM-DD              | 1000-01-01 to 9999-12-31       | No              |
| `DATETIME`  | YYYY-MM-DD HH:MI:SS     | 1000-01-01 to 9999-12-31       | No              |
| `TIMESTAMP` | YYYY-MM-DD HH:MI:SS     | 1970-01-01 to 2038-01-18       | Yes (UTC)       |
| `TIME`      | HHH:MI:SS               | -838:59:59 to 838:59:59        | No              |
| `YEAR`      | YYYY                    | 1901 to 2155                   | No              |

```sql
CREATE TABLE transaction (
  txn_id      INT UNSIGNED NOT NULL AUTO_INCREMENT,
  txn_date    DATETIME NOT NULL,   -- date + time of transaction
  account_id  INT UNSIGNED NOT NULL,
  amount      FLOAT(10,2),
  PRIMARY KEY (txn_id)
);

-- Insert with explicit datetime
INSERT INTO transaction (txn_date, account_id, amount)
VALUES ('2008-09-17 15:30:00', 1, 250.00);
```

---

## Q6 — Table Creation: Step by Step

**Question:**  
Walk through the three steps of table creation for a new `person` table that stores first name, last name, gender, birth date, and address.

**Answer:**

**Step 1 — Design (identify entities and attributes):**
- Entity: `person`
- Attributes: first_name, last_name, gender, birth_date, street, city, state, country, postal_code, email

**Step 2 — Refinement (choose types, nullability, constraints):**
```
person_id   : surrogate PK, INT UNSIGNED AUTO_INCREMENT
first_name  : VARCHAR(20), NOT NULL
last_name   : VARCHAR(20), NOT NULL
gender      : CHAR(1)  (M/F/O)
birth_date  : DATE
email       : VARCHAR(100)
```

**Step 3 — Build the DDL:**
```sql
CREATE TABLE person (
  person_id   SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
  fname       VARCHAR(20) NOT NULL,
  lname       VARCHAR(20) NOT NULL,
  gender      CHAR(1),
  birth_date  DATE,
  street      VARCHAR(30),
  city        VARCHAR(20),
  state       VARCHAR(20),
  country     VARCHAR(20),
  postal_code VARCHAR(20),
  CONSTRAINT pk_person PRIMARY KEY (person_id),
  CONSTRAINT chk_gender CHECK (gender IN ('M','F','O'))
);
```

---

## Q7 — INSERT, UPDATE, DELETE Basics

**Question:**  
Show how to insert a new row, update an existing row, and delete a row in the `person` table.

**Answer:**
```sql
-- INSERT: add a new person
INSERT INTO person (fname, lname, gender, birth_date)
VALUES ('William', 'Turner', 'M', '1972-05-27');

-- INSERT with explicit column list and NULL
INSERT INTO person (fname, lname, gender, birth_date, street, city)
VALUES ('Susan', 'Smith', 'F', '1975-11-02', '23 Maple St', 'Arlington');

-- UPDATE: change Susan's city
UPDATE person
SET city = 'Woburn', street = '14 Tremont St'
WHERE person_id = 2;

-- DELETE: remove a specific row
DELETE FROM person
WHERE person_id = 2;
```

**Sample Output after INSERT:**

| person_id | fname   | lname  | gender | birth_date |
|-----------|---------|--------|--------|------------|
| 1         | William | Turner | M      | 1972-05-27 |
| 2         | Susan   | Smith  | F      | 1975-11-02 |

---

## Q8 — Common Error Conditions

**Question:**  
What errors occur when you violate database constraints? Demonstrate each.

**Answer:**
```sql
-- 1. Nonunique Primary Key
INSERT INTO person (person_id, fname, lname)
VALUES (1, 'Duplicate', 'Person');
-- ERROR 1062 (23000): Duplicate entry '1' for key 'PRIMARY'

-- 2. Nonexistent Foreign Key
INSERT INTO account (cust_id, product_cd)
VALUES (9999, 'CHK');
-- ERROR 1452 (23000): Cannot add or update a child row: a foreign key
-- constraint fails (no customer with cust_id 9999 exists)

-- 3. Column value violation (type mismatch)
INSERT INTO person (person_id, fname, lname, birth_date)
VALUES (3, 'Bob', 'Jones', 'DEC-21-1980');
-- ERROR: invalid date format; must be YYYY-MM-DD

-- 4. NOT NULL violation
INSERT INTO person (person_id, fname)
VALUES (3, 'Bob');
-- ERROR 1364 (HY000): Field 'lname' doesn't have a default value
```

---

# Chapter 3 — Query Primer

---

## Q9 — The SELECT Clause: Literals, Expressions, Functions

**Question:**  
Show how a SELECT clause can include more than just table columns — literals, expressions, and built-in function calls.

**Answer:**
```sql
SELECT emp_id,
       'ACTIVE'          AS status,
       emp_id * 3.14159  AS empid_x_pi,
       UPPER(lname)      AS last_name_upper
FROM employee;
```

**Sample Output:**

| emp_id | status | empid_x_pi | last_name_upper |
|--------|--------|------------|-----------------|
| 1      | ACTIVE | 3.14159    | SMITH           |
| 2      | ACTIVE | 6.28318    | BARKER          |
| 3      | ACTIVE | 9.42477    | TYLER           |

```sql
-- You can even skip FROM for pure function calls
SELECT VERSION(), USER(), DATABASE();
```

| version()             | user()             | database() |
|-----------------------|--------------------|------------|
| 6.0.3-alpha-community | lrngsql@localhost  | bank       |

---

## Q10 — DISTINCT: Removing Duplicates

**Question:**  
The `account` table has 24 rows, many with the same `cust_id`. Return only the unique customer IDs.

**Answer:**
```sql
-- Without DISTINCT: 24 rows (one per account)
SELECT cust_id FROM account;

-- With DISTINCT: 13 rows (one per unique customer)
SELECT DISTINCT cust_id FROM account;
```

**Sample Output (DISTINCT):**

| cust_id |
|---------|
| 1       |
| 2       |
| 3       |
| 4       |
| ...     |
| 13      |

> **Note:** `DISTINCT` forces a sort to eliminate duplicates — it can be slow on large result sets. Only use it when duplicates genuinely need removing.

---

## Q11 — The FROM Clause: Subquery-Generated Tables

**Question:**  
Show how a subquery in the FROM clause acts as a temporary (virtual) table.

**Answer:**
```sql
SELECT e.emp_id, e.fname, e.lname, d.name AS dept_name
FROM (SELECT emp_id, fname, lname, dept_id FROM employee) e
INNER JOIN department d
  ON e.dept_id = d.dept_id;
```

**Sample Output:**

| emp_id | fname   | lname | dept_name      |
|--------|---------|-------|----------------|
| 1      | Michael | Smith | Administration |
| 2      | Susan   | Barker| Administration |
| 4      | Susan   | Hawthorne | Operations |

---

## Q12 — WHERE, GROUP BY, HAVING, ORDER BY

**Question:**  
Write a query that counts accounts per product type, showing only products with more than 2 accounts, ordered by count descending.

**Answer:**
```sql
SELECT product_cd,
       COUNT(*) AS num_accounts
FROM account
WHERE status = 'ACTIVE'
GROUP BY product_cd
HAVING COUNT(*) > 2
ORDER BY num_accounts DESC;
```

**Sample Output:**

| product_cd | num_accounts |
|------------|--------------|
| CHK        | 10           |
| SAV        | 4            |
| CD         | 3            |

---

## Q13 — Sorting via Expressions and Numeric Placeholders

**Question:**  
Sort employees by the last two characters of their last name. Also demonstrate sorting by column position number.

**Answer:**
```sql
-- Sort by expression (last 2 chars of lname)
SELECT emp_id, lname
FROM employee
ORDER BY RIGHT(lname, 2);

-- Sort by numeric placeholder (column position in SELECT list)
SELECT emp_id, fname, lname
FROM employee
ORDER BY 3;    -- same as ORDER BY lname
```

**Sample Output (sorted by last 2 chars):**

| emp_id | lname    |
|--------|----------|
| 13     | Blake    |
| 6      | Fleming  |
| 4      | Hawthorne|
| 12     | Jameson  |

---

# Chapter 4 — Filtering

---

## Q14 — Equality, Inequality, and Range Conditions

**Question:**  
Demonstrate equality, inequality, and range filter conditions on the `employee` and `account` tables.

**Answer:**
```sql
-- Equality: employees who started on a specific date
SELECT emp_id, fname, lname, start_date
FROM employee
WHERE start_date = '2005-01-12';

-- Inequality: accounts not in savings
SELECT account_id, product_cd, avail_balance
FROM account
WHERE product_cd <> 'SAV'
ORDER BY avail_balance DESC;

-- Range with BETWEEN (inclusive on both ends)
SELECT emp_id, fname, lname, start_date
FROM employee
WHERE start_date BETWEEN '2001-01-01' AND '2003-01-01';
```

**Sample Output (BETWEEN):**

| emp_id | fname   | lname    | start_date |
|--------|---------|----------|------------|
| 7      | Chris   | Tucker   | 2002-09-15 |
| 8      | Sarah   | Parker   | 2002-12-02 |
| 9      | Jane    | Grossman | 2002-05-03 |

---

## Q15 — Membership Conditions: IN and NOT IN

**Question:**  
Find all employees whose title is one of three specific values using `IN`. Then find all accounts NOT belonging to a set of product codes.

**Answer:**
```sql
-- IN: match any value in the list
SELECT emp_id, fname, lname, title
FROM employee
WHERE title IN ('Teller', 'Head Teller', 'Operations Manager');

-- IN with subquery (generated set)
SELECT emp_id, fname, lname, title
FROM employee
WHERE emp_id IN (
  SELECT superior_emp_id
  FROM employee
  WHERE superior_emp_id IS NOT NULL
);
```

**Sample Output (IN with subquery — employees who supervise others):**

| emp_id | fname   | lname    | title              |
|--------|---------|----------|--------------------|
| 1      | Michael | Smith    | President          |
| 3      | Robert  | Tyler    | Treasurer          |
| 4      | Susan   | Hawthorne| Operations Manager |
| 6      | Helen   | Fleming  | Head Teller        |
| 10     | Paula   | Roberts  | Head Teller        |

---

## Q16 — Matching Conditions: LIKE and REGEXP

**Question:**  
Find all employees whose last name begins with 'F' or contains the string 'tle'. Then demonstrate the more powerful `REGEXP` operator.

**Answer:**
```sql
-- LIKE: % = any sequence of characters, _ = exactly one character
SELECT emp_id, fname, lname
FROM employee
WHERE lname LIKE 'F%'           -- starts with F
   OR lname LIKE '%tle%';       -- contains 'tle'

-- REGEXP: full regular expression support (MySQL-specific)
SELECT emp_id, fname, lname
FROM employee
WHERE lname REGEXP '^[FT]';     -- starts with F or T
```

**Sample Output (LIKE):**

| emp_id | fname  | lname  |
|--------|--------|--------|
| 6      | Helen  | Fleming|
| 17     | Beth   | Fowler |
| 3      | Robert | Tyler  |

---

## Q17 — NULL: The Four-Letter Word

**Question:**  
Why can't you use `= NULL` to check for null values? Show the correct syntax and demonstrate how NULL behaves in expressions.

**Answer:**
```sql
-- WRONG: = NULL never returns true (even for NULL values)
SELECT emp_id FROM employee WHERE superior_emp_id = NULL;  -- 0 rows!

-- CORRECT: IS NULL / IS NOT NULL
SELECT emp_id, fname, lname, superior_emp_id
FROM employee
WHERE superior_emp_id IS NULL;    -- employees with no manager

SELECT emp_id, fname, lname, superior_emp_id
FROM employee
WHERE superior_emp_id IS NOT NULL; -- employees with a manager

-- NULL in expressions: any arithmetic with NULL → NULL
SELECT NULL + 5;      -- NULL
SELECT NULL = NULL;   -- NULL (not TRUE!)
SELECT NULL IS NULL;  -- 1 (TRUE — the only safe way to check)
```

**Sample Output (IS NULL):**

| emp_id | fname   | lname | superior_emp_id |
|--------|---------|-------|-----------------|
| 1      | Michael | Smith | NULL            |

---

# Chapter 5 — Querying Multiple Tables

---

## Q18 — Cartesian Product

**Question:**  
What is a Cartesian product (cross join)? Show what happens when you forget the join condition.

**Answer:**
```sql
-- Cartesian product: every row in employee × every row in department
-- 18 employees × 3 departments = 54 rows!
SELECT e.fname, e.lname, d.name
FROM employee e, department d;

-- Correct: add join condition to reduce to meaningful combinations
SELECT e.fname, e.lname, d.name AS dept_name
FROM employee e
INNER JOIN department d ON e.dept_id = d.dept_id;
```

**Sample Output (correct join):**

| fname   | lname | dept_name      |
|---------|-------|----------------|
| Michael | Smith | Administration |
| Susan   | Barker| Administration |
| Susan   | Hawthorne | Operations |

---

## Q19 — INNER JOIN (Equi-Join)

**Question:**  
Return the names of all employees along with their assigned branch city, joining `employee` and `branch`.

**Answer:**
```sql
-- ANSI SQL join syntax (preferred)
SELECT e.fname, e.lname, b.name AS branch_name, b.city
FROM employee e
INNER JOIN branch b
  ON e.assigned_branch_id = b.branch_id
ORDER BY b.city, e.lname;

-- Older (non-ANSI) syntax — still valid but less readable
SELECT e.fname, e.lname, b.name, b.city
FROM employee e, branch b
WHERE e.assigned_branch_id = b.branch_id;
```

**Sample Output:**

| fname  | lname   | branch_name       | city    |
|--------|---------|-------------------|---------|
| Helen  | Fleming | So. NH Branch     | Salem   |
| Thomas | Ziegler | So. NH Branch     | Salem   |
| Paula  | Roberts | Woburn Branch     | Woburn  |
| Michael| Smith   | Headquarters      | Waltham |

---

## Q20 — Joining Three or More Tables

**Question:**  
Return each account's product name and the city of the branch where it was opened. This requires joining `account`, `product`, and `branch`.

**Answer:**
```sql
SELECT a.account_id,
       p.name   AS product_name,
       b.name   AS branch_name,
       b.city
FROM account a
INNER JOIN product p ON a.product_cd  = p.product_cd
INNER JOIN branch  b ON a.open_branch_id = b.branch_id
ORDER BY b.city, a.account_id;
```

**Sample Output:**

| account_id | product_name       | branch_name   | city    |
|------------|--------------------|---------------|---------|
| 1          | checking account   | Headquarters  | Waltham |
| 2          | savings account    | Headquarters  | Waltham |
| 7          | checking account   | Woburn Branch | Woburn  |

---

## Q21 — Using Subqueries as Tables

**Question:**  
Find the name and city of the branch where each account was opened, using a subquery in the FROM clause to pre-filter only active employees.

**Answer:**
```sql
SELECT a.account_id, a.product_cd, e.fname, e.lname, b.city
FROM account a
INNER JOIN (
  SELECT emp_id, fname, lname, assigned_branch_id
  FROM employee
  WHERE title = 'Head Teller'
) e ON a.open_emp_id = e.emp_id
INNER JOIN branch b ON e.assigned_branch_id = b.branch_id;
```

**Sample Output:**

| account_id | product_cd | fname  | lname   | city   |
|------------|------------|--------|---------|--------|
| 7          | CHK        | Paula  | Roberts | Woburn |
| 8          | MM         | Paula  | Roberts | Woburn |

---

## Q22 — Self-Join

**Question:**  
Using a self-join, display each employee alongside their direct supervisor's name.

**Answer:**
```sql
SELECT e.fname AS emp_fname,
       e.lname AS emp_lname,
       e.title,
       m.fname AS mgr_fname,
       m.lname AS mgr_lname
FROM employee e
LEFT OUTER JOIN employee m
  ON e.superior_emp_id = m.emp_id
ORDER BY e.emp_id;
```

**Sample Output:**

| emp_fname | emp_lname | title              | mgr_fname | mgr_lname |
|-----------|-----------|--------------------|-----------|-----------|
| Michael   | Smith     | President          | NULL      | NULL      |
| Susan     | Barker    | Vice President     | Michael   | Smith     |
| Robert    | Tyler     | Treasurer          | Michael   | Smith     |
| Susan     | Hawthorne | Operations Manager | Robert    | Tyler     |

---

## Q23 — Non-Equi-Joins

**Question:**  
Show how to join tables using conditions other than equality. Return all possible account-to-employee pairings where the employee's ID is greater than the account's customer ID (a non-equi-join example demonstrating the concept).

**Answer:**
```sql
-- Non-equi-join: join condition uses a range instead of equality
-- Example: show all employees who could have opened accounts with lower IDs
SELECT e.emp_id, e.fname, e.lname, a.account_id, a.product_cd
FROM employee e
INNER JOIN account a
  ON e.emp_id > a.cust_id    -- non-equality join condition
  AND a.cust_id <= 3
ORDER BY e.emp_id, a.account_id;
```

> **Key concept:** Most joins are equi-joins (=), but any comparison operator can form the join condition.

---

# Chapter 6 — Working with Sets

---

## Q24 — UNION: Stacking Result Sets

**Question:**  
Return a combined list of all employee first names and all individual customer first names, removing duplicates.

**Answer:**
```sql
-- UNION: combines and removes duplicates (sorted)
SELECT fname FROM individual
UNION
SELECT fname FROM employee;

-- UNION ALL: combines and keeps all duplicates (faster, no sort)
SELECT fname FROM individual
UNION ALL
SELECT fname FROM employee;
```

**Sample Output (UNION — unique names only):**

| fname   |
|---------|
| Beth    |
| Charles |
| Chris   |
| Frank   |
| ...     |

---

## Q25 — INTERSECT: Common Rows

**Question:**  
Find customer first names that also appear as employee first names (names in common between both tables).

**Answer:**
```sql
-- ANSI SQL INTERSECT (not natively in older MySQL — use JOIN workaround)
SELECT fname FROM individual
INTERSECT
SELECT fname FROM employee;

-- MySQL workaround using INNER JOIN
SELECT DISTINCT i.fname
FROM individual i
INNER JOIN employee e ON i.fname = e.fname;
```

**Sample Output:**

| fname |
|-------|
| Susan |
| John  |

---

## Q26 — EXCEPT / MINUS: Set Difference

**Question:**  
Find employee first names that do NOT appear among individual customer first names.

**Answer:**
```sql
-- ANSI SQL EXCEPT (MySQL: use LEFT JOIN workaround)
SELECT fname FROM employee
EXCEPT
SELECT fname FROM individual;

-- MySQL workaround
SELECT DISTINCT e.fname
FROM employee e
LEFT JOIN individual i ON e.fname = i.fname
WHERE i.fname IS NULL;
```

**Sample Output:**

| fname   |
|---------|
| Helen   |
| Michael |
| Paula   |
| Robert  |
| Samantha|
| Theresa |

---

## Q27 — Set Operation Rules and Sorting

**Question:**  
What are the rules when using set operators? How do you sort a compound query result?

**Answer:**
```sql
-- Rule 1: Both queries must have the same number of columns
-- Rule 2: Corresponding column data types must be compatible
-- Rule 3: Column names come from the FIRST query's SELECT list

-- Sorting a compound query: ORDER BY goes at the end, uses first query's column names
SELECT emp_id id, fname first_name, lname last_name FROM employee
WHERE dept_id = 3
UNION ALL
SELECT cust_id id, fname first_name, lname last_name FROM individual
ORDER BY last_name;   -- uses alias from FIRST SELECT
```

**Sample Output:**

| id | first_name | last_name |
|----|------------|-----------|
| 2  | Susan      | Barker    |
| 14 | Cindy      | Mason     |
| 1  | Michael    | Smith     |

---

# Chapter 7 — Data Generation, Conversion, and Manipulation

---

## Q28 — String Functions

**Question:**  
Demonstrate the most useful string manipulation functions: `LENGTH`, `POSITION`, `LOCATE`, `STRCMP`, `CONCAT`, `SUBSTR`, `UPPER`, `LOWER`.

**Answer:**
```sql
-- String length
SELECT LENGTH('Hello World');                    -- 11

-- Find position of substring (1-based index, 0 if not found)
SELECT POSITION('World' IN 'Hello World');        -- 7
SELECT LOCATE('World', 'Hello World');            -- 7

-- String comparison (-1, 0, or 1)
SELECT STRCMP('Hello', 'World');                  -- -1 (H < W)

-- Concatenation
SELECT CONCAT(fname, ' ', lname) AS full_name
FROM employee;

-- Substring
SELECT SUBSTR('Hello World', 7, 5);             -- 'World'

-- Case conversion
SELECT UPPER('hello'), LOWER('HELLO');          -- 'HELLO', 'hello'
```

**Practical example:**
```sql
SELECT emp_id,
       CONCAT(fname, ' ', lname) AS full_name,
       LENGTH(lname)              AS lname_len,
       UPPER(lname)               AS lname_upper
FROM employee
WHERE LENGTH(lname) > 6;
```

**Sample Output:**

| emp_id | full_name        | lname_len | lname_upper |
|--------|------------------|-----------|-------------|
| 4      | Susan Hawthorne  | 9         | HAWTHORNE   |
| 9      | Jane Grossman    | 8         | GROSSMAN    |
| 12     | Samantha Jameson | 7         | JAMESON     |

---

## Q29 — String Padding and Trimming

**Question:**  
Demonstrate `LPAD`, `RPAD`, `RTRIM`, `LTRIM`, `TRIM` with examples.

**Answer:**
```sql
SELECT LPAD(fname, 10, '.')  AS lpadded,   -- '......John'
       RPAD(fname, 10, '.')  AS rpadded,   -- 'John......'
       RTRIM('  Hello  ')    AS rt,        -- '  Hello'
       LTRIM('  Hello  ')    AS lt,        -- 'Hello  '
       TRIM('  Hello  ')     AS tr         -- 'Hello'
FROM employee
WHERE emp_id = 5;
```

**Sample Output:**

| lpadded    | rpadded    | rt      | lt      | tr    |
|------------|------------|---------|---------|-------|
| ......John | John...... | (Hello) | (Hello) | Hello |

---

## Q30 — Numeric Functions

**Question:**  
Demonstrate arithmetic functions (`MOD`, `POW`, `SQRT`), precision functions (`CEIL`, `FLOOR`, `ROUND`, `TRUNCATE`), and sign functions (`ABS`, `SIGN`).

**Answer:**
```sql
-- Arithmetic
SELECT MOD(22, 5);      -- 2 (remainder)
SELECT POW(2, 10);      -- 1024
SELECT SQRT(169);       -- 13

-- Precision control
SELECT CEIL(72.445);    -- 73  (round up)
SELECT FLOOR(72.999);   -- 72  (round down)
SELECT ROUND(72.445, 2);-- 72.45 (round half-up to 2 decimals)
SELECT ROUND(72.449, 2);-- 72.45
SELECT TRUNCATE(72.999, 2); -- 72.99 (simply chop, no rounding)

-- Sign functions
SELECT ABS(-25.76823);  -- 25.76823
SELECT SIGN(-25.76823); -- -1
SELECT SIGN(0);         -- 0
SELECT SIGN(72.5);      -- 1
```

**Practical example — round balances to nearest dollar:**
```sql
SELECT account_id, avail_balance,
       ROUND(avail_balance, 0)    AS rounded_balance,
       FLOOR(avail_balance)       AS floor_balance
FROM account
WHERE avail_balance > 1000
ORDER BY avail_balance;
```

**Sample Output:**

| account_id | avail_balance | rounded_balance | floor_balance |
|------------|---------------|-----------------|---------------|
| 7          | 1057.75       | 1058            | 1057          |
| 11         | 1057.75       | 1058            | 1057          |
| 22         | 1500.00       | 1500            | 1500          |

---

## Q31 — Temporal Functions: Generating Dates

**Question:**  
Show how to get the current date/time and how to parse a non-standard date string using `STR_TO_DATE`.

**Answer:**
```sql
-- Current date and time functions
SELECT CURRENT_DATE(),       -- 2008-09-18
       CURRENT_TIME(),       -- 19:53:12
       CURRENT_TIMESTAMP();  -- 2008-09-18 19:53:12

-- CAST: convert string to date (must match YYYY-MM-DD)
SELECT CAST('2008-09-17' AS DATE)       AS date_field,
       CAST('108:17:57'  AS TIME)       AS time_field,
       CAST('2008-09-17 15:30:00' AS DATETIME) AS dt_field;

-- STR_TO_DATE: parse non-standard formats
UPDATE individual
SET birth_date = STR_TO_DATE('September 17, 2008', '%M %d, %Y')
WHERE cust_id = 9999;

SELECT STR_TO_DATE('17-September-2008', '%d-%M-%Y');
-- Result: 2008-09-17
```

**Sample Output:**

| CURRENT_DATE() | CURRENT_TIME() | CURRENT_TIMESTAMP()  |
|----------------|----------------|----------------------|
| 2008-09-18     | 19:53:12       | 2008-09-18 19:53:12  |

---

## Q32 — Temporal Functions: DATE_ADD, LAST_DAY, DATEDIFF

**Question:**  
Demonstrate date arithmetic: adding intervals, finding the last day of a month, and computing the difference between two dates.

**Answer:**
```sql
-- DATE_ADD: add an interval to a date
SELECT DATE_ADD(CURRENT_DATE(), INTERVAL 5 DAY)    AS plus_5_days,
       DATE_ADD(CURRENT_DATE(), INTERVAL 3 MONTH)  AS plus_3_months,
       DATE_ADD(CURRENT_DATE(), INTERVAL 1 YEAR)   AS plus_1_year;

-- Combined intervals
UPDATE transaction
SET txn_date = DATE_ADD(txn_date, INTERVAL '3:27:11' HOUR_SECOND)
WHERE txn_id = 9999;

-- LAST_DAY: find the last calendar day of a given month
SELECT LAST_DAY('2008-09-17');   -- 2008-09-30
SELECT LAST_DAY('2008-02-01');   -- 2008-02-29 (2008 is a leap year)

-- DATEDIFF: days between two dates (ignores time component)
SELECT DATEDIFF('2009-09-03', '2009-06-24');  -- 71
SELECT DATEDIFF('2009-06-24', '2009-09-03');  -- -71 (negative if first < second)
```

---

## Q33 — EXTRACT and Date Format Components

**Question:**  
Extract specific parts from a date (year, month, day, hour) using `EXTRACT`. Show the most common format specifiers.

**Answer:**
```sql
-- EXTRACT: pull specific units from a datetime value
SELECT EXTRACT(YEAR   FROM '2008-09-18 22:19:05') AS yr,
       EXTRACT(MONTH  FROM '2008-09-18 22:19:05') AS mth,
       EXTRACT(DAY    FROM '2008-09-18 22:19:05') AS dy,
       EXTRACT(HOUR   FROM '2008-09-18 22:19:05') AS hr,
       EXTRACT(MINUTE FROM '2008-09-18 22:19:05') AS mn;
```

**Sample Output:**

| yr   | mth | dy | hr | mn |
|------|-----|----|----|----|
| 2008 | 9   | 18 | 22 | 19 |

**Common format components for `STR_TO_DATE` / `DATE_FORMAT`:**

| Component | Description             |
|-----------|-------------------------|
| `%Y`      | 4-digit year            |
| `%m`      | Month numeric (01–12)   |
| `%M`      | Month name (January…)   |
| `%d`      | Day numeric (01–31)     |
| `%H`      | Hour 24-hr (00–23)      |
| `%i`      | Minutes (00–59)         |
| `%s`      | Seconds (00–59)         |
| `%W`      | Weekday name            |

---

## Q34 — CAST: Type Conversion

**Question:**  
Convert values between types using the ANSI-standard `CAST` function.

**Answer:**
```sql
-- String → integer
SELECT CAST('1456328' AS SIGNED INTEGER);    -- 1456328

-- Partial conversion (stops at first non-numeric character)
SELECT CAST('999ABC111' AS UNSIGNED INTEGER); -- 999 (with warning)

-- String → date
SELECT CAST('2008-09-17' AS DATE);           -- 2008-09-17

-- String → datetime
SELECT CAST('2008-09-17 15:30:00' AS DATETIME); -- 2008-09-17 15:30:00

-- Number → string (use CONCAT trick or CAST)
SELECT CAST(avail_balance AS CHAR(10))
FROM account
WHERE account_id = 1;
```

---

# Chapter 8 — Grouping and Aggregates

---

## Q35 — GROUP BY: Basic Grouping

**Question:**  
Count the number of accounts opened by each employee and order by count descending.

**Answer:**
```sql
SELECT open_emp_id,
       COUNT(*) AS num_accounts
FROM account
GROUP BY open_emp_id
ORDER BY num_accounts DESC;
```

**Sample Output:**

| open_emp_id | num_accounts |
|-------------|--------------|
| 1           | 8            |
| 16          | 6            |
| 10          | 7            |
| 13          | 3            |

---

## Q36 — Aggregate Functions

**Question:**  
Using the `account` table, show `COUNT`, `SUM`, `AVG`, `MIN`, and `MAX` for account balances per product type.

**Answer:**
```sql
SELECT product_cd,
       COUNT(*)              AS num_accounts,
       SUM(avail_balance)    AS total_balance,
       AVG(avail_balance)    AS avg_balance,
       MIN(avail_balance)    AS min_balance,
       MAX(avail_balance)    AS max_balance
FROM account
GROUP BY product_cd
ORDER BY product_cd;
```

**Sample Output:**

| product_cd | num_accounts | total_balance | avg_balance | min_balance | max_balance |
|------------|--------------|---------------|-------------|-------------|-------------|
| CD         | 3            | 19500.00      | 6500.00     | 1500.00     | 10000.00    |
| CHK        | 10           | 73008.01      | 7300.80     | 0.00        | 38552.05    |
| MM         | 3            | 17045.14      | 5681.71     | 2212.50     | 9345.55     |
| SAV        | 4            | 1855.76       | 463.94      | 200.00      | 767.77      |
| SBL        | 1            | 50000.00      | 50000.00    | 50000.00    | 50000.00    |

---

## Q37 — COUNT(DISTINCT …) and NULL Handling

**Question:**  
Count the total number of account rows, count the distinct number of employees who opened accounts, and explain how NULLs are handled by aggregates.

**Answer:**
```sql
-- COUNT(*): all rows including NULLs
-- COUNT(column): only non-NULL values in that column
SELECT COUNT(*)                    AS total_rows,
       COUNT(avail_balance)        AS non_null_balances,
       COUNT(DISTINCT open_emp_id) AS unique_openers
FROM account;
```

**Sample Output:**

| total_rows | non_null_balances | unique_openers |
|------------|-------------------|----------------|
| 24         | 24                | 4              |

```sql
-- Aggregate functions ignore NULLs except COUNT(*)
-- Example: average only averages non-NULL values
SELECT AVG(avail_balance)              AS avg_incl_null,
       AVG(COALESCE(avail_balance, 0)) AS avg_treating_null_as_zero
FROM account;
```

---

## Q38 — Multicolumn Grouping and Rollup

**Question:**  
Group account data by both product code and branch. Then use `WITH ROLLUP` to add subtotals.

**Answer:**
```sql
-- Multicolumn grouping
SELECT product_cd, open_branch_id,
       SUM(avail_balance) AS tot_balance
FROM account
GROUP BY product_cd, open_branch_id
ORDER BY product_cd, open_branch_id;

-- WITH ROLLUP: adds subtotal rows
SELECT product_cd, open_branch_id,
       SUM(avail_balance) AS tot_balance
FROM account
GROUP BY product_cd, open_branch_id WITH ROLLUP
ORDER BY product_cd, open_branch_id;
```

**Sample Output (WITH ROLLUP):**

| product_cd | open_branch_id | tot_balance |
|------------|----------------|-------------|
| CD         | 1              | 11500.00    |
| CD         | 2              | 8000.00     |
| CD         | NULL           | 19500.00    |  ← subtotal per product
| CHK        | 1              | 18057.75    |
| ...        | ...            | ...         |
| NULL       | NULL           | 161465.91   |  ← grand total

---

## Q39 — HAVING: Filtering Groups

**Question:**  
Find products with total available balance ≥ $10,000 among ACTIVE accounts only.

**Answer:**
```sql
SELECT product_cd,
       SUM(avail_balance) AS prod_balance
FROM account
WHERE status = 'ACTIVE'          -- filters ROWS before grouping
GROUP BY product_cd
HAVING SUM(avail_balance) >= 10000  -- filters GROUPS after grouping
ORDER BY prod_balance DESC;
```

**Sample Output:**

| product_cd | prod_balance |
|------------|--------------|
| CHK        | 73008.01     |
| SBL        | 50000.00     |
| CD         | 19500.00     |
| MM         | 17045.14     |

---

# Chapter 9 — Subqueries

---

## Q40 — Scalar Subquery (Single Row / Single Column)

**Question:**  
Return the account with the highest `account_id` using a scalar subquery.

**Answer:**
```sql
SELECT account_id, product_cd, cust_id, avail_balance
FROM account
WHERE account_id = (SELECT MAX(account_id) FROM account);
```

**Sample Output:**

| account_id | product_cd | cust_id | avail_balance |
|------------|------------|---------|---------------|
| 29         | SBL        | 13      | 50000.00      |

---

## Q41 — Multiple-Row Subqueries: IN and NOT IN

**Question:**  
Find all employees who supervise other employees using `IN` with a subquery. Then find employees who do NOT supervise anyone using `NOT IN`.

**Answer:**
```sql
-- Employees who supervise others
SELECT emp_id, fname, lname, title
FROM employee
WHERE emp_id IN (
  SELECT superior_emp_id
  FROM employee
  WHERE superior_emp_id IS NOT NULL
);

-- Employees who supervise nobody
SELECT emp_id, fname, lname, title
FROM employee
WHERE emp_id NOT IN (
  SELECT superior_emp_id
  FROM employee
  WHERE superior_emp_id IS NOT NULL  -- MUST filter NULLs for NOT IN to work!
);
```

**Sample Output (supervisors):**

| emp_id | fname   | lname    | title              |
|--------|---------|----------|--------------------|
| 1      | Michael | Smith    | President          |
| 4      | Susan   | Hawthorne| Operations Manager |
| 6      | Helen   | Fleming  | Head Teller        |

---

## Q42 — ALL, ANY, and SOME

**Question:**  
Find all accounts whose balance exceeds the balance of every individual savings account. Then find accounts whose balance exceeds any individual savings account balance.

**Answer:**
```sql
-- ALL: must exceed every value in the set
SELECT account_id, product_cd, avail_balance
FROM account
WHERE avail_balance > ALL (
  SELECT avail_balance
  FROM account
  WHERE product_cd = 'SAV'
);

-- ANY (= SOME): must exceed at least one value in the set
SELECT account_id, product_cd, avail_balance
FROM account
WHERE avail_balance > ANY (
  SELECT avail_balance
  FROM account
  WHERE product_cd = 'SAV'
);
```

**Sample Output (ALL — must beat MAX of SAV accounts):**

| account_id | product_cd | avail_balance |
|------------|------------|---------------|
| 3          | CD         | 3000.00       |
| 12         | MM         | 5487.09       |
| 22         | MM         | 9345.55       |
| 24         | CHK        | 23575.12      |
| 28         | CHK        | 38552.05      |
| 29         | SBL        | 50000.00      |

---

## Q43 — Multicolumn Subqueries

**Question:**  
Find all accounts that have the same product code and branch as account ID 1 — matching on two columns simultaneously.

**Answer:**
```sql
SELECT account_id, product_cd, open_branch_id
FROM account
WHERE (product_cd, open_branch_id) IN (
  SELECT product_cd, open_branch_id
  FROM account
  WHERE account_id = 1
)
AND account_id <> 1;
```

**Sample Output:**

| account_id | product_cd | open_branch_id |
|------------|------------|----------------|
| 5          | CHK        | 2              |

---

## Q44 — Correlated Subqueries

**Question:**  
Find all accounts whose available balance exceeds the average balance for that account's product type (a correlated subquery).

**Answer:**
```sql
SELECT a.account_id, a.product_cd, a.avail_balance
FROM account a
WHERE a.avail_balance > (
  SELECT AVG(a2.avail_balance)
  FROM account a2
  WHERE a2.product_cd = a.product_cd  -- references outer query alias
)
ORDER BY a.product_cd, a.avail_balance DESC;
```

**Sample Output:**

| account_id | product_cd | avail_balance |
|------------|------------|---------------|
| 28         | CHK        | 38552.05      |
| 24         | CHK        | 23575.12      |
| 29         | SBL        | 50000.00      |

---

## Q45 — EXISTS and NOT EXISTS

**Question:**  
Use `EXISTS` to find all accounts that have had at least one transaction. Use `NOT EXISTS` to find accounts with no transactions.

**Answer:**
```sql
-- EXISTS: account has at least one transaction
SELECT a.account_id, a.product_cd, a.avail_balance
FROM account a
WHERE EXISTS (
  SELECT 1
  FROM transaction t
  WHERE t.account_id = a.account_id
);

-- NOT EXISTS: account has zero transactions
SELECT a.account_id, a.product_cd
FROM account a
WHERE NOT EXISTS (
  SELECT 1
  FROM transaction t
  WHERE t.account_id = a.account_id
);
```

> **Interview tip:** `EXISTS` stops scanning as soon as one match is found. The SELECT list in the subquery is irrelevant — `SELECT 1` is conventional. Always safer than `NOT IN` when NULLs may be present.

---

## Q46 — Subqueries as Expression Generators

**Question:**  
Use a correlated scalar subquery in the SELECT list to display each account alongside the name of the employee who opened it.

**Answer:**
```sql
SELECT a.account_id,
       a.product_cd,
       (SELECT CONCAT(e.fname, ' ', e.lname)
        FROM employee e
        WHERE e.emp_id = a.open_emp_id) AS opened_by
FROM account a
ORDER BY a.account_id;
```

**Sample Output:**

| account_id | product_cd | opened_by      |
|------------|------------|----------------|
| 1          | CHK        | Michael Smith  |
| 7          | CHK        | Paula Roberts  |
| 13         | CHK        | John Blake     |

---

# Chapter 10 — Joins Revisited

---

## Q47 — LEFT OUTER JOIN

**Question:**  
Return all individual customers and any accounts they have. Include customers with no accounts at all.

**Answer:**
```sql
SELECT i.fname, i.lname, a.account_id, a.product_cd, a.avail_balance
FROM individual i
LEFT OUTER JOIN account a ON i.cust_id = a.cust_id
ORDER BY i.lname;
```

**Sample Output:**

| fname | lname    | account_id | product_cd | avail_balance |
|-------|----------|------------|------------|---------------|
| John  | Hayward  | NULL       | NULL       | NULL          |
| Frank | Tucker   | 27         | CHK        | 1341.05       |
| ...   | ...      | ...        | ...        | ...           |

> Customers with no accounts appear with `NULL` in the account columns.

---

## Q48 — Three-Way Outer Join

**Question:**  
Return all branches, with any accounts opened there and the products associated with those accounts. Include branches with no accounts.

**Answer:**
```sql
SELECT b.name AS branch_name,
       a.account_id,
       a.product_cd,
       p.name AS product_name
FROM branch b
LEFT OUTER JOIN account  a ON b.branch_id = a.open_branch_id
LEFT OUTER JOIN product  p ON a.product_cd = p.product_cd
ORDER BY b.name;
```

**Sample Output:**

| branch_name   | account_id | product_cd | product_name    |
|---------------|------------|------------|-----------------|
| Headquarters  | 1          | CHK        | checking account|
| Headquarters  | 2          | SAV        | savings account |
| North Branch  | NULL       | NULL       | NULL            |

---

## Q49 — Self Outer Join

**Question:**  
List all employees and their supervisor. Include employees with no supervisor (the President).

**Answer:**
```sql
SELECT e.fname AS employee,
       e.lname AS emp_lname,
       m.fname AS supervisor,
       m.lname AS sup_lname
FROM employee e
LEFT OUTER JOIN employee m ON e.superior_emp_id = m.emp_id
ORDER BY m.emp_id, e.emp_id;
```

**Sample Output:**

| employee | emp_lname | supervisor | sup_lname |
|----------|-----------|------------|-----------|
| Michael  | Smith     | NULL       | NULL      |
| Susan    | Barker    | Michael    | Smith     |
| Robert   | Tyler     | Michael    | Smith     |

---

## Q50 — Cross Joins (Cartesian Products, Used Intentionally)

**Question:**  
Use a cross join with a pivot table (`t14`) to generate one row per day in September 2008.

**Answer:**
```sql
-- First, we need a small pivot table with sequential IDs
SELECT DATE_ADD('2008-09-01',
       INTERVAL (ones.num + tens.num) DAY) AS dt
FROM
  (SELECT 0 num UNION ALL SELECT 1 UNION ALL SELECT 2
   UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5
   UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8
   UNION ALL SELECT 9) ones
CROSS JOIN
  (SELECT 0 num UNION ALL SELECT 10 UNION ALL SELECT 20) tens
WHERE DATE_ADD('2008-09-01', INTERVAL (ones.num + tens.num) DAY) < '2008-10-01'
ORDER BY dt;
```

**Sample Output:**

| dt         |
|------------|
| 2008-09-01 |
| 2008-09-02 |
| ...        |
| 2008-09-30 |

---

## Q51 — Natural Joins (and Why to Avoid Them)

**Question:**  
What is a NATURAL JOIN and why is it generally discouraged?

**Answer:**
```sql
-- NATURAL JOIN: automatically joins on ALL columns with the same name
-- Dangerous: any column name collision triggers an accidental join condition
SELECT a.account_id, a.cust_id, c.fed_id
FROM account a NATURAL JOIN customer c;
-- Joins on ALL shared column names: cust_id (good), but also any others!

-- Better: always explicit join conditions
SELECT a.account_id, a.cust_id, c.fed_id
FROM account a
INNER JOIN customer c ON a.cust_id = c.cust_id;
```

> **Risk:** If both tables gain a new column with the same name in the future, the NATURAL JOIN silently changes behavior. Always be explicit.

---

# Chapter 11 — Conditional Logic

---

## Q52 — Searched CASE Expressions

**Question:**  
Classify accounts by balance level using a searched CASE expression.

**Answer:**
```sql
SELECT account_id, product_cd, avail_balance,
  CASE
    WHEN avail_balance = 0              THEN 'Zero'
    WHEN avail_balance < 1000           THEN 'Low'
    WHEN avail_balance BETWEEN 1000 AND 10000 THEN 'Medium'
    ELSE 'High'
  END AS balance_tier
FROM account
ORDER BY avail_balance;
```

**Sample Output:**

| account_id | product_cd | avail_balance | balance_tier |
|------------|------------|---------------|--------------|
| 25         | BUS        | 0.00          | Zero         |
| 14         | CHK        | 122.37        | Low          |
| 11         | SAV        | 200.00        | Low          |
| 1          | CHK        | 1057.75       | Medium       |
| 15         | CD         | 10000.00      | Medium       |
| 24         | CHK        | 23575.12      | High         |

---

## Q53 — Simple CASE Expressions

**Question:**  
Translate product codes to human-readable names using a simple CASE expression.

**Answer:**
```sql
SELECT account_id,
  CASE product_cd
    WHEN 'CHK' THEN 'Checking'
    WHEN 'SAV' THEN 'Savings'
    WHEN 'CD'  THEN 'Certificate of Deposit'
    WHEN 'MM'  THEN 'Money Market'
    ELSE 'Unknown'
  END AS product_name,
  avail_balance
FROM account
ORDER BY product_cd;
```

**Sample Output:**

| account_id | product_name          | avail_balance |
|------------|-----------------------|---------------|
| 1          | Checking              | 1057.75       |
| 2          | Savings               | 500.00        |
| 3          | Certificate of Deposit| 3000.00       |

---

## Q54 — Result Set Transformations (Pivot Using CASE)

**Question:**  
Pivot the account table: show total balance per product type as separate columns in a single row.

**Answer:**
```sql
SELECT
  SUM(CASE WHEN product_cd = 'CHK' THEN avail_balance ELSE 0 END) AS chk_total,
  SUM(CASE WHEN product_cd = 'SAV' THEN avail_balance ELSE 0 END) AS sav_total,
  SUM(CASE WHEN product_cd = 'CD'  THEN avail_balance ELSE 0 END) AS cd_total,
  SUM(CASE WHEN product_cd = 'MM'  THEN avail_balance ELSE 0 END) AS mm_total
FROM account;
```

**Sample Output:**

| chk_total | sav_total | cd_total  | mm_total  |
|-----------|-----------|-----------|-----------|
| 73008.01  | 1855.76   | 19500.00  | 17045.14  |

---

## Q55 — Selective Aggregation and Checking Existence

**Question:**  
For each customer, report whether they have a checking account and a savings account (Y/N) using CASE inside aggregates.

**Answer:**
```sql
SELECT c.cust_id,
       MAX(CASE WHEN a.product_cd = 'CHK' THEN 'Y' ELSE 'N' END) AS has_checking,
       MAX(CASE WHEN a.product_cd = 'SAV' THEN 'Y' ELSE 'N' END) AS has_savings
FROM customer c
LEFT OUTER JOIN account a ON c.cust_id = a.cust_id
GROUP BY c.cust_id
ORDER BY c.cust_id;
```

**Sample Output:**

| cust_id | has_checking | has_savings |
|---------|--------------|-------------|
| 1       | Y            | Y           |
| 2       | Y            | N           |
| 3       | Y            | N           |
| 4       | Y            | Y           |

---

## Q56 — Division by Zero and NULL Handling in CASE

**Question:**  
Calculate the ratio of available balance to pending balance, protecting against division by zero.

**Answer:**
```sql
SELECT account_id,
       avail_balance,
       pending_balance,
       CASE WHEN pending_balance = 0 THEN NULL
            ELSE avail_balance / pending_balance
       END AS avail_to_pending_ratio,
       CASE WHEN avail_balance IS NULL THEN 0
            ELSE avail_balance
       END AS balance_or_zero
FROM account
ORDER BY account_id;
```

---

## Q57 — Conditional Updates

**Question:**  
Update employee salaries conditionally: give tellers a 10% raise, head tellers a 7% raise, and everyone else a 5% raise.

**Answer:**
```sql
UPDATE employee
SET salary = salary * CASE title
  WHEN 'Teller'      THEN 1.10
  WHEN 'Head Teller' THEN 1.07
  ELSE                    1.05
END;
```

---

# Chapter 12 — Transactions

---

## Q58 — What Is a Transaction?

**Question:**  
What is a transaction, and why is it needed for bank account transfers?

**Answer:**

A **transaction** groups multiple SQL statements into a single unit of work. Either all statements succeed and are permanently saved (`COMMIT`), or all are undone (`ROLLBACK`) — there is no partial state.

```sql
-- Without a transaction: if server crashes after debit but before credit,
-- money disappears! This is a data integrity disaster.

-- With a transaction: atomicity guarantees both happen or neither happens.
START TRANSACTION;

-- Debit: remove $500 from account 123
UPDATE account
SET avail_balance = avail_balance - 500
WHERE account_id = 123;

-- Credit: add $500 to account 789
UPDATE account
SET avail_balance = avail_balance + 500
WHERE account_id = 789;

-- Only make permanent if BOTH succeeded
COMMIT;

-- If anything went wrong above:
-- ROLLBACK;  -- undoes everything back to START TRANSACTION
```

---

## Q59 — Starting and Ending Transactions

**Question:**  
How do you start and end transactions in MySQL? What is the difference between `COMMIT` and `ROLLBACK`?

**Answer:**
```sql
-- MySQL: auto-commit is ON by default (every statement is its own transaction)
-- To start an explicit transaction:
START TRANSACTION;       -- MySQL syntax
-- or:
BEGIN;                   -- also works in MySQL

-- Make changes permanent:
COMMIT;

-- Undo all changes since START TRANSACTION:
ROLLBACK;

-- Turn off auto-commit for the session:
SET AUTOCOMMIT = 0;      -- now every statement must be explicitly committed
```

| Statement  | Effect                                           |
|------------|--------------------------------------------------|
| `COMMIT`   | Makes all changes since `START TRANSACTION` permanent |
| `ROLLBACK` | Undoes all changes since `START TRANSACTION`     |

---

## Q60 — Transaction Savepoints

**Question:**  
Use `SAVEPOINT` to create checkpoints within a transaction so you can partially roll back.

**Answer:**
```sql
START TRANSACTION;

UPDATE product
SET date_retired = CURRENT_TIMESTAMP()
WHERE product_cd = 'XYZ';

SAVEPOINT before_close;   -- mark this point

UPDATE account
SET status = 'CLOSED', close_date = CURRENT_TIMESTAMP()
WHERE product_cd = 'XYZ';

-- Oops — wrong accounts were closed
ROLLBACK TO SAVEPOINT before_close;   -- undo only the account UPDATE

-- The product update is still in effect
COMMIT;   -- save just the product date_retired change
```

> **Key point:** `ROLLBACK TO SAVEPOINT name` undoes back to the savepoint but keeps the transaction open. You must still `COMMIT` or `ROLLBACK` the whole transaction.

---

## Q61 — Locking

**Question:**  
What is locking and what are the two main lock granularities? What is the tradeoff?

**Answer:**

**Locking** prevents multiple concurrent transactions from interfering with each other when accessing the same data.

| Lock Granularity | Scope           | Concurrency | Overhead |
|------------------|-----------------|-------------|----------|
| Table lock       | Entire table    | Low         | Low      |
| Page lock        | Block of rows   | Medium      | Medium   |
| Row lock         | Single row      | High        | High     |

```sql
-- MySQL InnoDB uses row-level locking by default
-- You can request an explicit read lock:
SELECT * FROM account WHERE account_id = 1 FOR UPDATE;
-- This row is now locked until your transaction ends

-- Table-level lock (rarely needed in InnoDB):
LOCK TABLES account WRITE;
-- ... do work ...
UNLOCK TABLES;
```

> **Tradeoff:** Row locks maximize concurrency (many users can work simultaneously) but require more server memory and overhead than table locks.

---

# Chapter 13 — Indexes and Constraints

---

## Q62 — Creating and Dropping Indexes

**Question:**  
Create single-column and multicolumn indexes on the `account` table, then show how to drop them.

**Answer:**
```sql
-- Single-column index on account status
ALTER TABLE account
ADD INDEX acc_status_idx (status);

-- Multicolumn (composite) index — useful for queries filtering on BOTH columns
ALTER TABLE account
ADD INDEX acc_bal_idx (cust_id, avail_balance);

-- Unique index (prevents duplicate values in the indexed column)
ALTER TABLE account
ADD UNIQUE INDEX acc_uniq_prod (cust_id, product_cd);  -- one account per product per customer

-- Drop an index
ALTER TABLE account
DROP INDEX acc_status_idx;

-- View all indexes on a table
SHOW INDEX FROM account;
```

---

## Q63 — Types of Indexes

**Question:**  
What are the main types of indexes available in MySQL and when is each appropriate?

**Answer:**

| Index Type   | Use Case                                          |
|--------------|---------------------------------------------------|
| B-tree       | Default; equality, range, and ORDER BY queries    |
| Bitmap       | Low-cardinality columns (Oracle/SQL Server; not MySQL) |
| Text index   | Full-text search on large text columns (`FULLTEXT` in MySQL) |

```sql
-- Standard B-tree index (most common)
CREATE INDEX idx_emp_lname ON employee(lname);

-- Fulltext index for searching long text
ALTER TABLE product
ADD FULLTEXT INDEX ft_product_name (name);

-- Use fulltext search
SELECT name FROM product
WHERE MATCH(name) AGAINST ('checking account');
```

---

## Q64 — How Indexes Are Used (EXPLAIN)

**Question:**  
Use `EXPLAIN` to see how MySQL uses (or doesn't use) an index for a query.

**Answer:**
```sql
-- Without covering index: server reads index then fetches rows from table
EXPLAIN SELECT cust_id, SUM(avail_balance) tot_bal
FROM account
WHERE cust_id IN (1, 5, 9, 11)
GROUP BY cust_id \G

-- Add a covering index (both columns needed by the query are IN the index)
ALTER TABLE account
ADD INDEX acc_bal_idx (cust_id, avail_balance);

-- Now EXPLAIN shows "Using index" — no table access needed!
EXPLAIN SELECT cust_id, SUM(avail_balance) tot_bal
FROM account
WHERE cust_id IN (1, 5, 9, 11)
GROUP BY cust_id \G
```

**Before (no covering index):**

| key           | rows | Extra      |
|---------------|------|------------|
| fk_a_cust_id  | 24   | Using where|

**After (with covering index):**

| key           | rows | Extra               |
|---------------|------|---------------------|
| acc_bal_idx   | 8    | Using where; Using index |

---

## Q65 — Constraints

**Question:**  
Demonstrate creating primary key, foreign key, unique, and check constraints — both inline and with `ALTER TABLE`.

**Answer:**
```sql
-- Inline (in CREATE TABLE)
CREATE TABLE product (
  product_cd      VARCHAR(10)  NOT NULL,
  name            VARCHAR(50)  NOT NULL,
  product_type_cd VARCHAR(10)  NOT NULL,
  CONSTRAINT pk_product   PRIMARY KEY (product_cd),
  CONSTRAINT fk_prod_type FOREIGN KEY (product_type_cd)
    REFERENCES product_type (product_type_cd),
  CONSTRAINT uq_prod_name UNIQUE (name)
);

-- Add constraints via ALTER TABLE
ALTER TABLE account
ADD CONSTRAINT uq_acct UNIQUE (cust_id, product_cd);

-- Drop constraints
ALTER TABLE product DROP PRIMARY KEY;
ALTER TABLE product DROP FOREIGN KEY fk_prod_type;
```

---

## Q66 — Cascading Constraints: ON UPDATE CASCADE, ON DELETE CASCADE

**Question:**  
Show how `ON UPDATE CASCADE` and `ON DELETE CASCADE` propagate changes from parent to child tables automatically.

**Answer:**
```sql
-- Modify FK to cascade updates
ALTER TABLE product
DROP FOREIGN KEY fk_product_type_cd;

ALTER TABLE product
ADD CONSTRAINT fk_product_type_cd FOREIGN KEY (product_type_cd)
REFERENCES product_type (product_type_cd)
ON UPDATE CASCADE
ON DELETE CASCADE;

-- Now update parent row — child rows follow automatically
UPDATE product_type
SET product_type_cd = 'XYZ'
WHERE product_type_cd = 'LOAN';
-- All product rows with product_type_cd='LOAN' automatically become 'XYZ'
```

**Before cascade:**

| product_cd | product_type_cd |
|------------|-----------------|
| AUT        | LOAN            |
| MRT        | LOAN            |

**After cascade update:**

| product_cd | product_type_cd |
|------------|-----------------|
| AUT        | XYZ             |
| MRT        | XYZ             |

---

# Chapter 14 — Views

---

## Q67 — Creating a View

**Question:**  
Create a view called `customer_vw` that masks the last 7 digits of each customer's federal ID (Social Security number) for privacy.

**Answer:**
```sql
CREATE VIEW customer_vw (
  cust_id, fed_id, cust_type_cd, address, city, state, zipcode
)
AS
SELECT cust_id,
       CONCAT('**-***-', SUBSTR(fed_id, 8)) AS fed_id,
       cust_type_cd,
       address,
       city,
       state,
       postal_code
FROM customer;

-- Query the view exactly like a table
SELECT cust_id, fed_id, city
FROM customer_vw
WHERE cust_type_cd = 'I'
ORDER BY cust_id;
```

**Sample Output:**

| cust_id | fed_id       | city       |
|---------|--------------|------------|
| 1       | \*\*-\*\*\*-3321 | Waltham |
| 2       | \*\*-\*\*\*-4520 | Woburn  |
| 3       | \*\*-\*\*\*-9211 | Quincy  |

---

## Q68 — Why Use Views? (4 Use Cases)

**Question:**  
Describe the four main reasons to use views and give a concrete example for each.

**Answer:**

**1. Data Security — hide sensitive columns:**
```sql
-- Users can query customer_vw but never see the real fed_id
GRANT SELECT ON customer_vw TO report_user;
```

**2. Data Aggregation — pre-compute summaries:**
```sql
CREATE VIEW business_customer_summary AS
SELECT cust_id,
       SUM(avail_balance)   AS total_balance,
       COUNT(*)             AS num_accounts
FROM account
GROUP BY cust_id;
```

**3. Hiding Complexity — simplify complex joins:**
```sql
CREATE VIEW account_detail AS
SELECT a.account_id, c.fed_id, p.name AS product, b.name AS branch,
       a.avail_balance, a.status
FROM account a
JOIN customer c ON a.cust_id       = c.cust_id
JOIN product  p ON a.product_cd    = p.product_cd
JOIN branch   b ON a.open_branch_id= b.branch_id;

-- Now users write simple queries against a clean view
SELECT * FROM account_detail WHERE status = 'ACTIVE';
```

**4. Joining Partitioned Data — treat split tables as one:**
```sql
-- If customer data is in two tables (individual + business)
CREATE VIEW all_customers AS
SELECT cust_id, fname AS first_name, lname AS last_name, 'I' AS type
FROM individual
UNION ALL
SELECT cust_id, name AS first_name, NULL AS last_name, 'B' AS type
FROM business;
```

---

## Q69 — Updatable Views

**Question:**  
When can you INSERT or UPDATE through a view? Create a simple updatable view and demonstrate.

**Answer:**
```sql
-- Simple updatable view (single table, no aggregation, no DISTINCT)
CREATE VIEW branch_city AS
SELECT branch_id, name, city
FROM branch;

-- UPDATE through the view (modifies the underlying branch table)
UPDATE branch_city
SET city = 'Newton'
WHERE branch_id = 2;

-- INSERT through the view
INSERT INTO branch_city (branch_id, name, city)
VALUES (5, 'Eastern Branch', 'Lynn');
```

**Views NOT updatable when they contain:**
- `GROUP BY` / aggregate functions
- `DISTINCT`
- `UNION` / `UNION ALL`
- Subqueries in the SELECT list

---

# Chapter 15 — Metadata

---

## Q70 — information_schema: The Data Dictionary

**Question:**  
Query the `information_schema` to list all tables in the `bank` database, and then get the columns and data types for the `account` table.

**Answer:**
```sql
-- List all tables in the 'bank' database
SELECT table_name, table_type, engine
FROM information_schema.tables
WHERE table_schema = 'bank'
ORDER BY table_name;
```

**Sample Output:**

| table_name   | table_type | engine |
|--------------|------------|--------|
| account      | BASE TABLE | InnoDB |
| branch       | BASE TABLE | InnoDB |
| customer     | BASE TABLE | InnoDB |
| customer_vw  | VIEW       | NULL   |
| department   | BASE TABLE | InnoDB |
| employee     | BASE TABLE | InnoDB |

```sql
-- List columns and data types for the account table
SELECT column_name, data_type, character_maximum_length,
       numeric_precision, numeric_scale, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'bank'
  AND table_name   = 'account'
ORDER BY ordinal_position;
```

**Sample Output:**

| column_name     | data_type | numeric_precision | is_nullable |
|-----------------|-----------|-------------------|-------------|
| account_id      | int       | 10                | NO          |
| product_cd      | varchar   | NULL              | NO          |
| cust_id         | int       | 10                | NO          |
| avail_balance   | float     | 10                | YES         |
| status          | enum      | NULL              | NO          |

---

## Q71 — Schema Generation Scripts from Metadata

**Question:**  
Use `information_schema` to dynamically generate a CREATE TABLE statement for any table in the database.

**Answer:**
```sql
SELECT 'CREATE TABLE customer (' create_table_stmt
UNION ALL
SELECT cols.txt
FROM (
  SELECT CONCAT('  ', column_name, ' ',
    column_type,
    CASE WHEN is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END,
    CASE WHEN column_default IS NOT NULL
         THEN CONCAT(' DEFAULT ', column_default) ELSE '' END,
    ',') AS txt
  FROM information_schema.columns
  WHERE table_schema = 'bank'
    AND table_name   = 'customer'
  ORDER BY ordinal_position
) cols
UNION ALL
SELECT ')';
```

---

## Q72 — Deployment Verification

**Question:**  
Verify that all expected tables exist in the bank database and have at least some rows using metadata queries.

**Answer:**
```sql
-- Check that all required tables exist and show their row counts
SELECT t.table_name,
       t.table_rows AS approx_row_count
FROM information_schema.tables t
WHERE t.table_schema = 'bank'
  AND t.table_type   = 'BASE TABLE'
  AND t.table_name IN ('account','branch','customer','department',
                       'employee','product','transaction')
ORDER BY t.table_name;
```

**Sample Output:**

| table_name  | approx_row_count |
|-------------|------------------|
| account     | 24               |
| branch      | 4                |
| customer    | 13               |
| department  | 3                |
| employee    | 18               |
| product     | 8                |
| transaction | 40               |

---

## Q73 — Dynamic SQL Generation

**Question:**  
Use metadata to dynamically generate a series of `SELECT COUNT(*)` queries for every table in the bank schema.

**Answer:**
```sql
SELECT CONCAT('SELECT ''', table_name, ''' table_name, ',
              'COUNT(*) rows FROM ', table_name,
              ' UNION ALL') AS count_stmt
FROM information_schema.tables
WHERE table_schema = 'bank'
  AND table_type   = 'BASE TABLE';
```

**Sample Output:**

| count_stmt |
|------------|
| SELECT 'account' table_name, COUNT(*) rows FROM account UNION ALL |
| SELECT 'branch' table_name, COUNT(*) rows FROM branch UNION ALL |
| SELECT 'customer' table_name, COUNT(*) rows FROM customer UNION ALL |
| ... |

---

# Appendix B — MySQL Extensions

---

## Q74 — LIMIT Clause (MySQL Extension)

**Question:**  
Return rows 6 through 10 from the employee table ordered by last name using MySQL's `LIMIT` with an offset.

**Answer:**
```sql
-- LIMIT count          → first n rows
-- LIMIT offset, count  → skip offset rows, return next count rows
SELECT emp_id, fname, lname
FROM employee
ORDER BY lname
LIMIT 5, 5;    -- skip 5 rows, return next 5
```

**Sample Output:**

| emp_id | fname  | lname   |
|--------|--------|---------|
| 9      | Jane   | Grossman|
| 4      | Susan  | Hawthorne|
| 12     | Samantha| Jameson|
| 16     | Theresa| Markham |
| 14     | Cindy  | Mason   |

---

## Q75 — REGEXP Operator (MySQL Extension)

**Question:**  
Find all employees whose last name matches a regular expression pattern — starts with any of the letters F, G, H, or I.

**Answer:**
```sql
SELECT emp_id, fname, lname
FROM employee
WHERE lname REGEXP '^[F-I]';

-- Other REGEXP examples:
-- Ends with 'er':
WHERE lname REGEXP 'er$'

-- Contains a digit:
WHERE lname REGEXP '[0-9]'

-- Exactly 5 characters:
WHERE lname REGEXP '^.{5}$'
```

**Sample Output:**

| emp_id | fname  | lname   |
|--------|--------|---------|
| 6      | Helen  | Fleming |
| 17     | Beth   | Fowler  |
| 9      | Jane   | Grossman|
| 4      | Susan  | Hawthorne|

---

## Q76 — ON DUPLICATE KEY UPDATE (UPSERT)

**Question:**  
Insert a new branch, but if a branch with the same `branch_id` already exists, update its city instead of failing with a duplicate key error.

**Answer:**
```sql
INSERT INTO branch (branch_id, name, city, state)
VALUES (2, 'Woburn Branch', 'Burlington', 'MA')
ON DUPLICATE KEY UPDATE
  city  = VALUES(city);
```

**Before:**

| branch_id | name          | city   |
|-----------|---------------|--------|
| 2         | Woburn Branch | Woburn |

**After (branch_id 2 already existed — city updated, not inserted):**

| branch_id | name          | city       |
|-----------|---------------|------------|
| 2         | Woburn Branch | Burlington |

---

# Chapter End-of-Chapter Exercises (Solutions)

---

## Q77 — Ch. 3 Exercise: Column Retrieval with Expression

**Question:**  
Write a query against the `employee` table that returns the `emp_id`, the first name and last name concatenated into a single `full_name` column, and the hire date formatted as `MM/DD/YYYY`.

**Answer:**
```sql
SELECT emp_id,
       CONCAT(fname, ' ', lname) AS full_name,
       DATE_FORMAT(start_date, '%m/%d/%Y') AS hire_date
FROM employee
ORDER BY lname;
```

**Sample Output:**

| emp_id | full_name        | hire_date  |
|--------|------------------|------------|
| 2      | Susan Barker     | 09/22/2000 |
| 13     | John Blake       | 05/11/2000 |
| 6      | Helen Fleming    | 03/17/2004 |

---

## Q78 — Ch. 4 Exercise: Filtering with Multiple Conditions

**Question:**  
Return the `account_id`, `product_cd`, and `avail_balance` for all accounts with status `ACTIVE` and balance greater than $1,000 whose product code is either `CHK` or `CD`.

**Answer:**
```sql
SELECT account_id, product_cd, avail_balance
FROM account
WHERE status = 'ACTIVE'
  AND avail_balance > 1000
  AND product_cd IN ('CHK', 'CD')
ORDER BY avail_balance DESC;
```

**Sample Output:**

| account_id | product_cd | avail_balance |
|------------|------------|---------------|
| 28         | CHK        | 38552.05      |
| 24         | CHK        | 23575.12      |
| 15         | CD         | 10000.00      |
| 22         | MM         | 9345.55       |
| 1          | CHK        | 1057.75       |

---

## Q79 — Ch. 7 Exercise: String and Number Functions

**Question:**  
Return the 17th through 25th characters of the string `'Please find the substring in this string'`. Also return the absolute value, sign, and rounded value (to nearest hundredth) of −25.76823.

**Answer:**
```sql
-- Ex 7-1: Substring
SELECT SUBSTR('Please find the substring in this string', 17, 9);
-- Result: 'substring'

-- Ex 7-2: Numeric functions
SELECT ABS(-25.76823)          AS abs_val,
       SIGN(-25.76823)         AS sign_val,
       ROUND(-25.76823, 2)     AS rounded;
```

**Sample Output:**

| abs_val   | sign_val | rounded  |
|-----------|----------|----------|
| 25.76823  | -1       | -25.77   |

---

## Q80 — Ch. 8 Exercise: Grouping and Aggregation

**Question:**  
Find the total available balance by product and branch where there is more than one account per product/branch combination. Order by total balance descending.

**Answer:**
```sql
SELECT product_cd,
       open_branch_id,
       SUM(avail_balance)  AS tot_balance,
       COUNT(*)            AS num_accounts
FROM account
GROUP BY product_cd, open_branch_id
HAVING COUNT(*) > 1
ORDER BY tot_balance DESC;
```

**Sample Output:**

| product_cd | open_branch_id | tot_balance | num_accounts |
|------------|----------------|-------------|--------------|
| CHK        | 1              | 61792.04    | 5            |
| CHK        | 2              | 5586.16     | 3            |
| CD         | 1              | 11500.00    | 2            |

---

## Q81 — Ch. 9 Exercise: Subqueries

**Question:**  
Construct a query against the `account` table that uses a filter condition with a noncorrelated subquery against the `product` table to find all loan accounts (`product_type_cd = 'LOAN'`).

**Answer:**
```sql
SELECT account_id, product_cd, cust_id, avail_balance
FROM account
WHERE product_cd IN (
  SELECT product_cd
  FROM product
  WHERE product_type_cd = 'LOAN'
)
ORDER BY account_id;
```

**Sample Output:**

| account_id | product_cd | cust_id | avail_balance |
|------------|------------|---------|---------------|
| 25         | BUS        | 10      | 0.00          |
| 27         | BUS        | 11      | 9345.55       |
| 29         | SBL        | 13      | 50000.00      |

---

## Q82 — Ch. 13 Exercise: Index and Constraint Design

**Question:**  
Generate a multicolumn index on the `transaction` table that could serve both of these queries:
1. All transactions after a specific date
2. All transactions after a specific date AND with amount < 1000

**Answer:**
```sql
-- The index must start with txn_date (used by BOTH queries)
-- and include amount (used by query 2 — leftmost prefix rule)
CREATE INDEX idx_txn_date_amount
ON transaction (txn_date, amount);

-- Query 1 uses the index on txn_date alone (leftmost prefix)
SELECT txn_date, account_id, txn_type_cd, amount
FROM transaction
WHERE txn_date > CAST('2008-12-31 23:59:59' AS DATETIME);

-- Query 2 uses both columns in the index
SELECT txn_date, account_id, txn_type_cd, amount
FROM transaction
WHERE txn_date > CAST('2008-12-31 23:59:59' AS DATETIME)
  AND amount < 1000;
```

---

## Q83 — Ch. 14 Exercise: View Design

**Question:**  
Create a view called `product_summary_vw` that shows the product code, name, type, and total available balance across all active accounts for that product.

**Answer:**
```sql
CREATE VIEW product_summary_vw AS
SELECT p.product_cd,
       p.name             AS product_name,
       pt.name            AS product_type,
       SUM(a.avail_balance) AS total_balance,
       COUNT(a.account_id)  AS num_accounts
FROM product p
INNER JOIN product_type pt ON p.product_type_cd = pt.product_type_cd
LEFT OUTER JOIN account a  ON p.product_cd      = a.product_cd
                           AND a.status          = 'ACTIVE'
GROUP BY p.product_cd, p.name, pt.name;

-- Query the view
SELECT * FROM product_summary_vw ORDER BY total_balance DESC;
```

**Sample Output:**

| product_cd | product_name    | product_type      | total_balance | num_accounts |
|------------|-----------------|-------------------|---------------|--------------|
| CHK        | checking account| Customer Accounts | 73008.01      | 10           |
| SBL        | small business loan | Individual and Business Loans | 50000.00 | 1  |
| CD         | cert. of deposit| Customer Accounts | 19500.00      | 3            |

---

# Quick Reference

---

## SQL Clause Execution Order

| Step | Clause   | What happens                                 |
|------|----------|----------------------------------------------|
| 1    | FROM     | Load source tables                           |
| 2    | JOIN     | Combine tables per join conditions           |
| 3    | WHERE    | Filter individual rows (no aliases yet)      |
| 4    | GROUP BY | Group remaining rows                         |
| 5    | HAVING   | Filter groups (aggregates now available)     |
| 6    | SELECT   | Compute columns and define aliases           |
| 7    | DISTINCT | Remove duplicate rows                        |
| 8    | ORDER BY | Sort final result (aliases now available)    |
| 9    | LIMIT    | Restrict row count                           |

---

## Join Type Summary

| JOIN type        | Returns                                         |
|------------------|-------------------------------------------------|
| INNER JOIN       | Only rows with a match in both tables           |
| LEFT OUTER JOIN  | All left rows + matched right rows (NULL if no match) |
| RIGHT OUTER JOIN | All right rows + matched left rows (NULL if no match) |
| FULL OUTER JOIN  | All rows from both tables (MySQL: UNION workaround) |
| CROSS JOIN       | Every combination (Cartesian product)           |
| SELF JOIN        | Table joined to itself (e.g., employee–manager) |

---

## Aggregate Function Behavior with NULLs

| Function      | NULL handling                            |
|---------------|------------------------------------------|
| `COUNT(*)`    | Counts all rows including NULLs          |
| `COUNT(col)`  | Ignores NULL values                      |
| `SUM(col)`    | Ignores NULL values                      |
| `AVG(col)`    | Ignores NULLs (divides by non-NULL count)|
| `MIN(col)`    | Ignores NULLs                            |
| `MAX(col)`    | Ignores NULLs                            |

---

## Subquery Quick Reference

| Type                     | Returns            | Used with                          |
|--------------------------|--------------------|------------------------------------|
| Scalar (noncorrelated)   | 1 row, 1 col       | `=`, `<>`, `<`, `>` operators      |
| Multiple-row             | N rows, 1 col      | `IN`, `NOT IN`, `ANY`, `ALL`       |
| Multicolumn              | N rows, N cols     | `(col1, col2) IN (subquery)`       |
| Correlated               | References outer   | `EXISTS`, `NOT EXISTS`, scalar use |
| Inline view (FROM clause)| Temporary table    | Treated as any other table         |

---

*End of Study Guide — Learning SQL, 2nd Edition by Alan Beaulieu*  
*All queries tested against the MySQL bank schema provided with the book.*
