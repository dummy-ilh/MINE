# Microsoft SQL Server 2012 T-SQL Fundamentals
### By Itzik Ben-Gan — Comprehensive Study Notes

---

## Table of Contents
1. [Chapter 1: Background to T-SQL Querying and Programming](#chapter-1)
2. [Chapter 2: Single-Table Queries](#chapter-2)
3. [Chapter 3: Joins](#chapter-3)
4. [Chapter 4: Subqueries](#chapter-4)
5. [Chapter 5: Table Expressions](#chapter-5)
6. [Chapter 6: Set Operators](#chapter-6)
7. [Chapter 7: Beyond the Fundamentals of Querying](#chapter-7)
8. [Chapter 8: Data Modification](#chapter-8)
9. [Chapter 9: Transactions and Concurrency](#chapter-9)
10. [Chapter 10: Programmable Objects](#chapter-10)
11. [Appendix A: Getting Started](#appendix)

---

## Chapter 1: Background to T-SQL Querying and Programming {#chapter-1}

### 1.1 Theoretical Background

#### SQL
- SQL stands for **Structured Query Language** — a standard language for querying and managing data in relational database management systems (RDBMSs).
- Originally developed by IBM as **SEQUEL** (Structured English QUEry Language) for System R in the early 1970s; renamed SQL due to a trademark dispute.
- Became an ANSI standard in **1986**, ISO standard in **1987**.
- Major SQL standard versions: SQL-86, SQL-89, SQL-92, SQL:1999, SQL:2003, SQL:2006, SQL:2008, SQL:2011.
- SQL uses a **declarative paradigm** — you specify *what* you want, not *how* to get it. The RDBMS figures out the physical execution.
- T-SQL is Microsoft's **dialect/extension** of the SQL standard used in SQL Server.

**SQL Statement Categories:**
- **DDL (Data Definition Language):** `CREATE`, `ALTER`, `DROP` — deals with object definitions.
- **DML (Data Manipulation Language):** `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`, `MERGE` — deals with data retrieval and modification. Note: SELECT is DML, not just modification statements. TRUNCATE is DML, not DDL.
- **DCL (Data Control Language):** `GRANT`, `REVOKE` — deals with permissions.

#### Set Theory
- The relational model is rooted in **set theory**, developed by mathematician Georg Cantor.
- **Cantor's definition of a set:** "any collection M into a whole of definite, distinct objects m of our perception or of our thought."
- Key implications:
  - A set is a **single entity** — think of the whole collection, not individual elements.
  - Elements must be **distinct** (unique) — tables without keys are multisets/bags, not true sets.
  - **No order** among elements — curly notation `{a, b, c}` equals `{b, c, a}`. This is why rows in a table have no guaranteed order without ORDER BY.
  - **Subjective definition** — you define what constitutes a set based on your application's needs.

#### Predicate Logic
- **Predicate:** a property or expression that evaluates to TRUE or FALSE.
- Used to enforce integrity (e.g., `salary > 0`) and to filter data (e.g., `department = 'sales'`).
- In set theory, predicates define sets: "the set of all prime numbers" can be expressed as a predicate rather than listing all elements.

#### The Relational Model
- Created by **Dr. Edgar F. Codd** in 1969–1970 at IBM.
- Goal: consistent, non-redundant, complete data representation with built-in data integrity.
- Based on mathematical foundations → you can *prove* when a design is flawed.
- Key concepts:
  - **Proposition:** an assertion (e.g., "Employee Itzik Ben-Gan works in IT department") — either true or false. A true proposition manifests as a row. This is the **Closed World Assumption (CWA)** — if a fact isn't stored, it's assumed false.
  - **Predicate:** a parameterized proposition used to define the structure (heading) of a relation.
  - **Relation:** a set of related information. In SQL, the counterpart is a **table** (though not exact). A relation has a **header** (set of attributes/columns) and a **body** (set of tuples/rows).
  - **Attribute:** identified by name and type. Order doesn't matter (unlike SQL columns which have ordinal positions — a deviation from relational theory).
  - **Tuple:** a row identified by its key values, not by position.

> **Important Note:** The term "relational" comes from the mathematical term *relation*, not from relationships between tables.

#### Missing Values and NULL
- Debate exists whether predicates should be **two-valued** (TRUE/FALSE) or **three-valued** (TRUE/FALSE/UNKNOWN).
- Codd actually advocated **four-valued logic**: "missing but applicable" (A-Mark) vs. "missing but inapplicable" (I-Mark).
- SQL implements **three-valued predicate logic** with a single `NULL` mark for any missing value.
- NULL is the source of enormous confusion and complexity in SQL.

#### Constraints
- The relational model allows data integrity to be defined *as part of the model*.
- **Candidate key:** defined on one or more attributes; prevents duplicate rows; uniquely identifies a row. Multiple candidate keys can exist per table.
- **Primary key:** the chosen candidate key for identification. Others become **alternate keys**.
- **Foreign key:** references a candidate key in another (or same) table; enforces referential integrity.

#### Normalization (Normal Forms)
Normalization is a formal process to ensure each entity is represented by a single relation, eliminating anomalies and redundancy.

**1NF (First Normal Form):**
- Rows must be unique (enforced by a key).
- Attributes must be **atomic** — atomicity is subjective; depends on the application's needs.
- Does not forbid arrays per se; it restricts data to conform to proper set theory.

**2NF (Second Normal Form):**
- Must satisfy 1NF.
- Every **non-key attribute** must be **fully functionally dependent** on the *entire* candidate key — no partial dependencies.
- Example violation: In `Orders(orderid, productid, orderdate, qty, customerid, companyname)`, `orderdate` depends only on `orderid`, not on `(orderid, productid)` → split into `Orders` and `OrderDetails`.

**3NF (Third Normal Form):**
- Must satisfy 2NF.
- All non-key attributes must be **mutually independent** — no transitive dependencies.
- Example: if `companyname` depends on `customerid`, and `customerid` is a non-key attribute in `Orders`, move `companyname` to a separate `Customers` table.
- Informally: *"Every non-key attribute is dependent on the key, the whole key, and nothing but the key — so help me Codd."*

Higher normal forms (Boyce-Codd, 4NF, 5NF) exist but are beyond this book's scope.

### 1.2 The Data Life Cycle

Data moves through several environments:

```
OLTP → DSA → DW → BISM → DM
         ↑
        ETL
```

**OLTP (Online Transactional Processing):**
- Focus: data entry (INSERT, UPDATE, DELETE), not reporting.
- Normalized model → good for data entry, consistency, minimal redundancy.
- Poor for reporting (complex joins across many tables).
- Implemented in SQL Server, managed with T-SQL.

**DW (Data Warehouse) / Data Mart:**
- Focus: data retrieval and reporting.
- **Intentional redundancy**, fewer tables, simpler relationships → simpler, faster queries.
- **Star schema:** fact table (quantities, values) surrounded by dimension tables (Customer, Product, Employee, Time).
- **Snowflake schema:** normalized dimension tables.
- Preaggregated data at a certain granularity (e.g., by day), unlike OLTP which stores transaction-level data.
- **ETL (Extract, Transform, Load):** process to pull data from sources and load into DW. SQL Server Integration Services (SSIS) handles ETL.
- **DSA (Data Staging Area):** intermediate relational database used for data cleansing; not visible to end users.

**BISM (Business Intelligence Semantic Model):**
- Microsoft's model for the full BI stack.
- Three layers: data model, business logic/queries, data access.
- Deployment: Analysis Services (for IT/BI pros) or PowerPivot (for business users).
- Data models: multidimensional or tabular (relational).
- Languages: **MDX** (Multidimensional Expressions) and **DAX** (Data Analysis Expressions).
- Storage engines: MOLAP (preaggregated, multidimensional) and VertiPaq (columnstore, compressed, no preaggregation needed).

**DM (Data Mining):**
- Instead of users asking questions, algorithms *find* useful patterns, trends, anomalies.
- Analysis Services supports algorithms: clustering, decision trees, etc.
- Language: **DMX** (Data Mining Extensions).

### 1.3 SQL Server Architecture

#### The ABC Flavors of SQL Server
- **A — Appliance:** Complete hardware + software + services solution (e.g., Parallel Data Warehouse — PDW). Hosted at customer site. Uses DSQL (distributed SQL), not full T-SQL.
- **B — Box (On-Premises):** Traditional installation on customer's premises. Full T-SQL support. Customer manages everything.
- **C — Cloud (Windows Azure SQL Database):** Hosted by Microsoft. Same code base as on-premises but with differences. T-SQL mostly the same, some features missing. Updates deploy faster than on-premises releases. Connect to one database at a time; no cross-database queries.

#### SQL Server Instances
- An **instance** is a single installation of SQL Server Database Engine.
- Multiple instances can run on the same computer — each is completely independent (separate security, data, configuration).
- **Default instance:** connect using just the computer name (e.g., `Server1`).
- **Named instance:** connect using `Server1\InstanceName` (e.g., `Server1\Inst1`).
- One instance can be default; all others must be named.

**Use cases for multiple instances:**
- Support environments mimicking various production configurations.
- Teaching/demo purposes (different versions side-by-side).
- Providing security isolation for different clients.

#### Databases
- A database is a **container of schemas**, which contain **objects** (tables, views, procedures, etc.).
- Each instance has **system databases** + user databases.

**System Databases:**
| Database | Purpose |
|----------|---------|
| `master` | Instance-wide metadata, server config, info about all databases, initialization |
| `Resource` | Hidden, read-only; holds definitions of all system objects |
| `model` | Template for new databases; objects/settings placed here appear in all new databases |
| `tempdb` | Temporary data (work tables, sort space, row versioning, temp tables); re-created on restart |
| `msdb` | SQL Server Agent data (jobs, schedules, alerts), replication, Database Mail, backups |

- **Collation** at database level determines character data language support, case sensitivity, sort order.
- **Contained databases** (SQL Server 2012): user fully contained within the database, not tied to a server-level logon.

**Physical layout of a database:**
- **Data files (.mdf):** Primary data file. Contains object data and system catalog.
- **Secondary data files (.ndf):** Optional additional data files.
- **Log files (.ldf):** Transaction log; sequential writes only (no performance benefit from multiple log files).
- **Filegroups:** Logical groupings of data files. PRIMARY filegroup is mandatory. Objects target a filegroup, and data spreads across its files.
- SQL Server can write to multiple data files in parallel but only one log file at a time.

#### Schemas and Objects
- A database contains **schemas**, which contain **objects**.
- Schema = namespace + security boundary.
- **Two-part names:** `schema.object` (e.g., `Sales.Orders`). Always use two-part names to avoid ambiguity and implicit schema resolution overhead.
- Default schema: `dbo` (database owner) — created automatically in every database.

### 1.4 Creating Tables and Defining Data Integrity

#### Creating Tables

```sql
USE TSQL2012;
IF OBJECT_ID('dbo.Employees', 'U') IS NOT NULL
    DROP TABLE dbo.Employees;

CREATE TABLE dbo.Employees
(
    empid     INT          NOT NULL,
    firstname VARCHAR(30)  NOT NULL,
    lastname  VARCHAR(30)  NOT NULL,
    hiredate  DATE         NOT NULL,
    mgrid     INT          NULL,      -- NULL = no manager (e.g., CEO)
    ssn       VARCHAR(20)  NOT NULL,
    salary    MONEY        NOT NULL
);
```

- `OBJECT_ID('dbo.Employees', 'U')`: returns object ID if user table exists, NULL otherwise. Type `'U'` = user table.
- Always use `NOT NULL` unless there's a compelling reason for NULL; enforce nullability explicitly.
- Semicolons: T-SQL recommends terminating all statements with `;`. This is standard SQL. SQL Server has indicated that NOT terminating is a deprecated feature.

**Coding Style Notes:**
- Use white space for readability.
- Use the semicolon to terminate all statements.
- Consistency and readability matter most.

#### Defining Data Integrity (Declarative Constraints)

**Primary Key Constraint:**
```sql
ALTER TABLE dbo.Employees
ADD CONSTRAINT PK_Employees PRIMARY KEY(empid);
```
- Enforces uniqueness + NOT NULL on constraint columns.
- Only one primary key per table.
- SQL Server creates a **unique index** behind the scenes to enforce it.

**Unique Constraint:**
```sql
ALTER TABLE dbo.Employees
ADD CONSTRAINT UNQ_Employees_ssn UNIQUE(ssn);
```
- Can have multiple per table.
- Allows NULLs (but SQL Server only allows one NULL per unique constraint column — deviating from standard SQL which allows multiple NULLs).
- SQL Server creates a unique index behind the scenes.

**Foreign Key Constraint:**
```sql
-- First create Orders table with PK
CREATE TABLE dbo.Orders
(
    orderid  INT          NOT NULL,
    empid    INT          NOT NULL,
    custid   VARCHAR(10)  NOT NULL,
    orderts  DATETIME2    NOT NULL,
    qty      INT          NOT NULL,
    CONSTRAINT PK_Orders PRIMARY KEY(orderid)
);

-- FK: Orders.empid → Employees.empid
ALTER TABLE dbo.Orders
ADD CONSTRAINT FK_Orders_Employees
    FOREIGN KEY(empid) REFERENCES dbo.Employees(empid);

-- Self-referencing FK: Employees.mgrid → Employees.empid
ALTER TABLE dbo.Employees
ADD CONSTRAINT FK_Employees_Employees
    FOREIGN KEY(mgrid) REFERENCES dbo.Employees(empid);
```
- FK columns may contain NULLs even if the referenced PK column doesn't allow NULLs.
- Default referential action: **no action** — rejects deletes/updates if related rows exist.
- Other options: `ON DELETE CASCADE`, `ON DELETE SET DEFAULT`, `ON DELETE SET NULL` (same for `ON UPDATE`).
- `WITH NOCHECK` bypasses constraint checking for existing data (bad practice).

**Check Constraint:**
```sql
ALTER TABLE dbo.Employees
ADD CONSTRAINT CHK_Employees_salary CHECK(salary > 0.00);
```
- Row is **rejected** when the predicate evaluates to **FALSE**.
- Row is **accepted** when the predicate evaluates to **TRUE** or **UNKNOWN** (i.e., a NULL salary passes the check).
- This is consistent with "reject FALSE" logic (opposite of query filters which "accept TRUE").

**Default Constraint:**
```sql
ALTER TABLE dbo.Orders
ADD CONSTRAINT DFT_Orders_orderts DEFAULT(SYSDATETIME()) FOR orderts;
```
- When an explicit value isn't provided for `orderts` in an INSERT, SQL Server automatically uses `SYSDATETIME()`.

---

## Chapter 2: Single-Table Queries {#chapter-2}

### 2.1 Elements of the SELECT Statement

#### Logical Query Processing Order
SQL processes clauses in a **different order** than they are written:

| Logical Order | Clause | Written Order |
|:---:|--------|:---:|
| 1 | FROM | 3 |
| 2 | WHERE | 4 |
| 3 | GROUP BY | 5 |
| 4 | HAVING | 6 |
| 5 | SELECT | 1 |
| 6 | ORDER BY | 2 |

Within SELECT: Expressions → DISTINCT  
Within ORDER BY: TOP / OFFSET-FETCH

This is **logical** processing — SQL Server may physically optimize in a different order, but the result must match what logical processing would produce.

**Critical implication:** Column aliases assigned in SELECT cannot be referenced in WHERE, GROUP BY, or HAVING — those are processed *before* SELECT.

```sql
-- WRONG: orderyear alias not yet defined when WHERE is processed
SELECT orderid, YEAR(orderdate) AS orderyear
FROM Sales.Orders
WHERE orderyear > 2006;  -- Error: Invalid column name 'orderyear'

-- CORRECT: repeat the expression
SELECT orderid, YEAR(orderdate) AS orderyear
FROM Sales.Orders
WHERE YEAR(orderdate) > 2006;
```

**Sample query (used throughout this chapter):**
```sql
USE TSQL2012;
SELECT empid, YEAR(orderdate) AS orderyear, COUNT(*) AS numorders
FROM Sales.Orders
WHERE custid = 71
GROUP BY empid, YEAR(orderdate)
HAVING COUNT(*) > 1
ORDER BY empid, orderyear;
```

#### The FROM Clause
- First clause logically processed.
- Specifies source tables and table operators.
- Always use **schema-qualified** two-part names: `Sales.Orders` not just `Orders`.

```sql
SELECT orderid, custid, empid, orderdate, freight
FROM Sales.Orders;
-- Returns 830 rows (no filter)
```

**Delimiting identifiers:**
- Standard SQL: double quotes `"Order Details"`
- T-SQL specific: square brackets `[Order Details]`
- Both supported by SQL Server. Square brackets are the T-SQL way.
- If identifier follows regular naming rules, delimiters are optional (but allowed).

#### The WHERE Clause
- Returns only rows for which the predicate evaluates to **TRUE** (not FALSE, not UNKNOWN).
- Affects query performance — SQL Server evaluates indexes based on filter expressions.
- **Never apply functions to filtered columns** when you want index usage:

```sql
-- BAD for performance (can't use index on orderdate efficiently)
WHERE YEAR(orderdate) = 2007

-- GOOD: range filter allows index use
WHERE orderdate >= '20070101' AND orderdate < '20080101'
```

- Three-valued logic applies: WHERE filters out UNKNOWN (e.g., when comparing with NULL).

#### The GROUP BY Clause
- Groups rows by the specified expressions; produces one row per unique combination.
- After GROUP BY, all subsequent phases (HAVING, SELECT, ORDER BY) operate on **groups**, not individual rows.
- All SELECT expressions must either:
  - Be in the GROUP BY list (return a scalar per group by definition), or
  - Be wrapped in an aggregate function (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`).

```sql
-- Returns total freight and number of orders per employee per year
SELECT empid, YEAR(orderdate) AS orderyear,
       SUM(freight) AS totalfreight,
       COUNT(*) AS numorders
FROM Sales.Orders
WHERE custid = 71
GROUP BY empid, YEAR(orderdate);
```

**Aggregate function behavior with NULLs:**
- All aggregate functions **ignore NULLs**, except `COUNT(*)`.
- `COUNT(*)` counts rows; `COUNT(column)` counts non-NULL values.
- `COUNT(DISTINCT column)` counts distinct non-NULL values.
- Example with values `{30, 10, NULL, 10, 10}`:
  - `COUNT(*)` = 5 (all rows)
  - `COUNT(qty)` = 4 (ignores NULL)
  - `COUNT(DISTINCT qty)` = 2 (distinct: 10, 30)
  - `SUM(qty)` = 60; `SUM(DISTINCT qty)` = 40
  - `AVG(qty)` = 15; `AVG(DISTINCT qty)` = 20

#### The HAVING Clause
- Filters **groups** (after GROUP BY), not individual rows.
- Predicate can reference aggregate functions.
- Like WHERE: accepts TRUE, rejects FALSE and UNKNOWN.
- Column aliases from SELECT cannot be used here (HAVING is processed before SELECT).

```sql
HAVING COUNT(*) > 1  -- Only groups with more than 1 order
```

#### The SELECT Clause
- Returns the result columns.
- **Column aliasing** forms:
  - `expression AS alias` — recommended, clearest
  - `alias = expression` — alternative
  - `expression alias` (space, no AS) — NOT recommended; very hard to read
- **WARNING:** Missing comma bug — `SELECT orderid orderdate FROM Sales.Orders` — SQL Server interprets `orderdate` as an alias for `orderid`, returns only one column. Hard to detect!

**DISTINCT:**
```sql
SELECT DISTINCT empid, YEAR(orderdate) AS orderyear
FROM Sales.Orders
WHERE custid = 71;
-- Returns unique combinations (16 rows instead of 31 with duplicates)
```

**Avoid SELECT *:**
- Returns columns in ordinal position order (not relational).
- Client code using ordinal positions breaks if schema changes.
- New columns added to the table won't automatically appear in views using `SELECT *` — the compiled view form only knows columns at creation time.
- No practical performance difference, but `SELECT *` is still poor practice.

**All-at-once in SELECT:** You cannot reference a column alias created in the same SELECT clause within the same SELECT clause:
```sql
-- WRONG
SELECT orderid,
       YEAR(orderdate) AS orderyear,
       orderyear + 1 AS nextyear  -- Error: orderyear not yet defined
FROM Sales.Orders;

-- CORRECT: repeat the expression
SELECT orderid,
       YEAR(orderdate) AS orderyear,
       YEAR(orderdate) + 1 AS nextyear
FROM Sales.Orders;
```

#### The ORDER BY Clause
- **Last** clause processed; guarantees presentation ordering.
- Without ORDER BY, rows may return in **any order** — never assume ordering.
- A query WITH ORDER BY returns a **cursor** (not a relational table). This is why table expressions cannot have ORDER BY (except with TOP/OFFSET-FETCH).
- Only clause that can reference SELECT-clause aliases (because it's processed after SELECT).

```sql
ORDER BY empid, orderyear          -- uses alias from SELECT
ORDER BY empid ASC, orderyear DESC -- explicit ASC/DESC (ASC is default)
ORDER BY 1, 2                      -- bad practice: ordinal positions
```

- T-SQL allows ordering by columns NOT in the SELECT list (unless DISTINCT is used).
- With DISTINCT, ORDER BY is restricted to expressions in the SELECT list (otherwise the meaning is ambiguous).

#### The TOP and OFFSET-FETCH Filters

**TOP Filter (T-SQL proprietary):**
```sql
-- Return 5 most recent orders
SELECT TOP (5) orderid, orderdate, custid, empid
FROM Sales.Orders
ORDER BY orderdate DESC;

-- Return top 1 percent
SELECT TOP (1) PERCENT orderid, orderdate, custid, empid
FROM Sales.Orders
ORDER BY orderdate DESC;
-- Returns 9 rows (1% of 830, rounded up)

-- Add tiebreaker for deterministic results
SELECT TOP (5) orderid, orderdate, custid, empid
FROM Sales.Orders
ORDER BY orderdate DESC, orderid DESC;  -- orderid as tiebreaker

-- Include all ties
SELECT TOP (5) WITH TIES orderid, orderdate, custid, empid
FROM Sales.Orders
ORDER BY orderdate DESC;
-- Returns 8 rows (last date has 3 rows tied)
```

- **Non-deterministic** if ORDER BY has ties — SQL Server returns whichever rows it physically accesses first.
- TOP without ORDER BY is fully non-deterministic.
- ORDER BY serves *dual purpose* with TOP: presentation ordering AND filtering specification.
- `WITH TIES`: returns all rows tied with the last row in the TOP result.

**OFFSET-FETCH Filter (SQL Server 2012, ANSI standard):**
```sql
-- Skip 50 rows, return next 25 (page 3 of 25 rows per page)
SELECT orderid, orderdate, custid, empid
FROM Sales.Orders
ORDER BY orderdate, orderid
OFFSET 50 ROWS FETCH NEXT 25 ROWS ONLY;
```

- Requires ORDER BY.
- FETCH requires OFFSET (use `OFFSET 0 ROWS` if you don't want to skip).
- OFFSET without FETCH: skip rows, return the rest.
- Syntax is intentionally English-like: `ROW`/`ROWS`, `FIRST`/`NEXT` are interchangeable.
- Does NOT support PERCENT or WITH TIES (use TOP for those).
- Preferred over TOP for standard, page-able results.

#### Window Functions (Brief Introduction)

A **window function** computes a scalar result for each row based on a *window* (subset) of rows, defined with an `OVER` clause.

```sql
-- Row number per customer, ordered by value
SELECT orderid, custid, val,
       ROW_NUMBER() OVER(PARTITION BY custid ORDER BY val) AS rownum
FROM Sales.OrderValues
ORDER BY custid, val;
```

- `PARTITION BY custid`: window is per customer — row numbers reset per customer.
- `ORDER BY val`: within each partition, rows are ordered by val for numbering.
- **Important:** `OVER` ORDER BY ≠ presentation ORDER BY. Window ordering is for the calculation; you need a separate presentation ORDER BY for output ordering.
- `ROW_NUMBER()` must produce unique values per partition; if ORDER BY list is non-unique, the query is non-deterministic.
- Window functions are evaluated as part of SELECT, *before* DISTINCT.

Full coverage in Chapter 7.

### 2.2 Predicates and Operators

**Predicates:**
```sql
WHERE orderid IN(10248, 10249, 10250)           -- IN: equality to any in set
WHERE orderid BETWEEN 10300 AND 10310            -- BETWEEN: inclusive range
WHERE lastname LIKE N'D%'                        -- LIKE: pattern matching
```

Note: `N'D%'` — the `N` prefix denotes Unicode (NCHAR/NVARCHAR). Use it when the column is of a Unicode type.

**Comparison operators:**
- Standard: `=`, `>`, `<`, `>=`, `<=`, `<>`
- Non-standard (avoid): `!=`, `!>`, `!<` — use `<>` instead of `!=`

```sql
WHERE orderdate >= '20080101'
WHERE orderdate >= '20080101' AND empid IN(1, 3, 5)
```

**Arithmetic operators:** `+`, `-`, `*`, `/`, `%` (modulo)

```sql
SELECT orderid, productid, qty, unitprice, discount,
       qty * unitprice * (1 - discount) AS val
FROM Sales.OrderDetails;
```

**Data type precedence in expressions:**
- When operands are different types, the lower-precedence type is implicitly promoted.
- Integer / Integer = Integer (truncated): `5/2 = 2`, not `2.5`.
- To get decimal: `CAST(col1 AS NUMERIC(12,2)) / CAST(col2 AS NUMERIC(12,2))`
- Integer / Numeric = Numeric: `5/2.0 = 2.5`

**Operator precedence (high to low):**
1. `()` Parentheses
2. `*`, `/`, `%`
3. `+`, `-`, `+` (concatenation)
4. Comparison operators (`=`, `>`, `<`, etc.)
5. `NOT`
6. `AND`
7. `BETWEEN`, `IN`, `LIKE`, `OR`
8. `=` (assignment)

**Best practice:** Use parentheses for clarity even when not required:
```sql
WHERE (custid = 1 AND empid IN(1, 3, 5))
   OR (custid = 85 AND empid IN(2, 4, 6))
```

### 2.3 CASE Expressions

CASE is a **scalar expression** (not a statement) — returns a single value based on conditional logic. Can appear anywhere a scalar expression is allowed: SELECT, WHERE, HAVING, ORDER BY, CHECK constraints.

**Simple CASE form:**
```sql
SELECT productid, productname, categoryid,
    CASE categoryid
        WHEN 1 THEN 'Beverages'
        WHEN 2 THEN 'Condiments'
        WHEN 3 THEN 'Confections'
        ...
        ELSE 'Unknown Category'
    END AS categoryname
FROM Production.Products;
```
- Compares one value against a list of possible values (equality only).

**Searched CASE form:**
```sql
SELECT orderid, custid, val,
    CASE
        WHEN val < 1000.00                  THEN 'Less than 1000'
        WHEN val BETWEEN 1000.00 AND 3000.00 THEN 'Between 1000 and 3000'
        WHEN val > 3000.00                  THEN 'More than 3000'
        ELSE 'Unknown'
    END AS valuecategory
FROM Sales.OrderValues;
```
- Supports any predicates (not just equality).
- Returns value from the first WHEN that evaluates to TRUE.
- If no WHEN matches and no ELSE, returns NULL.

**CASE abbreviation functions:**
- `ISNULL(col, default)` — returns `col` if not NULL, otherwise `default` (T-SQL non-standard).
- `COALESCE(val1, val2, ...)` — returns first non-NULL value (**standard SQL**, preferred).
- `IIF(condition, val_if_true, val_if_false)` — SQL Server 2012+, non-standard (Access migration).
- `CHOOSE(index, val1, val2, ...)` — SQL Server 2012+, non-standard.

When there's a choice, prefer standard functions: use `COALESCE` instead of `ISNULL`.

**CASE for NULL sorting (NULL last):**
```sql
SELECT custid, region
FROM Sales.Customers
ORDER BY
    CASE WHEN region IS NULL THEN 1 ELSE 0 END,  -- NULLs sort last
    region;
```

### 2.4 NULL Marks

NULLs are the most confusing aspect of T-SQL. Key rules:

**Three-valued predicate logic: TRUE, FALSE, UNKNOWN**
- Any comparison with NULL yields **UNKNOWN**: `NULL = NULL` → UNKNOWN, `NULL > 5` → UNKNOWN.
- `NOT UNKNOWN` is still **UNKNOWN**.
- Query filters (WHERE, HAVING): **accept TRUE**, reject both FALSE and UNKNOWN.
- CHECK constraints: **reject FALSE**, accept both TRUE and UNKNOWN.

This means:
- `WHERE salary > 0`: rows with NULL salary are **filtered out** (UNKNOWN ≠ TRUE).
- `CHECK(salary > 0)`: rows with NULL salary are **accepted** (UNKNOWN ≠ FALSE).

**IS NULL / IS NOT NULL — the correct way to test for NULLs:**
```sql
-- WRONG: returns no rows even when looking for NULLs
SELECT * FROM Sales.Customers WHERE region = NULL;  -- always UNKNOWN

-- CORRECT
SELECT * FROM Sales.Customers WHERE region IS NULL;   -- works
SELECT * FROM Sales.Customers WHERE region IS NOT NULL; -- works
```

**Returning rows where region ≠ 'WA' including NULLs:**
```sql
SELECT custid, country, region, city
FROM Sales.Customers
WHERE region <> N'WA'
   OR region IS NULL;
-- Without OR region IS NULL, NULL rows would be excluded (UNKNOWN)
```

**NULL treatment in different contexts:**
- `GROUP BY`: Two NULLs are treated as **equal** — grouped together.
- `ORDER BY`: Two NULLs are treated as **equal** — sorted together. T-SQL sorts NULLs **before** non-NULLs by default.
- `WHERE`: NULL = NULL evaluates to UNKNOWN → row filtered out.
- `UNIQUE constraint`: SQL Server treats two NULLs as **equal** (only one NULL allowed) — this deviates from standard SQL which treats NULLs as distinct.

**Practical rule:** Always explicitly think about NULLs in every query. If default treatment is wrong, intervene with explicit IS NULL checks.

### 2.5 All-at-Once Operations

All expressions in the same logical processing phase are evaluated **simultaneously** — there is no left-to-right evaluation order within a phase.

**Consequence 1:** Cannot refer to aliases from the same SELECT clause:
```sql
SELECT orderid,
       YEAR(orderdate) AS orderyear,
       orderyear + 1 AS nextyear  -- Error: orderyear doesn't exist yet
FROM Sales.Orders;
```

**Consequence 2:** Short-circuit evaluation is NOT guaranteed in WHERE:
```sql
-- UNSAFE: SQL Server may evaluate col2/col1 first
SELECT col1, col2 FROM dbo.T1
WHERE col1 <> 0 AND col2/col1 > 2;

-- SAFE: CASE guarantees order of WHEN evaluation
SELECT col1, col2 FROM dbo.T1
WHERE
    CASE
        WHEN col1 = 0 THEN 'no'
        WHEN col2/col1 > 2 THEN 'yes'
        ELSE 'no'
    END = 'yes';

-- Or mathematical workaround (avoids division):
SELECT col1, col2 FROM dbo.T1
WHERE (col1 > 0 AND col2 > 2*col1)
   OR (col1 < 0 AND col2 < 2*col1);
```

> **Key insight:** `CASE` expressions guarantee sequential evaluation of WHEN clauses. This is the reliable way to control evaluation order.

**Consequence 3 (UPDATE):** In a single UPDATE, all assignments use the values *before* the update:
```sql
UPDATE dbo.T1 SET col1 = col2, col2 = col1;  -- Correctly swaps values
```

### 2.6 Working with Character Data

#### Data Types
| Type | Storage | Notes |
|------|---------|-------|
| `CHAR(n)` | Fixed n bytes | Pads with spaces; 1 byte per char |
| `VARCHAR(n)` | Variable, max n bytes + 2 | 1 byte per char |
| `NCHAR(n)` | Fixed 2n bytes | 2 bytes per Unicode char |
| `NVARCHAR(n)` | Variable, max 2n bytes + 2 | 2 bytes per Unicode char |
| `VARCHAR(MAX)` / `NVARCHAR(MAX)` | Up to ~2GB | Stored inline if ≤ 8000 bytes, else as LOB |

- Regular (`CHAR`/`VARCHAR`): one language + English, determined by collation.
- Unicode (`NCHAR`/`NVARCHAR`): multiple languages supported.
- Literals: regular → `'string'`; Unicode → `N'string'` (National prefix).
- Fixed-length types: better for write-heavy workloads (no row expansion). Variable-length: better for read-heavy (less storage).

#### Collation
Collation encapsulates: language support, sort order, case sensitivity, accent sensitivity.

```sql
SELECT name, description FROM sys.fn_helpcollations();
```

Example: `Latin1_General_CI_AS`
- `Latin1_General`: code page 1252 (Western European)
- `CI`: Case Insensitive (`a = A`)
- `AS`: Accent Sensitive (`à ≠ ä`)
- `BIN` (if present): binary sort (`A < B < a < b`)

Collation hierarchy: instance → database → column → expression (lowest wins).

**Convert collation in an expression:**
```sql
WHERE lastname COLLATE Latin1_General_CS_AS = N'davis'
-- Case-sensitive comparison even in a CI environment
```

**Database collation determines metadata** (object names, column names). In a case-sensitive database, `T1` and `t1` can be different tables.

#### Operators and Functions

**String Concatenation:**
```sql
-- Plus operator (NULL propagates)
SELECT empid, firstname + N' ' + lastname AS fullname
FROM HR.Employees;

-- COALESCE handles NULL in concatenation
SELECT custid, country + COALESCE(N',' + region, N'') + N',' + city AS location
FROM Sales.Customers;

-- CONCAT function (SQL Server 2012+): NULLs treated as empty strings
SELECT custid, CONCAT(country, N',' + region, N',' + city) AS location
FROM Sales.Customers;
```

**String Functions:**

| Function | Syntax | Description |
|----------|--------|-------------|
| `SUBSTRING` | `SUBSTRING(str, start, length)` | Extract substring; won't error past end of string |
| `LEFT` | `LEFT(str, n)` | First n characters |
| `RIGHT` | `RIGHT(str, n)` | Last n characters |
| `LEN` | `LEN(str)` | Character count; excludes trailing spaces |
| `DATALENGTH` | `DATALENGTH(str)` | Byte count; includes trailing spaces |
| `CHARINDEX` | `CHARINDEX(substr, str [, start])` | Position of first occurrence; returns 0 if not found |
| `PATINDEX` | `PATINDEX(pattern, str)` | Position of pattern (like LIKE patterns) |
| `REPLACE` | `REPLACE(str, old, new)` | Replace all occurrences |
| `REPLICATE` | `REPLICATE(str, n)` | Repeat string n times |
| `STUFF` | `STUFF(str, pos, len, insert)` | Delete len chars at pos, insert new string |
| `UPPER` / `LOWER` | `UPPER(str)` / `LOWER(str)` | Case conversion |
| `RTRIM` / `LTRIM` | `RTRIM(str)` / `LTRIM(str)` | Trim trailing/leading spaces |
| `FORMAT` | `FORMAT(val, format [, culture])` | .NET-style formatting (SQL Server 2012+) |

```sql
-- Count occurrences of 'e' in lastname
SELECT empid, lastname,
       LEN(lastname) - LEN(REPLACE(lastname, 'e', '')) AS numoccur
FROM HR.Employees;

-- Zero-pad supplier ID to 10 digits
SELECT supplierid,
       RIGHT(REPLICATE('0', 9) + CAST(supplierid AS VARCHAR(10)), 10) AS strsupplierid
FROM Production.Suppliers;

-- Equivalent using FORMAT (SQL Server 2012+)
SELECT FORMAT(1759, '000000000');  -- Returns '0000001759'

-- STUFF: replace 1 char at position 2 with 'abc'
SELECT STUFF('xyz', 2, 1, 'abc');  -- Returns 'xabcz'

-- Trim both leading and trailing spaces
SELECT RTRIM(LTRIM('   abc   '));  -- Returns 'abc'
```

#### The LIKE Predicate

Pattern matching in character string comparison.

| Wildcard | Description | Example |
|----------|-------------|---------|
| `%` | Any string (including empty) | `'D%'` matches 'Davis', 'D', 'Do...' |
| `_` | Single character | `'_e%'` matches 'Lew', 'Peled' (e is 2nd char) |
| `[ABC]` | Single char from list | `'[ABC]%'` matches 'Albert', 'Bob', 'Carol' |
| `[A-E]` | Single char in range | `'[A-E]%'` matches anything starting A through E |
| `[^A-E]` | Single char NOT in range | `'[^A-E]%'` excludes A through E start |

**Escape character:**
```sql
WHERE col1 LIKE '%!_%' ESCAPE '!'  -- Look for literal underscore
WHERE col1 LIKE '%[_]%'            -- Alternative using brackets
```

**Performance note:** LIKE with a leading `%` (like `'%abc%'`) cannot use an index scan efficiently. LIKE with a known prefix (`'D%'`) CAN use an index efficiently.

### 2.7 Working with Date and Time Data

#### Data Types

| Type | Storage | Range | Accuracy |
|------|---------|-------|----------|
| `DATETIME` | 8 bytes | Jan 1, 1753 – Dec 31, 9999 | 3⅓ milliseconds |
| `SMALLDATETIME` | 4 bytes | Jan 1, 1900 – Jun 6, 2079 | 1 minute |
| `DATE` | 3 bytes | Jan 1, 0001 – Dec 31, 9999 | 1 day |
| `TIME` | 3-5 bytes | N/A | 100 nanoseconds |
| `DATETIME2` | 6-8 bytes | Jan 1, 0001 – Dec 31, 9999 | 100 nanoseconds |
| `DATETIMEOFFSET` | 8-10 bytes | Jan 1, 0001 – Dec 31, 9999 | 100 nanoseconds + time zone |

- `TIME`, `DATETIME2`, `DATETIMEOFFSET` support precision specification: `TIME(0)` = 1 second, `TIME(3)` = 1 ms, `TIME(7)` = 100 ns.

#### Literals and Language Independence

**Critical rule:** Always use **language-neutral formats** to express date literals. Some formats are language-dependent and will be interpreted differently depending on session language/DATEFORMAT setting.

**Safe (language-neutral) formats:**
- `'YYYYMMDD'` — always interpreted as year-month-day regardless of DATEFORMAT setting.
- `'YYYY-MM-DDThh:mm:ss.nnn'` — ISO 8601 format.
- For `DATE`, `DATETIME2`, `DATETIMEOFFSET`: `'YYYY-MM-DD'` is also language-neutral.
- For `DATETIME` and `SMALLDATETIME`: `'YYYY-MM-DD'` is **language-dependent** (avoid it for these types).

```sql
-- UNSAFE: interpretation depends on DATEFORMAT
WHERE orderdate = '02/12/2007'  -- Could be Feb 12 or Dec 2!

-- SAFE
WHERE orderdate = '20070212'  -- Always Feb 12, 2007
```

If you must use a language-dependent format, use `CONVERT` with a style number or `PARSE` with a culture:
```sql
SELECT CONVERT(DATETIME, '02/12/2007', 101);  -- Style 101 = MM/DD/YYYY
SELECT CONVERT(DATETIME, '02/12/2007', 103);  -- Style 103 = DD/MM/YYYY
SELECT PARSE('02/12/2007' AS DATETIME USING 'en-US');
SELECT PARSE('02/12/2007' AS DATETIME USING 'en-GB');
```

**Working with date and time separately (pre-SQL Server 2008):**
- Store **dates only**: use midnight as the time component.
- Store **times only**: use January 1, 1900 as the base date.
- SQL Server 2008+: use dedicated `DATE` and `TIME` types.

#### Filtering Date Ranges (Performance Best Practice)

```sql
-- BAD: function on filtered column prevents efficient index use
WHERE YEAR(orderdate) = 2007

-- GOOD: range filter allows index seek
WHERE orderdate >= '20070101' AND orderdate < '20080101'

-- GOOD: for a specific month
WHERE orderdate >= '20070201' AND orderdate < '20070301'
```

#### Date and Time Functions

**Current date/time:**
```sql
SELECT GETDATE()              -- DATETIME, current date/time
SELECT CURRENT_TIMESTAMP      -- DATETIME, ANSI standard (no parentheses needed)
SELECT GETUTCDATE()           -- DATETIME, UTC
SELECT SYSDATETIME()          -- DATETIME2, current date/time
SELECT SYSUTCDATETIME()       -- DATETIME2, UTC
SELECT SYSDATETIMEOFFSET()    -- DATETIMEOFFSET, with time zone offset

-- Extract current date and time separately
SELECT CAST(SYSDATETIME() AS DATE) AS [current_date]
SELECT CAST(SYSDATETIME() AS TIME) AS [current_time]
```

Prefer `CURRENT_TIMESTAMP` (standard) and `SYSDATETIME()` for modern types.

**Conversion functions:**
```sql
CAST(value AS datatype)                         -- ANSI standard
CONVERT(datatype, value [, style_number])       -- supports style for date formatting
PARSE(value AS datatype [USING culture])        -- SQL Server 2012+, culture-aware
TRY_CAST(value AS datatype)                     -- Returns NULL instead of error
TRY_CONVERT(datatype, value [, style_number])   -- Returns NULL instead of error
TRY_PARSE(value AS datatype [USING culture])    -- Returns NULL instead of error
```

Prefer `CAST` when no style number needed (most standard). TRY_ versions added in SQL Server 2012.

**Date manipulation:**
```sql
-- DATEADD: add/subtract date parts
SELECT DATEADD(year, 1, '20090212');   -- 2010-02-12
SELECT DATEADD(month, -3, '20090212'); -- 2008-11-12
-- Parts: year, quarter, month, dayofyear, day, week, weekday, hour, minute, second, ...

-- DATEDIFF: difference in date part units
SELECT DATEDIFF(day, '20080212', '20090212');  -- 366
SELECT DATEDIFF(month, '20070101', '20090101'); -- 24

-- Get current date at midnight (pre-2008 technique):
SELECT DATEADD(day, DATEDIFF(day, '20010101', CURRENT_TIMESTAMP), '20010101')

-- Get first day of current month:
SELECT DATEADD(month, DATEDIFF(month, '20010101', CURRENT_TIMESTAMP), '20010101')

-- Get last day of current month:
SELECT DATEADD(month, DATEDIFF(month, '19991231', CURRENT_TIMESTAMP), '19991231')
```

**Date part extraction:**
```sql
SELECT DATEPART(month, '20090212');   -- Returns 2 (integer)
SELECT YEAR('20090212');              -- Returns 2009 (abbreviation of DATEPART(year,...))
SELECT MONTH('20090212');             -- Returns 2
SELECT DAY('20090212');               -- Returns 12
SELECT DATENAME(month, '20090212');   -- Returns 'February' (language-dependent string)
SELECT ISDATE('20090212');            -- Returns 1 (valid) or 0 (invalid)
```

**FROMPARTS functions (SQL Server 2012+):**
```sql
SELECT DATEFROMPARTS(2012, 02, 12)
SELECT DATETIME2FROMPARTS(2012, 02, 12, 13, 30, 5, 1, 7)
SELECT DATETIMEFROMPARTS(2012, 02, 12, 13, 30, 5, 997)
SELECT DATETIMEOFFSETFROMPARTS(2012, 02, 12, 13, 30, 5, 1, -8, 0, 7)
SELECT SMALLDATETIMEFROMPARTS(2012, 02, 12, 13, 30)
SELECT TIMEFROMPARTS(13, 30, 5, 1, 7)
```

**EOMONTH (SQL Server 2012+):**
```sql
SELECT EOMONTH(SYSDATETIME())         -- Last day of current month
SELECT EOMONTH('20090212', 3)         -- Last day 3 months from Feb 2009 = May 31, 2009

-- Orders placed on the last day of the month
SELECT orderid, orderdate FROM Sales.Orders
WHERE orderdate = EOMONTH(orderdate);
```

**SWITCHOFFSET / TODATETIMEOFFSET:**
```sql
-- Adjust DATETIMEOFFSET to a different timezone
SELECT SWITCHOFFSET(SYSDATETIMEOFFSET(), '-05:00')

-- Set timezone offset on a datetime value
SELECT TODATETIMEOFFSET(SYSDATETIME(), '-05:00')
```

### 2.8 Querying Metadata

**Catalog views (SQL Server-specific, most detailed):**
```sql
-- List tables in current database
SELECT SCHEMA_NAME(schema_id) AS table_schema_name, name AS table_name
FROM sys.tables;

-- Column details for a specific table
SELECT name AS column_name,
       TYPE_NAME(system_type_id) AS column_type,
       max_length, collation_name, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID(N'Sales.Orders');
```

**Information schema views (ANSI standard, less SQL Server-specific):**
```sql
-- List tables
SELECT TABLE_SCHEMA, TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = N'BASE TABLE';

-- Column info
SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH,
       COLLATION_NAME, IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = N'Sales' AND TABLE_NAME = N'Orders';
```

**System stored procedures and functions:**
```sql
EXEC sys.sp_tables;                          -- List queryable objects
EXEC sys.sp_help @objname = N'Sales.Orders'; -- Object info
EXEC sys.sp_columns @table_name = N'Orders', @table_owner = N'Sales';
EXEC sys.sp_helpconstraint @objname = N'Sales.Orders';

SELECT SERVERPROPERTY('ProductLevel');       -- Instance property (e.g., RTM, SP1)
SELECT DATABASEPROPERTYEX(N'TSQL2012', 'Collation');
SELECT OBJECTPROPERTY(OBJECT_ID(N'Sales.Orders'), 'TableHasPrimaryKey');
SELECT COLUMNPROPERTY(OBJECT_ID(N'Sales.Orders'), N'shipcountry', 'AllowsNull');
```

---

## Chapter 3: Joins {#chapter-3}

Joins are **table operators** in the FROM clause that operate on two input tables and return a result table. SQL Server supports: JOIN, APPLY, PIVOT, UNPIVOT.

**Three fundamental join types:**
- **Cross Join:** Cartesian Product only.
- **Inner Join:** Cartesian Product + Filter (ON predicate).
- **Outer Join:** Cartesian Product + Filter + Adding Outer Rows.

> **Important:** Logical query processing describes *what* happens; physical processing can be entirely different for optimization purposes. The result must be the same.

### 3.1 Cross Joins

A cross join produces a **Cartesian product** — every row from table A matched with every row from table B.  
If A has m rows and B has n rows, result has m × n rows.

**ANSI SQL-92 syntax (recommended):**
```sql
SELECT C.custid, E.empid
FROM Sales.Customers AS C
CROSS JOIN HR.Employees AS E;
-- 91 × 9 = 819 rows
```

**ANSI SQL-89 syntax (avoid):**
```sql
SELECT C.custid, E.empid
FROM Sales.Customers AS C, HR.Employees AS E;
-- Same result, but no JOIN keyword — harder to distinguish from forgotten join condition
```

**Self cross join:**
```sql
SELECT E1.empid, E1.firstname, E1.lastname,
       E2.empid, E2.firstname, E2.lastname
FROM HR.Employees AS E1
CROSS JOIN HR.Employees AS E2;
-- 9 × 9 = 81 rows — all possible pairs of employees
-- Table aliases are REQUIRED in self joins (can't use full table name as prefix)
```

**Practical use — generating numbers table:**
```sql
-- Create Digits table (0-9)
CREATE TABLE dbo.Digits(digit INT NOT NULL PRIMARY KEY);
INSERT INTO dbo.Digits(digit) VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9);

-- Generate numbers 1-1000 using 3 cross-joined instances
SELECT D3.digit * 100 + D2.digit * 10 + D1.digit + 1 AS n
FROM dbo.Digits AS D1
CROSS JOIN dbo.Digits AS D2
CROSS JOIN dbo.Digits AS D3
ORDER BY n;
-- For 1,000,000 rows: join 6 instances
```

### 3.2 Inner Joins

An inner join applies: Cartesian Product → Filter by ON predicate. Returns only rows that satisfy the join condition.

**ANSI SQL-92 syntax (strongly recommended):**
```sql
SELECT E.empid, E.firstname, E.lastname, O.orderid
FROM HR.Employees AS E
JOIN Sales.Orders AS O          -- INNER keyword is optional
ON E.empid = O.empid;
-- Returns 830 rows (all employees have orders)
```

**ANSI SQL-89 syntax (avoid):**
```sql
SELECT E.empid, E.firstname, E.lastname, O.orderid
FROM HR.Employees AS E, Sales.Orders AS O
WHERE E.empid = O.empid;
```

**Why ANSI SQL-92 is safer:**
- Without a JOIN condition in SQL-92 syntax → parser error. You catch the bug immediately.
- Without a WHERE condition in SQL-89 syntax → valid cross join. The bug may go unnoticed.

**ON clause also uses three-valued logic:** rows where ON evaluates to UNKNOWN are filtered out (same as WHERE).

#### More Join Examples

**Composite join (multiple join conditions):**
```sql
SELECT OD.orderid, OD.productid, OD.qty,
       ODA.dt, ODA.loginname, ODA.oldval, ODA.newval
FROM Sales.OrderDetails AS OD
JOIN Sales.OrderDetailsAudit AS ODA
    ON OD.orderid = ODA.orderid
    AND OD.productid = ODA.productid    -- composite PK match
WHERE ODA.columnname = N'qty';
```

**Non-equi join (join condition uses operator other than `=`):**
```sql
-- Generate unique pairs of employees (self join with <)
SELECT E1.empid, E1.firstname, E1.lastname,
       E2.empid, E2.firstname, E2.lastname
FROM HR.Employees AS E1
JOIN HR.Employees AS E2
ON E1.empid < E2.empid;
-- 9 employees → 9×8/2 = 36 unique pairs
-- ON E1.empid < E2.empid eliminates: self-pairs (equal) and mirrored pairs (only one direction qualifies)
```

**Multi-join queries:**
```sql
-- Join 3 tables: Customers → Orders → OrderDetails
SELECT C.custid, C.companyname, O.orderid, OD.productid, OD.qty
FROM Sales.Customers AS C
JOIN Sales.Orders AS O
    ON C.custid = O.custid
JOIN Sales.OrderDetails AS OD
    ON O.orderid = OD.orderid;
-- Table operators processed left-to-right
-- First join: Customers × Orders (filtered)
-- Second join: result × OrderDetails (filtered)
```

With cross/inner joins, the optimizer can rearrange join order for efficiency (it won't change the result).

### 3.3 Outer Joins

Outer joins preserve rows from one or both tables even when no matching row exists in the other table. For non-matching rows, NULL marks are placed in the attributes from the non-preserved side.

**Syntax:** Use `LEFT [OUTER] JOIN`, `RIGHT [OUTER] JOIN`, or `FULL [OUTER] JOIN`. OUTER keyword is optional.

**Three phases of outer join logical processing:**
1. Cartesian Product
2. Filter by ON predicate
3. **Add outer rows** (rows from the preserved table that had no match, with NULLs for the other side)

**Inner rows:** rows that found a match (returned by inner join).  
**Outer rows:** rows that did NOT find a match (added by the third phase).

**LEFT OUTER JOIN:**
```sql
-- Return all customers and their orders; include customers with no orders
SELECT C.custid, C.companyname, O.orderid
FROM Sales.Customers AS C
LEFT OUTER JOIN Sales.Orders AS O
ON C.custid = O.custid;
-- Returns 832 rows: 830 matched + 2 customers (22, 57) with no orders (NULL in order columns)
```

**Finding rows with no match (using NULL test):**
```sql
-- Return customers who placed NO orders
SELECT C.custid, C.companyname
FROM Sales.Customers AS C
LEFT OUTER JOIN Sales.Orders AS O
    ON C.custid = O.custid
WHERE O.orderid IS NULL;  -- Outer rows have NULL in orderid
-- Use PK, join column, or NOT NULL column for reliable NULL test
```

#### Beyond the Fundamentals of Outer Joins

**Including missing values:**
```sql
-- Return orders with all dates in range even if no orders on that date
SELECT DATEADD(day, Nums.n - 1, '20060101') AS orderdate,
       O.orderid, O.custid, O.empid
FROM dbo.Nums
LEFT OUTER JOIN Sales.Orders AS O
    ON DATEADD(day, Nums.n - 1, '20060101') = O.orderdate
WHERE Nums.n <= DATEDIFF(day, '20060101', '20081231') + 1
ORDER BY orderdate;
```

**Filtering attributes from the non-preserved side — common bug:**
```sql
-- BUG: WHERE on non-preserved side nullifies the outer join
SELECT C.custid, C.companyname, O.orderid, O.orderdate
FROM Sales.Customers AS C
LEFT OUTER JOIN Sales.Orders AS O
    ON C.custid = O.custid
WHERE O.orderdate >= '20070101';  -- NULL orderdate → UNKNOWN → filtered out
-- Outer rows (customers with no orders) are all eliminated!
-- The LEFT JOIN becomes effectively an INNER JOIN

-- CORRECT: put the filter in ON (non-final, affects matching only)
-- This returns ALL customers but only matches orders from 2007+
SELECT C.custid, C.companyname, O.orderid, O.orderdate
FROM Sales.Customers AS C
LEFT OUTER JOIN Sales.Orders AS O
    ON C.custid = O.custid
    AND O.orderdate >= '20070101';
```

> **Rule:** If a WHERE clause references a column from the non-preserved side with `<column> <operator> <value>` (not IS NULL), it usually indicates a bug — it silently converts the outer join to an inner join.

**Multi-join outer join bug:**
```sql
-- BUG: outer join followed by inner join nullifies the outer join
SELECT C.custid, O.orderid, OD.productid, OD.qty
FROM Sales.Customers AS C
LEFT OUTER JOIN Sales.Orders AS O      -- outer join: preserves customers with no orders
    ON C.custid = O.custid
JOIN Sales.OrderDetails AS OD          -- inner join: O.orderid is NULL for outer rows
    ON O.orderid = OD.orderid;         -- NULL = OD.orderid → UNKNOWN → filtered out
-- Customers 22 and 57 (with no orders) are lost!

-- FIXES:
-- Option 1: use LEFT OUTER JOIN for the second join too
LEFT OUTER JOIN Sales.OrderDetails AS OD ON O.orderid = OD.orderid

-- Option 2: flip order (inner join first, then outer join to customers)
FROM Sales.Orders AS O
JOIN Sales.OrderDetails AS OD ON O.orderid = OD.orderid
RIGHT OUTER JOIN Sales.Customers AS C ON O.custid = C.custid

-- Option 3: parentheses to make inner join independent
FROM Sales.Customers AS C
LEFT OUTER JOIN
    (Sales.Orders AS O JOIN Sales.OrderDetails AS OD ON O.orderid = OD.orderid)
ON C.custid = O.custid
```

**General rule:** A LEFT OUTER JOIN followed by an INNER JOIN or RIGHT OUTER JOIN on the non-preserved side will drop the outer rows.

**COUNT bug with outer joins:**
```sql
-- BUG: COUNT(*) counts outer rows too!
SELECT C.custid, COUNT(*) AS numorders
FROM Sales.Customers AS C
LEFT OUTER JOIN Sales.Orders AS O ON C.custid = O.custid
GROUP BY C.custid;
-- Customers 22 and 57 show up with count=1 (their outer row is counted)

-- CORRECT: COUNT(column) ignores NULLs
SELECT C.custid, COUNT(O.orderid) AS numorders  -- orderid is NULL for outer rows
FROM Sales.Customers AS C
LEFT OUTER JOIN Sales.Orders AS O ON C.custid = O.custid
GROUP BY C.custid;
-- Customers 22 and 57 correctly show count=0
```

> **Always use `COUNT(column)` instead of `COUNT(*)` when counting on the non-preserved side of an outer join. Choose a column that is NOT NULL except for outer rows (PK, join column, or NOT NULL column).**

---

## Chapter 4: Subqueries {#chapter-4}

A **subquery** is a query nested inside another query. The outer query uses the subquery result as an expression. Subqueries avoid storing intermediate results in variables.

**Types by dependency:**
- **Self-contained:** independent of the outer query; can be run standalone.
- **Correlated:** references columns from the outer query; cannot run standalone.

**Types by result shape:**
- **Scalar:** returns a single value.
- **Multivalued:** returns multiple values in a single column.
- **Table-valued:** returns an entire table (covered in Chapter 5 as table expressions).

### 4.1 Self-Contained Subqueries

#### Self-Contained Scalar Subqueries

Returns a single value; can appear anywhere a scalar expression can (WHERE, SELECT, etc.).

```sql
-- Using a variable (two steps):
DECLARE @maxid AS INT = (SELECT MAX(orderid) FROM Sales.Orders);
SELECT orderid, orderdate, empid, custid
FROM Sales.Orders WHERE orderid = @maxid;

-- Using a scalar subquery (single query):
SELECT orderid, orderdate, empid, custid
FROM Sales.Orders
WHERE orderid = (SELECT MAX(O.orderid) FROM Sales.Orders AS O);
```

**Critical rules for scalar subqueries:**
- Must return **no more than one value**. If it returns multiple values at runtime → error.
- If it returns **zero values** → returns NULL → comparison with NULL → UNKNOWN → row filtered out.

```sql
-- DANGEROUS: might fail at runtime if multiple employees match
WHERE empid = (SELECT E.empid FROM HR.Employees AS E
               WHERE E.lastname LIKE N'D%');
-- Currently works (only one 'D' employee) but will break if data changes

-- SAFE: use multivalued subquery with IN instead
WHERE empid IN (SELECT E.empid FROM HR.Employees AS E
                WHERE E.lastname LIKE N'D%');
```

#### Self-Contained Multivalued Subqueries

Returns multiple values (single column). Used with `IN`, `NOT IN`, `SOME`, `ANY`, `ALL`.

```sql
-- Orders handled by employees with last name starting with 'D'
SELECT orderid
FROM Sales.Orders
WHERE empid IN (SELECT E.empid FROM HR.Employees AS E
                WHERE E.lastname LIKE N'D%');
-- Returns 166 rows; works regardless of how many employees match

-- Orders placed by customers from USA
SELECT custid, orderid, orderdate, empid
FROM Sales.Orders
WHERE custid IN (SELECT C.custid FROM Sales.Customers AS C
                 WHERE C.country = N'USA');

-- Customers who placed NO orders
SELECT custid, companyname
FROM Sales.Customers
WHERE custid NOT IN (SELECT O.custid FROM Sales.Orders AS O);
```

**DISTINCT in subqueries:** Not necessary — the database engine is smart enough to remove duplicates internally when evaluating IN. No benefit to explicitly specifying DISTINCT.

**Multiple self-contained subqueries in one query:**
```sql
-- Find gaps in order IDs between min and max
SELECT n
FROM dbo.Nums
WHERE n BETWEEN (SELECT MIN(O.orderid) FROM dbo.Orders AS O)
            AND (SELECT MAX(O.orderid) FROM dbo.Orders AS O)
  AND n NOT IN (SELECT O.orderid FROM dbo.Orders AS O);
```

### 4.2 Correlated Subqueries

A correlated subquery references columns from the outer query. Logically, it is **evaluated separately for each outer row**.

```sql
-- For each order, return it only if it's the maximum order ID for that customer
SELECT custid, orderid, orderdate, empid
FROM Sales.Orders AS O1
WHERE orderid = (SELECT MAX(O2.orderid)
                 FROM Sales.Orders AS O2
                 WHERE O2.custid = O1.custid);  -- correlation: O1.custid
-- Returns 89 rows (last order for each customer)
```

**How to read a correlated subquery:**
1. Focus on one outer row (e.g., `orderid = 10248`, `custid = 85`).
2. Substitute the correlation value: the subquery becomes `SELECT MAX(O2.orderid) FROM Sales.Orders AS O2 WHERE O2.custid = 85`.
3. That returns `10739`. Compare with outer row's `10248`: no match → filtered out.
4. Repeat for every outer row.

**Debugging correlated subqueries:**
- You cannot highlight and run just the subquery — it references an undefined alias.
- Technique: substitute the correlation with a constant, test the subquery standalone, then put the correlation back.

**Percentage calculation using correlated subquery:**
```sql
-- Percentage of each order's value relative to customer's total
SELECT orderid, custid, val,
       CAST(100. * val / (SELECT SUM(O2.val)
                          FROM Sales.OrderValues AS O2
                          WHERE O2.custid = O1.custid)
            AS NUMERIC(5,2)) AS pct
FROM Sales.OrderValues AS O1
ORDER BY custid, orderid;
```

#### The EXISTS Predicate

`EXISTS(subquery)` returns TRUE if the subquery returns any rows, FALSE otherwise.

```sql
-- Customers from Spain who placed orders
SELECT custid, companyname
FROM Sales.Customers AS C
WHERE country = N'Spain'
  AND EXISTS (SELECT * FROM Sales.Orders AS O WHERE O.custid = C.custid);

-- Customers from Spain who placed NO orders
WHERE country = N'Spain'
  AND NOT EXISTS (SELECT * FROM Sales.Orders AS O WHERE O.custid = C.custid);
```

**Advantages of EXISTS:**
- English-like readability.
- **Short-circuit evaluation** — stops processing as soon as one row is found.
- Uses **two-valued logic** (not three-valued): always returns TRUE or FALSE, never UNKNOWN. NULL handling in the subquery is naturally excluded.
- `SELECT *` in EXISTS subquery is fine — the engine ignores the SELECT list anyway (only cares about row existence). Some prefer `SELECT 1` for clarity; no performance difference.

**EXISTS vs. NOT IN when NULLs are involved:**
- `NOT IN` can return an empty set if the subquery contains NULLs (see misbehaving subqueries).
- `NOT EXISTS` naturally handles NULLs correctly — it's safer.

### 4.3 Beyond the Fundamentals of Subqueries

#### Returning Previous or Next Values

"Previous" = "maximum value smaller than current value":
```sql
SELECT orderid, orderdate, empid, custid,
       (SELECT MAX(O2.orderid) FROM Sales.Orders AS O2
        WHERE O2.orderid < O1.orderid) AS prevorderid  -- NULL for first order
FROM Sales.Orders AS O1;
```

"Next" = "minimum value greater than current value":
```sql
SELECT orderid, orderdate, empid, custid,
       (SELECT MIN(O2.orderid) FROM Sales.Orders AS O2
        WHERE O2.orderid > O1.orderid) AS nextorderid  -- NULL for last order
FROM Sales.Orders AS O1;
```

> **Note:** SQL Server 2012 introduces `LAG` and `LEAD` window functions that handle this more elegantly and efficiently (Chapter 7).

#### Using Running Aggregates

```sql
-- Running total quantity by order year
SELECT orderyear, qty,
       (SELECT SUM(O2.qty)
        FROM Sales.OrderTotalsByYear AS O2
        WHERE O2.orderyear <= O1.orderyear) AS runqty
FROM Sales.OrderTotalsByYear AS O1
ORDER BY orderyear;
```

> **Note:** SQL Server 2012 window aggregate functions with framing handle this far more efficiently (Chapter 7).

#### Dealing with Misbehaving Subqueries

**NULL Trouble with NOT IN:**

If the subquery returns even one NULL, `NOT IN` always returns an empty result!

```sql
-- Insert a row with NULL custid
INSERT INTO Sales.Orders (custid, ...) VALUES(NULL, ...);

-- This query will now return EMPTY SET (not customers 22 and 57 as expected)
SELECT custid, companyname
FROM Sales.Customers
WHERE custid NOT IN (SELECT O.custid FROM Sales.Orders AS O);
-- Why: "22 NOT IN (1, 2, ..., NULL)" = NOT (... OR 22 = NULL)
--      = NOT (...FALSE... OR UNKNOWN) = NOT UNKNOWN = UNKNOWN → filtered out
```

**Fixes:**

Option 1 — Explicitly exclude NULLs:
```sql
WHERE custid NOT IN (SELECT O.custid FROM Sales.Orders AS O
                     WHERE O.custid IS NOT NULL)
```

Option 2 — Use NOT EXISTS (handles NULLs naturally):
```sql
SELECT custid, companyname
FROM Sales.Customers AS C
WHERE NOT EXISTS (SELECT * FROM Sales.Orders AS O WHERE O.custid = C.custid);
```

> **Best practice:** Always prefer `NOT EXISTS` over `NOT IN` when the subquery column could contain NULLs.

**Substitution Errors in Subquery Column Names:**

If a column name in the subquery doesn't exist in the subquery's table, SQL Server looks for it in the **outer table** — creating an unintended correlated subquery!

```sql
-- Table Sales.Orders has column "shipperid", NOT "shipper_id"
-- Table Sales.MyShippers has column "shipper_id" (with underscore)

-- BUG: meant to be self-contained, but becomes correlated
SELECT shipper_id, companyname
FROM Sales.MyShippers
WHERE shipper_id IN
    (SELECT shipper_id           -- No "shipper_id" in Orders → looks at outer table
     FROM Sales.Orders           -- Becomes: WHERE outer.shipper_id IN (SELECT outer.shipper_id ...)
     WHERE custid = 43);         -- Always TRUE (every shipper_id matches itself) → returns ALL shippers

-- CORRECT: prefix column names with table alias to force resolution error if wrong
WHERE shipper_id IN
    (SELECT O.shipperid          -- Error: column "shipper_id" not found in O → you catch the bug
     FROM Sales.Orders AS O
     WHERE O.custid = 43)
```

> **Best practice:** Always prefix column names in subqueries with the source table alias. If there's a naming error, you'll get a compile-time error instead of a silent logical bug.

---

## Chapter 5: Table Expressions {#chapter-5}

A **table expression** is a named query expression that represents a valid relational table. Can be used in DML statements like regular tables.

**Types:**
- **Derived tables** — defined in FROM clause, exist for the duration of the outer query.
- **CTEs (Common Table Expressions)** — defined with WITH, exist for the outer query.
- **Views** — stored in the database; reusable.
- **Inline TVFs (Table-Valued Functions)** — stored; support input parameters.

**Three requirements for any valid table expression query:**
1. **No guaranteed order** — no ORDER BY clause (unless with TOP or OFFSET-FETCH, but ORDER BY then serves filtering, not presentation).
2. **All columns must have names** — alias all expressions.
3. **All column names must be unique** — no duplicate column names.

Table expressions are **virtual** — not physically materialized. The outer query and inner query are merged into one query against the base tables.

### 5.1 Derived Tables

Defined in the FROM clause with parentheses, followed by an alias. Scope = the outer query only.

```sql
USE TSQL2012;
-- Simple derived table
SELECT *
FROM (SELECT custid, companyname
      FROM Sales.Customers
      WHERE country = N'USA') AS USACusts;
```

**Column aliasing in derived tables:**
- **Inline form** (preferred): `expression AS alias` inside the subquery.
- **External form**: list column names in parentheses after the derived table alias.

```sql
-- Problem: can't use GROUP BY with alias from SELECT
SELECT YEAR(orderdate) AS orderyear, COUNT(DISTINCT custid) AS numcusts
FROM Sales.Orders
GROUP BY orderyear;  -- ERROR: orderyear unknown at GROUP BY time

-- Solution using derived table (inline aliasing form):
SELECT orderyear, COUNT(DISTINCT custid) AS numcusts
FROM (SELECT YEAR(orderdate) AS orderyear, custid
      FROM Sales.Orders) AS D
GROUP BY orderyear;

-- External form alternative:
SELECT orderyear, COUNT(DISTINCT custid) AS numcusts
FROM (SELECT YEAR(orderdate), custid
      FROM Sales.Orders) AS D(orderyear, custid)
GROUP BY orderyear;
```

**Arguments in derived tables:**
```sql
DECLARE @empid AS INT = 3;
SELECT orderyear, COUNT(DISTINCT custid) AS numcusts
FROM (SELECT YEAR(orderdate) AS orderyear, custid
      FROM Sales.Orders
      WHERE empid = @empid) AS D    -- local variable used inside derived table
GROUP BY orderyear;
```

**Nesting (avoid when possible — hurts readability):**
```sql
-- Nested derived tables: D1 inside D2
SELECT orderyear, numcusts
FROM (SELECT orderyear, COUNT(DISTINCT custid) AS numcusts
      FROM (SELECT YEAR(orderdate) AS orderyear, custid
            FROM Sales.Orders) AS D1
      GROUP BY orderyear) AS D2
WHERE numcusts > 70;

-- Better alternative (no nesting, just repeat expression):
SELECT YEAR(orderdate) AS orderyear, COUNT(DISTINCT custid) AS numcusts
FROM Sales.Orders
GROUP BY YEAR(orderdate)
HAVING COUNT(DISTINCT custid) > 70;
```

**Multiple references problem:** Derived tables are defined *inside* the outer FROM clause — you cannot reference the same derived table twice. You must define it twice (duplicate code).

```sql
-- Must repeat the derived table definition for Cur and Prv:
SELECT Cur.orderyear,
       Cur.numcusts AS curnumcusts, Prv.numcusts AS prvnumcusts,
       Cur.numcusts - Prv.numcusts AS growth
FROM (SELECT YEAR(orderdate) AS orderyear, COUNT(DISTINCT custid) AS numcusts
      FROM Sales.Orders GROUP BY YEAR(orderdate)) AS Cur
LEFT OUTER JOIN
     (SELECT YEAR(orderdate) AS orderyear, COUNT(DISTINCT custid) AS numcusts
      FROM Sales.Orders GROUP BY YEAR(orderdate)) AS Prv
ON Cur.orderyear = Prv.orderyear + 1;
-- Code duplication is a maintenance problem
```

### 5.2 Common Table Expressions (CTEs)

CTEs are defined **before** the outer query using `WITH`. Key advantages over derived tables:
- No nesting (multiple CTEs separated by commas under one WITH).
- Can reference the same CTE multiple times.
- More readable and maintainable code.

**Basic syntax:**
```sql
WITH <CTE_Name>[(<column_list>)]
AS
(
    <inner_query>
)
<outer_query_against_CTE>;
```

**Important:** If there's a preceding statement in the same batch, terminate it with `;` before `WITH`. The CTE itself should also end with `;` (good practice).

```sql
WITH USACusts AS
(
    SELECT custid, companyname
    FROM Sales.Customers
    WHERE country = N'USA'
)
SELECT * FROM USACusts;
```

**Column aliasing (same options as derived tables):**
```sql
-- Inline form:
WITH C AS
(
    SELECT YEAR(orderdate) AS orderyear, custid
    FROM Sales.Orders
)
SELECT orderyear, COUNT(DISTINCT custid) AS numcusts
FROM C
GROUP BY orderyear;

-- External form:
WITH C(orderyear, custid) AS
(
    SELECT YEAR(orderdate), custid
    FROM Sales.Orders
)
SELECT orderyear, COUNT(DISTINCT custid) AS numcusts
FROM C
GROUP BY orderyear;
```

**Multiple CTEs (no nesting needed):**
```sql
WITH C1 AS
(
    SELECT YEAR(orderdate) AS orderyear, custid
    FROM Sales.Orders
),
C2 AS
(
    SELECT orderyear, COUNT(DISTINCT custid) AS numcusts
    FROM C1          -- C1 can reference previously defined CTEs
    GROUP BY orderyear
)
SELECT orderyear, numcusts
FROM C2
WHERE numcusts > 70;
```

**Multiple references (CTE advantage over derived tables):**
```sql
WITH YearlyCount AS
(
    SELECT YEAR(orderdate) AS orderyear,
           COUNT(DISTINCT custid) AS numcusts
    FROM Sales.Orders
    GROUP BY YEAR(orderdate)
)
SELECT Cur.orderyear,
       Cur.numcusts AS curnumcusts, Prv.numcusts AS prvnumcusts,
       Cur.numcusts - Prv.numcusts AS growth
FROM YearlyCount AS Cur
LEFT OUTER JOIN YearlyCount AS Prv    -- Same CTE referenced twice!
ON Cur.orderyear = Prv.orderyear + 1;
-- Code maintained in one place; much cleaner
```

> **Note:** Both CTE references are expanded during execution — no materialization occurs. If the inner query is expensive, consider persisting to a temp table for multiple references.

#### Recursive CTEs

A CTE with at least one **anchor member** (non-recursive) and one **recursive member** (references the CTE itself), separated by UNION ALL.

```sql
-- Return employee 2 and all subordinates at all levels
WITH EmpsCTE AS
(
    -- Anchor: starting point
    SELECT empid, mgrid, firstname, lastname
    FROM HR.Employees
    WHERE empid = 2

    UNION ALL

    -- Recursive: find direct subordinates of each row in previous result
    SELECT C.empid, C.mgrid, C.firstname, C.lastname
    FROM EmpsCTE AS P                -- P = previous result (parent)
    JOIN HR.Employees AS C           -- C = children/subordinates
        ON C.mgrid = P.empid
)
SELECT empid, mgrid, firstname, lastname
FROM EmpsCTE;
```

**How it works:**
1. Anchor executes once → returns employee 2.
2. Recursive member executes with previous result as input → returns employees 3, 5.
3. Recursive member executes again → returns employees 4, 6, 7, 8, 9.
4. Recursive member executes again → empty set → recursion stops.
5. Result = UNION ALL of all executions.

**Safety limit:** Default max recursion = 100. Customize with `OPTION(MAXRECURSION n)`. Use `MAXRECURSION 0` to remove limit (dangerous for infinite cycles).

**Traversing hierarchy upward (management chain):**
```sql
WITH EmpsCTE AS
(
    SELECT empid, mgrid, firstname, lastname
    FROM HR.Employees WHERE empid = 9  -- Anchor: start with employee 9

    UNION ALL

    SELECT P.empid, P.mgrid, P.firstname, P.lastname
    FROM EmpsCTE AS C                   -- C = current level (child/lower employee)
    JOIN HR.Employees AS P              -- P = parent/manager
        ON C.mgrid = P.empid
)
SELECT empid, mgrid, firstname, lastname FROM EmpsCTE;
-- Returns: 9→5→2→1 (Zoya → Sven → Don → Sara)
```

### 5.3 Views

Views are **reusable, stored** table expressions. Definitions stored in the database as permanent objects.

```sql
-- Create view
IF OBJECT_ID('Sales.USACusts') IS NOT NULL DROP VIEW Sales.USACusts;
GO
CREATE VIEW Sales.USACusts
AS
SELECT custid, companyname, contactname, contacttitle, address,
       city, region, postalcode, country, phone, fax
FROM Sales.Customers
WHERE country = N'USA';
GO

-- Query the view like a table
SELECT custid, companyname FROM Sales.USACusts;
```

**Views and ORDER BY:**
- ORDER BY is NOT allowed in a view definition (a view represents a relational table = no order).
- If you need ordered output, add ORDER BY in the outer query:
  ```sql
  SELECT custid, companyname, region FROM Sales.USACusts ORDER BY region;
  ```
- **Workaround using TOP (bad practice):**
  ```sql
  -- Technically valid but ORDER BY is NOT guaranteed to produce ordered output!
  CREATE VIEW Sales.USACusts AS
  SELECT TOP (100) PERCENT ...
  FROM Sales.Customers
  WHERE country = N'USA'
  ORDER BY region;  -- ORDER BY here serves TOP, NOT presentation
  ```
  > The view may return rows in any order even though ORDER BY is specified. Never rely on this.

**Refresh view metadata after table changes:**
```sql
EXEC sp_refreshview 'Sales.USACusts';
EXEC sp_refreshsqlmodule 'Sales.USACusts';
```
Better: always explicitly list column names (never `SELECT *` in a view).

#### View Options

**ENCRYPTION option:**
```sql
ALTER VIEW Sales.USACusts WITH ENCRYPTION
AS ...
-- Obfuscates the view definition; cannot be retrieved via OBJECT_DEFINITION or sp_helptext
```

**SCHEMABINDING option:**
```sql
ALTER VIEW Sales.USACusts WITH SCHEMABINDING
AS
SELECT custid, companyname, ...
FROM Sales.Customers       -- Must use two-part names (schema.object)
WHERE country = N'USA';    -- Cannot use SELECT *
-- Now: cannot DROP Customers or DROP/ALTER referenced columns
-- Prevents breaking the view through schema changes
```
Requirements for SCHEMABINDING:
- Use explicit two-part names (`Sales.Customers`, not just `Customers`).
- Cannot use `SELECT *`.

**CHECK OPTION:**
```sql
ALTER VIEW Sales.USACusts WITH SCHEMABINDING
AS
SELECT ...
FROM Sales.Customers
WHERE country = N'USA'
WITH CHECK OPTION;  -- Prevents INSERT/UPDATE through view if it would violate the filter
-- Now: inserting a UK customer through this view will fail
```

### 5.4 Inline Table-Valued Functions (Inline TVFs)

Inline TVFs are like **parameterized views** — views that accept input parameters.

```sql
-- Create inline TVF
IF OBJECT_ID('dbo.GetCustOrders') IS NOT NULL DROP FUNCTION dbo.GetCustOrders;
GO
CREATE FUNCTION dbo.GetCustOrders
(@cid AS INT) RETURNS TABLE
AS
RETURN
SELECT orderid, custid, empid, orderdate, ...
FROM Sales.Orders
WHERE custid = @cid;
GO

-- Query the TVF (pass arguments in parentheses)
SELECT orderid, custid FROM dbo.GetCustOrders(1) AS O;

-- Join with TVF
SELECT O.orderid, O.custid, OD.productid, OD.qty
FROM dbo.GetCustOrders(1) AS O
JOIN Sales.OrderDetails AS OD ON O.orderid = OD.orderid;
```

Like views, inline TVFs are expanded during query processing — no physical materialization.

**When to use views vs. inline TVFs:**
- No parameters needed → use a view.
- Parameters needed → use an inline TVF.

### 5.5 The APPLY Operator

APPLY is a T-SQL extension (not standard SQL; standard equivalent is `LATERAL`). Used in the FROM clause.

**Purpose:** Apply a table expression (right side) to each row of another table (left side). The right side can reference columns from the left side — making it like a correlated table subquery.

**Two forms:**
- `CROSS APPLY`: like INNER JOIN — excludes left rows where right returns empty set.
- `OUTER APPLY`: like LEFT OUTER JOIN — includes left rows where right returns empty set (with NULLs).

```sql
-- Return 3 most recent orders for each customer (CROSS APPLY)
SELECT C.custid, A.orderid, A.orderdate
FROM Sales.Customers AS C
CROSS APPLY
    (SELECT TOP (3) orderid, empid, orderdate, requireddate
     FROM Sales.Orders AS O
     WHERE O.custid = C.custid            -- correlation: references left table
     ORDER BY orderdate DESC, orderid DESC) AS A;
-- Customers with no orders (22, 57) are excluded (like INNER JOIN)

-- OUTER APPLY: include customers with no orders
SELECT C.custid, A.orderid, A.orderdate
FROM Sales.Customers AS C
OUTER APPLY
    (SELECT TOP (3) orderid, empid, orderdate, requireddate
     FROM Sales.Orders AS O
     WHERE O.custid = C.custid
     ORDER BY orderdate DESC, orderid DESC) AS A;
-- Customers 22, 57 included with NULLs in order columns

-- Using OFFSET-FETCH (SQL Server 2012+):
FROM Sales.Customers AS C
CROSS APPLY
    (SELECT orderid, ...
     FROM Sales.Orders AS O
     WHERE O.custid = C.custid
     ORDER BY orderdate DESC, orderid DESC
     OFFSET 0 ROWS FETCH FIRST 3 ROWS ONLY) AS A
```

**APPLY with inline TVF (cleaner encapsulation):**
```sql
-- Create TVF that returns top N orders for a given customer
CREATE FUNCTION dbo.TopOrders (@custid INT, @n INT) RETURNS TABLE AS RETURN
SELECT TOP (@n) orderid, empid, orderdate, requireddate
FROM Sales.Orders WHERE custid = @custid
ORDER BY orderdate DESC, orderid DESC;
GO

-- Apply to each customer
SELECT C.custid, C.companyname, A.orderid, A.empid, A.orderdate
FROM Sales.Customers AS C
CROSS APPLY dbo.TopOrders(C.custid, 3) AS A;
```

---

## Chapter 6: Set Operators {#chapter-6}

Set operators combine the result sets of two queries. The two queries must have:
- The **same number of columns**.
- **Compatible data types** (corresponding columns implicitly convertible).
- **No ORDER BY** in individual queries (ORDER BY can appear at the end, applied to the overall result).
- Column names in the result are determined by the **first query** (alias there if needed).

**Two flavors:**
- **DISTINCT** (default): eliminates duplicates → returns a true set.
- **ALL**: doesn't eliminate duplicates → returns a multiset.

> SQL Server supports ALL only for UNION. For INTERSECT ALL and EXCEPT ALL, workarounds exist.

**NULL comparison:** Set operators treat two NULLs as **equal** (unlike WHERE or JOIN, where NULL = NULL is UNKNOWN). This is a powerful advantage over equivalent JOIN/EXISTS solutions for NULL-containing data.

### 6.1 The UNION Operator

Unifies (combines) result sets of two queries.

**UNION ALL (multiset — no dedup, faster):**
```sql
SELECT country, region, city FROM HR.Employees
UNION ALL
SELECT country, region, city FROM Sales.Customers;
-- 9 + 91 = 100 rows; duplicates preserved
-- Faster than UNION (no dedup step)
```

**UNION (distinct — removes duplicates):**
```sql
SELECT country, region, city FROM HR.Employees
UNION
SELECT country, region, city FROM Sales.Customers;
-- 71 distinct rows; duplicates removed
```

**When to use which:**
- No potential duplicates → use UNION ALL (avoids overhead of dedup check).
- Potential duplicates, need all → use UNION ALL.
- Potential duplicates, need distinct → use UNION.

### 6.2 The INTERSECT Operator

Returns rows that appear in **both** input queries. NULLs in both sides are treated as equal.

**INTERSECT (distinct):**
```sql
SELECT country, region, city FROM HR.Employees
INTERSECT
SELECT country, region, city FROM Sales.Customers;
-- Returns: (UK, NULL, London), (USA, WA, Kirkland), (USA, WA, Seattle)
-- NULL handling: (UK, NULL, London) is matched because NULLs are treated as equal
```

Compare with alternatives:
```sql
-- INNER JOIN alternative (does NOT handle NULLs the same way)
SELECT DISTINCT E.country, E.region, E.city
FROM HR.Employees AS E
JOIN Sales.Customers AS C
    ON E.country = C.country
    AND E.region = C.region      -- NULL = NULL → UNKNOWN → (UK, NULL, London) EXCLUDED
    AND E.city = C.city;
-- This would miss the (UK, NULL, London) row!
```

**INTERSECT ALL (workaround — SQL Server 2012 doesn't have built-in ALL):**
Uses ROW_NUMBER to number occurrences, then INTERSECT:
```sql
WITH INTERSECT_ALL AS
(
    SELECT ROW_NUMBER()
               OVER(PARTITION BY country, region, city
                    ORDER BY (SELECT 0)) AS rownum,  -- ORDER BY (SELECT 0) = don't care about order
           country, region, city
    FROM HR.Employees
    INTERSECT
    SELECT ROW_NUMBER()
               OVER(PARTITION BY country, region, city
                    ORDER BY (SELECT 0)),
           country, region, city
    FROM Sales.Customers
)
SELECT country, region, city FROM INTERSECT_ALL;
-- Returns all intersecting occurrences (e.g., 4 rows for (UK, NULL, London))
```

**Tip:** `ORDER BY (SELECT 0)` or `ORDER BY (SELECT <constant>)` tells SQL Server that ordering doesn't matter — avoids sort overhead.

### 6.3 The EXCEPT Operator

Returns rows that appear in the **first** query but **NOT** in the second. EXCEPT is **asymmetric** (order matters).

**EXCEPT (distinct):**
```sql
-- Employee locations NOT in customer locations
SELECT country, region, city FROM HR.Employees
EXCEPT
SELECT country, region, city FROM Sales.Customers;
-- Returns: (USA, WA, Redmond), (USA, WA, Tacoma)

-- Customer locations NOT in employee locations
SELECT country, region, city FROM Sales.Customers
EXCEPT
SELECT country, region, city FROM HR.Employees;
-- Returns 66 rows
```

**EXCEPT ALL (workaround):**
```sql
WITH EXCEPT_ALL AS
(
    SELECT ROW_NUMBER() OVER(PARTITION BY country, region, city ORDER BY (SELECT 0)),
           country, region, city
    FROM HR.Employees
    EXCEPT
    SELECT ROW_NUMBER() OVER(PARTITION BY country, region, city ORDER BY (SELECT 0)),
           country, region, city
    FROM Sales.Customers
)
SELECT country, region, city FROM EXCEPT_ALL;
```

### 6.4 Precedence

`INTERSECT` has **higher precedence** than `UNION` and `EXCEPT` (which are equal).

```sql
-- INTERSECT is evaluated first (even though it appears second)
SELECT country, region, city FROM Production.Suppliers
EXCEPT
SELECT country, region, city FROM HR.Employees
INTERSECT                                    -- ← evaluated first
SELECT country, region, city FROM Sales.Customers;
-- Meaning: Supplier locations that are NOT (in both Employee AND Customer locations)

-- Use parentheses to override:
(SELECT country, region, city FROM Production.Suppliers
 EXCEPT
 SELECT country, region, city FROM HR.Employees)
INTERSECT
SELECT country, region, city FROM Sales.Customers;
-- Meaning: (Supplier locations MINUS Employee locations) that are also in Customer locations
```

**Always use parentheses** to make operator precedence explicit and clear.

### 6.5 Circumventing Unsupported Logical Phases

Individual queries in set operators cannot have ORDER BY. To add other logical phases to the combined result, use a table expression:

```sql
-- GROUP BY on the result of a UNION
SELECT country, COUNT(*) AS numlocations
FROM (SELECT country, region, city FROM HR.Employees
      UNION
      SELECT country, region, city FROM Sales.Customers) AS U
GROUP BY country;
```

**Ordering segments:** Use a sort column to separate the two inputs:
```sql
SELECT country, region, city
FROM (SELECT 1 AS sortcol, country, region, city FROM HR.Employees
      UNION ALL
      SELECT 2, country, region, city FROM Production.Suppliers) AS D
ORDER BY sortcol, country, region, city;
-- Employees first (sortcol=1), Suppliers second (sortcol=2), each sorted internally
```

**Using TOP within set operators (via table expressions):**
```sql
-- 2 most recent orders for employee 3, combined with 2 most recent for employee 5
SELECT empid, orderid, orderdate
FROM (SELECT TOP (2) empid, orderid, orderdate
      FROM Sales.Orders WHERE empid = 3
      ORDER BY orderdate DESC, orderid DESC) AS D1
UNION ALL
SELECT empid, orderid, orderdate
FROM (SELECT TOP (2) empid, orderid, orderdate
      FROM Sales.Orders WHERE empid = 5
      ORDER BY orderdate DESC, orderid DESC) AS D2;
```

---

## Chapter 7: Beyond the Fundamentals of Querying {#chapter-7}

### 7.1 Window Functions

A **window function** computes a scalar result for each row based on a calculation against a **subset (window) of rows** from the underlying query's result set. The window is defined with an `OVER` clause.

**Advantages over alternatives:**
- Unlike GROUP BY: preserves row-level detail while also providing aggregate information.
- Unlike subqueries: operates on the underlying query's result set (filters, joins, etc. already applied) without repeating logic.
- Allows separate ordering for the calculation vs. for presentation.
- Highly optimized for common patterns.

**Window specification components in OVER:**
1. **PARTITION BY:** divide rows into partitions; calculation restarts per partition.
2. **ORDER BY:** order rows within the partition (for rank/offset/framing purposes). ≠ presentation ORDER BY.
3. **Frame clause (ROWS/RANGE BETWEEN):** restrict to a subset within the ordered partition.

```sql
-- Running total values per employee
SELECT empid, ordermonth, val,
       SUM(val) OVER(PARTITION BY empid
                     ORDER BY ordermonth
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS runval
FROM Sales.EmpOrders;
```

**Where window functions are allowed:** Only in SELECT and ORDER BY clauses (evaluated during SELECT phase). To use in WHERE or GROUP BY, wrap in a table expression.

**Two major milestones:**
- SQL Server 2005: ranking window functions (complete), partial aggregate support (PARTITION BY only).
- SQL Server 2012: full aggregate support (ORDER BY + framing), plus LAG, LEAD, FIRST_VALUE, LAST_VALUE.

#### Ranking Window Functions

```sql
SELECT orderid, custid, val,
       ROW_NUMBER()   OVER(ORDER BY val) AS rownum,
       RANK()         OVER(ORDER BY val) AS rank,
       DENSE_RANK()   OVER(ORDER BY val) AS dense_rank,
       NTILE(100)     OVER(ORDER BY val) AS ntile
FROM Sales.OrderValues
ORDER BY val;
```

| Function | Ties | Description |
|----------|------|-------------|
| `ROW_NUMBER()` | Unique (arbitrary for ties) | Incrementing integers; must be unique |
| `RANK()` | Same rank for ties; gaps after | How many rows have a lower value |
| `DENSE_RANK()` | Same rank for ties; no gaps | How many **distinct** lower values |
| `NTILE(n)` | Distributes evenly | Assigns a tile number 1..n |

**Examples:**
- Values `{10, 20, 30, 30, 40}`:
  - ROW_NUMBER: 1, 2, 3, 4, 5 (ties get arbitrary distinct numbers)
  - RANK: 1, 2, 3, 3, 5 (gap after ties)
  - DENSE_RANK: 1, 2, 3, 3, 4 (no gap)

**ROW_NUMBER with DISTINCT:**
```sql
-- Non-deterministic if ORDER BY is not unique:
SELECT DISTINCT val, ROW_NUMBER() OVER(ORDER BY val) AS rownum
FROM Sales.OrderValues;
-- ROW_NUMBER is calculated BEFORE DISTINCT → assigns 830 unique numbers
-- DISTINCT then removes duplicates, but row numbers stay (no renumbering)
-- Result: 830 rows, not 795 distinct values!

-- To assign row numbers to distinct values, use GROUP BY:
SELECT val, ROW_NUMBER() OVER(ORDER BY val) AS rownum
FROM Sales.OrderValues
GROUP BY val;
-- GROUP BY reduces to 795 rows first, then ROW_NUMBER assigns 1-795
```

**PARTITION BY with ranking:**
```sql
-- Row number per customer, ordered by value
SELECT orderid, custid, val,
       ROW_NUMBER() OVER(PARTITION BY custid ORDER BY val) AS rownum
FROM Sales.OrderValues
ORDER BY custid, val;
-- Row numbers reset for each customer
```

#### Offset Window Functions

**LAG and LEAD (SQL Server 2012+):**
Access a value from a row at a specified offset before (LAG) or after (LEAD) the current row.

```sql
SELECT custid, orderid, val,
       LAG(val)  OVER(PARTITION BY custid ORDER BY orderdate, orderid) AS prevval,
       LEAD(val) OVER(PARTITION BY custid ORDER BY orderdate, orderid) AS nextval
FROM Sales.OrderValues;
-- LAG with 3 rows back and default 0: LAG(val, 3, 0) OVER(...)
```

Syntax: `LAG(element [, offset [, default]]) OVER(partition ORDER BY)`
- Default offset = 1 (immediately previous/next).
- Default value when no row at offset = NULL.
- No window frame clause (not applicable).

**FIRST_VALUE and LAST_VALUE (SQL Server 2012+):**
Return element from first/last row in the window frame.

```sql
SELECT custid, orderid, val,
       FIRST_VALUE(val) OVER(PARTITION BY custid
                              ORDER BY orderdate, orderid
                              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS firstval,
       LAST_VALUE(val)  OVER(PARTITION BY custid
                              ORDER BY orderdate, orderid
                              ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS lastval
FROM Sales.OrderValues
ORDER BY custid, orderdate, orderid;
```

**Critical note for LAST_VALUE:** The default frame when ORDER BY is specified (without ROWS/RANGE) is `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. This means without explicit framing, LAST_VALUE returns the current row's value! Always specify the frame explicitly with these functions.

#### Aggregate Window Functions

```sql
-- Grand total and customer total alongside each order
SELECT orderid, custid, val,
       SUM(val) OVER()                        AS totalvalue,    -- all rows
       SUM(val) OVER(PARTITION BY custid)      AS custtotalvalue -- per customer
FROM Sales.OrderValues;

-- Percentage calculations
SELECT orderid, custid, val,
       100. * val / SUM(val) OVER()                        AS pctall,
       100. * val / SUM(val) OVER(PARTITION BY custid)     AS pctcust
FROM Sales.OrderValues;
```

**With ORDER BY and framing (SQL Server 2012+):**
```sql
-- Running total
SUM(val) OVER(PARTITION BY empid
              ORDER BY ordermonth
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

-- Moving average (last 3 rows)
AVG(val) OVER(PARTITION BY empid
              ORDER BY ordermonth
              ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)

-- Full range
SUM(val) OVER(PARTITION BY empid
              ORDER BY ordermonth
              ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)

-- From 2 rows back to 1 row ahead
SUM(val) OVER(ORDER BY ordermonth ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING)
```

**Frame delimiters:**
- `UNBOUNDED PRECEDING` — beginning of partition
- `n PRECEDING` — n rows before current
- `CURRENT ROW` — current row
- `n FOLLOWING` — n rows after current
- `UNBOUNDED FOLLOWING` — end of partition

> **Note:** Avoid using the `RANGE` window frame unit in SQL Server 2012 — it's implemented in a very limited form.

### 7.2 Pivoting Data

**Pivoting = rotating rows to columns**, aggregating values.

Three phases:
1. **Grouping phase** (GROUP BY): what appears in rows.
2. **Spreading phase** (ON COLS): what appears in columns.
3. **Aggregation phase**: what value appears in the intersection.

**Setup table:**
```sql
CREATE TABLE dbo.Orders (orderid INT, orderdate DATE, empid INT, custid VARCHAR(5), qty INT, ...);
-- Contains orders for customers A, B, C, D from employees 1, 2, 3
```

**Goal:** Rows = employees, Columns = customers, Values = SUM(qty).

**Standard SQL pivoting:**
```sql
SELECT empid,
       SUM(CASE WHEN custid = 'A' THEN qty END) AS A,
       SUM(CASE WHEN custid = 'B' THEN qty END) AS B,
       SUM(CASE WHEN custid = 'C' THEN qty END) AS C,
       SUM(CASE WHEN custid = 'D' THEN qty END) AS D
FROM dbo.Orders
GROUP BY empid;
-- CASE returns qty only for matching custid; SUM aggregates
```

**Native PIVOT operator (T-SQL):**
```sql
SELECT empid, A, B, C, D
FROM (SELECT empid, custid, qty       -- Only include: grouping, spreading, aggregation columns!
      FROM dbo.Orders) AS D           -- Table expression filters out extra columns
PIVOT(SUM(qty) FOR custid IN(A, B, C, D)) AS P;
```

**Critical rule:** The source of PIVOT must contain **only** the three pivoting columns (grouping + spreading + aggregation). Any extra columns become additional grouping elements, which changes the result!

```sql
-- WRONG: operates directly on dbo.Orders (has orderid, orderdate, etc. → all become group keys)
SELECT empid, A, B, C, D FROM dbo.Orders PIVOT(SUM(qty) FOR custid IN(A, B, C, D)) AS P;
-- Returns one row per orderid × empid, not one row per empid!
```

**Pivoting with integer column names (must bracket):**
```sql
SELECT custid, [1], [2], [3]
FROM (SELECT empid, custid, qty FROM dbo.Orders) AS D
PIVOT(SUM(qty) FOR empid IN([1], [2], [3])) AS P;
```

**Dynamic pivoting (when you don't know the column values in advance):**
Requires dynamic SQL — construct the IN list by querying the data at runtime. See Chapter 10.

### 7.3 Unpivoting Data

**Unpivoting = rotating columns to rows**.

**Setup:**
```sql
CREATE TABLE dbo.EmpCustOrders (empid INT, A VARCHAR(5), B VARCHAR(5), C VARCHAR(5), D VARCHAR(5));
-- empid=1: A=NULL, B=20, C=34, D=NULL
-- empid=2: A=52,   B=27, C=NULL, D=NULL
-- empid=3: A=20,   B=NULL, C=22, D=30
```

**Standard SQL unpivoting (3 phases: copies → extract → filter NULLs):**
```sql
SELECT *
FROM (SELECT empid, custid,
             CASE custid           -- Phase 2: extract value for current custid
                 WHEN 'A' THEN A
                 WHEN 'B' THEN B
                 WHEN 'C' THEN C
                 WHEN 'D' THEN D
             END AS qty
      FROM dbo.EmpCustOrders
      CROSS JOIN (VALUES('A'),('B'),('C'),('D')) AS Custs(custid)  -- Phase 1: create copies
     ) AS D
WHERE qty IS NOT NULL;   -- Phase 3: eliminate irrelevant (NULL) intersections
```

**Native UNPIVOT operator:**
```sql
SELECT empid, custid, qty
FROM dbo.EmpCustOrders
UNPIVOT(qty FOR custid IN(A, B, C, D)) AS U;
-- UNPIVOT automatically eliminates NULL rows (always — unlike standard SQL approach)
-- So if you want to keep NULLs, use the standard approach instead
```

> **Note:** Unpivoting a pivoted table does NOT restore the original data — the aggregation in the pivot step loses detail information.

### 7.4 Grouping Sets

A **grouping set** is a set of attributes to group by. Normally, a single query defines a single grouping set. SQL Server supports requesting multiple grouping sets in one query.

**The problem:** To return aggregates for multiple grouping sets, you'd normally use UNION ALL:
```sql
SELECT empid, custid, SUM(qty) AS sumqty FROM dbo.Orders GROUP BY empid, custid
UNION ALL
SELECT empid, NULL, SUM(qty) FROM dbo.Orders GROUP BY empid
UNION ALL
SELECT NULL, custid, SUM(qty) FROM dbo.Orders GROUP BY custid
UNION ALL
SELECT NULL, NULL, SUM(qty) FROM dbo.Orders;
-- Multiple table scans; verbose code
```

#### GROUPING SETS subclause

```sql
SELECT empid, custid, SUM(qty) AS sumqty
FROM dbo.Orders
GROUP BY GROUPING SETS
(
    (empid, custid),   -- grouping set 1
    (empid),           -- grouping set 2
    (custid),          -- grouping set 3
    ()                 -- grand total (empty grouping set)
);
-- Single table scan; same result as UNION ALL version above
```

#### CUBE subclause

Generates **all possible subsets** (power set) of the specified elements:

```sql
GROUP BY CUBE(empid, custid)
-- Equivalent to GROUPING SETS((empid, custid), (empid), (custid), ())
-- CUBE(a, b, c) = all 8 possible grouping sets from {a, b, c}
```

#### ROLLUP subclause

Generates grouping sets assuming a **hierarchy** among the elements (rollup from most to least granular):

```sql
GROUP BY ROLLUP(YEAR(orderdate), MONTH(orderdate), DAY(orderdate))
-- Equivalent to GROUPING SETS(
--     (YEAR(orderdate), MONTH(orderdate), DAY(orderdate)),  -- most granular
--     (YEAR(orderdate), MONTH(orderdate)),
--     (YEAR(orderdate)),
--     ()                                                     -- grand total
-- )
-- ROLLUP(a, b, c) = 4 grouping sets (n+1 for n elements)
```

Useful for time hierarchies (year → month → day), organizational hierarchies, etc.

#### GROUPING and GROUPING_ID Functions

When a column is NULL in grouping sets output, it's ambiguous: is it a placeholder (grouping not participating) or actual NULL data?

**GROUPING(column):** Returns 0 if the column IS part of the current grouping set, 1 if it is NOT:
```sql
SELECT
    GROUPING(empid) AS grpemp,    -- 1 = empid not in this grouping set
    GROUPING(custid) AS grpcust,
    empid, custid, SUM(qty) AS sumqty
FROM dbo.Orders
GROUP BY CUBE(empid, custid);
```

**GROUPING_ID(col1, col2, ...):** Returns an integer bitmap where each bit represents whether a column is in the grouping set (0) or not (1):
```sql
SELECT
    GROUPING_ID(empid, custid) AS groupingset,
    empid, custid, SUM(qty) AS sumqty
FROM dbo.Orders
GROUP BY CUBE(empid, custid);
-- groupingset=0 (binary 00): both empid and custid are grouping columns
-- groupingset=1 (binary 01): empid only (custid not participating)
-- groupingset=2 (binary 10): custid only (empid not participating)
-- groupingset=3 (binary 11): neither (grand total)
```

**Practical use:** Identify rows by their grouping set for conditional formatting, reporting, filtering.

---
---

## Chapter 8: Data Modification {#chapter-8}

DML includes: SELECT, INSERT, UPDATE, DELETE, TRUNCATE, MERGE.

### 8.1 Inserting Data

#### The INSERT VALUES Statement

```sql
-- Basic single-row insert
INSERT INTO dbo.Orders(orderid, orderdate, empid, custid)
VALUES(10001, '20090212', 3, 'A');

-- Without specifying a column (uses default expression for orderdate):
INSERT INTO dbo.Orders(orderid, empid, custid)
VALUES(10002, 5, 'B');  -- orderdate gets DEFAULT (SYSDATETIME())

-- Multi-row insert (SQL Server 2008+):
INSERT INTO dbo.Orders(orderid, orderdate, empid, custid)
VALUES
    (10003, '20090213', 4, 'B'),
    (10004, '20090214', 1, 'A'),
    (10005, '20090213', 1, 'C'),
    (10006, '20090215', 3, 'C');
-- Atomic: if any row fails, none are inserted
```

**Column value resolution:**
- Explicit value specified → use it.
- No value specified → use DEFAULT if defined.
- No default → use NULL if column allows it.
- No default, NOT NULL → error.

**Table value constructor (standard SQL, SQL Server 2008+):**
```sql
-- Use VALUES as a derived table
SELECT *
FROM (VALUES
    (10003, '20090213', 4, 'B'),
    (10004, '20090214', 1, 'A')
) AS O(orderid, orderdate, empid, custid);
```

#### The INSERT SELECT Statement

```sql
INSERT INTO dbo.Orders(orderid, orderdate, empid, custid)
SELECT orderid, orderdate, empid, custid
FROM Sales.Orders
WHERE shipcountry = 'UK';
-- Atomic; same NULL/default handling as INSERT VALUES
```

**Legacy multi-row VALUES using UNION ALL (pre-2008 technique):**
```sql
INSERT INTO dbo.Orders(orderid, orderdate, empid, custid)
SELECT 10007, '20090215', 2, 'B' UNION ALL
SELECT 10008, '20090215', 1, 'C' UNION ALL
SELECT 10009, '20090216', 2, 'C';
-- Non-standard (SELECT without FROM); use VALUES clause instead in 2008+
```

#### The INSERT EXEC Statement

Insert a result set returned by a stored procedure or dynamic SQL batch:

```sql
INSERT INTO dbo.Orders(orderid, orderdate, empid, custid)
EXEC Sales.usp_getorders @country = 'France';
```

#### The SELECT INTO Statement

Creates a new table and populates it with query results. **Non-standard T-SQL**.

```sql
IF OBJECT_ID('dbo.Orders', 'U') IS NOT NULL DROP TABLE dbo.Orders;
SELECT orderid, orderdate, empid, custid
INTO dbo.Orders        -- creates the table
FROM Sales.Orders;
```

**What it copies:** Column names, data types, nullability, identity property, data.  
**What it does NOT copy:** Constraints, indexes, triggers, permissions.

**Performance benefit:** Minimally logged (when Recovery Model ≠ FULL) → very fast.

**With set operations:**
```sql
SELECT country, region, city
INTO dbo.Locations
FROM Sales.Customers
EXCEPT
SELECT country, region, city FROM HR.Employees;
```

> **Note:** Windows Azure SQL Database does NOT support SELECT INTO (creates heap; SQL Database requires clustered index). Use CREATE TABLE + INSERT SELECT instead.

#### The BULK INSERT Statement

```sql
BULK INSERT dbo.Orders FROM 'c:\temp\orders.txt'
WITH
(
    DATAFILETYPE   = 'char',
    FIELDTERMINATOR = ',',
    ROWTERMINATOR   = '\n'
);
-- Can run in minimally logged mode if requirements are met
```

### 8.2 The Identity Property and the Sequence Object

#### Identity Property

Automatically generates values for a column on INSERT.

```sql
CREATE TABLE dbo.T1
(
    keycol  INT          NOT NULL IDENTITY(1, 1)   -- seed=1, increment=1
                         CONSTRAINT PK_T1 PRIMARY KEY,
    datacol VARCHAR(10)  NOT NULL
);

-- INSERT: completely ignore the identity column
INSERT INTO dbo.T1(datacol) VALUES('AAAAA');  -- keycol auto-assigned = 1
INSERT INTO dbo.T1(datacol) VALUES('CCCCC');  -- keycol = 2
INSERT INTO dbo.T1(datacol) VALUES('BBBBB');  -- keycol = 3

-- Reference identity column generically:
SELECT $identity FROM dbo.T1;  -- Same as SELECT keycol
```

**Getting newly generated identity value:**
```sql
-- @@identity: last identity generated by the session (any scope, any table)
-- SCOPE_IDENTITY(): last identity in the current scope (RECOMMENDED)
-- IDENT_CURRENT('table'): current identity in the table, regardless of session

DECLARE @new_key AS INT;
INSERT INTO dbo.T1(datacol) VALUES('AAAAA');
SET @new_key = SCOPE_IDENTITY();
SELECT @new_key AS new_key;  -- Returns 4
```

> **Always use SCOPE_IDENTITY()** — `@@identity` can return wrong value if a trigger on the table does another INSERT to a different table.

**IDENT_CURRENT from any session:**
```sql
SELECT SCOPE_IDENTITY() AS [SCOPE_IDENTITY],
       @@identity        AS [@@identity],
       IDENT_CURRENT('dbo.T1') AS [IDENT_CURRENT];
-- From a new session: SCOPE_IDENTITY and @@identity = NULL; IDENT_CURRENT = last value
```

**Gaps in identity values:**
- A failed INSERT still **consumes** the identity value. The next INSERT gets seed+increment.
- Rollbacks do NOT undo identity changes.
- Result: gaps are normal when some INSERTs fail.

**Explicitly inserting a specific value:**
```sql
SET IDENTITY_INSERT dbo.T1 ON;
INSERT INTO dbo.T1(keycol, datacol) VALUES(5, 'FFFFF');  -- explicit value
SET IDENTITY_INSERT dbo.T1 OFF;
-- If explicit value > current identity → current identity updates
-- If explicit value < current identity → current identity stays (no change)
```

**Limitations of identity:**
- Cannot add identity property to an existing column.
- Cannot remove it from an existing column.
- Cannot update it.
- Tied to a specific table — you can't share a sequence across tables.
- Does NOT enforce uniqueness by itself (add PK or unique constraint for that).

#### Sequence Object (SQL Server 2012+)

A database object (not tied to any table) that generates a sequence of numbers.

```sql
-- Create a sequence
CREATE SEQUENCE dbo.SeqOrderIDs AS INT
    MINVALUE 1
    CYCLE;      -- CYCLE means it wraps around; default is NO CYCLE
-- Not specified: MAXVALUE defaults to max of INT, INCREMENT defaults to 1, START WITH defaults to MINVALUE

-- Modify a sequence
ALTER SEQUENCE dbo.SeqOrderIDs NO CYCLE;
ALTER SEQUENCE dbo.SeqOrderIDs RESTART WITH 1;

-- Generate next value
SELECT NEXT VALUE FOR dbo.SeqOrderIDs;  -- Returns 1 (first call)
SELECT NEXT VALUE FOR dbo.SeqOrderIDs;  -- Returns 2

-- Store in variable and use in INSERT
DECLARE @neworderid AS INT = NEXT VALUE FOR dbo.SeqOrderIDs;
INSERT INTO dbo.T1(keycol, datacol) VALUES(@neworderid, 'a');

-- Use directly in INSERT
INSERT INTO dbo.T1(keycol, datacol)
VALUES(NEXT VALUE FOR dbo.SeqOrderIDs, 'b');

-- Use in UPDATE!
UPDATE dbo.T1 SET keycol = NEXT VALUE FOR dbo.SeqOrderIDs;

-- Query current value
SELECT current_value FROM sys.sequences WHERE OBJECT_ID = OBJECT_ID('dbo.SeqOrderIDs');
```

**Controlled ordering in multi-row INSERT:**
```sql
INSERT INTO dbo.T1(keycol, datacol)
SELECT NEXT VALUE FOR dbo.SeqOrderIDs OVER(ORDER BY hiredate),  -- ordered assignment
       LEFT(firstname, 1) + LEFT(lastname, 1)
FROM HR.Employees;
```

**As a default constraint:**
```sql
ALTER TABLE dbo.T1
ADD CONSTRAINT DFT_T1_keycol
    DEFAULT (NEXT VALUE FOR dbo.SeqOrderIDs)
    FOR keycol;
-- Now you can INSERT without specifying keycol
INSERT INTO dbo.T1(datacol) VALUES('c');
-- This is impossible with identity! Identity can't be added to existing columns.
```

**Allocate a range at once:**
```sql
DECLARE @first AS SQL_VARIANT;
EXEC sys.sp_sequence_get_range
    @sequence_name     = N'dbo.SeqOrderIDs',
    @range_size        = 1000,
    @range_first_value = @first OUTPUT;
SELECT @first;  -- First value in the range; next 999 are already "reserved"
```

**Sequence advantages over identity:**
- Not tied to a table — can use across multiple tables.
- Can generate value before using it (store in variable).
- Can use in UPDATE statements.
- Can add/remove default constraint to existing table.
- Supports cycling, min/max values independently of type limits.

**Limitation:** Like identity, gaps can occur when transactions are rolled back.

### 8.3 Deleting Data

#### The DELETE Statement

```sql
-- Delete rows matching filter
DELETE FROM dbo.Orders
WHERE orderdate < '20070101';
-- Fully logged; may be slow for large deletions

-- DELETE without WHERE: deletes all rows (fully logged, slow)
DELETE FROM dbo.T1;
```

#### The TRUNCATE Statement

```sql
TRUNCATE TABLE dbo.T1;
-- Deletes ALL rows; minimally logged → much faster
-- Atomic (transactional despite common misconception)
-- Resets identity column to seed value (DELETE does not reset identity)
-- NOT allowed if table is referenced by any FK constraint (even disabled, even if referencing table is empty)
```

**Protect production table from accidental TRUNCATE/DROP:**
Create a dummy table with a foreign key pointing to the production table (even if disabled). Prevents truncation or dropping.

#### DELETE Based on a Join (T-SQL non-standard)

Delete rows from one table based on a filter involving another table:

```sql
-- Delete orders placed by US customers
DELETE FROM O
FROM dbo.Orders AS O
JOIN dbo.Customers AS C
    ON O.custid = C.custid
WHERE C.country = N'USA';
-- First FROM specifies FROM clause (with JOIN); second FROM is the target alias
```

**Standard alternative using subquery:**
```sql
DELETE FROM dbo.Orders
WHERE EXISTS
    (SELECT * FROM dbo.Customers AS C
     WHERE Orders.custid = C.custid AND C.country = N'USA');
```

### 8.4 Updating Data

#### The UPDATE Statement

```sql
-- Basic update
UPDATE dbo.OrderDetails
SET discount = discount + 0.05
WHERE productid = 51;

-- Compound assignment operators (SQL Server 2008+)
UPDATE dbo.OrderDetails SET discount += 0.05 WHERE productid = 51;
-- Other: -=, *=, /=, %=

-- Swapping column values (all-at-once)
UPDATE dbo.T1 SET col1 = col2, col2 = col1;
-- Both assignments use pre-update values → correct swap
```

#### UPDATE Based on a Join (T-SQL non-standard)

```sql
-- Update discount for orders of customer 1
UPDATE OD
SET discount += 0.05
FROM dbo.OrderDetails AS OD
JOIN dbo.Orders AS O
    ON OD.orderid = O.orderid
WHERE O.custid = 1;
```

**When JOIN version is significantly better than subquery:**
```sql
-- Update T1 columns from T2 (join version — one access to T2)
UPDATE T1
SET col1 = T2.col1, col2 = T2.col2, col3 = T2.col3
FROM dbo.T1 JOIN dbo.T2 ON T2.keycol = T1.keycol
WHERE T2.col4 = 'ABC';

-- Standard subquery version (three separate accesses to T2!)
UPDATE dbo.T1
SET col1 = (SELECT col1 FROM dbo.T2 WHERE T2.keycol = T1.keycol),
    col2 = (SELECT col2 FROM dbo.T2 WHERE T2.keycol = T1.keycol),
    col3 = (SELECT col3 FROM dbo.T2 WHERE T2.keycol = T1.keycol)
WHERE EXISTS (SELECT * FROM dbo.T2 WHERE T2.keycol = T1.keycol AND T2.col4 = 'ABC');
-- Much more verbose and less efficient
```

#### Assignment UPDATE (T-SQL non-standard)

Updates data AND assigns values to variables in the same statement — one table access:

```sql
-- Custom sequence mechanism (guarantees no gaps)
CREATE TABLE dbo.Sequences (id VARCHAR(10) NOT NULL PRIMARY KEY, val INT NOT NULL);
INSERT INTO dbo.Sequences VALUES('SEQ1', 0);

DECLARE @nextval AS INT;
UPDATE dbo.Sequences
SET @nextval = val += 1  -- val = val + 1; then @nextval = new val
WHERE id = 'SEQ1';
SELECT @nextval;  -- Next value in the sequence
-- Atomic; single table access; more efficient than separate UPDATE + SELECT
```

### 8.5 Merging Data (MERGE Statement)

MERGE applies different actions (INSERT, UPDATE, DELETE) based on whether source rows match target rows. Standard SQL with T-SQL extensions.

```sql
-- Upsert: update existing customers, insert new customers
MERGE INTO dbo.Customers AS TGT
USING dbo.CustomersStage AS SRC
    ON TGT.custid = SRC.custid           -- join condition defines match/non-match
WHEN MATCHED THEN
    UPDATE SET TGT.companyname = SRC.companyname,
               TGT.phone = SRC.phone,
               TGT.address = SRC.address
WHEN NOT MATCHED THEN                    -- source row has no match in target
    INSERT (custid, companyname, phone, address)
    VALUES (SRC.custid, SRC.companyname, SRC.phone, SRC.address);
-- IMPORTANT: MERGE must be terminated with a semicolon!
```

**Three clause types:**
- `WHEN MATCHED THEN`: action when source row has a matching target row.
- `WHEN NOT MATCHED [BY TARGET] THEN`: action when source row has no match (INSERT only).
- `WHEN NOT MATCHED BY SOURCE THEN`: action when target row has no matching source row (DELETE or UPDATE).

```sql
-- Full upsert with delete of non-matching target rows
MERGE dbo.Customers AS TGT
USING dbo.CustomersStage AS SRC ON TGT.custid = SRC.custid
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...
WHEN NOT MATCHED BY SOURCE THEN DELETE;
```

**Conditional matching with AND:**
```sql
-- Only update if attributes actually changed (avoids unnecessary updates)
WHEN MATCHED AND
    (TGT.companyname <> SRC.companyname
     OR TGT.phone <> SRC.phone
     OR TGT.address <> SRC.address) THEN
    UPDATE SET ...
```

### 8.6 Modifying Data Through Table Expressions

You can use INSERT, UPDATE, DELETE, MERGE on derived tables, CTEs, views, and inline TVFs — the modification is applied to the underlying base tables.

**Restrictions:**
- Can only modify one side of a join in one statement.
- Cannot update computed columns.
- INSERT must provide values for columns without implicit values.

**Primary use case 1 — Troubleshooting (view SELECT vs UPDATE):**
```sql
-- CTE approach: highlight just the SELECT for debugging, then change to UPDATE
WITH C AS
(
    SELECT custid, OD.orderid, productid, discount, discount + 0.05 AS newdiscount
    FROM dbo.OrderDetails AS OD
    JOIN dbo.Orders AS O ON OD.orderid = O.orderid
    WHERE O.custid = 1
)
UPDATE C
SET discount = newdiscount;
```

**Primary use case 2 — Window functions in UPDATE:**
```sql
-- Cannot use ROW_NUMBER() directly in UPDATE SET clause
-- Window functions only allowed in SELECT and ORDER BY

-- WORKAROUND: use CTE
WITH C AS
(
    SELECT col1, col2, ROW_NUMBER() OVER(ORDER BY col1) AS rownum
    FROM dbo.T1
)
UPDATE C SET col2 = rownum;
```

### 8.7 Modifications with TOP and OFFSET-FETCH

```sql
-- Delete 50 rows (arbitrary — no order guarantee)
DELETE TOP(50) FROM dbo.Orders;

-- Update 50 rows (arbitrary)
UPDATE TOP(50) dbo.Orders SET freight += 10.00;

-- To control which rows are affected: use table expressions with TOP
WITH C AS
(
    SELECT TOP(50) * FROM dbo.Orders ORDER BY orderid       -- 50 lowest order IDs
)
DELETE FROM C;

WITH C AS
(
    SELECT TOP(50) * FROM dbo.Orders ORDER BY orderid DESC  -- 50 highest order IDs
)
UPDATE C SET freight += 10.00;

-- OFFSET-FETCH equivalent (SQL Server 2012+)
WITH C AS
(
    SELECT * FROM dbo.Orders ORDER BY orderid
    OFFSET 0 ROWS FETCH FIRST 50 ROWS ONLY
)
DELETE FROM C;
```

### 8.8 The OUTPUT Clause

Returns information from modified rows. Syntax similar to SELECT clause.

**Prefixes:**
- `inserted.*`: row image **after** the change (available for INSERT, UPDATE, MERGE).
- `deleted.*`: row image **before** the change (available for DELETE, UPDATE, MERGE).

```sql
-- Redirect output to a table: add INTO <table>
-- Return to caller AND store: use two OUTPUT clauses (one with INTO, one without)
```

#### INSERT with OUTPUT

```sql
-- Return all newly generated identity values
INSERT INTO dbo.T1(datacol)
OUTPUT inserted.keycol, inserted.datacol  -- Return inserted rows
SELECT lastname FROM HR.Employees WHERE country = N'USA';

-- Direct to table variable:
DECLARE @NewRows TABLE(keycol INT, datacol NVARCHAR(40));
INSERT INTO dbo.T1(datacol)
OUTPUT inserted.keycol, inserted.datacol
INTO @NewRows
SELECT lastname FROM HR.Employees WHERE country = N'UK';
SELECT * FROM @NewRows;
```

#### DELETE with OUTPUT

```sql
DELETE FROM dbo.Orders
OUTPUT deleted.orderid, deleted.orderdate, deleted.empid, deleted.custid
WHERE orderdate < '20080101';
-- Returns information about deleted rows
-- Add INTO <archive_table> to archive deleted rows
```

#### UPDATE with OUTPUT

```sql
UPDATE dbo.OrderDetails
SET discount += 0.05
OUTPUT
    inserted.productid,
    deleted.discount AS olddiscount,    -- before update
    inserted.discount AS newdiscount    -- after update
WHERE productid = 51;
-- Returns both old and new values for auditing
```

#### MERGE with OUTPUT

```sql
MERGE INTO dbo.Customers AS TGT
USING dbo.CustomersStage AS SRC ON TGT.custid = SRC.custid
WHEN MATCHED THEN UPDATE SET ...
WHEN NOT MATCHED THEN INSERT ...
OUTPUT $action AS theaction,    -- 'INSERT', 'UPDATE', or 'DELETE'
       inserted.custid,
       deleted.companyname AS oldcompanyname,
       inserted.companyname AS newcompanyname;
-- For INSERT actions: deleted.* columns are NULL
```

#### Composable DML

Insert only a *subset* of modified rows into a target table:

```sql
INSERT INTO dbo.ProductsAudit(productid, colname, oldval, newval)
SELECT productid, N'unitprice', oldval, newval
FROM (UPDATE dbo.Products
      SET unitprice *= 1.15
      OUTPUT inserted.productid,
             deleted.unitprice AS oldval,
             inserted.unitprice AS newval
      WHERE supplierid = 1) AS D         -- D is the OUTPUT result set
WHERE oldval < 20.0 AND newval >= 20.0;  -- Filter only the interesting changes
-- Only rows crossing the $20 threshold are audited
```

---

## Chapter 9: Transactions and Concurrency {#chapter-9}

### 9.1 Transactions

A **transaction** is a unit of work that is treated atomically — all or nothing.

**Explicit transaction boundaries:**
```sql
BEGIN TRAN;
    INSERT INTO Sales.Orders (...) VALUES (...);
    SET @neworderid = SCOPE_IDENTITY();
    INSERT INTO Sales.OrderDetails (...) VALUES (@neworderid, ...);
COMMIT TRAN;   -- Confirm and make permanent

-- To cancel:
ROLLBACK TRAN;  -- Undo all changes since BEGIN TRAN
```

**Implicit transactions (default):** Each individual statement is its own transaction (auto-committed). Change with `SET IMPLICIT_TRANSACTIONS ON` (then you must explicitly COMMIT or ROLLBACK).

**Check if in an open transaction:** `SELECT @@TRANCOUNT` — 0 if no open transaction, >0 if one is open.

#### ACID Properties

| Property | Description |
|----------|-------------|
| **Atomicity** | All or nothing. On system failure, SQL Server rolls forward (redo) committed changes and rolls back uncommitted ones on restart. |
| **Consistency** | Database transitions from one consistent state to another; integrity constraints are maintained. |
| **Isolation** | Transactions are isolated from each other; controlled via isolation levels. |
| **Durability** | Committed changes are permanent. Written to transaction log first (Write-Ahead Logging); recovered on restart. |

### 9.2 Locks and Blocking

#### Lock Modes and Compatibility

**Two main lock modes:**

**Exclusive lock (X):**
- Required to **modify** data.
- Cannot be obtained if another transaction holds ANY lock on the resource.
- No other transaction can obtain ANY lock on a resource with an exclusive lock.
- Held until **end of transaction** (always — cannot be changed).

**Shared lock (S):**
- Required to **read** data (under default READ COMMITTED isolation in on-premises SQL Server).
- Multiple transactions can hold shared locks on the same resource simultaneously.
- Cannot be obtained if another transaction holds an exclusive lock.
- Duration and behavior can be controlled via isolation level.

**Lock compatibility table:**

| Requested \ Granted | X (Exclusive) | S (Shared) | IX (Intent Exclusive) | IS (Intent Shared) |
|---------------------|:---:|:---:|:---:|:---:|
| X (Exclusive) | No | No | No | No |
| S (Shared) | No | Yes | No | Yes |
| IX (Intent Exclusive) | No | No | Yes | Yes |
| IS (Intent Shared) | No | Yes | Yes | Yes |

**Summary:** Data being modified (X lock) cannot be read OR modified by others (at default isolation). Data being read (S lock) cannot be modified by others (at default isolation).

**Intent locks (IX, IS):** Obtained at higher granularity levels (table, page) to indicate intent to lock lower levels. Allows efficient conflict detection at coarser granularity without scanning all lower-level locks.

#### Lockable Resource Types

Resource types (granularity from finest to coarsest): `KEY` (row in index), `PAGE`, `OBJECT` (table), `DATABASE`, plus others (extents, allocation units, etc.)

**To lock a row (KEY), SQL Server also acquires:**
- Intent Exclusive/Shared lock on the PAGE containing the row.
- Intent Exclusive/Shared lock on the OBJECT (table).

**Lock escalation:** When a statement acquires ≥ 5,000 locks, SQL Server attempts to escalate to a table lock. Can be controlled with `ALTER TABLE ... SET LOCK_ESCALATION`.

#### Troubleshooting Blocking

**Default behavior:** Blocked requests wait indefinitely until the blocker releases the lock.

**Set lock timeout (per session):**
```sql
SET LOCK_TIMEOUT 5000;   -- 5 seconds; returns error after timeout
SET LOCK_TIMEOUT 0;      -- Immediate timeout (error if lock not immediately available)
SET LOCK_TIMEOUT -1;     -- Default: wait indefinitely
```

**Querying lock information:**
```sql
-- Active locks and waits
SELECT request_session_id AS spid, resource_type AS restype,
       DB_NAME(resource_database_id) AS dbname,
       resource_description AS res, resource_associated_entity_id AS resid,
       request_mode AS mode, request_status AS status
FROM sys.dm_tran_locks;

-- Connection info (last SQL batch run by each connection)
SELECT session_id, text
FROM sys.dm_exec_connections
CROSS APPLY sys.dm_exec_sql_text(most_recent_sql_handle) AS ST
WHERE session_id IN(52, 53);

-- Session details (who, what host, program)
SELECT session_id AS spid, login_time, host_name, program_name,
       login_name, last_request_start_time
FROM sys.dm_exec_sessions
WHERE session_id IN(52, 53);

-- Blocking details (which session is blocking which)
SELECT session_id AS spid, blocking_session_id, command,
       wait_type, wait_time, wait_resource
FROM sys.dm_exec_requests
WHERE blocking_session_id > 0;  -- Only blocked requests

-- Terminate a blocking session:
KILL 52;  -- Rolls back the blocking transaction; not available in SQL Database
```

### 9.3 Isolation Levels

Isolation levels control the consistency/concurrency trade-off for **readers**.

- Writers always use exclusive locks held until end of transaction (cannot change).
- Readers: default behavior varies by edition/configuration.
- **On-premises SQL Server default:** READ COMMITTED (readers use shared locks).
- **Windows Azure SQL Database default:** READ COMMITTED SNAPSHOT (readers use row versioning).

**Set isolation level:**
```sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;  -- session level
SELECT ... FROM T1 WITH (READCOMMITTEDLOCK)       -- query level hint
SELECT ... FROM T1 WITH (NOLOCK)                  -- = READ UNCOMMITTED
SELECT ... FROM T1 WITH (HOLDLOCK)                -- = SERIALIZABLE
```

#### Traditional Isolation Levels (Based on Locking)

**READ UNCOMMITTED (lowest):**
- Reader does NOT request a shared lock.
- Can read **dirty reads** (uncommitted changes by other transactions).
- Reader never blocks writers; writers never block readers.
- Use when data consistency is not critical and maximum concurrency is needed.
```sql
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
```

**READ COMMITTED (default on-premises):**
- Reader requests shared lock; releases it immediately after reading the resource.
- Prevents dirty reads; allows non-repeatable reads (data can change between reads in same transaction).
- Reader blocked by writers (waits for exclusive lock to release).
- Writer blocked by readers holding shared locks? No — shared lock is released as soon as the read finishes (not held to end of transaction).
```sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
```

**REPEATABLE READ:**
- Shared locks held until **end of transaction**.
- Prevents dirty reads AND non-repeatable reads.
- Readers blocked by writers; writers blocked by readers.
- **Lost update prevention:** Both transactions holding shared locks can't escalate to exclusive → deadlock detected.
- Still allows **phantom reads** (new rows satisfying the filter can be inserted by others).
```sql
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

**SERIALIZABLE (highest traditional):**
- Like REPEATABLE READ + prevents phantom reads.
- Locks the *key range* that satisfies the query filter — blocks insertion of new rows that would qualify for the query's filter.
- Highest consistency, lowest concurrency.
```sql
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

#### Isolation Levels Based on Row Versioning

Store previous committed versions of rows in **tempdb**. Readers don't take shared locks → never blocked by writers.

**Enabling (on-premises):**
```sql
-- Enable SNAPSHOT isolation:
ALTER DATABASE TSQL2012 SET ALLOW_SNAPSHOT_ISOLATION ON;

-- Enable READ_COMMITTED_SNAPSHOT (changes meaning of READ COMMITTED):
-- Must be the only connection to the database!
ALTER DATABASE TSQL2012 SET READ_COMMITTED_SNAPSHOT ON;
```

**Note:** Enabling row versioning increases overhead on DELETE and UPDATE operations (must copy row to version store in tempdb).

**READ COMMITTED SNAPSHOT:**
- Reader gets the last committed version of the row that was available when the **statement** started.
- No shared locks; readers never block.
- Allows non-repeatable reads (consistent with standard READ COMMITTED behavior).
- **Default in Windows Azure SQL Database.**
- Enabled by `SET READ_COMMITTED_SNAPSHOT ON` on the database.
- Changes the semantics of READ COMMITTED isolation level automatically.

**SNAPSHOT:**
- Reader gets the last committed version of the row that was available when the **transaction** started.
- No shared locks; readers never block.
- Prevents: dirty reads, non-repeatable reads, phantom reads (like SERIALIZABLE).
- **Detects update conflicts:** if another transaction modified data between your read and write → transaction fails with error 3960.
- Must be explicitly enabled and explicitly set in session.

```sql
-- Using SNAPSHOT
SET TRANSACTION ISOLATION LEVEL SNAPSHOT;
BEGIN TRAN;
SELECT ... FROM Production.Products WHERE productid = 2;
-- Gets version from when transaction started
COMMIT TRAN;
```

**Conflict detection (SNAPSHOT):**
```sql
-- Session 1: read, calculate, write
BEGIN TRAN;
SELECT unitprice FROM Production.Products WHERE productid = 2;  -- reads 19.00
-- Meanwhile Session 2: UPDATE Production.Products SET unitprice = 25.00 WHERE productid = 2
UPDATE Production.Products SET unitprice = 20.00 WHERE productid = 2;
-- Error 3960: Snapshot isolation transaction aborted due to update conflict
```

**Resolution:** Use error handling to retry the transaction.

**Disabling row versioning isolation levels:**
```sql
ALTER DATABASE TSQL2012 SET ALLOW_SNAPSHOT_ISOLATION OFF;
ALTER DATABASE TSQL2012 SET READ_COMMITTED_SNAPSHOT OFF;
```

#### Summary of Isolation Levels

| Isolation Level | Dirty Read | Non-Repeatable Read | Lost Update | Phantom Read | Update Conflict Detection | Row Versioning |
|----------------|:---:|:---:|:---:|:---:|:---:|:---:|
| READ UNCOMMITTED | Yes | Yes | Yes | Yes | No | No |
| READ COMMITTED | No | Yes | Yes | Yes | No | No |
| READ COMMITTED SNAPSHOT | No | Yes | Yes | Yes | No | **Yes** |
| REPEATABLE READ | No | No | No | Yes | No | No |
| SERIALIZABLE | No | No | No | No | No | No |
| SNAPSHOT | No | No | No | No | **Yes** | **Yes** |

### 9.4 Deadlocks

A **deadlock** occurs when two or more processes block each other — each waiting for the other to release a lock. Without intervention, they'd wait forever.

SQL Server **detects deadlocks** (typically within a few seconds) and terminates one transaction (the "victim") with error 1205.

**Victim selection:**
- By default: the transaction that has done the least work (cheapest rollback).
- Override with `SET DEADLOCK_PRIORITY <-10 to 10>`: lower priority → more likely to be chosen as victim. In ties, amount of work decides.

```sql
SET DEADLOCK_PRIORITY LOW;   -- This session prefers to be the victim
SET DEADLOCK_PRIORITY HIGH;  -- This session prefers not to be the victim
```

**Classic deadlock example:**
```sql
-- Session 1: modifies Products.productid=2 (X lock)
BEGIN TRAN; UPDATE Production.Products SET unitprice += 1.00 WHERE productid = 2;
-- Session 2: modifies OrderDetails for productid=2 (X lock on OD)
BEGIN TRAN; UPDATE Sales.OrderDetails SET unitprice += 1.00 WHERE productid = 2;
-- Session 1: tries to read OD.productid=2 → blocked by S2's X lock
SELECT ... FROM Sales.OrderDetails WHERE productid = 2;  -- BLOCKED
-- Session 2: tries to read Products.productid=2 → blocked by S1's X lock
SELECT ... FROM Production.Products WHERE productid = 2;  -- BLOCKED
-- DEADLOCK: S1 waits for S2; S2 waits for S1. SQL Server kills one.
```

#### Deadlock Mitigation Strategies

**1. Keep transactions short:** Move activities outside the transaction that don't need to be part of the unit of work.

**2. Consistent access order:** If both transactions access the same resources, access them in the **same order**.
```sql
-- DEADLOCK-PRONE: S1 accesses Products then OrderDetails; S2 accesses OrderDetails then Products
-- DEADLOCK-FREE: both sessions access Products first, then OrderDetails
```

**3. Good index design:** Without good indexes, queries scan (and lock) all rows even for selective predicates, causing unnecessary lock conflicts. Proper indexes allow targeted, narrow locks.

**4. Use row versioning isolation:** Under READ COMMITTED SNAPSHOT, readers don't take shared locks — deadlocks involving shared locks are eliminated.

---

## Chapter 10: Programmable Objects {#chapter-10}

### 10.1 Variables

Variables store temporary values within a batch.

```sql
-- Declaration and initialization
DECLARE @i AS INT;
SET @i = 10;

-- SQL Server 2008+: declare and initialize in one statement
DECLARE @i AS INT = 10;

-- Assignment with scalar subquery
DECLARE @empname AS NVARCHAR(31);
SET @empname = (SELECT firstname + N' ' + lastname
                FROM HR.Employees WHERE empid = 3);
-- Error if subquery returns more than one row

-- Multi-variable assignment from same row (nonstandard assignment SELECT):
DECLARE @firstname AS NVARCHAR(10), @lastname AS NVARCHAR(20);
SELECT @firstname = firstname, @lastname = lastname
FROM HR.Employees WHERE empid = 3;
-- If multiple rows qualify: variables get values from the LAST row accessed (undefined order!)
-- SET with scalar subquery is safer (fails if multiple rows)
```

### 10.2 Batches

A **batch** is one or more T-SQL statements sent to SQL Server as a single unit for parsing, resolution, optimization, and execution.

**GO command:** Client tool command (SSMS, SQLCMD) marking end of a batch. Not a T-SQL server command.

**GO n:** Execute the batch n times:
```sql
INSERT INTO dbo.T1 DEFAULT VALUES;
GO 100    -- Inserts 100 rows with incrementing identity values
```

**Batch rules:**

1. **Syntax error in batch → entire batch is rejected** (not submitted):
```sql
PRINT 'First batch'; USE TSQL2012; GO   -- Valid: executes
PRINT 'Second batch'; SELECT custid FOM Sales.Customers; GO  -- Syntax error: entire batch rejected
PRINT 'Third batch'; SELECT empid FROM HR.Employees; GO      -- Valid: executes
```

2. **Variables are local to their batch** — cannot be referenced in another batch:
```sql
DECLARE @i AS INT = 10;
PRINT @i;  -- Works: 10
GO
PRINT @i;  -- Error: "Must declare scalar variable @i"
```

3. **Statements that cannot be combined with others in same batch:**
CREATE DEFAULT, CREATE FUNCTION, CREATE PROCEDURE, CREATE RULE, CREATE SCHEMA, CREATE TRIGGER, CREATE VIEW must be the **first statement** in a batch.
```sql
-- WRONG:
IF OBJECT_ID('Sales.MyView', 'V') IS NOT NULL DROP VIEW Sales.MyView;
CREATE VIEW Sales.MyView AS SELECT ...; -- Error: CREATE VIEW must be first in batch
GO

-- CORRECT:
IF OBJECT_ID('Sales.MyView', 'V') IS NOT NULL DROP VIEW Sales.MyView;
GO                                       -- End of first batch
CREATE VIEW Sales.MyView AS SELECT ...;  -- First statement of new batch
GO
```

4. **Batches are units of resolution:** Object/column existence is checked at parse time. Schema changes and data access in the same batch can fail:
```sql
ALTER TABLE dbo.T1 ADD col2 INT;
SELECT col1, col2 FROM dbo.T1;  -- Error: col2 doesn't exist at parse time (batch beginning)
GO
-- FIX: separate into different batches
ALTER TABLE dbo.T1 ADD col2 INT; GO
SELECT col1, col2 FROM dbo.T1;  -- Now works
```

### 10.3 Flow Elements

#### IF ... ELSE

```sql
IF YEAR(SYSDATETIME()) <> YEAR(DATEADD(day, 1, SYSDATETIME()))
    PRINT 'Today is the last day of the year.';
ELSE
    PRINT 'Today is not the last day of the year.';

-- Nested IF
IF YEAR(SYSDATETIME()) <> YEAR(DATEADD(day, 1, SYSDATETIME()))
    PRINT 'Last day of the year.';
ELSE
    IF MONTH(SYSDATETIME()) <> MONTH(DATEADD(day, 1, SYSDATETIME()))
        PRINT 'Last day of the month but not year.';
    ELSE
        PRINT 'Not the last day of the month.';

-- Multiple statements: use BEGIN...END blocks
IF DAY(SYSDATETIME()) = 1
BEGIN
    PRINT 'First day of month.';
    /* ... process code ... */
END
ELSE
BEGIN
    PRINT 'Not first day.';
    /* ... process code ... */
END
```

**Three-valued logic in IF:** ELSE activates when predicate is FALSE **OR UNKNOWN**. If you need different treatment for UNKNOWN (e.g., NULL), add explicit IS NULL test.

#### WHILE

```sql
DECLARE @i AS INT = 1;
WHILE @i <= 10
BEGIN
    PRINT @i;
    SET @i = @i + 1;
END;
-- Prints 1 through 10

-- BREAK: exit the loop immediately
WHILE @i <= 10
BEGIN
    IF @i = 6 BREAK;  -- Loop exits when @i reaches 6
    PRINT @i;
    SET @i = @i + 1;
END;

-- CONTINUE: skip rest of current iteration
DECLARE @i AS INT = 0;
WHILE @i < 10
BEGIN
    SET @i = @i + 1;
    IF @i = 6 CONTINUE;  -- Skip PRINT for @i = 6
    PRINT @i;
END;
-- Prints 1, 2, 3, 4, 5, 7, 8, 9, 10
```

**Create and populate a Numbers table:**
```sql
CREATE TABLE dbo.Numbers(n INT NOT NULL PRIMARY KEY);
DECLARE @i AS INT = 1;
WHILE @i <= 1000
BEGIN
    INSERT INTO dbo.Numbers(n) VALUES(@i);
    SET @i = @i + 1;
END
```

### 10.4 Cursors

A **cursor** allows row-by-row processing of a result set. The author strongly recommends defaulting to **set-based solutions** instead of cursors.

**Why set-based is usually better:**
- Cursors go against the relational model (set theory).
- Row-by-row overhead is much slower than set-based operations.
- Cursor code is lengthy, less readable, harder to maintain.

**When cursors might be appropriate:**
- Administrative tasks that must be performed for each row (e.g., backup each database in instance).
- Running aggregates in SQL Server versions before 2012 (window functions are now much better).
- When set-based solutions perform badly and cursor solution genuinely uses less data access.

**Cursor pattern (6 steps):**
```sql
-- 1. Declare cursor
DECLARE C CURSOR FAST_FORWARD  -- read-only, forward-only (most efficient)
FOR SELECT custid, ordermonth, qty
    FROM Sales.CustOrders ORDER BY custid, ordermonth;

-- 2. Open cursor
OPEN C;

-- 3. Fetch first row
FETCH NEXT FROM C INTO @custid, @ordermonth, @qty;

-- 4. Loop while rows exist
WHILE @@FETCH_STATUS = 0
BEGIN
    -- process current row
    FETCH NEXT FROM C INTO @custid, @ordermonth, @qty;  -- advance cursor
END

-- 5. Close cursor
CLOSE C;

-- 6. Deallocate cursor
DEALLOCATE C;
```

**Running aggregates with cursor (pre-2012 approach):**
```sql
DECLARE @Result TABLE (custid INT, ordermonth DATETIME, qty INT, runqty INT, PRIMARY KEY(custid, ordermonth));
DECLARE @custid INT, @prvcustid INT, @ordermonth DATETIME, @qty INT, @runqty INT;

DECLARE C CURSOR FAST_FORWARD FOR
    SELECT custid, ordermonth, qty FROM Sales.CustOrders ORDER BY custid, ordermonth;
OPEN C;
FETCH NEXT FROM C INTO @custid, @ordermonth, @qty;
SELECT @prvcustid = @custid, @runqty = 0;

WHILE @@FETCH_STATUS = 0
BEGIN
    IF @custid <> @prvcustid SELECT @prvcustid = @custid, @runqty = 0;  -- reset per customer
    SET @runqty = @runqty + @qty;
    INSERT INTO @Result VALUES(@custid, @ordermonth, @qty, @runqty);
    FETCH NEXT FROM C INTO @custid, @ordermonth, @qty;
END
CLOSE C; DEALLOCATE C;

SELECT custid, CONVERT(VARCHAR(7), ordermonth, 121) AS ordermonth, qty, runqty
FROM @Result ORDER BY custid, ordermonth;
```

**SQL Server 2012+ replacement for running aggregates:**
```sql
SELECT custid, ordermonth, qty,
       SUM(qty) OVER(PARTITION BY custid
                     ORDER BY ordermonth
                     ROWS UNBOUNDED PRECEDING) AS runqty
FROM Sales.CustOrders ORDER BY custid, ordermonth;
-- Far more concise and efficient!
```

### 10.5 Temporary Tables

Three kinds of temporary tables — all physically stored in `tempdb`.

#### Local Temporary Tables

```sql
-- Name starts with single # 
CREATE TABLE #MyOrderTotalsByYear (orderyear INT NOT NULL PRIMARY KEY, qty INT NOT NULL);
INSERT INTO #MyOrderTotalsByYear ...
SELECT * FROM #MyOrderTotalsByYear;
```

- Visible only to the **creating session** (and inner levels of the call stack).
- Created at the outermost batch level: lasts until the session disconnects.
- Created inside a procedure: destroyed when the procedure ends.
- SQL Server adds a unique suffix internally to avoid name conflicts between sessions.

**Useful for:** Storing expensive intermediate results that need to be accessed multiple times (avoids re-executing the expensive query).

```sql
-- Store aggregated result once; join twice
CREATE TABLE #MyOrderTotalsByYear (orderyear INT NOT NULL PRIMARY KEY, qty INT NOT NULL);
INSERT INTO #MyOrderTotalsByYear
SELECT YEAR(O.orderdate) AS orderyear, SUM(OD.qty) AS qty
FROM Sales.Orders AS O JOIN Sales.OrderDetails AS OD ON OD.orderid = O.orderid
GROUP BY YEAR(orderdate);

SELECT Cur.orderyear, Cur.qty AS curyearqty, Prv.qty AS prvyearqty
FROM #MyOrderTotalsByYear AS Cur
LEFT OUTER JOIN #MyOrderTotalsByYear AS Prv
    ON Cur.orderyear = Prv.orderyear + 1;

DROP TABLE #MyOrderTotalsByYear;  -- Clean up explicitly
```

#### Global Temporary Tables

```sql
-- Name starts with ##
CREATE TABLE dbo.##Globals (id sysname NOT NULL PRIMARY KEY, val SQL_VARIANT NOT NULL);
INSERT INTO dbo.##Globals(id, val) VALUES(N'i', CAST(10 AS INT));
SELECT val FROM dbo.##Globals WHERE id = N'i';
DROP TABLE dbo.##Globals;  -- Explicit drop
```

- Visible to **all sessions**.
- Automatically destroyed when creating session disconnects AND no active references remain.
- Anyone can modify or drop it — no special permissions needed (and no protection).
- Useful for sharing temporary data across sessions.
- **Not supported by Windows Azure SQL Database.**

#### Table Variables

```sql
DECLARE @MyOrderTotalsByYear TABLE
(
    orderyear INT NOT NULL PRIMARY KEY,
    qty       INT NOT NULL
);
INSERT INTO @MyOrderTotalsByYear ...
SELECT * FROM @MyOrderTotalsByYear;
-- No explicit drop needed; goes out of scope at end of batch
```

- Visible only to the **current batch** (not to inner batches in call stack, not to subsequent batches).
- Changes are NOT fully rolled back on ROLLBACK TRAN (unlike temp tables).
- Physically in tempdb, not in memory (common misconception).
- Performance: Use table variables for very small data sets (few rows); use local temp tables for larger data.

#### Table Types (SQL Server 2008+)

Reusable table definition stored in the database:

```sql
CREATE TYPE dbo.OrderTotalsByYear AS TABLE
(
    orderyear INT NOT NULL PRIMARY KEY,
    qty       INT NOT NULL
);

-- Use as variable type:
DECLARE @MyOrderTotalsByYear AS dbo.OrderTotalsByYear;

-- Most powerful use: input parameters in stored procedures and functions
```

### 10.6 Dynamic SQL

Construct a T-SQL batch as a character string and execute it.

**Use cases:**
- Automate admin tasks (e.g., backup each database).
- Improve plan reuse for ad-hoc queries with parameters.
- Dynamic pivoting (column list not known at design time).

**Security warning:** Never concatenate unvalidated user input into SQL strings — exposes you to **SQL injection attacks**. Use parameters instead.

#### EXEC Command

```sql
DECLARE @sql AS VARCHAR(100);
SET @sql = 'PRINT ''This message was printed by dynamic SQL.'';';
EXEC(@sql);
-- Supports both regular and Unicode character strings
```

#### sp_executesql Stored Procedure

More secure and better for plan reuse:

```sql
DECLARE @sql AS NVARCHAR(100);
SET @sql = N'SELECT orderid, custid, empid, orderdate
             FROM Sales.Orders
             WHERE orderid = @orderid;';  -- parameter in the query string

EXEC sp_executesql
    @stmt   = @sql,
    @params = N'@orderid AS INT',  -- parameter declaration
    @orderid = 10248;              -- parameter value assignment
-- Even when called again with a different orderid, the query string stays the same
-- → execution plan can be reused from cache
```

Advantages over EXEC:
- **Input/output parameters** → prevents SQL injection (parameters are not code).
- Better **execution plan reuse** (parameterized query string is stable).
- Only supports **Unicode** (NVARCHAR) strings.

#### Dynamic PIVOT

```sql
-- Determine columns dynamically from data
DECLARE @sql AS NVARCHAR(1000), @orderyear AS INT, @first AS INT = 1;
SET @sql = N'SELECT * FROM (SELECT shipperid, YEAR(orderdate) AS orderyear, freight FROM Sales.Orders) AS D
             PIVOT(SUM(freight) FOR orderyear IN(';

DECLARE C CURSOR FAST_FORWARD FOR
    SELECT DISTINCT YEAR(orderdate) AS orderyear FROM Sales.Orders ORDER BY orderyear;
OPEN C;
FETCH NEXT FROM C INTO @orderyear;

WHILE @@FETCH_STATUS = 0
BEGIN
    IF @first = 0 SET @sql = @sql + N',';
    ELSE SET @first = 0;
    SET @sql = @sql + QUOTENAME(@orderyear);  -- QUOTENAME wraps in [brackets] for safety
    FETCH NEXT FROM C INTO @orderyear;
END
CLOSE C; DEALLOCATE C;

SET @sql = @sql + N')) AS P;';
EXEC sp_executesql @stmt = @sql;
```

### 10.7 Routines

Routines encapsulate code. Three types: UDFs, stored procedures, triggers.
T-SQL or .NET CLR code can implement routines. Use T-SQL for data manipulation; .NET CLR for computation-heavy, string, or iterative tasks.

#### User-Defined Functions (UDFs)

- Calculates something; returns a result. No side effects allowed.
- Side effects prohibited: cannot modify data/schema, cannot use RAND() or NEWID() (they have side effects on internal state).
- Scalar UDFs: return a single value; can appear in SELECT, WHERE, etc.
- Table UDFs: return a table; can appear in FROM clause.

```sql
-- Create scalar UDF: calculate age
IF OBJECT_ID('dbo.GetAge') IS NOT NULL DROP FUNCTION dbo.GetAge;
GO
CREATE FUNCTION dbo.GetAge
(
    @birthdate AS DATE,
    @eventdate AS DATE
)
RETURNS INT
AS
BEGIN
    RETURN
        DATEDIFF(year, @birthdate, @eventdate)
        - CASE WHEN 100 * MONTH(@eventdate) + DAY(@eventdate)
                    < 100 * MONTH(@birthdate) + DAY(@birthdate)
               THEN 1 ELSE 0 END;
    -- Trick: 100*month + day creates a comparable integer (e.g., Feb 12 → 212)
    -- Subtract 1 if the birthday hasn't occurred yet this year
END;
GO

-- Use in query
SELECT empid, firstname, lastname, birthdate,
       dbo.GetAge(birthdate, SYSDATETIME()) AS age
FROM HR.Employees;
```

**Note:** Schema-qualify function calls with two-part name (`dbo.GetAge`, not just `GetAge`).

#### Stored Procedures

Encapsulate T-SQL code with input/output parameters. Can have side effects (modify data, schema).

```sql
IF OBJECT_ID('Sales.GetCustomerOrders', 'P') IS NOT NULL DROP PROC Sales.GetCustomerOrders;
GO
CREATE PROC Sales.GetCustomerOrders
    @custid    AS INT,
    @fromdate  AS DATETIME = '19000101',   -- default value
    @todate    AS DATETIME = '99991231',
    @numrows   AS INT OUTPUT                -- output parameter
AS
SET NOCOUNT ON;  -- Suppress "n rows affected" messages
SELECT orderid, custid, empid, orderdate
FROM Sales.Orders
WHERE custid = @custid
  AND orderdate >= @fromdate
  AND orderdate < @todate;
SET @numrows = @@ROWCOUNT;   -- @@ROWCOUNT = rows affected by last statement
GO

-- Execute procedure
DECLARE @rc AS INT;
EXEC Sales.GetCustomerOrders
    @custid    = 1,
    @fromdate  = '20070101',
    @todate    = '20080101',
    @numrows   = @rc OUTPUT;   -- OUTPUT keyword required to receive output value
SELECT @rc AS numrows;
```

**Benefits of stored procedures:**
- **Encapsulation:** Change logic in one place, affects all callers.
- **Security:** Grant EXECUTE on procedure without granting direct table access. Prevents SQL injection when procedures replace ad-hoc SQL.
- **Error handling:** All error logic in one place.
- **Performance:** Plan reuse by default; reduced network traffic (only proc name + params sent, not full SQL).

#### Triggers

Special stored procedures that fire automatically when an event occurs. Cannot be executed explicitly.

**DML triggers (on data events):**
- `AFTER`: fires after the DML statement completes. Only on permanent tables.
- `INSTEAD OF`: fires instead of the DML statement. On tables or views.
- Triggers fire **per statement**, not per modified row.

**Special tables inside trigger body:**
- `inserted`: new image of affected rows (available in INSERT and UPDATE triggers).
- `deleted`: old image of affected rows (available in DELETE and UPDATE triggers).

**AFTER INSERT trigger example:**
```sql
CREATE TRIGGER trg_T1_insert_audit ON dbo.T1 AFTER INSERT
AS
SET NOCOUNT ON;
INSERT INTO dbo.T1_Audit(keycol, datacol)
SELECT keycol, datacol FROM inserted;  -- inserted = newly inserted rows
GO
```

**DDL triggers (on schema events):**
```sql
CREATE TRIGGER trg_audit_ddl_events
ON DATABASE FOR DDL_DATABASE_LEVEL_EVENTS  -- all DDL events on this database
AS
SET NOCOUNT ON;
DECLARE @eventdata AS XML = eventdata();   -- XML with event info
INSERT INTO dbo.AuditDDLEvents(posttime, eventtype, loginname, schemaname, objectname, ...)
VALUES(
    @eventdata.value('(/EVENT_INSTANCE/PostTime)[1]', 'VARCHAR(23)'),
    @eventdata.value('(/EVENT_INSTANCE/EventType)[1]', 'sysname'),
    @eventdata.value('(/EVENT_INSTANCE/LoginName)[1]', 'sysname'),
    ...
);
GO

-- Scopes: ON DATABASE (database-level events) or ON ALL SERVER (server-level events)
-- SQL Database: only supports ON DATABASE
```

**Trigger notes:**
- Trigger is part of the transaction that caused it to fire.
- `ROLLBACK TRAN` inside a trigger undoes both the trigger's work AND the causing statement.

### 10.8 Error Handling

**TRY...CATCH:**
```sql
BEGIN TRY
    -- Normal code
    PRINT 10/0;  -- Error occurs here
    PRINT 'No error';  -- This line is skipped
END TRY
BEGIN CATCH
    PRINT 'Error!';   -- Control jumps here
END CATCH;
```

- If no error in TRY block → CATCH is skipped.
- If error in TRY block → control goes to CATCH block.
- If error is handled in CATCH → caller doesn't see the error.

**Error functions (available in CATCH block):**
| Function | Returns |
|----------|---------|
| `ERROR_NUMBER()` | Error number (most important for conditional handling) |
| `ERROR_MESSAGE()` | Error message text |
| `ERROR_SEVERITY()` | Error severity (integer) |
| `ERROR_STATE()` | Error state (integer) |
| `ERROR_LINE()` | Line number where error occurred |
| `ERROR_PROCEDURE()` | Procedure name (NULL if not in a procedure) |

**Comprehensive error handling:**
```sql
BEGIN TRY
    INSERT INTO dbo.Employees(empid, empname, mgrid)
    VALUES(1, 'Emp1', NULL);
END TRY
BEGIN CATCH
    IF ERROR_NUMBER() = 2627
        PRINT 'Handling PK violation...';
    ELSE IF ERROR_NUMBER() = 547
        PRINT 'Handling CHECK/FK constraint violation...';
    ELSE IF ERROR_NUMBER() = 515
        PRINT 'Handling NULL violation...';
    ELSE IF ERROR_NUMBER() = 245
        PRINT 'Handling conversion error...';
    ELSE
    BEGIN
        PRINT 'Re-throwing error...';
        THROW;  -- SQL Server 2012+: re-raise the error to the caller
    END
    
    -- Log error details
    PRINT 'Error Number : ' + CAST(ERROR_NUMBER() AS VARCHAR(10));
    PRINT 'Error Message: ' + ERROR_MESSAGE();
    PRINT 'Error Line   : ' + CAST(ERROR_LINE() AS VARCHAR(10));
    PRINT 'Error Proc   : ' + COALESCE(ERROR_PROCEDURE(), 'Not within proc');
END CATCH;
```

**Reusable error handling in a procedure:**
```sql
CREATE PROC dbo.ErrInsertHandler AS
SET NOCOUNT ON;
IF ERROR_NUMBER() = 2627 PRINT 'Handling PK violation...';
ELSE IF ERROR_NUMBER() = 547 PRINT 'Handling CHECK/FK violation...';
...
GO

-- Use in calling code
BEGIN TRY
    INSERT INTO dbo.Employees ... ;
END TRY
BEGIN CATCH
    IF ERROR_NUMBER() IN (2627, 547, 515, 245)
        EXEC dbo.ErrInsertHandler;
    ELSE
        THROW;
END CATCH;
```

**Common error numbers:**
- 2627: Primary key violation.
- 547: Check constraint or foreign key violation.
- 515: NULL constraint violation.
- 245: Conversion error.
- 1205: Deadlock (transaction chosen as victim).
- 1222: Lock timeout expired.

---

## Appendix A: Getting Started {#appendix}

### Sample Database: TSQL2012

The book uses the **TSQL2012** sample database. Download source code and the creation script from **http://tsql.solidq.com**.

**Key tables and their schemas:**
| Schema | Table | Description |
|--------|-------|-------------|
| Sales | Orders | Order headers |
| Sales | OrderDetails | Order line items |
| Sales | Customers | Customer info |
| Sales | Shippers | Shipping companies |
| HR | Employees | Employee info (with self-referencing FK for management hierarchy) |
| Production | Products | Product catalog |
| Production | Categories | Product categories |
| Production | Suppliers | Product suppliers |
| Stats | Tests, Scores | Statistics sample data |
| dbo | Nums | Auxiliary numbers table (integers 1..N) |

**Key views:**
- `Sales.OrderValues` — Orders with total values
- `Sales.EmpOrders` — Employee order summary by month
- `Sales.CustOrders` — Customer order summary by month
- `Sales.OrderTotalsByYear` — Total quantities by year

### Installing SQL Server

**Editions to use for practice:** Any SQL Server 2012 edition except SQL Server Compact.
- Free trial: http://www.microsoft.com/sqlserver/en/us/get-sql-server/try-it.aspx
- MSDN subscribers: SQL Server 2012 Developer edition.

**Installation steps:**
1. Create a service account (local user with Password Never Expires).
2. Install prerequisites (.NET Framework 3.5 SP1, .NET Framework 4).
3. Run setup.exe → choose "New SQL Server Stand-Alone Installation".
4. Select features: Database Engine Services, Client Tools Connectivity, Documentation Components, Management Tools - Complete.
5. Choose instance configuration (default or named instance).
6. Set service accounts to use the created service account.
7. Set Authentication Mode to Windows Authentication; add current user as administrator.

### Working with SQL Server Management Studio (SSMS)

SSMS is the primary tool for writing and executing T-SQL code.

**Key features:**
- **Object Explorer:** Browse databases, tables, views, procedures, etc. Drag items to query window.
- **Query Window:** Write and execute T-SQL. Press **F5** to execute (or click Execute button).
- **Results To Grid** (default), **Results To Text**, **Results To File** — control output format.
- If code is highlighted, only the highlighted portion executes (partial execution).
- **Alt + highlight** = rectangular block selection for copying/executing rectangular regions.
- **Shift+F1** = load Books Online for the keyword at cursor position.
- **Tip:** Drag a table's Columns folder to the query window → SQL Server lists all column names.

**Connection options:**
- On-premises: Windows Authentication, server name only.
- SQL Database: SQL Server Authentication, server.database.windows.net.

### SQL Server Books Online

The official SQL Server documentation — your best reference for syntax, behavior, and system objects.

**Access:**
- From program group: SQL Server program group → Documentation & Community → SQL Server Documentation.
- Online: http://msdn.microsoft.com/en-us/library/ms130214(v=SQL.110).aspx
- Can download and install locally (check Help Library Manager for updates).

**Search methods:**
- **Index tab:** Type the keyword or function name to navigate sorted list.
- **Contents tab:** Navigate tree for broad topics ("What's New", "T-SQL Programming Reference").
- **Search box:** Full-text search across all documentation.
- **Ctrl+F:** Find text within the current open article.
- **Add to Favorites:** Save frequently referenced pages.

---

## Quick Reference: Key T-SQL Patterns

### Logical Query Processing Order
```
FROM → WHERE → GROUP BY → HAVING → SELECT (Expressions → DISTINCT) → ORDER BY (TOP/OFFSET-FETCH)
```

### NULL Handling Summary
```sql
WHERE col IS NULL          -- Correct way to test for NULL
WHERE col IS NOT NULL      -- Correct way to test for not NULL
COALESCE(col, 'default')   -- Replace NULL with a value (standard)
ISNULL(col, 'default')     -- T-SQL equivalent (non-standard)

-- In GROUP BY/ORDER BY: NULLs are equal (grouped/sorted together)
-- In WHERE/JOIN: NULL = NULL → UNKNOWN → filtered out
-- In UNIQUE constraint (SQL Server): NULLs are equal (only one NULL allowed)
-- In set operators (UNION/INTERSECT/EXCEPT): NULLs are equal
```

### Window Function Templates
```sql
-- Ranking
ROW_NUMBER() OVER([PARTITION BY cols] ORDER BY cols)
RANK()        OVER([PARTITION BY cols] ORDER BY cols)
DENSE_RANK()  OVER([PARTITION BY cols] ORDER BY cols)
NTILE(n)      OVER([PARTITION BY cols] ORDER BY cols)

-- Aggregate
SUM(col) OVER([PARTITION BY cols] [ORDER BY cols] [ROWS BETWEEN ... AND ...])
COUNT(*) OVER([PARTITION BY cols])
AVG(col) OVER([PARTITION BY cols] [ORDER BY cols] [ROWS BETWEEN ... AND ...])

-- Offset (SQL Server 2012+)
LAG(col [, offset [, default]])  OVER([PARTITION BY cols] ORDER BY cols)
LEAD(col [, offset [, default]]) OVER([PARTITION BY cols] ORDER BY cols)
FIRST_VALUE(col) OVER([PARTITION BY cols] ORDER BY cols ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
LAST_VALUE(col)  OVER([PARTITION BY cols] ORDER BY cols ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
```

### Pivoting Template
```sql
-- Standard SQL:
SELECT grouping_col,
       SUM(CASE WHEN spreading_col = 'val1' THEN aggregation_col END) AS val1,
       SUM(CASE WHEN spreading_col = 'val2' THEN aggregation_col END) AS val2
FROM source_table GROUP BY grouping_col;

-- Native PIVOT:
SELECT grouping_col, val1, val2
FROM (SELECT grouping_col, spreading_col, aggregation_col FROM source_table) AS D
PIVOT(SUM(aggregation_col) FOR spreading_col IN(val1, val2)) AS P;
```

### Unpivoting Template
```sql
-- Standard SQL:
SELECT * FROM (
    SELECT id_col, custid,
           CASE custid WHEN 'A' THEN A WHEN 'B' THEN B END AS qty
    FROM source_table
    CROSS JOIN (VALUES('A'),('B')) AS Custs(custid)
) AS D WHERE qty IS NOT NULL;

-- Native UNPIVOT:
SELECT id_col, custid, qty
FROM source_table
UNPIVOT(qty FOR custid IN(A, B)) AS U;
```

### MERGE Template
```sql
MERGE INTO target_table AS TGT
USING source_table AS SRC ON TGT.key = SRC.key
WHEN MATCHED [AND condition] THEN UPDATE SET ...
WHEN NOT MATCHED [BY TARGET] THEN INSERT (...) VALUES (...)
WHEN NOT MATCHED BY SOURCE THEN DELETE;  -- optional
-- Semicolon is REQUIRED after MERGE
```

### Error Handling Template
```sql
BEGIN TRY
    -- Your code here
END TRY
BEGIN CATCH
    IF ERROR_NUMBER() = <specific_error>
        -- Handle specific error
    ELSE
        THROW;  -- Re-raise for unknown errors
    -- Optionally log: ERROR_NUMBER(), ERROR_MESSAGE(), ERROR_LINE(), ERROR_PROCEDURE()
END CATCH;
```

---

*Notes compiled from: Microsoft SQL Server 2012 T-SQL Fundamentals by Itzik Ben-Gan (Microsoft Press / O'Reilly, 2012)*
