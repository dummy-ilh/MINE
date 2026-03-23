# SQL Cookbook — Study Guide
> Based on *SQL Cookbook* by Anthony Molinaro (O'Reilly, 2005)  
> Covers DB2, Oracle, PostgreSQL, MySQL, and SQL Server dialects  
> Core tables: **EMP** (14 rows) and **DEPT** (4 rows)

---

## Reference: Core Tables

```sql
-- EMP table (14 rows)
select * from emp;
```

| EMPNO | ENAME  | JOB       | MGR  | HIREDATE    | SAL  | COMM | DEPTNO |
|-------|--------|-----------|------|-------------|------|------|--------|
| 7369  | SMITH  | CLERK     | 7902 | 17-DEC-1980 | 800  |      | 20     |
| 7499  | ALLEN  | SALESMAN  | 7698 | 20-FEB-1981 | 1600 | 300  | 30     |
| 7521  | WARD   | SALESMAN  | 7698 | 22-FEB-1981 | 1250 | 500  | 30     |
| 7566  | JONES  | MANAGER   | 7839 | 02-APR-1981 | 2975 |      | 20     |
| 7654  | MARTIN | SALESMAN  | 7698 | 28-SEP-1981 | 1250 | 1400 | 30     |
| 7698  | BLAKE  | MANAGER   | 7839 | 01-MAY-1981 | 2850 |      | 30     |
| 7782  | CLARK  | MANAGER   | 7839 | 09-JUN-1981 | 2450 |      | 10     |
| 7788  | SCOTT  | ANALYST   | 7566 | 09-DEC-1982 | 3000 |      | 20     |
| 7839  | KING   | PRESIDENT |      | 17-NOV-1981 | 5000 |      | 10     |
| 7844  | TURNER | SALESMAN  | 7698 | 08-SEP-1981 | 1500 | 0    | 30     |
| 7876  | ADAMS  | CLERK     | 7788 | 12-JAN-1983 | 1100 |      | 20     |
| 7900  | JAMES  | CLERK     | 7698 | 03-DEC-1981 | 950  |      | 30     |
| 7902  | FORD   | ANALYST   | 7566 | 03-DEC-1981 | 3000 |      | 20     |
| 7934  | MILLER | CLERK     | 7782 | 23-JAN-1982 | 1300 |      | 10     |

```sql
-- DEPT table (4 rows)
select * from dept;
```

| DEPTNO | DNAME      | LOC      |
|--------|------------|----------|
| 10     | ACCOUNTING | NEW YORK |
| 20     | RESEARCH   | DALLAS   |
| 30     | SALES      | CHICAGO  |
| 40     | OPERATIONS | BOSTON   |

---

# Chapter 1 — Retrieving Records

---

## Recipe 1.1 — Retrieving All Rows and Columns

**Question:**  
You have a table and want to see all data in it. How do you retrieve every row and column?

**Answer:**
```sql
select * from emp;
```

**Sample Output:**

| EMPNO | ENAME | JOB     | MGR  | HIREDATE    | SAL  | COMM | DEPTNO |
|-------|-------|---------|------|-------------|------|------|--------|
| 7369  | SMITH | CLERK   | 7902 | 17-DEC-1980 | 800  |      | 20     |
| 7499  | ALLEN | SALESMAN| 7698 | 20-FEB-1981 | 1600 | 300  | 30     |
| ...   | ...   | ...     | ...  | ...         | ...  | ...  | ...    |

> **Note:** In application code, list columns explicitly instead of using `*` for clarity and safety.

---

## Recipe 1.2 — Retrieving a Subset of Rows

**Question:**  
You want to see only rows that satisfy a specific condition — for example, only employees in department 10.

**Answer:**
```sql
select *
from emp
where deptno = 10;
```

**Sample Output:**

| EMPNO | ENAME  | JOB       | MGR  | HIREDATE    | SAL  | COMM | DEPTNO |
|-------|--------|-----------|------|-------------|------|------|--------|
| 7782  | CLARK  | MANAGER   | 7839 | 09-JUN-1981 | 2450 |      | 10     |
| 7839  | KING   | PRESIDENT |      | 17-NOV-1981 | 5000 |      | 10     |
| 7934  | MILLER | CLERK     | 7782 | 23-JAN-1982 | 1300 |      | 10     |

---

## Recipe 1.3 — Rows Satisfying Multiple Conditions

**Question:**  
You want rows that satisfy multiple conditions using a combination of AND, OR, and parentheses. Return employees in dept 10, OR who earn a commission, OR who are in dept 20 and earn at most $2,000.

**Answer:**
```sql
select *
from emp
where deptno = 10
   or comm is not null
   or sal <= 2000 and deptno = 20;
```

**Sample Input (subset):**

| EMPNO | ENAME  | JOB      | SAL  | COMM | DEPTNO |
|-------|--------|----------|------|------|--------|
| 7369  | SMITH  | CLERK    | 800  |      | 20     |
| 7499  | ALLEN  | SALESMAN | 1600 | 300  | 30     |
| 7782  | CLARK  | MANAGER  | 2450 |      | 10     |

**Sample Output:**

| EMPNO | ENAME  | JOB       | SAL  | COMM | DEPTNO |
|-------|--------|-----------|------|------|--------|
| 7369  | SMITH  | CLERK     | 800  |      | 20     |
| 7499  | ALLEN  | SALESMAN  | 1600 | 300  | 30     |
| 7521  | WARD   | SALESMAN  | 1250 | 500  | 30     |
| 7654  | MARTIN | SALESMAN  | 1250 | 1400 | 30     |
| 7782  | CLARK  | MANAGER   | 2450 |      | 10     |
| 7839  | KING   | PRESIDENT | 5000 |      | 10     |
| 7844  | TURNER | SALESMAN  | 1500 | 0    | 30     |
| 7876  | ADAMS  | CLERK     | 1100 |      | 20     |
| 7934  | MILLER | CLERK     | 1300 |      | 10     |

---

## Recipe 1.4 — Retrieving a Subset of Columns

**Question:**  
You want to see only specific columns — name, department, and salary — rather than all columns.

**Answer:**
```sql
select ename, deptno, sal
from emp;
```

**Sample Output:**

| ENAME  | DEPTNO | SAL  |
|--------|--------|------|
| SMITH  | 20     | 800  |
| ALLEN  | 30     | 1600 |
| WARD   | 30     | 1250 |
| JONES  | 20     | 2975 |

---

## Recipe 1.5 — Providing Meaningful Column Names (Aliases)

**Question:**  
You want to rename column headers in your result set to something more descriptive. Rename `sal` to `salary` and `comm` to `commission`.

**Answer:**
```sql
select sal as salary, comm as commission
from emp;
```

**Sample Output:**

| salary | commission |
|--------|------------|
| 800    |            |
| 1600   | 300        |
| 1250   | 500        |
| 2975   |            |

---

## Recipe 1.6 — Referencing an Aliased Column in the WHERE Clause

**Question:**  
You defined an alias in the SELECT clause and want to use it in the WHERE clause. Why does this fail, and what is the workaround?

**Answer:**
```sql
-- This FAILS because aliases cannot be used in WHERE directly:
select sal as salary, comm as commission
from emp
where salary < 5000;  -- ERROR

-- Correct workaround: wrap in an inline view
select *
from (
  select sal as salary, comm as commission
  from emp
) x
where salary < 5000;
```

**Sample Output:**

| salary | commission |
|--------|------------|
| 800    |            |
| 1600   | 300        |
| 1250   | 500        |
| 2975   |            |

> **Key concept:** The WHERE clause is evaluated before the SELECT clause, so column aliases are not yet defined.

---

## Recipe 1.7 — Concatenating Column Values

**Question:**  
You want to merge values from multiple columns into one string. Return each employee's name and job as a single readable sentence.

**Answer:**
```sql
-- DB2, Oracle, PostgreSQL
select ename || ' WORKS AS A ' || job as msg
from emp
where deptno = 10;

-- MySQL
select concat(ename, ' WORKS AS A ', job) as msg
from emp
where deptno = 10;

-- SQL Server
select ename + ' WORKS AS A ' + job as msg
from emp
where deptno = 10;
```

**Sample Output:**

| msg                         |
|-----------------------------|
| CLARK WORKS AS A MANAGER    |
| KING WORKS AS A PRESIDENT   |
| MILLER WORKS AS A CLERK     |

---

## Recipe 1.8 — Conditional Logic in SELECT (CASE)

**Question:**  
You want to apply if-else logic in a SELECT statement. Return a status label based on each employee's salary: "UNDERPAID" if below 2000, "OVERPAID" if above 4000, otherwise "OK".

**Answer:**
```sql
select ename, sal,
  case when sal <= 2000 then 'UNDERPAID'
       when sal >= 4000 then 'OVERPAID'
       else 'OK'
  end as status
from emp;
```

**Sample Output:**

| ENAME  | SAL  | status    |
|--------|------|-----------|
| SMITH  | 800  | UNDERPAID |
| ALLEN  | 1600 | UNDERPAID |
| JONES  | 2975 | OK        |
| KING   | 5000 | OVERPAID  |
| SCOTT  | 3000 | OK        |

---

## Recipe 1.9 — Limiting the Number of Rows Returned

**Question:**  
You want to return only the first five rows from a query.

**Answer:**
```sql
-- DB2
select * from emp fetch first 5 rows only;

-- MySQL and PostgreSQL
select * from emp limit 5;

-- Oracle
select * from emp where rownum <= 5;

-- SQL Server
select top 5 * from emp;
```

**Sample Output:**

| EMPNO | ENAME  | JOB      | SAL  | DEPTNO |
|-------|--------|----------|------|--------|
| 7369  | SMITH  | CLERK    | 800  | 20     |
| 7499  | ALLEN  | SALESMAN | 1600 | 30     |
| 7521  | WARD   | SALESMAN | 1250 | 30     |
| 7566  | JONES  | MANAGER  | 2975 | 20     |
| 7654  | MARTIN | SALESMAN | 1250 | 30     |

---

## Recipe 1.10 — Returning n Random Records

**Question:**  
You want to return 5 random rows from EMP each time the query is executed.

**Answer:**
```sql
-- MySQL
select ename, job
from emp
order by rand() limit 5;

-- PostgreSQL
select ename, job
from emp
order by random() limit 5;

-- Oracle (using DBMS_RANDOM)
select * from (
  select ename, job
  from emp
  order by dbms_random.value()
) where rownum <= 5;

-- SQL Server
select top 5 ename, job
from emp
order by newid();
```

**Sample Output (varies per run):**

| ENAME  | JOB      |
|--------|----------|
| KING   | PRESIDENT|
| ADAMS  | CLERK    |
| BLAKE  | MANAGER  |
| TURNER | SALESMAN |
| FORD   | ANALYST  |

---

## Recipe 1.11 — Finding NULL Values

**Question:**  
You want to find all rows where a specific column is NULL — for example, employees who have no commission.

**Answer:**
```sql
select *
from emp
where comm is null;
```

**Sample Output:**

| EMPNO | ENAME | JOB      | SAL  | COMM | DEPTNO |
|-------|-------|----------|------|------|--------|
| 7369  | SMITH | CLERK    | 800  |      | 20     |
| 7566  | JONES | MANAGER  | 2975 |      | 20     |
| 7698  | BLAKE | MANAGER  | 2850 |      | 30     |

> **Key concept:** You cannot use `= NULL`; you must use `IS NULL`.

---

## Recipe 1.12 — Transforming NULLs into Real Values (COALESCE)

**Question:**  
You want to substitute a real value for any NULL returned in a query. Replace NULL commissions with 0.

**Answer:**
```sql
select ename, coalesce(comm, 0) as comm
from emp;
```

**Sample Output:**

| ENAME  | comm |
|--------|------|
| SMITH  | 0    |
| ALLEN  | 300  |
| WARD   | 500  |
| JONES  | 0    |
| MARTIN | 1400 |

---

## Recipe 1.13 — Searching for Patterns (LIKE)

**Question:**  
You want to search for rows that match a partial string. Return all employees whose name begins with "S" or ends in "ES".

**Answer:**
```sql
select ename, job
from emp
where ename like 'S%'
   or ename like '%ES';
```

**Sample Output:**

| ENAME | JOB     |
|-------|---------|
| SMITH | CLERK   |
| JONES | MANAGER |
| SCOTT | ANALYST |
| JAMES | CLERK   |

---

# Chapter 2 — Sorting Query Results

---

## Recipe 2.1 — Returning Results in a Specified Order

**Question:**  
You want to display employees' names, jobs, and salaries sorted by salary in ascending order.

**Answer:**
```sql
select ename, job, sal
from emp
order by sal asc;
```

**Sample Output:**

| ENAME  | JOB       | SAL  |
|--------|-----------|------|
| SMITH  | CLERK     | 800  |
| JAMES  | CLERK     | 950  |
| ADAMS  | CLERK     | 1100 |
| WARD   | SALESMAN  | 1250 |
| MARTIN | SALESMAN  | 1250 |

---

## Recipe 2.2 — Sorting by Multiple Fields

**Question:**  
You want to sort employees first by department number (ascending), then by salary (descending) within each department.

**Answer:**
```sql
select empno, deptno, sal, ename, job
from emp
order by deptno asc, sal desc;
```

**Sample Output:**

| EMPNO | DEPTNO | SAL  | ENAME  | JOB       |
|-------|--------|------|--------|-----------|
| 7839  | 10     | 5000 | KING   | PRESIDENT |
| 7782  | 10     | 2450 | CLARK  | MANAGER   |
| 7934  | 10     | 1300 | MILLER | CLERK     |
| 7788  | 20     | 3000 | SCOTT  | ANALYST   |
| 7902  | 20     | 3000 | FORD   | ANALYST   |

---

## Recipe 2.3 — Sorting by Substrings

**Question:**  
You want to sort results based on a specific part of a string — for example, the last two characters of each employee's name.

**Answer:**
```sql
-- DB2, MySQL, Oracle, PostgreSQL
select ename
from emp
order by substr(ename, length(ename)-1);

-- SQL Server
select ename
from emp
order by substring(ename, len(ename)-1, 2);
```

**Sample Output:**

| ENAME  |
|--------|
| ALLEN  |
| TURNER |
| MILLER |
| JONES  |
| JAMES  |

---

## Recipe 2.4 — Sorting Mixed Alphanumeric Data

**Question:**  
You have a column that contains both letters and numbers mixed together (e.g., `CLARK 7782`) and want to sort by either the numeric or character portion only.

**Answer:**
```sql
-- Sort by name only (strip numbers):
select data
from V
order by replace(
  translate(data, '0123456789', '##########'),
  '#', '');

-- Sort by number only (strip letters):
select data
from V
order by cast(
  replace(
    translate(data, repeat('#', length(data)),
      replace(translate(data, '##########', '0123456789'), '#', '')),
    '#', '') as integer);
```

**Sample Input (view V):**

| data               |
|--------------------|
| CLARK 7782         |
| KING 7839          |
| MILLER 7934        |
| SMITH 7369         |

**Sample Output (sorted by number):**

| data        |
|-------------|
| SMITH 7369  |
| CLARK 7782  |
| KING 7839   |
| MILLER 7934 |

---

## Recipe 2.5 — Dealing with NULLs when Sorting

**Question:**  
You want to sort results by commission, but NULL commissions should appear last regardless of sort direction.

**Answer:**
```sql
-- Non-NULL commissions first, then NULLs last (ascending):
select ename, sal, comm
from (
  select ename, sal, comm,
    case when comm is null then 0 else 1 end as is_null
  from emp
) x
order by is_null desc, comm;
```

**Sample Output:**

| ENAME  | SAL  | COMM |
|--------|------|------|
| TURNER | 1500 | 0    |
| ALLEN  | 1600 | 300  |
| WARD   | 1250 | 500  |
| MARTIN | 1250 | 1400 |
| SMITH  | 800  |      |
| JONES  | 2975 |      |

---

## Recipe 2.6 — Sorting on a Data Dependent Key

**Question:**  
You want the sort order to depend on the value of a column — for example, sort by `comm` if the employee is a salesman, otherwise sort by `ename`.

**Answer:**
```sql
select ename, sal, job, comm
from emp
order by
  case when job = 'SALESMAN' then comm
       else ename
  end;
```

**Sample Output:**

| ENAME  | SAL  | JOB       | COMM |
|--------|------|-----------|------|
| TURNER | 1500 | SALESMAN  | 0    |
| ALLEN  | 1600 | SALESMAN  | 300  |
| WARD   | 1250 | SALESMAN  | 500  |
| MARTIN | 1250 | SALESMAN  | 1400 |
| ADAMS  | 1100 | CLERK     |      |
| BLAKE  | 2850 | MANAGER   |      |

---

# Chapter 3 — Working with Multiple Tables

---

## Recipe 3.1 — Stacking One Rowset atop Another (UNION ALL)

**Question:**  
You want to return the name and department number of employees in department 10 and the name and department number of each department from the DEPT table, all in one result set.

**Answer:**
```sql
select ename as ename_and_dname, deptno
from emp
where deptno = 10
union all
select '----------', null
from t1
union all
select dname, deptno
from dept;
```

**Sample Output:**

| ename_and_dname | deptno |
|-----------------|--------|
| CLARK           | 10     |
| KING            | 10     |
| MILLER          | 10     |
| ----------      |        |
| ACCOUNTING      | 10     |
| RESEARCH        | 20     |
| SALES           | 30     |
| OPERATIONS      | 40     |

---

## Recipe 3.2 — Combining Related Rows (JOIN)

**Question:**  
You want to display the name and location of each employee in department 10. The name is in EMP and the location is in DEPT.

**Answer:**
```sql
select e.ename, d.loc
from emp e, dept d
where e.deptno = d.deptno
  and e.deptno = 10;
```

**Sample Output:**

| ENAME  | LOC      |
|--------|----------|
| CLARK  | NEW YORK |
| KING   | NEW YORK |
| MILLER | NEW YORK |

---

## Recipe 3.3 — Finding Rows in Common Between Two Tables (INTERSECT)

**Question:**  
You have a view V containing only CLERK-type employees. You want to find which employees in EMP match the rows in V based on name, job, and salary.

**Answer:**
```sql
-- DB2, Oracle, PostgreSQL (using INTERSECT)
select empno, ename, job, sal, deptno
from emp
where (ename, job, sal) in (
  select ename, job, sal from emp
  intersect
  select ename, job, sal from V
);

-- MySQL, SQL Server (using a JOIN)
select e.empno, e.ename, e.job, e.sal, e.deptno
from emp e, V
where e.ename = v.ename
  and e.job   = v.job
  and e.sal   = v.sal;
```

**Sample Input (view V — CLERK employees only):**

| ENAME  | JOB   | SAL  |
|--------|-------|------|
| SMITH  | CLERK | 800  |
| ADAMS  | CLERK | 1100 |
| JAMES  | CLERK | 950  |
| MILLER | CLERK | 1300 |

**Sample Output:**

| EMPNO | ENAME  | JOB   | SAL  | DEPTNO |
|-------|--------|-------|------|--------|
| 7369  | SMITH  | CLERK | 800  | 20     |
| 7876  | ADAMS  | CLERK | 1100 | 20     |
| 7900  | JAMES  | CLERK | 950  | 30     |
| 7934  | MILLER | CLERK | 1300 | 10     |

---

## Recipe 3.4 — Values in One Table Not Existing in Another (EXCEPT / MINUS)

**Question:**  
You want to find departments that exist in DEPT but have no employees in EMP. DEPT has department 40 (OPERATIONS) which has no employees.

**Answer:**
```sql
-- DB2 and PostgreSQL
select deptno from dept
except
select deptno from emp;

-- Oracle
select deptno from dept
minus
select deptno from emp;

-- MySQL and SQL Server
select deptno from dept
where deptno not in (select deptno from emp);
```

**Sample Output:**

| DEPTNO |
|--------|
| 40     |

> **Warning:** `NOT IN` with NULLs can produce unexpected results. Use `NOT EXISTS` as a safer alternative.

---

## Recipe 3.5 — Rows Without a Match in Another Table (Anti-Join)

**Question:**  
You want to find departments that have no employees, returning the full department row including name and location.

**Answer:**
```sql
-- DB2, MySQL, PostgreSQL, SQL Server
select d.*
from dept d
left outer join emp e on (d.deptno = e.deptno)
where e.deptno is null;
```

**Sample Output:**

| DEPTNO | DNAME      | LOC    |
|--------|------------|--------|
| 40     | OPERATIONS | BOSTON |

---

## Recipe 3.6 — Adding Joins Without Interfering with Other Joins

**Question:**  
You want to retrieve employee name, location, and (if any) the date of any bonus received — but outer joining to the BONUS table should not affect the rows returned for employees without bonuses.

**Answer:**
```sql
select e.ename, d.loc, eb.received
from emp e
join dept d on (e.deptno = d.deptno)
left join emp_bonus eb on (e.empno = eb.empno)
order by 2;
```

**Sample Output:**

| ENAME  | LOC      | RECEIVED    |
|--------|----------|-------------|
| CLARK  | NEW YORK |             |
| KING   | NEW YORK | 17-MAR-2005 |
| MILLER | NEW YORK |             |
| SMITH  | DALLAS   |             |

---

## Recipe 3.7 — Determining Whether Two Tables Have the Same Data

**Question:**  
You have a view V and the EMP table. You want to confirm whether they contain exactly the same rows.

**Answer:**
```sql
-- Find rows in EMP not in V, plus rows in V not in EMP:
(
  select empno, ename, job, mgr, hiredate, sal, comm, deptno,
         count(*) as cnt
  from emp
  group by empno, ename, job, mgr, hiredate, sal, comm, deptno
  except
  select empno, ename, job, mgr, hiredate, sal, comm, deptno,
         count(*) as cnt
  from V
  group by empno, ename, job, mgr, hiredate, sal, comm, deptno
)
union all
(
  select empno, ename, job, mgr, hiredate, sal, comm, deptno,
         count(*) as cnt
  from V
  group by empno, ename, job, mgr, hiredate, sal, comm, deptno
  except
  select empno, ename, job, mgr, hiredate, sal, comm, deptno,
         count(*) as cnt
  from emp
  group by empno, ename, job, mgr, hiredate, sal, comm, deptno
);
```

**Sample Output (if tables match):** *(no rows returned)*

**Sample Output (if tables differ):**

| EMPNO | ENAME | JOB   | SAL  | DEPTNO | cnt |
|-------|-------|-------|------|--------|-----|
| 7369  | SMITH | CLERK | 800  | 20     | 1   |

---

## Recipe 3.8 — Identifying and Avoiding Cartesian Products

**Question:**  
You want to see employees in department 10 with their department's location, but accidentally produce a Cartesian product with multiple locations. How do you fix it?

**Answer:**
```sql
-- Cartesian product (WRONG — returns 12 rows):
select e.ename, d.loc
from emp e, dept d
where e.deptno = 10;

-- Correct (join on DEPTNO — returns 3 rows):
select e.ename, d.loc
from emp e, dept d
where e.deptno = d.deptno
  and e.deptno = 10;
```

**Sample Output (correct):**

| ENAME  | LOC      |
|--------|----------|
| CLARK  | NEW YORK |
| KING   | NEW YORK |
| MILLER | NEW YORK |

---

## Recipe 3.9 — Performing Joins when Using Aggregates

**Question:**  
You want to find the sum of salaries for employees in department 10, but also need to join to the EMP_BONUS table. The join can inflate aggregate results if not handled carefully.

**Answer:**
```sql
-- Use DISTINCT inside SUM to avoid double-counting:
select deptno,
       sum(distinct sal) as total_sal,
       sum(bonus) as total_bonus
from (
  select e.empno, e.ename, e.sal, e.deptno,
         e.sal * case when eb.type = 1 then 0.1
                      when eb.type = 2 then 0.2
                      else 0.3 end as bonus
  from emp e, emp_bonus eb
  where e.empno = eb.empno
    and e.deptno = 10
) x
group by deptno;
```

**Sample Output:**

| DEPTNO | total_sal | total_bonus |
|--------|-----------|-------------|
| 10     | 8750      | 2135        |

---

## Recipe 3.12 — Using NULLs in Operations and Comparisons

**Question:**  
You want to find all employees whose commission is less than WARD's commission of 500, including those with no commission at all (NULL).

**Answer:**
```sql
select ename, comm
from emp
where coalesce(comm, 0) < (
  select comm from emp where ename = 'WARD'
);
```

**Sample Output:**

| ENAME  | COMM |
|--------|------|
| SMITH  |      |
| JONES  |      |
| BLAKE  |      |
| CLARK  |      |
| SCOTT  |      |
| KING   |      |
| TURNER | 0    |
| ADAMS  |      |
| JAMES  |      |
| FORD   |      |
| MILLER |      |
| ALLEN  | 300  |

---

# Chapter 4 — Inserting, Updating, Deleting

---

## Recipe 4.1 — Inserting a New Record

**Question:**  
You want to add a new row to the DEPT table.

**Answer:**
```sql
insert into dept (deptno, dname, loc)
values (50, 'PROGRAMMING', 'BALTIMORE');
```

**Result:**
```sql
select * from dept;
```

| DEPTNO | DNAME       | LOC       |
|--------|-------------|-----------|
| 10     | ACCOUNTING  | NEW YORK  |
| 20     | RESEARCH    | DALLAS    |
| 30     | SALES       | CHICAGO   |
| 40     | OPERATIONS  | BOSTON    |
| 50     | PROGRAMMING | BALTIMORE |

---

## Recipe 4.4 — Copying Rows from One Table into Another

**Question:**  
You want to copy rows from one table into another existing table. For example, copy all rows from DEPT into DEPT_EAST where the location is 'NEW YORK' or 'BOSTON'.

**Answer:**
```sql
insert into dept_east (deptno, dname, loc)
select deptno, dname, loc
from dept
where loc in ('NEW YORK', 'BOSTON');
```

**Sample Output (DEPT_EAST after insert):**

| DEPTNO | DNAME      | LOC      |
|--------|------------|----------|
| 10     | ACCOUNTING | NEW YORK |
| 40     | OPERATIONS | BOSTON   |

---

## Recipe 4.8 — Modifying Records in a Table (UPDATE)

**Question:**  
You want to increase the salaries of employees in department 20 by 10%.

**Answer:**
```sql
update emp
set sal = sal * 1.10
where deptno = 20;
```

**Sample Output (DEPT 20 employees after update):**

| ENAME | Old SAL | New SAL |
|-------|---------|---------|
| SMITH | 800     | 880     |
| JONES | 2975    | 3272    |
| SCOTT | 3000    | 3300    |
| ADAMS | 1100    | 1210    |
| FORD  | 3000    | 3300    |

---

## Recipe 4.9 — Updating when Corresponding Rows Exist

**Question:**  
You want to give a 10% raise to employees who appear in the EMP_BONUS table.

**Answer:**
```sql
update emp
set sal = sal * 1.10
where empno in (select empno from emp_bonus);
```

---

## Recipe 4.12 — Deleting All Records from a Table

**Question:**  
You want to delete all rows from a table.

**Answer:**
```sql
delete from emp;
```

> **Note:** `TRUNCATE TABLE emp` is faster for large tables as it is not logged row-by-row, but has fewer rollback options.

---

## Recipe 4.16 — Deleting Duplicate Records

**Question:**  
You have a table with duplicate names and want to keep only the row with the lowest EMPNO for each name.

**Answer:**
```sql
delete from dupes
where id not in (
  select min(id)
  from dupes
  group by name
);
```

**Sample Input:**

| ID | NAME  |
|----|-------|
| 1  | CLARK |
| 2  | CLARK |
| 3  | KING  |
| 4  | MILLER|
| 5  | MILLER|

**Sample Output (after delete):**

| ID | NAME   |
|----|--------|
| 1  | CLARK  |
| 3  | KING   |
| 4  | MILLER |

---

# Chapter 5 — Metadata Queries

---

## Recipe 5.1 — Listing Tables in a Schema

**Question:**  
You want to see the names of all tables in the current schema/database.

**Answer:**
```sql
-- Oracle
select table_name
from all_tables
where owner = 'YOUR_SCHEMA';

-- MySQL
show tables;

-- DB2
select tabname
from syscat.tables
where tabschema = 'YOUR_SCHEMA';

-- PostgreSQL
select tablename
from pg_tables
where schemaname = 'public';

-- SQL Server
select name from sys.tables;
```

---

## Recipe 5.2 — Listing a Table's Columns

**Question:**  
You want to see the column names and data types of the EMP table.

**Answer:**
```sql
-- Oracle
select column_name, data_type, nullable
from all_tab_columns
where table_name = 'EMP';

-- MySQL
describe emp;

-- PostgreSQL
select column_name, data_type
from information_schema.columns
where table_name = 'emp';
```

**Sample Output:**

| COLUMN_NAME | DATA_TYPE | NULLABLE |
|-------------|-----------|----------|
| EMPNO       | NUMBER    | N        |
| ENAME       | VARCHAR2  | Y        |
| JOB         | VARCHAR2  | Y        |
| SAL         | NUMBER    | Y        |
| COMM        | NUMBER    | Y        |
| DEPTNO      | NUMBER    | Y        |

---

## Recipe 5.6 — Using SQL to Generate SQL

**Question:**  
You want to write a query that generates other SQL statements — for example, generate `COUNT(*)` statements for every table in your schema.

**Answer:**
```sql
-- Oracle
select 'select count(*) from '|| table_name || ';' as cnt_sql
from all_tables;
```

**Sample Output:**

| cnt_sql                       |
|-------------------------------|
| select count(*) from EMP;     |
| select count(*) from DEPT;    |
| select count(*) from BONUS;   |

---

# Chapter 6 — Working with Strings

---

## Recipe 6.1 — Walking a String

**Question:**  
You want to traverse a string and return each character as a separate row. For example, break 'KING' into individual characters.

**Answer:**
```sql
-- DB2
select substr(e.ename, iter.pos, 1) as C
from emp e,
     (select id as pos from t10) iter
where e.ename = 'KING'
  and iter.pos <= length(e.ename);

-- Oracle
select substr(e.ename, iter.pos, 1) as C
from (select ename from emp where ename = 'KING') e,
     (select rownum as pos from emp) iter
where iter.pos <= length(e.ename);
```

**Sample Output:**

| C |
|---|
| K |
| I |
| N |
| G |

---

## Recipe 6.3 — Counting Occurrences of a Character

**Question:**  
You want to count how many times a specific character or substring appears in a string. Count the number of commas in `'10,CLARK,MANAGER'`.

**Answer:**
```sql
select (length('10,CLARK,MANAGER') -
        length(replace('10,CLARK,MANAGER', ',', ''))) / length(',')
  as cnt
from t1;
```

**Sample Output:**

| cnt |
|-----|
| 2   |

---

## Recipe 6.7 — Extracting Initials from a Name

**Question:**  
Given the name `'Stewie Griffin'`, return the initials `'S.G.'`.

**Answer:**
```sql
-- MySQL
select case
  when cnt = 2 then
    trim(trailing '.' from
      concat_ws('.',
        substr(substring_index(name,' ',1),1,1),
        substr(name, length(substring_index(name,' ',1))+2,1),
        '.'))
  else
    concat(substr(name,1,1),'.')
  end as initials
from (
  select name,
    length(name) - length(replace(name,' ','')) as cnt
  from (select 'Stewie Griffin' as name from t1) t
) t;
```

**Sample Output:**

| initials |
|----------|
| S.G.     |

---

## Recipe 6.10 — Creating a Delimited List from Table Rows

**Question:**  
You want to return each department's employees as a single comma-delimited string per department.

**Answer:**
```sql
-- MySQL (simplest)
select deptno,
       group_concat(ename order by empno separator ',') as emps
from emp
group by deptno;
```

**Sample Input:**

| DEPTNO | ENAME  |
|--------|--------|
| 10     | CLARK  |
| 10     | KING   |
| 10     | MILLER |
| 20     | SMITH  |

**Sample Output:**

| DEPTNO | emps                              |
|--------|-----------------------------------|
| 10     | CLARK,KING,MILLER                 |
| 20     | SMITH,JONES,SCOTT,ADAMS,FORD      |
| 30     | ALLEN,WARD,MARTIN,BLAKE,TURNER,JAMES |

---

# Chapter 7 — Working with Numbers

---

## Recipe 7.1 — Computing an Average

**Question:**  
You want to find the average salary for all employees and the average salary per department.

**Answer:**
```sql
-- Overall average
select avg(sal) as avg_sal
from emp;

-- Average per department
select deptno, avg(sal) as avg_sal
from emp
group by deptno;
```

**Sample Output:**

| DEPTNO | avg_sal |
|--------|---------|
| 10     | 2916.67 |
| 20     | 2175.00 |
| 30     | 1566.67 |

---

## Recipe 7.2 — Finding Min/Max Value in a Column

**Question:**  
You want to find the highest and lowest salaries in EMP.

**Answer:**
```sql
select min(sal) as min_sal, max(sal) as max_sal
from emp;
```

**Sample Output:**

| min_sal | max_sal |
|---------|---------|
| 800     | 5000    |

---

## Recipe 7.6 — Generating a Running Total

**Question:**  
You want to compute a cumulative sum of salaries ordered by employee number.

**Answer:**
```sql
-- DB2, Oracle, SQL Server (window function)
select empno, ename, sal,
       sum(sal) over (order by sal, empno) as running_total
from emp;

-- MySQL / Portable alternative (correlated subquery)
select e.empno, e.ename, e.sal,
       (select sum(sal) from emp d where d.empno <= e.empno) as running_total
from emp e
order by 3, 1;
```

**Sample Output:**

| EMPNO | ENAME  | SAL  | running_total |
|-------|--------|------|---------------|
| 7369  | SMITH  | 800  | 800           |
| 7900  | JAMES  | 950  | 1750          |
| 7876  | ADAMS  | 1100 | 2850          |
| 7521  | WARD   | 1250 | 4100          |
| 7654  | MARTIN | 1250 | 5350          |

---

## Recipe 7.9 — Calculating a Mode

**Question:**  
You want to find the salary that occurs most frequently (the mode) in EMP.

**Answer:**
```sql
select sal
from emp
group by sal
having count(*) >= all (
  select count(*) from emp group by sal
);
```

**Sample Output:**

| sal  |
|------|
| 1250 |
| 3000 |

---

## Recipe 7.10 — Calculating a Median

**Question:**  
You want to find the median salary from EMP.

**Answer:**
```sql
-- Oracle (simplest)
select median(sal)
from emp;

-- Portable (using percentile logic)
select avg(sal)
from (
  select sal,
         count(*) over () cnt,
         row_number() over (order by sal) rn
  from emp
) x
where rn in (floor((cnt+1)/2.0), ceil((cnt+1)/2.0));
```

**Sample Output:**

| median |
|--------|
| 1575   |

---

## Recipe 7.11 — Determining the Percentage of a Total

**Question:**  
You want to find what percentage of total salary each department contributes.

**Answer:**
```sql
select deptno,
       sum(sal) as dept_total,
       sum(sal) / (select sum(sal) from emp) * 100 as pct_of_total
from emp
group by deptno
order by deptno;
```

**Sample Output:**

| DEPTNO | dept_total | pct_of_total |
|--------|------------|--------------|
| 10     | 8750       | 29.68        |
| 20     | 10875      | 36.87        |
| 30     | 9400       | 31.87        |

---

# Chapter 8 — Date Arithmetic

---

## Recipe 8.1 — Adding and Subtracting Days, Months, and Years

**Question:**  
You want to add and subtract intervals of days, months, and years from a date. Using HIREDATE for CLARK (09-JUN-1981), compute ±5 days, ±5 months, and ±5 years.

**Answer:**
```sql
-- Oracle
select hiredate,
       hiredate - 5             as hd_minus_5d,
       hiredate + 5             as hd_plus_5d,
       add_months(hiredate,-5)  as hd_minus_5m,
       add_months(hiredate, 5)  as hd_plus_5m,
       add_months(hiredate,-60) as hd_minus_5y,
       add_months(hiredate, 60) as hd_plus_5y
from emp
where ename = 'CLARK';
```

**Sample Output:**

| HIREDATE    | hd_minus_5d | hd_plus_5d  | hd_minus_5m | hd_plus_5m  |
|-------------|-------------|-------------|-------------|-------------|
| 09-JUN-1981 | 04-JUN-1981 | 14-JUN-1981 | 09-JAN-1981 | 09-NOV-1981 |

---

## Recipe 8.2 — Number of Days Between Two Dates

**Question:**  
You want to find the difference in days between ALLEN's and WARD's hire dates.

**Answer:**
```sql
-- Oracle
select ward_hd - allen_hd as diff
from (
  select max(case when ename='WARD'  then hiredate end) as ward_hd,
         max(case when ename='ALLEN' then hiredate end) as allen_hd
  from emp
) x;
```

**Sample Output:**

| diff |
|------|
| 2    |

---

## Recipe 8.3 — Number of Business Days Between Two Dates

**Question:**  
You want to count only working days (Monday–Friday) between two dates, such as BLAKE's and JONES's hire dates.

**Answer:**
```sql
-- Oracle (using pivot table T500)
select sum(
  case when to_char(jones_hd + (rownum-1), 'DY') in ('SAT','SUN')
       then 0 else 1 end
) as biz_days
from t500, (
  select max(case when ename='BLAKE' then hiredate end) as blake_hd,
         max(case when ename='JONES' then hiredate end) as jones_hd
  from emp
)
where rownum <= jones_hd - blake_hd + 1;
```

**Sample Output:**

| biz_days |
|----------|
| 21       |

---

# Chapter 9 — Date Manipulation

---

## Recipe 9.1 — Determining if a Year Is a Leap Year

**Question:**  
You want to find whether the current year is a leap year by examining the last day of February.

**Answer:**
```sql
-- Oracle
select case
  when to_char(
    last_day(to_date(extract(year from sysdate) || '02', 'YYYYMM')),
    'DD') = '29'
  then 'Leap Year'
  else 'Not a Leap Year'
  end as is_leap
from dual;
```

**Sample Output (for a leap year):**

| is_leap   |
|-----------|
| Leap Year |

---

## Recipe 9.3 — Extracting Units of Time from a Date

**Question:**  
You want to extract the hour, minute, second, day, month, and year from the current timestamp as numbers.

**Answer:**
```sql
-- DB2
select hour(current_timestamp)   as hr,
       minute(current_timestamp) as min,
       second(current_timestamp) as sec,
       day(current_timestamp)    as dy,
       month(current_timestamp)  as mth,
       year(current_timestamp)   as yr
from t1;

-- Oracle
select to_number(to_char(sysdate,'HH24')) as hr,
       to_number(to_char(sysdate,'MI'))   as min,
       to_number(to_char(sysdate,'SS'))   as sec,
       to_number(to_char(sysdate,'DD'))   as dy,
       to_number(to_char(sysdate,'MM'))   as mth,
       to_number(to_char(sysdate,'YYYY')) as yr
from dual;
```

**Sample Output:**

| hr | min | sec | dy | mth | yr   |
|----|-----|-----|----|-----|------|
| 14 | 32  | 17  | 15 | 3   | 2005 |

---

## Recipe 9.4 — First and Last Day of a Month

**Question:**  
You want to determine the first and last day of the current month.

**Answer:**
```sql
-- DB2
select (current_date - day(current_date) day + 1 day) as firstday,
       (current_date + 1 month - day(current_date) day) as lastday
from t1;

-- Oracle
select trunc(sysdate, 'MM')  as firstday,
       last_day(sysdate)     as lastday
from dual;

-- MySQL
select date_add(current_date,
         interval -day(current_date)+1 day) as firstday,
       last_day(current_date) as lastday;
```

**Sample Output (for March 2005):**

| firstday    | lastday     |
|-------------|-------------|
| 01-MAR-2005 | 31-MAR-2005 |

---

## Recipe 9.7 — Creating a Calendar

**Question:**  
You want to generate a formatted calendar for the current month with weeks as rows and days Mon–Sun as columns.

**Answer:**
```sql
-- Oracle (abbreviated)
select max(case when to_char(dy,'DY')='MON' then to_char(dy,'DD') end) as Mo,
       max(case when to_char(dy,'DY')='TUE' then to_char(dy,'DD') end) as Tu,
       max(case when to_char(dy,'DY')='WED' then to_char(dy,'DD') end) as We,
       max(case when to_char(dy,'DY')='THU' then to_char(dy,'DD') end) as Th,
       max(case when to_char(dy,'DY')='FRI' then to_char(dy,'DD') end) as Fr,
       max(case when to_char(dy,'DY')='SAT' then to_char(dy,'DD') end) as Sa,
       max(case when to_char(dy,'DY')='SUN' then to_char(dy,'DD') end) as Su
from (
  select trunc(sysdate,'MM')+level-1 as dy,
         to_char(trunc(sysdate,'MM')+level-1,'IW') as wk
  from dual connect by level <= 31
  where trunc(sysdate,'MM')+level-1 <= last_day(sysdate)
)
group by wk
order by wk;
```

**Sample Output (June 2005):**

| Mo | Tu | We | Th | Fr | Sa | Su |
|----|----|----|----|----|----|----|
|    |    | 01 | 02 | 03 | 04 | 05 |
| 06 | 07 | 08 | 09 | 10 | 11 | 12 |
| 13 | 14 | 15 | 16 | 17 | 18 | 19 |
| 20 | 21 | 22 | 23 | 24 | 25 | 26 |
| 27 | 28 | 29 | 30 |    |    |    |

---

# Chapter 10 — Working with Ranges

---

## Recipe 10.1 — Locating a Range of Consecutive Values

**Question:**  
Given a list of projects with start and end dates, you want to identify which projects form a consecutive chain (where one project's end date is the next project's start date).

**Answer:**
```sql
select a.proj_id, a.proj_start, a.proj_end
from V a, V b
where a.proj_end = b.proj_start;
```

**Sample Input (view V):**

| PROJ_ID | PROJ_START  | PROJ_END    |
|---------|-------------|-------------|
| 1       | 01-JAN-2005 | 02-JAN-2005 |
| 2       | 02-JAN-2005 | 03-JAN-2005 |
| 5       | 06-JAN-2005 | 07-JAN-2005 |

**Sample Output:**

| PROJ_ID | PROJ_START  | PROJ_END    |
|---------|-------------|-------------|
| 1       | 01-JAN-2005 | 02-JAN-2005 |
| 2       | 02-JAN-2005 | 03-JAN-2005 |

---

## Recipe 10.2 — Differences Between Rows in the Same Group

**Question:**  
For each employee, show the difference in salary compared to the next employee hired in the same department.

**Answer:**
```sql
-- DB2, Oracle (window function)
select deptno, ename, sal, hiredate,
       sal - lead(sal) over (partition by deptno order by hiredate) as diff
from emp;
```

**Sample Output:**

| DEPTNO | ENAME  | SAL  | HIREDATE    | diff  |
|--------|--------|------|-------------|-------|
| 10     | CLARK  | 2450 | 09-JUN-1981 | -2550 |
| 10     | KING   | 5000 | 17-NOV-1981 | 3700  |
| 10     | MILLER | 1300 | 23-JAN-1982 |       |

---

## Recipe 10.5 — Generating Consecutive Numeric Values

**Question:**  
You want to generate a series of sequential integers from 1 to 10 without manually listing them.

**Answer:**
```sql
-- PostgreSQL (cleanest)
select generate_series(1, 10) as id;

-- Oracle (connect by)
select rownum as id
from dual
connect by rownum <= 10;

-- DB2 / SQL Server (recursive CTE)
with x (id) as (
  select 1
  union all
  select id + 1 from x where id < 10
)
select * from x;
```

**Sample Output:**

| id |
|----|
| 1  |
| 2  |
| 3  |
| ...| 
| 10 |

---

# Chapter 11 — Advanced Searching

---

## Recipe 11.1 — Paginating Through a Result Set

**Question:**  
You want to return rows 5 through 10 from EMP, ordered by EMPNO — useful for pagination in applications.

**Answer:**
```sql
-- DB2
select empno, ename, sal
from emp
order by empno
offset 4 rows fetch next 6 rows only;

-- Oracle (using ROWNUM in a subquery)
select *
from (
  select rownum as rn, empno, ename, sal
  from (select empno, ename, sal from emp order by empno)
)
where rn between 5 and 10;

-- MySQL / PostgreSQL
select empno, ename, sal
from emp
order by empno
limit 6 offset 4;
```

**Sample Output:**

| EMPNO | ENAME  | SAL  |
|-------|--------|------|
| 7654  | MARTIN | 1250 |
| 7698  | BLAKE  | 2850 |
| 7782  | CLARK  | 2450 |
| 7788  | SCOTT  | 3000 |
| 7839  | KING   | 5000 |
| 7844  | TURNER | 1500 |

---

## Recipe 11.5 — Selecting the Top n Records

**Question:**  
You want to return the top 5 earners from EMP (allowing for ties with DENSE_RANK).

**Answer:**
```sql
-- DB2, Oracle, SQL Server
select ename, sal
from (
  select ename, sal,
         dense_rank() over (order by sal desc) dr
  from emp
) x
where dr <= 5;

-- MySQL / PostgreSQL
select ename, sal
from emp
order by sal desc
limit 5;
```

**Sample Output:**

| ENAME | SAL  |
|-------|------|
| KING  | 5000 |
| FORD  | 3000 |
| SCOTT | 3000 |
| JONES | 2975 |
| BLAKE | 2850 |

---

## Recipe 11.6 — Highest and Lowest Values

**Question:**  
You want to find the employees with the highest and lowest salaries in a single query.

**Answer:**
```sql
-- DB2, Oracle, SQL Server (window functions)
select ename
from (
  select ename, sal,
         min(sal) over () as min_sal,
         max(sal) over () as max_sal
  from emp
) x
where sal in (min_sal, max_sal);
```

**Sample Output:**

| ENAME |
|-------|
| SMITH |
| KING  |

---

## Recipe 11.9 — Ranking Results

**Question:**  
You want to rank employees by salary, allowing for ties (where two employees with the same salary share a rank).

**Answer:**
```sql
-- DB2, Oracle, SQL Server
select dense_rank() over (order by sal) as rnk, sal
from emp;
```

**Sample Output:**

| rnk | sal  |
|-----|------|
| 1   | 800  |
| 2   | 950  |
| 3   | 1100 |
| 4   | 1250 |
| 4   | 1250 |
| 5   | 1300 |

---

## Recipe 11.11 — Finding Knight Values

**Question:**  
You want to return each employee's data along with the salary of the last employee hired in their department — a "knight value" is the most recent non-NULL value in an ordered sequence.

**Answer:**
```sql
-- DB2, Oracle (window function)
select deptno, ename, sal, hiredate,
       max(sal) keep (dense_rank last order by hiredate)
         over (partition by deptno) as latest_sal
from emp
order by 1, 4 desc;
```

**Sample Output:**

| DEPTNO | ENAME  | SAL  | HIREDATE    | latest_sal |
|--------|--------|------|-------------|------------|
| 10     | MILLER | 1300 | 23-JAN-1982 | 1300       |
| 10     | KING   | 5000 | 17-NOV-1981 | 1300       |
| 10     | CLARK  | 2450 | 09-JUN-1981 | 1300       |

---

# Chapter 12 — Reporting and Warehousing

---

## Recipe 12.1 — Pivoting a Result Set into One Row

**Question:**  
You want to convert the count of employees per department (multiple rows) into a single row with one column per department.

**Answer:**
```sql
select sum(case when deptno = 10 then 1 else 0 end) as deptno_10,
       sum(case when deptno = 20 then 1 else 0 end) as deptno_20,
       sum(case when deptno = 30 then 1 else 0 end) as deptno_30
from emp;
```

**Sample Input:**

| DEPTNO | CNT |
|--------|-----|
| 10     | 3   |
| 20     | 5   |
| 30     | 6   |

**Sample Output:**

| deptno_10 | deptno_20 | deptno_30 |
|-----------|-----------|-----------|
| 3         | 5         | 6         |

---

## Recipe 12.3 — Reverse Pivoting a Result Set

**Question:**  
You have a single row with one column per department count. You want to convert those columns back into rows.

**Answer:**
```sql
select dept.deptno,
       case dept.deptno
         when 10 then emp_cnts.deptno_10
         when 20 then emp_cnts.deptno_20
         when 30 then emp_cnts.deptno_30
       end as counts_by_dept
from (select 3 as deptno_10, 5 as deptno_20, 6 as deptno_30 from t1) emp_cnts,
     (select deptno from dept where deptno in (10,20,30)) dept;
```

**Sample Output:**

| DEPTNO | counts_by_dept |
|--------|----------------|
| 10     | 3              |
| 20     | 5              |
| 30     | 6              |

---

## Recipe 12.9 — Creating Horizontal Histograms

**Question:**  
You want to create a horizontal histogram of employee counts per job using repeated characters.

**Answer:**
```sql
select job,
       rpad('*', count(*), '*') as cnt
from emp
group by job
order by 2 desc;
```

**Sample Output:**

| JOB       | cnt      |
|-----------|----------|
| CLERK     | \*\*\*\* |
| SALESMAN  | \*\*\*\* |
| MANAGER   | \*\*\*   |
| ANALYST   | \*\*     |
| PRESIDENT | \*       |

---

## Recipe 12.12 — Calculating Simple Subtotals (ROLLUP)

**Question:**  
You want to generate subtotals for salary by job, plus a grand total for all jobs.

**Answer:**
```sql
select job, sum(sal) as total_sal
from emp
group by rollup(job)
order by job;
```

**Sample Output:**

| JOB       | total_sal |
|-----------|-----------|
| ANALYST   | 6000      |
| CLERK     | 4150      |
| MANAGER   | 8275      |
| PRESIDENT | 5000      |
| SALESMAN  | 5600      |
|           | 29025     |

> The row with NULL job is the grand total, produced by ROLLUP.

---

## Recipe 12.18 — Aggregations over Different Groups Simultaneously

**Question:**  
You want to compute each employee's salary alongside the maximum salary for their department and their job, all in one query — without self-joining.

**Answer:**
```sql
select empno, deptno, job, ename, sal,
       max(sal) over (partition by deptno) as max_by_dept,
       max(sal) over (partition by job)    as max_by_job
from emp;
```

**Sample Output:**

| EMPNO | DEPTNO | JOB      | ENAME  | SAL  | max_by_dept | max_by_job |
|-------|--------|----------|--------|------|-------------|------------|
| 7782  | 10     | MANAGER  | CLARK  | 2450 | 5000        | 2975       |
| 7839  | 10     | PRESIDENT| KING   | 5000 | 5000        | 5000       |
| 7934  | 10     | CLERK    | MILLER | 1300 | 5000        | 1300       |

---

# Chapter 13 — Hierarchical Queries

---

## Recipe 13.1 — Expressing a Parent-Child Relationship

**Question:**  
You want to display each employee with their manager's name using a self-join on the EMP table.

**Answer:**
```sql
select e.ename as employee,
       m.ename as manager
from emp e
left join emp m on (e.mgr = m.empno);
```

**Sample Output:**

| employee | manager |
|----------|---------|
| SMITH    | FORD    |
| ALLEN    | BLAKE   |
| JONES    | KING    |
| BLAKE    | KING    |
| KING     |         |

---

## Recipe 13.3 — Creating a Hierarchical View of a Table

**Question:**  
You want to display the entire organizational hierarchy from KING (top) down, formatted with indentation to show level.

**Answer:**
```sql
-- Oracle (CONNECT BY)
select lpad(' ', 2*(level-1)) || ename as org_chart
from emp
start with mgr is null
connect by prior empno = mgr;
```

**Sample Output:**

```
KING
  JONES
    SCOTT
      ADAMS
    FORD
      SMITH
  BLAKE
    ALLEN
    ...
  CLARK
    MILLER
```

---

# Chapter 14 — Odds 'n' Ends

---

## Recipe 14.1 — Cross-Tab Reports Using SQL Server's PIVOT

**Question:**  
You want to create a cross-tab (pivot) report using SQL Server's native PIVOT operator to show employee counts per department per job.

**Answer:**
```sql
-- SQL Server
select *
from (select job, deptno from emp) src
pivot (
  count(deptno)
  for deptno in ([10],[20],[30])
) as pvt;
```

**Sample Output:**

| JOB       | 10 | 20 | 30 |
|-----------|----|----|----|
| ANALYST   | 0  | 2  | 0  |
| CLERK     | 1  | 2  | 1  |
| MANAGER   | 1  | 1  | 1  |
| PRESIDENT | 1  | 0  | 0  |
| SALESMAN  | 0  | 0  | 4  |

---

## Recipe 14.12 — Calculating Percent Relative to Total

**Question:**  
You want to show each employee's salary alongside the percentage it represents of the total payroll.

**Answer:**
```sql
-- Oracle
select ename, sal,
       round(sal / (select sum(sal) from emp) * 100, 2) as pct_of_total
from emp
order by 3 desc;
```

**Sample Output:**

| ENAME  | SAL  | pct_of_total |
|--------|------|--------------|
| KING   | 5000 | 17.26        |
| SCOTT  | 3000 | 10.35        |
| FORD   | 3000 | 10.35        |
| JONES  | 2975 | 10.27        |
| BLAKE  | 2850 | 9.84         |
| CLARK  | 2450 | 8.46         |
| ALLEN  | 1600 | 5.52         |
| TURNER | 1500 | 5.18         |
| MILLER | 1300 | 4.49         |
| MARTIN | 1250 | 4.31         |
| WARD   | 1250 | 4.31         |
| ADAMS  | 1100 | 3.80         |
| JAMES  | 950  | 3.28         |
| SMITH  | 800  | 2.76         |

---

# Appendix A — Window Function Refresher

---

## A.1 — Grouping vs. Windowing

**Question:**  
What is the difference between a regular `GROUP BY` aggregate and a window function aggregate?

**Answer:**

```sql
-- GROUP BY collapses rows — you lose individual row detail:
select deptno, count(*) as cnt
from emp
group by deptno;

-- Window function keeps all rows AND adds the aggregate:
select deptno, ename,
       count(*) over (partition by deptno) as cnt
from emp;
```

**GROUP BY Output:**

| DEPTNO | cnt |
|--------|-----|
| 10     | 3   |
| 20     | 5   |
| 30     | 6   |

**Window Function Output:**

| DEPTNO | ENAME  | cnt |
|--------|--------|-----|
| 10     | CLARK  | 3   |
| 10     | KING   | 3   |
| 10     | MILLER | 3   |
| 20     | SMITH  | 5   |
| 20     | JONES  | 5   |

---

## A.2 — Windowing (Framing) Clause

**Question:**  
How do you use the framing clause in a window function to compute a running total that only accumulates rows up to and including the current row?

**Answer:**
```sql
select ename, sal,
       sum(sal) over (
         order by sal
         rows between unbounded preceding and current row
       ) as running_total
from emp;
```

**Sample Output:**

| ENAME  | SAL  | running_total |
|--------|------|---------------|
| SMITH  | 800  | 800           |
| JAMES  | 950  | 1750          |
| ADAMS  | 1100 | 2850          |
| WARD   | 1250 | 4100          |
| MARTIN | 1250 | 5350          |
| MILLER | 1300 | 6650          |

---

# Appendix B — Rozenshtein Revisited

---

## B.2 — Answering Questions Involving Negation

**Question:**  
Return all departments that have **no** clerks.

**Answer:**
```sql
select deptno
from emp
group by deptno
having sum(case when job = 'CLERK' then 1 else 0 end) = 0;
```

**Sample Output:**

*(No rows — every department has at least one CLERK in the standard EMP data.)*

---

## B.4 — Answering Questions Involving "At Least"

**Question:**  
Return departments that have **at least** two clerks.

**Answer:**
```sql
select deptno
from emp
group by deptno
having sum(case when job = 'CLERK' then 1 else 0 end) >= 2;
```

**Sample Output:**

| DEPTNO |
|--------|
| 20     |

---

## B.5 — Answering Questions Involving "Exactly"

**Question:**  
Return departments that have **exactly** two analysts.

**Answer:**
```sql
select deptno
from emp
group by deptno
having sum(case when job = 'ANALYST' then 1 else 0 end) = 2;
```

**Sample Output:**

| DEPTNO |
|--------|
| 20     |

---

*End of Study Guide — SQL Cookbook by Anthony Molinaro*  
*All queries use EMP / DEPT tables. Vendor-specific syntax varies — see each recipe's Discussion section for alternatives.*
