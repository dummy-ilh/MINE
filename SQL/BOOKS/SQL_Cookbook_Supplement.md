# SQL Cookbook — Missing Recipes Supplement
> Covers all recipes omitted from the original study guide  
> Same EMP / DEPT tables apply throughout

---

# Chapter 3 — Working with Multiple Tables (cont.)

---

## Recipe 3.10 — Performing Outer Joins when Using Aggregates

**Question:**  
You want to count bonuses received per department, but only some employees have bonuses. A regular join would exclude departments with no bonuses. How do you include all departments and still aggregate correctly?

**Answer:**
```sql
select deptno,
       sum(distinct sal)  as total_sal,
       sum(bonus)         as total_bonus
from (
  select e.empno, e.ename, e.sal, e.deptno,
         e.sal * case when eb.type = 1 then 0.1
                      when eb.type = 2 then 0.2
                      else 0.3 end as bonus
  from emp e left join emp_bonus eb on (e.empno = eb.empno)
  where e.deptno = 10
) x
group by deptno;
```

**Sample Output:**

| DEPTNO | total_sal | total_bonus |
|--------|-----------|-------------|
| 10     | 8750      | 1750        |

> **Key concept:** Use `SUM(DISTINCT sal)` to prevent salary from being counted multiple times when an employee has more than one bonus row.

---

## Recipe 3.11 — Returning Missing Data from Multiple Tables

**Question:**  
You want a full outer join result — all employees and all departments — even where there is no match on either side. Since SQLite/MySQL lack `FULL OUTER JOIN`, simulate it.

**Answer:**
```sql
-- DB2, PostgreSQL, SQL Server
select d.deptno, d.dname, e.ename
from dept d full outer join emp e on (d.deptno = e.deptno);

-- MySQL / simulation using UNION of two outer joins
select d.deptno, d.dname, e.ename
from dept d left join emp e on (d.deptno = e.deptno)
union
select d.deptno, d.dname, e.ename
from emp e left join dept d on (d.deptno = e.deptno);
```

**Sample Output:**

| DEPTNO | DNAME      | ENAME  |
|--------|------------|--------|
| 10     | ACCOUNTING | CLARK  |
| 10     | ACCOUNTING | KING   |
| 10     | ACCOUNTING | MILLER |
| 20     | RESEARCH   | SMITH  |
| ...    | ...        | ...    |
| 40     | OPERATIONS |        |

> Department 40 appears with no employee — the FULL OUTER JOIN reveals it.

---

# Chapter 4 — Inserting, Updating, Deleting (cont.)

---

## Recipe 4.2 — Inserting Default Values

**Question:**  
You want to insert a row into a table and have some columns take their default values automatically.

**Answer:**
```sql
-- Explicit DEFAULT keyword (DB2, Oracle, SQL Server, MySQL, PostgreSQL)
insert into D (id, foo)
values (default, 'Brighten');

-- Omit the column entirely (works everywhere)
insert into D (foo)
values ('Brighten');
```

**Sample Output:**

| ID | FOO      |
|----|----------|
| 1  | Brighten |

> `DEFAULT` instructs the database to apply whatever default was defined for that column at table-creation time.

---

## Recipe 4.3 — Overriding a Default Value with NULL

**Question:**  
A column has a default value, but you want to explicitly store NULL in it for a particular row.

**Answer:**
```sql
insert into D (id, foo)
values (null, 'Brighten');
```

**Sample Output:**

| ID   | FOO      |
|------|----------|
|      | Brighten |

> Only works if the column allows NULLs. Using `NULL` overrides the column's default entirely.

---

## Recipe 4.5 — Copying a Table Definition

**Question:**  
You want to create a new empty table with the same column structure as an existing table, but without copying any rows.

**Answer:**
```sql
-- DB2, Oracle, MySQL, PostgreSQL
create table dept_2
as
select * from dept where 1 = 0;

-- SQL Server
select * into dept_2
from dept where 1 = 0;
```

**Result:**  
`dept_2` exists with columns DEPTNO, DNAME, LOC — but zero rows.

---

## Recipe 4.6 — Inserting into Multiple Tables at Once

**Question:**  
You want to take rows from EMP and insert them into multiple target tables simultaneously based on conditions (Oracle-specific).

**Answer:**
```sql
-- Oracle only
insert all
  when deptno = 10 then
    into dept_10 values (empno, ename, job, mgr, hiredate, sal, comm, deptno)
  when deptno = 20 then
    into dept_20 values (empno, ename, job, mgr, hiredate, sal, comm, deptno)
  when deptno = 30 then
    into dept_30 values (empno, ename, job, mgr, hiredate, sal, comm, deptno)
select * from emp;
```

**Result:**  
CLARK, KING, MILLER → inserted into `dept_10`  
SMITH, JONES, etc. → inserted into `dept_20`  
ALLEN, WARD, etc. → inserted into `dept_30`

---

## Recipe 4.7 — Blocking Inserts to Certain Columns

**Question:**  
You want to prevent users from inserting values into specific columns (e.g., SAL and COMM) on the EMP table.

**Answer:**
```sql
-- Create a view that exposes only the safe columns
create view new_emps as
select empno, ename, job, mgr, hiredate, deptno
from emp;

-- Grant INSERT only on the view, not the base table
grant insert on new_emps to user_x;
```

> Users inserting via `new_emps` cannot supply values for SAL or COMM — those columns simply won't appear.

---

## Recipe 4.10 — Updating with Values from Another Table

**Question:**  
You have a table `new_sal` with updated salaries. You want to update EMP's SAL and COMM columns using values from `new_sal` where the departments match.

**Answer:**
```sql
-- MySQL
update emp e, new_sal ns
set e.sal  = ns.sal,
    e.comm = ns.sal / 2
where e.deptno = ns.deptno;

-- DB2 / SQL Server (using correlated update)
update emp
set sal  = (select sal  from new_sal where deptno = emp.deptno),
    comm = (select sal/2 from new_sal where deptno = emp.deptno)
where deptno in (select deptno from new_sal);
```

**Sample Input (new_sal):**

| DEPTNO | SAL  |
|--------|------|
| 10     | 4000 |

**Sample Output (dept 10 employees after update):**

| ENAME  | SAL  | COMM |
|--------|------|------|
| CLARK  | 4000 | 2000 |
| KING   | 4000 | 2000 |
| MILLER | 4000 | 2000 |

---

## Recipe 4.11 — Merging Records (UPSERT)

**Question:**  
You want to conditionally insert or update rows in a target table based on whether matching rows exist — an "upsert" operation.

**Answer:**
```sql
-- Oracle / DB2 (MERGE statement)
merge into emp_commission ec
using (select * from emp) emp
   on (ec.empno = emp.empno)
when matched then
  update set ec.comm = emp.sal * 0.1
when not matched then
  insert (ec.empno, ec.ename, ec.deptno, ec.comm)
  values (emp.empno, emp.ename, emp.deptno, emp.sal * 0.1);
```

**Result:**  
- Employees already in `emp_commission` → commission updated to 10% of SAL  
- Employees not yet in `emp_commission` → new row inserted

---

## Recipe 4.13 — Deleting Specific Records

**Question:**  
You want to delete only employees in department 10.

**Answer:**
```sql
delete from emp
where deptno = 10;
```

**Sample Output (remaining rows):**

| EMPNO | ENAME  | DEPTNO |
|-------|--------|--------|
| 7369  | SMITH  | 20     |
| 7499  | ALLEN  | 30     |
| ...   | ...    | ...    |

---

## Recipe 4.14 — Deleting a Single Record

**Question:**  
You want to delete exactly one specific row — for example, employee CLARK (EMPNO 7782).

**Answer:**
```sql
delete from emp
where empno = 7782;
```

> Using the primary key (EMPNO) guarantees exactly one row is deleted.

---

## Recipe 4.15 — Deleting Referential Integrity Violations

**Question:**  
You want to delete rows from EMP where the DEPTNO does not exist in the DEPT table (orphaned records).

**Answer:**
```sql
delete from emp
where deptno not in (select deptno from dept);

-- Safer alternative using NOT EXISTS:
delete from emp
where not exists (
  select 1 from dept
  where dept.deptno = emp.deptno
);
```

---

## Recipe 4.17 — Deleting Records Referenced from Another Table

**Question:**  
You want to delete employees who are in departments listed in a table called `dept_accidents`.

**Answer:**
```sql
delete from emp
where deptno in (select deptno from dept_accidents);
```

**Sample Input (dept_accidents):**

| DEPTNO |
|--------|
| 20     |

**Result:** All employees in department 20 are deleted from EMP.

---

# Chapter 5 — Metadata Queries (cont.)

---

## Recipe 5.3 — Listing Indexed Columns for a Table

**Question:**  
You want to see which columns of a table have indexes defined on them.

**Answer:**
```sql
-- Oracle
select table_name, index_name, column_name, column_position
from all_ind_columns
where table_name = 'EMP';

-- DB2
select indname, colname, colseq
from syscat.indexcoluse
where indschema = 'YOUR_SCHEMA';

-- MySQL
show index from emp;

-- PostgreSQL
select a.attname, i.relname
from pg_class t
join pg_index ix  on t.oid = ix.indrelid
join pg_class i   on i.oid = ix.indexrelid
join pg_attribute a on a.attrelid = t.oid and a.attnum = any(ix.indkey)
where t.relname = 'emp';

-- SQL Server
select a.name as index_name, b.name as column_name
from sys.indexes a
join sys.index_columns ic on a.object_id = ic.object_id and a.index_id = ic.index_id
join sys.columns b on ic.object_id = b.object_id and ic.column_id = b.column_id
where a.object_id = object_id('emp');
```

**Sample Output:**

| TABLE_NAME | INDEX_NAME | COLUMN_NAME | COLUMN_POSITION |
|------------|------------|-------------|-----------------|
| EMP        | PK_EMP     | EMPNO       | 1               |
| EMP        | IDX_DEPTNO | DEPTNO      | 1               |

---

## Recipe 5.4 — Listing Constraints on a Table

**Question:**  
You want to list all constraints (primary key, foreign key, unique, check) on the EMP table.

**Answer:**
```sql
-- Oracle
select constraint_name, constraint_type
from all_constraints
where table_name = 'EMP';

-- MySQL
select constraint_name, constraint_type
from information_schema.table_constraints
where table_name = 'emp';

-- SQL Server
select cc.name as constraint_name,
       cc.type_desc as constraint_type
from sys.check_constraints cc
where parent_object_id = object_id('emp');
```

**Sample Output:**

| CONSTRAINT_NAME | CONSTRAINT_TYPE |
|-----------------|-----------------|
| PK_EMP          | P               |
| FK_DEPTNO       | R               |
| CHK_SAL         | C               |

---

## Recipe 5.5 — Listing Foreign Keys Without Corresponding Indexes

**Question:**  
You want to find foreign key columns that have no supporting index — a common performance pitfall.

**Answer:**
```sql
-- Oracle
select a.table_name, a.constraint_name, a.column_name, a.position
from all_cons_columns a
join all_constraints b
  on (a.constraint_name = b.constraint_name
  and b.constraint_type = 'R'
  and b.owner = a.owner)
where a.owner = 'YOUR_SCHEMA'
  and not exists (
    select null
    from all_ind_columns c
    where c.table_name  = a.table_name
      and c.column_name = a.column_name
      and c.column_position = a.position
  );
```

**Sample Output:**

| TABLE_NAME | CONSTRAINT_NAME | COLUMN_NAME | POSITION |
|------------|-----------------|-------------|----------|
| EMP        | FK_DEPTNO       | DEPTNO      | 1        |

---

## Recipe 5.7 — Describing Data Dictionary Views in Oracle

**Question:**  
You want to see which Oracle data dictionary views are available and what they describe.

**Answer:**
```sql
-- List all dictionary views and their descriptions
select table_name, comments
from dictionary
order by table_name;

-- Describe columns of a specific view
select column_name, comments
from col_comments
where table_name = 'ALL_TAB_COLUMNS';
```

**Sample Output:**

| TABLE_NAME         | COMMENTS                                     |
|--------------------|----------------------------------------------|
| ALL_CONSTRAINTS    | Constraints on accessible tables             |
| ALL_IND_COLUMNS    | Columns comprising indexes on accessible tbl |
| ALL_TAB_COLUMNS    | Columns of user's tables, views, clusters    |

---

# Chapter 6 — Working with Strings (cont.)

---

## Recipe 6.2 — Embedding Quotes Within String Literals

**Question:**  
You want to include a single quote character inside a string literal, e.g., produce the output `g'day`.

**Answer:**
```sql
-- Standard: double up the single quote
select 'g''day' from t1;

-- Alternative: use two separate strings (some vendors)
select 'g' || chr(39) || 'day' from t1;  -- Oracle/DB2
select concat('g', char(39), 'day')  from t1;  -- MySQL/SQL Server
```

**Sample Output:**

| result |
|--------|
| g'day  |

---

## Recipe 6.4 — Removing Unwanted Characters from a String

**Question:**  
You want to strip all zeros and vowels from employee names and salaries.

**Answer:**
```sql
-- Oracle / DB2 (TRANSLATE + REPLACE)
select ename,
       translate(ename, 'AEIOU', 'xxxxx') as stripped1,
       replace(translate(ename,'AEIOU','AAAAA'),'A') as stripped2,
       sal,
       replace(cast(sal as char(4)), '0', '') as sal_stripped
from emp;
```

**Sample Input:**

| ENAME | SAL  |
|-------|------|
| ALLEN | 1600 |
| JONES | 2975 |

**Sample Output:**

| ENAME | stripped1 | sal_stripped |
|-------|-----------|--------------|
| ALLEN | xLLxN     | 16           |
| JONES | JxNxS     | 2975         |

---

## Recipe 6.5 — Separating Numeric and Character Data

**Question:**  
A column stores data like `'SMITH800'` where name and salary are concatenated. You want to split them into two columns.

**Answer:**
```sql
-- Oracle
select replace(
         translate(data, '0123456789', '9999999999'), '9', '') as ename,
       to_number(
         translate(data,
           replace(translate(data,'0123456789','9999999999'),'9',''),
           rpad('9',20,'9'))) as sal
from (select ename||sal as data from emp);
```

**Sample Input:**

| data       |
|------------|
| SMITH800   |
| ALLEN1600  |

**Sample Output:**

| ename | sal  |
|-------|------|
| SMITH | 800  |
| ALLEN | 1600 |

---

## Recipe 6.6 — Determining Whether a String Is Alphanumeric

**Question:**  
Given a mixed set of strings, return only those that contain exclusively letters and digits (no special characters).

**Answer:**
```sql
-- Oracle / DB2 (using TRANSLATE)
select data
from V
where translate(lower(data),
        '0123456789abcdefghijklmnopqrstuvwxyz',
        rpad('a', 36, 'a')) = rpad('a', length(data), 'a');

-- MySQL (using REGEXP)
select data
from V
where data regexp '^[a-zA-Z0-9]+$';
```

**Sample Input (view V):**

| data        |
|-------------|
| CLARK       |
| MILLER      |
| 123         |
| 7,369       |
| SCOTT820    |

**Sample Output:**

| data     |
|----------|
| CLARK    |
| MILLER   |
| 123      |
| SCOTT820 |

---

## Recipe 6.8 — Ordering by Parts of a String

**Question:**  
You want to sort employee names by the last two characters of their name.

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
| MARTIN |
| BLAKE  |
| ADAMS  |
| KING   |
| CLARK  |
| SMITH  |
| SCOTT  |
| FORD   |
| WARD   |

---

## Recipe 6.9 — Ordering by a Number in a String

**Question:**  
A view returns strings like `'CLARK 7782 ACCOUNTING'`. You want to sort these rows by the embedded employee number.

**Answer:**
```sql
-- Oracle
select data
from V
order by to_number(
  replace(
    translate(data,
      replace(translate(data,'0123456789',rpad('#',10,'#')),'#',''),
      rpad('#',20,'#')),
    '#',''));

-- SQL Server
select data
from V
order by cast(
  replace(
    translate(data, /* ... same approach */ ) as integer);
```

**Sample Input (view V):**

| data                   |
|------------------------|
| CLARK 7782 ACCOUNTING  |
| KING 7839 ACCOUNTING   |
| SMITH 7369 RESEARCH    |

**Sample Output:**

| data                  |
|-----------------------|
| SMITH 7369 RESEARCH   |
| CLARK 7782 ACCOUNTING |
| KING 7839 ACCOUNTING  |

---

## Recipe 6.11 — Converting Delimited Data into a Multi-Valued IN-List

**Question:**  
You have a comma-delimited string `'7654,7698,7782,7788'` and want to use it as an IN-list to query EMP.

**Answer:**
```sql
-- DB2 (walk the string into rows, then join)
select empno, ename, sal, deptno
from emp
where empno in (
  select cast(substr(c,2,locate(',',c,2)-2) as integer) as empno
  from (
    select substr(csv.emps, iter.pos) as c
    from (select ','||'7654,7698,7782,7788'||',' emps from t1) csv,
         (select id as pos from t100) iter
    where iter.pos <= length(csv.emps)
  ) x
  where c like ',%'
);
```

**Sample Output:**

| EMPNO | ENAME  | SAL  | DEPTNO |
|-------|--------|------|--------|
| 7654  | MARTIN | 1250 | 30     |
| 7698  | BLAKE  | 2850 | 30     |
| 7782  | CLARK  | 2450 | 10     |
| 7788  | SCOTT  | 3000 | 20     |

---

## Recipe 6.12 — Alphabetizing a String

**Question:**  
You want to alphabetize the individual characters within each employee's name. For example, `'KING'` becomes `'GIKN'`.

**Answer:**
```sql
-- Oracle (using CONNECT BY to walk + re-aggregate)
select ename, listagg(c) within group (order by c) as alphabetized
from (
  select ename, substr(ename, level, 1) as c
  from emp
  connect by level <= length(ename)
    and prior ename = ename
    and prior dbms_random.value is not null
)
group by ename;
```

**Sample Output:**

| ENAME  | alphabetized |
|--------|--------------|
| ADAMS  | AADMS        |
| ALLEN  | AELLN        |
| KING   | GIKN         |
| SMITH  | HIMST        |

---

## Recipe 6.13 — Identifying Strings That Can Be Treated as Numbers

**Question:**  
A column holds mixed data — some values are numbers stored as strings, some are pure text. Return only those that can be cast to a number.

**Answer:**
```sql
-- Oracle
select to_number(case
  when translate(data,'0123456789','9999999999') = rpad('9',length(data),'9')
  then data
  end) as data
from V
where translate(data,'0123456789','9999999999') = rpad('9',length(data),'9');

-- SQL Server
select cast(data as integer) as data
from V
where isnumeric(data) = 1;
```

**Sample Input:**

| data   |
|--------|
| 123    |
| CLARK  |
| 456    |
| 7,369  |

**Sample Output:**

| data |
|------|
| 123  |
| 456  |

---

## Recipe 6.14 — Extracting the nth Delimited Substring

**Question:**  
Given strings like `'mo,larry,curly'`, extract the 2nd comma-delimited token (`'larry'`).

**Answer:**
```sql
-- MySQL
select substring_index(
         substring_index('mo,larry,curly', ',', 2),
         ',',-1) as sub;

-- Oracle
select regexp_substr('mo,larry,curly','[^,]+',1,2) as sub from dual;

-- PostgreSQL
select split_part('mo,larry,curly',',',2) as sub;
```

**Sample Output:**

| sub   |
|-------|
| larry |

---

## Recipe 6.15 — Parsing an IP Address

**Question:**  
Given an IP address string `'111.22.3.4'`, return each octet as a separate column.

**Answer:**
```sql
-- MySQL
select substring_index(ip,'.',1) as a,
       substring_index(substring_index(ip,'.',2),'.',-1) as b,
       substring_index(substring_index(ip,'.',3),'.',-1) as c,
       substring_index(ip,'.',-1) as d
from (select '111.22.3.4' as ip from t1) t;

-- Oracle
select regexp_substr(ip,'[^.]+',1,1) as a,
       regexp_substr(ip,'[^.]+',1,2) as b,
       regexp_substr(ip,'[^.]+',1,3) as c,
       regexp_substr(ip,'[^.]+',1,4) as d
from (select '111.22.3.4' as ip from dual);
```

**Sample Output:**

| a   | b  | c | d |
|-----|----|---|---|
| 111 | 22 | 3 | 4 |

---

# Chapter 7 — Working with Numbers (cont.)

---

## Recipe 7.3 — Summing the Values in a Column

**Question:**  
You want to calculate the total salary for all employees and the total salary per department.

**Answer:**
```sql
-- Grand total
select sum(sal) as total_sal
from emp;

-- Per department
select deptno, sum(sal) as total_sal
from emp
group by deptno;
```

**Sample Output:**

| DEPTNO | total_sal |
|--------|-----------|
| 10     | 8750      |
| 20     | 10875     |
| 30     | 9400      |
| (all)  | 29025     |

---

## Recipe 7.4 — Counting Rows in a Table

**Question:**  
You want to count the total number of rows in EMP and also count rows per department.

**Answer:**
```sql
-- Total rows
select count(*) as cnt from emp;

-- Per department
select deptno, count(*) as cnt
from emp
group by deptno;
```

**Sample Output:**

| DEPTNO | cnt |
|--------|-----|
| 10     | 3   |
| 20     | 5   |
| 30     | 6   |

---

## Recipe 7.5 — Counting Values in a Column

**Question:**  
You want to count how many employees have a non-NULL commission value.

**Answer:**
```sql
select count(comm) as cnt_comm,
       count(*)    as cnt_all
from emp;
```

**Sample Output:**

| cnt_comm | cnt_all |
|----------|---------|
| 4        | 14      |

> `COUNT(column)` ignores NULLs; `COUNT(*)` counts every row including those with NULLs.

---

## Recipe 7.7 — Generating a Running Product

**Question:**  
You want to compute a cumulative product of EMPNOs (a mathematical running product) as rows are processed in order.

**Answer:**
```sql
-- DB2, Oracle, SQL Server (using LOG/EXP trick)
select empno, ename,
       exp(sum(ln(empno)) over (order by empno)) as running_product
from emp
where deptno = 10;
```

**Sample Output:**

| EMPNO | ENAME  | running_product |
|-------|--------|-----------------|
| 7782  | CLARK  | 7782            |
| 7839  | KING   | 60434898        |
| 7934  | MILLER | 477842591412    |

---

## Recipe 7.8 — Calculating a Running Difference

**Question:**  
You want to subtract each subsequent salary from a running total — producing a running difference ordered by EMPNO.

**Answer:**
```sql
select empno, ename, sal,
       case when rn = 1 then sal
            else sal * -1 end as adj_sal
from (
  select empno, ename, sal,
         row_number() over (order by empno) rn
  from emp
  where deptno = 10
)
-- Then take the running sum of adj_sal
```

**Simplified Oracle version:**
```sql
select empno, ename, sal,
       sum(case when rn = 1 then sal else -sal end)
         over (order by empno rows unbounded preceding) as running_diff
from (
  select empno, ename, sal,
         row_number() over (order by empno) rn
  from emp where deptno = 10
);
```

**Sample Output:**

| EMPNO | ENAME  | SAL  | running_diff |
|-------|--------|------|--------------|
| 7782  | CLARK  | 2450 | 2450         |
| 7839  | KING   | 5000 | -2550        |
| 7934  | MILLER | 1300 | -3850        |

---

## Recipe 7.12 — Aggregating Nullable Columns

**Question:**  
You want to compute the average commission, but NULLs should be treated as 0 rather than excluded from the average.

**Answer:**
```sql
-- Average ignoring NULLs (default behavior — 4 non-null rows)
select avg(comm) as avg_comm_ignore_null
from emp;

-- Average treating NULLs as 0 (all 14 rows)
select avg(coalesce(comm, 0)) as avg_comm_with_zero
from emp;
```

**Sample Output:**

| avg_comm_ignore_null | avg_comm_with_zero |
|----------------------|--------------------|
| 550                  | 157.14             |

---

## Recipe 7.13 — Computing Averages Without High and Low Values

**Question:**  
You want to calculate the average salary in EMP after excluding the single highest and single lowest salaries (a trimmed mean).

**Answer:**
```sql
select avg(sal) as avg_without_extremes
from emp
where sal not in (
  (select min(sal) from emp),
  (select max(sal) from emp)
);
```

**Sample Output:**

| avg_without_extremes |
|----------------------|
| 1885.71              |

---

## Recipe 7.14 — Converting Alphanumeric Strings into Numbers

**Question:**  
A column holds numeric values stored as strings (e.g., `'1234'`). You want to cast them to actual numbers for arithmetic.

**Answer:**
```sql
-- Oracle
select to_number(replace('$1,000.00', ',', '')) as num from dual;

-- SQL Server
select cast(replace('$1,000.00', ',', '') as decimal(10,2));

-- MySQL / PostgreSQL
select cast(replace('1000.00', ',', '') as decimal);
```

**Sample Output:**

| num     |
|---------|
| 1000.00 |

---

## Recipe 7.15 — Changing Values in a Running Total

**Question:**  
You want a running total of transactions, but purchases should add and withdrawals should subtract. Given a table with TRXTYPE ('PR' = purchase, 'PY' = payment) and AMT:

**Answer:**
```sql
select case when trxtype = 'PY' then 'PAYMENT' else 'PURCHASE' end as trx,
       amt,
       sum(case when trxtype = 'PY' then -amt else amt end)
         over (order by id
               rows between unbounded preceding and current row) as balance
from transactions;
```

**Sample Input:**

| ID | TRXTYPE | AMT |
|----|---------|-----|
| 1  | PR      | 100 |
| 2  | PR      | 200 |
| 3  | PY      | 50  |
| 4  | PR      | 150 |

**Sample Output:**

| trx      | amt | balance |
|----------|-----|---------|
| PURCHASE | 100 | 100     |
| PURCHASE | 200 | 300     |
| PAYMENT  | 50  | 250     |
| PURCHASE | 150 | 400     |

---

# Chapter 8 — Date Arithmetic (cont.)

---

## Recipe 8.4 — Number of Months or Years Between Two Dates

**Question:**  
You want to find the difference between ALLEN's and WARD's hire dates expressed in months and years.

**Answer:**
```sql
-- Oracle
select months_between(ward_hd, allen_hd) as months_diff,
       floor(months_between(ward_hd, allen_hd) / 12) as years_diff
from (
  select max(case when ename='ALLEN' then hiredate end) as allen_hd,
         max(case when ename='WARD'  then hiredate end) as ward_hd
  from emp
);

-- MySQL
select timestampdiff(month,  allen_hd, ward_hd) as months_diff,
       timestampdiff(year,   allen_hd, ward_hd) as years_diff
from (...);
```

**Sample Output:**

| months_diff | years_diff |
|-------------|------------|
| 0.07        | 0          |

---

## Recipe 8.5 — Seconds, Minutes, or Hours Between Two Dates

**Question:**  
You want to find the difference between two dates expressed in hours, minutes, and seconds.

**Answer:**
```sql
-- Oracle
select d1 - d2 as day_diff,
       (d1 - d2) * 24 as hour_diff,
       (d1 - d2) * 24 * 60 as minute_diff,
       (d1 - d2) * 24 * 60 * 60 as second_diff
from (
  select max(case when ename='ALLEN' then hiredate end) as d1,
         max(case when ename='WARD'  then hiredate end) as d2
  from emp
);

-- SQL Server
select datediff(hour,   d2, d1) as hour_diff,
       datediff(minute, d2, d1) as minute_diff,
       datediff(second, d2, d1) as second_diff
from (...);
```

**Sample Output:**

| day_diff | hour_diff | minute_diff | second_diff |
|----------|-----------|-------------|-------------|
| 2        | 48        | 2880        | 172800      |

---

## Recipe 8.6 — Counting Occurrences of Weekdays in a Year

**Question:**  
You want to count how many times each weekday (Monday through Sunday) appears in the current year.

**Answer:**
```sql
-- Oracle (using CONNECT BY to generate each day)
select to_char(d,'DY') as dow,
       count(*) as cnt
from (
  select trunc(sysdate,'YEAR') + level - 1 as d
  from dual
  connect by level <= (
    trunc(add_months(trunc(sysdate,'YEAR'),12)) - trunc(sysdate,'YEAR')
  )
)
group by to_char(d,'DY')
order by min(d);
```

**Sample Output:**

| DOW | cnt |
|-----|-----|
| MON | 52  |
| TUE | 52  |
| WED | 53  |
| THU | 52  |
| FRI | 52  |
| SAT | 52  |
| SUN | 52  |

---

## Recipe 8.7 — Date Difference Between Current and Next Record

**Question:**  
For each employee, you want to know how many days passed between their hire date and the hire date of the next person hired.

**Answer:**
```sql
-- DB2, Oracle, SQL Server (LEAD window function)
select ename, hiredate,
       lead(hiredate) over (order by hiredate) as next_hiredate,
       lead(hiredate) over (order by hiredate) - hiredate as days_diff
from emp;

-- MySQL / correlated subquery
select e.ename, e.hiredate,
       (select min(hiredate) from emp e2
        where e2.hiredate > e.hiredate) as next_hiredate
from emp e
order by hiredate;
```

**Sample Output:**

| ENAME  | HIREDATE    | next_hiredate | days_diff |
|--------|-------------|---------------|-----------|
| SMITH  | 17-DEC-1980 | 20-FEB-1981   | 65        |
| ALLEN  | 20-FEB-1981 | 22-FEB-1981   | 2         |
| WARD   | 22-FEB-1981 | 02-APR-1981   | 39        |

---

# Chapter 9 — Date Manipulation (cont.)

---

## Recipe 9.2 — Determining the Number of Days in a Year

**Question:**  
You want to find the total number of days in the current year (365 or 366 if leap year).

**Answer:**
```sql
-- Oracle
select add_months(trunc(sysdate,'YEAR'),12) - trunc(sysdate,'YEAR')
  as days_in_year
from dual;

-- DB2
select days(current_date + 1 year - dayofyear(current_date) day + 1 day)
     - days(current_date - dayofyear(current_date) day + 1 day)
  as days_in_year
from t1;

-- MySQL
select datediff(
  date_add(makedate(year(now()),1), interval 1 year),
  makedate(year(now()),1)) as days_in_year;
```

**Sample Output:**

| days_in_year |
|--------------|
| 365          |

---

## Recipe 9.5 — All Dates for a Particular Weekday Throughout a Year

**Question:**  
You want to return a list of all Fridays in the current year.

**Answer:**
```sql
-- Oracle (CONNECT BY)
select dy
from (
  select trunc(sysdate,'YEAR') + level - 1 as dy
  from dual
  connect by level <= 366
)
where to_char(dy,'DY') = 'FRI'
  and extract(year from dy) = extract(year from sysdate);

-- MySQL
select dy
from (
  select adddate(makedate(year(now()),1), t.n) as dy
  from (select a.n + b.n * 10 + c.n * 100 as n
        from (select 0 n union select 1 union ... select 9) a, ...) t
  where adddate(makedate(year(now()),1),t.n) < makedate(year(now())+1,1)
) x
where dayname(dy) = 'Friday';
```

**Sample Output (partial):**

| DY          |
|-------------|
| 07-JAN-2005 |
| 14-JAN-2005 |
| 21-JAN-2005 |
| 28-JAN-2005 |
| ...         |

---

## Recipe 9.6 — First and Last Occurrence of a Weekday in a Month

**Question:**  
You want to find the first and last Monday of the current month.

**Answer:**
```sql
-- Oracle
select next_day(trunc(sysdate,'MM')-1, 'MONDAY') as first_monday,
       next_day(last_day(trunc(sysdate,'MM'))-7, 'MONDAY') as last_monday
from dual;

-- MySQL
select case when dayname(first_of_month) = 'Monday' then first_of_month
            else date_add(first_of_month,
                   interval (7 - weekday(first_of_month)) day) end
       as first_monday
from (select date_add(current_date, interval -day(current_date)+1 day)
      as first_of_month) t;
```

**Sample Output (March 2005):**

| first_monday | last_monday |
|--------------|-------------|
| 07-MAR-2005  | 28-MAR-2005 |

---

## Recipe 9.8 — Listing Quarter Start and End Dates for the Year

**Question:**  
You want to list the start and end dates of all four quarters of the current year.

**Answer:**
```sql
-- Oracle (using CONNECT BY to generate 4 rows)
select rownum as qtr,
       add_months(trunc(sysdate,'YEAR'), (rownum-1)*3) as q_start,
       add_months(trunc(sysdate,'YEAR'), rownum*3) - 1 as q_end
from emp
where rownum <= 4;
```

**Sample Output (2005):**

| QTR | Q_START     | Q_END       |
|-----|-------------|-------------|
| 1   | 01-JAN-2005 | 31-MAR-2005 |
| 2   | 01-APR-2005 | 30-JUN-2005 |
| 3   | 01-JUL-2005 | 30-SEP-2005 |
| 4   | 01-OCT-2005 | 31-DEC-2005 |

---

## Recipe 9.9 — Quarter Start and End Dates for a Given Quarter

**Question:**  
Given a YYYYQ integer (e.g., `20053` = Q3 2005), return the quarter's start and end dates.

**Answer:**
```sql
-- Oracle
select qtr,
       to_date(yr || lpad((qtr*3)-2, 2, 0), 'YYYYMM') as q_start,
       to_date(yr || lpad( qtr*3,    2, 0), 'YYYYMM') + interval '1' month
         - interval '1' day as q_end
from (
  select mod(yrq,10) as qtr,
         trunc(yrq/10) as yr
  from (select 20053 as yrq from t1)
);
```

**Sample Output:**

| QTR | Q_START     | Q_END       |
|-----|-------------|-------------|
| 3   | 01-JUL-2005 | 30-SEP-2005 |

---

## Recipe 9.10 — Filling in Missing Dates

**Question:**  
You want to count employees hired each month from 1980–1983, but some months have no hirings. You want those months to appear with a count of 0.

**Answer:**
```sql
-- Oracle (generate all months in range, then left-join to hirings)
select lpad(rownum,2,'0')||'/'||yr as mth,
       coalesce(num_hired, 0) as num_hired
from (
  select to_char(hiredate,'YYYY') as yr,
         to_char(hiredate,'MM') as mth,
         count(*) as num_hired
  from emp
  group by to_char(hiredate,'YYYY'), to_char(hiredate,'MM')
) x
right join (
  select level as rownum, yr
  from (select distinct to_char(hiredate,'YYYY') yr from emp),
       (select level from dual connect by level <= 12)
) months on x.mth = lpad(months.rownum,2,'0') and x.yr = months.yr
order by yr, mth;
```

**Sample Output (partial):**

| MTH     | num_hired |
|---------|-----------|
| 01/1981 | 0         |
| 02/1981 | 2         |
| 03/1981 | 0         |
| 04/1981 | 1         |
| 05/1981 | 1         |

---

## Recipe 9.11 — Searching on Specific Units of Time

**Question:**  
You want to find employees hired in February or December, or hired on a Tuesday, regardless of year.

**Answer:**
```sql
-- DB2 / MySQL
select ename, hiredate
from emp
where monthname(hiredate) in ('February','December')
   or dayname(hiredate) = 'Tuesday';

-- Oracle
select ename, hiredate
from emp
where to_char(hiredate,'DY')  = 'TUE'
   or to_char(hiredate,'MON') in ('FEB','DEC');
```

**Sample Output:**

| ENAME  | HIREDATE    |
|--------|-------------|
| ALLEN  | 20-FEB-1981 |
| WARD   | 22-FEB-1981 |
| JONES  | 02-APR-1981 |
| MARTIN | 28-SEP-1981 |

---

## Recipe 9.12 — Comparing Records Using Specific Parts of a Date

**Question:**  
You want to find pairs of employees who were hired in the same month and on the same weekday, regardless of year.

**Answer:**
```sql
-- Oracle
select a.ename || ' was hired on the same month and weekday as ' ||
       b.ename as msg
from emp a, emp b
where to_char(a.hiredate,'DY') = to_char(b.hiredate,'DY')
  and to_char(a.hiredate,'MM') = to_char(b.hiredate,'MM')
  and a.empno < b.empno;
```

**Sample Output:**

| MSG                                               |
|---------------------------------------------------|
| JAMES was hired on the same month and weekday as FORD  |
| SCOTT was hired on the same month and weekday as JAMES |
| SCOTT was hired on the same month and weekday as FORD  |

---

## Recipe 9.13 — Identifying Overlapping Date Ranges

**Question:**  
You have a table `EMP_PROJECT` showing project start and end dates. Find cases where an employee started a new project before finishing their current one.

**Answer:**
```sql
select a.empno, a.ename,
       a.proj_id as proj_a, a.proj_start as a_start, a.proj_end as a_end,
       b.proj_id as proj_b, b.proj_start as b_start, b.proj_end as b_end
from emp_project a, emp_project b
where a.empno = b.empno
  and b.proj_start >= a.proj_start
  and b.proj_start <= a.proj_end
  and a.proj_id != b.proj_id
order by a.empno;
```

**Sample Input (emp_project):**

| EMPNO | ENAME | PROJ_ID | PROJ_START  | PROJ_END    |
|-------|-------|---------|-------------|-------------|
| 7782  | CLARK | 1       | 16-JUN-2005 | 18-JUN-2005 |
| 7782  | CLARK | 4       | 19-JUN-2005 | 24-JUN-2005 |
| 7782  | CLARK | 7       | 22-JUN-2005 | 25-JUN-2005 |

**Sample Output:**

| EMPNO | ENAME | PROJ_A | A_START     | A_END       | PROJ_B | B_START     |
|-------|-------|--------|-------------|-------------|--------|-------------|
| 7782  | CLARK | 4      | 19-JUN-2005 | 24-JUN-2005 | 7      | 22-JUN-2005 |

---

# Chapter 10 — Working with Ranges (cont.)

---

## Recipe 10.3 — Beginning and End of a Range of Consecutive Values

**Question:**  
Using the same project view V from Recipe 10.1, find the start and end dates of each consecutive sequence of projects.

**Answer:**
```sql
-- Oracle / DB2 (using LAG/LEAD)
select proj_start, proj_end
from (
  select proj_start, proj_end,
         lag(proj_end) over (order by proj_start) as prior_end
  from V
)
where prior_end is null
   or prior_end != proj_start;

-- Then pair with the matching range end using LEAD
```

**Sample Input:**

| PROJ_ID | PROJ_START  | PROJ_END    |
|---------|-------------|-------------|
| 1       | 01-JAN-2005 | 02-JAN-2005 |
| 2       | 02-JAN-2005 | 03-JAN-2005 |
| 3       | 03-JAN-2005 | 04-JAN-2005 |
| 5       | 06-JAN-2005 | 07-JAN-2005 |

**Sample Output:**

| range_start | range_end   |
|-------------|-------------|
| 01-JAN-2005 | 04-JAN-2005 |
| 06-JAN-2005 | 07-JAN-2005 |

---

## Recipe 10.4 — Filling in Missing Values in a Range

**Question:**  
You want to count employees hired each year from 1980 to 1983, including years with zero hirings.

**Answer:**
```sql
-- Oracle (generate the year range, then left-join)
select yr.yr, coalesce(emp.cnt, 0) as cnt
from (
  select 1980 + level - 1 as yr
  from dual
  connect by level <= 4
) yr
left join (
  select extract(year from hiredate) as yr, count(*) as cnt
  from emp
  group by extract(year from hiredate)
) emp on yr.yr = emp.yr
order by yr.yr;
```

**Sample Output:**

| YR   | CNT |
|------|-----|
| 1980 | 1   |
| 1981 | 10  |
| 1982 | 2   |
| 1983 | 1   |

---

# Chapter 11 — Advanced Searching (cont.)

---

## Recipe 11.2 — Skipping n Rows from a Table

**Question:**  
You want to return every other row from EMP — i.e., skip alternating rows. Return rows where the row number is odd.

**Answer:**
```sql
-- DB2, Oracle, SQL Server
select ename
from (
  select ename,
         row_number() over (order by ename) as rn
  from emp
) x
where mod(rn, 2) = 1;

-- MySQL / PostgreSQL
select ename
from (
  select ename,
         @rn := @rn + 1 as rn
  from emp, (select @rn:=0) r
  order by ename
) x
where mod(rn, 2) = 1;
```

**Sample Output:**

| ENAME  |
|--------|
| ADAMS  |
| BLAKE  |
| FORD   |
| JAMES  |
| KING   |
| MARTIN |
| SCOTT  |

---

## Recipe 11.3 — Incorporating OR Logic when Using Outer Joins

**Question:**  
You want to outer-join EMP to EMP_BONUS, but include bonus rows for employees who received a type 1 OR type 2 bonus — not just one condition.

**Answer:**
```sql
select e.ename, e.deptno, eb.type
from emp e
left join emp_bonus eb
  on (e.empno = eb.empno
  and eb.type in (1, 2));
```

**Sample Output:**

| ENAME  | DEPTNO | type |
|--------|--------|------|
| CLARK  | 10     |      |
| KING   | 10     | 1    |
| MILLER | 10     | 2    |
| SMITH  | 20     |      |
| ...    | ...    | ...  |

---

## Recipe 11.4 — Determining Which Rows Are Reciprocals

**Question:**  
You have a table of test results with columns `TEST1` and `TEST2`. You want to find pairs of rows where the scores are swapped — e.g., row A has (20, 20) and row B has (20, 20), or one row is the mirror of another.

**Answer:**
```sql
select e1.empno, e1.test1, e1.test2
from emp_test e1, emp_test e2
where e1.test1 = e2.test2
  and e1.test2 = e2.test1
  and e1.empno > e2.empno;
```

**Sample Input:**

| EMPNO | TEST1 | TEST2 |
|-------|-------|-------|
| 1     | 90    | 50    |
| 2     | 50    | 90    |
| 3     | 80    | 80    |

**Sample Output:**

| EMPNO | TEST1 | TEST2 |
|-------|-------|-------|
| 2     | 50    | 90    |

---

## Recipe 11.7 — Investigating Future Rows

**Question:**  
You want to find any employees who earn less than the employee hired immediately after them.

**Answer:**
```sql
-- DB2, Oracle, SQL Server (LEAD)
select ename, sal, hiredate, next_sal
from (
  select ename, sal, hiredate,
         lead(sal) over (order by hiredate) as next_sal
  from emp
) x
where sal < next_sal;
```

**Sample Output:**

| ENAME  | SAL  | HIREDATE    | next_sal |
|--------|------|-------------|----------|
| SMITH  | 800  | 17-DEC-1980 | 1600     |
| WARD   | 1250 | 22-FEB-1981 | 2975     |
| TURNER | 1500 | 08-SEP-1981 | 1250     |

---

## Recipe 11.8 — Shifting Row Values

**Question:**  
You want to return each employee's salary along with the next highest and next lowest salary. If no higher salary exists, wrap around to the lowest; if no lower salary exists, wrap around to the highest.

**Answer:**
```sql
-- DB2, Oracle, SQL Server
select ename, sal,
       coalesce(lead(sal) over (order by sal), min(sal) over ()) as forward,
       coalesce(lag(sal)  over (order by sal), max(sal) over ()) as rewind
from emp;
```

**Sample Output:**

| ENAME  | SAL  | FORWARD | REWIND |
|--------|------|---------|--------|
| SMITH  | 800  | 950     | 5000   |
| JAMES  | 950  | 1100    | 800    |
| ADAMS  | 1100 | 1250    | 950    |
| WARD   | 1250 | 1250    | 1100   |
| MARTIN | 1250 | 1300    | 1250   |

---

## Recipe 11.10 — Suppressing Duplicates

**Question:**  
You want to return only the distinct job types from EMP without duplicates, using methods beyond just `DISTINCT`.

**Answer:**
```sql
-- Simple DISTINCT
select distinct job from emp;

-- Using GROUP BY
select job from emp group by job;

-- DB2, Oracle, SQL Server (ROW_NUMBER — useful in subqueries)
select job
from (
  select job,
         row_number() over (partition by job order by job) as rn
  from emp
)
where rn = 1;
```

**Sample Output:**

| JOB       |
|-----------|
| ANALYST   |
| CLERK     |
| MANAGER   |
| PRESIDENT |
| SALESMAN  |

---

## Recipe 11.12 — Generating Simple Forecasts

**Question:**  
You have orders with a process date 2 days after the order date. For each order, generate 3 rows: the original, plus a verification row (process + 1 day) and a shipment row (process + 2 days).

**Answer:**
```sql
-- Oracle (using CONNECT BY to multiply rows)
select id,
       order_date,
       process_date,
       case iter.n
         when 1 then 'Ordered'
         when 2 then 'Verified'
         when 3 then 'Shipped'
       end as status,
       case iter.n
         when 1 then order_date
         when 2 then process_date + 1
         when 3 then process_date + 2
       end as status_date
from orders, (select level as n from dual connect by level <= 3) iter
order by id, n;
```

**Sample Input:**

| ID | ORDER_DATE  | PROCESS_DATE |
|----|-------------|--------------|
| 1  | 25-SEP-2005 | 27-SEP-2005  |
| 2  | 26-SEP-2005 | 28-SEP-2005  |

**Sample Output:**

| ID | ORDER_DATE  | STATUS   | STATUS_DATE |
|----|-------------|----------|-------------|
| 1  | 25-SEP-2005 | Ordered  | 25-SEP-2005 |
| 1  | 25-SEP-2005 | Verified | 28-SEP-2005 |
| 1  | 25-SEP-2005 | Shipped  | 29-SEP-2005 |
| 2  | 26-SEP-2005 | Ordered  | 26-SEP-2005 |
| 2  | 26-SEP-2005 | Verified | 29-SEP-2005 |
| 2  | 26-SEP-2005 | Shipped  | 30-SEP-2005 |

---

# Chapter 12 — Reporting and Warehousing (cont.)

---

## Recipe 12.2 — Pivoting a Result Set into Multiple Rows

**Question:**  
You want to pivot job types into columns, with one employee name per row — resulting in multiple rows when a job has multiple employees.

**Answer:**
```sql
select max(case when job='CLERK'     then ename else null end) as clerks,
       max(case when job='ANALYST'   then ename else null end) as analysts,
       max(case when job='MANAGER'   then ename else null end) as managers,
       max(case when job='SALESMAN'  then ename else null end) as salesmen,
       max(case when job='PRESIDENT' then ename else null end) as presidents
from (
  select job, ename,
         row_number() over (partition by job order by empno) as rn
  from emp
)
group by rn
order by rn;
```

**Sample Output:**

| clerks | analysts | managers | salesmen | presidents |
|--------|----------|----------|----------|------------|
| SMITH  | SCOTT    | JONES    | ALLEN    | KING       |
| ADAMS  | FORD     | BLAKE    | WARD     |            |
| JAMES  |          | CLARK    | MARTIN   |            |
| MILLER |          |          | TURNER   |            |

---

## Recipe 12.4 — Reverse Pivoting a Result Set into One Column

**Question:**  
You want to return every column from three employees as a single column, with blank rows separating each employee.

**Answer:**
```sql
-- Oracle / DB2
select case rn
         when 1 then ename
         when 2 then job
         when 3 then cast(sal as varchar(10))
         when 4 then ' '
       end as emps
from (
  select ename, job, sal,
         row_number() over (order by empno) as emp_rn
  from emp where deptno = 10
), (select level as rn from dual connect by level <= 4)
order by emp_rn, rn;
```

**Sample Output:**

| emps      |
|-----------|
| CLARK     |
| MANAGER   |
| 2450      |
|           |
| KING      |
| PRESIDENT |
| 5000      |
|           |
| MILLER    |
| CLERK     |
| 1300      |
|           |

---

## Recipe 12.5 — Suppressing Repeating Values from a Result Set

**Question:**  
You want to display DEPTNO and ENAME, but show the department number only on the first row for each department.

**Answer:**
```sql
-- Oracle / DB2 (using LAG to detect change)
select case when lag(deptno) over (order by deptno) = deptno
            then null
            else deptno
       end as deptno,
       ename
from emp
order by deptno;
```

**Sample Output:**

| DEPTNO | ENAME  |
|--------|--------|
| 10     | CLARK  |
|        | KING   |
|        | MILLER |
| 20     | SMITH  |
|        | ADAMS  |
| 30     | ALLEN  |
|        | BLAKE  |

---

## Recipe 12.6 — Pivoting to Facilitate Inter-Row Calculations

**Question:**  
You want to compare department salaries side by side — show dept 10, 20, and 30 salary totals in one row and compute the differences between them.

**Answer:**
```sql
select d20_sal - d10_sal as d20_vs_d10,
       d20_sal - d30_sal as d20_vs_d30
from (
  select sum(case when deptno=10 then sal end) as d10_sal,
         sum(case when deptno=20 then sal end) as d20_sal,
         sum(case when deptno=30 then sal end) as d30_sal
  from emp
) x;
```

**Sample Output:**

| d20_vs_d10 | d20_vs_d30 |
|------------|------------|
| 2125       | 1475       |

---

## Recipe 12.7 — Creating Buckets of Data of a Fixed Size

**Question:**  
You want to divide EMP rows into buckets of 5 employees each.

**Answer:**
```sql
select ceil(row_number() over (order by empno) / 5.0) as grp,
       empno, ename
from emp;
```

**Sample Output:**

| GRP | EMPNO | ENAME  |
|-----|-------|--------|
| 1   | 7369  | SMITH  |
| 1   | 7499  | ALLEN  |
| 1   | 7521  | WARD   |
| 1   | 7566  | JONES  |
| 1   | 7654  | MARTIN |
| 2   | 7698  | BLAKE  |
| 2   | 7782  | CLARK  |
| 2   | 7788  | SCOTT  |
| 2   | 7839  | KING   |
| 2   | 7844  | TURNER |
| 3   | 7876  | ADAMS  |
| 3   | 7900  | JAMES  |
| 3   | 7902  | FORD   |
| 3   | 7934  | MILLER |

---

## Recipe 12.8 — Creating a Predefined Number of Buckets

**Question:**  
You want to divide EMP into exactly 4 equally sized buckets (using NTILE).

**Answer:**
```sql
select ntile(4) over (order by empno) as grp,
       empno, ename, sal
from emp;
```

**Sample Output:**

| GRP | EMPNO | ENAME  | SAL  |
|-----|-------|--------|------|
| 1   | 7369  | SMITH  | 800  |
| 1   | 7499  | ALLEN  | 1600 |
| 1   | 7521  | WARD   | 1250 |
| 1   | 7566  | JONES  | 2975 |
| 2   | 7654  | MARTIN | 1250 |
| 2   | 7698  | BLAKE  | 2850 |
| 2   | 7782  | CLARK  | 2450 |
| 3   | 7788  | SCOTT  | 3000 |
| 3   | 7839  | KING   | 5000 |
| 3   | 7844  | TURNER | 1500 |
| 4   | 7876  | ADAMS  | 1100 |
| 4   | 7900  | JAMES  | 950  |
| 4   | 7902  | FORD   | 3000 |
| 4   | 7934  | MILLER | 1300 |

---

## Recipe 12.10 — Creating Vertical Histograms

**Question:**  
You want to create a vertical histogram showing employee counts per department, stacked upward with asterisks.

**Answer:**
```sql
select max(case when deptno=10 then rpad('*',cnt,'*') else ' ' end) as d10,
       max(case when deptno=20 then rpad('*',cnt,'*') else ' ' end) as d20,
       max(case when deptno=30 then rpad('*',cnt,'*') else ' ' end) as d30
from (
  select deptno, count(*) as cnt,
         row_number() over (partition by deptno order by deptno) as rn
  from emp
  group by deptno
);
```

**Sample Output (conceptual vertical representation):**

```
D10  D20  D30
 *    *    *
 *    *    *
 *    *    *
      *    *
      *    *
           *
```

---

## Recipe 12.11 — Returning Non-GROUP BY Columns

**Question:**  
You want the name of the employee with the highest salary in each department — a non-aggregated column — alongside the grouped MAX(SAL).

**Answer:**
```sql
-- Using a window function (cleanest)
select deptno, ename, sal
from (
  select deptno, ename, sal,
         max(sal) over (partition by deptno) as max_sal
  from emp
)
where sal = max_sal;

-- Using a correlated subquery
select deptno, ename, sal
from emp e
where sal = (select max(sal) from emp where deptno = e.deptno);
```

**Sample Output:**

| DEPTNO | ENAME | SAL  |
|--------|-------|------|
| 10     | KING  | 5000 |
| 20     | SCOTT | 3000 |
| 20     | FORD  | 3000 |
| 30     | BLAKE | 2850 |

---

## Recipe 12.13 — Subtotals for All Expression Combinations (CUBE)

**Question:**  
You want subtotals for every possible combination of DEPTNO and JOB groupings — all combinations, not just per-department rollups.

**Answer:**
```sql
-- DB2, Oracle, SQL Server (CUBE)
select deptno, job, sum(sal) as total_sal
from emp
group by cube(deptno, job)
order by deptno, job;
```

**Sample Output (partial):**

| DEPTNO | JOB       | total_sal |
|--------|-----------|-----------|
|        |           | 29025     |
|        | ANALYST   | 6000      |
|        | CLERK     | 4150      |
|        | MANAGER   | 8275      |
|        | PRESIDENT | 5000      |
|        | SALESMAN  | 5600      |
| 10     |           | 8750      |
| 10     | CLERK     | 1300      |
| 10     | MANAGER   | 2450      |
| 10     | PRESIDENT | 5000      |
| 20     |           | 10875     |
| ...    | ...       | ...       |

---

## Recipe 12.14 — Identifying Rows That Are Not Subtotals

**Question:**  
After using ROLLUP or CUBE, you want to flag which rows are subtotals vs. detail rows so you can format them differently.

**Answer:**
```sql
-- Oracle (GROUPING function returns 1 for subtotal rows, 0 for detail)
select deptno, job, sum(sal) as total_sal,
       grouping(deptno) as is_deptno_subtotal,
       grouping(job)    as is_job_subtotal
from emp
group by rollup(deptno, job);
```

**Sample Output:**

| DEPTNO | JOB     | total_sal | is_deptno_subtotal | is_job_subtotal |
|--------|---------|-----------|--------------------|-----------------|
| 10     | CLERK   | 1300      | 0                  | 0               |
| 10     | MANAGER | 2450      | 0                  | 0               |
| 10     |         | 8750      | 0                  | 1               |
|        |         | 29025     | 1                  | 1               |

---

## Recipe 12.15 — Using CASE Expressions to Flag Rows

**Question:**  
You want to mark each employee row to indicate whether they are in department 10, 20, or 30 using binary flags.

**Answer:**
```sql
select ename,
       case when deptno = 10 then 1 else 0 end as is_dept10,
       case when deptno = 20 then 1 else 0 end as is_dept20,
       case when deptno = 30 then 1 else 0 end as is_dept30
from emp;
```

**Sample Output:**

| ENAME  | is_dept10 | is_dept20 | is_dept30 |
|--------|-----------|-----------|-----------|
| SMITH  | 0         | 1         | 0         |
| ALLEN  | 0         | 0         | 1         |
| CLARK  | 1         | 0         | 0         |
| KING   | 1         | 0         | 0         |

---

## Recipe 12.16 — Creating a Sparse Matrix

**Question:**  
You want a matrix showing the count of employees per department per job, with NULLs where no employees exist in that combination.

**Answer:**
```sql
select deptno,
       sum(case when job='CLERK'     then 1 end) as clerks,
       sum(case when job='ANALYST'   then 1 end) as analysts,
       sum(case when job='MANAGER'   then 1 end) as managers,
       sum(case when job='SALESMAN'  then 1 end) as salesmen,
       sum(case when job='PRESIDENT' then 1 end) as presidents
from emp
group by deptno
order by deptno;
```

**Sample Output:**

| DEPTNO | clerks | analysts | managers | salesmen | presidents |
|--------|--------|----------|----------|----------|------------|
| 10     | 1      |          | 1        |          | 1          |
| 20     | 2      | 2        | 1        |          |            |
| 30     | 1      |          | 1        | 4        |            |

---

## Recipe 12.17 — Grouping Rows by Units of Time

**Question:**  
You want to count how many employees were hired per 5-year period (e.g., 1980–1984, 1985–1989).

**Answer:**
```sql
select period, count(*) as cnt
from (
  select case
    when hiredate between '01-JAN-1980' and '31-DEC-1984' then '1980-1984'
    when hiredate between '01-JAN-1985' and '31-DEC-1989' then '1985-1989'
    when hiredate between '01-JAN-1990' and '31-DEC-1994' then '1990-1994'
  end as period
  from emp
)
group by period;
```

**Sample Output:**

| period    | cnt |
|-----------|-----|
| 1980-1984 | 14  |

---

## Recipe 12.19 — Aggregations over a Moving Range of Values

**Question:**  
You want a 90-day moving average of salary, where for each row you average only the salaries of employees hired within the previous 90 days.

**Answer:**
```sql
-- Oracle / DB2 (using RANGE window clause)
select hiredate, sal,
       avg(sal) over (
         order by hiredate
         range between interval '90' day preceding and current row
       ) as moving_avg
from emp
order by hiredate;
```

**Sample Output:**

| HIREDATE    | SAL  | moving_avg |
|-------------|------|------------|
| 17-DEC-1980 | 800  | 800.00     |
| 20-FEB-1981 | 1600 | 1200.00    |
| 22-FEB-1981 | 1250 | 1216.67    |
| 02-APR-1981 | 2975 | 1941.67    |
| 01-MAY-1981 | 2850 | 2358.33    |

---

## Recipe 12.20 — Pivoting a Result Set with Subtotals

**Question:**  
You want to pivot job counts per department (like Recipe 12.1) but also include a TOTAL column and a TOTAL row using ROLLUP with pivoting.

**Answer:**
```sql
select deptno,
       sum(case when job='CLERK'     then 1 else 0 end) as clerks,
       sum(case when job='ANALYST'   then 1 else 0 end) as analysts,
       sum(case when job='MANAGER'   then 1 else 0 end) as managers,
       sum(case when job='PRESIDENT' then 1 else 0 end) as presidents,
       sum(case when job='SALESMAN'  then 1 else 0 end) as salesmen,
       count(*) as total
from emp
group by rollup(deptno)
order by deptno;
```

**Sample Output:**

| DEPTNO | clerks | analysts | managers | presidents | salesmen | total |
|--------|--------|----------|----------|------------|----------|-------|
| 10     | 1      | 0        | 1        | 1          | 0        | 3     |
| 20     | 2      | 2        | 1        | 0          | 0        | 5     |
| 30     | 1      | 0        | 1        | 0          | 4        | 6     |
|        | 4      | 2        | 3        | 1          | 4        | 14    |

---

# Appendix B — Rozenshtein Revisited (cont.)

---

## B.1 — Rozenshtein's Example Tables

**Question:**  
What are the example tables used in Appendix B for demonstrating relational division?

**Answer:**
```sql
-- Students and their enrolled courses
select * from students;
```

| STUDENT | COURSE |
|---------|--------|
| Rudy    | SQL    |
| Rudy    | Java   |
| Amy     | SQL    |
| Amy     | Java   |
| Amy     | Python |
| Bob     | SQL    |

```sql
-- Required courses
select * from required_courses;
```

| COURSE |
|--------|
| SQL    |
| Java   |

---

## B.3 — Answering Questions Involving "At Most"

**Question:**  
Return departments that have **at most** 3 clerks (i.e., fewer than or equal to 3).

**Answer:**
```sql
select deptno
from emp
group by deptno
having sum(case when job = 'CLERK' then 1 else 0 end) <= 3;
```

**Sample Output:**

| DEPTNO |
|--------|
| 10     |
| 20     |
| 30     |

> All departments qualify here since none has more than 3 clerks in the standard data.

---

## B.6 — Answering Questions Involving "Any" or "All"

**Question:**  
Return employees whose salary is greater than the salary of **any** employee in department 30 (i.e., greater than the minimum salary in dept 30).

**Answer:**
```sql
-- "ANY" — greater than at least one (equivalent to > MIN)
select ename, sal, deptno
from emp
where sal > any (select sal from emp where deptno = 30);

-- Equivalent without ANY:
select ename, sal, deptno
from emp
where sal > (select min(sal) from emp where deptno = 30);
```

**Sample Output:**

| ENAME  | SAL  | DEPTNO |
|--------|------|--------|
| ALLEN  | 1600 | 30     |
| JONES  | 2975 | 20     |
| BLAKE  | 2850 | 30     |
| CLARK  | 2450 | 10     |
| SCOTT  | 3000 | 20     |
| KING   | 5000 | 10     |
| TURNER | 1500 | 30     |
| ADAMS  | 1100 | 20     |
| FORD   | 3000 | 20     |
| MILLER | 1300 | 10     |

```sql
-- "ALL" — greater than every value (equivalent to > MAX)
select ename, sal, deptno
from emp
where sal > all (select sal from emp where deptno = 30);
```

**Sample Output:**

| ENAME | SAL  | DEPTNO |
|-------|------|--------|
| KING  | 5000 | 10     |
| SCOTT | 3000 | 20     |
| FORD  | 3000 | 20     |

---

*End of Supplement — all remaining recipes from the SQL Cookbook TOC*
