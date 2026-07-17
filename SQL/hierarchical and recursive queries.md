
## What is a recursive CTE?

A recursive CTE is a `WITH` clause that references itself. It lets SQL traverse tree and graph structures — org charts, category trees, folder hierarchies, bill of materials — without knowing the depth in advance.

```sql
WITH RECURSIVE cte_name AS (
    -- Anchor: the starting point (non-recursive)
    SELECT ...

    UNION ALL

    -- Recursive member: joins cte back to itself to go one level deeper
    SELECT ... FROM source JOIN cte_name ON ...
)
SELECT * FROM cte_name;
```

The keyword is `RECURSIVE` in PostgreSQL, MySQL 8+, SQLite 3.35+, and BigQuery. SQL Server uses `WITH cte AS (...)` without the `RECURSIVE` keyword but the structure is identical.

---

## How execution works (the iteration model)

Understanding the engine's internal loop prevents most recursive CTE bugs.

```
Step 1 — Run the anchor query. Result = "working table" (iteration 0).
Step 2 — Run the recursive member using only the CURRENT working table as input.
Step 3 — Append new rows to the result set. Replace working table with just the new rows.
Step 4 — Repeat from Step 2 until the recursive member returns zero rows.
Step 5 — Return the full accumulated result set.
```

The recursive member only ever sees the **most recently added rows**, not the full accumulated result. This is why cycle detection matters — if a row can reach itself, Step 2 never produces zero rows and the query runs forever.

---

## Use case 1 — Org chart (top-down traversal)

### Table: `employees`

| id | name    | manager_id |
|----|---------|------------|
| 1  | CEO     | NULL       |
| 2  | VP Eng  | 1          |
| 3  | VP Sales| 1          |
| 4  | Eng Lead| 2          |
| 5  | Alice   | 4          |
| 6  | Bob     | 4          |
| 7  | Carol   | 3          |

### Goal: print the full org chart under a given manager, with depth and path

```sql
WITH RECURSIVE org AS (
    -- Anchor: start at the CEO (or any root node)
    SELECT id,
           name,
           manager_id,
           0            AS depth,
           name::TEXT   AS path
    FROM   employees
    WHERE  manager_id IS NULL      -- root node(s)

    UNION ALL

    -- Recursive: one level down per iteration
    SELECT e.id,
           e.name,
           e.manager_id,
           org.depth + 1,
           org.path || ' → ' || e.name
    FROM   employees e
    JOIN   org ON e.manager_id = org.id
)
SELECT depth,
       REPEAT('  ', depth) || name AS indented_name,
       path
FROM   org
ORDER  BY path;
```

### Output

```
depth | indented_name    | path
0     | CEO              | CEO
1     |   VP Eng         | CEO → VP Eng
1     |   VP Sales       | CEO → VP Sales
2     |     Eng Lead     | CEO → VP Eng → Eng Lead
3     |       Alice      | CEO → VP Eng → Eng Lead → Alice
3     |       Bob        | CEO → VP Eng → Eng Lead → Bob
2     |     Carol        | CEO → VP Sales → Carol
```

### Key techniques

`REPEAT('  ', depth)` indents by level — purely cosmetic but makes output readable.

`path` accumulates the full ancestry string. It serves two purposes:
1. Readable display of where each node sits in the tree.
2. Cycle detection proxy — if a node name appears twice in the path, there is a cycle.

### Why it fails

| Mistake | Result |
|---|---|
| Anchor selects the wrong root | Subtree is wrong or empty |
| `UNION` instead of `UNION ALL` | Deduplication removes rows with identical columns — silent data loss in symmetric trees |
| No termination condition and a cycle in data | Infinite loop; query runs until max recursion depth or timeout |
| Casting path to wrong type | Type mismatch error — cast explicitly on the anchor (`name::TEXT`) |

---

## Use case 2 — Bottom-up traversal (find all ancestors)

### Goal: given employee Alice (id=5), find all managers up to the CEO

```sql
WITH RECURSIVE ancestry AS (
    -- Anchor: start at the target employee
    SELECT id, name, manager_id, 0 AS depth
    FROM   employees
    WHERE  id = 5        -- Alice

    UNION ALL

    -- Recursive: walk up to the manager
    SELECT e.id, e.name, e.manager_id, a.depth - 1
    FROM   employees e
    JOIN   ancestry a ON e.id = a.manager_id
)
SELECT depth, name
FROM   ancestry
ORDER  BY depth;
```

### Output

```
depth | name
0     | Alice
-1    | Eng Lead
-2    | VP Eng
-3    | CEO
```

The depth goes negative because we are walking upward — each parent is one level above the anchor. This is useful for breadcrumb rendering or permission inheritance.

---

## Use case 3 — Category tree (e-commerce hierarchy)

### Table: `categories`

| id | name        | parent_id |
|----|-------------|-----------|
| 1  | Electronics | NULL      |
| 2  | Computers   | 1         |
| 3  | Phones      | 1         |
| 4  | Laptops     | 2         |
| 5  | Desktops    | 2         |
| 6  | Gaming      | 5         |

### Goal: full path for every category (for breadcrumb display or URL slugs)

```sql
WITH RECURSIVE cat_tree AS (
    SELECT id,
           name,
           parent_id,
           name::TEXT          AS full_path,
           ARRAY[id]           AS id_path,
           1                   AS depth
    FROM   categories
    WHERE  parent_id IS NULL

    UNION ALL

    SELECT c.id,
           c.name,
           c.parent_id,
           ct.full_path || ' > ' || c.name,
           ct.id_path || c.id,
           ct.depth + 1
    FROM   categories c
    JOIN   cat_tree ct ON c.parent_id = ct.id
)
SELECT id,
       depth,
       full_path,
       id_path
FROM   cat_tree
ORDER  BY id_path;
```

### Output

```
id | depth | full_path                             | id_path
1  | 1     | Electronics                           | {1}
2  | 2     | Electronics > Computers               | {1,2}
3  | 2     | Electronics > Phones                  | {1,3}
4  | 3     | Electronics > Computers > Laptops     | {1,2,4}
5  | 3     | Electronics > Computers > Desktops    | {1,2,5}
6  | 4     | Electronics > Computers > Desktops > Gaming | {1,2,5,6}
```

`ARRAY[id]` (PostgreSQL) builds an integer array of the ancestor chain. Ordering by `id_path` gives a depth-first tree order without needing a complex `ORDER BY`. The array can also be checked for cycles: if `c.id = ANY(ct.id_path)`, a cycle exists.

---

## Use case 4 — Bill of materials (multi-level composition)

### Table: `bom` (bill of materials)

| parent_part | child_part | quantity |
|-------------|------------|----------|
| Bike        | Frame      | 1        |
| Bike        | Wheel      | 2        |
| Bike        | Drivetrain | 1        |
| Drivetrain  | Chain      | 1        |
| Drivetrain  | Gears      | 1        |
| Wheel       | Rim        | 1        |
| Wheel       | Tire       | 1        |
| Wheel       | Spoke      | 32       |

### Goal: total quantity of each raw part needed to make one Bike

```sql
WITH RECURSIVE exploded AS (
    -- Anchor: direct children of the top-level assembly
    SELECT child_part,
           quantity           AS total_qty,
           ARRAY[parent_part] AS visited
    FROM   bom
    WHERE  parent_part = 'Bike'

    UNION ALL

    -- Recursive: expand sub-assemblies, multiplying quantities
    SELECT b.child_part,
           e.total_qty * b.quantity,
           e.visited || b.parent_part
    FROM   bom b
    JOIN   exploded e ON b.parent_part = e.child_part
    WHERE  b.parent_part != ALL(e.visited)   -- cycle guard
)
SELECT child_part,
       SUM(total_qty) AS total_needed
FROM   exploded
WHERE  child_part NOT IN (SELECT DISTINCT parent_part FROM bom)  -- leaf parts only
GROUP  BY child_part
ORDER  BY child_part;
```

### Output

```
child_part | total_needed
Chain      | 1
Gears      | 1
Rim        | 2
Spoke      | 64
Tire       | 2
```

`total_qty * b.quantity` compounds the quantity down the tree — 2 Wheels × 32 Spokes = 64 Spokes total. The `WHERE child_part NOT IN (SELECT parent_part ...)` filter keeps only leaf nodes (parts that are not assemblies themselves).

---

## Cycle detection and prevention

Real-world data — especially graphs vs pure trees — can have cycles. An employee who is listed as their own manager, a category that is its own ancestor, a part that contains itself.

### Method 1 — Path-based string check (any database)

```sql
-- In the recursive member's WHERE clause:
WHERE  org.path NOT LIKE '%' || e.name || '%'
```

Fragile if names contain the separator or repeat naturally. Better for display than safety.

### Method 2 — Array membership check (PostgreSQL)

```sql
-- In the recursive member's WHERE clause:
WHERE  e.id != ALL(org.id_path)
```

Exact and fast. If the child's id is already in the ancestor id array, it is a cycle — skip it.

### Method 3 — Depth cap (any database)

```sql
WHERE  org.depth < 20
```

Crude but effective as a safety net. Always add this in production to prevent runaway queries on dirty data.

### Method 4 — `CYCLE` clause (PostgreSQL 14+, SQL:1999 standard)

```sql
WITH RECURSIVE org AS (
    SELECT id, name, manager_id FROM employees WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id
    FROM employees e JOIN org ON e.manager_id = org.id
)
CYCLE id SET is_cycle USING cycle_path
SELECT * FROM org WHERE NOT is_cycle;
```

`CYCLE id` tells the engine to track the `id` column for cycles. `is_cycle` is a boolean column the engine adds — filter it out of final results.

---

## Finding all descendants vs all ancestors

Two traversal directions, both common:

```sql
-- Top-down: all nodes UNDER a given node (descendants)
WHERE  anchor condition = specific_root

-- Bottom-up: all nodes ABOVE a given node (ancestors)
JOIN   cte ON source.id = cte.parent_id    ← walk up to parent
```

The join direction in the recursive member is what flips the traversal:

| Direction | Join condition |
|---|---|
| Top-down (descendants) | `JOIN cte ON child.parent_id = cte.id` |
| Bottom-up (ancestors) | `JOIN cte ON parent.id = cte.parent_id` |

---

## Finding the depth of each node / height of the tree

```sql
WITH RECURSIVE depth_calc AS (
    SELECT id, name, 0 AS depth FROM employees WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, d.depth + 1
    FROM   employees e JOIN depth_calc d ON e.manager_id = d.id
)
SELECT MAX(depth) AS tree_height,
       AVG(depth) AS avg_depth
FROM   depth_calc;
```

---

## Flattening for joins — connecting hierarchy to fact tables

A common pattern: compute the full subtree once in a CTE, then join it to a fact table.

```sql
-- Goal: total sales under VP Eng (id=2), including all their reports' reports
WITH RECURSIVE subtree AS (
    SELECT id FROM employees WHERE id = 2
    UNION ALL
    SELECT e.id FROM employees e JOIN subtree s ON e.manager_id = s.id
)
SELECT SUM(s.amount) AS total_sales
FROM   sales s
WHERE  s.rep_id IN (SELECT id FROM subtree);
```

The recursive CTE produces the full set of employee IDs in the subtree. The outer query treats it like a regular subquery.

---

## Database compatibility

| Feature | PostgreSQL | MySQL 8+ | SQL Server | BigQuery | SQLite 3.35+ |
|---|---|---|---|---|---|
| `WITH RECURSIVE` | Yes | Yes | `WITH` (no keyword) | Yes | Yes |
| `CYCLE` clause | 14+ | No | No | No | No |
| `ARRAY` path tracking | Yes | No | No | No | No |
| Max recursion depth | Default 100 (configurable) | Default 1000 | Default 100 (`MAXRECURSION`) | Unlimited | No limit |
| Override depth | `SET max_parallel_workers` | `max_sp_recursion_depth` | `OPTION (MAXRECURSION n)` | N/A | N/A |

SQL Server syntax note:

```sql
-- SQL Server: no RECURSIVE keyword, hint goes at the end
WITH org AS (
    SELECT id, name, manager_id, 0 AS depth FROM employees WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, o.depth + 1 FROM employees e JOIN org o ON e.manager_id = o.id
)
SELECT * FROM org
OPTION (MAXRECURSION 500);   -- safety cap
```

---

## Common failure modes summary

| Mistake | Result |
|---|---|
| `UNION` instead of `UNION ALL` | Silent row deduplication on symmetric trees |
| Anchor returns zero rows | Entire CTE returns empty |
| Recursive member references the full result, not the working table | Logic error — you can only reference the CTE name, which always means the working table |
| No cycle guard on graph data (non-tree) | Infinite loop |
| Forgetting to cast path string in anchor | Type mismatch error on string concatenation in recursive member |
| Querying a recursive CTE multiple times in the outer query | Some databases re-execute the recursion each time — materialize with a temp table or subquery |
| Depth cap too low on deep trees | Query silently truncates results mid-tree |

