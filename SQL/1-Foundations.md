# SQL for ML Interviews — Complete Fundamentals Tutorial
---

## Table of Contents

1. [SELECT, WHERE, ORDER BY, LIMIT](#1-select-where-order-by-limit)
2. [Aggregations — COUNT, SUM, AVG, GROUP BY](#2-aggregations--count-sum-avg-group-by)
3. [JOINs — INNER, LEFT, RIGHT, FULL, CROSS, SELF](#3-joins--inner-left-right-full-cross-self)
4. [Anti-Joins — LEFT JOIN ... WHERE NULL pattern](#4-anti-joins--left-join--where-null-pattern)
5. [EXISTS/NOT EXISTS vs IN/NOT IN](#5-existsnot-exists-vs-innot-in)
6. [Subqueries vs CTEs](#6-subqueries-vs-ctes)
7. [Recursive CTEs](#7-recursive-ctes)
8. [CASE WHEN](#8-case-when)
9. [NULL Behavior — IS NULL, COALESCE, NULLIF](#9-null-behavior--is-null-coalesce-nullif)
10. [Safe Division / Divide-by-Zero Handling](#10-safe-division--divide-by-zero-handling)
11. [WHERE vs HAVING](#11-where-vs-having)
12. [COUNT Variations — COUNT(*), COUNT(DISTINCT), COUNT(col)](#12-count-variations)
13. [Set Operations — UNION, INTERSECT, EXCEPT](#13-set-operations--union-intersect-except)

---

## 1. SELECT, WHERE, ORDER BY, LIMIT

### Concept

`SELECT` is the entry point of every query. The clauses execute in this **logical order** (not the order you write them):

```
FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

Understanding this order prevents most beginner mistakes (e.g., why you can't use a SELECT alias in WHERE).

---

### Sample Table: `model_runs`

| run_id | model_name | accuracy | dataset  | created_at |
|--------|------------|----------|----------|------------|
| 1      | XGBoost    | 0.91     | train    | 2024-01-10 |
| 2      | XGBoost    | 0.85     | test     | 2024-01-10 |
| 3      | RandomForest | 0.88   | train    | 2024-01-11 |
| 4      | RandomForest | 0.82   | test     | 2024-01-11 |
| 5      | LightGBM   | 0.94     | train    | 2024-01-12 |
| 6      | LightGBM   | 0.89     | test     | 2024-01-12 |

---

### Variant 1 — Basic SELECT with WHERE filter

**Goal:** Get all test-set runs with accuracy above 0.85.

```sql
SELECT
    run_id,
    model_name,
    accuracy
FROM model_runs
WHERE dataset = 'test'
  AND accuracy > 0.85;
```

**Output:**

| run_id | model_name | accuracy |
|--------|------------|----------|
| 6      | LightGBM   | 0.89     |

---

### Variant 2 — ORDER BY with LIMIT (Top-N pattern)

**Goal:** Find the top 3 models by accuracy across all runs.

```sql
SELECT
    run_id,
    model_name,
    accuracy,
    dataset
FROM model_runs
ORDER BY accuracy DESC
LIMIT 3;
```

**Output:**

| run_id | model_name | accuracy | dataset |
|--------|------------|----------|---------|
| 5      | LightGBM   | 0.94     | train   |
| 1      | XGBoost    | 0.91     | train   |
| 6      | LightGBM   | 0.89     | test    |

---

### Variant 3 — Computed column + aliasing

**Goal:** Show accuracy as a percentage and tag high performers.

```sql
SELECT
    model_name,
    dataset,
    ROUND(accuracy * 100, 1) AS accuracy_pct,
    accuracy > 0.90 AS is_high_performer   -- returns boolean in most DBs
FROM model_runs
ORDER BY accuracy DESC;
```

**Output:**

| model_name   | dataset | accuracy_pct | is_high_performer |
|--------------|---------|--------------|-------------------|
| LightGBM     | train   | 94.0         | true              |
| XGBoost      | train   | 91.0         | true              |
| LightGBM     | test    | 89.0         | false             |
| RandomForest | train   | 88.0         | false             |
| XGBoost      | test    | 85.0         | false             |
| RandomForest | test    | 82.0         | false             |

---

### ⚠️ Pitfalls

| Pitfall | Example | Fix |
|---------|---------|-----|
| Using SELECT alias in WHERE | `WHERE accuracy_pct > 90` (alias not yet defined) | Use the original expression: `WHERE accuracy * 100 > 90` |
| ORDER BY without LIMIT on large tables | Full sort of 100M rows | Always pair with LIMIT in production |
| LIMIT without ORDER BY | Results are non-deterministic | Always ORDER BY before LIMIT |
| `SELECT *` in production | Fetches unused columns, breaks on schema changes | Always name columns explicitly |

---

## 2. Aggregations — COUNT, SUM, AVG, GROUP BY

### Concept

Aggregation collapses many rows into one summary row **per group**. The key rule: every column in SELECT must either be in GROUP BY **or** be inside an aggregate function.

---

### Sample Table: `predictions`

| pred_id | model_name   | label | predicted | score |
|---------|--------------|-------|-----------|-------|
| 1       | XGBoost      | 1     | 1         | 0.92  |
| 2       | XGBoost      | 0     | 1         | 0.78  |
| 3       | XGBoost      | 1     | 0         | 0.41  |
| 4       | RandomForest | 1     | 1         | 0.88  |
| 5       | RandomForest | 0     | 0         | 0.21  |
| 6       | RandomForest | 1     | 1         | 0.95  |
| 7       | LightGBM     | 0     | 0         | 0.11  |
| 8       | LightGBM     | 1     | 1         | 0.97  |

---

### Variant 1 — Per-model accuracy

**Goal:** Compute accuracy (correct predictions / total) per model.

```sql
SELECT
    model_name,
    COUNT(*)                                         AS total_preds,
    SUM(CASE WHEN label = predicted THEN 1 ELSE 0 END) AS correct,
    ROUND(
        AVG(CASE WHEN label = predicted THEN 1.0 ELSE 0 END),
        3
    )                                                AS accuracy
FROM predictions
GROUP BY model_name
ORDER BY accuracy DESC;
```

**Output:**

| model_name   | total_preds | correct | accuracy |
|--------------|-------------|---------|----------|
| LightGBM     | 2           | 2       | 1.000    |
| RandomForest | 3           | 3       | 1.000    |
| XGBoost      | 3           | 1       | 0.333    |

---

### Variant 2 — GROUP BY multiple columns (confusion matrix counts)

**Goal:** Get prediction counts by model, actual label, and predicted label.

```sql
SELECT
    model_name,
    label        AS actual,
    predicted,
    COUNT(*)     AS count
FROM predictions
GROUP BY model_name, label, predicted
ORDER BY model_name, actual, predicted;
```

**Output:**

| model_name   | actual | predicted | count |
|--------------|--------|-----------|-------|
| LightGBM     | 0      | 0         | 1     |
| LightGBM     | 1      | 1         | 1     |
| RandomForest | 0      | 0         | 1     |
| RandomForest | 1      | 1         | 2     |
| XGBoost      | 0      | 1         | 1     |
| XGBoost      | 1      | 0         | 1     |
| XGBoost      | 1      | 1         | 1     |

---

### Variant 3 — Average score by actual label

**Goal:** Understand if the model's score distribution separates classes.

```sql
SELECT
    label,
    COUNT(*)        AS n,
    ROUND(AVG(score), 3) AS avg_score,
    ROUND(MIN(score), 3) AS min_score,
    ROUND(MAX(score), 3) AS max_score
FROM predictions
GROUP BY label;
```

**Output:**

| label | n | avg_score | min_score | max_score |
|-------|---|-----------|-----------|-----------|
| 0     | 3 | 0.367     | 0.110     | 0.780     |
| 1     | 5 | 0.826     | 0.410     | 0.970     |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Non-aggregated column not in GROUP BY | `SELECT model_name, score FROM ... GROUP BY model_name` — `score` is ambiguous | Add `score` to GROUP BY or wrap in `MAX(score)` |
| AVG on integers returns integer in some DBs | `AVG(correct)` may truncate to 0 | Cast: `AVG(correct::FLOAT)` or `AVG(correct * 1.0)` |
| Filtering groups with WHERE instead of HAVING | `WHERE COUNT(*) > 2` is a syntax error | Use `HAVING COUNT(*) > 2` |
| GROUP BY ordinal (position number) | `GROUP BY 1` works but is fragile — breaks if columns reorder | Always name the column |

---

## 3. JOINs — INNER, LEFT, RIGHT, FULL, CROSS, SELF

### Concept

A JOIN combines rows from two tables based on a condition.

```
INNER JOIN  → Only rows with a match in BOTH tables
LEFT JOIN   → All rows from left table; NULLs where no right match
RIGHT JOIN  → All rows from right table; NULLs where no left match
FULL JOIN   → All rows from both; NULLs where no match on either side
CROSS JOIN  → Every row in left × every row in right (Cartesian product)
SELF JOIN   → Join a table to itself using aliases
```

---

### Sample Tables

**`models`**

| model_id | model_name   | team    |
|----------|--------------|---------|
| 1        | XGBoost      | Search  |
| 2        | RandomForest | Ads     |
| 3        | LightGBM     | Search  |
| 4        | BERT         | NLP     |

**`deployments`**

| deploy_id | model_id | env     | deployed_at |
|-----------|----------|---------|-------------|
| 101       | 1        | prod    | 2024-01-01  |
| 102       | 1        | staging | 2024-01-05  |
| 103       | 2        | prod    | 2024-01-03  |
| 104       | 5        | prod    | 2024-01-10  |  ← model_id 5 doesn't exist

---

### Variant 1 — INNER JOIN (only matched rows)

**Goal:** Get all deployments with their model name. Ignore orphan records on either side.

```sql
SELECT
    d.deploy_id,
    m.model_name,
    d.env,
    d.deployed_at
FROM deployments d
INNER JOIN models m ON d.model_id = m.model_id
ORDER BY d.deploy_id;
```

**Output:**

| deploy_id | model_name   | env     | deployed_at |
|-----------|--------------|---------|-------------|
| 101       | XGBoost      | prod    | 2024-01-01  |
| 102       | XGBoost      | staging | 2024-01-05  |
| 103       | RandomForest | prod    | 2024-01-03  |

> deploy_id 104 dropped (model_id 5 has no match). BERT (model_id 4) also dropped (no deployment).

---

### Variant 2 — LEFT JOIN (keep all models, show if deployed)

**Goal:** Show every model and whether it has been deployed.

```sql
SELECT
    m.model_name,
    m.team,
    d.env,
    d.deployed_at
FROM models m
LEFT JOIN deployments d ON m.model_id = d.model_id
ORDER BY m.model_id;
```

**Output:**

| model_name   | team   | env     | deployed_at |
|--------------|--------|---------|-------------|
| XGBoost      | Search | prod    | 2024-01-01  |
| XGBoost      | Search | staging | 2024-01-05  |
| RandomForest | Ads    | prod    | 2024-01-03  |
| LightGBM     | Search | NULL    | NULL        |
| BERT         | NLP    | NULL    | NULL        |

> LightGBM and BERT have no deployments — shown with NULLs.

---

### Variant 3 — SELF JOIN (compare runs within same model)

**Goal:** For each model, find pairs of runs where train accuracy > test accuracy.

```sql
-- Using model_runs from earlier
SELECT
    a.model_name,
    a.accuracy AS train_acc,
    b.accuracy AS test_acc,
    ROUND(a.accuracy - b.accuracy, 3) AS gap
FROM model_runs a
JOIN model_runs b
  ON a.model_name = b.model_name
  AND a.dataset = 'train'
  AND b.dataset = 'test'
ORDER BY gap DESC;
```

**Output:**

| model_name   | train_acc | test_acc | gap   |
|--------------|-----------|----------|-------|
| XGBoost      | 0.91      | 0.85     | 0.060 |
| RandomForest | 0.88      | 0.82     | 0.060 |
| LightGBM     | 0.94      | 0.89     | 0.050 |

> This is the classic **overfitting detection** query in ML interviews.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Forgetting INNER JOIN drops unmatched rows | Expected all users but got fewer | Use LEFT JOIN and check for NULLs |
| CROSS JOIN on large tables | 1000 × 1000 rows = 1M rows; often unintentional | Always add an ON condition unless you explicitly want Cartesian product |
| RIGHT JOIN confusion | Harder to read; equivalent to swapping table order with LEFT JOIN | Prefer LEFT JOIN consistently |
| Duplicate rows from one-to-many JOIN | Joining orders to order_items inflates revenue SUM | Always check cardinality before aggregating after a JOIN |
| JOIN on NULLs | `NULL = NULL` is false in SQL | Use `IS NOT DISTINCT FROM` or COALESCE the key |

---

## 4. Anti-Joins — LEFT JOIN ... WHERE NULL pattern

### Concept

An **anti-join** returns rows from the left table that have **no matching row** in the right table. This is one of the most common interview patterns.

Two equivalent approaches:
- `LEFT JOIN ... WHERE right.key IS NULL`
- `NOT EXISTS (...)`

---

### Sample Tables

**`all_users`** — everyone who signed up

| user_id | name    |
|---------|---------|
| 1       | Alice   |
| 2       | Bob     |
| 3       | Carol   |
| 4       | Dave    |

**`purchasers`** — users who made a purchase

| user_id | amount |
|---------|--------|
| 1       | 49.99  |
| 3       | 29.99  |

---

### Variant 1 — LEFT JOIN anti-join (find non-purchasers)

```sql
SELECT
    u.user_id,
    u.name
FROM all_users u
LEFT JOIN purchasers p ON u.user_id = p.user_id
WHERE p.user_id IS NULL;
```

**Output:**

| user_id | name  |
|---------|-------|
| 2       | Bob   |
| 4       | Dave  |

---

### Variant 2 — NOT EXISTS (semantically clearer)

```sql
SELECT
    u.user_id,
    u.name
FROM all_users u
WHERE NOT EXISTS (
    SELECT 1
    FROM purchasers p
    WHERE p.user_id = u.user_id
);
```

**Output:** Same as above.

> `SELECT 1` is conventional — it signals "we only care about existence, not the values."

---

### Variant 3 — Anti-join to find models never deployed to prod

```sql
SELECT
    m.model_name,
    m.team
FROM models m
LEFT JOIN deployments d
    ON m.model_id = d.model_id
    AND d.env = 'prod'          -- ← filter goes in ON, not WHERE
WHERE d.deploy_id IS NULL;
```

**Output:**

| model_name | team   |
|------------|--------|
| LightGBM   | Search |
| BERT       | NLP    |

> Notice: the `env = 'prod'` filter is in the **ON clause**, not WHERE. If it were in WHERE, we'd accidentally convert this back to an INNER JOIN.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Putting the filter in WHERE instead of ON | `WHERE d.env = 'prod'` eliminates NULLs, defeating the anti-join | Move env filter to the ON clause |
| `WHERE right.col IS NULL` on a non-key column | Could match NULL data values, not just "no match" | Always NULL-check the JOIN key column, not a data column |
| Assuming NOT IN is equivalent | NOT IN breaks with NULLs in the subquery (see Topic 5) | Prefer LEFT JOIN or NOT EXISTS |

---

## 5. EXISTS/NOT EXISTS vs IN/NOT IN

### Concept

Both filter rows based on a subquery, but they behave **very differently with NULLs**.

- `IN` checks if a value equals any value in a list. If the list contains a NULL, comparisons against NULL return UNKNOWN — not TRUE or FALSE.
- `EXISTS` checks if the subquery returns **at least one row**. NULLs in data don't affect the result.

---

### Sample Tables

**`experiments`**

| exp_id | model_name |
|--------|------------|
| 1      | XGBoost    |
| 2      | LightGBM   |
| 3      | BERT       |

**`approved_models`**

| model_name   |
|--------------|
| XGBoost      |
| LightGBM     |
| NULL         |   ← a bad data row

---

### Variant 1 — IN (works when no NULLs in list)

```sql
SELECT exp_id, model_name
FROM experiments
WHERE model_name IN (
    SELECT model_name FROM approved_models
);
```

**Output:**

| exp_id | model_name |
|--------|------------|
| 1      | XGBoost    |
| 2      | LightGBM   |

> Works correctly here because BERT simply doesn't match.

---

### Variant 2 — NOT IN BREAKS with NULLs ⚠️

```sql
-- Trying to find experiments NOT in the approved list
SELECT exp_id, model_name
FROM experiments
WHERE model_name NOT IN (
    SELECT model_name FROM approved_models
);
```

**Output: 0 rows** ← WRONG!

**Why:** The subquery returns `('XGBoost', 'LightGBM', NULL)`. SQL evaluates `'BERT' NOT IN (..., NULL)` as `BERT != NULL`, which is UNKNOWN, not TRUE. So BERT is also excluded. Every row is filtered out.

---

### Variant 3 — NOT EXISTS always safe

```sql
SELECT e.exp_id, e.model_name
FROM experiments e
WHERE NOT EXISTS (
    SELECT 1
    FROM approved_models a
    WHERE a.model_name = e.model_name
);
```

**Output:**

| exp_id | model_name |
|--------|------------|
| 3      | BERT       |

> Correct result. NOT EXISTS is immune to NULLs in the subquery.

---

### EXISTS vs IN — Performance

- `IN` with a **static list** (`WHERE id IN (1, 2, 3)`) is fast.
- `IN` with a **subquery** on large tables can be slow (may not short-circuit).
- `EXISTS` short-circuits as soon as it finds the first matching row — often faster for large correlated subqueries.
- Modern optimizers often rewrite them equivalently, but `EXISTS` is safer to write explicitly.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| NOT IN with NULLs in subquery | Returns 0 rows silently | Always use NOT EXISTS or `WHERE x NOT IN (...) OR x IS NOT NULL` |
| EXISTS returns TRUE/FALSE only | Can't use `SELECT col` inside EXISTS meaningfully | Use `SELECT 1` or `SELECT NULL` — column values are ignored |
| IN with 10,000+ values | Can exceed query limits or be slow | Use a JOIN or temp table instead |

---

## 6. Subqueries vs CTEs

### Concept

Both let you define an intermediate result set. A **CTE (Common Table Expression)** is written with `WITH` and is generally preferred for readability. A **subquery** is embedded inline.

```sql
-- Subquery style
SELECT * FROM (SELECT ...) AS sub

-- CTE style
WITH sub AS (
    SELECT ...
)
SELECT * FROM sub
```

CTEs can be referenced multiple times in the same query. Subqueries cannot (without repeating the SQL).

---

### Sample Table: `feature_importance`

| feature      | model_name | importance |
|--------------|------------|------------|
| age          | XGBoost    | 0.30       |
| income       | XGBoost    | 0.45       |
| credit_score | XGBoost    | 0.25       |
| age          | LightGBM   | 0.20       |
| income       | LightGBM   | 0.55       |
| credit_score | LightGBM   | 0.25       |

---

### Variant 1 — Subquery (inline)

**Goal:** Find features with above-average importance per model.

```sql
SELECT
    f.feature,
    f.model_name,
    f.importance
FROM feature_importance f
JOIN (
    SELECT model_name, AVG(importance) AS avg_imp
    FROM feature_importance
    GROUP BY model_name
) avg_by_model
    ON f.model_name = avg_by_model.model_name
WHERE f.importance > avg_by_model.avg_imp
ORDER BY f.model_name, f.importance DESC;
```

---

### Variant 2 — CTE (same logic, far more readable)

```sql
WITH avg_by_model AS (
    SELECT
        model_name,
        AVG(importance) AS avg_imp
    FROM feature_importance
    GROUP BY model_name
)
SELECT
    f.feature,
    f.model_name,
    f.importance
FROM feature_importance f
JOIN avg_by_model a ON f.model_name = a.model_name
WHERE f.importance > a.avg_imp
ORDER BY f.model_name, f.importance DESC;
```

**Output (both):**

| feature | model_name | importance |
|---------|------------|------------|
| income  | LightGBM   | 0.55       |
| income  | XGBoost    | 0.45       |

---

### Variant 3 — Multiple CTEs chained

**Goal:** Find models where the top feature alone explains more than 50% of importance.

```sql
WITH totals AS (
    SELECT model_name, SUM(importance) AS total_imp
    FROM feature_importance
    GROUP BY model_name
),
top_features AS (
    SELECT model_name, MAX(importance) AS top_imp
    FROM feature_importance
    GROUP BY model_name
)
SELECT
    t.model_name,
    ROUND(tf.top_imp / t.total_imp, 3) AS top_feature_share
FROM totals t
JOIN top_features tf ON t.model_name = tf.model_name
WHERE tf.top_imp / t.total_imp > 0.5;
```

**Output:**

| model_name | top_feature_share |
|------------|-------------------|
| LightGBM   | 0.550             |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Forgetting CTE alias in FROM | `WITH cte AS (...) SELECT * FROM feature_importance` — CTE is never used | Reference the CTE name in FROM |
| Reusing subquery logic | Copy-pasting same subquery in multiple places | Refactor to a CTE |
| CTEs are not always materialized | In PostgreSQL, CTEs are optimization fences (pre-v12); in newer versions they may be inlined | Use `MATERIALIZED` keyword explicitly if needed |
| Deeply nested subqueries | Becomes unreadable past 2 levels | Flatten with CTEs |

---

## 7. Recursive CTEs

### Concept

A **recursive CTE** calls itself to process hierarchical or graph data — org charts, folder trees, bill of materials, etc.

Structure:
```sql
WITH RECURSIVE cte AS (
    -- 1. Anchor: starting point (non-recursive)
    SELECT ...

    UNION ALL

    -- 2. Recursive: join cte back to itself to go one level deeper
    SELECT ...
    FROM cte
    JOIN ...
)
SELECT * FROM cte;
```

---

### Sample Table: `employees`

| emp_id | name    | manager_id |
|--------|---------|------------|
| 1      | CEO     | NULL       |
| 2      | VP Eng  | 1          |
| 3      | VP Data | 1          |
| 4      | Sr MLE  | 2          |
| 5      | MLE     | 4          |
| 6      | DS      | 3          |

---

### Variant 1 — Org chart: all reports under a VP

**Goal:** Find everyone who reports (directly or indirectly) to VP Eng (emp_id = 2).

```sql
WITH RECURSIVE reports AS (
    -- Anchor: start with VP Eng
    SELECT emp_id, name, manager_id, 0 AS depth
    FROM employees
    WHERE emp_id = 2

    UNION ALL

    -- Recursion: find direct reports of current level
    SELECT e.emp_id, e.name, e.manager_id, r.depth + 1
    FROM employees e
    JOIN reports r ON e.manager_id = r.emp_id
)
SELECT emp_id, name, depth
FROM reports
ORDER BY depth, emp_id;
```

**Output:**

| emp_id | name   | depth |
|--------|--------|-------|
| 2      | VP Eng | 0     |
| 4      | Sr MLE | 1     |
| 5      | MLE    | 2     |

---

### Variant 2 — Full hierarchy with path string

```sql
WITH RECURSIVE org AS (
    SELECT emp_id, name, manager_id,
           name AS path
    FROM employees
    WHERE manager_id IS NULL   -- start from root (CEO)

    UNION ALL

    SELECT e.emp_id, e.name, e.manager_id,
           o.path || ' > ' || e.name
    FROM employees e
    JOIN org o ON e.manager_id = o.emp_id
)
SELECT emp_id, name, path
FROM org
ORDER BY path;
```

**Output:**

| emp_id | name    | path                        |
|--------|---------|-----------------------------|
| 1      | CEO     | CEO                         |
| 3      | VP Data | CEO > VP Data               |
| 6      | DS      | CEO > VP Data > DS          |
| 2      | VP Eng  | CEO > VP Eng                |
| 4      | Sr MLE  | CEO > VP Eng > Sr MLE       |
| 5      | MLE     | CEO > VP Eng > Sr MLE > MLE |

---

### Variant 3 — Detect depth (level in hierarchy)

```sql
WITH RECURSIVE levels AS (
    SELECT emp_id, name, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    SELECT e.emp_id, e.name, l.level + 1
    FROM employees e
    JOIN levels l ON e.manager_id = l.emp_id
)
SELECT name, level FROM levels ORDER BY level, name;
```

**Output:**

| name    | level |
|---------|-------|
| CEO     | 1     |
| VP Data | 2     |
| VP Eng  | 2     |
| DS      | 3     |
| Sr MLE  | 3     |
| MLE     | 4     |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Infinite loop (circular reference) | If A reports to B and B reports to A, recursion never ends | Add `WHERE depth < 100` or use cycle detection (`CYCLE` clause in PG 14+) |
| Using UNION instead of UNION ALL | UNION deduplicates every iteration — extremely slow | Always use UNION ALL in recursive CTEs |
| Anchor selects wrong root | Starting with wrong emp causes partial tree | Verify anchor with a simple SELECT first |
| Not all DBs support RECURSIVE | MySQL < 8.0, SQLite < 3.8.3 don't support it | Check DB version |

---

## 8. CASE WHEN

### Concept

`CASE WHEN` is SQL's if-else. It can appear in SELECT, WHERE, ORDER BY, and GROUP BY.

```sql
CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ELSE default_result
END
```

---

### Sample Table: `model_scores`

| model_name | dataset | accuracy | f1_score |
|------------|---------|----------|----------|
| XGBoost    | test    | 0.91     | 0.89     |
| LightGBM   | test    | 0.78     | 0.74     |
| BERT       | test    | 0.65     | 0.60     |
| ResNet     | test    | 0.55     | 0.50     |

---

### Variant 1 — Classify performance tier

```sql
SELECT
    model_name,
    accuracy,
    CASE
        WHEN accuracy >= 0.90 THEN 'Excellent'
        WHEN accuracy >= 0.75 THEN 'Good'
        WHEN accuracy >= 0.60 THEN 'Fair'
        ELSE 'Poor'
    END AS performance_tier
FROM model_scores
ORDER BY accuracy DESC;
```

**Output:**

| model_name | accuracy | performance_tier |
|------------|----------|------------------|
| XGBoost    | 0.91     | Excellent         |
| LightGBM   | 0.78     | Good              |
| BERT       | 0.65     | Fair              |
| ResNet     | 0.55     | Poor              |

---

### Variant 2 — Conditional aggregation (pivot-like)

**Goal:** Count models in each tier without a separate GROUP BY query.

```sql
SELECT
    COUNT(CASE WHEN accuracy >= 0.90 THEN 1 END) AS excellent_count,
    COUNT(CASE WHEN accuracy >= 0.75 AND accuracy < 0.90 THEN 1 END) AS good_count,
    COUNT(CASE WHEN accuracy >= 0.60 AND accuracy < 0.75 THEN 1 END) AS fair_count,
    COUNT(CASE WHEN accuracy < 0.60 THEN 1 END) AS poor_count
FROM model_scores;
```

**Output:**

| excellent_count | good_count | fair_count | poor_count |
|-----------------|------------|------------|------------|
| 1               | 1          | 1          | 1          |

---

### Variant 3 — CASE in ORDER BY (custom sort order)

**Goal:** Sort by tier importance, not alphabetically.

```sql
SELECT model_name, accuracy
FROM model_scores
ORDER BY
    CASE
        WHEN accuracy >= 0.90 THEN 1
        WHEN accuracy >= 0.75 THEN 2
        WHEN accuracy >= 0.60 THEN 3
        ELSE 4
    END;
```

**Output:**

| model_name | accuracy |
|------------|----------|
| XGBoost    | 0.91     |
| LightGBM   | 0.78     |
| BERT       | 0.65     |
| ResNet     | 0.55     |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Overlapping conditions | First matching WHEN wins — later conditions are never evaluated | Order conditions from most specific to most general |
| Forgetting ELSE | Returns NULL if no condition matches | Add `ELSE 'Unknown'` or `ELSE NULL` explicitly |
| Using CASE WHERE instead of CASE WHEN | Syntax error | It's always `CASE WHEN` |
| CASE in WHERE | Valid but verbose | Often `WHERE col BETWEEN x AND y` is cleaner |

---

## 9. NULL Behavior — IS NULL, COALESCE, NULLIF

### Concept

NULL means **unknown/missing** — not zero, not empty string. Any arithmetic or comparison with NULL returns NULL (i.e., UNKNOWN). This surprises most people.

```sql
NULL = NULL    → NULL (not TRUE!)
NULL != NULL   → NULL
1 + NULL       → NULL
```

Key functions:
- `IS NULL` / `IS NOT NULL` — the only correct way to check for NULL
- `COALESCE(a, b, c)` — returns the first non-NULL argument
- `NULLIF(a, b)` — returns NULL if a = b, else returns a (useful to avoid divide-by-zero)

---

### Sample Table: `model_metrics`

| model_name | accuracy | precision | recall |
|------------|----------|-----------|--------|
| XGBoost    | 0.91     | 0.88      | 0.85   |
| LightGBM   | 0.89     | NULL      | 0.90   |
| BERT       | NULL     | 0.72      | NULL   |
| ResNet     | NULL     | NULL      | NULL   |

---

### Variant 1 — IS NULL to find incomplete records

```sql
SELECT model_name
FROM model_metrics
WHERE accuracy IS NULL
   OR precision IS NULL
   OR recall IS NULL;
```

**Output:**

| model_name |
|------------|
| LightGBM   |
| BERT       |
| ResNet     |

---

### Variant 2 — COALESCE to fill defaults

```sql
SELECT
    model_name,
    COALESCE(accuracy, 0)   AS accuracy,
    COALESCE(precision, 0)  AS precision,
    COALESCE(recall, 0)     AS recall
FROM model_metrics;
```

**Output:**

| model_name | accuracy | precision | recall |
|------------|----------|-----------|--------|
| XGBoost    | 0.91     | 0.88      | 0.85   |
| LightGBM   | 0.89     | 0.00      | 0.90   |
| BERT       | 0.00     | 0.72      | 0.00   |
| ResNet     | 0.00     | 0.00      | 0.00   |

---

### Variant 3 — NULLIF to handle "unknown" placeholders

Sometimes missing data is encoded as -1 or 0, not NULL.

```sql
SELECT
    model_name,
    -- treat -1 as "no data"
    NULLIF(accuracy, -1)  AS clean_accuracy
FROM model_metrics;
```

Also common: avoid divide-by-zero using NULLIF:

```sql
SELECT
    model_name,
    2.0 * precision * recall / NULLIF(precision + recall, 0) AS f1_score
FROM model_metrics;
```

If both precision and recall are 0, NULLIF makes the denominator NULL → result is NULL instead of a crash.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `WHERE col = NULL` | Always returns 0 rows | Use `WHERE col IS NULL` |
| `COUNT(col)` ignores NULLs | `COUNT(accuracy)` won't count BERT or ResNet | Use `COUNT(*)` to count all rows regardless |
| `AVG` ignores NULLs | May silently misrepresent the mean | Use `AVG(COALESCE(accuracy, 0))` if 0 is the right fill |
| COALESCE with wrong type | `COALESCE(accuracy, 'N/A')` causes type error | Ensure all arguments are the same data type |
| ORDER BY puts NULLs at end (Postgres) or start (others) | Inconsistent across DBs | Use `ORDER BY col NULLS LAST` / `NULLS FIRST` |

---

## 10. Safe Division / Divide-by-Zero Handling

### Concept

In SQL, dividing by zero throws a **runtime error** (not NULL). You must guard against it explicitly. Two main approaches:

1. `NULLIF(denominator, 0)` — turns 0 into NULL, making division return NULL
2. `CASE WHEN denominator = 0 THEN NULL ELSE numerator / denominator END`

---

### Sample Table: `ab_test_results`

| variant | impressions | clicks | conversions |
|---------|-------------|--------|-------------|
| control | 10000       | 500    | 50          |
| test_A  | 8000        | 400    | 60          |
| test_B  | 0           | 0      | 0           |   ← zero impressions

---

### Variant 1 — Safe CTR (Click-Through Rate)

```sql
SELECT
    variant,
    clicks,
    impressions,
    ROUND(clicks * 1.0 / NULLIF(impressions, 0), 4) AS ctr
FROM ab_test_results;
```

**Output:**

| variant | clicks | impressions | ctr    |
|---------|--------|-------------|--------|
| control | 500    | 10000       | 0.0500 |
| test_A  | 400    | 8000        | 0.0500 |
| test_B  | 0      | 0           | NULL   |

> test_B returns NULL gracefully instead of crashing.

---

### Variant 2 — CASE WHEN approach (more explicit)

```sql
SELECT
    variant,
    CASE
        WHEN impressions = 0 THEN NULL
        ELSE ROUND(clicks * 1.0 / impressions, 4)
    END AS ctr
FROM ab_test_results;
```

**Output:** Same as above.

---

### Variant 3 — Lift calculation (conversion rate lift vs control)

```sql
WITH rates AS (
    SELECT
        variant,
        conversions * 1.0 / NULLIF(impressions, 0) AS conv_rate
    FROM ab_test_results
),
control_rate AS (
    SELECT conv_rate FROM rates WHERE variant = 'control'
)
SELECT
    r.variant,
    ROUND(r.conv_rate, 4) AS conv_rate,
    ROUND(
        (r.conv_rate - c.conv_rate) / NULLIF(c.conv_rate, 0) * 100,
        2
    ) AS lift_pct
FROM rates r
CROSS JOIN control_rate c;
```

**Output:**

| variant | conv_rate | lift_pct |
|---------|-----------|----------|
| control | 0.0050    | 0.00     |
| test_A  | 0.0075    | 50.00    |
| test_B  | NULL      | NULL     |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Integer division truncates | `5 / 2 = 2` in SQL (not 2.5) | Cast: `5 * 1.0 / 2` or `CAST(5 AS FLOAT) / 2` |
| Forgetting zero vs NULL denominator | NULLIF only catches zero, not NULL | NULLIF(COALESCE(denom, 0), 0) covers both |
| Lift of NULL control rate | `(x - NULL) / ...` → NULL silently | Validate control group exists before running |

---

## 11. WHERE vs HAVING

### Concept

Both filter rows, but at different stages of query execution:

| Clause | Filters | When it runs | Can use aggregates? |
|--------|---------|--------------|---------------------|
| WHERE  | Individual rows | Before GROUP BY | ❌ No |
| HAVING | Groups (after aggregation) | After GROUP BY | ✅ Yes |

---

### Sample Table: `daily_predictions`

| model_name | prediction_date | n_predictions | n_correct |
|------------|-----------------|---------------|-----------|
| XGBoost    | 2024-01-01      | 100           | 90        |
| XGBoost    | 2024-01-02      | 120           | 100       |
| LightGBM   | 2024-01-01      | 80            | 72        |
| LightGBM   | 2024-01-02      | 60            | 48        |
| BERT       | 2024-01-01      | 200           | 120       |

---

### Variant 1 — WHERE filters before grouping

**Goal:** Only consider days with more than 80 predictions, then compute accuracy.

```sql
SELECT
    model_name,
    SUM(n_predictions)  AS total_preds,
    SUM(n_correct)      AS total_correct,
    ROUND(SUM(n_correct) * 1.0 / SUM(n_predictions), 3) AS accuracy
FROM daily_predictions
WHERE n_predictions > 80        -- filters rows BEFORE grouping
GROUP BY model_name;
```

**Output:** (LightGBM's Jan 2 row dropped — only 60 predictions)

| model_name | total_preds | total_correct | accuracy |
|------------|-------------|---------------|----------|
| XGBoost    | 220         | 190           | 0.864    |
| LightGBM   | 80          | 72            | 0.900    |
| BERT       | 200         | 120           | 0.600    |

---

### Variant 2 — HAVING filters after grouping

**Goal:** Only show models with total predictions > 150.

```sql
SELECT
    model_name,
    SUM(n_predictions)  AS total_preds,
    ROUND(SUM(n_correct) * 1.0 / SUM(n_predictions), 3) AS accuracy
FROM daily_predictions
GROUP BY model_name
HAVING SUM(n_predictions) > 150;   -- filters GROUPS after aggregation
```

**Output:**

| model_name | total_preds | accuracy |
|------------|-------------|----------|
| XGBoost    | 220         | 0.864    |
| BERT       | 200         | 0.600    |

---

### Variant 3 — WHERE and HAVING together

**Goal:** Among days with 80+ predictions, show only models averaging > 0.80 accuracy.

```sql
SELECT
    model_name,
    SUM(n_predictions) AS total_preds,
    ROUND(SUM(n_correct) * 1.0 / SUM(n_predictions), 3) AS accuracy
FROM daily_predictions
WHERE n_predictions >= 80             -- row-level filter first
GROUP BY model_name
HAVING SUM(n_correct) * 1.0 / SUM(n_predictions) > 0.80;  -- group-level filter
```

**Output:**

| model_name | total_preds | accuracy |
|------------|-------------|----------|
| XGBoost    | 220         | 0.864    |
| LightGBM   | 80          | 0.900    |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `WHERE COUNT(*) > 5` | Syntax error — aggregates not allowed in WHERE | Move to `HAVING COUNT(*) > 5` |
| `HAVING model_name = 'XGBoost'` | Valid but inefficient | Filter non-aggregated columns in WHERE |
| Forgetting HAVING filters groups not rows | Expecting HAVING to remove individual rows | Use WHERE for row-level filters |

---

## 12. COUNT Variations

### Concept

| Function | Counts | Ignores NULLs? |
|----------|--------|----------------|
| `COUNT(*)` | All rows in the group | No — counts everything |
| `COUNT(col)` | Rows where col is NOT NULL | Yes |
| `COUNT(DISTINCT col)` | Unique non-NULL values in col | Yes |

---

### Sample Table: `events`

| event_id | user_id | session_id | event_type |
|----------|---------|------------|------------|
| 1        | 101     | S1         | click      |
| 2        | 101     | S1         | purchase   |
| 3        | 102     | S2         | click      |
| 4        | NULL    | S3         | view       |
| 5        | 103     | S2         | click      |
| 6        | 101     | S4         | click      |

---

### Variant 1 — COUNT(*) vs COUNT(col)

```sql
SELECT
    COUNT(*)            AS total_events,
    COUNT(user_id)      AS events_with_user,
    COUNT(session_id)   AS events_with_session
FROM events;
```

**Output:**

| total_events | events_with_user | events_with_session |
|--------------|-----------------|---------------------|
| 6            | 5               | 6                   |

> Row 4 has NULL user_id so `COUNT(user_id)` = 5, not 6.

---

### Variant 2 — COUNT(DISTINCT col) for unique users/sessions

```sql
SELECT
    COUNT(*)                    AS total_events,
    COUNT(DISTINCT user_id)     AS unique_users,
    COUNT(DISTINCT session_id)  AS unique_sessions
FROM events;
```

**Output:**

| total_events | unique_users | unique_sessions |
|--------------|-------------|-----------------|
| 6            | 3           | 4               |

> user_id 101 appears 3 times but counts as 1. NULL user_id is excluded from DISTINCT count.

---

### Variant 3 — Conditional COUNT (event funnel)

```sql
SELECT
    COUNT(*)                                            AS total_events,
    COUNT(CASE WHEN event_type = 'click' THEN 1 END)    AS clicks,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchases,
    ROUND(
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) * 1.0
        / NULLIF(COUNT(CASE WHEN event_type = 'click' THEN 1 END), 0),
        3
    ) AS click_to_purchase_rate
FROM events;
```

**Output:**

| total_events | clicks | purchases | click_to_purchase_rate |
|--------------|--------|-----------|------------------------|
| 6            | 4      | 1         | 0.250                  |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using COUNT(col) expecting to count NULLs | NULLs silently excluded | Use `COUNT(*)` or check for NULLs separately |
| COUNT(DISTINCT *) not valid | Syntax error | Count DISTINCT on a specific column or concatenate: `COUNT(DISTINCT col1 \|\| col2)` |
| COUNT DISTINCT across multiple columns | No native multi-column DISTINCT COUNT | Use a subquery: `SELECT COUNT(*) FROM (SELECT DISTINCT col1, col2 FROM t)` |
| Slow COUNT(DISTINCT) on large tables | Full scan required | Use HyperLogLog approximation (`APPROX_COUNT_DISTINCT` in BigQuery/Redshift) |

---

## 13. Set Operations — UNION, INTERSECT, EXCEPT

### Concept

Set operations combine results of two SELECT statements **vertically** (stacking rows).

| Operation | Returns |
|-----------|---------|
| `UNION` | All rows from both, **deduplicates** |
| `UNION ALL` | All rows from both, **keeps duplicates** (faster) |
| `INTERSECT` | Rows that appear in **both** results |
| `EXCEPT` | Rows in the **first** result but **not the second** |

**Rules:**
- Both queries must have the **same number of columns**
- Columns must have **compatible data types**
- Column names come from the **first** query

---

### Sample Tables

**`prod_models`**

| model_name | version |
|------------|---------|
| XGBoost    | v2      |
| LightGBM   | v1      |
| BERT       | v1      |

**`staging_models`**

| model_name | version |
|------------|---------|
| LightGBM   | v1      |
| ResNet     | v1      |
| BERT       | v2      |

---

### Variant 1 — UNION ALL (full combined list)

```sql
SELECT model_name, version, 'prod'    AS env FROM prod_models
UNION ALL
SELECT model_name, version, 'staging' AS env FROM staging_models;
```

**Output:**

| model_name | version | env     |
|------------|---------|---------|
| XGBoost    | v2      | prod    |
| LightGBM   | v1      | prod    |
| BERT       | v1      | prod    |
| LightGBM   | v1      | staging |
| ResNet     | v1      | staging |
| BERT       | v2      | staging |

---

### Variant 2 — INTERSECT (in both prod and staging, same version)

```sql
SELECT model_name, version FROM prod_models
INTERSECT
SELECT model_name, version FROM staging_models;
```

**Output:**

| model_name | version |
|------------|---------|
| LightGBM   | v1      |

> BERT appears in both but with different versions (v1 vs v2), so it's excluded.

---

### Variant 3 — EXCEPT (in prod but not in staging)

```sql
SELECT model_name, version FROM prod_models
EXCEPT
SELECT model_name, version FROM staging_models;
```

**Output:**

| model_name | version |
|------------|---------|
| XGBoost    | v2      |
| BERT       | v1      |

> These are in prod but don't have matching (name + version) rows in staging.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| UNION when you want UNION ALL | UNION deduplicates — slower and may drop valid duplicates | Use UNION ALL unless deduplication is intentional |
| Mismatched column counts | `SELECT a, b UNION SELECT c` → error | Match column count in both queries |
| Column types mismatched | `SELECT 1 UNION SELECT 'text'` → error or cast | Cast explicitly: `CAST(1 AS VARCHAR)` |
| EXCEPT order matters | `A EXCEPT B ≠ B EXCEPT A` | Always put the "source of truth" table first |
| INTERSECT vs JOIN confusion | INTERSECT checks full row equality; JOIN matches on keys | Use JOIN when you need to match on specific columns only |

---

## Quick Reference — Logical Query Execution Order

```
1. FROM          ← Which tables?
2. JOIN          ← Combine tables
3. WHERE         ← Filter rows
4. GROUP BY      ← Group remaining rows
5. HAVING        ← Filter groups
6. SELECT        ← Choose columns / compute expressions
7. DISTINCT      ← Remove duplicates
8. ORDER BY      ← Sort
9. LIMIT/OFFSET  ← Paginate
```

**Why this matters for interviews:**
- You cannot reference a SELECT alias in WHERE (alias not yet created)
- You cannot use aggregate functions in WHERE (aggregation hasn't happened yet)
- HAVING runs after GROUP BY, so aggregates are valid there

---

## Interview Cheat Sheet

| Scenario | Pattern |
|----------|---------|
| Find rows with no match | LEFT JOIN ... WHERE right.key IS NULL |
| Rows in A but not B | EXCEPT or NOT EXISTS |
| Safe division | `numerator / NULLIF(denominator, 0)` |
| Top N per group | ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) ← Topic 14+ |
| Filter on aggregate | HAVING |
| Replace NULLs | COALESCE(col, default) |
| Train vs test accuracy gap | SELF JOIN on same table |
| Hierarchies | Recursive CTE |
| Conditional counts | COUNT(CASE WHEN ... THEN 1 END) |
| Unique users | COUNT(DISTINCT user_id) |

---

*End of Fundamentals Module — Dates & Strings module continues from Topic 14.*
