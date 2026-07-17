# SQL for ML Interviews — Advanced Aggregation Patterns

> **Module 4** — Deduplication, Pivoting, Grouping Sets, and Array Aggregation. Same format: concept → variants with input/output → pitfalls. No other source needed.

---

## Table of Contents

26. [Deduplication](#26-deduplication)
27. [Pivot — CROSSTAB / Conditional Aggregation](#27-pivot--crosstab--conditional-aggregation)
28. [GROUPING SETS, ROLLUP, CUBE](#28-grouping-sets-rollup-cube)
29. [STRING_AGG / ARRAY_AGG and Unnesting](#29-string_agg--array_agg-and-unnesting)

---

## 26. Deduplication

### Concept

Duplicate rows are one of the most common data quality issues in ML pipelines — duplicate events inflate metrics, duplicate training rows bias models, duplicate users distort cohort analysis.

There are four main deduplication strategies:

| Strategy | When to use |
|----------|-------------|
| `SELECT DISTINCT` | All columns are identical — true exact duplicates |
| `ROW_NUMBER() = 1` | Keep one row per key, choose which one (newest, highest score, etc.) |
| `GROUP BY + MIN/MAX` | Collapse duplicates by keeping an aggregate of the non-key columns |
| `NOT EXISTS` / anti-join | Remove rows that already exist in another table |

**Key insight for interviews:** `SELECT DISTINCT` is blunt — it only works when every column is identical. In real data, duplicates usually have different timestamps or IDs, so `ROW_NUMBER()` is almost always what you actually need.

```sql
SELECT *
FROM (
    SELECT
        t.*,
        ROW_NUMBER() OVER (
            PARTITION BY user_id          -- the "duplicate key"
            ORDER BY updated_at DESC      -- the tie-break rule
        ) AS rn
    FROM events t
) ranked
WHERE rn = 1;```
hy not RANK() or DENSE_RANK()? Both can assign rank 1 to multiple rows when the ORDER BY column ties (e.g., two rows with the identical updated_at). If you use RANK() = 1 for dedup and there's a tie, you get both rows back — silently re-introducing the duplicate you were trying to remove. ROW_NUMBER() always breaks ties by physical row order, guaranteeing exactly one row per partition. This is the single most common interview trap.
---

### Sample Table: `raw_predictions` (messy, with duplicates)

| pred_id | model_name | user_id | score | created_at          |
|---------|------------|---------|-------|---------------------|
| 1       | XGBoost    | u1      | 0.91  | 2024-01-15 10:00:00 |
| 2       | XGBoost    | u1      | 0.91  | 2024-01-15 10:00:00 |  ← exact duplicate of row 1
| 3       | XGBoost    | u2      | 0.78  | 2024-01-15 10:01:00 |
| 4       | XGBoost    | u2      | 0.82  | 2024-01-15 10:05:00 |  ← same user, later update
| 5       | LightGBM   | u1      | 0.88  | 2024-01-15 10:00:00 |
| 6       | LightGBM   | u1      | 0.88  | 2024-01-15 10:00:00 |  ← exact duplicate of row 5

---

### Variant 1 — DISTINCT for exact duplicates

**Goal:** Remove rows where every column is identical.

```sql
SELECT DISTINCT
    model_name,
    user_id,
    score,
    created_at
FROM raw_predictions
ORDER BY model_name, user_id;
```

**Output:**

| model_name | user_id | score | created_at          |
|------------|---------|-------|---------------------|
| LightGBM   | u1      | 0.88  | 2024-01-15 10:00:00 |
| XGBoost    | u1      | 0.91  | 2024-01-15 10:00:00 |
| XGBoost    | u2      | 0.78  | 2024-01-15 10:01:00 |
| XGBoost    | u2      | 0.82  | 2024-01-15 10:05:00 |

> Rows 2 and 6 (exact duplicates) removed. Rows 3 and 4 (same user, different scores/timestamps) are both kept — DISTINCT can't distinguish "real updates" from "unwanted duplicates".

---

### Variant 2 — ROW_NUMBER() to keep the latest prediction per user per model

**Goal:** For each (model, user) pair, keep only the most recent prediction.

```sql
WITH ranked AS (
    SELECT
        pred_id,
        model_name,
        user_id,
        score,
        created_at,
        ROW_NUMBER() OVER (
            PARTITION BY model_name, user_id
            ORDER BY created_at DESC    -- latest first
        ) AS rn
    FROM raw_predictions
)
SELECT pred_id, model_name, user_id, score, created_at
FROM ranked
WHERE rn = 1
ORDER BY model_name, user_id;
```

**Output:**

| pred_id | model_name | user_id | score | created_at          |
|---------|------------|---------|-------|---------------------|
| 5       | LightGBM   | u1      | 0.88  | 2024-01-15 10:00:00 |
| 1       | XGBoost    | u1      | 0.91  | 2024-01-15 10:00:00 |
| 4       | XGBoost    | u2      | 0.82  | 2024-01-15 10:05:00 |

> For u2/XGBoost: row 4 (score 0.82, later time) was kept over row 3 (score 0.78, earlier).
> For LightGBM/u1: either row 5 or 6 is valid (identical) — ROW_NUMBER breaks the tie arbitrarily.

---

### Variant 3 — GROUP BY + aggregation to deduplicate while summarising

**Goal:** Deduplicate by collapsing to one row per (model, user), keeping max score and latest timestamp.

```sql
SELECT
    model_name,
    user_id,
    MAX(score)       AS best_score,
    MIN(score)       AS worst_score,
    MAX(created_at)  AS latest_prediction,
    COUNT(*)         AS raw_row_count
FROM raw_predictions
GROUP BY model_name, user_id
ORDER BY model_name, user_id;
```

**Output:**

| model_name | user_id | best_score | worst_score | latest_prediction   | raw_row_count |
|------------|---------|------------|-------------|---------------------|---------------|
| LightGBM   | u1      | 0.88       | 0.88        | 2024-01-15 10:00:00 | 2             |
| XGBoost    | u1      | 0.91       | 0.91        | 2024-01-15 10:00:00 | 2             |
| XGBoost    | u2      | 0.82       | 0.78        | 2024-01-15 10:05:00 | 2             |

> `raw_row_count = 2` for all — confirms duplicates existed. Good for auditing.

---

### Variant 4 — Dedup check: count duplicates before cleaning

**Goal:** Audit the table to find which (model, user) pairs have more than one row.

```sql
SELECT
    model_name,
    user_id,
    COUNT(*)                               AS row_count,
    COUNT(DISTINCT score)                  AS distinct_scores,
    COUNT(DISTINCT created_at)             AS distinct_timestamps
FROM raw_predictions
GROUP BY model_name, user_id
HAVING COUNT(*) > 1
ORDER BY row_count DESC;
```

**Output:**

| model_name | user_id | row_count | distinct_scores | distinct_timestamps |
|------------|---------|-----------|-----------------|---------------------|
| LightGBM   | u1      | 2         | 1               | 1                   |
| XGBoost    | u1      | 2         | 1               | 1                   |
| XGBoost    | u2      | 2         | 2               | 2                   |

> `distinct_scores = 1` and `distinct_timestamps = 1` → true exact duplicate (safe to drop either).
> `distinct_scores = 2` → real updates, not exact dupes — need business logic to decide which to keep.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| DISTINCT on a table with an ID column | If pred_id is different, DISTINCT keeps all rows even if everything else is identical | Drop the ID from DISTINCT, or use ROW_NUMBER |
| ROW_NUMBER tie-breaking is non-deterministic | Two rows with same created_at — which wins? | Add a tiebreaker: `ORDER BY created_at DESC, pred_id DESC` |
| Deduping training data vs serving data differently | Dedup logic for model training (keep all real labels) differs from serving (keep latest score) | Define dedup rules per use case explicitly |
| Dedup after a JOIN causes row explosion | Joining before deduping multiplies duplicates | Dedup each table first, then join |
| Using DISTINCT COUNT vs COUNT DISTINCT | `SELECT DISTINCT COUNT(*)` is wrong; `COUNT(DISTINCT col)` is right | Always `COUNT(DISTINCT col)` |
---

# SQL Deduplication

## 2. The Core Pattern: ROW_NUMBER() + PARTITION BY

```sql
SELECT *
FROM (
    SELECT
        t.*, --- or just *
        ROW_NUMBER() OVER (
            PARTITION BY user_id          -- the "duplicate key"
            ORDER BY updated_at DESC      -- the tie-break rule
        ) AS rn
    FROM events t
) ranked
WHERE rn = 1;
```

**Why not `RANK()` or `DENSE_RANK()`?** Both can assign rank 1 to multiple rows when the `ORDER BY` column ties (e.g., two rows with the identical `updated_at`). If you use `RANK() = 1` for dedup and there's a tie, you get *both* rows back — silently re-introducing the duplicate you were trying to remove. `ROW_NUMBER()` always breaks ties by physical row order, guaranteeing exactly one row per partition. This is the single most common interview trap.


Common trap: people write

```sql
SELECT user_id, MAX(updated_at) AS updated_at, status
FROM events
GROUP BY user_id
```

This is **invalid in strict SQL** (and misleading even where permitted, e.g. MySQL's non-strict mode) because `status` isn't functionally dependent on `user_id` in the `GROUP BY` — the engine picks an arbitrary `status` value, not necessarily the one that goes with the max `updated_at`. This is a classic L5 interview trap: candidates reach for `GROUP BY + MAX` when they actually need `ROW_NUMBER`.

---


```sql
SELECT user_id, COUNT(*) AS cnt
FROM events
GROUP BY user_id
HAVING COUNT(*) > 1;
```

To see the actual duplicate rows (not just counts):

```sql
SELECT *
FROM events e
WHERE user_id IN (
    SELECT user_id FROM events GROUP BY user_id HAVING COUNT(*) > 1
)
ORDER BY user_id, updated_at;
```


 Why is `SELECT customer_id, MAX(created_at), status FROM orders GROUP BY customer_id` problematic if you want "the status of the most recent order per customer"? What's the correct rewrite?
Common trap: people write

SELECT user_id, MAX(updated_at) AS updated_at, status
FROM events
GROUP BY user_id

This is invalid in strict SQL (and misleading even where permitted, e.g. MySQL's non-strict mode) because status isn't functionally dependent on user_id in the GROUP BY — the engine picks an arbitrary status value, not necessarily the one that goes with the max updated_at. This is a classic L5 interview trap: candidates reach for GROUP BY + MAX when they actually need ROW_NUMBER.

---

## 27. Pivot — CROSSTAB / Conditional Aggregation

### Concept

**Pivoting** transforms rows into columns — turning a long/narrow table into a wide table.

**Long format (normalized):**

| student | subject | score |
|---|---|---|
| Alice | Math | 90 |
| Alice | Science | 85 |
| Bob | Math | 78 |
| Bob | Science | 92 |

**Wide format (pivoted):**

| student | Math | Science |
|---|---|---|
| Alice | 90 | 85 |
| Bob | 78 | 92 |

Pivoting turns distinct **values** in one column into **separate columns**.

In SQL, there are two approaches:

| Approach | Syntax | Best for |
|----------|--------|----------|
| **Conditional aggregation** | `SUM(CASE WHEN col = 'x' THEN val END)` | Works in all SQL dialects; column names are hardcoded |
| **CROSSTAB** | `crosstab(...)` via `tablefunc` extension | PostgreSQL only; cleaner syntax |

**Unpivot** (wide → long) uses `UNION ALL` or `UNNEST`.

**When you'll see this in ML interviews:**
- Confusion matrix (actual vs predicted as rows → pivot to a matrix)
- Feature importance by model (model names as columns)
- A/B test results (metric per variant as columns)
- Time-series with dates as columns

---

### Sample Table: `model_scores_long` (long format)

| model_name   | dataset | accuracy |
|--------------|---------|----------|
| XGBoost      | train   | 0.960    |
| XGBoost      | val     | 0.930    |
| XGBoost      | test    | 0.910    |
| LightGBM     | train   | 0.975    |
| LightGBM     | val     | 0.945    |
| LightGBM     | test    | 0.940    |
| BERT         | train   | 0.940    |
| BERT         | val     | 0.905    |
| BERT         | test    | 0.890    |

---

### Variant 1 — Conditional aggregation pivot (works in all SQL)

**Goal:** Show train / val / test accuracy as separate columns per model.

```sql
SELECT
    model_name,
    MAX(CASE WHEN dataset = 'train' THEN accuracy END) AS train_acc,
    MAX(CASE WHEN dataset = 'val'   THEN accuracy END) AS val_acc,
    MAX(CASE WHEN dataset = 'test'  THEN accuracy END) AS test_acc,
    -- Bonus: generalisation gap
    ROUND(
        MAX(CASE WHEN dataset = 'train' THEN accuracy END) -
        MAX(CASE WHEN dataset = 'test'  THEN accuracy END),
        3
    )                                                  AS train_test_gap
FROM model_scores_long
GROUP BY model_name
ORDER BY test_acc DESC;
```

| model_name | train_acc | val_acc | test_acc | train_test_gap |
|------------|-----------|---------|----------|----------------|
| LightGBM   | 0.975     | 0.945   | 0.940    | 0.035          |
| XGBoost    | 0.960     | 0.930   | 0.910    | 0.050          |
| BERT       | 0.940     | 0.905   | 0.890    | 0.050          |

> `train_test_gap` reveals overfitting — LightGBM generalises best despite highest train accuracy.

```sql
SELECT
    student,
    SUM(CASE WHEN subject = 'Math'    THEN score END) AS math,
    SUM(CASE WHEN subject = 'Science' THEN score END) AS science
FROM   scores
GROUP  BY student;
```

#### How it works, step by step

1. `GROUP BY student` — one output row per student.
2. For each group, `SUM(CASE WHEN subject = 'Math' THEN score END)` walks every row in that group.
   - If `subject = 'Math'`, it contributes the `score` value.
   - Otherwise it contributes `NULL` (implicit `ELSE NULL`).
3. `SUM(NULL)` = `NULL` but `SUM(90, NULL, NULL)` = `90`.
4. Result: one column per subject, populated only where that condition is true.

#### Why `SUM` and not `MAX` or `MIN`?

| Aggregate | Use when |
|---|---|
| `SUM` | Scores, amounts — you want totals |
| `MAX` | You want the single value (when there's only one per group) |
| `MIN` | Same as MAX — choose based on semantics |
| `COUNT` | Counting occurrences |

If a student can only have one score per subject, `MAX` and `SUM` give the same result. `MAX` is often clearer semantically in that case ("give me the value, not the sum")
**Output:**

---

### Variant 2 — Confusion matrix pivot

**Goal:** Turn prediction results into a 2×2 confusion matrix.

```sql
-- Source: predictions table (label = actual, predicted = model output)
SELECT
    model_name,
    SUM(CASE WHEN label = 1 AND predicted = 1 THEN 1 ELSE 0 END) AS true_positive,
    SUM(CASE WHEN label = 0 AND predicted = 1 THEN 1 ELSE 0 END) AS false_positive,
    SUM(CASE WHEN label = 1 AND predicted = 0 THEN 1 ELSE 0 END) AS false_negative,
    SUM(CASE WHEN label = 0 AND predicted = 0 THEN 1 ELSE 0 END) AS true_negative,
    -- Derived metrics
    ROUND(
        SUM(CASE WHEN label = predicted THEN 1.0 ELSE 0 END) / COUNT(*),
        3
    )                                                             AS accuracy,
    ROUND(
        SUM(CASE WHEN label = 1 AND predicted = 1 THEN 1.0 END) /
        NULLIF(SUM(CASE WHEN predicted = 1 THEN 1 END), 0),
        3
    )                                                             AS precision,
    ROUND(
        SUM(CASE WHEN label = 1 AND predicted = 1 THEN 1.0 END) /
        NULLIF(SUM(CASE WHEN label = 1 THEN 1 END), 0),
        3
    )                                                             AS recall
FROM predictions
GROUP BY model_name;
```

**Sample Output:**

| model_name | true_positive | false_positive | false_negative | true_negative | accuracy | precision | recall |
|------------|---------------|----------------|----------------|---------------|----------|-----------|--------|
| XGBoost    | 42            | 8              | 12             | 38            | 0.800    | 0.840     | 0.778  |
| LightGBM   | 45            | 5              | 9              | 41            | 0.860    | 0.900     | 0.833  |

---

### Variant 3 — Unpivot (wide → long) using UNION ALL

**Goal:** Convert a wide table back to long format.

```sql
-- Wide source table: model_metrics_wide
-- Columns: model_name | train_acc | val_acc | test_acc

SELECT model_name, 'train' AS dataset, train_acc AS accuracy FROM model_metrics_wide
UNION ALL
SELECT model_name, 'val',              val_acc               FROM model_metrics_wide
UNION ALL
SELECT model_name, 'test',             test_acc              FROM model_metrics_wide
ORDER BY model_name, dataset;
```

**Output:**

| model_name | dataset | accuracy |
|------------|---------|----------|
| BERT       | test    | 0.890    |
| BERT       | train   | 0.940    |
| BERT       | val     | 0.905    |
| LightGBM   | test    | 0.940    |
| ...        | ...     | ...      |

> This is the standard way to feed wide feature tables into long-format ML training pipelines.

---

### Variant 4 — PostgreSQL CROSSTAB (cleaner but DB-specific)

```sql
-- Requires: CREATE EXTENSION IF NOT EXISTS tablefunc;
SELECT *
FROM crosstab(
    -- Source query (must return: row_name, category, value)
    $$
    SELECT model_name, dataset, accuracy
    FROM model_scores_long
    ORDER BY model_name, dataset
    $$,
    -- Category list (defines column order)
    $$VALUES ('test'), ('train'), ('val')$$
) AS pivot_table (
    model_name TEXT,
    test_acc   FLOAT,
    train_acc  FLOAT,
    val_acc    FLOAT
);
```

**Output:** Same as Variant 1 (but column names come from the `AS` clause).

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using SUM vs MAX in pivot | If there's only one row per (model, dataset), both work — but if duplicates exist, SUM inflates the value | Use MAX (or deduplicate first) unless you actually want a sum |
| Hardcoded category names | Column names are static in conditional aggregation — new categories need schema changes | For dynamic pivots use a stored procedure or handle in application layer |
| CROSSTAB category mismatch | Categories in the VALUES list must exactly match the data; missing ones return NULL | Always verify with a `SELECT DISTINCT category` first |
| NULL vs 0 in pivot | `CASE WHEN ... THEN val END` returns NULL (not 0) when condition is false | Use `COALESCE(MAX(CASE WHEN ... THEN val END), 0)` if 0 is the right default |
| Unpivot column count mismatch | UNION ALL requires same number of columns with compatible types | Cast as needed: `CAST(train_acc AS FLOAT)` |




### Extended example — multiple aggregates in one pivot

**Table: `sales`**

| rep | region | amount |
|---|---|---|
| Alice | North | 100 |
| Alice | South | 200 |
| Bob | North | 150 |
| Bob | North | 50 |
| Bob | South | 300 |

**Goal:** Total and count of sales per rep, broken out by region.

```sql
SELECT
    rep,
    SUM(CASE WHEN region = 'North' THEN amount END) AS north_total,
    COUNT(CASE WHEN region = 'North' THEN 1 END)    AS north_count,
    SUM(CASE WHEN region = 'South' THEN amount END) AS south_total,
    COUNT(CASE WHEN region = 'South' THEN 1 END)    AS south_count
FROM   sales
GROUP  BY rep;
```

**Output:**

| rep | north_total | north_count | south_total | south_count |
|---|---|---|---|---|
| Alice | 100 | 1 | 200 | 1 |
| Bob | 200 | 2 | 300 | 1 |

Note: `COUNT(CASE WHEN ... THEN 1 END)` counts non-NULL values — since the CASE returns 1 (non-NULL) only when the condition is true, this counts matching rows per group.

---

### Unpivoting — wide to long

The reverse operation: turn column headers back into row values.

**Table: `wide_scores`**

| student | math | science |
|---|---|---|
| Alice | 90 | 85 |
| Bob | 78 | 92 |

**Using UNION ALL (portable):**

```sql
SELECT student, 'Math'    AS subject, math    AS score FROM wide_scores
UNION ALL
SELECT student, 'Science' AS subject, science AS score FROM wide_scores;
```

**Output:**

| student | subject | score |
|---|---|---|
| Alice | Math | 90 |
| Alice | Science | 85 |
| Bob | Math | 78 |
| Bob | Science | 92 |

#### Why `UNION ALL` not `UNION`?

`UNION` deduplicates rows. If two students happened to have the same name, subject, and score, `UNION` would silently remove one row. `UNION ALL` keeps all rows — which is correct for unpivoting.

---

## Summary: when to use what

| Need | Technique |
|---|---|
| Both sides of a within-table relationship (hierarchy, co-purchase, pairs) | Self-join with two aliases |
| All unordered pairs | Self-join with `a.id < b.id` |
| Friends of friends / multi-hop | Chain of self-joins |
| Turn row values into columns | `SUM(CASE WHEN ...)` + `GROUP BY` |
| Turn columns into rows | `UNION ALL` per column, or `CROSS JOIN LATERAL (VALUES ...)` |
---

## 28. GROUPING SETS, ROLLUP, CUBE

### Concept

These are extensions to GROUP BY that compute **multiple levels of aggregation in a single query** — instead of running several GROUP BY queries and UNION ALLing them together.

| Clause | What it computes |
|--------|-----------------|
| `GROUPING SETS ((a,b), (a), ())` | Explicitly listed groupings only |
| `ROLLUP(a, b)` | Hierarchy: (a,b) → (a) → () — useful for subtotals |
| `CUBE(a, b)` | All combinations: (a,b), (a), (b), () — useful for cross-tab totals |

**The `GROUPING()` function** returns 1 if a column was aggregated away (i.e., that column is NULL because it's a subtotal row), 0 if it was used in the grouping. Use it to distinguish "NULL because subtotal" from "NULL because missing data".

---

### Sample Table: `experiment_results`

| model_name   | dataset | team   | accuracy |
|--------------|---------|--------|----------|
| XGBoost      | train   | Search | 0.960    |
| XGBoost      | test    | Search | 0.910    |
| LightGBM     | train   | Ads    | 0.975    |
| LightGBM     | test    | Ads    | 0.940    |
| BERT         | train   | NLP    | 0.940    |
| BERT         | test    | NLP    | 0.890    |

---

### Variant 1 — GROUPING SETS (explicit grouping combinations)

**Goal:** In one query, get: per (model, dataset), per model only, and grand total.

```sql
SELECT
    model_name,
    dataset,
    ROUND(AVG(accuracy), 3) AS avg_accuracy,
    COUNT(*)                AS n_rows,
    GROUPING(model_name)    AS is_model_subtotal,   -- 1 = this column was rolled up
    GROUPING(dataset)       AS is_dataset_subtotal
FROM experiment_results
GROUP BY GROUPING SETS (
    (model_name, dataset),   -- group 1: per model per dataset
    (model_name),            -- group 2: per model (across all datasets)
    ()                       -- group 3: grand total
)
ORDER BY model_name NULLS LAST, dataset NULLS LAST;
```

**Output:**

| model_name | dataset | avg_accuracy | n_rows | is_model_subtotal | is_dataset_subtotal |
|------------|---------|--------------|--------|-------------------|---------------------|
| BERT       | test    | 0.890        | 1      | 0                 | 0                   |
| BERT       | train   | 0.940        | 1      | 0                 | 0                   |
| BERT       | NULL    | 0.915        | 2      | 0                 | 1 ← dataset rolled up |
| LightGBM   | test    | 0.940        | 1      | 0                 | 0                   |
| LightGBM   | train   | 0.975        | 1      | 0                 | 0                   |
| LightGBM   | NULL    | 0.958        | 2      | 0                 | 1                   |
| XGBoost    | test    | 0.910        | 1      | 0                 | 0                   |
| XGBoost    | train   | 0.960        | 1      | 0                 | 0                   |
| XGBoost    | NULL    | 0.935        | 2      | 0                 | 1                   |
| NULL       | NULL    | 0.936        | 6      | 1                 | 1 ← grand total    |

---

### Variant 2 — ROLLUP (hierarchical subtotals — the most common pattern)

**Goal:** Show accuracy by (team → model → dataset), with subtotals at each level.

```sql
SELECT
    team,
    model_name,
    dataset,
    ROUND(AVG(accuracy), 3) AS avg_accuracy,
    COUNT(*)                AS n
FROM experiment_results
GROUP BY ROLLUP(team, model_name, dataset)
ORDER BY team NULLS LAST, model_name NULLS LAST, dataset NULLS LAST;
```

**Output:**

| team   | model_name | dataset | avg_accuracy | n |
|--------|------------|---------|--------------|---|
| Ads    | LightGBM   | test    | 0.940        | 1 |
| Ads    | LightGBM   | train   | 0.975        | 1 |
| Ads    | LightGBM   | NULL    | 0.958        | 2 | ← LightGBM subtotal
| Ads    | NULL       | NULL    | 0.958        | 2 | ← Ads team subtotal
| NLP    | BERT       | test    | 0.890        | 1 |
| NLP    | BERT       | train   | 0.940        | 1 |
| NLP    | BERT       | NULL    | 0.915        | 2 | ← BERT subtotal
| NLP    | NULL       | NULL    | 0.915        | 2 | ← NLP team subtotal
| Search | XGBoost    | test    | 0.910        | 1 |
| Search | XGBoost    | train   | 0.960        | 1 |
| Search | XGBoost    | NULL    | 0.935        | 2 | ← XGBoost subtotal
| Search | NULL       | NULL    | 0.935        | 2 | ← Search team subtotal
| NULL   | NULL       | NULL    | 0.936        | 6 | ← Grand total

> ROLLUP produces subtotals from right to left: removes dataset first, then model_name, then team.

---

### Variant 3 — CUBE (all combinations — useful for dashboards)

**Goal:** All possible aggregation combinations of (model_name, dataset).

```sql
SELECT
    COALESCE(model_name, 'ALL MODELS') AS model_name,
    COALESCE(dataset,    'ALL DATASETS') AS dataset,
    ROUND(AVG(accuracy), 3)            AS avg_accuracy,
    COUNT(*)                           AS n
FROM experiment_results
GROUP BY CUBE(model_name, dataset)
ORDER BY
    GROUPING(model_name),
    GROUPING(dataset),
    model_name,
    dataset;
```

**Output:**

| model_name  | dataset      | avg_accuracy | n |
|-------------|--------------|--------------|---|
| BERT        | test         | 0.890        | 1 |
| BERT        | train        | 0.940        | 1 |
| LightGBM    | test         | 0.940        | 1 |
| LightGBM    | train        | 0.975        | 1 |
| XGBoost     | test         | 0.910        | 1 |
| XGBoost     | train        | 0.960        | 1 |
| BERT        | ALL DATASETS | 0.915        | 2 | ← model subtotal
| LightGBM    | ALL DATASETS | 0.958        | 2 |
| XGBoost     | ALL DATASETS | 0.935        | 2 |
| ALL MODELS  | test         | 0.913        | 3 | ← dataset subtotal
| ALL MODELS  | train        | 0.958        | 3 |
| ALL MODELS  | ALL DATASETS | 0.936        | 6 | ← grand total

> CUBE(a, b) computes: (a,b), (a), (b), () — 2² = 4 grouping combinations.
> CUBE(a, b, c) computes 2³ = 8 combinations. Use carefully — can get expensive.

---

### Variant 4 — GROUPING() to label subtotal rows cleanly

```sql
SELECT
    CASE WHEN GROUPING(model_name) = 1 THEN '** TOTAL **'
         ELSE model_name END             AS model_name,
    CASE WHEN GROUPING(dataset) = 1 THEN '** ALL **'
         ELSE dataset END                AS dataset,
    ROUND(AVG(accuracy), 3)             AS avg_accuracy
FROM experiment_results
GROUP BY ROLLUP(model_name, dataset)
ORDER BY GROUPING(model_name), model_name, GROUPING(dataset), dataset;
```

**Output:**

| model_name  | dataset   | avg_accuracy |
|-------------|-----------|--------------|
| BERT        | test      | 0.890        |
| BERT        | train     | 0.940        |
| BERT        | ** ALL ** | 0.915        |
| LightGBM    | test      | 0.940        |
| LightGBM    | train     | 0.975        |
| LightGBM    | ** ALL ** | 0.958        |
| XGBoost     | test      | 0.910        |
| XGBoost     | train     | 0.960        |
| XGBoost     | ** ALL ** | 0.935        |
| ** TOTAL ** | ** ALL ** | 0.936        |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| NULL ambiguity | `WHERE model_name IS NULL` matches both subtotal rows AND rows where model_name was actually NULL in data | Use `GROUPING(model_name) = 1` to identify subtotal rows precisely |
| ROLLUP column order matters | `ROLLUP(a, b)` gives subtotals (a,b)→(a)→() but NOT (b) alone | If you need (b) alone too, use CUBE or explicit GROUPING SETS |
| CUBE on many columns is expensive | CUBE(a,b,c,d) = 16 grouping combinations | Limit to 3-4 columns max; use GROUPING SETS to control exactly what's computed |
| COALESCE masking real NULLs | Using COALESCE to label subtotals hides actual NULL values in data | Use GROUPING() function instead of COALESCE for subtotal labelling |
| Not all DBs support all three | MySQL < 8.0 lacks GROUPING SETS and CUBE | In BigQuery, use GROUPING SETS; ROLLUP supported everywhere |

---

## 29. STRING_AGG / ARRAY_AGG and Unnesting

### Concept

These functions do the **opposite of pivoting/joining** — they collapse multiple rows into a single aggregated value per group.

| Function | Output type | DB support |
|----------|-------------|------------|
| `STRING_AGG(col, delim ORDER BY ...)` | Delimited string | PostgreSQL, BigQuery, SQL Server |
| `GROUP_CONCAT(col ORDER BY ... SEPARATOR ',')` | Delimited string | MySQL |
| `LISTAGG(col, ',') WITHIN GROUP (ORDER BY ...)` | Delimited string | Oracle, Redshift |
| `ARRAY_AGG(col ORDER BY ...)` | Native array | PostgreSQL, BigQuery |
| `UNNEST(array)` | Rows | PostgreSQL, BigQuery |
| `STRING_TO_ARRAY(str, delim)` | Array from string | PostgreSQL |

---

### Sample Table: `model_features`

| model_name   | feature_name   | importance | feature_type |
|--------------|----------------|------------|--------------|
| XGBoost      | user_age       | 0.30       | demographic  |
| XGBoost      | income         | 0.45       | financial    |
| XGBoost      | credit_score   | 0.25       | financial    |
| LightGBM     | user_age       | 0.20       | demographic  |
| LightGBM     | income         | 0.55       | financial    |
| LightGBM     | credit_score   | 0.15       | financial    |
| LightGBM     | clicks_7d      | 0.10       | behavioral   |
| BERT         | token_count    | 0.40       | text         |
| BERT         | sentiment      | 0.35       | text         |
| BERT         | entity_count   | 0.25       | text         |

---

### Variant 1 — STRING_AGG: collapse features per model (ordered by importance)

```sql
SELECT
    model_name,
    COUNT(*)                                                  AS feature_count,
    STRING_AGG(feature_name, ', ' ORDER BY importance DESC)  AS features_by_importance,
    STRING_AGG(feature_name, ', ' ORDER BY feature_name)     AS features_alphabetical
FROM model_features
GROUP BY model_name
ORDER BY model_name;
```

**Output:**

| model_name | feature_count | features_by_importance                   | features_alphabetical                    |
|------------|---------------|------------------------------------------|------------------------------------------|
| BERT       | 3             | token_count, sentiment, entity_count     | entity_count, sentiment, token_count     |
| LightGBM   | 4             | income, user_age, credit_score, clicks_7d | clicks_7d, credit_score, income, user_age |
| XGBoost    | 3             | income, user_age, credit_score           | credit_score, income, user_age           |

---

### Variant 2 — ARRAY_AGG: collect features as an array + filter NULLs

```sql
SELECT
    model_name,
    ARRAY_AGG(feature_name ORDER BY importance DESC)            AS feature_array,
    ARRAY_AGG(importance   ORDER BY importance DESC)            AS importance_array,
    ARRAY_AGG(
        CASE WHEN importance > 0.25 THEN feature_name END
    )                                                           AS top_features_raw,     -- includes NULLs
    ARRAY_REMOVE(
        ARRAY_AGG(
            CASE WHEN importance > 0.25 THEN feature_name END
            ORDER BY importance DESC
        ),
        NULL
    )                                                           AS top_features_clean    -- NULLs removed
FROM model_features
GROUP BY model_name
ORDER BY model_name;
```

**Output:**

| model_name | feature_array                                 | top_features_clean      |
|------------|-----------------------------------------------|-------------------------|
| BERT       | {token_count, sentiment, entity_count}        | {token_count, sentiment} |
| LightGBM   | {income, user_age, credit_score, clicks_7d}   | {income}                |
| XGBoost    | {income, user_age, credit_score}              | {income, user_age}      |

---

### Variant 3 — UNNEST: explode an array column back to rows

```sql
-- Suppose we have a table with a pre-aggregated array column:
-- model_tags: model_name | tags (array)

-- XGBoost | {classification, tabular, prod}
-- BERT    | {nlp, text, classification, prod}

SELECT
    model_name,
    UNNEST(tags) AS tag
FROM model_tags
ORDER BY model_name, tag;
```

**Output:**

| model_name | tag            |
|------------|----------------|
| BERT       | classification |
| BERT       | nlp            |
| BERT       | prod           |
| BERT       | text           |
| XGBoost    | classification |
| XGBoost    | prod           |
| XGBoost    | tabular        |

---

### Variant 4 — Full pipeline: aggregate → filter → unnest → re-aggregate

**Goal:** Find features shared by 2+ models (intersection via aggregation).

```sql
-- Step 1: collect features per model as array
WITH model_feature_sets AS (
    SELECT
        model_name,
        ARRAY_AGG(feature_name ORDER BY feature_name) AS feature_set
    FROM model_features
    GROUP BY model_name
),

-- Step 2: explode all features with their model
exploded AS (
    SELECT
        model_name,
        UNNEST(feature_set) AS feature_name
    FROM model_feature_sets
),

-- Step 3: count how many models use each feature
feature_coverage AS (
    SELECT
        feature_name,
        COUNT(DISTINCT model_name)               AS model_count,
        STRING_AGG(model_name, ', ' ORDER BY model_name) AS used_by_models
    FROM exploded
    GROUP BY feature_name
)

-- Step 4: keep only shared features
SELECT *
FROM feature_coverage
WHERE model_count >= 2
ORDER BY model_count DESC, feature_name;
```

**Output:**

| feature_name | model_count | used_by_models             |
|--------------|-------------|----------------------------|
| credit_score | 2           | LightGBM, XGBoost          |
| income       | 2           | LightGBM, XGBoost          |
| user_age     | 2           | LightGBM, XGBoost          |

---

### Variant 5 — STRING_TO_ARRAY + UNNEST: parse comma-separated column

```sql
-- Source table: model_registry
-- model_name | tags_csv (text column, comma-separated)
-- XGBoost    | 'classification,tabular,prod'
-- BERT       | 'nlp,text,classification'

WITH exploded AS (
    SELECT
        model_name,
        TRIM(UNNEST(STRING_TO_ARRAY(tags_csv, ','))) AS tag
    FROM model_registry
)
SELECT
    tag,
    COUNT(DISTINCT model_name) AS model_count,
    STRING_AGG(model_name, ', ' ORDER BY model_name) AS models
FROM exploded
GROUP BY tag
ORDER BY model_count DESC, tag;
```

**Output:**

| tag            | model_count | models             |
|----------------|-------------|--------------------|
| classification | 2           | BERT, XGBoost      |
| nlp            | 1           | BERT               |
| prod           | 1           | XGBoost            |
| tabular        | 1           | XGBoost            |
| text           | 1           | BERT               |

---

### Variant 6 — ARRAY_AGG with DISTINCT (deduplicate within aggregation)

```sql
SELECT
    feature_type,
    ARRAY_AGG(DISTINCT model_name ORDER BY model_name) AS models_using_type,
    COUNT(DISTINCT model_name)                         AS model_count
FROM model_features
GROUP BY feature_type
ORDER BY model_count DESC;
```

**Output:**

| feature_type | models_using_type          | model_count |
|--------------|----------------------------|-------------|
| financial    | {LightGBM, XGBoost}        | 2           |
| behavioral   | {LightGBM}                 | 1           |
| demographic  | {LightGBM, XGBoost}        | 2           |
| text         | {BERT}                     | 1           |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| STRING_AGG without ORDER BY | Output order is non-deterministic — different every run | Always include `ORDER BY` inside STRING_AGG |
| NULL values silently dropped | STRING_AGG and ARRAY_AGG both skip NULLs by default | Use `COALESCE(col, 'UNKNOWN')` before aggregating if NULLs matter |
| ARRAY_AGG with DISTINCT doesn't allow ORDER BY in some DBs | `ARRAY_AGG(DISTINCT col ORDER BY col)` errors in some Postgres versions | Use a subquery: aggregate without DISTINCT, then dedupe the array |
| UNNEST with multiple columns causes row multiplication | `UNNEST(arr1), UNNEST(arr2)` in same SELECT produces a cross join if arrays differ in length | Use `UNNEST(arr1) WITH ORDINALITY` or zip carefully |
| String delimiter collision | If a feature name contains the delimiter (e.g. comma), parsing breaks | Use an unlikely delimiter like `' | '` or use ARRAY_AGG instead |
| Very long STRING_AGG output | Can hit VARCHAR size limits or slow down comparisons | Consider ARRAY_AGG or store in a separate child table |
| Confusing aggregation direction | STRING_AGG is many-rows-to-one; UNNEST is one-row-to-many | Draw the direction before writing the query |

---

## Aggregation Patterns — Combined Cheat Sheet

### Deduplication Decision Tree

```
Does every column need to be identical to count as a duplicate?
    YES → SELECT DISTINCT (but exclude surrogate keys like pred_id)
    NO  → Do you need to choose WHICH duplicate to keep?
              YES → ROW_NUMBER() OVER (PARTITION BY key ORDER BY tiebreaker) = 1
              NO  → GROUP BY key + aggregates (MAX, MIN, COUNT)
```

---

### Pivot Pattern Reference

| Need | Pattern |
|------|---------|
| Rows → columns (pivot) | `MAX(CASE WHEN cat = 'x' THEN val END) AS x` per group |
| Columns → rows (unpivot) | `UNION ALL` one SELECT per column |
| Confusion matrix | Conditional SUM per (label, predicted) combination |
| A/B test comparison | Conditional AVG per variant |
| Dynamic pivot | Not natively possible in SQL — handle in Python/application layer |

---

### ROLLUP vs CUBE vs GROUPING SETS

| Clause | Combinations produced | Use case |
|--------|-----------------------|----------|
| `ROLLUP(a, b, c)` | (a,b,c), (a,b), (a), () | Hierarchical subtotals (e.g. year → month → day) |
| `CUBE(a, b)` | (a,b), (a), (b), () | Cross-tab with all marginal totals |
| `GROUPING SETS((a,b),(a),())` | Exactly what you list | Full control; most explicit |

---

### STRING_AGG / ARRAY_AGG Quick Reference

```sql
-- Collapse rows to string
STRING_AGG(col, ', ' ORDER BY col)

-- Collapse rows to array
ARRAY_AGG(col ORDER BY col)

-- Collapse with deduplication
ARRAY_AGG(DISTINCT col ORDER BY col)

-- Explode array to rows
UNNEST(array_col)

-- Parse string to array then explode
UNNEST(STRING_TO_ARRAY(csv_col, ','))

-- Remove NULLs from array
ARRAY_REMOVE(ARRAY_AGG(col), NULL)

-- Check if array contains a value
'value' = ANY(array_col)

-- Array length
ARRAY_LENGTH(array_col, 1)   -- PostgreSQL
ARRAY_SIZE(array_col)        -- BigQuery / Spark SQL
```

---

# SQL deep dive: self-joins and pivoting

---

## Part 1 — Self-join

### What is a self-join?

A self-join joins a table to itself. SQL has no special `SELF JOIN` keyword — you just reference the same table twice using two different aliases.

```sql
SELECT a.col, b.col
FROM my_table a
JOIN my_table b ON a.some_col = b.other_col;
```

The aliases (`a`, `b`) are mandatory — without them, the database cannot distinguish which copy of the table each column reference belongs to.

---

### Why does it exist?

Some relationships live inside a single table:

| Scenario | Same table contains... |
|---|---|
| Org hierarchy | Employee → their Manager (also an employee) |
| Friend graph | User A ↔ User B (both users) |
| Co-purchase | Order rows for the same customer |
| A/B pairing | Experiment arms in one log table |

Pulling both sides of the relationship into one row requires reading the table twice — hence two aliases.

---

### Use case 1 — Users who bought both product A and product B

**Table: `orders`**

| user_id | product |
|---|---|
| 1 | A |
| 1 | B |
| 2 | A |
| 3 | B |
| 4 | A |
| 4 | B |

**Goal:** Find every `user_id` who bought **A and B**.

```sql
SELECT a.user_id
FROM   orders a
JOIN   orders b
  ON   a.user_id = b.user_id   -- same customer
 AND   a.product = 'A'          -- left copy: bought A
 AND   b.product = 'B';         -- right copy: bought B
```

**Output:**

| user_id |
|---|
| 1 |
| 4 |

#### Why it works

- `a` is every row where `product = 'A'`.
- `b` is every row where `product = 'B'`.
- The `ON a.user_id = b.user_id` condition stitches them together — only users who appear in **both** filtered sets survive.

#### Why it fails (common mistakes)

| Mistake | What happens |
|---|---|
| Forget `AND a.product = 'A'` | Joins every row to every row for the same user — massive cross-product |
| Use `WHERE` instead of `AND` in the join | Logically equivalent for inner joins, but confusing; gets wrong results with outer joins |
| Use `a.product = b.product` | Only matches rows with the same product — returns nothing useful |

---

### Use case 2 — Pairwise comparison (every distinct pair of users in the same city)

**Table: `users`**

| id | name | city |
|---|---|---|
| 1 | Alice | NYC |
| 2 | Bob | NYC |
| 3 | Carol | NYC |
| 4 | Dan | LA |

**Goal:** All unordered pairs of users in the same city.

```sql
SELECT a.name AS user_1,
       b.name AS user_2,
       a.city
FROM   users a
JOIN   users b
  ON   a.city = b.city
 AND   a.id < b.id;   -- prevents (Alice,Bob) AND (Bob,Alice); also removes self-pairs
```

**Output:**

| user_1 | user_2 | city |
|---|---|---|
| Alice | Bob | NYC |
| Alice | Carol | NYC |
| Bob | Carol | NYC |

#### The `a.id < b.id` trick

Without it you get three problems in the result:

1. **Self-pairs** — `(Alice, Alice)`.
2. **Duplicates** — `(Alice, Bob)` and `(Bob, Alice)` both appear.
3. Unnecessary row count explosion.

`a.id < b.id` enforces a strict ordering — only the lower ID goes on the left. This eliminates all three at once.

Using `a.id != b.id` removes self-pairs but still produces duplicates.
Using `a.id > b.id` is equivalent to `<` — just mirror-image; pick one and be consistent.

---

### Use case 3 — Friend recommendations (friends of friends)

**Table: `friendships`**

| user_a | user_b |
|---|---|
| 1 | 2 |
| 1 | 3 |
| 2 | 4 |
| 3 | 4 |

**Goal:** Recommend users who are 2 hops away but not already friends.

```sql
SELECT DISTINCT f1.user_a AS user_id,
                f2.user_b AS recommended
FROM   friendships f1
JOIN   friendships f2
  ON   f1.user_b = f2.user_a           -- chain: user_a → bridge → recommended
WHERE  f2.user_b != f1.user_a          -- don't recommend yourself
  AND  NOT EXISTS (                     -- don't recommend existing friends
         SELECT 1 FROM friendships f3
          WHERE f3.user_a = f1.user_a
            AND f3.user_b = f2.user_b
       );
```

**Reading the chain:**

```
f1: user_a  →  bridge_user   (first hop)
f2:            bridge_user  →  recommended   (second hop)
```

The `JOIN ON f1.user_b = f2.user_a` is the pivot point — it connects the end of the first friendship to the start of the second.

#### Why it fails without the NOT EXISTS

Without the `NOT EXISTS`, you recommend people who are already friends, which is noisy and wrong for a recommendation engine. You could also use a `LEFT JOIN` + `WHERE f3.user_a IS NULL` pattern as an alternative.

---

