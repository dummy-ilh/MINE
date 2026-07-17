
# SQL for ML Interviews — Data Types & Advanced Functions

> **Module 6** — The topics that separate mid-level from senior. Same format throughout: concept → variants with input/output → pitfalls. No other source needed.

---

## Table of Contents

43. [Data Types and Casting](#43-data-types-and-casting)
44. [JSON and Array Functions](#44-json-and-array-functions)
45. [LATERAL Joins / CROSS APPLY](#45-lateral-joins--cross-apply)
46. [Approximate Aggregations](#46-approximate-aggregations)

---

## 43. Data Types and Casting

### Concept

SQL is **strictly typed** — every column has a declared type and mixing types causes errors or silent wrong results. Understanding the type system is critical because:

- Division between two integers truncates (no decimal)
- Comparing a VARCHAR to an INTEGER causes implicit cast → index not used
- JSON, ARRAY, UUID, BOOLEAN all behave differently from VARCHAR
- ML pipelines frequently ingest data as text and must cast on the way in

---

### PostgreSQL Type Hierarchy (know this map)

```
Numeric
├── INTEGER (INT)        — 4 bytes, -2.1B to 2.1B
├── BIGINT               — 8 bytes, ~9.2 × 10^18
├── NUMERIC(p, s)        — arbitrary precision (use for money)
├── FLOAT / REAL         — 4-byte floating point
└── DOUBLE PRECISION     — 8-byte floating point

Text
├── VARCHAR(n)           — variable up to n chars
├── TEXT                 — unlimited
└── CHAR(n)              — fixed width, right-padded with spaces

Temporal
├── DATE                 — year-month-day
├── TIME                 — time of day
├── TIMESTAMP            — date + time, no timezone
└── TIMESTAMPTZ          — date + time + timezone (store this in prod)

Other
├── BOOLEAN              — true / false / NULL
├── UUID                 — 128-bit universally unique ID
├── JSON / JSONB         — semi-structured (JSONB = binary, indexed)
├── ARRAY                — typed arrays (e.g. INTEGER[], TEXT[])
└── ENUM                 — user-defined set of allowed values
```

---

### Two Cast Syntaxes — Always Equivalent

```sql
CAST(col AS type)    -- SQL standard, works everywhere
col::type            -- PostgreSQL shorthand, cleaner for chaining
```

---

### Sample Table: `raw_model_imports` (everything arrived as TEXT)

| id | model_name | accuracy_txt | created_txt      | is_active_txt | tags_txt                  |
|----|------------|--------------|------------------|---------------|---------------------------|
| 1  | XGBoost    | 0.910        | 2024-01-15       | true          | classification,tabular    |
| 2  | LightGBM   | 0.945        | 2024-01-20       | TRUE          | regression,tabular        |
| 3  | BERT       | N/A          | 2024-02-03       | false         | nlp,text                  |
| 4  | ResNet     | 0.88         | not-a-date       | 1             | vision                    |

---

### Variant 1 — Safe casting of all common types

```sql
SELECT
    id,
    model_name,

    -- FLOAT: guard against non-numeric strings
    CASE
        WHEN accuracy_txt ~ '^[0-9]+\.?[0-9]*$'   -- regex: valid number
        THEN accuracy_txt::FLOAT
        ELSE NULL
    END                                             AS accuracy,

    -- DATE: guard against bad date strings
    CASE
        WHEN created_txt ~ '^\d{4}-\d{2}-\d{2}$'
        THEN created_txt::DATE
        ELSE NULL
    END                                             AS created_date,

    -- BOOLEAN: handle multiple representations
    CASE
        WHEN LOWER(is_active_txt) IN ('true', '1', 'yes') THEN TRUE
        WHEN LOWER(is_active_txt) IN ('false','0', 'no')  THEN FALSE
        ELSE NULL
    END                                             AS is_active

FROM raw_model_imports;
```

**Output:**

| id | model_name | accuracy | created_date | is_active |
|----|------------|----------|--------------|-----------|
| 1  | XGBoost    | 0.910    | 2024-01-15   | true      |
| 2  | LightGBM   | 0.945    | 2024-01-20   | true      |
| 3  | BERT       | NULL     | 2024-02-03   | false     |
| 4  | ResNet     | 0.88     | NULL         | true      |

---

### Variant 2 — Numeric type precision traps

```sql
-- Integer division silently truncates
SELECT
    7 / 2                        AS int_div,          -- 3 (WRONG if you want 3.5)
    7.0 / 2                      AS float_div,        -- 3.5
    CAST(7 AS FLOAT) / 2        AS cast_div,         -- 3.5
    7::NUMERIC / 2              AS numeric_div,      -- 3.5000...

    -- Numeric precision for money (avoid FLOAT for money)
    CAST('123456789.99' AS NUMERIC(12,2))  AS safe_money,

    -- FLOAT has precision limits
    0.1::FLOAT + 0.2::FLOAT     AS float_sum,        -- 0.30000000000000004
    0.1::NUMERIC + 0.2::NUMERIC AS numeric_sum;      -- 0.3 (exact)
```

**Output:**

| int_div | float_div | cast_div | numeric_sum | float_sum          |
|---------|-----------|----------|-------------|---------------------|
| 3       | 3.5       | 3.5      | 0.3         | 0.30000000000000004 |

---

### Variant 3 — Timestamp and timezone casting

```sql
SELECT
    -- String → timestamp
    '2024-01-15 08:30:00'::TIMESTAMP                            AS ts_naive,
    '2024-01-15 08:30:00+05:30'::TIMESTAMPTZ                   AS ts_with_tz,

    -- Convert between timezones
    '2024-01-15 08:30:00+05:30'::TIMESTAMPTZ
        AT TIME ZONE 'UTC'                                       AS ts_in_utc,

    -- Extract after cast
    EXTRACT(HOUR FROM '2024-01-15 08:30:00'::TIMESTAMP)        AS hour_part,

    -- Date arithmetic only works after casting
    '2024-01-15'::DATE + INTERVAL '7 days'                     AS one_week_later,

    -- Epoch (seconds since 1970-01-01)
    EXTRACT(EPOCH FROM '2024-01-15 00:00:00'::TIMESTAMP)       AS unix_ts;
```

**Output:**

| ts_naive            | ts_in_utc           | hour_part | one_week_later | unix_ts    |
|---------------------|---------------------|-----------|----------------|------------|
| 2024-01-15 08:30:00 | 2024-01-15 03:00:00 | 8         | 2024-01-22     | 1705276800 |

---

### Variant 4 — UUID, ENUM and type-specific patterns

```sql
-- UUID: generate for new records
SELECT gen_random_uuid() AS new_experiment_id;
-- Output: 'a3f1c2d4-89ab-4e23-b1c2-0d9f3e4a5b6c'

-- Compare UUIDs safely (no string comparison issues)
SELECT * FROM experiments
WHERE exp_id = 'a3f1c2d4-89ab-4e23-b1c2-0d9f3e4a5b6c'::UUID;

-- ENUM: restrict allowed values at schema level
CREATE TYPE model_status AS ENUM ('training', 'deployed', 'retired', 'failed');

-- Cast integer codes to labels
SELECT
    status_code,
    CASE status_code
        WHEN 1 THEN 'training'
        WHEN 2 THEN 'deployed'
        WHEN 3 THEN 'retired'
        WHEN 4 THEN 'failed'
    END::model_status AS status
FROM model_runs;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Integer division truncates silently | `5/2 = 2`, not `2.5` — no error, just wrong | Always cast one operand: `5::FLOAT/2` or `5.0/2` |
| FLOAT for money | `0.1 + 0.2 = 0.30000000000000004` | Use `NUMERIC(p,s)` for financial values |
| Implicit cast defeats indexes | `WHERE int_col = '5'` casts string to int or int to string, skips index | Match types: `WHERE int_col = 5` |
| TIMESTAMP vs TIMESTAMPTZ | TIMESTAMP ignores timezone — dangerous for global products | Always store as TIMESTAMPTZ in production |
| `::DATE` from TIMESTAMPTZ uses server timezone | `ts::DATE` in UTC gives different date than local midnight | Cast explicitly: `ts AT TIME ZONE 'America/New_York')::DATE` |
| VARCHAR(n) silently truncates on INSERT in some DBs | PostgreSQL raises an error; MySQL silently truncates | Use TEXT in PostgreSQL; be explicit in MySQL |
| CHAR(n) pads with spaces | `CHAR(10)` stores `'abc       '` — equality checks fail | Use VARCHAR or TEXT instead |

---

## 44. JSON and Array Functions

### Concept

Modern ML pipelines store semi-structured data in SQL — model configs, feature vectors, hyperparameters, API payloads — as JSON or arrays. Two types in PostgreSQL:

| Type | Description | When to use |
|------|-------------|-------------|
| `JSON` | Stored as text, re-parsed on every access | Rarely — just use JSONB |
| `JSONB` | Binary format, indexed, faster for queries | Always prefer this |

**Key operators:**

| Operator | Returns | Example |
|----------|---------|---------|
| `col->>'key'` | TEXT value of key | `config->>'learning_rate'` |
| `col->'key'` | JSON sub-object | `config->'layers'` |
| `col#>>'{a,b}'` | TEXT at nested path | `config#>>'{optimizer,lr}'` |
| `col#>'{a,b}'`  | JSON at nested path | `config#>'{optimizer}'` |
| `jsonb_each(col)` | Set of (key, value) rows | Explode all keys to rows |
| `jsonb_object_keys(col)` | Set of keys | List all keys in object |
| `jsonb_array_elements(col)` | Rows from JSON array | Explode JSON array |
| `@>` | Contains operator (uses index) | `config @> '{"type":"xgb"}'` |

---

### Sample Table: `ml_experiments`

| exp_id | model_name | config                                                           | metrics                              | feature_scores           |
|--------|------------|------------------------------------------------------------------|--------------------------------------|--------------------------|
| 1      | XGBoost    | `{"lr":0.1,"depth":6,"n_estimators":200,"type":"boosting"}`    | `{"accuracy":0.91,"f1":0.89}`        | `[0.45, 0.30, 0.25]`     |
| 2      | LightGBM   | `{"lr":0.05,"depth":8,"n_estimators":500,"type":"boosting"}`   | `{"accuracy":0.94,"f1":0.93}`        | `[0.55, 0.25, 0.20]`     |
| 3      | BERT       | `{"layers":12,"hidden":768,"type":"transformer","lr":0.00002}` | `{"accuracy":0.87,"f1":0.85,"auc":0.91}` | `[0.40, 0.35, 0.25]`|

---

### Variant 1 — Extract scalar fields from JSONB

```sql
SELECT
    exp_id,
    model_name,

    -- Extract as text then cast
    (config->>'lr')::FLOAT                  AS learning_rate,
    (config->>'depth')::INT                 AS max_depth,
    config->>'type'                         AS model_type,

    -- Metrics
    (metrics->>'accuracy')::FLOAT           AS accuracy,
    (metrics->>'f1')::FLOAT                 AS f1_score,

    -- Nested path (BERT has optimizer nested)
    (config#>>'{lr}')::FLOAT                AS lr_via_path,

    -- NULL-safe: key may not exist in all rows
    (metrics->>'auc')::FLOAT               AS auc    -- NULL for XGBoost and LightGBM

FROM ml_experiments
ORDER BY (metrics->>'accuracy')::FLOAT DESC;
```

**Output:**

| exp_id | model_name | learning_rate | max_depth | model_type  | accuracy | f1_score | auc  |
|--------|------------|---------------|-----------|-------------|----------|----------|------|
| 2      | LightGBM   | 0.05          | 8         | boosting    | 0.94     | 0.93     | NULL |
| 1      | XGBoost    | 0.10          | 6         | boosting    | 0.91     | 0.89     | NULL |
| 3      | BERT       | 0.00002       | NULL      | transformer | 0.87     | 0.85     | 0.91 |

---

### Variant 2 — Filter using JSONB containment operator (@>)

```sql
-- Find all boosting-type models with lr < 0.1
SELECT
    exp_id,
    model_name,
    config->>'type'    AS model_type,
    (config->>'lr')::FLOAT AS lr
FROM ml_experiments
WHERE config @> '{"type": "boosting"}'        -- uses GIN index if created
  AND (config->>'lr')::FLOAT < 0.1;
```

**Output:**

| exp_id | model_name | model_type | lr   |
|--------|------------|------------|------|
| 2      | LightGBM   | boosting   | 0.05 |

---

### Variant 3 — Explode JSON object keys to rows (jsonb_each)

```sql
-- Show each hyperparameter as its own row — useful for comparing configs
SELECT
    exp_id,
    model_name,
    key   AS param_name,
    value AS param_value
FROM ml_experiments,
     jsonb_each(config) AS kv(key, value)     -- implicit LATERAL
ORDER BY exp_id, key;
```

**Output:**

| exp_id | model_name | param_name   | param_value |
|--------|------------|--------------|-------------|
| 1      | XGBoost    | depth        | 6           |
| 1      | XGBoost    | lr           | 0.1         |
| 1      | XGBoost    | n_estimators | 200         |
| 1      | XGBoost    | type         | "boosting"  |
| 2      | LightGBM   | depth        | 8           |
| 2      | LightGBM   | lr           | 0.05        |
| ...    | ...        | ...          | ...         |
| 3      | BERT       | hidden       | 768         |
| 3      | BERT       | layers       | 12          |
| 3      | BERT       | lr           | 0.00002     |
| 3      | BERT       | type         | "transformer"|

---

### Variant 4 — Explode JSON array (feature_scores)

```sql
-- Unnest the feature_scores array with position index
SELECT
    exp_id,
    model_name,
    idx - 1                          AS feature_index,   -- 0-indexed
    score::FLOAT                     AS importance_score
FROM ml_experiments,
     jsonb_array_elements_text(feature_scores) WITH ORDINALITY AS t(score, idx)
ORDER BY exp_id, feature_index;
```

**Output:**

| exp_id | model_name | feature_index | importance_score |
|--------|------------|---------------|------------------|
| 1      | XGBoost    | 0             | 0.45             |
| 1      | XGBoost    | 1             | 0.30             |
| 1      | XGBoost    | 2             | 0.25             |
| 2      | LightGBM   | 0             | 0.55             |
| 2      | LightGBM   | 1             | 0.25             |
| 2      | LightGBM   | 2             | 0.20             |
| 3      | BERT       | 0             | 0.40             |
| 3      | BERT       | 1             | 0.35             |
| 3      | BERT       | 2             | 0.25             |

---

### Variant 5 — Build JSONB from query results (jsonb_build_object, jsonb_agg)

```sql
-- Aggregate experiment results into a JSON summary per model type
SELECT
    config->>'type'                                    AS model_type,
    COUNT(*)                                           AS experiment_count,
    jsonb_agg(
        jsonb_build_object(
            'exp_id',    exp_id,
            'model',     model_name,
            'accuracy',  (metrics->>'accuracy')::FLOAT
        )
        ORDER BY (metrics->>'accuracy')::FLOAT DESC
    )                                                  AS experiments_json
FROM ml_experiments
GROUP BY config->>'type';
```

**Output:**

| model_type  | experiment_count | experiments_json |
|-------------|------------------|------------------|
| boosting    | 2                | `[{"exp_id":2,"model":"LightGBM","accuracy":0.94},{"exp_id":1,"model":"XGBoost","accuracy":0.91}]` |
| transformer | 1                | `[{"exp_id":3,"model":"BERT","accuracy":0.87}]` |

---

### Native Array Functions

```sql
-- Array literal
SELECT ARRAY[0.45, 0.30, 0.25] AS scores;

-- Array length
SELECT ARRAY_LENGTH(ARRAY[1,2,3], 1);     -- 3 (1 = first dimension)

-- Access by index (1-indexed)
SELECT (ARRAY[10, 20, 30])[2];            -- 20

-- Append to array
SELECT ARRAY_APPEND(ARRAY[1,2], 3);       -- {1,2,3}

-- Check membership
SELECT 2 = ANY(ARRAY[1, 2, 3]);           -- true
SELECT 5 = ALL(ARRAY[5, 5, 5]);           -- true

-- Array contains
SELECT ARRAY[1,2,3] @> ARRAY[2,3];        -- true

-- Unnest (rows)
SELECT UNNEST(ARRAY['a','b','c']);         -- three rows

-- Array aggregation
SELECT ARRAY_AGG(accuracy ORDER BY accuracy DESC)
FROM model_scores;

-- Remove element
SELECT ARRAY_REMOVE(ARRAY[1,2,3,2], 2);  -- {1,3}

-- Convert to string
SELECT ARRAY_TO_STRING(ARRAY[1,2,3], ','); -- '1,2,3'

-- String to array
SELECT STRING_TO_ARRAY('a,b,c', ',');     -- {a,b,c}
```

---

### Variant 6 — Find experiments where feature 0 has importance > 0.4 (array indexing)

```sql
-- feature_scores is stored as a native FLOAT[] column
SELECT
    exp_id,
    model_name,
    feature_scores[1]               AS top_feature_score,   -- 1-indexed!
    feature_scores[1] > 0.4        AS top_feature_dominant,
    ARRAY_LENGTH(feature_scores, 1) AS n_features
FROM ml_experiments_array   -- version with native array column
WHERE feature_scores[1] > 0.4
ORDER BY feature_scores[1] DESC;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `->` returns JSON, `->>` returns TEXT | `(config->'lr') + 0.1` errors — can't add to JSON | Always use `->>` then cast: `(config->>'lr')::FLOAT + 0.1` |
| Missing key returns NULL, not error | `config->>'nonexistent'` = NULL silently | Use `COALESCE(config->>'key', 'default')` |
| JSON vs JSONB | JSON stores text, re-parses each time; can't index | Always use JSONB in production |
| GIN index required for @> | Without index, `@>` does full table scan | `CREATE INDEX idx ON t USING GIN(config)` |
| Array indexing is 1-based in SQL | `arr[1]` is the first element — opposite of Python's `arr[0]` | Remember: SQL arrays start at 1 |
| `jsonb_each` output type | `value` column is of type JSONB, not TEXT | Cast to text: `value::TEXT` or use `jsonb_each_text` |
| JSON key ordering not preserved in JSONB | JSONB may reorder keys during storage | Don't rely on key order; always access by name |

---

## 45. LATERAL Joins / CROSS APPLY

### Concept

A **LATERAL join** allows a subquery on the right side of a JOIN to **reference columns from the left side** — something ordinary subqueries cannot do. It's like a for-each loop in SQL.

```sql
SELECT ...
FROM left_table l
JOIN LATERAL (
    SELECT ...
    FROM right_table r
    WHERE r.col = l.col    -- ← can reference l.col here
    LIMIT 3
) sub ON TRUE;
```

**Without LATERAL:** The subquery must be self-contained — it cannot see the current row of the left table.
**With LATERAL:** The subquery is re-evaluated for each row of the left table — like a correlated subquery, but it can return multiple rows.

**Equivalents:**
- PostgreSQL: `JOIN LATERAL ... ON TRUE` or `LEFT JOIN LATERAL ... ON TRUE`
- SQL Server / Sybase: `CROSS APPLY` (inner) / `OUTER APPLY` (left)
- BigQuery: Implicit lateral with `UNNEST` always works as lateral

**When LATERAL is the right tool:**
- Top-N per group (alternative to ROW_NUMBER window)
- Calling a set-returning function per row (`jsonb_array_elements`, `UNNEST`, `regexp_matches`)
- Computing a correlated aggregation that needs to return multiple rows
- Passing per-row parameters to a subquery

---

### Sample Tables

**`models`**

| model_id | model_name | team   |
|----------|------------|--------|
| 1        | XGBoost    | Search |
| 2        | LightGBM   | Ads    |
| 3        | BERT       | NLP    |

**`experiment_results`**

| exp_id | model_id | accuracy | run_date   |
|--------|----------|----------|------------|
| 1      | 1        | 0.910    | 2024-01-10 |
| 2      | 1        | 0.915    | 2024-01-11 |
| 3      | 1        | 0.905    | 2024-01-12 |
| 4      | 2        | 0.940    | 2024-01-10 |
| 5      | 2        | 0.945    | 2024-01-11 |
| 6      | 2        | 0.935    | 2024-01-12 |
| 7      | 3        | 0.870    | 2024-01-10 |
| 8      | 3        | 0.885    | 2024-01-11 |

---

### Variant 1 — Top-2 experiments per model using LATERAL

```sql
SELECT
    m.model_name,
    top_exps.exp_id,
    top_exps.accuracy,
    top_exps.run_date
FROM models m
JOIN LATERAL (
    SELECT exp_id, accuracy, run_date
    FROM experiment_results e
    WHERE e.model_id = m.model_id          -- reference to outer table
    ORDER BY accuracy DESC
    LIMIT 2
) top_exps ON TRUE                         -- ON TRUE = always join
ORDER BY m.model_name, top_exps.accuracy DESC;
```

**Output:**

| model_name | exp_id | accuracy | run_date   |
|------------|--------|----------|------------|
| BERT       | 8      | 0.885    | 2024-01-11 |
| BERT       | 7      | 0.870    | 2024-01-10 |
| LightGBM   | 5      | 0.945    | 2024-01-11 |
| LightGBM   | 4      | 0.940    | 2024-01-10 |
| XGBoost    | 2      | 0.915    | 2024-01-11 |
| XGBoost    | 1      | 0.910    | 2024-01-10 |

> This is the cleanest way to write Top-N per group without a CTE + ROW_NUMBER pattern.

---

### Variant 2 — LEFT JOIN LATERAL (include models with no experiments)

```sql
SELECT
    m.model_name,
    latest.accuracy AS latest_accuracy,
    latest.run_date AS latest_run_date
FROM models m
LEFT JOIN LATERAL (
    SELECT accuracy, run_date
    FROM experiment_results e
    WHERE e.model_id = m.model_id
    ORDER BY run_date DESC
    LIMIT 1
) latest ON TRUE;
```

**Output:**

| model_name | latest_accuracy | latest_run_date |
|------------|-----------------|-----------------|
| XGBoost    | 0.905           | 2024-01-12      |
| LightGBM   | 0.935           | 2024-01-12      |
| BERT       | 0.885           | 2024-01-11      |

> `LEFT JOIN LATERAL` preserves models with no experiments (they'd show NULL). Regular `JOIN LATERAL` drops them.

---

### Variant 3 — LATERAL to call a set-returning function per row

```sql
-- Expand JSONB config keys for each experiment — LATERAL is implicit here
SELECT
    e.exp_id,
    e.model_name,
    kv.key   AS param,
    kv.value AS value
FROM ml_experiments e,
     jsonb_each(e.config) AS kv         -- implicit LATERAL: jsonb_each sees e.config
ORDER BY e.exp_id, kv.key;
```

> The comma syntax (`FROM t1, function(t1.col)`) is implicit LATERAL in PostgreSQL.
> Same as: `FROM ml_experiments e JOIN LATERAL jsonb_each(e.config) AS kv ON TRUE`

---

### Variant 4 — LATERAL for per-row statistics (correlated multi-row return)

```sql
-- For each model, compute the accuracy trend (slope approximation)
-- using the last 3 experiments
SELECT
    m.model_name,
    trend.avg_accuracy,
    trend.max_accuracy,
    trend.min_accuracy,
    trend.n_runs,
    ROUND(trend.max_accuracy - trend.min_accuracy, 4) AS accuracy_range
FROM models m
JOIN LATERAL (
    SELECT
        AVG(accuracy)  AS avg_accuracy,
        MAX(accuracy)  AS max_accuracy,
        MIN(accuracy)  AS min_accuracy,
        COUNT(*)       AS n_runs
    FROM experiment_results e
    WHERE e.model_id = m.model_id
    ORDER BY run_date DESC
    LIMIT 3
) trend ON TRUE
ORDER BY trend.avg_accuracy DESC;
```

**Output:**

| model_name | avg_accuracy | max_accuracy | min_accuracy | n_runs | accuracy_range |
|------------|--------------|--------------|--------------|--------|----------------|
| LightGBM   | 0.9400       | 0.945        | 0.935        | 3      | 0.0100         |
| XGBoost    | 0.9100       | 0.915        | 0.905        | 3      | 0.0100         |
| BERT       | 0.8775       | 0.885        | 0.870        | 2      | 0.0150         |

---

### Variant 5 — UNNEST with LATERAL (explode array column per row)

```sql
-- Each model has a FLOAT[] column of experiment accuracies
-- Compute per-element statistics

SELECT
    model_name,
    idx,
    score,
    AVG(score) OVER (PARTITION BY model_name) AS model_avg
FROM ml_experiments e,
     UNNEST(feature_scores) WITH ORDINALITY AS t(score, idx)  -- implicit LATERAL
ORDER BY model_name, idx;
```

---

### Variant 6 — CROSS APPLY (SQL Server / Azure SQL syntax)

```sql
-- SQL Server equivalent of JOIN LATERAL
SELECT
    m.model_name,
    top_e.accuracy,
    top_e.run_date
FROM models m
CROSS APPLY (
    SELECT TOP 2 accuracy, run_date
    FROM experiment_results e
    WHERE e.model_id = m.model_id
    ORDER BY accuracy DESC
) top_e;

-- LEFT / OUTER version (keeps rows with no match)
SELECT m.model_name, latest_e.accuracy
FROM models m
OUTER APPLY (
    SELECT TOP 1 accuracy
    FROM experiment_results e
    WHERE e.model_id = m.model_id
    ORDER BY run_date DESC
) latest_e;
```

---

### LATERAL vs Alternatives — When to use which

| Pattern | Readability | Handles Top-N | Handles multi-row return | Works in all DBs |
|---------|------------|---------------|--------------------------|-----------------|
| LATERAL | High | ✅ | ✅ | PostgreSQL, BigQuery |
| CROSS APPLY | High | ✅ | ✅ | SQL Server, Sybase |
| ROW_NUMBER CTE | Medium | ✅ | ✅ | All DBs ✅ |
| Correlated subquery | Low | ✅ (TOP 1 only) | ❌ (single value only) | All DBs |
| Self-JOIN | Low | Complicated | ❌ | All DBs |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Forgetting `ON TRUE` | `JOIN LATERAL (...) ` without `ON TRUE` is a syntax error | Always end with `ON TRUE` for a LATERAL that should always match |
| JOIN vs LEFT JOIN LATERAL | `JOIN LATERAL` drops outer rows with no lateral result (like INNER JOIN) | Use `LEFT JOIN LATERAL` to preserve all left-side rows |
| LATERAL in BigQuery | BigQuery doesn't use the `LATERAL` keyword; use comma syntax or `CROSS JOIN UNNEST(...)` | `FROM t, UNNEST(t.arr)` is implicit lateral in BigQuery |
| Performance: LATERAL re-evaluates per row | A LATERAL subquery with an expensive operation runs once per left-table row | Ensure the inner query is indexed; push filters into the lateral |
| Implicit lateral with set-returning functions | `FROM t, jsonb_each(t.col)` is implicitly lateral — easy to forget it's a LATERAL | Intentional syntax; just know why it works |

---

## 46. Approximate Aggregations

### Concept

On very large tables (hundreds of millions to billions of rows), **exact aggregations are slow and expensive**. Approximate algorithms provide answers within a small, known error margin in a fraction of the time and memory.

| Function | Exact equivalent | Error guarantee | DB support |
|----------|-----------------|-----------------|------------|
| `APPROX_COUNT_DISTINCT` | `COUNT(DISTINCT col)` | ~2% relative error | BigQuery, Redshift, Athena, Spark |
| HyperLogLog (HLL) | `COUNT(DISTINCT col)` | Configurable error | PostgreSQL (hll extension), Redshift |
| `APPROX_QUANTILES(col, n)` | `PERCENTILE_CONT` | ~1% relative error | BigQuery |
| `APPROX_PERCENTILE(col, p)` | `PERCENTILE_CONT(p)` | ~1% | Redshift, Athena, Presto |
| `APPROX_TOP_K(col, k)` | Most frequent values | Approximate | BigQuery |
| `APPROX_TOP_COUNT(col, n)` | Top-N by frequency | Approximate | BigQuery |

**Why this matters for ML interviews:**
- Feature cardinality estimation (is this a high-cardinality column?)
- Daily active user counts at scale
- Latency percentile monitoring (P99) on billions of requests
- Distinct item counts in recommendation systems

---

### Variant 1 — APPROX_COUNT_DISTINCT (BigQuery / Redshift)

```sql
-- BigQuery
SELECT
    model_name,
    -- Exact: expensive on 1B rows
    COUNT(DISTINCT user_id)        AS exact_distinct_users,

    -- Approximate: ~100x faster, ~2% error
    APPROX_COUNT_DISTINCT(user_id) AS approx_distinct_users

FROM model_predictions
GROUP BY model_name;
```

**Conceptual Output** (on 1B row table):

| model_name | exact_distinct_users | approx_distinct_users | diff_pct |
|------------|---------------------|----------------------|----------|
| XGBoost    | 48,291,044          | 48,204,391           | 0.18%    |
| LightGBM   | 31,005,229          | 30,977,108           | 0.09%    |

---

### Variant 2 — HyperLogLog in PostgreSQL (hll extension)

```sql
-- Setup (once)
CREATE EXTENSION IF NOT EXISTS hll;

-- Build HLL sketches daily (cheap to store)
CREATE TABLE daily_user_hll AS
SELECT
    model_name,
    DATE_TRUNC('day', prediction_ts)::DATE AS day,
    hll_add_agg(hll_hash_text(user_id))   AS user_hll   -- build sketch
FROM model_predictions
GROUP BY model_name, day;

-- Query: merge sketches to get weekly distinct users (no need to re-scan raw data)
SELECT
    model_name,
    DATE_TRUNC('week', day)::DATE              AS week,
    hll_cardinality(hll_union_agg(user_hll))   AS approx_weekly_dau
FROM daily_user_hll
GROUP BY model_name, week
ORDER BY model_name, week;
```

> **The power of HLL sketches:** Once you have daily sketches, you can compute weekly/monthly distinct users by **merging sketches** — no re-scanning 1B rows. This is how companies like Facebook and Netflix compute MAU at scale.

---

### Variant 3 — APPROX_QUANTILES for latency percentiles (BigQuery)

```sql
-- BigQuery: approximate percentile profile
SELECT
    model_name,
    APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)]  AS p50,
    APPROX_QUANTILES(latency_ms, 100)[OFFSET(90)]  AS p90,
    APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)]  AS p95,
    APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)]  AS p99,
    APPROX_QUANTILES(latency_ms, 100)[OFFSET(100)] AS p100_max
FROM model_inference_log
GROUP BY model_name;
```

> `APPROX_QUANTILES(col, 100)` returns an array of 101 values (0th through 100th percentile). `[OFFSET(99)]` is the 99th percentile.

---

### Variant 4 — APPROX_TOP_K (most common feature values, BigQuery)

```sql
-- Find the 5 most common values for a feature (approximate)
SELECT
    feature_name,
    APPROX_TOP_COUNT(feature_value, 5) AS top_5_values
FROM feature_logs
GROUP BY feature_name;
```

**Output:**

| feature_name | top_5_values |
|--------------|--------------|
| country      | `[{value: "US", count: 4200000}, {value: "UK", count: 1100000}, ...]` |
| device_type  | `[{value: "mobile", count: 6500000}, {value: "desktop", count: 3200000}, ...]` |

---

### Variant 5 — Exact vs approximate: when to use which

```sql
-- Decision rule: use approximate when n > ~10M rows AND error < 1-2% is acceptable

-- For ML feature cardinality check (exact not needed, just "high" or "low"):
SELECT
    column_name,
    APPROX_COUNT_DISTINCT(value) AS approx_cardinality,
    CASE
        WHEN APPROX_COUNT_DISTINCT(value) > 100  THEN 'high_cardinality'
        WHEN APPROX_COUNT_DISTINCT(value) > 10   THEN 'medium_cardinality'
        ELSE 'low_cardinality'
    END                          AS cardinality_label
FROM feature_values
GROUP BY column_name
ORDER BY approx_cardinality DESC;
```

---

### Variant 6 — Redshift APPROXIMATE PERCENTILE_DISC

```sql
-- Redshift
SELECT
    model_name,
    APPROXIMATE PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
    APPROXIMATE PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
    APPROXIMATE PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99
FROM model_inference_log
GROUP BY model_name;
```

---

### The HyperLogLog Algorithm — What the Interviewer Wants You to Know

```
HyperLogLog estimates COUNT(DISTINCT) using:
1. Hash each value to a binary string
2. Track the maximum number of leading zeros seen across all hashes
3. More leading zeros → more distinct values seen
4. Use many hash buckets (registers) + harmonic mean to reduce variance

Key properties:
- Memory: fixed ~1.5 KB per sketch regardless of data size
- Error:  ±2% with 2048 registers (standard)
- Merge:  UNION of two HLL sketches = HLL of the combined data
- Idempotent: hashing the same value twice doesn't change the sketch
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using approximate for financial/legal counts | 2% error on $100M transaction count = $2M off | Use exact COUNT(DISTINCT) for regulated metrics |
| APPROX_COUNT_DISTINCT error on small n | On 100 rows, 2% relative error is just 2 users — fine. On 100 rows where true answer is 50, it may be off by 10% | For small tables, exact is fast anyway — no need to approximate |
| HLL merge requires same hash seed | Merging HLL sketches built with different hash functions gives wrong results | Always use the same hashing function across all sketches |
| APPROX_QUANTILES array offset confusion | BigQuery: `[OFFSET(99)]` is the 99th percentile (0-indexed, 100 buckets) | `APPROX_QUANTILES(col, 100)[OFFSET(p)]` = pth percentile |
| Not available in PostgreSQL natively | `APPROX_COUNT_DISTINCT` doesn't exist in vanilla PG | Install `hll` extension or use `pg_stats` estimates |
| Sketch-based counts change on recompute | Approximate results are non-deterministic across runs (different hash seeds) | Set a fixed seed if reproducibility matters |

---

## Module 6 — Complete Cheat Sheet

### Data Types Quick Reference

| Need | Type | Notes |
|------|------|-------|
| Whole numbers up to 2.1B | `INTEGER` | Default for counts, IDs |
| Large IDs, row counts | `BIGINT` | Auto-increment PKs on big tables |
| Money / exact decimals | `NUMERIC(p,s)` | Never use FLOAT for money |
| ML scores, ratios | `FLOAT` / `DOUBLE PRECISION` | 2% imprecision at 15th decimal — fine for ML |
| Short text | `VARCHAR(n)` | Use TEXT in PostgreSQL; VARCHAR in MySQL |
| Long text | `TEXT` | Unlimited in PostgreSQL |
| Date only | `DATE` | `'2024-01-15'::DATE` |
| Date + time (global product) | `TIMESTAMPTZ` | Always store UTC |
| True/false | `BOOLEAN` | `true`, `false`, `NULL` |
| Unique identifiers | `UUID` | `gen_random_uuid()` |
| Semi-structured config | `JSONB` | Index with GIN; use `->>` to extract |
| Feature vectors | `FLOAT[]` | Native array; 1-indexed |

---

### Cast Pattern Reference

```sql
col::FLOAT                    -- string/int to float
col::INTEGER                  -- truncates decimal
col::TEXT                     -- any type to string
col::DATE                     -- string to date ('YYYY-MM-DD')
col::TIMESTAMP                -- string to timestamp
col::BOOLEAN                  -- 'true'/'false' string to bool
col::UUID                     -- string to UUID
col::JSONB                    -- text to JSONB
col::FLOAT[]                  -- text representation to float array
CAST(5 AS FLOAT) / 2         -- safe division
```

---

### JSON Operator Reference

```sql
col->>'key'                   -- extract text value
col->'key'                    -- extract JSON sub-object
col#>>'{a,b,c}'               -- extract nested text
col @> '{"key":"val"}'        -- contains (uses GIN index)
col ? 'key'                   -- key exists
jsonb_each(col)               -- explode to (key, value) rows
jsonb_array_elements(col)     -- explode JSON array to rows
jsonb_array_elements_text(col) -- same, returns text
jsonb_build_object('k',v,...) -- build JSONB from key-value pairs
jsonb_agg(expr)               -- aggregate rows into JSON array
jsonb_object_keys(col)        -- get all keys
```

---

### LATERAL Pattern Reference

```sql
-- Top-N per group
FROM left l
JOIN LATERAL (SELECT ... FROM right r WHERE r.id = l.id ORDER BY x LIMIT N) sub ON TRUE

-- Latest per group
FROM left l
LEFT JOIN LATERAL (SELECT ... FROM right r WHERE r.id = l.id ORDER BY ts DESC LIMIT 1) sub ON TRUE

-- Explode JSONB per row
FROM table t, jsonb_each(t.config) AS kv(key, value)

-- Explode array per row with index
FROM table t, UNNEST(t.arr) WITH ORDINALITY AS x(val, idx)

-- SQL Server equivalent
FROM left l CROSS APPLY (SELECT TOP N ... FROM right r WHERE r.id = l.id) sub
FROM left l OUTER APPLY (SELECT TOP 1 ... FROM right r WHERE r.id = l.id) sub
```

---

### Approximate vs Exact — Decision Guide

| Situation | Use |
|-----------|-----|
| n < 10M rows | Exact — it's fast enough |
| n > 100M, error < 2% acceptable | `APPROX_COUNT_DISTINCT` |
| Need to merge daily → weekly distinct users | HyperLogLog sketches |
| Latency P50/P90/P99 at scale | `APPROX_QUANTILES` (BigQuery) / `APPROXIMATE PERCENTILE_DISC` (Redshift) |
| Feature cardinality check for ML preprocessing | Approximate — exact overkill |
| Revenue metrics, legal compliance | Always exact |
| Real-time dashboards on petabyte scale | HLL + pre-aggregated sketch tables |

---

