# SQL for ML Interviews — Window Functions: Complete Module

> **Module 3** — Every major window function pattern. Same format: concept → variants with input/output → pitfalls. No other source needed.

---

## Table of Contents

25. [Window Function Anatomy — The OVER() Clause](#25-window-function-anatomy--the-over-clause)
26. [Ranking Functions — ROW_NUMBER, RANK, DENSE_RANK, NTILE](#26-ranking-functions--row_number-rank-dense_rank-ntile)
27. [Top-N Per Group Pattern](#27-top-n-per-group-pattern)
28. [LAG and LEAD — Accessing Adjacent Rows](#28-lag-and-lead--accessing-adjacent-rows)
29. [Running Totals and Cumulative Aggregates](#29-running-totals-and-cumulative-aggregates)
30. [Window Frame Clauses — ROWS vs RANGE, BETWEEN](#30-window-frame-clauses--rows-vs-range-between)
31. [Moving Averages and Rolling Windows](#31-moving-averages-and-rolling-windows)
32. [Percentiles — PERCENTILE_CONT, PERCENTILE_DISC, PERCENT_RANK, CUME_DIST](#32-percentiles--percentile_cont-percentile_disc-percent_rank-cume_dist)
33. [Median](#33-median)
34. [FIRST_VALUE, LAST_VALUE, NTH_VALUE](#34-first_value-last_value-nth_value)
35. [Gaps and Islands Problem](#35-gaps-and-islands-problem)
36. [Session Detection with Window Functions](#36-session-detection-with-window-functions)
37. [Year-over-Year and Period Comparisons](#37-year-over-year-and-period-comparisons)

---

## 25. Window Function Anatomy — The OVER() Clause

### Concept

A **window function** computes a value across a set of rows **related to the current row** — without collapsing them into a single output row like GROUP BY does. Every window function uses an `OVER()` clause.

```sql
function_name(column)
    OVER (
        PARTITION BY  col1, col2     -- divide rows into independent groups (optional)
        ORDER BY      col3 DESC      -- sort rows within each partition (optional)
        ROWS BETWEEN  ... AND ...    -- define the frame (optional, has a default)
    )
```

**The three parts:**

| Part | What it controls | Optional? |
|------|-----------------|-----------|
| `PARTITION BY` | Resets the window for each unique group — like GROUP BY but without collapsing rows | Yes |
| `ORDER BY` | Defines row order within the partition — required for ranking, running totals, LAG/LEAD | For most functions |
| Frame (`ROWS`/`RANGE BETWEEN`) | Which rows around the current row are included in the calculation | Yes (has default) |

**Key insight:** Window functions run **after** WHERE, GROUP BY, and HAVING but **before** the final SELECT output. You cannot filter on a window function result in WHERE — use a CTE or subquery.

---

### Sample Table: `daily_model_metrics`

| date       | model_name   | accuracy | requests |
|------------|--------------|----------|----------|
| 2024-01-01 | XGBoost      | 0.910    | 1200     |
| 2024-01-01 | LightGBM     | 0.940    | 800      |
| 2024-01-02 | XGBoost      | 0.905    | 1300     |
| 2024-01-02 | LightGBM     | 0.935    | 850      |
| 2024-01-03 | XGBoost      | 0.920    | 1100     |
| 2024-01-03 | LightGBM     | 0.945    | 900      |
| 2024-01-04 | XGBoost      | 0.915    | 1250     |
| 2024-01-04 | LightGBM     | 0.930    | 780      |

---

### Anatomy Query — showing all three OVER parts together

```sql
SELECT
    date,
    model_name,
    accuracy,
    -- No PARTITION: global rank across all rows
    RANK() OVER (ORDER BY accuracy DESC)                          AS global_rank,

    -- PARTITION only: count of days per model (same for all rows of that model)
    COUNT(*) OVER (PARTITION BY model_name)                       AS model_day_count,

    -- PARTITION + ORDER: cumulative max accuracy per model
    MAX(accuracy) OVER (
        PARTITION BY model_name
        ORDER BY date
    )                                                             AS cumulative_best
FROM daily_model_metrics
ORDER BY date, model_name;
```

**Output:**

| date       | model_name | accuracy | global_rank | model_day_count | cumulative_best |
|------------|------------|----------|-------------|-----------------|-----------------|
| 2024-01-01 | LightGBM   | 0.940    | 2           | 4               | 0.940           |
| 2024-01-01 | XGBoost    | 0.910    | 6           | 4               | 0.910           |
| 2024-01-02 | LightGBM   | 0.935    | 4           | 4               | 0.940           |
| 2024-01-02 | XGBoost    | 0.905    | 8           | 4               | 0.910           |
| 2024-01-03 | LightGBM   | 0.945    | 1           | 4               | 0.945           |
| 2024-01-03 | XGBoost    | 0.920    | 5           | 4               | 0.920           |
| 2024-01-04 | LightGBM   | 0.930    | 3           | 4               | 0.945           |
| 2024-01-04 | XGBoost    | 0.915    | 7           | 4               | 0.920           |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Filtering on window result in WHERE | `WHERE RANK() OVER(...) = 1` is a syntax error | Wrap in CTE: `WITH r AS (...) SELECT * FROM r WHERE rank = 1` |
| Forgetting PARTITION BY | Window spans the entire table instead of per-group | Add `PARTITION BY model_name` to reset per model |
| ORDER BY in OVER changes default frame | Adding ORDER BY changes the frame from ALL ROWS to RANGE UNBOUNDED PRECEDING | Be explicit with your frame clause |
| Window functions in GROUP BY queries | Window runs after grouping — sees grouped result set, not original rows | Intentional; just know the order |

---

## 26. Ranking Functions — ROW_NUMBER, RANK, DENSE_RANK, NTILE

### Concept

All four assign a number to each row within a partition. They differ only in how they handle **ties**.

| Function | Ties handled how | Gaps after ties? | Typical use |
|----------|-----------------|-----------------|-------------|
| `ROW_NUMBER()` | Arbitrarily breaks ties — every row gets a unique number | N/A | Deduplication, pagination |
| `RANK()` | Tied rows get the same rank, next rank skips | Yes (1,1,3) | Leaderboards where gaps matter |
| `DENSE_RANK()` | Tied rows get same rank, next rank does NOT skip | No (1,1,2) | When you want contiguous ranks |
| `NTILE(n)` | Divides rows into n equal buckets | N/A | Quartiles, deciles |

---

### Sample Table: `model_leaderboard`

| model_name   | f1_score |
|--------------|----------|
| LightGBM     | 0.945    |
| XGBoost_v2   | 0.920    |
| XGBoost_v1   | 0.920    |
| BERT         | 0.910    |
| ResNet       | 0.885    |
| LogisticReg  | 0.870    |

---

### Variant 1 — All four side by side

```sql
SELECT
    model_name,
    f1_score,
    ROW_NUMBER()  OVER (ORDER BY f1_score DESC) AS row_num,
    RANK()        OVER (ORDER BY f1_score DESC) AS rnk,
    DENSE_RANK()  OVER (ORDER BY f1_score DESC) AS dense_rnk,
    NTILE(3)      OVER (ORDER BY f1_score DESC) AS tercile
FROM model_leaderboard
ORDER BY f1_score DESC;
```

**Output:**

| model_name  | f1_score | row_num | rnk | dense_rnk | tercile |
|-------------|----------|---------|-----|-----------|---------|
| LightGBM    | 0.945    | 1       | 1   | 1         | 1       |
| XGBoost_v2  | 0.920    | 2       | 2   | 2         | 1       |
| XGBoost_v1  | 0.920    | 3       | 2   | 2         | 2       |
| BERT        | 0.910    | 4       | 4   | 3         | 2       |
| ResNet      | 0.885    | 5       | 5   | 4         | 3       |
| LogisticReg | 0.870    | 6       | 6   | 5         | 3       |

> RANK skips 3 (two models tied at rank 2). DENSE_RANK does not skip. ROW_NUMBER breaks ties arbitrarily.

---

### Variant 2 — RANK within a partition (per dataset type)

```sql
SELECT
    model_name,
    dataset,
    f1_score,
    RANK() OVER (PARTITION BY dataset ORDER BY f1_score DESC) AS rank_in_dataset
FROM model_scores_by_dataset
ORDER BY dataset, rank_in_dataset;
```

**Output (sample):**

| model_name | dataset | f1_score | rank_in_dataset |
|------------|---------|----------|-----------------|
| LightGBM   | test    | 0.945    | 1               |
| XGBoost    | test    | 0.910    | 2               |
| LightGBM   | train   | 0.980    | 1               |
| XGBoost    | train   | 0.960    | 2               |

---

### Variant 3 — NTILE for score quartiles

```sql
SELECT
    model_name,
    f1_score,
    NTILE(4) OVER (ORDER BY f1_score) AS quartile   -- 1=bottom, 4=top
FROM model_leaderboard;
```

**Output:**

| model_name  | f1_score | quartile |
|-------------|----------|----------|
| LogisticReg | 0.870    | 1        |
| ResNet      | 0.885    | 1        |
| BERT        | 0.910    | 2        |
| XGBoost_v1  | 0.920    | 2        |
| XGBoost_v2  | 0.920    | 3        |
| LightGBM    | 0.945    | 3        |

> Only 6 rows for NTILE(4) — larger buckets always come first.

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| ROW_NUMBER tie-breaking is non-deterministic | Two rows with same ORDER BY value may swap between runs | Add a tiebreaker: `ORDER BY f1_score DESC, model_name` |
| RANK vs DENSE_RANK confusion | "Give me rank 2" — does the questioner mean RANK or DENSE_RANK? | Clarify; default to DENSE_RANK unless gaps are meaningful |
| NTILE with uneven division | 10 rows, NTILE(3) gives buckets of 4,3,3 | Larger buckets always come first |
| Using RANK to deduplicate | RANK allows ties; dedup needs ROW_NUMBER | Always use ROW_NUMBER for dedup |

---

## 27. Top-N Per Group Pattern

### Concept

"Give me the top 3 models by accuracy for each dataset" — asked in nearly every ML data interview. The pattern is always:
1. Assign `ROW_NUMBER()` within each group
2. Wrap in a CTE
3. Filter `WHERE rn <= N`

---

### Sample Table: `experiment_results`

| exp_id | model_name   | dataset | accuracy |
|--------|--------------|---------|----------|
| 1      | XGBoost      | train   | 0.960    |
| 2      | LightGBM     | train   | 0.975    |
| 3      | BERT         | train   | 0.940    |
| 4      | ResNet       | train   | 0.930    |
| 5      | LogisticReg  | train   | 0.890    |
| 6      | XGBoost      | test    | 0.910    |
| 7      | LightGBM     | test    | 0.945    |
| 8      | BERT         | test    | 0.905    |
| 9      | ResNet       | test    | 0.880    |
| 10     | LogisticReg  | test    | 0.860    |

---

### Variant 1 — Top 2 per dataset

```sql
WITH ranked AS (
    SELECT
        exp_id,
        model_name,
        dataset,
        accuracy,
        ROW_NUMBER() OVER (
            PARTITION BY dataset
            ORDER BY accuracy DESC
        ) AS rn
    FROM experiment_results
)
SELECT exp_id, model_name, dataset, accuracy
FROM ranked
WHERE rn <= 2
ORDER BY dataset, rn;
```

**Output:**

| exp_id | model_name | dataset | accuracy |
|--------|------------|---------|----------|
| 7      | LightGBM   | test    | 0.945    |
| 6      | XGBoost    | test    | 0.910    |
| 2      | LightGBM   | train   | 0.975    |
| 1      | XGBoost    | train   | 0.960    |

---

### Variant 2 — Top-1 per group (best model per dataset)

```sql
WITH ranked AS (
    SELECT
        model_name,
        dataset,
        accuracy,
        ROW_NUMBER() OVER (PARTITION BY dataset ORDER BY accuracy DESC) AS rn
    FROM experiment_results
)
SELECT model_name, dataset, accuracy
FROM ranked
WHERE rn = 1;
```

**Output:**

| model_name | dataset | accuracy |
|------------|---------|----------|
| LightGBM   | test    | 0.945    |
| LightGBM   | train   | 0.975    |

---

### Variant 3 — Bottom N (worst performing, for triage)

```sql
WITH ranked AS (
    SELECT
        model_name,
        dataset,
        accuracy,
        ROW_NUMBER() OVER (PARTITION BY dataset ORDER BY accuracy ASC) AS rn
    FROM experiment_results
)
SELECT model_name, dataset, accuracy
FROM ranked
WHERE rn <= 2
ORDER BY dataset, accuracy;
```

**Output:**

| model_name  | dataset | accuracy |
|-------------|---------|----------|
| LogisticReg | test    | 0.860    |
| ResNet      | test    | 0.880    |
| LogisticReg | train   | 0.890    |
| ResNet      | train   | 0.930    |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using RANK instead of ROW_NUMBER | Two tied models both get rn=1; `WHERE rn=1` returns both — sometimes intentional | Use RANK if you want all ties; ROW_NUMBER for strict top-1 |
| Filtering in WHERE without CTE | `WHERE ROW_NUMBER() OVER(...) <= 2` is illegal | Always wrap in CTE or subquery first |
| Forgetting PARTITION BY | Returns global top-N instead of per-group | Double-check the PARTITION BY column |
| Non-deterministic top-1 on ties | Two models tied — which wins? | Add secondary sort: `ORDER BY accuracy DESC, model_name` |

---

## 28. LAG and LEAD — Accessing Adjacent Rows

### Concept

`LAG` and `LEAD` let you look **backwards** or **forwards** within an ordered window without a self-join.

```sql
LAG (col, offset, default)  OVER (PARTITION BY ... ORDER BY ...)
LEAD(col, offset, default)  OVER (PARTITION BY ... ORDER BY ...)
```

- `offset` — how many rows to look back/forward (default = 1)
- `default` — value when there is no previous/next row (default = NULL)

Classic uses: day-over-day change, detecting state transitions, computing inter-event time.

---

### Sample Table: `daily_model_metrics` (reused)

| date       | model_name | accuracy | requests |
|------------|------------|----------|----------|
| 2024-01-01 | XGBoost    | 0.910    | 1200     |
| 2024-01-02 | XGBoost    | 0.905    | 1300     |
| 2024-01-03 | XGBoost    | 0.920    | 1100     |
| 2024-01-04 | XGBoost    | 0.915    | 1250     |
| 2024-01-01 | LightGBM   | 0.940    | 800      |
| 2024-01-02 | LightGBM   | 0.935    | 850      |
| 2024-01-03 | LightGBM   | 0.945    | 900      |
| 2024-01-04 | LightGBM   | 0.930    | 780      |

---

### Variant 1 — Day-over-day accuracy change

```sql
SELECT
    date,
    model_name,
    accuracy,
    LAG(accuracy) OVER (
        PARTITION BY model_name
        ORDER BY date
    )                                                AS prev_day_accuracy,
    ROUND(
        accuracy - LAG(accuracy) OVER (
            PARTITION BY model_name ORDER BY date
        ), 4
    )                                                AS accuracy_delta
FROM daily_model_metrics
ORDER BY model_name, date;
```

**Output:**

| date       | model_name | accuracy | prev_day_accuracy | accuracy_delta |
|------------|------------|----------|-------------------|----------------|
| 2024-01-01 | LightGBM   | 0.940    | NULL              | NULL           |
| 2024-01-02 | LightGBM   | 0.935    | 0.940             | -0.0050        |
| 2024-01-03 | LightGBM   | 0.945    | 0.935             | +0.0100        |
| 2024-01-04 | LightGBM   | 0.930    | 0.945             | -0.0150        |
| 2024-01-01 | XGBoost    | 0.910    | NULL              | NULL           |
| 2024-01-02 | XGBoost    | 0.905    | 0.910             | -0.0050        |
| 2024-01-03 | XGBoost    | 0.920    | 0.905             | +0.0150        |
| 2024-01-04 | XGBoost    | 0.915    | 0.920             | -0.0050        |

---

### Variant 2 — LEAD to see next day's value

```sql
SELECT
    date,
    model_name,
    requests,
    LEAD(requests, 1, 0) OVER (
        PARTITION BY model_name
        ORDER BY date
    ) AS next_day_requests
FROM daily_model_metrics
ORDER BY model_name, date;
```

**Output (LightGBM):**

| date       | model_name | requests | next_day_requests |
|------------|------------|----------|-------------------|
| 2024-01-01 | LightGBM   | 800      | 850               |
| 2024-01-02 | LightGBM   | 850      | 900               |
| 2024-01-03 | LightGBM   | 900      | 780               |
| 2024-01-04 | LightGBM   | 780      | 0  (default)      |

---

### Variant 3 — Detect accuracy regression (drop > 0.01 from prior day)

```sql
WITH deltas AS (
    SELECT
        date,
        model_name,
        accuracy,
        LAG(accuracy) OVER (PARTITION BY model_name ORDER BY date) AS prev_acc
    FROM daily_model_metrics
)
SELECT
    date,
    model_name,
    ROUND(prev_acc, 4)             AS prev_accuracy,
    ROUND(accuracy, 4)             AS curr_accuracy,
    ROUND(accuracy - prev_acc, 4)  AS delta
FROM deltas
WHERE accuracy - prev_acc < -0.01
ORDER BY delta;
```

**Output:**

| date       | model_name | prev_accuracy | curr_accuracy | delta   |
|------------|------------|---------------|---------------|---------|
| 2024-01-04 | LightGBM   | 0.9450        | 0.9300        | -0.0150 |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Missing PARTITION BY | LAG looks across ALL models, mixing rows | Always PARTITION BY the entity |
| LAG with date gaps | LAG returns the previous row, not previous calendar day | Join to a date spine before using LAG |
| Using LAG result in WHERE | Can't filter in WHERE on window functions | Wrap in CTE then filter |
| Default value type mismatch | `LAG(accuracy, 1, 0)` — 0 must match float type | Use `0.0` for floats |

---

## 29. Running Totals and Cumulative Aggregates

### Concept

Running totals use this frame when ORDER BY is present:
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`

This means: accumulate everything from the first row in the partition up to and including the current row.

---

### Sample Table: `daily_requests` (XGBoost only)

| date       | model_name | requests |
|------------|------------|----------|
| 2024-01-01 | XGBoost    | 1200     |
| 2024-01-02 | XGBoost    | 1300     |
| 2024-01-03 | XGBoost    | 1100     |
| 2024-01-04 | XGBoost    | 1250     |

---

### Variant 1 — Running total of requests

```sql
SELECT
    date,
    requests,
    SUM(requests) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_requests
FROM daily_requests
WHERE model_name = 'XGBoost';
```

**Output:**

| date       | requests | cumulative_requests |
|------------|----------|---------------------|
| 2024-01-01 | 1200     | 1200                |
| 2024-01-02 | 1300     | 2500                |
| 2024-01-03 | 1100     | 3600                |
| 2024-01-04 | 1250     | 4850                |

---

### Variant 2 — Running total as % of grand total

```sql
WITH totals AS (
    SELECT
        date,
        requests,
        SUM(requests) OVER (ORDER BY date)  AS running_total,
        SUM(requests) OVER ()               AS grand_total   -- no ORDER BY = all rows
    FROM daily_requests
    WHERE model_name = 'XGBoost'
)
SELECT
    date,
    requests,
    running_total,
    ROUND(running_total * 100.0 / grand_total, 1) AS cumulative_pct
FROM totals;
```

**Output:**

| date       | requests | running_total | cumulative_pct |
|------------|----------|---------------|----------------|
| 2024-01-01 | 1200     | 1200          | 24.7           |
| 2024-01-02 | 1300     | 2500          | 51.5           |
| 2024-01-03 | 1100     | 3600          | 74.2           |
| 2024-01-04 | 1250     | 4850          | 100.0          |

---

### Variant 3 — Running max (accuracy high-water mark)

```sql
SELECT
    date,
    model_name,
    accuracy,
    MAX(accuracy) OVER (
        PARTITION BY model_name
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS best_so_far,
    accuracy = MAX(accuracy) OVER (
        PARTITION BY model_name
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS is_new_record
FROM daily_model_metrics
ORDER BY model_name, date;
```

**Output:**

| date       | model_name | accuracy | best_so_far | is_new_record |
|------------|------------|----------|-------------|---------------|
| 2024-01-01 | LightGBM   | 0.940    | 0.940       | true          |
| 2024-01-02 | LightGBM   | 0.935    | 0.940       | false         |
| 2024-01-03 | LightGBM   | 0.945    | 0.945       | true          |
| 2024-01-04 | LightGBM   | 0.930    | 0.945       | false         |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| RANGE vs ROWS on tied values | RANGE includes all rows with the same ORDER BY value — inflates sum on ties | Use `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` explicitly |
| `SUM() OVER ()` confusion | No ORDER BY = grand total on every row, not running total | Intentional; add ORDER BY for running total |
| Running total resets unexpectedly | Forgot PARTITION BY | Add `PARTITION BY model_name` |

---

## 30. Window Frame Clauses — ROWS vs RANGE, BETWEEN

### Concept

The frame clause specifies **which rows** relative to the current row are included in the calculation.

```sql
ROWS  BETWEEN <start> AND <end>   -- physical row positions
RANGE BETWEEN <start> AND <end>   -- value-based grouping (ties treated as one unit)
```

**Frame boundaries:**

| Keyword | Meaning |
|---------|---------|
| `UNBOUNDED PRECEDING` | First row of the partition |
| `N PRECEDING` | N rows before current |
| `CURRENT ROW` | Current row only |
| `N FOLLOWING` | N rows after current |
| `UNBOUNDED FOLLOWING` | Last row of the partition |

**Default when ORDER BY is present:** `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
**Default when no ORDER BY:** `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`

---

### Sample Table: `scores_with_ties`

| id | score |
|----|-------|
| 1  | 10    |
| 2  | 20    |
| 3  | 20    |
| 4  | 30    |

---

### Variant 1 — ROWS vs RANGE on tied values

```sql
SELECT
    id,
    score,
    SUM(score) OVER (ORDER BY score ROWS  BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rows_sum,
    SUM(score) OVER (ORDER BY score RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS range_sum
FROM scores_with_ties;
```

**Output:**

| id | score | rows_sum | range_sum |
|----|-------|----------|-----------|
| 1  | 10    | 10       | 10        |
| 2  | 20    | 30       | 70        |
| 3  | 20    | 50       | 70        |
| 4  | 30    | 80       | 80        |

> RANGE treats rows 2 and 3 as the same unit (same score = same "current range"). Both get 70 (10+20+20+... wait — RANGE includes all rows with score <= current, so rows 1,2,3 = 10+20+20 = 50... actually RANGE UNBOUNDED PRECEDING TO CURRENT ROW means all rows up to AND INCLUDING all ties at the current value). ROWS is strictly positional.

---

### Variant 2 — N PRECEDING: 3-row trailing window

```sql
SELECT
    date,
    requests,
    AVG(requests) OVER (
        ORDER BY date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS trailing_3day_avg
FROM daily_requests
WHERE model_name = 'XGBoost';
```

**Output:**

| date       | requests | trailing_3day_avg |
|------------|----------|-------------------|
| 2024-01-01 | 1200     | 1200.0            |
| 2024-01-02 | 1300     | 1250.0            |
| 2024-01-03 | 1100     | 1200.0            |
| 2024-01-04 | 1250     | 1216.7            |

---

### Variant 3 — Suffix sum (requests from current day to end)

```sql
SELECT
    date,
    requests,
    SUM(requests) OVER (
        ORDER BY date
        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
    ) AS requests_remaining
FROM daily_requests
WHERE model_name = 'XGBoost';
```

**Output:**

| date       | requests | requests_remaining |
|------------|----------|--------------------|
| 2024-01-01 | 1200     | 4850               |
| 2024-01-02 | 1300     | 3650               |
| 2024-01-03 | 1100     | 2350               |
| 2024-01-04 | 1250     | 1250               |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Invisible RANGE default inflates running totals on ties | Default is RANGE, not ROWS | Always write `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` explicitly |
| N PRECEDING is rows, not time | 2 PRECEDING = 2 rows back, not 2 days | Fill date gaps with a spine first |
| Frame clause requires ORDER BY | Frame without ORDER BY = error | Always pair frame with ORDER BY |
| LAST_VALUE with default frame | Returns current row, not partition's last row | Use `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` |

---

## 31. Moving Averages and Rolling Windows

### Concept

A moving average smooths time-series noise. Used in ML for monitoring accuracy drift, request volume trends, and anomaly baselines.

```sql
AVG(metric) OVER (
    PARTITION BY model_name
    ORDER BY date
    ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW   -- N-period moving average
)
```

---

### Sample Table (XGBoost daily accuracy, 7 days)

| date       | accuracy |
|------------|----------|
| 2024-01-01 | 0.910    |
| 2024-01-02 | 0.905    |
| 2024-01-03 | 0.920    |
| 2024-01-04 | 0.915    |
| 2024-01-05 | 0.900    |
| 2024-01-06 | 0.925    |
| 2024-01-07 | 0.918    |

---

### Variant 1 — 3-day moving average

```sql
SELECT
    date,
    accuracy,
    ROUND(AVG(accuracy) OVER (
        PARTITION BY model_name
        ORDER BY date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 4) AS ma_3day
FROM daily_model_metrics
WHERE model_name = 'XGBoost'
ORDER BY date;
```

**Output:**

| date       | accuracy | ma_3day |
|------------|----------|---------|
| 2024-01-01 | 0.910    | 0.9100  |
| 2024-01-02 | 0.905    | 0.9075  |
| 2024-01-03 | 0.920    | 0.9117  |
| 2024-01-04 | 0.915    | 0.9133  |
| 2024-01-05 | 0.900    | 0.9117  |
| 2024-01-06 | 0.925    | 0.9133  |
| 2024-01-07 | 0.918    | 0.9143  |

---

### Variant 2 — Exclude warm-up rows (only full windows)

```sql
WITH ma AS (
    SELECT
        date,
        accuracy,
        AVG(accuracy) OVER (
            ORDER BY date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS ma_3day,
        COUNT(*) OVER (
            ORDER BY date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS window_size
    FROM daily_model_metrics
    WHERE model_name = 'XGBoost'
)
SELECT date, accuracy, ROUND(ma_3day, 4) AS ma_3day
FROM ma
WHERE window_size = 3;
```

**Output:**

| date       | accuracy | ma_3day |
|------------|----------|---------|
| 2024-01-03 | 0.920    | 0.9117  |
| 2024-01-04 | 0.915    | 0.9133  |
| 2024-01-05 | 0.900    | 0.9117  |
| 2024-01-06 | 0.925    | 0.9133  |
| 2024-01-07 | 0.918    | 0.9143  |

---

### Variant 3 — Anomaly detection: today vs prior 7-day baseline

```sql
WITH ma AS (
    SELECT
        date,
        model_name,
        accuracy,
        AVG(accuracy) OVER (
            PARTITION BY model_name
            ORDER BY date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING  -- exclude current row from baseline
        ) AS prior_7day_avg
    FROM daily_model_metrics
)
SELECT
    date,
    model_name,
    accuracy,
    ROUND(prior_7day_avg, 4)              AS baseline,
    ROUND(accuracy - prior_7day_avg, 4)   AS deviation,
    CASE
        WHEN accuracy < prior_7day_avg - 0.01 THEN 'ALERT: drop'
        WHEN accuracy > prior_7day_avg + 0.01 THEN 'ALERT: spike'
        ELSE 'normal'
    END                                   AS status
FROM ma
WHERE prior_7day_avg IS NOT NULL;
```

> This is the **ML model monitoring** pattern — detect drift by comparing today to a rolling baseline that explicitly excludes today (`1 PRECEDING` as upper bound).

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Partial windows at series start | AVG of 1-2 rows looks smoothed but isn't | Filter `WHERE window_size = N` or document warm-up period |
| Gaps in date series | 3 PRECEDING = 3 rows, may span 5 calendar days | Fill date gaps with a spine before computing MAs |
| Including current row in prior baseline | Baseline inflated by the value you're comparing to | Use `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` |

---

## 32. Percentiles — PERCENTILE_CONT, PERCENTILE_DISC, PERCENT_RANK, CUME_DIST

### Concept

| Function | Type | What it returns |
|----------|------|----------------|
| `PERCENTILE_CONT(p)` | Ordered-set aggregate | Interpolated value at percentile p — may not be in the data |
| `PERCENTILE_DISC(p)` | Ordered-set aggregate | Nearest actual data value at or above percentile p |
| `PERCENT_RANK()` | Window function | Relative rank: `(rank - 1) / (n - 1)` → [0, 1] |
| `CUME_DIST()` | Window function | Fraction of rows <= current row → (0, 1] |

---

### Sample Table: `model_inference_latency`

| request_id | model_name | latency_ms |
|------------|------------|------------|
| 1          | XGBoost    | 12         |
| 2          | XGBoost    | 15         |
| 3          | XGBoost    | 18         |
| 4          | XGBoost    | 22         |
| 5          | XGBoost    | 25         |
| 6          | XGBoost    | 30         |
| 7          | XGBoost    | 45         |
| 8          | XGBoost    | 120        |

---

### Variant 1 — PERCENTILE_CONT vs DISC, P50/P90/P99

```sql
SELECT
    model_name,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_cont,
    PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_disc,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms) AS p90,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99
FROM model_inference_latency
GROUP BY model_name;
```

**Output:**

| model_name | p50_cont | p50_disc | p90   | p95    | p99    |
|------------|----------|----------|-------|--------|--------|
| XGBoost    | 23.5     | 22       | 67.5  | 93.75  | 116.25 |

> p50_cont = 23.5 (interpolated between 22 and 25). p50_disc = 22 (actual row at the 50th percentile position).

---

### Variant 2 — PERCENT_RANK and CUME_DIST as window functions

```sql
SELECT
    request_id,
    latency_ms,
    ROUND(PERCENT_RANK() OVER (ORDER BY latency_ms), 3) AS pct_rank,
    ROUND(CUME_DIST()    OVER (ORDER BY latency_ms), 3) AS cume_dist
FROM model_inference_latency
WHERE model_name = 'XGBoost'
ORDER BY latency_ms;
```

**Output:**

| request_id | latency_ms | pct_rank | cume_dist |
|------------|------------|----------|-----------|
| 1          | 12         | 0.000    | 0.125     |
| 2          | 15         | 0.143    | 0.250     |
| 3          | 18         | 0.286    | 0.375     |
| 4          | 22         | 0.429    | 0.500     |
| 5          | 25         | 0.571    | 0.625     |
| 6          | 30         | 0.714    | 0.750     |
| 7          | 45         | 0.857    | 0.875     |
| 8          | 120        | 1.000    | 1.000     |

> PERCENT_RANK of first row is always 0. CUME_DIST is always > 0.

---

### Variant 3 — Flag P95 SLA breaches (portable approach)

```sql
WITH p95 AS (
    SELECT
        model_name,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency
    FROM model_inference_latency
    GROUP BY model_name
)
SELECT
    r.request_id,
    r.model_name,
    r.latency_ms,
    ROUND(p.p95_latency, 1)        AS p95_threshold,
    r.latency_ms > p.p95_latency  AS sla_breach
FROM model_inference_latency r
JOIN p95 p ON r.model_name = p.model_name
ORDER BY r.latency_ms DESC;
```

**Output:**

| request_id | model_name | latency_ms | p95_threshold | sla_breach |
|------------|------------|------------|---------------|------------|
| 8          | XGBoost    | 120        | 93.8          | true       |
| 7          | XGBoost    | 45         | 93.8          | false      |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| CONT vs DISC confusion | CONT may return a value not in the data; DISC always returns an actual value | For latency SLAs, CONT is standard |
| `WITHIN GROUP` vs `OVER()` | PERCENTILE_CONT uses `WITHIN GROUP`, not `OVER` | These are different syntax paths entirely |
| PERCENT_RANK starts at 0, CUME_DIST never reaches 0 | PERCENT_RANK = (rank-1)/(n-1); CUME_DIST = rank/n | Know the formulas |
| Percentile without PARTITION BY | Returns global percentile | Add PARTITION BY or compute per group in CTE |

---

## 33. Median

### Concept

SQL has no universal `MEDIAN()` function. Three approaches:

| Approach | DB | Notes |
|----------|----|-------|
| `PERCENTILE_CONT(0.5)` | PostgreSQL, Redshift, BigQuery | Standard, interpolated |
| `MEDIAN()` | Oracle, some others | Direct function |
| Manual ROW_NUMBER approach | All DBs | Verbose but portable |

---

### Variant 1 — Standard median + mean vs median gap (skew detection)

```sql
SELECT
    model_name,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS median_latency,
    ROUND(AVG(latency_ms), 2)                               AS mean_latency,
    ROUND(
        AVG(latency_ms) -
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms),
        2
    )                                                        AS mean_minus_median
FROM model_inference_latency
GROUP BY model_name;
```

**Output:**

| model_name | median_latency | mean_latency | mean_minus_median |
|------------|----------------|--------------|-------------------|
| XGBoost    | 23.5           | 35.88        | 12.38             |

> Large positive gap: mean > median → **right-skewed distribution** (long-tail latency from outliers). Critical insight for ML serving.

---

### Variant 2 — Full latency percentile profile per model

```sql
SELECT
    model_name,
    COUNT(*)                                                         AS n,
    ROUND(MIN(latency_ms))                                           AS min_ms,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY latency_ms)) AS p25,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms)) AS p50,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY latency_ms)) AS p75,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms)) AS p90,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms)) AS p99,
    ROUND(MAX(latency_ms))                                           AS max_ms
FROM model_inference_latency
GROUP BY model_name;
```

---

### Variant 3 — Manual median (portable, works everywhere)

```sql
WITH ordered AS (
    SELECT
        latency_ms,
        ROW_NUMBER() OVER (ORDER BY latency_ms) AS rn,
        COUNT(*)     OVER ()                    AS total
    FROM model_inference_latency
    WHERE model_name = 'XGBoost'
)
SELECT AVG(latency_ms) AS median
FROM ordered
WHERE rn IN (
    FLOOR((total + 1) / 2.0),
    CEIL( (total + 1) / 2.0)
);
```

**Output:**

| median |
|--------|
| 23.5   |

> FLOOR and CEIL pick the same row for odd N, two middle rows for even N. AVG handles both cases.

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using AVG as a proxy for median | Outliers skew the mean significantly | Always compute both; report median for latency and income distributions |
| Manual median for even N | Floor and Ceil must pick two different rows | Verify: 8 rows → positions 4 and 5 |
| BigQuery median | No PERCENTILE_CONT in standard BigQuery | Use `APPROX_QUANTILES(col, 100)[OFFSET(50)]` |

---

## 34. FIRST_VALUE, LAST_VALUE, NTH_VALUE

### Concept

These pull a value from a **specific row** within the window frame:
- `FIRST_VALUE(col)` — value from the first row of the frame
- `LAST_VALUE(col)` — value from the last row of the frame
- `NTH_VALUE(col, n)` — value from the nth row

**Critical:** `LAST_VALUE` with the default frame (`RANGE ... CURRENT ROW`) returns the **current row's value**, not the partition's last row. Always extend the frame.

---

### Variant 1 — Gap from personal best (each day vs model's best day)

```sql
SELECT
    date,
    model_name,
    accuracy,
    FIRST_VALUE(accuracy) OVER (
        PARTITION BY model_name
        ORDER BY accuracy DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                                             AS best_accuracy,
    ROUND(
        accuracy / FIRST_VALUE(accuracy) OVER (
            PARTITION BY model_name
            ORDER BY accuracy DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) - 1,
        4
    )                                                             AS gap_from_best
FROM daily_model_metrics
ORDER BY model_name, date;
```

**Output:**

| date       | model_name | accuracy | best_accuracy | gap_from_best |
|------------|------------|----------|---------------|---------------|
| 2024-01-01 | LightGBM   | 0.940    | 0.945         | -0.0053       |
| 2024-01-02 | LightGBM   | 0.935    | 0.945         | -0.0106       |
| 2024-01-03 | LightGBM   | 0.945    | 0.945         | 0.0000        |
| 2024-01-04 | LightGBM   | 0.930    | 0.945         | -0.0159       |

---

### Variant 2 — LAST_VALUE (must extend frame)

```sql
SELECT
    date,
    model_name,
    accuracy,
    LAST_VALUE(accuracy) OVER (
        PARTITION BY model_name
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- critical!
    ) AS latest_accuracy
FROM daily_model_metrics
ORDER BY model_name, date;
```

---

### Variant 3 — NTH_VALUE: second-best accuracy

```sql
SELECT DISTINCT
    model_name,
    FIRST_VALUE(accuracy) OVER w  AS best,
    NTH_VALUE(accuracy, 2) OVER w AS second_best
FROM daily_model_metrics
WINDOW w AS (
    PARTITION BY model_name
    ORDER BY accuracy DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
);
```

> The `WINDOW` keyword lets you name a reusable window definition — cleaner than repeating the OVER clause.

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| LAST_VALUE with default frame | Returns current row's value, not partition's last | Always add `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` |
| NTH_VALUE returns NULL if fewer than N rows | Partition has only 1 row, NTH_VALUE(col, 2) = NULL | Use COALESCE or check count first |
| FIRST_VALUE ORDER BY direction | `ORDER BY accuracy DESC` gives best; `ASC` gives worst | Confirm direction matches intent |

---

## 35. Gaps and Islands Problem

### Concept

**Islands** = groups of consecutive values. **Gaps** = breaks between islands. The canonical technique subtracts `ROW_NUMBER()` from the sequence value — consecutive integers produce the same constant (the island's group ID).

```
date values:  Jan1, Jan2, Jan3,  Jan5, Jan6,  Jan9, Jan10
row_number:   1,    2,    3,     4,    5,     6,    7
date - rn:    Dec31,Dec31,Dec31, Jan1, Jan1,  Jan3, Jan3
              ← island 1 ──────► ← island 2 ►  ← island 3 ►
```

---

### Sample Table: `model_uptime_days`

| date       | model_name | status |
|------------|------------|--------|
| 2024-01-01 | XGBoost    | up     |
| 2024-01-02 | XGBoost    | up     |
| 2024-01-03 | XGBoost    | up     |
| 2024-01-05 | XGBoost    | up     |
| 2024-01-06 | XGBoost    | up     |
| 2024-01-09 | XGBoost    | up     |
| 2024-01-10 | XGBoost    | up     |

---

### Variant 1 — Identify islands (consecutive uptime periods)

```sql
WITH numbered AS (
    SELECT
        date,
        model_name,
        ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY date) AS rn
    FROM model_uptime_days
),
grouped AS (
    SELECT
        date,
        model_name,
        (date - rn * INTERVAL '1 day')::DATE AS grp   -- constant within each island
    FROM numbered
)
SELECT
    model_name,
    MIN(date) AS island_start,
    MAX(date) AS island_end,
    COUNT(*)  AS days_in_island
FROM grouped
GROUP BY model_name, grp
ORDER BY island_start;
```

**Output:**

| model_name | island_start | island_end | days_in_island |
|------------|--------------|------------|----------------|
| XGBoost    | 2024-01-01   | 2024-01-03 | 3              |
| XGBoost    | 2024-01-05   | 2024-01-06 | 2              |
| XGBoost    | 2024-01-09   | 2024-01-10 | 2              |

---

### Variant 2 — Find the gaps (downtime windows)

```sql
WITH islands AS (
    WITH numbered AS (
        SELECT
            date,
            ROW_NUMBER() OVER (ORDER BY date) AS rn
        FROM model_uptime_days
        WHERE model_name = 'XGBoost'
    )
    SELECT
        MIN(date) AS island_start,
        MAX(date) AS island_end,
        (date - rn * INTERVAL '1 day')::DATE AS grp
    FROM numbered
    GROUP BY grp
)
SELECT
    LAG(island_end) OVER (ORDER BY island_start) + INTERVAL '1 day' AS gap_start,
    island_start - INTERVAL '1 day'                                  AS gap_end,
    island_start - LAG(island_end) OVER (ORDER BY island_start) - 1 AS gap_days
FROM islands
WHERE LAG(island_end) OVER (ORDER BY island_start) IS NOT NULL;
```

**Output:**

| gap_start  | gap_end    | gap_days |
|------------|------------|----------|
| 2024-01-04 | 2024-01-04 | 1        |
| 2024-01-07 | 2024-01-08 | 2        |

---

### Variant 3 — Integer gap detection (missing experiment IDs)

```sql
-- Find missing IDs between min and max
WITH all_ids AS (
    SELECT GENERATE_SERIES(
        (SELECT MIN(exp_id) FROM experiments),
        (SELECT MAX(exp_id) FROM experiments)
    ) AS exp_id
)
SELECT a.exp_id AS missing_id
FROM all_ids a
LEFT JOIN experiments e ON a.exp_id = e.exp_id
WHERE e.exp_id IS NULL;
```

> `GENERATE_SERIES` is PostgreSQL. BigQuery: `UNNEST(GENERATE_ARRAY(min, max))`.

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Date arithmetic type issues | `date - rn` in PG gives interval not date | Cast: `(date - rn * INTERVAL '1 day')::DATE` |
| Missing PARTITION BY | Islands bleed across models | Always PARTITION BY entity |
| Non-1 step sequences | Works only for step=1; step-N sequences need `value/step - rn` | Adjust arithmetic to normalise step size |
| GENERATE_SERIES portability | PG-only | BigQuery: `GENERATE_ARRAY`; others: use a numbers table |

---

## 36. Session Detection with Window Functions

### Concept

Group user events into sessions where any gap > threshold (e.g., 30 minutes) starts a new session.

**Algorithm:**
1. LAG to get the previous event timestamp per user
2. Flag rows where gap > threshold as session start (= 1)
3. Cumulative SUM of those flags = session number

---

### Sample Table: `user_events`

| event_id | user_id | event_ts            | event_type |
|----------|---------|---------------------|------------|
| 1        | u1      | 2024-01-15 10:00:00 | click      |
| 2        | u1      | 2024-01-15 10:05:00 | click      |
| 3        | u1      | 2024-01-15 10:45:00 | purchase   |
| 4        | u1      | 2024-01-15 10:50:00 | click      |
| 5        | u2      | 2024-01-15 11:00:00 | click      |
| 6        | u2      | 2024-01-15 11:10:00 | click      |

---

### Variant 1 — Assign session IDs (30-minute timeout)

```sql
WITH lagged AS (
    SELECT
        event_id,
        user_id,
        event_ts,
        event_type,
        LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts
    FROM user_events
),
flagged AS (
    SELECT
        *,
        CASE
            WHEN prev_ts IS NULL                                          THEN 1
            WHEN EXTRACT(EPOCH FROM (event_ts - prev_ts)) > 1800         THEN 1
            ELSE 0
        END AS is_new_session
    FROM lagged
)
SELECT
    event_id,
    user_id,
    event_ts,
    event_type,
    SUM(is_new_session) OVER (
        PARTITION BY user_id
        ORDER BY event_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS session_id
FROM flagged
ORDER BY user_id, event_ts;
```

**Output:**

| event_id | user_id | event_ts            | event_type | session_id |
|----------|---------|---------------------|------------|------------|
| 1        | u1      | 2024-01-15 10:00:00 | click      | 1          |
| 2        | u1      | 2024-01-15 10:05:00 | click      | 1          |
| 3        | u1      | 2024-01-15 10:45:00 | purchase   | 2          |
| 4        | u1      | 2024-01-15 10:50:00 | click      | 2          |
| 5        | u2      | 2024-01-15 11:00:00 | click      | 1          |
| 6        | u2      | 2024-01-15 11:10:00 | click      | 1          |

---

### Variant 2 — Session summary

```sql
WITH sessions AS (
    -- paste session assignment CTE chain here
    ...
)
SELECT
    user_id,
    session_id,
    MIN(event_ts)                                          AS session_start,
    MAX(event_ts)                                          AS session_end,
    EXTRACT(EPOCH FROM MAX(event_ts) - MIN(event_ts))      AS duration_seconds,
    COUNT(*)                                               AS event_count,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END)    AS purchases
FROM sessions
GROUP BY user_id, session_id
ORDER BY user_id, session_id;
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Session IDs are per-user, not global | user1 session 1 != user2 session 1 | Concatenate: `user_id || '-' || session_id` for global uniqueness |
| Timezone-naive timestamps | Gap calculation wrong across DST | Store all timestamps in UTC |
| Very short sessions pollute metrics | Bot traffic creates 1-event sessions | Filter: `HAVING COUNT(*) > 1` or `duration_seconds > 0` |

---

## 37. Year-over-Year and Period Comparisons

### Concept

Two approaches to period comparisons:
1. **LAG with offset** — `LAG(metric, 12)` over monthly data = same month last year
2. **Self-join on shifted date** — `curr.month = prev.month + INTERVAL '1 year'`

The self-join approach is safer when data has gaps (missing months break the LAG offset).

---

### Sample Table: `monthly_model_requests`

| month      | model_name | requests |
|------------|------------|----------|
| 2023-01-01 | XGBoost    | 10000    |
| 2023-02-01 | XGBoost    | 11000    |
| 2023-03-01 | XGBoost    | 10500    |
| 2024-01-01 | XGBoost    | 13000    |
| 2024-02-01 | XGBoost    | 14500    |
| 2024-03-01 | XGBoost    | 13800    |

---

### Variant 1 — YoY using LAG offset 12

```sql
SELECT
    month,
    model_name,
    requests,
    LAG(requests, 12) OVER (
        PARTITION BY model_name
        ORDER BY month
    )                                                        AS prev_year_requests,
    ROUND(
        (requests - LAG(requests, 12) OVER (
            PARTITION BY model_name ORDER BY month
        )) * 100.0
        / NULLIF(LAG(requests, 12) OVER (
            PARTITION BY model_name ORDER BY month
        ), 0),
        1
    )                                                        AS yoy_pct
FROM monthly_model_requests
ORDER BY model_name, month;
```

**Output:**

| month      | model_name | requests | prev_year_requests | yoy_pct |
|------------|------------|----------|--------------------|---------|
| 2023-01-01 | XGBoost    | 10000    | NULL               | NULL    |
| 2023-02-01 | XGBoost    | 11000    | NULL               | NULL    |
| 2023-03-01 | XGBoost    | 10500    | NULL               | NULL    |
| 2024-01-01 | XGBoost    | 13000    | 10000              | 30.0    |
| 2024-02-01 | XGBoost    | 14500    | 11000              | 31.8    |
| 2024-03-01 | XGBoost    | 13800    | 10500              | 31.4    |

---

### Variant 2 — YoY via self-join (handles gaps)

```sql
SELECT
    curr.month,
    curr.model_name,
    curr.requests                                 AS curr_requests,
    prev.requests                                 AS prev_year_requests,
    ROUND(
        (curr.requests - prev.requests) * 100.0
        / NULLIF(prev.requests, 0), 1
    )                                             AS yoy_pct
FROM monthly_model_requests curr
LEFT JOIN monthly_model_requests prev
    ON  curr.model_name = prev.model_name
    AND curr.month      = prev.month + INTERVAL '1 year'
ORDER BY curr.model_name, curr.month;
```

---

### Variant 3 — MoM growth rate

```sql
SELECT
    month,
    model_name,
    requests,
    LAG(requests, 1) OVER (PARTITION BY model_name ORDER BY month) AS prev_month,
    ROUND(
        (requests - LAG(requests, 1) OVER (PARTITION BY model_name ORDER BY month))
        * 100.0
        / NULLIF(LAG(requests, 1) OVER (PARTITION BY model_name ORDER BY month), 0),
        1
    )                                                               AS mom_pct
FROM monthly_model_requests
ORDER BY model_name, month;
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| LAG offset assumes no gaps | Missing month shifts all offsets | Self-join is safer for sparse data |
| Dividing by 0 when prior year is 0 | New model launch has undefined YoY | `NULLIF(prev, 0)` |
| Feb 29 date arithmetic | `2024-02-29 + INTERVAL '1 year'` errors in some DBs | Use `DATE_TRUNC('month', ...)` to normalise |

---

## Complete Window Function Cheat Sheet

### The OVER() Clause Builder

```sql
<function>() OVER (
    PARTITION BY  <group columns>           -- who resets the window?
    ORDER BY      <sort column> ASC/DESC    -- order within the partition
    ROWS BETWEEN  UNBOUNDED PRECEDING       -- frame start
              AND CURRENT ROW              -- frame end
)
```

---

### Function Reference

| Category | Function | Key Note |
|----------|----------|----------|
| **Ranking** | `ROW_NUMBER()` | Unique number per row; no args |
| | `RANK()` | Tied rows share rank; gaps follow |
| | `DENSE_RANK()` | Tied rows share rank; no gaps |
| | `NTILE(n)` | Splits into n buckets |
| **Navigation** | `LAG(col, n, default)` | n rows back |
| | `LEAD(col, n, default)` | n rows forward |
| | `FIRST_VALUE(col)` | First row in frame |
| | `LAST_VALUE(col)` | Last row — always extend frame! |
| | `NTH_VALUE(col, n)` | nth row in frame |
| **Aggregates** | `SUM / AVG / MIN / MAX / COUNT` | All work as window functions |
| **Distribution** | `PERCENT_RANK()` | [0,1]; first row = 0 |
| | `CUME_DIST()` | (0,1]; never 0 |
| **Ordered-set** | `PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY col)` | Interpolated; uses WITHIN GROUP not OVER |
| | `PERCENTILE_DISC(p) WITHIN GROUP (ORDER BY col)` | Actual value from data |

---

### Frame Quick Reference

| Use Case | Frame |
|----------|-------|
| Running total | `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` |
| Grand total on every row | `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` |
| 7-day trailing window | `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` |
| Centered 3-row window | `ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING` |
| Remaining total | `ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING` |
| Prior rows only (exclude current) | `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` |

---

### Interview Pattern Index

| Asked for | Pattern |
|-----------|---------|
| Top N per group | `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) <= N` |
| Day-over-day change | `value - LAG(value) OVER (PARTITION BY ... ORDER BY date)` |
| Running total | `SUM(col) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` |
| 7-day moving average | `AVG(col) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` |
| Column as % of total | `col * 1.0 / SUM(col) OVER ()` |
| Rank within group | `DENSE_RANK() OVER (PARTITION BY group ORDER BY metric DESC)` |
| Median / P95 latency | `PERCENTILE_CONT(0.5 / 0.95) WITHIN GROUP (ORDER BY col)` |
| Consecutive streak length | Islands: `(date - rn * INTERVAL '1 day')::DATE` as group key |
| Session ID | `SUM(is_new_session_flag) OVER (PARTITION BY user ORDER BY ts)` |
| YoY growth | `LAG(metric, 12) OVER (PARTITION BY entity ORDER BY month)` or self-join |
| Anomaly vs baseline | `value vs AVG() OVER (... ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)` |
| High-water mark | `MAX(metric) OVER (PARTITION BY ... ORDER BY date ROWS UNBOUNDED PRECEDING...)` |
| Skew detection | `AVG(col) - PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)` |

---

*End of Module 3 — Window Functions.*
*Module 4: Data Modelling for ML — fact/dimension tables, slowly changing dimensions, feature stores, event schema design.*
