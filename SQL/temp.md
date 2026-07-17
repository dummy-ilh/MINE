# SQL for ML Interviews — Window Functions: Complete Module

> **Master Edition** — Merged from all window-function study material into one file. Same format throughout: **concept → variants with input/output → pitfalls**. No other source needed.

---

## Table of Contents

1. [Window Function Anatomy — The OVER() Clause](#1-window-function-anatomy--the-over-clause)
2. [Ranking Functions — ROW_NUMBER, RANK, DENSE_RANK, NTILE](#2-ranking-functions--row_number-rank-dense_rank-ntile)
3. [Top-N Per Group Pattern](#3-top-n-per-group-pattern)
4. [LAG and LEAD — Accessing Adjacent Rows](#4-lag-and-lead--accessing-adjacent-rows)
5. [Running Totals and Cumulative Aggregates](#5-running-totals-and-cumulative-aggregates)
6. [Window Frame Clauses — ROWS vs RANGE, BETWEEN](#6-window-frame-clauses--rows-vs-range-between)
7. [Moving Averages and Rolling Windows](#7-moving-averages-and-rolling-windows)
8. [Weighted Averages](#8-weighted-averages)
9. [Percentiles — PERCENTILE_CONT, PERCENTILE_DISC, PERCENT_RANK, CUME_DIST](#9-percentiles--percentile_cont-percentile_disc-percent_rank-cume_dist)
10. [Median](#10-median)
11. [FIRST_VALUE, LAST_VALUE, NTH_VALUE](#11-first_value-last_value-nth_value)
12. [Gaps and Islands Problem](#12-gaps-and-islands-problem)
13. [Session Detection with Window Functions](#13-session-detection-with-window-functions)
14. [Year-over-Year and Period Comparisons](#14-year-over-year-and-period-comparisons)
15. [Advanced FAANG Patterns — Attribution, Threshold Crossing, Rolling Active Users](#15-advanced-faang-patterns--attribution-threshold-crossing-rolling-active-users)
16. [Deduplication via Window Functions](#16-deduplication-via-window-functions)
17. [Pivot / Unpivot (Bonus — Adjacent Skill)](#17-pivot--unpivot-bonus--adjacent-skill)
18. [Complete Cheat Sheet](#18-complete-cheat-sheet)
19. [Practice Questions (Curated, with Answers)](#19-practice-questions-curated-with-answers)

---

## 1. Window Function Anatomy — The OVER() Clause

### Concept

A **window function** computes a value across a set of rows **related to the current row** — without collapsing them into a single output row the way `GROUP BY` does. Every window function uses an `OVER()` clause.

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
| `PARTITION BY` | Resets the window for each unique group — like `GROUP BY` but without collapsing rows | Yes |
| `ORDER BY` | Defines row order within the partition — required for ranking, running totals, LAG/LEAD | For most functions |
| Frame (`ROWS`/`RANGE BETWEEN`) | Which rows around the current row are included in the calculation | Yes (has default) |

**Key insight:** Window functions run **after** `WHERE`, `GROUP BY`, and `HAVING` but **before** the final `SELECT` output ordering. You cannot filter on a window function result directly in `WHERE` — wrap it in a CTE or subquery.

**GROUP BY vs window function, side by side:**

```sql
-- GROUP BY collapses rows — you lose individual row detail
SELECT department, AVG(salary) FROM employees GROUP BY department;
-- Result: 1 row per department

-- Window function keeps ALL rows + adds the aggregate alongside
SELECT name, department, salary,
  AVG(salary) OVER (PARTITION BY department) AS dept_avg
FROM employees;
-- Result: every employee row + their dept avg on the same line
```

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

### Variant 1 — PARTITION BY only (aggregate per group, all rows kept)

```sql
SELECT name, department, salary,
  AVG(salary)  OVER (PARTITION BY department) AS dept_avg,
  SUM(salary)  OVER (PARTITION BY department) AS dept_total,
  COUNT(*)     OVER (PARTITION BY department) AS dept_headcount
FROM employees;
```

### Variant 2 — ORDER BY only (running/cumulative across all rows)

```sql
SELECT name, order_date, amount,
  SUM(amount) OVER (ORDER BY order_date) AS running_total
FROM orders;
```

### Variant 3 — Anatomy query: all three OVER parts together

```sql
SELECT
    date, model_name, accuracy,
    -- No PARTITION: global rank across all rows
    RANK() OVER (ORDER BY accuracy DESC)                          AS global_rank,
    -- PARTITION only: count of days per model (same for all rows of that model)
    COUNT(*) OVER (PARTITION BY model_name)                       AS model_day_count,
    -- PARTITION + ORDER: cumulative max accuracy per model
    MAX(accuracy) OVER (PARTITION BY model_name ORDER BY date)    AS cumulative_best
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
| ORDER BY in OVER changes default frame | Adding ORDER BY changes the frame from ALL ROWS to `RANGE UNBOUNDED PRECEDING` | Be explicit with your frame clause |
| Window functions in GROUP BY queries | Window runs after grouping — sees the grouped result set, not original rows | Intentional; just know the order of operations |

---

## 2. Ranking Functions — ROW_NUMBER, RANK, DENSE_RANK, NTILE

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
    model_name, f1_score,
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

> RANK skips to 4 after the tie at 2. DENSE_RANK does not skip. ROW_NUMBER breaks ties arbitrarily.

### Variant 2 — RANK within a partition (per dataset type)

```sql
SELECT model_name, dataset, f1_score,
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

### Variant 3 — NTILE for score quartiles + human-readable labels

```sql
WITH bucketed AS (
  SELECT model_name, f1_score,
    NTILE(4) OVER (ORDER BY f1_score) AS quartile   -- 1=bottom, 4=top
  FROM model_leaderboard
)
SELECT model_name, f1_score,
  CASE quartile
    WHEN 1 THEN 'Bottom 25%'
    WHEN 2 THEN 'Lower Mid'
    WHEN 3 THEN 'Upper Mid'
    WHEN 4 THEN 'Top 25%'
  END AS score_band
FROM bucketed;
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
| RANK vs DENSE_RANK confusion | "Give me rank 2" — does the interviewer mean RANK or DENSE_RANK? | Clarify; default to DENSE_RANK unless gaps are meaningful |
| NTILE with uneven division | 10 rows, NTILE(3) gives buckets of 4,3,3 | Larger buckets always come first |
| Using RANK to deduplicate | RANK allows ties; dedup needs strictly unique numbers | Always use ROW_NUMBER for dedup |

---

## 3. Top-N Per Group Pattern

### Concept

"Give me the top 3 models by accuracy for each dataset" — asked in nearly every ML data interview. The pattern is always:
1. Assign `ROW_NUMBER()` (or `DENSE_RANK()` if ties should all be included) within each group
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

### Variant 1 — Top 2 per group (ROW_NUMBER)

```sql
WITH ranked AS (
    SELECT exp_id, model_name, dataset, accuracy,
        ROW_NUMBER() OVER (PARTITION BY dataset ORDER BY accuracy DESC) AS rn
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

### Variant 2 — Top-1 per group, ties included (DENSE_RANK)

```sql
WITH ranked AS (
    SELECT name, department, salary,
        DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dr
    FROM employees
)
SELECT name, department, salary FROM ranked WHERE dr = 1;
```

> Use `DENSE_RANK` when the top spot might legitimately be tied and both winners should show; use `ROW_NUMBER` when exactly one row per group is required.

### Variant 3 — Bottom N (worst performers, for triage)

```sql
WITH ranked AS (
    SELECT model_name, dataset, accuracy,
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
| Using RANK instead of ROW_NUMBER | Two tied models both get rn=1; `WHERE rn=1` returns both — sometimes intentional, sometimes a bug | Use RANK/DENSE_RANK if you want all ties; ROW_NUMBER for strict top-1 |
| Filtering in WHERE without a CTE | `WHERE ROW_NUMBER() OVER(...) <= 2` is illegal SQL | Always wrap in CTE or subquery first |
| Forgetting PARTITION BY | Returns global top-N instead of per-group top-N | Double-check the PARTITION BY column |
| Non-deterministic top-1 on ties | Two models tied — which one "wins" can vary by run | Add a secondary sort: `ORDER BY accuracy DESC, model_name` |

---

## 4. LAG and LEAD — Accessing Adjacent Rows

### Concept

`LAG` and `LEAD` let you look **backwards** or **forwards** within an ordered window without a self-join.

```sql
LAG (col, offset, default)  OVER (PARTITION BY ... ORDER BY ...)
LEAD(col, offset, default)  OVER (PARTITION BY ... ORDER BY ...)
```

- `offset` — how many rows to look back/forward (default = 1)
- `default` — value used when there is no previous/next row (default = NULL)

Classic uses: day-over-day change, detecting state transitions, computing inter-event time, days-until-next-order.

---

### Sample Table: `daily_model_metrics` (reused from Section 1)

---

### Variant 1 — Day-over-day accuracy change

```sql
SELECT date, model_name, accuracy,
    LAG(accuracy) OVER (PARTITION BY model_name ORDER BY date) AS prev_day_accuracy,
    ROUND(accuracy - LAG(accuracy) OVER (PARTITION BY model_name ORDER BY date), 4) AS accuracy_delta
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

### Variant 2 — LEAD to see the next row's value (with a default)

```sql
SELECT date, model_name, requests,
    LEAD(requests, 1, 0) OVER (PARTITION BY model_name ORDER BY date) AS next_day_requests
FROM daily_model_metrics
ORDER BY model_name, date;
```

**Output (LightGBM):**

| date       | requests | next_day_requests |
|------------|----------|-------------------|
| 2024-01-01 | 800      | 850               |
| 2024-01-02 | 850      | 900               |
| 2024-01-03 | 900      | 780               |
| 2024-01-04 | 780      | 0 (default)        |

### Variant 3 — Related applications (days until next order, price movement, regression detection)

```sql
-- Days between orders per customer
SELECT customer_id, order_date, amount,
  LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order_date,
  DATEDIFF(
    LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date),
    order_date
  ) AS days_until_next_order
FROM orders;

-- Detect price movement direction
SELECT product_id, price_date, price,
  LAG(price) OVER (PARTITION BY product_id ORDER BY price_date) AS prev_price,
  CASE
    WHEN price > LAG(price) OVER (PARTITION BY product_id ORDER BY price_date) THEN 'Increased'
    WHEN price < LAG(price) OVER (PARTITION BY product_id ORDER BY price_date) THEN 'Decreased'
    ELSE 'No Change'
  END AS price_movement
FROM product_prices;

-- Flag accuracy regressions (drop > 0.01 vs prior day) — filter requires a CTE
WITH deltas AS (
    SELECT date, model_name, accuracy,
        LAG(accuracy) OVER (PARTITION BY model_name ORDER BY date) AS prev_acc
    FROM daily_model_metrics
)
SELECT date, model_name, ROUND(prev_acc, 4) AS prev_accuracy,
    ROUND(accuracy, 4) AS curr_accuracy, ROUND(accuracy - prev_acc, 4) AS delta
FROM deltas
WHERE accuracy - prev_acc < -0.01
ORDER BY delta;
```

**Output (regression detection):**

| date       | model_name | prev_accuracy | curr_accuracy | delta   |
|------------|------------|----------------|---------------|---------|
| 2024-01-04 | LightGBM   | 0.9450         | 0.9300        | -0.0150 |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Missing PARTITION BY | LAG looks across ALL entities, mixing rows from different groups | Always PARTITION BY the entity (model, customer, user, product) |
| LAG with date gaps | LAG returns the previous *row*, not the previous *calendar day* | Join to a date spine before using LAG if calendar continuity matters |
| Using LAG result directly in WHERE | Can't filter in WHERE on a window function | Wrap in a CTE, then filter |
| Default value type mismatch | `LAG(accuracy, 1, 0)` — 0 must match the column's type | Use `0.0` for floats, not integer `0` |

---

## 5. Running Totals and Cumulative Aggregates

### Concept

Running totals use this frame when ORDER BY is present:
`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`

This means: accumulate everything from the first row in the partition up to and including the current row. Always write this explicitly — see Section 6 for why relying on the default is risky.

---

### Sample Table: `daily_requests` (XGBoost only)

| date       | model_name | requests |
|------------|------------|----------|
| 2024-01-01 | XGBoost    | 1200     |
| 2024-01-02 | XGBoost    | 1300     |
| 2024-01-03 | XGBoost    | 1100     |
| 2024-01-04 | XGBoost    | 1250     |

---

### Variant 1 — Basic running total

```sql
SELECT date, requests,
    SUM(requests) OVER (
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_requests
FROM daily_requests
WHERE model_name = 'XGBoost';
```

**Output:**

| date       | requests | cumulative_requests |
|------------|----------|----------------------|
| 2024-01-01 | 1200     | 1200                 |
| 2024-01-02 | 1300     | 2500                 |
| 2024-01-03 | 1100     | 3600                 |
| 2024-01-04 | 1250     | 4850                 |

### Variant 2 — Running total as % of grand total

```sql
WITH totals AS (
    SELECT date, requests,
        SUM(requests) OVER (ORDER BY date)  AS running_total,
        SUM(requests) OVER ()               AS grand_total   -- no ORDER BY = all rows
    FROM daily_requests
    WHERE model_name = 'XGBoost'
)
SELECT date, requests, running_total,
    ROUND(running_total * 100.0 / grand_total, 1) AS cumulative_pct
FROM totals;
```

**Output:**

| date       | requests | running_total | cumulative_pct |
|------------|----------|----------------|-----------------|
| 2024-01-01 | 1200     | 1200           | 24.7            |
| 2024-01-02 | 1300     | 2500           | 51.5            |
| 2024-01-03 | 1100     | 3600           | 74.2            |
| 2024-01-04 | 1250     | 4850           | 100.0           |

### Variant 3 — Running total by group WITH RESET (e.g., resets every month)

The reset-per-period pattern is one of the most common production variants: `PARTITION BY` the entity **and** the period together.

```sql
-- Running revenue per sales rep, resets each calendar month
SELECT rep_id, sale_date, amount,
  SUM(amount) OVER (
    PARTITION BY rep_id, DATE_FORMAT(sale_date, '%Y-%m')
    ORDER BY sale_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS monthly_running_total
FROM sales;

-- Running order count per customer, resets each year
SELECT customer_id, order_date,
  COUNT(*) OVER (
    PARTITION BY customer_id, YEAR(order_date)
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS order_num_this_year
FROM orders;
```

### Variant 4 — Running max / high-water mark

```sql
SELECT date, model_name, accuracy,
    MAX(accuracy) OVER (
        PARTITION BY model_name ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS best_so_far,
    accuracy = MAX(accuracy) OVER (
        PARTITION BY model_name ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS is_new_record
FROM daily_model_metrics
ORDER BY model_name, date;
```

**Output:**

| date       | model_name | accuracy | best_so_far | is_new_record |
|------------|------------|----------|--------------|----------------|
| 2024-01-01 | LightGBM   | 0.940    | 0.940        | true           |
| 2024-01-02 | LightGBM   | 0.935    | 0.940        | false          |
| 2024-01-03 | LightGBM   | 0.945    | 0.945        | true           |
| 2024-01-04 | LightGBM   | 0.930    | 0.945        | false          |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| RANGE vs ROWS on tied values | Default RANGE includes all rows sharing the current ORDER BY value — inflates sum on ties | Use `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` explicitly |
| `SUM() OVER ()` confusion | No ORDER BY = grand total repeated on every row, not a running total | Intentional behavior; add ORDER BY to get a running total instead |
| Running total resets unexpectedly (or doesn't reset when it should) | Forgot PARTITION BY, or partitioned by entity only instead of entity + period | Add `PARTITION BY entity, period` for reset-per-period logic |

---

## 6. Window Frame Clauses — ROWS vs RANGE, BETWEEN

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
| `N PRECEDING` | N rows before the current row |
| `CURRENT ROW` | Current row only |
| `N FOLLOWING` | N rows after the current row |
| `UNBOUNDED FOLLOWING` | Last row of the partition |

**Defaults:**
- When ORDER BY is present: `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
- When no ORDER BY: `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`

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
SELECT id, score,
    SUM(score) OVER (ORDER BY score ROWS  BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rows_sum,
    SUM(score) OVER (ORDER BY score RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS range_sum
FROM scores_with_ties;
```

**Output:**

| id | score | rows_sum | range_sum |
|----|-------|----------|-----------|
| 1  | 10    | 10       | 10        |
| 2  | 20    | 30       | 50        |
| 3  | 20    | 50       | 50        |
| 4  | 30    | 80       | 80        |

> `RANGE` treats every row tied on the ORDER BY value (rows 2 and 3, both score=20) as part of the same logical group — they both get the total through the *entire tied group* (10+20+20=50), not their individual row position. `ROWS` is strictly positional, so row 2 only sees 10+20=30.

### Variant 2 — N PRECEDING: 3-row trailing window

```sql
SELECT date, requests,
    AVG(requests) OVER (
        ORDER BY date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS trailing_3day_avg
FROM daily_requests
WHERE model_name = 'XGBoost';
```

**Output:**

| date       | requests | trailing_3day_avg |
|------------|----------|--------------------|
| 2024-01-01 | 1200     | 1200.0             |
| 2024-01-02 | 1300     | 1250.0             |
| 2024-01-03 | 1100     | 1200.0             |
| 2024-01-04 | 1250     | 1216.7             |

### Variant 3 — Suffix sum (remaining total from current row to end)

```sql
SELECT date, requests,
    SUM(requests) OVER (
        ORDER BY date
        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
    ) AS requests_remaining
FROM daily_requests
WHERE model_name = 'XGBoost';
```

**Output:**

| date       | requests | requests_remaining |
|------------|----------|----------------------|
| 2024-01-01 | 1200     | 4850                 |
| 2024-01-02 | 1300     | 3650                 |
| 2024-01-03 | 1100     | 2350                 |
| 2024-01-04 | 1250     | 1250                 |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Invisible RANGE default inflates running totals on ties | The silent default is RANGE, not ROWS | Always write `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` explicitly |
| N PRECEDING is rows, not time | `2 PRECEDING` = 2 rows back, not 2 calendar days, if data has gaps | Fill date gaps with a spine first |
| Frame clause requires ORDER BY | A frame without ORDER BY is a syntax error | Always pair a frame clause with ORDER BY |
| LAST_VALUE with default frame | Returns the current row, not the partition's true last row | Use `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` (see Section 11) |

---

## 7. Moving Averages and Rolling Windows

### Concept

A moving average smooths time-series noise. Used in ML for monitoring accuracy drift, request volume trends, and anomaly baselines.

```sql
AVG(metric) OVER (
    PARTITION BY entity
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
SELECT date, accuracy,
    ROUND(AVG(accuracy) OVER (
        PARTITION BY model_name ORDER BY date
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

### Variant 2 — Exclude warm-up rows (only full windows)

```sql
WITH ma AS (
    SELECT date, accuracy,
        AVG(accuracy) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma_3day,
        COUNT(*) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS window_size
    FROM daily_model_metrics
    WHERE model_name = 'XGBoost'
)
SELECT date, accuracy, ROUND(ma_3day, 4) AS ma_3day
FROM ma
WHERE window_size = 3;
```

**Output:** (first two warm-up rows dropped)

| date       | accuracy | ma_3day |
|------------|----------|---------|
| 2024-01-03 | 0.920    | 0.9117  |
| 2024-01-04 | 0.915    | 0.9133  |
| 2024-01-05 | 0.900    | 0.9117  |
| 2024-01-06 | 0.925    | 0.9133  |
| 2024-01-07 | 0.918    | 0.9143  |

### Variant 3 — Anomaly detection: today vs prior 7-day baseline (excludes today)

```sql
WITH ma AS (
    SELECT date, model_name, accuracy,
        AVG(accuracy) OVER (
            PARTITION BY model_name ORDER BY date
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING  -- excludes current row from baseline
        ) AS prior_7day_avg
    FROM daily_model_metrics
)
SELECT date, model_name, accuracy,
    ROUND(prior_7day_avg, 4) AS baseline,
    ROUND(accuracy - prior_7day_avg, 4) AS deviation,
    CASE
        WHEN accuracy < prior_7day_avg - 0.01 THEN 'ALERT: drop'
        WHEN accuracy > prior_7day_avg + 0.01 THEN 'ALERT: spike'
        ELSE 'normal'
    END AS status
FROM ma
WHERE prior_7day_avg IS NOT NULL;
```

> This is the **ML model monitoring** pattern — detect drift by comparing today to a rolling baseline that explicitly excludes today (`1 PRECEDING` as the upper bound).

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Partial windows at the start of the series | AVG of 1–2 rows looks smoothed but isn't a true N-period average | Filter `WHERE window_size = N` or document the warm-up period |
| Gaps in the date series | `3 PRECEDING` = 3 rows, which may span more than 3 calendar days if data is missing | Fill date gaps with a spine before computing moving averages |
| Including the current row in a "prior" baseline | Baseline gets inflated by the value you're comparing against it | Use `ROWS BETWEEN N PRECEDING AND 1 PRECEDING` |

---

## 8. Weighted Averages

### Concept

A simple `AVG()` treats every row equally — but this creates **Simpson's Paradox** style errors when group sizes differ (e.g., averaging daily averages instead of weighting by daily volume). A true weighted average divides total value by total weight (e.g., total revenue / total orders), not the average of averages.

Weighted averages also show up as **recency-weighted rolling metrics**, where more recent periods matter more.

---

### Sample Table: `orders`

| category | amount |
|----------|--------|
| Electronics | 500 |
| Electronics | 100 |
| Electronics | 100 |
| Apparel     | 50  |
| Apparel     | 60  |

---

### Variant 1 — Weighted average vs simple average (Simpson's Paradox guard)

```sql
SELECT category,
  SUM(amount) / COUNT(*) AS weighted_avg_order_value,  -- weight by number of orders
  AVG(amount)            AS simple_avg
FROM orders
GROUP BY category;
```

> `SUM(amount) / COUNT(*)` weights every dollar equally across all orders. If you instead averaged pre-aggregated daily averages, days with few orders would carry the same influence as days with thousands — that's the Simpson's Paradox trap.

### Variant 2 — Rolling 3-month weighted average (recent months weighted higher)

```sql
SELECT month, revenue,
  ROUND(
    (revenue * 3
     + LAG(revenue, 1, 0) OVER (ORDER BY month) * 2
     + LAG(revenue, 2, 0) OVER (ORDER BY month) * 1
    ) / 6, 2
  ) AS weighted_3mo_avg
FROM monthly_revenue;
```

> Weights sum to 6 (3+2+1); current month contributes 3x, two months ago contributes 1x.

### Variant 3 — Weighted average combined with window ranking

```sql
SELECT category,
  SUM(amount) AS total_revenue,
  COUNT(*)    AS total_orders,
  SUM(amount) / COUNT(*) AS weighted_avg_order_value,
  RANK() OVER (ORDER BY SUM(amount) / COUNT(*) DESC) AS value_rank
FROM orders
GROUP BY category;
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Averaging pre-aggregated averages | `AVG(daily_avg)` treats a low-volume day the same as a high-volume day | Use `SUM(value) / SUM(weight)` instead |
| Weights that don't sum to a clean denominator | Arbitrary weights make the result hard to sanity-check | Pick weights that sum to a round number (e.g., 3+2+1=6) |
| Confusing "weighted" with "exponential" | A linear weighted average is not the same as an exponential moving average (EMA) | Be explicit in interviews about which one is being asked for |

---

## 9. Percentiles — PERCENTILE_CONT, PERCENTILE_DISC, PERCENT_RANK, CUME_DIST

### Concept

| Function | Type | What it returns |
|----------|------|------------------|
| `PERCENTILE_CONT(p)` | Ordered-set aggregate | Interpolated value at percentile p — may not exist in the data |
| `PERCENTILE_DISC(p)` | Ordered-set aggregate | Nearest actual data value at or above percentile p |
| `PERCENT_RANK()` | Window function | Relative rank: `(rank - 1) / (n - 1)` → range [0, 1] |
| `CUME_DIST()` | Window function | Fraction of rows ≤ current row → range (0, 1] |

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
SELECT model_name,
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

> `p50_cont` = 23.5 (interpolated between 22 and 25). `p50_disc` = 22 (an actual row's value at the 50th percentile position).

### Variant 2 — PERCENT_RANK and CUME_DIST as window functions

```sql
SELECT request_id, latency_ms,
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

> `PERCENT_RANK` of the first row is always 0. `CUME_DIST` is always > 0.

### Variant 3 — Flag P95 SLA breaches

```sql
WITH p95 AS (
    SELECT model_name,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency
    FROM model_inference_latency
    GROUP BY model_name
)
SELECT r.request_id, r.model_name, r.latency_ms,
    ROUND(p.p95_latency, 1) AS p95_threshold,
    r.latency_ms > p.p95_latency AS sla_breach
FROM model_inference_latency r
JOIN p95 p ON r.model_name = p.model_name
ORDER BY r.latency_ms DESC;
```

**Output:**

| request_id | model_name | latency_ms | p95_threshold | sla_breach |
|------------|------------|------------|----------------|------------|
| 8          | XGBoost    | 120        | 93.8           | true       |
| 7          | XGBoost    | 45         | 93.8           | false      |

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| CONT vs DISC confusion | CONT may return a value not present in the data; DISC always returns an actual value | For latency SLAs, CONT is the standard convention |
| `WITHIN GROUP` vs `OVER()` | `PERCENTILE_CONT` uses `WITHIN GROUP`, not `OVER` — a different syntax path entirely | Memorize both syntaxes separately |
| PERCENT_RANK starts at 0, CUME_DIST never reaches 0 | `PERCENT_RANK = (rank-1)/(n-1)`; `CUME_DIST = rank/n` | Know the formulas cold |
| Percentile without PARTITION BY | Returns a single global percentile across all groups | Add `PARTITION BY` or compute per-group in a CTE |

---

## 10. Median

### Concept

SQL has no universal `MEDIAN()` function. Three approaches:

| Approach | DB support | Notes |
|----------|-----------|-------|
| `PERCENTILE_CONT(0.5)` | PostgreSQL, Redshift, Snowflake, BigQuery (via extension) | Standard, interpolated |
| `MEDIAN()` | Oracle, some others | Direct built-in function |
| Manual `ROW_NUMBER()` approach | All databases including MySQL | Verbose but fully portable |

---

### Variant 1 — Standard median + mean-vs-median gap (skew detection)

```sql
SELECT model_name,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS median_latency,
    ROUND(AVG(latency_ms), 2) AS mean_latency,
    ROUND(AVG(latency_ms) - PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms), 2) AS mean_minus_median
FROM model_inference_latency
GROUP BY model_name;
```

**Output:**

| model_name | median_latency | mean_latency | mean_minus_median |
|------------|-----------------|--------------|---------------------|
| XGBoost    | 23.5            | 35.88        | 12.38               |

> A large positive gap means mean > median → **right-skewed distribution** (long-tail latency from outliers). This is a critical insight for ML serving.

### Variant 2 — Full percentile profile per group

```sql
SELECT model_name,
    COUNT(*)                                                        AS n,
    ROUND(MIN(latency_ms))                                          AS min_ms,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY latency_ms)) AS p25,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms)) AS p50,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY latency_ms)) AS p75,
    ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms)) AS p90,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms)) AS p99,
    ROUND(MAX(latency_ms))                                          AS max_ms
FROM model_inference_latency
GROUP BY model_name;
```

### Variant 3 — Manual median via ROW_NUMBER (works on every SQL engine, including MySQL)

```sql
WITH ranked AS (
    SELECT salary,
        ROW_NUMBER() OVER (ORDER BY salary) AS rn,
        COUNT(*)     OVER ()                AS total
    FROM employees
)
SELECT AVG(salary) AS median
FROM ranked
WHERE rn IN (FLOOR((total + 1) / 2.0), CEIL((total + 1) / 2.0));
```

**Output (8-row example):**

| median |
|--------|
| 23.5   |

> `FLOOR`/`CEIL` of `(total+1)/2` pick the *same* row for odd N (one middle value), and *two different* middle rows for even N. `AVG()` correctly handles both cases — for 8 rows this resolves to positions 4 and 5.

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using AVG as a proxy for median | Outliers skew the mean significantly | Always compute both; report median for latency and income-style distributions |
| Manual median for even N | Floor and Ceil must pick two different rows, not the same one twice | Verify with a small example: 8 rows → positions 4 and 5 |
| BigQuery has no native `PERCENTILE_CONT` | Standard BigQuery SQL lacks this ordered-set aggregate | Use `APPROX_QUANTILES(col, 100)[OFFSET(50)]` |

---

## 11. FIRST_VALUE, LAST_VALUE, NTH_VALUE

### Concept

These pull a value from a **specific row** within the window frame:
- `FIRST_VALUE(col)` — value from the first row of the frame
- `LAST_VALUE(col)` — value from the last row of the frame
- `NTH_VALUE(col, n)` — value from the nth row of the frame

**Critical:** `LAST_VALUE` with the default frame (`RANGE ... CURRENT ROW`) returns the **current row's own value**, not the partition's true last row. Always extend the frame explicitly — this is one of the most common SQL interview bugs.

---

### Variant 1 — Gap from personal best (each day vs. the model's best-ever day)

```sql
SELECT date, model_name, accuracy,
    FIRST_VALUE(accuracy) OVER (
        PARTITION BY model_name ORDER BY accuracy DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS best_accuracy,
    ROUND(
        accuracy / FIRST_VALUE(accuracy) OVER (
            PARTITION BY model_name ORDER BY accuracy DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) - 1, 4
    ) AS gap_from_best
FROM daily_model_metrics
ORDER BY model_name, date;
```

**Output:**

| date       | model_name | accuracy | best_accuracy | gap_from_best |
|------------|------------|----------|-----------------|-----------------|
| 2024-01-01 | LightGBM   | 0.940    | 0.945           | -0.0053         |
| 2024-01-02 | LightGBM   | 0.935    | 0.945           | -0.0106         |
| 2024-01-03 | LightGBM   | 0.945    | 0.945           | 0.0000          |
| 2024-01-04 | LightGBM   | 0.930    | 0.945           | -0.0159         |

### Variant 2 — LAST_VALUE (must extend the frame, or the "top earner per department" pattern)

```sql
-- Latest accuracy value carried onto every row (correct — frame extended)
SELECT date, model_name, accuracy,
    LAST_VALUE(accuracy) OVER (
        PARTITION BY model_name ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING  -- critical!
    ) AS latest_accuracy
FROM daily_model_metrics
ORDER BY model_name, date;

-- Same idea: top earner in department shown on every employee's row
SELECT name, department, salary,
  FIRST_VALUE(name) OVER (
    PARTITION BY department ORDER BY salary DESC
  ) AS top_earner_in_dept
FROM employees;

-- Department min salary via LAST_VALUE — frame is mandatory here
SELECT name, department, salary,
  LAST_VALUE(salary) OVER (
    PARTITION BY department
    ORDER BY salary DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS dept_min_salary
FROM employees;
```

> ⚠️ Without the explicit frame, `LAST_VALUE` only looks up to the current row — an extremely common bug that silently returns the wrong (current-row) value instead of the group's true last value.

### Variant 3 — NTH_VALUE: second-best accuracy, using a named WINDOW

```sql
SELECT DISTINCT model_name,
    FIRST_VALUE(accuracy) OVER w  AS best,
    NTH_VALUE(accuracy, 2) OVER w AS second_best
FROM daily_model_metrics
WINDOW w AS (
    PARTITION BY model_name
    ORDER BY accuracy DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
);
```

> The `WINDOW` keyword lets you name a reusable window definition — cleaner than repeating the `OVER` clause across multiple functions.

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| LAST_VALUE with default frame | Returns the current row's value, not the partition's true last value | Always add `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` |
| NTH_VALUE returns NULL if fewer than N rows exist | A partition with only 1 row makes `NTH_VALUE(col, 2)` return NULL | Use `COALESCE` or check the partition count first |
| FIRST_VALUE direction confusion | `ORDER BY accuracy DESC` gives the best value; `ASC` gives the worst | Always confirm sort direction matches business intent |

---

## 12. Gaps and Islands Problem

### Concept

**Islands** = groups of consecutive values. **Gaps** = breaks between islands. The canonical technique subtracts `ROW_NUMBER()` from the sequence value — consecutive integers/dates produce the same constant (the island's group ID).

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

### Variant 1 — Identify islands (consecutive uptime periods) and streak length

```sql
WITH numbered AS (
    SELECT date, model_name,
        ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY date) AS rn
    FROM model_uptime_days
),
grouped AS (
    SELECT date, model_name,
        (date - rn * INTERVAL '1 day')::DATE AS grp   -- constant within each island
    FROM numbered
)
SELECT model_name, MIN(date) AS island_start, MAX(date) AS island_end, COUNT(*) AS days_in_island
FROM grouped
GROUP BY model_name, grp
ORDER BY island_start;
```

**Output:**

| model_name | island_start | island_end | days_in_island |
|------------|----------------|-------------|------------------|
| XGBoost    | 2024-01-01     | 2024-01-03  | 3                |
| XGBoost    | 2024-01-05     | 2024-01-06  | 2                |
| XGBoost    | 2024-01-09     | 2024-01-10  | 2                |

> MySQL equivalent: `DATE_SUB(day, INTERVAL rn DAY)` instead of `date - rn * INTERVAL '1 day'`.

### Variant 2 — Find the gaps (downtime windows) between islands

```sql
WITH numbered AS (
    SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS rn
    FROM model_uptime_days WHERE model_name = 'XGBoost'
),
islands AS (
    SELECT MIN(date) AS island_start, MAX(date) AS island_end,
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

### Variant 3 — Longest streak per user (login streak variant)

```sql
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
islands AS (
  SELECT user_id,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp,
    COUNT(*) AS streak_len
  FROM numbered
  GROUP BY user_id, DATE_SUB(login_date, INTERVAL rn DAY)
)
SELECT user_id, MAX(streak_len) AS longest_streak
FROM islands
GROUP BY user_id
ORDER BY longest_streak DESC;

-- Bonus: integer gap detection (missing sequential IDs), not date-based
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
-- BigQuery: UNNEST(GENERATE_ARRAY(min, max)) instead of GENERATE_SERIES
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Date arithmetic type issues | `date - rn` in Postgres yields an interval, not a date | Cast explicitly: `(date - rn * INTERVAL '1 day')::DATE` |
| Missing PARTITION BY | Islands bleed across different entities (users, models) | Always `PARTITION BY` the entity before computing ROW_NUMBER |
| Non-1 step sequences | The trick only works cleanly for step=1; step-N sequences need `value/step - rn` | Normalize the step size in the arithmetic |
| GENERATE_SERIES portability | PostgreSQL-only function | BigQuery: `GENERATE_ARRAY`; other engines: build a numbers table |

---

## 13. Session Detection with Window Functions

### Concept

Group user events into sessions where any gap greater than a threshold (e.g., 30 minutes) starts a new session.

**Algorithm:**
1. `LAG` to get the previous event timestamp per user
2. Flag rows where the gap exceeds the threshold (or where there's no previous row) as a session start (= 1)
3. Cumulative `SUM` of those flags = the session number

---

### Sample Table: `user_events`

| event_id | user_id | event_ts            | event_type |
|----------|---------|----------------------|------------|
| 1        | u1      | 2024-01-15 10:00:00  | click      |
| 2        | u1      | 2024-01-15 10:05:00  | click      |
| 3        | u1      | 2024-01-15 10:45:00  | purchase   |
| 4        | u1      | 2024-01-15 10:50:00  | click      |
| 5        | u2      | 2024-01-15 11:00:00  | click      |
| 6        | u2      | 2024-01-15 11:10:00  | click      |

---

### Variant 1 — Assign session IDs (30-minute timeout)

```sql
WITH lagged AS (
    SELECT event_id, user_id, event_ts, event_type,
        LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS prev_ts
    FROM user_events
),
flagged AS (
    SELECT *,
        CASE
            WHEN prev_ts IS NULL THEN 1
            WHEN EXTRACT(EPOCH FROM (event_ts - prev_ts)) > 1800 THEN 1
            ELSE 0
        END AS is_new_session
    FROM lagged
)
SELECT event_id, user_id, event_ts, event_type,
    SUM(is_new_session) OVER (
        PARTITION BY user_id ORDER BY event_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS session_id
FROM flagged
ORDER BY user_id, event_ts;
```

**Output:**

| event_id | user_id | event_ts             | event_type | session_id |
|----------|---------|------------------------|------------|------------|
| 1        | u1      | 2024-01-15 10:00:00    | click      | 1          |
| 2        | u1      | 2024-01-15 10:05:00    | click      | 1          |
| 3        | u1      | 2024-01-15 10:45:00    | purchase   | 2          |
| 4        | u1      | 2024-01-15 10:50:00    | click      | 2          |
| 5        | u2      | 2024-01-15 11:00:00    | click      | 1          |
| 6        | u2      | 2024-01-15 11:10:00    | click      | 1          |

### Variant 2 — MySQL/generic equivalent using minute differences

```sql
WITH gaps AS (
  SELECT user_id, event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_time
  FROM events
),
sessions AS (
  SELECT user_id, event_time,
    SUM(CASE
      WHEN TIMESTAMPDIFF(MINUTE, prev_time, event_time) > 30
        OR prev_time IS NULL THEN 1 ELSE 0
    END) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
  FROM gaps
)
SELECT user_id, session_id, COUNT(*) AS events_in_session
FROM sessions
GROUP BY user_id, session_id;
```

### Variant 3 — Session summary rollup

```sql
WITH sessions AS (
    -- paste the session assignment CTE chain from Variant 1 here
    SELECT event_id, user_id, event_ts, event_type, session_id FROM ( /* ... */ ) t
)
SELECT user_id, session_id,
    MIN(event_ts) AS session_start,
    MAX(event_ts) AS session_end,
    EXTRACT(EPOCH FROM MAX(event_ts) - MIN(event_ts)) AS duration_seconds,
    COUNT(*) AS event_count,
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchases
FROM sessions
GROUP BY user_id, session_id
ORDER BY user_id, session_id;
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Session IDs are per-user, not global | `user1` session 1 ≠ `user2` session 1 | Concatenate `user_id || '-' || session_id` for a globally unique key |
| Timezone-naive timestamps | Gap calculations can be wrong across DST boundaries | Store all timestamps in UTC |
| Very short sessions pollute metrics | Bot traffic creates many 1-event sessions | Filter with `HAVING COUNT(*) > 1` or `duration_seconds > 0` |

---

## 14. Year-over-Year and Period Comparisons

### Concept

Two approaches to period comparisons:
1. **LAG with offset** — `LAG(metric, 12)` over monthly data = same month last year
2. **Self-join on shifted date** — `curr.month = prev.month + INTERVAL '1 year'`

The self-join approach is safer when data has gaps, because a missing month silently breaks the LAG offset.

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

### Variant 1 — YoY using LAG offset 12 (monthly) or LAG over years (annual)

```sql
-- Monthly data, 12-month offset
SELECT month, model_name, requests,
    LAG(requests, 12) OVER (PARTITION BY model_name ORDER BY month) AS prev_year_requests,
    ROUND(
        (requests - LAG(requests, 12) OVER (PARTITION BY model_name ORDER BY month)) * 100.0
        / NULLIF(LAG(requests, 12) OVER (PARTITION BY model_name ORDER BY month), 0), 1
    ) AS yoy_pct
FROM monthly_model_requests
ORDER BY model_name, month;

-- Annual data, simple LAG(1) over years
WITH yearly AS (
  SELECT YEAR(order_date) AS yr, SUM(amount) AS revenue
  FROM orders GROUP BY YEAR(order_date)
)
SELECT yr, revenue,
  LAG(revenue) OVER (ORDER BY yr) AS prev_year,
  ROUND((revenue - LAG(revenue) OVER (ORDER BY yr)) * 100.0 /
    NULLIF(LAG(revenue) OVER (ORDER BY yr), 0), 2) AS yoy_pct
FROM yearly;
```

**Output (monthly):**

| month      | model_name | requests | prev_year_requests | yoy_pct |
|------------|------------|----------|-----------------------|---------|
| 2024-01-01 | XGBoost    | 13000    | 10000                 | 30.0    |
| 2024-02-01 | XGBoost    | 14500    | 11000                 | 31.8    |
| 2024-03-01 | XGBoost    | 13800    | 10500                 | 31.4    |

### Variant 2 — YoY via self-join (handles gaps safely)

```sql
SELECT curr.month, curr.model_name,
    curr.requests AS curr_requests,
    prev.requests AS prev_year_requests,
    ROUND((curr.requests - prev.requests) * 100.0 / NULLIF(prev.requests, 0), 1) AS yoy_pct
FROM monthly_model_requests curr
LEFT JOIN monthly_model_requests prev
    ON  curr.model_name = prev.model_name
    AND curr.month      = prev.month + INTERVAL '1 year'
ORDER BY curr.model_name, curr.month;
```

### Variant 3 — Same-week-last-year comparison (common at Meta/Google) and MoM growth

```sql
-- Same calendar week, year over year
WITH weekly AS (
  SELECT YEAR(sale_date) AS yr, WEEK(sale_date) AS wk, SUM(revenue) AS weekly_rev
  FROM daily_sales GROUP BY YEAR(sale_date), WEEK(sale_date)
)
SELECT yr, wk, weekly_rev,
  LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr) AS same_wk_last_yr,
  ROUND(
    (weekly_rev - LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr)) * 100.0 /
    NULLIF(LAG(weekly_rev) OVER (PARTITION BY wk ORDER BY yr), 0), 2
  ) AS yoy_pct
FROM weekly ORDER BY yr, wk;

-- Month-over-month growth
SELECT month, model_name, requests,
    LAG(requests, 1) OVER (PARTITION BY model_name ORDER BY month) AS prev_month,
    ROUND(
        (requests - LAG(requests, 1) OVER (PARTITION BY model_name ORDER BY month)) * 100.0
        / NULLIF(LAG(requests, 1) OVER (PARTITION BY model_name ORDER BY month), 0), 1
    ) AS mom_pct
FROM monthly_model_requests
ORDER BY model_name, month;
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| LAG offset assumes no gaps | A missing month silently shifts every subsequent offset | Self-join is safer for sparse or irregular data |
| Dividing by zero when the prior period is 0 | A newly launched model has an undefined YoY% | Always wrap the denominator in `NULLIF(prev, 0)` |
| Feb 29 date arithmetic | `2024-02-29 + INTERVAL '1 year'` errors or behaves oddly on some engines | Use `DATE_TRUNC('month', ...)` to normalize before comparing |

---

## 15. Advanced FAANG Patterns — Attribution, Threshold Crossing, Rolling Active Users

### Concept

Three patterns that show up repeatedly in senior-level interviews and don't fit neatly into the categories above: **first-touch attribution**, **threshold crossing**, and **rolling N-day active user counts**.

---

### Variant 1 — First-touch channel attribution

```sql
WITH ranked AS (
  SELECT user_id, channel, purchase_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY purchase_date ASC) AS rn
  FROM purchases
)
SELECT channel,
  COUNT(*) AS first_purchases,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_attribution
FROM ranked
WHERE rn = 1
GROUP BY channel
ORDER BY first_purchases DESC;
```

> `SUM(COUNT(*)) OVER ()` with no PARTITION BY sums the already-aggregated group counts into a grand total — a nested aggregate + window combo worth memorizing.

### Variant 2 — Threshold crossing (find the exact order that pushed cumulative spend over $1000)

```sql
WITH running AS (
  SELECT user_id, order_id, amount, order_date,
    SUM(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_spend,
    SUM(amount) OVER (
      PARTITION BY user_id ORDER BY order_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prev_cumulative
  FROM orders
)
SELECT user_id, order_id, amount, order_date, cumulative_spend
FROM running
WHERE prev_cumulative < 1000
  AND cumulative_spend >= 1000;
```

> The general threshold-crossing template: compute a running total twice — once through the current row, once through the prior row — then filter for the row where the boundary was crossed.

### Variant 3 — Rolling 7-day active users (self-join, not a window frame)

```sql
SELECT DISTINCT a.event_date,
  COUNT(DISTINCT b.user_id) AS rolling_7d_users
FROM events a
JOIN events b
  ON b.event_date BETWEEN a.event_date - INTERVAL 6 DAY AND a.event_date
GROUP BY a.event_date
ORDER BY a.event_date;
```

> Rolling *distinct* counts (like WAU) cannot be done with a simple `ROWS BETWEEN 6 PRECEDING` window frame, because window aggregates don't support `COUNT(DISTINCT ...)` on most engines. A self-join over a date range is the standard workaround.

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Trying `COUNT(DISTINCT x) OVER (...)` | Not supported on most SQL engines (Postgres, MySQL, SQL Server) | Use the self-join pattern in Variant 3, or `APPROX_COUNT_DISTINCT` in BigQuery |
| Off-by-one on threshold crossing | Using `<=` instead of `<` on `prev_cumulative` double-counts or misses the boundary row | Always use strict `<` for "before" and `>=` for "at or after" |
| Attribution double-counting | Forgetting `ROW_NUMBER() = 1` filter includes every purchase, not just the first | Always filter to `rn = 1` before aggregating by channel |

---

## 16. Deduplication via Window Functions

### Concept

`ROW_NUMBER()` partitioned by the "should be unique" key, ordered to prefer the row you want to keep, is the standard dedup pattern — far more flexible than `DISTINCT`.

---

### Variant 1 — Keep only the first (or most recent) row per duplicate group

```sql
-- Keep earliest row per email
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY email ORDER BY created_at ASC) AS rn
    FROM users
)
SELECT * FROM ranked WHERE rn = 1;
```

### Variant 2 — Delete duplicates in place, keeping the most recent row

```sql
WITH ranked AS (
    SELECT ctid,                           -- Postgres physical row id (use PK elsewhere)
           ROW_NUMBER() OVER (PARTITION BY email ORDER BY created_at DESC) AS rn
    FROM users
)
DELETE FROM users
WHERE ctid IN (SELECT ctid FROM ranked WHERE rn > 1);
```

### Variant 3 — Dedup on multiple columns, and audit duplicates without deleting

```sql
-- Composite key dedup
SELECT * FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id, product_id, order_date ORDER BY id) AS rn
    FROM orders
) t
WHERE rn = 1;

-- Audit: find duplicate rows without deleting anything
SELECT * FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY email) AS cnt
    FROM users
) t
WHERE cnt > 1;
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Using RANK for dedup | RANK allows ties, so `rn = 1` can return more than one row | Always use `ROW_NUMBER()` for strict one-row-per-group dedup |
| No secondary sort key | Ties in the ORDER BY produce non-deterministic "keep" choices | Add a tiebreaker column, e.g. `ORDER BY created_at ASC, id ASC` |
| Deleting before verifying | An off-by-one in the CTE can delete the wrong rows permanently | Always `SELECT` and review the CTE output before running `DELETE` |

---

## 17. Pivot / Unpivot (Bonus — Adjacent Skill)

### Concept

Not a window function, but frequently paired with one in the same interview. Turns row values into columns (`PIVOT`) or columns back into rows (`UNPIVOT`).

---

### Variant 1 — Manual pivot via CASE + aggregate (portable to every engine)

```sql
SELECT product_id,
    SUM(CASE WHEN quarter = 'Q1' THEN sales ELSE 0 END) AS q1,
    SUM(CASE WHEN quarter = 'Q2' THEN sales ELSE 0 END) AS q2,
    SUM(CASE WHEN quarter = 'Q3' THEN sales ELSE 0 END) AS q3,
    SUM(CASE WHEN quarter = 'Q4' THEN sales ELSE 0 END) AS q4
FROM sales
GROUP BY product_id;
```

### Variant 2 — Native PIVOT syntax by engine

```sql
-- PostgreSQL (needs tablefunc extension)
CREATE EXTENSION IF NOT EXISTS tablefunc;
SELECT * FROM crosstab(
    'SELECT product_id, quarter, sales FROM sales ORDER BY 1,2'
) AS ct(product_id INT, q1 NUMERIC, q2 NUMERIC, q3 NUMERIC, q4 NUMERIC);

-- SQL Server
SELECT product_id, [Q1], [Q2], [Q3], [Q4]
FROM sales
PIVOT (SUM(sales) FOR quarter IN ([Q1], [Q2], [Q3], [Q4])) AS pvt;

-- BigQuery / Snowflake
SELECT * FROM sales
PIVOT (SUM(sales) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4'));
```

### Variant 3 — UNPIVOT (columns back to rows)

```sql
-- Portable via UNION ALL
SELECT product_id, 'q1' AS quarter, q1 AS sales FROM pivoted_sales
UNION ALL
SELECT product_id, 'q2', q2 FROM pivoted_sales
UNION ALL
SELECT product_id, 'q3', q3 FROM pivoted_sales
UNION ALL
SELECT product_id, 'q4', q4 FROM pivoted_sales;

-- SQL Server native UNPIVOT
SELECT product_id, quarter, sales
FROM pivoted_sales
UNPIVOT (sales FOR quarter IN (q1, q2, q3, q4)) AS unpvt;
```

---

### Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| PIVOT syntax is not portable | Every engine has different native syntax (or none at all, like MySQL) | Default to the manual `CASE` + `SUM` approach unless you know the target engine |
| Hardcoded column lists | Native PIVOT requires listing every output column value upfront | Use dynamic SQL if categories are unknown ahead of time (engine-specific) |

---

## 18. Complete Cheat Sheet

### The OVER() Clause Builder

```sql
<function>() OVER (
    PARTITION BY  <group columns>           -- who resets the window?
    ORDER BY      <sort column> ASC/DESC    -- order within the partition
    ROWS BETWEEN  UNBOUNDED PRECEDING       -- frame start
              AND CURRENT ROW              -- frame end
)
```

### Function Reference

| Category | Function | Key Note |
|----------|----------|----------|
| **Ranking** | `ROW_NUMBER()` | Unique number per row; no ties |
| | `RANK()` | Tied rows share rank; gaps follow (1,1,3) |
| | `DENSE_RANK()` | Tied rows share rank; no gaps (1,1,2) |
| | `NTILE(n)` | Splits into n roughly-equal buckets |
| **Navigation** | `LAG(col, n, default)` | n rows back |
| | `LEAD(col, n, default)` | n rows forward |
| | `FIRST_VALUE(col)` | First row in frame |
| | `LAST_VALUE(col)` | Last row — always extend the frame! |
| | `NTH_VALUE(col, n)` | nth row in frame |
| **Aggregates** | `SUM / AVG / MIN / MAX / COUNT` | All work as window functions |
| **Distribution** | `PERCENT_RANK()` | [0,1]; first row = 0 |
| | `CUME_DIST()` | (0,1]; never 0 |
| **Ordered-set** | `PERCENTILE_CONT(p) WITHIN GROUP (ORDER BY col)` | Interpolated; uses `WITHIN GROUP`, not `OVER` |
| | `PERCENTILE_DISC(p) WITHIN GROUP (ORDER BY col)` | Actual value from the data |

### Frame Quick Reference

| Use Case | Frame |
|----------|-------|
| Running total | `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` |
| Grand total on every row | `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` |
| 7-day trailing window | `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` |
| Centered 3-row window | `ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING` |
| Remaining/suffix total | `ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING` |
| Prior rows only (exclude current) | `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` |

### Interview Pattern Index

| Asked for | Pattern |
|-----------|---------|
| Top N per group | `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) <= N` |
| Day-over-day change | `value - LAG(value) OVER (PARTITION BY ... ORDER BY date)` |
| Running total | `SUM(col) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` |
| Running total, resets per period | `PARTITION BY entity, period` + running-total frame |
| 7-day moving average | `AVG(col) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` |
| Column as % of total | `col * 1.0 / SUM(col) OVER ()` |
| Weighted average | `SUM(value) / SUM(weight)`, not `AVG(daily_avg)` |
| Rank within group | `DENSE_RANK() OVER (PARTITION BY group ORDER BY metric DESC)` |
| Median / P95 latency | `PERCENTILE_CONT(0.5 / 0.95) WITHIN GROUP (ORDER BY col)` |
| Consecutive streak length | Islands: `(date - rn * INTERVAL '1 day')::DATE` as group key |
| Session ID | `SUM(is_new_session_flag) OVER (PARTITION BY user ORDER BY ts)` |
| YoY growth | `LAG(metric, 12) OVER (PARTITION BY entity ORDER BY month)` or self-join |
| Anomaly vs. baseline | `value vs AVG() OVER (... ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)` |
| High-water mark | `MAX(metric) OVER (PARTITION BY ... ORDER BY date ROWS UNBOUNDED PRECEDING...)` |
| Skew detection | `AVG(col) - PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)` |
| First-touch attribution | `ROW_NUMBER() OVER (PARTITION BY user ORDER BY date ASC) = 1` |
| Threshold crossing | Running SUM twice (thru current, thru prior) + `WHERE prev < X AND curr >= X` |
| Rolling distinct active users | Self-join on date range, `COUNT(DISTINCT user_id)` (not a window frame) |
| Deduplication | `ROW_NUMBER() OVER (PARTITION BY dedup_key ORDER BY tiebreaker) = 1` |

### Master Pitfall List (all sections, one place)

| # | Pitfall | One-line fix |
|---|---------|----------------|
| 1 | Filtering a window function in WHERE | Wrap in a CTE first |
| 2 | Forgetting PARTITION BY | Add it — window silently spans the whole table otherwise |
| 3 | RANK vs DENSE_RANK vs ROW_NUMBER confusion | Use ROW_NUMBER for dedup/strict top-1; RANK/DENSE_RANK when ties matter |
| 4 | NTILE bucket sizes uneven | Larger buckets always come first |
| 5 | LAG/LEAD ignoring calendar gaps | Join to a date spine first |
| 6 | Silent RANGE default on running totals | Always write `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` |
| 7 | LAST_VALUE returns current row, not group's last | Always extend frame to `UNBOUNDED FOLLOWING` |
| 8 | Moving average includes partial warm-up windows | Filter on window row count, or document it |
| 9 | Averaging pre-aggregated averages (Simpson's Paradox) | Use `SUM(value)/SUM(weight)` |
| 10 | PERCENTILE_CONT vs DISC confusion | CONT interpolates; DISC returns real data value |
| 11 | Using AVG as a median proxy | Compute both; report median for skewed distributions |
| 12 | Date arithmetic type mismatch in islands trick | Cast explicitly to DATE |
| 13 | Session IDs collide across users | Concatenate user_id with session_id |
| 14 | LAG(12) breaks on missing months | Use a self-join for YoY on sparse data |
| 15 | Dividing by a zero prior-period value | Wrap denominator in `NULLIF(x, 0)` |
| 16 | `COUNT(DISTINCT x) OVER (...)` unsupported | Use a self-join for rolling distinct counts |
| 17 | Deleting dedup rows without reviewing first | Always SELECT and inspect the CTE before DELETE |

---

## 19. Practice Questions (Curated, with Answers)

🟢 **Q1 — Easy (Ranking)**
`employees(emp_id, name, department, salary)` — For each employee, show name, salary, and their `DENSE_RANK` within department by salary (highest = rank 1).

```sql
SELECT name, department, salary,
  DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
```

---

🟢 **Q2 — Easy (Percentiles)**
`employees(emp_id, name, department, salary)` — For each department, show the median salary, the 90th percentile salary, and each employee's percentile rank within their department.

```sql
SELECT name, department, salary,
  PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) OVER (PARTITION BY department) AS median_sal,
  PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY salary) OVER (PARTITION BY department) AS p90_sal,
  ROUND(PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) * 100, 1) AS dept_percentile_rank
FROM employees
ORDER BY department, salary;
```

---

🟡 **Q3 — Medium (LAG)**
`orders(order_id, customer_id, amount, order_date)` — For each customer, show each order and the previous order's amount, plus the difference vs. the current order.

```sql
SELECT customer_id, order_id, amount, order_date,
  LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_order_amount,
  amount - LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS amount_diff
FROM orders;
```

---

🟡 **Q4 — Medium (Gaps & Islands)**
`user_logins(user_id, login_date)` — Find each user's longest consecutive login streak. Return `user_id` and `longest_streak` in days, sorted descending.

```sql
WITH numbered AS (
  SELECT user_id, login_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS rn
  FROM (SELECT DISTINCT user_id, login_date FROM user_logins) t
),
islands AS (
  SELECT user_id,
    DATE_SUB(login_date, INTERVAL rn DAY) AS grp,
    COUNT(*) AS streak_len
  FROM numbered
  GROUP BY user_id, DATE_SUB(login_date, INTERVAL rn DAY)
)
SELECT user_id, MAX(streak_len) AS longest_streak
FROM islands
GROUP BY user_id
ORDER BY longest_streak DESC;
```

---

🔴 **Q5 — Hard (Moving Average + Filtering)**
`daily_sales(sale_date, product_id, revenue)` — For each product, show each day's revenue and a 7-day moving average. Only include products where the **latest** 7-day moving average is above 500.

```sql
WITH moving_avg AS (
  SELECT product_id, sale_date, revenue,
    AVG(revenue) OVER (
      PARTITION BY product_id ORDER BY sale_date
      ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
  FROM daily_sales
),
latest_avg AS (
  SELECT product_id,
    LAST_VALUE(moving_avg_7d) OVER (
      PARTITION BY product_id ORDER BY sale_date
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS latest_7d_avg
  FROM moving_avg
)
SELECT DISTINCT m.product_id, m.sale_date, m.revenue, m.moving_avg_7d
FROM moving_avg m
JOIN latest_avg l ON m.product_id = l.product_id
WHERE l.latest_7d_avg > 500;
```

---

🔴 **Q6 — Hard (Attribution + MoM Growth)**
`orders(order_id, user_id, amount, channel, order_date)` — For each channel in 2025, show: total first-time purchases attributed to it, % of all first purchases that month, average order value of those first purchases, and month-over-month growth in first purchases.

```sql
WITH first_purchases AS (
  SELECT user_id, channel, amount, order_date,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS rn
  FROM orders
),
first_only AS (SELECT * FROM first_purchases WHERE rn = 1),
monthly AS (
  SELECT channel,
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    COUNT(*)    AS first_purchases,
    AVG(amount) AS avg_order_value
  FROM first_only
  WHERE YEAR(order_date) = 2025
  GROUP BY channel, DATE_FORMAT(order_date, '%Y-%m')
)
SELECT channel, month, first_purchases,
  ROUND(first_purchases * 100.0 / SUM(first_purchases) OVER (PARTITION BY month), 2) AS pct_of_month,
  ROUND(avg_order_value, 2) AS avg_order_value,
  ROUND(
    (first_purchases - LAG(first_purchases) OVER (PARTITION BY channel ORDER BY month)) * 100.0 /
    NULLIF(LAG(first_purchases) OVER (PARTITION BY channel ORDER BY month), 0), 2
  ) AS mom_growth_pct
FROM monthly
ORDER BY channel, month;
```

---

*End of master module. Sections 1–14 cover every core window function and pattern; Section 15 covers advanced attribution/threshold/rolling-count patterns that don't reduce to a simple frame; Sections 16–17 cover closely adjacent techniques (dedup, pivot); Section 18 is the single-page cheat sheet; Section 19 is curated practice with full answers.*
