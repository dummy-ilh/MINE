# 2 — Dates, Strings & Interval Arithmetic


## Table of Contents

14. [Date Functions — EXTRACT, DATE_TRUNC, DATE_PART](#14-date-functions--extract-date_trunc-date_part)
15. [Interval Arithmetic — Adding/Subtracting Time](#15-interval-arithmetic--addingsubtracting-time)
16. [Current Timestamps — NOW, CURRENT_DATE, CURRENT_TIMESTAMP](#16-current-timestamps--now-current_date-current_timestamp)
17. [Date Differences — DATEDIFF, AGE, Epoch Arithmetic](#17-date-differences--datediff-age-epoch-arithmetic)
18. [String Functions — UPPER, LOWER, TRIM, LENGTH, REPLACE](#18-string-functions--upper-lower-trim-length-replace)
19. [String Slicing — SUBSTRING, LEFT, RIGHT, POSITION/STRPOS](#19-string-slicing--substring-left-right-positionstrpos)
20. [Pattern Matching — LIKE, ILIKE, SIMILAR TO, Regex](#20-pattern-matching--like-ilike-similar-to-regex)
21. [String Splitting & Parsing — SPLIT_PART, STRING_TO_ARRAY](#21-string-splitting--parsing--split_part-string_to_array)
22. [String Aggregation — STRING_AGG, ARRAY_AGG, GROUP_CONCAT](#22-string-aggregation--string_agg-array_agg-group_concat)
23. [Type Casting — CAST, :: , TO_DATE, TO_TIMESTAMP](#23-type-casting--cast---to_date-to_timestamp)
24. [Combining Date + String — Formatting & Labelling for Reports](#24-combining-date--string--formatting--labelling-for-reports)

---

## 14. Date Functions — EXTRACT, DATE_TRUNC, DATE_PART

### Concept

Dates in SQL are stored as a single value but you often need to **extract a component** (year, month, day, hour) or **truncate** to a bucket (round down to the nearest week/month/year).

| Function | What it does | Returns |
|----------|-------------|---------|
| `EXTRACT(part FROM date)` | Pull one component out of a date | Number |
| `DATE_PART('part', date)` | Same as EXTRACT (PostgreSQL alias) | Float |
| `DATE_TRUNC('part', date)` | Round date *down* to the start of the period | Timestamp |

**Part names:** `year`, `month`, `day`, `hour`, `minute`, `second`, `week`, `quarter`, `dow` (day of week 0=Sun), `doy` (day of year)

---

### Sample Table: `model_training_runs`

| run_id | model_name   | started_at              | finished_at             | status   |
|--------|--------------|-------------------------|-------------------------|----------|
| 1      | XGBoost      | 2024-01-15 08:30:00     | 2024-01-15 09:45:00     | success  |
| 2      | LightGBM     | 2024-01-15 10:00:00     | 2024-01-15 10:20:00     | success  |
| 3      | BERT         | 2024-02-03 14:00:00     | 2024-02-03 22:10:00     | success  |
| 4      | ResNet       | 2024-02-14 09:00:00     | 2024-02-14 09:05:00     | failed   |
| 5      | XGBoost      | 2024-03-01 06:00:00     | 2024-03-01 07:30:00     | success  |

---

### Variant 1 — EXTRACT individual date parts

**Goal:** Pull year, month, day, and hour of each run start.

```sql
SELECT
    run_id,
    model_name,
    EXTRACT(YEAR  FROM started_at) AS run_year,
    EXTRACT(MONTH FROM started_at) AS run_month,
    EXTRACT(DAY   FROM started_at) AS run_day,
    EXTRACT(HOUR  FROM started_at) AS start_hour,
    EXTRACT(DOW   FROM started_at) AS day_of_week  -- 0=Sun, 1=Mon ... 6=Sat
FROM model_training_runs;
```

**Output:**

| run_id | model_name | run_year | run_month | run_day | start_hour | day_of_week |
|--------|------------|----------|-----------|---------|------------|-------------|
| 1      | XGBoost    | 2024     | 1         | 15      | 8          | 1 (Mon)     |
| 2      | LightGBM   | 2024     | 1         | 15      | 10         | 1 (Mon)     |
| 3      | BERT       | 2024     | 2         | 3       | 14         | 6 (Sat)     |
| 4      | ResNet     | 2024     | 2         | 14      | 9          | 3 (Wed)     |
| 5      | XGBoost    | 2024     | 3         | 1       | 6          | 5 (Fri)     |

---

### Variant 2 — DATE_TRUNC to bucket by month

**Goal:** Count successful runs per month.

```sql
SELECT
    DATE_TRUNC('month', started_at) AS month_start,
    COUNT(*)                         AS runs,
    COUNT(CASE WHEN status = 'success' THEN 1 END) AS successful
FROM model_training_runs
GROUP BY DATE_TRUNC('month', started_at)
ORDER BY month_start;
```

**Output:**

| month_start         | runs | successful |
|---------------------|------|------------|
| 2024-01-01 00:00:00 | 2    | 2          |
| 2024-02-01 00:00:00 | 2    | 1          |
| 2024-03-01 00:00:00 | 1    | 1          |

> `DATE_TRUNC('month', '2024-01-15')` → `2024-01-01 00:00:00` — always the first day of the month at midnight.

---

### Variant 3 — DATE_TRUNC with week, finding off-hours runs

**Goal:** Find runs that started outside business hours (before 9am or after 6pm).

```sql
SELECT
    run_id,
    model_name,
    started_at,
    EXTRACT(HOUR FROM started_at) AS start_hour
FROM model_training_runs
WHERE EXTRACT(HOUR FROM started_at) < 9
   OR EXTRACT(HOUR FROM started_at) >= 18;
```

**Output:**

| run_id | model_name | started_at          | start_hour |
|--------|------------|---------------------|------------|
| 5      | XGBoost    | 2024-03-01 06:00:00 | 6          |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `DATE_TRUNC` returns a timestamp, not a date | Comparing to a plain DATE may fail or coerce | Cast: `DATE_TRUNC('month', ts)::DATE` |
| `EXTRACT(DOW ...)` is 0-indexed in Postgres, 1-indexed in MySQL | Sunday = 0 in PG, Sunday = 1 in MySQL | Check DB docs; use `TO_CHAR(date, 'Day')` for readability |
| Grouping by a timestamp instead of truncated date | Every unique second gets its own group | Always `DATE_TRUNC` before grouping by time |
| `EXTRACT(WEEK ...)` ambiguity | ISO week vs calendar week differ at year boundaries | Use `EXTRACT(ISOYEAR ...)` + `EXTRACT(WEEK ...)` together for ISO week numbers |

---

## 15. Interval Arithmetic — Adding/Subtracting Time

### Concept

SQL lets you add or subtract **intervals** directly to dates and timestamps. This is how you compute "7 days ago", "next month", "90-day windows", etc.

```sql
-- PostgreSQL / Standard SQL
date_col + INTERVAL '7 days'
date_col - INTERVAL '3 months'
date_col + INTERVAL '1 year 6 months'

-- BigQuery
DATE_ADD(date_col, INTERVAL 7 DAY)
DATE_SUB(date_col, INTERVAL 3 MONTH)

-- MySQL (same as BigQuery syntax)
DATE_ADD(date_col, INTERVAL 7 DAY)
```

---

### Sample Table: `experiments`

| exp_id | model_name | start_date | duration_days |
|--------|------------|------------|---------------|
| 1      | XGBoost    | 2024-01-10 | 14            |
| 2      | LightGBM   | 2024-02-01 | 7             |
| 3      | BERT       | 2024-03-15 | 30            |
| 4      | ResNet     | 2024-04-01 | 60            |

---

### Variant 1 — Compute experiment end date

```sql
-- PostgreSQL
SELECT
    exp_id,
    model_name,
    start_date,
    duration_days,
    start_date + (duration_days || ' days')::INTERVAL AS end_date
FROM experiments;
```

**Output:**

| exp_id | model_name | start_date | duration_days | end_date   |
|--------|------------|------------|---------------|------------|
| 1      | XGBoost    | 2024-01-10 | 14            | 2024-01-24 |
| 2      | LightGBM   | 2024-02-01 | 7             | 2024-02-08 |
| 3      | BERT       | 2024-03-15 | 30            | 2024-04-14 |
| 4      | ResNet     | 2024-04-01 | 60            | 2024-05-31 |

> `(duration_days || ' days')::INTERVAL` converts the integer into a dynamic interval string.

---

### Variant 2 — Find experiments active in the last 30 days

```sql
-- Assuming today is 2024-04-20
SELECT
    exp_id,
    model_name,
    start_date,
    start_date + (duration_days || ' days')::INTERVAL AS end_date
FROM experiments
WHERE start_date + (duration_days || ' days')::INTERVAL >= CURRENT_DATE - INTERVAL '30 days'
  AND start_date <= CURRENT_DATE;
```

**Output (as of 2024-04-20):**

| exp_id | model_name | start_date | end_date   |
|--------|------------|------------|------------|
| 3      | BERT       | 2024-03-15 | 2024-04-14 |
| 4      | ResNet     | 2024-04-01 | 2024-05-31 |

---

### Variant 3 — Rolling 7-day window label

**Goal:** Assign each experiment to a weekly cohort bucket.

```sql
SELECT
    exp_id,
    model_name,
    start_date,
    DATE_TRUNC('week', start_date)                        AS week_start,
    DATE_TRUNC('week', start_date) + INTERVAL '6 days'   AS week_end
FROM experiments
ORDER BY start_date;
```

**Output:**

| exp_id | model_name | start_date | week_start | week_end   |
|--------|------------|------------|------------|------------|
| 1      | XGBoost    | 2024-01-10 | 2024-01-08 | 2024-01-14 |
| 2      | LightGBM   | 2024-02-01 | 2024-01-29 | 2024-02-04 |
| 3      | BERT       | 2024-03-15 | 2024-03-11 | 2024-03-17 |
| 4      | ResNet     | 2024-04-01 | 2024-04-01 | 2024-04-07 |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Adding integer directly to a date | `date_col + 7` works in some DBs (MySQL) but not others | Explicit: `date_col + INTERVAL '7 days'` |
| Month arithmetic crosses month-end | `Jan 31 + 1 month` = Feb 28 (or 29), not Mar 2 | Expected; use `DATE_TRUNC` to normalise if needed |
| Interval with float days | `INTERVAL '1.5 days'` is valid but confusing | Use hours: `INTERVAL '36 hours'` |
| Dynamic intervals from a column | `date + duration_days * INTERVAL '1 day'` is cleanest | Avoid string concatenation when multiplying works |

---

## 16. Current Timestamps — NOW, CURRENT_DATE, CURRENT_TIMESTAMP

### Concept

| Function | Returns | Timezone-aware? |
|----------|---------|-----------------|
| `NOW()` | Current date + time (with timezone) | Yes |
| `CURRENT_TIMESTAMP` | Same as NOW() — SQL standard | Yes |
| `CURRENT_DATE` | Today's date only, no time | No |
| `CURRENT_TIME` | Current time only, no date | Yes |
| `LOCALTIMESTAMP` | Current date + time, no timezone | No |

**Key rule:** `NOW()` is evaluated **once per query/transaction** and stays fixed throughout. This is important — it won't drift mid-query.

---

### Sample Table: `deployments`

| deploy_id | model_name | deployed_at             | retired_at              |
|-----------|------------|-------------------------|-------------------------|
| 1         | XGBoost    | 2024-01-01 00:00:00 UTC | NULL                    |
| 2         | LightGBM   | 2024-02-15 00:00:00 UTC | 2024-03-15 00:00:00 UTC |
| 3         | BERT       | 2024-03-01 00:00:00 UTC | NULL                    |

---

### Variant 1 — Find currently active deployments

```sql
SELECT
    deploy_id,
    model_name,
    deployed_at
FROM deployments
WHERE deployed_at <= NOW()
  AND (retired_at IS NULL OR retired_at > NOW());
```

**Output (as of 2024-04-20):**

| deploy_id | model_name | deployed_at             |
|-----------|------------|-------------------------|
| 1         | XGBoost    | 2024-01-01 00:00:00 UTC |
| 3         | BERT       | 2024-03-01 00:00:00 UTC |

---

### Variant 2 — Days since deployment

```sql
SELECT
    model_name,
    deployed_at::DATE AS deploy_date,
    (CURRENT_DATE - deployed_at::DATE) AS days_live
FROM deployments
WHERE retired_at IS NULL
ORDER BY days_live DESC;
```

**Output:**

| model_name | deploy_date | days_live |
|------------|-------------|-----------|
| XGBoost    | 2024-01-01  | 110       |
| BERT       | 2024-03-01  | 50        |

---

### Variant 3 — Audit log: tag records created in the last 24 hours

```sql
SELECT
    deploy_id,
    model_name,
    deployed_at,
    CASE
        WHEN deployed_at >= NOW() - INTERVAL '24 hours' THEN 'recent'
        ELSE 'older'
    END AS recency_tag
FROM deployments
ORDER BY deployed_at DESC;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `NOW()` vs `CURRENT_DATE` type mismatch | Comparing a TIMESTAMPTZ column to CURRENT_DATE can cause implicit casts | Cast explicitly: `deployed_at::DATE = CURRENT_DATE` |
| Timezone confusion | `NOW()` returns UTC in most cloud DBs; your app may show local time | Always store and compare in UTC; convert at display layer |
| Using `NOW()` in WHERE on indexed timestamp columns | Some DBs can't use index if timestamp is cast to date | Compare ranges: `WHERE ts >= CURRENT_DATE AND ts < CURRENT_DATE + 1` |
| `GETDATE()` in SQL Server | Same as `NOW()` but SQL Server syntax | Check which DB you're on in the interview |

---

## 17. Date Differences — DATEDIFF, AGE, Epoch Arithmetic

### Concept

Computing the gap between two dates is a top-3 interview date skill. Every DB has slightly different syntax:

| DB | Syntax | Returns |
|----|--------|---------|
| PostgreSQL | `date1 - date2` | integer (days) when both are DATE |
| PostgreSQL | `AGE(ts1, ts2)` | interval (e.g. "2 months 5 days") |
| BigQuery | `DATE_DIFF(date1, date2, DAY)` | integer |
| MySQL / Redshift | `DATEDIFF(date1, date2)` | integer (days) |
| All DBs | Epoch subtraction | seconds, then divide |

---

### Sample Table: `model_training_runs` (reused)

| run_id | model_name | started_at          | finished_at         |
|--------|------------|---------------------|---------------------|
| 1      | XGBoost    | 2024-01-15 08:30:00 | 2024-01-15 09:45:00 |
| 2      | LightGBM   | 2024-01-15 10:00:00 | 2024-01-15 10:20:00 |
| 3      | BERT       | 2024-02-03 14:00:00 | 2024-02-03 22:10:00 |
| 4      | ResNet     | 2024-02-14 09:00:00 | 2024-02-14 09:05:00 |

---

### Variant 1 — Training duration in minutes (PostgreSQL)

```sql
SELECT
    run_id,
    model_name,
    EXTRACT(EPOCH FROM (finished_at - started_at)) / 60 AS duration_minutes
FROM model_training_runs
ORDER BY duration_minutes DESC;
```

**How it works:** `finished_at - started_at` gives an INTERVAL. `EXTRACT(EPOCH FROM interval)` converts it to total seconds. Divide by 60 for minutes.

**Output:**

| run_id | model_name | duration_minutes |
|--------|------------|-----------------|
| 3      | BERT       | 490.00           |
| 1      | XGBoost    | 75.00            |
| 2      | LightGBM   | 20.00            |
| 4      | ResNet     | 5.00             |

---

### Variant 2 — AGE function for human-readable difference

```sql
SELECT
    run_id,
    model_name,
    AGE(finished_at, started_at) AS duration
FROM model_training_runs;
```

**Output:**

| run_id | model_name | duration        |
|--------|------------|-----------------|
| 1      | XGBoost    | 01:15:00        |
| 2      | LightGBM   | 00:20:00        |
| 3      | BERT       | 08:10:00        |
| 4      | ResNet     | 00:05:00        |

> Good for display; bad for arithmetic. Use EPOCH for calculations.

---

### Variant 3 — Days between deployment and first success (multi-table)

```sql
-- deployments table has: deploy_id, model_name, deployed_at
-- model_training_runs has: run_id, model_name, started_at, status

WITH first_success AS (
    SELECT
        model_name,
        MIN(started_at) AS first_success_at
    FROM model_training_runs
    WHERE status = 'success'
    GROUP BY model_name
)
SELECT
    d.model_name,
    d.deployed_at::DATE AS deploy_date,
    fs.first_success_at::DATE AS first_success_date,
    (fs.first_success_at::DATE - d.deployed_at::DATE) AS days_to_first_success
FROM deployments d
JOIN first_success fs ON d.model_name = fs.model_name
ORDER BY days_to_first_success;
```

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `date1 - date2` on TIMESTAMP returns INTERVAL, not integer | `EXTRACT` on an interval can be surprising for multi-day intervals | Use EPOCH division for any duration in consistent units |
| `DATEDIFF` argument order varies | MySQL: `DATEDIFF(end, start)` = positive. Swap order → negative | Always test with a known pair |
| AGE() argument order | `AGE(later, earlier)` = positive interval | Remember: AGE(end, start) |
| Month/year differences are imprecise in days | Feb has 28/29 days; "1 month" is ambiguous | For day-precision use EPOCH; for calendar units use EXTRACT(YEAR/MONTH) |

---

## 18. String Functions — UPPER, LOWER, TRIM, LENGTH, REPLACE

### Concept

These are the "cleanup" functions — used constantly for data normalisation before joins or comparisons.

| Function | What it does |
|----------|-------------|
| `UPPER(str)` | Convert to uppercase |
| `LOWER(str)` | Convert to lowercase |
| `TRIM(str)` | Remove leading + trailing spaces |
| `LTRIM(str)` | Remove leading spaces only |
| `RTRIM(str)` | Remove trailing spaces only |
| `LENGTH(str)` | Number of characters |
| `REPLACE(str, from, to)` | Replace all occurrences of a substring |
| `CONCAT(a, b, ...)` or `a \|\| b` | Concatenate strings |

trim('ccchelloccc',c) --hello
---

### Sample Table: `raw_model_registry`

| id | model_name        | framework   | team_email            |
|----|-------------------|-------------|-----------------------|
| 1  | ' XGBoost '      | xgboost     | SEARCH@company.com    |
| 2  | 'lightgbm_v2'    | LightGBM    | ads@Company.COM       |
| 3  | 'BERT-base'      | transformers| NLP@COMPANY.com       |
| 4  | 'resnet-50  '    | pytorch     | vision@company.com    |

---

### Variant 1 — Normalise messy model names and emails

```sql
SELECT
    id,
    TRIM(LOWER(model_name))       AS clean_name,
    LOWER(team_email)             AS clean_email,
    LENGTH(TRIM(model_name))      AS name_length
FROM raw_model_registry;
```

**Output:**

| id | clean_name    | clean_email            | name_length |
|----|---------------|------------------------|-------------|
| 1  | xgboost       | search@company.com     | 7           |
| 2  | lightgbm_v2   | ads@company.com        | 12          |
| 3  | bert-base     | nlp@company.com        | 9           |
| 4  | resnet-50     | vision@company.com     | 9           |

---

### Variant 2 — REPLACE to standardise separators

**Goal:** Normalise model names — replace hyphens and underscores with spaces.

```sql
SELECT
    id,
    REPLACE(
        REPLACE(TRIM(LOWER(model_name)), '_', ' '),
        '-', ' '
    ) AS normalised_name
FROM raw_model_registry;
```

**Output:**

| id | normalised_name |
|----|-----------------|
| 1  | xgboost         |
| 2  | lightgbm v2     |
| 3  | bert base       |
| 4  | resnet 50       |

---

### Variant 3 — CONCAT to build labels for reports

```sql
SELECT
    id,
    UPPER(TRIM(model_name)) || ' (' || LOWER(framework) || ')' AS display_label,
    'mailto:' || LOWER(team_email) AS mailto_link
FROM raw_model_registry;
```

**Output:**

| id | display_label           | mailto_link                     |
|----|-------------------------|---------------------------------|
| 1  | XGBOOST (xgboost)       | mailto:search@company.com       |
| 2  | LIGHTGBM_V2 (lightgbm)  | mailto:ads@company.com          |
| 3  | BERT-BASE (transformers) | mailto:nlp@company.com         |
| 4  | RESNET-50 (pytorch)      | mailto:vision@company.com      |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Case-sensitive JOINs fail on dirty data | `'XGBoost' = 'xgboost'` → false | `LOWER(a.name) = LOWER(b.name)` on both sides |
| TRIM only removes spaces | Doesn't remove tabs (`\t`) or newlines (`\n`) | Use `REGEXP_REPLACE(col, '\s+', ' ')` for all whitespace |
| LENGTH vs CHAR_LENGTH | For multi-byte (UTF-8) strings, LENGTH may return bytes not chars | Use `CHAR_LENGTH` in MySQL; `LENGTH` in PG counts chars for text type |
| `||` with NULL | `'hello' || NULL` → NULL in SQL standard | Use `CONCAT_WS` or `COALESCE` |

---

## 19. String Slicing — SUBSTRING, LEFT, RIGHT, POSITION/STRPOS

### Concept

| Function | Syntax | What it does |
|----------|--------|-------------|
| `SUBSTRING` | `SUBSTRING(str, start, length)` | Extract from position `start` for `length` chars (1-indexed) |
| `LEFT` | `LEFT(str, n)` | First n characters |
| `RIGHT` | `RIGHT(str, n)` | Last n characters |
| `POSITION` | `POSITION(sub IN str)` | Find start position of substring (0 = not found) |
| `STRPOS` | `STRPOS(str, sub)` | Same, PostgreSQL syntax |

---

### Sample Table: `log_events`

| log_id | log_message                              |
|--------|------------------------------------------|
| 1      | ERROR:2024-01-15:model_inference_timeout |
| 2      | INFO:2024-01-16:batch_job_complete       |
| 3      | WARN:2024-02-01:memory_threshold_reached |
| 4      | ERROR:2024-02-14:gpu_out_of_memory       |

---

### Variant 1 — Extract log level and date using LEFT/POSITION

```sql
SELECT
    log_id,
    SUBSTRING(log_message, 1, POSITION(':' IN log_message) - 1)  AS log_level,
    SUBSTRING(log_message, POSITION(':' IN log_message) + 1, 10) AS log_date
FROM log_events;
```

**How it works:**
- `POSITION(':' IN log_message)` finds the first colon → e.g. position 6 for "ERROR:"
- `SUBSTRING(str, 1, 5)` grabs characters 1 through 5 = "ERROR"
- Then shift past the colon and grab 10 chars for the date

**Output:**

| log_id | log_level | log_date   |
|--------|-----------|------------|
| 1      | ERROR     | 2024-01-15 |
| 2      | INFO      | 2024-01-16 |
| 3      | WARN      | 2024-02-01 |
| 4      | ERROR     | 2024-02-14 |

---

### Variant 2 — RIGHT to extract file extension

```sql
-- Model artifact filenames: 'xgboost_v2.pkl', 'bert_base.onnx'
SELECT
    filename,
    RIGHT(filename, LENGTH(filename) - STRPOS(filename, '.')) AS extension,
    LEFT(filename, STRPOS(filename, '.') - 1)                 AS base_name
FROM model_artifacts;
```

**Output** (sample):

| filename        | extension | base_name   |
|-----------------|-----------|-------------|
| xgboost_v2.pkl  | pkl       | xgboost_v2  |
| bert_base.onnx  | onnx      | bert_base   |

---

### Variant 3 — SUBSTRING with regex (PostgreSQL)

```sql
-- Extract model name from free-text description
SELECT
    log_id,
    SUBSTRING(log_message FROM 'model_([a-z_]+)') AS extracted_model
FROM log_events;
```

**Output:**

| log_id | extracted_model   |
|--------|-------------------|
| 1      | inference_timeout |
| 2      | NULL              |
| 3      | NULL              |
| 4      | NULL              |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| SQL strings are 1-indexed | `SUBSTRING(str, 0, 3)` behaves like `SUBSTRING(str, 1, 2)` in PG | Always start from index 1 |
| POSITION returns 0 when not found | `SUBSTRING(str, 0, ...)` on a missing delimiter gives wrong results | Guard: `CASE WHEN POSITION(':' IN col) > 0 THEN ... END` |
| Different DB names | `STRPOS` (PG) = `LOCATE` (MySQL) = `CHARINDEX` (SQL Server) | Know your DB in the interview |
| SUBSTRING vs SUBSTR | Both work in PG; MySQL uses `SUBSTR` | Safe to use `SUBSTRING` universally |

---

## 20. Pattern Matching — LIKE, ILIKE, SIMILAR TO, Regex

### Concept

| Operator | Case-sensitive | Wildcards | Power |
|----------|---------------|-----------|-------|
| `LIKE` | Yes | `%` (any chars), `_` (one char) | Low |
| `ILIKE` | No (PG only) | Same as LIKE | Low |
| `SIMILAR TO` | Yes | LIKE + basic regex (`|`, `*`, `+`) | Medium |
| `~` (PG) | Yes | Full POSIX regex | High |
| `~*` (PG) | No | Full POSIX regex | High |
| `REGEXP_LIKE` | Varies | Full regex | High |

---

### Sample Table: `feature_store`

| feature_id | feature_name            | data_type |
|------------|-------------------------|-----------|
| 1          | user_age                | int       |
| 2          | user_click_rate_7d      | float     |
| 3          | item_price_usd          | float     |
| 4          | user_session_count_30d  | int       |
| 5          | model_score_xgb         | float     |
| 6          | USER_COUNTRY            | string    |

---

### Variant 1 — LIKE for prefix/suffix matching

```sql
-- All user-related features
SELECT feature_name
FROM feature_store
WHERE feature_name LIKE 'user_%';
```

**Output:**

| feature_name           |
|------------------------|
| user_age               |
| user_click_rate_7d     |
| user_session_count_30d |

```sql
-- All features ending in a time window (_7d, _30d)
SELECT feature_name
FROM feature_store
WHERE feature_name LIKE '%_7d'
   OR feature_name LIKE '%_30d';
```

**Output:**

| feature_name           |
|------------------------|
| user_click_rate_7d     |
| user_session_count_30d |

---

### Variant 2 — ILIKE for case-insensitive search (PostgreSQL)

```sql
SELECT feature_name
FROM feature_store
WHERE feature_name ILIKE 'user%';  -- matches 'user_age' AND 'USER_COUNTRY'
```

**Output:**

| feature_name           |
|------------------------|
| user_age               |
| user_click_rate_7d     |
| user_session_count_30d |
| USER_COUNTRY           |

---

### Variant 3 — Regex for precise patterns

```sql
-- Features that end with a number (time windows like _7d, _30d, _90d)
SELECT feature_name
FROM feature_store
WHERE feature_name ~ '_\d+d$';   -- underscore, one or more digits, 'd', end of string
```

**Output:**

| feature_name           |
|------------------------|
| user_click_rate_7d     |
| user_session_count_30d |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `LIKE '%x%'` is slow on large tables | Can't use a B-tree index for leading wildcard | Use full-text search (`TSVECTOR`) or trigram index (`pg_trgm`) |
| `_` in LIKE is a wildcard, not literal | `WHERE name LIKE 'user_age'` matches "userXage" too | Escape: `WHERE name LIKE 'user\_age' ESCAPE '\'` |
| SIMILAR TO is rarely used | Confusing mix of LIKE and regex; most devs use `~` instead | Stick to LIKE or POSIX regex |
| `~` is PostgreSQL only | MySQL uses `REGEXP`, BigQuery uses `REGEXP_CONTAINS` | Always check DB syntax |

---

## 21. String Splitting & Parsing — SPLIT_PART, STRING_TO_ARRAY

### Concept

When a column contains delimited values (comma-separated tags, colon-separated log fields, pipe-separated categories), you need to split them.

| Function | DB | What it does |
|----------|----|-------------|
| `SPLIT_PART(str, delim, n)` | PostgreSQL | Returns the nth segment |
| `STRING_TO_ARRAY(str, delim)` | PostgreSQL | Returns an array of parts |
| `SUBSTRING_INDEX(str, delim, n)` | MySQL | Returns left n segments |
| `SPLIT(str, delim)` | BigQuery | Returns array |
| `UNNEST(array)` | PG/BQ | Explodes array to rows |

---

### Sample Table: `model_tags`

| model_id | model_name | tags                          |
|----------|------------|-------------------------------|
| 1        | XGBoost    | classification,tabular,prod   |
| 2        | BERT       | nlp,text,classification,prod  |
| 3        | ResNet     | vision,image,staging          |
| 4        | LightGBM   | tabular,regression,prod       |

---

### Variant 1 — SPLIT_PART to extract the first tag

```sql
SELECT
    model_name,
    SPLIT_PART(tags, ',', 1) AS primary_tag,
    SPLIT_PART(tags, ',', 2) AS secondary_tag
FROM model_tags;
```

**Output:**

| model_name | primary_tag    | secondary_tag |
|------------|----------------|---------------|
| XGBoost    | classification | tabular       |
| BERT       | nlp            | text          |
| ResNet     | vision         | image         |
| LightGBM   | tabular        | regression    |

---

### Variant 2 — UNNEST to explode tags into rows

```sql
SELECT
    model_name,
    UNNEST(STRING_TO_ARRAY(tags, ',')) AS tag
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
| LightGBM   | prod           |
| LightGBM   | regression     |
| LightGBM   | tabular        |
| ResNet     | image          |
| ResNet     | staging        |
| ResNet     | vision         |
| XGBoost    | classification |
| XGBoost    | prod           |
| XGBoost    | tabular        |

---

### Variant 3 — Count models per tag (after exploding)

```sql
WITH exploded AS (
    SELECT
        model_name,
        UNNEST(STRING_TO_ARRAY(tags, ',')) AS tag
    FROM model_tags
)
SELECT
    tag,
    COUNT(DISTINCT model_name) AS model_count
FROM exploded
GROUP BY tag
ORDER BY model_count DESC, tag;
```

**Output:**

| tag            | model_count |
|----------------|-------------|
| prod           | 3           |
| classification | 2           |
| tabular        | 2           |
| image          | 1           |
| nlp            | 1           |
| regression     | 1           |
| staging        | 1           |
| text           | 1           |
| vision         | 1           |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `SPLIT_PART` returns empty string (not NULL) if index exceeds parts | `SPLIT_PART('a,b', ',', 5)` → `''` | Use `NULLIF(SPLIT_PART(...), '')` |
| UNNEST multiplies rows | A model with 4 tags becomes 4 rows | Remember to GROUP BY model_name if aggregating back |
| Leading/trailing spaces in tags | `'prod '` ≠ `'prod'` after split | `TRIM(UNNEST(...))` |
| Storing lists in a column | Violates 1NF; makes queries complex | Ideal schema uses a separate `model_tags` table with one tag per row |

---

## 22. String Aggregation — STRING_AGG, ARRAY_AGG, GROUP_CONCAT

### Concept

The inverse of UNNEST — collapse multiple rows into a single delimited string or array.

| Function | DB | Output type |
|----------|----|-------------|
| `STRING_AGG(col, delim)` | PostgreSQL, BigQuery | String |
| `GROUP_CONCAT(col SEPARATOR ',')` | MySQL | String |
| `LISTAGG(col, ',')` | Oracle, Redshift | String |
| `ARRAY_AGG(col)` | PostgreSQL, BigQuery | Array |

All support `ORDER BY` inside the aggregation.

---

### Sample Table: `model_experiments` (one row per model-tag)

| model_name | tag            | accuracy |
|------------|----------------|----------|
| XGBoost    | classification | 0.91     |
| XGBoost    | tabular        | 0.91     |
| XGBoost    | prod           | 0.91     |
| BERT       | nlp            | 0.87     |
| BERT       | classification | 0.87     |
| LightGBM   | tabular        | 0.94     |

---

### Variant 1 — STRING_AGG to collapse tags per model

```sql
SELECT
    model_name,
    MAX(accuracy)                                    AS accuracy,
    STRING_AGG(tag, ', ' ORDER BY tag)              AS all_tags,
    COUNT(tag)                                       AS tag_count
FROM model_experiments
GROUP BY model_name
ORDER BY accuracy DESC;
```

**Output:**

| model_name | accuracy | all_tags                      | tag_count |
|------------|----------|-------------------------------|-----------|
| LightGBM   | 0.94     | tabular                       | 1         |
| XGBoost    | 0.91     | classification, prod, tabular | 3         |
| BERT       | 0.87     | classification, nlp           | 2         |

---

### Variant 2 — ARRAY_AGG for array output

```sql
SELECT
    model_name,
    ARRAY_AGG(tag ORDER BY tag) AS tag_array
FROM model_experiments
GROUP BY model_name;
```

**Output:**

| model_name | tag_array                          |
|------------|------------------------------------|
| XGBoost    | {classification, prod, tabular}    |
| BERT       | {classification, nlp}              |
| LightGBM   | {tabular}                          |

---

### Variant 3 — Find models that share a tag (self-join after aggregation)

```sql
WITH model_tags AS (
    SELECT model_name, STRING_AGG(tag, ',' ORDER BY tag) AS tags
    FROM model_experiments
    GROUP BY model_name
)
SELECT a.model_name AS model_a, b.model_name AS model_b, a.tags AS shared_tag_signature
FROM model_tags a
JOIN model_tags b ON a.tags = b.tags AND a.model_name < b.model_name;
```

> No output here (no two models share the exact same tag set), but this pattern finds duplicate feature profiles — useful in ML for detecting identical feature groups.

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| `STRING_AGG` without ORDER BY | Non-deterministic output order | Always add `ORDER BY` inside the function |
| NULLs in STRING_AGG | NULLs are silently skipped | Use `COALESCE(col, 'UNKNOWN')` before aggregating if you need them |
| Very long concatenated strings | Can hit VARCHAR size limits | Know your DB's limit; consider ARRAY_AGG instead |
| `GROUP_CONCAT` default separator is comma | MySQL's default; easy to forget when parsing output | Always specify: `GROUP_CONCAT(col SEPARATOR '|')` |

---

## 23. Type Casting — CAST, ::, TO_DATE, TO_TIMESTAMP

### Concept

SQL is strictly typed — mixing types causes errors or silent wrong results. Casting converts a value from one type to another.

```sql
-- Two equivalent cast syntaxes (PostgreSQL)
CAST(col AS INTEGER)
col::INTEGER

-- String to date
TO_DATE('2024-01-15', 'YYYY-MM-DD')
TO_TIMESTAMP('2024-01-15 08:30:00', 'YYYY-MM-DD HH24:MI:SS')

-- Date to string (formatting)
TO_CHAR(date_col, 'YYYY-MM')      -- '2024-01'
TO_CHAR(date_col, 'Month YYYY')   -- 'January 2024'
TO_CHAR(number, 'FM999,999.00')   -- '1,234.56'
```

---

### Sample Table: `raw_imports`

| id | score_text | date_text        | flag |
|----|------------|------------------|------|
| 1  | '0.91'     | '15/01/2024'     | '1'  |
| 2  | '0.85'     | '03-Feb-2024'    | '0'  |
| 3  | '0.78'     | '2024-03-01'     | '1'  |
| 4  | 'N/A'      | '2024-04-10'     | '0'  |

---

### Variant 1 — Safe cast of scores (handle N/A)

```sql
SELECT
    id,
    CASE
        WHEN score_text = 'N/A' THEN NULL
        ELSE CAST(score_text AS FLOAT)
    END AS score,
    CAST(flag AS INTEGER) AS flag_int
FROM raw_imports;
```

**Output:**

| id | score | flag_int |
|----|-------|----------|
| 1  | 0.91  | 1        |
| 2  | 0.85  | 0        |
| 3  | 0.78  | 1        |
| 4  | NULL  | 0        |

---

### Variant 2 — TO_DATE for mixed date formats

```sql
SELECT
    id,
    CASE
        WHEN date_text LIKE '__/__/____'
            THEN TO_DATE(date_text, 'DD/MM/YYYY')
        WHEN date_text LIKE '__-___-____'
            THEN TO_DATE(date_text, 'DD-Mon-YYYY')
        ELSE
            date_text::DATE
    END AS parsed_date
FROM raw_imports;
```

**Output:**

| id | parsed_date |
|----|-------------|
| 1  | 2024-01-15  |
| 2  | 2024-02-03  |
| 3  | 2024-03-01  |
| 4  | 2024-04-10  |

---

### Variant 3 — TO_CHAR for report-friendly formatting

```sql
SELECT
    model_name,
    TO_CHAR(started_at, 'DD Mon YYYY HH24:MI')  AS run_label,
    TO_CHAR(started_at, 'YYYY-"W"WW')           AS iso_week,
    TO_CHAR(accuracy * 100, 'FM990.0"%"')       AS accuracy_display
FROM model_training_runs
JOIN model_scores USING (model_name);
```

**Output (sample):**

| model_name | run_label         | iso_week  | accuracy_display |
|------------|-------------------|-----------|-----------------|
| XGBoost    | 15 Jan 2024 08:30 | 2024-W03  | 91.0%           |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Implicit cast of string to date | `WHERE date_col = '2024-01-15'` may work but relies on DB auto-cast | Cast explicitly for clarity |
| `::DATE` fails on ambiguous formats | `'01/02/2024'` — is it Jan 2 or Feb 1? | Use `TO_DATE` with explicit format string |
| Integer division before cast | `1/2 = 0`, then cast to float = `0.0` | Cast first: `1::FLOAT / 2` or `1.0 / 2` |
| `TO_CHAR` decimal formatting | `TO_CHAR(0.91, '0.99')` → `' 0.91'` with leading space | Use `FM` prefix: `TO_CHAR(0.91, 'FM0.99')` = `'0.91'` |

---

## 24. Combining Date + String — Formatting & Labelling for Reports

### Concept

In ML interviews, you'll often be asked to produce **cohort labels**, **time-bucketed summaries**, and **human-readable report columns**. This topic ties date functions and string functions together.

---

### Sample Table: `user_events`

| user_id | event_type | event_ts            | country |
|---------|------------|---------------------|---------|
| 1       | signup     | 2024-01-03 09:00:00 | US      |
| 2       | signup     | 2024-01-15 14:00:00 | UK      |
| 3       | purchase   | 2024-02-05 10:30:00 | US      |
| 4       | signup     | 2024-02-20 08:00:00 | DE      |
| 5       | purchase   | 2024-03-10 16:00:00 | US      |

---

### Variant 1 — Monthly cohort label

```sql
SELECT
    user_id,
    event_type,
    TO_CHAR(event_ts, 'YYYY-MM')                          AS cohort_month,
    TO_CHAR(event_ts, 'Mon YYYY')                         AS cohort_label,
    country || '-' || TO_CHAR(event_ts, 'YYYY-MM')        AS country_cohort
FROM user_events
ORDER BY event_ts;
```

**Output:**

| user_id | event_type | cohort_month | cohort_label | country_cohort |
|---------|------------|--------------|--------------|----------------|
| 1       | signup     | 2024-01      | Jan 2024     | US-2024-01     |
| 2       | signup     | 2024-01      | Jan 2024     | UK-2024-01     |
| 3       | purchase   | 2024-02      | Feb 2024     | US-2024-02     |
| 4       | signup     | 2024-02      | Feb 2024     | DE-2024-02     |
| 5       | purchase   | 2024-03      | Mar 2024     | US-2024-03     |

---

### Variant 2 — Cohort retention summary (signups → purchases)

```sql
WITH signups AS (
    SELECT user_id, DATE_TRUNC('month', event_ts) AS signup_month
    FROM user_events
    WHERE event_type = 'signup'
),
purchases AS (
    SELECT user_id, DATE_TRUNC('month', event_ts) AS purchase_month
    FROM user_events
    WHERE event_type = 'purchase'
)
SELECT
    TO_CHAR(s.signup_month, 'YYYY-MM')        AS cohort,
    COUNT(DISTINCT s.user_id)                 AS signups,
    COUNT(DISTINCT p.user_id)                 AS converted,
    ROUND(
        COUNT(DISTINCT p.user_id) * 100.0
        / NULLIF(COUNT(DISTINCT s.user_id), 0),
        1
    )                                         AS conversion_pct
FROM signups s
LEFT JOIN purchases p ON s.user_id = p.user_id
GROUP BY s.signup_month
ORDER BY s.signup_month;
```

**Output:**

| cohort  | signups | converted | conversion_pct |
|---------|---------|-----------|----------------|
| 2024-01 | 2       | 1         | 50.0           |
| 2024-02 | 1       | 0         | 0.0            |

---

### Variant 3 — Day-of-week traffic heatmap label

```sql
SELECT
    TO_CHAR(event_ts, 'Dy')   AS day_abbrev,   -- Mon, Tue, ...
    EXTRACT(DOW FROM event_ts) AS dow_number,   -- 0=Sun
    COUNT(*)                   AS event_count
FROM user_events
GROUP BY TO_CHAR(event_ts, 'Dy'), EXTRACT(DOW FROM event_ts)
ORDER BY dow_number;
```

**Output:**

| day_abbrev | dow_number | event_count |
|------------|------------|-------------|
| Wed        | 3          | 1           |
| Mon        | 1          | 1           |
| Tue        | 2          | 1           |
| Tue        | 2          | 1           |
| Sun        | 0          | 1           |

---

### ⚠️ Pitfalls

| Pitfall | Explanation | Fix |
|---------|-------------|-----|
| Cohort label as string vs date | Grouping by `TO_CHAR(ts, 'YYYY-MM')` works but sorts as string, not date | Group by `DATE_TRUNC('month', ts)` and format only in SELECT |
| Mixing timezone-aware and naive timestamps in cohorts | Cohort boundaries shift depending on timezone | Always convert to a consistent timezone first with `AT TIME ZONE` |
| Country + cohort concatenation with NULLs | `NULL || '-2024-01'` = NULL | `COALESCE(country, 'Unknown') || '-' || ...` |

---

## Cross-DB Syntax Cheat Sheet

| Operation | PostgreSQL | BigQuery | MySQL |
|-----------|-----------|----------|-------|
| Current date | `CURRENT_DATE` | `CURRENT_DATE()` | `CURDATE()` |
| Current timestamp | `NOW()` | `CURRENT_TIMESTAMP()` | `NOW()` |
| Add interval | `+ INTERVAL '7 days'` | `DATE_ADD(d, INTERVAL 7 DAY)` | `DATE_ADD(d, INTERVAL 7 DAY)` |
| Date diff (days) | `date1 - date2` | `DATE_DIFF(d1, d2, DAY)` | `DATEDIFF(d1, d2)` |
| Truncate to month | `DATE_TRUNC('month', ts)` | `DATE_TRUNC(ts, MONTH)` | `DATE_FORMAT(d, '%Y-%m-01')` |
| Extract year | `EXTRACT(YEAR FROM ts)` | `EXTRACT(YEAR FROM ts)` | `YEAR(d)` |
| Format date | `TO_CHAR(ts, 'YYYY-MM')` | `FORMAT_DATE('%Y-%m', d)` | `DATE_FORMAT(d, '%Y-%m')` |
| Case-insensitive LIKE | `ILIKE` | `LIKE` (always case-insensitive) | `LIKE` (case-insensitive by default) |
| String split (nth part) | `SPLIT_PART(s, ',', 2)` | `SPLIT(s, ',')[OFFSET(1)]` | `SUBSTRING_INDEX(s, ',', 2)` |
| Concat strings | `\|\|` or `CONCAT()` | `CONCAT()` or `\|\|` | `CONCAT()` |
| Regex match | `col ~ 'pattern'` | `REGEXP_CONTAINS(col, pattern)` | `col REGEXP 'pattern'` |

---

## Interview Quick Patterns

| Scenario | Pattern |
|----------|---------|
| Monthly active users | `DATE_TRUNC('month', event_ts)` + `COUNT(DISTINCT user_id)` |
| Rolling 7-day window | `WHERE event_ts >= CURRENT_DATE - INTERVAL '7 days'` |
| Training duration in minutes | `EXTRACT(EPOCH FROM (finished_at - started_at)) / 60` |
| Cohort label for grouping | `DATE_TRUNC('month', signup_ts)` — never `TO_CHAR` |
| Parse log fields | `SPLIT_PART(log, ':', n)` or `SUBSTRING` + `POSITION` |
| Normalise dirty strings before JOIN | `LOWER(TRIM(col))` on both sides |
| Safe cast from text | `CASE WHEN col ~ '^\d+\.?\d*$' THEN col::FLOAT ELSE NULL END` |
| Explode comma-separated tags | `UNNEST(STRING_TO_ARRAY(tags, ','))` |
| Collapse rows to comma list | `STRING_AGG(col, ', ' ORDER BY col)` |
| Find records in last N days | `WHERE ts >= NOW() - INTERVAL 'N days'` |

---

```sql
-- ============================================
-- DATE DIFF
-- ============================================
-- PostgreSQL
SELECT order_date - ship_date AS day_diff;                     -- integer days
SELECT AGE(order_date, ship_date);                              -- interval (y/m/d)
SELECT EXTRACT(EPOCH FROM (ts2 - ts1))/3600 AS hour_diff;        -- hours between timestamps

-- MySQL
SELECT DATEDIFF(order_date, ship_date) AS day_diff;              -- days only
SELECT TIMESTAMPDIFF(HOUR, ts1, ts2) AS hour_diff;               -- any unit: SECOND/MINUTE/HOUR/DAY/MONTH/YEAR

-- SQL Server
SELECT DATEDIFF(DAY, ship_date, order_date) AS day_diff;
SELECT DATEDIFF(HOUR, ts1, ts2) AS hour_diff;

-- Snowflake / BigQuery
SELECT DATEDIFF(day, ship_date, order_date) AS day_diff;         -- Snowflake
SELECT DATE_DIFF(order_date, ship_date, DAY) AS day_diff;        -- BigQuery (args reversed!)

-- ============================================
-- DATE_TRUNC — round down to unit
-- ============================================
-- PostgreSQL / Redshift / Snowflake
SELECT DATE_TRUNC('day', created_at);      -- 2026-07-09 00:00:00
SELECT DATE_TRUNC('month', created_at);    -- 2026-07-01 00:00:00
SELECT DATE_TRUNC('year', created_at);     -- 2026-01-01 00:00:00
SELECT DATE_TRUNC('week', created_at);     -- start of week (Mon, Postgres default)
SELECT DATE_TRUNC('hour', created_at);     -- zeroes out minutes/seconds

-- BigQuery
SELECT DATE_TRUNC(created_at, MONTH);      -- unit unquoted, arg order flipped

-- MySQL (no native DATE_TRUNC, emulate manually)
SELECT DATE_FORMAT(created_at, '%Y-%m-01') AS month_trunc;
SELECT DATE(created_at) AS day_trunc;

-- SQL Server (no native DATE_TRUNC pre-2022, emulate manually)
SELECT DATEFROMPARTS(YEAR(created_at), MONTH(created_at), 1) AS month_trunc;
SELECT DATE_TRUNC(MONTH, created_at);      -- SQL Server 2022+ only

-- ============================================
-- EXTRACT — pull out a date part
-- ============================================
-- PostgreSQL / MySQL / Snowflake (ANSI standard, widely supported)
SELECT EXTRACT(YEAR FROM created_at);
SELECT EXTRACT(MONTH FROM created_at);
SELECT EXTRACT(DAY FROM created_at);
SELECT EXTRACT(DOW FROM created_at);        -- day of week (0=Sun, Postgres)
SELECT EXTRACT(DOY FROM created_at);        -- day of year
SELECT EXTRACT(QUARTER FROM created_at);
SELECT EXTRACT(HOUR FROM created_at);
SELECT EXTRACT(EPOCH FROM created_at);      -- unix timestamp (Postgres)

-- SQL Server (no EXTRACT pre-2022, use DATEPART)
SELECT DATEPART(YEAR, created_at);
SELECT DATEPART(MONTH, created_at);
SELECT DATEPART(WEEKDAY, created_at);

-- BigQuery
SELECT EXTRACT(YEAR FROM created_at);
SELECT EXTRACT(DAYOFWEEK FROM created_at);  -- 1=Sunday

-- ============================================
-- QUICK REFERENCE TABLE (as comments)
-- ============================================
-- | Task              | Postgres            | MySQL                    | SQL Server            | BigQuery                  |
-- |--------------------|----------------------|--------------------------|------------------------|----------------------------|
-- | Diff in days       | date1 - date2        | DATEDIFF(date1,date2)    | DATEDIFF(DAY,d2,d1)    | DATE_DIFF(d1,d2,DAY)       |
-- | Diff in hours      | EXTRACT(EPOCH..)/3600| TIMESTAMPDIFF(HOUR,..)   | DATEDIFF(HOUR,..)      | TIMESTAMP_DIFF(t1,t2,HOUR)|
-- | Truncate to month  | DATE_TRUNC('month',x)| DATE_FORMAT(x,'%Y-%m-01')| DATEFROMPARTS(...)     | DATE_TRUNC(x, MONTH)       |
-- | Extract year       | EXTRACT(YEAR FROM x) | YEAR(x)                 | DATEPART(YEAR,x)       | EXTRACT(YEAR FROM x)       |
-- | Extract day-of-week| EXTRACT(DOW FROM x)  | DAYOFWEEK(x)             | DATEPART(WEEKDAY,x)    | EXTRACT(DAYOFWEEK FROM x)  |

-- ============================================
-- COMMON PATTERN: Monthly bucketed aggregation
-- ============================================
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(*) AS orders,
    SUM(amount) AS revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- ============================================
-- COMMON PATTERN: Age in days since signup
-- ============================================
SELECT
    user_id,
    CURRENT_DATE - signup_date AS days_since_signup   -- Postgres
FROM users;
```
```sql
-- ================================================================
-- INTERVAL FUNCTIONS — adding/subtracting time, generating ranges
-- ================================================================

-- --- Basic interval arithmetic ---
-- PostgreSQL
SELECT NOW() + INTERVAL '1 day';
SELECT NOW() - INTERVAL '7 days';
SELECT NOW() + INTERVAL '3 months';
SELECT NOW() + INTERVAL '1 year 2 months 3 days';
SELECT order_date + INTERVAL '30 days' AS due_date FROM orders;

-- MySQL
SELECT NOW() + INTERVAL 1 DAY;
SELECT NOW() - INTERVAL 7 DAY;
SELECT DATE_ADD(NOW(), INTERVAL 3 MONTH);
SELECT DATE_SUB(NOW(), INTERVAL 1 YEAR);

-- SQL Server (no INTERVAL keyword, use DATEADD)
SELECT DATEADD(DAY, 1, GETDATE());
SELECT DATEADD(MONTH, -3, GETDATE());
SELECT DATEADD(YEAR, 1, GETDATE());

-- Snowflake
SELECT DATEADD(DAY, 1, CURRENT_TIMESTAMP());
SELECT DATEADD('month', 3, CURRENT_TIMESTAMP());

-- BigQuery
SELECT TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY);
SELECT DATE_ADD(CURRENT_DATE(), INTERVAL 3 MONTH);


-- --- Interval between two timestamps (as a duration, not a number) ---
-- PostgreSQL: subtracting timestamps gives an INTERVAL type
SELECT ship_date - order_date AS duration;             -- interval e.g. '3 days 04:00:00'
SELECT AGE(ship_date, order_date);                      -- human-friendly interval
SELECT justify_interval(ship_date - order_date);        -- normalizes days/months/years

-- Extract parts out of an interval
SELECT EXTRACT(DAY FROM (ship_date - order_date)) AS days_part
FROM orders;
SELECT EXTRACT(EPOCH FROM (ship_date - order_date)) AS total_seconds
FROM orders;


-- --- Common patterns: filtering by relative time windows ---
-- Last 7 days
SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '7 days';       -- Postgres
SELECT * FROM orders WHERE order_date >= NOW() - INTERVAL 7 DAY;                -- MySQL
SELECT * FROM orders WHERE order_date >= DATEADD(DAY, -7, GETDATE());           -- SQL Server
SELECT * FROM orders WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY); -- BigQuery

-- Last full calendar month
SELECT * FROM orders
WHERE order_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
  AND order_date <  DATE_TRUNC('month', CURRENT_DATE);                          -- Postgres

-- Rolling 30-day window per row (correlated comparison)
SELECT a.customer_id, a.order_date,
       COUNT(b.order_id) AS orders_in_prior_30_days
FROM orders a
JOIN orders b
  ON b.customer_id = a.customer_id
 AND b.order_date BETWEEN a.order_date - INTERVAL '30 days' AND a.order_date
GROUP BY a.customer_id, a.order_date;


-- --- Generating a series of dates using intervals ---
-- PostgreSQL: generate_series with interval step
SELECT generate_series(
    '2026-01-01'::date,
    '2026-01-31'::date,
    INTERVAL '1 day'
) AS day;

-- Weekly buckets
SELECT generate_series(
    '2026-01-01'::date,
    '2026-12-31'::date,
    INTERVAL '1 week'
) AS week_start;

-- BigQuery: GENERATE_DATE_ARRAY with interval-like step
SELECT day
FROM UNNEST(GENERATE_DATE_ARRAY('2026-01-01', '2026-01-31', INTERVAL 1 DAY)) AS day;

-- Snowflake: recursive CTE (no native generate_series for dates pre-2023 versions)
WITH RECURSIVE dates AS (
    SELECT '2026-01-01'::date AS d
    UNION ALL
    SELECT DATEADD(day, 1, d) FROM dates WHERE d < '2026-01-31'
)
SELECT * FROM dates;


-- --- Age / tenure calculations using intervals ---
SELECT
    user_id,
    signup_date,
    AGE(CURRENT_DATE, signup_date) AS tenure                -- Postgres: e.g. '2 years 3 mons 10 days'
FROM users;

SELECT
    user_id,
    TIMESTAMPDIFF(YEAR, signup_date, CURDATE()) AS tenure_years   -- MySQL
FROM users;

SELECT
    user_id,
    DATEDIFF(YEAR, signup_date, GETDATE()) AS tenure_years        -- SQL Server
FROM users;


-- --- Interval comparisons / boolean checks ---
SELECT * FROM subscriptions
WHERE (CURRENT_DATE - start_date) > INTERVAL '1 year';           -- Postgres

SELECT * FROM subscriptions
WHERE DATEDIFF(CURDATE(), start_date) > 365;                     -- MySQL (no interval compare, use days)


-- --- Truncating an interval-based bucket for grouping ---
SELECT
    DATE_TRUNC('week', order_date) AS week_bucket,
    SUM(amount) AS weekly_revenue
FROM orders
GROUP BY DATE_TRUNC('week', order_date)
ORDER BY week_bucket;


-- --- Quick reference: interval keywords by dialect ---
-- | Task                  | Postgres              | MySQL                     | SQL Server           | BigQuery                    |
-- |------------------------|------------------------|---------------------------|------------------------|-------------------------------|
-- | Add N days             | date + INTERVAL 'N day'| DATE_ADD(date, INTERVAL N DAY) | DATEADD(DAY,N,date) | DATE_ADD(date, INTERVAL N DAY)|
-- | Subtract N months      | date - INTERVAL 'N mon'| DATE_SUB(date, INTERVAL N MONTH)| DATEADD(MONTH,-N,date)| DATE_SUB(date, INTERVAL N MONTH)|
-- | Diff as interval        | date2 - date1 (interval)| N/A (numeric only)        | N/A (numeric only)    | N/A (numeric only)             |
-- | Generate date series    | generate_series(...)   | recursive CTE / calendar table | recursive CTE     | GENERATE_DATE_ARRAY(...)       |
```

```

SELECT
  LOCATE('world', 'hello world'),   -- 7 (position of substring)
  LOCATE('@', 'john@gmail.com'),    -- 5
  INSTR('hello world', 'world');    -- 7 (same as LOCATE, different arg order)

sql-- Find users with gmail
SELECT email FROM users
WHERE LOCATE('@gmail.com', email) > 0;

-- Or simpler with LIKE
SELECT email FROM users WHERE email LIKE '%@gmail.com';

SELECT
  REPLACE('hello world', 'world', 'SQL'), -- 'hello SQL'
  REPLACE(phone, '-', ''),               -- remove dashes
  REPLACE(phone, ' ', '');               -- remove spaces

-- CONCAT
SELECT
  CONCAT(first_name, ' ', last_name) AS full_name,
  CONCAT('user_', user_id)           AS user_label
FROM users;

-- CONCAT_WS — skips NULLs safely
SELECT CONCAT_WS(', ', city, state, country) AS full_address
FROM users;
-- NULL state → 'New York, USA' not 'New York, , USA'


SELECT
  user_id,
  LOWER(TRIM(email))              AS clean_email,
  CONCAT('****', RIGHT(phone, 4)) AS masked_phone,
  UPPER(TRIM(country))            AS clean_country
FROM users;


```
