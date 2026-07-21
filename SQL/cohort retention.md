# Cohort Retention Analysis — End to End

---

## What Is Cohort Retention?

Cohort retention measures what percentage of users who started in a given period (the cohort) came back and performed an action in later periods. It answers: **"Of the users who joined in Month X, how many were still active in Month X+1, X+2, X+3…?"**

---

## The Sample Data

We'll use a simple `events` table throughout. Every example builds on this.

```sql
-- Raw events table
SELECT * FROM events LIMIT 10;
```

| user_id | event_date | event_type |
|---------|------------|------------|
| 101     | 2024-01-05 | login      |
| 102     | 2024-01-12 | login      |
| 101     | 2024-02-03 | login      |
| 103     | 2024-02-18 | login      |
| 101     | 2024-03-07 | login      |
| 102     | 2024-03-22 | login      |
| 104     | 2024-01-09 | login      |
| 103     | 2024-03-14 | login      |
| 104     | 2024-02-27 | login      |
| 102     | 2024-04-11 | login      |

---

## Stage 1 — Find Each User's First Activity Date (Cohort Assignment)

Every user belongs to exactly one cohort: the month they first appeared.

**Variables:**
- `cohort_month` — The truncated month of the user's first-ever event
- `MIN(event_date)` — Finds the earliest event per user

```sql
-- Stage 1: Assign each user to their cohort month
SELECT
    user_id,
    DATE_TRUNC('month', MIN(event_date)) AS cohort_month   -- first month seen
FROM events
GROUP BY user_id;
```

**Output:**

| user_id | cohort_month |
|---------|--------------|
| 101     | 2024-01-01   |
| 102     | 2024-01-01   |
| 103     | 2024-02-01   |
| 104     | 2024-01-01   |

> Users 101, 102, 104 → January cohort. User 103 → February cohort.

---

## Stage 2 — Find All Activity Months Per User

Now we find every month each user was active (not just their first).

**Variables:**
- `activity_month` — Each distinct month a user had any event

```sql
-- Stage 2: All months each user was active
SELECT
    user_id,
    DATE_TRUNC('month', event_date) AS activity_month    -- each month they acted
FROM events
GROUP BY user_id, DATE_TRUNC('month', event_date);
```

**Output:**

| user_id | activity_month |
|---------|----------------|
| 101     | 2024-01-01     |
| 101     | 2024-02-01     |
| 101     | 2024-03-01     |
| 102     | 2024-01-01     |
| 102     | 2024-03-01     |
| 102     | 2024-04-01     |
| 103     | 2024-02-01     |
| 103     | 2024-03-01     |
| 104     | 2024-01-01     |
| 104     | 2024-02-01     |

> Notice user 102 skipped February — they're inactive that month.

---

## Stage 3 — Join Cohort Month to Activity Month

We combine Stages 1 and 2 so each activity row also knows the user's cohort.

**Variables:**
- `c.cohort_month` — When the user first appeared
- `a.activity_month` — When they were active in this row

```sql
-- Stage 3: Combine cohort assignment with activity
WITH cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', MIN(event_date)) AS cohort_month
    FROM events
    GROUP BY user_id
),
activity AS (
    SELECT
        user_id,
        DATE_TRUNC('month', event_date) AS activity_month
    FROM events
    GROUP BY user_id, DATE_TRUNC('month', event_date)
)
SELECT
    a.user_id,
    c.cohort_month,
    a.activity_month
FROM activity a
JOIN cohorts c ON a.user_id = c.user_id;
```

**Output:**

| user_id | cohort_month | activity_month |
|---------|--------------|----------------|
| 101     | 2024-01-01   | 2024-01-01     |
| 101     | 2024-01-01   | 2024-02-01     |
| 101     | 2024-01-01   | 2024-03-01     |
| 102     | 2024-01-01   | 2024-01-01     |
| 102     | 2024-01-01   | 2024-03-01     |
| 102     | 2024-01-01   | 2024-04-01     |
| 103     | 2024-02-01   | 2024-02-01     |
| 103     | 2024-02-01   | 2024-03-01     |
| 104     | 2024-01-01   | 2024-01-01     |
| 104     | 2024-01-01   | 2024-02-01     |

---

## Stage 4 — Calculate the Period Number (Month Index)

Instead of calendar months, we want relative periods: Month 0 (signup month), Month 1 (one month later), Month 2, etc.

**Variables:**
- `period_number` — How many months after cohort_month the user was active
- `DATEDIFF` / `EXTRACT` — Computes the gap between cohort month and activity month

```sql
-- Stage 4: Compute relative period (0 = signup month, 1 = first month after, etc.)

-- PostgreSQL / BigQuery syntax:
SELECT
    user_id,
    cohort_month,
    activity_month,
    EXTRACT(YEAR FROM AGE(activity_month, cohort_month)) * 12
    + EXTRACT(MONTH FROM AGE(activity_month, cohort_month)) AS period_number
FROM (... Stage 3 query ...);

-- Snowflake / Redshift syntax:
SELECT
    user_id,
    cohort_month,
    activity_month,
    DATEDIFF('month', cohort_month, activity_month) AS period_number
FROM (... Stage 3 query ...);
```

**Output:**

| user_id | cohort_month | activity_month | period_number |
|---------|--------------|----------------|---------------|
| 101     | 2024-01-01   | 2024-01-01     | 0             |
| 101     | 2024-01-01   | 2024-02-01     | 1             |
| 101     | 2024-01-01   | 2024-03-01     | 2             |
| 102     | 2024-01-01   | 2024-01-01     | 0             |
| 102     | 2024-01-01   | 2024-03-01     | 2             |
| 102     | 2024-01-01   | 2024-04-01     | 3             |
| 103     | 2024-02-01   | 2024-02-01     | 0             |
| 103     | 2024-02-01   | 2024-03-01     | 1             |
| 104     | 2024-01-01   | 2024-01-01     | 0             |
| 104     | 2024-01-01   | 2024-02-01     | 1             |

> Period 0 = the signup month. Period 2 = two months later. User 102 has no Period 1 (skipped February).

---

## Stage 5 — Count Cohort Size and Active Users Per Period

Now we aggregate: how many users started in each cohort, and how many came back in each period?

**Variables:**
- `cohort_size` — Total users who belong to the cohort (COUNT at Period 0)
- `active_users` — Users active in that cohort + period combination

```sql
-- Stage 5: Count users per cohort per period
WITH cohort_data AS (
    -- ... Stages 1–4 combined ...
    SELECT
        user_id,
        cohort_month,
        DATEDIFF('month', cohort_month, activity_month) AS period_number
    FROM activity a
    JOIN cohorts c ON a.user_id = c.user_id
)
SELECT
    cohort_month,
    period_number,
    COUNT(DISTINCT user_id) AS active_users        -- users active this period
FROM cohort_data
GROUP BY cohort_month, period_number
ORDER BY cohort_month, period_number;
```

**Output:**

| cohort_month | period_number | active_users |
|--------------|---------------|--------------|
| 2024-01-01   | 0             | 3            |
| 2024-01-01   | 1             | 2            |
| 2024-01-01   | 2             | 2            |
| 2024-01-01   | 3             | 1            |
| 2024-02-01   | 0             | 1            |
| 2024-02-01   | 1             | 1            |

---

## Stage 6 — Compute Cohort Size (Denominator)

We need the cohort size (Period 0 count) as a denominator for every row in that cohort.

**Variables:**
- `cohort_size` — Pulled from Period 0 using a window function or self-join
- `FIRST_VALUE` / subquery — Fetches the Period 0 count for each cohort

```sql
-- Stage 6: Attach cohort size to every row
WITH period_counts AS (
    -- Stage 5 result
    SELECT
        cohort_month,
        period_number,
        COUNT(DISTINCT user_id) AS active_users
    FROM cohort_data
    GROUP BY cohort_month, period_number
)
SELECT
    cohort_month,
    period_number,
    active_users,
    FIRST_VALUE(active_users) OVER (
        PARTITION BY cohort_month          -- reset per cohort
        ORDER BY period_number             -- Period 0 comes first
    ) AS cohort_size                       -- always the Period 0 count
FROM period_counts;
```

**Output:**

| cohort_month | period_number | active_users | cohort_size |
|--------------|---------------|--------------|-------------|
| 2024-01-01   | 0             | 3            | 3           |
| 2024-01-01   | 1             | 2            | 3           |
| 2024-01-01   | 2             | 2            | 3           |
| 2024-01-01   | 3             | 1            | 3           |
| 2024-02-01   | 0             | 1            | 1           |
| 2024-02-01   | 1             | 1            | 1           |

---

## Stage 7 — Calculate Retention Rate

Divide `active_users` by `cohort_size` to get the retention percentage.

**Variables:**
- `retention_rate` — `active_users / cohort_size * 100`
- `ROUND` — Formats the percentage

```sql
-- Stage 7: Final retention rate
SELECT
    cohort_month,
    period_number,
    active_users,
    cohort_size,
    ROUND(100.0 * active_users / cohort_size, 1) AS retention_rate   -- % retained
FROM (... Stage 6 query ...);
```

**Output:**

| cohort_month | period_number | active_users | cohort_size | retention_rate |
|--------------|---------------|--------------|-------------|----------------|
| 2024-01-01   | 0             | 3            | 3           | 100.0%         |
| 2024-01-01   | 1             | 2            | 3           | 66.7%          |
| 2024-01-01   | 2             | 2            | 3           | 66.7%          |
| 2024-01-01   | 3             | 1            | 3           | 33.3%          |
| 2024-02-01   | 0             | 1            | 1           | 100.0%         |
| 2024-02-01   | 1             | 1            | 1           | 100.0%         |

> Period 0 is always 100% (every user counts themselves). Period 1+ shows true retention.

---

## Stage 8 — Pivot Into a Retention Triangle

The final step reshapes the data into the classic retention grid — cohorts as rows, periods as columns.

**Variables:**
- `MAX(CASE WHEN ...)` — Conditional aggregation to pivot periods into columns
- Each column alias (`m0`, `m1`, `m2`…) = a period

```sql
-- Stage 8: Pivot to retention triangle
SELECT
    cohort_month,
    cohort_size,
    MAX(CASE WHEN period_number = 0 THEN retention_rate END) AS m0,
    MAX(CASE WHEN period_number = 1 THEN retention_rate END) AS m1,
    MAX(CASE WHEN period_number = 2 THEN retention_rate END) AS m2,
    MAX(CASE WHEN period_number = 3 THEN retention_rate END) AS m3
FROM (... Stage 7 query ...)
GROUP BY cohort_month, cohort_size
ORDER BY cohort_month;
```

**Output — The Retention Triangle:**

| cohort_month | cohort_size | m0     | m1    | m2    | m3    |
|--------------|-------------|--------|-------|-------|-------|
| 2024-01-01   | 3           | 100.0% | 66.7% | 66.7% | 33.3% |
| 2024-02-01   | 1           | 100.0% | 100.0%| —     | —     |

> Blank cells (—) = that period hasn't happened yet. This is the "triangle" shape.

---

## The Full Query (All Stages Combined)

```sql
WITH
-- Stage 1: Cohort assignment
cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', MIN(event_date)) AS cohort_month
    FROM events
    GROUP BY user_id
),

-- Stage 2: All activity months
activity AS (
    SELECT
        user_id,
        DATE_TRUNC('month', event_date) AS activity_month
    FROM events
    GROUP BY user_id, DATE_TRUNC('month', event_date)
),

-- Stages 3 & 4: Join + compute period number
cohort_activity AS (
    SELECT
        a.user_id,
        c.cohort_month,
        DATEDIFF('month', c.cohort_month, a.activity_month) AS period_number
    FROM activity a
    JOIN cohorts c ON a.user_id = c.user_id
),

-- Stage 5: Count active users per cohort per period
period_counts AS (
    SELECT
        cohort_month,
        period_number,
        COUNT(DISTINCT user_id) AS active_users
    FROM cohort_activity
    GROUP BY cohort_month, period_number
),

-- Stage 6: Attach cohort size
with_cohort_size AS (
    SELECT
        cohort_month,
        period_number,
        active_users,
        FIRST_VALUE(active_users) OVER (
            PARTITION BY cohort_month
            ORDER BY period_number
        ) AS cohort_size
    FROM period_counts
),

-- Stage 7: Retention rate
retention AS (
    SELECT
        cohort_month,
        period_number,
        active_users,
        cohort_size,
        ROUND(100.0 * active_users / cohort_size, 1) AS retention_rate
    FROM with_cohort_size
)

-- Stage 8: Pivot
SELECT
    cohort_month,
    cohort_size,
    MAX(CASE WHEN period_number = 0 THEN retention_rate END) AS m0,
    MAX(CASE WHEN period_number = 1 THEN retention_rate END) AS m1,
    MAX(CASE WHEN period_number = 2 THEN retention_rate END) AS m2,
    MAX(CASE WHEN period_number = 3 THEN retention_rate END) AS m3,
    MAX(CASE WHEN period_number = 4 THEN retention_rate END) AS m4,
    MAX(CASE WHEN period_number = 5 THEN retention_rate END) AS m5
FROM retention
GROUP BY cohort_month, cohort_size
ORDER BY cohort_month;
```

---

## Key Variables Summary

| Variable | What It Is | Where Used |
|----------|-----------|------------|
| `cohort_month` | Truncated month of user's first event | Stages 1, 3–8 |
| `activity_month` | Truncated month of any event | Stage 2 |
| `period_number` | Months since cohort_month (0, 1, 2…) | Stages 4–8 |
| `active_users` | COUNT DISTINCT users active in that cohort + period | Stages 5–8 |
| `cohort_size` | COUNT DISTINCT users at Period 0 (denominator) | Stages 6–8 |
| `retention_rate` | active_users / cohort_size × 100 | Stages 7–8 |

---

## How to Read the Triangle

```
           M0      M1      M2      M3
Jan cohort 100%   66.7%   66.7%   33.3%
Feb cohort 100%  100.0%    —       —
```

- **Read across a row** → how a single cohort decays over time
- **Read down a column** → compare the same period across different cohorts
- **The diagonal** → all users at the same calendar month regardless of cohort
- **Blank cells** → future periods that haven't occurred yet

---

## Common Variants

**Day-level retention** — replace `DATE_TRUNC('month', ...)` with `DATE_TRUNC('day', ...)` and `DATEDIFF('month', ...)` with `DATEDIFF('day', ...)`. Used for mobile apps where D1, D7, D30 matter most.

**Event-specific retention** — filter the `activity` CTE with `WHERE event_type = 'purchase'` to measure whether users came back to buy, not just log in.

**Unbounded retention** — instead of checking activity in month N exactly, use `period_number >= N` to count anyone still active *at least* N months later. Removes the gap/skip problem shown with user 102 above.

**Rolling 28-day active** — define `activity_month` as any event within a 28-day window rather than a calendar month, to smooth seasonality effects.
