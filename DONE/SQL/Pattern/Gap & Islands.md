# SQL Islands & Gaps

## The problem, in plain English

You have rows with some kind of sequence — usually a date or an ID — and you want to find:
- **Islands**: consecutive/unbroken runs of values
- **Gaps**: the missing spots between runs

Classic example: a table of user login dates. You want to know "what are the streaks?" (islands) or "which days were missed?" (gaps).

```
date
2024-01-01
2024-01-02
2024-01-03
2024-01-05
2024-01-06
2024-01-10
```

Islands here: `[01-01 to 01-03]`, `[01-05 to 01-06]`, `[01-10]`
Gaps: `01-04` is missing, `01-07 to 01-09` is missing.

## The one mental model: rows marching in step

You have a sorted column (`col`) that's *supposed* to increase by exactly 1 each row if nothing's missing. Assign every row a counter that *also* increases by exactly 1 each row, no matter what (`ROW_NUMBER()`).

- **While the data is unbroken**, `col` and the counter climb in lockstep → their difference is frozen.
- **The instant something's missing**, `col` jumps ahead of the counter → the difference jumps too.

That single fact gives you both islands and gaps — they're just two different questions you ask about the same broken lockstep.

## Islands: "group the rows that stayed in lockstep"

```sql
SELECT
    col,
    ROW_NUMBER() OVER (ORDER BY col) AS rn,
    col - ROW_NUMBER() OVER (ORDER BY col) AS grp
FROM my_table;
```

Every row **inside the same island** gets the exact same `grp` value. That's the whole trick — everything else is just using `grp` as a `GROUP BY` key.

**Universal template:**

```sql
WITH numbered AS (
    SELECT
        col,
        ROW_NUMBER() OVER (ORDER BY col) AS rn
    FROM my_table
),
grouped AS (
    SELECT
        col,
        col - rn AS grp        -- this constant = island ID
    FROM numbered
)
SELECT
    MIN(col) AS island_start,
    MAX(col) AS island_end,
    COUNT(*) AS island_length
FROM grouped
GROUP BY grp
ORDER BY island_start;
```

## Gaps: "find the exact row where lockstep broke"

You don't need `grp` for this — just look one row ahead and check the jump size.

**Step 1 — get the next value in each row using `LEAD`:**

```sql
SELECT
    col AS current_val,
    LEAD(col) OVER (ORDER BY col) AS next_val
FROM my_table;
```

`LEAD(col)` means "grab the value from the next row down, and put it next to this row." With our dates:

| current_val | next_val |
|---|---|
| 01-01 | 01-02 |
| 01-02 | 01-03 |
| 01-03 | 01-05 |
| 01-05 | 01-06 |
| 01-06 | 01-10 |
| 01-10 | NULL |

**Step 2 — a gap exists wherever the jump is more than 1.** Row 3 (`01-03 → 01-05`) jumps 2 days. Row 5 (`01-06 → 01-10`) jumps 4 days. Those are the gaps.

**Step 3 — filter to just those rows.** A window function's result can't be used directly in `WHERE`, so wrap it in a subquery:

```sql
SELECT
    current_val + 1 AS gap_start,
    next_val - 1 AS gap_end
FROM (
    SELECT
        col AS current_val,
        LEAD(col) OVER (ORDER BY col) AS next_val
    FROM my_table
) t
WHERE next_val - current_val > 1;
```

The `+1` / `-1` shifts the boundary values inward so you report the actual missing range, not the two dates surrounding it:

| gap_start | gap_end |
|---|---|
| 01-04 | 01-04 |
| 01-07 | 01-09 |

**So: islands asks "which rows share a frozen difference" (needs `ROW_NUMBER` + `GROUP BY`). Gaps asks "where did the difference jump" (needs `LEAD` + `WHERE`).** Same underlying fact about consecutiveness, two different questions.

## Variants — same trick, different "consecutive" definition

The pattern only changes in **what "consecutive" means** for your data type.

| Variant | What changes | Trick |
|---|---|---|
| Integers | consecutive means `+1` | `col - ROW_NUMBER() OVER (ORDER BY col)` |
| Dates | consecutive means `+1 day` | same subtraction — date arithmetic gives you a number back (cast `rn` to interval in Postgres: `col - (rn || ' days')::interval`) |
| Per-category (e.g. islands per user) | counter should restart per group | `PARTITION BY user_id` inside `ROW_NUMBER()`, and `GROUP BY user_id, grp` after |
| Status runs (non-numeric — group repeated identical values regardless of value) | "consecutive" means *same value repeated*, not `+1` | two counters: `ROW_NUMBER() OVER (ORDER BY id) - ROW_NUMBER() OVER (PARTITION BY status ORDER BY id)`. The global counter always climbs by 1; the partitioned one only climbs when that status shows up, so it "falls behind" the moment status changes, breaking the difference |
| Custom gap threshold (session windows) | gap isn't exactly 1, it's e.g. "> 30 minutes" | `LAG` + a `CASE` flag for "new session," then `SUM(flag) OVER (ORDER BY ...)` as a running island ID — more verbose but handles thresholds `ROW_NUMBER` can't |

Every row in that table is still the same lockstep idea — only the definition of "in step" changes.

## Worked problems

### Problem 1 — Plain integers (find islands)

**Input `nums`:**

| n |
|---|
| 1 |
| 2 |
| 3 |
| 7 |
| 8 |
| 10 |

```sql
SELECT MIN(n) AS start, MAX(n) AS end_, COUNT(*) AS len
FROM (
    SELECT n, n - ROW_NUMBER() OVER (ORDER BY n) AS grp
    FROM nums
) t
GROUP BY grp
ORDER BY start;
```

**Output:**

| start | end_ | len |
|---|---|---|
| 1 | 3 | 3 |
| 7 | 8 | 2 |
| 10 | 10 | 1 |

### Problem 2 — Dates, per user (login streaks)

**Input `logins`:**

| user_id | login_date |
|---|---|
| A | 2024-01-01 |
| A | 2024-01-02 |
| A | 2024-01-03 |
| A | 2024-01-05 |
| B | 2024-01-01 |
| B | 2024-01-02 |

```sql
SELECT user_id, MIN(login_date) AS streak_start, MAX(login_date) AS streak_end
FROM (
    SELECT
        user_id,
        login_date,
        login_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_date) AS grp
    FROM logins
) t
GROUP BY user_id, grp
ORDER BY user_id, streak_start;
```

`PARTITION BY user_id` restarts the counter at 1 for every user, so each user's `grp` values are only ever comparable within that user. The outer query must `GROUP BY user_id, grp` — grouping by `grp` alone would wrongly merge different users who happen to land on the same constant.

**Output:**

| user_id | streak_start | streak_end |
|---|---|---|
| A | 01-01 | 01-03 |
| A | 01-05 | 01-05 |
| B | 01-01 | 01-02 |

### Problem 3 — Gaps (missing dates)

**Input `logins2`:**

| d |
|---|
| 01-01 |
| 01-02 |
| 01-03 |
| 01-07 |
| 01-08 |

```sql
SELECT d + 1 AS gap_start, next_d - 1 AS gap_end
FROM (
    SELECT d, LEAD(d) OVER (ORDER BY d) AS next_d
    FROM logins2
) t
WHERE next_d - d > 1;
```

**Output:**

| gap_start | gap_end |
|---|---|
| 01-04 | 01-06 |

### Problem 4 — Repeated identical values (status runs)

**Input `logs`:**

| id | status |
|---|---|
| 1 | active |
| 2 | active |
| 3 | inactive |
| 4 | inactive |
| 5 | inactive |
| 6 | active |

```sql
SELECT status, MIN(id) AS run_start, MAX(id) AS run_end
FROM (
    SELECT
        id,
        status,
        ROW_NUMBER() OVER (ORDER BY id)
        - ROW_NUMBER() OVER (PARTITION BY status ORDER BY id) AS grp
    FROM logs
) t
GROUP BY status, grp
ORDER BY run_start;
```

`GROUP BY status, grp` (not just `status`) matters here: the two separate "active" runs (rows 1–2 and row 6) get different `grp` values and must stay separate runs.

**Output:**

| status | run_start | run_end |
|---|---|---|
| active | 1 | 2 |
| inactive | 3 | 5 |
| active | 6 | 6 |

### Problem 5 — Custom gap threshold (session windows)

**Input `clicks`:**

| click_time |
|---|
| 10:00 |
| 10:10 |
| 10:15 |
| 10:50 |
| 10:55 |

Rule: a new session starts if the gap since the last click is **> 30 minutes**.

```sql
SELECT
    click_time,
    SUM(is_new_session) OVER (ORDER BY click_time) AS session_id
FROM (
    SELECT
        click_time,
        CASE
            WHEN click_time - LAG(click_time) OVER (ORDER BY click_time) > INTERVAL '30 minutes'
            THEN 1 ELSE 0
        END AS is_new_session
    FROM clicks
) t;
```

`LAG` grabs the *previous* row's timestamp (opposite direction of `LEAD`). The first row's `LAG` is NULL, so the `CASE` correctly leaves it flagged `0` instead of erroring. `SUM(is_new_session) OVER (ORDER BY click_time)` is a running total of that flag — every `1` permanently steps the total up, so all rows before the next `1` share the same session ID.

**Output:**

| click_time | session_id |
|---|---|
| 10:00 | 0 |
| 10:10 | 0 |
| 10:15 | 0 |
| 10:50 | 1 |
| 10:55 | 1 |

## Practice order

1. Integer islands/gaps on a plain number list.
2. Date islands/gaps (login streaks).
3. Add `PARTITION BY` (per-user streaks).
4. Same-value run grouping (status changes) — the double-`ROW_NUMBER` trick.
5. Session windowing with a custom gap threshold (`LAG` + `SUM(flag)` style) — this is what "sessionize user clickstream, 30-min timeout" interview questions want.
