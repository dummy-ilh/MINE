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

# Islands & Gaps — Interview Q&A (Google / Apple / Meta style) + Cheatsheet

These are the actual flavors these companies ask this pattern in. Same trick every time — only the "what counts as consecutive" and "what to report" changes.

---

## Q1 (Meta / Facebook — LeetCode 1225, "Report Contiguous Dates")

**Setup:** A `Failed` table (`fail_date`) and a `Succeeded` table (`success_date`) log server status by day. Every day has exactly one row in one of the two tables. Report the **periods** of consecutive `failed` or `succeeded` days, in order, with `period_state`, `start_date`, `end_date`.

**Sample input:**

`Failed`: 2019-01-01, 2019-01-02, 2019-01-03, 2019-01-17
`Succeeded`: 2019-01-04, 2019-01-05, 2019-01-06, 2019-01-10, 2019-01-11, 2019-01-12, 2019-01-13, 2019-01-14, 2019-01-15, 2019-01-16

**Expected output:**

| period_state | start_date | end_date |
|---|---|---|
| failed | 2019-01-01 | 2019-01-03 |
| succeeded | 2019-01-04 | 2019-01-06 |
| succeeded | 2019-01-10 | 2019-01-16 |
| failed | 2019-01-17 | 2019-01-17 |

**Solution:**

```sql
WITH all_days AS (
    SELECT fail_date AS d, 'failed' AS state FROM Failed
    UNION ALL
    SELECT success_date AS d, 'succeeded' AS state FROM Succeeded
),
grouped AS (
    SELECT
        d, state,
        d - ROW_NUMBER() OVER (PARTITION BY state ORDER BY d) AS grp
    FROM all_days
)
SELECT state AS period_state, MIN(d) AS start_date, MAX(d) AS end_date
FROM grouped
GROUP BY state, grp
ORDER BY start_date;
```

**Why:** This is the "status runs" variant, but simplified — since the two tables are already split by state, you just need `PARTITION BY state` inside `ROW_NUMBER()` and the standard `date - rn` island trick does the rest. `GROUP BY state, grp` is required for the same reason as always: two separate runs of the same state must not merge.

**Trap interviewers watch for:** forgetting `date` restricted to a `WHERE` range in the real problem (dates only within a filter window) — always re-check the exact date bounds asked for before running the island logic.

---

## Q2 (Amazon / Meta — "N consecutive active days", LeetCode 1454 "Active Users")

**Setup:** A `sessions` table (`user_id`, `session_date`). Find users active for **5 or more consecutive days**.

**Solution:**

```sql
WITH distinct_days AS (
    SELECT DISTINCT user_id, session_date
    FROM sessions
),
grouped AS (
    SELECT
        user_id, session_date,
        session_date - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY session_date) AS grp
    FROM distinct_days
),
streaks AS (
    SELECT user_id, grp, COUNT(*) AS streak_len, MIN(session_date) AS start_date, MAX(session_date) AS end_date
    FROM grouped
    GROUP BY user_id, grp
)
SELECT user_id, start_date, end_date, streak_len
FROM streaks
WHERE streak_len >= 5
ORDER BY user_id, start_date;
```

**Why `DISTINCT` first:** if a user has two sessions on the same day, `ROW_NUMBER()` still only increments once per row — duplicate same-day rows silently break the `date - rn` arithmetic. Dedup to one row per `(user, date)` before applying the trick. This is the single most common bug in a live interview attempt at this problem.

---

## Q3 (Google — gap-detection, "missing dates over N days")

**Setup:** A `sensor_readings` table (`sensor_id`, `reading_date`). Flag any sensor with a **gap of more than 3 days** between consecutive readings — this indicates the sensor went offline.

**Solution:**

```sql
WITH ordered AS (
    SELECT
        sensor_id, reading_date,
        LEAD(reading_date) OVER (PARTITION BY sensor_id ORDER BY reading_date) AS next_date
    FROM sensor_readings
)
SELECT
    sensor_id,
    reading_date AS offline_after,
    next_date AS back_online,
    next_date - reading_date AS gap_days
FROM ordered
WHERE next_date - reading_date > 3;
```

**Why:** Pure gap-detection — no `grp`, no `ROW_NUMBER` needed at all. `PARTITION BY sensor_id` inside `LEAD` keeps each sensor's timeline independent, exactly like `PARTITION BY user_id` does for islands.

**Follow-up interviewers ask:** "What if a sensor never reports again after its last reading — is that a gap?" Answer: no, `LEAD` returns `NULL` for the last row per partition, and `NULL - reading_date` is `NULL`, which fails the `> 3` filter — so trailing "gaps to now" are silently excluded. If the interviewer wants those flagged too, you need a separate `WHERE next_date IS NULL AND CURRENT_DATE - reading_date > 3` branch, unioned in.

---

## Q4 (Apple / Bloomberg — LeetCode 601, "Human Traffic of Stadium")

**Setup:** A `stadium` table (`id`, `visit_date`, `people`), with `id` consecutive integers in visit order. Find all records where **three or more consecutive rows** (by `id`) each have `people >= 100`.

**Sample input:**

| id | visit_date | people |
|---|---|---|
| 1 | 2017-01-01 | 10 |
| 2 | 2017-01-02 | 109 |
| 3 | 2017-01-03 | 150 |
| 4 | 2017-01-04 | 99 |
| 5 | 2017-01-05 | 145 |
| 6 | 2017-01-06 | 1455 |
| 7 | 2017-01-07 | 199 |
| 8 | 2017-01-08 | 188 |

**Expected output:** ids 5, 6, 7, 8 (the run of `people >= 100` from row 5 onward is length 4; row 2-3 is only length 2, so it's excluded).

**Solution:**

```sql
WITH filtered AS (
    SELECT id, visit_date, people
    FROM stadium
    WHERE people >= 100
),
grouped AS (
    SELECT
        id, visit_date, people,
        id - ROW_NUMBER() OVER (ORDER BY id) AS grp
    FROM filtered
)
SELECT s.id, s.visit_date, s.people
FROM grouped g
JOIN stadium s ON s.id = g.id
WHERE g.grp IN (
    SELECT grp FROM grouped GROUP BY grp HAVING COUNT(*) >= 3
)
ORDER BY s.id;
```

**Why this is the hardest of the four:** it's islands (`id - ROW_NUMBER()`) applied *after* a `WHERE` filter, then a second pass (`HAVING COUNT(*) >= 3`) to keep only the islands that are long enough. This two-stage "filter → island → filter by island size" shape is the giveaway that a question is a harder variant of the base pattern — recognize it as: filter rows first, then run the *exact same* `id - rn` trick on the filtered id column (it still works because `id` stays the original sequential id, not a re-numbered one).

**Trap:** the `grp` value here is only meaningful because `id` in the original table is itself already gapless/sequential. If `id` weren't guaranteed sequential, you'd need to `ROW_NUMBER()` the *unfiltered* table first to get a clean sequential column before filtering.

---

## Cheatsheet

**One idea underneath everything:** a sorted column and a same-order counter climb in lockstep while nothing's missing; the difference between them is frozen during a run and jumps the instant something breaks.

| Goal | Tool | Core line |
|---|---|---|
| Islands (basic) | `ROW_NUMBER` + `GROUP BY` | `col - ROW_NUMBER() OVER (ORDER BY col) AS grp` |
| Islands per group (per user, per sensor) | add `PARTITION BY` | `ROW_NUMBER() OVER (PARTITION BY key ORDER BY col)`; then `GROUP BY key, grp` |
| Gaps (basic) | `LEAD` + `WHERE` | `LEAD(col) OVER (ORDER BY col) - col > 1` |
| Gaps per group | `LEAD` + `PARTITION BY` | `LEAD(col) OVER (PARTITION BY key ORDER BY col)` |
| Same-value runs (status changes, not numeric) | double `ROW_NUMBER` | `ROW_NUMBER() OVER (ORDER BY id) - ROW_NUMBER() OVER (PARTITION BY status ORDER BY id)` |
| Custom/variable gap threshold (sessions) | `LAG` + flag + running `SUM` | `SUM(CASE WHEN gap > threshold THEN 1 ELSE 0 END) OVER (ORDER BY col)` as session id |
| Islands after filtering, kept only if long enough | filter → island trick → `HAVING COUNT(*) >= n` | `WHERE condition` first, then `id - ROW_NUMBER()`, then filter `grp`s by size |

**Checklist before you write a line of SQL in the interview:**
1. Is "consecutive" numeric/date (`+1`), or same-value-repeated, or a custom time threshold? → picks which row above you're in.
2. Does it need to be scoped per entity (user/sensor/host)? → add `PARTITION BY` to every window function, and to the final `GROUP BY`.
3. Duplicate rows for the same key on the same date/id? → `DISTINCT` or pre-aggregate before applying `ROW_NUMBER`, or the arithmetic silently breaks.
4. Are you filtering rows *before* finding islands (e.g. `people >= 100`)? → filter first, then apply the trick on the filtered set, then optionally re-filter by island size with `HAVING`.
5. Trailing/leading edge cases — first row's `LAG` is `NULL`, last row's `LEAD` is `NULL` → decide explicitly whether those count, don't let them silently vanish from your `WHERE`.
