# LAG / LEAD Masterclass: Period-over-Period Analysis

Every pattern below follows the same teaching method: **sample input → CTE by CTE → sample output at each stage**. Once you see the row-by-row transformation, the syntax stops being magic.

---

## 0. The Core Mental Model

`LAG` and `LEAD` are **offset window functions**. They don't aggregate rows into one — they let one row "reach across" to a neighboring row, based on an ordering you define.

```sql
LAG(column, offset, default)  OVER (PARTITION BY ... ORDER BY ...)
LEAD(column, offset, default) OVER (PARTITION BY ... ORDER BY ...)
```

| Piece | Meaning |
|---|---|
| `column` | value to pull from the neighboring row |
| `offset` | how many rows away (default `1`) |
| `default` | value to use when there's no such row (default `NULL`) |
| `PARTITION BY` | resets the "neighbor" search per group (customer, store, product…) |
| `ORDER BY` | defines what "previous" / "next" even means (usually a date) |

`LAG` looks **backward** (previous row). `LEAD` looks **forward** (next row). That's the entire idea — everything below is just this idea applied to different shapes of data.

---

## 1. Day-over-Day Delta

**Business question:** "How much did daily revenue change vs. yesterday?"

**Raw table `daily_sales`:**

| sale_date | revenue |
|---|---|
| 2026-07-01 | 1000 |
| 2026-07-02 | 1200 |
| 2026-07-03 | 900 |
| 2026-07-04 | 1500 |

```sql
WITH base AS (
    SELECT sale_date, revenue
    FROM daily_sales
),
with_prev AS (
    SELECT
        sale_date,
        revenue,
        LAG(revenue, 1) OVER (ORDER BY sale_date) AS prev_day_revenue
    FROM base
),
with_delta AS (
    SELECT
        sale_date,
        revenue,
        prev_day_revenue,
        revenue - prev_day_revenue AS dod_delta,
        ROUND(
            (revenue - prev_day_revenue) * 100.0 / NULLIF(prev_day_revenue, 0), 2
        ) AS dod_pct_change
    FROM with_prev
)
SELECT * FROM with_delta ORDER BY sale_date;
```

**`base` output** — unchanged passthrough (CTEs don't have to transform anything; this one just scopes the columns you care about):

| sale_date | revenue |
|---|---|
| 07-01 | 1000 |
| 07-02 | 1200 |
| 07-03 | 900 |
| 07-04 | 1500 |

**`with_prev` output** — each row now carries yesterday's value alongside it. Notice the first row has no "yesterday," so `LAG` returns `NULL` (no default was supplied):

| sale_date | revenue | prev_day_revenue |
|---|---|---|
| 07-01 | 1000 | NULL |
| 07-02 | 1200 | 1000 |
| 07-03 | 900 | 1200 |
| 07-04 | 1500 | 900 |

**`with_delta` output** — final answer, computed as plain arithmetic on the two columns now sitting side by side:

| sale_date | revenue | prev_day_revenue | dod_delta | dod_pct_change |
|---|---|---|---|---|
| 07-01 | 1000 | NULL | NULL | NULL |
| 07-02 | 1200 | 1000 | 200 | 20.0 |
| 07-03 | 900 | 1200 | -300 | -25.0 |
| 07-04 | 1500 | 900 | 600 | 66.67 |

**The key lesson:** `LAG` only *fetches* the neighbor value. The delta/percent math happens in a *separate, later* step. Splitting these into two CTEs (`with_prev`, `with_delta`) isn't strictly required — you could inline it all in one `SELECT` — but separating "fetch the neighbor" from "do math on it" makes debugging trivial: if a delta looks wrong, check `with_prev` first and see if the neighbor itself is wrong.

---

## 2. Week-over-Week (Aggregate First, Then LAG)

**Business question:** "How does this week's revenue compare to last week's?"

This pattern is different from #1 because you must **collapse to weekly grain before** you can lag by week. Lagging daily rows by 7 rows is fragile — it breaks the moment a day is missing.

**Raw table:** same `daily_sales` as above, but imagine a full month of rows.

```sql
WITH weekly AS (
    SELECT
        DATE_TRUNC('week', sale_date) AS week_start,
        SUM(revenue) AS weekly_revenue
    FROM daily_sales
    GROUP BY DATE_TRUNC('week', sale_date)
),
with_prev_week AS (
    SELECT
        week_start,
        weekly_revenue,
        LAG(weekly_revenue, 1) OVER (ORDER BY week_start) AS prev_week_revenue
    FROM weekly
)
SELECT
    week_start,
    weekly_revenue,
    prev_week_revenue,
    weekly_revenue - prev_week_revenue AS wow_delta,
    ROUND((weekly_revenue - prev_week_revenue) * 100.0
          / NULLIF(prev_week_revenue, 0), 2) AS wow_pct_change
FROM with_prev_week
ORDER BY week_start;
```

**`weekly` output** (this CTE did real aggregation — grain changed from daily to weekly):

| week_start | weekly_revenue |
|---|---|
| 2026-06-29 | 6200 |
| 2026-07-06 | 7100 |
| 2026-07-13 | 6800 |

**`with_prev_week` output** — `LAG` now operates on *one row per week*, so offset `1` correctly means "the previous week," not "the previous day":

| week_start | weekly_revenue | prev_week_revenue |
|---|---|---|
| 06-29 | 6200 | NULL |
| 07-06 | 7100 | 6200 |
| 07-13 | 6800 | 7100 |

**Final output:**

| week_start | weekly_revenue | prev_week_revenue | wow_delta | wow_pct_change |
|---|---|---|---|---|
| 06-29 | 6200 | NULL | NULL | NULL |
| 07-06 | 7100 | 6200 | 900 | 14.52 |
| 07-13 | 6800 | 7100 | -300 | -4.23 |

**Rule of thumb:** whenever the period you're comparing (week, month, quarter) doesn't match the natural grain of the raw rows (day, transaction), **aggregate in one CTE, then lag in the next.** Never try to lag by a row-count offset (`LAG(x, 7)`) as a proxy for "7 days ago" unless you're certain there are zero gaps in the dates — one missing day silently shifts everything.

---

## 3. Month-over-Month and Year-over-Year — Two Different Techniques

### 3a. Month-over-Month (adjacent row, same idea as week-over-week)

```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', sale_date) AS month_start,
           SUM(revenue) AS monthly_revenue
    FROM daily_sales
    GROUP BY 1
)
SELECT
    month_start,
    monthly_revenue,
    LAG(monthly_revenue) OVER (ORDER BY month_start) AS prev_month_revenue,
    monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month_start) AS mom_delta
FROM monthly
ORDER BY month_start;
```
Same pattern as week-over-week — just a different `DATE_TRUNC` grain. Nothing new conceptually.

### 3b. Year-over-Year (offset ≠ 1)

**The trap:** YoY is *not* `LAG(x, 12)` unless your rows are guaranteed one-per-month with zero gaps for every month in history. A cleaner, gap-proof approach: partition by month-number, order by year, and lag by 1 within that partition.

**Raw `monthly` output (multi-year):**

| month_start | monthly_revenue |
|---|---|
| 2025-07-01 | 40000 |
| 2025-08-01 | 42000 |
| 2026-07-01 | 47000 |
| 2026-08-01 | 44000 |

```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', sale_date) AS month_start,
           SUM(revenue) AS monthly_revenue
    FROM daily_sales
    GROUP BY 1
),
tagged AS (
    SELECT
        month_start,
        monthly_revenue,
        EXTRACT(MONTH FROM month_start) AS month_num,
        EXTRACT(YEAR FROM month_start)  AS year_num
    FROM monthly
),
with_yoy AS (
    SELECT
        month_start,
        monthly_revenue,
        LAG(monthly_revenue, 1) OVER (
            PARTITION BY month_num       -- "same calendar month" bucket
            ORDER BY year_num            -- walk forward one year at a time
        ) AS same_month_last_year
    FROM tagged
)
SELECT
    month_start,
    monthly_revenue,
    same_month_last_year,
    ROUND((monthly_revenue - same_month_last_year) * 100.0
          / NULLIF(same_month_last_year, 0), 2) AS yoy_pct_change
FROM with_yoy
ORDER BY month_start;
```

**`tagged` output** — this CTE's whole job is to create the two columns the next CTE's window function needs (`month_num` for partitioning, `year_num` for ordering):

| month_start | monthly_revenue | month_num | year_num |
|---|---|---|---|
| 2025-07-01 | 40000 | 7 | 2025 |
| 2025-08-01 | 42000 | 8 | 2025 |
| 2026-07-01 | 47000 | 7 | 2026 |
| 2026-08-01 | 44000 | 8 | 2026 |

**`with_yoy` output** — because `PARTITION BY month_num` groups all "Julys" together and all "Augusts" together, `LAG(...,1)` inside each partition correctly means "same month, previous year," *regardless of how many other months sit between them in the raw table*:

| month_start | monthly_revenue | same_month_last_year |
|---|---|---|
| 2025-07-01 | 40000 | NULL |
| 2025-08-01 | 42000 | NULL |
| 2026-07-01 | 47000 | 40000 |
| 2026-08-01 | 44000 | 42000 |

**Final output:**

| month_start | monthly_revenue | same_month_last_year | yoy_pct_change |
|---|---|---|---|
| 2025-07-01 | 40000 | NULL | NULL |
| 2025-08-01 | 42000 | NULL | NULL |
| 2026-07-01 | 47000 | 40000 | 17.5 |
| 2026-08-01 | 44000 | 42000 | 4.76 |

**The lesson:** this is the single most important trick in the whole masterclass — **`PARTITION BY` doesn't have to be a "real" grouping column like customer_id. It can be a derived bucket (month number, day-of-week, product category) that redefines what "adjacent" means for the offset function.**

---

## 4. LEAD — Looking Forward

`LEAD` is `LAG`'s mirror. Same syntax, opposite direction. Use it when the question is phrased forward-looking: "what happens next," rather than backward-looking: "what changed since."

**Business question:** "For each customer order, how many days until their *next* order?" (a classic churn/repeat-purchase signal)

**Raw `orders`:**

| customer_id | order_date |
|---|---|
| C1 | 2026-01-05 |
| C1 | 2026-02-10 |
| C1 | 2026-06-01 |
| C2 | 2026-03-01 |
| C2 | 2026-03-20 |

```sql
WITH ordered AS (
    SELECT customer_id, order_date
    FROM orders
),
with_next AS (
    SELECT
        customer_id,
        order_date,
        LEAD(order_date, 1) OVER (
            PARTITION BY customer_id
            ORDER BY order_date
        ) AS next_order_date
    FROM ordered
),
with_gap AS (
    SELECT
        customer_id,
        order_date,
        next_order_date,
        next_order_date - order_date AS days_to_next_order
    FROM with_next
)
SELECT * FROM with_gap ORDER BY customer_id, order_date;
```

**`with_next` output** — `PARTITION BY customer_id` keeps C1's "next order" from ever leaking into C2's row. The *last* order per customer has no "next," so `LEAD` naturally returns `NULL` — this is often the exact signal you want ("their most recent order — did they ever come back?"):

| customer_id | order_date | next_order_date |
|---|---|---|
| C1 | 01-05 | 02-10 |
| C1 | 02-10 | 06-01 |
| C1 | 06-01 | NULL |
| C2 | 03-01 | 03-20 |
| C2 | 03-20 | NULL |

**`with_gap` final output:**

| customer_id | order_date | next_order_date | days_to_next_order |
|---|---|---|---|
| C1 | 01-05 | 02-10 | 36 |
| C1 | 02-10 | 06-01 | 111 |
| C1 | 06-01 | NULL | NULL |
| C2 | 03-01 | 03-20 | 19 |
| C2 | 03-20 | NULL | NULL |

A `NULL` in `days_to_next_order` on the *most recent* row per customer is expected and meaningful (no repeat purchase *yet*). A `NULL` anywhere else would indicate a data problem.

---

## 5. Multiple Offsets — LAG(x,1), LAG(x,2)... in One Query

You are not limited to one lag per query. Stack several offsets to compare a row against several points in its history simultaneously.

**Business question:** "Show this month, last month, and two months ago, side by side."

```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', sale_date) AS month_start,
           SUM(revenue) AS monthly_revenue
    FROM daily_sales
    GROUP BY 1
)
SELECT
    month_start,
    monthly_revenue                                   AS this_month,
    LAG(monthly_revenue, 1) OVER (ORDER BY month_start) AS month_minus_1,
    LAG(monthly_revenue, 2) OVER (ORDER BY month_start) AS month_minus_2,
    LAG(monthly_revenue, 3) OVER (ORDER BY month_start) AS month_minus_3
FROM monthly
ORDER BY month_start;
```

**Output** (each column is the *same window function*, just a different offset — no extra CTE needed since there's no intermediate math yet):

| month_start | this_month | month_minus_1 | month_minus_2 | month_minus_3 |
|---|---|---|---|---|
| 2026-04 | 38000 | NULL | NULL | NULL |
| 2026-05 | 41000 | 38000 | NULL | NULL |
| 2026-06 | 39500 | 41000 | 38000 | NULL |
| 2026-07 | 44000 | 39500 | 41000 | 38000 |

This is the backbone of "trailing N-period" tables in dashboards — every offset is an independent window call, evaluated against the same ordered partition.

---

## 6. LAG and LEAD Together — Detecting Local Peaks/Valleys

Combining both in one row lets you compare a point to *both* neighbors at once — useful for spike detection or "is this a local max?" logic.

**Business question:** "Flag any day where revenue was higher than both the day before *and* the day after (a local peak)."

```sql
WITH base AS (
    SELECT sale_date, revenue FROM daily_sales
),
with_neighbors AS (
    SELECT
        sale_date,
        revenue,
        LAG(revenue, 1)  OVER (ORDER BY sale_date) AS prev_revenue,
        LEAD(revenue, 1) OVER (ORDER BY sale_date) AS next_revenue
    FROM base
)
SELECT
    sale_date,
    revenue,
    prev_revenue,
    next_revenue,
    CASE
        WHEN revenue > prev_revenue AND revenue > next_revenue THEN TRUE
        ELSE FALSE
    END AS is_local_peak
FROM with_neighbors
ORDER BY sale_date;
```

**`with_neighbors` output** — one row now sees both directions at once:

| sale_date | revenue | prev_revenue | next_revenue |
|---|---|---|---|
| 07-01 | 1000 | NULL | 1200 |
| 07-02 | 1200 | 1000 | 900 |
| 07-03 | 900 | 1200 | 1500 |
| 07-04 | 1500 | 900 | NULL |

**Final output:**

| sale_date | revenue | is_local_peak |
|---|---|---|
| 07-01 | 1000 | FALSE |
| 07-02 | 1200 | TRUE |
| 07-03 | 900 | FALSE |
| 07-04 | 1500 | FALSE (no next value to confirm) |

Note the edge case: the last row can never be flagged `TRUE` here, because `next_revenue` is `NULL` and any comparison against `NULL` is `NULL`/false — this is usually correct behavior (you can't confirm a peak without a "day after" to compare to), but confirm that's actually what your business logic wants.

---

## 7. Handling Gaps — the `default` Argument and `IGNORE NULLS`

### 7a. Supplying a default instead of NULL

```sql
LAG(revenue, 1, 0) OVER (ORDER BY sale_date)   -- returns 0 instead of NULL on the first row
```
Use this when "no prior period" should mean "treat as zero" (e.g., a brand-new product with no prior month of sales) rather than "unknown," which would otherwise poison downstream math (`NULL - anything = NULL`).

### 7b. Skipping NULLs in the underlying data (dialect-dependent: Snowflake, BigQuery, Postgres 16+, Oracle)

**Raw data with missing days:**

| sale_date | revenue |
|---|---|
| 07-01 | 1000 |
| 07-02 | NULL |
| 07-03 | 900 |

Default `LAG` behavior treats the `NULL` row like any other row — it becomes the "previous value" for 07-03, which is wrong if `NULL` means "no sale recorded" rather than "sale of nothing":

```sql
LAG(revenue) OVER (ORDER BY sale_date)          -- 07-03 sees NULL (from 07-02)
LAG(revenue) IGNORE NULLS OVER (ORDER BY sale_date)  -- 07-03 sees 1000 (skips 07-02 entirely)
```

| sale_date | revenue | lag_default | lag_ignore_nulls |
|---|---|---|---|
| 07-01 | 1000 | NULL | NULL |
| 07-02 | NULL | 1000 | 1000 |
| 07-03 | 900 | NULL | 1000 |

**Lesson:** `IGNORE NULLS` changes *which row counts as "1 back."* Without it, a `NULL` row still occupies a slot in the offset count; with it, `NULL` rows are transparent and the function reaches through them to the last *real* value. Not all engines support this clause (MySQL and older Postgres don't) — the portable workaround is a `COALESCE`-based "last non-null" pattern using `LAG` combined with a running `MAX` over a flag, but that's a topic for a gaps-and-islands deep dive.

---

## 8. Gaps and Islands with LAG — Detecting Session Breaks

**Business question:** "Group a user's page views into sessions, where a new session starts after 30+ minutes of inactivity."

**Raw `page_views`:**

| user_id | viewed_at |
|---|---|
| U1 | 10:00 |
| U1 | 10:05 |
| U1 | 10:50 |
| U1 | 10:55 |

```sql
WITH ordered AS (
    SELECT user_id, viewed_at
    FROM page_views
),
with_prev AS (
    SELECT
        user_id,
        viewed_at,
        LAG(viewed_at) OVER (PARTITION BY user_id ORDER BY viewed_at) AS prev_viewed_at
    FROM ordered
),
with_flag AS (
    SELECT
        user_id,
        viewed_at,
        prev_viewed_at,
        CASE
            WHEN prev_viewed_at IS NULL THEN 1
            WHEN viewed_at - prev_viewed_at > INTERVAL '30 minutes' THEN 1
            ELSE 0
        END AS is_new_session
    FROM with_prev
),
with_session_id AS (
    SELECT
        user_id,
        viewed_at,
        SUM(is_new_session) OVER (
            PARTITION BY user_id ORDER BY viewed_at
        ) AS session_id
    FROM with_flag
)
SELECT * FROM with_session_id ORDER BY user_id, viewed_at;
```

**`with_prev` output:**

| user_id | viewed_at | prev_viewed_at |
|---|---|---|
| U1 | 10:00 | NULL |
| U1 | 10:05 | 10:00 |
| U1 | 10:50 | 10:05 |
| U1 | 10:55 | 10:50 |

**`with_flag` output** — `LAG` supplies the raw gap; a plain `CASE` turns that gap into a binary "session boundary" signal:

| user_id | viewed_at | is_new_session |
|---|---|---|
| U1 | 10:00 | 1 |
| U1 | 10:05 | 0 |
| U1 | 10:50 | 1 |
| U1 | 10:55 | 0 |

**`with_session_id` final output** — a *running sum* of the boundary flags turns "is this a new session" into "which session number is this," a classic pairing of an offset function feeding a cumulative one:

| user_id | viewed_at | session_id |
|---|---|---|
| U1 | 10:00 | 1 |
| U1 | 10:05 | 1 |
| U1 | 10:50 | 2 |
| U1 | 10:55 | 2 |

This is the canonical **gaps-and-islands** pattern: `LAG` detects the gap → `CASE` turns it into a flag → running `SUM`/`COUNT` turns the flag into a group ID.

---

## 9. Running Growth Rate — Chaining LAG Across Many Rows at Once

**Business question:** "Show cumulative growth vs. the very first period" — different from period-over-period, this compares every row back to a fixed anchor rather than its immediate neighbor.

```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', sale_date) AS month_start,
           SUM(revenue) AS monthly_revenue
    FROM daily_sales
    GROUP BY 1
),
with_baseline AS (
    SELECT
        month_start,
        monthly_revenue,
        FIRST_VALUE(monthly_revenue) OVER (ORDER BY month_start) AS baseline_revenue
    FROM monthly
)
SELECT
    month_start,
    monthly_revenue,
    baseline_revenue,
    ROUND((monthly_revenue - baseline_revenue) * 100.0
          / NULLIF(baseline_revenue, 0), 2) AS pct_growth_since_start
FROM with_baseline
ORDER BY month_start;
```

Not `LAG`/`LEAD` at all — `FIRST_VALUE` — included deliberately so you know when to reach for a *different* offset function: **`LAG`/`LEAD` for "the neighbor"; `FIRST_VALUE`/`LAST_VALUE` for "the anchor."**

| month_start | monthly_revenue | baseline_revenue | pct_growth_since_start |
|---|---|---|---|
| 04 | 38000 | 38000 | 0.0 |
| 05 | 41000 | 38000 | 7.89 |
| 06 | 39500 | 38000 | 3.95 |
| 07 | 44000 | 38000 | 15.79 |

---

## 10. Running & Rolling Metrics — Cumulative Sums, Moving Averages, Rolling Active Users

`LAG`/`LEAD` reach a *single* neighboring row. Running and rolling metrics are a different family — they aggregate *over a range of rows* (a "frame") relative to the current one. Same `OVER (PARTITION BY ... ORDER BY ...)` skeleton, but now with an explicit **frame clause**:

```sql
SUM(x) OVER (
    ORDER BY date_col
    ROWS BETWEEN <start> AND <end>
)
```

| Frame | Meaning |
|---|---|
| `UNBOUNDED PRECEDING AND CURRENT ROW` | everything from the start through today → **running total** |
| `6 PRECEDING AND CURRENT ROW` | today + previous 6 rows (7 total) → **rolling 7-period window** |
| `1 PRECEDING AND 1 FOLLOWING` | one row each side → **centered moving average** |
| `CURRENT ROW AND UNBOUNDED FOLLOWING` | today through the end → **remaining/reverse-running total** |

If you omit the frame clause entirely, most engines default to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` when there's an `ORDER BY` — which quietly gives you a running total even when you didn't ask for one. **Always write the frame clause explicitly**; relying on the default is a common source of "why is my average wrong" bugs.

### 10a. Cumulative Sum (Running Total)

**Business question:** "Running total of revenue through each day."

**Raw `daily_sales`:**

| sale_date | revenue |
|---|---|
| 07-01 | 1000 |
| 07-02 | 1200 |
| 07-03 | 900 |
| 07-04 | 1500 |

```sql
WITH base AS (
    SELECT sale_date, revenue FROM daily_sales
),
with_running_total AS (
    SELECT
        sale_date,
        revenue,
        SUM(revenue) OVER (
            ORDER BY sale_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_total
    FROM base
)
SELECT * FROM with_running_total ORDER BY sale_date;
```

**`with_running_total` output** — each row's frame is "everything so far," so the frame literally grows by one row each time you move down:

| sale_date | revenue | running_total |
|---|---|---|
| 07-01 | 1000 | 1000 |
| 07-02 | 1200 | 2200 |
| 07-03 | 900 | 3100 |
| 07-04 | 1500 | 4600 |

Swap `SUM` for `MAX`/`MIN` and you get **running max / running min** ("highest revenue day so far") with zero other changes — the frame clause is doing all the work, the aggregate function is interchangeable.

### 10b. Moving Average — Fixed Trailing Window

**Business question:** "3-day trailing moving average of revenue" (smooths out day-to-day noise).

```sql
WITH base AS (
    SELECT sale_date, revenue FROM daily_sales
)
SELECT
    sale_date,
    revenue,
    ROUND(AVG(revenue) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2) AS moving_avg_3d,
    COUNT(*) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS days_in_window
FROM base
ORDER BY sale_date;
```

**Output** — note the first two rows: the frame *asks* for 3 rows but there simply aren't 2 prior rows yet, so the engine silently uses however many exist. This is why `days_in_window` is included — **always surface the actual row count in early development**, so a 1-day "average" doesn't get mistaken for a real 3-day one:

| sale_date | revenue | moving_avg_3d | days_in_window |
|---|---|---|---|
| 07-01 | 1000 | 1000.00 | 1 |
| 07-02 | 1200 | 1100.00 | 2 |
| 07-03 | 900 | 1033.33 | 3 |
| 07-04 | 1500 | 1200.00 | 3 |

**Common variant — fixed N instead of "2 preceding":** for a "rolling 7-day average," just change the number: `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW`. The pattern is identical regardless of window length.

### 10c. Centered Moving Average

**Business question:** "Smooth the trend line by averaging each day with the day before *and* after" (used for charts/trend lines, not for live dashboards — it needs future data, so it's not usable for "today").

```sql
SELECT
    sale_date,
    revenue,
    ROUND(AVG(revenue) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ), 2) AS centered_moving_avg_3d
FROM daily_sales
ORDER BY sale_date;
```

| sale_date | revenue | centered_moving_avg_3d |
|---|---|---|
| 07-01 | 1000 | 1100.00 *(avg of 1000,1200 — no "before" exists)* |
| 07-02 | 1200 | 1033.33 *(avg of 1000,1200,900)* |
| 07-03 | 900 | 1200.00 *(avg of 1200,900,1500)* |
| 07-04 | 1500 | 1200.00 *(avg of 900,1500 — no "after" exists)* |

**Trailing vs. centered is a real design decision, not a stylistic one:** trailing (10b) is what you'd put on a live ops dashboard, since it only uses data that existed at the time. Centered (10c) is what you'd use in a retrospective report or chart, since it "peeks" at future rows — using it live would mean tomorrow's number silently changes today's chart.

### 10d. Rolling N-Day *Active Users* (the hard one — needs `COUNT DISTINCT`)

**Why this is a different problem:** `SUM`/`AVG`/`MAX` over a frame Just Work, because adding/removing one row from a running total is simple addition/subtraction. But `COUNT(DISTINCT user_id)` is **not** window-frame-friendly in most SQL engines — you can't say `COUNT(DISTINCT user_id) OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` in Postgres/MySQL (Snowflake/BigQuery are more lenient, but it's still slow at scale). The standard, portable workaround is a **self-join or date-spine join**, not a plain frame clause.

**Raw `events`:**

| event_date | user_id |
|---|---|
| 07-01 | U1 |
| 07-01 | U2 |
| 07-02 | U1 |
| 07-03 | U3 |
| 07-04 | U2 |
| 07-04 | U3 |

```sql
WITH dates AS (
    -- one row per calendar day you want a rolling metric for
    SELECT generate_series(
        (SELECT MIN(event_date) FROM events),
        (SELECT MAX(event_date) FROM events),
        INTERVAL '1 day'
    )::date AS report_date
),
windowed AS (
    -- for each report_date, pull every event that falls inside its trailing 7-day window
    SELECT
        d.report_date,
        e.user_id
    FROM dates d
    JOIN events e
        ON e.event_date > d.report_date - INTERVAL '7 days'
       AND e.event_date <= d.report_date
),
rolling_dau AS (
    SELECT
        report_date,
        COUNT(DISTINCT user_id) AS rolling_7d_active_users
    FROM windowed
    GROUP BY report_date
)
SELECT * FROM rolling_dau ORDER BY report_date;
```

**`dates` output** — the "spine" that guarantees every calendar day appears even if it had zero events (window functions alone can't invent missing dates; a generated date spine is the standard fix):

| report_date |
|---|
| 07-01 |
| 07-02 |
| 07-03 |
| 07-04 |

**`windowed` output** (partial, showing `report_date = 07-04`'s matches as an example) — this CTE's whole job is to explode each report date into all the raw event-rows that fall in its trailing window, so the next step can just `COUNT DISTINCT` per date like normal `GROUP BY` aggregation:

| report_date | user_id |
|---|---|
| 07-04 | U1 |
| 07-04 | U2 |
| 07-04 | U3 |

**`rolling_dau` final output:**

| report_date | rolling_7d_active_users |
|---|---|
| 07-01 | 2 |
| 07-02 | 2 |
| 07-03 | 3 |
| 07-04 | 3 |

**The lesson:** whenever a rolling metric involves `DISTINCT`, reach for **join-to-a-date-spine**, not a frame clause. Frame clauses are for aggregates that can be incrementally added/removed one row at a time (`SUM`, `AVG`, `MIN`, `MAX`, `COUNT(*)`); `COUNT(DISTINCT ...)` can't be incrementally maintained that way because removing a row might or might not reduce the distinct count, depending on duplicates elsewhere in the window.

### 10e. Weighted Moving Average (unequal weights per lag)

**Business question:** "Weight today 50%, yesterday 30%, the day before 20%" (recency-weighted smoothing).

This can't be a plain frame `AVG` (which weights every row in the frame equally) — combine multiple `LAG`s with manual weights instead, tying this back to Section 5:

```sql
SELECT
    sale_date,
    revenue,
    ROUND(
        revenue * 0.5
        + LAG(revenue, 1) OVER (ORDER BY sale_date) * 0.3
        + LAG(revenue, 2) OVER (ORDER BY sale_date) * 0.2
    , 2) AS weighted_moving_avg
FROM daily_sales
ORDER BY sale_date;
```

| sale_date | revenue | weighted_moving_avg |
|---|---|---|
| 07-01 | 1000 | NULL *(missing lags → NULL)* |
| 07-02 | 1200 | NULL |
| 07-03 | 900 | 1110.00 |
| 07-04 | 1500 | 1220.00 |

**Lesson:** a "moving average" isn't always a frame-clause job — as soon as the weighting is uneven, you're back to Section 5's stacked-`LAG` technique with arithmetic on top, not a `ROWS BETWEEN` frame.

### 10f. Rolling Metric Per Group (`PARTITION BY` + frame together)

Every pattern above works identically per-entity by adding `PARTITION BY` — the frame then resets at each partition boundary automatically.

```sql
SELECT
    store_id,
    sale_date,
    revenue,
    SUM(revenue) OVER (
        PARTITION BY store_id
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total_per_store
FROM store_daily_sales
ORDER BY store_id, sale_date;
```

| store_id | sale_date | revenue | running_total_per_store |
|---|---|---|---|
| S1 | 07-01 | 500 | 500 |
| S1 | 07-02 | 600 | 1100 |
| S2 | 07-01 | 300 | 300 |
| S2 | 07-02 | 400 | 700 |

Note S2's running total starts fresh at 300, not 1400 — the frame never crosses a `PARTITION BY` boundary, no matter how the frame clause is written.

### 10g. `ROWS` vs. `RANGE` — the Subtle Trap with Duplicate Order Keys

`ROWS BETWEEN 1 PRECEDING AND CURRENT ROW` means "physically 1 row back," counted by row position. `RANGE BETWEEN 1 PRECEDING AND CURRENT ROW` means "1 unit back **in the value being ordered by**" — and if two rows share the same `ORDER BY` value (e.g., two events at the identical timestamp, or two sales rows on the same date before aggregation), `RANGE` treats them as the *same point* and includes both in every frame that touches that point, while `ROWS` treats them as distinct positions.

**Rule of thumb:** default to `ROWS` unless you specifically need "ties are one unit" semantics — `ROWS` is more predictable and is what people usually mean by "trailing N rows."

---

## 11. Quick Reference — Which Pattern for Which Question

| Question shape | Function | Partition key | Order key |
|---|---|---|---|
| "vs. yesterday / last row" | `LAG(x, 1)` | none (or entity) | date |
| "vs. next occurrence" | `LEAD(x, 1)` | entity | date |
| "vs. same period last year" | `LAG(x, 1)` | calendar sub-unit (month, week-of-year) | year |
| "trailing 3 periods side by side" | `LAG(x,1)`, `LAG(x,2)`, `LAG(x,3)` | entity | date |
| "is this a local spike" | `LAG` + `LEAD` together | entity | date |
| "time until next event" | `LEAD(date_col)` then subtract | entity | date |
| "session/streak grouping" | `LAG` → `CASE` → running `SUM` | entity | date |
| "vs. a fixed starting point" | `FIRST_VALUE` (not lag/lead) | entity | date |
| "running total to date" | `SUM(x) OVER (ROWS UNBOUNDED PRECEDING AND CURRENT ROW)` | entity | date |
| "trailing N-period average" | `AVG(x) OVER (ROWS N-1 PRECEDING AND CURRENT ROW)` | entity | date |
| "smoothed trend line (retrospective)" | `AVG(x) OVER (ROWS 1 PRECEDING AND 1 FOLLOWING)` | entity | date |
| "rolling distinct-user count" | join to a date spine + `COUNT(DISTINCT)`, not a frame clause | n/a | n/a |
| "recency-weighted average" | stacked `LAG`s × manual weights | entity | date |

---

## Practice — Test Yourself

Try writing these before checking your instinct against the patterns above:

1. Daily active users, with the percent change from **the same day last week** (hint: this is #1 and #3b's technique combined — partition by day-of-week, order by week number).
2. For each employee, the number of days since their **previous** promotion and until their **next** one, in the same row (hint: #4 + #1 combined, one `LAG` and one `LEAD` on the same date column).
3. Flag every stock-price row where price dropped for **three consecutive days** (hint: #8's gaps-and-islands skeleton, but the flag condition compares to `LAG(price,1)` *and* `LAG(price,2)`).
4. Running total of sign-ups per acquisition channel, reset each calendar year (hint: #10a's frame, but think about what has to change in the `PARTITION BY` to make the total reset on Jan 1).
5. Rolling 30-day *distinct* active users, but you only have a raw `events` table with millions of rows — sketch why a naive `COUNT(DISTINCT ...) OVER (ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)` would be a bad idea here, and what you'd do instead (hint: #10d).
