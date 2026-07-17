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

Want me to build out full solutions (with sample data and CTE-by-CTE traces, same style as above) for any of these five?

---

# Part II: Product Analytics Patterns

Same teaching method continues: sample input → CTE by CTE → sample output. These patterns lean on window functions from Part I (`LAG`, running `SUM`, `PARTITION BY`) but combine them with `GROUP BY`, self-joins, and conditional aggregation — the actual toolkit behind product-analytics dashboards.

## 12. DAU / WAU / MAU + Stickiness

**Business question:** "How engaged are users, relative to how many we have?" Stickiness = DAU / MAU, the classic proxy for habitual use.

**Raw `events`:**

| event_date | user_id |
|---|---|
| 07-01 | U1 |
| 07-01 | U2 |
| 07-15 | U1 |
| 07-15 | U3 |
| 07-20 | U2 |

```sql
WITH daily AS (
    SELECT event_date, COUNT(DISTINCT user_id) AS dau
    FROM events
    GROUP BY event_date
),
weekly AS (
    SELECT DATE_TRUNC('week', event_date) AS week_start,
           COUNT(DISTINCT user_id) AS wau
    FROM events
    GROUP BY 1
),
monthly AS (
    SELECT DATE_TRUNC('month', event_date) AS month_start,
           COUNT(DISTINCT user_id) AS mau
    FROM events
    GROUP BY 1
),
joined AS (
    SELECT
        d.event_date,
        d.dau,
        m.mau,
        ROUND(d.dau * 1.0 / NULLIF(m.mau, 0), 3) AS dau_mau_stickiness
    FROM daily d
    JOIN monthly m
        ON DATE_TRUNC('month', d.event_date) = m.month_start
)
SELECT * FROM joined ORDER BY event_date;
```

**`daily` / `monthly` outputs** — each is an ordinary `COUNT(DISTINCT)` per grain, computed independently (you cannot derive MAU by summing DAU — a user active 20 separate days still counts once in MAU, so MAU must always come from its *own* `GROUP BY`, never a rollup of the daily counts):

| event_date | dau |
|---|---|
| 07-01 | 2 |
| 07-15 | 2 |
| 07-20 | 1 |

| month_start | mau |
|---|---|
| 07-01 | 3 |

**`joined` final output** — the stickiness ratio is just DAU divided by the MAU of the month that day belongs to:

| event_date | dau | mau | dau_mau_stickiness |
|---|---|---|---|
| 07-01 | 2 | 3 | 0.667 |
| 07-15 | 2 | 3 | 0.667 |
| 07-20 | 1 | 3 | 0.333 |

**Lesson:** DAU, WAU, MAU are each **independent `COUNT(DISTINCT)` aggregations at their own grain** — never try to compute MAU by summing or averaging daily counts, since that double-counts repeat users.

---

## 13. L1 / L7 / L28 Active User Windows

**Business question:** distinct from DAU/MAU — "of the users active *today*, what fraction were also active at least once in the trailing 7 (or 28) days?" This is a per-user retention-density metric, common in growth teams (e.g., "L7/L28 ratio" as a habit-strength signal), and it needs the date-spine + join technique from Part I §10d, since it's a rolling `COUNT(DISTINCT)`.

```sql
WITH dates AS (
    SELECT generate_series(
        (SELECT MIN(event_date) FROM events),
        (SELECT MAX(event_date) FROM events),
        INTERVAL '1 day'
    )::date AS report_date
),
l1 AS (
    SELECT d.report_date, COUNT(DISTINCT e.user_id) AS l1_users
    FROM dates d
    JOIN events e ON e.event_date = d.report_date
    GROUP BY d.report_date
),
l7 AS (
    SELECT d.report_date, COUNT(DISTINCT e.user_id) AS l7_users
    FROM dates d
    JOIN events e
        ON e.event_date > d.report_date - INTERVAL '7 days'
       AND e.event_date <= d.report_date
    GROUP BY d.report_date
),
l28 AS (
    SELECT d.report_date, COUNT(DISTINCT e.user_id) AS l28_users
    FROM dates d
    JOIN events e
        ON e.event_date > d.report_date - INTERVAL '28 days'
       AND e.event_date <= d.report_date
    GROUP BY d.report_date
)
SELECT
    l1.report_date, l1_users, l7_users, l28_users,
    ROUND(l1_users * 1.0 / NULLIF(l7_users, 0), 3) AS l1_l7_ratio,
    ROUND(l1_users * 1.0 / NULLIF(l28_users, 0), 3) AS l1_l28_ratio
FROM l1
JOIN l7 USING (report_date)
JOIN l28 USING (report_date)
ORDER BY report_date;
```

| report_date | l1_users | l7_users | l28_users | l1_l7_ratio | l1_l28_ratio |
|---|---|---|---|---|---|
| 07-20 | 1 | 2 | 3 | 0.500 | 0.333 |

**Lesson:** `L1/L7`, `L1/L28` are structurally identical to DAU/MAU stickiness (§12) — just with rolling windows instead of calendar-month windows. Once you've built one rolling `COUNT(DISTINCT)` CTE, every "LN active users" variant is a copy-paste with a different interval.

---

## 14. Retention Analysis (Day-N Retention & Cohorts)

**Business question:** "Of users who signed up on a given day, what % were still active exactly N days later?"

**Raw tables:** `signups(user_id, signup_date)`, `events(user_id, event_date)`.

```sql
WITH cohort AS (
    SELECT user_id, signup_date FROM signups
),
activity AS (
    SELECT DISTINCT user_id, event_date FROM events
),
joined AS (
    SELECT
        c.user_id,
        c.signup_date,
        a.event_date,
        (a.event_date - c.signup_date) AS days_since_signup
    FROM cohort c
    JOIN activity a ON a.user_id = c.user_id
                    AND a.event_date >= c.signup_date
),
day_n_flags AS (
    SELECT
        signup_date,
        user_id,
        MAX(CASE WHEN days_since_signup = 1  THEN 1 ELSE 0 END) AS retained_day1,
        MAX(CASE WHEN days_since_signup = 7  THEN 1 ELSE 0 END) AS retained_day7,
        MAX(CASE WHEN days_since_signup = 28 THEN 1 ELSE 0 END) AS retained_day28
    FROM joined
    GROUP BY signup_date, user_id
)
SELECT
    signup_date,
    COUNT(*) AS cohort_size,
    ROUND(AVG(retained_day1) , 3) AS day1_retention,
    ROUND(AVG(retained_day7) , 3) AS day7_retention,
    ROUND(AVG(retained_day28), 3) AS day28_retention
FROM day_n_flags
GROUP BY signup_date
ORDER BY signup_date;
```

**`day_n_flags` output** — one flag row per user, `MAX(CASE ...)` collapses potentially many activity rows per user down to a single 0/1 "were they active on exactly day N":

| signup_date | user_id | retained_day1 | retained_day7 |
|---|---|---|---|
| 07-01 | U1 | 1 | 1 |
| 07-01 | U2 | 0 | 1 |

**Final output** — `AVG` of a 0/1 flag *is* the retention rate, a trick worth internalizing (`AVG(boolean_as_int)` = percentage):

| signup_date | cohort_size | day1_retention | day7_retention |
|---|---|---|---|
| 07-01 | 2 | 0.5 | 1.0 |

**Lesson:** the `MAX(CASE WHEN days_since_signup = N THEN 1 ELSE 0 END)` idiom, then `AVG()` of that flag, is the single most reusable trick in product analytics SQL — it converts "did this ever happen" into a clean percentage without a self-join per day-N.

---

## 15. Conversion Funnel Basics

**Business question:** "Of users who viewed the product page, what % added to cart, and what % of *those* checked out?" — an unordered, presence-based funnel (each stage only checks "did this event ever happen," not sequence).

**Raw `events(user_id, event_name, event_time)`**

```sql
WITH flags AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_name = 'view_product' THEN 1 ELSE 0 END) AS viewed,
        MAX(CASE WHEN event_name = 'add_to_cart'   THEN 1 ELSE 0 END) AS added_to_cart,
        MAX(CASE WHEN event_name = 'checkout'      THEN 1 ELSE 0 END) AS checked_out
    FROM events
    GROUP BY user_id
)
SELECT
    SUM(viewed)                                   AS step1_viewed,
    SUM(added_to_cart)                             AS step2_added,
    SUM(checked_out)                               AS step3_checkout,
    ROUND(SUM(added_to_cart) * 1.0 / NULLIF(SUM(viewed), 0), 3)       AS view_to_cart_rate,
    ROUND(SUM(checked_out)   * 1.0 / NULLIF(SUM(added_to_cart), 0), 3) AS cart_to_checkout_rate
FROM flags;
```

| step1_viewed | step2_added | step3_checkout | view_to_cart_rate | cart_to_checkout_rate |
|---|---|---|---|---|
| 500 | 150 | 60 | 0.300 | 0.400 |

**Caveat this basic version ignores:** it doesn't check *order* — a user who added to cart before ever viewing the product page (e.g., via a saved link) still counts as "converted" at every stage. That's fine for a rough funnel; it's wrong for anything claiming causal step-by-step drop-off. That's what the next pattern fixes.

---

## 16. Ordered Funnel (Sequential Steps Must Happen in Order)

**Business question:** same funnel, but a step only counts if it happened **at or after** the previous step's timestamp — i.e., genuine sequential progression.

```sql
WITH view_evt AS (
    SELECT user_id, MIN(event_time) AS viewed_at
    FROM events WHERE event_name = 'view_product'
    GROUP BY user_id
),
cart_evt AS (
    SELECT e.user_id, MIN(e.event_time) AS added_at
    FROM events e
    JOIN view_evt v ON v.user_id = e.user_id
    WHERE e.event_name = 'add_to_cart'
      AND e.event_time >= v.viewed_at
    GROUP BY e.user_id
),
checkout_evt AS (
    SELECT e.user_id, MIN(e.event_time) AS checked_out_at
    FROM events e
    JOIN cart_evt c ON c.user_id = e.user_id
    WHERE e.event_name = 'checkout'
      AND e.event_time >= c.added_at
    GROUP BY e.user_id
)
SELECT
    (SELECT COUNT(*) FROM view_evt)     AS step1_viewed,
    (SELECT COUNT(*) FROM cart_evt)     AS step2_added_in_order,
    (SELECT COUNT(*) FROM checkout_evt) AS step3_checkout_in_order;
```

**Lesson:** each CTE **only builds on the previous CTE's output**, not the raw table directly — `cart_evt` joins against `view_evt`, and `checkout_evt` joins against `cart_evt`. This chaining is what actually enforces "in order": a user can only appear in `checkout_evt` if they made it through `cart_evt` first, which itself required making it through `view_evt` first. This is the core structural difference from §15 — the funnel's *shape* comes from CTEs referencing each other in sequence, not from a single flat `GROUP BY`.

| step1_viewed | step2_added_in_order | step3_checkout_in_order |
|---|---|---|
| 500 | 120 | 55 |

Notice `step2_added_in_order` (120) is lower than §15's `step2_added` (150) — the 30-user gap is exactly the users who added to cart *before* their tracked page view, and the ordered funnel correctly excludes them.

---

## 17. Session Analysis (Duration, Events per Session, Bounce Rate)

Builds directly on Part I §8's gaps-and-islands session-ID technique — session grouping is a prerequisite, not a separate skill.

```sql
WITH with_session_id AS (
    -- (this is exactly Part I §8's with_session_id CTE — session_id per user via LAG + running SUM)
    SELECT user_id, viewed_at, session_id FROM with_session_id_from_part_1
),
session_stats AS (
    SELECT
        user_id,
        session_id,
        MIN(viewed_at) AS session_start,
        MAX(viewed_at) AS session_end,
        COUNT(*) AS events_in_session,
        EXTRACT(EPOCH FROM (MAX(viewed_at) - MIN(viewed_at))) / 60.0 AS duration_minutes
    FROM with_session_id
    GROUP BY user_id, session_id
)
SELECT
    COUNT(*) AS total_sessions,
    ROUND(AVG(duration_minutes), 2) AS avg_session_minutes,
    ROUND(AVG(events_in_session), 2) AS avg_events_per_session,
    ROUND(AVG(CASE WHEN events_in_session = 1 THEN 1 ELSE 0 END), 3) AS bounce_rate
FROM session_stats;
```

**Lesson:** "bounce rate" is the same `AVG(CASE WHEN ... THEN 1 ELSE 0 END)` percentage idiom from §14 — a single-event session is a "bounce," and averaging that 0/1 flag across all sessions gives the rate directly.

---

## 18. RFM Segmentation (Recency, Frequency, Monetary)

**Business question:** "Score every customer on how recently they bought, how often, and how much, then bucket them into segments (e.g., 'Champions', 'At Risk')."

```sql
WITH customer_stats AS (
    SELECT
        customer_id,
        MAX(order_date) AS last_order_date,
        COUNT(*) AS order_count,
        SUM(order_amount) AS total_spend
    FROM orders
    GROUP BY customer_id
),
with_recency AS (
    SELECT
        customer_id,
        (CURRENT_DATE - last_order_date) AS days_since_last_order,
        order_count,
        total_spend
    FROM customer_stats
),
scored AS (
    SELECT
        customer_id,
        NTILE(5) OVER (ORDER BY days_since_last_order DESC) AS recency_score,  -- more recent = higher score
        NTILE(5) OVER (ORDER BY order_count ASC)             AS frequency_score,
        NTILE(5) OVER (ORDER BY total_spend ASC)             AS monetary_score
    FROM with_recency
)
SELECT
    customer_id,
    recency_score, frequency_score, monetary_score,
    (recency_score + frequency_score + monetary_score) AS rfm_total,
    CASE
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champion'
        WHEN recency_score <= 2 AND frequency_score >= 4 THEN 'At Risk (was loyal)'
        WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost'
        ELSE 'Regular'
    END AS segment
FROM scored;
```

**Lesson:** `NTILE(5)` is the workhorse here — it buckets customers into quintiles based on rank, which is what turns a continuous number (days since last order, total spend) into a comparable 1–5 score across totally different units. The segment labels are then just `CASE` logic layered on top of three independently-computed scores — nothing about RFM requires anything beyond `NTILE` + `CASE`.

---

## 19. A/B Test Results

**Business question:** "Which variant converted better, and by roughly how much?" (SQL computes the rates; a proper significance test — z-test/chi-square — typically happens downstream in Python/R/a stats tool, but the SQL groundwork is identical every time.)

```sql
WITH assignment AS (
    SELECT user_id, variant FROM experiment_assignments
),
conversions AS (
    SELECT DISTINCT user_id FROM orders
),
joined AS (
    SELECT
        a.variant,
        a.user_id,
        CASE WHEN c.user_id IS NOT NULL THEN 1 ELSE 0 END AS converted
    FROM assignment a
    LEFT JOIN conversions c ON c.user_id = a.user_id
)
SELECT
    variant,
    COUNT(*) AS users,
    SUM(converted) AS conversions,
    ROUND(AVG(converted), 4) AS conversion_rate,
    ROUND(
        SQRT(AVG(converted) * (1 - AVG(converted)) / COUNT(*))
    , 4) AS standard_error
FROM joined
GROUP BY variant
ORDER BY variant;
```

| variant | users | conversions | conversion_rate | standard_error |
|---|---|---|---|---|
| control | 10000 | 820 | 0.0820 | 0.0027 |
| treatment | 10000 | 910 | 0.0910 | 0.0029 |

**Lesson:** the `LEFT JOIN` + `CASE WHEN ... IS NOT NULL` idiom is how you turn "did this user appear in a different table at all" into a 0/1 flag — combined with `AVG()`, this is the same conversion-rate pattern as §14/§17, just applied per experiment arm. The standard error column is included because a bare rate difference (9.1% vs 8.2%) means nothing without it — always carry variance/SE alongside a rate comparison, even before a formal significance test.

---

## 20. Feature Adoption & Power Users

**Business question:** "What % of active users have adopted a new feature, and who are the 'power users' driving most of its usage?"

```sql
WITH feature_usage AS (
    SELECT
        user_id,
        COUNT(*) AS feature_uses
    FROM events
    WHERE event_name = 'used_new_feature'
    GROUP BY user_id
),
active_users AS (
    SELECT DISTINCT user_id FROM events
    WHERE event_date >= CURRENT_DATE - INTERVAL '28 days'
),
adoption AS (
    SELECT
        a.user_id,
        COALESCE(f.feature_uses, 0) AS feature_uses,
        CASE WHEN f.user_id IS NOT NULL THEN 1 ELSE 0 END AS adopted
    FROM active_users a
    LEFT JOIN feature_usage f ON f.user_id = a.user_id
),
ranked AS (
    SELECT
        user_id,
        feature_uses,
        PERCENT_RANK() OVER (ORDER BY feature_uses) AS usage_percentile
    FROM adoption
    WHERE adopted = 1
)
SELECT
    (SELECT AVG(adopted) FROM adoption) AS adoption_rate,
    COUNT(*) FILTER (WHERE usage_percentile >= 0.9) AS power_user_count
FROM ranked;
```

**Lesson:** `PERCENT_RANK()` (a windowed sibling of `NTILE`) is the natural fit for "power user" cutoffs, because it's threshold-based ("top 10% of usage") rather than fixed-bucket-based — swap the `0.9` to redefine "power user" without restructuring the query. `COALESCE` + `LEFT JOIN` again turns "absence from a table" into a real 0 rather than a `NULL`, which matters because `AVG()` silently ignores `NULL`s and would inflate the adoption rate if non-adopters weren't explicitly zeroed out.

---

## 21. Search & Ads Metrics (CTR, RPM, Quality Score, nDCG)

**CTR (click-through rate) and RPM (revenue per mille):**

```sql
WITH impressions AS (
    SELECT query_id, COUNT(*) AS impression_count
    FROM ad_impressions
    GROUP BY query_id
),
clicks AS (
    SELECT query_id, COUNT(*) AS click_count, SUM(revenue) AS total_revenue
    FROM ad_clicks
    GROUP BY query_id
)
SELECT
    i.query_id,
    i.impression_count,
    COALESCE(c.click_count, 0) AS click_count,
    ROUND(COALESCE(c.click_count, 0) * 1.0 / NULLIF(i.impression_count, 0), 4) AS ctr,
    ROUND(COALESCE(c.total_revenue, 0) * 1000.0 / NULLIF(i.impression_count, 0), 2) AS rpm
FROM impressions i
LEFT JOIN clicks c ON c.query_id = i.query_id;
```

RPM is literally "revenue, scaled to a per-1000-impressions rate" — the `* 1000.0` is the entire definition; everything else is the same ratio-with-`NULLIF` pattern used throughout this masterclass.

**nDCG (normalized Discounted Cumulative Gain)** — measures ranking quality: a relevant result ranked #1 is worth more than the same relevant result ranked #10.

```sql
WITH ranked_results AS (
    SELECT
        query_id,
        result_id,
        relevance_score,             -- e.g. 0-3, human/click-derived label
        ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY displayed_rank) AS position
    FROM search_results
),
dcg AS (
    SELECT
        query_id,
        SUM(relevance_score / LOG(2, position + 1)) AS dcg
    FROM ranked_results
    GROUP BY query_id
),
ideal_dcg AS (
    SELECT
        query_id,
        SUM(relevance_score / LOG(2, ideal_position + 1)) AS idcg
    FROM (
        SELECT
            query_id,
            relevance_score,
            ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY relevance_score DESC) AS ideal_position
        FROM ranked_results
    ) ideal
    GROUP BY query_id
)
SELECT
    d.query_id,
    ROUND(d.dcg / NULLIF(i.idcg, 0), 4) AS ndcg
FROM dcg d
JOIN ideal_dcg i ON i.query_id = d.query_id;
```

**Lesson:** `dcg` uses `ROW_NUMBER()` on the *actual displayed order*; `idcg` uses `ROW_NUMBER()` on relevance sorted *descending* — the same window function, applied to two different orderings of the same rows, is exactly what "actual vs. ideal ranking" means mathematically. nDCG is just `dcg / idcg`, capped at 1.0 when the actual order matches the ideal order.

**Quality Score** (ads) is typically a weighted blend of CTR, landing-page relevance, and ad relevance — structurally, it's the same weighted-combination idiom as Part I §10e's weighted moving average, just combining different metrics instead of different time lags:

```sql
SELECT
    query_id,
    ROUND(ctr_score * 0.5 + relevance_score * 0.3 + landing_page_score * 0.2, 3) AS quality_score
FROM ad_component_scores;
```

---

## 22. Ranking & Scoring Systems

**Business question:** "Rank products within each category by sales, handling ties correctly."

```sql
WITH sales_by_product AS (
    SELECT category, product_id, SUM(units_sold) AS units_sold
    FROM product_sales
    GROUP BY category, product_id
)
SELECT
    category,
    product_id,
    units_sold,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY units_sold DESC) AS row_num_rank,
    RANK()       OVER (PARTITION BY category ORDER BY units_sold DESC) AS rank_with_gaps,
    DENSE_RANK() OVER (PARTITION BY category ORDER BY units_sold DESC) AS dense_rank_no_gaps
FROM sales_by_product;
```

**Output — the three ranking functions diverge exactly at a tie**, which is the whole reason all three exist:

| category | product_id | units_sold | row_num_rank | rank_with_gaps | dense_rank_no_gaps |
|---|---|---|---|---|---|
| Shoes | P1 | 500 | 1 | 1 | 1 |
| Shoes | P2 | 400 | 2 | 2 | 2 |
| Shoes | P3 | 400 | 3 | 2 | 2 |
| Shoes | P4 | 300 | 4 | 4 | 3 |

**Lesson:** `ROW_NUMBER()` always assigns unique, arbitrary-among-ties numbers (P2/P3 tied at 400 still get 2 and 3) — never use it when ties should be treated as equal. `RANK()` gives ties the same number but **skips** the next number(s) (jumps 2→4). `DENSE_RANK()` gives ties the same number and does **not** skip (2→3). Picking the wrong one is a common, silent bug — e.g., "top 3 products" using `ROW_NUMBER()` when ties existed will arbitrarily exclude a product that's tied for 3rd.

---

## 23. Large-Scale Aggregations with ROLLUP / CUBE / GROUPING SETS

**Business question:** "Total revenue by region and product, *plus* subtotals per region, *plus* the grand total — all in one result set" (classic reporting/pivot-table need).

```sql
SELECT
    COALESCE(region, 'ALL REGIONS') AS region,
    COALESCE(product, 'ALL PRODUCTS') AS product,
    SUM(revenue) AS total_revenue,
    GROUPING(region) AS is_region_subtotal,
    GROUPING(product) AS is_product_subtotal
FROM sales
GROUP BY ROLLUP(region, product)
ORDER BY region, product;
```

**Output** — `ROLLUP(region, product)` produces every row you'd get from `GROUP BY region, product`, **plus** one subtotal row per region (product = NULL), **plus** one grand-total row (both NULL) — in a single query instead of a `UNION ALL` of three separate `GROUP BY`s:

| region | product | total_revenue | is_region_subtotal | is_product_subtotal |
|---|---|---|---|---|
| APAC | Shoes | 20000 | 0 | 0 |
| APAC | Shirts | 15000 | 0 | 0 |
| APAC | ALL PRODUCTS | 35000 | 0 | 1 |
| EMEA | Shoes | 18000 | 0 | 0 |
| EMEA | ALL PRODUCTS | 18000 | 0 | 1 |
| ALL REGIONS | ALL PRODUCTS | 53000 | 1 | 1 |

**Lesson:** the `GROUPING()` function is essential, not decorative — it's the only reliable way to tell a *real* NULL region (e.g., unattributed sales) apart from a NULL produced *by* the rollup as a subtotal marker. Without it, `COALESCE(region, 'ALL REGIONS')` would wrongly relabel genuinely-unknown regions as grand totals. `CUBE(region, product)` is the same idea but adds subtotals for *every* combination (including product-only, ignoring region) — use it when you need cross-tab subtotals in both directions, not just a hierarchy.

**Partitioning note (physical, not `PARTITION BY` the window-function clause):** at real scale, "large-scale aggregation" usually also means the underlying table is physically partitioned (e.g., by `sale_date`) so the engine prunes irrelevant partitions before scanning — this is a table/storage design decision made by whoever owns the warehouse schema, not something expressed inside the `SELECT`, but it's the difference between a rollup query scanning terabytes vs. scanning the one month you actually filtered on.

---

## 24. Edge Cases: NULLs, Division by Zero, Sparse Data

These aren't a separate "topic" so much as a checklist to run against every pattern above before shipping it.

**a) Division by zero / by NULL:**
```sql
-- Wrong: errors or returns NULL unpredictably across engines when denominator is 0
SELECT clicks / impressions AS ctr FROM stats;

-- Right: NULLIF turns 0 into NULL, so the division returns NULL (not an error) — then COALESCE if you want 0 instead
SELECT COALESCE(clicks * 1.0 / NULLIF(impressions, 0), 0) AS ctr FROM stats;
```

**b) `AVG`/`SUM` silently ignoring NULLs — the "missing user" trap:**
`AVG(converted)` (§14, §19) only works as a clean rate *because* every row has an explicit 0 or 1. The moment a `LEFT JOIN` leaves some rows as `NULL` instead of `0` (as in §18's `feature_uses` before the `COALESCE`), `AVG()` quietly excludes those rows from both numerator and denominator, inflating the rate. **Always `COALESCE` a joined flag/count to 0 before aggregating it** — this single habit prevents the majority of "why is my metric too high" bugs in this whole document.

**c) Sparse data — dates/categories with zero rows don't appear at all:**
`GROUP BY` can only group rows that exist. A day with zero events simply won't appear in the output — it won't show up as `0`, it won't show up as anything. This is why §10d, §13, and §14 all build an explicit date spine (`generate_series`) and `LEFT JOIN`/`JOIN` the real data onto it, rather than trusting `GROUP BY` to surface every date on its own. Whenever a rolling or cohort metric needs to show *zero* for an inactive day rather than *skip* that day, a spine is mandatory, not optional polish.

**d) Small-sample noise in ratios:** a category with 3 impressions and 1 click has a 33% CTR — technically correct, meaningless in practice. When ranking or segmenting by a ratio metric (CTR, conversion rate, retention), pair it with its raw denominator (as in §19's `users` column) so downstream consumers can filter out statistically meaningless small-sample rows, rather than presenting a bare percentage that looks equally confident at every sample size.

---

# Other Product Analytics Topics Worth Mastering

Beyond what's built out above, these come up often enough in real product-analytics/growth SQL work to be worth knowing exist, even in brief:

- **North Star metric trees** — decomposing one top-line metric into the multiplicative/additive drivers that feed it (e.g., Revenue = Sessions × Conversion Rate × AOV), each driver individually queryable with the patterns above.
- **Cohort LTV curves** — cumulative revenue per signup-cohort over time since signup, combining §14's cohort join with Part I §10a's running total.
- **Churn & reactivation classification** — labeling each user each period as `new` / `retained` / `churned` / `resurrected` by comparing this period's activity flag to last period's via `LAG` (Part I) over a per-user, per-period grid.
- **K-factor / virality coefficient** — invites sent per user × conversion rate of those invites, a ratio-of-ratios built from the same conditional-aggregation idioms as §15–16.
- **AARRR ("Pirate Metrics") funnel** — Acquisition/Activation/Retention/Referral/Revenue as one connected ordered funnel, directly extending §16.
- **Multi-touch attribution** — splitting credit for a conversion across several earlier touchpoints (first-touch, last-touch, linear, time-decay), usually built with `ROW_NUMBER()`/`LAG`/`LEAD` over a user's touchpoint history.
- **Percentile-based segmentation (`PERCENTILE_CONT`/`PERCENTILE_DISC`)** — for continuous cutoffs (e.g., "median session length") rather than `NTILE`'s fixed bucket counts.
- **Novelty/recency-decay weighting** — down-weighting older events in an engagement score, the same weighted-`LAG` idea from Part I §10e generalized to arbitrary decay functions (e.g., exponential half-life).
- **Seasonality-adjusted comparisons** — comparing a metric to the same weekday last month/year rather than a raw adjacent period, extending Part I §3b's "partition by calendar sub-unit" trick to day-of-week.
- **Time-to-value / time-to-first-key-action** — first occurrence of an activation event minus signup time, a `MIN()`-per-user pattern feeding directly into retention analysis.
- **Basket/cross-sell analysis (market basket, product affinity)** — self-joining an order-items table to itself to count co-purchase frequency, then computing lift over independent purchase rates.
- **Anomaly detection via z-scores** — `(value - AVG(value) OVER (...)) / STDDEV(value) OVER (...)`, directly reusing Part I's window-frame machinery with `STDDEV` instead of `SUM`/`AVG`.
- **Survivorship bias & Simpson's paradox checks** — sanity patterns (segment-level breakdowns before trusting an aggregate trend) rather than a single query pattern, but worth actively watching for whenever an aggregate metric moves in a direction that a segment-level `GROUP BY` doesn't confirm.
- **Guardrail metrics in experiments** — extending §19's A/B pattern to also monitor secondary/negative metrics (latency, complaints, unsubscribes) per variant, not just the primary conversion metric.

Want full CTE-traced build-outs (same depth as the sections above) for any of these — churn/reactivation state labeling and cohort LTV curves are usually the two most requested next.
