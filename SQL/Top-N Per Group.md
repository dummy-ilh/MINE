# Top-N Per Group — Practice Session Notes

## The Core Pattern (Reference)

**When to use it:** Anytime the question says "per [group]" + "top/first/last/most recent N."

**The skeleton:**
```sql
SELECT *
FROM (
    SELECT
        <group_col>,
        <metric>,
        ROW_NUMBER() OVER (PARTITION BY <group_col> ORDER BY <metric> DESC) AS rn
    FROM <table>
) ranked
WHERE rn <= N;
```

**Ranking function decision table:**

| Function | Ties | Gaps after ties | Use when |
|---|---|---|---|
| `ROW_NUMBER()` | Breaks ties arbitrarily/by tiebreaker | No gaps | Need **exactly N** rows, ties don't matter or are broken deterministically |
| `RANK()` | Same rank for ties | Leaves gaps | Want ties included, and gaps in ranking are fine |
| `DENSE_RANK()` | Same rank for ties | No gaps | Want ties included, ranking by **distinct value tiers** |

**Key distinction drilled throughout the session:**
- **Window functions** → keep individual rows, annotate with position within a group.
- **`GROUP BY` + aggregates** (`MIN`, `MAX`, `SUM`, `COUNT`, `AVG`) → collapse rows into a summary, one row per group, no row-level detail preserved.

**Failure modes to watch for:**
1. Filtering `WHERE rn <= 3` in the same `SELECT` as the window function — illegal, window functions evaluate after `WHERE` in logical query order.
2. Using `ROW_NUMBER()` when ties should be included → wrong row count.
3. Forgetting a deterministic tiebreaker in `ORDER BY` when interviewer wants exactly N rows.
4. Using `RANK()` when you actually want "top N distinct values" → should be `DENSE_RANK()`.
5. Using `MAX()`/aggregate when the question actually wants a specific row (row-level detail) that happens to have the max value — that's a window function job, not a `GROUP BY` job.

---

## Q1 — Basic Top-N, No Ties

**Question:** Top 2 highest-paid employees in each department. `employees(employee_id, department, salary)`. Assume no ties matter.

**Attempt 1:**
```sql
Select employee_id, department, salary 
From (
select *,
DENSE_RANK() OVER ( PARTITION BY department order by salary) as rn 
from employees) 
where rn <3
```
**Notes/misses:**
- Missing `DESC` in `ORDER BY salary` — ranked ascending, so `rn <= 2` returned the **lowest**-paid, not highest-paid. Classic "runs but wrong answer" bug — always sanity-check sort direction against what "top" means.
- Used `DENSE_RANK()` where `ROW_NUMBER()` is the safer default for "exactly N" when ties aren't supposed to matter — `DENSE_RANK()` could return 3+ rows if there's a tie at rank 2.

**Final (correct):**
```sql
Select employee_id, department, salary From ( select *, ROW_NUMBER() OVER ( PARTITION BY department order by salary desc) as rn from employees) where rn <3
```

---

## Q2 — Top-N Distinct Values, Ties Included

**Question:** Top 2 **distinct** salary values per department — if 3 people tie for 2nd highest, include all of them.

**Attempt 1:**
```sql
New as ( select *, DENSE_RANK() OVER ( PARTITION BY department order by salary desc) as rn from employees)

select distinct(salary) 
from new 
where rn <3
```
**Notes/misses:**
- `DENSE_RANK()` choice was correct.
- But answered the wrong shape of question — the ask was for employee-level rows (who are the people in the top tiers), not just the distinct salary values. `SELECT DISTINCT(salary)` collapses back to unique salary numbers with no employee identity attached.
- Syntax note: `DISTINCT(salary)` — parentheses don't scope `DISTINCT` to one column; `DISTINCT` always applies to the whole selected row. `SELECT DISTINCT(salary)` behaves identically to `SELECT DISTINCT salary`. If more columns were added, e.g. `SELECT DISTINCT(salary), employee_id`, the parens would do nothing and every row would still show since `employee_id` makes rows unique.

**Final (correct):**
```sql
WITH new AS (
    SELECT *,
           DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rn
    FROM employees
)
SELECT employee_id, department, salary
FROM new
WHERE rn <= 2;
```
**Lesson locked in:** skeleton never changes between Q1 and Q2 — only the ranking function changes based on how the question treats ties.

---

## Q3 — Compound ORDER BY Tiebreak

**Question:** `products(product_id, category, revenue, launch_date)`. For each category, find the most recently launched product; tie-break by higher revenue.

**Final answer (correct on first full attempt):**
```sql
WITH new AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY launch_date DESC, revenue DESC) AS rn
    FROM products
)
SELECT product_id, category, launch_date, revenue
FROM new
WHERE rn = 1;
```
**Notes/misses:**
- Correct compound `ORDER BY` instinct (primary sort + tiebreak column) on first try.
- Minor style note: backticks around `` `ROW_NUMBER` `` and `` `products` `` are MySQL-specific escaping for reserved words — unnecessary here since neither is reserved. Habitually backticking non-reserved identifiers can read as dialect uncertainty in an interview.

---

## Q4 — GROUP BY vs. Window Function (the pivot point)

**Question:** `orders(order_id, customer_id, order_date, amount)`. For each customer, find first order date and most recent order date — one row per customer.

**Attempt 1 (window-function approach, incorrect):**
```sql
WITH new AS
 ( SELECT *, 
ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC ) AS rn,
ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date ASC ) AS rn2,
 FROM orders)

 SELECT order_id, customer_id, order_date,rn, rn2  FROM new WHERE rn =1 and rn2=1;
```
**Notes/misses:**
- Good instinct to use two window functions for two endpoints, but the filter logic was broken: `rn = 1 AND rn2 = 1` on the *same row* only holds true if a customer has exactly one order total. Any customer with 2+ orders gets filtered out entirely, since no single row can simultaneously be both the most recent and the first.
- Shape problem: returned `rn`/`rn2` instead of `first_order_date`/`last_order_date`, and only one `order_date` column instead of two side-by-side values.
- Root issue: reached for window functions out of pattern-matching momentum from Q1–Q3, but the prompt ("one row per customer," no need for row-level order_id) was actually a signal for `GROUP BY` + aggregates, not ranking.

**Final (correct, after direct explanation):**
```sql
SELECT customer_id,
       MIN(order_date) AS first_order_date,
       MAX(order_date) AS last_order_date
FROM orders
GROUP BY customer_id;
```
**Key distinction locked in here:**
- Window functions → keep individual rows, annotate position within group.
- `GROUP BY` + aggregates → collapse rows into a summary; use when output is one row per group and row-level detail isn't needed.

---

## Q4b — Confirming the Pivot Stuck

**Question:** `reviews(review_id, product_id, rating, review_date)`. For each product, return number of reviews and average rating.

**Final answer (correct, fast):**
```sql
SELECT product_id,
       COUNT(rating) AS rating_cnt,
       AVG(rating) AS rating_avg
FROM reviews
GROUP BY product_id;
```
**Notes/misses:**
- Correctly and quickly recognized this as a `GROUP BY` job — confirmed the Q4 lesson landed.
- Minor note (not a bug): `COUNT(rating)` vs `COUNT(*)` — these differ if `rating` can be `NULL` (a review left without a rating). If the question means "number of reviews" literally, `COUNT(*)` is more precise since it counts every review row regardless of null ratings; `COUNT(rating)` would undercount in that case.
- Backticks around `` `reviews` `` unnecessary, same recurring note.

---

## Q5 — Combining Aggregate + Ranking

**Question:** `orders(order_id, customer_id, order_date, amount)`. Find top 3 customers by total spend, showing their total spend.

**Attempt 1 (syntax errors):**
```sql
With data as  (SELECT customer_id, SUM(amount) AS rating_cnt, FROM `orders `GROUP BY customer_id;)

select * from data
order by  rating_cnt desc
limit 3
```
**Notes/misses:**
- Correct high-level plan: aggregate first (SUM per customer), then rank/limit — recognized this needs both tools.
- Syntax bugs: trailing comma before `FROM`; semicolon inside the CTE parens before it was closed (terminates the statement prematurely); stray space inside backticked `` `orders ` `` breaking identifier resolution.
- Naming nit: column named `rating_cnt` for a `SUM(amount)` — leftover copy-paste from a previous question.

**Attempt 2 (correct, but conceptually incomplete):**
```sql
With data as (SELECT customer_id, SUM(amount) AS amt_s FROM orders GROUP BY customer_id)
select * from data order by amt_s desc limit 3
```
**Notes/misses:**
- Syntax clean, correctly answers the literal ask ("top 3 customers," exact count implied).
- Conceptual gap flagged: `LIMIT N` is blind to ties, same as `ROW_NUMBER() ... WHERE rn <= N` — it forces an arbitrary cutoff at exactly N regardless of ties, unlike `RANK()`/`DENSE_RANK()`.

**Alternative shown (tie-aware version, for reference):**
```sql
WITH data AS (
    SELECT customer_id, SUM(amount) AS amt_s
    FROM orders
    GROUP BY customer_id
),
ranked AS (
    SELECT *, DENSE_RANK() OVER (ORDER BY amt_s DESC) AS rnk
    FROM data
)
SELECT customer_id, amt_s
FROM ranked
WHERE rnk <= 3;
```
**Takeaway:** `LIMIT N` and `ROW_NUMBER() ... WHERE rn <= N` are functionally equivalent — both give exact row counts, both blind to ties.

---

## Q6 — Per-Row Detail with Automatic Filter Resolution

**Question:** `page_views(view_id, user_id, page, view_time)`. For each user, find their 2nd most recently viewed page. Exclude users with fewer than 2 page views.

**Attempt 1 (solved a different problem — session paused here for a rest break):**
```sql
WITH data AS (
    SELECT user_id, Count(page) AS amt_s
    FROM page_views
group by user_id
Having Count(page) >2
),
ranked AS (
    SELECT *, ROW_number() OVER (ORDER BY amt_s DESC) AS rnk
    FROM data
)
SELECT user_id, amt_s
FROM ranked
WHERE rnk < 3;
```
**Notes/misses:**
- This ranked *users* by total view count and returned the top 2 users overall — a global top-N-users query, not a per-user "2nd most recent page" query.
- Tell-tale sign: no `page` or `view_time` in the output at all, despite the question being about "2nd most recent **page**."
- This was actually the same shape as Q3/Q4 (per-row detail via window function), not a new pattern — momentary miscategorization, likely fatigue-driven (session was paused here).

**Final (correct, after redo):**
```sql
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY view_time DESC) AS rn
    FROM page_views
)
SELECT user_id, page, view_time
FROM ranked
WHERE rn = 2;
```
**Key insight:** the "exclude users with fewer than 2 views" condition resolves itself automatically — if a user has only 1 view, no row for them ever reaches `rn = 2`, so `WHERE rn = 2` filters them out with no extra logic needed. Worth stating this reasoning explicitly in an interview.

---

## Q7 — Ranking by a Computed/Derived Metric

**Question:** `order_items(order_id, product_id, quantity, unit_price)`. For each order, find the product with the highest line total (`quantity × unit_price`). Return order_id, product_id, line total.

**Attempt 1 (incomplete syntax + wrong tool):**
```sql
with data as (
order_id, quantity*unit_price as linetotal from `order_items`)
select order_id,max(linetotal ) from data grouby order_id
```
**Notes/misses:**
- Missing `SELECT` keyword inside the CTE entirely — invalid SQL.
- Missing `product_id` in the CTE's select list, though it's needed in the final output.
- `groupby` typo (missing space) → `GROUP BY`.
- Conceptual issue (the important one): used `MAX(linetotal)` with `GROUP BY order_id` — but the question needs a **specific row** (which product) that happens to have the max value, not just the max number itself. `MAX()` collapses to a summary value and can't reliably carry `product_id` alongside it unless it's functionally tied — most databases would error or return an arbitrary product_id. This is the Q1-vs-Q4 distinction again: "keep a row" job vs. "collapse to a summary" job.

**Attempt 2 (missing pieces):**
```sql
with data as ( select
order_id,product_id, quantity*unit_price as linetotal from `order_items`)
select*  from (select order_id, product_id, Row_number() over (partition by order_id order by linetotal desc) )
where rn =1
```
**Notes/misses:**
- Correct pivot to window function based on the row-level-detail reasoning.
- Bugs: `ROW_NUMBER()` output never aliased (`AS rn` missing), so there was no `rn` column to filter on; inner subquery had no `FROM data` clause at all.
- `linetotal` missing from the final output despite the question requiring it.

**Attempt 3 (inner piece fixed):**
```sql
select order_id, product_id, Row_number() over (partition by order_id order by linetotal desc) as rn from data
```
**Notes/misses:** alias and `FROM data` both correctly added.

**Attempt 4 (typo relapse):**
```sql
with data as ( select order_id,product_id, quantity*unit_price as linetotal from order_items) 
select * from (select order_id, product_id, Row_number() over (partition by order_id order by line total desc)as rn from data  ) where rn =1
```
**Notes/misses:**
- Typo: `line total` (with a space) instead of `linetotal` — would be read as two separate identifiers or throw a syntax error.
- `linetotal` still missing from the inner `SELECT` list, so it wouldn't survive to the outer `SELECT *`.

**Final (correct):**
```sql
with data as ( select order_id,product_id, quantity*unit_price as linetotal from order_items)
select * from (select order_id, product_id, linetotal,  Row_number() over (partition by order_id order by linetotal desc)as rn from data ) where rn =1
```

---

## Q8 — Multi-Criteria Tiebreak + Correct Tie-Inclusive Ranking Function

**Question:** `products(product_id, category, revenue, launch_date)`. Top 3 products by revenue per category; tie-break by earlier launch date; if still tied after that, include all.

**Warm-up exchange:** confirmed compound `ORDER BY revenue DESC, launch_date ASC` — revenue as primary "top" criterion, ascending launch_date so earlier date wins tiebreak.

**Attempt 1 (garbled syntax):**
```sql
with data as ( select *, `DENSE_RANK`() over (partition by product_id over(order by ORDER BY revenue DESC, launch_date ASC )as rn  from products )

select * from data where  rn <4
```
**Notes/misses:**
- Correct high-level call: `DENSE_RANK()` chosen correctly for "ties should all survive, no gaps corrupting the top-3 cutoff."
- Conceptual bug: `PARTITION BY product_id` — since `product_id` is unique per row, every row becomes its own partition of size 1, so every product gets `rn = 1` and the filter does nothing. Needed `PARTITION BY category` (the actual grouping column), same role `department` played in Q1/Q2.
- Syntax: stray duplicated `over(` and `ORDER BY` keyword, unclosed parens, CTE never closed before the outer query, unneeded backticks around `` `DENSE_RANK` ``.
- Style note: `rn < 4` is mechanically equivalent to `rn <= 3` for integers, but `<=` more directly mirrors "top 3" without requiring the reader to do subtraction.

**Attempt 2 (one stray paren):**
```sql
with data as ( select *,`DENSE_RANK`() over (partition by category  ORDER BY revenue DESC, launch_date ASC )as rn from products ))
select * from data where rn <=3
```
**Notes/misses:** logic fully correct (right partition column, right compound ORDER BY, right ranking function, right filter) — just one extra stray `)` after `from products`.

**Final (correct):**
```sql
WITH data AS (
    SELECT *, DENSE_RANK() OVER (PARTITION BY category ORDER BY revenue DESC, launch_date ASC) AS rn
    FROM products
)
SELECT * FROM data WHERE rn <= 3;
```

---

## Session Summary — 10/10 Confirmed On:

1. Instantly recognizing when to use a ranking window function vs. `GROUP BY`/aggregates (locked in via Q4, Q4b, Q5).
2. Correctly choosing `ROW_NUMBER()` vs. `RANK()` vs. `DENSE_RANK()` based on how the question treats ties (Q1, Q2, Q8).
3. Ranking by a computed/derived metric rather than a raw column (Q7).
4. Chaining multi-column `ORDER BY` for tiebreak logic inside a ranking function (Q3, Q8).
5. Recognizing when a stated filter condition (e.g., "exclude users with fewer than 2 views") resolves itself automatically via the ranking filter, rather than needing extra logic (Q6).

**Note on the errors overall:** nearly all misses were either (a) momentary tool miscategorization — reaching for a window function when `GROUP BY` was right, or vice versa — which self-corrected with one nudge each time, or (b) typing-speed syntax noise (missing keywords, stray parens, dangling commas) rather than conceptual gaps. Both categories should continue to shrink with more reps in a real SQL editor with syntax highlighting.

**Suggested next patterns:** funnel conversion, cohort retention.

---
---

# Cohort Retention — Practice Session Notes

## The Core Idea

Group users by **when they started** (their cohort — usually signup week/month), then track what fraction of each cohort is still active N periods later. The mental model: **rows = cohort × period-since-signup**, **value = % of that cohort still active in that period.**

Output shape is almost always one of two things:
- **Long format** — one row per (cohort, period): `cohort_month | period_number | retention_rate`
- **Pivot/heatmap format** — one row per cohort, columns `m0, m1, m2, m3...`

---

## The Canonical Query, CTE by CTE

This is the query built across the session. Every CTE explained in full — what it computes, why it's a separate step, and what would break if you skipped it.

```sql
WITH cohorts AS (
    SELECT user_id, DATE_TRUNC('month', MIN(signup_date)) AS cohort_month
    FROM signups
    GROUP BY user_id
),
activity AS (
    SELECT
        c.user_id,
        c.cohort_month,
        DATEDIFF('month', c.cohort_month, DATE_TRUNC('month', a.login_date)) AS period_number
    FROM cohorts c
    JOIN logins a ON a.user_id = c.user_id
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
    FROM cohorts
    GROUP BY cohort_month
)
SELECT
    act.cohort_month,
    act.period_number,
    COUNT(DISTINCT act.user_id) AS active_users,
    cs.cohort_size,
    ROUND(COUNT(DISTINCT act.user_id) * 1.0 / cs.cohort_size, 3) AS retention_rate
FROM activity act
JOIN cohort_sizes cs ON cs.cohort_month = act.cohort_month
GROUP BY act.cohort_month, act.period_number, cs.cohort_size
ORDER BY act.cohort_month, act.period_number;
```

### CTE 1 — `cohorts`: assign every user to one cohort

```sql
SELECT user_id, DATE_TRUNC('month', MIN(signup_date)) AS cohort_month
FROM signups
GROUP BY user_id
```
**What it does:** produces exactly one row per user, labeled with the calendar month they first signed up.

**Why `MIN(signup_date)`:** guards against a user having more than one row in a raw signup/events table. If `signups` is guaranteed one-row-per-user already, `MIN()` is a no-op — but it's a cheap defensive habit, since assuming clean data is a common way this pattern breaks silently.

**Why `DATE_TRUNC('month', ...)`:** collapses a specific date (e.g. Jan 17) down to the first-of-month (Jan 1), so all signups within the same calendar month land in the same bucket. Without this, every user would effectively be their own "cohort" since exact dates rarely match.

**This CTE is the foundation everything else joins back to** — every later step needs to know "which cohort does this user belong to," and this is the only place that's computed.

### CTE 2 — `activity`: attach a period number to every login event

```sql
SELECT
    c.user_id,
    c.cohort_month,
    DATEDIFF('month', c.cohort_month, DATE_TRUNC('month', a.login_date)) AS period_number
FROM cohorts c
JOIN logins a ON a.user_id = c.user_id
```
**What it does:** for every login event, figures out how many calendar months after that user's cohort month the login happened.

**Why join `cohorts` to `logins` (not query `logins` alone):** `logins` only has raw dates — it has no idea what a user's cohort is. You need `cohort_month` (computed in CTE 1) sitting next to every login row before you can do the subtraction. This is the most common bug spot: joining `logins` against the raw `signups` table instead of against the `cohorts` CTE — the raw table never has `cohort_month` as a column, only the CTE does, since that's where it was calculated.

**Why `DATEDIFF('month', ..., DATE_TRUNC('month', a.login_date))` instead of raw day subtraction:** a user who signs up Jan 31 and logs in Feb 1 is "1 calendar period later," but that's only 1 day apart — dividing days by 30 would call this period 0, which is wrong. Truncating both sides to the month grain *first*, then diffing the truncated values, avoids month-boundary bugs entirely.

**One row per login event survives here** — this CTE is still row-level, not aggregated yet. That's intentional: you need per-event granularity before you can count distinct active users per period in the next step.

### CTE 3 — `cohort_sizes`: freeze the denominator

```sql
SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
FROM cohorts
GROUP BY cohort_month
```
**What it does:** for each cohort month, counts how many total users started in it — a single fixed number per cohort.

**Why this has to be its own CTE, computed from `cohorts` (not from `activity`):** the denominator for a retention rate is always "how many people were in this cohort *to begin with*," never "how many were active in this specific period." If you accidentally computed cohort size from the `activity` CTE instead, you'd be counting only users who logged in at least once — silently excluding anyone in the cohort who never returned, which inflates every retention rate. This is the single most common conceptual bug in cohort retention queries.

### Final `SELECT`: aggregate and compute the rate

```sql
SELECT
    act.cohort_month,
    act.period_number,
    COUNT(DISTINCT act.user_id) AS active_users,
    cs.cohort_size,
    ROUND(COUNT(DISTINCT act.user_id) * 1.0 / cs.cohort_size, 3) AS retention_rate
FROM activity act
JOIN cohort_sizes cs ON cs.cohort_month = act.cohort_month
GROUP BY act.cohort_month, act.period_number, cs.cohort_size
ORDER BY act.cohort_month, act.period_number;
```
**Why `COUNT(DISTINCT act.user_id)`, never `COUNT(*)`:** a single user can log in multiple times within the same period (e.g. daily logins in the same month). `COUNT(*)` would count every login event, not every unique person — this can push a "retention rate" above 100%, which is an instant tell to an interviewer that event granularity wasn't considered.

**Why the join to `cohort_sizes` happens last:** you need the row-level `activity` data grouped down to (cohort_month, period_number) first — the join just attaches the fixed denominator onto each of those grouped rows so the division can happen in the same `SELECT`.

**Why `* 1.0` before dividing:** forces floating-point division. In many SQL dialects, integer ÷ integer truncates to an integer (e.g., `3/7` becomes `0`, not `0.428`), silently producing all-zero retention rates. Multiplying by `1.0` first converts one side to a float, forcing correct decimal division.

---

## Q1 — Per-Row Mechanics (Cohort + Period Number)

**Question:** Given `signups(user_id, signup_date)` and `logins(user_id, login_date)`, return each user's cohort month and their login's period number — one row per login event, no aggregation yet.

**Attempt 1 (wrong schema entirely):**
```sql
WITH cohorts AS ( SELECT user_id, DATE_TRUNC('month', MIN(signup_date)) AS cohort_month FROM users GROUP BY user_id ), 
activity AS ( SELECT c.user_id, c.cohort_month, DATEDIFF('month', c.cohort_month, DATE_TRUNC('month', a.activity_date)) AS period_number FROM cohorts c JOIN logins a ON a.user_id = c.user_id )
```
**Notes/misses:**
- Used `FROM users` and `a.activity_date` — neither exists in this problem's schema. Both were copied from the teaching example instead of the actual given tables (`signups`, `logins` with `login_date`). Always re-read the exact schema given, don't autopilot from the last example seen.
- Missing the final `SELECT` entirely — the query as written returns nothing.

**Attempt 2 (join pointed at the wrong table):**
```sql
WITH cohorts AS ( SELECT user_id, DATE_TRUNC('month', MIN(signup_date)) AS cohort_month FROM `signups`GROUP BY user_id ),
activity AS ( SELECT c.user_id, c.cohort_month, DATEDIFF('month', c.cohort_month, DATE_TRUNC('month', a.activity_date)) AS period_number FROM `signups `c JOIN logins a ON a.user_id = c.user_id )
Select * from activity
```
**Notes/misses:**
- Critical bug: `FROM `signups `c JOIN logins a` — joined `logins` against the raw `signups` table instead of the `cohorts` CTE. `c.cohort_month` doesn't exist on `signups`; it only exists on the `cohorts` CTE, which was built correctly but never actually used downstream.
- `a.activity_date` typo persisted — should be `a.login_date`.
- Stray space inside backticks (`` `signups `c ``) would break identifier parsing regardless; backticks weren't needed at all here.

**Final (correct):**
```sql
WITH cohorts AS (
    SELECT user_id, DATE_TRUNC('month', MIN(signup_date)) AS cohort_month
    FROM signups
    GROUP BY user_id
),
activity AS (
    SELECT
        c.user_id,
        c.cohort_month,
        DATEDIFF('month', c.cohort_month, DATE_TRUNC('month', a.login_date)) AS period_number
    FROM cohorts c
    JOIN logins a ON a.user_id = c.user_id
)
SELECT * FROM activity;
```

---

## Q2 — Full Aggregation (Cohort Size + Retention Rate)

**Question:** Aggregate into the full retention table — for each `cohort_month` and `period_number`, return distinct active user count, cohort size, and retention rate.

**Attempt (one syntax bug):**
```sql
WITH cohorts AS (...), activity AS (...) cohort_sizes AS ( SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size FROM cohorts GROUP BY cohort_month ),

SELECT act.cohort_month, act.period_number, COUNT(DISTINCT act.user_id) AS active_users, cs.cohort_size, ROUND(COUNT(DISTINCT act.user_id) * 1.0 / cs.cohort_size, 3) AS retention_rate
FROM activity act JOIN cohort_sizes cs ON cs.cohort_month = act.cohort_month
GROUP BY act.cohort_month, act.period_number, cs.cohort_size ORDER BY act.cohort_month, act.period_number;
```
**Notes/misses:**
- Only bug: a trailing comma left after the last CTE (`cohort_sizes AS (...),`) right before the final `SELECT`. Once the last CTE in a `WITH` clause is declared, no comma should follow — a comma signals "another CTE follows," but the final `SELECT` isn't a CTE. Simple deletion fixes it.
- Everything else — join logic, `COUNT(DISTINCT ...)`, `ROUND(... * 1.0 / ...)`, `GROUP BY` columns — was correct on this attempt.

**Final:** matches the canonical query at the top of this section, in full.

---

## Q3 — The Definition Trap: Sticky (No-Gaps) Retention

**Question twist:** "I want 'sticky' retention — a user counts as retained in period N only if they were active in period N **and every period before it since signup**."

This is a genuinely different metric from the standard version, and it's a common way interviewers probe whether a candidate just memorized the skeleton or actually understands what the numbers mean.

### Building the running distinct-period count

```sql
sticky AS (
    SELECT
        user_id,
        cohort_month,
        period_number,
        COUNT(DISTINCT period_number) OVER (
            PARTITION BY user_id
            ORDER BY period_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS periods_active_so_far
    FROM activity
)
```

**Breaking down `COUNT(DISTINCT period_number) OVER (PARTITION BY user_id ORDER BY period_number ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` piece by piece:**

1. **`COUNT(DISTINCT period_number)`** — the aggregate itself: how many *distinct* period numbers appear. `DISTINCT` guards against a user having two logins in the same period being double-counted as "extra progress."
2. **`OVER (...)`** — turns this into a window function: computed per row rather than collapsed into one value per group, unlike a plain `GROUP BY` aggregate.
3. **`PARTITION BY user_id`** — resets the count separately for each user. Without this, you'd be counting distinct periods across *all* users combined, which is meaningless.
4. **`ORDER BY period_number ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`** — the "running" part: for the current row, look at every row from the very first (`UNBOUNDED PRECEDING`) through the current one, in period order. This gives a cumulative count *as of this point in the sequence*, not a total across the user's whole history.

**Reading it as a sentence:** *"For each user, going through their login periods in order, tell me — as of this row — how many distinct periods I've seen so far, starting from their very first period."*

**Concrete trace — user active in periods `[0, 1, 3]` (skipped period 2):**

| row (ordered by period_number) | period_number | periods_active_so_far |
|---|---|---|
| 1 | 0 | 1 (seen so far: {0}) |
| 2 | 1 | 2 (seen so far: {0,1}) |
| 3 | 3 | 3 (seen so far: {0,1,3}) |

At row 3, `period_number = 3` but `periods_active_so_far = 3` — these don't match (`period_number + 1` would need to be 4). That mismatch is exactly the gap-detection signal: had the user been active every period through 3, `periods_active_so_far` would read 4, matching `period_number + 1`.

### The filter condition

```sql
WHERE s.periods_active_so_far = s.period_number + 1
```
This keeps only rows where the user hit *every* period from 0 up through the current one, with no skips. Everything downstream of this (join to `cohort_sizes`, `COUNT(DISTINCT user_id)`, rate calculation) is structurally identical to Q2 — only the input rows being aggregated have changed.

### Full sticky retention query

```sql
WITH cohorts AS (
    SELECT user_id, DATE_TRUNC('month', MIN(signup_date)) AS cohort_month
    FROM signups
    GROUP BY user_id
),
activity AS (
    SELECT
        c.user_id,
        c.cohort_month,
        DATEDIFF('month', c.cohort_month, DATE_TRUNC('month', a.login_date)) AS period_number
    FROM cohorts c
    JOIN logins a ON a.user_id = c.user_id
),
sticky AS (
    SELECT
        user_id,
        cohort_month,
        period_number,
        COUNT(DISTINCT period_number) OVER (
            PARTITION BY user_id
            ORDER BY period_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS periods_active_so_far
    FROM activity
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(DISTINCT user_id) AS cohort_size
    FROM cohorts
    GROUP BY cohort_month
)
SELECT
    s.cohort_month,
    s.period_number,
    COUNT(DISTINCT s.user_id) AS sticky_retained_users,
    cs.cohort_size,
    ROUND(COUNT(DISTINCT s.user_id) * 1.0 / cs.cohort_size, 3) AS sticky_retention_rate
FROM sticky s
JOIN cohort_sizes cs ON cs.cohort_month = s.cohort_month
WHERE s.periods_active_so_far = s.period_number + 1
GROUP BY s.cohort_month, s.period_number, cs.cohort_size
ORDER BY s.cohort_month, s.period_number;
```

**Notes/misses along the way:**
- First instinct on how to detect "no gaps" was "self-join and row numbers" — not wrong in spirit (both are tools for comparing a row to others), but not the precise mechanism. The actual answer needed a **running distinct count via a window frame**, not a self-join.
- Needed the running-count logic broken down piece by piece (aggregate function → window → partition → frame) before it clicked — this is a good one to revisit cold in a few days to check it's retained, since it was built through guided steps rather than solved independently start to finish.

---

## Additional Retention Sub-Patterns (from external reference material — not yet drilled)

These showed up in a supplementary FAANG SQL prep document and are worth knowing exist, even without having practiced them live yet.

**1. Retention heatmap / pivot format** — same underlying data as the long-format query, but reshaped to one row per cohort with `m0, m1, m2, m3...` as columns:
```sql
MAX(CASE WHEN month_num = 0 THEN retention_rate END) AS m0,
MAX(CASE WHEN month_num = 1 THEN retention_rate END) AS m1,
...
```
`MAX()` here is just a mechanism to pivot — each `CASE WHEN` only produces a non-null value for one specific month_num per user-group, so `MAX()` (or `MIN()`, doesn't matter) picks out that single value per pivoted column after grouping. The phrase "one row per cohort" in a question is the signal to reach for this shape instead of long format.

**2. Three distinct "retention" definitions — terminology varies by company, so the safest move is to ask which is meant:**
   - **Exact retention** (what Q1/Q2 built): active in *precisely* period N.
   - **Rolling retention** (from the reference doc): active at *any point* within a range, e.g. `DATEDIFF BETWEEN 1 AND 7` — a simple ranged `CASE WHEN`, no window function required.
   - **Sticky retention** (what Q3 built): active in *every* period from 0 through N with no gaps — the strictest of the three, requiring the running-count window function.

**3. Churn segmentation by recency bucket** — a related but distinct query shape: bucket users by `days_inactive` into labels like Active/At-Risk/Churning/Churned via `CASE WHEN`, then compute each bucket's share of the total using `SUM(COUNT(*)) OVER ()` as the denominator.

**4. LTV by cohort** — reuses the exact same running-window mechanic as the sticky-retention `periods_active_so_far` calculation, but sums **revenue** instead of counting periods:
```sql
SUM(monthly_revenue) OVER (
    PARTITION BY cohort_date ORDER BY month_number
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
) AS cumulative_ltv
```
Good confirmation that `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is a general-purpose "running total" tool, not something specific to period-counting.

**5. Average retention curve across all cohorts** — collapses the cohort dimension entirely: `AVG(retention_rate) GROUP BY month_number` answers "what does retention typically look like by month N, averaged across all cohorts" — a different question from any single cohort's curve.

**6. Weekly cohorts** — cohort grain doesn't have to be monthly; truncating to week start (e.g., `DATE_SUB(date, INTERVAL DAYOFWEEK(date)-1 DAY)` in MySQL) is the same idea as `DATE_TRUNC('month', ...)` at a different grain. Exact function differs by dialect, but the concept — truncate to the start of the period, then diff truncated values — is identical to the month-based version already drilled.

**Not new — already fully covered in Q1–Q3:** the core cohort-base CTE, the join-then-diff-periods pattern, `COUNT(DISTINCT user_id)` for dedup, and the fixed cohort-size denominator.

---

## Cohort Retention — Key Traps to Remember

1. Cohort assignment must use `MIN(signup_date)` (or a dedicated one-row-per-user signup table) — never assign cohort from just any date column in a multi-row events table.
2. Always `COUNT(DISTINCT user_id)`, never `COUNT(*)` — multiple events per user per period will otherwise inflate rates past 100%.
3. Cohort size (the denominator) must come from the cohort-assignment step, frozen once — never recomputed from the activity/join step, or it silently excludes anyone who never returned.
4. Truncate both dates to the period grain *before* diffing them — never divide a raw day-count by 30, since that breaks at month boundaries.
5. Force float division (`* 1.0`) before dividing counts, or integer division silently truncates rates to 0.
6. "Retention" is ambiguous across exact / rolling / sticky definitions — asking the interviewer which one they mean is a legitimate, expected clarifying question, not a stalling tactic.
