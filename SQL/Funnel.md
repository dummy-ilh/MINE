# FUNNEL
Say you have a basic events table:

```sql
-- events(user_id, event_name, event_time)
```

## Variant 1: Conditional aggregation (the easy, go-to way)

Flag each user with 1/0 for whether they hit each stage, then sum it up. This is the simplest and fastest to write for most funnels.

```sql
SELECT
  COUNT(DISTINCT CASE WHEN event_name = 'visit'    THEN user_id END) AS visited,
  COUNT(DISTINCT CASE WHEN event_name = 'signup'   THEN user_id END) AS signed_up,
  COUNT(DISTINCT CASE WHEN event_name = 'add_cart' THEN user_id END) AS added_cart,
  COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN user_id END) AS purchased
FROM events
WHERE event_time BETWEEN '2026-06-01' AND '2026-06-30';
```

Nice because it's one pass over the data, easy to read, and easy to add stages to. The catch: this counts *anyone who ever did X*, not necessarily in order. If someone purchased without visiting in your date window (e.g. they visited last month), they'd still count as "purchased" here. Fine for a lot of use cases, but not strictly sequential.

## Variant 2: Self-joins (enforces order)

Join each stage to the next, requiring the later event to happen after the earlier one, per user. This gives you a true "did A then B" funnel.

```sql
SELECT
  COUNT(DISTINCT v.user_id) AS visited,
  COUNT(DISTINCT s.user_id) AS signed_up,
  COUNT(DISTINCT c.user_id) AS added_cart,
  COUNT(DISTINCT p.user_id) AS purchased
FROM (SELECT DISTINCT user_id, MIN(event_time) AS t FROM events WHERE event_name = 'visit' GROUP BY 1) v
LEFT JOIN (SELECT DISTINCT user_id, MIN(event_time) AS t FROM events WHERE event_name = 'signup') s
  ON s.user_id = v.user_id AND s.t >= v.t
LEFT JOIN (SELECT DISTINCT user_id, MIN(event_time) AS t FROM events WHERE event_name = 'add_cart') c
  ON c.user_id = s.user_id AND c.t >= s.t
LEFT JOIN (SELECT DISTINCT user_id, MIN(event_time) AS t FROM events WHERE event_name = 'purchase') p
  ON p.user_id = c.user_id AND p.t >= c.t;
```

More correct, but verbose — every extra stage means another join. Gets messy past 4-5 stages.

## Variant 3: Window functions (best for strict, ordered funnels)

Use `LAG()` per user ordered by time, then check that each stage immediately follows a valid prior stage. This is the cleanest way to handle "must happen in this order" without stacking joins.

```sql
WITH ranked AS (
  SELECT
    user_id,
    event_name,
    event_time,
    ROW_NUMBER() OVER (PARTITION BY user_id, event_name ORDER BY event_time) AS rn
  FROM events
),
first_touch AS (
  SELECT user_id, event_name, event_time
  FROM ranked
  WHERE rn = 1  -- first time each user did each event
),
pivoted AS (
  SELECT
    user_id,
    MAX(CASE WHEN event_name = 'visit'    THEN event_time END) AS t_visit,
    MAX(CASE WHEN event_name = 'signup'   THEN event_time END) AS t_signup,
    MAX(CASE WHEN event_name = 'add_cart' THEN event_time END) AS t_cart,
    MAX(CASE WHEN event_name = 'purchase' THEN event_time END) AS t_purchase
  FROM first_touch
  GROUP BY user_id
)
SELECT
  COUNT(*) FILTER (WHERE t_visit IS NOT NULL) AS visited,
  COUNT(*) FILTER (WHERE t_signup   >= t_visit) AS signed_up,
  COUNT(*) FILTER (WHERE t_cart     >= t_signup) AS added_cart,
  COUNT(*) FILTER (WHERE t_purchase >= t_cart) AS purchased
FROM pivoted;
```

(`FILTER` is Postgres/Snowflake syntax — swap for `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` on MySQL/BigQuery.)

This "pivot to one row per user" pattern is the real workhorse of funnel SQL — it's a variant of #1 but with per-stage timestamps instead of just flags, so you can enforce ordering without joins.

## Which one to actually reach for

- **Quick gut-check funnel, order doesn't matter much** → Variant 1 (conditional aggregation). This is the "easy way" — 90% of the time this is all you need.
- **Need strict sequential order, few stages** → Variant 2 (self-join) is intuitive but scales badly.
- **Need strict order, more than 3-4 stages, or want conversion time between stages too** → Variant 3 (pivot + window functions). This scales the best and is what most analytics teams standardize on.

One more thing worth knowing: once you have the per-user pivoted table (Variant 3's `pivoted` CTE), computing conversion rates and time-between-stages is trivial:

```sql
SELECT
  ROUND(100.0 * COUNT(*) FILTER (WHERE t_signup >= t_visit) / COUNT(*) FILTER (WHERE t_visit IS NOT NULL), 1) AS visit_to_signup_pct,
  AVG(t_signup - t_visit) FILTER (WHERE t_signup >= t_visit) AS avg_time_to_signup
FROM pivoted;
```
---
COUNT(DISTINCT CASE WHEN event_name = 'visit' THEN user_id END)

break and explain
**1. The `CASE WHEN` part**
```sql
CASE WHEN event_name = 'visit' THEN user_id END
```
This runs row by row. For every row in the table:
- If that row's `event_name` is `'visit'`, this expression returns the `user_id` from that row.
- If it's not `'visit'` (say it's `'signup'` or `'purchase'`), there's no `ELSE`, so it implicitly returns `NULL`.

So conceptually, it turns your `user_id` column into a "masked" version — only showing the user_id when the row was a visit event, and hiding it (as NULL) otherwise.

Example on 4 rows:

| user_id | event_name | CASE result |
|---|---|---|
| 1 | visit | 1 |
| 1 | signup | NULL |
| 2 | visit | 2 |
| 3 | purchase | NULL |

**2. Wrapping it in `COUNT(...)`**

`COUNT()` counts non-NULL values. Since the CASE expression nulled out anything that wasn't a visit, plain `COUNT(CASE WHEN ...)` would count *rows* where a visit happened — so if user 1 visited 3 times, that's 3 toward the count.

**3. Adding `DISTINCT`**

`COUNT(DISTINCT CASE WHEN ...)` first deduplicates the values that come out of the CASE expression, *then* counts them. So if user 1 visited 3 times, you still only count them once. NULLs are automatically ignored by `COUNT`, so all the non-visit rows just drop out of consideration entirely.

**Put together:** this expression answers "how many unique users had at least one row where `event_name = 'visit'`?" — which is exactly what you want for a funnel stage: unique users reaching that stage, not raw event counts.

That's also why you can stack several of these side by side in one `SELECT` — each one independently filters-and-counts its own stage from the *same* set of rows, in a single pass over the table, which is what makes conditional aggregation efficient.

---
(SELECT DISTINCT user_id, MIN(event_time) AS t FROM events WHERE event_name = 'visit' GROUP BY 1)


Good catch to look at this one closely — it actually has a subtle redundancy worth pointing out.

**Breaking it down:**

```sql
SELECT DISTINCT user_id, MIN(event_time) AS t
FROM events
WHERE event_name = 'visit'
GROUP BY 1
```

**Step by step:**

1. `WHERE event_name = 'visit'` — first, filter the whole table down to only rows where the event was a visit. Every other event type is thrown out before anything else happens.

2. `GROUP BY 1` — group the remaining rows by column 1 in the SELECT list, which is `user_id`. (`GROUP BY 1` is positional shorthand for `GROUP BY user_id` — some people like it, some find it fragile since reordering the SELECT list silently changes what you're grouping by.)

3. `MIN(event_time) AS t` — within each user's group of visit-rows, take the earliest timestamp. So if user 1 visited 5 times, this collapses those 5 rows into one, keeping only the first visit time.

4. `SELECT DISTINCT user_id, ...` — this is the redundant part. Once you `GROUP BY user_id`, each group already produces exactly one row per user_id — so `user_id` is already unique in the output. Adding `DISTINCT` here does nothing except potentially cost a wasted dedup pass, since there's nothing left to deduplicate.

**What it actually produces:** one row per user, with their earliest visit timestamp.

| user_id | t |
|---|---|
| 1 | 2026-06-01 09:00 |
| 2 | 2026-06-03 14:22 |

**Cleaner equivalent** (same result, no redundant DISTINCT):
```sql
SELECT user_id, MIN(event_time) AS t
FROM events
WHERE event_name = 'visit'
GROUP BY user_id
```

This is why it's used in the self-join funnel — you need exactly one row per user per stage (their first time reaching it) so that joining stage-to-stage doesn't accidentally multiply rows if someone triggered an event multiple times.

---
MAX(CASE WHEN event_name = 'visit' THEN event_time END) AS t_visit,

Same CASE-masking trick as before, but now pulling out a timestamp instead of a user_id, and using `MAX` instead of `COUNT(DISTINCT ...)`.

**The CASE part:**
```sql
CASE WHEN event_name = 'visit' THEN event_time END
```
Row by row: if this row's event is `'visit'`, return its `event_time`. Otherwise (no `ELSE`), return `NULL`.

So on a set of rows for one user:

| event_name | event_time | CASE result |
|---|---|---|
| visit | 09:00 | 09:00 |
| signup | 09:15 | NULL |
| visit | 09:40 | 09:40 |
| purchase | 10:00 | NULL |

**Why `MAX` here, and why it matters:**

This line lives inside a `GROUP BY user_id` (from the `pivoted` CTE in the earlier query), aggregating many rows down to one row per user. `MAX` picks the largest non-NULL value in that group — so out of `09:00`, `NULL`, `09:40`, `NULL`, it returns `09:40`: the *latest* visit time for that user.

That's a deliberate choice, and worth flagging: earlier in the conversation we used `MIN(event_time)` to get a user's *first* visit. Here it's `MAX`. Practically it usually doesn't change much for a "visit" stage (people don't visit that many times before signing up), but it does mean this version answers "what's the latest time this user did X" rather than "the first time." If you want strictly first-touch timestamps per stage (generally the safer choice for funnel ordering), swap `MAX` → `MIN` here to match the self-join version's logic.

**Why `MAX` (or `MIN`) instead of `COUNT`:** because we're aggregating a *timestamp*, not counting occurrences. `MAX`/`MIN` on a column of mostly-NULLs-plus-one-real-value effectively acts like "pull out the one value that matters, or NULL if this user never did this stage" — which is exactly what you want when pivoting from long (one row per event) to wide (one row per user, one column per stage).

Good questions — both get at subtle behavior.

## Why `MAX`?

Here's the thing: by the time we get to the `pivoted` CTE, `first_touch` already has **at most one row per (user_id, event_name)** — we filtered to `rn = 1` specifically to guarantee that. So for any given user, there's only ever one possible non-NULL value the `CASE WHEN` could produce for `t_visit`.

So why wrap it in `MAX` at all if there's nothing to compare?

Because of `GROUP BY user_id`. SQL requires every column in the SELECT list to either be in the `GROUP BY` or wrapped in an aggregate function — you can't just say `SELECT user_id, event_time FROM first_touch GROUP BY user_id`, because SQL doesn't know which `event_time` out of the group to show you. `MAX()` (or `MIN()`, or even `SUM()` if it were numeric) is just the mechanism to satisfy that rule and collapse the group into one row.

Since each group has only one non-NULL candidate value (from the CASE) plus a bunch of NULLs, `MAX` and `MIN` produce the *identical* result here — `MAX(9:00, NULL, NULL) = MIN(9:00, NULL, NULL) = 9:00`. It's a completely arbitrary pick between the two; `MAX` isn't doing anything semantically special in this specific query. (It mattered more back in the version *without* the `rn = 1` filter, where a user could have multiple visit rows and you had to actively choose "first" vs "latest.")

## What happens if a column is NULL?

This is where the real logic lives, in the final SELECT's comparisons:

```sql
COUNT(*) FILTER (WHERE t_signup >= t_visit) AS signed_up
```

In SQL, **any comparison involving NULL evaluates to NULL** (not true, not false) — and `WHERE` only keeps rows where the condition is `TRUE`. `NULL` is treated as "not true," so the row gets excluded.

Walk through the cases for a user:

| t_visit | t_signup | `t_signup >= t_visit` | counted in `signed_up`? |
|---|---|---|---|
| 09:00 | 09:15 | TRUE | yes |
| 09:00 | NULL | NULL (never signed up) | no |
| NULL | 09:15 | NULL (signed up but no visit record — odd data) | no |
| NULL | NULL | NULL | no |

So the NULL handling does exactly the right thing for a funnel without you having to write extra `IS NOT NULL` checks: anyone missing the *later* stage's timestamp automatically drops out, because comparing anything to NULL just goes to NULL and fails the filter. That one line is quietly doing both "did they reach this stage" and "did it happen in the right order" at once.

Here's the same funnel, but broken into one CTE per stage — each stage explicit, with its own timestamp column, and explicit order checks between adjacent stages (`t1`, `t2`, `t3`, `t4`).

```sql
WITH
-- Stage 1: everyone who visited, and when they first did
visit AS (
  SELECT user_id, MIN(event_time) AS t1
  FROM events
  WHERE event_name = 'visit'
    AND event_time BETWEEN '2026-06-01' AND '2026-06-30'
  GROUP BY user_id
),

-- Stage 2: everyone who signed up, and when
signup AS (
  SELECT user_id, MIN(event_time) AS t2
  FROM events
  WHERE event_name = 'signup'
    AND event_time BETWEEN '2026-06-01' AND '2026-06-30'
  GROUP BY user_id
),

-- Stage 3: everyone who added to cart, and when
add_cart AS (
  SELECT user_id, MIN(event_time) AS t3
  FROM events
  WHERE event_name = 'add_cart'
    AND event_time BETWEEN '2026-06-01' AND '2026-06-30'
  GROUP BY user_id
),

-- Stage 4: everyone who purchased, and when
purchase AS (
  SELECT user_id, MIN(event_time) AS t4
  FROM events
  WHERE event_name = 'purchase'
    AND event_time BETWEEN '2026-06-01' AND '2026-06-30'
  GROUP BY user_id
),

-- Stitch all four stages together per user, keeping everyone who at least visited
funnel AS (
  SELECT
    v.user_id,
    v.t1,
    s.t2,
    c.t3,
    p.t4
  FROM visit v
  LEFT JOIN signup   s ON s.user_id = v.user_id
  LEFT JOIN add_cart c ON c.user_id = v.user_id
  LEFT JOIN purchase p ON p.user_id = v.user_id
)

-- Final counts, enforcing that each stage happened on/after the previous one
SELECT
  COUNT(*) FILTER (WHERE t1 IS NOT NULL)      AS visited,
  COUNT(*) FILTER (WHERE t2 >= t1)            AS signed_up,
  COUNT(*) FILTER (WHERE t3 >= t2)            AS added_cart,
  COUNT(*) FILTER (WHERE t4 >= t3)            AS purchased
FROM funnel;
```

**What each piece is doing, and why it's split this way:**

- `visit`, `signup`, `add_cart`, `purchase` — four separate, parallel CTEs, one per funnel stage. Each is independently: filter to that event type, filter to the date range, group by user, and grab their earliest (`MIN`) timestamp for that stage. Naming them `t1`–`t4` instead of `t_visit`/`t_signup`/etc. makes the ordering explicit and visually obvious in the final checks.

- `funnel` — joins all four stage-tables together on `user_id`, anchored to `visit` (`LEFT JOIN` from `v`). Anyone who visited gets a row here, even if they never signed up, added to cart, or purchased — those columns just come back `NULL` via the `LEFT JOIN`.

- Final `SELECT` — same NULL-comparison logic as before: `t2 >= t1` is `NULL` (and gets excluded) if the user never reached that stage, and enforces chronological order when they did.

**How this differs from your original 4-line query:**

Your original just asks "how many distinct users ever had each event type" — independent counts, no ordering, no per-user linkage between stages. This version asks "how many users went through the stages *in order*" — a user who added to cart before signing up (weird, but possible in messy data) would count toward `added_cart` in the original but get excluded here, since `t3 >= t2` would fail.

Here are reusable templates you can adapt by just swapping stage names and table/column names.

## Template 1: Generic N-stage conditional aggregation (quick, unordered)

```sql
SELECT
  COUNT(DISTINCT CASE WHEN event_name = 'STAGE_1' THEN user_id END) AS stage_1,
  COUNT(DISTINCT CASE WHEN event_name = 'STAGE_2' THEN user_id END) AS stage_2,
  COUNT(DISTINCT CASE WHEN event_name = 'STAGE_3' THEN user_id END) AS stage_3,
  COUNT(DISTINCT CASE WHEN event_name = 'STAGE_4' THEN user_id END) AS stage_4
FROM events
WHERE event_time BETWEEN 'START_DATE' AND 'END_DATE';
```
Use when: quick sanity check, order doesn't matter, want it fast.

## Template 2: Ordered funnel with per-stage CTEs (readable, extensible)

```sql
WITH
stage_1 AS (
  SELECT user_id, MIN(event_time) AS t1
  FROM events
  WHERE event_name = 'STAGE_1_NAME'
    AND event_time BETWEEN 'START_DATE' AND 'END_DATE'
  GROUP BY user_id
),
stage_2 AS (
  SELECT user_id, MIN(event_time) AS t2
  FROM events
  WHERE event_name = 'STAGE_2_NAME'
    AND event_time BETWEEN 'START_DATE' AND 'END_DATE'
  GROUP BY user_id
),
stage_3 AS (
  SELECT user_id, MIN(event_time) AS t3
  FROM events
  WHERE event_name = 'STAGE_3_NAME'
    AND event_time BETWEEN 'START_DATE' AND 'END_DATE'
  GROUP BY user_id
),
funnel AS (
  SELECT s1.user_id, s1.t1, s2.t2, s3.t3
  FROM stage_1 s1
  LEFT JOIN stage_2 s2 ON s2.user_id = s1.user_id
  LEFT JOIN stage_3 s3 ON s3.user_id = s1.user_id
)
SELECT
  COUNT(*) FILTER (WHERE t1 IS NOT NULL) AS stage_1_count,
  COUNT(*) FILTER (WHERE t2 >= t1)       AS stage_2_count,
  COUNT(*) FILTER (WHERE t3 >= t2)       AS stage_3_count
FROM funnel;
```
Use when: you need to add/remove stages easily and want clear, debuggable steps. Copy the CTE block per stage.

## Template 3: Time-boxed conversion (with conversion window)

```sql
WITH
stage_1 AS (
  SELECT user_id, MIN(event_time) AS t1
  FROM events WHERE event_name = 'STAGE_1_NAME'
  GROUP BY user_id
),
stage_2 AS (
  SELECT user_id, MIN(event_time) AS t2
  FROM events WHERE event_name = 'STAGE_2_NAME'
  GROUP BY user_id
),
funnel AS (
  SELECT s1.user_id, s1.t1, s2.t2
  FROM stage_1 s1
  LEFT JOIN stage_2 s2 ON s2.user_id = s1.user_id
)
SELECT
  COUNT(*) FILTER (WHERE t1 IS NOT NULL) AS stage_1_count,
  COUNT(*) FILTER (
    WHERE t2 >= t1 AND t2 <= t1 + INTERVAL 'N days'
  ) AS stage_2_count_within_window
FROM funnel;
```
Use when: conversions only count within a meaningful time window (e.g. "signed up within 7 days of visiting"). Swap `INTERVAL 'N days'` per your business logic.

## Template 4: Pivot + window function (scales to many stages, most standard)

```sql
WITH ranked AS (
  SELECT user_id, event_name, event_time,
         ROW_NUMBER() OVER (PARTITION BY user_id, event_name ORDER BY event_time) AS rn
  FROM events
  WHERE event_time BETWEEN 'START_DATE' AND 'END_DATE'
),
first_touch AS (
  SELECT user_id, event_name, event_time
  FROM ranked WHERE rn = 1
),
pivoted AS (
  SELECT
    user_id,
    MAX(CASE WHEN event_name = 'STAGE_1' THEN event_time END) AS t1,
    MAX(CASE WHEN event_name = 'STAGE_2' THEN event_time END) AS t2,
    MAX(CASE WHEN event_name = 'STAGE_3' THEN event_time END) AS t3,
    MAX(CASE WHEN event_name = 'STAGE_4' THEN event_time END) AS t4
  FROM first_touch
  GROUP BY user_id
)
SELECT
  COUNT(*) FILTER (WHERE t1 IS NOT NULL) AS stage_1,
  COUNT(*) FILTER (WHERE t2 >= t1)       AS stage_2,
  COUNT(*) FILTER (WHERE t3 >= t2)       AS stage_3,
  COUNT(*) FILTER (WHERE t4 >= t3)       AS stage_4
FROM pivoted;
```
Use when: 4+ stages, want one clean scan instead of stacking joins. This is the one most teams standardize on.

## Template 5: Conversion rates + time-to-convert (add-on to Template 4)

```sql
SELECT
  ROUND(100.0 * COUNT(*) FILTER (WHERE t2 >= t1) 
        / NULLIF(COUNT(*) FILTER (WHERE t1 IS NOT NULL), 0), 1) AS stage1_to_stage2_pct,
  AVG(t2 - t1) FILTER (WHERE t2 >= t1) AS avg_time_stage1_to_stage2
FROM pivoted;
```
Use when: you want % conversion between adjacent stages and average time-to-convert, not just raw counts. `NULLIF(..., 0)` guards against divide-by-zero if a stage has no users.

---

## 1. Conversion Funnel Basics
A funnel tracks users moving through sequential steps. The key metric is drop-off rate at each step.
```sql
-- Table: events(user_id, event_type, event_date)
-- Funnel: signup → onboarding → first_purchase → repeat_purchase

WITH funnel AS (
  SELECT
    COUNT(DISTINCT CASE WHEN event_type = 'signup'
          THEN user_id END)           AS step1_signup,
    COUNT(DISTINCT CASE WHEN event_type = 'onboarding'
          THEN user_id END)           AS step2_onboarding,
    COUNT(DISTINCT CASE WHEN event_type = 'first_purchase'
          THEN user_id END)           AS step3_purchase,
    COUNT(DISTINCT CASE WHEN event_type = 'repeat_purchase'
          THEN user_id END)           AS step4_repeat
  FROM events
)
SELECT
  step1_signup,
  step2_onboarding,
  step3_purchase,
  step4_repeat,
  ROUND(step2_onboarding  * 100.0 / step1_signup,    2) AS signup_to_onboard_pct,
  ROUND(step3_purchase    * 100.0 / step2_onboarding, 2) AS onboard_to_purchase_pct,
  ROUND(step4_repeat      * 100.0 / step3_purchase,   2) AS purchase_to_repeat_pct,
  ROUND(step4_repeat      * 100.0 / step1_signup,     2) AS overall_conversion_pct
FROM funnel;
```

---

## 2. Ordered Funnel (Steps Must Be Sequential)
Real funnels require steps to happen in order. This is harder.
```sql
-- User must: view_product → add_to_cart → checkout → purchase
-- Each step must come AFTER the previous one

WITH step1 AS (
  SELECT DISTINCT user_id, MIN(event_date) AS s1_date
  FROM events WHERE event_type = 'view_product'
  GROUP BY user_id
),
step2 AS (
  SELECT DISTINCT e.user_id, MIN(e.event_date) AS s2_date
  FROM events e
  JOIN step1 s ON e.user_id = s.user_id
  WHERE e.event_type = 'add_to_cart'
    AND e.event_date > s.s1_date   -- must happen AFTER step 1
  GROUP BY e.user_id
),
step3 AS (
  SELECT DISTINCT e.user_id, MIN(e.event_date) AS s3_date
  FROM events e
  JOIN step2 s ON e.user_id = s.user_id
  WHERE e.event_type = 'checkout'
    AND e.event_date > s.s2_date
  GROUP BY e.user_id
),
step4 AS (
  SELECT DISTINCT e.user_id, MIN(e.event_date) AS s4_date
  FROM events e
  JOIN step3 s ON e.user_id = s.user_id
  WHERE e.event_type = 'purchase'
    AND e.event_date > s.s3_date
  GROUP BY e.user_id
)
SELECT
  COUNT(DISTINCT step1.user_id) AS viewed,
  COUNT(DISTINCT step2.user_id) AS added_to_cart,
  COUNT(DISTINCT step3.user_id) AS checked_out,
  COUNT(DISTINCT step4.user_id) AS purchased,
  ROUND(COUNT(DISTINCT step2.user_id) * 100.0 /
        COUNT(DISTINCT step1.user_id), 2) AS view_to_cart_pct,
  ROUND(COUNT(DISTINCT step4.user_id) * 100.0 /
        COUNT(DISTINCT step1.user_id), 2) AS overall_pct
FROM step1
LEFT JOIN step2 USING (user_id)
LEFT JOIN step3 USING (user_id)
LEFT JOIN step4 USING (user_id);
```

> 💡 Each CTE filters `event_date > prev step date` — enforces ordering.

---
