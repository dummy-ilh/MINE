# Conversion Funnel Analysis — End to End
### Built slowly, stage by stage — with "why" and "why not" at every step

---

## What Is a Conversion Funnel?

A conversion funnel measures how many users make it through a sequence of ordered steps — e.g., **Viewed Product → Added to Cart → Started Checkout → Purchased**. It answers: **"Of everyone who did Step 1, what fraction made it to Step 2? Step 3? All the way to the end?"**

**Why build it at all?** Without a funnel, you only see the final conversion rate (e.g., "2% of visitors buy") — a single number that hides *where* people are dropping off. A funnel splits that one number into a story: maybe 80% view a product, but only 20% of those add to cart — telling you exactly which step to fix first.

**Why not just look at total purchases vs. total visitors?** Because that ratio conflates every possible failure point into one metric. A 2% overall conversion rate could mean "people love the cart page but hate checkout" or "people barely make it past the homepage" — completely different problems requiring completely different fixes. The funnel disaggregates the single number into diagnosable stages.

---

## The Sample Data

We'll use a simple `events` table throughout, same style as a cohort analysis but now the events themselves form an ordered journey.

```sql
SELECT * FROM events LIMIT 12;
```

| user_id | event_time          | event_type       |
|---------|---------------------|------------------|
| 201     | 2024-05-01 09:02:00 | view_product     |
| 201     | 2024-05-01 09:05:00 | add_to_cart      |
| 201     | 2024-05-01 09:07:00 | start_checkout   |
| 201     | 2024-05-01 09:10:00 | purchase         |
| 202     | 2024-05-01 10:15:00 | view_product     |
| 202     | 2024-05-01 10:16:00 | add_to_cart      |
| 203     | 2024-05-02 08:00:00 | view_product     |
| 204     | 2024-05-02 11:20:00 | view_product     |
| 204     | 2024-05-02 11:25:00 | add_to_cart      |
| 204     | 2024-05-02 11:40:00 | start_checkout   |
| 205     | 2024-05-02 14:00:00 | view_product     |
| 205     | 2024-05-02 14:02:00 | add_to_cart      |

**Why this shape of table?** A "long" event log (one row per action) rather than a "wide" table (one row per user with columns for each step) is the natural format raw product analytics data arrives in — every stage of this analysis is really just different ways of reshaping this same long table. **Why not store it wide from the start?** Because a user might repeat a step (view a product twice), skip a step, or do steps out of order — a long log captures all of that faithfully, while a wide table forces premature assumptions about exactly one row per user.

---

## Stage 1 — Define the Funnel Steps (In Order)

Before writing any SQL, we fix the ordered list of steps we're measuring.

**Variables:**
- `step_order` — A manually defined ranking (1, 2, 3, 4) of which event types count as which funnel stage

```sql
-- Stage 1: Define step ordering as a lookup (often just a CASE statement or small mapping table)
SELECT
    event_type,
    CASE event_type
        WHEN 'view_product'    THEN 1
        WHEN 'add_to_cart'     THEN 2
        WHEN 'start_checkout'  THEN 3
        WHEN 'purchase'        THEN 4
    END AS step_order
FROM events
GROUP BY event_type;
```

**Output:**

| event_type      | step_order |
|-----------------|------------|
| view_product    | 1          |
| add_to_cart     | 2          |
| start_checkout  | 3          |
| purchase        | 4          |

**Why define this explicitly instead of trusting timestamps alone?** Timestamps tell you *when* something happened, not which conceptual funnel stage it belongs to — you need a human-defined mapping from raw event names to funnel position, because "the funnel" is a business definition, not something the data announces on its own. **Why not just count distinct event types and assume they're already in the right order?** Event names in a database are rarely stored in funnel order (e.g., alphabetically `add_to_cart` comes before `purchase`), so relying on implicit ordering will silently miscount the funnel — always pin the order down explicitly.

---

## Stage 2 — Find Each User's Furthest Step Reached

For each user, we need to know the *highest* step_order they ever reached — this tells us how far they got, regardless of how many times they repeated earlier steps.

**Variables:**
- `max_step` — The highest step_order value seen for that user, across all their events

```sql
-- Stage 2: Find the furthest step each user reached
WITH step_events AS (
    SELECT
        user_id,
        event_type,
        CASE event_type
            WHEN 'view_product'    THEN 1
            WHEN 'add_to_cart'     THEN 2
            WHEN 'start_checkout'  THEN 3
            WHEN 'purchase'        THEN 4
        END AS step_order
    FROM events
)
SELECT
    user_id,
    MAX(step_order) AS max_step        -- furthest stage reached
FROM step_events
GROUP BY user_id;
```

**Output:**

| user_id | max_step |
|---------|----------|
| 201     | 4        |
| 202     | 2        |
| 203     | 1        |
| 204     | 3        |
| 205     | 2        |

**Why `MAX` and not `COUNT` of events?** A user could view the product 5 times and never add to cart — `COUNT` would overstate their progress, while `MAX(step_order)` correctly captures that they only ever reached Step 1, no matter how many times they repeated it. **Why not just check "did they do the very last step" (a boolean)?** Because that only tells you pass/fail on the *entire* funnel — it throws away exactly the information a funnel exists to show: *which intermediate step* they stalled at.

---

## Stage 3 — A Critical Assumption: Strict Order vs. "Ever Reached"

Before continuing, we must decide: does a user only count as "reaching Step 3" if they did Steps 1, 2, and 3 **in that order**, or does it just matter that they eventually did the Step 3 *event*, regardless of order?

**Why does this matter?** Real user behavior is messy — someone might add-to-cart, remove it, browse more, then start checkout without technically "viewing the product" again in that session. If your funnel logic silently assumes perfect order, you can undercount real conversions.

**Two approaches, with tradeoffs:**

```sql
-- Approach A: "Ever reached" (simpler, used in Stage 2 above)
--   Just take MAX(step_order) per user — ignores sequencing entirely.
--   Why use it: simple, fast, matches most standard funnel dashboards (Amplitude/Mixpanel default).
--   Why not: can overcount — a user who added to cart BEFORE ever viewing a product
--   (e.g., re-ordering from history) still counts as reaching Step 2.

-- Approach B: Strict in-order sequence (stricter, more correct for "true" funnel behavior)
SELECT
    user_id,
    event_type,
    event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event_time
FROM events;
--   Why use it: guarantees the steps happened in the expected sequence for each user.
--   Why not always: significantly more complex SQL, and for most product-analytics
--   use cases the simpler "ever reached" approach is the industry standard because
--   it's more forgiving of realistic, slightly non-linear user journeys.
```

**Decision for this walkthrough:** We proceed with Approach A ("ever reached") for the rest of this document, since it's what Stage 2 already computed and it's the standard behind most funnel tools — but a senior analyst should always state out loud which assumption they're making, since it changes the numbers.

---

## Stage 4 — Count Users Reaching Each Step (The Raw Funnel Counts)

Now we count, for each step, how many users reached *at least* that far.

**Variables:**
- `users_reaching_step` — COUNT DISTINCT users whose `max_step >= step_order`

```sql
-- Stage 4: Count users reaching each step or further
WITH user_progress AS (
    -- Stage 2 result
    SELECT user_id, MAX(step_order) AS max_step
    FROM step_events
    GROUP BY user_id
),
funnel_steps AS (
    SELECT 1 AS step_order, 'view_product' AS step_name
    UNION ALL SELECT 2, 'add_to_cart'
    UNION ALL SELECT 3, 'start_checkout'
    UNION ALL SELECT 4, 'purchase'
)
SELECT
    f.step_order,
    f.step_name,
    COUNT(DISTINCT CASE WHEN u.max_step >= f.step_order THEN u.user_id END) AS users_reaching_step
FROM funnel_steps f
CROSS JOIN user_progress u
GROUP BY f.step_order, f.step_name
ORDER BY f.step_order;
```

**Output:**

| step_order | step_name        | users_reaching_step |
|------------|------------------|----------------------|
| 1          | view_product     | 5                    |
| 2          | add_to_cart      | 4                    |
| 3          | start_checkout   | 2                    |
| 4          | purchase         | 1                    |

**Why `CROSS JOIN` here instead of a simple `GROUP BY max_step`?** A plain `GROUP BY max_step` only shows you how many people's *furthest* step was exactly N — it wouldn't tell you that everyone who reached Step 3 also, by definition, passed through Step 1 and 2. The `CROSS JOIN` + `>=` comparison deliberately generates the *cumulative* "reached this far or beyond" count, which is what a funnel chart actually shows (each bar is normally smaller than or equal to the one before it, never larger). **Why not just filter `WHERE max_step = f.step_order` and then manually sum going backward?** You could, but it's more error-prone to hand-roll a running total in application code than to let `>=` do the cumulative logic directly in SQL.

---

## Stage 5 — Compute Step-over-Step Conversion Rate

The raw counts are useful, but the *rate* from one step to the next is what actually reveals drop-off.

**Variables:**
- `conversion_from_previous` — `users_reaching_step / LAG(users_reaching_step)`
- `LAG` — Window function that looks at the previous row's value

```sql
-- Stage 5: Step-over-step and overall conversion rates
SELECT
    step_order,
    step_name,
    users_reaching_step,
    ROUND(100.0 * users_reaching_step
        / LAG(users_reaching_step) OVER (ORDER BY step_order), 1) AS pct_of_previous_step,
    ROUND(100.0 * users_reaching_step
        / FIRST_VALUE(users_reaching_step) OVER (ORDER BY step_order), 1) AS pct_of_step1
FROM (... Stage 4 query ...)
ORDER BY step_order;
```

**Output:**

| step_order | step_name        | users_reaching_step | pct_of_previous_step | pct_of_step1 |
|------------|------------------|----------------------|-----------------------|--------------|
| 1          | view_product     | 5                    | —                     | 100.0%       |
| 2          | add_to_cart      | 4                    | 80.0%                 | 80.0%        |
| 3          | start_checkout   | 2                    | 50.0%                 | 40.0%        |
| 4          | purchase         | 1                    | 50.0%                 | 20.0%        |

**Why report *both* "% of previous step" and "% of step 1"?** They answer different questions. "% of previous step" (also called the *marginal* conversion rate) tells you exactly where the biggest relative drop-off happens — here, Step 2→3 loses half the remaining users, which is the step to investigate first. "% of step 1" (the *cumulative* conversion rate) tells you the end-to-end health of the whole funnel — useful for exec reporting ("20% of everyone who viewed a product bought one") but useless for diagnosing *which* step to fix, since a low number could come from a bad drop anywhere. **Why not report only the cumulative number?** Because two funnels can have the identical overall 20% conversion rate while failing at completely different steps — the cumulative number alone can't distinguish "everyone struggles at checkout" from "everyone struggles just to add to cart."

---

## Stage 6 — Add a Time Window (Why Funnels Need One)

So far we've silently assumed every event in the table "counts" toward the funnel, no matter how much time passed between steps. In practice, a user who viewed a product in January and purchased in June probably wasn't on a single continuous "funnel journey" — that's two different sessions/intents.

**Variables:**
- `window_days` — Maximum allowed gap between the first step and the step being measured
- `DATEDIFF` — Computes days between the user's Step 1 event and their later event

```sql
-- Stage 6: Restrict to events within N days of the user's first step
WITH first_step AS (
    SELECT
        user_id,
        MIN(event_time) AS first_seen
    FROM events
    WHERE event_type = 'view_product'
    GROUP BY user_id
),
windowed_events AS (
    SELECT
        e.user_id,
        e.event_type,
        e.event_time,
        f.first_seen
    FROM events e
    JOIN first_step f ON e.user_id = f.user_id
    WHERE e.event_time <= f.first_seen + INTERVAL '7 days'   -- 7-day funnel window
)
SELECT * FROM windowed_events;
```

**Why add a window at all?** Without one, a funnel silently conflates unrelated visits across arbitrarily long time spans, inflating "conversion" with events that have nothing to do with each other — a purchase six months after a single product view is far more likely to be a coincidence (or a second, unrelated visit) than the outcome of that specific browsing session. **Why not make the window infinite to be "safe" and not miss any conversions?** Because a funnel's entire purpose is to measure a specific, intentional journey — an infinite window makes the funnel unfalsifiable (nearly everyone "eventually" does everything given enough time), which defeats the point of measuring where people drop off *within a session or journey*. **Why 7 days specifically, and not 1 day or 30?** The right window length depends on the product's natural consideration cycle — impulse purchases (fast food app) might use a same-day window, while considered purchases (booking a vacation) might reasonably use 30+ days; there's no universal answer, and this choice should be stated explicitly and revisited if it changes the funnel numbers materially.

---

## Stage 7 — Break Out the Funnel by Segment

A single funnel is a starting point, but the real diagnostic value comes from comparing funnels across segments (e.g., new vs. returning users, or traffic source).

**Variables:**
- `segment` — A grouping dimension (e.g., `traffic_source`, `device_type`, `is_new_user`)

```sql
-- Stage 7: Funnel broken out by a segment column
SELECT
    e.traffic_source,                -- example segment
    f.step_order,
    f.step_name,
    COUNT(DISTINCT CASE WHEN u.max_step >= f.step_order THEN u.user_id END) AS users_reaching_step
FROM funnel_steps f
CROSS JOIN user_progress u
JOIN events e ON e.user_id = u.user_id
GROUP BY e.traffic_source, f.step_order, f.step_name
ORDER BY e.traffic_source, f.step_order;
```

**Why segment the funnel instead of stopping at the aggregate view?** An aggregate funnel can hide the fact that one segment is dragging down the average while another is already performing well — segmenting turns "40% drop off at checkout" into an actionable finding like "60% of *mobile* users drop off at checkout, but only 15% of desktop users do," which points straight at a mobile-checkout UX problem instead of a vague, undirected fix. **Why not segment by every possible dimension at once?** Slicing too finely (e.g., by device × traffic source × time-of-day × user tier simultaneously) fragments the data into tiny sample sizes per cell, where random noise can look like a meaningful pattern — start with one or two business-relevant segments and only drill further into a segment once it's shown a real signal.

---

## Stage 8 — Pivot Into the Classic Funnel Chart Shape

Finally, reshape the step-by-step rows into a single row per segment with one column per step — the shape you'd actually feed into a bar chart.

**Variables:**
- `MAX(CASE WHEN ...)` — Same conditional-aggregation pivot trick used in cohort analysis

```sql
-- Stage 8: Pivot funnel steps into columns
SELECT
    traffic_source,
    MAX(CASE WHEN step_order = 1 THEN users_reaching_step END) AS step1_view,
    MAX(CASE WHEN step_order = 2 THEN users_reaching_step END) AS step2_cart,
    MAX(CASE WHEN step_order = 3 THEN users_reaching_step END) AS step3_checkout,
    MAX(CASE WHEN step_order = 4 THEN users_reaching_step END) AS step4_purchase
FROM (... Stage 7 query ...)
GROUP BY traffic_source
ORDER BY traffic_source;
```

**Output (example, aggregate + two segments):**

| traffic_source | step1_view | step2_cart | step3_checkout | step4_purchase |
|-----------------|------------|------------|------------------|------------------|
| organic          | 3          | 3          | 2                | 1                |
| paid_social      | 2          | 1          | 0                | 0                |

**Why pivot at the very end and not earlier?** Every prior stage (2 through 7) needed the "long" row-per-step shape to make joins, window functions, and `GROUP BY`s work naturally — pivoting early would have made every subsequent calculation harder (you'd be doing arithmetic across columns instead of rows). **Why not skip the pivot and just hand analysts the long format?** Because the long format, while easier for SQL to compute, isn't how humans read a funnel — a wide, one-row-per-segment table is what maps directly onto a funnel visualization or a simple spreadsheet comparison, so the pivot is purely for human/downstream-tool readability, done only once all the hard computation is finished.

---

## The Full Query (All Stages Combined)

```sql
WITH
-- Stage 1: Step definitions
step_events AS (
    SELECT
        user_id,
        event_type,
        event_time,
        CASE event_type
            WHEN 'view_product'    THEN 1
            WHEN 'add_to_cart'     THEN 2
            WHEN 'start_checkout'  THEN 3
            WHEN 'purchase'        THEN 4
        END AS step_order
    FROM events
),

-- Stage 6: Restrict to a 7-day window from each user's first product view
first_step AS (
    SELECT user_id, MIN(event_time) AS first_seen
    FROM step_events
    WHERE step_order = 1
    GROUP BY user_id
),
windowed AS (
    SELECT s.*
    FROM step_events s
    JOIN first_step f ON s.user_id = f.user_id
    WHERE s.event_time <= f.first_seen + INTERVAL '7 days'
),

-- Stage 2: Furthest step reached per user (within window)
user_progress AS (
    SELECT user_id, MAX(step_order) AS max_step
    FROM windowed
    GROUP BY user_id
),

-- Step lookup
funnel_steps AS (
    SELECT 1 AS step_order, 'view_product' AS step_name
    UNION ALL SELECT 2, 'add_to_cart'
    UNION ALL SELECT 3, 'start_checkout'
    UNION ALL SELECT 4, 'purchase'
),

-- Stage 4: Cumulative users reaching each step
step_counts AS (
    SELECT
        f.step_order,
        f.step_name,
        COUNT(DISTINCT CASE WHEN u.max_step >= f.step_order THEN u.user_id END) AS users_reaching_step
    FROM funnel_steps f
    CROSS JOIN user_progress u
    GROUP BY f.step_order, f.step_name
)

-- Stage 5: Conversion rates
SELECT
    step_order,
    step_name,
    users_reaching_step,
    ROUND(100.0 * users_reaching_step
        / LAG(users_reaching_step) OVER (ORDER BY step_order), 1) AS pct_of_previous_step,
    ROUND(100.0 * users_reaching_step
        / FIRST_VALUE(users_reaching_step) OVER (ORDER BY step_order), 1) AS pct_of_step1
FROM step_counts
ORDER BY step_order;
```

---

## Key Variables Summary

| Variable | What It Is | Where Used |
|----------|-----------|------------|
| `step_order` | Manually assigned rank of each event type in the funnel | Stages 1–8 |
| `max_step` | Furthest step_order a user ever reached | Stages 2, 4 |
| `users_reaching_step` | COUNT DISTINCT users whose max_step >= this step | Stages 4–8 |
| `pct_of_previous_step` | Marginal (step-over-step) conversion rate | Stage 5 |
| `pct_of_step1` | Cumulative conversion rate from the very start | Stage 5 |
| `window_days` | Max allowed time gap from first step to count as "in-funnel" | Stage 6 |
| `segment` | Grouping dimension (traffic source, device, etc.) | Stage 7 |

---

## How to Read the Funnel

```
Step:            1 (View)   2 (Cart)   3 (Checkout)   4 (Purchase)
Users:              5          4            2               1
% of previous:      —         80%          50%             50%
% of step 1:       100%       80%          40%             20%
```

- **Read left to right** → where users drop off, in absolute and relative terms
- **The biggest % drop between adjacent steps** → your highest-priority fix
- **% of step 1 at the final step** → the single "headline" conversion rate, but never diagnostic on its own
- **Comparing segments side by side** → reveals whether a drop-off is universal or specific to one channel/device/cohort

---

## Common Variants — Why Each Exists, and Why Not to Default to It

**Time-to-convert distribution** — instead of just counting who reached each step, compute `event_time - first_seen` for each step and look at the distribution (median, p90) of how *long* it took. **Why:** two funnels can have identical conversion rates but wildly different speeds — slow conversions often signal friction even when users eventually get through. **Why not always:** adds real complexity for a question you may not need answered yet — start with the basic funnel counts first, and only add time-to-convert once you've identified a step worth investigating deeper.

**Strict ordered-sequence funnel** (from Stage 3, Approach B) — required when step order genuinely matters for the business question (e.g., "did they complete checkout *after* seeing the loyalty discount banner, in that order"). **Why not by default:** the join/window-function logic needed to enforce strict ordering is considerably more complex, and most standard product funnels (and the tools that generate them, like Amplitude/Mixpanel) default to the simpler "ever reached" definition — reach for strict ordering only when the business question specifically requires it.

**Multi-path / branching funnels** — real products often have more than one valid path to conversion (e.g., "buy now" vs. "add to cart then checkout later"). **Why:** forcing every user into one single linear funnel undercounts people who converted via a different but equally valid path. **Why not model every possible path from day one:** it multiplies analysis complexity quickly — start with the single dominant path, and only branch the funnel once data shows a second path carries meaningful volume.

**Funnel with re-entry allowed** — some funnels should let a user "restart" after a long gap (e.g., they abandoned cart, came back a month later, and completed a full new session). **Why:** without allowing re-entry, a user who fails once is permanently counted as a drop-off even if they later fully succeed in a new session. **Why not always allow re-entry:** it can double-count a single underlying user's persistence as if it were two separate successes, inflating apparent funnel health — decide explicitly whether you're measuring "session-level" or "user-level, ever" conversion, and be consistent.
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
