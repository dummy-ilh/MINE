# SQL Funnels

A **funnel** measures how many users progress through a sequence of events.

Given a basic events table:

```sql
-- events(user_id, event_name, event_time)
```

---

# Table of Contents

1. Conditional Aggregation (Fastest)
2. Self-Joins (Strict Sequential Funnel)
3. Pivot + Window Functions (Recommended)
4. Understanding the SQL
   - `COUNT(DISTINCT CASE WHEN...)`
   - `GROUP BY + MIN()`
   - `MAX(CASE WHEN...)`
   - Why NULL comparisons work
5. Funnel Templates
6. Which Approach Should You Use?

---

# 1. Conditional Aggregation (Fastest)

The simplest way to build a funnel.

Count users that performed each event independently.

```sql
SELECT
    COUNT(DISTINCT CASE WHEN event_name = 'visit'    THEN user_id END) AS visited,
    COUNT(DISTINCT CASE WHEN event_name = 'signup'   THEN user_id END) AS signed_up,
    COUNT(DISTINCT CASE WHEN event_name = 'add_cart' THEN user_id END) AS added_cart,
    COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN user_id END) AS purchased
FROM events
WHERE event_time BETWEEN '2026-06-01' AND '2026-06-30';
```

### Pros

- Very short
- Single pass over the table
- Easy to add/remove stages

### Cons

Does **not** enforce order.

Someone who purchased without visiting in this time window will still count as a purchaser.

---

# 2. Self-Joins (Strict Sequential Funnel)

Instead of counting each event independently, find each user's first event at every stage and join them together.

```sql
SELECT
    COUNT(DISTINCT v.user_id) AS visited,
    COUNT(DISTINCT s.user_id) AS signed_up,
    COUNT(DISTINCT c.user_id) AS added_cart,
    COUNT(DISTINCT p.user_id) AS purchased
FROM (
    SELECT user_id,
           MIN(event_time) AS t
    FROM events
    WHERE event_name='visit'
    GROUP BY user_id
) v

LEFT JOIN (
    SELECT user_id,
           MIN(event_time) AS t
    FROM events
    WHERE event_name='signup'
    GROUP BY user_id
) s
ON s.user_id=v.user_id
AND s.t>=v.t

LEFT JOIN (
    SELECT user_id,
           MIN(event_time) AS t
    FROM events
    WHERE event_name='add_cart'
    GROUP BY user_id
) c
ON c.user_id=s.user_id
AND c.t>=s.t

LEFT JOIN (
    SELECT user_id,
           MIN(event_time) AS t
    FROM events
    WHERE event_name='purchase'
    GROUP BY user_id
) p
ON p.user_id=c.user_id
AND p.t>=c.t;
```

### Pros

- Enforces order
- Easy to understand

### Cons

- Gets very long
- Every new stage requires another join

---

# 3. Pivot + Window Functions (Recommended)

This is the approach most analytics teams eventually standardize on.

Instead of multiple joins:

1. Find each user's first occurrence of every event.
2. Pivot those timestamps into columns.
3. Compare timestamps.

```sql
WITH ranked AS (

SELECT
    user_id,
    event_name,
    event_time,
    ROW_NUMBER() OVER(
        PARTITION BY user_id,event_name
        ORDER BY event_time
    ) rn
FROM events

),

first_touch AS (

SELECT
    user_id,
    event_name,
    event_time
FROM ranked
WHERE rn=1

),

pivoted AS (

SELECT

    user_id,

    MAX(CASE WHEN event_name='visit'
             THEN event_time END) AS t_visit,

    MAX(CASE WHEN event_name='signup'
             THEN event_time END) AS t_signup,

    MAX(CASE WHEN event_name='add_cart'
             THEN event_time END) AS t_cart,

    MAX(CASE WHEN event_name='purchase'
             THEN event_time END) AS t_purchase

FROM first_touch
GROUP BY user_id

)

SELECT

COUNT(*) FILTER(
WHERE t_visit IS NOT NULL
) AS visited,

COUNT(*) FILTER(
WHERE t_signup>=t_visit
) AS signed_up,

COUNT(*) FILTER(
WHERE t_cart>=t_signup
) AS added_cart,

COUNT(*) FILTER(
WHERE t_purchase>=t_cart
) AS purchased

FROM pivoted;
```

### Why this scales well

The final table looks like this:

| user_id | t_visit | t_signup | t_cart | t_purchase |
|----------|----------|-----------|---------|------------|
| 1 | 09:00 | 09:10 | 09:20 | 09:45 |
| 2 | 10:00 | NULL | NULL | NULL |
| 3 | 11:00 | 11:05 | NULL | NULL |

Once data looks like this, almost every funnel metric becomes easy.

---

# Understanding the SQL

---

## COUNT(DISTINCT CASE WHEN...)

Example:

```sql
COUNT(DISTINCT CASE
    WHEN event_name='visit'
    THEN user_id
END)
```

This is probably the most common interview question.

---

### Step 1

The CASE expression runs on every row.

```sql
CASE
WHEN event_name='visit'
THEN user_id
END
```

If the event is a visit:

```
returns user_id
```

Otherwise:

```
returns NULL
```

Example:

| user_id | event_name | CASE result |
|----------|------------|-------------|
| 1 | visit | 1 |
| 1 | signup | NULL |
| 2 | visit | 2 |
| 3 | purchase | NULL |

---

### Step 2

`COUNT()` ignores NULLs.

So

```sql
COUNT(CASE WHEN ...)
```

counts visit **rows**.

---

### Step 3

`DISTINCT` removes duplicate users.

Without DISTINCT

```
User 1 visited 5 times

Count = 5
```

With DISTINCT

```
User 1 visited 5 times

Count = 1
```

So

```sql
COUNT(DISTINCT CASE WHEN ...)
```

means

> Count unique users that had at least one visit.

---

## GROUP BY + MIN()

Example:

```sql
SELECT
    user_id,
    MIN(event_time) AS t
FROM events
WHERE event_name='visit'
GROUP BY user_id;
```

### What happens?

Imagine

| user_id | event_time |
|----------|------------|
| 1 | 09:00 |
| 1 | 09:10 |
| 1 | 09:30 |

After grouping

```
User 1

↓

09:00
```

Only the earliest visit remains.

---

### Why not DISTINCT?

Sometimes you'll see

```sql
SELECT DISTINCT
    user_id,
    MIN(event_time)
...
GROUP BY user_id
```

The `DISTINCT` is redundant.

`GROUP BY user_id` already guarantees one row per user.

---

## MAX(CASE WHEN...)

Example

```sql
MAX(
CASE
WHEN event_name='visit'
THEN event_time
END
)
```

This is the pivot trick.

Suppose one user has

| event | CASE result |
|--------|-------------|
| visit | 09:00 |
| signup | NULL |
| purchase | NULL |

MAX becomes

```
MAX(09:00,NULL,NULL)

↓

09:00
```

It simply pulls the timestamp into its own column.

---

### Why MAX instead of MIN?

In the previous CTE (`first_touch`), we already kept only the first event.

So there is only one non-null timestamp.

That means

```sql
MAX(...)
```

and

```sql
MIN(...)
```

produce exactly the same result.

The aggregate exists because SQL requires one when using `GROUP BY`.

---

## Why NULL comparisons work

Suppose

```sql
COUNT(*) FILTER(
WHERE t_signup>=t_visit
)
```

If a user never signed up

```
t_signup=NULL
```

Then SQL evaluates

```
NULL >= something

↓

NULL
```

A WHERE clause only keeps rows where the condition is TRUE.

NULL is not TRUE.

So the row is automatically excluded.

| t_visit | t_signup | Comparison | Counted? |
|----------|-----------|------------|----------|
| 09:00 | 09:10 | TRUE | ✅ |
| 09:00 | NULL | NULL | ❌ |
| NULL | 09:10 | NULL | ❌ |
| NULL | NULL | NULL | ❌ |

This is why funnel SQL often doesn't need explicit `IS NOT NULL` checks.

---

# Reusable Funnel Templates

---

## Template 1 — Conditional Aggregation

```sql
SELECT
COUNT(DISTINCT CASE WHEN event_name='STAGE_1' THEN user_id END) stage1,
COUNT(DISTINCT CASE WHEN event_name='STAGE_2' THEN user_id END) stage2,
COUNT(DISTINCT CASE WHEN event_name='STAGE_3' THEN user_id END) stage3
FROM events;
```

Use when:

- quick dashboard
- order doesn't matter

---

## Template 2 — Ordered Funnel

```sql
WITH

stage1 AS (

SELECT user_id,
MIN(event_time) t1

FROM events

WHERE event_name='STAGE_1'

GROUP BY user_id

),

stage2 AS (

SELECT user_id,
MIN(event_time) t2

FROM events

WHERE event_name='STAGE_2'

GROUP BY user_id

)

SELECT *
FROM stage1
LEFT JOIN stage2
ON stage1.user_id=stage2.user_id
AND t2>=t1;
```

Use when:

- strict ordering
- few stages

---

## Template 3 — Conversion Window

```sql
WHERE t2>=t1
AND t2<=t1+INTERVAL '7 days'
```

Useful for questions like

> "How many users signed up within 7 days?"

---

## Template 4 — Conversion Rate

```sql
SELECT

ROUND(

100.0 *

COUNT(*) FILTER(
WHERE t_signup>=t_visit
)

/

NULLIF(

COUNT(*) FILTER(
WHERE t_visit IS NOT NULL
),

0

),

1

) AS conversion_rate

FROM pivoted;
```

---

## Template 5 — Average Time Between Stages

```sql
SELECT

AVG(t_signup-t_visit)

FILTER(

WHERE t_signup>=t_visit

)

FROM pivoted;
```

Useful for

- average signup time
- average purchase time
- average onboarding duration

---

# Which Approach Should You Use?

| Situation | Best Choice |
|------------|------------|
| Quick dashboard | Conditional Aggregation |
| Small ordered funnel | Self-Joins |
| 4+ stages | Pivot + Window Functions |
| Need conversion time | Pivot + Window Functions |
| Production analytics | Pivot + Window Functions |

---

# Summary

| Approach | Pros | Cons |
|-----------|------|------|
| Conditional Aggregation | Fastest, simplest | Doesn't enforce order |
| Self-Joins | Strict ordering | Doesn't scale well |
| Pivot + Window Functions | Clean, scalable, production-friendly | Slightly more advanced |

**Rule of thumb**

- **Need counts only?** → Conditional aggregation.
- **Need strict ordering with a few stages?** → Self-joins.
- **Need scalable production funnels, conversion rates, or time-to-convert?** → Pivot + Window Functions.
