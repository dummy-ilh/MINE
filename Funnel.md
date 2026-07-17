# PART 1 — FUNNEL CONVERSION

A **funnel** measures how many users progress through a sequence of events.

Given a basic events table:

```sql
-- events(user_id, event_name, event_time)
```

---

# Table of Contents

1. Part 1 — Funnel Conversion (Interview Walkthrough)
   - 1.0 The Setup
   - 1.1 Conditional Aggregation (90% Solution)
   - 1.2 Self-Joins (Strict Order)
   - 1.3 Conversion Rates
   - 1.4 Variants
   - 1.5 Interview Traps
2. Part 2 — Core Funnel SQL Patterns
   - Conditional Aggregation
   - Self-Joins
   - Pivot + Window Functions
3. Part 3 — Understanding the SQL
4. Part 4 — Reusable Templates
5. Which Approach Should You Use?

---

# PART 1 — FUNNEL CONVERSION

## 1.0 The Setup

Almost every funnel interview starts with a single events table.

```
events
+---------+-------------+---------------------+
| user_id | event_name  | event_time          |
+---------+-------------+---------------------+
| 1       | signup      | 2024-01-01 10:00:00 |
| 1       | activation  | 2024-01-01 10:05:00 |
| 1       | purchase    | 2024-01-02 09:00:00 |
| 2       | signup      | 2024-01-01 11:00:00 |
| 2       | activation  | 2024-01-03 12:00:00 |
| 3       | signup      | 2024-01-01 12:00:00 |
| 4       | signup      | 2024-01-02 08:00:00 |
| 4       | purchase    | 2024-01-02 09:30:00 | -- purchased WITHOUT activation
+---------+-------------+---------------------+
```

### Users move through the funnel differently

- ✅ User 1: signup → activation → purchase
- ⚠️ User 2: signup → activation → drop-off
- ⚠️ User 3: signup only
- ⚠️ User 4: signup → purchase (skipped activation)

This last case is what separates correct funnel SQL from naive solutions.

**Interview question**

> For each stage, how many users reached it, and what is the conversion between stages?

---

## 1.1 Conditional Aggregation (The 90% Solution)

**Idea**

Collapse every user's events into one row of boolean flags.

```sql
WITH user_flags AS (
    SELECT
        user_id,
        MAX(CASE WHEN event_name='signup' THEN 1 ELSE 0 END) AS did_signup,
        MAX(CASE WHEN event_name='activation' THEN 1 ELSE 0 END) AS did_activation,
        MAX(CASE WHEN event_name='purchase' THEN 1 ELSE 0 END) AS did_purchase
    FROM events
    GROUP BY user_id
)

SELECT
    SUM(did_signup)     AS signup_count,
    SUM(did_activation) AS activation_count,
    SUM(did_purchase)   AS purchase_count
FROM user_flags;
```

Output

```
signup_count | activation_count | purchase_count
4            | 2                | 2
```

### Interview takeaway
Problem: this counts user 4 as "activated → purchased" implicitly through independence, but it does NOT enforce order. If your funnel doesn't require strict sequence, this is genuinely fine and is what most product analytics dashboards do (Mixpanel/Amplitude funnels are configurable between "any order" and "in order"). If the interviewer says "step-by-step, in order

---

## 1.2 Self-Joins (Strict Sequential Funnel)

To enforce order, every stage must happen **after** the previous stage.

```sql
WITH signup AS (
    SELECT user_id,
           MIN(event_time) AS signup_time
    FROM events
    WHERE event_name='signup'
    GROUP BY user_id
),
activation AS (
    SELECT user_id,
           MIN(event_time) AS activation_time
    FROM events
    WHERE event_name='activation'
    GROUP BY user_id
),
purchase AS (
    SELECT user_id,
           MIN(event_time) AS purchase_time
    FROM events
    WHERE event_name='purchase'
    GROUP BY user_id
)

SELECT
    s.user_id,
    s.signup_time,
    a.activation_time,
    p.purchase_time
FROM signup s
LEFT JOIN activation a
    ON s.user_id=a.user_id
   AND a.activation_time>s.signup_time
LEFT JOIN purchase p
    ON a.user_id=p.user_id
   AND p.purchase_time>a.activation_time;
```

Result

user_id | signup_time         | activation_time     | purchase_time
1       | 2024-01-01 10:00:00 | 2024-01-01 10:05:00 | 2024-01-02 09:00:00
2       | 2024-01-01 11:00:00 | 2024-01-03 12:00:00 | NULL
3       | 2024-01-01 12:00:00 | NULL                | NULL
4       | 2024-01-02 08:00:00 | NULL                | NULL   <- purchase dropped! joined off activation
```

This is the key interview trap. User 4 purchased but never activated. Because the purchase join is chained off activation_time (to enforce strict in-order sequence), user 4's purchase disappears — which is correct if the funnel truly requires activation before purchase, but wrong if you meant "reached this step at all, regardless of order." Always ask the interviewer: "does the funnel require strict sequential order, or just 'did they eventually do X'?" This single clarifying question is a strong L5 signal — it shows you know the two interpretations produce different numbers.

### Interview takeaway

Always ask:

> "Does the funnel require strict sequential order, or do you simply want users that eventually reached each stage?"

That one clarification changes the answer.

---

## 1.3 Conversion Rates

Once you have ordered counts, computing conversion becomes easy.

```sql
WITH funnel_counts AS (
    SELECT
        COUNT(*)               AS signup_count,
        COUNT(activation_time) AS activation_count,
        COUNT(purchase_time)   AS purchase_count
    FROM ordered_funnel
)

SELECT
    signup_count,
    activation_count,
    purchase_count,

    ROUND(
        100.0 * activation_count /
        NULLIF(signup_count,0),
        1
    ) AS signup_to_activation_pct,

    ROUND(
        100.0 * purchase_count /
        NULLIF(activation_count,0),
        1
    ) AS activation_to_purchase_pct,

    ROUND(
        100.0 * purchase_count /
        NULLIF(signup_count,0),
        1
    ) AS overall_conversion_pct

FROM funnel_counts;
```

Output

```
signup_count     = 4
activation_count = 2
purchase_count   = 1

signup → activation = 50%

activation → purchase = 50%

overall conversion = 25%
```

```
Signup (4)
   │
 50%
   ▼
Activation (2)
   │
 50%
   ▼
Purchase (1)

Overall = 25%
```

### Interview takeaway

Always use `NULLIF()` to avoid divide-by-zero errors.

---

## 1.4 Variants

### Variant A — Window Functions

Same logic as self-joins, but only one scan over the table.

(Your existing SQL goes here.)

---

### Variant B — Time-Bounded Funnels

Example:

```
Activation must occur within 7 days of signup.
```

Add

```sql
AND activation_time <= signup_time + INTERVAL '7 days'
```

---

### Variant C — Generic N-Step Funnel

Use a step definition table plus `LAG()` instead of writing eight joins.

(Your existing SQL goes here.)

---

### Variant D — Segmented Funnel

Group by channel, country, device, etc.

Compute percentages **inside** each segment.

---

## 1.5 Common Interview Mistakes

- Fan-out from multiple events per step (always aggregate before joining)
- Forgetting to clarify strict order vs. "ever reached"
- Divide-by-zero (use `NULLIF`)
- Duplicate events causing inflated counts
- Timezone / date-window ambiguity
- Joining purchases directly from signup instead of activation

---

# PART 2 — Core Funnel SQL Patterns

## 2.1 Conditional Aggregation (Fastest)

...your existing section...

---

## 2.2 Self-Joins (Strict Sequential Funnel)

...existing section...

---

## 2.3 Pivot + Window Functions (Recommended)

...existing section...

---

# PART 3 — Understanding the SQL

- `COUNT(DISTINCT CASE WHEN ...)`
- `GROUP BY + MIN()`
- `MAX(CASE WHEN ...)`
- `FILTER()`
- Why NULL comparisons work

...existing explanations...

---

# PART 4 — Reusable Templates

...your existing template section...

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

| Method | Order Enforced | Scales | Interview Frequency |
|----------|---------------|--------|---------------------|
| Conditional Aggregation | ❌ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Self-Joins | ✅ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Pivot + Window | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Generic N-Step | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Rule of thumb**

- **Need quick counts?** → Conditional Aggregation.
- **Need strict ordering?** → Self-Joins.
- **Need scalable production funnels?** → Pivot + Window Functions.
