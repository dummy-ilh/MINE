# SQL Cheat Sheet — FAANG DS/MLE (Medium–Hard)

---

## 0. Order of Execution (know this cold)

```
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT
```
⚠️ You can't use a SELECT alias in WHERE (alias doesn't exist yet) but you CAN in ORDER BY.

---

## 1. JOINs

```sql
-- INNER: only matching rows
SELECT * FROM a JOIN b ON a.id = b.id;

-- LEFT: all of a, NULLs from b if no match
SELECT * FROM a LEFT JOIN b ON a.id = b.id;

-- SELF JOIN: compare rows within same table
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```
⚠️ **Fan-out trap**: joining a 1-row table to a many-row table duplicates rows → inflates SUM/COUNT. Always check granularity before aggregating post-join. Aggregate first, then join, when possible.

---

## 2. GROUP BY / HAVING / CASE WHEN

```sql
-- WHERE filters rows BEFORE grouping; HAVING filters groups AFTER
SELECT dept_id, COUNT(*) AS n
FROM employees
WHERE active = 1
GROUP BY dept_id
HAVING COUNT(*) > 5;

-- Conditional aggregation = mini pivot
SELECT
  dept_id,
  SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) AS female_count,
  SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) AS male_count
FROM employees
GROUP BY dept_id;
```
⚠️ `COUNT(*)` counts all rows; `COUNT(col)` skips NULLs. This silently changes results when col has NULLs.

---

## 3. Window Functions (the #1 FAANG topic)

```sql
SELECT *,
  ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn,
  RANK()       OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rk,
  DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS drk
FROM employees;
```
- `ROW_NUMBER`: always unique 1,2,3,4... (no ties)
- `RANK`: ties share rank, then **skips** (1,2,2,4)
- `DENSE_RANK`: ties share rank, **no skip** (1,2,2,3)

⚠️ Pick based on whether ties should "cost" a position. Nth-highest problems almost always want `DENSE_RANK`.

```sql
-- Nth highest salary per department
SELECT * FROM (
  SELECT *, DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rnk
  FROM employees
) t
WHERE rnk = 2;
```

```sql
-- LAG/LEAD: compare to previous/next row
SELECT
  user_id, order_date,
  LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS prev_date,
  order_date - LAG(order_date) OVER (PARTITION BY user_id ORDER BY order_date) AS days_since_last
FROM orders;
```

```sql
-- Running total / moving average
SELECT
  order_date, revenue,
  SUM(revenue) OVER (ORDER BY order_date) AS running_total,
  AVG(revenue) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_avg_7d
FROM daily_sales;
```
⚠️ Default frame for `ORDER BY` without explicit `ROWS`/`RANGE` is `RANGE UNBOUNDED PRECEDING AND CURRENT ROW` — fine for running totals, but be explicit (`ROWS BETWEEN n PRECEDING AND CURRENT ROW`) for moving windows or you'll get wrong results when ORDER BY column has duplicates.

```sql
-- Top-N per group (most common interview ask)
SELECT * FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS rn
  FROM products
) t
WHERE rn <= 3;
```

---

## 4. Gaps & Islands (streaks / sessions)

```sql
-- Consecutive login streak: subtract row_number from date → constant for a streak
SELECT user_id,
  date - (ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date))::int AS grp
FROM logins;
-- Rows with same grp = one continuous streak. COUNT(*) per grp = streak length.
```

```sql
-- Sessionization: new session if gap > 30 min since last event
SELECT *,
  SUM(CASE WHEN gap_minutes > 30 OR gap_minutes IS NULL THEN 1 ELSE 0 END)
    OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
FROM (
  SELECT *,
    EXTRACT(EPOCH FROM event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time))/60 AS gap_minutes
  FROM events
) t;
```
⚠️ This pattern (date minus row_number, or cumulative flag-on-gap) is the single most reused trick across "find consecutive X" and "sessionize events" questions — recognize the shape, not the story.

---

## 5. Pivoting

```sql
-- Rows to columns via conditional aggregation (portable, no PIVOT keyword needed)
SELECT
  user_id,
  MAX(CASE WHEN metric = 'clicks' THEN value END) AS clicks,
  MAX(CASE WHEN metric = 'views'  THEN value END) AS views
FROM events_long
GROUP BY user_id;
```

---

## 6. A/B Testing Pattern

```sql
SELECT
  experiment_group,
  COUNT(DISTINCT user_id) AS users,
  COUNT(DISTINCT CASE WHEN converted = 1 THEN user_id END) AS converters,
  COUNT(DISTINCT CASE WHEN converted = 1 THEN user_id END)::float
    / COUNT(DISTINCT user_id) AS conversion_rate
FROM ab_test_events
GROUP BY experiment_group;
```
⚠️ Use `COUNT(DISTINCT user_id)`, not `COUNT(*)` — event logs have multiple rows per user, and counting rows instead of users is the classic "user-level vs event-level" trap Google likes to test.

---

## 7. Funnel Analysis

```sql
-- Step-by-step drop-off using conditional MAX (did user reach each step?)
SELECT
  COUNT(DISTINCT CASE WHEN step = 'view'     THEN user_id END) AS viewed,
  COUNT(DISTINCT CASE WHEN step = 'cart'     THEN user_id END) AS carted,
  COUNT(DISTINCT CASE WHEN step = 'purchase' THEN user_id END) AS purchased
FROM funnel_events;
```
⚠️ This only works if a user logically can't reach a later step without the earlier one. If steps can be skipped, you need a self-join checking `cart.user_id IN (SELECT user_id FROM view_step)` per stage instead.

---

## 8. Cohort / Retention

```sql
-- D1/D7/D30 retention by signup cohort
WITH cohort AS (
  SELECT user_id, DATE_TRUNC('week', signup_date) AS cohort_week
  FROM users
)
SELECT
  c.cohort_week,
  COUNT(DISTINCT c.user_id) AS cohort_size,
  COUNT(DISTINCT CASE WHEN e.event_date = c.cohort_week + INTERVAL '1 day'  THEN e.user_id END) AS d1,
  COUNT(DISTINCT CASE WHEN e.event_date = c.cohort_week + INTERVAL '7 days' THEN e.user_id END) AS d7
FROM cohort c
LEFT JOIN events e ON c.user_id = e.user_id
GROUP BY c.cohort_week;
```

---

## 9. Weighted Average (metric aggregation trap)

```sql
-- WRONG: AVG(rate) treats every row equally regardless of volume
-- RIGHT: weight by denominator
SELECT
  SUM(conversions) * 1.0 / SUM(visits) AS true_avg_conversion_rate
FROM daily_metrics;
```
⚠️ `AVG(daily_conversion_rate)` over days with wildly different traffic gives a misleading "average of averages." Always aggregate numerator/denominator separately, then divide.

---

## 10. CTEs vs Subqueries

```sql
WITH high_value AS (
  SELECT user_id, SUM(amount) AS total
  FROM orders
  GROUP BY user_id
  HAVING SUM(amount) > 1000
)
SELECT u.name, h.total
FROM high_value h JOIN users u ON h.user_id = u.id;
```
⚠️ CTEs aren't always materialized (depends on engine/optimizer) — don't assume a CTE referenced twice is computed once. For genuinely reused intermediate results, check the engine or use a temp table.

---

## 11. Recursive CTE (hierarchies — lighter-weight, but Google has asked)

```sql
WITH RECURSIVE org_chart AS (
  SELECT id, name, manager_id, 1 AS depth
  FROM employees WHERE manager_id IS NULL          -- anchor: top of hierarchy
  UNION ALL
  SELECT e.id, e.name, e.manager_id, oc.depth + 1
  FROM employees e
  JOIN org_chart oc ON e.manager_id = oc.id          -- recursive: go one level down
)
SELECT * FROM org_chart;
```
⚠️ Always have an anchor (base case) and a terminating join condition, or you get an infinite loop.

---

## 12. More Window Functions

```sql
-- NTILE: split into n equal buckets (quartiles/deciles)
SELECT user_id, total_spend,
  NTILE(4) OVER (ORDER BY total_spend DESC) AS spend_quartile
FROM user_spend;
-- quartile 1 = top 25% of spenders

-- FIRST_VALUE / LAST_VALUE: grab boundary value without a subquery
SELECT user_id, order_date, amount,
  FIRST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY order_date) AS first_purchase_amt,
  LAST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY order_date
    RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_purchase_amt
FROM orders;
```
⚠️ `LAST_VALUE` with the default frame only sees up to the current row, so it looks "broken" (returns current row's value, not the true last). You must explicitly widen the frame to `UNBOUNDED FOLLOWING` to get the actual last value.

```sql
-- PERCENT_RANK / CUME_DIST: relative standing, 0 to 1
SELECT user_id, score,
  PERCENT_RANK() OVER (ORDER BY score) AS pct_rank,
  CUME_DIST()    OVER (ORDER BY score) AS cume_dist
FROM scores;

-- Median: no MEDIAN() keyword — use PERCENTILE_CONT
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;
-- per-group median: add OVER (PARTITION BY dept_id) where supported, or GROUP BY dept_id with the WITHIN GROUP form
```

**Frame clause: `ROWS` vs `RANGE` vs `GROUPS`**
```sql
-- ROWS: physical row count — exactly N rows back, regardless of ties
SUM(x) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)

-- RANGE: logical value range — includes ALL rows with the same ORDER BY value as current row
SUM(x) OVER (ORDER BY date RANGE BETWEEN 6 PRECEDING AND CURRENT ROW)
```
⚠️ If `date` has duplicate values, `RANGE` pulls in every tied row, silently changing your "7-day window" into something wider. Default `ORDER BY` alone uses `RANGE UNBOUNDED PRECEDING` — for moving windows, always use `ROWS` explicitly unless you specifically want tie-inclusive behavior.

---

## 13. EXISTS vs IN vs JOIN

```sql
-- EXISTS: stops at first match, handles NULLs safely — best for "does a related row exist"
SELECT * FROM users u
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);

-- IN: simple but breaks silently with NULLs in the subquery
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);
-- NOT IN is the dangerous one:
SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM orders WHERE user_id IS NOT NULL);
```
⚠️ `NOT IN` against a subquery that can return NULL returns **zero rows**, even when matches clearly should exist — NULL poisons the whole comparison. Use `NOT EXISTS` instead of `NOT IN` whenever the subquery column might contain NULLs.

```sql
-- Correlated subquery: inner query references the outer row
SELECT e.name, e.salary
FROM employees e
WHERE e.salary > (
  SELECT AVG(salary) FROM employees WHERE department_id = e.department_id
);
-- "employees earning more than their own department's average"
```

---

## 14. Set Operations

```sql
UNION       -- combines + removes duplicates (does an implicit sort/dedup — slower)
UNION ALL   -- combines, keeps duplicates (faster, use unless you need dedup)
INTERSECT   -- rows present in both queries
EXCEPT      -- rows in first query but not second (MINUS in Oracle)
```
⚠️ Default to `UNION ALL` unless you specifically need deduplication — `UNION` silently drops legitimate duplicate rows and costs more.

---

## 15. STRING_AGG / ARRAY_AGG (collapse rows to a list)

```sql
-- One row per user, all their products as a comma-separated string
SELECT user_id, STRING_AGG(product_name, ', ' ORDER BY product_name) AS products
FROM purchases
GROUP BY user_id;

-- Same idea but as an actual array (Postgres)
SELECT user_id, ARRAY_AGG(product_name ORDER BY purchase_date) AS products_in_order
FROM purchases
GROUP BY user_id;
```

---

## 16. Churn (different shape than retention!)

```sql
-- Users active in period N but NOT active in period N+1
SELECT a.user_id
FROM monthly_activity a
WHERE a.month = '2026-05'
  AND NOT EXISTS (
    SELECT 1 FROM monthly_activity b
    WHERE b.user_id = a.user_id AND b.month = '2026-06'
  );
```
⚠️ Retention asks "did they come back?" (positive join), churn asks "did they NOT come back?" (anti-join / NOT EXISTS). People often try to force churn into the cohort/retention CASE WHEN template — it doesn't fit cleanly; reach for anti-join instead.

---

## 17. Strict-Order Funnel (steps can be skipped or out of order)

```sql
-- For each user, find the LATEST step they reached, respecting required order
WITH ranked AS (
  SELECT user_id, step,
    CASE step WHEN 'view' THEN 1 WHEN 'cart' THEN 2 WHEN 'purchase' THEN 3 END AS step_order,
    event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_time
  FROM funnel_events
),
-- a step only "counts" if a strictly earlier step happened before it for that user
valid_steps AS (
  SELECT r.user_id, r.step_order
  FROM ranked r
  WHERE r.step_order = 1
     OR EXISTS (
       SELECT 1 FROM funnel_events e
       WHERE e.user_id = r.user_id
         AND e.event_time < r.event_time
         AND CASE e.step WHEN 'view' THEN 1 WHEN 'cart' THEN 2 WHEN 'purchase' THEN 3 END = r.step_order - 1
     )
)
SELECT step_order, COUNT(DISTINCT user_id) AS users_reached
FROM valid_steps
GROUP BY step_order
ORDER BY step_order;
```
⚠️ The naive funnel (section 7) just counts who has *any* row at each step — fine if steps are physically impossible to skip. The moment a user can jump straight to "purchase" without "cart" (e.g. buy-now links, support-added items), you need this order-aware version or your funnel overcounts later steps.

---

## 18. Sampling

```sql
-- Random sample of rows, engine-native (Postgres / similar)
SELECT * FROM events
ORDER BY RANDOM()
LIMIT 1000;
-- ⚠️ ORDER BY RANDOM() on a huge table is SLOW (sorts the whole table). Fine for interviews, bad in prod.

-- TABLESAMPLE: fast, block-level sampling — doesn't guarantee exact row count but scales to big tables
SELECT * FROM events TABLESAMPLE BERNOULLI (1);   -- ~1% of rows, Postgres
SELECT * FROM events TABLESAMPLE SYSTEM (1);      -- faster, less random (page-level)

-- Deterministic sampling by hashing an ID — same users always selected (good for stable A/B holdouts)
SELECT * FROM users
WHERE MOD(ABS(HASHTEXT(user_id::text)), 100) < 10;   -- ~10% of users, consistently
```
⚠️ For A/B test assignment specifically, always sample by a stable key (user_id hash), never `RANDOM()` per row — otherwise the same user can land in different groups across sessions/queries, contaminating the experiment.

---

## 19. NULL Gotchas

```sql
-- NULL is never "= NULL" — must use IS NULL / IS NOT NULL
SELECT * FROM t WHERE col IS NULL;

-- AVG/SUM/COUNT(col) silently skip NULLs
SELECT AVG(score) FROM t;  -- ignores NULL scores, NOT treated as 0

-- COALESCE to default NULLs before math
SELECT SUM(COALESCE(discount, 0)) FROM orders;
```
⚠️ If you want NULL treated as 0 (common in revenue/discount fields), you must `COALESCE` explicitly — SQL won't do it for you, and this changes both AVG and SUM results.

---

## 20. Quick Pattern → Trigger-Word Map

| Question mentions...                          | Use this pattern                    |
|------------------------------------------------|--------------------------------------|
| "consecutive", "streak", "in a row"            | Gaps & islands (date − row_number)   |
| "session", "within X minutes of"               | Gaps & islands (cumulative flag)     |
| "Nth highest/lowest per group"                 | DENSE_RANK + filter                  |
| "top N per category"                           | ROW_NUMBER + filter                  |
| "compare to previous/next row"                 | LAG / LEAD                           |
| "running total", "cumulative sum"              | Window SUM with ORDER BY             |
| "moving average", "rolling N days"             | Window AVG with ROWS BETWEEN         |
| "quartile", "decile", "top X%"                 | NTILE                                |
| "first/last purchase", "first event per user"  | FIRST_VALUE / LAST_VALUE             |
| "percentile", "relative standing"              | PERCENT_RANK / CUME_DIST             |
| "median"                                       | PERCENTILE_CONT(0.5)                 |
| "manager chain", "org hierarchy", "depth"      | Self-join or Recursive CTE           |
| "average earner in own dept", "above own group avg" | Correlated subquery             |
| "has at least one order", "no matching row"    | EXISTS / NOT EXISTS (not IN/NOT IN)  |
| "list of products per user", "comma-separated" | STRING_AGG / ARRAY_AGG               |
| "conversion rate", "variant A vs B"            | A/B pattern: DISTINCT user counts    |
| "step 1 → step 2 → step 3 drop-off"            | Funnel: conditional COUNT(DISTINCT)  |
| "drop-off but steps can be skipped"            | Strict-order funnel (self-join)      |
| "retention", "D1/D7/D30", "cohort"             | Cohort pattern with DATE_TRUNC       |
| "churn", "stopped using", "did not return"     | Anti-join / NOT EXISTS               |
| "pivot", "wide format", "one row per user"     | CASE WHEN + MAX/SUM + GROUP BY       |
| "average of rates/percentages"                 | Weighted avg trap — check it!        |
| "random sample", "X% of users", "holdout group"| TABLESAMPLE or hashed deterministic sampling |

---

## 21. Final Sanity Checklist (run mentally before submitting any query)

- [ ] Did a JOIN possibly duplicate rows before my aggregation? (fan-out check)
- [ ] Should I be counting DISTINCT users instead of rows/events?
- [ ] Are there NULLs that AVG/SUM/COUNT will silently drop or that need COALESCE?
- [ ] Am I using NOT IN with a subquery that could contain NULLs? (use NOT EXISTS instead)
- [ ] RANK vs DENSE_RANK vs ROW_NUMBER — did I pick the one ties actually require?
- [ ] Is my window frame explicit (ROWS BETWEEN) where the default RANGE behavior could break ties?
- [ ] If using LAST_VALUE, did I widen the frame to UNBOUNDED FOLLOWING?
- [ ] Am I averaging a rate directly instead of aggregating numerator/denominator first?
- [ ] For a funnel question — can steps actually be skipped? If so, is my funnel order-aware?
- [ ] For sampling/A-B assignment — am I sampling on a stable hashed key, not RANDOM() per row?
