# SQL Deduplication

## 1. Intuition

Duplicates in a table come in two flavors, and confusing them is the #1 mistake:

- **Exact duplicates**: every column value is identical across rows. `SELECT DISTINCT *` or `GROUP BY` on all columns handles this.
- **Logical duplicates**: rows share a *key* (e.g., `user_id`, `order_id`, `email`) but differ in other columns (e.g., timestamp, status). This is the harder, far more common interview scenario — you need to decide *which* row to keep, not just collapse identical ones.

Almost every real dedup problem is the second kind: "for each `user_id`, keep only the most recent row." That reframes deduplication as a **ranking problem**, which is why `ROW_NUMBER()` is the workhorse tool, not `DISTINCT`.

Mental model: **partition the rows into groups that should collapse to one, rank the rows within each group by your tie-break rule, keep rank 1 (or discard rank > 1).**

---

## 2. The Core Pattern: ROW_NUMBER() + PARTITION BY

```sql
SELECT *
FROM (
    SELECT
        t.*,
        ROW_NUMBER() OVER (
            PARTITION BY user_id          -- the "duplicate key"
            ORDER BY updated_at DESC      -- the tie-break rule
        ) AS rn
    FROM events t
) ranked
WHERE rn = 1;
```

- `PARTITION BY` defines what counts as "the same logical record."
- `ORDER BY` inside the window defines which duplicate wins (most recent, highest priority, lowest ID, etc.).
- Filtering `rn = 1` keeps exactly one row per partition — no ties, ever, unlike `RANK()`.

**Why not `RANK()` or `DENSE_RANK()`?** Both can assign rank 1 to multiple rows when the `ORDER BY` column ties (e.g., two rows with the identical `updated_at`). If you use `RANK() = 1` for dedup and there's a tie, you get *both* rows back — silently re-introducing the duplicate you were trying to remove. `ROW_NUMBER()` always breaks ties by physical row order, guaranteeing exactly one row per partition. This is the single most common interview trap.

---

## 3. Worked Example

Table `events`:

| event_id | user_id | updated_at          | status  |
|----------|---------|---------------------|---------|
| 1        | 101     | 2026-01-01 10:00    | pending |
| 2        | 101     | 2026-01-02 09:00    | shipped |
| 3        | 101     | 2026-01-02 09:00    | shipped |
| 4        | 102     | 2026-01-03 11:00    | pending |

Goal: one row per `user_id`, the most recent `status`.

```sql
SELECT user_id, status, updated_at
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC, event_id DESC) AS rn
    FROM events
) t
WHERE rn = 1;
```

Result:

| user_id | status  | updated_at       |
|---------|---------|------------------|
| 101     | shipped | 2026-01-02 09:00 |
| 102     | pending | 2026-01-03 11:00 |

Note the tie between event_id 2 and 3 (identical `updated_at`). Adding `event_id DESC` as a secondary sort key makes the result **deterministic** — this is deliberate, not incidental. Without it, the engine picks arbitrarily and the query becomes non-reproducible across runs.

---

## 4. Deleting Duplicates (not just selecting)

Selecting deduplicated rows is easy; actually removing duplicates from the table requires a CTE (most engines don't allow window functions directly in `DELETE`):

```sql
WITH ranked AS (
    SELECT
        event_id,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
    FROM events
)
DELETE FROM events
WHERE event_id IN (SELECT event_id FROM ranked WHERE rn > 1);
```

Or, in engines supporting it (Postgres):

```sql
DELETE FROM events e
USING ranked r
WHERE e.event_id = r.event_id AND r.rn > 1;
```

**Always keep a primary key or unique row identifier** (`event_id` here) — without one, you cannot target specific duplicate rows for deletion, only whole groups.

---

## 5. DISTINCT vs GROUP BY vs ROW_NUMBER — when to use which

| Technique | Use when | Limitation |
|---|---|---|
| `SELECT DISTINCT *` | Rows are byte-for-byte identical | Can't pick "best" row when non-key columns differ |
| `GROUP BY key` + `MAX()`/`MIN()` | You only need one derived value per key (e.g., latest timestamp) | Awkward when you need the *entire row* tied to that max value — MAX(updated_at) doesn't guarantee the other columns come from the same row |
| `ROW_NUMBER() OVER (PARTITION BY ...)` | You need the full row associated with the "winning" duplicate | Slightly more verbose; requires subquery/CTE |

Common trap: people write

```sql
SELECT user_id, MAX(updated_at) AS updated_at, status
FROM events
GROUP BY user_id
```

This is **invalid in strict SQL** (and misleading even where permitted, e.g. MySQL's non-strict mode) because `status` isn't functionally dependent on `user_id` in the `GROUP BY` — the engine picks an arbitrary `status` value, not necessarily the one that goes with the max `updated_at`. This is a classic L5 interview trap: candidates reach for `GROUP BY + MAX` when they actually need `ROW_NUMBER`.

---

## 6. Self-Join Alternative

Before window functions were universally supported, dedup was done with a self-join anti-pattern:

```sql
SELECT e1.*
FROM events e1
LEFT JOIN events e2
    ON e1.user_id = e2.user_id
    AND e1.updated_at < e2.updated_at
WHERE e2.user_id IS NULL;
```

This keeps only rows with no "later" row for the same `user_id`. It works, but:
- It's O(n²) in naive execution without a good index on `(user_id, updated_at)`.
- It breaks silently on exact ties (both tied rows survive, since neither is strictly "less than" the other).

Worth knowing for legacy codebases or engines with poor window function support, but **default to `ROW_NUMBER()`** in any modern interview or production code — it's clearer and handles ties deterministically with a secondary sort key.

---

## 7. Finding Duplicates (diagnostic, not removal)

Sometimes you just need to *detect* duplicates, not remove them:

```sql
SELECT user_id, COUNT(*) AS cnt
FROM events
GROUP BY user_id
HAVING COUNT(*) > 1;
```

To see the actual duplicate rows (not just counts):

```sql
SELECT *
FROM events e
WHERE user_id IN (
    SELECT user_id FROM events GROUP BY user_id HAVING COUNT(*) > 1
)
ORDER BY user_id, updated_at;
```

---

## 8. Production Considerations

- **Idempotency keys**: in event-driven systems (Kafka, webhooks, retries), duplicates usually arrive because of at-least-once delivery. The real fix is upstream — an idempotency key (`event_id`, `request_id`) enforced via a unique constraint or `INSERT ... ON CONFLICT DO NOTHING` (Postgres) / `MERGE` (SQL Server, Snowflake) — rather than deduping downstream every time you query.
- **Upserts**: `INSERT ... ON CONFLICT (key) DO UPDATE` is the production pattern for "keep the latest version" — it prevents duplicates from ever landing, instead of cleaning them up after the fact.
- **Streaming/ETL dedup**: in batch pipelines, dedup is usually done once at ingestion (staging → clean table) using the `ROW_NUMBER` pattern, not on every downstream query — repeatedly deduping the same data at query time is wasted compute at scale.
- **Watermarking**: in streaming systems (Flink, Spark Structured Streaming), late-arriving duplicate events are handled with watermarks + dedup state stores keyed by event ID within a time window — conceptually the same partition-and-keep-first logic, just online rather than batch.
- **Index the partition key**: `ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at)` benefits enormously from an index on `(user_id, updated_at)` — the engine can then avoid a full sort per partition.

---

## 9. Interview Traps (quick-fire)

1. **Using `RANK()` instead of `ROW_NUMBER()`** for dedup → ties leak duplicates back through.
2. **No tie-breaker in `ORDER BY`** → non-deterministic "duplicate removal," different rows kept on different runs.
3. **`GROUP BY` with a non-aggregated column that isn't functionally dependent on the group key** → arbitrary/wrong values, sometimes silently allowed by the engine.
4. **Forgetting a stable row identifier** → can't write a targeted `DELETE`, only mass deletes by group.
5. **Deduping downstream repeatedly instead of fixing the root cause** (missing unique constraint upstream) — an L5 answer should flag this as a design smell, not just patch it with SQL every time.

---

## 10. L5-Differentiating Talking Points

- Recognize dedup as a **ranking problem**, and explain *why* `ROW_NUMBER` is correct where `RANK`/`DISTINCT`/naive `GROUP BY` are not — this is the single clearest signal of SQL maturity in an interview.
- Distinguish **query-time dedup** (read path, cheap to iterate, doesn't fix root cause) from **write-time dedup** (unique constraints, upserts, idempotency keys — fixes root cause, more design work).
- Talk about **cost at scale**: dedup via window function requires a sort per partition; for very large tables, pre-aggregating or maintaining a deduped materialized table/incremental pipeline beats re-running `ROW_NUMBER()` over the full history on every query.
- Mention **exactly-once vs at-least-once** semantics when the interviewer frames this in a streaming/event context — this signals you're connecting SQL dedup to the broader data engineering picture, not just memorizing syntax.
- Know when dedup is a **symptom of a missing constraint** (no unique index on the natural key) versus a genuine business rule (e.g., "keep the latest status update," which is inherently a many-to-one relationship even in a well-designed schema).

---

## 11. Comprehension Checks

1. Why does `RANK() = 1` fail to reliably deduplicate when there are ties in the `ORDER BY` column, but `ROW_NUMBER() = 1` does not?
2. Write a query to delete all but the earliest row per `order_id` from a table `orders(order_id, customer_id, created_at)`, keeping the row with the smallest `created_at`, breaking ties by the smallest `order_id`.
3. Why is `SELECT customer_id, MAX(created_at), status FROM orders GROUP BY customer_id` problematic if you want "the status of the most recent order per customer"? What's the correct rewrite?
4. In a streaming pipeline with at-least-once delivery, why is `ON CONFLICT DO NOTHING` on ingest usually a better fix than a nightly dedup job? When would you still want the nightly job as well?
