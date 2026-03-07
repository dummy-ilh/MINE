# Day 18 — Data Quality & Validation in SQL
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Why Data Quality Matters
2. NULL Analysis
3. Duplicate Detection
4. Range & Constraint Validation
5. Referential Integrity Checks
6. Data Freshness Checks
7. Distribution & Skew Checks
8. Schema Drift Detection
9. Full DQ Report
10. FAANG DQ Patterns

---

## 1. Why Data Quality Matters

```
Bad data → wrong features → wrong model → wrong decisions → lost revenue

At scale (billions of rows), even 0.1% bad data = millions of corrupted records.
DQ checks run automatically in every production pipeline.
```

**Interview answer to "what's the first thing you do with a new dataset?"**
→ Validate it: NULLs, duplicates, range checks, referential integrity, freshness.

---

## 2. NULL Analysis

```sql
-- Count NULLs and % per column
SELECT
  COUNT(*)                                                           AS total_rows,
  SUM(CASE WHEN user_id     IS NULL THEN 1 ELSE 0 END)              AS null_user_id,
  SUM(CASE WHEN email       IS NULL THEN 1 ELSE 0 END)              AS null_email,
  SUM(CASE WHEN signup_date IS NULL THEN 1 ELSE 0 END)              AS null_signup_date,
  ROUND(SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS null_email_pct
FROM users;

-- Flag columns above 5% NULL threshold
WITH null_counts AS (
  SELECT 'user_id'   AS col, SUM(CASE WHEN user_id   IS NULL THEN 1 ELSE 0 END) AS nulls, COUNT(*) AS total FROM users UNION ALL
  SELECT 'email',           SUM(CASE WHEN email      IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM users UNION ALL
  SELECT 'country',         SUM(CASE WHEN country    IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM users
)
SELECT col, nulls, total,
  ROUND(nulls*100.0/total, 2) AS null_pct,
  CASE WHEN nulls*100.0/total > 5 THEN '⚠️ FAIL' ELSE '✅ PASS' END AS dq_check
FROM null_counts ORDER BY null_pct DESC;
```

---

## 3. Duplicate Detection

```sql
-- Duplicate primary keys
SELECT user_id, COUNT(*) AS cnt FROM users
GROUP BY user_id HAVING COUNT(*) > 1;

-- Duplicate business keys (e.g. email)
SELECT email, COUNT(*) AS accounts FROM users
GROUP BY email HAVING COUNT(*) > 1 ORDER BY accounts DESC;

-- Deduplicate — keep latest record per user
WITH deduped AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY updated_at DESC
    ) AS rn
  FROM users
)
SELECT * FROM deduped WHERE rn = 1;
```

---

## 4. Range & Constraint Validation

```sql
SELECT
  SUM(CASE WHEN amount < 0            THEN 1 ELSE 0 END) AS negative_amounts,
  SUM(CASE WHEN order_date > CURRENT_DATE THEN 1 ELSE 0 END) AS future_dates,
  SUM(CASE WHEN amount = 0            THEN 1 ELSE 0 END) AS zero_amounts,
  SUM(CASE WHEN amount > 100000       THEN 1 ELSE 0 END) AS huge_amounts,
  COUNT(*) AS total_rows
FROM orders;

-- Invalid categoricals
SELECT * FROM orders
WHERE status NOT IN ('pending', 'completed', 'cancelled', 'refunded');

-- Date logic violations
SELECT
  SUM(CASE WHEN signup_date > CURRENT_DATE                    THEN 1 ELSE 0 END) AS future_signups,
  SUM(CASE WHEN birth_date  > signup_date                     THEN 1 ELSE 0 END) AS born_after_signup,
  SUM(CASE WHEN DATEDIFF(CURRENT_DATE, birth_date)/365 < 13   THEN 1 ELSE 0 END) AS underage,
  SUM(CASE WHEN DATEDIFF(CURRENT_DATE, birth_date)/365 > 120  THEN 1 ELSE 0 END) AS impossible_age
FROM users;
```

---

## 5. Referential Integrity Checks

```sql
-- Orphaned orders (no matching user)
SELECT COUNT(*) AS orphaned_orders
FROM orders o
LEFT JOIN users u ON o.user_id = u.user_id
WHERE u.user_id IS NULL;

-- Orders placed before user signed up
SELECT o.order_id, o.user_id, o.order_date, u.signup_date
FROM orders o
JOIN users u ON o.user_id = u.user_id
WHERE o.order_date < u.signup_date;

-- Orphaned events (no matching session)
SELECT COUNT(*) AS orphaned_events
FROM events e
LEFT JOIN sessions s ON e.session_id = s.session_id
WHERE s.session_id IS NULL;
```

---

## 6. Data Freshness Checks

```sql
-- Last update check
SELECT
  MAX(created_at) AS latest_record,
  DATEDIFF(CURRENT_TIMESTAMP, MAX(created_at)) AS hours_stale,
  CASE WHEN DATEDIFF(CURRENT_TIMESTAMP, MAX(created_at)) > 24
       THEN '⚠️ STALE' ELSE '✅ FRESH' END AS status
FROM orders;

-- Daily row count trend — detect pipeline failures
SELECT DATE(created_at) AS load_date, COUNT(*) AS rows_loaded,
  LAG(COUNT(*)) OVER (ORDER BY DATE(created_at)) AS prev_day,
  ROUND((COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY DATE(created_at))) * 100.0 /
    NULLIF(LAG(COUNT(*)) OVER (ORDER BY DATE(created_at)), 0), 2) AS pct_change,
  CASE WHEN ABS((COUNT(*) - LAG(COUNT(*)) OVER (ORDER BY DATE(created_at))) * 100.0 /
    NULLIF(LAG(COUNT(*)) OVER (ORDER BY DATE(created_at)), 0)) > 50
  THEN '⚠️ ANOMALY' ELSE '✅ NORMAL' END AS status
FROM orders
GROUP BY DATE(created_at) ORDER BY load_date;
```

---

## 7. Distribution & Skew Checks

```sql
SELECT
  COUNT(*)                AS total_rows,
  ROUND(MIN(amount), 2)   AS min_val,
  ROUND(MAX(amount), 2)   AS max_val,
  ROUND(AVG(amount), 2)   AS mean_val,
  ROUND(STDDEV(amount), 2) AS std_val,
  ROUND(STDDEV(amount) / NULLIF(AVG(amount), 0), 4) AS coeff_variation,
  MAX(CASE WHEN pct_rank <= 0.25 THEN amount END) AS p25,
  MAX(CASE WHEN pct_rank <= 0.50 THEN amount END) AS p50_median,
  MAX(CASE WHEN pct_rank <= 0.75 THEN amount END) AS p75,
  MAX(CASE WHEN pct_rank <= 0.95 THEN amount END) AS p95,
  MAX(CASE WHEN pct_rank <= 0.99 THEN amount END) AS p99
FROM (
  SELECT amount,
    PERCENT_RANK() OVER (ORDER BY amount) AS pct_rank
  FROM orders
) t;
```

---

## 8. Schema Drift Detection

```sql
-- New/dropped/type-changed columns vs yesterday
WITH today AS (
  SELECT column_name, data_type FROM information_schema.columns
  WHERE table_name = 'orders' AND table_schema = 'production'
),
yesterday AS (
  SELECT column_name, data_type FROM column_snapshots
  WHERE table_name = 'orders'
    AND snapshot_date = CURRENT_DATE - INTERVAL 1 DAY
)
SELECT 'ADDED'        AS change_type, t.column_name, t.data_type
FROM today t LEFT JOIN yesterday y ON t.column_name = y.column_name WHERE y.column_name IS NULL
UNION ALL
SELECT 'DROPPED',      y.column_name, y.data_type
FROM yesterday y LEFT JOIN today t ON y.column_name = t.column_name WHERE t.column_name IS NULL
UNION ALL
SELECT 'TYPE_CHANGED', t.column_name, CONCAT(y.data_type,' → ',t.data_type)
FROM today t JOIN yesterday y ON t.column_name = y.column_name WHERE t.data_type != y.data_type;
```

---

## 9. Full DQ Scorecard

```sql
WITH checks AS (
  SELECT COUNT(*) AS total_rows,
    SUM(CASE WHEN user_id    IS NULL THEN 1 ELSE 0 END) AS null_user_id,
    SUM(CASE WHEN amount     IS NULL THEN 1 ELSE 0 END) AS null_amount,
    SUM(CASE WHEN order_date IS NULL THEN 1 ELSE 0 END) AS null_date,
    SUM(CASE WHEN amount < 0         THEN 1 ELSE 0 END) AS negative_amount,
    SUM(CASE WHEN order_date > CURRENT_DATE THEN 1 ELSE 0 END) AS future_date,
    COUNT(*) - COUNT(DISTINCT order_id)                 AS duplicate_ids,
    DATEDIFF(CURRENT_DATE, MAX(order_date))             AS days_since_latest
  FROM orders
)
SELECT total_rows,
  ROUND((1 - null_user_id  *1.0/total_rows)*100, 2) AS user_id_completeness,
  ROUND((1 - null_amount   *1.0/total_rows)*100, 2) AS amount_completeness,
  CASE WHEN negative_amount = 0 THEN '✅' ELSE CONCAT('⚠️  ', negative_amount) END AS amount_validity,
  CASE WHEN future_date     = 0 THEN '✅' ELSE CONCAT('⚠️  ', future_date)     END AS date_validity,
  CASE WHEN duplicate_ids   = 0 THEN '✅' ELSE CONCAT('⚠️  ', duplicate_ids)   END AS pk_uniqueness,
  CASE WHEN days_since_latest <= 1 THEN '✅ FRESH'
       ELSE CONCAT('⚠️  STALE: ', days_since_latest, 'd') END AS freshness
FROM checks;
```

---

## 10. FAANG DQ Patterns

```sql
-- Google: Event deduplication by idempotency key
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY event_id ORDER BY received_at DESC
    ) AS rn
  FROM raw_events
)
INSERT INTO clean_events SELECT * FROM ranked WHERE rn = 1;

-- Amazon: Inventory reconciliation
SELECT sku_id, system_quantity, physical_quantity,
  system_quantity - physical_quantity AS discrepancy,
  CASE WHEN ABS(system_quantity - physical_quantity) > 10
        OR ABS(system_quantity - physical_quantity)*1.0/NULLIF(system_quantity,0) > 0.05
       THEN '⚠️  INVESTIGATE' ELSE '✅ OK' END AS status
FROM inventory_reconciliation
WHERE discrepancy != 0 ORDER BY ABS(discrepancy) DESC;

-- Meta: Late-arriving data
SELECT DATE(event_time) AS event_date, COUNT(*) AS total_events,
  SUM(CASE WHEN TIMESTAMPDIFF(MINUTE, event_time, received_at) > 60
      THEN 1 ELSE 0 END) AS late_events,
  ROUND(SUM(CASE WHEN TIMESTAMPDIFF(MINUTE, event_time, received_at) > 60
      THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS late_pct
FROM events GROUP BY DATE(event_time) ORDER BY event_date;
```

---

## Practice Questions

### Q1 — Easy ✅
DQ report: NULLs, negative amounts, invalid status, future dates.

```sql
WITH checks AS (
  SELECT 'null_order_id'  AS check_name,
    SUM(CASE WHEN order_id   IS NULL THEN 1 ELSE 0 END) AS failures, COUNT(*) AS total FROM orders UNION ALL
  SELECT 'null_user_id',
    SUM(CASE WHEN user_id    IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM orders UNION ALL
  SELECT 'null_amount',
    SUM(CASE WHEN amount     IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM orders UNION ALL
  SELECT 'null_status',
    SUM(CASE WHEN status     IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM orders UNION ALL
  SELECT 'null_order_date',
    SUM(CASE WHEN order_date IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM orders UNION ALL
  SELECT 'negative_amount',
    SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END),         COUNT(*) FROM orders UNION ALL
  SELECT 'invalid_status',
    SUM(CASE WHEN status NOT IN ('pending','completed','cancelled')
      THEN 1 ELSE 0 END),                                COUNT(*) FROM orders UNION ALL
  SELECT 'future_order_date',
    SUM(CASE WHEN order_date > CURRENT_DATE THEN 1 ELSE 0 END), COUNT(*) FROM orders
)
SELECT check_name, failures, total,
  ROUND(failures*100.0/total, 2) AS failure_pct,
  CASE WHEN failures = 0 THEN '✅ PASS' ELSE '⚠️  FAIL' END AS result
FROM checks ORDER BY failures DESC;
```

### Q2 — Medium ✅
Referential integrity violations summary.

```sql
WITH orphaned_orders AS (
  SELECT 'orphaned_orders' AS violation_type, COUNT(*) AS violation_count
  FROM orders o LEFT JOIN users u ON o.user_id = u.user_id
  WHERE u.user_id IS NULL
),
pre_signup_orders AS (
  SELECT 'order_before_signup', COUNT(*)
  FROM orders o JOIN users u ON o.user_id = u.user_id
  WHERE o.order_date < u.signup_date
),
duplicate_orders AS (
  SELECT 'suspected_duplicate_orders', COUNT(*)
  FROM (
    SELECT user_id, order_date, amount, COUNT(*) AS cnt
    FROM orders GROUP BY user_id, order_date, amount HAVING COUNT(*) > 1
  ) dupes
)
SELECT * FROM orphaned_orders
UNION ALL SELECT * FROM pre_signup_orders
UNION ALL SELECT * FROM duplicate_orders
ORDER BY violation_count DESC;
```

### Q3 — Hard ✅
Daily pipeline health monitor.

```sql
WITH daily_stats AS (
  SELECT event_date,
    COUNT(*)                                                                AS total_events,
    ROUND(SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS null_amount_pct,
    COUNT(*) - COUNT(DISTINCT event_id)                                     AS duplicate_event_ids,
    ROUND(SUM(CASE WHEN event_type NOT IN ('click','view','purchase','signup')
      THEN 1 ELSE 0 END)*100.0/COUNT(*), 2)                                AS invalid_type_pct
  FROM events GROUP BY event_date
),
with_rolling AS (
  SELECT *,
    AVG(total_events) OVER (
      ORDER BY event_date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS rolling_7d_avg
  FROM daily_stats
)
SELECT event_date, total_events,
  ROUND(rolling_7d_avg, 0)                                                  AS rolling_7d_avg,
  ROUND(total_events*100.0/NULLIF(rolling_7d_avg,0)-100, 2)                 AS vs_avg_pct,
  null_amount_pct, duplicate_event_ids, invalid_type_pct,
  CASE
    WHEN null_amount_pct    > 5   THEN '⚠️  HIGH NULL RATE'
    WHEN duplicate_event_ids > 0  THEN '⚠️  DUPLICATES'
    WHEN invalid_type_pct   > 1   THEN '⚠️  INVALID TYPES'
    WHEN ABS(total_events*100.0/NULLIF(rolling_7d_avg,0)-100) > 50
                                  THEN '⚠️  VOLUME ANOMALY'
    ELSE '✅ HEALTHY'
  END AS pipeline_status
FROM with_rolling ORDER BY event_date;
```

---

## Key Takeaways

| DQ Dimension | Technique |
|---|---|
| Completeness | `SUM(CASE WHEN col IS NULL ...)` per column |
| Uniqueness | `GROUP BY pk HAVING COUNT(*) > 1` |
| Validity | Range, enum, logic checks |
| Referential integrity | `LEFT JOIN ... WHERE right.id IS NULL` |
| Freshness | `DATEDIFF(NOW(), MAX(created_at))` |
| Volume trend | `LAG(row_count)` day over day |
| Deduplication | `ROW_NUMBER() PARTITION BY key ORDER BY timestamp DESC` |
| Schema drift | `INFORMATION_SCHEMA.COLUMNS` diff vs snapshot |

- **Interview answer** → first thing with new data: check NULLs, dupes, ranges, RI
- **Orphaned records** → LEFT JOIN + WHERE IS NULL
- **Pipeline anomaly** → row count vs 7d rolling average > 50% change
- **Idempotent deduplication** → ROW_NUMBER on natural key, keep rn = 1
- **Late-arriving data** → TIMESTAMPDIFF(event_time, received_at)

---

*Day 18 complete — 12 days to go 🚀*
