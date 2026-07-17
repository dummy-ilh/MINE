
## Handling NULLs — Advanced

```sql
-- NULLIF — returns NULL if two values equal (prevents divide by zero)
SELECT revenue / NULLIF(sessions, 0) AS revenue_per_session
FROM metrics;

-- COALESCE chain
SELECT user_id,
  COALESCE(mobile_phone, work_phone, home_phone, 'No phone') AS contact
FROM users;

-- NULL-safe comparison (Postgres)
SELECT * FROM users
WHERE email IS NOT DISTINCT FROM 'test@gmail.com';

-- MySQL equivalent
SELECT * FROM users WHERE email <=> 'test@gmail.com';

-- Count NULLs per column (data quality check)
SELECT
  COUNT(*)                                      AS total_rows,
  COUNT(*) - COUNT(email)                       AS null_email,
  COUNT(*) - COUNT(phone)                       AS null_phone,
  ROUND((COUNT(*) - COUNT(email)) * 100.0
        / COUNT(*), 2)                          AS pct_null_email
FROM users;
```

---

## 5. Advanced WHERE Filtering

```sql
-- Exclude test/bot accounts
SELECT * FROM users
WHERE email NOT LIKE '%@test.com'
  AND email NOT LIKE '%@example.com'
  AND email NOT REGEXP '^(test|bot|spam|admin)@'
  AND user_id > 1000;

-- Filter where ANY of multiple columns match
SELECT * FROM products
WHERE 'electronics' IN (category_1, category_2, category_3);

-- Dynamic last complete month filter
SELECT * FROM orders
WHERE order_date >= DATE_FORMAT(NOW() - INTERVAL 1 MONTH, '%Y-%m-01')
  AND order_date <  DATE_FORMAT(NOW(), '%Y-%m-01');
```

---

## 6. Data Quality Report Pattern

```sql
WITH quality AS (
  SELECT
    COUNT(*)                                AS total_rows,
    COUNT(DISTINCT user_id)                 AS unique_users,
    COUNT(*) - COUNT(DISTINCT user_id)      AS duplicate_users,
    COUNT(*) - COUNT(email)                 AS null_emails,
    COUNT(*) - COUNT(phone)                 AS null_phones,
    SUM(CASE WHEN email NOT LIKE '%@%'
             THEN 1 ELSE 0 END)             AS invalid_emails,
    SUM(CASE WHEN age < 0 OR age > 120
             THEN 1 ELSE 0 END)             AS invalid_ages,
    MIN(created_at)                         AS earliest_signup,
    MAX(created_at)                         AS latest_signup
  FROM users
)
SELECT *,
  ROUND(duplicate_users * 100.0 / total_rows, 2) AS pct_duplicates,
  ROUND(null_emails     * 100.0 / total_rows, 2) AS pct_null_email
FROM quality;
```


🔴 Q3 — Hard

Tables: events(user_id, event_type, event_date), users(user_id, signup_date, plan_type)
Find users who: signed up in 2024, are on the 'premium' plan, triggered a 'purchase' event within 7 days of signup, but have NOT triggered any event in the last 30 days (potentially churned premium users). Return user_id, signup_date, days_since_last_event.

### Q3 — Hard ✅
Churned premium users who purchased within 7 days of signup.

```sql
WITH early_purchase AS (
  SELECT DISTINCT u.user_id
  FROM users u
  JOIN events e ON u.user_id = e.user_id
  WHERE u.plan_type = 'premium'
    AND YEAR(u.signup_date) = 2024
    AND e.event_type = 'purchase'
    AND DATEDIFF(e.event_date, u.signup_date) <= 7
),
last_activity AS (
  SELECT user_id, MAX(event_date) AS last_event_date
  FROM events
  GROUP BY user_id
)
SELECT
  u.user_id, u.signup_date,
  DATEDIFF(CURRENT_DATE, l.last_event_date) AS days_since_last_event
FROM users u
JOIN early_purchase ep ON u.user_id = ep.user_id
JOIN last_activity l   ON u.user_id = l.user_id
WHERE DATEDIFF(CURRENT_DATE, l.last_event_date) > 30
ORDER BY days_since_last_event DESC;
```




🟢 Q1 — Easy

Table: orders(order_id, user_id, product_id, amount, order_date)
Find the top 10 most frequently co-purchased product pairs (bought in the same order). Return product_a, product_b, co_purchase_count, sorted descending.


### Q1 — Easy ✅
Top 10 most co-purchased product pairs.

```sql
SELECT a.product_id AS product_a, b.product_id AS product_b,
  COUNT(DISTINCT a.order_id) AS co_purchase_count
FROM orders a
JOIN orders b ON a.order_id = b.order_id AND a.product_id < b.product_id
GROUP BY a.product_id, b.product_id
ORDER BY co_purchase_count DESC LIMIT 10;
🟡 Q2 — Medium

Same table plus products(product_id, name, category).
For user_id = 200, recommend the top 10 products they haven't bought yet, based on what similar users purchased. Define similarity as number of products in common. Exclude products already purchased by user 200.

```

### Q2 — Medium ✅
Top 10 personalized recommendations for user 200 based on similar users.

```sql
WITH user_products AS (SELECT DISTINCT user_id, product_id FROM orders),
similar_users AS (
  SELECT b.user_id AS similar_user, COUNT(*) AS common_products
  FROM user_products a
  JOIN user_products b ON a.product_id = b.product_id AND b.user_id != 200
  WHERE a.user_id = 200
  GROUP BY b.user_id ORDER BY common_products DESC LIMIT 10
),
already_bought AS (SELECT DISTINCT product_id FROM orders WHERE user_id = 200)
SELECT up.product_id, p.name, p.category,
  SUM(su.common_products)         AS recommendation_score,
  COUNT(DISTINCT su.similar_user) AS supporting_users
FROM similar_users su
JOIN user_products up ON su.similar_user = up.user_id
JOIN products p       ON up.product_id   = p.product_id
WHERE up.product_id NOT IN (SELECT product_id FROM already_bought)
GROUP BY up.product_id, p.name, p.category
ORDER BY recommendation_score DESC LIMIT 10;
```

🔴 Q3 — Hard

Tables: orders(order_id, user_id, product_id, amount, order_date), products(product_id, name, category)
Build a full recommendation pipeline for user 200: (1) item-item recs based on co-purchases (score = sum of co_counts), (2) popularity fallback for categories where item-item score is low, (3) ensure no more than 3 items per category (diversity constraint). Return top 15 recommendations with source (item_item or popularity_fallback).
---
### Q3 — Hard ✅
Full pipeline: item-item → popularity fallback → diversity constraint → top 15.

```sql
WITH already_bought AS (
  SELECT DISTINCT product_id FROM orders WHERE user_id = 200
),
item_item AS (
  SELECT b.product_id,
    COUNT(*) AS item_item_score, 'item_item' AS source
  FROM orders a
  JOIN orders b ON a.order_id = b.order_id AND a.product_id != b.product_id
  WHERE a.product_id IN (SELECT product_id FROM already_bought)
    AND b.product_id NOT IN (SELECT product_id FROM already_bought)
  GROUP BY b.product_id
),
popularity AS (
  SELECT o.product_id,
    COUNT(DISTINCT o.order_id) AS pop_score, 'popularity_fallback' AS source
  FROM orders o
  WHERE o.product_id NOT IN (SELECT product_id FROM already_bought)
  GROUP BY o.product_id
),
combined AS (
  SELECT COALESCE(ii.product_id, pop.product_id) AS product_id,
    COALESCE(ii.item_item_score, pop.pop_score * 0.1) AS final_score,
    COALESCE(ii.source, pop.source) AS source
  FROM item_item ii
  FULL OUTER JOIN popularity pop ON ii.product_id = pop.product_id
),
with_category AS (
  SELECT c.product_id, p.name, p.category, c.final_score, c.source,
    ROW_NUMBER() OVER (
      PARTITION BY p.category ORDER BY c.final_score DESC
    ) AS category_rank
  FROM combined c JOIN products p ON c.product_id = p.product_id
)
SELECT product_id, name, category,
  ROUND(final_score, 4) AS score, source, category_rank
FROM with_category
WHERE category_rank <= 3
ORDER BY final_score DESC LIMIT 15;
```

---

Consecutive 7-day active users (island-gap trick)
WITH daily AS (
  SELECT DISTINCT user_id, DATE(order_date) AS day FROM orders
),
with_grp AS (
  SELECT user_id, day,
    DATE_SUB(day, INTERVAL ROW_NUMBER() OVER (
      PARTITION BY user_id ORDER BY day) DAY) AS grp
  FROM daily
)
SELECT user_id FROM with_grp
GROUP BY user_id, grp
HAVING COUNT(*) >= 7;

-- Session analysis (30-min gap = new session)
WITH gaps AS (
  SELECT user_id, event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_time
  FROM events
),
sessions AS (
  SELECT user_id, event_time,
    SUM(CASE
      WHEN TIMESTAMPDIFF(MINUTE, prev_time, event_time) > 30
        OR prev_time IS NULL THEN 1 ELSE 0
    END) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
  FROM gaps
)
SELECT user_id, session_id, COUNT(*) AS events_in_session
FROM sessions
GROUP BY user_id, session_id;

---
# Day 18 — Data Quality & Validation in SQL
**FAANG SQL 30-Day Prep**

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
🟢 Q1 — Easy

Table: orders(order_id, user_id, amount, status, order_date)
Write a data quality report that checks: NULL counts for all columns, negative amounts, invalid status values (only pending/completed/cancelled are valid), and future order dates. Return one row per check with a pass/fail flag.

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

🟡 Q2 — Medium

Tables: orders(order_id, user_id, amount, order_date), users(user_id, signup_date)
Find all referential integrity violations: orders with no matching user, orders placed before the user's signup date, and users with more than one order on the same date with the same amount (suspected duplicates). Return a summary count per violation type.




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

🔴 Q3 — Hard

Table: events(event_id, user_id, event_type, event_date, amount)
Build a daily pipeline health monitor: for each day show total events, NULL rate for amount, duplicate event_id count, % of invalid event_types (valid = click/view/purchase/signup), row count vs 7-day average, and flag days where any metric is out of threshold.
### Q1 — Easy ✅
DQ report: NULLs, negative amounts, invalid status, future dates.
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


