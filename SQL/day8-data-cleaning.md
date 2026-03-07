# Day 8 — Advanced Filtering, Data Cleaning & Deduplication
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Deduplication — Finding & Removing Duplicates
2. Handling NULLs — Advanced
3. Type Casting
4. CASE WHEN for Data Cleaning
5. Advanced WHERE Filtering
6. Data Quality Checks
7. DISTINCT vs GROUP BY
8. EXCEPT / INTERSECT / UNION

---

## 1. Deduplication

```sql
-- Find duplicate rows
SELECT email, COUNT(*) AS cnt
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- See actual duplicate rows
SELECT * FROM users
WHERE email IN (
  SELECT email FROM users
  GROUP BY email HAVING COUNT(*) > 1
)
ORDER BY email;

-- Keep only LATEST row per email (standard FAANG pattern)
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY email
      ORDER BY created_at DESC
    ) AS rn
  FROM users
)
SELECT * FROM ranked WHERE rn = 1;

-- Delete duplicates keeping latest
DELETE FROM users
WHERE user_id NOT IN (
  SELECT user_id FROM (
    SELECT MAX(user_id) AS user_id
    FROM users
    GROUP BY email
  ) AS keep
);
```

> 💡 ROW_NUMBER + PARTITION BY = the standard FAANG deduplication pattern.

---

## 2. Handling NULLs — Advanced

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

## 3. Type Casting

```sql
-- CAST
SELECT
  CAST('123' AS UNSIGNED)      AS str_to_int,
  CAST(123   AS CHAR)          AS int_to_str,
  CAST('2026-03-07' AS DATE)   AS str_to_date,
  CAST(salary AS DECIMAL(10,2)) AS clean_salary
FROM users;

-- Safe division — avoid integer truncation
SELECT
  total_revenue * 1.0 / total_orders            AS correct,
  CAST(total_revenue AS DECIMAL) / total_orders AS also_correct
FROM summary;

-- String to date when stored incorrectly
SELECT STR_TO_DATE('07/03/2026', '%d/%m/%Y') AS parsed_date;  -- MySQL
SELECT TO_DATE('07/03/2026', 'DD/MM/YYYY')   AS parsed_date;  -- Postgres
```

---

## 4. CASE WHEN for Data Cleaning

```sql
-- Fix inconsistent status values
SELECT order_id,
  CASE
    WHEN LOWER(TRIM(status)) IN ('completed','complete','done') THEN 'Completed'
    WHEN LOWER(TRIM(status)) IN ('cancelled','canceled','cancel') THEN 'Cancelled'
    WHEN LOWER(TRIM(status)) IN ('pending','pend','in progress') THEN 'Pending'
    ELSE 'Unknown'
  END AS clean_status
FROM orders;

-- Flag anomalies
SELECT order_id, amount,
  CASE
    WHEN amount < 0      THEN 'Negative — data error'
    WHEN amount = 0      THEN 'Zero order'
    WHEN amount > 100000 THEN 'Suspiciously large'
    ELSE 'Normal'
  END AS anomaly_flag
FROM orders;

-- Impute missing values (ML feature engineering)
SELECT user_id,
  COALESCE(age, ROUND((SELECT AVG(age) FROM users))) AS imputed_age,
  COALESCE(country, 'Unknown')                       AS imputed_country,
  CASE WHEN salary IS NULL THEN 0 ELSE 1 END         AS has_salary
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

---

## 7. DISTINCT vs GROUP BY

```sql
-- DISTINCT: deduplicates result set
SELECT DISTINCT email FROM users;

-- GROUP BY: use when you need aggregates too
SELECT email, COUNT(*) AS occurrences
FROM users
GROUP BY email;
-- DISTINCT can't do this
```

---

## 8. EXCEPT / INTERSECT / UNION

```sql
-- EXCEPT: rows in A but not B (churned users)
SELECT user_id FROM users_2024
EXCEPT
SELECT user_id FROM users_2025;

-- INTERSECT: rows in both (retained users)
SELECT user_id FROM users_2024
INTERSECT
SELECT user_id FROM users_2025;

-- UNION: combine + deduplicate
SELECT user_id FROM table_a
UNION
SELECT user_id FROM table_b;

-- UNION ALL: combine keeping duplicates (faster)
SELECT user_id FROM table_a
UNION ALL
SELECT user_id FROM table_b;
```

> ⚠️ MySQL doesn't support INTERSECT/EXCEPT — use JOINs instead.

---

## Practice Questions

### Q1 — Easy ✅
Data quality report on orders table.

```sql
SELECT
  COUNT(*)                                         AS total_rows,
  COUNT(*) - COUNT(amount)                         AS null_amount,
  COUNT(*) - COUNT(status)                         AS null_status,
  SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END)      AS negative_amounts,
  SUM(CASE WHEN status NOT IN
    ('completed','pending','cancelled')
    THEN 1 ELSE 0 END)                             AS unknown_status
FROM orders;
```

### Q2 — Medium ✅
Keep earliest account per email.

```sql
WITH ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY email ORDER BY created_at ASC
    ) AS rn
  FROM users
)
SELECT user_id, email, created_at, country
FROM ranked WHERE rn = 1;
```

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

---

## Key Takeaways

- **Dedup pattern** → ROW_NUMBER + PARTITION BY email + WHERE rn = 1
- **NULLIF(col, 0)** → prevents divide by zero — always use in rate calculations
- **CAST as DECIMAL** → prevents integer division errors
- **CASE WHEN for cleaning** → standardize messy categorical values
- **COUNT(*) - COUNT(col)** → count NULLs in any column
- **UNION ALL** → faster than UNION when duplicates are acceptable
- **EXCEPT/INTERSECT** → not supported in MySQL, use LEFT JOIN + IS NULL instead

---

*Day 8 complete — 22 days to go 🚀*
