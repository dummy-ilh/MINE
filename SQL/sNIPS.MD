
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

