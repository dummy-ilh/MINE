
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


