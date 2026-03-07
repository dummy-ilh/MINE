# Day 6 — Date & Time Functions
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Current Date/Time
2. Extracting Date Parts
3. Date Arithmetic
4. DATE_FORMAT & DATE_TRUNC
5. FAANG Date Patterns (Cohorts, Retention, A/B Testing)

---

## 1. Getting Current Date/Time

```sql
SELECT
  CURRENT_DATE,       -- 2026-03-07 (date only)
  CURRENT_TIMESTAMP,  -- 2026-03-07 14:30:00 (date + time)
  NOW(),              -- same as CURRENT_TIMESTAMP
  CURDATE(),          -- MySQL alias for CURRENT_DATE
  CURTIME();          -- MySQL: time only → 14:30:00
```

---

## 2. Extracting Date Parts

```sql
SELECT
  YEAR(order_date),      -- 2026
  MONTH(order_date),     -- 3
  DAY(order_date),       -- 7
  HOUR(created_at),      -- 14
  MINUTE(created_at),    -- 30
  DAYOFWEEK(order_date), -- 1=Sunday … 7=Saturday
  DAYNAME(order_date),   -- 'Saturday'
  MONTHNAME(order_date), -- 'March'
  QUARTER(order_date),   -- 1 (Jan–Mar)
  WEEK(order_date)       -- 10 (week of year)
FROM orders;

-- EXTRACT — portable across DBs (Postgres, BigQuery)
SELECT
  EXTRACT(YEAR  FROM order_date) AS yr,
  EXTRACT(MONTH FROM order_date) AS mo,
  EXTRACT(DOW   FROM order_date) AS day_of_week
FROM orders;
```

---

## 3. Date Arithmetic

```sql
-- Add/subtract time
SELECT
  order_date + INTERVAL 7 DAY   AS plus_7_days,
  order_date - INTERVAL 1 MONTH AS minus_1_month,
  order_date + INTERVAL 1 YEAR  AS plus_1_year,
  created_at + INTERVAL 30 MINUTE AS plus_30_mins
FROM orders;

-- DATEDIFF — days between two dates
SELECT DATEDIFF('2026-03-07', '2026-01-01');  -- 65

-- Days since signup
SELECT user_id,
  DATEDIFF(CURRENT_DATE, signup_date) AS days_since_signup
FROM users;

-- TIMESTAMPDIFF — any unit
SELECT
  TIMESTAMPDIFF(DAY,    start_date, end_date) AS days,
  TIMESTAMPDIFF(MONTH,  start_date, end_date) AS months,
  TIMESTAMPDIFF(HOUR,   start_ts,   end_ts)   AS hours,
  TIMESTAMPDIFF(MINUTE, start_ts,   end_ts)   AS minutes
FROM events;
```

---

## 4. DATE_FORMAT & DATE_TRUNC

```sql
-- DATE_FORMAT (MySQL)
SELECT
  DATE_FORMAT(order_date, '%Y-%m')    AS year_month,  -- '2026-03'
  DATE_FORMAT(order_date, '%Y-%m-%d') AS formatted,   -- '2026-03-07'
  DATE_FORMAT(order_date, '%W, %M %d') AS readable    -- 'Saturday, March 07'
FROM orders;

-- DATE_TRUNC (Postgres / BigQuery)
SELECT
  DATE_TRUNC('month', order_date) AS month_start,  -- 2026-03-01
  DATE_TRUNC('week',  order_date) AS week_start,   -- 2026-03-02
  DATE_TRUNC('year',  order_date) AS year_start    -- 2026-01-01
FROM orders;
```

```sql
-- Group revenue by month — MySQL
SELECT DATE_FORMAT(order_date, '%Y-%m') AS month,
  SUM(amount) AS revenue
FROM orders
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month;

-- Group revenue by month — Postgres/BigQuery
SELECT DATE_TRUNC('month', order_date) AS month,
  SUM(amount) AS revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
```

---

## 5. FAANG Date Patterns

### Last N Days Filter
```sql
-- Last 30 days
SELECT * FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL 30 DAY;

-- Last 7 days (with timestamp)
SELECT * FROM events
WHERE event_time >= NOW() - INTERVAL 7 DAY;
```

### Age / Tenure Calculation
```sql
SELECT name, hire_date,
  TIMESTAMPDIFF(YEAR,  hire_date, CURRENT_DATE) AS years_at_company,
  TIMESTAMPDIFF(MONTH, hire_date, CURRENT_DATE) AS months_at_company
FROM employees;
```

### A/B Test Window
```sql
SELECT user_id, event_type, event_date
FROM events
WHERE event_date BETWEEN '2026-01-01' AND '2026-02-28'
  AND user_id IN (
    SELECT user_id FROM ab_assignments
    WHERE experiment = 'exp_01'
  );
```

### Cohort Retention Table
```sql
WITH cohorts AS (
  SELECT user_id,
    DATE_TRUNC('month', signup_date) AS cohort_month
  FROM users
),
activity AS (
  SELECT DISTINCT user_id,
    DATE_TRUNC('month', event_date) AS active_month
  FROM events
),
combined AS (
  SELECT
    c.cohort_month,
    TIMESTAMPDIFF(MONTH, c.cohort_month, a.active_month) AS month_number,
    COUNT(DISTINCT a.user_id) AS active_users
  FROM cohorts c
  JOIN activity a ON c.user_id = a.user_id
  WHERE TIMESTAMPDIFF(MONTH, c.cohort_month, a.active_month) BETWEEN 0 AND 3
  GROUP BY c.cohort_month, month_number
)
SELECT
  cohort_month,
  MAX(CASE WHEN month_number = 0 THEN active_users END) AS month_0,
  MAX(CASE WHEN month_number = 1 THEN active_users END) AS month_1,
  MAX(CASE WHEN month_number = 2 THEN active_users END) AS month_2,
  MAX(CASE WHEN month_number = 3 THEN active_users END) AS month_3
FROM combined
GROUP BY cohort_month
ORDER BY cohort_month;
-- CASE WHEN + MAX pivot trick → turns rows into columns
```

---

## Practice Questions

### Q1 — Easy ✅
Users signed up in last 90 days with days ago count.

```sql
SELECT
  user_id, name, signup_date,
  DATEDIFF(CURRENT_DATE, signup_date) AS days_ago
FROM users
WHERE signup_date >= CURRENT_DATE - INTERVAL 90 DAY;
```

### Q2 — Medium ✅
Monthly revenue 2025 with MoM % change.

```sql
WITH monthly AS (
  SELECT
    DATE_FORMAT(order_date, '%Y-%m') AS month,
    SUM(amount) AS revenue
  FROM orders
  WHERE YEAR(order_date) = 2025
  GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT
  month, revenue,
  LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
  ROUND(
    (revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0 /
    NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 2
  ) AS mom_pct_change
FROM monthly
ORDER BY month;
```

### Q3 — Hard ✅
Cohort retention table — month 0 through 3.

```sql
WITH cohorts AS (
  SELECT user_id,
    DATE_TRUNC('month', signup_date) AS cohort_month
  FROM users
),
activity AS (
  SELECT DISTINCT user_id,
    DATE_TRUNC('month', event_date) AS active_month
  FROM events
),
combined AS (
  SELECT c.cohort_month,
    TIMESTAMPDIFF(MONTH, c.cohort_month, a.active_month) AS month_number,
    COUNT(DISTINCT a.user_id) AS active_users
  FROM cohorts c
  JOIN activity a ON c.user_id = a.user_id
  WHERE TIMESTAMPDIFF(MONTH, c.cohort_month, a.active_month) BETWEEN 0 AND 3
  GROUP BY c.cohort_month, month_number
)
SELECT cohort_month,
  MAX(CASE WHEN month_number = 0 THEN active_users END) AS month_0,
  MAX(CASE WHEN month_number = 1 THEN active_users END) AS month_1,
  MAX(CASE WHEN month_number = 2 THEN active_users END) AS month_2,
  MAX(CASE WHEN month_number = 3 THEN active_users END) AS month_3
FROM combined
GROUP BY cohort_month
ORDER BY cohort_month;
```

---

## Key Takeaways

- **CURRENT_DATE vs NOW()** — date only vs date+time
- **DATEDIFF(a, b)** — days between; put later date first for positive result
- **TIMESTAMPDIFF(unit, start, end)** — flexible, any time unit
- **DATE_FORMAT** (MySQL) vs **DATE_TRUNC** (Postgres/BQ) — know both
- **DATE_TRUNC** snaps to period start — best for GROUP BY on time periods
- **Cohort pivot** — CASE WHEN + MAX turns row-per-month into columns
- **Last N days** — always `>= CURRENT_DATE - INTERVAL N DAY`, never LIKE on dates

---

*Day 6 complete — 24 days to go 🚀*
