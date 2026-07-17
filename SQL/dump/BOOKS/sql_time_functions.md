# SQL Time-Based Functions — Cheat Sheet

> Examples assume `now() = 2024-03-15 10:30:45` and a table `orders(order_id, created_at, shipped_at)`

---

## 1. Get current date / time

| Function | Result | Notes |
|----------|--------|-------|
| `NOW()` | `2024-03-15 10:30:45` | Current date + time |
| `CURRENT_TIMESTAMP` | `2024-03-15 10:30:45` | Same as NOW(), ANSI standard |
| `CURDATE()` / `CURRENT_DATE` | `2024-03-15` | Date only |
| `CURTIME()` / `CURRENT_TIME` | `10:30:45` | Time only |
| `SYSDATE()` | `2024-03-15 10:30:45` | Evaluated at execution time (not query start) |
| `UTC_TIMESTAMP()` | `2024-03-15 02:30:45` | UTC equivalent of NOW() |

---

## 2. Extract parts of a date

| Function | Example | Result |
|----------|---------|--------|
| `YEAR(date)` | `YEAR('2024-03-15')` | `2024` |
| `MONTH(date)` | `MONTH('2024-03-15')` | `3` |
| `DAY(date)` | `DAY('2024-03-15')` | `15` |
| `HOUR(datetime)` | `HOUR('2024-03-15 10:30:45')` | `10` |
| `MINUTE(datetime)` | `MINUTE('2024-03-15 10:30:45')` | `30` |
| `SECOND(datetime)` | `SECOND('2024-03-15 10:30:45')` | `45` |
| `DAYOFWEEK(date)` | `DAYOFWEEK('2024-03-15')` | `6` (1=Sun, 7=Sat) |
| `DAYOFYEAR(date)` | `DAYOFYEAR('2024-03-15')` | `75` |
| `WEEK(date)` | `WEEK('2024-03-15')` | `11` |
| `QUARTER(date)` | `QUARTER('2024-03-15')` | `1` |
| `DAYNAME(date)` | `DAYNAME('2024-03-15')` | `Friday` |
| `MONTHNAME(date)` | `MONTHNAME('2024-03-15')` | `March` |

**EXTRACT — ANSI standard, works everywhere:**

| Example | Result |
|---------|--------|
| `EXTRACT(YEAR FROM created_at)` | `2024` |
| `EXTRACT(MONTH FROM created_at)` | `3` |
| `EXTRACT(DOW FROM created_at)` | `5` (PostgreSQL: 0=Sun) |
| `EXTRACT(EPOCH FROM created_at)` | Unix timestamp (PostgreSQL) |
| `EXTRACT(QUARTER FROM created_at)` | `1` |
| `EXTRACT(WEEK FROM created_at)` | `11` |

---

## 3. Format a date

| Function | Example | Result |
|----------|---------|--------|
| `DATE_FORMAT(date, fmt)` *(MySQL)* | `DATE_FORMAT(created_at, '%Y-%m')` | `2024-03` |
| `DATE_FORMAT(date, fmt)` *(MySQL)* | `DATE_FORMAT(created_at, '%d %M %Y')` | `15 March 2024` |
| `DATE_FORMAT(date, fmt)` *(MySQL)* | `DATE_FORMAT(created_at, '%W')` | `Friday` |
| `TO_CHAR(date, fmt)` *(PostgreSQL)* | `TO_CHAR(created_at, 'YYYY-MM')` | `2024-03` |
| `TO_CHAR(date, fmt)` *(PostgreSQL)* | `TO_CHAR(created_at, 'Day DD Mon YYYY')` | `Friday 15 Mar 2024` |
| `FORMAT(date, fmt)` *(SQL Server)* | `FORMAT(created_at, 'yyyy-MM')` | `2024-03` |

**Common format codes:**

| Code | Meaning | MySQL | PostgreSQL |
|------|---------|-------|------------|
| Year (4-digit) | 2024 | `%Y` | `YYYY` |
| Month (2-digit) | 03 | `%m` | `MM` |
| Month name | March | `%M` | `Month` |
| Day (2-digit) | 15 | `%d` | `DD` |
| Day name | Friday | `%W` | `Day` |
| Hour (24h) | 10 | `%H` | `HH24` |
| Minute | 30 | `%i` | `MI` |
| Second | 45 | `%s` | `SS` |

---

## 4. Add / subtract time

| Function | Example | Result |
|----------|---------|--------|
| `DATE_ADD(date, INTERVAL n unit)` | `DATE_ADD('2024-03-15', INTERVAL 7 DAY)` | `2024-03-22` |
| `DATE_ADD(date, INTERVAL n unit)` | `DATE_ADD('2024-03-15', INTERVAL 1 MONTH)` | `2024-04-15` |
| `DATE_ADD(date, INTERVAL n unit)` | `DATE_ADD('2024-03-15', INTERVAL 2 HOUR)` | `2024-03-15 02:00:00` |
| `DATE_SUB(date, INTERVAL n unit)` | `DATE_SUB('2024-03-15', INTERVAL 30 DAY)` | `2024-02-14` |
| `date + INTERVAL n unit` *(shorthand)* | `created_at + INTERVAL 1 DAY` | Next day |
| `date - INTERVAL n unit` *(shorthand)* | `created_at - INTERVAL 3 MONTH` | 3 months ago |
| `DATEADD(unit, n, date)` *(SQL Server)* | `DATEADD(DAY, 7, created_at)` | +7 days |

**Valid INTERVAL units:**

`SECOND` · `MINUTE` · `HOUR` · `DAY` · `WEEK` · `MONTH` · `QUARTER` · `YEAR`

---

## 5. Difference between two dates

| Function | Example | Result |
|----------|---------|--------|
| `DATEDIFF(end, start)` *(MySQL)* | `DATEDIFF('2024-03-15', '2024-01-01')` | `74` (days) |
| `DATEDIFF(unit, start, end)` *(SQL Server)* | `DATEDIFF(DAY, '2024-01-01', '2024-03-15')` | `74` |
| `end - start` *(PostgreSQL)* | `'2024-03-15'::date - '2024-01-01'::date` | `74` (days) |
| `TIMESTAMPDIFF(unit, start, end)` | `TIMESTAMPDIFF(HOUR, shipped_at, NOW())` | Hours since shipped |
| `TIMESTAMPDIFF(unit, start, end)` | `TIMESTAMPDIFF(MONTH, '2023-01-01', '2024-03-15')` | `14` |
| `AGE(end, start)` *(PostgreSQL)* | `AGE('2024-03-15', '2000-06-01')` | `23 years 9 mons 14 days` |

---

## 6. Truncate / round a date

Used to group by day, week, month etc. without exact timestamp equality issues.

| Function | Example | Result |
|----------|---------|--------|
| `DATE(datetime)` | `DATE('2024-03-15 10:30:45')` | `2024-03-15` |
| `DATE_TRUNC(unit, date)` *(PostgreSQL)* | `DATE_TRUNC('month', created_at)` | `2024-03-01 00:00:00` |
| `DATE_TRUNC(unit, date)` *(PostgreSQL)* | `DATE_TRUNC('week', created_at)` | `2024-03-11 00:00:00` |
| `DATE_TRUNC(unit, date)` *(PostgreSQL)* | `DATE_TRUNC('hour', created_at)` | `2024-03-15 10:00:00` |
| `TRUNC(date, fmt)` *(Oracle)* | `TRUNC(created_at, 'MM')` | `2024-03-01` |
| `CONVERT(date, DATE)` *(MySQL)* | `CONVERT(created_at, DATE)` | `2024-03-15` |

**Equivalent of DATE_TRUNC in MySQL** (no native support):
```sql
DATE_FORMAT(created_at, '%Y-%m-01')          -- start of month
DATE_FORMAT(created_at, '%Y-01-01')          -- start of year
DATE_SUB(created_at, INTERVAL WEEKDAY(created_at) DAY)  -- start of week
```

---

## 7. Convert & cast

| Function | Example | Result |
|----------|---------|--------|
| `STR_TO_DATE(str, fmt)` *(MySQL)* | `STR_TO_DATE('15-03-2024', '%d-%m-%Y')` | `2024-03-15` |
| `TO_DATE(str, fmt)` *(PostgreSQL/Oracle)* | `TO_DATE('15-03-2024', 'DD-MM-YYYY')` | `2024-03-15` |
| `CAST(val AS DATE)` | `CAST('2024-03-15' AS DATE)` | `2024-03-15` |
| `CAST(val AS DATETIME)` | `CAST('2024-03-15' AS DATETIME)` | `2024-03-15 00:00:00` |
| `UNIX_TIMESTAMP(date)` *(MySQL)* | `UNIX_TIMESTAMP('2024-03-15')` | `1710460800` |
| `FROM_UNIXTIME(ts)` *(MySQL)* | `FROM_UNIXTIME(1710460800)` | `2024-03-15 00:00:00` |
| `TO_TIMESTAMP(ts)` *(PostgreSQL)* | `TO_TIMESTAMP(1710460800)` | `2024-03-15 00:00:00+00` |

---

## 8. Timezone handling

| Function | Example | Notes |
|----------|---------|-------|
| `CONVERT_TZ(dt, from_tz, to_tz)` *(MySQL)* | `CONVERT_TZ(created_at, '+00:00', '+05:30')` | UTC → IST |
| `AT TIME ZONE` *(PostgreSQL/SQL Server)* | `created_at AT TIME ZONE 'America/New_York'` | Convert to NY time |
| `TIMEZONE('UTC', now())` *(PostgreSQL)* | — | Current time in UTC |
| `NOW() AT TIME ZONE 'UTC'` *(PostgreSQL)* | — | Same as above |

---

## 9. Common real-world patterns

| Task | SQL |
|------|-----|
| Orders placed today | `WHERE DATE(created_at) = CURDATE()` |
| Orders in last 7 days | `WHERE created_at >= NOW() - INTERVAL 7 DAY` |
| Orders in last 30 days | `WHERE created_at >= NOW() - INTERVAL 30 DAY` |
| Orders this month | `WHERE YEAR(created_at) = YEAR(NOW()) AND MONTH(created_at) = MONTH(NOW())` |
| Orders this year | `WHERE YEAR(created_at) = YEAR(NOW())` |
| Group by month | `GROUP BY DATE_FORMAT(created_at, '%Y-%m')` |
| Group by week | `GROUP BY YEARWEEK(created_at)` |
| Group by hour of day | `GROUP BY HOUR(created_at)` |
| Group by day of week | `GROUP BY DAYNAME(created_at)` |
| Days since order placed | `DATEDIFF(NOW(), created_at)` |
| Hours since shipped | `TIMESTAMPDIFF(HOUR, shipped_at, NOW())` |
| Age from birthdate | `TIMESTAMPDIFF(YEAR, dob, CURDATE())` |
| First day of current month | `DATE_FORMAT(NOW(), '%Y-%m-01')` |
| Last day of current month | `LAST_DAY(NOW())` |
| Same day last year | `DATE_SUB(CURDATE(), INTERVAL 1 YEAR)` |
| Weekend orders only | `WHERE DAYOFWEEK(created_at) IN (1, 7)` |
| Business hours orders | `WHERE HOUR(created_at) BETWEEN 9 AND 17` |

---

## 10. Window functions with time

| Task | SQL |
|------|-----|
| Running total by date | `SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` |
| 7-day rolling average | `AVG(amount) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` |
| Previous day's value | `LAG(amount, 1) OVER (ORDER BY order_date)` |
| Day-over-day change | `amount - LAG(amount, 1) OVER (ORDER BY order_date)` |
| Next day's value | `LEAD(amount, 1) OVER (ORDER BY order_date)` |
| Month-over-month % change | `ROUND((amount - LAG(amount,1) OVER (ORDER BY month)) / LAG(amount,1) OVER (ORDER BY month) * 100, 1)` |
| First order date per user | `FIRST_VALUE(order_date) OVER (PARTITION BY user_id ORDER BY order_date)` |
| Rank orders by recency | `ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date DESC)` |
| Days between consecutive orders | `DATEDIFF(order_date, LAG(order_date,1) OVER (PARTITION BY user_id ORDER BY order_date))` |

---

## 11. Dialect quick-reference

| Task | MySQL | PostgreSQL | SQL Server |
|------|-------|------------|------------|
| Current timestamp | `NOW()` | `NOW()` | `GETDATE()` |
| Add 7 days | `DATE_ADD(d, INTERVAL 7 DAY)` | `d + INTERVAL '7 days'` | `DATEADD(DAY,7,d)` |
| Difference in days | `DATEDIFF(end, start)` | `end - start` | `DATEDIFF(DAY,start,end)` |
| Extract year | `YEAR(d)` | `EXTRACT(YEAR FROM d)` | `YEAR(d)` |
| Truncate to month | `DATE_FORMAT(d,'%Y-%m-01')` | `DATE_TRUNC('month',d)` | `DATETRUNC(MONTH,d)` |
| String to date | `STR_TO_DATE(s, fmt)` | `TO_DATE(s, fmt)` | `CONVERT(DATE, s)` |
| Last day of month | `LAST_DAY(d)` | `DATE_TRUNC('month',d) + INTERVAL '1 month - 1 day'` | `EOMONTH(d)` |

---
