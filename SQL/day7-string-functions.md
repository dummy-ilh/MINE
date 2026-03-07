# Day 7 — String Functions & REGEX
**FAANG SQL 30-Day Prep**

---

## Concepts Covered

1. Basic String Functions
2. SUBSTRING, LEFT, RIGHT
3. LOCATE & INSTR
4. REPLACE & CONCAT
5. LIKE vs REGEXP
6. COALESCE + String Cleaning
7. String Aggregation
8. FAANG String Patterns

---

## 1. Basic String Functions

```sql
SELECT
  UPPER('hello')       AS upper_case,   -- 'HELLO'
  LOWER('HELLO')       AS lower_case,   -- 'hello'
  LENGTH('hello')      AS length,       -- 5
  CHAR_LENGTH('hello') AS char_length,  -- 5 (use for unicode)
  TRIM('  hello  ')    AS trimmed,      -- 'hello'
  LTRIM('  hello  ')   AS left_trim,    -- 'hello  '
  RTRIM('  hello  ')   AS right_trim,   -- '  hello'
  REVERSE('hello')     AS reversed,     -- 'olleh'
  REPEAT('ha', 3)      AS repeated;     -- 'hahaha'

-- Clean messy user input
SELECT
  LOWER(TRIM(email))   AS clean_email,
  UPPER(TRIM(country)) AS clean_country
FROM users;
```

---

## 2. SUBSTRING, LEFT, RIGHT

```sql
SELECT
  SUBSTRING('hello world', 1, 5)  AS sub1,  -- 'hello' (start, length)
  SUBSTRING('hello world', 7)     AS sub2,  -- 'world' (from pos 7 to end)
  LEFT('hello world', 5)          AS left5, -- 'hello'
  RIGHT('hello world', 5)         AS right5;-- 'world'

-- Extract year from a string date column (when stored as VARCHAR)
SELECT LEFT(order_date_str, 4) AS year_str FROM orders;  -- '2026'

-- Extract domain from email
SELECT
  email,
  SUBSTRING(email, LOCATE('@', email) + 1) AS domain
FROM users;
-- 'john@gmail.com' → 'gmail.com'
```

---

## 3. LOCATE & INSTR

```sql
SELECT
  LOCATE('world', 'hello world'),   -- 7 (position of substring)
  LOCATE('@', 'john@gmail.com'),    -- 5
  INSTR('hello world', 'world');    -- 7 (same as LOCATE, different arg order)

sql-- Find users with gmail
SELECT email FROM users
WHERE LOCATE('@gmail.com', email) > 0;

-- Or simpler with LIKE
SELECT email FROM users WHERE email LIKE '%@gmail.com';
```

---

## 4. REPLACE & CONCAT

```sql
SELECT
  REPLACE('hello world', 'world', 'SQL'), -- 'hello SQL'
  REPLACE(phone, '-', ''),               -- remove dashes
  REPLACE(phone, ' ', '');               -- remove spaces

-- CONCAT
SELECT
  CONCAT(first_name, ' ', last_name) AS full_name,
  CONCAT('user_', user_id)           AS user_label
FROM users;

-- CONCAT_WS — skips NULLs safely
SELECT CONCAT_WS(', ', city, state, country) AS full_address
FROM users;
-- NULL state → 'New York, USA' not 'New York, , USA'
```

> ⚠️ `CONCAT` with any NULL returns NULL. Use `CONCAT_WS` when NULLs are possible.

---

## 5. LIKE vs REGEXP

```sql
-- LIKE — simple pattern matching
-- % = any number of chars, _ = exactly one char
SELECT * FROM users WHERE name LIKE 'A%';       -- starts with A
SELECT * FROM users WHERE name LIKE '%son';     -- ends with son
SELECT * FROM users WHERE name LIKE '_ohn';     -- John, Rohn etc.
SELECT * FROM users WHERE name LIKE '%an%';     -- contains 'an'
SELECT * FROM users WHERE name LIKE 'J_h_';    -- exactly 4 chars, J_h_
```

```sql
-- REGEXP — full regex (MySQL)
SELECT * FROM users WHERE email REGEXP '^[a-z]+@gmail\\.com$';
SELECT * FROM users WHERE phone REGEXP '^[0-9]{10}$'; -- exactly 10 digits
SELECT * FROM users WHERE name  REGEXP '^[A-Z][a-z]+$'; -- Proper case
```

**REGEXP Cheatsheet:**
```
^     = start of string
$     = end of string
.     = any single character
*     = 0 or more of previous
+     = 1 or more of previous
[0-9] = any digit
[a-z] = any lowercase letter
{n}   = exactly n times
{n,m} = between n and m times
\\    = escape special character
```

---

## 6. String Cleaning (DS Pattern)

```sql
-- Standardize messy country values
SELECT user_id,
  CASE
    WHEN LOWER(TRIM(country)) IN ('india', 'in', 'ind') THEN 'India'
    WHEN LOWER(TRIM(country)) IN ('usa', 'us', 'united states') THEN 'USA'
    WHEN LOWER(TRIM(country)) IN ('uk', 'united kingdom', 'gb') THEN 'UK'
    ELSE TRIM(country)
  END AS clean_country
FROM users;
```

```sql
-- SUBSTRING_INDEX — split on delimiter
-- SUBSTRING_INDEX(str, delimiter, count)
-- count > 0 → from left, count < 0 → from right

-- Parse 'color:red,size:large,brand:nike'
SELECT product_id,
  SUBSTRING_INDEX(SUBSTRING_INDEX(tags, 'color:', -1), ',', 1) AS color,
  SUBSTRING_INDEX(SUBSTRING_INDEX(tags, 'size:',  -1), ',', 1) AS size,
  SUBSTRING_INDEX(SUBSTRING_INDEX(tags, 'brand:', -1), ',', 1) AS brand
FROM products;


-- Parse structured strings (e.g. 'category:electronics|brand:apple')
SELECT
  product_id,
  SUBSTRING_INDEX(
    SUBSTRING_INDEX(tags, 'category:', -1), '|', 1
  ) AS category
FROM products;
-- SUBSTRING_INDEX(str, delimiter, count)
-- count > 0 → left side, count < 0 → right side
```

---

## 7. String Aggregation

```sql
-- GROUP_CONCAT (MySQL)
SELECT department,
  GROUP_CONCAT(name ORDER BY name SEPARATOR ', ') AS employees_list
FROM employees
GROUP BY department;
-- 'Engineering' → 'Alice, Bob, Carol'

-- STRING_AGG (Postgres)
SELECT department,
  STRING_AGG(name, ', ' ORDER BY name) AS employees_list
FROM employees
GROUP BY department;
```

---

## 8. FAANG String Patterns

```sql
-- Validate email format
SELECT email,
  CASE
    WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'
    THEN 'Valid' ELSE 'Invalid'
  END AS email_status
FROM users;
```

```sql
-- Extract UTM params from URL
-- URL: 'https://site.com?utm_source=google&utm_medium=cpc'
SELECT url,
  SUBSTRING_INDEX(SUBSTRING_INDEX(url, 'utm_source=',   -1), '&', 1) AS utm_source,
  SUBSTRING_INDEX(SUBSTRING_INDEX(url, 'utm_medium=',   -1), '&', 1) AS utm_medium,
  SUBSTRING_INDEX(SUBSTRING_INDEX(url, 'utm_campaign=', -1), '&', 1) AS utm_campaign
FROM page_views
WHERE url LIKE '%utm_source=%';
```

```sql
-- Count words in a string
SELECT description,
  LENGTH(description) - LENGTH(REPLACE(description, ' ', '')) + 1 AS word_count
FROM products;
```

```sql
-- Find duplicate emails (case-insensitive)
SELECT LOWER(TRIM(email)) AS clean_email, COUNT(*) AS cnt
FROM users
GROUP BY LOWER(TRIM(email))
HAVING COUNT(*) > 1;
```

```sql
-- Mask PII — show only last 4 digits
SELECT user_id,
  CONCAT('****', RIGHT(phone, 4)) AS masked_phone
FROM users;
```

---

## Practice Questions

### Q1 — Easy ✅
Clean email, masked phone, clean country.

```sql
SELECT
  user_id,
  LOWER(TRIM(email))              AS clean_email,
  CONCAT('****', RIGHT(phone, 4)) AS masked_phone,
  UPPER(TRIM(country))            AS clean_country
FROM users;
```

### Q2 — Medium ✅
Parse tags column, filter by color.

```sql
SELECT product_id, product_name,
  SUBSTRING_INDEX(SUBSTRING_INDEX(tags, 'color:', -1), ',', 1) AS color,
  SUBSTRING_INDEX(SUBSTRING_INDEX(tags, 'size:',  -1), ',', 1) AS size,
  SUBSTRING_INDEX(SUBSTRING_INDEX(tags, 'brand:', -1), ',', 1) AS brand
FROM products
HAVING color IN ('red', 'blue');
```

### Q3 — Hard ✅
UTM source monthly analysis.

```sql
WITH parsed AS (
  SELECT user_id, view_date,
    SUBSTRING_INDEX(SUBSTRING_INDEX(url, 'utm_source=',   -1), '&', 1) AS utm_source,
    SUBSTRING_INDEX(SUBSTRING_INDEX(url, 'utm_medium=',   -1), '&', 1) AS utm_medium,
    SUBSTRING_INDEX(SUBSTRING_INDEX(url, 'utm_campaign=', -1), '&', 1) AS utm_campaign
  FROM page_views
  WHERE YEAR(view_date) = 2025
    AND url LIKE '%utm_source=%'
)
SELECT utm_source,
  DATE_FORMAT(view_date, '%Y-%m') AS month,
  COUNT(*)                        AS total_views,
  COUNT(DISTINCT user_id)         AS unique_users
FROM parsed
GROUP BY utm_source, DATE_FORMAT(view_date, '%Y-%m')
HAVING COUNT(*) > 100
ORDER BY month, total_views DESC;
```

---

## Key Takeaways

- **TRIM + LOWER** first — always clean before comparing strings
- **CONCAT_WS** over CONCAT when NULLs are possible
- **LIKE** for simple patterns, **REGEXP** for complex validation
- **SUBSTRING_INDEX** — best tool for delimiter-based parsing
- **GROUP_CONCAT / STRING_AGG** — aggregate rows into a single string
- **Mask PII** with CONCAT + RIGHT — comes up in data governance questions
- **HAVING after parsing** — parse in SELECT, filter in HAVING when alias needed

---

Summary Cheatsheet
FunctionPurposeUPPER / LOWERChange caseTRIM / LTRIM / RTRIMRemove whitespaceLENGTH / CHAR_LENGTHString lengthSUBSTRING(str, pos, len)Extract substringLEFT / RIGHTExtract from endsLOCATE(sub, str)Find position of substringREPLACE(str, from, to)Replace occurrencesCONCAT / CONCAT_WSJoin strings (WS skips NULLs)LIKESimple pattern matchingREGEXPFull regex matchingGROUP_CONCATAggregate rows into stringSUBSTRING_INDEXSplit on delimiter




