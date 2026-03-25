# SQL String Functions — Cheat Sheet

> Examples assume a table `users(user_id, full_name, email, bio)` with a sample row:
> `full_name = 'Alice Johnson'`, `email = 'alice.johnson@gmail.com'`, `bio = '  Senior Dev  '`

---

## 1. Case conversion

| Function | Example | Result |
|----------|---------|--------|
| `UPPER(str)` | `UPPER('alice johnson')` | `ALICE JOHNSON` |
| `LOWER(str)` | `LOWER('ALICE JOHNSON')` | `alice johnson` |
| `INITCAP(str)` *(PostgreSQL/Oracle)* | `INITCAP('alice johnson')` | `Alice Johnson` |

---

## 2. Length

| Function | Example | Result | Notes |
|----------|---------|--------|-------|
| `LENGTH(str)` | `LENGTH('Alice Johnson')` | `13` | Bytes in MySQL, chars in PostgreSQL |
| `CHAR_LENGTH(str)` | `CHAR_LENGTH('Alice Johnson')` | `13` | Always character count — use this |
| `LEN(str)` *(SQL Server)* | `LEN('Alice Johnson')` | `13` | Excludes trailing spaces |
| `OCTET_LENGTH(str)` | `OCTET_LENGTH('café')` | `5` | Byte count (UTF-8 multibyte aware) |

---

## 3. Trim whitespace

| Function | Example | Result |
|----------|---------|--------|
| `TRIM(str)` | `TRIM('  Senior Dev  ')` | `Senior Dev` |
| `LTRIM(str)` | `LTRIM('  Senior Dev  ')` | `Senior Dev  ` |
| `RTRIM(str)` | `RTRIM('  Senior Dev  ')` | `  Senior Dev` |
| `TRIM(LEADING 'x' FROM str)` | `TRIM(LEADING '0' FROM '0042')` | `42` |
| `TRIM(TRAILING 'x' FROM str)` | `TRIM(TRAILING '!' FROM 'Hello!!!')` | `Hello` |
| `TRIM(BOTH 'x' FROM str)` | `TRIM(BOTH '*' FROM '**hello**')` | `hello` |

---

## 4. Substring — extract part of a string

| Function | Example | Result | Notes |
|----------|---------|--------|-------|
| `SUBSTRING(str, start, len)` | `SUBSTRING('Alice Johnson', 7, 7)` | `Johnson` | 1-indexed |
| `SUBSTR(str, start, len)` | `SUBSTR('Alice Johnson', 1, 5)` | `Alice` | Alias, works everywhere |
| `LEFT(str, n)` | `LEFT('Alice Johnson', 5)` | `Alice` | First n chars |
| `RIGHT(str, n)` | `RIGHT('Alice Johnson', 7)` | `Johnson` | Last n chars |
| `MID(str, start, len)` *(MySQL)* | `MID('Alice Johnson', 7, 7)` | `Johnson` | Same as SUBSTRING |

**Extract domain from email:**
```sql
SELECT SUBSTRING(email, LOCATE('@', email) + 1) AS domain FROM users;
-- alice.johnson@gmail.com → gmail.com
```

**Extract username from email:**
```sql
SELECT LEFT(email, LOCATE('@', email) - 1) AS username FROM users;
-- alice.johnson@gmail.com → alice.johnson
```

---

## 5. Search inside a string

| Function | Example | Result | Notes |
|----------|---------|--------|-------|
| `LOCATE(substr, str)` *(MySQL)* | `LOCATE('@', 'alice@gmail.com')` | `6` | 0 if not found |
| `LOCATE(substr, str, start)` | `LOCATE('l', 'Alice Johnson', 3)` | `4` | Start search at pos 3 |
| `POSITION(substr IN str)` | `POSITION('@' IN 'alice@gmail.com')` | `6` | ANSI standard |
| `INSTR(str, substr)` *(MySQL/Oracle)* | `INSTR('alice@gmail.com', '@')` | `6` | Same as LOCATE |
| `STRPOS(str, substr)` *(PostgreSQL)* | `STRPOS('alice@gmail.com', '@')` | `6` | PostgreSQL version |
| `CHARINDEX(substr, str)` *(SQL Server)* | `CHARINDEX('@', 'alice@gmail.com')` | `6` | SQL Server version |

---

## 6. Replace & remove

| Function | Example | Result |
|----------|---------|--------|
| `REPLACE(str, from, to)` | `REPLACE('alice.johnson', '.', ' ')` | `alice johnson` |
| `REPLACE(str, from, to)` | `REPLACE('(123) 456-7890', '-', '')` | `(123) 456 7890` |
| `REPLACE(str, from, '')` | `REPLACE('  hello  ', ' ', '')` | `hello` |
| `REGEXP_REPLACE(str, pattern, to)` | `REGEXP_REPLACE('ph: 123-456', '[0-9]', '#')` | `ph: ###-###` |
| `TRANSLATE(str, from, to)` *(PostgreSQL/Oracle)* | `TRANSLATE('abc', 'abc', 'ABC')` | `ABC` | Char-by-char swap |

---

## 7. Concatenate

| Function | Example | Result |
|----------|---------|--------|
| `CONCAT(s1, s2, ...)` | `CONCAT('Alice', ' ', 'Johnson')` | `Alice Johnson` |
| `CONCAT_WS(sep, s1, s2, ...)` | `CONCAT_WS(', ', 'Alice', 'Johnson')` | `Alice, Johnson` |
| `s1 \|\| s2` *(PostgreSQL/Oracle)* | `'Alice' \|\| ' ' \|\| 'Johnson'` | `Alice Johnson` |
| `+` *(SQL Server)* | `'Alice' + ' ' + 'Johnson'` | `Alice Johnson` |
| `GROUP_CONCAT(col)` *(MySQL)* | `GROUP_CONCAT(tag ORDER BY tag SEPARATOR ', ')` | `design, dev, sql` |
| `STRING_AGG(col, sep)` *(PostgreSQL/SQL Server)* | `STRING_AGG(tag, ', ')` | `design, dev, sql` |

> `CONCAT_WS` skips NULLs automatically — use it when any field might be NULL.

---

## 8. Pad a string

| Function | Example | Result | Use case |
|----------|---------|--------|----------|
| `LPAD(str, len, pad)` | `LPAD('42', 6, '0')` | `000042` | Zero-pad IDs |
| `RPAD(str, len, pad)` | `RPAD('Alice', 10, '.')` | `Alice.....` | Fixed-width output |
| `LPAD(str, len, ' ')` | `LPAD('99', 5, ' ')` | `   99` | Right-align numbers |

---

## 9. Repeat & reverse

| Function | Example | Result |
|----------|---------|--------|
| `REPEAT(str, n)` | `REPEAT('ab', 3)` | `ababab` |
| `REVERSE(str)` | `REVERSE('hello')` | `olleh` |
| `SPACE(n)` | `SPACE(5)` | `     ` (5 spaces) |

---

## 10. Pattern matching

### LIKE

| Pattern | Meaning | Example | Matches |
|---------|---------|---------|---------|
| `%` | Any sequence of chars | `LIKE 'a%'` | `alice`, `arnold` |
| `_` | Exactly one char | `LIKE 'a_ice'` | `alice` |
| `%@gmail.com` | Ends with | `LIKE '%@gmail.com'` | `alice@gmail.com` |
| `%dev%` | Contains | `LIKE '%dev%'` | `senior dev`, `devops` |

```sql
-- Emails not from gmail
SELECT * FROM users WHERE email NOT LIKE '%@gmail.com';

-- Names starting with A or B
SELECT * FROM users WHERE full_name LIKE 'A%' OR full_name LIKE 'B%';
```

### REGEXP / RLIKE (MySQL) / ~ (PostgreSQL)

| Pattern | Meaning | Example |
|---------|---------|---------|
| `^alice` | Starts with alice | `REGEXP '^alice'` |
| `com$` | Ends with com | `REGEXP 'com$'` |
| `[0-9]+` | One or more digits | `REGEXP '[0-9]+'` |
| `[a-z]{3}` | Exactly 3 lowercase letters | `REGEXP '^[a-z]{3}$'` |
| `(gmail\|yahoo)` | Either gmail or yahoo | `REGEXP 'gmail\|yahoo'` |

```sql
-- MySQL: valid email format
SELECT * FROM users WHERE email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';

-- PostgreSQL: same
SELECT * FROM users WHERE email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
```

---

## 11. Split a string

| Function | Example | Result | Notes |
|----------|---------|--------|-------|
| `SUBSTRING_INDEX(str, delim, n)` *(MySQL)* | `SUBSTRING_INDEX('a,b,c', ',', 2)` | `a,b` | First 2 parts |
| `SUBSTRING_INDEX(str, delim, -1)` *(MySQL)* | `SUBSTRING_INDEX('a,b,c', ',', -1)` | `c` | Last part |
| `SPLIT_PART(str, delim, n)` *(PostgreSQL)* | `SPLIT_PART('a,b,c', ',', 2)` | `b` | 1-indexed |
| `STRING_TO_ARRAY(str, delim)` *(PostgreSQL)* | `STRING_TO_ARRAY('a,b,c', ',')` | `{a,b,c}` | Returns array |

**Extract first and last name:**
```sql
-- MySQL
SELECT
  SUBSTRING_INDEX(full_name, ' ', 1)  AS first_name,
  SUBSTRING_INDEX(full_name, ' ', -1) AS last_name
FROM users;

-- PostgreSQL
SELECT
  SPLIT_PART(full_name, ' ', 1) AS first_name,
  SPLIT_PART(full_name, ' ', 2) AS last_name
FROM users;
```

---

## 12. Compare & soundex

| Function | Example | Result | Notes |
|----------|---------|--------|-------|
| `STRCMP(s1, s2)` *(MySQL)* | `STRCMP('alice', 'alice')` | `0` | 0=equal, -1/1=less/greater |
| `SOUNDEX(str)` | `SOUNDEX('Johnson')` | `J525` | Phonetic code |
| `SOUNDS LIKE` *(MySQL)* | `'Johnson' SOUNDS LIKE 'Jonson'` | `1` (true) | Phonetic match |
| `DIFFERENCE(s1, s2)` *(SQL Server)* | `DIFFERENCE('Johnson', 'Jonson')` | `4` | 0-4, 4=most similar |
| `LEVENSHTEIN(s1, s2)` *(PostgreSQL extension)* | `LEVENSHTEIN('kitten', 'sitting')` | `3` | Edit distance |

---

## 13. Type conversion with strings

| Function | Example | Result |
|----------|---------|--------|
| `CAST(val AS CHAR)` | `CAST(42 AS CHAR)` | `'42'` |
| `CAST(str AS UNSIGNED)` | `CAST('42abc' AS UNSIGNED)` | `42` |
| `CONVERT(val, CHAR)` *(MySQL)* | `CONVERT(3.14, CHAR)` | `'3.14'` |
| `TO_CHAR(val, fmt)` *(PostgreSQL)* | `TO_CHAR(1234567, '999,999')` | `1,234,567` |
| `FORMAT(n, decimals)` *(MySQL)* | `FORMAT(1234567.891, 2)` | `1,234,567.89` |

---

## 14. Common real-world patterns

| Task | SQL |
|------|-----|
| Normalize names to title case | `CONCAT(UPPER(LEFT(full_name,1)), LOWER(SUBSTRING(full_name,2)))` |
| Extract email domain | `SUBSTRING(email, LOCATE('@', email) + 1)` |
| Extract email username | `LEFT(email, LOCATE('@', email) - 1)` |
| Remove all spaces | `REPLACE(full_name, ' ', '')` |
| Clean phone number | `REPLACE(REPLACE(REPLACE(phone,'(',''),')',''),'-','')` |
| Check string contains digit | `full_name REGEXP '[0-9]'` |
| Truncate bio to 100 chars | `IF(CHAR_LENGTH(bio)>100, CONCAT(LEFT(bio,97),'...'), bio)` |
| Pad order ID to 8 digits | `LPAD(order_id, 8, '0')` |
| Split first / last name | `SUBSTRING_INDEX(full_name,' ',1)` / `SUBSTRING_INDEX(full_name,' ',-1)` |
| Concatenate tags per user | `GROUP_CONCAT(tag ORDER BY tag SEPARATOR ', ')` |
| Find rows with empty or null | `WHERE bio IS NULL OR TRIM(bio) = ''` |
| Count occurrences of char | `CHAR_LENGTH(str) - CHAR_LENGTH(REPLACE(str, '@', ''))` |
| Reverse a string | `REVERSE(full_name)` |
| Case-insensitive search | `WHERE LOWER(full_name) = LOWER('alice johnson')` |
| Starts with vowel | `WHERE full_name REGEXP '^[AEIOUaeiou]'` |

---

## 15. Dialect quick-reference

| Task | MySQL | PostgreSQL | SQL Server |
|------|-------|------------|------------|
| Concatenate | `CONCAT(a,b)` | `a \|\| b` | `a + b` |
| String length | `CHAR_LENGTH(s)` | `LENGTH(s)` | `LEN(s)` |
| Find position | `LOCATE(sub, str)` | `STRPOS(str, sub)` | `CHARINDEX(sub, str)` |
| Substring | `SUBSTRING(s, pos, len)` | `SUBSTRING(s, pos, len)` | `SUBSTRING(s, pos, len)` |
| Title case | *(manual)*  | `INITCAP(s)` | *(manual)* |
| Regex match | `s REGEXP 'pat'` | `s ~ 'pat'` | `s LIKE '%pat%'` *(limited)* |
| Split string | `SUBSTRING_INDEX(s, d, n)` | `SPLIT_PART(s, d, n)` | `STRING_SPLIT(s, d)` |
| Aggregate to string | `GROUP_CONCAT(col)` | `STRING_AGG(col, sep)` | `STRING_AGG(col, sep)` |
| Regex replace | `REGEXP_REPLACE(s, p, r)` | `REGEXP_REPLACE(s, p, r)` | *(no native, use CLR)* |

---
