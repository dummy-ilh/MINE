Since browsing failed, I’ll generate a **complete table of PostgreSQL built-in functions** with **usage and examples**, based on official documentation and advanced SQL references.

---

# 🐘 **PostgreSQL Built-in Functions – Complete Table**

| Category | Function                 | Description       | Usage Example                      |
| -------- | ------------------------ | ----------------- | ---------------------------------- |
| **Math** | `ABS(x)`                 | Absolute value    | `SELECT ABS(-42); → 42`            |
|          | `CEIL(x)` / `CEILING(x)` | Round up          | `SELECT CEIL(1.4); → 2`            |
|          | `FLOOR(x)`               | Round down        | `SELECT FLOOR(1.9); → 1`           |
|          | `ROUND(x, n)`            | Round to n digits | `SELECT ROUND(42.837, 2); → 42.84` |
|          | `TRUNC(x, n)`            | Truncate decimals | `SELECT TRUNC(42.837, 1); → 42.8`  |
|          | `POWER(x, y)`            | Exponentiation    | `SELECT POWER(2, 3); → 8`          |
|          | `SQRT(x)`                | Square root       | `SELECT SQRT(9); → 3`              |
|          | `EXP(x)`                 | Exponential e^x   | `SELECT EXP(1); → 2.7182...`       |
|          | `LN(x)`                  | Natural log       | `SELECT LN(2.718); → ~1`           |
|          | `LOG(b, x)`              | Log base b        | `SELECT LOG(10, 1000); → 3`        |
|          | `MOD(a, b)`              | Modulo            | `SELECT MOD(10, 3); → 1`           |
|          | `GREATEST(a, b, …)`      | Max of inputs     | `SELECT GREATEST(1,5,3); → 5`      |
|          | `LEAST(a, b, …)`         | Min of inputs     | `SELECT LEAST(1,5,3); → 1`         |

---

| **String**                  | Function           | Description                                                      | Example |
| --------------------------- | ------------------ | ---------------------------------------------------------------- | ------- |
| `LENGTH(s)`                 | String length      | `SELECT LENGTH('hello'); → 5`                                    |         |
| `UPPER(s)`                  | To uppercase       | `SELECT UPPER('abc'); → ABC`                                     |         |
| `LOWER(s)`                  | To lowercase       | `SELECT LOWER('ABC'); → abc`                                     |         |
| `INITCAP(s)`                | Capitalize words   | `SELECT INITCAP('hello world'); → Hello World`                   |         |
| `CONCAT(s1, s2, ...)`       | Join strings       | `SELECT CONCAT('a', 'b'); → ab`                                  |         |
| `SUBSTRING(s FROM m FOR n)` | Extract substring  | `SELECT SUBSTRING('abcdef' FROM 2 FOR 3); → bcd`                 |         |
| `LEFT(s, n)`                | Leftmost chars     | `SELECT LEFT('hello', 2); → he`                                  |         |
| `RIGHT(s, n)`               | Rightmost chars    | `SELECT RIGHT('hello', 2); → lo`                                 |         |
| `TRIM()` / `BTRIM()`        | Trim spaces/chars  | `SELECT TRIM(' abc '); → abc`                                    |         |
| `REPLACE(s, from, to)`      | Replace text       | `SELECT REPLACE('2024', '4', '5'); → 2025`                       |         |
| `REPEAT(s, n)`              | Repeat string      | `SELECT REPEAT('ha', 3); → hahaha`                               |         |
| `POSITION(substr IN str)`   | Substring position | `SELECT POSITION('ll' IN 'hello'); → 3`                          |         |
| `STRPOS(s, substr)`         | Like POSITION      | `SELECT STRPOS('abc', 'b'); → 2`                                 |         |
| `OVERLAY()`                 | Replace substring  | `SELECT OVERLAY('abcdef' PLACING 'XYZ' FROM 3 FOR 2); → abXYZef` |         |

---

| **Date/Time**                 | Function           | Description                                | Example |
| ----------------------------- | ------------------ | ------------------------------------------ | ------- |
| `CURRENT_DATE`                | Today's date       | `SELECT CURRENT_DATE;`                     |         |
| `CURRENT_TIME`                | Time now           | `SELECT CURRENT_TIME;`                     |         |
| `CURRENT_TIMESTAMP` / `NOW()` | Date + time        | `SELECT NOW();`                            |         |
| `AGE(date)`                   | Age diff           | `SELECT AGE('2020-01-01'); → 5 years …`    |         |
| `EXTRACT(field FROM date)`    | Extract component  | `SELECT EXTRACT(YEAR FROM CURRENT_DATE);`  |         |
| `DATE_TRUNC(unit, date)`      | Truncate to unit   | `SELECT DATE_TRUNC('month', NOW());`       |         |
| `JUSTIFY_DAYS()`              | Normalize interval | `SELECT JUSTIFY_DAYS(INTERVAL '30 days');` |         |
| `MAKE_DATE(y, m, d)`          | Construct date     | `SELECT MAKE_DATE(2025, 1, 15);`           |         |

---

| **Aggregate**            | Function               | Description                                 | Example |
| ------------------------ | ---------------------- | ------------------------------------------- | ------- |
| `COUNT(*)`               | Row count              | `SELECT COUNT(*) FROM users;`               |         |
| `SUM(x)`                 | Total                  | `SELECT SUM(salary) FROM emp;`              |         |
| `AVG(x)`                 | Average                | `SELECT AVG(score);`                        |         |
| `MIN(x)` / `MAX(x)`      | Min / Max              | `SELECT MAX(age);`                          |         |
| `STRING_AGG(str, delim)` | Concatenate with delim | `SELECT STRING_AGG(name, ', ') FROM users;` |         |
| `ARRAY_AGG(x)`           | Aggregate into array   | `SELECT ARRAY_AGG(id) FROM table;`          |         |

---

| **Window Functions**             | Description             | Example                                                            |
| -------------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `ROW_NUMBER() OVER(...)`         | Row index per partition | `SELECT ROW_NUMBER() OVER(PARTITION BY dept ORDER BY salary DESC)` |
| `RANK()`                         | Gapped ranking          | `SELECT RANK() OVER(ORDER BY score)`                               |
| `DENSE_RANK()`                   | Continuous ranking      | `SELECT DENSE_RANK() OVER(ORDER BY score)`                         |
| `LAG(expr)` / `LEAD(expr)`       | Prev/Next row           | `LAG(salary) OVER(ORDER BY id)`                                    |
| `FIRST_VALUE()` / `LAST_VALUE()` | First/last row value    | `FIRST_VALUE(salary) OVER(...)`                                    |
| `NTH_VALUE(x, n)`                | nth row value           | `NTH_VALUE(name, 2) OVER(...)`                                     |

---

\| **Type Conversion** | Function | Example |
\|---------------------|----------|
\| `CAST(x AS type)` | `SELECT CAST('123' AS INTEGER);` |
\| `::` syntax | `SELECT '123'::INT;` |
\| `TO_CHAR(value, format)` | `SELECT TO_CHAR(NOW(), 'YYYY-MM-DD');` |
\| `TO_DATE(str, fmt)` | `SELECT TO_DATE('20250808', 'YYYYMMDD');` |

---

\| **Conditional** | Function | Example |
\|----------------|----------|
\| `CASE WHEN ... THEN ... ELSE ... END` | `SELECT CASE WHEN salary > 1000 THEN 'High' ELSE 'Low' END` |
\| `COALESCE(a, b, c)` | First non-null | `SELECT COALESCE(NULL, NULL, 'x'); → x` |
\| `NULLIF(a, b)` | NULL if equal | `SELECT NULLIF(5, 5); → NULL` |

---

\| **JSON** | Function | Example |
\|---------|----------|
\| `->` / `->>` | Access JSON/JSONB | `SELECT data->'name' FROM users;` |
\| `#>` / `#>>` | Nested JSON access | `SELECT data#>'{a,b}' FROM table;` |
\| `jsonb_set()` | Set value | `SELECT jsonb_set(data, '{a,b}', '"new"')` |
\| `jsonb_array_elements()` | Explode array | `SELECT jsonb_array_elements('[1,2,3]'::jsonb);` |
\| `to_json()` / `to_jsonb()` | Convert | `SELECT to_json(ARRAY[1,2,3]);` |

---

\| **Array** | Function | Example |
\|----------|----------|
\| `ARRAY[]` | Create array | `SELECT ARRAY[1,2,3];` |
\| `UNNEST()` | Expand array | `SELECT UNNEST(ARRAY[1,2,3]);` |
\| `array_length(arr, dim)` | Length of dimension | `SELECT array_length(ARRAY[1,2,3], 1);` |
\| `array_append(arr, val)` | Add element | `SELECT array_append(ARRAY[1], 2);` |

---


Let me know how deep you want to go.
