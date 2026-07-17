# SQL Joins — Complete Guide

---

## What is a JOIN?

A JOIN combines rows from two tables based on a shared column (usually a foreign key).
The type of JOIN controls what happens when a row in one table has **no match** in the other.

---

## 1. INNER JOIN

**Returns:** Only rows that have a match in **both** tables. Unmatched rows are silently dropped.

**Use when:**
- You only want complete data — no gaps
- You're reporting on things that definitely exist on both sides
- Example: Show all customers who have placed at least one order

**Do NOT use when:**
- You need to see customers who haven't ordered yet (they'll vanish from results)
- You're auditing for missing data

**Real-world examples:**
- Show all employees and their assigned department names
- List all products that have been ordered at least once
- Get all students who have enrolled in a course

```sql
SELECT c.name, o.total
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id;
```

---

## 2. LEFT JOIN (LEFT OUTER JOIN)

**Returns:** All rows from the **left** table + matching rows from the right. If no match, right-side columns are `NULL`.

**Use when:**
- The left table is your main focus and the right table is optional extra info
- You want to include records even if they have no related data
- You want to find rows with NO match (add `WHERE right.id IS NULL`)

**Do NOT use when:**
- You strictly need both sides to have data (use INNER JOIN instead)

**Real-world examples:**
- List all customers, showing their last order (or NULL if they've never ordered)
- Show all employees and their manager's name (new hires may have no manager yet)
- Find all products that have **never** been ordered: `WHERE o.id IS NULL`
- List all blog posts and their comments (posts with no comments still appear)

```sql
-- All customers, with or without orders
SELECT c.name, o.total
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;

-- Find customers with NO orders at all
SELECT c.name
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;
```

---

## 3. RIGHT JOIN (RIGHT OUTER JOIN)

**Returns:** All rows from the **right** table + matching rows from the left. Left-side columns are `NULL` if no match.

**Use when:**
- The right table is your main focus
- You inherited a query where the table order is fixed

**Do NOT use when:**
- You can just swap the table order and use a LEFT JOIN — which is almost always clearer to read

**Real-world examples:**
- List all departments, even those with no employees assigned yet
- Show all job positions, with applicants if any exist

```sql
SELECT c.name, o.total
FROM customers c
RIGHT JOIN orders o ON c.id = o.customer_id;
-- Same result as: orders LEFT JOIN customers
```

> 💡 **Tip:** Most developers avoid RIGHT JOIN entirely — just swap the table order and use LEFT JOIN for consistency.

---

## 4. FULL OUTER JOIN

**Returns:** All rows from **both** tables. Unmatched rows on either side show `NULL` for the other table's columns.

**Use when:**
- No table is more important than the other
- You're reconciling two data sources and need to see what's missing on either side
- You're merging datasets and want to catch gaps

**Do NOT use when:**
- You only need matched rows (use INNER JOIN — FULL OUTER is expensive on large tables)
- Your database is MySQL — it doesn't support FULL OUTER JOIN natively (use UNION of LEFT + RIGHT)

**Real-world examples:**
- Compare this month's sales vs last month's — see products that appeared in one but not the other
- Sync two systems: find records in System A missing from System B and vice versa
- Audit employee records across two HR databases

```sql
SELECT c.name, o.total
FROM customers c
FULL OUTER JOIN orders o ON c.id = o.customer_id;
```

---

## 5. CROSS JOIN

**Returns:** Every possible combination of rows from both tables. If Table A has 4 rows and Table B has 3, you get 4 × 3 = 12 rows.

**Use when:**
- You intentionally need every combination
- Generating test/seed data
- Building combination matrices

**Do NOT use when:**
- You forgot to write an ON clause — an accidental CROSS JOIN on large tables can crash your database
- Tables are large (1000 × 1000 = 1,000,000 rows)

**Real-world examples:**
- Generate all size + color combinations for a clothing product (S/M/L × Red/Blue/Green)
- Create a schedule grid: every employee × every shift slot
- Build a multiplication table or probability matrix

```sql
SELECT s.size, c.color
FROM sizes s
CROSS JOIN colors c;
```

---

## 6. SELF JOIN

**Returns:** Rows from a table joined against itself, using two aliases to tell them apart.

**Use when:**
- Rows in a table have a relationship with other rows in the **same** table
- Working with hierarchical or peer data

**Do NOT use when:**
- The relationship spans two different tables (use a normal JOIN)
- You haven't aliased both sides — it will error

**Real-world examples:**
- Show each employee alongside their manager's name (both in the `employees` table)
- Find pairs of products in the same category
- Detect duplicate rows: join a table to itself on identical columns
- Find all flights departing from and arriving at the same city on the same day

```sql
SELECT e.name AS employee, m.name AS manager
FROM employees e
JOIN employees m ON e.manager_id = m.id;
```

---

## Quick Decision Cheatsheet

| Scenario | Use |
|---|---|
| Only want complete matched data | `INNER JOIN` |
| Left table is main, right is optional | `LEFT JOIN` |
| Find rows with NO match in another table | `LEFT JOIN ... WHERE right.id IS NULL` |
| Right table is main, left is optional | `RIGHT JOIN` (or flip + LEFT JOIN) |
| Need everything from both, gaps included | `FULL OUTER JOIN` |
| Every combination of two tables | `CROSS JOIN` |
| Table relates to itself (hierarchy, peers) | `SELF JOIN` |

---

## Common Mistakes

- **Accidentally getting a CROSS JOIN** — forgetting the `ON` clause in old-style SQL
- **Using INNER JOIN when you meant LEFT JOIN** — silently losing rows with no match
- **FULL OUTER JOIN in MySQL** — not supported; use `LEFT JOIN UNION RIGHT JOIN` instead
- **Forgetting aliases in SELF JOIN** — both sides must have different names

---

## 9. Tricky FAANG JOIN Patterns

---

### Pattern 1: Range / Inequality JOIN

`orders`
| order_id | amount |
|----------|--------|
| 1001 | 45 |
| 1002 | 130 |
| 1003 | 280 |

`discount_tiers`
| tier_name | min_amount | max_amount | discount_pct |
|-----------|------------|------------|--------------|
| Bronze | 0 | 99 | 5% |
| Silver | 100 | 199 | 10% |
| Gold | 200 | 999 | 15% |

**Result:**
| order_id | amount | tier_name | discount_pct |
|----------|--------|-----------|--------------|
| 1001 | 45 | Bronze | 5% |
| 1002 | 130 | Silver | 10% |
| 1003 | 280 | Gold | 15% |

---

### Pattern 2: Time-Window JOIN

`impressions`
| user_id | imp_time |
|---------|----------|
| u1 | 10:00 |
| u2 | 10:05 |

`clicks`
| user_id | click_time |
|---------|------------|
| u1 | 10:42 |
| u1 | 11:30 |
| u2 | 12:00 |

**Result** (within 1 hour of impression):
| user_id | imp_time | click_time |
|---------|----------|------------|
| u1 | 10:00 | 10:42 |

> u1's 11:30 click excluded (>1hr). u2 excluded (no match in window).

---

### Pattern 3: JOIN on Subquery Aggregate

`employees`
| name | department | salary |
|------|------------|--------|
| Alice | Eng | 120k |
| Bob | Eng | 95k |
| Carol | HR | 80k |
| Dave | HR | 72k |

**Subquery intermediate:**
| department | max_sal |
|------------|---------|
| Eng | 120k |
| HR | 80k |

**Result:**
| name | department | salary |
|------|------------|--------|
| Alice | Eng | 120k |
| Carol | HR | 80k |

---

### Pattern 4: CROSS JOIN + Anti-JOIN

`users`
| user_id |
|---------|
| u1 |
| u2 |

`products`
| product_id |
|------------|
| p1 |
| p2 |
| p3 |

`purchases` (already bought)
| user_id | product_id |
|---------|------------|
| u1 | p1 |
| u2 | p2 |

**Result** (2 × 3 = 6 combos minus 2 existing):
| user_id | product_id |
|---------|------------|
| u1 | p2 |
| u1 | p3 |
| u2 | p1 |
| u2 | p3 |

---

### Pattern 5: Chained LEFT JOINs — ON vs WHERE

`employees`
| name | dept_id |
|------|---------|
| Alice | 1 |
| Bob | 2 |
| Carol | NULL |

`departments`
| dept_id | dept_name | location |
|---------|-----------|----------|
| 1 | Engineering | NYC |
| 2 | HR | LA |

**❌ WHERE result** (Carol silently dropped):
| name | dept_name |
|------|-----------|
| Alice | Engineering |

**✅ ON result** (Carol preserved):
| name | dept_name |
|------|-----------|
| Alice | Engineering |
| Bob | NULL |
| Carol | NULL |

---

### Pattern 6: Consecutive Events

`logins` (self-joined)
| user_id | login_date |
|---------|------------|
| u1 | Mar 1 |
| u1 | Mar 2 |
| u1 | Mar 5 |
| u2 | Mar 1 |
| u2 | Mar 3 |

**Result:**
| user_id |
|---------|
| u1 |

> u1 matched Mar 1 → Mar 2. u2 skipped Mar 2, no consecutive pair.

---

### Pattern 7: USING vs ON

`employees`
| name | dept_id |
|------|---------|
| Alice | 1 |
| Bob | 2 |

`departments`
| dept_id | dept_name |
|---------|-----------|
| 1 | Engineering |
| 2 | HR |

**ON result** (dept_id appears twice):
| name | e.dept_id | d.dept_id | dept_name |
|------|-----------|-----------|-----------|
| Alice | 1 | 1 | Engineering |
| Bob | 2 | 2 | HR |

**USING result** (dept_id appears once):
| name | dept_id | dept_name |
|------|---------|-----------|
| Alice | 1 | Engineering |
| Bob | 2 | HR |

---
